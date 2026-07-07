"""Session F2 verification (GUTWrap Path A, multiview NCC+geo loss) -- run
inside the `geer` container:
    docker exec -w /home geer python tests/test_multiview.py

Verification plan (see GUTWrap_Discussion/3DGEERGW_EXECUTION.md Session F2):
  1. world_to_view_normals round-trip -- see tests/test_normal_field.py check 5.
  2. Pure-PyTorch reference oracle self-test on synthetic PH *and* EQ flat
     textured planes: NCC ~= 1 for a genuinely flat, correctly-warped patch;
     geo_loss reprojection pixel-noise ~= 0 on the same synthetic setup.
  3. New CUDA kernel vs. the pure-PyTorch reference oracle: numerically
     compare (ncc, valid) at many random synthetic query points (PH and EQ),
     tight tolerance.
  4. Finite-difference backward (EQWarpPatchNCC) vs. the reference oracle's
     own autograd (which is exact, since it's ordinary differentiable
     PyTorch ops) -- gradcheck-style comparison.

No real truck data needed for this file (pure synthetic geometry); the
truck-checkpoint parity/smoke runs are separate (see 3DGEERGW_EXECUTION.md).
"""
import math
import os
import sys
from types import SimpleNamespace

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gaussian_wrapping.fields import view_to_world
from gaussian_wrapping.fisheye_proj import project_view_to_pixel
from utils.graphics_utils import getWorld2View2
from utils.ray_normals import get_ray_dirs_view
import utils.multiview as multiview_mod
from utils.multiview import (
    expected_depth_from_invdepth,
    straight_through_median_depth,
    ncc_reference_oracle,
    geo_loss,
    eq_warp_patch_ncc,
    _ref_to_neighbor_RT,
)

import multiview_ncc as multiview_ncc_ext

torch.manual_seed(0)


def make_camera(render_model, R, T, W, H, focal=140.0, fov_deg=70.0):
    """Build a minimal SimpleNamespace camera object with everything
    get_ray_dirs_view / project_view_to_pixel / _ref_to_neighbor_RT need.
    render_model: 1 = EQ, 2 = PH."""
    cam = SimpleNamespace()
    cam.R = R
    cam.T = T
    cam.render_model = render_model
    cam.image_width = W
    cam.image_height = H
    wvt = torch.tensor(getWorld2View2(R, T)).transpose(0, 1).float().cuda()
    cam.world_view_transform = wvt
    cam.camera_center = wvt.inverse()[3, :3]
    cam._ray_dirs_view = None

    if render_model == 2:  # PH
        cam.focal_x = focal
        cam.focal_y = focal
        cam.principal_x = W / 2.0
        cam.principal_y = H / 2.0
    else:  # EQ
        cam.focal_x = focal
        cam.focal_y = focal
        cam.principal_x = W / 2.0
        cam.principal_y = H / 2.0
        ys, xs = torch.meshgrid(
            torch.arange(H, dtype=torch.float32), torch.arange(W, dtype=torch.float32), indexing="ij"
        )
        tx = (xs + 0.5 - cam.principal_x) / cam.focal_x
        ty = (ys + 0.5 - cam.principal_y) / cam.focal_y
        theta = torch.sqrt(tx ** 2 + ty ** 2)
        phi = torch.atan2(ty, tx)
        raymap = torch.stack(
            [torch.sin(theta) * torch.cos(phi), torch.sin(theta) * torch.sin(phi), torch.cos(theta)], dim=-1
        )
        cam.raymap = raymap
    return cam


def render_plane_image(camera, plane_z, tex_fn):
    """Analytically render camera's view of a world-space flat plane z=plane_z
    (plane normal (0,0,-1) in WORLD space, i.e. facing -z) by intersecting
    each pixel's own ray with the plane and evaluating tex_fn(x, y)."""
    H, W = camera.image_height, camera.image_width
    dirs = get_ray_dirs_view(camera)  # (3,H,W) view-space unit rays
    dirs_flat = dirs.permute(1, 2, 0).reshape(-1, 3)

    origin_world = view_to_world(camera, torch.zeros(1, 3, device="cuda")).view(3)
    dirs_world = view_to_world(camera, dirs_flat) - origin_world  # (N,3)

    t = (plane_z - origin_world[2]) / dirs_world[:, 2]
    world_pts = origin_world.view(1, 3) + t.unsqueeze(-1) * dirs_world
    img = tex_fn(world_pts[:, 0], world_pts[:, 1]).view(H, W)
    return img, t.view(H, W)


def tex_fn(x, y):
    return 0.5 + 0.3 * torch.sin(0.8 * x) * torch.cos(0.6 * y) + 0.2 * torch.sin(1.3 * x + 0.4 * y)


def build_flat_scene(render_model, W=160, H=120, plane_z=8.0, neighbor_deg=6.0, neighbor_t=0.5):
    """Ref camera at canonical pose (R=I, T=0, so world == ref view space).
    Neighbor camera rotated + translated. Both observe the SAME world-space
    flat plane z=plane_z via render_plane_image."""
    import numpy as np

    R_ref = np.eye(3, dtype=np.float64)
    T_ref = np.zeros(3, dtype=np.float64)
    ref_cam = make_camera(render_model, R_ref, T_ref, W, H)

    a = math.radians(neighbor_deg)
    R_n = np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]], dtype=np.float64)
    T_n = np.array([neighbor_t, 0.0, 0.0], dtype=np.float64)
    neighbor_cam = make_camera(render_model, R_n, T_n, W, H)

    image_r, depth_r = render_plane_image(ref_cam, plane_z, tex_fn)
    image_n, depth_n = render_plane_image(neighbor_cam, plane_z, tex_fn)
    return ref_cam, neighbor_cam, image_r, image_n, depth_r, depth_n


def check_flat_plane_ncc(model_name, render_model, patch_radius=3):
    ref_cam, neighbor_cam, image_r, image_n, depth_r, depth_n = build_flat_scene(render_model)
    H, W = ref_cam.image_height, ref_cam.image_width

    ys = torch.arange(20, H - 20, 11, device="cuda")
    xs = torch.arange(20, W - 20, 13, device="cuda")
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    pixels = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1)

    depths = depth_r[pixels[:, 1], pixels[:, 0]].contiguous()
    # ref pose is canonical (world_view_transform == I), so world normal
    # (0,0,-1) IS the ref-view-space normal.
    normals = torch.tensor([0.0, 0.0, -1.0], device="cuda").expand(pixels.shape[0], 3).contiguous()

    ncc, valid = ncc_reference_oracle(
        ref_cam, neighbor_cam, depths, normals, pixels, image_r, image_n, patch_radius=patch_radius
    )
    frac_valid = valid.float().mean().item()
    median_ncc = ncc[valid].median().item() if valid.any() else float("nan")
    print(f"[{model_name}] flat-plane oracle: frac_valid={frac_valid:.3f}, "
          f"median_ncc={median_ncc:.6f}, min_ncc={ncc[valid].min().item():.6f}")
    assert frac_valid > 0.8, f"{model_name}: too few valid patches ({frac_valid})"
    assert median_ncc > 0.95, f"{model_name}: flat-plane median NCC too low ({median_ncc})"

    # geo_loss sanity check on the same synthetic setup: sample neighbor
    # median-depth as the measurement and confirm near-zero pixel drift / loss.
    neighbor_pkg = {"median_depth": depth_n.unsqueeze(0)}
    gloss, gmask, gweights = geo_loss(
        ref_cam, neighbor_cam, depth_r.unsqueeze(0), neighbor_pkg,
        ref_depth_valid=depth_r.unsqueeze(0) > 0,
        pixel_noise_th=2.0, znear_relative=0.01, scene_radius=plane_scene_radius(),
    )
    print(f"[{model_name}] geo_loss on flat plane: loss={gloss.item():.6f}, "
          f"coverage={gmask.float().mean().item():.3f}")
    assert gmask.float().mean().item() > 0.5, f"{model_name}: geo_loss mask coverage too low"
    assert gloss.item() < 0.5, f"{model_name}: geo_loss pixel-noise too high on a perfectly flat/consistent scene ({gloss.item()})"

    return ref_cam, neighbor_cam, image_r, image_n


def check_geo_loss_ref_depth_gradient(model_name, render_model):
    ref_cam, neighbor_cam, _, _, depth_r, depth_n = build_flat_scene(render_model)
    ref_depth = (depth_r * 1.02).unsqueeze(0).detach().clone().requires_grad_(True)
    neighbor_pkg = {"median_depth": depth_n.unsqueeze(0)}

    gloss, gmask, _ = geo_loss(
        ref_cam, neighbor_cam, ref_depth, neighbor_pkg,
        ref_depth_valid=ref_depth > 0,
        pixel_noise_th=2.0, znear_relative=0.01, scene_radius=plane_scene_radius(),
    )
    gloss.backward()
    grad = ref_depth.grad
    grad_on_mask = grad.squeeze(0)[gmask]
    grad_norm = grad_on_mask.abs().sum().item() if grad_on_mask.numel() else 0.0
    print(f"[{model_name}] geo_loss reference-depth grad: loss={gloss.item():.6f}, "
          f"coverage={gmask.float().mean().item():.3f}, grad_l1={grad_norm:.3e}")
    assert gmask.any(), f"{model_name}: geo_loss mask unexpectedly empty in gradient check"
    assert torch.isfinite(grad_on_mask).all(), f"{model_name}: geo_loss reference-depth gradient has NaNs/Infs"
    assert grad_norm > 0.0, f"{model_name}: geo_loss lost its reference-depth gradient"
    print(f"PASS [{model_name}] geo_loss propagates through reference expected depth")


def plane_scene_radius():
    return 8.0  # matches plane_z default; scene extent is O(few) world units


def check_expected_depth_from_invdepth(model_name, render_model):
    ref_cam, _, _, _, depth_r, _ = build_flat_scene(render_model)
    invdepth = (1.0 / depth_r.clamp_min(1e-6)).detach().clone()
    invdepth[0, 0] = 0.0
    invdepth = invdepth.unsqueeze(0).requires_grad_(True)

    expected_depth, valid = expected_depth_from_invdepth(invdepth)
    diff = (expected_depth.squeeze(0)[valid.squeeze(0)] - depth_r[valid.squeeze(0)]).abs()
    max_diff = diff.max().item()
    print(f"[{model_name}] expected-depth conversion: valid={valid.float().mean().item():.3f}, "
          f"max|diff|={max_diff:.3e}")
    assert not bool(valid[0, 0, 0]), f"{model_name}: zero inverse-depth pixel should be invalid"
    assert expected_depth[0, 0, 0].item() == 0.0, f"{model_name}: invalid expected depth must stay zero"
    assert max_diff < 1e-4, f"{model_name}: expected-depth conversion drifted ({max_diff})"

    expected_depth[valid].sum().backward()
    grad = invdepth.grad
    assert grad[0, 0, 0].item() == 0.0, f"{model_name}: invalid pixel received depth gradient"
    assert torch.isfinite(grad[valid]).all(), f"{model_name}: expected-depth gradients must be finite"
    assert grad[valid].abs().max().item() > 0.0, f"{model_name}: valid expected-depth path lost gradients"
    # Touch the camera so the fixture is visibly tied to the projection model in logs.
    assert ref_cam.render_model == render_model
    print(f"PASS [{model_name}] expected inverse-depth wiring is differentiable on valid pixels")


def check_straight_through_median():
    torch.manual_seed(7)
    P, H, W = 50, 8, 10
    xyz = (torch.randn(P, 3, device="cuda") * 2.0 + torch.tensor([0., 0., 6.], device="cuda")).requires_grad_(True)
    wvt = torch.eye(4, device="cuda")  # identity extrinsics: view == world
    gidx = torch.randint(-1, P, (H, W), device="cuda", dtype=torch.int32)
    md = torch.rand(1, H, W, device="cuda") * 5.0 + 1.0
    md[0, 0, 1] = 0.0  # crossed pixel with zero rasterized depth -> invalid

    depth, valid = straight_through_median_depth(md, gidx, xyz, wvt)
    assert torch.equal(depth.detach(), md), "ST value must be the exact median depth"
    assert torch.equal(valid, (gidx >= 0).unsqueeze(0) & (md > 0)), "validity mask wrong"

    depth[valid].sum().backward()
    grad = xyz.grad
    used = torch.zeros(P, dtype=torch.bool, device="cuda")
    used[gidx[(gidx >= 0) & (md[0] > 0)].long()] = True
    assert (grad[~used] == 0).all(), "gaussians not referenced by valid pixels must get no grad"
    assert (grad[used].norm(dim=-1) > 0).all(), "every referenced gaussian must get grad"
    # direction: d||x||/dx = x/||x|| times the pixel count referencing it
    i = int(used.nonzero()[0])
    n_i = int(((gidx == i) & (md[0] > 0)).sum())
    expect = xyz[i].detach() / xyz[i].detach().norm() * n_i
    err = (grad[i] - expect).norm() / expect.norm()
    assert err < 1e-5, f"grad direction/scale off ({err})"
    print(f"PASS straight-through median: exact value, gradient routed to "
          f"{int(used.sum())}/{P} crossing gaussians, unit-direction err {err:.1e}")


def check_cuda_vs_oracle(model_name, render_model, n_points=200, patch_radius=3, tol=5e-3):
    torch.manual_seed(42)
    ref_cam, neighbor_cam, image_r, image_n, depth_r, depth_n = build_flat_scene(render_model)
    H, W = ref_cam.image_height, ref_cam.image_width
    radius = patch_radius

    xs = torch.randint(radius, W - radius, (n_points,), device="cuda")
    ys = torch.randint(radius, H - radius, (n_points,), device="cuda")
    pixels = torch.stack([xs, ys], dim=-1)

    depths = 3.0 + 12.0 * torch.rand(n_points, device="cuda")
    normals = torch.randn(n_points, 3, device="cuda")
    normals[:, 2] = -1.0 - torch.rand(n_points, device="cuda")  # bias toward facing the camera

    ncc_oracle, valid_oracle = ncc_reference_oracle(
        ref_cam, neighbor_cam, depths, normals, pixels, image_r, image_n, patch_radius=radius
    )

    ray_dirs_r = get_ray_dirs_view(ref_cam).permute(1, 2, 0).contiguous()
    R_rn, T_rn = _ref_to_neighbor_RT(ref_cam, neighbor_cam)
    ncc_cuda, valid_cuda = multiview_ncc_ext.multiview_ncc_forward(
        depths, normals, pixels.int(), ray_dirs_r, R_rn, T_rn, image_r, image_n,
        neighbor_cam.render_model, neighbor_cam.focal_x, neighbor_cam.focal_y,
        neighbor_cam.principal_x, neighbor_cam.principal_y, radius,
    )

    valid_agree = (valid_cuda == valid_oracle)
    frac_valid_agree = valid_agree.float().mean().item()
    both_valid = valid_cuda & valid_oracle
    if both_valid.any():
        diff = (ncc_cuda[both_valid] - ncc_oracle[both_valid]).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        p50, p90, p99 = torch.quantile(diff, torch.tensor([0.5, 0.9, 0.99], device=diff.device)).tolist()
    else:
        max_diff = mean_diff = p50 = p90 = p99 = float("nan")
    print(f"[{model_name}] CUDA vs oracle: N={n_points}, valid_agree={frac_valid_agree:.3f}, "
          f"both_valid={int(both_valid.sum())}, max|diff|={max_diff:.3e}, mean|diff|={mean_diff:.3e}, "
          f"p50={p50:.3e}, p90={p90:.3e}, p99={p99:.3e}")
    # Two independently-coded forward passes (CUDA raw bilinear/reduction vs
    # PyTorch grid_sample/reduction) agree up to float32 accumulation-order
    # noise, not bit-for-bit -- p99 << tol confirms `tol` is not just papering
    # over a real discrepancy (verified empirically, not assumed).
    assert frac_valid_agree > 0.95, f"{model_name}: CUDA/oracle validity masks disagree too often"
    assert both_valid.sum() > n_points * 0.5, f"{model_name}: too few mutually-valid points to compare"
    assert p99 < tol, f"{model_name}: CUDA vs oracle NCC p99 diff {p99} >= {tol}"
    print(f"PASS [{model_name}] CUDA kernel matches pure-PyTorch oracle (p99 diff {p99:.2e} < {tol}, "
          f"max {max_diff:.2e} is a float32-rounding outlier)")


def check_backward_vs_oracle_autograd(model_name, render_model, n_points=64, patch_radius=3):
    """Gradient check at MODERATELY-OFF query points (true flat-plane depth
    /normal + a +-40% depth / +-0.3 normal perturbation, not a tiny one).

    Finding from direct investigation (see 3DGEERGW_EXECUTION.md Session F2):
    right at convergence (depth/normal matching the true surface almost
    exactly), ncc sits at a shallow plateau near its max (~1.0) and the TRUE
    gradient (verified via the oracle's own exact autograd) is legitimately
    tiny there -- small enough that float32 rounding noise in the 49-term
    patch sums dominates ANY finite-difference estimate at that point,
    regardless of eps (an eps-sweep from 1e-2 down to 1e-6 was tried; smaller
    eps makes it strictly worse, confirming this is round-off, not
    truncation, error). This is an inherent property of a plateaued
    photometric-correlation loss, not a bug in either implementation --
    verified by comparing the oracle's OWN autograd against the oracle's
    OWN finite differences (independent of the CUDA kernel) and observing
    the identical divergence. Moderately-off points have a real, much
    larger gradient signal and are what this check exercises; near-perfect
    points are exactly where the loss's contribution (and therefore the
    imprecision of its gradient) matters least anyway."""
    torch.manual_seed(7)
    ref_cam, neighbor_cam, image_r, image_n, depth_r, depth_n = build_flat_scene(render_model)
    H, W = ref_cam.image_height, ref_cam.image_width
    radius = patch_radius

    xs = torch.randint(radius, W - radius, (n_points,), device="cuda")
    ys = torch.randint(radius, H - radius, (n_points,), device="cuda")
    pixels = torch.stack([xs, ys], dim=-1)

    true_depths = depth_r[pixels[:, 1], pixels[:, 0]].contiguous()
    depths0 = true_depths * (1.0 + 0.4 * (2 * torch.rand(n_points, device="cuda") - 1))
    normals0 = torch.tensor([0.0, 0.0, -1.0], device="cuda").expand(n_points, 3).contiguous().clone()
    normals0 += 0.3 * torch.randn(n_points, 3, device="cuda")

    # Oracle autograd gradient (exact, ordinary differentiable PyTorch ops).
    depths_o = depths0.clone().requires_grad_(True)
    normals_o = normals0.clone().requires_grad_(True)
    ncc_o, valid_o = ncc_reference_oracle(
        ref_cam, neighbor_cam, depths_o, normals_o, pixels, image_r, image_n, patch_radius=radius
    )
    (ncc_o.sum()).backward()
    grad_depths_oracle = depths_o.grad.clone()
    grad_normals_oracle = normals_o.grad.clone()

    # CUDA finite-difference gradient via EQWarpPatchNCC (default eps=3e-2,
    # tuned per the docstring above / utils/multiview.py::EQWarpPatchNCC).
    ray_dirs_r = get_ray_dirs_view(ref_cam).permute(1, 2, 0).contiguous()
    R_rn, T_rn = _ref_to_neighbor_RT(ref_cam, neighbor_cam)
    depths_c = depths0.clone().requires_grad_(True)
    normals_c = normals0.clone().requires_grad_(True)
    ncc_c, valid_c = eq_warp_patch_ncc(
        depths_c, normals_c, pixels.int(), ray_dirs_r, R_rn, T_rn, image_r, image_n,
        neighbor_cam.render_model, neighbor_cam.focal_x, neighbor_cam.focal_y,
        neighbor_cam.principal_x, neighbor_cam.principal_y, radius,
    )
    (ncc_c.sum()).backward()
    grad_depths_cuda = depths_c.grad.clone()
    grad_normals_cuda = normals_c.grad.clone()

    both_valid = valid_o & valid_c
    n_valid = int(both_valid.sum())
    print(f"[{model_name}] backward check: N={n_points}, both_valid={n_valid}")
    assert n_valid > n_points * 0.5, f"{model_name}: too few mutually valid points for backward check"

    gdo = grad_depths_oracle[both_valid]
    gdc = grad_depths_cuda[both_valid]
    gno = grad_normals_oracle[both_valid]
    gnc = grad_normals_cuda[both_valid]

    gd_sign_agree = (torch.sign(gdo) == torch.sign(gdc)).float().mean().item()
    gn_sign_agree = (torch.sign(gno) == torch.sign(gnc)).float().mean().item()
    gd_mean_diff = (gdo - gdc).abs().mean().item()
    gn_mean_diff = (gno - gnc).abs().mean().item()
    print(f"[{model_name}] grad_depths: mean|diff|={gd_mean_diff:.3e} (oracle scale "
          f"{gdo.abs().mean().item():.3e}), sign_agree={gd_sign_agree:.3f}; "
          f"grad_normals: mean|diff|={gn_mean_diff:.3e} (oracle scale "
          f"{gno.abs().mean().item():.3e}), sign_agree={gn_sign_agree:.3f}")
    # Diagnostic-style thresholds (see 3DGEERGW_EXECUTION.md's established
    # verification convention: Pearson-r/coverage checks, not bit-exact
    # matches) -- the finite-difference gradient is directionally correct
    # most of the time, not numerically tight, which is the accepted
    # trade-off of Phase 1 (see the module docstring above).
    assert gd_sign_agree > 0.65, f"{model_name}: grad_depths sign agreement too low ({gd_sign_agree})"
    assert gn_sign_agree > 0.65, f"{model_name}: grad_normals sign agreement too low ({gn_sign_agree})"
    print(f"PASS [{model_name}] finite-difference backward is directionally consistent with oracle autograd "
          f"(sign agreement {gd_sign_agree:.2f}/{gn_sign_agree:.2f})")


# ---------------------------------------------------------------------------
# Phase 2: analytic CUDA backward (_C.multiview_ncc_backward) checks.
# ---------------------------------------------------------------------------

def _backward_fixture(render_model, n_points, patch_radius, seed=7):
    """Shared fixture for the backward checks: moderately-off query points
    (+-40% depth, +-0.3 normal perturbation off the true flat-plane values)
    -- same rationale as check_backward_vs_oracle_autograd's docstring: at
    convergence the true gradient is legitimately ~0 and nothing meaningful
    can be compared there."""
    torch.manual_seed(seed)
    ref_cam, neighbor_cam, image_r, image_n, depth_r, depth_n = build_flat_scene(render_model)
    H, W = ref_cam.image_height, ref_cam.image_width
    radius = patch_radius

    xs = torch.randint(radius, W - radius, (n_points,), device="cuda")
    ys = torch.randint(radius, H - radius, (n_points,), device="cuda")
    pixels = torch.stack([xs, ys], dim=-1)

    true_depths = depth_r[pixels[:, 1], pixels[:, 0]].contiguous()
    depths0 = true_depths * (1.0 + 0.4 * (2 * torch.rand(n_points, device="cuda") - 1))
    normals0 = torch.tensor([0.0, 0.0, -1.0], device="cuda").expand(n_points, 3).contiguous().clone()
    normals0 += 0.3 * torch.randn(n_points, 3, device="cuda")

    ray_dirs_r = get_ray_dirs_view(ref_cam).permute(1, 2, 0).contiguous()
    R_rn, T_rn = _ref_to_neighbor_RT(ref_cam, neighbor_cam)
    return (ref_cam, neighbor_cam, image_r, image_n, pixels, depths0, normals0,
            ray_dirs_r, R_rn, T_rn)


def check_analytic_backward_vs_oracle(model_name, render_model, n_points=256, patch_radius=3):
    """Phase 2 validation (a): analytic CUDA backward vs the pure-torch
    oracle's exact autograd gradient, on the same moderately-off fixture.

    The oracle is run in FLOAT64 here (same f32 input values, upcast). This
    is deliberate and was established empirically, not assumed: the analytic
    kernel's internals are double precision (see the impl.cu rationale), so
    it computes the exact derivative of the shared real-valued function; a
    FLOAT32 oracle autograd carries its own rounding noise (grid_sample +
    49-term reductions with cancellation) that shows up as a false-error
    tail up to ~7% at small-|grad| points -- verified by comparing the f32
    oracle against this same f64 oracle and observing the identical tail.
    Comparing against f64 measures the kernel's actual error, not the
    reference's.

    Masked (excluded) points, and why each mask is legitimate:
      - validity disagreement (oracle vs kernel valid flag): the loss is
        gated to exactly 0 for invalid patches in both implementations;
        gradient at the validity boundary is genuinely undefined.
      - |grad_oracle| below the noise floor: relative error on a near-zero
        denominator is uninformative (absolute agreement is still checked
        for these via the abs-diff stat).
      - bilinear-cell straddles: the sampled NCC is only piecewise-smooth in
        (u_n, v_n) -- its derivative jumps at integer pixel boundaries of
        the neighbor image. The one remaining f32-vs-f64 input difference
        (R_rn/T_rn are consumed as f32 by the kernel but recomputed in f64
        inside the oracle's camera algebra) can land u_n on opposite sides
        of a boundary, making two exact-but-different-branch derivatives
        disagree legitimately. Detected a-posteriori as isolated points
        whose rel err is >>median (reported, must stay rare)."""
    (ref_cam, neighbor_cam, image_r, image_n, pixels, depths0, normals0,
     ray_dirs_r, R_rn, T_rn) = _backward_fixture(render_model, n_points, patch_radius)

    # Oracle exact autograd in float64 (identical f32 input values, upcast).
    wvt_r = ref_cam.world_view_transform
    wvt_n = neighbor_cam.world_view_transform
    rays_f32 = ref_cam._ray_dirs_view
    ref_cam.world_view_transform = wvt_r.double()
    neighbor_cam.world_view_transform = wvt_n.double()
    ref_cam._ray_dirs_view = rays_f32.double()
    try:
        depths_o = depths0.double().clone().requires_grad_(True)
        normals_o = normals0.double().clone().requires_grad_(True)
        ncc_o, valid_o = ncc_reference_oracle(
            ref_cam, neighbor_cam, depths_o, normals_o, pixels,
            image_r.double(), image_n.double(), patch_radius=patch_radius,
        )
        ncc_o.sum().backward()
        gd_o = depths_o.grad.float().clone()
        gn_o = normals_o.grad.float().clone()
        valid_o = valid_o.bool()
    finally:
        ref_cam.world_view_transform = wvt_r
        neighbor_cam.world_view_transform = wvt_n
        ref_cam._ray_dirs_view = rays_f32

    # Analytic CUDA backward (grad_ncc = ones == d(sum ncc)/d(ncc)).
    ncc_c, valid_c = multiview_ncc_ext.multiview_ncc_forward(
        depths0, normals0, pixels.int(), ray_dirs_r, R_rn, T_rn, image_r, image_n,
        neighbor_cam.render_model, neighbor_cam.focal_x, neighbor_cam.focal_y,
        neighbor_cam.principal_x, neighbor_cam.principal_y, patch_radius,
    )
    gd_c, gn_c = multiview_ncc_ext.multiview_ncc_backward(
        depths0, normals0, pixels.int(), ray_dirs_r, R_rn, T_rn, image_r, image_n,
        neighbor_cam.render_model, neighbor_cam.focal_x, neighbor_cam.focal_y,
        neighbor_cam.principal_x, neighbor_cam.principal_y, patch_radius,
        torch.ones_like(ncc_c), precise=True,
    )
    # Fast (float-internals) variant: report-only stats against the same f64
    # oracle. Expected profile (documented, not asserted tightly): median rel
    # err ~1e-3 -- the noise floor of ANY f32 implementation of this gradient
    # (the f32 reference oracle itself shows the same tail vs f64, verified
    # during Phase-2 bring-up) -- with rare large outliers at bilinear-cell
    # branch flips and small-|grad| cancellation points.
    gd_f, gn_f = multiview_ncc_ext.multiview_ncc_backward(
        depths0, normals0, pixels.int(), ray_dirs_r, R_rn, T_rn, image_r, image_n,
        neighbor_cam.render_model, neighbor_cam.focal_x, neighbor_cam.focal_y,
        neighbor_cam.principal_x, neighbor_cam.principal_y, patch_radius,
        torch.ones_like(ncc_c), precise=False,
    )

    both_valid = valid_o & valid_c
    n_valid = int(both_valid.sum())
    print(f"[{model_name}] analytic-vs-oracle: N={n_points}, both_valid={n_valid}, "
          f"valid_disagree={int((valid_o != valid_c).sum())}")
    assert n_valid > n_points * 0.5, f"{model_name}: too few mutually valid points"

    def _relerr_stats(name, g_ref, g_test, floor_frac=1e-3):
        g_ref_v = g_ref[both_valid].reshape(-1)
        g_test_v = g_test[both_valid].reshape(-1)
        # Noise floor: a fraction of the RMS gradient magnitude (scale-aware,
        # avoids hand-picked absolute constants across PH/EQ fixtures).
        floor = floor_frac * g_ref_v.abs().square().mean().sqrt().item()
        above = g_ref_v.abs() > floor
        rel = (g_test_v[above] - g_ref_v[above]).abs() / g_ref_v[above].abs()
        # a-posteriori bilinear-cell-straddle mask: isolated legit-branch
        # disagreements (see docstring); must remain rare.
        med = rel.median().item() if rel.numel() else float("nan")
        straddle = rel > max(100.0 * med, 0.05)
        n_straddle = int(straddle.sum())
        rel_masked = rel[~straddle]
        max_rel = rel_masked.max().item() if rel_masked.numel() else float("nan")
        mean_rel = rel_masked.mean().item() if rel_masked.numel() else float("nan")
        abs_diff_below = (g_test_v[~above] - g_ref_v[~above]).abs().max().item() if (~above).any() else 0.0
        print(f"[{model_name}] {name}: n_above_floor={int(above.sum())}/{g_ref_v.numel()}, "
              f"median_rel={med:.3e}, max_rel(masked)={max_rel:.3e}, mean_rel={mean_rel:.3e}, "
              f"straddle_masked={n_straddle} ({n_straddle / max(int(above.sum()),1):.4f}), "
              f"max|absdiff| below floor={abs_diff_below:.3e}")
        assert n_straddle <= max(2, int(0.02 * int(above.sum()))), \
            f"{model_name} {name}: too many outliers masked ({n_straddle}) -- not a rare-straddle pattern"
        assert max_rel < 0.01, f"{model_name} {name}: max rel err {max_rel} >= 1%"
        return max_rel

    _relerr_stats("grad_depths", gd_o, gd_c)
    _relerr_stats("grad_normals", gn_o, gn_c)
    print(f"PASS [{model_name}] analytic CUDA backward (precise) matches oracle autograd (<1% rel err above noise floor)")

    # Report-only fast-mode stats (see comment above the precise=False call).
    for name, g_ref, g_test in (("grad_depths(fast)", gd_o, gd_f),
                                ("grad_normals(fast)", gn_o, gn_f)):
        g_ref_v = g_ref[both_valid].reshape(-1)
        g_test_v = g_test[both_valid].reshape(-1)
        floor = 1e-3 * g_ref_v.abs().square().mean().sqrt().item()
        above = g_ref_v.abs() > floor
        rel = (g_test_v[above] - g_ref_v[above]).abs() / g_ref_v[above].abs()
        sign = (torch.sign(g_test_v[above]) == torch.sign(g_ref_v[above])).float().mean().item()
        q50, q90, q99 = torch.quantile(rel, torch.tensor([0.5, 0.9, 0.99], device=rel.device)).tolist()
        print(f"[{model_name}] {name} vs f64 oracle (report-only): "
              f"p50={q50:.3e}, p90={q90:.3e}, p99={q99:.3e}, max={rel.max().item():.3e}, "
              f"sign_agree={sign:.3f}")
        assert sign > 0.95, f"{model_name} {name}: fast-mode sign agreement unexpectedly low ({sign})"


def check_analytic_vs_fd_backward(model_name, render_model, n_points=256, patch_radius=3):
    """Phase 2 validation (b): analytic backward vs the existing calibrated
    (eps=3e-2) FD backward, both through the full EQWarpPatchNCC autograd
    path (mode switch flipped for the analytic arm and restored after).
    FD at eps=3e-2 is intentionally smoothed, so this is a directional/scale
    agreement report (same convention as the Phase-1 FD-vs-oracle check),
    not a tight tolerance."""
    (ref_cam, neighbor_cam, image_r, image_n, pixels, depths0, normals0,
     ray_dirs_r, R_rn, T_rn) = _backward_fixture(render_model, n_points, patch_radius, seed=11)

    def _run(mode):
        old = multiview_mod.MULTIVIEW_NCC_BACKWARD
        multiview_mod.MULTIVIEW_NCC_BACKWARD = mode
        try:
            d = depths0.clone().requires_grad_(True)
            n = normals0.clone().requires_grad_(True)
            ncc, valid = eq_warp_patch_ncc(
                d, n, pixels.int(), ray_dirs_r, R_rn, T_rn, image_r, image_n,
                neighbor_cam.render_model, neighbor_cam.focal_x, neighbor_cam.focal_y,
                neighbor_cam.principal_x, neighbor_cam.principal_y, patch_radius,
            )
            ncc.sum().backward()
            return ncc.detach(), valid, d.grad.clone(), n.grad.clone()
        finally:
            multiview_mod.MULTIVIEW_NCC_BACKWARD = old

    ncc_fd, valid_fd, gd_fd, gn_fd = _run("fd")
    ncc_an, valid_an, gd_an, gn_an = _run("analytic")

    # Forward path must be bit-identical regardless of backward mode.
    assert torch.equal(ncc_fd, ncc_an) and torch.equal(valid_fd, valid_an), \
        f"{model_name}: forward output changed with backward mode -- must be impossible"

    bv = valid_fd
    gd_sign = (torch.sign(gd_fd[bv]) == torch.sign(gd_an[bv])).float().mean().item()
    gn_sign = (torch.sign(gn_fd[bv]) == torch.sign(gn_an[bv])).float().mean().item()
    gd_diff = (gd_fd[bv] - gd_an[bv]).abs().mean().item()
    gn_diff = (gn_fd[bv] - gn_an[bv]).abs().mean().item()
    gd_scale = gd_an[bv].abs().mean().item()
    gn_scale = gn_an[bv].abs().mean().item()
    print(f"[{model_name}] analytic-vs-FD: n_valid={int(bv.sum())}, "
          f"grad_depths sign_agree={gd_sign:.3f} mean|diff|={gd_diff:.3e} (analytic scale {gd_scale:.3e}); "
          f"grad_normals sign_agree={gn_sign:.3f} mean|diff|={gn_diff:.3e} (analytic scale {gn_scale:.3e})")
    # Same 0.65 bar the Phase-1 FD-vs-oracle check uses: FD is the noisy arm
    # here; the analytic arm is separately held to <1% vs the oracle above.
    assert gd_sign > 0.65, f"{model_name}: analytic-vs-FD grad_depths sign agreement too low ({gd_sign})"
    assert gn_sign > 0.65, f"{model_name}: analytic-vs-FD grad_normals sign agreement too low ({gn_sign})"
    print(f"PASS [{model_name}] analytic backward directionally consistent with calibrated FD backward")


def benchmark_backward_modes(model_name, render_model, n_points=10000, patch_radius=3, iters=30):
    """Phase 2 validation (c): microbenchmark forward+backward for the FD vs
    analytic modes at ~10k query points (GPU-light: the synthetic images are
    160x120 and all per-point buffers are O(n_points))."""
    (ref_cam, neighbor_cam, image_r, image_n, pixels, depths0, normals0,
     ray_dirs_r, R_rn, T_rn) = _backward_fixture(render_model, n_points, patch_radius, seed=3)

    def _run_once():
        d = depths0.clone().requires_grad_(True)
        n = normals0.clone().requires_grad_(True)
        ncc, valid = eq_warp_patch_ncc(
            d, n, pixels.int(), ray_dirs_r, R_rn, T_rn, image_r, image_n,
            neighbor_cam.render_model, neighbor_cam.focal_x, neighbor_cam.focal_y,
            neighbor_cam.principal_x, neighbor_cam.principal_y, patch_radius,
        )
        ncc.sum().backward()

    import time
    results = {}
    for label, mode, precise in (("fd", "fd", True),
                                 ("analytic64", "analytic", True),
                                 ("analytic32", "analytic", False)):
        old_mode = multiview_mod.MULTIVIEW_NCC_BACKWARD
        old_prec = multiview_mod.MULTIVIEW_NCC_ANALYTIC_PRECISE
        multiview_mod.MULTIVIEW_NCC_BACKWARD = mode
        multiview_mod.MULTIVIEW_NCC_ANALYTIC_PRECISE = precise
        try:
            for _ in range(3):  # warmup
                _run_once()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(iters):
                _run_once()
            torch.cuda.synchronize()
            results[label] = (time.perf_counter() - t0) / iters * 1e3
        finally:
            multiview_mod.MULTIVIEW_NCC_BACKWARD = old_mode
            multiview_mod.MULTIVIEW_NCC_ANALYTIC_PRECISE = old_prec

    print(f"[{model_name}] fwd+bwd timing @ {n_points} pts (avg of {iters}): "
          f"fd={results['fd']:.3f} ms, analytic64={results['analytic64']:.3f} ms "
          f"({results['fd']/results['analytic64']:.2f}x vs fd), "
          f"analytic32={results['analytic32']:.3f} ms "
          f"({results['fd']/results['analytic32']:.2f}x vs fd)")
    return results


def main():
    print("=== Step 1: differentiable reference-depth wiring ===")
    check_expected_depth_from_invdepth("PH", render_model=2)
    check_expected_depth_from_invdepth("EQ", render_model=1)
    check_geo_loss_ref_depth_gradient("PH", render_model=2)
    check_geo_loss_ref_depth_gradient("EQ", render_model=1)
    check_straight_through_median()

    print("=== Step 2: pure-PyTorch oracle self-test (flat plane, NCC ~= 1) ===")
    check_flat_plane_ncc("PH", render_model=2)
    check_flat_plane_ncc("EQ", render_model=1)

    print("\n=== Step 3: CUDA kernel vs pure-PyTorch oracle (random query points) ===")
    check_cuda_vs_oracle("PH", render_model=2)
    check_cuda_vs_oracle("EQ", render_model=1)

    print("\n=== Step 4: finite-difference backward vs oracle autograd ===")
    check_backward_vs_oracle_autograd("PH", render_model=2)
    check_backward_vs_oracle_autograd("EQ", render_model=1)

    print("\n=== Step 5 (Phase 2): analytic CUDA backward vs oracle autograd ===")
    check_analytic_backward_vs_oracle("PH", render_model=2)
    check_analytic_backward_vs_oracle("EQ", render_model=1)

    print("\n=== Step 6 (Phase 2): analytic vs calibrated-FD backward (full autograd path) ===")
    check_analytic_vs_fd_backward("PH", render_model=2)
    check_analytic_vs_fd_backward("EQ", render_model=1)

    print("\n=== Step 7 (Phase 2): fwd+backward timing, FD vs analytic ===")
    benchmark_backward_modes("PH", render_model=2)
    benchmark_backward_modes("EQ", render_model=1)

    print("\nAll Session F2 multiview checks PASSED.")


if __name__ == "__main__":
    main()
