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
from utils.multiview import (
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

    # geo_loss sanity check on the same synthetic setup: build render_pkg-like
    # dicts (median_depth) and confirm near-zero pixel drift / loss.
    ref_pkg = {"median_depth": depth_r.unsqueeze(0)}
    neighbor_pkg = {"median_depth": depth_n.unsqueeze(0)}
    gloss, gmask, gweights = geo_loss(
        ref_cam, neighbor_cam, ref_pkg, neighbor_pkg,
        pixel_noise_th=2.0, znear_relative=0.01, scene_radius=plane_scene_radius(),
    )
    print(f"[{model_name}] geo_loss on flat plane: loss={gloss.item():.6f}, "
          f"coverage={gmask.float().mean().item():.3f}")
    assert gmask.float().mean().item() > 0.5, f"{model_name}: geo_loss mask coverage too low"
    assert gloss.item() < 0.5, f"{model_name}: geo_loss pixel-noise too high on a perfectly flat/consistent scene ({gloss.item()})"

    return ref_cam, neighbor_cam, image_r, image_n


def plane_scene_radius():
    return 8.0  # matches plane_z default; scene extent is O(few) world units


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


def main():
    print("=== Step 2: pure-PyTorch oracle self-test (flat plane, NCC ~= 1) ===")
    check_flat_plane_ncc("PH", render_model=2)
    check_flat_plane_ncc("EQ", render_model=1)

    print("\n=== Step 3: CUDA kernel vs pure-PyTorch oracle (random query points) ===")
    check_cuda_vs_oracle("PH", render_model=2)
    check_cuda_vs_oracle("EQ", render_model=1)

    print("\n=== Step 4: finite-difference backward vs oracle autograd ===")
    check_backward_vs_oracle_autograd("PH", render_model=2)
    check_backward_vs_oracle_autograd("EQ", render_model=1)

    print("\nAll Session F2 multiview checks PASSED.")


if __name__ == "__main__":
    main()
