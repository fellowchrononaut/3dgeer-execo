# ported/adapted from GaussianWrapping gaussian_wrapping/regularization/
# multiview_gggs.py (compute_nearest_cameras: near-verbatim, pure
# camera-pose/extrinsics geometry, no projection-model dependency) and
# gaussian_wrapping/regularization/regularizer/multiview.py (wiring shape).
#
# The photometric (NCC) half is NOT a port of multiview_gggs.py's
# PatchMatch/warp_patch_ncc -- that machinery is pinhole-only (a
# plane-induced homography). This module instead wraps the NEW general
# ray-based EQ/PH CUDA kernel in submodules/multiview_ncc/, plus a
# pure-PyTorch reference oracle (test-only) implementing the exact same math
# for numerical verification. See GUTWrap_Discussion/3DGEERGW_EXECUTION.md,
# Session F2, for the full design rationale.
"""Multiview photometric (NCC) + geometric consistency loss, generalized to
3DGEER's PH and EQ (fisheye) camera models via utils/ray_normals.py::
get_ray_dirs_view and gaussian_wrapping/fisheye_proj.py::project_view_to_pixel.
"""
import math
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from gaussian_wrapping.fisheye_proj import project_view_to_pixel
from utils.ray_normals import get_ray_dirs_view

try:
    import multiview_ncc as _multiview_ncc_ext
    _HAVE_MULTIVIEW_NCC_EXT = True
except ImportError:
    _HAVE_MULTIVIEW_NCC_EXT = False

# Backward-mode switch for EQWarpPatchNCC (Phase 1 vs Phase 2, see
# 3DGEERGW_EXECUTION.md Session F2 "backward-gradient strategy"):
#   "fd"       -- Phase 1 finite-difference backward (9 forward kernel calls,
#                 calibrated eps=3e-2; the validated production default).
#   "analytic" -- Phase 2 hand-derived analytic backward (single CUDA
#                 backward-kernel call, submodules/multiview_ncc
#                 _C.multiview_ncc_backward; validated against the
#                 pure-torch oracle autograd + the FD backward in
#                 tests/test_multiview.py, but NOT the default pending the
#                 A/B-run evidence + user decision).
# Override without code changes via the environment variable
# MULTIVIEW_NCC_BACKWARD=analytic|fd.
#
# MULTIVIEW_NCC_ANALYTIC_PRECISE (default "1") selects the analytic kernel's
# internal precision: "1" = double (validated machine-exact vs a float64
# oracle; roughly FD speed on consumer fp64-throttled GPUs), "0" = float
# (several times faster; f32 noise floor, still far tighter than FD's
# eps=3e-2 sign-agreement-grade gradients). Only consulted when the
# backward mode is "analytic".
import os as _os
MULTIVIEW_NCC_BACKWARD = _os.environ.get("MULTIVIEW_NCC_BACKWARD", "fd")
assert MULTIVIEW_NCC_BACKWARD in ("fd", "analytic"), \
    f"MULTIVIEW_NCC_BACKWARD must be 'fd' or 'analytic', got {MULTIVIEW_NCC_BACKWARD!r}"
MULTIVIEW_NCC_ANALYTIC_PRECISE = _os.environ.get("MULTIVIEW_NCC_ANALYTIC_PRECISE", "1") != "0"


# ---------------------------------------------------------------------------
# compute_nearest_cameras -- pure extrinsics geometry, ported near-verbatim
# from multiview_gggs.py:19-83. Deviation from the original: this version
# takes `scene_radius` as a parameter (from scene.cameras_extent, which
# already computes the identical diagonal*1.1 quantity as GW's
# get_cameras_spatial_extent -- see scene/dataset_readers.py::getNerfppNorm)
# instead of recomputing it internally, to avoid porting a second copy of
# that utility.
# ---------------------------------------------------------------------------
def compute_nearest_cameras(
    train_cameras: List,
    scene_radius: float,
    multi_view_max_angle: float = 30.0,
    multi_view_min_dis_relative: float = 0.002,
    multi_view_max_dis_relative: float = 0.3,
    multi_view_num: int = 8,
) -> Dict[int, Dict[str, List]]:
    multi_view_min_dis = scene_radius * multi_view_min_dis_relative
    multi_view_max_dis = scene_radius * multi_view_max_dis_relative

    world_view_transforms = []
    camera_centers = []
    center_rays = []
    for cur_cam in train_cameras:
        world_view_transforms.append(cur_cam.world_view_transform)
        camera_centers.append(cur_cam.camera_center)
        R = torch.as_tensor(cur_cam.R, dtype=torch.float32, device="cuda")
        center_ray = torch.tensor([0.0, 0.0, 1.0], device="cuda")
        center_ray = center_ray @ R.transpose(-1, -2)
        center_rays.append(center_ray)

    camera_centers = torch.stack(camera_centers, dim=0)
    center_rays = torch.stack(center_rays, dim=0)
    center_rays = F.normalize(center_rays, dim=-1)

    diss = torch.norm(camera_centers[:, None] - camera_centers[None], dim=-1).detach().cpu().numpy()
    tmp = torch.sum(center_rays[:, None] * center_rays[None], dim=-1).clamp(-1.0, 1.0)
    angles = (torch.arccos(tmp) * 180.0 / math.pi).detach().cpu().numpy()

    nearest_cameras = {}
    for idx, cur_cam in enumerate(train_cameras):
        sorted_indices = np.lexsort((angles[idx], diss[idx]))
        mask = (
            (angles[idx][sorted_indices] < multi_view_max_angle)
            & (diss[idx][sorted_indices] > multi_view_min_dis)
            & (diss[idx][sorted_indices] < multi_view_max_dis)
        )
        sorted_indices = sorted_indices[mask]
        n = min(multi_view_num, len(sorted_indices))
        nearest_cameras[idx] = {
            "nearest_id": [int(i) for i in sorted_indices[:n]],
            "nearest_names": [train_cameras[i].image_name for i in sorted_indices[:n]],
        }
    return nearest_cameras


# ---------------------------------------------------------------------------
# ref-view <-> neighbor-view extrinsics (pure R/T composition, no projection
# model involved -- same formula GW itself uses at multiview_gggs.py:266-267,
# ported verbatim since it's projection-model-independent camera algebra).
# ---------------------------------------------------------------------------
def _ref_to_neighbor_RT(ref_cam, neighbor_cam) -> Tuple[torch.Tensor, torch.Tensor]:
    R_rn = neighbor_cam.world_view_transform[:3, :3].transpose(-1, -2) @ ref_cam.world_view_transform[:3, :3]
    T_rn = -R_rn @ ref_cam.world_view_transform[3, :3] + neighbor_cam.world_view_transform[3, :3]
    return R_rn, T_rn


# Public alias (train.py wiring uses this name; the leading-underscore name
# above is kept for tests/test_multiview.py's existing references).
ref_to_neighbor_RT = _ref_to_neighbor_RT


# ---------------------------------------------------------------------------
# geo_loss -- reimplementation of multiview_gggs.py's sample_depth +
# PatchMatch.__call__'s geo-consistency term (:91-153, :196-260), using
# get_ray_dirs_view for unprojection and project_view_to_pixel for
# reprojection instead of GW's raw perspective-divide (Fx/Fy/Cx/Cy) formulas,
# so this works for EQ as well as PH. Per the brief's expected_depth note:
# GW's own "ours" renderer sets expected_depth == median_depth, so we use
# median_depth directly on both sides (no depth_ratio blend needed).
# ---------------------------------------------------------------------------
def geo_loss(
    ref_cam,
    neighbor_cam,
    ref_render_pkg: Dict[str, torch.Tensor],
    neighbor_render_pkg: Dict[str, torch.Tensor],
    pixel_noise_th: float = 1.0,
    znear_relative: float = 0.02,
    scene_radius: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (geo_loss scalar, mask (H,W) bool, weights (H,W) float)."""
    H, W = ref_cam.image_height, ref_cam.image_width
    device = ref_render_pkg["median_depth"].device
    znear = znear_relative * scene_radius

    dirs_ref = get_ray_dirs_view(ref_cam)                       # (3,H,W)
    depth_ref = ref_render_pkg["median_depth"]                  # (1,H,W)
    pts_ref_view = (dirs_ref * depth_ref).permute(1, 2, 0).reshape(-1, 3)  # (H*W,3)

    R_rn, T_rn = _ref_to_neighbor_RT(ref_cam, neighbor_cam)
    pts_neighbor_view = pts_ref_view @ R_rn.transpose(-1, -2) + T_rn        # X_n = R@X_r + T

    u_n, v_n, valid_proj = project_view_to_pixel(neighbor_cam, pts_neighbor_view)

    Hn, Wn = neighbor_cam.image_height, neighbor_cam.image_width
    gx = (2.0 * u_n + 1.0) / Wn - 1.0
    gy = (2.0 * v_n + 1.0) / Hn - 1.0
    grid = torch.stack([gx, gy], dim=-1).view(1, -1, 1, 2)

    neighbor_depth = neighbor_render_pkg["median_depth"]        # (1,Hn,Wn)
    sampled_depth = F.grid_sample(
        neighbor_depth[None], grid, mode="bilinear", padding_mode="border", align_corners=False
    ).view(-1)

    dirs_neighbor = get_ray_dirs_view(neighbor_cam)              # (3,Hn,Wn)
    sampled_dir = F.grid_sample(
        dirs_neighbor[None], grid, mode="bilinear", padding_mode="border", align_corners=False
    ).view(3, -1).transpose(0, 1)                                # (N,3)

    pts_neighbor_from_depth = sampled_dir * sampled_depth.unsqueeze(-1)  # (N,3) neighbor view space

    R_nr = R_rn.transpose(-1, -2)
    T_nr = -R_nr @ T_rn
    pts_back_ref_view = pts_neighbor_from_depth @ R_nr.transpose(-1, -2) + T_nr

    u_back, v_back, valid_back = project_view_to_pixel(ref_cam, pts_back_ref_view)

    ys, xs = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing="ij",
    )
    pixel_noise = torch.sqrt((u_back - xs.reshape(-1)) ** 2 + (v_back - ys.reshape(-1)) ** 2)

    depth_ref_flat = depth_ref.view(-1)
    mask = (
        valid_proj
        & valid_back
        & (sampled_depth > 0)
        & (pts_neighbor_from_depth.norm(dim=-1) > znear)
        & (pts_back_ref_view.norm(dim=-1) > znear)
        & (pixel_noise < pixel_noise_th)
        & (depth_ref_flat > 0)
    )
    weights = torch.exp(-pixel_noise)
    weights = torch.where(mask, weights, torch.zeros_like(weights))

    if not bool(mask.any()):
        return torch.zeros((), device=device), mask.view(H, W), weights.view(H, W)

    loss = (weights * pixel_noise)[mask].mean()
    return loss, mask.view(H, W), weights.view(H, W)


# ---------------------------------------------------------------------------
# Pure-PyTorch reference oracle for the NCC math -- TEST-ONLY (not a
# production code path). Implements the exact same per-offset math as the
# CUDA kernel (submodules/multiview_ncc/cuda_multiview_ncc/
# multiview_ncc_impl.cu), composed from get_ray_dirs_view + grid_sample +
# project_view_to_pixel, used to numerically verify the CUDA kernel's
# forward output (and, via ordinary autograd, its finite-difference
# backward) on synthetic geometry. See tests/test_multiview.py.
# ---------------------------------------------------------------------------
def ncc_reference_oracle(
    ref_cam,
    neighbor_cam,
    depths: torch.Tensor,          # (P,)
    normals: torch.Tensor,         # (P,3) ref-view-space, need not be unit
    pixels: torch.Tensor,          # (P,2) int (x,y) ref-image center pixels
    image_r: torch.Tensor,         # (Hr,Wr)
    image_n: torch.Tensor,         # (Hn,Wn)
    patch_radius: int = 3,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns (ncc (P,), valid (P,) bool). Differentiable w.r.t. depths/normals."""
    device = depths.device
    P = depths.shape[0]
    Hr, Wr = ref_cam.image_height, ref_cam.image_width
    Hn, Wn = neighbor_cam.image_height, neighbor_cam.image_width
    R_rn, T_rn = _ref_to_neighbor_RT(ref_cam, neighbor_cam)

    dirs_ref = get_ray_dirs_view(ref_cam)  # (3,Hr,Wr)
    pixels = pixels.long()

    radius = patch_radius
    offsets = torch.arange(-radius, radius + 1, device=device)
    dv, du = torch.meshgrid(offsets, offsets, indexing="ij")   # (L,L)
    L = 2 * radius + 1

    px = pixels[:, 0].view(P, 1, 1) + du.view(1, L, L)   # (P,L,L)
    py = pixels[:, 1].view(P, 1, 1) + dv.view(1, L, L)

    in_bounds = (px >= 0) & (px <= Wr - 1) & (py >= 0) & (py <= Hr - 1)
    px_c = px.clamp(0, Wr - 1)
    py_c = py.clamp(0, Hr - 1)

    ray_off = dirs_ref[:, py_c, px_c].permute(1, 2, 3, 0)        # (P,L,L,3)
    ray_c = dirs_ref[:, pixels[:, 1].long(), pixels[:, 0].long()].permute(1, 0)  # (P,3)

    P0 = ray_c * depths.view(P, 1)                               # (P,3)
    N = normals                                                  # (P,3)
    N_dot_P0 = (N * P0).sum(-1, keepdim=True)                    # (P,1)

    denom = (N.view(P, 1, 1, 3) * ray_off).sum(-1)               # (P,L,L)
    bad_denom = denom.abs() < 1e-8
    safe_denom = torch.where(bad_denom, torch.full_like(denom, 1e-8), denom)
    t = N_dot_P0.view(P, 1, 1) / safe_denom                      # (P,L,L)
    bad = bad_denom | (t <= 0)

    X = ray_off * t.unsqueeze(-1)                                # (P,L,L,3)
    Xn = X @ R_rn.transpose(-1, -2) + T_rn                       # (P,L,L,3)

    u_n, v_n, proj_valid = project_view_to_pixel(neighbor_cam, Xn.reshape(-1, 3))
    u_n = u_n.view(P, L, L)
    v_n = v_n.view(P, L, L)
    proj_valid = proj_valid.view(P, L, L)

    gx = (2.0 * u_n + 1.0) / Wn - 1.0
    gy = (2.0 * v_n + 1.0) / Hn - 1.0
    grid = torch.stack([gx, gy], dim=-1).view(1, P * L * L, 1, 2)
    c_n = F.grid_sample(
        image_n[None, None].expand(1, 1, Hn, Wn), grid,
        mode="bilinear", padding_mode="border", align_corners=False,
    ).view(P, L, L)

    c_r = image_r[py_c, px_c]                                    # (P,L,L)

    all_inside = in_bounds & proj_valid & (~bad)
    all_inside = all_inside.view(P, -1).all(dim=-1)              # (P,)

    total = float(L * L)
    sum_c_r = c_r.reshape(P, -1).sum(-1)
    sum_c_n = c_n.reshape(P, -1).sum(-1)
    sum_c_r2 = (c_r * c_r).reshape(P, -1).sum(-1)
    sum_c_n2 = (c_n * c_n).reshape(P, -1).sum(-1)
    sum_c_rn = (c_r * c_n).reshape(P, -1).sum(-1)

    cross = sum_c_rn - sum_c_r * sum_c_n / total
    variance_r = sum_c_r2 - sum_c_r * sum_c_r / total
    variance_n = sum_c_n2 - sum_c_n * sum_c_n / total
    ncc = cross * cross / (variance_r * variance_n + 1e-8)

    valid = all_inside & (variance_r > 5e-6) & (variance_n > 5e-6)
    ncc = torch.where(valid, ncc, torch.zeros_like(ncc))
    return ncc, valid


# ---------------------------------------------------------------------------
# EQWarpPatchNCC -- torch.autograd.Function wrapping the CUDA forward kernel
# with a FINITE-DIFFERENCE backward (Phase 1, see 3DGEERGW_EXECUTION.md
# Session F2 "backward-gradient strategy"). Structural template:
# submodules/GaussianWrapping/submodules/Geometry-Grounded-Gaussian-Splatting/
# submodules/warp-patch-ncc/warp_patch_ncc/__init__.py::_WarpPatchNCC -- its
# ANALYTIC backward is what's being replaced here, not its overall shape.
#
# Phase 2 (IMPLEMENTED, opt-in): the hand-derived analytic EQ/PH backward
# now exists as _C.multiview_ncc_backward (see cuda_multiview_ncc/
# multiview_ncc_impl.cu::multiview_ncc_backward_kernel for the derivation)
# and is selected via MULTIVIEW_NCC_BACKWARD="analytic" (module constant /
# env var above; default remains "fd" pending the A/B-run evidence + user
# decision to flip). This finite-difference approach costs
# ~9x the forward-kernel cost per active iteration (1 center + 2 depth + 6
# normal-component evaluations via central differences), which was judged an
# acceptable trade against the derivation-bug risk of hand-deriving a full
# atan2-chain analytic gradient (rejected explicitly, see the Session F2
# "CORRECTION" journal entry) -- each of the 9 calls is itself a fast fused
# CUDA kernel, so this is still much cheaper than a naive pure-PyTorch
# reimplementation of the whole patch-warp+NCC computation.
#
# eps_depth/eps_normal default = 3e-2, NOT 1e-3 -- empirically tuned (see
# tests/test_multiview.py check_backward_vs_oracle_autograd and the Session
# F2 journal entry): near a converged match, ncc is very close to its local
# max (~1.0) and the TRUE gradient is legitimately tiny (the loss surface is
# a shallow plateau there). At eps=1e-3, central-difference noise from
# float32 rounding in the 49-term patch sums (absolute float32 precision
# near ncc~1 is ~1e-7, divided by 2*eps=2e-3 gives a gradient noise floor of
# ~5e-5 -- but the *true* derivative near convergence is often smaller than
# that) completely swamps the real signal; eps=3e-2 was the smallest value
# where finite differences reliably tracked the oracle's exact autograd
# gradient (verified by direct eps-sweep, not assumed).
# ---------------------------------------------------------------------------
class EQWarpPatchNCC(torch.autograd.Function):
    """Backward mode is selected by the module-level MULTIVIEW_NCC_BACKWARD
    constant (env-overridable, default "fd") read at forward() time:

    - "fd" (Phase 1, production default): 9 forward-kernel calls at forward
      time (center + central differences on depth and the 3 normal
      components at the calibrated eps=3e-2 -- see the eps rationale in the
      comment block above); backward is a cheap multiply with the stashed
      FD gradients.
    - "analytic" (Phase 2): single forward-kernel call at forward time;
      backward calls the hand-derived analytic CUDA backward kernel
      (_C.multiview_ncc_backward), which recomputes the forward reduction
      per point and chains NCC -> patch stats -> bilinear -> PH/EQ
      projection Jacobian -> ray-plane intersection analytically. Exact
      (matches the pure-torch oracle's autograd to float32 noise), no eps.
    """
    @staticmethod
    def forward(ctx, depths, normals, uvs, ray_dirs_r, R, T, image_r, image_n,
                 render_model_n, fx_n, fy_n, cx_n, cy_n, patch_radius,
                 eps_depth=3e-2, eps_normal=3e-2):
        if not _HAVE_MULTIVIEW_NCC_EXT:
            raise RuntimeError(
                "multiview_ncc CUDA extension not built -- see "
                "submodules/multiview_ncc/setup.py "
                "(pip install --no-build-isolation -e .)"
            )

        def _fwd(d, n):
            return _multiview_ncc_ext.multiview_ncc_forward(
                d, n, uvs, ray_dirs_r, R, T, image_r, image_n,
                render_model_n, fx_n, fy_n, cx_n, cy_n, patch_radius,
            )

        ncc, valid = _fwd(depths, normals)

        mode = MULTIVIEW_NCC_BACKWARD
        ctx.backward_mode = mode
        if mode == "analytic":
            ctx.kernel_args = (render_model_n, fx_n, fy_n, cx_n, cy_n, patch_radius)
            ctx.save_for_backward(depths, normals, uvs, ray_dirs_r, R, T,
                                  image_r, image_n)
            return ncc, valid

        with torch.no_grad():
            ncc_dp, _ = _fwd(depths + eps_depth, normals)
            ncc_dm, _ = _fwd(depths - eps_depth, normals)
            grad_depths = (ncc_dp - ncc_dm) / (2.0 * eps_depth)

            grad_normals = torch.zeros_like(normals)
            for k in range(3):
                n_p = normals.clone(); n_p[:, k] += eps_normal
                n_m = normals.clone(); n_m[:, k] -= eps_normal
                ncc_np, _ = _fwd(depths, n_p)
                ncc_nm, _ = _fwd(depths, n_m)
                grad_normals[:, k] = (ncc_np - ncc_nm) / (2.0 * eps_normal)

        ctx.save_for_backward(grad_depths, grad_normals)
        return ncc, valid

    @staticmethod
    def backward(ctx, grad_ncc, grad_valid):
        if ctx.backward_mode == "analytic":
            (depths, normals, uvs, ray_dirs_r, R, T,
             image_r, image_n) = ctx.saved_tensors
            render_model_n, fx_n, fy_n, cx_n, cy_n, patch_radius = ctx.kernel_args
            grad_depths, grad_normals = _multiview_ncc_ext.multiview_ncc_backward(
                depths, normals, uvs, ray_dirs_r, R, T, image_r, image_n,
                render_model_n, fx_n, fy_n, cx_n, cy_n, patch_radius,
                grad_ncc.contiguous().float(),
                precise=MULTIVIEW_NCC_ANALYTIC_PRECISE,
            )
            # The analytic kernel already multiplies by the upstream grad_ncc
            # inside the kernel (g_upstream), so return its outputs directly.
            return (grad_depths, grad_normals) + (None,) * 14

        grad_depths, grad_normals = ctx.saved_tensors
        # 16 forward inputs (depths, normals, uvs, ray_dirs_r, R, T, image_r,
        # image_n, render_model_n, fx_n, fy_n, cx_n, cy_n, patch_radius,
        # eps_depth, eps_normal) -> 16 return values, grad only for the first two.
        return (
            grad_ncc * grad_depths,
            grad_ncc.unsqueeze(-1) * grad_normals,
            None,  # uvs
            None,  # ray_dirs_r
            None,  # R
            None,  # T
            None,  # image_r
            None,  # image_n
            None,  # render_model_n
            None,  # fx_n
            None,  # fy_n
            None,  # cx_n
            None,  # cy_n
            None,  # patch_radius
            None,  # eps_depth
            None,  # eps_normal
        )


def eq_warp_patch_ncc(depths, normals, uvs, ray_dirs_r, R, T, image_r, image_n,
                       render_model_n, fx_n, fy_n, cx_n, cy_n, patch_radius,
                       eps_depth=3e-2, eps_normal=3e-2):
    return EQWarpPatchNCC.apply(
        depths, normals, uvs, ray_dirs_r, R, T, image_r, image_n,
        render_model_n, fx_n, fy_n, cx_n, cy_n, patch_radius,
        eps_depth, eps_normal,
    )
