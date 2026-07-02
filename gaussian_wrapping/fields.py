# ported/adapted from GaussianWrapping gaussian_wrapping/regularization/sdf/depth_fusion.py
# (transform_points_world_to_view, AdaptiveTSDF.integrate)
"""TSDF depth-fusion field evaluated at arbitrary 3D points, fisheye-aware.

Convention (matches GaussianWrapping's AdaptiveTSDF): for a single camera,
``sdf_cam = d_obs - ||x_view||`` where ``d_obs`` is the camera's observed
(median) depth sampled at the point's projection and ``||x_view||`` is the
point's own distance from the camera. sdf_cam > 0 means the point is nearer
to the camera than the observed surface (i.e. in front of / outside the
surface -> "empty"); sdf_cam < 0 means the point is farther than the
observed surface (i.e. behind it -> "occupied"). Truncated to
[-1, 1] * trunc_margin and averaged with equal weight (1.0) across all
cameras for which the point (a) projects inside the image, (b) has a valid
(> 0) observed depth, and (c) has truncated sdf >= -1 (GaussianWrapping drops
points that fall more than one truncation margin *behind* the observed
surface from the running average -- see AdaptiveTSDF.integrate, the
`valid_mask & (sdf[..., 0] >= -1.)` line -- so a single very-close occluding
surface cannot poison the average for a point that is actually far behind a
*different* surface in that view).

field > 0 = empty space, field < 0 = occupied, surface at 0.
Points seen by no camera default to occupied (tsdf = -1), matching
AdaptiveTSDF's `initial_sdf_value=-1.0` default.
"""
from typing import List, Optional

import torch
import torch.nn.functional as F

from gaussian_wrapping.fisheye_proj import project_view_to_pixel


def world_to_view(camera, points: torch.Tensor) -> torch.Tensor:
    """World-space (N,3) -> this camera's view-space (N,3).

    Uses the same row-vector convention as GaussianWrapping's
    `transform_points_world_to_view` (points_h @ world_view_transform),
    which is consistent with the Session-C-pinned view->world rotation
    convention `view_normals_to_world(cam, n) = world_view_transform[:3,:3] @ n`
    (see utils/ray_normals.py): a point AT camera.camera_center maps to
    x_view ~= 0, and a point 5 units along the camera's forward axis in
    world space maps to x_view ~= (0, 0, 5). Both are asserted in
    tests/test_fisheye_proj.py.
    """
    ones = torch.ones(points.shape[0], 1, device=points.device, dtype=points.dtype)
    points_h = torch.cat([points, ones], dim=-1)  # (N,4)
    wvt = camera.world_view_transform.to(device=points.device, dtype=points.dtype)
    view_points = points_h @ wvt  # (N,4)
    return view_points[:, :3]


@torch.no_grad()
def render_depth_maps(cameras: List, gaussians, pipe) -> List[torch.Tensor]:
    """Render and return the median_depth map (1,H,W) for each camera.

    A training job may be sharing the GPU (see 3DGEERGW_EXECUTION.md Session
    E hard rules): retry once after `torch.cuda.empty_cache()` on OOM.
    """
    from gaussian_renderer import render

    background = torch.zeros(3, device="cuda")
    depth_maps = []
    for cam in cameras:
        try:
            pkg = render(cam, gaussians, pipe, background)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            pkg = render(cam, gaussians, pipe, background)
        depth_maps.append(pkg["median_depth"].detach())
    return depth_maps


def fuse_tsdf_at_points(
    points: torch.Tensor,
    cameras: List,
    depth_maps: List[torch.Tensor],
    trunc_margin: float,
    chunk_size: Optional[int] = 500_000,
) -> torch.Tensor:
    """Fuse per-camera rendered median-depth maps into a TSDF at `points`.

    Args:
        points (torch.Tensor): (N, 3) world-space query points.
        cameras (List[Camera]): cameras aligned 1:1 with `depth_maps`.
        depth_maps (List[torch.Tensor]): each (1, H, W) (or (H, W)),
            typically from `render_depth_maps`.
        trunc_margin (float): truncation distance (world units).
        chunk_size: process points in chunks to bound peak memory
            (H*W grid_sample grids scale with N); None disables chunking.

    Returns:
        torch.Tensor: (N,) tsdf values in [-1, 1].
    """
    assert points.shape[-1] == 3
    assert len(cameras) == len(depth_maps)
    N = points.shape[0]
    device = points.device

    if chunk_size is None or N <= chunk_size:
        chunks = [points]
    else:
        chunks = list(torch.split(points, chunk_size, dim=0))

    out_chunks = []
    for chunk in chunks:
        n = chunk.shape[0]
        tsdf_sum = torch.zeros(n, device=device)
        weight_sum = torch.zeros(n, device=device)

        for cam, depth_map in zip(cameras, depth_maps):
            x_view = world_to_view(cam, chunk)  # (n,3)
            u, v, valid_proj = project_view_to_pixel(cam, x_view)

            H, W = cam.image_height, cam.image_width
            dm = depth_map.to(device=device, dtype=chunk.dtype).view(1, 1, H, W)

            # align_corners=False mapping consistent with the -0.5 pixel-center
            # convention baked into project_view_to_pixel (u,v = pixel index).
            gx = (2.0 * u + 1.0) / W - 1.0
            gy = (2.0 * v + 1.0) / H - 1.0
            grid = torch.stack([gx, gy], dim=-1).view(1, n, 1, 2)

            d_obs = F.grid_sample(
                dm, grid, mode="bilinear", padding_mode="border", align_corners=False
            ).view(n)

            r = x_view.norm(dim=-1)
            sdf_cam = (d_obs - r) / trunc_margin

            valid = valid_proj & (d_obs > 0) & (sdf_cam >= -1.0)
            sdf_cam = sdf_cam.clamp(max=1.0)

            w = valid.float()
            tsdf_sum = tsdf_sum + sdf_cam * w
            weight_sum = weight_sum + w

        tsdf = torch.where(
            weight_sum > 0, tsdf_sum / weight_sum.clamp(min=1e-8),
            torch.full_like(tsdf_sum, -1.0),
        )
        out_chunks.append(tsdf)

    return torch.cat(out_chunks, dim=0)
