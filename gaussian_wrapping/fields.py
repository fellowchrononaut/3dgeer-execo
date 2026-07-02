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
import math
from typing import List, Optional

import torch
import torch.nn.functional as F

from gaussian_wrapping.fisheye_proj import project_view_to_pixel
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer, integrate_points

# geer-rasterizer tile size (cuda_rasterizer/config.h BLOCK_X/BLOCK_Y);
# the Python-side tile binning below MUST match this exactly.
_BLOCK_X = 16
_BLOCK_Y = 16


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


# ---------------------------------------------------------------------------
# Session E2: exact ray-integrated occupancy field.
# ported/adapted from GaussianWrapping submodules/diff-gaussian-rasterization/
# cuda_rasterizer/forward.cu (integrateCUDA / Rasterizer::integrate) and
# gaussian_wrapping/gaussian_renderer/ours.py (integrate_ours) -- see the
# "Session E2" section of 3DGEERGW_EXECUTION.md for the exact semantics
# ported: per query point x, per view v, A_v(x) = accumulated alpha along
# the camera ray through x's own pixel, counting only Gaussians whose
# (sorted) center depth <= t_x = ||x_view||; occupancy(x) = min over valid
# views of A_v(x) (points valid in no view -> occupancy 0). This is GW's
# actual quality extraction path ("ours"/exact_computation); the TSDF
# depth-fusion field above is only their initialization helper.
#
# field(x) = 0.5 + iso - occupancy(x); surface at field = 0 (SAME sign
# convention as fuse_tsdf_at_points above: field > 0 = empty, < 0 = occupied).
# ---------------------------------------------------------------------------

def _build_raster_settings(camera, pc, pipe, bg_color: torch.Tensor) -> GaussianRasterizationSettings:
    """Minimal replica of gaussian_renderer.render()'s raster_settings
    construction (kept in sync by hand -- factored out here rather than
    refactoring render() itself so train.py's hot path stays untouched)."""
    tanfovx = math.tan(camera.FoVx * 0.5)
    tanfovy = math.tan(camera.FoVy * 0.5)
    return GaussianRasterizationSettings(
        image_height=camera.image_height,
        image_width=camera.image_width,
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=1.0,
        viewmatrix=camera.world_view_transform,
        mirror_transformed_tan_theta=camera.mirror_transformed_tan_theta.cuda(),
        mirror_transformed_tan_phi=camera.mirror_transformed_tan_phi.cuda(),
        tan_theta=camera.tan_theta.cuda(),
        tan_phi=camera.tan_phi.cuda(),
        focal_x=float(camera.focal_x or 0.0),
        focal_y=float(camera.focal_y or 0.0),
        principal_x=float(camera.principal_x or 0.0),
        principal_y=float(camera.principal_y or 0.0),
        distortion_coeffs=camera.distortion_coeffs.cuda() if camera.render_model == 1 else torch.empty(0, device="cuda"),
        raymap=camera.raymap.cuda() if camera.render_model == 1 else torch.empty(0, device="cuda"),
        sh_degree=pc.active_sh_degree,
        campos=camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        antialiasing=pipe.antialiasing,
        render_mode=camera.render_model,
        near_threshold=0.2,
        asso_mode=0,
    )


@torch.no_grad()
def _forward_buffers_for_view(camera, gaussians, pipe):
    """No-grad forward pass that returns the raster_settings + rasterizer
    state buffers for one view (retried once after `empty_cache()` on OOM,
    matching `render_depth_maps`'s pattern above)."""
    raster_settings = _build_raster_settings(camera, gaussians, pipe, torch.zeros(3, device="cuda"))
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    P = gaussians.get_xyz.shape[0]
    colors_precomp = torch.zeros(P, 3, device="cuda")  # unused (no color output needed); skips SH eval
    kwargs = dict(
        means3D=gaussians.get_xyz,
        opacities=gaussians.get_opacity,
        colors_precomp=colors_precomp,
        scales=gaussians.get_scaling,
        rotations=gaussians.get_rotation,
    )
    try:
        buf = rasterizer.forward_with_buffers(**kwargs)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        buf = rasterizer.forward_with_buffers(**kwargs)
    return raster_settings, buf


@torch.no_grad()
def _integrate_alpha_chunk(camera, raster_settings, buf, x_view: torch.Tensor):
    """A_v(x) for one view's already-built buffers + one chunk of
    view-space points. Returns (alpha (n,), valid (n,) bool); `alpha` is
    only meaningful where `valid` is True."""
    n = x_view.shape[0]
    device = x_view.device
    u, v, valid = project_view_to_pixel(camera, x_view)
    alpha_full = torch.zeros(n, device=device)
    if not bool(valid.any()):
        return alpha_full, valid

    W, H = camera.image_width, camera.image_height
    grid_x = (W + _BLOCK_X - 1) // _BLOCK_X
    grid_y = (H + _BLOCK_Y - 1) // _BLOCK_Y
    num_tiles = grid_x * grid_y

    valid_idx = valid.nonzero(as_tuple=True)[0]
    px = u[valid_idx].round().long().clamp(0, W - 1)
    py = v[valid_idx].round().long().clamp(0, H - 1)
    pix_id = (py * W + px).to(torch.int32).contiguous()
    tval = x_view[valid_idx].norm(dim=-1).float().contiguous()

    tile = (py // _BLOCK_Y) * grid_x + (px // _BLOCK_X)
    order = torch.argsort(tile)
    tile_sorted = tile[order]
    q_point_order = order.to(torch.int32).contiguous()  # indices into (pix_id, tval)

    counts = torch.bincount(tile_sorted, minlength=num_tiles)
    ends = torch.cumsum(counts, dim=0).to(torch.int32)
    starts = ends - counts.to(torch.int32)
    q_ranges = torch.stack([starts, ends], dim=-1).to(torch.int32).contiguous()

    alpha_valid = integrate_points(
        raster_settings, buf["geomBuffer"], buf["binningBuffer"], buf["imgBuffer"],
        buf["num_rendered"], buf["P"],
        pix_id, tval, q_ranges, q_point_order,
    )  # (n_valid,) aligned to valid_idx's own order

    alpha_full[valid_idx] = alpha_valid
    return alpha_full, valid


@torch.no_grad()
def evaluate_occupancy_integrated(
    points: torch.Tensor,
    cameras: List,
    gaussians,
    pipe,
    iso: float = 0.0,
    chunk: int = 2_000_000,
) -> torch.Tensor:
    """Exact ray-integrated occupancy field at `points` (N,3) world-space.

    One forward render (buffer build) per camera (cannot cache all views'
    8M-Gaussian state at once); each point chunk within that view reuses the
    same buffers. occupancy(x) = min over valid views of A_v(x); points
    valid in no view get occupancy 0 (vacant). Returns field = 0.5 + iso -
    occupancy (SAME sign convention as fuse_tsdf_at_points: >0 empty, <0
    occupied, surface at 0).
    """
    assert points.shape[-1] == 3
    N = points.shape[0]
    device = points.device

    occupancy = torch.zeros(N, device=device)
    any_valid = torch.zeros(N, dtype=torch.bool, device=device)

    for camera in cameras:
        raster_settings, buf = _forward_buffers_for_view(camera, gaussians, pipe)

        for start in range(0, N, chunk):
            end = min(start + chunk, N)
            x_view = world_to_view(camera, points[start:end])
            alpha, valid = _integrate_alpha_chunk(camera, raster_settings, buf, x_view)

            occ_slice = occupancy[start:end]
            valid_prev = any_valid[start:end]
            both_valid = valid & valid_prev
            updated = torch.where(
                both_valid, torch.minimum(occ_slice, alpha),
                torch.where(valid, alpha, occ_slice),
            )
            occupancy[start:end] = updated
            any_valid[start:end] = valid_prev | valid

    occupancy = torch.where(any_valid, occupancy, torch.zeros_like(occupancy))
    return 0.5 + iso - occupancy
