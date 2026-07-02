# ported/adapted from GaussianWrapping gaussian_wrapping/extraction/pivots.py
# (get_intersecting_pivots_from_normals, n_pivots=2 case) and
# gaussian_wrapping/regularization/normal_field.py (get_gaussian_std_in_direction).
#
# NOTE on file attribution: 3DGEERGW_EXECUTION.md Session E / the task brief
# point at `functional/pivots.py::extract_gaussian_pivots` for "the exact
# 2-pivot formula", but the on-disk `functional/pivots.py::extract_gaussian_pivots`
# actually produces 9 pivots per Gaussian from an *axis-aligned box* (8 corners
# + center; no normals involved at all -- see submodules/GaussianWrapping/
# gaussian_wrapping/functional/pivots.py:178). The "2 pivots: center + offset
# along the oriented normal, std_factor=3.0" formula the task describes is
# `extraction/pivots.py::get_intersecting_pivots_from_normals(n_pivots=2, ...)`.
# This is drift in the planning doc (same class of drift as D4 in the
# execution plan), not a real ambiguity in the requested formula -- ported
# from the correct (normal-based) function below.
"""2-pivot-per-Gaussian extraction for pivot-based marching tetrahedra.

For each Gaussian i with center mu_i, world-space rotation R_i (columns =
local axes in world space, build_rotation() convention), diagonal scale
s_i = (sx, sy, sz), and unit oriented normal n_i, GaussianWrapping computes
the Gaussian's standard deviation along the normal direction as

    sigma_i = || S_i . R_i^T . n_i ||        (S_i = diag(s_i))

(regularization/normal_field.py::get_gaussian_std_in_direction, the
`transposed_scaled_rotation = (R @ S)^T = S @ R^T` branch) and then spawns
two pivots per Gaussian:

    pivot_front_i = mu_i + std_factor * sigma_i * n_i     (std_factor = 3.0)
    pivot_center_i = mu_i

(extraction/pivots.py::get_intersecting_pivots_from_normals, n_pivots=2,
n_pivots_no_center == 1 branch).
"""
from typing import Callable, Optional, Tuple

import torch

from utils.general_utils import build_rotation


def _subsample_by_opacity(means, scales, rotations, normals, opacities, max_pivots, n_pivots_per_gaussian):
    """Shared subsampling logic (used by both `extract_gaussian_pivots` and
    `get_searched_pivots`): if the requested pivot budget can't fit
    `n_pivots_per_gaussian` pivots per Gaussian, keep only the
    largest-opacity Gaussians."""
    P = means.shape[0]
    if max_pivots is not None and n_pivots_per_gaussian * P > max_pivots:
        keep = max(1, max_pivots // n_pivots_per_gaussian)
        if opacities is None:
            raise ValueError(
                "max_pivots requires `opacities` to select which Gaussians "
                "to keep (largest-opacity subsampling)."
            )
        keep_idx = torch.topk(opacities.view(-1), keep, largest=True).indices
        means, scales, rotations, normals = means[keep_idx], scales[keep_idx], rotations[keep_idx], normals[keep_idx]
    return means, scales, rotations, normals


def _std_along_direction(scales: torch.Tensor, rotations: torch.Tensor, directions: torch.Tensor) -> torch.Tensor:
    """sigma_i = || S_i . R_i^T . d_i || (regularization/normal_field.py::
    get_gaussian_std_in_direction, `normalize_directions=False` branch).
    `directions` must already be unit-normalized."""
    R = build_rotation(rotations)  # (P,3,3), columns = local axes in world space
    Rt_d = torch.bmm(R.transpose(1, 2), directions.unsqueeze(-1)).squeeze(-1)  # (P,3)
    return (scales * Rt_d).norm(dim=-1)  # (P,)


def extract_gaussian_pivots(
    means: torch.Tensor,
    scales: torch.Tensor,
    rotations: torch.Tensor,
    normals: torch.Tensor,
    opacities: Optional[torch.Tensor] = None,
    std_factor: float = 3.0,
    max_pivots: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract 2 pivots per Gaussian: center and normal-offset point.

    Args:
        means (torch.Tensor): (P, 3) Gaussian centers (world space).
        scales (torch.Tensor): (P, 3) Gaussian scales (world-space std along
            local axes, i.e. `gaussians.get_scaling`).
        rotations (torch.Tensor): (P, 4) Gaussian rotation quaternions
            (i.e. `gaussians.get_rotation`).
        normals (torch.Tensor): (P, 3) oriented world-space normals
            (i.e. `gaussians.get_normals`); need not be pre-normalized.
        opacities (torch.Tensor, optional): (P,) or (P, 1) opacities. Only
            required if `max_pivots` forces subsampling: Delaunay above
            ~1.5M points is the practical scipy.spatial.Delaunay ceiling, so
            when `2*P > max_pivots` the Gaussians with the largest opacity
            are kept.
        std_factor (float): GaussianWrapping's exact constant (3.0).
        max_pivots (int, optional): cap on the total number of returned
            pivots (2 per kept Gaussian).

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            pivots (2*P', 3): first P' rows are the normal-offset pivots,
                last P' rows are the Gaussian centers (P' <= P).
            pivot_scales (2*P', 1): GaussianWrapping's pivot scale
                convention, `3 * max(scales, dim=-1)`, repeated for both
                pivots of each kept Gaussian.
    """
    assert means.shape[-1] == 3 and scales.shape[-1] == 3 and rotations.shape[-1] == 4

    means, scales, rotations, normals = _subsample_by_opacity(
        means, scales, rotations, normals, opacities, max_pivots, n_pivots_per_gaussian=2)

    normals = torch.nn.functional.normalize(normals, dim=-1)  # (P,3)
    normal_std = _std_along_direction(scales, rotations, normals)  # (P,)

    front = means + std_factor * normal_std.unsqueeze(-1) * normals  # (P,3)
    pivots = torch.cat([front, means], dim=0)  # (2P,3)

    pivot_scale = 3.0 * scales.detach().max(dim=-1, keepdim=True).values  # (P,1)
    pivot_scales = torch.cat([pivot_scale, pivot_scale], dim=0)  # (2P,1)

    return pivots, pivot_scales


@torch.no_grad()
def get_searched_pivots(
    means: torch.Tensor,
    scales: torch.Tensor,
    rotations: torch.Tensor,
    normals: torch.Tensor,
    sdf_function: Callable[[torch.Tensor], torch.Tensor],
    opacities: Optional[torch.Tensor] = None,
    std_factor: float = 3.33,
    max_pivots: Optional[int] = None,
    search_iter: int = 5,
    step_size: float = 0.33,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Searched-pivot variant of `extract_gaussian_pivots` (ported from
    GaussianWrapping `extraction/pivots.py::get_searched_pivots`): when a
    Gaussian's center is itself occupied (sdf(center) <= 0) and the initial
    front pivot hasn't crossed the surface yet (same sdf sign as the
    center), the front pivot is walked further out along the normal in
    `step_size`-sized increments (up to `search_iter` times) until it
    crosses -- or the budget runs out. This finds a tighter surface
    bracket for thin/occluded structures than the fixed `std_factor` offset
    alone. Requires `search_iter + 2` calls to `sdf_function` in the worst
    case (2 initial evals + up to `search_iter` refinement evals on a
    shrinking candidate set), so it is intended for use with the exact
    ray-integrated field, not (necessarily) the fast TSDF preview.

    Args:
        sdf_function: callable (M,3) world points -> (M,) field values,
            SAME sign convention as `fuse_tsdf_at_points`/
            `evaluate_occupancy_integrated` (>0 empty, <0 occupied).
        Other args: SAME as `extract_gaussian_pivots` (GW's own defaults
            here are std_factor=3.33, search_iter=10, step_size=1.0; this
            port's pinned defaults -- search_iter=5, step_size=0.33 -- trade
            some search depth for wall-clock time, since each iteration is
            a full multi-view field evaluation; see 3DGEERGW_EXECUTION.md
            Session E2).

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: SAME shape/ordering convention as
            `extract_gaussian_pivots`: pivots (2*P',3) (front pivots first,
            centers last), pivot_scales (2*P',1).
    """
    assert means.shape[-1] == 3 and scales.shape[-1] == 3 and rotations.shape[-1] == 4

    means, scales, rotations, normals = _subsample_by_opacity(
        means, scales, rotations, normals, opacities, max_pivots, n_pivots_per_gaussian=2)

    normals = torch.nn.functional.normalize(normals, dim=-1)  # (P,3)
    normal_std = _std_along_direction(scales, rotations, normals)  # (P,)
    normal_ray = normal_std.unsqueeze(-1) * normals  # (P,3): one "unit" (std_factor=1) step

    center = means
    front0 = center + std_factor * normal_ray
    pivots0 = torch.stack([front0, center], dim=1).reshape(-1, 3)  # (2P,3) interleaved per-Gaussian
    sdf0 = sdf_function(pivots0).view(-1, 2)  # (P,2)
    front_sdf = sdf0[:, 0].clone()
    center_sdf = sdf0[:, 1]

    ray_multiple = torch.full_like(normal_std.unsqueeze(-1), std_factor)  # (P,1)
    center_is_occupied = center_sdf <= 0.0  # (P,)
    same_sign = (center_sdf * front_sdf) > 0.0  # (P,): front hasn't crossed the surface yet

    for _ in range(search_iter):
        search_mask = center_is_occupied & same_sign
        if not bool(search_mask.any()):
            break

        center_to_search = center[search_mask]
        ray_to_search = normal_ray[search_mask]
        mult_to_search = ray_multiple[search_mask] + step_size
        front_to_search = center_to_search + ray_to_search * mult_to_search
        ray_multiple[search_mask] = mult_to_search

        front_sdf_search = sdf_function(front_to_search)  # (M,)
        center_sdf_search = center_sdf[search_mask]
        front_sdf[search_mask] = front_sdf_search
        same_sign[search_mask] = (front_sdf_search * center_sdf_search) > 0.0

    front = center + normal_ray * ray_multiple  # (P,3), final searched front pivots
    pivots = torch.cat([front, center], dim=0)  # (2P,3)

    pivot_scale = 3.0 * scales.detach().max(dim=-1, keepdim=True).values  # (P,1)
    pivot_scales = torch.cat([pivot_scale, pivot_scale], dim=0)  # (2P,1)

    return pivots, pivot_scales
