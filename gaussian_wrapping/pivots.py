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
from typing import Optional, Tuple

import torch

from utils.general_utils import build_rotation


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
    P = means.shape[0]

    if max_pivots is not None and 2 * P > max_pivots:
        keep = max(1, max_pivots // 2)
        if opacities is None:
            raise ValueError(
                "extract_gaussian_pivots: max_pivots requires `opacities` to "
                "select which Gaussians to keep (largest-opacity subsampling)."
            )
        opac = opacities.view(-1)
        keep_idx = torch.topk(opac, keep, largest=True).indices
        means = means[keep_idx]
        scales = scales[keep_idx]
        rotations = rotations[keep_idx]
        normals = normals[keep_idx]

    normals = torch.nn.functional.normalize(normals, dim=-1)  # (P,3)
    R = build_rotation(rotations)  # (P,3,3), columns = local axes in world space

    # sigma_i = || S_i . (R_i^T . n_i) ||
    Rt_n = torch.bmm(R.transpose(1, 2), normals.unsqueeze(-1)).squeeze(-1)  # (P,3)
    normal_std = (scales * Rt_n).norm(dim=-1)  # (P,)

    front = means + std_factor * normal_std.unsqueeze(-1) * normals  # (P,3)
    pivots = torch.cat([front, means], dim=0)  # (2P,3)

    pivot_scale = 3.0 * scales.detach().max(dim=-1, keepdim=True).values  # (P,1)
    pivot_scales = torch.cat([pivot_scale, pivot_scale], dim=0)  # (2P,1)

    return pivots, pivot_scales
