# ported/adapted (structural template only) from GaussianWrapping
# submodules/Geometry-Grounded-Gaussian-Splatting/submodules/warp-patch-ncc/
# warp_patch_ncc/__init__.py -- unlike that file, this is a THIN forward-only
# wrapper with no autograd.Function: the finite-difference backward (Phase 1,
# see GUTWrap_Discussion/3DGEERGW_EXECUTION.md Session F2) lives in
# utils/multiview.py::EQWarpPatchNCC, which calls `multiview_ncc_forward`
# below multiple times per training step.
import torch
from . import _C


def multiview_ncc_forward(
    depths: torch.Tensor,
    normals: torch.Tensor,
    uvs: torch.Tensor,
    ray_dirs_r: torch.Tensor,
    R: torch.Tensor,
    T: torch.Tensor,
    image_r: torch.Tensor,
    image_n: torch.Tensor,
    render_model_n: int,
    fx_n: float,
    fy_n: float,
    cx_n: float,
    cy_n: float,
    patch_radius: int,
    debug: bool = False,
):
    """Forward-only EQ/PH-general multiview patch-NCC. Returns (ncc (P,),
    valid (P,) bool). No gradients attached here -- see
    utils/multiview.py::EQWarpPatchNCC for the finite-difference
    torch.autograd.Function wrapper used in training."""
    ncc, valid = _C.multiview_ncc_forward(
        depths.contiguous(),
        normals.contiguous(),
        uvs.contiguous().int(),
        ray_dirs_r.contiguous(),
        R.contiguous(),
        T.contiguous(),
        image_r.contiguous(),
        image_n.contiguous(),
        int(render_model_n),
        float(fx_n), float(fy_n), float(cx_n), float(cy_n),
        int(patch_radius),
        bool(debug),
    )
    return ncc, valid
