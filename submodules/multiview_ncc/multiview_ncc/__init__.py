# ported/adapted (structural template only) from GaussianWrapping
# submodules/Geometry-Grounded-Gaussian-Splatting/submodules/warp-patch-ncc/
# warp_patch_ncc/__init__.py -- unlike that file, this is a THIN wrapper with
# no autograd.Function: both the Phase-1 finite-difference backward and the
# Phase-2 analytic backward (see GUTWrap_Discussion/3DGEERGW_EXECUTION.md
# Session F2) live in utils/multiview.py::EQWarpPatchNCC, which calls
# `multiview_ncc_forward` (Phase 1, multiple times per training step) and/or
# `multiview_ncc_backward` (Phase 2, once per backward call) below.
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


def multiview_ncc_backward(
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
    grad_ncc: torch.Tensor,
    precise: bool = True,
    debug: bool = False,
):
    """Phase 2 analytic backward: recomputes the forward pass internally and
    returns (grad_depths (P,), grad_normals (P,3)) w.r.t. `depths`/`normals`,
    given the upstream dL/dncc in `grad_ncc` (P,). `precise=True` (default)
    runs double internals (matches a float64 oracle autograd to machine
    precision; roughly FD-backward speed on consumer fp64-throttled GPUs);
    `precise=False` runs float internals (much faster; f32 noise floor --
    still far tighter than the eps=3e-2 FD backward). See
    utils/multiview.py::EQWarpPatchNCC (MULTIVIEW_NCC_BACKWARD="analytic")
    for the autograd.Function wiring, and cuda_multiview_ncc/
    multiview_ncc_impl.cu::multiview_ncc_backward_kernel for the derivation."""
    grad_depths, grad_normals = _C.multiview_ncc_backward(
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
        grad_ncc.contiguous(),
        bool(precise),
        bool(debug),
    )
    return grad_depths, grad_normals
