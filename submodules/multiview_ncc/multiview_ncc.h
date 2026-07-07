// ported/adapted (structural template only) from GaussianWrapping
// submodules/Geometry-Grounded-Gaussian-Splatting/submodules/warp-patch-ncc/
// warp_patch_ncc.h -- the Torch-facing entry point declaration. The internal
// math (cuda_multiview_ncc/multiview_ncc_impl.cu) is new: a per-offset
// ray-intersect-reproject kernel generalized to PH and EQ (fisheye)
// projection models, NOT a plane-induced-homography warp (see
// GUTWrap_Discussion/3DGEERGW_EXECUTION.md Session F2 for why the homography
// approach in warp_patch_ncc cannot be reused for fisheye).
//
// This is a FORWARD-ONLY kernel (Phase 1 backward strategy: finite-difference
// the forward pass from Python, see utils/multiview.py::EQWarpPatchNCC). No
// analytic gradient is computed on the CUDA side.
#pragma once
#include <cstdio>
#include <string>
#include <torch/extension.h>
#include <tuple>

// render_model_n: 1 = KB/EQ (equidistant fisheye, undistorted), 2 = PH (pinhole).
// Matches scene/cameras.py::Camera.render_model / gaussian_wrapping/fisheye_proj.py.
std::tuple<torch::Tensor, torch::Tensor>
MultiviewNCC(const torch::Tensor& depths,       // (P,) float, ref median depth at center pixel
             const torch::Tensor& normals,      // (P,3) float, ref-view-space normal (need not be unit)
             const torch::Tensor& uvs,          // (P,2) int32, ref center pixel (x, y)
             const torch::Tensor& ray_dirs_r,   // (Hr,Wr,3) float, ref view-space unit ray dirs
             const torch::Tensor& R,            // (3,3) float, ref-view -> neighbor-view rotation
             const torch::Tensor& T,            // (3,) float, ref-view -> neighbor-view translation
             const torch::Tensor& image_r,      // (Hr,Wr) float grayscale
             const torch::Tensor& image_n,      // (Hn,Wn) float grayscale
             const int render_model_n,
             const float fx_n, const float fy_n,
             const float cx_n, const float cy_n,
             const int patch_radius,
             const bool debug);

// Phase 2 analytic backward -- see cuda_multiview_ncc/multiview_ncc_impl.cu
// (multiview_ncc_backward_kernel) for the full chain-rule derivation.
// Recomputes the forward pass internally (same inputs as MultiviewNCC above,
// plus the upstream dL/dncc); returns (grad_depths (P,), grad_normals (P,3)).
std::tuple<torch::Tensor, torch::Tensor>
MultiviewNCCBackward(const torch::Tensor& depths,
                      const torch::Tensor& normals,
                      const torch::Tensor& uvs,
                      const torch::Tensor& ray_dirs_r,
                      const torch::Tensor& R,
                      const torch::Tensor& T,
                      const torch::Tensor& image_r,
                      const torch::Tensor& image_n,
                      const int render_model_n,
                      const float fx_n, const float fy_n,
                      const float cx_n, const float cy_n,
                      const int patch_radius,
                      const torch::Tensor& grad_ncc,
                      const bool precise,
                      const bool debug);

// refer to Gaussian Splatting / warp_patch_ncc.h
#define CHECK_CUDA(A, debug)                                                                                           \
    A;                                                                                                                 \
    if (debug) {                                                                                                       \
        auto ret = cudaDeviceSynchronize();                                                                            \
        if (ret != cudaSuccess) {                                                                                      \
            std::cerr << "\n[CUDA ERROR] in " << __FILE__ << "\nLine " << __LINE__ << ": " << cudaGetErrorString(ret); \
            throw std::runtime_error(cudaGetErrorString(ret));                                                         \
        }                                                                                                              \
    }
