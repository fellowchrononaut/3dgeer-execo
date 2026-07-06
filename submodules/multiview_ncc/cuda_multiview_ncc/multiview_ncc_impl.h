#pragma once

// Raw-pointer forward-only launcher. New math (NOT a port of
// warp_patch_ncc_impl's homography warp) -- see GUTWrap_Discussion/
// 3DGEERGW_EXECUTION.md Session F2 and multiview_ncc.h for the design.
//
// Per query point idx (patch centered at ref-camera pixel uvs[idx]):
//   for each of the (2*patch_radius+1)^2 integer pixel offsets (du,dv)
//   around uvs[idx] in the REFERENCE image:
//     1. look up that offset pixel's view-space ray direction in ray_dirs_r
//     2. intersect the ray with the local tangent plane anchored at the
//        query point's own 3D location P0 = depth*ray_dirs_r[uvs[idx]],
//        plane normal = normals[idx] (ref view space, need not be unit)
//     3. transform the intersection point into the neighbor camera's view
//        space via X_n = R @ X + T
//     4. reproject X_n into the neighbor camera's pixel coords via its own
//        projection model (render_model_n: 1=EQ, 2=PH)
//     5. bilinear-sample image_n there; sample image_r at the exact integer
//        offset pixel (no interpolation needed -- offsets are integer ref
//        pixels, not the half-step optimization from warp_patch_ncc)
//   accumulate sum_c_r, sum_c_n, sum_c_r2, sum_c_n2, sum_c_r_c_n over the
//   patch and compute ncc = cross^2 / (var_r*var_n + eps), exactly as
//   warp_patch_ncc_impl.cu:231-234 (this reduction step IS projection-model
//   independent and is intentionally identical).
void multiview_ncc_forward_launcher(
    const int P,
    const float* depths,
    const float* normals,      // (P,3)
    const int* uvs,            // (P,2) int32 (x, y)
    const float* ray_dirs_r,   // (Hr,Wr,3)
    const float* R,            // (3,3) row-major, ref-view -> neighbor-view
    const float* T,            // (3,)
    const float* image_r,      // (Hr,Wr)
    const float* image_n,      // (Hn,Wn)
    const int render_model_n,  // 1 = KB/EQ, 2 = PH
    const float fx_n, const float fy_n,
    const float cx_n, const float cy_n,
    const int Hr, const int Wr,
    const int Hn, const int Wn,
    const int patch_radius,
    float* ncc,
    bool* valid);
