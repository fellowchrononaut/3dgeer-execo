// New kernel (NOT a port of warp_patch_ncc_impl.cu's plane-induced-homography
// warp -- that construction only exists because a homography is a pinhole-only
// concept; see GUTWrap_Discussion/3DGEERGW_EXECUTION.md Session F2 "CORRECTION"
// entry). This kernel instead does the ray-intersect-reproject math explicitly
// per output pixel, so it generalizes to the EQ (fisheye) projection model.
//
// Sign/shape conventions mirror this project's own Python primitives so the
// pure-PyTorch reference oracle in utils/multiview.py can be numerically
// compared against this kernel directly:
//   - ray directions / depth convention: utils/ray_normals.py
//     (dist_map is Euclidean distance along the ray; pts = dirs * dist)
//   - PH/EQ pixel projection formulas: gaussian_wrapping/fisheye_proj.py
//     ::project_view_to_pixel (ported to CUDA math intrinsics here)
#include "multiview_ncc_impl.h"
#include <cmath>

#define EQ_MAX_THETA 1.5533430342749535f  // 89 degrees in radians

struct Float3 {
    float x, y, z;
};

__device__ __forceinline__ Float3 make_f3(float x, float y, float z) { return Float3{x, y, z}; }

__device__ __forceinline__ float dot3(const Float3& a, const Float3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ __forceinline__ Float3 scale3(const Float3& a, float s) {
    return make_f3(a.x * s, a.y * s, a.z * s);
}

__device__ __forceinline__ Float3 add3(const Float3& a, const Float3& b) {
    return make_f3(a.x + b.x, a.y + b.y, a.z + b.z);
}

// R is row-major 3x3: Xn = R @ X (row 0 -> Xn.x, row 1 -> Xn.y, row 2 -> Xn.z)
__device__ __forceinline__ Float3 matvec3(const float* R, const Float3& v) {
    return make_f3(
        R[0] * v.x + R[1] * v.y + R[2] * v.z,
        R[3] * v.x + R[4] * v.y + R[5] * v.z,
        R[6] * v.x + R[7] * v.y + R[8] * v.z);
}

// Bilinear sample with the pixel-CENTER convention used throughout this
// project (pixel i is centered at continuous coordinate i, see
// fisheye_proj.py's module docstring / project_view_to_pixel's "-0.5" terms).
__device__ __forceinline__ float bilinear_sample(const float* img, int H, int W, float u, float v) {
    int x0 = static_cast<int>(floorf(u));
    int y0 = static_cast<int>(floorf(v));
    int x1 = x0 + 1;
    int y1 = y0 + 1;
    float wx1 = u - static_cast<float>(x0);
    float wx0 = 1.f - wx1;
    float wy1 = v - static_cast<float>(y0);
    float wy0 = 1.f - wy1;
    x0 = min(max(x0, 0), W - 1);
    x1 = min(max(x1, 0), W - 1);
    y0 = min(max(y0, 0), H - 1);
    y1 = min(max(y1, 0), H - 1);
    float c00 = img[y0 * W + x0];
    float c01 = img[y0 * W + x1];
    float c10 = img[y1 * W + x0];
    float c11 = img[y1 * W + x1];
    return (c00 * wx0 + c01 * wx1) * wy0 + (c10 * wx0 + c11 * wx1) * wy1;
}

// Project a neighbor-view-space point to the neighbor camera's pixel coords.
// Mirrors gaussian_wrapping/fisheye_proj.py::project_view_to_pixel exactly
// (PH branch and undistorted-EQ branch), ported to CUDA math intrinsics.
__device__ __forceinline__ bool project_to_neighbor_pixel(
    const Float3& x, const int render_model_n,
    const float fx_n, const float fy_n, const float cx_n, const float cy_n,
    const int Wn, const int Hn,
    float& u_out, float& v_out) {
    if (render_model_n == 2) {  // PH (pinhole)
        bool valid_z  = x.z > 1e-6f;
        float safe_z  = valid_z ? x.z : 1.f;
        u_out         = fx_n * (x.x / safe_z) + 0.5f * Wn - 0.5f;
        v_out         = fy_n * (x.y / safe_z) + 0.5f * Hn - 0.5f;
        return valid_z
               && u_out >= -0.5f && u_out <= Wn - 0.5f
               && v_out >= -0.5f && v_out <= Hn - 0.5f;
    } else {  // KB/EQ (equidistant fisheye, undistorted)
        float r     = sqrtf(x.x * x.x + x.y * x.y);
        float theta = atan2f(r, x.z);
        float phi   = atan2f(x.y, x.x);
        u_out       = cx_n + fx_n * theta * cosf(phi) - 0.5f;
        v_out       = cy_n + fy_n * theta * sinf(phi) - 0.5f;
        return theta < EQ_MAX_THETA
               && u_out >= -0.5f && u_out <= Wn - 0.5f
               && v_out >= -0.5f && v_out <= Hn - 0.5f;
    }
}

__global__ void multiview_ncc_forward_kernel(
    const int P,
    const float* __restrict__ depths,
    const Float3* __restrict__ normals,
    const int2* __restrict__ uvs,
    const Float3* __restrict__ ray_dirs_r,  // flattened (Hr*Wr), row-major y*Wr+x
    const float* __restrict__ R,
    const float* __restrict__ T,
    const float* __restrict__ image_r,
    const float* __restrict__ image_n,
    const int render_model_n,
    const float fx_n, const float fy_n, const float cx_n, const float cy_n,
    const int Hr, const int Wr,
    const int Hn, const int Wn,
    const int radius,
    float* __restrict__ ncc,
    bool* __restrict__ valid) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= P) return;

    ncc[idx]   = 0.f;
    valid[idx] = false;

    const int2 uvc = uvs[idx];
    // Reference-image bbox check (mirrors warp_patch_ncc_impl.cu's early return).
    if (uvc.x - radius < 0 || uvc.x + radius > Wr - 1 ||
        uvc.y - radius < 0 || uvc.y + radius > Hr - 1) {
        return;
    }

    const float depth   = depths[idx];
    const Float3 normal = normals[idx];
    const Float3 ray_c  = ray_dirs_r[uvc.y * Wr + uvc.x];
    const Float3 P0     = scale3(ray_c, depth);
    const float N_dot_P0 = dot3(normal, P0);

    const Float3 Tv = make_f3(T[0], T[1], T[2]);

    float sum_c_r = 0.f, sum_c_n = 0.f, sum_c_r2 = 0.f, sum_c_n2 = 0.f, sum_c_rn = 0.f;
    bool all_inside = true;

    for (int dv = -radius; dv <= radius; dv++) {
        const int py = uvc.y + dv;
        for (int du = -radius; du <= radius; du++) {
            const int px = uvc.x + du;
            const Float3 ray_off = ray_dirs_r[py * Wr + px];
            const float denom    = dot3(normal, ray_off);

            bool bad          = fabsf(denom) < 1e-8f;
            const float sdenom = bad ? copysignf(1e-8f, denom == 0.f ? 1.f : denom) : denom;
            const float t      = N_dot_P0 / sdenom;
            bad                = bad || (t <= 0.f);

            const Float3 X  = scale3(ray_off, t);
            const Float3 Xn = add3(matvec3(R, X), Tv);

            float u_n, v_n;
            const bool proj_valid = project_to_neighbor_pixel(
                Xn, render_model_n, fx_n, fy_n, cx_n, cy_n, Wn, Hn, u_n, v_n);
            all_inside = all_inside && proj_valid && !bad;

            const float c_n = bilinear_sample(image_n, Hn, Wn, u_n, v_n);
            const float c_r = image_r[py * Wr + px];

            sum_c_r += c_r;
            sum_c_n += c_n;
            sum_c_r2 += c_r * c_r;
            sum_c_n2 += c_n * c_n;
            sum_c_rn += c_r * c_n;
        }
    }

    const int line          = 2 * radius + 1;
    const float total_inv   = 1.f / static_cast<float>(line * line);
    const float cross       = sum_c_rn - sum_c_r * sum_c_n * total_inv;
    const float variance_r  = sum_c_r2 - sum_c_r * sum_c_r * total_inv;
    const float variance_n  = sum_c_n2 - sum_c_n * sum_c_n * total_inv;
    const float output_ncc  = cross * cross / (variance_r * variance_n + 1e-8f);

    const bool valid_patch = all_inside && variance_r > 5e-6f && variance_n > 5e-6f;
    ncc[idx]   = valid_patch ? output_ncc : 0.f;
    valid[idx] = valid_patch;
}

void multiview_ncc_forward_launcher(
    const int P,
    const float* depths,
    const float* normals,
    const int* uvs,
    const float* ray_dirs_r,
    const float* R,
    const float* T,
    const float* image_r,
    const float* image_n,
    const int render_model_n,
    const float fx_n, const float fy_n,
    const float cx_n, const float cy_n,
    const int Hr, const int Wr,
    const int Hn, const int Wn,
    const int patch_radius,
    float* ncc,
    bool* valid) {
    if (P == 0) return;
    const int threads = 256;
    const int blocks   = (P + threads - 1) / threads;
    multiview_ncc_forward_kernel<<<blocks, threads>>>(
        P,
        depths,
        reinterpret_cast<const Float3*>(normals),
        reinterpret_cast<const int2*>(uvs),
        reinterpret_cast<const Float3*>(ray_dirs_r),
        R,
        T,
        image_r,
        image_n,
        render_model_n,
        fx_n, fy_n, cx_n, cy_n,
        Hr, Wr, Hn, Wn,
        patch_radius,
        ncc,
        valid);
}
