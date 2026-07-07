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
    // Reference-image bbox check (mirrors warp_patch_ncc_impl.cu's early
    // return). Overflow-proof form: all arithmetic is on the trusted image
    // dims, never on the user-supplied uvc values (uvc.x + radius could wrap
    // for garbage int inputs and slip past a `> Wr - 1` comparison -- found
    // in the 2026-07-07 OOB audit as the only theoretical OOB path; every
    // image read is otherwise bbox-gated or index-clamped before access).
    if (uvc.x < radius || uvc.x > Wr - 1 - radius ||
        uvc.y < radius || uvc.y > Hr - 1 - radius) {
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

// ===========================================================================
// Phase 2 -- analytic backward.
//
// Chain (per query point idx, per patch-offset pixel k = (du,dv)):
//
//   NCC = cross^2 / (var_r*var_n + eps)                     [patch statistics]
//     cross  = sum(c_r*c_n) - sum(c_r)*sum(c_n)/M
//     var_n  = sum(c_n^2)   - sum(c_n)^2/M         (var_r and c_r are
//                                                    grad-free: they never
//                                                    depend on depth/normal)
//   => d(NCC)/d(c_n_k) = A*(c_r_k - Sr/M) - B*(c_n_k - Sn/M)
//        A = 2*cross/den,  B = 2*cross^2*var_r/den^2,  den = var_r*var_n+eps
//
//   c_n_k = bilinear_sample(image_n, u_k, v_k)               [bilinear]
//     dc_n_k/du_k, dc_n_k/dv_k = standard corner-difference form
//
//   (u_k,v_k) = project(Xn_k)                                 [PH/EQ Jacobian]
//     Xn_k = R @ (t_k * ray_off_k) + T  =>  dXn_k/dt_k = R @ ray_off_k
//     (exact -- t is the only free scalar), so
//     du_k/dt_k = J_proj row u . (R @ ray_off_k), same for v.
//
//   t_k = depth*a / b_k                                       [ray-plane]
//     a   = dot(normal, ray_c)      (center ray, shared by all offsets)
//     b_k = dot(normal, ray_off_k)  (the forward kernel's `denom`)
//     dt_k/d(depth)  = a / b_k
//     dt_k/d(normal) = depth*(b_k*ray_c - a*ray_off_k) / b_k^2  (quotient rule)
//
// dL/d(depth) = grad_ncc * sum_k d(NCC)/d(c_n_k) * dc_n_k/dt_k * dt_k/d(depth)
// (same shape for normal, with the vector dt_k/d(normal)).
//
// The backward recomputes the whole forward reduction per point (Pass A)
// instead of stashing forward intermediates across the autograd boundary --
// simpler plumbing, negligible cost. Within the kernel, Pass A stashes the
// transcendental-heavy projection results per offset so Pass B (the chain
// rule) never recomputes them.
//
// PRECISION: the kernel is templated on the internal scalar type T.
//   T = double ("precise", the default): computes the exact derivative of
//     the shared real-valued function to f64 accuracy -- matches a float64
//     run of utils/multiview.py::ncc_reference_oracle's autograd to
//     machine precision (0 rel err after f32 rounding, see
//     tests/test_multiview.py). fp64 runs at 1/64 rate on consumer
//     GPUs, so this lands at roughly FD-backward speed, not faster.
//   T = float ("fast"): ~identical math in f32 + fast-math. Carries the
//     same noise floor as ANY f32 implementation of this gradient
//     (including the f32 reference oracle itself, verified empirically
//     against a float64 oracle): median rel err ~1e-3, with a tail at
//     small-|grad| points from (a) cancellation in the dNCC/dc_n
//     accumulation and (b) ~1e-3 px fast-math u_n noise flipping
//     bilinear-cell derivative branches. Still far more accurate than the
//     eps=3e-2 finite-difference backward, and much faster than it.
//
// Points whose recomputed valid_patch is false get exactly zero gradient,
// mirroring the forward's `ncc[idx] = valid_patch ? output_ncc : 0.f` gate
// -- the loss is genuinely disconnected there (patch bbox / EQ theta-cone /
// degenerate ray-plane / low-variance boundaries), not something to smooth.
// ===========================================================================

// Compile-time cap on the patch radius for the per-thread stash arrays
// (production multiview_patch_size = 3; enforced host-side in
// multiview_ncc.cu). Local arrays live in per-thread local memory -- at the
// production radius only (2*3+1)^2 = 49 of the 121 slots are touched.
#define BWD_MAX_RADIUS 5
#define BWD_MAX_OFFS ((2 * BWD_MAX_RADIUS + 1) * (2 * BWD_MAX_RADIUS + 1))

template <typename T>
struct Vec3 {
    T x, y, z;
};

template <typename T>
__device__ __forceinline__ Vec3<T> make_v3(T x, T y, T z) { return Vec3<T>{x, y, z}; }

template <typename T>
__device__ __forceinline__ Vec3<T> v3_of(const Float3& a) { return Vec3<T>{(T)a.x, (T)a.y, (T)a.z}; }

template <typename T>
__device__ __forceinline__ T dot3t(const Vec3<T>& a, const Vec3<T>& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

template <typename T>
__device__ __forceinline__ Vec3<T> scale3t(const Vec3<T>& a, T s) {
    return make_v3<T>(a.x * s, a.y * s, a.z * s);
}

template <typename T>
__device__ __forceinline__ Vec3<T> add3t(const Vec3<T>& a, const Vec3<T>& b) {
    return make_v3<T>(a.x + b.x, a.y + b.y, a.z + b.z);
}

template <typename T>
__device__ __forceinline__ Vec3<T> matvec3t(const float* R, const Vec3<T>& v) {
    return make_v3<T>(
        (T)R[0] * v.x + (T)R[1] * v.y + (T)R[2] * v.z,
        (T)R[3] * v.x + (T)R[4] * v.y + (T)R[5] * v.z,
        (T)R[6] * v.x + (T)R[7] * v.y + (T)R[8] * v.z);
}

// EQ projection via the exact identity cos(phi) = x/r, sin(phi) = y/r
// (r > 0), avoiding the atan2(y,x)+cos+sin of the forward kernel entirely:
//   u = cx + fx*theta*(x/r) - 0.5,  v = cy + fy*theta*(y/r) - 0.5.
// Identical real-valued function as project_to_neighbor_pixel (the r->0
// limit of theta*(x/r) is 0 either way, since theta ~ r/z); also hands back
// r and theta for the Jacobian in Pass B.
template <typename T>
__device__ __forceinline__ bool project_eq_identity(
    const Vec3<T>& x,
    const T fx_n, const T fy_n, const T cx_n, const T cy_n,
    const int Wn, const int Hn,
    T& u_out, T& v_out, T& r_out, T& theta_out) {
    const T r     = sqrt(x.x * x.x + x.y * x.y);
    const T theta = atan2(r, x.z);
    const T inv_r = T(1) / max(r, T(1e-12));
    u_out     = cx_n + fx_n * theta * (x.x * inv_r) - T(0.5);
    v_out     = cy_n + fy_n * theta * (x.y * inv_r) - T(0.5);
    r_out     = r;
    theta_out = theta;
    return theta < (T)EQ_MAX_THETA
           && u_out >= T(-0.5) && u_out <= T(Wn) - T(0.5)
           && v_out >= T(-0.5) && v_out <= T(Hn) - T(0.5);
}

template <typename T>
__device__ __forceinline__ T bilinear_sample_t(const float* img, int H, int W, T u, T v) {
    int x0 = static_cast<int>(floor(u));
    int y0 = static_cast<int>(floor(v));
    int x1 = x0 + 1;
    int y1 = y0 + 1;
    T wx1 = u - (T)x0;
    T wx0 = T(1) - wx1;
    T wy1 = v - (T)y0;
    T wy0 = T(1) - wy1;
    x0 = min(max(x0, 0), W - 1);
    x1 = min(max(x1, 0), W - 1);
    y0 = min(max(y0, 0), H - 1);
    y1 = min(max(y1, 0), H - 1);
    T c00 = img[y0 * W + x0];
    T c01 = img[y0 * W + x1];
    T c10 = img[y1 * W + x0];
    T c11 = img[y1 * W + x1];
    return (c00 * wx0 + c01 * wx1) * wy0 + (c10 * wx0 + c11 * wx1) * wy1;
}

template <typename T>
__device__ __forceinline__ void bilinear_sample_grad_t(
    const float* img, int H, int W, T u, T v, T& dc_du, T& dc_dv) {
    int x0 = static_cast<int>(floor(u));
    int y0 = static_cast<int>(floor(v));
    int x1 = x0 + 1;
    int y1 = y0 + 1;
    T wx1 = u - (T)x0;
    T wy1 = v - (T)y0;
    T wx0 = T(1) - wx1;
    T wy0 = T(1) - wy1;
    x0 = min(max(x0, 0), W - 1);
    x1 = min(max(x1, 0), W - 1);
    y0 = min(max(y0, 0), H - 1);
    y1 = min(max(y1, 0), H - 1);
    T c00 = img[y0 * W + x0];
    T c01 = img[y0 * W + x1];
    T c10 = img[y1 * W + x0];
    T c11 = img[y1 * W + x1];
    dc_du = (c01 - c00) * wy0 + (c11 - c10) * wy1;
    dc_dv = (c10 - c00) * wx0 + (c11 - c01) * wx1;
}

template <typename T>
struct Jac2x3 {
    T du_dx, du_dy, du_dz;
    T dv_dx, dv_dy, dv_dz;
};

template <typename T>
__device__ __forceinline__ Jac2x3<T> ph_projection_jacobian(const Vec3<T>& x, T fx, T fy) {
    bool valid_z = x.z > T(1e-6);
    T safe_z     = valid_z ? x.z : T(1);
    T inv_z      = T(1) / safe_z;
    Jac2x3<T> J;
    J.du_dx = fx * inv_z;
    J.du_dy = T(0);
    J.du_dz = -fx * x.x * inv_z * inv_z;
    J.dv_dx = T(0);
    J.dv_dy = fy * inv_z;
    J.dv_dz = -fy * x.y * inv_z * inv_z;
    return J;
}

// EQ Jacobian. theta = atan2(r,z), r = sqrt(x^2+y^2), cphi = x/r, sphi = y/r;
// u = cx+fx*theta*cphi, v = cy+fy*theta*sphi. Symbolic form (R2 = r^2+z^2):
//   du/dx = fx*( z*x*cphi/(r*R2) + theta*y^2/r^3 )
//   du/dy = fx*( z*y*cphi/(r*R2) - theta*x*y/r^3 )
//   du/dz = -fx*r*cphi/R2
//   dv/*  = same with sphi<->cphi and the phi-term signs mirrored.
// (The theta*'/r^3 terms below are written as theta*sphi*k2*y etc. with
// k2 = 1/r^2 -- algebraically identical via sphi = y/r.)
// r and theta are passed in from the Pass-A stash (project_eq_identity).
// rho->0 guard: the true Jacobian is bounded-but-direction-dependent as
// r->0 (theta->0 at the same rate the 1/r-type terms grow); floor the
// r/rho^2 denominators at a small epsilon rather than dividing by zero.
template <typename T>
__device__ __forceinline__ Jac2x3<T> eq_projection_jacobian(
    const Vec3<T>& x, T fx, T fy, T r, T theta) {
    const T rho2      = r * r;
    const T r_safe    = max(r, T(1e-9));
    const T rho2_safe = max(rho2, T(1e-18));
    const T R2_safe   = max(rho2 + x.z * x.z, T(1e-18));
    const T inv_r     = T(1) / r_safe;
    const T inv_R2    = T(1) / R2_safe;
    const T cphi      = x.x * inv_r;
    const T sphi      = x.y * inv_r;
    const T k1        = x.z * inv_r * inv_R2;  // d(theta)/dr / r  (radial factor)
    const T k2        = T(1) / rho2_safe;       // used for d(phi)/d(x,y)

    Jac2x3<T> J;
    J.du_dx = fx * (k1 * x.x * cphi + theta * sphi * k2 * x.y);
    J.du_dy = fx * (k1 * x.y * cphi - theta * sphi * k2 * x.x);
    J.du_dz = fx * (-r * inv_R2) * cphi;
    J.dv_dx = fy * (k1 * x.x * sphi - theta * cphi * k2 * x.y);
    J.dv_dy = fy * (k1 * x.y * sphi + theta * cphi * k2 * x.x);
    J.dv_dz = fy * (-r * inv_R2) * sphi;
    return J;
}

template <typename T>
__global__ void multiview_ncc_backward_kernel(
    const int P,
    const float* __restrict__ depths,
    const Float3* __restrict__ normals,
    const int2* __restrict__ uvs,
    const Float3* __restrict__ ray_dirs_r,
    const float* __restrict__ R,
    const float* __restrict__ Tr,
    const float* __restrict__ image_r,
    const float* __restrict__ image_n,
    const int render_model_n,
    const float fx_nf, const float fy_nf, const float cx_nf, const float cy_nf,
    const int Hr, const int Wr,
    const int Hn, const int Wn,
    const int radius,
    const float* __restrict__ grad_ncc,
    float* __restrict__ grad_depths,
    Float3* __restrict__ grad_normals) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= P) return;

    grad_depths[idx]  = 0.f;
    grad_normals[idx] = make_f3(0.f, 0.f, 0.f);

    const int2 uvc = uvs[idx];
    // Overflow-proof bbox check -- same rationale as the forward kernel's.
    if (uvc.x < radius || uvc.x > Wr - 1 - radius ||
        uvc.y < radius || uvc.y > Hr - 1 - radius ||
        radius > BWD_MAX_RADIUS) {
        return;
    }

    const T fx_n = fx_nf, fy_n = fy_nf, cx_n = cx_nf, cy_n = cy_nf;

    const T depth          = depths[idx];
    const Vec3<T> normal   = v3_of<T>(normals[idx]);
    const Vec3<T> ray_c    = v3_of<T>(ray_dirs_r[uvc.y * Wr + uvc.x]);
    const T a              = dot3t(normal, ray_c);  // forward's dot(N,ray_c)
    const T N_dot_P0       = depth * a;              // == dot(N, depth*ray_c)

    const Vec3<T> Tv = make_v3<T>(Tr[0], Tr[1], Tr[2]);

    // Per-offset stash: projection results from Pass A, reused by Pass B so
    // the transcendental-heavy work is done exactly once per offset.
    T st_u[BWD_MAX_OFFS], st_v[BWD_MAX_OFFS], st_cn[BWD_MAX_OFFS];
    T st_r[BWD_MAX_OFFS], st_th[BWD_MAX_OFFS];   // EQ Jacobian inputs (unused for PH)

    // ---- Pass A: forward reduction + stash. ----
    T sum_c_r = T(0), sum_c_n = T(0), sum_c_r2 = T(0), sum_c_n2 = T(0), sum_c_rn = T(0);
    bool all_inside = true;
    int k = 0;

    for (int dv = -radius; dv <= radius; dv++) {
        const int py = uvc.y + dv;
        for (int du = -radius; du <= radius; du++, k++) {
            const int px = uvc.x + du;
            const Vec3<T> ray_off = v3_of<T>(ray_dirs_r[py * Wr + px]);
            const T denom         = dot3t(normal, ray_off);

            bool bad       = fabs(denom) < T(1e-8);
            const T sdenom = bad ? copysign(T(1e-8), denom == T(0) ? T(1) : denom) : denom;
            const T t      = N_dot_P0 / sdenom;
            bad            = bad || (t <= T(0));

            const Vec3<T> X  = scale3t(ray_off, t);
            const Vec3<T> Xn = add3t(matvec3t<T>(R, X), Tv);

            T u_n, v_n, r_eq = T(0), th_eq = T(0);
            bool proj_valid;
            if (render_model_n == 2) {  // PH
                bool valid_z = Xn.z > T(1e-6);
                T safe_z     = valid_z ? Xn.z : T(1);
                u_n = fx_n * (Xn.x / safe_z) + T(0.5) * T(Wn) - T(0.5);
                v_n = fy_n * (Xn.y / safe_z) + T(0.5) * T(Hn) - T(0.5);
                proj_valid = valid_z
                             && u_n >= T(-0.5) && u_n <= T(Wn) - T(0.5)
                             && v_n >= T(-0.5) && v_n <= T(Hn) - T(0.5);
            } else {  // KB/EQ
                proj_valid = project_eq_identity<T>(
                    Xn, fx_n, fy_n, cx_n, cy_n, Wn, Hn, u_n, v_n, r_eq, th_eq);
            }
            all_inside = all_inside && proj_valid && !bad;

            const T c_n = bilinear_sample_t<T>(image_n, Hn, Wn, u_n, v_n);
            const T c_r = image_r[py * Wr + px];

            st_u[k]  = u_n;
            st_v[k]  = v_n;
            st_cn[k] = c_n;
            st_r[k]  = r_eq;
            st_th[k] = th_eq;

            sum_c_r += c_r;
            sum_c_n += c_n;
            sum_c_r2 += c_r * c_r;
            sum_c_n2 += c_n * c_n;
            sum_c_rn += c_r * c_n;
        }
    }

    const int line      = 2 * radius + 1;
    const T total_inv   = T(1) / (T)(line * line);
    const T cross       = sum_c_rn - sum_c_r * sum_c_n * total_inv;
    const T variance_r  = sum_c_r2 - sum_c_r * sum_c_r * total_inv;
    const T variance_n  = sum_c_n2 - sum_c_n * sum_c_n * total_inv;
    const T den_ncc     = variance_r * variance_n + T(1e-8);

    const bool valid_patch = all_inside && variance_r > T(5e-6) && variance_n > T(5e-6);
    if (!valid_patch) {
        return;   // matches forward's ncc[idx]=0.f gate -- disconnected, zero grad.
    }

    const T g_upstream = grad_ncc[idx];
    // d(NCC)/d(cross) = 2*cross/den_ncc ; d(NCC)/d(var_n) = -cross^2*var_r/den_ncc^2
    const T A = T(2) * cross / den_ncc;
    const T B = T(2) * cross * cross * variance_r / (den_ncc * den_ncc);

    const T mean_c_r = sum_c_r * total_inv;
    const T mean_c_n = sum_c_n * total_inv;

    T acc_ddepth = T(0);
    Vec3<T> acc_dnormal = make_v3<T>(T(0), T(0), T(0));

    // ---- Pass B: per-offset analytic chain over the stash. ----
    k = 0;
    for (int dv = -radius; dv <= radius; dv++) {
        const int py = uvc.y + dv;
        for (int du = -radius; du <= radius; du++, k++) {
            const int px = uvc.x + du;
            const Vec3<T> ray_off = v3_of<T>(ray_dirs_r[py * Wr + px]);
            const T b             = dot3t(normal, ray_off);   // == denom

            bool bad       = fabs(b) < T(1e-8);
            const T sdenom = bad ? copysign(T(1e-8), b == T(0) ? T(1) : b) : b;
            const T inv_b  = T(1) / sdenom;
            const T t      = N_dot_P0 * inv_b;

            const Vec3<T> ray_off_n = matvec3t<T>(R, ray_off);   // == dXn/dt
            const Vec3<T> Xn        = add3t(scale3t(ray_off_n, t), Tv);

            const T u_n = st_u[k];
            const T v_n = st_v[k];
            const T c_n = st_cn[k];
            const T c_r = image_r[py * Wr + px];

            // d(NCC)/d(c_n_k)
            const T dncc_dcn = A * (c_r - mean_c_r) - B * (c_n - mean_c_n);

            // d(c_n_k)/d(u,v) -- bilinear corner-difference form.
            T dc_du, dc_dv;
            bilinear_sample_grad_t<T>(image_n, Hn, Wn, u_n, v_n, dc_du, dc_dv);

            // d(u,v)/d(Xn) -- PH or EQ projection Jacobian, then directional
            // derivative along ray_off_n (== dXn/dt) to get d(u,v)/dt.
            Jac2x3<T> J = (render_model_n == 2)
                              ? ph_projection_jacobian<T>(Xn, fx_n, fy_n)
                              : eq_projection_jacobian<T>(Xn, fx_n, fy_n, st_r[k], st_th[k]);
            const T du_dt = J.du_dx * ray_off_n.x + J.du_dy * ray_off_n.y + J.du_dz * ray_off_n.z;
            const T dv_dt = J.dv_dx * ray_off_n.x + J.dv_dy * ray_off_n.y + J.dv_dz * ray_off_n.z;

            const T dcn_dt = dc_du * du_dt + dc_dv * dv_dt;   // d(c_n_k)/dt

            // Ray-plane intersection: t = depth*a / b  (quotient rule).
            //   dt/d(depth)  = a/b
            //   dt/d(normal) = depth*(b*ray_c - a*ray_off)/b^2
            const T dt_ddepth = a * inv_b;
            const Vec3<T> dt_dnormal = scale3t(
                add3t(scale3t(ray_c, sdenom), scale3t(ray_off, -a)),
                depth * inv_b * inv_b);

            acc_ddepth += dncc_dcn * dcn_dt * dt_ddepth;
            acc_dnormal = add3t(acc_dnormal, scale3t(dt_dnormal, dncc_dcn * dcn_dt));
        }
    }

    grad_depths[idx]  = static_cast<float>(g_upstream * acc_ddepth);
    grad_normals[idx] = make_f3(
        static_cast<float>(g_upstream * acc_dnormal.x),
        static_cast<float>(g_upstream * acc_dnormal.y),
        static_cast<float>(g_upstream * acc_dnormal.z));
}

void multiview_ncc_backward_launcher(
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
    const bool precise,
    const float* grad_ncc,
    float* grad_depths,
    float* grad_normals) {
    if (P == 0) return;
    const int threads = 256;
    const int blocks   = (P + threads - 1) / threads;
    if (precise) {
        multiview_ncc_backward_kernel<double><<<blocks, threads>>>(
            P, depths,
            reinterpret_cast<const Float3*>(normals),
            reinterpret_cast<const int2*>(uvs),
            reinterpret_cast<const Float3*>(ray_dirs_r),
            R, T, image_r, image_n,
            render_model_n, fx_n, fy_n, cx_n, cy_n,
            Hr, Wr, Hn, Wn, patch_radius, grad_ncc,
            grad_depths, reinterpret_cast<Float3*>(grad_normals));
    } else {
        multiview_ncc_backward_kernel<float><<<blocks, threads>>>(
            P, depths,
            reinterpret_cast<const Float3*>(normals),
            reinterpret_cast<const int2*>(uvs),
            reinterpret_cast<const Float3*>(ray_dirs_r),
            R, T, image_r, image_n,
            render_model_n, fx_n, fy_n, cx_n, cy_n,
            Hr, Wr, Hn, Wn, patch_radius, grad_ncc,
            grad_depths, reinterpret_cast<Float3*>(grad_normals));
    }
}
