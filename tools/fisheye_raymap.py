"""Per-pixel ray map for an OPENCV_FISHEYE / Kannala-Brandt camera.

Same output as data/scnt/scnt_raymap.py (float32 [H, W, 3] unit camera rays
(sin(theta) * n_hat, cos(theta)) on the integer pixel grid, read from
<scene>/nerfstudio/transforms.json, written to <scene>/raymap_fisheye.npy),
but inverts theta_d = theta (1 + k1 theta^2 + k2 theta^4 + k3 theta^6 + k4 theta^8)
by bisection instead of 3 Newton steps from a linear guess.

Why: scnt_raymap.py's initial guess overshoots by ~40% on a square fisheye
image (it scales by the corner radius), which is harmless for k = 0 (one Newton
step is exact) but diverges near the image-circle edge for real coefficients
(NaNs and rays at 130 deg where the model says ~81 deg, on the X5 lens).
Bisection over the monotonic part of the polynomial cannot overshoot. For
k = 0 this reproduces scnt_raymap.py's output inside the monotonic range.

usage: python tools/fisheye_raymap.py <scene_dir>
"""
import json
import os
import sys

import numpy as np


def main(scene_dir):
    info = json.load(open(os.path.join(scene_dir, "nerfstudio", "transforms.json")))
    W, H = int(info["w"]), int(info["h"])
    fx, fy, cx, cy = info["fl_x"], info["fl_y"], info["cx"], info["cy"]
    k = np.array([info.get(f"k{i}", 0.0) for i in range(1, 5)], dtype=np.float64)

    def theta_d(t):
        t2 = t * t
        return t * (1 + t2 * (k[0] + t2 * (k[1] + t2 * (k[2] + t2 * k[3]))))

    def dtheta_d(t):
        t2 = t * t
        return 1 + t2 * (3 * k[0] + t2 * (5 * k[1] + t2 * (7 * k[2] + t2 * 9 * k[3])))

    # Upper end of the monotonic (invertible) range, capped at pi.
    ts = np.linspace(0, np.pi, 20001)
    bad = np.nonzero(dtheta_d(ts) <= 0)[0]
    t_max = ts[bad[0] - 1] if len(bad) else np.pi
    d_max = theta_d(t_max)

    v, u = np.mgrid[0:H, 0:W].astype(np.float64)
    x, y = (u - cx) / fx, (v - cy) / fy
    delta = np.hypot(x, y)

    lo = np.zeros_like(delta)
    hi = np.full_like(delta, t_max)
    target = np.minimum(delta, d_max)  # beyond the invertible range: clamp to its edge
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        above = theta_d(mid) > target
        hi = np.where(above, mid, hi)
        lo = np.where(above, lo, mid)
    theta = 0.5 * (lo + hi)

    s = np.where(delta > 1e-6, np.sin(theta) / np.maximum(delta, 1e-12), 0.0)
    rays = np.stack([x * s, y * s, np.cos(theta)], axis=-1)
    rays[delta <= 1e-6] = (0.0, 0.0, 1.0)

    out = os.path.join(scene_dir, "raymap_fisheye.npy")
    np.save(out, rays.astype(np.float32))
    print(f"k={k.tolist()}  invertible to {np.degrees(t_max):.1f} deg (radius {d_max * fx:.0f} px)")
    print(f"saved {out}  {rays.shape}")


if __name__ == "__main__":
    main(sys.argv[1])
