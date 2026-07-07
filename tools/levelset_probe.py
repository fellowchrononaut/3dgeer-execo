#!/usr/bin/env python3
"""Measure the integrated-occupancy isosurface roughness over a locally-flat
patch (ground plane), WITHOUT meshing — the decisive splat-geometry-quality
probe from the 2026-07-06 mesh-noise investigation (13.3mm RMS on
sessD_full30k established that extraction was innocent and training-side
multiview regularization was the gap).

Method: take a block of pixels on the ground plane in one training view,
bisect the field along each pixel ray around the rendered median depth,
fit a plane to the recovered isosurface points, report residual RMS/p95.
Compare runs with the SAME --patch args; RMS (mm) is resolution-independent.

Usage:
  python tools/levelset_probe.py -m <model_dir> -s <source> \
      --dataset COLMAP --camera_model PINHOLE --render_model PH \
      [--resolution 2 --eval] [--iteration 30000] [--cy 0.85 --cx 0.5 --S 40]
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams
from scene import Scene, GaussianModel
from gaussian_renderer import render
from gaussian_wrapping.fields import evaluate_occupancy_integrated
from utils.ray_normals import get_ray_dirs_view


def main():
    parser = ArgumentParser()
    lp = ModelParams(parser); op = OptimizationParams(parser); pp = PipelineParams(parser)
    parser.add_argument("--iteration", type=int, default=30000)
    parser.add_argument("--render_model", type=str, default="PH")
    parser.add_argument("--cam_idx", type=int, default=0)
    parser.add_argument("--cy", type=float, default=0.85, help="patch center y (fraction of H)")
    parser.add_argument("--cx", type=float, default=0.5, help="patch center x (fraction of W)")
    parser.add_argument("--S", type=int, default=40, help="patch side length in pixels")
    parser.add_argument("--band", type=float, default=0.10, help="bisection band as fraction of mean depth")
    args = parser.parse_args()
    dataset = lp.extract(args); opt = op.extract(args); pipe = pp.extract(args)
    dataset.render_model = args.render_model
    dataset.fov_mod = None; dataset.sample_step = 0.002
    dataset.focal_scaling = 1.0; dataset.distortion_scaling = 1.0
    dataset.mirror_shift = 0.0; dataset.raymap = None

    g = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, g, load_iteration=args.iteration, shuffle=False)
    cams = scene.getTrainCameras()
    cam = cams[args.cam_idx]
    bg = torch.zeros(3, device="cuda")
    with torch.no_grad():
        pkg = render(cam, g, pipe, bg)
    md = pkg["median_depth"][0]
    H, W = md.shape
    cy, cx, S = int(H * args.cy), int(W * args.cx), args.S
    patch_d = md[cy:cy + S, cx:cx + S]
    if not (patch_d > 0).all():
        raise SystemExit("patch has depth holes; adjust --cy/--cx/--S")
    dirs = get_ray_dirs_view(cam)[:, cy:cy + S, cx:cx + S].reshape(3, -1).T
    d0 = patch_d.reshape(-1)
    R = cam.world_view_transform[:3, :3]
    C = cam.camera_center

    def to_world(t):
        return (dirs * t[:, None]) @ R.T + C

    band = args.band * float(d0.mean())
    lo, hi = d0 - band, d0 + band
    f_lo = evaluate_occupancy_integrated(to_world(lo), cams, g, pipe)
    f_hi = evaluate_occupancy_integrated(to_world(hi), cams, g, pipe)
    ok = (f_lo > 0) & (f_hi < 0)
    print(f"bracketing rays: {ok.float().mean():.3f} of {S}x{S} patch")
    lo, hi = lo.clone(), hi.clone()
    for _ in range(14):
        mid = 0.5 * (lo + hi)
        f = evaluate_occupancy_integrated(to_world(mid), cams, g, pipe)
        go_lo = f > 0
        lo = torch.where(go_lo & ok, mid, lo)
        hi = torch.where((~go_lo) & ok, mid, hi)
    pts = to_world(0.5 * (lo + hi))[ok].cpu().numpy()
    c = pts.mean(0); q = pts - c
    _, _, V = np.linalg.svd(q, full_matrices=False)
    res = q @ V[2]
    print(f"patch extent (m): {q.max(0) - q.min(0)}, npts={len(pts)}")
    print(f"LEVEL-SET roughness: RMS={res.std() * 1000:.2f} mm, "
          f"p95|res|={np.percentile(np.abs(res), 95) * 1000:.2f} mm")
    pts_md = to_world(d0)[ok].cpu().numpy()
    c2 = pts_md.mean(0); q2 = pts_md - c2
    _, _, V2 = np.linalg.svd(q2, full_matrices=False)
    res2 = q2 @ V2[2]
    print(f"MEDIAN-DEPTH roughness: RMS={res2.std() * 1000:.2f} mm, "
          f"p95={np.percentile(np.abs(res2), 95) * 1000:.2f} mm")


if __name__ == "__main__":
    main()
