#!/usr/bin/env python3
"""Surface-quality metrics on flat image patches of one training view, for
comparing runs (e.g. --depth_normal_grad none/median/st on x5_kb).

Per patch (top-left = (cy*H, cx*W), side S, as in tools/levelset_probe.py):
  attr / shape / depth : mean angle (deg) of each pixel's normal from the
      patch-mean normal, for the learned normal attribute, the Gaussians' own
      shortest axis, and the normal of the rendered median-depth surface.
      attr/shape small but depth large = well-oriented Gaussians stacked at
      different depths ("shingled"), which is what makes meshes jagged.
  mesh_rms : plane-fit residual RMS of mesh vertices that project into the
      patch at the visible depth (only with --mesh).
Units follow the scene (x5 COLMAP/Metashape units, not guaranteed metric).

Usage (in geer, from /home):
  python tools/surface_patch_metrics.py -m <model_dir> -s <source> --iteration 28000 \
      --render_model KB --raymap_path <source>/raymap_fisheye.npy [--mesh <mesh.ply>] \
      [--patch desk:0.63,0.355,40 ...]
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gaussian_wrapping.extract_mesh import build_scene
from gaussian_wrapping.fields import world_to_view
from gaussian_wrapping.fisheye_proj import project_view_to_pixel
from gaussian_renderer import render, render_normal_field
from utils.ray_normals import depth_to_normals_via_rays, view_normals_to_world

# Flat patches on x5 camera 0 (frame_000001_0), confirmed by the user 2026-10-15.
DEFAULT_PATCHES = ["desk:0.63,0.355,40", "table:0.595,0.70,30", "wall:0.46,0.15,60", "whiteboard:0.45,0.62,40"]


def spread(n, y0, x0, S):
    p = n[:, y0:y0 + S, x0:x0 + S].reshape(3, -1)
    m = F.normalize(p.mean(1), dim=0)
    return torch.rad2deg(torch.acos((p * m[:, None]).sum(0).clamp(-1, 1))).mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model_path", required=True)
    ap.add_argument("-s", "--source_path", required=True)
    ap.add_argument("--iteration", type=int, required=True)
    ap.add_argument("--dataset", default="SCANNETPP")
    ap.add_argument("--camera_model", default="FISHEYE")
    ap.add_argument("--render_model", default="KB")
    ap.add_argument("--raymap_path", default=None)
    ap.add_argument("--cam_idx", type=int, default=0)
    ap.add_argument("--mesh", default=None)
    ap.add_argument("--band", type=float, default=0.10)
    ap.add_argument("--patch", nargs="*", default=DEFAULT_PATCHES, help="name:cy,cx,S")
    a = ap.parse_args()

    _, _, pipe, g, scene = build_scene(
        source_path=a.source_path, model_path=a.model_path, dataset_type=a.dataset,
        camera_model=a.camera_model, iteration=a.iteration, render_model=a.render_model,
        sample_step=0.002, focal_scaling=1.0, distortion_scaling=1.0, mirror_shift=0.0,
        raymap_path=a.raymap_path)
    cam = scene.getTrainCameras()[a.cam_idx]
    with torch.no_grad():
        n_attr = F.normalize(render_normal_field(cam, g, pipe, shape=False), dim=0)
        n_shape = F.normalize(render_normal_field(cam, g, pipe, shape=True), dim=0)
        md = render(cam, g, pipe, torch.zeros(3, device="cuda"))["median_depth"]
        nv, _ = depth_to_normals_via_rays(cam, md)
        n_depth = F.normalize(view_normals_to_world(cam, nv), dim=0)
    md = md[0]
    H, W = md.shape

    mesh_v = None
    if a.mesh:
        import trimesh
        V = torch.tensor(np.asarray(trimesh.load(a.mesh, process=False).vertices), dtype=torch.float32, device="cuda")
        xv = world_to_view(cam, V)
        u, v, valid = project_view_to_pixel(cam, xv)
        mesh_v = (V, u, v, valid, xv.norm(dim=1))

    print(f"model {a.model_path} @ {a.iteration}  ({g.get_xyz.shape[0]} Gaussians, cam {a.cam_idx})")
    f_px = float(getattr(cam, "focal_x", 0.0) or 0.0) or (W / 2.0) / np.tan(cam.FoVx / 2.0)
    print(f"{'patch':12s} {'attr':>6s} {'shape':>6s} {'depth':>6s}   {'mesh_rms':>9s} {'rms/dist':>9s} {'rms_px':>7s}  "
          f"(normal spreads in deg; mesh_rms in scene units x1000; rms/dist in permille; rms_px = rms at that distance in pixels, f={f_px:.0f}px)")
    for spec in a.patch:
        name, rest = spec.split(":")
        cy, cx, S = (float(t) for t in rest.split(","))
        y0, x0, S = int(H * cy), int(W * cx), int(S)
        row = f"{name:12s} {spread(n_attr, y0, x0, S):6.1f} {spread(n_shape, y0, x0, S):6.1f} {spread(n_depth, y0, x0, S):6.1f}"
        if mesh_v is not None:
            V, u, v, valid, dist = mesh_v
            pd = md[y0:y0 + S, x0:x0 + S]
            d0 = pd[pd > 0].median()
            sel = valid & (u >= x0) & (u < x0 + S) & (v >= y0) & (v < y0 + S) & ((dist - d0).abs() < a.band * d0)
            P = V[sel].double().cpu().numpy()
            if len(P) >= 10:
                c = P - P.mean(0)
                res = c @ np.linalg.svd(c, full_matrices=False)[2][-1]
                rms, dist = res.std(), float(d0)
                row += f"   {rms * 1000:9.2f} {rms / dist * 1000:9.2f} {rms / dist * f_px:7.2f}  (n={len(P)}, dist={dist:.2f})"
            else:
                row += f"   {'n/a':>9s}  (n={len(P)})"
        print(row)


if __name__ == "__main__":
    main()
