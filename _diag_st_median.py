"""Diagnostic 2: fidelity of the straight-through median-depth surrogate.
Gather per-pixel ||view-space center|| of the T=0.5-crossing gaussian (gidx)
and compare against the rasterizer's median depth on sessD_full30k @30000."""
import torch
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams
from scene import Scene, GaussianModel
from gaussian_renderer import render

parser = ArgumentParser()
lp = ModelParams(parser); pp = PipelineParams(parser)
args = parser.parse_args([
    "-m", "/home/output/sessD_full30k", "-s", "/home/data/tt/datasets/truck",
    "--dataset", "COLMAP", "--camera_model", "PINHOLE", "--resolution", "1", "--eval",
])
dataset = lp.extract(args); pipe = pp.extract(args)
dataset.render_model = "PH"
dataset.fov_mod = None; dataset.sample_step = 0.002
dataset.focal_scaling = 1.0; dataset.distortion_scaling = 1.0
dataset.mirror_shift = 0.0; dataset.raymap = None

g = GaussianModel(dataset.sh_degree)
scene = Scene(dataset, g, load_iteration=30000, shuffle=False)
bg = torch.zeros(3, device="cuda")
for cam_idx in (0, 60, 120):
    cam = scene.getTrainCameras()[cam_idx]
    with torch.no_grad():
        pkg = render(cam, g, pipe, bg)
    md = pkg["median_depth"][0]
    gidx = pkg["gidx"]
    W2V = cam.world_view_transform.cuda()
    xyz_view = g.get_xyz @ W2V[:3, :3] + W2V[3, :3]
    d_center = xyz_view.norm(dim=-1)
    sel = gidx >= 0
    surr = d_center[gidx[sel].long()]
    m = md[sel]
    rel = (surr - m).abs() / m.clamp_min(1e-6)
    print(f"cam {cam_idx}: gidx valid {sel.float().mean():.3f}, "
          f"|d_center[gidx]-md|/md: p50={rel.quantile(0.5):.4f} p90={rel.quantile(0.9):.4f} "
          f"p99={rel.quantile(0.99):.4f} max={rel.max():.3f}")
