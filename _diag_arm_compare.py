"""3-arm comparison renders + test PSNR at a fixed iteration.
Usage (in geer, from /home):
    python _diag_arm_compare.py <model_dir> <iteration> <tag>
Renders train cams 0/60/120: RGB, shape-normal map, median-depth vis
-> /home/output/arm_compare/<tag>_cam{N}_{rgb,shape,depth}.png
and prints test-set PSNR at that iteration's point cloud.
"""
import os
import sys
import numpy as np
import cv2
import torch
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams
from scene import Scene, GaussianModel
from gaussian_renderer import render, render_normal_field
from utils.image_utils import psnr

model_dir, it, tag = sys.argv[1], int(sys.argv[2]), sys.argv[3]
parser = ArgumentParser(); lp = ModelParams(parser); pp = PipelineParams(parser)
args = parser.parse_args(["-m", model_dir, "-s", "/home/data/tt/datasets/truck",
    "--dataset", "COLMAP", "--camera_model", "PINHOLE", "--resolution", "1", "--eval"])
d = lp.extract(args); pipe = pp.extract(args)
d.render_model = "PH"; d.fov_mod = None; d.sample_step = 0.002
d.focal_scaling = 1.0; d.distortion_scaling = 1.0; d.mirror_shift = 0.0; d.raymap = None
g = GaussianModel(d.sh_degree)
scene = Scene(d, g, load_iteration=it, shuffle=False)
bg = torch.zeros(3, device="cuda")
out_dir = "/home/output/arm_compare"
os.makedirs(out_dir, exist_ok=True)

with torch.no_grad():
    for ci in (0, 60, 120):
        cam = scene.getTrainCameras()[ci]
        pkg = render(cam, g, pipe, bg)
        rgb = (pkg["render"].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        cv2.imwrite(f"{out_dir}/{tag}_cam{ci}_rgb.png", rgb[:, :, [2, 1, 0]])
        sm = render_normal_field(cam, g, pipe, shape=True)
        sv = ((sm * 0.5 + 0.5).clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        cv2.imwrite(f"{out_dir}/{tag}_cam{ci}_shape.png", sv[:, :, [2, 1, 0]])
        md = pkg["median_depth"][0]
        dv = (md / md.max().clamp(min=1e-8)).clamp(0, 1).cpu().numpy()
        cv2.imwrite(f"{out_dir}/{tag}_cam{ci}_depth.png", (dv * 255).astype(np.uint8))

    p_sum, n = 0.0, 0
    for cam in scene.getTestCameras():
        img = render(cam, g, pipe, bg)["render"].clamp(0, 1)
        gt = cam.sampled_image.cuda().clamp(0, 1)
        p_sum += psnr(img, gt).mean().item(); n += 1
print(f"ARM {tag} iter {it}: test PSNR {p_sum / n:.3f} over {n} views; "
      f"P={g.get_xyz.shape[0]}; renders in {out_dir}/")
