"""Session G V5 baseline: median-depth stats on sessD_full30k cam 0 with the
CURRENT (pre-Session-G) build. Compare after the forward interpolation change."""
import torch
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams
from scene import Scene, GaussianModel
from gaussian_renderer import render

parser = ArgumentParser(); lp = ModelParams(parser); pp = PipelineParams(parser)
args = parser.parse_args(["-m", "/home/output/sessD_full30k", "-s", "/home/data/tt/datasets/truck",
    "--dataset", "COLMAP", "--camera_model", "PINHOLE", "--resolution", "1", "--eval"])
d = lp.extract(args); pipe = pp.extract(args)
d.render_model = "PH"; d.fov_mod = None; d.sample_step = 0.002
d.focal_scaling = 1.0; d.distortion_scaling = 1.0; d.mirror_shift = 0.0; d.raymap = None
g = GaussianModel(d.sh_degree); s = Scene(d, g, load_iteration=30000, shuffle=False)
cam = s.getTrainCameras()[0]
with torch.no_grad():
    pkg = render(cam, g, pipe, torch.zeros(3, device="cuda"))
md = pkg["median_depth"][0]; sel = md > 0
print(f"MD-BASELINE cam0: valid {sel.float().mean():.4f}  p50 {md[sel].median():.4f}  "
      f"p10 {md[sel].quantile(0.1):.4f}  p90 {md[sel].quantile(0.9):.4f}")
