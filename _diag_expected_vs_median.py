"""Diagnostic: how biased is 1/expected_invdepth vs median_depth on a healthy
checkpoint (sessD_full30k @30000)? Run inside geer from /home."""
import sys, torch
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
cam = scene.getTrainCameras()[0]
bg = torch.zeros(3, device="cuda")
with torch.no_grad():
    pkg = render(cam, g, pipe, bg)
    # accumulated alpha via colors_precomp=ones trick
    ones = torch.ones_like(g.get_xyz)
    pkg_a = render(cam, g, pipe, torch.zeros(3, device="cuda"),
                   override_color=ones)
    alpha = pkg_a["render"][0]

md = pkg["median_depth"][0]
inv = pkg["depth"].squeeze(0) if pkg["depth"].dim() == 3 else pkg["depth"]
exp_d = torch.where(inv > 1e-6, 1.0 / inv.clamp_min(1e-6), torch.zeros_like(inv))
exp_d_norm = torch.where(inv > 1e-6, alpha / inv.clamp_min(1e-6), torch.zeros_like(inv))

both = (md > 0) & (exp_d > 0)
rel = ((exp_d - md).abs() / md.clamp_min(1e-6))[both]
rel_n = ((exp_d_norm - md).abs() / md.clamp_min(1e-6))[both]
print(f"pixels: {md.numel()}, md>0: {(md>0).float().mean():.3f}, inv>1e-6: {(inv>1e-6).float().mean():.3f}")
print(f"alpha: p5={alpha.quantile(0.05):.3f} p50={alpha.quantile(0.5):.3f} p95={alpha.quantile(0.95):.3f}, frac alpha<0.9: {(alpha<0.9).float().mean():.3f}")
print(f"RAW  |1/inv - md|/md   : p50={rel.quantile(0.5):.4f} p90={rel.quantile(0.9):.4f} p99={rel.quantile(0.99):.4f} max={rel.max():.2f}")
print(f"NORM |alpha/inv - md|/md: p50={rel_n.quantile(0.5):.4f} p90={rel_n.quantile(0.9):.4f} p99={rel_n.quantile(0.99):.4f} max={rel_n.max():.2f}")
sat = both & (alpha > 0.99)
print(f"saturated (alpha>0.99, {sat.float().mean():.3f} of px): RAW p50={((exp_d-md).abs()/md.clamp_min(1e-6))[sat].quantile(0.5):.4f} p99={((exp_d-md).abs()/md.clamp_min(1e-6))[sat].quantile(0.99):.4f}")
unsat = both & (alpha < 0.9)
if unsat.any():
    print(f"unsaturated (alpha<0.9, {unsat.float().mean():.3f} of px): RAW p50={((exp_d-md).abs()/md.clamp_min(1e-6))[unsat].quantile(0.5):.4f} p99={((exp_d-md).abs()/md.clamp_min(1e-6))[unsat].quantile(0.99):.4f}")
