# throwaway: audit per-camera dim consistency (gray_image vs ray_dirs vs H/W) at r2
import torch
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams
from scene import Scene, GaussianModel
from utils.ray_normals import get_ray_dirs_view

parser = ArgumentParser()
lp = ModelParams(parser); op = OptimizationParams(parser); pp = PipelineParams(parser)
args = parser.parse_args(["-s","/home/data/tt/datasets/truck","-m","/home/output/_dims_audit",
                          "--dataset","COLMAP","--camera_model","PINHOLE","--resolution","2","--eval"])
dataset = lp.extract(args); opt = op.extract(args); pipe = pp.extract(args)
dataset.render_model="PH"; dataset.fov_mod=None; dataset.sample_step=0.002
dataset.focal_scaling=1.0; dataset.distortion_scaling=1.0; dataset.mirror_shift=0.0; dataset.raymap=None
g = GaussianModel(dataset.sh_degree)
scene = Scene(dataset, g, shuffle=False)
cams = scene.getTrainCameras()
sizes = {}
bad = 0
for c in cams:
    gi = c.gray_image
    rd = get_ray_dirs_view(c)
    key = (c.image_height, c.image_width, tuple(gi.shape), tuple(rd.shape))
    sizes[key] = sizes.get(key, 0) + 1
    # ray_dirs: (3,H,W); gray: (1,H,W) or (H,W)
    gH, gW = gi.shape[-2], gi.shape[-1]
    rH, rW = rd.shape[-2], rd.shape[-1]
    if not (gH == rH == c.image_height and gW == rW == c.image_width):
        bad += 1
        print("MISMATCH:", c.image_name, "cam H/W", c.image_height, c.image_width,
              "gray", tuple(gi.shape), "rays", tuple(rd.shape))
print("distinct size signatures:", len(sizes))
for k, v in sizes.items():
    print("  ", k, "x", v)
print("mismatched cameras:", bad, "/", len(cams))
