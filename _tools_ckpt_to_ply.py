"""Reconstruct point_cloud/iteration_N/point_cloud.ply from a chkpntN.pth
(for runs whose --save_iterations missed N). Usage (in geer, from /home):
    python _tools_ckpt_to_ply.py /home/output/sessF2_mv_r1_fix6 21000
"""
import os
import sys
import torch
from scene.gaussian_model import GaussianModel

model_dir, it = sys.argv[1], int(sys.argv[2])
ckpt = os.path.join(model_dir, f"chkpnt{it}.pth")
model_args, loaded_iter = torch.load(ckpt, weights_only=False)
assert loaded_iter == it, f"checkpoint says iter {loaded_iter}, expected {it}"

g = GaussianModel(3)
# capture() layout: [0] active_sh_degree, [1] _xyz, [2] _features_dc,
# [3] _features_rest, [4] _scaling, [5] _rotation, [6] _opacity,
# [7] max_radii2D, [8] xyz_grad_accum, [9] denom, [10] opt state,
# [11] spatial_lr_scale, [12] _normals, [13] normal_error_accum.
# Assign the parameter tensors directly -- no optimizer/training_setup needed
# for a pure save_ply export (restore() would require an OptimizationParams).
(g.active_sh_degree, g._xyz, g._features_dc, g._features_rest,
 g._scaling, g._rotation, g._opacity) = model_args[:7]
g.spatial_lr_scale = model_args[11]
g._normals = model_args[12]

out_dir = os.path.join(model_dir, "point_cloud", f"iteration_{it}")
os.makedirs(out_dir, exist_ok=True)
out = os.path.join(out_dir, "point_cloud.ply")
g.save_ply(out)
print(f"wrote {out}  ({g._xyz.shape[0]} gaussians, sh_degree {g.active_sh_degree})")
