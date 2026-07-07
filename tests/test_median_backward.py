"""Session G verification -- differentiable median depth (implicit backward).
Run inside the geer container:
    docker exec -w /home geer python tests/test_median_backward.py

Tests (synthetic gaussians on a real truck camera, PH mode):
  1. Single opaque gaussian: analytic unit response -- the stored (Euclidean)
     median's gradient w.r.t. an along-ray center shift is exactly 1 in the
     continuous model; FD-verify the full chain within tolerance, and verify
     the opacity gradient sign (more opaque => crossing earlier => md down).
  2. Wall + faint floater: the OPACITY RELIEF VALVE -- the floater (in front,
     t_delta > 0, shape-gradient damped 1000x) must receive a NEGATIVE
     opacity gradient when the loss wants the median deeper, matching FD.
  3. No-median-loss regression: color-only backward still works, grads finite.
"""
import math
import os
import sys

import torch
import torch.nn.functional as F
from torch import nn
from argparse import ArgumentParser

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arguments import ModelParams, PipelineParams, OptimizationParams
from scene import Scene
from scene.gaussian_model import GaussianModel
from gaussian_renderer import render
from utils.general_utils import inverse_sigmoid

TRUCK_SOURCE = "/home/data/tt/datasets/truck"
TRUCK_MODEL = "/home/output/tt/truck"


def build_camera():
    parser = ArgumentParser()
    lp = ModelParams(parser); op = OptimizationParams(parser); pp = PipelineParams(parser)
    args = parser.parse_args([
        "-s", TRUCK_SOURCE, "-m", TRUCK_MODEL,
        "--dataset", "COLMAP", "--camera_model", "PINHOLE",
    ])
    dataset = lp.extract(args); pipe = pp.extract(args)
    dataset.render_model = "PH"
    dataset.fov_mod = None; dataset.sample_step = 0.002
    dataset.focal_scaling = 1.0; dataset.distortion_scaling = 1.0
    dataset.mirror_shift = 0.0; dataset.raymap = None
    g_dummy = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, g_dummy, load_iteration=None, shuffle=False)
    cam = scene.getTrainCameras()[0]
    view_dir_world = cam.world_view_transform.inverse()[2, :3].cuda()
    view_dir_world = F.normalize(view_dir_world, dim=0)
    return cam, pipe, view_dir_world


def make_gaussians(positions, opacities, scale=0.05):
    n = len(positions)
    g = GaussianModel(3)
    g._xyz = nn.Parameter(torch.stack(positions).contiguous())
    g._features_dc = nn.Parameter(torch.rand(n, 1, 3, device="cuda") * 0.5)
    g._features_rest = nn.Parameter(torch.zeros(n, (g.max_sh_degree + 1) ** 2 - 1, 3, device="cuda"))
    g._scaling = nn.Parameter(torch.full((n, 3), math.log(scale), device="cuda"))
    g._rotation = nn.Parameter(torch.tensor([[1.0, 0.0, 0.0, 0.0]], device="cuda").repeat(n, 1))
    g._opacity = nn.Parameter(inverse_sigmoid(torch.tensor(opacities, device="cuda").view(n, 1)))
    g._normals = nn.Parameter(F.normalize(torch.randn(n, 3, device="cuda"), dim=-1))
    g.active_sh_degree = g.max_sh_degree
    return g


def render_md(cam, g, pipe):
    bg = torch.zeros(3, device="cuda")
    pkg = render(cam, g, pipe, bg)
    return pkg["median_depth"], pkg["gidx"]


def pick_pixel(md, gidx, want_id):
    sel = (gidx == want_id) & (md[0] > 0)
    assert sel.any(), f"gaussian {want_id} owns no median pixels"
    ys, xs = sel.nonzero(as_tuple=True)
    cy, cx = ys.float().mean(), xs.float().mean()
    k = ((ys.float() - cy) ** 2 + (xs.float() - cx) ** 2).argmin()
    return int(ys[k]), int(xs[k])


def test_single_gaussian(cam, pipe, view_dir):
    center = cam.camera_center.cuda() + 3.0 * view_dir
    g = make_gaussians([center], [0.95])
    md, gidx = render_md(cam, g, pipe)
    y, x = pick_pixel(md, gidx, 0)
    md0 = md[0, y, x].item()

    loss = md[0, y, x]
    loss.backward()
    grad_xyz = g._xyz.grad[0].clone()
    grad_op_raw = g._opacity.grad[0, 0].item()
    assert torch.isfinite(grad_xyz).all() and grad_xyz.norm() > 0, "xyz grad missing/NaN"

    # analytic unit response: d(md)/d(shift along +ray) == 1 in the model
    along = torch.dot(grad_xyz, view_dir).item()
    print(f"[single] md={md0:.4f}  grad_along_ray={along:.4f} (expect ~1)  "
          f"grad_perp={ (grad_xyz - along * view_dir).norm().item():.4f}  "
          f"grad_opacity_raw={grad_op_raw:.6f}")
    assert 0.5 < along < 1.5, f"along-ray response {along} not ~1 -- formula mis-port"

    # FD cross-check of the full chain (shift the center along the ray)
    eps = 0.01
    with torch.no_grad():
        g_p = make_gaussians([center + eps * view_dir], [0.95])
        md_p, _ = render_md(cam, g_p, pipe)
        g_m = make_gaussians([center - eps * view_dir], [0.95])
        md_m, _ = render_md(cam, g_m, pipe)
    fd = (md_p[0, y, x].item() - md_m[0, y, x].item()) / (2 * eps)
    print(f"[single] FD d(md)/d(along-ray shift) = {fd:.4f} vs autograd {along:.4f}")
    assert fd > 0.5, f"FD sanity failed ({fd}) -- forward interpolation broken?"
    assert abs(along - fd) / max(abs(fd), 1e-6) < 0.5, "autograd vs FD disagree >50%"

    # opacity sign: more opaque => T crosses 0.5 earlier => md decreases
    d_op = 0.05
    with torch.no_grad():
        g_p = make_gaussians([center], [0.95])
        g_p._opacity += d_op
        md_p, _ = render_md(cam, g_p, pipe)
        g_m = make_gaussians([center], [0.95])
        g_m._opacity -= d_op
        md_m, _ = render_md(cam, g_m, pipe)
    fd_op = (md_p[0, y, x].item() - md_m[0, y, x].item()) / (2 * d_op)
    print(f"[single] FD d(md)/d(raw opacity) = {fd_op:.5f} vs autograd {grad_op_raw:.5f}")
    assert fd_op < 0, "FD opacity sign wrong -- forward broken?"
    assert grad_op_raw < 0, "autograd opacity sign wrong (expected negative)"
    print("PASS single-gaussian: unit along-ray response, FD agreement, opacity sign")


def test_floater_relief_valve(cam, pipe, view_dir):
    c = cam.camera_center.cuda()
    wall = c + 3.0 * view_dir
    floater = c + 1.5 * view_dir
    g = make_gaussians([wall, floater], [0.95, 0.30])
    md, gidx = render_md(cam, g, pipe)
    y, x = pick_pixel(md, gidx, 0)   # median owned by the wall

    md[0, y, x].backward()           # loss = +md => wants median DEEPER
    g_op = g._opacity.grad
    g_xyz = g._xyz.grad
    assert torch.isfinite(g_op).all() and torch.isfinite(g_xyz).all()

    # FD on the floater's raw opacity
    d_op = 0.05
    with torch.no_grad():
        g_p = make_gaussians([wall, floater], [0.95, 0.30]); g_p._opacity[1] += d_op
        md_p, _ = render_md(cam, g_p, pipe)
        g_m = make_gaussians([wall, floater], [0.95, 0.30]); g_m._opacity[1] -= d_op
        md_m, _ = render_md(cam, g_m, pipe)
    fd_fl = (md_p[0, y, x].item() - md_m[0, y, x].item()) / (2 * d_op)
    print(f"[floater] autograd d(md)/d(op_floater)={g_op[1,0].item():.5f}  FD={fd_fl:.5f}  "
          f"wall grads: op={g_op[0,0].item():.5f} xyz_along={torch.dot(g_xyz[0], view_dir).item():.4f}")
    assert fd_fl < 0, "FD floater-opacity sign unexpected -- adjust scene params"
    assert g_op[1, 0].item() < 0, "RELIEF VALVE MISSING: floater opacity grad not negative"
    assert torch.dot(g_xyz[0], view_dir).item() > 0.2, "wall along-ray grad missing"
    print("PASS floater: opacity relief valve present with FD-consistent sign")


def test_color_only_regression(cam, pipe, view_dir):
    g = make_gaussians([cam.camera_center.cuda() + 3.0 * view_dir], [0.95])
    bg = torch.zeros(3, device="cuda")
    pkg = render(cam, g, pipe, bg)
    pkg["render"].sum().backward()
    for name, p in [("xyz", g._xyz), ("opacity", g._opacity), ("f_dc", g._features_dc)]:
        assert p.grad is not None and torch.isfinite(p.grad).all(), f"{name} grad broken"
    print("PASS color-only backward regression (median path dormant)")


def main():
    torch.manual_seed(0)
    cam, pipe, view_dir = build_camera()
    print(f"camera: {cam.image_name}, {cam.image_width}x{cam.image_height}")
    test_single_gaussian(cam, pipe, view_dir)
    test_floater_relief_valve(cam, pipe, view_dir)
    test_color_only_regression(cam, pipe, view_dir)
    print("\nAll Session G median-backward checks PASSED.")


if __name__ == "__main__":
    main()
