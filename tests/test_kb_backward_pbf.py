"""KB backward vs finite differences, interior and at the 90 deg edge.
Run inside the geer container:
    docker exec -w /home geer python tests/test_kb_backward_pbf.py

The forward renderCUDA skips, per ray, every Gaussian whose tangent-space PBF
does not contain the ray (mode 1 = KB/EQ); the backward must skip the same
ones, or its T = T / (1 - alpha) replays alphas that were never composited.
Near and beyond 90 deg (ray z <= 0) those alphas are large: T overflows and
the Gaussians on that pixel get NaN gradients in every parameter (found on
x5_vid012 v2: ~3k all-NaN Gaussians by 7k iterations).

Synthetic Gaussians on a real X5 KB camera (x5_vid012_v2_view):
  1. interior (theta 20-70 deg): analytic grads match central differences.
  2. edge (theta 80-98 deg, large): grads finite and match central differences.
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
from utils.raymap_utils import load_raymap

SOURCE = "/home/data/x5_vid012_v2_view"


def build_camera():
    parser = ArgumentParser()
    lp = ModelParams(parser); OptimizationParams(parser); pp = PipelineParams(parser)
    args = parser.parse_args([
        "-s", SOURCE, "-m", "/tmp/test_kb_backward_pbf", "--resolution", "2",
        "--dataset", "SCANNETPP", "--camera_model", "FISHEYE", "--near_threshold", "-0.01",
    ])
    dataset = lp.extract(args); pipe = pp.extract(args)
    dataset.render_model = "KB"
    dataset.fov_mod = None; dataset.sample_step = 0.002
    dataset.focal_scaling = 1.0; dataset.distortion_scaling = 1.0; dataset.mirror_shift = 0.0
    dataset.raymap = load_raymap(SOURCE + "/raymap_fisheye_cam{cam_id}.npy")
    os.makedirs(dataset.model_path, exist_ok=True)
    scene = Scene(dataset, GaussianModel(dataset.sh_degree), load_iteration=None, shuffle=False)
    return scene.getTrainCameras()[0], pipe


def make_gaussians(cam, n, theta_deg, dist, scale, opacity, seed):
    """n Gaussians at view-space angles theta_deg=(lo,hi) from the axis."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    u = lambda lo, hi: lo + (hi - lo) * torch.rand(n, device="cuda", generator=g)
    th = torch.deg2rad(u(*theta_deg)); ph = u(0, 2 * math.pi); d = u(*dist)
    p_view = torch.stack([d * th.sin() * ph.cos(), d * th.sin() * ph.sin(), d * th.cos()], -1)
    v2w = torch.inverse(cam.world_view_transform.T.cuda())
    xyz = (p_view @ v2w[:3, :3].T + v2w[:3, 3]).contiguous()
    gm = GaussianModel(3)
    gm._xyz = nn.Parameter(xyz)
    gm._features_dc = nn.Parameter(torch.rand(n, 1, 3, device="cuda", generator=g))
    gm._features_rest = nn.Parameter(torch.zeros(n, (gm.max_sh_degree + 1) ** 2 - 1, 3, device="cuda"))
    gm._scaling = nn.Parameter(torch.log(u(*scale))[:, None].repeat(1, 3) + 0.3 * torch.randn(n, 3, device="cuda", generator=g))
    gm._rotation = nn.Parameter(F.normalize(torch.randn(n, 4, device="cuda", generator=g), dim=-1))
    gm._opacity = nn.Parameter(inverse_sigmoid(u(*opacity))[:, None])
    gm._normals = nn.Parameter(F.normalize(torch.randn(n, 3, device="cuda", generator=g), dim=-1))
    gm.active_sh_degree = gm.max_sh_degree
    return gm


def loss_fn(cam, gm, pipe, w):
    img = render(cam, gm, pipe, torch.zeros(3, device="cuda"))["render"]
    return (img.double() * w).sum()


def check(name, cam, pipe, gm, w, n_fd=12, eps=1e-3):
    for p in (gm._xyz, gm._opacity, gm._features_dc):
        p.grad = None
    loss = loss_fn(cam, gm, pipe, w)
    loss.backward()
    grads = {"xyz": gm._xyz.grad.clone(), "opacity": gm._opacity.grad.clone()}
    nonfinite = sum(int((~torch.isfinite(gm_p.grad)).any(dim=-1).sum())
                    for gm_p in (gm._xyz, gm._opacity, gm._features_dc, gm._scaling, gm._rotation))
    # FD on the Gaussians with the largest opacity gradients (they are on screen)
    order = grads["opacity"].abs().flatten().argsort(descending=True)[:n_fd]
    ana, fd = [], []
    with torch.no_grad():
        for i in order.tolist():
            for pname, comps in (("opacity", [0]), ("xyz", [0, 1, 2])):
                p = getattr(gm, "_" + pname)
                for c in comps:
                    old = p[i, c].item()
                    p[i, c] = old + eps; lp = loss_fn(cam, gm, pipe, w).item()
                    p[i, c] = old - eps; lm = loss_fn(cam, gm, pipe, w).item()
                    p[i, c] = old
                    fd.append((lp - lm) / (2 * eps)); ana.append(grads[pname][i, c].item())
    ana = torch.tensor(ana, dtype=torch.float64); fd = torch.tensor(fd, dtype=torch.float64)
    cos = F.cosine_similarity(ana, fd, dim=0).item()
    rel = ((ana - fd).norm() / fd.norm().clamp_min(1e-12)).item()
    ok = nonfinite == 0 and cos > 0.99 and rel < 0.15
    print(f"[{name}] non-finite-grad Gaussians {nonfinite}; FD over {len(fd)} comps: "
          f"cos {cos:.4f}, rel err {rel:.3f} -> {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    cam, pipe = build_camera()
    g = torch.Generator(device="cuda").manual_seed(0)
    w = torch.randn(3, cam.image_height, cam.image_width, device="cuda", generator=g, dtype=torch.float64)
    ok = check("interior", cam, pipe, make_gaussians(cam, 60, (20, 70), (1.5, 4.0), (0.03, 0.2), (0.3, 0.9), 1), w)
    ok &= check("edge", cam, pipe, make_gaussians(cam, 60, (80, 98), (1.0, 3.0), (0.15, 0.6), (0.5, 0.95), 2), w)
    print("ALL PASS" if ok else "SOME FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
