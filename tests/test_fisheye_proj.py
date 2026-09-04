"""Session E verification (GUTWrap Path A) -- run inside the `geer` container:
    docker exec -w /home geer python tests/test_fisheye_proj.py

Checks (see GUTWrap_Discussion/3DGEERGW_EXECUTION.md, Session E, E1/E2):
  1. PH round-trip: for a grid of real truck-camera pixels, unproject via
     utils/ray_normals.get_ray_dirs_view (already validated in Session C),
     place a point at a fixed distance along that ray in view space, and
     reproject with gaussian_wrapping.fisheye_proj.project_view_to_pixel.
     Must agree with the original pixel to < 0.5 px.
  2. EQ self-consistency: build a synthetic equidistant raymap analytically
     from the *same* model implemented in fisheye_proj.py (no real fisheye
     camera/raymap involved), round-trip pixel -> ray -> 3D point -> pixel,
     must agree to < 0.5 px. (fisheye_proj implements the *undistorted*
     equidistant model; the real KB raymap includes distortion coefficients
     baked in at data-prep time, so this is intentionally a self-consistency
     check, not a round-trip against real X5 data -- see fisheye_proj.py's
     module docstring.)
  3. world_to_view consistency (gaussian_wrapping/fields.py): a point AT
     camera.camera_center must map to x_view ~= 0, and a point 5 units along
     the camera's forward axis (in world space) must map to x_view ~= (0,0,5).
  4. KB round-trip against a *real* distorted raymap (ScanNet++ 1d003b07bd
     dslr, genuine fitted k1-k4): unproject via get_ray_dirs_view (which for
     render_model==1 just reads camera.raymap directly, i.e. the true
     distorted ray directions baked in by data/scnt/scnt_raymap.py), place a
     point at a fixed distance along that ray, and reproject with
     project_view_to_pixel. Must agree with the original pixel to < 0.5px --
     this is the regression check for the fisheye_proj.py distortion fix
     (previously this check would have failed by tens of pixels since the
     old code never applied k1-k4 at all).
"""
import math
import os
import sys
from argparse import ArgumentParser
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arguments import ModelParams, PipelineParams, OptimizationParams
from scene import Scene
from scene.gaussian_model import GaussianModel
from utils.ray_normals import get_ray_dirs_view

from gaussian_wrapping.fisheye_proj import project_view_to_pixel
from gaussian_wrapping.fields import world_to_view

TRUCK_SOURCE = "/home/data/tt/datasets/truck"
TRUCK_MODEL = "/home/output/tt/truck"
TRUCK_ITER = 30000

SCNT_SOURCE = "/home/data/scnt/datasets/1d003b07bd/dslr"
SCNT_RAYMAP = "/home/data/scnt/datasets/1d003b07bd/dslr/raymap_fisheye.npy"
SCNT_MODEL = "/home/output/scnt_kb_vanilla"
SCNT_ITER = 30000


def build_scene():
    parser = ArgumentParser()
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    args = parser.parse_args([
        "-s", TRUCK_SOURCE,
        "-m", TRUCK_MODEL,
        "--dataset", "COLMAP",
        "--camera_model", "PINHOLE",
    ])
    dataset = lp.extract(args)
    opt = op.extract(args)
    pipe = pp.extract(args)

    dataset.fov_mod = None
    dataset.sample_step = 0.002
    dataset.render_model = "PH"
    dataset.focal_scaling = 1.0
    dataset.distortion_scaling = 1.0
    dataset.mirror_shift = 0.0
    dataset.raymap = None

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=TRUCK_ITER, shuffle=False)
    return dataset, opt, pipe, gaussians, scene


def build_scene_kb():
    parser = ArgumentParser()
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    args = parser.parse_args([
        "-s", SCNT_SOURCE,
        "-m", SCNT_MODEL,
        "--dataset", "SCANNETPP",
        "--camera_model", "FISHEYE",
    ])
    dataset = lp.extract(args)
    opt = op.extract(args)
    pipe = pp.extract(args)

    dataset.fov_mod = None
    dataset.sample_step = 0.002
    dataset.render_model = "KB"
    dataset.focal_scaling = 1.0
    dataset.distortion_scaling = 1.0
    dataset.mirror_shift = 0.0
    dataset.raymap = np.load(SCNT_RAYMAP)

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=SCNT_ITER, shuffle=False)
    return dataset, opt, pipe, gaussians, scene


def check4_kb_round_trip_real_distortion(camera, distance=5.0, stride=53):
    """KB round-trip against the real distorted raymap (genuine k1-k4)."""
    H, W = camera.image_height, camera.image_width
    dirs = get_ray_dirs_view(camera)  # (3,H,W) unit view-space rays, from the real raymap

    ys = torch.arange(0, H, stride, device="cuda")
    xs = torch.arange(0, W, stride, device="cuda")
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    grid_y = grid_y.reshape(-1)
    grid_x = grid_x.reshape(-1)

    ray = dirs[:, grid_y, grid_x].permute(1, 0)  # (N,3)
    points_view = ray * distance

    u, v, valid = project_view_to_pixel(camera, points_view)

    err_u = (u - grid_x.float()).abs()
    err_v = (v - grid_y.float()).abs()
    err = torch.sqrt(err_u ** 2 + err_v ** 2)

    n_valid = int(valid.sum())
    n_total = valid.numel()
    max_err = err[valid].max().item() if n_valid else float("nan")
    mean_err = err[valid].mean().item() if n_valid else float("nan")
    print(f"[check4] KB round-trip vs real distorted raymap: N={n_total}, valid={n_valid}, "
          f"mean_err={mean_err:.6f}px, max_err={max_err:.6f}px")
    assert n_valid == n_total, f"check4: {n_total - n_valid} / {n_total} points invalid"
    assert max_err < 0.5, f"check4 FAILED: max round-trip error {max_err:.4f}px >= 0.5px"
    print("PASS check4: KB round-trip against real distortion < 0.5px")


def check1_ph_round_trip(camera, distance=5.0, stride=37):
    """PH round-trip against get_ray_dirs_view on a real truck camera."""
    H, W = camera.image_height, camera.image_width
    dirs = get_ray_dirs_view(camera)  # (3,H,W) unit view-space rays

    ys = torch.arange(0, H, stride, device="cuda")
    xs = torch.arange(0, W, stride, device="cuda")
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    grid_y = grid_y.reshape(-1)
    grid_x = grid_x.reshape(-1)

    ray = dirs[:, grid_y, grid_x].permute(1, 0)  # (N,3)
    points_view = ray * distance  # (N,3) points along the ray at fixed distance

    u, v, valid = project_view_to_pixel(camera, points_view)

    err_u = (u - grid_x.float()).abs()
    err_v = (v - grid_y.float()).abs()
    err = torch.sqrt(err_u ** 2 + err_v ** 2)

    assert valid.all(), f"check1: {int((~valid).sum())} / {valid.numel()} points invalid"
    max_err = err.max().item()
    mean_err = err.mean().item()
    print(f"[check1] PH round-trip vs get_ray_dirs_view: N={err.numel()}, "
          f"mean_err={mean_err:.6f}px, max_err={max_err:.6f}px")
    assert max_err < 0.5, f"check1 FAILED: max round-trip error {max_err:.4f}px >= 0.5px"
    print("PASS check1: PH round-trip < 0.5px")


def check2_eq_self_consistency():
    """EQ self-consistency: build a synthetic EQ raymap analytically from the
    same model as fisheye_proj.py, round-trip pixel -> ray -> point -> pixel."""
    W, H = 640, 480
    focal_x = focal_y = 300.0
    principal_x, principal_y = W / 2.0, H / 2.0
    cam = SimpleNamespace(
        render_model=1,
        focal_x=focal_x, focal_y=focal_y,
        principal_x=principal_x, principal_y=principal_y,
        image_width=W, image_height=H,
    )

    # Sample a grid of pixels well within the <89deg valid cone.
    us = torch.linspace(20, W - 20, 25)
    vs = torch.linspace(20, H - 20, 25)
    grid_v, grid_u = torch.meshgrid(vs, us, indexing="ij")
    u0 = grid_u.reshape(-1)
    v0 = grid_v.reshape(-1)

    # Analytic EQ unprojection, exactly inverting fisheye_proj's EQ formula
    # (no distortion_coeffs on this SimpleNamespace cam -> theta_d == theta,
    # and no "-0.5" pixel-center shift -- see fisheye_proj.py's module docstring):
    #   u = principal_x + focal_x * theta * cos(phi)
    #   v = principal_y + focal_y * theta * sin(phi)
    tx = (u0 - principal_x) / focal_x
    ty = (v0 - principal_y) / focal_y
    theta = torch.sqrt(tx ** 2 + ty ** 2)
    phi = torch.atan2(ty, tx)
    ray = torch.stack([
        torch.sin(theta) * torch.cos(phi),
        torch.sin(theta) * torch.sin(phi),
        torch.cos(theta),
    ], dim=-1)  # (N,3) unit ray

    distance = 7.0
    points_view = ray * distance

    u1, v1, valid = project_view_to_pixel(cam, points_view)

    assert valid.all(), f"check2: {int((~valid).sum())} / {valid.numel()} points invalid"
    err = torch.sqrt((u1 - u0) ** 2 + (v1 - v0) ** 2)
    max_err = err.max().item()
    mean_err = err.mean().item()
    print(f"[check2] EQ self-consistency: N={err.numel()}, "
          f"mean_err={mean_err:.6f}px, max_err={max_err:.6f}px")
    assert max_err < 0.5, f"check2 FAILED: max round-trip error {max_err:.4f}px >= 0.5px"
    print("PASS check2: EQ self-consistency < 0.5px")


def check3_world_to_view_consistency(camera):
    """world_to_view: camera_center -> ~0; point 5m along forward axis -> (0,0,5)."""
    center = camera.camera_center.view(1, 3)
    x_view_center = world_to_view(camera, center).view(3)
    err_center = x_view_center.norm().item()
    print(f"[check3] ||world_to_view(camera_center)|| = {err_center:.6f}")
    assert err_center < 1e-2, f"check3 FAILED: camera_center did not map to ~0 ({err_center})"

    # Forward axis in world space: view-space +Z direction mapped to world via
    # the Session-C-pinned convention world_view_transform[:3,:3] @ e_z.
    R = camera.world_view_transform[:3, :3]
    e_z = torch.tensor([0.0, 0.0, 1.0], device="cuda")
    forward_world = F.normalize(torch.einsum("ij,j->i", R, e_z), dim=0)

    point_world = camera.camera_center + 5.0 * forward_world
    x_view_point = world_to_view(camera, point_world.view(1, 3)).view(3)
    expected = torch.tensor([0.0, 0.0, 5.0], device="cuda")
    err_point = (x_view_point - expected).norm().item()
    print(f"[check3] world_to_view(camera_center + 5*forward) = "
          f"{x_view_point.tolist()} (expected ~[0,0,5]), err={err_point:.6f}")
    assert err_point < 1e-2, f"check3 FAILED: forward-axis point error {err_point}"
    print("PASS check3: world_to_view consistency (camera_center -> 0, forward*5 -> (0,0,5))")


def main():
    dataset, opt, pipe, gaussians, scene = build_scene()
    camera = scene.getTrainCameras()[0]

    check1_ph_round_trip(camera)
    check2_eq_self_consistency()
    check3_world_to_view_consistency(camera)

    _, _, _, _, scnt_scene = build_scene_kb()
    scnt_camera = scnt_scene.getTrainCameras()[0]
    check4_kb_round_trip_real_distortion(scnt_camera)

    print("\nAll Session E fisheye_proj/fields checks PASSED.")


if __name__ == "__main__":
    main()
