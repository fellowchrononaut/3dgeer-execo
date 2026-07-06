"""Session C verification (GUTWrap Path A) -- run inside the `geer` container:
    docker exec -w /home geer python tests/test_normal_field.py

Loads the pre-Session-A truck 30k checkpoint (zero nx/ny/nz -> exercises the
load_ply all-zero-normal guard added in scene/gaussian_model.py) and checks:
  1. median_depth -> depth_to_normals_via_rays -> PNG dump (+ depth vis)
  2. render_normal_field -> PNG dump
  3. gradient sanity: render_normal_field(...).sum().backward() populates
     gaussians._normals.grad
  4. convention pin: a synthetic on-axis Gaussian, 3 units in front of a real
     truck camera, with normal = -view_dir_world, decodes back to
     -view_dir_world through render_normal_field; also round-trips
     view_normals_to_world against its own inverse.
"""
import math
import os
import sys

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torch import nn
from argparse import ArgumentParser

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arguments import ModelParams, PipelineParams, OptimizationParams
from scene import Scene
from scene.gaussian_model import GaussianModel
from gaussian_renderer import render, render_normal_field
from utils.ray_normals import depth_to_normals_via_rays, view_normals_to_world, world_to_view_normals
from utils.general_utils import inverse_sigmoid

OUT_DIR = "/home/output/sessC_test"
TRUCK_SOURCE = "/home/data/tt/datasets/truck"
TRUCK_MODEL = "/home/output/tt/truck"
TRUCK_ITER = 30000


def save_normal_png(path, normal_map_chw):
    """normal_map_chw: (3,H,W) tensor already in [-1,1]. Saves as (n*0.5+0.5)."""
    vis = (normal_map_chw * 0.5 + 0.5).clamp(0, 1).permute(1, 2, 0).detach().cpu().numpy()
    vis = (vis * 255).astype(np.uint8)
    cv2.imwrite(path, vis[:, :, [2, 1, 0]])  # RGB -> BGR for cv2


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

    # mirror train.py:40-49
    dataset.fov_mod = None
    dataset.sample_step = 0.002
    dataset.render_model = "PH"
    dataset.focal_scaling = 1.0
    dataset.distortion_scaling = 1.0
    dataset.mirror_shift = 0.0
    dataset.raymap = None

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=TRUCK_ITER, shuffle=False)
    # NOTE: gaussians.training_setup(opt) is NOT called here -- it reads/builds
    # self._exposure / self.pretrained_exposures, which are only populated by
    # create_from_pcd(), not by load_ply() (pre-existing gap, out of Session C
    # scope). load_ply() already returns every parameter (incl. _normals) with
    # requires_grad_(True), which is all check 3 needs -- verified below.
    assert gaussians._normals.requires_grad, "load_ply() did not set _normals.requires_grad"
    return dataset, opt, pipe, gaussians, scene


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    dataset, opt, pipe, gaussians, scene = build_scene()
    print(f"[INFO] Loaded {gaussians.get_xyz.shape[0]} gaussians from "
          f"{TRUCK_MODEL}/point_cloud/iteration_{TRUCK_ITER}")

    # all-zero-normal guard sanity: every stored normal must be (near) unit norm
    norms = gaussians.get_normals.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-3), \
        "load_ply all-zero-normal guard failed: get_normals is not unit-norm everywhere"
    print(f"[INFO] get_normals unit-norm check OK (min={norms.min().item():.6f}, "
          f"max={norms.max().item():.6f})")

    viewpoint_cam = scene.getTrainCameras()[0]
    background = torch.zeros(3, device="cuda")

    # ---------------------------------------------------------------
    # Check 1: median_depth -> depth_to_normals_via_rays
    # ---------------------------------------------------------------
    with torch.no_grad():
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        median_depth = render_pkg["median_depth"]              # (1,H,W)
        target_n_view, valid = depth_to_normals_via_rays(viewpoint_cam, median_depth)
        target_n_world = view_normals_to_world(viewpoint_cam, target_n_view)

    coverage_1 = valid.float().mean().item()
    save_normal_png(os.path.join(OUT_DIR, "check1_depth_normal_world.png"), target_n_world)

    dm = median_depth.squeeze(0)
    valid_d = dm > 0
    if valid_d.any():
        dmin, dmax = dm[valid_d].min(), dm[valid_d].max()
        dm_norm = torch.where(valid_d, (dm - dmin) / (dmax - dmin + 1e-8), torch.zeros_like(dm))
    else:
        dm_norm = torch.zeros_like(dm)
    depth_vis = (dm_norm.cpu().numpy() * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(OUT_DIR, "check1_depth_vis.png"), depth_vis)

    assert coverage_1 > 0.3, f"median_depth coverage too low ({coverage_1:.3f})"
    print(f"PASS check1: depth_to_normals_via_rays -- coverage={coverage_1:.4f}, "
          f"PNGs -> check1_depth_normal_world.png / check1_depth_vis.png")

    # ---------------------------------------------------------------
    # Check 2: render_normal_field
    # ---------------------------------------------------------------
    with torch.no_grad():
        normal_map = render_normal_field(viewpoint_cam, gaussians, pipe)
    save_normal_png(os.path.join(OUT_DIR, "check2_normal_field.png"), normal_map)
    coverage_2 = (normal_map.norm(dim=0) > 1e-3).float().mean().item()
    assert coverage_2 > 0.3, f"render_normal_field coverage too low ({coverage_2:.3f})"
    print(f"PASS check2: render_normal_field -- coverage={coverage_2:.4f}, "
          f"PNG -> check2_normal_field.png")

    # ---------------------------------------------------------------
    # Check 3: gradient sanity
    # ---------------------------------------------------------------
    if gaussians._normals.grad is not None:
        gaussians._normals.grad = None
    normal_map_grad = render_normal_field(viewpoint_cam, gaussians, pipe)
    normal_map_grad.sum().backward()
    grad = gaussians._normals.grad
    assert grad is not None, "gaussians._normals.grad is None after backward"
    grad_norm = grad.norm().item()
    assert grad_norm > 1e-6, f"gradient norm too small ({grad_norm})"
    print(f"PASS check3: gradient sanity -- grad_norm={grad_norm:.6f}")

    # ---------------------------------------------------------------
    # Check 4: convention pin
    # ---------------------------------------------------------------
    R = viewpoint_cam.world_view_transform[:3, :3]
    view_to_world_full = viewpoint_cam.world_view_transform.inverse()
    # Ground truth forward direction, derived independently of view_normals_to_world
    # via the full 4x4 inverse (row-vector convention): a point at view-space
    # depth d=1 along the optical axis maps to world position
    # view_to_world_full[2,:3] + camera_center, so the direction is row 2.
    view_dir_world_gt = F.normalize(view_to_world_full[2, :3], dim=0)

    n_view_axis = torch.tensor([0.0, 0.0, 1.0], device="cuda")
    candidate = F.normalize(torch.einsum("ij,j->i", R, n_view_axis), dim=0)
    axis_err = (candidate - view_dir_world_gt).norm().item()
    print(f"[INFO] view_normals_to_world convention check: "
          f"||R@e_z - WVT^-1_row2|| = {axis_err:.6f} (R = world_view_transform[:3,:3])")
    if axis_err > 1e-2:
        raise AssertionError(
            "view_normals_to_world convention (R @ n) does not match the analytic "
            "forward axis derived from world_view_transform.inverse(); the "
            "transpose form (R.T @ n) should be tried instead -- see utils/ray_normals.py")
    view_dir_world = view_dir_world_gt

    # round-trip: view_normals_to_world(cam, world_to_view(n)) == n
    n_world_probe = F.normalize(torch.randn(3, device="cuda"), dim=0)
    n_view_probe = torch.einsum("ij,i->j", R, n_world_probe)          # world_to_view = R^T @ n = n @ R
    n_world_roundtrip = view_normals_to_world(
        viewpoint_cam, n_view_probe.view(3, 1, 1)).view(3)
    roundtrip_err = (n_world_roundtrip - n_world_probe).norm().item()
    assert roundtrip_err < 1e-3, f"view_normals_to_world round-trip failed ({roundtrip_err})"
    print(f"[INFO] view_normals_to_world round-trip error = {roundtrip_err:.6f}")

    # synthetic on-axis Gaussian, 3 units in front of the camera
    gaussian_world_pos = viewpoint_cam.camera_center + 3.0 * view_dir_world
    gaussian_world_normal = F.normalize(-view_dir_world, dim=0)

    synth = GaussianModel(3)
    synth._xyz = nn.Parameter(gaussian_world_pos.view(1, 3).contiguous())
    synth._features_dc = nn.Parameter(torch.zeros(1, 1, 3, device="cuda"))
    synth._features_rest = nn.Parameter(
        torch.zeros(1, (synth.max_sh_degree + 1) ** 2 - 1, 3, device="cuda"))
    synth._scaling = nn.Parameter(torch.full((1, 3), math.log(0.1), device="cuda"))
    synth._rotation = nn.Parameter(torch.tensor([[1.0, 0.0, 0.0, 0.0]], device="cuda"))
    synth._opacity = nn.Parameter(inverse_sigmoid(torch.tensor([[0.99]], device="cuda")))
    synth._normals = nn.Parameter(gaussian_world_normal.view(1, 3).contiguous())
    synth.active_sh_degree = synth.max_sh_degree

    with torch.no_grad():
        synth_normal_map = render_normal_field(viewpoint_cam, synth, pipe)  # (3,H,W)

    coverage_map = synth_normal_map.norm(dim=0)
    peak = coverage_map.max().item()
    assert peak > 1e-3, "synthetic Gaussian did not project into the image"
    y, x = torch.nonzero(coverage_map == coverage_map.max())[0].tolist()
    decoded = synth_normal_map[:, y, x]
    decoded_unit = F.normalize(decoded, dim=0)
    conv_err = (decoded_unit - gaussian_world_normal).norm().item()
    print(f"[INFO] check4 pixel=({y},{x}), peak_coverage={peak:.4f}, "
          f"decoded={decoded.tolist()}, decoded_unit={decoded_unit.tolist()}, "
          f"expected={gaussian_world_normal.tolist()}")
    assert conv_err < 0.05, f"convention check failed: error={conv_err:.4f}"
    print(f"PASS check4: convention pin -- axis_err={axis_err:.6f}, "
          f"roundtrip_err={roundtrip_err:.6f}, decode_err={conv_err:.6f}, "
          f"peak_coverage={peak:.4f}")

    # ---------------------------------------------------------------
    # Check 5 (Session F2): world_to_view_normals round-trip
    # ---------------------------------------------------------------
    # (a) round-trip: world_to_view_normals(view_normals_to_world(n_view)) == n_view,
    # on a batch of random unit-ish view-space normal fields (reuse the (3,H,W)
    # map shape so this exercises the same einsum path as production use).
    torch.manual_seed(0)
    n_view_rand = F.normalize(torch.randn(3, 8, 8, device="cuda"), dim=0)
    n_world_rand = view_normals_to_world(viewpoint_cam, n_view_rand)
    n_view_roundtrip = world_to_view_normals(viewpoint_cam, n_world_rand)
    rt_err = (n_view_roundtrip - n_view_rand).norm(dim=0).max().item()
    assert rt_err < 1e-4, f"world_to_view_normals round-trip failed (max err {rt_err})"
    print(f"[INFO] check5a world_to_view_normals(view_normals_to_world(n)) round-trip max err = {rt_err:.6f}")

    # (b) independent cross-check using the SAME known forward-axis fact as
    # check 4: the camera's own optical axis is exactly [0,0,1] in its own
    # view space, and view_dir_world (derived above from world_view_transform
    # .inverse() row 2, NOT from view_normals_to_world) is that axis in world
    # space. world_to_view_normals(view_dir_world) must recover [0,0,1].
    forward_view_recovered = world_to_view_normals(
        viewpoint_cam, view_dir_world.view(3, 1, 1)).view(3)
    forward_expected = torch.tensor([0.0, 0.0, 1.0], device="cuda")
    forward_err = (forward_view_recovered - forward_expected).norm().item()
    assert forward_err < 1e-3, f"world_to_view_normals forward-axis check failed (err {forward_err})"
    print(f"[INFO] check5b world_to_view_normals(view_dir_world) = {forward_view_recovered.tolist()} "
          f"(expected [0,0,1]), err={forward_err:.6f}")
    print(f"PASS check5: world_to_view_normals round-trip={rt_err:.6f}, forward-axis err={forward_err:.6f}")

    print("\nAll Session C+F2 checks PASSED.")


if __name__ == "__main__":
    main()
