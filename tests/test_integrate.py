"""Session E2 verification (GUTWrap Path A) -- run inside the `geer` container:
    docker exec -w /home geer python tests/test_integrate.py            # analytic only
    docker exec -w /home geer python tests/test_integrate.py --crosscheck  # + field-vs-TSDF

Analytic single-Gaussian test of the forward-only `integrate_points` CUDA
kernel (geer-rasterizer) through the full Python path
(gaussian_wrapping/fields.py::evaluate_occupancy_integrated):
  1. one opaque Gaussian (opacity 0.99, scale 0.1) placed 5 units in front of
     a real truck camera along its optical axis;
  2. query point 3 units out (in FRONT of the Gaussian): its ray-integrated
     alpha must be ~0 (< 0.05) -- the Gaussian's sorted depth (5) exceeds the
     point's own ray parameter (3), so it is not counted;
  3. query point 7 units out (BEHIND the Gaussian): alpha must be > 0.5
     (~0.99 = the Gaussian's peak alpha on the central ray);
  4. the behind-point's integrated alpha must match the rendered pixel's
     total accumulated alpha at its projection (1 - final_T, recovered via
     the bg trick: black colors + white bg => pixel value == final_T);
  5. a point that projects into NO view -> occupancy 0 -> field == +0.5
     (vacant), exactly.

--crosscheck additionally loads the 30k truck model and reports (does NOT
gate) the sign-agreement fraction between the integrated field and the
Session-E TSDF field on 200k random pivots (expect >0.7 but NOT 1.0 -- they
are different fields).
"""
import math
import os
import sys
import time
from argparse import ArgumentParser

import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arguments import ModelParams, PipelineParams, OptimizationParams
from scene import Scene
from scene.gaussian_model import GaussianModel
from gaussian_renderer import render
from utils.general_utils import inverse_sigmoid

from gaussian_wrapping.fields import (
    evaluate_occupancy_integrated,
    fuse_tsdf_at_points,
    render_depth_maps,
    world_to_view,
)
from gaussian_wrapping.fisheye_proj import project_view_to_pixel
from gaussian_wrapping.pivots import extract_gaussian_pivots

TRUCK_SOURCE = "/home/data/tt/datasets/truck"
TRUCK_MODEL = "/home/output/sessD_full30k"
TRUCK_ITER = 30000


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

    # mirror train.py:40-49 / tests/test_normal_field.py::build_scene
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


def make_single_gaussian(world_pos, scale=0.1, opacity=0.99):
    synth = GaussianModel(3)
    synth._xyz = nn.Parameter(world_pos.view(1, 3).contiguous())
    synth._features_dc = nn.Parameter(torch.zeros(1, 1, 3, device="cuda"))
    synth._features_rest = nn.Parameter(
        torch.zeros(1, (synth.max_sh_degree + 1) ** 2 - 1, 3, device="cuda"))
    synth._scaling = nn.Parameter(torch.full((1, 3), math.log(scale), device="cuda"))
    synth._rotation = nn.Parameter(torch.tensor([[1.0, 0.0, 0.0, 0.0]], device="cuda"))
    synth._opacity = nn.Parameter(inverse_sigmoid(torch.tensor([[opacity]], device="cuda")))
    synth._normals = nn.Parameter(torch.tensor([[0.0, 0.0, 1.0]], device="cuda"))
    synth.active_sh_degree = synth.max_sh_degree
    return synth


def analytic_test(pipe, camera):
    # Camera optical axis in world space (same derivation as
    # tests/test_normal_field.py check4, row-vector convention).
    view_to_world_full = camera.world_view_transform.inverse()
    view_dir_world = F.normalize(view_to_world_full[2, :3], dim=0)
    cam_center = camera.camera_center

    gaussian_pos = cam_center + 5.0 * view_dir_world
    synth = make_single_gaussian(gaussian_pos, scale=0.1, opacity=0.99)

    p_front = cam_center + 3.0 * view_dir_world     # in front of the Gaussian
    p_behind = cam_center + 7.0 * view_dir_world    # behind the Gaussian
    p_nowhere = cam_center - 10.0 * view_dir_world  # behind the camera: no view
    points = torch.stack([p_front, p_behind, p_nowhere], dim=0)

    field = evaluate_occupancy_integrated(points, [camera], synth, pipe)
    alpha = 0.5 - field  # occupancy = alpha_integrated (iso=0)

    a_front, a_behind, a_nowhere = alpha.tolist()
    f_nowhere = field[2].item()
    print(f"[analytic] alpha_integrated: front(3u)={a_front:.6f}, "
          f"behind(7u)={a_behind:.6f}, nowhere={a_nowhere:.6f}, "
          f"field(nowhere)={f_nowhere:.6f}")

    assert a_front < 0.05, f"front point alpha too high: {a_front}"
    assert a_behind > 0.5, f"behind point alpha too low: {a_behind}"
    assert abs(f_nowhere - 0.5) < 1e-6, \
        f"no-view point must be exactly vacant (field=+0.5), got {f_nowhere}"
    print("PASS analytic: front < 0.05, behind > 0.5, no-view field == +0.5")

    # Cross-check: integrated alpha(behind) == rendered pixel's total
    # accumulated alpha at the behind-point's projection. bg trick:
    # colors_precomp = 0, bg = 1 => out_color = C + T*bg = final_T.
    with torch.no_grad():
        x_view = world_to_view(camera, p_behind.view(1, 3))
        u, v, valid = project_view_to_pixel(camera, x_view)
        assert bool(valid[0]), "behind point should project inside the image"
        px = int(u[0].round().clamp(0, camera.image_width - 1).item())
        py = int(v[0].round().clamp(0, camera.image_height - 1).item())

        white_bg = torch.ones(3, device="cuda")
        pkg = render(camera, synth, pipe, white_bg,
                     override_color=torch.zeros(1, 3, device="cuda"))
        final_T = pkg["render"][0, py, px].item()
        alpha_render = 1.0 - final_T

    diff = abs(a_behind - alpha_render)
    print(f"[analytic] render cross-check @ pixel ({py},{px}): "
          f"alpha_render={alpha_render:.6f}, alpha_integrated={a_behind:.6f}, "
          f"|diff|={diff:.6f}")
    assert diff < 0.02, f"integrated vs rendered alpha mismatch: {diff}"
    print("PASS analytic: integrated alpha(behind) matches rendered pixel alpha")


@torch.no_grad()
def field_crosscheck(pipe, gaussians, cameras, n_pivots=200_000):
    """Report-only: sign agreement between integrated and TSDF fields on
    random pivots from the trained model (different fields -- expect >0.7,
    NOT 1.0)."""
    means = gaussians.get_xyz.detach()
    scales = gaussians.get_scaling.detach()
    rotations = gaussians.get_rotation.detach()
    normals = gaussians.get_normals.detach()

    # Same scene-radius crop as mesh_extraction (far-field sky Gaussians are
    # meaningless for a surface-field comparison).
    cam_centers = torch.stack([c.camera_center for c in cameras], dim=0)
    avg_center = cam_centers.mean(dim=0, keepdim=True)
    scene_radius = (cam_centers - avg_center).norm(dim=-1).max().item() * 1.1
    keep = (means - avg_center).norm(dim=-1) <= 2.0 * scene_radius
    means, scales, rotations, normals = means[keep], scales[keep], rotations[keep], normals[keep]

    g = torch.Generator(device="cuda").manual_seed(0)
    sel = torch.randperm(means.shape[0], generator=g, device="cuda")[: n_pivots // 2]
    pivots, _ = extract_gaussian_pivots(
        means[sel], scales[sel], rotations[sel], normals[sel], std_factor=3.33)
    print(f"[crosscheck] {pivots.shape[0]} pivots from {sel.shape[0]} random Gaussians "
          f"(radius-cropped pool: {means.shape[0]})")

    trunc_margin = max(2e-3 * scene_radius, 5.0 * scales.mean().item())

    t0 = time.time()
    depth_maps = render_depth_maps(cameras, gaussians, pipe)
    tsdf = fuse_tsdf_at_points(pivots, cameras, depth_maps, trunc_margin)
    t_tsdf = time.time() - t0
    del depth_maps
    torch.cuda.empty_cache()

    t0 = time.time()
    integrated = evaluate_occupancy_integrated(pivots, cameras, gaussians, pipe)
    t_int = time.time() - t0

    agree = ((tsdf > 0) == (integrated > 0)).float().mean().item()
    print(f"[crosscheck] TSDF field: {t_tsdf:.1f}s | integrated field: {t_int:.1f}s "
          f"({len(cameras)} views)")
    print(f"[crosscheck] sign stats: tsdf +{(tsdf > 0).float().mean().item():.4f} "
          f"| integrated +{(integrated > 0).float().mean().item():.4f}")
    print(f"[crosscheck] SIGN AGREEMENT FRACTION = {agree:.4f} "
          f"(report-only; different fields, expect >0.7 but not 1.0)")
    return agree


def main():
    parser = ArgumentParser()
    parser.add_argument("--crosscheck", action="store_true",
                        help="also run the 200k-pivot integrated-vs-TSDF "
                             "sign-agreement report (loads the full 30k model fields)")
    args = parser.parse_args()

    dataset, opt, pipe, gaussians, scene = build_scene()
    print(f"[INFO] Loaded {gaussians.get_xyz.shape[0]} gaussians from "
          f"{TRUCK_MODEL}/point_cloud/iteration_{TRUCK_ITER}")

    camera = scene.getTrainCameras()[0]
    analytic_test(pipe, camera)

    if args.crosscheck:
        field_crosscheck(pipe, gaussians, scene.getTrainCameras())

    print("\nAll Session E2 integrate checks PASSED.")


if __name__ == "__main__":
    main()
