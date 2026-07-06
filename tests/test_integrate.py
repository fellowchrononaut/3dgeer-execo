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

Session E2.1 (field-continuity fix) additions:
  - the behind-point analytic-vs-rendered-pixel check above now tolerates
    1e-2 (not an exact match) -- the kernel evaluates each query point's own
    exact sub-pixel ray, which by design differs slightly from the rendered
    pixel's pixel-center ray.
  - `continuity_test`: samples the field along two 1000-point segments
    through real truck geometry (lateral, perpendicular to a camera ray at
    fixed depth; and along-ray/depth) spanning ~2 pixel footprints near a
    real surface crossing, and asserts no adjacent-sample field jump exceeds
    1e-3 away from the true (legitimate) crossing -- the pre-fix field showed
    O(alpha) jumps here from pixel-snapping + the hard depth gate.
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
    view_to_world,
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
    # Session E2.1: integrate_points now evaluates the query point's own
    # EXACT sub-pixel ray (through p_behind's precise projection), while the
    # rendered pixel's alpha is accumulated along the PIXEL-CENTER ray -- the
    # two rays are no longer identical in general (that identity was exactly
    # the staircase bug this session fixes), so this is a close-agreement
    # check, not an exact match. 1e-2 comfortably separates "same Gaussian,
    # slightly different ray" from a real regression.
    assert diff < 1e-2, f"integrated vs rendered alpha mismatch: {diff}"
    print("PASS analytic: integrated alpha(behind) matches rendered pixel alpha (within 1e-2)")


@torch.no_grad()
def continuity_test(pipe, gaussians, camera):
    """Session E2.1 verification (b)(ii): continuity probe for the
    field-continuity fix. Before the fix, `integrateCUDA` snapped every
    query point to its nearest pixel center to look up `rayf`, so ALL points
    inside one ~pixel-sized 3D cell saw an identical ray and got identical
    alpha contributions -- the field was piecewise-constant, jumping by
    O(alpha_i) at pixel-footprint boundaries and at every Gaussian-depth
    crossing (the hard `depths[gid] <= t_x` gate). This probes two real
    segments through the truck for exactly that signature:
      (a) LATERAL: perpendicular to a real camera ray, at a fixed depth near
          the surface, spanning ~2 pixel footprints (world-space units).
      (b) DEPTH: along that same ray, spanning an equivalent range in depth
          (crosses whichever Gaussian(s) dominate the surface there).
    Both segments genuinely cross the true isosurface once (a real, legitimately
    steep transition -- not the bug), so the true crossing sample is excluded
    from the "no unexpected jump" assertion; everywhere else, adjacent-sample
    |delta field| must stay far below the old O(alpha) step size.
    """
    W, H = camera.image_width, camera.image_height
    assert camera.render_model == 2, "continuity_test assumes PH (pinhole) truck cameras"
    px, py = W // 2, H // 2  # a real pixel -- image center, expected to hit the truck

    # View-space ray direction (unnormalized, z=1) through this pixel's exact
    # center -- algebraic inverse of fisheye_proj.py's PH projection formula.
    dir_view = torch.tensor(
        [(px + 0.5 - W / 2.0) / camera.focal_x, (py + 0.5 - H / 2.0) / camera.focal_y, 1.0],
        device="cuda")

    def points_at_depths(depths: torch.Tensor) -> torch.Tensor:
        pts_view = dir_view.view(1, 3) * depths.view(-1, 1)
        return view_to_world(camera, pts_view)

    def max_jump_excluding_crossing(field: torch.Tensor, margin: int = 20):
        diffs = (field[1:] - field[:-1]).abs()
        sample_signs = torch.sign(field)
        cross = (sample_signs[:-1] * sample_signs[1:] < 0).nonzero(as_tuple=True)[0]
        keep = torch.ones_like(diffs, dtype=torch.bool)
        for c in cross.tolist():
            lo, hi = max(0, c - margin), min(diffs.shape[0], c + margin + 1)
            keep[lo:hi] = False
        excl_max = diffs[keep].max().item() if bool(keep.any()) else 0.0
        return excl_max, diffs.max().item(), cross.numel()

    # 1) Coarse depth scan along the central ray to locate a real surface
    # crossing (field sign change) to probe near.
    coarse_depths = torch.linspace(0.3, 20.0, 400, device="cuda")
    coarse_field = evaluate_occupancy_integrated(
        points_at_depths(coarse_depths), [camera], gaussians, pipe)
    coarse_signs = torch.sign(coarse_field)
    crossings = (coarse_signs[:-1] * coarse_signs[1:] < 0).nonzero(as_tuple=True)[0]
    assert crossings.numel() > 0, (
        "no surface crossing found along the image-center ray in [0.3,20]u -- "
        "pick a different pixel/camera for this probe")
    d_surf = 0.5 * (coarse_depths[crossings[0]] + coarse_depths[crossings[0] + 1]).item()

    # world-space size of one pixel at depth d_surf (PH: pixel angular size
    # ~= 1/focal_x rad); half-span = 1 pixel each side -> a ~2-pixel-wide
    # segment total, spanning adjacent pixels, for both segments below.
    pixel_world_size = d_surf / camera.focal_x
    span = pixel_world_size

    # 2) LATERAL segment: perpendicular to the ray, fixed depth = d_surf.
    right_view = torch.tensor([1.0, 0.0, 0.0], device="cuda")
    offsets = torch.linspace(-span, span, 1000, device="cuda")
    lateral_pts_view = dir_view.view(1, 3) * d_surf + offsets.view(-1, 1) * right_view.view(1, 3)
    lateral_field = evaluate_occupancy_integrated(
        view_to_world(camera, lateral_pts_view), [camera], gaussians, pipe)

    # 3) DEPTH segment: along the ray, spanning the same order-of-magnitude
    # range around d_surf.
    depths = torch.linspace(d_surf - span, d_surf + span, 1000, device="cuda")
    depth_field = evaluate_occupancy_integrated(points_at_depths(depths), [camera], gaussians, pipe)

    lateral_excl_max, lateral_raw_max, lateral_n_cross = max_jump_excluding_crossing(lateral_field)
    depth_excl_max, depth_raw_max, depth_n_cross = max_jump_excluding_crossing(depth_field)

    print(f"[continuity] pixel=({px},{py}) d_surf={d_surf:.4f}u pixel_world_size={pixel_world_size:.5f}u "
          f"span=+-{span:.5f}u")
    print(f"[continuity] LATERAL: max|delta field| excl. crossing = {lateral_excl_max:.6f} "
          f"(raw incl. crossing = {lateral_raw_max:.6f}, {lateral_n_cross} crossing(s) over 1000 samples)")
    print(f"[continuity] DEPTH:   max|delta field| excl. crossing = {depth_excl_max:.6f} "
          f"(raw incl. crossing = {depth_raw_max:.6f}, {depth_n_cross} crossing(s) over 1000 samples)")

    assert lateral_excl_max < 1e-3, f"lateral field discontinuity away from the true crossing: {lateral_excl_max}"
    assert depth_excl_max < 1e-3, f"depth field discontinuity away from the true crossing: {depth_excl_max}"
    print("PASS continuity: no staircase jumps > 1e-3 away from the true surface crossing "
          "(old pixel-snapped/hard-gate field showed O(alpha) jumps here)")


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
    continuity_test(pipe, gaussians, camera)

    if args.crosscheck:
        field_crosscheck(pipe, gaussians, scene.getTrainCameras())

    print("\nAll Session E2 integrate checks PASSED.")


if __name__ == "__main__":
    main()
