# ported/adapted from GaussianWrapping gaussian_wrapping/primal_adaptive_meshing_extraction.py
# (candidate-point generation / refine-and-filter / Delaunay tet-classification
# pipeline order) + gaussian_wrapping/utils/primal_adaptive_meshing_utils.py
# (MeshFromDelaunay, sample_mesh_proportional_to_camera -- see pam_utils.py).
"""CLI: Primal Adaptive Meshing (PAM), GaussianWrapping's second extraction
stage (GUTWrap Path A, Session E3). Takes an INPUT MESH (typically our own
pivot-based marching-tetrahedra output from extract_mesh.py) purely as a
*candidate-point generator*: it samples points on that mesh, pulls them onto
OUR occupancy field's isosurface with a gradient/Newton refinement, filters
out points that didn't converge, then rebuilds a (denser, better-placed)
surface via Delaunay tetrahedralization + tet-inside/outside classification.

Example (truck, PH mode, from the Session E2 integrated pivot-MTet mesh):
    python gaussian_wrapping/pam_extraction.py \\
        --model_path /home/output/sessD_full30k \\
        --source_path /home/data/tt/datasets/truck \\
        --dataset COLMAP --camera_model PINHOLE --render_model PH \\
        --iteration 30000 \\
        --input_mesh /home/output/sessE_mesh/truck_mtet_30k_integrated.ply \\
        --output_mesh /home/output/sessE_mesh/truck_pam.ply \\
        --bounding_box_method scene --max_points 1000000

DEVIATIONS from the on-disk GaussianWrapping implementation (Session E3 hard
rule: document semantic gaps in this header):

1. **Gradient field**: GW's Newton-step direction comes from their own
   analytic Gaussian-mixture log-density gradient (`GaussianVectorField` /
   `utils/primal_adaptive_meshing_utils.py::get_vector_field_quantities_aux`),
   which needs GW-specific `GaussianModel` machinery we don't have
   (`convert_features_to_normals`, `get_scaling_with_3D_filter`,
   `learn_occupancy` opacity-shift mode, ...). The task spec explicitly asks
   for refinement "using our field callable"; this port therefore replaces
   GW's analytic vector field with a **central-difference numerical gradient
   of our own field_fn** (`pam_utils.numerical_field_gradient`, same field_fn
   as `mesh_extraction.py --sdf_mode {integrated,exact}`). This costs 6 extra
   field evaluations per refinement step (vs. GW's ~0, since theirs is a
   closed-form KNN lookup) but requires zero new Gaussian-model machinery.
2. **Sign/threshold convention**: GW works in an "occupancy in [0,1],
   threshold ~0.5" convention with a separate `iso_surface_value` /
   `occupancy_threshold` algebra layer. This port instead works directly in
   OUR field's zero-centered convention (field > 0 = empty, < 0 = occupied,
   surface at field == 0, exactly as `mesh_extraction.py`), with
   `--iso_surface_value` as a direct offset on that field (and an optional
   `--auto_iso`, reusing `fields.sample_depth_surface_points`, exactly like
   `extract_mesh.py`). This avoids re-deriving GW's threshold-transform
   bookkeeping while being value-for-value equivalent.
3. **`--bounding_box_method ground_truth`** (GW's COLMAP_SfM.log-based crop
   volume) is explicitly out of scope per the task spec ("skip ground_truth").
4. **`--bounding_box_method scene`**: GW centers this AABB at the WORLD
   ORIGIN (`min_bound = [-scene_radius]*3`), which silently assumes a
   pre-centered/normalized scene. This port centers it at the camera-centroid
   instead (same convention `mesh_extraction.py`'s own radius-crop already
   uses for T&T `truck`, an un-normalized raw-COLMAP scene) -- centering at
   the true origin here would clip the geometry out entirely on truck.
5. Mesh IO / post-processing / "largest connected component" reuses
   `mesh_extraction.py`'s trimesh-based `_largest_connected_component`
   instead of GW's open3d `post_process_mesh` (open3d is not installed in
   the `geer` container; trimesh is already a dependency of this pipeline).
6. **`--mesh_sampling_method surface_even`** uses trimesh's own
   `trimesh.sample.sample_surface` instead of GW's
   `eval/TNTUniScanEvals/uniform_sampling_eval.sample_surface` (functionally
   equivalent area-weighted point+normal sampling; avoids porting the whole
   eval harness for one helper).
7. **View-cost control**: like `mesh_extraction.py --max_field_eval_sec`,
   the (many-call) Newton-refinement loop calibrates on a small point sample
   and, if a single field eval exceeds `--max_field_eval_sec`, subsamples
   views (every 2nd camera) for the refinement loop only; the final
   tet-classification occupancy pass always uses the full camera set.
8. `--no_force_use_all_points` (GW: skip the oversampling retry loop) is
   dropped -- this port always uses the oversample-and-retry loop (simpler,
   single code path; GW's flag existed only to skip it for speed).
"""
import copy
import json
import os
import sys
import time
from argparse import ArgumentParser

import numpy as np
import torch
import trimesh
from scipy.spatial import ConvexHull

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arguments import ModelParams, PipelineParams, OptimizationParams  # noqa: E402

from gaussian_wrapping.extract_mesh import build_scene  # noqa: E402
from gaussian_wrapping.fields import evaluate_occupancy_integrated, sample_depth_surface_points  # noqa: E402
from gaussian_wrapping.mesh_extraction import _largest_connected_component  # noqa: E402
from gaussian_wrapping.pam_utils import (  # noqa: E402
    MeshFromDelaunay, TorchMesh, numerical_field_gradient,
    sample_mesh_proportional_to_camera, sample_surface_even, plot_histogram,
)


# ---------------------------------------------------------------------------
# Bounding-volume loading / cropping
# ---------------------------------------------------------------------------

def _in_convex_hull(hull: ConvexHull, points: np.ndarray, tol: float = 0.0) -> np.ndarray:
    """True where `points` are inside (or within `tol` of) the convex hull's
    half-space representation (ported from GW primal_adaptive_meshing_
    extraction.py::_in_convex_hull)."""
    signed_dist = points @ hull.equations[:, :-1].T + hull.equations[:, -1]
    return signed_dist.max(axis=1) <= tol


def _load_blender_hull(json_path: str):
    print(f"[pam] loading Blender bounding volume from {json_path}")
    with open(json_path, "r") as f:
        data = json.load(f)
    if data.get("class_name") != "GaussianWrappingBoundingVolume":
        raise ValueError(
            f"Expected class_name 'GaussianWrappingBoundingVolume', got "
            f"'{data.get('class_name')}'. Make sure the JSON was exported "
            "with the GW Bounding Volume Blender add-on."
        )
    vertices = np.array(data["vertices"], dtype=np.float64)
    return ConvexHull(vertices)


def _crop_mesh_to_bbox(mesh: trimesh.Trimesh, bbox_method: str, bbox_file, bbox_scaling: float,
                        scene_center: np.ndarray, scene_radius: float):
    """Returns (cropped trimesh.Trimesh, crop_kind, crop_obj) where crop_kind
    is "aabb" (crop_obj = (min_bound, max_bound)) or "hull" (crop_obj =
    scipy ConvexHull), so `main()` can re-apply the SAME crop to sampled
    candidate points later (mirroring GW: mesh AND resampled points are both
    clipped to the bounding volume)."""
    if bbox_method == "scene":
        r = scene_radius * bbox_scaling
        min_bound = scene_center - r
        max_bound = scene_center + r
        verts = mesh.vertices
        inside = np.all((verts >= min_bound) & (verts <= max_bound), axis=1)
        face_mask = inside[mesh.faces].all(axis=1)
        cropped = mesh.copy()
        cropped.update_faces(face_mask)
        cropped.remove_unreferenced_vertices()
        return cropped, "aabb", (min_bound, max_bound)

    elif bbox_method == "blender":
        if bbox_file is None:
            raise ValueError("--bounding_box_file is required when --bounding_box_method=blender")
        hull = _load_blender_hull(bbox_file)
        inside = _in_convex_hull(hull, mesh.vertices.astype(np.float64))
        face_mask = inside[mesh.faces].all(axis=1)
        cropped = mesh.copy()
        cropped.update_faces(face_mask)
        cropped.remove_unreferenced_vertices()
        return cropped, "hull", hull

    else:
        raise ValueError(
            f"--bounding_box_method {bbox_method!r} not supported (choices: "
            "scene, blender; 'ground_truth' is explicitly out of scope, see "
            "pam_extraction.py module docstring)."
        )


def _crop_points_to_bbox(points: np.ndarray, crop_kind: str, crop_obj) -> np.ndarray:
    if crop_kind == "aabb":
        min_bound, max_bound = crop_obj
        keep = np.all((points >= min_bound) & (points <= max_bound), axis=1)
    else:
        keep = _in_convex_hull(crop_obj, points.astype(np.float64), tol=1e-6)
    return points[keep]


# ---------------------------------------------------------------------------
# Refinement / filtering (Newton step onto the field's zero level set)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _gradient_descent_refinement(points: np.ndarray, field_fn, grad_fn, n_steps: int,
                                  min_grad_norm: float, target: float = 0.0,
                                  chunk_size: int = 1_000_000, device: str = "cuda",
                                  plot_dir=None):
    """Damped Newton iteration pulling `points` toward field_fn(x) == target.

    delta = (target - field(x)) / |grad(x)|^2 * grad(x), x <- x + 0.5*clamp(delta_coeff,-1,1)*grad(x)
    (same damping/clamping constants as GW's gradient_descent_refinement,
    re-derived for OUR field's sign convention -- see module docstring
    deviation #1/#2). Points where |grad| is too small (no local field
    variation, e.g. far outside the object) are left in place.
    """
    xi = torch.from_numpy(points).float()

    for step in range(n_steps):
        xi_new_chunks = []
        last_field_values = None
        for xi_chunk in torch.chunk(xi, max(1, xi.shape[0] // chunk_size + 1)):
            xi_chunk = xi_chunk.to(device)
            grad = grad_fn(xi_chunk)
            field_values = field_fn(xi_chunk).unsqueeze(-1)
            norm_sq = grad.norm(dim=-1, keepdim=True) ** 2
            valid = (norm_sq > (min_grad_norm ** 2)).squeeze(-1)

            alpha = torch.zeros_like(field_values)
            alpha[valid] = ((target - field_values[valid]) / norm_sq[valid]).clamp(min=-1.0, max=1.0)
            new_chunk = xi_chunk + 0.5 * alpha * grad
            xi_new_chunks.append(new_chunk.cpu())
            last_field_values = field_values
        xi = torch.cat(xi_new_chunks, dim=0)

        if plot_dir is not None and last_field_values is not None:
            plot_histogram(last_field_values, filename=os.path.join(plot_dir, f"pam_field_hist_{step}.png"),
                            title="PAM refinement field values", iso_value=target)

    normal_chunks, norm_chunks = [], []
    for xi_chunk in torch.chunk(xi, max(1, xi.shape[0] // chunk_size + 1)):
        g = grad_fn(xi_chunk.to(device))
        norms = g.norm(dim=-1, keepdim=True)
        normal_chunks.append((g / (norms + 1e-8)).cpu())
        norm_chunks.append(norms.cpu())

    refined_normals = torch.cat(normal_chunks, dim=0).numpy()
    grad_norms = torch.cat(norm_chunks, dim=0)
    return xi.numpy(), refined_normals, grad_norms


@torch.no_grad()
def _filter_points_by_field(points: np.ndarray, field_fn, target: float, threshold: float,
                             chunk_size: int = 1_000_000, device: str = "cuda"):
    if points.shape[0] == 0:
        return points, np.zeros(0, dtype=bool)
    field_chunks = []
    for chunk in torch.chunk(torch.from_numpy(points).float(), max(1, points.shape[0] // chunk_size + 1)):
        field_chunks.append(field_fn(chunk.to(device)).cpu())
    field_values = torch.cat(field_chunks, dim=0)
    mask = (field_values - target).abs() <= threshold
    return points[mask.numpy()], mask.numpy()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = ArgumentParser(description="GUTWrap Session E3 -- Primal Adaptive Meshing (PAM)")
    parser.add_argument("--model_path", "-m", required=True, type=str)
    parser.add_argument("--source_path", "-s", required=True, type=str)
    parser.add_argument("--iteration", type=int, default=30000)
    parser.add_argument("--dataset", type=str, default="COLMAP",
                         choices=["AUTO", "COLMAP", "BLENDER", "SCANNETPP", "MVL"])
    parser.add_argument("--camera_model", type=str, default="PINHOLE", choices=["FISHEYE", "PINHOLE"])
    parser.add_argument("--render_model", type=str, default="PH", choices=["BEAP", "KB", "EQ", "PH"])
    parser.add_argument("--sample_step", type=float, default=0.002)
    parser.add_argument("--focal_scaling", type=float, default=1.0)
    parser.add_argument("--distortion_scaling", type=float, default=1.0)
    parser.add_argument("--mirror_shift", type=float, default=0.0)
    parser.add_argument("--raymap_path", type=str, default=None)

    parser.add_argument("--input_mesh", type=str, required=True)
    parser.add_argument("--output_mesh", type=str, required=True)

    parser.add_argument("--sdf_mode", type=str, default="integrated", choices=["integrated", "exact"],
                         help="Field driving PAM's refinement/classification -- SAME field_fn as "
                              "extract_mesh.py --sdf_mode (evaluate_occupancy_integrated).")
    parser.add_argument("--iso_surface_value", type=float, default=0.0,
                         help="Offset on OUR field's zero-centered convention (field>0 empty, "
                              "field<0 occupied, surface at 0) -- see module docstring deviation #2.")
    parser.add_argument("--auto_iso", action="store_true",
                         help="Same recipe as extract_mesh.py --auto_iso; overrides --iso_surface_value.")
    parser.add_argument("--auto_iso_n_points", type=int, default=1_000_000)
    parser.add_argument("--auto_iso_reduction", type=str, default="median", choices=["median", "mean"])
    parser.add_argument("--field_chunk", type=int, default=2_000_000)
    parser.add_argument("--max_field_eval_sec", type=float, default=180.0,
                         help="If a calibration field eval exceeds this, the (many-call) refinement "
                              "loop subsamples views (every 2nd camera); final tet classification "
                              "always uses the full camera set.")
    parser.add_argument("--grad_eps", type=float, default=None,
                         help="Absolute world-space step for the central-difference field gradient "
                              "used by refinement (default: 0.002 * scene_radius).")
    parser.add_argument("--min_grad_norm", type=float, default=1e-4,
                         help="Points with |grad field| below this are left in place during refinement "
                              "(no reliable local surface direction).")

    parser.add_argument("--max_points", type=int, default=1_000_000, help="Target number of final candidate points")
    parser.add_argument("--p_per_tet", type=int, default=10, help="Points per tet for occupancy check (1 = tet barycenter)")
    parser.add_argument("--n_steps", type=int, default=10, help="Newton refinement steps")
    parser.add_argument("--vacancy_threshold", type=float, default=0.1, help="|field - iso| filter threshold")
    parser.add_argument("--mesh_sampling_method", type=str, default="proportional_to_camera",
                         choices=["surface_even", "proportional_to_camera"])
    parser.add_argument("--oversampling_factor", type=int, default=2,
                         help="Sample this multiple of max_points upfront per resample round")
    parser.add_argument("--max_resample_rounds", type=int, default=20,
                         help="Safety cap on the oversample-and-retry loop.")
    parser.add_argument("--bounding_box_method", type=str, default="scene", choices=["scene", "blender"],
                         help="'ground_truth' is explicitly out of scope, see module docstring.")
    parser.add_argument("--bounding_box_scaling", type=float, default=1.0)
    parser.add_argument("--bounding_box_file", type=str, default=None,
                         help="Blender bounding-volume JSON (required for --bounding_box_method=blender)")
    parser.add_argument("--post_process", action="store_true", help="Keep only the largest connected component")
    parser.add_argument("--save_candidate_points", action="store_true")
    parser.add_argument("--plot_vacancy_histogram", action="store_true")

    args = parser.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.output_mesh)), exist_ok=True)
    stats = {}
    t_start = time.time()

    # -- 1. scene / cameras --------------------------------------------------
    dataset, opt, pipe, gaussians, scene = build_scene(
        source_path=args.source_path, model_path=args.model_path, dataset_type=args.dataset,
        camera_model=args.camera_model, iteration=args.iteration, render_model=args.render_model,
        sample_step=args.sample_step, focal_scaling=args.focal_scaling,
        distortion_scaling=args.distortion_scaling, mirror_shift=args.mirror_shift,
        raymap_path=args.raymap_path,
    )
    cameras = scene.getTrainCameras()
    print(f"[pam] loaded {gaussians.get_xyz.shape[0]} Gaussians, {len(cameras)} cameras "
          f"from {args.model_path} @ iteration {args.iteration}")

    cam_centers = torch.stack([c.camera_center for c in cameras], dim=0)
    avg_center = cam_centers.mean(dim=0, keepdim=True)
    scene_radius = (cam_centers - avg_center).norm(dim=-1).max().item() * 1.1
    stats["scene_radius"] = scene_radius
    print(f"[pam] scene_radius = {scene_radius:.3f}")

    occ_mode = "exact" if args.sdf_mode == "exact" else "integrated"

    def _make_field_fn(iso_value, camera_list):
        return lambda pts: evaluate_occupancy_integrated(
            pts, camera_list, gaussians, pipe, iso=iso_value, chunk=args.field_chunk, mode=occ_mode)

    field_fn_iso0 = _make_field_fn(0.0, cameras)

    # -- 2. calibrate + optional view-subsampling for the refinement loop ---
    t0 = time.time()
    calib_pts = gaussians.get_xyz.detach()[:min(50_000, gaussians.get_xyz.shape[0])]
    _ = field_fn_iso0(calib_pts)
    stats["t_field_eval_calibration_sec"] = time.time() - t0
    if stats["t_field_eval_calibration_sec"] > args.max_field_eval_sec and len(cameras) > 4:
        refine_cameras = cameras[::2]
        stats["refine_view_subsampled"] = True
        print(f"[pam] field eval took {stats['t_field_eval_calibration_sec']:.1f}s > "
              f"{args.max_field_eval_sec:.0f}s budget -- subsampling views "
              f"({len(refine_cameras)}/{len(cameras)}) for the refinement loop")
    else:
        refine_cameras = cameras
        stats["refine_view_subsampled"] = False

    # -- 3. iso (manual or auto) ---------------------------------------------
    iso = args.iso_surface_value
    if args.auto_iso:
        t0 = time.time()
        surface_pts = sample_depth_surface_points(cameras, gaussians, pipe, n_points=args.auto_iso_n_points)
        raw = field_fn_iso0(surface_pts)
        sdf_isosurface_value = raw.median().item() if args.auto_iso_reduction == "median" else raw.mean().item()
        iso = -sdf_isosurface_value
        stats["t_auto_iso_sec"] = time.time() - t0
        stats["auto_iso_sdf_isosurface_value"] = sdf_isosurface_value
        print(f"[pam] auto_iso: {args.auto_iso_reduction} raw field at "
              f"{surface_pts.shape[0]} surface points = {sdf_isosurface_value:.6f} -> iso = {iso:.6f}")
    stats["iso"] = iso

    field_fn = _make_field_fn(iso, cameras)                 # full-camera, for final occupancy/classification
    field_fn_refine = _make_field_fn(iso, refine_cameras)   # possibly view-subsampled, for the Newton loop

    grad_eps = args.grad_eps if args.grad_eps is not None else 0.002 * scene_radius
    stats["grad_eps"] = grad_eps
    grad_fn_refine = lambda pts: numerical_field_gradient(field_fn_refine, pts, eps=grad_eps)  # noqa: E731

    # -- 4. load + crop input mesh -------------------------------------------
    t0 = time.time()
    if not os.path.exists(args.input_mesh):
        raise FileNotFoundError(f"Input mesh not found: {args.input_mesh}")
    mesh_raw = trimesh.load(args.input_mesh, process=False)
    stats["n_input_verts"] = int(mesh_raw.vertices.shape[0])
    stats["n_input_faces"] = int(mesh_raw.faces.shape[0])

    mesh_cropped, crop_kind, crop_obj = _crop_mesh_to_bbox(
        mesh_raw, args.bounding_box_method, args.bounding_box_file, args.bounding_box_scaling,
        scene_center=avg_center.squeeze(0).cpu().numpy().astype(np.float64), scene_radius=scene_radius,
    )
    stats["n_cropped_verts"] = int(mesh_cropped.vertices.shape[0])
    stats["n_cropped_faces"] = int(mesh_cropped.faces.shape[0])
    stats["t_load_crop_mesh_sec"] = time.time() - t0
    print(f"[pam] input mesh: {stats['n_input_verts']} v / {stats['n_input_faces']} f -> "
          f"cropped {stats['n_cropped_verts']} v / {stats['n_cropped_faces']} f "
          f"({stats['t_load_crop_mesh_sec']:.1f}s)")
    if stats["n_cropped_faces"] == 0:
        raise RuntimeError(
            "Cropped input mesh is empty -- check --bounding_box_method/--bounding_box_scaling "
            "(scene_radius may not match this mesh's extent)."
        )

    torch_mesh = TorchMesh(
        verts=torch.as_tensor(np.asarray(mesh_cropped.vertices), dtype=torch.float32, device="cuda"),
        faces=torch.as_tensor(np.asarray(mesh_cropped.faces), dtype=torch.long, device="cuda"),
    )

    # -- 5. candidate-point generation: sample -> refine -> filter, retry ---
    plot_dir = args.model_path if args.plot_vacancy_histogram else None

    def _sample(n_sample, face_probs):
        if args.mesh_sampling_method == "surface_even":
            pts, _ = sample_surface_even(mesh_cropped, n_sample)
            return pts, None
        else:
            pts, _, fp = sample_mesh_proportional_to_camera(torch_mesh, cameras, n_sample, face_probs=face_probs)
            return pts, fp

    def _refine_filter(pts):
        refined, _, _ = _gradient_descent_refinement(
            pts, field_fn_refine, grad_fn_refine, n_steps=args.n_steps,
            min_grad_norm=args.min_grad_norm, target=0.0, plot_dir=plot_dir,
        )
        filtered, _ = _filter_points_by_field(refined, field_fn, target=0.0, threshold=args.vacancy_threshold)
        return filtered

    t0 = time.time()
    n_sample = args.max_points * args.oversampling_factor
    sampled_pts, face_probs = _sample(n_sample, None)
    all_points = []
    points_to_extract = args.max_points
    round_i = 0
    while points_to_extract > 0 and round_i < args.max_resample_rounds:
        print(f"[pam] refine/filter round {round_i}: {sampled_pts.shape[0]} candidate points "
              f"(need {points_to_extract} more)")
        curr = _refine_filter(sampled_pts)
        all_points.append(curr)
        points_to_extract -= curr.shape[0]
        round_i += 1
        if points_to_extract > 0:
            n_sample = points_to_extract * args.oversampling_factor
            sampled_pts, face_probs = _sample(n_sample, face_probs)
    points = np.concatenate(all_points, axis=0)[:args.max_points] if all_points else np.zeros((0, 3))
    points = _crop_points_to_bbox(points, crop_kind, crop_obj)
    stats["t_candidate_points_sec"] = time.time() - t0
    stats["n_resample_rounds"] = round_i
    stats["n_candidate_points"] = int(points.shape[0])
    print(f"[pam] {stats['n_candidate_points']} candidate points after {round_i} round(s), "
          f"{stats['t_candidate_points_sec']:.1f}s")

    if args.save_candidate_points:
        cand_path = args.output_mesh.replace(".ply", "_candidate_points.ply")
        trimesh.points.PointCloud(points).export(cand_path)
        print(f"[pam] wrote candidate points to {cand_path}")

    if points.shape[0] < 4:
        raise RuntimeError(f"Only {points.shape[0]} candidate points survived -- cannot tetrahedralize.")

    # -- 6. Delaunay + tet classification + boundary-triangle extraction ----
    t0 = time.time()
    mfd = MeshFromDelaunay(points, add_corners=False)
    stats["n_tets"] = int(mfd.simplices.shape[0])
    stats["t_delaunay_sec"] = time.time() - t0
    print(f"[pam] Delaunay: {stats['n_tets']} tets ({stats['t_delaunay_sec']:.1f}s)")

    t0 = time.time()
    if args.p_per_tet == 1:
        query_points = mfd.barycenters
    else:
        query_points = mfd.sample_random_tet_points(args.p_per_tet).reshape(-1, 3)
    torch_query_points = torch.tensor(query_points, dtype=torch.float32, device="cuda")

    field_chunks = []
    for chunk in torch.chunk(torch_query_points, max(1, torch_query_points.shape[0] // 2_000_000 + 1)):
        field_chunks.append(field_fn(chunk).cpu())
    field_values = torch.cat(field_chunks, dim=0)

    if args.p_per_tet > 1:
        field_values = field_values.reshape(args.p_per_tet, -1).mean(0)
    mfd.tet_colors = (field_values.numpy() < 0.0)  # field<0 == occupied (our sign convention)
    stats["t_tet_occupancy_sec"] = time.time() - t0
    stats["frac_tets_occupied"] = float(mfd.tet_colors.mean())
    print(f"[pam] tet occupancy evaluated on {torch_query_points.shape[0]} points "
          f"({stats['t_tet_occupancy_sec']:.1f}s); {stats['frac_tets_occupied']:.4f} fraction occupied")

    t0 = time.time()
    surface_verts, surface_faces = mfd.get_surface()
    stats["t_extract_surface_sec"] = time.time() - t0
    stats["n_raw_pam_vertices"] = int(surface_verts.shape[0])
    stats["n_raw_pam_triangles"] = int(surface_faces.shape[0])
    print(f"[pam] boundary surface: {stats['n_raw_pam_vertices']} v / {stats['n_raw_pam_triangles']} f "
          f"({stats['t_extract_surface_sec']:.1f}s)")

    # -- 7. post-process + export --------------------------------------------
    t0 = time.time()
    if args.post_process and surface_faces.shape[0] > 0:
        largest_mesh, n_components, n_largest_faces = _largest_connected_component(surface_verts, surface_faces)
        stats["n_components"] = n_components
        stats["n_largest_component_triangles"] = n_largest_faces
        out_mesh = largest_mesh
    else:
        out_mesh = trimesh.Trimesh(vertices=surface_verts, faces=surface_faces, process=False)
    stats["t_postprocess_sec"] = time.time() - t0
    stats["final_n_vertices"] = int(out_mesh.vertices.shape[0])
    stats["final_n_triangles"] = int(out_mesh.faces.shape[0])

    out_mesh.export(args.output_mesh)
    stats["output_path"] = args.output_mesh
    stats["t_total_sec"] = time.time() - t_start
    print(f"[pam] wrote {args.output_mesh} ({stats['final_n_vertices']} v / "
          f"{stats['final_n_triangles']} f) total {stats['t_total_sec']:.1f}s")

    stats_path = os.path.splitext(os.path.abspath(args.output_mesh))[0] + "_stats.txt"
    with open(stats_path, "w") as f:
        for k, v in stats.items():
            f.write(f"{k}: {v}\n")


if __name__ == "__main__":
    with torch.no_grad():
        main()
