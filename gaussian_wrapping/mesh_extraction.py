# ported/adapted from GaussianWrapping gaussian_wrapping/utils/tetmesh.py
# (marching tetrahedra, itself adapted by GW from NVIDIA kaolin) and
# gaussian_wrapping/pivot_based_mesh_extraction.py (pipeline order,
# post_process_mesh).
"""Pivot-based marching-tetrahedra mesh extraction (GaussianWrapping's
primary mesh-extraction method), ported onto 3DGEER's median-depth TSDF
field (gaussian_wrapping/fields.py) instead of GW's own renderer stack.

Pipeline (see extract_mesh_pivot_mtet):
  1. pivots <- gaussian_wrapping.pivots.extract_gaussian_pivots
  2. tsdf at pivots <- gaussian_wrapping.fields.fuse_tsdf_at_points
  3. scipy.spatial.Delaunay tetrahedralization of the pivots (CPU/numpy;
     everything else stays in torch)
  4. marching tetrahedra at iso 0 (ported from GW's utils/tetmesh.py,
     unbatched -- 3DGEER never needs the batch dimension)
  5. 8-step binary-search refinement of each crossing vertex, re-evaluating
     fuse_tsdf_at_points on candidates along the crossing edge
  6. drop degenerate triangles (repeated vertex index / near-zero area)
  7. largest-connected-component filtering (trimesh, not open3d --
     lighter dependency, per Session E hard rules)
  8. write PLY (trimesh)
"""
import time
from typing import Optional

import numpy as np
import torch
import trimesh
from scipy.spatial import Delaunay

from gaussian_wrapping.pivots import extract_gaussian_pivots, get_searched_pivots
from gaussian_wrapping.fields import render_depth_maps, fuse_tsdf_at_points, evaluate_occupancy_integrated

# ---------------------------------------------------------------------------
# Marching tetrahedra (ported from GaussianWrapping utils/tetmesh.py, which is
# itself adapted from NVIDIA kaolin's ops.conversions.tetmesh). Unbatched:
# 3DGEER always has a single "batch" (one scene), so the list-of-batches
# wrapper in the original is dropped.
# ---------------------------------------------------------------------------

_triangle_table = torch.tensor([
    [-1, -1, -1, -1, -1, -1],
    [1, 0, 2, -1, -1, -1],
    [4, 0, 3, -1, -1, -1],
    [1, 4, 2, 1, 3, 4],
    [3, 1, 5, -1, -1, -1],
    [2, 3, 0, 2, 5, 3],
    [1, 4, 0, 1, 5, 4],
    [4, 2, 5, -1, -1, -1],
    [4, 5, 2, -1, -1, -1],
    [4, 1, 0, 4, 5, 1],
    [3, 2, 0, 3, 5, 2],
    [1, 3, 5, -1, -1, -1],
    [4, 1, 2, 4, 3, 1],
    [3, 0, 4, -1, -1, -1],
    [2, 0, 1, -1, -1, -1],
    [-1, -1, -1, -1, -1, -1],
], dtype=torch.long)

_num_triangles_table = torch.tensor(
    [0, 1, 1, 2, 1, 2, 2, 1, 1, 2, 2, 1, 2, 1, 1, 0], dtype=torch.long)
_base_tet_edges = torch.tensor([0, 1, 0, 2, 0, 3, 1, 2, 1, 3, 2, 3], dtype=torch.long)
_v_id = torch.pow(2, torch.arange(4, dtype=torch.long))


def marching_tetrahedra(
    vertices: torch.Tensor,
    tets: torch.Tensor,
    sdf: torch.Tensor,
    scales: Optional[torch.Tensor] = None,
    chunk_size: int = 32 * 1024 * 1024,
):
    """Unbatched marching tetrahedra.

    Args:
        vertices (torch.Tensor): (P, 3) pivot positions.
        tets (torch.Tensor): (T, 4) long tetrahedron vertex indices.
        sdf (torch.Tensor): (P,) signed field values (surface at 0).
        scales (torch.Tensor, optional): (P, 1) per-pivot scale, carried
            through to the output edges for optional large-edge filtering.
            If None, a dummy all-ones tensor is used.

    Returns:
        end_points (torch.Tensor): (N_verts, 2, 3) the two pivot positions
            bracketing each crossing edge.
        end_sdf (torch.Tensor): (N_verts, 2, 1) sdf at those two pivots.
        end_scales (torch.Tensor): (N_verts, 2, 1) scale at those two pivots.
        faces (torch.Tensor): (N_faces, 3) long triangle indices into the
            N_verts crossing-edge array.
        edge_pivot_idx (torch.Tensor): (N_verts, 2) long indices into the
            original `vertices`/`sdf` arrays for each crossing-edge endpoint.
    """
    device = vertices.device
    if scales is None:
        scales = torch.ones(vertices.shape[0], 1, device=device)

    if tets.shape[0] > chunk_size:
        merged = None
        for tet_chunk in torch.chunk(tets, tets.shape[0] // chunk_size + 1):
            torch.cuda.empty_cache()
            chunk_res = marching_tetrahedra(vertices, tet_chunk, sdf, scales, chunk_size)
            if merged is None:
                merged = chunk_res
            else:
                merged = _merge_mtet_chunks(merged, chunk_res)
        return merged

    with torch.no_grad():
        occ_n = sdf > 0
        occ_fx4 = occ_n[tets.reshape(-1)].reshape(-1, 4)
        occ_sum = torch.sum(occ_fx4, -1)

        valid_tets = (occ_sum > 0) & (occ_sum < 4)

        all_edges = tets[valid_tets][:, _base_tet_edges.to(device)].reshape(-1, 2)
        order = (all_edges[:, 0] > all_edges[:, 1]).bool()
        all_edges[order] = all_edges[order][:, [1, 0]]

        unique_edges, idx_map = torch.unique(all_edges, dim=0, return_inverse=True)
        unique_edges = unique_edges.long()
        mask_edges = occ_n[unique_edges.reshape(-1)].reshape(-1, 2).sum(-1) == 1
        mapping = torch.ones((unique_edges.shape[0],), dtype=torch.long, device=device) * -1
        mapping[mask_edges] = torch.arange(mask_edges.sum(), dtype=torch.long, device=device)
        idx_map = mapping[idx_map]

        interp_v = unique_edges[mask_edges]  # (N_verts, 2) pivot indices

    end_points = vertices[interp_v.reshape(-1)].reshape(-1, 2, 3)
    end_sdf = sdf[interp_v.reshape(-1)].reshape(-1, 2, 1)
    end_scales = scales[interp_v.reshape(-1)].reshape(-1, 2, 1)

    idx_map = idx_map.reshape(-1, 6)
    tetindex = (occ_fx4[valid_tets] * _v_id.to(device).unsqueeze(0)).sum(-1)
    num_triangles = _num_triangles_table.to(device)[tetindex]
    triangle_table_device = _triangle_table.to(device)

    faces = torch.cat((
        torch.gather(input=idx_map[num_triangles == 1], dim=1,
                     index=triangle_table_device[tetindex[num_triangles == 1]][:, :3]).reshape(-1, 3),
        torch.gather(input=idx_map[num_triangles == 2], dim=1,
                     index=triangle_table_device[tetindex[num_triangles == 2]][:, :6]).reshape(-1, 3),
    ), dim=0)

    return end_points, end_sdf, end_scales, faces, interp_v


def _merge_mtet_chunks(a, b):
    end_points_a, end_sdf_a, end_scales_a, faces_a, interp_v_a = a
    end_points_b, end_sdf_b, end_scales_b, faces_b, interp_v_b = b
    device = end_points_a.device

    all_edges = torch.cat([interp_v_a, interp_v_b], dim=0)
    unique_edges, idx_map = torch.unique(all_edges, dim=0, return_inverse=True)

    n_a = end_points_a.shape[0]
    merged_points = torch.zeros((unique_edges.shape[0], 2, 3), device=device)
    merged_sdf = torch.zeros((unique_edges.shape[0], 2, 1), device=device)
    merged_scales = torch.zeros((unique_edges.shape[0], 2, 1), device=device)
    merged_points[idx_map[:n_a]] = end_points_a
    merged_points[idx_map[n_a:]] = end_points_b
    merged_sdf[idx_map[:n_a]] = end_sdf_a
    merged_sdf[idx_map[n_a:]] = end_sdf_b
    merged_scales[idx_map[:n_a]] = end_scales_a
    merged_scales[idx_map[n_a:]] = end_scales_b

    faces_a_remapped = idx_map[faces_a.reshape(-1)].reshape(-1, 3)
    faces_b_remapped = idx_map[(faces_b.reshape(-1) + n_a)].reshape(-1, 3)
    merged_faces = torch.cat([faces_a_remapped, faces_b_remapped], dim=0)

    return merged_points, merged_sdf, merged_scales, merged_faces, unique_edges


# ---------------------------------------------------------------------------
# Binary-search refinement + degenerate-triangle removal + CC filtering
# ---------------------------------------------------------------------------

@torch.no_grad()
def _binary_search_refine(end_points, end_sdf, field_fn, n_steps=8):
    """n_steps-step bisection of each crossing edge, re-evaluating the field
    (`field_fn`, either `fuse_tsdf_at_points` or
    `evaluate_occupancy_integrated`, both (M,3) -> (M,)) at the midpoint
    instead of trusting the pivot-level linear interpolation. end_points:
    (Nv,2,3); end_sdf: (Nv,2,1)."""
    lo = end_points[:, 0, :].clone()
    hi = end_points[:, 1, :].clone()
    lo_sdf = end_sdf[:, 0, 0].clone()
    hi_sdf = end_sdf[:, 1, 0].clone()

    for _ in range(n_steps):
        mid = 0.5 * (lo + hi)
        mid_sdf = field_fn(mid)
        same_as_lo = (mid_sdf * lo_sdf) >= 0
        lo = torch.where(same_as_lo.unsqueeze(-1), mid, lo)
        lo_sdf = torch.where(same_as_lo, mid_sdf, lo_sdf)
        hi = torch.where(~same_as_lo.unsqueeze(-1), mid, hi)
        hi_sdf = torch.where(~same_as_lo, mid_sdf, hi_sdf)

    denom = (lo_sdf.abs() + hi_sdf.abs()).clamp(min=1e-8)
    t = (lo_sdf.abs() / denom).unsqueeze(-1)
    verts = lo + t * (hi - lo)
    return verts


def _large_edge_vertex_mask(end_points: torch.Tensor, end_scales: torch.Tensor) -> torch.Tensor:
    """GaussianWrapping's `filter_large_edges` safeguard (pivot_based_mesh_
    extraction.py / functional/pivots.py::extract_mesh): a crossing edge
    between two pivots is only trusted if the distance between the pivots is
    no larger than the sum of their (per-Gaussian) scales. Without this,
    Delaunay tetrahedra connecting far-apart outlier pivots (e.g. background/
    sky Gaussians with large scale in an unbounded outdoor scene) can flip
    TSDF sign across a huge tet and inject spurious triangles spanning the
    whole scene into the "largest connected component" -- the LichtFeld TSDF
    failure mode this pipeline is explicitly checked against (Session E
    Verify step 3 sanity gates)."""
    edge_len = (end_points[:, 0, :] - end_points[:, 1, :]).norm(dim=-1)
    edge_scale_sum = end_scales[:, 0, 0] + end_scales[:, 1, 0]
    return edge_len <= edge_scale_sum


def _drop_degenerate_triangles(verts: torch.Tensor, faces: torch.Tensor, area_eps: float = 1e-12):
    """Drop faces with a repeated vertex index or near-zero area."""
    v0, v1, v2 = faces[:, 0], faces[:, 1], faces[:, 2]
    not_repeated = (v0 != v1) & (v1 != v2) & (v0 != v2)

    p0, p1, p2 = verts[v0], verts[v1], verts[v2]
    cross = torch.cross(p1 - p0, p2 - p0, dim=-1)
    area = 0.5 * cross.norm(dim=-1)
    non_degenerate = not_repeated & (area > area_eps)

    return faces[non_degenerate]


def _largest_connected_component(verts: np.ndarray, faces: np.ndarray) -> trimesh.Trimesh:
    """Keep only the largest connected component (by face count), mirroring
    GaussianWrapping's pivot_based_mesh_extraction.py::post_process_mesh, but
    with trimesh instead of open3d (lighter dependency)."""
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    components = mesh.split(only_watertight=False)
    if len(components) == 0:
        return mesh, 0, 0
    largest = max(components, key=lambda c: len(c.faces))
    return largest, len(components), len(largest.faces)


# ---------------------------------------------------------------------------
# Top-level pipeline
# ---------------------------------------------------------------------------

def extract_mesh_pivot_mtet(
    gaussians,
    cameras,
    pipe,
    output_path: str,
    iso: float = 0.0,
    std_factor: float = 3.33,
    max_pivots: Optional[int] = 1_500_000,
    trunc_margin: Optional[float] = None,
    n_binary_steps: int = 10,
    max_radius_factor: Optional[float] = 2.0,
    sdf_mode: str = "integrated",
    use_searched_pivots: Optional[bool] = None,
    search_iter: int = 5,
    search_step_size: float = 0.33,
    field_chunk: Optional[int] = None,
    max_field_eval_sec: float = 180.0,
    stats_path: Optional[str] = None,
):
    """Full pivot-based marching-tetrahedra pipeline. Returns a dict of
    pipeline statistics (also written to `stats_path` if given).

    Args (Session E2 additions):
        sdf_mode: "integrated" (GW's actual quality path -- exact
            ray-integrated occupancy via the geer-rasterizer's forward-only
            `integrate_points` CUDA kernel, see gaussian_wrapping/fields.py::
            evaluate_occupancy_integrated) or "tsdf" (Session E's
            depth-fusion field, kept as the fast preview mode).
        use_searched_pivots: if True, pivots are refined via
            `pivots.get_searched_pivots` (walks the front pivot outward
            along the normal until it crosses the surface) instead of the
            fixed `std_factor` offset. Defaults to True for "integrated"
            (where the extra field evals are affordable/worthwhile) and
            False for "tsdf" (where fuse_tsdf_at_points is cheap enough
            that the fixed offset is not the bottleneck, and GW itself only
            pairs searched pivots with the exact field).
        field_chunk: point-chunk size passed to the field evaluator (None
            uses each field function's own default: 500k for tsdf, 2M for
            integrated).
        max_field_eval_sec: if a calibration field eval (integrated mode
            only) exceeds this, subsequent search/refine-stage field evals
            use every 2nd camera to bound wall time (per
            3DGEERGW_EXECUTION.md Session E2 End-to-end verification note).
            The final pivot-SDF eval (which fixes mesh topology) and the
            "tsdf" mode are unaffected.
    """
    stats = {}
    t_start = time.time()

    # -- scene extent / trunc_margin (computed first: needed for the pivot
    #    radius crop below) -------------------------------------------------
    cam_centers = torch.stack([c.camera_center for c in cameras], dim=0)
    avg_center = cam_centers.mean(dim=0, keepdim=True)
    scene_radius = (cam_centers - avg_center).norm(dim=-1).max().item() * 1.1
    stats["scene_radius"] = scene_radius

    # -- 1. Field callable + pivots ------------------------------------------
    t0 = time.time()
    means = gaussians.get_xyz.detach()
    scales = gaussians.get_scaling.detach()
    rotations = gaussians.get_rotation.detach()
    normals = gaussians.get_normals.detach()
    opacities = gaussians.get_opacity.detach().view(-1)

    if trunc_margin is None:
        gw_margin = 2e-3 * scene_radius
        pivot_margin = 5.0 * scales.mean().item()
        trunc_margin = max(gw_margin, pivot_margin)
    stats["trunc_margin"] = trunc_margin

    stats["n_gaussians"] = int(means.shape[0])
    # Pre-crop the Gaussians by the same radius BEFORE opacity subsampling:
    # otherwise `max_pivots` fills its budget with far-field (sky/background)
    # Gaussians that step 1b would drop anyway, starving the in-scene mesh.
    if max_radius_factor is not None:
        g_keep = (means - avg_center.to(means.device)).norm(dim=-1) \
                 <= max_radius_factor * scene_radius
        means, scales, rotations = means[g_keep], scales[g_keep], rotations[g_keep]
        normals, opacities = normals[g_keep], opacities[g_keep]
        stats["n_gaussians_in_radius"] = int(means.shape[0])

    # -- field callable (used for pivot evaluation AND binary-search
    #    refinement below) -----------------------------------------------
    if sdf_mode == "tsdf":
        t0 = time.time()
        depth_maps = render_depth_maps(cameras, gaussians, pipe)
        stats["t_render_depth_sec"] = time.time() - t0
        print(f"[mesh_extraction] rendered {len(depth_maps)} depth maps in "
              f"{stats['t_render_depth_sec']:.1f}s")
        tsdf_chunk = field_chunk if field_chunk is not None else 500_000
        field_fn = lambda pts: fuse_tsdf_at_points(pts, cameras, depth_maps, trunc_margin, chunk_size=tsdf_chunk)
        field_fn_refine = field_fn
        if use_searched_pivots is None:
            use_searched_pivots = False
    elif sdf_mode == "integrated":
        integrated_chunk = field_chunk if field_chunk is not None else 2_000_000
        field_fn = lambda pts: evaluate_occupancy_integrated(
            pts, cameras, gaussians, pipe, iso=0.0, chunk=integrated_chunk)
        if use_searched_pivots is None:
            use_searched_pivots = True

        # Calibrate on a modest, representative sample (post radius-crop
        # Gaussian centers) to decide whether the (many-call) search/refine
        # stages should subsample views to stay within budget. The final
        # pivot-SDF eval below always uses the full camera set.
        t0 = time.time()
        calib_n = min(50_000, means.shape[0])
        _ = field_fn(means[:calib_n])
        stats["t_field_eval_calibration_sec"] = time.time() - t0
        if stats["t_field_eval_calibration_sec"] > max_field_eval_sec and len(cameras) > 4:
            refine_cameras = cameras[::2]
            field_fn_refine = lambda pts: evaluate_occupancy_integrated(
                pts, refine_cameras, gaussians, pipe, iso=0.0, chunk=integrated_chunk)
            stats["refine_view_subsampled"] = True
            print(f"[mesh_extraction] field eval on {calib_n} pts took "
                  f"{stats['t_field_eval_calibration_sec']:.1f}s > {max_field_eval_sec:.0f}s budget -- "
                  f"subsampling views ({len(refine_cameras)}/{len(cameras)}) for search/refine stages")
        else:
            field_fn_refine = field_fn
            stats["refine_view_subsampled"] = False
    else:
        raise ValueError(f"Unknown sdf_mode: {sdf_mode!r} (expected 'integrated' or 'tsdf')")
    stats["sdf_mode"] = sdf_mode
    stats["use_searched_pivots"] = use_searched_pivots

    if use_searched_pivots:
        pivots, pivot_scales = get_searched_pivots(
            means, scales, rotations, normals, field_fn_refine,
            opacities=opacities, std_factor=std_factor, max_pivots=max_pivots,
            search_iter=search_iter, step_size=search_step_size,
        )
    else:
        pivots, pivot_scales = extract_gaussian_pivots(
            means, scales, rotations, normals,
            opacities=opacities, std_factor=std_factor, max_pivots=max_pivots,
        )
    stats["n_pivots_raw"] = int(pivots.shape[0])

    # -- 1b. Radius crop (NOT in the original Session E spec; added to pass
    #    the "final mesh bbox within ~2x the camera bbox" sanity gate -- see
    #    deviations in the Session E report). T&T `truck` is an unbounded
    #    outdoor COLMAP scene: ~20% of Gaussians sit beyond 3x the camera
    #    trajectory radius (some >300 units out, scale up to ~109), which is
    #    normal background/sky modeling in vanilla 3DGS but explodes the
    #    Delaunay hull and injects scene-spanning triangles into the "largest
    #    connected component" (the LichtFeld TSDF failure mode this pipeline
    #    is checked against). GaussianWrapping's own reference pipeline has
    #    an equivalent safeguard (`compute_valid_mask`, per-camera frustum
    #    culling of pivots before Delaunay); this is a cheaper radius-based
    #    approximation of the same idea.
    if max_radius_factor is not None:
        crop_radius = max_radius_factor * scene_radius
        dist_from_center = (pivots - avg_center.to(pivots.device)).norm(dim=-1)
        keep = dist_from_center <= crop_radius
        pivots = pivots[keep]
        pivot_scales = pivot_scales[keep]
        stats["max_radius_factor"] = max_radius_factor
        stats["crop_radius"] = crop_radius
        stats["n_pivots_dropped_by_radius_crop"] = int((~keep).sum().item())
    stats["n_pivots"] = int(pivots.shape[0])
    stats["t_pivots_sec"] = time.time() - t0
    print(f"[mesh_extraction] pivots: {stats['n_pivots']} kept / "
          f"{stats['n_pivots_raw']} raw (from {stats['n_gaussians']} Gaussians) "
          f"in {stats['t_pivots_sec']:.1f}s")

    # -- 2. Field at pivots ---------------------------------------------------
    t0 = time.time()
    pivot_sdf = field_fn(pivots)
    stats["t_fuse_field_sec"] = time.time() - t0
    n_pos = int((pivot_sdf > 0).sum().item())
    n_neg = int((pivot_sdf < 0).sum().item())
    n_trunc_pos = int((pivot_sdf >= 0.999).sum().item())
    n_trunc_neg = int((pivot_sdf <= -0.999).sum().item())
    stats["field_frac_positive"] = n_pos / pivot_sdf.numel()
    stats["field_frac_negative"] = n_neg / pivot_sdf.numel()
    stats["field_frac_truncated_positive"] = n_trunc_pos / pivot_sdf.numel()
    stats["field_frac_truncated_negative"] = n_trunc_neg / pivot_sdf.numel()
    print(f"[mesh_extraction] pivot field ({sdf_mode}) evaluated in {stats['t_fuse_field_sec']:.1f}s -- "
          f"frac positive={stats['field_frac_positive']:.4f}, "
          f"frac negative={stats['field_frac_negative']:.4f}")

    # -- 3. Delaunay --------------------------------------------------------
    t0 = time.time()
    pivots_np = pivots.detach().cpu().double().numpy()
    delaunay = Delaunay(pivots_np)
    tets = torch.from_numpy(delaunay.simplices.astype(np.int64)).to(pivots.device)
    stats["t_delaunay_sec"] = time.time() - t0
    stats["n_tets"] = int(tets.shape[0])
    print(f"[mesh_extraction] Delaunay: {stats['n_tets']} tets in "
          f"{stats['t_delaunay_sec']:.1f}s")

    # -- 4. Marching tetrahedra ----------------------------------------------
    t0 = time.time()
    end_points, end_sdf, end_scales, faces, edge_pivot_idx = marching_tetrahedra(
        vertices=pivots, tets=tets, sdf=(pivot_sdf - iso), scales=pivot_scales,
    )
    stats["t_mtet_sec"] = time.time() - t0
    stats["n_raw_triangles"] = int(faces.shape[0])
    stats["n_raw_vertices"] = int(end_points.shape[0])
    print(f"[mesh_extraction] marching tetrahedra: {stats['n_raw_vertices']} verts, "
          f"{stats['n_raw_triangles']} tris in {stats['t_mtet_sec']:.1f}s")

    # Linear-interpolation vertex positions (pre-refinement)
    norm_sdf = end_sdf.abs() / end_sdf.abs().sum(dim=1, keepdim=True).clamp(min=1e-8)
    verts_linear = end_points[:, 0, :] * norm_sdf[:, 1, :] + end_points[:, 1, :] * norm_sdf[:, 0, :]

    # -- 4b. Large-edge filtering (GW filter_large_edges safeguard) -----------
    # Drop faces whose crossing edge spans farther than the sum of the two
    # bracketing pivots' scales -- guards against Delaunay tets connecting
    # far-apart outlier pivots (e.g. large background Gaussians in an
    # unbounded scene) from injecting spurious scene-spanning triangles.
    vertex_ok = _large_edge_vertex_mask(end_points, end_scales)
    edge_filter_mask = vertex_ok[faces].all(dim=1)
    faces = faces[edge_filter_mask]
    stats["n_triangles_after_edge_filter"] = int(faces.shape[0])
    print(f"[mesh_extraction] {stats['n_triangles_after_edge_filter']} / "
          f"{stats['n_raw_triangles']} triangles kept after large-edge filter")

    # -- 5. Binary-search refinement ------------------------------------------
    t0 = time.time()
    if n_binary_steps > 0:
        verts = _binary_search_refine(
            end_points, end_sdf - iso, field_fn_refine, n_steps=n_binary_steps,
        )
    else:
        verts = verts_linear
    stats["t_refine_sec"] = time.time() - t0
    print(f"[mesh_extraction] binary-search refinement ({n_binary_steps} steps) "
          f"in {stats['t_refine_sec']:.1f}s")

    # -- 6. Drop degenerate triangles -----------------------------------------
    faces_clean = _drop_degenerate_triangles(verts, faces)
    stats["n_postrefine_vertices"] = int(verts.shape[0])
    stats["n_nondegenerate_triangles"] = int(faces_clean.shape[0])
    print(f"[mesh_extraction] {stats['n_nondegenerate_triangles']} / "
          f"{stats['n_triangles_after_edge_filter']} triangles kept after degeneracy check")

    # -- 7. Largest connected component ---------------------------------------
    t0 = time.time()
    verts_np = verts.detach().cpu().numpy()
    faces_np = faces_clean.detach().cpu().numpy()
    largest_mesh, n_components, n_largest_faces = _largest_connected_component(verts_np, faces_np)
    stats["t_postprocess_sec"] = time.time() - t0
    stats["n_components"] = n_components
    stats["n_largest_component_triangles"] = n_largest_faces
    stats["final_n_vertices"] = int(largest_mesh.vertices.shape[0])
    stats["final_n_triangles"] = int(largest_mesh.faces.shape[0])
    bbox_min = largest_mesh.vertices.min(axis=0).tolist()
    bbox_max = largest_mesh.vertices.max(axis=0).tolist()
    stats["mesh_bbox_min"] = bbox_min
    stats["mesh_bbox_max"] = bbox_max
    print(f"[mesh_extraction] {n_components} connected components; "
          f"largest = {n_largest_faces} tris "
          f"({stats['t_postprocess_sec']:.1f}s)")

    # -- 8. Write PLY -----------------------------------------------------------
    largest_mesh.export(output_path)
    stats["output_path"] = output_path
    stats["t_total_sec"] = time.time() - t_start
    print(f"[mesh_extraction] wrote {output_path} "
          f"({stats['final_n_vertices']} verts / {stats['final_n_triangles']} tris) "
          f"total {stats['t_total_sec']:.1f}s")

    if stats_path is not None:
        with open(stats_path, "w") as f:
            for k, v in stats.items():
                f.write(f"{k}: {v}\n")

    return stats
