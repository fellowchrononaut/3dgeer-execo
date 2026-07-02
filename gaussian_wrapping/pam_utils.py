# ported/adapted from GaussianWrapping gaussian_wrapping/utils/primal_adaptive_meshing_utils.py
"""Support utilities for gaussian_wrapping/pam_extraction.py (Session E3 PAM
port): the Delaunay-tet mesh container (`MeshFromDelaunay`, pure numpy/scipy,
ported near-verbatim), camera-aware surface sampling
(`sample_mesh_proportional_to_camera`), and a numerical-gradient substitute
for GW's analytic `GaussianVectorField` (see pam_extraction.py's module
docstring, deviation #1, for why).

NOT ported from the GW file: `GaussianVectorField` / `get_vector_field_
quantities_aux` (needs GW-specific GaussianModel attributes we don't have --
see pam_extraction.py header) and `plot_histogram`'s exact styling (kept, but
trivial/optional -- see below).
"""
import os
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch
from scipy.spatial import Delaunay

from gaussian_wrapping.fields import world_to_view
from gaussian_wrapping.fisheye_proj import project_view_to_pixel

# ---------------------------------------------------------------------------
# MeshFromDelaunay (verbatim port -- pure numpy/scipy, no GW renderer deps)
# ---------------------------------------------------------------------------

def tet_circumcenter(verts: np.ndarray) -> np.ndarray:
    ba_x = verts[:, 1, 0] - verts[:, 0, 0]
    ba_y = verts[:, 1, 1] - verts[:, 0, 1]
    ba_z = verts[:, 1, 2] - verts[:, 0, 2]
    ca_x = verts[:, 2, 0] - verts[:, 0, 0]
    ca_y = verts[:, 2, 1] - verts[:, 0, 1]
    ca_z = verts[:, 2, 2] - verts[:, 0, 2]
    da_x = verts[:, 3, 0] - verts[:, 0, 0]
    da_y = verts[:, 3, 1] - verts[:, 0, 1]
    da_z = verts[:, 3, 2] - verts[:, 0, 2]
    len_ba = ba_x * ba_x + ba_y * ba_y + ba_z * ba_z
    len_ca = ca_x * ca_x + ca_y * ca_y + ca_z * ca_z
    len_da = da_x * da_x + da_y * da_y + da_z * da_z
    cross_cd_x = ca_y * da_z - da_y * ca_z
    cross_cd_y = ca_z * da_x - da_z * ca_x
    cross_cd_z = ca_x * da_y - da_x * ca_y
    cross_db_x = da_y * ba_z - ba_y * da_z
    cross_db_y = da_z * ba_x - ba_z * da_x
    cross_db_z = da_x * ba_y - ba_x * da_y
    cross_bc_x = ba_y * ca_z - ca_y * ba_z
    cross_bc_y = ba_z * ca_x - ca_z * ba_x
    cross_bc_z = ba_x * ca_y - ca_x * ba_y
    div_den = (ba_x * cross_cd_x + ba_y * cross_cd_y + ba_z * cross_cd_z)
    mask_div_den = np.abs(div_den) == 0
    div_den[mask_div_den] = 1
    denominator = 0.5 / div_den
    circ_x = (len_ba * cross_cd_x + len_ca * cross_db_x + len_da * cross_bc_x) * denominator
    circ_y = (len_ba * cross_cd_y + len_ca * cross_db_y + len_da * cross_bc_y) * denominator
    circ_z = (len_ba * cross_cd_z + len_ca * cross_db_z + len_da * cross_bc_z) * denominator
    out = np.column_stack((circ_x, circ_y, circ_z)) + verts[:, 0]
    out[mask_div_den] = verts[mask_div_den].mean(1)
    return out


class MeshFromDelaunay(Delaunay):
    """Delaunay tetrahedralization + tet-classification -> boundary-triangle
    surface extraction. Verbatim port of GW's utils/primal_adaptive_meshing_
    utils.py::MeshFromDelaunay (pure numpy/scipy)."""

    def __init__(self, points: np.ndarray, add_corners: bool = False, **kwargs) -> None:
        super().__init__(points, **kwargs)
        self.add_corners = add_corners
        self.circum_centers = tet_circumcenter(self.points[self.simplices])
        self.barycenters = self.points[self.simplices].mean(1)
        self.triangle_faces, self.triangle_faces_neighbors = self.get_triangle_faces()
        self.in_mask = self.order_neighbors()
        self.triangle_areas = np.sqrt((np.cross(
            self.points[self.triangle_faces[:, 1]] - self.points[self.triangle_faces[:, 0]],
            self.points[self.triangle_faces[:, 2]] - self.points[self.triangle_faces[:, 0]]) ** 2).sum(-1))
        self.triangle_max_length = self.get_triangle_max_length()

    def get_triangle_faces(self):
        opp_face = [[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]]
        ii = np.arange(len(self.neighbors))
        triangle_faces = -np.ones((len(self.neighbors) * 4, 3), dtype=int)
        for j in range(4):
            triangle_faces[4 * ii + j] = self.simplices[:, opp_face[j]]
        triangle_faces_neighbors = np.column_stack((
            np.arange(len(self.neighbors)).repeat(4), self.neighbors.reshape(len(self.neighbors) * 4)))
        return triangle_faces, triangle_faces_neighbors

    def get_triangle_max_length(self):
        l1 = ((self.points[self.triangle_faces[:, 1]] - self.points[self.triangle_faces[:, 0]]) ** 2).sum(-1)
        l2 = ((self.points[self.triangle_faces[:, 2]] - self.points[self.triangle_faces[:, 0]]) ** 2).sum(-1)
        l3 = ((self.points[self.triangle_faces[:, 2]] - self.points[self.triangle_faces[:, 1]]) ** 2).sum(-1)
        return np.max(np.stack((l1, l2, l3)), 0)

    def face_orientation(self, p1, p2, p3, vp1):
        return (np.cross(p2 - p1, p3 - p1) * (vp1 - (p1 + p2 + p3) / 3.)).sum(-1) > 0

    def order_neighbors(self):
        opp_vert = self.simplices.reshape(len(self.simplices[self.triangle_faces_neighbors]))
        in_mask = self.face_orientation(
            *np.transpose(self.points[self.triangle_faces], (1, 0, 2)), self.points[opp_vert])
        in_triangle = self.triangle_faces
        flipped_triangles = np.fliplr(in_triangle)
        self.triangle_faces = in_triangle * in_mask[:, None] + (1 - in_mask[:, None]) * flipped_triangles
        return in_mask

    def sample_random_tet_points(self, n: int = 1) -> np.ndarray:
        weights = np.random.rand(n, *self.simplices.shape)
        weights /= weights.sum(-1, keepdims=True)
        x = self.points[self.simplices]
        bary_x = (weights[..., None] * x[None, ...]).sum(-2)
        return bary_x

    def get_surface(self, treshold: float = 0, len_threshold: Optional[float] = None):
        neigh_color = self.tet_colors[self.triangle_faces_neighbors]
        neigh_color[self.triangle_faces_neighbors == -1] = 0
        neigh_color = neigh_color > treshold
        in_t = (neigh_color[:, 0] == 0) * (neigh_color[:, 1] == 1)
        in_triangle = self.triangle_faces[in_t]

        un = np.unique(in_triangle)
        if un.shape[0] == 0:
            return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64)
        inv = np.arange(in_triangle.max() + 1)
        inv[un] = np.arange(len(un))
        nvertices = self.points[un]
        in_triangle = inv[in_triangle]

        if len_threshold is not None:
            in_scores = self.triangle_max_length[in_t]
            in_triangle = in_triangle[in_scores < len_threshold]
        return nvertices, in_triangle


# ---------------------------------------------------------------------------
# Numerical gradient of OUR field callable (substitute for GW's analytic
# GaussianVectorField -- see pam_extraction.py module docstring, deviation #1)
# ---------------------------------------------------------------------------

@torch.no_grad()
def numerical_field_gradient(field_fn: Callable, points: torch.Tensor, eps: float) -> torch.Tensor:
    """(N,3) -> (N,3) central-difference gradient of a scalar field_fn at
    `points`. 6 extra field_fn calls (2 per axis). `eps` is an absolute
    world-space step (see pam_extraction.py's `--grad_eps`, auto-scaled from
    the scene radius by default)."""
    grads = torch.zeros_like(points)
    for axis in range(3):
        offset = torch.zeros(1, 3, device=points.device, dtype=points.dtype)
        offset[0, axis] = eps
        f_plus = field_fn(points + offset)
        f_minus = field_fn(points - offset)
        grads[:, axis] = (f_plus - f_minus) / (2.0 * eps)
    return grads


# ---------------------------------------------------------------------------
# Camera-aware surface sampling (ported)
# ---------------------------------------------------------------------------

class TorchMesh:
    """Minimal (verts, faces, face_normals) container -- GW's own `Meshes`
    class is a much richer differentiable-rendering mesh type; the two PAM
    sampling paths (surface_even / proportional_to_camera) only need these
    three tensors, so this trims the port to what's actually used."""

    def __init__(self, verts: torch.Tensor, faces: torch.Tensor):
        self.verts = verts
        self.faces = faces
        v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
        n = torch.cross(v1 - v0, v2 - v0, dim=-1)
        self.face_normals = torch.nn.functional.normalize(n, dim=-1)


@torch.no_grad()
def is_in_view_frustum(points: torch.Tensor, camera) -> torch.Tensor:
    """(N,3) world points -> (N,) bool, True where the point projects inside
    `camera`'s image and in front of it. Ported in spirit from GW's utils/
    geometry_utils.py::is_in_view_frustum, but built on OUR fisheye_proj /
    world_to_view helpers (PH + KB/EQ) instead of GW's pinhole-only Fx/Fy/Cx/Cy."""
    x_view = world_to_view(camera, points)
    _, _, valid = project_view_to_pixel(camera, x_view)
    return valid


@torch.no_grad()
def compute_face_to_camera_minimum_distance(mesh: TorchMesh, cameras, batch_size: int = 100_000) -> torch.Tensor:
    """Minimum distance from each face's centroid to any camera that sees it
    (ported from GW utils/primal_adaptive_meshing_utils.py)."""
    all_cam_centers = torch.cat([c.camera_center[None] for c in cameras], dim=0)
    min_face_to_cam_distance = 1_000_000. * torch.ones(mesh.faces.shape[0], device=mesh.faces.device)

    for i in range(0, mesh.faces.shape[0], batch_size):
        batch_faces = mesh.faces[i:i + batch_size]
        batch_face_centers = mesh.verts[batch_faces].mean(dim=1)

        in_view_mask = torch.zeros(batch_faces.shape[0], len(cameras), device=mesh.faces.device, dtype=torch.bool)
        for j, camera in enumerate(cameras):
            in_view_mask[:, j] = is_in_view_frustum(batch_face_centers, camera)

        batch_face_dist = (batch_face_centers[:, None] - all_cam_centers[None]).norm(dim=2)
        batch_face_dist[~in_view_mask] = 1_000_000.
        min_face_to_cam_distance[i:i + batch_size] = batch_face_dist.min(dim=1).values

    return min_face_to_cam_distance


@torch.no_grad()
def sample_mesh_proportional_to_camera(mesh: TorchMesh, cameras, num_points: int, face_probs: Optional[torch.Tensor] = None):
    """Sample points (+normals) from a mesh with per-face probability
    proportional to (face_area / min_distance_to_any_seeing_camera^2) --
    ported verbatim (modulo `is_in_view_frustum`, see above)."""
    face_verts = mesh.verts[mesh.faces]

    if face_probs is None:
        min_face_to_cam_distance = compute_face_to_camera_minimum_distance(mesh, cameras, batch_size=100_000)
        face_areas = torch.linalg.norm(
            torch.cross(face_verts[:, 1] - face_verts[:, 0], face_verts[:, 2] - face_verts[:, 0], dim=-1), dim=1,
        ) / 2.0
        face_probs = face_areas / (min_face_to_cam_distance ** 2).clamp(min=1e-12)
        face_probs = face_probs / torch.sum(face_probs)

    sampled_face_idx = torch.multinomial(input=face_probs, num_samples=num_points, replacement=True)
    sampled_face_verts = face_verts[sampled_face_idx]

    b1 = torch.rand(num_points, 1, device=mesh.faces.device)
    b2 = torch.rand(num_points, 1, device=mesh.faces.device) * (1. - b1)
    b3 = 1. - b1 - b2
    bary = torch.cat([b1, b2, b3], dim=1)

    means = (sampled_face_verts * bary[..., None]).sum(dim=1)
    normals = mesh.face_normals[sampled_face_idx]

    return means.cpu().numpy(), normals.cpu().numpy(), face_probs


def sample_surface_even(mesh_trimesh, num_points: int):
    """"surface_even" sampling: area-weighted uniform surface sampling.
    Substitute for GW's eval/TNTUniScanEvals/uniform_sampling_eval.py::
    sample_surface (trimesh.sample.sample_surface is functionally
    equivalent -- area-weighted point + face-normal sampling -- and avoids
    porting the eval-harness module)."""
    import trimesh
    points, face_idx = trimesh.sample.sample_surface(mesh_trimesh, num_points)
    normals = mesh_trimesh.face_normals[face_idx]
    return np.asarray(points), np.asarray(normals)


def plot_histogram(occupancy_values, filename: str, title: str, delta: float = 0.05, iso_value: float = 0.0):
    """Best-effort port of GW's histogram debug plot (skips silently if
    matplotlib is unavailable -- this is a debug aid, not load-bearing)."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"[pam] matplotlib not available, skipping histogram plot {filename}")
        return
    if hasattr(occupancy_values, "detach"):
        occupancy_values = occupancy_values.detach().cpu().numpy()
    occupancy_values = np.asarray(occupancy_values).flatten()
    lower, upper = iso_value - delta, iso_value + delta
    zoomed = occupancy_values[(occupancy_values >= lower) & (occupancy_values <= upper)]
    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
    n_points = occupancy_values.shape[0]
    plt.figure(figsize=(8, 6))
    plt.hist(zoomed, bins=60, color="#1386e3", alpha=0.85, edgecolor="black", linewidth=1.2)
    plt.title(f"{title} (zoom [{lower:.3f}, {upper:.3f}])")
    plt.xlabel("Field value")
    plt.ylabel("Count")
    plt.xlim([lower, upper])
    plt.ylim([0, n_points])
    plt.grid(axis="y", alpha=0.25, linestyle="--")
    plt.tight_layout()
    plt.savefig(filename, dpi=120, bbox_inches="tight")
    plt.close()
