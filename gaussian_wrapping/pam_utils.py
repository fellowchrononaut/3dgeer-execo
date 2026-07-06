# ported/adapted from GaussianWrapping gaussian_wrapping/utils/primal_adaptive_meshing_utils.py
"""Support utilities for gaussian_wrapping/pam_extraction.py (Session E3 PAM
port): the Delaunay-tet mesh container (`MeshFromDelaunay`, pure numpy/scipy,
ported near-verbatim), camera-aware surface sampling
(`sample_mesh_proportional_to_camera`), a numerical-gradient fallback for
Newton refinement (`numerical_field_gradient`, still used by
`--gradient_mode numerical`), and -- as of the Session E3 PAM-noise fix --
GW's own **analytic** Newton-step gradient (`GaussianVectorField` /
`get_vector_field_quantities_aux` / `robust_sigma_inv` /
`robust_gaussian_eval_shifted_points`, `--gradient_mode analytic`).

This analytic path was originally skipped (see pam_extraction.py's old
deviation #1) because it was judged to need GW-specific `GaussianModel`
machinery we didn't have. We since diagnosed that PAM's noisiest failure mode
(median dihedral 24 deg but component count 6x the raw mesh's, on the truck
checkpoint) traces to `numerical_field_gradient` differentiating our own
*rendered* occupancy field, which inherits that field's per-view
nearest-pixel binning discretization (`fields.py`'s `.round()` lookup, not
bilinear) -- a numerical artifact, not a real surface feature. GW's analytic
gradient sidesteps this entirely: it is a closed-form Gaussian-mixture
log-vacancy gradient computed from a KD-tree k=32-nearest-Gaussian lookup
(by center), using ONLY each Gaussian's own mean/normal/scaling/rotation/
opacity -- zero cameras, zero rasterization, zero pixel binning anywhere. We
now have all of those accessors (`gaussians.get_xyz/get_normals/get_scaling/
get_rotation/get_opacity`, all already activated/normalized) with no GW-style
mip-filter (`_with_3D_filter`) variants needed, so the port is tractable
without any new GaussianModel machinery. It is a paired-but-mismatched
scalar/gradient Newton scheme (GW's own comment: "Non normalized normal
field is grad log v") -- a production-proven heuristic, not an exact
gradient of our own occupancy scalar -- ported faithfully as-is.

NOT ported from the GW file: `plot_histogram`'s exact styling (kept, but
trivial/optional -- see below).
"""
import os
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch
from scipy.spatial import Delaunay, KDTree

from gaussian_wrapping.fields import world_to_view
from gaussian_wrapping.fisheye_proj import project_view_to_pixel
from utils.general_utils import build_scaling_rotation

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
# Analytic Newton-step gradient (GW's GaussianVectorField, ported verbatim --
# see module docstring above for why this is now tractable / what it is).
# Ported from GW utils/general_utils.py::robust_sigma_inv,
# robust_gaussian_eval_shifted_points (lines ~147-214) and
# utils/primal_adaptive_meshing_utils.py::get_vector_field_quantities_aux,
# GaussianVectorField (lines ~179-291), variable names adapted only.
# ---------------------------------------------------------------------------

def robust_sigma_inv(g_scales: torch.Tensor, g_rotation: torch.Tensor,
                      return_invscale_rot: bool = False):
    """Inverse covariance Sigma^-1 = (S^-1 R^T)^T (S^-1 R^T) for an
    anisotropic Gaussian, given per-Gaussian scale/rotation. Accepts
    (N,3)/(N,4) or batched (B,k,3)/(B,k,4) inputs (ported verbatim from GW's
    utils/general_utils.py::robust_sigma_inv, using OUR
    `build_scaling_rotation` -- identical implementation already in
    utils/general_utils.py).

    NOTE (verbatim from GW): uses `.view()`, not `.reshape()` -- callers must
    pass contiguous (N,3)/(N,4) or (B,k,3)/(B,k,4) tensors (true for
    `GaussianVectorField`'s fancy-indexed gather below; test code building
    synthetic batches should `.repeat()`/`.contiguous()`, not `.expand()`).
    """
    using_batches = g_scales.ndim == 3
    if using_batches:
        B, k, _ = g_scales.shape
        g_scales = g_scales.view(-1, 3)
        g_rotation = g_rotation.view(-1, 4)
    M = build_scaling_rotation(s=1.0 / g_scales, r=g_rotation).transpose(-1, -2)  # (..., 3, 3)
    sigma_inv = M.transpose(-1, -2) @ M  # (..., 3, 3)
    if using_batches:
        sigma_inv = sigma_inv.view(B, k, 3, 3)
        M = M.view(B, k, 3, 3)
    if return_invscale_rot:
        return sigma_inv, M
    return sigma_inv


def robust_gaussian_eval_shifted_points(shifted_points: torch.Tensor,
                                         gaussian_invscale_rot: torch.Tensor,
                                         gaussian_opacity: torch.Tensor) -> torch.Tensor:
    """Numerically-stable anisotropic Gaussian density eval at points already
    shifted by (x - mu). (N,3),(N,3,3),(N,1) -> (N,1). Ported verbatim from
    GW's utils/general_utils.py::robust_gaussian_eval_shifted_points."""
    transformed_shifts = torch.bmm(
        gaussian_invscale_rot,             # (N, 3, 3)
        shifted_points.unsqueeze(-1),      # (N, 3, 1)
    ).squeeze(-1)                          # (N, 3)
    dist_sq = (transformed_shifts ** 2).sum(dim=-1, keepdim=True)  # (N, 1)
    gaussian_density = gaussian_opacity * torch.exp(-0.5 * dist_sq)  # (N, 1)
    return gaussian_density


def get_vector_field_quantities_aux(points: torch.Tensor, g_means: torch.Tensor,
                                     g_normals: torch.Tensor, g_scales: torch.Tensor,
                                     g_rotation: torch.Tensor, g_opacity: torch.Tensor) -> torch.Tensor:
    """GW's analytic Newton-step gradient formula, evaluated from a fixed set
    of k neighbor Gaussians per query point.

    Shapes: points (B,3); g_means/g_normals/g_scales (B,k,3); g_rotation
    (B,k,4); g_opacity (B,k,1). Returns (B,3), the summed
    indicator-clipped weighted `Sigma_i^-1 @ (x - mu_i)` vector field (GW's
    own comment: "Non normalized normal field is grad log v" -- this is NOT
    the gradient of OUR render-based `field_fn`; see module docstring above
    and `GaussianVectorField` below for the empirically-verified sign
    convention that lets it stand in for one anyway).

    Ported verbatim from GW's utils/primal_adaptive_meshing_utils.py::
    get_vector_field_quantities_aux (variable names adapted only; GW returns
    a dict with one key, we return the tensor directly since nothing else in
    this port needs curl).
    """
    B, k_neighbors = g_means.shape[0], g_means.shape[1]
    p = points.unsqueeze(1) - g_means  # (B, k, 3)

    # G_i(x) via S^-1 @ R^T
    g_invscale_rot = build_scaling_rotation(
        s=1.0 / g_scales.view(-1, 3),
        r=g_rotation.view(-1, 4),
    ).transpose(-1, -2)  # (B*k, 3, 3)
    gi_x = robust_gaussian_eval_shifted_points(
        shifted_points=p.view(-1, 3),
        gaussian_invscale_rot=g_invscale_rot,
        gaussian_opacity=g_opacity.view(-1, 1),
    ).view(B, k_neighbors, 1)  # (B, k, 1)

    # Half-space clip: only Gaussians whose outward normal faces x contribute.
    n_dot_x_minus_mu = torch.sum(g_normals * p, dim=-1, keepdim=True)  # (B, k, 1)
    indicator_function = (n_dot_x_minus_mu >= 0).float()  # (B, k, 1)

    # Sigma_i^-1 @ (x - mu_i)
    sigma_inv = robust_sigma_inv(g_scales, g_rotation)  # (B, k, 3, 3)
    transformed_points = torch.einsum("bkij, bkj -> bki", sigma_inv, p)  # (B, k, 3)

    gaussian_quotient = gi_x / (1.0 - gi_x + 1e-8)  # (B, k, 1)
    nabla_log = gaussian_quotient * transformed_points  # (B, k, 3)
    normal_field = torch.sum(indicator_function * nabla_log, dim=1)  # (B, 3)
    return normal_field


class GaussianVectorField:
    """GW's analytic Newton-step gradient source: a KD-tree over raw
    Gaussian centers (built once), queried for k nearest neighbors per query
    point, feeding `get_vector_field_quantities_aux`. Zero cameras, zero
    rasterization, zero pixel binning -- this is exactly why it doesn't
    inherit our rendered field's discretization noise (see module docstring).

    SIGN CONVENTION (empirically verified, see
    tests/test_pam_analytic_gradient.py -- do not re-derive by hand only,
    the half-space-indicator clipping makes it easy to get backwards): the
    raw output of `get_vector_field_quantities_aux`, used AS-IS with no sign
    flip, already points toward increasing OUR field's vacancy convention
    (field>0 empty, field<0 occupied) -- i.e. it is compatible with
    `pam_extraction.py::_gradient_descent_refinement`'s existing Newton-step
    formula unmodified. `tests/test_pam_analytic_gradient.py` confirms this
    with both a direct sign check on the raw vector and an end-to-end
    refinement-step displacement check (points move toward field==0 from
    both sides).
    """

    @torch.no_grad()
    def __init__(self, gaussians, k_neighbors: int = 32) -> None:
        self.k_neighbors = k_neighbors
        means = gaussians.get_xyz.detach().cpu().numpy()  # (N, 3)
        self.tree = KDTree(means)

    @torch.no_grad()
    def gradient(self, query_points: torch.Tensor, gaussians) -> torch.Tensor:
        """(N,3) query points -> (N,3) analytic gradient."""
        device = query_points.device
        q_np = query_points.detach().cpu().numpy()
        _, nn_idx_np = self.tree.query(q_np, k=self.k_neighbors, workers=-1)
        nn_idx = torch.from_numpy(nn_idx_np).to(device=device, dtype=torch.long)
        if nn_idx.ndim == 1:  # scipy squeezes the batch dim when N==1
            nn_idx = nn_idx.unsqueeze(0)

        means = gaussians.get_xyz
        normals = gaussians.get_normals
        scales = gaussians.get_scaling
        rotations = gaussians.get_rotation
        opacity = gaussians.get_opacity

        N, k = nn_idx.shape
        flat_idx = nn_idx.reshape(-1)
        return get_vector_field_quantities_aux(
            query_points,
            means[flat_idx].view(N, k, 3),
            normals[flat_idx].view(N, k, 3),
            scales[flat_idx].view(N, k, 3),
            rotations[flat_idx].view(N, k, 4),
            opacity[flat_idx].view(N, k, 1),
        )


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
