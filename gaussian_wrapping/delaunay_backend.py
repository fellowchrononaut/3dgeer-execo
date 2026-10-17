"""3D Delaunay tetrahedralization backends for mesh extraction and PAM.

scipy (Qhull): the original backend; single-threaded and memory-hungry, it
ran out of host memory above ~1.5-2 M points (hence extract_mesh.py's
--max_pivots 1.5 M cap and the 1.4 M PAM limit on x5).
cgal: CGAL's Delaunay_triangulation_3 through Tetra-NeRF's extension
(GaussianWrapping's own backend, `tetranerf.utils.extension.cpp.triangulate`;
GW scene/mesh.py `method == "tetranerf"`): ~35 s for 4 M points. Identical
tetrahedra to scipy (tested on random points), possibly in another order.
"""
import numpy as np
import torch

_OPPOSITE_FACE = [[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]]


def delaunay_tets(points: np.ndarray, backend: str = "scipy") -> np.ndarray:
    """(N,3) points -> (T,4) int64 vertex indices of the Delaunay tetrahedra."""
    if backend == "scipy":
        from scipy.spatial import Delaunay
        return Delaunay(np.asarray(points, dtype=np.float64)).simplices.astype(np.int64)
    if backend == "cgal":
        try:
            from tetranerf.utils.extension import cpp
        except ImportError as e:
            raise ImportError("--delaunay cgal needs Tetra-NeRF's CGAL extension (tetranerf); "
                              "see hpc/build_extensions.sh / docker setup") from e
        tets = cpp.triangulate(torch.as_tensor(np.asarray(points), dtype=torch.float32).contiguous())
        return tets.cpu().numpy().astype(np.int64)
    raise ValueError(f"unknown Delaunay backend {backend!r} (scipy | cgal)")


@torch.no_grad()
def tet_neighbors(simplices: np.ndarray) -> np.ndarray:
    """scipy.spatial.Delaunay.neighbors for given tetrahedra: neighbors[i, j] is
    the tetrahedron sharing the face of tet i opposite its vertex j, -1 on the
    hull. Computed by matching sorted faces (on the GPU when available)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    s = torch.from_numpy(np.ascontiguousarray(simplices)).to(device=device, dtype=torch.int64)
    T = s.shape[0]
    faces = torch.stack([s[:, o] for o in _OPPOSITE_FACE], 1).reshape(-1, 3)   # row 4i+j: face opposite vertex j
    faces, _ = faces.sort(dim=1)
    _, inv = torch.unique(faces, dim=0, return_inverse=True)
    del faces
    order = torch.argsort(inv, stable=True)
    inv_sorted = inv[order]
    same = inv_sorted[1:] == inv_sorted[:-1]                                    # interior faces appear twice
    a, b = order[:-1][same], order[1:][same]
    nb = torch.full((4 * T,), -1, dtype=torch.int64, device=device)
    nb[a] = b // 4
    nb[b] = a // 4
    return nb.view(T, 4).cpu().numpy()
