#!/usr/bin/env python3
"""
tools/mesh_dihedral_stats.py

Session E2.1 verification (GUTWrap Path A): standalone dihedral-angle
roughness metric for extracted meshes, matching the method used in the
2026-07-06 QA session-log entries of 3DGEERGW_EXECUTION.md (a real flat
panel should have near-0deg dihedral between adjacent triangle face normals,
spiking only at genuine edges; ~88deg median on the raw pre-fix pivot-MTet
mesh was close to random face orientation).

Metric: `trimesh.Trimesh.face_adjacency_angles` gives the angle (radians)
between the two face normals sharing each mesh edge directly -- this IS the
dihedral angle, no re-derivation needed. Connected components via
`mesh.split(only_watertight=False)` (SAME method `gaussian_wrapping/
mesh_extraction.py::_largest_connected_component` already uses for
post-processing, for consistency with the rest of this codebase).

Usage:
    python tools/mesh_dihedral_stats.py mesh1.ply [mesh2.ply ...]
"""
import sys

import numpy as np
import trimesh


def dihedral_stats(path: str) -> dict:
    mesh = trimesh.load(path, process=False)
    angles_deg = np.degrees(mesh.face_adjacency_angles)

    components = mesh.split(only_watertight=False)
    n_components = len(components)

    return {
        "path": path,
        "n_vertices": int(mesh.vertices.shape[0]),
        "n_faces": int(mesh.faces.shape[0]),
        "n_edges_adjacency": int(angles_deg.shape[0]),
        "median_dihedral_deg": float(np.median(angles_deg)) if angles_deg.size else float("nan"),
        "mean_dihedral_deg": float(np.mean(angles_deg)) if angles_deg.size else float("nan"),
        "frac_edges_gt_30deg": float((angles_deg > 30.0).mean()) if angles_deg.size else float("nan"),
        "n_components": n_components,
    }


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    for path in sys.argv[1:]:
        stats = dihedral_stats(path)
        print(f"\n=== {stats['path']} ===")
        print(f"  vertices: {stats['n_vertices']}, faces: {stats['n_faces']}, "
              f"adjacency edges: {stats['n_edges_adjacency']}")
        print(f"  median dihedral: {stats['median_dihedral_deg']:.2f} deg")
        print(f"  mean dihedral:   {stats['mean_dihedral_deg']:.2f} deg")
        print(f"  frac edges > 30deg: {stats['frac_edges_gt_30deg']:.4f}")
        print(f"  connected components: {stats['n_components']}")


if __name__ == "__main__":
    main()
