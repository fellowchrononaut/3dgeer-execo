"""Loading per-pixel ray maps for KB/EQ (fisheye) cameras.

--raymap_path accepts either
  * a single .npy file: one ray map shared by every camera (previous behavior), or
  * a path containing ``{cam_id}``, e.g. ``data/x5/raymap_fisheye_cam{cam_id}.npy``:
    one ray map per COLMAP camera id (e.g. one per lens of a dual-fisheye rig).
    Each image gets the ray map of its own COLMAP camera, matching the per-camera
    intrinsics/distortion that scene/dataset_readers.py already reads from
    cameras_fish.txt. Generate them with ``tools/fisheye_raymap.py --from_colmap``.
"""
import glob
import os
import re

import numpy as np

CAM_ID_PLACEHOLDER = "{cam_id}"


def load_raymap(raymap_path):
    """None | np.ndarray (shared) | dict {camera_id: np.ndarray} (per camera).

    A missing single-file path returns None, as the callers did before; a
    per-camera pattern that matches nothing is an error.
    """
    if not raymap_path:
        return None
    if CAM_ID_PLACEHOLDER not in raymap_path:
        return np.load(raymap_path) if os.path.exists(raymap_path) else None

    regex = re.compile(re.escape(raymap_path).replace(re.escape(CAM_ID_PLACEHOLDER), r"(\d+)"))
    raymaps = {}
    for f in sorted(glob.glob(raymap_path.replace(CAM_ID_PLACEHOLDER, "*"))):
        m = regex.fullmatch(f)
        if m:
            raymaps[int(m.group(1))] = np.load(f)
    if not raymaps:
        raise FileNotFoundError(f"no per-camera ray maps match {raymap_path!r}")
    shapes = {k: v.shape for k, v in raymaps.items()}
    print(f"[raymap] per-camera ray maps for COLMAP cameras {sorted(raymaps)}: {shapes}")
    return raymaps


def raymap_for_camera(raymap, camera_id):
    """The ray map for one image's COLMAP camera (or the shared one)."""
    if isinstance(raymap, dict):
        if camera_id not in raymap:
            raise KeyError(f"no ray map for COLMAP camera {camera_id}; "
                           f"available: {sorted(raymap)} (check --raymap_path)")
        return raymap[camera_id]
    return raymap
