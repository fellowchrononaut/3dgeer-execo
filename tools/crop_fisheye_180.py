#!/usr/bin/env python3
"""
tools/crop_fisheye_180.py
# ported/adapted (concept only) from GaussianWrapping's data-prep utilities;
# this implementation is 3DGEER/EQ-specific. 3DGEER's EQ render mode caps at
# 180 deg FoV (GitHub issue #41) while the Insta360 X5 lenses are ~190 deg, so
# X5 frames must be masked down to a <=180 deg (<=90 deg half-FoV) circular
# FoV before training / Metashape alignment.

Given a directory of fisheye frames plus the lens focal length (in pixels,
equidistant model: r = f * theta) and principal point, this script:
  1. computes a circular mask of radius r = focal_px * deg2rad(90 - margin_deg)
     centered at (cx, cy), i.e. a (180 - 2*margin_deg) degree FoV circle,
  2. writes that mask as a single-channel PNG (--mask-out), directly reusable
     as train.py's --mask_path or as a Metashape/COLMAP per-image mask,
  3. blacks out every pixel outside the circle in each frame, either into a
     separate --output directory or in-place (default) when --output is
     omitted.

Usage:
    python tools/crop_fisheye_180.py --input frames/ --focal-px 950 \
        --mask-out frames/mask.png --output frames_masked/
"""
import argparse
import glob
import math
import os

import cv2
import numpy as np

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def list_images(input_dir):
    paths = []
    for ext in IMG_EXTS:
        paths.extend(glob.glob(os.path.join(input_dir, f"*{ext}")))
        paths.extend(glob.glob(os.path.join(input_dir, f"*{ext.upper()}")))
    return sorted(set(paths))


def make_circular_mask(height, width, cx, cy, radius_px):
    yy, xx = np.mgrid[0:height, 0:width]
    dist2 = (xx - cx) ** 2 + (yy - cy) ** 2
    return np.where(dist2 <= radius_px ** 2, 255, 0).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(
        description="Circular fisheye mask/crop for 180deg-capped EQ training "
                     "(X5 lenses are ~190deg; 3DGEER EQ mode caps at 180deg).")
    parser.add_argument("--input", required=True, help="Directory of input frames.")
    parser.add_argument("--output", default=None,
                         help="Directory to write masked frames. If omitted, "
                              "frames are modified in-place.")
    parser.add_argument("--focal-px", type=float, required=True,
                         help="Lens focal length in pixels (equidistant model).")
    parser.add_argument("--cx", type=float, default=None,
                         help="Principal point x (px). Defaults to image width/2.")
    parser.add_argument("--cy", type=float, default=None,
                         help="Principal point y (px). Defaults to image height/2.")
    parser.add_argument("--margin_deg", type=float, default=5.0,
                         help="Shrink the 90deg half-FoV circle by this many "
                              "degrees (default 5).")
    parser.add_argument("--mask-out", default="mask.png",
                         help="Path to write the circular mask PNG (default mask.png).")
    args = parser.parse_args()

    images = list_images(args.input)
    if not images:
        raise SystemExit(f"No images found in {args.input}")

    first = cv2.imread(images[0], cv2.IMREAD_UNCHANGED)
    if first is None:
        raise SystemExit(f"Could not read {images[0]}")
    height, width = first.shape[:2]

    cx = args.cx if args.cx is not None else width / 2.0
    cy = args.cy if args.cy is not None else height / 2.0

    half_fov_rad = math.radians(90.0 - args.margin_deg)
    radius_px = args.focal_px * half_fov_rad

    mask = make_circular_mask(height, width, cx, cy, radius_px)
    mask_out_dir = os.path.dirname(args.mask_out)
    if mask_out_dir:
        os.makedirs(mask_out_dir, exist_ok=True)
    cv2.imwrite(args.mask_out, mask)
    print(f"Wrote mask ({width}x{height}, r={radius_px:.1f}px, "
          f"center=({cx:.1f},{cy:.1f})) to {args.mask_out}")

    if args.output is not None:
        os.makedirs(args.output, exist_ok=True)

    mask_bool = mask > 0
    n_done = 0
    for path in images:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"Warning: could not read {path}, skipping.")
            continue
        if img.shape[0] != height or img.shape[1] != width:
            print(f"Warning: {path} is {img.shape[1]}x{img.shape[0]} "
                  f"!= reference {width}x{height}; recomputing mask for it.")
            frame_mask = make_circular_mask(img.shape[0], img.shape[1], cx, cy, radius_px) > 0
        else:
            frame_mask = mask_bool
        out = img.copy()
        out[~frame_mask] = 0
        dest = os.path.join(args.output, os.path.basename(path)) if args.output else path
        cv2.imwrite(dest, out)
        n_done += 1

    print(f"Processed {n_done} frame(s).")


if __name__ == "__main__":
    main()
