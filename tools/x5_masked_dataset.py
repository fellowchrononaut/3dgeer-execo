"""Build an X5 dual-fisheye training dataset with per-image loss masks.

Each mask (white = pixel used by the loss) is
  * the lens circle, cut by viewing angle: theta <= --theta_max_deg, computed per
    lens from that image's OPENCV_FISHEYE camera in <src>/colmap_ba/cameras.txt
    (the bundle-adjusted model, at the image resolution), AND
  * the photographer mask of that frame and lens (VID_001/dual_fisheye/
    {front,rear}_masks/frame_NNNNNN.png, white = keep), with the masked region
    grown by --dilate_px.

The images are the unbaked frames (content to the edge of the lens circle),
instead of the copies with an ~80 deg circle burnt in that x5_eq/resized_images
holds. Train with --masks masks (see arguments/__init__.py) and no --mask_path.

  python tools/x5_masked_dataset.py --src data/x5_ba_1920 \\
      --images ../x5_eq/images \\
      --person_masks '/media/GS_Captures/Insta360x5/Insta360x5/VID_001/dual_fisheye/{lens}_masks/{stem}.png' \\
      --out data/x5_ba_masked_1920

Output (<out>): colmap -> <src>/colmap, raymap_fisheye_cam*.npy -> <src>,
resized_images -> --images (the reader's fixed folder name for SCANNETPP KB/EQ),
masks/<image_name>.png at the image resolution, README.txt.
"""
import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from colmap_ba_fisheye import read_text_model  # noqa: E402

LENS_NAMES = {0: "front", 1: "rear"}


def theta_map(W, H, p):
    """Per-pixel angle from the optical axis (rad) of an OPENCV_FISHEYE camera,
    COLMAP pixel convention (pixel centre at +0.5)."""
    fx, fy, cx, cy, k1, k2, k3, k4 = p
    t = np.linspace(0.0, np.pi, 200001)
    t2 = t * t
    td = t * (1 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4))))
    mono = np.nonzero(np.diff(td) <= 0)[0]
    n = mono[0] + 1 if len(mono) else len(t)  # monotonic (invertible) part
    v, u = np.mgrid[0:H, 0:W].astype(np.float64)
    delta = np.hypot((u + 0.5 - cx) / fx, (v + 0.5 - cy) / fy)
    return np.interp(delta, td[:n], t[:n], right=np.inf)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="BA dataset (colmap/, colmap_ba/, raymap_fisheye_cam*.npy)")
    ap.add_argument("--images", required=True, help="unbaked images folder, as the symlink target from <out>")
    ap.add_argument("--person_masks", required=True,
                    help="photographer mask pattern with {lens} (front/rear) and {stem} (image name without _<idx>)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--theta_max_deg", type=float, default=90.0)
    ap.add_argument("--dilate_px", type=int, default=16, help="grow the photographer region (image px)")
    a = ap.parse_args()

    cams, images, _ = read_text_model(os.path.join(a.src, "colmap_ba"))
    os.makedirs(os.path.join(a.out, "masks"), exist_ok=True)
    src_rel = os.path.relpath(os.path.abspath(a.src), os.path.abspath(a.out))
    links = {"colmap": os.path.join(src_rel, "colmap"), "resized_images": a.images}
    for c in cams:
        links[f"raymap_fisheye_cam{c}.npy"] = os.path.join(src_rel, f"raymap_fisheye_cam{c}.npy")
    for name, target in links.items():
        dst = os.path.join(a.out, name)
        if os.path.lexists(dst):
            os.remove(dst)
        os.symlink(target, dst)
    images_dir = os.path.join(a.out, "resized_images")

    circles = {}
    for c, (_, W, H, p) in cams.items():
        circles[c] = theta_map(W, H, p) <= np.radians(a.theta_max_deg)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * a.dilate_px + 1, 2 * a.dilate_px + 1))

    used = []
    for im in sorted(images.values(), key=lambda m: m["name"]):
        stem_full = os.path.splitext(im["name"])[0]
        stem = stem_full.rsplit("_", 1)[0]
        img = cv2.imread(os.path.join(images_dir, im["name"]), cv2.IMREAD_REDUCED_GRAYSCALE_2)
        if img is None:
            sys.exit(f"cannot read {os.path.join(images_dir, im['name'])}")
        circle = circles[im["cam"]]
        if (2 * img.shape[0], 2 * img.shape[1]) != circle.shape:
            sys.exit(f"{im['name']}: image {2 * img.shape[1]}x{2 * img.shape[0]} != camera {circle.shape[1]}x{circle.shape[0]}")
        person = cv2.imread(a.person_masks.format(lens=LENS_NAMES[im["cam"]], stem=stem), cv2.IMREAD_GRAYSCALE)
        if person is None:
            sys.exit(f"missing photographer mask for {im['name']}: "
                     f"{a.person_masks.format(lens=LENS_NAMES[im['cam']], stem=stem)}")
        person_region = cv2.dilate((person <= 127).astype(np.uint8), kernel) > 0
        keep = circle & ~person_region
        cv2.imwrite(os.path.join(a.out, "masks", stem_full + ".png"), keep.astype(np.uint8) * 255)
        used.append((keep.mean(), person_region[circle].mean()))

    used = np.array(used)
    summary = (f"{len(used)} masks, theta_max {a.theta_max_deg} deg, photographer dilation {a.dilate_px} px; "
               f"used fraction of the square: mean {used[:, 0].mean():.3f} (min {used[:, 0].min():.3f}); "
               f"photographer share of the circle: mean {used[:, 1].mean():.3f}, max {used[:, 1].max():.3f}")
    print(summary)
    with open(os.path.join(a.out, "README.txt"), "w") as f:
        f.write(f"""{os.path.basename(os.path.normpath(a.out))}: {os.path.basename(os.path.normpath(a.src))} (per-lens BA) with
unbaked images and per-image loss masks, made by tools/x5_masked_dataset.py:

  python tools/x5_masked_dataset.py --src {a.src} --images {a.images} \\
      --person_masks '{a.person_masks}' \\
      --theta_max_deg {a.theta_max_deg} --dilate_px {a.dilate_px} --out {a.out}

masks/<image_name>.png (white = used) = lens circle up to {a.theta_max_deg} deg from the
axis, per lens from colmap_ba/cameras.txt, minus the photographer mask grown by
{a.dilate_px} px. {summary}.

Train with --masks masks and WITHOUT --mask_path, --render_model KB,
--raymap_path <this dir>/raymap_fisheye_cam{{cam_id}}.npy.
""")


if __name__ == "__main__":
    main()
