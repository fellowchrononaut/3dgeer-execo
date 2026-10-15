"""Per-lens bundle adjustment of a dual-fisheye COLMAP text model (pycolmap).

Splits a model whose images all share one OPENCV_FISHEYE camera into one camera
per lens (rear = image-name suffix >= --rear_from, e.g. the X5's _103.._205),
then refines per lens fx, fy, cx, cy, k1..k4 together with all poses and
points: a robust (soft-L1) pass, removal of observations above --max_error px,
and a second pass. Writes a training dataset:

  <out>/colmap_ba/        consistent BA model at the observation resolution
  <out>/colmap/           what scene/dataset_readers.py reads: cameras(_fish).txt
                          with one camera per lens, intrinsics scaled to
                          --train_size, images.txt (observations stay at the
                          observation resolution, as in the input datasets),
                          points3D.txt, points3D.ply
  <out>/raymap_fisheye_cam{0,1}.npy   for --raymap_path <out>/raymap_fisheye_cam{cam_id}.npy
  <out>/ba_report.txt     reprojection error before / after, per lens, by angle

The gauge is fixed by holding image 0's pose and one position coordinate of a
second image constant, so the world frame and scale stay those of the input.

Must run where pycolmap is installed (the geer container):
  python tools/colmap_ba_fisheye.py --colmap data/x5_eq/colmap \\
      --init_k 0.02431 0.01870 -0.00645 -0.00008 --rear_from 103 \\
      --train_size 1920 --out data/x5_ba_1920
"""
import argparse
import os
import shutil
import sys

import numpy as np
import pycolmap

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fisheye_raymap import main_from_colmap  # noqa: E402

LENS_NAMES = {0: "front", 1: "rear"}


# --- text model I/O (input models come from Metashape exports, not COLMAP) ---

def read_text_model(d):
    cams = {}
    for line in open(os.path.join(d, "cameras.txt")):
        if line.startswith("#") or not line.strip():
            continue
        t = line.split()
        cams[int(t[0])] = (t[1], int(t[2]), int(t[3]), np.array(list(map(float, t[4:]))))
    lines = [l for l in open(os.path.join(d, "images.txt")) if not l.startswith("#")]
    images = {}
    for i in range(0, len(lines) - 1, 2):
        h = lines[i].split()
        if len(h) < 10:
            continue
        obs = np.array(lines[i + 1].split(), dtype=float).reshape(-1, 3)
        images[int(h[0])] = dict(q=np.array(list(map(float, h[1:5]))), t=np.array(list(map(float, h[5:8]))),
                                 cam=int(h[8]), name=h[9], obs=obs)
    points = {}
    for line in open(os.path.join(d, "points3D.txt")):
        if line.startswith("#") or not line.strip():
            continue
        t = line.split()
        points[int(t[0])] = (np.array(list(map(float, t[1:4]))), np.array(list(map(float, t[4:7]))))
    return cams, images, points


def qvec2rot(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
        [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
        [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y]])


def project_fisheye(Xc, p):
    fx, fy, cx, cy, k1, k2, k3, k4 = p
    rho = np.hypot(Xc[:, 0], Xc[:, 1])
    th = np.arctan2(rho, Xc[:, 2])
    t2 = th * th
    thd = th * (1 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4))))
    s = np.where(rho > 1e-12, thd / np.maximum(rho, 1e-12), 0.0)
    return np.stack([fx * Xc[:, 0] * s + cx, fy * Xc[:, 1] * s + cy], 1), th


def lens_of(name, rear_from):
    return 1 if int(os.path.splitext(name)[0].rsplit("_", 1)[1]) >= rear_from else 0


def reprojection_stats(d, rear_from):
    """Per-observation error of a text model, independent of pycolmap."""
    cams, images, points = read_text_model(d)
    E, TH, LENS = [], [], []
    for im in images.values():
        obs = im["obs"][im["obs"][:, 2] >= 0]
        obs = obs[[int(p) in points for p in obs[:, 2]]]
        if not len(obs):
            continue
        X = np.stack([points[int(p)][0] for p in obs[:, 2]])
        Xc = X @ qvec2rot(im["q"]).T + im["t"]
        uv, th = project_fisheye(Xc, cams[im["cam"]][3])
        E.append(np.linalg.norm(uv - obs[:, :2], axis=1))
        TH.append(np.degrees(th))
        LENS.append(np.full(len(obs), lens_of(im["name"], rear_from)))
    return np.concatenate(E), np.concatenate(TH), np.concatenate(LENS), cams, images, points


def format_stats(title, E, TH, LENS, scale, f):
    out = [f"{title}  (px at the training size; mrad = px / f)",
           "                  n      median   mean    p90    mean mrad"]
    for L in (0, 1, None):
        m = np.ones_like(E, bool) if L is None else LENS == L
        e = E[m] * scale
        out.append(f"  {('both' if L is None else LENS_NAMES[L]):6s}  {m.sum():8d}   {np.median(e):6.3f} {e.mean():6.3f} "
                   f"{np.percentile(e, 90):6.3f}   {1000 * e.mean() / f:6.3f}")
    out.append("  by angle from the optical axis (median px)   front   rear")
    for a, b in zip([0, 30, 50, 70, 80, 90], [30, 50, 70, 80, 90, 100]):
        m = (TH >= a) & (TH < b)
        if m.sum() < 20:
            continue
        med = [np.median(E[m & (LENS == L)]) * scale if (m & (LENS == L)).sum() > 10 else float("nan") for L in (0, 1)]
        out.append(f"    {a:3d}-{b:3d} deg                              {med[0]:6.3f}  {med[1]:6.3f}")
    return "\n".join(out)


def write_split_model(src, dst, init_k, rear_from):
    """The input model with its single camera duplicated into one per lens."""
    os.makedirs(dst, exist_ok=True)
    cams, images, points = read_text_model(src)
    if len(cams) != 1:
        sys.exit(f"expected one shared camera in {src}, found {len(cams)}")
    model, W, H, p = next(iter(cams.values()))
    if model != "OPENCV_FISHEYE":
        sys.exit(f"camera model {model} is not OPENCV_FISHEYE")
    p = p.copy()
    if init_k is not None:
        p[4:8] = init_k
    with open(os.path.join(dst, "cameras.txt"), "w") as f:
        for c in (0, 1):
            f.write(f"{c} OPENCV_FISHEYE {W} {H} " + " ".join(repr(float(v)) for v in p) + "\n")
    with open(os.path.join(dst, "images.txt"), "w") as f:
        for i, im in sorted(images.items()):
            f.write(f"{i} " + " ".join(repr(float(v)) for v in (*im["q"], *im["t"]))
                    + f" {lens_of(im['name'], rear_from)} {im['name']}\n")
            f.write(" ".join(f"{float(x)!r} {float(y)!r} {int(pid)}" for x, y, pid in im["obs"]) + "\n")
    shutil.copy(os.path.join(src, "points3D.txt"), os.path.join(dst, "points3D.txt"))
    return W, H


def run_ba(rec, loss, loss_scale, fixed_image, scale_image):
    opts = pycolmap.BundleAdjustmentOptions()
    opts.refine_focal_length = True
    opts.refine_principal_point = True
    opts.refine_extra_params = True
    opts.refine_extrinsics = True
    opts.loss_function_type = loss
    opts.loss_function_scale = loss_scale
    opts.print_summary = False
    cfg = pycolmap.BundleAdjustmentConfig()
    for image_id in rec.reg_image_ids():
        cfg.add_image(image_id)
    cfg.set_constant_cam_pose(fixed_image)
    cfg.set_constant_cam_positions(scale_image, [0])
    ba = pycolmap.BundleAdjuster(opts, cfg)
    # solve() returns a ceres Summary, which this pycolmap build cannot convert
    # to Python (no pyceres); the before/after error is reported from the model.
    try:
        ba.solve(rec)
    except TypeError as e:
        if "ceres::Solver::Summary" not in str(e):
            raise


def filter_observations(rec, max_error):
    """Delete observations whose reprojection error exceeds max_error px."""
    n = 0
    for image_id in rec.reg_image_ids():
        image = rec.image(image_id)
        cam = rec.camera(image.camera_id)
        cam_from_world = image.cam_from_world
        for idx, p2d in enumerate(image.points2D):
            if not p2d.has_point3D():
                continue
            xyz = rec.point3D(p2d.point3D_id).xyz
            uv = cam.img_from_cam(cam_from_world * xyz)
            if uv is None or np.linalg.norm(np.asarray(uv) - p2d.xy) > max_error:
                rec.delete_observation(image_id, idx)
                n += 1
    return n


def write_training_colmap(ba_dir, out_colmap, scale):
    """cameras.txt / cameras_fish.txt at the training size; images/points as the BA model."""
    os.makedirs(out_colmap, exist_ok=True)
    cams, _, points = read_text_model(ba_dir)
    lines = ["# Camera list with one line of data per camera:",
             "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]",
             f"# Number of cameras: {len(cams)}"]
    for c, (model, W, H, p) in sorted(cams.items()):
        q = p.copy()
        q[:4] *= scale  # same convention as the earlier x5 datasets: c_train = c_obs * scale
        lines.append(f"{c} {model} {round(W * scale)} {round(H * scale)} " + " ".join(repr(float(v)) for v in q))
    for fn in ("cameras.txt", "cameras_fish.txt"):
        open(os.path.join(out_colmap, fn), "w").write("\n".join(lines) + "\n")
    for fn in ("images.txt", "points3D.txt"):
        shutil.copy(os.path.join(ba_dir, fn), os.path.join(out_colmap, fn))
    from plyfile import PlyData, PlyElement
    xyz = np.stack([v[0] for v in points.values()])
    rgb = np.stack([v[1] for v in points.values()])
    if rgb.max() <= 1.0:
        rgb = rgb * 255.0
    v = np.empty(len(xyz), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
                                  ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    v["x"], v["y"], v["z"] = xyz.T
    v["nx"] = v["ny"] = v["nz"] = 0
    v["red"], v["green"], v["blue"] = np.clip(rgb, 0, 255).astype(np.uint8).T
    PlyData([PlyElement.describe(v, "vertex")]).write(os.path.join(out_colmap, "points3D.ply"))


def rig_report(images_txt, rear_from):
    """Spread of the front->rear relative pose over simultaneous frames (not enforced by this BA)."""
    from scipy.spatial.transform import Rotation as Rot
    lines = [l for l in open(images_txt) if not l.startswith("#")]
    P = {}
    for i in range(0, len(lines) - 1, 2):
        h = lines[i].split()
        if len(h) < 10:
            continue
        q = list(map(float, h[1:5]))
        P[int(os.path.splitext(h[9])[0].rsplit("_", 1)[1])] = (
            Rot.from_quat([q[1], q[2], q[3], q[0]]).as_matrix(), np.array(list(map(float, h[5:8]))))
    pairs = [(P[i], P[i + rear_from]) for i in range(rear_from) if i in P and i + rear_from in P]
    R = Rot.from_matrix(np.stack([r[0] @ f[0].T for f, r in pairs]))
    t = np.stack([r[1] - (r[0] @ f[0].T) @ f[1] for f, r in pairs])
    d = np.degrees((R.mean().inv() * R).magnitude())
    tn = np.linalg.norm(t, axis=1)
    return (f"rig (front->rear) over {len(pairs)} frames: rotation deviation from mean median {np.median(d):.3f} deg, "
            f"p90 {np.percentile(d, 90):.3f} deg; baseline |t| mean {tn.mean():.4f}, std {tn.std():.4f} "
            f"(spread of t {np.linalg.norm(t - t.mean(0), axis=1).mean():.4f}) scene units")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--colmap", required=True, help="input text model (one shared OPENCV_FISHEYE camera)")
    ap.add_argument("--init_k", type=float, nargs=4, default=None, help="initial k1..k4 (default: the input's)")
    ap.add_argument("--rear_from", type=int, default=103, help="image-name suffix of the first rear-lens image")
    ap.add_argument("--train_size", type=int, default=1920, help="width of the training images / ray maps")
    ap.add_argument("--max_error", type=float, default=4.0, help="drop observations above this (obs px) after pass 1")
    ap.add_argument("--loss_scale", type=float, default=1.0, help="soft-L1 scale of pass 1 (obs px)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    split_dir = os.path.join(a.out, "colmap_split_init")
    ba_dir = os.path.join(a.out, "colmap_ba")
    W, _ = write_split_model(a.colmap, split_dir, a.init_k, a.rear_from)
    scale = a.train_size / W

    E0, TH0, L0, cams0, _, _ = reprojection_stats(split_dir, a.rear_from)
    f_train = cams0[0][3][0] * scale
    report = [f"input {os.path.abspath(a.colmap)}, observations at {W} px, training size {a.train_size} px",
              format_stats("BEFORE (per-lens split, initial k)", E0, TH0, L0, scale, f_train),
              rig_report(os.path.join(split_dir, "images.txt"), a.rear_from)]
    print("\n".join(report))

    rec = pycolmap.Reconstruction()
    rec.read_text(split_dir)
    ids = sorted(rec.reg_image_ids())
    fixed_image, scale_image = ids[0], ids[len(ids) // 4]
    print(rec.summary())

    print("\n=== pass 1: soft-L1 ===")
    run_ba(rec, pycolmap.LossFunctionType.SOFT_L1, a.loss_scale, fixed_image, scale_image)
    n_obs = rec.compute_num_observations()
    n_del = filter_observations(rec, a.max_error)
    report.append(f"removed {n_del} of {n_obs} observations above {a.max_error} px after pass 1 "
                  f"({rec.num_points3D()} points left)")
    print(report[-1])
    print("\n=== pass 2: squared loss ===")
    run_ba(rec, pycolmap.LossFunctionType.TRIVIAL, 1.0, fixed_image, scale_image)

    os.makedirs(ba_dir, exist_ok=True)
    rec.write_text(ba_dir)
    E1, TH1, L1, cams1, _, _ = reprojection_stats(ba_dir, a.rear_from)
    report.append(format_stats("AFTER per-lens BA", E1, TH1, L1, scale, f_train))
    report.append(rig_report(os.path.join(ba_dir, "images.txt"), a.rear_from))
    report.append(f"mean track length {rec.compute_mean_track_length():.3f}")
    report.append("intrinsics at the observation size (fx fy cx cy k1 k2 k3 k4):")
    for c in (0, 1):
        report.append(f"  {LENS_NAMES[c]:5s} (camera {c}): " + " ".join(f"{v:.6g}" for v in cams1[c][3]))

    write_training_colmap(ba_dir, os.path.join(a.out, "colmap"), scale)
    main_from_colmap(os.path.join(a.out, "colmap", "cameras_fish.txt"),
                     os.path.join(a.out, "raymap_fisheye_cam{cam_id}.npy"))
    shutil.rmtree(split_dir)

    text = "\n\n".join(report) + "\n"
    open(os.path.join(a.out, "ba_report.txt"), "w").write(text)
    print("\n" + text)


if __name__ == "__main__":
    main()
