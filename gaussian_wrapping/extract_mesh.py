# ported/adapted from GaussianWrapping gaussian_wrapping/pivot_based_mesh_extraction.py
# (CLI arg handling / __main__ pattern), scene construction copied from
# tests/test_normal_field.py::build_scene (Session C/E convention for
# standing up a Scene outside train.py).
"""CLI: extract a pivot-based marching-tetrahedra mesh from a trained 3DGEER
checkpoint (GUTWrap Path A, Session E).

Example (truck, PH mode, Session-D checkpoint):
    python gaussian_wrapping/extract_mesh.py \\
        --model_path /home/output/sessD_smoke3 \\
        --source_path /home/data/tt/datasets/truck \\
        --dataset COLMAP --camera_model PINHOLE --render_model PH \\
        --iteration 3000 \\
        --output /home/output/sessE_mesh/truck_mtet.ply \\
        --max_pivots 1200000
"""
import os
import sys
from argparse import ArgumentParser

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arguments import ModelParams, PipelineParams, OptimizationParams
from scene import Scene
from scene.gaussian_model import GaussianModel

from gaussian_wrapping.mesh_extraction import extract_mesh_pivot_mtet


def build_scene(source_path, model_path, dataset_type, camera_model, iteration,
                 render_model, sample_step=0.002, focal_scaling=1.0,
                 distortion_scaling=1.0, mirror_shift=0.0, raymap_path=None):
    parser = ArgumentParser()
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    args = parser.parse_args([
        "-s", source_path,
        "-m", model_path,
        "--dataset", dataset_type,
        "--camera_model", camera_model,
    ])
    dataset = lp.extract(args)
    opt = op.extract(args)
    pipe = pp.extract(args)

    # mirror train.py:40-49 / tests/test_normal_field.py::build_scene
    dataset.fov_mod = None
    dataset.sample_step = sample_step
    dataset.render_model = render_model
    dataset.focal_scaling = focal_scaling
    dataset.distortion_scaling = distortion_scaling
    dataset.mirror_shift = mirror_shift
    dataset.raymap = None
    if raymap_path is not None and os.path.exists(raymap_path):
        import numpy as np
        dataset.raymap = np.load(raymap_path)

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    return dataset, opt, pipe, gaussians, scene


def main():
    parser = ArgumentParser(description="GUTWrap Session E -- pivot-based mesh extraction")
    parser.add_argument("--model_path", "-m", required=True, type=str)
    parser.add_argument("--source_path", "-s", required=True, type=str)
    parser.add_argument("--iteration", type=int, default=30000)
    parser.add_argument("--output", "-o", required=True, type=str)
    parser.add_argument("--dataset", type=str, default="COLMAP",
                         choices=["AUTO", "COLMAP", "BLENDER", "SCANNETPP", "MVL"])
    parser.add_argument("--camera_model", type=str, default="PINHOLE",
                         choices=["FISHEYE", "PINHOLE"])
    parser.add_argument("--render_model", type=str, default="PH",
                         choices=["BEAP", "KB", "EQ", "PH"])
    parser.add_argument("--sample_step", type=float, default=0.002)
    parser.add_argument("--focal_scaling", type=float, default=1.0)
    parser.add_argument("--distortion_scaling", type=float, default=1.0)
    parser.add_argument("--mirror_shift", type=float, default=0.0)
    parser.add_argument("--raymap_path", type=str, default=None)

    parser.add_argument("--iso", type=float, default=0.0)
    parser.add_argument("--std_factor", type=float, default=3.0)
    parser.add_argument("--max_pivots", type=int, default=1_500_000)
    parser.add_argument("--trunc_margin", type=float, default=None)
    parser.add_argument("--n_binary_steps", type=int, default=8)
    parser.add_argument("--max_radius_factor", type=float, default=2.0,
                         help="Drop pivots farther than max_radius_factor * scene_radius "
                              "from the camera-center centroid before Delaunay (guards "
                              "against unbounded-scene background Gaussians exploding the "
                              "mesh bbox / connected-component filter). Set to a large "
                              "number or a negative value's absence (i.e. omit) to disable "
                              "-- pass e.g. 1e9 to effectively disable.")

    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    dataset, opt, pipe, gaussians, scene = build_scene(
        source_path=args.source_path,
        model_path=args.model_path,
        dataset_type=args.dataset,
        camera_model=args.camera_model,
        iteration=args.iteration,
        render_model=args.render_model,
        sample_step=args.sample_step,
        focal_scaling=args.focal_scaling,
        distortion_scaling=args.distortion_scaling,
        mirror_shift=args.mirror_shift,
        raymap_path=args.raymap_path,
    )
    print(f"[extract_mesh] loaded {gaussians.get_xyz.shape[0]} Gaussians from "
          f"{args.model_path} @ iteration {args.iteration}")

    cameras = scene.getTrainCameras()
    stats_path = os.path.join(os.path.dirname(os.path.abspath(args.output)), "stats.txt")

    extract_mesh_pivot_mtet(
        gaussians=gaussians,
        cameras=cameras,
        pipe=pipe,
        output_path=args.output,
        iso=args.iso,
        std_factor=args.std_factor,
        max_pivots=args.max_pivots,
        trunc_margin=args.trunc_margin,
        n_binary_steps=args.n_binary_steps,
        max_radius_factor=args.max_radius_factor,
        stats_path=stats_path,
    )


if __name__ == "__main__":
    main()
