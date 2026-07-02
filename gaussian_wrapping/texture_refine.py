# ported/adapted from GaussianWrapping gaussian_wrapping/texture_mesh.py
# (per-vertex-color Adam optimization against rendered GT images, L1+SSIM
# loss) -- rewritten on top of a minimal, self-contained nvdiffrast
# differentiable rasterizer instead of porting GW's full scene/mesh.py
# `Meshes` / `MeshRasterizer` / `MeshRenderer` / `ScalableMeshRenderer`
# machinery (1387 lines; frustum culling, batched multi-mesh support,
# normals/depth AOVs, etc. -- none of which texture_mesh.py's own training
# loop actually needs beyond `nvdiff_rasterization` + `dr.interpolate` +
# `dr.antialias`, see scene/mesh.py:23-60 and :1149-1190).
"""CLI: per-vertex-color texture refinement (GUTWrap Path A, Session E3,
GW's 3rd extraction-pipeline stage). Optimizes RGB colors at each vertex of
an input mesh (e.g. our own pivot-MTet or PAM output) so that a
differentiable nvdiffrast Gouraud rasterization of the mesh matches the
trained 3DGEER splat's own rendered RGB images at the training cameras
(L1 + SSIM, exactly GW's loss).

REQUIRES nvdiffrast (`pip install --no-build-isolation
/home/submodules/GaussianWrapping/submodules/nvdiffrast` inside the `geer`
container -- built successfully in Session E3, pure-CUDA rasterizer
backend, no EGL/OpenGL headers needed: this fork's setup.py only compiles
the `cudaraster` sources, and `RasterizeGLContext` itself is deprecated to
alias `RasterizeCudaContext` internally, see nvdiffrast/torch/ops.py).

Output: `<mesh>_texture_refined.ply` -- per-vertex RGB colors (NOT a UV
texture atlas; GW's own texture_mesh.py produces the same Gouraud
per-vertex representation, despite the "texture" naming -- there is no
`texture.png`/UV-unwrap step anywhere in the ported reference file).

Example (truck, PH mode, from the Session E3 PAM mesh):
    python gaussian_wrapping/texture_refine.py \\
        --model_path /home/output/sessD_full30k \\
        --source_path /home/data/tt/datasets/truck \\
        --dataset COLMAP --camera_model PINHOLE --render_model PH \\
        --iteration 30000 \\
        --mesh /home/output/sessE_mesh/truck_pam.ply \\
        --n_iter 1000

DEVIATIONS from the on-disk GaussianWrapping implementation (documented per
Session E3 hard rules):
1. Differentiable rasterization: a ~30-line `_rasterize_and_shade` (below)
   replaces GW's `Meshes`/`MeshRasterizer`/`MeshRenderer`/
   `ScalableMeshRenderer` stack -- same core math (`pos = verts_h @
   camera.full_proj_transform`; `dr.rasterize` -> `dr.interpolate` ->
   `dr.antialias`), no frustum culling (truck's cameras all see the full
   object; `frustum_cull_mesh` in GW is an optimization, not a correctness
   requirement) or scalable/chunked-face variant (not needed at truck's
   mesh face counts).
2. Loss: `fused_ssim` (a separate pip package GW depends on) is not
   installed in the `geer` container; this port uses OUR OWN
   `utils/loss_utils.py::ssim` (windowed SSIM, already used elsewhere in
   this codebase, e.g. train.py) instead -- numerically different
   implementation, same L1+SSIM loss *shape*.
3. **Vertex-color initialization** (not present in GW's own script -- GW
   relies on `pivot_based_mesh_extraction.py` having already written
   `evaluate_mesh_colors_all_vertices`-based vertex colors onto the input
   mesh before `texture_mesh.py` ever runs). OUR `extract_mesh.py` /
   `pam_extraction.py` meshes carry NO vertex colors. This port adds
   `_init_vertex_colors`: for each vertex, average the RGB of every
   training camera that observes it with `|view-space depth - rendered
   median_depth at that pixel| < tolerance` (i.e. an un-occluded
   observation) -- reusing the SAME median-depth machinery already in this
   repo (`gaussian_wrapping/fields.py`) instead of porting GW's own
   TSDF-based `evaluate_mesh_colors`/`evaluate_mesh_colors_all_vertices`
   (`regularization/sdf/depth_fusion.py`). Vertices with zero unoccluded
   observations fall back to the mean color of all covered vertices.
4. `--use_scalable_renderer` is dropped (see deviation #1).
"""
import os
import random
import sys
import time
from argparse import ArgumentParser
from random import randint

import numpy as np
import torch
import torch.nn.functional as F
import trimesh

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import nvdiffrast.torch as dr
except ImportError as e:
    raise ImportError(
        "gaussian_wrapping/texture_refine.py requires nvdiffrast. Build it inside "
        "the geer container with:\n"
        "    pip install --no-build-isolation "
        "/home/submodules/GaussianWrapping/submodules/nvdiffrast\n"
        "(pure-CUDA rasterizer backend, no EGL/OpenGL headers needed; see this "
        "file's module docstring)."
    ) from e

from gaussian_renderer import render  # noqa: E402
from utils.loss_utils import l1_loss, ssim  # noqa: E402
from gaussian_wrapping.extract_mesh import build_scene  # noqa: E402
from gaussian_wrapping.fields import world_to_view  # noqa: E402
from gaussian_wrapping.fisheye_proj import project_view_to_pixel  # noqa: E402


# ---------------------------------------------------------------------------
# Minimal differentiable Gouraud rasterizer (see module docstring deviation #1)
# ---------------------------------------------------------------------------

def _rasterize_and_shade(glctx, camera, verts: torch.Tensor, faces_i32: torch.Tensor, features: torch.Tensor):
    """Rasterize `verts`/`faces_i32` from `camera` and antialiased-interpolate
    per-vertex `features` (N,C). Returns (C,H,W). Ported math from GW's
    scene/mesh.py::nvdiff_rasterization + MeshRenderer.forward's
    interpolate/antialias calls (same `full_proj_transform` row-vector
    convention as the rest of this repo's camera handling)."""
    H, W = camera.image_height, camera.image_width
    ones = torch.ones(verts.shape[0], 1, device=verts.device, dtype=verts.dtype)
    pos = torch.cat([verts, ones], dim=-1)
    pos_clip = torch.matmul(pos, camera.full_proj_transform)[None]  # (1,N,4)
    rast_out, _ = dr.rasterize(glctx, pos_clip, faces_i32, resolution=[H, W])
    feat_img, _ = dr.interpolate(features[None], rast_out, faces_i32)
    feat_img = dr.antialias(feat_img, rast_out, pos_clip, faces_i32)
    return feat_img[0].permute(2, 0, 1)  # (C,H,W)


# ---------------------------------------------------------------------------
# Vertex-color initialization (see module docstring deviation #3)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _render_rgb_and_depth(cameras, gaussians, pipe):
    background = torch.zeros(3, device="cuda")
    rgb_maps, depth_maps = [], []
    for cam in cameras:
        try:
            pkg = render(cam, gaussians, pipe, background)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            pkg = render(cam, gaussians, pipe, background)
        rgb_maps.append(pkg["render"].detach())
        depth_maps.append(pkg["median_depth"].detach())
    return rgb_maps, depth_maps


@torch.no_grad()
def _init_vertex_colors(verts: torch.Tensor, cameras, rgb_maps, depth_maps, depth_tol: float):
    N = verts.shape[0]
    color_sum = torch.zeros(N, 3, device=verts.device)
    weight_sum = torch.zeros(N, device=verts.device)

    for cam, rgb, dmap in zip(cameras, rgb_maps, depth_maps):
        x_view = world_to_view(cam, verts)
        u, v, valid = project_view_to_pixel(cam, x_view)
        H, W = cam.image_height, cam.image_width
        gx = (2.0 * u + 1.0) / W - 1.0
        gy = (2.0 * v + 1.0) / H - 1.0
        grid = torch.stack([gx, gy], dim=-1).view(1, N, 1, 2)

        dm = dmap.to(device=verts.device, dtype=verts.dtype).view(1, 1, H, W)
        d_obs = F.grid_sample(dm, grid, mode="bilinear", padding_mode="border", align_corners=False).view(N)
        r = x_view.norm(dim=-1)
        unoccluded = valid & (d_obs > 0) & ((r - d_obs).abs() < depth_tol)
        if not bool(unoccluded.any()):
            continue

        rgb_batched = rgb.to(device=verts.device, dtype=verts.dtype).unsqueeze(0)  # (1,3,H,W)
        sampled = F.grid_sample(rgb_batched, grid, mode="bilinear", padding_mode="border", align_corners=False)
        sampled = sampled.view(3, N).permute(1, 0)  # (N,3)

        w = unoccluded.float().unsqueeze(-1)
        color_sum += sampled * w
        weight_sum += unoccluded.float()

    has_color = weight_sum > 0
    colors = torch.where(has_color.unsqueeze(-1), color_sum / weight_sum.clamp(min=1e-8).unsqueeze(-1),
                          torch.zeros(N, 3, device=verts.device))
    if bool(has_color.any()):
        fallback = colors[has_color].mean(dim=0)
    else:
        fallback = torch.full((3,), 0.5, device=verts.device)
    colors = torch.where(has_color.unsqueeze(-1), colors, fallback.unsqueeze(0).expand(N, 3))
    return colors, int(has_color.sum().item())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = ArgumentParser(description="GUTWrap Session E3 -- per-vertex texture refinement")
    parser.add_argument("--model_path", "-m", required=True, type=str)
    parser.add_argument("--source_path", "-s", required=True, type=str)
    parser.add_argument("--iteration", type=int, default=30000)
    parser.add_argument("--dataset", type=str, default="COLMAP",
                         choices=["AUTO", "COLMAP", "BLENDER", "SCANNETPP", "MVL"])
    parser.add_argument("--camera_model", type=str, default="PINHOLE", choices=["FISHEYE", "PINHOLE"])
    parser.add_argument("--render_model", type=str, default="PH", choices=["BEAP", "KB", "EQ", "PH"])
    parser.add_argument("--sample_step", type=float, default=0.002)
    parser.add_argument("--focal_scaling", type=float, default=1.0)
    parser.add_argument("--distortion_scaling", type=float, default=1.0)
    parser.add_argument("--mirror_shift", type=float, default=0.0)
    parser.add_argument("--raymap_path", type=str, default=None)

    parser.add_argument("--mesh", type=str, required=True)
    parser.add_argument("--output", type=str, default=None,
                         help="Default: <mesh>_texture_refined.ply next to the input mesh.")
    parser.add_argument("--n_iter", type=int, default=1000)
    parser.add_argument("--lambda_dssim", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=0.0025)
    parser.add_argument("--init_depth_tol_factor", type=float, default=0.02,
                         help="Vertex-color-init unocclusion tolerance, as a fraction of scene_radius.")
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_path = args.output or (os.path.splitext(args.mesh)[0] + "_texture_refined.ply")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    dataset, opt, pipe, gaussians, scene = build_scene(
        source_path=args.source_path, model_path=args.model_path, dataset_type=args.dataset,
        camera_model=args.camera_model, iteration=args.iteration, render_model=args.render_model,
        sample_step=args.sample_step, focal_scaling=args.focal_scaling,
        distortion_scaling=args.distortion_scaling, mirror_shift=args.mirror_shift,
        raymap_path=args.raymap_path,
    )
    cameras = scene.getTrainCameras()
    print(f"[texture_refine] loaded {gaussians.get_xyz.shape[0]} Gaussians, {len(cameras)} cameras "
          f"from {args.model_path} @ iteration {args.iteration}")

    cam_centers = torch.stack([c.camera_center for c in cameras], dim=0)
    scene_radius = (cam_centers - cam_centers.mean(dim=0, keepdim=True)).norm(dim=-1).max().item() * 1.1

    t0 = time.time()
    rgb_maps, depth_maps = _render_rgb_and_depth(cameras, gaussians, pipe)
    print(f"[texture_refine] rendered {len(rgb_maps)} GT images in {time.time() - t0:.1f}s")

    print(f"[texture_refine] loading mesh from {args.mesh}")
    mesh = trimesh.load(args.mesh, process=False)
    verts = torch.from_numpy(np.asarray(mesh.vertices)).float().cuda()
    faces = torch.from_numpy(np.asarray(mesh.faces)).long().cuda()
    faces_i32 = faces.to(torch.int32).contiguous()
    print(f"[texture_refine] mesh: {verts.shape[0]} verts / {faces.shape[0]} faces")

    t0 = time.time()
    depth_tol = args.init_depth_tol_factor * scene_radius
    init_colors, n_covered = _init_vertex_colors(verts, cameras, rgb_maps, depth_maps, depth_tol)
    print(f"[texture_refine] vertex-color init: {n_covered}/{verts.shape[0]} vertices had an "
          f"unoccluded observation (tol={depth_tol:.4f}), {time.time() - t0:.1f}s")

    verts_colors = torch.nn.Parameter(init_colors.clone().requires_grad_(True))
    optimizer = torch.optim.Adam([{"params": [verts_colors], "lr": args.lr, "name": "verts_colors"}], lr=0.0, eps=1e-15)

    glctx = dr.RasterizeCudaContext()

    ema_loss = 0.0
    viewpoint_idx_stack = []
    t0 = time.time()
    for i_iter in range(args.n_iter):
        if not viewpoint_idx_stack:
            viewpoint_idx_stack = list(range(len(cameras)))
        viewpoint_idx = viewpoint_idx_stack.pop(randint(0, len(viewpoint_idx_stack) - 1))
        camera = cameras[viewpoint_idx]

        mesh_rgb = _rasterize_and_shade(glctx, camera, verts, faces_i32, verts_colors)
        gt_image = rgb_maps[viewpoint_idx]

        Ll1 = l1_loss(mesh_rgb, gt_image)
        ssim_value = ssim(mesh_rgb.unsqueeze(0), gt_image.unsqueeze(0))
        loss = (1.0 - args.lambda_dssim) * Ll1 + args.lambda_dssim * (1.0 - ssim_value)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            ema_loss = 0.4 * loss.item() + 0.6 * ema_loss
        if i_iter % 100 == 0 or i_iter == args.n_iter - 1:
            print(f"[texture_refine] iter {i_iter}/{args.n_iter} loss={loss.item():.5f} ema={ema_loss:.5f}")
        if i_iter % 200 == 0:
            torch.cuda.empty_cache()

    t_train = time.time() - t0
    print(f"[texture_refine] refinement done: {args.n_iter} iters in {t_train:.1f}s, final ema loss {ema_loss:.5f}")

    with torch.no_grad():
        final_colors = (verts_colors.detach().clamp(0.0, 1.0).cpu().numpy() * 255).astype(np.uint8)
    out_mesh = trimesh.Trimesh(
        vertices=verts.detach().cpu().numpy(), faces=faces.detach().cpu().numpy(),
        vertex_colors=final_colors, process=False,
    )
    out_mesh.export(output_path)
    print(f"[texture_refine] wrote {output_path}")

    stats_path = os.path.splitext(os.path.abspath(output_path))[0] + "_stats.txt"
    with open(stats_path, "w") as f:
        f.write(f"n_verts: {verts.shape[0]}\n")
        f.write(f"n_faces: {faces.shape[0]}\n")
        f.write(f"n_covered_init: {n_covered}\n")
        f.write(f"n_iter: {args.n_iter}\n")
        f.write(f"final_ema_loss: {ema_loss}\n")
        f.write(f"t_train_sec: {t_train}\n")
        f.write(f"output_path: {output_path}\n")


if __name__ == "__main__":
    main()
