"""Session G diagnostic: per-gaussian gradient scale of each loss on the
healthy sessD_full30k checkpoint. Photometric vs 0.02*geo vs 0.6*ncc,
identical wiring to train.py's multiview block. Run in geer from /home."""
import os
import torch
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams
from scene import Scene, GaussianModel
from gaussian_renderer import render, render_normal_field
from utils.loss_utils import l1_loss, ssim
from utils.multiview import compute_nearest_cameras, geo_loss, eq_warp_patch_ncc, ref_to_neighbor_RT
from utils.ray_normals import get_ray_dirs_view, world_to_view_normals

parser = ArgumentParser(); lp = ModelParams(parser); pp = PipelineParams(parser)
args = parser.parse_args(["-m", "/home/output/sessD_full30k", "-s", "/home/data/tt/datasets/truck",
    "--dataset", "COLMAP", "--camera_model", "PINHOLE", "--resolution", "1", "--eval"])
d = lp.extract(args); pipe = pp.extract(args)
d.render_model = "PH"; d.fov_mod = None; d.sample_step = 0.002
d.focal_scaling = 1.0; d.distortion_scaling = 1.0; d.mirror_shift = 0.0; d.raymap = None
g = GaussianModel(d.sh_degree); scene = Scene(d, g, load_iteration=30000, shuffle=False)
cams = scene.getTrainCameras()
cam = cams[0]
nearest = compute_nearest_cameras(cams, scene_radius=scene.cameras_extent)
neighbor = cams[nearest[0]["nearest_id"][0]]
bg = torch.zeros(3, device="cuda")
print(f"ref {cam.image_name} neighbor {neighbor.image_name}, P={g.get_xyz.shape[0]}")


def zero():
    for p in [g._xyz, g._opacity, g._scaling, g._rotation, g._features_dc, g._features_rest]:
        p.grad = None


def stats(name):
    gx = g._xyz.grad.norm(dim=-1) if g._xyz.grad is not None else torch.zeros(1)
    go = g._opacity.grad.abs().view(-1) if g._opacity.grad is not None else torch.zeros(1)
    n_nan = int((~torch.isfinite(gx)).sum()) + int((~torch.isfinite(go)).sum())
    sx = gx[gx > 0]; so = go[go > 0]
    def q(t, p): return t.quantile(p).item() if t.numel() else 0.0
    print(f"{name:12s} NAN/INF={n_nan} | xyz: n={sx.numel():8d} p50={q(sx,0.5):.3e} p99={q(sx,0.99):.3e} "
          f"max={sx.max().item() if sx.numel() else 0:.3e} | "
          f"opa: n={so.numel():8d} p50={q(so,0.5):.3e} p99={q(so,0.99):.3e} "
          f"max={so.max().item() if so.numel() else 0:.3e}")
    return q(sx, 0.5), (sx.max().item() if sx.numel() else 0.0)


# --- A: photometric (train.py weights: 0.8 L1 + 0.2 (1-ssim)) ---
zero()
pkg = render(cam, g, pipe, bg)
gt = cam.sampled_image.cuda()
loss = 0.8 * l1_loss(pkg["render"], gt) + 0.2 * (1.0 - ssim(pkg["render"], gt))
loss.backward()
photo_p50, _ = stats("photo")

# --- B: geo only, weight 0.02 ---
zero()
pkg = render(cam, g, pipe, bg)
with torch.no_grad():
    npkg = render(neighbor, g, pipe, bg)
md = pkg["median_depth"]
gl, gmask, gweights = geo_loss(cam, neighbor, md, npkg, ref_depth_valid=md > 0,
                               pixel_noise_th=1.0, znear_relative=0.02,
                               scene_radius=scene.cameras_extent)
print(f"geo_loss={gl.item():.4f} mask_frac={gmask.float().mean().item():.3f}")
(0.02 * gl).backward()
stats("geo*0.02")

# --- C: ncc only, weight 0.6 (exact train.py block) ---
zero()
pkg = render(cam, g, pipe, bg)
with torch.no_grad():
    npkg = render(neighbor, g, pipe, bg)
md = pkg["median_depth"]
gl, gmask, gweights = geo_loss(cam, neighbor, md, npkg, ref_depth_valid=md > 0,
                               pixel_noise_th=1.0, znear_relative=0.02,
                               scene_radius=scene.cameras_extent)
shape_map = render_normal_field(cam, g, pipe, shape=True)
nmv = world_to_view_normals(cam, shape_map)
H, W = cam.image_height, cam.image_width
ys, xs = torch.meshgrid(torch.arange(H, device="cuda"), torch.arange(W, device="cuda"), indexing="ij")
vidx = gmask.view(-1).nonzero(as_tuple=True)[0]
depths_sel = md.view(-1)[vidx]
normals_sel = nmv.reshape(3, -1)[:, vidx].transpose(0, 1).contiguous()
pixels_sel = torch.stack([xs.reshape(-1)[vidx], ys.reshape(-1)[vidx]], dim=-1).int()
weights_sel = gweights.view(-1)[vidx]
rays = get_ray_dirs_view(cam).permute(1, 2, 0).contiguous()
R_rn, T_rn = ref_to_neighbor_RT(cam, neighbor)
ncc, nvalid = eq_warp_patch_ncc(depths_sel, normals_sel, pixels_sel, rays, R_rn, T_rn,
                                cam.gray_image.squeeze(0).cuda(), neighbor.gray_image.squeeze(0).cuda(),
                                neighbor.render_model, neighbor.focal_x, neighbor.focal_y,
                                neighbor.principal_x, neighbor.principal_y, 3)
ncc_term = torch.clamp(1.0 - ncc, 0.0, 2.0)
nmask = (ncc_term < 0.9) & nvalid
ncc_loss = (ncc_term * weights_sel)[nmask].mean()
print(f"ncc_loss={ncc_loss.item():.4f} n_pts={int(nmask.sum())}")
(0.6 * ncc_loss).backward()
stats("ncc*0.6")

# --- D: geo again, top offenders ---
zero()
pkg = render(cam, g, pipe, bg)
with torch.no_grad():
    npkg = render(neighbor, g, pipe, bg)
md = pkg["median_depth"]
gl, gmask, _ = geo_loss(cam, neighbor, md, npkg, ref_depth_valid=md > 0,
                        pixel_noise_th=1.0, znear_relative=0.02,
                        scene_radius=scene.cameras_extent)
(0.02 * gl).backward()
gx = g._xyz.grad.norm(dim=-1)
top = gx.topk(10).values
print("geo top-10 xyz grad norms:", [f"{v:.2e}" for v in top.tolist()])
print(f"geo: n gaussians with |grad| > 100x photo_p50({photo_p50:.1e}): "
      f"{int((gx > 100 * photo_p50).sum())}")
