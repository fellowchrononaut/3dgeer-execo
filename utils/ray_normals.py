# ported/adapted from GaussianWrapping gaussian_wrapping/densification/normal_error.py
# (view<->world normal convention) and gaussian_wrapping/utils/geometry_utils.py
# (finite-difference depth-to-normal idea); ray directions are 3DGEER-specific
# (mirror the PBF/PH/KB-EQ ray formulas in cuda_rasterizer/forward.cu::renderCUDA).
import torch
import torch.nn.functional as F


def get_ray_dirs_view(camera):
    """(3,H,W) unit ray directions in view space. Cached on the camera."""
    if getattr(camera, "_ray_dirs_view", None) is not None:
        return camera._ray_dirs_view
    H, W = camera.image_height, camera.image_width
    if camera.render_model == 1:          # KB/EQ: per-pixel raymap (H,W,3)
        dirs = camera.raymap.cuda().permute(2, 0, 1).float()
    elif camera.render_model == 2:        # PH: analytic pinhole rays
        ys, xs = torch.meshgrid(torch.arange(H, device="cuda"),
                                 torch.arange(W, device="cuda"), indexing="ij")
        dirs = torch.stack([(xs + 0.5) / camera.focal_x - W / (2.0 * camera.focal_x),
                             (ys + 0.5) / camera.focal_y - H / (2.0 * camera.focal_y),
                             torch.ones_like(xs, dtype=torch.float32)], dim=0)
    else:                                 # BEAP: tan grids
        tt = camera.tan_theta.cuda(); tp = camera.tan_phi.cuda()
        dirs = torch.stack([tt[None, :].expand(H, W), tp[:, None].expand(H, W),
                             torch.ones(H, W, device="cuda")], dim=0)
    camera._ray_dirs_view = F.normalize(dirs, dim=0)
    return camera._ray_dirs_view


@torch.no_grad()
def depth_to_normals_via_rays(camera, dist_map):
    """dist_map (1,H,W) Euclidean distance along the ray (median depth).
    Returns view-space unit normals (3,H,W) + validity mask (H,W)."""
    dirs = get_ray_dirs_view(camera)                     # (3,H,W)
    pts = dirs * dist_map                                # (3,H,W)
    dx = pts[:, :, 1:] - pts[:, :, :-1]
    dy = pts[:, 1:, :] - pts[:, :-1, :]
    dx = F.pad(dx, (0, 1, 0, 0)); dy = F.pad(dy, (0, 0, 0, 1))
    n = F.normalize(torch.cross(dx, dy, dim=0), dim=0)
    # orient toward camera: n.ray must be negative
    flip = (n * dirs).sum(0, keepdim=True) > 0
    n = torch.where(flip, -n, n)
    valid = (dist_map > 0).squeeze(0)
    # also invalidate pixels whose right/bottom neighbor had no depth
    v = valid.clone(); v[:, -1] = False; v[-1, :] = False
    v = v & valid.roll(-1, dims=1) & valid.roll(-1, dims=0)
    return torch.where(v[None], n, torch.zeros_like(n)), v


def view_normals_to_world(camera, n):
    """(3,H,W) view-space -> world-space. world_view_transform is stored
    transposed (Inria convention: v_view = v_world_row @ W2V), so rows of
    world_view_transform[:3,:3] are the V2W rotation's *columns* transposed
    the other way -- validated by tests/test_normal_field.py check 4, which
    pins whichever of (R @ n) / (R.T @ n) reproduces GW's convention."""
    R = camera.world_view_transform[:3, :3]              # (3,3), transposed storage
    return torch.einsum("ij,jhw->ihw", R, n)             # == R_v2w @ n
