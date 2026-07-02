# throwaway preview script (Session E2 QA) -- renders a mesh PLY headlessly
# with a simple z-buffered flat shader in torch (no GL needed), from a few
# orbit viewpoints + one real training-camera-like viewpoint.
import sys

import numpy as np
import torch
import trimesh
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ply_path = sys.argv[1]
out_png = sys.argv[2]

mesh = trimesh.load(ply_path, process=False)
V = torch.tensor(np.asarray(mesh.vertices), dtype=torch.float32, device="cuda")
F = torch.tensor(np.asarray(mesh.faces), dtype=torch.long, device="cuda")
print(f"loaded {V.shape[0]} verts / {F.shape[0]} faces")

center = V.median(dim=0).values
extent = (V.max(dim=0).values - V.min(dim=0).values).norm().item()


def render_view(azim_deg, elev_deg, dist_factor=0.7, res=900, n_samples=6, focus_radius=None):
    az, el = np.radians(azim_deg), np.radians(elev_deg)
    cam_dir = torch.tensor([np.cos(el) * np.cos(az), np.sin(el), np.cos(el) * np.sin(az)],
                           dtype=torch.float32, device="cuda")
    cam_pos = center + dist_factor * extent * cam_dir
    fwd = torch.nn.functional.normalize(center - cam_pos, dim=0)
    up0 = torch.tensor([0.0, -1.0, 0.0], device="cuda")  # COLMAP-ish: y down
    right = torch.nn.functional.normalize(torch.cross(fwd, up0, dim=0), dim=0)
    up = torch.cross(right, fwd, dim=0)

    # face samples (area-uniform barycentric) + normals
    tri = V[F]  # (T,3,3)
    if focus_radius is not None:
        m = (tri.mean(dim=1) - center).norm(dim=-1) < focus_radius
        tri = tri[m]
    n = torch.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0], dim=-1)
    nn = torch.nn.functional.normalize(n, dim=-1)
    cs, nns = [], []
    g = torch.Generator(device="cuda").manual_seed(0)
    for _ in range(n_samples):
        r1 = torch.rand(tri.shape[0], 1, device="cuda", generator=g).sqrt()
        r2 = torch.rand(tri.shape[0], 1, device="cuda", generator=g)
        w = torch.cat([1 - r1, r1 * (1 - r2), r1 * r2], dim=-1)  # (T,3)
        cs.append((tri * w.unsqueeze(-1)).sum(dim=1))
        nns.append(nn)
    c = torch.cat(cs, dim=0)
    nn = torch.cat(nns, dim=0)

    rel = c - cam_pos
    z = rel @ fwd
    x = rel @ right
    y = rel @ up
    valid = z > 1e-3
    f = res * 1.2
    u = (f * x / z + res / 2).long()
    v = (f * -y / z + res / 2).long()
    inside = valid & (u >= 0) & (u < res) & (v >= 0) & (v < res)

    shade = (nn @ torch.nn.functional.normalize(torch.tensor([0.4, 1.0, 0.3], device="cuda"), dim=0)).abs()
    shade = (0.25 + 0.75 * shade).clamp(0, 1)

    # z-buffer via scatter_reduce amin on flattened pixel ids
    pid = (v[inside] * res + u[inside])
    zbuf = torch.full((res * res,), float("inf"), device="cuda")
    zbuf.scatter_reduce_(0, pid, z[inside], reduce="amin")
    keep = z[inside] <= zbuf[pid] * 1.001
    img = torch.zeros(res * res, device="cuda")
    img.scatter_reduce_(0, pid[keep], shade[inside][keep], reduce="amax")
    return img.view(res, res).cpu().numpy()


focus = 0.28 * extent  # ~truck-scale central region
views = [(-30, 30), (60, 30), (150, 30), (240, 30)]
fig, axes = plt.subplots(2, 2, figsize=(14, 14))
for ax, (a, e) in zip(axes.ravel(), views):
    ax.imshow(render_view(a, e, dist_factor=0.28, focus_radius=focus),
              cmap="gray", vmin=0, vmax=1)
    ax.set_title(f"azim={a} elev={e} (central {focus:.1f}u)")
    ax.axis("off")
plt.tight_layout()
plt.savefig(out_png, dpi=110)
print("wrote", out_png)
