# throwaway: inspect + replay the captured failing NCC batch
import torch, sys
c = torch.load("/home/output/ncc_failing_batch.pt", weights_only=False)
d, n, uvs = c["depths"], c["normals"], c["uvs"]
print("P =", d.shape[0])
print("depths: finite", torch.isfinite(d).all().item(), "min/max", d.min().item(), d.max().item())
print("normals: finite", torch.isfinite(n).all().item(), "norm min/max",
      n.norm(dim=-1).min().item(), n.norm(dim=-1).max().item())
print("uvs: dtype", uvs.dtype, "x range", uvs[:,0].min().item(), uvs[:,0].max().item(),
      "y range", uvs[:,1].min().item(), uvs[:,1].max().item())
print("ray_dirs:", tuple(c["ray_dirs_r"].shape), "image_r:", tuple(c["image_r"].shape),
      "image_n:", tuple(c["image_n"].shape))
print("R finite:", torch.isfinite(c["R"]).all().item(), "T finite:", torch.isfinite(c["T"]).all().item())
print("intrinsics:", c["fx_n"], c["fy_n"], c["cx_n"], c["cy_n"], "radius", c["patch_radius"],
      "model", c["render_model_n"])
if len(sys.argv) > 1 and sys.argv[1] == "replay":
    import multiview_ncc as ext
    args = [c["depths"].cuda(), c["normals"].cuda(), c["uvs"].cuda().int(),
            c["ray_dirs_r"].cuda(), c["R"].cuda(), c["T"].cuda(),
            c["image_r"].cuda(), c["image_n"].cuda(),
            c["render_model_n"], c["fx_n"], c["fy_n"], c["cx_n"], c["cy_n"],
            c["patch_radius"], True]
    ncc, valid = ext.multiview_ncc_forward(*args)
    torch.cuda.synchronize()
    print("REPLAY OK:", ncc.shape, "valid frac", valid.float().mean().item())
