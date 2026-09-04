# Running 3DGEER on the MODUS HPC

Everything here is additive: no existing script or Dockerfile was modified.
`docker/Dockerfile_h200` sits alongside the 3090/4090/5090 variants and differs
from `Dockerfile_5090` in exactly three lines.

## The cluster, as verified 2026-09-04

| | |
|---|---|
| Login node | `modus02`, `$HOME=/home/isae/s.jois` |
| Scratch | `/scratch/isae/deos/s.jois` -> `/gpfs/scratch/isae/deos/s.jois` |
| Quota | **scratch 95 GB soft / 100 GB hard**, home 45/50 GB. Inode count unlimited |
| Training GPU | `gpu01`: 2x H200 NVL, **sm_90**, 141 GB, driver 580.126.20, + 14 MIG 1g slices |
| Viewing GPU | `modus-visu[01-03]`: RTX 6000 Ada, **sm_89**, 48 GB, DCV desktop, no Slurm GRES |
| Wall time | gpu partition: 6 h default, **24 h max**. Over-requests are rejected, not clamped |
| Contention | gpu01 idle; one other GPU user in 14 days; no QOS caps |
| Network | No proxy needed. GitHub direct. **ghcr.io works, Docker Hub does not** |
| Apptainer | `module load apptainer/1.4.5-3`. Unprivileged `docker://` builds verified |

## The core idea

The container holds **only the environment**. Your code lives on GPFS and is
bind-mounted in. So a repo change usually never touches the image:

| You changed | Needed on MODUS | Cost |
|---|---|---|
| Python (`train.py`, `utils/`, `scene/`) | `sync.sh code`, run | seconds |
| `.cu` / `.h` / `.cpp` | `sync.sh code`, then `build_extensions.sh` | 2-5 min |
| pip dep, CUDA or PyTorch version | rebuild image, push ghcr, re-pull SIF | 30-40 min, rare |

Only the third case rebuilds the image. That is the whole reason to bind-mount
rather than bake the code in.

Because the SIF's `site-packages` is read-only, extensions are built
`--inplace` into the bind-mounted repo and picked up via `PYTHONPATH`, which
precedes `site-packages` in `sys.path`. No reinstall after a CUDA edit.

## First-time setup

On the dev PC:

```bash
# 1. build the three-arch image and push it (needs a GitHub PAT, write:packages)
echo $PAT | docker login ghcr.io -u fellowchrononaut --password-stdin
bash hpc/build_and_push.sh

# 2. push code and one dataset
bash hpc/sync.sh code
bash hpc/sync.sh data x5_eq_1920
```

On the MODUS login node:

```bash
cd /scratch/isae/deos/s.jois/3dgeer-execo
bash hpc/build_sif.sh          # ghcr -> ~/containers/geer_h200.sif
```

On a GPU node, once, and after every CUDA edit:

```bash
bash hpc/interactive.sh 1 mig               # or: full
# inside the container:
bash hpc/build_extensions.sh
```

## Everyday loop

```bash
# dev PC
bash hpc/sync.sh code

# MODUS
cd /gpfs/scratch/isae/deos/s.jois/3dgeer-execo
sbatch --job-name=x5eq_h200 hpc/train.slurm hpc/jobs/x5_eq.sh
squeue -u $USER
tail -f /gpfs/scratch/isae/deos/s.jois/logs/x5eq_h200_*.out

# dev PC, when it finishes
bash hpc/sync.sh fetch x5eq_h200 --prune
```

`bash hpc/sync.sh quota` shows remaining scratch at any time.

## Two things that will bite

**Architecture-specific binaries must not travel.** The dev PC builds sm_120,
gpu01 needs sm_90. An sm_120 `.so` imports cleanly on the H200 and then dies at
kernel launch with *"no kernel image is available for execution on the device"* --
a confusing failure, not an obvious one. `hpc/sync.sh` excludes `*.so` and
`build/` for this reason.

`submodules/multiview_ncc/multiview_ncc/_C.cpython-311-x86_64-linux-gnu.so` is
currently **tracked in git**, so a `git clone` on MODUS reintroduces the trap
even though rsync avoids it. Worth untracking:

```bash
git rm --cached submodules/multiview_ncc/multiview_ncc/_C.cpython-311-x86_64-linux-gnu.so
echo '*.so' >> .gitignore
```

**Checkpoint size scales with gaussian count, and bigger scenes are the point.**
A 2.1 GB `.pth` is roughly 3M gaussians. The H200's 141 GB could train 15-20M,
which makes checkpoints 10-14 GB each -- one run on the current four-checkpoint
schedule would eat half the scratch quota. At x5_eq scale (~10 GB per run,
6-8 runs resident) this is comfortable, so it is not urgent yet. When scene size
grows, `train.py` needs a rolling window of two `.pth` plus a save-on-SIGUSR1
handler; `hpc/train.slurm` already requests the signal and has the hook point.

## Viewing

`modus-visu[01-03]` have a 48 GB RTX 6000 Ada and a browser-reachable DCV
desktop, on the same GPFS as gpu01. Checkpoints can be inspected in place
rather than pulled home. The image keeps its SIBR build for this; running
containerized SIBR against DCV is untested and GL-in-containers is fiddly, so
expect some fiddling on the first attempt.

Those nodes are the shared visualization resource and their GPU is not
Slurm-managed. Fine for a look at a checkpoint; not the place for training runs.

## Open with the admin

- **Raise the scratch quota.** The filesystem is 2% used (8 of 428 TB) and
  95 GB is the binding constraint on scene size. Highest-leverage ask.
- **Scratch purge policy** -- still unknown; determines whether datasets can
  live on scratch between sessions or must be re-staged each time.

## Files

| File | Runs on | Purpose |
|---|---|---|
| `../docker/Dockerfile_h200` | dev PC | Three-arch image, `8.9;9.0;12.0+PTX` |
| `build_and_push.sh` | dev PC | Build and push to ghcr.io |
| `sync.sh` | dev PC | `code` / `data` / `fetch` / `quota` |
| `build_sif.sh` | modus02 | ghcr -> SIF, with the cache kept off the home quota |
| `env.sh` | MODUS | Shared paths and the `geer_exec` container wrapper |
| `build_extensions.sh` | GPU node, in container | In-place CUDA extension build |
| `train.slurm` | modus02 | sbatch wrapper: provenance stamp, signal hook |
| `interactive.sh` | modus02 | `salloc` + container shell for debugging |
| `jobs/x5_eq.sh` | GPU node, in container | Example run |
