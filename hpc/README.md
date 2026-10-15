# Running 3DGEER on the MODUS HPC

Everything here is additive: no existing script or Dockerfile was modified.
The container image is built **on MODUS** from `hpc/geer.def`, a step-for-step
translation of `docker/Dockerfile_5090` (which stays the dev PC's recipe). The
header of `geer.def` lists every difference. Keep the two recipes in step.

## The cluster, as verified 2026-09-04

| | |
|---|---|
| Login node | `modus02`, `$HOME=/home/isae/s.jois` |
| Scratch | `/scratch/isae/deos/s.jois` -> `/gpfs/scratch/isae/deos/s.jois` |
| Quota | **scratch 445 GB soft / 450 GB hard** (raised from 95/100, 2026-10), home 45/50 GB. Inode count unlimited |
| Training GPU | `gpu01`: 2x H200 NVL, **sm_90**, 141 GB, driver 580.126.20, + 14 MIG 1g slices |
| Viewing GPU | `modus-visu[01-03]`: RTX 6000 Ada, **sm_89**, 48 GB, DCV desktop, no Slurm GRES |
| Wall time | gpu partition: 6 h default, **24 h max**. Over-requests are rejected, not clamped |
| Contention | gpu01 idle; one other GPU user in 14 days; no QOS caps |
| Network | GitHub direct. Docker Hub, gitlab.inria.fr, Ubuntu apt, PyPI via `module load proxy` -- from login **and** CPU nodes |
| Apptainer | `module load apptainer/1.4.5-3`. **`--fakeroot` builds from a `.def` work without root**, incl. with `APPTAINER_TMPDIR` on GPFS |
| CPU nodes | diskless: `/tmp` is a RAM-backed overlay, so big temp files go on GPFS |

## The core idea

The container holds **only the environment**. Your code lives on GPFS and is
bind-mounted in. So a repo change usually never touches the image:

| You changed | Needed on MODUS | Cost |
|---|---|---|
| Python (`train.py`, `utils/`, `scene/`) | `sync.sh code`, run | seconds |
| `.cu` / `.h` / `.cpp` | `sync.sh code`, then `build_extensions.sh` | 2-5 min |
| pip dep, CUDA or PyTorch version | edit `geer.def` (and `Dockerfile_5090`), `sbatch hpc/build_sif.slurm` | ~1 h, rare |

Only the third case rebuilds the image. That is the whole reason to bind-mount
rather than bake the code in.

Because the SIF's `site-packages` is read-only, extensions are built
`--inplace` into the bind-mounted repo and picked up via `PYTHONPATH`, which
precedes `site-packages` in `sys.path`. No reinstall after a CUDA edit.

## First-time setup

On the dev PC, push code and one dataset:

```bash
bash hpc/sync.sh code
bash hpc/sync.sh data x5_eq_1920
```

On the MODUS login node, build the image as a CPU-node job:

```bash
cd /gpfs/scratch/isae/deos/s.jois/3dgeer-execo
sbatch hpc/build_sif.slurm     # -> ~/containers/geer_h200.sif
squeue -u $USER
tail -f /gpfs/scratch/isae/deos/s.jois/logs/geer_sif_*.out
```

No Docker, registry or token is involved: Apptainer pulls the PyTorch base image
from Docker Hub through the ISAE proxy and builds with `--fakeroot`.

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

`*.so` is in `.gitignore` too, so a `git clone` on MODUS is equally safe.
(`multiview_ncc/_C.cpython-311-x86_64-linux-gnu.so` used to be tracked and
contained only sm_120 code; it was untracked for exactly this reason.)

**Checkpoint size scales with gaussian count, and bigger scenes are the point.**
A 2.1 GB `.pth` is roughly 3M gaussians. The H200's 141 GB could train 15-20M,
which makes checkpoints 10-14 GB each, so one run on the current four-checkpoint
schedule is ~50-60 GB. With 445 GB of scratch that is seven or so big runs
resident, or ~40 at x5_eq scale (~10 GB each) -- enough that pruning is
housekeeping rather than a constraint. Keep fetching and pruning finished runs
anyway: the purge policy is still unknown, and the dev PC is the archive.

A rolling window of two `.pth` is no longer needed for space. A save-on-SIGUSR1
handler in `train.py` is still worth adding for runs that approach the 24 h wall
clock; `hpc/train.slurm` already requests the signal and has the hook point.

## Viewing

`modus-visu[01-03]` have a 48 GB RTX 6000 Ada and a browser-reachable DCV
desktop, on the same GPFS as gpu01. Checkpoints can be inspected in place
rather than pulled home. The image keeps its SIBR build for this; running
containerized SIBR against DCV is untested and GL-in-containers is fiddly, so
expect some fiddling on the first attempt.

Those nodes are the shared visualization resource and their GPU is not
Slurm-managed. Fine for a look at a checkpoint; not the place for training runs.

## Open with the admin

- ~~Raise the scratch quota~~ -- done, 445 GB soft / 450 GB hard (2026-10).
- **Scratch purge policy** -- still unknown; determines whether datasets can
  live on scratch between sessions or must be re-staged each time.

## Files

| File | Runs on | Purpose |
|---|---|---|
| `geer.def` | built on MODUS | Container recipe, `8.9;9.0+PTX`; mirrors `docker/Dockerfile_5090` |
| `sync.sh` | dev PC | `code` / `data` / `fetch` / `quota` |
| `build_sif.slurm` | modus02 (`sbatch`) | Builds `geer.def` on a CPU node -> `~/containers/geer_h200.sif` |
| `env.sh` | MODUS | Shared paths and the `geer_exec` container wrapper |
| `build_extensions.sh` | GPU node, in container | In-place CUDA extension build |
| `train.slurm` | modus02 | sbatch wrapper: provenance stamp, signal hook |
| `interactive.sh` | modus02 | `salloc` + container shell for debugging |
| `jobs/x5_eq.sh` | GPU node, in container | Example run |
