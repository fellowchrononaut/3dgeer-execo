# Shared configuration for MODUS-side scripts. Source it, don't run it.
#   source hpc/env.sh
#
# Paths use the real GPFS path, not the /scratch symlink: Apptainer bind mounts
# don't carry the host's /scratch -> /gpfs/scratch symlink into the container,
# so using the resolved path keeps paths identical inside and outside.

MODUS_USER="${MODUS_USER:-s.jois}"
SCRATCH="${SCRATCH:-/gpfs/scratch/isae/deos/${MODUS_USER}}"

REPO="${REPO:-${SCRATCH}/3dgeer-execo}"
DATA="${DATA:-${SCRATCH}/data}"
OUTPUT="${OUTPUT:-${SCRATCH}/output}"
LOGS="${LOGS:-${SCRATCH}/logs}"

# The SIF lives in $HOME (45 GB quota, barely used) so it doesn't eat the
# 95 GB scratch quota that datasets and checkpoints have to share.
SIF="${SIF:-${HOME}/containers/geer_h200.sif}"

APPTAINER_MODULE="${APPTAINER_MODULE:-apptainer/1.4.5-3}"
GHCR_IMAGE="${GHCR_IMAGE:-ghcr.io/fellowchrononaut/3dgeer:h200}"

# PYTHONPATH makes the in-place-built extensions in the bind-mounted repo win
# over the copies baked into the read-only SIF. PYTHONPATH precedes
# site-packages in sys.path, so no reinstall is needed after a CUDA edit.
GEER_PYTHONPATH="${REPO}/submodules/geer-rasterizer:${REPO}/submodules/multiview_ncc"

# Run a command inside the container with the GPU and GPFS visible.
#   geer_exec python train.py ...
#
# Deliberately NOT using --cleanenv: it would strip CUDA_VISIBLE_DEVICES, which
# Slurm sets to scope the job to its allocated GPU. PYTHONNOUSERSITE=1 is the
# targeted fix for the one thing --cleanenv would have protected us from
# (~/.local/lib/python3.11 shadowing the container's site-packages via the
# default $HOME bind).
geer_exec() {
    apptainer exec --nv \
        --bind "${SCRATCH}:${SCRATCH}" \
        --env PYTHONNOUSERSITE=1 \
        --env PYTHONPATH="${GEER_PYTHONPATH}" \
        --pwd "${REPO}" \
        "${SIF}" "$@"
}
