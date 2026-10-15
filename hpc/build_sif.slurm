#!/bin/bash
#SBATCH --job-name=geer_sif
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --output=/gpfs/scratch/isae/deos/s.jois/logs/%x_%j.out
#SBATCH --error=/gpfs/scratch/isae/deos/s.jois/logs/%x_%j.err
#
# Build the container image (hpc/geer.def) on MODUS, on a CPU node.
#
#   cd /gpfs/scratch/isae/deos/s.jois/3dgeer-execo
#   sbatch hpc/build_sif.slurm
#
# Only needed when the ENVIRONMENT changes (pip deps, CUDA, PyTorch, the
# vendored geer-rasterizer baseline). Code and CUDA-kernel edits never need it:
# the repo is bind-mounted and extensions are rebuilt in place.
#
# Why a CPU node: compiling the CUDA extensions needs no GPU, and 30+ min of
# heavy compilation does not belong on the shared login node. CPU nodes reach
# Docker Hub, gitlab.inria.fr and PyPI through `module load proxy`.
#
# Why temp dirs on GPFS: the unpacked image is ~33 GB, and CPU nodes are
# diskless -- their /tmp is a RAM-backed overlay. Fakeroot builds with
# APPTAINER_TMPDIR on GPFS were verified to work.
set -euo pipefail

source "${SLURM_SUBMIT_DIR}/hpc/env.sh"
module load "${APPTAINER_MODULE}" proxy
apptainer --version

WORK="${SCRATCH}/.apptainer_build/${SLURM_JOB_ID}"
export APPTAINER_CACHEDIR="${WORK}/cache"
export APPTAINER_TMPDIR="${WORK}/tmp"
mkdir -p "${APPTAINER_CACHEDIR}" "${APPTAINER_TMPDIR}" "$(dirname "${SIF}")"
trap 'rm -rf "${WORK}"' EXIT

# %files paths in geer.def are relative to the build directory.
cd "${REPO}"

# Build next to the live image and swap only on success, so a failed build
# never leaves running jobs without a SIF.
NEW="${SIF}.new"
echo "==> $(date -Is) building ${NEW} with ${SLURM_CPUS_PER_TASK} jobs"
apptainer build --fakeroot --force \
    --build-arg JOBS="${SLURM_CPUS_PER_TASK}" \
    "${NEW}" hpc/geer.def

echo "==> $(date -Is) sanity check (no GPU on this node, so imports only)"
apptainer exec "${NEW}" python -c "
import torch, simple_knn._C, diff_gaussian_rasterization
print('torch', torch.__version__, 'cuda', torch.version.cuda)
print('extensions import OK')
"
apptainer exec "${NEW}" ls /workspace/3dgs_dependency/sibr_core/install/bin/ | grep -i gaussian

# If the check above fails, set -e stops here: the live SIF is untouched and
# ${NEW} is left in place for inspection.
mv -f "${NEW}" "${SIF}"
ls -lh "${SIF}"
echo "==> $(date -Is) done. Next, on a GPU node: bash hpc/interactive.sh 1, then bash hpc/build_extensions.sh"
