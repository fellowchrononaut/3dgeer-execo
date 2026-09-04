#!/bin/bash
# Build the Apptainer image on the MODUS login node from the ghcr.io image.
# Run on modus02, not on a compute node.
#
#   bash hpc/build_sif.sh
#
# Docker Hub is unreachable from MODUS (i/o timeout to index.docker.io);
# ghcr.io works. Push from the dev PC first: bash hpc/build_and_push.sh
set -euo pipefail
source "$(dirname "$0")/env.sh"

module load "${APPTAINER_MODULE}"
apptainer --version

mkdir -p "$(dirname "${SIF}")"

# The OCI layer cache is ~11 GB for this image. Its default home
# (~/.apptainer/cache) would take a quarter of the 45 GB home quota, so keep it
# on scratch and remove it once the SIF exists.
CACHE="${SCRATCH}/.apptainer_cache"
export APPTAINER_CACHEDIR="${CACHE}"
mkdir -p "${CACHE}"
trap 'rm -rf "${CACHE}"' EXIT

echo "==> building ${SIF} from docker://${GHCR_IMAGE}"
apptainer build --force "${SIF}" "docker://${GHCR_IMAGE}"

ls -lh "${SIF}"
echo "==> sanity check (no GPU needed on the login node)"
apptainer exec "${SIF}" python -c "import torch; print('torch', torch.__version__)"
echo "==> done. GPU check happens in the first job; see hpc/train.slurm"
