#!/bin/bash
# Build the three-arch image on the dev PC and push it to ghcr.io.
# MODUS cannot reach Docker Hub, but ghcr.io works, so ghcr is the delivery path.
#
#   bash hpc/build_and_push.sh
#
# One-time setup: a GitHub PAT with write:packages, then
#   echo $PAT | docker login ghcr.io -u fellowchrononaut --password-stdin
#
# Only needed when the ENVIRONMENT changes (pip deps, CUDA, PyTorch). Ordinary
# code and CUDA-kernel edits never require this -- the repo is bind-mounted and
# extensions are rebuilt in place. See hpc/README.md.
set -euo pipefail
cd "$(cd "$(dirname "$0")/.." && pwd)"

GHCR_IMAGE="${GHCR_IMAGE:-ghcr.io/fellowchrononaut/3dgeer:h200}"

echo "==> building (three arches: 8.9 + 9.0 + 12.0+PTX -- expect ~30-40 min)"
bash docker/build.sh h200

echo "==> tagging and pushing ${GHCR_IMAGE}"
docker tag zixunh/3dgeer:latest "${GHCR_IMAGE}"
docker push "${GHCR_IMAGE}"

echo "==> done. On the MODUS login node: bash hpc/build_sif.sh"
