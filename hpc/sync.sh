#!/bin/bash
# Move code, data and results between the dev PC and MODUS.
# Run on the dev PC (deos-simrobot-l), not on MODUS.
#
#   bash hpc/sync.sh code                 push the working tree
#   bash hpc/sync.sh data x5_eq_1920      push one dataset
#   bash hpc/sync.sh fetch sessH_x5eq     pull one run's output back
#   bash hpc/sync.sh fetch sessH_x5eq --prune   ... and free the scratch copy
#
# rsync rather than git for the fast loop: no PAT needed on MODUS, no commit
# required, and it mirrors exactly what is on disk right now. Use git for
# milestones you want a SHA on -- hpc/train.slurm records the SHA either way.
set -euo pipefail

MODUS_HOST="${MODUS_HOST:-modus}"
MODUS_USER="${MODUS_USER:-s.jois}"
REMOTE="${MODUS_USER}@${MODUS_HOST}"
RSCRATCH="/gpfs/scratch/isae/deos/${MODUS_USER}"

LOCAL_REPO="$(cd "$(dirname "$0")/.." && pwd)"

# '*.so' and 'build/' are excluded on purpose: compiled extensions are
# architecture-specific. The dev PC builds sm_120, gpu01 needs sm_90, and an
# sm_120 .so imports fine on the H200 before failing at kernel launch. Note
# that submodules/multiview_ncc/multiview_ncc/_C*.so is currently TRACKED in
# git, so a git clone on MODUS reintroduces this trap; see hpc/README.md.
EXCLUDES=(
    --exclude='data/'      --exclude='output*/'   --exclude='.git/'
    --exclude='build/'     --exclude='*.so'       --exclude='*.egg-info/'
    --exclude='__pycache__/' --exclude='*.pyc'    --exclude='tmp/'
    --exclude='.vscode/'   --exclude='*.pth'      --exclude='normal_dumps/'
)

case "${1:-}" in
  code)
    echo "==> ${LOCAL_REPO} -> ${REMOTE}:${RSCRATCH}/3dgeer-execo"
    ssh "${REMOTE}" "mkdir -p ${RSCRATCH}/3dgeer-execo ${RSCRATCH}/logs ${RSCRATCH}/output ${RSCRATCH}/data"
    rsync -avh --delete "${EXCLUDES[@]}" \
        "${LOCAL_REPO}/" "${REMOTE}:${RSCRATCH}/3dgeer-execo/"
    ;;
  data)
    DS="${2:?usage: sync.sh data <dataset-dir-name>}"
    echo "==> data/${DS} -> ${REMOTE}:${RSCRATCH}/data/${DS}"
    du -sh "${LOCAL_REPO}/data/${DS}"
    rsync -avh --progress "${LOCAL_REPO}/data/${DS}/" "${REMOTE}:${RSCRATCH}/data/${DS}/"
    ;;
  fetch)
    RUN="${2:?usage: sync.sh fetch <run-name> [--prune]}"
    DEST="${LOCAL_REPO}/output/${RUN}_modus"
    echo "==> ${REMOTE}:${RSCRATCH}/output/${RUN} -> ${DEST}"
    mkdir -p "${DEST}"
    rsync -avh --progress "${REMOTE}:${RSCRATCH}/output/${RUN}/" "${DEST}/"
    if [ "${3:-}" = "--prune" ]; then
        echo "==> freeing scratch: removing remote ${RUN}"
        ssh "${REMOTE}" "rm -rf ${RSCRATCH}/output/${RUN}"
    fi
    echo "==> remaining scratch usage:"
    ssh "${REMOTE}" "mmlsquota --block-size auto gpfs 2>/dev/null | tail -3"
    ;;
  quota)
    ssh "${REMOTE}" "mmlsquota --block-size auto gpfs; du -sh ${RSCRATCH}/* 2>/dev/null | sort -rh"
    ;;
  *)
    sed -n '2,12p' "$0"; exit 1
    ;;
esac
