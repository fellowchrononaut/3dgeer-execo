#!/bin/bash
# Delete all but the newest <keep_n> chkpnt<iter>.pth of a model dir, plus any
# listed milestone iterations. Checkpoints hold the Gaussians and both Adam
# moments (~3x the point cloud, 2-3 GB each on x5); resuming needs only the
# newest, and meshes/viewers use point_cloud/iteration_*/ instead.
#
#   tools/prune_checkpoints.sh <model_dir> <keep_n> [milestone_iter ...]
#   tools/prune_checkpoints.sh <model_dir> <keep_n> [milestone_iter ...] --loop <pid>
#       repeat every 60 s while <pid> runs (used by train_autoresume.sh)
#
# The newest file is always kept, so a checkpoint being written is never touched.
set -uo pipefail
MODEL_DIR="${1:?model_dir}"; KEEP_N="${2:?keep_n}"; shift 2
MILESTONES=(); LOOP_PID=""
while [ $# -gt 0 ]; do
    if [ "$1" = "--loop" ]; then LOOP_PID="$2"; shift 2; else MILESTONES+=("$1"); shift; fi
done
[ "${KEEP_N}" -ge 1 ] || { echo "keep_n must be >= 1" >&2; exit 1; }

prune() {
    local files n i f it keep m
    mapfile -t files < <(ls "${MODEL_DIR}"/chkpnt*.pth 2>/dev/null | sort -V)
    n=${#files[@]}
    for ((i = 0; i < n - KEEP_N; i++)); do
        f="${files[$i]}"; it="$(basename "$f" .pth)"; it="${it#chkpnt}"
        keep=0
        for m in "${MILESTONES[@]}"; do [ "$it" = "$m" ] && keep=1; done
        if [ "$keep" = 0 ]; then
            rm -f "$f" && echo "$(date '+%F %T') pruned $(basename "$f")" >> "${MODEL_DIR}/prune.log"
        fi
    done
}

if [ -n "${LOOP_PID}" ]; then
    while kill -0 "${LOOP_PID}" 2>/dev/null; do prune; sleep 60; done
fi
prune
