#!/bin/bash
# Interactive shell inside the container on a GPU node, for debugging.
# Run on the MODUS login node, ideally inside tmux so an ssh drop doesn't
# kill the allocation.
#
#   tmux new -s geer
#   bash hpc/interactive.sh            # full H200, 4 h
#   bash hpc/interactive.sh 2 mig      # MIG slice, 2 h, for compiles and imports
#
# gpu01 is effectively uncontended (one other GPU user over the last 14 days,
# no QOS caps), so the full H200 is usually the right default even for debugging.
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/env.sh"

HOURS="${1:-4}"
KIND="${2:-full}"

case "${KIND}" in
    mig)  GRES="gpu:h200_nvl_1g:1"; CPUS=8  ;;
    full) GRES="gpu:h200:1";        CPUS=16 ;;
    *)    echo "usage: interactive.sh [hours] [full|mig]"; exit 1 ;;
esac

echo "==> salloc ${GRES} for ${HOURS}h"
salloc -N 1 -p gpu --gres="${GRES}" --cpus-per-task="${CPUS}" \
    --time="${HOURS}:00:00" -- \
    bash -c "
        module load ${APPTAINER_MODULE}
        source '$(cd "$(dirname "$0")" && pwd)/env.sh'
        nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv
        echo '==> entering container at ${REPO}'
        geer_exec bash
    "
