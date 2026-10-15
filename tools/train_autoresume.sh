#!/bin/bash
# Run train.py until it reaches its final iteration, resuming from the newest
# checkpoint after a crash (e.g. the stochastic illegal-memory-access in the
# geer-rasterizer, see GUTWrap memory "EQ rasterizer heisenbug").
#
#   tools/train_autoresume.sh <model_dir> <max_attempts> [--initial <ckpt>] -- <train.py args>
#
# <train.py args> must NOT include -m/--model_path or --start_checkpoint; this
# script sets both. --initial seeds attempt 1 from a checkpoint of another run
# (e.g. continue a 7k run to 30k into a new model dir). Pass --iterations and
# --checkpoint_iterations as usual; resume granularity = checkpoint spacing.
# Success = train.py exits 0 and point_cloud/iteration_<--iterations> exists.
set -uo pipefail

# The normal phase (from normal_from_iter) allocates differently sized buffers
# than the image-only phase; with the default caching allocator that fragmented
# a 32 GB 5090 into OOM at ~20.2k on x5_kb (15.6 GiB allocated, 13.1 GiB reserved
# but unusable). Expandable segments avoid that; results are unaffected.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

MODEL_DIR="${1:?model_dir}"; MAX_ATTEMPTS="${2:?max_attempts}"; shift 2
INITIAL=""
if [ "${1:-}" = "--initial" ]; then INITIAL="$2"; shift 2; fi
[ "${1:-}" = "--" ] && shift

TARGET=""
prev=""
for a in "$@"; do [ "$prev" = "--iterations" ] && TARGET="$a"; prev="$a"; done
TARGET="${TARGET:-30000}"

mkdir -p "${MODEL_DIR}"
LOG="${MODEL_DIR}/autoresume.log"
for attempt in $(seq 1 "${MAX_ATTEMPTS}"); do
    LATEST="$(ls "${MODEL_DIR}"/chkpnt*.pth 2>/dev/null | sort -V | tail -n 1)"
    START="${LATEST:-${INITIAL}}"
    echo "$(date '+%F %T') === attempt ${attempt}, resume=${START:-scratch} ===" | tee -a "${LOG}"
    python train.py -m "${MODEL_DIR}" ${START:+--start_checkpoint "${START}"} "$@"
    rc=$?
    echo "$(date '+%F %T') attempt ${attempt} exit=${rc}" | tee -a "${LOG}"
    if [ "${rc}" -eq 0 ] && [ -d "${MODEL_DIR}/point_cloud/iteration_${TARGET}" ]; then
        echo "$(date '+%F %T') SUCCESS reached ${TARGET} after ${attempt} attempt(s)" | tee -a "${LOG}"
        exit 0
    fi
done
echo "$(date '+%F %T') GAVE UP after ${MAX_ATTEMPTS} attempts" | tee -a "${LOG}"
exit 1
