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
#
# KEEP_CKPTS=N keeps only the newest N checkpoints while training runs (see
# tools/prune_checkpoints.sh), plus the iterations in KEEP_CKPT_ITERS
# (e.g. "20000 30000"). Unset = keep every checkpoint (previous behaviour).
#
# Gives up early when two attempts in a row fail without writing a newer
# checkpoint: the crash is then deterministic (e.g. an out-of-memory at a fixed
# iteration) and further retries would only replay it.
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
PRUNE=(bash "$(dirname "$0")/prune_checkpoints.sh" "${MODEL_DIR}" "${KEEP_CKPTS:-}" ${KEEP_CKPT_ITERS:-})
if [ -n "${KEEP_CKPTS:-}" ]; then
    # Background pruner tied to this script's PID; train.py stays in the
    # foreground (Ctrl-C and signals reach it as before). Final pass on exit.
    "${PRUNE[@]}" --loop $$ &
    trap '"${PRUNE[@]}"' EXIT
fi
NO_PROGRESS=0
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
    NEWEST="$(ls "${MODEL_DIR}"/chkpnt*.pth 2>/dev/null | sort -V | tail -n 1)"
    if [ -n "${LATEST}" ] && [ "${NEWEST}" = "${LATEST}" ]; then
        NO_PROGRESS=$((NO_PROGRESS + 1))
        if [ "${NO_PROGRESS}" -ge 2 ]; then
            echo "$(date '+%F %T') GAVE UP: two attempts in a row crashed after $(basename "${LATEST}") without a newer checkpoint (deterministic failure)" | tee -a "${LOG}"
            exit 1
        fi
    else
        NO_PROGRESS=0
    fi
done
echo "$(date '+%F %T') GAVE UP after ${MAX_ATTEMPTS} attempts" | tee -a "${LOG}"
exit 1
