#!/bin/bash
#SBATCH --job-name=geer
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=23:55:00
#SBATCH --signal=B:USR1@600
#SBATCH --output=/gpfs/scratch/isae/deos/s.jois/logs/%x_%j.out
#SBATCH --error=/gpfs/scratch/isae/deos/s.jois/logs/%x_%j.err
#
# Generic wrapper: sets up the container, stamps the run for reproducibility,
# then executes a job script inside it.
#
#   cd /gpfs/scratch/isae/deos/s.jois/3dgeer-execo
#   sbatch hpc/train.slurm hpc/jobs/x5_eq.sh
#   sbatch --job-name=x5eq_st hpc/train.slurm hpc/jobs/x5_eq.sh
#
# Notes:
#   - 23:55:00 rather than 24:00:00 matches local practice. The gpu partition's
#     MaxTime is 1-00:00:00 and over-requests are REJECTED, not clamped.
#   - #SBATCH lines cannot use variables, so the log paths are literal. Change
#     them together with MODUS_USER in hpc/env.sh if the account changes.
#   - The log directory must exist at submit time; hpc/sync.sh code creates it.
set -euo pipefail

JOB_SCRIPT="${1:?usage: sbatch hpc/train.slurm <job-script> [args...]}"
shift || true

# sbatch copies this script into Slurm's spool, so $BASH_SOURCE does not point
# at the repo. SLURM_SUBMIT_DIR is the directory sbatch was run from.
source "${SLURM_SUBMIT_DIR}/hpc/env.sh"

module load "${APPTAINER_MODULE}"
mkdir -p "${OUTPUT}" "${LOGS}"

echo "===== SLURM ====="
echo "job        ${SLURM_JOB_ID}   name ${SLURM_JOB_NAME}"
echo "node       $(hostname)"
echo "cpus       ${SLURM_CPUS_PER_TASK}"
echo "start      $(date -Is)"
echo "sif        ${SIF}"
echo "repo       ${REPO}"

echo "===== GPU ====="
# compute_cap must read 9.0 here. If it does not, the extensions built by
# hpc/build_extensions.sh are for the wrong architecture.
nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv

echo "===== PROVENANCE ====="
GIT_SHA="$(git -C "${REPO}" rev-parse --short HEAD 2>/dev/null || echo unknown)"
GIT_DIRTY="$(git -C "${REPO}" status --porcelain 2>/dev/null | wc -l)"
echo "git        ${GIT_SHA} (${GIT_DIRTY} uncommitted files)"
echo "sif mtime  $(stat -c %y "${SIF}" 2>/dev/null || echo missing)"
echo "job script ${JOB_SCRIPT}"

# SIGUSR1 arrives 600 s before the wall clock ends. train.py does not yet act on
# it -- a save-on-signal handler is the pending rolling-window patch -- so for
# now this surfaces the resume command rather than losing the information.
on_usr1() {
    echo "===== WALL CLOCK WARNING at $(date -Is): ~10 min left ====="
    LAST_CKPT="$(ls -1t "${OUTPUT}"/*/chkpnt*.pth 2>/dev/null | head -1 || true)"
    if [ -n "${LAST_CKPT}" ]; then
        echo "latest checkpoint: ${LAST_CKPT}"
        echo "resume with: --start_checkpoint ${LAST_CKPT}"
    else
        echo "no checkpoint written yet -- this run will be lost on timeout"
    fi
}
trap on_usr1 USR1

echo "===== TRAINING: ${JOB_SCRIPT} ====="
# Backgrounded + wait so the USR1 trap can fire; a foreground command would
# block signal handling until it exited.
geer_exec bash "${JOB_SCRIPT}" "$@" &
TRAIN_PID=$!

set +e
wait "${TRAIN_PID}"
STATUS=$?
# A trapped signal makes wait return 128+signum while the child is still alive;
# in that case keep waiting for the real exit status.
while [ "${STATUS}" -gt 128 ] && kill -0 "${TRAIN_PID}" 2>/dev/null; do
    wait "${TRAIN_PID}"
    STATUS=$?
done
set -e

echo "===== DONE ====="
echo "exit       ${STATUS}"
echo "end        $(date -Is)"
echo "scratch usage:"
mmlsquota --block-size auto gpfs 2>/dev/null | tail -3 || true
du -sh "${OUTPUT}" 2>/dev/null || true
echo
echo "fetch results from the dev PC with:"
echo "  bash hpc/sync.sh fetch <run-name> --prune"
exit "${STATUS}"
