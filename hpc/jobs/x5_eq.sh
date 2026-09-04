#!/bin/bash
# x5_eq EQ-mode training on the H200. Runs INSIDE the container, launched by
# hpc/train.slurm -- do not run it directly on the login node.
#
#   sbatch --job-name=x5eq_h200 hpc/train.slurm hpc/jobs/x5_eq.sh
#
# Mirrors scripts/train_kb.sh in structure. Flags reconstructed from the
# cfg_args of output/x5_eq_hq and output/x5_eq_mesh:
#   --dataset SCANNETPP --camera_model FISHEYE --resolution 2, source x5_eq_1920
# CHECK the render-model and raymap lines against your last local run before
# trusting the numbers; cfg_args only records ModelParams, not the optimizer or
# pipeline flags, so those two could not be recovered from disk.
set -euo pipefail

RUN_NAME="${RUN_NAME:-x5eq_h200}"
SCRATCH="${SCRATCH:-/gpfs/scratch/isae/deos/s.jois}"
DATASET_DIR="${SCRATCH}/data/x5_eq_1920"
OUTPUT_DIR="${SCRATCH}/output/${RUN_NAME}"

ITERS_NUM=30000

# Deliberately NOT passing --densify_grad_threshold. The PH-truck value of
# 0.002 throttles EQ densification roughly tenfold (105k vs 1.07M gaussians,
# verified by A/B on x5_eq 2026-07-08). The default 0.0002 is correct for EQ.

python train.py \
    -s "${DATASET_DIR}" -m "${OUTPUT_DIR}" \
    --iterations "${ITERS_NUM}" \
    --checkpoint_iterations 3000 7000 15000 30000 \
    --save_iterations 3000 7000 15000 30000 \
    --test_iterations 3000 7000 15000 30000 \
    --resolution 2 --eval \
    --render_model EQ \
    --dataset SCANNETPP \
    --camera_model FISHEYE \
    --raymap_path "${DATASET_DIR}/raymap_fisheye.npy"

# Checkpoints are ~2.1 GB each at ~3M gaussians and scale with gaussian count.
# Four of them plus PLYs is ~10 GB, which fits the 95 GB scratch quota fine at
# this scene size. A markedly larger scene will not: see hpc/README.md.
echo "==> output:"
du -sh "${OUTPUT_DIR}"
