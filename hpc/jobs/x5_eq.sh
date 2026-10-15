#!/bin/bash
# x5_eq EQ-mode training on the H200. Runs INSIDE the container, launched by
# hpc/train.slurm -- do not run it directly on the login node.
#
#   sbatch --job-name=x5eq_h200 hpc/train.slurm hpc/jobs/x5_eq.sh
#   RUN_NAME=smoke ITERS=1000 ... hpc/jobs/x5_eq.sh     # short smoke run
#   DATASET=x5_kb_1920 RENDER_MODEL=KB ... hpc/jobs/x5_eq.sh
#     # same images/poses with the fitted lens distortion (data/x5_kb_1920/README.txt)
#   START_CKPT=<path/chkpntN.pth> EXTRA_ARGS="--depth_normal_grad st" ...
#     # continue another run's checkpoint into this RUN_NAME, with extra train.py flags
#
# Runs through tools/train_autoresume.sh: a crash (the stochastic rasterizer
# illegal-memory-access) resumes from the newest checkpoint in OUTPUT_DIR.
#
# Mirrors scripts/train_kb.sh in structure. Flags, and where each came from:
#   --dataset SCANNETPP --camera_model FISHEYE --resolution 2, source
#     x5_eq_1920: cfg_args of output/x5_eq_hq and output/x5_eq_mesh
#   --mask_path mask_180.png: printed in the logs of output/x5_eq_ctrl_vanilla
#     and output/x5_eq_r1_st_30k
#   --render_model EQ: forces distortion_scaling=0 (scene/cameras.py); the
#     x5 OPENCV_FISHEYE coefficients are 0 anyway
#   --raymap_path: required for EQ. The raymap is 1920x1920 and sets the render
#     size, so do NOT lower --resolution to save memory without checking that
#     the GT images still match it.
set -euo pipefail

RUN_NAME="${RUN_NAME:-x5eq_h200}"
SCRATCH="${SCRATCH:-/gpfs/scratch/isae/deos/s.jois}"
DATASET_DIR="${SCRATCH}/data/${DATASET:-x5_eq_1920}"
# EQ forces the distortion coefficients to 0; KB uses k1..k4 from
# colmap/cameras_fish.txt and needs a raymap generated with the same model.
RENDER_MODEL="${RENDER_MODEL:-EQ}"
OUTPUT_DIR="${SCRATCH}/output/${RUN_NAME}"

ITERS_NUM="${ITERS:-30000}"
# Full runs keep the usual schedule; shorter runs save/test once, at the end.
if [ "${ITERS_NUM}" = 30000 ]; then DEFAULT_SCHEDULE="3000 7000 15000 30000"; else DEFAULT_SCHEDULE="${ITERS_NUM}"; fi
SCHEDULE="${SCHEDULE:-${DEFAULT_SCHEDULE}}"   # override, e.g. SCHEDULE="3000 5000 7000"

# Deliberately NOT passing --densify_grad_threshold. The PH-truck value of
# 0.002 throttles EQ densification roughly tenfold (105k vs 1.07M gaussians,
# verified by A/B on x5_eq 2026-07-08). The default 0.0002 is correct for EQ.

bash tools/train_autoresume.sh "${OUTPUT_DIR}" "${MAX_ATTEMPTS:-6}" ${START_CKPT:+--initial "${START_CKPT}"} -- \
    -s "${DATASET_DIR}" \
    --iterations "${ITERS_NUM}" \
    --checkpoint_iterations ${SCHEDULE} \
    --save_iterations ${SCHEDULE} \
    --test_iterations ${SCHEDULE} \
    --resolution 2 --eval \
    --render_model "${RENDER_MODEL}" \
    --dataset SCANNETPP \
    --camera_model FISHEYE \
    --raymap_path "${DATASET_DIR}/raymap_fisheye.npy" \
    --mask_path "${DATASET_DIR}/mask_180.png" \
    --ip 127.0.0.1 --port "${VIEWER_PORT:-6009}" \
    ${EXTRA_ARGS:-}

# Checkpoints are ~2.1 GB each at ~3M gaussians and scale with gaussian count.
# Four of them plus PLYs is ~10 GB per run at this scene size; a 15-20M gaussian
# scene is ~50-60 GB. Both fit the 445 GB scratch quota; see hpc/README.md.
echo "==> output:"
du -sh "${OUTPUT_DIR}"
