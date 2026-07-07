#!/usr/bin/env bash
set -euo pipefail

# Run inside the geer container from /home:
#   docker exec -w /home geer bash scripts/smoke_multiview_depth_wiring.sh
#
# Expected installs in that container:
#   pip install --no-build-isolation /home/submodules/geer-rasterizer
#   cd /home/submodules/multiview_ncc && pip install --no-build-isolation -e .

python - <<'PY'
from diff_gaussian_rasterization import _C as geer_rasterizer
import multiview_ncc

assert hasattr(geer_rasterizer, "rasterize_gaussians")
assert hasattr(multiview_ncc, "multiview_ncc_forward")
print("OK: geer rasterizer and multiview_ncc extensions import")
PY

SCENE_ID=${SCENE_ID:-truck}
DATA_ROOT=${DATA_ROOT:-data/tt/datasets}
OUTPUT_DIR=${OUTPUT_DIR:-output/sessF2_mv_depth_wiring_smoke}
DATASET_DIR=${DATA_ROOT}/${SCENE_ID}

MULTIVIEW_NCC_BACKWARD=${MULTIVIEW_NCC_BACKWARD:-analytic} \
python train.py \
    -s "${DATASET_DIR}" -m "${OUTPUT_DIR}" \
    --iterations 150 \
    --checkpoint_iterations 150 \
    --save_iterations 150 \
    --test_iterations 150 \
    --resolution 1 --eval \
    --render_model PH \
    --dataset COLMAP \
    --camera_model PINHOLE \
    --densify_grad_threshold 0.002 \
    --normal_from_iter 50 \
    --multiview \
    --multiview_from_iter 60 \
    --disable_viewer
