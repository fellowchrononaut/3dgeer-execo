#!/bin/bash
# Rebuild the CUDA extensions in place, inside the container, on a GPU node.
# Needed after any .cu/.cuh/.h/.cpp change, and once after the first code sync.
#
# Run it through the container, from a node with the target GPU:
#   source hpc/env.sh && geer_exec bash hpc/build_extensions.sh
#
# Extensions are built --inplace into the bind-mounted repo rather than pip
# installed, because a SIF's site-packages is read-only. hpc/env.sh puts these
# directories on PYTHONPATH so the fresh sm_90 build shadows the sm_89/sm_90/
# sm_120 fat binary baked into the image.
#
# The compiled .so is architecture-specific and MUST NOT travel between
# machines. hpc/sync.sh excludes '*.so' and 'build/' for exactly this reason:
# an sm_120 binary from the dev PC imports cleanly on the H200 and then dies at
# kernel launch with "no kernel image is available for execution on the device".
set -euo pipefail

REPO_DIR="${REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "${REPO_DIR}"

# Match the GPU we are actually on, rather than building all three targets.
CC="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)"
export TORCH_CUDA_ARCH_LIST="${CC}"
echo "==> building for compute capability ${CC}"

NPROC="${SLURM_CPUS_PER_TASK:-8}"
export MAX_JOBS="${NPROC}"

for ext in submodules/geer-rasterizer submodules/multiview_ncc; do
    echo "==> ${ext}"
    ( cd "${ext}" && python setup.py build_ext --inplace )
done

echo "==> verifying imports resolve to the in-place builds"
python - <<'PY'
import diff_gaussian_rasterization as d, multiview_ncc as m, torch
print("torch                    ", torch.__version__)
print("diff_gaussian_rasterization", d.__file__)
print("multiview_ncc            ", m.__file__)
if torch.cuda.is_available():
    from simple_knn._C import distCUDA2
    x = distCUDA2(torch.randn(1000, 3, device="cuda").contiguous())
    print("kernel launch OK         ", x.shape, torch.cuda.get_device_name(0))
PY
