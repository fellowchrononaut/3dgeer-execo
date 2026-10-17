#!/bin/bash
# Build Tetra-NeRF's CGAL Delaunay extension (python package `tetranerf`), the
# backend behind `--delaunay cgal` in gaussian_wrapping/extract_mesh.py and
# pam_extraction.py (gaussian_wrapping/delaunay_backend.py). GaussianWrapping
# uses the same extension (its scene/mesh.py `method == "tetranerf"`).
#
#   bash tools/build_cgal_triangulation.sh [BUILD_DIR] [--pip]
#
# BUILD_DIR defaults to <repo>/build/tetra_triangulation (git-ignored, excluded
# by hpc/sync.sh: the .so is architecture-specific). --pip additionally
# installs it editable (local docker); without it, put BUILD_DIR on
# PYTHONPATH (hpc/env.sh does, for MODUS, where site-packages is read-only).
#
# Reproducible on any machine with the training image (torch, nvcc, CMake,
# Boost headers): no system CGAL needed.
# - CGAL 5.6.1 (header-only release) and pybind11 v2.13.6 are pinned by
#   version + SHA-256 and unpacked into <repo>/build/_deps. Put the two archives
#   there by hand on a machine without network access.
# - CGAL runs without GMP/MPFR (CGAL_DISABLE_GMP: exact predicates through
#   Boost.Multiprecision), so the same build works in the MODUS SIF, which has
#   Boost but no GMP headers.
# - Licences: CGAL's 3D triangulation is GPL-3 (or commercial from
#   GeometryFactory); the Tetra-NeRF wrapper is MIT; pybind11 is BSD.
#
# Two fixes against the submodule as shipped, applied to the copy only:
# - its cmake/FindTorch.cmake forces -D_GLIBCXX_USE_CXX11_ABI=0, which leaves
#   torch symbols unresolved with torch >= 2.x wheels (ABI 1); use torch's own;
# - CMake >= 4 rejects the pybind11 v2.9.2 its CMakeLists fetches; use v2.13.6.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
SRC="${REPO}/submodules/GaussianWrapping/submodules/tetra_triangulation"
BUILD="${REPO}/build/tetra_triangulation"; PIP=0
for a in "$@"; do
    if [ "${a}" = "--pip" ]; then PIP=1; else BUILD="${a}"; fi
done
DEPS="${REPO}/build/_deps"

CGAL_VER=5.6.1
CGAL_URL="https://github.com/CGAL/cgal/releases/download/v${CGAL_VER}/CGAL-${CGAL_VER}-library.tar.xz"
CGAL_SHA=6640f91dad9956764db27440932ac3a2f6269bf8066690f639dd7e995e3e1925
PYBIND_VER=2.13.6
PYBIND_URL="https://github.com/pybind/pybind11/archive/refs/tags/v${PYBIND_VER}.tar.gz"
PYBIND_SHA=e08cb87f4773da97fa7b5f035de8763abc656d87d5773e62f6da0587d1f0ec20

fetch() {   # url sha256 archive-name
    local f="${DEPS}/$3"
    [ -f "${f}" ] || python -c 'import sys, urllib.request; urllib.request.urlretrieve(sys.argv[1], sys.argv[2])' "$1" "${f}"
    echo "$2  ${f}" | sha256sum -c --quiet - || { echo "checksum mismatch: ${f}" >&2; exit 1; }
}
mkdir -p "${DEPS}"
fetch "${CGAL_URL}" "${CGAL_SHA}" "CGAL-${CGAL_VER}-library.tar.xz"
fetch "${PYBIND_URL}" "${PYBIND_SHA}" "pybind11-${PYBIND_VER}.tar.gz"
[ -d "${DEPS}/CGAL-${CGAL_VER}" ] || tar -xf "${DEPS}/CGAL-${CGAL_VER}-library.tar.xz" -C "${DEPS}"
[ -d "${DEPS}/pybind11-${PYBIND_VER}" ] || tar -xzf "${DEPS}/pybind11-${PYBIND_VER}.tar.gz" -C "${DEPS}"

rm -rf "${BUILD}"; mkdir -p "$(dirname "${BUILD}")"
cp -r "${SRC}" "${BUILD}"
rm -rf "${BUILD}/build" "${BUILD}"/*.egg-info "${BUILD}/CMakeCache.txt" "${BUILD}/CMakeFiles"

ABI="$(python -c 'import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))')"
sed -i "s/-D_GLIBCXX_USE_CXX11_ABI=0/-D_GLIBCXX_USE_CXX11_ABI=${ABI}/" "${BUILD}/cmake/FindTorch.cmake"

if command -v nvidia-smi >/dev/null && nvidia-smi >/dev/null 2>&1; then
    ARCH="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d .)"
else
    ARCH="${CUDA_ARCH:-120}"
fi
CUDA_INC="$(dirname "$(dirname "$(command -v nvcc)")")/include"
TORCH_CMAKE="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')"
echo "==> tetra_triangulation: CGAL ${CGAL_VER} (no GMP), pybind11 ${PYBIND_VER}, sm_${ARCH}, C++11 ABI ${ABI}"

cd "${BUILD}"
cmake . -DCMAKE_BUILD_TYPE=Release -DPYTHON_EXECUTABLE="$(command -v python)" \
    -DCMAKE_CUDA_ARCHITECTURES="${ARCH}" -DCMAKE_PREFIX_PATH="${TORCH_CMAKE}" \
    -DCGAL_DIR="${DEPS}/CGAL-${CGAL_VER}" -DCGAL_DISABLE_GMP=ON \
    -DFETCHCONTENT_SOURCE_DIR_PYBIND11="${DEPS}/pybind11-${PYBIND_VER}" -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
    -DCMAKE_CXX_FLAGS="-I${CUDA_INC}" > cmake_configure.log
cmake --build . -j "${MAX_JOBS:-$(nproc)}" > cmake_build.log

if [ "${PIP}" = 1 ]; then
    pip install --no-build-isolation --no-deps -e "${BUILD}" > pip_install.log
fi

cd /
PYTHONPATH="${BUILD}:${PYTHONPATH:-}" python - <<'PY'
import numpy as np, torch
import tetranerf
from tetranerf.utils.extension import cpp
p = np.random.default_rng(0).random((20000, 3)).astype(np.float32)
a = {tuple(sorted(t)) for t in cpp.triangulate(torch.from_numpy(p)).tolist()}
try:
    from scipy.spatial import Delaunay
    b = {tuple(sorted(t)) for t in Delaunay(p.astype(np.float64)).simplices.tolist()}
    check = "identical" if a == b else f"DIFFERENT ({len(a & b)}/{len(b)})"
except ImportError:   # e.g. the MODUS image has no scipy
    check = f"{len(a)} tets (scipy not installed, no reference check)"
print("==> tetranerf from", tetranerf.__file__, "| CGAL on 20k points:", check)
assert len(a) > 100000
PY
