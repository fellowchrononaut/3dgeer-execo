# ported/adapted (structural template only, internal math is new) from
# GaussianWrapping submodules/Geometry-Grounded-Gaussian-Splatting/submodules/
# warp-patch-ncc/setup.py -- see GUTWrap_Discussion/3DGEERGW_EXECUTION.md,
# Session F2, for why this is a NEW standalone extension rather than a reuse
# of warp_patch_ncc (that kernel is pinhole-only via a homography; ours is a
# general ray-based per-offset EQ/PH reprojection kernel).
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os
os.path.dirname(os.path.abspath(__file__))

setup(
    name="multiview_ncc",
    packages=['multiview_ncc'],
    ext_modules=[
        CUDAExtension(
            name="multiview_ncc._C",
            sources=[
                "cuda_multiview_ncc/multiview_ncc_impl.cu",
                "multiview_ncc.cu",
                "ext.cpp"],
            extra_compile_args={"nvcc": [
                "-O3",
                "--use_fast_math"]})
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
