import os
import sys
from setuptools import setup, find_packages

# Optional compilation of the native Strata-DiT CUDA/C++ extension (csrc/)
# Set OPENFAL_BUILD_CUDA=1 when building on a machine with nvcc / CUDA toolkit.
ext_modules = []
cmdclass = {}

if os.environ.get("OPENFAL_BUILD_CUDA", "0") == "1":
    try:
        from torch.utils.cpp_extension import BuildExtension, CUDAExtension

        ext_modules.append(
            CUDAExtension(
                name="openfal._strata_C",
                sources=[
                    "csrc/strata_ops.cpp",
                    "csrc/strata_dma.cu",
                ],
                extra_compile_args={
                    "cxx": ["-O3", "-std=c++17"],
                    "nvcc": [
                        "-O3",
                        "--use_fast_math",
                        "-std=c++17",
                        "-U__CUDA_NO_HALF_OPERATORS__",
                        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                    ],
                },
            )
        )
        cmdclass["build_ext"] = BuildExtension
    except Exception as exc:
        sys.stderr.write(f"[openfal] Skipping native CUDA extension build: {exc}\n")

setup(
    name="openfal",
    version="0.1.0",
    packages=find_packages(include=["openfal", "openfal.*"]),
    package_data={"openfal": ["ui/*.html"]},
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)
