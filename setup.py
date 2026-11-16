import os
import sys
from setuptools import setup, find_packages

# Optional compilation of the native Strata-DiT CUDA/C++ extension (csrc/)
# Set OPENFAL_BUILD_CUDA=1 when building on a machine with nvcc / CUDA toolkit.
ext_modules = []
cmdclass = {}

if os.environ.get("OPENFAL_BUILD_CUDA", "0").strip() == "1":
    try:
        from torch.utils.cpp_extension import BuildExtension, CUDAExtension

        is_windows = sys.platform == "win32"
        nvcc_flags = [
            "-O3",
            "--use_fast_math",
            "-std=c++17",
            "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        ]
        library_dirs = []
        if is_windows:
            # CUDA 12.x does not recognize the MSVC 19.50+ toolchain shipped with
            # Visual Studio 2026 / Build Tools 18 and aborts with C1189. Let nvcc
            # proceed with the newer host compiler instead of failing the build.
            nvcc_flags.append("-allow-unsupported-compiler")

            # Conda-packaged CUDA toolkits keep cudart.lib in Library/lib while
            # PyTorch only probes the standard lib/x64 layout.
            from torch.utils.cpp_extension import CUDA_HOME

            if CUDA_HOME:
                for candidate in (
                    os.path.join(CUDA_HOME, "lib", "x64"),
                    os.path.join(CUDA_HOME, "lib"),
                ):
                    if os.path.isfile(os.path.join(candidate, "cudart.lib")):
                        library_dirs.append(candidate)

        ext_modules.append(
            CUDAExtension(
                name="openfal._strata_C",
                sources=[
                    "csrc/strata_ops.cpp",
                    "csrc/strata_dma.cu",
                ],
                library_dirs=library_dirs,
                extra_compile_args={
                    "cxx": ["/O2", "/std:c++17"] if is_windows else ["-O3", "-std=c++17"],
                    "nvcc": nvcc_flags,
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
