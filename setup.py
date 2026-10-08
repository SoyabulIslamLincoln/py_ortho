import os
import sys

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup

if sys.platform == "win32":
    compile_args = ["/O2"]
else:
    compile_args = ["-O3", "-fno-math-errno"]
    if os.environ.get("ORTHO_NATIVE") == "1":  # opt-in: tune for this CPU only
        compile_args.append("-march=native")

# _fast reproduces NumPy expressions operation by operation (bit-identical results), so the
# compiler must not fuse a multiply and an add into one FMA there (MSVC does not by default)
exact_args = [] if sys.platform == "win32" else ["-ffp-contract=off"]
# GCC (Linux) contracts across statements by default, so on aarch64 the serial and threaded
# bundle-adjustment kernels fuse differently and stop being bit-identical. Clang (macOS) only
# contracts within an expression and x86-64 baseline has no FMA, so this changes Linux arm64 only.
if sys.platform.startswith("linux"):
    compile_args.append("-ffp-contract=off")

extensions = [
    Extension(
        f"orthomosaic.{name}",
        [f"orthomosaic/{name}.pyx"],
        depends=[f"orthomosaic/{header}" for header in {
            "_core": ["_hamming.h"], "_ba": ["_ba_kern.h"], "_dense": ["_sweep.h"],
        }.get(name, [])],
        include_dirs=[np.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        extra_compile_args=compile_args + (exact_args if name == "_fast" else []),
    )
    for name in ("_core", "_ba", "_mvs", "_dense", "_fast")
]

setup(ext_modules=cythonize(extensions, compiler_directives={"language_level": "3"}))
