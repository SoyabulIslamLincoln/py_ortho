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
# GCC can fuse the serial and threaded bundle-adjustment expressions differently.
# Scope this parity fix to _ba: disabling contraction in every Linux kernel also
# changes the 0.8.0 dense-matching arithmetic and can reduce its throughput.
ba_args = ["-ffp-contract=off"] if sys.platform.startswith("linux") else []

extensions = [
    Extension(
        f"orthomosaic.{name}",
        [f"orthomosaic/{name}.pyx"],
        depends=[f"orthomosaic/{header}" for header in {
            "_ba": ["_ba_kern.h"], "_dense": ["_sweep.h"],
        }.get(name, [])],
        include_dirs=[np.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        extra_compile_args=compile_args + (exact_args if name == "_fast" else ba_args if name == "_ba" else []),
    )
    for name in ("_core", "_ba", "_mvs", "_dense", "_fast")
]

setup(ext_modules=cythonize(extensions, compiler_directives={"language_level": "3"}))
