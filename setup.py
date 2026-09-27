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

extensions = [
    Extension(
        "orthomosaic._core",
        ["orthomosaic/_core.pyx"],
        include_dirs=[np.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        extra_compile_args=compile_args,
    )
]

setup(ext_modules=cythonize(extensions, compiler_directives={"language_level": "3"}))
