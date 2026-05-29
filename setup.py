from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
import numpy as np

# C++ kernels (templated on T in {float, double}, namespace QuantsNet) live under
# src/mathpf/_kernels/ and are compiled directly into each consuming Python
# extension.  Each .so/.pyd carries its own copy of the extern "C" symbols
# (mathpf_MillsRatio, mathpf_MillsRatioDiff, ...).  No cross-extension dynamic
# linkage required -- each is self-contained.
_KERNEL_INCLUDES = [np.get_include(), "src/mathpf", "src/mathpf/_kernels"]
# mills.cpp: R, R1, R3, Rrel_below1, R013_CF (Mills primitives).
# mills_dd.cpp: R_DD, R_DD_CF (symmetric divided differences; uses mills primitives
# internally via the QuantsNet:: templated names, so mills.cpp is required too).
_MILLS_SRC    = ["src/mathpf/_kernels/mills.cpp"]
_MILLS_DD_SRC = ["src/mathpf/_kernels/mills.cpp", "src/mathpf/_kernels/mills_dd.cpp"]

extensions = [
    Extension(
        name="mathpf.avg_funcs",
        sources=["src/mathpf/avg_funcs.pyx"],
        include_dirs=[np.get_include()],
    ),
    Extension(
        name="mathpf.mills",
        sources=["src/mathpf/mills.pyx"] + _MILLS_SRC,
        include_dirs=_KERNEL_INCLUDES,
        language="c++",
    ),
    Extension(
        name="mathpf.mills_dd",
        sources=["src/mathpf/mills_dd.pyx"] + _MILLS_DD_SRC,
        include_dirs=_KERNEL_INCLUDES,
        language="c++",
    ),
]

setup(
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    ext_modules=cythonize(
        extensions,
        language_level=3,
        compiler_directives={
            "boundscheck": False,
            "wraparound": False,
            "cdivision": True,
        },
    )
)
