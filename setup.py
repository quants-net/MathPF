from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
import numpy as np

# The Mills-ratio C++ kernel (templated on T in {float, double}, namespace
# QuantsNet, declared in _kernels/mills.h, implemented in _kernels/mills.cpp).
# Compiled into both `mills` and `mills_dd` extensions so each .so/.pyd carries
# its own copy of the extern "C" symbols (mathpf_MillsRatio etc.).  No cross-
# extension dynamic linkage required -- each is self-contained.
_KERNEL_SOURCES = ["src/mathpf/_kernels/mills.cpp"]
_KERNEL_INCLUDES = [np.get_include(), "src/mathpf", "src/mathpf/_kernels"]

extensions = [
    Extension(
        name="mathpf.avg_funcs",
        sources=["src/mathpf/avg_funcs.pyx"],
        include_dirs=[np.get_include()],
    ),
    Extension(
        name="mathpf.mills",
        sources=["src/mathpf/mills.pyx"] + _KERNEL_SOURCES,
        include_dirs=_KERNEL_INCLUDES,
        language="c++",
    ),
    Extension(
        name="mathpf.mills_dd",
        sources=["src/mathpf/mills_dd.pyx"] + _KERNEL_SOURCES,
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
