from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        name="mathpf.avg_funcs",
        sources=["src/mathpf/avg_funcs.pyx"],
        include_dirs=[np.get_include()],
    ),
    Extension(
        name="mathpf.mills",
        sources=["src/mathpf/mills.pyx"],
        include_dirs=[np.get_include(), "src/mathpf"],  # for _mills_coef.h
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
