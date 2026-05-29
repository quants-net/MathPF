# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
# distutils: language = c++
"""Mills ratio R(x) = N(-x)/n(x), -R'(x), -R'''(x), plus (sqrt(pi/2) - R(x))/x on [0, 1].

This module is a thin numpy-vectorisation layer over the C++ kernels in
src/mathpf/_kernels/mills.{h,cpp}.  The kernels are templated on T in
{float, double}; the Cython binding (this file) and the qna pricer share the
same canonical math source.

The scalar entry points (_R, _R1, _R3, _R013_CF, _Rrel_below1) are cimport-able
from other Cython modules via the cdef extern declarations in mills.pxd.
"""
import numpy as np
cimport numpy as np
np.import_array()


# -- Python-callable numpy-vectorized wrappers --
def millsratio(x):
    """Mills ratio R(x) = N(-x)/n(x); scalar or ndarray, any x."""
    cdef double[::1] f, o
    cdef Py_ssize_t i
    x_arr = np.asarray(x, dtype=np.float64)
    scalar = x_arr.ndim == 0
    flat = np.ascontiguousarray(x_arr.ravel())
    out = np.empty_like(flat)
    f = flat
    o = out
    for i in range(f.shape[0]):
        o[i] = _R(f[i])
    return float(out[0]) if scalar else out.reshape(x_arr.shape)


def millsratio_d1(x):
    """-R'(x) = 1 - x R(x); scalar or ndarray, any x."""
    cdef double[::1] f, o
    cdef Py_ssize_t i
    x_arr = np.asarray(x, dtype=np.float64)
    scalar = x_arr.ndim == 0
    flat = np.ascontiguousarray(x_arr.ravel())
    out = np.empty_like(flat)
    f = flat
    o = out
    for i in range(f.shape[0]):
        o[i] = _R1(f[i])
    return float(out[0]) if scalar else out.reshape(x_arr.shape)


def millsratio_d3(x):
    """-R'''(x); scalar or ndarray, any x."""
    cdef double[::1] f, o
    cdef Py_ssize_t i
    x_arr = np.asarray(x, dtype=np.float64)
    scalar = x_arr.ndim == 0
    flat = np.ascontiguousarray(x_arr.ravel())
    out = np.empty_like(flat)
    f = flat
    o = out
    for i in range(f.shape[0]):
        o[i] = _R3(f[i])
    return float(out[0]) if scalar else out.reshape(x_arr.shape)


def millsratio_rel_below1(x):
    """(sqrt(pi/2) - R(x))/x; scalar or ndarray, 0 <= x <= 1."""
    cdef double[::1] f, o
    cdef Py_ssize_t i
    x_arr = np.asarray(x, dtype=np.float64)
    if np.any((x_arr < 0.0) | (x_arr > 1.0)):
        raise ValueError('millsratio_rel_below1 requires 0 <= x <= 1.')
    scalar = x_arr.ndim == 0
    flat = np.ascontiguousarray(x_arr.ravel())
    out = np.empty_like(flat)
    f = flat
    o = out
    for i in range(f.shape[0]):
        o[i] = _Rrel_below1(f[i])
    return float(out[0]) if scalar else out.reshape(x_arr.shape)
