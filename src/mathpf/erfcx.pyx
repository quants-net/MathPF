# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
# distutils: language = c++
"""Scaled complementary error function erfcx(z) = exp(z^2) * erfc(z) and its
1st and 3rd derivatives w.r.t. z.

This module is a thin numpy-vectorisation layer over the C++ kernels in
src/mathpf/_kernels/erfcx.{h,cpp}, which express the surface via mathpf's
Mills-ratio primitives (R, R_1, R_3) -- cancellation-free for both small and
large |z|.

Equivalent to scipy.special.erfcx (interface-wise) but routed through
mathpf's tiered-CF + Chebyshev Mills implementation, which is the same
machinery option-pricing callers (lbp::Bachelier, lbp::Black) consume.

The scalar entry points (_erfcx, _erfcx_d1, _erfcx_d3) are cimport-able from
other Cython modules via the cdef extern declarations in erfcx.pxd.
"""
import numpy as np
cimport numpy as np
np.import_array()


# -- Python-callable numpy-vectorized wrappers --
def erfcx(z):
    """Scaled complementary error function erfcx(z) = exp(z^2) * erfc(z);
    scalar or ndarray, any z."""
    cdef double[::1] f, o
    cdef Py_ssize_t i
    z_arr = np.asarray(z, dtype=np.float64)
    scalar = z_arr.ndim == 0
    flat = np.ascontiguousarray(z_arr.ravel())
    out = np.empty_like(flat)
    f = flat
    o = out
    for i in range(f.shape[0]):
        o[i] = _erfcx(f[i])
    return float(out[0]) if scalar else out.reshape(z_arr.shape)


def erfcx_d1(z):
    """First derivative d/dz [erfcx(z)] = 2 z erfcx(z) - 2/sqrt(pi); scalar or
    ndarray, any z.  Computed cancellation-free via mathpf's R_1 primitive."""
    cdef double[::1] f, o
    cdef Py_ssize_t i
    z_arr = np.asarray(z, dtype=np.float64)
    scalar = z_arr.ndim == 0
    flat = np.ascontiguousarray(z_arr.ravel())
    out = np.empty_like(flat)
    f = flat
    o = out
    for i in range(f.shape[0]):
        o[i] = _erfcx_d1(f[i])
    return float(out[0]) if scalar else out.reshape(z_arr.shape)


def erfcx_d3(z):
    """Third derivative d^3/dz^3 [erfcx(z)]; scalar or ndarray, any z.
    Computed cancellation-free via mathpf's R_3 primitive."""
    cdef double[::1] f, o
    cdef Py_ssize_t i
    z_arr = np.asarray(z, dtype=np.float64)
    scalar = z_arr.ndim == 0
    flat = np.ascontiguousarray(z_arr.ravel())
    out = np.empty_like(flat)
    f = flat
    o = out
    for i in range(f.shape[0]):
        o[i] = _erfcx_d3(f[i])
    return float(out[0]) if scalar else out.reshape(z_arr.shape)
