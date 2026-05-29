# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
# distutils: language = c++
"""Symmetric divided differences of the Mills ratio R(x) = N(-x)/n(x).

For a midpoint x and half-step dx (so the two evaluation points are x - dx and x + dx):

    millsratio_dd(x, dx, theta=+1) = (R(x-dx) - R(x+dx)) / (2 dx)  (difference branch)
    millsratio_dd(x, dx, theta=-1) = (R(dx-x) + R(x+dx)) / (2 dx)  (sum branch; first
                                     argument reflected so both R-evaluations land on
                                     positive arguments)
    millsratio_dd_cf(x, dx, n)     = deep-asymptotic cancellation-free divided difference
                                     via the coupled (V, T, Pa, Pb) recurrence on the
                                     depth-n CF convergent of R; table-free.

Note: theta=-1 is NOT the literal arithmetic divided difference -- the first argument is
reflected (R(dx - x) instead of R(x - dx)) so that both R-evaluations land on positive
arguments, avoiding cancellation when x < 0.

This module is a thin numpy-vectorisation layer over the C++ kernels in
src/mathpf/_kernels/mills_dd.{h,cpp}.  The kernels are templated on T in {float, double}
and share the same canonical math source with the qna pricer.

The scalar entry points (_R_DD, _R_DD_CF) are cimport-able from other Cython modules
via the cdef extern declarations in mills_dd.pxd.
"""
import numpy as np
cimport numpy as np
np.import_array()


# -- Python-callable numpy-vectorized wrappers (broadcasting on (x, dx)) --
def millsratio_dd(x, dx, theta=1):
    """Mills-ratio symmetric divided difference:
        theta = +1: returns (R(x-dx) - R(x+dx)) / (2 dx)  (difference branch)
        theta = -1: returns (R(dx-x) + R(x+dx)) / (2 dx)  (sum branch; first argument
                    reflected so both R-evaluations land on positive arguments)
    Scalars or broadcastable ndarrays for x, dx; theta scalar (+/-1).  For theta=+1 a
    three-regime split (CF asymp / 5-term R'''-seeded Taylor / direct R difference)
    bounds the relative error at the worst-case boundary; theta=-1 is a pure sum of two
    Mills ratios (no cancellation).
    """
    cdef const double[::1] fx, fdx
    cdef double[::1] o
    cdef Py_ssize_t i, n
    cdef int th = <int>theta
    x_arr = np.asarray(x, dtype=np.float64)
    dx_arr = np.asarray(dx, dtype=np.float64)
    bshape = np.broadcast_shapes(x_arr.shape, dx_arr.shape)
    scalar = (len(bshape) == 0)
    flat_x = np.ascontiguousarray(np.broadcast_to(x_arr, bshape).ravel())
    flat_dx = np.ascontiguousarray(np.broadcast_to(dx_arr, bshape).ravel())
    out = np.empty_like(flat_x)
    fx = flat_x
    fdx = flat_dx
    o = out
    n = fx.shape[0]
    for i in range(n):
        o[i] = _R_DD(fx[i], fdx[i], th)
    return float(out[0]) if scalar else out.reshape(bshape)


def millsratio_dd_cf(x, dx, n_terms=4):
    """Deep-asymptotic cancellation-free divided difference via the (V, T, P^a, P^b) CF
    recurrence (table-free, algorithmic loop).  n_terms >= 0 supported.  Scalars or
    broadcastable ndarrays for x, dx; n_terms scalar.
    """
    cdef const double[::1] fx, fdx
    cdef double[::1] o
    cdef Py_ssize_t i, n
    cdef int nt = <int>n_terms
    x_arr = np.asarray(x, dtype=np.float64)
    dx_arr = np.asarray(dx, dtype=np.float64)
    bshape = np.broadcast_shapes(x_arr.shape, dx_arr.shape)
    scalar = (len(bshape) == 0)
    flat_x = np.ascontiguousarray(np.broadcast_to(x_arr, bshape).ravel())
    flat_dx = np.ascontiguousarray(np.broadcast_to(dx_arr, bshape).ravel())
    out = np.empty_like(flat_x)
    fx = flat_x
    fdx = flat_dx
    o = out
    n = fx.shape[0]
    for i in range(n):
        o[i] = _R_DD_CF(fx[i], fdx[i], nt)
    return float(out[0]) if scalar else out.reshape(bshape)
