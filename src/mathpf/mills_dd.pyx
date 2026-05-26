# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""Symmetric divided differences of the Mills ratio R(x) = N(-x)/n(x), in the form used
by the Black-Scholes price-to-vega ratio.

For a midpoint x and half-step dx (so the two evaluation points are x - dx and x + dx;
in BS notation x = m0 = logk/sigma, dx = sigma/2, x - dx = -d1, x + dx = -d2):

    millsratio_dd(x, dx, theta=+1) = (R(x-dx) - R(x+dx)) / (2 dx)    = Cv/sigma  (call branch)
    millsratio_dd(x, dx, theta=-1) = (R(dx-x) + R(x+dx)) / (2 dx)    = (1-Cv)/sigma (above branch)
    millsratio_dd_asymp(x, dx, n)  = deep-OTM cancellation-free closed form (n=2..5)

Note: theta=-1 is NOT the literal arithmetic divided difference -- it is the BS
"above-the-inflection" representation (1-Cv)/sigma, which evaluates R at two positive
arguments (R(dx-x) and R(x+dx)) to avoid the cancellation that would arise in (1-Cv)/sigma
if computed via the literal divided difference for x < 0.

Built on the mathpf.mills cdef kernels (_R, _R3) via cimport.  Scalar (cdef) kernels are
nogil; the def wrappers vectorize over numpy arrays with broadcasting on (x, dx).
"""
from .mills cimport _R, _R3
import numpy as np
cimport numpy as np
np.import_array()


# -- C-level scalar kernels (cimport-able: from mathpf.mills_dd cimport _R_DD, _R_DD_asymp) --
cdef double _R_DD_asymp(double x, double dx, int n_terms) noexcept nogil:
    """Cancellation-free divided difference [R(x-dx) - R(x+dx)] / (2 dx) via the analytic
    DD of the Mills-ratio asymptotic in powers of 1/(z^2+3); deep-OTM (x - dx >= 32.7).
    n_terms = 2,3,4,5 reach ~100*eps for x - dx >= 352, 99.6, 51.2, 32.7."""
    cdef double M, t, P, q, c0, c1, c2, c3, c4
    M = x*x - dx*dx                       # = (x-dx)(x+dx) = a b = d1 d2
    t = 4.0*dx*dx                         # sigma^2 (sigma = 2 dx); polynomial coeffs in t
    P = M*M + 6.0*M + 3.0*t + 9.0        # Puw = (a^2+3)(b^2+3)
    q = 1.0/P                             # Horner in q = 1/Puw (c_j = coeff of Puw^j)
    if n_terms <= 2:                      # R^[2] convergent (CF n=2 pole)
        return (1.0 + q*(-3.0*M - t - 3.0)) / M
    if n_terms == 3:
        c0 = ((-72.0*M - 96.0*t - 504.0)*M + (-42.0*t - 432.0)*t - 1080.0)*M + ((-6.0*t - 90.0)*t - 432.0)*t - 648.0
        c1 = 30.0*M + 12.0*t + 54.0
        c2 = -3.0*M - t - 3.0
        return (1.0 + q*(c2 + q*(c1 + q*c0))) / M
    if n_terms == 4:
        c0 = (((576.0*M + 1056.0*t + 5760.0)*M + (720.0*t + 7776.0)*t + 20736.0)*M + ((216.0*t + 3456.0)*t + 18144.0)*t + 31104.0)*M + (((24.0*t + 504.0)*t + 3888.0)*t + 12960.0)*t + 15552.0
        c1 = ((-72.0*M - 96.0*t - 888.0)*M + (-42.0*t - 768.0)*t - 2808.0)*M + ((-6.0*t - 162.0)*t - 1152.0)*t - 2376.0
        c2 = 30.0*M + 12.0*t + 78.0
        c3 = -3.0*M - t - 3.0
        return (1.0 + q*(c3 + q*(c2 + q*(c1 + q*c0)))) / M
    # n_terms >= 5
    c0 = ((((-12096.0*M - 28224.0*t - 157248.0)*M + (-26208.0*t - 290304.0)*t - 798336.0)*M + ((-12096.0*t - 199584.0)*t - 1088640.0)*t - 1959552.0)*M + (((-2772.0*t - 60480.0)*t - 489888.0)*t - 1741824.0)*t - 2286144.0)*M + ((((-252.0*t - 6804.0)*t - 72576.0)*t - 381024.0)*t - 979776.0)*t - 979776.0
    c1 = (((576.0*M + 1056.0*t + 16848.0)*M + (720.0*t + 22896.0)*t + 102384.0)*M + ((216.0*t + 10260.0)*t + 90720.0)*t + 221616.0)*M + (((24.0*t + 1512.0)*t + 19764.0)*t + 94608.0)*t + 151632.0
    c2 = ((-72.0*M - 96.0*t - 888.0)*M + (-42.0*t - 768.0)*t - 4572.0)*M + ((-6.0*t - 162.0)*t - 1908.0)*t - 6156.0
    c3 = 30.0*M + 12.0*t + 78.0
    c4 = -3.0*M - t - 3.0
    return (1.0 + q*(c4 + q*(c3 + q*(c2 + q*(c1 + q*c0))))) / M


cdef double _R_DD(double x, double dx, int theta) noexcept nogil:
    """Symmetric divided difference of R about x with half-step dx:
        _R_DD(x, dx, theta) = (R(x - dx) - theta * R(x + dx)) / (2 dx)
    theta = -1: sum branch, no cancellation.  theta = +1: three regimes
    (asymp / R'''-seeded Taylor / direct mc.R difference) to keep error near eps."""
    cdef double a, xsq, r_d3, r_d1, r_d5, r_d7, dx2
    cdef int n_terms
    if theta < 0:                                       # above: sum of R's, no cancellation
        return (_R(dx - x) + _R(x + dx)) / (2.0*dx)
    a = x - dx                                          # = -d1 (smaller Mills argument)
    if a >= 51.2:                                       # deep OTM: cancellation-free DD
        if   a >= 352.0: n_terms = 2
        elif a >= 99.6:  n_terms = 3
        else:            n_terms = 4
        return _R_DD_asymp(x, dx, n_terms)
    if 2.0*dx < 3.7e-2*(1.25 + x):                      # small dx: Taylor seeded by R'''
        xsq = x*x
        r_d3 = _R3(x)                                   # = -R'''(x)
        r_d1 = (r_d3 + 1.0) / (xsq + 3.0)               # descend: -R'(x), cancellation-free (odd -> odd)
        r_d5 = (xsq + 7.0)*r_d3 - 6.0*r_d1              # ascend from accurate R''': all-x robust
        r_d7 = (xsq + 11.0)*r_d5 - 20.0*r_d3
        dx2 = dx*dx
        return r_d1 + dx2*(r_d3/6.0 + dx2*(r_d5/120.0 + dx2*r_d7/5040.0))
    # direct difference for larger dx: arguments not near-equal; mild cancellation
    return (_R(x - dx) - _R(x + dx)) / (2.0*dx)


# -- Python-callable numpy-vectorized wrappers (broadcasting on (x, dx)) --
def millsratio_dd(x, dx, theta=1):
    """Mills-ratio divided difference for BS price-to-vega ratio:
        theta = +1: returns (R(x-dx) - R(x+dx)) / (2 dx)  = Cv/sigma  (call branch)
        theta = -1: returns (R(dx-x) + R(x+dx)) / (2 dx)  = (1-Cv)/sigma  (above branch)
    Scalars or broadcastable ndarrays for x, dx; theta scalar (+/-1).  For theta=+1 a
    three-regime split (asymp / R'''-seeded Taylor / direct R difference) keeps relative
    error near eps; theta=-1 is a pure sum of two Mills ratios (no cancellation).
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


def millsratio_dd_asymp(x, dx, n_terms=4):
    """Deep-OTM cancellation-free divided difference [R(x-dx) - R(x+dx)] / (2 dx) via
    the analytic DD of the Mills-ratio 1/(z^2+3) asymptotic; valid for x - dx >= ~32.
    n_terms in {2, 3, 4, 5} (default 4) reaches ~100*eps for x - dx >= 352, 99.6, 51.2,
    32.7 respectively.  Scalars or broadcastable ndarrays for x, dx; n_terms scalar.
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
        o[i] = _R_DD_asymp(fx[i], fdx[i], nt)
    return float(out[0]) if scalar else out.reshape(bshape)
