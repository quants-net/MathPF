# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""Symmetric divided differences of the Mills ratio R(x) = N(-x)/n(x), in the form used
by the Black-Scholes price-to-vega ratio.

For a midpoint x and half-step dx (so the two evaluation points are x - dx and x + dx;
in BS notation x = m0 = logk/sigma, dx = sigma/2, x - dx = -d1, x + dx = -d2):

    millsratio_dd(x, dx, theta=+1)   = (R(x-dx) - R(x+dx)) / (2 dx)  = Cv/sigma  (call)
    millsratio_dd(x, dx, theta=-1)   = (R(dx-x) + R(x+dx)) / (2 dx)  = (1-Cv)/sigma (above)
    millsratio_dd_asymp_x2(x, dx, n) = deep-OTM cancellation-free divided difference via
                                       the plain 1/x^2 asymptotic; algorithmic loop over
                                       orders 0..n, dispatched by _R_DD from a static
                                       (a_min, n_terms) table.  Internal: _R_DD_asymp_x2p3
                                       (the 1/(x^2+3) "shift 3" form, n=2..5) is kept as
                                       a cdef-only cross-check, not part of the public API.

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


# Precomputed signed, double-factorial-folded coefficients for _R_DD_asymp_x2.
# Row j (length j+1) = [ (-1)^j * (2j-1)!! * C(2j+1, 2k+1) for k = 0..j ]
# (2j-1)!! convention: (-1)!! = 1.  Stored as a dense 13x13 numpy array (unused
# entries are 0); typed memoryview gives nogil-safe access from the cdef kernel.
# Rows up to j=12 support n_terms <= 12; entries with |v| > 2^53 (in j=12) lose
# a few lower bits but their contribution to R_DD is scaled by q^12 ~ eps so the
# loss is harmless (verified <= 1 ulp vs mpmath across the asymp band).
_ASYMP_X2_TABLE_PY = np.array([
    [                  1.0,                  0.0,                  0.0,                  0.0,                  0.0,                  0.0,                  0.0,                  0.0,                  0.0,                  0.0,                  0.0,                  0.0,                  0.0],
    [                 -3.0,                 -1.0,                  0.0,                  0.0,                  0.0,                  0.0,                  0.0,                  0.0,                  0.0,                  0.0,                  0.0,                  0.0,                  0.0],
    [                 15.0,                 30.0,                  3.0,                  0.0,                  0.0,                  0.0,                  0.0,                  0.0,                  0.0,                  0.0,                  0.0,                  0.0,                  0.0],
    [               -105.0,               -525.0,               -315.0,                -15.0,                  0.0,                  0.0,                  0.0,                  0.0,                  0.0,                  0.0,                  0.0,                  0.0,                  0.0],
    [                945.0,               8820.0,              13230.0,               3780.0,                105.0,                  0.0,                  0.0,                  0.0,                  0.0,                  0.0,                  0.0,                  0.0,                  0.0],
    [             -10395.0,            -155925.0,            -436590.0,            -311850.0,             -51975.0,               -945.0,                  0.0,                  0.0,                  0.0,                  0.0,                  0.0,                  0.0,                  0.0],
    [             135135.0,            2972970.0,           13378365.0,           17837820.0,            7432425.0,             810810.0,              10395.0,                  0.0,                  0.0,                  0.0,                  0.0,                  0.0,                  0.0],
    [           -2027025.0,          -61486425.0,         -405810405.0,         -869593725.0,         -676350675.0,         -184459275.0,          -14189175.0,            -135135.0,                  0.0,                  0.0,                  0.0,                  0.0,                  0.0],
    [           34459425.0,         1378377000.0,        12543230700.0,        39421582200.0,        49276977750.0,        25086461400.0,         4824319500.0,          275675400.0,            2027025.0,                  0.0,                  0.0,                  0.0,                  0.0],
    [         -654729075.0,       -33391182825.0,      -400694193900.0,     -1736341506900.0,     -3183292762650.0,     -2604512260350.0,      -934953119100.0,      -133564731300.0,        -5892561675.0,          -34459425.0,                  0.0,                  0.0,                  0.0],
    [        13749310575.0,       870789669750.0,     13323081947175.0,     76131896841000.0,    192444517014750.0,    230933420417700.0,    133230819471750.0,     35528218525800.0,      3918553513875.0,       137493105750.0,          654729075.0,                  0.0,                  0.0],
    [      -316234143225.0,    -24350029028325.0,   -462650551538175.0,  -3370739732635275.0, -11235799108784250.0, -18590140343624848.0, -15730118752297950.0,  -6741479465270550.0,  -1387951654614525.0,   -121750145141625.0,     -3478575575475.0,       -13749310575.0,                  0.0],
    [      7905853580625.0,    727338529417500.0,  16801520029544250.0, 152013752648257504.0, 646058448755094400.0, 1409582070011115008.0, 1644512415012967424.0, 1033693518008151040.0, 342030943458579392.0, 56005066765147504.0,  4000361911796250.0,     94870242967500.0,       316234143225.0],
], dtype=np.float64)
cdef double[:, ::1] _ASYMP_X2_TABLE = _ASYMP_X2_TABLE_PY

cdef int _ASYMP_X2_K_MAX = 12            # row index range [0, K_MAX]
cdef double _ASYMP_X2_MIN_A     = 14.2   # smallest a = x - dx covered by asymp_x2 (n=12)
cdef double _ASYMP_X2_MAX_DX_X  = 0.9    # asymp only when dx/x < 0.9 (else direct is ~1 ulp)


# -- C-level scalar kernels (cimport-able: from mathpf.mills_dd cimport _R_DD, _R_DD_asymp_x2, _R_DD_asymp_x2p3) --
cdef double _R_DD_asymp_x2(double x, double dx, int n_terms) noexcept nogil:
    """[R(x-dx) - R(x+dx)] / (2 dx) via the plain 1/x^2 Mills-ratio asymptotic, computed
    as an algorithmic loop (vs. Jaeckel's inline hand-rolled polynomial), with order
    n_terms tunable at call time.

    With p = x^2 - dx^2, q = x^2/p^2 (Jaeckel's q), and e = (dx/x)^2:

        R_DD_asymp_x2 = (1/p) sum_{j=0}^{n_terms} (-1)^j (2j-1)!! M_j(e) q^j

    where M_j(e) = sum_{k=0}^j C(2j+1, 2k+1) e^k (odd-indexed binomials of 2j+1).
    The precomputed table (_ASYMP_X2_TABLE) folds in the (-1)^j (2j-1)!! prefactor,
    so the loop reduces to Horner-in-e + multiply-accumulate in q.  Cancellation-free
    for any x > dx > 0.  n_terms valid in [0, 10]; out-of-range returns 0.0.
    """
    cdef double p, q, e, q_acc, s, inner
    cdef int j, k
    if n_terms < 0 or n_terms > _ASYMP_X2_K_MAX:
        return 0.0
    p = x*x - dx*dx
    q = (x*x) / (p*p)
    e = (dx*dx) / (x*x)
    q_acc = 1.0
    s = 0.0
    for j in range(n_terms + 1):
        inner = _ASYMP_X2_TABLE[j, j]
        for k in range(j - 1, -1, -1):
            inner = inner * e + _ASYMP_X2_TABLE[j, k]
        s = s + q_acc * inner
        q_acc = q_acc * q
    return s / p


cdef double _R_DD_asymp_x2p3(double x, double dx, int n_terms) noexcept nogil:
    """Cancellation-free divided difference [R(x-dx) - R(x+dx)] / (2 dx) via the analytic
    DD of the Mills-ratio asymptotic in powers of 1/(z^2+3) ("shift 3"); deep-OTM.
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
    (asymp_x2 / 5-term R'''-seeded Taylor / direct R difference) keep relative error
    within ~38 eps at the worst-case Taylor/direct gate boundary."""
    cdef double a, xsq, r_d3, r_d1, r_d5, r_d7, r_d9, dx2
    cdef int n_terms
    if theta < 0:                                       # above: sum of R's, no cancellation
        return (_R(dx - x) + _R(x + dx)) / (2.0*dx)
    a = x - dx                                          # = -d1 (smaller Mills argument)
    if a >= _ASYMP_X2_MIN_A and dx < _ASYMP_X2_MAX_DX_X*x:  # deep OTM AND dx/x < 0.9
        if   a >= 883.0: n_terms = 2
        elif a >= 213.0: n_terms = 3
        elif a >=  92.7: n_terms = 4
        elif a >=  54.0: n_terms = 5
        elif a >=  37.1: n_terms = 6
        elif a >=  28.2: n_terms = 7
        elif a >=  22.9: n_terms = 8
        elif a >=  19.5: n_terms = 9
        elif a >=  17.1: n_terms = 10
        else:            n_terms = 12         # a in [14.2, 17.1): ~103 mults
        return _R_DD_asymp_x2(x, dx, n_terms)
    # dx/x >= 0.9 (with a in asymp band) falls through to the direct difference: R(x-dx)/R(x+dx)
    # >= 19 there, so subtraction loses < 1 bit (~1 ulp).
    if dx < 3.92e-2*(1.25 + x):                         # small dx (dx < 0.0392(1.25+x)): 5-term Taylor seeded by R''' (N=5 balance, ~38 eps at gate)
        xsq = x*x
        r_d3 = _R3(x)                                   # = -R'''(x)
        r_d1 = (r_d3 + 1.0) / (xsq + 3.0)               # descend: -R'(x), cancellation-free (odd -> odd)
        r_d5 = (xsq +  7.0)*r_d3 -  6.0*r_d1            # ascend: r_{2k+1} = (x^2+4k-1) r_{2k-1} - (2k-1)(2k-2) r_{2k-3}
        r_d7 = (xsq + 11.0)*r_d5 - 20.0*r_d3
        r_d9 = (xsq + 15.0)*r_d7 - 42.0*r_d5
        dx2 = dx*dx                                     # Taylor in dx; nested with consecutive denominators (2k+1)(2k+2)
        return r_d1 + dx2/6.0*(r_d3 + dx2/20.0*(r_d5 + dx2/42.0*(r_d7 + dx2/72.0*r_d9)))
    # direct difference for larger dx: arguments not near-equal; mild cancellation
    return (_R(x - dx) - _R(x + dx)) / (2.0*dx)


# -- Python-callable numpy-vectorized wrappers (broadcasting on (x, dx)) --
def millsratio_dd(x, dx, theta=1):
    """Mills-ratio divided difference for BS price-to-vega ratio:
        theta = +1: returns (R(x-dx) - R(x+dx)) / (2 dx)  = Cv/sigma  (call branch)
        theta = -1: returns (R(dx-x) + R(x+dx)) / (2 dx)  = (1-Cv)/sigma  (above branch)
    Scalars or broadcastable ndarrays for x, dx; theta scalar (+/-1).  For theta=+1 a
    three-regime split (asymp_x2 / 5-term R'''-seeded Taylor / direct R difference)
    keeps relative error within ~38 eps at the worst-case boundary; theta=-1 is a pure
    sum of two Mills ratios (no cancellation).
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


def millsratio_dd_asymp_x2(x, dx, n_terms=4):
    """Deep-OTM cancellation-free divided difference via the plain 1/x^2 Mills-ratio
    asymptotic (algorithmic loop over orders).  n_terms in [0, 10]; for the dispatch
    used inside _R_DD see the (a_min, n_terms) table in the module source.  Scalars
    or broadcastable ndarrays for x, dx; n_terms scalar."""
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
        o[i] = _R_DD_asymp_x2(fx[i], fdx[i], nt)
    return float(out[0]) if scalar else out.reshape(bshape)


