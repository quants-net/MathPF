# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""Symmetric divided differences of the Mills ratio R(x) = N(-x)/n(x).

For a midpoint x and half-step dx (so the two evaluation points are x - dx and x + dx):

    millsratio_dd(x, dx, theta=+1) = (R(x-dx) - R(x+dx)) / (2 dx)  (difference branch)
    millsratio_dd(x, dx, theta=-1) = (R(dx-x) + R(x+dx)) / (2 dx)  (sum branch; first
                                     argument reflected so both R-evaluations land on
                                     positive arguments)
    millsratio_dd_cf(x, dx, n)     = deep-asymptotic cancellation-free divided difference via
                                     the coupled (V, T, Pa, Pb) recurrence on the
                                     depth-n CF convergent of R; algorithmic loop with
                                     no precomputed coefficient table, table-free.

Note: theta=-1 is NOT the literal arithmetic divided difference -- the first argument
is reflected (R(dx-x) instead of R(x-dx)) so that both R-evaluations land on positive
arguments, avoiding cancellation when x < 0.

Built on the mathpf.mills cdef kernels (_R, _R3) via cimport.  Scalar (cdef) kernels are
nogil; the def wrappers vectorize over numpy arrays with broadcasting on (x, dx).

History note: an earlier version dispatched the asymp branch to a 1/x^2 algorithmic
loop with a 9x9 precomputed coefficient table (millsratio_dd_asymp_x2).  After raising
_ASYMP_X2_MIN_A from 14.2 to 22.9 (the asymp branch now fires much less frequently),
the table's amortised cost no longer paid for itself, so the dispatch was replaced with
the cancellation-free CF recurrence (slightly more multiplies per call, but no table
and only one division -- and at the asymp band's low frequency the per-call cost is in
the noise).
"""
from .mills cimport _R, _R3, _R013_CF
import numpy as np
cimport numpy as np
np.import_array()

cdef extern from "_mills_coef.h":      # single source of truth for the CF cutoffs (shared with mills.pyx)
    const double XCF_R1[6]             # R_hat_1's CF cutoffs (\epsilon at n = 12, 10, 8, 6, 4, 2); the asymp
                                       # branch's a-dispatch ladder reads XCF_R1[2..5] (= 21.2, 41, 165, 12800)
                                       # so its Taylor/CF split and n-step thresholds stay structurally aligned
                                       # with R_hat_1's CF convergent rate (n+1)!/a^{2n} ~ eps.
    const double XCF_R3[5]             # XCF_R3[0] = 17.1: smallest x where R3's CF n=12 convergent is at eps;
                                       # the Taylor branch's upper sub-range [17.1, ~22) bypasses _R3's full
                                       # dispatch and calls _R013_CF(u, 12, 3) directly, sharing u = x^2 + 3
                                       # with the Taylor ascent recurrence.


# -- C-level scalar kernels (cimport-able: from mathpf.mills_dd cimport _R_DD, _R_DD_CF) --
cdef double _R_DD_CF(double x, double dx, int n_terms) noexcept nogil:
    """[R(x-dx) - R(x+dx)] / (2 dx) via the coupled (V, T, P^a, P^b) recurrence on
    the depth-n_terms CF convergent of R.  Cancellation-free for any x > dx > 0
    (both a = x - dx and b = x + dx positive).  Zero divisions in the loop; one final
    divide.  Per iteration: 11 mults; total ~ 11 n_terms + setup.

    With a = x - dx and b = x + dx, the recurrence is

        V_{k+1}  = a b V_k + k^2 V_{k-1} + k T_k
        T_{k+1}  = (a^2+b^2) V_k + 2k a b V_{k-1} + k(k-1) T_{k-1}
        P^[a]_{k+1} = a P^[a]_k + k P^[a]_{k-1}   (similarly P^[b])

    seeded by V_0 = 0, V_1 = 1, T_1 = -1, T_2 = a^2 + b^2 + 1.  The final slope is
    V_n / (P^[a]_n P^[b]_n).  The setup quantities a b and a^2 + b^2 are formed
    directly from the already-computed a, b -- one fewer multiply than via
    (x, dx), and cancellation-safe at dx -> x by Sterbenz on (x - dx) (the
    explicit x^2 - dx^2 form would lose precision proportional to 1/(1-(dx/x)^2)).
    """
    cdef double a, b, ab, s2
    cdef double V_p, V_c, V_n, T_p, T_c, T_n
    cdef double Pa_p, Pa_c, Pa_n, Pb_p, Pb_c, Pb_n
    cdef int k
    a = x - dx
    b = x + dx
    if n_terms == 0:
        return 1.0 / (a * b)
    ab  = a * b                                          # = x^2 - dx^2; Sterbenz-safe at dx -> x via the exact (x - dx)
    s2  = a*a + b*b                                      # = 2 (x^2 + dx^2); one fewer mult than 2.0 * (x*x + dx*dx)
    V_p,  V_c  = 1.0,  ab - 1.0                          # V_1, V_2
    T_p,  T_c  = -1.0, s2 + 1.0                          # T_1, T_2
    Pa_p, Pa_c = a, a*a + 1.0                            # P^[a]_1, P^[a]_2
    Pb_p, Pb_c = b, b*b + 1.0                            # P^[b]_1, P^[b]_2
    for k in range(2, n_terms + 1):
        V_n  = ab * V_c + k * (k * V_p + T_c)
        T_n  = s2 * V_c + k * (2.0 * ab * V_p + (k - 1) * T_p)   # 2 a b V_p inlined (compiler hoists 2*ab as loop-invariant)
        Pa_n = a * Pa_c + k * Pa_p
        Pb_n = b * Pb_c + k * Pb_p
        V_p,  V_c  = V_c,  V_n
        T_p,  T_c  = T_c,  T_n
        Pa_p, Pa_c = Pa_c, Pa_n
        Pb_p, Pb_c = Pb_c, Pb_n
    return V_c / (Pa_c * Pb_c)


cdef double _R_DD(double x, double dx, int theta) noexcept nogil:
    """Symmetric divided difference of R about x with half-step dx:
        _R_DD(x, dx, theta) = (R(x - dx) - theta * R(x + dx)) / (2 dx)
    theta = -1: sum branch, no cancellation.  theta = +1: three regimes
    (CF asymp / 5-term R'''-seeded Taylor / direct R difference) keep relative error
    bounded across the (x, dx) plane (overall worst ~64 eps, set by the Taylor branch's
    truncation at its gate corner)."""
    cdef double a, u, r_d3, r_d1, r_d5, r_d7, r_d9, dx2
    if theta < 0:                                       # above: sum of R's, no cancellation
        return (_R(dx - x) + _R(x + dx)) / (2.0*dx)
    a = x - dx                                          # = -d1 (smaller Mills argument)
    if dx < 3.92e-2*(1.25 + x):                         # unified small-dx gate: asymp ladder or Taylor by a
        # Asymp CF DD ladder, even-only n aligned with R_hat_1's CF convergent rate (n+1)!/a^{2n} ~ eps:
        # thresholds are exactly R_hat_1's CF cutoffs at n = 2, 4, 6, 8 (XCF_R1[5..2]).  Falls through
        # to Taylor when a < XCF_R1[2] (= 21.2).
        if   a >= XCF_R1[5]: return _R_DD_CF(x, dx, 2)  # a >= 12800
        elif a >= XCF_R1[4]: return _R_DD_CF(x, dx, 4)  # a >= 165
        elif a >= XCF_R1[3]: return _R_DD_CF(x, dx, 6)  # a >= 41
        elif a >= XCF_R1[2]: return _R_DD_CF(x, dx, 8)  # a in [21.2, 41): bottom row
        # a < XCF_R1[2] (21.2): 5-term Taylor seeded by R''' (N=5 balance, ~38 eps at gate)
        u = x*x + 3.0                                   # kernel's shifted variable; shared with _R013_CF, descent, and ascent
        if x >= XCF_R3[0]:                              # x in [17.1, ~22): _R3's CF n=12 path; call directly with our u
            r_d3 = _R013_CF(u, 12, 3)                   # skips _R3's sign/U_MAX/ladder dispatch + redundant x*x + 3
        else:                                           # x < 17.1: segmented Chebyshev path inside _R3
            r_d3 = _R3(x)                               # = -R'''(x)
        r_d1 = (r_d3 + 1.0) / u                         # descend: -R'(x), cancellation-free (odd -> odd); u = x^2 + 3
        r_d5 = (u +  4.0)*r_d3 -  6.0*r_d1              # ascend: r_{2k+1} = (x^2+4k-1) r_{2k-1} - (2k-1)(2k-2) r_{2k-3};
        r_d7 = (u +  8.0)*r_d5 - 20.0*r_d3              # (x^2 + 7/11/15) rewritten as (u + 4/8/12) since u = x^2 + 3
        r_d9 = (u + 12.0)*r_d7 - 42.0*r_d5
        dx2 = dx*dx                                     # Taylor in dx; nested with consecutive denominators (2k+1)(2k+2)
        return r_d1 + dx2/6.0*(r_d3 + dx2/20.0*(r_d5 + dx2/42.0*(r_d7 + dx2/72.0*r_d9)))
    # dx >= gate*(1.25+x): direct difference (cancellation bounded by m >= m*, ~eps/m)
    return (_R(x - dx) - _R(x + dx)) / (2.0*dx)


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
    recurrence (table-free, algorithmic loop).  n_terms >= 0 supported; for the dispatch
    thresholds used inside _R_DD, see the (a_min, n_terms) ladder in _R_DD's source.
    Scalars or broadcastable ndarrays for x, dx; n_terms scalar.
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
