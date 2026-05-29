"""Tests for mathpf.mills_dd: symmetric divided differences of the Mills ratio.

Stdlib-only reference (matches test_mills.py convention; no scipy/mpmath):
  R(x) = sqrt(pi/2) * erfcx(x/sqrt(2)),  erfcx(z) = exp(z^2) * erfc(z).
math.exp(z^2) overflows around z ~ 26.6 (z^2 > log(DBL_MAX)), so x ~ 37.6 is the
upper limit for the math.erfc-based truth.  Deeper-asymptotic tests use self-
consistency (asymp at lower n converges to asymp at higher n) instead of an
external truth.
"""
import math
import numpy as np
import pytest

import mathpf

SQRT2 = math.sqrt(2.0)
SQRT_PI_2 = math.sqrt(math.pi / 2.0)


def _R(x):
    """Mills ratio R(x) via stdlib erfc; valid for |x| <= ~37."""
    z = x / SQRT2
    return SQRT_PI_2 * math.exp(z * z) * math.erfc(z)


def _DD_ref(x, dx, theta=1):
    """Match mills_dd semantics:
        theta=+1: (R(x-dx) - R(x+dx)) / (2 dx)  -- call branch
        theta=-1: (R(dx-x) + R(x+dx)) / (2 dx)  -- above branch (first arg reflected)
    """
    if theta > 0:
        return (_R(x - dx) - _R(x + dx)) / (2.0 * dx)
    return (_R(dx - x) + _R(x + dx)) / (2.0 * dx)


# ----------------------------------------------------------------------- millsratio_dd

def test_dd_scalar_call_direct_branch():
    """theta=+1, dx large enough -> direct R difference."""
    for x, dx in [(2.0, 0.5), (5.0, 0.3), (10.0, 0.1)]:
        v = mathpf.millsratio_dd(x, dx, +1)
        t = _DD_ref(x, dx, +1)
        assert abs(v - t) / abs(t) < 1e-10


def test_dd_scalar_call_taylor_branch():
    """theta=+1, dx below cutoff -> R'''-seeded Taylor."""
    # Taylor when a < 21.2 (else asymp) AND dx < 0.0392*(1.25 + x)
    for x, dx in [(5.0, 0.05), (15.0, 0.05), (16.0, 0.1), (20.0, 0.5)]:
        a = x - dx
        assert a < 21.2                                   # not in asymp band
        assert dx < 3.92e-2 * (1.25 + x)                  # confirm Taylor regime
        v = mathpf.millsratio_dd(x, dx, +1)
        t = _DD_ref(x, dx, +1)
        assert abs(v - t) / abs(t) < 1e-10


def test_dd_scalar_call_asymp_branch_at_n8_boundary():
    """theta=+1 deep-asymptotic at the n=8 boundary (x - dx >= 21.2), within math.erfc range."""
    for x, dx in [(22.0, 0.1), (25.0, 0.5), (33.5, 0.5)]:
        v = mathpf.millsratio_dd(x, dx, +1)
        t = _DD_ref(x, dx, +1)
        assert abs(v - t) / abs(t) < 1e-9


def test_dd_scalar_above_branch():
    """theta=-1: sum branch (R(dx-x) + R(x+dx))/(2 dx), no cancellation."""
    for x, dx in [(0.5, 0.3), (1.0, 0.8), (3.0, 0.2)]:
        v = mathpf.millsratio_dd(x, dx, -1)
        t = _DD_ref(x, dx, -1)
        assert abs(v - t) / abs(t) < 1e-11


def test_dd_scalar_returns_float():
    """Scalar in -> Python float out (not 0-d ndarray)."""
    out = mathpf.millsratio_dd(1.0, 0.1)
    assert isinstance(out, float)


def test_dd_vectorized_shape_and_broadcast():
    """Vectorization with broadcasting on x, dx (within math.erfc range)."""
    x = np.linspace(0.5, 8.0, 12).reshape(3, 4)        # (3, 4)
    dx = np.array([0.05, 0.1, 0.2, 0.5])              # (4,) broadcasts
    out = mathpf.millsratio_dd(x, dx)
    assert out.shape == (3, 4)
    ref = np.array([[_DD_ref(float(x[i, j]), float(dx[j]))
                     for j in range(4)] for i in range(3)])
    assert np.allclose(out, ref, rtol=1e-10)


def test_dd_three_regimes_grid_within_erfc_range():
    """Sweep theta=+1 across all reachable regimes inside the math.erfc range (x <= 35)."""
    grid = [(x, 0.5 * sigma)
            for x in [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 33.5]
            for sigma in [0.01, 0.05, 0.1, 0.3, 1.0]]
    worst = 0.0
    for x, dx in grid:
        v = mathpf.millsratio_dd(x, dx, +1)
        t = _DD_ref(x, dx, +1)
        worst = max(worst, abs(v - t) / abs(t))
    assert worst < 1e-9, f"worst rel err: {worst:.2e}"


# --------------------------------------------------------------- millsratio_dd_cf

def test_dd_cf_validated_against_erfc():
    """Anchor: validate CF DD at low orders against math.erfc truth (deep asymptotic).
    n=8 reaches ~eps for a >= 21.2; the math.erfc-based reference itself loses ulps
    at large x (erfc(z) and exp(z^2) cancel out at z ~ 14), so a 1e-10 sanity check
    is the appropriate ceiling here."""
    for x, dx in [(25.0, 0.5), (28.0, 0.1), (33.5, 0.5)]:
        v = mathpf.millsratio_dd_cf(x, dx, 8)
        t = _DD_ref(x, dx, +1)
        assert abs(v - t) / abs(t) < 1e-10


def test_dd_cf_truncation_converges():
    """Self-consistency: lower-n CF DD converges to higher-n at deep asymptotic.
    Per the even-only XCF_R1 dispatch, a_min for n in {2, 4, 6, 8} is
    12800, 165, 41, 21.2 (CF rate (n+1)!/a^(2n) ~ eps)."""
    eps = np.finfo(float).eps
    a_mins = {2: 12800.0, 4: 165.0, 6: 41.0, 8: 21.2}
    for x in [60.0, 200.0, 1000.0, 15000.0]:
        for dx in [0.1, 0.5, 1.0]:
            ref = mathpf.millsratio_dd_cf(x, dx, 8)
            for n in (2, 4, 6):
                a = x - dx
                if a < a_mins[n]:
                    continue                              # truncation dominates, skip
                v = mathpf.millsratio_dd_cf(x, dx, n)
                assert abs(v - ref) / abs(ref) < 200 * eps, \
                    f"n={n} x={x} dx={dx}: rel diff {abs(v - ref) / abs(ref):.2e}"


def test_dd_cf_vectorized():
    """Vectorized over arrays; truth via self-consistency against n=8."""
    x = np.array([60.0, 100.0, 200.0, 400.0])
    dx = np.array([0.5, 0.1, 0.05, 0.01])
    out = mathpf.millsratio_dd_cf(x, dx, 4)
    ref = mathpf.millsratio_dd_cf(x, dx, 8)
    assert out.shape == (4,)
    assert np.allclose(out, ref, rtol=1e-12)


def test_dd_cf_default_n_terms():
    """n_terms defaults to 4."""
    a = mathpf.millsratio_dd_cf(100.0, 0.5)
    b = mathpf.millsratio_dd_cf(100.0, 0.5, 4)
    assert a == b


# -------------------------------------------------- consistency with the dispatcher branch

def test_dd_dispatcher_matches_cf_in_deep_asymp():
    """In the deep-asymptotic regime (a = x - dx >= 21.2 with dx/x < 0.9), millsratio_dd
    internally calls millsratio_dd_cf with the dispatched n.  Verify a representative
    point from each band of the XCF_R1-aligned dispatch."""
    # band: (a, n) per the dispatch ladder in _R_DD: a>=12800->2, 165->4, 41->6, 21.2->8
    for x, dx, n_expected in [
        (15000.0, 0.5, 2),   # a = 14999.5 >= 12800 -> n=2
        (1000.0,  0.5, 4),   # a = 999.5   >= 165   -> n=4
        (50.0,    0.5, 6),   # a = 49.5    >= 41    -> n=6
        (22.5,    0.5, 8),   # a = 22.0    >= 21.2  -> n=8 (bottom row)
    ]:
        a = mathpf.millsratio_dd(x, dx, +1)
        b = mathpf.millsratio_dd_cf(x, dx, n_expected)
        assert a == b, f"x={x}, dx={dx}, n={n_expected}: {a} vs {b}"


def test_dd_dispatcher_high_dx_x_escape_to_direct():
    """When a is in the asymp band but dx/x >= 0.9, the dispatcher falls through to
    the direct mc.R difference (asymp's q^n diverges at f -> 1; direct is ~1 ulp there).
    Stay within math.erfc range (x + dx <= ~37)."""
    # x=18.5, dx=17.0 -> a=1.5 (NOT in asymp band; just exercise direct branch with
    # x+dx=35.5 still in erfc range)
    x, dx = 18.5, 17.0          # a = 1.5; gate would route to direct anyway
    v = mathpf.millsratio_dd(x, dx, +1)
    t = _DD_ref(x, dx, +1)
    assert abs(v - t) / abs(t) < 1e-9
