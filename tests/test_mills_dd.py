"""Tests for mathpf.mills_dd: Mills-ratio divided differences for BS price-to-vega.

Stdlib-only reference (matches test_mills.py convention; no scipy/mpmath):
  R(x) = sqrt(pi/2) * erfcx(x/sqrt(2)),  erfcx(z) = exp(z^2) * erfc(z).
math.exp(z^2) overflows around z ~ 26.6 (z^2 > log(DBL_MAX)), so x ~ 37.6 is the
upper limit for the math.erfc-based truth.  Deeper-OTM tests use self-consistency
(asymp at lower n converges to asymp at higher n) instead of an external truth.
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
    # Taylor when a < 17.1 (else asymp) AND dx < 0.0392*(1.25 + x)
    for x, dx in [(5.0, 0.05), (15.0, 0.05), (16.0, 0.1)]:
        a = x - dx
        assert a < 17.1                                   # not in asymp band
        assert dx < 3.92e-2 * (1.25 + x)                  # confirm Taylor regime
        v = mathpf.millsratio_dd(x, dx, +1)
        t = _DD_ref(x, dx, +1)
        assert abs(v - t) / abs(t) < 1e-10


def test_dd_scalar_call_asymp_branch_at_n10_boundary():
    """theta=+1 deep-OTM at the n=10 boundary (x - dx >= 17.1), within math.erfc range."""
    for x, dx in [(17.5, 0.1), (20.0, 0.5), (33.5, 0.5)]:
        v = mathpf.millsratio_dd(x, dx, +1)
        t = _DD_ref(x, dx, +1)
        assert abs(v - t) / abs(t) < 1e-9


def test_dd_scalar_above_branch():
    """theta=-1: BS above-inflection form (R(dx-x) + R(x+dx))/(2 dx), no cancellation."""
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


# --------------------------------------------------------------- millsratio_dd_asymp_x2

def test_dd_asymp_x2_validated_against_erfc():
    """Anchor: validate asymp_x2 at low orders against math.erfc truth (deep OTM).
    n=10 reaches ~eps for a >= 17.1; the math.erfc-based reference itself loses ulps
    at large x (erfc(z) and exp(z^2) cancel out at z ~ 14), so a 1e-10 sanity check
    is the appropriate ceiling here."""
    for x, dx in [(20.0, 0.5), (25.0, 0.1), (33.5, 0.5)]:
        v = mathpf.millsratio_dd_asymp_x2(x, dx, 10)
        t = _DD_ref(x, dx, +1)
        assert abs(v - t) / abs(t) < 1e-10


def test_dd_asymp_x2_truncation_converges():
    """Self-consistency: lower-n asymp_x2 converges to higher-n at deep OTM.
    Per the static table, a_min for n=2..10 is 883, 213, 92.7, 54.0, 37.1, 28.2, 22.9, 19.5, 17.1."""
    eps = np.finfo(float).eps
    a_mins = {2: 883.0, 3: 213.0, 4: 92.7, 5: 54.0, 6: 37.1,
              7: 28.2, 8: 22.9, 9: 19.5, 10: 17.1}
    for x in [60.0, 100.0, 200.0, 400.0, 1000.0]:
        for dx in [0.1, 0.5, 1.0]:
            ref = mathpf.millsratio_dd_asymp_x2(x, dx, 10)
            for n in range(2, 10):
                a = x - dx
                if a < a_mins[n]:
                    continue                              # truncation dominates, skip
                v = mathpf.millsratio_dd_asymp_x2(x, dx, n)
                assert abs(v - ref) / abs(ref) < 200 * eps, \
                    f"n={n} x={x} dx={dx}: rel diff {abs(v - ref) / abs(ref):.2e}"


def test_dd_asymp_x2_vectorized():
    """Vectorized over arrays; truth via self-consistency against n=10."""
    x = np.array([60.0, 100.0, 200.0, 400.0])
    dx = np.array([0.5, 0.1, 0.05, 0.01])
    out = mathpf.millsratio_dd_asymp_x2(x, dx, 4)
    ref = mathpf.millsratio_dd_asymp_x2(x, dx, 10)
    assert out.shape == (4,)
    assert np.allclose(out, ref, rtol=1e-12)


def test_dd_asymp_x2_default_n_terms():
    """n_terms defaults to 4."""
    a = mathpf.millsratio_dd_asymp_x2(100.0, 0.5)
    b = mathpf.millsratio_dd_asymp_x2(100.0, 0.5, 4)
    assert a == b


# -------------------------------------------------- consistency with the dispatcher branch

def test_dd_dispatcher_matches_asymp_x2_in_deep_otm():
    """In the deep-OTM regime (a = x - dx >= 17.1 with dx/x < 0.9), millsratio_dd
    internally calls millsratio_dd_asymp_x2 with the table-dispatched n.  Verify a
    few points from each band."""
    # band: (a, n) per _ASYMP_X2_N_TABLE: a>=883->2, 213->3, 92.7->4, 54->5, ...,17.1->10
    for x, dx, n_expected in [
        (1000.0, 0.5, 2),    # a = 999.5 >= 883 -> n=2
        (300.0, 0.5, 3),     # a = 299.5 >= 213 -> n=3
        (100.0, 0.5, 4),     # a = 99.5 >= 92.7 -> n=4
        (60.0, 0.5, 5),      # a = 59.5 >= 54.0 -> n=5
        (40.0, 0.5, 6),      # a = 39.5 >= 37.1 -> n=6
        (30.0, 0.5, 7),      # a = 29.5 >= 28.2 -> n=7
        (24.0, 0.5, 8),      # a = 23.5 >= 22.9 -> n=8
        (20.0, 0.1, 9),      # a = 19.9 >= 19.5 -> n=9
        (18.0, 0.1, 10),     # a = 17.9 >= 17.1 -> n=10
    ]:
        a = mathpf.millsratio_dd(x, dx, +1)
        b = mathpf.millsratio_dd_asymp_x2(x, dx, n_expected)
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
