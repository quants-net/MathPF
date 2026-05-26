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
    """theta=+1, sigma >= cutoff -> direct R difference."""
    for x, dx in [(2.0, 0.5), (5.0, 0.3), (10.0, 0.1)]:
        v = mathpf.millsratio_dd(x, dx, +1)
        t = _DD_ref(x, dx, +1)
        assert abs(v - t) / abs(t) < 1e-10


def test_dd_scalar_call_taylor_branch():
    """theta=+1, sigma below cutoff -> R'''-seeded Taylor."""
    # Taylor when 2*dx < 0.037*(1.25 + x)
    for x, dx in [(5.0, 0.05), (15.0, 0.05), (30.0, 0.05)]:
        assert 2.0 * dx < 3.7e-2 * (1.25 + x)            # confirm we're in Taylor regime
        v = mathpf.millsratio_dd(x, dx, +1)
        t = _DD_ref(x, dx, +1)
        assert abs(v - t) / abs(t) < 1e-10


def test_dd_scalar_call_asymp_branch_at_n5_boundary():
    """theta=+1 deep-OTM at the n=5 boundary (x - dx ~ 33), still within math.erfc range."""
    for x, dx in [(33.5, 0.5), (34.0, 0.1)]:
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


# ------------------------------------------------------------------ millsratio_dd_asymp

def test_dd_asymp_n5_validated_against_erfc():
    """Anchor: validate n=5 at its accuracy boundary (x - dx >= 32.7) against math.erfc."""
    for x, dx in [(33.5, 0.5), (35.0, 0.1)]:
        v = mathpf.millsratio_dd_asymp(x, dx, 5)
        t = _DD_ref(x, dx, +1)
        assert abs(v - t) / abs(t) < 1e-10


def test_dd_asymp_truncation_converges():
    """Self-consistency: lower-n asymp converges to higher-n asymp at deep-OTM.
    At x >> n_terms boundary, asymp(n) - asymp(n+1) is the next truncation term and ~ 0."""
    eps = np.finfo(float).eps
    for x in [60.0, 100.0, 200.0, 400.0]:
        for dx in [0.1, 0.5, 1.0]:
            ref = mathpf.millsratio_dd_asymp(x, dx, 5)
            for n in [2, 3, 4]:
                v = mathpf.millsratio_dd_asymp(x, dx, n)
                # tolerance: 200*eps once we're well past the n-boundary
                # (n=2 wants x-dx>=352; n=3 wants >=99.6; n=4 wants >=51.2)
                if n == 2 and x < 352.0:
                    continue                              # truncation dominates, skip
                if n == 3 and x < 99.6:
                    continue
                assert abs(v - ref) / abs(ref) < 200 * eps, \
                    f"n={n} x={x} dx={dx}: rel diff {abs(v - ref) / abs(ref):.2e}"


def test_dd_asymp_vectorized():
    """Vectorized over arrays; truth via self-consistency against n=5."""
    x = np.array([60.0, 100.0, 200.0, 400.0])
    dx = np.array([0.5, 0.1, 0.05, 0.01])
    out = mathpf.millsratio_dd_asymp(x, dx, 4)
    ref = mathpf.millsratio_dd_asymp(x, dx, 5)
    assert out.shape == (4,)
    assert np.allclose(out, ref, rtol=1e-12)


def test_dd_asymp_default_n_terms():
    """n_terms defaults to 4."""
    a = mathpf.millsratio_dd_asymp(60.0, 0.5)
    b = mathpf.millsratio_dd_asymp(60.0, 0.5, 4)
    assert a == b


# -------------------------------------------------- consistency with the dispatcher branch

def test_dd_dispatcher_matches_asymp_in_deep_otm():
    """In the deep-OTM regime (x - dx >= 51.2 with x - dx < 99.6), millsratio_dd
    internally calls millsratio_dd_asymp(n=4)."""
    for x, dx in [(60.0, 0.5), (80.0, 0.3), (95.0, 0.1)]:
        a = mathpf.millsratio_dd(x, dx, +1)
        b = mathpf.millsratio_dd_asymp(x, dx, 4)
        assert a == b
