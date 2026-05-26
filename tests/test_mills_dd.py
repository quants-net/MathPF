"""Tests for mathpf.mills_dd: Mills-ratio divided differences for BS price-to-vega.

Reference truth uses scipy.special.erfcx (no overflow up to x ~ 27, where erfcx itself
becomes the limiting factor for the math.erfc-based shortcut).
"""
import math
import numpy as np
import scipy.special as _spsp
import pytest

import mathpf

SQRT1_2 = math.sqrt(0.5)
SQRT_PI_2 = math.sqrt(math.pi / 2.0)


def _R(x):
    return SQRT_PI_2 * _spsp.erfcx(x * SQRT1_2)


def _DD_ref(x, dx, theta=1):
    """Match mills_dd semantics:
        theta=+1: (R(x-dx) - R(x+dx)) / (2 dx)  -- call branch
        theta=-1: (R(dx-x) + R(x+dx)) / (2 dx)  -- above branch (R(|x-dx|), reflected)
    """
    if theta > 0:
        return (_R(x - dx) - _R(x + dx)) / (2.0 * dx)
    return (_R(dx - x) + _R(x + dx)) / (2.0 * dx)


# ----------------------------------------------------------------------- millsratio_dd

def test_dd_scalar_call_branch():
    """theta=+1 across the three internal regimes (asymp / Taylor / direct)."""
    # direct branch:    x=2,  dx=0.5  -> a = 1.5
    # Taylor branch:    x=5,  dx=0.05 -> a = 4.95, 2dx=0.1 < 0.037*(1.25+5)=0.231
    # asymp branch:     x=60, dx=0.5  -> a = 59.5 (>= 51.2 -> n=4)
    for x, dx, tol in [(2.0, 0.5, 1e-11),
                       (5.0, 0.05, 1e-11),
                       (60.0, 0.5, 1e-10)]:
        v = mathpf.millsratio_dd(x, dx, +1)
        t = _DD_ref(x, dx, +1)
        assert abs(v - t) / abs(t) < tol


def test_dd_scalar_above_branch():
    """theta=-1: sum branch, no cancellation."""
    for x, dx in [(0.5, 0.3), (1.0, 0.8), (3.0, 0.2)]:
        v = mathpf.millsratio_dd(x, dx, -1)
        t = _DD_ref(x, dx, -1)
        assert abs(v - t) / abs(t) < 1e-11


def test_dd_scalar_returns_float():
    """Scalar in -> Python float out (not 0-d ndarray)."""
    out = mathpf.millsratio_dd(1.0, 0.1)
    assert isinstance(out, float)


def test_dd_vectorized_shape_and_broadcast():
    """Vectorization with broadcasting on x, dx."""
    x = np.linspace(0.5, 8.0, 12).reshape(3, 4)        # (3, 4)
    dx = np.array([0.05, 0.1, 0.2, 0.5])              # (4,) broadcasts
    out = mathpf.millsratio_dd(x, dx)
    assert out.shape == (3, 4)
    ref = np.array([[_DD_ref(float(x[i, j]), float(dx[j]))
                     for j in range(4)] for i in range(3)])
    assert np.allclose(out, ref, rtol=1e-10)


def test_dd_three_regimes_grid():
    """Sweep all three theta=+1 regimes against the erfc-based truth."""
    grid = []
    for x in [0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 200.0, 400.0]:
        for sigma in [0.01, 0.05, 0.1, 0.3, 1.0]:
            grid.append((x, 0.5 * sigma))
    worst = 0.0
    for x, dx in grid:
        v = mathpf.millsratio_dd(x, dx, +1)
        t = _DD_ref(x, dx, +1)
        worst = max(worst, abs(v - t) / abs(t))
    assert worst < 1e-10, f"worst rel err: {worst:.2e}"


# ------------------------------------------------------------------ millsratio_dd_asymp

def test_dd_asymp_truncation_targets():
    """n_terms = 2/3/4/5 should reach ~100 eps at a >= 352/99.6/51.2/32.7."""
    eps = np.finfo(float).eps
    for a_min, n_terms, tol in [(352.0, 2, 200 * eps),
                                (99.6,  3, 200 * eps),
                                (51.2,  4, 200 * eps),
                                (32.7,  5, 200 * eps)]:
        x = a_min + 0.5
        dx = 0.5
        v = mathpf.millsratio_dd_asymp(x, dx, n_terms)
        t = _DD_ref(x, dx, +1)
        assert abs(v - t) / abs(t) < tol, \
            f"n={n_terms} at a={a_min}: rel err {abs(v - t) / abs(t):.2e} vs tol {tol:.2e}"


def test_dd_asymp_vectorized():
    x = np.array([60.0, 100.0, 200.0, 400.0])
    dx = np.array([0.5, 0.1, 0.05, 0.01])
    out = mathpf.millsratio_dd_asymp(x, dx, 4)
    assert out.shape == (4,)
    ref = np.array([_DD_ref(float(xi), float(di)) for xi, di in zip(x, dx)])
    assert np.allclose(out, ref, rtol=1e-12)


def test_dd_asymp_default_n_terms():
    """n_terms defaults to 4."""
    a = mathpf.millsratio_dd_asymp(60.0, 0.5)
    b = mathpf.millsratio_dd_asymp(60.0, 0.5, 4)
    assert a == b


# -------------------------------------------------- consistency with the dispatcher branch

def test_dd_dispatcher_matches_asymp_in_deep_otm():
    """In the deep-OTM regime, millsratio_dd internally calls millsratio_dd_asymp(n=4)."""
    for x, dx in [(60.0, 0.5), (80.0, 0.3), (95.0, 0.1)]:
        a = mathpf.millsratio_dd(x, dx, +1)
        b = mathpf.millsratio_dd_asymp(x, dx, 4)
        assert a == b
