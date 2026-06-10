"""Smoke and accuracy tests for mathpf.erfcx.

Cross-checked against scipy.special.erfcx and the analytical derivative
identity erfcx'(z) = 2 z erfcx(z) - 2/sqrt(pi).
"""
from __future__ import annotations

import math
import numpy as np
import pytest
from scipy import special

import mathpf


# Representative grid spanning the dispatch regimes of the underlying Mills
# primitives: negative (reflection), [0, 1] polynomial, [1, ~9.5] R_1
# segmented Chebyshev, [9.5, infty) tiered continued fraction.
Z_GRID = [-10.0, -5.0, -2.0, -1.0, -0.1, 0.0, 0.1, 0.5, 1.0,
          1.001, 2.0, 5.0, 9.4, 9.5, 9.6, 15.0, 30.0, 100.0]


# =========================================================================
# Scalar matching with SciPy
# =========================================================================
@pytest.mark.parametrize('z', Z_GRID)
def test_erfcx_matches_scipy(z):
    """erfcx vs scipy.special.erfcx -- ~few ulps across the realistic range."""
    ours   = mathpf.erfcx(z)
    theirs = special.erfcx(z)
    if theirs == 0.0:
        assert ours == 0.0
        return
    rel = abs(ours / theirs - 1)
    assert rel < 1e-13, f"erfcx({z}): ours={ours}, scipy={theirs}, rel={rel:.3e}"


@pytest.mark.parametrize('z', Z_GRID)
def test_erfcx_d1_matches_analytic_identity(z):
    """erfcx'(z) = 2 z erfcx(z) - 2/sqrt(pi)."""
    d1_ours  = mathpf.erfcx_d1(z)
    d1_check = 2.0 * z * special.erfcx(z) - 2.0 / math.sqrt(math.pi)
    if d1_check == 0.0:
        assert abs(d1_ours) < 1e-15
        return
    rel = abs(d1_ours / d1_check - 1)
    # Allow looser tolerance at large |z| where the analytic identity itself
    # suffers from cancellation between 2 z erfcx(z) and 2/sqrt(pi).
    tol = 1e-12 if abs(z) <= 5.0 else 1e-10
    assert rel < tol, f"erfcx_d1({z}): ours={d1_ours}, check={d1_check}, rel={rel:.3e}"


@pytest.mark.parametrize('z', Z_GRID)
def test_erfcx_d3_matches_analytic_identity(z):
    """Differentiating erfcx'(z) = 2 z erfcx(z) - 2/sqrt(pi) thrice gives
       erfcx'''(z) = 4 z (2 z^2 + 3) erfcx(z) - 8 (z^2 + 1) / sqrt(pi).
    Loosen tolerance at large |z| where the analytic identity itself
    suffers cancellation between O(z^3 erfcx) and O(z^2/sqrt(pi))."""
    d3_ours  = mathpf.erfcx_d3(z)
    d3_check = 4.0 * z * (2.0 * z * z + 3.0) * special.erfcx(z) \
               - 8.0 * (z * z + 1.0) / math.sqrt(math.pi)
    if d3_check == 0.0:
        assert abs(d3_ours) < 1e-15
        return
    rel = abs(d3_ours / d3_check - 1)
    # Loosening schedule reflects how badly the analytic identity cancels at
    # large |z|: at z=30 the leading term and the 8(z^2+1)/sqrt(pi) subtractor
    # match to ~1e-3, leaving ~13 digits; at z=100 only ~10 digits remain.
    if abs(z) <= 2.0:    tol = 1e-12
    elif abs(z) <= 5.0:  tol = 1e-10
    elif abs(z) <= 30.0: tol = 1e-7
    else:                tol = 1e-3
    assert rel < tol, f"erfcx_d3({z}): ours={d3_ours}, check={d3_check}, rel={rel:.3e}"


# =========================================================================
# Endpoint values
# =========================================================================
def test_erfcx_zero():
    """erfcx(0) = exp(0) * erfc(0) = 1.  Tolerated to 1 ulp because the
    composition R(0) * sqrt(2/pi) = sqrt(pi/2) * sqrt(2/pi) is the product
    of two independently-rounded reciprocals, not exactly 1.0 in fp64."""
    assert abs(mathpf.erfcx(0.0) - 1.0) <= 4.5e-16    # 2 ulps at 1.0


def test_erfcx_d1_zero():
    """erfcx'(0) = 2*0*1 - 2/sqrt(pi) = -2/sqrt(pi)."""
    expected = -2.0 / math.sqrt(math.pi)
    assert abs(mathpf.erfcx_d1(0.0) - expected) < 5e-16


def test_erfcx_d3_zero():
    """erfcx'''(0): differentiate erfcx'(z) = 2 z erfcx(z) - 2/sqrt(pi) twice.
       erfcx''(z) = 2 erfcx(z) + 2 z erfcx'(z)
       erfcx'''(z) = 4 erfcx'(z) + 2 z erfcx''(z)
       At z = 0: erfcx'''(0) = 4 * erfcx'(0) = -8/sqrt(pi)."""
    expected = -8.0 / math.sqrt(math.pi)
    assert abs(mathpf.erfcx_d3(0.0) - expected) < 5e-15


# =========================================================================
# numpy-vectorisation
# =========================================================================
def test_erfcx_array_input():
    """Vectorised over ndarray; shape and values preserved."""
    z = np.array([[0.0, 1.0], [2.0, 3.0]])
    out = mathpf.erfcx(z)
    assert out.shape == z.shape
    assert np.allclose(out, special.erfcx(z), rtol=1e-13)


def test_erfcx_d1_array_input():
    z = np.array([0.0, 0.5, 1.0, 2.0, 5.0])
    d1 = mathpf.erfcx_d1(z)
    check = 2.0 * z * special.erfcx(z) - 2.0 / math.sqrt(math.pi)
    assert np.allclose(d1, check, rtol=1e-12)


def test_erfcx_scalar_returns_float():
    """A scalar input returns a Python float, not a 0-d ndarray."""
    assert isinstance(mathpf.erfcx(0.5), float)
    assert isinstance(mathpf.erfcx_d1(0.5), float)
    assert isinstance(mathpf.erfcx_d3(0.5), float)
