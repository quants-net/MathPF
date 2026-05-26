"""Tests for mathpf.mills against a stdlib reference.

Reference:  R(x) = N(-x)/n(x) = sqrt(pi/2) * erfcx(x/sqrt2),  erfcx(z) = exp(z^2) erfc(z).
Only math.erfc is needed (no scipy/mpmath), kept to moderate |x| so exp(z^2) is finite.
The compiled kernels match this to a few ulp over the tested range.
"""
import math

import numpy as np
import pytest

import mathpf

SQRT2 = math.sqrt(2.0)
C = math.sqrt(math.pi / 2.0)          # R(0) = sqrt(pi/2)


def R_ref(x):
    z = x / SQRT2
    return C * math.exp(z * z) * math.erfc(z)


def R1_ref(x):
    return 1.0 - x * R_ref(x)          # -R'(x)


def R3_ref(x):
    return (x * x + 3.0) * R1_ref(x) - 1.0   # -R'''(x) = (x^2+3)(-R') - 1


def Rrel_ref(x):
    return (C - R_ref(x)) / x          # (sqrt(pi/2) - R(x)) / x  on (0,1]


XS_ALL = np.concatenate([np.linspace(-6.0, -0.01, 50),
                         np.linspace(0.0, 8.0, 200)])
XS_POS = np.linspace(1e-6, 8.0, 200)
XS_01 = np.linspace(1e-6, 1.0, 200)


def _max_rel(fn, ref, xs):
    err = 0.0
    for x in xs:
        r = ref(float(x))
        err = max(err, abs(fn(float(x)) - r) / abs(r))
    return err


def test_millsratio_scalar():
    assert _max_rel(mathpf.millsratio, R_ref, XS_ALL) < 1e-11
    assert mathpf.millsratio(0.0) == pytest.approx(C, rel=1e-14)


def test_millsratio_d1_scalar():
    assert _max_rel(mathpf.millsratio_d1, R1_ref, XS_ALL) < 1e-11   # any sign (reflection for x<0)


def test_millsratio_d3_scalar():
    assert _max_rel(mathpf.millsratio_d3, R3_ref, XS_ALL) < 1e-10   # any sign (reflection for x<0)


def test_millsratio_rel_scalar():
    assert _max_rel(mathpf.millsratio_rel_below1, Rrel_ref, XS_01) < 1e-10
    assert mathpf.millsratio_rel_below1(0.0) == pytest.approx(1.0, rel=1e-14)


def test_vectorized_and_shape():
    x = XS_POS.reshape(20, 10)
    out = mathpf.millsratio(x)
    assert out.shape == x.shape
    ref = np.array([[R_ref(float(v)) for v in row] for row in x])
    assert np.allclose(out, ref, rtol=1e-11)
    # scalar in -> python float out
    assert isinstance(mathpf.millsratio(0.5), float)


def test_recovery_identity():
    # R1 = 1 - x R   and   R = sqrt(pi/2) - x*Rrel  must hold for the kernels themselves
    x = np.linspace(0.01, 1.0, 50)
    assert np.allclose(mathpf.millsratio_d1(x), 1.0 - x * mathpf.millsratio(x), rtol=1e-12)
    assert np.allclose(mathpf.millsratio(x), C - x * mathpf.millsratio_rel_below1(x), rtol=1e-12)


def test_domain_validation():
    # millsratio / _d1 / _d3 accept any x (reflection for x < 0)
    for fn in (mathpf.millsratio, mathpf.millsratio_d1, mathpf.millsratio_d3):
        assert np.isfinite(fn(-3.0))
    # millsratio_rel_below1 is strict to [0, 1]
    with pytest.raises(ValueError):
        mathpf.millsratio_rel_below1(-0.5)
    with pytest.raises(ValueError):
        mathpf.millsratio_rel_below1(1.5)


def test_negative_x_saturation():
    """For x < X_NEG_MAX (=-37.5), the reflection branch's exp(x^2/2) would overflow.
    R / R1 / R3 saturate to +inf deterministically rather than silently propagating
    exp's overflow (which still produces +inf, but +inf - finite = +inf or NaN
    depending on sign and downstream context)."""
    # Just inside the safe range -- still finite (very large, but well below DBL_MAX)
    for fn in (mathpf.millsratio, mathpf.millsratio_d1, mathpf.millsratio_d3):
        assert np.isfinite(fn(-37.0)), f"{fn.__name__}(-37.0) should be finite (~1e297)"
    # Past the saturation threshold -- explicit +inf
    for x in (-37.5 - 1e-9, -40.0, -100.0, -1e10):
        for fn in (mathpf.millsratio, mathpf.millsratio_d1, mathpf.millsratio_d3):
            v = fn(x)
            assert np.isposinf(v), f"{fn.__name__}({x}) should be +inf, got {v}"
    # Vectorized: a single array containing safe + saturating arguments
    x = np.array([-40.0, -37.0, 0.0, 3.0, 100.0])
    out = mathpf.millsratio(x)
    assert np.isposinf(out[0])
    assert np.all(np.isfinite(out[1:]))
    # Positive x is unaffected: x = 100 should give R(100) ~ 1/100 = 0.01
    assert abs(out[-1] - 0.01) < 1e-3
