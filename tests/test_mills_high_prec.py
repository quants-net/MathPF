"""High-precision conformance tests for the Mills primitives.

Compares the compiled kernels against mpmath-generated reference values
stored at half-integer x in [0, 20] (see tests/_ref_table.py and the
generator tools/gen_mills_ref.py).  Tolerance is in ulps relative to the
reference value; passing here is a stronger guarantee than the previous
stdlib-erfc-based tests since the reference itself is bit-exact at double
precision (rounded from 50 decimal digits).

mpmath is NOT a test-time dependency -- the reference table is embedded as
plain float constants in tests/_ref_table.py.
"""
import numpy as np
import pytest

import mathpf
from _ref_table import R as R_REF
from _ref_table import R1 as R1_REF
from _ref_table import R3 as R3_REF
from _ref_table import R_rel as R_REL_REF


_EPS = np.finfo(float).eps


def _rel_err(v, ref):
    """Return |v - ref| / |ref| , protecting against ref == 0."""
    scale = max(abs(ref), 1.0)
    return abs(v - ref) / scale


def _ulps(v, ref):
    """Approximate number of ulps between v and ref."""
    return _rel_err(v, ref) / _EPS


# Ulp tolerances against the mpmath reference.  Observed max-drift on a
# clean Windows MSVC build (after R_rel switched to segmented deg-7):
#     R:     0.80 ulps  at x = 0.0
#     R1:    0.50 ulps  at x = 1.0
#     R3:    0.50 ulps  at x = 0.5
#     R_rel: 0.50 ulps  at x = 0.0
# All four primitives are sub-ulp.  Uniform 4-ulp ceiling gives ~5x headroom
# for libm cross-platform LSB drift on macOS-arm64 Clang and Linux gcc.
TOL_ULP = 4


@pytest.mark.parametrize("x", sorted(R_REF.keys()))
def test_millsratio_high_prec(x):
    """mathpf.millsratio matches mpmath truth at half-integer x in [0, 20]."""
    ref = R_REF[x]
    v   = mathpf.millsratio(float(x))
    assert _ulps(v, ref) <= TOL_ULP, (
        f"R drift at x={x}: kernel={v!r}, ref={ref!r}, ulps={_ulps(v, ref):.2f}"
    )


@pytest.mark.parametrize("x", sorted(R1_REF.keys()))
def test_millsratio_d1_high_prec(x):
    """mathpf.millsratio_d1 (= -R'(x) = 1 - x R(x)) matches mpmath truth."""
    ref = R1_REF[x]
    v   = mathpf.millsratio_d1(float(x))
    assert _ulps(v, ref) <= TOL_ULP, (
        f"R1 drift at x={x}: kernel={v!r}, ref={ref!r}, ulps={_ulps(v, ref):.2f}"
    )


@pytest.mark.parametrize("x", sorted(R3_REF.keys()))
def test_millsratio_d3_high_prec(x):
    """mathpf.millsratio_d3 (= -R'''(x)) matches mpmath truth."""
    ref = R3_REF[x]
    v   = mathpf.millsratio_d3(float(x))
    assert _ulps(v, ref) <= TOL_ULP, (
        f"R3 drift at x={x}: kernel={v!r}, ref={ref!r}, ulps={_ulps(v, ref):.2f}"
    )


@pytest.mark.parametrize("x", sorted(R_REL_REF.keys()))
def test_millsratio_rel_below1_high_prec(x):
    """mathpf.millsratio_rel_below1 matches mpmath truth on [0, 1]."""
    ref = R_REL_REF[x]
    v   = mathpf.millsratio_rel_below1(float(x))
    assert _ulps(v, ref) <= TOL_ULP, (
        f"R_rel drift at x={x}: kernel={v!r}, ref={ref!r}, ulps={_ulps(v, ref):.2f}"
    )
