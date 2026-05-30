"""High-precision conformance tests for mathpf.millsratio_dd / millsratio_dd_cf.

Three test groups, one per branch of the MillsRatioDiff dispatcher (theta=+1):

  CF branch       (millsratio_dd_cf directly, a >= XCF_R1[0] and small dx):
                  XCF_R1-tier (a, n_terms) pairings cross dx grid.
  Taylor branch   (millsratio_dd, dx < 0.0392*(1.25+x) and a < 21.2):
                  per-row (a, dx) cells with dx scaled to the Taylor regime.
  Direct branch   (millsratio_dd, dx >= 0.0392*(1.25+x)):
                  (a) tiers cross uniform dx grid.

Reference values are pre-computed at 50 decimal digits and embedded in
tests/_ref_table.py (no mpmath at test time).
"""
import numpy as np
import pytest

import mathpf
from _ref_table import MRDD as MRDD_REF
from _ref_table import MRDD_TIERS, MRDD_DXS
from _ref_table import MRDD_TAYLOR
from _ref_table import MRDD_DIRECT, MRDD_DIRECT_TIERS, MRDD_DIRECT_DXS


_EPS = np.finfo(float).eps


def _ulps(v, ref):
    scale = max(abs(ref), 1.0)
    return abs(v - ref) / scale / _EPS


# Uniform 4-ulp tolerance across all three branches.  Observed max-drift on a
# clean Windows MSVC build:
#     CF branch       0.01 ulps  (bit-exact)
#     Taylor branch   3.41 ulps  (worst at a=5, dx=0.01 -- still under 4)
#     Direct branch   0.25 ulps  (bit-exact)
# 4 ulps absorbs libm cross-platform LSB drift on macOS-arm64 Clang and Linux gcc.
TOL_ULP = 4


# ----------------------------------------------------------------------------
# CF branch
# ----------------------------------------------------------------------------
_PARAMS_CF = [
    (a, n, dx, MRDD_REF[i][dx])
    for i, (a, n) in enumerate(MRDD_TIERS)
    for dx in MRDD_DXS
]


@pytest.mark.parametrize("a,n_terms,dx,ref", _PARAMS_CF,
                         ids=[f"a={a}_n={n}_dx={dx}" for (a, n, dx, _) in _PARAMS_CF])
def test_millsratio_dd_cf_high_prec(a, n_terms, dx, ref):
    """mathpf.millsratio_dd_cf matches mpmath truth at the XCF_R1-tier pairings."""
    x = a + dx
    v = mathpf.millsratio_dd_cf(x, dx, n_terms)
    assert _ulps(v, ref) <= TOL_ULP, (
        f"MRDD_CF drift at a={a}, n_terms={n_terms}, dx={dx}: "
        f"kernel={v!r}, ref={ref!r}, ulps={_ulps(v, ref):.2f}"
    )


# ----------------------------------------------------------------------------
# Taylor branch (dispatcher routes here when a < 21.2 and dx < 0.0392*(1.25+x))
# ----------------------------------------------------------------------------
@pytest.mark.parametrize("a,dx,ref", MRDD_TAYLOR,
                         ids=[f"a={a}_dx={dx}" for (a, dx, _) in MRDD_TAYLOR])
def test_millsratio_dd_taylor_high_prec(a, dx, ref):
    """mathpf.millsratio_dd routes (a < 21.2, dx in Taylor regime) to the 5-term
    R'''-seeded Taylor; verify it matches mpmath truth."""
    x = a + dx
    # Sanity: confirm the dispatcher actually routes here.
    assert a < 21.2 and dx < 3.92e-2 * (1.25 + x), \
        f"cell (a={a}, dx={dx}) does not route to Taylor"
    v = mathpf.millsratio_dd(x, dx, +1)
    assert _ulps(v, ref) <= TOL_ULP, (
        f"MRDD Taylor drift at a={a}, dx={dx}: "
        f"kernel={v!r}, ref={ref!r}, ulps={_ulps(v, ref):.2f}"
    )


# ----------------------------------------------------------------------------
# Direct branch (dispatcher routes here when dx >= 0.0392*(1.25+x))
# ----------------------------------------------------------------------------
_PARAMS_DIRECT = [
    (a, dx, MRDD_DIRECT[i][dx])
    for i, a in enumerate(MRDD_DIRECT_TIERS)
    for dx in MRDD_DIRECT_DXS
]


@pytest.mark.parametrize("a,dx,ref", _PARAMS_DIRECT,
                         ids=[f"a={a}_dx={dx}" for (a, dx, _) in _PARAMS_DIRECT])
def test_millsratio_dd_direct_high_prec(a, dx, ref):
    """mathpf.millsratio_dd routes (large dx) to the direct (R(x-dx) - R(x+dx))/(2dx)
    formula; cancellation-bounded since the difference is order-of-magnitude similar."""
    x = a + dx
    assert dx >= 3.92e-2 * (1.25 + x), \
        f"cell (a={a}, dx={dx}) does not route to Direct"
    v = mathpf.millsratio_dd(x, dx, +1)
    assert _ulps(v, ref) <= TOL_ULP, (
        f"MRDD Direct drift at a={a}, dx={dx}: "
        f"kernel={v!r}, ref={ref!r}, ulps={_ulps(v, ref):.2f}"
    )
