"""High-precision conformance tests for mathpf.millsratio_dd / millsratio_dd_cf.

Two test groups:

  CF primitive    millsratio_dd_cf directly, at the XCF_R1-tier (a, n_terms)
                  pairings cross dx grid.

  DD dispatcher   millsratio_dd(x, dx, +1) -- branch-agnostic on a 7x6 grid
                  in (a, m) coordinates where m = dx/(1.25+x) is the gate
                  fraction.  Two cells straddle the Taylor/Direct gate at
                  m = 0.0391 / 0.0393 (the threshold is 0.0392) to surface
                  any gate-corner regression; the test makes no assumption
                  about which internal branch handles each cell, so future
                  shifts to the gate constant don't require test changes.

Reference values are pre-computed at 50 decimal digits and embedded in
tests/_ref_table.py (no mpmath at test time).
"""
import numpy as np
import pytest

import mathpf
from _ref_table import MRDD as MRDD_REF
from _ref_table import MRDD_TIERS, MRDD_DXS
from _ref_table import MRDD_DD


_EPS = np.finfo(float).eps


def _ulps(v, ref):
    scale = max(abs(ref), 1.0)
    return abs(v - ref) / scale / _EPS


# Tolerances per primitive.  Observed max-drift on a clean Windows MSVC build:
#     CF primitive            0.01 ulps  (bit-exact)
#     DD dispatcher           7.50 ulps  (worst at a=0.5, m=0.01 -- small-a,
#                                          small-dx cancellation in the direct
#                                          subtraction)
# 4 ulps gates the CF primitive (huge headroom).  16 ulps gates the DD
# dispatcher (~2x worst observed); both absorb libm cross-platform LSB drift.
TOL_ULP_CF = 4
TOL_ULP_DD = 16


# ----------------------------------------------------------------------------
# CF primitive
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
    assert _ulps(v, ref) <= TOL_ULP_CF, (
        f"MRDD_CF drift at a={a}, n_terms={n_terms}, dx={dx}: "
        f"kernel={v!r}, ref={ref!r}, ulps={_ulps(v, ref):.2f}"
    )


# ----------------------------------------------------------------------------
# DD dispatcher (branch-agnostic)
# ----------------------------------------------------------------------------
@pytest.mark.parametrize("a,m,dx,ref", MRDD_DD,
                         ids=[f"a={a}_m={m}" for (a, m, _, _) in MRDD_DD])
def test_millsratio_dd_dispatcher_high_prec(a, m, dx, ref):
    """mathpf.millsratio_dd(x, dx, +1) matches mpmath truth across a 7x6 grid
    in (a, m) with m = dx/(1.25+x) straddling the Taylor/Direct gate (0.0392)
    via the m = 0.0391 / 0.0393 column pair.  No assumption about internal
    branch routing -- if the gate constant moves in a future kernel change,
    this test stays valid."""
    x = a + dx
    v = mathpf.millsratio_dd(x, dx, +1)
    assert _ulps(v, ref) <= TOL_ULP_DD, (
        f"MRDD_DD drift at a={a}, m={m}, dx={dx}: "
        f"kernel={v!r}, ref={ref!r}, ulps={_ulps(v, ref):.2f}"
    )
