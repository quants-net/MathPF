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
from _ref_table import MRDD_DD, MRDD_DD_MS


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


# ----------------------------------------------------------------------------
# DD dispatcher vs CF cross-validation (no stored truth)
# ----------------------------------------------------------------------------
def _n_terms_for_a(a):
    """Return the CF order n_terms appropriate for `a`'s XCF_R1 tier.  Caller
    must ensure a >= 11.5 (where the n=12 bridge first reaches ~ulp accuracy)."""
    if a >= 12800.0: return 2
    if a >= 165.0:   return 4
    if a >= 41.0:    return 6
    if a >= 21.2:    return 8
    if a >= 14.5:    return 10
    if a >= 11.5:    return 12
    raise ValueError(f"a={a} below CF accuracy range (need a >= 11.5)")


# Reuse the same m grid; pair with a values where the CF primitive is
# accurate (a >= 11.5).  The dispatcher may internally route any of these
# cells to CF / Taylor / Direct depending on (a, m); the test compares its
# output to millsratio_dd_cf with the tier-appropriate n_terms -- since the
# previous MRDD_CF test confirmed millsratio_dd_cf matches mpmath to
# sub-ulp at these tiers, this cross-check transitively validates the
# dispatcher at a >= 11.5 without storing more reference values.
MRDD_DD_VS_CF_AS = (11.5, 14.5, 15.0, 21.2, 30.0, 100.0)


_PARAMS_DD_VS_CF = [
    (a, m)
    for a in MRDD_DD_VS_CF_AS
    for m in MRDD_DD_MS
]


@pytest.mark.parametrize("a,m", _PARAMS_DD_VS_CF,
                         ids=[f"a={a}_m={m}" for (a, m) in _PARAMS_DD_VS_CF])
def test_millsratio_dd_dispatcher_vs_cf(a, m):
    """Cross-validate millsratio_dd against millsratio_dd_cf at a >= 11.5
    where CF is sub-ulp accurate per the previous MRDD_CF test.  No stored
    truth -- the CF call itself is the reference."""
    dx = m * (1.25 + a) / (1.0 - m)
    x = a + dx
    n = _n_terms_for_a(a)
    ref = mathpf.millsratio_dd_cf(x, dx, n)
    v   = mathpf.millsratio_dd(x, dx, +1)
    assert _ulps(v, ref) <= TOL_ULP_DD, (
        f"MRDD vs CF drift at a={a}, m={m}, dx={dx}, n_terms={n}: "
        f"dispatcher={v!r}, cf_ref={ref!r}, ulps={_ulps(v, ref):.2f}"
    )
