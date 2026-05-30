"""High-precision conformance test for mathpf.millsratio_dd_cf.

Compares the compiled MillsRatioDiff_CF kernel against mpmath truth at the
(a, n_terms) tier pairings that match the XCF_R1 dispatcher thresholds,
cross-multiplied with a representative dx grid (spanning the sub-Sterbenz
small-dx regime to moderate dx).  Reference values are pre-computed at 50
decimal digits and embedded in tests/_ref_table.py (no mpmath at test time).

Why these tiers: the production MillsRatioDiff dispatcher routes a >= 21.2
to MillsRatioDiff_CF with n_terms in {2, 4, 6, 8} depending on a's XCF_R1
bracket.  We additionally probe the BRIDGE corners (a = 11.5 with n=12 and
a = 14.5 with n=10) which lie just inside the Taylor region in production
but exercise the CF primitive's deep-convergence behaviour.
"""
import numpy as np
import pytest

import mathpf
from _ref_table import MRDD as MRDD_REF
from _ref_table import MRDD_TIERS, MRDD_DXS


_EPS = np.finfo(float).eps


def _ulps(v, ref):
    scale = max(abs(ref), 1.0)
    return abs(v - ref) / scale / _EPS


# Observed max-drift on a clean Windows MSVC build is 0.01 ulps -- effectively
# bit-exact.  4 ulps gives ~400x headroom for libm cross-platform LSB drift.
TOL_ULP = 4


# Build flat parametrize list of (a, n_terms, dx, ref) so each cell shows up
# as its own pytest case.
_PARAMS = [
    (a, n, dx, MRDD_REF[i][dx])
    for i, (a, n) in enumerate(MRDD_TIERS)
    for dx in MRDD_DXS
]


@pytest.mark.parametrize("a,n_terms,dx,ref", _PARAMS,
                         ids=[f"a={a}_n={n}_dx={dx}" for (a, n, dx, _) in _PARAMS])
def test_millsratio_dd_cf_high_prec(a, n_terms, dx, ref):
    """mathpf.millsratio_dd_cf matches mpmath truth at the XCF_R1-tier pairings."""
    x = a + dx
    v = mathpf.millsratio_dd_cf(x, dx, n_terms)
    assert _ulps(v, ref) <= TOL_ULP, (
        f"MRDD_CF drift at a={a}, n_terms={n_terms}, dx={dx}: "
        f"kernel={v!r}, ref={ref!r}, ulps={_ulps(v, ref):.2f}"
    )
