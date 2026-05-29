"""Bit-equality consistency between mathpf._pyref (Python reference) and the
compiled top-level mathpf surface (Cython binding to the production kernel).

The _pyref modules are the canonical algorithmic specification of what the
compiled binding computes; this test enforces that they stay byte-identical
across every dispatch regime.  Drift in either side fails CI immediately.
"""
import numpy as np
import pytest

import mathpf
from mathpf._pyref import mills_cheby as ref_mc
from mathpf._pyref import mills_dd as ref_dd


# ----------------------------------------------------------------- Mills ratio R and derivatives

# x grid spanning every dispatch path:
#   x <= 1            : Rrel_below1 polynomial
#   1 < x < XCF_R[0]  : segmented Chebyshev for R1 (R derived as (1-R1)/x)
#   x >= XCF_R[0]=9.5 : tiered CF for R (n=12 -> 10 -> 8 -> 6 -> 4 -> 2 -> 1/x)
#   x = negative      : reflection R(x) = sqrt(2pi) exp(x^2/2) - R(-x)
X_GRID_R = [
    -3.0, -1.0, -0.1,
    0.0, 0.1, 0.5, 1.0,
    1.001, 2.0, 5.0, 9.4,
    9.5, 9.6, 11.5, 15.2, 24.5, 60.0, 600.0, 7.0e7, 1.0e10,
]


@pytest.mark.parametrize("x", X_GRID_R)
def test_R_pyref_matches_compiled(x):
    assert ref_mc.R(x) == mathpf.millsratio(x), f"R drift at x={x}"


@pytest.mark.parametrize("x", X_GRID_R)
def test_R1_pyref_matches_compiled(x):
    assert ref_mc.R1(x) == mathpf.millsratio_d1(x), f"R1 drift at x={x}"


# R3 grid: x <= 0 also covered; CF tiers at 17.1, 25.4, 50.5, 210, 17300
X_GRID_R3 = [
    -2.0, -0.5, 0.0, 0.5, 2.0, 5.0, 10.0, 17.0,
    17.1, 17.2, 25.5, 50.6, 210.5, 17400.0, 1.0e8,
]


@pytest.mark.parametrize("x", X_GRID_R3)
def test_R3_pyref_matches_compiled(x):
    assert ref_mc.R3(x) == mathpf.millsratio_d3(x), f"R3 drift at x={x}"


# Rrel_below1: only defined on [0, 1]
@pytest.mark.parametrize("x", [0.0, 0.05, 0.25, 0.5, 0.75, 1.0])
def test_Rrel_below1_pyref_matches_compiled(x):
    assert ref_mc.Rrel_below1(x) == mathpf.millsratio_rel_below1(x), f"Rrel_below1 drift at x={x}"


# -------------------------------------------------- Symmetric divided differences (R_DD)

# (x, dx) grid spanning all three regimes of R_DD's dispatch:
#   - direct R difference  (dx >= 0.0392 (1.25 + x))
#   - Taylor (a = x - dx < 21.2, dx < gate)
#   - CF asymp ladder (a >= 21.2, dx < gate); cutoffs at 21.2, 41, 165, 12800
DD_GRID = []
for x in [0.5, 2.0, 5.0, 10.0, 15.0, 17.0, 17.1, 17.2, 18.0, 21.0,
          21.5, 25.0, 50.0, 200.0, 1000.0, 15000.0]:
    gate = 3.92e-2 * (1.25 + x)
    for dx in [0.01, gate*0.5, gate*0.95, gate*1.5, gate*5.0]:
        if dx < x:
            DD_GRID.append((x, dx))


@pytest.mark.parametrize("x,dx", DD_GRID)
def test_R_DD_call_branch_pyref_matches_compiled(x, dx):
    """theta = +1 (difference branch) across all three regimes."""
    assert ref_dd.R_DD(x, dx, +1) == mathpf.millsratio_dd(x, dx, +1), f"R_DD(+1) drift at x={x}, dx={dx}"


@pytest.mark.parametrize("x,dx", [(0.5, 0.3), (1.0, 0.8), (3.0, 0.2), (-1.0, 0.5)])
def test_R_DD_above_branch_pyref_matches_compiled(x, dx):
    """theta = -1 (sum branch)."""
    assert ref_dd.R_DD(x, dx, -1) == mathpf.millsratio_dd(x, dx, -1), f"R_DD(-1) drift at x={x}, dx={dx}"


@pytest.mark.parametrize("x,dx,n", [
    (15000.0, 0.5, 2),    # XCF_R1[5] tier
    (1000.0, 0.5, 4),     # XCF_R1[4] tier
    (50.0, 0.5, 6),       # XCF_R1[3] tier
    (22.5, 0.5, 8),       # XCF_R1[2] tier (bottom row)
    (100.0, 0.1, 4),
    (60.0, 1.0, 4),
])
def test_R_DD_CF_pyref_matches_compiled(x, dx, n):
    """Cancellation-free CF DD at all four dispatch orders."""
    assert ref_dd.R_DD_CF(x, dx, n) == mathpf.millsratio_dd_cf(x, dx, n), \
        f"R_DD_CF drift at x={x}, dx={dx}, n={n}"
