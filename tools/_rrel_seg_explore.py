"""Explore segmented Chebyshev fits for R_rel(x) = (sqrt(pi/2) - R(x))/x on
[0, 1], targeting deg-7 (8 coefficients per segment) to match R1's per-segment
cost.

Goal: find the smallest number of uniform segments on [0, 1] that brings the
per-segment relative error below ~2*EPS (the same gate used for R1 / R3 fits
in tools/cheby_fit.py).  If 2-4 segments suffice, this is a worthwhile swap
from the current 14-coef single-segment form -- it halves the per-call cost
for R_rel and (via R = sqrt(pi/2) - x * R_rel) for R near small x.
"""
from __future__ import annotations

import mpmath as mp
import numpy as np

EPS = np.finfo(float).eps
DPS = 45


def R_ref(x):
    """High-precision Mills ratio."""
    x = mp.mpf(x)
    return mp.ncdf(-x) / mp.npdf(x)


def R_rel_ref(x):
    """R_rel(x) = (sqrt(pi/2) - R(x))/x on (0, 1]; R_rel(0) := 1."""
    xm = mp.mpf(x)
    if xm == 0:
        return mp.mpf(1)
    return (mp.sqrt(mp.pi / 2) - R_ref(xm)) / xm


def seg_fit(a, b, ncoef):
    """Fit R_rel on [a, b] with monomial-form polynomial of degree ncoef-1.
    Returns (coefs_hi_first, rel_err) where rel_err = sup-norm error / min |f|."""
    def f(s):
        return R_rel_ref(mp.mpf(a) + s * (mp.mpf(b) - mp.mpf(a)))
    with mp.workdps(DPS):
        coef, abserr = mp.chebyfit(f, [0, 1], ncoef, error=True)
        scale = min(abs(f(mp.mpf(0))), abs(f(mp.mpf(1))))
        rel = float(abserr / scale)
    return [float(c) for c in coef], rel


def worst_rel_err(nseg, ncoef):
    """Worst per-segment relative error over nseg uniform segments on [0, 1]."""
    boundaries = [i / nseg for i in range(nseg + 1)]
    worst = 0.0
    worst_i = -1
    for i in range(nseg):
        a, b = boundaries[i], boundaries[i + 1]
        _, rel = seg_fit(a, b, ncoef)
        if rel > worst:
            worst, worst_i = rel, i
    return worst, worst_i


def emit(nseg, ncoef):
    """Emit the segmented coefficient block for _mills_coef.h."""
    boundaries = [i / nseg for i in range(nseg + 1)]
    all_coefs = []
    worst = 0.0
    for i in range(nseg):
        a, b = boundaries[i], boundaries[i + 1]
        coefs, rel = seg_fit(a, b, ncoef)
        # Reverse to lowest-first so Horner reads C[o+ncoef-1] as the high coef
        coefs = list(reversed(coefs))
        all_coefs.extend(coefs)
        worst = max(worst, rel)
    print(f"/* Rrel_below1: deg-{ncoef-1}, {nseg} segments uniform on [0, 1]; "
          f"max per-seg rel-err = {worst:.2e} */")
    print(f"inline constexpr int    NSEG_RREL = {nseg};")
    print(f"inline constexpr double C_Rrel[{nseg * ncoef}] = {{", end="")
    for k, v in enumerate(all_coefs):
        if k % 4 == 0:
            print("\n   ", end=" ")
        print(f"{v:>23.17g},", end=" ")
    print("\n};")


def main():
    print("Segmented R_rel fit on [0, 1], degree 7 (ncoef = 8):")
    print(f"  target rel-err <= 2*EPS = {2*EPS:.2e}")
    print()
    print(f"{'nseg':>4} {'worst rel':>14} {'worst-seg-idx':>14}  cost: 8 mults Horner")
    print("-" * 60)
    for nseg in [1, 2, 3, 4, 6, 8, 10, 12, 16, 20]:
        worst, idx = worst_rel_err(nseg, ncoef=8)
        flag = "  <- PASSES" if worst <= 2 * EPS else ""
        print(f"{nseg:>4} {worst:14.3e} {idx:>14d}{flag}")
    print()
    print("/* ===== emit nseg=12 ===== */")
    emit(nseg=12, ncoef=8)


if __name__ == "__main__":
    main()
