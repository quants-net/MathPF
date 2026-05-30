"""Probe MillsRatioDiff_CF ulp drift vs mpmath truth across the (a, n) tiers
matching XCF_R1, paired with a dx grid.  Goal: pick a sensible tolerance for
the future test_mills_dd_high_prec.py.
"""
import mpmath as mp
import numpy as np
import mathpf

mp.mp.dps = 50
EPS = np.finfo(float).eps

# (a, n_terms) pairs.  a = x - dx is the smaller Mills argument; n_terms is the
# CF order paired with that tier per XCF_R1.
PAIRS = [
    (11.5,    12),
    (14.5,    10),
    (21.2,     8),
    (41.0,     6),
    (165.0,    4),
    (12800.0,  2),
]
DXS = [0.0001, 0.01, 1.0, 2.0, 4.0, 8.0]


def R_mp(x):
    x = mp.mpf(x)
    return mp.ncdf(-x) / mp.npdf(x)


def truth_dd(x, dx):
    """Exact divided difference (R(x-dx) - R(x+dx)) / (2 dx) at 50 dps."""
    xm, dxm = mp.mpf(x), mp.mpf(dx)
    return (R_mp(xm - dxm) - R_mp(xm + dxm)) / (mp.mpf(2) * dxm)


def main():
    print(f"{'a':>8} {'n':>3} | {'dx':>8} {'mathpf':>17} {'truth':>17} {'ulps':>8}")
    print("-" * 75)
    for a, n in PAIRS:
        for dx in DXS:
            x = a + dx
            v = mathpf.millsratio_dd_cf(x, dx, n)
            ref = float(truth_dd(x, dx))
            scale = max(abs(ref), 1.0)
            ulps = abs(v - ref) / scale / EPS
            flag = "  HIGH" if ulps > 16 else ""
            print(f"{a:8.1f} {n:>3} | {dx:8.4f} {v:17.9e} {ref:17.9e} {ulps:8.2f}{flag}")
        print()


if __name__ == "__main__":
    main()
