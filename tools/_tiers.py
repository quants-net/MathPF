"""Verify tiered-CF accuracy of mathpf._pyref.mills_cheby (R / R1 / R3) vs mpmath.
Sweeps each function across the bands [XCF, t1), [t1, t2), ... and reports the
worst relative error (in eps) per band, with extra points right at each threshold.

Run from the MathPF repo root:
    python tools/_tiers.py
"""
import numpy as np, mpmath as mp
from mathpf._pyref import mills_cheby as mc

mp.mp.dps = 120
EPS = 2.0 ** -52


def R_ref(x):
    x = mp.mpf(x)
    return mp.sqrt(mp.pi / 2) * mp.erfc(x / mp.sqrt(2)) * mp.e ** (x * x / 2)


def Rd1_ref(x):
    x = mp.mpf(x)
    return 1 - x * R_ref(x)            # -R'


def Rd3_ref(x):
    x = mp.mpf(x)
    return (x * x + 3) * (1 - x * R_ref(x)) - 1   # -R'''


CASES = [
    ("R   ", mc.R,    R_ref,   24.1, [59.3, 548.0]),
    ("R1", mc.R1, Rd1_ref, 21.2, [41.0, 165.0, 12800.0]),
    ("R3", mc.R3, Rd3_ref, 25.4, [50.5, 210.0, 17300.0]),
]

for name, fn, ref, xcf, thr in CASES:
    edges = [xcf] + thr + [1.0e6]
    print(f"\n== {name}  (x_cf={xcf}, tier thresholds {thr}) ==")
    for lo, hi in zip(edges[:-1], edges[1:]):
        xs = np.concatenate([
            np.geomspace(lo, hi, 40),
            [lo, lo * 1.0000001, hi * 0.9999999],   # band endpoints
        ])
        worst = 0.0
        xw = lo
        for x in xs:
            got = mp.mpf(fn(float(x)))
            r = ref(x)
            e = abs((got - r) / r) / EPS
            if e > worst:
                worst, xw = e, x
        print(f"  band [{lo:>9.4g}, {hi:>9.4g}):  worst = {float(worst):6.2f} eps  at x={float(xw):.4g}")
