"""Pick total NSEG for R_d1 (deg6) and R_d3 (deg7) so the buckets covering [0,XCF]
plus a contiguous headroom band beyond XCF all stay <= TGT eps (bucket-local).
Report used-region worst, headroom bucket count, ntab, and XMAX reached."""
import numpy as np, mpmath as mp
mp.mp.dps = 50
EPS = 2.0 ** -52
NN = 3.5
TGT = 2.5


def Rmp(x):
    x = mp.mpf(x)
    return mp.sqrt(mp.pi / 2) * mp.erfc(x / mp.sqrt(2)) * mp.e ** (x * x / 2)


def Rd1(x):
    x = mp.mpf(x); return 1 - x * Rmp(x)


def Rd3(x):
    x = mp.mpf(x); return (x * x + 3) * (1 - x * Rmp(x)) - 1


def berr(fn, nseg, ncoef, i):
    def f(s, i=i):
        w = (i + s) / nseg
        return fn(NN * w / (1 - w))
    coef = [float(c) for c in mp.chebyfit(f, [0, 1], ncoef, error=False)][::-1]
    bw = 0.0
    for s in np.linspace(0, 1, 21):
        p = coef[-1]
        for k in range(ncoef - 2, -1, -1):
            p = p * s + coef[k]
        bw = max(bw, abs((p - float(f(mp.mpf(s)))) / float(f(mp.mpf(s)))))
    return bw / EPS


def evaluate(name, fn, xcf, ncoef, nsegs):
    print(f"== {name}  (XCF={xcf}, deg={ncoef-1}, target {TGT} eps) ==")
    for nseg in nsegs:
        i_star = int(xcf / (NN + xcf) * nseg)
        used = max(berr(fn, nseg, ncoef, i) for i in range(i_star + 1))
        head, i = 0, i_star + 1
        while i < nseg - 1 and berr(fn, nseg, ncoef, i) <= TGT:
            head += 1; i += 1
        ntab = i_star + 1 + head
        xmax = NN * ntab / (nseg - ntab)
        print(f"  NSEG={nseg:3d}  used worst={used:4.2f}eps  headroom={head} bucket(s)  "
              f"ntab={ntab}  XMAX~{xmax:.0f} ({xmax/xcf:.2f}x XCF)")


evaluate("R_d1", Rd1, 21.2, 7, (90, 100, 110, 120, 130))
evaluate("R_d3", Rd3, 25.4, 8, (110, 120, 130, 140, 150))
