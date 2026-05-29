"""Generation & verification harness for the fast double-precision Mills ratio
primitives that mathpf ships:
    R(x)    = N(-x)/n(x)
    R_d1(x) = 1 - x R(x) = -R'(x)          (cancellation-free log-derivative)
for x >= 0.  Negative arguments use the reflection
    R(x) = sqrt(2*pi) e^{x^2/2} - R(-x).

Approximation regions (variable t = 1/x):
  region 1  small x  (0 .. x_lo)   :  Chebyshev in x      (segments uniform in x)
  region 2  mid   x  (x_lo .. x_cf):  Chebyshev in t=1/x  (segments uniform in t)
  region 3  large x  (x >= x_cf)   :  Laplace continued fraction (separate)

Conventions (kept consistent with the C++/Cython surface):
  * each segment is fit on a LOCAL coordinate s in [0,1];
  * the polynomial is stored in monomial form and evaluated by HORNER;
  * the error measure is mp.chebyfit(..., error=True) -- its built-in max ABSOLUTE
    error (a high-precision sample at the N Chebyshev extrema, reliable as a sup-norm
    estimate) -- normalized to RELATIVE by dividing by the smallest |f| on the segment.

Goal: match the earlier "200 segments @ degree 6" accuracy with fewer segments by
trading polynomial degree against segment count.

Run from the MathPF repo root:
    python tools/cheby_fit.py        # full refit + verify + emit
"""
import numpy as np
import mpmath as mp


EPS = np.finfo(float).eps          # 2.220446049250313e-16
DPS = 45                           # mpmath digits for fit + error estimate


# ----------------------------------------------------------------------------
# MPMR -- arbitrary-precision (mpmath) Mills-ratio reference used to fit and
# verify the segmented Chebyshev / CF approximations.  Pure math; no application
# vocabulary in this module.
# ----------------------------------------------------------------------------
class MPMR:
    """High-precision Mills ratio R(x) = N(-x)/n(x) and odd-derivative ladder."""

    @staticmethod
    def R(x_):
        x = mp.mpf(x_)
        return mp.ncdf(-x) / mp.npdf(x)

    @staticmethod
    def R_d1(x_):
        """-R'(x) = 1 - x R(x)."""
        x = mp.mpf(x_)
        return mp.mpf('1') - x * MPMR.R(x)

    @staticmethod
    def R_dn(x_, n):
        """-R^(n)(x) for odd n (>0 for x>=0), via the ladder R^(k+1) = x R^(k) + k R^(k-1)."""
        x = mp.mpf(x_)
        r_prev, r = MPMR.R(x), x * MPMR.R(x) - 1   # R, R'
        for k in range(1, n):
            r_prev, r = r, x * r + k * r_prev
        return -r

    @staticmethod
    def R_d3(x_):
        """-R'''(x) (>0 for x>=0)."""
        return MPMR.R_dn(x_, 3)

    @staticmethod
    def R_t(t, NN, nn, kk):
        x_ = mp.mpf(f'{NN*nn}') / (mp.mpf(t) + mp.mpf(f'{kk}')) - mp.mpf(f'{nn}')
        return MPMR.R(x_)

    @staticmethod
    def R_d1_t(t, NN, nn, kk):
        x_ = mp.mpf(f'{NN*nn}') / (mp.mpf(t) + mp.mpf(f'{kk}')) - mp.mpf(f'{nn}')
        return mp.mpf('1') - x_ * MPMR.R(x_)


# --------------------------------------------------------------------------
# per-segment fit (local s in [0,1])
# --------------------------------------------------------------------------
def _seg_fn(u_a, u_b, kind, der):
    """High-precision f(s), s in [0,1], for a segment [u_a,u_b] of the working
    variable u (= x if kind=='x', = t=1/x if kind=='inv'). der -> R_d1 else R."""
    ua, ub = mp.mpf(u_a), mp.mpf(u_b)

    def f(s):
        u = ua + s * (ub - ua)
        if kind == "inv":
            x = 1 / u
        elif kind == "inv1":          # t = 1/(1+x)  ->  x = 1/t - 1
            x = 1 / u - 1
        else:
            x = u
        return MPMR.R_d1(x) if der else MPMR.R(x)
    return f


def seg_fit(u_a, u_b, kind, der, ncoef):
    """Return (double monomial coeffs hi-first, relative error) for one segment,
    using chebyfit's built-in error normalized by the segment's smallest |f|."""
    f = _seg_fn(u_a, u_b, kind, der)
    with mp.workdps(DPS):
        coef, abserr = mp.chebyfit(f, [0, 1], ncoef, error=True)
        scale = min(abs(f(mp.mpf(0))), abs(f(mp.mpf(1))))   # relative normalization
        rel = float(abserr / scale)
    return [float(c) for c in coef], rel


def _region_worst(kind, u_lo, u_hi, der, ncoef, nseg):
    """Worst per-segment relative error over nseg uniform segments in u."""
    edges = np.linspace(u_lo, u_hi, nseg + 1)
    worst = 0.0
    for j in range(nseg):
        _, rel = seg_fit(edges[j], edges[j + 1], kind, der, ncoef)
        worst = max(worst, rel)
    return worst


def min_segments(kind, x_lo, x_hi, der, ncoef, tol_eps=2.0, seg_cap=1200):
    """Smallest uniform-segment count reaching tol_eps*EPS relative error.
    kind=='x'   -> segments uniform in x over [x_lo, x_hi]
    kind=='inv' -> segments uniform in t=1/x over [1/x_hi, 1/x_lo]
    (relative error decreases ~monotonically with nseg, so bisect.)"""
    if kind == "inv":
        u_lo, u_hi = 1.0 / x_hi, 1.0 / x_lo
    elif kind == "inv1":
        u_lo, u_hi = 1.0 / (1.0 + x_hi), 1.0 / (1.0 + x_lo)
    else:
        u_lo, u_hi = x_lo, x_hi
    tol = tol_eps * EPS

    def ok(nseg):
        return _region_worst(kind, u_lo, u_hi, der, ncoef, nseg) <= tol

    lo, hi = 1, 1
    if ok(1):
        return 1, _region_worst(kind, u_lo, u_hi, der, ncoef, 1)
    hi = 2
    while hi <= seg_cap and not ok(hi):
        lo, hi = hi, hi * 2
    if hi > seg_cap:
        return None, _region_worst(kind, u_lo, u_hi, der, ncoef, seg_cap)
    while hi - lo > 1:                          # smallest ok in (lo, hi]
        mid = (lo + hi) // 2
        if ok(mid):
            hi = mid
        else:
            lo = mid
    return hi, _region_worst(kind, u_lo, u_hi, der, ncoef, hi)


# --------------------------------------------------------------------------
# degree vs segment-count trade table
# --------------------------------------------------------------------------
def trade_table(label, kind, x_lo, x_hi, degrees=(4, 5, 6), tol_eps=2.0):
    u_lo, u_hi = (1.0 / x_hi, 1.0 / x_lo) if kind == "inv" else (x_lo, x_hi)
    span = u_hi - u_lo
    var = "t=1/x" if kind == "inv" else "x"
    print(f"\n=== {label}: Chebyshev in {var}, x in [{x_lo},{x_hi}] "
          f"(equal segments in {var} over [{u_lo:.4f},{u_hi:.4f}], span {span:.4f}); "
          f"rel err <= {tol_eps:.0f}*eps ===")
    print(f"{'deg':>4} | {'R segs':>7} {f'd{var}':>9} {'err/eps':>8} "
          f"| {'R_d1 segs':>9} {f'd{var}':>9} {'err/eps':>8}")
    for deg in degrees:
        nc = deg + 1
        sR, eR = min_segments(kind, x_lo, x_hi, False, nc, tol_eps)
        sD, eD = min_segments(kind, x_lo, x_hi, True, nc, tol_eps)
        wR = f"{span/sR:.5f}" if sR else "-"
        wD = f"{span/sD:.5f}" if sD else "-"
        sRs = f"{sR}" if sR else ">cap"
        sDs = f"{sD}" if sD else ">cap"
        print(f"{deg:>4} | {sRs:>7} {wR:>9} {eR/EPS:>8.1f} "
              f"| {sDs:>9} {wD:>9} {eD/EPS:>8.1f}")


# --------------------------------------------------------------------------
# where is the error worst? / boundary diagnostics
# --------------------------------------------------------------------------
def error_profile(kind, x_lo, x_hi, der, ncoef, nseg):
    """Per-segment relative error; returns list of (x_mid, rel) ordered by the
    working variable (region 2 ends at x~=1, the boundary)."""
    u_lo, u_hi = (1.0 / x_hi, 1.0 / x_lo) if kind == "inv" else (x_lo, x_hi)
    edges = np.linspace(u_lo, u_hi, nseg + 1)
    out = []
    for j in range(nseg):
        _, rel = seg_fit(edges[j], edges[j + 1], kind, der, ncoef)
        um = 0.5 * (edges[j] + edges[j + 1])
        out.append(((1.0 / um if kind == "inv" else um), rel))
    return out


def show_profile(label, kind, x_lo, x_hi, der, ncoef, nseg, npts=8, bnd=1.0):
    prof = error_profile(kind, x_lo, x_hi, der, ncoef, nseg)
    xmax, emax = max(prof, key=lambda p: p[1])
    xbnd, ebnd = min(prof, key=lambda p: abs(p[0] - bnd))   # segment nearest the region boundary
    print(f"{label}: {nseg} segs, deg {ncoef-1}  ->  MAX {emax/EPS:5.2f} eps @ x={xmax:.4f}"
          f"   | seg@x~={xbnd:.3f}: {ebnd/EPS:.2f} eps")
    idx = np.unique(np.linspace(0, nseg - 1, npts).astype(int))
    print("    " + "  ".join(f"x={prof[i][0]:.3f}:{prof[i][1]/EPS:4.1f}" for i in idx))


def boundary_probe(ncoef, w=0.05, x0s=(0.7, 0.85, 1.0, 1.15, 1.3, 1.6, 2.0, 3.0)):
    print(f"\nequal-width probe (deg {ncoef-1}, physical x-width {w}): "
          f"rel err/eps of x-param vs 1/x-param on the SAME x-interval")
    print(f"{'x0':>6} | {'R: x':>7} {'1/x':>7} | {'R_d1: x':>8} {'1/x':>7} | lower(sum)")
    for x0 in x0s:
        xa, xb = x0 - w / 2, x0 + w / 2
        _, eRx = seg_fit(xa, xb, "x", False, ncoef)
        _, eDx = seg_fit(xa, xb, "x", True, ncoef)
        _, eRi = seg_fit(1.0 / xb, 1.0 / xa, "inv", False, ncoef)
        _, eDi = seg_fit(1.0 / xb, 1.0 / xa, "inv", True, ncoef)
        better = "x" if (eRx + eDx) <= (eRi + eDi) else "1/x"
        print(f"{x0:>6.2f} | {eRx/EPS:>7.2f} {eRi/EPS:>7.2f} | {eDx/EPS:>8.2f} "
              f"{eDi/EPS:>7.2f} | {better}")


def _probe_pair(ncoef, x0, w):
    xa, xb = x0 - w / 2, x0 + w / 2
    _, eRx = seg_fit(xa, xb, "x", False, ncoef)
    _, eRi = seg_fit(1.0 / xb, 1.0 / xa, "inv", False, ncoef)
    _, eDx = seg_fit(xa, xb, "x", True, ncoef)
    _, eDi = seg_fit(1.0 / xb, 1.0 / xa, "inv", True, ncoef)
    return eRx, eRi, eDx, eDi


def find_boundaries(ncoef, w=0.05, x0s=None):
    """Crossover x* where 1/x-param first beats x-param, separately for R and R_d1."""
    if x0s is None:
        x0s = np.round(np.arange(1.25, 1.86, 0.05), 2)
    print(f"--- crossover probe (deg {ncoef-1}, w={w}): rel err/eps, x vs 1/x ---")
    print(f"{'x0':>5} | {'R x':>7} {'R 1/x':>7} {'win':>4} | {'Rd1 x':>7} {'Rd1 1/x':>8} {'win':>4}")
    xbR = xbD = None
    for x0 in x0s:
        eRx, eRi, eDx, eDi = _probe_pair(ncoef, x0, w)
        wR = "1/x" if eRi <= eRx else "x"
        wD = "1/x" if eDi <= eDx else "x"
        if xbR is None and eRi <= eRx:
            xbR = x0
        if xbD is None and eDi <= eDx:
            xbD = x0
        print(f"{x0:>5.2f} | {eRx/EPS:>7.2f} {eRi/EPS:>7.2f} {wR:>4} | "
              f"{eDx/EPS:>7.2f} {eDi/EPS:>8.2f} {wD:>4}")
    return xbR, xbD


def reassess(xbR, xbD, x_cf=6.0, ncoef=7):
    print(f"\n--- reassess (deg {ncoef-1}, x_cf={x_cf}): segments with per-function boundary ---")
    print(f"{'fn':>5} {'boundary':>9} | {'reg1 (x)':>9} {'reg2 (1/x)':>11} {'total':>6} "
          f"| {'baseline@1.0':>12}")
    for name, der, xb, base in (("R", False, xbR, 82), ("R_d1", True, xbD, 99)):
        s1, _ = min_segments("x", 0.0, xb, der, ncoef)
        s2, _ = min_segments("inv", xb, x_cf, der, ncoef)
        tot = (s1 or 0) + (s2 or 0)
        print(f"{name:>5} {xb:>9.2f} | {str(s1):>9} {str(s2):>11} {tot:>6} | {base:>12}")


def boundary_scan(boundaries, x_cf=6.0, ncoef=7):
    print(f"\n--- total segments vs boundary (deg {ncoef-1}, x_cf={x_cf}) ---")
    print(f"{'xb':>5} | {'R r1':>5} {'R r2':>5} {'R tot':>6} | "
          f"{'Rd1 r1':>7} {'Rd1 r2':>7} {'Rd1 tot':>8}")
    for xb in boundaries:
        sR1, _ = min_segments("x", 0.0, xb, False, ncoef)
        sR2, _ = min_segments("inv", xb, x_cf, False, ncoef)
        sD1, _ = min_segments("x", 0.0, xb, True, ncoef)
        sD2, _ = min_segments("inv", xb, x_cf, True, ncoef)
        print(f"{xb:>5.2f} | {sR1:>5} {sR2:>5} {sR1+sR2:>6} | "
              f"{sD1:>7} {sD2:>7} {sD1+sD2:>8}")


def _frac_ok(nn, x_cf, der, ncoef, nseg, tol):
    """Can nseg uniform segments of t = nn/(nn+x) over x in [0, x_cf] hold tol?
    (early-exit: returns False on the first segment that fails)."""
    edges = np.linspace(nn / (nn + x_cf), 1.0, nseg + 1)   # x=0 -> t=1 ; x=x_cf -> t=nn/(nn+x_cf)
    for j in range(nseg):
        ua, ub = mp.mpf(edges[j]), mp.mpf(edges[j + 1])

        def f(s, ua=ua, ub=ub):
            t = ua + s * (ub - ua)
            x = nn / t - nn
            return MPMR.R_d1(x) if der else MPMR.R(x)
        with mp.workdps(DPS):
            coef, abserr = mp.chebyfit(f, [0, 1], ncoef, error=True)
            scale = min(abs(f(mp.mpf(0))), abs(f(mp.mpf(1))))
            if float(abserr / scale) > tol:
                return False
    return True


def frac_min_segments(nn, x_cf, der, ncoef, tol_eps=2.0, seg_cap=400):
    tol = tol_eps * EPS
    if _frac_ok(nn, x_cf, der, ncoef, 1, tol):
        return 1
    lo, hi = 1, 2
    while hi <= seg_cap and not _frac_ok(nn, x_cf, der, ncoef, hi, tol):
        lo, hi = hi, hi * 2
    if hi > seg_cap:
        return None
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if _frac_ok(nn, x_cf, der, ncoef, mid, tol):
            hi = mid
        else:
            lo = mid
    return hi


_SPECS = (
    # name, MPMR fn, x_cf (= highest-n bridge, also the Chebyshev cutoff), nseg, ncoef, reflect.
    # cf  = bridge convergent evaluated at x_cf;  cf_n = its order n.
    # cf_tiers = cheaper lower-n convergents for larger x, as (x>=threshold, n, expr) in
    #   DESCENDING threshold order; each threshold is that order's own x_cf (where its
    #   relative truncation reaches ~eps).  All share u = x*x.  Then doc.
    # R, R1, Rrel are emitted by emit_R_R1_Rrel() (custom hybrid split at x=1).  Only R3 is wlin.
    dict(name="R3", fn="R_d3", xcf=25.4, wlin=True, nseg_total=50, tgt=2.5, ncoef=9, cf_n=10, ovf=True,
         # reflection for x<0 (R'''(x)-R'''(-x) = M_SQRT2PI (x^3+3x) e^{x^2/2}; {exp}/{name} per target):
         refl="M_SQRT2PI * (-x) * (x*x + 3.0) * {exp}(0.5 * x * x) + {name}(-x)",
         cf="6.0*(1545.0 + u*(545.0 + u*(45.0 + u))) / (10395.0 + u*(17325.0 + u*(6930.0 + u*(990.0 + u*(55.0 + u)))))",
         cf_tiers=((17300.0, 4, "6.0 / (15.0 + u*(10.0 + u))"),
                   (210.0,   6, "6.0*(11.0 + u) / (105.0 + u*(105.0 + u*(21.0 + u)))"),
                   (50.5,    8, "6.0*(123.0 + u*(26.0 + u)) / (945.0 + u*(1260.0 + u*(378.0 + u*(36.0 + u))))")),
         doc="R3(x) = -R'''(x), scalar float (any sign); tiered CF convergents -R'''^[10..4] beyond x_cf."),
)


def _horner(ncoef, var="s", arr="c", off=None):
    """Ascending-coefficient Horner expression (constant first) in `var` over `arr`.
    With `off` (a local name) the flat array is indexed arr[off + k] (k=0 -> arr[off]),
    for bucket-major flat coefficient arrays; off=None gives the plain arr[k]."""
    def idx(k):
        if off is None:
            return f"{arr}[{k}]"
        return f"{arr}[{off}]" if k == 0 else f"{arr}[{off} + {k}]"
    h = idx(ncoef - 1)
    for k in range(ncoef - 2, -1, -1):
        h = f"({h})*{var} + {idx(k)}"
    return h


# Tiered continued-fraction ladders, shared by the pure-Python (.py) and the Cython (.pyx,
# MathPF) emitters so the two targets cannot drift -- a typo here fails verify().  Each entry
# is (XCF index, order n, expr in u=x*x and x); DESCENDING threshold (cheapest/largest-x
# first); the last entry is the bridge at XCF[0].  The expr text is valid in Python AND Cython.
R_CF = ((5, 0, "1.0 / x"),
        (4, 2, "(2.0 + u) / (x*(3.0 + u))"),
        (3, 4, "(8.0 + u*(9.0 + u)) / (x*(15.0 + u*(10.0 + u)))"),
        (2, 6, "(48.0 + u*(87.0 + u*(20.0 + u))) / (x*(105.0 + u*(105.0 + u*(21.0 + u))))"),
        (1, 8, "(384.0 + u*(975.0 + u*(345.0 + u*(35.0 + u)))) / (x*(945.0 + u*(1260.0 + u*(378.0 + u*(36.0 + u)))))"),
        (0, 10, "(3840.0 + u*(12645.0 + u*(6090.0 + u*(938.0 + u*(54.0 + u))))) / (x*(10395.0 + u*(17325.0 + u*(6930.0 + u*(990.0 + u*(55.0 + u))))))"))
R1_CF = ((4, 2, "1.0 / (3.0 + u)"),
         (3, 4, "(7.0 + u) / (15.0 + u*(10.0 + u))"),
         (2, 6, "(57.0 + u*(18.0 + u)) / (105.0 + u*(105.0 + u*(21.0 + u)))"),
         (1, 8, "(561.0 + u*(285.0 + u*(33.0 + u))) / (945.0 + u*(1260.0 + u*(378.0 + u*(36.0 + u))))"),
         (0, 10, "(6555.0 + u*(4680.0 + u*(840.0 + u*(52.0 + u)))) / (10395.0 + u*(17325.0 + u*(6930.0 + u*(990.0 + u*(55.0 + u)))))"))


def _cf_ladder(cf, arr, indent="    ", ovf_name=None):
    """Render the tiered-CF if-ladder lines for `cf` over threshold tuple `arr` (e.g. 'XCF_R').
    Sets u=x*x, optional overflow guard `if u > ovf_name: return 0.0`, then checks the
    largest threshold first and falls through to the bridge.  Identical text for .py / .pyx."""
    g = indent + "    "
    out = [f"{indent}if x >= {arr}[0]:", f"{g}u = x * x"]
    if ovf_name:
        out += [f"{g}if u > {ovf_name}:", f"{g}    return 0.0"]
    last = len(cf) - 1
    for j, (idx, n, expr) in enumerate(cf):
        if j < last:
            out += [f"{g}if x >= {arr}[{idx}]:  # n={n} convergent", f"{g}    return {expr}"]
        else:
            out += [f"{g}return {expr}  # n={n} (bridge)"]
    return out


def _verify_seg(coef, ncoef, f):
    """Worst relative double-Horner error of `coef` vs mpmath `f` over s in [0,1]."""
    worst = 0.0
    for s in np.linspace(0.0, 1.0, 21):
        p = coef[-1]
        for k in range(ncoef - 2, -1, -1):
            p = p * s + coef[k]
        with mp.workdps(DPS):
            ref = float(f(mp.mpf(s)))
        worst = max(worst, abs((p - ref) / ref))
    return worst


def wlin_fit(nn, xcf, ncoef, fn, nseg, head_factor):
    """XCF-independent bucketing.  w = x/(nn+x) in [0,1] is split into `nseg` equal
    buckets; bucket i (local s in [0,1]) maps to w=(i+s)/nseg, x=nn*w/(1-w).  Fit a
    degree-(ncoef-1) Chebyshev per bucket.  Store buckets 0..i_cap covering
    [0, head_factor*xcf]: the buckets covering [0, xcf] (0..i_star) plus a headroom band
    out to ~head_factor*xcf, so a moderate rise of xcf needs no refit.  The partition --
    hence the coefficients -- does not depend on xcf.
    Returns (rows, used_worst_eps, head_worst_eps, headroom, i_star)."""
    i_star = int(xcf / (nn + xcf) * nseg)             # bucket containing xcf
    xc = head_factor * xcf
    i_cap = int(xc / (nn + xc) * nseg)                # bucket containing head_factor*xcf
    rows, used_worst, head_worst = [], 0.0, 0.0
    for i in range(i_cap + 1):
        def f(s, i=i):
            w = (i + s) / nseg
            return fn(nn * w / (1.0 - w))
        with mp.workdps(DPS):
            coef = [float(c) for c in mp.chebyfit(f, [0, 1], ncoef, error=False)][::-1]
        bw = _verify_seg(coef, ncoef, f)
        if i <= i_star:
            used_worst = max(used_worst, bw)
        else:
            head_worst = max(head_worst, bw)
        rows.append("    " + ", ".join(repr(c) for c in coef) + ",")  # flat: one bucket per line
    return rows, used_worst / EPS, head_worst / EPS, i_cap - i_star, i_star


def int_arg_anchors(mmax=5):
    """Lines defining three integer-argument anchor arrays (m = 1 .. mmax), as
    correctly-rounded doubles.  These are tabulated values of R, R1, and log R1 at
    small positive integer arguments; useful to downstream packages that need exact
    integer-m anchors (not part of mathpf's surface, which exposes the continuous
    R/R1/R3).  Stored as separate 1-D columns (stdlib array('d')), indexed [m-1]:

        T_NR1[m-1]   = n(m) R1(m) = n(m) - m N(-m)
        T_R1[m-1]    = R1(m) = -R'(m) = 1 - m R(m)
        T_LOGR1[m-1] = log R1(m)

    Not called by emit_module() -- run this standalone and paste the output into
    whichever downstream module wants the anchors.
    """
    nr1, r1, logr1 = [], [], []
    with mp.workdps(60):
        for m in range(1, mmax + 1):
            R1m = MPMR.R_d1(m)                       # 1 - m R(m) = -R'(m)
            nr1.append(float(mp.npdf(m) * R1m))
            r1.append(float(R1m))
            logr1.append(float(mp.log(R1m)))
    out = [f"# Integer-argument anchor arrays (m = 1..{mmax}).  Separate 1-D columns,",
           "# indexed [m-1], R1(m) = -R'(m) = 1 - m R(m):",
           "#   T_NR1   = n(m) R1(m) = n(m) - m N(-m)",
           "#   T_R1    = R1(m) (anchor slope),  T_LOGR1 = log R1(m).",
           "T_NR1 = array('d', (" + ", ".join(repr(v) for v in nr1) + "))",
           "T_R1 = array('d', (" + ", ".join(repr(v) for v in r1) + "))",
           "T_LOGR1 = array('d', (" + ", ".join(repr(v) for v in logr1) + "))",
           ""]
    return out


def _lobatto_fit(fn, ncoef, dps=60):
    """Interpolate fn at `ncoef` Chebyshev-Lobatto nodes on [0,1] (x_k=(1-cos(pi k/N))/2,
    N=ncoef-1; the set INCLUDES the endpoints 0 and 1).  Pinning x=0 forces the constant
    term to fn(0) exactly, killing the node-gap ringing that a node-interior fit (chebfit)
    leaves in the unconstrained sliver near 0.  Returns ascending monomial coeffs (doubles)."""
    N = ncoef - 1
    with mp.workdps(dps):
        nodes = [(1 - mp.cos(mp.pi * k / N)) / 2 for k in range(N + 1)]
        V, b = mp.matrix(ncoef, ncoef), mp.matrix(ncoef, 1)
        for r, xn in enumerate(nodes):
            xp = mp.mpf(1)
            for c in range(ncoef):
                V[r, c] = xp
                xp *= xn
            b[r] = fn(xn)
        sol = mp.lu_solve(V, b)
    return [float(sol[k]) for k in range(ncoef)]


def emit_R_R1_Rrel(xcf_r=(11.4, 15.1, 24.1, 59.3, 548.0, 67000000.0),
                   xcf_r1=(14.5, 21.2, 41.0, 165.0, 12800.0),
                   nfrac_r1=2.5, nseg_r1=32, deg_rrel=14, deg_r1=7):
    """Build the R / R1 / Rrel constant lines + function sources for the hybrid split at x=1.

    Storage is memory-minimal: each primitive is stored only where it is well-conditioned.
      * C_Rrel : a single deg-`deg_rrel` poly of  Rrel_below1(x) = (sqrt(pi/2) - R(x))/x  over
                 x in [0,1], in DIRECT x (one segment, s=x so dx/ds=1 -- immune to the bucket-
                 width rounding that hurts a wide t-map).  Fit at Chebyshev-LOBATTO nodes so the
                 node at x=0 pins Rrel_below1(0) = -R'(0) = 1 exactly (c0 = 1.0), avoiding the
                 near-0 ringing of an interior-node fit.  It is positive on (0,1].  Recovers
                 R(x)  = sqrt(pi/2) - x*Rrel_below1(x)   and   R1(x) = 1 - x*R(x).
      * C_R1   : a deg-`deg_r1` Chebyshev table of R1(x) = -R'(x) for x>1, bucketed in the
                 SHIFTED t = (x-1)/(nfrac+x) in [0,1] (t=0 at x=1), `nseg` equal buckets.
                 R recovers for x>1 as  R = (1 - R1)/x  (amplification R1/(1-R1) < 0.53).
    Beyond x_cf each uses its own tiered Laplace continued fraction (XCF_R / XCF_R1).  R is
    the hot path: for 1<x<XCF_R[0] it reads C_R1 directly (it reaches only x<XMAX_R1), so no
    R1 continued fraction is inlined into R.  Returns (const_lines, [Rrel_src, R_src, R1_src])."""
    C = mp.sqrt(mp.pi / 2)
    Cf = float(C)
    nc_rrel, nc_r1 = deg_rrel + 1, deg_r1 + 1

    # ---- Rrel_below1: one deg-deg_rrel poly of (sqrt(pi/2)-R(x))/x over x in [0,1] (direct x,
    #      Chebyshev-Lobatto so x=0 is a node and the constant term is pinned to 1) ----
    def frel(x):
        x = mp.mpf(x)
        if x == 0:
            return mp.mpf(1)                      # Rrel_below1(0) = -R'(0) = 1
        return (C - MPMR.R(x)) / x
    c_rrel = _lobatto_fit(frel, nc_rrel)
    # real double-eval: R = sqrt(pi/2) - x*Horner(Rrel_below1), R1 = 1 - x*R, and Rrel itself
    rrel_R_eps = rrel_R1_eps = rrel_self_eps = 0.0
    for x in np.linspace(0.0, 1.0, 401):
        p = c_rrel[-1]
        for k in range(nc_rrel - 2, -1, -1):
            p = p * x + c_rrel[k]
        Rd = Cf - x * p
        with mp.workdps(DPS):
            rref = float(MPMR.R(mp.mpf(x)))
        rrel_R_eps = max(rrel_R_eps, abs((Rd - rref) / rref))
        if x > 0.0:
            R1d = 1.0 - x * Rd
            with mp.workdps(DPS):
                r1ref = float(MPMR.R_d1(mp.mpf(x)))
                fref = float(frel(mp.mpf(x)))
            rrel_R1_eps = max(rrel_R1_eps, abs((R1d - r1ref) / r1ref))
            rrel_self_eps = max(rrel_self_eps, abs((p - fref) / fref))
    rrel_R_eps /= EPS
    rrel_R1_eps /= EPS
    rrel_self_eps /= EPS

    # ---- R1: deg-deg_r1 Chebyshev buckets of t=(x-1)/(nfrac+x) over x>1 ----
    x_top = max(xcf_r1[0], xcf_r[0])              # table must reach here (R reads it to XCF_R[0])
    i_cap = int(nseg_r1 * (x_top - 1.0) / (nfrac_r1 + x_top))
    nb = i_cap + 1                                # buckets 0..nb-1; bucket nb-1 extends past x_top
    r1_rows = []
    for i in range(nb):
        def f(s, i=i):
            t = (i + s) / nseg_r1
            x = (1.0 + nfrac_r1 * t) / (1.0 - t)
            return MPMR.R_d1(x)
        with mp.workdps(DPS):
            coef = [float(c) for c in mp.chebyfit(f, [0, 1], nc_r1, error=False)][::-1]
        r1_rows.append(coef)
    xmax_r1 = (1.0 + nfrac_r1 * nb / nseg_r1) / (1.0 - nb / nseg_r1)
    # real double-eval over [1, x_top] using the RUNTIME closed-form s (catches dx/ds blow-up)
    r1_tab_eps = 0.0
    for x in np.linspace(1.0 + 1e-6, x_top, 800):
        d = nfrac_r1 + x
        i = int(nseg_r1 * (x - 1.0) / d)
        if i >= nb:
            i = nb - 1
        s = ((nseg_r1 - i) * x - (nseg_r1 + i * nfrac_r1)) / d
        c = r1_rows[i]
        p = c[-1]
        for k in range(nc_r1 - 2, -1, -1):
            p = p * s + c[k]
        with mp.workdps(DPS):
            ref = float(MPMR.R_d1(mp.mpf(x)))
        r1_tab_eps = max(r1_tab_eps, abs((p - ref) / ref))
    r1_tab_eps /= EPS

    print(f"Rrel_below1: deg{deg_rrel} Lobatto on [0,1] (c0={c_rrel[0]:.16f})  -> "
          f"self={rrel_self_eps:.2f} R={rrel_R_eps:.2f} R1={rrel_R1_eps:.2f} eps")
    print(f"R1:   deg{deg_r1} t-map NF={nfrac_r1} NSEG={nseg_r1} NB={nb} (x in [1,{xmax_r1:.2f}], "
          f"reads to {x_top})  -> table={r1_tab_eps:.2f} eps")

    # M_SQRT2PI_2 and XCF_R are emitted at module top, next to M_SQRT2PI / _U_MAX (general consts).
    top = [
        f"M_SQRT2PI_2 = {Cf!r}  # R(0) = sqrt(pi/2) = M_SQRT2PI/2",
        f"XCF_R = {tuple(float(v) for v in xcf_r)!r}  # (bridge x_cf=n=10, then ascending CF tiers n=8,6,4,2,0)",
    ]
    const = [
        "# --- Rrel_below1: the [0,1] primitive shared by R and R1 (see emit_R_R1_Rrel) ---",
        f"# C_Rrel: deg-{deg_rrel} Horner of Rrel_below1(x) = (sqrt(pi/2)-R(x))/x over x in [0,1]",
        f"#   ({nc_rrel} coeffs, ascending, >0).  Recover R = sqrt(pi/2) - x*Rrel_below1, R1 = 1 - x*R.",
        "C_Rrel = array('d', (" + ", ".join(repr(c) for c in c_rrel) + "))",
        "",
        "# --- R / R1: hybrid split at x=1 (memory-minimal; see emit_R_R1_Rrel) ---",
        "# x<=1: use C_Rrel above.  x>1: R1 has a deg-%d table (C_R1) in t=(x-1)/(N_FRAC_R1+x)," % deg_r1,
        "#   R = (1-R1)/x.  Beyond x_cf each uses its own tiered CF (XCF_R at top, XCF_R1 below).",
        f"XCF_R1 = {tuple(float(v) for v in xcf_r1)!r}  # (bridge x_cf=n=10, then ascending CF tiers n=8,6,4,2)",
        f"N_FRAC_R1 = {nfrac_r1!r}  # shifted-t scale: t = (x-1)/(N_FRAC_R1 + x), t=0 at x=1",
        f"NSEG_R1 = {nseg_r1}  # equal t-buckets over [0,1] (XCF-independent partition)",
        f"TMAX_R1 = {nb}  # stored buckets 0..TMAX_R1-1 of C_R1; cover x in [1, XMAX_R1]",
        f"XMAX_R1 = {xmax_r1!r}  # max x covered by C_R1 (R reads C_R1 up to XCF_R[0] < XMAX_R1)",
        f"# C_R1: flat array('d'), {nb} buckets x {nc_r1} coeffs (deg-{deg_r1}); bucket i = C_R1[i*{nc_r1}+k], k=0..{nc_r1 - 1}.",
        "#   i = int(NSEG_R1*(x-1)/(N_FRAC_R1+x)), s = ((NSEG_R1-i)*x-(NSEG_R1+i*N_FRAC_R1))/(N_FRAC_R1+x) (closed form)",
        "C_R1 = array('d', (",
    ]
    const += ["    " + ", ".join(repr(c) for c in row) + "," for row in r1_rows]
    const += ["))", ""]

    rrel_src = "\n".join([
        "def Rrel_below1(x):",
        f'    """Rrel_below1(x) = (sqrt(pi/2) - R(x))/x for x in [0,1] (deg-{deg_rrel}, >0);',
        '    R(x) = M_SQRT2PI_2 - x*Rrel_below1(x).  Pinned so Rrel_below1(0) = -R\'(0) = 1."""',
        "    c = C_Rrel",
        f"    return {_horner(nc_rrel, 'x', 'c')}",
    ])
    r_src = "\n".join([
        "def R(x):",
        '    """Mills ratio R(x) = N(-x)/n(x), scalar float (any sign)."""',
        "    if x < 0.0:",
        "        return M_SQRT2PI * math.exp(0.5 * x * x) - R(-x)",
        "    if x <= 1.0:                        # R = sqrt(pi/2) - x*Rrel_below1(x)  (inlined)",
        "        c = C_Rrel",
        f"        return M_SQRT2PI_2 - x * ({_horner(nc_rrel, 'x', 'c')})",
        *_cf_ladder(R_CF, "XCF_R"),
        "    d = N_FRAC_R1 + x                   # 1 < x < XCF_R[0]: R = (1 - R1)/x, R1 from C_R1",
        "    i = int(NSEG_R1 * (x - 1.0) / d)",
        "    if i >= TMAX_R1:",
        "        i = TMAX_R1 - 1",
        "    s = ((NSEG_R1 - i) * x - (NSEG_R1 + i * N_FRAC_R1)) / d",
        f"    o = i * {nc_r1}",
        "    c = C_R1",
        f"    return (1.0 - ({_horner(nc_r1, 's', 'c', off='o')})) / x",
    ])
    r1_src = "\n".join([
        "def R1(x):",
        '    """R1(x) = 1 - x R(x) = -R\'(x), scalar float (any sign)."""',
        "    if x < 0.0:                         # reflection: R1(x) = R1(-x) - M_SQRT2PI*x*exp(x^2/2)",
        "        return M_SQRT2PI * (-x) * math.exp(0.5 * x * x) + R1(-x)",
        "    if x <= 1.0:                        # R1 = 1 - x*R,  R = sqrt(pi/2)-x*Rrel_below1 (inlined)",
        "        c = C_Rrel",
        f"        return 1.0 - x * (M_SQRT2PI_2 - x * ({_horner(nc_rrel, 'x', 'c')}))",
        *_cf_ladder(R1_CF, "XCF_R1", ovf_name="_U_MAX"),
        "    d = N_FRAC_R1 + x                   # 1 < x < XCF_R1[0]: deg-%d t-bucket" % deg_r1,
        "    i = int(NSEG_R1 * (x - 1.0) / d)",
        "    if i >= TMAX_R1:",
        "        i = TMAX_R1 - 1",
        "    s = ((NSEG_R1 - i) * x - (NSEG_R1 + i * N_FRAC_R1)) / d",
        f"    o = i * {nc_r1}",
        "    c = C_R1",
        f"    return {_horner(nc_r1, 's', 'c', off='o')}",
    ])
    return top, const, [rrel_src, r_src, r1_src]


def verify(npts=400):
    """Re-import the freshly generated mathpf._pyref.mills_cheby and sweep R, R1, Rrel, R3
    against mpmath in real double precision (the definitive check -- catches closed-form-s
    rounding)."""
    import importlib
    from mathpf._pyref import mills_cheby
    importlib.reload(mills_cheby)
    m = mills_cheby
    worst_all = 0.0

    def sweep(label, fn, ref, xs):
        nonlocal worst_all
        worst, xw = 0.0, 0.0
        for x in xs:
            x = float(x)
            got = fn(x)
            with mp.workdps(50):
                r = float(ref(mp.mpf(x)))
            e = abs(got - r) / abs(r) if r != 0.0 else abs(got)
            if e > worst:
                worst, xw = e, x
        worst_all = max(worst_all, worst)
        print(f"  {label:8s} {worst/EPS:8.2f} eps @ x={xw:< 12.5g} ({len(xs)} pts)")

    print("verify (real double eval vs mpmath):")
    sweep("R<=1", m.R, MPMR.R, np.linspace(0.0, 1.0, npts))
    sweep("R mid", m.R, MPMR.R, np.linspace(1.0 + 1e-6, m.XCF_R[0] - 0.01, npts))
    sweep("R cf", m.R, MPMR.R, np.linspace(m.XCF_R[0], 1000.0, npts))
    sweep("R huge", m.R, MPMR.R, np.geomspace(1e3, 1e8, 80))
    sweep("R neg", m.R, MPMR.R, np.linspace(-30.0, -0.001, npts))
    sweep("Rrelb1", m.Rrel_below1, lambda x: (mp.sqrt(mp.pi / 2) - MPMR.R(x)) / x, np.linspace(0.001, 1.0, npts))
    sweep("R1<=1", m.R1, MPMR.R_d1, np.linspace(0.001, 1.0, npts))
    sweep("R1 mid", m.R1, MPMR.R_d1, np.linspace(1.0 + 1e-6, m.XCF_R1[0] - 0.01, npts))
    sweep("R1 cf", m.R1, MPMR.R_d1, np.linspace(m.XCF_R1[0], 1000.0, npts))
    sweep("R1 huge", m.R1, MPMR.R_d1, np.geomspace(1e3, 1e10, 80))
    sweep("R3", m.R3, MPMR.R_d3, np.linspace(0.001, 25.39, npts))
    sweep("R3 cf", m.R3, MPMR.R_d3, np.linspace(25.4, 500.0, npts))
    print(f"WORST overall: {worst_all/EPS:.2f} eps")
    return worst_all / EPS


def emit_module(path="src/mathpf/_pyref/mills_cheby.py", nn=3.5, specs=_SPECS, head_factor=1.2):
    """Generate the scalar Mills-ratio module (mathpf._pyref.mills_cheby).  R / R1 / Rrel
    use a memory-minimal hybrid split at x=1 (Rrel deg poly on [0,1] in direct x; R1 deg poly
    table in t=(x-1)/(N_FRAC_R1+x) for x>1; R recovers from whichever side is well-conditioned)
    -- see emit_R_R1_Rrel.  R3 uses the XCF-independent w=x/(nn+x) bucketing (wlin_fit) over
    [0, head_factor*x_cf].  All use tiered continued-fraction convergents beyond x_cf.
    Verifies, then writes `path`.  Default path assumes invocation from the MathPF repo root."""
    out = ['"""Reference scalar implementations of the Mills ratio  R(x) = N(-x)/n(x),',
           "R1(x) = -R'(x) = 1 - x R(x), R3(x) = -R'''(x), and Rrel_below1(x) = (sqrt(pi/2) - R(x))/x.",
           "Auto-generated by tools/cheby_fit.emit_module(); do not edit by hand.",
           "",
           "Layout (memory-minimal -- each primitive is stored only where it is well-conditioned):",
           "  R, R1, Rrel_below1  hybrid split at x=1:",
           "    * x <= 1 : Rrel_below1 = (sqrt(pi/2)-R)/x is one deg-N poly in x (C_Rrel, >0);",
           "               R = sqrt(pi/2) - x*Rrel_below1,   R1 = 1 - x*R.",
           "    * x  > 1 : R1 has a deg-M Chebyshev table (C_R1) bucketed in t=(x-1)/(N_FRAC_R1+x);",
           "               R recovers as  R = (1 - R1)/x.",
           "    * x >= x_cf : tiered Laplace continued fraction (XCF_R / XCF_R1).",
           "  R3  segmented Chebyshev in w=x/(N_FRAC_R3+x) over [0,x_cf] (C_R3), CF beyond (XCF_R3).",
           'Coefficients are ascending (constant first); each piece is a Horner evaluation."""',
           "import math, sys", "from array import array", "", "M_SQRT2PI = 2.5066282746310002"]
    rrel_top, rrel_const, rrel_funcs = emit_R_R1_Rrel()
    out += rrel_top                                # M_SQRT2PI_2, XCF_R -- with the general consts
    out += ["# _U_MAX = sqrt(DBL_MAX) ~ 1.34e154.  Past it x*x overflows the quadratic CF",
            "# denominators of R1/R3; there the functions are ~0, so the guard returns 0.",
            "_U_MAX = math.sqrt(sys.float_info.max)",
            ""]
    # Note: integer-argument anchor tables (T_NR1, T_R1, T_LOGR1) are NOT emitted into
    # mathpf -- they are downstream-application-specific.  Run int_arg_anchors() standalone
    # if you need to refresh them, and paste the output into the consuming package.
    out += rrel_const
    funcs = list(rrel_funcs)
    for sp in specs:
        name, xcf, ncoef = sp["name"], sp["xcf"], sp["ncoef"]
        fn = getattr(MPMR, sp["fn"])
        nseg = sp["nseg_total"]
        rows, used_worst, head_worst, headroom, i_star = wlin_fit(nn, xcf, ncoef, fn, nseg, head_factor)
        ntab = len(rows)
        x_max = nn * ntab / (nseg - ntab)             # max x reached by the stored buckets
        tiers = sp.get("cf_tiers", ())                # descending threshold
        ntier = len(tiers)
        xcf_tuple = (xcf,) + tuple(thr for thr, _, _ in reversed(tiers))   # ascending tier thresholds
        flag = "" if max(used_worst, head_worst) <= sp.get("tgt", 2.5) else "  <-- ABOVE tgt!"
        print(f"{name}: XCF={xcf} NSEG={nseg} deg={ncoef-1}  buckets={ntab} "
              f"(0..{i_star} cover [0,XCF], +{headroom} headroom to x~{x_max:.0f}={x_max/xcf:.2f}xXCF)  "
              f"used={used_worst:.2f} head={head_worst:.2f} eps{flag}")
        out += [f"# --- {name}: XCF-independent w-bucketing,  w = x/(N_FRAC_{name}+x) in [0,1] cut into NSEG_{name}",
                f"#   equal buckets (FIXED partition).  i = int(x/(N_FRAC_{name}+x)*NSEG_{name});  local coordinate",
                f"#   s = ((NSEG_{name}-i)*x - i*N_FRAC_{name})/(N_FRAC_{name}+x) (not NSEG*w-i) keeps precision at large x.",
                f"N_FRAC_{name} = {nn!r}  # w-bucket scale: w = x/(N_FRAC_{name} + x)",
                f"XCF_{name} = {xcf_tuple!r}  # (bridge x_cf, then ascending CF tier thresholds); XCF_{name}[0] in bucket {i_star}",
                f"NSEG_{name} = {nseg}  # total equal-width buckets of w=x/(N_FRAC_{name}+x) over [0,1] (XCF-independent)",
                f"TMAX_{name} = {ntab}  # coefficients cover buckets 0..TMAX_{name}-1, i.e. t in [0, TMAX_{name})",
                f"XMAX_{name} = {x_max!r}  # max x covered by C_{name} (~{x_max/xcf:.2f}x bridge); XCF_{name}[0] may rise to XMAX_{name} w/o refit",
                f"# C_{name}: flat array('d'), {ntab} buckets x {ncoef} coeffs; bucket i = C_{name}[i*{ncoef}+k], k=0..{ncoef - 1}",
                f"#   (0..{i_star} cover [0,XCF_{name}[0]], +{headroom} headroom to ~{head_factor:g}x)",
                f"C_{name} = array('d', (", *rows, "))", ""]
        body = [f"def {name}(x):", f'    """{sp["doc"]}"""']
        if sp.get("refl"):
            body += ["    if x < 0.0:",
                     "        return " + sp["refl"].format(exp="math.exp", name=name)]
        body += [f"    if x >= XCF_{name}[0]:", "        u = x * x"]
        if sp.get("ovf"):                             # overflow guard: huge x -> denominator blows up, asymptote 0
            body += ["        if u > _U_MAX:", "            return 0.0"]
        for j, (thr, ncf, expr) in enumerate(tiers):  # largest threshold first
            body += [f"        if x >= XCF_{name}[{ntier - j}]:  # n={ncf} convergent",
                     f"            return {expr}"]
        body += [f"        return {sp['cf']}  # n={sp['cf_n']} convergent (x_cf bridge)",
                 f"    d = N_FRAC_{name} + x",
                 f"    i = int(x / d * NSEG_{name})",
                 f"    s = ((NSEG_{name} - i) * x - i * N_FRAC_{name}) / d",
                 f"    o = i * {ncoef}",
                 f"    c = C_{name}",
                 f"    return {_horner(ncoef, off='o')}"]
        funcs.append("\n".join(body))
    out.append("")
    out.append("\n\n\n".join(funcs))
    with open(path, "w") as fh:
        fh.write("\n".join(out) + "\n")
    print(f"wrote {path}")


def emit_mathpf(dest="D:/Github/MathPF/src/mathpf"):
    """Generate the MathPF Cython module (scope #1: Mills-ratio kernels) from the committed
    coefficient data in the generated mills_cheby.  Writes three files into `dest`:
      _mills_coef.h : static const double[] data only (so a future C/C++ port can #include it),
      mills.pxd     : cimport-able scalar kernels  _R / _R1 / _R3 / _Rrel_below1,
      mills.pyx     : the kernels (libc.math, branch + Horner + tiered CF) and numpy-vectorized
                      wrappers millsratio / millsratio_d1 / millsratio_d3 / millsratio_rel.
    The CF ladders (R_CF/R1_CF, R3 from _SPECS) and Horner come from the SAME helpers as the
    .py emitter, so .py and .pyx cannot drift (a CF typo would fail verify()).  Building needs
    Cython + a C compiler -- run in the MathPF env / cibuildwheel CI, not here."""
    import importlib, os
    import mills_cheby
    importlib.reload(mills_cheby)
    mc = mills_cheby
    nc_rrel = len(mc.C_Rrel)                       # 15 (deg 14)
    nc_r1 = len(mc.C_R1) // mc.TMAX_R1             # 8  (deg 7)
    nc_r3 = len(mc.C_R3) // mc.TMAX_R3             # 9  (deg 8)

    def carr(name, vals):
        return f"static const double {name}[{len(vals)}] = {{" + ", ".join(repr(float(v)) for v in vals) + "};"

    h = ["/* Auto-generated by cheby_fit.emit_mathpf -- do not edit.",
         " * Coefficient data for the Mills-ratio kernels (mathpf.mills).  Data only, so a C/C++",
         " * port can #include it too; the kernels live in mills.pyx. */",
         "#ifndef MATHPF_MILLS_COEF_H", "#define MATHPF_MILLS_COEF_H", "",
         f"static const double M_SQRT2PI   = {float(mc.M_SQRT2PI)!r};",
         f"static const double M_SQRT2PI_2 = {float(mc.M_SQRT2PI_2)!r};  /* R(0) = sqrt(pi/2) */",
         f"static const double U_MAX       = {float(mc._U_MAX)!r};  /* sqrt(DBL_MAX); R1/R3 -> 0 beyond */",
         f"static const double X_NEG_MAX   = -37.5;  /* x < this: R/R1/R3 reflection's exp(x^2/2) overflows; saturate to +inf */",
         "", carr("XCF_R", mc.XCF_R), carr("XCF_R1", mc.XCF_R1), carr("XCF_R3", mc.XCF_R3), "",
         f"static const double N_FRAC_R1 = {float(mc.N_FRAC_R1)!r};",
         f"static const int    NSEG_R1   = {int(mc.NSEG_R1)};",
         f"static const int    TMAX_R1   = {int(mc.TMAX_R1)};",
         f"static const double N_FRAC_R3 = {float(mc.N_FRAC_R3)!r};",
         f"static const int    NSEG_R3   = {int(mc.NSEG_R3)};",
         f"static const int    TMAX_R3   = {int(mc.TMAX_R3)};", "",
         f"/* Rrel_below1: deg-{nc_rrel - 1}, single segment on [0,1] */", carr("C_Rrel", mc.C_Rrel),
         f"/* R1 = -R': flat {mc.TMAX_R1} x {nc_r1} (deg-{nc_r1 - 1}); bucket i = C_R1[i*{nc_r1}+k] */", carr("C_R1", mc.C_R1),
         f"/* R3 = -R''': flat {mc.TMAX_R3} x {nc_r3} (deg-{nc_r3 - 1}); bucket i = C_R3[i*{nc_r3}+k] */", carr("C_R3", mc.C_R3),
         "", "#endif", ""]

    pxd = ["# Cython header for mathpf.mills -- cimport the scalar kernels, e.g.",
           "#   from mathpf.mills cimport _R, _R1, _R3, _Rrel_below1",
           "# Auto-generated by cheby_fit.emit_mathpf -- do not edit.",
           "cdef double _R(double x) noexcept nogil",
           "cdef double _R1(double x) noexcept nogil",
           "cdef double _R3(double x) noexcept nogil",
           "cdef double _Rrel_below1(double x) noexcept nogil", ""]

    sp3 = next(s for s in _SPECS if s["name"] == "R3")
    t3 = sp3["cf_tiers"]
    nt = len(t3)
    r3_cf = [(nt - j, n, e) for j, (thr, n, e) in enumerate(t3)] + [(0, sp3["cf_n"], sp3["cf"])]

    extern = ['cdef extern from "_mills_coef.h":',
              "    const double M_SQRT2PI", "    const double M_SQRT2PI_2", "    const double U_MAX",
              "    const double X_NEG_MAX",
              f"    const double XCF_R[{len(mc.XCF_R)}]",
              f"    const double XCF_R1[{len(mc.XCF_R1)}]",
              f"    const double XCF_R3[{len(mc.XCF_R3)}]",
              "    const double N_FRAC_R1", "    const int NSEG_R1", "    const int TMAX_R1",
              "    const double N_FRAC_R3", "    const int NSEG_R3", "    const int TMAX_R3",
              f"    const double C_Rrel[{nc_rrel}]",
              f"    const double C_R1[{len(mc.C_R1)}]",
              f"    const double C_R3[{len(mc.C_R3)}]"]

    krrel = ["cdef double _Rrel_below1(double x) noexcept nogil:",
             '    """(sqrt(pi/2) - R(x))/x on [0,1] (deg-%d); R = M_SQRT2PI_2 - x*Rrel_below1."""' % (nc_rrel - 1),
             "    cdef const double* c = &C_Rrel[0]",
             f"    return {_horner(nc_rrel, 'x', 'c')}"]
    kR = ["cdef double _R(double x) noexcept nogil:",
          '    """Mills ratio R(x) = N(-x)/n(x); any sign.  Saturates to +inf for x < X_NEG_MAX."""',
          "    cdef double u, s, d", "    cdef int i, o", "    cdef const double* c",
          "    if x < 0.0:",
          "        if x < X_NEG_MAX:                                  # exp(x^2/2) overflow -> saturate",
          "            return INFINITY",
          "        return M_SQRT2PI * exp(0.5 * x * x) - _R(-x)",
          "    if x <= 1.0:", "        c = &C_Rrel[0]",
          f"        return M_SQRT2PI_2 - x * ({_horner(nc_rrel, 'x', 'c')})",
          *_cf_ladder(R_CF, "XCF_R"),
          "    d = N_FRAC_R1 + x",
          "    i = <int>(NSEG_R1 * (x - 1.0) / d)",
          "    if i >= TMAX_R1:", "        i = TMAX_R1 - 1",
          "    s = ((NSEG_R1 - i) * x - (NSEG_R1 + i * N_FRAC_R1)) / d",
          f"    o = i * {nc_r1}", "    c = &C_R1[0]",
          f"    return (1.0 - ({_horner(nc_r1, 's', 'c', off='o')})) / x"]
    kR1 = ["cdef double _R1(double x) noexcept nogil:",
           '    """R1(x) = 1 - x R(x) = -R\'(x); any sign.  Saturates to +inf for x < X_NEG_MAX."""',
           "    cdef double u, s, d", "    cdef int i, o", "    cdef const double* c",
           "    if x < 0.0:",                              # reflection: R1(x) = R1(-x) - M_SQRT2PI*x*exp(x^2/2)
           "        if x < X_NEG_MAX:                                  # exp(x^2/2) overflow -> saturate",
           "            return INFINITY",
           "        return M_SQRT2PI * (-x) * exp(0.5 * x * x) + _R1(-x)",
           "    if x <= 1.0:", "        c = &C_Rrel[0]",
           f"        return 1.0 - x * (M_SQRT2PI_2 - x * ({_horner(nc_rrel, 'x', 'c')}))",
           *_cf_ladder(R1_CF, "XCF_R1", ovf_name="U_MAX"),
           "    d = N_FRAC_R1 + x",
           "    i = <int>(NSEG_R1 * (x - 1.0) / d)",
           "    if i >= TMAX_R1:", "        i = TMAX_R1 - 1",
           "    s = ((NSEG_R1 - i) * x - (NSEG_R1 + i * N_FRAC_R1)) / d",
           f"    o = i * {nc_r1}", "    c = &C_R1[0]",
           f"    return {_horner(nc_r1, 's', 'c', off='o')}"]
    kR3 = ["cdef double _R3(double x) noexcept nogil:",
           '    """R3(x) = -R\'\'\'(x); any sign.  Saturates to +inf for x < X_NEG_MAX."""',
           "    cdef double u, s, d", "    cdef int i, o", "    cdef const double* c",
           "    if x < 0.0:",
           "        if x < X_NEG_MAX:                                  # exp(x^2/2) overflow -> saturate",
           "            return INFINITY",
           "        return " + sp3["refl"].format(exp="exp", name="_R3"),
           *_cf_ladder(r3_cf, "XCF_R3", ovf_name="U_MAX"),
           "    d = N_FRAC_R3 + x",
           "    i = <int>(x / d * NSEG_R3)",
           "    s = ((NSEG_R3 - i) * x - i * N_FRAC_R3) / d",
           f"    o = i * {nc_r3}", "    c = &C_R3[0]",
           f"    return {_horner(nc_r3, 's', 'c', off='o')}"]

    def wrapper(pub, kern, doc, validate=None, msg=None):
        L = [f"def {pub}(x):", f'    """{doc}"""',
             "    cdef double[::1] f, o", "    cdef Py_ssize_t i",
             "    x_arr = np.asarray(x, dtype=np.float64)"]
        if validate:
            L += [f"    if np.any({validate}):", f"        raise ValueError({msg!r})"]
        L += ["    scalar = x_arr.ndim == 0",
              "    flat = np.ascontiguousarray(x_arr.ravel())",
              "    out = np.empty_like(flat)", "    f = flat", "    o = out",
              "    for i in range(f.shape[0]):", f"        o[i] = {kern}(f[i])",
              "    return float(out[0]) if scalar else out.reshape(x_arr.shape)"]
        return L

    wraps = (wrapper("millsratio", "_R", "Mills ratio R(x)=N(-x)/n(x); scalar or ndarray, any x.") + [""]
             + wrapper("millsratio_d1", "_R1", "-R'(x) = 1 - x R(x); scalar or ndarray, any x.") + [""]
             + wrapper("millsratio_d3", "_R3", "-R'''(x); scalar or ndarray, any x.") + [""]
             + wrapper("millsratio_rel_below1", "_Rrel_below1", "(sqrt(pi/2)-R(x))/x; scalar or ndarray, 0 <= x <= 1.",
                       "(x_arr < 0.0) | (x_arr > 1.0)", "millsratio_rel_below1 requires 0 <= x <= 1."))

    pyx = ["# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True",
           '"""Mills ratio R(x)=N(-x)/n(x) and -R\'(x), -R\'\'\'(x), plus (sqrt(pi/2)-R(x))/x on [0,1].',
           "Auto-generated by cheby_fit.emit_mathpf -- do not edit.  Coefficient data: _mills_coef.h.",
           'Segmented Chebyshev (flat bucket-major tables) + tiered Laplace continued fraction."""',
           "from libc.math cimport exp, INFINITY", "import numpy as np", "cimport numpy as np",
           "np.import_array()", "", *extern, "",
           "# -- C-level scalar kernels (cimport-able: from mathpf.mills cimport _R, _R1, _R3, _Rrel_below1) --",
           "\n".join(krrel), "", "\n".join(kR), "", "\n".join(kR1), "", "\n".join(kR3), "",
           "# -- Python-callable numpy-vectorized wrappers --", "\n".join(wraps), ""]

    os.makedirs(dest, exist_ok=True)
    for fn, lines in (("_mills_coef.h", h), ("mills.pxd", pxd), ("mills.pyx", pyx)):
        with open(os.path.join(dest, fn), "w") as fh:
            fh.write("\n".join(lines) + "\n")
        print(f"wrote {os.path.join(dest, fn)}")
    print(f"sizes: C_Rrel={nc_rrel}, C_R1={len(mc.C_R1)} ({mc.TMAX_R1}x{nc_r1}), "
          f"C_R3={len(mc.C_R3)} ({mc.TMAX_R3}x{nc_r3})")


if __name__ == "__main__":
    emit_module()
