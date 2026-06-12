"""Compare Jaeckel's two normalised-Black expansions against mathpf's R_DD,
restricted to the (h, t) cells where Jaeckel's dispatcher actually calls
each expansion.

Background
----------
LBR (Jaeckel 2013-2023) ships two file-scope expansions for b/vega:

  asymptotic_expansion_of_normalised_black_call_over_vega(h, t)
      17th-order asymp series.  Dispatch gate inside Jaeckel's
      normalised_black_call:    h < ETA = -10   AND   h + t < ETA + TAU.

  small_t_expansion_of_normalised_black_call_over_vega(h, t)
      12th-order Taylor in t (variable w := t**2).  Dispatch gate:
      (AEXP did NOT fire)   AND   t < TAU ~= 0.211 .

After the dispatcher's x > 0 reflection, h <= 0 below the expansions.  The
AEXP exclusion narrows STEXP's effective region to {-10 <= h <= 0, t < TAU}.

mathpf's symmetric divided difference of the Mills ratio,

  R_DD(x, dx, +1) := [R(x - dx) - R(x + dx)] / (2 dx) ,

is the algorithmic counterpart, related by

  b/vega = R(-h-t) - R(-h+t) = 2 t * R_DD(-h, t, +1) .

R_DD dispatches across an even-order CF asymp ladder (n in {2, 4, 6, 8}
keyed off XCF_R1 thresholds 12800, 165, 41, 21.2), a 5-term Taylor seeded
by R'''(x) for small dx with a < 21.2, and the direct mc.R difference where
cancellation is benign.

Findings
--------
AEXP inside its dispatch gate:  q = (h/((h+t)(h-t)))**2 is bounded by ~0.01
throughout the gate (the |h+t| > 9.79 floor cuts off the divergence corner),
so the 17th-order truncation is ~q**17 ~ 1e-34 -- vastly below eps.  LBR is
at full double precision there; mathpf's R_DD lands in its Taylor branch
near the gate-boundary cell (-14.2, 0.605) at ~1.4e-14 (~64 eps), which is
the documented N=5 Taylor truncation cost.

STEXP inside its dispatch region:  worst LBR cell is at the h = -10 corner,
where Jaeckel's coefficient

  a := 1 + h * Y(h)        with  Y(h) := Phi(h)/phi(h)

suffers cancellation -- Y(h) -> -1/h as h -> -inf, so |a| ~ 1/h**2 (~0.01
at h = -10, ~ 2 decimal digits killed before the series even runs).  Worst
single cell at (h, t) = (-9.4, 0.17): LBR 4.28e-14 (~192 eps) vs mathpf
bit-exact in double.  mathpf's R_DD avoids the 1 + h*Y(h) coefficient
entirely -- the Taylor branch is seeded by R'''(x) and descended to R'(x)
via the cancellation-free (r_d3 + 1) / (x**2 + 3) relation.

Run
---
    python -m mathpf.demos.lbr_vs_mathpf

Requires mpmath (truth side):  pip install mathpf[dev]
"""
from __future__ import annotations

from mathpf import jaeckel as L

try:
    import mpmath as mp
except ImportError as e:                                  # pragma: no cover
    raise SystemExit(
        "This demo needs mpmath for the high-precision truth side.\n"
        "Install with:  pip install mathpf[dev]"
    ) from e

# mathpf ships a pure-Python reference scalar that mirrors the compiled
# kernel; we use it here to compare R_DD's value at exactly the points the
# compiled mathpf.mills_dd path would evaluate.
from mathpf._pyref import mills_dd as mdd

mp.mp.dps = 80


def _R_mp(z) -> mp.mpf:
    """Mills ratio R(z) = sqrt(pi/2) * erfcx(z/sqrt2) at mpmath precision.

    Uses mp.erfc instead of mp.ncdf because mp.ncdf carries an internal
    precision cap that bites at the 1e-12 level around |z| ~ 10 (a footgun
    we hit once while writing this demo)."""
    z = mp.mpf(z)
    return mp.sqrt(mp.pi / 2) * mp.erfc(z / mp.sqrt(2)) * mp.exp(z * z / 2)


def truth_b_over_vega(h: float, t: float) -> float:
    """High-precision b/vega = R(-h-t) - R(-h+t).

    NOTE: convert h, t to mpmath BEFORE doing -h-t / -h+t.  Computing them
    in double first and then wrapping shifts the inputs by ~5e-15 vs what
    the algorithms actually see, surfacing as a spurious ~3e-12 relerr in
    the truth at deep-OTM cells."""
    H, T = mp.mpf(h), mp.mpf(t)
    return float(_R_mp(-H - T) - _R_mp(-H + T))


def _Y(h: float) -> float:
    """Y(h) = Phi(h)/phi(h) = sqrt(pi/2) * erfcx(-h/sqrt2).  For computing
    Jaeckel's cancellation-prone coefficient a := 1 + h * Y(h)."""
    return float(
        mp.sqrt(mp.pi / 2) * mp.erfc(-mp.mpf(h) / mp.sqrt(2)) * mp.exp(mp.mpf(h) ** 2 / 2)
    )


def relerr(approx: float, truth: float) -> float:
    if truth == 0.0:
        return 0.0 if approx == 0.0 else float("inf")
    return abs(approx - truth) / abs(truth)


def _fmt(re: float) -> str:
    if re == 0.0:
        return "    0    "
    return f"{re:>9.1e}"


def section_aexp() -> None:
    print("=" * 78)
    print("AEXP inside its dispatch gate: h < ETA = -10  AND  h + t < ETA + TAU ~= -9.79")
    print("=" * 78)
    print("Inside this gate q = (h/((h+t)(h-t)))^2 <= ~0.01 (the |h+t| > 9.79 floor")
    print("blocks the divergence corner), so the 17th-order series truncation is")
    print("~q^17 ~ 1e-34 -- vastly below eps.  Walking the gate boundary:")
    print()
    print(f"  {'h':>8}  {'t':>7}  {'q':>10}  {'LBR aexp re':>12}  {'mathpf re':>11}")
    for h in (-10.001, -10.01, -10.1, -10.5, -11.0, -15.0, -20.0, -30.0):
        t = (-9.79) - h - 1e-4
        if t <= 0:
            continue
        r = (h + t) * (h - t)
        q = (h / r) ** 2
        tr = truth_b_over_vega(h, t)
        v = L.aexp_over_vega(h, t)
        rd = 2.0 * t * mdd.R_DD(-h, t, 1)
        print(
            f"  {h:>8.3f}  {t:>7.3f}  {q:>10.3e}  "
            f"{_fmt(relerr(v, tr))}     {_fmt(relerr(rd, tr))}"
        )
    print()
    print("Verdict: AEXP is at full double precision throughout its dispatch gate.")
    print("mathpf's worst point inside the gate sits in its Taylor branch at")
    print("(-14.2, 0.605) ~ 1.4e-14 (~64 eps); see mathpf.demos.lbr_vs_mathpf docstring.")
    print()


def section_stexp() -> None:
    print("=" * 78)
    print(f"STEXP inside its dispatch region: -10 <= h <= 0  AND  0 < t < TAU = {L.TAU:.4f}")
    print("=" * 78)
    print("Jaeckel: 'the main bottleneck for precision is the coefficient")
    print("a := 1 + h * Y(h) when |h| > 1'.  Y(h) -> -1/h as h -> -inf, so |a|")
    print("collapses at rate 1/h^2 -- at h=-10, |a| ~ 0.01 (2 digits gone before")
    print("the series even begins).  mathpf's R_DD seeds its Taylor branch from")
    print("R'''(x), so no  1 + h*Y(h)  cancellation appears.")
    print()
    print(f"  {'h':>8}  {'t':>7}  {'|a|':>10}  {'LBR stexp re':>12}  {'mathpf re':>11}")
    for h in (-2.0, -5.0, -8.0, -9.0, -9.4, -9.9, -10.0):
        t = 0.20
        a = 1.0 + h * _Y(h)
        tr = truth_b_over_vega(h, t)
        v = L.stexp_over_vega(h, t)
        rd = 2.0 * t * mdd.R_DD(-h, t, 1)
        print(
            f"  {h:>8.3f}  {t:>7.3f}  {a:>10.2e}  "
            f"{_fmt(relerr(v, tr))}     {_fmt(relerr(rd, tr))}"
        )
    print()
    print("Worst LBR cells found by 100x21 grid scan over the STEXP region:")
    print(f"  {'h':>8}  {'t':>7}  {'LBR stexp re':>12}  {'mathpf re':>11}  {'LBR/mathpf':>11}")
    grid = []
    for hi in range(-100, 1):
        h = hi / 10.0
        for ti in range(1, 21):
            t = ti * 0.01
            tr = truth_b_over_vega(h, t)
            if tr == 0:
                continue
            v = L.stexp_over_vega(h, t)
            rd = 2.0 * t * mdd.R_DD(-h, t, 1)
            re_l = relerr(v, tr)
            re_m = relerr(rd, tr)
            if re_l < 1e-15:
                continue
            ratio_str = "inf" if re_m == 0 else f"{re_l / re_m:>8.1f}x"
            grid.append((re_l, h, t, re_m, ratio_str))
    grid.sort(reverse=True)
    for re_l, h, t, re_m, ratio in grid[:8]:
        print(
            f"  {h:>8.2f}  {t:>7.3f}  {_fmt(re_l)}     {_fmt(re_m)}    {ratio:>11}"
        )
    print()


def section_summary() -> None:
    print("=" * 78)
    print("Summary")
    print("=" * 78)
    print(
        """
AEXP inside its dispatch gate:  full double precision (~eps).  No interesting
  failure -- q <= ~0.01 inside the gate makes the 17th-order series error
  ~1e-34.  LBR matches mathpf at the limit of the truth side.

STEXP inside its dispatch region:  worst at the h = -10 corner, ~ 200 eps
  loss (cell (-9.4, 0.17): LBR ~ 4.3e-14, mathpf ~ 0).  Cause: cancellation
  in a := 1 + h * Y(h) -- Jaeckel calls this out in his source comments.
  mathpf's R_DD avoids the coefficient entirely (seeds from R'''(x), then
  descends to R'(x) via the cancellation-free (r_d3 + 1)/(x^2 + 3)).
"""
    )


def main() -> int:
    section_aexp()
    section_stexp()
    section_summary()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
