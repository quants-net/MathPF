"""Cancellation analysis of the Taylor branch: R'''-seed vs R'-seed, at the
n=4 (m0=51.2) and n=3 (m0=99.6) corners on the r*=0.037 border.

For each scheme we report, in units of eps = 2^-52:
  trunc  : 4-term Taylor with EXACT (mpmath) odd derivatives vs the true ratio
           -> truncation only.
  total  : 4-term Taylor evaluated in float64 with the seeded recurrence
           -> truncation + cancellation.
  cancel : |total_float - trunc_exact| reduced to eps  -> isolated cancellation.
"""
import numpy as np
import mpmath as mp

mp.mp.dps = 80
EPS = 2.0 ** -52
RT = 0.037


def R_mp(x):
    return mp.sqrt(mp.pi / 2) * mp.erfc(x / mp.sqrt(2)) * mp.e ** (x * x / 2)


def derivs_mp(x, nmax):
    """R^(0..nmax) at x via ladder R^(n+1)=x R^(n)+n R^(n-1), R'=xR-1."""
    x = mp.mpf(x)
    R = [R_mp(x)]
    R.append(x * R[0] - 1)            # R'
    for n in range(1, nmax):
        R.append(x * R[n] + n * R[n - 1])
    return R


def true_ratio(m0, sig):
    a = mp.mpf(m0) - mp.mpf(sig) / 2
    b = mp.mpf(m0) + mp.mpf(sig) / 2
    return (R_mp(a) - R_mp(b)) / mp.mpf(sig)


def taylor_exact(m0, sig):
    """4-term Taylor with exact odd derivatives (truncation only)."""
    D = derivs_mp(m0, 7)
    s2 = mp.mpf(sig) ** 2
    Rx1 = -D[1]; Rx3 = -D[3]; Rx5 = -D[5]; Rx7 = -D[7]
    return Rx1 + s2 * (Rx3 / 24 + s2 * (Rx5 / 1920 + s2 * Rx7 / 322560))


def taylor_float_R3seed(m0, sig):
    """float64, seeded by accurate -R'''(m0), propagate down to Rx1, up to Rx5,Rx7."""
    m0 = np.float64(m0); sig = np.float64(sig); m0sq = m0 * m0
    Rx3 = np.float64(-float(derivs_mp(m0, 3)[3]))      # correctly-rounded seed
    Rx1 = (Rx3 + 1.0) / (m0sq + 3.0)
    Rx5 = (m0sq + 7.0) * Rx3 - 6.0 * Rx1
    Rx7 = (m0sq + 11.0) * Rx5 - 20.0 * Rx3
    s2 = sig * sig
    return Rx1 + s2 * (Rx3 / 24.0 + s2 * (Rx5 / 1920.0 + s2 * Rx7 / 322560.0))


def taylor_float_R1seed(m0, sig):
    """float64, seeded by accurate -R'(m0), climb up: Rx3,Rx5,Rx7 from Rx1."""
    m0 = np.float64(m0); sig = np.float64(sig); m0sq = m0 * m0
    Rx1 = np.float64(-float(derivs_mp(m0, 1)[1]))      # correctly-rounded seed
    Rx3 = (m0sq + 3.0) * Rx1 - 1.0                     # -R''' = (x^2+3)(-R') - 1
    Rx5 = (m0sq + 7.0) * Rx3 - 6.0 * Rx1
    Rx7 = (m0sq + 11.0) * Rx5 - 20.0 * Rx3
    s2 = sig * sig
    return Rx1 + s2 * (Rx3 / 24.0 + s2 * (Rx5 / 1920.0 + s2 * Rx7 / 322560.0))


print(f"{'m0':>7} {'logk':>8} {'sigma':>9} | "
      f"{'trunc':>7} | {'R3:total':>9} {'R3:cancel':>9} | "
      f"{'R1:total':>9} {'R1:cancel':>9}")
for m0 in (51.2, 70.0, 90.0, 99.6):
    sig = RT * (1.25 + m0)
    logk = m0 * sig
    truth = true_ratio(m0, sig)
    texact = taylor_exact(m0, sig)
    trunc = abs((texact - truth) / truth) / EPS

    f3 = mp.mpf(taylor_float_R3seed(m0, sig))
    tot3 = abs((f3 - truth) / truth) / EPS
    can3 = abs((f3 - texact) / truth) / EPS

    f1 = mp.mpf(taylor_float_R1seed(m0, sig))
    tot1 = abs((f1 - truth) / truth) / EPS
    can1 = abs((f1 - texact) / truth) / EPS

    print(f"{m0:7.1f} {float(logk):8.1f} {sig:9.4f} | "
          f"{float(trunc):7.1f} | {float(tot3):9.1f} {float(can3):9.1f} | "
          f"{float(tot1):9.1f} {float(can1):9.1f}")
