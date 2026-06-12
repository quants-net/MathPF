"""Pure-Python ports of Jaeckel's two file-scope expansions for the
normalised Black price (over vega), used by the LBR comparison demo
(``mathpf.demos.lbr_vs_mathpf``).

These are direct transcriptions of the two static functions

    asymptotic_expansion_of_normalised_black_call_over_vega(h, t)
    small_t_expansion_of_normalised_black_call_over_vega   (h, t)

inside Jaeckel's ``lets_be_rational.cpp`` (Jaeckel 2013-2023, "Let's Be
Rational").  They live here -- and not inside any compiled mathpf
kernel -- because their purpose is documentary: they let a reader run
the comparison against mathpf's ``R_DD`` without installing or building
Jaeckel's C++ library.

Math correspondence with mathpf's symmetric divided difference of the
Mills ratio::

    b/vega = R(-h-t) - R(-h+t) = 2 * t * R_DD(-h, t, theta=+1)

so each Jaeckel expansion is the algorithmic counterpart of one R_DD
branch:

    aexp_over_vega(h, t) / (2 t) == R_DD asymp branch  (at -h, t)
    stexp_over_vega(h, t) / (2 t) == R_DD Taylor branch (at -h, t)

The erfcx call inside ``stexp_over_vega`` uses ``mathpf.erfcx`` (the
segmented-Chebyshev kernel built on mathpf's Mills primitives) rather
than Jaeckel's ``erfcx_cody``; the two agree to a handful of ulps, and
the cancellation in ``a := 1 + h * Y(h)`` -- which is the documented
small-t bottleneck for ``|h| > 1`` -- is independent of which erfcx is
used.  See ``mathpf.demos.lbr_vs_mathpf`` for the side-by-side accuracy
sweep against an mpmath truth.
"""
from __future__ import annotations

import math
import sys

import mathpf


# Jaeckel's hard-coded dispatch thresholds (file-scope constants in his
# C++).  Exposed here so callers can reproduce his dispatch geometry.
ETA = -10.0
"""Asymptotic-expansion dispatch threshold: ``h < ETA``.

Jaeckel's dispatcher uses the 17th-order asymptotic series below
``h = -10`` (with the additional gate ``h + t < ETA + TAU ~= -9.79``).
"""

TAU = 2.0 * sys.float_info.epsilon ** (1.0 / 16.0)
"""Small-t-expansion dispatch threshold: ``t < TAU`` (~= 0.2114).

Jaeckel's dispatcher uses the 12th-order Taylor in ``t`` below this.
The literal value is ``2 * eps^(1/16)``.
"""


# Constants used by stexp_over_vega.  Hardcoded to the correctly-rounded
# fp64 values (math.sqrt would be 1 ulp off for SQRT_PI_OVER_TWO).
_SQRT_PI_OVER_TWO = 1.2533141373155003   # sqrt(pi / 2) = R(0)
_SQRT_TWO         = 1.4142135623730951   # sqrt(2)


def aexp_over_vega(h: float, t: float) -> float:
    """Jaeckel's 17th-order asymptotic expansion of b/vega in (h, t).

    Pure-Python transcription of
    ``asymptotic_expansion_of_normalised_black_call_over_vega(h, t)``
    inside Jaeckel's ``lets_be_rational.cpp``.

    Jaeckel's stated validity gate is::

        h < ETA (= -10)   AND   h + t < ETA + TAU (~= -9.79),

    inside which the relative accuracy is claimed to be 1.64e-16.
    Outside the gate the series is divergent and amplification by
    cancellation grows fast -- the comparison demo deliberately probes
    outside the gate to show the degradation.

    Math correspondence::

        b/vega = R(-h-t) - R(-h+t) = 2 t * R_DD(-h, t, theta=+1)

    so this returns ``2 * t * R_DD_asymp(-h, t)``.

    Notes
    -----
    The kernel is a single hardcoded polynomial in three quantities
    derived from ``(h, t)``: ``e = (t/h)**2``, ``r = (h+t)(h-t)``, and
    ``q = (h/r)**2``.  Inside the gate, ``q`` stays below ~0.01, so the
    17th-order truncation error ``~ q**17`` is ~1e-34, far below eps.
    """
    e = (t / h) ** 2
    r = (h + t) * (h - t)
    q = (h / r) ** 2
    # 17th-order asymptotic expansion of A(h,t) in q, transcribed
    # verbatim from Jaeckel's C++ source.  The (--very long--) sum is
    # Horner-nested in q and inside each q-level Horner-nested in e.
    s = (2.0 + q * (-6.0e0 - 2.0 * e + 3.0 * q * (1.0e1 + e * (2.0e1 + 2.0 * e) + 5.0 * q * (-1.4e1 + e * (-7.0e1 + e * (-4.2e1 - 2.0 * e)) + 7.0 * q * (1.8e1 + e * (1.68e2 + e * (2.52e2 + e * (7.2e1 + 2.0 * e))) + 9.0 * q * (-2.2e1 + e * (-3.3e2 + e * (-9.24e2 + e * (-6.6e2 + e * (-1.1e2 - 2.0 * e)))) + 1.1e1 * q * (2.6e1 + e * (5.72e2 + e * (2.574e3 + e * (3.432e3 + e * (1.43e3 + e * (1.56e2 + 2.0 * e))))) + 1.3e1 * q * (-3.0e1 + e * (-9.1e2 + e * (-6.006e3 + e * (-1.287e4 + e * (-1.001e4 + e * (-2.73e3 + e * (-2.1e2 - 2.0 * e)))))) + 1.5e1 * q * (3.4e1 + e * (1.36e3 + e * (1.2376e4 + e * (3.8896e4 + e * (4.862e4 + e * (2.4752e4 + e * (4.76e3 + e * (2.72e2 + 2.0 * e))))))) + 1.7e1 * q * (-3.8e1 + e * (-1.938e3 + e * (-2.3256e4 + e * (-1.00776e5 + e * (-1.84756e5 + e * (-1.51164e5 + e * (-5.4264e4 + e * (-7.752e3 + e * (-3.42e2 - 2.0 * e)))))))) + 1.9e1 * q * (4.2e1 + e * (2.66e3 + e * (4.0698e4 + e * (2.3256e5 + e * (5.8786e5 + e * (7.05432e5 + e * (4.0698e5 + e * (1.08528e5 + e * (1.197e4 + e * (4.2e2 + 2.0 * e))))))))) + 2.1e1 * q * (-4.6e1 + e * (-3.542e3 + e * (-6.7298e4 + e * (-4.90314e5 + e * (-1.63438e6 + e * (-2.704156e6 + e * (-2.288132e6 + e * (-9.80628e5 + e * (-2.01894e5 + e * (-1.771e4 + e * (-5.06e2 - 2.0 * e)))))))))) + 2.3e1 * q * (5.0e1 + e * (4.6e3 + e * (1.0626e5 + e * (9.614e5 + e * (4.08595e6 + e * (8.9148e6 + e * (1.04006e7 + e * (6.53752e6 + e * (2.16315e6 + e * (3.542e5 + e * (2.53e4 + e * (6.0e2 + 2.0 * e))))))))))) + 2.5e1 * q * (-5.4e1 + e * (-5.85e3 + e * (-1.6146e5 + e * (-1.77606e6 + e * (-9.37365e6 + e * (-2.607579e7 + e * (-4.01166e7 + e * (-3.476772e7 + e * (-1.687257e7 + e * (-4.44015e6 + e * (-5.9202e5 + e * (-3.51e4 + e * (-7.02e2 - 2.0 * e)))))))))))) + 2.7e1 * q * (5.8e1 + e * (7.308e3 + e * (2.3751e5 + e * (3.12156e6 + e * (2.003001e7 + e * (6.919458e7 + e * (1.3572783e8 + e * (1.5511752e8 + e * (1.0379187e8 + e * (4.006002e7 + e * (8.58429e6 + e * (9.5004e5 + e * (4.7502e4 + e * (8.12e2 + 2.0 * e))))))))))))) + 2.9e1 * q * (-6.2e1 + e * (-8.99e3 + e * (-3.39822e5 + e * (-5.25915e6 + e * (-4.032015e7 + e * (-1.6934463e8 + e * (-4.1250615e8 + e * (-6.0108039e8 + e * (-5.3036505e8 + e * (-2.8224105e8 + e * (-8.870433e7 + e * (-1.577745e7 + e * (-1.472562e6 + e * (-6.293e4 + e * (-9.3e2 - 2.0 * e)))))))))))))) + 3.1e1 * q * (6.6e1 + e * (1.0912e4 + e * (4.74672e5 + e * (8.544096e6 + e * (7.71342e7 + e * (3.8707344e8 + e * (1.14633288e9 + e * (2.07431664e9 + e * (2.33360622e9 + e * (1.6376184e9 + e * (7.0963464e8 + e * (1.8512208e8 + e * (2.7768312e7 + e * (2.215136e6 + e * (8.184e4 + e * (1.056e3 + 2.0 * e))))))))))))))) + 3.3e1 * (-7.0e1 + e * (-1.309e4 + e * (-6.49264e5 + e * (-1.344904e7 + e * (-1.4121492e8 + e * (-8.344518e8 + e * (-2.9526756e9 + e * (-6.49588632e9 + e * (-9.0751353e9 + e * (-8.1198579e9 + e * (-4.6399188e9 + e * (-1.6689036e9 + e * (-3.67158792e8 + e * (-4.707164e7 + e * (-3.24632e6 + e * (-1.0472e5 + e * (-1.19e3 - 2.0 * e))))))))))))))))) * q)))))))))))))))))
    b_over_vega = (t / r) * s
    return abs(max(b_over_vega, 0.0))


def stexp_over_vega(h: float, t: float) -> float:
    """Jaeckel's 12th-order small-t expansion of b/vega in (h, t).

    Pure-Python transcription of
    ``small_t_expansion_of_normalised_black_call_over_vega(h, t)``
    inside Jaeckel's ``lets_be_rational.cpp``.

    Jaeckel's stated validity gate is::

        h <= 0   AND   t < TAU (~= 0.211),

    inside which the relative accuracy is claimed to be ~ eps.  The
    bottleneck for ``|h| > 1`` is the coefficient

        a := 1 + h * Y(h),    Y(h) := Phi(h)/phi(h)

    which suffers catastrophic cancellation as ``h -> -infinity``
    (``Y(h) -> -1/h`` so ``1 + h*Y(h) -> 0`` with vanishing leading
    digits).  At ``h = -10``, ``|a| ~ 0.01`` -- two decimal digits are
    killed before the series even runs.

    Math correspondence: see ``aexp_over_vega``; this returns
    ``2 * t * R_DD_taylor(-h, t)``.

    Notes
    -----
    The erfcx call inside ``a := 1 + h*Y(h)`` here uses ``mathpf.erfcx``
    (the segmented-Chebyshev kernel) rather than Jaeckel's ``erfcx_cody``.
    The two implementations agree to a handful of ulps; the cancellation
    in ``a`` is independent of which erfcx is plugged in.
    """
    # Y(h) := Phi(h)/phi(h) = sqrt(pi/2) * erfcx(-h/sqrt2)
    a = 1.0 + h * _SQRT_PI_OVER_TWO * float(mathpf.erfcx(-h / _SQRT_TWO))
    w = t * t
    h2 = h * h
    # 12th-order Taylor expansion in w = t^2 of (Y(h+t) - Y(h-t))/(2t),
    # Horner-nested in w with each level Horner-nested in h2.  Transcribed
    # verbatim from Jaeckel's C++ source.
    b_over_vega = 2 * t * (a + w * ((-1 + 3 * a + a * h2) / 6 + w * ((-7 + 15 * a + h2 * (-1 + 10 * a + a * h2)) / 120 + w * ((-57 + 105 * a + h2 * (-18 + 105 * a + h2 * (-1 + 21 * a + a * h2))) / 5040 + w * ((-561 + 945 * a + h2 * (-285 + 1260 * a + h2 * (-33 + 378 * a + h2 * (-1 + 36 * a + a * h2)))) / 362880 + w * ((-6555 + 10395 * a + h2 * (-4680 + 17325 * a + h2 * (-840 + 6930 * a + h2 * (-52 + 990 * a + h2 * (-1 + 55 * a + a * h2))))) / 39916800 + ((-89055 + 135135 * a + h2 * (-82845 + 270270 * a + h2 * (-20370 + 135135 * a + h2 * (-1926 + 25740 * a + h2 * (-75 + 2145 * a + h2 * (-1 + 78 * a + a * h2)))))) * w) / 6227020800.0))))))
    return abs(max(b_over_vega, 0.0))
