"""Reference scalar implementations of the symmetric divided differences of the
Mills ratio R(x) = N(-x)/n(x).

Built on the segmented-Chebyshev / CF kernels in mills_cheby (mc.R, mc.R3); no scipy /
mpmath dependency.  For a midpoint x and half-step dx (so the two evaluation points
are x - dx and x + dx):

    R_DD(x, dx, theta = +1)   = (R(x - dx) - R(x + dx)) / (2 dx)   (difference branch)
    R_DD(x, dx, theta = -1)   = (R(dx - x) + R(x + dx)) / (2 dx)   (sum branch; first
                                argument reflected so both R-evaluations land on
                                positive arguments, no cancellation)
    R_DD_CF(x, dx, n_terms)   = cancellation-free DD via the (V, T, P^a, P^b) recurrence
                                on the depth-n_terms CF convergent of R.  Table-free,
                                algorithmic.  R_DD dispatches to this in the asymp band.
"""
from . import mills_cheby as mc


# Asymp-branch dispatch reads directly from mc.XCF_R1[2..5] (= 21.2, 41, 165, 12800), which
# are R_hat_1's CF convergent cutoffs at n = 8, 6, 4, 2 respectively (the CF rate
# (n+1)!/a^(2n) ~ eps at those a).  Walked top-down: largest threshold matches first (smallest
# n).  Even-only orders inherited from the CF tier structure.  The asymp/Taylor split sits at
# XCF_R1[2] = 21.2 (the n=8 cutoff); below that the Taylor branch absorbs the call, since its
# R'''-seeded cancellation amplification (~delta^8 x^4 / 60480) stays below eps across a < 21.2.
_R3_XCF          = mc.XCF_R3[0]                       # 17.1 -- smallest x where mc.R3's CF n=12 convergent
                                                      # is at eps; Taylor branch's upper sub-range bypasses
                                                      # mc.R3's dispatch and calls R013_CF(u, 12, 3) directly


def R_DD(x, dx, theta=1):
    """Mills-ratio symmetric divided difference about x with half-step dx:

        theta = +1:  R_DD = (R(x - dx) - R(x + dx)) / (2 dx)  (difference branch)
        theta = -1:  R_DD = (R(dx - x) + R(x + dx)) / (2 dx)  (sum branch; first argument
                     reflected so both R-evaluations land on positive arguments even when
                     x < 0, avoiding cancellation in the literal difference)

    theta = -1 is a sum of Mills ratios, no cancellation.
    For theta = +1, the difference [R(x - dx) - R(x + dx)] / (2 dx) is split into
    three regimes to keep relative error near eps:
      - dx < 0.0392 (1.25 + x) AND a = x - dx >= mc.XCF_R1[2] = 21.2 (deep asymptotic, small dx):
        cancellation-free divided difference via the (V, T, P^a, P^b) CF recurrence
        (R_DD_CF), with order n stepped down through the XCF_R1 ladder: n = 8 for
        a >= XCF_R1[2] (21.2), 6 for a >= XCF_R1[3] (41), 4 for a >= XCF_R1[4] (165),
        2 for a >= XCF_R1[5] (12800).  The Taylor branch absorbs everything below,
        since its R'''-seeded cancellation amplification (~delta^8 x^4 / 60480) stays
        below eps across a < 21.2 (max Taylor error ~64 eps, set by truncation at
        the gate corner).
      - else small dx (dx < 0.0392 (1.25 + x), a < 21.2): 5-term Taylor in dx in the odd
        derivatives -R^(2k+1)(x), seeded by the central -R'''(x) = mc.R3(x) and
        propagated DOWN to -R'(x) = (r_d3 + 1) / (x^2 + 3) (cancellation-free,
        odd -> odd) and UP to -R^(5), -R^(7), -R^(9) via the Hermite-like recurrence
        r_{2k+1} = (x^2 + 4k-1) r_{2k-1} - (2k-1)(2k-2) r_{2k-3}.  Seeding from R'''
        (vs. climbing from -R') keeps the upward cancellation ~x^4 smaller, so it
        stays accurate for all x.  The gate dx < 0.0392(1.25 + x) sits at the
        n=5 truncation/direct-cancellation balance ((dx/x)^(2n) = N eps x/(2dx) with
        N=3), giving a balanced error of ~38 eps at the boundary.
      - else dx >= 0.0392 (1.25 + x): the direct mc.R difference.  Cancellation is
        bounded by m >= m*, giving ~eps/m (worst ~38 eps at the gate boundary).
        For very large dx/x the cancellation is benign anyway: R(x-dx)/R(x+dx) > 19
        once dx/x >= 0.9, so the subtraction loses < 1 bit.
    """
    if theta < 0:                                        # above: sum of Mills ratios, no cancellation
        return (mc.R(dx - x) + mc.R(x + dx)) / (2.0*dx)
    # below (theta = +1): difference [R(x - dx) - R(x + dx)] / (2 dx)
    a = x - dx                                            # = -d1 (smaller Mills argument)
    if dx < 3.92e-2*(1.25 + x):                          # unified small-dx gate: asymp ladder or Taylor by a
        # Asymp CF DD ladder, even-only n aligned with R_hat_1's CF convergent rate (n+1)!/a^(2n) ~ eps:
        # thresholds are exactly R_hat_1's CF cutoffs at n = 2, 4, 6, 8 (mc.XCF_R1[5..2]).  Falls
        # through to Taylor when a < mc.XCF_R1[2] (= 21.2).
        if   a >= mc.XCF_R1[5]: return R_DD_CF(x, dx, mc.NCF_R1[5]) # a >= 12800:          CF n=2
        elif a >= mc.XCF_R1[4]: return R_DD_CF(x, dx, mc.NCF_R1[4]) # a in [165, 12800):   CF n=4
        elif a >= mc.XCF_R1[3]: return R_DD_CF(x, dx, mc.NCF_R1[3]) # a in [41, 165):      CF n=6
        elif a >= mc.XCF_R1[2]: return R_DD_CF(x, dx, mc.NCF_R1[2]) # a in [21.2, 41):     CF n=8 bottom row
        # a < mc.XCF_R1[2] (21.2): 5-term Taylor seeded by R''' (N=5 balance, ~38 eps at gate)
        u = x*x + 3.0                                   # kernel's shifted variable; shared with R013_CF, descent, and ascent
        if x >= _R3_XCF:                                # x in [17.1, ~22): mc.R3's CF n=12 path; call directly with our u
            r_d3 = mc.R013_CF(u, mc.NCF_R3[0], 3)        # skips mc.R3's sign/U_MAX/ladder dispatch + redundant x*x + 3
        else:                                           # x < 17.1: segmented Chebyshev path inside mc.R3
            r_d3 = mc.R3(x)                             # = -R'''(x)
        r_d1 = (r_d3 + 1.0) / u                        # descend: -R'(x), cancellation-free (odd -> odd); u = x^2 + 3
        r_d5 = (u +  4.0)*r_d3 -  6.0*r_d1             # ascend from accurate R''': r_{2k+1} = (x^2+4k-1) r_{2k-1} - (2k-1)(2k-2) r_{2k-3};
        r_d7 = (u +  8.0)*r_d5 - 20.0*r_d3             # (x^2 + 7/11/15) rewritten as (u + 4/8/12) since u = x^2 + 3
        r_d9 = (u + 12.0)*r_d7 - 42.0*r_d5
        dx2 = dx*dx                                       # Taylor in dx; nested with consecutive denominators (2k+1)(2k+2)
        return r_d1 + dx2/6.0*(r_d3 + dx2/20.0*(r_d5 + dx2/42.0*(r_d7 + dx2/72.0*r_d9)))
    # dx >= gate*(1.25+x): direct difference (cancellation bounded by m >= m*, ~eps/m)
    return (mc.R(x - dx) - mc.R(x + dx)) / (2.0*dx)


def R_DD_CF(x, dx, n_terms):
    """Cancellation-free divided difference (R(x - dx) - R(x + dx)) / (2 dx) via the
    coupled (V, T) recurrence on the depth-(n_terms+1) CF convergent of R.

    With a = x - dx and b = x + dx, the recurrence is

        V_{k+1} = (a b) V_k  + k^2 V_{k-1}  + k T_k
        T_{k+1} = (a^2+b^2) V_k + 2 k (a b) V_{k-1} + k (k-1) T_{k-1}
        P^[a]_{k+1} = a P^[a]_k + k P^[a]_{k-1},   similarly for P^[b]

    seeded by V_0 = 0, V_1 = 1, T_1 = -1, T_2 = a^2 + b^2 + 1.  The final slope is
    V_n / (P^[a]_n P^[b]_n).  Zero divisions inside the loop; one final divide.
    Cancellation-free for any (x, dx) with x > dx > 0 (so a, b both positive).

    The setup quantities ab = a*b and s2 = a^2 + b^2 reuse the already-computed
    a, b -- one fewer mult than 2*(x^2+dx^2) for s2, and cancellation-safe at
    dx -> x by Sterbenz on (x - dx) (the explicit x^2 - dx^2 form would lose
    precision proportional to 1/(1 - (dx/x)^2)).  Useful in the regime where
    R_DD's direct mc.R difference cancels and Taylor is not enough -- the natural
    bridge between the Taylor band and the asymp band on a compiled scalar path
    where div >> mul.
    """
    a, b = x - dx, x + dx
    if n_terms == 0:
        return 1.0/(a*b)
    ab  = a * b                                     # = x^2 - dx^2; Sterbenz-safe at dx -> x via the exact (x - dx)
    s2  = a*a + b*b                                 # = 2*(x^2 + dx^2); one fewer mult than 2.0*(x*x + dx*dx)
    V_p,  V_c  = 1.0, ab - 1.0                      # V_1, V_2
    T_p,  T_c  = -1.0, s2 + 1.0                     # T_1, T_2
    Pa_p, Pa_c = a, a*a + 1.0                       # P^[a]_1, P^[a]_2
    Pb_p, Pb_c = b, b*b + 1.0                       # P^[b]_1, P^[b]_2
    for k in range(2, n_terms + 1):
        V_n  = ab * V_c + k * (k * V_p + T_c)
        T_n  = s2 * V_c + k * (2.0 * ab * V_p + (k - 1) * T_p)  # 2 a b V_p inlined (loop-invariant; hoisted by compiler)
        Pa_n = a * Pa_c + k * Pa_p
        Pb_n = b * Pb_c + k * Pb_p
        V_p,  V_c  = V_c,  V_n
        T_p,  T_c  = T_c,  T_n
        Pa_p, Pa_c = Pa_c, Pa_n
        Pb_p, Pb_c = Pb_c, Pb_n
    return V_c/(Pa_c*Pb_c)
