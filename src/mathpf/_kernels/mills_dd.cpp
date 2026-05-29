/* Templated symmetric divided differences of the Mills ratio.  Ported from the
 * original Cython prototype (src/mathpf/mills_dd.pyx); see mills_dd.h for the API.
 *
 * Calls into the templated math primitives declared in mills.h (MillsRatio,
 * MillsRatioDeriv3, MillsRatio_CF).  Bit-equality with the Cython prototype is
 * enforced by tests/test_pyref_consistency.py.
 */
#include "mills_dd.h"
#include "mills.h"          /* MillsRatio, MillsRatioDeriv3, MillsRatio_CF */
#include "_mills_coef.h"    /* XCF_R1, XCF_R3 */

QNSPACE
{
    /* ----------------------------------------------------------------------
     * MillsRatioDiff_CF(x, dx, n_terms) -- cancellation-free divided difference
     * via the coupled (V, T, P^a, P^b) recurrence on the depth-n CF convergent
     * of R.  Cancellation-free for any x > dx > 0.  Per iteration: 11 mults;
     * total ~ 11 n_terms + setup.
     *
     * With a = x - dx and b = x + dx:
     *   V_{k+1}      = a b V_k + k^2 V_{k-1} + k T_k
     *   T_{k+1}      = (a^2+b^2) V_k + 2k a b V_{k-1} + k(k-1) T_{k-1}
     *   P^[a]_{k+1}  = a P^[a]_k + k P^[a]_{k-1}    (similarly P^[b])
     * seeded by V_0 = 0, V_1 = 1, T_1 = -1, T_2 = a^2+b^2+1.  Final slope is
     * V_n / (P^[a]_n * P^[b]_n).  Setup quantities a*b and a^2+b^2 form from a, b
     * directly -- cancellation-safe at dx -> x by Sterbenz on (x - dx).
     * ---------------------------------------------------------------------- */
    template<typename T>
    T MillsRatioDiff_CF(T x, T dx, int n_terms)
    {
        T a = x - dx;
        T b = x + dx;
        if (n_terms == 0)
            return T(1.0) / (a * b);

        T ab = a * b;                       /* = x^2 - dx^2; Sterbenz-safe at dx -> x */
        T s2 = a * a + b * b;               /* = 2(x^2 + dx^2); one fewer mult than 2*(x*x + dx*dx) */
        T V_p  = T(1.0),  V_c  = ab - T(1.0);   /* V_1, V_2 */
        T T_p  = T(-1.0), T_c  = s2 + T(1.0);   /* T_1, T_2 */
        T Pa_p = a, Pa_c = a * a + T(1.0);      /* P^[a]_1, P^[a]_2 */
        T Pb_p = b, Pb_c = b * b + T(1.0);      /* P^[b]_1, P^[b]_2 */

        T V_n, T_n, Pa_n, Pb_n;
        for (int k = 2; k <= n_terms; ++k) {
            V_n  = ab * V_c + T(k) * (T(k) * V_p + T_c);
            T_n  = s2 * V_c + T(k) * (T(2.0) * ab * V_p + T(k - 1) * T_p);  /* 2 a b V_p inlined */
            Pa_n = a  * Pa_c + T(k) * Pa_p;
            Pb_n = b  * Pb_c + T(k) * Pb_p;
            V_p  = V_c;   V_c  = V_n;
            T_p  = T_c;   T_c  = T_n;
            Pa_p = Pa_c;  Pa_c = Pa_n;
            Pb_p = Pb_c;  Pb_c = Pb_n;
        }
        return V_c / (Pa_c * Pb_c);
    }

    /* ----------------------------------------------------------------------
     * MillsRatioDiff(x, dx, theta) -- the main dispatcher.
     *
     * theta = -1: sum branch (R(dx-x) + R(x+dx))/(2 dx) -- no cancellation, both
     *             evaluations land on positive arguments.
     *
     * theta = +1: difference branch -- three regimes by (x, dx) location:
     *   (1) dx >= 0.0392 (1.25 + x)                                : direct R difference
     *   (2) dx <  0.0392 (1.25 + x), a := x-dx >= 21.2 (= XCF_R1[2]): CF asymp ladder
     *   (3) dx <  0.0392 (1.25 + x), a < 21.2                       : 5-term R'''-seeded Taylor
     *
     * Overall worst relative error ~64 eps, set by the Taylor branch's truncation
     * at its gate corner.
     * ---------------------------------------------------------------------- */
    template<typename T>
    T MillsRatioDiff(T x, T dx, int theta)
    {
        if (theta < 0)                                                  /* sum branch */
            return (MillsRatio<T>(dx - x) + MillsRatio<T>(x + dx)) / (T(2.0) * dx);

        T a = x - dx;                                                   /* = -d1 (smaller Mills argument) */
        if (dx < T(3.92e-2) * (T(1.25) + x)) {                          /* unified small-dx gate */
            /* Asymp CF DD ladder, even-only n aligned with R_hat_1's CF cutoffs
             * XCF_R1[5..2] = 12800, 165, 41, 21.2.  Falls through to Taylor when
             * a < XCF_R1[2] = 21.2. */
            if      (a >= T(XCF_R1[5])) return MillsRatioDiff_CF<T>(x, dx, 2);  /* a >= 12800 */
            else if (a >= T(XCF_R1[4])) return MillsRatioDiff_CF<T>(x, dx, 4);  /* a >= 165 */
            else if (a >= T(XCF_R1[3])) return MillsRatioDiff_CF<T>(x, dx, 6);  /* a >= 41 */
            else if (a >= T(XCF_R1[2])) return MillsRatioDiff_CF<T>(x, dx, 8);  /* a in [21.2, 41) */

            /* a < 21.2: 5-term Taylor seeded by R''' (N=5 balance, ~38 eps at gate) */
            T u = x * x + T(3.0);                                       /* shared shifted variable */
            T r_d3;
            if (x >= T(XCF_R3[0]))                                      /* x in [17.1, ~22): CF n=12 directly */
                r_d3 = MillsRatio_CF<T>(u, 12, 3);
            else                                                        /* x < 17.1: segmented Chebyshev path */
                r_d3 = MillsRatioDeriv3<T>(x);
            T r_d1 = (r_d3 + T(1.0)) / u;                               /* descend: -R'(x), cancellation-free */
            T r_d5 = (u + T(4.0))  * r_d3 - T(6.0)  * r_d1;             /* ascend: r_{2k+1} = (x^2+4k-1) r_{2k-1} - ... */
            T r_d7 = (u + T(8.0))  * r_d5 - T(20.0) * r_d3;             /* (x^2 + 7/11/15) rewritten as (u + 4/8/12) */
            T r_d9 = (u + T(12.0)) * r_d7 - T(42.0) * r_d5;
            T dx2 = dx * dx;
            return r_d1 + dx2 / T(6.0) * (r_d3 + dx2 / T(20.0) * (r_d5 + dx2 / T(42.0) * (r_d7 + dx2 / T(72.0) * r_d9)));
        }
        /* dx >= gate*(1.25+x): direct difference (cancellation bounded) */
        return (MillsRatio<T>(x - dx) - MillsRatio<T>(x + dx)) / (T(2.0) * dx);
    }

    /* Explicit float / double instantiations. */
    template double MillsRatioDiff<double>(double, double, int);
    template float  MillsRatioDiff<float>(float, float, int);
    template double MillsRatioDiff_CF<double>(double, double, int);
    template float  MillsRatioDiff_CF<float>(float, float, int);

}  /* QNSPACE */


/* --------------------------------------------------------------------------
 * extern "C" shims -- double-only.  Same wheel-internal-linkage rationale as
 * mills.cpp; see mills.h for the discussion.
 * -------------------------------------------------------------------------- */
extern "C" {

double mathpf_MillsRatioDiff(double x, double dx, int theta) {
    return QuantsNet::MillsRatioDiff<double>(x, dx, theta);
}
double mathpf_MillsRatioDiff_CF(double x, double dx, int n_terms) {
    return QuantsNet::MillsRatioDiff_CF<double>(x, dx, n_terms);
}

}  /* extern "C" */
