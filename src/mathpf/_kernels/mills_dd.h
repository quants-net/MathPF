/* Symmetric divided differences of the Mills ratio.  Two surface functions:
 *
 *     MillsRatioDiff(x, dx, theta) = (R(x-dx) - theta*R(x+dx)) / (2 dx)
 *
 *       theta = +1: difference branch -- three-regime dispatch
 *                   (CF asymp / 5-term R'''-seeded Taylor / direct R difference)
 *                   keeps relative error bounded across the (x, dx) plane.
 *       theta = -1: sum branch -- R(dx - x) + R(x + dx)  (first arg reflected so
 *                   both R-evaluations land on positive arguments).
 *
 *     MillsRatioDiff_CF(x, dx, n_terms) = cancellation-free divided difference
 *                   via the coupled (V, T, P^a, P^b) recurrence on the depth-n
 *                   CF convergent of R.  Used by MillsRatioDiff's asymp band.
 *
 * Built on the templated Mills primitives in mills.h (MillsRatio, MillsRatioDeriv3,
 * MillsRatio_CF) -- include both this header and mills.h to get the full surface.
 * Coefficient data (XCF_R1, XCF_R3 thresholds) lives in _mills_coef.h.
 *
 * Templates live in `namespace mathpf`.  Vendoring this file into another
 * tree requires _mills_coef.h, mills.h, and mills.cpp to land in the same
 * directory.
 */
#ifndef MATHPF_KERNELS_MILLS_DD_H
#define MATHPF_KERNELS_MILLS_DD_H

namespace mathpf
{
    template<typename T> T MillsRatioDiff(T x, T dx, int theta);
    template<typename T> T MillsRatioDiff_CF(T x, T dx, int n_terms);
}

#ifdef __cplusplus
extern "C" {
#endif

/* Stable C ABI for Cython / ctypes / other-language bindings.  Double-only. */
double mathpf_MillsRatioDiff(double x, double dx, int theta);
double mathpf_MillsRatioDiff_CF(double x, double dx, int n_terms);

#ifdef __cplusplus
}
#endif

#endif
