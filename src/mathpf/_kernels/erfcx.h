/* Scaled complementary error function erfcx(z) = exp(z^2) * erfc(z) and its
 * derivatives w.r.t. z, expressed via the Mills-ratio primitives in mills.h.
 *
 * Identities (any sign of z; reflection handles z < 0 inside the Mills
 * primitives themselves):
 *     erfcx(z)   =  R(z * sqrt(2))   * sqrt(2 / pi)
 *     erfcx'(z)  = -R_1(z * sqrt(2)) * (2 / sqrt(pi))
 *     erfcx'''(z)= -R_3(z * sqrt(2)) * 2 * (2 / sqrt(pi))
 *
 * (The 2nd derivative is intentionally not exposed -- mathpf's Mills surface
 *  also skips R_2 since R_2(x) = R(x) - x*R_1(x) is a trivial composition;
 *  same here, erfcx''(z) = 2*erfcx(z) + 2z*erfcx'(z).)
 *
 * All three are cancellation-free thanks to the underlying Mills primitives
 * (MillsRatio uses tiered CF + segmented Chebyshev; MillsRatioDeriv1 is the
 * 1 - x R primitive; MillsRatioDeriv3 = -R'''(x) directly).
 *
 * Vendoring note: not part of MathPF's vendoring contract with downstream
 * pricer libraries.  erfcx is a standalone numerical surface for callers who
 * want a SciPy-style erfcx interface backed by mathpf's primitives.
 */
#ifndef MATHPF_KERNELS_ERFCX_H
#define MATHPF_KERNELS_ERFCX_H

namespace mathpf
{
    template<typename T> T Erfcx(T z);          /* exp(z^2) * erfc(z) */
    template<typename T> T ErfcxDeriv1(T z);    /* d/dz [erfcx(z)] */
    template<typename T> T ErfcxDeriv3(T z);    /* d^3/dz^3 [erfcx(z)] */
}

#ifdef __cplusplus
extern "C" {
#endif

/* Stable C ABI for Cython / ctypes / other-language bindings.  Double-only. */
double mathpf_Erfcx(double z);
double mathpf_ErfcxDeriv1(double z);
double mathpf_ErfcxDeriv3(double z);

#ifdef __cplusplus
}
#endif

#endif
