/* Shared math constants for the MathPF C++ kernels.
 *
 * Templated `inline constexpr` variable templates in `namespace mathpf`.
 * Each TU that #includes this header gets the constants at the precision
 * required for its T = {float, double} instantiations.
 *
 * Names follow the `<math.h>` `M_*` convention, scoped by `namespace mathpf`
 * so they cannot clash with any preprocessor `M_*` macros that platform
 * headers may define.
 *
 * 22 sigfigs each: round-trip-stable to the nearest fp64 regardless of the
 * platform's decimal-to-double parser.  Generated via mpmath at mp.dps = 30.
 */
#ifndef MATHPF_KERNELS_MATHCONSTANTS_H
#define MATHPF_KERNELS_MATHCONSTANTS_H

#include <cmath>
#include <limits>

namespace mathpf
{
    /* Used by mills.cpp (reflection branch) and the erfcx interface. */
    template<typename T> inline constexpr T M_SQRT2     = T(1.414213562373095048802);   /* sqrt(2)       */
    template<typename T> inline constexpr T M_SQRT2PI   = T(2.506628274631000502416);   /* sqrt(2 pi)    */
    template<typename T> inline constexpr T M_SQRT2PI_2 = T(1.253314137315500251208);   /* sqrt(pi / 2)  = R(0) */

    /* Used by the erfcx interface (erfcx via Mills):
     *     erfcx(z)   =  R(z * M_SQRT2)   * M_SQRT_2_PI    (= 2 / sqrt(2 pi))
     *     erfcx'(z)  = -R_1(z * M_SQRT2) * M_2_SQRT_PI
     *     erfcx'''(z)= -R_3(z * M_SQRT2) * 2 * M_2_SQRT_PI
     * Note M_SQRT_2_PI is the reciprocal of M_SQRT2PI_2; kept as its own
     * named constant so erfcx call sites read by intent rather than as a
     * reciprocal. */
    template<typename T> inline constexpr T M_SQRT_2_PI = T(0.7978845608028653558799);  /* sqrt(2 / pi) */
    template<typename T> inline constexpr T M_2_SQRT_PI = T(1.128379167095512573896);   /* 2 / sqrt(pi) */

    /* Type-dependent overflow thresholds for the reflection branch of Mills R.
     *   X_NEG_MAX = -sqrt(2 log T_max);  x < X_NEG_MAX -> exp(x^2/2) overflows.
     *   U_MAX     =  sqrt(T_max);        u > U_MAX     -> R_1(x)/R_3(x) underflow.
     * Both derive from std::numeric_limits<T>::max() so float instantiations
     * use FLT_MAX-derived bounds instead of DBL_MAX-derived ones.
     *
     * `inline const` (not constexpr) because std::sqrt and std::log are not
     * constexpr until C++23/26; the codebase targets C++17.  Initialised once
     * per T at static-init time; subsequent accesses are constant-fold-equivalent. */
    template<typename T>
    inline T const X_NEG_MAX = -std::sqrt(T(2) * std::log(std::numeric_limits<T>::max()));
    template<typename T>
    inline T const U_MAX     =  std::sqrt(std::numeric_limits<T>::max());
}

#endif
