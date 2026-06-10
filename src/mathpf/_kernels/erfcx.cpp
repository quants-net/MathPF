/* Templated implementations of the scaled complementary error function
 * erfcx(z) = exp(z^2) * erfc(z) and its 1st and 3rd derivatives, expressed
 * via the Mills-ratio primitives in mills.h.  See erfcx.h for the API.
 *
 * Derivations (any sign of z; Mills primitives handle reflection internally):
 *
 *   Let x = z * sqrt(2);  d/dz [.] = sqrt(2) * d/dx [.].
 *
 *   erfcx(z)   = R(x) * sqrt(2/pi)
 *
 *   erfcx'(z)  = sqrt(2) * R'(x) * sqrt(2/pi)
 *              = sqrt(2) * (-R_1(x)) * sqrt(2/pi)
 *              = -R_1(x) * (2/sqrt(pi))
 *
 *   erfcx'''(z)= (sqrt(2))^3 * R'''(x) * sqrt(2/pi)
 *              = 2*sqrt(2) * R'''(x) * sqrt(2/pi)
 *              = R'''(x) * (4/sqrt(pi))
 *              = -R_3(x) * (4/sqrt(pi))                    [R_3 := -R''' by mathpf convention]
 *              = -R_3(x) * 2 * (2/sqrt(pi))
 *              = -R_3(x) * 2 * M_2_SQRT_PI
 */
#include "erfcx.h"
#include "mills.h"           /* MillsRatio, MillsRatioDeriv1, MillsRatioDeriv3 */
#include "MathConstants.h"   /* M_SQRT2, M_SQRT_2_PI, M_2_SQRT_PI */

namespace mathpf
{
    template<typename T>
    T Erfcx(T z)
    {
        return MillsRatio<T>(z * M_SQRT2<T>) * M_SQRT_2_PI<T>;
    }

    template<typename T>
    T ErfcxDeriv1(T z)
    {
        return -MillsRatioDeriv1<T>(z * M_SQRT2<T>) * M_2_SQRT_PI<T>;
    }

    template<typename T>
    T ErfcxDeriv3(T z)
    {
        return -MillsRatioDeriv3<T>(z * M_SQRT2<T>) * T(2) * M_2_SQRT_PI<T>;
    }

    /* Explicit float / double instantiations. */
    template double Erfcx<double>(double);
    template float  Erfcx<float>(float);
    template double ErfcxDeriv1<double>(double);
    template float  ErfcxDeriv1<float>(float);
    template double ErfcxDeriv3<double>(double);
    template float  ErfcxDeriv3<float>(float);

}  /* namespace mathpf */


/* --------------------------------------------------------------------------
 * extern "C" shims -- double-only.  Same wheel-internal-linkage rationale as
 * mills.cpp / mills_dd.cpp; see mills.h for the discussion.
 * -------------------------------------------------------------------------- */
extern "C" {

double mathpf_Erfcx(double z)         { return mathpf::Erfcx<double>(z); }
double mathpf_ErfcxDeriv1(double z)   { return mathpf::ErfcxDeriv1<double>(z); }
double mathpf_ErfcxDeriv3(double z)   { return mathpf::ErfcxDeriv3<double>(z); }

}  /* extern "C" */
