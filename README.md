# MathPF

**Numerically-precise special functions for quantitative finance and statistics**, written as templated C++17 kernels with thin Cython wrappers for Python.

The name **MathPF** stands for **Math for [PyFENG](https://github.com/PyFE/PyFENG)** — the numerical-primitives layer extracted from the [PyFENG](https://github.com/PyFE/PyFENG) (Python Financial ENGineering) ecosystem so it can be used standalone or vendored into any C++ project.

The core surface centres on the **Mills ratio** family `R(x) = N(-x)/n(x)` — the cancellation-free primitive that sits underneath Black-Scholes and Bachelier option pricing, normal-tail probabilities, hazard rates, and many Bayesian-statistics expressions — together with a **scaled complementary error function** `erfcx(z) = exp(z²) erfc(z)` interface routed through the same machinery.

The implementation pairs **tiered continued-fraction asymptotics** for large arguments with **segmented Chebyshev interpolation** on `[0, 1]` and `[1, ~9.5]`, giving worst-case accuracy of **~1–3 ulps** on the real axis at native FP64 cost.  The numerical design is inspired by Steven G. Johnson's [Faddeeva package](http://ab-initio.mit.edu/wiki/index.php/Faddeeva_Package), which powers SciPy's and libcerf's `erfcx`; MathPF restricts attention to the real axis and exposes a Mills-flavoured surface (`R`, `R₁`, `R₃`, divided differences) better suited to option-pricing primitives than the complex-plane Faddeeva surface.

## Why MathPF

| | MathPF | `scipy.special` | Boost.Math |
|---|---|---|---|
| Mills ratio `R(x)`, `R₁(x) = -R'(x)`, `R₃(x) = -R'''(x)` | ✅ direct | compose via `erfcx` | compose via `erfc` |
| Symmetric divided difference `R_DD(x, dx)` | ✅ direct | not exposed | not exposed |
| `erfcx(z)`, `erfcx'(z)`, `erfcx'''(z)` | ✅ direct, via Mills | ✅ value only | ✅ value only |
| Cancellation-free `1 - x R(x)` (= `R₁`) | ✅ pinned in coefficients | ❌ subtraction loss for large `x` | ❌ same |
| Compile-time templated for `T ∈ {float, double}` | ✅ | n/a | ✅ |
| Vendorable as `_kernels/` for C++ consumers | ✅ | n/a (Python wheel) | ✅ header-only |

**MathPF is the right pick when** you need derivative-order >1 in a cancellation-free form, when you're computing tail probabilities or hazard ratios where `1 − Φ(x)` loses digits, or when you want the *same* implementation used inside pricing/IV kernels accessible both from Python and from your own C++ tree.

## Install

```bash
pip install mathpf
```

Wheels are built for Python 3.10–3.13 on Linux x86_64, macOS arm64 + x86_64, and Windows AMD64.

To install from source (requires a C++17 compiler):

```bash
git clone https://github.com/quants-net/MathPF
cd MathPF
pip install -e .
```

## Quick start

```python
import numpy as np
import mathpf

# --- Mills ratio family ----------------------------------------------------
mathpf.millsratio(0.5)              # R(x)              -> 1.141087  (scalar)
mathpf.millsratio_d1(0.5)           # -R'(x) = 1 - xR   -> 0.42946   (scalar)
mathpf.millsratio_d3(0.5)           # -R'''(x)          -> ...
mathpf.millsratio_rel_below1(0.5)   # (sqrt(pi/2) - R(x))/x on [0, 1]

# Vectorised over numpy arrays:
xs = np.linspace(-5, 5, 11)
mathpf.millsratio(xs)               # ndarray out, same shape

# Symmetric divided difference R_DD(x, dx) = (R(x-dx) + theta R(x+dx)) / (2 dx)
mathpf.millsratio_dd(3.0, 0.1, theta=+1)   # difference branch (numerically stable)
mathpf.millsratio_dd(3.0, 0.1, theta=-1)   # sum branch

# --- erfcx interface (same machinery, scipy-compatible signature) ----------
mathpf.erfcx(2.0)                   # exp(z^2) * erfc(z) = 0.255395
mathpf.erfcx_d1(2.0)                # d/dz erfcx(z)
mathpf.erfcx_d3(2.0)                # d^3/dz^3 erfcx(z)

# --- Average-value helpers (separate module: avg_funcs) ---------------------
mathpf.logrel(0.001)                # log(1+x)/x, cancellation-free near 0
mathpf.powrel(0.001, n=3)           # ((1+x)^n - 1) / (n x), similarly
```

## API overview

### `mathpf.mills` — Mills ratio and derivatives

| Function | Math | Domain | Notes |
|---|---|---|---|
| `millsratio(x)` | `R(x) = N(-x) / n(x)` | any `x` | Reflection-handled internally for `x < 0` |
| `millsratio_d1(x)` | `R₁(x) = -R'(x) = 1 - x R(x)` | any `x` | Cancellation-free (not `1 - small`) |
| `millsratio_d3(x)` | `R₃(x) = -R'''(x)` | any `x` | Direct, not from `R` |
| `millsratio_rel_below1(x)` | `(√(π/2) − R(x)) / x` | `x ∈ [0, 1]` | Polynomial primitive shared by `R`, `R₁` |

### `mathpf.mills_dd` — Symmetric divided differences

| Function | Math | Use case |
|---|---|---|
| `millsratio_dd(x, dx, theta)` | `(R(x − dx) ± R(x + dx)) / (2 dx)` | Volatility-derivative primitive |
| `millsratio_dd_cf(x, dx, n_terms)` | Continued-fraction direct form | When `dx ≪ x` (asymptotic ladder) |

### `mathpf.erfcx` — Scaled complementary error function

All routed through the Mills primitives, so accuracy mirrors `MillsRatio`.

| Function | Math | Identity |
|---|---|---|
| `erfcx(z)` | `exp(z²) · erfc(z)` | `R(z·√2) · √(2/π)` |
| `erfcx_d1(z)` | `d/dz erfcx(z)` | `−R₁(z·√2) · 2/√π` |
| `erfcx_d3(z)` | `d³/dz³ erfcx(z)` | `−R₃(z·√2) · 4/√π` |

The 2nd derivative is intentionally not exposed; `erfcx''(z) = 2 erfcx(z) + 2 z erfcx'(z)` composes trivially from what's already there (the same convention as MathPF's Mills surface, which exposes `R, R₁, R₃` and skips `R₂`).

### `mathpf.avg_funcs` — Cancellation-free averaging primitives

| Function | Math | Notes |
|---|---|---|
| `logrel(x)` | `log(1 + x) / x` | `→ 1` as `x → 0`, accurate near 0 |
| `powrel(x, n)` | `((1 + x)ⁿ − 1) / (n x)` | `→ 1` as `x → 0`, integer `n` |

## Accuracy

Cross-checked against an mpmath reference at 80-digit precision on a dense grid spanning every dispatch regime:

| Function | Worst absolute ulps |
|---|---|
| `millsratio`, `millsratio_d1` | ≤ 3 |
| `millsratio_d3` | ≤ 5 |
| `millsratio_rel_below1` | ≤ 2 |
| `millsratio_dd` (difference branch) | ≤ 64 (Taylor truncation at gate corner) |
| `erfcx` | ≤ 15 (cross-checked vs `scipy.special.erfcx`) |
| `erfcx_d1` | ≤ 15 (via analytic identity vs `scipy.special.erfcx`) |
| `erfcx_d3` | ≤ 15 (small `|z|`); analytic identity loses precision at large `|z|` |

Python reference (`mathpf._pyref`) is bit-equal to the compiled binding via `tests/test_pyref_consistency.py`.

## Using the C++ kernels in another project

MathPF's templated C++ kernels are designed to be vendored at the source level. The kernel tree is:

```
src/mathpf/
├── _kernels/                       # C++ source (vendor this)
│   ├── MathConstants.h             # M_SQRT2PI, M_SQRT2PI_2, ..., X_NEG_MAX, U_MAX
│   ├── mills.h / mills.cpp
│   ├── mills_dd.h / mills_dd.cpp
│   └── erfcx.h / erfcx.cpp
└── _mills_coef.h                   # Auto-generated coefficient tables (vendor this)
```

To vendor into a downstream C++ project:

1. Copy `_kernels/*` and `_mills_coef.h` into your own tree.
2. Compile the `.cpp` files into your binary alongside your own sources.
3. Call from C++ with the `mathpf::` namespace, or use the `extern "C"` shim symbols (`mathpf_MillsRatio`, `mathpf_Erfcx`, etc.) for FFI.

```cpp
#include "mills.h"      // mathpf::MillsRatio, MillsRatioDeriv1, ...
#include "erfcx.h"      // mathpf::Erfcx, ErfcxDeriv1, ErfcxDeriv3

// Inside your own namespace, pull mathpf:: in for terse call sites:
namespace mylib {
    using namespace mathpf;

    double price(double sig, double k) {
        double m = k / sig;
        return sig * MillsRatioDeriv1(m);   // unqualified, resolves to mathpf::
    }
}
```

## Math identities reference

The cancellation-free identities below are the contract MathPF's kernels implement; the same equations cross-check the Python reference (`_pyref`) against the compiled binding bit-for-bit in `tests/test_pyref_consistency.py`.

### Mills ratio family

Let `n(x) = exp(−x²/2)/√(2π)` and `N(x) = ∫₋∞ˣ n(t) dt`.

| Identity | Notes |
|---|---|
| `R(x) := N(−x) / n(x)` | The Mills ratio (inverse hazard) |
| `R(0) = √(π/2)` | Endpoint value, hard-pinned in coefficients |
| `R(x) = √(2π) · exp(x²/2) − R(−x)` | Reflection for `x < 0`; saturates beyond `X_NEG_MAX<T>` |
| `R(x) ~ 1/x − 1/x³ + 3/x⁵ − ⋯` | Asymptotic expansion for `x → +∞` |
| `R₁(x) := 1 − x R(x) = −R'(x)` | Cancellation-free `−R'`; the `1 − x R` form preserves precision at large `x` |
| `R₁(0) = 1` | |
| `R₁(x) ~ 1/x² − 3/x⁴ + 15/x⁶ − ⋯` | |
| `R₃(x) := −R'''(x)` | Direct kernel, not differentiated from `R` |
| `R₃(x) ~ −6/x⁴ + 60/x⁶ − ⋯` | |
| Laplace continued fraction | `R(x) = 1/(x + 1/(x + 2/(x + 3/(x + ⋯ ))))` |

The shared `[0, 1]` primitive `Rrel_below1(x) = (√(π/2) − R(x)) / x` is the polynomial routed by `R` and `R₁` to avoid evaluating `R` near the singularity at `x = 0`.

### Symmetric divided differences

For `MillsRatioDiff(x, dx, θ)` with `θ ∈ {+1, −1}`:

| `θ` | Formula | Limit `dx → 0` |
|:---:|---|---|
| `+1` | `[R(x − dx) − R(x + dx)] / (2 dx)` | `−R'(x) = R₁(x)` |
| `−1` | `[R(x − dx) + R(x + dx)] / (2 dx)` | `R(x) / dx`  (singular; not a derivative limit) |

Three internal regimes (gated on `dx` vs. `0.0392 · (1.25 + x)`): direct subtraction at moderate `dx`, asymptotic continued-fraction ladder at small `dx` with `x − dx ≥ 21.2`, and a 5-term `R'''`-seeded Taylor expansion below that gate.

### erfcx via Mills

Let `x = z · √2`. Then:

| Identity | Mills form |
|---|---|
| `erfcx(z) := exp(z²) · erfc(z)` | `= R(x) · √(2/π)` |
| `erfcx'(z) = 2 z erfcx(z) − 2/√π` | `= −R₁(x) · 2/√π` |
| `erfcx''(z) = 2 erfcx(z) + 2 z erfcx'(z)` | (not exposed; trivial composition) |
| `erfcx'''(z)` | `= −R₃(x) · 4/√π` |
| `erfcx(0) = 1` | (1 ulp due to FP64 reciprocal rounding of `√(π/2) · √(2/π)`) |
| `erfcx(z) → 1/(z √π) − ⋯` as `z → +∞` | Inherits Mills' asymptotic series |

### Average-value helpers

For `x → 0`:

| Identity | Limit |
|---|---|
| `logrel(x) := log(1 + x) / x` | `→ 1` |
| `powrel(x, n) := ((1 + x)ⁿ − 1) / (n x)`, integer `n` | `→ 1` |

Both implemented to avoid the `0/0` form at `x = 0`.

### Constants in `MathConstants.h`

All in `mathpf::` namespace, 22 sig-fig precision via mpmath (round-trip-stable to the nearest FP64):

| Name | Value | Used by |
|---|---|---|
| `M_SQRT_2<T>` | `√2 ≈ 1.41421…` | erfcx wrappers (spelled with underscore to avoid clashing with POSIX `<math.h>` macro `M_SQRT2`) |
| `M_SQRT2PI<T>` | `√(2π) ≈ 2.50662…` | Mills reflection |
| `M_SQRT2PI_2<T>` | `√(π/2) = R(0)` | `R` segment-zero anchor |
| `M_SQRT_2_PI<T>` | `√(2/π) = 1/M_SQRT2PI_2` | erfcx |
| `M_2_SQRT_PI<T>` | `2/√π ≈ 1.12838…` | erfcx derivatives |
| `X_NEG_MAX<T>` | `−√(2 log T_max)` | Reflection-branch overflow gate |
| `U_MAX<T>` | `√(T_max)` | `R₁/R₃` underflow gate |

The last two are type-dependent (`inline const`, derived from `std::numeric_limits<T>::max()`):

```cpp
template<typename T>
inline T const X_NEG_MAX = -std::sqrt(T(2) * std::log(std::numeric_limits<T>::max()));
template<typename T>
inline T const U_MAX     =  std::sqrt(std::numeric_limits<T>::max());
```

So `MillsRatio<float>` saturates at the actual `FLT_MAX`-derived boundary (`x ≈ −13.3`), not the conservative `DBL_MAX` one (`x ≈ −37.68`).

(`X_NEG_MAX` and `U_MAX` were the only `inline const`s in MathConstants.h needing the runtime initialisation; everything else is `constexpr`. C++17 `std::sqrt` / `std::log` are non-constexpr, so we use `inline const` instead; C++23's `constexpr <cmath>` would let us tighten this to a true `constexpr` — purely cosmetic, no runtime impact.)

## Development

```bash
git clone https://github.com/quants-net/MathPF
cd MathPF
pip install -e ".[dev]"
pip install scipy                     # optional, for erfcx cross-check tests
pytest tests/                          # 486 tests, < 1 s
```

To regenerate coefficient tables from scratch (requires `mpmath`):

```bash
python tools/cheby_fit.py              # writes src/mathpf/_mills_coef.h
```

## Citing / referencing

If you use MathPF in published research, please reference the project URL until a Zenodo DOI is available:

> Choi, J. (2026). MathPF: numerically-precise special functions for quantitative finance. https://github.com/quants-net/MathPF

## License

MIT — see [`LICENSE`](LICENSE).
