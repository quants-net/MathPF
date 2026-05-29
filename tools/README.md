# `tools/` — coefficient-refit and numerical-analysis scripts

Maintainer-only utilities. **Not shipped in the wheel** — these live outside
`src/mathpf/` and `find_packages(where="src")` does not pick them up.

## Setup

These scripts depend on `mpmath` (and `pytest` for verification). They are
declared as the `dev` extras of the package:

```sh
pip install -e ".[dev]"
```

## What's here

| Script | Purpose |
|--------|---------|
| `cheby_fit.py` | Generate `src/mathpf/_pyref/mills_cheby.py` (the Mills-ratio Chebyshev / CF coefficient tables and dispatch thresholds). Includes `MPMR` (mpmath reference) inline. |
| `_cancel.py` | Cancellation-budget analysis for the `R'''-seeded` Taylor branch of `R_DD`. |
| `_oddcf.py` | Even-vs-odd convergent-order analysis for the `-R'''` continued fraction. |
| `_wlin.py` | Bucket-count analysis for the `w = x/(N+x)` segmented Chebyshev fit of `R3`. |
| `_tiers.py` | Tiered-CF accuracy sweep of the live `mathpf._pyref.mills_cheby` against an mpmath reference. |

## Typical refit workflow

```sh
# 1) From the MathPF repo root, run the emitter; rewrites _pyref/mills_cheby.py:
python tools/cheby_fit.py

# 2) Rebuild the Cython extensions (they consume _mills_coef.h, also auto-emitted):
pip install -e .

# 3) Run the consistency CI gate; fails if _pyref drifted from the compiled binding:
pytest tests/test_pyref_consistency.py -q
```

The bit-equality test in `tests/test_pyref_consistency.py` is the safety net:
if anything in this directory or in the C/Cython kernels drifts, the test
catches it before a release.
