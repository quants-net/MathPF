"""Generate high-precision reference values for the Mills primitives R, R1,
R3, R_rel (at x = 0, 0.5, 1.0, ..., 20.0) AND the divided-difference primitive
MillsRatioDiff_CF (at the XCF_R1-tier (a, n_terms) pairings cross dx grid),
using mpmath at 50 decimal digits.  Emits a Python module of plain float
constants for the test suite to import.

The generated file (tests/_ref_table.py) is committed; mpmath is NOT a
test-time dependency.  Re-run this generator only when extending the
anchor grid or revising precision.

Usage:
    python tools/gen_mills_ref.py            # prints to stdout
    python tools/gen_mills_ref.py --write    # overwrites tests/_ref_table.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mpmath as mp


# 50 decimal digits is comfortably more than double precision (~17 digits)
# minus headroom for any cancellation inside the formulas below.
mp.mp.dps = 50


def R(x):
    """Mills ratio R(x) = N(-x)/n(x) = sqrt(pi/2) exp(x^2/2) erfc(x/sqrt(2)).
    Computed with the erfc-scaling identity for numerical stability."""
    xm = mp.mpf(x)
    return mp.sqrt(mp.pi / 2) * mp.exp(xm * xm / 2) * mp.erfc(xm / mp.sqrt(2))


def R1(x):
    """R1(x) = -R'(x) = 1 - x R(x)."""
    xm = mp.mpf(x)
    return 1 - xm * R(xm)


def R3(x):
    """R3(x) = -R'''(x) = (x^2 + 3) R1(x) - 1
    Equivalent to (x^2 + 3)(1 - x R) - 1."""
    xm = mp.mpf(x)
    return (xm * xm + 3) * R1(xm) - 1


def R_rel(x):
    """Cancellation-free near-zero form on [0, 1]:
        R_rel(x) = (sqrt(pi/2) - R(x)) / x ,
        R_rel(0) := 1   (by L'Hopital: R(x) = sqrt(pi/2) - x + O(x^2))
    The test grid for this primitive is x in [0, 1]."""
    xm = mp.mpf(x)
    if xm == 0:
        return mp.mpf(1)
    return (mp.sqrt(mp.pi / 2) - R(xm)) / xm


def MRDD(x, dx):
    """Mills-ratio divided difference (symmetric form):
        MRDD(x, dx) = (R(x - dx) - R(x + dx)) / (2 dx) .
    Equivalent to MillsRatioDiff(x, dx, theta=+1) -- the difference branch
    (theta=-1 sum branch is a separate primitive)."""
    xm, dxm = mp.mpf(x), mp.mpf(dx)
    return (R(xm - dxm) - R(xm + dxm)) / (mp.mpf(2) * dxm)


# (a, n_terms) pairs for the MillsRatioDiff_CF test, matching XCF_R1 tiers.
# a = x - dx is the smaller Mills argument; n_terms is the CF order paired
# with that tier in the production dispatcher (XCF_R1[k] -> n_terms 12, 10,
# 8, 6, 4, 2 going outward; the bridge n=12 is the in-tier order for
# a >= XCF_R1[0]).
MRDD_TIERS = (
    (11.5,    12),
    (14.5,    10),
    (21.2,     8),
    (41.0,     6),
    (165.0,    4),
    (12800.0,  2),
)
MRDD_DXS   = (0.0001, 0.01, 1.0, 2.0, 4.0, 8.0)

# (a, m) grid for the branch-AGNOSTIC test of millsratio_dd(x, dx, +1).
# m = dx / (1.25 + x) is the gate-fraction; the dispatcher's Taylor / Direct
# threshold sits at 0.0392, so m = 0.0391 / 0.0393 straddle the gate corner
# from below / above respectively.
#
# Why parameterise by m: lets the test grid stay stable even if the gate
# constant moves in the future.  The test calls the public dispatcher with
# no assertion about which internal branch handles the cell.
#
# For each (a, m): dx = m * (1.25 + a) / (1 - m)  -- inverts m = dx/(1.25+x).
MRDD_DD_AS = (0.1, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
MRDD_DD_MS = (0.01,     # deep small-dx regime
              0.0391,   # just below the Taylor/Direct gate
              0.0393,   # just above the gate
              0.1,      # moderate
              0.25,
              0.75)     # large dx (deep Direct, dx > x)


def _fmt_dict_lines(name: str, mapping: dict) -> list[str]:
    """Emit `name = {x: value, ...}` with each entry on its own line and
    values printed at 18 significant digits (well past double precision)."""
    lines = [f"{name} = {{"]
    for x, v in mapping.items():
        lines.append(f"    {float(x):>4}: {float(v):.17e},")
    lines.append("}")
    return lines


def main(argv):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--write", action="store_true",
                   help="Write tests/_ref_table.py.  Without this, print to stdout.")
    args = p.parse_args(argv)

    # x grids:
    #   - R, R1, R3 (defined on R+): m = 0..40 at step 0.5 (41 nodes, x in [0, 20])
    #   - R_rel (cancellation-free near-zero form): k = 0..20 at step 0.05
    #     (21 nodes, x in [0, 1] -- finer because the valid domain is small)
    xs_full = [0.5  * m for m in range(0, 41)]   # 0, 0.5, 1.0, ..., 20.0
    xs_rel  = [round(0.05 * k, 2) for k in range(0, 21)]   # 0, 0.05, 0.10, ..., 1.0

    R_table     = {x: R(x)     for x in xs_full}
    R1_table    = {x: R1(x)    for x in xs_full}
    R3_table    = {x: R3(x)    for x in xs_full}
    Rrel_table  = {x: R_rel(x) for x in xs_rel}

    # MillsRatioDiff_CF (symmetric divided difference) reference: 2D table
    # indexed by (a, n_terms) tier first, then dx.  Stored as a list-of-dicts
    # paralleling MRDD_TIERS / MRDD_DXS for direct index lookup in the test.
    MRDD_table = []
    for a, _n in MRDD_TIERS:
        row = {}
        for dx in MRDD_DXS:
            x = a + dx                  # x = a + dx by construction (a = x - dx)
            row[dx] = MRDD(x, dx)
        MRDD_table.append(row)

    # Branch-agnostic MRDD reference for the dispatcher: flat list of
    # (a, m, dx, value) tuples.  dx = m * (1.25 + a) / (1 - m); x = a + dx.
    MRDD_DD_table = []
    for a in MRDD_DD_AS:
        for m in MRDD_DD_MS:
            dx = m * (1.25 + a) / (1.0 - m)
            x = a + dx
            MRDD_DD_table.append((a, m, dx, MRDD(x, dx)))

    header = (
        '"""High-precision reference values for the Mills primitives.\n'
        "\n"
        "Generated by tools/gen_mills_ref.py with mpmath at 50 decimal digits;\n"
        "each value is rounded to 17 significant decimal digits (just past double\n"
        "precision).  Tests import these constants directly so mpmath is not a\n"
        "test-time dependency.  Re-run the generator if the anchor grid changes.\n"
        "\n"
        "Reference definitions:\n"
        "    R(x)       = N(-x)/n(x) = sqrt(pi/2) exp(x^2/2) erfc(x/sqrt(2))\n"
        "    R1(x)      = -R'(x)     = 1 - x R(x)\n"
        "    R3(x)      = -R'''(x)   = (x^2 + 3)(1 - x R(x)) - 1\n"
        "    R_rel(x)   = (sqrt(pi/2) - R(x)) / x,  defined on [0, 1] with R_rel(0) := 1\n"
        "    MRDD(x,dx) = (R(x-dx) - R(x+dx)) / (2 dx)  -- symmetric divided difference\n"
        "                  (= MillsRatioDiff(x, dx, theta=+1) difference branch)\n"
        "\"\"\"\n"
    )

    lines = [header]
    lines.extend(_fmt_dict_lines("R", R_table))
    lines.append("")
    lines.extend(_fmt_dict_lines("R1", R1_table))
    lines.append("")
    lines.extend(_fmt_dict_lines("R3", R3_table))
    lines.append("")
    lines.extend(_fmt_dict_lines("R_rel", Rrel_table))
    lines.append("")
    # MRDD reference: 2D structure that mirrors MRDD_TIERS / MRDD_DXS.
    lines.append("# MillsRatioDiff_CF reference: (a, n_terms) tier pairings cross dx grid.")
    lines.append("# Each MRDD[i] is a dict keyed by dx; tier i corresponds to MRDD_TIERS[i].")
    lines.append(f"MRDD_TIERS = {MRDD_TIERS!r}")
    lines.append(f"MRDD_DXS   = {MRDD_DXS!r}")
    lines.append("MRDD = [")
    for row in MRDD_table:
        lines.append("    {")
        for dx, v in row.items():
            lines.append(f"        {float(dx):>8}: {float(v):.17e},")
        lines.append("    },")
    lines.append("]")
    lines.append("")

    # Branch-agnostic dispatcher reference: list of (a, m, dx, value) tuples.
    lines.append("# MillsRatioDiff branch-agnostic dispatcher reference: (a, m) grid.")
    lines.append("# m = dx/(1.25+x) is the gate fraction; dispatcher gates at m = 0.0392.")
    lines.append("# Each row: (a, m, dx, expected_value) -- dx is derived for ease of use.")
    lines.append(f"MRDD_DD_AS = {MRDD_DD_AS!r}")
    lines.append(f"MRDD_DD_MS = {MRDD_DD_MS!r}")
    lines.append("MRDD_DD = [")
    for a, m, dx, v in MRDD_DD_table:
        # dx emitted at full repr() precision so the parsed-back float is
        # bit-identical to what was used to compute the mpmath truth value.
        lines.append(f"    ({float(a):>5}, {float(m):>7}, {float(dx)!r:>23}, {float(v):.17e}),")
    lines.append("]")
    lines.append("")
    out = "\n".join(lines)

    if args.write:
        target = Path(__file__).resolve().parent.parent / "tests" / "_ref_table.py"
        target.write_text(out, encoding="utf-8", newline="\n")
        print(f"wrote {target}  ({len(out)} bytes)")
    else:
        sys.stdout.write(out)


if __name__ == "__main__":
    main(sys.argv[1:])
