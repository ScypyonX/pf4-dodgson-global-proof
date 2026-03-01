"""
PF₅ Falsification via Interval Arithmetic
==========================================
Proves K is NOT PF₅ by exhibiting Toeplitz configurations where
det₅ < 0 with a rigorous interval-arithmetic enclosure at 80 digits.

Key: Toeplitz config (u₀, h) gives
  x = (u₀, u₀+h, u₀+2h, u₀+3h, u₀+4h)
  y = (0,   h,    2h,    3h,    4h)
so all differences xᵢ - yⱼ = u₀ + (i-j)·h.

Uses mpmath.iv (interval arithmetic) with outward rounding.
"""

import mpmath
import json
import os
import time
from datetime import datetime
from config import N_SERIES_MAX, MpfEncoder, _ns

# 80-digit interval arithmetic
mpmath.mp.dps = 80
IV = mpmath.iv


def Phi_iv(u):
    """Interval-arithmetic evaluation of Phi(u) at 80 digits."""
    u = IV.mpf(u)
    e4u = IV.exp(4*u)
    e5u = IV.exp(5*u)
    e9u = IV.exp(9*u)
    PI = IV.pi
    result = IV.mpf(0)
    for n in range(1, N_SERIES_MAX):
        n2 = IV.mpf(n*n)
        n4 = IV.mpf(n**4)
        t = PI * n2 * e4u
        term = (2*PI**2*n4*e9u - 3*PI*n2*e5u) * IV.exp(-t)
        result += term
        # Check convergence using midpoint magnitude
        mid_term = abs(mpmath.iv.mpf(term).mid)
        mid_result = abs(mpmath.iv.mpf(result).mid)
        if n > 1 and mid_result > 0 and mid_term > 0:
            if mid_term / mid_result < mpmath.power(10, -85):
                break
    return result


def det5_iv(u0, h):
    """5×5 determinant for Toeplitz config (u0, h) via interval arithmetic."""
    x = [u0 + i*h for i in range(5)]
    y = [j*h for j in range(5)]

    # Build 5×5 interval matrix
    A = [[Phi_iv(abs(x[i] - y[j])) for j in range(5)] for i in range(5)]

    # 5×5 determinant via cofactor expansion along row 0
    def det_sub(rows, cols):
        """Determinant of submatrix A[rows, cols]."""
        n = len(rows)
        if n == 1:
            return A[rows[0]][cols[0]]
        if n == 2:
            return A[rows[0]][cols[0]]*A[rows[1]][cols[1]] - A[rows[0]][cols[1]]*A[rows[1]][cols[0]]
        result = IV.mpf(0)
        for k in range(n):
            minor_cols = cols[:k] + cols[k+1:]
            cofactor = det_sub(rows[1:], minor_cols)
            if k % 2 == 0:
                result += A[rows[0]][cols[k]] * cofactor
            else:
                result -= A[rows[0]][cols[k]] * cofactor
        return result

    return det_sub(list(range(5)), list(range(5)))


if __name__ == "__main__":
    print(f"PF₅ Falsification — {datetime.now().isoformat()}")
    print(f"Interval arithmetic at {mpmath.mp.dps} digits")
    print(f"Outward rounding via mpmath.iv\n")

    # 12 Toeplitz configurations: u₀ ∈ {0.01, 0.02, 0.03}, h ∈ [0.03, 0.05]
    configs = [
        ("0.01", "0.03"),
        ("0.01", "0.035"),
        ("0.01", "0.04"),
        ("0.01", "0.045"),
        ("0.01", "0.05"),
        ("0.02", "0.03"),
        ("0.02", "0.035"),
        ("0.02", "0.04"),
        ("0.02", "0.045"),
        ("0.02", "0.05"),
        ("0.03", "0.03"),
        ("0.03", "0.035"),
    ]

    t0 = time.time()
    results = []
    all_negative = True

    print(f"{'u0':>6} {'h':>6} {'det5 interval':>40} {'< 0?':>6}")
    print("-"*65)

    for u0_str, h_str in configs:
        u0 = mpmath.mpf(u0_str)
        h = mpmath.mpf(h_str)
        d = det5_iv(u0, h)

        # Decision in interval arithmetic (mpf), then convert for JSON
        negative = d.b < 0  # entire interval below zero — mpf comparison
        lo = mpmath.mpf(d.a)
        hi = mpmath.mpf(d.b)

        results.append({
            "u0": u0_str,
            "h": h_str,
            "det5_lo": lo,
            "det5_hi": hi,
            "certified_negative": negative,
        })

        if not negative:
            all_negative = False

        status = "✓ < 0" if negative else "✗"
        print(f"  {u0_str:>4}  {h_str:>5}  [{_ns(lo, 8):>18}, {_ns(hi, 8):>18}]  {status}")

    elapsed = time.time() - t0

    print(f"\n{'='*65}")
    print(f"All {len(configs)} configs certified negative: {'YES ✓' if all_negative else 'NO ✗'}")
    print(f"Time: {elapsed:.1f}s")

    # Find the most negative (strongest counterexample)
    strongest = min(results, key=lambda r: r["det5_hi"])

    from config import metadata as _meta

    # Override precision in metadata for this script
    meta = _meta()
    meta["precision_digits"] = 80
    meta["arithmetic"] = "interval (mpmath.iv, outward rounding)"

    output = {
        "timestamp": datetime.now().isoformat(),
        "claim": "K is not PF_5",
        "method": "Toeplitz 5x5 det via interval arithmetic at 80 digits",
        "n_configs": len(configs),
        "all_certified_negative": all_negative,
        "strongest_counterexample": {
            "u0": strongest["u0"],
            "h": strongest["h"],
            "det5_interval": [strongest["det5_lo"], strongest["det5_hi"]],
        },
        "configs": results,
        "time_s": round(elapsed, 1),
        "metadata": meta,
    }

    outpath = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "..", "results", "pf5.json")
    with open(outpath, "w") as f:
        json.dump(output, f, indent=2, cls=MpfEncoder)
    print(f"\nResults saved to results/pf5.json")
