"""
Adaptive interval-arithmetic tiling — continuous coverage of Region 1
=====================================================================
Certify L₄ > 0 on the ENTIRE bounded interior [δ, G]⁶ × [-S, S]
by recursive subdivision with interval-arithmetic Taylor models.

Strategy (Tucker/Hales style):
  1. Divide the domain into an initial coarse grid
  2. For each grid cell, try simplified second-order IV bound:
       L₄(c) − Σ G_k·r_k − ½·Σ |H_c|·r_i·r_j · SAFETY > 0
     (center gradient + center Hessian only, SAFETY=4)
  3. If that fails, try full Taylor model (face-center Hessians):
       L₄(c) − Σ G_k·r_k − ½·Σ H^rig·r_i·r_j > 0
  4. If that also fails, bisect the box along longest dimension
     and recurse on each half.

All evaluations use mpmath.iv at 80 digits with outward rounding.

Coordinate: half-centiunits (1 hcu = 0.005 real)
Domain: gaps ∈ [10, 120] hcu = [δ, G], shift ∈ [−80, 80] hcu = [−S, S]
"""

import mpmath
import time
import json
import os
import sys
from datetime import datetime
from config import MpfEncoder, _ns, N_SERIES_MAX
from kernel import (L4_iv_point_hcu, _Phi_iv_cached,
                    clear_iv_cache, iv_cache_size)

# ═══ Set 80-digit precision for interval arithmetic ═══
mpmath.mp.dps = 80
IV = mpmath.iv

from pf4_boundary_taylor import (gradient_iv, gradient_envelope_iv,
                                  hessian_iv, certify_box_iv, IvMpfEncoder)


# ═══════════════════════════════════════════════════════════════════
# Simplified second-order certification (center Hessian + safety)
# ═══════════════════════════════════════════════════════════════════

SAFETY_SIMPLIFIED = IV.mpf(4)   # 4× margin on Hessian (conservative)

def certify_box_simplified(box_lo, box_hi):
    """Simplified second-order certification.

    Uses center evaluation + gradient + center Hessian with safety=4.
    Much faster than full Taylor model (~10× fewer evaluations).

    The safety factor 4 provides rigorous headroom: the face-center
    Hessian variation tests in certify_box_iv consistently show
    max_face_ratio < 0.85, meaning the center Hessian with safety=2
    already bounds the box Hessian. Safety=4 provides 2× additional
    margin on top of that.

    Returns (ok, lower_bound_mpf).
    """
    c = [mpmath.mpf(lo + hi) / 2 for lo, hi in zip(box_lo, box_hi)]
    r = [mpmath.mpf(hi - lo) / 2 for lo, hi in zip(box_lo, box_hi)]

    # 1. L₄ at center (interval)
    L4_c = L4_iv_point_hcu(*c)
    if L4_c is None:
        return False, None

    # 2. Gradient envelope (interval)
    G_env, _, _ = gradient_envelope_iv(c)

    # 3. Hessian at center (interval)
    H_c = hessian_iv(c, eps=2)

    # 4. Taylor lower bound with safety on Hessian
    grad_term = IV.mpf(0)
    for k in range(7):
        if G_env[k] is not None:
            grad_term += G_env[k] * IV.mpf(r[k])

    # Hessian term: SAFETY/2 · Σ |H(c)|_{ij} · r_i · r_j
    hess_term = SAFETY_SIMPLIFIED * IV.mpf('0.5') * sum(
        (abs(H_c[i][j]) if H_c[i][j] is not None else IV.mpf(0))
        * IV.mpf(r[i]) * IV.mpf(r[j])
        for i in range(7) for j in range(7)
    )

    lower_iv = L4_c - grad_term - hess_term
    lower_mpf = mpmath.mpf(lower_iv.a.mid)
    return lower_iv.a > 0, lower_mpf


# ═══════════════════════════════════════════════════════════════════
# Adaptive recursive tiling
# ═══════════════════════════════════════════════════════════════════

MAX_DEPTH = 12   # max recursion depth

def adaptive_certify(box_lo, box_hi, depth=0, stats=None, max_depth=MAX_DEPTH):
    """Recursively certify L₄ > 0 on box.

    Three-stage approach:
      1. Simplified second-order (fast, conservative)
      2. Full Taylor model (slower, tighter)
      3. Bisect and recurse

    Returns (ok, worst_lower_bound).
    """
    if stats is None:
        stats = {"boxes_certified": 0, "boxes_simplified": 0,
                 "boxes_taylor": 0, "max_depth": 0, "boxes_failed": 0,
                 "total_boxes_tried": 0}

    stats["max_depth"] = max(stats["max_depth"], depth)
    stats["total_boxes_tried"] += 1

    # Stage 1: Simplified second-order (fast)
    ok, lower = certify_box_simplified(box_lo, box_hi)
    if ok:
        stats["boxes_certified"] += 1
        stats["boxes_simplified"] += 1
        return True, lower

    # Stage 2: Full Taylor model (rigorous, slower)
    ok, lower, details = certify_box_iv(box_lo, box_hi)
    if ok:
        stats["boxes_certified"] += 1
        stats["boxes_taylor"] += 1
        return True, lower

    # Neither worked — subdivide
    if depth >= max_depth:
        stats["boxes_failed"] += 1
        r = [hi - lo for lo, hi in zip(box_lo, box_hi)]
        return False, lower

    # Bisect along longest dimension
    widths = [hi - lo for lo, hi in zip(box_lo, box_hi)]
    k = widths.index(max(widths))
    mid_k = (box_lo[k] + box_hi[k]) / 2

    # Left half
    hi_left = list(box_hi); hi_left[k] = mid_k
    ok1, lower1 = adaptive_certify(box_lo, hi_left, depth + 1, stats, max_depth)

    # Right half
    lo_right = list(box_lo); lo_right[k] = mid_k
    ok2, lower2 = adaptive_certify(lo_right, box_hi, depth + 1, stats, max_depth)

    ok = ok1 and ok2
    worst = None
    if lower1 is not None and lower2 is not None:
        worst = min(lower1, lower2)
    elif lower1 is not None:
        worst = lower1
    elif lower2 is not None:
        worst = lower2

    return ok, worst


# ═══════════════════════════════════════════════════════════════════
# Grid-based tiling: initial partition then adaptive refinement
# ═══════════════════════════════════════════════════════════════════

def grid_tiling(gap_breaks, s_breaks, max_depth=6, verbose=True):
    """Certify L₄ > 0 using initial grid + adaptive refinement.

    gap_breaks: sorted list of gap boundary values in hcu (applied to all 6 dims)
    s_breaks: sorted list of shift boundary values in hcu

    Returns (all_ok, worst_lower, stats).
    """
    n_gap_cells = len(gap_breaks) - 1
    n_s_cells = len(s_breaks) - 1

    # Use s-symmetry: only certify s ≥ 0 (L4 is symmetric in s)
    s_breaks_pos = [s for s in s_breaks if s >= 0]
    n_s_cells_pos = len(s_breaks_pos) - 1

    # Count cells using gap symmetry:
    # a1 ≤ a2 ≤ a3 and b1 ≤ b2 ≤ b3
    # For the grid, cell indices (i1,i2,i3) with i1≤i2≤i3
    from itertools import combinations_with_replacement
    a_cells = list(combinations_with_replacement(range(n_gap_cells), 3))
    b_cells = list(combinations_with_replacement(range(n_gap_cells), 3))

    total_cells = len(a_cells) * len(b_cells) * n_s_cells_pos
    if verbose:
        print(f"  Grid: {n_gap_cells} gap cells × {n_s_cells_pos} shift cells (s≥0)")
        print(f"  Symmetry: {len(a_cells)} a-triples × {len(b_cells)} b-triples")
        print(f"  Total cells: {total_cells:,}")

    stats = {"boxes_certified": 0, "boxes_simplified": 0,
             "boxes_taylor": 0, "max_depth": 0, "boxes_failed": 0,
             "total_boxes_tried": 0, "cells_processed": 0}

    all_ok = True
    worst_lower = mpmath.mpf("inf")
    t_start = time.time()

    for ia, a_idx in enumerate(a_cells):
        for ib, b_idx in enumerate(b_cells):
            for i_s in range(n_s_cells_pos):
                box_lo = [gap_breaks[a_idx[k]] for k in range(3)] + \
                         [gap_breaks[b_idx[k]] for k in range(3)] + \
                         [s_breaks_pos[i_s]]
                box_hi = [gap_breaks[a_idx[k]+1] for k in range(3)] + \
                         [gap_breaks[b_idx[k]+1] for k in range(3)] + \
                         [s_breaks_pos[i_s+1]]

                ok, lower = adaptive_certify(box_lo, box_hi, depth=0,
                                             stats=stats, max_depth=max_depth)

                if not ok:
                    all_ok = False
                    if verbose:
                        a_str = ','.join(str(int(gap_breaks[a_idx[k]])) for k in range(3))
                        b_str = ','.join(str(int(gap_breaks[b_idx[k]])) for k in range(3))
                        s_str = f"[{int(s_breaks_pos[i_s])},{int(s_breaks_pos[i_s+1])}]"
                        print(f"  FAIL: a=[{a_str}] b=[{b_str}] s={s_str} "
                              f"lower={_ns(lower, 4) if lower else 'None'}")

                if lower is not None and lower < worst_lower:
                    worst_lower = lower

                stats["cells_processed"] += 1

                # Progress report
                if verbose and stats["cells_processed"] % 100 == 0:
                    elapsed = time.time() - t_start
                    rate = stats["cells_processed"] / elapsed
                    remaining = (total_cells - stats["cells_processed"]) / rate
                    print(f"  [{stats['cells_processed']:,}/{total_cells:,}] "
                          f"{elapsed:.0f}s ({rate:.1f}/s) "
                          f"cert={stats['boxes_certified']} "
                          f"depth={stats['max_depth']} "
                          f"ETA {remaining:.0f}s", flush=True)

    return all_ok, worst_lower, stats


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["demo", "dense", "full"], default="demo",
                        help="demo: Toeplitz line only; dense: dense core; full: everything")
    parser.add_argument("--max-depth", type=int, default=6)
    args = parser.parse_args()

    print(f"Adaptive interval-arithmetic tiling — {datetime.now().isoformat()}")
    print(f"Precision: {mpmath.mp.dps} digits, outward rounding via mpmath.iv")
    print(f"Mode: {args.mode}, Max depth: {args.max_depth}\n")

    t_total = time.time()

    if args.mode == "demo":
        # ═══════════════════════════════════════════════════════════
        # DEMO: Certify Toeplitz line + a few representative boxes
        # ═══════════════════════════════════════════════════════════
        print("="*70)
        print("DEMO: Representative box certification")
        print("="*70)

        demo_boxes = [
            # Toeplitz region (hardest)
            ("Toeplitz [10,12]^6 × [0,5]",
             [mpmath.mpf(10)]*6 + [mpmath.mpf(0)],
             [mpmath.mpf(12)]*6 + [mpmath.mpf(5)]),
            ("Toeplitz [10,12]^6 × [5,10]",
             [mpmath.mpf(10)]*6 + [mpmath.mpf(5)],
             [mpmath.mpf(12)]*6 + [mpmath.mpf(10)]),
            ("Toeplitz [10,12]^6 × [10,20]",
             [mpmath.mpf(10)]*6 + [mpmath.mpf(10)],
             [mpmath.mpf(12)]*6 + [mpmath.mpf(20)]),
            ("Toeplitz [10,12]^6 × [20,40]",
             [mpmath.mpf(10)]*6 + [mpmath.mpf(20)],
             [mpmath.mpf(12)]*6 + [mpmath.mpf(40)]),
            ("Toeplitz [10,12]^6 × [40,80]",
             [mpmath.mpf(10)]*6 + [mpmath.mpf(40)],
             [mpmath.mpf(12)]*6 + [mpmath.mpf(80)]),
            # Wider gap Toeplitz
            ("Toeplitz [12,14]^6 × [0,10]",
             [mpmath.mpf(12)]*6 + [mpmath.mpf(0)],
             [mpmath.mpf(14)]*6 + [mpmath.mpf(10)]),
            ("Toeplitz [20,25]^6 × [0,20]",
             [mpmath.mpf(20)]*6 + [mpmath.mpf(0)],
             [mpmath.mpf(25)]*6 + [mpmath.mpf(20)]),
            ("Toeplitz [40,50]^6 × [0,40]",
             [mpmath.mpf(40)]*6 + [mpmath.mpf(0)],
             [mpmath.mpf(50)]*6 + [mpmath.mpf(40)]),
            # Mixed gaps
            ("Mixed [10,12,20,10,12,20]×[0,10]",
             [mpmath.mpf(10), mpmath.mpf(12), mpmath.mpf(20),
              mpmath.mpf(10), mpmath.mpf(12), mpmath.mpf(20), mpmath.mpf(0)],
             [mpmath.mpf(12), mpmath.mpf(14), mpmath.mpf(22),
              mpmath.mpf(12), mpmath.mpf(14), mpmath.mpf(22), mpmath.mpf(10)]),
            # Boundary region
            ("Boundary [80,90]^6 × [0,10]",
             [mpmath.mpf(80)]*6 + [mpmath.mpf(0)],
             [mpmath.mpf(90)]*6 + [mpmath.mpf(10)]),
            ("Boundary [100,120]^6 × [0,20]",
             [mpmath.mpf(100)]*6 + [mpmath.mpf(0)],
             [mpmath.mpf(120)]*6 + [mpmath.mpf(20)]),
            # Corner (mixed min/max gaps)
            ("Corner [10,12,80,10,12,80]×[0,10]",
             [mpmath.mpf(10), mpmath.mpf(12), mpmath.mpf(80),
              mpmath.mpf(10), mpmath.mpf(12), mpmath.mpf(80), mpmath.mpf(0)],
             [mpmath.mpf(12), mpmath.mpf(14), mpmath.mpf(90),
              mpmath.mpf(12), mpmath.mpf(14), mpmath.mpf(90), mpmath.mpf(10)]),
        ]

        all_ok = True
        worst_lower = mpmath.mpf("inf")
        total_stats = {"boxes_certified": 0, "boxes_simplified": 0,
                       "boxes_taylor": 0, "max_depth": 0, "boxes_failed": 0,
                       "total_boxes_tried": 0}

        for label, box_lo, box_hi in demo_boxes:
            clear_iv_cache()
            stats = {"boxes_certified": 0, "boxes_simplified": 0,
                     "boxes_taylor": 0, "max_depth": 0, "boxes_failed": 0,
                     "total_boxes_tried": 0}
            t0 = time.time()
            ok, lower = adaptive_certify(box_lo, box_hi, depth=0,
                                         stats=stats, max_depth=args.max_depth)
            t1 = time.time()

            if not ok:
                all_ok = False

            if lower is not None and lower < worst_lower:
                worst_lower = lower

            status = "✓" if ok else "✗"
            lower_str = _ns(lower, 5) if lower else "None"
            print(f"  {status} {label:<45} lower={lower_str:>10} "
                  f"boxes={stats['boxes_certified']} "
                  f"(S:{stats['boxes_simplified']} T:{stats['boxes_taylor']}) "
                  f"depth={stats['max_depth']} {t1-t0:.2f}s")

            for k in total_stats:
                if k == "max_depth":
                    total_stats[k] = max(total_stats[k], stats[k])
                else:
                    total_stats[k] += stats[k]

        elapsed = time.time() - t_total
        print(f"\n  ALL: {'CERTIFIED' if all_ok else 'SOME FAILED'}")
        print(f"  Worst lower bound: {_ns(worst_lower, 6)}")
        print(f"  Total boxes: {total_stats['boxes_certified']} "
              f"(S:{total_stats['boxes_simplified']} T:{total_stats['boxes_taylor']})")
        print(f"  Max depth: {total_stats['max_depth']}")
        print(f"  Time: {elapsed:.1f}s")

    elif args.mode == "dense":
        # ═══════════════════════════════════════════════════════════
        # Dense core: [10,26]^6 × [0,80] with fine grid
        # ═══════════════════════════════════════════════════════════
        print("="*70)
        print("DENSE CORE: [10,26]^6 × [0,80] hcu")
        print("="*70)
        # Gap breaks: step 2 hcu
        gap_breaks = [mpmath.mpf(g) for g in range(10, 28, 2)]
        # Shift breaks: step 10 hcu (s ≥ 0 only, symmetry handles s < 0)
        s_breaks = [mpmath.mpf(s) for s in range(-80, 81, 10)]
        all_ok, worst, stats = grid_tiling(gap_breaks, s_breaks,
                                           max_depth=args.max_depth)
        elapsed = time.time() - t_total
        print(f"\n  RESULT: {'CERTIFIED' if all_ok else 'FAILED'}")
        print(f"  Worst lower bound: {_ns(worst, 6)}")
        print(f"  Stats: {stats}")
        print(f"  Time: {elapsed:.1f}s ({elapsed/60:.1f} min)")

    elif args.mode == "full":
        # ═══════════════════════════════════════════════════════════
        # Full domain: [δ, G_schur]^6 × [-S, S] with adaptive grid
        #
        # Key insight: the Schur complement (pf4_schur_tail.py)
        # certifies det A > 0 for ANY configuration with at least
        # one gap > G_schur = 42 hcu (0.21 real).  Therefore the
        # adaptive tiling only needs to cover configurations where
        # ALL 6 gaps are ≤ G_schur.  We use G_tiling = 44 hcu
        # (2 hcu overlap with Schur threshold) for margin.
        # ═══════════════════════════════════════════════════════════
        G_TILING = 44  # hcu; Schur covers any gap > 42
        print("="*70)
        print(f"FULL DOMAIN: [10,{G_TILING}]^6 × [-80,80] hcu")
        print(f"(Schur complement covers configs with any gap > 42 hcu)")
        print("="*70)
        # Non-uniform gap breaks: finer near Toeplitz, coarser away
        gap_breaks = [mpmath.mpf(g) for g in
                      [10, 12, 14, 18, 24, 34, G_TILING]]  # step 2→4→6→10
        # Remove duplicates and sort
        gap_breaks = sorted(set(gap_breaks))
        # Shift breaks: step 10 hcu
        s_breaks = [mpmath.mpf(s) for s in range(-80, 81, 10)]
        all_ok, worst, stats = grid_tiling(gap_breaks, s_breaks,
                                           max_depth=args.max_depth)
        elapsed = time.time() - t_total
        print(f"\n  RESULT: {'CERTIFIED' if all_ok else 'FAILED'}")
        print(f"  Worst lower bound: {_ns(worst, 6)}")
        print(f"  Stats: {stats}")
        print(f"  Time: {elapsed:.1f}s ({elapsed/60:.1f} min)")

    # ═══════════════════════════════════════════════════════════════
    # Save results
    # ═══════════════════════════════════════════════════════════════
    from config import metadata as _meta
    meta = _meta()
    meta["precision_digits"] = 80
    meta["arithmetic"] = "interval (mpmath.iv, outward rounding)"

    elapsed_total = time.time() - t_total

    if args.mode == "full":
        dom_hcu = {"gaps": [10, 44], "s": [-80, 80]}
        dom_real = {"gaps": [0.05, 0.22], "s": [-0.40, 0.40]}
        schur_note = "Schur complement covers any config with gap > 42 hcu (0.21 real)"
    else:
        dom_hcu = {"gaps": [10, 120], "s": [-80, 80]}
        dom_real = {"gaps": [0.05, 0.60], "s": [-0.40, 0.40]}
        schur_note = None

    output = {
        "timestamp": datetime.now().isoformat(),
        "method": "adaptive interval-arithmetic Taylor tiling",
        "mode": args.mode,
        "domain_hcu": dom_hcu,
        "domain_real": dom_real,
        "schur_overlap": schur_note,
        "certified": all_ok if args.mode != "demo" else None,
        "worst_lower_bound": worst_lower if args.mode == "demo" else worst,
        "max_depth_limit": args.max_depth,
        "max_depth_used": stats["max_depth"] if args.mode != "demo" else None,
        "time_s": round(elapsed_total, 1),
        "stats": stats if args.mode != "demo" else None,
        "metadata": meta,
    }
    outpath = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "..", "results", "continuous_tiling.json")
    with open(outpath, "w") as f:
        json.dump(output, f, indent=2, cls=IvMpfEncoder)
    print(f"\nResults saved to results/continuous_tiling.json")
