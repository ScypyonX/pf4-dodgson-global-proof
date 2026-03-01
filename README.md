# Global total positivity of order four for the de Bruijn–Newman kernel

Computational certificates for the paper:

> W. Michałowski, *Global total positivity of order four for the de Bruijn–Newman kernel with a rigorous interval certificate* (2026).

The kernel K(u) = Φ(|u|) associated with the de Bruijn–Newman constant is shown to be a Pólya frequency function of order 4. The proof partitions the seven-dimensional parameter space into four regions. Grid certificates use 60-digit multiprecision; the Taylor bridge, adaptive tiling, and PF₅ falsification use interval arithmetic (`mpmath.iv`) at 80 digits with outward rounding.

## Requirements

Python ≥ 3.10, mpmath ≥ 1.3.

```
pip install -r requirements.txt
```

## Quick check (~30 s)

```
python certs/pf4_quick_check.py
```

## Full reproduction

```
bash verify_all.sh
```

Or run individual components:

| Command | Certificate | Runtime |
|---------|-------------|---------|
| `python certs/pf4_master_certificate.py` | K2+E0, PF₃, Region 1 grid, Schur | ~3 min |
| `python certs/pf4_boundary_taylor.py` | Taylor bridge (mini-cell) | ~30 s |
| `python certs/pf4_continuous_tiling.py --mode full` | Adaptive IV tiling (Region 1) | ~35 min |
| `python certs/pf4_schur_tail.py` | Schur complement | ~10 s |
| `python certs/pf4_shift_tail.py` | Shift tail | ~20 s |
| `python certs/pf4_coalescence.py` | Coalescence | ~15 s |
| `python certs/verify_pf5.py` | PF₅ falsification (interval arithmetic) | ~1 s |
| `python tests/test_kernel.py` | Unit tests | ~5 s |

Each script outputs a JSON file to `results/` recording the result, precision, and environment metadata. Frozen outputs are provided for reference; re-running overwrites them.

## Repository structure

```
certs/
  kernel.py                  Canonical kernel Φ, K, L₄ (single source of truth)
  config.py                  Shared parameters, units, reproducibility metadata
  pf4_master_certificate.py  K2+E0, PF₃, Region 1 grid, Schur spot-check
  pf4_boundary_taylor.py     Taylor bridge: IV enclosure near Toeplitz minimum
  pf4_continuous_tiling.py   Adaptive IV tiling for continuous L₄ > 0
  pf4_schur_tail.py          Schur complement ratio for large gaps
  pf4_shift_tail.py          Local Lipschitz cert for large shifts
  pf4_coalescence.py         Leading coefficient at coalescence boundary
  pf4_quick_check.py         Reduced grid for quick verification
  verify_pf5.py              PF₅ falsification via interval arithmetic
results/                     Frozen JSON outputs (all certified: true)
paper/
  main.tex                   arXiv source
  refs.bib                   Bibliography
tests/
  test_kernel.py             Unit tests for kernel properties
docs/
  proof_structure.md         Internal proof structure notes
  checklist.md               Internal audit checklist
verify_all.sh                One-command verification of all certificates
```

## Proof architecture

Parameters: δ = 0.05, S = 0.4, G = 0.60. Working unit: 1 hcu = 0.005 real.

**Zero-float policy.** All PASS/FAIL decisions are performed in `mpmath.mpf` (60 digits) for grid certificates, and in `mpmath.iv` (80 digits, outward rounding) for the Taylor bridge, adaptive tiling, and PF₅ falsification. Floating-point conversion occurs only at JSON serialization.

1. **Bounded interior** (gaps ∈ [δ, G], |s| ≤ S): Adaptive interval-arithmetic tiling on [δ, 44 hcu]⁶ × [-S, S] certifies L₄ > 0 continuously (25,088 boxes, worst bound 5.0 × 10⁻⁴). The Schur complement covers any configuration with a gap > 42 hcu, providing a 2 hcu overlap.
2. **Unbounded tail** (∃ gap > G): Schur complement decomposition; ratio ρ < 0.986.
3. **Shift tail** (|s| > S): Local Lipschitz bounds per interval; worst margin 1.810.
4. **Coalescence** (∃ gap < δ): Taylor expansion of det₄; leading coefficient Ñ ≥ 8.25 × 10⁻²³.

## Reproducibility

All JSON files in `results/` are frozen reference outputs with `certified: true`. Each JSON includes a `metadata` block with Python version, platform, mpmath version, and domain parameters, so any discrepancy can be traced to environment differences. Run `bash verify_all.sh` to regenerate and verify all certificates.

## Citation

See `CITATION.cff`.

## License

MIT. See `LICENSE`.
