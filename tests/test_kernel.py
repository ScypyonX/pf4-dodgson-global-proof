"""
Unit tests for the de Bruijn-Newman kernel Φ and derived quantities.

Run: python -m pytest tests/ -v
  or: python tests/test_kernel.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'certs'))

import mpmath
mpmath.mp.dps = 60

# Import from canonical kernel module
from kernel import Phi, Phi_prime, Phi_double_prime, K_cached_cu as K_cached, L4_cu as L4

PI = mpmath.pi


def test_phi_positive():
    """K1: Φ(u) > 0 for u ≥ 0."""
    for u in [0, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0]:
        val = Phi(mpmath.mpf(u))
        assert val > 0, f"Φ({u}) = {val} ≤ 0"


def test_phi_decreasing():
    """K2: Φ'(u) < 0 for u > 0."""
    for u in [0.01, 0.05, 0.1, 0.5, 1.0, 2.0]:
        val = Phi_prime(mpmath.mpf(u))
        assert val < 0, f"Φ'({u}) = {val} ≥ 0"


def test_g_convex():
    """E0: g''(t) > 0 where g = -log Φ."""
    for t in [0.01, 0.05, 0.1, 0.5, 1.0]:
        t_mp = mpmath.mpf(t)
        pv = Phi(t_mp)
        pp = Phi_prime(t_mp)
        ppp = Phi_double_prime(t_mp)
        gpp = -ppp/pv + (pp/pv)**2
        assert gpp > 0, f"g''({t}) = {gpp} ≤ 0"


def test_phi_symmetry():
    """K(u) = Φ(|u|): check K_cached is symmetric."""
    for u_hcu in [5, 10, 20, 50, 100]:
        assert K_cached(u_hcu) == K_cached(-u_hcu), \
            f"K({u_hcu}) ≠ K({-u_hcu})"


def test_phi_monotone_decay():
    """Φ(u₁) > Φ(u₂) for 0 ≤ u₁ < u₂."""
    vals = [Phi(mpmath.mpf(u)/10) for u in range(0, 21)]
    for i in range(len(vals)-1):
        assert vals[i] > vals[i+1], \
            f"Φ({i/10}) = {vals[i]} ≤ Φ({(i+1)/10}) = {vals[i+1]}"


def test_det3x3_manual():
    """Verify 3×3 determinant formula against mpmath.matrix.det."""
    A = mpmath.matrix([
        [K_cached(0), K_cached(5), K_cached(10)],
        [K_cached(5), K_cached(0), K_cached(5)],
        [K_cached(10), K_cached(5), K_cached(0)],
    ])
    det_mp = mpmath.det(A)
    # Compute via our d3 formula
    M = [[K_cached(abs(5*i - 5*j)) for j in range(3)] for i in range(3)]
    det_manual = (M[0][0]*(M[1][1]*M[2][2] - M[1][2]*M[2][1])
                 -M[0][1]*(M[1][0]*M[2][2] - M[1][2]*M[2][0])
                 +M[0][2]*(M[1][0]*M[2][1] - M[1][1]*M[2][0]))
    rel_err = abs(det_mp - det_manual) / abs(det_mp) if det_mp != 0 else abs(det_manual)
    assert rel_err < mpmath.mpf('1e-50'), \
        f"3×3 det mismatch: mpmath={det_mp}, manual={det_manual}, rel_err={rel_err}"


def test_L4_toeplitz_positive():
    """L₄ at Toeplitz corner (δ,...,δ,0) is positive."""
    L = L4(5, 5, 5, 5, 5, 5, 0)
    assert L is not None and L > 0, f"L₄(δ,...,δ,0) = {L}"


def test_L4_toeplitz_value():
    """L₄ at Toeplitz corner ≈ 0.3077 (known grid minimum)."""
    L = L4(5, 5, 5, 5, 5, 5, 0)
    assert abs(float(L) - 0.3077) < 0.001, \
        f"L₄(δ,...,δ,0) = {float(L)}, expected ≈ 0.3077"


def test_L4_symmetric_in_s():
    """L₄ is symmetric in s: L₄(..., s) = L₄(..., -s) at Toeplitz."""
    L_pos = L4(5, 5, 5, 5, 5, 5, 20)
    L_neg = L4(5, 5, 5, 5, 5, 5, -20)
    assert L_pos is not None and L_neg is not None
    rel = abs(L_pos - L_neg) / abs(L_pos)
    assert rel < mpmath.mpf('1e-50'), \
        f"L₄ asymmetric: L(+s)={L_pos}, L(-s)={L_neg}"


def test_adaptive_truncation_large_u():
    """Φ(u) is nonzero for large u (where fixed cutoff t>200 would fail)."""
    for u in [1.5, 2.0, 3.0]:
        val = Phi(mpmath.mpf(u))
        assert val > 0, f"Φ({u}) = {val} (should be > 0; fixed cutoff bug?)"


def test_no_zero_in_bug_zone():
    """Regression: Φ(u) > 0 on [1.05, 3.0] where fixed cutoff t>200 killed terms.

    The old bug: πn²e^{4u} > 200 cutoff kills dominant n=1 term for u > 1.04.
    This test scans the danger zone at 20 points to prevent regression.
    """
    for i in range(20):
        u = mpmath.mpf("1.05") + mpmath.mpf(i) * mpmath.mpf("0.1")
        val = Phi(u)
        assert val > 0, f"Φ({float(u):.2f}) = {val} (zero in bug zone!)"


def test_phi_positive_at_critical_points():
    """Φ at u = 1.04, 1.1, 1.5 — the boundary of the old cutoff bug."""
    for u in [1.04, 1.1, 1.5]:
        val = Phi(mpmath.mpf(u))
        assert val > 0, f"Φ({u}) = {val} ≤ 0 (near cutoff threshold)"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"  ✓ {t.__name__}")
            passed += 1
        except AssertionError as e:  # noqa: F821
            print(f"  ✗ {t.__name__}: {e}")
    print(f"\n{passed}/{len(tests)} tests passed")
