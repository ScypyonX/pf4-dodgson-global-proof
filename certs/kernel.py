"""
Canonical de Bruijn-Newman kernel and derived quantities.
==========================================================

All certificates import from this single module to guarantee
kernel identity across the proof.

Kernel: K(u) = Phi(|u|), where Phi is defined by the theta-type
series (1). Adaptive truncation: series terminates when
|term/sum| < 10^{-(dps+10)}.

Units: certificates use two internal unit systems:
  - centiunits (cu): 1 cu = 0.01 real  (used in master cert)
  - half-centiunits (hcu): 1 hcu = 0.005 real  (used in Taylor bridge,
    shift tail, coalescence)

This module provides caching wrappers for both unit systems.
"""

import mpmath
from config import DPS, N_SERIES_MAX

mpmath.mp.dps = DPS
PI = mpmath.pi


# ═══════════════════════════════════════════════════════════════════
# Core kernel functions
# ═══════════════════════════════════════════════════════════════════

def Phi(u):
    """De Bruijn-Newman kernel Phi(u), u >= 0. Adaptive truncation."""
    u = mpmath.mpf(u)
    e4u = mpmath.exp(4*u)
    e5u = mpmath.exp(5*u)
    e9u = mpmath.exp(9*u)
    result = mpmath.mpf(0)
    for n in range(1, N_SERIES_MAX):
        n2 = mpmath.mpf(n*n)
        n4 = mpmath.mpf(n**4)
        t = PI * n2 * e4u
        term = (2*PI**2*n4*e9u - 3*PI*n2*e5u) * mpmath.exp(-t)
        result += term
        if n > 1 and result != 0 and term != 0:
            if mpmath.fabs(term/result) < mpmath.power(10, -(mpmath.mp.dps+10)):
                break
    if result <= 0:
        raise ValueError(f"Phi({mpmath.nstr(u, 6)}) = {result} <= 0 — precision failure or bug")
    return result


def Phi_prime(u):
    """First derivative Phi'(u)."""
    u = mpmath.mpf(u)
    e4u = mpmath.exp(4*u)
    e5u = mpmath.exp(5*u)
    e9u = mpmath.exp(9*u)
    result = mpmath.mpf(0)
    for n in range(1, N_SERIES_MAX):
        n2 = mpmath.mpf(n*n)
        n4 = mpmath.mpf(n**4)
        A = 2*PI**2*n4
        B = 3*PI*n2
        val = A*e9u - B*e5u
        dval = 9*A*e9u - 5*B*e5u
        dt = 4*PI*n2*e4u
        t = PI*n2*e4u
        term = (dval - val*dt) * mpmath.exp(-t)
        result += term
        if n > 1 and result != 0 and term != 0:
            if mpmath.fabs(term/result) < mpmath.power(10, -(mpmath.mp.dps+10)):
                break
    return result


def Phi_double_prime(u):
    """Second derivative Phi''(u)."""
    u = mpmath.mpf(u)
    e4u = mpmath.exp(4*u)
    e5u = mpmath.exp(5*u)
    e9u = mpmath.exp(9*u)
    result = mpmath.mpf(0)
    for n in range(1, N_SERIES_MAX):
        n2 = mpmath.mpf(n*n)
        n4 = mpmath.mpf(n**4)
        A = 2*PI**2*n4
        B = 3*PI*n2
        val = A*e9u - B*e5u
        dval = 9*A*e9u - 5*B*e5u
        d2val = 81*A*e9u - 25*B*e5u
        dt = 4*PI*n2*e4u
        d2t = 16*PI*n2*e4u
        t = PI*n2*e4u
        term = (d2val - 2*dval*dt + val*(dt*dt - d2t)) * mpmath.exp(-t)
        result += term
        if n > 1 and result != 0 and term != 0:
            if mpmath.fabs(term/result) < mpmath.power(10, -(mpmath.mp.dps+10)):
                break
    return result


# ═══════════════════════════════════════════════════════════════════
# Caching wrappers
# ═══════════════════════════════════════════════════════════════════

_phi_cache = {}


def K_cached_cu(u_cu):
    """K(u) with u in centiunits (1 cu = 0.01 real). Cached."""
    key = abs(u_cu)
    if key not in _phi_cache:
        _phi_cache[key] = Phi(mpmath.mpf(key) / 100)
    return _phi_cache[key]


def K_cached_hcu(u_hcu):
    """K(u) with u in half-centiunits (1 hcu = 0.005 real). Cached."""
    key = abs(u_hcu)
    if key not in _phi_cache:
        _phi_cache[key] = Phi(mpmath.mpf(key) / 200)
    return _phi_cache[key]


def cache_size():
    """Return current cache size."""
    return len(_phi_cache)


def clear_cache():
    """Clear the Phi cache."""
    _phi_cache.clear()


# ═══════════════════════════════════════════════════════════════════
# L4 Dodgson log-ratio
# ═══════════════════════════════════════════════════════════════════

def _det3(A, rows, cols):
    """3x3 determinant of A[rows, cols]."""
    r, c = rows, cols
    return (A[r[0]][c[0]] * (A[r[1]][c[1]]*A[r[2]][c[2]] - A[r[1]][c[2]]*A[r[2]][c[1]])
          - A[r[0]][c[1]] * (A[r[1]][c[0]]*A[r[2]][c[2]] - A[r[1]][c[2]]*A[r[2]][c[0]])
          + A[r[0]][c[2]] * (A[r[1]][c[0]]*A[r[2]][c[1]] - A[r[1]][c[1]]*A[r[2]][c[0]]))


def L3_cu(a1, a2, b1, b2, s):
    """Dodgson log-ratio L3, all args in centiunits. Returns mpf or None."""
    x = [0, a1, a1+a2]
    y = [s, s+b1, s+b1+b2]
    A = [[K_cached_cu(x[i]-y[j]) for j in range(3)] for i in range(3)]
    # 2x2 minors: Delta_{33} = rows{0,1},cols{0,1}, etc.
    D33 = A[0][0]*A[1][1] - A[0][1]*A[1][0]
    D11 = A[1][1]*A[2][2] - A[1][2]*A[2][1]
    D31 = A[0][1]*A[1][2] - A[0][2]*A[1][1]
    D13 = A[1][0]*A[2][1] - A[1][1]*A[2][0]
    if D33 > 0 and D11 > 0 and D31 > 0 and D13 > 0:
        return mpmath.log(D33) + mpmath.log(D11) - mpmath.log(D31) - mpmath.log(D13)
    return None


def L4_cu(a1, a2, a3, b1, b2, b3, s):
    """Dodgson log-ratio L4, all args in centiunits. Returns mpf or None."""
    x = [0, a1, a1+a2, a1+a2+a3]
    y = [s, s+b1, s+b1+b2, s+b1+b2+b3]
    A = [[K_cached_cu(x[i]-y[j]) for j in range(4)] for i in range(4)]
    D44 = _det3(A, [0,1,2], [0,1,2])
    D11 = _det3(A, [1,2,3], [1,2,3])
    D41 = _det3(A, [0,1,2], [1,2,3])
    D14 = _det3(A, [1,2,3], [0,1,2])
    if D44 > 0 and D11 > 0 and D41 > 0 and D14 > 0:
        return mpmath.log(D44) + mpmath.log(D11) - mpmath.log(D41) - mpmath.log(D14)
    return None


def L4_hcu(a1, a2, a3, b1, b2, b3, s):
    """Dodgson log-ratio L4, all args in half-centiunits. Returns mpf or None."""
    x = [0, a1, a1+a2, a1+a2+a3]
    y = [s, s+b1, s+b1+b2, s+b1+b2+b3]
    A = [[K_cached_hcu(x[i]-y[j]) for j in range(4)] for i in range(4)]
    D44 = _det3(A, [0,1,2], [0,1,2])
    D11 = _det3(A, [1,2,3], [1,2,3])
    D41 = _det3(A, [0,1,2], [1,2,3])
    D14 = _det3(A, [1,2,3], [0,1,2])
    if D44 > 0 and D11 > 0 and D41 > 0 and D14 > 0:
        return mpmath.log(D44) + mpmath.log(D11) - mpmath.log(D41) - mpmath.log(D14)
    return None


def det4_hcu(a1, a2, a3, b1, b2, b3, s):
    """Full 4x4 determinant, all args in half-centiunits. Returns mpf."""
    x = [0, a1, a1+a2, a1+a2+a3]
    y = [s, s+b1, s+b1+b2, s+b1+b2+b3]
    A = [[K_cached_hcu(x[i]-y[j]) for j in range(4)] for i in range(4)]
    d = (A[0][0] * _det3(A, [1,2,3], [1,2,3])
       - A[0][1] * _det3(A, [1,2,3], [0,2,3])
       + A[0][2] * _det3(A, [1,2,3], [0,1,3])
       - A[0][3] * _det3(A, [1,2,3], [0,1,2]))
    return d


# ═══════════════════════════════════════════════════════════════════
# Interval-arithmetic enclosures (Tucker-certified)
# ═══════════════════════════════════════════════════════════════════
#
# All functions below use mpmath.iv (outward rounding) at the DPS
# set by the caller (typically 80 digits).
#
# Monotonicity of Phi (Phi'(u) < 0 for u > 0, proved analytically
# in E0) is used to convert point evaluations into tight enclosures
# on interval inputs, avoiding the exponential dependency blowup
# that direct interval Phi evaluation would cause.
# ═══════════════════════════════════════════════════════════════════

_phi_iv_cache = {}


def clear_iv_cache():
    """Clear the interval Phi cache."""
    _phi_iv_cache.clear()


def iv_cache_size():
    """Return current interval cache size."""
    return len(_phi_iv_cache)


def _Phi_iv_point(u_iv):
    """Phi at a single (degenerate) interval point with outward rounding.

    u_iv: ivmpf (degenerate or near-degenerate)
    Returns: narrow ivmpf rigorously enclosing true Phi(u).
    Operates at current mpmath.mp.dps.
    Raises ValueError if rigorous tail bound cannot be established.

    All terms of the Phi series are positive:
      t_n = pi*n^2*e^{5u}*(2*pi*n^2*e^{4u} - 3) * exp(-pi*n^2*e^{4u})
    For u >= 0, n >= 1: 2*pi*n^2*e^{4u} >= 2*pi > 3, so t_n > 0.

    The ratio |t_{n+1}|/|t_n| decreases monotonically for
    sufficiently large n (dominated by exp(-pi*(2n+1)*e^{4u})).
    Once the ratio q = t_N/t_{N-1} satisfies q < 1 (verified in IV),
    the tail R_N = sum_{k>N} t_k satisfies:
      0 < R_N <= t_N * q / (1 - q)    (geometric series bound)

    The loop terminates ONLY when this rigorous condition holds.
    The enclosure is [partial_sum.a, partial_sum.b + tail_ub].
    """
    IV = mpmath.iv
    e4u = IV.exp(4 * u_iv)
    e5u = IV.exp(5 * u_iv)
    e9u = IV.exp(9 * u_iv)
    PI_iv = IV.pi
    result = IV.mpf(0)
    prev_abs_term = None
    last_abs_term = None
    tail_certified = False
    for n in range(1, N_SERIES_MAX):
        n2 = IV.mpf(n * n)
        n4 = IV.mpf(n**4)
        t = PI_iv * n2 * e4u
        term = (2 * PI_iv**2 * n4 * e9u - 3 * PI_iv * n2 * e5u) * IV.exp(-t)
        result += term
        prev_abs_term = last_abs_term
        last_abs_term = abs(term)

        # Rigorous stop: require q_ub < 1 AND tail_ub negligible vs result
        if n > 1 and prev_abs_term is not None and prev_abs_term.a > 0:
            q_ub = last_abs_term.b / prev_abs_term.a
            if q_ub < 1 and result.a > 0:
                tail_ub = last_abs_term.b * q_ub / (1 - q_ub)
                if tail_ub < result.a * mpmath.power(10, -(mpmath.mp.dps - 5)):
                    # Tail is negligible vs partial sum — certify and stop
                    result = mpmath.mpi(result.a, result.b + tail_ub)
                    tail_certified = True
                    break

    if not tail_certified:
        raise ValueError(
            f"_Phi_iv_point: rigorous tail bound not achieved after "
            f"{N_SERIES_MAX} terms at u={mpmath.nstr(u_iv.mid, 6)}"
        )

    return result


def _Phi_iv_cached(u_iv):
    """Cached interval Phi at a (degenerate) interval point."""
    key = mpmath.nstr(u_iv.mid, 50)
    if key not in _phi_iv_cache:
        _phi_iv_cache[key] = _Phi_iv_point(u_iv)
    return _phi_iv_cache[key]


def K_iv(diff_real_iv):
    """Interval kernel K = Phi(|u|) on interval u in REAL coordinates.

    Uses monotonicity: Phi'(u) < 0 for u > 0 (proved in E0).
    Phi([a,b]) ⊆ [Phi(b), Phi(a)]  for 0 ≤ a ≤ b.

    diff_real_iv: ivmpf interval of (x_i - y_j) in real units.
    Returns: ivmpf enclosing K(diff) for all diff in interval.
    """
    IV = mpmath.iv
    u_abs = abs(diff_real_iv)   # |u| ≥ 0
    u_lo = u_abs.a              # lower bound (degenerate ivmpf)
    u_hi = u_abs.b              # upper bound (degenerate ivmpf)

    # Monotonicity: Phi(u_lo) ≥ Phi(u) ≥ Phi(u_hi) for u ∈ [u_lo, u_hi]
    phi_at_lo = _Phi_iv_cached(u_lo)   # larger value
    phi_at_hi = _Phi_iv_cached(u_hi)   # smaller value

    # Rigorous enclosure: [lower of phi_hi, upper of phi_lo]
    return mpmath.mpi(phi_at_hi.a, phi_at_lo.b)


def L4_iv_point_hcu(a1, a2, a3, b1, b2, b3, s):
    """Dodgson log-ratio L₄ at a POINT with interval rounding (hcu units).

    All args: mpf values (or numbers convertible to mpf) in half-centiunits.
    Uses interval arithmetic for rigorous rounding control: every arithmetic
    operation uses outward rounding, so the returned interval is guaranteed
    to enclose the true L₄ value.

    Returns: ivmpf interval enclosing L₄, or None if any minor ≤ 0.
    """
    IV = mpmath.iv
    a1_iv, a2_iv, a3_iv = IV.mpf(a1), IV.mpf(a2), IV.mpf(a3)
    b1_iv, b2_iv, b3_iv = IV.mpf(b1), IV.mpf(b2), IV.mpf(b3)
    s_iv = IV.mpf(s)

    x = [IV.mpf(0), a1_iv, a1_iv + a2_iv, a1_iv + a2_iv + a3_iv]
    y = [s_iv, s_iv + b1_iv, s_iv + b1_iv + b2_iv,
         s_iv + b1_iv + b2_iv + b3_iv]

    # 4×4 interval kernel matrix (hcu → real: divide by 200)
    hcu_to_real = IV.mpf(1) / IV.mpf(200)
    A = [[_Phi_iv_cached(abs((x[i] - y[j]) * hcu_to_real))
          for j in range(4)] for i in range(4)]

    # Four 3×3 Dodgson minors (interval arithmetic)
    D44 = _det3(A, [0, 1, 2], [0, 1, 2])
    D11 = _det3(A, [1, 2, 3], [1, 2, 3])
    D41 = _det3(A, [0, 1, 2], [1, 2, 3])
    D14 = _det3(A, [1, 2, 3], [0, 1, 2])

    # All four must be provably positive to take log
    if D44.a > 0 and D11.a > 0 and D41.a > 0 and D14.a > 0:
        return IV.log(D44) + IV.log(D11) - IV.log(D41) - IV.log(D14)

    return None
