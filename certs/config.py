"""
Shared configuration and reproducibility metadata for PF₄ certificates.

NOTE: Zero float() policy — all computation in mpf.
      MpfEncoder handles mpf → JSON number at the serialization boundary.
      _ns() handles mpf → string for print display.
"""

import sys
import platform
import json
import mpmath

# ═══════════════════════════════════════════════════════════════════
# Precision
# ═══════════════════════════════════════════════════════════════════
DPS = 60
mpmath.mp.dps = DPS
PI = mpmath.pi

# ═══════════════════════════════════════════════════════════════════
# Domain parameters (half-centiunits: 1 hcu = 0.005 real)
# ═══════════════════════════════════════════════════════════════════
DELTA_HCU = 10       # δ = 0.05 real
S_HCU = 80           # S = 0.40 real
G_HCU = 120          # G = 0.60 real
HCU_TO_REAL = 0.005  # 1 hcu = 0.005 real

# Series truncation
N_SERIES_MAX = 500   # maximum terms in Φ series

# ═══════════════════════════════════════════════════════════════════
# Zero-float display and serialization helpers
# ═══════════════════════════════════════════════════════════════════

class MpfEncoder(json.JSONEncoder):
    """JSON encoder: mpf → JSON number at the serialization boundary ONLY."""
    def default(self, obj):
        if isinstance(obj, mpmath.mpf):
            return obj.__float__()
        return super().default(obj)


def _ns(x, n=6):
    """mpf → string with n significant digits. Display only, never for decisions."""
    if x is None:
        return "0"
    return mpmath.nstr(x, n)


# ═══════════════════════════════════════════════════════════════════
# Reproducibility metadata
# ═══════════════════════════════════════════════════════════════════

def metadata():
    """Return dict of reproducibility metadata for JSON output."""
    return {
        "repository": "https://github.com/ScypyonX/pf4-dodgson-global-proof",
        "python_version": sys.version,
        "platform": platform.platform(),
        "mpmath_version": mpmath.__version__,
        "precision_digits": DPS,
        "n_series_max": N_SERIES_MAX,
        "hcu_to_real": HCU_TO_REAL,
        "delta_real": DELTA_HCU * HCU_TO_REAL,
        "S_real": S_HCU * HCU_TO_REAL,
        "G_real": G_HCU * HCU_TO_REAL,
    }
