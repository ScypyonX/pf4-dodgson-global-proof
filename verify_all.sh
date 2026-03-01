#!/bin/bash
# verify_all.sh — Run all PF4 proof certificates and verify results
#
# Usage: bash verify_all.sh [--skip-tiling]
#
# With --skip-tiling, the adaptive tiling (which takes ~1h) is skipped
# and only the frozen JSON is checked.

set -e
cd "$(dirname "$0")"

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'
FAIL=0

check_json() {
    local json="$1"
    local field="$2"
    local expected="$3"
    local val
    val=$(python3 -c "import json; d=json.load(open('$json')); print(d.get('$field', 'MISSING'))")
    if [ "$val" = "$expected" ]; then
        echo -e "  ${GREEN}✓${NC} $json: $field = $val"
    else
        echo -e "  ${RED}✗${NC} $json: $field = $val (expected $expected)"
        FAIL=1
    fi
}

echo "═══════════════════════════════════════════════════"
echo "PF4 Proof Verification Suite"
echo "═══════════════════════════════════════════════════"
echo

# Phase 1: Master certificate (K2+E0, PF3, dense core, boundary, Schur)
echo "[1/7] Master certificate..."
python3 certs/pf4_master_certificate.py
check_json results/master.json all_certified True
echo

# Phase 2: Boundary Taylor bridge
echo "[2/7] Taylor bridge (mini-cell)..."
python3 certs/pf4_boundary_taylor.py
check_json results/boundary_taylor.json certified True
echo

# Phase 3: Adaptive tiling
if [ "$1" = "--skip-tiling" ]; then
    echo "[3/7] Adaptive tiling (SKIPPED — checking frozen JSON)..."
else
    echo "[3/7] Adaptive tiling (this takes ~1 hour)..."
    python3 certs/pf4_continuous_tiling.py --mode full --max-depth 8
fi
check_json results/continuous_tiling.json certified True
check_json results/continuous_tiling.json mode full
echo

# Phase 4: Schur tail
echo "[4/7] Schur complement..."
python3 certs/pf4_schur_tail.py
check_json results/schur_tail.json certified True
echo

# Phase 5: Shift tail
echo "[5/7] Shift tail..."
python3 certs/pf4_shift_tail.py
check_json results/shift_tail.json certified True
echo

# Phase 6: Coalescence
echo "[6/7] Coalescence..."
python3 certs/pf4_coalescence.py
check_json results/coalescence.json certified True
echo

# Phase 7: PF5 falsification
echo "[7/7] PF5 falsification..."
python3 certs/verify_pf5.py
check_json results/pf5.json all_certified_negative True
echo

# Unit tests
echo "[bonus] Unit tests..."
python3 tests/test_kernel.py
echo

echo "═══════════════════════════════════════════════════"
if [ $FAIL -eq 0 ]; then
    echo -e "${GREEN}ALL CERTIFICATES VERIFIED ✓${NC}"
else
    echo -e "${RED}SOME CERTIFICATES FAILED ✗${NC}"
    exit 1
fi
