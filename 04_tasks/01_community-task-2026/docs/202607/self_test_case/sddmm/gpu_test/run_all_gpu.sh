#!/usr/bin/env bash
# gpu_test runner: accuracy (GPU vs CPU golden) and optional GPU perf benchmark.
#
# Accuracy (correctness):
#   ./run_all_gpu.sh                    # 200 cases × 3 ops
#   ./run_all_gpu.sh --start 0 --end 3  # first 3 fixed smoke cases per op
#
# Performance (3 fixed reference cases per op, paste --markdown into task doc):
#   ./run_perf_gpu.sh
#   ./run_perf_gpu.sh --op sddmm --markdown
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

if [[ "${1:-}" == "--perf" ]]; then
  shift
  exec "$DIR/run_perf_gpu.sh" "$@"
fi

EXTRA=("$@")
if [[ ! -f cases/sddmm_cases.json ]]; then
  echo "[run_all_gpu] cases/*.json not found, generating..."
  python3 generate_cases.py
fi

echo "========== SDDMM accuracy (200 cases) =========="
python3 run_sddmm_gpu.py "${EXTRA[@]}"
echo
echo "========== SpGEMM accuracy (200 cases) =========="
python3 run_spgemm_gpu.py "${EXTRA[@]}"
echo
echo "========== SpSM accuracy (200 cases) =========="
python3 run_spsm_gpu.py "${EXTRA[@]}"
echo
echo "All accuracy tests passed. For GPU perf: ./run_perf_gpu.sh"
