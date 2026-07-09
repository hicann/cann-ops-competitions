#!/usr/bin/env bash
# Unified ATK runner for aclSparse SDDMM / SpGEMM / SpSM.
#
# Usage:
#   ./run_sparse_atk.sh <sddmm|spgemm|spsm> [backend] [--gen] [extra atk task args...]
#
# backend: auto | npu | gpu | cpu   (default: auto)
#   auto  - prefer npu, then gpu, then cpu
#   npu   - NPU under test + CPU golden (cv_fused_double_benchmark)
#   gpu   - GPU under test + CPU golden
#   cpu   - CPU-only smoke (high-precision reference path)
#
# Examples:
#   ./run_sparse_atk.sh sddmm auto --gen
#   ./run_sparse_atk.sh spgemm gpu -s 0 -e 10
#   ./run_sparse_atk.sh spsm npu -rn 5
set -euo pipefail

ATK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OP="${1:?Usage: $0 <sddmm|spgemm|spsm> [backend] [--gen] [...]}"
BACKEND="${2:-auto}"
shift 2 || true

GEN_CASES=0
EXTRA_ARGS=()
for arg in "$@"; do
  if [[ "$arg" == "--gen" ]]; then
    GEN_CASES=1
  else
    EXTRA_ARGS+=("$arg")
  fi
done

case "${OP,,}" in
  sddmm)
    OP_DIR="aclSparseSddmm"
    YAML="aclSparseSddmm.yaml"
    GENERATOR="generator_aclSparseSddmm.py"
    EXECUTOR="executor_aclSparseSddmm.py"
    JSON="result/aclSparseSddmm/json/all_aclSparseSddmm.json"
    ;;
  spgemm)
    OP_DIR="aclSparseSpgemm"
    YAML="aclSparseSpgemm.yaml"
    GENERATOR="generator_aclSparseSpgemm.py"
    EXECUTOR="executor_aclSparseSpgemm.py"
    JSON="result/aclSparseSpgemm/json/all_aclSparseSpgemm.json"
    ;;
  spsm)
    OP_DIR="aclSparseSpsm"
    YAML="aclSparseSpsm.yaml"
    GENERATOR="generator_aclSparseSpsm.py"
    EXECUTOR="executor_aclSparseSpsm.py"
    JSON="result/aclSparseSpsm/json/all_aclSparseSpsm.json"
    ;;
  *)
    echo "Unknown operator: $OP (expected sddmm|spgemm|spsm)" >&2
    exit 1
    ;;
esac

WORKDIR="${ATK_ROOT}/${OP_DIR}"
cd "$WORKDIR"

detect_backend() {
  if python3 - <<'PY' 2>/dev/null; then echo npu; return; fi
import torch
import torch_npu
raise SystemExit(0 if torch.npu.is_available() else 1)
PY
  if python3 - <<'PY' 2>/dev/null; then echo gpu; return; fi
import torch
raise SystemExit(0 if torch.cuda.is_available() else 1)
PY
  echo cpu
}

if [[ "$BACKEND" == "auto" ]]; then
  BACKEND="$(detect_backend)"
fi

NODES_FILE="${WORKDIR}/nodes_accu_${BACKEND}.yaml"
case "$BACKEND" in
  npu)
    cat > "$NODES_FILE" <<'EOF'
nodes:
  - backend: npu
    task: ['accuracy']
    devices: [0]
  - backend: gpu
    task: ['accuracy']
EOF
    BM_DEVICE="CPU"
    ;;
  gpu)
    cat > "$NODES_FILE" <<'EOF'
nodes:
  - backend: gpu
    task: ['accuracy']
EOF
    BM_DEVICE="CPU"
    ;;
  cpu)
    cat > "$NODES_FILE" <<'EOF'
nodes:
  - backend: cpu
    task: ['accuracy']
EOF
    BM_DEVICE="CPU"
    ;;
  *)
    echo "Unknown backend: $BACKEND" >&2
    exit 1
    ;;
esac

if [[ "$GEN_CASES" -eq 1 ]] || [[ ! -f "$JSON" ]]; then
  echo "[run_sparse_atk] generating cases: atk case -f $YAML -p $GENERATOR"
  atk case -f "$YAML" -p "$GENERATOR"
fi

if [[ ! -f "$JSON" ]]; then
  echo "Case json not found: $JSON" >&2
  exit 1
fi

# Ensure executor can import shared helpers under ATK/
export PYTHONPATH="${ATK_ROOT}:${PYTHONPATH:-}"

echo "[run_sparse_atk] operator=$OP backend=$BACKEND nodes=$NODES_FILE"
echo "[run_sparse_atk] running: atk task -c $JSON -n $NODES_FILE -p $EXECUTOR -bd $BM_DEVICE ${EXTRA_ARGS[*]:-<none>}"

atk task \
  -c "$JSON" \
  -n "$NODES_FILE" \
  -p "$EXECUTOR" \
  -bd "$BM_DEVICE" \
  "${EXTRA_ARGS[@]}"
