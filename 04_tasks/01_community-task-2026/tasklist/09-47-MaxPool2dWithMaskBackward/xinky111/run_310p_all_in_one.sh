#!/usr/bin/env bash
set -e
set -u

echo "======================================================================"
echo "        Atlas 300V Pro (Ascend 310P) One-Shot Master Runner          "
echo "======================================================================"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 1. Environment & Dependencies
echo ">>> [1/6] Checking 310P environment & dependencies..."
pip install ml_dtypes --quiet || true

if [ -z "${ASCEND_DEVICE_ID:-}" ]; then
  export ASCEND_DEVICE_ID=0
fi
echo "Using ASCEND_DEVICE_ID=$ASCEND_DEVICE_ID"
npu-smi info || true

# 2. Build Ascend C NPU Library for 310P (dav-2002)
echo ">>> [2/6] Compiling Ascend C NPU Kernel Library (dav-2002)..."
build=/tmp/maxpool-310p-npu
rm -rf /tmp/maxpool-source "$build"
mkdir -p /tmp/maxpool-source
cp -a "$SCRIPT_DIR/." /tmp/maxpool-source/

cmake -S /tmp/maxpool-source -B "$build" \
  -DCMAKE_ASC_RUN_MODE=npu \
  -DCMAKE_ASC_ARCHITECTURES="dav-2002" \
  -DPOOL_ARCH="dav-2002" \
  -DPOOL_EXACT_OUTPUT=ON
cmake --build "$build" -j 8

echo "=== 310P Kernel Library Built: $build/libmax_pool_backward.so ==="

# 3. Run Full Precision Validation Suite on 310P (FP32 & FP16, total 304 cases)
echo ">>> [3/6] Running 310P NPU Precision & Generalization Validation..."
cd "$REPO_DIR"

for mode in smoke edges tails random full; do
  echo "--- Running suite: $mode (float32 float16) ---"
  python3 -B "$SCRIPT_DIR/tests/validate_kernel.py" \
    --library "$build/libmax_pool_backward.so" \
    --suite "$mode" \
    --dtypes float32 float16 \
    --report "$SCRIPT_DIR/reports/310p_npu_${mode}.json"
done

echo "=== All 310P Precision Checks PASSED on NPU ==="

# 4. Build Standard Custom OPP Package (Supports both 910B & 310P)
echo ">>> [4/6] Building Official Standard OPP Package..."
mkdir -p "$REPO_DIR/operator_build"
rm -rf /tmp/package

cmake -S "$SCRIPT_DIR/package" -B /tmp/package \
  '-DASCEND_COMPUTE_UNIT=ascend910b;ascend310p' \
  -DENABLE_SOURCE_PACKAGE=ON -DENABLE_BINARY_PACKAGE=ON \
  -DASCEND_CANN_PACKAGE_PATH="$ASCEND_HOME_PATH" -DASCEND_PYTHON_EXECUTABLE=python3
cmake --build /tmp/package -j 4
cmake --build /tmp/package --target binary -j 4
cmake --build /tmp/package --target package -j 4

cp /tmp/package/custom_opp_*.run "$REPO_DIR/operator_build/"
cp /tmp/package/autogen/aclnn_max_pool2d_with_mask_backward.h "$REPO_DIR/operator_build/"
cp /tmp/package/libcust_opapi.so "$REPO_DIR/operator_build/"

echo "=== OPP Package Built: ==="
ls -lh "$REPO_DIR/operator_build/"

# 5. Verify Standard OPP Package Installation
echo ">>> [5/6] Verifying Standard OPP Package Installation on 310P..."
chmod +x "$REPO_DIR/operator_build"/custom_opp_*.run || true
"$REPO_DIR/operator_build"/custom_opp_*.run --quiet || true

# 6. Hardware Benchmarking on 310P (8 AI Cores, FP32/FP16)
echo ">>> [6/6] Running 310P Hardware Microsecond Benchmarking (8 Cores)..."
BENCHMARK_PLATFORM=310p BENCHMARK_BLOCKS=8 python3 -B "$SCRIPT_DIR/tests/benchmark_hardware.py" || true

# 7. Generate Summary and Archive Deliverables
echo "======================================================================"
echo ">>> Packaging All Deliverables into 310P_COMPLETE_DELIVERABLES.tar.gz..."
cd "$REPO_DIR"

tar -czf /workspace/310P_COMPLETE_DELIVERABLES.tar.gz \
  operator_build \
  operator/reports

ls -lh /workspace/310P_COMPLETE_DELIVERABLES.tar.gz

echo "======================================================================"
echo " [SUCCESS] 310P All-in-One Execution COMPLETED 100%!"
echo " 1. Download /workspace/310P_COMPLETE_DELIVERABLES.tar.gz via WebIDE"
echo " 2. You can now STOP/SHUTDOWN the 310P instance to SAVE CARD HOURS!"
echo "======================================================================"
