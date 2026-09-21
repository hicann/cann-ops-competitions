#!/usr/bin/env bash
set -e
source /usr/local/Ascend/ascend-toolkit/set_env.sh
set -u

platform=${1:-a2}
suite=${2:-all}
case "$platform" in
  a2) arch=dav-2201; types="float32 float16 bfloat16" ;;
  310p) arch=dav-2002; types="float32 float16" ;;
  *) echo "Unknown platform: $platform" >&2; exit 2 ;;
esac
case "$suite" in all|smoke|edges|full|random|tails) ;; *) echo "Unknown suite: $suite" >&2; exit 2 ;; esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
build=/tmp/maxpool-${platform}-npu

echo "=== Building MaxPool2dWithMaskBackward for NPU ($platform, $arch) ==="
mkdir -p /tmp/maxpool-source
cp -a "$SCRIPT_DIR/." /tmp/maxpool-source/

cmake -S /tmp/maxpool-source -B "$build" \
  -DCMAKE_ASC_RUN_MODE=npu \
  -DCMAKE_ASC_ARCHITECTURES="$arch" \
  -DPOOL_ARCH="$arch" \
  -DPOOL_EXACT_OUTPUT=ON
cmake --build "$build" -j 8

echo "=== Build Complete: $build/libmax_pool_backward.so ==="

# Auto-detect device ID
if [ -z "${ASCEND_DEVICE_ID:-}" ]; then
  export ASCEND_DEVICE_ID=0
fi
echo "Using ASCEND_DEVICE_ID=$ASCEND_DEVICE_ID"

cd "$REPO_DIR"
for mode in smoke edges tails random full; do
  if [[ "$suite" == all || "$suite" == "$mode" ]]; then
    echo ">>> Running NPU test suite: $mode ($types) on $platform ..."
    python3 -B "$SCRIPT_DIR/tests/validate_kernel.py" \
      --library "$build/libmax_pool_backward.so" \
      --suite "$mode" \
      --dtypes $types \
      --report "$SCRIPT_DIR/reports/${platform}_npu_${mode}.json"
  fi
done

if [[ "$platform" == a2 && "$suite" == all ]]; then
  echo ">>> Running Host Tiling and Interface Validation..."
  g++ -std=c++17 -shared -fPIC "$SCRIPT_DIR/tests/geometry_bridge.cpp" -o /tmp/libpool_geometry.so
  python3 -B "$SCRIPT_DIR/tests/validate_tiling.py"
  python3 -B "$SCRIPT_DIR/tests/validate_interface.py" \
    --library "$build/libmax_pool_backward.so" \
    --report "$SCRIPT_DIR/reports/a2_npu_interface.json"
fi

echo "=== ALL NPU TESTS PASSED! ==="
