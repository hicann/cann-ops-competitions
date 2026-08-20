#!/usr/bin/env bash
# This script must keep LF line endings for Linux CI.
set -euo pipefail

if [ -n "${BASE_LIBS_PATH:-}" ]; then
    export ASCEND_HOME_PATH="${BASE_LIBS_PATH}"
elif [ -n "${ASCEND_HOME_PATH:-}" ]; then
    export ASCEND_HOME_PATH
elif [ -n "${ASCEND_AICPU_PATH:-}" ]; then
    export ASCEND_HOME_PATH="${ASCEND_AICPU_PATH}"
else
    echo "Please set BASE_LIBS_PATH, ASCEND_HOME_PATH, or ASCEND_AICPU_PATH."
    exit 1
fi

echo "Using ASCEND_HOME_PATH: ${ASCEND_HOME_PATH}"
SCRIPT_PATH="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_PATH}/build_out"
TARGET="${1:-package}"
JOBS="${BUILD_JOBS:-$(nproc)}"

cmake -S "${SCRIPT_PATH}" -B "${BUILD_DIR}" --preset=default
cmake --build "${BUILD_DIR}" --target binary -j "${JOBS}"
cmake --build "${BUILD_DIR}" --target "${TARGET}" -j "${JOBS}"
