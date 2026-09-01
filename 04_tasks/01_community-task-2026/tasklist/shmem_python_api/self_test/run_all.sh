#!/usr/bin/env bash

set -u
set -o pipefail

SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_DIR="$SELF_DIR/tests"
LOG_DIR="$SELF_DIR/logs"

SHMEM_REPO="${SHMEM_REPO:-/workspace/shmem}"
VENV="${SHMEM_VENV:-/workspace/.venvs/shmem-python-api}"
CANN_ENV="${CANN_ENV:-/usr/local/Ascend/cann-9.1.0-beta.3/set_env.sh}"

mkdir -p "$LOG_DIR"

echo "============================================================"
echo "SHMEM PYTHON API FORMAL SELF-TEST"
echo "============================================================"

echo "SHMEM_REPO=$SHMEM_REPO"
echo "TEST_DIR=$TEST_DIR"
echo "LOG_DIR=$LOG_DIR"

if [ ! -d "$SHMEM_REPO" ]; then
    echo "ERROR: SHMEM_REPO not found: $SHMEM_REPO"
    exit 1
fi

if [ ! -f "$VENV/bin/activate" ]; then
    echo "ERROR: venv not found: $VENV"
    exit 1
fi

if [ ! -f "$CANN_ENV" ]; then
    echo "ERROR: CANN environment script not found: $CANN_ENV"
    exit 1
fi

source "$VENV/bin/activate"
source "$CANN_ENV"

unset ASCEND_VISIBLE_DEVICES
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1}"
export SHMEM_INSTANCE_PORT_RANGE="${SHMEM_INSTANCE_PORT_RANGE:-19000:19020}"

# Avoid simultaneously loading source-tree and wheel copies of libshmem.so.
export LD_LIBRARY_PATH="$(
    printf '%s' "${LD_LIBRARY_PATH:-}" |
    tr ':' '\n' |
    grep -v "^${SHMEM_REPO}/build/lib$" |
    awk 'NF' |
    paste -sd: -
)"

echo
echo "ASCEND_RT_VISIBLE_DEVICES=$ASCEND_RT_VISIBLE_DEVICES"
echo "SHMEM_INSTANCE_PORT_RANGE=$SHMEM_INSTANCE_PORT_RANGE"

echo
echo "============================================================"
echo "PYTHON SYNTAX CHECK"
echo "============================================================"

for f in "$TEST_DIR"/*.py; do
    echo "CHECK: $f"
    python -m py_compile "$f" || exit 1
done

PASS_COUNT=0
FAIL_COUNT=0

run_case()
{
    NAME="$1"
    SCRIPT="$2"

    LOG="$LOG_DIR/${NAME}.log"
    RCFILE="$LOG_DIR/${NAME}.rc"

    echo
    echo "============================================================"
    echo "RUN: $NAME"
    echo "============================================================"

    timeout 300s \
    torchrun \
        --standalone \
        --nproc-per-node=2 \
        "$SCRIPT" \
        2>&1 | tee "$LOG"

    RC=${PIPESTATUS[0]}
    echo "$RC" > "$RCFILE"

    echo "${NAME}_RC=$RC"

    if [ "$RC" -eq 0 ]; then
        echo "$NAME: PASS"
        PASS_COUNT=$((PASS_COUNT + 1))
    else
        echo "$NAME: FAIL"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
}

run_case \
    putmem_on_stream_bitexact \
    "$TEST_DIR/test_putmem_on_stream_bitexact.py"

run_case \
    multi_instance_create_switch \
    "$TEST_DIR/test_multi_instance_create_switch.py"

run_case \
    multi_instance_heap_isolation \
    "$TEST_DIR/test_multi_instance_heap_isolation.py"

run_case \
    sdma_put_stream \
    "$TEST_DIR/test_sdma_put_stream.py"

run_case \
    core_wrappers_hccl_fixed \
    "$TEST_DIR/test_core_wrappers_hccl_fixed.py"

echo
echo "============================================================"
echo "SELF-TEST SUMMARY"
echo "============================================================"

echo "PASS_COUNT=$PASS_COUNT"
echo "FAIL_COUNT=$FAIL_COUNT"

echo
echo "Full RDMA-dependent handle_wait is NOT counted as PASS."
echo "Current status: BLOCKED by validation environment."

if [ "$FAIL_COUNT" -eq 0 ]; then
    echo
    echo "SUPPORTED_SELF_TESTS_ALL_PASS"
    exit 0
else
    echo
    echo "SUPPORTED_SELF_TESTS_FAILED"
    exit 1
fi
