#!/bin/bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
#
# Usage:
#   bash tests/run.sh                          # run all functional + performance
#   bash tests/run.sh --test-name test_barrier_sync
#   bash tests/run.sh --nproc 4 --suite functional
#   bash tests/run.sh --skip-install           # assume shmem already installed
#
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$(readlink -f "$0")")" && pwd)
NPROC=${NPROC:-2}
SUITE=all
TEST_NAME=""
SKIP_INSTALL=0
SHMEM_ROOT=${SHMEM_ROOT:-}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --test-name)
            TEST_NAME="$2"
            shift 2
            ;;
        --nproc)
            NPROC="$2"
            shift 2
            ;;
        --suite)
            SUITE="$2"
            shift 2
            ;;
        --skip-install)
            SKIP_INSTALL=1
            shift
            ;;
        --shmem-root)
            SHMEM_ROOT="$2"
            shift 2
            ;;
        -h|--help)
            sed -n '12,20p' "$0"
            exit 0
            ;;
        *)
            echo "Unknown arg: $1"
            exit 1
            ;;
    esac
done

function ensure_cann_env()
{
    if [[ -z "${ASCEND_HOME_PATH:-}" ]]; then
        echo "[WARN] ASCEND_HOME_PATH not set. Please source CANN setenv first, e.g.:"
        echo "       source \${ASCEND_HOME_PATH}/bin/setenv.bash"
    else
        if [[ -f "${ASCEND_HOME_PATH}/bin/setenv.bash" ]]; then
            # shellcheck disable=SC1090
            source "${ASCEND_HOME_PATH}/bin/setenv.bash"
        elif [[ -f "${ASCEND_HOME_PATH}/set_env.sh" ]]; then
            # shellcheck disable=SC1090
            source "${ASCEND_HOME_PATH}/set_env.sh"
        fi
    fi
}

function install_shmem_wheel()
{
    if [[ ${SKIP_INSTALL} -eq 1 ]]; then
        echo "[INFO] --skip-install: skip wheel install"
        return 0
    fi

    if pip show shmem >/dev/null 2>&1; then
        echo "[INFO] uninstall existing shmem wheel"
        pip uninstall --yes shmem || true
    fi

    local search_roots=()
    if [[ -n "${SHMEM_ROOT}" ]]; then
        search_roots+=("${SHMEM_ROOT}")
    fi
    search_roots+=("${SCRIPT_DIR}/../../../../.." "${PWD}" "${HOME}")

    local whl=""
    for root in "${search_roots[@]}"; do
        [[ -d "${root}" ]] || continue
        whl=$(find "${root}" -name "shmem*.whl" -type f 2>/dev/null | head -n 1 || true)
        [[ -n "${whl}" ]] && break
    done

    if [[ -z "${whl}" ]]; then
        echo "[ERROR] No shmem*.whl found. Build with: (cd <shmem-root> && pip wheel .)"
        echo "        Or pass --shmem-root /path/to/shmem (or built dist)."
        exit 1
    fi
    echo "[INFO] installing ${whl}"
    pip install "${whl}"
}

function list_tests()
{
    local suite="$1"
    local files=()
    if [[ "${suite}" == "all" || "${suite}" == "functional" ]]; then
        while IFS= read -r f; do files+=("$f"); done < <(ls "${SCRIPT_DIR}/functional"/test_*.py 2>/dev/null | sort)
    fi
    if [[ "${suite}" == "all" || "${suite}" == "performance" ]]; then
        while IFS= read -r f; do files+=("$f"); done < <(ls "${SCRIPT_DIR}/performance"/test_*.py 2>/dev/null | sort)
    fi
    printf '%s\n' "${files[@]}"
}

function run_one()
{
    local pyfile="$1"
    local name
    name=$(basename "${pyfile}" .py)
    echo "======== RUN ${name} (nproc=${NPROC}) ========"
    torchrun --nproc-per-node "${NPROC}" "${pyfile}"
    echo "======== PASS ${name} ========"
}

function main()
{
    ensure_cann_env
    install_shmem_wheel

    if [[ -n "${TEST_NAME}" ]]; then
        local match=""
        match=$(find "${SCRIPT_DIR}/functional" "${SCRIPT_DIR}/performance" \
            -name "${TEST_NAME}.py" -o -name "${TEST_NAME}" 2>/dev/null | head -n 1 || true)
        if [[ -z "${match}" ]]; then
            echo "[ERROR] test not found: ${TEST_NAME}"
            echo "Available:"
            list_tests all
            exit 1
        fi
        run_one "${match}"
        echo "All selected tests passed."
        return 0
    fi

    local failed=0
    while IFS= read -r pyfile; do
        [[ -z "${pyfile}" ]] && continue
        if ! run_one "${pyfile}"; then
            failed=1
        fi
    done < <(list_tests "${SUITE}")

    if [[ ${failed} -ne 0 ]]; then
        echo "[ERROR] some tests failed"
        exit 1
    fi
    echo "All Python community tests passed!"
}

main
