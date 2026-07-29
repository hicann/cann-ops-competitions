#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

if [ -z "${ASCEND_HOME_PATH:-}" ]; then
  if [ -d /usr/local/Ascend/ascend-toolkit/8.3.RC1 ]; then
    export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/8.3.RC1
  else
    export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest
  fi
fi
SELECTED_ASCEND_HOME_PATH="${ASCEND_HOME_PATH}"

clean_conflicting_ascend_paths() {
  local name="$1"
  local value="${!name:-}"
  local filtered=""
  local entry

  if [ -z "${value}" ]; then
    return
  fi

  IFS=':' read -r -a entries <<< "${value}"
  for entry in "${entries[@]}"; do
    case "${entry}" in
      /usr/local/Ascend/cann|/usr/local/Ascend/cann/*|/usr/local/Ascend/cann-*|/usr/local/Ascend/cann-*/*|/usr/local/Ascend/ascend-toolkit/latest|/usr/local/Ascend/ascend-toolkit/latest/*)
        continue
        ;;
    esac
    if [ -z "${filtered}" ]; then
      filtered="${entry}"
    else
      filtered="${filtered}:${entry}"
    fi
  done

  export "${name}=${filtered}"
}

prepend_existing_paths() {
  local name="$1"
  shift
  local current="${!name:-}"
  local prefix=""
  local entry

  for entry in "$@"; do
    if [ -d "${entry}" ]; then
      if [ -z "${prefix}" ]; then
        prefix="${entry}"
      else
        prefix="${prefix}:${entry}"
      fi
    fi
  done

  if [ -n "${prefix}" ]; then
    if [ -n "${current}" ]; then
      export "${name}=${prefix}:${current}"
    else
      export "${name}=${prefix}"
    fi
  fi
}

load_ascend_environment() {
  local env_script="$1"
  local env_entry

  while IFS= read -r -d '' env_entry; do
    case "${env_entry}" in
      BASH_FUNC_*)
        continue
        ;;
    esac
    export "${env_entry}"
  done < <(bash -c 'set +eu; source "$1" >/dev/null 2>&1 || true; env -0' bash "${env_script}")
}

if [ "${SELECTED_ASCEND_HOME_PATH}" != "/usr/local/Ascend/ascend-toolkit/latest" ]; then
  clean_conflicting_ascend_paths PATH
  clean_conflicting_ascend_paths LD_LIBRARY_PATH
  clean_conflicting_ascend_paths PYTHONPATH
fi

if [ -f "${ASCEND_HOME_PATH}/set_env.sh" ]; then
  load_ascend_environment "${ASCEND_HOME_PATH}/set_env.sh"
elif [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
  load_ascend_environment /usr/local/Ascend/ascend-toolkit/set_env.sh
fi

export ASCEND_HOME_PATH="${SELECTED_ASCEND_HOME_PATH}"
if [ "${ASCEND_HOME_PATH}" != "/usr/local/Ascend/ascend-toolkit/latest" ]; then
  clean_conflicting_ascend_paths PATH
  clean_conflicting_ascend_paths LD_LIBRARY_PATH
  clean_conflicting_ascend_paths PYTHONPATH
fi
prepend_existing_paths PATH \
  "${ASCEND_HOME_PATH}/bin" \
  "${ASCEND_HOME_PATH}/python/site-packages/bin" \
  "${ASCEND_HOME_PATH}/compiler/ccec_compiler/bin" \
  "${ASCEND_HOME_PATH}/tools/profiler/bin"
prepend_existing_paths LD_LIBRARY_PATH \
  "${ASCEND_HOME_PATH}/lib64" \
  "${ASCEND_HOME_PATH}/lib64/plugin/opskernel" \
  "${ASCEND_HOME_PATH}/lib64/plugin/nnengine" \
  "${ASCEND_HOME_PATH}/opp/built-in/op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64"
prepend_existing_paths PYTHONPATH \
  "${ASCEND_HOME_PATH}/python/site-packages" \
  "${ASCEND_HOME_PATH}/opp/built-in/op_impl/ai_core/tbe"

if ! command -v cmake >/dev/null 2>&1; then
  if [ -x /root/miniconda3/envs/llm_test/bin/cmake ]; then
    export PATH="/root/miniconda3/envs/llm_test/bin:${PATH}"
  fi
fi

echo "[GeluV2] ASCEND_HOME_PATH=${ASCEND_HOME_PATH}"
echo "[GeluV2] cmake=$(command -v cmake || true)"
echo "[GeluV2] msprof=$(command -v msprof || true)"
echo "[GeluV2] ccec=$(command -v ccec || true)"

ASC_DIR_DEFAULT="${ASCEND_HOME_PATH}/aarch64-linux/tikcpp/ascendc_kernel_cmake"
cmake --preset release -DASC_DIR="${ASC_DIR:-${ASC_DIR_DEFAULT}}"
cmake --build --preset release
cmake --build "${SCRIPT_DIR}/build_out" --target binary -j "${BUILD_JOBS:-8}"
cmake --build "${SCRIPT_DIR}/build_out" --target package -j "${BUILD_JOBS:-8}"
