#!/usr/bin/env bash
# Benchmark task-book 3 fixed reference cases per op (cuSPARSE on local GPU).
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"
exec python3 run_perf_gpu.py "$@"
