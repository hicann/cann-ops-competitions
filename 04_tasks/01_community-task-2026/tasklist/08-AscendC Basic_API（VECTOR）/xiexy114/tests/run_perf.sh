#!/bin/bash
# Performance comparison: AddRelu Tensor path (baseline) vs pointer path, device cycles.
# Prereq: source set_env.sh; export ASCEND_HOME_PATH=<staging root>; (old compiler) __nop workaround applied.
set -eu

HERE="$(cd "$(dirname "$0")" && pwd)"
BUILD="${BUILD:-/tmp/perfbuild}"
mkdir -p "$BUILD" && cd "$BUILD"
cmake "$HERE/perf" -DCMAKE_ASC_ARCHITECTURES=dav-3510 >/dev/null
cmake --build . >/dev/null

echo "== AddRelu cyclic benchmark (512 half, 10000 iters) =="
for i in 1 2 3; do
    ./demo | grep -E "cycles=|speedup"
    echo "---"
done

echo
echo "== per-element cycle sweep (1000 iters, count 512..16384) =="
./demo_sweep
