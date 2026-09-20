#!/bin/bash
# msprof analysis of the AddRelu benchmark: AIV time / vec ratio / MTE.
# Prereq: source set_env.sh; export ASCEND_HOME_PATH=<staging root>; (old compiler) __nop workaround applied.
set -eu

HERE="$(cd "$(dirname "$0")" && pwd)"
BUILD="${BUILD:-/tmp/perfbuild}"
OUT="${OUT:-/tmp/prof_bench}"
mkdir -p "$BUILD" && cd "$BUILD"
cmake "$HERE/perf" -DCMAKE_ASC_ARCHITECTURES=dav-3510 >/dev/null
cmake --build . >/dev/null
rm -rf "$OUT"
msprof --output="$OUT" --ai-core=on --task-time=l1 ./demo >/tmp/msprof_bench.log 2>&1

SUMMARY=$(find "$OUT" -path "*mindstudio_profiler_output*" -name "op_summary_*.csv" | head -1)
echo "== op_summary ($SUMMARY) =="
python3 - "$SUMMARY" <<'PY'
import csv, sys
with open(sys.argv[1]) as f:
    for row in csv.DictReader(f):
        name = row['Op Name']
        tag = 'pointer' if 'ELb1EE' in name else 'tensor'
        print(f"{tag:8s} duration={row['Task Duration(us)']:>9} us  "
              f"aiv_time={row['aiv_time(us)']:>8} us  aiv_total_cycles={row['aiv_total_cycles']:>8}  "
              f"aiv_vec_ratio={row['aiv_vec_ratio']:>6}  aiv_scalar_ratio={row['aiv_scalar_ratio']:>6}  "
              f"mte2={row['aiv_mte2_time(us)']:>6} mte3={row['aiv_mte3_time(us)']:>6}")
PY
