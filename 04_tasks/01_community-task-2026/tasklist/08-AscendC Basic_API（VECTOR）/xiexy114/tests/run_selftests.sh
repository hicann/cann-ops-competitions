#!/bin/bash
# Correctness self-test for the VECTOR Basic API pointer extension (Ascend950PR, dav-3510).
#
# Env:
#   TESTCASES_DIR  : task-provided test-cases dir (the folder containing element_wise_arithmetic/, cast/, ...)
#   ASC_DEVKIT     : asc-devkit repo path used to build (default /workspace/asc-devkit)
#   WORK           : build dir (default /tmp/selftest)
#
# Prereq:
#   source /usr/local/Ascend/cann-9.1.0/set_env.sh
#   export ASCEND_HOME_PATH=<staging root whose asc/ -> ${ASC_DEVKIT}>
set -u

ASC_DEVKIT="${ASC_DEVKIT:-/workspace/asc-devkit}"
WORK="${WORK:-/tmp/selftest}"
ARCH="dav-3510"
PY="${PYTHON:-python3}"

: "${TESTCASES_DIR:?please export TESTCASES_DIR to the task test-cases directory}"
[ -d "$TESTCASES_DIR" ] || { echo "TESTCASES_DIR not found: $TESTCASES_DIR"; exit 2; }
mkdir -p "$WORK"

pass=0; fail=0
run_one() { # case tag cmake_extra gen_args verify_args
    local case="$1" tag="$2" cextra="$3" gargs="$4" vargs="$5"
    local dir="$WORK/${case}${tag}"
    rm -rf "$dir" && mkdir -p "$dir" && cd "$dir" || return
    if ! cmake "$TESTCASES_DIR/$case" -DCMAKE_ASC_ARCHITECTURES=$ARCH $cextra >/tmp/st_cm.log 2>&1; then
        echo "FAIL(config) $case$tag"; fail=$((fail+1)); return; fi
    if ! timeout 300 cmake --build . >/tmp/st_b.log 2>&1; then
        echo "FAIL(build)  $case$tag"; grep -m3 -i error /tmp/st_b.log; fail=$((fail+1)); return; fi
    "$PY" "$TESTCASES_DIR/$case/scripts/gen_data.py" $gargs >/dev/null 2>&1
    ./demo >/tmp/st_r.log 2>&1
    local res
    if [ -f "$TESTCASES_DIR/$case/scripts/verify_result.py" ]; then
        res=$("$PY" "$TESTCASES_DIR/$case/scripts/verify_result.py" $vargs output/output.bin output/golden.bin 2>&1 \
              | grep -oE "test pass!|error ratio: [0-9.]+")
    else
        res=$(grep -o "test pass!" /tmp/st_r.log | tail -1)
    fi
    if echo "$res" | grep -q "test pass!"; then
        echo "PASS  $case$tag   ($res)"; pass=$((pass+1))
    else
        echo "FAIL  $case$tag   ($res)"; fail=$((fail+1))
    fi
}

echo "==================== correctness ===================="
run_one element_wise_arithmetic        ""    ""                  ""               ""
run_one element_wise_compound_compute  "_s1" "-DSCENARIO_NUM=1"  "-scenarioNum=1" ""
run_one element_wise_compound_compute  "_s2" "-DSCENARIO_NUM=2"  "-scenarioNum=2" ""
run_one create_vec_index               ""    ""                  ""               ""
run_one duplicate                      ""    ""                  ""               ""
run_one cast                           "_s0" "-DSCENARIO_NUM=0"  "-scenarioNum=0" ""
run_one cast                           "_s1" "-DSCENARIO_NUM=1"  "-scenarioNum=1" ""
run_one reduce                         ""    ""                  ""               ""
run_one reduce_repeat                  ""    ""                  ""               ""
run_one reduce_data_block              ""    ""                  ""               ""
run_one reduce_pair_elem               ""    ""                  ""               ""
for s in 1 2 3 4 5 6; do
    run_one reduce_computation "_s$s" "-DSCENARIO_NUM=$s" "-scenarioNum=$s" "-scenarioNum=$s"
done

echo "==================== summary ===================="
echo "PASS=$pass FAIL=$fail"
[ "$fail" -eq 0 ] && echo "ALL TESTS PASSED" || echo "SOME TESTS FAILED"
[ "$fail" -eq 0 ]
