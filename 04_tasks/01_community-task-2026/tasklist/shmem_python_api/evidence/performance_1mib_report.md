# SHMEM Python API 1 MiB Performance Validation

## Test scope

- API: `aclshmemx_putmem_on_stream`
- Payload size: 1 MiB (1048576 bytes)
- Warmup: 10 iterations
- Measured iterations: 100
- PE configurations: 2 PE, 4 PE, and 8 PE
- Repeated paired measurements: 30 runs per PE configuration
- Primary metric: PE0 Python overhead versus the same-path C++ baseline
- Overhead formula: `(Python latency - C++ latency) / C++ latency * 100%`
- Measurement order:
  - odd runs: C++ -> Python
  - even runs: Python -> C++
- Confidence interval: two-sided 95% Student-t interval
- Acceptance criterion used for this validation: 95% CI upper bound <= 5%

Both Python and C++ measurements use the same Host RMA API path, payload size,
peer pattern, warmup count, measured iteration count, and stream synchronization
boundaries.

## Stability quality control

A latency stability rule was applied independently to the raw C++ and Python
latencies before final statistical aggregation.

For each PE configuration, the median C++ latency and median Python latency were
calculated separately. A paired run was flagged when either raw latency deviated
by more than 5% from its corresponding median.

Eight unstable pairs were identified:

- 2 PE: run 5
- 4 PE: runs 17, 19, 21, 23, 25, 27, and 29
- 8 PE: no unstable pairs

The same paired measurements were rerun using the original PE configuration and
measurement order. All reruns completed successfully with C++ RC=0 and Python
RC=0.

The flagged-run list and rerun log are included as separate evidence files.

## Final results

| PE | Runs | Mean overhead | Standard deviation | 95% CI | 95% CI upper | Result |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 2 PE | 30 | -0.228% | 0.510% | [-0.418%, -0.038%] | -0.038% | PASS |
| 4 PE | 30 | 0.260% | 0.869% | [-0.064%, 0.585%] | 0.585% | PASS |
| 8 PE | 30 | 0.103% | 0.144% | [0.049%, 0.157%] | 0.157% | PASS |

All three PE configurations satisfy the 5% overhead target with substantial
margin.

The negative 2 PE point estimate is interpreted as no measurable Python wrapper
overhead within the measurement resolution, rather than as evidence that Python
accelerates the underlying C++ operation.

## Evidence files

- `performance_1mib_summary.csv`: final statistical summary
- `performance_1mib_results.csv`: final paired measurements
- `performance_1mib_flagged_runs.txt`: runs identified by the stability rule
- `performance_1mib_rerun_flagged.log`: rerun results for flagged pairs
