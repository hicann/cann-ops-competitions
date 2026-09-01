# SHMEM Python API Validation Evidence

本目录保存 SHMEM Python API 社区任务的测试证据。

## 1. Functional Logs

```text
logs/examples_python_extension_run_v2.log
logs/step2_putmem_on_stream_bitexact.log
logs/step2_multi_instance_create_switch_v4.log
logs/step2_multi_instance_heap_isolation.log
logs/step3_core_wrappers_hccl_fixed.log
```

## 2. Performance

```text
performance_acceptance_summary.txt
```

性能结果：

```text
2 PE: -4.87%
4 PE:  1.36%
8 PE:  0.31%
```

均满足：

```text
Observed Python Binding Overhead <= 5%
```

## 3. Handle Wait / RDMA Blocker

```text
step3_handle_wait_environment_blocker.txt
step3_handle_wait_failure_extract.txt
step3_roce_init_failure_extract.txt
step3_official_blockers_summary.txt
```

当前状态：

```text
Handle Wait Binding              PASS
Full RDMA Remote Completion      BLOCKED
RDMA Runtime                     BLOCKED
```

原因：

```text
HNS RDMA HCA visible in sysfs
/dev/infiniband not exposed
ROCE initialization ret=-3
MTE-only handle_wait triggers AICore 507015
```

完整 RDMA-dependent Handle Wait 不能标记为 PASS。

## 4. Formal Self-Test

正式自测试脚本：

```text
../self_test/run_all.sh
```

正式结果：

```text
PASS_COUNT=5
FAIL_COUNT=0
SUPPORTED_SELF_TESTS_ALL_PASS
SELF_TEST_OVERALL_RC=0
```
