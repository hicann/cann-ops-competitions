# SHMEM Python API Self-Test

## 1. 测试说明

本目录用于 SHMEM Python API 社区任务的自测试。

当前自测试覆盖：

- `aclshmemx_putmem_on_stream` bit-exact 正确性；
- Multi-instance 创建与 Context 切换；
- Multi-instance Heap 隔离；
- SDMA `putmem_on_stream` Smoke；
- `shmem.core` Barrier / Sync / Multi-instance Wrapper。

完整测试报告：

```text
../docs/self_test_report.md
```

测试证据：

```text
../evidence/
```

## 2. 测试环境

```text
OS: openEuler 24.03 LTS-SP2
Architecture: aarch64
NPU: Ascend 910B4
CANN: 9.1.0-beta.3
Python: 3.11.6
PyTorch: 2.7.1+cpu
torch_npu: 2.7.1.post8
```

默认 SHMEM 代码仓库：

```text
/workspace/shmem
```

默认 Python 虚拟环境：

```text
/workspace/.venvs/shmem-python-api
```

## 3. 一键运行

```bash
cd /workspace/cann-ops-competitions/04_tasks/01_community-task-2026/tasklist/shmem_python_api
bash self_test/run_all.sh
```

成功结果：

```text
PASS_COUNT=5
FAIL_COUNT=0
SUPPORTED_SELF_TESTS_ALL_PASS
```

退出码：

```text
0
```

## 4. 自测试文件

```text
tests/test_putmem_on_stream_bitexact.py
tests/test_multi_instance_create_switch.py
tests/test_multi_instance_heap_isolation.py
tests/test_sdma_put_stream.py
tests/test_core_wrappers_hccl_fixed.py
```

## 5. 测试内容

### 5.1 putmem_on_stream Bit-exact

验证：

```text
2 PE
256-byte deterministic pattern
cross-rank transfer
Device -> Host readback
byte-by-byte exact comparison
```

通过标志：

```text
PUTMEM_ON_STREAM_BITEXACT_OVERALL_PASS
```

### 5.2 Multi-instance Create / Switch

验证：

```text
Create instance 1
Create instance 2
Context 1 -> 2 -> 1
Finalize instance 2
Finalize instance 1
```

通过标志：

```text
MULTI_INSTANCE_CREATE_SWITCH_OVERALL_PASS
```

### 5.3 Multi-instance Heap Isolation

验证：

```text
Instance A write Pattern A
Instance B write Pattern B
A/B repeated context switch
Verify A/B data unchanged
Destroy B
Verify A remains valid
```

通过标志：

```text
MULTI_INSTANCE_HEAP_ISOLATION_OVERALL_PASS
```

### 5.4 SDMA Smoke

验证：

```text
SDMA init
malloc
putmem_on_stream
stream synchronize
finalize
```

通过标志：

```text
SDMA_INIT_OK
SDMA_PUT_STREAM_OK
SDMA_FINALIZE_OK
```

### 5.5 Core Wrapper

验证：

```text
core barrier/sync
core multi_instance
core.handle_wait symbol
```

校正版测试通过标志：

```text
test_core_wrappers_hccl_fixed.py running success!
```

## 6. Handle Wait 环境限制

当前环境中：

```text
/sys/class/infiniband/hns_0 exists
/sys/class/infiniband/hns_1 exists
/dev/infiniband does not exist
```

独立 ROCE 初始化结果：

```text
ret=-3
ACLSHMEM_SMEM_ERROR (-3)
```

MTE-only Handle Wait 完整执行路径出现：

```text
AICore error 507015
```

因此：

```text
Handle Wait Binding             PASS
Handle Wait Wrapper Surface     PASS
Full RDMA Remote Completion     BLOCKED
```

完整 RDMA-dependent Handle Wait 不能标记为 PASS。

## 7. 性能

性能数据见：

```text
../evidence/performance_acceptance_summary.txt
```

结果：

| PE | C++ | Python | Observed Overhead |
|---:|---:|---:|---:|
| 2 | 11.402 us | 10.847 us | -4.87% |
| 4 | 10.811 us | 10.958 us | 1.36% |
| 8 | 10.858 us | 10.892 us | 0.31% |

全部满足：

```text
Observed Python Binding Overhead <= 5%
```

负值仅表示本次测量没有观察到 Python 的额外开销，不表示 Python 在机制上比 C++ 更快。

## 8. 一键测试结果

正式运行：

```text
PASS_COUNT=5
FAIL_COUNT=0

SUPPORTED_SELF_TESTS_ALL_PASS

SELF_TEST_OVERALL_RC=0
```

当前环境支持的正式自测试全部通过。

完整 RDMA-dependent Handle Wait 独立记录为：

```text
BLOCKED
```
