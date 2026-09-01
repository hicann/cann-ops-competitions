# SHMEM Python API 自测试报告

## 1. 测试目的

本报告记录 SHMEM Python API 社区任务的功能正确性、RMA 数据正确性、Multi-instance 隔离、Engine Smoke、Barrier/Sync 和 Python/C++ 性能验证结果。

测试状态严格区分：

```text
PASS
BLOCKED
NOT VERIFIED
```

环境限制项不描述为 PASS。

---

## 2. 代码信息

代码仓库：

```text
cann/shmem
```

开发分支：

```text
feat/shmem-python-api
```

验证提交：

```text
b41a0b0d9c02151c9da9a60d7a39f2c7d6bfed7c
```

代码 PR：

```text
https://gitcode.com/cann/shmem/merge_requests/632
```

文档 PR：

```text
https://gitcode.com/cann/shmem/pull/666
```

关联社区 Issue：

```text
https://gitcode.com/cann/shmem/issues/489
```

---

## 3. 测试环境

```text
OS: openEuler 24.03 LTS-SP2
Architecture: aarch64
NPU: 8 × Ascend 910B4
CANN: 9.1.0-beta.3
Python: 3.11.6
PyTorch: 2.7.1+cpu
torch_npu: 2.7.1.post8
```

NPU 0-7 健康状态均为：

```text
Health = OK
```

---

## 4. Python Extension 全量测试

测试入口：

```text
examples/python_extension/run.sh
```

最终结果：

```text
test_prof running success!
test_handle_wait_binding running success!
All Python tests passed!
EXAMPLES_PYTHON_EXTENSION_FINAL_RC=0
```

状态：

```text
PASS
```

说明：

```text
test_handle_wait_binding
```

验证的是 Binding、Type、Public Export 和 Parameter Surface，不代表完整 RDMA Remote Completion。

证据：

```text
../evidence/logs/examples_python_extension_run_v2.log
```

---

## 5. Public Lazy API

顶层 Public API 已完成：

```text
__all__
Lazy Access
Native Lazy Loading
```

专项验证结果：

```text
missing_from___all__ = []
missing_or_error = []
ALL_REQUIRED_LAZY_PUBLIC_APIS_OK
```

状态：

```text
PASS
```

---

## 6. RMA 正确性

### 6.1 Core RMA

现有 Python RMA 测试覆盖：

```text
put
get
put_signal
signal_op
signal_wait
quiet
quiet_on_stream
stream put
stream get
```

测试会读取实际 Device Memory 并校验结果。

状态：

```text
PASS
```

### 6.2 putmem_on_stream Bit-exact

测试条件：

```text
2 PE
256 bytes
deterministic rank-dependent pattern
cross-rank transfer
Device -> Host readback
byte-by-byte exact comparison
```

结果：

```text
PE0: PUTMEM_ON_STREAM_BITEXACT_PASS
PE1: PUTMEM_ON_STREAM_BITEXACT_PASS
PUTMEM_ON_STREAM_BITEXACT_OVERALL_PASS
BITEXACT_RC=0
```

状态：

```text
PASS
```

证据：

```text
../evidence/logs/step2_putmem_on_stream_bitexact.log
```

---

## 7. Multi-instance 创建与 Context 切换

专项测试真实创建：

```text
instance_id=1
instance_id=2
```

结果：

```text
INSTANCE_CREATE_PASS id=1
INSTANCE_CREATE_PASS id=2
MULTI_INSTANCE_SWITCH_PASS
FINALIZE_B ret=0
FINALIZE_A ret=0
MULTI_INSTANCE_CREATE_SWITCH_OVERALL_PASS
RC=0
```

状态：

```text
PASS
```

证据：

```text
../evidence/logs/step2_multi_instance_create_switch_v4.log
```

---

## 8. Multi-instance Heap 隔离

验证流程：

```text
Create Instance A
Write Pattern A

Create Instance B
Write Pattern B

Switch B -> A
Verify Pattern A

Switch A -> B
Verify Pattern B

Repeated A/B Switch
Verify both patterns unchanged

Destroy Instance B
Switch to A
Verify A remains valid
```

结果：

```text
HEAP_A_WRITE_PASS
HEAP_B_WRITE_PASS
HEAP_A_ISOLATION_PASS
HEAP_B_ISOLATION_PASS
REPEATED_SWITCH_DATA_PASS
DESTROY_B_KEEP_A_PASS
MULTI_INSTANCE_HEAP_ISOLATION_OVERALL_PASS
RC=0
```

状态：

```text
PASS
```

证据：

```text
../evidence/logs/step2_multi_instance_heap_isolation.log
```

---

## 9. Barrier / Sync

进行了：

```text
2 PE
4 PE
8 PE
```

多卡验证。

结果：

```text
2 PE RC=0
4 PE RC=0
8 PE RC=0
```

未发现明显：

```text
Deadlock
Timeout
```

状态：

```text
PASS
```

---

## 10. Engine 验证

### 10.1 MTE

MTE 主路径完成：

```text
Heap
RMA
putmem_on_stream
Barrier
Sync
Multi-instance
```

状态：

```text
PASS
```

### 10.2 SDMA

结果：

```text
SDMA_INIT_OK
SDMA_PUT_STREAM_OK
SDMA_FINALIZE_OK
RC=0
```

状态：

```text
PASS
```

### 10.3 RDMA / ROCE

当前系统 sysfs 可见：

```text
/sys/class/infiniband/hns_0
/sys/class/infiniband/hns_1
```

但容器未暴露：

```text
/dev/infiniband
```

独立 ROCE 初始化：

```text
initialize shmem failed, ret=-3
ACLSHMEM_SMEM_ERROR (-3)
```

状态：

```text
BLOCKED
```

### 10.4 UDMA

当前环境未形成正式 Runtime 验证结论。

状态：

```text
NOT VERIFIED
```

---

## 11. Core Wrapper

原始归档 `test_core_wrappers.py` 使用：

```text
HCCL ProcessGroup
+
CPU Tensor UID broadcast
```

测试会在真正进入 SHMEM Wrapper 功能前发生 Backend / Device 不匹配。

校正版将 UID Broadcast Tensor 放置到 NPU。

结果：

```text
core.init ok
core.barrier_on_stream ok
core barrier/sync wrappers ok
core multi_instance ok
core.handle_wait symbol ok
test_core_wrappers_hccl_fixed.py running success!
CORE_WRAPPERS_FIXED_RC=0
```

状态：

```text
CORRECTED VALIDATION PASS
```

说明：

这表示校正版 Wrapper 功能测试通过，不表示原始归档测试文件未经修改即 PASS。

证据：

```text
../evidence/logs/step3_core_wrappers_hccl_fixed.log
```

---

## 12. Handle Wait

### 12.1 Binding / Wrapper

已经验证：

```text
Handle Python type
aclshmemx_handle_wait Public Export
shmem.core.handle_wait symbol
parameter surface
```

状态：

```text
PASS
```

### 12.2 Full Remote Completion

当前 Device 实现的 Handle Wait 完整路径涉及：

```text
aclshmemi_roce_quiet
```

MTE-only 环境运行完整 Handle Wait 时出现：

```text
error code 507015
The aicore execution is abnormal.
```

同时错误信息显示：

```text
The address for the scalar to access the internal buffer of AICore is out of bounds.
```

独立 ROCE Init：

```text
ret=-3
```

当前容器：

```text
/dev/infiniband not exposed
```

因此完整 RDMA-dependent Handle Wait Remote Completion 状态：

```text
BLOCKED
```

不能标记为 PASS。

证据：

```text
../evidence/step3_handle_wait_environment_blocker.txt
../evidence/step3_handle_wait_failure_extract.txt
../evidence/step3_roce_init_failure_extract.txt
../evidence/step3_official_blockers_summary.txt
```

---

## 13. Python / C++ 性能

比较 API：

```text
aclshmemx_putmem_on_stream
```

统一参数：

```text
Engine: MTE
Payload: 64 KiB
Warmup: 10
Iterations: 100
Same Stream
Same Peer Rule
```

结果：

| PE | C++ | Python | Observed Overhead | Result |
|---:|---:|---:|---:|---|
| 2 | 11.402 us | 10.847 us | -4.87% | PASS |
| 4 | 10.811 us | 10.958 us | 1.36% | PASS |
| 8 | 10.858 us | 10.892 us | 0.31% | PASS |

性能要求：

```text
Observed Python Binding Overhead <= 5%
```

结果：

```text
2 PE PASS
4 PE PASS
8 PE PASS
```

负 Overhead 仅代表该次测量没有观察到 Python 的额外开销，不表示 Python 在机制上快于 C++。

证据：

```text
../evidence/performance_acceptance_summary.txt
```

---

## 14. 正式 Self-Test

正式交付脚本：

```text
../self_test/run_all.sh
```

实际执行结果：

```text
PASS_COUNT=5
FAIL_COUNT=0

Full RDMA-dependent handle_wait is NOT counted as PASS.
Current status: BLOCKED by validation environment.

SUPPORTED_SELF_TESTS_ALL_PASS

SELF_TEST_OVERALL_RC=0
```

状态：

```text
PASS
```

这里的 PASS 指当前环境支持执行的 5 个正式自测试全部通过。

---

## 15. 官方归档用例状态

| Test | Status |
|---|---|
| `test_init_multi_instance.py` | PASS |
| `test_heap_mem_type.py` | PASS |
| `test_engine_config.py` | PASS |
| `test_barrier_sync.py` | PASS |
| `test_prof.py` | PASS |
| original `test_core_wrappers.py` | Test compatibility issue before SHMEM execution |
| corrected `test_core_wrappers_hccl_fixed.py` | PASS |
| full `test_handle_wait.py` | BLOCKED by current RDMA environment |

因此本报告不生成虚假的：

```text
Official tests all PASS
```

结论。

---

## 16. 测试结果总表

| 验收内容 | 状态 |
|---|---|
| Python Host API Binding | PASS |
| Public API | PASS |
| Lazy Loading | PASS |
| Python Extension Suite | PASS |
| RMA Put/Get | PASS |
| RMA Bit-exact | PASS |
| Barrier / Sync | PASS |
| Multi-instance Create | PASS |
| Multi-instance Context Switch | PASS |
| Multi-instance Heap Isolation | PASS |
| Destroy B Keep A | PASS |
| MTE | PASS |
| SDMA | PASS |
| Core Wrapper Corrected Validation | PASS |
| Performance 2 PE | PASS |
| Performance 4 PE | PASS |
| Performance 8 PE | PASS |
| Python Binding Overhead <= 5% | PASS |
| Handle Wait Binding | PASS |
| Full Handle Wait RDMA Completion | BLOCKED |
| RDMA Runtime | BLOCKED |
| UDMA Runtime | NOT VERIFIED |

---

## 17. 结论

SHMEM Python API 主要 Host Binding、高层 Wrapper、RMA、Barrier/Sync、Multi-instance、Typed Heap、MTE、SDMA 及 Python/C++ 性能验证均已完成。

正式 Self-Test 结果：

```text
PASS_COUNT=5
FAIL_COUNT=0
SUPPORTED_SELF_TESTS_ALL_PASS
SELF_TEST_OVERALL_RC=0
```

当前环境支持执行的正式自测试全部通过。

完整 RDMA-dependent `handle_wait` Remote Completion 无法在当前容器完成验证。

原因：

```text
/dev/infiniband not exposed
ROCE initialization ret=-3
MTE-only Handle Wait triggers AICore 507015
```

因此该项保持：

```text
BLOCKED
```

待具备完整 RDMA/ROCE Runtime 的环境后补充最终验证。


---

## 18. 证据索引

### 18.1 Python Extension
- `../evidence/logs/examples_python_extension_run_v2.log`

### 18.2 Public Lazy API
- `../evidence/logs/public_lazy_api.log`
- `../evidence/logs/public_lazy_api.rc`

### 18.3 RMA Bit-exact
- `../evidence/logs/step2_putmem_on_stream_bitexact.log`

### 18.4 Multi-instance
- `../evidence/logs/step2_multi_instance_create_switch_v4.log`
- `../evidence/logs/step2_multi_instance_heap_isolation.log`

### 18.5 Barrier 2/4/8 PE
- `../evidence/logs/barrier_multicard_summary.txt`
- `../evidence/logs/barrier_2npu.rc`
- `../evidence/logs/barrier_4npu.rc`
- `../evidence/logs/barrier_8npu.rc`

### 18.6 SDMA
- `../evidence/logs/sdma_put_stream_formal.log`
- `../evidence/logs/sdma_put_stream_formal.rc`

### 18.7 Core Wrapper
- `../evidence/logs/step3_core_wrappers_hccl_fixed.log`

### 18.8 Performance
- `../evidence/performance_acceptance_summary.txt`

### 18.9 Formal Self-Test
- `../evidence/formal_self_test_summary.txt`
- `../evidence/logs/formal_self_test_result.txt`
- `../self_test/logs/`

### 18.10 Handle Wait / RDMA Environment Blocker
- `../evidence/step3_handle_wait_environment_blocker.txt`
- `../evidence/step3_handle_wait_failure_extract.txt`
- `../evidence/step3_roce_init_failure_extract.txt`
- `../evidence/step3_official_blockers_summary.txt`

状态说明：

- Handle Wait Binding：PASS
- Full RDMA-dependent Handle Wait Remote Completion：BLOCKED
- RDMA Runtime：BLOCKED
- UDMA Runtime：NOT VERIFIED


---

## 19. 验收截图

为便于验收复核，归档以下终端执行结果截图：

- `../evidence/screenshots/01_formal_self_test.png`
  - Formal Self-Test 汇总；
  - `PASS_COUNT=5`；
  - `FAIL_COUNT=0`；
  - `SELF_TEST_OVERALL_RC=0`。

- `../evidence/screenshots/02_correctness.png`
  - RMA putmem_on_stream bit-exact；
  - Multi-instance create/switch；
  - Multi-instance heap isolation。

- `../evidence/screenshots/03_performance.png`
  - 2 PE / 4 PE / 8 PE Python 与 C++ same-path 性能对比；
  - Python binding observed overhead 均不超过 5%。

- `../evidence/screenshots/04_api_barrier_sdma.png`
  - Public Lazy API；
  - Barrier 2/4/8 PE；
  - SDMA smoke validation。

- `../evidence/screenshots/05_handle_wait_blocker.png`
  - 当前环境 `/dev/infiniband` 不存在；
  - ROCE 初始化返回 `-3`；
  - Full RDMA-dependent Handle Wait Remote Completion 状态为 `BLOCKED`；
  - 该项未记为 PASS。
