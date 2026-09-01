# SHMEM Python API 设计文档

## 1. 需求背景

### 1.1 需求来源

本任务来源于 CANN 社区任务 2026 年 7 月 **SHMEM Python API 补齐任务**。

SHMEM（ACLSHMEM）面向昇腾集群提供对称共享内存、单边 RMA、集合通信及同步等能力。现有 SHMEM 主线已经具备 Host C++ API 和部分 Python 能力，但在 Host 接口到 Python 绑定、高层 Python 封装、测试覆盖以及用户文档之间仍存在能力缺口。

本任务目标是在现有工程基础上，通过 pybind11 补齐 SHMEM Host C++ API 的 Python Binding，并完善：

- `_pyshmem` 底层 Python Binding；
- `shmem.core` Pythonic 高层封装；
- `shmem` 顶层公开 API；
- Multi-instance 支持；
- Stream API；
- Engine Config；
- Profiling；
- Handle / handle_wait；
- 多卡功能测试；
- 正确性测试；
- 性能验证；
- Python API 文档；
- QuickStart 文档。

本次代码实现仓库：

```text
cann/shmem
```

开发分支：

```text
feat/shmem-python-api
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

## 2. 背景介绍

### 2.1 SHMEM Python 调用分层

本设计采用以下 Python API 分层：

```text
用户 Python 程序
        │
        ▼
shmem 顶层公开 API
        │
        ├── __all__
        ├── Lazy Loading
        └── Native Library Guard
        │
        ▼
shmem.core
        │
        ├── Pythonic API
        ├── 参数检查
        ├── 异常转换
        └── Stream / Multi-instance 封装
        │
        ▼
_pyshmem
        │
        ├── pybind11
        ├── Python/C++ 类型转换
        ├── Pointer / Stream 转换
        └── GIL 管理
        │
        ▼
ACLSHMEM Host C++ API
        │
        ▼
SHMEM Runtime
        │
        ├── MTE
        ├── SDMA
        ├── RDMA
        └── UDMA
```

其中：

- `_pyshmem` 层尽量保持原生 C++ API 语义；
- `shmem.core` 层负责 Pythonic 高层封装；
- `shmem` 顶层负责公开 API、Lazy Loading 和原生库加载保护；
- Python 层不改变底层 SHMEM 的通信、同步和对称内存语义。

---

## 3. 当前问题

任务开始前主要存在以下问题：

1. 部分 Host C++ API 尚未提供 Python Binding；
2. barrier / sync 能力无法完整从 Python 使用；
3. Multi-instance Python 支持不完整；
4. 指定 Host / Device 内存类型的 Heap API 不完整；
5. Stream 参数缺少统一 Python 表达约定；
6. Handle 类型及 `handle_wait` 缺少完整 Python 暴露；
7. Engine Config 缺少 Python 接口；
8. Profiling API 涉及原生内部 Buffer / Pointer，不适合直接暴露；
9. 部分接口只有底层能力，没有 `shmem.core` Pythonic 封装；
10. 缺少多 PE 功能验证；
11. 缺少 RMA bit-exact 正确性验证；
12. 缺少 Multi-instance Heap 隔离验证；
13. 缺少 Python 与 C++ 同路径性能对比；
14. Python API 和 QuickStart 文档不足。

---

## 4. 需求分析

### 4.1 总体目标

在不修改 ACLSHMEM 原生通信语义的前提下，补齐 Host API 的 Python 支持，使 Python 用户能够完成：

- SHMEM 初始化；
- Unique ID 属性构建；
- Multi-instance Context 管理；
- 指定 Memory Type 的对称内存申请；
- barrier；
- sync；
- Stream barrier；
- Stream RMA；
- handle_wait；
- MTE / SDMA / RDMA / UDMA Engine Config；
- Profiling；
- 高层 `shmem.core` 调用。

同时要求：

- Python 行为与 C++ Host API 尽量保持一致；
- Python 层不修改 Collective 语义；
- Python Binding 的额外性能开销满足任务要求；
- 功能测试覆盖新增能力；
- 正确性测试覆盖 RMA 和 Multi-instance；
- 文档说明参数、返回值、异常、Stream 和 Pointer 使用约定。

---

## 5. 接口需求拆解

### 5.1 pybind11 Host API 补齐

本次主要补充以下 Python Host API：

| 类别 | Python API |
|---|---|
| Multi-instance | `aclshmemx_set_attr_uniqueid_args` |
| Multi-instance | `aclshmemx_instance_ctx_get` |
| Multi-instance | `aclshmemx_instance_ctx_set` |
| Memory | `aclshmemx_malloc` |
| Memory | `aclshmemx_calloc` |
| Memory | `aclshmemx_align` |
| Memory | `aclshmemx_free` |
| Synchronization | `aclshmem_barrier` |
| Synchronization | `aclshmem_barrier_all` |
| Synchronization | `aclshmem_sync` |
| Synchronization | `aclshmem_sync_all` |
| Stream Synchronization | `aclshmemx_barrier_on_stream` |
| Stream Synchronization | `aclshmemx_barrier_all_on_stream` |
| Stream RMA | `aclshmemx_putmem_on_stream` |
| Handle | `aclshmemx_handle_wait` |
| Engine | `aclshmemx_set_mte_config` |
| Engine | `aclshmemx_set_sdma_config` |
| Engine | `aclshmemx_set_rdma_config` |
| Engine | `aclshmemx_set_udma_config` |
| Profiling | `aclshmemx_get_prof` |
| Profiling | `aclshmemx_show_prof` |

同时补充：

```text
Handle
MemType
InitAttr.instance_id
```

### 5.2 顶层公开 API

顶层 `shmem` 模块负责公开新增 API，并通过：

```python
shmem.__all__
```

向用户暴露公共接口。

顶层 API 使用 Lazy Loading，避免普通 `import shmem` 时立即加载全部 Native Library。

---

## 6. 详细设计

### 6.1 `_pyshmem` Binding 层

实现位置：

```text
src/host/python_wrapper/pyshmem.cpp
```

主要职责：

1. 使用 pybind11 注册 ACLSHMEM Host API；
2. 完成 Python/C++ 参数转换；
3. 完成 Enum 和 Class Binding；
4. 完成 Python Integer 与 C++ Pointer 的转换；
5. 完成 Python Integer 与 `aclrtStream` 的转换；
6. 对适合的 Native API 管理 Python GIL；
7. 对 Profiling 等特殊接口进行安全包装；
8. 尽量保持原生返回值和语义。

设计原则：

```text
_pyshmem = 原生能力的薄封装
```

---

### 6.2 `shmem.core` 层

实现位置：

```text
src/python/shmem/core/
```

主要职责：

1. 提供 Pythonic API；
2. 对输入参数进行检查；
3. 将 Native Error Code 转换为 Python 异常；
4. 隐藏部分底层 Pointer / Handle 细节；
5. 对 Stream API 提供统一调用入口；
6. 提供 Multi-instance Context 高层接口。

典型接口：

```python
core.barrier()
core.barrier_all()
core.sync()
core.sync_all()

core.instance_ctx_get()
core.instance_ctx_set(instance_id)

core.handle_wait(handle, stream)
```

---

### 6.3 顶层 `shmem` 模块

实现位置：

```text
src/python/shmem/__init__.py
```

主要职责：

- 导出公共 API；
- 管理 `__all__`；
- Lazy Load `_pyshmem`；
- 检查 Native Library；
- 处理原生库加载错误。

---

## 7. Lazy Loading 设计

### 7.1 设计目标

普通：

```python
import shmem
```

不立即强制加载所有 Native Library。

只有当用户真正访问 Native API 时，再加载：

```text
_pyshmem
libshmem.so
```

流程：

```text
import shmem
    ↓
加载 Python 顶层模块
    ↓
访问 Native API
    ↓
__getattr__()
    ↓
_ensure_native()
    ↓
_load_native()
    ↓
加载 _pyshmem
```

### 7.2 Native Library Guard

为了避免同一 Python 进程同时加载不同来源的：

```text
libshmem.so
```

增加 Native Library 一致性检查。

若同时加载：

```text
site-packages/shmem/backends/910/libshmem.so
```

与：

```text
/workspace/shmem/build/lib/libshmem.so
```

可能造成：

- Symbol 冲突；
- ABI 不一致；
- Runtime 状态污染；
- SHMEM 初始化状态异常。

因此发现多份不同路径的 `libshmem.so` 后主动抛出异常，提示清理重复安装或 `LD_LIBRARY_PATH`。

---

## 8. Barrier / Sync 设计

### 8.1 Host Barrier

对应：

```text
aclshmem_barrier
aclshmem_barrier_all
aclshmem_sync
aclshmem_sync_all
```

Python Binding 保持 C++ Collective 语义。

原则：

- 参与 PE 按正确顺序调用；
- Python 层不自动补齐缺失 PE；
- Python 层不改变同步范围；
- Python 层不修改原生 Team 语义。

### 8.2 Stream Barrier

对应：

```text
aclshmemx_barrier_on_stream
aclshmemx_barrier_all_on_stream
```

Python 中 Stream 使用整数地址表示：

```python
stream_ptr: int
```

概念转换：

```text
Python int
    ↓
intptr_t
    ↓
aclrtStream
```

Python 不拥有 Stream 生命周期。

---

## 9. Multi-instance 设计

### 9.1 `InitAttr.instance_id`

Python `InitAttr` 补充：

```python
instance_id
```

从而支持：

```text
instance_id > 0
```

的多实例初始化。

### 9.2 Default Multi-instance

Default Multi-instance 模式使用：

```text
SHMEM_INSTANCE_PORT_RANGE=start:end
```

输入 Port 为：

```text
0
```

实际端口按：

```text
start_port + instance_id
```

分配。

专项测试真实创建：

```text
instance_id = 1
instance_id = 2
```

### 9.3 Context Get

原生：

```text
aclshmemx_instance_ctx_get()
```

返回 Context Pointer。

Python 侧以整数地址表示：

```text
aclshmem_instance_ctx *
        ↓
intptr_t
        ↓
Python int
```

### 9.4 Context Set

原生：

```text
aclshmemx_instance_ctx_set(instance_id)
```

负责切换当前活动实例。

Python 层只传递 `instance_id`，不在 Python 侧维护额外 Runtime State。

### 9.5 生命周期

普通：

```text
aclshmem_finalize()
```

用于释放当前活动实例。

Multi-instance 特定销毁通过：

```text
aclshmemx_finalize(instance_id)
```

执行。

专项验证结果：

```text
INSTANCE_CREATE_PASS id=1
INSTANCE_CREATE_PASS id=2

CONTEXT_SWITCH_PASS id=1
CONTEXT_SWITCH_PASS id=2

MULTI_INSTANCE_SWITCH_PASS

FINALIZE_B ret=0
FINALIZE_A ret=0

MULTI_INSTANCE_CREATE_SWITCH_OVERALL_PASS
RC=0
```

---

## 10. Multi-instance Heap 隔离

专项测试流程：

```text
Create Instance A
        ↓
malloc A
        ↓
write Pattern A
        ↓
Create Instance B
        ↓
malloc B
        ↓
write Pattern B
        ↓
Switch B → A
        ↓
verify Pattern A
        ↓
Switch A → B
        ↓
verify Pattern B
        ↓
Repeated A ↔ B
        ↓
verify both
        ↓
Destroy B
        ↓
Switch A
        ↓
verify A still valid
```

测试结果：

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

验证当前范围内：

- A/B Context 独立；
- A/B Heap 数据互不污染；
- 多次切换后数据保持；
- 销毁 B 后 A 继续有效。

---

## 11. 指定 Memory Type 的对称 Heap

新增：

```text
aclshmemx_malloc
aclshmemx_calloc
aclshmemx_align
aclshmemx_free
```

并暴露：

```python
MemType
```

包括：

```text
HOST_SIDE
DEVICE_SIDE
```

Native Pointer：

```text
void *
```

Python 表示：

```text
int
```

转换：

```text
void *
  ↓
intptr_t
  ↓
Python int
```

Python 不接管 Native Pointer 生命周期。

---

## 12. Stream RMA

公开：

```text
aclshmemx_putmem_on_stream
```

Python 参数中的：

- destination address；
- source address；
- stream；

均使用整数地址形式与 Native 层交互。

Python Binding 不复制 Payload，不修改原生 RMA 语义。

---

## 13. RMA 正确性设计

现有 Python Core RMA 测试覆盖：

- `put`；
- `get`；
- `put_signal`；
- `signal_op`；
- `signal_wait`；
- `quiet`；
- `quiet_on_stream`；
- Stream Put；
- Stream Get。

测试会真实读取 Device Memory 并验证结果，不只是检查 API 是否存在。

### 13.1 `putmem_on_stream` Bit-exact

专项参数：

```text
PE 数：2
Payload：256 Bytes
数据：确定性的 rank-dependent pattern
传输：cross-rank
读取：Device → Host
比较：逐字节 exact compare
```

结果：

```text
PE0: PUTMEM_ON_STREAM_BITEXACT_PASS
PE1: PUTMEM_ON_STREAM_BITEXACT_PASS

PUTMEM_ON_STREAM_BITEXACT_OVERALL_PASS
BITEXACT_RC=0
```

---

## 14. Handle / handle_wait

### 14.1 Handle

Python 暴露：

```python
Handle
```

用于包装 Native Handle 信息。

### 14.2 handle_wait

公开：

```text
aclshmemx_handle_wait
```

同时提供：

```text
shmem.core.handle_wait
```

Python Binding 负责：

- Handle 参数转换；
- Stream 地址转换；
- 调用 Native Handle Wait。

Python 不重写 Native Completion 机制。

### 14.3 环境限制

当前 Device 实现中，Handle Wait 完整 Completion 路径涉及：

```text
aclshmemi_roce_quiet
```

当前验证主机：

```text
8 × Ascend 910B4
```

NPU 均为 Healthy。

sysfs 可见：

```text
/sys/class/infiniband/hns_0
/sys/class/infiniband/hns_1
```

但容器不存在：

```text
/dev/infiniband
```

ROCE 初始化专项测试：

```text
initialize shmem failed, ret=-3
```

MTE-only Handle Wait 运行时出现：

```text
507015
The aicore execution is abnormal.
```

并出现：

```text
The address for the scalar to access the internal buffer of AICore is out of bounds.
```

因此：

```text
Handle Wait Binding              PASS
Handle Wait Public Export        PASS
Handle Wait Wrapper Surface      PASS
Full RDMA Remote Completion      BLOCKED
```

不能把 Full RDMA Handle Wait 写成 PASS。

---

## 15. Engine Config

新增：

```text
aclshmemx_set_mte_config
aclshmemx_set_sdma_config
aclshmemx_set_rdma_config
aclshmemx_set_udma_config
```

Python Binding 不修改 Native 配置逻辑和参数限制。

### 15.1 MTE

当前完成：

```text
初始化
Heap
RMA
putmem_on_stream
Barrier
Sync
Multi-instance
```

结果：

```text
MTE 主路径 PASS
```

### 15.2 SDMA

专项 Smoke：

```text
2 PE
malloc
putmem_on_stream
stream synchronize
finalize
```

结果：

```text
SDMA_INIT_OK
SDMA_PUT_STREAM_OK
SDMA_FINALIZE_OK
RC=0
```

### 15.3 RDMA

当前：

```text
ROCE init ret=-3
```

因此：

```text
RDMA Full Runtime Validation = BLOCKED
```

### 15.4 UDMA

当前环境未形成正式 UDMA Runtime 验证结论。

因此：

```text
UDMA Runtime Validation = NOT VERIFIED
```

---

## 16. Profiling

Native Profiling 接口可能涉及 Runtime 管理的数据和 Pointer。

Python 侧避免把需要用户手工维护生命周期的内部 Buffer 直接暴露出去。

公开：

```text
aclshmemx_get_prof
aclshmemx_show_prof
```

Profiling 数据转换为 Python 可管理对象，使 Native Buffer Ownership 继续由 SHMEM Runtime 管理。

---

## 17. 参数类型映射

### 17.1 Integer

C++：

```text
int
uint32_t
uint64_t
size_t
```

Python：

```text
int
```

### 17.2 Pointer

C++：

```text
void *
aclshmem_instance_ctx *
```

Python：

```text
int
```

### 17.3 Stream

C++：

```text
aclrtStream
```

Python：

```text
int
```

### 17.4 Unique ID

C++：

```text
aclshmemx_uniqueid_t
```

Python：

```text
bytes
```

### 17.5 Enum

通过：

```text
py::enum_
```

暴露。

例如：

```python
MemType.DEVICE_SIDE
MemType.HOST_SIDE
```

### 17.6 Struct / Class

通过：

```text
py::class_
```

暴露：

```text
InitAttr
Handle
```

---

## 18. 返回值与异常

底层 `_pyshmem` 尽量保留 Native 返回值：

```text
Native int status
      ↓
Python int
```

`shmem.core` 提供 Pythonic 参数检查和异常处理。

原则：

```text
_pyshmem
    ↓
Native Semantics

shmem.core
    ↓
Pythonic Semantics
```

明显的 Python 参数类型错误由 Python / pybind11 抛出：

```text
TypeError
```

---

## 19. GIL

对于适合进入 Native Runtime 或可能阻塞的 API，可使用：

```text
py::gil_scoped_release
```

降低 Native 通信期间长期占用 Python GIL 的影响。

Python 参数解析和返回对象构造仍需要在持有 GIL 时完成。

---

## 20. Public API Lazy Exposure 验证

专项检查结果：

```text
missing_from___all__ = []
missing_or_error = []

ALL_REQUIRED_LAZY_PUBLIC_APIS_OK
```

同时观察：

```text
_NATIVE_LOADED
False → True
```

说明 Lazy Loading 正常。

---

## 21. Python Extension 全量测试

入口：

```bash
bash examples/python_extension/run.sh
```

覆盖：

```text
init_test
tls_test
test.py
barrier_sync_test
test_init_final
test_memory
test_rma
test_direct
test_collective
test_multi_instance
engine config
profiling
handle_wait binding
```

结果：

```text
test_prof running success!
test_handle_wait_binding running success!
All Python tests passed!
EXAMPLES_PYTHON_EXTENSION_FINAL_RC=0
```

其中 `test_handle_wait_binding` 只代表：

```text
Binding / Type / Export / Parameter Surface
```

不等价于完整 RDMA Completion PASS。

---

## 22. Core Wrapper 验证

官方归档 `test_core_wrappers.py` 使用：

```text
HCCL backend
+
CPU Tensor broadcast
```

在 SHMEM 功能执行前出现 Backend 兼容问题。

修正版将 Unique ID Broadcast Tensor 放到 NPU。

结果：

```text
core barrier/sync wrappers ok
core multi_instance ok
core.handle_wait symbol ok

test_core_wrappers_hccl_fixed.py running success!
CORE_WRAPPERS_FIXED_RC=0
```

因此 Core Wrapper 功能验证通过。

原始测试问题应记录为测试兼容问题，而不是 SHMEM Python API 功能失败。

---

## 23. Barrier / Sync 多卡

执行：

```text
2 PE
4 PE
8 PE
```

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

---

## 24. 性能设计

性能比较使用同路径：

```text
aclshmemx_putmem_on_stream
```

统一参数：

```text
Engine      = MTE
Payload     = 64 KiB
Warmup      = 10
Iterations  = 100
Peer        = next rank
Stream      = same path
```

### 24.1 C++ Baseline

| PE 数 | C++ 平均延迟 |
|---:|---:|
| 2 | 11.402 us |
| 4 | 10.811 us |
| 8 | 10.858 us |

### 24.2 Python

| PE 数 | Python 平均延迟 | Observed Overhead |
|---:|---:|---:|
| 2 | 10.847 us | -4.87% |
| 4 | 10.958 us | 1.36% |
| 8 | 10.892 us | 0.31% |

全部满足：

```text
Observed Python Binding Overhead <= 5%
```

2 PE 的负值仅代表本次测试中没有观察到 Python 的额外性能开销，不表示 Python 机制上比 C++ 更快。

---

## 25. 可测试性设计

整体验证按照：

```text
API 存在性
    ↓
参数与类型
    ↓
Lazy Export
    ↓
单接口功能
    ↓
2 PE 功能
    ↓
RMA 正确性
    ↓
Multi-instance 隔离
    ↓
2/4/8 PE
    ↓
Python / C++ 性能
    ↓
Engine Smoke
```

进行。

---

## 26. 功能验证结果

| 测试项目 | 结果 | 说明 |
|---|---|---|
| Public API Lazy Export | PASS | 所需 API 可访问 |
| `examples/python_extension/run.sh` | PASS | RC=0 |
| Barrier | PASS | 多 PE |
| Barrier on Stream | PASS | 多 PE |
| Sync | PASS | 多 PE |
| Sync All | PASS | 多 PE |
| Typed Heap | PASS | Memory Type |
| Engine Config | PASS | Binding / Config |
| Profiling | PASS | Python 测试 |
| RMA Put | PASS | 数据验证 |
| RMA Get | PASS | 数据验证 |
| Put Signal | PASS | Signal + Data |
| Quiet | PASS | RMA |
| Quiet on Stream | PASS | Stream |
| `putmem_on_stream` Bit-exact | PASS | 256 Bytes |
| Multi-instance Create | PASS | Instance 1/2 |
| Multi-instance Context Switch | PASS | A ↔ B |
| Multi-instance Heap Isolation | PASS | Pattern A/B |
| Destroy B Keep A | PASS | A 保持 |
| Core Wrapper Fixed Validation | PASS | RC=0 |
| MTE | PASS | 主路径 |
| SDMA | PASS | Smoke |
| RDMA Full Runtime | BLOCKED | 当前容器缺 `/dev/infiniband` |
| UDMA Runtime | NOT VERIFIED | 当前环境未验证 |
| Handle Wait Binding | PASS | Binding |
| Handle Wait Full Completion | BLOCKED | RDMA 环境限制 |

---

## 27. Handle Wait 环境阻塞说明

当前：

```text
/sys/class/infiniband/hns_0
/sys/class/infiniband/hns_1
```

存在。

但是：

```text
/dev/infiniband
```

不存在。

ROCE 初始化：

```text
ret=-3
```

MTE-only Handle Wait：

```text
507015
```

因此最终报告统一描述为：

```text
The archived handle_wait validation uses a path whose device-side
completion logic depends on RDMA/ROCE.

On the current validation host, HNS RDMA HCAs are visible through
sysfs, but the container does not expose /dev/infiniband device
nodes. A separate ROCE initialization probe fails before handle_wait
execution with ACLSHMEM_SMEM_ERROR (-3).

Running the handle_wait path under MTE-only initialization triggers
an Ascend AICore 507015 exception.

An isolated MTE putmem_on_stream test passes, and the Python
putmem_on_stream bit-exact validation also passes. Therefore the
observed failure is specific to the RDMA-dependent handle_wait path
rather than the Python RMA binding itself.

Full RDMA-dependent handle_wait remote-completion semantics could not
be exercised in this environment and are therefore reported as
BLOCKED rather than PASS.
```

---

## 28. 性能验收结果

| PE 数 | C++ | Python | Observed Overhead | 结果 |
|---:|---:|---:|---:|---|
| 2 | 11.402 us | 10.847 us | -4.87% | PASS |
| 4 | 10.811 us | 10.958 us | 1.36% | PASS |
| 8 | 10.858 us | 10.892 us | 0.31% | PASS |

均满足：

```text
Observed Python Binding Overhead <= 5%
```

---

## 29. 兼容性

### 29.1 C++ API

本任务主要新增 Python Binding 与 Python Wrapper。

不改变已有 Native Host API 调用方式。

### 29.2 Python

新增 API 通过：

```text
__all__
__getattr__
Lazy Native Load
```

公开。

### 29.3 Memory ABI

Python 不复制或改变 SHMEM Heap 内存布局。

Native Pointer 继续由 Runtime 管理。

---

## 30. 可靠性

### 30.1 Native Library Guard

发现多个 `libshmem.so` 时主动报错，避免：

```text
Symbol Conflict
ABI Mismatch
Runtime State Corruption
```

### 30.2 Multi-instance

Runtime State 由 Native Context 管理。

Python 不额外复制 Context State。

### 30.3 Pointer Ownership

Python 不拥有 Native SHMEM Pointer。

### 30.4 Engine

Python 不绕过原生 Engine 参数约束。

---

## 31. 可维护性

代码结构：

```text
src/host/python_wrapper/pyshmem.cpp
        ↓
Native Binding

src/python/shmem/core/
        ↓
Pythonic Wrapper

src/python/shmem/__init__.py
        ↓
Public API / Lazy Loading

examples/python_extension/test/
        ↓
Functional Tests

docs/api/pythonAPI.md
        ↓
API Documentation

docs/quickstart.md
        ↓
QuickStart
```

后续新增 API 可以按照：

```text
C++ Host API
    ↓
pybind11 Binding
    ↓
shmem.core
    ↓
Public Export
    ↓
Test
    ↓
Documentation
```

扩展。

---

## 32. 文档设计

产品仓库文档包括：

```text
docs/api/pythonAPI.md
docs/quickstart.md
```

其中：

`pythonAPI.md` 描述：

- API；
- 参数；
- 返回值；
- Pointer；
- Stream；
- Multi-instance；
- Engine；
- Profiling；
- Handle。

`quickstart.md` 描述：

- Wheel 安装；
- 环境；
- 初始化；
- Heap；
- RMA；
- Collective；
- Multi-instance。

---

## 33. 测试环境

```text
OS:
openEuler 24.03 LTS-SP2

Architecture:
aarch64

NPU:
8 × Ascend 910B4

CANN:
9.1.0-beta.3

Python:
3.11.6

PyTorch:
2.7.1+cpu

torch_npu:
2.7.1.post8
```

Wheel：

```text
cann_shmem-1.0.0-cp311-cp311-linux_aarch64.whl
```

---

## 34. 验收标准对应情况

| 验收项 | 当前状态 |
|---|---|
| Host C++ API Python Binding | 已完成 |
| `shmem.core` 高层封装 | 已完成 |
| Public API Exposure | 已完成 |
| Lazy Loading | 已完成 |
| Python Extension 全量测试 | PASS |
| Barrier / Sync | PASS |
| Multi-instance | PASS |
| Multi-instance Heap Isolation | PASS |
| Typed Heap | PASS |
| Engine Config | PASS |
| Profiling | PASS |
| MTE | PASS |
| SDMA Smoke | PASS |
| RMA Bit-exact | PASS |
| 2 PE Performance | PASS |
| 4 PE Performance | PASS |
| 8 PE Performance | PASS |
| Binding Overhead <= 5% | PASS |
| Core Wrapper Validation | PASS |
| Handle Wait Binding | PASS |
| Handle Wait Full RDMA Completion | BLOCKED |
| RDMA Runtime | BLOCKED |
| UDMA Runtime | NOT VERIFIED |
| Python API 文档 | 已完成独立文档提交 |
| QuickStart | 已完成独立文档提交 |

---

## 35. 已知限制

1. Stream Pointer 以 Python Integer 表示，用户需要保证 Stream 有效；
2. Heap Pointer 以 Python Integer 表示，Python 不管理生命周期；
3. Collective 操作仍要求各 PE 正确参与；
4. Multi-instance Team 高级操作继续遵循 Native 限制；
5. Handle Wait Full Completion 依赖 RDMA/ROCE；
6. 当前容器无 `/dev/infiniband`；
7. UDMA 未完成正式 Runtime 验证；
8. Profiling Native Buffer 不直接交由 Python 用户管理；
9. Python Binding 不绕过 Native Engine 限制；
10. 性能结果仅代表当前测试环境实测结果。

---

## 36. 风险分析

### 36.1 Native Library 重复加载

风险：

```text
多个 libshmem.so
    ↓
Symbol Conflict
    ↓
ABI Mismatch
    ↓
Runtime State Corruption
```

通过 Native Library Guard 提前阻止。

### 36.2 Pointer 风险

可能包括：

- 已释放 Pointer；
- 错误 Instance Pointer；
- Memory Type 不一致；
- Stream 地址无效。

通过文档和测试明确 Ownership。

### 36.3 Multi-instance 风险

专项覆盖：

```text
Context A/B
Heap A/B
Repeated Switch
Destroy B Keep A
```

均验证通过。

### 36.4 RDMA 风险

RDMA Runtime 可用性同时受：

- Driver；
- HCA；
- Container Device Mapping；
- `/dev/infiniband`；
- Network；
- SHMEM Backend；

影响。

不能只根据 sysfs 存在判断 RDMA 可用。

---

## 37. 测试结论

当前已经完成：

```text
Python Extension Full Suite         PASS
Public Lazy API                     PASS
MTE                                 PASS
SDMA Smoke                          PASS
RMA Put/Get                         PASS
RMA Bit-exact                       PASS
Barrier / Sync                      PASS
2/4/8 PE Barrier                    PASS
Multi-instance Create               PASS
Multi-instance Context Switch       PASS
Multi-instance Heap Isolation       PASS
Destroy B Keep A                    PASS
Core Wrapper Corrected Validation   PASS
Python/C++ Performance <=5%         PASS
```

性能结果：

```text
2 PE:
C++     11.402 us
Python  10.847 us
Observed Overhead = -4.87%

4 PE:
C++     10.811 us
Python  10.958 us
Observed Overhead = 1.36%

8 PE:
C++     10.858 us
Python  10.892 us
Observed Overhead = 0.31%
```

全部满足：

```text
Observed Python Binding Overhead <= 5%
```

当前未能在本容器完成：

```text
RDMA-dependent handle_wait remote-completion semantics
```

原因：

```text
HNS RDMA HCA:
visible in sysfs

/dev/infiniband:
not exposed

ROCE initialization:
ret=-3

MTE-only handle_wait:
AICore 507015
```

因此该项为：

```text
BLOCKED BY VALIDATION ENVIRONMENT
```

而非 PASS。

---

## 38. 总体结论

本设计通过：

```text
ACLSHMEM Host C++ API
        ↓
pybind11 / _pyshmem
        ↓
shmem.core
        ↓
shmem Top-level Public API
```

建立 SHMEM Python Host API 使用链路。

本次工作覆盖：

- Host API Python Binding；
- Python / C++ 类型映射；
- Pointer / Stream 约定；
- Multi-instance；
- Typed Heap；
- Barrier / Sync；
- Handle；
- Engine Config；
- Profiling；
- Lazy Loading；
- Native Library Guard；
- Pythonic Core Wrapper；
- Python API 文档；
- Multi-PE 测试；
- RMA 正确性；
- Multi-instance Heap 隔离；
- Python/C++ 性能测试。

当前主要功能和性能路径已完成验证。

对于完整 RDMA Handle Wait 能力，已经形成：

```text
Runtime Error
+
Device Environment
+
Source Path
```

证据，能够明确区分 Python Binding 问题与当前验证环境 RDMA Runtime 不可用问题。

后续在具备：

```text
/dev/infiniband
+
可正常初始化的 SHMEM ROCE Runtime
```

的环境中，可继续执行：

```text
RDMA Initialization
        ↓
RDMA RMA Operation
        ↓
Handle Wait
        ↓
Remote Data Visibility
        ↓
Full Completion Validation
```

完成该环境相关能力的最终验证。
