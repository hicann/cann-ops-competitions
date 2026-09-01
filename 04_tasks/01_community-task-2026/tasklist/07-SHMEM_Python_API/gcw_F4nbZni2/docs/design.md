# SHMEM Python Host API 设计说明

> 文档性质：可公开评审的设计材料。本文不包含候选源码、二进制、私有仓库地址、
> 机器凭据或未公开性能原始数据。

# 需求背景（required）

## 需求来源

本文对应 2026 年 7 月社区任务《SHMEM Python 接口开发任务书》。任务要求在现有
ACLSHMEM Python 工程上补齐 20 个 Host C++ API 的 pybind11 绑定、13 个
`shmem.core` 高层入口、测试、文档和性能证据，并保持既有 API 兼容。

## 背景介绍

当前主线已经提供初始化、基础 Buffer 和 RMA 等 Python 入口，但多实例上下文、
带 `mem_type` 的对称堆、引擎配置、集合同步、stream barrier、RDMA handle wait
和 profiling 尚未完整对齐 Host C++ API。缺失项不仅是名称导出，还涉及 UID 与
Buffer 生命周期、native 指针的安全快照、GIL/锁边界、stream 完成语义和多卡
能力裁决。

# 需求分析（required）

## 需求描述

目标是提供类型安全、生命周期明确、完成边界可解释的 Host Python API；支持
Python 3.9 及以上、CANN 9.0.0 或更新社区版本，并在相同 shape、stream 和
拓扑下比较 Python Host RMA 与公开 C++ API。适配硬件遵循任务书，面向 Atlas
A2/A3 训练系列或推理系列产品。正式门禁使用单节点 4 PE；2 PE 可作为附加回归
证据。本文不以未执行的 A3 或 8 PE 外推验证结论。

Device Kernel API、改变原生通信语义、隐藏 stream synchronize、用单节点结果
外推跨节点 RDMA，以及在公开设计评审中披露候选源码均不在本设计目标内。

## 需求拆解

1. 在 `shmem._pyshmem` 独立导出任务书要求的 20 个 Host API；除 context getter
   为避免解锁后裸指针竞态而使用同一 native 状态锁内 ID 快照外，可调用型 wrapper
   恰好委托一次对应 native API，不用 Python alias 冒充低层实现。
2. 在 `shmem.core` 提供 13 个 Pythonic 入口，统一参数校验、异常、Buffer、
   stream 和生命周期约定。
3. 保持 legacy `Buffer`、初始化、RMA 和默认 stream 行为兼容。
4. 覆盖 UID、context、typed heap、collective、profiling 及并发释放等关键边界。
5. 使用官方 64 KiB 自测和公开 C++ 比较器完成 4 PE 正式性能门禁；2 PE 作为
   附加功能与回归证据。条件能力必须实际探测，不得依据符号存在或配置返回值
   推断 PASS。

# 详细设计（required）

## 总体架构

```text
用户程序
  ├─ shmem.core        Pythonic 校验、异常、Buffer 所有权、进程级生命周期锁
  └─ shmem._pyshmem    精确宽度转换、值对象、GIL 边界、一次 native 委托
           │
           └─ ACLSHMEM Host C/C++ API（语义与状态来源）
```

`_pyshmem` 是低层 expert escape hatch：保留原生返回值、空指针和状态，不虚构
成功。`shmem.core` 负责把无效 Python 参数、空 context、错误实例和重复释放转换
为稳定异常。两层都不改变 collective 参与者集合、调用顺序或 stream 完成语义。

## 接口映射

| 族 | 低层 `_pyshmem` | 高层 `shmem.core` / 说明 |
|-|-|-|
| UID 初始化 | `aclshmemx_set_attr_uniqueid_args` | 与既有 `get_unique_id`、`init` 链接；不新增重复高层 setter |
| 多实例 | `aclshmemx_instance_ctx_get/set` | `ctx_get`、`ctx_set` |
| 对称堆 | `aclshmemx_malloc/calloc/align/free` | `malloc`、`calloc`、`align`、`free` |
| 引擎配置 | `aclshmemx_set_mte/sdma/rdma/udma_config` | 保留原生状态；能力由依赖操作验证 |
| 集合通信 | `aclshmem_barrier/barrier_all/sync/sync_all` | 同名 Pythonic 包装 |
| Stream 集合 | `aclshmemx_barrier_on_stream/barrier_all_on_stream` | 同名包装；返回仅表示提交 |
| RDMA handle | `aclshmemx_handle_wait` | `handle_wait` |
| Profiling | `aclshmemx_get_prof/show_prof` | Python 自有快照；旧 show 接口保留并弃用 |

公开导出的 20 个低层名称和 13 个高层入口均为独立符号。为使设计评审可直接核验，
冻结签名和返回边界逐项列出如下；完整异常表仍由 Python API 文档和静态契约测试维护。

| ID | 低层冻结签名 | 返回边界 |
|-|-|-|
| L01 | `aclshmemx_set_attr_uniqueid_args(my_pe: int, n_pes: int, local_mem_size: int, uid: Union[UniqueId, bytes, bytearray, memoryview], attr: InitAttr) -> int` | 原生状态 |
| L02 | `aclshmemx_instance_ctx_get() -> Optional[InstanceContext]` | 空指针为 `None`，否则为 ID 快照 |
| L03 | `aclshmemx_instance_ctx_set(instance_id: int) -> int` | 原生状态 |
| L04 | `aclshmemx_malloc(size: int, mem_type: MemType = MemType.DEVICE_SIDE) -> int` | 原生地址，空指针为 `0` |
| L05 | `aclshmemx_calloc(count: int, size: int, mem_type: MemType = MemType.DEVICE_SIDE) -> int` | 原生地址，空指针为 `0` |
| L06 | `aclshmemx_align(alignment: int, size: int, mem_type: MemType = MemType.DEVICE_SIDE) -> int` | 原生地址，空指针为 `0` |
| L07 | `aclshmemx_free(ptr: int, mem_type: MemType = MemType.DEVICE_SIDE) -> None` | 原生 `void`，不虚构状态 |
| L08 | `aclshmemx_set_mte_config(offset: int, ub_size: int, sync_id: int) -> int` | 原生状态，不等同能力 PASS |
| L09 | `aclshmemx_set_sdma_config(offset: int, ub_size: int, sync_id: int) -> int` | 同上 |
| L10 | `aclshmemx_set_rdma_config(offset: int, ub_size: int, sync_id: int) -> int` | 同上 |
| L11 | `aclshmemx_set_udma_config(offset: int, ub_size: int, sync_id: int) -> int` | 同上 |
| L12 | `aclshmem_barrier(team: int) -> None` | 主机阻塞 |
| L13 | `aclshmem_barrier_all() -> None` | 主机阻塞 |
| L14 | `aclshmem_sync(team: int) -> None` | 只依赖规范的 sync 保证 |
| L15 | `aclshmem_sync_all() -> None` | 只依赖规范的 sync 保证 |
| L16 | `aclshmemx_barrier_on_stream(team: int, stream: Optional[int] = None) -> None` | 只入队，不表示完成 |
| L17 | `aclshmemx_barrier_all_on_stream(stream: Optional[int] = None) -> None` | 只入队，不表示完成 |
| L18 | `aclshmemx_handle_wait(handle: Handle, stream: Optional[int] = None) -> None` | 按值复制 handle 后只入队 |
| L19 | `aclshmemx_get_prof(verbose: bool = False) -> Optional[ProfSnapshot]` | 无记录为 `None`；兼容旧 out-param 调用形式 |
| L20 | `aclshmemx_show_prof() -> None` | 保留旧显示行为并发出弃用警告 |

| ID | 高层冻结签名 | 高层策略 |
|-|-|-|
| H01 | `barrier(team: int = 0) -> None` | 校验 team 后阻塞 |
| H02 | `barrier_all() -> None` | WORLD barrier |
| H03 | `sync(team: int = 0) -> None` | 只承诺 sync 语义 |
| H04 | `sync_all() -> None` | WORLD sync |
| H05 | `barrier_on_stream(team: int = 0, stream: Optional[int] = None) -> None` | 只入队 |
| H06 | `barrier_all_on_stream(stream: Optional[int] = None) -> None` | 只入队 |
| H07 | `ctx_get() -> InstanceContext` | 无活动实例转 `AclshmemError` |
| H08 | `ctx_set(instance_id: int) -> None` | 限制在安全的 `uint32` 域 |
| H09 | `malloc(size: int, mem_type: MemType = MemType.DEVICE_SIDE) -> Buffer` | 正整数与空地址检查 |
| H10 | `calloc(count: int, size: int, mem_type: MemType = MemType.DEVICE_SIDE) -> Buffer` | 乘积溢出预检 |
| H11 | `align(alignment: int, size: int, mem_type: MemType = MemType.DEVICE_SIDE) -> Buffer` | 正的 2 的幂对齐 |
| H12 | `free(buf: Buffer) -> None` | provenance 与单次释放状态机 |
| H13 | `handle_wait(handle: Handle, stream: Optional[int] = None) -> None` | 校验后只入队 |

既有性能比较入口 `aclshmemx_putmem_on_stream(dst: int, src: int, elem_size: int, pe: int, stream: int) -> None`
不计入上述新增 20 个低层 API；它单独执行 exact-int/宽度校验并作为官方性能代表。

## 类型、宽度与返回值

- 本任务新增低层接口及性能比较入口的指针先按非负 `intptr_t` 校验，再转换为
  原生指针；不接受 `bool` 冒充整数。
- `instance_id`、size、rank、team、stream 等分别按原生 `uint64_t`、`size_t`、
  `int32_t` 或 `intptr_t` 边界检查。
- `MemType`、SignalOp、CmpOp 使用精确枚举，不接受任意整数替代。
- 原生 `void` 返回 Python `None`；原生状态码保持 `int`；原生空指针保持 `0`
  或在高层按契约转换为异常。
- 低层 `aclshmemx_malloc/calloc/align` 对零尺寸不做高层语义改写；`size=0` 或
  `count=0` 原样下沉，native 空指针保持整数 `0`。高层 `malloc/calloc/align` 只接受
  正数，零值使用 `AclshmemInvalid`，合法正参数分配返回空地址时使用 `AclshmemError`。

## UID 所有权与初始化链

`InitAttr` 使用 Python 不可访问的 C++ 私有存储持有 UID owner。`UniqueId` 输入保持
强引用；bytes-like 输入必须可读、C-contiguous 且长度精确匹配原生结构，然后
复制到稳定私有存储。setter 先保存完整 native attr、bootstrap flag 和旧 owner，
仅在 native 成功后提交新所有权并切换为 `ACLSHMEMX_INIT_WITH_UNIQUEID`；失败时
恢复全部字段和旧 owner。随后 `aclshmem_init(attr)` 使用保存的 bootstrap flag，
形成完整的 builder → init 链。测试覆盖调用方 `del`、GC、重复绑定以及尝试从
Python 删除或覆盖 owner 后仍不会产生悬空指针。

## Context 与进程并发

`InstanceContext` 是只含不可变 ID 的 Python 值快照，不暴露 native 指针。
`instance_ctx_get` 在 native 生命周期互斥锁内检查活动实例并复制 ID；互斥锁
释放后才构造 Python 值对象。GIL 不能替代该 native 锁，因为低层 `finalize`
会先释放 GIL 再进入生命周期路径。
`init`、`finalize`、`ctx_get`、`ctx_set`、allocator/free 使用同一进程级 `RLock`
串行化单次高层调用。该锁不把“ctx_set + 任意用户操作”升级为事务，跨多个 API
的业务序列仍由应用锁保护；直接调用 native API 的外部 C++ 线程不在此锁域内。

非零实例通过设置 `InitAttr.instance_id` 后调用初始化入口创建；随后
`ctx_set(instance_id)` 选择活动实例，`ctx_get()` 只返回当前 ID 快照。每个实例
保存独立 host state、team pool 和 team mask；context 切换同步保存并恢复这些
状态，finalize 后的实例不能被重新切换为活动状态。

## Buffer 所有权与释放状态机

allocator 返回的 Buffer 保存原始地址、长度、MemType、instance ID 和 allocator
kind。为保持历史兼容，`Buffer(addr, length)` 是 owning legacy 包装，并由
`aclshmem_free` 释放；调用方必须保证地址来自匹配的 legacy allocator。只有显式
peer-view factory 产生 non-owning 对象，并强引用 origin。

手工 Buffer 构造和释放前都要求 exact `int`（拒绝 `bool`），并在任何状态变化
前校验非负 `intptr_t` 地址、正 `size_t` 长度及 provenance。高层 `free` 在
进程锁内完成类型、origin、地址、MemType 和实例检查；在进入可能
释放 GIL 的 native 调用前先标记 release-called。该状态只表示“已发起释放且禁止
再次调用”，不宣称 native 已确认成功。即使 native 抛出异常也不回滚，因为地址
可能已被消费，重试会引发 double free。Buffer 没有 `__del__`，不会在不可预测的
GC 时机发起 collective free。

## Stream、collective 与跨节点边界

`None` 和整数 0 只表示 native default stream。on-stream API 和 handle wait 的
返回只表示提交，不表示 stream 已完成；实现不得暗中 synchronize、quiet 或插入
barrier。所有 collective 必须由正确 PE 以匹配参数和顺序进入。

`Handle` 是调用方按 `team_id` 构造的 team-scoped wait condition，只保存该值；
它不是 RMA 操作返回的 future，也不拥有 stream、team 或 runtime 生命周期。
`handle_wait` 在进入 native 前复制 Handle 值，调用方仍负责提供匹配的 team、
stream 和原生操作顺序。

`aclshmem_ptr`/peer Buffer 只有在 runtime 为对应 PE 提供直接地址映射时才可用于
load/store。跨节点数据可见性必须通过 RMA API 验证，并在同一 stream 排入所需
`handle_wait` 后同步精确 stream；不能把单节点 peer 地址可解引用的结果外推到
跨节点。

## Profiling

`get_prof` 与 `show_prof` 共用专用 C++ mutex。等待该 mutex 时释放 GIL。
`get_prof` 在锁内完成“native 调用 +
复制到本地 C++ 值对象”，`show_prof` 在同一锁内完成 native 调用；锁外再构造
Python 对象。native `get_prof` 每次只返回当前单个 PE 的数据，`ProfSnapshot` 保存
不可变 `pe_id` 以及 Python 自有、只读的 NumPy `int64` `ccount`/`cycles` 数组，
二者 shape 均为 `(64, 1024)`，不借用 native 生命周期。如需跨 rank 汇总，只由
测试框架基于逐 rank 快照完成，binding 不遍历或推断其他 PE 指针。

NumPy 是 profiling 快照的运行时依赖，`setup.py`/wheel metadata 的
`install_requires` 固定为 `numpy>=1.24.4,<=1.26.4`。契约测试核对源码声明、wheel
`Requires-Dist`、已安装 metadata，并要求当前运行版本位于该范围；这不表示范围内
每个版本都分别完成了硬件重放。构造 profiling 快照时若无法导入 NumPy，则传播
`ImportError`，不静默降级或借用 native 内存。`get_prof(verbose=False)` 是主 API；
`get_prof(None, verbose)` 兼容官方自测中的历史 out-param 写法，返回 `None` 或单元素
tuple。`show_prof` 保留并发出 `DeprecationWarning`。

## 异常与能力裁决

低层转换失败使用 `TypeError`/`OverflowError`；高层参数或 provenance 失败使用
`AclshmemInvalid`；缺失 context、native 非零状态或空分配使用 `AclshmemError`。
SDMA/RDMA/UDMA、HOST_SIDE 和跨节点能力不得仅依据符号存在或 config 返回 0
判定 PASS，必须运行实际依赖操作；不具备能力时记录带原因的 SKIP。MTE 是必测
主路径，不能被条件 SKIP。

## 修改范围

| 路径 | 设计目的 |
|-|-|
| `src/host/python_wrapper/pyshmem.cpp` | 20 个低层绑定、值对象、UID/profiling 生命周期及受控 RMA 快路径 |
| `src/host/python_wrapper/CMakeLists.txt` | 构建 Python 扩展和可回退快路径 |
| `src/python/shmem/core/` | context、collective、typed heap、异常和导出 |
| `src/host/init/shmem_init.cpp`、`src/host/team/` | 保存/恢复每实例 team 状态并拒绝 finalized context |
| `setup.py`、`scripts/build.sh` | Python ≥3.9 依赖和确定性解释器选择 |
| `docs/api/pythonAPI.md`、`docs/quickstart.md` | API、完成边界、能力限制和复现步骤 |
| `examples/python_extension/` | 官方用例接入、功能/性能重放及证据身份校验 |

现有 `shmem.core.put` 的签名、异常和 stream 语义保持不变。AArch64 受控快路径
使用独立 host 前端，但与公开 `aclshmemx_putmem_on_stream` C++ 比较器汇入同一
`aclshmemi_prepare_and_post_rma` dispatcher 和 device path；本文不把它们误写为
同一 public C++ 入口。快路径每次读取当前 block 配置，失败时记录错误；缓存命中
与 cache-fill 持有 GIL，通用 public-API fallback 在一次 native 调用期间释放 GIL。
禁用快路径或不满足平台条件时继续使用 fallback，并以跨 PE bit-exact 用例验证
成功路径等价性。

# 可维可测分析

## 功能与正确性标准

| 层 | 必测内容 | 证据 |
|-|-|-|
| 零卡静态 | 20/13 接口、委托次数、宽度、GIL、锁、UID 回滚、Buffer 并发释放 | 命令、完整日志、退出码、源码哈希 |
| 构建装载 | CANN ≥ 9.0.0 configure/build/wheel/import | wheel/native library 哈希及加载路径 |
| 4 PE 正式、2 PE 附加 | allocator、context、collective、stream、profile、MTE 主路径 | 逐 rank 日志、机器信息、截图 |
| 条件能力 | HOST_SIDE、SDMA/RDMA/UDMA、跨节点 handle_wait | 能力探测和实际数据可见性 oracle |

RMA put/get 结果必须与 C++ 路径 bit-exact 或符合接口约定误差；Signal/handle_wait
完成后验证对端数据可见性；多实例切换后 heap/team 状态不得互相污染。旧用例和
官方自测必须零 FAIL，SKIP 只能来自经过认证的环境能力缺失。

## 性能标准

性能采用同一源码身份、同一 wheel/native library、同 shape、同 stream、相同
warmup/采样/排空方式的成对对照。C++ 分母调用公开
`aclshmemx_putmem_on_stream`；交付 Python 热路径使用不同 host 前端，但从
`aclshmemi_prepare_and_post_rma` 起共享 dispatcher 与 device path。记录逐 rank
原始样本、worst-rank 点估计和 95% 置信上界。

官方公开性能自测固定 `MSG_SIZE = 64 * 1024`、`ITERS = 100`、`WARMUP = 10`，
并在同一 stream 上完成 100 次入队后执行一次批次结束同步。64 KiB 是正式性能
负载；Python 必须与同 shape、同 stream、同批次完成边界的 C++ 路径比较，并
显式设置有效 C++ baseline。4 PE 的 overhead 不得超过 5%。本提交的冻结协议
把低层和高层路径的 worst-rank 点估计及 95% 置信上界都不超过 5% 作为内部
补充质量指标，而不是重定义官方门槛。
2 PE 是附加证据，不能替代 4 PE；4 KiB 和 1 MiB 仅作为诊断数据保留，不得替代
或重定义 64 KiB 官方门槛。2026-08-11 合入的任务书修订不再把 2/8 PE 列为
当前性能门禁。

如需解释新版“算子执行耗时”的组成，可额外记录 device-event/profiler 分解；该
诊断不能替代官方 `perf_counter + 100 次入队 + 同 stream 批末同步` 的主结果。

## 兼容性与风险

- 保留旧 `Buffer(addr, length)` owning/free 行为和 profiling 兼容调用形式。
- AArch64 可启用受控 fast path，其他平台及禁用配置使用 pybind11 fallback。
- native team/heap 状态变更只为多实例隔离服务，必须由独立测试证明不污染实例。
- 任何影响候选 tree 的修改都会生成新身份；旧 wheel/NPU/性能证据不得继承。
- 静态 PASS 不等于硬件 PASS；config 返回 0 不等于对应 engine 数据路径可用。

## 交付与公开边界

设计评审 PR 只提交本文，不包含实现源码。实现、wheel、native library、原始日志
和截图保存在仅审核人员可访问的私有仓库；最终向 `cann/shmem` 提交源码 PR 前，
需同步最新上游、重建并对新身份完成重放。每份报告都记录 source commit/tree、
wheel、native library、官方测试版本、测试协议和证据文件 SHA-256。

验收申请提供个人代码仓、固定 branch/commit、完整 README、自测脚本与报告，并
按组织方要求邀请 `Ascend-CANN` 作为 Developer。只有设计评审和验收完成后，才
以该冻结身份为基线向 `cann/shmem` 提交源码 PR；后续代码变化必须重新验收。
