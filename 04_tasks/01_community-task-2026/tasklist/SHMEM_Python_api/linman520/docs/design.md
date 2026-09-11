# 需求背景（required）

## 1. 来源

这次改动针对 CANN 2026 年 7 月的 SHMEM Python 接口任务：在已有工程上补齐 20 个 Host C++ API 的 pybind11 导出，再把它们接到 `shmem.core`，同时补测试和使用说明。

实现主要落在这些位置：

- Host 绑定入口：`src/host/python_wrapper/pyshmem.cpp`；
- Python 包：`src/python/shmem/`；
- Python API 文档：`docs/api/pythonAPI.md`；
- 快速开始：`docs/quickstart.md`；
- Python 测试和样例：`examples/python_extension/`；
- RDMA handle 语义参考：`examples/rdma_handlewait_test/`。

## 2. 背景

SHMEM（ACLSHMEM）提供昇腾集群的对称共享内存、单边 RMA、集合通信和同步能力。主线 Python 工程已经有初始化、基础内存分配、RMA 和动态库加载代码，缺的主要是下面这些接口：

1. UID 初始化属性构造和多实例上下文切换；
2. 带 `mem_type` 的对称堆 `malloc/calloc/align/free`；
3. MTE、SDMA、RDMA、UDMA 引擎配置；
4. Team/world barrier、sync 和 on-stream barrier；
5. `handle_wait`；
6. profiling 数据获取和展示。

这项工作不改 AI Core Kernel、`aclshmemx_roce_*` 等 Device API、通信协议或 Host C API ABI。Python 层只做参数转换、对象生命周期管理和异常适配，把已有 Host 能力接到 Python 上。

## 3. 现有代码

### 3.1 现有代码和缺口

| 位置 | 现在负责什么 | 这次怎么改 |
| --- | --- | --- |
| `src/host/python_wrapper/pyshmem.cpp` | pybind11 模块、初始化、基础内存、RMA、Stream 和日志相关绑定 | 沿用已有命名、指针和 GIL 处理；把任务书中的内部调用补成公开绑定 |
| `src/python/shmem/core/init_final.py` | `init/finalize` 等初始化高层接口 | 扩展 UID 属性、多实例参数时保持旧调用兼容 |
| `src/python/shmem/core/memory.py` | `buffer/free/get_peer_buffer` | 增加 `mem_type`、Buffer 元数据和安全释放 |
| `src/python/shmem/core/rma.py` | put/get/signal/quiet 等 RMA 封装 | 增加 `handle_wait` 和引擎配置，保持现有 RMA 语义 |
| `src/python/shmem/core/__init__.py` | 汇总已有高层导出 | 汇总新增 collective、multi-instance、config、profiling 接口 |
| `examples/python_extension/run.sh` | 安装 wheel 并运行已有 torchrun 测试 | 保留原有测试，在同一入口加入新增用例；任一 rank 失败都返回非零 |

`pyshmem.cpp` 目前在初始化流程内部调用 `aclshmemx_set_attr_uniqueid_args`，也已经绑定了基础内存和 `aclshmemx_get_heap_base`。新增绑定沿用这里已有的地址转换、Stream 处理和异常方式，不再另起一套封装。

### 3.2 参数表示

这里不是 Tensor 算子，不处理 ND/NCHW 之类的数据格式。Python 只需要正确表示 C/C++ 标量、枚举、指针、Stream、Team、Handle 和 profiling 结构。

| 原生值 | Python 侧表示 | 进入 C++ 前的处理 |
| --- | --- | --- |
| `int`、`int32_t` | `int` | 类型检查；按 C++ 目标范围拒绝溢出 |
| `uint32_t`、`uint64_t` | `int` | 禁止负数，检查上界 |
| `size_t`、`int64_t` | `int` | 分配参数非负；`calloc` 检查乘法溢出 |
| `void *` | `intptr_t` 或 `Buffer` | `0` 只按 C++ 接口允许的空指针语义处理 |
| `aclrtStream` | `int`/`intptr_t` 或 `None` | 只在高层约定允许时将 `None` 解释为默认流 |
| `aclshmem_team_t` | `int` | `ACLSHMEM_TEAM_WORLD` 的数值以头文件为准 |
| `aclshmem_mem_type_t` | `MemType` | 只能使用 Host 头文件定义的枚举值 |
| `aclshmem_handle_t` | `Handle` | 只表示真实异步操作产生的完成句柄，不伪造完成状态 |
| profiling 输出结构 | Python 不可变快照 | 不跨越原生调用保存内部指针 |

### 3.3 范围

下面的 20 个 Host API 是新增清单。`pyshmem.cpp` 里已有的 `aclshmemx_get_heap_base` 等接口继续兼容，但不放进这次新增统计。

# 需求分析（required）

## 1. 要做什么

用 pybind11 把 20 个 Host C++ API 暴露到 `shmem._pyshmem`，不改变 C++ 的参数顺序、返回值、集合语义、Stream 排队语义和错误行为。`shmem.core` 按现有 `init`、`buffer`、`put`、`get` 的用法提供高层入口，并同步更新 API 文档、QuickStart、测试脚本和自测报告。

## 2. 改动清单

1. 在 `_pyshmem` 中补齐 20 个 API，并逐项对照头文件；
2. 在 `shmem.core` 中补齐集合通信、多实例、`mem_type` 内存和 `handle_wait`；
3. 沿用已有的指针、Stream、GIL 和异常约定；
4. 用 C++ 路径对照 RMA、Signal/handle、多实例隔离和性能；
5. 更新 API 文档、QuickStart、README 和自测报告。

## 3. 外部依赖与环境

| 依赖 | 要求 | 用途 |
| --- | --- | --- |
| Python | >= 3.9 | Python 包和测试 |
| CANN | 9.0.0 或较新社区版 | Runtime、Host 编译和运行 |
| pybind11 | 沿用仓库版本 | C++/Python 绑定 |
| torch + torch_npu | 版本相互匹配 | `torchrun` 多 PE 测试和 Stream |
| Atlas A2/A3 | 至少 2 卡主路径 | MTE、集合、多实例验证 |
| RDMA/SDMA/UDMA 环境 | 可选 | 对应引擎 smoke 和跨机 handle 验证 |

不新增 MPI、NumPy 或其他 Python 运行时依赖。构建前加载 CANN 环境，例如 `source /usr/local/Ascend/ascend-toolkit/set_env.sh`；实际安装路径以测试机器为准。

## 4. 低层 20 项接口清单

实现时按下面的清单逐项添加绑定。20 个名字都要能从 `_pyshmem` 直接导入；只在 `shmem.core` 里间接调用不算完成。

| 序号 | C++ Host API | `_pyshmem` 入口 | 实现要点 |
| ---: | --- | --- | --- |
| 1 | `aclshmemx_set_attr_uniqueid_args` | 同名函数 | UID、rank、PE 数和内存参数按 C++ 语义校验；返回原生错误码 |
| 2 | `aclshmemx_instance_ctx_get` | 同名函数 | 返回安全的上下文快照或 C++ 约定的空值 |
| 3 | `aclshmemx_instance_ctx_set` | 同名函数 | 按 `instance_id` 切换；失败不应伪称切换成功 |
| 4 | `aclshmemx_malloc` | 同名函数 | `size + mem_type`，返回 `intptr_t` 地址 |
| 5 | `aclshmemx_calloc` | 同名函数 | `count + size + mem_type`，检查乘法溢出 |
| 6 | `aclshmemx_align` | 同名函数 | `alignment + size + mem_type`，对齐约束交给头文件/运行时最终判断 |
| 7 | `aclshmemx_free` | 同名函数 | 地址和 `mem_type` 与分配保持一致 |
| 8 | `aclshmemx_set_mte_config` | 同名函数 | offset/UB size/sync_id，保留原生返回码 |
| 9 | `aclshmemx_set_sdma_config` | 同名函数 | 同上，平台能力由原生层判断 |
| 10 | `aclshmemx_set_rdma_config` | 同名函数 | 同上，跨机能力不由 Python 伪造 |
| 11 | `aclshmemx_set_udma_config` | 同名函数 | 以真实 Host 声明为准，不擅自增加参数 |
| 12 | `aclshmem_barrier` | 同名函数 | Team 集合调用，阻塞语义保持一致 |
| 13 | `aclshmem_barrier_all` | 同名函数 | 全局集合调用，阻塞语义保持一致 |
| 14 | `aclshmem_sync` | 同名函数 | Team 内存可见性语义，不当作 barrier |
| 15 | `aclshmem_sync_all` | 同名函数 | 全局 sync，不当作 barrier |
| 16 | `aclshmemx_barrier_on_stream` | 同名函数 | 将 Team barrier 排入指定 ACL Stream |
| 17 | `aclshmemx_barrier_all_on_stream` | 同名函数 | 将全局 barrier 排入指定 ACL Stream |
| 18 | `aclshmemx_handle_wait` | 同名函数 | 等待真实 RDMA/Stream handle；不以线程睡眠替代 |
| 19 | `aclshmemx_get_prof` | 同名函数 | 返回 Python 持有的深拷贝快照 |
| 20 | `aclshmemx_show_prof` | 同名函数 | 保留原生展示行为，明确诊断用途 |

如果头文件声明与表中参数描述不同，以头文件为准，并同步修改 API 文档和测试；不要为了迁就表格去改 C++ ABI。

## 5. 高层接口清单

高层接口按使用场景分成三组：

| 组 | Python API | 责任 |
| --- | --- | --- |
| 集合与同步 | `barrier`、`barrier_all`、`sync`、`sync_all`、on-stream 变体 | 默认 Team、Stream 分派、异常转换和集合约束说明 |
| 多实例与内存 | `instance_ctx_get/set`、`multi_instance`、`buffer/calloc/aligned_buffer/free` | 上下文恢复、`mem_type`、Buffer 归属和生命周期 |
| 完成等待 | `handle_wait` | 真实 Handle/Stream 校验和原生等待语义 |

引擎配置和 profiling 另提供高层函数，但不改变任务书中 20 个低层接口和 3 组高层功能的统计方式。

# 详细设计（required）

## 1. 算子分析

这里没有新的数学算子，也没有 Tensor shape、dtype、广播或 tiling 数据。计算和通信仍由 SHMEM Host C++ API、Runtime 及 MTE/SDMA/RDMA/UDMA 引擎完成；Python 只做接口适配。

对应的调用链如下：

```text
Python 用户代码
    -> shmem.core（Python 参数、状态、生命周期）
    -> shmem._pyshmem（pybind11 类型转换和 GIL）
    -> SHMEM Host C++ API
    -> Runtime / 通信引擎
```

## 2. 算子实现

### 2.1 Host 侧总体方案

低层绑定尽量一一对应 C++，Pythonic 行为放在高层完成：

| 层 | 做什么 | 不做什么 |
| --- | --- | --- |
| `_pyshmem` | 注册函数、枚举和结构；转换指针、Stream、Team、Handle；释放阻塞调用的 GIL；保留返回码 | 不替其他 PE 调集合操作，不替调用者同步 Stream，不改 C++ 参数顺序 |
| `shmem.core` | 做参数和状态检查；把错误码转成异常；管理 Buffer 和上下文；提供默认值 | 不改 barrier/sync/handle 的完成语义，不把没有硬件条件的 RDMA 说成可用 |

### 2.2 pybind11 绑定规则

1. 指针使用 `intptr_t` 传递，Python 整数到原生指针的转换在调用边界完成；不能把裸指针保存在 Python 对象中跨越 `finalize`。
2. `aclrtStream` 使用与现有 `put/get/on_stream` 绑定一致的整数句柄转换。低层 `None` 是否可用由具体 C++ API 的现有约定决定；高层只在文档明确的函数中提供默认 Stream。
3. C++ `void` 映射为 Python `None`；C++ 错误码函数低层保留 `int`，高层统一检查非零返回值并抛出 `AclshmemError`。
4. 绑定层对可能阻塞或耗时的 Host 调用使用 `py::gil_scoped_release`，包括 barrier、sync、内存分配/释放、实例切换和 handle wait；纯参数转换在持有 GIL 时完成。
5. C++ 输出结构含二级指针时，在低层调用返回前完成复制；Python 对象只保存值，不保存原生 `out_profs` 等内部地址。
6. pybind11 类型名称、枚举名称和异常命名沿用当前仓库风格；若已有导出符号，不新增同义别名造成 API 分裂。

### 2.3 API 映射与高层原型

#### 2.3.1 初始化与多实例

```python
_pyshmem.aclshmemx_set_attr_uniqueid_args(..., uid, attr) -> int
_pyshmem.aclshmemx_instance_ctx_get() -> InstanceContext | None
_pyshmem.aclshmemx_instance_ctx_set(instance_id: int) -> int

core.instance_ctx_get() -> int
core.instance_ctx_set(instance_id: int) -> None
core.multi_instance(instance_id: int)  # context manager
```

`set_attr_uniqueid_args` 的 uid、rank、PE 数、内存大小和 `InitAttr` 字段跟 Host 头文件走。绑定层接受当前 `get_unique_id()` 返回的 `bytes`，也接受实现已支持的 buffer-like 输入；进入 C++ 调用前把 UID 复制到绑定层拥有的连续存储中，并让 `InitAttr` 的 Python 包装对象持有这份 owner，直到初始化调用返回。这样 C++ 只会在有效期内使用稳定地址，不把临时 Python buffer 的地址交给 C++ 后马上释放；若后续绑定改为异步保存参数，必须改为由 `InitAttr` 自身拥有副本，不能只依赖 `py::keep_alive`。

实例上下文是进程内状态，不是线程局部的 Python 普通变量。`multi_instance` 采用以下流程：

```text
读取 old_instance
    -> 切换 new_instance
    -> 执行用户代码
    -> finally 恢复 old_instance
```

切换失败就不进入用户代码；恢复失败直接抛错，不能静默处理。原生库没有线程安全保证时，同一进程不要让多个 Python 线程同时切实例。`Buffer` 和 `Handle` 记录实例或 Team，释放和等待时据此检查上下文。

#### 2.3.2 对称堆和 Buffer

```python
core.buffer(size, release=False, except_on_del=True,
            mem_type=MemType.DEVICE_SIDE) -> Buffer
core.calloc(count, size, mem_type=...) -> Buffer
core.aligned_buffer(alignment, size, mem_type=...) -> Buffer
core.free(buf, mem_type=None) -> None
```

`Buffer` 至少保存：

```text
addr: int
length: int
mem_type: MemType
instance_id: int
freed: bool
release: bool
except_on_del: bool
```

分配按下面的顺序走：

1. 检查参数是整数、非负且在 `size_t`/C++ 目标类型范围内；
2. `calloc` 在调用 C++ 前检查 `count * size` 是否超过 `SIZE_MAX`；
3. `align` 检查正数和基础格式，平台特有约束交给 Host API 最终判断；
4. 读取当前实例，调用对应 `aclshmemx_*` 接口；
5. 空指针或非零错误码按高层异常策略处理；
6. 构造带实例和 `mem_type` 的 Buffer。

释放按下面的顺序走：

1. 检查对象类型、未释放状态和当前实例；
2. 检查显式 `mem_type` 与 Buffer 记录一致；
3. 调用 `aclshmemx_free`，成功后再设置 `freed=True`；
4. 重复释放、错误实例和错误内存类型抛出 `AclshmemInvalid`；
5. 析构函数不在不确定的集合上下文中隐式执行集体 free。

所有 PE 仍要按相同顺序、大小和 `mem_type` 分配和释放；Python 不替其他 PE 补调用。

#### 2.3.3 引擎配置

高层提供：

```python
core.set_mte_config(offset, ub_size, sync_id) -> None
core.set_sdma_config(offset, ub_size, sync_id) -> None
core.set_rdma_config(offset, ub_size, sync_id) -> None
core.set_udma_config(offset, ub_size, sync_id) -> None
```

Python 层检查无符号范围：`offset >= 0`、`ub_size > 0`、`sync_id >= 0`，上界跟 C++ 声明走。UB 对齐、引擎可用性、链路状态和平台最小值交给原生实现判断；接口能调用不代表机器具备对应引擎。配置作用于当前实例，并在相关 RMA 操作前完成。

MTE 是主路径；SDMA、RDMA、UDMA 有环境再做 smoke。跨机 RDMA 要用 C++ 示例和实际链路验证。

#### 2.3.4 集合通信和 Stream

```python
core.barrier(team=TEAM_WORLD, stream=None) -> None
core.barrier_all(stream=None) -> None
core.sync(team=TEAM_WORLD) -> None
core.sync_all() -> None
core.barrier_on_stream(team, stream) -> None
core.barrier_all_on_stream(stream) -> None
```

分派规则：

| 高层调用 | 原生路径 | 完成边界 |
| --- | --- | --- |
| `barrier(team, stream=None)` | `aclshmem_barrier` | Host 阻塞式 Team barrier |
| `barrier(team, stream=s)` | `aclshmemx_barrier_on_stream` | 排入 `s`，Host 是否完成由 Stream 同步确认 |
| `barrier_all(stream=None)` | `aclshmem_barrier_all` | Host 阻塞式全局 barrier |
| `barrier_all(stream=s)` | `aclshmemx_barrier_all_on_stream` | 排入 `s`，不由 Python 自动同步 |
| `sync(team)` | `aclshmem_sync` | C++ 定义的 Team 可见性语义 |
| `sync_all()` | `aclshmem_sync_all` | C++ 定义的全局可见性语义 |

`sync` 不是 `barrier` 的别名，不能拿来等远端更新完成。Team 内的 PE 要按相同顺序调用相同集合接口；一个 rank 出错时其他 rank 可能一直等，所以测试要有超时并能结束整个测试组。

#### 2.3.5 `handle_wait` 约定

```python
core.handle_wait(handle: Handle, stream: int | None = None) -> None
```

`Handle` 是 team-scoped 的等待条件，不是某次 RMA 返回的 future。当前 `aclshmem_handle_t` 只包含 `team_id`；Host RMA API 不为每次操作返回一个 Handle，调用方按 team 主动构造 Handle，官方 `rdma_handlewait_test` 也遵循这一约定。因此高层可以接收显式构造的 `Handle`，并保留 team id 和默认 `TEAM_WORLD` 的兼容入口，最终由底层按 C++ 规则解释。`Handle` 不拥有 Stream、Team 或 runtime 资源，也不负责这些资源的生命周期。

不能用 Python 线程、sleep、barrier 或 sync 代替 `aclshmemx_handle_wait`。Handle 的 Team/实例关联信息用于参数和上下文校验；Stream 必须与产生异步操作的路径匹配。`None` 只在现有 Stream 约定表示默认 Stream 时使用。

验证沿用 `examples/rdma_handlewait_test/` 的调用顺序：提交异步操作、调用 handle wait、检查数据可见性。没有 RDMA 时只报告接口 smoke，不把它写成跨机完成语义通过。

#### 2.3.6 profiling

低层 `aclshmemx_get_prof` 返回前完成深拷贝，高层返回只读 Python 快照；`aclshmemx_show_prof` 保留原生打印兼容性。快照不得引用 `aclshmem_prof_pe_t **` 等原生输出指针。profiling 是显式诊断接口，不进入 RMA 热路径；大 PE 数量下测试需记录快照内存成本。

Python 侧可以用不可变对象保存快照：

```text
ProfPe(pe_id: int, block_prof: tuple[ProfBlock, ...])
ProfBlock(ccount: tuple[int, ...], cycles: tuple[int, ...])
```

具体字段以 Host 头文件为准；结构新增字段时一起复制，不能只保留示例中的字段。

### 2.4 模块改动规划

| 文件 | 改动 | 检查方式 |
| --- | --- | --- |
| `src/host/python_wrapper/pyshmem.cpp` | 20 个函数、必要枚举/结构、GIL 和指针处理 | import、签名和返回值测试 |
| `src/python/shmem/core/init_final.py` | UID 属性和实例初始化兼容扩展 | 初始化、多实例测试 |
| `src/python/shmem/core/memory.py` | `mem_type` 分配、Buffer 元数据和安全 free | mem_type/lifecycle 测试 |
| `src/python/shmem/core/collective.py` 或现有同步模块 | barrier/sync/on-stream 高层封装 | CC 多 PE 测试 |
| `src/python/shmem/core/multi_instance.py` | ctx get/set 和上下文管理器 | 两实例隔离测试 |
| `src/python/shmem/core/config.py` | 四类引擎配置和参数检查 | MTE/可选引擎 smoke |
| `src/python/shmem/core/rma.py` | `handle_wait` | C++ 同构测试 |
| `src/python/shmem/core/profiling.py` | profiling 快照和展示 | 生命周期测试 |
| `src/python/shmem/core/__init__.py` | 新接口公开导出 | 导入测试 |
| `docs/api/pythonAPI.md` | 完整 API、异常、集合/Stream/引擎限制 | 文档审查 |
| `docs/quickstart.md` | 构建安装、多卡和最小调用 | 按文档执行 |
| `examples/python_extension/test/` | 新增功能和错误路径 | 测试报告 |
| `examples/python_extension/README.md` | 环境、构建、安装、测试和日志保存 | 交付审查 |

模块可以按主线维护者意见合并；合并后仍要覆盖表中的职责和测试。

## 3. Host 调用后的后端路径

这里不新增或修改 AI Core Kernel，也不使用 Ascend C `DataCopy`、`LocalTensor`、tilingKey 或自定义 UB 队列。Python 调用 Host C++ API 后，Runtime 仍按实例初始化、引擎配置和设备能力选择原有路径。

```text
Python 参数/状态检查
    -> pybind11 参数转换
    -> 释放 GIL
    -> Host C++ API
    -> Runtime 选择既有通信引擎
    -> 返回码/指针/快照安全转换
    -> Python 异常或对象
```

# 支持硬件

| 硬件 | 覆盖范围 | 最低验证要求 |
| --- | :---: | --- |
| Atlas A2 训练系列 | 是 | 2 PE MTE 主路径、集合、内存和回归 |
| Atlas A3 训练系列 | 是 | 2/4 PE MTE 主路径；具备条件时补 SDMA/RDMA |
| Atlas A2/A3 推理系列 | 是 | 按可用设备完成对应多卡验证 |
| Ascend 950 | 不在这次范围内 | 不纳入通过结论 |

RDMA、SDMA、UDMA 的结果写明硬件、链路、CANN 和引擎条件；没有真实环境时记录为未执行或 smoke，不写成“已支持”。

# 使用限制

1. 20 个接口的原生签名、枚举和错误码以 Host 头文件为准，Python 不改 C++ ABI。
2. `barrier`、`sync` 和对称堆分配/释放涉及集合语义；所有 PE 要同步、同序、同参数调用。
3. `sync` 不等价于 barrier；Stream barrier 只排队，不由 Python 偷偷同步。
4. Stream 要属于正确设备且仍然有效；非法或已销毁 Stream 由 Python 预检和原生层共同拒绝。
5. 实例切换影响后续 heap、Team、引擎配置和 finalize；Buffer/Handle 不能跨实例使用。
6. `mem_type` 要与分配一致；错误类型、错误实例和重复 free 要能稳定复现，并抛出明确异常。
7. `handle_wait` 不能伪造 Handle，不能用线程等待、sleep、barrier 或 sync 替代。
8. RDMA/SDMA/UDMA 依赖硬件和链路；未验证项在报告中单列。
9. `get_prof` 仅用于诊断，不应在训练或通信热循环中高频调用。
10. Device API、通信协议、transport 算法和已有 RMA 功能不在这次修改范围。

# 可维可测分析

## 1. 怎么判断正确

| 类别 | 通过条件 | 记录 |
| --- | --- | --- |
| 绑定 | 20 个符号可导入，参数/返回类型和 C++ 语义一致 | import/signature 日志 |
| RMA | put/get 与 C++ 路径 bit-exact 或落在约定误差内 | 2/4/8 PE 结果 |
| Signal/Handle | wait 后对端数据可见性与 C++ 示例一致 | `rdma_handlewait_test` 日志 |
| 多实例 | heap、Team、配置和数据不串 | 切换前后断言 |
| 集合 | barrier/sync/on-stream 不超时、不死锁 | torchrun 日志 |
| 生命周期 | 错误实例、错误 mem_type、重复 free、空指针行为明确 | 错误路径测试 |
| profiling | 返回和 finalize 后都没有悬空引用 | 生命周期测试 |

## 2. 功能用例

| 用例组 | 最小覆盖 |
| --- | --- |
| UID/InitAttr | 合法 UID、短/非法 UID、rank/PE 边界、内存大小、初始化失败 |
| context | get、set 合法实例、set 不存在实例、异常恢复 |
| mem_type heap | DEVICE/HOST 的 malloc/calloc/align/free、清零、对齐、溢出 |
| free 安全性 | 重复释放、错误实例、错误类型、非法地址、集合顺序 |
| engine config | `test_config_diag.py` 覆盖 MTE/SDMA/RDMA/UDMA config smoke、边界和错误码 |
| collective | WORLD/Team barrier、sync、2/4/8 PE（具备设备时）、超时保护 |
| Stream | 默认/显式 Stream、前后事件、on-stream 不自动同步 |
| handle | `test_sync.py` 完成 Handle 构造和接口 smoke；RDMA 环境具备时再验证跨机可见性 |
| profiling | 初始化前后、verbose/展示、快照内容、finalize 后生命周期 |
| 回归 | `examples/python_extension/run.sh` 原有全部测试 |

多 PE 用例用 `torchrun` 启动。脚本设置超时；一个 rank 失败时结束整个测试组，并保存各 rank 的日志。

## 3. 性能比较

Python Host RMA 以 `putmem_on_stream` 为代表，与同 shape、同 PE 数、同 Stream、同引擎、同同步边界的 C++ 路径比较：

```text
相对开销 = (Python 路径耗时 - C++ 路径耗时) / C++ 路径耗时 × 100%
```

任务要求 2/4/8 卡。卡数不足时只记录已跑的规模，并注明缺少的规模。性能脚本至少覆盖 4 KiB、64 KiB、1 MiB 和 8 MiB message，其中 64 KiB 对齐官方验收脚本作为基础门禁，并保留小包结果以暴露 pybind11、GIL 和参数转换的固定开销；不能只用 8 MiB 作为唯一门禁。每个消息尺寸在 Python/C++ 两条路径上使用相同 shape、PE 数、Stream、引擎和同步边界，使用 5 次 warmup、30 次正式迭代，输出 average、带宽、原始轮次、硬件拓扑、CANN/Python/torch 版本和环境变量；Python 相对 C++ 的 overhead 要 <= 5%。

barrier 和 barrier_on_stream 另做正确性/超时测试，不把一次成功调用直接当作性能结论。

## 4. 运行顺序

README 至少写清下面这条路径，具体构建目标跟主线脚本保持一致：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
## 构建扩展并生成 wheel
bash scripts/build.sh -python_extension
python3 setup.py bdist_wheel

## 安装本次生成的 wheel，然后运行原有回归测试
pip3 install dist/shmem-*.whl --force-reinstall
bash examples/python_extension/run.sh
PYTHON_EXTENSION_NPROC_PER_NODE=4 bash examples/python_extension/run.sh
```

专项测试由现有 `run.sh` 串起；同步/handle 用例集中在 `test_sync.py`，引擎和诊断用例集中在 `test_config_diag.py`。性能入口为：

```bash
bash examples/python_extension/perf/run_putmem_perf.sh --pe 2
bash examples/python_extension/perf/run_putmem_perf.sh --pe 4
bash examples/python_extension/perf/run_putmem_perf.sh --pe 8
```

UT 日志和截图属于本地验收产物或 CI artifact，默认不提交 Git，以免引入大文件、机器信息和不可复现内容。仓内只保留测试脚本、README 和结构化的小型报告；报告中记录 artifact 的获取位置或 CI 链接。RDMA handle 的 runtime smoke 通过环境变量显式打开：

```bash
ACLSHMEM_RUN_RDMA_HANDLE_WAIT=1 bash examples/python_extension/run.sh
```

## 5. 报告和交付物

报告记录测试名称、命令、PE 数、硬件/引擎、返回码、关键输出、日志和截图的本地/CI artifact 位置；RDMA/8PE 等未执行项写明原因。性能部分保留 Python/C++ 原始数据和 overhead 计算，不能只写“通过”。日志和截图不作为仓内交付文件。

交付件清单：

1. 本文档；
2. `_pyshmem` 绑定和 `shmem.core` 实现；
3. `docs/api/pythonAPI.md`、`docs/quickstart.md` 和 `examples/python_extension/README.md`；
4. `examples/python_extension/perf/` 中的结构化 2PE/4PE/8PE 性能报告（已执行规模）；
5. `docs/api/python_extension_selftest_report.md` 自测报告，以及个人仓库、分支和 PR 信息。UT 日志、完整原始轮次、环境信息和截图作为本地/CI artifact 保存，不直接提交到仓库。

# 兼容性分析

| 兼容项 | 风险 | 处理方式 |
| --- | --- | --- |
| 旧 Python API | 在尾部增加参数可能影响位置调用 | 新增 `mem_type` 放在尾部并提供旧默认值；旧 `buffer/free/put/get` 调用回归 |
| `_pyshmem` ABI | 改动 C++ 类型或签名会破坏二进制 | 只增加 pybind 注册，不改 Host 头文件和既有函数 ABI |
| Buffer | 新旧 Buffer 元数据不同 | 保留 `addr/length` 访问，新增字段不破坏旧消费者 |
| GIL | 阻塞调用阻塞解释器 | 原生阻塞区释放 GIL，参数转换仍在 GIL 内 |
| Stream | 自动同步会改变性能和完成边界 | 不自动同步；文档明确排队和完成边界 |
| 多实例 | 全局上下文被错误恢复或跨实例释放 | 上下文管理器 `finally` 恢复，Buffer/Handle 记录归属 |
| 引擎 | “接口可调用”被误认为“硬件可用” | 报告拆分 MTE 必测和其他引擎条件性 smoke |
| profiling | 原生输出指针悬空 | 显式调用时深拷贝为 Python 快照 |
| degraded mode | 原生库缺失时导入行为变化 | 沿用现有加载守卫和异常风格，不静默伪造接口成功 |
