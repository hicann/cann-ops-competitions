# SHMEM Python Host API Extension Design

| 项目 | 内容 |
|-|-|
| 社区任务 | 7 月社区任务—SHMEM Python 接口开发 |
| 团队名称 | 土豆饼 |
| 实现仓库 | `cann/shmem` |
| 目标硬件 | Atlas A2/A3 训练系列产品或推理系列产品 |

## 需求背景

### 需求来源

本设计对应社区任务《SHMEM Python 接口开发任务书》，目标是在现有 `cann/shmem` Python 工程中补齐尚未导出的 Host C++ API 的 pybind11 绑定，并在 `shmem.core` 提供风格一致的高层 Pythonic 封装。

### 背景介绍

ACLSHMEM 已在 Host C++ 层提供初始化、多实例、对称 heap、集合通信、stream barrier、handle wait 与 profiling 等能力。当前 Python 包已有 `init`、`buffer`、`put`、`get` 等基础封装，但部分 Host API 仍只能从 C++ 使用，导致 Python 用户无法完整覆盖多实例、mem_type heap、CC 同步和诊断场景。

## 需求分析

### 需求描述

本变更需要完成两层封装：

1. `_pyshmem`：使用 pybind11 直接导出任务书列出的 20 个 Host API，保持底层命名和 C++ 语义。
2. `shmem.core`：在底层绑定上提供高层封装，统一 `Buffer`、stream 整数句柄、异常和默认值约定。
3. 文档与示例测试：更新 API 文档、QuickStart 和 `examples/python_extension` 测试入口。

Device 侧 Kernel API，例如 `aclshmemx_roce_*`，不属于本任务范围。

### 需求拆解与接口映射

下表中的“异常/返回”同时定义底层绑定和高层包装的错误边界。Python 对象、枚举和所有借用指针的检查/值复制均在持有 GIL 时完成；只有在不再访问 Python 对象或借用指针后，阻塞、等待设备或参与集合通信的原生调用才释放 GIL。实例状态转换还必须持有下述实例锁，profiling 使用独立的 C++ 锁。

| # | C++ Host API | `_pyshmem` 签名 | `shmem.core` 接口 | 异常/返回约定 |
|-:|-|-|-|-|
| 1 | `aclshmemx_set_attr_uniqueid_args(my_pe, n_pes, local_mem_size, uid*, attr*) -> int` | `aclshmemx_set_attr_uniqueid_args(rank, nranks, mem_size, uid, attr) -> int` | `set_attr_uniqueid_args(rank, nranks, mem_size, uid, instance_id=0) -> InitAttr` | 底层保留 C++ 的输出参数、返回码和错误行为；`InitAttr` 的 C++ 包装私有持有 UID 深拷贝，并以 candidate + commit 保证失败不改变旧值；高层另将非 0 返回码转换为异常。 |
| 2 | `aclshmemx_instance_ctx_get() -> aclshmem_instance_ctx*` | `aclshmemx_instance_ctx_get() -> InstanceCtx`（内部 snapshot helper 复制值） | `ctx_get() -> int` | 空指针转换为异常；`InstanceCtx` 是仅含 `id` 的不可变值快照，不向 Python 暴露全局 context 借用指针。getter 的 native 调用、判空和 `id` 复制全程持有 GIL 及 `g_aclshmem_ctx_mutex`。 |
| 3 | `aclshmemx_instance_ctx_set(instance_id) -> int` | 同 C++ | `ctx_set(instance_id) -> None` | 高层将非 0 返回码转换为 `AclshmemError`。 |
| 4 | `aclshmemx_malloc(size, mem_type) -> void*` | `aclshmemx_malloc(size, mem_type=None) -> int \| None` | `malloc(size, mem_type=None) -> Buffer` | 指针用 `intptr_t`；底层对零长度或分配失败均原样返回 `0`/`None`；高层预先拒绝零长度，对非零请求返回空指针抛 `AclshmemError`；默认 `DEVICE_SIDE`。 |
| 5 | `aclshmemx_calloc(count, size, mem_type) -> void*` | `aclshmemx_calloc(nmemb, size, mem_type=None) -> int \| None` | `calloc(...) -> Buffer` | 底层保留全部空指针返回；高层拒绝零元素或零元素大小，并对非零请求返回空指针抛 `AclshmemError`。 |
| 6 | `aclshmemx_align(alignment, size, mem_type) -> void*` | `aclshmemx_align(alignment, size, mem_type=None) -> int \| None` | `align(...) -> Buffer` | 底层保留全部空指针返回；高层拒绝零长度，并对非零请求返回空指针抛 `AclshmemError`；alignment 原样传递给 C++ 校验。 |
| 7 | `aclshmemx_free(ptr, mem_type) -> void` | `aclshmemx_free(ptr, mem_type=None) -> None` | `free(buf, mem_type=None) -> None` | 高层只允许释放当前实例中由分配接口创建且尚未释放的 owned `Buffer`；在持有 GIL 和实例锁时校验实例及归一化后的 `mem_type`，立即置 `release_called`，再释放 GIL 调用原生 void 接口，状态不回滚。 |
| 8 | `aclshmemx_set_mte_config(offset, ub_size, sync_id) -> int` | 同 C++ | `set_mte_config(...) -> None` | 高层将非 0 返回码转换为 `AclshmemError`。 |
| 9 | `aclshmemx_set_sdma_config(offset, ub_size, sync_id) -> int` | 同 C++ | `set_sdma_config(...) -> None` | 同上。 |
| 10 | `aclshmemx_set_rdma_config(offset, ub_size, sync_id) -> int` | 同 C++ | `set_rdma_config(...) -> None` | 同上；RDMA staging `ub_size >= 128`。 |
| 11 | `aclshmemx_set_udma_config(offset, ub_size, sync_id) -> int` | 同 C++ | `set_udma_config(...) -> None` | 同上；UDMA staging `ub_size >= 128`。 |
| 12 | `aclshmem_barrier(team) -> void` | 同 C++ | `barrier(team=0, stream=None) -> None` | 集合调用；无 C++ 返回码。 |
| 13 | `aclshmem_barrier_all() -> void` | 同 C++ | `barrier_all(stream=None) -> None` | 全 PE 集合调用。 |
| 14 | `aclshmem_sync(team) -> void` | 同 C++ | `sync(team=0) -> None` | 集合调用；不保证 ACLSHMEM RMA/AMO 完成。 |
| 15 | `aclshmem_sync_all() -> void` | 同 C++ | `sync_all() -> None` | 全 PE 集合调用；语义同 sync。 |
| 16 | `aclshmemx_barrier_on_stream(team, aclrtStream) -> void` | `aclshmemx_barrier_on_stream(team, stream: int) -> None` | `barrier_on_stream(team, stream)`，亦可用 `barrier(..., stream=...)` | `0`/`None` 表示默认 stream；调用只负责入队。 |
| 17 | `aclshmemx_barrier_all_on_stream(aclrtStream) -> void` | `aclshmemx_barrier_all_on_stream(stream: int) -> None` | `barrier_all_on_stream(stream)`，亦可用 `barrier_all(stream=...)` | 同上。 |
| 18 | `aclshmemx_handle_wait(handle, aclrtStream) -> void` | `aclshmemx_handle_wait(Handle, stream: int) -> None` | `handle_wait(handle, stream) -> None` | `Handle` 是调用方按 team 主动构造的等待条件，原生载荷仅含 `team_id`，不是 RMA 返回的 future，也不拥有 stream/team/runtime 资源；等待所用 team、实例和显式非零 stream 必须与待完成的 RDMA 路径匹配。低层保留原生 `0`/`nullptr` 语义。 |
| 19 | `aclshmemx_get_prof(out_profs**, verbose) -> void` | `aclshmemx_get_prof(out_profs, verbose) -> list[ProfRecord] \| None` | `get_prof(verbose=True) -> ProfData \| None` | 底层兼容 `aclshmemx_get_prof(None, False)` 等原生调用形式；专用 C++ mutex 的锁区覆盖原生查询、判空以及 `pe_id` 和全部记录的值复制，返回值不依赖共享 C++ 缓冲区生命周期。 |
| 20 | `aclshmemx_show_prof() -> void` | 同 C++ | `show_prof() -> None` | 保留 deprecated 打印入口并与 `get_prof` 共用 profiling mutex；新代码优先使用 `get_prof`。 |

## 详细设计

### 分层设计

#### 底层 pybind11 绑定

文件：`src/host/python_wrapper/pyshmem.cpp`

- 保留 C++ Host API 名称，新增绑定导出到 `shmem._pyshmem`。
- 指针参数和返回值统一按 `intptr_t` 暴露给 Python。
- `aclrtStream` 由 Python `int` 传入，`0` 表示 `nullptr`。
- 阻塞或集合类接口仅在完成 Python 参数转换、借用指针值复制和必要的状态预提交后使用 `py::gil_scoped_release`；不得把原生借用指针带出持有 GIL/对应 mutex 的区间。
- 新增 `MemType`、`Handle` 和不可变 `InstanceCtx` 值类型。`Handle` 的原生载荷只对应 `aclshmem_handle_t.team_id`；`InstanceCtx` 只复制 `aclshmem_instance_ctx.id`，不保存 `aclshmem_instance_ctx *` 或其 `instance` 字段。须在保存实例状态的 C++ 翻译单元增加内部 snapshot helper：它在 `g_aclshmem_ctx_mutex` 内取得当前 context、判空并复制 `id`，再返回值对象；pybind 不得把公开 getter 返回的裸指针直接包装为 Python 引用。context getter 从调用该 helper、判空到复制 `id` 全程持有 GIL，helper 内部同时持有同一 C++ instance mutex；高层 `init/set/finalize` 由实例 `RLock` 串行，native 函数自行管理其内部 context mutex，绑定层不得持有该 mutex 再重入 native。仅在 Python 状态预检查完成后，阻塞 native 段释放 GIL，避免锁顺序反转。
- `InitAttr` 使用专用 C++ wrapper，私有持有 `aclshmemx_init_attr_t attr_` 和可选的 `aclshmemx_uniqueid_t uid_owner_` 深拷贝，不依赖 `py::dynamic_attr` 或运行时 Python 属性。`aclshmemx_set_attr_uniqueid_args` 严格接收 `(rank, nranks, mem_size, uid, attr)` 并采用 candidate + commit：先把 `UniqueId`/临时 bytes 规范化并深拷贝为 `candidate_owner`，复制旧 `attr_` 为 `candidate_attr`，令 `candidate_attr.comm_args` 指向 candidate；调用原生接口失败时丢弃 candidate，`attr_`、`uid_owner_` 及旧 `comm_args` 均保持不变；成功时再交换/提交 owner 和 attr，并将提交后的 `attr_.comm_args` 重定向到最终 `uid_owner_` 的稳定地址。重复设置使用同一流程，旧 owner 仅在成功提交后释放。底层原样返回 C++ `int`，高层便利接口复用该入口并负责把非零返回码转为异常。
- 新增 `ProfData`/`ProfRecord` 快照类型和唯一的 `capture_prof_snapshot(verbose) -> ProfData | None`。等待专用 profiling mutex 时先释放 GIL；取得 mutex 后在同一锁区调用 `aclshmemx_get_prof`、判空并把 `profs->pe_id` 及全部非零 `(block_id, frame_id, count, cycles)` 记录复制到本地 C++ 值对象，释放 mutex 后重新持有 GIL，再构造 Python 快照。`aclshmemx_show_prof` 共用该 mutex，避免覆盖共享 `g_host_profs`。公共兼容入口保留 `(out_profs, verbose)` 两个位置参数并返回 `records`，私有入口返回完整快照；两者共享上述实现，未选择当前 PE 时返回 `None`。
- `aclshmemx_malloc/calloc/align` 不对任何空指针作异常转换：无论来自零长度、heap 耗尽、未初始化、`HOST_SIDE` 不支持、非法 alignment 或内部失败，C++ 的 `nullptr` 均按既有绑定约定暴露为 `0` 或 `None`。高层 `Buffer` 分配接口先拒绝零长度；对合法的非零请求调用底层后，必须在构造 `Buffer` 前检查返回地址，若为 `0`/`None` 则抛出包含 API 名称、请求大小和 `mem_type` 的 `AclshmemError`，不得创建 `Buffer(addr=0, ...)`。

#### 高层 Python 封装

文件：`src/python/shmem/core/`

- `init_final.py`：新增 `set_attr_uniqueid_args`，`init` 增加 `instance_id` 参数，`finalize` 可按实例 id 调用 `aclshmemx_finalize`。`init`、`ctx_get`、`ctx_set`、`finalize` 和 `use_instance` 共用进程级可重入锁 `threading.RLock`，使嵌套的实例切换/恢复不会自锁，并防止 getter 取值期间与切换或销毁交错。
- `memory.py`：`buffer` 改用 `aclshmemx_malloc`，新增 `malloc/calloc/align/free` 的 mem_type 参数。扩展 `Buffer`，增加只读 `mem_type`、`instance_id`、`owned`、`release_called`：`malloc/calloc/align/buffer` 返回 `owned=True`，`get_peer_buffer` 及手工地址 view 返回 `owned=False`。`free` 同时持有 GIL 和实例 `RLock`，检查 owned、未释放、当前/显式实例与 `instance_id` 一致、实参 `mem_type` 与记录值一致，随后先置 `release_called=True`；调用原生 void free 时仅释放 GIL并继续持有实例锁，native 返回后才释放实例锁。即使原生调用或后续同步异常也不回滚，确保并发 double-free 只有一次进入 native，且释放期间实例不会被切换或销毁。
- `sync.py`：新增 barrier/sync、stream barrier、多实例 ctx、engine config、handle_wait 和 profiling 封装。
- `__init__.py`：统一导出新增 high-level API，保持 `import shmem.core as core` 的使用方式。

### 参数与异常约定

| 参数/返回 | 约定 |
|-|-|
| 指针 | Python 侧使用 `int`，底层按 `intptr_t` 转为 C++ 指针。 |
| Stream | Python 侧传 ACL runtime stream 句柄整数；底层统一保留 `0` 表示 `nullptr` 的原生语义，barrier 等允许默认流的高层接口也可将 `stream=None` 转为 `0`。这是底层兼容规则，不适用于高层 `handle_wait`：后者在入队前同时拒绝 `None` 和数值 `0`，必须传入待等待 RDMA 路径所用的同一显式非零 stream。 |
| mem_type | 使用 `core.MemType.DEVICE_SIDE` 或 `core.MemType.HOST_SIDE`；默认 `DEVICE_SIDE`。 |
| Context | 底层 `aclshmemx_instance_ctx_get()` 返回不可变的 `InstanceCtx(id)` 值快照，高层 `ctx_get()` 返回该快照的 `id`；均不暴露 runtime 借用指针。 |
| Handle | `core.Handle(team_id=0)` 是调用方主动构造的 team-scoped 等待条件；原生字段仅为 `team_id`，对象不由某次 RMA 产生，也不拥有 stream、team 或 runtime。 |
| Profiling | `get_prof()` 通过私有低层快照入口返回 `ProfData(pe_id, records)`；`pe_id` 直接复制自同一次调用得到的原生 `profs->pe_id`，每条 `ProfRecord` 包含 block/frame id、count 和 cycles。未通过 `SHMEM_CYCLE_PROF_PE` 选择当前 PE 时返回 `None`。 |
| 返回码 | 高层封装将非 0 返回码转换为 `AclshmemError`；底层保留 C++ 返回码或 void 风格。 |

### 集合语义与限制

- `barrier`、`barrier_all`、`sync`、`sync_all` 必须由参与 team 的所有 PE 按相同顺序调用。
- `aclshmemx_malloc`、`aclshmemx_calloc`、`aclshmemx_align` 和 `aclshmemx_free` 均是全 PE 集合调用，内部包含全局 control barrier。所有 PE 必须按相同顺序调用相同种类的接口；每次分配的 `size`、`count`、`alignment` 和 `mem_type` 必须一致或满足原生实现要求的兼容关系；每次释放必须对应各 PE 上同一轮对称分配得到的本地指针，并使用与分配时一致的 `mem_type`。不得让某些 PE 跳过、提前返回或改变调用次序，否则其余 PE 可能永久阻塞，且对称 heap 关系会被破坏。
- 高层零长度校验是进入原生集合调用前的本地校验，不提供跨 PE 协调能力。调用方必须在调用前通过同一配置来源、broadcast/all-gather 或等价机制确保所有 PE 的参数和控制流一致；尤其禁止某个 PE 因零长度在高层抛异常、其他 PE 却以非零长度进入底层分配。高层文档不得把本地异常描述为能够安全处理跨 PE 参数分歧。
- Host `barrier` 保证调用该 barrier 前由 CPU 发起的 store、AMO 和 RMA 更新完成；它不替 NPU 完成由 NPU 发起的通信。若需从 Host 观察 NPU 操作，应同步对应 stream/device 或采用 stream 接口。
- `sync` 只保证此前普通 memory store 的完成和可见性，不保证通过 ACLSHMEM API 发起的远端 RMA/AMO 更新完成，不能用来替代 `barrier`/`quiet`。
- 仅 HCCS 的 scale-up 系统上 barrier 后更新全局可见；同时存在 HCCS/RDMA 的系统只保证写入某 PE 内存的更新对该 PE 可见。
- RDMA/UDMA config 的 UB scratch 至少满足 Host API 要求的 128 字节 staging block；MTE/SDMA 使用各自 Host API 的平台约束，不统一套用 128 字节限制。
- `handle_wait` 与 C++ 示例一致：调用方在 RDMA/stream RMA 之后按 team 主动构造 `Handle(team_id)`，它不是某次操作返回的 future。调用方必须保证 handle 的 team、当前实例及传入的显式非零 ACL stream 与待完成的 RDMA 路径匹配；高层校验可获得的 team/实例元数据并拒绝 `None`/`0`，无法从仅含 `team_id` 的原生 handle 推导 stream，因此 stream 匹配仍是必须明确遵守的调用契约。`putmem_on_stream(nullptr)` 与 wait 可能解析到不同运行时流，破坏入队顺序和跨机数据可见性。低层入口为保持 C++ 参数语义仍允许 `0`。默认 MTE smoke 只覆盖入口和主动构造的 `Handle`，完整可见性由 RDMA 专项脚本验证。

#### Signal 跨机支持矩阵

Python API 文档和 QuickStart 必须沿用 C++ 按具体 API 路径定义的 Signal 传输限制，不得仅根据操作符为 `SET` 就推断支持跨机 RDMA：

| Python / C++ API 路径 | 操作 | 跨机 RDMA | Python 侧约束 |
|-|-|-|-|
| `core.put_signal` / `aclshmemx_putmem_signal`、`aclshmemx_putmem_signal_nbi` 及 typed put-signal 变体 | 携带 signal 的 put（包括 `SET`） | 不支持，仅 MTE | 即使 signal 操作为 `SET`，该 API 路径也不得用于 RDMA；单机/HCCS 示例与文档必须标注 MTE-only。 |
| `core.signal_op` / `aclshmemx_signal_op_on_stream(..., SET, ...)` | `SET` | 条件支持 | 这是 Signal 跨机 RDMA 的唯一受支持路径；仅在目标 PE、signal 地址注册、transport 等 RDMA 条件满足时使用，并遵守传入 stream 的入队与同步规则。 |
| `core.signal_op` / `aclshmemx_signal_op_on_stream(..., ADD, ...)` | `ADD` | 不支持 | 不得用于 RDMA 跨机场景；跨机累加需改用受支持的同步/通信方案。 |
| Signal wait API | wait | 不发起跨机等待 | wait 只轮询本卡地址；跨机通知时，远端 PE 通过受支持的 on-stream `SET` 更新该 PE 的本地 signal 地址，再由本 PE 执行 wait。 |

## 可维可测分析

### 功能测试

| 测试文件 | 覆盖点 |
|-|-|
| `examples/python_extension/test/core/test_memory.py` | `MemType` heap 的 `malloc/calloc/align/free`；底层零长度和分配失败仍返回 `0`/`None`，高层拒绝零长度或对非零空返回抛 `AclshmemError`。逐项断言分配对象记录只读 `mem_type/instance_id/owned=True/release_called=False`，peer buffer 与手工 view 为 `owned=False`；错误 `mem_type`、错误实例、非 owned、重复释放均在 native 前失败。用两个 Python 线程同时 free 同一 Buffer，断言仅一个线程在持有 GIL 时把 `release_called` 从 false 置 true、仅一次进入 native，且 native void 调用异常后状态不回滚；`HOST_SIDE` 按 CANN/runtime 能力软验证。 |
| `examples/python_extension/test/core/test_sync.py` | 底层五参数 `set_attr_uniqueid_args` 与 `get_prof(None, False)` 兼容入口。UID 用例覆盖临时 bytes/`UniqueId`、删除原始对象并强制 GC、同一 `InitAttr` 重复设置、native 失败时 attr/owner/`comm_args` 保持旧值、成功时三者一次性提交，不检查或依赖 `_uid_owner` Python 属性。Context 用例断言返回不可变 id 快照，并以多线程交错 getter 与 `ctx_set/finalize` 验证无借用指针逃逸；profiling 用例并发执行 `get_prof` 与 `show_prof`，断言每个快照的 `pe_id`/records 来自同一锁区且互不覆盖。另覆盖 engine config、barrier/sync、stream barrier、主动构造 Handle，以及高层 wait 缺失/零 stream 的拒绝和低层零流兼容。 |
| `examples/python_extension/test/core/test_multi_instance.py` | 创建两个 UID 多实例，验证 `init/ctx_get/ctx_set/finalize/use_instance` 共用可重入锁，嵌套 `use_instance` 可恢复且与并发 finalize 串行；验证显式 `mem_type` heap 分配释放、world team 查询和按实例 barrier。所有 PE 按相同 malloc/calloc/align/free 顺序和兼容参数执行；测试辅助层先 all-gather 操作描述，对 rank 间参数或顺序不一致的负向输入统一报错，并断言没有 PE 进入原生集合调用。 |
| `examples/python_extension/test/core/test_handle_wait_rdma.py` | ROCE 初始化后由调用方按 RDMA team 主动构造 `Handle(team_id)`，在匹配的实例和同一显式 ACL stream 上依次提交 Host RMA 与 `core.handle_wait`，验证同步后对端数据可见；断言 RMA API 不返回 Handle，Handle 不拥有 stream/team/runtime 资源。另覆盖 team、实例和 stream 不匹配的校验或诊断路径；C++ device-kernel handle_wait 对照通过现有 `use_handlewait` 样例验收。 |
| `examples/python_extension/test/core/test_signal.py`、`test_signal_rdma.py` | 单机/HCCS 路径覆盖 `core.put_signal`、`core.signal_op` 的 SET/ADD 和本卡 signal wait，并验证 put-signal 系列标记为 MTE-only；RDMA 环境只调用 `core.signal_op(..., SET, ..., stream)`（底层为 `aclshmemx_signal_op_on_stream`）验证跨机 SET 与目标 PE 本地 wait 数据可见性，禁止用 `core.put_signal(..., SET)` 代替；put-signal 系列和 signal-op ADD 的跨机输入作为能力限制测试，在进入不支持路径前明确拒绝或标记为 unsupported，且不得误报成功。 |
| `examples/python_extension/run.sh` | 默认接入原有 Python 用例和新增 MTE/CC/多实例 smoke 用例。 |
| `docs/202607/self_test_case/shmem_python/tests/run.sh` | 比赛仓官方归档验收套件，覆盖 20 个底层绑定、core wrapper、barrier 无死锁及 putmem_on_stream 性能入口；`functional/test_engine_config.py` 按引擎传入合法 UB 大小（MTE/SDMA 为 64 字节，RDMA/UDMA 为 128 字节）；实现完成后不得以 `[SKIP]` 代替 PASS。 |

具备 Ascend/CANN 环境时，推荐使用 `examples/python_extension/run_acceptance.sh` 统一收集静态检查、wheel 构建、默认用例、可选 Python RDMA smoke 和 C++ `use_handlewait` 对照日志。统一入口将 `PYTHON_BIN` 和 `NPROC` 显式传递到 wheel 安装及所有分布式子进程，避免构建、安装、测试使用不同 Python 环境，或多卡配置静默退回默认值。

验收入口把本轮构建出的绝对 wheel 路径显式传给 `run.sh`。安装完成后通过
`importlib.metadata.distribution("cann-shmem")` 解析发行目录，并确认实际导入的
`shmem.__file__` 位于该目录下，避免历史 wheel 或源码目录遮蔽本轮产物。

QuickStart 的对称内存示例必须让分配参数来自所有 PE 共享或广播的配置，并用 `try/finally` 保证所有 PE 以相同顺序释放各自对应的对称指针。示例旁明确标注 `malloc/calloc/align/free` 是全 PE 集合操作；单个 PE 的条件分支、异常提前返回或不同 `mem_type` 均不安全。若示例接受每个 rank 的外部输入，应先 all-gather 并在所有 rank 上一致地通过或拒绝，再调用原生分配，而不能依赖高层零长度校验协调各 PE。

QuickStart 的 Signal 示例按上述具体 API 路径区分单机/HCCS 与跨机 RDMA：`core.put_signal` 及其 nbi/typed 对应接口只出现在 MTE 示例中，即使使用 `SET` 也不得写成跨机 RDMA 示例；跨机示例必须显式调用 `core.signal_op(..., SET, ..., stream)`，对应底层 `aclshmemx_signal_op_on_stream`，再由目标 PE 在本卡地址上 wait。示例同时标注 signal-op `ADD` 不支持 RDMA、wait 本身不访问远端地址。Python API reference 在 `put_signal`、`signal_op` 的 SET/ADD、wait 及其变体旁分别链接该矩阵，避免用户从通用 Signal 示例推断错误能力。

### 性能测试

性能用例位于 `examples/python_extension/perf/`，使用同一进程、同一 SHMEM context、同一对称内存和同一 ACL stream 比较以下路径：

1. Python 路径：性能计时循环调用 `_pyshmem.aclshmemx_putmem_on_stream`；正确性阶段还调用 `_pyshmem.aclshmemx_getmem_on_stream`。
2. C++ 基线：通过 `ctypes` 调用 `libputmem_on_stream_baseline.so`。helper 不直接链接、`dlopen` 或静态包含第二份 SHMEM runtime；它在 Python 扩展已加载 SHMEM 后使用 `dlsym(RTLD_DEFAULT, ...)` 解析 `aclshmemx_putmem_on_stream` 和 `aclshmemx_getmem_on_stream`，从而与 Python 路径调用同一份进程内 runtime。运行前用 `dladdr` 和 `/proc/self/maps` 的规范路径/inode 校验目标符号只归属一份 `libshmem.so`，并确认 Python 扩展与 helper 解析地址的模块一致；无法证明唯一性时终止测试，不输出性能通过结论。结果元数据记录 helper 文件和实际加载 `libshmem.so` 的规范路径及 SHA-256；性能门槛仍按任务书以 put 路径为代表。
3. 两条路径均执行最多 100 次预热；每个 shape 执行 9 个测量批次。迭代数按 shape 调整，使每批至少调用 20 次且目标传输量约为 64 MiB。每批计时从第一次 Host API 调用前开始，到同步同一 stream 返回后结束，覆盖完整 RMA 路径并避免只测入队开销。各批次交替 Python/C++ 测量顺序，避免把调度或频率瞬变持续归入同一实现。
4. shape 固定为 4 KiB、64 KiB、1 MiB 和 16 MiB；目标 PE 为 `(rank + 1) % nranks`。各 rank 的结果取最大 overhead，避免异常 PE 被平均值隐藏。
5. 每个 rank/shape 在计时前分别执行 Python 与 C++ 的 put/get 路径，使用 rank 特征字节填充整个 payload，并逐字节校验目标 PE 的完整 put 结果和本地 get 结果 bit-exact；正确性失败时不产生性能通过结论。
6. 分别使用 `torchrun --nproc-per-node 2/4/8` 执行。输出 CSV 字段为 `world_size,shape_bytes,rank,iterations,python_us,cpp_us,overhead_percent,pass`；其中 `rank` 标识该 shape 下开销最大的 PE。
7. 每批成对开销按 `(python_sample / cpp_sample - 1) * 100%` 计算，使用 9 个成对开销的中位数作为门槛判定值；CSV 中的 `python_us` 和 `cpp_us` 分别记录两组样本的中位数。所有 2/4/8 卡、所有 shape 均须不超过 5%。若环境不足 8 卡，缺少的规模必须明确标为未验证，不得以较小规模代替。

为降低噪声，测试前固定设备频率策略、关闭其他 NPU 负载，并记录 SoC、CANN、Python、torch/torch_npu、SHMEM commit、可见设备、唯一 runtime 校验结果，以及 helper 和实际加载 `libshmem.so` 的 SHA-256。原始 CSV、完整日志及汇总截图统一写入 `acceptance_logs/<timestamp>/performance/`。

### 正确性与性能标准

| 验收标准 | 描述 | 标准来源 |
|-|-|-|
| 正确性 | RMA put/get 结果与 C++ 参考路径 bit-exact；signal/handle_wait 后对端数据可见性符合 C++ 语义；多实例切换后 heap/team 状态互不污染。 | 任务书 |
| 性能 | Python Host RMA 以 `putmem_on_stream` 为代表，相对同 shape、同 stream 的 C++ 路径 overhead 不超过 5%，覆盖 2/4/8 卡场景。 | 任务书 |
| 稳定性 | barrier/barrier_on_stream 功能正确，无异常超时或死锁。 | 任务书 |

### 兼容性分析

- `_pyshmem.aclshmemx_set_attr_uniqueid_args` 保留官方测试使用的五参数输出对象形式并返回原生 `int`；`InitAttr` C++ wrapper 私有持有 UID 深拷贝，以 candidate + commit 保证重复设置和失败回滚，不要求 `py::dynamic_attr`；返回 `InitAttr` 的便利形式仅由 `shmem.core` 提供。
- `_pyshmem.aclshmemx_instance_ctx_get` 保留 `InstanceCtx` 类型名，但返回只含 `id` 的不可变值快照，不再暴露全局 context 引用；高层 `ctx_get()` 的整数返回保持不变。
- `_pyshmem.aclshmemx_get_prof` 保留 `(out_profs, verbose)` 两参数入口，包括 `get_prof(None, False)`；兼容入口、私有完整快照入口和 `show_prof` 共用 profiling mutex，后者保留同一锁区复制的原生 `pe_id`。
- `_pyshmem.aclshmemx_malloc/calloc/align` 对所有失败场景保留 C++ 空指针返回语义；高层 `Buffer` API 单独拒绝零长度，并将非零请求得到的空指针转换为 `AclshmemError`。
- `core.handle_wait` 不再提供默认 stream，并显式拒绝 `None` 和数值 `0`。`Handle` 由调用方按 team 主动构造，不描述为 RMA 返回对象；调用方必须确保其 team、当前实例及显式 stream 与待等待 RDMA 路径匹配。低层入口仍接受 `0`，保持原生参数兼容性。
- `core.buffer(size)` 保留原调用方式，默认仍分配 device-side 对称 heap。
- `core.finalize()` 不传参时保留原语义；传入 `instance_id` 才调用多实例 finalize。
- `aclshmem_init_using_unique_id` 新增 `instance_id=0` 默认参数，既有 4 参数调用保持兼容；`uid` 支持 `UniqueId` 对象和既有 bytes 表示；`core.get_unique_id()` 保持既有 bytes 返回。
- 顶层 `shmem` 继续导出已有低层 API，同时新增本任务所需 API。

## 支持硬件

目标硬件与任务书一致：Atlas A2/A3 训练系列或推理系列产品，CANN 9.0.0 或社区较新版本。RDMA/SDMA/UDMA smoke 需依赖对应硬件和通信环境能力。
