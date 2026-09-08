# 【社区任务】SHMEM Python 接口开发设计说明书

## 一、需求背景

### 1.1 需求来源

CANN 2026 年 7 月社区任务，任务书见 `04_tasks/01_community-task-2026/docs/202607/SHMEM_Python_api_task_doc.md`。

### 1.2 基线信息

| 项 | 值 |
| --- | --- |
| 目标仓库 | https://gitcode.com/cann/shmem |
| 基线分支 | master |
| 基线 commit | `1c89eec` |
| 硬件 | Atlas A2 / A3 |
| CANN | ≥ 9.0.0 |
| Python | ≥ 3.9 |

### 1.3 现状分析

`src/host/python_wrapper/pyshmem.cpp` 已绑定 41 个 Host API，覆盖初始化、对称堆、RMA、team 与 P2P 同步。以下五类接口尚无 Python 入口：

| 分类 | 接口 | 数量 |
| --- | --- | --- |
| 初始化与多实例 | `aclshmemx_set_attr_uniqueid_args`、`aclshmemx_instance_ctx_get`、`aclshmemx_instance_ctx_set` | 3 |
| 对称堆（带 `mem_type`） | `aclshmemx_malloc`、`aclshmemx_calloc`、`aclshmemx_align`、`aclshmemx_free` | 4 |
| 引擎 UB 配置 | `aclshmemx_set_mte_config`、`set_sdma_config`、`set_rdma_config`、`set_udma_config` | 4 |
| 集合通信与同步 | `aclshmem_barrier`、`barrier_all`、`sync`、`sync_all`、`aclshmemx_barrier_on_stream`、`barrier_all_on_stream`、`aclshmemx_handle_wait` | 7 |
| Profiling | `aclshmemx_get_prof`、`aclshmemx_show_prof` | 2 |

Python 侧 `src/python/shmem/core/` 现有 `init_final.py`、`memory.py`、`rma.py`、`direct.py`、`utils.py` 五个模块，本次新增的封装需与其惯例一致。

### 1.4 任务目标

将上述 20 个接口绑定至 `shmem._pyshmem`，并在 `shmem.core` 之上提供高层封装，同步补齐 `docs/api/pythonAPI.md` 与测试。

### 1.5 验收标准

| 项 | 标准 |
| --- | --- |
| 功能 | 新增接口均有 Python 用例；`examples/python_extension/run.sh` 既有用例全部通过 |
| 性能 | Python Host RMA 相对 C++ 的开销 ≤ 5%（2/4/8 卡） |
| 正确性 | RMA put/get 结果与 C++ 参考一致 |
| 文档 | 已知限制须明确记录，不得隐瞒 |

## 二、需求分析

### 2.1 总体分层

```
用户代码
   │
   ├── shmem.core          高层封装：Pythonic 接口、Buffer 契约、异常
   │      ├── cc.py               barrier / sync / handle_wait
   │      └── multi_instance.py   实例上下文、typed 堆、引擎配置、profiling
   │
   └── shmem._pyshmem      pybind11 绑定：与 C 接口一一对应
          └── pyshmem_ext.cpp
                 │
                 └── libshmem.so   ACLSHMEM Host API
```

高层封装不引入新语义，只做三件事：合并同族接口的调用形式、把返回码转成异常、把裸地址包成 `Buffer`。

### 2.2 需求拆解与目标 API 映射

#### 2.2.1 初始化与多实例

`aclshmemx_set_attr_uniqueid_args` 填充 `InitAttr`；`instance_ctx_get/set` 切换当前激活实例。

前者的常规路径已由 `aclshmem_init_using_unique_id` 在 C++ 内部覆盖（其实现即调用它），故仅提供裸绑定，不做高层封装。后两者封装为 `instance_ctx_get() -> int` 与 `instance_ctx_set(id) -> None`。

#### 2.2.2 对称堆内存

四个 `aclshmemx_*` 接口相对已绑定的 `aclshmem_*` 版本，增量仅是 `mem_type` 参数。高层封装命名为 `buffer_typed` / `calloc_typed` / `align_typed` / `free_typed`，与 `memory.py` 的 `buffer` / `free` 区分，并额外提供上下文管理器 `symmetric_buffer`。

#### 2.2.3 引擎配置

四个接口签名一致，均为 `(uint64_t offset, uint32_t ub_size, uint32_t sync_id)`。注意头文件中同名的 device 侧接口返回 `void`，host 侧返回 `int`，绑定的是后者。

#### 2.2.4 集合通信与同步

`barrier` 与 `sync` 各有 team / 全局两个变体，`barrier` 另有两个 on-stream 变体。高层封装合并为 `barrier(team=None, stream=None)` 与 `sync(team=None)` 两个函数，由参数组合决定实际调用哪一个。

`handle_wait` 的 `aclshmem_handle_t` 只含一个 `team_id` 字段，不是不透明句柄，绑定时按值构造。

#### 2.2.5 Profiling

`get_prof` 的原生输出指向库内部缓冲区，绑定层深拷贝后交出 Python 自有的只读快照；`show_prof` 在头文件中已标注 deprecated，绑定保留但文档标注。

### 2.3 兼容性要求

不改动任何已有绑定的签名与行为。`pyshmem.cpp` 仅新增两行：include 头文件、在模块末尾调用注册函数。

## 三、详细设计

### 3.1 pybind11 公共约定

#### 3.1.1 指针与整数

所有 C 指针以 `intptr_t` 在 Python 侧表示，与 `pyshmem.cpp` 现有约定一致。

#### 3.1.2 ACL stream

沿用 `pyshmem.cpp` 中 `putmem_on_stream` 等六处的判空模式：

```cpp
aclrtStream acl_stream = nullptr;
if (stream != 0) {
    acl_stream = reinterpret_cast<aclrtStream>(stream);
}
```

传 0 表示使用默认流。

#### 3.1.3 GIL

多数绑定使用 `py::call_guard<py::gil_scoped_release>()`。pybind11 在参数转换完成之后才构造 call guard,因此按引用接收已注册类型(`aclshmemx_uniqueid_t&`、`aclshmemx_init_attr_t&`)是安全的。

两处**例外**,原因是它们涉及库内可回收存储或 Python 对象操作:

| 接口 | 例外做法 | 原因 |
| --- | --- | --- |
| `aclshmemx_instance_ctx_get` | 全程持 GIL | 返回的是指向库内上下文的指针,释放 GIL 会在"取指针"与"读 `ctx->id`"之间留下可被并发 `instance_ctx_set` 回收的窗口。该调用只是一次字段读取,持 GIL 不损失并发 |
| `aclshmemx_set_attr_uniqueid_args` | 全程持 GIL | 需构造并挂载 Python 侧的 UID 副本(见 3.2);原生调用只做字段赋值、不阻塞 |

`aclshmemx_get_prof` 仍释放 GIL,但另加 C++ 互斥量(见 3.8)。锁序固定为**先释放 GIL、再取互斥量**,不存在持 GIL 等互斥量的路径,故不会与之形成死锁。

#### 3.1.4 错误语义

分两层：

- 绑定层：分配类接口拿到空指针时抛 `std::runtime_error`，与 `pyshmem.cpp` 中 `aclshmem_malloc` 一致；返回码原样透传。
- 封装层：非零返回码转 `AclshmemError`，参数缺失转 `AclshmemInvalid`，与 `init_final.py`、`rma.py` 一致。

绑定层不吞错误码，也不把空指针当作合法返回值交给调用方。

### 3.2 UID 初始化与 InitAttr

`aclshmemx_set_attr_uniqueid_args` 的实现(`src/host/init/shmem_init.cpp:218`)是 `attr->comm_args = (void*)uid`——**保存的是 UID 的地址,不是内容**。因此 UID 的生命周期必须覆盖到后续的 `aclshmemx_init_attr`,只在本次函数调用期间有效是不够的:Python 侧的 `UniqueId` 一旦被 GC,`comm_args` 即成悬空指针。

绑定层因此建立显式的 owner 关系。**owner 必须放在 Python 触碰不到的地方**:早期版本给 `InitAttr` 加 `py::dynamic_attr()` 并把副本挂在 `attributes._uniqueid_owner` 上,但凡是 Python 可达的属性,调用方就能 `del attr._uniqueid_owner` 或将其重新赋值,一旦最后一个引用被丢弃,`comm_args` 立刻悬空——生命周期保证形同虚设。现改为:

- 副本存放在扩展模块内的 `g_uid_owners`(`unordered_map<const aclshmemx_init_attr_t*, unique_ptr<aclshmemx_uniqueid_t>>`),以 `InitAttr` 实例地址为键,由 `g_uid_owner_mutex` 保护。`InitAttr` 不再需要 `dynamic_attr()`,Python 侧不存在任何指向该副本的名字。
- 绑定内部将传入的 UID **深拷贝**进该表,`comm_args` 指向表内副本。调用方随后可立即删除自己的 UID。
- **成功时才安装新 owner**;失败时保留旧的,因为此时 `comm_args` 仍指向上一次成功安装的副本,替换它反而制造悬空。同一个 `InitAttr` 重复调用是替换而非累积,赋值时旧副本随 `unique_ptr` 析构。
- 表按地址索引,条目会在该地址被新的 `InitAttr` 复用时才回收,故其规模以存活的 `InitAttr` 数量为界、而非调用次数。用这一有界滞留换掉悬空指针是有意为之:一个 `UniqueId` 仅 128 字节,而 `InitAttr` 的正常用法是每次初始化创建一个,不在循环里产生。

该绑定持 GIL 全程执行:它操作 Python 对象,而原生调用只做结构体字段赋值、不阻塞。

用例分两层验证:`test_uid_outlives_caller` 断言 `_uniqueid_owner` 这一属性**不存在**,
且 `delattr` / `setattr` 两种改写尝试都被 `AttributeError` 拒绝——即调用方无从触及副本;
`test_init_after_uid_collected` 则在 `del uid` + `gc.collect()` 之后真正调用
`aclshmem_init` 并要求成功。前者证明攻击面已封闭,后者才能证明 `comm_args`
所指的仍是有效存储,因为该指针在原生 bootstrap 内部被消费,悬空不会表现为
Python 层的异常。

失败时 `attributes` 内容未定义,docstring 明确写出,避免调用方将半初始化的对象传给 `aclshmem_init`。

### 3.3 多实例上下文

C 接口 `aclshmemx_instance_ctx_get` 返回 `aclshmem_instance_ctx *`。该指针指向库内部持有的上下文，可能被回收，直接透出裸地址对调用方既不安全也没有用途。

分两层处理，以保持低层与 C 接口的语义对应：

- 低层 `_pyshmem.aclshmemx_instance_ctx_get()` 返回只读的 `InstanceContext` 值快照，字段在原生调用返回后立即拷出。目前仅携带 `id`——结构体中的 `instance` 字段在头文件中标注为保留。保留结构体形态是为了后续字段扩展时不必改签名。
- `shmem.core.instance_ctx_get()` 在其上取 `id` 返回 Python `int`，与 `instance_ctx_set` 形成闭环。

并发保护分两级:

- **绑定层**:该绑定是 3.1.3 所述的 GIL 例外,从调用 native、检查空指针到复制 `ctx->id` 全程持 GIL,消除悬空读取窗口。
- **封装层**:`shmem.core` 的 `instance_ctx_get` / `instance_ctx_set` / `init` / `finalize` 共用 `utils._instance_lock`(一把 `threading.RLock`)。这四个函数都会读或改"当前激活实例",串行化后 `get` 不会撞上正在拆除上下文的 `set`。用 RLock 是因为 `init`/`finalize` 会再进入取同一把锁的辅助函数。

**限制**:上述两级都只覆盖经 Python 进入的调用方。若有原生 C++ 线程直接调用 `aclshmemx_instance_ctx_set` 并发切换实例,本设计不作承诺,该约束写入限制表。

### 3.4 对称堆与 Buffer

#### 3.4.1 低层接口

四个绑定的 `mem_type` 默认值取 `DEVICE_SIDE`，与 `include/host/mem/shmem_host_heap.h` 中的 C++ 默认参数一致。

#### 3.4.2 Buffer 契约

`memory.py` 的 `buffer()` 返回 `Buffer(addr, length)`，而 `rma.py` 的 `put` / `get` / `put_signal` 均依赖 `.length` 推导传输字节数。若新增的分配接口返回裸 `int`，其结果无法直接传给同一个包内的 RMA 接口，调用方必须手工包一层。

因此 `buffer_typed` / `calloc_typed` / `align_typed` 统一返回 `Buffer`，`calloc_typed` 的长度取 `count * size`。

`Buffer` 相应扩展两个字段。原因是 `aclshmemx_free` 返回 `void`——重复释放、按错误的堆释放、以及释放一个并非本进程分配的对端地址，这三种误用底层都无从检测、更无从报告，一旦发出即为未定义行为:

| 字段 | 含义 |
| --- | --- |
| `mem_type` | 分配时所用的堆。`None` 表示该 `Buffer` 并不拥有一块分配——`get_peer_buffer` 返回的对端地址、以及调用方从裸指针自行构造的 `Buffer` 属此类。`memory.py` 的 `buffer()` 走 `aclshmem_malloc`(device 侧堆),故其返回值带 `DEVICE_SIDE`,可交给 `free_typed` 释放 |
| `released` | 是否已发起过释放。**在调用 native 之前**置位,且永不清除 |

`free_typed` 据此在持 GIL 期间做三项检查,任一不满足即抛 `AclshmemInvalid` 而不进入原生调用:已 `released`、`mem_type` 为 `None`(非本模块分配)、`mem_type` 与传入值不符。先置位再调用的顺序是有意的:原生释放不可撤销,故并发的第二次释放应当在 Python 这一层输掉竞争,而不是在分配器里。

`free()` 做对称的处理,否则防护会留下两个缺口:一是绕过 `free_typed` 重复释放,二是把 host 侧分配送进只认 device 侧的 `aclshmem_free`。故 `free()` 同样先检查 `released`、再检查 `mem_type` 是否为 `DEVICE_SIDE`(`None` 视为兼容,保留裸 `Buffer` 的既有用法),然后置位、调用。

两条路径因此互相识别:`buffer()` 走 `aclshmem_malloc`(device 侧堆),其返回的 `Buffer` 带 `DEVICE_SIDE`,既可 `free()` 也可 `free_typed()`;`get_peer_buffer` 的返回值 `mem_type` 为 `None`,经 `free_typed` 被拒。构造函数的 `mem_type` 参数带默认值,既有的两参调用形式不受影响。防护覆盖经本模块分配器取得的 `Buffer`;调用方拿裸地址自建 `Buffer` 仍可绕过,这一边界在 README 限制表中写明。

#### 3.4.3 上下文管理器

`symmetric_buffer(size, mem_type)` 在 `with` 块退出时释放。需注意对称堆的分配与释放是集合操作，含隐式 barrier：若块内部分 PE 抛异常，这些 PE 会进入集合性的释放而其余 PE 不会，导致作业死锁。该约束在 docstring 中以 warning 标注。

### 3.5 引擎配置

四个封装函数共用一个私有辅助函数：

```python
def _set_engine_config(setter, engine_name, offset, ub_size, sync_id):
    ret = setter(offset, ub_size, sync_id)
    if ret != 0:
        logger.error("Configure %s unified buffer fails, ret: %d.", engine_name, ret)
        raise AclshmemError(f"Configure {engine_name} unified buffer fails.")
```

UDMA 的 `ub_size` 不得小于 `ACLSHMEM_UDMA_MTE_STAGING_UB_SIZE`（128），该约束写入 docstring。

### 3.6 Collective 与同步语义

`barrier(team=None, stream=None)` 的四条分支：

| team | stream | 实际调用 |
| --- | --- | --- |
| None | None | `aclshmem_barrier_all` |
| 有值 | None | `aclshmem_barrier` |
| None | 有值 | `aclshmemx_barrier_all_on_stream` |
| 有值 | 有值 | `aclshmemx_barrier_on_stream` |

on-stream 变体**仅将 barrier 入队到指定 NPU stream**,函数返回只表示入队完成。当该 stream 执行到这个 barrier 时,它保证同一 stream 上此前 NPU 发起操作的顺序与完成。Host 若需确认 barrier 已执行完毕,应再同步对应 stream(如 `aclrtSynchronizeStream`)。相对地,CPU 阻塞版 barrier 才是覆盖 CPU 发起路径的那一个。该语义写入 docstring 与限制表。

`sync` 与 `barrier` 的区别是仅保证本地存储可见，不保证远端更新完成，测试用例据此区别设计断言。

### 3.7 handle_wait

`aclshmem_handle_t` 定义为 `struct { aclshmem_team_t team_id; }`，是按值传递的结构体而非指针，绑定时构造后传入。Python 侧参数命名为 `team` 而非 `handle`，避免调用方误以为需要传某个由 API 返回的不透明对象。

**完成语义是片上异步,不是 Host 阻塞。** host 实现 `aclshmemx_handle_wait`(`src/host/sync/shmemi_sync.cpp:43`)转调 `aclshmemi_handle_wait_on_stream`,后者(`src/device/gm2gm/shmemi_device_cc_kernel.cpp:42`)只把 `k_aclshmem_handle_wait` kernel 入队到指定 stream 便返回,**不调用 `aclrtSynchronizeStream`**。故封装层维持该语义并在 docstring 与 API 文档中以 warning 写明两条约束:异步 RMA 与 `handle_wait` 须下发到**同一** stream(不同 stream 上的操作不受此约束);Host 读取结果前须自行同步该 stream,把函数返回当作"数据已就位"会产生竞态。

用例覆盖的是上述使用契约:同流下发 `putmem_on_stream` → `handle_wait` → Host 同步 → 跨 PE 读回校验。需如实说明其证明力的边界——**删掉 `handle_wait` 后该断言依然通过**,因为 `aclrtSynchronizeStream` 本身已排空该 put,从 Host 视角二者不可区分。要让 `handle_wait` 成为必需,须由 device kernel 在不同步 stream 的前提下依赖该顺序,这超出 Python 绑定的测试范围。故该用例定位为使用契约与 API 行为测试,不声称是必要性证明。

头文件注释称底层实现不支持跨节点句柄。实测在节点间 RDMA 路径上该接口正常工作（见 7.1），因此限制表中按「跨节点行为未在文档中承诺，实测可用但不宜依赖」表述。

### 3.8 Profiling

`aclshmemx_get_prof(&profs, verbose)` 的输出指针指向库内部的静态全局：内容会被下次调用覆盖，`finalize` 之后失效。把它作为返回值透给 Python 既不安全，也无法据此验证 profiling 内容是否正确。

绑定改为在原生调用返回后立即深拷贝，交出 Python 自有的只读快照：未采集数据的 rank 返回 `None`，否则返回 `(pe_id, blocks)`，其中 `blocks` 含 `ACLSHMEM_CYCLE_PROF_MAX_BLOCK` 项，每项是一对 `ACLSHMEM_CYCLE_PROF_FRAME_CNT` 长的 `(ccount, cycles)` 元组。快照与库的生命周期无关，`finalize` 后仍可读。

深拷贝解决了返回对象的长期生命周期问题,但不解决**并发**:`g_host_profs`(`shmem_init.cpp:78`)是共享静态全局,从 native 返回到拷贝完成之间存在被另一线程覆盖的窗口。故用一个 C++ `std::mutex` 覆盖「调用 native + 复制到本地 C++ 快照(`std::vector<aclshmem_prof_block_t>`)」的完整区段,退出临界区后才构造 Python 对象——构造过程不触碰共享状态,留在锁外以缩短临界区。

锁序见 3.1.3:先释放 GIL 再取该互斥量,单向且唯一,无死锁路径。

**限制**:该串行化只覆盖经本绑定进入的调用方,不覆盖直接调用 `aclshmemx_get_prof` 的原生 C++ 线程。此约束写入 docstring 与限制表,并由 `test_binding_lifetime.py` 中 8 线程 × 200 次并发调用的用例守住绑定侧不变量。

对应的用例校验 shape、字段类型、二次调用不改动已交出的快照、非采集 PE 的返回值，以及 `finalize` 之后快照仍然有效。

### 3.9 高层模块划分

按接口性质而非按 C 头文件划分：

- `cc.py`：集合通信与同步，含 `handle_wait`
- `multi_instance.py`：实例上下文、typed 堆、引擎配置、profiling

两者均通过 `core/__init__.py` 的 star-import 导出，`__all__` 与既有五个模块无符号冲突。

## 四、实现范围与提交策略

### 4.1 修改文件清单

| 文件 | 类型 | 内容 |
| --- | --- | --- |
| `src/host/python_wrapper/pyshmem_ext.h` | 新增 | `DefineShmemExt` 声明 |
| `src/host/python_wrapper/pyshmem_ext.cpp` | 新增 | 20 个 pybind11 绑定;`fast_putmem_on_stream` vectorcall 绑定与 `g_fast_methods` 注册;`_bench_*` 探针;`g_prof_mutex` |
| `src/host/python_wrapper/pyshmem.cpp` | 修改 | include 与注册新模块(2 行);`InitAttr` 的 owner 说明注释(UID 副本改由扩展模块持有,不再用 `py::dynamic_attr()`);暴露 `InitAttr.instance_id`(1 行,非 0 实例的创建入口) |
| `src/host/python_wrapper/CMakeLists.txt` | 修改 | 加入新编译单元 |
| `src/python/shmem/core/cc.py` | 新增 | `barrier` / `sync` / `handle_wait` |
| `src/python/shmem/core/multi_instance.py` | 新增 | 实例上下文、typed 堆、引擎配置、profiling |
| `src/python/shmem/core/rma.py` | 修改 | `put` 的下沉目标改为 `fast_putmem_on_stream`(1 行 + docstring) |
| `src/python/shmem/core/utils.py` | 修改 | 新增共享 `_instance_lock`;`Buffer` 扩展 `mem_type` / `released` |
| `src/python/shmem/core/memory.py` | 修改 | `buffer()` 记录 `DEVICE_SIDE`;`free()` 补重复释放与堆类型检查;`get_peer_buffer` 注明对端 `Buffer` 不持有分配 |
| `src/python/shmem/core/init_final.py` | 修改 | `init` / `finalize` 纳入 `_instance_lock` |
| `src/python/shmem/core/__init__.py` | 修改 | 导出新模块 |
| `examples/python_extension/test/core/test_cc.py` | 新增 | 集合同步用例 |
| `examples/python_extension/test/core/test_multi_instance.py` | 新增 | 多实例、typed 堆、引擎、profiling 用例 |
| `examples/python_extension/test/core/test_rma_fast.py` | 新增 | vectorcall 热路径与 pybind 版的跨 PE 一致性 |
| `examples/python_extension/test/core/test_binding_lifetime.py` | 新增 | UID owner 生命周期(含 GC 后仍可 init 的端到端用例)、get_prof 与 instance_ctx 并发约束 |
| `examples/python_extension/test/core/test_instance_create.py` | 新增 | 创建非 0 实例并在其上分配(5.2.6) |
| `examples/python_extension/bench/bench_rma_overhead.py` | 新增 | RMA 绝对时延基准 |
| `examples/python_extension/bench/bench_call_overhead.py` | 新增 | pybind 绑定层开销差分 |
| `examples/python_extension/bench/bench_fastcall_probe.py` | 新增 | pybind vs vectorcall 差分(5.6.5) |
| `examples/python_extension/bench/bench_gate_e2e.py` | 新增 | 门禁端到端:`core.put` 与 C++ 同口径(5.6.3) |
| `examples/python_extension/bench/run_gate.sh` | 新增 | 多轮采集 launcher |
| `examples/python_extension/bench/summarize_gate.py` | 新增 | 原始样本打印与中位判定 |
| `examples/python_extension/bench/cpp_baseline/main.cpp` | 新增 | C++ 基线;含门禁用的 `RmaOp::PutOnStream` 分支 |
| `examples/python_extension/bench/cpp_baseline/CMakeLists.txt` | 新增 | C++ 基线的构建定义 |
| `examples/python_extension/bench/cpp_baseline/run.sh` | 新增 | C++ 基线的多进程 launcher |
| `examples/python_extension/run.sh` | 修改 | 登记 5 个新用例 |
| `examples/CMakeLists.txt` | 修改 | 纳入 bench 子目录 |
| `docs/api/pythonAPI.md` | 修改 | 补齐新公共 API 条目;`handle_wait` 的片上完成语义 |
| `docs/quickstart.md` | 修改 | §6.2 最小可运行示例(端到端:初始化→分配→跨 PE RMA→同步→校验→收尾)与易踩约定表 |

#### 4.1.1 兼容性保证

热路径改用 vectorcall 属**内部下沉目标的替换**,对外签名、异常与流语义均不变:

| 面 | 保证 | 依据 |
| --- | --- | --- |
| `shmem.core.put` 签名 | `put(dst, src, remote_pe=-1, stream=None)` 未变 | 仅函数体最后一行的被调对象改变 |
| `stream` 语义 | 传 `None` 或 `0` 均用默认流,与原 `putmem_on_stream` 一致 | `0 if stream is None else stream`,与 pybind 版判空模式等价 |
| 异常 | 参数非法时抛 `TypeError`,与 pybind 版一致 | `PyLong_As*` 失败即 `PyErr_Occurred()` 返回,由 `test_rma_fast.py` 校验 |
| `_pyshmem` 既有符号 | `aclshmemx_putmem_on_stream` 的 pybind 绑定原样保留 | vectorcall 是**新增**符号 `fast_putmem_on_stream`,非替换 |
| 20 个必做 API | 全部仍为 pybind11 | 符合任务书"使用 pybind11 封装" |
| 测量探针 | `_bench_` 前缀,不进公共面 | 命名约定 |

`core.put` 与两条底层路径的等价性由 `test_rma_fast.py` 守住:同一组数据分别经 pybind 与 vectorcall 下发,跨 PE 读回逐字节比对。

### 4.2 提交策略

单 commit 提交。`pyshmem.cpp` 格式化变动（972 行）与功能改动（2 行）合并于同一 commit。

## 五、测试设计

### 5.1 测试层级

| 层级 | 内容 |
| --- | --- |
| 回归 | `run.sh` 下 7 个既有用例(`init_test` / `tls_test` / `unique_id_test` / `test_init_final` / `test_memory` / `test_rma` / `test_direct`) |
| 功能 | 新增 2 个:`test_cc.py`(集合同步)、`test_multi_instance.py`(实例上下文 / typed 堆 / 引擎配置 / profiling),覆盖本次 20 个接口 |
| 一致性 | 新增 1 个:`test_rma_fast.py`,vectorcall 与 pybind 两条路径逐字节比对 |
| 生命周期与并发 | 新增 1 个:`test_binding_lifetime.py`,单进程,覆盖 UID owner、`get_prof` / `instance_ctx_get` 的并发约束 |
| 多实例创建 | 新增 1 个:`test_instance_create.py`,单进程,创建非 0 实例并在其上分配 |
| 性能 | Host RMA 时延基准 + 口径一致的 C++ 基线 |

新增合计 5 个文件,均已登记进 `run.sh`;既有 7 个不变,故 `run.sh` 一轮共执行 12 个。

### 5.2 功能用例矩阵

`test/core/test_cc.py`：

| 步骤 | 覆盖 |
| --- | --- |
| 1 | `barrier()` 全局 |
| 2 | `barrier(team=...)` |
| 3 | `barrier(stream=...)` |
| 4 | `barrier(team=..., stream=...)` |
| 5 | `sync()` 与 `sync(team=...)` |
| 6 | `handle_wait(team=..., stream=...)` 正常路径 |
| 7 | `handle_wait` 缺 stream，须抛 `AclshmemInvalid` |

`test/core/test_multi_instance.py`：

| 步骤 | 覆盖 |
| --- | --- |
| 1 | `buffer_typed` 返回 Buffer 且长度正确 |
| 2 | `calloc_typed` 长度为 `count * size` 且内存已清零 |
| 3 | `align_typed` 地址按指定字节对齐 |
| 4 | 显式 `DEVICE_SIDE` 与默认行为一致 |
| 5 | `symmetric_buffer` 上下文管理器 |
| 6 | 四种引擎配置 |
| 7 | 实例上下文往返 |
| 8 | `get_prof` 地址跨调用稳定 |

### 5.3 多进程执行规则

沿用 `test/core/` 既有用例的写法：`torchrun --nproc-per-node N`，gloo backend，UID 经 `torch.zeros(512, uint8)` 广播，`core.init` / `core.finalize` 成对调用。不直接使用 `_pyshmem` 的低层初始化接口，以保证走的是与用户相同的路径。

### 5.4 正确性口径

集合同步不以「调用未抛异常」作为通过标准。各 PE 向自身 slot 写入可区分的值，同步后经对端地址读回校验。

关键在于断言必须能够失败。为此在用例中注入跨 PE 时序偏斜：rank 0 以外的 PE 在写入前延迟 0.2 秒。没有这一偏斜时，`quiet()` 加设备同步已足以让写入可见，即使删掉集合调用断言依然通过——即断言无效。

该设计经反向验证：将 `collective()` 替换为空操作后重跑，结果为

```
[rank0]: AssertionError: [FAIL] barrier_all: PE 1 store not visible after sync, expected 101, got 0
[rank1]: AssertionError: [FAIL] barrier(team): PE 0 store not visible after sync, expected 200, got 100
```

rank 0 读到 0（对端尚未写入），rank 1 读到上一轮的陈旧值，符合同步缺失的预期症状。

### 5.5 性能口径

Python 与 C++ 两侧使用完全相同的计时方法，否则数字不可比。具体参数、对照对象与计算公式见 5.6，实测数据见 5.7。

barrier 置于计时窗口外是因为它内部含 `aclrtSynchronizeStream` 与跨 PE 同步，PE 间时钟偏斜会污染单次时延。

C++ 基线由新增的 `bench/cpp_baseline/` 采集，输出的 JSON 由 `bench/summarize_gate.py` 消费。

### 5.6 门禁的测量方法

#### 5.6.1 对照对象必须是同一个操作

`shmem.core.put` 下沉到 `aclshmemx_putmem_on_stream`,因此 C++ 侧的对照必须也是这个入口。早期版本以阻塞的 `aclshmem_putmem` 作分母,那是**两个不同的操作**:异步入队一侧会因为不等待完成而显得更快,得到的差值与绑定开销无关。`bench/cpp_baseline/main.cpp` 现在增加 `RmaOp::PutOnStream` 分支,输出 `put_on_stream` 一列专用于门禁判定。

#### 5.6.2 计时窗口两侧逐项对齐

| 项 | 值 |
| --- | --- |
| 操作 | `aclshmemx_putmem_on_stream`,默认流 |
| warmup | 50 次 |
| 单次窗口 | 500 次操作 + 一次 `aclrtSynchronizeDevice` |
| 窗口内取值 | 总耗时 / 500 |
| 每轮采样 | 40 个窗口取最小值 |
| barrier 位置 | 窗口之外 |
| 引导方式 | 两侧同为 uid + ip_port,同一 launcher 形态 |

上述七项两侧取值完全相同,对应 `cpp_baseline/main.cpp` 与 `bench_gate_e2e.py` 中同名的 `g_warmup_iters` / `g_timed_iters` / `g_repeats` 三个常量(50 / 500 / 40)。窗口数必须一致:取的是最小值,采样越多越容易取到更低的 min,两侧不等会系统性地偏向采样多的一方。

排空必须是**主机侧真实等待**。若改用 `aclshmemx_quiet_on_stream` 只入队不等待,各尺寸会测出相同耗时(实测 4 KB 与 1 MB 都是 3.5 μs),那是入队吞吐而非传输时延——此类读数已废弃。

#### 5.6.3 计算公式

```
overhead_us      = python_put_min - cpp_put_min
overhead_percent = overhead_us / cpp_put_min × 100
```

两侧同机、同卡数、同尺寸、同计时口径,故可直接相减。脚本 `bench/run_gate.sh` + `bench/summarize_gate.py`。

#### 5.6.4 为什么必须多轮取中位

4 KB 处偶发离群:同一配置下多数轮次落在 4.27–4.35 μs,但每若干轮会有一轮跳到 4.7–4.9 μs,两侧都会出现。该抖动幅度(约 0.5 μs)远大于绑定开销(约 0.16 μs),单轮采样会把它误读成系统性开销——本任务早期就据此得出过 15.4% 的错误结论。现按 9 轮取中位判定(理由见 5.7 末「关于轮数的选择」),并**完整打印每轮原始样本**,使离群可见而不被平滑掉。轮数须为奇数:偶数个样本的中位是中间两值的平均,会把离群折进结果而非排除,`run_gate.sh` 会校验并拒绝。

#### 5.6.5 差分探针的定位

`bench/bench_fastcall_probe.py` 用差分法比较 pybind11 与 vectorcall 两种绑定,给出的是**两者之差**,用于回答"换绑定机制值不值",不能据此推出任一方相对 C++ 的绝对开销。门禁判定以 5.6.3 的端到端对照为准。其共同基线为空 Python 循环:

```
pybind 绑定开销     = loop(_bench_pybind5) - loop(lambda: None)
vectorcall 绑定开销 = loop(_bench_fast5)   - loop(lambda: None)
vectorcall 下界     = loop(_bench_fast_noop0) - loop(lambda: None)
```

前两个探针接收与 `putmem_on_stream` 相同的五个整数参数、做相同转换、均不做 RMA,故差值即分发机制本身的成本;第三个为无参 vectorcall,给出跨界固定成本的下界,用于区分"过边界"与"转参数"两部分。采样为 2000 次 warmup + 20000 次计时窗口,取 9 轮中位(`g_warmup_iters` / `g_timed_iters` / `g_repeats`)。

### 5.7 门禁实测结果

环境:Ascend 910(Atlas 800T A3),CANN 9.0.0,Python 3.11.15,pybind11 3.0.4,Kunpeng 920 @ 2.9GHz。每卡数 9 轮,每轮两侧各一次。

采集时主机 load average 稳定在 22/640 核(约 3.4%)。该条件须记录:绑定开销是纯主机 CPU 工作,主机繁忙时两侧受影响不对称,详见 5.8.1。

**每轮原始样本**(`putmem_on_stream` 最小值,μs):

| 卡数 | 尺寸 | C++ 九轮 | Python 九轮 |
| --- | --- | --- | --- |
| 2 | 4 KB | 4.275 4.254 4.272 4.270 4.271 4.254 4.264 4.259 4.264 | **4.540** 4.269 4.346 4.326 4.264 4.268 4.258 4.300 **4.829** |
| 2 | 64 KB | 5.482 5.481 5.492 5.491 5.494 5.473 5.482 5.479 5.500 | 5.494 5.490 5.498 5.490 5.485 5.486 5.485 5.490 5.498 |
| 2 | 1 MB | 33.218 33.206 33.229 33.240 33.223 33.225 33.225 33.236 33.244 | 33.248 33.229 33.252 33.246 33.237 33.243 33.245 33.240 33.237 |
| 4 | 4 KB | 4.264 4.261 4.303 4.284 4.274 4.266 4.278 4.269 4.276 | **4.573** 4.285 4.278 **4.480** **4.837** 4.271 4.279 4.291 4.285 |
| 4 | 64 KB | 5.493 5.480 5.503 5.489 5.503 5.482 5.498 5.487 5.481 | 5.498 5.485 5.485 5.501 5.516 5.506 5.501 5.501 5.493 |
| 4 | 1 MB | 32.556 32.590 32.605 32.583 32.512 32.590 32.568 32.604 32.580 | 32.569 32.550 32.562 32.621 32.598 32.499 32.589 32.556 32.552 |
| 8 | 4 KB | 4.292 4.273 4.282 4.300 4.333 4.300 4.329 4.309 4.250 | 4.279 4.276 4.265 4.344 4.408 **4.539** **4.542** 4.321 4.275 |
| 8 | 64 KB | 5.536 5.498 5.516 5.531 5.514 5.534 5.547 5.518 5.514 | 5.495 5.511 5.519 5.513 5.507 5.544 5.529 5.524 5.499 |
| 8 | 1 MB | 32.589 32.543 32.574 32.622 32.547 32.591 32.573 32.599 32.482 | 32.561 32.549 32.531 32.568 32.584 32.565 32.613 32.527 32.567 |

粗体为 5.6.4 所述的离群轮次,全部保留在表内,未做任何剔除。离群只出现在 4 KB 且集中在 Python 侧,这与"绑定开销是主机 CPU 工作、更易被调度扰动"一致;64 KB 与 1 MB 两侧均无离群,九轮跨度分别在 0.03 μs 与 0.14 μs 以内。

**判定**(各取九轮中位):

| 卡数 | 尺寸 | C++ 中位 | Python 中位 | overhead | 占比 | 判定 |
| --- | --- | --- | --- | --- | --- | --- |
| 2 | 4 KB | 4.264 | 4.300 | +0.036 | 0.84% | 达标 |
| 2 | 64 KB | 5.482 | 5.490 | +0.008 | 0.14% | 达标 |
| 2 | 1 MB | 33.225 | 33.243 | +0.018 | 0.05% | 达标 |
| 4 | 4 KB | 4.274 | 4.285 | +0.012 | 0.28% | 达标 |
| 4 | 64 KB | 5.489 | 5.501 | +0.013 | 0.23% | 达标 |
| 4 | 1 MB | 32.583 | 32.562 | -0.021 | -0.06% | 达标 |
| 8 | 4 KB | 4.300 | 4.321 | +0.021 | 0.50% | 达标 |
| 8 | 64 KB | 5.518 | 5.513 | -0.005 | -0.10% | 达标 |
| 8 | 1 MB | 32.574 | 32.565 | -0.009 | -0.03% | 达标 |

**三尺寸 × 三卡数全部达标,最差 0.84%。**

overhead 绝对值落在 -0.02 ~ +0.04 μs,低于 5.7.1 单独测得的 vectorcall 绑定开销(约 0.16 μs)——端到端测量中该开销部分被流水线掩盖。两个独立测法在量级上互相印证。

个别格子出现小幅负值(-0.03% ~ -0.10%),量级在运行间抖动之内,应读作"在测量分辨率内两者无差别",而非 Python 快于 C++。

**关于轮数的选择**:为减小离群噪声,判定采用 9 轮取中位。此前 5 轮那批(日志 `logs/gate_discarded/`)九格同样达标,但因主机 load 升至 70–90 属重度 CPU 争抢,采集条件无效而整批弃用,详见 5.9。

#### 5.7.1 绑定机制的选择依据

同机差分(9 轮 × 20000 次调用取中位)测得 `putmem_on_stream`(5 参)的绑定层开销。两侧数据同由 `bench/bench_fastcall_probe.py` 采集——`_bench_pybind5` 与 `_bench_fast5` 两个探针在同一次运行内减去同一个空循环基线,故两者可直接比较;公式见 5.6.5。(`bench/bench_call_overhead.py` 是另一支脚本,测的是 `my_pe` / `aclshmem_ptr` / `aclshmem_putmem` 的 0 参与 2 参开销,不含 5 参探针,不参与本表。)

| 绑定方式 | 绑定层开销 | 机制 |
| --- | --- | --- |
| pybind11 | ≈ 0.267 μs | 重载分发 + 每参 type_caster + 参数元组 + 异常包裹 |
| **vectorcall** | **≈ 0.155 μs** | 直取解释器参数向量,CPython C-API 转换,调同一 C 入口 |

vectorcall 降低约 42%(节省约 0.112 μs/次)。该开销为纯主机 CPU 工作,与传输量、卡数无关。按 5.6.5 所述,此表用于说明为何选 vectorcall,门禁判定不依赖它。

上表为 15 次独立运行的中位,其中 6 次绑核复测以排除 CPU 争抢,结果一致。原始日志 `logs/probe/n46_fastcall_probe.log`(含一轮离群,按 5.6.4 的规矩保留不剔除)。

### 5.8 门禁结论

**任务书门禁「Python Host RMA 相对同路径 C++ overhead ≤ 5%(2/4/8 卡)」已达成**,依据是 5.7 的端到端实测:交付路径 `shmem.core.put` 与 C++ `aclshmemx_putmem_on_stream` 同机、同卡数、同尺寸、同计时口径直接对照,2/4/8 卡 × 三尺寸共九格全部 ≤ 5%,最差 0.84%。

结论限定于该实测环境(Ascend 910 / Atlas 800T A3,CANN 9.0.0,Python 3.11.15,Kunpeng 920 @ 2.9GHz,8 卡,主机 load ≈ 3.4%)。门禁的物理裕度来自:绑定开销约 0.155 μs 对 4 KB 的 C++ 时延约 4.26 μs 占约 3.6%,这是**理论上界**;端到端实测因该开销部分被流水线掩盖而更低(4 KB 三个卡数分别为 0.84% / 0.28% / 0.50%)。

#### 5.8.1 主机负载敏感性(实测,重要)

绑定开销是纯主机 CPU 工作,而 C++ 分母由 NPU/驱动主导。**主机 CPU 繁忙时,两侧受影响并不对称**,这一点在本任务中被实测到,不是推论。

在另一台同款机器(同为 Kunpeng 920 @ 2.9GHz、8 卡 Ascend 910)上,当其 load average 升至 70–90(640 核,约 12–14%)时,同一份代码测得:

| 侧 | 4 KB 五次连测(μs) | 离群比例 |
| --- | --- | --- |
| C++ | 4.290 4.284 4.278 **4.496** 4.281 | 1/5,幅度 +0.2 μs |
| Python | **4.555** **5.452** **5.433** 4.297 **5.662** | 4/5,幅度可达 +1.2 μs |

该状态下 3 轮门禁判定给出 4 KB 占比 11.58%,超出预算。成因是主机 CPU 被抢占后 Python 侧(解释器 + 更多用户态指令)受损远大于 C++ 侧,与绑定实现无关——同一二进制在空闲机器上测得 0.84%。

由此得出两条须写入交付说明的结论:

1. **门禁判定必须在低负载主机上进行。** 建议采集前确认 load average 与核数之比在 10% 以内,并在报告中记录该值。本节 5.7 的数据即在 3.4% 下采集。
2. **该敏感性本身是 Python 绑定的固有属性**,不因实现优化而消失:任何跨 Python↔C 边界的路径都比纯 C++ 路径更依赖主机 CPU 的可用性。vectorcall 已将其压到 pybind11 的约一半,这是实现层能做到的限度。

作为风险边界的估算:即使某主机的 CPU 稳定慢到使绑定开销翻倍至约 0.31 μs(而非上述被抢占的情形),4 KB 占比也只到约 7%,再计入流水线掩盖才是实际值——**门禁余量不足以吸收一倍的主机 CPU 性能差,更不足以吸收重度 CPU 争抢。** 若验收环境负载较高,可用交付的 `bench/run_gate.sh` 现场复现判定,并同时记录 load。

### 5.9 关于此前数值的更正

此前 PR 与文档中出现过 +6.5%、+9.5%、+14.4%、+8.87%、15.4% 等数值,均已废弃,成因分三类:

1. **对照对象不同**:以阻塞 `aclshmem_putmem` 作分母对照异步 `putmem_on_stream`,两者不是同一操作。
2. **排空口径不同**:Python 侧只入队 quiet 不等待完成,导致各尺寸读数相同(4 KB 与 1 MB 皆 3.5 μs),甚至出现"Python 比 C++ 快 89%"的荒谬结果。
3. **单轮采样撞上离群**:4 KB 偶发抖动被当作系统性开销,曾据此得出 15.4%。

此外,评审过程中曾报出"最差 0.74%"与"最差 1.40%"两版数据。前者的 C++ 侧每轮采 40 个窗口而 Python 侧只采 21 个,而判定取的是最小值——采样多的一侧更易取到更低的 min;其偏差方向是**高估** Python 开销(分母被压低),故当时结论仍成立,但既然文档声称"两侧逐项对齐",该不一致本身即错误,已统一为 40 轮。后者口径正确,但采集主机当时 load 升至 70–90,属 5.8.1 所述的重度 CPU 争抢状态,测量条件无效,故整批弃用(**非按数值挑选**:该批数据九格同样达标)。现改在低负载主机(load ≈ 3.4%)以 9 轮重测,以 5.7 为准。

本节以 5.6 的方法与 5.7 的数据为准。原始日志随交付包提供。

## 六、验证环境规划

### 6.1 已具备环境

单节点验证在河套集群完成，跨节点验证在洛书调度平台完成。

| 项 | 单节点 | 跨节点 |
| --- | --- | --- |
| 芯片 | Ascend 910（Atlas A2，`IT22HMDA_2_S`） | Ascend 910 |
| CANN | 9.0.0-beta.2 | 9.0.0 |
| 驱动 | 25.5.2，固件 7.8.0.7.220 | — |
| 容器 | `cann9.0.0.beta2-dev:20260509` | `mindspeed-llm:26.0.0-a3-openeuler24.03-py3.11-aarch64` |
| Python | 3.11.15 | 3.11.14 |
| torch / torch_npu | 2.10.0 | 2.7.1.post4 |
| 构建参数 | `RDMA_SUPPORT=OFF` | `RDMA_SUPPORT=ON` |
| 拓扑 | 单节点 8 卡 | 2 节点 × 2 卡 |

跨节点的两个 pod 由调度器分配在不同物理机，设备编号不重叠（master 为 `/dev/davinci10,11`，worker 为 `/dev/davinci4,5`）。

### 6.2 使用顺序

先在 2 卡上验证功能，再扩到 4 / 8 卡采集性能数据；跨节点部分在洛书平台以 2 节点 × 2 卡验证。

### 6.3 环境与数据保存

构建产物与测试日志保存在共享盘，自测报告记录完整的环境参数与实测数值。

## 七、风险与应对

| 风险 | 应对 |
| --- | --- |
| 小尺寸传输的 Python 开销超出 5% | 热路径改用 vectorcall（`METH_FASTCALL`）绑定，绑定开销从约 0.267 μs 降至约 0.155 μs（-42%），`core.put` 已路由至此。同机 2/4/8 卡实测三尺寸全部达标（最差 0.84%）。4 KB 裕度的主机 CPU 相关性详见 5.8 |
| 用例的对端读取方式依赖单节点地址映射 | 原用例经 `aclshmem_ptr` 取对端地址后直接 `memcpy`，跨节点读到无效数据。已改为经 `aclshmem_getmem` 取回，单节点与跨节点均通过。详见 7.1 |
| `HOST_SIDE` 在验证环境不可用 | 已查明为构建配置而非环境限制:该路径由 `HAS_ACLRT_MEM_FABRIC_HANDLE` 门控,须以 `-DENABLE_CANN_BUILD=ON` 构建。加该开关后在 Ascend 910 实测分配与释放均通过;默认构建参数不含该开关,故随包用例仍只覆盖 `DEVICE_SIDE`,该条件写入限制表 |
| 多实例语义无法实质验证 | 已解决:创建入口是 `InitAttr.instance_id`(此前未绑定),补一行 `def_readwrite` 后即可从 Python 创建非 0 实例。`test_instance_create.py` 覆盖创建、活动实例校验与堆分配,并经反向验证。`core.init` 因在 C++ 内部构造 `InitAttr` 仍固定为实例 0,该边界写入限制表 |
| 格式化噪声淹没功能改动 | 单 commit 交付,格式变动与功能改动同处一次提交,已在 4.2 说明 |

### 7.1 跨节点验证结果

跨节点场景已在洛书平台完成验证，此前列为「环境不具备」的一项现有实测结论。

| 用例 | 单节点 2 PE | 跨节点 4 PE |
| --- | --- | --- |
| `test_cc.py` | 2/2 通过 | 4/4 通过 |
| `test_multi_instance.py` | 2/2 通过 | 4/4 通过 |

RMA 时延对照（微秒，最小值）。注意 `bench_rma_overhead.py` 中对端取 `(pe + 1) % world_size`，4 PE 分布在 2 节点时 rank0 的对端与其同处一节点，故该脚本在多节点下打印的仍是同节点时延；下表另用 `pe ^ 1` 与 `(pe + 2) % ws` 分别指定对端测得：

| 尺寸 | 同节点 | 跨节点 | 比值 |
| --- | --- | --- | --- |
| 4 KB | 4.573 | 6.557 | 1.43× |
| 64 KB | 5.613 | 15.832 | 2.82× |
| 1 MB | 33.543 | 192.290 | 5.73× |

关于 `handle_wait`：3.7 节中「底层实现不支持跨节点句柄」的表述源自头文件注释，实测在节点间 RDMA 路径上该接口正常工作，用例通过。更准确的表述是其跨节点行为未在文档中承诺，实测可用但不宜依赖。

## 八、交付件

| 交付件 | 路径 |
| --- | --- |
| 设计文档 | 本文件 |
| 实现 | 个人仓分支 `feat/python-host-api-extension` |
| 自测报告 | 仓库根目录 `SELFTEST_python_ext.md` |
| README | 仓库根目录 `README_python_ext.md` |
| API 文档 | `docs/api/pythonAPI.md` |
| 跨节点验证报告 | 仓库根目录 `CROSSNODE_REPORT.md` |

## 九、实施门禁

| 门禁 | 状态 |
| --- | --- |
| 20 个绑定编译通过 | 已达成 |
| 7 个既有用例无回归 | 已达成 |
| 5 个新增用例通过 | 已达成(功能 2 + 一致性 1 + 生命周期 1 + 多实例创建 1,分类见 5.1) |
| 断言有效性反向验证 | 已达成 |
| clang-format 零差异 | 已达成 |
| OAT License 头 | 已达成 |
| `pythonAPI.md` 全量覆盖 | 已达成（两个章节共 37 条） |
| 5% 性能预算 | **达成**（交付的 vectorcall 热路径）。同机 2/4/8 卡 × 三尺寸共九格全部达标，最差 0.84%，详见 5.7。4 KB 裕度的主机 CPU 相关性见 5.8 |
| 跨节点功能验证 | 已达成（2 节点 × 2 卡，用例全通过）|

## 十、任务书追踪矩阵

| 任务书要求 | 落实位置 | 状态 |
| --- | --- | --- |
| 20 个 Host API 绑定 | `pyshmem_ext.cpp` | 完成 |
| `shmem.core` 高层封装 | `cc.py`、`multi_instance.py` | 完成 |
| `intptr_t` 指针、GIL 释放、风格对齐 | 3.1 pybind11 公共约定 | 完成 |
| 已知限制不得隐瞒 | 七、风险与应对；自测报告限制表 | 完成 |
| 新增接口均有用例 | 5.2 功能用例矩阵 | 完成 |
| `run.sh` 既有用例通过 | 7/7 | 完成 |
| MTE 路径 | 引擎配置用例 | 完成 |
| RDMA/SDMA/UDMA 冒烟 | 引擎配置用例 | 完成 |
| Python 相对 C++ 开销 ≤ 5%（2/4/8 卡） | 5.6 测量方法；5.7 实测结果；5.8 门禁结论 | **达成**：交付热路径经 vectorcall 绑定（`core.put` 已路由），同机 2/4/8 卡三尺寸全部达标；4 KB 裕度的主机 CPU 相关性已如实说明 |
| RMA 结果与 C++ 一致 | 功能用例数据校验 | 完成 |
| 设计文档 | 本文件 | 完成 |
| 自测报告 | `SELFTEST_python_ext.md` | 完成 |
| 个人仓 + README | 分支 + `README_python_ext.md` | 完成 |
| 邀请 `Ascend-CANN` | — | 待办 |
