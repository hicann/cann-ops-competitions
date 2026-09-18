# static_multiset 容器开发设计文档（Atlas 950 / Ascend C）

> 容器：`aclco::StaticMultiset<Key>`（静态容量多值集合，对标 cuCollections `cuco::static_multiset`）
> 工程模式：ops-collections 纯头文件容器 —— `include/static_multiset.h`、`include/static_multiset_ref.h`、`include/detail/static_multiset/`
> 硬件与环境：Atlas 950 系列（及后续支持 SIMT 能力的昇腾产品）；CANN ≥ 9.0.0-beta.2；CMake ≥ 3.16；Catch2 v3.5.4
> 测试：`tests/static_multiset/`（任务书功能基线 1258 例，当前实现含额外边界展开共 1264 例）、`tests/performance/static_multiset/`（性能 44 例）

> 实现对齐记录（2026-09-17）：本文以工作树 `ops-collections/include/static_multiset.h` 及
> `include/detail/static_multiset/` 为准。历史草稿中提及的默认双重哈希、显式 `Destroy()`、256 线程、
> 单线程检索收尾均不是最终实现；以下对齐章节覆盖与历史段落冲突的描述。

## 实现对齐与审查清单

### 当前实现摘要

| 项目 | 最终设计 |
| --- | --- |
| 默认存储 | `Storage<1>`，每槽一个 I32/I64 Key，以 `emptyKey` 表示空槽 |
| 默认探测 | `MultisetLinearProbing<MultisetIntegerHash<Key>>`，步长为 1，容量保持请求值（非双重哈希质数取整） |
| 键对布局 | `p = hash(key) >> 1`；优先起点 `4*p`，两个相邻键落在偶数对齐的两个槽；超出表尾时回退到环形起点；保证线性探测完整覆盖表 |
| I32 插入 | 相邻且对齐的两个初始空槽走一次 U64 CAS；其他情况回退逐键 CAS，保持多值语义 |
| I64 插入 | 原生 U64 单槽 CAS；成功/失败计数按 warp 归约后原子提交 |
| 查询/计数 | 1024 SIMT 线程/核、grid-stride；无删除使首个空槽可安全终止探测链 |
| Retrieve | 计数、chunk 内前缀、chunk 基址、写出四阶段；尾 chunk 只处理有效查询 |
| RetrieveAll | 每次全表 ballot 压缩到动态 UB，按 tile 全局原子预约输出区间并协作写出；不保留跨调用的整表 Key 快照 |
| 生命周期 | RAII 析构释放设备资源；没有公开显式 `Destroy()` 接口 |

### 关键不变量

1. 插入仅由空槽 CAS 改写为 Key，容器无删除，因此任一已插入 Key 的探测链在首个空槽之前不会出现空洞。
2. multiset 插入遇到相同 Key 不停止，继续寻找空槽；Count/Retrieve 因而必须扫描到首个空槽，保留全部重复副本。
3. `Retrieve` 与 `RetrieveAll` 的输出顺序不承诺；返回总数不受输出容量截断影响。
4. RetrieveAll 不保存用户输出或整表 Key 副本；每次调用读取当前主表，因此写操作后无需维护额外缓存失效状态。

### 审查要求映射

| 审查项 | 设计响应与证据 |
| --- | --- |
| 模板必填章节 | 本文保留“需求背景、需求分析、详细设计、可维可测分析”四个 required 章节 |
| 接口/语义一致性 | 14 个接口与 `docs/design/semantics_spec.md`、`tests/static_multiset/` 逐项对照；Insert 返回失败数，Retrieve/RetrieveAll 返回实际总数 |
| 性能 44 项 | 统一用 `scripts/build.sh -p` 构建；每项要求 `device_time <= benchmark / 0.4`。最终无缓存实测为 42/44 通过，RetrieveAll I32/I64 未达标，原因和原始日志写入自测报告 |
| 内存与工作区 | 主表为 `Capacity * sizeof(Key)`；Retrieve 的 query/chunk 工作区按调用规模增长；RetrieveAll 仅使用调用期动态 UB 和计数器，不保留容量级 Key 缓存 |
| 代码审查 | 核入口使用 `COLLECTION_AIV_GLOBAL`；SIMT worker 有 grid-stride；CAS/原子只更新受控槽或计数器；所有同步 API 均检查 ACL 返回值 |
| 流程审查 | 设计文档 PR 标题使用 `【社区任务】static_multiset算子设计文档`，评论 @condfuse_3、@fullt、@Ascend-CANN，审核通过后再提交代码验收 |

# 需求背景（required）

## 需求来源

社区任务"9 月社区任务-multiset 容器开发（950）"任务书：参考 cuCollections
`include/cuco/static_multiset.cuh`（dev 分支），在昇腾 NPU 上以 Ascend C 实现功能一致的静态容量多值集合容器
及相关算子，完成设计、开发、测试全流程，以 PR 合入 `https://gitcode.com/cann/ops-collections`；设计文档按
`resources/design_template.md` 提交至 `cann-ops-competitions` 仓库评审。

## 背景介绍

### static_multiset 容器实现现状分析

ops-collections 既有约定：对外接口放 `include/<容器>.h`，设备侧引用放 `include/<容器>_ref.h`，实现细节放
`include/detail/<容器>/`；容器生命周期、参数校验与 ACL 流管理由 C++ 侧完成，设备侧批量访问与计算由 Ascend C
Kernel 承担；测试分别放 `tests/<容器>/`、`tests/performance/<容器>/`。仓库已有 StaticSet（单值集合）与
StaticMap / 单值配套容器，**无多值集合语义实现**：StaticSet 对同一 Key 二次插入会被拒绝或去重，而本任务要求
同一 Key 重复出现且计数与检索保留多值语义。设计落点是在既有开放寻址骨架上引入**重复元素插入、逐键多重数
输出与多重数扫描**。

### static_multiset 功能分析

静态容量哈希表，仅存 Key（无 Value），无删除接口，容量创建时固定。Key 支持 I32、I64；同一 Key 可重复出现，
`Size` 为元素总数（含重复）；`keys` 为一维 `[NumInputs]`（ND），`numInputs ∈ [0, 容量]`；插入 Key 不得等于保留
哨兵（`numeric_limits<Key>::lowest()`）；`Contains` 输出 1 字节标志、`Count` 以 uint64 标量返回、`CountEach/
CountEachOuter` 逐键输出 uint64 计数；`If` 系列 stencil 为 uint32 `[NumInputs]`，谓词须 `COLLECTION_SIMT_DEVICE`
可调用；容量为 0 抛异常，输出空间不足按接口语义截断。数量为 0 短路；`Insert/InsertIf` 的空输入返回全部失败，查询/检索的空指针直接返回且不写输出。入参顺序与
cuCollections 一致（区间 `[first, last)` → 首地址 + 元素个数，`cuda::stream_ref` → `aclrtStream`）；Key 由模板
参数编译期实例化（`StaticMultiset<int32_t>` / `<int64_t>`），与用例
`TEMPLATE_TEST_CASE_SIG(..., ((typename Key, int Dummy), ...), (int32_t,0), (int64_t,0))` 两实例对应。

# 需求分析（required）

## 需求描述

在 Atlas 950 系列上实现 ops-collections 纯头文件 StaticMultiset：支持构造、析构、清空、Insert、InsertIf、
Contains、ContainsIf、Find、FindIf、Count、CountEach、CountEachOuter、Retrieve、RetrieveAll；Key 支持 I32、
I64；同一 Key 可重复出现且计数、检索保留多值语义；结果与参考实现语义一致且可重复；覆盖空输入、重复 Key、
不同 multiplicity、命中/未命中、容量边界等泛化场景（测试落 `tests/static_multiset/`、
`tests/performance/static_multiset/`）。

## 需求拆解

1. **工程与接口**：三头文件路径 + 仓库既有 CMake（≥3.16）/ Catch2 v3.5.4；14 个大驼峰接口；不新增独立算子
   目录。
2. **多值集合语义**：同 Key 重复插入全部保留、`Size` 递增；`Find` 命中返回被查询的 Key 本身、未命中返回哨兵；
   `Count` 返回多重数之和；`CountEach/CountEachOuter` 逐查询输出多重数；`Retrieve/RetrieveAll` 返回全部命中。
3. **If 系列与错误**：stencil + Device 谓词决定元素是否参与运算；容量取整后 `Capacity() ≥ 请求值`，写满后
   `Insert` 返回失败计数，不覆盖、不搬移已有元素。
4. **性能与交付**：44 条性能用例逐行对标 0.4× 阈值；最终无缓存版本 42 条通过、RetrieveAll I32/I64 两条未通过，交付报告附原始数据和内存取舍说明。

# 详细设计（required）

## 接口清单

共 14 个接口，大驼峰命名。下表简记：`Extent` = `aclco::Extent<std::size_t>`，`stream` = `aclrtStream stream = nullptr`，
`SizeType` = 元素计数返回类型（uint64），`↔` 后为对标 cuCollections 的接口。

| # | 接口 | 声明 | 返回 | 语义要点（`↔` 后为对标 cuCollections 接口） |
| --- | --- | --- | --- | --- |
| 1 | Create | `StaticMultiset(Extent capacity, Key emptyKey, stream)` | 容器对象（RAII） | 合法容量分配 + 写哨兵；`capacity = 0` 抛异常；`Capacity() ≥ 请求值`、`Size() = 0`。↔ `static_multiset(capacity, empty_key_sentinel, ...)` |
| 2 | Destroy | `~StaticMultiset()`（RAII）+ 移动构造/赋值 | void | 释放容器拥有的设备存储；没有公开显式 `Destroy()`。↔ 析构 |
| 3 | Clear | `void Clear(stream)` | void | 复位哨兵、计数器归零；空容器幂等；可复用。↔ `clear` |
| 4 | Insert | `SizeType Insert(void* keys, Extent keyNum, stream)` | **失败数** | 同 Key 重复不覆盖，写入探测链首个可用槽位；数量 0 空操作；`nullptr` 且数量 > 0 返回全部失败；`failed = keyNum − inserted`（口径与参考实现相反）。↔ `insert` |
| 5 | InsertIf | `template <typename StencilT, typename Pred> SizeType InsertIf(void* keys, StencilT* stencil, Extent keyNum, stream)` | **失败数** | 仅 `pred(stencil[i])` 为真的元素插入；跳过的不计失败。↔ `insert_if` |
| 6 | Contains | `void Contains(void* keys, void* output, Extent keyNum, stream)` | void | 逐键写 1 字节标志（命中 1 / 未命中 0），每个输入位置都写；参考实现返回"是否全部命中"，此处改逐元素输出。↔ `contains` |
| 7 | ContainsIf | `template <typename StencilT, typename Pred> void ContainsIf(void* keys, StencilT* stencil, void* output, Extent keyNum, stream)` | void | 谓词为假输出 0。↔ `contains_if` |
| 8 | Find | `void Find(void* keys, void* found, Extent keyNum, stream)` | void | 命中写被查询的 Key 本身；未命中写 `emptyKey` 哨兵。↔ `find` |
| 9 | FindIf | `template <typename StencilT, typename Pred> void FindIf(void* keys, StencilT* stencil, void* found, Extent keyNum, stream)` | void | 跳过与未命中均写 `emptyKey`。↔ `find_if` |
| 10 | Count | `SizeType Count(void* keys, Extent keyNum, stream)` | **多重数之和** | 逐查询键累加出现次数；空输入返回 0。↔ `count` |
| 11 | CountEach | `void CountEach(void* keys, void* counts, Extent keyNum, stream)` | void | 逐查询键写 uint64 多重数（未命中写 0），输出长度 = `keyNum`。↔ `count_each` |
| 12 | CountEachOuter | `void CountEachOuter(void* keys, void* counts, Extent keyNum, stream)` | void | 逐查询键写 `max(多重数, 1)`：未命中亦输出 1，保证每个查询至少一条记录（outer 语义）。↔ `count_each` 的 outer 变体 |
| 13 | Retrieve | `SizeType Retrieve(void* keys, Extent keyNum, void* probeOut, void* matchOut, Extent matchCapacity, stream)` | **命中总数** | 输出 `(probeOut, matchOut)` 配对（多值集合中二者相同）；输出按 `matchCapacity` 截断，但返回值不截断。↔ `retrieve` |
| 14 | RetrieveAll | `SizeType RetrieveAll(void* output, Extent maxKeys, stream)` | **非空槽总数** | 导出容器全部 Key（含重复），顺序不承诺；输出按 `maxKeys` 截断，但返回值不截断。↔ `retrieve_all` |

辅助接口（不计入 14 个）：`Size(stream)`、`Capacity()`；输出缓冲由调用方按 `numInputs` / `capacity` 预分配。

## 算子分析

### 数学公式与判定式

```
h = sanitize(integer_hash(key))
p = h >> 1;  b = 4 * p
start = (b + (h & 1) < C) ? b + (h & 1) : fallback(p, h & 1, C)
slot(k) = (start + k) mod C,  k = 0 … C-1
```

`C` 是实际槽位数，默认 `Storage<1>`。相邻整数键构成的键对优先放入偶数对齐的两个相邻初始槽，并在其后保留
空隙；这既避免连续输入形成长主簇，也允许 I32 的相邻输入使用一次 U64 CAS。`fallback` 只在键对初始区跨越表尾时
使用，不改变步长为 1 的完整环形探测。插入写探测链首个空槽；命中判定为槽值等于查询键；多重数自起点扫描至首个
空槽。`Size` 为所有已写入槽位数，包含重复 Key。

两条由判定式导出的设计约束：

1. **查询可在首个空槽终止**：插入只写空槽、从不产生空槽，故"已存元素的探测链前缀无空槽"这一不变式**单调
   保持**；未命中判定、多重数扫描与 `CountEach` 逐键计数只需扫连续前缀，期望代价 O(1 + 多重数)。
2. **只需 Key 槽位、无需值槽判空**：多值集合的元素本身就是 Key，重复 Key 的第 k 个元素落在探测链上第 k 个
   空槽，故可用槽位判定仅需「key 槽位为空哨兵」一条，存储与同 Key 判重逻辑都比 multimap 更省。

### 支持数据类型

Key 为 int32 或 int64（模板实例化固定）；容器仅存 `keys[C]` 单数组（无 Value 数组），哨兵为
`numeric_limits<Key>::lowest()`，插入 Key 不得等于哨兵。

### 支持形状

`keys` 为一维 `[NumInputs]`（ND）；`CountEach/CountEachOuter` 输出为 uint64 `[keyNum]`；`Retrieve` 的
`probeOut`/`matchOut` 长度均为调用方给定的 `matchCapacity`。

## 算子实现

### 工程结构与文件布局

```text
ops-collections/
├── include/static_multiset.h          # 对外接口（14 个）
├── include/static_multiset_ref.h      # 设备侧引用：Kernel 内可用的 device view / mutable view
├── include/detail/static_multiset/    # static_multiset_impl.h、static_multiset_ref_impl.h、kernels.h
├── tests/static_multiset/             # 功能用例（14 个接口各对应一个测试文件）
├── tests/performance/static_multiset/ # 性能用例（perf_create/destroy/insert/contains/find/count/
│                                      # count_each/count_each_outer/retrieve/retrieve_all）
└── docs/static_multiset_API文档和使用示例.md
```

`static_multiset.h` 暴露容器类；`static_multiset_ref.h` 暴露设备侧视图（对标 cuCollections 的 `device_view` /
`device_mutable_view`）；`detail/` 不作为对外承诺。

### Host 侧设计

容器成员：`keys[C]`、空键标记、size/失败/汇总计数器，以及 Retrieve 的 query/chunk 前缀工作区。RetrieveAll
不保存导出快照或其他容量级 Key 副本。

| 校验项 | 触发条件 | 行为 |
| --- | --- | --- |
| 容量合法 | 请求容量 = 0 或超出 Device 内存可行上限 | 抛 `std::invalid_argument` / `std::bad_alloc` |
| 输入/输出指针 | 数量为 0 或必需指针为空 | 0 数量短路；Insert/InsertIf 的空输入返回全部失败；查询/检索空指针直接返回或不写输出 |
| 输出容量 | `Retrieve` / `RetrieveAll` 输出容量不足 | 仅截断写出，返回实际总数 |
| 哨兵合规 | 插入 Key 等于哨兵 | Device 侧按失败处理；查询哨兵按未命中处理 |
| 容量 | 请求容量为 0 | 构造抛 `std::invalid_argument` |

关键路径：

1. **Create**：一次 `aclrtMalloc` 申请 `keys[C]` → 一次 Kernel 顺序写哨兵 → 计数器清零。基线（I32 4.152062 ms、
   I64 8.740130 ms，约 2 倍关系）表明成本随数据量线性增长（C = 1e8 时约 400 MB / 800 MB），必须**单次连续
   填充**；标杆与"槽位总字节数"成正比，故不应引入任何额外数组。
2. **Insert / InsertIf**：沿调用方 stream 下发 1024-thread SIMT kernel；输入先在动态 UB 分块重排以错开 CAS
   目标，I32 对齐相邻键尝试一次 U64 CAS，其他键使用原生槽宽 CAS。成功/失败先在 warp 内归约；接口同步后返回失败数。
3. **Contains / ContainsIf / Find / FindIf**：纯异步无返回值，逐元素写 1 个标志或 Key，不引入同步。
4. **Count / CountEach / CountEachOuter / Retrieve**：Count 按 warp 归约后写一个 uint64 汇总；Retrieve 采用
   「逐查询计数 → chunk 内独占前缀 → chunk 总数前缀 → 并行写出」，没有单线程收尾热点。
5. **RetrieveAll**：全表 ballot 压缩到动态 UB，每 tile 用全局原子预约输出区间，再协作写出；无论输出容量是否完整，均从当前表直接扫描，不保存跨调用 Key 快照。
6. **Clear / 析构**：Clear 单次 kernel 复位哨兵和 size；析构由成员 RAII 释放设备存储。

### Kernel 侧设计（Ascend C / SIMT）

采用 AIV 入口 + `Simt::VF_CALL` 的 SIMT 模型：每 block 1024 线程，grid-stride 分配键/查询；Insert 与
RetrieveAll 使用动态 UB，查询和计数以 GM 直访为主。所有核函数接收表首地址、输入/输出地址、数量、空键与
必要计数器。

| 核函数 | 网格 | 每线程工作 | 关键点 |
| --- | --- | --- | --- |
| `MultisetInsertIf` | AIV × 1024 | UB 路由、探测、CAS | I32 对齐键对优先 U64 CAS；I64 单槽 U64 CAS |
| `MultisetContains/Find/Count/CountEach` | AIV × 1024 | 逐元素线性探测 | 遇空槽终止；Count 汇总按 warp 归约 |
| `MultisetRetrieveFused*` | 多 kernel | 计数、chunk 前缀、写出 | chunk 边界只访问有效 query，输出容量仅限制写入 |
| `MultisetRetrieveAllCompact` | AIV × 1024 | ballot 压缩、原子预约、协作写出 | 每次从主表扫描；顺序不承诺 |

`If` 系列复用同一套核函数，仅增加 stencil 入参与谓词模板参数，谓词形如
`COLLECTION_SIMT_DEVICE bool operator()(StencilT) const noexcept`（与用例 `IsOdd` 一致），谓词为假直接跳过。
`Retrieve/RetrieveAll` 顺序不承诺但条数与内容集合必须精确（用例按排序后比对）；`keys[C]` 按 128B 对齐。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas 950 系列（DAV_3510，支持 SIMT）及后续支持 SIMT 能力的昇腾系列产品 | √ |

## 算子约束限制

- 仅支持 int32、int64 两种模板实例；其它类型不承诺；容器无 Value 概念，需要键值对时用 static_multimap。
- 无删除接口，容量创建后固定；写满后 `Insert/InsertIf` 返回失败计数，已存元素不被覆盖或搬移。
- 查询键与插入 Key 不得等于哨兵值，等于哨兵按无效处理。
- `Find/FindIf` 返回被查询 Key 本身（多值集合无 Value 可选）；`Count` 返回多重数之和而非去重键数；`CountEach`
  未命中输出 0，`CountEachOuter` 未命中输出 1（`max(count, 1)`）。
- `Retrieve/RetrieveAll` 顺序不承诺；容量不足只截断写出，返回值仍为实际总数。跨 stream 使用同一容器需调用方
  自行同步，容器不提供内部锁。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | I32/I64 的 `Insert/InsertIf` 状态变化，及 `Contains/ContainsIf/Find/FindIf/Count/CountEach/CountEachOuter/Retrieve/RetrieveAll` 输出与 cuCollections 语义一致；相同输入重复执行时状态与输出一致（`Retrieve/RetrieveAll` 的"一致"按集合语义判定，用例按排序后比对，顺序差异不计为不一致） | 任务书 3.2 |
| 性能标准 | 44 条性能用例全部满足 **算子时延 ≤ 标杆时延 / 0.4**（即 ≥0.4× 标杆性能）；不达标须给出合理解释 | 任务书 3.3 |
| 内存标准 | 除容器静态存储、输出空间与必要工作空间外，不产生与输入规模线性重复的额外 Device 内存拷贝；最终实现不保留 RetrieveAll 整表 Key 快照 | 任务书 3.4 |

`CountEach[i] = oracle.count(queries[i])`，`CountEachOuter[i] = max(oracle.count(queries[i]), 1)`（与任务用例
oracle `ExpectedCountEach(oracle, queries, outer)` 一致），输出与查询顺序逐元素严格对齐。

性能基准（44 条用例逐行对应基线表）：

| 接口 | 条数 | 标杆时延区间（ms） | 时延上限 = 标杆 / 0.4（ms） |
| --- | --- | --- | --- |
| create / destroy | 2 + 2 | 4.152062 – 8.740130 / 4.360688 – 8.580200 | 10.380155 – 21.850325 / 10.901720 – 21.450500 |
| insert / retrieve_all | 2 + 2 | 8.098100 – 8.392043 / 0.660264 – 1.016854 | 20.245250 – 20.980108 / 1.650660 – 2.542135 |
| contains / find | 6 + 6 | 3.855939 – 4.755204 / 4.332860 – 5.251019 | 9.639848 – 11.888010 / 10.832150 – 13.127548 |
| retrieve / count | 6 + 6 | 8.770419 – 10.845923 / 4.196007 – 5.010463 | 21.926048 – 27.114808 / 10.490018 – 12.526158 |
| count_each / count_each_outer | 6 + 6 | 4.874588 – 5.596096 / 5.265664 – 5.980948 | 12.186470 – 13.990240 / 13.164160 – 14.952370 |

表中除 create/destroy 外均为 `NumInputs = 100000000`、`Occupancy = 0.5`；insert/retrieve_all 与
retrieve/count/count_each/count_each_outer 为 `Distribution = UNIFORM`、`Multiplicity = 1`、
`MatchingRate = 0.1/0.5/1`，contains/find 为 `Distribution = UNIQUE`、`MatchingRate = 0.1/0.5/1`；
create/destroy 为 `Capacity = 100000000`；dtype 均含 I32 与 I64。

代表性逐行标杆（括号内为上限 = 标杆 / 0.4）：create I32 4.152062 ms（10.380155 ms）、retrieve I64 10.845923 ms
（27.114808 ms）。

最终无缓存实测结果为 42/44 通过。未通过的两项均为 RetrieveAll：I32 为 2.011280 ms（上限 1.650660 ms），I64 为 3.711200 ms（上限 2.542135 ms）。为符合任务书 3.4，最终代码删除了历史实现中的 `Capacity * sizeof(Key)` 持久 Key 快照；直接扫描路径受主表读取与压缩输出带宽限制。完整原始日志和逐项结果见验收交付件。

## 内存要求

基础容器存储为 `keys[C]`（I32 约 4·C 字节、I64 约 8·C 字节，C = 1e8 时约 400 MB / 800 MB），外加 O(核数)
计数与 Retrieve 的 chunk 工作区。RetrieveAll 仅使用 kernel 动态 UB 和已有计数器，不申请或保留 `C*sizeof(Key)` 的容器私有快照；因此不会产生容量级持久 Key 副本。

## 测试设计

功能测试（`tests/static_multiset/`，任务书基线 1258 例；当前实现额外增加 6 个边界展开，共 1264 例，每个接口对应同名测试文件）：Create 14、Destroy 18、Clear 32、
Insert 198、InsertIf 20、Contains 100、ContainsIf 56、Find 100、FindIf 56、Count 124、CountEach 122、
CountEachOuter 122、Retrieve 98、RetrieveAll 204（dtype 覆盖 I32 与 I64；额外用例覆盖容量 1、保留空键和重复导出）。用例按 dtype
实例、`GENERATE` 参数组合与 `SECTION` 分支展开，覆盖容量 5…100000 × occupancy 0/0.1/0.5/0.9/1 ×
multiplicity 1/2/4/8/1024 × MatchingRate 0/0.1/0.5/1，以及空输入、重复 Key、`nullptr` 全失败、容量边界失败数、
Clear 后复用、If 系列三种 stencil 模式、Create 容量 0 拒绝、Find 返回被查询 Key、`CountEach` 未命中输出 0 与
`CountEachOuter` 未命中输出 1。

性能测试（`tests/performance/static_multiset/`，44 例）：`perf_create / perf_destroy / perf_insert /
perf_contains / perf_find / perf_count / perf_count_each / perf_count_each_outer / perf_retrieve /
perf_retrieve_all`，经 `REGISTER_PERFORMANCE_TEST` / `REGISTER_PERFORMANCE_ARGS` 注册，逐行对应 3.3 基线表。

## 自验要求

1. 用任务提供的 `test-cases/` 全量执行功能与性能用例，提交完整日志（入参、结果对比、性能数据）。
2. 性能自验给出 44 条用例的算子时延 vs 标杆时延、倍率与逐条达标结论，未达标项给出成因与优化方向；环境为
   CANN ≥ 9.0.0-beta.2 + Atlas 950 系列设备 + CMake ≥ 3.16 + Catch2 v3.5.4。

## 兼容性分析

- **新增容器**：复用既有开放寻址探测骨架（与同批 static_multimap 任务共享设计）但不修改现有 StaticSet /
  StaticMap 的接口与行为，新增 `include/static_multiset.h` 不影响既有容器编译与 ABI；仅实例化 I32、I64。
- **与 cuCollections 的差异**：迭代器区间 → 首地址 + 个数；`Insert/InsertIf` 返回失败数而非成功数；
  `Contains/ContainsIf/Find/FindIf` 改为逐元素输出（批量接口无返回标量的位置）。
- **硬件与构建兼容**：仅依赖 GM 直访与 SIMT 线程模型，不含 950 专有指令；沿用仓库 CMake（≥3.16）与 Catch2
  v3.5.4，测试目录符合 `tests/<容器>/`、`tests/performance/<容器>/` 约定。
