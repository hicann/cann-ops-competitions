# static_multiset 容器开发设计文档（Atlas 950 / Ascend C）

> 容器：`aclco::StaticMultiset<Key>`（静态容量多值集合，对标 cuCollections `cuco::static_multiset`）
> 工程模式：ops-collections 纯头文件容器 —— `include/static_multiset.h`、`include/static_multiset_ref.h`、`include/detail/static_multiset/`
> 硬件与环境：Atlas 950 系列（及后续支持 SIMT 能力的昇腾产品）；CANN ≥ 9.0.0-beta.2；CMake ≥ 3.16；Catch2 v3.5.4
> 测试：`tests/static_multiset/`（功能 1258 例）、`tests/performance/static_multiset/`（性能 44 例）

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
可调用；容量为 0、输出空间不足、stencil 长度不匹配、空指针（数量 > 0）与非法流均须报错。入参顺序与
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
4. **性能与交付**：44 条性能用例满足 ≥0.4× 标杆；交付设计文档、代码与测试、自测报告、个人仓地址并更新 README。

# 详细设计（required）

## 接口清单

共 14 个接口，大驼峰命名。下表简记：`Extent` = `aclco::Extent<std::size_t>`，`stream` = `aclrtStream stream = nullptr`，
`SizeType` = 元素计数返回类型（uint64），`↔` 后为对标 cuCollections 的接口。

| # | 接口 | 声明 | 返回 | 语义要点（`↔` 后为对标 cuCollections 接口） |
| --- | --- | --- | --- | --- |
| 1 | Create | `StaticMultiset(Extent capacity, Key emptyKey, stream)` | 容器对象（RAII） | 合法容量分配 + 写哨兵；`capacity = 0` 抛异常；`Capacity() ≥ 请求值`、`Size() = 0`。↔ `static_multiset(capacity, empty_key_sentinel, ...)` |
| 2 | Destroy | `~StaticMultiset()` + 显式 `void Destroy()` + 移动构造/赋值 | void | 释放存储、置句柄无效；显式 Destroy 幂等。↔ 析构 |
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
| 13 | Retrieve | `SizeType Retrieve(void* keys, Extent keyNum, void* probeOut, void* matchOut, Extent matchCapacity, stream)` | **实际条数** | 输出 `(probeOut, matchOut)` 配对（多值集合中二者为同一 Key）；条数 = min(命中总数, `matchCapacity`)。↔ `retrieve` |
| 14 | RetrieveAll | `SizeType RetrieveAll(void* keys, Extent capacity, stream)` | **实际条数** | 导出容器全部 Key（含重复），顺序不承诺；条数 = min(`Size()`, `capacity`)。↔ `retrieve_all` |

辅助接口（不计入 14 个）：`Size(stream)`、`Capacity()`；输出缓冲由调用方按 `numInputs` / `capacity` 预分配。

## 算子分析

### 数学公式与判定式

```
h1 = hash1(key) ; h2 = hash2(key) | 1               // h2 取奇数，与 2 的幂容量互质 ⇒ 探测可覆盖全表
slot(k) = (h1 + k · h2) & (C - 1),  k = 0 … C-1     // C = Capacity()，2 的幂
插入：写入探测链上首个空槽（同一 Key 的重复元素顺次占用后续空槽）
命中判定：key 槽位 == 查询键
多重数：自 h1 起连续统计 key 槽位 == 查询键 的个数，遇首个空槽终止
Size = 已写入槽位总数（含重复 Key 的每个元素）
```

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
├── include/detail/static_multiset/    # probing.h（双散列探测）、storage.h（Key 数组+计数器）
│                                      # kernels.h（init/insert/query/reduce）、predicates.h
├── tests/static_multiset/             # 功能用例（14 个接口各对应一个测试文件）
├── tests/performance/static_multiset/ # 性能用例（perf_create/destroy/insert/contains/find/count/
│                                      # count_each/count_each_outer/retrieve/retrieve_all）
└── docs/static_multiset_API文档和使用示例.md
```

`static_multiset.h` 暴露容器类；`static_multiset_ref.h` 暴露设备侧视图（对标 cuCollections 的 `device_view` /
`device_mutable_view`）；`detail/` 不作为对外承诺。

### Host 侧设计

容器成员：默认 stream、请求容量与 `Capacity()`、`empty_key_sentinel`、Device 侧 `keys[C]`、元素计数器、逐核
局部计数缓冲。

| 校验项 | 触发条件 | 行为 |
| --- | --- | --- |
| 容量合法 | 请求容量 = 0 或超出 Device 内存可行上限 | 抛 `std::invalid_argument` / `std::bad_alloc` |
| 指针非空 | 载入类接口数量 > 0 但输入/输出指针为空（Insert 例外） | 抛 `std::invalid_argument` |
| 数量合法 | `numInputs > Capacity()`；`RetrieveAll` 输出容量为 0 而 `Size() > 0` | 抛 `std::invalid_argument` |
| 哨兵合规 | 插入 Key 等于哨兵 | 按"不得为保留空键值"拒绝（可判定项 Host 校验，Device 侧跳过） |
| 句柄/流有效 | `Destroy()` 之后调用接口；stream 非空但非本设备有效流 | 分别抛 `std::logic_error` / `std::runtime_error` |

关键路径：

1. **Create**：一次 `aclrtMalloc` 申请 `keys[C]` → 一次 Kernel 顺序写哨兵 → 计数器清零。基线（I32 4.152062 ms、
   I64 8.740130 ms，约 2 倍关系）表明成本随数据量线性增长（C = 1e8 时约 400 MB / 800 MB），必须**单次连续
   填充**；标杆与"槽位总字节数"成正比，故不应引入任何额外数组。
2. **Insert / InsertIf**：沿调用方 stream 异步下发，每线程处理一个 Key；失败按核局部计数，收尾归约为 1 个标量
   回读（8B D2H + 一次同步）后作为返回值；数量为 0 直接返回 0。
3. **Contains / ContainsIf / Find / FindIf**：纯异步无返回值，逐元素写 1 个标志或 Key，不引入同步。
4. **Count / CountEach / CountEachOuter / Retrieve / RetrieveAll**：`Count` 与检索类接口采用「Device 侧归约 →
   8B D2H → 一次同步」；`CountEach/CountEachOuter` 逐键输出、输出长度已知为 `keyNum`，除边界外保持纯异步。
   检索写入在 Device 侧单线程收尾，保证条数精确且不越界写。
5. **Clear / Destroy**：单次 Kernel 复位哨兵 + 计数器清零；Destroy 直接释放存储并置空句柄，不引入额外流同步。

### Kernel 侧设计（Ascend C / SIMT）

采用 SIMT 编程模型（每 block 256 线程，`__simt_vf__` 设备函数 + `asc_vf_call` 由核函数调用），纯 GM 直访、
不依赖 UB，因此**无 tiling 结构**，核函数仅接收标量参数（keys 首地址、C、数量、哨兵）。

| 核函数 | 网格 | 每线程工作 | 关键点 |
| --- | --- | --- | --- |
| `MultisetInitKernel` | useNumBlocks | grid-stride 写哨兵、计数器清零 | 单次顺序写 |
| `MultisetInsertKernel<Key>` | useNumBlocks | 探测 → CAS 占用槽位；失败局部计数 | 原子 CAS 保证单一写入者，失败者继续探测下一位置（不重试同一槽位） |
| `MultisetQueryKernel<Key,Op>` | useNumBlocks | 逐元素探测；Op ∈ {Contains, Find, Count, CountEach, CountEachOuter, Retrieve} | Count/CountEach/Retrieve 做前缀连续扫描（遇空槽终止）；Find 命中即返回被查询 Key |
| `MultisetReduceKernel` | 1 block | 归约各核局部计数与 `Size`，写 1 个标量 | 整数累加 ⇒ 与分块方式无关，返回值可复现 |

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
  未命中输出 0，`CountEachOuter` 未命中输出 1（`max(多重数, 1)`）；`Retrieve/RetrieveAll` 顺序不承诺，容量不足
  时按给定容量截断（返回值即实际条数）。
- `Count/Retrieve/RetrieveAll/Insert/InsertIf` 含 D2H 回读与流同步，`Contains/ContainsIf/Find/FindIf` 为纯异步；
  跨 stream 使用同一容器需调用方自行同步，容器不提供内部锁。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | I32/I64 的 `Insert/InsertIf` 状态变化，及 `Contains/ContainsIf/Find/FindIf/Count/CountEach/CountEachOuter/Retrieve/RetrieveAll` 输出与 cuCollections 语义一致；相同输入重复执行时状态与输出一致（`Retrieve/RetrieveAll` 的"一致"按集合语义判定，用例按排序后比对，顺序差异不计为不一致） | 任务书 3.2 |
| 性能标准 | 44 条性能用例全部满足 **算子时延 ≤ 标杆时延 / 0.4**（即 ≥0.4× 标杆性能）；不达标须给出合理解释 | 任务书 3.3 |
| 内存标准 | 除容器静态存储、输出空间与必要工作空间外，不产生与输入规模线性重复的额外 Device 内存拷贝 | 任务书 3.4 |

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

## 内存要求

容器静态存储为 `keys[C]` 单数组（I32 约 4·C 字节、I64 约 8·C 字节，C = 1e8 时约 400 MB / 800 MB），外加
O(核数) 局部计数缓冲；批量接口直接读写调用方缓冲、仅回拷 1 个标量，故不产生与输入规模线性重复的额外 Device
拷贝（任务书 3.4）。相比需同时存键与值的映射型容器，本容器省去 Value 数组。

## 测试设计

功能测试（`tests/static_multiset/`，1258 例，每个接口对应同名测试文件）：Create 14、Destroy 18、Clear 32、
Insert 194、InsertIf 20、Contains 100、ContainsIf 56、Find 100、FindIf 56、Count 124、CountEach 122、
CountEachOuter 122、Retrieve 98、RetrieveAll 202（dtype 覆盖 I32 与 I64；与任务书 3.5 一致）。用例按 dtype
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
