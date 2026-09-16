# static_multimap 容器开发设计文档（Atlas 950 / Ascend C）

> 容器：`aclco::StaticMultimap<Key, Value>`（静态容量多值键值映射，对标 cuCollections `cuco::static_multimap`）
> 工程模式：ops-collections 纯头文件容器 —— `include/static_multimap.h`、`include/static_multimap_ref.h`、`include/detail/static_multimap/`
> 硬件与环境：Atlas 950 系列（及后续支持 SIMT 能力的昇腾产品）；CANN ≥ 9.0.0-beta.2；CMake ≥ 3.16；Catch2 v3.5.4
> 测试：`tests/static_multimap/`（功能 1086 例）、`tests/performance/static_multimap/`（性能 32 例）

# 需求背景（required）

## 需求来源

社区任务"9 月社区任务-multimap 容器开发（950）"任务书：参考 cuCollections
`include/cuco/static_multimap.cuh`（dev 分支），在昇腾 NPU 上以 Ascend C 实现功能一致的静态多值键值映射
容器及相关算子，完成设计、开发、测试全流程，以 PR 合入 `https://gitcode.com/cann/ops-collections`；
设计文档按 `resources/design_template.md` 提交至 `cann-ops-competitions` 仓库评审。

## 背景介绍

### static_multimap 容器实现现状分析

ops-collections 既有约定：对外接口放 `include/<容器>.h`，设备侧引用放 `include/<容器>_ref.h`，实现细节放
`include/detail/<容器>/`；容器生命周期、参数校验与 ACL 流管理由 C++ 侧完成，设备侧批量访问与计算由 Ascend C
Kernel 承担；测试分别放 `tests/<容器>/`、`tests/performance/<容器>/`。仓库已有 StaticMap（单值映射）与
StaticSet（单值集合），**无多值语义实现**——StaticMap 对同一 Key 后写覆盖前写，本任务要求同一 Key 可关联多个
Value 且全部保留，故设计落点是在既有开放寻址骨架上引入**多值插入与多重数扫描**。

### static_multimap 功能分析

静态容量哈希表，无删除接口，容量创建时固定。Key/Value 均支持 I32、I64（验收场景同类型），同一 Key 可关联
多个 Value；`keys`/`values` 为一维 `[NumInputs]`（ND），`numInputs ∈ [0, 容量]`；插入值不得等于保留哨兵
（`numeric_limits<T>::lowest()`）；`Contains` 输出 1 字节标志、`Count` 以 uint64 标量返回；`If` 系列 stencil 为
uint32 `[NumInputs]`，谓词须 `COLLECTION_SIMT_DEVICE` 可调用；容量为 0、输出空间不足、stencil 长度不匹配、
空指针（数量 > 0）与非法流均须报错。入参顺序与 cuCollections 一致：区间 `[first, last)` 替换为「首地址 +
元素个数」；Key/Value 由模板参数编译期实例化，与用例 `TEMPLATE_TEST_CASE_SIG(..., (int32_t,int32_t,0),
(int64_t,int64_t,0))` 两实例对应，故"dtype 不一致"退化为编译期检查；键值对用 `aclco::Pair<Key, Value>`
（`pair.h`，成员 `first`/`second`）。

# 需求分析（required）

## 需求描述

在 Atlas 950 系列上实现 ops-collections 纯头文件 StaticMultimap：支持构造、析构、清空、Insert、InsertIf、
Contains、ContainsIf、Find、FindIf、Count、Retrieve、RetrieveAll；Key/Value 支持 I32、I64 且同类型；同一 Key
可关联多个 Value；查询、计数与检索结果与参考实现语义一致且可重复；覆盖空输入、重复 Key、不同 multiplicity、
命中/未命中与容量边界等泛化场景（测试落 `tests/static_multimap/`、`tests/performance/static_multimap/`）。

## 需求拆解

1. **工程与接口**：三头文件路径 + 仓库既有 CMake（≥3.16）/ Catch2 v3.5.4；12 个大驼峰接口；测试落
   `tests/static_multimap/`、`tests/performance/static_multimap/`，不新增独立算子目录。
2. **多值语义**：同 Key 重复插入不覆盖、全部保留；`Size` 为元素总数（含重复）；`Find` 返回该 Key 关联的任一
   Value；`Count` 返回多重数之和；`Retrieve` 返回全部命中配对。
3. **If 系列与错误**：stencil + Device 谓词决定元素是否参与运算；容量取整后 `Capacity() ≥ 请求值`，写满后
   `Insert` 返回失败计数，不覆盖、不搬移已有元素。
4. **性能与交付**：32 条性能用例满足 ≥0.4× 标杆；交付设计文档、代码与测试、自测报告、个人仓地址并更新 README。

# 详细设计（required）

## 接口清单

共 12 个接口，大驼峰命名。下表简记：`Extent` = `aclco::Extent<std::size_t>`，`stream` = `aclrtStream stream = nullptr`，
`SizeType` = 元素计数返回类型（uint64），`↔` 后为对标 cuCollections 的接口。

| # | 接口 | 声明 | 返回 | 语义要点（`↔` 后为对标 cuCollections 接口） |
| --- | --- | --- | --- | --- |
| 1 | Create | `StaticMultimap(Extent capacity, Key emptyKey, Value emptyValue, stream)` | 容器对象（RAII） | 合法容量分配 + 写哨兵；`capacity = 0` 抛异常；`Capacity() ≥ 请求值`、`Size() = 0`。↔ `static_multimap(capacity, empty_key_sentinel, empty_value_sentinel, ...)` |
| 2 | Destroy | `~StaticMultimap()` + 显式 `void Destroy()` + 移动构造/赋值 | void | 释放存储、置句柄无效；显式 Destroy 幂等。↔ 析构 |
| 3 | Clear | `void Clear(stream)` | void | 复位哨兵、计数器归零；空容器幂等；可复用。↔ `clear` |
| 4 | Insert | `SizeType Insert(void* values, Extent valueNum, stream)` | **失败数** | `Pair` 数组首地址；同 Key 重复不覆盖，写入探测链首个可用槽位；数量 0 空操作；`nullptr` 且数量 > 0 返回全部失败；`failed = valueNum − inserted`（口径与参考实现相反）。↔ `insert` |
| 5 | InsertIf | `template <typename StencilT, typename Pred> SizeType InsertIf(void* values, StencilT* stencil, Extent valueNum, stream)` | **失败数** | 仅 `pred(stencil[i])` 为真的元素插入；跳过的不计失败。↔ `insert_if` |
| 6 | Contains | `void Contains(void* keys, void* output, Extent keyNum, stream)` | void | 逐键写 1 字节标志（命中 1 / 未命中 0），每个输入位置都写；参考实现返回"是否全部命中"，此处改逐元素输出。↔ `contains` |
| 7 | ContainsIf | `template <typename StencilT, typename Pred> void ContainsIf(void* keys, StencilT* stencil, void* output, Extent keyNum, stream)` | void | 谓词为假输出 0。↔ `contains_if` |
| 8 | Find | `void Find(void* keys, void* values, Extent keyNum, stream)` | void | 命中写该 Key 关联**任一** Value（须真实存在）；未命中写 `emptyValue`。↔ `find` |
| 9 | FindIf | `template <typename StencilT, typename Pred> void FindIf(void* keys, StencilT* stencil, void* values, Extent keyNum, stream)` | void | 跳过与未命中均写 `emptyValue`。↔ `find_if` |
| 10 | Count | `SizeType Count(void* keys, Extent keyNum, stream)` | **多重数之和** | 逐查询键累加出现次数；空输入返回 0。↔ `count` |
| 11 | Retrieve | `SizeType Retrieve(void* keys, Extent keyNum, void* probeOut, void* matchOut, Extent matchCapacity, stream)` | **实际条数** | 输出 `(probeOut, matchOut)` 配对；条数 = min(命中总数, `matchCapacity`)。↔ `retrieve` |
| 12 | RetrieveAll | `SizeType RetrieveAll(void* keys, void* values, Extent capacity, stream)` | **实际条数** | 导出全部键值对（含重复 Key 每个 Value）；条数 = min(`Size()`, `capacity`)。↔ `retrieve_all` |

辅助接口（不计入 12 个）：`Size(stream)`、`Capacity()`；输出缓冲由调用方按 `numInputs` / `capacity` 预分配。

## 算子分析

### 数学公式与判定式

```
h1 = hash1(key) ; h2 = hash2(key) | 1               // h2 取奇数，与 2 的幂容量互质 ⇒ 探测可覆盖全表
slot(k) = (h1 + k · h2) & (C - 1),  k = 0 … C-1     // C = Capacity()，2 的幂
插入：写入探测链上首个「空槽」或「同 Key 且 value 槽为空」的槽位
命中判定：key 槽位 == 查询键
多重数：自 h1 起连续统计 key 槽位 == 查询键 的个数，遇首个空 key 槽位终止
Size = 已写入槽位总数（含重复 Key 的每个 Value）
```

两条由判定式导出的设计约束：

1. **查询可在首个空槽终止**：插入只把元素写进空槽或同 Key 的空值槽、从不产生空槽，故"已存元素的探测链前缀
   无空槽"这一不变式**单调保持**；未命中判定与多重数扫描只需扫连续前缀，期望代价 O(1 + 多重数)。
2. **多值需要"值槽判空"**：同 Key 的第 k 个 Value 需独立槽位，可用槽位判定为「key 为空哨兵」或「key 相同且
   value 为空哨兵」——这也是 Create 必须同时接收 emptyKey 与 emptyValue 的原因。

### 支持数据类型

Key/Value 为 int32 或 int64（模板实例化固定，验收同类型）。

### 支持形状

`keys`/`values` 为一维 `[NumInputs]`（ND）。输入为 `Pair<Key,Value>` 数组（AoS），内部按 **keys[C] + values[C]
双数组（SoA）** 存放，使查询只搬运 Key 槽位。

## 算子实现

### 工程结构与文件布局

```text
ops-collections/
├── include/static_multimap.h          # 对外接口（12 个）
├── include/static_multimap_ref.h      # 设备侧引用：Kernel 内可用的 device view / mutable view
├── include/detail/static_multimap/    # probing.h（双散列探测）、storage.h（双数组+计数器）
│                                      # kernels.h（init/insert/query/reduce）、predicates.h
├── tests/static_multimap/             # 功能用例（12 个接口各对应一个测试文件）
├── tests/performance/static_multimap/ # 性能用例（perf_create/destroy/insert/contains/find/
│                                      # retrieve/retrieve_all/count）
└── docs/static_multimap_API文档和使用示例.md
```

`static_multimap.h` 暴露容器类；`static_multimap_ref.h` 暴露设备侧视图（对标 cuCollections 的 `device_view` /
`device_mutable_view`）；`detail/` 不作为对外承诺。

### Host 侧设计

容器成员：默认 stream、请求容量与 `Capacity()`、`empty_key_sentinel`、`empty_value_sentinel`、Device 侧
`keys[C]`/`values[C]`、元素计数器、逐核局部计数缓冲。

| 校验项 | 触发条件 | 行为 |
| --- | --- | --- |
| 容量合法 | 请求容量 = 0 或超出 Device 内存可行上限 | 抛 `std::invalid_argument` / `std::bad_alloc` |
| 指针非空 | 载入类接口数量 > 0 但输入/输出指针为空（Insert 例外） | 抛 `std::invalid_argument` |
| 数量合法 | `numInputs > Capacity()`；`RetrieveAll` 输出容量为 0 而 `Size() > 0` | 抛 `std::invalid_argument` |
| 哨兵合规 | 插入的 Key/Value 等于对应哨兵 | 按"不得为保留空键值"拒绝（可判定项 Host 校验，Device 侧跳过） |
| 句柄/流有效 | `Destroy()` 之后调用接口；stream 非空但非本设备有效流 | 分别抛 `std::logic_error` / `std::runtime_error` |

关键路径：

1. **Create**：一次 `aclrtMalloc` 申请 `keys[C] + values[C]` → 一次 Kernel 顺序写哨兵 → 计数器清零。基线
   （I32/I32 7.635412 ms、I64/I64 15.419803 ms，约 2 倍关系）表明成本随数据量线性增长（C = 1e8 时约 800 MB /
   1.6 GB），必须**单次连续填充**，不得逐元素多次下发或 Host 侧循环写槽位。
2. **Insert / InsertIf**：沿调用方 stream 异步下发，每线程处理一个 `Pair`；失败按核局部计数，收尾归约为 1 个
   标量回读（8B D2H + 一次同步）后作为返回值；数量为 0 直接返回 0。
3. **Contains / ContainsIf / Find / FindIf**：纯异步无返回值，逐元素写 1 个标志或 Value，不引入同步。
4. **Count / Retrieve / RetrieveAll**：同样「Device 侧归约 → 8B D2H → 一次同步」；检索写入在 Device 侧单线程
   收尾，保证条数精确且不越界写（上限为调用方给定的 `matchCapacity` / `capacity`）。
5. **Clear / Destroy**：单次 Kernel 复位哨兵 + 计数器清零；Destroy 直接释放存储并置空句柄，不引入额外流同步。

### Kernel 侧设计（Ascend C / SIMT）

采用 SIMT 编程模型（每 block 256 线程，`__simt_vf__` 设备函数 + `asc_vf_call` 由核函数调用），纯 GM 直访、
不依赖 UB，因此**无 tiling 结构**，核函数仅接收标量参数（keys/values 首地址、C、数量、两个哨兵）。

| 核函数 | 网格 | 每线程工作 | 关键点 |
| --- | --- | --- | --- |
| `MultimapInitKernel` | useNumBlocks | grid-stride 写哨兵、计数器清零 | 单次顺序写 |
| `MultimapInsertKernel<Key,Value>` | useNumBlocks | 探测 → CAS 占用 key 槽位 → 写 value 槽位；失败局部计数 | 原子 CAS 保证单一写入者，失败者继续探测下一位置（不重试同一槽位） |
| `MultimapQueryKernel<Key,Value,Op>` | useNumBlocks | 逐元素探测；Op ∈ {Contains, Find, Count, Retrieve} | Count/Retrieve 做前缀连续扫描（遇空槽终止）；Find 命中即返回首个匹配 Value |
| `MultimapReduceKernel` | 1 block | 归约各核局部计数与 `Size`，写 1 个标量 | 整数累加 ⇒ 与分块方式无关，返回值可复现 |

`If` 系列复用同一套核函数，仅增加 stencil 入参与谓词模板参数，谓词形如
`COLLECTION_SIMT_DEVICE bool operator()(StencilT) const noexcept`（与用例 `IsOdd` 一致），谓词为假直接跳过。
`Retrieve/RetrieveAll` 顺序不承诺但条数与内容集合必须精确（用例按排序后比对）；keys/values 双数组按 128B 对齐。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas 950 系列（DAV_3510，支持 SIMT）及后续支持 SIMT 能力的昇腾系列产品 | √ |

## 算子约束限制

- 仅支持 (int32,int32)、(int64,int64) 两种模板实例；其它组合不承诺。
- 无删除接口，容量创建后固定；写满后 `Insert/InsertIf` 返回失败计数，已存元素不被覆盖或搬移。
- 查询键与插入的 Key/Value 不得等于对应哨兵值，等于哨兵按无效处理。
- `Find/FindIf` 只返回**一个** Value（属该 Key 的 Value 集合），需全部匹配值时用 `Retrieve`；`Count` 返回多重数
  之和而非去重键数；`Retrieve/RetrieveAll` 顺序不承诺，容量不足时按给定容量截断（返回值即实际条数）。
- `Count/Retrieve/RetrieveAll/Insert/InsertIf` 含 D2H 回读与流同步，`Contains/ContainsIf/Find/FindIf` 为纯异步；
  跨 stream 使用同一容器需调用方自行同步，容器不提供内部锁。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | I32/I64 的 `Insert/InsertIf` 状态变化，及 `Contains/ContainsIf/Find/FindIf/Count/Retrieve/RetrieveAll` 输出与 cuCollections 语义一致；相同输入重复执行时状态与输出一致（`Retrieve/RetrieveAll` 的"一致"按集合语义判定，用例按排序后比对，顺序差异不计为不一致） | 任务书 3.2 |
| 性能标准 | 32 条性能用例全部满足 **算子时延 ≤ 标杆时延 / 0.4**（即 ≥0.4× 标杆性能）；不达标须给出合理解释 | 任务书 3.3 |
| 内存标准 | 除容器静态存储、输出空间与必要工作空间外，不产生与输入规模线性重复的额外 Device 内存拷贝 | 任务书 3.4 |

性能基准（32 条用例逐行对应基线表）：

| 接口 | 条数 | 标杆时延区间（ms） | 时延上限 = 标杆 / 0.4（ms） |
| --- | --- | --- | --- |
| create / destroy | 2 + 2 | 7.635412 – 15.419803 / 9.038761 – 17.766071 | 19.088530 – 38.549508 / 22.596903 – 44.415178 |
| insert | 2 | 9.984867 – 13.994128 | 24.962168 – 34.985320 |
| retrieve_all | 2 | 1.469109 – 3.421753 | 3.672773 – 8.554383 |
| contains / find | 6 + 6 | 6.272652 – 9.062888 / 7.065370 – 9.708089 | 15.681630 – 22.657220 / 17.663425 – 24.270223 |
| retrieve / count | 6 + 6 | 12.055540 – 16.426197 / 5.868191 – 8.985895 | 30.138850 – 41.065493 / 14.670478 – 22.464738 |

表中除 create/destroy 外均为 `Distribution = UNIFORM`、`NumInputs = 100000000`、`Occupancy = 0.5`、
`Multiplicity = 1`，查询类接口扫描 `MatchingRate = 0.1/0.5/1`；create/destroy 为 `Capacity = 100000000`；
dtype 均含 I32/I32 与 I64/I64。

代表性逐行标杆（括号内为上限 = 标杆 / 0.4）：create I32/I32 7.635412 ms（19.088530 ms）、insert I32/I32
9.984867 ms（24.962168 ms）、count I64/I64 8.985895 ms（22.464738 ms）。

## 内存要求

容器静态存储为 `keys[C] + values[C]` 双数组（I32 对约 8·C 字节、I64 对约 16·C 字节，C = 1e8 时约 800 MB /
1.6 GB），外加 O(核数) 局部计数缓冲；批量接口直接读写调用方缓冲、仅回拷 1 个标量，故不产生与输入规模线性
重复的额外 Device 拷贝（任务书 3.4）。

## 测试设计

功能测试（`tests/static_multimap/`，1086 例，每个接口对应同名测试文件）：Create 14、Destroy 18、Clear 32、
Insert 290、InsertIf 20、Contains 100、ContainsIf 56、Find 100、FindIf 56、Count 100、Retrieve 98、
RetrieveAll 202（dtype 覆盖 I32/I32 与 I64/I64；与任务书 3.5 一致）。用例按 dtype 实例、`GENERATE` 参数组合与
`SECTION` 分支展开，覆盖容量 1…1000000 × occupancy 0/0.1/0.5/0.9/1 × multiplicity 1/2/4/8 ×
MatchingRate 0/0.1/0.5/1，以及空输入、重复 Key/重复配对、`nullptr` 全失败、容量边界失败数、Clear 后复用、
If 系列三种 stencil 模式、Create 容量 0 拒绝、未命中写哨兵与 Retrieve 条数精确。

性能测试（`tests/performance/static_multimap/`，32 例）：`perf_create / perf_destroy / perf_insert /
perf_contains / perf_find / perf_count / perf_retrieve / perf_retrieve_all`，经 `REGISTER_PERFORMANCE_TEST` /
`REGISTER_PERFORMANCE_ARGS` 注册，逐行对应 3.3 基线表。

## 自验要求

1. 用任务提供的 `test-cases/` 全量执行功能与性能用例，提交完整日志（入参、结果对比、性能数据）。
2. 性能自验给出 32 条用例的算子时延 vs 标杆时延、倍率与逐条达标结论，未达标项给出成因与优化方向；环境为
   CANN ≥ 9.0.0-beta.2 + Atlas 950 系列设备 + CMake ≥ 3.16 + Catch2 v3.5.4。

## 兼容性分析

- **新增容器**：复用既有探测骨架但不修改 StaticMap/StaticSet 的接口与行为，新增
  `include/static_multimap.h` 不影响既有容器编译与 ABI；仅实例化 (I32,I32)、(I64,I64)。
- **与 cuCollections 的差异**：迭代器区间 → 首地址 + 个数；`Insert/InsertIf` 返回失败数而非成功数；
  `Contains/ContainsIf/Find/FindIf` 改为逐元素输出（批量接口无返回标量的位置）。
- **硬件与构建兼容**：仅依赖 GM 直访与 SIMT 线程模型，不含 950 专有指令；沿用仓库 CMake（≥3.16）与 Catch2
  v3.5.4，测试目录符合 `tests/<容器>/`、`tests/performance/<容器>/` 约定。
