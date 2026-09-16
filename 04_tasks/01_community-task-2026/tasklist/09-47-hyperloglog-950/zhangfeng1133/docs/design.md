# hyperloglog 容器开发设计文档（Atlas 950 / Ascend C）

> 容器：`aclco::HyperLogLog<Key>`（HyperLogLog 基数估算，对标 cuCollections `cuco::hyperloglog`）
> 工程模式：ops-collections 纯头文件容器 —— `include/hyperloglog.h`、`include/hyperloglog_ref.h`、`include/detail/hyperloglog/`
> 硬件与环境：Atlas 950 系列（及后续支持 SIMT 能力的昇腾产品）；CANN ≥ 9.0.0-beta.2；CMake ≥ 3.16；Catch2 v3.5.4
> 测试：`tests/hyperloglog/`（功能 1280 例）、`tests/performance/hyperloglog/`（性能 72 例）

# 需求背景（required）

## 需求来源

社区任务"9 月社区任务-hyperloglog 容器开发（950）"任务书：参考 cuCollections
`include/cuco/hyperloglog.cuh`（dev 分支），在昇腾 NPU 上以 Ascend C 实现功能一致的 HyperLogLog 容器及相关
算子，完成设计、开发、测试全流程，以 PR 合入 `https://gitcode.com/cann/ops-collections`；设计文档按
`resources/design_template.md` 提交至 `cann-ops-competitions` 仓库评审。

## 背景介绍

### HyperLogLog 容器实现现状分析

ops-collections 既有约定：对外接口放 `include/<容器>.h`，设备侧引用放 `include/<容器>_ref.h`，实现细节放
`include/detail/<容器>/`；容器生命周期、参数校验与 ACL 流管理由 C++ 侧完成，设备侧批量访问与计算由 Ascend C
Kernel 承担；功能与性能测试分别放 `tests/<容器>/`、`tests/performance/<容器>/`。**仓库现无任何 HyperLogLog 实现**，属全新容器，无历史实现迁移问题。其"算子"载体是
`Clear / Add / Merge / Estimate` 四个状态操作加 `Create / Destroy` 生命周期操作（有状态定长 Sketch），
设计落点为「定长寄存器的设备侧批量更新 + 确定性归约估算」。

### HyperLogLog 功能分析

以 m 个 8 位寄存器记录输入元素哈希的最大前导零秩，用调和平均估计去重基数：

| 参数 | 含义 | 输入/输出/属性 | dtype 与形状 | 值域约束与异常行为 |
| --- | --- | --- | --- | --- |
| hll | 容器句柄 | 输入/输出（Create 输出） | handle，标量 | 非空、已初始化；空句柄报错 |
| keys | 待计数元素数组首地址 | 输入 | I32、I64，ND `[NumInputs]` | `keyNum>0` 时指针须非空，否则报错 |
| sketchSizeKb | Sketch 容量 | 属性 | uint32，标量 | 8/16/32/64/128/256 KB；非法容量报错 |
| otherHll | Merge 的另一容器 | 输入 | handle，标量 | 与 hll 类型和 Sketch 配置兼容；不兼容报错 |
| numInputs | 本次处理元素数量 | 属性 | uint64，标量 | 非负整数；超范围报错 |
| estimate | 估算基数 | 输出 | uint64，标量 | 非负整数 |
| stream | ACL 执行流 | 输入 | aclrtStream，标量 | 有效 ACL 流（`nullptr` 为默认流）；非法流报错 |

入参顺序与 cuCollections 一致：device iterator 区间 `[first, last)` 统一替换为「首地址 + 元素个数」
（`void* keys, Extent keyNum`），`cuda::stream_ref` 替换为 `aclrtStream`，参数含义不变。dtype 由模板参数编译期
实例化（`HyperLogLog<int32_t>` / `HyperLogLog<int64_t>`），与任务用例 `TEMPLATE_TEST_CASE_SIG` 的 I32/I64 两实例
一一对应，运行期不再校验 dtype。

# 需求分析（required）

## 需求描述

在 Atlas 950 系列上以 Ascend C + C++ 实现 ops-collections 纯头文件 HyperLogLog 容器：支持按 SketchSizeKB /
标准差 / 精度三种方式构造，支持 Destroy、Clear、Add、Merge、Estimate；Key 支持 I32、I64；覆盖空输入、重复
输入、清空后重用、不同 SketchSizeKB、合法合并等泛化场景。

## 需求拆解

1. **工程与接口**：三头文件路径 + 仓库既有 CMake（≥3.16）/ Catch2 v3.5.4；6 个接口（三种 Create 构造入口、
   Destroy、Clear、Add、Merge、Estimate），不新增独立算子目录。
2. **算法与生命周期**：批量哈希 → 寄存器原子 max；逐寄存器 max 合并 + 配置兼容校验；整数化归约求 Estimate，
   空 Sketch 返回 0 且不改状态；RAII + 移动语义，Destroy 幂等。
3. **精度、性能与交付**：相对误差 ≤ 3×1.04/√m，分批与一次性 Add 逐位一致；72 条性能用例满足 ≥0.4× 标杆；
   交付设计文档、代码与测试、自测报告、个人仓地址并更新 README。

# 详细设计（required）

## 接口清单

对外接口共 6 个，大驼峰命名，`stream` 默认为 `nullptr`。

| # | 接口 | 声明 | 返回/生效方式 | 语义要点与对标 cuCollections |
| --- | --- | --- | --- | --- |
| 1 | Create | `CreateWithSketchSizeKB(uint32_t, aclrtStream)`；`CreateWithStandardDeviation(double, aclrtStream)`；`CreateWithPrecision(uint32_t, aclrtStream)`（static 工厂） | 返回容器对象（RAII） | 三种构造；非法值抛 `std::invalid_argument`；新容器全零、Estimate 为 0。对标按 `sketch_size_bytes` 构造 + `set_sketch_size_bytes`，反向使用 `standard_deviation()` / `estimate_error(precision)` |
| 2 | Destroy | `~HyperLogLog()` + 显式 `void Destroy()` + 移动构造/移动赋值 | void | 释放 Device 寄存器存储、置句柄无效；显式 `Destroy()` 幂等。参考实现以析构承担，本任务增幂等显式入口 |
| 3 | Clear | `void Clear(aclrtStream stream = nullptr)` | void | 全部寄存器归零；空 Sketch 上幂等；Clear 后可继续 Add 复用。对标 `clear(stream)` |
| 4 | Add | `void Add(void* keys, aclco::Extent<std::size_t> keyNum, aclrtStream stream = nullptr)` | void，经容器状态生效 | 哈希 → 寄存器 max；`keyNum = 0` 为合法空操作。对标 `add(first, last, stream)`，区间 → 首地址 + 个数 |
| 5 | Merge | `void Merge(const HyperLogLog& otherHll, aclrtStream stream = nullptr)` | void | 逐寄存器 max 原地合并；配置不兼容抛异常；自合并安全。对标 `merge(other, stream)` |
| 6 | Estimate | `uint64_t Estimate(aclrtStream stream = nullptr) const` | 返回 uint64 | 纯读、不改状态；空 Sketch 返回 0。对标 `estimate(stream)` → `size_type`，即任务书参数表的 `estimate` 输出 |

辅助查询（不计入 6 个接口）：`SketchSizeKB()`、`RegisterCount()`、`Precision()`。

## 算子分析
### 数学公式

```
h    = hash64(key)                          // Key 的 64 位位模式全参与
idx  = h >> (64 - p)                        // p = log2(m) ∈ [13,18]
w    = h << p ; rank = (w == 0) ? (64 - p + 1) : (clz(w) + 1)
M[idx] = max(M[idx], rank)                  // 交换、结合，与处理顺序无关
E = alpha_m · m² / Σ 2^(-M[j]),  alpha_m = 0.7213 / (1 + 1.079/m)
小值域：E ≤ 2.5m 且 V > 0 时 E = m · ln(m / V)      // V 为零寄存器个数
大值域：E > 2^64 / 30 时 E = -2^64 · ln(1 - E/2^64)  // 本任务 NumInputs ≤ 1e8，远低于阈值，不触发
Estimate = round_half_up(E)
```

三条由公式导出的设计约束：

1. **小值域修正是必选路径**：全零时 V = m，未修正的 E ≈ 0.72m ≠ 0，只有 `m·ln(m/V) = 0` 满足用例
   "新建容器 / Clear 后 `Estimate == 0`"。
2. **Σ2^(-M[j]) 必须整数累加**：取 `maxM = max_j M[j]`，则 `Σ2^(-M[j]) = (Σ 2^(maxM-M[j])) · 2^(-maxM)`，括号内为
   64 位整数，与并行归约分块无关 ⇒ Estimate 逐位可复现（浮点累加会因求和顺序漂移）。
3. **rank 上界 64-p+1**（p = 13 时为 52），8 位寄存器无损容纳。

### 支持数据类型

Key 支持 int32、int64（模板实例化固定；int32 符号扩展后按 64 位位模式哈希，避免与全 1 位模式混淆）；
寄存器与存储为 uint8 × m，1 字节/寄存器，m = SketchSizeKB × 1024 ⇒ 存储恰为 SketchSizeKB KB。

### 支持形状

`keys` 为一维 `[NumInputs]`（ND），覆盖 0/1/2/3/7/31/128/1024/4096/8195/32768/131072 与 1 亿（性能档），长度与容量
用 `aclco::Extent<std::size_t>`；非 256 倍数长度由 grid-stride 尾块覆盖。

档位与精度指标（同时是标准差异构造的反解表）：

| SketchSizeKB | m | p | σ = 1.04/√m | 误差上界 3σ |
| --- | --- | --- | --- | --- |
| 8 | 8192 | 13 | 1.15% | 3.45% |
| 16 | 16384 | 14 | 0.81% | 2.44% |
| 32 | 32768 | 15 | 0.57% | 1.72% |
| 64 | 65536 | 16 | 0.41% | 1.22% |
| 128 | 131072 | 17 | 0.29% | 0.86% |
| 256 | 262144 | 18 | 0.20% | 0.61% |

按标准差构造的吸附规则：取满足「σ_actual ≤ 请求 σ」的**最小档位**，用例给出的六个待测标准差恰为上表六档 σ
向上取整到 1e-4，故可精确还原六个档位。

## 算子实现

### 工程结构与文件布局

```text
ops-collections/
├── include/hyperloglog.h              # 对外接口（Create/Destroy/Clear/Add/Merge/Estimate）
├── include/hyperloglog_ref.h          # 设备侧引用：Kernel 内可调用的 Sketch 视图（更新/归约）
├── include/detail/hyperloglog/        # hll_config.h / hll_hash.h / hll_sketch.h / hll_kernels.h
├── tests/hyperloglog/                 # 功能用例（create/clear/add/merge/estimate/destroy）
├── tests/performance/hyperloglog/     # 性能用例（perf_*）
└── docs/hyperloglog_API文档和使用示例.md
```

`hyperloglog.h` 只暴露容器类与工厂；`hyperloglog_ref.h` 暴露设备侧 Sketch 视图（对标 cuCollections 的
device view / device mutable view）；`detail/` 不作为对外承诺。

### Host 侧设计

容器成员：默认 stream、`Extent<std::size_t> register_count_`、`precision_`、Sketch 容量（KB）、哈希种子、
Device 寄存器指针（m 字节）、配置指纹（容量 + 精度 + 种子，用于 Merge 兼容判定）；Sketch 存储即 m 字节，
无额外元数据数组。

| 校验项 | 触发条件 | 行为 |
| --- | --- | --- |
| 配置合法 | 容量 ∉ {8,16,32,64,128,256}（含 0、7、512）；precision ∉ [13,18]（含 0）；σ ≤ 0 或非有限 | 抛 `std::invalid_argument` |
| 指针与句柄 | `Add` 中 `keyNum > 0` 且 `keys == nullptr`；或 `Destroy()` 之后调用任何操作 | 前者抛 `std::invalid_argument`，后者抛 `std::logic_error` |
| 合并与流 | Merge 时另一容器容量/精度/种子不一致；stream 非空但非本设备有效流 | 前者抛 `std::invalid_argument`，后者包装为 `std::runtime_error` |

错误口径与仓库既有容器一致：配置与参数类错误抛 C++ 异常（用例以 `REQUIRE_THROWS` 校验）；设备侧运行期失败
（malloc/launch）抛 `std::runtime_error` / `std::bad_alloc`。

关键路径：

1. **Create**：一次 `aclrtMalloc` 申请 m 字节 → 一次 `HllClearKernel`（或 `aclrtMemsetAsync`）异步清零 → 记录
   默认流；基线 create 时延在 8…256 KB 间仅 0.4035–0.5142 ms、与容量弱相关 ⇒ 须避免多次 malloc 与多次下发。
2. **Add**：`keyNum = 0` 直接返回；否则沿调用方 stream 异步下发，`useNumBlocks = min(coreNum, ceil(keyNum /
   256))`（coreNum 取平台 AIV 核数），核内 grid-stride 覆盖尾块。
3. **Merge / Clear / Estimate**：Merge 原地逐寄存器 max、不申请额外存储；Clear 为单次 Kernel 清零 m 字节；
   Estimate 为 Device 侧并行归约 + `aclrtMemcpyAsync`(8B D2H) + 一次同步（基线 0.0371–0.1157 ms，m 增 32 倍
   时延仅增 3.1 倍 ⇒ 归约主体必须在 Device 侧完成）。三者均不回拷中间数组、不引入额外内存申请。

### Kernel 侧设计（Ascend C / SIMT）

采用 SIMT 编程模型（每 block 256 线程，`__simt_vf__` 设备函数 + `asc_vf_call` 由核函数调用），纯 GM 直访、
不依赖 UB，因此**无 tiling 结构**，核函数仅接收标量参数（寄存器首地址、m、keys 首地址、count、种子）。
哈希取高 p 位做寄存器索引（与 rank 位段不重叠）；寄存器 1 字节紧凑、128B 对齐；并发更新用原子 max，其
交换结合性与归约整数化共同构成**确定性来源**。

| 核函数 | 网格 | 每线程工作 | 关键点 |
| --- | --- | --- | --- |
| `HllClearKernel` | useNumBlocks | grid-stride 写 0 | 单次顺序写，避免逐元素多次小写 |
| `HllAddKernel<Key>` | useNumBlocks | 取一个 key → 哈希 → 算 idx/rank → 对 `M[idx]` 原子 max | 必须**单指令原子 max**，禁止"读-改-写"非原子组合 |
| `HllMergeKernel` | useNumBlocks | grid-stride 做 max(dst, src) | 原地安全；自合并幂等 |
| `HllReduceKernel` | 1 block | 归约 `Σ2^(maxM-M[j])`、`V`、`maxM` → 收尾算出 Estimate 写 1 个 uint64 | 整数累加保证与分块无关 |

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas 950 系列（DAV_3510，支持 SIMT）及后续支持 SIMT 能力的昇腾系列产品 | √ |

## 算子约束限制

- 仅支持 Key = int32 / int64 两种模板实例；其它类型不承诺（编译期静态断言提示）。
- SketchSizeKB 仅 8/16/32/64/128/256 六档，precision 仅 13…18，其余值在 Create 阶段拒绝。
- Merge 双方必须容量、精度、哈希种子一致，否则抛异常而非静默取小值。
- `Add` 为异步 void 接口，跨 stream 使用须由调用方自行同步；`Estimate` 会同步 stream，不可高频循环调用；
  `keyNum > 0` 且 `keys == nullptr` 视为参数错误，`keyNum = 0` 时 `keys` 允许为 `nullptr`。
- 近似算法：不保证精确基数，仅保证 3σ 误差上界与同输入可重复；不同 Key 类型实例之间结果不可比。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | `\|Estimate − N\| / N ≤ 3 × 1.04 / √m`（N 为精确基数）；N = 0 时 Estimate 必须为 0；Clear/Add/Merge 状态变化与 cuCollections 一致；结果可重复且 Estimate 不改状态 | 任务书 3.2 |
| 性能标准 | 72 条性能用例全部满足 **算子时延 ≤ 标杆时延 / 0.4**（即 ≥0.4× 标杆性能）；不达标须给出合理解释 | 任务书 3.3 |
| 内存标准 | 除容器静态存储、输出空间与必要工作空间外，不产生与输入规模线性重复的额外 Device 内存拷贝 | 任务书 3.4 |

性能基准（72 条性能用例与基线表逐行对应：Create/Destroy/Clear 36 条、Estimate/Merge 24 条、Add 12 条）：

| 接口 | 参数组合 | 标杆时延区间（ms） | 时延上限 = 标杆 / 0.4（ms） |
| --- | --- | --- | --- |
| create | 8…256 KB × I32/I64（12 条） | 0.403518 – 0.514238 | 1.008795 – 1.285595 |
| destroy | 8…256 KB × I32/I64（12 条） | 0.442138 – 0.487931 | 1.105345 – 1.219828 |
| clear | 8…256 KB × I32/I64（12 条） | 0.029108 – 0.033233 | 0.072770 – 0.083083 |
| estimate | NumInputs = 1e8，8…256 KB（12 条） | 0.037083 – 0.115725 | 0.092708 – 0.289313 |
| merge | NumInputs = 1e8，8…256 KB（12 条） | 0.030822 – 0.057901 | 0.077055 – 0.144753 |
| add | UNIFORM、NumInputs = 1e8、Multiplicity = 1、8…256 KB（12 条） | 0.316691 – 2.138625 | 0.791728 – 5.346563 |

代表性逐行标杆（create 档，完整逐行数据以任务书 3.3 为准）：I32/8 KB 0.509452 ms（上限 1.273630 ms）、
I32/256 KB 0.447700 ms（1.119250 ms）、I64/8 KB 0.510325 ms（1.275813 ms）、I64/256 KB 0.403518 ms
（1.008795 ms）。由基线反推（非实测结论）：Add 时延对容量极敏感（I32 8 KB 0.3167 → I32 256 KB 2.0314 ms），
瓶颈是随机寄存器访问局部性；destroy 基线高于 create，析构应直接释放存储、不做遍历与额外同步。

## 内存要求

容器静态存储为寄存器数组 m 字节（= SketchSizeKB KB，8…256 KB），无影子数组与逐档冗余；工作空间为
O(核数) 级、不随 `NumInputs` 增长；Add 直接读调用方 Device 缓冲、Estimate 仅回拷 1 个 uint64，因而不产生与
输入规模线性重复的额外 Device 拷贝（任务书 3.4）。

## 测试设计

功能测试（`tests/hyperloglog/`，1280 例）按 dtype 实例、`GENERATE` 参数组合与 `SECTION` 分支展开统计：

| 接口 | 文件 | 用例数 | 覆盖维度 |
| --- | --- | --- | --- |
| Create | `create_test.cpp` | 38 | 三组六档取值（容量/标准差/精度）+ 非法配置（0、7、512、precision=0） |
| Destroy | `destroy_test.cpp` | 24 | 8 轮 create→add→destroy；移动构造转移所有权后 Estimate 仍正确 |
| Clear | `clear_test.cpp` | 180 | 空容器重复 Clear；有数据后 Clear；Clear 后复用 |
| Add | `add_test.cpp` | 726 | 12 档 keyNum × 3 档 distinct + 分批等价性 + 空指针拒绝 + 1 亿大基数 |
| Merge | `merge_test.cpp` | 192 | 空∩空、空∩非空、自合并幂等、重叠/不相交、容量不兼容拒绝 |
| Estimate | `estimate_test.cpp` | 120 | 基数 0…1048576；两容器逐位一致 + 误差上界 + 重复 Estimate 不改状态 |
| 合计 | — | 1280 | dtype 均为 I32、I64；与任务书 3.5 功能用例表一致 |

性能测试（`tests/performance/hyperloglog/`，72 例）：`perf_create / perf_clear / perf_add / perf_merge /
perf_estimate / perf_destroy`，经 `REGISTER_PERFORMANCE_TEST` / `REGISTER_PERFORMANCE_ARGS` 注册，逐行对应
3.3 基线表；用例驱动出的强约束（`keyNum = 0` 成功且不改状态、`Add(nullptr, 1)` 抛异常、分批与一次性 Add 的
Estimate 逐位相等、Clear 幂等、Merge 不兼容必抛）已体现在上文设计中。

## 自验要求

1. 用任务提供的 `test-cases/` 全量执行功能与性能用例，提交完整日志（入参、结果对比、性能数据）；功能自验
   覆盖六档容量 × 三种构造、空/重复/清空复用、合并兼容与不兼容、大基数 1e8、非对齐长度（8195、65537）。
2. 性能自验给出 72 条用例的算子时延 vs 标杆时延、倍率与逐条达标结论，未达标项给出成因与优化方向；环境为
   CANN ≥ 9.0.0-beta.2 + Atlas 950 系列设备 + CMake ≥ 3.16 + Catch2 v3.5.4。

## 兼容性分析

- **新增容器，无历史契约**：不影响既有 BloomFilter / StaticMap / StaticSet 的编译与 ABI；仅实例化 I32、I64。
  与 cuCollections 的差异收敛为两点——迭代器区间 → 首地址 + 个数，新增三种构造入口与幂等 `Destroy`；参数
  顺序、语义与返回类型含义不变，便于后续对拍。
- **硬件与构建兼容**：仅依赖 GM 直访与 SIMT 线程模型，不含 950 专有指令，后续支持 SIMT 的昇腾产品可复用
  （非 SIMT 平台不在本任务范围）；沿用仓库 CMake（≥3.16）与 Catch2 v3.5.4，测试目录符合
  `tests/<容器>/`、`tests/performance/<容器>/` 约定。
