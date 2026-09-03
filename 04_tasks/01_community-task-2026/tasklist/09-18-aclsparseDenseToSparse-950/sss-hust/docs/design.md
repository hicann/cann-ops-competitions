# aclsparseDenseToSparse 算子设计文档（Ascend 950PR / A5）

> 交付件①。按 `resources/design_template.md` 组织，四个必需一级章节齐全。
> 标注 **[待确认]** 的条目需要评审确认后才能定稿，逐条列在文末「希望检视人员了解」。

---

# 需求背景（required）

## 需求来源

CANN 生态 9 月社区任务「aclsparseDenseToSparse 算子开发(950)」A5 档。目标是在
`ops-sparse` 仓的 Ascend 950（DAV_3510，`arch35`）实现上，把稠密矩阵到稀疏矩阵的
三阶段转换能力补齐到任务书规格：补 complex64、补 Blocked-ELL 非整除尾块、
新增 Python/ATen 适配、优化性能并完善测试与文档。

代码交付至 `https://gitcode.com/cann/ops-sparse` 的 `master`；设计文档交付至
`https://gitcode.com/cann/cann-ops-competitions`。

## 背景介绍

### 与上一个任务的根本差别：这不是数值计算算子

`DenseToSparse` 做两件事：**发现结构**（哪些位置非零、有多少个）与**搬运位**
（把非零元素的原始 bit 模式写到目标格式的 values 里）。它不做任何算术，
因此任务书 §3.2 要求的是 **exact match / bit-wise 一致**，而不是混合容差。

这条决定了整套设计的重心：**没有精度误差可言，风险全在结构语义的边界与确定性上**。
正负零是否入结构、INF/NAN 是否保留、base 0/1 的 offsets 端点、尾块 padding 写什么、
重复执行是否逐位可复现——这些是本算子的「精度」。

### `ops-sparse` 现状分析

`sparse/densetosparse/arch35/` 已有实现，规模如下（基于 `origin/master` 实测）：

| 文件 | 行数 | 职责 |
|---|---|---|
| `densetosparse_host.cpp` | 658 | 三个公开接口、参数校验、workspace 布局、tiling、kernel 启动 |
| `densetosparse_kernel.cpp` | 622 | Analysis / Convert 两个 SIMT kernel |
| `densetosparse_kernel.h` | 30 | 两个 `*_kernel_do` 启动声明 |
| `densetosparse_tiling_data.h` | 40 | `DenseToSparseTilingData` |
| `test/densetosparse/arch35/densetosparse_test.cpp` | 1274 | C++ UT |
| `test/densetosparse/densetosparse_golden.h` | 300 | CPU Golden |

已具备的能力（依据 `sparse/densetosparse/README.md` 与源码核对）：

| 维度 | 现状 |
|---|---|
| values dtype | INT8 / FP16 / BF16 / FP32 |
| 目标格式 | CSR / CSC / COO / Blocked-ELL |
| Dense layout | `ACL_SPARSE_ORDER_ROW`（`ld≥n`）与 `ACL_SPARSE_ORDER_COL`（`ld≥m`） |
| index base | 0 / 1 |
| 索引类型 | I32 / I64；CSR/CSC 的 offset 与 element index 必须同型，混合组合返回 `NOT_SUPPORTED` |
| 零语义 | `±0` 均不入结构；NaN（含 payload）、`±Inf` 视为非零，按位搬运 |
| 边界 | 零维、全零、全非零已支持 |
| 三阶段 | `GetBufferSize`（纯查询，不记状态）→ `Analysis`（写 offsets、更新 `nnz`）→ `Convert`（写 indices/values） |

内核已把非零判定写成掩掉符号位后看剩余位的形式，这一点对本任务很关键（见详细设计）：

```cpp
template <typename T> inline bool IsNonzero(T bits)       { return bits != 0; }        // int8
template <> inline bool IsNonzero<uint16_t>(uint16_t bits) { return (bits & 0x7FFFU) != 0; }      // fp16/bf16
template <> inline bool IsNonzero<uint32_t>(uint32_t bits) { return (bits & 0x7FFFFFFFU) != 0; }  // fp32
```

### 三个功能缺口

**缺口一：complex64 未支持。** `densetosparse_host.cpp` 的 `GetElementBytes()` 只识别
1/2/4 字节，其余返回 0；调用处对 `matA`、`matB` 的 values 类型各判一次，
拿到 0 就返回 `NOT_SUPPORTED`。README 亦明确「不支持 float64 和任何复数类型」。注意描述符层已经放行——`sparse/common/aclsparse_descr.cpp` 的
`IsValidSparseValueType()` 已包含 `ACL_COMPLEX64`，所以创建 complex64 描述符本就可行，
**缺口只在算子本体**，这是一个边界清晰、影响面可控的改动。

**缺口二：Blocked-ELL 不支持非整除尾块。** 现有 host 在
`densetosparse_host.cpp:132` 明确拒绝：

```cpp
if (sparse->rows % sparse->ellBlockSize != 0 ||
    sparse->cols % sparse->ellBlockSize != 0 ||
    sparse->ellCols % sparse->ellBlockSize != 0) { /* INVALID_VALUE */ }
```

而任务书 §2.4 要求「Blocked-ELL 支持 rows、cols、ellCols 非 blockSize 整除的尾块，
并对越界逻辑元素写正零」。这是**语义扩展而非参数放宽**，会连带影响 BELL 的容量公式与
`nnz` 定义，是本任务技术风险最高的一项（见详细设计与 [待确认] 第 3 条）。

**缺口三：Python/ATen 适配完全缺失。** 全仓没有任何 `torch` / `aten` / `python` 目录，
而任务书把它列为**必选交付**：四个公开入口 `Tensor.to_sparse/to_sparse_csr/
to_sparse_csc/to_sparse_bsr` 映射到 `aten::_to_sparse/_to_sparse_csr/_to_sparse_csc/
_to_sparse_bsr`，需在 NPU 完成注册且不得 CPU fallback。

### 功能分析：三阶段契约决定了两遍扫描

`Analysis` 必须先产出 `nnz`，调用方据此分配 payload 并 `SetPointers`，`Convert` 才能写。
`nnz` 要更新到 Host 描述符，所以 **Analysis 内必然有一次 Device→Host 回读同步**；
而 Convert 时 payload 才存在，所以 **Dense 必然被读两遍**。这不是实现缺陷，
是 cuSPARSE 这套接口语义的固有代价，设计上只能围绕它做取舍（见详细设计 §3.5）。

---

# 需求分析（required）

## 需求描述

在 `arch35` 上把 `aclsparseDenseToSparseGetBufferSize / Analysis / Convert` 三个接口的
能力补齐到：五种 values dtype（INT8/FP16/BF16/FP32/complex64）× 四种目标格式
（CSR/CSC/COO/Blocked-ELL）× 两种 Dense layout（ROW/COL）× 两种 index base（0/1），
Device 索引 I32（现有 I64 能力作为超集保留），并新增 Python/ATen 适配层；
性能达到各自标杆的 0.3 倍以上，内存满足 §3.4 二选一条件。

## 需求拆解

| # | 需求 | 来源 | 性质 |
|---|---|---|---|
| R1 | values 支持 complex64，实/虚部分别 bit-wise 一致 | §1、§3.2 | 新增 |
| R2 | complex64 的非零规则：任一分量非零或为 NaN 则保留 | §2.1 | 新增 |
| R3 | Blocked-ELL 支持 rows/cols/ellCols 非 blockSize 整除的尾块，越界逻辑元素写正零 | §2.4 | 语义扩展 |
| R4 | Blocked-ELL 的 values 类型扩到含 complex64 | §1、§2.4 | 新增 |
| R5 | Python/ATen 四入口在 NPU 注册，不得 CPU fallback | §2.0、§3.5 | 新增 |
| R6 | 输出坐标顺序确定性，重复执行 nnz/结构/values 逐位一致 | §2.1、§3.2 | 已有，需保持并补测 |
| R7 | 输入 Dense 与 BELL pattern 全程只读 | §2.1、§7.3 | 已有，需补测 |
| R8 | 性能 ≥ 各自标杆 0.3 倍，报告分阶段与端到端耗时 | §3.3 | 优化 |
| R9 | workspace 仅按查询值分配，Analysis/Convert 可复用；核心路径不得建与 Dense 或 payload 成比例的额外 Host 缓冲 | §3.4 | 约束 |
| R10 | 接 200 条精度用例、320 条性能用例、内存对比脚本 | §3.4、§3.5 | 新增 |
| R11 | 与 A2/A3 的公共 Host 代码交叉回归、处理合入冲突 | §3.5、§5 | 流程 |

---

# 详细设计（required）

## 算子分析

### 数学公式

设 Dense 为 \(A \in \mathbb{K}^{m \times n}\)，\(\mathbb{K}\) 为 INT8/FP16/BF16/FP32/complex64
之一。定义非零谓词 \(\mathrm{nz}(a)\)（见 §3.2），则

\[
S = \{(i,j) \mid \mathrm{nz}(A_{ij})\},\quad \mathrm{nnz} = |S|
\]

CSR/CSC/COO 输出 \(S\) 的坐标与对应 values；Blocked-ELL 不发现结构，
而是按调用方预置的 block-column pattern \(P\) 提取固定块。**输出 values 是输入 bit
模式的搬运，不做任何算术或类型转换。**

### 支持数据类型

| values dtype | 枚举 | 字节 | 现状 | 本任务 |
|---|---|---|---|---|
| int8 | `ACL_INT8` | 1 | 已支持 | 保持 |
| float16 | `ACL_FLOAT16` | 2 | 已支持 | 保持 |
| bfloat16 | `ACL_BF16` | 2 | 已支持 | 保持 |
| float32 | `ACL_FLOAT` | 4 | 已支持 | 保持 |
| **complex64** | `ACL_COMPLEX64` | **8** | 不支持 | **新增（R1/R4）** |

索引与 offsets：Device 侧 I32 为任务书规定；现有实现同时支持 I64，作为超集保留。
CSR/CSC 的 offset 与 element index 必须同型（与 cuSPARSE 实际行为一致），
混合组合返回 `NOT_SUPPORTED`。`matA` 与 `matB` 的 values 类型必须相同。

### 支持形状

`rows`、`cols`、CSR/CSC/COO 的实际 `nnz` 取值范围 `[0, INT32_MAX]`；
选 I32 时所有输出 offset/index 还须能由 I32 表示。ROW 要求 `ld ≥ cols`，
COL 要求 `ld ≥ rows`。支持零维、全零、全非零。Blocked-ELL 几何见 §3.3。

## 算子实现

### 3.1 complex64：把非零谓词统一成「掩符号位后看剩余位」

现有三个 `IsNonzero` 特化本质上是同一条规则：**去掉符号位，剩余位非零即非零**。
这条规则恰好把 complex64 的要求也覆盖了：

- `±0` 的幅值位全为 0 → 不入结构 ✓
- NaN 的指数位全 1、尾数非零 → 幅值位非零 → 保留 ✓
- `±Inf` 的指数位全 1 → 幅值位非零 → 保留 ✓

complex64 是交织的 `(re, im)` 两个 fp32。按 `uint64_t` 一次加载两个分量，
谓词是两个半字各自掩符号位后取或：

```cpp
template <> __simt_callee__ __aicore__ inline bool IsNonzero<uint64_t>(uint64_t bits)
{
    const uint32_t lo = static_cast<uint32_t>(bits) & 0x7FFFFFFFU;          // Re
    const uint32_t hi = static_cast<uint32_t>(bits >> 32) & 0x7FFFFFFFU;    // Im
    return (lo | hi) != 0;
}
```

这一条同时满足 R2 的「任一分量非零或为 NaN 时保留」——因为 NaN 的幅值位本就非零，
不需要单独的 `isnan` 分支。**把语义收敛成一条位运算规则，而不是给 complex64 开一条
特殊路径**，是本设计在 complex64 上的核心取舍：结构发现逻辑与值宽度解耦，
kernel 只需按 `elementBytes` 多一个 8 字节分支，values 搬运仍是纯按位拷贝。

配套改动：
- host `GetElementBytes()` 增加 `ACL_COMPLEX64 → 8`（该函数是唯一的 dtype 门禁，
  `matA`/`matB` 两处校验都走它，故这一处改动即打开 complex64 通路）；
- dtype 校验白名单加入 `ACL_COMPLEX64`（CSR/CSC/COO 与 BELL 都要加，见 R4）；
- Golden 侧按 `std::complex<float>` 的位表示实现同一条谓词，用于对拍。

**风险点**：8 字节元素的对齐与合并访存。fp32 已是 4 字节，complex64 的 8 字节
在 Dense 侧是自然对齐的连续两个 fp32，读取无额外代价；写 values 时每个非零元素写
8 字节，比 fp32 更"宽"，反而对写效率有利（上一个任务里 sub-word 写的惩罚很明显，
8 字节写不存在这个问题）。

### 3.2 非零语义汇总（这是本算子的「精度」）

| 输入 | 是否入结构 | values 写什么 |
|---|---|---|
| `+0` / `-0`（浮点） | 否 | — |
| `0`（int8） | 否 | — |
| 普通有限非零 | 是 | 原始 bit |
| `+Inf` / `-Inf` | 是 | 原始 bit |
| NaN（含任意 payload） | 是 | 原始 bit，payload 不变 |
| complex64，任一分量幅值位非零 | 是 | 原始 8 字节 bit |
| complex64，`(+0, -0)` 等两分量皆零 | 否 | — |
| BELL pattern 未覆盖 / 越界 slot | （容量固定） | **正零**（全字节 0） |

「正零」对五种 dtype 都是全字节 0，实现上是 `memset` 语义，不需要按 dtype 分支。

### 3.3 Blocked-ELL 尾块（R3，本任务风险最高项）

**现状**：要求 `rows % b == 0 && cols % b == 0 && ellCols % b == 0`，否则
`INVALID_VALUE`；容量为 `ellColInd[blockRows*slots]`、`ellValue[rows*ellCols]`，
且 `aclsparseSpMatGetSize` 报的 `nnz = rows*ellCols`。

**任务书要求**：三者均可不整除，越界逻辑元素写正零。

**设计**：把「整除」从**前置条件**改为**布局的一种特例**，几何量改用向上取整：

```
blockRows = ceil(rows / b)
slots     = ceil(ellCols / b)
ellColInd 元素数 = blockRows * slots
ellValue  标量数 = blockRows * b * slots * b     // 注意：不再是 rows * ellCols
```

关键取舍在 `ellValue` 的容量定义。有两种选法：

| 方案 | `ellValue` 容量 | 优点 | 代价 |
|---|---|---|---|
| **A. 按补齐后的块网格** | `blockRows*b × slots*b` | 块内寻址公式不变（`task*b*b + innerCol*b + innerRow`），kernel 改动最小 | 与现有 `rows*ellCols` 不同，`nnz` 语义随之改变 |
| B. 仍按 `rows × ellCols` | `rows*ellCols` | 兼容现有容量与 `nnz` | 尾块在 values 里不再是完整 `b*b`，块内寻址要按行截断，kernel 与 Golden 都要引入尾块特例 |

**本设计选 A**，理由：越界逻辑元素要求写正零，说明**存储上这些位置是存在的**
（否则无处可写）；A 让「每个 slot 恒占 `b*b` 个标量」这一条不变量在尾块上继续成立，
kernel 里越界判断退化为对 `row < rows && col < cols` 的一次比较，
不需要第二套寻址公式。B 虽然容量更省，但会让块内寻址在尾块与非尾块之间分叉，
是正确性风险更高的选择——上一个任务里「循环按上界展开而非真实路数」导致的
串值 bug，性质与此类似。

选 A 的连带后果是 `aclsparseSpMatGetSize` 报的 `nnz` 从 `rows*ellCols` 变为
`blockRows*b*slots*b`，两者仅在不整除时不同，整除时完全一致，**因此不破坏现有整除
用例的行为**。但这是对外可见的语义变化，需要评审确认（[待确认] 第 3 条）。

越界元素的判定（`b*b` 内每个逻辑位置）：

```
row = blockRow * b + innerRow
col = blockColumn * b + innerCol      // blockColumn 由 pattern 解码
写正零 当且仅当  pattern 项无效（padding/越界）  或  row >= rows  或  col >= cols
```

pattern 有效性判定沿用现有规则：对 base 为 `base` 的项 `encoded`，
有效当且仅当 `base <= encoded < base + ceil(cols/b)`；`-1`、小于 `base` 的值、
以及 1-based 下的 `0` 均按 padding 处理。注意上界随 `cols` 改用向上取整。

### 3.4 Python/ATen 适配（R5）

仓内无先例，需新建目录。参照四个入口的实际派发链设计注册点：

| 公开入口 | ATen 算子 | 目标格式 |
|---|---|---|
| `Tensor.to_sparse()` | `aten::_to_sparse` | COO |
| `Tensor.to_sparse_csr()` | `aten::_to_sparse_csr` | CSR |
| `Tensor.to_sparse_csc()` | `aten::_to_sparse_csc` | CSC |
| `Tensor.to_sparse_bsr()` | `aten::_to_sparse_bsr` | BSR（≈ Blocked-ELL，语义差异见下） |

适配层职责：参数/dtype/shape/layout/stride/device 校验 → 构造 `aclsparseConstDnMat`
与目标 `aclsparseSpMat` 描述符 → 走三阶段 → 用 Analysis 得到的 `nnz` 分配 payload 并
`SetPointers` → 构造 PyTorch Sparse Tensor 输出。不支持的组合返回明确错误，
不得回退 CPU。

**[待确认] 第 5 条：`to_sparse_bsr` 与 Blocked-ELL 不是同一种格式。** BSR 按块存储
**所有**非零块并带 `crow_indices/col_indices`，块结构由数据发现；Blocked-ELL 是每
block row 固定 `slots` 个 slot、pattern 由调用方预置、不发现结构。二者不能直接互映。
可选做法：(a) 用 CSR 路径先得到块级结构再组装 BSR，Blocked-ELL 只在 C++ 层暴露；
(b) 只注册前三个入口，`to_sparse_bsr` 明确返回不支持并在文档说明。倾向 (a)，
但需确认验收口径。

派发落点需在真机实测确认（上一个任务的教训：CSR 入口最终缺的符号是
`aten::addmm.out`、后端 key 实名 `SparseCsrnpu`，与文档推断不同）。因此**第一步是
写派发探测脚本、在 950PR 上跑出真实的算子链与缺失符号**，再定注册清单。

### 3.5 性能设计：两个标杆对应两种完全不同的瓶颈

任务书 §3.3 说「二者 API 和计时工作不同，不作横向混算」。核对
`baseline_results/gpu_full_results.tsv`（其 README 声明为本任务包的权威导出）后，
两个标杆的覆盖面是**互补**的：

| 标杆来源 | P-01/02/03 有效条数 | 覆盖格式 | P-01 median_us 区间 |
|---|---|---|---|
| 原生 cuSPARSE（C++） | 30 | **仅 Blocked-ELL**（5 dtype × 2 base × 3 场景） | 16.576 ~ 18.880 |
| PyTorch GPU | 90 | **CSR / CSC / COO**（各 30） | 622.960 ~ 7317.728 |

也就是说 **Blocked-ELL 对标原生 cuSPARSE，CSR/CSC/COO 对标 PyTorch GPU**，
二者不重叠。这直接决定了两条路径的优化方向完全不同。

**Blocked-ELL：小数据量、受启动延迟支配。** BELL 只读 pattern 覆盖的块。
三个场景的 `ellCols` 均为 64（由 `nnz/rows` 反推），实际工作量：

| 场景 | rows×cols | b | ellCols | 块数 | 标量数 | 读+写字节（int8 ~ complex64） |
|---|---|---|---|---|---|---|
| P-01 | 8192×28672 | 16 | 64 | 2048 | 524,288 | 1.0 MB ~ 8.4 MB |
| P-02 | 4096×1536 | 32 | 64 | 256 | 262,144 | 0.5 MB ~ 4.2 MB |
| P-03 | 7168×2048 | 64 | 64 | 112 | 458,752 | 0.9 MB ~ 7.3 MB |

标杆 16.6us 完成 1.0 MB 只相当于 63 GB/s，远低于 H100 峰值——**标杆本身就不是带宽
受限，而是被固定开销支配**。0.3x 目标给我们 45~63us 的预算。因此 BELL 的优化目标是
**压低每次调用的固定成本**：kernel 启动次数、Host↔Device 往返、tiling 计算。
BELL 的 `GetBufferSize` 返回 0、Analysis 不扫 Dense 不启 kernel，这两点现有实现已经
做对了，Convert 应保持单 kernel 一次启动。

**CSR/CSC/COO：全量扫描、受带宽支配，且天然两遍。** NPU 需读完整 Dense。
按每个 case 自己的标杆算 0.3x 预算所需的有效读带宽（单遍）：

| | 最大 | 中位 | 最小 |
|---|---|---|---|
| 所需带宽（单遍，90 条） | 349 GB/s | 51 GB/s | 7 GB/s |

最紧的是 `P-01-coo-complex64`（1879 MB / 5380us）。考虑到 §「功能分析」里说明的
**Analysis 与 Convert 各读一遍 Dense**，真实需求约为其两倍（~700 GB/s 峰值场景）。
950PR 的 HBM 带宽在此之上，**目标可达，但必须把两遍都做成接近纯流式的顺序读**，
没有余量做低效访存。

是否可以只扫一遍？可以，代价是在 workspace 里缓存非零掩码（`rows*cols/8` 字节，
P-01 为 29 MB），但这与 §3.4 的第 2 项条件（固有 workspace 不超过 L2 容量）冲突。
**本设计不缓存掩码**，走两遍扫描，内存合规性走 §3.4 的第 1 项条件（额外内存不超过
GPU 总用量的 50%）——PyTorch GPU 侧 P-01 的 `extra_peak_allocated_bytes` 达 2.13 GB，
而我们的 workspace 只有 O(核数 × 分块) 量级，这一项有很大余量。

**优化顺序（吸取上一个任务的教训）**：第一步就用 `msprof --task-time=on` 拿逐 kernel
归因，不靠阶段级计时反推瓶颈。已知的候选方向：Analysis 的行/列计数与前缀和是否为
并行扫描（上一个任务里串行前缀和曾占 81%）、Convert 的写是否合并、
COL layout 下的跨步读是否需要分块转置。

### 3.6 workspace 与生命周期（R9）

沿用现有语义：workspace 是每次调用的临时 scratch，不保存跨阶段身份，
Analysis 与 Convert 可用不同地址；`bufferSize == 0` 时可传 `nullptr`（BELL 与空输入）。
新增 complex64 不改变 workspace 布局——workspace 里放的是计数与前缀和等
**索引量**，与 values 宽度无关。

R9 的「不得创建与 Dense 或 payload 成比例的额外 Host 缓冲」在现有实现里已满足：
Analysis 只把标量 `nnz` 回读到 Host。新增 ATen 层时要特别注意不要为了构造
Sparse Tensor 而在 Host 上中转 payload。

## 支持硬件

| 芯片 | 支持情况 |
|---|---|
| Ascend 950PR / Ascend 950DT | 支持（arch35 / dav-3510） |

## 算子约束限制

1. `alg` 仅支持 `ACL_SPARSE_DENSETOSPARSE_ALG_DEFAULT`，其余枚举返回参数错误。
2. CSR/CSC 的 offset 类型与 element index 类型必须相同；I32/I64 与 I64/I32 混合组合
   可用于创建描述符，但本算子返回 `NOT_SUPPORTED`。
3. `matA` 与 `matB` 的 values 类型必须相同；不支持 float64 与 complex128。
4. Blocked-ELL 不发现结构：pattern 未覆盖的 Dense 非零被忽略，不扩充、不重排、
   不报容量不足。
5. Convert 不校验 Analysis 时的快照（Dense values/shape/`ld`/order/指针），
   只消费调用时的当前描述符字段；一致性由调用方保证。
6. Convert 保持 stream 异步，Device 侧检出的不一致不会转成本次 API 的同步错误返回。
7. 描述符只借用 Device 指针，不拥有内存；须先同步 stream 再销毁描述符、释放内存。

---

# 可维可测分析

## 精度标准/性能标准

**精度：exact match，无容差。** 依据任务书 §3.2：CSR/CSC/COO 的 `nnz`、坐标、
indices、values 与 CPU Golden 完全一致；五种 dtype 保持非零元素 bit 模式；
complex64 实部虚部分别 bit-wise 一致。这比上一个任务的混合容差**更严也更简单**
——不存在「差多少算对」的判断，一个 bit 不同就是错。

Golden 用 CPU 实现，按 §3.2 的谓词表逐条对照；由于不做算术，Golden 不需要加宽类型
（这与 SpGEMM 需要 fp64/complex128 加宽形成对比）。

**性能：各自标杆的 0.3 倍以上**，口径为「标杆 GPU 设备 Event 调用耗时 / NPU 同调用
范围总耗时」，每 case 预热 ≥10 次、正式采样 ≥30 次，报告中位数、p90、Analysis、
Convert、完整三阶段、端到端、实际 nnz、稀疏度、workspace 峰值和各 Kernel 耗时。
按 §3.5 的结论，Blocked-ELL 对标原生 cuSPARSE、CSR/CSC/COO 对标 PyTorch GPU。

**内存：** §3.4 二选一，本设计走第 1 项（额外内存 ≤ GPU 总用量 50%），
用 `collect_sparse_ops_{gpu,npu}_memory.py` 采集、`compare_sparse_ops_memory.py`
按 case id 对比；Blocked-ELL 无等价 Torch API，按 `GetBufferSize` 查询值与 Profiler
验收固有 workspace 不超过 L2 容量，输出 payload 字节数另行报告。

## 测试策略

分四层，从不依赖硬件到完整上板：

**第一层：主机侧算法验证（不需要 NPU）。** 把非零谓词与 BELL 尾块寻址抽成可用主机
编译器编译的头文件，与 Golden 对拍。上一个任务证明这一层能把算法 bug 的定位周期从
「排队上板」压到秒级，尤其适合 BELL 尾块这种寻址易错项。

**第二层：C++ UT。** 在现有 1274 行基础上补：

| 覆盖项 | 用例要点 |
|---|---|
| complex64 | 四格式 × base 0/1 × ROW/COL；手算小例；实虚部独立的 `±0`、Inf、NaN 组合 |
| BELL 尾块 | rows/cols/ellCols 三者分别不整除、两两不整除、三者同时不整除；`b` 取 16/32/64 |
| BELL padding | `-1`、小于 base 的值、1-based 的 `0`、超上界的值；越界逻辑元素为正零 |
| 零语义 | `±0` 不入结构；全零；全非零；零维 |
| base 端点 | base 0 的 offsets 首尾 `0/nnz`；base 1 为 `1/nnz+1` |
| 确定性 | 同输入重复执行，nnz/结构/values 逐位一致 |
| 只读性 | Dense 与 pattern 调用前后逐字节比对 |
| 三阶段状态 | 跳阶段、pointer rebinding、workspace 复用与不足、`bufferSize==0` 传 `nullptr` |
| 异常 | 空描述符/指针、非法 shape/dtype/format/index/base/`ld`/alg、溢出 |
| guard 区 | 输出缓冲前后哨兵，检测越界写 |

**第三层：任务包的 200 条精度用例与 320 条性能用例。** 精度用例的判定类型是
`dense_to_sparse_exact`（custom），与 §3.2 的 exact match 一致。
**这一层要尽早接通**——上一个任务里官方性能脚本因为 ATen 层没打通而一路没跑，
到交付前才发现两套口径从未交叉验证，是明确的教训。

**第四层：ATen 端到端与 Profiler 证据。** 四个入口的公开语义、参数/dtype/shape/
layout/stride/device 校验、异常、alias/in-place（适用时）；用 NPU Dispatch 与
Profiler 证明各 Kernel 都在 AI Core 上、无 CPU fallback。

## 兼容性分析

**对现有行为的影响。** R1（complex64）与 R4 是纯新增，对现有四种 dtype 无影响。
R3（BELL 尾块）把整除从前置条件改为特例，**整除输入的容量公式与 `nnz` 完全不变**，
故现有整除用例行为不变；不整除输入从「返回 `INVALID_VALUE`」变为「成功并按尾块语义
输出」，是能力扩展方向的行为变化。

**与 A2/A3 的公共代码冲突。** 任务书 §5 要求处理公共 Host 冲突并完成交叉回归。
按上一个任务的经验，冲突集中在 `include/cann_ops_sparse.h` 与公共 CMake：
本任务不新增公开接口（三个接口已存在），头文件改动预计仅限于 dtype 支持范围的注释，
冲突面比 SpGEMM 小得多。仍将在提 PR 前用
`git merge-tree --write-tree HEAD <A2A3 分支>` 预演，并把双方新增内容取齐到逐字相同
以便自动合并。

**版本兼容。** PyTorch 2.7+、torch_npu 26.0.0+。注意 torch_npu 的 `26.1.0` 是 Ascend
发布列车号，PyPI 上以配套 PyTorch 版本号发布（`v26.1.0-pytorch2.12.0` 的
`version.txt` 内容即 `2.12.0`），不要为凑版本号降 PyTorch。

---

# 风险点

| # | 风险 | 影响 | 应对 |
|---|---|---|---|
| 1 | BELL 尾块的 `ellValue` 容量与 `nnz` 语义变化 | 对外可见的语义变更 | 选方案 A 保证整除输入行为不变；提交评审确认（[待确认] 3） |
| 2 | `to_sparse_bsr` 与 Blocked-ELL 语义不等价 | ATen 入口可能无法覆盖 | 先实测派发链，再定注册清单；备选只注册前三入口并说明（[待确认] 5） |
| 3 | CSR/CSC/COO 两遍全量扫描的带宽压力 | 最紧场景需 ~700 GB/s | 两遍都做成纯流式顺序读；先用 profiler 定位再优化 |
| 4 | COL layout 下的跨步访存 | 可能显著慢于 ROW | 分块处理以恢复局部性；作为独立优化项量测 |
| 5 | 标杆数据两份不一致 | 直接影响性能目标 | 见 [待确认] 1，以 TSV 为准 |

# 交付物

1. **算子设计文档**：本文件，PR 至 `cann-ops-competitions` 的
   `04_tasks/01_community-task-2026/tasklist/09-aclsparseDenseToSparse-950/sss-hust/docs/design.md`。
   提 PR 时 `upstream/master` 上尚无任何 `09-*` 目录，故按仓内既有惯例
   （`08-15-aclsparseSpGemm-950` 等 `月份-序号-任务名-平台` 形式）取此名，序号待定；
   若已规划标准编号，按评审意见调整。
2. **代码**：PR 至 `ops-sparse` 的 `master`：
   - `include/cann_ops_sparse.h`：dtype 支持范围更新
   - `sparse/densetosparse/arch35/`：host、kernel、tiling 与构建配置
   - `sparse/densetosparse/README.md`：接口说明与限制说明更新
   - `test/densetosparse/arch35/` 与 `test/densetosparse/`：C++ UT、NPU wrapper、参数与 Golden
   - Python/ATen 适配层与端到端 UT
3. **自测用例与测试代码**：四层测试、一键 `verify_all.sh`、`REPRODUCE.md`。
4. **自测报告**：用例参数、输出结构与精度结果、性能数据、峰值内存、截图、
   Profiler 证据、失败项说明。

---

# 希望检视人员了解

以下五条依赖评审确认，其中前两条直接影响性能验收结论。

**1. 标杆数据两份不一致，本设计以 TSV 为准。** 任务包里
`aclsparseDenseToSparse_testCase/gpu_performance_result_benchmark.md` 与
`baseline_results/gpu_full_results.tsv` 对同一 case 给出不同数值。例如
`dense-to-sparse-P-01-csr-int8-base0` 的 PyTorch GPU `median_us`：md 为 **17937.904**，
TSV 为 **1362.528**，相差约 13 倍。判断依据：TSV 的 README 声明它是
"the single complete, operator-filtered GPU baseline export for this standalone task
package"，且**任务书 §3.3 目标表里引用的四个区间与 TSV 完全吻合**
（C++ 16.576–18.880 / 13.632–18.208 / 14.272–19.008；PyTorch P-01 622.960–7317.728、
P-02 103.024–293.936、P-03 147.360–549.776）。md 自称数据源为 `../cuda_result` 与
`../cuda_result2`，而这两个目录未随包交付。请确认以 TSV 为准。

**2. 标杆 GPU 是 H100。** `baseline_results/manifest.json` 的 `source_inputs` 记录源
文件名为 `cuda_result_H100` 与 `cuda_result2_H100`；而 md 正文写「用户确认本次采集
设备为 标杆」，是未替换的占位符。报告中将按 H100 表述，请确认。

**3. Blocked-ELL 尾块导致 `ellValue` 容量与 `nnz` 语义变化（详细设计 §3.3）。**
选按补齐后的块网格（`blockRows*b × slots*b`）而非 `rows*ellCols`，使「每 slot 恒占
`b*b` 标量」在尾块上继续成立、块内寻址公式不分叉。整除输入下两者完全相同，
故不破坏现有行为；不整除时 `aclsparseSpMatGetSize` 报的 `nnz` 会随之改变。
请确认该语义，或指定按 `rows*ellCols` 截断的方案 B。

**4. 任务书 §7.1 说「Device 索引固定为 I32」，而现有实现同时支持 I64。**
本设计保留 I64 作为超集（不移除既有能力），验收按 I32 口径。请确认这样处理可接受。

**5. `to_sparse_bsr` 与 Blocked-ELL 不是同一种格式（详细设计 §3.4）。**
BSR 存储所有非零块、块结构由数据发现并带 `crow_indices/col_indices`；
Blocked-ELL 每 block row 固定 slots、pattern 由调用方预置、不发现结构。
拟按 (a) 走 CSR 路径得到块级结构再组装 BSR、Blocked-ELL 仅在 C++ 层暴露；
备选 (b) 只注册前三个入口并明确返回不支持。请指定验收口径。

另外说明一点方法论上的安排：ATen 层的注册清单**不按文档推断，以 950PR 实机的派发
探测结果为准**。上一个任务中 `torch.sparse.mm(csr, csr)` 的实际缺失符号是
`aten::addmm.out`、后端 key 实名 `SparseCsrnpu`，与仅凭文档得出的结论不同，
故本任务把「写派发探测脚本并上板实测」列为 ATen 工作的第一步。
