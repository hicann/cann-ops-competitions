# aclsparseDenseToSparse 算子设计文档（Atlas A2/A3 / arch22）

> 交付件①。按 `resources/design_template.md` 组织，四个必需一级章节齐全。

---

# 需求背景（required）

## 需求来源

CANN 生态 9 月社区任务「aclsparseDenseToSparse 算子开发(A2A3)」。在 `ops-sparse` 仓为
Atlas A2/A3（DAV_2201，`arch22`，覆盖 910B3/910B4/910_93 系列）从零实现稠密矩阵到
稀疏矩阵的三阶段转换：CSR / CSC / COO / Blocked-ELL 四种目标格式、五种 values
dtype、I32 Device 索引、index base 0/1，并交付 Python/ATen 适配层。

代码交付至 `https://gitcode.com/cann/ops-sparse` 的 `master`；设计文档交付至
`https://gitcode.com/cann/cann-ops-competitions`。

## 背景介绍

### 这不是数值计算算子

`DenseToSparse` 做两件事：**发现结构**（哪些逻辑坐标非零、共有多少个）与**搬运位**
（把非零元素的原始 bit 模式写到目标格式的 values 里）。它不做任何算术，因此任务书
§3.2 要求的是 **exact match / bit-wise 一致**，而不是混合容差。这条决定了整套设计的
重心：**没有精度误差可言，风险全在结构语义的边界、确定性与 arch22 移植正确性上**。
正负零是否入结构、INF/NAN 是否保留、base 0/1 的 offsets 端点、尾块 padding 写什么、
重复执行是否逐位可复现——这些是本算子的「精度」。

### `ops-sparse` 现状分析

`sparse/densetosparse/` 已有 **arch35（A5，DAV-3510）实现**，无 arch22 实现。基于
`origin/master`（e0015bb）实测：

| 文件 | 行数 | 职责 |
|---|---|---|
| `arch35/densetosparse_host.cpp` | 658 | 三个公开接口、参数校验、workspace 布局、tiling、kernel 启动 |
| `arch35/densetosparse_kernel.cpp` | 622 | Analysis / Convert 的 SIMT kernel 群（count/scan/offsets/convert/bell） |
| `arch35/densetosparse_kernel.h` | 30 | 两个 `*_kernel_do` 启动函数声明 |
| `arch35/densetosparse_tiling_data.h` | 40 | `DenseToSparseTilingData` |
| `test/densetosparse/arch35/densetosparse_test.cpp` | 1274 | C++ UT |
| `test/densetosparse/densetosparse_golden.h` | 300 | CPU Golden（与 arch 共享） |

arch35 已具备的能力（依据 `sparse/densetosparse/README.md` 与源码核对）：

| 维度 | 现状 |
|---|---|
| values dtype | INT8 / FP16 / BF16 / FP32（**无 complex64**） |
| 目标格式 | CSR / CSC / COO / Blocked-ELL（**BELL 强制整除，无尾块**） |
| Dense layout | ROW（`ld≥n`）/ COL（`ld≥m`） |
| index base | 0 / 1 |
| 索引类型 | I32 / I64（CSR/CSC offset 与 element index 必须同型） |
| 零语义 | `±0` 不入结构；NaN（含 payload）、`±Inf` 按位搬运 |
| 三阶段 | GetBufferSize（纯查询）→ Analysis（写 offsets、更新 `nnz`）→ Convert（写 indices/values） |

arch22 侧的仓库先例（作为工程模式参照）：

- `sparse/sddmm/arch22/`：**标准 Ascend C** 工程模式（`TPipe`、`op.Init/op.Process`、
  `GetAivCoreCount()` 分核、`KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY)`）；
- `sparse/spmv/arch22/kernels/spmv_kernel.h:643`：`AscendC::SyncAll()` 多核同步先例；
- 构建分发：根 CMake `get_soc_arch_dirs()` 把 `ascend910b*`、`ascend910_93*` 映射到
  `arch22`，`sparse/CMakeLists.txt` 按 arch 目录 glob 源文件——**新增
  `sparse/densetosparse/arch22/` 目录即可自动参与 arch22 构建，无需改构建脚本**。

公共代码的关键约束（本任务必须触碰、且与 A5 任务共享）：

- `sparse/common/aclsparse_descr.cpp` 的 `aclsparseCreateBlockedEll()`（444 行起）
  **在公共描述符层强制 `rows/cols/ellCols % blockSize == 0`**，并把 BELL 的
  `nnz` 定为 `rows * ellCols`、pattern 容量定为
  `(rows/b) * (ellCols/b)`。BELL 尾块语义必须改这里，是跨任务冲突点；
- `IsValidSparseValueType()` 已包含 `ACL_COMPLEX64`——描述符层已放行 complex64，
  缺口只在算子本体的 dtype 门禁与 kernel，影响面可控。

### 与 950（A5）任务的关系：同一算子的两个平台档

社区同期存在「aclsparseDenseToSparse(950)」任务（增强既有 arch35 实现）。两个任务
共享：公开头文件、公共描述符、`test/densetosparse/` 公共 Golden/参数/用例、算子
README。**本任务在 arch22 上是全新实现**，但语义必须与 A5 对齐，公共代码改动双方
取齐到逐字相同以便自动合并（任务书 §5 要求处理公共 Host 冲突并完成 A2/A3 与 A5
交叉回归）。

### 功能缺口

**缺口一：arch22 整体缺失。** 无 host、无 kernel、无构建、无测试。这不是简单搬运：
arch35 kernel 用 SIMT 编程模型（`__simt_vf__`、`asc_vf_call`、`blockIdx/threadIdx`，
DAV-3510 特性），arch22 必须重写为标准 Ascend C 向量 kernel（见详细设计 §3.3）。
Host 侧校验/workspace/tiling 逻辑与架构无关，可按函数等价移植。

**缺口二：complex64 未支持**（与 A5 任务相同缺口）：`GetElementBytes()` 只识别
1/2/4 字节，complex64 返回 0 → `NOT_SUPPORTED`。

**缺口三：Blocked-ELL 不支持非整除尾块**（与 A5 任务相同缺口）：host 与公共描述符
两层都拒绝不整除几何；任务书 §2.4 要求支持尾块并对越界逻辑元素写正零。

**缺口四：Python/ATen 适配完全缺失**（必选交付）：全仓无 torch/aten/python 目录。
四个公开入口 `Tensor.to_sparse / to_sparse_csr / to_sparse_csc / to_sparse_bsr`
需在 NPU 注册且不得 CPU fallback。

### 功能分析：三阶段契约决定两遍扫描

`Analysis` 必须先产出 `nnz`，调用方据此分配 payload 并 `SetPointers`，`Convert` 才能
写。`nnz` 要更新到 Host 描述符，所以 **Analysis 内必然有一次 Device→Host 回读同步**；
Convert 时 payload 才存在，所以 **Dense 必然被读两遍**。这不是实现缺陷，是 cuSPARSE
这套接口语义的固有代价，设计上只能围绕它做取舍（见 §3.7 性能设计）。

---

# 需求分析（required）

## 需求描述

在 `arch22` 上实现 `aclsparseDenseToSparseGetBufferSize / Analysis / Convert` 三个
接口，能力达到：五种 values dtype（INT8/FP16/BF16/FP32/complex64）× 四种目标格式
（CSR/CSC/COO/Blocked-ELL）× 两种 Dense layout（ROW/COL）× 两种 index base（0/1），
Device 索引 I32（现有 I64 作为超集保留），并新增 Python/ATen 适配层；
每个有效 case 性能倍率达到 0.25 倍性能标杆以上，内存满足 §3.4 二选一条件。

## 需求拆解

| # | 需求 | 来源 | 性质 |
|---|---|---|---|
| R1 | arch22 全新实现：Host 分发 + Ascend C Kernel + 构建 + 测试 + 文档 | §1、§2.2、§5 | 新增（本任务主体） |
| R2 | values 支持 complex64，实/虚部分别 bit-wise 一致 | §1、§3.2 | 新增 |
| R3 | complex64 非零规则：任一分量非零或为 NaN 则保留 | §2.1 | 新增 |
| R4 | Blocked-ELL 支持 rows/cols/ellCols 非 blockSize 整除的尾块，越界逻辑元素写正零 | §2.4 | 语义扩展（含公共描述符改动） |
| R5 | Device 索引 I32；Host 尺寸元数据保持 `int64_t` | §7.1 | 规定 |
| R6 | Python/ATen 四入口在 NPU 注册，不得 CPU fallback | §2.0、§3.5 | 新增 |
| R7 | 输出坐标顺序确定性，重复执行 nnz/结构/values 逐位一致 | §2.1、§3.2 | 需保证并补测 |
| R8 | 输入 Dense 与 BELL pattern 全程只读 | §2.1、§7.3 | 需保证并补测 |
| R9 | 性能倍率 ≥ 0.25，报告分阶段与端到端耗时 | §3.3 | 约束 |
| R10 | workspace 仅按查询值分配，可跨阶段复用；不得建与 Dense/payload 成比例的额外 Host 缓冲 | §3.4 | 约束 |
| R11 | 交付 C++ UT、ATen UT、Python 端到端 UT；接 200 条精度 / 320 条性能 / 内存用例；提交 910B3、910B4、A3 三型号报告 | §3.4、§3.5 | 新增 |
| R12 | 与 A5 公共 Host 交叉回归、处理合入冲突 | §3.5、§5 | 流程 |

---

# 详细设计（required）

## 算子分析

### 数学公式

设 Dense 为 \(A \in \mathbb{K}^{m \times n}\)，\(\mathbb{K}\) 为
INT8/FP16/BF16/FP32/complex64 之一。定义非零谓词 \(\mathrm{nz}(a)\)（§3.4），则

\[ S = \{(i,j) \mid \mathrm{nz}(A_{ij})\},\quad \mathrm{nnz} = |S| \]

CSR/CSC/COO 输出 \(S\) 的坐标与对应 values；Blocked-ELL 不发现结构，按调用方预置的
block-column pattern \(P\) 提取固定块。**输出 values 是输入 bit 模式的搬运，不做任何
算术或类型转换。**

### 支持数据类型

| values dtype | 枚举 | 字节 | arch35 现状 | 本任务（arch22） |
|---|---|---|---|---|
| int8 | `ACL_INT8` | 1 | 已支持 | 移植 |
| float16 | `ACL_FLOAT16` | 2 | 已支持 | 移植 |
| bfloat16 | `ACL_BF16` | 2 | 已支持 | 移植 |
| float32 | `ACL_FLOAT` | 4 | 已支持 | 移植 |
| **complex64** | `ACL_COMPLEX64` | **8** | 不支持 | **新增（R2）** |

索引与 offsets：Device 侧 I32 为任务书规定；同时实现 I64 作为超集，保持与 arch35
公共 C++ UT 的用例对等。CSR/CSC 的 offset 与 element index 必须
同型（与 cuSPARSE 实际行为一致），混合组合返回 `NOT_SUPPORTED`。`matA` 与 `matB`
的 values 类型必须相同。

### 支持形状

`rows`、`cols`、CSR/CSC/COO 的实际 `nnz` 取值范围 `[0, INT32_MAX]`；选 I32 时所有
输出 offset/index 还须能由 I32 表示（`nnz + base ≤ INT32_MAX`）。ROW 要求
`ld ≥ cols`，COL 要求 `ld ≥ rows`。支持零维、全零、全非零。Blocked-ELL 几何见
§3.5。

## 算子实现

### 3.1 总体结构：build 级 arch 分发 + 分层复用

```text
include/cann_ops_sparse.h          公开三接口声明（已存在，不改签名）
sparse/common/                     描述符/handle/host utils（复用，仅 BELL 尾块改动）
sparse/densetosparse/arch22/       本任务新增：
  densetosparse_host.cpp             三接口 + 校验 + workspace + tiling + launch
  densetosparse_tiling_data.h        arch22 tiling 结构
  densetosparse_kernel.h             *_kernel_do 声明（与 arch35 同名同构）
  densetosparse_kernel.cpp           Ascend C kernel 群 + launch
sparse/densetosparse/README.md     支持矩阵更新（arch22 行 + complex64）
test/densetosparse/arch22/         C++ UT + NPU wrapper（镜像 arch35 结构）
test/densetosparse/                公共 golden/param/csv 扩展（complex64、BELL 尾块）
python/                            ATen 适配 + pybind 扩展 + e2e UT（§3.6）
```

构建：根 CMake 按 SOC 把 `arch22` 目录纳入 `ARCH_SRC_FILES`（glob 机制，见「现状
分析」），arch22 与 arch35 的 host 在不同 SOC 构建下各自提供同名公开符号，互不
链接——**这是任务书所说「新增 arch22 Host 分发」的落点**。`OP_LIST=densetosparse`
可单独编译本算子。

### 3.2 Host 侧设计

#### 整体架构

```text
┌──────────────────────────────────────────────────────────────────┐
│                     aclsparseDenseToSparse (Host)                  │
│                                                                    │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌───────────────┐ │
│  │ 参数校验   │──→│ workspace│──→│ tiling   │──→│ kernel 启动    │ │
│  │ 六层校验   │   │ 布局计算  │   │ 构建     │   │ (绑定 stream)  │ │
│  │ §3.2.1    │   │ §3.2.4   │   │          │   │               │ │
│  └──────────┘   └──────────┘   └──────────┘   └───────┬───────┘ │
│                                                        │         │
└────────────────────────────────────────────────────────┼─────────┘
                                                         │
                 ┌───────────────────────────────────────▼───────┐
                 │      AIV kernel 群 (满核, KERNEL_TYPE_AIV_ONLY) │
                 │                                               │
                 │  Analysis:  count → scan → addbase →          │
                 │             offsets(I32/I64) → total          │
                 │  Convert:   count → scan → total(校验) →       │
                 │             validate_offsets → convert        │
                 │  BELL:      bell (单 kernel)                   │
                 └───────────────────────────────────────────────┘
```

#### Host 调用流程图

```text
┌─────────────────────── Host ────────────────────────┐
│                                                      │
│  aclsparseDenseToSparseGetBufferSize(A, B, alg)     │
│  aclsparseDenseToSparseAnalysis(A, B, alg, buffer)  │
│  aclsparseDenseToSparseConvert(A, B, alg, buffer)   │
│                                                      │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐       │
│  │ 标量/枚举  │───→│ 描述符配置 │───→│ shape/ld │       │
│  │ 校验      │    │ dtype门禁 │    │ span 校验 │      │
│  └──────────┘    └──────────┘    └────┬─────┘       │
│                                      │               │
│                          ┌───────────▼──────────┐    │
│                          │ format 分派           │    │
│                          └───┬──────────┬───────┘    │
│                     BELL      │          │ CSR/CSC/COO│
│                   ┌───────────▼──┐  ┌────▼─────────┐ │
│                   │ bufferSize=0 │  │ unit 分解     │ │
│                   │ Analysis直接 │  │ workspace布局 │ │
│                   │ 返回 SUCCESS │  │ 多级 scan 规模│ │
│                   └──────────────┘  └────┬─────────┘ │
│                                          │            │
│                    ┌─────────────────────▼─────────┐  │
│                    │ GetLaunchBlocks(work)          │  │
│                    │ = min(AIV核数, ceil(work/4096)) │  │
│                    └─────────────────────┬─────────┘  │
│                                          │            │
│              ┌───────────────────────────▼──────────┐ │
│              │ Analysis: D2H 回读 status/nnz (8B)   │ │
│              │         + SynchronizeStream          │ │
│              │         → matB->nnz = actualNnz      │ │
│              └──────────────────────────────────────┘ │
└──────────────────────────────────────────────────────┘
```

校验、workspace、tiling 逻辑与架构无关，**按 arch35 逐函数等价移植**（保持两 arch
行为一致，便于交叉回归），要点：

1. **校验分层**（全部 Host 完成，任何非法输入在 launch 前返回）：
   标量/枚举（handle/matA/matB 非空、alg 仅 DEFAULT、format ∈ {CSR,CSC,COO,BELL}、
   base ∈ {0,1}）→ 描述符配置（dtype 门禁走 `GetElementBytes()`，complex64 → 8；
   CSR/CSC offset 与 index 同型）→ shape 一致（`matB.rows/cols == matA`、
   `≤ INT32_MAX`、values 非空）→ Dense span（ld 合法性 + 地址/字节溢出检查）→
   稀疏存储（BELL 几何与 pattern/values 指针；CSR/CSC offsets 指针）→
   Convert 前的 payload 指针（indices/values 按 `nnz > 0` 判非空）。
2. **统一 unit 分解**（本任务对 arch35 的结构化改进）：
   CSR 与 COO 按 `(row, col-chunk)` 分解，CSC 按 `(col, row-chunk)` 分解；
   `kMinorChunk = 4096`，`unitCount = major × ceil(minor / 4096)`。
   - 三种格式的 count/scan/convert 共用同一套 unit 机制，COO 不再需要 arch35 的
     逐元素 `pos / cols` 除法（unit 自带 major/minor 基址），kernel 更简单；
   - unit 按 row-major 顺序编号，天然保证 COO 输出按逻辑 row-major 顺序、CSR 行内
     列递增、CSC 列内行递增——**确定性顺序由分解方式直接构造出来**。
3. **分核策略**：满核优先。`blocks = min(GetAivCoreCount(),
   ceil(work / kWorkPerCore))`，`kWorkPerCore = 4096`（work = unitCount 或 BELL
   task 数）；`GetAivCoreCount()` 为 0 时报 `INTERNAL_ERROR`。
4. **workspace 布局**（沿用 arch35 语义，R10 合规）：
   `[status 4B @0][pad][totalNnz u64 @32][level0 @64][level1...]`，各级
   `unitCount × 8B`（u64）按 32B 对齐，级间 `ceil(count / kScanChunk)` 收敛
   （`kScanChunk` 取 4096，比 arch35 的 1024 少一级、少两次 kernel launch）。
   BELL 与空输入 `bufferSize == 0`，`buffer` 可传空。
5. **Analysis 回读**：`aclrtMemcpyAsync` D2H 拷 `status` 与 `nnz` +
   `aclrtSynchronizeStream`（描述符绑定的 stream），校验
   `nnz ≤ INT32_MAX 且 ≤ m×n` 后写入 `matB->nnz`——这是流程中唯一的 Host 同步点。
6. **tiling**：`DenseToSparseTilingData`（arch22 版）携带 rows/cols/ld/nnz/
   ellBlockSize/ellCols/unitCount/各 workspace 偏移/format/order/base/
   offsetType/indexType/elementBytes/numBlocks，与 kernel 侧共享同一头文件。

### 3.3 Kernel 侧设计：SIMT → 标准 Ascend C 的移植（本任务技术主体）

arch35 kernel 是 SIMT 模型（每元素一个线程）；arch22 重写为 **AIV 向量 kernel + UB
数据流**。编程模型映射：

| arch35（SIMT） | arch22（标准 Ascend C） |
|---|---|
| `blockIdx.x * blockDim.x + threadIdx.x` 线程号 | `GetBlockIdx()` 核号 × 每核循环处理多个 unit |
| `__gm__` 逐元素读写 | `DataCopy` GM↔UB 批量搬运 + `GlobalTensor/LocalTensor` |
| `asc_vf_call<F>(dim3{256}, ...)` | `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY)` + 向量指令（And/Or/Compare/ReduceSum/GatherMask） |
| u64 标量算术自由使用 | 向量通路 int64 支持受限 → **计数/压缩在向量通路按 ≤32b 处理，u64 只出现在 prefix 标量通路**（见下） |

**Kernel 流水线**（与 arch35 同构，命名保持一致以便对照；同一 stream 上串行
launch，kernel 间无额外间隙）：

#### Analysis 流程图

```text
┌────────────────── Device (AIV × blocks) ───────────────────┐
│                                                             │
│  ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌──────────┐   │
│  │ count   │──→│ scan    │──→│ addbase │──→│ offsets  │   │
│  │ 每unit向 │   │ 逐级前缀 │   │ 高层级前 │   │ CSR/CSC  │   │
│  │ 量统计非 │   │ 和      │   │ 缀加回低 │   │ I32/I64  │   │
│  │ 零数    │   │ level0→ │   │ 层级    │   │ 末端值+  │   │
│  │ →level0 │   │ levelN  │   │         │   │ base     │   │
│  └─────────┘   └─────────┘   └─────────┘   └──────────┘   │
│                       │                                     │
│                       ▼                                     │
│                 ┌──────────┐                                │
│                 │ total    │ → nnz 槽 (workspace) ──→ D2H  │
│                 │ 末级末值  │   status 校验 nnz+base≤I32max  │
│                 └──────────┘                                │
│  (unitCount == 0: empty_offsets 填 base, COO 直接 nnz=0)    │
└─────────────────────────────────────────────────────────────┘
```

#### Convert 流程图

```text
┌────────────────── Device (AIV × blocks) ───────────────────┐
│                                                             │
│  CSR/CSC/COO:                                               │
│  ┌─────────┐   ┌─────────┐   ┌──────────┐   ┌──────────┐  │
│  │ count+  │──→│ total   │──→│ validate │──→│ convert  │  │
│  │ scan    │   │ (重算,  │   │ _offsets │   │ (流压缩)  │  │
│  │ (第二遍) │   │ 比对nnz)│   │ (与当前  │   │          │  │
│  └─────────┘   └─────────┘   │ offsets  │   └──────────┘  │
│                              │ 比对)    │                  │
│                              └──────────┘                  │
│  BELL:  ┌─────────────────────────────┐                    │
│         │ bell (单 kernel)            │                    │
│         │ pattern→task→b×b块搬运/正零 │                    │
│         └─────────────────────────────┘                    │
└─────────────────────────────────────────────────────────────┘
```

#### convert kernel 单 unit 内部（流压缩，128 元素子块）

```text
for each unit (4096 元素) 认领于本核:
  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
  │ DataCopy     │──→│ 向量非零判定  │──→│ 子块循环 ×32 │
  │ tile → UB    │   │ (掩符号位,   │   └──────┬───────┘
  │ (4096 elems) │   │  §3.4 谓词)  │          │
  └──────────────┘   └──────────────┘          ▼
                             ┌─────────────────────────────┐
                             │ ① ReduceSum(mask[sub]) → c_s │
                             │ ② GatherMask(values[sub])    │
                             │   → 压缩 values (≤128)       │
                             │ ③ GatherMask(iota[sub])      │
                             │   + base + minorBase → 索引  │
                             │ ④ DataCopy → GM 输出区        │
                             │   (unitOutStart + prefix_s)  │
                             └─────────────────────────────┘
  unitOutStart: CSR/CSC = offsets[major] + unit 级前缀差
                COO     = unit 级全局前缀
```

**count kernel（全向量化）**：每 unit 处理 4096 个元素：`DataCopy` tile 进 UB →
按 dtype 位宽视图做「掩符号位看剩余位」非零判定（§3.4）→ `Compare` 得 0/1 mask →
`ReduceSum` 得 unit 计数 → 写 GM level0。complex64 的 tile 以 u32 视图处理（8192B =
2048 u32），判定等效（见 §3.4）。UB 占用：tile 32KB + mask/规约临时 < 8KB，
远低于 arch22 可用 UB（实测约 192KB，见风险 2）。

**scan / addbase kernel（标量-in-UB）**：前缀和的数值可达 `nnz ≤ INT32_MAX`，但
u64 累加在 arch22 向量通路不可靠。设计：**级数组保持 u64（与 arch35 语义一致，
保留 `nnz > INT32_MAX` 的 Device 侧检出能力），但扫描在 UB 内以标量循环完成**
（每核认领连续段：`DataCopy` 段进 UB → `GetValue/SetValue` 累加 → `DataCopy`
写回 + 段和写下一级）。典型规模（P-01 CSR 57344 unit = 458KB）每核约 12KB、千次
量级标量操作，代价可忽略；count/convert 的重活仍在向量通路。**不使用 GM 标量
读+写混合访问**（arch22 实测该模式会丢写，见风险 2 引用的硬件约束集）。

**convert kernel（压缩写，本移植的核心难点）**：arch35 的 SIMT 逐元素
`if (nz) out++ 写` 在向量机上的等价物是**流压缩（stream compaction）**。arch22 无
vscatter，设计采用「**128 元素子块 GatherMask 压缩**」，要点：

- 输出区恰好由各 unit 顺序拼接，**每个输出元素只被一个 unit 写一次，无原子、
  无竞争 → bit-wise 确定性（R7）**；
- 每子块两次 GatherMask + 两次小 DataCopy，MTE/向量调用次数为
  `unitCount × 32` 量级，P-01 约 1.8M 子块 / 满核分摊，预算内（§3.7）；
- 宽 dtype 的 GatherMask 通道拆分（fp32 拆 2×16b、complex64 拆 4×16b，mask 复用）
  与单次调用包络（上个任务实测 repeat ≤ 128 可靠、256 输出错位）列为**上板首日
  探针项**（风险 2），不达标时回退到 128 元素标量内联循环（仅小矩阵兜底）。

**Dense layout × 格式的访存矩阵**：CSR+ROW、CSC+COL、COO+ROW 的 unit 读取是
`ld` 内连续段，直接 `DataCopy`；CSR+COL、CSC+ROW、COO+COL 为跨步访问，设计为
**向量构造索引 + vgather**（索引 `base + i*ld` 为等差数列，广播+iota 向量生成；
vgather 单次 8192 元素包络实测可靠），兜底方案为 workspace 内转置一遍（额外
一次读写，预算仍宽松，见 §3.7）。P-01/02/03 全部性能用例为 row 布局，主路径
不受影响。

**BELL kernel**：task = blockRow × slots + slot；pattern 批量 `DataCopy` 进 UB 后
逐 task 消费；有效 pattern（`base ≤ enc < base + ceil(cols/b)`）按 §3.5 几何读
b×b 块（ROW 布局 2D 搬运 + UB 内 16×16 `Transpose` 拼块；COL 布局天然列主序直写，
免转置），无效/越界槽位写正零；`ellValue` 每 task 恒占 `b×b` 连续标量，单次
`DataCopy` 写出。pattern 与 Dense 全程只读（R8）。

#### BELL kernel 流程图

```text
┌────────────────── Device (AIV × blocks, 单 kernel) ──────────────────┐
│                                                                       │
│  for task = blockIdx → tasks step blocks:   (task = br*slots + slot)  │
│                                                                       │
│    ┌────────────────┐    ┌────────────────────────────────┐          │
│    │ 读 pattern 项   │───→│ 有效? base ≤ enc < base+       │          │
│    │ ellColInd[task]│    │ ceil(cols/b)                   │          │
│    └────────────────┘    └───────┬───────────────┬────────┘          │
│                                    │ 是             │ 否               │
│                                    ▼                ▼                 │
│                    ┌──────────────────┐   ┌──────────────────┐      │
│                    │ 读 Dense b×b 块   │   │ 写 b×b 正零      │      │
│                    │ ROW: 2D DataCopy  │   │ (padding/越界)   │      │
│                    │ + UB Transpose    │   └──────────────────┘      │
│                    │ COL: 列主序直写   │                              │
│                    └────────┬─────────┘                              │
│                             │                                        │
│                    ┌────────▼─────────────────────────┐              │
│                    │ 尾块越界元素 (row≥rows 或         │              │
│                    │ col≥cols) → 写正零               │              │
│                    └────────┬─────────────────────────┘              │
│                             │                                        │
│                    ┌────────▼─────────────────────────┐              │
│                    │ DataCopy b×b → ellValue           │              │
│                    │ [task*b*b ...] (块内列主序)       │              │
│                    └──────────────────────────────────┘              │
└───────────────────────────────────────────────────────────────────────┘
```

### 3.4 非零语义（本算子的「精度」），含 complex64

统一成一条位运算规则：**去掉符号位，剩余幅值位非零即非零**：

```cpp
IsNonzero(int8)   : bits != 0
IsNonzero(u16)    : (bits & 0x7FFF) != 0        // fp16 / bf16
IsNonzero(u32)    : (bits & 0x7FFFFFFF) != 0    // fp32
IsNonzero(u64)    : (lo & 0x7FFFFFFF) | (hi & 0x7FFFFFFF) != 0   // complex64
```

complex64 是交织的 (re, im) 两个 fp32，两半字各自掩符号位后取或——**NaN 幅值位
本就非零，不需要 isnan 分支**；向量通路上把 8 字节元素视为两个 u32 字处理，
OR 语义与「任一分量非零或为 NaN 则保留」严格等价（R3）。Golden 按同一条谓词实现。

| 输入 | 是否入结构 | values 写什么 |
|---|---|---|
| `+0` / `-0`（浮点）、`0`（int8） | 否 | — |
| 普通有限非零 / `±Inf` / NaN（含 payload） | 是 | 原始 bit |
| complex64，任一分量幅值位非零 | 是 | 原始 8 字节 bit |
| complex64，两分量皆（正/负）零 | 否 | — |
| BELL pattern 无效/越界、尾块越界逻辑元素 | （容量固定） | **正零**（全字节 0） |

### 3.5 Blocked-ELL 尾块（R4，含公共描述符改动）

**现状**：公共 `aclsparseCreateBlockedEll()` 与 arch22 将移植的 host 校验都强制
`rows % b == cols % b == ellCols % b == 0`；容量 `ellValue = rows × ellCols`、
`nnz = rows × ellCols`。

**设计（与 A5 任务方案对齐，取「补齐块网格」方案 A）**：整除从**前置条件**改为
**布局特例**，几何量改用向上取整：

```text
blockRows = ceil(rows / b)        slots = ceil(ellCols / b)
ellColInd 元素数 = blockRows × slots
ellValue  标量数 = blockRows × b × slots × b     // 每 slot 恒占 b×b，寻址不分叉
valuePos = task * b * b + innerCol * b + innerRow // 与整除情形完全同式
```

「每个 slot 恒占 `b×b` 标量」的不变量在尾块上继续成立，kernel 越界判定退化为
`row < rows && col < cols` 一次比较，不引入第二套寻址公式。pattern 有效性上界随
`cols` 改用 `ceil(cols/b)`；越界/padding 槽位与尾块越界元素统一写正零（§3.4 表）。

整除输入下新旧容量与 `nnz` 完全一致，**不破坏现有行为**；不整除输入从
`INVALID_VALUE` 变为按尾块语义成功输出（能力扩展方向）。`aclsparseSpMatGetSize`
对 BELL 报的 `nnz` 随之变为 `blockRows*b*slots*b`，是对外可见的语义变化
（见风险 3）。

**公共代码改动清单**（与 A5 任务逐字对齐，见兼容性分析）：
`aclsparseCreateBlockedEll()` 放宽整除并改容量/nnz 公式；Golden 与 UT 公共部分同步
扩展。

### 3.6 Python/ATen 适配（R6）

仓内无先例，新建 `python/` 顶层目录，两层结构：

```text
python/
  ops_sparse/            # torch 层：四入口的 NPU 实现 + 绑定安装
  ext/                   # C++（pybind11 + torch 头）扩展模块：
                         #   TORCH_LIBRARY_IMPL(aten, <NPU key>, ...) 注册四算子
                         #   内部构造 aclsparse 描述符并驱动三阶段
  tests/                 # ATen UT + Python 端到端 UT
```

| 公开入口 | ATen 算子 | 目标格式 | 实现路径 |
|---|---|---|---|
| `Tensor.to_sparse()` | `aten::_to_sparse` | COO | 直接走三阶段（COO） |
| `Tensor.to_sparse_csr()` | `aten::_to_sparse_csr` | CSR | 直接走三阶段（CSR，base 0，I32） |
| `Tensor.to_sparse_csc()` | `aten::_to_sparse_csc` | CSC | 直接走三阶段（CSC） |
| `Tensor.to_sparse_bsr(bs)` | `aten::_to_sparse_bsr` | BSR | **BSR ≠ BELL**：块级结构发现（见下） |

- **适配层职责**：校验参数/dtype/shape/layout/stride/device（2D、步幅可表示为
  `ld`、dtype ∈ 五种）→ 在当前 NPU stream 上构造 `aclsparseConstDnMat` 与目标
  `aclsparseSpMat` → GetBufferSize/Analysis → 按 `nnz` 以 NPU allocator 分配
  payload → `SetPointers` → Convert → 以 `torch::from_blob` + deleter 或 DLPack
  包装为 Sparse Tensor（**payload 不经 Host 中转**，R10）→ 异常路径返回与
  ATen 语义一致的错误，**不回退 CPU**。

#### ATen 适配调用链流程图

```text
┌───────────────── Python ─────────────────┐
│  Tensor.to_sparse / to_sparse_csr /      │
│  to_sparse_csc / to_sparse_bsr           │
└──────────────────┬───────────────────────┘
                   ▼
┌───────────── Dispatcher (NPU key) ───────┐
│  aten::_to_sparse / _to_sparse_csr /     │
│  _to_sparse_csc / _to_sparse_bsr         │
└──────────────────┬───────────────────────┘
                   ▼
┌───────────────── C++ 适配层 ─────────────────────────────┐
│  ┌──────────┐   ┌──────────────────┐   ┌──────────────┐ │
│  │ 参数校验   │──→│ 构造描述符        │──→│ GetBufferSize │ │
│  │ dtype/    │   │ DnMat(A) +      │   │ (查询)        │ │
│  │ shape/ld/ │   │ SpMat(B)        │   └──────┬───────┘ │
│  │ stride/   │   └──────────────────┘          │         │
│  │ device    │                        ┌────────▼───────┐ │
│  └──────────┘                        │ Analysis       │ │
│                                      │ → nnz (D2H 8B) │ │
│  BSR 额外前置:                        └────────┬───────┘ │
│  块占用矩阵 → 本算子 CSR 路径          ┌────────▼───────┐ │
│  (块级 offsets/indices)              │ NPU 分配 payload │ │
│                                      │ + SetPointers  │ │
│                                      └────────┬───────┘ │
│                                      ┌────────▼───────┐ │
│                                      │ Convert        │ │
│                                      └────────┬───────┘ │
│                                      ┌────────▼───────┐ │
│                                      │ 包装 Sparse    │ │
│                                      │ Tensor (NPU)   │ │
│                                      └────────────────┘ │
└──────────────────────────────────────────────────────────┘
         （全程 NPU stream，无 CPU fallback，payload 不经 Host）
```
- **`to_sparse_bsr` 映射**（与 A5 任务同题）：BSR 按数据发现非零块并带
  `crow_indices/col_indices`，与「pattern 预置、不发现结构」的 BELL 不是同一格式。
  设计采用块级 CSR 路径：以块为粒度复用本算子的 count kernel 生成
  `ceil(m/b)×ceil(n/b)` 块占用矩阵（int8）→ 调 `aclsparseDenseToSparse`（CSR，
  int8）得到块级 offsets/indices（**调用链仍经过本算子**）→ 派生自 BELL kernel
  的块搬运 kernel 聚合 b×b values。全程 NPU。
- **派发注册以实机探测为准**（上个任务的教训：实际缺失符号与后端 key 实名和文档
  推断不同）：第一步在 910B3 上跑派发探测脚本，确认 `aten::_to_sparse*` 的真实
  派发链、缺失符号与 torch_npu 的 NPU key 实名，再定注册清单。
- 版本：PyTorch 2.7+、torch_npu 26.0.0+（注意 torch_npu `26.1.0` 是 Ascend 发布
  列车号，PyPI 以配套 PyTorch 版本号发布，不为凑版本号降 PyTorch）。

### 3.7 性能设计：两类标杆、两种瓶颈

任务书 §3.3 口径：性能倍率 = 标杆 GPU 设备 Event 调用耗时 / NPU 同调用范围全部
Kernel 总耗时，目标 ≥ 0.25（即 **NPU 预算 = 4 × 标杆**）。两份标杆独立、互补、
不混算（已核对任务包 `gpu_performance_result_benchmark.md` 与
`baseline_results/gpu_full_results.tsv`，**两者数值一致**；标杆设备按
`manifest.json` 为 A100）：

| 标杆来源 | 覆盖格式 | P-01 median_us | P-02 | P-03 |
|---|---|---|---|---|
| 原生 cuSPARSE（C++，10 条/场景） | 仅 BELL | 34.688–39.808 | 31.424–37.312 | 34.432–39.616 |
| PyTorch GPU（30 条/场景） | CSR/CSC/COO | 4156.656–48839.441 | 234.384–1319.840 | 382.704–2638.320 |

**BELL：小数据量、固定开销支配。** 三场景 ellCols=64（每行 64 非零），实际搬运
0.5–8.4MB。最紧预算 = 4×31.424 ≈ 126us（P-02），最紧带宽 ≈ 60GB/s（P-01
complex64，8.4MB / 138.8us）。对策：GetBufferSize 返回 0、Analysis 不启 kernel、
Convert 保持**单 kernel 一次启动**；核心路径零 D2H 同步、零 tiling 重算。

**CSR/CSC/COO：全量两遍扫描、带宽支配。** 按 0.25× 折算，全部 90 条 PyTorch 标杆
case 的单遍有效读带宽需求约 7–75GB/s（P-02），最紧为 P-01 coo-complex64：
1.879GB × 2 遍 / 26.0ms ≈ **144GB/s**。910B3 HBM 带宽在其上，**可达但两遍都必须
接近纯流式顺序读**，没有余量做低效访存。是否缓存非零掩码换一遍扫描？掩码
`m×n/8`（P-01 29MB）会使 workspace 与 Dense 成比例——仅 BELL 受 §3.4 条件 2 约束，
CSR/CSC/COO 走条件 1（额外内存 ≤ GPU 总用量 50%），数值上放得过（A100 侧
P-01 CSR 额外峰值 2.13GB，掩码仅 29MB），但两遍流式读已满足预算，**本设计不缓存
掩码**，保持 workspace 只含计数/前缀等索引量。

**优化顺序**（吸取上个任务教训）：第一步就用 `msprof --task-time=on` 拿逐 kernel
归因，不靠阶段级计时反推；已知候选：scan 级数与 kernel 数（kScanChunk 已从 1024
提至 4096）、convert 子块合并写（128→更大粒度 DataCopy）、COL/ROW 跨步组合的
vgather 效率。kernel 间零间隙（arch22 实测），多 kernel 流水线的代价可控。

### 3.8 workspace 与生命周期（R10）

沿用 arch35 语义：workspace 是每次调用的临时 scratch，不保存跨阶段身份，Analysis
与 Convert 可复用同一地址也可各用各的；`bufferSize == 0` 时可传空。workspace 只含
status/nnz 槽与多级 u64 前缀（索引量），与 values 宽度无关。核心路径除 Analysis
的 8B 标量回读外无任何 Host 中转；ATen 层输出构造同样不落 Host。描述符只借用
Device 指针：先同步 stream，再销毁描述符、释放内存。

## 支持硬件

| 芯片 | 支持情况 |
|---|---|
| Atlas 800I A2（910B3 / 910B4） | 支持（arch22 / DAV-2201），CANN 9.1.0+ |
| Atlas A3（任务环境提供型号，910_93 系列） | 支持（arch22 / DAV-2201） |

自验证优先 910B3；交付含 910B3、910B4、A3 三型号的功能/精度/性能报告。

## 算子约束限制

1. `alg` 仅支持 `ACL_SPARSE_DENSETOSPARSE_ALG_DEFAULT`。
2. CSR/CSC 的 offset 与 element index 类型必须相同；I32/I64、I64/I32 混合可创建
   描述符但本算子返回 `NOT_SUPPORTED`。
3. `matA` 与 `matB` values 类型必须相同；不支持 float64 与 complex128。
4. BELL 不发现结构：pattern 未覆盖的 Dense 非零被忽略，不扩充、不重排 pattern。
5. Convert 不校验 Analysis 时的快照，只消费调用时的当前描述符字段；Device 侧检出
   的不一致置 status，不转为同步错误返回，调用者须同步 stream 后读取。
6. 描述符只借用 Device 指针，生命周期由调用方保证。
7. ATen 入口支持 2D、可 `ld` 化步幅、五种 dtype；`to_sparse_bsr` 块大小语义对齐
   PyTorch；不支持的组合返回明确错误，不回退 CPU。

---

# 可维可测分析（required）

## 精度标准/性能标准

**精度：exact match，无容差**（任务书 §3.2，参考 opbase experimental_standard）：
CSR/CSC/COO 的 nnz、坐标、indices、values 与 CPU Golden 完全一致；五种 dtype 保持
非零元素 bit 模式；complex64 实/虚部分别 bit-wise 一致。Golden 用 CPU 实现 §3.4
谓词表，因不做算术无需加宽类型。

**性能：各自标杆的 0.25 倍以上**，口径为「标杆 GPU 设备 Event 调用耗时 / NPU 同
调用范围全部 Kernel 总耗时」；C++ 标杆与 PyTorch 标杆分别对齐来源与调用范围；
每 case 预热 ≥ 10 次、正式采样 ≥ 30 次；报告中位数、p90、Analysis、Convert、完整
三阶段、端到端、实际 nnz、稀疏度、workspace 峰值和各 Kernel 耗时。

**内存**：§3.4 二选一。CSR/CSC/COO 走条件 1：用任务包
`collect_sparse_ops_{gpu,npu}_memory.py` 采集、`compare_sparse_ops_memory.py` 按
case id 对比（额外峰值 ≤ GPU 50%）；BELL 走条件 2：`GetBufferSize` 恒 0，固有
workspace 不超过 L2 容量，输出 payload 字节数另行报告。

## 测试策略

分四层，从不依赖硬件到完整上板：

**第一层：主机侧算法验证（无需 NPU）。** 把非零谓词、unit 分解、BELL 尾块寻址、
输出基址计算抽成主机可编译头文件，与 Golden 对拍。移植类 bug（unit 边界、base
偏移、尾块几何）在这一层秒级定位，是 SIMT→Ascend C 移植正确性的第一道闸。

**第二层：C++ UT**（`test/densetosparse/arch22/`，镜像 arch35 结构 + SOC 网关）：

| 覆盖项 | 用例要点 |
|---|---|
| 五 dtype × 四格式 × base 0/1 × ROW/COL | 全组合矩阵抽样 + 手算小例 |
| complex64 | 实/虚部独立 `±0`、Inf、NaN 组合；bit 级比对 |
| BELL 尾块 | rows/cols/ellCols 分别/两两/三者不整除；b=16/32/64；越界正零 |
| BELL padding | `-1`、小于 base、1-based 的 `0`、超上界 |
| 零语义/边界 | `±0`、全零、全非零、零维、空指针/描述符、非法 shape/dtype/format/index/base/ld/alg、溢出 |
| 三阶段状态 | 跳阶段、pointer rebinding、workspace 复用与不足、`bufferSize==0` 传空 |
| 确定性/只读性 | 重复执行 bit-wise 比对；Dense 与 pattern 前后逐字节比对 |
| guard 区 | 输出缓冲前后哨兵检测越界写 |

**第三层：任务包用例。** 200 条精度用例（ATK，判定 exact）+ 320 条性能用例 +
内存三脚本。**尽早接通**（上个任务的教训：官方脚本与自测口径到交付前才交叉
验证）。

**第四层：ATen 端到端与 Profiler 证据。** 四入口的参数/dtype/shape/layout/
stride/device 校验、异常、输出语义；NPU Dispatch 与 Profiler 证明全部 kernel 在
AI Core 上、无 CPU fallback。

## 兼容性分析

**对现有行为的影响。** arch22 为全新目录，不触碰 arch35 任何行为。公共代码改动
三处（`aclsparseCreateBlockedEll` 尾块、公共 Golden/UT 参数扩展、README 支持矩阵）
与 I32/I64/dtype 门禁注释——**与 A5（950）任务的需求重叠，双方改动取齐到逐字相同
以便自动合并**；提 PR 前用 `git merge-tree --write-tree` 预演双方分支，冲突集中在
公共文件时以语义等价、文本相同为原则处理，并跑 A2/A3 与 A5 双向交叉回归
（任务书 §3.5、§5）。

**SOC 网关。** 测试框架按 SOC 版本网关跳过不适型号（沿用仓内 `skipped_tests.list`
机制），arch22 目标 SOC 为 `ascend910b*` 与 `ascend910_93*`。

**版本兼容。** CANN 9.1.0 及后续配套；PyTorch 2.7+、torch_npu 26.0.0+。

---

# 风险点

| # | 风险 | 影响 | 应对 |
|---|---|---|---|
| 1 | SIMT→Ascend C 移植的正确性（unit 分解、压缩写、scan） | 结构错误即 exact mismatch | 第一层主机对拍 + guard 区 + 小例手算；kernel 与 arch35 同名同构便于逐段比对 |
| 2 | arch22 硬件微观约束（GatherMask 包络、vgather 形态、UB ~192KB、GM 标量读+写丢写、无 vscatter） | 压缩写/跨步读路径返工 | 上板首日跑探针脚本（上个任务已有探针基建与实测定律集）；每条路径均有兜底（标量子块、workspace 转置） |
| 3 | BELL 尾块 `nnz`/容量语义变化波及公共描述符 | 对外语义变更 + 与 A5 冲突 | 方案 A 保证整除输入行为不变；与 A5 逐字对齐；提交评审确认（见风险表） |
| 4 | `to_sparse_bsr` 与 BELL 语义不等价 | ATen 入口覆盖与验收口径 | 块级 CSR 路径保调用链；派发探测先行；备选明确报不支持并说明 |
| 5 | 两遍全量扫描带宽（最紧 144GB/s）+ COL 跨步组合 | 性能不达标 | 两遍均流式化；msprof 逐 kernel 归因后再优化；跨步走 vgather/转置双路备选 |
| 6 | 多 kernel launch 开销 vs BELL ~126us 预算 | BELL 性能风险 | BELL 严格单 kernel、零同步；CSR/CSC/COO 预算宽松（ms 级） |
| 7 | A3 具体型号环境晚到位 | 三型号报告进度 | 架构同源（arch22），先 910B3 全量自验，A3 到位后补跑功能/精度/性能 |

# 交付物

1. **算子设计文档**：本文件，PR 至 `cann-ops-competitions`
   `04_tasks/01_community-task-2026/tasklist/09-aclsparseDenseToSparse-A2A3/zhanghua145/docs/design.md`
   （upstream 尚无该任务目录，按仓内 `09-*` 惯例自建，编号待评审调整）。
2. **代码**（PR 至 `ops-sparse` `master`，任务书 §5 结构）：
   `include/cann_ops_sparse.h` dtype 支持范围注释；`sparse/densetosparse/arch22/`
   host/kernel/tiling；`sparse/common/aclsparse_descr.cpp` BELL 尾块（与 A5 对齐）；
   `sparse/densetosparse/README.md`；`test/densetosparse/arch22/` 与公共目录；
   `python/` ATen 适配与 e2e UT。
3. **自测用例与测试代码**：四层测试、复现 README（环境/编译/步骤，验收人可复现）。
4. **自测报告**：用例参数、输出结构与精度结果、性能数据（aclsparse Kernel 与
   Python/ATen 端到端双口径）、峰值内存、截图、Profiler 证据、失败项说明；
   910B3/910B4/A3 三型号 + A5 交叉回归记录。
