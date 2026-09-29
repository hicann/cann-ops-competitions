# 需求背景（required）

## 需求来源

CANN 社区任务：算子实操工坊-广州站 — aclblasCgeru 算子开发（A2/A3）。

- 任务书：《aclblasCgeru_Atlas800IA3_task_doc.md》（aclblasCgeru A2/A3 算子开发任务书）
- 目标合入仓库：https://gitcode.com/cann/ops-blas （算子目录 `blas/ger/`，测试目录 `test/ger/cgeru/arch22/`）
- 对标基线：cuBLAS `cublasCgeru`；语义参考 Netlib BLAS `cgeru`（https://www.netlib.org/blas/cgeru.f ）
- 精度标准来源：生态算子开源精度标准 experimental_standard.md

## 背景介绍

### aclblasCgeru算子实现路径

基于 ops-blas 开源仓的 Ascend C 工程框架，使用 Ascend C kernel 直调方式，在昇腾 NPU（Atlas 800I A2 / Atlas 800I A3 系列产品，对应架构目录 arch22，SOC ascend910b3/ascend910b4）上实现单精度复数（complex64）无共轭秩 1 更新算子 `aclblasCgeru`。通过 handle 绑定 stream 直调 NPU kernel，接口为句柄式 BLAS 接口。

算子实现路径：`blas/ger/arch22/`

| 文件 | 职责 |
| --- | --- |
| `blas/ger/arch22/cgeru_host.cpp` | host 侧：参数校验、no-op 快速返回、分核与 tiling 计算、进程级常驻常量表管理、kernel 选择与启动 |
| `blas/ger/arch22/cgeru_tiling_data.h` | tiling 结构体 `CgeruTilingData` 定义（按值作为 kernel 参数传递） |
| `blas/ger/arch22/cgeru_kernel.cpp` | 通用 kernel 入口 `cgeru_kernel`（支持任意 incx/incy） |
| `blas/ger/arch22/cgeru_kernel_fast.cpp` | AOS 快速 kernel 入口 `cgeru_kernel_fast` 与 alpha=(+1,+0) 编译期特化入口 `cgeru_kernel_fast_unit` |
| `blas/ger/arch22/cgeru_kernel_impl.h` | kernel 侧全部计算流水：通用路径 `Process`/`ComputeColumn`/`LoadXChunk` 与快速路径 `ProcessFast<kAlphaUnit>` |
| `blas/ger/README.md` | 算子参数说明与产品支持表（Atlas A2/A3：支持） |
| `include/cann_ops_blas.h` | 新增 `aclblasCgeru` 接口声明（与同族 `aclblasCgerc` 同形） |

### 同类算子现状分析

通过对仓内同族算子的分析，当前相关能力如下：

| 算子 | 语义 | 仓内现状（ops-blas） |
| --- | --- | --- |
| `aclblasSger`（实数秩 1 更新） | `A = alpha·x·yᵀ + A` | 已有实现：`blas/ger/arch22/sger_host.cpp`、`sger_kernel.cpp`，具备完整 host 校验、tiling 按值传参、全异步启动约定与 CSV 测试工程 |
| `aclblasCgerc`（复数共轭秩 1 更新） | `A = alpha·x·yᴴ + A`（y 取共轭） | 已有实现及测试工程，但 arch22 版本为简化实现（tiling 不含 lda/incx/incy、host 校验不全），不能直接满足本任务的步长/padding/异常参数要求 |
| `aclblasCgeru`（复数无共轭秩 1 更新） | `A = alpha·x·yᵀ + A`（**y 不取共轭**） | 仓内原无实现、无接口声明，本次新增 |

本算子以 sger 的 arch22 工程为骨架（完整参数校验、按列切分、tiling 按值传参、全异步），复数计算在 kernel 侧独立实现；随性能调优演进为"通用 kernel + AOS 快速 kernel + alpha 单位特化 kernel"三入口形态。geru 与 gerc 的唯一语义差异为 y 是否取共轭，实现与 golden 均按"不共轭"落实。

### aclblasCgeru算子功能分析

算子功能：`A = alpha * x * y^T + A`。

- 输入：复数标量 alpha（Host 内存）、复数向量 x（长度 m，Device 内存）、复数向量 y（长度 n，Device 内存）、复数矩阵 A 的旧值（m×n，列主序，Device 内存）
- 输出：A 原地覆写为新值
- 支持数据类型：COMPLEX64（实部/虚部各为 FLOAT32）
- 支持正/负步长 incx/incy（负步长反向索引）、lda ≥ max(1,m) 的列间 padding
- 不支持广播：x/y/A 为独立操作数，无广播语义

# 需求分析（required）

## 需求描述

使用 Ascend C 编程语言在 ops-blas 仓内实现 `aclblasCgeru` 句柄式 BLAS 接口：通过 handle 绑定 stream 直调 NPU kernel，完成 complex64 无共轭秩 1 更新 `A = alpha * x * y^T + A`；实现代码放在 `blas/ger/`（arch22），接口声明新增至 `include/cann_ops_blas.h`，测试代码合入 `test/ger/cgeru/arch22/`；精度满足 COMPLEX64（实部/虚部按 FLOAT32）开源精度标准，性能不高于任务书标杆耗时。

## 需求拆解

1. 新增 `include/cann_ops_blas.h` 中 `aclblasCgeru` 接口声明，参数序列与 `cublasCgeru` 一一对应，不定义产品私有平行接口。
2. host 侧实现：参数合法性校验、m=0/n=0/alpha=(0,0) no-op 快速返回、分核与 tiling 计算、kernel 选择与启动；接口全异步（不做 stream 同步、无每次调用的 device 端 tiling 分配与 H2D 拷贝）。
3. kernel 侧实现复数秩 1 更新，y **不取共轭**；覆盖任意 incx/incy（含负步长）、lda padding 与 m/n 全形状范围。
4. 边界行为对齐 cuBLAS/Netlib：零维与 alpha=(0,0) 返回 `ACLBLAS_STATUS_SUCCESS` 且不写 A；非法参数返回 `ACLBLAS_STATUS_INVALID_VALUE` 等对应状态码。
5. 精度（rtol=2^-10、atol=2^-16、matched_ratio ≥ 0.99、max_abs_error ≤ 1e-2 或 32ULP）与性能（512²/1024²/2048²/4096² 四标杆）达标。
6. 提供算子 README（产品支持表标注 Atlas A2/A3 支持）与 CSV 驱动的 GTest 自验工程。

# 详细设计（required）

## 算子分析

### 数学公式

```
A = alpha * x * y^T + A
```

设 alpha = αr + j·αi，x[i] = xr(i) + j·xi(i)，y[j] = yr(j) + j·yi(j)。y **不取共轭**，逐元素语义为：

```
A[i][j] = alpha * x[i] * y[j] + A[i][j]    (i = 0..m-1, j = 0..n-1)
```

实现按"先算列系数 beta(j) = alpha·y[j]，再与整列 x 相乘"组织（每列一次复数标量乘、每行复用）：

```
beta_r(j) = αr·yr(j) − αi·yi(j)
beta_i(j) = αr·yi(j) + αi·yr(j)

A[i][j].real = beta_r(j)·xr(i) − beta_i(j)·xi(i) + A[i][j].real
A[i][j].imag = beta_r(j)·xi(i) + beta_i(j)·xr(i) + A[i][j].imag
```

> 与 Cgerc 的差异：gerc 对 y 取共轭，实部/虚部的交叉项符号相反。本算子实部为"减"、虚部为"加"，kernel 计算与 cblas golden 均按此落实。含步长时按 incx/incy 索引：x 逻辑下标 i 对应物理下标 i·incx，y 逻辑下标 j 对应物理下标 j·incy，负步长从尾部反向索引。

### 支持数据类型

| 数据 | 类型 | 说明 |
| --- | --- | --- |
| alpha | COMPLEX64（`aclblasComplex`，实部/虚部各 float32） | Host 内存标量指针，取值为 FLOAT32 全集 |
| x、y | COMPLEX64，ND 格式 | Device 内存只读向量，支持正/负步长 incx/incy |
| A | COMPLEX64，ND（列主序） | Device 内存矩阵，输入旧值、原地覆写输出 |

复数类型以 `include/cann_ops_blas_common.h` 中的定义为准：

```cpp
typedef struct aclblasComplex {
    float real;
    float imag;
} aclblasComplex;
```

### 支持形状

- x 逻辑长度 m，物理长度 `1 + (m-1)·|incx|`；
- y 逻辑长度 n，物理长度 `1 + (n-1)·|incy|`；
- A 维度 lda×n（列主序），仅更新前 m×n 部分，lda ≥ max(1,m)；
- m、n 为运行时入参，不要求 dynamic shape 编译期特化；无广播。
- kernel 侧对单行长度设上限分块：incx==1 时每块最多 MAX_COMPLEX=4096 个复数；|incx|>1 时受跨步 32B 槽位限制每块最多 STRIDE_CHUNK_COMPLEX=1024 个复数，外层按行块循环覆盖任意 m。

### 参数说明（对应任务书 §2.4）

| 参数 | 输入/输出 | 内存位置 | dtype/格式 | shape | 异常行为 |
| --- | --- | --- | --- | --- | --- |
| handle | 输入 | Host | 句柄（携带 stream） | - | nullptr → `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`；stream 未初始化 → `ACLBLAS_STATUS_NOT_INITIALIZED` |
| m | 输入 | Host | int 标量 | - | m<0 → `INVALID_VALUE`；m=0 为合法 no-op |
| n | 输入 | Host | int 标量 | - | n<0 → `INVALID_VALUE`；n=0 为合法 no-op |
| alpha | 输入 | Host | COMPLEX64 标量指针 | - | nullptr → `INVALID_VALUE`；(0,0) 为 no-op，不写 A |
| x | 输入（只读） | Device | COMPLEX64 / ND | 逻辑长 m，物理 1+(m-1)·\|incx\| | 需计算时为 nullptr → `INVALID_VALUE` |
| incx | 输入 | Host | int 标量 | - | =0 或 INT_MIN → `INVALID_VALUE`，支持负值 |
| y | 输入（只读，**不取共轭**） | Device | COMPLEX64 / ND | 逻辑长 n，物理 1+(n-1)·\|incy\| | 需计算时为 nullptr → `INVALID_VALUE` |
| incy | 输入 | Host | int 标量 | - | =0 或 INT_MIN → `INVALID_VALUE`，支持负值 |
| A | 输入/输出（原地） | Device | COMPLEX64 / ND（列主序） | lda×n，更新前 m×n | 需计算时为 nullptr → `INVALID_VALUE` |
| lda | 输入 | Host | int 标量 | - | < max(1,m) → `INVALID_VALUE` |

返回值 `aclblasStatus_t` 以 `include/cann_ops_blas_common.h` 定义为准。

## 算子实现

### 实现方案

总体采用 host 侧按列划分（列主序下同一核负责的列区间连续且互不重叠），kernel 侧提供三个编译入口，由 host 根据形状与 alpha 位模式在启动时静态选择：

| kernel 入口 | 触发条件 | 算法特征 |
| --- | --- | --- |
| `cgeru_kernel_do`（通用路径） | 其余全部形状（任意 incx） | x 实部/虚部分离计算，行方向分块（4096/1024），支持任意 \|incx\|>1 与负步长 |
| `cgeru_kernel_fast_do`（AOS 快速路径） | incx==1 且 m ≤ 4096 | x 全程保持交错（AOS），预构建旋转向量 Jx，每列仅 2 条 2m-lane 稠密向量 op |
| `cgeru_kernel_fast_unit_do`（单位 alpha 特化） | 快速路径条件 **且** alpha 位级精确等于 (+1.0f, +0.0f) | `ProcessFast<true>` 模板特化，beta=y，每列 4 乘 2 加标量运算编译期消除 |

#### 3.2.1 host侧设计：

**tiling策略：**

（1）接口校验与 no-op 快速返回

`aclblasCgeru` 校验顺序镜像 sger，避免对零值/负值先做运算再报错：

| 顺序 | 校验项 | 判定条件 | 返回码 |
| --- | --- | --- | --- |
| 1 | handle | 为 nullptr | `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| 2 | 零维 quick return | m==0 \|\| n==0 | `ACLBLAS_STATUS_SUCCESS`（不启动 kernel、不写 A） |
| 3 | 维度 | m<0 或 n<0 | `ACLBLAS_STATUS_INVALID_VALUE` |
| 4 | 步长 | incx==0 \|\| incx==INT_MIN；incy 同理 | `ACLBLAS_STATUS_INVALID_VALUE` |
| 5 | alpha 指针 | 为 nullptr | `ACLBLAS_STATUS_INVALID_VALUE` |
| 6 | alpha quick return | alpha->real==0 且 alpha->imag==0 | `ACLBLAS_STATUS_SUCCESS`（不写 A） |
| 7 | 数据指针 | x/y/A 任一为 nullptr | `ACLBLAS_STATUS_INVALID_VALUE` |
| 8 | lda | lda < max(1,m) | `ACLBLAS_STATUS_INVALID_VALUE` |
| 9 | stream | handle 携带 stream 为空 | `ACLBLAS_STATUS_NOT_INITIALIZED` |
| 10 | 常驻常量表 | 首次分配/上传失败 | `ACLBLAS_STATUS_ALLOC_FAILED` |

（2）tiling 按值传参（sger 约定）

tiling 结构体 `CgeruTilingData` 直接以 `const CgeruTilingData&` 按值作为 kernel 参数下发，**不分配 device 端 tiling 缓冲、不做每次调用的同步 H2D 拷贝**：

```cpp
struct CgeruTilingData {
    uint64_t A;             // 矩阵 A 设备基址
    uint64_t x;             // 向量 x 设备基址
    uint64_t y;             // 向量 y 设备基址
    uint32_t m;             // A 行数 / x 逻辑长度（按复数元素计数）
    uint32_t n;             // A 列数 / y 逻辑长度
    uint32_t lda;           // A 前导维（列主序），lda >= max(1, m)
    uint32_t useCoreNum;    // 实际参与 AIV 核数
    uint32_t colsPerBlock;  // 每核负责列数 = ceil(n / useCoreNum)
    int32_t  incx;          // x 步长，负值反向索引
    int32_t  incy;          // y 步长，负值反向索引
    float    alphaReal;     // alpha 实部
    float    alphaImag;     // alpha 虚部
    uint32_t rowBands = 1;  // 二维分块行带数（1 为纯按列一维划分）
    uint32_t colGroups = 0; // 二维分块列组数（0 表示取 useCoreNum）
};
```

所有维度/步长字段均按**复数元素**计数（1 个复数 = 2 个交错 float32）。

（3）进程级常驻常量表（96KB，单例）

host 维护函数内静态单例 `CgeruOffsetTable`，首次调用时一次性 `aclrtMalloc` + H2D 上传，进程内所有 handle/stream 共享、直至进程退出才释放；表内容只依赖 MAX_COMPLEX，与输入无关。共三段：

| 设备偏移 | 长度 | 内容 | 使用者 |
| --- | --- | --- | --- |
| 0 ~ 32KB | 8192 个 u32 | 重交错表：`offset[2i]=4i`、`offset[2i+1]=4(i+4096)`（字节偏移），把分离的 [实部半区\|虚部半区] Gather 回交错 [r0,i0,r1,i1,…] | 通用 kernel |
| 32 ~ 64KB | 8192 个 u32 | Jx 旋转表：`jx[2i]=8i+4`、`jx[2i+1]=8i`，从交错 x 中取出 [xi0,xr0,xi1,xr1,…] | AOS 快速 kernel |
| 64 ~ 96KB | 8192 个 float | 循环符号模式 [-1,+1,-1,+1,…]，全掩码张量乘把 [xi,xr,…] 变为 Jx=[-xi,xr,…]，避免逐 lane 掩码向量指令 | AOS 快速 kernel |

> 用全掩码张量乘替代逐 lane 掩码 Muls，是因为 ascend910b3 上掩码类向量指令会在向量单元留下粘滞状态，可能毒化后续 kernel launch（实测硅片缺陷规避）。

（4）kernel 选择与全异步启动

- `fastEligible = (incx == 1) && (m <= 4096)`；
- 单位 alpha 判定按位精确比较：real 位模式 `0x3F800000` 且 imag 位模式 `0x00000000`（-0.0f 与 NaN 位模式故意不匹配）；
- 启动函数：非 fast → `cgeru_kernel_do`；fast 且单位 alpha → `cgeru_kernel_fast_unit_do`；其余 fast → `cgeru_kernel_fast_do`；
- 启动参数仅 x/y/A 基址、常量表设备指针、tiling（按值）、blockDim、stream；
- **API 完全异步**：host 不调用 `aclrtSynchronizeStream`，连续 launch 依赖 stream ordering 背靠背执行，调用方在读 A 前自行同步（任务书 §2.5）。

##### 1. 分核策略：

总体为按列均分：`colsPerBlock = ceil(n / useCoreNum)`，核 b 负责列区间 `[b·colsPerBlock, min((b+1)·colsPerBlock, n))`，列区间互不重叠；`blockIdx·colsPerBlock ≥ n` 的尾核直接退出。列主序下每核的列区间天然连续，且不存在跨核写同一 A 元素。

核数由 `CalCgeruCoreNum(m, n)` 依据同窗口 A/B 实测调优结果分层确定（基础上限取 `GetAivCoreCount()`，910B3 为 40 核），最终 `coreNum = min(n, cap)`：

**分层 cap 规则（按判定顺序）：**

| 条件 | cap | 调优依据（摘要） |
| --- | --- | --- |
| n ≤ 64 | min(aiv, 24) | 列少时 32/40 核闲置且增加固定 prologue 争用 |
| n ≤ 256 且 m·n ≤ 64K | min(aiv, 24) | 总工作量小，24 核最优（256² 4.24→3.79us） |
| n ≤ 512 且 m·n ≥ 700K | aiv（40） | 大面积形状满核领先 3~5% |
| 其余 n ≤ 512 | min(aiv, 32) | 512² 等形状 32 核稳定优于 40 核 |
| m%32==0 且 m ≤ 256 且 1024 ≤ n < 2048 | min(aiv, 24)（S4-9 窄带） | 短对齐行+约千列时每核列数过少，固定 prologue 摊薄不足，40→24 约 −8~−16% |
| m%32==0 且 128 ≤ m ≤ 256 且 2048 ≤ n < 4096 | min(aiv, 32)（S5-5） | (256,2048) 40→32 约 −10% |
| m%4==0 且 400 ≤ m ≤ 600 且 860 ≤ n < 1100 | min(aiv, 32)（S5-5） | (540,1092)、(492,862) 类形状 −5~−9% |
| 257 ≤ m ≤ 280 且 1900 ≤ n < 4096 | min(aiv, 32)（S5-5） | (276,1983) 约 −6%；区间刻意收窄以防更重行数回退 |
| m ≤ 128 且 1400 ≤ n < 1500 | min(aiv, 32)（S5-5） | (103,1421) 约 −4% |
| 其余 n > 512 | aiv（40） | 列绑定型，满核最优 |

**精确 (m,n) 形状键覆盖（在 min(n,cap) 基础上直接改写 coreNum）：**

| 形状键 (m,n) | coreNum | 说明 |
| --- | --- | --- |
| (64, 2048) | 32（S6-3） | 40 核划分严重不均（39×52+1×20），32×64 整除，约 −10.2% |
| (1342, 690) | 30（S6-3） | 30×23 整除，约 −3.3% |
| (136, 2678) | 34（S7-5） | 同窗口 A/B 中位 −3.0% |
| (3943, 114) | 24（S7-5） | m=3943 单核 prologue 极重，32→24 约 −7.2%，仍保证每核 ≥4 列 |
| (540, 1092) | 30（S9-2） | 40 核墙长由 28 列尾核决定；30 核每核 37 列且 prologue 更轻，安静窗中位 −1.37% |

**二维列组×行带分块（S3-10，仅快速路径）：**

对高瘦窄矩阵，一维按列划分的固定 prologue（x/Jx 表搬运、Jx 构建、|x| 归约）相对收益变差。当 `fastEligible && m%4==0` 且命中以下两个窗口时启用 `colGroups × rowBands = min(20,n) × 2` 网格：

- 高瘦：m ≥ 3800 且 n ≤ 64（如 4096×64，约 −7%）；
- 高宽：m ≥ 3200 且 160 ≤ n ≤ 256（如 4096×256 约 −12.5%、3272×192 约 −6.5%）。

行带 0 覆盖 `segM0 = (m/2) & ~7` 行，行带 1 覆盖其余 `m − segM0` 行；`blockIdx % colGroups` 选列组、`blockIdx / colGroups` 选行带。两行带处理的 A 行区间不相交，因此**不需要原子加**；窗口外形状保持一维划分（双倍 prologue 会使二维更慢）。

##### 2. 数据分块和内存优化策略：

- **行方向分块（通用路径）**：incx==1 连续搬运时单块 4096 个复数；|incx|>1 跨步 DataCopyPad 每个 8B 复数占一个 32B 目的槽，UB 占用放大 4 倍，单块收缩为 1024 个复数；m 任意大时外层 `rowOff` 循环逐块覆盖。
- **列方向分组（快速路径）**：每核列块在 ubBulk（62KB）内按 64-lane 向量颗粒打包成组，组内先批量预取旧 A、再逐列 2 条 Axpy、组末批量写回；组容量 `groupCap = regionBytes / granuleBytes`。当整块列 tile 超过半区且单 tile 能放入半区时，ubBulk 拆成两个 31KB ping/pong 半区，组 k 的 MTE3 写回与组 k+1 的 MTE2 读入并行（独立 free flag EVENT_ID7/EVENT_ID2）；全部列能放入整块 62KB（仅 1 组）时退回单缓冲，避免无谓拆区。
- **x 一次搬入全列复用**：x 对一个核负责的所有列完全相同，核内只搬一次；快速路径额外一次性构建 Jx 旋转向量供全部列复用。
- **y 标量预取**：incy==1 且空间满足时，将本核连续 y 段一次搬入 UB 后用 GetValue 读取，否则每列经 S 管标量 GM 读（负 incy 按 `(n-1-j)·|incy|` 反向索引）。快速路径受 ascend910b3 短 MTE2 拷贝毒化后续 launch 的硅片问题约束，y 段只允许暂存在已验证干净的 ubX 区尾部（门限折合 m ≲ 4000），独立 y 窗口保留布局但不启用。
- **无 per-call device 内存**：tiling 按值传参，常量表进程级常驻，单次调用不发生 `aclrtMalloc/Memcpy`，启动路径仅寄存器配置与 enqueue。
- **UB 192KB 全量利用**：两条路径各自静态划分 192KB UB（详见 kernel 侧设计），缓冲偏移针对 ascend910b3 实测硬件约束规避（Gather 目的不能与源共享 64KB 区、不能落在 32KB 边界；长向量槽不跨 64KB 边界）。

##### 3. tilingkey规划策略：

不使用运行时 tilingKey 寄存器字段。host 感知到的分支信息在 **launch 时静态分发到三个独立 `__global__` kernel 入口**（等价于编译期 tilingKey：0=通用、1=AOS 快速、2=单位 alpha 特化），单位 alpha 特化进一步以模板常量 `kAlphaUnit` 消除每列标量运算分支，运行时零分支开销；二维分块通过 tiling 字段 `rowBands/colGroups` 表达，kernel 内仅做 blockIdx 映射，无额外 tilingKey。无数据类型多态（仅 COMPLEX64）、无广播分支。

#### 3.2.2 kernel侧设计：

三个 kernel 入口均为 `__global__ __aicore__ __vector__` 向量核程序，入口先 `AscendC::SetMaskNorm()` 将向量掩码置为正常全掩码模式，退出前 `PIPE_BARRIER(ALL)`。快速 kernel 与通用 kernel 分属不同翻译单元（`cgeru_kernel_fast.cpp` 定义 `CGERU_FAST_KERNEL` 后再 include 实现头），防止调度器把某一分支的 WAIT 信号量跨分支提升。

**（1）Init 阶段**

- 由 kernel 参数（按值 tiling）设置 GM Tensor（x/y/A，通用路径另设重交错表 Tensor，快速路径在常量表基址 +32KB/+64KB 处分别设置 Jx 表与符号模式 Tensor）；
- 取 `blockIdx`，计算本核列起始 `colStart = colBlock·colsPerBlock` 与列数 `colCount = min(colsPerBlock, n-colStart)`，越界核直接退出；
- 二维分块时按行带平移 x/A 基址并把 m 收缩为本带行数（segM0 为 8 对齐、两行带长度均为 4 的倍数，保持快速路径全部对齐不变量）。

**（2）通用路径 Process —— CopyIn / Compute / CopyOut**

UB 布局（192KB）：

```
[  0,  32KB) ubIn           分离式 x：实部半区 [0,4096) / 虚部半区 [4096,8192)（float）
[ 32KB, 64KB) ubGatherOffset 重交错偏移表（仅载入 maxCalNum*2 项）；尾部 [48KB,64KB)
                             为本核 y 预取窗口（16KB，incy==1 且容量满足时启用）
[ 64KB, 96KB) ubOutPing      每列交错输出 ping
[ 96KB,128KB) ubOutPong      每列交错输出 pong
[128KB,160KB) ubCalcPing     每列分离计算 ping（Gather 后复用为 A 旧值 tile）
[160KB,192KB) ubCalcPong     每列分离计算 pong
```

CopyIn（行块外层循环，每块一次）：

- incx==1：`DataCopyPad` 连续搬入交错 x；
- incx>1：跨步 `DataCopyPad`（blockLen=8B、每复数落一个 32B 槽、srcGap=(|incx|−1)·8B），再在 UB 内压实为连续复数；
- incx<0：按物理地址正序跨步搬运（起点 `(m-rowOff-calNum)·|incx|`），在 UB 内翻转为逻辑正序；
- 随后两条 `GatherMask` 把实部、虚部分离到 ubSep 的两个连续半区（半区间隔 HALF_SPLIT_OFFSET=4096 floats）。

Compute（核内逐列，功能正确优先、列间以 ALL barrier 串行）：

- 取 y[j] 复数（yInUb 时从 UB 窗口 GetValue，否则 S 管 GM 标量读），SET/WAIT_FLAG(S,V,EVENT_ID1) 保证标量就绪；
- 标量复数乘 beta = alpha·y[j]（不共轭）：`br=αr·yr−αi·yi`、`bi=αr·yi+αi·yr`；
- 4 段 `muls_v` + `sub_v`/`add_v`（显式 ASCEND_V220 simd intrinsics，V 管按序发射，RAW 由硬件消解）：
  `real = br·xr − bi·xi`、`imag = br·xi + bi·xr`；
- `Gather` 借重交错表把 [r0,r1,…,i0,i1,…] 还原为交错 [r0,i0,r1,i1,…]。

CopyOut（UB 内 read-modify-write，**不使用硬件原子加**）：

- SET/WAIT_FLAG(V,MTE2) 后 `DataCopyPad` 把 A 旧值列块搬入分离计算缓冲（该缓冲在 Gather 后已死亡，直接复用），SET/WAIT_FLAG(MTE2,V)；
- fp32 `add_v` 在 UB 内完成 旧A＋新贡献；
- SET/WAIT_FLAG(V,MTE3) 后 `DataCopyPad` 写回列主序第 j 列（地址 `(j·lda + rowOff)·2` float 起，连续 calNum 个复数）；ping/pong 使用 EVENT_ID3/ID4 与 ID5/ID6 两组事件。

> 不用硬件 float atomic-add 的原因：硬件 float 原子加保留的尾数位过少，在随机数据数百列累加、强抵消场景丢精度；列区间核间互不重叠，UB 内读-改-写既可保证精度又无竞争。

**（3）AOS 快速路径 ProcessFast<kAlphaUnit> —— CopyIn / Compute / CopyOut**

采用 chpr 模板的 AOS（交错）算法，x 全程不分离，每列计算降为 2 条稠密 2m-lane 向量 op。

UB 布局（192KB，偏移依据 ascend910b3 实测硬件约束选定）：

```
ubJx   @ 256          旋转向量 Jx=[-xi0,xr0,…]（≤32KB），全列复用
ubY    @ 32KB+512      独立 y 窗口（保留布局；因硅片拷贝毒化问题默认不启用）
ubBulk @ 64KB          按列打包的稠密槽区（62KB），组内旧 A 直接预取入槽
ubX    @ 126KB+256     交错 x（≤32KB，连续搬入），尾部富余空间可暂存 y 段
ubScr  @ 158KB+256     先作 |x| 归约 scratch，后作 rescue 缩放/选择 scratch（≈34KB）
```

CopyIn / 预处理：

- 连续搬入整段交错 x；yInUb 门控满足时把本核 y 段搬到 ubX 区尾部 x 之后；
- `BuildRotatedX`：载入 Jx 表与符号模式（按 64-float 颗粒整粒度载入，残余 lane 为死 lane），一次 `Gather` 得到 [xi0,xr0,…]，再与常驻 [-1,+1] 模式做全掩码张量 `Mul` 得到 Jx=[-xi0,xr0,…]，一次构建、全部列复用；
- rescue 能力门控（m ≤ RESCUE_MAX_M=1024）：`ComputeMaxAbsInterleaved` 在 2m 个 lane 上求 max(|xr|,|xi|)——末块残余 lane 由 S 管补零后，`Abs` 长度取整到 32B，再把归约长度向上取整到 64-lane 整倍数，用 2201 跨块硬件归约 `BlockReduceMax`（vcmax，每 64 lane 折叠为 8 lane，非整倍数尾块会被丢弃）逐级折叠，不足 64 lane 的尾（1/2/4… 个 8-float 块）由 `VecBlockMax` 成对 Max 树 + S 管收尾；结果只取指数位 `maxAbsExpBits`。

Compute（组内逐列）：

- 取 y 标量并计算 beta；`kAlphaUnit=true` 时 beta 直接等于 y（4 乘 2 加在编译期消除，运行时零标量 flop）；
- 旧 A tile 已在组预取阶段直接装入该列 ubBulk 槽，列计算只有两条 Axpy：
  `slot += x · beta_r`（2m lanes）；`slot += Jx · beta_i`（2m lanes），
  交错布局下实部 lane 自然得到 `A_r + br·xr − bi·xi`、虚部 lane 得到 `A_i + br·xi + bi·xr`。

组预取/写回按 m 对齐度分两支：

- `m%32==0`（2m 为 64 的倍数）：分组预取模式，旧 A 用一次分组 `DataCopyPad` 直接落入各列槽（本片硅忽略 UB 侧非零 gap，故 srcGap/dstGap=0），每列严格 2 条稠密 op；
- 否则逐列预取：当 tileBytes 与列间 GM gap 均 32B 对齐（m%4==0 且 lda%4==0）时用一次普通 PLAIN `DataCopy` 双侧 gap  fan-in，否则逐列单发 `DataCopyPad`；
- 组末 SET/WAIT_FLAG(V,MTE3,ID4) 后批量 `DataCopyPad`/`DataCopy` 写回本组各列，并置 MTE3→MTE2 free flag 放行下一组（首组前置 SET_FLAG 配对，末组 WAIT 平衡，防止 SET_FLAG 不配对累积导致 AIV 事件硬件挂死）。

rescue 防溢出分支（极端大值用例，如官方 m=n=32 场景；纯整数指数域判定，阈值 `rescueThrExp = 378 − maxAbsExpBits`）：

- 当 max(beta 实,beta 虚) 指数位超过阈值时，`scaleDn = betaExpBits − rescueThrExp`（钳到 ≤127），仅用指数位构造 2 的幂 `scaleDnF=2^(−s)`、`scaleUp=2^s`（无 libm）；
- 在槽内旧 A 未被破坏前先算缩放结果 `scl = oldA + scaleUp·(betaS·x + betaS·Jx)`；
- 正常分支结果若全部有限（`Abs` + `Compares(LE, 0x7F7FFFFF)` 生成掩码，`Select` 按 lane 选择有限的正常结果、溢出 lane 取 scl），再 `muls(1.0)` 写回槽；
- rescue 的 Compares/Select 是快速路径唯一的逐 lane 掩码操作，kernel 退出前重新 `SetMaskNorm()` 恢复全掩码模式。

**（4）事件与流水小结**

| 路径 | 关键同步 |
| --- | --- |
| 通用 | ID1 S→V（y 标量就绪）；ID3/ID4、ID5/ID6 两组 V↔MTE2 乒乓（当前功能路径固定用 ping 组）；ID0 V→MTE3；列间 ALL barrier 串行 |
| 快速 | ID7/ID2 MTE3→MTE2 组 free（首组预置、末组 WAIT 平衡）；ID3 MTE2→V；ID4 V→MTE3；组前一条 ALL barrier 排干可选 y 搬入并保证 S 管首次 GetValue 顺序；ping/pong 两 31KB 半区并行 |

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas 800I A2（910B3） | √ |
| Atlas 800I A3（910B4） | √ |
| Atlas 800T A2（910B3，性能测试设备） | √ |

- CANN 版本：CANN 9.1.0（任务书 §3.1）；
- 对应 SOC 架构目录 arch22（ascend910b3/ascend910b4），CMake 按 SOC 自动选择架构目录。

## 算子约束限制

| 约束项 | 内容 |
| --- | --- |
| 参数合法性 | m ≥ 0、n ≥ 0；incx ≠ 0 且 ≠ INT_MIN、incy 同理；lda ≥ max(1,m)；alpha 不可为 nullptr；非法参数返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| 无共轭语义 | y 不取共轭（与 gerc 的共轭转置相区别），kernel 与 golden 公式均按实部"减"、虚部"加"落实 |
| 数据类型 | 仅支持 COMPLEX64（实部/虚部各 FLOAT32），单类型无多态分支 |
| 非连续 Tensor 支持 | 仅支持 incx/incy/lda 语义内的跨步与列间 padding，不支持除此之外的非连续内存访问 |
| broadcast 规则 | 不涉及，x/y/A 为独立操作数，无广播 |
| dynamic shape | 不要求编译期特化，m/n/lda/incx/incy 均为运行时入参 |
| 原地与视图语义 | A 原地覆写（UB 内读旧值-相加-写回），不返回视图 |
| 空 Tensor 与 0 维 | m=0 或 n=0 合法 no-op 返回成功且不计算；alpha=(0,0) 不写 A |
| 异步执行 | 接口全异步，host 不做 stream 同步；依赖 `aclblasSetStream` 绑定 stream，调用方读回 A 前须同步 |
| 资源约束 | UB 192KB；进程级常驻常量表 96KB；快速路径限 incx==1 且 m ≤ 4096（其余形状走通用路径，通用路径支持任意 m/incx） |
| 并发写 | 按列（二维分块时按列组×行带）划分，各核写区间互不相交，不使用硬件原子加 |

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述（不涉及说明原因） | 标准来源 |
| --- | --- | --- |
| 精度标准 | golden 由 cblas（Netlib `cgeru`，y 不取共轭）单标杆生成，输出 A 全矩阵验证，实部、虚部分别按 FLOAT32 比对；逐元素 `\|actual − golden\| ≤ atol + rtol·\|golden\|`，用例同时满足 matched_ratio ≥ 0.99 且 max_abs_error ≤ 1e-2（或 32ULP）判定通过；浮点乘加非 bit-exact | 任务书 §3.2；生态算子开源精度标准 experimental_standard.md |
| 精度参数 | COMPLEX64 实部/虚部分量：rtol = 2^-10（9.77e-4），atol = 2^-16（1.53e-5），required_matched_ratio = 0.99，max_abs_error_limit = 1e-2 或 32·ULP | 同上 |
| 性能标准 | Atlas 800I A2（910B3）、COMPLEX64，warmup 后有效采样 >50 次取平均单次耗时：m=n=512（incx=incy=1）≤ 5.7us；1024 ≤ 10.1us；2048 ≤ 55.71us；4096 ≤ 219.35us | 任务书 §3.3 |
| 内存要求 | 不涉及（任务书 §3.4）；实现侧无 per-call device 分配，仅进程级 96KB 常驻常量表 | 任务书 §3.4 |

**自验方案（test/ger/cgeru/）：** 使用仓内 CSV 驱动 GTest 框架——`cgeru_param.h` 解析用例参数、`cgeru_golden.h` 以 cblas cgeru（不共轭）生成 golden、`arch22/cgeru_npu_wrapper.h` RAII 管理设备缓冲并封装调用、`arch22/cgeru_test.cpp` 加载 `cgeru_test.csv` 执行精度/性能用例。覆盖：

| 类别 | 覆盖内容 |
| --- | --- |
| 基础用例 | m/n=0、1、小质数、2 的幂及 ±1、非对齐值；incx/incy = ±1/±2/±3 全组合；alpha 含 (0,0)、(1,0)、纯虚数、大值 |
| shape 扫描 | m×n 矩形组合直至 4096；lda 紧凑（=max(1,m)）与多 padding 场景 |
| 边界/负向 | 空指针 x/y/A、incx/incy=0 与 INT_MIN、负维度、lda 越界、handle/stream 异常，状态码与 cuBLAS 行为对齐 |
| 特殊值 | x/y/A 的 Inf/NaN、全零、交替、极端值填充（含触发 kernel rescue 分支的大值用例） |
| 性能用例 | 4 个标杆 case，先 warmup 再采样 >50 次取平均 |

性能采样支持 `CBLAS3_PERF_MODE=1` 环境变量与 `TC_PF_` 用例前缀触发性能模式（跳过精度校验、隔离计时），采用探测自适应迭代次数与冷却窗口，输出标准化 `[PERF]` 记录供验收脚本解析；测试步骤见 `test/ger/cgeru/README.md`。

## 兼容性分析

1. **接口兼容**：`aclblasCgeru` 为新增接口，声明加入 `include/cann_ops_blas.h`，与同族 `aclblasCgerc` 声明形式一致、与 950PR 同名接口共用同一声明，不定义产品私有平行 API，不破坏既有 ABI；参数序列与 `cublasCgeru` 一一对应，无需参数映射。
2. **平台兼容**：实现位于 `blas/ger/arch22/`，CMake 按 SOC 自动选择架构目录，arch35 等其他架构目录不受影响；README 产品支持表标注 Atlas A2/A3 系列产品支持。
3. **数据布局兼容**：仅支持 complex64 ND（列主序），与仓内既有复数算子（cgerc/cgemv 等）布局约定一致，无新增布局假设。
4. **精度兼容**：实部/虚部分别按 FLOAT32 标准判定，不引入跨列累加顺序假设；原地更新采用 UB 内 fp32 读-改-写而非硬件低尾数原子加，保证强抵消场景精度；浮点乘加非 bit-exact 属于任务书明确允许的范围。
5. **运行时兼容**：接口全异步且无 per-call device 内存分配，与 stream ordering 语义及既有 BLAS 句柄使用模式一致；常驻常量表进程级共享、引用计数无关，多 handle/多 stream 并发调用安全。
