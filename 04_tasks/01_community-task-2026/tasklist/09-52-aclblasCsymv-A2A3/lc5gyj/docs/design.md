# aclblasCsymv 算子设计文档

## 一、需求背景

### 1.1 需求来源

通过**社区任务**完成开源仓算子贡献的需求。在昇腾 NPU（Atlas A2 / A3，`arch22` / DAV_2201）上使用 Ascend C 开发 `aclblasCsymv`（complex64 对称矩阵-向量乘）算子，对齐 cuBLAS `cublasCsymv` / Netlib `ssymv`（复数路径按实数语义扩展）语义，纳入 `ops-blas` 开源仓。

- 任务书算子：`aclblasCsymv`
- 目标硬件：Atlas A2 / A3 系列产品（arch22 / DAV_2201，CANN 9.1.0）
- 参考标杆：cuBLAS `cublasCsymv`（语义参考 Netlib `ssymv`）
- 代码仓：`ops-blas`（Ascend BLAS 算子库），源码路径 `blas/symv/arch22/`

### 1.2 背景介绍

#### 1.2.1 aclblasCsymv 算子实现优化

本任务在 `ops-blas` 开源仓中新增 `aclblasCsymv` 算子实现，属 Level-2 级 BLAS 矩阵-向量乘算子（symmetric matrix-vector multiply），对齐 cuBLAS 语义进行原生实现。

**参考资源路径：**

- 接口定义（算子信息库 / 头文件）：`ops-blas/include/cann_ops_blas.h`（`aclblasCsymv` 声明**当前不存在**，需新增；参考同仓已有 `aclblasSsymv` 声明，第 152 行）
- 算子实现目录：`ops-blas/blas/symv/arch22/`（与现有 `aclblasSsymv` 实现同目录）
- 测试目录：`ops-blas/test/symv/csymv/arch22/`
- 语义参考：cuBLAS `cublasCsymv`（`cublas_docs/cublas_symv.md`）、Netlib BLAS `ssymv`（`https://www.netlib.org/blas/ssymv.f`）
- 同族已有实现（复用参考）：`ops-blas/blas/symv/arch22/ssymv_{host,kernel}.cpp`（实数版对称矩阵-向量乘，AIV 分块 + 行轮转分核 + `SetAtomicAdd` 范式）、`ops-blas/blas/gemv/arch22/ascblasCgemv_utils.h`（complex64 Level-2 解交织/重交织/复数标量乘范式）

#### 1.2.2 aclblasCsymv 算子现状分析

`ops-blas` 仓内已有实数版 `aclblasSsymv`（对称矩阵-向量乘），但缺少复数版 `aclblasCsymv`。本任务补齐该接口。复数版与实数版的核心差异在于：复数乘法（4 次实数乘 + 2 次实数加）、复数点积（实/虚两路归约）与复数元素存储（每元素实部/虚部各 float，需解/重交织）。另需注意同族 `arch22/ssymv_kernel.cpp` **只实现了 LOWER 分支**（`colLen = row + 1` 固定），本算子必须自行补齐 UPPER。

##### 1.2.2.1 标杆算子支持的数据类型和数据格式

与算子信息库/接口声明保持一致：

| 项 | 取值 |
|----|------|
| 输入类型 | `aclblasComplex`（complex64，float real + float imag，8 字节对齐） |
| 输出类型 | `aclblasComplex`（complex64 向量） |
| 矩阵布局 | A 为 n×n 列主序（Column-Major）复数对称矩阵（A = A^T，**不取共轭**），lda×n 物理布局，仅 `uplo` 指定三角被引用，另一三角由对称性推断 |
| 向量布局 | x、y 逻辑长度 n，支持任意步长 incx/incy（含负步长） |
| 索引语义 | `y[i] = alpha * Σ_j A_sym[i][j] * x[j] + beta * y[i]`，列主序 A 元素 `A(i,j) = A[i + j*lda]`（0-based） |

##### 1.2.2.2 标杆算子实现描述

cuBLAS `cublasCsymv` / Netlib `ssymv`（复数扩展）计算定义：

```
y = alpha * A * x + beta * y
```

按 Netlib `ssymv` 逐列展开（UPPER/LOWER 仅决定扫描方向）：

```
对每条列 j (0..n-1)：
  temp1 = alpha * x[j]；temp2 = 0
  UPPER：for i = 0 .. j-1    (列 j 上方元素)
  LOWER：for i = j+1 .. n-1  (列 j 下方元素)
      y[i] += temp1 * A(i,j)        # 复数 axpy
      temp2 += A(i,j) * x[i]        # 复数点积
  y[j] += temp1 * A(j,j) + alpha * temp2
```

- 上三角（UPPER）：主对角及上方元素 `A[i][j]`（j≥i）直接存储；下方元素由对称性用 `A[j][i]`（j<i）
- 下三角（LOWER）：主对角及下方元素 `A[i][j]`（j≤i）直接存储；上方元素由对称性用 `A[j][i]`（j>i）
- 复数乘法：`(a+bi)(c+di) = (ac-bd) + (ad+bc)i`
- **对称而非厄米特**：推断未存储三角时直接取对称元素、**不对虚部取共轭**（非 chemv 的 A = A^H）

**边界语义（对齐 cuBLAS / Netlib）：**

- `n = 0`：合法 quick return，返回 SUCCESS 不执行计算
- `alpha = (0,0)` 且 `beta = (1,0)`：quick return，不写 y
- `alpha = (0,0)` 且 `beta ≠ (1,0)`：退化为 `y = beta*y`，不读 A、x
- `beta = (0,0)`：y 输入值不被使用（可为无效输入）
- 负步长（`incx<0`/`incy<0`）：合法，从反向起点遍历
- 参数校验失败返回 `INVALID_VALUE` / `HANDLE_IS_NULLPTR`

##### 1.2.2.3 标杆算子实现流程图

```
┌────────────────────────────────────────────┐
│        aclblasCsymv (对标入口)             │
└───────────────────┬────────────────────────┘
                    ▼
        ┌───────────────────────────┐
        │  参数校验                 │
        │  uplo/handle/incx/incy/   │──→ INVALID_VALUE / HANDLE_IS_NULLPTR
        │  lda/alpha/beta/null 检查 │
        └────────────┬──────────────┘
                     ▼
        ┌───────────────────────────┐
        │  quick return ?           │──yes──→ 返回 SUCCESS（不计算/不写 y）
        │  n==0 或 (α=0 且 β=1)     │
        └────────────┬──────────────┘
                     ▼ no
        ┌───────────────────────────┐
        │  α=0 且 β≠1 ?             │──yes──→ y = β·y（不读 A、x）
        └────────────┬──────────────┘
                     ▼ no
        ┌───────────────────────────┐
        │  β≠1：先 y = β·y          │
        └────────────┬──────────────┘
                     ▼
        ┌───────────────────────────┐
        │  逐列 j = 0 .. n-1         │
        │    temp1 = α·x[j]         │
        │    temp2 = 0              │
        │    UPPER: i=0..j-1        │
        │    LOWER: i=j+1..n-1      │
        │      y[i] += temp1·A(i,j) │
        │      temp2 += A(i,j)·x[i] │
        │    y[j] += temp1·A(j,j)   │
        │           + α·temp2       │
        └────────────┬──────────────┘
                     ▼
               返回 SUCCESS
```

## 二、需求分析

### 2.1 外部组件依赖

| 依赖组件 | 版本/说明 | 用途 |
|----------|-----------|------|
| ACL (AscendCL) | CANN 9.1.0 | 设备内存管理、stream、kernel launch |
| AscendC kernel 运行时 | CANN 9.1.0 | `__gm__` 地址空间、`DataCopy/DataCopyPad`、`Mul/Muls/Add/ReduceSum` 向量指令、`SetAtomicAdd` |
| GTest | 系统已装 | 参数化测试框架 |
| CBLAS/LAPACK | 系统已装 | 实数 ssymv 对照（复数 golden 自实现） |

### 2.2 内部适配模块

| 内部模块 | 说明 |
|----------|------|
| `common/helper/aclblas_handle_internal.h` | handle 结构、stream 绑定 |
| `common/helper/kernel_constant.h` | UB 容量、buffer 数量等常量 |
| `common/helper/host_utils.h` | host 侧工具函数 |
| `log/log.h` | 日志输出（`OP_LOGE`/`OP_LOGD`） |
| `test/frame`、`test/utils`（fill.h） | 测试框架、数据填充 |

### 2.3 需求模块设计

#### 2.3.1 Ascend C 算子原型

```c
aclblasStatus_t aclblasCsymv(
    aclblasHandle_t handle,        // aclblas 句柄
    aclblasFillMode_t uplo,        // ACLBLAS_UPPER / ACLBLAS_LOWER
    int n,                         // 对称矩阵 A 的行/列数
    const aclblasComplex* alpha,   // 复数标量 alpha
    const aclblasComplex* A,       // 复数对称矩阵（列主序，lda×n）
    int lda,                       // A 前导维度
    const aclblasComplex* x,       // 输入向量 x（设备指针）
    int incx,                      // x 步长
    const aclblasComplex* beta,    // 复数标量 beta
    aclblasComplex* y,             // 输入/输出向量 y（设备指针）
    int incy                       // y 步长
);
```

#### 2.3.2 Ascend C 算子相关约束

与标杆算子（cuBLAS `cublasCsymv`）相比，当前设计的功能范围：

| 功能点 | cuBLAS | 本设计 | 说明 |
|--------|--------|--------|------|
| 数据类型 | complex64 | complex64 | 一致 |
| 对称语义 | symmetric (A=A^T) 不共轭 | 对称不共轭 | 一致，非 Hermitian |
| uplo 分支 | UPPER/LOWER | UPPER/LOWER | 一致（本算子自实现双分支） |
| 正负步长 | 支持 | 支持 | 负步长按 Netlib 语义 |
| alpha/beta 特殊值 | 支持 | 支持 | no-op / 退化分支 |
| 前导维 lda | 任意 ≥ max(1,n) | 支持 | 含 padding |
| 非连续 Tensor | 任意视图 | 不支持超 lda/inc 语义的视图 | 任务书明确本批次不支持 |

### 2.4 参数校验顺序与错误码

多参数同时非法时，按下列固定顺序校验（实现与用例据此对齐）：

| 序 | 检查项 | 返回 |
|----|--------|------|
| 1 | `handle == nullptr` | `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| 2 | `uplo` 非法（非 UPPER/LOWER） | `ACLBLAS_STATUS_INVALID_VALUE`（对齐用例 `TC_ED_190`；任务书正文写 `INVALID_ENUM`，见 §3.4 说明） |
| 3 | `n < 0` / `lda < max(1,n)` / `incx == 0` / `incy == 0` / `alpha == nullptr` / `beta == nullptr` | `ACLBLAS_STATUS_INVALID_VALUE` |
| 4 | `n == 0` | 立即 `SUCCESS`（不启动 kernel、不读 A/x/y） |
| 5 | `alpha == (0,0)` 且 `beta == (1,0)` | 立即 `SUCCESS`（不写 y） |
| 6 | `alpha ≠ (0,0)` 时才要求 A、x 非空；需写回时才要求 y 非空 | `ACLBLAS_STATUS_INVALID_VALUE` |

所有长度、负步长起点与地址运算以 **64 位**计算；对 `inc < 0` 先扩宽为 int64 再取负，避免 `INT_MIN` 取绝对值溢出。空指针为**条件化检查**：仅在实际需要解引用时才判定（如 α=(0,0) 时不要求 A/x 有效）。

## 三、需求详细设计

### 3.1 调用方式

本任务采用 **Kernel 直调（ops-blas 内部 API）** 方式：

- 上层通过 `aclblasCsymv(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy)` 调用
- host 侧（`csymv_host.cpp`）完成参数校验、quick return 判定、tiling 计算，调用 `csymv_kernel_do` 启动 kernel
- kernel 侧（`csymv_kernel.cpp`）执行对称矩阵-向量乘
- 输出 y 为设备指针，由调用方负责 D2H 拷贝

### 3.2 需求总体设计

整体采用**多核 AIV 行轮转分核**架构（参照同仓 `aclblasSsymv` 范式，扩展复数支持）：

```
host: 参数校验 → quick return 判定 → tiling 计算 → launch kernel
kernel: [核0: y = β·y 折叠] → [各核行轮转计算 A*x，原子累加写 y]
```

#### 3.2.1 host 侧设计

##### 3.2.1.1 分核策略

- 按输出向量 y 的行维度分核，**行轮转**：`for (row = vecIdx; row < n; row += useCoreNum)`
- 使用核数 `useCoreNum = min(n, 8)`，对齐同族 `ssymv`/`cgemv` 的 `numBlocks = 8` 惯例（同族 tiling 预留 `SYMV_MAX_CORE_NUM = 50` 上限）
- 非单位步长（`incx!=1 || incy!=1`）：`useCoreNum = 1`（单核，避免非对齐标量访存）

> **设计取向说明**：本算子沿用同族"固定 `numBlocks = 8`"的仓内惯例，而非"运行时查询平台 AIV 核数"。理由：与同族 `ssymv`/`cgemv` 保持实现一致、tiling 数值确定可复现；且性能用例规模（n≤4096）下 8 核已足够。若评审要求跨代际自适应核数，可改为经 `aclrtGetDeviceInfo` 查询向量核数后取 `min(查询值, n)`。

##### 3.2.1.2 数据分块和内存优化策略

**UB 规划（每核）：** 单核内以行为最外层单位，行内沿列方向（存储连续段）按 `maxDataCount` 分块，双 buffer 流水（`BUFFER_NUM = 2`）。

同族 `ssymv` 块大小 `maxDataCount = 30*1024/4 = 7680`（float 元素/块，即每 buffer 30 KB 预算）。复数化后每 buffer 需实/虚两平面，UB 预算按下式核算：

| Buffer | 内容 | 大小（复数） |
|--------|------|-------------|
| A_tile | 当前行块 A 实/虚平面 | 2 × maxDataCount × 4B |
| x_tile | x 实/虚平面（列段） | 2 × maxDataCount × 4B |
| z_tile | 点积/axpy 结果实/虚平面 | 2 × maxDataCount × 4B |

```
Σ(buffer 容量) ≤ UB_SIZE      // UB_SIZE 由运行时 GetCoreMemSize(UB) 获取，禁止硬编码
```

> 任务书未规定 UB 大小，本设计以平台运行时查询值为准；块大小初值参照同族 30 KB 预算，按复数缓冲数（实/虚双平面）下调。

**workspace 规划：** 各核行轮转写 y 存在跨核冲突，采用 `SetAtomicAdd<float>()` 原子累加消解，**无需跨核归约 workspace**。

**数据访问：** A 列主序，`A(row,col)` 物理地址 `A + row*2 + col*lda*2`（float 计），同列元素连续、同行按 lda 跨步；复数元素按实部/虚部分离平面处理（解交织 `vreducev2`、重交织 `Gather` + host 预生成 mask）。

##### 3.2.1.3 tilingKey 规划策略

本算子为单形态（对称矩阵-向量乘）算子，**不引入 tilingKey 分支**（对齐同族，运行期 `if` 处理 uplo/步长/退化分支）。tiling 数据（`CsymvTilingData`）整体由 host 计算后随 kernel 参数传入：

```cpp
struct CsymvTilingData {
    uint32_t n;            // 矩阵维度
    uint32_t lda;          // 前导维度
    uint32_t uplo;         // ACLBLAS_UPPER / ACLBLAS_LOWER
    uint32_t useCoreNum;   // 使用核数
    uint32_t scaleOnly;    // 1: 仅做 y=β·y（α=0 且 β≠1），不读 A/x
    float alphaReal;       // alpha 实部
    float alphaImag;       // alpha 虚部
    float betaReal;        // beta 实部
    float betaImag;        // beta 虚部
    int64_t incx;          // x 步长（含符号）
    int64_t incy;          // y 步长（含符号）
};
```

##### 3.2.1.4 alpha/beta 标量位置（Host/Device）处理与评审点

任务书 §参数表要求 `alpha`、`beta` 的指针可位于 **Host 或 Device 内存**（原文："指向复数标量乘数的指针，Host/Device 内存"）。

**现状与张力**：仓内最接近的复数 Level-2 同族 `blas/gemv/arch22/cgemv_host.cpp`（第 82–85 行）**直接以 Host 指针解引用** `alpha->real/imag`、`beta->real/imag`；仓内 `_aclblas_handle` 只含 stream/workspace，**无 BLAS pointer-mode 字段或设置接口**。即任务书的"Host/Device 双位置"要求与仓内既有实现惯例存在真实张力。

本设计的处理：

- **默认路径（Host 标量）**：与同族一致，host 侧直接读取 `alpha/beta` 的实/虚部写入 tiling（`alphaReal/alphaImag/betaReal/betaImag`），kernel 经 tiling 取标量，无需在 Device 侧访问标量指针。
- **Device 标量路径（拟）**：host 侧先用 `aclrtPointerGetAttributes` 查询指针 `location.type`——普通栈/malloc → `UNREGISTERED`(2)、`aclrtMallocHost` → `HOST`(0)、`aclrtMalloc` → `DEVICE`(1)；判定为 `DEVICE` 时，先同步 handle 绑定的 stream（保证标量写入完成），将 8 字节标量 D2H 拷回 Host，再执行依赖标量值的 quick return / 条件空指针判定；一般计算仍异步提交同一 stream。查询失败直接映射运行时错误，**不解引用未知指针**，也不把查询失败的指针默认当 Host。
- **用例覆盖**：任务配套用例 `csymv_test.csv` 的 alpha/beta 均以标量值（`alpha_real/imag`、`beta_real/imag`）给出，**仅覆盖 Host 标量路径**，Device 标量路径不在给定用例范围内。

**提请评审确认**：当前公共句柄无 pointer-mode 字段。上述"自动识别指针位置"是否符合统一接口约定？若要求 Device 标量路径完全异步，则依赖标量内容的 quick return / 条件空指针错误码如何返回需先明确。本稿**不新增公共 pointer-mode API**。

#### 3.2.2 kernel 侧设计

##### 3.2.2.1 kernel 侧实现描述

kernel 采用 AIV-only（`KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY)`），行轮转分核，各核独立计算：

1. **原子加开启**：`SetAtomicAdd<float>()`（复数交织下逐 float 原子加等价于复数加法）
2. **β 缩放（核0 折叠）**：`β=(1,0)` 跳过；`β=(0,0)` 覆写 y=0（清 NaN）；其余 `y += (β-1)·y` 原子加 → `y = β·y`。`scaleOnly` 时仅核0 缩放后结束（不读 A/x）
3. **行轮转遍历**：`for row = vecIdx; row < n; row += useCoreNum`
4. **列段循环**：`LOWER: col∈[0,row]`；`UPPER: col∈[row,n-1]`，按 `maxDataCount` 分块
   - **复数点积**：`Σ A(row,col)·x[col] → 乘 α → y[row]`（实/虚两路 `ReduceSum`）
   - **复数 axpy**：`α·x[row]·A(row,col) → y[col]`（段含对角列时少写 1 个，避免与点积项重复累加）
5. **复数搬运**：CopyIn 解交织（交织 complex64 → 实/虚平面），CopyOut 重交织（实/虚平面 → 交织 complex64），全程不取共轭
6. **原子加关闭**：`SetAtomicNone()`

##### 3.2.2.2 Ascend C 实现流程图

```
┌────────────────────────────────────────────────────────┐
│                csymv_kernel                             │
│  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY)         │
└───────────────────────────┬────────────────────────────┘
                            ▼
        ┌────────────────────────────────────────┐
        │  vecIdx = GetBlockIdx()                │
        │  vecIdx >= useCoreNum ? → 直接返回      │
        └───────────────────┬────────────────────┘
                            ▼
        ┌────────────────────────────────────────┐
        │  SetAtomicAdd<float>()（跨核写 y 冲突） │
        │  核0：y = β·y 折叠                     │
        │    β=(1,0) 跳过 / β=(0,0) 覆写 0 +     │
        │    SyncAll / 其余 y += (β-1)·y         │
        │  scaleOnly：核0 缩放后结束（不读 A/x） │
        └───────────────────┬────────────────────┘
                            ▼
        ┌────────────────────────────────────────┐
        │  行轮转：for row = vecIdx; row<n;      │
        │           row += useCoreNum            │
        │   列段：LOWER col∈[0,row]              │
        │         UPPER col∈[row,n-1]            │
        │         （按 maxDataCount 分块）        │
        │   解交织 → 复数点积（→ y[row]）        │
        │          + 复数 axpy（→ y[col]，对角   │
        │            少写 1）                     │
        │   重交织 → 原子累加写 y                │
        └───────────────────┬────────────────────┘
                            ▼
        ┌────────────────────────────────────────┐
        │  SetAtomicNone()                       │
        └────────────────────────────────────────┘
```

##### 3.2.2.3 Ascend C 实现流程图与标杆算子流程图存在的差异点和原因

| 差异点 | 标杆（Netlib/cuBLAS） | Ascend C 本设计 | 原因 |
|--------|------------------------|-----------------|------|
| 循环结构 | 逐列标量循环（顺序累加） | 行轮转分块 + 向量化 | 多核并行 + 向量指令提升吞吐 |
| 行并行 | 单核串行 | 最多 8 核行轮转 | A2/A3 多核并行降低延迟 |
| 复数乘 | 通用复数乘 | 实/虚平面分离 + 向量化（4 实数乘 + 2 加） | 利用 AIV 向量指令，避免标量复数指令 |
| β*y 融合 | 单独循环 | 核0 折叠 `(β-1)·y` 原子加 | 减少 y 二次访问，天然得到 β·y |
| β=0 清 NaN | — | 核0 覆写 y=0 + SyncAll | 折叠法无法清 NaN（用例 `beta0_y_nan` 要求） |
| α=0 退化 | 早退返回 | `scaleOnly` 只做 y=β·y、不读 A/x | 用例 `alpha0_scale` 要求 A/x 可为空 |
| 对称访问 | 逐元素判断 uplo | 行内按 UPPER/LOWER 统一列区间 | 避免逐元素分支，提升流水效率 |
| 复数搬运 | 原生复数连续访问 | 交织↔实/虚平面互转（解/重交织 + host mask） | NPU 向量单元按 float 平面运算 |
| quick return | host 侧判断 | host 侧判断（n==0 或 α=0,β=1） | 语义一致，减少无谓 kernel launch |

### 3.3 支持硬件

与《算子任务书》要求一致：**Atlas A2 / A3 系列产品（arch22 / DAV_2201）**，CANN 9.1.0。

| 支持的芯片版本 | 涉及勾选 |
|----------------|----------|
| Atlas A2 训练系列产品 | √ |
| Atlas A2 推理系列产品 | √ |
| Atlas A3 训练系列产品 | √ |
| Atlas A3 推理系列产品 | √ |
| Atlas 800I A2（910B3，性能测试设备） | √ |

### 3.4 算子约束限制

| 约束 | 说明 |
|------|------|
| 数据类型 | 仅支持 complex64（`aclblasComplex`）；不支持 fp16/fp64/bf16 输入 |
| 矩阵布局 | 列主序，lda × n 物理布局，lda ≥ max(1, n) |
| 对称语义 | symmetric (A=A^T)，不取共轭；非 Hermitian |
| uplo | 仅支持 ACLBLAS_UPPER / ACLBLAS_LOWER，非法枚举返回 **INVALID_VALUE**（已核对配套用例 `TC_ED_190 invalid_uplo` 期望 `INVALID_VALUE`，且与同族 `aclblasSsymv`、cuBLAS 一致；任务书正文写 `INVALID_ENUM`，此处以用例为准——见下方差异说明） |
| n | `n ≥ 0`；`n < 0` 返回 INVALID_VALUE，`n = 0` 合法 quick return（与同族 `ssymv` 的 `n<=0→SUCCESS` 有意不同） |
| 步长 | 支持任意非零整数步长（正/负）；incx==0/incy==0 返回 INVALID_VALUE |
| 空指针 | 条件化检查：A/x 仅在 α≠(0,0) 时要求非空，y 仅在真正写回时要求非空 |
| 特殊值 | α=(0,0)且β=(1,0) quick return；α=(0,0)退化为 y=β·y；β=(0,0)时 y 可为无效输入 |
| 核数 | 单位步长最多使用 8 个 AIV 核；非单位步长单核 |

**与任务书正文的差异说明（非法 uplo 返回码）**：任务书正文 §参数表将非法 uplo 记为 `ACLBLAS_STATUS_INVALID_ENUM`；但任务配套用例 `csymv_test.csv` 第 `TC_ED_190 invalid_uplo` 的 `expect_result` 为 `ACLBLAS_STATUS_INVALID_VALUE`，且仓内同族 `aclblasSsymv`、cuBLAS `cublasCsymv` 亦返回 `INVALID_VALUE`。**本设计以配套用例为准返回 `INVALID_VALUE`**；若评审要求严格对齐任务书正文，可一行改为 `INVALID_ENUM`（不影响其余逻辑）。

## 四、特性交叉分析

| 特性 | 说明 |
|------|------|
| 多核并行 | 按输出行轮转分核，跨核写 y 冲突由 `SetAtomicAdd` 消解，结果在容差内浮动 |
| workspace | 各核直接原子累加写 y，无跨核归约 workspace；仅 mask（重交织）需 host 预生成小缓冲 |
| 精度 | 与 §5.1 精度标准对齐；golden 按 Netlib ssymv 语义自实现复数参考（对称不共轭） |
| 对称性 | 只读 uplo 指定三角，对称位置直接引用，不重复计算 |
| 并发 | kernel 为读 A/x + 写 y，无跨算子共享状态，可安全并发（依赖 handle 绑定 stream） |

## 五、可维可测分析

### 5.1 精度标准 / 性能标准

**精度标准（对齐任务书 §3.2）：**

- 输出向量 y 全量验证，实部/虚部分别按 FLOAT32 标准判定
- `|actual - golden| ≤ atol + rtol × |golden|`，其中 `rtol = 2^-10`，`atol = 2^-16`
- `required_matched_ratio ≥ 0.99`，`max_abs_error ≤ 1e-2` 或 `32 × ULP`
- golden 由测试工程内按 Netlib `ssymv` 语义自实现的复数参考（对称不共轭）
- Inf/NaN 特殊值按一致性判定

**性能标准（对齐任务书 §3.3，测试设备 Atlas 800I A2 / 910B3）：**

| case | uplo | n | 标杆耗时（Avg, us） |
|------|------|-----|---------------------|
| 1 | ACLBLAS_UPPER | 512 | 15.45 |
| 2 | ACLBLAS_UPPER | 2048 | 46.14 |
| 3 | ACLBLAS_LOWER | 2048 | 38.57 |
| 4 | ACLBLAS_UPPER | 4096 | 146 |
| 5 | ACLBLAS_LOWER | 4096 | 99.55 |

- 性能数据须先 warmup 再有效采样 >50 次取平均

### 5.2 兼容性分析

| 兼容性项 | 说明 |
|----------|------|
| 接口兼容 | `aclblasCsymv` 为仓内新增接口声明，与同仓 `aclblasSsymv` 同构（实数→复数），供其他产品线共用，禁止 A2/A3 私有平行 API |
| 数据格式 | complex64 内存布局与仓内其他复数算子（cgemv/cgerc 等）一致 |
| 多代际 | arch22 专用实现位于 `blas/symv/arch22/`，不影响 arch35 等路径 |
| 语义对齐 | quick return/退化/负步长/非法参数等边界语义与 cuBLAS 对齐 |
| README | `blas/symv/README.md` 新增 `aclblasCsymv` 参数说明章节与产品支持表（标注 Atlas A2/A3：支持） |

## 附录：测试工程与自测方案

### 测试工程目录

```text
include/cann_ops_blas.h                        新增 aclblasCsymv 公共声明
blas/symv/README.md                            新增参数说明与 A2/A3 产品支持表
blas/symv/arch22/csymv_host.cpp                参数校验、tiling、调度
blas/symv/arch22/csymv_kernel.cpp              Ascend C kernel 与直调封装
blas/symv/arch22/csymv_kernel.h                kernel 启动声明
blas/symv/arch22/csymv_tiling_data.h           tiling 结构定义
test/symv/csymv/CMakeLists.txt                 注册 GTest 工程（ops_blas_add_gtest_tests）
test/symv/csymv/csymv_param.h                  CSV 解析与输入生成参数
test/symv/csymv/csymv_golden.h                 独立 CPU 复数参考（Netlib ssymv 语义，对称不共轭）
test/symv/csymv/arch22/csymv_test.cpp          参数化测试主体
test/symv/csymv/arch22/csymv_npu_wrapper.h     算子调用封装
test/symv/csymv/csymv_test.csv                 1200 条 CSV 驱动用例
```

`blas/CMakeLists.txt` 自动收集 arch22 源码；不复制 competitions 通用 ACLNN 示例目录。

### 用例规模与分类（共 1200 条 = 1000 精度 + 200 性能）

| 类别 | 前缀 | 条数 | 说明 |
|------|------|------|------|
| L0 基础 | TC_L0 | 4 | uplo × 小尺寸 (4,8) |
| L1 尺寸扫描 | TC_SQ | 46 | 23 种尺寸（1→2048，含奇数/边界/非对齐）× uplo |
| L2 标量 | TC_AB | 16 | 8 组 alpha/beta（0/1/-1/复数/纯虚数）× uplo |
| L4 前导维 | TC_LD | 6 | lda = n+4 padding × uplo |
| L4 步长 | TC_INC | 72 | incx×incy ∈ ±1/±2/±3 全组合 × uplo |
| L5 填充/特殊值 | TC_FL | 18 | 均匀/全零/交替/极端/Inf/NaN |
| L5b 覆盖 | TC_CV | 16 | 中等尺寸（10~400）× uplo |
| L6 边界负向 | TC_ED | 18 | 零维/α=0 退化/β=0 不读 y/空指针/非法枚举/非法 lda/负维度/零步长 |
| EX 扩展 | TC_EX | 804 | 尺寸池 × uplo × 标量 × 步长 × padding 确定性采样 |
| PF 性能 | TC_PF | 200 | 5 条任务书典型 case + 尺寸扫描（均连续访存） |

### 验收口径

- **精度**：`verify_accuracy.py` 编译并跑 GTest，逐条比对 `expect_result`（排除 TC_PF）；实/虚部分别按 FLOAT32：`atol=2⁻¹⁶`、`rtol=2⁻¹⁰`、`matched_ratio ≥ 0.99`、`max_abs_error ≤ 1e-2` 或 `32 ULP`；并出仓内 MERE(<2⁻¹³)/MARE(×10) 双口径。golden 为测试工程内自实现复数参考。
- **性能**：`verify_performance.py` 采 TC_PF，warmup 后有效采样 >50 次取平均，与 `gpu_baseline.csv` 比对（倍率 0.8 对标 A100 基线）。
- **已核对的边界用例**：`TC_ED_190 invalid_uplo → INVALID_VALUE`、`TC_ED_192 neg_n → INVALID_VALUE`、`TC_ED_183 beta0_y_nan → SUCCESS`、`TC_ED_184 alpha0_scale → SUCCESS`、`TC_ED_182 alpha0_beta1_qr → SUCCESS` 及空指针系列（`TC_ED_185~189`）均须通过。
- **补充用例**：给定用例未覆盖 Device 指针标量、非单位步长与 uplo 组合的深挖、未引用三角填 NaN 毒值、lda padding 毒值、重复调用等，自验时补充。
