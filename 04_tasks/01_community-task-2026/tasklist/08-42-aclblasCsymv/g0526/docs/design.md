# aclblasCsymv 算子设计文档

## 一、需求背景

### 1.1 需求来源

通过**社区任务**完成开源仓算子贡献的需求。在昇腾 NPU（Ascend 950PR / arch35）上使用 Ascend C 开发 `aclblasCsymv`（complex64 对称矩阵-向量乘）算子，对齐 cuBLAS `cublasCsymv` / Netlib `ssymv`（复数路径按实数语义扩展）语义，纳入 `ops-blas` 开源仓。

- 任务书算子：`aclblasCsymv`
- 目标硬件：Ascend 950PR（arch35 / DAV_3510）
- 参考标杆：cuBLAS `cublasCsymv`（语义参考 Netlib `ssymv`）
- 代码仓：`ops-blas`（Ascend BLAS 算子库），源码路径 `blas/symv/arch35/`

### 1.2 背景介绍

#### 1.2.1 aclblasCsymv 算子实现优化

本任务在 `ops-blas` 开源仓中新增 `aclblasCsymv` 算子实现，属 Level-2 级 BLAS 矩阵-向量乘算子（symmetric matrix-vector multiply），对齐 cuBLAS 语义进行原生实现。

**参考资源路径：**
- 接口定义（算子信息库 / 头文件）：`ops-blas/include/cann_ops_blas.h`（`aclblasCsymv` 声明**当前不存在**，需新增；参考同仓已有 `aclblasSsymv` 声明，第 154-158 行）
- 算子实现目录：`ops-blas/blas/symv/arch35/`（与现有 `aclblasSsymv` 实现同目录）
- 测试目录：`ops-blas/test/symv/csymv/`
- 语义参考：cuBLAS `cublasCsymv`（`cublas_docs/cublas_symv.md`）、Netlib BLAS `ssymv`（`https://www.netlib.org/blas/ssymv.f`）
- 同族已有实现（复用参考）：`ops-blas/blas/symv/arch35/ssymv_kernel.cpp`（实数版对称矩阵-向量乘，AIV 分块范式）、`ops-blas/blas/asum/arch35/scasum_kernel.cpp`（complex64 SIMT 归约范式）

#### 1.2.2 aclblasCsymv 算子现状分析

`ops-blas` 仓内已有实数版 `aclblasSsymv`（对称矩阵-向量乘），但缺少复数版 `aclblasCsymv`。本任务补齐该接口。复数版与实数版的核心差异在于：复数乘法（4 次实数乘 + 2 次实数加）与对称矩阵的复数元素存储（每元素实部/虚部各 float）。

##### 1.2.2.1 标杆算子支持的数据类型和数据格式

与算子信息库/接口声明保持一致：

| 项 | 取值 |
|----|------|
| 输入类型 | `aclblasComplex`（complex64，float real + float imag，8 字节对齐） |
| 输出类型 | `aclblasComplex`（complex64 向量） |
| 矩阵布局 | A 为 n×n 列主序（Column-Major）复数对称矩阵（A = A^T，**不取共轭**），lda×n 物理布局，仅 `uplo` 指定三角被引用，另一三角由对称性推断 |
| 向量布局 | x、y 逻辑长度 n，支持任意步长 incx/incy（含负步长） |
| 索引语义 | `y[i] = alpha * Σ_{j} A[i][j] * x[j] + beta * y[i]`，列主序 A 元素 `A(i,j) = A[i + j*lda]`（0-based） |

##### 1.2.2.2 标杆算子实现描述

cuBLAS `cublasCsymv` / Netlib `ssymv`（复数扩展）计算定义：

```
y = alpha * A * x + beta * y
```

展开（对称不共轭，A = A^T）：

```
对每个 i (0..n-1):
  if uplo == UPPER:
    y[i] = alpha * ( Σ_{j>=i} A[i][j]*x[j] + Σ_{j<i} A[j][i]*x[j] ) + beta * y[i]
  else (LOWER):
    y[i] = alpha * ( Σ_{j<=i} A[i][j]*x[j] + Σ_{j>i} A[j][i]*x[j] ) + beta * y[i]
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
- 参数校验失败返回 `INVALID_VALUE` / `INVALID_ENUM` / `HANDLE_IS_NULLPTR`

##### 1.2.2.3 标杆算子实现流程图

```
┌─────────────────────────────────────────┐
│           aclblasCsymv (入口)            │
└──────────────────┬──────────────────────┘
                   ▼
        ┌─────────────────────┐
        │  参数校验            │
        │  handle/uplo/lda/   │───→ INVALID_VALUE / INVALID_ENUM /
        │  incx/incy/alpha/   │     NOT_INITIALIZED
        │  beta/null 检查     │
        └─────────┬───────────┘
                  ▼
        ┌─────────────────────┐
        │  quick return ?      │───yes──→ 返回 SUCCESS（不计算/不写 y）
        │  n==0 或 (α=0,β=1)   │
        └─────────┬───────────┘
                  ▼ no
        ┌─────────────────────┐
        │  α=0 且 β≠1 ?        │───yes──→ y = beta*y（不读 A、x）
        └─────────┬───────────┘
                  ▼ no
        ┌─────────────────────┐
        │  计算 tiling         │
        │  分核 / 分块 / uplo  │
        └─────────┬───────────┘
                  ▼
        ┌─────────────────────┐
        │  启动 kernel         │
        │  多核并行 A*x        │
        └─────────┬───────────┘
                  ▼
        ┌─────────────────────┐
        │  每行融合 beta*y     │
        │  → y                │
        └─────────┬───────────┘
                  ▼
           返回 SUCCESS
```

## 二、需求分析

### 2.1 外部组件依赖

| 依赖组件 | 版本/说明 | 用途 |
|----------|-----------|------|
| ACL (AscendCL) | CANN 9.1.0 | 设备内存管理、stream、kernel launch |
| AscendC kernel 运行时 | CANN 9.1.0 | `__gm__`/`__ubuf__` 地址空间、`DataCopy`、`Adds`、`Mul` 等向量指令、`CrossCoreSetFlag/WaitFlag` |
| SIMT API (`simt_api/asc_simt.h`) | CANN 9.1.0 | SIMT 编程模型（备选路径） |
| GTest | 系统已装 | 参数化测试框架 |
| CBLAS/LAPACK | 系统已装 | 实数 ssymv 对照（复数 golden 自实现） |

### 2.2 内部适配模块

| 内部模块 | 说明 |
|----------|------|
| `common/helper/aclblas_handle_internal.h` | handle 结构、workspace 管理（`GetEffectiveWorkspace`） |
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
| uplo 分支 | UPPER/LOWER | UPPER/LOWER | 一致 |
| 正负步长 | 支持 | 支持 | 负步长按 Netlib 语义 |
| alpha/beta 特殊值 | 支持 | 支持 | no-op / 退化分支 |
| 前导维 lda | 任意 ≥ max(1,n) | 支持 | 含 padding |

## 三、需求详细设计

### 3.1 调用方式

本任务采用 **Kernel 直调（ops-blas 内部 API）** 方式：
- 上层通过 `aclblasCsymv(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy)` 调用
- host 侧（`csymv_host.cpp`）完成参数校验、quick return 判定、tiling 计算、workspace 准备，调用 `csymv_kernel_do` 启动 kernel
- kernel 侧（`csymv_kernel.cpp`）执行对称矩阵-向量乘
- 输出 y 为设备指针，由调用方负责 D2H 拷贝

### 3.2 需求总体设计

整体采用**多核 AIV 行分块**架构（参照同仓 `aclblasSsymv` 范式，扩展复数支持）：

```
host: 参数校验 → quick return 判定 → tiling 计算 → workspace 准备 → launch kernel
kernel: [各核按行分块计算部分 A*x] → [融合 beta*y → 写入 y]
```

#### 3.2.1 host 侧设计

##### 3.2.1.1 分核策略

- 按输出向量 y 的行维度分核：每核处理一段连续的行区间
- 使用核数 `useCoreNum = min(核数, n)`，每核行数 `rowsPerCore = ceil(n / useCoreNum)`
- 每核行区间 `[rowStart[i], rowStart[i]+rowCount[i])`，其中 `rowStart[i] = Σ_{k<i} rowCount[k]`
- 核内按行遍历：每行计算 `y[i] = alpha * Σ_j A_sym[i][j] * x[j] + beta * y[i]`

##### 3.2.1.2 数据分块和内存优化策略

**UB 规划（每核）：**

以 n=2048、每核行块为例，A 行向量（对称访问）与 x 向量分块载入 UB：

| Buffer | 内容 | 大小 |
|--------|------|------|
| A_tile | 当前行块 A[i][j]（复数，实部/虚部分离视图） | rowsPerCore × tileN × 2 × 4B |
| x_tile | 当前列块 x[j]（复数） | tileN × 2 × 4B |
| y_partial | 行部分和（复数） | rowsPerCore × 2 × 4B |

**分块策略：**
- 列维度按 `tileN` 分块（如 256/512），逐列块载入 x_tile 与 A 列块，累加部分和
- 行块内一次性载入多行，复用 x 数据降低 GM 访问
- 950PR UB 容量 248KB，`rowsPerCore × tileN × 8B ≤ 约 96KB`（留余量），保证双 buffer 流水

**workspace 规划：**
- 各核独立计算行分块，输出直接写 y（不同核写不同行），无需跨核归约 workspace
- 仅需核间同步保证各自完成（`CrossCoreSetFlag/WaitFlag`）

**数据访问：**
- A 为列主序：`A[i][j]` 物理地址 `A + i*sizeof(complex) + j*lda*sizeof(complex)`，同列元素连续、同列行遍历时按 lda 跨步
- 复数元素按实部/虚部视图处理（对齐 scasum/cdotc 的 float2 或 AIV 分离视图经验）

##### 3.2.1.3 tilingKey 规划策略

本算子为单形态（对称矩阵-向量乘）算子，**不引入 tilingKey 分支**。tiling 数据（`CsymvTilingData`）整体由 host 计算后随 kernel 参数传入：

```cpp
struct CsymvTilingData {
    int32_t n;                 // 矩阵维度
    int32_t lda;               // 前导维度
    aclblasFillMode_t uplo;    // UPPER/LOWER
    int32_t useCoreNum;        // 使用核数
    uint32_t rowStart[MAX_CORE_NUM];  // 每核起始行
    uint32_t rowCount[MAX_CORE_NUM];  // 每核行数
    uint32_t tileN;            // 列分块大小
    int32_t incx, incy;        // 步长
    int32_t alphaIsZero;       // alpha 是否为 (0,0)
    int32_t betaIsZero;        // beta 是否为 (0,0)
};
```

#### 3.2.2 kernel 侧设计

##### 3.2.2.1 kernel 侧实现描述

kernel 按输出行分块，各核独立计算：

1. **行块遍历**（每核）：处理本核 `rowCount` 行
2. **列块循环**：对每行，按 `tileN` 分块遍历 A 的行向量（对称访问）
   - UPPER：主对角及上方 `A[i][j]`（j≥i）+ 下方对称 `A[j][i]`（j<i）
   - LOWER：主对角及下方 `A[i][j]`（j≤i）+ 上方对称 `A[j][i]`（j>i）
3. **复数乘累加**：`acc += A_sym[i][j] * x[j]`（AIV 复数乘法按实部/虚部分离计算）
4. **beta*y 融合**：每行计算完成 `y[i] = alpha * acc + beta * y[i]`，写回 y
5. **核间同步**：`CrossCoreSetFlag/WaitFlag` 保证各核写回完成

##### 3.2.2.2 Ascend C 实现流程图

```
┌────────────────────────────────────────────────────────┐
│                csymv_kernel                             │
│  KERNEL_TASK_TYPE_DEFAULT(AIV_ONLY)                     │
└─────────────────────────┬──────────────────────────────┘
                          ▼
        ┌────────────────────────────────────┐
        │  blockIdx = GetBlockIdx()          │
        │  rowStart/rowCount = tiling[...]   │
        └─────────────────┬──────────────────┘
                          ▼
        ┌────────────────────────────────────┐
        │  遍历本核行区间 [rowStart, rowEnd)  │
        └─────────────────┬──────────────────┘
                          ▼
        ┌────────────────────────────────────┐
        │  初始化 acc[2] = {0,0} (复数)       │
        │  UPLO? → 行向量对称访问模式        │
        └─────────────────┬──────────────────┘
                          ▼
        ┌────────────────────────────────────┐
        │  列块循环 (tileN 分块):             │
        │  │ 载入 x_tile（复数）             │
        │  │ 载入 A 行块（对称, UPPER/LOWER）│
        │  │ acc += A_sym[i][j] * x[j]      │
        │  │ (复数乘: 实部/虚部分离)          │
        └─────────────────┬──────────────────┘
                          ▼
        ┌────────────────────────────────────┐
        │  y[i] = alpha*acc + beta*y[i]      │
        │  (含 alpha=0/beta=0 快速路径)       │
        └─────────────────┬──────────────────┘
                          ▼
        ┌────────────────────────────────────┐
        │  CrossCoreSetFlag<0,MTE3>(0)       │
        │  CrossCoreWaitFlag<0,MTE3>(0)      │
        └────────────────────────────────────┘
```

`行内对称向量访问`（UPLO=UPPER，对行 i）：

```
j = 0 .. i-1:    A_sym[i][j] = A[j][i]        # 下方，读上三角对称位置
j = i .. n-1:    A_sym[i][j] = A[i][j]        # 上方，直接读
```

##### 3.2.2.3 Ascend C 实现流程图与标杆算子流程图存在的差异点和原因

| 差异点 | 标杆（Netlib/cuBLAS） | Ascend C 本设计 | 原因 |
|--------|------------------------|-----------------|------|
| 循环结构 | 双层标量循环（行内顺序累加） | 行块并行 + 列块分块 + 向量化 | 多核并行 + 向量指令提升吞吐 |
| 行并行 | 单核串行 | 最多 16 核按行分块 | 950PR 多核并行降低延迟 |
| 复数乘 | 通用复数乘 | 实部/虚部分离 + 向量化 | 利用 AIV 向量指令，避免标量复数指令 |
| beta*y 融合 | 单独循环 | 每行计算完成后就地融合 | 减少 y 二次访问 |
| 对称访问 | 逐元素判断 uplo | 分块内按 UPPER/LOWER 统一访问模式 | 避免逐元素分支，提升流水效率 |
| quick return | host 侧判断 | host 侧判断（n==0 或 α=0,β=1） | 语义一致，减少无谓 kernel launch |

### 3.3 支持硬件

与《算子任务书》要求一致：**Ascend 950PR（arch35 / DAV_3510）**，CANN 9.1.0。

### 3.4 算子约束限制

| 约束 | 说明 |
|------|------|
| 数据类型 | 仅支持 complex64（`aclblasComplex`）；不支持 fp16/fp64/bf16 输入 |
| 矩阵布局 | 列主序，lda × n 物理布局，lda ≥ max(1, n) |
| 对称语义 | symmetric (A=A^T)，不取共轭；非 Hermitian |
| uplo | 仅支持 ACLBLAS_UPPER / ACLBLAS_LOWER，非法枚举返回 INVALID_ENUM |
| 步长 | 支持任意非零整数步长（正/负）；incx==0/incy==0 返回 INVALID_VALUE |
| 特殊值 | alpha=(0,0)且beta=(1,0) quick return；alpha=(0,0)退化为 y=beta*y；beta=(0,0)时 y 可为无效输入 |
| 核数 | 最多使用 16 个 AIV 核 |

## 四、特性交叉分析

| 特性 | 说明 |
|------|------|
| 多核并行 | 按输出行分块，各核独立写不同 y 行，无写冲突 |
| workspace | 各核直接写 y，无跨核归约 workspace；核间仅需完成同步 |
| 精度 | 与 §3.2 精度标准对齐；golden 按 Netlib ssymv 语义自实现复数参考（对称不共轭），double 精度累加 |
| 对称性 | 只读 uplo 指定三角，对称位置直接引用，不重复计算 |
| 并发 | kernel 为读 A/x + 写 y（各核不同行），无跨算子共享状态，可安全并发 |

## 五、可维可测分析

### 5.1 精度标准 / 性能标准

**精度标准（对齐任务书 §3.2）：**
- 输出向量 y 全量验证，实部/虚部分别按 FLOAT32 标准判定
- `|actual - golden| ≤ atol + rtol × |golden|`，其中 `rtol = 2^-10`，`atol = 2^-16`
- `required_matched_ratio ≥ 0.99`，`max_abs_error ≤ 1e-2` 或 `32 × ULP`
- golden 由测试工程内按 Netlib `ssymv` 语义自实现的复数参考（double 精度累加）
- Inf/NaN 特殊值按一致性判定

**性能标准（对齐任务书 §3.3）：**

| case | uplo | n | 标杆耗时（Avg, us） |
|------|------|-----|---------------------|
| 1 | ACLBLAS_UPPER | 512 | 3.64 |
| 2 | ACLBLAS_UPPER | 2048 | 8.90 |
| 3 | ACLBLAS_LOWER | 2048 | 6.44 |

- 性能数据须先 warmup 再有效采样 >50 次取平均

### 5.2 兼容性分析

参考 https://gitcode.com/cann/cann-ops-competitions/pull/1182 的评审口径：

| 兼容性项 | 说明 |
|----------|------|
| 接口兼容 | `aclblasCsymv` 为仓内新增接口声明，与同仓 `aclblasSsymv` 同构（实数→复数），供其他产品线共用，禁止 950PR 私有平行 API |
| 数据格式 | complex64 内存布局与仓内其他复数算子（cdotc/cdotu/scasum/caxpy）一致 |
| 多代际 | arch35 专用实现位于 `blas/symv/arch35/`，不影响其他架构（arch22 等）路径 |
| 语义对齐 | quick return/退化/负步长/非法参数等边界语义与 cuBLAS 对齐 |
| README | `blas/symv/README.md` 新增 `aclblasCsymv` 参数说明章节与产品支持表（Ascend 950PR：支持） |

## 附录：自测方案概要

- 测试框架：ops-blas test 目录 CSV 驱动 GTest 参数化测试
- 用例覆盖：小尺寸基础、尺寸扫描（0/1/质数/2 幂±1/大规模）、alpha/beta 特殊值、uplo 双分支、前导维 padding、步长组合（±1/±2/±3）、边界负向（n<0/lda 非法/零步长/空指针）、Inf/NaN 特殊值、性能用例（512/2048/4096/8192）
- golden：测试工程内按 Netlib `ssymv` 语义自实现复数参考（对称不共轭，double 累加）
- 性能：warmup + 有效采样 >50 次取平均，与 §3.3 标杆对比