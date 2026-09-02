# aclblasCtrmv 算子设计文档

## 一、需求背景

### 1.1 需求来源

通过**社区任务**完成开源仓算子贡献的需求。在昇腾 NPU（Ascend 950PR / arch35）上使用 Ascend C 开发 `aclblasCtrmv`（complex64 三角矩阵-向量乘）算子，对齐 cuBLAS `cublasCtrmv` / Netlib `ctrmv` 语义，纳入 `ops-blas` 开源仓。

- 任务书算子：`aclblasCtrmv`
- 目标硬件：Ascend 950PR（arch35 / DAV_3510）
- 参考标杆：cuBLAS `cublasCtrmv`（语义参考 Netlib `ctrmv`）
- 代码仓：`ops-blas`（Ascend BLAS 算子库），源码路径 `blas/trmv/arch35/`

### 1.2 背景介绍

#### 1.2.1 aclblasCtrmv 算子实现优化

本任务在 `ops-blas` 开源仓中新增 `aclblasCtrmv` 算子实现，属 Level-2 级 BLAS 三角矩阵-向量乘算子（triangular matrix-vector multiply），对齐 cuBLAS 语义进行原生实现。

**参考资源路径：**
- 接口定义（算子信息库 / 头文件）：`ops-blas/include/cann_ops_blas.h`（`aclblasCtrmv` 声明已存在，第 350-352 行）
- 算子实现目录：`ops-blas/blas/trmv/arch35/`（与现有 `aclblasStrmv` 实现同目录）
- 测试目录：`ops-blas/test/trmv/ctrmv/`（当前仅有硬编码冒烟测试，CSV 驱动测试参照同族 `test/trmv/strmv/` 新建）
- 语义参考：cuBLAS `cublasCtrmv`、Netlib BLAS `ctrmv`（`https://www.netlib.org/blas/ctrmv.f`）
- 同族已有实现（复用参考）：`ops-blas/blas/trmv/arch35/strmv_kernel.cpp`（实数版三角矩阵-向量乘，含 uplo/trans/diag 分支处理范式）、`ops-blas/blas/asum/arch35/scasum_kernel.cpp`（complex64 SIMT 归约范式）

#### 1.2.2 aclblasCtrmv 算子现状分析

`ops-blas` 仓内已有实数版 `aclblasStrmv`（三角矩阵-向量乘），但缺少复数版 `aclblasCtrmv`。本任务补齐该接口。复数版与实数版的核心差异在于：复数乘法（4 次实数乘 + 2 次实数加）、共轭转置（trans=OP_C 时对虚部取反）、三角矩阵的复数元素存储。

##### 1.2.2.1 标杆算子支持的数据类型和数据格式

与算子信息库/接口声明保持一致：

| 项 | 取值 |
|----|------|
| 输入类型 | `aclblasComplex`（complex64，float real + float imag，8 字节对齐） |
| 输出类型 | `aclblasComplex`（complex64 向量，原地覆写） |
| 矩阵布局 | A 为 n×n 列主序（Column-Major）复数三角矩阵，lda×n 物理布局，仅 `uplo` 指定三角被引用 |
| 向量布局 | x 逻辑长度 n，原地输入输出，支持任意步长 incx（含负步长） |
| 索引语义 | `x[i] = op(A)[i][j] * x[j]` 累加，列主序 A 元素 `A(i,j) = A[i + j*lda]`（0-based） |

##### 1.2.2.2 标杆算子实现描述

cuBLAS `cublasCtrmv` / Netlib `ctrmv` 计算定义：

```
x = op(A) * x
```

其中 op(A) 由 trans 决定：
- `trans = ACLBLAS_OP_N`：`op(A) = A`
- `trans = ACLBLAS_OP_T`：`op(A) = A^T`（转置，不取共轭）
- `trans = ACLBLAS_OP_C`：`op(A) = A^H`（共轭转置，取共轭）

展开（以 OP_N + UPPER 为例，其余分支同理）：

```
对每个 i (0..n-1):
  if uplo == UPPER:
    x[i] = Σ_{j>=i} A[i][j]*x[j]   (diag=UNIT 时 j==i 项 A[i][i] 视为 1)
  else (LOWER):
    x[i] = Σ_{j<=i} A[i][j]*x[j]   (diag=UNIT 时 j==i 项 A[i][i] 视为 1)
```

- 上三角（UPPER）：仅引用主对角及上方元素 `A[i][j]`（j≥i），下方不访问
- 下三角（LOWER）：仅引用主对角及下方元素 `A[i][j]`（j≤i），上方不访问
- `diag = ACLBLAS_UNIT`：主对角元素视为 1，**不访问** A 的对角存储
- `diag = ACLBLAS_NON_UNIT`：使用存储的对角元素
- 复数乘法：`(a+bi)(c+di) = (ac-bd) + (ad+bc)i`
- 共轭转置（OP_C）：访问 `A[j][i]` 时取共轭 `conj(A[j][i])`
- x 原地覆写：输入旧值在计算中被读取，输出新值写回同一向量

**边界语义（对齐 cuBLAS / Netlib）：**
- `n = 0`：合法 no-op，返回 SUCCESS 不执行计算
- 负步长（`incx<0`）：合法，从反向起点遍历
- 参数校验失败返回 `INVALID_VALUE` / `INVALID_ENUM` / `HANDLE_IS_NULLPTR`

##### 1.2.2.3 标杆算子实现流程图

```
┌─────────────────────────────────────────┐
│           aclblasCtrmv (入口)            │
└──────────────────┬──────────────────────┘
                   ▼
        ┌─────────────────────┐
        │  参数校验            │
        │  handle/uplo/trans/ │───→ INVALID_VALUE / INVALID_ENUM /
        │  diag/lda/incx/null │     NOT_INITIALIZED
        └─────────┬───────────┘
                  ▼
        ┌─────────────────────┐
        │  n == 0 ?            │───yes──→ 返回 SUCCESS（不计算）
        └─────────┬───────────┘
                  ▼ no
        ┌─────────────────────┐
        │  计算 tiling         │
        │  分核 / 分块 /       │
        │  uplo×trans×diag 分支│
        └─────────┬───────────┘
                  ▼
        ┌─────────────────────┐
        │  启动 kernel         │
        │  多核并行 op(A)*x    │
        └─────────┬───────────┘
                  ▼
        ┌─────────────────────┐
        │  x 原地覆写          │
        │  → x                │
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
| CBLAS/LAPACK | 系统已装 | cblas `ctrmv` golden 生成 |

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
aclblasStatus_t aclblasCtrmv(
    aclblasHandle_t handle,        // aclblas 句柄
    aclblasFillMode_t uplo,        // ACLBLAS_UPPER / ACLBLAS_LOWER
    aclblasOperation_t trans,      // ACLBLAS_OP_N / OP_T / OP_C
    aclblasDiagType_t diag,        // ACLBLAS_NON_UNIT / ACLBLAS_UNIT
    int n,                         // 三角矩阵 A 的行/列数
    const aclblasComplex* A,       // 复数三角矩阵（列主序，lda×n）
    int lda,                       // A 前导维度
    aclblasComplex* x,             // 输入/输出向量 x（设备指针，原地覆写）
    int incx                       // x 步长
);
```

#### 2.3.2 Ascend C 算子相关约束

与标杆算子（cuBLAS `cublasCtrmv`）相比，当前设计的功能范围：

| 功能点 | cuBLAS | 本设计 | 说明 |
|--------|--------|--------|------|
| 数据类型 | complex64 | complex64 | 一致 |
| uplo 分支 | UPPER/LOWER | UPPER/LOWER | 一致 |
| trans 分支 | N/T/C | N/T/C | 一致（含共轭转置） |
| diag 分支 | UNIT/NON_UNIT | UNIT/NON_UNIT | 一致（UNIT 不访问对角） |
| 正负步长 | 支持 | 支持 | 负步长按 Netlib 语义 |
| 原地覆写 | 支持 | 支持 | x 原地 |

## 三、需求详细设计

### 3.1 调用方式

本任务采用 **Kernel 直调（ops-blas 内部 API）** 方式：
- 上层通过 `aclblasCtrmv(handle, uplo, trans, diag, n, A, lda, x, incx)` 调用
- host 侧（`ctrmv_host.cpp`）完成参数校验、no-op 判定、tiling 计算、workspace 准备，调用 `ctrmv_kernel_do` 启动 kernel
- kernel 侧（`ctrmv_kernel.cpp`）执行三角矩阵-向量乘
- 输出 x 为设备指针（原地），由调用方负责 D2H 拷贝

### 3.2 需求总体设计

整体采用**多核 AIV 行分块**架构（参照同仓 `aclblasStrmv` 范式，扩展复数与 trans/diag 支持）：

```
host: 参数校验 → no-op 判定 → tiling 计算 → workspace 准备 → launch kernel
kernel: [各核按行分块计算部分 op(A)*x] → [x 原地覆写]
```

#### 3.2.1 host 侧设计

##### 3.2.1.1 分核策略

- 按输出向量 x 的行维度分核：每核处理一段连续的行区间
- 使用核数 `useCoreNum = min(核数, n)`，每核行数 `rowsPerCore = ceil(n / useCoreNum)`
- 每核行区间 `[rowStart[i], rowStart[i]+rowCount[i])`，其中 `rowStart[i] = Σ_{k<i} rowCount[k]`
- 核内按行遍历：每行计算 `x[i] = Σ_j op(A)[i][j] * x[j]`
- trans 分支影响访问方向：
  - OP_N：按行访问 A[i][j]，行内 j 从对角向边界扩展
  - OP_T/OP_C：按列访问 A[j][i]，等效转置，访问方向调整

##### 3.2.1.2 数据分块和内存优化策略

**UB 规划（每核）：**

以 n=2048、每核行块为例，A 行向量与 x 向量分块载入 UB：

| Buffer | 内容 | 大小 |
|--------|------|------|
| A_tile | 当前行块 A[i][j]（复数，实部/虚部分离视图） | rowsPerCore × tileN × 2 × 4B |
| x_tile | 当前列块 x[j]（复数） | tileN × 2 × 4B |
| acc | 行部分和（复数） | rowsPerCore × 2 × 4B |

**分块策略：**
- 列维度按 `tileN` 分块（如 256/512），逐列块载入 x_tile 与 A 列块，累加部分和
- 行块内一次性载入多行，复用 x 数据降低 GM 访问
- 950PR UB 容量 248KB，`rowsPerCore × tileN × 8B ≤ 约 96KB`（留余量），保证双 buffer 流水
- diag=UNIT 时跳过对角列的载入与乘加，减少无效访问

**workspace 规划：**
- 各核独立计算行分块，输出直接写 x 的不同行，无需跨核归约 workspace
- 仅需核间同步保证各自完成（`CrossCoreSetFlag/WaitFlag`）

**数据访问：**
- A 为列主序：`A[i][j]` 物理地址 `A + i*sizeof(complex) + j*lda*sizeof(complex)`，同列元素连续
- 复数元素按实部/虚部视图处理（对齐 scasum/cdotc 的 float2 或 AIV 分离视图经验）

##### 3.2.1.3 tilingKey 规划策略

本算子为多形态算子（uplo × trans × diag 组合），但计算路径统一为三角矩阵-向量乘，**不引入 tilingKey 分支**，各形态由 kernel 内分支（编译期常量或运行时参数）统一处理。tiling 数据（`CtrmvTilingData`）整体由 host 计算后随 kernel 参数传入：

```cpp
struct CtrmvTilingData {
    int32_t n;                 // 矩阵维度
    int32_t lda;               // 前导维度
    aclblasFillMode_t uplo;    // UPPER/LOWER
    aclblasOperation_t trans;  // OP_N/OP_T/OP_C
    aclblasDiagType_t diag;    // NON_UNIT/UNIT
    int32_t useCoreNum;        // 使用核数
    uint32_t rowStart[MAX_CORE_NUM];  // 每核起始行
    uint32_t rowCount[MAX_CORE_NUM];  // 每核行数
    uint32_t tileN;            // 列分块大小
    int32_t incx;              // 步长
};
```

#### 3.2.2 kernel 侧设计

##### 3.2.2.1 kernel 侧实现描述

kernel 按输出行分块，各核独立计算：

1. **行块遍历**（每核）：处理本核 `rowCount` 行
2. **列块循环**：对每行，按 `tileN` 分块遍历 A 的行向量（三角访问，受 uplo/trans/diag 约束）
   - uplo=UPPER：行 i 访问 `j ∈ [i, n)`（OP_N 时 A[i][j]）
   - uplo=LOWER：行 i 访问 `j ∈ [0, i]`（OP_N 时 A[i][j]）
   - trans=OP_T/OP_C：访问方向转置（读 A[j][i]），OP_C 时取共轭
   - diag=UNIT：跳过 j==i 的对角项（视为 1）
3. **复数乘累加**：`acc += A_tri[i][j] * x[j]`（实部/虚部分离计算）
4. **x 原地覆写**：每行计算完成 `x[i] = acc`，写回 x（注意原地依赖：行 i 的结果依赖 x 的低索引行，需保证计算顺序）
5. **核间同步**：`CrossCoreSetFlag/WaitFlag` 保证各核写回完成

##### 3.2.2.2 Ascend C 实现流程图

```
┌────────────────────────────────────────────────────────┐
│                ctrmv_kernel                             │
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
        │  uplo/trans/diag → 三角访问模式    │
        └─────────────────┬──────────────────┘
                          ▼
        ┌────────────────────────────────────┐
        │  列块循环 (tileN 分块):             │
        │  │ 载入 x_tile（复数）             │
        │  │ 载入 A 行块（三角约束）          │
        │  │ diag=UNIT → 跳过对角列          │
        │  │ trans=C → 取共轭                │
        │  │ acc += A_tri[i][j] * x[j]      │
        │  │ (复数乘: 实部/虚部分离)          │
        └─────────────────┬──────────────────┘
                          ▼
        ┌────────────────────────────────────┐
        │  x[i] = acc (原地覆写)             │
        └─────────────────┬──────────────────┘
                          ▼
        ┌────────────────────────────────────┐
        │  CrossCoreSetFlag<0,MTE3>(0)       │
        │  CrossCoreWaitFlag<0,MTE3>(0)      │
        └────────────────────────────────────┘
```

`行内三角访问`（OP_N + UPPER + NON_UNIT，对行 i）：

```
j = i .. n-1:   acc += A[i][j] * x[j]    # 上三角直接读
（diag=UNIT 时 j 从 i+1 开始，跳过 A[i][i]）
```

`行内三角访问`（OP_C + LOWER，对行 i）：

```
j = 0 .. i:     acc += conj(A[j][i]) * x[j]   # 下三角共轭转置
```

##### 3.2.2.3 Ascend C 实现流程图与标杆算子流程图存在的差异点和原因

| 差异点 | 标杆（Netlib/cuBLAS） | Ascend C 本设计 | 原因 |
|--------|------------------------|-----------------|------|
| 循环结构 | 双层标量循环（行内顺序累加） | 行块并行 + 列块分块 + 向量化 | 多核并行 + 向量指令提升吞吐 |
| 行并行 | 单核串行 | 最多 16 核按行分块 | 950PR 多核并行降低延迟 |
| 复数乘 | 通用复数乘 | 实部/虚部分离 + 向量化 | 利用 AIV 向量指令，避免标量复数指令 |
| trans/diag | 逐元素判断 | 分块内统一访问模式 + 跳过对角列 | 避免逐元素分支，提升流水效率 |
| 原地覆写 | 单核顺序保证依赖 | 多核需注意行依赖（低索引行先算） | 保证原地语义正确性 |
| no-op | host 侧判断 | host 侧判断（n==0） | 语义一致，减少无谓 kernel launch |

### 3.3 支持硬件

与《算子任务书》要求一致：**Ascend 950PR（arch35 / DAV_3510）**，CANN 9.1.0。

### 3.4 算子约束限制

| 约束 | 说明 |
|------|------|
| 数据类型 | 仅支持 complex64（`aclblasComplex`）；不支持 fp16/fp64/bf16 输入 |
| 矩阵布局 | 列主序，lda × n 物理布局，lda ≥ max(1, n) |
| 三角引用 | 仅引用 uplo 指定三角，另一部分不读取；diag=UNIT 时不访问对角 |
| uplo/trans/diag | 非法枚举返回 INVALID_ENUM |
| 步长 | 支持任意非零整数步长（正/负）；incx==0 返回 INVALID_VALUE |
| 原地语义 | x 原地覆写，输出依赖输入旧值 |
| 核数 | 最多使用 16 个 AIV 核 |

## 四、特性交叉分析

| 特性 | 说明 |
|------|------|
| 多核并行 | 按输出行分块，各核独立写不同 x 行；注意行依赖（OP_N 时行 i 依赖低索引行） |
| workspace | 各核直接写 x，无跨核归约 workspace；核间仅需完成同步 |
| 精度 | 与 §3.2 精度标准对齐；golden 由 cblas `ctrmv` 生成 |
| 三角性 | 只读 uplo 指定三角；diag=UNIT 时对角按 1 处理不访问 |
| trans 分支 | OP_T 转置（不共轭）、OP_C 共轭转置，访问模式由 kernel 内分支统一处理 |
| 并发 | kernel 为读 A/x + 写 x（各核不同行），无跨算子共享状态，可安全并发 |

## 五、可维可测分析

### 5.1 精度标准 / 性能标准

**精度标准（对齐任务书 §3.2）：**
- 输出向量 x 全向量验证，实部/虚部分别按 FLOAT32 标准判定
- `|actual - golden| ≤ atol + rtol × |golden|`，其中 `rtol = 2^-10`，`atol = 2^-16`
- `required_matched_ratio ≥ 0.99`，`max_abs_error ≤ 1e-2` 或 `32 × ULP`
- golden 由 cblas（Netlib `ctrmv`）生成，输出向量 x 全量验证
- diag=UNIT 用例 golden 计算时同样不访问 A 对角元素（对角按 1 处理）
- Inf/NaN 特殊值按一致性判定

**性能标准（对齐任务书 §3.3）：**

| case | n | uplo | trans | diag | incx | 标杆耗时（Avg, us） |
|------|-----|------|-------|------|------|---------------------|
| 1 | 512 | UPPER | N | NON_UNIT | 1 | 11.86 |
| 2 | 1024 | LOWER | T | NON_UNIT | 1 | 35.90 |
| 3 | 2048 | UPPER | C | UNIT | 1 | 75.59 |

- 性能数据须先 warmup 再有效采样 >50 次取平均

### 5.2 兼容性分析

| 兼容性项 | 说明 |
|----------|------|
| 接口兼容 | `aclblasCtrmv` 接口声明已存在于仓内 `include/cann_ops_blas.h`（第 350-352 行），供其他产品线共用，禁止 950PR 私有平行 API |
| 数据格式 | complex64 内存布局与仓内其他复数算子（cdotc/cdotu/scasum/caxpy）一致 |
| 多代际 | arch35 专用实现位于 `blas/trmv/arch35/`，不影响其他架构（arch22 等）路径 |
| 语义对齐 | no-op/负步长/非法参数等边界语义与 cuBLAS 对齐 |
| README | 按 ops-blas 仓库规范补充 `aclblasCtrmv` 算子 README，产品支持表标注 Ascend 950PR：支持 |