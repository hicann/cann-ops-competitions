# aclblasCtbmv 算子设计文档（Ascend 950PR / arch35）

| 文档版本 | 日期 | 作者/团队 | 说明 |
| --- | --- | --- | --- |
| V1.0 | 2026-09-06 | will13546 | 按社区任务设计文档模板整理 Ascend 950PR（arch35）评审稿 |

> 本文档按照 CANN 社区任务 2026 `design_template.md` 编写。目标代码位于 `ops-blas/blas/tbmv/arch35/`，测试位于 `ops-blas/test/tbmv/ctbmv/arch35/`。本文只描述 Ascend 950PR（DAV_3510 / arch35）实现，不包含 A2/A3（arch22）方案。

# 一、需求背景

## 1.1 需求来源

本需求来源于 2026 年 8 月社区任务 `aclblasCtbmv`（Ascend 950PR）。任务要求在 `ops-blas` 中补齐单精度复数三角带状矩阵-向量乘接口，在 Ascend 950PR 上使用 Ascend C 完成 Host、Kernel、测试和性能验证，并与 cuBLAS `cublasCtbmv`、Netlib BLAS `ctbmv` 的功能及参数语义保持一致。

## 1.2 背景介绍

TBMV 属于 BLAS Level 2 算子。相比完整三角矩阵-向量乘，三角带状矩阵只保存主对角线附近 `k` 条副对角线或超对角线，存储量由 `n*n` 降为 `(k+1)*n`，理论计算量由 `O(n^2)` 降为 `O(n*k)`。该算子用于带状线性系统、有限差分、信号处理和稀疏结构明确的科学计算。

ops-blas 仓内已有实数同族接口 `aclblasStbmv`（arch35 已支持），以及公共 Handle/stream 机制与 BLAS 测试框架。本任务在复用公共工程能力的基础上新增：

1. complex64（`aclblasComplex`，实部/虚部各 float32）数据访问与复数乘加；
2. `trans=OP_C` 共轭转置路径（复数路径专有语义，取负虚部）；
3. 带步长复数向量的正向/反向逻辑索引（`incx<0` 从末端反向遍历）；
4. 原地覆盖输出、带状格式列主序寻址、`diag=UNIT` 时对角不读；
5. 与公共头文件 `include/cann_ops_blas.h` 共用、与 `aclblasStbmv` 同型的 `aclblasCtbmv` ABI。

## 1.3 功能分析

核心功能：`x = op(A) * x`（x 原地覆写）。`op(A)` 根据 `trans` 取值：

- `ACLBLAS_OP_N`：`op(A) = A`
- `ACLBLAS_OP_T`：`op(A) = Aᵀ`（**纯转置，不取共轭**）
- `ACLBLAS_OP_C`：`op(A) = Aᴴ`（**共轭转置，复数路径取共轭**）

A 按带状格式（banded storage）列主序存于 `lda×n` 数组，仅带内元素被引用：

- `uplo=UPPER`：主对角存于第 k+1 行，元素 `A(i,j)` 位于 `A(1+k+i-j, j)`（1-based）；数组左上角 k×k 三角区域不被引用
- `uplo=LOWER`：主对角存于第 1 行，元素 `A(i,j)` 位于 `A(1+i-j, j)`（1-based）；数组右下角 k×k 三角区域不被引用

`diag=UNIT` 时对角不读、视为 `(1,0)`；`diag=NON_UNIT` 时读取实际对角。`k ∈ [0, n-1]`，`k=0` 退化为对角矩阵。`n=0` 为合法 no-op。无 alpha/beta 标量，无确定性要求。

# 二、需求分析

## 2.1 需求描述

使用 Ascend C（SIMT）在 Ascend 950PR 上实现 `aclblasCtbmv`，支持：

- 数据类型：`aclblasComplex`（complex64）
- `uplo`：`ACLBLAS_UPPER` / `ACLBLAS_LOWER`
- `trans`：`ACLBLAS_OP_N` / `ACLBLAS_OP_T` / `ACLBLAS_OP_C`
- `diag`：`ACLBLAS_NON_UNIT` / `ACLBLAS_UNIT`
- 带状格式列主序、负步长、n=0 no-op、原地覆写

接口声明放入 `include/cann_ops_blas.h`，与 `aclblasStbmv` 同型（float→aclblasComplex），禁止定义 950PR 私有平行接口。

## 2.2 需求拆解

- 参数序列与 `cublasCtbmv` 一致：`handle, uplo, trans, diag, n, k, A, lda, x, incx`
- 仅引用 uplo 指定三角的带内元素；`diag=UNIT` 时不读对角
- `trans=OP_C` 时对取出的 A 元素取共轭（负虚部）
- 原地覆写 x：先写中间结果，再按 incx 步长写回；`k=0` 时对角就地缩放
- 非法枚举返回 `ACLBLAS_STATUS_INVALID_ENUM`；非法数值/空指针返回 `ACLBLAS_STATUS_INVALID_VALUE`；handle 为空返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`

# 三、详细设计

## 3.1 算子分析

### 数学公式

`x = op(A) * x`，其中 op 定义见 §1.3。

### 支持数据类型

A / x：`aclblasComplex { float real; float imag; }`（8 字节）。精度对标时实部、虚部分别按 `ACL_FLOAT`（FLOAT32）判定。

### 支持形状

- 方阵 `n×n`，`n ≥ 0`；半带宽 `k ∈ [0, n-1]`
- A 以带状格式列主序存储于 `lda×n`（`lda ≥ k+1`），仅带内元素有效
- x 为 `n` 元素向量，`incx ≠ 0`，支持负步长

## 3.2 算子实现

### 实现方案

采用「单值复数乘加 + 中间结果写回」的 SIMT fallback 方案，支持全部 uplo/trans/diag/incx 组合。

#### Host 侧（`ctbmv_host.cpp`）

tiling 策略：

1. **参数校验**：非法枚举（uplo/trans/diag）返回 `ACLBLAS_STATUS_INVALID_ENUM`；非法数值（n<0、k<0、lda<k+1、incx=0、空指针）返回 `ACLBLAS_STATUS_INVALID_VALUE`；handle 为空返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`；`n=0` 合法 no-op 返回成功。
2. **分核策略**：`numBlocks = max(min(ceil(n / SIMT_MIN_THREAD_NUM), aivCoreNum), 1)`，将 n 行任务均分到 AIV 核。
3. **线程/块内切分**：`numThreads = min(CeilAlign(CeilDiv(n, numBlocks), SIMT_MIN_THREAD_NUM), SIMT_MAX_THREAD_NUM)`，`rowsPerBlock = CeilDiv(n, numBlocks)`。
4. **UB x 缓存开关**：`useUb = (incx == 1 && n >= 32 && rowsPerBlock + k <= CTBMV_UB_X_ELEMS)`，将块内所需 x 窗口缓存到 UB。
5. **Workspace**：`k>0` 时需 `n × sizeof(aclblasComplex)` 的 device 中间结果缓冲（从 handle workspace 获取）；`k=0` 时退化为对角就地缩放，无需 workspace。
6. 通过 handle 绑定 stream，`aclblasCtbmv` 以句柄式接口直调 NPU kernel。

#### Kernel 侧（`ctbmv_fallback_kernel.cpp`）

SIMT 核内实现，分为三类 kernel：

1. **compute kernel**（k>0）：对每个输出行 `row` 求解列区间 `[colStart, colEnd)`，累加 `op(A)[row][col] * x[col]`，结果暂存 `midOutGm[row]`（workspace）。利用 `__simt_vf__` 行并行分布到 block/thread。
   - 模板参数：`UPLO_IS_UPPER`、`TRANS_IS_N`、`CONJ`、`DIAG_IS_UNIT`。
   - 带状索引（0-based，列主序）：
     - UPPER：`A(row, col) 位于 band[ k + row - col + col*lda ]`
     - LOWER：`A(row, col) 位于 band[ row - col + col*lda ]`
   - 列区间：`trans=N` 取 `A` 的带内行/列；`trans=T/C` 交换 row/col 访问（即取 `Aᵀ/Aᴴ` 的带内行/列）。
   - `trans=C` 时对取出的 A 元素取负虚部（共轭）。
   - 复数乘加按实部/虚部分别累加：`accRe += aRe*xRe - aIm*xIm; accIm += aRe*xIm + aIm*xRe`。
2. **copy kernel**：将 `midOutGm[n]` 按 `incx` 步长写回 x（`CtbmvCopyBack`）。
3. **diag kernel**（k=0）：就地缩放 `x[row] = op(A)[row][row] * x[row]`；`trans=C` 时对 A 对角共轭。

###### 模板化分派

- `CtbmvDispatchTrans<UPLO_IS_UPPER>`：按 `trans`（N/T/C）选择 `CONJ` 与 `TRANS_IS_N` 组合，按 `useUb` 选择 GM 直算 / UB 缓存路径。
- 每个模板实例仅展开实际使用的分支，避免运行时分支开销，保证只读取带内/对角元素。

### 支持硬件

支持的芯片版本 | 涉及勾选
---|---
Ascend 950PR | √

（A2/A3、A3 训练/推理系列不在本方案范围。）

### 算子约束限制

- 仅引用 uplo 指定三角的带内元素；`diag=UNIT` 时不读对角。
- 不支持超出 `lda`/`incx` 语义的非连续内存访问（任务明确不要求）。
- 不涉及广播；`n`/`k` 为运行时入参，不要求 dynamic shape。
- 不要求确定性计算。

# 四、可维可测分析

## 4.1 精度标准 / 性能标准

| 验收标准 | 描述 | 标准来源 |
|---|---|---|
| 精度标准 | golden 由 cblas（Netlib `ctbmv`）生成，x 全量验证。实部/虚部分别按 FLOAT32：atol=2⁻¹⁶、rtol=2⁻¹⁰、matched_ratio ≥ 0.99、max_abs_error ≤ 1e-2 或 32×ULP | 生态算子开源精度标准 |
| 性能标准 | 各性能 case 平均单次耗时 ≤ 标杆耗时/倍率（倍率阈值 0.4）；先 warmup 再有效采样 >50 次取平均 | 任务书 §3.3 |

## 4.2 兼容性分析

- 接口声明放入 `include/cann_ops_blas.h`，与 `aclblasStbmv` 同型，可与其他产品线共用。
- `aclblasComplex` 与 OpenBLAS/cblas 复数布局二进制兼容，便于 CPU golden 直接复用 `cblas_ctbmv`。
- 不在 950PR 上定义私有平行接口。

# 五、交付文件清单

| 类型 | 路径 | 说明 |
|---|---|---|
| 算子实现 | `ops-blas/blas/tbmv/arch35/ctbmv_common.h` | 公共定义、tiling 结构、kernel_do 声明 |
| | `ops-blas/blas/tbmv/arch35/ctbmv_host.cpp` | Host 校验、tiling、kernel 启动 |
| | `ops-blas/blas/tbmv/arch35/ctbmv_fallback_kernel.cpp` | SIMT fallback kernel（全部组合） |
| 接口声明 | `ops-blas/include/cann_ops_blas.h` | 新增 `aclblasCtbmv` 声明 |
| 测试代码 | `ops-blas/test/tbmv/ctbmv/ctbmv_param.h` | CSV 参数解析 |
| | `ops-blas/test/tbmv/ctbmv/ctbmv_golden.h` | cblas CPU golden |
| | `ops-blas/test/tbmv/ctbmv/ctbmv_test_util.h` | 复数带状/步长数据生成与精度校验 |
| | `ops-blas/test/tbmv/ctbmv/arch35/ctbmv_test.cpp` | GTest 驱动用例 |
| | `ops-blas/test/tbmv/ctbmv/arch35/ctbmv_npu_wrapper.h` | Device 数据搬运封装 |
| | `ops-blas/test/tbmv/ctbmv/arch35/ctbmv_test.csv` | 1200 条自测用例 |
| | `ops-blas/test/tbmv/ctbmv/CMakeLists.txt` | 测试注册 |
| | `ops-blas/test/tbmv/ctbmv/README.md` | 测试步骤说明 |
| 算子 README | `ops-blas/blas/tbmv/README.md` | 更新产品支持表与接口说明 |
