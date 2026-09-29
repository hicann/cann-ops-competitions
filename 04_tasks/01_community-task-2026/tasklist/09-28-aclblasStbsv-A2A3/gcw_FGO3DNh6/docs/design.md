# aclblasStbsv 算子设计文档（Atlas A2/A3, arch22）

# 需求背景（required）

## 需求来源

本需求来自 CANN 社区任务「9月社区任务-aclblasStbsv算子开发(A2/A3)」。任务要求在昇腾 NPU（Atlas A2 910B3 / A3）上使用 Ascend C 开发单精度实数三角带状求解算子 `aclblasStbsv`，功能、参数语义对标 cuBLAS `cublasStbsv` 与 Netlib `stbsv`，代码最终合入昇腾算子开源仓 ops-blas（https://gitcode.com/cann/ops-blas ）的 `blas/stbsv/arch22/` 目录，测试代码合入 `test/stbsv/arch22/`。

## 背景介绍

### aclblasStbsv 算子实现

BLAS Level-2 的 TBSV（Triangular Banded Solve）完成

```
x := op(A) * x
```

其中 A 为 n×n 三角带状矩阵（半带宽 k），以列主序带状（banded column-major）格式存储；x 为长度 n 的实数向量，原地（in-place）输入输出。TBSV 广泛用于三角方程的回代/前代求解、带状线性方程组的直接法以及科学计算中的三角变换。

本次任务的实现路径：

- 目标硬件：Atlas 800T A2（910B3）/ Atlas A3，CANN 9.1.0
- 编程语言：Ascend C（arch22 编程模型）
- 工程模式：**Ascend C kernel 直调**。Host 侧提供句柄式 BLAS 接口（`aclblasHandle_t` 携带 stream），通过 `<<<blocks, nullptr, stream>>>` 直接下发 NPU kernel，不经过 aclnn 图/原型注册链路
- 实现目录：`blas/stbsv/arch22/`（Host + Kernel + Tiling）
- 接口声明位置：`include/cann_ops_blas.h`（仓内已声明，本次补充 arch22 落地）

### 对标接口现状分析

| 接口 | 现状 | 说明 |
|---|---|---|
| `cublasStbsv`（cuBLAS） | 语义基线 | 参数序列 `handle, uplo, trans, diag, n, k, A, lda, x, incx`；支持 N/T、UPPER/LOWER、UNIT/NON_UNIT |
| `stbsv`（Netlib） | 参考实现 + golden | 与 cuBLAS 语义一致；golden 由 cblas 生成 |
| `aclblasStbsv`（ops-blas 本次新增） | 本次新增实现 | `include/cann_ops_blas.h` 已声明，需补充 arch22（A2/A3）实现 |

### aclblasStbsv 算子功能分析

- 计算 `x = op(A) * x`，`op(A) ∈ {A, Aᵀ}`（实数档 OP_C 按 Aᵀ 处理），原地覆写
- A 为三角带状矩阵，仅引用 uplo 指定的三角带内元素，带外恒为 0
- `diag = ACLBLAS_UNIT` 时主对角视为 1，**不读取 A 中对应对角位置**
- `n = 0` 为合法 no-op；`k = 0` 且 UNIT 时退化为单位阵求解；`incx` 支持正/负步长；`incx = 0` 非法
- 数据类型：float32

# 需求分析（required）

## 需求描述

使用 Ascend C 在 Atlas A2（910B3）/ A3 上实现 `aclblasStbsv`，接口签名、参数语义、异常返回与 cuBLAS `cublasStbsv` 逐参数对齐：

```cpp
aclblasStatus_t aclblasStbsv(
    aclblasHandle_t handle,
    aclblasFillMode_t uplo,
    aclasOperation_t trans,
    aclblasDiagType_t diag,
    int n,
    int k,
    const float* A,
    int lda,
    float* x,
    int incx);
```

接口声明在 `include/cann_ops_blas.h`（仓内已存在，本次仅补充实现），实现置于 `blas/stbsv/arch22/`，测试工程置于 `test/stbsv/arch22/`。

## 需求拆解

1. **功能**：uplo×trans×diag 8 组枚举组合全部正确；支持 incx 正/负步长；支持 n=0、k=0&&UNIT 等退化场景 quick return；支持 diag=UNIT 对角不读。
2. **接口**：与仓内已声明的 `aclblasStbsv` 逐参数对齐，遵循 `cann_ops_blas_common.h` 返回值约定，禁止私有平行 API。
3. **精度**：float32 混合容差，matched_ratio ≥ 0.99、max_abs_error ≤ 1e-2（具体阈值以任务书 §3.2 与生态算子开源精度标准为准）。
4. **性能**：任务书 §3.3 五档代表 case 平均单次耗时不高于标杆。
5. **工程**：Host + Kernel + Tiling 三件套，kernel 直调；测试工程含 CSV 用例与自测报告。
6. **自验**：覆盖基础、尺寸扫描、步长、填充（含 Inf/NaN）、边界负向、性能用例，输出自测报告。

# 详细设计（required）

## 算子分析

### 数学公式

```
x := op(A) · x

        ⎧ A      trans = ACLBLAS_OP_N
op(A) = ⎨ Aᵀ     trans = ACLBLAS_OP_T   （实数档 OP_C 按 Aᵀ 处理）
```

逐分量求解（以输出分量 i 为例）：取 `sum = x[i]`，减去已知分量的带内贡献 `Σ_{j≠i} A[i,j]·x[j]`，再按对角处理（`NON_UNIT` 时 `x[i] = sum / A[i,i]`；`UNIT` 时 `x[i] = sum`）。求解方向由 uplo 与 trans 决定（LOWER 且 N，或 UPPER 且 T → 前代；其余 → 回代）。

### 支持数据类型

| 张量/标量 | 类型 | 说明 |
|---|---|---|
| A | `const float*`（FLOAT32） | Device 内存，只读，列主序带状存储，长度 lda × n |
| x | `float*`（FLOAT32） | Device 内存，输入/输出，长度 1+(n-1)·\|incx\| |
| n / k / lda / incx | `int` | Host 标量 |
| uplo / trans / diag | 枚举（int） | Host 标量 |

### 支持形状

- A：n×n 三角带状矩阵，半带宽 k，列主序带状存储（lda ≥ k+1）
- x：长度 `1 + (n-1)·|incx|` 的一维数组（incx < 0 时从尾部反向遍历）
- 不涉及广播；n、k 为运行时入参

### 参数与异常行为矩阵

| 参数 | 合法值域 | 非法时返回 |
|---|---|---|
| handle | 非 nullptr | `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| uplo | `{ACLBLAS_UPPER, ACLBLAS_LOWER}` | `ACLBLAS_STATUS_INVALID_ENUM` |
| trans | `{ACLBLAS_OP_N, ACLBLAS_OP_T, ACLBLAS_OP_C}` | `ACLBLAS_STATUS_INVALID_ENUM` |
| diag | `{ACLBLAS_NON_UNIT, ACLBLAS_UNIT}` | `ACLBLAS_STATUS_INVALID_ENUM` |
| n | n ≥ 0（n = 0 为合法 no-op） | n < 0 → `ACLBLAS_STATUS_INVALID_VALUE` |
| k | k ≥ 0 | k < 0 → `ACLBLAS_STATUS_INVALID_VALUE` |
| lda | lda ≥ max(1, k+1) | lda ≤ k → `ACLBLAS_STATUS_INVALID_VALUE` |
| A / x | n > 0 时非 nullptr | `ACLBLAS_STATUS_INVALID_VALUE` |
| incx | incx ≠ 0 | `ACLBLAS_STATUS_INVALID_VALUE` |

校验顺序：handle → 枚举 → n/k/lda → 指针/incx → 退化场景 quick return。异常分支均在 Host 侧返回，不下发 kernel。

## 算子实现

### 工程结构与文件清单

```
include/cann_ops_blas.h                       # aclblasStbsv 声明（仓内已存在，本次补充 arch22 落地）
blas/stbsv/arch22/stbsv_host.cpp              # Host：校验、退化 quick-return、tiling、launch
blas/stbsv/arch22/stbsv_kernel.cpp            # Kernel：AIV-only 计算核 + 核函数下发
blas/stbsv/arch22/stbsv_tiling_data.h         # Tiling 结构体（Host/Kernel 共享）
test/stbsv/arch22/stbsv_test.cpp              # GTest 用例
test/stbsv/arch22/stbsv_test.csv              # 驱动用例
test/stbsv/stbsv_param.h                      # CSV 参数解析（共享，扩展 fill 解析）
test/stbsv/stbsv_golden.h                     # cblas stbsv golden（共享）
```

Host/Kernel 结构与仓内同族 `stbsv` 实现（`blas/stbsv/arch35/`）对齐：Host 声明 `void stbsv_kernel_do(...)`，Kernel 中以 `<<<numBlocks, nullptr, stream>>>` 下发；kernel 入口声明 `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY)`。

### 实现方案

#### Host 侧设计

Host 侧负责参数校验、退化场景 quick-return、tiling 数据构造与 kernel 下发：

- **参数校验与速返**：按上表逐项校验；`n == 0` 或 `k == 0 && UNIT` 等退化场景在启动前直接返回 `SUCCESS`（但仍保证负向用例的指针非空校验先执行）。
- **tiling 构造**：将 a/x 设备地址、n/k/uplo/trans/diag/lda/incx 等标量封装进 `StbsvTilingData`，传递给 kernel 侧。
- **kernel 下发**：通过 `stbsv_kernel_do(tiling, h->stream)` 异步下发；不主动 synchronize，与测试框架的 stream 同步协同。

#### Kernel 侧设计

Kernel 在 Ascend C AIV 核上完成逐分量三角回代/前代计算：

- **编译期特化**：针对 uplo×trans×diag 8 组枚举组合分别实例化计算核，消除运行时分支；trans 场景通过索引访问的对称处理统一折叠，无需单独代码路径。
- **原地求解与数据管理**：当前分量由 GM 取入、完成对角处理后写回 GM；已求解分量按需缓存在 UB，避免对历史 x 的反复读取。
- **对角处理**：`diag = UNIT` 时主对角项以 1 代入，对应对角地址的读取被编译期剔除，与 golden 语义一致（`diag = UNIT` 不读对角位的约定）。
- **incx 处理**：正/负步长统一映射为物理偏移，对正/负步长一致处理。
- **边界与退化**：`n = 0` 不启动 kernel；`k = 0 && UNIT` 在 Host 侧 quick-return；带内越界由带状索引天然规避。
- **数据类型与精度**：全程 float32；浮点结合顺序尽量与 Netlib golden 对齐，残余差异为 ULP 级噪声，满足任务书精度标准。

> 说明：本设计为开发前的方案说明，kernel 内部的访存组织与并行细节在开发过程中按硬件特性进一步细化与调优，最终以提交代码与自测报告为准。

### 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas A2 训练系列产品（910B3） / Atlas A2 推理系列产品 | √ |
| Atlas A3 训练系列产品 / Atlas A3 推理系列产品 | √ |

### 算子约束限制

- 仅支持 float32；不支持双精度及其他 dtype
- 不支持 `incx = 0`
- 不支持非连续 Tensor（超出 incx 语义的内存访问不支持）
- 不支持广播、不支持 dynamic shape 编译期特化（n/k 为运行时入参）
- A 必须为三角带状矩阵，列主序带状存储，lda ≥ k+1
- `diag = UNIT` 时实现不读取也不校验 A 对角位的内容
- 依赖 `aclblasSetStream` 绑定 stream，读回 Device 结果前需由调用方同步 stream

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | float32 混合容差：`\|actual−golden\| ≤ atol + rtol·\|golden\|`（FLOAT32 atol=rtol=2⁻¹³）；用例级 matched_ratio ≥ 0.99 且 max_abs_error ≤ 1e-2 | 任务书 §3.2 + 生态算子开源精度标准 |
| 性能标准 | 任务书 §3.3 五档代表 case 平均单次耗时 ≤ 标杆 | 任务书 §3.3 |
| 采集口径 | 先 warmup，再有效采样取均值（msprof 实测） | 任务书 §3.3 |

## 测试设计

测试工程基于 ops-blas `test/frame`（CSV 驱动 + GTest），golden 由 Netlib `cblas_stbsv` 生成，逐元素混合容差比对。

### 用例集（覆盖方向）

| 类别 | 覆盖点 |
|---|---|
| 基础 | uplo×trans×diag 8 组枚举 × 小尺寸 |
| 尺寸扫描 | 1 → 数千（含 1、质数、2 的幂及 ±1、非对齐值）× 枚举轮转 |
| 步长 | incx ∈ {±1, ±2, …} × 枚举组合 |
| 填充 | 均匀随机 / 全零 / 极端值 / Inf / NaN |
| 边界负向 | n=0 / k=0&&UNIT quick-return、空指针、非法枚举、n/k/lda/incx 非法 |
| 性能 | 任务书 §3.3 五档代表 case + 规模扫描 |

CSV 列格式与仓内 `stbsv_param.h` 的按名解析逻辑对齐（含 `uplo, trans, diag, n, k, lda, incx, a_fill, x_fill` 等字段，用于区分空指针负向场景的 buffer 分配）。

### 自验

- 精度：`build_debug/test/stbsv/stbsv_test` 全量用例，比对 golden
- 性能：`msprof` 对代表 case 采样，warmup 后取均值，与任务书 §3.3 标杆对照
- 交付：设计文档 PR、测试用例、用例结果自测报告、测试步骤指导文档

## 兼容性分析

- **新算子落地**：`include/cann_ops_blas.h` 原有声明不受影响，本次仅在 `blas/stbsv/arch22/` 补充实现，无 ABI 破坏
- 与同族 `stbsv` 实现（`blas/stbsv/arch35/`）共享测试框架与 golden，参数语义逐参数对齐
- 未定义私有平行 API，其他产品线可直接复用该声明
- 不修改现有已合入算子的任何行为

## 风险与待确认项

| # | 事项 | 影响 | 处置 |
|---|---|---|---|
| 1 | 大规模 case 的性能是否达标需经真机实测调优 | 中 | 开发完成后以 msprof 实测为准，必要时做访存/并行优化 |
| 2 | 仓库 `blas/stbsv/arch22/`、`test/stbsv/arch22/` 目录需新建 | 低 | 按仓内同族实现惯例搭建 |
| 3 | 任务书 §3.2/§3.3 具体阈值以官方任务书为准 | 低 | 自测报告按任务书口径如实填报 |
