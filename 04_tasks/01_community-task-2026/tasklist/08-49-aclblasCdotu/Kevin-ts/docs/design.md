# aclblasCdotu 算子设计文档

> 任务：9 月社区任务——Ascend 950PR aclblasCdotu 算子开发
> 目标硬件：Ascend 950PR（`arch35`）
> CANN 版本：9.1.0
> 合入仓库：https://gitcode.com/cann/ops-blas（目录 `blas/dot/arch35/`）

## 1. 需求背景

### 1.1 需求来源

社区任务要求在 Ascend 950PR 上使用 Ascend C 编程语言开发单精度复数（complex64）向量无共轭点积算子 `aclblasCdotu`，完成算子设计、开发、测试全流程。接口风格与 cuBLAS `cublasCdotu` 完全对齐，语义参考 Netlib `cdotu`。

计算公式为：

```text
result = Σ( x[k] × y[j] )   (k = 1+(i-1)·incx, j = 1+(i-1)·incy, i = 1..n, 1-based)
```

复数乘法语义：`(a+bi)(c+di) = (ac-bd) + (ad+bc)i`

```text
result.real = Σ( x[k].real × y[j].real − x[k].imag × y[j].imag )
result.imag = Σ( x[k].real × y[j].imag + x[k].imag × y[j].real )
```

**与 cublasCdotc 的关键区别**：cuBLAS 文档明确仅函数名以 'c' 结尾时对向量 x 取共轭，`cublasCdotu` 不取共轭。本算子实现不取共轭。

### 1.2 背景介绍

`aclblasCdotu` 是 BLAS Level-1 复数点积算子，计算两个复数向量的无共轭点积。这是昇腾 ops-blas 仓 arch35 路径下首个复数 BLAS 算子，其开发将建立复数类型在 Ascend C 下的归约计算模式，为后续复数 BLAS 算子（`cdotc`、`chemm`、`cgemm` 等）提供参考模板。

本设计以精度、正确性和可维护性为优先，复用 arch35 已有实数 `sdot` 的成熟多核分段归约框架，基于 Ascend C kernel 直调方式实现。

## 2. 需求分析

### 2.1 需求描述

首版必须支持：

- `aclblasComplex` 类型（`{float real, float imag}`），与 `cann_ops_blas_common.h` 定义一致；
- 复数向量无共轭点积，累加实部与虚部；
- 句柄式 BLAS 接口，通过 handle 绑定 stream 直调 NPU kernel；
- 支持任意非零 `incx`/`incy`（含负步长），负步长索引语义对齐 Netlib `cdotu`；
- `n = 0` 为合法 no-op，直接返回成功并将 result 置 (0, 0)；
- `n < 0`、`incx = 0`、`incy = 0`、空指针等非法参数返回对应错误码；
- 接口声明放入 `include/cann_ops_blas.h`（已存在），禁止定义 950PR 私有平行接口。

### 2.2 需求拆解

| 子问题 | 设计结论 |
|---|---|
| 复数乘法 | 逐元素计算 `(ac-bd) + (ad+bc)i`，实部虚部分别累加 |
| 数据类型 | 内核以 `float` 数组形式处理复数（`[real, imag, real, imag, ...]`），每元素 2×float |
| 连续路径 | 当 `incx=1, incy=1, incx>0, incy>0` 时走连续路径，块内用 Mul+ReduceSum 加速 |
| 跨步路径 | 任意步长走逐元素跨步路径，含负步长支持 |
| 多核分段 | 对 n 个复数元素平均分配到各 core，每 core 独立计算部分和 |
| 跨核归约 | core0 收集所有 core 的部分和，ReduceSum 得到最终结果 |
| 空操作 | `n = 0` 时直接返回成功，将 result 置 (0, 0) |
| 参数校验 | handle 空指针、n 负数、incx=0、incy=0、x/y/result 空指针分别返回对应错误码 |
| 构建系统 | 复用仓库 blas/CMakeLists.txt 自动收集机制，按 `arch35` 目录组织 |

## 3. 详细设计

### 3.1 算子分析

对 `n` 个复数元素，第 `i` 个参与计算的元素地址为：

- 正步长：`x[i * incx]`、`y[i * incy]`
- 负步长：`x[(n-1+i) * (-incx)]`、`y[(n-1+i) * (-incy)]`（等效于末尾反向遍历）

每个参与计算的复数元素对乘积的贡献为：

```text
prod.real = x_elem.real × y_elem.real − x_elem.imag × y_elem.imag
prod.imag = x_elem.real × y_elem.imag + x_elem.imag × y_elem.real
```

所有 n 个乘积的实部与虚部分别累加，得到最终结果。

### 3.2 总体组件

```text
Public API: aclblasCdotu
  ├─ Validate: handle 空指针 → HANDLE_IS_NULLPTR
  │             n < 0 → INVALID_VALUE
  │             incx == 0, incy == 0 → INVALID_VALUE
  │             x / y / result 空指针 (n>0) → INVALID_VALUE
  ├─ No-op: n == 0 → aclrtMemset result 置零, 返回 SUCCESS
  ├─ Tiling: 计算 useCoreNum、每 core 的 startIdx 和 calCount
  ├─ Launch: 分配 workspace（useCoreNum × 2 × sizeof(float)），启动 kernel
  └─ 依赖: aclrtSynchronizeDevice 由调用方在新 stream 或读回前执行
```

### 3.3 算子实现

主要文件职责：

| 路径 | 职责 |
|---|---|
| `blas/dot/arch35/cdotu_tiling_data.h` | Host/Device 共用 tiling 数据结构 |
| `blas/dot/arch35/cdotu_host.cpp` | 参数校验、tiling 计算、workspace 分配、kernel 启动 |
| `blas/dot/arch35/cdotu_kernel.cpp` | Ascend C kernel 实现：复数乘加、多核分段归约、跨核汇总 |
| `test/dot/cdotu/arch35/cdotu_test.cpp` | GTest 测试入口，CSV 驱动 |
| `test/dot/cdotu/arch35/cdotu_test.csv` | 测试用例矩阵 |
| `test/dot/cdotu/arch35/cdotu_npu_wrapper.h` | H2D/D2H 封装 |
| `test/dot/cdotu/arch35/cdotu_golden.h` | CPU golden（cblas_cdotu_sub） |
| `test/dot/cdotu/arch35/cdotu_param.h` | 测试参数结构体 |
| `test/dot/cdotu/CMakeLists.txt` | 测试构建配置 |

#### 3.3.1 接口定义

```cpp
aclblasStatus_t aclblasCdotu(
    aclblasHandle_t handle, int n, const aclblasComplex* x, int incx,
    const aclblasComplex* y, int incy, aclblasComplex* result);
```

已在 `include/cann_ops_blas.h:184` 声明，禁止定义 950PR 私有平行接口。

#### 3.3.2 参数校验

```text
handle == nullptr          → ACLBLAS_STATUS_HANDLE_IS_NULLPTR
n < 0                      → ACLBLAS_STATUS_INVALID_VALUE
n > 0 && incx == 0         → ACLBLAS_STATUS_INVALID_VALUE
n > 0 && incy == 0         → ACLBLAS_STATUS_INVALID_VALUE
n > 0 && x == nullptr       → ACLBLAS_STATUS_INVALID_VALUE
n > 0 && y == nullptr       → ACLBLAS_STATUS_INVALID_VALUE
n > 0 && result == nullptr  → ACLBLAS_STATUS_INVALID_VALUE
n == 0                     → 直接返回 SUCCESS（result 置零）
```

#### 3.3.3 Tiling 数据

```cpp
struct CdotuTilingData {
    int64_t n;          // 复数元素个数
    int64_t incx;       // x 步长
    int64_t incy;       // y 步长
    uint32_t useCoreNum; // 实际使用的 AI Core 数量
};
```

Tiling 计算逻辑：

1. `useCoreNum = min(GetAivCoreCount(), n)`，n=0 时取 1；
2. 将 n 个复数元素平均分配到 useCoreNum 个 core；
3. 每 core 的 `startIdx` 和 `calCount` 在 kernel 端按 `blockIdx` 计算，无需放入 tiling。

#### 3.3.4 Workspace 计算

```text
workspaceSize = useCoreNum × 2 × sizeof(float)  // 实部+虚部，每 core 一对 float
```

不额外分配 tiling device buffer——tiling 数据通过 kernel 启动参数传递（参考 sdot 模式）。

#### 3.3.5 Kernel 启动

```cpp
cdotu_kernel_do(x_gm, y_gm, result_gm, workspace_gm, useCoreNum, tiling, stream);
```

### 3.4 Kernel 设计

#### 3.4.1 算法概述

每个 AI Core 负责一段连续或跨步的复数元素，独立计算部分和（实部、虚部各一），写入 workspace 中对应 core 的位置。所有 core 完成计算后通过 CrossCore 同步，core 0 收集所有 partial sum 并做最终 ReduceSum。

#### 3.4.2 连续路径（ProcessContiguous）

触发条件：`incx > 0 && incy > 0 && incxAbs == 1 && incyAbs == 1`

- 使用双缓冲（xQueue、yQueue）从 GM 批量搬运复数数据到 UB；
- 复数数据以 `float` 数组形式存储：`[real0, imag0, real1, imag1, ...]`；
- 每 chunk 处理 chunkSize 个复数元素（即 2×chunkSize 个 float）；
- 在 UB 中分离实部/虚部，计算复数乘法，分别累加：
  - `accReal += xRe * yRe - xIm * yIm`
  - `accImag += xRe * yIm + xIm * yRe`
- 使用 ReduceSum 对 chunk 内结果归约，再累加到累加器。

UB 分配策略（参考 sdot 的 SAFETY_MARGIN 方案）：

```text
UB_SIZE = 248 × 1024 = 253952 bytes
availFloats = (UB_SIZE - SAFETY_MARGIN) / sizeof(float)  // SAFETY_MARGIN = 20480
fixedCost = ACCUMULATOR + SHARED_TMP                                    // 256 + 64 = 320
chunkSize = (availFloats - fixedCost) / (BUFFER_NUM × 2 + 1)           // 除以 5
```

累加器：`ACCUMULATOR_FLOATS = 64`，实际使用前 2 个 float（实部、虚部），其余 zero-padding 对齐 vector 单次操作宽度。

#### 3.4.3 跨步路径（ProcessStrided）

触发条件：任意 `incx ≠ 1` 或 `incy ≠ 1` 或负步长

- 逐元素处理，每元素从 GM 读取一对复数到 UB；
- 负步长下标计算：
  ```text
  xOffset = incxPos ? (startIdx + i) × incxAbs : (n - 1 - (startIdx + i)) × incxAbs
  yOffset = incyPos ? (startIdx + i) × incyAbs : (n - 1 - (startIdx + i)) × incyAbs
  ```
- 每次读取一个复数（2×float），计算复数乘法后累加；
- 不使用双缓冲和 ReduceSum，直接逐元素 Mul+Add。

#### 3.4.4 跨核归约

所有 core 完成部分和计算后，通过 `CrossCoreSetFlag`/`CrossCoreWaitFlag` 同步。core 0 从 workspace 读取所有 core 的部分和（useCoreNum × 2 个 float），对实部、虚部分别做 ReduceSum，得到最终结果并写入 result。

#### 3.4.5 空操作

当 `calCount == 0`（该 core 无分配元素）时，直接将部分和置零并写入 workspace。

### 3.5 状态码

| 场景 | 返回状态码 |
|---|---|
| handle 为 nullptr | `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| n < 0 | `ACLBLAS_STATUS_INVALID_VALUE` |
| incx == 0（n > 0） | `ACLBLAS_STATUS_INVALID_VALUE` |
| incy == 0（n > 0） | `ACLBLAS_STATUS_INVALID_VALUE` |
| x 为 nullptr（n > 0） | `ACLBLAS_STATUS_INVALID_VALUE` |
| y 为 nullptr（n > 0） | `ACLBLAS_STATUS_INVALID_VALUE` |
| result 为 nullptr（n > 0） | `ACLBLAS_STATUS_INVALID_VALUE` |
| n == 0 | `ACLBLAS_STATUS_SUCCESS`（result 置零） |
| workspace 不足 | `ACLBLAS_STATUS_EXECUTION_FAILED` |
| GetAivCoreCount 失败 | `ACLBLAS_STATUS_EXECUTION_FAILED` |
| Kernel 正常执行 | `ACLBLAS_STATUS_SUCCESS` |

## 4. 支持硬件

| 硬件 | 目录 | 设计定位 |
|---|---|---|
| Ascend 950PR | `blas/dot/arch35` | 目标硬件与实现目录 |
| A2/A3 | `blas/dot/arch22` | 既有实现目录（仅支持 inc=1） |

arch35 新增实现不替换 arch22 的已有 `cdotu`/`cdotc` 实现，两者通过 CMake 的 SOC_ARCH_DIRS 机制自动区分。接口声明 `include/cann_ops_blas.h` 为公共头文件，各架构共用。

## 5. 算子约束限制

- 仅支持 `aclblasComplex` 类型（`{float real, float imag}`），不支持双精度复数；
- `n` 为 `int` 类型，取值范围 `[0, INT_MAX]`；
- `incx`、`incy` 为 `int` 类型，取值范围 `INT_MIN` 到 `INT_MAX`，但 `0` 为非法值；
- `x`、`y`、`result` 为 Device 内存指针，非 Host 内存；
- 不涉及非连续 Tensor（向量非连续由 incx/incy 表达，无额外 leading dimension padding）；
- 不涉及 broadcast 规则；
- 不涉及 dynamic shape（n 为运行时入参）；
- 不涉及原地更新（result 为独立输出标量）；
- 不要求确定性计算（归约累加顺序不保证跨次一致）；
- 异步执行：依赖 `aclblasSetStream` 绑定 stream，读回 Device 结果前须同步 stream。

## 6. 可维可测分析

### 6.1 测试分层

| 层级 | 内容 | 判定 |
|---|---|---|
| 参数校验 | handle 空、n<0、incx=0、incy=0、x/y/result 空指针、n=0 no-op | 返回码与预期一致 |
| 小 shape 基础 | n=1, 2, 3, 7, 8（小质数、2 的幂、单个元素） | 精度通过 |
| shape 扫描 | 2 的幂及 2 的幂 ±1，直至大规模（≥2^20） | 精度通过 |
| 填充模式 | 全零、交替符号、均匀分布、正态分布 | 精度通过 |
| 步长扫描 | incx/incy = 1, 2, -1, -2 及非常规步长 | 精度通过 |
| 边界与负向 | n=0、incx=0、incy=0、空指针、负步长大 n | 返回码/精度正确 |
| 特殊值 | INF、NaN、INF×NaN 等 | 行为对齐 cublas |
| 性能 | 3 个标杆 case（n=1M, 2M, 4M, inc=1） | 平均耗时 ≤ 标杆 |
| 内存 | 无内存泄漏，workspace 检查 | 正常 |

### 6.2 数据生成协议

测试数据由 CSV 文件描述，Python 脚本调用 C++ GTest 工程加载 CSV 调用 `aclblasCdotu` 接口执行。

- 均匀分布 `[-5, 5]` 占 50%，正态分布（μ∈[-5,5], σ∈[0.1,2]）占 50%，实部/虚部独立采样；
- 包含 INF/NaN 特殊值用例；
- 精度 golden 由 cblas（Netlib BLAS 复数实现 `cdotu`）生成；
- 随机种子固定，保证用例可复现。

### 6.3 精度标准

按生态算子开源精度标准，本算子输入输出为 COMPLEX64，实部、虚部分别按 FLOAT32 分量判定：

| 数据类型 | rtol | atol | required_matched_ratio | max_abs_error_limit |
|---|---|---|---|---|
| COMPLEX64（实部/虚部按 FLOAT32 分量） | 2^-10 (9.77e-4) | 2^-16 (1.53e-5) | 0.99 | 1e-2 或 32 × ULP |

逐元素通过条件：`|actual − golden| ≤ atol + rtol × |golden|`

当用例同时满足 `matched_ratio ≥ required_matched_ratio` 且 `max_abs_error ≤ max_abs_error_limit` 时，判定该用例精度通过。

**补充说明**：本算子为 n 项复数乘加的归约累加运算，误差随 n 累积；`matched_ratio ≥ 0.99` 允许少量离群点，以适配大 n 下的累加舍入误差。

### 6.4 性能标准

测试设备：Ascend 950PR。性能数据为 COMPLEX64 输入场景下的平均单次耗时（Avg time，单位 us），须先 warmup 再有效采样 >50 次取平均。

| case | n | incx | incy | 标杆耗时（Avg time, us） |
|---|---|---|---|---|
| 1 | 1048576 | 1 | 1 | 25.31 |
| 2 | 2097152 | 1 | 1 | 30.25 |
| 3 | 4194304 | 1 | 1 | 74.85 |

### 6.5 可维护性

- 参数校验、tiling、kernel 分层，职责清晰；
- 连续路径与跨步路径共用同一套复数乘法语义，通过 `ProcessContiguous`/`ProcessStrided` 分离数据搬运策略；
- 常量（SAFETY_MARGIN、ACCUMULATOR_FLOATS、ELEMENTS_PER_BLOCK 等）集中定义，性能调优不改变公共 API 语义；
- 测试报告绑定源代码版本和输入指纹，不依赖历史结果。

## 7. 兼容性分析

- `cann_ops_blas.h` 中 `aclblasCdotu` 声明已存在，本设计不新增或修改公共头文件声明；
- arch35 新实现不替换 arch22 的已有 `cdotu`/`cdotc` 实现，但公共头文件为各架构共用，接口签名须保持一致；
- arch22 的 `cdotu` 实现仅支持 `incx=1, incy=1`，arch35 实现支持任意非零步长（含负步长），功能超集；
- 构建系统通过 `SOC_ARCH_DIRS` 自动选择对应架构源码，无需修改顶层 CMakeLists.txt；
- 测试代码合入 `test/dot/cdotu/arch35/`，参考主仓 `test/dot/sdot/arch35/` 结构。

## 8. 交付件清单

| 序号 | 交付件 | 说明 |
|---|---|---|
| 1 | 算子设计文档 | 本文档 |
| 2 | 算子实现代码 | `blas/dot/arch35/` 目录下 cdotu_tiling_data.h、cdotu_host.cpp、cdotu_kernel.cpp |
| 3 | 自测用例及测试代码 | `test/dot/cdotu/arch35/` 目录下 CSV 驱动的 GTest 测试 |
| 4 | 自测报告 | 含用例参数、精度对比（实部/虚部分别）、性能数据 |
| 5 | README 文档 | 产品支持表标注 Ascend 950PR：支持 |