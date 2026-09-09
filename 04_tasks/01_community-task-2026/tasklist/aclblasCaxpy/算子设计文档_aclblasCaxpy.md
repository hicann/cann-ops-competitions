# 需求背景（required）

## 需求来源

昇腾 CANN 训练营社区任务：`aclblasCaxpy` 算子开发（Atlas 950PR），任务书编号 aclblasCaxpy_Atlas950PR_task_doc.md。

## 背景介绍

### aclblasCaxpy 算子实现

基于 ops-blas 开源仓（https://gitcode.com/cann/ops-blas ）使用 Ascend C 编程语言为 Ascend 950PR（arch35）实现 BLAS Level-1 单精度复数向量线性组合更新算子 `aclblasCaxpy`。

对标接口：cuBLAS `cublasCaxpy`（语义参考 Netlib `caxpy`，https://www.netlib.org/blas/caxpy.f ）。

实现现状分析：

- ops-blas 仓 `blas/axpy/arch22/` 已存在旧版 caxpy（910B 平台，C220 底层 intrinsic 实现，无步长处理）；
- `blas/axpy/arch35/` 仅有 saxpy（float32 单精度）实现，**arch35 的 caxpy 需全新实现**，这是本任务主体；
- 接口声明已在 `include/cann_ops_blas.h`（309-311 行）存在：
  `aclblasStatus_t aclblasCaxpy(aclblasHandle_t handle, int n, const aclblasComplex* alpha, const aclblasComplex* x, int incx, aclblasComplex* y, int incy)`，**无需新增声明**；
- `aclblasComplex`（`{float real; float imag;}`）定义于 `include/cann_ops_blas_common.h`。

### aclblasCaxpy 算子功能分析

| 参数 | 参数含义 | 输入/输出 | 数据类型 | 支持数据类型 | 约束 | 形状 |
| --- | --- | --- | --- | --- | --- | --- |
| handle | 上下文句柄，携带 stream | 输入 | scalar | - | 有效句柄 | - |
| n | 向量 x、y 的元素个数 | 输入 | scalar | int | n ≥ 0 | - |
| alpha | 复数标量乘数指针（Host 内存） | 输入 | scalar | COMPLEX64 | n>0 时非空 | - |
| x | 复数向量（Device 内存，只读） | 输入 | tensor | COMPLEX64 | n>0 时非空 | 逻辑一维 [n]，物理长度 1+(n-1)\|incx\| |
| incx | x 连续元素步长，可正可负 | 输入 | scalar | int | ≠ 0 | - |
| y | 复数向量（Device 内存，原地更新） | 输出 | tensor | COMPLEX64 | n>0 时非空 | 逻辑一维 [n]，物理长度 1+(n-1)\|incy\| |
| incy | y 连续元素步长，可正可负 | 输入 | scalar | int | ≠ 0 | - |

计算公式（逐元素，i = 1..n，k = 1+(i-1)·incx，j = 1+(i-1)·incy，1-based 兼容 Fortran）：

```
y[j] = alpha * x[k] + y[j]
```

复数乘法按 `(a+bi)(c+di) = (ac-bd) + (ad+bc)i` 展开为实部/虚部分别计算。

支持数据类型：COMPLEX64（实部/虚部各 float32）。

支持正/负步长：指针指向首访问元素（逻辑元素 0），负步长沿低地址方向访问。

# 需求分析（required）

## 需求描述

使用 Ascend C 编程语言为 Ascend 950PR（arch35）实现 `aclblasCaxpy`，支持 complex64 数据类型、正/负步长、非对齐偏移、边界与负向输入语义，满足生态算子开源精度标准与任务书性能标杆，并完成 CSV 驱动 GTest 自测工程。

## 需求拆解

1. 功能语义对齐 cuBLAS `cublasCaxpy`：`y[j] = alpha * x[k] + y[j]`，复数乘法实部/虚部分别计算；
2. 支持正/负步长（incx/incy 可正可负，指针指向首访问元素，`ptr[i·inc] == 逻辑元素 i`）；
3. 返回码语义对齐任务书 §2.4：`n < 0` → `ACLBLAS_STATUS_INVALID_VALUE`；`n == 0` → 合法 no-op 返回成功；`n > 0` 时 alpha/x/y 空指针、incx/incy 为 0 → `ACLBLAS_STATUS_INVALID_VALUE`；handle 为空 → `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`；
4. 精度满足生态算子开源精度标准：实部/虚部分别按 FLOAT32（rtol=2^-10、atol=2^-16、matched_ratio≥0.99、max_abs_error≤1e-2）；
5. 性能满足任务书 §3.3 标杆耗时（warmup 后有效采样 >50 次取平均）及 gpu_baseline.csv 逐条倍率 ≥ 0.4；
6. 交付自测用例及测试代码（覆盖任务方 1200 条用例，CSV 驱动 GTest，可复现测试步骤）。

# 详细设计（required）

## 算子分析

### 数学公式

对每个逻辑元素 i（复数 alpha = ar + ai·i，x[i] = xr + xi·i，y[i] = yr + yi·i）：

```
tr = ar·xr − ai·xi        （实部乘项，与 Netlib caxpy 同运算次序）
ti = ar·xi + ai·xr        （虚部乘项）
yr' = tr + yr
yi' = ti + yi
```

### 支持数据类型

COMPLEX64（`aclblasComplex`，实部/虚部各 float32，32B 对齐单元 = 8 个 float = 4 个复数）。

### 支持形状

逻辑一维 [n]（n ≥ 0，int）；物理长度 `1 + (n-1)·|inc|`，任意对齐偏移（以复数计 4B 对齐即可，非对齐偏移 1..7 复数由 DataCopyPad 承载）。

## 算子实现

### 实现方案

#### 3.2.1 host 侧设计：

tiling 策略：按步长形态规划两条执行路径（tilingkey），数据视为一维逻辑向量，不考虑维度信息：

- **tilingkey = 0（连续路径）**：`incx == 1 && incy == 1`，走 AIV 向量内核，tile 流水 + 双缓冲；
- **tilingkey = 1（步长路径）**：其余情况（含正/负步长、非对齐偏移），走 SIMT 内核，带符号步长索引逐元素计算。

`CaxpyTilingData`（`blas/axpy/arch35/caxpy_tiling_data.h`）字段：`totalN / perCoreN / remainder / tileSize / incx(int64) / incy(int64) / useCoreNum / startOffset[64] / calCount[64] / nthreads / alphaReal / alphaImag`。alpha 以**值**传递（host 解引用 `*alpha` 填充两个字段），不分配设备内存。

##### 1. 分核策略：

优先使用满核的原则（`GetAivCoreCount` / 64 核上限）：

- 连续路径：`numBlocks = min(totalN, coreNum)`；每核 `perCoreN = CeilDiv(totalN, coreNum)` 按 8 元素（32B）对齐，余数归末核（`remainder = totalN − perCoreN·(coreNum−1)`）；
- 步长路径：`numBlocks = min(CeilDiv(n, 128), coreNum)`，`nthreads = min(CeilAlign(CeilDiv(n, coreNum), 128), 2048)`（约束在 [128, 2048] 内）；`startOffset[core]` / `calCount[core]` 逐核均分，保证分块不重不漏。

##### 2. 数据分块和内存优化策略：

充分使用 UB 空间的原则，同时考虑 double buffer 与计算临时空间：

- 连续路径单核内按 tile 切分：`tileSize = min(UB 容量约束, 3072)` 对齐 8；
- UB 预算核算：3 组深度 2 双缓冲输入队列（x 实/虚、y 实/虚共 4 个平面按 3 组组织）× tileSize×4B，加 6 个计算平面（ar·xr / ai·xi / ar·xi / ai·xr / 实部累加 / 虚部累加）：
  **单 tile 峰值 18 × 3072 × 4B = 216KB ≤ UB_SIZE 248KB**；
- 搬入搬出采用 `DataCopyPad`（支持任意对齐突发，规避 DataCopy 32B 对齐限制），`BUFFER_NUM = 2` 双缓冲流水。

##### 3. tilingkey 规划策略：

host 侧根据 incx/incy 形态确定 tilingkey 并填充对应 tiling 字段；tilingkey=0 时 `incx/incy` 字段为 1（AIV 内核直接使用），tilingkey=1 时携带带符号步长供 SIMT 内核索引。

#### 3.2.2 kernel 侧设计：

进行 Init 和 Process 两个阶段，Process 包括数据搬入（CopyIn）、计算（Compute）、搬出（CopyOut）三个阶段。复数运算拆解为实部/虚部两个 float32 平面并行处理。

**AIV 连续路径（tilingkey=0，`caxpy_kernel.cpp`）**：

1. CopyIn：`DataCopyPad` 按 2 元素突发从交织存储的复数组抽取 x 实部/虚部、y 实部/虚部四个 float 流（参考 cgeam/cherk 的 DeInterleave 思路），进入深度 2 双缓冲队列；
2. Compute：Muls 生成 ar·xr / ai·xi / ar·xi / ai·xr，Sub/Add 完成 `yr' = ar·xr − ai·xi + yr`、`yi' = ar·xi + ai·xr + yi`（与 Netlib 同运算次序，保证 golden 一致）；
3. CopyOut：Interleave 将实/虚平面交织为复数排列，`DataCopyPad` 写回 y。

**SIMT 步长路径（tilingkey=1）**：`asc_vf_call` 虚拟线程，`dim3{nthreads,1,1}` + `LAUNCH_BOUND(SIMT_MAX_THREAD_NUM)`，每个虚拟线程计算逻辑元素 i：

```
y[base + i·incy] = alpha·x[base + i·incx] + y[base + i·incy]
```

步长以带符号 int64 索引参与寻址，天然支持负步长（基址指向首访问元素，向低地址方向递增），无需额外反转。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas 950PR | √ |

## 算子约束限制

- 不支持广播（BLAS Level-1 向量语义，逐元素线性运算）；
- `n < 0` → `ACLBLAS_STATUS_INVALID_VALUE`；`incx`/`incy` 为 0 → `ACLBLAS_STATUS_INVALID_VALUE`；`n > 0` 时 alpha/x/y 为空 → `ACLBLAS_STATUS_INVALID_VALUE`；`n == 0` 为合法 no-op；
- 无额外设备内存分配（alpha 按值传递，tiling 经内核形参传递），不涉及任务书 §3.4 内存要求；
- Inf/NaN 输入按浮点语义传递（不报错），精度比对按特殊值规则匹配。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述(不涉及说明原因) | 标准来源 |
| --- | --- | --- |
| 精度标准 | 实部/虚部分别按 FLOAT32 生态精度标准：rtol=2^-10、atol=2^-16、required_matched_ratio≥0.99、max_abs_error_limit≤1e-2；golden 由 cblas（Netlib `caxpy`）单标杆生成，实部/虚部分别比对；Inf/NaN 按特殊值规则匹配 | 任务书 §3.2 / 生态算子开源精度标准 |
| 性能标准 | 任务书 §3.3 标杆：n=1048576/2097152/4194304（inc=1）→ 14.53/28.36/89.76 us（Avg，warmup 后有效采样 >50 次取平均）；gpu_baseline.csv 200 条逐条倍率 = 基线耗时/实测耗时 ≥ 0.4 | 任务书 §3.3 / gpu_baseline.csv |
| 内存标准 | 任务书 §3.4 明确"不涉及"；设计核算 UB 预算 216KB ≤ 248KB，无额外设备内存分配 | 任务书 §3.4 |

## 兼容性分析

- 接口复用 `include/cann_ops_blas.h` 既有声明（309-311 行），未新增/修改任何接口，可与其他产品线共用；
- arch35 为全新实现，不影响 arch22 旧版 caxpy（910B）与其他平台行为，无 ABI 变更；
- 测试工程与 ops-blas 仓 test 框架（BlasTest / csv_loader / fill / verify）完全兼容，`CMakeLists.txt` 按 `SOC_VERSION == ascend950` 分支接入 GTest 目标。
