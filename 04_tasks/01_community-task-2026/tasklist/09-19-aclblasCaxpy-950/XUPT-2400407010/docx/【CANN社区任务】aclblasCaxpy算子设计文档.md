# aclblasCaxpy 算子设计文档（Ascend 950PR）

| 项目 | 内容 |
| --- | --- |
| 算子名称 | aclblasCaxpy（BLAS L1 复数向量线性组合更新） |
| 对标接口 | cuBLAS `cublasCaxpy` / Netlib BLAS `caxpy` |
| 开发语言 | Ascend C / CATLASS |
| 适配硬件 | Ascend 950PR |
| CANN 版本 | CANN 9.1.0 |
| 数据类型 | COMPLEX64（实部/虚部各 float32） |
| 代码目录 | ops-blas 仓 `blas/axpy/arch35/` |
| 接口声明 | ops-blas 仓 `include/cann_ops_blas.h` |
| 文档日期 | 2026-09-10 |

---

# 需求背景（required）

## 需求来源

本算子来自 CANN 训练营西安邮电大学社区任务「aclblasCaxpy 算子开发」，验收通过后合入昇腾算子开源仓（ops-blas）。

- 任务书：`aclblasCaxpy_Atlas950PR_task_doc.md`
- 开源仓：https://gitcode.com/cann/ops-blas
- 设计文档模板：https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md
- 生态算子开源精度标准：https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md

## 背景介绍

### aclblasCaxpy 算子开发

在昇腾 NPU（Ascend 950PR）上使用 Ascend C/CATLASS 编程语言开发单精度复数（complex64）向量线性组合更新算子 `aclblasCaxpy`，完成算子设计、开发、测试全流程工作。算子基于 ops-blas 开源仓工程框架，采用 Ascend C kernel 直调方式开发，通过 handle 绑定 stream 直调 NPU kernel，对标 GPU 生态 cuBLAS 同名接口，补齐昇腾生态 BLAS L1 复数向量能力。

### 现状分析

| 现状项 | 说明 |
| --- | --- |
| 参考实现（GPU） | cuBLAS `cublasCaxpy`，语义参考 Netlib BLAS `caxpy`（https://www.netlib.org/blas/caxpy.f ） |
| 生态现状（昇腾） | ops-blas 仓 `axpy` 族当前仅有硬编码冒烟测试（`test/axpy/caxpy/arch22/caxpy_test.cpp`），尚无 arch35 实现、无 CSV 驱动测试工程，属从零开发并配套测试工程 |
| 接口现状 | `include/cann_ops_blas.h` 中已有 `aclblasCaxpy` 声明，本任务按既有声明实现，禁止定义 950PR 私有平行接口 |
| 精度 golden | 由 cblas（Netlib BLAS 复数实现）生成，随测试工程提供，无其他三方软件依赖 |

### aclblasCaxpy 算子功能分析

- **功能公式**：对 n 个元素执行 `y[j] = alpha * x[k] + y[j]`（i = 1..n，k = 1+(i-1)\*incx，j = 1+(i-1)\*incy，1-based 索引兼容 Fortran）
- **输入**：`handle`、`n`、`alpha`（复数标量）、`x`（复数向量）、`incx`、`incy`
- **输出**：`y`（原地更新为 `alpha*x + y`）
- **数据语义**：复数乘法 `(a+bi)(c+di) = (ac-bd) + (ad+bc)i`；复数加法实部/虚部分别相加；复数类型 `aclblasComplex` 以 ops-blas 仓 `include/cann_ops_blas_common.h` 定义为准（实部/虚部各 float32）
- **步长语义**：`incx`/`incy` 可正可负，负步长表示从低地址起始、向回取数；x/y 逻辑一维 `[n]`，物理长度 `1+(n-1)*|inc|`

# 需求分析（required）

## 需求描述

在 Ascend 950PR 上实现 BLAS L1 复数向量线性组合更新算子 `aclblasCaxpy`：`y[j] = alpha * x[k] + y[j]`，其中 `alpha`、`x`、`y` 均为单精度复数（COMPLEX64），支持 `incx`/`incy` 正负步长与原地更新语义，通过句柄式 BLAS 接口（handle 绑定 stream）直调 NPU kernel。

## 需求拆解

1. **接口兼容**：接口签名与 ops-blas 仓 `include/cann_ops_blas.h` 中既有声明完全一致，参数序列与 `cublasCaxpy` 一一对应；禁止定义 950PR 私有平行接口。
2. **功能正确**：实现复数乘加语义 `y = alpha*x + y`（实部/虚部分别计算），支持 `incx`/`incy` 正负步长寻址与原地更新。
3. **数据类型**：支持 COMPLEX64（实部/虚部各 float32），精度按实部/虚部分别判定。
4. **边界与异常**：`n = 0` 为合法 no-op（返回成功且不修改 y）；`n < 0`、`incx = 0`、`incy = 0`、`n > 0` 时 `alpha`/`x`/`y` 为空指针均返回 `ACLBLAS_STATUS_INVALID_VALUE`；`handle` 为空返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`。
5. **精度达标**：满足生态算子开源精度标准（complex64 按 FLOAT32 分量判定），不低于 golden（cblas 生成）精度。
6. **性能达标**：各性能 case 平均单次耗时（Avg time，us）不高于任务书标杆耗时；性能测试须先 warmup 再有效采样 >50 次取平均。
7. **工程规范**：代码合入 ops-blas 仓 `blas/axpy/arch35/`，测试代码合入 `test/axpy/caxpy/arch35/`（CSV 驱动，结构参考主仓 blas 算子测试代码），交付算子 readme 与产品支持表（Ascend 950PR：支持）。

# 详细设计（required）

## 算子分析

### 数学公式

```
y[j] = alpha * x[k] + y[j]
  i = 1..n
  k = 1 + (i-1) * incx
  j = 1 + (i-1) * incy
```

复数展开（设 `alpha = a + bi`，`x = c + di`，`y = e + fi`）：

```
实部：y.re = (a*c - b*d) + e
虚部：y.im = (a*d + b*c) + f
```

### 数据流与内存视图

- x/y 为逻辑一维向量，逻辑长度 n；物理长度为 `1 + (n-1) * |inc|`（步长为 1 时物理长度等于 n，为连续访问）。
- 步长可正可负：正步长从起始元素向高地址方向取数；负步长从起始元素向低地址方向取数。kernel 侧按逻辑索引 `(i-1)*inc` 映射物理地址，`|inc| = 1` 时访存连续、无空洞。
- y 为原地输出：计算完成直接写回 y 原地址，不返回视图。

### 支持数据类型

| 类型 | 说明 |
| --- | --- |
| COMPLEX64 | 单精度复数，实部/虚部各 float32，共 8 字节/元素；与 `aclblasComplex` 定义一致 |

### 支持形状与广播

- 本算子为两个一维向量的逐元素线性组合更新，**不涉及广播**（任务书 §2.5 明确 broadcast 规则为"不涉及"）。
- 非连续 Tensor（leading dimension padding 场景）本批次不支持，向量非连续统一由 `incx`/`incy` 表达。
- dynamic shape：不要求，n 为运行时入参。

### 精度分析

本算子为逐元素线性运算，误差来源仅为单次复数乘加的浮点舍入（1 次乘 + 1 次加，按实部/虚部两次 FLOAT32 运算），正常实现应远优于生态精度标准阈值。无随机数生成、无迭代累积误差，无需额外数值稳定处理。

## 算子实现

### 接口定义

```cpp
aclblasStatus_t aclblasCaxpy(
    aclblasHandle_t handle,
    int n,
    const aclblasComplex* alpha,
    const aclblasComplex* x,
    int incx,
    aclblasComplex* y,
    int incy);
```

### 参数说明

| 参数名 | 输入／输出/属性 | 描述 | 数据类型 | dtype类型 | 数据排布格式 | 维度(shape) | 值域范围 | 异常行为 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| handle | 输入 | ops-blas 库上下文句柄，携带 stream，Host 内存 | scalar | - | - | - | 指向已创建的有效句柄 | handle 为 nullptr 时返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| n | 输入 | 向量 x、y 的元素个数，Host 内存 | scalar | int | - | - | n ≥ 0 | n < 0 时返回 `ACLBLAS_STATUS_INVALID_VALUE`；n = 0 为合法 no-op，直接返回成功 |
| alpha | 输入 | 指向复数标量乘数的指针，Host 内存 | scalar | COMPLEX64 | - | - | 实部/虚部取值于 FLOAT32 全集 | n > 0 时 alpha 为 nullptr 返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| x | 输入 | 复数向量，Device 内存，只读 | tensor | COMPLEX64 | ND | 逻辑一维 [n]，物理长度 1+(n-1)\|incx\| | 实部/虚部取值于 FLOAT32 全集 | n > 0 时 x 为 nullptr 返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| incx | 输入 | x 中连续元素之间的步长，Host 内存，可正可负 | scalar | int | - | - | incx ≠ 0 | incx = 0 时返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| y | 输出（原地输出） | 复数向量，Device 内存，原地更新为 alpha\*x+y | tensor | COMPLEX64 | ND | 逻辑一维 [n]，物理长度 1+(n-1)\|incy\| | 实部/虚部取值于 FLOAT32 全集 | n > 0 时 y 为 nullptr 返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| incy | 输入 | y 中连续元素之间的步长，Host 内存，可正可负 | scalar | int | - | - | incy ≠ 0 | incy = 0 时返回 `ACLBLAS_STATUS_INVALID_VALUE` |

**返回值**：`aclblasStatus_t`，状态码语义与 ops-blas 仓 `include/cann_ops_blas_common.h` 定义一致（`ACLBLAS_STATUS_SUCCESS` / `ACLBLAS_STATUS_INVALID_VALUE` / `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` 等）。

### 工程模式

采用 Ascend C kernel 直调方式：基于 ops-blas 仓工程框架，实现句柄式 BLAS 接口，通过 `handle` 绑定 stream（`aclblasSetStream`）直调 NPU kernel。实现代码位于 `blas/axpy/arch35/`，接口声明位于公共头文件 `include/cann_ops_blas.h`，供其他产品线共用。

#### 3.2.1 host 侧设计

**1. 参数校验**

host 侧先完成参数合法性校验，校验失败直接返回对应错误码，不发起 kernel 调用：

| 非法条件 | 返回值 |
| --- | --- |
| handle == nullptr | `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| n < 0 | `ACLBLAS_STATUS_INVALID_VALUE` |
| n > 0 且 alpha / x / y 为 nullptr | `ACLBLAS_STATUS_INVALID_VALUE` |
| incx == 0 或 incy == 0 | `ACLBLAS_STATUS_INVALID_VALUE` |
| n == 0 | 合法 no-op，直接返回 `ACLBLAS_STATUS_SUCCESS`，不修改 y |

**2. 数据规模计算**

根据 n、|incx|、|incy| 计算搬运规模：x/y 逻辑元素数为 n，物理元素数为 `1 + (n-1)*|inc|`；COMPLEX64 单元素 8 字节。据此确定 host 侧分配量与 kernel 侧搬运量，避免越界访问。

**3. 分核与 tiling 策略**

- **分核策略**：优先使用满核。根据 n 与单核可处理 tile 数计算核数 `coreNum`，核间数据块均分；不能均分时，将余出的数据块分配到前几个核上，保证各核负载均衡。
- **数据分块**：根据 UB 内存大小（`GetCoreMemSize`）与预定义 `BLOCK_SIZE`、`BUFFER_NUM`（double buffer 开关）计算单 tile 数据量 `tileDataNum`，将 n 切分为若干 tile；单核处理多 tile 时循环搬运计算，尾块按剩余数据量单独处理，保证不完整块并入计算流程、不产生数据碎片。
- **UB 空间权衡**：综合考虑不同硬件的 UB 大小、是否开启 double buffer、kernel 侧计算过程是否需要临时存储，确定单核内切分大小，充分使用 UB 空间。

**4. TilingData 传递**

将计算出的切分参数写入 `CaxpyTilingData` 结构体，随 kernel 启动参数传给 kernel 侧：

```cpp
struct CaxpyTilingData {
  int32_t n;           // 向量长度
  int32_t incx;        // x 步长（可正可负）
  int32_t incy;        // y 步长（可正可负）
  int32_t coreNum;     // 使用的核数
  int32_t tileDataNum; // 单 tile 元素数
  int32_t tileNum;     // 单核 tile 数
  int32_t tailDataNum; // 尾块元素数（< tileDataNum 时为非 0）
};
```

#### 3.2.2 kernel 侧设计

kernel 侧进行 Init 和 Process 两个阶段，其中 Process 包括数据搬入（CopyIn）、计算（Compute）、搬出（CopyOut）三个阶段。

**1. Init 阶段**

- 解析 TilingData（n、incx、incy、tile 参数），计算各核负责的数据区间；
- 绑定 x、y 全局内存地址，申请 UB 缓冲（含 double buffer 时申请多块）。

**2. CopyIn 阶段（数据搬入）**

- 按逻辑索引搬入：对第 i 个元素（i = 1..n），物理地址偏移为 `(i-1)*inc`，`|inc| = 1` 时按连续地址整块搬入；`|inc| > 1` 时按步长逐元素（或按实部/虚部连续段）搬入；
- 负步长（inc < 0）时按负偏移寻址取数，起始地址不变，仅偏移方向相反；
- 无广播参与，无需 shape 补全与广播填充逻辑。

**3. Compute 阶段（计算）**

逐元素执行复数乘加，实部/虚部分别计算：

```cpp
// 伪代码：单元素计算（alpha = a + bi, x = c + di, y = e + fi）
res.re = a * c - b * d + e;
res.im = a * d + b * c + f;
```

**4. CopyOut 阶段（结果搬出）**

将计算结果按原步长寻址写回 y 对应位置，完成原地更新。

**5. 性能与内存优化**

- 连续场景（incx = incy = 1）：整块 vector 化搬运与计算，避免逐元素寻址开销；
- double buffer：CopyIn 与 Compute 流水重叠，隐藏搬运延迟；
- 多核并行：各核独立处理分配的 tile 区间，无核间通信与同步开销；
- 大 n 场景按 tile 循环处理，UB 复用，避免一次性占用过大内存。

### 算子流程

```
host: 参数校验 → 数据规模计算 → 分核/分块(tiling) → 填充 TilingData
        → aclblasSetStream 绑定 stream → 启动 kernel
kernel: Init（解析 TilingData/绑定地址/申请 UB）
        → Process: CopyIn（按步长搬入 x/y）→ Compute（复数乘加）→ CopyOut（写回 y）
```

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas 950PR（Ascend 950PR） | √ |

## 算子约束限制

| 约束项 | 内容 |
| --- | --- |
| 参数合法性 | n ≥ 0；incx ≠ 0；incy ≠ 0；非法参数返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| broadcast 规则 | 不涉及，本算子为两个一维向量的逐元素线性组合 |
| 非连续 Tensor 支持 | 不要求（本批次不支持额外 leading dimension padding 场景；向量非连续由 incx/incy 表达） |
| dynamic shape | 不要求，n 为运行时入参 |
| 原地与视图语义 | y 原地更新，不返回视图 |
| 确定性计算 | 不要求 |
| 空 Tensor 与 0 维处理 | n = 0 为合法 no-op，返回成功且不修改 y |
| 异步执行 | 依赖 `aclblasSetStream` 绑定 stream；读回 Device 结果前须同步 stream |
| 数据范围 | 输入实部/虚部取值于 FLOAT32 全集，含 Inf/NaN 传递语义（对齐 cublas） |

# 可维可测分析

## 精度标准/性能标准

### 精度标准

golden 由 cblas（Netlib BLAS `caxpy`）单标杆比对生成，实部、虚部分别比对。按生态算子开源精度标准，本算子输入输出为单精度复数，比对时实部、虚部分别按 FLOAT32 标准判定：

| 数据类型 | rtol | atol | required_matched_ratio | max_abs_error_limit |
| --- | --- | --- | --- | --- |
| COMPLEX64（实部/虚部按 FLOAT32 分量） | 2^-10 (9.77e-4) | 2^-16 (1.53e-5) | 0.99 | 1e-2 或 32 \* ULP |

逐元素通过条件：`|actual - golden| ≤ atol + rtol × |golden|`；当用例同时满足 `matched_ratio ≥ required_matched_ratio` 且 `max_abs_error ≤ max_abs_error_limit` 时，判定该用例精度通过。

### 性能标准

- 测试设备：Ascend 950PR。性能数据为 COMPLEX64 输入场景下的平均单次耗时（Avg time，单位 us），须先 warmup 再有效采样 >50 次取平均。
- 验收判定：NPU 平均单次耗时 ≤ 任务书标杆耗时（下表），等价于与 GPU H100 基线（`gpu_baseline.csv`）的倍率 `ratio = gpu_ms / npu_ms ≥ 0.4`。

| case | n | incx | incy | 标杆耗时（Avg time，us） |
| --- | --- | --- | --- | --- |
| 1 | 1048576 | 1 | 1 | 14.53 |
| 2 | 2097152 | 1 | 1 | 28.36 |
| 3 | 4194304 | 1 | 1 | 89.76 |

> 口径说明：任务书 §3.3 标杆耗时 = GPU H100 原始耗时（`gpu_baseline.csv` 前 3 条：5.811 / 11.343 / 35.903 us）÷ 倍率阈值 0.4，即允许的 NPU 最大平均耗时上限。

## 测试方案

采用 ops-blas 仓 test 目录的测试框架完成自验：测试用例以 CSV 文件描述（`caxpy_test.csv`，1200 条），Python 脚本调用 C++ GTest 工程加载 CSV 调用 `aclblasCaxpy` 接口执行，精度 golden 由 cblas 生成。CSV 驱动测试工程参照 `test/copy/scopy`、`test/chemm` 模式新建于 `test/axpy/caxpy/arch35/`。

### 用例分类

| 类别 | 前缀 | 条数 | 说明 |
| --- | --- | --- | --- |
| L0 基础 | TC_L0 | 8 | 小尺寸 × 基础步长组合（±1/±2） |
| L1 尺寸 | TC_SQ | 38 | 38 种尺寸（1→1048576，含质数/边界/非对齐） |
| L2 步长 | TC_INC | 36 | incx × incy 全组合（±1/±2/±3） |
| L3 标量 | TC_AB | 18 | 9 组 alpha 特殊值（0/1/纯虚/负/大值/1e10）× 2 尺寸 |
| L5 填充 | TC_FL | 12 | x/y 分别覆盖：均匀随机/全零/交替/极端值/Inf/NaN |
| L5b 对齐 | TC_AL | 5 | x/y 对齐偏移组合（非对齐地址访问） |
| L6 边界 | TC_ED | 8 | n=0 no-op / 空指针 / incx=0 / incy=0 / 负维度（负向期望 INVALID_VALUE） |
| EX 扩展 | TC_EX | 875 | 尺寸 × 步长 × 标量 × 对齐的确定性采样（扩展精度条数的主力类别） |
| PF 性能 | TC_PF | 200 | 3 条任务书典型 case + 小尺寸 + 规模扫描 + 步长组合 + 非对齐 + 混合 |

> 用例由 `gen_csv.py` 以固定随机种子（默认 20260823）确定性生成，可复现；条数可扩展（`--accuracy` / `--perf`），固定类别（L0~L6）保持全集覆盖。性能用例一律连续访存（incx=incy=1、无对齐偏移），非连续场景由精度用例 TC_INC/TC_AL/TC_EX 覆盖。

### 验证工具

| 脚本 | 用途 | 关键命令 |
| --- | --- | --- |
| `gen_csv.py` | 生成/扩展测试用例与基线占位 | `python gen_csv.py --accuracy 1500 --perf 300 --seed 12345` |
| `verify_accuracy.py` | 编译并运行 GTest，解析逐条 PASS/FAIL | `python verify_accuracy.py --repo <ops-blas> --soc ascend950 --csv ./caxpy_test.csv` |
| `verify_performance.py` | 执行 TC_PF 采集 NPU 耗时，与基线比对 | `python verify_performance.py --repo <ops-blas> --soc ascend950 --timeout 3600` |

### 入参覆盖规则

| 参数名 | Tensor 值域分布 | Attr 覆盖规则 |
| --- | --- | --- |
| handle | - | 固定为已创建的有效句柄，不随机生成 |
| n | - | 覆盖 0、1、小质数（2/3/7）、2 的幂及 2 的幂 ±1，直至大规模（≥2^20） |
| alpha | 均匀分布 [-5, 5] 占 50%、正态分布（μ∈[-5,5]，σ∈[0.1,2]）占 50%，实部/虚部独立采样；另含特殊值 (0,0)、(1,0) | - |
| x | 均匀分布 [-5, 5] 占 50%、正态分布（μ∈[-5,5]，σ∈[0.1,2]）占 50%，实部/虚部独立采样；含 Inf/NaN 特殊值用例 | - |
| incx | - | 覆盖 1、2、-1、-2 及非常规步长；incx = 0 作为负向用例（期望返回 `ACLBLAS_STATUS_INVALID_VALUE`） |
| y | 同 x 的分布规则 | - |
| incy | 同 incx 的覆盖规则 | - |

> 说明：当前 ops-blas 测试工程仅支持均匀分布（RANDOM_NORM_5_5 = [-5,5]），生态标准要求的 50% 正态分布依赖测试工程扩展；当前用例按均匀分布生成，扩展完成后通过 `--dist mixed` 切换。

## 兼容性分析

本算子为 ops-blas 仓新增算子（axpy 族首次 arch35 实现），不涉及对既有算子行为的兼容性影响：

- 接口声明放入公共头文件 `include/cann_ops_blas.h`，供各产品线共用，不定义 950PR 私有平行接口；
- 参数序列与对标接口 `cublasCaxpy` 一一对应，无需额外映射说明；
- 精度 golden 由 cblas（Netlib BLAS）生成，无三方软件依赖；
- 边界、负向用例及特殊值（Inf/NaN）用例行为对齐 cublas。

## 自验与交付要求

1. 自验用例覆盖：小 shape 基础用例、shape 扫描、填充模式、对齐偏移、边界与负向用例（零维、空指针 x/y、非法步长、负维度等）、规格允许的 INF/NAN 场景，以及性能/内存用例；任务配套用例未覆盖的场景自行补充。
2. 自测报告：包含用例参数、精度对比结果及截图（实部/虚部分别）、性能数据及截图、内存占用数据。
3. 交付件：算子设计文档（本文件）、自测用例及测试代码（含 README 说明测试步骤）、自测报告、待验收代码地址（个人仓邀请 Ascend-CANN 为开发者）。
4. 测试通过后按 ops-blas 仓库规范提交 PR：实现合入 `blas/axpy/arch35/`，测试代码合入 `test/axpy/caxpy/arch35/`；README 产品支持表标注 Ascend 950PR：支持。
