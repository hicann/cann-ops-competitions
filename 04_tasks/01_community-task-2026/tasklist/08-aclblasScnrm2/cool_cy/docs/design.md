# aclblasScnrm2 算子设计文档

# 需求背景（required）

## 需求来源

本任务来自 CANN 2026 年 8 月社区任务，要求在 Ascend 950PR 上使用 Ascend C 编程语言开发单精度复数向量欧几里得范数算子 `aclblasScnrm2`，并提交到 `ops-blas` 开源仓。

接口语义需与 cuBLAS `cublasScnrm2` 保持一致，验收以 `aclblasScnrm2` 的 BLAS 层行为为准。

## 背景介绍

`Scnrm2` 是 BLAS Level-1 归约算子，用于计算复数向量的 L2 范数。其数学定义为：

```text
result = ||x||_2 = sqrt(Σ (real(x[k])^2 + imag(x[k])^2))
```

其中 `k = 1 + (i - 1) * incx`，`i = 1..n`，兼容 Fortran 风格 1-based 步长语义。

## 接口现状分析

原型与任务书保持一致：

```cpp
aclblasStatus_t aclblasScnrm2(
    aclblasHandle_t handle,
    int n,
    const aclblasComplex* x,
    int incx,
    float* result);
```

| 参数 | 参数含义 | 数据类型 | 支持数据类型 | 约束 | 形状 |
| --- | --- | --- | --- | --- | --- |
| handle | 计算上下文句柄 | handle | 有效句柄 | 非空 | - |
| n | 复数元素个数 | int | int32 | `n <= 0` 合法 quick return | 标量 |
| x | 输入向量 | tensor | complex64 | `n > 0` 时非空 | 逻辑一维 `[n]` |
| incx | 元素步长 | int | int32 | 非 0，可正可负 | 标量 |
| result | 输出范数 | tensor | float32 | 非空 | 标量 |

### 算子功能分析

1. `n <= 0` 时直接返回 `0.0f`，不触发 kernel。
2. `n > 0` 时按 `incx` 遍历复数向量。
3. 对每个复数元素计算模平方 `real^2 + imag^2`。
4. 采用多阶段累加或等效 scaled sum-of-squares 方法，避免中间溢出和下溢。
5. 最终返回单个 `float32` 标量结果。

# 需求分析（required）

## 需求描述

实现 `aclblasScnrm2`，与 cuBLAS `cublasScnrm2` 语义一致，满足 Ascend 950PR 平台上的功能、精度和性能验收要求。

## 需求拆解

1. 支持 `complex64` 输入和 `float32` 输出。
2. 支持 `incx = 1, 2, 3, -1, -2, -3 ...` 等正负步长。
3. 支持 `n = 0`、负数 `n` quick return。
4. `incx = 0`、空指针等非法参数返回错误码。
5. 采用多阶段累加，保证范数计算稳定性。
6. 接口声明沿用 `include/cann_ops_blas.h`，不引入 950PR 私有平行 API。
7. 满足任务书性能标杆。

# 详细设计（required）

## 算子分析

### 数学公式

设输入复数向量为 `x`，则：

```text
result = sqrt(Σ_{i=1..n} (real(x[k])^2 + imag(x[k])^2))
```

其中：

```text
k = 1 + (i - 1) * incx
```

对于 `incx < 0` 的情况，逻辑遍历顺序与 cuBLAS 保持一致，由调用侧规范化起始地址或 kernel 内部做等价偏移处理。

### 支持数据类型

| 数据 | 类型 |
| --- | --- |
| x | complex64 |
| result | float32 |

### 支持形状

```text
x:      [n]
result: scalar
```

其中物理可访问长度为 `1 + (n - 1) * abs(incx)`。

## 算子实现

### 实现方案

实现采用 `ops-blas` 现有 BLAS 直调框架，代码落点为 `blas/nrm2/arch35/`。整体分为 host 侧参数处理、kernel 侧分块归约、最终标量回写三个阶段。

#### 3.2.1 host 侧设计

1. 参数校验：
   - `handle` 非空。
   - `result` 非空。
   - `incx != 0`。
   - `n > 0` 时 `x` 非空。
2. quick return：
   - `n <= 0` 时直接写 `result = 0.0f` 并返回成功。
3. 起始位置规范化：
   - `incx > 0` 时直接使用输入首地址。
   - `incx < 0` 时按逻辑遍历顺序调整起始偏移，保证与 cuBLAS 语义一致。
4. tiling 规划：
   - 按元素规模和 UB 容量划分 tile。
   - 以连续访问为优先，减少 GM 访存次数。
5. kernel 启动：
   - 从 handle 绑定的 stream 发起 kernel 调度，将 `n`、`abs_incx`、起始偏移、tile 配置传入 kernel。
6. 结果回写：
   - 保持 `result` 为 Device 标量输出，供调用侧按 stream 同步后读取。

#### 3.2.2 kernel 侧设计

kernel 采用分块扫描 + 多阶段规约方式实现 `Scnrm2`：

1. 每个 core 读取一个或多个 tile。
2. 对 tile 内复数元素计算 `real^2 + imag^2`。
3. 使用 scaled sum-of-squares 维护局部 `scale` 和 `ssq`，避免直接累加平方导致的数值风险。
4. core 内得到局部 `(scale, ssq)` 后写入中间缓冲。
5. 第二阶段对各 core 的局部结果进行 merge，仍使用同一套 scaled 规约规则。
6. 最终由一个 core 完成 `result = scale * sqrt(ssq)` 并写回输出。

该方案满足“多阶段累加或等效手段”的任务要求，同时在复数场景下保持稳定性和可复现性。

### 3.2.3 数值稳定性设计

采用 BLAS 常用的 scaled sum-of-squares 规约：

```text
if abs(x) != 0:
    if scale < abs(x):
        ssq = 1 + ssq * (scale / abs(x))^2
        scale = abs(x)
    else:
        ssq += (abs(x) / scale)^2
```

其中 `abs(x)` 对复数由 `sqrt(real^2 + imag^2)` 得到。该策略比直接累加模平方更稳健，适合大规模向量和极值输入。

### 3.2.4 工程文件设计

| 模块 | 说明 |
| --- | --- |
| `include/cann_ops_blas.h` | 保留/复用 `aclblasScnrm2` 声明 |
| `blas/nrm2/arch35/` | 950PR 实现目录 |
| `blas/nrm2/arch35/scnrm2_host.cpp` / `scnrm2_kernel.cpp` | host 参数检查、quick return、tiling、kernel 实现 |
| `test/nrm2/scnrm2/arch35/` | CSV 和 GTest 测试 |

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Ascend 950PR | √ |

## 算子约束限制

| 约束项 | 说明 |
| --- | --- |
| 数据类型 | 输入仅支持 complex64，输出为 float32 |
| 步长 | `incx != 0`，支持正负步长 |
| 零维 | `n <= 0` 为合法 quick return |
| 空指针 | `n > 0` 时 `x` 不能为空，`result` 不能为空 |
| 反向 | 不支持反向传播 |
| 动态形状 | 运行时由 `n` 和 `incx` 决定 |

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述(不涉及说明原因) | 标准来源 |
| --- | --- | --- |
| 精度标准 | 与 cblas / Netlib `scnrm2` 结果对齐，满足 float32 标量误差阈值 | 任务书 |
| 性能标准 | 各性能 case 平均耗时不高于任务书标杆 | 任务书 |

精度验证采用标量比对方式，满足：

```text
|actual - golden| <= atol + rtol * |golden|
```

任务书建议阈值按 float32 标量标准执行。

## 测试设计

### 精度测试

| 编号 | 场景 | 说明 |
| --- | --- | --- |
| TC-01 | 小尺寸基础用例 | 覆盖最小规模、单元素和短向量 |
| TC-02 | 随机输入 | 覆盖均匀分布和正态分布 |
| TC-03 | 正负步长 | 覆盖 `incx > 0` 和 `incx < 0` |
| TC-04 | 零维 quick return | 覆盖 `n = 0` 和负数 `n` |
| TC-05 | 非法参数 | 覆盖 `incx = 0`、空指针 |
| TC-06 | 极值输入 | 覆盖 Inf / NaN / 全零 / 大数 |

### 性能测试

| case | n | incx | 标杆耗时（Avg time, us） |
| --- | --- | --- | --- |
| 1 | 1048576 | 1 | 5.23 |
| 2 | 4194304 | 1 | 5.75 |
| 3 | 16777216 | 1 | 6.73 |

测试方式为 warmup 后有效采样 50 次以上取平均值。

### 自测要求

1. 精度 golden 由 cblas / Netlib `scnrm2` 生成。
2. CSV 驱动测试覆盖正向和负向用例。
3. 自测报告需要包含用例参数、精度结果和性能结果。

## 兼容性分析

该算子为新增接口，不影响现有 `ops-blas` 算子行为。输入不满足 dtype、shape、步长或空指针要求时，由 host 侧直接返回错误码，避免错误结果下发到 kernel。

# 参考资料

1. 任务书：`aclblasScnrm2_Atlas950PR_task_doc.md`
2. Ascend C 算子开发文档
3. ops-blas 开源仓
4. cuBLAS `cublasScnrm2` 文档
5. Netlib BLAS `scnrm2.f`
