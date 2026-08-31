# 需求背景（required）

## 需求来源

昇腾CANN社区 2026年8月社区任务：aclblasScnrm2算子开发（Ascend 950PR）。

## 背景介绍

### aclblasScnrm2算子概述

aclblasScnrm2 是 cuBLAS 标准 BLAS Level-1 接口 `cublasScnrm2` 的昇腾等价实现，用于计算单精度复数（complex64）向量的欧几里得范数（L2 范数）。

数学公式：`result = ||x||_2 = sqrt(Σ (|real(x[k])|² + |imag(x[k])|²))`，其中 k = 1+(i-1)*incx，i = 1..n。

输入为 complex64 复数向量（实部/虚部各 float32，交错存储），输出为 float32 实数标量。

### 对标接口

- cuBLAS: `cublasScnrm2`
- Netlib BLAS: `scnrm2`（参考：https://www.netlib.org/blas/scnrm2.f）

### 与仓内已有算子关系

ops-blas 仓已有 `aclblasSnrm2`（实数 float32 向量范数）和 `aclblasSnrm2Ex`（多精度扩展），本算子为复数版本，复用类似的多核并行架构。

# 需求分析（required）

## 需求描述

在昇腾 NPU（Ascend 950PR）上使用 Ascend C 编程语言实现 `aclblasScnrm2` 接口，通过 handle 绑定 stream 直调 NPU kernel，实现代码放在 `blas/nrm2/arch35/`。

## 需求拆解

1. 实现 complex64 输入向量的 L2 范数计算（输出 float32 标量）
2. 支持连续访存（incx=1）和非连续访存（incx≠1，含负步长）
3. 采用多阶段累加避免中间结果溢出/下溢
4. 正确处理 Inf/NaN 特殊值传播
5. 实现零维 quick return（n≤0 时 result=0.0f）
6. 参数合法性校验（空指针、incx=0 等负向用例）
7. 性能达标（全部性能用例 incx=1，充分利用多核并行带宽）

# 详细设计（required）

## 算子分析

### 数学公式

```
result = sqrt(Σᵢ (real(x[k])² + imag(x[k])²))
```
其中 k = 1+(i-1)*|incx|，i = 1..n（范数与遍历方向无关，负步长取绝对值）。

### 接口定义

```cpp
aclblasStatus_t aclblasScnrm2(
    aclblasHandle_t handle,
    int n,
    const aclblasComplex* x,
    int incx,
    float* result);
```

### 参数说明

| 参数 | 方向 | 数据类型 | 说明 |
|------|------|----------|------|
| handle | 输入 | aclblasHandle_t | ops-blas 库上下文句柄，携带 stream |
| n | 输入 | int | 向量复数元素个数 |
| x | 输入 | const aclblasComplex* | 复数向量，Device 内存 |
| incx | 输入 | int | 元素步长，可正可负，不为0 |
| result | 输出 | float* | L2 范数结果，Device 内存 |

### 支持数据类型

- 输入：COMPLEX64（aclblasComplex = {float real; float imag;}，交错存储）
- 输出：FLOAT32（实数标量）

### 返回值

| 状态码 | 条件 |
|--------|------|
| ACLBLAS_STATUS_SUCCESS | 正常完成或 n≤0 quick return |
| ACLBLAS_STATUS_HANDLE_IS_NULLPTR | handle 为空 |
| ACLBLAS_STATUS_INVALID_VALUE | incx=0 或 n>0时 x/result 为空 |

## 算子实现

### 实现方案

采用双路径架构，根据 incx 值选择最优计算路径：

#### Host 侧设计

**参数校验**：
- handle == nullptr → HANDLE_IS_NULLPTR
- incx == 0 → INVALID_VALUE
- result == nullptr → INVALID_VALUE
- n > 0 && x == nullptr → INVALID_VALUE

**Quick return**：n ≤ 0 时，通过 aclrtMemcpy 写 0.0f 到 device result，返回 SUCCESS。

**Tiling 策略**：
- useCoreNum = min(aivCoreCount, n)，上限 SCNRM2_MAX_CORE_NUM=64
- perCoreN = n / useCoreNum（每核基础处理量）
- remainder = n % useCoreNum（前 remainder 个核多处理 1 个元素）
- incx ≠ ±1 时计算 SIMT 线程数 nthreads

**Workspace**：
- SIMD 路径：paddedCoreNum × sizeof(float) 字节（每核 1 个部分和）
- SIMT 路径：paddedCoreNum × 2 × sizeof(float) 字节（每核 scale+ssq 对）
- 通过 aclrtMemsetAsync 零初始化

**Kernel 启动**：单次 kernel launch（使用 SyncAll 合并计算与归约）。

#### Kernel 侧设计

##### 路径一：SIMD（incx=1，性能关键路径）

**核心思想**：complex64 数据在内存中为交错的 [real₀, imag₀, real₁, imag₁, ...]，视为 2*n 个连续 float32。对其逐元素平方后求和，等价于 Σ(real² + imag²)。

**流程**：
1. 每核计算自己负责的 perCoreN 个复数元素（= 2*perCoreN 个 float）
2. 分 chunk 加载到 UB（maxDataCount=27392 float/chunk），每 chunk 做：
   - DataCopy: GM → UB（32B 对齐）
   - Mul(x, x, x): 逐元素平方
   - ReduceSum: 向量求和得单值
   - Add: 累加到 accBuf
3. 最终将部分和写入 workspace[blockIdx]
4. SyncAll 同步所有核
5. Block 0 执行最终归约：ReduceSum(workspace) → Sqrt → 写入 result

**UB 内存布局**：
- inQueue: BUFFER_NUM=2 × 27392×4B = 214KB（双缓冲流水）
- outQueue: 2 × 32B = 64B
- workBuf: (27392/64 对齐)×4B + 32B ≈ 1.7KB
- accBuf: 32B
- 总计 < 248KB - 32KB(safety) = 216KB ✓

##### 路径二：SIMT（incx≠1，正确性关键路径）

**核心思想**：非连续访存无法利用 SIMD 向量化，采用多线程标量计算 + LAPACK 式 scaled-ssq 累加防止溢出。

**流程**：
1. 每个 SIMT 线程维护 (tScale, tSsq) 对，遍历自己负责的元素：
   - 对 |real| 和 |imag| 分别更新 (scale, ssq)
   - 遇到 NaN 标记 flag=NaN，遇到 Inf 标记 flag=Inf
2. 线程间树形归约合并 (scale, ssq) 对
3. Thread 0 将 (scale, ssq) 或 flag 写入 workspace[blockIdx*2]
4. SyncAll 同步
5. Block 0 执行最终归约：合并所有 (scale, ssq) 对，result = globalScale × sqrt(globalSsq)

**Scaled-ssq 更新规则**（防溢出）：
```
if ax > scale:
    ssq = 1 + ssq × (scale/ax)²
    scale = ax
elif ax ≠ 0:
    ssq += (ax/scale)²
```

### 关键设计决策

1. **单 Kernel 架构**：使用 SyncAll 合并计算与归约，消除第二次 kernel launch 的 15us 调度开销
2. **Complex→Float 转换**：incx=1 时将 complex64 视为 2n float 处理，最大化 SIMD 吞吐
3. **双路径分流**：SIMD 路径优化带宽利用率，SIMT 路径保证非连续访存正确性
4. **Overflow-safe**：SIMT 路径采用 scaled-ssq 算法，正确处理 FLT_MAX 级别输入
5. **特殊值传播**：NaN 优先于 Inf，对齐 cblas_scnrm2 语义

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
|---------------|----------|
| Ascend 950PR | √ |

## 算子约束限制

| 约束项 | 内容 |
|--------|------|
| 参数合法性 | n≥0（n≤0 走 quick return）；incx≠0 |
| 非连续 Tensor | 由 incx 表达步长，含负步长（取绝对值） |
| broadcast | 不涉及，单向量归约到标量 |
| dynamic shape | 不要求，n 为运行时入参 |
| 确定性计算 | 不要求（归约顺序不保证逐位一致） |
| 异步执行 | 依赖 aclblasSetStream 绑定 stream |

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
|----------|------|----------|
| 精度标准 | atol=2⁻¹⁶, rtol=2⁻¹⁰, max_abs_error≤1e-2 | 生态算子开源精度标准（FLOAT32档） |
| 性能标准 | n=1M/4M/16M incx=1 ≤ 5.23/5.75/6.73 us | 任务书 §3.3 |

## 兼容性分析

接口声明已存在于 `include/cann_ops_blas.h`，与其他产品线（arch22 等）共用同一 API，无兼容性问题。
