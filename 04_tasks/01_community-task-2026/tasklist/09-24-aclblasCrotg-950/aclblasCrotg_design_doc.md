# 需求背景（required）

## 需求来源

昇腾社区任务 2026 年 8 月社区任务 - aclblasCrotg 算子开发（950PR）。

## 背景介绍

### aclblasCrotg 算子实现

基于 ops-blas 开源仓既有实数算子 aclblasSrotg 的 arch35 实现模式，使用 Ascend C 编程语言开发复数 Givens 旋转参数构造算子 aclblasCrotg，适配 Ascend 950PR NPU。

aclblasCrotg 算子对标 cuBLAS cublasCrotg 和 Netlib BLAS crotg 参考实现，为昇腾 NPU 提供复数 Givens 旋转参数构造能力。Givens 旋转是 QR 分解、最小二乘求解、特征值计算等数值线性代数算法的基本构建块。

### aclblasCrotg 算子功能分析

| 参数 | 参数含义 | 数据类型 | 支持数据类型 | 约束 | 形状 |
| --- | --- | --- | --- | --- | --- |
| handle | 库上下文句柄，携带 stream | aclblasHandle_t | - | 必须为有效句柄 | - |
| a | 输入/输出复数标量，运算后覆写为 r | aclblasComplex* | complex64 | 不可为 nullptr | [1] |
| b | 输入复数标量，只读不覆写 | aclblasComplex* | complex64 | 不可为 nullptr | [1] |
| c | 输出旋转余弦（实数 float） | float* | float32 | 不可为 nullptr | [1] |
| s | 输出旋转正弦（复数） | aclblasComplex* | complex64 | 不可为 nullptr | [1] |

计算公式：
c = |a| / sqrt(|a|^2 + |b|^2)
s = sgn(a) * conjg(b) / sqrt(|a|^2 + |b|^2)
r = sgn(a) * sqrt(|a|^2 + |b|^2)

其中 sgn(a) = a/|a|（a = 0 时取 1），|x| = sqrt(Re(x)^2 + Im(x)^2)。

### aclblasCrotg 算子功能分析

输入：复数标量 a、b
输出：实数标量 c、复数标量 s、a 被覆写为 r
支持数据类型：complex64（单精度复数）
特殊分支：b=0 时 c=1,s=0,r=a；a=0 且 b≠0 时 c=0,s=conjg(b)/|b|,r=|b|；a=b=0 时 c=1,s=0,r=0

# 需求分析（required）

## 需求描述

使用 Ascend C 编程语言实现 aclblasCrotg 算子，支持 complex64 数据类型，实现 host/device 双路径（全部 host 指针直接 CPU 计算，全部 device 指针启动 SIMT kernel），混合指针返回 INVALID_VALUE。性能不低于 cuBLAS cublasCrotg 对标版本。

## 需求拆解

1. 支持 complex64 数据类型（实部/虚部各 float32）
2. 实现 host/device 双路径判别与分发
3. 数值稳定性：使用 ::hypotf 安全缩放，避免极端数量级输入下的溢出/下溢
4. 特殊分支语义与 Netlib crotg 参考实现一致
5. 性能不低于 cuBLAS cublasCrotg 对标版本

# 详细设计（required）

## 算子分析

### 数学公式

c = |a| / sqrt(|a|^2 + |b|^2)
s = sgn(a) * conjg(b) / sqrt(|a|^2 + |b|^2)
r = sgn(a) * sqrt(|a|^2 + |b|^2)

满足正交归一性：c^2 + |s|^2 = 1

### 支持数据类型

complex64（单精度复数，实部/虚部各 float32）

### 支持形状

纯标量算子，无形状/维度概念。4 个参数均为单元素指针。

## 算子实现

### 实现方案

#### host 侧设计：

本算子为纯标量算子，不涉及 tiling 策略。host 侧主要职责为参数校验和路径分发：

1. 参数合法性校验：handle/a/b/c/s 任一为 nullptr 返回对应错误码
2. 指针位置判断：通过 aclrtPointerGetAttributes 判断 a 指针位于 host 还是 device
3. 一致性检查：b/c/s 必须与 a 同侧（全部 host 或全部 device），混合返回 INVALID_VALUE
4. 路径分发：
   - Host 路径：调用 CrotgCpuCompute() 直接 CPU 计算
   - Device 路径：调用 crotg_kernel_do() 启动 SIMT kernel，绑定 handle 中的 stream

##### 分核策略：

本算子为纯标量算子，不涉及多核分派。Device 路径使用单线程 SIMT 核（asc_vf_call 配 dim3{SIMT_MIN_THREAD_NUM}），因算子仅处理 4 个标量参数。

##### 数据分块和内存优化策略：

不涉及。算子为纯标量，无数据搬运需求。Device 路径的 H2D/D2H 由测试框架的 crotg_npu_wrapper.h 封装，算子本身仅操作已位于 device 的指针。

#### kernel 侧设计：

采用 Ascend C SIMT 编程模型，single-thread dispatch：

1. 从 GM 内存读取 a/b 的实部/虚部（4 个 float）
2. 计算 absA = ::hypotf(ar, ai)，absB = ::hypotf(br, bi)
3. 按 Netlib Algorithm 978 安全缩放逻辑计算 norm/c/s/r
4. 将 r 写回 a 的 GM 位置，c/s 写入对应输出位置

特殊分支处理：
- b = 0：c = 1, s = (0,0), r = a
- a = 0 且 b ≠ 0：c = 0, s = conjg(b)/|b|, r = |b|（实数结果）
- a = b = 0：c = 1, s = (0,0), r = (0,0)

数值稳定性：全部使用 ::hypotf() 计算复数模，避免直接平方相加导致的溢出/下溢。host/golden/kernel 三路统一使用 ::hypotf 确保浮点语义一致。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Ascend 950PR | √ |

## 算子约束限制

1. 本算子为纯标量算子，无向量长度、步长、维度参数，不支持 batch 模式（需多次调用）
2. a/b/c/s 必须全部位于 host 或全部位于 device，混合指针返回 INVALID_VALUE
3. Inf/NaN 输入按 IEEE 754 语义传播，不保证结果的数学正确性（与 cuBLAS 行为一致）
4. 次正规数（denormal）处理依赖 NPU hardware flush-to-zero 行为，可能与 host 侧有微小差异但在 MERE 容差范围内
5. b 为只读输入，不被覆写（与 cuBLAS/Netlib 语义一致，实数 rotg 的 z 恢复规则不适用于复数版本）

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | 逐标量比对 matched_ratio ≥ 0.99，max_abs_error ≤ 1e-2 或 32 ULP；数学性质验证（正交归一/旋转零化/模长保持）全部通过 | 生态算子开源精度标准 |
| 性能标准 | NPU 平均单次耗时 ≤ 任务书标杆耗时（单次一般值 11.85us、单次 a=0 9.97us、单次大数 9.49us、批量 1000 次 11.84us、批量 10000 次 11.53us） | 任务书 §3.3 |

## 兼容性分析

新算子，不涉及兼容性分析。接口签名对标 cuBLAS cublasCrotg 逐参数一致，可与其他产品线共用。
