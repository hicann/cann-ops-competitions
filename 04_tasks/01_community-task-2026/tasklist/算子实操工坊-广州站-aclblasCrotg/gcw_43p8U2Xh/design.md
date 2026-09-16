# 需求背景（required）

## 1.1 需求来源
华为昇腾 CANN 社区任务（广州站）——算子实操工坊。本任务要求在 Atlas A2/A3 系列产品上，使用 Ascend C 编程语言原生开发单精度复数 Givens 旋转参数构造算子 `aclblasCrotg`，并最终合入昇腾算子开源仓（ops-blas）。

## 1.2 背景介绍

### 1.2.1 aclblasCrotg 算子实现优化
基于 BLAS 库历史 CUDA/cuBLAS 版本（`cublasCrotg`）及 Netlib `crotg` 参考实现，使用 Ascend C 编程语言在昇腾 NPU 上进行算子原生开发与性能优化。该算子在 QR 分解、特征值求解等线性代数计算中具有重要作用，属于 BLAS Level 1 基础标量算子。由于复数算子涉及相位计算和共轭处理，且面临极小值下溢和极大值溢出的风险，其数值稳定性要求极高。

### 1.2.2 aclblasCrotg 算子参考实现现状分析
当前 `ops-blas` 开源仓中仅有同族实数算子 `aclblasSrotg`（位于 `blas/rotg/` 目录），尚无复数版本实现。实数算子仅涉及简单的开方和除法，而复数算子需要处理复数模长、共轭及相位旋转，且必须引入安全缩放机制。当前算子实现需严格参照 `srotg` 的工程框架，构建句柄式 BLAS 接口并实现 Host/Device 双路径直接调用。

### 1.2.3 aclblasCrotg 算子功能分析
**输入**：复数标量 $a$、$b$，数据类型为 `aclblasComplex`（实部/虚部均为 float32）。
**输出**：$a$ 原地覆写为 $r$，$c$（实数 float32），$s$（复数 complex64），$b$ 为只读输入不覆写。
**支持数据类型**：complex64。
**对标基线接口**：cuBLAS `cublasCrotg`（语义参考 Netlib `crotg` 实现）。

# 需求分析（required）

## 2.1 需求描述
使用 Ascend C 编程语言实现单精度复数 Givens 旋转参数构造算子 `aclblasCrotg`。算子需提供句柄式 BLAS 接口，支持 Host/Device 双路径直接调用，实现 complex64 数据类型的标量计算，严格满足精度和性能要求，并保证极端数量级输入下的数值稳定性，防止溢出和下溢。

## 2.2 需求拆解
1. **功能实现**：正确构造 Givens 旋转矩阵，完成 $r$、$c$、$s$ 计算，原地覆写 $a$ 为 $r$。支持特殊分支（$b=0$、$a=0$、$a=b=0$、$-0.0$）。
2. **数值稳定性**：采用全局缩放算法，避免直接计算 $|a|^2 + |b|^2$ 造成的下溢/溢出。
3. **接口与约束**：新增头文件声明，校验空指针（$a$、$b$、$c$、$s$）及 Host/Device 内存一致性。
4. **性能达标**：Host 路径单次调用延迟满足任务书 §3.3 标杆要求。

# 详细设计（required）

## 3.1 算子分析

### 3.1.1 数学公式
Givens 旋转矩阵作用于二维向量 $(a, b)^T$，使其第二个分量为零：

$$
\begin{bmatrix}
c & s \\
-\text{conjg}(s) & c
\end{bmatrix}
\begin{bmatrix}
a \\
b
\end{bmatrix}
=
\begin{bmatrix}
r \\
0
\end{bmatrix}
$$

其中，$c$ 为实数，$s$ 为复数，$r$ 为复数结果。数学定义为：

$$
\begin{aligned}
c &= \frac{|a|}{\sqrt{|a|^2 + |b|^2}} \\
s &= \text{sgn}(a) \cdot \frac{\text{conjg}(b)}{\sqrt{|a|^2 + |b|^2}} \\
r &= \text{sgn}(a) \cdot \sqrt{|a|^2 + |b|^2}
\end{aligned}
$$

其中：
- $|x|$ 表示复数模长，即 $\sqrt{\text{Re}(x)^2 + \text{Im}(x)^2}$
- $\text{conjg}(x)$ 表示复数共轭，即 $\text{Re}(x) - \text{Im}(x) \cdot i$
- $\text{sgn}(a) = a / |a|$，当 $a = 0$ 时，取 $\text{sgn}(0) = 1$
- 正交归一性约束：$c^2 + |s|^2 = 1$

**特殊值分支语义（与 Netlib 参考实现对齐）**：
- $b = 0$ 时：$c = 1$，$s = (0,0)$，$r = a$
- $a = 0$ 且 $b \neq 0$ 时：$c = 0$，$s = \text{conjg}(b)/|b|$，$r = |b|$（实数结果）
- $a = b = 0$ 时：$c = 1$，$s = (0,0)$，$r = (0,0)$

### 3.1.2 支持数据类型
- 输入/输出：`aclblasComplex`（complex64，实部/虚部均为单精度 float32）
- 标量参数：`float`（float32）

### 3.1.3 支持形状
纯标量算子，每个参数均为单元素指针，形状为 $[1]$。无向量长度、无步长、无维度轴。

## 3.2 算子实现

### 3.2.1 Host 侧设计

**接口设计**：在 `include/cann_ops_blas.h` 中新增如下声明：

```cpp
aclblasStatus_t aclblasCrotg(
    aclblasHandle_t handle,
    aclblasComplex* a,
    aclblasComplex* b,
    float* c,
    aclblasComplex* s);
```

**参数校验策略**：
1. 校验 `handle` 是否为空，为空返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`。
2. 校验 $a$、$b$、$c$、$s$ 指针是否为空，为空返回 `ACLBLAS_STATUS_INVALID_VALUE`。
3. **内存一致性检测**：通过 `aclrtPointerGetAttributes` 判断四个指针的内存属性。
   - 若均为 Host 侧：直接进行 CPU 计算，将结果写回 Host 内存。
   - 若均为 Device 侧：绑定 stream，下发 Device 侧 Kernel 进行计算。
   - 若混合 Host/Device 指针：返回 `ACLBLAS_STATUS_INVALID_VALUE`。

**Kernel 下发与同步**：Device 路径采用 kernel 直调方式。由于 kernel 下行为异步操作，Host 侧函数返回时 kernel 可能尚未执行完成。为保证 Host 侧后续读取 Device 内存时结果可见，**kernel 下发后必须调用 `aclrtSynchronizeStream(handle->stream)` 同步 stream**。这是本算子的关键工程细节，缺少此同步将导致 Host 侧读回的数据仍为原始输入值。

**无 Tiling 策略**：由于本算子为纯标量算子，仅有一个数据点，不需要分核、Tiling 切分及批量搬运策略。Host 侧固定以单核（Block Dim = 1）下发 Kernel，以最小化调度开销。

### 3.2.2 Kernel 侧设计

**整体流程**：采用 Kernel 直调方式，包含数据搬入（CopyIn）、计算（Compute）、搬出（CopyOut）三个阶段。

1. **CopyIn 阶段**：从 Global Memory 中读取 $a$、$b$ 的实部和虚部标量数据（读取 4 个 float 数值），存入寄存器。
2. **Compute 阶段**：
   - **全局缩放**：计算四个分量（$a_r, a_i, b_r, b_i$）绝对值的最大值 $ma = \max(|a_r|, |a_i|, |b_r|, |b_i|)$，所有分量除以 $ma$ 归一化到 $O(1)$ 量级。
   - **模长计算**：在归一化后的分量上计算 $|a|$、$|b|$，避免直接平方导致下溢。
   - **特殊分支判断**：优先判断 $b=0$、$a=0$、$a=b=0$。
   - **复数计算**：将复数乘除分解为实部虚部运算，计算 $c$、$s$、$r$；最终结果 $r$ 乘以缩放因子 $ma$ 还原量级。
3. **CopyOut 阶段**：将计算得到的 $r_{real}$、$r_{imag}$ 原地写回 $a$ 的 GM 地址；将 $c$、$s_{real}$、$s_{imag}$ 写回对应指针的 GM 地址。由于是单次标量计算，不使用队列（TQue），直接使用标量读写接口操作 Global Memory。

**核心算法设计**：

```cpp
// 读取输入
float a_r = aGM.GetValue(0), a_i = aGM.GetValue(1);
float b_r = bGM.GetValue(0), b_i = bGM.GetValue(1);
float c, s_r, s_i, r_r, r_i;

// 全局缩放：4 个分量绝对值的最大值
float ma = max(max(abs(a_r), abs(a_i)), max(abs(b_r), abs(b_i)));

if (ma == 0.0f) {
    // 全零输入
    c = 1.0f; s_r = 0.0f; s_i = 0.0f; r_r = 0.0f; r_i = 0.0f;
} else {
    // 归一化到 O(1)
    float a_sr = a_r / ma, a_si = a_i / ma;
    float b_sr = b_r / ma, b_si = b_i / ma;
    float abs_a = sqrt(a_sr * a_sr + a_si * a_si);
    float abs_b = sqrt(b_sr * b_sr + b_si * b_si);

    if (abs_b == 0.0f) {
        c = 1.0f; s_r = 0.0f; s_i = 0.0f; r_r = a_r; r_i = a_i;
    } else if (abs_a == 0.0f) {
        c = 0.0f;
        s_r = b_sr / abs_b; s_i = -b_si / abs_b;
        r_r = abs_b * ma; r_i = 0.0f;
    } else {
        float norm_s = sqrt(abs_a * abs_a + abs_b * abs_b);
        c = abs_a / norm_s;
        float sgn_ar = a_sr / abs_a, sgn_ai = a_si / abs_a;
        s_r = (sgn_ar * b_sr + sgn_ai * b_si) / norm_s;
        s_i = (sgn_ai * b_sr - sgn_ar * b_si) / norm_s;
        r_r = sgn_ar * norm_s * ma;
        r_i = sgn_ai * norm_s * ma;
    }
}
```

**特殊值传播与边界处理**：
- 遇到 Inf/NaN 输入时，仅需保证算子不崩溃且返回 `ACLBLAS_STATUS_SUCCESS`。原因是 IEEE 754 标准对 NaN 的传播路径定义为"实现相关"（implementation-defined），Host 侧 CPU 与 NPU kernel 的不同编译器实现对 NaN 的比较、max、sqrt 等操作的返回结果可能不同，因此不作为精度判定依据。

## 3.3 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas 800I/T A2 (910B3) | √ |
| Atlas 800I A3 | √ |

## 3.4 算子约束限制
1. 四个指针 $a$、$b$、$c$、$s$ 均不可为空。
2. $a$、$b$、$c$、$s$ 必须全部位于 Host 侧或全部位于 Device 侧，不支持混合。
3. 作为纯标量算子，不涉及非连续 Tensor、Broadcast、动态 Shape 和空 Tensor 的概念。
4. 异步执行依赖 `aclblasSetStream` 绑定 stream；Device 路径下发 kernel 后须调用 `aclrtSynchronizeStream` 保证 Host 侧可见。

# 可维可测分析

## 4.1 精度标准/性能标准

### 4.1.1 精度验收标准
本算子计算精度需满足生态算子开源精度标准。在单精度复数与单精度实数标量输入下，比对阈值如下：

| 数据类型 | rtol | atol | required_matched_ratio | max_abs_error_limit |
| --- | --- | --- | --- | --- |
| COMPLEX64 / FLOAT32 | $2^{-10}$ (9.77e-4) | $2^{-16}$ (1.53e-5) | 0.99 | 1e-2 或 32 * ULP |

逐元素通过条件：$|actual - golden| \le atol + rtol \times |golden|$。当用例同时满足 matched_ratio $\ge$ required_matched_ratio 且 max_abs_error $\le$ max_abs_error_limit 时，判定该用例精度通过。Golden 由 Netlib `crotg` 参考实现生成。

**数学性质验证口径（与逐标量比对并行执行）**：
每条精度用例须同时验证以下数学性质，其中 $atol = 2^{-16} \approx 1.53 \times 10^{-5}$，$rtol = 2^{-10} \approx 9.77 \times 10^{-4}$：
1. **正交归一性**：$|c^2 + |s|^2 - 1| \le atol + rtol$
2. **旋转零化**：
   - $|c \cdot a + s \cdot b - r| \le atol + rtol \cdot \sqrt{|a|^2+|b|^2}$
   - $|-\text{conjg}(s) \cdot a + c \cdot b| \le atol + rtol \cdot \sqrt{|a|^2+|b|^2}$
3. **模长保持**：$||r| - \sqrt{|a|^2+|b|^2}| \le atol + rtol \cdot \sqrt{|a|^2+|b|^2}$

### 4.1.2 性能验收标准
测试设备：Atlas 800I A3 (910B3)。性能数据为 COMPLEX64 输入场景下的平均单次耗时（Avg time，单位 us），须先 warmup 再有效采样 >50 次取平均。

| case | 场景 | $a$ | $b$ | 连续调用次数 | 标杆耗时（us） |
| --- | --- | --- | --- | --- | --- |
| 1 | 单次调用延迟（一般值） | (3.0, 4.0) | (1.0, 2.0) | 1 | 4.65 |
| 2 | 单次调用延迟（特殊分支 a=0） | (0.0, 0.0) | (1.0, 2.0) | 1 | 4.95 |
| 3 | 单次调用延迟（大数缩放路径） | (1e30, 1e30) | (1e30, -1e30) | 1 | 4.66 |
| 4 | 批量连续调用平均延迟 | (1.5, -2.5) | (0.5, 0.25) | 1000 | 4.70 |
| 5 | 批量连续调用平均延迟 | (1.5, -2.5) | (0.5, 0.25) | 10000 | 4.20 |

## 4.2 自验用例覆盖
基于 `crotg_test.csv` 的用例集（1200 条），自验覆盖以下类别：
- **L0 基础组合**（12 条）：一般复数、纯实、纯虚、等模、符号组合。
- **SP 特殊分支**（18 条）：$b=0$（$a$ 八相位）、$a=0$（$b$ 八相位）、$a=b=0$（含 -0.0）。
- **MAG 数量级**（16 条）：大数溢出保护（1e30/3e38）、小数下溢保护（1e-30/1e-38/denormal 1.4e-45）、数量级差异（1e±20、1e±40 倍差）。
- **PH 相位组合**（16 条）：单位圆 4×4 相位网格。
- **FL 特殊浮点**（8 条）：Inf/NaN 传播。
- **ED 边界负向**（5 条）：$a/b/c/s$ 空指针负向（INVALID_VALUE）以及全零合法组合。
- **EX 扩展**（925 条）：50% 均匀 [-5,5] + 50% 对数数量级（1e-30~1e30）× 随机相位。
- **PF 性能**（200 条）：5 条任务书典型 case + 取值路径 × iters 扫描 + 批量规模扫描。

## 4.3 兼容性分析
本算子为新增算子，不涉及存量代码的兼容性改造。接口签名严格对齐 `cublasCrotg`，并在 `include/cann_ops_blas.h` 中声明供其他产品线共用。测试工程参照 `srotg` 模式新建，golden 由 Netlib `crotg` 参考实现生成。在 `blas/rotg/README.md` 产品支持表中新增 `aclblasCrotg` 并标注 Atlas A2/A3 系列产品支持。