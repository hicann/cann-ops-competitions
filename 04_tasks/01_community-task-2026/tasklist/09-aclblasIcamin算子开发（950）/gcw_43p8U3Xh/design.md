# aclblasIcamin 算子设计文档

## 1 需求背景（required）

### 1.1 需求来源
华为昇腾 CANN 社区任务（广州站）——算子实操工坊。本任务要求在 Ascend 950PR 上，使用 Ascend C 编程语言开发单精度复数向量最小模元素索引算子 `aclblasIcamin`，完成算子设计、开发、测试全流程，并合入昇腾算子开源仓（ops-blas）。

### 1.2 背景介绍

#### 1.2.1 aclblasIcamin 算子实现优化
基于 cuBLAS `cublasIcamin` 核心功能与参数语义，使用 Ascend C 在昇腾 NPU 上进行原生开发与性能优化。标准 Netlib BLAS 无 `icamin` 例程，golden 由测试工程内 cblas 风格 CPU 参考循环生成。该算子用于在复数向量中查找最小模元素索引，属于 BLAS Level 1 归约算子，在复数线性代数计算中用于索引定位。

#### 1.2.2 aclblasIcamin 算子参考实现现状分析
当前 `ops-blas` 开源仓中已有同族实数算子 `aclblasIsamin`（位于 `blas/iamin/arch35/`），尚无复数版本实现。本算子需参照 `isamin` 的 arch35 归约实现框架，实现句柄式 BLAS 接口，通过 handle 绑定 stream 直调 NPU kernel，代码放在 `blas/iamin/arch35/`，测试工程参照 `test/isamin/` 模式新建于 `test/iamin/icamin/arch35/`。

#### 1.2.3 aclblasIcamin 算子功能分析
**输入**：`n`、复数向量 `x`、步长 `incx`。  
**输出**：`result` 为 1-based 最小模元素索引。  
**支持数据类型**：输入 COMPLEX64，输出 INT32。  
**数学表达式**：`result = argmin_i (|Re(x[k])| + |Im(x[k])|)`，`i = 1..n`，`k = 1+(i-1)*incx`。  
复数“模”按 BLAS icamin 惯例定义为 `|Re(x[k])| + |Im(x[k])|`，非欧几里得模；多个元素模相同时返回最小索引。  
**对标基线接口**：cuBLAS `cublasIcamin`。

## 2 需求分析（required）

### 2.1 需求描述
使用 Ascend C 编程语言实现 `aclblasIcamin`，提供句柄式 BLAS 接口，支持 handle 绑定 stream 直调 NPU kernel，完成 complex64 向量最小模元素索引计算。输出单个 INT32 整数索引，精度判定为与 golden 精确一致（bit-exact）。满足 quick return、非法参数、边界与特殊值语义，并达到性能标杆。

### 2.2 需求拆解
1. **功能实现**：按 `|Re| + |Im|` 计算复数模，遍历 `i=1..n`，维护最小模及对应最小 1-based 索引；模相同时严格返回最小索引。
2. **边界与 quick return**：`n = 0` 或 `incx < 1`（含 0 与负步长）不触发 kernel，写 `result = 0` 并返回成功；`n < 0` 返回 `ACLBLAS_STATUS_INVALID_VALUE`；`n > 0` 且 `incx >= 1` 时 `x` 空指针返回 `ACLBLAS_STATUS_INVALID_VALUE`。
3. **接口与约束**：在 `include/cann_ops_blas.h` 新增声明，签名与 cuBLAS `cublasIcamin` 逐参数对齐；禁止定义 950PR 私有平行接口；校验 handle、result、x 空指针。
4. **性能达标**：Ascend 950PR 上 COMPLEX64 输入场景，平均单次耗时满足任务书 §3.3 标杆。

## 3 详细设计（required）

### 3.1 算子分析

#### 3.1.1 数学公式
查找单精度复数向量 `x` 中最小模元素的 1-based 索引：

$$
result = \arg\min_{i=1..n} \left( |\mathrm{Re}(x[k])| + |\mathrm{Im}(x[k])| \right)
$$

其中：

$$
k = 1 + (i-1) \cdot incx
$$

- 输出为 1-based 索引，兼容 Fortran 惯例。
- 当多个元素的模相同时，返回最小索引。
- 复数类型 `aclblasComplex` 实部/虚部各 float32，交错存储，n 个复数元素对应 2*n 个 float。
- `n = 0` 或 `incx < 1` 为合法 quick return：不触发 kernel，写 `result = 0` 并返回 `ACLBLAS_STATUS_SUCCESS`。
- `n < 0` 返回 `ACLBLAS_STATUS_INVALID_VALUE`。
- 含 NaN 时，按任务书 Q7 对齐：跳过 NaN 元素；首元素为 NaN 时基准置 `FLT_MAX`，最终以研发确认结论为准。

#### 3.1.2 支持数据类型
- 输入向量：`aclblasComplex`（COMPLEX64，实部/虚部均为 float32）。
- 输出索引：`int`（INT32）。
- 标量参数：`n`、`incx` 为 `int`。

#### 3.1.3 支持形状
逻辑一维 `[n]`，物理长度 `1+(n-1)*|incx|`。纯向量归约到标量索引，不涉及 broadcast、动态 shape、非连续 Tensor 和原地更新。

### 3.2 算子实现

#### 3.2.1 Host 侧设计

**接口设计**：在 `include/cann_ops_blas.h` 中新增如下声明：

```cpp
aclblasStatus_t aclblasIcamin(
    aclblasHandle_t handle,
    int n,
    const aclblasComplex* x,
    int incx,
    int* result);
```

**参数校验策略**：
1. 校验 `handle` 是否为空，为空返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`。
2. 校验 `result` 是否为空，为空返回 `ACLBLAS_STATUS_INVALID_VALUE`。
3. 校验 `n < 0`，返回 `ACLBLAS_STATUS_INVALID_VALUE`。
4. `n = 0` 或 `incx < 1`：quick return，不触发 kernel，通过运行时接口将 `result` 置 0，返回 `ACLBLAS_STATUS_SUCCESS`。
5. `n > 0` 且 `incx >= 1` 时，校验 `x` 是否为空，为空返回 `ACLBLAS_STATUS_INVALID_VALUE`。
6. `n > 0` 且 `incx >= 1` 且 `x` 非空：绑定 handle 携带的 stream，下发 Device 侧 Kernel。

**Kernel 下发与同步**：Device 路径采用 kernel 直调方式。Host 侧完成 Tiling 后下发 kernel，kernel 异步执行；调用方读回 Device 结果前须同步 `handle` 绑定的 stream。测试工程负责 warmup、采样与同步。

**Tiling 策略**：
- 纯归约算子，按 `n` 和核数切分。
- 小 `n` 使用单核；大 `n` 使用多核并行，每个 core 处理连续 `i` 区间 `[start, end)`。
- 每个 core 根据 `i` 区间映射物理索引 `k = 1+(i-1)*incx`，计算局部最小模和 1-based 索引。
- 局部结果写入 workspace，最终通过二级归约合并为全局最小索引写入 `result`。
- 归约操作满足结合律：模更小者优先；模相等时索引更小者优先。

#### 3.2.2 Kernel 侧设计

**整体流程**：采用 Kernel 直调方式，包含数据搬入、计算、搬出/归约三个阶段。

1. **CopyIn 阶段**：按分配到的 `i` 区间读取复数元素，实部/虚部交错读取。
2. **Compute 阶段**：
   - 计算 `mag = |Re| + |Im|`。
   - 维护局部 `best_mag` 与 `best_idx`。
   - 若 `mag < best_mag`，更新 `best_mag` 和 `best_idx`。
   - 若 `mag == best_mag` 且 `i < best_idx`，更新 `best_idx`。
   - 含 NaN 元素按任务书 Q7 跳过。
3. **归约/搬出阶段**：将局部 `(best_mag, best_idx)` 写入 workspace；二级归约后由最终核将 `best_idx` 写入 `result`。

**核心算法伪代码**：

```cpp
best_mag = FLT_MAX;
best_idx = 0;

for (i = start; i < end; ++i) {
    k = 1 + (i - 1) * incx;
    real = x[2 * (k - 1)];
    imag = x[2 * (k - 1) + 1];
    mag = abs(real) + abs(imag);

    if (isnan(mag)) {
        continue; // 按任务书 Q7 跳过 NaN
    }
    if (mag < best_mag) {
        best_mag = mag;
        best_idx = i;
    } else if (mag == best_mag && i < best_idx) {
        best_idx = i;
    }
}

// 局部结果写入 workspace
workspace[block_id].mag = best_mag;
workspace[block_id].idx = best_idx;
```

**特殊值处理**：
- 全零输入：所有模为 0，返回最小索引 1。
- Inf 输入：按 `|Re| + |Im|` 正常参与比较。
- NaN 输入：按任务书 Q7 跳过 NaN 元素；首元素为 NaN 时基准置 `FLT_MAX`。
- `n = 0` 或 `incx < 1`：Host 侧 quick return，不进入 Kernel。

#### 3.2.3 内存与步长
- `x` 为 Device 只读内存，逻辑一维 `[n]`，物理长度 `1+(n-1)*|incx|`。
- `result` 为 Device 输出标量，INT32。
- `incx >= 1` 正常计算；`incx < 1` 不反向遍历，统一 quick return。
- 不要求非连续 Tensor 支持，步长由 `incx` 表达。

### 3.3 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Ascend 950PR | √ |

### 3.4 算子约束限制
1. `handle` 不可为空，否则返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`。
2. `result` 不可为空，否则返回 `ACLBLAS_STATUS_INVALID_VALUE`。
3. `n >= 0`；`n < 0` 返回 `ACLBLAS_STATUS_INVALID_VALUE`。
4. `n = 0` 或 `incx < 1` 为合法 quick return，写 `result = 0` 并返回成功。
5. `n > 0` 且 `incx >= 1` 时 `x` 不可为空，否则返回 `ACLBLAS_STATUS_INVALID_VALUE`。
6. 作为归约到标量索引的算子，不涉及非连续 Tensor、Broadcast、动态 Shape 和空 Tensor 的额外概念。
7. 异步执行依赖 `aclblasSetStream` 绑定 stream；读回 Device 结果前须同步 stream。
8. 接口声明放入 `include/cann_ops_blas.h`，禁止定义 950PR 私有平行接口。

## 4 可维可测分析

### 4.1 精度标准/性能标准

#### 4.1.1 精度验收标准
本算子计算精度需满足生态算子开源精度标准。输入为单精度复数 COMPLEX64，分量按 FLOAT32 档判定；输出为 INT32 整数索引，精度验证退化为精确一致判定。

| 数据类型 | rtol | atol | required_matched_ratio | max_abs_error_limit |
| --- | --- | --- | --- | --- |
| COMPLEX64（分量 FLOAT32 档） | 2^-10 (9.77e-4) | 2^-16 (1.53e-5) | 0.99 | 1e-2 或 32 * ULP |

通过条件：输出为整数索引时，浮点阈值退化为 `actual == golden`，即 1-based 索引逐位相等；matched_ratio 退化为单值判定。quick return 用例判定 `result == 0`。含 NaN 用例按任务书 Q7 的 golden 语义对齐。

#### 4.1.2 性能验收标准
测试设备：Ascend 950PR。性能数据为 COMPLEX64 输入场景下的平均单次耗时（Avg time，单位 us），须先 warmup 再有效采样 >50 次取平均。

| case | n | incx | 标杆耗时（Avg time，us） |
| --- | --- | --- | --- |
| 1 | 1048576 | 1 | 24.77 |
| 2 | 2097152 | 1 | 24.59 |
| 3 | 4194304 | 1 | 29.69 |

### 4.2 自验用例覆盖
基于 `icamin_test.csv` 的用例集（1200 条），自验覆盖以下类别：

| 类别 | 前缀 | 条数 | 说明 |
| --- | --- | --- | --- |
| L0 基础 | TC_L0 | 6 | 小尺寸（n=1/8）× 基础步长（1/2）+ tie 语义用例（全零填充多元素模相同，期望最小索引 1） |
| L1 尺寸 | TC_SQ | 38 | 38 种尺寸（1→1048576，含质数/边界/非对齐），incx=1 |
| L2 步长 | TC_INC | 17 | 正步长（1/2/3 × n∈{16,64,256}）正常计算 + 非正步长（0/-1/-2/-3 × n∈{16,64}）quick return |
| L5 填充 | TC_FL | 12 | 均匀随机/全零/交替/极端值/Inf/NaN × 2 尺寸 |
| L6 边界 | TC_ED | 12 | n=0 quick return（×5 步长）、负 n（×3，期望 INVALID_VALUE）、x 空指针（×2，期望 INVALID_VALUE）、quick return 优先于指针检查组合（×2，期望 SUCCESS） |
| EX 扩展 | TC_EX | 915 | 尺寸 × 步长 × 填充的确定性采样 |
| PF 性能 | TC_PF | 200 | 3 条任务书典型 case + 小尺寸 + 规模扫描 + 特殊填充 + 混合，均为连续访存（incx=1） |

精度 golden 由测试工程内 cblas 风格 CPU 参考循环生成，整数索引按 `EXPECT_EQ` 精确比对。

### 4.3 兼容性分析
本算子为新增算子，不涉及存量代码的兼容性改造。接口签名严格对齐 `cublasIcamin`，并在 `include/cann_ops_blas.h` 中声明供其他产品线共用。测试工程参照 `test/isamin/` 模式新建，CSV 列格式对齐 `test/isamin/isamin_param.h` 的参数定义。README 产品支持表标注 Ascend 950PR：支持。算子代码合入 `blas/iamin/arch35/`，测试代码合入 `test/iamin/icamin/arch35/`。