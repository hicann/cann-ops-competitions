# 【社区任务】ELU 算子设计文档评审申请

## 一、需求背景（Required 强制模块）

### 1.1 任务来源

本任务是昇腾 2026 年 8 月社区任务「ELU 算子开发」，任务书见
[asc_elu_task_doc.md](../cann-ops-competitions/04_tasks/01_community-task-2026/docs/202608/asc_elu_task_doc.md)。

### 1.2 开发目标

基于 **Ascend C C API** 接口，开发纯 **Vector-Core** 算子 `ELU`（Exponential Linear Unit），
核心约束如下：

- **禁止调用 Cube-Core 单元**：全程仅使用 Vector 指令完成计算；
- **禁止 Host 侧串行计算**：所有计算必须在 Device 侧批量完成；
- **内存安全**：合理分配 Local Memory，无缓冲区溢出；
- **对齐标准**：功能逻辑与 Ascend C 官方算子库（ops-nn `activation/elu`）实现完全对齐；
- **数据类型**：必须支持 `float`、`float16`；
- **泛化能力**：支持各类合法输入场景（任意长度、任意 alpha/scale/input_scale）。

### 1.3 数学公式

$$
ELU(x) = \begin{cases}
scale \cdot x, & x > 0 \\
\alpha \cdot scale \cdot \left(e^{x \cdot inputScale} - 1\right), & x \le 0
\end{cases}
$$

其中 $\alpha$（激活系数）、$scale$（缩放系数）、$inputScale$（输入缩放系数）为可配置标量。

### 1.4 官方实现现状分析

官方 elu 算子（ops-nn `activation/elu/op_kernel/elu_apt.cpp` + `arch35/elu_dag.h`）的实现要点：

1. `z = x * inputScale`；
2. `expm1(z) = exp(z) - 1` 采用分段实现：
   - 当 `|z| < 0.1` 时使用泰勒多项式近似（float 取 5 项：`z + z²/2 + z³/6 + z⁴/24 + z⁵/120`，half 取 4 项：`z + z²/2 + z³/6 + z⁴/24`）；
   - 否则使用 `exp(z) - 1`；
3. `neg = expm1(z) * alpha`；
4. `out = (x > 0) ? x : neg`；
5. `out = out * scale`。

本算子设计与该逻辑完全一致。

---

## 二、需求分析（Required 强制模块）

### 2.1 核心需求描述

使用 Ascend C C API（Reg 矢量计算模式）实现 ELU 算子，支持 `float` / `float16` 两种数据类型，
功能逻辑与官方 elu 完全对齐，满足生态算子开源精度标准，全部计算在 Vector 单元完成。

### 2.2 全量需求拆解与验收标准

| 拆解子项 | 验收标准 |
| --- | --- |
| 数据类型覆盖 | 支持 FLOAT、FLOAT16，与任务书参数表完全一致 |
| 泛化能力 | 支持任意合法输入长度（含非 32B 对齐尾块）、任意 alpha/scale/input_scale 标量 |
| 精度对齐 | 满足生态算子开源精度标准：FLOAT32 rtol=2⁻¹⁰/atol=2⁻¹⁶、FLOAT16 rtol=2⁻⁹/atol=2⁻⁹，通过率 ≥ 0.99 |
| 计算约束 | 纯 Vector 计算（不使用 Cube），Host 侧无逐元素串行计算 |
| 内存安全 | Local Memory 合理分配，无缓冲区溢出 |

### 2.3 外部组件依赖

不涉及外部组件依赖；golden 生成依赖 `torch.ops.aten.elu`（与官方算子库 golden 逻辑一致）。

---

## 三、详细设计（Required 强制模块）

### 3.1 算子基础分析

#### 3.1.1 数学语义

严格遵循任务书公式与官方实现：`x > 0` 时输出 `scale*x`，`x ≤ 0` 时输出 `alpha*scale*expm1(x*inputScale)`。

#### 3.1.2 参数规范

| 参数名 | 输入/输出 | 描述 | 数据类型 | 数据格式 |
| :--- | :--- | :--- | :--- | ---- |
| `src` | Input | 源操作数（公式中的 x） | FLOAT / FLOAT16 | ND |
| `alpha` | Input | 激活系数（公式中的 α） | FLOAT | 标量 |
| `scale` | Input | 缩放系数（公式中的 scale） | FLOAT | 标量 |
| `input_scale` | Input | 输入缩放系数（公式中的 inputScale） | FLOAT | 标量 |
| `dst` | Output | 目的操作数 | FLOAT / FLOAT16 | ND |

核函数名：`elu_custom`，入参为 `(__gm__ DT* src, __gm__ DT* dst, float alpha, float scale, float input_scale)`，
编译期宏 `ELU_DTYPE_HALF`（0/1）选择数据类型，`SIZE` 指定数据长度。

#### 3.1.3 支持形状

一维 ND 张量，任意长度（含非 32B 对齐长度）。shape 支持：1 / 32 / 1024 / 2048（任务书要求），
并支持任意大尺寸（多 block 分片）。

### 3.2 Host 侧设计

#### 3.2.1 Tiling / 分核策略

- 数据视为一维向量，仅考虑元素个数，不考虑维度信息；
- **多 block 分片**：每个 block 处理 `TILE_LENGTH`（8192）个元素，Host 侧按
  `blockNum = ceil(total_length / TILE_LENGTH)` 启动核数；最后一个 block 处理尾块（不足 8192 的元素）；
- 无大小核区分，各 block 负载均匀（前 `blockNum-1` 个 block 满 8192，最后 1 个 block 为尾块）；
- 每个 block 内部单次完成 CopyIn → Compute → CopyOut，不引入多 tile 循环。

#### 3.2.2 UB 内存分配

每个 block 在 UB 内静态分配两块缓冲区：

```cpp
__ubuf__ DT x_local[TILE_LENGTH];   // 输入缓冲区：float 32KB / half 16KB
__ubuf__ DT y_local[TILE_LENGTH];   // 输出缓冲区：float 32KB / half 16KB
```

GM ↔ UB 搬运使用 `asc_copy_gm2ub_align` / `asc_copy_ub2gm_align`（`n_burst=1`），
支持非 32B 对齐数据自动 32B 补齐（硬件填充 dummy，GM 写出时丢弃），保证尾块搬运无越界。

### 3.3 Kernel 侧设计

核函数为 `__vector__ __global__`，流程如下：

```mermaid
flowchart TD
    A[Kernel 启动 asc_init] --> B[计算本 block 的 offset 与 cur_len]
    B --> C[CopyIn: GM→UB 批量搬运输入分片]
    C --> D[流水同步 MTE2→V]
    D --> E[Compute: 寄存器级向量计算 ELU]
    E --> E1[z = x * inputScale]
    E1 --> E2[expm1 分段: |z|<0.1 多项式 / exp(z)-1]
    E2 --> E3[neg = expm1 * alpha]
    E3 --> E4[sel = x>0 ? x : neg]
    E4 --> E5[out = sel * scale]
    E5 --> F[流水同步 V→MTE3]
    F --> G[CopyOut: UB→GM 写回输出分片]
    G --> H[结束]
```

#### 3.3.1 float 路径

每寄存器组 4 个元素（b32 mask），使用的 C API 指令：

| 计算步骤 | 指令 |
| --- | --- |
| z = x * inputScale | `asc_mul_scalar` |
| 多项式（Horner，5 项） | `asc_mul_scalar` / `asc_add_scalar` / `asc_mul` |
| exp(z) - 1 | `asc_exp` + `asc_add_scalar(-1)` |
| \|z\| < 0.1 选择 | `asc_abs` + `asc_lt_scalar` + `asc_select` |
| * alpha | `asc_mul_scalar` |
| x > 0 选择 | `asc_gt_scalar` + `asc_select` |
| * scale | `asc_mul_scalar` |

#### 3.3.2 float16 路径

每寄存器组 8 个元素（b16 mask），计算流程与 float 路径同构，多项式取 4 项，
全部计算在 half 精度下使用 `vector_half` 指令完成（`asc_exp`、`asc_mul_scalar`、
`asc_add_scalar`、`asc_abs`、`asc_lt_scalar`、`asc_gt_scalar`、`asc_select` 均有 half 版本）。

#### 3.3.3 掩码与尾块处理

循环内 `vmask = asc_update_mask_b16/b32(data_len - i * one_rep_size)`（POST_UPDATE 自动递减），
最后一次迭代的掩码只覆盖剩余元素，保证非对齐长度（如 1001、12345）计算正确；
未掩码的寄存器通道不写回，尾块多余数据由搬运接口自动补齐且不写入 GM。

#### 3.3.4 流水同步

每个 block 内采用 `asc_sync_notify(PIPE_MTE2, PIPE_V, EVENT_ID0)` / `asc_sync_wait(...)`
与 `asc_sync_notify(PIPE_V, PIPE_MTE3, EVENT_ID0)` / `asc_sync_wait(...)` 完成 MTE2/V/MTE3
流水线同步，末尾 `asc_sync()` 收尾。

### 3.4 实现要点说明

1. 单 block 单次搬运（无多 tile 循环），规避了 Reg 模式下多 tile 循环复用 `EVENT_ID0`
   的 flag 同步异常问题（详见自测报告「易用性问题」）。
2. Host 侧除数据文件读写与核函数启动外无任何逐元素运算。
3. 精度验证按生态算子开源精度标准实现（混合容差 + 通过率 + 最大绝对误差上限）。

---

## 四、支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Ascend 950PR / 950DT | √ |
| Atlas A3 / A2 训练推理系列 | 不支持（C API Reg 模式 3510 专用） |

## 五、算子约束限制

- 仅支持 ND 格式、一维输入（元素级算子，任意长度）；
- 不支持广播（任务书无广播要求，输入输出 shape 一致）；
- 仅支持 float / float16（任务书要求）。

---

## 六、可维可测分析

### 6.1 精度标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | 满足生态算子开源精度标准（混合容差），golden 与官方一致（torch.ops.aten.elu） | 任务书 3.3 / opbase 精度标准 |

### 6.2 性能标准

任务书明确「性能要求：无」。自测报告中提供参考性能数据（含 Host 启动开销）。

### 6.3 兼容性分析

新算子，不涉及历史版本兼容性分析。CANN 兼容范围：CANN 9.0.0 ~ CANN 9.1.0（任务书要求）。

### 6.4 自测计划

| 类别 | 用例 |
| --- | --- |
| 功能 | float/half × shape {1, 32, 1024, 2048}；非默认 alpha/scale/input_scale；特殊值（0/-0/±inf/NaN/阈值边界）；大尺寸（1001/12345/65537/100000）多 seed |
| 精度 | matched_ratio ≥ 0.99 且 max_abs_error ≤ 上限（FLOAT32 1e-2 / FLOAT16 1e-1） |
| 性能 | `--perf` 模式循环执行核函数，统计平均耗时与吞吐（参考） |
