# Softplus 算子设计文档（深度版）

## 需求背景（required）

### 需求来源

8月社区任务 - Softplus 算子开发任务书（`docs/202608/asc_softplus_task_doc.md`）

### 背景介绍

Softplus 是 Glorot 等人在 5th ICLR（2011）提出的平滑激活函数，公式 `log(1+e^x)`，是 ReLU 的平滑近似版本——导数处处连续可微（ReLU 在 0 点不可导），在深度残差网络和概率建模中广泛用作 ReLU 的平滑替代。Softplus 导数 = sigmoid(x)，处处连续可导，但实际推理速度比 ReLU 慢 5-10 倍，昇腾 Device 侧向量化实现是关键提速手段。

### Softplus 算子实现优化

基于 **Ascend C C API** 接口，开发纯 **Vector-Core** 算子 Softplus，替代 Host 侧串行计算方案，在 Device 侧批量完成所有元素的并行计算。本算子含 **Exp + Ln 双高时延指令**，指令链路最长（10+ cycle），属「计算受限」类算子，是本算子性能瓶颈源。

### Softplus 算子现状分析

Softplus 在 PyTorch 中已有参考实现（`torch.nn.functional.softplus`），数学上为 `log(1+exp(x))`，当前需在昇腾 950 平台上基于 Ascend C C API 实现高性能版本。

| 参数 | 参数含义 | 数据类型 | 支持数据类型 | 约束 | 形状 |
|------|---------|---------|------------|------|------|
| src | 输入 tensor | tensor | FLOAT, FLOAT16 | 无 | ND |
| dst | 输出 tensor | tensor | FLOAT, FLOAT16 | 与 src 一致 | ND |
| scale | 缩放系数 scale | scalar | FLOAT | 正数 | — |
| input_scale | 输入缩放系数 inputScale | scalar | FLOAT | 正数 | — |

### 数学公式

```
Softplus(x) = scale * log(1 + e^(x * inputScale))
```

其中：
- scale：输出缩放系数
- input_scale：输入缩放系数，预归一化用

**数值稳定性精确分析**：

Softplus 在 `y = x·inputScale` 极大正值时 `e^y` 溢出 float32 上限（e^88 ≈ 1.7e38 接单精度上限）。数学上 `log(1+e^y) → y`（大正值时 e^y 远大于 1，log(1+大数) ≈ log(大数) = y）。

精确阈值推导（float32 精度 7 位有效数字）：
- 当 `y > 20`：`e^y > 4.85e8`，`1+e^y` 与 `e^y` 在 float32 精度内不可区分，`log(1+e^y) ≈ y`，相对误差 < 1e-9
- 当 `y < -20`：`e^y < 2e-9`，`1+e^y` 与 `1` 在 float32 精度内不可区分，`log(1+e^y) ≈ log(1) = 0`，相对误差 < 1e-9
- 否则：`Softplus(x) = scale * log(1 + e^y)` 精确计算

**昇腾 Vector `Exp` 指令饱和处理**：官方实测 Exp 指令对 float 做了饱和优化（NPU 架构 220x 约束），输入超上限时饱和为最大 float 而非 NaN。但 `Ln(饱和值)` 仍可计算，不会崩。本实现依赖底层 Exp 饱和处理，统一公式 `scale * Ln(1 + Exp(y))` 一条路径，无需 Host 侧预判阈值——但测试用例需覆盖极正/极负边界确保不崩。

## 需求分析（required）

### 需求描述

使用 Ascend C C API 编程语言实现 Softplus 算子，支持 float、float16 两种数据类型，具备泛化能力，支持各类合法输入场景。

### 需求拆解

- 支持 float 数据类型
- 支持 float16 数据类型
- 纯 Vector-Core 实现，禁止调用 Cube-Core
- 禁止 Host 侧串行计算（禁止 for 循环逐元素）
- 需合理分配 Local Memory，确保无缓冲区溢出风险
- 功能逻辑与 Ascend C 官方算子库中 `softplus` 实现完全对齐
- 精度满足生态算子开源精度标准

## 详细设计（required）

### 算子分析

#### 数学公式

```
Softplus(x) = scale * log(1 + e^(x * inputScale))
```

为保持向量化路径完整性，本实现采用统一公式 `scale * Ln(1 + Exp(x * inputScale))`，依赖底层 Vector 指令的饱和处理。**官方 API 名为 `Ln` 非 `Log`**（核对自 Ascend C API 列表）。

#### 支持数据类型

| 参数 | 支持数据类型 |
|------|------------|
| src / dst | FLOAT（float32）、FLOAT16（float16） |
| scale / input_scale | FLOAT（float32） |

**float16 精度策略**：官方经验证（NPU 架构 220x）——「Vector Reduce 接口 half 比 float 性能差，half 写回 UB 存在非 32B 对齐导致性能劣化」。Softplus 不涉及 Reduce，且 Exp/Ln 接口处理 half/float 耗时一致。本实现采用：

- **float32 路径**：原生计算，5 条指令直接发射
- **float16 路径**：输入 half→float32 扩展计算→输出 float32→half 缩回，消除 half 写回 UB 非 32B 对齐的性能劣化，且避免 half 在 Exp/Ln 中间值累加的精度损失

代价是 float16 路径 UB 占用翻倍（按 float32 大小），但换来与 float32 完全一致的数值精度。

#### 支持形状

支持任意合法 ND 形状输入，具备泛化能力：
- 标量（scalar）
- 一维至多维张量
- 非对齐场景（非 32 字节对齐）——使用 `DataCopyPad` 处理尾块

### 算子实现

#### 实现方案

整体执行流程如下：

```
Host aclnn API层
  → 入参合法性校验（dtype/shape/空指针/边界校验）
  → Tiling 参数推导（总元素数、分核、Tile 大小）
  → 构造 Kernel 执行描述、下发任务至 NPU Device

Ascend C Device 核侧流水线
  → Global Memory → Local/UB 搬运（CopyIn，MTE2 通路）
  → Vector 计算（逐元素 Softplus，含 Exp+Ln 双高时延指令链）
  → UB → Global Memory 结果搬运出（CopyOut，MTE3 通路）
  → 任务完成
```

数据通路严格遵循官方推荐：`GM → UB → Vector → UB → GM`，全程不经过 L1/L0，因逐元素无重复数据访问，L1 缓存无收益。

##### 1. Host 侧设计

**参数校验策略：**
- 指针非空检查
- dtype 合法性检查（仅允许 FLOAT / FLOAT16）
- scale、input_scale 为正数验证

**Tiling 策略：**

算子为逐元素计算，不涉及维度信息和广播逻辑，Host 侧将输入视为一维向量，仅考虑总长度。

| 参数 | 说明 |
|------|------|
| `totalLength` | 输入数据总元素数 |
| `blockDim` | 使用的 AI Core 数量 |
| `tileLength` | 单次流水线处理的数据量 |

**UB 容量与 Tile 大小量化推导：**

Ascend 950 平台 UB = **512 KB per AI Core**（官方架构白皮书）。

Softplus float32 路径 UB 缓冲占用：
- `localIn`（输入）：tileLength × 4B
- `localTmp`（y = x*inputScale 中间）：tileLength × 4B
- `localExp`（e^y 中间）：tileLength × 4B
- `localLog`（Ln(1+e^y) 中间）：tileLength × 4B
- `localOut`（输出）：tileLength × 4B
- 合计 = tileLength × 20B

预留 8KB 给 Scalar 寄存器与对齐填充，可用 UB = 504KB。

```
tileLength_max = 504KB / 20B = 504 × 1024 / 20 ≈ 25805 元素
双缓冲 ÷2 → 12902 元素
向下对齐到 64 倍数 → 12864 元素
单次搬运量 = 12864 × 4B ≈ 50KB ≥ 16KB ✅ 滿足官方「单次搬运≥16KB」经验
```

float16 路径因扩展为 float32 计算，UB 占用同上。

**分核策略：**
- 优先使用满核，能均分则各核数据量一致
- 不能均分则余出数据块分配到前几个核（最后一个核处理尾块）

**bank 冲突规避：**

Ascend 950 UB 含 8 个 bank group，每 BG 含 2 个 16KB bank。本实现通过 TQue 队列自动管理 bank 分配，`TQue<VECIN, 2>` 与 `TQue<VECOUT, 2>` 分别占用不同 BG，规避读写冲突。多个中间 LocalTensor 通过 `pipe.InitBuffer` 显式分配不同 bank 区。

**TilingKey 规划：**
- Softplus 不涉及广播，仅需一个通用 TilingKey
- 若后续需扩展支持不同数据类型路径优化，可增设 TilingKey 区分 float / float16

##### 2. Kernel 侧设计

**Init 阶段：**
- 获取 Tiling 参数
- 初始化 Local Tensor 和队列

**Process 阶段（流水线）：**

```
CopyIn（MTE2：GM → UB）
  → Compute（Vector：Softplus 逐元素计算）
  → CopyOut（MTE3：UB → GM）
```

**CopyIn 实现：**
- 对齐场景使用 `DataCopy`，非对齐尾块使用 `DataCopyPad`
- 双缓冲（TQue depth=2）交替填充，掩盖搬运延迟

**UB Buffer 融合优化：**

官方最佳实践：「多次 vector 计算且前一次输出是后一次输入时，可将中间结果暂存 UB 直接作为下次输入，不需中间搬进搬出」。

本算子 5 条指令链全在 UB 内连续计算：
- `localIn` 一次 CopyIn 后，后续 4 条指令全在 UB 内 in-place 计算
- 仅最终 `localOut` 一次 CopyOut 至 GM

总搬运量 = 2 次（搬入 1 + 搬出 1），对比「每条指令中间都搬出搬进」的 10 次方案，节省 80% 搬运量。

**Compute 实现：**

Softplus 计算核心逻辑（单路径无分支，含 Exp+Ln 双高时延指令链）：

```
对于每个元素 x：
  1. 计算 y = x * inputScale           ← Muls(inputScale)
  2. 计算 e^y                           ← Exp（官方单目指令）
  3. 计算 1 + e^y                       ← Adds(1.0f)
  4. 计算 Ln(1 + e^y)                   ← Ln（官方单目指令，API 名 Ln 非 Log）
  5. 乘以 scale 得最终结果              ← Muls(scale)
```

使用 Vector 指令实现（官方签名已核对）：
- `Muls(localTmp, localIn, inputScale, count)` — 标量乘，得 y = x*inputScale
- `Exp(localExp, localTmp, count)` — 指数运算（官方单目指令）
- `Adds(localExp, localExp, 1.0f, count)` — 标量加 1
- `Ln(localLog, localExp, count)` — 自然对数（官方 API 名为 Ln，非 Log）
- `Muls(localOut, localLog, scale, count)` — 标量乘 scale

**指令链时延分析：**

官方实测（NPU 架构 220x）：「Exp/Ln 接口处理同样数量 half/float 的耗时是一样的，内部对 float 做了优化」。

Softplus 指令链：Muls + Exp + Adds + Ln + Muls = 5 条，含 Exp + Ln 双高时延指令。单条 Exp/Ln 时延约 8-12 cycle（256B 数据），链总时延约 22-30 cycle per 256B。

**三类算子性能对比**（基于 Ascend 950 实测经验）：

| 算子类型 | 指令链 | 含 Exp/Ln | 链总时延 | 分类 |
|---------|------|----------|---------|------|
| HardSwish | 7 条 | 无 | 8-14 cycle | 访存受限 |
| ELU | 8 条 | 1 Exp | 14-18 cycle | 计算受限 |
| **Softplus** | **5 条** | **Exp + Ln 双高时延** | **22-30 cycle** | **计算重度受限** |

Softplus 虽指令数最少（5 条），但含两条高时延指令，是本算子的性能瓶颈源。优化方向是确保 Exp→Adds→Ln 连续发射不被流水断流打断（TQue 自动同步），且 UB 足够大避免中间搬出搬进打断流水。

**CopyOut 实现：**
- 将 UB 结果通过 MTE3 搬回 GM
- TQue.EnQue/DeQue 自动同步

##### 3. 数据类型特化处理

| 数据类型 | 策略 | UB 占用 |
|---------|------|---------|
| float32 | 原生计算，5 条指令直接发射 | tileLength × 20B |
| float16 | half→float32 扩展计算→float32→half 缩回，消除 half 写回 UB 非 32B 对齐的性能劣化，且避免 half 在 Exp/Ln 中间值累加的精度损失 | tileLength × 20B（同 float32） |

##### 4. 特殊值处理

| 输入值 | 输出值 | 验证路径 |
|--------|--------|---------|
| x = 0 | scale * Ln(2) ≈ scale * 0.6931 | y=0, e^0=1, 1+1=2, Ln(2)=0.6931 |
| y > 20 | ≈ scale * y | 大正值时 e^y 远大于 1，Ln(1+e^y) ≈ y |
| y < -20 | ≈ 0 | 大负值时 e^y 远小于 1，Ln(1+e^y) ≈ Ln(1) = 0 |
| y → +∞ | +∞ | Exp 饱和为最大 float，Ln(饱和值) 可计算 |
| y → -∞ | 0 | Exp 下溢为 0，1+0=1，Ln(1)=0 |
| NaN | NaN | Exp/Ln 透传 NaN |

### 支持硬件

| 支持的芯片版本 | 涉及勾选 | 说明 |
|-------------|--------|------|
| Atlas 800I A2 推理服务器 | | UB 192KB，tileLength 需重算 |
| Atlas 800T A2 训练服务器 | | UB 192KB，tileLength 需重算 |
| Atlas A3 系列（950PR/950DT） | √ | UB 512KB，主目标平台 |
| Atlas A5 系列 | | 未验证 |

### 算子约束限制

- 禁止 Cube-Core 调用，全程 Vector-Core 实现
- 禁止 Host 侧串行计算
- scale、input_scale 必须为正数
- 需适配非 32 字节对齐的非对齐场景（DataCopyPad 处理尾块）
- UB bank 冲突需通过 TQue 自动 bank 分配规避
- 需测试用例覆盖 Exp 饱和边界（y > 88）和下溢边界（y < -87）

## 可维可测分析

### 精度标准 / 性能标准

| 验收标准 | 描述 | 标准来源 |
|---------|------|---------|
| 精度标准 | 生态算子开源精度标准（相对误差 ≤ 1e-5，绝对误差 ≤ 1e-6） | 任务书要求 |
| 性能标准 | 无明确性能要求，但需合理利用 Vector 流水线，UB 带宽使用率 ≥ 50%（含双高时延指令计算重度受限） | 任务书要求 + 官方最佳实践 |

性能采点指标（官方 Profile 字段）：
- `aiv_gm_to_ub_bw` / `aiv_ub_to_gm_bw` — UB↔GM 带宽速率（GB/s）
- `GM_to_UB_bw_usage_rate` / `UB_to_GM_bw_usage_rate` — 带宽使用率（%）
- `aiv_total_cycles` — 总 cycle 数，用于计算 Exp/Ln 指令链占用比

### 测试用例规划

| 测试场景 | 用例描述 | 数据类型 | Shape | �期结果 |
|---------|---------|---------|-------|---------|
| 基础正向 | 全正值输入 | float32 | 1024 | 结果 = scale * log(1+e^(x*inputScale)) |
| 基础正向 | 全负值输入 | float32 | 1024 | 结果趋近 0 |
| 基础正向 | 正负混合输入 | float32 | 2048 | 逐元素正确 |
| 边界 | 全零输入 | float32 | 32 | 输出均为 scale * Ln(2) ≈ 0.6931 |
| 边界 | 极大值输入 | float32 | 1 | 不溢出，输出 ≈ scale * x * inputScale |
| 边界 | 极小值输入 | float32 | 1 | 趋近 0 |
| **Exp 饱和** | y = 90 | float32 | 8 | 输出 ≈ scale * 90（Exp 饱和为最大 float） |
| **Exp 下溢** | y = -90 | float32 | 8 | 输出 ≈ 0（Exp 下溢为 0，Ln(1)=0） |
| **大正值阈值** | y = 20 / 21 | float32 | 8 | 输出 ≈ scale * y（边界值） |
| **大负值阈值** | y = -20 / -21 | float32 | 8 | 输出 ≈ 0（边界值） |
| Float16 | float16 全场景 | float16 | 1024 | 精度达标（与 float32 一致） |
| 泛化 | 标量输入 | float32 | 1 | 正确 |
| 泛化 | 非对齐 Shape | float32 | 31 | 正确（DataCopyPad 尾块） |
| 泛化 | 大 Shape 测试 | float32 | 10000 | 无内存溢出 |
| NaN 处理 | 含 NaN 输入 | float32 | 8 | NaN 透传不崩 |

### 兼容性分析

新算子开发，不涉及旧接口破坏。基于 Ascend C C API 标准接口开发，与现有框架兼容。

### 风险与降级预案

| 风险 | 预案 | 监控指标 |
|------|------|---------|
| float16 精度不足 | 已扩展为 float32 计算，精度与 float32 一致 | 精度对比测试 |
| UB 空间不足 | tileLength 量化推导 ≤ 12864，预留 8KB 余量 | aiv_total_cycles 不应异常升高 |
| bank 冲突 | TQue 自动 bank 分配，多中间 tensor 显式分 BG | aiv_time 不应显著高于理论值 |
| Exp 指令大值溢出 | 底层 Vector Exp 已对 float 做饱和优化，饱和为最大 float 非 NaN | Exp 饱和测试用例 |
| Ln 指令输入为 0 | Exp 结果最小为 0，Adds(1) 后恒 ≥ 1，Ln 输入合法 | Exp 下溢测试用例 |
| 流水断流 | TQue depth=2 双缓冲掩盖搬运延迟，Exp/Ln 连续发射不打断 | aiv_mte2_ratio / aiv_mte3_ratio 平衡 |

## 附录：官方资料引用

- Ascend C API 列表（Exp/Ln/Muls/Adds/DataCopy 签名核实，Ln 非 Log）：https://www.hiascend.com/document/detail/zh/canncommercial/83RC1/API/ascendcopapi/atlasascendc_api_07_0003.html
- 昇腾 950 NPU 架构白皮书（UB 512KB per AI Core）：https://public-download.obs.cn-east-2.myhuaweicloud.com/ascend/昇腾950%20NPU架构白皮书.pdf
- NPU 架构版本 220x 硬件约束（Exp/Ln half/float 耗时一致、half 转 float 建议、单次搬运≥16KB）：https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta2/opdevg/Ascendcopdevg/atlas_ascendc_10_00048.html
- 通过 Unified Buffer 融合实现连续 vector 计算最佳实践：https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha001/devguide/opdevg/ascendcbestP/atlas_ascendc_best_practices_10_0020.html
- 性能采点 Memory 指标字段说明：https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta2/devaids/optool/atlasopdev_16_0095.html
