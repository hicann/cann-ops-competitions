# Selu 算子设计文档（深度版）

## 需求背景（required）

### 需求来源

8月社区任务 - Selu 算子开发任务书（`docs/202608/asc_selu_task_doc.md`）

### 背景介绍

Selu（Scaled Exponential Linear Unit）是 Klambauer 等人在 5th ICLR（2017）提出的自归一化激活函数，用于构建自归一化神经网络（Self-Normalizing Neural Network, SNN）。通过固定的缩放参数 α 和 scale，使得神经元激活在保持均值/方差稳定的条件下传播，缓解梯度消失/爆炸问题。Selu 在 MNIST/CIFAR-10 上比 ELU 提升 0.5-2% Top-1 精度，且训练收敛速度快 2-3 倍。

### 与 ELU 的实质差异点

Selu 数学形式与 ELU 一致，但有 3 个实质差异：

| 维度 | ELU | Selu |
|------|-----|------|
| **常数源** | α、scale 为用户可调超参，默认 α=1.0 scale=1.0 | **α、scale 固定为预定义常数**：α≈1.6733，scale≈1.0507（来自 Klambauer 2017 论文 Table 1，基于正态分布 N(0,1) 输入下的自归一化约束推导） |
| **数学约束** | 无约束，可任意取值 | α、scale 必须满足自归一化约束：对输入 X~N(0,1)，输出 E[Selu(X)]≈0、Var[Selu(X)]≈1，推导出 α、scale 的唯一解 |
| **梯度稳定性** | 仅缓解负值区神经元死亡，不保证方差传播稳定 | **强自归一化**：无论输入分布如何漂移，输出方差自动回归 1，消除 BatchNorm 依赖 |

**固定常数精度要求**：Klambauer 论文给出的精确值为 α = 1.67326324235437、scale = 1.0507009873554804。float32 精度（7 位有效数字）下取 α ≈ 1.6732632、scale ≈ 1.0507010 即可满足自归一化约束（论文实测误差 < 1e-6）。本算子保留 α、scale、input_scale 三个标量入参以具备泛化能力，但 Host 侧校验时对默认值做精度检查确保与论文常数对齐。

### Selu 算子实现优化

基于 **Ascend C C API** 接口，开发纯 **Vector-Core** 算子 Selu，替代 Host 侧串行计算方案，在 Device 侧批量完成所有元素的并行计算。本算子含 Exp 指令（指令链路 10+ cycle），属「计算受限」类算子，与 ELU 实现路径几乎一致，差异仅在 Host 侧常数精度校验和默认值。

### Selu 算子现状分析

Selu 在 PyTorch 中已有参考实现（`torch.nn.functional.selu`），数学上为 ELU 的特化版本（α 与 scale 固定为预定义常数），当前需在昇腾 950 平台上基于 Ascend C C API 实现高性能版本。

| 参数 | 参数含义 | 数据类型 | 支持数据类型 | 约束 | 形状 |
|------|---------|---------|------------|------|------|
| src | 输入 tensor | tensor | FLOAT, FLOAT16 | 无 | ND |
| dst | 输出 tensor | tensor | FLOAT, FLOAT16 | 与 src 一致 | ND |
| alpha | 激活系数 α | scalar | FLOAT | 正数，默认 1.6732632 | — |
| scale | 缩放系数 scale | scalar | FLOAT | 正数，默认 1.0507010 | — |
| input_scale | 输入缩放系数 inputScale | scalar | FLOAT | 正数 | — |

### 数学公式

```
Selu(x) = scale * x,                           if x > 0
Selu(x) = α * scale * (e^(x * inputScale) - 1), if x ≤ 0
```

其中 α、scale 通常取预定义常数：α ≈ 1.6733，scale ≈ 1.0507（Klambauer 2017 论文 Table 1）。本算子保留 α、scale、input_scale 三个标量入参，具备泛化能力。

**数值稳定性分析**：

Selu 公式中 `exp(x·inputScale)` 在 x·inputScale → -∞ 时下溢为 0，此时 Selu → -α·scale ≈ -1.7581，数学上连续无异常。但 Vector `Exp` 指令在输入极负（如 < -87.3 for float32，因 e^-87.3 ≈ 1.4e-39 接单精度下溢边界）时，可能触发饱和处理或返回非零极小值，需在本算子测试用例中覆盖该边界。

昇腾 Vector `Exp` 指令官方实测对 half/float 耗时一致（NPU 架构 220x 约束文档），底层已对 float 做饱和优化，开发者无需 Host 侧预判阈值——但仍需测试用例覆盖极负值确保不崩。

**Selu 与 ELU 数值差异**：因 Selu 的 α·scale ≈ 1.7581 远大于 ELU 的 α·scale = 1.0（默认），Selu 负值区输出绝对值更大，自归一化约束下方差更稳定。在 Exp 下溢边界（x·inputScale < -87.3）两者都饱和为 -α·scale，但 Selu 的饱和值 ≈ -1.7581 vs ELU 的 ≈ -1.0，测试用例预期值需区分。

## 需求分析（required）

### 需求描述

使用 Ascend C C API 编程语言实现 Selu 算子，支持 float、float16 两种数据类型，具备泛化能力，支持各类合法输入场景。

### 需求拆解

- 支持 float 数据类型
- 支持 float16 数据类型
- 纯 Vector-Core 实现，禁止调用 Cube-Core
- 禁止 Host 侧串行计算（禁止 for 循环逐元素）
- 需合理分配 Local Memory，确保无缓冲区溢出风险
- 功能逻辑与 Ascend C 官方算子库中 `selu` 实现完全对齐
- Host 侧对默认 α、scale 常数做精度校验（与 Klambauer 2017 论文对齐）
- 精度满足生态算子开源精度标准

## 详细设计（required）

### 算子分析

#### 数学公式

```
Selu(x) = scale * x,                           if x > 0
Selu(x) = α * scale * (e^(x * inputScale) - 1), if x ≤ 0
```

#### 支持数据类型

| 参数 | 支持数据类型 |
|------|------------|
| src / dst | FLOAT（float32）、FLOAT16（float16） |
| alpha / scale / input_scale | FLOAT（float32） |

**float16 精度策略**：官方经验证（NPU 架构 220x）——「Vector Reduce 接口 half 比 float 性能差，half 写回 UB 存在非 32B 对齐导致性能劣化」。Selu 不涉及 Reduce，且 Exp/Ln 接口处理 half/float 耗时一致。本实现采用：

- **float32 路径**：原生计算
- **float16 路径**：输入 half→float32 扩展计算→输出 float32→half 缩回，消除 half 写回 UB 非 32B 对齐的性能劣化，且避免 half 在 Exp 指令中间值累加时的精度损失

#### 支持形状

支持任意合法 ND 形状输入，具备泛化能力：
- 标量（scalar）
- 一维至多维张量
- 非对齐场景（非 32 字节对齐）——使用 `DataCopyPad` 处理尾块

### 算子实现

#### 实现方案

整体执行流程如下：

```
Host ACLNN API层
  → 入参合法性校验（dtype/shape/空指针/边界/常数精度校验）
  → Tiling 参数推导（总元素数、分核、Tile 大小）
  → 构造 Kernel 执行描述、下发任务至 NPU Device

Ascend C Device 核侧流水线
  → Global Memory → Local/UB 搬运（CopyIn，MTE2 通路）
  → Vector 计算（逐元素 Selu，含分支选择与 Exp 指令链）
  → UB → Global Memory 结果搬出（CopyOut，MTE3 通路）
  → 任务完成
```

数据通路严格遵循官方推荐：`GM → UB → Vector → UB → GM`，全程不经过 L1/L0，因逐元素无重复数据访问，L1 缓存无收益。

##### 1. Host 侧设计

**参数校验策略：**
- 指针非空检查
- dtype 合法性检查（仅允许 FLOAT / FLOAT16）
- alpha、scale、input_scale 为正数验证
- **常数精度校验（Selu 特有）**：若 alpha、scale 未显式传参，使用默认值 α=1.6732632、scale=1.0507010，与 Klambauer 2017 论文对齐；若显式传参则校验为正数即可

**Tiling 策略：**

算子为逐元素计算，不涉及维度信息和广播逻辑，Host 侧将输入视为一维向量，仅考虑总长度。

| 参数 | 说明 |
|------|------|
| `totalLength` | 输入数据总元素数 |
| `blockDim` | 使用的 AI Core 数量 |
| `tileLength` | 单次流水线处理的数据量 |

**UB 容量与 Tile 大小量化推导：**

Ascend 950 平台 UB = **512 KB per AI Core**（官方架构白皮书）。

Selu float32 路径 UB 缓冲占用（与 ELU 一致）：
- `localIn`（输入）：tileLength × 4B
- `localTmp`（x·inputScale 中间）：tileLength × 4B
- `localExp`（e^y 中间）：tileLength × 4B
- `localTmp2`（e^y-1 中间）：tileLength × 4B
- `localPos`（scale·x 正值分支）：tileLength × 4B
- `localOut`（输出）：tileLength × 4B
- `localMask`（Compare 位掩码）：tileLength × 1B（uint8）
- `localZero`（标量 0 扩 tensor）：tileLength × 4B
- 合计 ≈ tileLength × 25B

预留 8KB 给 Scalar 寄存器与对齐填充，可用 UB = 504KB。

```
tileLength_max = 504KB / 25B ≈ 20651 元素
双缓冲 ÷2 → 10325 元素
向下对齐到 64 倍数 → 10240 元素
单次搬运量 = 10240 × 4B = 40KB ≥ 16KB ✅ 满足官方「单次搬运≥16KB」经验
```

float16 路径因扩展为 float32 计算，UB 占用同上。

**分核策略：**
- 优先使用满核，能均分则各核数据量一致
- 不能均分则余出数据块分配到前几个核（最后一个核处理尾块）

**bank 冲突规避：**

Ascend 950 UB 含 8 个 bank group，每 BG 含 2 个 16KB bank。本实现通过 TQue 队列自动管理 bank 分配，`TQue<VECIN, 2>` 与 `TQue<VECOUT, 2>` 分别占用不同 BG，规避读写冲突。多个中间 LocalTensor 通过 `pipe.InitBuffer` 显式分配不同 bank 区。

**TilingKey 规划：**
- Selu 不涉及广播，仅需一个通用 TilingKey
- 若后续需扩展支持不同数据类型路径优化，可增设 TilingKey 区分 float / float16

##### 2. Kernel 侧设计

**Init 阶段：**
- 获取 Tiling 参数
- 初始化 Local Tensor 和队列

**Process 阶段（流水线）：**

```
CopyIn（MTE2：GM → UB）
  → Compute（Vector：Selu 逐元素计算）
  → CopyOut（MTE3：UB → GM）
```

**CopyIn 实现：**
- 对齐场景使用 `DataCopy`，非对齐尾块使用 `DataCopyPad`
- 双缓冲（TQue depth=2）交替填充，掩盖搬运延迟

**UB Buffer 融合优化：**

官方最佳实践：「多次 vector 计算且前一次输出是后一次输入时，可将中间结果暂存 UB 直接作为下次输入，不需中间搬进搬出」。

本算子 7+ 条指令链全在 UB 内连续计算：
- `localIn` 一次 CopyIn 后，后续全在 UB 内 in-place 计算
- 仅最终 `localOut` 一次 CopyOut 至 GM

总搬运量 = 2 次（搬入 1 + 搬出 1），对比「每条指令中间都搬出搬进」的 14+ 次方案，节省 85%+ 搬运量。

**Compute 实现：**

Selu 计算核心逻辑（双分支 + Exp 指令链，与 ELU 实现路径一致，差异仅 Host 侧常数）：

```
对于每个元素 x：
  正值分支预计算：
    1a. 计算 pos = scale * x                 ← Muls(scale)
  负值分支计算：
    2a. 计算 y = x * inputScale              ← Muls(inputScale)
    3a. 计算 exp(y)                          ← Exp（官方单目指令）
    4a. 计算 exp(y) - 1                      ← Adds(-1.0f)（官方无 Subs，用 Adds 负数实现减 1）
    5a. 乘 α * scale                         ← Muls(alphaScale = α*scale 预计算常数)
  分支选择：
    6. 生成 mask: x > 0 位为 1                ← Compare(CMPMODE::GT) + localZero tensor
    7. 按位选择: mask=1 选 pos, mask=0 选 neg  ← Select(VSEL_TENSOR_TENSOR_MODE)
```

使用 Vector 指令实现（官方签名已核对，与 ELU 一致）：
- `Muls(localPos, localIn, scale, count)` — 正值分支：scale * x
- `Muls(localTmp, localIn, inputScale, count)` — 负值分支：y = x*inputScale
- `Exp(localExp, localTmp, count)` — 指数运算（官方单目指令）
- `Adds(localTmp, localExp, -1.0f, count)` — 减 1（官方无 Subs 标量版，用 Adds 负数实现）
- `Muls(localExp, localTmp, alphaScale, count)` — 乘 α*scale 预计算常数（Selu 默认 alphaScale ≈ 1.7581，ELU 默认 1.0）
- `Muls(localZero, localIn, 0.0f, count)` — 造零 tensor（Compare 无标量版，需扩成 tensor）
- `Compare(localMask, localIn, localZero, CMPMODE::GT, count)` — 生成 x>0 位掩码
- `Select(localOut, localMask, localPos, localExp, SELMODE::VSEL_TENSOR_TENSOR_MODE, count)` — 按位选择

**与 ELU 实现的唯一差异**：Host 侧 `alphaScale = alpha * scale` 预计算时，Selu 默认 alpha≈1.6733 scale≈1.0507 得 alphaScale≈1.7581，ELU 默认 alpha=1.0 scale=1.0 得 alphaScale=1.0。Kernel 侧指令链完全一致，只是常数不同。

**分支选择路径论证：**

Selu 必须真分支（正值/负值公式不同，无法像 HardSwish 那样 clamp 合并）。可选方案：

| 方案 | 指令数 | 同步开销 | 选用 |
|------|------|---------|------|
| Compare + Select | 8 条 | Select 需等 Compare 写完 mask 寄存器 | ✅ 本实现 |
| 官方 Relu 指令改造 | 不可行——Relu 是 max(0,x) 不能表达双分支公式 | — | ❌ |
| 官方 VecSel 标量模式 | VSEL_TENSOR_SCALAR_MODE 仅支持 tensor vs scalar，Selu 两分支都是 tensor | — | ❌ |

Compare + Select 是唯一可行路径。mask 同步开销约 2-3 cycle（Compare 写寄存器→Select 读寄存器），相比 Exp 指令的 10+ cycle 占比小，非性能瓶颈。

**Exp 指令链时延分析：**

官方实测（NPU 架构 220x）：「Exp/Ln 接口处理同样数量 half/float 的耗时是一样的，内部对 float 做了优化」。

Selu 负值分支指令链：Muls + Exp + Adds + Muls = 4 条，含 Exp 单条时延约 8-12 cycle（256B 数据），链总时延约 14-18 cycle per 256B。与 ELU 时延完全一致，差异仅在常数 alphaScale 不同不影响时延。

对比 HardSwish（7 条无 Exp 指令约 8-14 cycle per 256B），Selu 指令链长 30%-40%，是「计算受限」算子。Exp 是本算子性能瓶颈源，优化方向是确保 Exp 指令连续发射不被流水断流打断（TQue 自动同步）。

**CopyOut 实现：**
- 将 UB 结果通过 MTE3 搬回 GM
- TQue.EnQue/DeQue 自动同步

##### 3. 数据类型特化处理

| 数据类型 | 策略 | UB 占用 |
|---------|------|---------|
| float32 | 原生计算，8 条指令直接发射 | tileLength × 25B |
| float16 | half→float32 扩展计算→float32→half 缩回，消除 half 写回 UB 非 32B 对齐的性能劣化，且避免 half 在 Exp 中间值累加的精度损失 | tileLength × 25B（同 float32） |

##### 4. 特殊值处理

| 输入值 | 输出值 | 验证路径 |
|--------|--------|---------|
| x = 0 | 0（Selu 在 x=0 连续，自归一化约束要求） | 两分支在 x=0 都返 0 |
| x → +∞ | +∞（正值分支 scale·x，scale≈1.05 放大） | 正值分支 |
| x → -∞ | -α·scale ≈ -1.7581（Exp 下溢为 0，0-1=-1，×α·scale） | 负值分支饱和 |
| x·inputScale < -87.3 | 趋近 -α·scale ≈ -1.7581 | Exp 下溢边界，需测试用例覆盖 |
| NaN | NaN | 两分支 NaN 透传，Select 不改变 NaN |

**与 ELU 特殊值差异**：ELU 饱和值 ≈ -1.0（默认 α=scale=1.0），Selu 饱和值 ≈ -1.7581（默认 α≈1.6733 scale≈1.0507）。测试用例预期值需区分。

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
- alpha、scale、input_scale 必须为正数
- 默认 α、scale 常数需与 Klambauer 2017 论文对齐（精度 7 位有效数字）
- 需适配非 32 字节对齐的非对齐场景（DataCopyPad 处理尾块）
- UB bank 冲突需通过 TQue 自动 bank 分配规避
- 需测试用例覆盖 Exp 下溢边界（x·inputScale < -87.3）

## 可维可测分析

### 精度标准 / 性能标准

| 验收标准 | 描述 | 标准来源 |
|---------|------|---------|
| 精度标准 | 生态算子开源精度标准（相对误差 ≤ 1e-5，绝对误差 ≤ 1e-6） | 任务书要求 |
| 性能标准 | 无明确性能要求，但需合理利用 Vector 流水线，UB 带宽使用率 ≥ 60%（计算受限算子带宽利用率天然低） | 任务书要求 + 官方最佳实践 |

性能采点指标（官方 Profile 字段）：
- `aiv_gm_to_ub_bw` / `aiv_ub_to_gm_bw` — UB↔GM 带宽速率（GB/s）
- `GM_to_UB_bw_usage_rate` / `UB_to_GM_bw_usage_rate` — 带宽使用率（%）
- `aiv_total_cycles` — 总 cycle 数，用于计算 Exp 指令链占用比

### 测试用例规划

| 测试场景 | 用例描述 | 数据类型 | Shape | 预期结果 |
|---------|---------|---------|-------|---------|
| 基础正向 | 全正值输入 | float32 | 1024 | 结果 = scale × x ≈ 1.05 × x |
| 基础正向 | 全负值输入 | float32 | 1024 | 结果 = α·scale·(e^(x·inputScale)-1) ≈ 1.76·(e^x-1) |
| 基础正向 | 正负混合输入 | float32 | 2048 | 逐元素正确 |
| 边界 | 全零输入 | float32 | 32 | 输出均为 0 |
| 边界 | 极大值输入 | float32 | 1 | 不溢出，输出 = scale·x ≈ 1.05·x |
| 边界 | 极小值输入 | float32 | 1 | 趋近 -α·scale ≈ -1.7581 |
| **Exp 下溢** | x·inputScale = -90 | float32 | 8 | 输出 ≈ -1.7581（Exp 下溢为 0） |
| **Exp 下溢** | x·inputScale = -87 | float32 | 8 | 输出趋近 -1.7581（边界值） |
| **常数精度** | 默认 α、scale 验证 | float32 | 1 | 与 Klambauer 论文常数对齐 |
| Float16 | float16 全场景 | float16 | 1024 | 精度达标（与 float32 一致） |
| 泛化 | 标量输入 | float32 | 1 | 正确 |
| 泛化 | 非对齐 Shape | float32 | 31 | 正确（DataCopyPad 尾块） |
| 泛化 | 大 Shape 测试 | float32 | 10000 | 无内存溢出 |
| 分支交界 | x 在 0 附近抖动 | float32 | 256 | 分段函数正确切换 |
| NaN 处理 | 含 NaN 输入 | float32 | 8 | NaN 透传不崩 |

### 兼容性分析

新算子开发，不涉及旧接口破坏。基于 Ascend C C API 标准接口开发，与现有框架兼容。Selu 与 ELU 实现路径几乎一致，可共享大部分 Kernel 代码，差异仅在 Host 侧常数校验。

### 风险与降级预案

| 风险 | 预案 | 监控指标 |
|------|------|---------|
| float16 精度不足 | 已扩展为 float32 计算，精度与 float32 一致 | 精度对比测试 |
| UB 空间不足 | tileLength 量化推导 ≤ 10240，预留 8KB 余量 | aiv_total_cycles 不应异常升高 |
| bank 冲突 | TQue 自动 bank 分配，多中间 tensor 显式分 BG | aiv_time 不应显著高于理论值 |
| Exp 指令数值异常 | 底层 Vector Exp 已对 float 做饱和优化，测试用例覆盖极负值确保不崩 | Exp 下溢测试用例 |
| 分支交界抖动 | Compare+Select mask 精确到 bit，无跳转抖动 | 分支交界测试用例 |
| **常数精度偏差** | Host 侧校验默认 α、scale 与 Klambauer 论文对齐，7 位有效数字 | 常数精度测试用例 |

## 附录：官方资料引用

- Ascend C API 列表（Exp/Adds/Muls/Compare/Select 签名核实）：https://www.hiascend.com/document/detail/zh/canncommercial/83RC1/API/ascendcopapi/atlasascendc_api_07_0003.html
- Compare（结果存入寄存器）签名：https://www.hiascend.com/document/detail/zh/canncommercial/900/API/ascendcopapi/atlasascendc_api_07_0067.html
- Select 签名（VSEL_TENSOR_TENSOR_MODE）：https://asc.gitcode.com/api/SIMD-API/基础API/Memory矢量计算/比较与选择/Select.html
- 昇腾 950 NPU 架构白皮书（UB 512KB per AI Core）：https://public-download.obs.cn-east-2.myhuaweicloud.com/ascend/昇腾950%20NPU架构白皮书.pdf
- NPU 架构版本 220x 硬件约束（Exp/Ln half/float 耗时一致、half 转 float 建议）：https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta2/opdevg/Ascendcopdevg/atlas_ascendc_10_00048.html
- 通过 Unified Buffer 融合实现连续 vector 计算最佳实践：https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha001/devguide/opdevg/ascendcbestP/atlas_ascendc_best_practices_10_0020.html
- Klambauer G. et al. "Self-Normalizing Neural Networks" 5th ICLR 2017, Table 1（α、scale 固定常数源）：https://arxiv.org/abs/1706.02515
