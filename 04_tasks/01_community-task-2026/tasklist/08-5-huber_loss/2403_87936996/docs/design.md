# HuberLoss 算子设计文档

> 任务：CANN 社区任务 2026 · 8 月社区任务 - huber_loss 算子开发
> 算子实现提交至 `cann/ops-nn` 仓 `experimental/loss/huber_loss`；
> 本文档只描述设计，实测数据与测试记录见随验收提交的自测报告。

# 需求背景（required）

## 需求来源

CANN 社区任务 2026 · 8 月社区任务 - huber_loss 算子开发。

PyTorch 的 `aten::huber_loss` 未被 NPU 后端支持，训练时自动 fallback 到 CPU 执行，
导致训练耗时显著增长，阻塞模型迭代。需要基于 Ascend C 在昇腾 NPU 上实现功能一致的前向算子。

## 背景介绍

### PyTorch 原生实现分析

参考实现：`aten/src/ATen/native/Loss.cpp` 的 `huber_loss` 与 `apply_loss_reduction`。

计算式（设 `e = input - target`）：

$$
loss_i =
\begin{cases}
0.5 e_i^2, & |e_i| \leq delta \\
delta \cdot (|e_i| - 0.5 \cdot delta), & |e_i| > delta
\end{cases}
$$

reduction 语义：

| reduction | 行为 |
| --- | --- |
| none(0) | 输出逐元素 loss，shape 与输入一致 |
| mean(1) | 输出全部元素 loss 的均值，0 维标量 |
| sum(2) | 输出全部元素 loss 的和，0 维标量 |

空张量边界（PyTorch 行为）：`none` 输出空张量；`sum` 输出 `0`；`mean` 输出 `NaN`（0/0）。

### 昇腾侧现状分析

`ops-nn` 的 `experimental/loss/huber_loss` 已有一个合入版本（PR `!7904`），当前能力：

| 项 | 已合入版本 | 本任务要求 |
| --- | --- | --- |
| 参数名 | `predictions` / `targets` / `loss` | `input` / `target` / `output` |
| reduction | **不支持**（仅逐元素） | 必须支持 none / mean / sum |
| 输出 shape | 恒同输入 | mean/sum 为 0 维标量 |
| 非连续 Tensor | 仅处理连续存储 | 必须支持 |
| dtype | FLOAT32 / FLOAT16 / BFLOAT16 | 同 |

因此本任务的核心增量是：**reduction 属性与跨核归约**、**非连续 Tensor 支持**、**参数名对齐任务书**。

# 需求分析（required）

## 需求描述

使用 Ascend C 实现 HuberLoss 前向算子，与 PyTorch `aten::huber_loss` 前向功能完全对齐，
支持 FLOAT32 / FLOAT16 / BFLOAT16、ND 格式、任意维度（各维 ≥ 0）、非连续 Tensor，
支持 `reduction ∈ {0,1,2}` 与 `delta > 0`，并满足泛化输入场景。

## 需求拆解

1. **计算正确性**：分段公式与边界（`|e| = delta` 处两式相等）严格对齐 PyTorch。
2. **reduction**：新增 int 属性，默认 1；mean/sum 输出 0 维标量，需要跨核归约。
3. **数据类型**：FLOAT16 / BFLOAT16 在 Kernel 内升 FP32 计算后转回输出 dtype，避免精度损失。
4. **非连续 Tensor**：由框架自动连续化，算子侧按 storage shape 计算。
5. **空张量**：`none` 空输出、`sum` 为 0、`mean` 为 NaN，与 PyTorch 一致。
6. **精度**：满足 AscendOpTest 工具默认阈值，且与 CPU `aten::huber_loss` 对齐。
7. **性能**：达到 80% 的 compute bound 或 memory bound。
8. **参数名**：`input` / `target` / `output`，属性 `delta`（默认 1.0）、`reduction`（默认 1）。

# 详细设计（required）

## 算子分析

### 数学公式

设 `e = input - target`：

$$
loss_i =
\begin{cases}
0.5 e_i^2, & |e_i| \leq delta \\
delta \cdot (|e_i| - 0.5 \cdot delta), & |e_i| > delta
\end{cases}
$$

`reduction=mean` 时 `output = (Σ loss_i) / N`；`reduction=sum` 时 `output = Σ loss_i`。

**公式改写（实现关键）**：直接实现分段函数需要 mask/select 与额外缓冲。令 `m = min(|e|, delta)`，则

$$
loss = 0.5 m^2 + delta \cdot (|e| - m)
$$

- `|e| ≤ delta` 时 `m = |e|`，得 `0.5e²` ✓
- `|e| > delta` 时 `m = delta`，得 `0.5delta² + delta|e| - delta² = delta(|e| - 0.5delta)` ✓
- `|e| = delta` 时两式相等，边界无歧义 ✓

该改写用 `Mins` 替代 mask/select，节省约 4 个 UB 缓冲，且无分支。
改写式与原分段式在数学上恒等，量化差异需通过 ULP 审计验证控制在阈值内。

### 与 PyTorch CPU 的精度语义对齐

任务书要求「与 CPU `aten::huber_loss` 结果对齐」，且 AscendOpTest 的 bfloat16 阈值为 4e-3，
而 bf16 的 1 ulp 相对误差为 `2^-8 ~ 2^-7`（3.9e-3 ~ 7.8e-3），**差 1 ulp 即可能超差**。
因此逐元素路径必须精确复刻 PyTorch 的运算序列，而不能只保证数学等价。

PyTorch CPU 的低精度运算序列通过逐 bit 假设检验确定
（`verify/remote_src/probe_torch_bf16_path.py`）：**float16 全程 float32 计算、仅最终转换一次；
bfloat16 则会把差值量化回 bfloat16 后再继续计算**。据此本算子分三层对齐：

1. **逐元素计算精度**：fp16 全程 FP32；**bf16 在 `Sub` 后把 `e` 量化回 bfloat16**
   （`kQuantizeDiffToDtype`，一次 `Cast(CAST_RINT)` + `Cast(CAST_NONE)` 往返，
   复用已消费完的 `inputLocal` 作中转，不额外占 UB）
2. **reduction 的输入**：PyTorch 的 huber_loss ≡ `apply_loss_reduction(elementwise_loss, reduction)`，
   而 elementwise_loss 是与输入同 dtype 的张量，故归约作用在**已量化**的逐元素结果上。
   典型可判别场景：fp16、`delta=10`、`input=[65504,1,1,1]`、`target=0` 时 CPU 给出 `mean → inf`，
   而归约全程保持 FP32 会得到有限值，与 CPU 不一致。
   故 `kQuantizeElemBeforeReduce = true`，归约前对逐元素 loss 做一次 dtype 往返。
3. **累加精度**：对已量化的值在 FP32 域累加，配 Kahan 补偿。

代价：第 2 项在 fp16/bf16 归约路径上增加两条全长 Cast 指令，会降低该路径的有效带宽。
任务书中精度为硬性条件、性能「不满足要求的场景需要特别说明」，故取精度优先。

### 参数与规格

| 参数名 | 输入/输出/属性 | 数据类型 | 格式 | shape | 非连续 |
| --- | --- | --- | --- | --- | --- |
| input | 输入 | FLOAT32 / FLOAT16 / BFLOAT16 | ND | 任意维度，各维 ≥ 0 | 支持 |
| target | 输入 | 同 input | ND | 与 input 一致 | 支持 |
| delta | 可选属性 | FLOAT | - | 默认 1.0，必须 > 0 | - |
| reduction | 可选属性 | INT | - | 默认 1，取值 0/1/2 | - |
| output | 输出 | 与输入一致 | ND | none 同输入；mean/sum 为 0 维标量 | - |

## 算子实现

### host 侧设计

#### 1. 校验与 InferShape

- 校验 `input`/`target` 的 storage shape 与 dtype 完全一致（不支持广播）。
- 校验 `reduction ∈ {0,1,2}`、`delta > 0`，否则返回 `GRAPH_FAILED`。
- InferShape：`reduction=0` 时 `output = input`；否则 `output` 置为 0 维标量（`SetDimNum(0)`）。
- 非连续支持：`op_def` 中三个 Tensor 均声明 `AutoContiguous()`，tiling 使用 `GetStorageShape()`。

#### 2. tiling 策略

tiling 的纯计算部分抽取为不依赖 gert 的 `ResolveTiling`（`op_host/huber_loss_tiling_plan.h`），
真机 tiling 与主机侧仿真验证共用同一份实现，避免公式出现双实现漂移。

**分核策略（512B 对齐粒度）**

官方明确 GM→Local 搬运时 GM 地址需 512 字节对齐才能最大化带宽效率，
因此**核切分以 512B 对齐块为粒度**，而非按元素严格均分：

```text
alignElements   = 512B / elemBytes            // float32:128, float16/bfloat16:256
totalBlocks     = ceil(total / alignElements)
coreNum         = min(启发式核数, totalBlocks, 平台 AIV 核数)
blocksPerCore   = totalBlocks / coreNum
tailBlockNum    = totalBlocks % coreNum       // 前 tailBlockNum 个核各多一个对齐块
smallCoreDataNum = blocksPerCore * alignElements
bigCoreDataNum   = smallCoreDataNum + alignElements
```

这样每个核的 GM 起始偏移都是 512B 的整数倍；仅最后一个有数据的核由 kernel 侧夹取到实际元素数
（尾块不足一个对齐块时）。覆盖性约束：各核跨度之和 ≥ total 且超出量 < 一个对齐块。

**核数启发式**（元素数越大并行度越高，reduction 路径需权衡跨核归约开销）：

| 元素数 | reduction=none | reduction=mean/sum |
| --- | --- | --- |
| ≤ 128 | 1 | 1（≤1024 均为 1） |
| ≤ 1024 | 2 | 1 |
| ≤ 8192 | 8（float32 且 ≤8192 时降为 4） | 8 |
| ≤ 32768 | 12 | 16 |
| ≤ 131072 | ceil(total/4096) | 32 |
| ≤ 524288 | ceil(total/4096) | min(ceil(total/2048), 48) |
| 更大 | ceil(total/4096) | ceil(total/2048) |

各档位来自等梯度全排列调优（`verify/tune_huber_params.py`），云端 Profiling 后按实测校准。

**UB 切分（tile 求解）**

```text
queueBytesPerElement = (reduction==0) ? 6*elemBytes : 4*elemBytes   // 双缓冲，none 含输出队列
tmpFloatBytesPerElement = (float32 且 reduction==0) ? 4 : 8         // float 逐元素路径仅 1 个 float 临时
reduceFixedBytes = (reduction==0) ? 0 : (96 + 32 + coreNum*32)      // 归约三段 + 标量中转 + 跨核槽
usableUb  = ubSize - 512(安全余量) - reduceFixedBytes
tileDataNum = floor(usableUb / bytesPerElement / alignElements) * alignElements
```

`tileDataNum` 同样按 512B 对齐，使每个 tile 的搬运起始地址保持对齐；
若不足一个对齐块则 tiling 失败。192KB UB 下 tile 字节数约 19～32KB，
满足官方"单次搬运 > 16KB 可获得最优带宽"的建议。

#### 3. workspace 规划

```text
workspace[0] = sysWorkspaceSize(GetLibApiWorkSpaceSize) + usrWorkspaceBytes
usrWorkspaceBytes = (reduction==0) ? 0 : coreNum * (2*32 + 8*sizeof(float))
                                                   sync region   归约槽
```

- 系统预留区**始终申请**（与 CANN 9.0.0 权威工程 celu_v2 / binary_cross_entropy_grad_v2 一致）。
- **kernel 必须经 `SetSysWorkspace` + `GetUserWorkspace` 取得用户区基址**
  （权威依据：ops-math `reduce_sum` kernel 的同一写法）：

  ```cpp
  GM_ADDR userWorkspace = nullptr;
  if (workspace != nullptr) {                       // none 路径不用 workspace，
      AscendC::SetSysWorkspace(workspace);          // 故此处不可在为空时直接 return
      userWorkspace = AscendC::GetUserWorkspace(workspace);
  }
  ```

  若直接把 `workspace` 指针当用户区，每核归约槽会写进系统预留区首部，
  覆盖 `SyncAll` 硬同步的框架标志位。真机表现为 **reduction 路径在核数由少变多时 Kernel 卡死**：
  `mean N=2048`(8 核) → `mean N=65536`(32 核) 必现；核数递减、同核数重复、`none` 路径均正常；
  对 workspace 整体 `aclrtMemset(0)` 后又恢复正常（等于把被覆盖的标志位还原成初值）。
  该缺陷曾被批量测试的 `maxN=2048` 上限掩盖，定位过程见 `verify/remote_src/sequence_probe.cpp`
  的 14 组对照实验。
- 用户区布局：`[sync region coreNum*64B][归约槽 coreNum*32B]`。
  - **sync region**：为 `SyncAll` 软同步原型预留（官方要求 gmWorkspace ≥ 核数×32B 且需初始化为 0）。
    当前实现使用**无参硬同步**，该区域未被占用，保留以便后续切换同步方式。
  - **归约槽**：每核 8 个 float（32B），从 sync region 之后开始（`workspaceFloatOffset = coreNum*64/4`）。

#### 4. 多核同步的合规配置

reduction 路径使用 **无参硬同步 `SyncAll()`**（硬件全核 barrier，不占用 workspace）：

- **无参硬同步**：官方文档指出"在分离模式下建议使用硬同步接口而非软同步接口，软同步仅适用于纯 Vector 场景且性能较低"。挂死风险由「dtype × reduction × 元素量」组合矩阵与「核数由少变多」的序列场景专项覆盖（`verify/remote_src/stall_probe.cpp`、`sequence_probe.cpp`）。
- **不声明 `KERNEL_TASK_TYPE_DEFAULT`**：与已合入 smooth_l1_loss_v2 等 reduction loss 算子一致；声明 `KERNEL_TYPE_MIX_AIV_1_0`/`AIV_ONLY` 会导致 SyncAll 死锁。
- **不调 `SetScheduleMode(1)`**：smooth_l1 等合入算子未使用；单 stream 验收场景不触发多 stream 死锁。
- **必须调 `SetSysWorkspace` + `GetUserWorkspace`**：否则会覆盖系统预留区中的硬同步标志位（见上一节）。
- **DataCopy 与 SyncAll 之间不加 PipeBarrier**：与 smooth_l1 一致（此前多加一个导致死锁）。

#### 5. tilingkey 规划

本算子 kernel 侧按 `tilingData.reduction` 运行时分支，不同 reduction 的数据流在同一实现内统一处理，
故 **tilingkey 固定为 0**，不做 key 分派。

### kernel 侧设计

整体为 `Init` + `Process` 两阶段。`Process` 按 reduction 走两条数据流：

```text
reduction = none :  CopyIn -> Compute -> CopyOut            (逐 tile 流水)
reduction = mean/sum:  CopyIn -> ComputeReduceAccumulate    (逐 tile 累加)
                       -> GlobalReduction                   (跨核汇总 + 缩放 + 写标量)
```

#### 1. Init

- 核切分**直接采用 host 侧 tiling 的结论**（单一事实来源），按 512B 对齐块计算：

  ```text
  globalBufferIndex = coreId * smallCoreDataNum + min(coreId, tailBlockNum) * alignElements
  coreSpan          = smallCoreDataNum + (coreId < tailBlockNum ? alignElements : 0)
  coreDataNum       = (globalBufferIndex >= total) ? 0 : min(coreSpan, total - globalBufferIndex)
  ```

  尾部对齐块可能整体落在总量之外，此时该核 `coreDataNum = 0`，但**仍须参与跨核同步**（否则 SyncAll 死锁）。

- `reduction != 0` 时 `outputGm` 长度置为 1，防止归约路径越界覆盖。
- UB 缓冲分配：输入/输出队列（双缓冲）、两个 float 临时缓冲、归约四段缓冲
  （累加器 / Kahan 补偿量 / 临时量 / 朴素累加器）、标量输出中转缓冲。
  单 tile 即可覆盖本核数据时收缩 tile，但**必须保持 32B 对齐**（`DataCopyPad` 的尾部补齐
  会写到对齐边界）；float 临时缓冲按 8 元素向上对齐分配。
  注意收缩后的容量**只保证 32B 对齐、不保证 64 元素对齐**，因此 tile 内归约不得依赖
  「可以补齐到 64 的倍数」这一假设（详见性能优化章节的尾部折叠方案）。

#### 2. CopyIn —— DataCopyPad 的填充语义（关键设计点）

搬入使用 `DataCopyPad` 支持非对齐长度。**填充参数必须显式指定 `rightPadding`**：

官方规定，当 `blockLen + leftPadding + rightPadding` 不满足 32B 对齐时框架会追加 dummy 补齐，
而**若 leftPadding 与 rightPadding 均为 0，dummy 取"待搬运数据块的第一个元素值"，不是 0**。
若依赖尾部为 0 做按块对齐的计算与归约，会把 `Huber(input[0] - target[0])` 多算进结果。

因此实现为：

```cpp
computeDataNum = AlignUp(processDataNum, ioElementsPerBlock)   // 按 32B 块对齐的计算长度
padElements    = computeDataNum - processDataNum
DataCopyExtParams      copyParams{1, processDataNum * sizeof(T), 0, 0, 0};
DataCopyPadExtParams<T> padParams{true, 0, padElements, 0};    // 显式右侧补 0
```

这样 `blockLen + rightPadding` 恰好 32B 对齐，落入官方"已对齐 + isPad"的确定分支，填入 `paddingValue = 0`。
收益：计算长度按块对齐后，尾部元素的 `e = 0 - 0 = 0`、`Huber(0) = 0`，
于是**无需标量补零**，省掉每 tile 的标量 `SetValue` 循环与两次 `PipeBarrier<PIPE_ALL>` 全流水冲刷；
而向量指令本身按 32B 块执行，多算这几个元素不产生额外开销。

#### 3. Compute —— 指令序列

FLOAT32 路径（零拷贝，直接在队列缓冲上计算并写入输出缓冲）：

```text
Sub (out, input, target)      // e
Abs (absDiff, out)            // |e|
Mins(out, absDiff, delta)     // m = min(|e|, delta)
Sub (absDiff, absDiff, out)   // |e| - m
Mul (out, out, out)           // m²
Muls(out, out, 0.5)           // 0.5m²
Muls(absDiff, absDiff, delta) // delta(|e| - m)
Add (out, out, absDiff)       // loss
```

FLOAT16 / BFLOAT16 路径：先 `Cast(CAST_NONE)` 将两个输入升到 FP32，
**在 FP32 域相减**（避免低精度相减的舍入损失），其余同上；
最后 `Cast(CAST_RINT)` 回写输出 dtype。

舍入模式依据（Atlas A2 Cast 支持矩阵）：`half/bfloat16_t → float` 仅支持 `CAST_NONE`；
`float → bfloat16_t` **不支持** `CAST_NONE`，故降精度统一使用 `CAST_RINT`（四舍六入五成双）。

#### 4. 归约设计

**tile 内**：二叉树级联归约到 8 个部分和（`half` 取 ≥ 一半且为 8 的倍数，
保证 dst 与 src 或完全重合或完全不重合，且偏移 32B 对齐），然后以 **Kahan 补偿求和**
累加到每核 8 槽累加器：

```text
y = val - c;  t = accum + y;  c = (t - accum) - y;  accum = t
```

Kahan 的必要性：AscendOpTest 对 FLOAT32 的容差与错误率均为 1e-4，
而归约输出只有 1 个元素（错误率无法摊薄，超差即失败）。tile 间朴素顺序累加的相对误差约
`tileNum × eps`，在 1e8 量级元素时仅剩约 3 倍余量；Kahan 将误差拉回 `eps` 量级，
代价是每 tile 增加 4 条 8 元素向量指令，相对 tile 主体可忽略。

**跨核**：各核将 8 槽局部和 `DataCopy` 到 workspace **用户区**归约槽（每核 32B，正好一个 block），
无参 `SyncAll()` 硬同步后由 core0 读回全部槽位，再做一次向量级联归约得到全局和。
DataCopy 与 SyncAll 之间不插入 PipeBarrier（与 smooth_l1_loss_v2 一致）。

**缩放与写出**：`mean` 除以 `totalDataNum`；空张量时按 PyTorch 语义返回 `NaN`
（直接构造 quiet NaN 位模式，不依赖标量单元的除零行为），`sum` 保持 0。

标量结果**不使用 `GlobalTensor::SetValue`** 写出——官方说明 `SetValue` 只更新核内 DCache，
需额外配 `DataCacheCleanAndInvalid` 才能落到 GM。实现改为写入 UB 中转缓冲后经
`DataCopyPad`（MTE3）直写 GM，从根上规避一致性问题，并与 none 路径的写出方式统一。

BFLOAT16 标量转换保留**手写 RNE**：官方标量 Cast 对 bf16 为截断、对 half 为 `CAST_ODD`，
与向量 `CAST_RINT`(RNE) 不一致；手写 RNE 可保证 none / mean / sum 三条路径舍入一致。

## 支持硬件

| 支持的芯片版本 | 是否勾选 |
| --- | --- |
| Atlas A2 训练系列产品 | √ |

`op_def` 中通过 `OpAICoreConfig` 声明 `ascend910b` 配置，开启动态 shape / 动态 rank 支持。

## 算子约束限制

- `input` 与 `target` 的 shape、dtype 必须完全一致，**不支持广播**。
- 输出 dtype 与输入一致。
- `reduction` 仅支持 0/1/2，其他值为非法输入。
- `delta` 必须为正数。
- 数据格式仅支持 ND。
- 支持非连续 Tensor，由框架自动插入连续化处理。
- 当前作为独立 loss 算子实现，不涉及图融合。

# 可维可测分析

## 精度标准 / 性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | 满足 AscendOpTest 工具默认阈值，且与 CPU `aten::huber_loss` 结果对齐。生效项为 `[tolerance, errorRate]`：float32 `[1e-4, 1e-4]`、float16 `[1e-3, 1e-3]`、bfloat16 `[4e-3, 4e-3]`；`\|expect\| ≥ 1` 看相对误差，`< 1` 看绝对误差，双 NaN 视为通过 | 任务书 + AscendOpTest `compare/compare/accuracy_config.py` |
| 性能标准 | 达到 80% 的 compute bound **或者** memory bound（"或者"表示满足其一即达标）。`memory 达成率 = 有效带宽 / 同访存模式官方算子实测带宽`；`compute 达成率 = aiv_vec_time / Task Duration`（msprof 硬件计数器）。取两者较高者判定 | 任务书 |
| 精度标准（补充） | 官方生态算子精度标准：golden 取"更高精度的实现的单一精度标杆"，要求 `MERE < Threshold` 且 `MARE < 10*Threshold`（float32 `2^-13`、float16 `2^-10`、bfloat16 `2^-7`） | `opbase` 仓 `ops_precision_standard/experimental_standard.md` |

**memory 基线标定方法**：不使用厂商标称带宽（官方开发文档未公开 Atlas A2 的 HBM 带宽数值），
也不拿被测算子自身当峰值（属循环论证）。约定用 **CANN 自带的官方 memory-bound 算子**
在同机、同访存模式、同规模、同一次会话内连续测量取带宽上界作分母：

| 访存模式 | 标定算子 | 对应本算子用例 |
|---|---|---|
| 读 2N + 写 N | `aclnnAdd` / `aclnnMul` / `aclnnMaximum` / `aclnnSub` 取最快者 | `reduction=none` |
| 读 N + 写 1（纯读） | `aclnnReduceSum` | `reduction=mean` / `sum` |
| 读 N + 写 N | `aclnnAbs` | 交叉参考 |

纯读基线必须与被测用例**同规模**（`mean`/`sum` 读两个输入张量，故取 512MB 规模的基线），
否则会因规模效应高估达成率。

> 口径说明：不采用 `aiv_mte2_time / Task Duration` 作 memory 达成率——
> `mte2_time` 是 MTE2 指令活跃时间而非数据传输时间，会给出与有效带宽矛盾的结论。

### 精度保障措施

| 措施 | 目的 |
| --- | --- |
| FLOAT16/BFLOAT16 升 FP32 计算，且在 FP32 域相减 | 避免低精度相减的舍入损失 |
| 归约内部 FP32 + Kahan 补偿 | 将累加误差从 `tileNum × eps` 降到 `eps` 量级 |
| tile 内与跨核均为二叉树级联归约 | 误差为 log 级而非线性 |
| 输出转换统一 `CAST_RINT`(RNE)，标量 bf16 手写 RNE | 保证 none/mean/sum 三条路径舍入一致 |
| 公式改写 `0.5m² + delta(\|e\| - m)` | `\|e\| = delta` 处两式恒等，边界无歧义 |

按官方生态算子精度标准（golden 取更高精度实现）评估时约定两项剔除规则：
「高精度真值低于该 dtype 最小正规数」的采样点（该处 dtype 唯一可表示的结果是 0，
公式中 `+1e-7` 保护项会主导相对误差，属表示范围限制而非计算误差）、
以及「目标精度域内溢出为 Inf」的采样点（该行为与 PyTorch 一致，单独计数）。

实测数据见自测报告，不在本设计文档中重复。

### 性能保障措施

| 措施 | 依据 |
| --- | --- |
| 核切分与 tile 长度均按 512B 对齐 | 官方：GM→Local 搬运时 GM 地址 512B 对齐才能最大化带宽效率 |
| tile 实际字节数 19～32KB | 官方：单次搬运 > 16KB 可获得最优带宽 |
| 输入/输出队列双缓冲 | 搬运与计算流水重叠 |
| FLOAT32 逐元素路径零 UB→UB 拷贝 | 直接在队列缓冲上计算并写入输出缓冲 |
| 公式改写省去 mask/select 与约 4 个 UB 缓冲 | 提高可用 tile 大小 |
| 免标量补零（显式 rightPadding） | 消除每 tile 的标量循环与两次全流水冲刷 |
| 移除计算与归约中的手工 `PipeBarrier<PIPE_V>` | 官方：算子工程默认开启自动同步，BiSheng 按依赖自动插入；手工 barrier 会阻断向量指令间及与下一 tile 搬入的流水重叠 |

**bound 归属预期**：算术强度约 0.67 FLOP/Byte，`reduction=none` 为典型 memory bound。
`reduction=mean`/`sum` 在 float32 下仍为 memory bound；在 float16/bfloat16 下因元素数翻倍、
升 FP32 域计算、且为与 CPU 逐 bit 对齐需要 dtype 往返 Cast，转为 compute bound，
按任务书「compute bound 或者 memory bound 满足其一即达标」的口径以 compute 达成率判定。

性能主判定用例统一取 ≥256MB 数据量以确保超出 L2（L2 命中时带宽会跳到 2300 GB/s 量级，
特征显著不同，此类用例仅作对照不参与判定）。实测数据见自测报告，不在本设计文档中重复。

**归约优化**：每 tile 末尾把逐元素 loss 归约成 8 个部分和，原用级联二叉树
（tile 长约 8192 需约 10 步、步间串行依赖，造成向量流水气泡）。
依官方 `ReduceSum` 文档提示（该接口为软件仿真，性能可能不及硬件指令 `BlockReduceSum`/
`WholeReduceSum`，且 `BlockReduceSum` 支持 Atlas A2），改为先用 `BlockReduceSum`
每步压缩 8 倍（8192 → 1024 → 128 → 16，3 步）再用 1 步级联 Add 收尾，依赖链降至 4 步。

实现要点：`BlockReduceSum` 每次只处理整数个 repeat（每 repeat 64 个 float），
因此每轮循环都必须显式处理不足一个 repeat 的尾部，否则 `len/64` 的整数除法会
静默丢弃它（例如 `len=1000` 时只处理 960 个），导致归约结果偏小。

尾部的处理方式是**把残余用一条 `Add` 折叠进前段**，而不是补零到 64 的倍数。
补零方案会造成 UB 越界：kernel 在单 tile 即可覆盖本核数据时会把 tile 缓冲收缩到
`AlignUp(coreDataNum, ioElementsPerBlock)`，而 `ioElementsPerBlock` 为 16（fp16/bf16）
或 8（float32），**都不保证是 64 的倍数**。例如 bf16 末核 `coreDataNum = 1000` 时
缓冲只有 1008 个 float，而补齐目标是 1024，`Duplicate` 与 `BlockReduceSum` 会各越界
16 个 float 踩到相邻 UB。折叠方案的所有访问都落在 `[0, len)` 内，从根上消除该风险：

```cpp
const uint64_t whole = (len / 64) * 64;
const uint64_t rest  = len - whole;                  // 8 的倍数，且 <= 56
if (rest > 0) { Add<float>(src, src, src[whole], rest); }
BlockReduceSum<float>(dst, src, whole / 64, 64, 1, 1, 8);
len = whole / 8;
```

合规性：`rest ≤ 56 < 64 ≤ whole`，故 `dst[0, rest)` 与 `src[whole, whole + rest)` 不重叠；
两侧偏移均为 8 的倍数，起始地址天然 32B 对齐。
`dst` 与 `src` 不可重叠，故复用本 tile 已消费完的 `absDiff` 作乒乓缓冲，不额外占 UB。

## 测试方案

| 层次 | 内容 |
| --- | --- |
| InferShape UT | 默认 reduction、同形输出、标量、空张量、shape 不一致失败场景 |
| Tiling UT | 三 dtype × 三 reduction、512B 对齐断言、workspace 精确断言、非法入参、UB 不足 |
| Kernel UT | 三 dtype、三分支边界、自定义 delta、单元素、非均匀多核、多 tile、跨核归约、512B 对齐尾块夹取、空张量 NaN/0 |
| aclnn 样例 | 两段式接口 `aclnnHuberLossGetWorkspaceSize` + `aclnnHuberLoss` |
| 精度测试 | 边界用例 + 随机泛化用例，golden 由 PyTorch 官方实现直接生成，按 AscendOpTest 默认阈值 + 官方生态精度标准双判据判定 |
| 性能测试 | ≥256MB 主判定用例（3 dtype × 3 reduction）+ L2 对照用例，按主导瓶颈判定达成率，memory 基线由官方 aclnn 算子同机标定 |
| 非有限值专项 | 「Inf/NaN × 多 tile」用例，覆盖非有限值分别落在首/中/末 tile，含 ±Inf 混合与 Inf+NaN |
| 专项诊断 | 执行序列敏感性、挂死边界矩阵、PyTorch 语义逐 bit 假设检验、公式量化差异审计 |

**golden 生成纪律**：

1. **一律由 PyTorch 官方实现在 CPU 上生成**（`verify/remote_src/gen_torch_golden.py`），
   不使用自行复刻的实现——自实现容易在 reduction 的量化时机上产生系统性偏差。
2. **必须显式固定 `torch.set_num_threads(1)`**。`at::parallel_for` 的 grain size 为 32768，
   元素数超过它会分块并行，分块改变 cascade 求和的结合顺序进而改变舍入，
   使 golden 随机器核数与负载而变、不可复现。单线程是确定性参考语义。

测试步骤与用例清单详见算子目录下 `tests/README.md`，实测结果详见自测报告。

### 设计取舍：bfloat16 归约的可对齐边界

归约路径的累加顺序无法与 PyTorch 逐 bit 复刻，需要在设计阶段明确其可对齐边界：

1. **PyTorch CPU 的 `sum`/`mean` 结果本身依赖 `torch.set_num_threads()`**。
   `at::parallel_for` 的 grain size 为 32768，元素数超过它才分块并行，
   分块改变 cascade 求和的结合顺序进而改变舍入；差异只出现在单线程与多线程之间。
   因此判定基准必须固定为单线程（确定性参考语义），详见上文 golden 生成纪律。
2. **bfloat16 的判定阈值小于其 1 个 ulp**。AscendOpTest 的 bfloat16 阈值为 `4e-3`，
   而 bf16 最坏情况 1 ulp 相对误差为 `2^-7 = 7.8e-3 > 4e-3`；又因归约输出只有 1 个元素，
   判定式 `invalid > total × errorRate` 中 `total = 1`，**错误率无法摊薄，差 1 ulp 即判失败**。
   当精确和落在两个 bf16 网格点的中点附近时，任何 1e-7 级的实现差异都会使结果翻到相邻格点。
   float32 / float16 不受此约束（其 1 ulp 相对误差远小于对应阈值）。
3. **PyTorch 对 bfloat16 归约使用 bf16 累加器 + 向量化 cascade**（逐 bit 假设检验确认
   `huber_loss(reduction='sum')` ≡ `torch.sum(elementwise_loss)`，而 `torch.sum` 对 bf16
   张量每级向量加法都量化回 bf16）。

**设计选择：采用 FP32 高精度累加（硬件 `BlockReduceSum` pairwise 归约）+ Kahan 补偿，
不模仿 PyTorch 的 bf16 逐级量化累加。** 理由：其 cascade 结构依赖向量宽度与线程数，
不可靠复现；且刻意降低累加精度会偏离精确值、劣化官方生态算子精度标准（golden 取更高精度实现）的余量。
该选择的代价是 bfloat16 归约在极少数「精确和落在 bf16 网格中点附近」的用例上可能出现
1 ulp 分歧，实测分歧率与逐例证据见自测报告，不在本设计文档中展开。

## 兼容性分析

已有合入版本（PR `!7904`）的参数名为 `predictions`/`targets`/`loss` 且不支持 reduction，
本次按任务书要求改为 `input`/`target`/`output` 并新增 `reduction` 属性（默认 1）。

- `reduction` 为**可选属性且默认值为 1（mean）**，与 PyTorch 默认行为一致。
- 参数名变更会影响已有调用方的 GE 图构造与 aclnn 接口签名，属于按任务书要求的接口对齐，
  在 `experimental` 目录下演进，不涉及正式接口的兼容性承诺。
- 新增能力（reduction、非连续）为纯增量，`reduction=0` 时行为与原逐元素版本一致。
