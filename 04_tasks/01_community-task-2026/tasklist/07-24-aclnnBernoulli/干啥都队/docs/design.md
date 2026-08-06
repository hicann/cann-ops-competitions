# aclnnBernoulli 算子设计文档

> **文档类型**：算子设计文档（design.md）
> **任务来源**：2026 年 7 月社区任务 aclnnBernoulli
> **算子类别**：Math / 随机采样 / 内存优化
> **版本**：v1.2  |  **日期**：2026-08-04

---

## 一、需求背景

### 1.1 需求来源

本任务面向 `ops-math/random/stateless_bernoulli` 中已有的 `aclnnBernoulli` 实现，目标是解决 `Tensor.bernoulli_` 对应算子的内存一致性问题。任务书明确要求：在不影响原有功能和语义的前提下，消除小算子拼接产生的冗余中间输出，使 Ascend 侧内存占用与 GPU 的差距低于 5%。

任务基本信息如下：

| 项目 | 内容 |
|------|------|
| 任务书 | [aclnnBernoulli_task_doc.md](https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/docs/202607/aclnnBernoulli_task_doc.md) |
| 开源仓地址 | [ops-math/random/stateless_bernoulli](https://gitcode.com/cann/ops-math/tree/master/random/stateless_bernoulli) |
| CANN 版本 | CANN 8.5.0 及以上 |
| 适配硬件 | Atlas A2/A3 训练系列产品 |
| 开发语言 | Ascend C |

### 1.2 背景介绍

#### 1.2.1 算子功能

对输入张量的每个位置按概率 `prob` 进行伯努利采样：

$$
out_i \sim Bernoulli(prob), \qquad P(out_i = 1) = prob
$$

公开接口保持任务书定义不变：`self` 与 `out` 的 shape、dtype 一致，`prob` 为 Host 侧 `aclScalar`。输出元素只能为 0 或 1，并支持任务书列出的数据类型、ND 格式、0 到 8 维以及非连续 Tensor。

本设计不新增 `aclnnBernoulliTensor`、`aclnnInplaceBernoulli` 等公开接口，也不扩展 tensor-prob 或 broadcast 语义。任务中的 inplace 指内部临时缓冲区复用，不等同于新增公开 API。

#### 1.2.2 现状与问题

当前 `aclnnBernoulli` 采用小算子拼接实现，内部包含 `genmask(DSA) + fill + DropoutDoMask` 等步骤。现有实现中 `Fill`、`DropoutDoMask`、`Mask` 等步骤会产生中间临时输出，导致同一份数据在多个全量缓冲区中同时存活。

任务书给出的现状数据为：

| 场景 | 当前内存膨胀情况 | 主要原因 |
|------|------------------|----------|
| BFLOAT16 / FLOAT32 | 约 50% | 全量中间输出与目标输出同时存在，并可能引入 FP32 临时量 |
| FLOAT16 / INT64 | 约 25% | 拼接路径保留了额外的中间结果 |
| GPU 参考实现 | 单个 Kernel | 不保留同等规模的拼接中间输出 |

该任务的根因不是缺少 Bernoulli API，而是已有实现的中间缓冲区生命周期过长。因此设计重点是优化现有调用链和缓冲区所有权，而不是重新实现随机数算法或扩展算子接口。

#### 1.2.3 任务目标

1. 将 `fill + DropoutDoMask` 融合为 inplace 缓冲区复用，消除不必要的全量中间输出。
2. BFLOAT16/FLOAT32 场景最多保留任务书允许的 1 份 FP32 量级临时空间，并将总内存差距控制在 5% 以下。
3. 计算结果与 `torch.bernoulli`、`Tensor.bernoulli_` 语义一致，不改变原有接口、支持范围和随机采样行为。
4. 优化后的性能不低于原算子。

---

## 二、需求分析

### 2.1 算子参数与支持范围

以下范围直接来自任务书，设计和测试不得擅自扩展为 tensor-prob、broadcast 或其他公开接口。

| 参数名 | 输入/输出 | 类型或数据类型 | 数据格式 | Shape | 使用说明 |
|--------|-----------|----------------|----------|-------|----------|
| `self` (`aclTensor*`) | 输入 | FLOAT16、FLOAT、DOUBLE、UINT8、INT8、INT16、INT32、INT64、BOOL、BFLOAT16 | ND | 0-8 维 | 支持空 Tensor；dtype 和 shape 需与 `out` 一致；支持非连续 Tensor |
| `prob` (`aclScalar*`) | 输入 | FLOAT16、FLOAT、DOUBLE、BFLOAT16 | - | - | 满足 `0 <= prob <= 1` |
| `seed` (`int64_t`) | 输入 | INT64 | - | - | 设置随机数生成器种子 |
| `offset` (`int64_t`) | 输入 | INT64 | - | - | 必须满足 `offset % 4 == 0`，否则调用失败 |
| `out` (`aclTensor*`) | 输出 | FLOAT16、FLOAT、DOUBLE、UINT8、INT8、INT16、INT32、INT64、BOOL、BFLOAT16 | ND | 0-8 维 | 支持空 Tensor；dtype 和 shape 需与 `self` 一致；支持非连续 Tensor |
| `workspaceSize` (`uint64_t*`) | 输出 | - | - | - | 返回 Device 侧所需 workspace 大小 |
| `executor` (`aclOpExecutor**`) | 输出 | - | - | - | 返回算子执行器 |

任务书的“算子约束限制”项为“无”。上表中的概率值域、self/out 一致性、空 Tensor、非连续 Tensor 和 offset 对齐属于接口参数说明及验收范围，不在此基础上增加新的公开约束。

### 2.2 需求拆解

1. 保持现有 `aclnnBernoulli` 两段式接口及参数检查行为，不新增公共 API。
2. 定位现有 `op_api` 或算子 Kernel 中 `genmask`、`Fill`、`DropoutDoMask` 的调用关系和每个中间张量的生命周期。
3. 让 `Fill` 与 `DropoutDoMask` 共享同一个可写缓冲区，或者直接调用对应的 inplace 形式，避免同时保留 Fill 输出和 DropoutDoMask 输出。
4. 对 BFLOAT16/FLOAT32 仍需要的 FP32 临时量进行单份复用，并按 tile 而非完整输出分配临时空间。
5. 对全部任务书 dtype、空 Tensor、非连续 Tensor、offset 对齐和边界概率进行正确性验证。
6. 提供原实现、优化实现和 GPU 参考之间的内存、精度和性能对比结果。

### 2.3 修改边界

| 模块 | 本任务处理方式 |
|------|----------------|
| 现有 `aclnnBernoulli` 的 `op_api` 调用链 | 若小算子拼接发生在此处，在此处调整输出复用和 workspace 绑定 |
| `random/stateless_bernoulli` Kernel | 若当前实现通过 Kernel 生成中间输出，则改为单 Kernel 或 tile 级临时缓冲复用 |
| 公开接口、算子原型、参数语义 | 保持不变，不增加 tensor-prob、broadcast 或 Inplace 公共接口 |
| 其他随机算子和 `aclnnInplaceUniform` | 不修改，不把本任务扩展为替代或重构其他算子 |
| 测试和文档 | 增加内存峰值、精度、性能对比及完整自测交付件 |

---

## 三、详细设计

### 3.1 总体方案

本任务最终采用单 Kernel 方案：保留现有随机掩码生成逻辑，在 Kernel 内完成 `Fill` 与 `DropoutDoMask` 的等价计算，并通过 tile 级临时缓冲区复用降低峰值内存。该方案不新增公共 API，代码和测试报告均以单 Kernel 路径为准。

主方案的核心原则如下：

- 保留现有随机掩码生成路径以及 `seed`、`offset` 的处理方式，避免因内存优化改变采样语义。
- `Fill` 的输出不再作为独立的全量 Tensor 长期存活；它应作为 `DropoutDoMask` 的输入或目标缓冲区被直接复用。
- `DropoutDoMask` 结果写回同一缓冲区，或者直接写入最终 `out`；禁止同时保留 Fill 全量输出和 DropoutDoMask 全量输出。
- 所有临时数据优先按 tile 放入 UB/workspace。只有 CANN API 或数据类型转换确实需要时，才保留一份 FP32 全量临时空间。
- 公开 `aclnnBernoulli` 的 `self`、`prob`、`seed`、`offset`、`out` 参数和返回语义不改变。

### 3.2 数据流设计

#### 3.2.1 优化前的逻辑

```text
self/prob/seed/offset
          |
          v
   genmask(DSA) ----> mask 临时结果
          |
          v
       Fill --------> Fill 全量临时输出
          |
          v
   DropoutDoMask ----> DropoutDoMask 全量输出
          |
          v
         out
```

当下游算子不能复用上游输出时，Fill 和 DropoutDoMask 的结果会重叠存活，形成任务书所述的内存膨胀。

#### 3.2.2 优化后的逻辑

```text
 self/prob/seed/offset
           |
           v
    genmask(DSA) ----> mask（沿用现有生命周期）
           |
           v
  Fill / Fill-inplace --> buffer A
           |
           v
  DropoutDoMask-inplace 复用 buffer A
           |
           v
          out
```

单 Kernel 在 Kernel 内按 tile 生成/读取掩码、完成 Fill 与 DropoutDoMask 的等价计算，并直接写出 `out`，必须满足同样的 buffer 生命周期和内存验收标准。

#### 3.2.3 DSA packed mask 与连续输出同址复用

`genmask(DSA)` 输出的是 packed mask，每个元素只占 1 bit，逻辑所需空间为：

```text
packedMaskBytes = ceil(totalElements / 8)
```

对于连续 Tensor，Host 侧在 Tiling 阶段判断 packed mask 的地址、容量和输出地址是否满足同址复用条件。满足条件时，将 packed mask 所在的临时缓冲区直接作为 0/1 输出的展开目标，避免另外申请一份同规模的 Fill 输出；用户传入的 `out` 仍按原有 API 语义接收最终结果。packed mask 的 bit 顺序、实际对齐大小、DSA 输出地址以及 alias 能力必须以目标分支的真实实现和 CANN API 为准，本文公式仅表示逻辑大小。

同址复用由 executor/workspace 统一规划，仅用于内部临时缓冲区；不改变用户 `out` 的地址语义，也不要求用户侧 Tensor 直接 alias。

同址复用必须同时满足以下条件：

1. DSA 写入已经完成，后续展开阶段不会与 DSA 发生并发写冲突。
2. packed mask 在展开过程中没有其他消费者，且临时缓冲区容量覆盖展开后的输出范围。
3. 输出地址、长度和 32B 对齐满足目标 CANN 接口及 Vector 指令要求。
4. 输出 dtype 的写入宽度、尾部元素和 `out` 的实际地址映射均已在 Host 侧计算并传入 TilingData。

展开逻辑保持原有 Bernoulli 语义：

```text
bit_i = packedMask[i / 8] >> (i % 8) & 1
out_i = bit_i ? 1 : 0
```

同址复用只改变中间缓冲区的生命周期，不改变 `seed`、`offset`、mask 生成顺序或最终输出地址。

#### 3.2.4 高地址到低地址 wave 原地展开

packed mask 展开后，单位元素空间会从 1 bit 扩展为至少 1 byte，可能还需要更宽的 dtype。若展开目标与 packed mask 使用同一基地址，低地址到高地址写入会覆盖尚未读取的 packed mask。因此连续 Tensor 采用高地址到低地址的 wave 处理顺序：

1. Host 根据 UB 容量将连续输出划分为若干 wave，并把每个 wave 的起止元素和 packed mask 偏移写入 TilingData。
2. Kernel 等待 DSA mask 完成后，从当前 wave 的最高元素向最低元素逆序读取 bit 并写入展开后的输出地址。
3. 当前 wave 完成后再处理下一个低地址 wave；尾 wave 使用有效元素数限制写入范围。
4. 若 stride 造成输出地址不单调、地址范围不满足安全复用条件，直接切换到非连续 Tensor 的临时 tile 路径，不使用同址展开。

高到低的顺序保证写入地址不会提前覆盖低地址区域中尚未消费的 packed mask。Kernel 与 DSA 之间使用现有执行器的依赖或同步机制，不能依赖隐式执行顺序。

### 3.3 缓冲区与内存优化

设输出元素个数为 `N`，输出元素字节数为 `sizeof(out)`，定义：

```text
M_peak = 常驻输入/输出内存
       + 同时存活的全量临时缓冲区
       + workspace
       + 运行时分配器统计的峰值保留内存

memory_gap = (M_ascend - M_gpu) / M_gpu * 100%
```

具体平台的 UB 大小、workspace 对齐和 double buffer 数量以 CANN 8.5 及以上目标环境的运行时查询和现有 Kernel 框架为准，不在设计文档中硬编码 910B 的单一规格。

| 缓冲区 | 优化前 | 优化后 |
|--------|--------|--------|
| `mask` | 沿用现有实现 | 沿用现有实现，确认不会额外复制 |
| `Fill` 全量输出 | 与后续结果同时存活 | 与 `DropoutDoMask` 共享或被直接覆盖 |
| `DropoutDoMask` 全量输出 | 独立分配 | 复用 Fill 缓冲区或直接写 `out` |
| FP32 临时量 | 可能与上述全量输出重叠 | BFLOAT16/FLOAT32 仅保留任务允许的一份，并优先 tile 化 |
| workspace | 由原执行器申请 | 不新增无必要的全量 workspace；记录优化前后实际峰值 |

任务书给出的验收基线和目标如下，最终数值必须用实测数据填写，不能用理论估算替代：

| dtype | 优化前相对 GPU 的基线 | 优化目标 |
|-------|------------------------|----------|
| BFLOAT16 / FLOAT32 | 约 50% 膨胀 | 总差距小于 5%，保留的 FP32 临时量不超过 1 份量级 |
| FLOAT16 / INT64 | 约 25% 膨胀 | 总差距小于 5% |
| 其他任务书 dtype | 以原实现实测为准 | 总差距小于 5% |

### 3.3.1 Host 与 Tiling 设计

Host 侧不重新计算随机数，而是负责识别数据布局、选择安全路径和规划内存：

1. 获取 `self/out` 的 shape、元素个数、dtype、元素字节数、连续性和 stride 信息。
2. 计算 `packedMaskBytes`、wave 数量、每个 wave 的元素范围以及输出地址范围。
3. 根据布局和规模选择 TilingKey：

| TilingKey | 适用场景 | 处理方式 |
|-----------|----------|----------|
| `PACKED_REUSE` | 连续 Tensor、规模达到 wave 阈值、地址可安全复用 | packed mask 同址复用，高地址到低地址展开 |
| `STRIDED_TILE` | 非连续 Tensor或地址不满足单调复用条件 | packed mask 和输出在 UB 中按 tile 搬运，再按 stride 写出 |
| `TINY` | 极小 Tensor | 单 wave/单 block 路径，减少 DSA、同步和 Tiling 固定开销 |
| `FP64_FALLBACK` | FP64 不适合当前 Vector 展开路径 | 使用 tile 级 FP64 转换或原有安全 fallback，保证结果正确 |

4. 依据目标平台实际可用 UB 和对齐要求计算 wave 大小，保证任一路径都不会发生 workspace 或 UB 越界。
5. 将路径、wave 起始地址、wave 元素数、packed mask 偏移、输出 stride、元素字节数和 fallback 标志写入 TilingData。

推荐的 TilingData 逻辑字段如下，实际字段名以代码为准：

```text
totalElements       // 输出总元素数
packedMaskBytes     // packed mask 总字节数
waveElements        // 单个 wave 的元素数
waveCount           // wave 数量
elementBytes        // out dtype 元素字节数
isContiguous        // 是否满足连续地址复用
tilingKey           // PACKED_REUSE / STRIDED_TILE / TINY / FP64_FALLBACK
outStride[]         // 非连续 Tensor 的 stride 信息
```

### 3.3.2 Kernel 与 UB 预算设计

Kernel 统一分为 DSA mask 完成、路径选择、按 tile 展开/写出三个阶段：

```text
DSA packed mask 完成
          |
          v
等待依赖并读取 TilingData
          |
  +-------+------------------+
  |                          |
连续 PACKED_REUSE       非连续 STRIDED_TILE
高地址到低地址展开       UB tile 解包并按 stride 写出
  |                          |
  +-------------+------------+
                |
                v
              out
```

UB 预算按当前 wave 计算，不使用固定的全量临时 Tensor：

```text
maskTileBytes   = alignUp(ceil(waveElements / 8), 32)
outTileBytes    = alignUp(waveElements * elementBytes, 32)
fp64TileBytes   = only for FP64 fallback, allocated by tile
ubRequired      = maskTileBytes + outTileBytes + fp64TileBytes
                  + alignment/reserved bytes
```

开启 double buffer 时，`ubRequired` 需要乘以实际启用的 buffer 数量，并与平台查询到的可用 UB 比较。连续路径优先复用 packed mask 缓冲区，非连续路径只在 UB 中保留当前 tile；FP64 fallback 不能再额外申请一份完整输出。

### 3.3.3 非连续 Tensor、极小 Tensor 与 FP64 fallback

| 场景 | 路径 | 设计要求 |
|------|------|----------|
| 连续 Tensor | `PACKED_REUSE` | 使用高到低 wave 原地展开，充分复用 packed mask 地址 |
| 非连续 Tensor | `STRIDED_TILE` | 不强行做全局同址复用；在 UB 中展开当前 tile，按照 stride 写入 `out` |
| 极小 Tensor | `TINY` | 合并为单个小 wave，避免多核调度和 DSA 固定开销；空 Tensor 直接返回 |
| FP64 | `FP64_FALLBACK` | 使用可用的 FP64 Vector/标量转换路径，必要时回退到原有安全实现；结果和边界行为优先于吞吐 |

fallback 路径必须在测试中单独统计 workspace、峰值内存和延迟。若 fallback 无法达到 5% 内存差距，应在提交前继续优化其临时缓冲区，而不能仅以连续 FLOAT/BFLOAT16 路径的结果代表全部 dtype。

### 3.4 API 和参数校验

本任务不重新定义 API。实现应复用现有 `aclnnBernoulliGetWorkspaceSize` 和 `aclnnBernoulli` 的签名、错误处理和执行器生命周期。需要确认并覆盖以下行为：

1. `self`、`prob`、`out`、`workspaceSize`、`executor` 的空指针检查沿用现有接口约定。
2. `prob` 支持 FLOAT16、FLOAT、DOUBLE、BFLOAT16，并满足 `0 <= prob <= 1`。
3. `self` 与 `out` 的 dtype 和 shape 一致，支持任务书列出的 10 种 dtype。
4. `self` 和 `out` 使用 ND 格式，支持非连续 Tensor 和空 Tensor。
5. `offset % 4 == 0` 时才允许执行；不满足时调用失败。
6. 不增加 tensor-prob、broadcast、额外 rank 限制或未在任务书中定义的错误码语义。

### 3.5 Kernel/算子调用链设计

#### 3.5.1 单 Kernel 实现

使用一个 Bernoulli Kernel 完成：

```text
读取 seed/offset 和 prob
        |
按 tile 生成与原实现一致的随机掩码
        |
在 UB 中完成 Fill 与 DropoutDoMask 的等价计算
        |
将 0/1 结果按 out dtype 写入 out
```

单 Kernel 只作为解决 buffer alias 限制的实现手段，不应引入新的公共 API。Kernel 需要复用仓库现有的随机数实现和 tiling 基础设施，避免在本任务中重新设计 Philox、broadcast 或新的 stateless 接口。

### 3.6 数据类型和边界处理

输出只包含 0 和 1。对 FLOAT16、FLOAT、DOUBLE、BFLOAT16 以及整数/BOOL 输出，转换应确保 0 和 1 可精确表示。`prob` 为 BFLOAT16 时，按现有算子约定完成转换和比较，不得因优化路径丢失该类型。

| 场景 | 处理要求 |
|------|----------|
| `prob = 0` | 输出全为 0 |
| `prob = 1` | 输出全为 1 |
| 空 Tensor | 返回空输出，不访问越界地址 |
| 非连续 Tensor | 按原有 stride/布局写入，不能因 buffer 复用破坏地址映射 |
| 最后一个不完整 tile | 只处理有效元素，不能覆盖尾部无效空间 |
| `offset % 4 != 0` | 调用失败，不能静默修正 offset |

### 3.7 支持硬件与兼容性

| 项目 | 设计要求 |
|------|----------|
| CANN | 8.5.0 及以上 |
| 主要硬件 | Atlas A2/A3 训练系列产品 |
| API 兼容性 | 保持现有 `aclnnBernoulli` API，不新增公开接口 |
| 现有功能 | 不修改其他随机算子和 `aclnnInplaceUniform` |
| 平台差异 | 通过现有平台查询、编译配置和 Kernel 框架处理，不以单一 UB 常量替代实测 |

---

## 四、可维可测分析

### 4.1 内存一致性标准

内存一致性是本任务的核心验收项。必须在相同输入、相同 shape、相同 dtype、相同 `prob`、相同 `seed/offset` 和相同预热条件下比较：

1. 原始 Ascend 实现的峰值内存。
2. 优化后 Ascend 实现的峰值内存。
3. GPU 参考实现的峰值内存。

每个用例至少记录 `peak allocated`、`peak reserved`、workspace 大小和测量环境。建议测试流程为：清理缓存、预热若干次、重置峰值统计、执行固定次数、记录峰值，并在报告中注明统计 API 和测量单位。

报告使用以下公式计算差距：

```text
memory_gap_percent = abs(peak_ascend - peak_gpu) / peak_gpu * 100%
```

验收要求：所有任务书覆盖场景的总内存差距低于 5%；BFLOAT16/FLOAT32 需要额外说明剩余 FP32 临时量的大小和生命周期。

#### 4.1.1 A2/A3 验证结果记录

以下表格用于提交前填入真实测量结果。原实现、优化实现和 GPU 参考必须使用相同 shape、dtype、prob、seed/offset、预热次数和同步方式；不能用估算值代替实测值。

| 平台 | 路径 | dtype | shape | 原实现峰值 | 优化后峰值 | GPU 峰值 | 内存差距 | 延迟对比 | 结论 |
|------|------|-------|-------|------------|------------|----------|----------|----------|------|
| Atlas A2 | PACKED_REUSE | FLOAT16/BFLOAT16/FLOAT32 | 连续大 Tensor | 待实测 | 待实测 | 待实测 | 待计算 | 待实测 | 待填写 |
| Atlas A2 | STRIDED_TILE | 任务书覆盖 dtype | 非连续 Tensor | 待实测 | 待实测 | 待实测 | 待计算 | 待实测 | 待填写 |
| Atlas A2 | TINY/FP64_FALLBACK | 极小 Tensor/DOUBLE | 小 Tensor | 待实测 | 待实测 | 待实测 | 待计算 | 待实测 | 待填写 |
| Atlas A3 | PACKED_REUSE | FLOAT16/BFLOAT16/FLOAT32 | 连续大 Tensor | 待实测 | 待实测 | 待实测 | 待计算 | 待实测 | 待填写 |
| Atlas A3 | STRIDED_TILE | 任务书覆盖 dtype | 非连续 Tensor | 待实测 | 待实测 | 待实测 | 待计算 | 待实测 | 待填写 |
| Atlas A3 | TINY/FP64_FALLBACK | 极小 Tensor/DOUBLE | 小 Tensor | 待实测 | 待实测 | 待实测 | 待计算 | 待实测 | 待填写 |

### 4.2 精度和语义标准

| 验收项 | 测试方法 |
|--------|----------|
| AscendOpTest 默认阈值 | 使用任务书指定的默认阈值完成原实现与优化实现的结果比对 |
| `prob = 0/1` | 分别验证所有输出 dtype 的全 0 和全 1 |
| 输出值域 | 验证结果只包含 0 和 1 |
| 固定随机状态 | 固定 `seed` 和满足约束的 `offset`，对比原实现、优化实现的结果；若底层序列不是公开契约，则至少验证分布和边界行为一致 |
| PyTorch 语义 | 与 `torch.bernoulli`/`Tensor.bernoulli_` 在相同概率和 dtype 下进行统计分布及输出值域比对 |
| BFLOAT16 | 覆盖 `prob` 为 BFLOAT16 及 self/out 为 BFLOAT16 的组合 |

随机采样不以逐元素完全复现 GPU 的随机序列作为唯一标准；测试报告应说明随机种子策略、逐元素比对范围和统计检验方法。

### 4.3 性能标准

优化后性能不得低于原算子。对每个主要 dtype 和代表性 shape，使用相同的预热次数、重复次数和同步方式，记录原实现与优化实现的：

- kernel/端到端延迟中位数；
- 必要时的 P90 延迟；
- workspace 和峰值内存；
- 是否出现额外的同步或临时分配。

内存优化不能通过引入明显额外同步来换取结果。若 inplace 融合带来性能风险，必须在报告中给出量化对比，并以“不低于原算子”为最终门槛。

### 4.4 测试覆盖矩阵

| 测试维度 | 覆盖内容 |
|----------|----------|
| self/out dtype | FLOAT16、FLOAT、DOUBLE、UINT8、INT8、INT16、INT32、INT64、BOOL、BFLOAT16 |
| prob dtype | FLOAT16、FLOAT、DOUBLE、BFLOAT16 |
| 概率边界 | 0、1、0.5 及接近 0/1 的值 |
| shape | 标量、1-8 维、空 Tensor、小尺寸和大尺寸 |
| layout | ND 连续 Tensor 与任务书要求的非连续 Tensor |
| 随机参数 | 合法的 `seed` 和 `offset`，以及 `offset % 4 != 0` 的失败用例 |
| 结果正确性 | 原实现、优化实现、PyTorch 参考的精度和分布对比 |
| 内存 | 原实现 vs 优化实现 vs GPU，覆盖 BFLOAT16/FLOAT32、FLOAT16/INT64 及其他 dtype |
| 性能 | 代表性 shape 下原实现与优化实现对比 |
| 路径 | `PACKED_REUSE`、`STRIDED_TILE`、`TINY`、`FP64_FALLBACK` 均至少覆盖一组用例 |

不包含 tensor-prob、broadcast、四套新 API 或与本任务无关的其他随机算子测试。

### 4.5 验收交付件

设计与实现完成后，交付件必须包含：

1. 按社区模板填写并通过评审的 `design.md`，其中包含 inplace 融合方案和内存优化分析。
2. 任务书指定的 [aclnn_bernoulli 自测用例目录](https://gitcode.com/cann/cann-ops-competitions/tree/master/04_tasks/01_community-task-2026/docs/202607/self_test_case/aclnn_bernoulli/) 下的可运行自测用例、测试脚本及脚本运行 README。
3. 覆盖全部自测用例的结果报告，包括内存对比数据、精度结果和整体测试通过截图。
4. 个人代码仓链接、仓库路径、分支名、算子目录和完整 README。
5. 按任务书要求邀请账号 `Ascend-CANN` 作为个人仓开发者。
6. 测试通过后，向 [ops-math/experimental/random](https://gitcode.com/cann/ops-math/tree/master/experimental/random) 目录提交 PR。

---

## 五、实现前检查清单

| 检查项 | 通过条件 |
|--------|----------|
| 任务范围 | 只优化现有 `aclnnBernoulli`，没有新增 tensor-prob 或公开 Inplace API |
| inplace 融合 | Fill 和 DropoutDoMask 不再同时保留两份全量输出 |
| 地址安全 | 连续路径按高地址到低地址 wave 展开，非连续路径不强行同址复用 |
| 路径覆盖 | 连续、非连续、极小 Tensor 和 FP64 fallback 均有独立测试 |
| 参数一致 | BFLOAT16、offset 对齐、10 种 self/out dtype 均与任务书一致 |
| 内存目标 | 所有验收用例与 GPU 差距低于 5% |
| 精度目标 | 满足 AscendOpTest 默认阈值，并符合 PyTorch Bernoulli 语义 |
| 性能目标 | 优化后不低于原算子 |
| 交付完整 | 自测脚本、README、报告、内存数据、精度数据和截图齐全 |

---

> **文档状态**：已确定采用单 Kernel 路径；packed mask 的具体 bit 顺序、对齐和地址复用能力以目标分支的实际 CANN API 为准。
> **下一步**：完成原始实现的内存基线和 `Fill`/`DropoutDoMask` 缓冲区生命周期分析，实现单 Kernel 的 tile 级复用，并补齐自测报告。
