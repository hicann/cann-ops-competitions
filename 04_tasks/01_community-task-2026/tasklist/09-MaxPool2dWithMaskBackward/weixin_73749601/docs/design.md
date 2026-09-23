# 需求背景（required）

## 需求来源

昇腾社区任务广场——9月社区任务-MaxPool2dWithMaskBackward算子开发（任务ID: 41ecbfa3d3cd4306adaf8236460aed08）

## 背景介绍

MaxPool2dWithMaskBackward 是 MaxPool2dWithMask 前向算子的反向传播函数。前向已基于 Ascend C 实现并通过全部精度测试（164/164）。

### TBE 实现现状分析

| 参数 | 参数含义 | 数据类型 | 支持数据类型 | 约束 | 形状 |
| --- | --- | --- | --- | --- | --- |
| gradOutput | 上游梯度 | tensor | float16, float32, bfloat16 | - | 4D |
| self | 正向输入 | tensor | float16, float32, bfloat16 | - | 4D |
| indices | argmax掩码 | tensor | int8 | int32流容器 | 4D |
| gradInput | 输出梯度 | tensor | float16, float32, bfloat16 | - | 4D |

# 需求分析（required）

## 需求描述

使用 Ascend C 实现 MaxPool2dWithMaskBackward 算子：根据 indices 中的 argmax 位置，将 gradOutput 梯度回填到 gradInput 对应位置（scatter-add）。

## 需求拆解

1. 支持 float16/float32/bfloat16 数据类型
2. 支持非对称核（kH≠kW）
3. indices int8 容器头部 int32 LE 流解码
4. 步长 < 核尺寸时的梯度累加（overlap 场景）
5. fp32 域累加保证精度（与 golden np.float32 一致）

# 详细设计（required）

## 算子分析

### 数学公式

gradInput(n, c, ih, iw) += gradOutput(n, c, ho, wo) 当且仅当 indices(n, c, ho, wo) == ih*W + iw

未被 argmax 选中的输入位置梯度为 0。

## 算子实现

### Kernel 侧设计

**架构**：AIV-only，按 (nc) 通道划分核——每核独占若干通道的 gradInput 平面，无跨核写冲突。

**数据流**（per (nc, ih) 输入行）：
1. 零填 fp32 行缓冲
2. 遍历受影响输出行 [hoStart, hoEnd]：
   - GM GetValue 读 indices int32 行
   - DataCopyPad 读 gradOutput 行 + 整行 Cast 到 fp32
   - 标量 scatter-add（idx ∈ [ih*W, (ih+1)*W) 范围筛选 → rowBuf[iw] += grad）
3. fp32 行缓冲一次性 Cast 回 T 写出 gradInput 行

**关键设计决策**：
1. **fp32 域累加**：bf16/fp16 的 gradOutput 整行 Cast 到 fp32 后累加，最终一次性 Cast 回 T（与 golden np.float32 一致）
2. **范围筛选免除法**：idx ∈ [ih*W, (ih+1)*W) 等价于解码到本输入行，避免逐元素除法
3. **按输入行处理**：每行独立零填→累加→写回，天然处理 stride < kernel 时的梯度累加

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas 800I/T A2 | √ |
| Atlas 800I A3 | √ |
| Atlas 300V Pro | √ |

## 算子约束限制

- dilation 仅支持 1
- 输入暂不支持 NaN/-Inf
- 输入维度仅支持 4D

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | FLOAT32: rtol=2^-10, atol=2^-16, matched_ratio≥0.99 | 生态算子开源精度标准 |
| 性能标准 | 不低于 TBE 版本 95%（910B3） | 任务书 |

## 兼容性分析

新算子，不涉及兼容性分析

# 测试验证

## 精度测试

140/140 用例全部通过：
- 覆盖 float32(59)/float16(44)/bfloat16(37) 全 dtype
- 覆盖 20 种 kernel/stride/padding 组合（含非对称核 k3x5, k5x3, k7x3 等）
- 覆盖 ceilMode=True/False
- 覆盖 stride < kernel（overlap 累加）和 stride ≥ kernel（无 overlap）场景

代码仓：https://gitcode.com/weixin_73749601/max_pool2d_with_mask_backward
