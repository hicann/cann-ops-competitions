# 需求背景（required）

## 需求来源

昇腾社区任务广场——9月社区任务-MaxPool2dWithMask算子开发（任务ID: 6e72d6cde6a84ddb867f6b7d599914a5）

## 背景介绍

### MaxPool2dWithMask 算子概述

MaxPool2dWithMask 算子执行 2D 最大池化并输出 argmax 掩码，用于反向传播的梯度路由。原实现为 TBE 版本，本任务要求基于 Ascend C 编程语言重新实现。

### TBE 实现现状分析

| 参数 | 参数含义 | 数据类型 | 支持数据类型 | 约束 | 形状 |
| --- | --- | --- | --- | --- | --- |
| x | 输入tensor | tensor | float16, float32, bfloat16 | 不支持NaN/-Inf | 3D/4D |
| out | 输出tensor | tensor | float16, float32, bfloat16 | - | 3D/4D |
| indices | argmax掩码 | tensor | int8（int32流容器） | 逐位一致 | 3D/4D |

属性：kernelSize（list_int，长度1-2）、stride（list_int，可空默认等于kernelSize）、padding（list_int，长度1-2）、dilation（仅支持1）、ceilMode（bool，可选）

计算公式：out(n,c,h,w) = max(input(n,c,h*sH-pT+kh, w*sW-pL+kw))，indices 记录 argmax 展平位置

# 需求分析（required）

## 需求描述

使用 Ascend C 编程语言实现 MaxPool2dWithMask 算子，支持 float16/float32/bfloat16 数据类型、3D/4D 输入、ceilMode，输出值精确匹配 + argmax 掩码逐位一致。

## 需求拆解

1. 支持 float16/float32/bfloat16 数据类型
2. 支持 3D [C,H,W] 和 4D [N,C,H,W] 输入
3. 支持 ceilMode（含尾窗钳制语义）
4. 支持 1 长度属性广播（kernelSize=[k] → [k,k]）
5. 支持 stride 为空列表时默认等于 kernelSize
6. indices 输出为 int8 容器（头部 int32 LE argmax 流 + 尾部清零）
7. 性能不低于 TBE 版本的 95%

# 详细设计（required）

## 算子分析

### 数学公式

out(N_j, C_j, h, w) = max_{m∈[0,kH), n∈[0,kW)} input(N_j, C_j, stride[0]×h + m - padT, stride[1]×w + n - padL)

indices 为 argmax 的展平线性索引 ih*W+iw（tie 取先到/最小线性索引）

### 支持数据类型

float16、float32、bfloat16

### 支持形状

3D [C,H,W] 和 4D [N,C,H,W]，非连续输入自动 Contiguous

## 算子实现

### 实现方案

#### Host 侧设计

**tiling 策略**：按输出行（NC×Ho）做行并行，大/小核均分。dtype 经 TPL DATATYPE 模板参数分发（fp16/fp32/bf16 三变体）。schMode 按 sW 分三档（0=sW=1 连续平面 / 1=sW=2 跨距平面 / 2=generic）。

**核心参数**：
- woAlign = align64(wOut)（CB22 掩码 64 对齐）
- planeStrideT/F = 32B 对齐平面行距
- UB 预算 = (kW+1)×planeStrideT×sizeof(T) + kW×planeStrideF×4 + 固定缓冲

#### Kernel 侧设计

**架构**：AIV-only，每核独立处理分配的输出行子集。

**数据流**（每输出行）：
1. LoadPlanes：按 (kh, kw) 加载 kW 张 32B 对齐输入平面（-inf 预填边界）
2. Accumulate：fp32 域成对 (max, argmax) 向量链更新
3. WriteRow：T 域值直写 out + fp32 索引 Cast 后写 indices

**成对更新核心**（对每个 (kh, kw)）：
```
Adds(candIdxF, rampF, base_scalar, woAlign)      // 候选索引 = ow*sW + base
Compare(gtMask, candF32, curVal, GT, woAlign)     // fp32 域比较
Select(curIdxF, gtMask, candIdxF, curIdxF, ...)   // fp32 索引 Select
Select(curValT, gtMask, candT, curValT, ...)      // T 域值 Select（位级精确）
Max(curVal, curVal, cand, woAlign)                // fp32 值更新
```

**关键设计决策**：
1. **fp32 域全计算**：bf16/fp16 加载后 Cast 到 fp32 比较/选择，避免低精度比较误差
2. **索引 fp32 真实值存储**：避免 int32-as-float ReinterpretCast Select 的可靠性问题（大 shape 下 Select 不生效）
3. **T 域值跟踪**：fp16/bf16 的输出值通过 T 域 Select 保证位级精确（免 Cast 精度损失）
4. **V→MTE 跨管 PIPE_ALL 屏障**：PipeBarrier<PIPE_V>() 不足以让 MTE 管看到 V 管写入
5. **单一大 TBuf 手工切分**：规避多 TBuf<VECCALC> 混叠问题
6. **indices 容器尾区清零**：各核分片 DataCopyPad 写零到 streamEnd 之后的区域

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas 800I/T A2 | √ |
| Atlas 800I A3 | √ |
| Atlas 300V Pro | √ |

## 算子约束限制

- dilation 仅支持 1
- 输入暂不支持 NaN/-Inf
- 输入维度仅支持 3D/4D

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | FLOAT32: rtol=2^-10, atol=2^-16, matched_ratio≥0.99；indices 逐位一致 | 生态算子开源精度标准 |
| 性能标准 | 不低于 TBE 版本 95%（910B3） | 任务书 |

## 兼容性分析

新算子，不涉及兼容性分析

# 测试验证

## 精度测试

164/164 用例全部通过（153 任务书用例 + 11 自补泛化用例）：
- 覆盖 float16/float32/bfloat16 全 dtype
- 覆盖 3D/4D 输入
- 覆盖 ceilMode=True/False
- 覆盖非对称核（1×3, 3×1, 2×5）
- 覆盖 sW=3 generic 路径
- 覆盖全 tie 常值输入
- 覆盖大 kernel 9×9

## 性能测试

在 910_93 上完成基线采集，sW=1 已向量化（单块 DataCopyPad），sW≥2 走标量路径待优化。

代码仓：https://gitcode.com/weixin_73749601/max_pool2d_with_mask
