# 需求背景（required）

## 需求来源

CANN 训练营 2026 暑期季西安交通大学专场 `SeluGrad` 算子开发任务。任务要求参考昇腾内置 `aclnnSeluBackward` 的 TBE 实现，使用 Ascend C 实现功能一致的自定义算子，并完成设计、开发、测试和 PR 交付材料。

## 背景介绍

### SeluGrad 算子实现优化

`SeluGrad` 用于 SELU 激活函数反向传播，输入为上游梯度 `gradients` 和前向输出 `outputs`，输出 `y` 为反向梯度。算子按元素计算，不涉及跨元素归约。

计算公式如下：

```text
scale = 1.0507009873554804934
scale_alpha = 1.75809934084737686
y_i = gradients_i * scale                      if outputs_i >= 0
y_i = gradients_i * (outputs_i + scale_alpha)  if outputs_i < 0
```

### SeluGrad 算子现状分析

当前实现支持 `float32`、`float16`、`bfloat16` 三种输入输出类型，format 为 ND，输入 shape 必须一致。实现重点是同 shape 连续张量的多核切分、非 32 对齐尾块搬运，以及正负分支的逻辑值比较。

# 需求分析

## 外部组件依赖

不依赖第三方组件。算子通过 msopgen 生成的自定义算子工程接入 ACLNN 调用链，并在 OPP 包中注册原型、tiling、kernel 和 opapi。

## 内部适配模块

- `op_host/selu_grad_def.cpp`：注册算子输入、输出、dtype、format 和 soc 配置。
- `op_host/selu_grad_infershape.cpp`：校验 `gradients` 与 `outputs` shape 一致，并设置输出 shape。
- `op_host/selu_grad_tiling.cpp`：根据元素数、UB 大小和 AIV 核数生成 tiling 参数。
- `op_kernel/selu_grad.h`：实现 Ascend C kernel 的搬入、计算和搬出流程。

## 需求模块设计

### 算子原型

| 名称 | 类别 | dtype | format | shape | 说明 |
| --- | --- | --- | --- | --- | --- |
| `gradients` | 输入 | `float32/float16/bfloat16` | ND | all | 上游梯度 |
| `outputs` | 输入 | `float32/float16/bfloat16` | ND | 与 `gradients` 相同 | SELU 前向输出 |
| `y` | 输出 | 与输入一致 | ND | 同输入 | 反向梯度输出 |

### 相关约束

- `gradients` 与 `outputs` 的 dtype 和 shape 必须一致。
- 当前提交面向连续 ND 张量。
- 当前注册并验证的 soc 为 `ascend910b`、`ascend910_93`；A3/`ascend950` 需在对应 CANN 环境中单独验证后再扩展配置。

# 需求详细设计

## 使能方式

| 上层框架 | 涉及的框架勾选 |
| --- | --- |
| TF训练/推理 |  |
| Pytorch训练/推理 |  |
| ATC推理 | √ |
| Aclnn直调 | √ |
| OPAT调优 |  |
| SGAT子图切分 |  |

## 需求总体设计

### Host 侧设计

1. 从输入 desc 获取 dtype、shape 和元素总数 `totalNum`。
2. 校验两个输入 shape 相等，输出 shape 直接继承输入 shape。
3. 查询平台 AIV 核数与 UB 大小，按 `ceil(totalNum / coreNum)` 生成核间连续区间。
4. 根据输入、输出队列、double buffer、比较掩码和 FP32 临时缓冲估算 `ubFactor`，并按 64 元素对齐，便于向量比较和尾块处理。
5. 按 dtype 设置 tiling key，使 kernel 选择对应模板实例。

### Kernel 侧设计

每个 AIV 核处理一段连续元素，流程为 `CopyIn -> Compute -> CopyOut`：

1. `CopyIn` 使用 `DataCopyPad` 将 `gradients` 和 `outputs` 按 tile 搬入 UB，尾块不足对齐长度时由 padding 处理。
2. `Compute` 对 `currentNum <= 64` 的小 tile 使用低开销逐元素路径；更大 tile 使用 `CompareScalar` 和 `Select` 完成 `outputs >= 0` 分支判断，分别计算 `gradients * scale` 或 `gradients * (outputs + scale_alpha)`。
3. `CopyOut` 使用 `DataCopyPad` 写回 `y`，真实写回长度由 tile 实际元素数控制。

### 数据切分和同步策略

算子无跨核数据依赖，各核写回区间互不重叠。核内使用 `TPipe` 和 `TQue` 管理 MTE2、VEC、MTE3 阶段之间的同步，避免显式 event 管理。

### Workspace

不需要 workspace，`workspace_bytes=0`。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| 香橙派 OrangePi AIpro |  |
| Atlas 200I/500 A2 推理产品 |  |
| Atlas 800I/T A2 | √ |

## 算子约束限制

不支持 broadcast；输入需为 dtype、shape 完全一致的 ND 连续张量。

# 特性交叉分析

本算子为逐元素反向梯度算子，不涉及融合、归约、atomic、多输出或跨核同步特性。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | 满足 CANN Judge 默认精度阈值；自验证用 CPU golden 逐元素比较 | 任务书 |
| 性能标准 | 所有核参与场景性能不低于 TBE 基线的 95%，正式结果以 CANN Judge/评审环境为准 | 任务书 |

## 兼容性分析

新开发算子，不涉及历史版本兼容。本次复验在 Ascend 910B、CANN 9.0.0 notebook 环境完成，覆盖 6 个 TBE/Ascend C 对比 case；自验证表已列出 TBE 基线运行成功日志、Ascend C 算子运行成功日志、精度结论和性能数据。
