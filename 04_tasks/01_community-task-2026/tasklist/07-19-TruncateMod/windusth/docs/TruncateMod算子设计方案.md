# 需求背景（required）

## 需求来源

CANN 训练营 2026 暑期季西安交通大学专场 `TruncateMod` 算子开发任务。任务要求参考昇腾 CANN 内置 `TruncateMod` 实现，使用 Ascend C 实现功能一致的自定义算子，并完成设计、开发、测试和 PR 交付材料。

## 背景介绍

### TruncateMod 算子实现优化

`TruncateMod` 对输入 `x1` 和 `x2` 执行逐元素取模，商按向 0 截断规则计算，余数与 `TruncateDiv` 的商定义保持一致。算子需要支持 broadcast，使两个输入扩展到统一输出 shape 后再计算。

计算公式如下：

```text
q_i = trunc(x1_i / x2_i)
y_i = x1_i - q_i * x2_i
```

### TruncateMod 算子现状分析

当前实现支持 `float32`、`float16`、`bfloat16` 三种输入输出类型，format 为 ND，支持 same-shape、rank 内 broadcast、跨 rank broadcast 和 rank-0 scalar broadcast。实现重点是 broadcast offset 映射、连续输入的向量快路径、一般 broadcast 的 UB gather、向 0 截断商计算和余数写回。

# 需求分析

## 外部组件依赖

不依赖第三方组件。算子通过 msopgen 生成的自定义算子工程接入 ACLNN 调用链，并在 OPP 包中注册原型、tiling、kernel 和 opapi。

## 内部适配模块

- `op_host/truncate_mod_def.cpp`：注册输入、输出、dtype、format 和 soc 配置。
- `op_host/truncate_mod_infershape.cpp`：按 broadcast 规则推导输出 shape。
- `op_host/truncate_mod_tiling.cpp`：生成 broadcast 后的 shape、stride、多核切分和 UB tile 参数。
- `op_kernel/truncate_mod.h`：实现 Ascend C kernel 的连续向量快路径、一般 broadcast gather 路径和截断取模。

## 需求模块设计

### 算子原型

| 名称 | 类别 | dtype | format | shape | 说明 |
| --- | --- | --- | --- | --- | --- |
| `x1` | 输入 | `float32/float16/bfloat16` | ND | all | 被取模数 |
| `x2` | 输入 | `float32/float16/bfloat16` | ND | 可 broadcast 到输出 shape | 模数 |
| `y` | 输出 | 与输入一致 | ND | `x1` 与 `x2` 的 broadcast shape | 截断取模结果 |

### 相关约束

- 两个输入 dtype 必须一致。
- broadcast 维度必须满足相等或其中一个为 1。
- rank 最大支持 8。
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

1. 读取 `x1`、`x2` 的 shape 和 dtype。
2. 从末尾维度对齐，按 broadcast 规则推导输出 shape；不合法维度直接返回错误。
3. 将输入 shape 补齐到统一 rank，计算 `x1Stride`、`x2Stride` 和 `outStride`。
4. 计算输出元素总数 `totalNum`，小张量按每核至少 32 个元素的策略限制实际使用核数，避免把极小 tensor 切成大量单元素任务。
5. 根据实际使用核数切分连续输出区间，并生成 `blockFactor`。
6. 根据 UB 大小、输入输出队列和低精度 dtype 的 FP32 临时缓冲计算每次处理的 `ubFactor`，并按 64 元素对齐。
7. 按 dtype 设置 tiling key，使 kernel 选择对应模板实例。

### Kernel 侧设计

每个 AIV 核处理一段输出线性 index：

1. 如果 `x1`、`x2` 均为 same-shape 或 rank-0 scalar broadcast，则输入在输出线性顺序上连续，kernel 使用 `DataCopyPad` 或 `Duplicate` 搬入 UB 后直接执行向量计算。
2. 对一般 rank 内或跨 rank broadcast，kernel 对每个输出 index 使用 `outStride` 还原多维坐标，再根据 `x1Stride/x2Stride` 映射输入 offset，将一个 tile gather 到 UB。
3. gather 后复用同一套向量计算路径：`Div -> Trunc -> Mul -> Sub`，得到 `q=trunc(x1/x2)` 和 `y=x1-q*x2`。
4. `float16/bfloat16` 先提升到 FP32 做除法、截断和余数计算，再用 `CAST_RINT` 写回目标 dtype；输出按 tile 聚合写回，尾块使用 `DataCopyPad` 保证非对齐长度正确。

### 数据切分和同步策略

输出区间按连续线性范围分配给不同核，各核写回地址不重叠。核内通过 `TPipe` 和 `TQue` 管理搬入、计算、搬出的同步关系。

### Workspace

不需要 workspace，`workspace_bytes=0`。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| 香橙派 OrangePi AIpro |  |
| Atlas 200I/500 A2 推理产品 |  |
| Atlas 800I/T A2 | √ |

## 算子约束限制

当前实现支持 ND 连续张量，rank 最大为 8；输入 dtype 必须一致且仅覆盖浮点 dtype。除数为 0 时不做额外分支修正，按 Ascend 向量 `Div/Trunc` 浮点语义传播 NaN/Inf，本次自验证已覆盖 float32 0 除数 NaN 语义。

# 特性交叉分析

本算子为逐元素二输入算子，涉及 broadcast 但不涉及归约、atomic、多输出或跨核同步。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | 满足 CANN Judge 默认精度阈值；自验证用 CPU golden 按 broadcast 逻辑逐元素比较 | 任务书 |
| 性能标准 | 所有核参与场景性能不低于 TBE 基线的 95%，正式结果以 CANN Judge/评审环境为准 | 任务书 |

## 兼容性分析

新开发算子，不涉及历史版本兼容。本次复验在 Ascend 910B、CANN 9.0.0 notebook 环境完成，覆盖 16 个 TBE/Ascend C 对比 case，包含 same-shape、scalar broadcast、多维 broadcast、非对齐长度和 float 特殊值；自验证表已列出 TBE 基线运行成功日志、Ascend C 算子运行成功日志、精度结论和性能数据。
