# 需求背景（required）

## 需求来源

CANN 训练营 2026 暑期季西安交通大学专场 `SyncBatchNormBackwardReduce` 算子开发任务。任务要求参考昇腾内置 `SyncBatchNormBackwardReduce` TBE 实现，在 Ascend NPU 上使用 Ascend C 开发同功能算子，并完成设计、开发、测试和 PR 交付。

## 背景介绍

`SyncBatchNormBackwardReduce` 用于同步 BatchNorm 反向计算中的局部归约后处理。算子接收 `sum_dy`、`sum_dy_dx_pad`、`mean`、`invert_std` 四个输入，输出 `sum_dy_xmu` 和 `y` 两个张量。计算公式为：

```text
sum_dy_xmu_i = sum_dy_dx_pad_i - mean_i * sum_dy_i
y_i = sum_dy_xmu_i * invert_std_i
```

## 现状分析

内置 TBE 原型支持 `float32`、`float16`、`bfloat16`，format 为 ND，四个输入 shape 一致，两个输出 shape 与输入一致。当前 Ascend C 实现与该范围对齐，重点覆盖多输入、多输出、非 32 对齐长度、near-zero 消减场景和不同 dtype 的中间结果舍入。

# 需求分析（required）

## 外部组件依赖

不依赖第三方组件。算子工程由 `msopgen` 生成，通过 CANN `opbuild` 构建 OPP 包，通过 ACLNN API 调用自定义算子；性能和精度对照使用 CANN 9.0.0 notebook 环境中的内置 TBE `SyncBatchNormBackwardReduce`。

## 内部适配模块

- `op_host/sync_batch_norm_backward_reduce_def.cpp`：注册四个输入、两个输出、dtype、format 和 soc 配置。
- `op_host/sync_batch_norm_backward_reduce_infershape.cpp`：以 `sum_dy` 的 shape 设置两个输出 shape；四输入 shape 一致性由算子原型、调用约束和测试覆盖保证。
- `op_host/sync_batch_norm_backward_reduce_tiling.cpp`：生成多核切分、tiling key、UB tile 参数和 workspace。
- `op_kernel/sync_batch_norm_backward_reduce.h`：实现 Ascend C kernel 的搬入、计算和搬出流程。

## 需求模块设计

### 算子原型

| 名称 | 类别 | dtype | format | shape | 说明 |
| --- | --- | --- | --- | --- | --- |
| `sum_dy` | 输入 | `float32/float16/bfloat16` | ND | all | dy 的通道归约和 |
| `sum_dy_dx_pad` | 输入 | `float32/float16/bfloat16` | ND | 与 `sum_dy` 相同 | dy 与 x 相关项的 padding 后归约结果 |
| `mean` | 输入 | `float32/float16/bfloat16` | ND | 与 `sum_dy` 相同 | 前向保存的均值 |
| `invert_std` | 输入 | `float32/float16/bfloat16` | ND | 与 `sum_dy` 相同 | 前向保存的标准差倒数 |
| `sum_dy_xmu` | 输出 | 与输入一致 | ND | 与输入相同 | 中间结果 |
| `y` | 输出 | 与输入一致 | ND | 与输入相同 | 最终输出 |

### 约束

- 四个输入 dtype、format、shape 必须一致。
- 当前实现面向连续 ND 张量。
- 不涉及 broadcast、reduce axis 属性或可选输入。
- `float16/bfloat16` 输出按目标 dtype 舍入后写回。

# 需求详细设计（required）

## 使能方式

算子以自定义 OPP 包形式使能。构建后通过 `ASCEND_CUSTOM_OPP_PATH` 指向生成的 vendor 目录，ACLNN 调用 `aclnnSyncBatchNormBackwardReduceGetWorkspaceSize` 和 `aclnnSyncBatchNormBackwardReduce`。

## Host 侧设计

1. 读取输入 shape，计算总元素数 `totalNum`。
2. 查询平台 AIV 核数与 UB 大小。
3. 根据 dtype 设置 tiling key：`float16/float32/bfloat16` 三个模板。
4. 按 32B 对齐粒度生成 `blockFactor`，避免多核写回落在同一 32B cache line 内。
5. 按四输入两输出、double buffer 估算 `ubFactor`，并按最小搬运粒度对齐。
6. workspace 固定为 0。

## Kernel 侧设计

每个 AIV core 处理一段连续元素：

1. `CopyIn` 使用 `DataCopyPad` 分别搬入 `sum_dy`、`sum_dy_dx_pad`、`mean`、`invert_std`。
2. `Compute` 在 UB local tensor 上逐元素计算 `sum_dy_xmu` 和 `y`。
3. `float32` 直接在目标 dtype local tensor 上执行 `Mul/Sub/Mul`；`float16/bfloat16` 先提升到 FP32 临时缓冲计算，再用 `CAST_RINT` 写回目标 dtype。
4. `CopyOut` 使用 `DataCopyPad` 分别写回两个输出。
5. 对 `bfloat16` 使用 round-to-nearest-even 方式从 FP32 写回 BF16，匹配 TBE `round(..., "bfloat16")` 语义。

### 数据切分和同步策略

算子无跨核数据依赖，各核写回区间互不重叠。核内通过 `TPipe` 和 `TQue` 管理四路输入队列与两路输出队列的生产消费关系，由队列框架负责 MTE2、VEC、MTE3 阶段同步。

### Workspace

不需要 workspace，`workspace_bytes=0`。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| 香橙派 OrangePi AIpro |  |
| Atlas 200I/500 A2 推理产品 |  |
| Atlas 800I/T A2 | √ |

当前 `op_host` 已注册并验证 `ascend910b`、`ascend910_93` 两类 soc 配置。

## 算子约束限制

当前实现支持 ND 连续张量，要求四个输入 shape 完全一致，输出 shape 与输入一致。

# 特性交叉分析

本算子为逐元素四输入二输出算子，不涉及 broadcast、归约、atomic 或跨核同步。实现需要关注多输出队列生命周期、尾块 `DataCopyPad` 写回，以及 BF16 舍入语义。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | 满足 CANN Judge 默认精度阈值；自验证用 CPU golden 分别比较 `sum_dy_xmu` 和 `y` | 任务书 |
| 性能标准 | 所有核参与场景性能不低于 TBE 基线的 95%，正式结果以 CANN Judge/评审环境为准 | 任务书 |

## 兼容性分析

新开发算子，不涉及历史版本兼容。本次复验在 Ascend 910B、CANN 9.0.0 notebook 环境完成，覆盖 18 个 TBE/Ascend C 对比 case；自验证表已列出 TBE 基线运行成功日志、Ascend C 算子运行成功日志、精度结论和性能数据。
