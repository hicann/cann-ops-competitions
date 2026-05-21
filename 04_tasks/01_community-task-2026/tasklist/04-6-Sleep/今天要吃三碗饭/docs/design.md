# Sleep 算子设计文档

# 1. 需求背景（required）

## 需求来源
为了提升 NPU 对主流深度学习框架（如 PyTorch）及三方库（如 VeOmni）的兼容性，需在 CANN 算子库中补齐 `torch.cuda._sleep` 的对标功能。

## 背景介绍
`torch.cuda._sleep` 主要用于开发测试场景，通过在设备侧执行 busy-spin（忙等待）来模拟特定耗时的计算。
### 应用场景
* **异步性测试**：验证 CUDA/NPU Stream 的异步执行行为。
* **同步语义验证**：测试 Event、Barrier、Fence 等同步原语的正确性。
* **性能模拟**：在不编写复杂 Kernel 的情况下模拟特定耗时的算子执行，用于流水线分析。

### Sleep 算子实现现状分析
当前 NPU 侧缺失直接对标的 API。现有的算子多为数据处理型，缺乏纯粹基于硬件周期的控制流算子。需基于 Ascend C 编程语言，通过读取 AI Core 内部计数器实现精确的周期级延时。

| 参数 | 参数含义 | 数据类型 | 支持数据类型 | 约束 | 形状 |
| --- | --- | --- | --- | --- | --- |
| cycles | 目标延时周期数 | 属性 (int64) | int64 | 非负数 | 标量 |

# 2. 需求分析（required）

## 需求描述
在 `ops-nn/control/sleep` 目录下实现 `Sleep` 算子。该算子接收一个 `int64` 类型的 `cycles` 参数，要求 AI Core 在执行时通过忙等待方式阻塞执行流，直到达到指定的硬件周期数。

## 需求拆解
1.  **接口实现**：提供 `aclnnSleepGetWorkspaceSize` 和 `aclnnSleep` 两段式接口。
2.  **核心逻辑**：在物理核上实现 `busy-spin` 逻辑。
3.  **硬件适配**：适配 Atlas A2 训练系列产品及 Atlas A3 系列产品。
4.  **测试绑定**：通过 pybind11 封装，确保可在 PyTorch 环境下通过类似 `torch_npu._sleep(cycles)` 的方式调用。

# 3. 详细设计（required）

## 3.1 算子分析

### 数学逻辑
该算子无数学公式，其核心逻辑为时延控制：
$$T_{delay} = \frac{cycles}{f_{core}}$$
其中 $f_{core}$ 为 AI Core 的工作频率。算子通过循环读取硬件计数器，当 $Count_{current} - Count_{start} \ge cycles$ 时退出。

### 支持数据类型
* `cycles`: `int64_t`

## 3.2 Ascend C 算子实现

Ascend C 算子流程图
![mermaid-diagram-2026-04-30-001606.png](https://raw.gitcode.com/user-images/assets/7665709/00108804-7da5-4697-aee9-a89d12569fce/mermaid-diagram-2026-04-30-001606.png 'mermaid-diagram-2026-04-30-001606.png')

### 3.2.1 host 侧设计：

**tiling 策略：**
由于 `Sleep` 算子不涉及大规模数据的搬运和切分，其 Tiling 过程极其简化：
1.  **参数透传**：将 `cycles` 参数直接封装进 `SleepTilingData` 结构体中。
2.  **核数规划**：默认启动 1 个核心（Block）执行即可满足时延阻塞需求。若需模拟多核同时占用的场景，可根据需求配置 `blockDim`。
3.  **Workspace**：该算子不需要额外的内存空间进行中间计算，故 `workspaceSize` 返回 0。

**接口定义：**
* `aclnnSleepGetWorkspaceSize`: 仅负责计算并将 `cycles` 存入 `executor`。
* `aclnnSleep`: 负责启动 Kernel。

### 3.2.2 kernel 侧设计：

**实现方案：**
1.  **初始化 (Init)**：从 Tiling 数据中获取目标 `cycles`。
2.  **执行 (Process)**：
    * 调用 Ascend C 的硬件读取接口（如 `get_tick()` 或通过汇编读取专用寄存器）获取起始 tick 数。
    * 进入 `while` 循环不断读取当前 tick 值。
    * 计算当前 tick 与起始 tick 的差值。
    * 当差值大于等于目标 `cycles` 时，跳出循环。

## 3.3 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas A2 训练系列产品 | √ |
| Atlas A3 系列产品 | √ |

## 3.4 算子约束限制
1.  `cycles` 必须为正整数。
2.  受硬件计数器频率影响，极小的 `cycles` 值可能因为指令调度开销导致实际时延略大于预期值。

# 4. 可维可测分析

## 4.1 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | 实际延时周期数应 $\ge$ 设定周期数，误差应在指令级开销范围内（通常小于几百个 cycles）。 | 对标 CUDA `_sleep` 实现 |
| 性能标准 | 不涉及（算子本身即为消耗时间的算子） | 不涉及 |

## 4.2 兼容性分析
本算子为新增算子，旨在补齐 NPU 在算子开发测试工具链上的短板，不涉及对已有算子的兼容性影响。