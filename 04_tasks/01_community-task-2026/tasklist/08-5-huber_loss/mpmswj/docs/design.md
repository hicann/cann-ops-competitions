# HuberLoss 算子设计文档

> 文档性质：目标态设计与实现同步稿。  
> 代码基线：ops-nn/experimental/loss/huber_loss。  
> 当前状态：代码已在 Atlas A2（Ascend 910B3）、CANN 9.1.0 完成 Host UT 32/32、ACLNN L2 9/9、Kernel 仿真 11/11、生产编译、`.run` 打包及指定路径安装。真实 NPU 精度矩阵 198/198、负向矩阵 18/18、AscendOpTest 默认阈值 9/9、同批 CPU `aten::huber_loss` 对齐 9/9 均通过。strict_final2 大规模 msprof 的 9 个 Case、900 个正式样本全部为 `MIX_AIV` 且 AICPU 为 0，按名义 memory roofline 为 9/9 达到 80%。超过 100% 的值仅是名义逻辑模型结果，不表示物理 HBM 超过 100%。已新增 PyTorch 2.9 独立路由扩展，完成 forward/out/backward/backward.out 四个 PrivateUse1 注册；路由 UT 9/9，最终 profile 中 11 个 `HuberLoss` 前向任务均为 `MIX_AIV`、1 个 backward `SmoothL1` 任务为 `AI_VECTOR_CORE`，AICPU=0。

| 项目 | 内容 |
| --- | --- |
| 任务 | 8月社区任务 - huber_loss 算子开发 |
| 目标算子 | aten::huber_loss 前向算子 |
| 目标目录 | ops-nn/experimental/loss/huber_loss |
| 目标硬件 | Atlas A2 训练系列产品 |
| 软件基线 | CANN 9.0.0 及以上 |
| 输入格式 | ND |
| 支持类型 | float32、float16、bfloat16 |

# 需求背景（required）

## 需求来源

任务启动前，当前 NPU 后端未完整支持 aten::huber_loss，调用会回退到 CPU，造成设备间同步和数据搬运开销，显著拉长训练时间。本任务要求参考 PyTorch Huber Loss 语义，在 Atlas A2 训练系列产品上使用 Ascend C 实现前向算子，完成设计、开发、精度测试、性能测试和验收交付，并最终向 ops-nn 的 experimental/loss 目录提交代码。

任务书核心交付为 Atlas A2 上的 HuberLoss 前向计算，任务书定义的 input、target、reduction、delta、output 为 Ascend C 公共接口契约。PyTorch 新版本提供的 weight 参数属于 functional 前端组合能力，不作为该 Ascend C 公共接口的输入。

## 背景介绍

### HuberLoss 算子功能

Huber Loss 在误差较小时使用平方损失，在误差较大时使用线性损失，兼顾均方误差对小误差的平滑性和绝对误差对异常值的鲁棒性。

设第 i 个元素的误差为：

$$
e_i = input_i - target_i
$$

逐元素损失为：

$$
h_i =
\begin{cases}
0.5e_i^2, & |e_i| < delta \\
delta(|e_i| - 0.5delta), & |e_i| \geq delta
\end{cases}
$$

令输入元素总数为 N，则输出为：

$$
output =
\begin{cases}
\{h_i\}_{i=0}^{N-1}, & reduction = 0\ (none) \\
\frac{1}{N}\sum_{i=0}^{N-1}h_i, & reduction = 1\ (mean) \\
\sum_{i=0}^{N-1}h_i, & reduction = 2\ (sum)
\end{cases}
$$

delta 默认值为 1.0，且必须严格大于 0；reduction 默认值为 1。

### PyTorch 参考实现语义

PyTorch 原生实现先计算逐元素 Huber Loss，再通过 mean、sum 或不归约返回结果。非 none 模式将输出调整为 0 维标量，delta 非正时返回错误。

任务书对 PyTorch 语义做了两项明确收敛：

- input 与 target 必须具有完全相同的 shape 和 dtype，不支持广播；
- 只支持 float32、float16、bfloat16 和 ND 格式。

因此，本设计以任务书契约为最高优先级，不在 Ascend C 核心接口中新增广播或 weight 输入；PyTorch 前端兼容能力在补充集成章节单独说明。

### 当前 Ascend C 代码现状

当前实现已统一使用 input、target、output 命名，执行链如下：

~~~mermaid
flowchart TD
    A["ACLNN 两段式调用"] --> B["参数、dtype、shape、reduction、delta 校验"]
    B --> C{"空 Tensor?"}
    C -- "是" --> D["none 直接返回；mean/sum Fill NaN 或 0"]
    C -- "否" --> E["input/target Contiguous"]
    E --> F["L0 HuberLoss 与 InferShape"]
    F --> G["Host Tiling：none/mean/sum 三个 TilingKey"]
    G --> H["Kernel：有限 delta clipped 公式；+Inf 特殊位模式路径"]
    H --> I{"reduction=none?"}
    I -- "是" --> J["按原 dtype 逐元素 CopyOut"]
    I -- "否" --> K["8-lane tile 折叠与 FP32 单核/多核归约"]
    K --> L["mean 缩放或 sum，写严格 0 维标量"]
    J --> M["ViewCopy 到调用方 output"]
    L --> M
~~~

当前源码具有以下能力：

| 模块 | 已实现内容 |
| --- | --- |
| OpDef | input/target/output；三组严格对齐的 FP32/FP16/BF16、ND；reduction 默认 1、delta 默认 1.0；按任务书适配 Atlas A2（ascend910b） |
| 非连续输入 | OpDef 声明 AutoContiguous；手写 ACLNN 显式调用 Contiguous；三种 dtype 的 input-only、target-only、both 非连续 view 已纳入 198 个 A2 真机 Case |
| InferShape | 兼容动态 shape；拒绝已知不一致 shape 和非法 reduction；none 保持输入 shape，mean/sum 生成严格 0 维输出 |
| Host Tiling | 校验 dtype/shape/属性/输出；按 64 元素对齐计算 UB tile；选择 none、单核 reduce 或多核 reduce，并设置三种 TilingKey |
| Kernel | 有限 delta 使用 clipped 无分支公式；`delta=+Inf` 使用特殊位模式路径，并区分有限二次项溢出的 `+Inf`、真正无限差值的 NaN 标记及输入 NaN，在单核/多核规约中保留标记。FP16 按源 dtype 逐步计算并舍入；BF16 以 FP32 运算加 BF16 RNE 往返复现源 dtype 物化；Host 将 delta 按输入 dtype RNE 量化；低精度逐元素 loss 物化后再以 FP32 规约，最终转回输入 dtype。有限 `delta=1.0` 的非有限输入和有限差值溢出已真机 12/12 验证可由普通 clipped 热路径正确处理，无需增加特殊分类指令 |
| Workspace | none 和单核 reduce 为 0；多核 reduce 为平台系统同步 workspace 加每核一个 UB block 对齐 partial |
| ACLNN/L0 | 手写两段式接口，完成严格校验、空 Tensor 分流、Contiguous、AICore launcher 和 ViewCopy |
| 测试源码 | UT 覆盖 reduction/dtype/空 Tensor/尾块/多核、`delta=+Inf`、有限溢出与低精度 CPU aten 严格语义；ACLNN 矩阵覆盖 198 个精度和 18 个负例，其中有限 `delta=1.0` 非有限输入/差值溢出 12/12 通过；这 12 个为真实 NPU 专项，因 CPU Kernel simulator 对有限 delta 下 NaN/Inf 的行为与 910B 真机不同而未并入 Kernel UT，Kernel UT 仍为 11/11；AscendOpTest 覆盖 9 个基础组合；benchmark 支持 9 组大规模性能复现 |

当前构建与验收证据如下：

| 验证项 | 当前证据 | 完成条件 |
| --- | --- | --- |
| CANN 构建与仓内 UT | Host 32/32、L2 9/9、Kernel 11/11；三套 ascend910b 生产 Kernel、`.run` 和指定路径安装均成功；最终 `.run` SHA-256 为 `1fe41d6ef576939f3b3b54f2e851bc70cc79ed4b44a39a344d20bb5240bc1598` | 通过 |
| CPU aten/AscendOpTest 精度 | AscendOpTest 默认阈值 9/9；相同输入对 PyTorch 2.9.0 CPU aten 9/9 | 通过 |
| 非连续、空 Tensor 与低精度语义 | 198/198 精度矩阵覆盖既有 153 个 dtype/layout/empty/低精度严格语义场景及总计 45 个特殊值/溢出场景；第二批 12 个有限 `delta=1.0` 非有限输入和差值溢出场景真机 12/12 通过，确认无需改动普通热路径；`delta=0.7` 与 tiny delta 为 exact 零容差 | 通过 |
| AICore 路由 | 最终性能矩阵 900 个正式 `HuberLoss` 样本均为 `MIX_AIV`，全流程 AICPU=0 | 通过 |
| 80% bound | `N=2^26` 的 strict_final2 9 组 Case 完整测量，最终 9/9 达标；原 FP16/BF16 mean/sum 四个未达场景优化后为 81.29%–117.42% | 通过 |
| PyTorch PrivateUse1 补充兼容验证（非任务书核心验收项） | 独立扩展完成 forward/out/backward/backward.out 四个 schema 注册；路由 UT 9/9；profile 中 11 个 `HuberLoss/MIX_AIV`、1 个 backward `SmoothL1/AI_VECTOR_CORE`，AICPU=0 | 通过 |

# 需求分析（required）

## 需求描述

使用 Ascend C 实现 HuberLoss 前向算子，使其在 Atlas A2 训练系列产品上完成 NPU 原生计算。算子必须支持 FP32、FP16、BF16，支持 none、mean、sum 三种 reduction，支持任意合法 ND shape 和非连续 input/target，且精度与 CPU aten::huber_loss 对齐。

## 需求拆解

1. 实现 Huber 分段公式，正确覆盖 |e| 小于、等于和大于 delta 的场景。
2. 支持 float32、float16、bfloat16，输出 dtype 与输入一致。
3. 支持 reduction=0、1、2，默认值为 1。
4. reduction=0 时输出与 input 同 shape；reduction=1/2 时输出为 0 维标量。
5. input 与 target 的 shape、dtype 必须一致，不支持广播。
6. delta 默认 1.0，且必须大于 0。
7. input 和 target 支持非连续 Tensor，在进入 HuberLoss Kernel 前转换为连续布局。
8. delta 先按输入 dtype RNE 量化；FP16 按源 dtype 逐步计算并舍入，BF16 采用 FP32 运算加 BF16 RNE 往返复现源 dtype 物化。两种低精度的逐元素 loss 均先物化为源 dtype，再转 FP32 归约；partial/workspace 累加保持 FP32，最终转回原输出 dtype。`delta=+Inf` 另走位模式路径。
9. 支持标量、多维、动态 shape 的运行时具体化、非 32B 对齐尾块和空 Tensor。
10. 精度满足 AscendOpTest 默认阈值，并与 CPU aten::huber_loss 对齐。
11. 性能达到 80% compute bound 或 memory bound；未达到的场景给出数据和原因。
12. ACLNN 与 PyTorch profiler 均确认 forward 执行 HuberLoss AICore Kernel、backward 执行 NPU SmoothL1 Kernel，且不走 AICPU；四个 PrivateUse1 schema 均已注册。

## 接口契约

### 逻辑接口

~~~text
huber_loss(
    input: Tensor,
    target: Tensor,
    reduction: int = 1,
    delta: float = 1.0
) -> output: Tensor
~~~

### 参数定义

| 参数 | 类别 | 必选性 | dtype | format | shape | 说明 |
| --- | --- | --- | --- | --- | --- | --- |
| input | 输入 | 必选 | FP32、FP16、BF16 | ND | 任意维，各维大于或等于 0 | 预测值，支持非连续 Tensor |
| target | 输入 | 必选 | 与 input 相同 | ND | 与 input 完全相同 | 目标值，支持非连续 Tensor |
| reduction | 属性 | 可选 | int64 | - | - | 默认 1；0=none，1=mean，2=sum |
| delta | 属性 | 可选 | float | - | - | 默认 1.0，必须大于 0 |
| output | 输出 | 必选 | 与 input 相同 | ND | none 同输入；mean/sum 为 0 维 | Huber Loss 结果 |

### 合法性与边界语义

| 场景 | 预期行为 |
| --- | --- |
| input/target shape 不同 | 参数错误，不进行广播 |
| input/target dtype 不同 | 参数错误 |
| reduction 小于 0 或大于 2 | 参数错误 |
| delta 小于等于 0 或 NaN | 参数错误；以 delta > 0 判断 |
| delta 为正无穷 | 合法；任务书只要求 delta > 0，结果与 CPU 参考逐项对齐，不额外增加有限值限制 |
| 标量输入 | N=1；none 输出标量，mean/sum 也输出标量 |
| 空 Tensor + none | 返回与 input 同 shape 的空 Tensor |
| 空 Tensor + sum | 返回 0 维标量 0 |
| 空 Tensor + mean | 返回 0 维标量 NaN，与空 Tensor mean 语义对齐 |
| 动态 shape | InferShape 可保留动态信息；Kernel 启动前必须得到具体 storage shape |
| 非连续 input/target | 通过 AutoContiguous 或 L2 Contiguous 转为连续 Tensor 后进入 Kernel |

## 范围说明

本次包含：

- HuberLoss forward；
- ACLNN/OpDef、InferShape、Host Tiling、Ascend C Kernel；
- FP32/FP16/BF16、ND；
- none/mean/sum；
- Atlas A2；
- 精度、泛化、ACLNN/AICore 执行路径和性能验证。

任务书核心交付不声明以下能力：

- HuberLoss backward；
- input/target 广播；
- weight 参数；
- 图融合；
- ND 以外的数据格式；
- Atlas A2 以外 SoC 的适配和验证。

任务书的 Ascend C 核心交付为 HuberLoss forward。为关闭业务侧 CPU fallback，本算子目录额外提供 `pytorch/` 独立扩展，作为非任务书核心验收项的补充兼容能力：它使用匹配 PyTorch 2.9 的官方 op-plugin/TorchNPU 生成器产生 dispatcher stub，forward 调用本算子的 `aclnnHuberLoss`，backward 对有限 delta 复用 NPU SmoothL1 backward，并对 `delta=+Inf` 单独处理。该扩展已在指定环境完成端到端验证，但不等同于已向上游 TorchNPU/op-plugin 仓提交合入。

## 验收交付物

| 交付物 | 本任务需完成的内容 |
| --- | --- |
| 算子设计文档 | 按社区模板完成本文，并按任务书要求向 cann-competitions 对应任务目录提交评审 PR |
| 自测用例及测试代码 | 提供本设计列出的精度、泛化、异常、ACLNN/AICore 执行路径和性能用例；测试 README 必须给出可复现步骤 |
| 自测报告 | 记录完整 Case 参数、CPU/NPU 精度结果、AscendOpTest 判定、性能数据以及必要截图和 profiler 原始证据 |
| 待验收代码地址 | 提供个人仓链接、分支和 `experimental/loss/huber_loss` 目录，满足任务书的仓库协作权限要求 |
| ops-nn 合入 PR | 所有测试和评审通过后，向 ops-nn 的 `experimental/loss` 目录提交代码 PR |

上述交付物中的测试结果均以真实日志和 profiler 原始证据填写；提交号和 PR 地址在实际提交后补入，不以计划值代替。

## 模块设计

| 模块 | 设计职责 |
| --- | --- |
| OpDef | 声明输入输出 dtype/format 组合、reduction/delta 默认值、AutoContiguous 和 ascend910b 配置 |
| InferShape | 校验同 shape；按 reduction 推导同 shape或 0 维标量 |
| ACLNN 手写封装层 | 空指针、dtype、shape、reduction、delta 校验；非连续输入处理；空 Tensor 快速路径 |
| Host Tiling | 获取平台核数和 UB；选择 none/单核 reduce/多核 reduce；计算 workspace 和 tiling data |
| Kernel | 有限 delta clipped 两 buffer 公式、`+Inf`/大 delta 特殊值分类与标记保留、FP16 逐步舍入、BF16 RNE 往返、8-lane tile 折叠、跨核 FP32 partial 汇总、结果 cast 和写回 |
| UT/ST | 覆盖 Host、Kernel、ACLNN、泛化、非连续、空 Tensor、AICore 执行路径和性能 |

# 详细设计（required）

## 算子分析

### 数学公式

逐元素部分：

$$
\begin{aligned}
e_i &= input_i-target_i \\
a_i &= |e_i| \\
q_i &= 0.5e_i^2 \\
l_i &= delta(a_i-0.5delta) \\
h_i &= a_i < delta\ ?\ q_i:l_i
\end{aligned}
$$

归约部分：

$$
output =
\begin{cases}
h, & reduction=0 \\
sum(h)/N, & reduction=1 \\
sum(h), & reduction=2
\end{cases}
$$

有限 delta 时，Kernel 使用与分段定义等价的无分支公式：

$$
m=\min(|e|, delta),\qquad h=m\times(|e|-0.5m)
$$

`delta=+Inf` 时采用独立位模式路径：有限误差进入二次项，有限二次项溢出保留 `+Inf`；真正
无限差值与输入 NaN 使用独立标记，并在单核/多核规约中传播，最终物化为 NaN。FP16/BF16 的
大有限 delta 路径也恢复 A2 FP32→低精度 Cast 饱和前应有的 `+Inf`/NaN 类别，与 CPU aten
语义一致。

Host 先把 delta 按输入 dtype 做 RNE 量化。FP16 在源 dtype 逐步执行并舍入公式中间结果；A2 高层 Ascend C API 不支持 BF16 原生 Sub，因此 BF16 采用 FP32 运算并在 CPU aten 会物化的位置执行 BF16 round-to-nearest-even 往返。两种低精度的逐元素 h 均先物化到源 dtype，再转回 FP32 做 partial/workspace 累加，从而对齐 CPU `aten::huber_loss` 的“逐元素低精度物化、FP32 归约”语义；大规模规约按默认阈值验收，不承诺 bit-exact。

### 支持数据类型

OpDef 中三个输入输出 dtype 表必须按索引一一对应：

| 组合编号 | input | target | output | 内部计算 |
| --- | --- | --- | --- | --- |
| 0 | FP32 | FP32 | FP32 | FP32 |
| 1 | FP16 | FP16 | FP16 | 源 dtype 逐步舍入；逐元素 FP16 loss；FP32 归约 |
| 2 | BF16 | BF16 | BF16 | FP32 运算 + BF16 RNE 往返；逐元素 BF16 loss；FP32 归约 |

不允许混合 dtype。format 和 UnknownShapeFormat 均为三组 ND。

### 支持形状

- input 和 target 支持 0 维标量及任意维 ND Tensor；
- 各维允许为 0，因此总元素数可能为 0；
- 静态场景下 input 与 target 的 rank 和每一维必须一致；动态场景只允许未知维兼容，运行时具体 shape 仍须完全一致；
- reduction=0 时 output view shape 与 input 相同；
- reduction=1/2 时 output view shape 为严格 0 维，内部 storage 只保存 1 个元素；
- 不支持广播。

## 算子实现

### 实现方案

总体调用链：

~~~mermaid
flowchart LR
    A["aclnnHuberLossGetWorkspaceSize"] --> B["参数校验"]
    B --> C["空 Tensor 快速路径"]
    B --> D["Contiguous"]
    D --> E["HuberLoss L0/OpDef"]
    E --> F["InferShape"]
    F --> G["Host Tiling"]
    G --> H["AICore Kernel"]
    H --> I["必要时 Cast/ViewCopy"]
    I --> J["用户 output"]
~~~

公开两段式接口为：

~~~cpp
aclnnStatus aclnnHuberLossGetWorkspaceSize(
    const aclTensor* input,
    const aclTensor* target,
    int64_t reduction,
    float delta,
    aclTensor* output,
    uint64_t* workspaceSize,
    aclOpExecutor** executor);

aclnnStatus aclnnHuberLoss(
    void* workspace,
    uint64_t workspaceSize,
    aclOpExecutor* executor,
    aclrtStream stream);
~~~

当前实现采用手写 op_host/op_api 封装，参考同仓 MseLoss 完成严格参数校验、显式 Contiguous、空 Tensor 分流和最终 ViewCopy。由于公开 ACLNN 接口由手写代码提供，op_host/CMakeLists.txt 已设置 `ACLNNTYPE aclnn_exclude`，避免自动生成接口与手写 `aclnn_huber_loss.*` 重复定义或打包冲突；OpDef、InferShape、Tiling 和 L0 launcher 仍按手写接口的执行图参与构建。

#### L2/接口侧设计

第一段接口依次完成：

1. 检查 workspaceSize、executor、input、target、output 非空。
2. 检查 input、target、output dtype 属于 FP32/FP16/BF16 且三者一致。
3. 检查 input 与 target view shape 完全一致。
4. 检查 reduction 只能是 0、1、2。
5. 检查 delta > 0；该判断同时拒绝 NaN 和非正数。
6. reduction=0 时检查 output view shape 等于 input；reduction=1/2 时检查 output 为 0 维。
7. 空 Tensor 在 Contiguous 前走快速路径：
   - none：直接成功；
   - mean：向 output 填充 NaN；
   - sum：向 output 填充 0。
8. 对非空 input 和 target 执行 Contiguous；OpDef 的 AutoContiguous 作为同一能力的声明和兜底。
9. 非空输入构建 HuberLoss 执行图，最终按 output dtype 和 view 写回。

#### Host 侧设计

##### OpDef 与 InferShape

OpDef 的公开 schema 已统一为 `input`、`target`、`output`；L0 的 `OP_INPUT/OP_OUTPUT`、Kernel 入口参数、dtype 编译宏、文档、示例和测试名称保持一致，其中 dtype 宏统一为 `DTYPE_INPUT`。

OpDef 属性顺序为：

~~~cpp
this->Attr("reduction").AttrType(OPTIONAL).Int(1);
this->Attr("delta").AttrType(OPTIONAL).Float(1.0);
~~~

固定属性索引为 reduction=0、delta=1。公开 ACLNN、OpDef、InferShape 和 Tiling 全链路按 int64 保持 0/1/2 语义；Host 仅在设置 TilingKey 时映射一次，Kernel 模板参数继续使用相同数值。若实际生成物证明框架发生了属性表示转换，则将其作为 ABI 集成问题处理，并在单一边界函数中显式映射；不得默认混用 GetInt/GetStr 掩盖契约错误。

InferShape 逻辑：

1. 获取 input、target、output shape 和 reduction。
2. 静态维必须逐维相等；未知维可以兼容传播，但不得据此引入广播，运行时再次校验具体 shape 完全一致。
3. 校验 reduction 位于 0 到 2。
4. reduction=0 时复制 input shape。
5. reduction=1/2 时设置 output 为 gert::Shape()，即 0 维标量。

##### 数据展平

计算过程与维度语义无关，Host 将 input 按连续一维向量处理：

$$
N = \prod_{d \in input.shape} d
$$

需要检查 shape size 未知和乘法溢出。动态 shape 只有在运行时具体化后才能进入 Tiling。

##### 分核策略

none 路径为逐元素计算，优先多核。为使各核 GM 起始位置满足 32B 对齐，按 dtype 计算：

$$
gmAlignElem = \frac{32}{sizeof(inputType)}
$$

设平台可用 AIV 核数为 C：

$$
\begin{aligned}
rawBlockFactor &= \left\lceil \frac{N}{C} \right\rceil \\
blockFactor &= align\_up(rawBlockFactor, gmAlignElem) \\
blockNum &= \left\lceil \frac{N}{blockFactor} \right\rceil
\end{aligned}
$$

最后一核只处理剩余元素，尾块使用 DataCopyPad。

mean/sum 路径需要全量归约。小、中小 shape 若使用多核，会额外产生 workspace、SyncAll 和二次汇总开销。当前实现根据单 tile 的 UB 容量使用保守启发式阈值：

$$
singleCoreThresholdInitial = CalcTileCapacity(ubSize, dtype)
$$

单核可以循环处理多个 tile，因此该值不是能力上限，也不能直接视为最终性能拐点。最终 singleCoreThreshold 必须使用 Atlas A2 profiler 对不同 dtype、shape 和 reduction 实测调优。当 reduction 不为 none 且 N 小于等于当前调优阈值时使用单核快速路径：

$$
blockNum=1,\quad blockFactor=N,\quad workspace=0
$$

当 N 超过阈值时使用多核两阶段归约。先以每核至少承担一个 GM 对齐块为约束计算候选核数，再反算对齐后的每核元素数：

$$
\begin{aligned}
reduceCoreCap &= \min\left(C, \left\lceil\frac{N}{gmAlignElem}\right\rceil\right) \\
blockFactor &= align\_up\left(\left\lceil\frac{N}{reduceCoreCap}\right\rceil, gmAlignElem\right) \\
blockNum &= \left\lceil\frac{N}{blockFactor}\right\rceil
\end{aligned}
$$

最后一核只处理余数，所有非最后核的 GM 起始地址保持 32B 对齐。多核归约必须设置批处理调度模式，保证参与 SyncAll 的核可同时驻留。

空 Tensor 不进入常规分核：none 直接返回；mean/sum 由接口快速路径填充值。若由 Kernel 兜底处理，则启动 1 个核并根据 reduction 写入 NaN 或 0。

##### UB 切分策略

最终 Kernel 删除 quadratic、linear 和 mask，仅保留两个 FP32 计算 buffer。不同路径的每元素 UB 预算为：

$$
bytesPerElem_{none}=6\times sizeof(inputType)+2\times sizeof(float)
$$

$$
bytesPerElem_{reduce}=4\times sizeof(inputType)+2\times sizeof(float)
$$

none 的 6 份 inputType 空间对应 input、target、output 三条双缓冲队列；reduction 不保留完整输出 tile，仅 input、target 双缓冲，并为最终标量保留一个 32B 输出 buffer。两个 FP32 buffer 分别承载 diff/clipped-loss 与 abs/excess，低精度 Cast 复用输入队列和这两个计算 buffer。Host 因此对 none/reduction 分别按 6/4 条 I/O 队列预算 tile，具体 ubFactor 由运行时 UB 容量、dtype、64 元素对齐和 255 repeat 上限共同确定，不在设计中固化。

当前 tile 大小为：

$$
tileNum =
floor\_align\left(
\left\lfloor\frac{availableUb}{bytesPerElem}\right\rfloor,
64
\right)
$$

归约路径另需每核一个 32B local partial 和 0 号核汇总队列。每个 tile 先通过成对 Add 折叠到一个 8-lane FP32 block，再跨 tile 累加这 8 个 lane；仅在本核/跨核最终阶段调用一次 ReduceSum。Host 预留 16KiB 固定 UB 空间覆盖 partial、汇总队列和管线管理开销，再按实际 InitBuffer 列表计算 availableUb；tileNum 同时受 255 个矢量 repeat 上限约束。任何 tileNum=0 或字段溢出均在 Host 侧报错。

##### TilingData 字段

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| totalNum | int64 | 输入总元素数 |
| blockFactor | int64 | 每核最多处理的元素数 |
| ubFactor | int64 | 单次 tile 处理的元素数 |
| blockNum | int64 | 实际启动核数 |
| workspaceFloatsPerCore | int64 | 每核 partial 槽位，按 UB block 字节数除以 sizeof(float) 推导；A2 常见为 8 个 FP32，即 32B |
| invNumel | float | mean 的 1/N；其他模式为 0 |
| delta | float | Host 已按 input dtype 进行 RNE 量化的 Huber 阈值 |

实现已由 blockFactor、blockNum、totalNum 统一表达分核信息，避免旧大小核字段及 uint32 单核元素数限制；Host 还校验最后一核 offset 的 int64 乘法范围和非空不变量。

##### TilingKey 规划

按 reduction 拆分为三个编译期模板实例：

| TilingKey | 模式 | 说明 |
| --- | --- | --- |
| 0 | none | 逐元素输出，无跨核归约 |
| 1 | mean | FP32 归约后乘 1/N |
| 2 | sum | FP32 归约后直接输出总和 |

输入 dtype 由 schema 生成的 `DTYPE_INPUT` 编译宏生成 FP32、FP16、BF16 实例，Kernel 入口采用 `template <uint32_t reductionMode>`，不再在运行时重复分派 reduction。`huber_loss_tiling_key.h` 使用仓库统一的 `ASCENDC_TPL_ARGS_DECL`、`ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(...))` 和 `GET_TPL_TILING_KEY` 机制声明、生成和选择 0/1/2 三个 key；Host 设置的 key、动态二进制信息和 Kernel 模板实例逐项一致。

##### Workspace 规划

| HuberLoss 主 Kernel 路径 | Tiling 返回的主 Kernel workspace |
| --- | --- |
| none | 0 |
| 单核 mean/sum | 0 |
| 多核 mean/sum | 系统同步 workspace + 每核一个 UB block 对齐的 partial（A2 通常为 32B） |

多核归约按平台提供的系统同步 workspace 与用户 partial workspace 之和申请：

$$
workspaceSize =
systemWorkspaceSize(platform)
+ blockNum \times workspaceFloatsPerCore \times sizeof(float)
$$

其中 systemWorkspaceSize 优先通过平台接口或仓库公共能力查询；用户 workspace 每核保存一个 UB block 对齐的 partial。当前同仓 MseLoss 的 A2 实现以 16MiB 作为系统 workspace，但 HuberLoss 不应在设计中未经平台查询直接固化该常量。Host 同时显式设置 SetNeedAtomic(false)，多核归约不依赖 GM 原子累加。

Kernel 入口只在“多核且 reduction 模板参数不为 none”时执行 SetSysWorkspace(workspace)，随后通过 GetUserWorkspace(workspace) 获取 partial 起始地址；严禁把原始 workspace 指针直接当作用户 partial 区。none 和单核 reduction 的 workspace 可以为空，不调用 SetSysWorkspace 或 GetUserWorkspace。

上表只描述 HuberLoss 主 Kernel 的 Tiling workspace。手写 L2 中的 Contiguous、Fill、Cast 或 ViewCopy 等辅助算子仍可能使 aclnnHuberLossGetWorkspaceSize 返回非 0；公开接口的总 workspace 始终以 executor 返回值为准。

#### Kernel 侧设计

##### 初始化

Kernel 根据 TilingKey 选择 none、mean 或 sum 模板实例，并由 DTYPE_INPUT 选择输入类型。Init 阶段完成：

1. 读取 totalNum、blockFactor、ubFactor、invNumel、delta，reductionMode 由编译期模板参数确定。
2. 根据 GetBlockIdx 计算当前核的 GM offset 和有效 blockLength。
3. 初始化 inputGM、targetGM、outputGM。
4. 仅多核 reduce 通过 SetSysWorkspace 和 GetUserWorkspace 初始化系统区与用户 partial 区；none/单核路径不访问 workspace。
5. 初始化 input/target 双缓冲队列、两个 FP32 中间 buffer；none 再初始化 output 双缓冲，reduction 仅初始化一个 32B 标量输出 buffer、partial buffer 和必要的多核汇总队列。

##### 逐元素计算

数学分段采用 `|e| < delta` 时二次、否则线性；在分段点两式数值相同。对有限 delta，Kernel 使用
`m=min(abs(e), delta)`、`loss=m*(abs(e)-0.5*m)` 的等价 clipped 公式避免 Compare/Select。每个
tile 执行：

1. CopyIn：通过 DataCopyPad 搬入 input 和 target。
2. Host 已把 delta 按输入 dtype RNE 量化。FP16 在源 dtype 逐步执行差值、绝对值、clipped、半值和差值计算，每一步保留源 dtype 舍入；BF16 采用 FP32 运算并在 CPU aten 会物化的位置做 BF16 RNE 往返；FP32 直接运算。
3. 有限 delta 路径计算 clipped 公式，不再生成 quadratic、linear、mask，也不执行 Compare/Select。
4. `delta=+Inf` 进入特殊位模式路径：有限 `abs(e)` 直接计算二次项；有限二次项溢出的 `+Inf` 与真正无限差值产生的 NaN 标记分开保存，输入 NaN 同样进入 NaN 标记；规约阶段的折叠、tile 累加和跨核汇总均传播该标记，避免依赖 A2 低精度指令对 `Inf-Inf`、NaN 和溢出 Cast 的平台差异。
5. FP16/BF16 逐元素 loss 均先物化为源 dtype；mean/sum 再转回 FP32 作为 reduction 输入，none 在 CopyOut 时写源 dtype。对于低精度大有限 delta，物化前的 FP32 loss 用于恢复 Cast 饱和掉的 `+Inf`/NaN 类别。大规模规约遵循默认误差阈值，不声称 bit-exact。

##### none 路径

1. 将 FP32 Huber 结果转换为输入 dtype。
2. CopyOut 到对应 GM offset。
3. 每核独立完成，不使用 workspace 和核间同步。

##### mean/sum 单核路径

1. 每个 tile 得到符合 CPU aten 逐元素物化语义的 FP32 reduction 输入。
2. 通过成对 Add 将 tile 折叠到一个 8-lane/32B FP32 block，并跨 tile 累加对应 lane；每个 tile 不再执行 ReduceSum。特殊路径使用标记感知的 Add，保证 NaN 语义不被平台算术吞掉。
3. 所有 tile 完成后仅对 8-lane partial 执行一次 ReduceSum；特殊路径先检查并保留 NaN 标记：
   - mean：localSum 乘 invNumel；
   - sum：保持 localSum。
4. 转换为输出 dtype，并由 0 号核写入唯一标量；最终 GM CopyOut 的有效字节数严格为 `sizeof(output dtype)`，不能复用 32B partial 槽位的整块写回参数。

##### mean/sum 多核路径

1. 每核独立完成逐元素计算，并把所有 tile 累加为一个 8-lane FP32 partial block；特殊路径同时保留有限溢出与真正无限差值的类别标记。
2. 每核通过一次 32B 对齐的 CopyOut 将完整 partial block 写入独立用户 workspace 槽位。
4. 使用 PipeBarrier&lt;PIPE_ALL&gt; 确保本核 DMA 写回完成，再执行 SyncAll。
5. 仅 0 号核读取所有 partial block，并对 `blockNum×8` 个 FP32 lane 执行一次最终 ReduceSum。
6. mean 模式乘 invNumel，sum 模式保持总和。
7. 结果转换为输出 dtype，写入 output[0]；最终标量 CopyOut 只写 `sizeof(output dtype)` 字节，避免越过 0 维单元素输出的存储边界。

该方案不使用 Global AtomicAdd，减少原子冲突和非确定累加顺序对精度的影响。

##### 尾块处理和流水

- 非 32B 对齐尾块由 DataCopyPad 补齐搬入，所有 Cast/Sub/Abs/Mins/Axpy/Mul/折叠操作只处理有效 `currentNum`，none 的 CopyOut 也只写回有效元素；
- none 保留 input、target、output 三条双缓冲队列；reduction 仅保留 input/target 双缓冲，output 为一个 32B buffer；
- reduction 在消费当前 tile 前预取下一 tile 到第二个队列槽，使 GM 读取与当前 tile 的矢量计算/折叠重叠；
- 通过 clipped 公式、预取和 8-lane 折叠消除了原逐 tile ReduceSum 和冗余中间计算瓶颈；strict_final2 的 9 个性能 Case 均越过 80% 门槛。特殊值分类和标记扫描位于 `+Inf`/大 delta 冷路径，不参与 `delta=1.0` 性能矩阵；有限 `delta=1.0` 的 12 个补充非有限/溢出场景已依靠普通热路径真机 12/12 通过，因此无需为它们新增分类指令。

##### Kernel 流程图

~~~mermaid
flowchart TD
    A["读取 TilingData 与模式"] --> B{"reduction=none?"}
    B -- 是 --> C["多核分片"]
    C --> D["循环 tile：CopyIn"]
    D --> E{"delta=+Inf?"}
    E -- 否 --> F["有限 delta：clipped 两 buffer 公式；低精度逐步舍入"]
    E -- 是 --> F1["+Inf 位模式路径：区分有限溢出、真 Inf 与 NaN"]
    F1 --> G
    F --> G["Cast 并 CopyOut"]
    G --> Z["结束"]

    B -- 否 --> H{"N 是否为空?"}
    H -- 是 --> I["mean 写 NaN / sum 写 0"]
    I --> Z
    H -- 否 --> J{"是否单核 fast path?"}
    J -- 是 --> K["预取下一 tile；按有限/+Inf 语义计算并量化 loss"]
    K --> L["成对 Add 折叠至 8 lanes；特殊路径保留标记"]
    L --> L1["最终一次 ReduceSum"]
    L1 --> M{"reduction=mean?"}
    M -- 是 --> M1["乘 invNumel"]
    M -- 否 --> M2["保持 sum"]
    M1 --> N["Cast 并写标量"]
    M2 --> N
    N --> Z

    J -- 否 --> O["每核循环 tile 并累加 8-lane partial/特殊标记"]
    O --> P["写入每核独立 32B partial"]
    P --> Q["PipeBarrier 后 SyncAll"]
    Q --> R{"blockIdx=0?"}
    R -- 否 --> Z
    R -- 是 --> S["汇总所有 partial"]
    S --> T{"reduction=mean?"}
    T -- 是 --> T1["乘 invNumel"]
    T -- 否 --> T2["保持 sum"]
    T1 --> U["Cast 并写标量"]
    T2 --> U
    U --> Z
~~~

### 实现文件清单

| 文件/目录 | 已落地内容或状态 |
| --- | --- |
| op_host/huber_loss_def.cpp | schema 已统一为 input/target/output，包含 reduction Int 属性和三组严格对齐的 dtype/format |
| op_host/huber_loss_infershape.cpp | 已按 reduction 推导同 shape 或 0 维标量，并拒绝非法值 |
| op_host/huber_loss_tiling.cpp | 已实现 reduction 解析、单核/多核 reduce、workspace、按 input dtype 量化 delta 和新 TilingData |
| op_host/op_api/aclnn_huber_loss.cpp/.h | 已实现两段式 L2 接口、严格参数校验、空 Tensor、Contiguous 和 ViewCopy |
| op_host/op_api/huber_loss.cpp/.h | 已实现 L0 调用封装，使用 input/target/output，固定属性顺序并加入 launcher list |
| op_host/CMakeLists.txt | 已纳入手写 op_api 源文件，并设置 `ACLNNTYPE aclnn_exclude` |
| op_kernel/huber_loss_tiling_data.h | 已定义 totalNum、blockFactor、ubFactor、invNumel、workspace 等字段 |
| op_kernel/huber_loss_tiling_key.h | 已使用仓库模板宏声明 none/mean/sum 三个 reduction key，并与 Host 选择保持一致 |
| op_kernel/huber_loss.cpp | 入口使用 input/target/output 与 DTYPE_INPUT，按 none/mean/sum TilingKey 实例化和分发 Kernel 模板 |
| op_kernel/huber_loss.h | 实现有限 delta clipped 公式、`+Inf` 位模式路径、有限溢出/真 Inf/NaN 分类与规约标记保留、FP16 源 dtype 逐步舍入、BF16 FP32+RNE 往返、低精度逐元素 loss 物化、8-lane 折叠与单核/多核 FP32 归约 |
| README.md | 已更新接口、reduction、空 Tensor、非连续和测试说明 |
| docs/aclnnHuberLoss.md | 已记录公开 reduction 参数和两段式接口契约 |
| examples | 已增加三种 reduction 的非连续输入调用示例 |
| tests/ut/op_host | 已增加 reduction 推形、非法属性、标量和空 Tensor 用例 |
| tests/ut/op_kernel | 已增加 none/mean/sum、单核/多核 reduce、生产入口、尾块、`delta=+Inf`、有限溢出与 FP16/BF16 CPU aten 严格逐元素语义回归，并新增跨四核特殊标记规约用例，11/11 通过 |
| examples、tests/README.md | 已提供 198 个精度、18 个负例、9 个 AscendOpTest、CPU aten 直接对齐和 9 组性能复现步骤及结果 |
| tests/ut/op_api | 已新增 L2 mock UT，覆盖三种 dtype/reduction、空 Tensor 构图、非连续 view 构图，以及 null、shape、dtype、reduction、delta 负例；mock 不替代真机数据搬运和数值验证 |
| pytorch/ 独立路由扩展（补充兼容） | 已完成 forward/out/backward/backward.out 四个 PrivateUse1 注册、wheel 构建与指定路径安装；路由 UT 9/9，msprof 前后向均在 NPU 执行，AICPU=0；该项不是任务书核心验收项 |

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas A2 训练系列产品（ascend910b） | √ |

本次按任务书适配并验证 Atlas A2（构建配置 ascend910b）。其他 SoC 未纳入本次验收与验证范围，不据此判定为不支持。

本任务测试基线为 CANN 9.0.0 及以上。

## 算子约束限制

- 只支持 FP32、FP16、BF16；
- 只支持 ND；
- input、target、output dtype 必须一致；
- input 与 target shape 必须一致；
- 不支持广播；
- reduction 仅支持 0、1、2；
- delta 必须大于 0；
- none 输出与输入同 shape；
- mean/sum 输出为 0 维标量；
- 有限 delta 使用 clipped 无分支公式；`delta=+Inf` 使用特殊位模式路径，区分有限二次项溢出的 `+Inf`、真正无限差值与输入 NaN，并在单核/多核规约中传播 NaN 标记；低精度大有限 delta 同样恢复 Cast 饱和前的特殊值类别；
- FP16 按源 dtype 逐步计算并舍入；BF16 以 FP32 运算加 BF16 RNE 往返复现源 dtype 物化；delta 按输入 dtype 量化；低精度逐元素 loss 物化后再以 FP32 规约，最终输出仍为原 dtype；大规模规约不承诺 bit-exact；
- 当前作为独立算子，不参与图融合；
- 动态 shape 在 Kernel 启动前必须具体化；

## 本次交付范围与 PyTorch 补充集成

- 本任务的 Ascend C 公共接口按任务书实现 HuberLoss 前向，参数为 input、target、reduction、delta、output；不新增 backward Kernel 或 weight 输入；
- 独立 PyTorch 扩展属于补充兼容能力（非任务书核心验收项）。导入 `huber_loss_extension` 后，路由层使用已有 NPU 算子完成 autograd backward；在已验证的 PyTorch 2.9 环境中，`F.huber_loss(weight=...)` 由 functional 层在无权重 HuberLoss 结果上组合乘法与归约完成，不改变 Ascend C 公共接口。

## 特性交叉分析

| 特性 | 分析 |
| --- | --- |
| 非连续 Tensor | input/target 通过 AutoContiguous 或 L2 Contiguous 处理；需验证辅助 Kernel 和整体 workspace |
| 动态 shape/rank | 动态 shape 在运行时具体化后由 Tiling 处理；动态 rank 支持仍需专门 UT/ST 验证 |
| 空 Tensor | none 返回空；sum 返回 0；mean 返回 NaN，优先在接口层快速处理 |
| 混合精度 | 公开接口不支持混合 dtype；FP16 逐步按源 dtype 舍入，BF16 以 FP32+BF16 RNE 往返保留 CPU aten 的物化语义，partial/workspace 使用 FP32 |
| 跨核同步 | 只用于大 shape mean/sum；需 SetScheduleMode、系统 workspace 和 SyncAll |
| PyTorch 路由（补充兼容） | 独立扩展已注册四个 PrivateUse1 schema；forward/out 调用 `aclnnHuberLoss`，backward 留在 NPU，profile 全流程 AICPU=0；进程中需先导入扩展完成注册 |
| 图融合 | 不涉及 |
| backward | Ascend C 核心任务不新增 backward Kernel；PyTorch 集成层用现有 NPU SmoothL1 backward 组合完成 autograd 路由；huber_loss_grad 仍为独立算子 |

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 功能标准 | 数学语义、reduction、shape、dtype、异常处理与任务书一致 | 社区任务书 |
| 精度标准 | 满足 AscendOpTest 默认阈值，并与 CPU aten::huber_loss 对齐 | 社区任务书 / AscendOpTest |
| 本仓执行路径标准 | ACLNN profiler 中进入 HuberLoss AICore，不走 AICPU | 任务目标 / ops-nn 责任边界 |
| PyTorch 补充兼容验证（非任务书核心验收项） | 四个 PrivateUse1 schema 已注册；导入扩展后，前向、out、autograd 不回退 CPU/AICPU | 业务补充目标；独立扩展真机验证通过 |
| 性能标准 | 达到 80% compute bound 或 memory bound；未达标场景逐项说明 | 社区任务书 |

功能、精度、边界、异常、执行路由和性能测量证据均已补齐。精度相关门槛通过；最终性能 9/9 达到 80% memory bound 门槛，原四个半精度规约未达项已关闭。

## 功能与精度测试设计

### 基础组合矩阵

三种 dtype 与三种 reduction 必须全组合覆盖：

| 维度 | 取值 |
| --- | --- |
| dtype | FP32、FP16、BF16 |
| reduction | 0=none、1=mean、2=sum |
| delta | 默认 1.0、自定义 0.1、0.5、2.0、较小正数、较大正数、正无穷 |
| shape | 标量、空 Tensor、1D、2D、4D、8D、小/中/大 shape |
| layout | 连续、input 非连续、target 非连续、两者均非连续 |
| 数值区域 | e=0、正负误差、abs(e)<delta、abs(e)=delta、abs(e)>delta、混合区域 |

### 建议精度 Case

| Case | dtype | reduction | shape | 目的 |
| --- | --- | --- | --- | --- |
| P01 | 三种 | 三种 | 标量 | 0 维输入输出语义 |
| P02 | 三种 | 三种 | [1] | 单元素与标量区别 |
| P03 | 三种 | none | [7] | 正负误差和分段边界 |
| P04 | 三种 | 三种 | [63]、[64]、[65] | 向量宽度和尾块 |
| P05 | 三种 | 三种 | [2,3,5] | 多维展平 |
| P06 | 三种 | 三种 | [17,33] | 非 32B 对齐和非均匀分核 |
| P07 | 三种 | 三种 | [32,128,256] | 多 tile 与多核 |
| P08 | 三种 | 三种 | [1,1,1,1,1,1,1,2] | 高 rank |
| P09 | 三种 | 三种 | [0]、[2,0,3] | 空 Tensor 语义 |
| P10 | 三种 | 三种 | 非连续 view | AutoContiguous 端到端 |
| P11 | 三种 | 三种 | 动态 shape 多次运行 | 运行时 Tiling 泛化 |
| P12 | 三种 | 三种 | 大 shape | FP32 累加与跨核汇总 |

Golden 使用 CPU aten::huber_loss。每个 Case 记录输入参数、CPU 输出、NPU 输出、最大绝对误差、最大相对误差、AscendOpTest 判定和日志/截图。

## 性能测试设计

### 测量环境

每次报告必须记录：

- SoC 完整型号；
- CANN 版本和驱动/固件版本；
- 算子代码提交；
- dtype、shape、reduction、delta；
- AIV 核数、UB 大小、TilingKey、blockNum、ubFactor；
- warmup 次数、测量次数和统计量；
- 主 Kernel 时间、端到端时间及辅助 Kernel 时间。

建议至少 warmup 10 次、正式测量 100 次，报告中位数，并保留 profiler 原始数据。

### 性能 Case

| Case | shape | reduction | 关注点 |
| --- | --- | --- | --- |
| T01 | [1]、[64]、[1024] | 三种 | 小 shape 启动开销与单核 fast path |
| T02 | [4096]、[65536] | none | 多核逐元素吞吐 |
| T03 | [4096]、[65536] | mean/sum | 单核与多核阈值 |
| T04 | [1,3,224,224] | 三种 | 常见训练 Tensor |
| T05 | [32,128,256] | 三种 | 多 tile 与大 shape |
| T06 | 非 32B 对齐大 shape | 三种 | 尾块影响 |
| T07 | 非连续输入 | 三种 | Contiguous 端到端开销 |
| T08 | [67108864] | 三种 dtype × 三种 reduction | 超过 192 MiB L2 的 80% roofline 主验收矩阵 |

### Bound 计算口径

none 路径的最低逻辑 GM 流量可按两次读取和一次写回估算：

$$
logicalBytes_{none}=3N \times sizeof(inputType)
$$

mean/sum 的主数据流量以两次输入读取为主，另加标量输出和多核 partial：

$$
logicalBytes_{reduce}
=2N \times sizeof(inputType)
+ I_{multiCore} \times 2 \times blockNum \times workspaceSlotBytes
+ sizeof(outputType)
$$

其中 I_multiCore 在多核归约时为 1，单核归约时为 0；系数 2 对应每核 partial 槽位的一次写入和 0 号核的一次读取。系统同步区流量未计入该逻辑下界。正式报告优先使用 profiler 的实际 GM 读写字节计算有效带宽，并同时列出逻辑口径，避免 workspace、重复搬运或辅助 Kernel 被漏计。

实际带宽利用率为：

$$
BandwidthUtilization =
\frac{logicalBytes/kernelTime}{measuredPeakBandwidth}
$$

性能矩阵使用有限 `delta=1.0`：FP32/BF16 clipped 路径的核心矢量操作为 Sub、Abs、Mins、Axpy 和 Mul，FP16 路径按源 dtype 分解等价步骤；低精度还包含语义所需的 Cast/RNE 往返，归约路径包含折叠 Add。`+Inf` 特殊路径不参与该性能矩阵。最终 FLOP 口径在报告中固定为 none 每元素 5 FLOPs、mean 每元素 6 FLOPs、sum 为 `6N-1`，并与性能解析脚本保持一致：

$$
ComputeUtilization =
\frac{definedFLOPs/kernelTime}{measuredPeakFLOPS}
$$

根据算术强度和平台 roofline 判断场景属于 memory bound 或 compute bound，再使用相应利用率与 80% 目标比较。不得用端到端时间和主 Kernel 时间混算，也不得在没有峰值来源和 profiler 数据时声明达标。

## 兼容性分析

当前实现已在原逐元素 HuberLoss 基线上补齐 reduction 和手写 ACLNN 公开接口，并保持合法 none 场景的分段公式不变。源码层面的完整兼容目标为：

- 数学结果与 CPU aten::huber_loss 对齐；
- reduction 数值映射与任务书一致；
- 输出 dtype 和 shape 与任务书一致；
- FP16 源 dtype 逐步舍入、BF16 FP32+BF16 RNE 往返、delta 量化和逐元素 loss 物化保留 CPU aten 语义；partial/workspace 使用 FP32 累加，大规模规约按默认阈值验收而不承诺 bit-exact；
- 非连续输入在接口层转换后保持相同逻辑结果；
- 空 Tensor 的 none/sum/mean 与 CPU 参考一致；
- 本任务已适配并验证 Atlas A2（ascend910b）；其他 SoC 未纳入本次验收与验证范围，不改变其既有路由。

## 风险与控制措施

| 风险 | 控制措施 |
| --- | --- |
| 源码完成与验收通过混淆 | 文档分别记录实现能力与实测证据；没有日志或 profiler 数据时不填写通过结论 |
| reduction 映射混乱 | 全链路统一 0=none、1=mean、2=sum，并增加属性 UT |
| 标量被实现为 [1] | ACLNN 校验 view rank=0，InferShape 使用空 Shape |
| FP16/BF16 累加误差及 CPU 语义差异 | FP16 源 dtype 逐步舍入、BF16 FP32+BF16 RNE 往返、delta 源 dtype 量化、逐元素 loss 源 dtype 物化后以 FP32 累加；`delta=0.7` 与 tiny delta 零容差回归、AscendOpTest 与 CPU aten 均通过 |
| 多核归约竞争 | 每核完整清零并写入独立对齐槽位，DMA 屏障和 SyncAll 后仅 0 号核汇总 |
| 空 mean 除 0 | 空 Tensor 快速写 NaN，不进入常规 1/N |
| workspace 越界 | Host 统一计算系统和用户 workspace，Kernel 使用相同每核槽位 |
| 非连续输入语义 | 三种 dtype 的 input-only、target-only、both view 已纳入 198 个真机 Case并通过 |
| 半精度规约性能风险 | 保留优化前 5/9 profiler；strict_final2 通过两 buffer 公式、预取和 8-lane 折叠将四个原 FAIL 场景提升至 81.29%–117.42%，最终 9/9 PASS；BF16 两个规约场景余量较窄，后续优化不得删除 CPU aten 所需 RNE 语义 |
| PrivateUse1 集成回归 | 固定检查四个 schema 注册、路由 UT、前后向 profiler 路径和 wheel 哈希；测试必须确认 CPU fallback 告警与 AICPU 任务数均为 0 |

## 参考资料

1. 任务书：8月社区任务 - huber_loss 算子开发任务书.md。
2. PyTorch Huber Loss 文档：https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.huber_loss.html
3. PyTorch Loss.cpp：https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/Loss.cpp
4. 同仓 reduction 参考：experimental/loss/mse_loss/docs/mse_loss_design.md。
5. 当前实现：experimental/loss/huber_loss/op_host、experimental/loss/huber_loss/op_kernel。
