## 需求背景

### 需求来源

- 通过社区任务完成开源仓算子贡献的需求。

### 背景介绍

算子功能：实现带权重衰减的 Adam 优化器融合算子，用于训练场景中的参数更新。该算子将二阶矩更新、一阶矩更新以及带 decay 的参数更新三段计算融合为一个算子，减少中间张量落盘与访存次数，提升训练性能。

计算公式：

- output0 = input1 × const_mul2_x + input0² × const_mul3_x
- output1 = input2 × const_mul_x + input0 × const_mul1_x
- output2 = input3 - (output1 ÷ (sqrt(output0) + add2_y) + input3 × const_mul4_x) × input4

其中：

- input0：梯度，源码中的 `data_grad`
- input1：二阶矩历史值，源码中的 `data_v`
- input2：一阶矩历史值，源码中的 `data_m`
- input3：待更新参数，源码中的 `data_var`
- input4：步长或学习率缩放项，源码中的 `data_input4`
- const_mul_x ~ const_mul4_x：各阶段乘法系数
- add2_y：分母中的稳定项
- output0：更新后的二阶矩估计
- output1：更新后的一阶矩估计
- output2：带 decay 的参数更新结果

1. `adam_apply_one_with_decay` 算子实现信息
   - kernel实现：
      - /usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_impl/ai_core/tbe/impl/ops_legacy/dynamic/adam_apply_one_with_decay.py
   - 算子原型：
      - /usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_proto/inc/elewise_calculation_ops.h
   - 算子信息库：
      - /usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_impl/ai_core/tbe/config/ascend910b/aic-ascend910b-ops-info.json

2. `adam_apply_one_with_decay` 算子现状分析
   1. TBE 算子支持方式
      - 当前 Python 源码为动态 shape 实现。
      - 通过 `classify(dynamic_inputs, OpPatternMode.ELEWISE_WITH_BROADCAST)` 支持逐输入广播分类。
      - 当前源码未实现单独的 `op_select_format`，方案按通用 ND 广播场景设计。
      - `@register_operator_compute(..., support_fusion=True, support_bfp16=True)` 表明该实现支持融合，并声明支持 `BFLOAT16`。
   2. TBE 算子实现描述
![image.png](https://raw.gitcode.com/user-images/assets/7665709/a8222c0c-3a35-4344-8b04-349c8c783d72/image.png 'image.png')
   3. TBE 算子实现流程图
      - `adam_apply_one_with_decay` 算子 TBE 版本的整体流程如下图所示：
![mermaid-1776914847923.png](https://raw.gitcode.com/user-images/assets/7665709/4e58f5b9-043e-485a-8d09-5fa846e5eded/mermaid-1776914847923.png 'mermaid-1776914847923.png')

## 需求详细设计

### 使能方式

创建 `aclnnAdamApplyOneWithDecay` 相关接口，调用此算子。

### 需求总体设计

#### host 侧设计

1. 分核策略
   1. 优先采用满核分配策略，尽可能提升并行度。
   2. 当总数据量可以被核数整除时，按均匀分块处理。
   3. 当存在尾块时，将余数分配给前若干个核或最后一核统一处理。
   4. 当输入规模较小时，可退化为单核执行，减少调度开销。

2. 数据分块与内存规划策略
   1. 以单核可容纳的 UB 数据量为基础，沿连续维切块。
   2. 算子中间结果较多，需要规划输入缓冲、中间结果缓冲与输出缓冲。
   3. 由于 `output2` 依赖 `output0` 与 `output1`，建议按依赖顺序在 UB 中复用临时缓冲：
      - 第一阶段生成 `output0`
      - 第二阶段生成 `output1`
      - 第三阶段基于 `output0/output1` 继续完成 `output2`
   4. 搬运数据时需要满足 32 Byte 对齐要求。
   5. 若硬件与切块条件允许，可开启 double buffer 提升流水效率。

3. tilingKey 规划策略
   1. 由于源码侧支持广播，AscendC 方案建议区分两类场景：
      - `tilingKey = 0`：所有输入 shape 一致，无广播，走高效逐块计算路径
      - `tilingKey = 1`：存在广播关系，走带广播展开的通用路径
   2. 若本次 AscendC 落地阶段仅覆盖同 shape 场景，可先实现 `tilingKey = 0`，后续再补齐广播路径。

#### kernel 侧设计

1. kernel 侧实现描述
   - kernel 侧建议拆分为 `Init`、`Process` 两大阶段。
   - `Process` 内部进一步拆分为 `CopyIn`、`Compute`、`CopyOut`。
   - 需要从 GM 搬入 11 个输入张量：
     - `input0`、`input1`、`input2`、`input3`、`input4`
     - `const_mul_x`、`const_mul1_x`、`const_mul2_x`、`const_mul3_x`、`const_mul4_x`
     - `add2_y`
   - 计算顺序建议与 TBE 保持一致：
     1. **计算 output0**
        - `tmp0 = input0 × input0`
        - `tmp1 = tmp0 × const_mul3_x`
        - `tmp2 = input1 × const_mul2_x`
        - `output0 = tmp1 + tmp2`
     2. **计算 output1**
        - `tmp3 = input2 × const_mul_x`
        - `tmp4 = input0 × const_mul1_x`
        - `output1 = tmp3 + tmp4`
     3. **计算 output2**
        - `tmp5 = sqrt(output0)`
        - `tmp6 = tmp5 + add2_y`
        - `tmp7 = output1 ÷ tmp6`
        - `tmp8 = input3 × const_mul4_x`
        - `tmp9 = tmp7 + tmp8`
        - `tmp10 = tmp9 × input4`
        - `output2 = input3 - tmp10`
   - 对 `float16` 输入，建议在开方和除法链路上按硬件能力选择是否升精度到 `float32` 计算，以对齐 TBE 行为。
   - 若输入是 `bfloat16`，建议统一转换为 `float32` 完成计算，再在输出阶段回写为原始类型。
   - 若后续需要支持广播场景，kernel 需补充：
     - 输入索引展开
     - 广播维步长计算
     - 标量/长度为 1 维度的重复读或向量广播

2. AscendC 实现流程图
   - `adam_apply_one_with_decay` 算子流程见下图。

```mermaid
graph TD
    A["开始"] --> B["从 GM 搬入 11 个输入到 UB"]
    B --> B1["input0 ~ input4"]
    B --> B2["const_mul_x ~ const_mul4_x, add2_y"]

    B1 & B2 --> C{"输入类型是否需要升精度"}
    C -->|bf16 或 float16 特定链路| D["转换到 float32 计算路径"]
    C -->|否| E["保持原始类型"]

    D --> F["计算 output0"]
    E --> F

    F --> F1["tmp0 = input0 × input0"]
    F1 --> F2["tmp1 = tmp0 × const_mul3_x"]
    F2 --> F3["tmp2 = input1 × const_mul2_x"]
    F3 --> F4["output0 = tmp1 + tmp2"]

    F4 --> G["计算 output1"]
    G --> G1["tmp3 = input2 × const_mul_x"]
    G1 --> G2["tmp4 = input0 × const_mul1_x"]
    G2 --> G3["output1 = tmp3 + tmp4"]

    G3 --> H["计算 output2"]
    H --> H1["tmp5 = sqrt(output0)"]
    H1 --> H2["tmp6 = tmp5 + add2_y"]
    H2 --> H3["tmp7 = output1 ÷ tmp6"]
    H3 --> H4["tmp8 = input3 × const_mul4_x"]
    H4 --> H5["tmp9 = tmp7 + tmp8"]
    H5 --> H6["tmp10 = tmp9 × input4"]
    H6 --> H7["output2 = input3 - tmp10"]

    H7 --> I{"是否执行过类型提升"}
    I -->|是| J["输出转换回原始类型"]
    I -->|否| K["跳过类型回转"]

    J --> L["将 output0/output1/output2 搬出到 GM"]
    K --> L
    L --> M["结束"]
```

#### AscendC 实现流程图与 TBE 流程图存在的差异点和原因

1. TBE 源码显式支持广播分类，而 AscendC 首版可优先实现同 shape 路径，后续再扩展广播逻辑。
2. TBE 依赖框架自动调度与表达式图构建；AscendC 需要显式设计分核、切块、流水和 UB 复用策略。
3. TBE 通过 Python 构图支持动态 shape，多组 shape 组合会分别生成 schedule；AscendC 侧通常通过 tiling 与运行时参数控制覆盖不同输入规模。

### 支持硬件

| 产品                                                                            | 是否支持 |
| :------------------------------------------------------------------------------ | :------: |
| <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term> |    √     |

### 算子原型

| 参数名        | 类别     | 描述                                   | 数据类型                     | 数据格式 | shape |
| :------------ | :------- | :------------------------------------- | :--------------------------- | :------ | :---- |
| input0        | 输入张量 | 梯度输入，用于平方项和一阶矩更新。     | BFLOAT16、FLOAT16、FLOAT32   | ND      | all   |
| input1        | 输入张量 | 二阶矩历史值，用于计算 output0。       | BFLOAT16、FLOAT16、FLOAT32   | ND      | all   |
| input2        | 输入张量 | 一阶矩历史值，用于计算 output1。       | BFLOAT16、FLOAT16、FLOAT32   | ND      | all   |
| input3        | 输入张量 | 待更新参数，用于 decay 和最终减法。    | BFLOAT16、FLOAT16、FLOAT32   | ND      | all   |
| input4        | 输入张量 | 步长或学习率缩放项。                   | BFLOAT16、FLOAT16、FLOAT32   | ND      | all   |
| const_mul_x   | 输入张量 | `mul_0` 的乘数张量。                   | BFLOAT16、FLOAT16、FLOAT32   | ND      | all   |
| const_mul1_x  | 输入张量 | `mul_1` 的乘数张量。                   | BFLOAT16、FLOAT16、FLOAT32   | ND      | all   |
| const_mul2_x  | 输入张量 | `mul_2` 的乘数张量。                   | BFLOAT16、FLOAT16、FLOAT32   | ND      | all   |
| const_mul3_x  | 输入张量 | `mul_3` 的乘数张量。                   | BFLOAT16、FLOAT16、FLOAT32   | ND      | all   |
| const_mul4_x  | 输入张量 | decay 分支 `mul_4` 的乘数张量。        | BFLOAT16、FLOAT16、FLOAT32   | ND      | all   |
| add2_y        | 输入张量 | 分母稳定项加数张量。                   | BFLOAT16、FLOAT16、FLOAT32   | ND      | all   |
| output0       | 输出张量 | 更新后的二阶矩估计。                   | BFLOAT16、FLOAT16、FLOAT32   | ND      | all   |
| output1       | 输出张量 | 更新后的一阶矩估计。                   | BFLOAT16、FLOAT16、FLOAT32   | ND      | all   |
| output2       | 输出张量 | 带权重衰减的参数更新结果。             | BFLOAT16、FLOAT16、FLOAT32   | ND      | all   |

### 算子约束限制

1. 所有输入 dtype 需保持一致，输出 dtype 与输入 dtype 保持一致。
2. 当前 AscendC 方案建议优先支持所有输入 shape 一致的场景。

### 精度标准/性能标准

1. 精度不低于 TBE 实现。
2. 性能不低于 TBE 实现。
