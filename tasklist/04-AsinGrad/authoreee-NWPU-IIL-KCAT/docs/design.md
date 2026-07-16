# AsinGrad算子AscendC设计方案

## 需求背景

### 需求来源

依据CANN社区算子迁移任务要求，对历史TBE实现的AsinGrad算子进行AscendC重写，使其能够在Atlas A2/Ascend 910B环境下通过CANNJudge功能、精度和性能验证。

### 背景介绍

AsinGrad用于计算反正弦函数Asin的反向梯度，数学表达式为：

```text
z = dy / sqrt(1 - y * y)
```

其中`y`为前向输入，`dy`为上游梯度，`z`为输出梯度。原TBE实现采用DSL向量接口完成`vmul`、`vmuls`、`vadds`、`vsqrt`、`vdiv`等逐元素计算；本方案使用AscendC在AICore侧显式实现数据搬运、向量计算和结果写回。

AsinGrad算子TBE实现路径和相关API路径如下：

- AsinGrad算子实现路径为：`/usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_impl/ai_core/tbe/impl`
- AsinGrad算子实现中的API路径：`/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/tbe/dsl`


### AsinGrad算子现状分析

通过对AsinGrad历史TBE版本的功能分析，当前核心能力如下：

- 输入`y`、`dy`形状必须一致，输出`z`与输入保持相同shape。
- TBE历史版本主要支持`float16`、`float32`；CANNJudge任务模板扩展到`float16`、`float32`、`bfloat16`。
- `float16`路径在硬件支持时先转换为`float32`计算，再转换回`float16`，以降低`1 - y * y`接近0时的精度损失。
- 计算过程为纯逐元素操作，不涉及reduce、broadcast、shape变换或workspace依赖。

| 名称 | 类别 | dtype | format | shape | 介绍 |
| --- | --- | --- | --- | --- | --- |
| y | 输入 | fp16/fp32/bf16 | ND | all | Asin前向输入 |
| dy | 输入 | fp16/fp32/bf16 | ND | 同y | 上游梯度 |
| z | 输出 | fp16/fp32/bf16 | ND | 同y | 输出梯度 |

AsinGrad算子TBE版本的整体流程图如下图所示：
```mermaid
flowchart TD
    A["AsinGrad算子入口 asin_grad(y, dy, z, kernel_name)"] --> B["获取输入shape和dtype"]
    B --> C["check_shape(y), check_shape(dy)"]
    C --> D{"shape_y == shape_dy ?"}
    D -- 否 --> E["抛出shape不一致错误"]
    D -- 是 --> F["refine_shape_axes(shape_y / shape_dy)"]

    F --> G["check_dtype: float16 / float32"]
    G --> H{"dtype_y == dtype_dy ?"}
    H -- 否 --> I["抛出dtype不一致错误"]
    H -- 是 --> J["创建TVM placeholder: data_y, data_dy"]

    J --> K["调用 asin_grad_compute(data_y, data_dy, z)"]
    K --> L["生成计算表达式 res"]
    L --> M["with tvm.target.cce()"]
    M --> N["auto_schedule(res)"]
    N --> O["build(schedule, config)"]
    O --> P["生成TBE算子二进制"]
```

TBE Compute计算流程图
```mermaid
flowchart TD
    A["asin_grad_compute(y, dy, z)"] --> B["记录原始dtype = y.dtype"]
    B --> C{"dtype == float16 且支持float32向量API ?"}

    C -- 是 --> D["cast_to(y, float32)"]
    D --> E["cast_to(dy, float32)"]
    C -- 否 --> F["保持原dtype计算"]

    E --> G["vmul(y, y)"]
    F --> G

    G --> H["vmuls(data, -1)"]
    H --> I["vadds(data, 1)"]
    I --> J["vsqrt(1 - y*y)"]
    J --> K["vdiv(dy, sqrt_res)"]
    K --> L{"原始dtype == float16 ?"}
    L -- 是 --> M["cast_to(res, float16)"]
    L -- 否 --> N["直接输出res"]
    M --> O["返回结果z"]
    N --> O
```

接口具体实现图：
```mermaid
flowchart LR
    Y["输入 y"] --> C1{"float16路径?"}
    DY["输入 dy"] --> C1

    C1 -- 是 --> CY["cast_to(y, float32)"]
    C1 -- 是 --> CDY["cast_to(dy, float32)"]
    C1 -- 否 --> Y2["y保持原dtype"]
    C1 -- 否 --> DY2["dy保持原dtype"]

    CY --> MUL["vmul(y, y)"]
    Y2 --> MUL

    MUL --> MULS["vmuls(data, -1)"]
    MULS --> ADDS["vadds(data, 1)"]
    ADDS --> SQRT["vsqrt(1 - y*y)"]

    CDY --> DIV["vdiv(dy, sqrt_res)"]
    DY2 --> DIV
    SQRT --> DIV

    DIV --> C2{"原始dtype为float16?"}
    C2 -- 是 --> OUT16["cast_to(res, float16)"]
    C2 -- 否 --> OUT32["res保持float32"]
    OUT16 --> Z["输出 z"]
    OUT32 --> Z
```

数学表达式与TBE API对应关系：
```mermaid
flowchart TD
    A["目标公式: z = dy / sqrt(1 - y*y)"] --> B["y*y"]
    B --> B1["tbe.vmul(y, y)"]

    A --> C["-(y*y)"]
    C --> C1["tbe.vmuls(data, -1)"]

    A --> D["1 - y*y"]
    D --> D1["tbe.vadds(data, 1)"]

    A --> E["sqrt(1 - y*y)"]
    E --> E1["tbe.vsqrt(num_to_vrsqrt, 1)"]

    A --> F["dy / sqrt(...)"]
    F --> F1["tbe.vdiv(dy, vsqrt_res)"]

    A --> G["float16精度策略"]
    G --> G1["输入float16先cast_to float32计算"]
    G1 --> G2["最终cast_to float16输出"]
```

dtype分支逻辑图：
```mermaid
flowchart TD
    A["输入dtype"] --> B{"dtype == float16 ?"}
    B -- 是 --> C{"平台支持float32向量API ?"}
    C -- 是 --> D["y, dy 转float32"]
    D --> E["按float32执行 vmul/vmuls/vadds/vsqrt/vdiv"]
    E --> F["结果转回float16"]
    F --> G["输出z"]

    C -- 否 --> H["按float16执行计算"]
    H --> G

    B -- 否: float32 --> I["按float32执行计算"]
    I --> G
```


## 需求分析

### 外部组件依赖

不涉及外部组件依赖。

### 内部适配模块

适配Aclnn接口和图模式调用。


### 需求模块设计

#### 算子原型

- 原型设计：`AsinGrad(y, dy) -> z`。
- 相关约束：`y`、`dy`、`z` dtype保持一致；shape保持一致；format支持ND。
- Atlas A2训练系列产品/Atlas 800I A2推理产品支持`float16`、`float32`、`bfloat16`。

## 需求详细设计

### 使能方式

| 上层框架 | 涉及的框架勾选 |
| --- | --- |
| TF训练/推理 |  |
| Pytorch训练/推理 |  |
| ATC推理 | √ |
| Aclnn直调 | √ |
| OPAT调优 |  |

### 需求总体设计

#### host侧设计

AsinGrad计算过程不依赖原始维度的逐维信息，host侧将输入视为一维连续向量，仅关注总元素个数、数据类型、UB大小和AIV核数。Host侧通过`GetInputShape`获取`y`、`dy`的shape，检查输入输出shape一致；通过`GetInputDesc`/`GetOutputDesc`获取dtype并检查一致性。

**分核策略：**

- 优先使用可用AIV核，将`totalNum`按`coreNum`近似均分。
- 单核长度按cache line对应元素数向上对齐得到`blockFactor`。
- `usedCoreNum`由`totalNum`和`blockFactor`反推，避免启动空核。

**数据分块和内存优化策略：**

- 根据UB空间和kernel侧实际buffer数量计算`ubFactor`。
- FP32路径使用`y`、`dy`、`z`双缓冲队列以及`tmp`临时buffer。
- FP16/BF16路径使用低精度输入输出队列，并额外申请float32中间buffer完成升精度计算。
- 单次UB循环仅处理真实`currentNum`元素，尾块不对无效padding区域执行计算。

**tilingData规划：**

| 字段 | 含义 |
| --- | --- |
| totalNum | 展平后的总元素数 |
| blockFactor | 每个核处理的最大元素数 |
| ubFactor | 单次UB循环处理的元素数 |

**tilingKey规划：**

| schMode | 数据类型 | kernel分支 |
| --- | --- | --- |
| 0 | FP16 | `AsinGrad<half>` |
| 1 | FP32 | `AsinGrad<float>` |
| 2 | BF16 | `AsinGrad<bfloat16_t>` |

`schMode`使用2 bit声明，保证三种取值均可表达。

**数据检测：**

- shape约束：`y`、`dy`、`z`必须同shape。
- dtype约束：`y`、`dy`、`z`必须同dtype，支持`DT_FLOAT16`、`DT_FLOAT`、`DT_BF16`。
- format约束：支持ND格式。

#### kernel侧设计

Kernel侧分为`Init`和`Process`两个阶段，其中`Process`包括数据搬入`CopyIn`、计算`Compute`、搬出`CopyOut`三个阶段。

- `Init`阶段：根据`GetBlockIdx`获取当前核编号，结合`blockFactor`计算本核GM起始偏移和`blockLength`；初始化输入输出`GlobalTensor`以及UB队列/临时buffer。
- `CopyIn`阶段：使用`DataCopyPad`将`y`、`dy`从GM搬入UB队列。
- `Compute`阶段：FP32路径直接在float上计算；FP16/BF16路径先Cast到float32，完成`Mul`、`Muls`、`Adds`、`Sqrt`、`Div`后再Cast回原dtype。
- `CopyOut`阶段：将`z`从UB搬回GM，只写回当前tile真实元素数。
- 循环策略：每个核内部按`ubFactor`循环处理，尾块`currentNum`小于`ubFactor`时仅对真实元素执行向量计算，避免无效padding区域影响结果。

AscendC核心计算流程如下：

```text
CopyIn:   yGM, dyGM -> yLocal, dyLocal
Compute:  tmp = yLocal * yLocal
          tmp = 1.0 - tmp
          tmp = sqrt(tmp)
          zLocal = dyLocal / tmp
CopyOut:  zLocal -> zGM
```
Ascend C整体执行流程图：
```mermaid
flowchart TD
    A["调用 AsinGrad(y, dy)"] --> B["Host侧算子注册与校验"]
    B --> C["InferShape: z.shape = y.shape"]
    C --> D["TilingFunc"]
    D --> E["获取输入shape / dtype / format"]
    E --> F{"shape和dtype是否合法?"}
    F -- 否 --> G["返回GRAPH_FAILED"]
    F -- 是 --> H["获取平台信息: AIV核数 / UB大小"]

    H --> I["计算totalNum"]
    I --> J["计算blockFactor: 单核处理元素数"]
    J --> K["计算usedCoreNum并设置BlockDim"]
    K --> L["计算ubFactor: 单次UB处理元素数"]
    L --> M["设置TilingData: totalNum, blockFactor, ubFactor"]
    M --> N["根据dtype设置TilingKey"]

    N --> O["启动AICore Kernel asin_grad<schMode>"]
    O --> P["Kernel Init"]
    P --> Q["Kernel Process循环"]
    Q --> R["CopyIn: GM -> UB"]
    R --> S["Compute: dy / sqrt(1 - y*y)"]
    S --> T["CopyOut: UB -> GM"]
    T --> U{"本核数据处理完成?"}
    U -- 否 --> Q
    U -- 是 --> V["Kernel结束"]
```

Kernel侧Init流程图：
```mermaid
flowchart TD
    A["Kernel入口 asin_grad<schMode>"] --> B["读取TilingData"]
    B --> C["根据schMode选择模板类型"]
    C --> D["创建AsinGrad<T>对象"]
    D --> E["Init(y, dy, z, tilingData)"]

    E --> F["blockIdx = GetBlockIdx()"]
    F --> G["offset = blockFactor * blockIdx"]
    G --> H["remain = totalNum - offset"]
    H --> I{"remain <= 0 或 ubFactor <= 0 ?"}
    I -- 是 --> J["blockLength = 0, 返回"]
    I -- 否 --> K["blockLength = min(remain, blockFactor)"]
    K --> L["ubLength = ubFactor"]
    L --> M["设置yGM / dyGM / zGM GlobalTensor"]
    M --> N["初始化yQueue / dyQueue / zQueue"]
    N --> O{"T == float ?"}
    O -- 是 --> P["初始化tmpBuf"]
    O -- 否 --> Q["初始化yFp32Buf / dyFp32Buf / tmpBuf / zFp32Buf"]
```

Kernel侧Process流程图：
```mermaid
flowchart TD
    A["Process()"] --> B{"blockLength <= 0 ?"}
    B -- 是 --> C["直接返回"]
    B -- 否 --> D["loopCount = ceil(blockLength / ubLength)"]

    D --> E["for i in loopCount"]
    E --> F["currentNum = 当前tile真实元素数"]
    F --> G["CopyIn(i, currentNum)"]
    G --> H["Compute(currentNum)"]
    H --> I["CopyOut(i, currentNum)"]
    I --> J{"所有tile完成?"}
    J -- 否 --> E
    J -- 是 --> K["Process结束"]
```


### 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| 香橙派OrangePi AIpro |  |
| Atlas 200I/500 A2推理产品 |  |
| Atlas 800I/T A2 | √ |

### 算子约束限制

不支持广播；不支持`y`、`dy` dtype不一致；不支持非ND格式；不支持复数、整数、bool等非浮点输入。输入值通常应位于`[-1, 1]`范围内，超出定义域时结果遵循硬件`sqrt/div`对NaN或Inf的处理行为。

## 特性交叉分析

AsinGrad为逐元素数学算子，与reduce、broadcast、layout转换、原子写、多输出等特性无交叉依赖。算子不使用workspace，不依赖外部状态；可作为图模式或aclnn直调路径中的基础反向算子。

## 可维可测分析

### 精度标准/性能标准

| 验收标准 | 描述(不涉及说明原因) | 标准来源 |
| --- | --- | --- |
| 精度标准 | 不低于TBE版本。 | |
| 性能标准 | 不低于TBE版本。 | |

测试建议覆盖以下场景：

- 基础shape：一维、二维、多维、标量shape。
- 数据类型：`float16`、`float32`、`bfloat16`。
- 规模场景：小shape、单核、满核、多tile、大shape和尾块。
- 数值场景：普通随机值、接近0、接近±1、`dy`包含正负值。

### 兼容性分析

新算子，不涉及历史二进制兼容。
