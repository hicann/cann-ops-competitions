# InplaceRsqrt 算子开发设计文档

## 一、需求背景

### 1.1 需求来源

通过社区任务完成开源仓算子贡献的需求（昇腾 CANN 算子社区开源任务）。

### 1.2 背景介绍

#### 1.2.1 InplaceRsqrt算子实现优化

本次任务要求参考昇腾内置 TBE 算子 `Rsqrt`，在昇腾 NPU 上基于 Ascend C 实现功能一致的 `aclnnInplaceRsqrt` 算子。

* **TBE kernel 源文件获取路径：** `/usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_impl/ai_core/tbe/impl/dynamic/rsqrt.py`
* **算子信息库路径：** `/usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_impl/ai_core/tbe/config/ascend910b/`
* **算子原型路径：** `/usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_proto/inc/`

#### 1.2.2 InplaceRsqrt算子现状分析

* **1.2.2.1 TBE算子支持的数据类型和数据格式**
根据 TBE 源码及算子信息库，原 `Rsqrt` 算子仅支持以下浮点数据类型：`float16`, `float32`, `bfloat16`。支持的数据格式为 `ND`。
* **1.2.2.2 TBE算子实现描述**
原 TBE 算子实现逻辑为：计算 $y = \frac{1}{\sqrt{x}}$。具体过程为：
1. 检查输入数据类型，若为 `float16` 且平台支持 `float32` 的 `vadd` 操作，则将输入先转换为 `float32`（提升计算精度）。
2. 调用 `tbe.vsqrt` 接口计算平方根。
3. 调用 `tbe.broadcast` 生成与输入 shape 相同的全 `1.0` 张量。
4. 调用 `tbe.vdiv` 接口，将全 1 张量除以平方根结果。
5. 若原始输入为 `float16`，则将结果转换回 `float16` 输出。


* **1.2.2.3 TBE算子实现流程图**

![image.png](https://raw.gitcode.com/user-images/assets/9516645/6c45ab9d-9a6c-4854-a06a-1809b770cd53/image.png 'image.png')

---

## 二、需求分析

### 2.1 外部组件依赖

根据实际开发任务，需适配算子调用框架，当前主要外部依赖为 **ACLNN** 框架。

### 2.2 内部适配模块

算子内部需适配 Ascend C 的 Host 侧 Tiling 模块（`op_host/rsqrt_tiling.cpp`）与 Kernel 侧实现模块（`op_kernel/rsqrt.cpp`），并更新相关算子信息库与 API 注册模块。

### 2.3 需求模块设计

* **2.3.1 AscendC算子原型**
除对齐原 TBE 算子的 `float16`, `float32`, `bfloat16` 外，需新增支持 `bool`, `int8`, `int16`, `uint8`, `int32`。原型定义中，`selfRef` 既作输入也作输出，具体规格如下：

| 参数名 | 输入/输出/属性 | 描述 | 使用说明 | 数据类型 | 数据格式 | 维度(shape) | 非连续Tensor |
| --- | --- | --- | --- | --- | --- | --- | --- |
| selfRef (aclTensor*) | 输入/输出 | 公式中的 input，并原地存放 output | 支持空Tensor。 | float32、float16、bfloat16、bool、int8、int16、uint8、int32 | ND | 0-8 | √ |

* **2.3.2 AscendC算子相关约束**
* 需严格对齐 PyTorch 中 Rsqrt 的算子行为。
* 针对输入为 `0` 产生的 `inf` 值，依赖底层硬件的 `Cast` 指令（`CAST_TRUNC` 等）完成从浮点 `inf` 到整型的标准转换/截断行为，以完全对齐框架表现。



---

## 三、需求详细设计

### 3.1 使能方式

通过 ACLNN 进行使能调用。

### 3.2 需求总体设计

#### 3.2.1 host侧设计

* **3.2.1.1 分核策略**
获取输入 Tensor 的总数据量，根据硬件申请的 UB (Unified Buffer) 空间上限，计算出每个核最多能处理的数据量 (`coreDataNum`)。将总任务均匀分配到多个 AI Core 上，无法整除的尾部数据 (`tailDataNum`) 单独分配给末尾 Core 处理。
* **3.2.1.2 数据分块和内存优化策略**
在单核内，将 `coreDataNum` 继续划分为适合指令流水线的 `tileDataNum`。基于双缓冲（Double Buffer）机制，将内存划分为两块以实现计算（Vector）与搬运（MTE）指令并发。
* **3.2.1.3 tilingKey规划策略**
不同数据类型（特别是在处理新增整型类型和 bool 时）需要不同的流水线缓存区（如 `TBuf` 充当浮点转换中间态）及 Cast 策略，将数据类型映射为不同的 `tilingKey`，并在 Kernel 侧根据 `tilingKey` 实例化对应的泛化模板。

#### 3.2.2 kernel侧设计

* **3.2.2.1 kernel侧实现描述**
对于 `selfRef` (Inplace) 输入，Kernel 侧直接在流水线中读取并写回同一全局内存地址。
1. **FLOAT32:** 直接计算 $1 / \sqrt{x}$ (`Sqrt` -> `Div`)。
2. **FLOAT16 / BFLOAT16:** 提精计算。先 `Cast` 为 `float32`，计算 $1 / \sqrt{x}$ 后，通过 `CAST_RINT` 转回。
3. **BOOL:** 恒定输出。获取输入后将内存重解释为 `int16_t`，通过 `Duplicate` 填充 `0x0101`（等效全 1），直接输出。
4. **INT8 / INT16 / UINT8 / INT32:** 统一步骤。将源数据 `Cast` 至对应的浮点数（如 `int8/int16/uint8` 转 `half`，`int32` 转 `float`），计算完毕后，直接利用 `CAST_TRUNC` 截断模式转回原整型。输入为 0 产生的 `inf` 依赖 `Cast` 指令自动处理。


* **3.2.2.2 AscendC实现流程图**

![image.png](https://raw.gitcode.com/user-images/assets/9516645/d49879a9-ac21-481b-98ef-368db22b9b93/image.png 'image.png')

* **3.2.2.3 AscendC实现流程图与TBE流程图存在的差异点和原因**
**差异点：** Ascend C 实现在 TBE 的基础上增加了大量的数据类型判断和 `Cast` 逻辑。
**原因：** 本次开发针对 `InplaceRsqrt` 大幅增加了泛化数据类型支持。为保证算子在 NPU 上的高性能，整型需转换为浮点型才能使用硬件的高效 `Sqrt/Div` 指令；同时，遵循框架级行为（如 PyTorch 对齐），依靠硬件原生的 `Cast` 截断逻辑。

### 3.3 支持硬件

要求与《算子任务书》保持一致：支持 **Atlas A2 训练系列产品 / Atlas A3 系列产品**。

### 3.4 算子约束限制

* 支持空 Tensor 场景。
* 支持 0-8 维（ND）及非连续 Tensor。
* 由于为 Inplace 算子，输入/输出数据的底层存储空间复用，外层框架调用时须保证物理内存地址一致。

---

## 四、特性交叉分析

本算子为基础数学类逐点（Element-wise）运算算子，不涉及复杂的网络结构与维度变化，对其他特性的交叉影响极小。

---

## 五、可维可测分析

### 5.1 精度标准/性能标准

* **精度标准：** 计算精度不得低于原 TBE 算子，且需满足 `AscendOpTest` 工具默认阈值。新增的整型支持部分，需验证 `0` 作为输入时，其底层 `inf` 转整型的截断行为是否与 PyTorch 基准完全对齐。
* **性能标准：** 在所有核参与计算场景下，性能不得低于原 TBE 算子。若小 Shape（10us 以下场景）差距超 3us，需提供仿真图佐证并给出分析结论。

### 5.2 兼容性分析

本实现基于 Ascend C 标准接口进行开发，并注册至 ACLNN。算子逻辑是原 TBE `Rsqrt` 的超集，不仅向下兼容原有浮点场景的模型，也安全无缝地扩展了对新数据类型的支持，对外部框架高度兼容。