# InplaceSigmoid 算子开发设计文档

## 一、需求背景

### 1.1 需求来源

通过社区任务完成昇腾 CANN 算子开源仓贡献需求，丰富昇腾硬件上对神经网络激活函数数据类型的泛化支持。

### 1.2 背景介绍

#### 1.2.1 InplaceSigmoid 算子实现优化

本次任务要求参考昇腾内置 TBE 算子 `Sigmoid`，在 Atlas A2/A3 训练系列产品上基于 Ascend C 编程语言实现功能一致的 `aclnnInplaceSigmoid` 算子。

* **主要扩展：** 原算子仅支持浮点类型，本次需扩展支持 `int16`, `int8`, `uint8` 数据类型，并采用就地（Inplace）运算直接更新原 Tensor。
* **TBE 参考路径：** `/usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_impl/ai_core/tbe/impl/dynamic/sigmoid.py`

#### 1.2.2 InplaceSigmoid 算子现状分析

* **1.2.2.1 TBE 算子支持的数据类型和数据格式**
根据 TBE 源码及原型定义，原 `Sigmoid` 算子仅支持 `float16`, `float32`, `bfloat16` 数据类型。支持的数据格式为 `ND`。
* **1.2.2.2 TBE 算子实现描述**
原 TBE 算子的实现具有高度的硬件分支特征，针对不同计算平台应用了不同的算法：
1. **Ascend310P (高性能模式)：** 采用泰勒展开式逼近 $L(x) = ax + bx^3 + cx^5 + d$，然后结合极大极小值截断。
2. **Ascend v200/v300 (Tiny/Nano)：** 采用极值安全转换公式 $y = \frac{e^{\min(0, x)}}{1 + e^{-|x|}}$ 防溢出。
3. **Ascend910B / Ascend910_93 (本次目标硬件)：** 执行标准计算公式 $y = \frac{1}{1 + e^{-x}}$。在高精度（HIGH_PRECISION）模式下，如果存在特定的运算（如非卷积），会将 `float16`/`bfloat16` 转换为 `float32` 进行运算，完成后转回原类型。


* **1.2.2.3 TBE 算子实现流程图**

![image.png](https://raw.gitcode.com/user-images/assets/9516645/2012147d-e63f-4ad5-b82e-3f25635a77dd/image.png 'image.png')

---

## 二、需求分析

### 2.1 外部组件依赖

需要适配算子调用框架 ACLNN。

### 2.2 内部适配模块

算子内部需完成 Ascend C Host 侧的 Tiling 切分模块设计（`op_host/sigmoid_tiling.cpp`）和 Kernel 侧运算模块（`op_kernel/sigmoid.cpp`），并更新相应算子信息库。

### 2.3 需求模块设计

#### 2.3.1 Ascend C 算子原型

| 参数名 | 输入/输出/属性 | 描述 | 使用说明 | 数据类型 | 数据格式 | 维度(shape) | 非连续Tensor |
| --- | --- | --- | --- | --- | --- | --- | --- |
| self | 输入 | 待进行Sigmoid计算的入参。 | 支持空Tensor，shape需要与out一致。 | FLOAT、FLOAT16、BFLOAT16、INT16、INT8、UINT8 | ND | 0-8 | √ |
| out | 输出 | 计算的出参。 | shape需要与self一致（Inplace实现实际操作同一内存）。 | FLOAT、FLOAT16、BFLOAT16、INT16、INT8、UINT8 | ND | 0-8 | √ |

#### 2.3.2 Ascend C 算子相关约束

* 算子默认使用确定性实现。
* 对新增整型数据的处理需严格遵循指定转换逻辑：`int16` 转为 `float32` 计算，`int8 / uint8` 转为 `float16` 计算。转回时需考虑四舍五入或截断等常规 Cast 行为以对齐框架。

---

## 三、需求详细设计

### 3.1 使能方式

根据实际开发任务，使用 ACLNN 框架调用算子。

### 3.2 需求总体设计

#### 3.2.1 Host 侧设计

* **3.2.1.1 分核策略**
获取输入数据的总元素个数，基于物理设备可用的 AI Core 数量和硬件 UB (Unified Buffer) 内存上限，计算每个核处理的数据总量 (`coreDataNum`)。将任务均匀分配至多核，末尾多出的零散数据分配给最后一个核作为 `tailDataNum`。
* **3.2.1.2 数据分块和内存优化策略**
在每个 AI Core 内部，基于 Double Buffer 机制将当前核的数据拆解为 `tileDataNum`。分别申请用于数据流入 (`inQueueX`) 和流出 (`outQueueY`) 的缓存，同时针对需要数据类型转换的场景（如 `int16`、`int8` 等），需额外申请 `TBuf` 充当中间高精度计算类型缓存（如 `float32`、`float16` 缓存区），以避免直接覆盖源数据内存导致错乱。
* **3.2.1.3 tilingKey 规划策略**
由于新增了整型，不同数据类型的中间态 Cast 路径差异极大。依据输入 Tensor 的 `dtype` 生成对应的 `tilingKey`，以便 Kernel 侧直接实例化对应数据类型的计算模板。

#### 3.2.2 Kernel 侧设计

* **3.2.2.1 Kernel 侧实现描述**
Kernel 层面的处理流程高度专注于 **Atlas A2/A3** 架构，舍弃 TBE 中的其它历史遗留硬件分支，聚焦于公式 $y = \frac{1}{1 + e^{-x}}$。
针对不同类型的 Cast 策略设计如下：
1. **FLOAT32:** 原生直接计算。`Muls(x, -1.0)` $\rightarrow$ `Exp(x)` $\rightarrow$ `Adds(x, 1.0)` $\rightarrow$ `Div(1.0, x)`。
2. **FLOAT16 / BFLOAT16:** 参照 TBE 高精度模式，提取数据 `Cast` 到 `float32` 向量中完成上述计算，再 `Cast` 返回原类型。
3. **INT16 (新增):** `Cast` 为 `float32` 后进行计算，计算完成后 `Cast` (使用 `RoundMode::CAST_RINT` 或框架要求的转整型模式) 回 `int16`。
4. **INT8 / UINT8 (新增):** `Cast` 为 `float16` 后进行计算，完成后转回对应整型。
5. **Inplace 操作:** 在 CopyOut 阶段，直接覆盖最初传入的 Global Memory 地址 `selfGm`。


* **3.2.2.2 Ascend C 实现流程图**

![image.png](https://raw.gitcode.com/user-images/assets/9842739/997c89c0-8ecc-4978-bd14-2d87487a0a9a/image.png 'image.png')

* **3.2.2.3 Ascend C 实现流程图与 TBE 流程图存在的差异点和原因**
* **差异点：** 1. 抛弃了基于 Ascend 310P/310 的 `vabs`、`min` 极值处理和泰勒展开式的复杂分支判断，仅保留标准 Sigmoid 计算。
2. 引入了大量的源头数据到计算精度的转换路线 (`int16->fp32`, `int8->fp16` 等)。
* **原因：** 本次开发明确指定适配硬件为 Atlas A2/A3（对应的就是原 TBE 中的 `Ascend910B` 及其衍生型号），底层指令自带 `Inf`/`NaN` 支持，性能强劲，无需老旧硬件的防溢出 hack 代码。同时由于泛化了多种整型，必须通过 Ascend C 的 `Cast` 接口进行类型升降级以利用浮点指令管线。



### 3.3 支持硬件

适配 **Atlas A2 训练系列产品 / Atlas A3 系列产品**。

### 3.4 算子约束限制

* 支持 0-8 维（ND）及非连续 Tensor。
* Inplace 特性要求调用链路确保输入输出共享同一块内存块，算子内不额外申请外存。

---

## 四、特性交叉分析

该算子为纯 Element-wise 的原地更新激活函数运算，输入输出的 Shape 与 Stride 特性在执行中保持不变。对于非连续 Tensor，依靠 ACLNN 框架下发的一维展平偏移量或循环迭代进行读取，不引起网络结构层次的其他特性交叉冲突。

---

## 五、可维可测分析

### 5.1 精度标准/性能标准

* **精度：** 需满足 `AscendOpTest` 的默认阈值。新增的 `int16`, `int8`, `uint8` 的输入输出结果，需与 PyTorch 参考实现（由于其转浮点计算后再强制转整型会有精度舍入）完全对齐。
* **性能：** 在所有核参与计算的前提下，性能需达到原 TBE 的 95% 以上。小于 10us 的场景需提供仿真图作为调优依据。

### 5.2 兼容性分析

本实现采用 Ascend C 进行标准重构，作为原有 TBE 的功能超集（增加了整数泛化支持），向后完全兼容已有模型的 Sigmoid 调用场景，对现有模型业务运行无破坏性影响。