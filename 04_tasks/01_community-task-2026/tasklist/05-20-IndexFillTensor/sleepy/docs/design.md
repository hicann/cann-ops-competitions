# IndexFillTensor 算子开发设计文档

## 一、需求背景

### 1.1 需求来源

当前 `aclnnIndexFillTensor` 算子不支持 `int16`, `int8`, `uint8`, `double` 数据类型。本次任务旨在基于现有的 Ascend C `IndexFillD` 算子代码进行二次开发，扩展这四种数据类型的支持，以满足更广泛的模型泛化与底层算力调度需求，并将成果贡献至昇腾 CANN 算子开源仓。

### 1.2 背景介绍

#### 1.2.1 算子实现优化

本次开发的核心任务是在 Ascend C 侧补充对上述新数据类型的支持，并供 `aclnnIndexFillTensor` 接口调用。由于底层 NPU 对不同数据类型的指令支持差异，新增类型需要特定的类型转换策略（如小整型转 `half` 计算，int16转为`float`计算，`double` 降级切分运算）。

#### 1.2.2 IndexFillD 算子现状分析 (基于 Ascend C 现有实现)

`aclnnIndexFillTensor` 是 OPAPI 基于算子 `IndexFillD` 封装的 `aclnn` 接口。当前 Ascend C 的 `IndexFillD` 算子主要依靠 `assist1`（保留位置为 1，替换位置为 0）和 `assist2`（保留位置为 0，替换位置为 `value`）两个辅助张量来完成核心逻辑。
当前代码支持 `bfloat16`, `float`, `half`, `int32`, `bool`（内部通过 `int8_t` 实例化），其底层实现逻辑分支如下：

1. **bfloat16_t 分支：** 将数据 `Cast` 到 `float`，通过 `CompareScalar` 判断 `assist1 > 0.0` 生成掩码，调用 `Select` 进行元素选择，最后 `Cast` 截断回 `bfloat16_t`。
2. **float / half 分支：** 直接使用 `CompareScalar` 判断 `assist1 > 0.0` 生成掩码，调用 `Select` 选择 `xLocal` 或 `assist2Local`。
3. **int32_t 分支：** 直接调用整型乘加指令，计算 $y = x \times assist1 + assist2$。
4. **int8_t 分支 (当前多用于 bool 承载)：** 由于底层小整型乘加限制，先 `Cast` 到 `half` 精度，执行 `Mul` 和 `Add`，再 `Cast_RINT` 转回 `int8_t`。

### 1.3 新增章节：Ascend C 现有实现的流程图

目前IndexFillD算子实现还未合入库中，参考的实现逻辑在pr[!4146](https://gitcode.com/cann/ops-nn/pull/4146)中。

根据 `index_fill_d.cpp` 与 `index_fill_d.h` 的现有逻辑，其单次 `DoRunOp` 的执行流程如下：

![image.png](https://raw.gitcode.com/user-images/assets/10010611/9278de58-c2bc-4a33-bbf2-af04ba785f47/image.png 'image.png')

---

## 二、需求分析

### 2.1 外部组件依赖

本算子的主要外部依赖为 ACLNN 框架，框架层需完成入参校验、维度解析，并将 `dim`、`index` 参数在 Host 侧转化为适配层所需的掩码逻辑或由 Kernel 侧根据逻辑直接处理。

### 2.2 内部适配模块

算子内部需在 Ascend C 的 Host 侧 Tiling 模块（为新类型配置切分策略及 `TilingKey`）和 Kernel 侧实现模块补充新增类型的特化逻辑，并更新 API 层的参数注册。

### 2.3 需求模块设计

#### 2.3.1 AscendC算子原型

更新后的算子原型如下，已明确增加对 `INT16`、`INT8`、`UINT8`、`DOUBLE` 的支持：

| 参数名 | 输入/输出/属性 | 描述 | 使用说明 | 数据类型 | 数据格式 | 维度(shape) | 非连续Tensor |
| --- | --- | --- | --- | --- | --- | --- | --- |
| self | 输入 | 输入Tensor。 | 待被在指定位置的值用value替换的张量 | FLOAT16、FLOAT、INT32、INT64、BOOL、BFLOAT16、**INT16**、**INT8**、**UINT8**、**DOUBLE** | ND | [1,8] | √ |
| dim | 属性 | 指定了self将要填充的维度。 | dim取值范围在[-self.dim(), self.dim())，0维时为[-1, 1) | INT64 | - | - | - |
| index | 输入 | 指定self在dim维度将要填充的下标。 | 元素值小于self对应dim的维度大小 | INT32、INT64 | - | - | - |
| value | 输入 | 指定填充的数据值。 | 需要可转化为self的数据类型。 | FLOAT16、FLOAT、INT32、INT64、BOOL、BFLOAT16、**INT16**、**INT8**、**UINT8**、**DOUBLE** | - | - | - |
| out | 输出 | 指定的输出张量。 | - | 与self一致 | ND | 与self一致 | √ |

#### 2.3.2 算子相关约束

* 对于新增的 `double` 类型，鉴于部分 AI Core 缺乏原生的 64 位浮点乘加计算单元，必须使用软硬件协同的方式（切分 32 位运算）来保证精度与功能。

---

## 三、需求详细设计

### 3.1 使能方式

通过 ACLNN 接口进行算子的调用和使能。

### 3.2 需求总体设计

#### 3.2.1 host侧设计

* **3.2.1.1 分核策略**
获取输入总数据量，依据设备可用核心数进行均分。考虑到 `double` 类型的实现会消耗更多的 UB 内存（用于存储 64 位拆分出来的两份 32 位数据以及 `Gather` 重排索引），Host 侧需针对 `double` 制定特定的 `coreDataNum` 计算公式，防止内存溢出。
* **3.2.1.2 TilingKey 规划策略**
新增数据类型由于指令路线截然不同（转 `half` 计算 vs 转 `int32` 拆解重组），需通过 `dtype` 生成对应的 `tilingKey`。Kernel 侧依据 `tilingKey` 选用不同的泛化分支。

#### 3.2.2 kernel侧设计

* **3.2.2.1 kernel侧实现描述 (重点实现逻辑)**
针对新增数据类型，不使用原有的 `Select` 指令，而是统一采用掩码辅助张量 `assist1` 和 `assist2` 进行乘加操作 ($y = x \times assist1 + assist2$)。
1. **对于 int16, int8, uint8:** 受限于底层没有这几种类型的原生高性能矢量乘加指令，需先将数据 `Cast`（类型转换）为 浮点数，其中int16转为float，int8/uint8转为half。
* 步骤：读取输入 $\rightarrow$ 转至浮点数 $\rightarrow$ 执行 `Mul` ($x \times assist1$) $\rightarrow$ 执行 `Add` ($+ assist2$) $\rightarrow$ 运算完毕后 `Cast` 截断转回原整型 $\rightarrow$ CopyOut 写回。


2. **对于 double (float64):**
Ascend 架构对 64 位浮点的原生支持受限，需通过拆位与重排的逻辑来借用 32 位算术单元。
* **位解释：** 将输入 `double` 的 64 位内存重解释 (Reinterpret) 为 `int64`。
* **位切分：** 利用 `Cast` 将 `int64` 转化为 `int32`。
* **数据重排：** 此时数据排布不符合常规流水，需借助 `Gather` 算子按预设的索引对这批 `int32` 数据进行重排（将高低位分离到对应的运算位）。
* **模拟运算：** 针对切分好的 32 位数据执行 `Mul` 和 `Add` 运算，完成基于掩码的值替换。
* **聚合重写：** 将计算完成的高低位 `int32` 数据聚合拼装，重解释回 `double` 类型后写回。




* **3.2.2.2 AscendC实现流程图**

![image.png](https://raw.gitcode.com/user-images/assets/7665709/af995359-d9b7-4916-9aa3-7e387639a9e7/image.png 'image.png')

* **3.2.2.3 AscendC实现流程图与TBE流程图存在的差异点和原因**
**差异点：** TBE 的流程主要是在浮点的 `vcmpsel`（选择）和整型的 `vmul/vadd` 之间跳转。而 Ascend C 针对新增类型引入了深度的类型转换流（`int/uint -> half`）和复杂的内存重解释与指令拆解（`double -> int64 -> int32 -> Gather -> 聚合`）。
**原因：** Ascend C 贴近底层架构。在 A2/A3 上，8/16位整型的乘加开销较大且容易溢出，利用硬件强大的 `half`（FP16）算力能最大化矢量处理性能；而 `double` 由于 64 位宽度限制，必须通过 32 位指令集模拟和 `Gather` 重排的方式绕过硬件本身的限制，从而兼顾泛化要求和性能。

### 3.3 支持硬件

支持 **Atlas A2 训练系列产品 / Atlas A3 系列产品**。

### 3.4 算子约束限制

* `double` 的数据聚合策略占用较多临时 UB 缓存，大 Shape 下 Tiling 需预留足够的 `TBuf` 空间。

---

## 四、特性交叉分析

本算子是对已有 `IndexFillD` 功能在数据类型维度的补充，属于 Element-wise 与部分索引掩码的结合。针对新增数据类型的强转和位操作在核内闭环完成，不引发外部并发问题，与其他高级网络特性无破坏性交叉。

---

## 五、可维可测分析

### 5.1 精度标准/性能标准

* **精度标准：** 计算精度需满足 `AscendOpTest` 默认阈值。
* **性能标准：** 要求新增的 `int16, int8, uint8` 类型性能不劣于目前的 `int32` 数据类型；新增的 `double` 类型虽然有拆位重组开销，但总体性能不劣化于 `INT64` 类型。

### 5.2 兼容性分析

本设计向下兼容原有已实现的浮点和整型算子逻辑。外层 API 调用时，不同数据类型的转发基于 `TilingKey` 透明完成。这使得深度学习框架（如 PyTorch）可以无缝对齐底层新支持的整型和双精度 `IndexFillD`，极大增强了框架的兼容能力。