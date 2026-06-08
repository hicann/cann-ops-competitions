# AdamApplyOneWithDecay算子AscendC实现设计文档

## 一、 需求背景

### 1.1 需求来源

通过社区任务完成开源仓算子贡献的需求

### 1.2 背景介绍

#### 1.2.1 AdamApplyOneWithDecay算子实现优化

基于AdamApplyOneWithDecay算子历史TBE版本使用AscendC编程语言在910B芯片上进行优化替换
AdamApplyOneWithDecay算子（TBE）实现路径和相关API路径

- TBE实现路径（910B芯片原版）:
   静态实现： `/home/developer/Ascend/cann-9.0.0/opp/built-in/op_impl/ai_core/tbe/impl/ops_legacy/adam_apply_one_with_decay.py`
   动态实现： `/home/developer/Ascend/cann-9.0.0/opp/built-in/op_impl/ai_core/tbe/impl/ops_legacy/dynamic/adam_apply_one_with_decay.py`
   adam_apply_one_with_decay没有DSL api入口，因为它不是 TBE 框架提供的基础 DSL API，而是一个自定义组合算子。
- 950芯片AscendC实现路径（atvoss DAG框架）: `ops-nn/optim/adam_apply_one_with_decay/op_kernel/arch35/adam_apply_one_with_decay_dag.h`
- 950芯片AscendC编译入口: `/home/developer/Ascend/cann-9.0.0/opp/built-in/op_impl/ai_core/tbe/impl/ops_nn/dynamic/adam_apply_one_with_decay_apt.py`

#### 1.2.2 AdamApplyOneWithDecay算子现状分析

通过对AdamApplyOneWithDecay算子910B芯片TBE版本的功能分析，当前支持的能力如下：

- 使用Python DSL定义计算，通过`tvm.placeholder`创建输入占位符
- `adam_apply_one_with_decay_compute`函数中组合基础算子：`vmul`(平方)、`vmul`(乘)、`vadd`(加)、`vsqrt`(开方)、`vdiv`(除)、`vsub`(减)
- fp16输入时，sqrt_compute中通过`cast_to`提升精度到float32再计算`vsqrt`，之后cast回float16
- 使用`classify`自动分类输入shape模式（`ELEWISE_WITH_BROADCAST`）
- 使用`auto_schedule`自动调优，框架自动处理分核、UB切分、双缓冲
- 通过`broadcast_shapes` + `tbe.broadcast`处理输入间的广播

TBE算子的整体流程图如下图所示:
![nn_adam_apply_one_with_decay_docs_tbe.png](https://raw.gitcode.com/user-images/assets/7665709/7b05dcda-5bd2-479e-a639-00832ab39068/nn_adam_apply_one_with_decay_docs_tbe.png 'nn_adam_apply_one_with_decay_docs_tbe.png')

## 二、 需求分析

### 2.1 外部组件依赖

不涉及外部组件适配

### 2.2 内部适配模块

该算子支持 aclnn 接口（自动生成 aclnn 接口），需要适配 op_host、tiling、kernel 模块。

### 2.3 需求模块设计

#### 2.3.1 AscendC算子原型

| 名称 | 类别 | dtype | format | shape | 介绍 |
|--|--|--|--|--|--|
| input0 | 输入 | bf16/fp16/fp32 | ND | all | 广播张量输入0（gradient，梯度） |
| input1 | 输入 | bf16/fp16/fp32 | ND | all | 广播张量输入1（v，二阶矩） |
| input2 | 输入 | bf16/fp16/fp32 | ND | all | 广播张量输入2（m，一阶矩） |
| input3 | 输入 | bf16/fp16/fp32 | ND | all | 广播张量输入3（param，参数） |
| input4 | 输入 | bf16/fp16/fp32 | ND | (1,) | 标量：学习率lr |
| mul0_x | 输入 | bf16/fp16/fp32 | ND | (1,) | 标量：beta1 |
| mul1_x | 输入 | bf16/fp16/fp32 | ND | (1,) | 标量：1-beta1 |
| mul2_x | 输入 | bf16/fp16/fp32 | ND | (1,) | 标量：beta2（二阶矩衰减率） |
| mul3_x | 输入 | bf16/fp16/fp32 | ND | (1,) | 标量：1-beta2（v的更新系数） |
| mul4_x | 输入 | bf16/fp16/fp32 | ND | (1,) | 标量：weight_decay |
| add2_y | 输入 | bf16/fp16/fp32 | ND | (1,) | 标量：epsilon |
| output0 | 输出 | bf16/fp16/fp32 | ND | 广播后shape | 输出0（v_new，更新后的二阶矩） |
| output1 | 输出 | bf16/fp16/fp32 | ND | 广播后shape | 输出1（m_new，更新后的一阶矩） |
| output2 | 输出 | bf16/fp16/fp32 | ND | 广播后shape | 输出2（param_new，更新后的参数） |

- 相关约束:
    - 所有输入输出数据类型必须一致
    - input0与input1广播得到output0的shape
    - input0与input2广播得到output1的shape
    - input2与input3广播得到output2的shape
    - input4/mul0_x-mul4_x/add2_y为标量输入（shape为[1]或标量）
    - 支持 Atlas 800I/T A2（ascend910b）

## 三、 需求详细设计

### 3.1 使能方式

| 上层框架 | 涉及的框架 |
|---------|---------|
| **TF训练/推理** | - |
| **PyTorch训练/推理** | - |
| **ATC推理** | - |
| **Aclnn直调** | √ |
| **OPAT调优** | - |
| **SGAT子图切分** | - |

### 3.2 需求总体设计

#### 3.2.1 host侧设计

##### 3.2.1.1 分核策略

参考库上类似算子softsign，将总元素数均分到各AI Core

```
totalNum: 总元素个数
coreNum: AI Core总数
usedCoreNum: 实际使用的核数

分核逻辑：
blockFactor = CeilDiv(totalNum, coreNum)
usedCoreNum = CeilDiv(totalNum, blockFactor)
```

##### 3.2.1.2 数据分块和内存优化策略

UB总空间减去各buffer占用，根据数据类型和是否开启Double Buffer计算单次可处理的元素数。

**float32（直接计算路径）**：

| Buffer | 数量 | 单位大小 | 说明 |
|--------|:----:|--------|------|
| input0Queue | BUFFER_NUM | 4B | 广播张量输入0 |
| input1Queue | BUFFER_NUM | 4B | 广播张量输入1 |
| input2Queue | BUFFER_NUM | 4B | 广播张量输入2 |
| input3Queue | BUFFER_NUM | 4B | 广播张量输入3 |
| output0Queue | BUFFER_NUM | 4B | 输出0 |
| output1Queue | BUFFER_NUM | 4B | 输出1 |
| output2Queue | BUFFER_NUM | 4B | 输出2 |
| tmpBuf0-2 | 3 | 4B | 中间计算缓冲 |
| **合计** | 7*BUFFER_NUM+4 | 4B | |

- Double Buffer开启时(BUFFER_NUM=2): 17个buffer = 68 bytes/element
- Double Buffer关闭时(BUFFER_NUM=1): 10个buffer = 40 bytes/element

**float16（Cast计算路径 - 仅Sqrt时Cast）**：

| Buffer | 数量 | 单位大小 | 说明 |
|--------|:----:|--------|------|
| input0-3Queue | 4*BUFFER_NUM | 2B | half类型输入队列 |
| output0-2Queue | 3*BUFFER_NUM | 2B | half类型输出队列 |
| tmpBuf0-2 | 3 | 2B | 中间计算half缓冲（单buffer） |
| sqrtBuf | 1 | 4B | Sqrt计算float缓冲（单buffer） |
| **合计** | 7\*BUFFER_NUM\*2B + 4\*2B + 1\*4B | — | |

- Double Buffer开启时: 7\*2\*2B + 3\*2B + 1\*4B = 28B + 6B + 4B = 38 bytes/element
- Double Buffer关闭时: 7\*1\*2B + 3\*2B + 1\*4B = 14B + 6B + 4B = 24 bytes/element

**bfloat16（全Cast计算路径 - 全部计算在float精度下）**：

| Buffer | 数量 | 单位大小 | 说明 |
|--------|:----:|--------|------|
| input0-3Queue | 4*BUFFER_NUM | 2B | bf16类型输入队列 |
| output0-2Queue | 3*BUFFER_NUM | 2B | bf16类型输出队列 |
| floatInput0-3 | 4 | 4B | 输入cast到float缓冲 |
| floatOut0-2 | 3 | 4B | 输出cast到float缓冲 |
| tmpBuf0-3 | 3 | 4B | 中间计算float缓冲 |
| **合计** | 7\*BUFFER_NUM\*2B + 11\*4B | — | |

- Double Buffer开启时: 7\*2\*2B + 10\*4B = 28B + 40B = **68 bytes/element**
- Double Buffer关闭时: 7\*1\*2B + 10\*4B = 14B + 40B = **54 bytes/element**

> 注：bfloat16的TBuf使用4字节float是因为AscendC API限制（Muls/Adds等操作不完全支持bf16），需要cast到float进行计算。实际kernel实现中有10个TBuf（floatInput0-3、floatOut0-2、tmpBuf0-2）。

具体实现公式：
```cpp
// 根据数据类型和是否双缓冲计算每元素占用字节数
int64_t bytesPerElement = CalcBytesPerElement(dataType, useDoubleBuffer);
// 计算单次循环可处理的元素数（按UB块大小对齐）
int64_t ubFactor = FloorAlign(FloorDiv(ubSize, bytesPerElement), ubBlockSize);
```

**是否启用Double Buffer**
| 场景 | 是否启用 |
|------|---------|
| 单缓冲（totalNum <= 1024） | 不启用 |
| 双缓冲（totalNum > 1024） | 启用 |

```cpp
constexpr int64_t MIN_SPLIT_THRESHOLD = 1024;
uint64_t useDoubleBuffer = (totalNum > MIN_SPLIT_THRESHOLD) ? 1 : 0;
```
##### 3.2.1.3 tilingKey规划策略
使用tilingKey来区分不同数据类型的kernel分支：
- `tilingKey = 0` (FLOAT32_MODE)：使用 `AdamApplyOneWithDecayKernel` 直接在float精度下计算
- `tilingKey = 1` (FLOAT16_MODE)：使用 `AdamApplyOneWithDecayHalfKernel` 仅在Sqrt时cast到float
- `tilingKey = 2` (BF16_MODE)：使用 `AdamApplyOneWithDecayBf16Kernel` 全部输入cast到float，全部计算在float精度下，最后cast回bf16
- host侧根据输入数据类型设置对应的tilingKey值
- kernel侧通过 `TILING_KEY_IS()` 宏进行编译期分支选择

#### 3.2.2 kernel侧设计

##### 3.2.2.1 kernel侧实现描述

AdamApplyOneWithDecay算子针对不同数据类型采用不同的处理策略：

1. **float32（直接计算路径 AdamApplyOneWithDecayKernel）**：
   - 标量输入通过GM地址直接读取（使用GlobalTensor.GetValue()）
   - 广播张量输入通过 `DataCopyPad` 拷贝到UB
   - 所有计算直接在float精度下进行：`Mul`、`Muls`、`Add`、`Sub`、`Sqrt`、`Div`

2. **float16（Cast计算路径 AdamApplyOneWithDecayHalfKernel）**：
   - 标量输入从GM地址读取half类型值
   - 广播张量输入通过 `DataCopyPad` 拷贝half类型到UB
   - **大部分计算保持half精度**：output0、output1、除法、加法等计算直接在half类型下进行
   - **仅在Sqrt计算时临时cast到float**：计算output2时，将output0 cast到float32计算sqrt，**sqrt后立即cast回half**，然后adds在half下进行
   - **与TBE流程完全一致**：TBE的sqrt_compute中cast→sqrt→立即cast back，add_compute在原精度下进行

3. **bfloat16（Cast计算路径 AdamApplyOneWithDecayBf16Kernel）**：
   - 标量输入从GM地址读取bf16类型值
   - 广播张量输入通过 `DataCopyPad` 拷贝bf16类型到UB
   - **全部输入立即cast到float**，**所有计算在float精度下进行**，**最后所有输出cast回bf16**（API限制：Muls/Adds不完全支持bf16）
   - Cast流程：bf16输入 → 立即cast到float → 全部float计算 → 最后cast回bf16输出

##### 3.2.2.2 AscendC实现流程图
![nn_adam_apply_one_with_decay_docs_ascendc.png](https://raw.gitcode.com/user-images/assets/7665709/05304aa6-9c85-40a7-892c-4e3331680c3c/nn_adam_apply_one_with_decay_docs_ascendc.png 'nn_adam_apply_one_with_decay_docs_ascendc.png')

**三个模板的关键差异**：

| 特性 | Float32模板 | Float16模板 | BFloat16模板 |
|------|------------|-------------|--------------|
| 输入Queue类型 | float | half | bfloat16_t |
| 输出Queue类型 | float | half | bfloat16_t |
| tmpBuf类型 | float | half | float |
| sqrtBuf类型 | N/A | float | N/A |
| 计算精度 | 全float | 大部分half，sqrt用float | 全float |
| Cast操作 | 无 | 仅Sqrt时: half→float→half | 全部输入: bf16→float<br>全部输出: float→bf16 |
| API限制 | 无 | Sqrt需要float精度 | Muls/Adds不支持bf16 |

##### 3.2.2.3 910B芯片TBE与AscendC实现流程图存在的差异点和原因

- bf16数据类型，tbe是只在sqrt运算cast到fp32进行，其余运算在原数据类型。AscendC实现因为Muls，Adds不支持bf16数据类型，是先全部cast到fp32，所有计算完成后再cast回bf16数据类型的。
- 分核逻辑，TBE实现分核逻辑是框架`auto_schedule`自动调优的，910B芯片上原有的算子即为TBE实现。AscendC实现参考的库上softsign算子分核逻辑，所以可能会稍有差异。

### 3.3 支持硬件

| 支持的芯片版本 | 是否支持 |
|---------|---------|
| Atlas 800I/T A2（ascend910b） | 支持 |

### 3.4 算子约束限制

- 所有输入输出数据类型必须一致
- input0-input3必须满足广播规则
- 标量输入（input4/mul0_x-mul4_x/add2_y）
- 不支持动态广播（input0-input3需要shape一致或可广播）
- 支持 aclnn 接口自动生成，可直调使用

## 四、 特性交叉分析

不涉及特性交叉

## 五、 可维可测分析

### 5.1 精度标准/性能标准

精度方面，和TBE对齐

性能参考AdamApplyOneWithDecay算子任务书，有如下要求：
- 算子整体性能需与910B芯片原TBE实现持平
- 暂仅要求所有核参与计算场景下，性能不低于原TBE算子的 95%
- bf16类型因为AscendC api限制，只能在fp32数据类型下进行计算，性能从理论上要比只有sqrt在fp32计算的逻辑要稍差一些，具体还得实测后进行对比

### 5.2 兼容性分析

- 算子原型与910B芯片TBE版本完全一致（11输入3输出，相同名称和数据类型）
- InferShape逻辑与TBE版本一致（基于BroadcastShape推导，对应TBE的`_check_broadcast_shape`）
- 数学公式完全一致（TBE的`adam_apply_one_with_decay_compute`组合的算子序列与AscendC的Compute步骤一一对应）
- 该910B AscendC版本位于experimental目录，不影响910B TBE版本和950 DAG版本的已有功能
