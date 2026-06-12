# Relu算子设计文档

# 一、需求背景

## 1.1 需求来源

参考昇腾版本内置aclnnRelu算子的TBE实现，在昇腾NPU上基于Ascend C编程语言实现功能一致的算子，并扩展支持int16数据类型。

## 1.2 背景介绍

### 1.2.1 Relu算子实现优化

ReLU（Rectified Linear Unit）数学表达式：

$$
\text{ReLU}(x) = \max(0, x) = 
\begin{cases} 
x, & x > 0 \\
0, & x \leq 0
\end{cases}
$$

### 1.2.2 Relu算子现状分析

#### 1.2.2.1 TBE算子支持的数据类型

- **支持数据类型**：`FLOAT`、`FLOAT16`、`INT8`、`INT32`、`INT64`、`BFLOAT16`
- **Shape限制**：0 到 8维

#### 1.2.2.2 TBE算子实现描述

TBE算子核心逻辑（路径：`/usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_impl/ai_core/tbe/impl/ops_legacy/relu.py`）：

TBE ReLU 是逐元素单输入单输出算子，无广播、归约及跨元素依赖，实现流程如下：
参数校验：通过para_check校验输入、输出、kernel_name 合法性，同时校验输入数据类型是否在白名单（float16/float32/int8/int32/bfloat16/int64）内；
动态 Shape 分类：按逐元素模式对动态 shape 输入分类；
张量构造：获取动态 shape，用tvm.placeholder构建输入张量；
核心计算分支：
特殊场景：res_conv2d/dequant_remove_pad输入 + 平台支持tik.vcopy+ 非 int64 → 调用tbe.vlrelu(x, 0)；
类型适配计算：
int8 + 平台支持s82f16：先转 float16 计算，结果转回 int8；
bfloat16：先转 float32 计算，结果经tbe.round转回 bfloat16；
常规场景：平台支持对应 dtype 的tbe.vrelu→直接调用；否则构造全 0 张量，用tbe.vmax(x, 0)实现；
编译生成：自动生成调度，编译输出可执行 kernel。

**TBE算子实现策略**

| 实现路径 | 适用条件 | 说明 |
|---------|---------|------|
| `vlrelu(x, 0)` | 支持vcopy且输入来自res_conv2d | 专用leaky relu优化 |
| `vrelu(x)` | 支持vrelu指令的数据类型 | 硬件原生relu指令 |
| `vmax(x, 0)` | 不支持vrelu的数据类型 | 通用实现：与0取最大值 |

#### 1.2.2.3 TBE算子实现流程图

![exported_image (22).png](https://raw.gitcode.com/user-images/assets/10036508/5e6c4732-badf-44b3-90c5-33e0feb260f2/exported_image__22_.png 'exported_image (22).png')

# 二、需求分析

## 2.1 外部组件依赖

不涉及外部组件依赖。

## 2.2 内部适配模块

适配Aclnn接口，支持常规调用和原位调用两种模式。

## 2.3 需求模块设计

### 2.3.1 AscendC算子原型

| 参数名 | 输入/输出 | 数据类型 | 数据格式 | 维度 | 非连续Tensor |
|--------|----------|---------|---------|------|---------------|
| self | 输入 | FLOAT、FLOAT16、INT8、INT16、INT32、INT64、BFLOAT16 | ND | 0-8 | √ |
| out | 输出 | FLOAT、FLOAT16、INT8、INT16、INT32、INT64、BFLOAT16 | ND | 0-8 | √ |

### 2.3.2 AscendC算子相关约束

- 支持空Tensor（shape需与out一致）
- 输入输出shape必须相同
- 最大支持8维张量
- 支持原位计算模式（y = x）
- 支持非连续Tensor

# 三、需求详细设计

## 3.1 使能方式

| 上层框架 | 支持 |
|---------|------|
| Aclnn直调 | √ |

## 3.2 需求总体设计

### 3.2.1 host侧设计

#### 3.2.1.1 分核策略

- **INT64类型**：使用所有可用Core
- **其他类型**：根据总数据量决定，按16KB阈值分配，最多使用coreNum个Core
- **任务分配**：前formerNum个Core处理formerLength，最后一个Core处理tailLength

#### 3.2.1.2 数据分块和内存优化策略

- **UB容量适配**：根据数据类型计算bufferCoefficient
  - FP32：8（双缓冲：2×4字节）
  - FP16/INT8：4（数据量小）
  - BF16/INT16：12（需转FP32中间类型）
  - INT64：32（CompareScalar向量化）

- **内存对齐**：按32字节对齐tile大小，Cache Line对齐（512字节）分配Core任务

#### 3.2.1.3 tilingKey规划策略

将tiling参数（formerNum、formerLength、tailLength、tileLength）封装到ReluTilingData结构体中。

### 3.2.2 kernel侧设计

#### 3.2.2.1 kernel侧实现描述

Relu算子kernel侧采用模板化设计，针对不同数据类型实现3种Kernel类：

1. **KernelRelu<T>**：用于FP32、FP16、INT32，直接使用原生Relu指令
2. **KernelReluUpcast<T, MidT>**：用于BF16、INT16（MidT=float）、INT8（MidT=half），通过类型提升计算
3. **KernelReluVectorInt64<T>**：用于INT64，任意INT64整数 `x = (high_32 << 32) | low_32`，其符号仅由 `high_32` 的最高位决定：
```
ReLU(x) = x,  若 high_32 的 bit15 == 0（正数或零）
          0,  若 high_32 的 bit15 == 1（负数）
```

所有Kernel类共享基类KernelReluBase，实现通用的数据搬运和流程控制。

#### 3.2.2.2 AscendC实现流程图

![exported_image (14).png](https://raw.gitcode.com/user-images/assets/9516645/c550e4ff-8fbf-4b93-a0ef-67110d12904b/exported_image__14_.png 'exported_image (14).png')

int64子流程：
![exported_image (15).png](https://raw.gitcode.com/user-images/assets/9516645/78590961-0109-4adb-9616-2ae8104ca055/exported_image__15_.png 'exported_image (15).png')

#### 3.2.2.3 AscendC实现流程图与TBE流程图存在的差异点和原因

| 差异点 | TBE实现 | AscendC实现 | 原因 |
|--------|---------|-------------|------|
| 数据类型支持 | FP16/FP32/INT8/INT32/BF16/INT64 | 扩展支持INT16 | AscendC灵活性更高 |
| INT8处理 | Cast到Float16 | Cast到Half | 策略一致，类型名称不同 |
| 计算指令 | vrelu/vmax/vlrelu | Relu/CompareScalar+Select | AscendC API差异 |

## 3.3 支持硬件

| 芯片版本 | 支持 |
|---------|------|
| Atlas 800I/T A2 | √ |
| Atlas 910B | √ |
| Atlas A3系列 | √ |

## 3.4 算子约束限制

- 输入输出shape必须相同（无广播）
- 输入输出数据类型必须相同
- 最大支持8维张量
- INT64类型使用向量化计算，性能较其他类型略低
- 所有核参与场景下，性能不低于TBE算子的95%

# 四、特性交叉分析

本算子为独立激活函数算子，不涉及与其他算子的特性交叉。

# 五、可维可测分析

## 5.1 精度标准/性能标准

**精度标准**：
- 浮点类型：误差在硬件允许范围内，满足AscendOpTest默认阈值
- 整型类型：结果必须完全一致
- 对比基准：以TBE算子输出为基准

**性能标准**：
- 所有核参与计算场景下，性能不低于TBE算子的95%
- 小shape场景（10us以下）相差3us可接受

## 5.2 兼容性分析

- **向后兼容**：完全兼容TBE算子已支持的数据类型和格式
- **扩展支持**：INT16类型
- **调用模式**：支持常规计算和原位计算（aclnnInplaceRelu）
