# 【社区任务】ELU 算子设计文档

# 一、需求描述

## 1.1 需求来源

本需求来源于 [CANN 社区任务 2026 的“8 月社区任务 - ELU 算子开发任务书”](https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/docs/202608/asc_elu_task_doc.md)。任务要求基于 Ascend C C API 开发纯 Vector-Core ELU 算子，支持 `FLOAT` 和 `FLOAT16`，适配 Ascend 950 与 CANN 9.0.0~9.1.0。

代码目标目录：

```text
asc-devkit/examples/02_simd_c_api/03_c_api/02_reg_vector_compute/elu
```

## 1.2 需求分析

### 背景介绍

ELU（Exponential Linear Unit）是 ReLU 系列激活函数。正数区间保持线性输出，非正数区间使用指数曲线，使输出下界趋近于 `-alpha * scale`。本任务中的 ELU 还包含 `scale` 和 `input_scale` 两个缩放参数。

算子内部包含指数运算，指令依赖链长于 ReLU。实现需要减少无效正区间指数计算造成的数值溢出风险，并在不使用 Cube-Core 的前提下安排 GM/UB 搬运和 Reg Vector 计算。

输入要求（来自任务书）

| 参数            | 类型     | 含义                     | 数据类型       | 格式 |
| --------------- | -------- | ------------------------ | -------------- | ---- |
| `src`         | 输入     | 源 Tensor 连续首地址     | FLOAT、FLOAT16 | ND   |
| `dst`         | 输出     | 目的 Tensor 连续首地址   | FLOAT、FLOAT16 | ND   |
| `alpha`       | 输入属性 | 非正分支激活系数         | FLOAT          | 标量 |
| `scale`       | 输入属性 | 输出缩放系数             | FLOAT          | 标量 |
| `input_scale` | 输入属性 | 非正分支指数输入缩放系数 | FLOAT          | 标量 |

### 功能描述

实现与任务书公式一致的 ELU C API 样例：

$$
ELU(x)=
\begin{cases}
scale \cdot x, & x>0 \\
alpha \cdot scale \cdot (e^{x \cdot input\_scale}-1), & x\leq0
\end{cases}
$$

### 需求拆解

1. 支持 `float32` 和 `float16` 两条计算路径，输入输出 dtype 相同。
2. 支持元素总数不超过 `uint32_t` 范围的连续 ND Tensor；空 Tensor 由 Host 直接返回空输出，非空 Tensor 由 Kernel 展平为一维处理。
3. 覆盖任务指定的 shape `1/32/1024/2048` 和输入值域 `[-100, 100]`。
4. 支持非 32B 对齐的小 shape 和尾块，不能越界读写 GM。
5. 所有逐元素计算在 Device Vector Core 完成；Host 仅负责读取二进制、内存申请、Kernel 下发和输出写盘。
6. 不调用 Cube-Core，不申请 L0A/L0B/L0C，不执行矩阵指令。
7. 本任务无性能硬指标，仍应避免额外 UB 中间 Tensor 和 Host 串行计算。

### 外部依赖

| 组件             | 用途                                   |
| ---------------- | -------------------------------------- |
| Ascend 950       | `dav-3510` Reg Vector 指令执行       |
| CANN 9.0.0~9.1.0 | ASC 编译器、ACL Runtime 与设备环境     |
| asc-devkit       | C API 头文件、实现、CMake 包和样例目录 |
| Python 3 + NumPy | 输入生成、CPU Golden 和精度比对        |

# 二、方案设计

## 2.1 内部实现

### 数学公式

原始分段公式可改写为：

$$
y=scale\cdot\left(\max(x,0)+alpha\cdot\left(e^{\min(x,0)\cdot input\_scale}-1\right)\right)
$$

等价性：

- 当 `x > 0` 时，`min(x, 0)=0`，指数项为 `exp(0)-1=0`，结果为 `scale*x`。
- 当 `x <= 0` 时，`max(x, 0)=0`，结果为原始指数分支。

该形式无需生成比较 Mask 后再 Select；在有限属性输入下，正数区间的 `min(x,0)` 为 0，因此指数项退化为 `exp(0)-1=0`，不会计算无用的 `exp(x*input_scale)`。

### 支持数据类型

任务书描述：

- `src` 和 `dst` 支持 `float32/float16`，输入输出 dtype 相同；
- `alpha`、`scale`、`input_scale` 是独立的**标量**输入，任务书将三者的数据类型固定为 `FLOAT`，不随 Tensor dtype 改变。为保持该接口定义，Host 解析结果和 Kernel 下发参数均使用 `float`。

### 支持形状

[任务书 2.2 参数定义](../../../../docs/202608/asc_elu_task_doc.md#22-参数定义)规定 `src`、`dst` 的数据格式为 ND，[任务书 3 自验证计划](../../../../docs/202608/asc_elu_task_doc.md#3-自验证计划)列出 `1/32/1024/2048` 四种测试数据量，但没有规定具体 Shape、rank、stride，也没有要求 Shape 发生变化或进行 Shape 推导。

当前样例的对形状的处理方式：

- 不接收和处理 Shape 信息，只接收元素总数 `totalLength`；
- 默认输入、输出在 GM 中连续存储。Kernel 按连续地址处理 `totalLength` 个元素，
- 不执行 Shape 解析、Flatten、Reshape 或非连续 Tensor 转换；
- `length=0` 时由 Host 直接返回空输出。

### 整体实现方案

ELU 是逐元素算子，当前实现按以下固定数据流执行，不包含与 rank 相关的分支：

1. Host 校验参数，将连续输入表示为 `totalLength`，根据数据量确定实际启动核数并选择 float32 或 float16 Kernel。
2. 每个 Kernel Block 获得一段连续且互不重叠的输入区间，再将该区间循环切分为不超过 8192 个元素的 Tile。
3. 每个 Tile 依次经过 `GM -> UB` 搬入、Reg Vector ELU 计算和 `UB -> GM` 搬出；输入、输出 UB 使用双缓冲。
4. float32 和 float16 使用相同的数据流与数学公式，只改变 GM/UB/Reg Vector 数据类型；所有逐元素计算均在 Device 完成。

上述流程分别对应当前 `RunElu`、`GetAlignedCoreRange`、`elu_float_custom/elu_half_custom` 和 `EluFloatVF/EluHalfVF` 的实现。

### 已实现的性能优化

以下各项均已体现在当前代码中 。

| 优化项                | 当前实现及依据                                                                                                      |
| --------------------- | ------------------------------------------------------------------------------------------------------------------- |
| 无分支计算            | `EluFloatVF/EluHalfVF` 使用 `asc_min/asc_max` 实现第 2.1 节证明过的等价公式，不生成 Compare/Select 结果 Buffer  |
| Reg Vector 中间值驻留 | Kernel 的 UB 声明只有两组输入槽和两组输出槽；正负分支中间值均为`vector_float/vector_half`                         |
| 指令融合              | 使用`asc_axpy` 直接执行 `positive += alpha * negative`，在语义上合并一次标量乘法和一次向量加法                  |
| 常量与 Mask 复用      | 零向量和全宽 Mask 在每个 VF 函数调用中构造一次，完整 repeat 循环复用；仅尾 repeat 构造 Tail Mask                    |
| 数据量感知分核        | Host 结合 Tile 上限、完整 VL 工作量和设备可用核数选择实际核数；小输入核数扫描和大输入 profiling 结果见第 3.1 节     |
| 有效长度搬运          | CopyIn/CopyOut 的长度均为`tileBytes`，最后不足 256B 的 Reg Vector 计算由 Tail Mask 屏蔽；对应任务书的内存安全要求 |
| Ping/Pong 双缓冲      | `srcBuffer/dstBuffer` 均有两个槽，Tile 按奇偶选择槽和 `EVENT_ID0/1`                                             |

## 2.2 接口设计

### 2.2.1 Host 侧接口

#### 命令行参数与职责

Host 可执行程序样例接收：

```C++
elu_example <float32|float16> <length> <alpha> <scale> <input_scale> <input.bin> <output.bin> [blocks|auto]
```

Host 仅执行：

1. 校验 dtype、元素数、属性和输入文件大小。
2. **若元素数为 0**，直接写出空输出，不初始化 Device、不申请 GM 且不下发 Kernel。
3. 对非空 Tensor 初始化 ACL、Device 和 Stream。
4. 申请输入输出 GM 并搬入原始二进制输入。
5. 查询可用 Vector Core 数，根据可选核数参数和 dtype 下发 `elu_float_custom` 或 `elu_half_custom`。
6. 同步 Stream，搬回输出并写盘。
7. 释放 Stream、GM、Device 和 ACL 资源。

CPU Golden 由独立 Python 测试脚本生成，不属于算子运行路径。

#### 分核与分块策略

Host 确定并下发 `numBlocks`， 自动分核同时考虑 UB Tile 上限和实际 Vector repeat 工作量：

| 约束             | 计算方式                                                                        | 作用                                                                                                                        |
| ---------------- | ------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| Tile 核数上限    | $B_{tile}=\min(B_{available},\max(8,\lceil N/8192\rceil))$                    | `8` 是根据现有 `1/2/4/8` 核小输入扫描选取的经验阈值；总 Tile 数超过 8 后随 Tile 数扩核，且不超过设备可用 Vector Core 数 |
| 小输入工作量上限 | $B_{repeat}=\lceil N/R\rceil$，其中 float32 的 $R=64$，float16 的 $R=128$ | 核数不超过覆盖输入所需的 Vector repeat 数，避免小输入启动过多核；最后一个 repeat 可以不足 256B                              |

最终自动核数为：

$$
B_{auto}=\min(B_{tile},B_{repeat})
$$

其中 `8` 不是任务书要求或硬件固定值，而是当前测试范围内采用的经验参数，相关核数扫描结果见第 3.1 节。它只参与计算核数上限，最终实际核数仍由 $B_{repeat}$ 和设备可用 Vector Core 数共同限制。

- 例如 float32 的 32/64/128/256/512 个元素分别启动 1/1/2/4/8 个 Kernel Block。用户显式指定核数仅用于验证与性能对比，Host 仍会将其限制在设备可用核数及有效工作量范围内。

### 2.2.2 Kernel 侧接口

#### 入口、参数与约束

Device 侧按照输入 dtype 提供两个 Kernel 入口，以下函数签名直接对应 `elu.asc`：

```cpp
__vector__ __global__ void elu_float_custom(
    __gm__ float* src,
    __gm__ float* dst,
    uint32_t totalLength,
    uint32_t numBlocks,
    float alpha,
    float scale,
    float inputScale);

__vector__ __global__ void elu_half_custom(
    __gm__ half* src,
    __gm__ half* dst,
    uint32_t totalLength,
    uint32_t numBlocks,
    float alpha,
    float scale,
    float inputScale);
```

| 参数            | 方向 | 类型                                 | 作用                                             | 约束                                                                                                    |
| --------------- | ---- | ------------------------------------ | ------------------------------------------------ | ------------------------------------------------------------------------------------------------------- |
| `src`         | 输入 | `__gm__ float*` / `__gm__ half*` | 连续 ND 输入在 GM 中的首地址                     | 只支持 float32/float16，dtype 必须与`dst` 相同；Kernel 不接收 rank 或 stride，不支持非连续存储 Tensor |
| `dst`         | 输出 | `__gm__ float*` / `__gm__ half*` | GM 输出首地址                                    | dtype 和元素数必须与`src` 相同，并具有容纳 `totalLength` 个元素的空间                               |
| `totalLength` | 输入 | `uint32_t`                         | 连续输入的总元素数，用于计算逐核范围和 Tile 数量 | Kernel 入口取值范围为$[1,2^{32}-1]$；`length=0` 由 Host 返回空输出，不启动 Kernel                   |
| `numBlocks`   | 输入 | `uint32_t`                         | Host 最终确定的实际启动核数                      | 必须与`<<<numBlocks, 0, stream>>>` 的启动核数一致，并由 Host 限制在设备可用核数和有效工作量范围内     |
| `alpha`       | 输入 | `float` 标量                       | 广播到非正分支，控制指数结果的激活系数           | 始终以`float` 传入；float16 Kernel 在每个 Kernel Block 中转换一次为 `half`                          |
| `scale`       | 输入 | `float` 标量                       | 广播到全部有效元素，对合并结果进行最终缩放       | 始终以`float` 传入；float16 Kernel 在每个 Kernel Block 中转换一次为 `half`                          |
| `inputScale`  | 输入 | `float` 标量                       | 广播到非正分支，在指数计算前缩放输入             | 始终以`float` 传入；float16 Kernel 在每个 Kernel Block 中转换一次为 `half`                          |

非参数约束

- Kernel 形式： 使用`__vector__ __global__`，只执行 GM↔UB 搬运、Reg Vector 计算及 Scalar Core 循环控制，不使用 Cube 数据搬运、矩阵计算或 L0 矩阵内存
- 适配目标：面向 Ascend 950`dav-3510`，支持 Ascend 950PR/950DT。

Host 根据 dtype 选择入口并按同一顺序传参：

```cpp
if (dataType == EluDataType::FLOAT32) {
    elu_float_custom<<<numBlocks, 0, stream>>>(
        static_cast<float*>(srcDevice), static_cast<float*>(dstDevice),
        totalLength, numBlocks, alpha, scale, inputScale);
} else {
    elu_half_custom<<<numBlocks, 0, stream>>>(
        static_cast<half*>(srcDevice), static_cast<half*>(dstDevice),
        totalLength, numBlocks, alpha, scale, inputScale);
}
```

两个入口使用相同的分块、CopyIn、Compute、CopyOut 流程，差异只在 Tensor 和 Reg Vector 数据类型。Kernel 无返回值，计算结果直接写入 `dst`。

#### Kernel 执行流程

两个 Kernel 入口共享以下总体流程，`CopyIn`、`Compute` 和 `CopyOut` 通过两组 Ping/Pong UB 槽及同步事件衔接：

```text
(blockOffset, currentLength) = GetAlignedCoreRange(totalLength, numBlocks, block_idx)
准备计算标量，申请两组输入/输出 UB 槽并初始化同步事件

for tileIndex in [0, ceil(currentLength / 8192)):
    slot       = tileIndex & 1
    tileLength = min(8192, currentLength - tileIndex * 8192)
    gmOffset   = blockOffset + tileIndex * 8192

    CopyIn : 等待输入槽可复用，将 src[gmOffset : gmOffset + tileLength] 从 GM 搬入 UB[slot]
    Compute: 等待输入就绪及输出槽可复用，在 Reg Vector 中完成 ELU，结果写入输出 UB[slot]
    CopyOut: 等待计算完成，将输出 UB[slot] 的 tileLength 个元素搬回 dst 的对应区间

等待两组槽的最终事件，Kernel 结束
```

float32 路径使用 `vector_float/float`，float16 路径使用 `vector_half/half`。float16 Tensor 在 GM、UB 和 Reg Vector 计算中始终保持 half，不转换为 float；只有 `alpha`、`scale` 和 `inputScale` 三个 float 标量在每个 Kernel Block 开始计算前转换一次为 half。

##### 分块

Kernel 先调用 `GetAlignedCoreRange` 取得本核范围，再进行核内 Tile 切分：

```cpp
GetAlignedCoreRange(
    totalLength, numBlocks, elementsPerDataBlock,
    block_idx, blockOffset, currentLength);
tileCount = ceil(currentLength / 8192);
```

`blockOffset/currentLength` 表示本核负责的连续区间；循环中的 `tileLength=min(8192, currentLength-processed)`，`gmOffset=blockOffset+processed`。每核分配两组 Ping/Pong 输入和输出槽：

```text
srcBuffer[2][8192]
dstBuffer[2][8192]
```

最大 UB 用量：

| dtype   | 输入 UB | 输出 UB |     合计 |
| ------- | ------: | ------: | -------: |
| float32 | 65536 B | 65536 B | 131072 B |
| float16 | 32768 B | 32768 B |  65536 B |

Tile 根据 `tileIndex & 1` 选择槽和对应的 `EVENT_ID0/1`。当一个核负责的数据超过 8192 个元素时，循环继续处理下一 Tile；计算中间值只保存在 Reg Vector 中，不申请额外 UB Tensor。

##### CopyIn

CopyIn 的输入是当前 Tile 的 GM 地址和有效字节数，输出是当前 Ping/Pong 输入槽：

```cpp
asc_copy_gm2ub_align(
    srcLocal, src + gmOffset, 1, tileBytes,
    0, 0, false, 0, 0, 0);
```

当前槽先等待 `PIPE_V -> PIPE_MTE2` 释放事件，再调用 `asc_copy_gm2ub_align`，随后以 `PIPE_MTE2 -> PIPE_V` 事件通知 Compute 输入就绪。

这里采用 CANN 9.0.0~9.1.0 均已编译通过的 `dav-3510` 完整参数重载：单 burst、无 padding、关闭常量填充、stride 为 0。每个槽按完整 Tile 分配，能够容纳搬运接口在 UB 端生成的补齐数据。

##### Compute

Compute 输入为 `srcLocal`、`tileLength` 和三个计算标量，输出为 `dstLocal`。float32 路径使用 `vector_float/float`，float16 路径使用 `vector_half/half`，两条路径执行相同步骤：

1. **生成 repeat Mask 和 Tail Mask**

   完整 repeat 使用全宽 Mask 并在循环中复用；最后不足 256B 时，根据剩余元素数生成 Tail Mask，只启用有效元素。

   API：float32 使用 `asc_update_mask_b32`，float16 使用 `asc_update_mask_b16`。

2. **构造零向量**

   每次 VF 函数调用构造一次零向量，供后续 `min/max` 复用。

   API：`asc_duplicate_scalar`。

3. **将当前 repeat 从 UB 装入 Reg Vector**

   $$
   x=srcLocal[repeat]
   $$

   API：`asc_loadalign`。

4. **计算 1：取得非正分支输入**

   $$
   y=scale\cdot\left(\max(x,0)+alpha\cdot\left(e^{\overbrace{\min(x,0)}^{\text{计算 1}}\cdot input\_scale}-1\right)\right)
   $$

   API：`asc_min`。

5. **计算 2：应用输入缩放标量**

   $$
   y=scale\cdot\left(\max(x,0)+alpha\cdot\left(e^{\overbrace{\min(x,0)\cdot input\_scale}^{\text{计算 2}}}-1\right)\right)
   $$

   API：`asc_mul_scalar`，将同一个 `input_scale` 广播到 Mask 覆盖的元素。

6. **计算 3：执行指数运算**

   $$
   y=scale\cdot\left(\max(x,0)+alpha\cdot\left(\overbrace{e^{\min(x,0)\cdot input\_scale}}^{\text{计算 3}}-1\right)\right)
   $$

   API：`asc_exp`。

7. **计算 4：完成指数分支**

   $$
   y=scale\cdot\left(\max(x,0)+alpha\cdot\overbrace{\left(e^{\min(x,0)\cdot input\_scale}-1\right)}^{\text{计算 4}}\right)
   $$

   API：`asc_add_scalar`，广播标量 `-1`。

8. **计算 5：取得正分支输入**

   $$
   y=scale\cdot\left(\overbrace{\max(x,0)}^{\text{计算 5}}+alpha\cdot\left(e^{\min(x,0)\cdot input\_scale}-1\right)\right)
   $$

   API：`asc_max`。

9. **计算 6：合并正、负分支**

   $$
   y=scale\cdot\overbrace{\left(\max(x,0)+alpha\cdot\left(e^{\min(x,0)\cdot input\_scale}-1\right)\right)}^{\text{计算 6}}
   $$

   API：`asc_axpy`，将标量 `alpha` 的乘法与两分支相加合并为一次调用。

10. **计算 7：应用最终缩放标量**

    $$
    y=\overbrace{scale\cdot\left(\max(x,0)+alpha\cdot\left(e^{\min(x,0)\cdot input\_scale}-1\right)\right)}^{\text{计算 7}}
    $$

    API：`asc_mul_scalar`，将同一个 `scale` 广播到 Mask 覆盖的元素。

11. **将结果写回输出 UB**

    $$
    dstLocal[repeat]=y
    $$

    API：`asc_storealign`。

`sizeof(vector_float/vector_half)` 用于取得每个 repeat 的元素数，避免依赖早期 CANN 版本不存在的 `asc_get_vf_len()`。零向量配合向量版 `asc_min/asc_max`，依据是 CANN 9.0.0-beta.2 没有可供 Reg Vector VF 直接调用的 scalar 重载。

##### CopyOut

CopyOut 的输入是 `dstLocal` 和当前 Tile 的有效字节数，输出地址为 `dst+gmOffset`：

```cpp
asc_copy_ub2gm_align(
    dst + gmOffset, dstLocal, 1, tileBytes, 0, 0, 0);
```

Compute 通过 `PIPE_V -> PIPE_MTE3` 通知结果就绪，MTE3 只搬出 `tileBytes` 个有效字节，随后以 `PIPE_MTE3 -> PIPE_V` 释放当前输出槽。

Kernel 退出前消费两组槽的最终释放事件，确保异步操作完成。该时序直接对应当前 `elu_float_custom/elu_half_custom` 中的事件调用。

#### 标量与矢量的计算行为（评审意见说明）

本节专门回应评审提出的“`alpha`、`scale` 和 `input_scale` 均为标量，需要说明标量与矢量在计算时的行为”。`src` 是矢量输入；三个属性是一次 Kernel 调用共享的 `FLOAT` 标量，不是与 `src` 等长的 Tensor。标量 C API 将同一个标量值应用到当前 Mask 覆盖的每个有效元素：

| 对象            | 接口/类型                    | 每个有效元素`i` 的行为                 | 复用范围                                |
| --------------- | ---------------------------- | ---------------------------------------- | --------------------------------------- |
| `src`         | `vector_float/vector_half` | 每个元素对应一个`src[i]`，独立执行 ELU | 一个 VF repeat                          |
| `input_scale` | `asc_mul_scalar`           | `negative[i] *= input_scale`           | 全部核、Tile 和有效元素使用同一个属性值 |
| `alpha`       | `asc_axpy`                 | `positive[i] += alpha * negative[i]`   | 同上                                    |
| `scale`       | `asc_mul_scalar`           | `positive[i] *= scale`                 | 同上                                    |

Host 和 Kernel 入口始终以 `float` 传递三个属性；float32 路径直接使用，float16 路径在每核计算前各转换一次为 `half`，随后复用于该核全部计算。常量 `0` 显式广播为零矢量，常量 `-1` 由 `asc_add_scalar` 广播；尾 Mask 只启用有效元素。

## 2.3 测试用例设计

### 参考规范

精度与测试方案按照 [ELU 算子开发任务书](../../../../docs/202608/asc_elu_task_doc.md)第 3 节执行，并参考任务书指定的以下资料：

- [opbase《生态算子开源精度标准》](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md#2-%E8%AF%AF%E5%B7%AE%E6%8C%87%E6%A0%87%E4%B8%8E%E9%80%9A%E8%BF%87%E6%A0%87%E5%87%86)
- [asc-devkit Abs 样例验证脚本](https://gitcode.com/cann/asc-devkit/tree/master/examples/01_simd_cpp_api/03_basic_api/02_reg_vector_compute/abs/scripts)

### 精度比对方法

CPU Golden 从已量化到目标 dtype 的输入出发，以 float64 按原始分段公式计算，再转换为输出 dtype。测试数据使用固定随机种子 `20260803`，`mixed` 模式在 `[-100,100]` 内随机生成数据，并注入 `-100/-20/-1/-1e-3/-0/0/1e-3/1/20/100` 等边界值。

Golden 使用原始分段公式，而 Kernel 使用无分支等价公式。验证脚本直接采用第 3.1 节所列 opbase 阈值，报告 `matched_ratio` 和 `max_absolute_error`，且仅在两项均达标时通过。补充用例使用 `standard` 模式生成 50% `[-5,5]` 均匀分布和 50% 正态分布，并覆盖零值/负值属性及 `NAN/INF/-INF` 特殊输入。

### 输入上下界与 Corner Case

任务书规定的输入值域上下界为 `[-100,100]`；`mixed` 模式会把上下界和分段边界附近的值直接注入输入，而不是仅依赖随机采样。默认属性 `alpha=scale=input_scale=1` 时，理论输出范围为 `[exp(-100)-1,100]≈[-1,100]`。任务书没有规定三个属性的数值上下界，因此不对任意属性组合声明固定输出范围，而是以零值、正值和负值等价类补充验证。各边界的预期行为如下：

| 场景             | 输入                         | Golden/Kernel 预期行为                                             | 覆盖用例                |
| ---------------- | ---------------------------- | ------------------------------------------------------------------ | ----------------------- |
| 输入下界         | `x=-100`                   | `exp(-100)-1≈-1`，验证负区间饱和端                              | P01～P08 mixed          |
| 输入上界         | `x=100`                    | `y=100`；无分支公式只计算 `exp(min(x,0))=exp(0)`，不计算大指数 | P02～P04/P06～P08 mixed |
| 负侧接近零       | `x=-1e-3`                  | 以 float64`expm1` Golden 检查 `exp(x)-1` 的抵消误差            | P02～P04/P06～P08 mixed |
| 正侧接近零       | `x=1e-3`                   | 进入线性分支，`y=x`                                              | P02～P04/P06～P08 mixed |
| 分段边界         | `x=-0/+0`                  | 数值结果均为 0                                                     | G03/G08、S03            |
| 单分支           | 仅正值、仅负值               | 分别独立验证线性分支和指数分支                                     | G01/G02/G06/G07         |
| 非有限值         | `NAN/+INF/-INF`            | NaN 与 NaN 匹配；无穷要求符号与 Golden 一致                        | S03                     |
| 属性 Corner Case | 单个属性为 0，以及负属性组合 | 验证输出清零、指数输入缩放和符号组合                               | S02                     |

其中非有限值属于任务书值域外的健壮性补充。验证脚本不会将 NaN/Inf 代入有限值容差公式，而是分别检查 NaN 对应关系和同符号无穷。

### 功能与边界用例

任务书必测用例：

| ID  | dtype   | length | 输入模式                  | `alpha/scale/input_scale` | 覆盖目标                         |
| --- | ------- | -----: | ------------------------- | --------------------------- | -------------------------------- |
| P01 | float32 |      1 | mixed，范围`[-100,100]` | `1.0/1.0/1.0`             | 标量计算及非对齐搬运             |
| P02 | float32 |     32 | mixed，范围`[-100,100]` | `1.0/1.0/1.0`             | 单次 VF 计算                     |
| P03 | float32 |   1024 | mixed，范围`[-100,100]` | `1.0/1.0/1.0`             | 多 VF repeat                     |
| P04 | float32 |   2048 | mixed，范围`[-100,100]` | `1.0/1.0/1.0`             | 多核切分与合并                   |
| P05 | float16 |      1 | mixed，范围`[-100,100]` | `1.0/1.0/1.0`             | 标量及`float -> half` 属性转换 |
| P06 | float16 |     32 | mixed，范围`[-100,100]` | `1.0/1.0/1.0`             | b16 Mask 和 Reg Vector           |
| P07 | float16 |   1024 | mixed，范围`[-100,100]` | `1.0/1.0/1.0`             | 多 VF repeat                     |
| P08 | float16 |   2048 | mixed，范围`[-100,100]` | `1.0/1.0/1.0`             | 多核切分与合并                   |

泛化和边界用例：

| ID  | dtype   | length | 输入模式                              | `alpha/scale/input_scale` | 覆盖目标                      |
| --- | ------- | -----: | ------------------------------------- | --------------------------- | ----------------------------- |
| G00 | 两种    |      0 | 空 Tensor                             | `1.0/1.0/1.0`             | Host 空操作，不启动 Kernel    |
| G01 | float32 |      1 | `x=1`，仅正分支                     | `1.5/0.5/0.75`            | `y=scale*x`                 |
| G02 | float32 |      1 | `x=-1`，仅非正分支                  | `1.5/0.5/0.75`            | 指数分支                      |
| G03 | float32 |      1 | `x=0`                               | `1.5/0.5/0.75`            | 分支边界                      |
| G04 | float32 |     17 | mixed，非 32B 对齐                    | `1.5/0.5/0.75`            | 尾部 Mask 和 GM 搬运          |
| G05 | float32 |   8193 | mixed，全局长度超过单 Tile            | `0.25/1.25/0.5`           | 多核切分及全局尾块            |
| G06 | float16 |      1 | `x=1`，仅正分支                     | `1.5/0.5/0.75`            | `y=scale*x`                 |
| G07 | float16 |      1 | `x=-1`，仅非正分支                  | `1.5/0.5/0.75`            | 指数分支                      |
| G08 | float16 |      1 | `x=0`                               | `1.5/0.5/0.75`            | 分支边界                      |
| G09 | float16 |     17 | mixed，非 32B 对齐                    | `1.5/0.5/0.75`            | 尾部 Mask 和 GM 搬运          |
| G10 | float16 |   8193 | mixed，全局长度超过单 Tile            | `0.25/1.25/0.5`           | 多核切分及全局尾块            |
| G11 | float32 |  65537 | mixed，固定 8 核的单核跨 UB Tile 边界 | `0.25/1.25/0.5`           | 多 Tile Buffer 复用与事件依赖 |
| G12 | float16 |  65537 | mixed，固定 8 核的单核跨 UB Tile 边界 | `0.25/1.25/0.5`           | 多 Tile Buffer 复用与事件依赖 |

正式精度标准补充覆盖：

| 分组 | dtype           |  length | 输入/属性                                     | 覆盖目标         |
| ---- | --------------- | ------: | --------------------------------------------- | ---------------- |
| S01  | float32/float16 |    2048 | 50% 均匀 + 50% 正态，`1/1/1`                | opbase 常规分布  |
| S02  | float32/float16 | 32/1024 | 分别令单个属性为 0，并覆盖`-0.75/1.25/-0.5` | 标量属性等价类   |
| S03  | float32/float16 |       5 | `NAN/INF/-INF/-0/0`，`1/1/1`              | 非有限值和符号零 |

G11/G12 在脚本中显式固定为 8 核，使至少一个核分到超过 8192 个元素；其他用例使用默认自动分核。

G00 对应 FLOAT/FLOAT16 两个 Host 空操作用例，不进入 Kernel 精度统计；其余 32 个用例用于验证 Device 计算和任务书精度标准。

# 三、可维可测

## 3.1 精度标准/性能标准

以下精度标准直接采用任务书第 3 节指定的 [opbase《生态算子开源精度标准》](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md#2-%E8%AF%AF%E5%B7%AE%E6%8C%87%E6%A0%87%E4%B8%8E%E9%80%9A%E8%BF%87%E6%A0%87%E5%87%86)，阈值未作放宽。单个元素满足
`|actual-golden| <= atol + rtol*|golden|` 时记为匹配；用例还需同时满足匹配率和最大绝对误差上限。

| dtype   |  `rtol` |  `atol` | 最低匹配率 |       最大绝对误差上限 |
| ------- | --------: | --------: | ---------: | ---------------------: |
| FLOAT   | `2^-10` | `2^-16` |   `0.99` | `1e-2` 或 `32*ULP` |
| FLOAT16 |  `2^-9` |  `2^-9` |   `0.99` | `1e-1` 或 `32*ULP` |

当前验证脚本选用标准给出的固定硬上限分支，即 FLOAT/FLOAT16 分别使用 `1e-2/1e-1`。精度判定的上下界为：单元素绝对误差满足 `0 <= abs_error <= atol + rtol*|golden|` 时记为匹配；整例还需满足 `0.99 <= matched_ratio <= 1`，并且 `0 <= max_abs_error <= max_abs_error_limit`。任务书未设置性能门槛，性能优化不改变上述精度通过条件。

### 真机验证环境与结果

| 硬件                              | 系统环境                                               | CANN 环境                          |
| --------------------------------- | ------------------------------------------------------ | ---------------------------------- |
| Ascend950PR；`npu-smi 25.7.rc1` | Linux x86_64 容器，镜像`cann_9.0.0-beta.2-py3.12-a5` | `cann_pkg 9.0.0-beta.2-20260328` |
| Ascend950PR                       | Linux x86_64 容器                                      | CANN 9.0.0 正式版                  |
| Ascend950PR                       | Linux x86_64 容器                                      | CANN 9.1.0 正式版                  |

正式版安装包来源如下。CANN 9.0.0 的链接为安装文件直链，浏览器可能不展示下载页而是直接下载；也可将链接交给 `wget -c` 下载。

| 版本       | 安装包来源                                                                                                                                                                                                                                                                                                                       |
| ---------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| CANN 9.0.0 | [Ascend-cann-toolkit_9.0.0_linux-x86_64.run](<https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%209.0.0/Ascend-cann-toolkit_9.0.0_linux-x86_64.run>)、[Ascend-cann-950-ops_9.0.0_linux-x86_64.run](<https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%209.0.0/Ascend-cann-950-ops_9.0.0_linux-x86_64.run>) |
| CANN 9.1.0 | [Linux x86_64 安装包目录](https://ascend.devcloud.huaweicloud.com/cann/run/software/9.1.0/x86_64/)                                                                                                                                                                                                                                |

32 组用例完成后，逐份检查 `metrics.json` 中的阈值、匹配率、最大绝对误差和 `passed` 字段，核验结果如下：

| dtype   | 用例数 | 匹配率条件     | 最大绝对误差条件 | 结果       |
| ------- | -----: | -------------- | ---------------- | ---------- |
| FLOAT   |     16 | 全部`>=0.99` | 全部`<=1e-2`   | 16/16 通过 |
| FLOAT16 |     16 | 全部`>=0.99` | 全部`<=1e-1`   | 16/16 通过 |

以上结果覆盖 P01~P08、G01~G12 和 S01~S03，最终审计输出为 `[CONFIRMED] 32/32 NPU cases satisfy the task-book precision standard.`。

已有报告中的代表性 Corner Case 结果如下；表中数值来自已有逐用例报告，同一组 32 个 Device 用例已在 Ascend 950 + CANN 9.0.0、9.1.0 正式版全部通过：

| Corner Case                           | dtype   | matched_ratio |   max_abs_error |   固定硬上限 | 结果 |
| ------------------------------------- | ------- | ------------: | --------------: | -----------: | :--: |
| `[-100,100]` 上下界及近零值，N=1024 | FLOAT   |       `1.0` |   `5.9605e-8` |     `1e-2` | 通过 |
| `[-100,100]` 上下界及近零值，N=1024 | FLOAT16 |       `1.0` |   `4.8828e-4` |     `1e-1` | 通过 |
| 仅正值`x=1`                         | 两种    |       `1.0` |           `0` | 对应类型上限 | 通过 |
| 仅负值`x=-1`                        | 两种    |       `1.0` | `<=2.4414e-4` | 对应类型上限 | 通过 |
| 分段边界`x=0`                       | 两种    |       `1.0` |           `0` | 对应类型上限 | 通过 |
| `NAN/+INF/-INF/-0/+0`               | 两种    |       `1.0` |           `0` | 对应类型上限 | 通过 |
| 负属性组合`-0.75/1.25/-0.5`         | FLOAT   |       `1.0` |   `7.6294e-6` |     `1e-2` | 通过 |
| 负属性组合`-0.75/1.25/-0.5`         | FLOAT16 |       `1.0` |     `6.25e-2` |     `1e-1` | 通过 |

最新代码已分别在 Ascend 950 + CANN 9.0.0、9.1.0 正式版执行完整批量测试：32 组 Device 用例均满足任务书精度标准，FLOAT/FLOAT16 两组 `length=0` Host 空操作用例也均通过，即每套环境均为 34/34 通过。其中 G11/G12 使用 length 65537、固定 8 核，覆盖 Ping/Pong 跨 Tile 复用；空 Tensor 不进入 Kernel 精度统计。

### 性能用例与阶段性结果

性能测试仅在 Ascend 950 真机执行。每组用例先预热，再重复执行并统计平均耗时、P50、P99 和有效元素吞吐率。

| ID  | dtype           |  length | 目的                  |
| --- | --------------- | ------: | --------------------- |
| T01 | float32/float16 |    2048 | 小 shape 启动开销基线 |
| T02 | float32/float16 |   65536 | 多核负载均衡          |
| T03 | float32/float16 |   65537 | 单核跨 Tile 路径      |
| T04 | float32/float16 |  524288 | 大 shape 稳态吞吐率   |
| T05 | float32/float16 | 1048576 | 更大连续 Tensor       |

T04 float32 的阶段性 profiling 表明，数据量感知的自动扩核明显优于固定小核数，因而自动分核已作为当前 Host 设计的一部分。CANN 9.0.0 正式版在 1650 MHz 下使用相同输入各测试 5 次，P50 结果如下：

| 分核方式     | 单缓冲 P50/us | 双缓冲 P50/us |  耗时下降 |    加速比 |
| ------------ | ------------: | ------------: | --------: | --------: |
| 自动最大核数 |     `4.284` |     `3.946` |  `7.9%` | `1.09x` |
| 固定 8 核    |    `10.014` |     `6.346` | `36.6%` | `1.58x` |

双缓冲在两种分核方式下均取得稳定收益。固定 8 核的实测降幅更大；结合其每核处理 Tile 更多这一事实，该结果与较长稳态区间能够提供更多流水重叠机会相符，但性能结论仍以表中的真机数据为准。

在已测的 32～1024 元素范围内，小输入 profiling 表明过度扩核会使多核调度开销超过计算收益。因此自动模式将核数限制在覆盖输入所需的 256B Vector repeat 数以内；显式核数模式保持不变。

## 3.2 兼容性分析

本实现是 Ascend 950 `dav-3510` 专用 Reg Vector C API 样例。CANN 版本和 `asc-devkit` 源码必须配套；不承诺与 `dav-2201` C API 二进制兼容。
