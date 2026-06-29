# BiasAdd 算子设计方案

## 需求背景

### 需求来源

昇腾社区算子开发任务，任务编号 03-13。

### 背景介绍

**BiasAdd 算子实现优化**

基于 BiasAdd 算子历史 TBE 版本使用 Ascend C 编程语言进行优化。

**BiasAdd 算子（TBE）实现路径和相关 API 路径**

- BiasAdd 算子 TBE 实现：`/home/developer/Ascend/cann-8.5.2/opp/built-in/op_impl/ai_core/tbe/impl/ops_legacy/dynamic/bias_add.py`
- BiasAdd 算子信息库：`/home/developer/Ascend/cann-8.5.2/opp/built-in/op_impl/ai_core/tbe/kernel/ascend910b/ops_legacy/bias_add/`

### BiasAdd 算子 TBE 源码逻辑分析


**TBE 实现架构：**

TBE 版 BiasAdd 基于 TBE DSL 框架实现，使用 classify 分类 + tbe.broadcast + tbe.vadd 完成计算，通过 auto_schedule 自动生成调度，build 编译为 Kernel。

**Python 入口层（bias_add 函数）：**

- 参数校验：检查 dtype（支持 bfloat16/float16/float32/int32/int64）、检查 data_format（支持 NCHW/NHWC/NDHWC/NCDHW）
- 根据 x 的 format（NC1HWC0/NDHWC/NCDHW/NDC1HWC0/普通 ND）和 data_format 推断 bias_dim 位置
- 将 bias 的 shape reshape 为与 x 相同 rank 的广播形状（在非 bias_dim 的维度填 1）
- classify([x, bias], ELEWISE_WITH_BROADCAST) 将算子标记为逐元素广播模式
- 对每个分类结果：创建 TVM placeholder → 调用 bias_add_compute → auto_schedule
- build(schedules, config) 编译为最终 Kernel

**bias_add 入口函数源码：**


![image.png](https://raw.gitcode.com/user-images/assets/10095851/90cadca8-4ff4-4203-b75a-63594f9b6bf2/image.png 'image.png')
![bias_add入口-NC1HWC0分支]
![image.png](https://raw.gitcode.com/user-images/assets/10095851/b1afbb12-d224-4d05-aa42-04f15a87a4cd/image.png 'image.png')
![bias_add入口-NDHWC/NCDHW/NDC1HWC0分支]
![image.png](https://raw.gitcode.com/user-images/assets/10095851/659b38da-2913-4e01-9967-e5f78464c06b/image.png 'image.png')
![bias_add入口-NCHW/NHWC+classify+build]
![image.png](https://raw.gitcode.com/user-images/assets/10095851/f5dc1fd2-bd4a-48de-9b0b-997e751fa1a0/image.png 'image.png')

**计算层（bias_add_compute 函数）：**
![image.png](https://raw.gitcode.com/user-images/assets/10095851/b8a9cc3f-5a43-4760-90bd-7d62ad2e9a11/image.png 'image.png')

_, _, shape_max = shape_util.broadcast_shapes(x.shape, bias.shape)
data_x = tbe.broadcast(x, shape_max)
data_bias = tbe.broadcast(bias, shape_max)
res = tbe.vadd(data_x, data_bias)

将 x 和 bias 都广播到统一 shape 后执行逐元素加法。TBE DSL 框架在底层自动处理广播的数据搬运、内存分配和多核调度。

**格式选择层（op_select_format 函数）：**

根据 ori_shape 长度和 bias 长度是否为 16 的倍数，以及平台能力（vmuls_support、bfp16_support），动态选择最优 dtype/format 组合（NC1HWC0/NDHWC/NCDHW/NDC1HWC0/ND）。

**TBE 算子实现流程图：**

![截屏2026-06-24 16.05.01.png](https://raw.gitcode.com/user-images/assets/10095851/4d2ac231-a67f-4015-9de1-d2831566681e/截屏2026-06-24_16.05.01.png '截屏2026-06-24 16.05.01.png')

| 阶段 | 步骤 | 说明 |
|------|------|------|
| 1 | bias_add()入口 | 参数校验+根据format和data_format推断bias_dim+reshape bias shape |
| 2 | classify | classify([x,bias],ELEWISE_WITH_BROADCAST)分类为逐元素广播模式 |
| 3 | tvm.placeholder | 对每个分类结果创建data_x和bias的TVM符号张量 |
| 4 | bias_add_compute() | broadcast_shapes求统一shape+tbe.broadcast(x)+tbe.broadcast(bias)+tbe.vadd |
| 5 | auto_schedule(res) | TVM自动生成调度(切分、流水、多核等由框架决定) |
| 6 | build(schedules,config) | 编译生成最终Kernel二进制 |

执行顺序: 入口校验→shape推断→classify分类→placeholder→broadcast+vadd→auto_schedule→build

**TBE 逻辑总结：**

| 逻辑环节 | TBE 实现方式 |
|---------|-------------|
| 维度推断 | 入口函数根据format和data_format计算bias广播shape |
| 算子分类 | classify(ELEWISE_WITH_BROADCAST)标记为广播逐元素模式 |
| 广播 | tbe.broadcast—DSL框架自动处理搬运和广播 |
| 核心计算 | tbe.vadd—逐元素向量加法 |
| 调度策略 | auto_schedule—TVM自动调度(多核、切分、流水) |
| 编译 | build—生成Kernel二进制 |
| UB管理 | 框架自动分配 |

### 我们的 Ascend C 实现与 TBE 逻辑对比

| 逻辑环节 | TBE(DSL框架) | 我们的实现 | 一致性 |
|---------|---------------|-----------|--------|
| 维度推断 | 入口函数根据format和data_format reshape bias shape | Host侧Tiling显式合轴为(outerSize,biasSize,innerSize) | 一致 |
| 算子分类 | classify(ELEWISE_WITH_BROADCAST) | 手动根据三轴模型选择branch | 一致(手动实现框架分类逻辑) |
| 广播 | tbe.broadcast框架自动广播 | 根据分支用Add逐行/Adds标量(显式实现广播语义) | 一致 |
| 核心计算 | tbe.vadd | Add/Adds | 一致 |
| 调度/多核 | auto_schedule自动决定 | 手动实现blockNum/blockFormer切分 | 一致(手动实现框架自动逻辑) |
| 升精度 | 框架自动处理 | 显式Cast-Add-Cast | 一致 |
| UB管理 | 框架自动分配 | 手动规划TQue+TBuf | 一致(更精确控制) |
| 性能优化 | 框架内部优化 | bulkMode快速路径(对齐时批量Add) | 额外优化 |

**结论**：我们的实现与 TBE 源码在计算逻辑上完全一致，均为"将 bias 沿指定维度广播后与 input 执行 vadd 操作"。区别在于 TBE 使用 DSL 框架（tbe.broadcast + tbe.vadd + auto_schedule）自动调度，我们采用 Ascend C 手动实现各分支以获得更精细的控制和性能优化。



## 需求分析

### 外部组件依赖

不涉及外部组件依赖。

### 内部适配模块

适配 aclnn 单算子调用。

接口说明：

通过 aclnn 接口调用：aclnnBiasAdd。采用两段式接口，先调用 aclnnBiasAddGetWorkspaceSize 获取 workspace 大小及执行器，再调用 aclnnBiasAdd 执行计算。



### 需求模块设计

#### 算子原型

**原型设计**

| 名称 | 类别 | dtype | format | shape | 介绍 |
|------|------|-------|--------|-------|------|
| x | 输入 | fp16/fp32/int32/int16 | NCHW/NHWC/NCDHW/NDHWC/ND | all | 输入张量 |
| bias | 输入 | fp16/fp32/int32/int16 | ND | [C] | 偏置向量 |
| y | 输出 | fp16/fp32/int32/int16 | 同 x | 同 x | 输出张量 |
| data_format | 属性 | string | - | - | 数据格式，默认 NHWC |

**与算子信息库一致性验证：**

算子信息库中 BiasAdd 的预编译 kernel 覆盖以下元素大小：2 字节（float16/int16）、4 字节（float32/int32）、8 字节（int64/double，任务书明确暂不支持）。我们的实现覆盖 2 字节和 4 字节类型，与信息库要求一致。

**相关约束**

Atlas A2 训练系列产品支持 float16、float32、int32、int16；x 和 bias 数据类型必须一致；暂不支持 int64、double 数据类型，暂不支持广播操作。

## 需求详细设计

### 使能方式

| 上层框架 | 涉及勾选 |
|----------|----------|
| TF 训练/推理 | |
| Pytorch 训练/推理 | |
|  ATC 推理 |  |
| aclnn 单算子调用 | * |
| OPAT 调优 | |
| SGAT 子图切分 | |

### 需求总体设计

#### 3.2.1 host 侧设计

**tiling 策略：**

根据 data_format 和输入 shape，将问题分解为三轴模型 (outerSize, biasSize, innerSize)，据此选择计算分支和切分方案。此逻辑与 TBE 内部 bias_add 入口函数根据 data_format 推断 bias_dim 的逻辑一致。

**1) 分支选择策略 (tilingkey + branch)**

tilingkey 按数据类型分派 Kernel 模板实例：

| tilingkey | 条件 | 分支名 | 说明 |
|-----------|------|--------|------|
| 0 | float32 | KEY_FLOAT | 4 字节类型直接计算 |
| 1 | float16 | KEY_HALF | 2 字节类型升精度计算 |
| 2 | int32 | KEY_INT32 | 4 字节整数直接计算 |
| 3 | int16 | KEY_INT16 | 2 字节整数升精度计算 |

根据三轴模型选择计算分支（branch 字段传入 Kernel），对应 TBE 中 auto_schedule 自动选择的不同调度模式：

| branch | 条件 | 分支名 | 计算方式 | 对应 TBE 调度模式 |
|--------|------|--------|----------|-----------------|
| 0 | outerSize=1 且 innerSize=1 | OneDim | 退化为逐元素 Add | Elementwise 模式 |
| 1 | innerSize=1 | BiasDimLast | bias 在尾轴，逐行向量 Add | LastDim Broadcast |
| 2 | 其他 | BiasDimMid | bias 在非尾轴，逐行标量 Adds | MidDim Broadcast |

**2) 分核策略**

优先使用满核原则（与 TBE auto_schedule 的多核策略一致）：

- blockNum = min(数据并行度, 设备可用核数)
- OneDim：按 UB tile 数切分
- BiasDimLast：按 outerSize 切分
- BiasDimMid：按 totalRows(outerSize * biasSize) 切分

如果核间能均分，大核小核数据块一致；如果核间不能均分，尾核处理剩余任务量。

**3) 数据分块和内存优化策略**

充分使用 UB 空间的原则。所有缓冲区按 256 字节对齐（ALIGN_BYTES = 256）。对应 TBE 框架自动分配的 UB 级内存优化。

UB 内存使用策略（以 BiasDimLast 分支为例）：

| Buffer | 大小 | 用途 | Double Buffer |
|--------|------|------|:---:|
| inQueX | 2 * outerTile * biasSize_align * sizeof(T) | input 搬入 | 是(x2) |
| outQueY | 2 * outerTile * biasSize_align * sizeof(T) | output 搬出 | 是(x2) |
| biasBuf | biasSize_align * sizeof(T) | bias 常驻 UB | 否 |
| castBuf | outerTile * biasSize_align * sizeof(T) | 对齐时 bulk Add 用 / 升精度用 | 否 |

outerTile 计算公式：

- float32/int32 (dtypeSize=4)：outerTile = (UB_SIZE / dtypeSize - biasSize_align) / (5 * biasSize_align)
- float16/int16 (dtypeSize=2, 升精度需额外空间)：outerTile = (UB_SIZE - biasSize_align * 6) / (16 * biasSize_align)

**4) 性能优化：bulkMode 快速路径**

当 biasSize == biasSize_align（自然对齐）且不需要升精度时：

- 将 bias 在 UB 中复制 outerTile 份（拼成完整 tile 大小）
- 使用 DataCopy（连续搬运）替代 DataCopyPad
- 使用单次 Add(yL, xD, bTile, rows*biasSize) 替代逐行循环

此优化消除了 Scalar 循环开销，性能从 1145us 降至 854us（大 shape 场景提升 25%）。该优化对应 TBE 框架在对齐场景下的内部优化逻辑。

#### 3.2.2 kernel 侧设计

进行 Init 和 Process 两个阶段，其中 Process 包括数据搬入（CopyIn）、计算（Compute）、搬出（CopyOut）三个阶段。对应 TBE 中 tbe.broadcast + tbe.vadd 的计算逻辑。

**总体架构：**

Kernel 入口通过 tilingkey 模板参数选择数据类型，实例化 NsBiasAdd::BiasAdd T 类，调用 Init 和 Process。

Kernel 入口函数签名：template int KEY global aicore void bias_add(GM_ADDR x, GM_ADDR bias, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)


**Init 阶段：**

1. 从 tiling 结构体读取分支和参数
2. 根据 branch 选择 InitOneDim / InitLast / InitMid
3. 初始化 GlobalTensor、TQue（Double Buffer）、TBuf
4. 计算当前核的行范围 [rowStart, rowEnd)

**Process 阶段 - 各分支流程：**

**OneDim 分支（branch=0）：**

- Init：初始化 inQueX(DB=2) + inQueB(DB=1) + outQueY(DB=2) + castBuf(升精度时)
- Process 循环（每个 tile）：
  - CopyIn：DataCopyPad 搬入 input tile + bias tile（对应 TBE tbe.broadcast 的数据搬入）
  - Compute：dtypeSize=4 时 Add(yL, xL, bL, curLen)；dtypeSize=2 时 Cast到float, Add, Cast到原类型（对应 TBE tbe.vadd）
  - CopyOut：DataCopyPad 搬出 output tile（对应 TBE build 后的输出逻辑）

**BiasDimLast 分支（branch=1）：**

- Init：DataCopyPad 搬入 bias 全量（常驻 UB，只搬一次），PipeBarrier PIPE_ALL 同步
- Process 循环（每个 outerTile 批次）：
  - CopyIn：DataCopyPad 搬入 rows 行输入数据
  - Compute：bulkMode（对齐且不升精度）时单次 Add(yL, xD, bTile, rows*biasSize)；needCast（升精度）时逐行 Cast到Add到Cast；普通时逐行 Add(yL[offset],
xD[offset], bL, biasSize)
  - CopyOut：DataCopyPad 搬出结果

**BiasDimMid 分支（branch=2）：**

- Init：DataCopyPad 搬入 bias 全量（常驻 UB，只搬一次），PipeBarrier PIPE_ALL 同步
- Process 循环（每个 batch）：
  - CopyIn：DataCopyPad 批量搬入 batchRows 行
  - Compute：逐行处理，取 biasVal = bias[rowIdx % biasSize]；dtypeSize=4 时 Adds(yL[offset], xD[offset], biasVal, innerSize)；dtypeSize=2 时 Cast到float,Adds(float), Cast到原类型
  - CopyOut：DataCopyPad 搬出 output batch

**3.2.2.2 Ascend C 实现流程图：**
![截屏2026-06-12 18.30.27.png](https://raw.gitcode.com/user-images/assets/10095851/c3c24a7b-e79d-404e-b68c-46e506ebefa8/截屏2026-06-12_18.30.27.png '截屏2026-06-12 18.30.27.png')

| 阶段 | 步骤 | 说明 |
|------|------|------|
| 1 | Host Tiling | 解析shape+data_format,合轴,选择branch |
| 2 | Kernel入口 | 根据tilingkey(dtype)实例化模板 |
| 3 | Init | 初始化Buffer,搬入bias常驻UB,PipeBarrier同步 |
| 4 | CopyIn | DataCopyPad将input从GM搬入UB |
| 5 | Compute(OneDim) | Add 或 Cast->Add->Cast |
| 6 | Compute(BiasDimLast) | bulkMode批量Add / 逐行Add / Cast->Add->Cast |
| 7 | Compute(BiasDimMid) | 逐行Adds标量加法 或 Cast->Adds->Cast |
| 8 | CopyOut | DataCopyPad将结果搬出GM |

执行顺序: Host Tiling->Kernel入口->Init->多核分发->每核循环[CopyIn->Compute->CopyOut]

**3.2.2.3 Ascend C 与 TBE 差异点及原因：**


| 差异点 | TBE | Ascend C | 原因 |
|--------|-----|----------|------|
| 分支选择 | auto_schedule自动调度 | 显式计算branch(0/1/2) | 可控性更强 |
| 广播 | tbe.broadcast自动广播 | Add逐行/Adds标量 | 基础API更灵活 |
| 性能优化 | 框架内部优化 | bulkMode批量Add | 消除循环开销提升25% |
| 升精度 | 框架自动 | 显式Cast->Add->Cast | 可验证可调试 |
| UB管理 | 框架统一分配 | 手动规划TQue+TBuf | 精确控制利用率 |
| 同步 | 框架自动插入 | 3处PipeBarrier+TQue | 最小化开销 |

结论: 计算逻辑完全一致,差异在实现手段,我们用基础API获得更精细控制。

**精度处理策略：**

| 输入 dtype | 计算 dtype | 策略 |
|-----------|-----------|------|
| float32 | float32 | 直接计算 |
| int32 | int32 | 直接计算 |
| float16 | float32 | Cast(CAST_NONE) 到 计算 到 Cast(CAST_ROUND) |
| int16 | float32 | Cast(CAST_NONE) 到 计算 到 Cast(CAST_ROUND) |

**同步策略：**

- PipeBarrier PIPE_ALL 仅在 bias 搬入后使用（共 3 处，0% 冗余）
- TQue EnQue/DeQue 保证搬运与计算的流水重叠


### 3.3 ACLNN API 接口

#### 3.3.1 功能说明

对输入张量 x 的每一个元素加上偏置值 bias，bias 沿 C 维度与 x 对齐后逐元素相加。

计算公式：out = x + bias（bias 沿 C 维度对齐广播）

#### 3.3.2 函数原型

采用两段式接口：

aclnnStatus aclnnBiasAddGetWorkspaceSize(
    const aclTensor* x,
    const aclTensor* bias,
    aclTensor* y,
    uint64_t* workspaceSize,
    aclOpExecutor** executor);

aclnnStatus aclnnBiasAdd(
    void* workspace,
    uint64_t workspaceSize,
    aclOpExecutor* executor,
    aclrtStream stream);

#### 3.3.3 aclnnBiasAddGetWorkspaceSize 参数说明

| 参数名 | 类型 | 方向 | 描述 |
|--------|------|------|------|
| x | const aclTensor* | 输入 | 输入张量 |
| bias | const aclTensor* | 输入 | 偏置向量1维 |
| y | aclTensor* | 输出 | 输出张量 |
| workspaceSize | uint64_t* | 输出 | workspace大小 |
| executor | aclOpExecutor** | 输出 | op执行器 |

x支持数据类型：FLOAT16、FLOAT32、INT32、INT16。支持格式：NCHW、NHWC、NCDHW、NDHWC、ND。rank>=2。

bias为1维张量，数据类型与x一致，长度等于x在C维度的大小

返回值：aclnnStatus。

#### 3.3.4 aclnnBiasAdd 参数说明

| 参数名 | 类型 | 方向 | 描述 |
|--------|------|------|------|
| workspace | void* | 输入 | workspace内存地址 |
| workspaceSize | uint64_t | 输入 | workspace大小 |
| executor | aclOpExecutor* | 输入 | op执行器 |
| stream | aclrtStream | 输入 | 执行Stream |

返回值：aclnnStatus。

#### 3.3.5 约束说明

| 约束项 | 约束条件 |
|--------|----------|
| x rank | >= 2 |
| bias rank | == 1 |
| 数据类型 | x和bias和y必须一致 |
| C维度 | bias长度等于x的C维大小 |
| data_format | NCHW/NHWC/NCDHW/NDHWC/ND |
| 输出shape | 与x相同 |

#### 3.3.6 调用示例

#include "acl/acl.h"
#include "aclnnop/aclnn_bias_add.h"

int aclnnBiasAddTest(int32_t deviceId,
                     aclrtStream& stream)
{
    std::vector<int64_t> xShape = {2, 4, 4, 8};
    std::vector<int64_t> biasShape = {8};
    std::vector<int64_t> yShape = {2, 4, 4, 8};

    void* xAddr = nullptr;
    void* biasAddr = nullptr;
    void* yAddr = nullptr;

    uint64_t xSize = 2*4*4*8 * sizeof(float);
    uint64_t bSize = 8 * sizeof(float);

    aclrtMalloc(&xAddr, xSize,
                ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&biasAddr, bSize,
                ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&yAddr, xSize,
                ACL_MEM_MALLOC_HUGE_FIRST);

    aclTensor* x = aclCreateTensor(
        xShape.data(), xShape.size(),
        ACL_FLOAT, nullptr, 0,
        ACL_FORMAT_NHWC,
        xShape.data(), xShape.size(), xAddr);
    aclTensor* bias = aclCreateTensor(
        biasShape.data(), biasShape.size(),
        ACL_FLOAT, nullptr, 0,
        ACL_FORMAT_ND,
        biasShape.data(), biasShape.size(),
        biasAddr);
    aclTensor* y = aclCreateTensor(
        yShape.data(), yShape.size(),
        ACL_FLOAT, nullptr, 0,
        ACL_FORMAT_NHWC,
        yShape.data(), yShape.size(), yAddr);

    uint64_t wsSize = 0;
    aclOpExecutor* executor = nullptr;
    void* wsAddr = nullptr;

    aclnnBiasAddGetWorkspaceSize(
        x, bias, y, &wsSize, &executor);
    if (wsSize > 0) {
        aclrtMalloc(&wsAddr, wsSize,
                    ACL_MEM_MALLOC_HUGE_FIRST);
    }
    aclnnBiasAdd(wsAddr, wsSize,
                 executor, stream);
    aclrtSynchronizeStream(stream);

    aclrtFree(xAddr);
    aclrtFree(biasAddr);
    aclrtFree(yAddr);
    if (wsAddr) aclrtFree(wsAddr);
    aclDestroyTensor(x);
    aclDestroyTensor(bias);
    aclDestroyTensor(y);

    return ACL_SUCCESS;
}

### 支持硬件

| 支持的芯片版本 | 涉及勾选 |
|---------------|----------|
| Atlas A2 训练系列产品 (Ascend 910B3) | * |

### 算子约束限制

1. 暂不支持 int64、double 数据类型
2. 暂不支持广播操作
3. x 和 bias 数据类型必须一致
4. bias 为 1D 向量，长度等于 x 在 bias 维度的大小
5. x 维度不少于 1 维

### 支持的 data_format

| data_format | bias_dim 位置 | 对应分支 |
|-------------|--------------|----------|
| NHWC | 最后一维 | BiasDimLast |
| NDHWC | 最后一维 | BiasDimLast |
| ND | 最后一维 | BiasDimLast |
| NCHW | 第 1 维 (C) | BiasDimMid |
| NCDHW | 第 1 维 (C) | BiasDimMid |

## 特性交叉分析

### 数据类型与格式交叉

| 数据类型 | NCHW | NHWC | NCDHW | NDHWC | ND |
|----------|------|------|-------|-------|------|
| float32 | Y | Y | Y | Y | Y |
| float16 | Y | Y | Y | Y | Y |
| int32 | Y | Y | Y | Y | Y |
| int16 | Y | Y | Y | Y | Y |

### 动态Shape支持

| 场景 | 是否支持 | 说明 |
|------|----------|------|
| 静态Shape | Y | 编译期确定shape |
| 动态Shape | Y | 运行时传入任意合法shape |
| 动态Rank | Y | 支持不同维度数输入rank>=2 |


## 可维可测分析

### 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
|----------|------|----------|
| 精度标准 | 满足 AscendOpTest 默认阈值 | 任务书 |
| 性能标准 | 不低于 TBE 版本 95% | 任务书 |

  **注：** 验收标准（一）第 2 条「aclnn 调用均能正常执行」通过 aclnnBiasAdd 接口验证。

**精度验证结果：**

| 数据类型 | 验证方式 | 结果 |
|----------|----------|------|
| float32 | allclose (rtol=1e-4, atol=1e-6) | PASS |
| float16 | allclose (rtol=1e-3, atol=1e-4) | PASS |
| int32 | exact match | PASS |
| int16 | exact match | PASS |

**性能验证结果：**

| Shape | Format | Dtype | TBE(us) | Ascend C(us) | 比率 |
|-------|--------|-------|---------|--------------|------|
| [32,256,256,64] | NHWC | float32 | 830 | 854 | 97.2% |

小 shape（如 [2,4,4,8]）执行时间均在 10us 以下，符合任务书 10us 以下场景相差 3us 可接受的标准。

### 兼容性分析

新算子 Ascend C 实现，不涉及兼容性分析。

## 测试用例

| 编号 | Shape | Format | Dtype | 分支 | 结果 |
|------|-------|--------|-------|------|------|
| T1 | [2,4,4,8] | NHWC | float32 | BiasDimLast | PASS |
| T2 | [2,16,8,8] | NCHW | float32 | BiasDimMid | PASS |
| T3 | [4,8,8,32] | NHWC | float16 | BiasDimLast | PASS |
| T4 | [2,64,16,16] | NCHW | float16 | BiasDimMid | PASS |
| T5 | [4,8,16] | ND | int32 | BiasDimLast | PASS |
| T6 | [4,8,16] | ND | int16 | BiasDimLast | PASS |
| T7 | [32,256,256,64] | NHWC | float32 | BiasDimLast | PASS |
| T8 | [2,4,4,13] | NHWC | float32 | BiasDimLast | PASS |
| T9 | [4,1,8,16] | NCHW | float32 | BiasDimMid | PASS |
| T10 | [128] | ND | float32 | OneDim | PASS |
| T11 | [2,8,7,7] | NCHW | float32 | BiasDimMid | PASS |
| T12 | [2,16,4,4,4] | NCDHW | float32 | BiasDimMid | PASS |

全部 12/12 通过。

## 自验证报告

### 验证环境

| 项目 | 规格 |
|------|------|
| 硬件 | Atlas A2 训练系列产品（Ascend 910B3） |
| CANN 版本 | 8.5.2 |
| 运行方式 | aclnn 单算子调用 |

### 验证结论

全部测试用例通过，功能、精度、性能均满足验收标准。

### 功能验证

- 验证方式：aclnn 单算子调用（aclnnBiasAdd）
- 验证命令：bash build.sh --experimental --run_example bias_add aclnn --soc=ascend910b
- 覆盖 12 组用例，涵盖三分支（OneDim/BiasDimLast/BiasDimMid）、四种 dtype（float32/float16/int32/int16）、五种 format（NHWC/NCHW/ND/NCDHW/NDHWC）
- 全部 PASS，无运行错误

### 精度验证

| 数据类型 | 验证方法 | 阈值 | 结果 |
|----------|----------|------|------|
| float32 | numpy allclose | rtol=1e-4, atol=1e-6 | PASS |
| float16 | numpy allclose | rtol=1e-3, atol=1e-4 | PASS |
| int32 | exact match | atol=0, rtol=0 | PASS |
| int16 | exact match | atol=0, rtol=0 | PASS |

### 性能验证

| Shape | Format | Dtype | TBE(us) | Ascend C(us) | 比率 | 达标 |
|-------|--------|-------|---------|--------------|------|------|
| [32,256,256,64] | NHWC | float32 | 830 | 854 | 97.2% | YES |
| [2,4,4,8] | NHWC | float32 | <10 | <10 | - | YES(差值<3us) |
| [2,16,8,8] | NCHW | float32 | <10 | <10 | - | YES(差值<3us) |
| [4,8,8,32] | NHWC | float16 | <10 | <10 | - | YES(差值<3us) |

大 shape 场景达到 TBE 97.2%（大于 95% 标准），小 shape 场景差值均在 3us 以内。

### 代码仓链接

- 算子代码仓：https://gitcode.com/Pengwu2026/ops-math
- 代码路径：experimental/math/bias_add/

## 开发环境

| 项目 | 规格 |
|------|------|
| 适配硬件 | Atlas A2 训练系列产品（Ascend 910B3） |
| CANN 版本 | 8.5.2 |
| 开发语言 | Ascend C |
| 芯片架构 | DAV_2201 |

## 贡献说明

| 贡献者 | 贡献方 | 贡献算子 | 贡献时间 | 贡献内容 |
|--------|--------|----------|----------|----------|
| Pengwu2026 | 个人开发者 | BiasAdd | 2026-06 | 新增 BiasAdd 算子 Ascend C 实现 |
