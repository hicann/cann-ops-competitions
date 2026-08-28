# 需求背景（required）

## 需求来源

CANN 社区任务——8月社区任务 huber_loss 算子开发。

## 背景介绍

### huber_loss 算子缺失现状

`aten::huber_loss` 算子未被 NPU 后端支持，PyTorch 模型在昇腾 NPU 上运行时自动 fallback 至 CPU 执行，导致训练耗时呈指数级增长，严重阻塞模型迭代周期。

需要在昇腾 NPU 上基于 Ascend C 编程语言实现功能一致的 Huber Loss 前向算子，完成算子设计、开发、测试全流程工作。

### PyTorch huber_loss 功能分析

PyTorch 原生 `aten::huber_loss` 接口（[官方文档](https://pytorch.org/docs/stable/generated/torch.nn.functional.huber_loss.html)）核心功能：

**数学公式：**

设 `e = input - target`，则逐元素损失为：

$$
loss_i =
\begin{cases}
0.5 \cdot e_i^2, & |e_i| \leq \delta \\
\delta \cdot (|e_i| - 0.5 \cdot \delta), & |e_i| > \delta
\end{cases}
$$

随后根据 `reduction` 模式对逐元素损失进行归约（本任务中 `reduction` 为 int 类型属性：0=none, 1=mean, 2=sum，默认 1）：

- `reduction = 0`（none）：输出逐元素损失，shape 与 input 一致
- `reduction = 1`（mean）：输出所有元素损失的均值（标量）
- `reduction = 2`（sum）：输出所有元素损失的和（标量）

**PyTorch 源码参考：** [Loss.cpp](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/Loss.cpp)

### 现有实现分析

昇腾算子开源仓 `ops-nn` 中已存在 `huber_loss` 算子的基础实现（路径：`experimental/loss/huber_loss`），但该实现存在以下局限：

| 能力 | 现有实现 | 任务要求 |
| --- | --- | --- |
| reduction=none | 支持 | 支持 |
| reduction=mean | 不支持 | 支持 |
| reduction=sum | 不支持 | 支持 |
| 数据类型 | FLOAT32 / FLOAT16 / BFLOAT16 | FLOAT32 / FLOAT16 / BFLOAT16 |
| 数据格式 | ND | ND |
| 非连续 Tensor | 支持（AutoContiguous） | 支持 |

本任务需在现有 none 模式基础上扩展支持 mean / sum 两种归约模式，实现与 PyTorch `aten::huber_loss` 完全对齐的前向功能。

# 需求分析（required）

## 需求描述

使用 Ascend C 编程语言实现 `huber_loss` 算子，支持 `reduction` 归约模式（none / mean / sum），支持 float32、float16、bfloat16 数据类型，数据格式 ND，适配 Atlas A2 训练系列产品。

## 需求拆解

1. **reduction=none 模式**：逐元素输出 Huber 损失，输出 shape 与 input 一致。
2. **reduction=mean 模式**：对所有元素 Huber 损失取均值，输出为标量。
3. **reduction=sum 模式**：对所有元素 Huber 损失求和，输出为标量。
4. **数据类型支持**：float32 原生计算；float16 / bfloat16 在 Kernel 内部 upcast 为 float32 计算后转回输出 dtype，避免精度损失。
5. **reduction=mean 精度保障**：累加过程在 FP32 进行，最终乘以 `1/N` 缩放系数后再转回输出 dtype。
6. **性能要求**：达到 80% 的 compute bound 或 memory bound。
7. **精度要求**：满足 AscendOpTest 工具默认阈值，与 CPU `aten::huber_loss` 结果对齐。
8. **泛化能力**：支持任意维度、任意合法 shape 的输入，验收阶段采用泛化数据验收。

# 详细设计（required）

## 算子分析

### 数学公式

**逐元素 Huber 损失：**

$$
e_i = input_i - target_i
$$

$$
loss_i =
\begin{cases}
0.5 \cdot e_i^2, & |e_i| \leq \delta \\
\delta \cdot (|e_i| - 0.5 \cdot \delta), & |e_i| > \delta
\end{cases}
$$

**归约输出：**

- `reduction = 0`（none）：$output = loss$（逐元素）
- `reduction = 2`（sum）：$output = \sum_{i} loss_i$
- `reduction = 1`（mean）：$output = \frac{1}{N} \sum_{i} loss_i$，其中 $N$ 为元素总数

### 支持数据类型

| 输入/输出 | 数据类型 | Kernel 内部处理 |
| --- | --- | --- |
| input / target / output | FLOAT32 | 原生 float 计算 |
| input / target / output | FLOAT16 | upcast 为 float32 计算，结果 cast 回 half |
| input / target / output | BFLOAT16 | upcast 为 float32 计算，结果 cast 回 bfloat16 |

### 支持形状

- input / target：任意维度，各维度 ≥ 0，ND 格式
- output（reduction=none）：与 input 相同 shape
- output（reduction=mean/sum）：标量（0 维）

### 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 | Shape | 非连续 Tensor |
| --- | --- | --- | --- | --- | --- | --- |
| input | 输入 | 预测值张量 | FLOAT32 / FLOAT16 / BFLOAT16 | ND | 任意维度，各维度 ≥ 0 | 支持 |
| target | 输入 | 目标值张量 | FLOAT32 / FLOAT16 / BFLOAT16 | ND | 与 input 相同 | 支持 |
| reduction | 可选属性 | 归约模式，默认 1 | int | - | - | - |
| delta | 可选属性 | Huber 阈值，默认 1.0，须 > 0 | float | - | - | - |

> **delta 精度处理**：delta 在 Host 侧 TilingFunc 中按输入 dtype RNE（Round to Nearest Even）量化后写入 TilingData，Kernel 侧直接使用，避免运行时转换开销。FP16 模式下 delta 被量化为 half(delta)，BF16 模式下 delta 被量化为 bfloat16(delta) 后转 FP32 传递。FP32 模式下 delta 原样传递。
| output | 输出 | loss 计算结果 | FLOAT32 / FLOAT16 / BFLOAT16 | ND | reduction=none 时与 input 同 shape；mean/sum 时为标量 | - |

> **reduction 取值映射：** 0 = none（逐元素输出），1 = mean（取均值），2 = sum（求和）

### 算子约束限制

1. input 与 target 必须具有相同的 shape 和 dtype，不支持 broadcast。
2. reduction 取值仅支持 0（none）、1（mean）、2（sum），其他值属非法输入。
3. delta 必须为正数（> 0）。
4. 输出 dtype 与 input / target 一致；reduction=mean 时内部累加提升至 FP32 计算后再转回输出 dtype，避免精度损失。
5. fusion：当前作为独立 loss 算子实现，不涉及图融合。

## 算子实现

### 实现方案

#### 整体架构

算子采用 Host + Kernel 分离的 Ascend C 标准开发模式，文件结构如下：

```
huber_loss/
├── op_host/
│   ├── huber_loss_def.cpp          # 算子定义（输入/输出/属性/硬件配置）
│   ├── huber_loss_infershape.cpp   # InferShape 实现
│   └── huber_loss_tiling.cpp       # Tiling 策略实现
├── op_kernel/
│   ├── huber_loss.cpp              # Kernel 入口函数
│   ├── huber_loss.h                # Kernel 类定义
│   ├── huber_loss_tiling_data.h    # TilingData 结构定义
│   └── huber_loss_tiling_key.h     # TilingKey 模板参数定义
├── tests/                           # UT 测试
├── examples/                        # ACLNN 调用样例
├── docs/                            # 接口文档
├── CMakeLists.txt
└── README.md
```

#### Host 侧设计

##### 1. 算子定义（huber_loss_def.cpp）

```cpp
class HuberLoss : public OpDef {
    // 输入
    this->Input("input")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        .AutoContiguous();
    this->Input("target")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        .AutoContiguous();

    // 属性
    this->Attr("reduction").AttrType(OPTIONAL).Int(1);   // 默认 mean
    this->Attr("delta").AttrType(OPTIONAL).Float(1.0);     // 默认 1.0

    // 输出
    this->Output("output")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        .AutoContiguous();

    // 硬件配置
    this->AICore().AddConfig("ascend910b");
};
```

##### 2. InferShape 设计（huber_loss_infershape.cpp）

InferShape 逻辑根据 `reduction` 属性决定输出 shape：

```cpp
static ge::graphStatus InferShapeHuberLoss(gert::InferShapeContext* context) {
    // 1. 校验 input/target shape 一致
    // 2. 读取 reduction 属性
    // 3. 根据 reduction 设置 output shape:
    //    - reduction == 0 (none): output shape = input shape
    //    - reduction == 1 (mean) 或 2 (sum): output shape = 标量 {}
    // 4. 校验 output dtype 与 input dtype 一致
}
```

**InferShape 输出 shape 规则：**

| reduction | output shape |
| --- | --- |
| 0 (none) | 与 input 相同 |
| 1 (mean) | `{}`（0 维标量） |
| 2 (sum) | `{}`（0 维标量） |

##### 3. Tiling 策略（huber_loss_tiling.cpp）

Tiling 策略根据 `reduction` 模式分为两条路径：

**路径 A — reduction=none（逐元素输出）：**

沿用现有 huber_loss 的 tiling 策略：

- **分核策略**：优先满核。若核间能均分则无大小核区分；若不能均分则将余出数据块分配到前几个核上（大核多处理一个 tile）。
- **数据分块**：根据 UB 内存大小和各 buffer 的空间需求计算单次搬运的 tile 大小（`tileDataNum`），按 64 元素对齐。
- **UB 内存分配**：
  - 2 个输入 Queue（input/target）× 2 buffer × elemBytes
  - 1 个输出 Queue × 2 buffer × elemBytes
  - 4 个 float 计算 buffer（diff, abs, quadratic, linear）× 4 bytes
  - 1 个 mask buffer（uint8_t）
  - 若 FP16/BF16：额外 2 个 float upcast buffer（predictionsFloat, targetsFloat）
- **TilingKey**：0

**路径 B — reduction=mean/sum（归约输出）：**

参考 mse_loss 的 tiling 策略：

- **分核策略**：
  - 若总元素数较少（单核可处理），使用单核，无需 workspace。
  - 若总元素数较多，多核并行处理，每核独立累加部分和，再通过 workspace 进行跨核归约。
- **数据分块**：同 none 模式的 UB tile 大小计算逻辑。
- **Workspace**：
  - 当多核并行时，每核在 workspace 中分配一个 slot（`workspaceFloatsPerCore` 个 float，通常为 1，即每核一个部分和）。
  - 核 0 负责收集所有核的部分和，做最终 `ReduceSum`。
  - mean 模式：最终结果乘以 `1/N`（`meanScale`）。
- **TilingKey**：
  - 按数据类型区分：FP16 → key base + 0，FP32 → key base + 1，BF16 → key base + 2
  - 按 reduction 模式区分：none → 单独 key，mean/sum → 共享归约 kernel（mean 额外乘 scale）

**TilingKey 规划：**

TilingKey 同时编码数据类型和 reduction 路径，Kernel 侧通过模板参数 `schMode` 选择数据类型分支，通过 `reduction` 字段选择 none / reduce 路径。

| TilingKey | 数据类型 | reduction 路径 | schMode（模板参数） | 说明 |
| --- | --- | --- | --- | --- |
| 0 | FLOAT16 | none | 0 | FP16 逐元素输出 |
| 1 | FLOAT32 | none | 1 | FP32 逐元素输出 |
| 2 | BFLOAT16 | none | 2 | BF16 逐元素输出 |
| 3 | FLOAT16 | mean/sum | 0 | FP16 归约输出 |
| 4 | FLOAT32 | mean/sum | 1 | FP32 归约输出 |
| 5 | BFLOAT16 | mean/sum | 2 | BF16 归约输出 |

> **映射规则：** `TilingKey // 3` 决定 reduction 路径（0=none, 1=mean/sum），`TilingKey % 3` 决定 `schMode`（0=FP16, 1=FP32, 2=BF16）。Host 侧通过 `context->SetTilingKey()` 设置，Kernel 侧通过 `ASCENDC_TPL_ARGS` 自动展开模板参数。
>
> mean 与 sum 共用归约 Kernel，通过 TilingData 中的 `reduction` 字段区分（reduction==1 时额外乘 `meanScale`，reduction==2 时直接输出 sum）。

**TilingData 结构：**

```cpp
struct HuberLossTilingData {
    // —— 公共字段 ——
    float delta;                        // Huber 阈值
    int64_t reduction;                  // 0=none, 1=mean, 2=sum

    // —— reduction=none 模式字段（uint32_t） ——
    uint32_t smallCoreDataNum;          // 小核处理元素数
    uint32_t bigCoreDataNum;            // 大核处理元素数
    uint32_t finalBigTileNum;           // 大核 tile 循环次数
    uint32_t finalSmallTileNum;         // 小核 tile 循环次数
    uint32_t tileDataNum;               // 单 tile 元素数
    uint32_t smallTailDataNum;          // 小核尾块元素数
    uint32_t bigTailDataNum;            // 大核尾块元素数
    uint32_t tailBlockNum;              // 大核数量

    // —— reduction=mean/sum 模式字段（int64_t） ——
    int64_t totalNum;                   // 总元素数
    int64_t blockFactor;                // 每核处理元素数
    int64_t ubFactor;                   // 单 tile 元素数
    int64_t blockNum;                   // 实际使用核数
    int64_t workspaceFloatsPerCore;     // 每核 workspace float 槽位数
    float meanScale;                    // mean 模式缩放系数 = 1/N
};
```

> **字段类型**：none 模式字段为 uint32_t（元素数和 tile 数不会超过 2³²），reduce 模式字段为 int64_t（与 mse_loss 一致）。公共字段 delta 为 float，reduction 为 int64_t。Host/Kernel 两侧结构体定义完全一致，通过 `GET_TILING_DATA_WITH_STRUCT` 宏保证内存布局对齐。
>
> **double buffer 策略**：input/target/output Queue 均开启 double buffer（BUFFER_NUM=2），在 Compute 计算当前 tile 的同时 CopyIn 可预取下一个 tile，隐藏访存延迟。计算 buffer（diff/abs/quadratic/linear/loss）为单 buffer，因为计算阶段无 IO 重叠需求。

#### Kernel 侧设计

##### 1. Kernel 入口（huber_loss.cpp）

```cpp
__global__ __aicore__ void huber_loss(GM_ADDR input, GM_ADDR target, GM_ADDR output,
                                       GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(HuberLossTilingData);
    GET_TILING_DATA_WITH_STRUCT(HuberLossTilingData, tilingData, tiling);

    // 根据 TilingKey 选择数据类型模板
    // TilingKey % 3: 0=FP16, 1=FP32, 2=BF16
    uint32_t schMode = GetTilingKey() % 3;
    if (schMode == 0) { NsHuberLoss::KernelHuberLoss<half> op; ... }       // FP16
    if (schMode == 1) { NsHuberLoss::KernelHuberLoss<float> op; ... }     // FP32
    if (schMode == 2) { NsHuberLoss::KernelHuberLoss<bfloat16_t> op; ... } // BF16
}
```

##### 2. Kernel 类设计（huber_loss.h）

Kernel 类 `KernelHuberLoss<T>` 包含两条执行路径：

```cpp
template <typename T>
class KernelHuberLoss {
    using IoT = T;
    static constexpr bool kUpcast = !std::is_same<IoT, float>::value;

public:
    __aicore__ inline void Init(...) {
        // 1. 根据 tilingData 确定 reduction 模式
        // 2. 计算当前核的数据范围 (offset, count)
        // 3. 设置 GlobalBuffer
        // 4. 初始化 UB buffers (Queue + TBuf)
        // 5. 若 mean/sum 多核：初始化 workspace GlobalTensor
    }

    __aicore__ inline void Process() {
        if (totalNum_ == 0) {
            ProcessEmpty();     // 空 Tensor 快速路径
        } else if (reduction_ == 0) {
            ProcessNone();      // 逐元素路径
        } else {
            ProcessReduce();    // 归约路径
        }
    }

    // —— 空 Tensor 处理 ——
    __aicore__ inline void ProcessEmpty() {
        // none: output 已在 InferShape 设为空 shape，无需写回
        // mean: output 写 NaN（0/0）
        // sum:  output 写 0
        if (reduction_ != 0) {
            LocalTensor<IoT> outLocal = outputQueue_.AllocTensor<IoT>();
            float fillVal = (reduction_ == 1) ? NAN : 0.0f;
            // FP32 → IoT cast + 写标量
            Cast(outLocal, ..., 1);
            outputQueue_.EnQue(outLocal);
            auto out = outputQueue_.DeQue<IoT>();
            DataCopyExtParams params{1, sizeof(IoT), 0, 0, 0};
            DataCopyPad(outputGM_, out, params);
            outputQueue_.FreeTensor(out);
        }
    }

private:
    // —— 逐元素路径（reduction=none）——
    __aicore__ inline void ProcessNone() {
        for (uint32_t i = 0; i < tiles_; ++i) {
            CopyIn(i, n);
            ComputeNone(n);     // 计算 Huber loss 逐元素
            CopyOut(i, n);      // 写回 GM
        }
    }

    // —— 归约路径（reduction=mean/sum）——
    __aicore__ inline void ProcessReduce() {
        // 1. 初始化 partialLocal = 0
        // 2. 循环 tile：
        //    CopyIn → ComputeHuberElementWise → ReduceSum → Add(partialLocal)
        // 3. 跨核归约：
        //    - 单核：直接写结果（mean 乘 meanScale）
        //    - 多核：各核写部分和到 workspace → SyncAll → 核0收集并 ReduceSum
        LocalTensor<float> partialLocal = partialBuf_.Get<float>();
        Duplicate(partialLocal, 0.0f, workspaceFloatsPerCore_);
        for (uint32_t i = 0; i < tiles_; ++i) {
            CopyIn(i, n);
            AccumulateHuber(n, partialLocal);  // 逐元素 Huber + 累加
        }
        CopyOutReduce(partialLocal);
    }
};
```

##### 3. 核心计算逻辑

**逐元素 Huber 损失计算（ComputeNone / AccumulateHuber 共用）：**

```cpp
__aicore__ inline void ComputeHuberElementWise(uint32_t n, LocalTensor<float>& lossBuf) {
    auto p = inputQueue_.DeQue<IoT>();
    auto t = targetQueue_.DeQue<IoT>();

    auto diff = diffBuf_.Get<float>();
    auto absDiff = absBuf_.Get<float>();
    auto quadratic = quadraticBuf_.Get<float>();
    auto linear = linearBuf_.Get<float>();
    auto mask = maskBuf_.Get<uint8_t>();

    // 1. upcast（FP16/BF16 → FP32）
    if constexpr (kUpcast) {
        auto pf = inputFloatBuf_.Get<float>();
        auto tf = targetFloatBuf_.Get<float>();
        Cast(pf, p, RoundMode::CAST_NONE, n);
        Cast(tf, t, RoundMode::CAST_NONE, n);
        PipeBarrier<PIPE_V>();
        Sub(diff, pf, tf, n);
    } else {
        Sub(diff, p, t, n);
    }

    // 2. e = input - target
    PipeBarrier<PIPE_V>();

    // 3. |e|
    Abs(absDiff, diff, n);
    PipeBarrier<PIPE_V>();

    // 4. quadratic 分支: 0.5 * e²
    Mul(quadratic, diff, diff, n);
    PipeBarrier<PIPE_V>();
    Muls(quadratic, quadratic, 0.5f, n);
    PipeBarrier<PIPE_V>();

    // 5. linear 分支: delta * (|e| - 0.5 * delta)
    Adds(linear, absDiff, -0.5f * delta_, n);
    PipeBarrier<PIPE_V>();
    Muls(linear, linear, delta_, n);
    PipeBarrier<PIPE_V>();

    // 6. 分段选择: abs(e) ≤ delta ? quadratic : linear
    // Ascend C Compare 指令以 64 bits 为 mask 宽度（对应 8 个 float），
    // 一次 Compare 最多处理 256 个 float（32 × 8）。
    // 此处采用 64 元素为一次比较单元，repeat 次 covering 整个 tile。
    constexpr uint64_t kMask = 64;  // 64 个 float 对应 64 bytes mask
    const uint64_t width = n < kMask ? n : kMask;
    const int32_t repeat = (n + width - 1) / width;
    UnaryRepeatParams cmpParams{1, 1, 8, 8};
    CompareScalar(mask, absDiff, delta_, CMPMODE::LE, width, repeat, cmpParams);
    PipeBarrier<PIPE_V>();
    BinaryRepeatParams selParams{1, 1, 1, 8, 8, 8};
    Select(lossBuf, mask, quadratic, linear, SELMODE::VSEL_TENSOR_TENSOR_MODE, width, repeat, selParams);
    PipeBarrier<PIPE_V>();

    inputQueue_.FreeTensor(p);
    targetQueue_.FreeTensor(t);
}
```

**none 模式 CopyOut：**

```cpp
// lossBuf (float) → cast 回 IoT → 写回 GM
if constexpr (kUpcast) {
    Cast(out, lossBuf, RoundMode::CAST_RINT, n);
} else {
    // FP32 直接写回
    Adds(out, lossBuf, 0.0f, n);  // 利用 Adds 做类型转换
}
```

**归约模式 Accumulate：**

```cpp
__aicore__ inline void AccumulateHuber(uint32_t n, LocalTensor<float>& partialLocal) {
    ComputeHuberElementWise(n, lossBuf);  // 逐元素 Huber loss（FP32）
    PipeBarrier<PIPE_V>();
    ReduceSum<float>(lossBuf, lossBuf, reduceWorkBuf_, n);  // tile 内 reduce sum，workspace 用独立 buffer
    PipeBarrier<PIPE_V>();
    Add(partialLocal, partialLocal, lossBuf, 1);  // 累加到 partial
}
```

**归约模式 CopyOutReduce（跨核归约）：**

```cpp
__aicore__ inline void CopyOutReduce(LocalTensor<float>& partialLocal) {
    if (blockNum_ == 1) {
        // 单核：直接写结果
        if (reduction_ == 1) {  // mean
            Muls(partialLocal, partialLocal, meanScale_, 1);
            PipeBarrier<PIPE_V>();
        }
        WriteReduceResult(partialLocal);
        return;
    }

    // 多核：各核写部分和到 workspace
    PipeBarrier<PIPE_ALL>();
    DataCopy(workspaceGM_[GetBlockIdx() * workspaceFloatsPerCore_], partialLocal, workspaceFloatsPerCore_);
    PipeBarrier<PIPE_ALL>();
    SyncAll();

    if (GetBlockIdx() != 0) return;  // 只有核 0 做最终归约

    // 核 0：收集所有核部分和 → ReduceSum → (mean 乘 scale) → 写输出
    LocalTensor<float> mergeLocal = downloadQueue_.AllocTensor<float>();
    DataCopy(mergeLocal, workspaceGM_, blockNum_ * workspaceFloatsPerCore_);
    ReduceSum<float>(mergeLocal, mergeLocal, reduceWorkBuf_, blockNum_ * workspaceFloatsPerCore_);
    PipeBarrier<PIPE_V>();
    if (reduction_ == 1) {  // mean
        Muls(mergeLocal, mergeLocal, meanScale_, 1);
        PipeBarrier<PIPE_V>();
    }
    WriteReduceResult(mergeLocal);
}
```

**WriteReduceResult（FP32 → IoT 类型转换 + 写标量到 GM）：**

```cpp
__aicore__ inline void WriteReduceResult(LocalTensor<float>& resultLocal) {
    LocalTensor<IoT> outLocal = outputQueue_.AllocTensor<IoT>();
    if constexpr (kUpcast) {
        Cast(outLocal, resultLocal, RoundMode::CAST_RINT, 1);
    } else {
        Adds(outLocal, resultLocal, 0.0f, 1);
    }
    outputQueue_.EnQue(outLocal);
    auto out = outputQueue_.DeQue<IoT>();
    DataCopyExtParams params{1, sizeof(IoT), 0, 0, 0};
    DataCopyPad(outputGM_, out, params);
    outputQueue_.FreeTensor(out);
}
```

##### 4. UB 内存布局

**reduction=none 模式 UB 分配：**

| Buffer | 用途 | 数量 × 大小 |
| --- | --- | --- |
| inputQueue_ | 输入 input | 2 × tile × elemBytes |
| targetQueue_ | 输入 target | 2 × tile × elemBytes |
| outputQueue_ | 输出 loss | 2 × tile × elemBytes |
| diffBuf_ | e = input - target | 1 × tile × 4 (float) |
| absBuf_ | \|e\| | 1 × tile × 4 |
| quadraticBuf_ | 0.5 * e² | 1 × tile × 4 |
| linearBuf_ | delta * (\|e\| - 0.5 * delta) | 1 × tile × 4 |
| maskBuf_ | Compare 结果 mask | 1 × tile × 1 (uint8) |
| inputFloatBuf_ (kUpcast) | upcast 后的 input | 1 × tile × 4 |
| targetFloatBuf_ (kUpcast) | upcast 后的 target | 1 × tile × 4 |

> **每 tile UB 开销**（none 模式）= `2 × tile × elemBytes × 3（input+target+output Queue）` + `tile × 4 × (4 + kUpcast × 2)（diff/abs/quadratic/linear + upcast buf）` + `tile × 1（mask）`
>
> 其中 `elemBytes` 为输入 dtype 元素大小（FP32=4, FP16/BF16=2），`kUpcast` 在 FP16/BF16 时为 1，FP32 时为 0。`tileDataNum` 的最大值受 UB 总大小（ascend910b 通常为 192KB）约束。

**reduction=mean/sum 模式 UB 分配：**

| Buffer | 用途 | 数量 × 大小 |
| --- | --- | --- |
| inputQueue_ | 输入 input | 2 × tile × elemBytes |
| targetQueue_ | 输入 target | 2 × tile × elemBytes |
| outputQueue_ | 输出（标量） | 2 × elemBytes |
| downloadQueue_ | 多核归约时下载 workspace | 1 × blockNum × workspaceFloatsPerCore × 4 |
| diffBuf_ | e = input - target | 1 × tile × 4 |
| absBuf_ | \|e\| | 1 × tile × 4 |
| quadraticBuf_ | 0.5 * e² | 1 × tile × 4 |
| linearBuf_ | delta * (\|e\| - 0.5 * delta) | 1 × tile × 4 |
| lossBuf_ | Select 结果（逐元素 loss） | 1 × tile × 4 |
| maskBuf_ | Compare 结果 mask | 1 × tile × 1 |
| partialBuf_ | 部分和累加器 | 1 × workspaceFloatsPerCore × 4 |
| inputFloatBuf_ (kUpcast) | upcast 后的 input | 1 × tile × 4 |
| targetFloatBuf_ (kUpcast) | upcast 后的 target | 1 × tile × 4 |

##### 5. 算子执行流程图

###### 图 1：算子总体调用链

~~~mermaid
flowchart LR
    A["aclnnHuberLossGetWorkspaceSize"] --> B["参数、dtype、shape、reduction、delta 校验"]
    B --> C{"空 Tensor?"}
    C -- "是" --> D["none 直接返回；mean Fill NaN / sum Fill 0"]
    C -- "否" --> E["input/target Contiguous"]
    E --> F["L0 HuberLoss 与 InferShape"]
    F --> G["Host Tiling：none / mean / sum 三组 TilingKey"]
    G --> H["Kernel：有限 delta clipped 公式；delta=+Inf 特殊位模式路径"]
    H --> I{"reduction=none?"}
    I -- "是" --> J["按原 dtype 逐元素 CopyOut"]
    I -- "否" --> K["FP32 单核/多核归约"]
    K --> L["mean 缩放或 sum，写 0 维标量"]
    J --> M["ViewCopy 到调用方 output"]
    L --> M
~~~

> **图 1 说明**：ACLNN 两段式接口入口。第一段完成参数校验、空 Tensor 快速分流、Contiguous 转换和执行图构建；第二段启动 AICore Kernel。Kernel 完成后若 output 是非连续 view 或 dtype 不匹配，通过 ViewCopy 写回用户 output。input/target 非连续时在进入 Kernel 前转换为连续布局。

###### 图 2：Host Tiling 决策流程

~~~mermaid
flowchart TD
    T0["TilingFunc 入口"] --> T1["读取 input/target shape、dtype"]
    T1 --> T2["读取 reduction（默认 1）、delta（默认 1.0）"]
    T2 --> T3{"delta > 0?"}
    T3 -- "否" --> TERR["OP_LOGE 报错，return GRAPH_FAILED"]
    T3 -- "是" --> T4["delta 按输入 dtype RNE 量化"]
    T4 --> T5["计算总元素数 N"]
    T5 --> T6{"reduction 模式?"}

    T6 -- "none（0）" --> TN1["分核策略：bigCore / smallCore"]
    TN1 --> TN2["tileDataNum = UB 可用 / 单元素开销，对齐 32B"]
    TN2 --> TN3["计算 bigCoreDataNum、smallCoreDataNum、tileNum、tail"]
    TN3 --> TN4["TilingKey = dtype_offset + 0"]
    TN4 --> TN5["写入 TilingData，workspace = 0"]

    T6 -- "mean / sum（1/2）" --> TR1{"N ≤ 单核阈值?"}
    TR1 -- "是（单核 fast path）" --> TR2["blockNum = 1，workspace = 0"]
    TR1 -- "否（多核 reduce）" --> TR3["blockNum = min(N/ubFactor, maxCores)，分配 workspace"]
    TR2 --> TR4["ubFactor = UB 可用 / 单元素开销，对齐 32B"]
    TR3 --> TR4
    TR4 --> TR5["meanScale = 1.0 / N"]
    TR5 --> TR6["TilingKey = dtype_offset + 3"]
    TR6 --> TR6a["blockNum > 1 时 SetScheduleMode(1) 同步模式"]
    TR6a --> TR7["写入 TilingData，含 reduction、meanScale、delta"]

    TN5 --> DONE_T["Tiling 完成"]
    TR7 --> DONE_T
~~~

> **图 2 说明**：Host 侧根据 reduction 模式走两条 Tiling 路径。none 模式沿用 bigCore/smallCore 分核策略；mean / sum 模式参考 mse_loss，按元素总数决定单核 fast path 或多核 reduce。TilingKey 编码规则：TilingKey // 3 决定 reduction 路径（0=none, 1=mean/sum），TilingKey % 3 决定 schMode（0=FP16, 1=FP32, 2=BF16）。delta 在 Host 侧提前按输入 dtype RNE 量化，避免 Kernel 内运行时转换开销。

###### 图 3：Kernel 执行流程

~~~mermaid
flowchart TD
    A["读取 TilingData 与模式"] --> B{"reduction=none?"}
    B -- "是" --> C["多核分片"]
    C --> D["循环 tile：CopyIn（double buffer 预取）"]
    D --> E{"delta=+Inf?"}
    E -- "否" --> F["有限 delta：clipped 两 buffer 公式；低精度逐步舍入"]
    E -- "是" --> F1["+Inf 位模式路径：区分有限溢出、真 Inf 与 NaN"]
    F1 --> G
    F --> G["Cast 并 CopyOut"]
    G --> Z["结束"]

    B -- "否" --> H{"N 是否为空?"}
    H -- "是" --> I["mean 写 NaN / sum 写 0"]
    I --> Z
    H -- "否" --> J{"是否单核 fast path?"}
    J -- "是" --> K["预取下一 tile；按有限/+Inf 语义计算并量化 loss"]
    K --> L["成对 Add 折叠至 8 lanes；特殊路径保留标记"]
    L --> L1["最终一次 ReduceSum"]
    L1 --> M{"reduction=mean?"}
    M -- "是" --> M1["乘 meanScale（1/N）"]
    M -- "否" --> M2["保持 sum"]
    M1 --> N["Cast 并写标量"]
    M2 --> N
    N --> Z

    J -- "否" --> O["每核循环 tile 并累加 8-lane partial"]
    O --> P["写入每核独立 32B partial 到 workspace"]
    P --> Q["PipeBarrier 后 SyncAll"]
    Q --> R{"blockIdx=0?"}
    R -- "否" --> Z
    R -- "是" --> S["汇总所有 partial"]
    S --> T{"reduction=mean?"}
    T -- "是" --> T1["乘 meanScale（1/N）"]
    T -- "否" --> T2["保持 sum"]
    T1 --> U["Cast 并写标量"]
    T2 --> U
    U --> Z
~~~

> **图 3 说明**：Kernel 入口先检查空 Tensor（N=0），空 Tensor 时 mean 写 NaN、sum 写 0、none 无需写回。非空时通过 TilingKey 分流到 none 或 mean/sum 路径。
>
> **none 路径**：每个核独立处理数据分片，逐 tile 执行 CopyIn→Compute→Cast→CopyOut，无跨核通信。double buffer 预取下一 tile 隐藏访存延迟。
>
> **mean/sum 路径**：每个核先本地累加所有 tile 的 Huber loss 到 partialLocal（FP32），然后分两种情况：
> - **单核 fast path**（总元素数 ≤ 单核阈值）：直接在当前核完成 mean 缩放或 sum 输出，写标量到 GM。
> - **多核路径**：各核将 partial 写入 workspace GM，SyncAll 同步后核 0 汇总所有 partial 再做最终归约。
>
> **delta=+Inf 特殊路径**：当 delta 为 +Inf 时，走位模式路径，区分有限溢出、真 Inf 与 NaN，在单核/多核规约中保留标记。

#### ACLNN 接口设计

算子提供 ACLNN 两段式接口，遵循昇腾算子 ACLNN 标准规范。

> **ViewCopy 机制**：当用户传入的 output tensor 是非连续 view 或 dtype 与内部计算 dtype 不匹配时，ACLNN 内部分配连续的 output tensor，Kernel 完成后通过 ViewCopy 将内部结果写回用户 output。对于非连续 input/target，在进入 Kernel 前通过 Contiguous 转换为连续布局。

**接口定义：**

```cpp
// 第一段：计算 workspace 大小
aclnnStatus aclnnHuberLossGetWorkspaceSize(
    const aclTensor *input,
    const aclTensor *target,
    int64_t reduction,
    double delta,
    const aclTensor *output,
    uint64_t *workspaceSize,
    aclOpExecutor *executor);

// 第二段：执行算子
aclnnStatus aclnnHuberLoss(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);
```

**op_api 目录结构：**

```
op_host/
├── op_api/
│   └── aclnn_huber_loss.cpp     # ACLNN 接口实现
├── huber_loss_def.cpp
├── huber_loss_infershape.cpp
└── huber_loss_tiling.cpp
```

**异常处理路径：**

| 校验项 | 校验位置 | 失败行为 |
| --- | --- | --- |
| input/target 为空指针 | `aclnnHuberLossGetWorkspaceSize` | 返回 `ACLNN_ERR_INVALID_PARAM` |
| input/target shape 不一致 | `InferShape` | 返回 `ge::GRAPH_FAILED` |
| input/target dtype 不一致 | `InferShape` | 返回 `ge::GRAPH_FAILED` |
| reduction 不在 {0,1,2} | `TilingFunc` | OP_LOGE 报错，返回 `ge::GRAPH_FAILED` |
| delta ≤ 0 | `TilingFunc` | OP_LOGE 报错，返回 `ge::GRAPH_FAILED` |
| output dtype 与 input 不一致 | `InferShape` | 返回 `ge::GRAPH_FAILED` |

#### 编译与运行

**环境要求：**

- CANN 9.0.0 及以上
- Atlas A2 训练系列产品（ascend910b）

**编译命令：**

```bash
source /usr/local/Ascend/cann-9.0.0/set_env.sh

# 编译 Host 侧
bash build.sh -u --ophost --ops=huber_loss --soc=ascend910b --experimental

# 编译 Kernel 侧
bash build.sh -u --opkernel --ops=huber_loss --soc=ascend910b --experimental
```

**运行 UT 测试：**

```bash
bash build.sh -u --ops=huber_loss --soc=ascend910b --experimental
```

**Eager 调用验证：**

```bash
# 构建 custom package 后
bash build.sh --run_example huber_loss eager cust \
    --example_name=huber_loss --vendor_name=custom --experimental
```

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas 800I/T A2 | √ |

## 算子约束限制

1. input 与 target 必须具有相同的 shape 和 dtype，不支持 broadcast。
2. reduction 取值仅支持 0（none）、1（mean）、2（sum），其他值属非法输入。
3. delta 必须为正数（> 0）。
4. 输出 dtype 与 input / target 一致。
5. reduction=mean 时内部累加提升至 FP32 计算后再转回输出 dtype，避免精度损失。
6. 当前作为独立 loss 算子实现，不涉及图融合。
7. 支持动态 rank 和动态 shape。
8. 支持非连续 Tensor（通过 AutoContiguous 自动连续化）。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | 满足 AscendOpTest 工具默认阈值，与 CPU `aten::huber_loss` 结果对齐 | 任务书 |
| 性能标准 | 达到 80% 的 compute bound 或 memory bound | 任务书 |

### 精度测试方案

**测试工具：** [AscendOpTest](https://gitcode.com/HIT1920/AscendOpTest)

**精度对比基准：** PyTorch CPU `torch.nn.functional.huber_loss`

**精度阈值（AscendOpTest 默认）：**

| 数据类型 | 相对误差阈值 | 绝对误差阈值 |
| --- | --- | --- |
| FLOAT32 | 1e-5 | 1e-8 |
| FLOAT16 | 1e-3 | 1e-5 |
| BFLOAT16 | 1e-2 | 1e-3 |

**精度测试用例设计：**

| 用例编号 | 数据类型 | input shape | reduction | delta | 特殊场景 |
| --- | --- | --- | --- | --- | --- |
| UT-001 | FLOAT32 | (1024,) | none | 1.0 | 基础功能 |
| UT-002 | FLOAT32 | (1024,) | mean | 1.0 | 基础功能 |
| UT-003 | FLOAT32 | (1024,) | sum | 1.0 | 基础功能 |
| UT-004 | FLOAT16 | (2048, 512) | none | 1.0 | FP16 大张量 |
| UT-005 | FLOAT16 | (2048, 512) | mean | 1.0 | FP16 归约 |
| UT-006 | FLOAT16 | (2048, 512) | sum | 1.0 | FP16 归约 |
| UT-007 | BFLOAT16 | (4096,) | none | 1.0 | BF16 基础 |
| UT-008 | BFLOAT16 | (4096,) | mean | 1.0 | BF16 归约 |
| UT-009 | BFLOAT16 | (4096,) | sum | 1.0 | BF16 归约 |
| UT-010 | FLOAT32 | (3, 224, 224) | mean | 0.5 | 自定义 delta |
| UT-011 | FLOAT32 | (3, 224, 224) | sum | 2.0 | 自定义 delta |
| UT-012 | FLOAT32 | (1,) | mean | 1.0 | 单元素 |
| UT-013 | FLOAT32 | (0,) | mean | 1.0 | 空张量 |
| UT-014 | FLOAT32 | (4096, 4096) | mean | 1.0 | 大张量多核 |
| UT-015 | FLOAT16 | (0, 100) | none | 1.0 | 空张量带维度 |
| UT-016 | FLOAT32 | (7, 13, 17, 19) | none | 1.0 | 奇数维度泛化 |
| UT-017 | FLOAT32 | (7, 13, 17, 19) | mean | 3.5 | 奇数维度 + 自定义 delta |
| UT-018 | BFLOAT16 | (8192, 8192) | sum | 1.0 | BF16 大张量归约 |
| UT-019 | FLOAT32 | (1000000,) | mean | 1.0 | 超大张量泛化 |
| UT-020 | FLOAT32 | (256, 256, 32) | none | 0.1 | 多维 + 小 delta |

**边界值测试：**

| 用例编号 | 场景 | 预期行为 |
| --- | --- | --- |
| BV-001 | \|e\| == delta（恰好等于阈值） | 走 quadratic 分支（`|e| ≤ delta`） |
| BV-002 | \|e\| = delta + ε（略大于阈值） | 走 linear 分支 |
| BV-003 | e = 0（input == target） | loss = 0 |
| BV-004 | delta = 极小正数（1e-10） | 正常计算 |
| BV-005 | delta = 大数（1e6） | 正常计算 |
| BV-006 | reduction 非法值（3） | 报错返回 |
| BV-007 | delta = 0 | 报错返回 |
| BV-008 | delta = -1 | 报错返回 |
| BV-009 | input/target shape 不一致 | 报错返回 |
| BV-010 | input/target dtype 不一致 | 报错返回 |

### 性能测试方案

**测试硬件：** Atlas A2 训练系列产品

**性能测试用例：**

| 用例编号 | 数据类型 | shape | reduction | delta | 测试目标 |
| --- | --- | --- | --- | --- | --- |
| PERF-001 | FLOAT32 | (1024, 1024) | none | 1.0 | memory bound（逐元素） |
| PERF-002 | FLOAT32 | (1024, 1024) | mean | 1.0 | compute bound（归约） |
| PERF-003 | FLOAT32 | (1024, 1024) | sum | 1.0 | compute bound（归约） |
| PERF-004 | FLOAT16 | (4096, 4096) | none | 1.0 | memory bound（FP16 逐元素） |
| PERF-005 | FLOAT16 | (4096, 4096) | mean | 1.0 | compute bound（FP16 归约） |
| PERF-006 | BFLOAT16 | (4096, 4096) | mean | 1.0 | compute bound（BF16 归约） |
| PERF-007 | FLOAT32 | (8192, 8192) | mean | 1.0 | 大张量归约 |
| PERF-008 | FLOAT32 | (256,) | mean | 1.0 | 小张量（单核） |

**性能指标：**

- 逐元素模式（none）：理论访存量为 `(2 × input + 1 × output) × elemBytes`，目标达到 80% memory bound。
- 归约模式（mean/sum）：理论访存量为 `2 × input × elemBytes`（读 input + target，写 1 个标量可忽略），目标达到 80% memory bound。

## 兼容性分析

现有 `ops-nn` 仓中 `huber_loss` 算子**无 `reduction` 属性**，仅实现逐元素输出（等价于 `reduction=none`）。本次开发新增 `reduction` 属性（int 类型，默认 1=mean）和 `delta` 属性（float 类型，默认 1.0），并新增 mean / sum 归约路径。

**兼容性影响：**

- 现有算子定义中无 `reduction` 属性，现有调用方均以逐元素模式运行。本次新增 `reduction` 为 OPTIONAL 属性，若调用方未显式传 `reduction`，默认值为 1（mean），**行为与现有逐元素输出不同**。
- **缓解措施**：
  - ACLNN 接口层面提供 `aclnnHuberLossGetWorkspaceSize` 两段式接口，调用方可显式传 `reduction` 参数。若需保持原逐元素行为，需显式传 `reduction=0`。
  - 由于现有实现 README 明确标注"不支持 reduction"，且该算子尚在 `experimental` 目录，实际下游使用方极少，兼容性风险低。
  - PR 提交时在 PR 描述中显式说明 `reduction` 默认值变更，通知 reviewer。

# 附录

## 参考资料

| 资料 | 链接 |
| --- | --- |
| 任务书 | [huber_loss_task_doc.md](https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/docs/202608/huber_loss_task_doc.md) |
| 设计文档模板 | [design_template.md](https://gitcode.com/cann/cann-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md) |
| Ascend C 算子开发文档 | [链接](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/opdevg/Ascendcopdevg/atlas_ascendc_map_10_0002.html) |
| 算子开发接口文档 | [链接](https://www.hiascend.com/document/detail/zh/canncommercial/850/API/ascendcopapi/atlasascendc_api_07_0003.html) |
| Ascend C 在线课程 | [链接](https://www.hiascend.com/developer/courses/detail/1691696509765107713) |
| 代码样例（loss 目录） | [链接](https://gitcode.com/cann/ops-nn/tree/master/experimental/loss) |
| PyTorch huber_loss 官方文档 | [链接](https://pytorch.org/docs/stable/generated/torch.nn.functional.huber_loss.html) |
| PyTorch 源码实现 | [Loss.cpp](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/Loss.cpp) |
| AscendOpTest 精度阈值 | [链接](https://gitcode.com/HIT1920/AscendOpTest/blob/master/compare/compare/accuracy_config.py) |
| PR 合入目标目录 | [ops-nn/experimental/loss](https://gitcode.com/cann/ops-nn/tree/master/experimental/loss) |

## 术语表

| 术语 | 含义 |
| --- | --- |
| NPU | Neural Processing Unit，昇腾神经网络处理器 |
| CANN | Compute Architecture for Neural Networks，昇腾计算架构 |
| Ascend C | 昇腾算子开发编程语言 |
| UB | Unified Buffer，统一缓冲区，AI Core 片上存储 |
| GM | Global Memory，全局内存（HBM） |
| Tiling | 数据切分策略，将计算任务分配到多核和多次循环 |
| TilingKey | 切分键，Host 侧设置，Kernel 侧据此选择不同执行分支 |
| TilingData | 切分数据结构，Host 侧填充后传递到 Kernel 侧 |
| InferShape | 输出形状推导 |
| ACLNN | Ascend Computing Language Native NN API |
| FP32 / FP16 / BF16 | float32 / float16 / bfloat16 数据类型 |
| memory bound | 访存瓶颈，性能受限于数据搬运带宽 |
| compute bound | 计算瓶颈，性能受限于计算单元吞吐 |
| upcast | 低精度类型向高精度类型转换（如 FP16 → FP32） |
| double buffer | 双缓冲，Queue 分配 2 个 buffer 交替使用，隐藏访存延迟 |
| SyncAll | 跨核同步原语，等待所有核到达同步点 |
| tile | 单次循环处理的数据块 |
| PipeBarrier | 管道屏障，确保前一阶段指令完成后再执行后续指令 |
