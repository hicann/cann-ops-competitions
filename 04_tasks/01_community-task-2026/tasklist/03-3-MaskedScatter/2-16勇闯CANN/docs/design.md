# **需求背景**

## **需求来源**

基于 TBE MaskedScatter 算子历史版本，使用 Ascend C 进行改造与适配。改造目标是保持 TBE 语义和关键调度逻辑一致，并通过 aclnn 直调方式完成自定义 experimental 算子的编译、安装和 example 验证。

MaskedScatter 的语义为：按一维扁平顺序扫描 `x` 和 `mask`，当 `mask[i]` 为 false 时输出 `y[i] = x[i]`；当 `mask[i]` 为 true 时，按 true mask 出现顺序从 `updates` 中取值写入 `y[i]`。

等价伪代码如下：

```cpp
int64_t updateIndex = 0;
for (int64_t i = 0; i < numElemX; ++i) {
    if (mask[i] != 0) {
        y[i] = updates[updateIndex];
        updateIndex++;
    } else {
        y[i] = x[i];
    }
}
```

## **TBE 源码分析**

通过对 TBE 内置 MaskedScatter 算子源码（`masked_scatter.py` 中 `MaskedScatter` 类、`masked_scatter()` 顶层函数、`task_schedule()`、`calc_updates_start()`、`compute()`）进行逐行分析，当前支持的能力与核心逻辑如下。

TBE 算子源码路径：

```text
${ASCEND_INSTALL_PATH}/opp/built-in/op_impl/ai_core/tbe/impl/ops_legacy/dynamic/masked_scatter.py
```

算子原型路径：

```text
${ASCEND_INSTALL_PATH}/opp/built-in/op_proto/inc/
```

算子信息库路径：

```text
${ASCEND_INSTALL_PATH}/opp/built-in/op_impl/ai_core/tbe/kernel/config/ascend910b/ops_legacy/masked_scatter.json
```

### **1. 支持的数据类型**

TBE `masked_scatter.py` 的 dtype 字节表支持如下类型：

| 数据类型 | 字节大小 | 说明 |
| -------- | -------- | ---- |
| float16 | 2 | 正常处理 |
| float32 | 4 | 正常处理 |
| int64 | 8 | TBE 字节表包含 |
| int32 | 4 | 正常处理 |
| uint8 | 1 | 正常处理 |
| int8 | 1 | 正常处理 |
| bool | 1 | `mask` 使用 bool，TBE 内部按 int8 Tensor 读写 |
| int16 | 2 | 正常处理 |
| bfloat16 | 2 | TBE 初始化时映射为 float16 处理 |

当前 Ascend C experimental 算子注册支持：

| 数据类型 | 字节大小 | 说明 |
| -------- | -------- | ---- |
| float16 | 2 | 支持 |
| float32 | 4 | 支持 |
| uint8 | 1 | 支持 |
| int8 | 1 | 支持 |
| int16 | 2 | 支持 |
| int32 | 4 | 支持 |
| bool | 1 | 支持 |
| bfloat16 | 2 | 支持 |

差异说明：TBE 字节表包含 `int64`，当前 experimental 算子原型和 host dtype 校验未注册 `int64`，因此当前 Ascend C 版本不支持 `int64`。

### **2. 输入输出与 shape 约束**

TBE `check_supported()` 的显式约束如下：

| 检查项 | 规则 |
| ------ | ---- |
| 动态未知 shape | `x_shape` 包含 `-1` 或 `-2` 时返回 `"Unknown"` |
| `x` 与 `mask` shape | 必须完全相同，否则返回 `False` |

TBE 顶层接口为：

```python
def masked_scatter(x, mask, updates, y, kernel_name="masked_scatter")
```

输入输出含义如下：

| 名称 | 类别 | dtype | shape | 说明 |
| ---- | ---- | ----- | ----- | ---- |
| x | 输入 | dtype 表中的数据类型 | all | 原始输入 |
| mask | 输入 | bool | 与 x 相同 | 控制是否替换 |
| updates | 输入 | 与 x 相同 | all | 替换值来源 |
| y | 输出 | 与 x 相同 | 与 x 相同 | 输出结果 |

### **3. TBE 常量与 tiling 参数**

TBE 关键常量如下：

| 常量 | 值 | 说明 |
| ---- | -- | ---- |
| `BYTES_PER_BLOCK` | 32 | GM/UB 搬运按 32B 对齐 |
| `MAX_VEC_PROCESS_NUM` | 64 | 向量处理基础长度，也是 task 最小长度 |
| `TASK_ALIGN` | 4096 | 大输入 task 长度，也是 `calc_updates_start` 单次 reduce 的 mask 元素数 |
| `TILING_ARG_NUM` | 4 | tiling GM 中 int64 参数数量 |
| `BOOL_DTYPE` | int8 | bool mask 在 TIK Tensor 中按 int8 表示 |

TBE 从 `tiling_gm` 读取 4 个 int64 参数：

| tiling 下标 | 字段 | 说明 |
| ----------- | ---- | ---- |
| 0 | `num_elem_x` | `x` 总元素数 |
| 1 | `num_elem_mask` | `mask` 总元素数，TBE 中与 `num_elem_x` 一致 |
| 2 | `num_elem_updates` | `updates` 总元素数 |
| 3 | `tiling_core_num` | host 侧分配的 AI Core 数 |

### **4. TBE 分段与分核策略**

TBE 先通过 `get_task_block_size()` 计算每个逻辑 task 的元素数量 `aligned_elem_per_core`：

```text
block_size = num_elem_x / tiling_core_num
block_size_aligned = align_up(block_size, MAX_VEC_PROCESS_NUM)

if num_elem_x >= tiling_core_num * TASK_ALIGN:
    block_size_aligned = TASK_ALIGN

if num_elem_x <= MAX_VEC_PROCESS_NUM:
    block_size_aligned = MAX_VEC_PROCESS_NUM
```

因此 task 长度明确分为三类：

| 输入规模 | `aligned_elem_per_core` |
| -------- | ------------------------ |
| `num_elem_x <= 64` | 64 |
| `64 < num_elem_x < tiling_core_num * 4096` | `num_elem_x / tiling_core_num` 向上对齐到 64 |
| `num_elem_x >= tiling_core_num * 4096` | 4096 |

随后计算逻辑 task 数并分配到 AI Core：

```text
logi_task_num = ceil(num_elem_x / aligned_elem_per_core)
logi_task_num_per_aicore = logi_task_num / tiling_core_num
logi_task_tail = logi_task_num % tiling_core_num
```

每个核的任务数为：

```text
core_task_num = logi_task_num_per_aicore
if aicore_idx < logi_task_tail:
    core_task_num = core_task_num + 1
```

每个核开始前需要计算当前核之前已经处理的 mask true 数：

```text
prefix_len = pre_core_task_num * aligned_elem_per_core
task_updates_start = calc_updates_start(prefix_len)
```

这个 `task_updates_start` 是该核第一个 task 从 `updates` 读取的全局起始下标。本核内多个 task 之间通过 `compute()` 返回的新 `updates_start` 串联，不重复统计前缀。

### **5. TBE `calc_updates_start` 逻辑**

`calc_updates_start(num_elem_mask)` 统计 `mask[0 : num_elem_mask]` 中 true 元素数量。

完整流程：

1. `sum_res_int64 = 0`。
2. `data_move_length = TASK_ALIGN = 4096`。
3. 若 `0 < num_elem_mask < TASK_ALIGN`，则 `data_move_length = num_elem_mask`。
4. `iters = ceil(num_elem_mask / data_move_length)`。
5. 分配长度为 `data_move_length` 的 `bool_ub`、`fp16_ub`、`fp32_ub`、`work_tensor_ub`、`sum_fp32_ub`。
6. 每轮按 32B 对齐从 `mask_gm` 搬入 `bool_ub`。
7. `vec_conv(bool -> fp16)`。
8. `vec_conv(fp16 -> fp32)`。
9. `vec_reduce_add(fp32)` 得到本轮 mask true 数。
10. `scalar_conv(round)` 转为 int32。
11. 累加到 `sum_res_int64`。

关键对齐点：TBE 每次 reduce 的 mask 元素分段长度是 `TASK_ALIGN = 4096`，不是 `MAX_VEC_PROCESS_NUM = 64`。

### **6. TBE `compute` 逻辑**

`compute(input_offset, updates_start)` 处理一个逻辑 task。

完整流程：

1. 默认 `num_elem_per_input = aligned_elem_per_core`。
2. 若当前 task 越过 `num_elem_x` 尾部，则 `num_elem_per_input = num_elem_x - input_offset`。
3. 计算 `burst_x = ceil(num_elem_per_input / num_elem_per_block)`。
4. 计算 `burst_mask`，mask 按 1 字节元素和 32B block 对齐搬运。
5. 从 GM 搬入 `x[input_offset]` 到 `x_ub`。
6. 从 GM 搬入 `mask[input_offset]` 到 `mask_ub`。
7. 计算 `num_remain_updates = num_elem_updates - updates_start`。
8. 若 `num_remain_updates > num_elem_per_input`，则截断为 `num_elem_per_input`。
9. 若 `num_remain_updates > 0`，从 `updates_gm` 搬入本 task 最多需要的 updates。
10. 若对齐搬运会越过 `updates` 尾部，则通过 `updates_back_offset` 回退 GM 读取起点，避免越界。
11. 遍历 task 内 offset：
    - `mask_ub[offset] == 0`：保留 `x_ub[offset]`。
    - `mask_ub[offset] != 0`：写入 `updates_ub[updates_offset_ub + updates_back_offset]`，并递增 `updates_offset_ub`。
12. `updates_start = updates_start + updates_offset_ub`。
13. 将 `x_ub` 搬出到 `y_gm[input_offset]`。
14. 返回新的 `updates_start`。

### **TBE 整体流程图**

流程图源码文件：[`flowcharts/tbe/01_tbe_overall.mmd`](flowcharts/tbe/01_tbe_overall.mmd)

### **TBE task_schedule 流程图**

流程图源码文件：[`flowcharts/tbe/02_tbe_task_schedule.mmd`](flowcharts/tbe/02_tbe_task_schedule.mmd)

### **TBE calc_updates_start 流程图**

流程图源码文件：[`flowcharts/tbe/03_tbe_calc_updates_start.mmd`](flowcharts/tbe/03_tbe_calc_updates_start.mmd)

### **TBE compute 流程图**

流程图源码文件：[`flowcharts/tbe/04_tbe_compute.mmd`](flowcharts/tbe/04_tbe_compute.mmd)

# **需求分析**

## **外部组件依赖**

不涉及新增外部组件依赖。

本算子依赖 CANN 基础能力：

| 组件 | 作用 |
| ---- | ---- |
| Ascend C kernel API | GM/UB Tensor、DataCopyPad、Cast、WholeReduceSum、PipeBarrier 等 kernel 侧能力 |
| op_host tiling API | host 侧 dtype/shape 校验、tiling data 写入、block dim 设置 |
| GE op 注册框架 | 算子原型、infer shape、AICore config 注册 |
| aclnn 调用框架 | example 中通过两段式 aclnn API 调用 custom 算子 |

## **内部适配模块**

适配 Aclnn 接口调用，并在 experimental 目录中提供算子原型、host tiling、kernel、example 和设计文档。

| 模块 | 路径 | 作用 |
| ---- | ---- | ---- |
| 算子原型 | `op_graph/masked_scatter_proto.h` | 声明 `MaskedScatter` 输入输出 |
| host 注册 | `op_host/masked_scatter_def.cpp` | 注册 dtype/format 和 AICore config |
| infer shape | `op_host/masked_scatter_infershape.cpp` | 输出 shape 等于 `x` shape |
| tiling | `op_host/masked_scatter_tiling.cpp` | 校验 dtype/shape，写 tiling data，设置 block dim |
| tiling data | `op_kernel/masked_scatter_tiling_data.h` | 定义 kernel 读取的 4 个 int64 tiling 字段 |
| tiling key | `op_kernel/masked_scatter_tiling_key.h` | 声明 Ascend C 模板参数 |
| kernel 入口 | `op_kernel/masked_scatter.cpp` | 读取 tiling，调用 `Init` 和 `Process` |
| kernel 实现 | `op_kernel/masked_scatter.h` | 实现分核、mask 前缀计数、masked scatter 计算 |
| aclnn example | `examples/test_aclnn_masked_scatter.cpp` | 保留一条 eager custom 示例 |
| 设计文档 | `docs/masked_scatter_design.md` | 记录 TBE 和 Ascend C 设计 |
| 流程图源码 | `docs/flowcharts/` | 单独保存 Mermaid 流程图代码 |

## **需求模块设计**

### **算子原型**

| **名称** | **类别** | **dtype** | **format** | **shape** | **介绍** |
| -------- | -------- | --------- | ---------- | --------- | -------- |
| x | 输入 | float16 / float32 / uint8 / int8 / int16 / int32 / bool / bfloat16 | ND | all | 原始输入张量 |
| mask | 输入 | bool | ND | 与 x 相同 | 控制替换位置的 bool mask |
| updates | 输入 | 与 x 相同 | ND | all | 替换值来源 |
| y | 输出 | 与 x 相同 | ND | 与 x 相同 | masked scatter 后的输出张量 |

**属性**：

MaskedScatter 当前无属性。

### **shape 与 dtype 约束**

| **约束项** | **规则** | **实现位置** |
| ---------- | -------- | ------------ |
| mask dtype | 必须为 bool | `masked_scatter_tiling.cpp` |
| x/updates/y dtype | 必须完全一致 | `masked_scatter_tiling.cpp` |
| x/mask/y shape | 必须完全一致 | `masked_scatter_tiling.cpp` |
| y shape | 等于 x shape | `masked_scatter_infershape.cpp` |

## **算子支持型号**

Atlas A2 训练系列产品 / Atlas 800I A2 推理产品（ascend910b）。

# **需求详细设计**

## **使能方式**

| **上层框架** | **涉及的框架勾选** |
| ------------ | ------------------ |
| TF训练/推理 | |
| Pytorch训练/推理 | |
| ATC推理 | |
| Aclnn直调 | √ |
| OPAT调优 | |
| SGAT子图切分 | |

## **需求总体设计**

### **host侧设计方案**

MaskedScatter host 侧负责输出 shape 推导、输入合法性校验、tiling data 写入和 block dim 设置。

#### **1) InferShape**

输出 shape 与输入 `x` 完全一致，逐维度复制：

```cpp
const gert::Shape* xShape = context->GetInputShape(0);
gert::Shape* yShape = context->GetOutputShape(0);
*yShape = *xShape;
```

该逻辑与 TBE 语义一致：MaskedScatter 只改写元素值，不改变张量 rank 和 shape。

#### **2) dtype 校验**

host tiling 中定义支持 dtype 集合：

```cpp
const std::set<ge::DataType> supportedDtype = {
    ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_UINT8, ge::DT_INT8,
    ge::DT_INT16, ge::DT_INT32, ge::DT_BOOL, ge::DT_BF16};
```

校验规则：

1. `x` dtype 必须在支持集合中。
2. `mask` dtype 必须为 `ge::DT_BOOL`。
3. `x`、`updates`、`y` dtype 必须完全一致。

#### **3) shape 校验**

host tiling 中读取 `x`、`mask`、`y` 的 storage shape，并要求：

```text
xShape == maskShape
xShape == yShape
```

`updates` shape 不要求与 `x` 相同，语义上只要求按 mask true 顺序提供替换值。

#### **4) tiling 参数**

Ascend C 使用与 TBE `tiling_gm` 一致的 4 个字段：

```cpp
struct MaskedScatterTilingData {
    int64_t numElemX;
    int64_t numElemMask;
    int64_t numElemUpdates;
    int64_t tilingCoreNum;
};
```

host 侧写入规则：

```text
numElemX = input x storage shape size
numElemMask = numElemX
numElemUpdates = input updates storage shape size
tilingCoreNum = PlatformAscendC.GetCoreNumAiv()
```

#### **5) 分核策略**

host 侧设置：

```cpp
context->SetBlockDim(static_cast<uint32_t>(coreNum));
```

具体逻辑 task 长度、逻辑 task 数、每核 task 数在 kernel `Init()` 和 `Process()` 中计算。该设计与 TBE 一致：host 只传入 `tiling_core_num`，kernel 内按 `TASK_ALIGN` 和 `MAX_VEC_PROCESS_NUM` 计算分段。

#### **Ascend C Host Tiling 流程图**

流程图源码文件：[`flowcharts/ascend/05_ascend_host_tiling.mmd`](flowcharts/ascend/05_ascend_host_tiling.mmd)

### **kernel侧设计方案**

Kernel 侧执行 `Init` 和 `Process` 两个阶段。

#### **1) Kernel 入口**

`masked_scatter.cpp` 中入口逻辑如下：

```cpp
REGISTER_TILING_DEFAULT(MaskedScatterTilingData);
GET_TILING_DATA_WITH_STRUCT(MaskedScatterTilingData, tilingData, tiling);
NsMaskedScatter::MaskedScatter<DTYPE_X> op;
op.Init(x, mask, updates, y, &tilingData);
op.Process();
```

当前 `schMode` 仅作为模板参数声明存在，实际 kernel 使用单一路径，语义对应 TBE Python 版本的通用实现。

#### **2) Init**

`Init` 完成 GM 绑定、tiling 读取、dtype 信息计算、task 参数计算和 UB buffer 初始化。

GM 绑定：

| 成员 | GM 输入输出 | 说明 |
| ---- | ----------- | ---- |
| `xGm_` | x | 原始输入 |
| `maskGm_` | mask | 按 int8 访问，用于逐元素判断 |
| `maskGmUint8_` | mask | 按 uint8 访问，用于前缀 reduce 统计 |
| `updatesGm_` | updates | 替换值来源 |
| `yGm_` | y | 输出 |

task 参数计算：

```cpp
dtypeBytesSize_ = sizeof(DTYPE);
numElemPerBlock_ = BYTES_PER_BLOCK / dtypeBytesSize_;
alignedElemPerCore_ = GetTaskBlockSize();
logiTaskNum_ = CeilDiv(numElemX_, alignedElemPerCore_);
logiTaskNumPerAicore_ = logiTaskNum_ / tilingCoreNum_;
logiTaskTail_ = logiTaskNum_ % tilingCoreNum_;
```

UB buffer 初始化全部以 `TASK_ALIGN = 4096` 为最大 task 长度：

| buffer/queue | 大小 |
| ------------ | ---- |
| `inQueueX_` | `TASK_ALIGN * dtypeBytesSize_` |
| `inQueueMask_` | `TASK_ALIGN * 1` |
| `inQueueUpdates_` | `TASK_ALIGN * dtypeBytesSize_` |
| `outQueueY_` | `TASK_ALIGN * dtypeBytesSize_` |
| `boolBuf_` | `TASK_ALIGN * 1` |
| `fp16Buf_` | `TASK_ALIGN * sizeof(float16)` |
| `fp32Buf_` | `TASK_ALIGN * sizeof(float32)` |
| `sumBuf_` | `TASK_ALIGN * sizeof(float32)` |

#### **3) GetTaskBlockSize**

Ascend C 与 TBE 保持一致：

```cpp
int64_t blockSizeAligned = MAX_VEC_PROCESS_NUM;
if (tilingCoreNum_ != 0) {
    int64_t blockSize = numElemX_ / tilingCoreNum_;
    blockSizeAligned = AlignDiv(blockSize, MAX_VEC_PROCESS_NUM);
}
if (numElemX_ >= tilingCoreNum_ * TASK_ALIGN) {
    blockSizeAligned = TASK_ALIGN;
}
if (numElemX_ <= MAX_VEC_PROCESS_NUM) {
    blockSizeAligned = MAX_VEC_PROCESS_NUM;
}
return blockSizeAligned;
```

这里的 `TASK_ALIGN = 4096`、`MAX_VEC_PROCESS_NUM = 64` 与 TBE 常量保持一致。

#### **4) Process**

每个 AI Core 只处理分配给自己的逻辑 task：

1. `aicoreIdx = GetBlockIdx()`。
2. 计算本核 `coreTaskNum`。
3. 计算本核之前已经分配的 `preCoreTaskNum`。
4. 若 `coreTaskNum > 0`，先调用：

```cpp
taskUpdatesStart = CalcUpdatesStart(preCoreTaskNum * alignedElemPerCore_);
```

5. 逐 task 调用：

```cpp
taskUpdatesStart = Compute((preCoreTaskNum + taskIdx) * alignedElemPerCore_,
                           taskUpdatesStart);
```

该逻辑与 TBE `task_schedule()` 一致：每核只在开头统计一次前缀 mask true 数，本核内部 task 通过 `Compute` 返回值继续推进 `updatesStart`。

#### **5) CalcUpdatesStart**

Ascend C 的 mask 前缀计数流程与 TBE 对齐，核心常量为：

```cpp
constexpr int64_t TASK_ALIGN = 4096;
constexpr int64_t COUNT_REDUCE_LEN = TASK_ALIGN;
```

完整流程：

1. `sumCount = 0`。
2. `dataMoveLength = COUNT_REDUCE_LEN`，即 4096。
3. 若 `0 < numElemMask < COUNT_REDUCE_LEN`，则 `dataMoveLength = numElemMask`。
4. `iters = CeilDiv(numElemMask, dataMoveLength)`。
5. 每轮按 32B 对齐搬入 mask。
6. 计算 `curDataLen` 和 `alignedDataLen`。
7. 调用 `CountMaskNonZero(maskUb, curDataLen, alignedDataLen)`。
8. 累加每轮 true 数，返回 `sumCount`。

`CountMaskNonZero` 的逻辑：

1. 对 pad 区域置 0，避免尾部 32B 对齐补齐区域参与统计。
2. `Cast(uint8 mask -> fp16)`。
3. `Cast(fp16 -> fp32)`。
4. `Abs`。
5. `Mins(..., 1.0)`，把任意非零 mask 归一化为 1。
6. `WholeReduceSum`。
7. 通过 `V_S` 事件同步后读取 sum。
8. round 后返回 int64。

重点对齐项：此前 Ascend C 使用 `COUNT_REDUCE_LEN = 64` 会导致前缀 mask 计数分段与 TBE 不同；当前已对齐为 `COUNT_REDUCE_LEN = TASK_ALIGN = 4096`。

#### **6) Compute**

`Compute(inputOffset, updatesStart)` 处理一个逻辑 task。

完整流程：

1. 默认 `numElemPerInput = alignedElemPerCore_`。
2. 若 task 越过 `numElemX_` 尾部，则缩短为尾部实际长度。
3. 计算 `burstX` 和 `burstMask`。
4. 从 `xGm_` 搬入 `yUb`，相当于先默认 `y = x`。
5. 从 `maskGm_` 搬入 `maskUb`。
6. 计算当前 task 可读取的 updates 数：

```text
numRemainUpdates = numElemUpdates_ - updatesStart
numRemainUpdates = min(numRemainUpdates, numElemPerInput)
```

7. 若还有 updates 可读，则搬入 `updatesUb`。
8. 若按 block 对齐读取会超过 updates 尾部，则使用 `updatesBackOffset` 回退读起点。
9. 遍历 task 内 offset：
   - `maskUb[offset] == 0`：保留 `yUb[offset]`，即原始 `x` 值。
   - `maskUb[offset] != 0 && updatesOffsetUb < numRemainUpdates`：写 `updatesUb[updatesOffsetUb + updatesBackOffset]`。
10. `updatesStart += updatesOffsetUb`。
11. 将 `yUb` 搬出到 `yGm_[inputOffset]`。
12. 返回新的 `updatesStart`。

#### **7) TBE 与 Ascend C 对齐关系**

| 设计点 | TBE | Ascend C | 结论 |
| ------ | --- | -------- | ---- |
| 算子输入输出 | `x, mask, updates -> y` | `x, mask, updates -> y` | 一致 |
| tiling 字段 | 4 个 int64 | 同 4 个字段 | 一致 |
| task 长度计算 | `get_task_block_size()` | `GetTaskBlockSize()` | 一致 |
| 大输入 task 长度 | `TASK_ALIGN = 4096` | `TASK_ALIGN = 4096` | 一致 |
| 前缀 mask reduce 分段 | 每 4096 个 mask 元素 | `COUNT_REDUCE_LEN = TASK_ALIGN` | 一致 |
| 默认输出值 | 先搬入 x，再按 mask 覆盖 | 先搬入 x 到 yUb，再按 mask 覆盖 | 一致 |
| updates 尾部处理 | `updates_back_offset` | `updatesBackOffset` | 一致 |
| bfloat16 | 映射为 float16 | BF16 注册，元素字节数为 2 | task/搬运长度一致 |
| int64 | TBE 字节表包含 | 当前未注册 | 当前 experimental 不支持 |

### **Ascend C Kernel 流程图**

#### **1. Kernel 入口与 Init 流程图**

流程图源码文件：[`flowcharts/ascend/06_ascend_kernel_entry_init.mmd`](flowcharts/ascend/06_ascend_kernel_entry_init.mmd)

#### **2. Process 分核调度流程图**

流程图源码文件：[`flowcharts/ascend/07_ascend_process_schedule.mmd`](flowcharts/ascend/07_ascend_process_schedule.mmd)

#### **3. CalcUpdatesStart 流程图**

流程图源码文件：[`flowcharts/ascend/08_ascend_calc_updates_start.mmd`](flowcharts/ascend/08_ascend_calc_updates_start.mmd)

#### **4. Compute 流程图**

流程图源码文件：[`flowcharts/ascend/09_ascend_compute.mmd`](flowcharts/ascend/09_ascend_compute.mmd)

#### **5. Aclnn example 调用流程图**

流程图源码文件：[`flowcharts/ascend/10_aclnn_example.mmd`](flowcharts/ascend/10_aclnn_example.mmd)

### **Aclnn 示例设计**

示例文件：

```text
examples/test_aclnn_masked_scatter.cpp
```

当前 example 只保留一条测试，与 `gather_elements_v3` example 风格一致，走 eager custom aclnn 两段式调用。

输入：

```text
x       = [1, 2, 3, 4, 5, 6, 7, 8]
mask    = [1, 0, 1, 0, 1, 0, 1, 0]
updates = [10, 20, 30, 40]
```

调用：

```cpp
aclnnMaskedScatterGetWorkspaceSize(self, mask, source, out, &workspaceSize, &executor);
aclnnMaskedScatter(workspaceAddr, workspaceSize, executor, stream);
```

期望输出：

```text
y = [10, 2, 20, 4, 30, 6, 40, 8]
```

### **编译安装与验证**

编译命令：

```bash
bash build.sh --pkg --soc=ascend910b --ops=masked_scatter --experimental --vendor_name=custom
```

安装命令：

```bash
./build_out/cann-ops-nn-custom_linux-aarch64.run
```

example 验证命令：

```bash
export LD_LIBRARY_PATH=/home/developer/Ascend/cann-8.5.2/opp/vendors/custom_nn/op_api/lib/:${LD_LIBRARY_PATH}
bash build.sh --run_example masked_scatter eager cust --vendor_name=custom --experimental
```

当前验证输出：

```text
result[0] is: 10.000000
result[1] is: 2.000000
result[2] is: 20.000000
result[3] is: 4.000000
result[4] is: 30.000000
result[5] is: 6.000000
result[6] is: 40.000000
result[7] is: 8.000000
Run test_aclnn_masked_scatter success.
```

## **支持硬件**

| **支持的芯片版本** | **涉及勾选** |
| ------------------ | ------------ |
| 香橙派OrangePi AIpro | |
| Atlas 200I/500 A2推理产品 | |
| Atlas 800I/T A2 | √ |
| Atlas A2训练系列产品 | √ |

## **算子约束限制**

* `x`、`mask`、`y` shape 必须完全相同。
* `mask` dtype 必须为 bool。
* `x`、`updates`、`y` dtype 必须完全相同。
* 当前 experimental Ascend C 算子未注册 `int64`，即使 TBE dtype 字节表包含 `int64`。
* 当前 kernel 按扁平一维顺序处理所有元素，不区分原始 rank。
* 当前实现没有显式校验 `updates` 元素数量必须大于等于 mask true 数；当当前 task 可读 updates 不足时，`updatesOffsetUb < numRemainUpdates` 会阻止继续写入，后续位置保留原始 `x` 值。

# **特性交叉分析可维可测分析**

## **精度标准/性能标准**

| **验收标准** | **描述(不涉及说明原因)** | **标准来源** |
| ------------ | ------------------------ | ------------ |
| 精度标准 | 与 TBE 版本按元素比对一致 | 历史 TBE 对标 |
| 分核标准 | task 长度、逻辑 task 分配、每核前缀 updates 起点与 TBE 一致 | TBE 源码对齐 |
| 性能标准 | 大输入 task 长度为 4096，mask 前缀计数每 4096 个元素 reduce 一次 | TBE 源码对齐 |

## **关联的 Issue**

暂无。

## **文档更新**

本文档。

## **类型标签**

* [ ] Bug修复
* [ ] 新特性
* [ ] 性能优化
* [ ] 文档更新
* [x] 其他，请描述：社区任务算子设计文档
