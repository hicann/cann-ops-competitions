# 【社区任务】ScatterND 算子设计文档

## 一、需求背景

### 1.1 需求来源

通过社区任务完成 ScatterND 算子在 ops-nn 开源仓中的实现与适配。

### 1.2 背景介绍

#### 1.2.1 ScatterND 算子实现优化

ScatterND 算子用于将 updates 张量中的值根据 indices 索引分散更新到 input 张量的指定位置。

**TBE 算子源码路径（本次任务参考实现）：**



该文件为 ScatterNd TBE legacy 算子适配脚本，基于 Tik DSL 实现。原始 CANN toolkit 内置路径通常位于：

`/home/developer/Ascend/cann-9.0.0-beta.2/opp/built-in/op_impl/ai_core/tbe/impl/ops_legacy/dynamic/scatter_nd.py`

**kernel 实现：**
/home/developer/Ascend/ascend-toolkit/latest/opp/built-in/op_impl/ai_core/tbe/impl/dynamic/

**算子信息库路径：**

`/home/developer/Ascend/ascend-toolkit/latest/opp/built-in/op_impl/ai_core/tbe/config/ascend910b`



**算子原型路径：**

- `/home/developer/Ascend/ascend-toolkit/latest/opp/built-in/op_proto/inc/`

**参考文件正确性验证：**

aclnn 接口为 `aclnnScatterNd`，其输入为 `(input, indices, updates)`，输出为 `output`。TBE 源码 `scatter_nd.py` 中：

- 算子注册名为 `"ScatterNd"`（`@register_operator("ScatterNd")`），与 aclnn 接口名称及原型注册 `REG_OP(ScatterNd)` 一致；
- 入口函数 `scatter_nd(indices, x, shape, y, kernel_name="ScatterNd", impl_mode=None)` 的参数语义为：
  - `indices`：索引张量，对应 aclnn 的 `indices`；
  - `x`：更新张量 `updates`；
  - `shape`：输出张量 `output` 的 shape（由 input shape 推导）；
  - `y`：输出张量 `output`；
  - 因此该函数包含 3 个输入参数（indices/x/shape）和 1 个输出参数（y），与原型注册参数一致；
- 入口函数根据平台能力选择 Tik 或 DSL 实现路径：若平台不支持 `tbe.dsl.vexp` float32，走 Tik 路径（`scatter_nd_tik`）；否则走 DSL 路径（`scatter_nd_dsl`）；
- Tik 路径的 `ScatterNd` 类通过 Tik 直接 CCE 编程，包含 16 种 `tiling_mode` 分支，分为 atomic（mode 1-5, 16）和非 atomic（mode 6-15）两大类；
- DSL 路径调用 `tbe.scatter_nd(var_tensor, indices_tensor, updates_tensor, reduction)` 完成计算语义定义，参数语义与 aclnn 的 `input/indices/updates` 对应。

因此上述 TBE 源码文件为本算子的正确参考实现。

#### 1.2.2 ScatterND算子现状分析

##### 1.2.2.1 支持的数据类型和数据格式

根据算子信息库，ScatterND 的 TBE 版本在 Atlas A2 平台支持的数据类型为 FLOAT16、FLOAT、INT32。算子信息库声明支持动态编译、动态 format、动态 rank、动态 shape，输入输出 shape 能力为 all。
本算子最终支持 ND storage format，data/updates/output dtype 支持 FLOAT16、FLOAT、INT32，indices dtype 支持 INT32、INT64。

| 项目             | 本次任务范围                    | 当前实现                    |
|:-------------- |:------------------------- |:----------------------- |
| 输入             | input, indices, updates   | input, indices, updates |
| 输出             | output                    | output                  |
| dtype          | FLOAT16, FLOAT, INT32     | FLOAT16, FLOAT, INT32   |
| indices dtype  | INT32, INT64              | INT32, INT64            |
| storage format | ND                        | ND                      |
| dynamic        | dynamic shape/rank/format | 运行时推导 shape             |

##### 1.2.2.2 算子实现描述

TBE 适配脚本的主要逻辑如下：

1. `scatter_nd` 入口函数根据平台能力选择 Tik 或 DSL 实现路径。
2. `scatter_nd_tik`（Tik 路径）：创建 ScatterNd 类实例，通过 Tik 直接构建多核并行计算程序。
3. `scatter_nd_dsl`（DSL 路径）：通过 TBE DSL 的 `tbe.scatter_nd()` 接口完成计算，由框架自动调度。

**Tik 实现核心流程：**

1. `ScatterNd.__init__()` 完成参数校验（`check_input_params`）、缓冲区大小计算、Tik Tensor/Scalar 声明。
2. `scatter_nd_compute_tiling()` 从 GM 读取 35 参数 tiling 到 UB，解析 tiling_args，根据 `tiling_mode` 选择分核策略。
3. 分为两大类执行路径：
   - **atomic 模式（tiling_mode 1-5, 16）**：通过 `set_atomic_add` 直接将 updates 原子累加到 output GM。
   - **非 atomic 模式（tiling_mode 6-15）**：从 GM 读取 var → UB，在 UB 中 `vec_add`（var + updates）后写回 GM。
4. 每个模式根据 UB 容量、32B 对齐、数据量大小选择不同的搬运和计算策略。
5. `BuildCCE` 编译生成最终核函数。

**DSL 实现核心流程：**

1. 参数校验（dtype 检查）。
2. `classify` 对输入进行分类，`shape_util.variable_shape` 推导动态 shape。
3. 创建占位 Tensor（var_tensor、indices_tensor、updates_tensor、shape_tensor）。
4. 调用 `tbe.scatter_nd(var_tensor, indices_tensor, updates_tensor, reduction)` 完成计算语义定义。
5. `tbe.auto_schedule` 自动调度，`tbe.build` 编译构建。

核心计算语义如下：

```python
# indices.shape = (M, N), updates.shape = (M, *input.shape[N:])
for idx in range(M):
    # 根据 indices 计算多维偏移量
    outputIdx = compute_offset(indices[idx], output_shape)
    output[outputIdx] = updates[idx]
```

##### 1.2.2.3 算子实现流程图

以下为 TBE 参考实现的完整流程图，包含 DSL 与 Tik 两条路径，以及 Tik 路径中 16 种 `tiling_mode` 的详细分支。

![TBE 参考实现流程图](![ScatterNdTensor.png](https://raw.gitcode.com/user-images/assets/10130836/9f04043d-366d-43b3-9f93-e331db929be3/ScatterNdTensor.png 'ScatterNdTensor.png'))


## 二、需求分析

### 2.1 外部组件依赖

| 外部依赖           | 使用位置     | 作用                          |
|:-------------- |:-------- |:--------------------------- |
| TBE 框架         | TBE 适配脚本 | 参数校验、调度生成、编译构建              |
| Tik 框架         | Tik 路径   | 直接 CCE 编程，多核并行、UB 管理、vec 指令 |
| TBE DSL        | DSL 路径   | `tbe.scatter_nd` 计算语义，auto_schedule |
| ACL Runtime    | 核函数      | 设备内存管理和 kernel 启动           |

### 2.2 内部适配模块

| 模块           | 文件                  | 作用                                      |
|:------------ |:------------------- |:--------------------------------------- |
| TBE Tik 适配模块 | `scatter_nd.py`     | 参数校验、Tik 多核并行构建、tiling 分发、编译调用        |
| TBE DSL 适配模块 | `scatter_nd.py`     | DSL compute + auto_schedule + build 路径 |
| Tik 计算类      | `ScatterNd`（同文件内）   | Tik 程序主体，包含 16 种 tiling_mode 分支        |

### 2.3 需求模块设计

#### 2.3.1 算子原型

| 名称      | 类别  | dtype                   | shape/取值              | 说明   |
|:------- |:--- |:----------------------- |:--------------------- |:---- |
| input   | 输入  | ND: FLOAT16/FLOAT/INT32 | 任意 shape              | 输入张量 |
| indices | 输入  | ND: INT32, INT64        | (M, N)                | 索引张量 |
| updates | 输入  | 与 input 相同              | (M, *input.shape[N:]) | 更新张量 |
| output  | 输出  | 与 input 相同              | 与 input 相同            | 输出张量 |

#### 2.3.2 算子相关约束

最终支持范围与 TBE/原型范围保持一致。约束如下：

1. `input` 和 `updates` 的 dtype 必须一致。
2. `indices` 的 dtype 必须为 INT32 或 INT64。
3. `indices` 和 `updates` 的 shape 必须完全一致。
4. `input` 的前 N 维必须与 `indices` 的 shape 匹配（N = len(indices.shape)）。

## 三、需求详细设计

### 3.1 使能方式

当前实现适配 TBE 调用框架。调用流程为：

```python
scatter_nd(input, indices, updates)
```

接口完成参数校验、输出 shape 推导、调度生成和核函数调用。

### 3.2 需求总体设计

#### 3.2.1 host 侧设计

host 侧在执行阶段解析输入 shape。TBE 参考实现中的关键参数与分核策略由 `tiling` 在运行前计算，核函数通过 `tiling_gm` 传入 35 个 int64 参数，包括：

- `tiling_mode`：执行模式选择（1-16）；
- `core_num`：实际使用的核数；
- `indice_step`：每个核负责的索引步长；
- `update_data_num`：每个索引对应的更新数据量；
- `indices_loop_num / indices_last_num`：indices 循环次数和尾数；
- `var_each_core_data / var_last_core_data`：每个核负责的 output 数据量；
- `var_each_core_set_zero_loop_num / var_last_core_set_zero_loop_num`：output 初始化清零参数。

#### 3.2.2 kernel 侧设计

##### 3.2.2.1 kernel 侧实现描述

本次任务 kernel 侧设计与 TBE 参考实现保持一致，采用 Tik 手写实现：

1. **入口阶段**：`scatter_nd` 根据平台能力选择 `scatter_nd_tik` 或 `scatter_nd_dsl`。
2. **Tik 初始化阶段**：`ScatterNd.__init__()` 校验参数，计算 UB 容量、数据块大小、atomic 支持等，并声明 Tik Tensor/Scalar。
3. **tiling 解析阶段**：`scatter_nd_compute_tiling()` 从 GM 读取 tiling 参数，解析后按 `tiling_mode` 分发。
4. **多核并行阶段**：每个 AI Core 执行 `init_ub_tensor()` 后，调用 `traversing_indices(mode)` 处理 indices。
5. **索引计算阶段**：`get_var_read_index(indices_ub_index)` 根据 indices 的每一行和 `var_offset_index_tiling` 计算 output 偏移。
6. **更新阶段**：
   - **atomic 模式**：`move_indices` 选择 atomic 分支，通过 `set_atomic_add` 将 updates 直接写回 `out_gm`；
   - **非 atomic 模式**：`move_indices` 选择非 atomic 分支，根据具体 `tiling_mode` 将 `out_gm` 中对应区域读取到 `var_ub`，与 `updates` 做 `vec_add` 后写回 `out_gm`。`traversing_var_single_core / traversing_var_mul_core` 等函数在源码中作为清零/初始化辅助函数存在，用于特定场景。
7. **编译阶段**：`scatter_nd_operator()` 调用 `BuildCCE` 生成核函数。

核心计算公式：

```text
outputIdx = sum(indices[idx][k] * var_offset_index_tiling[k])   # k = 0..indices_last_dim-1
output[outputIdx * update_data_num + innerIdx] = updates[idx * update_data_num + innerIdx]
```

##### 3.2.2.2 AscendC 实现流程图


![acendec流程图.png](https://raw.gitcode.com/user-images/assets/10130836/773c161f-e2f7-4875-a694-d927d2ff4286/acendec流程图.png 'acendec流程图.png')



##### 3.2.2.3 实现流程图与 TBE 流程图存在的差异点和原因

| 差异点      | TBE 参考实现（E:\download\scantter_nd\tbe\scatter_nd.py） | 本次任务设计（与 TBE 一致） | 原因            |
|:-------- |:------------------------------------------------- |:---------------------- |:------------- |
| 实现路径     | 同时存在 Tik 路径和 DSL 路径，运行时根据平台能力选择               | 与 TBE 一致              | 保持最大兼容性和性能 |
| 调度方式     | Tik 路径为显式手动分核调度（tiling_mode + 多核 for_range）       | 与 TBE 一致              | 手动控制并行粒度和 UB 使用 |
| 动态 shape | 由 TBE 框架 / tiling 参数在运行时推导                        | 与 TBE 一致              | 实现形式由框架决定 |
| 计算表达     | Tik 路径显式使用 `set_atomic_add` 或 `vec_add` 完成更新        | 与 TBE 一致              | 数学语义一致，与硬件能力匹配 |
| 初始化阶段    | 非 atomic 模式下读取 out_gm 对应区域到 var_ub，与 updates 做 `vec_add` 后写回；`traversing_var_single_core / traversing_var_mul_core` 等清零辅助函数在源码中定义，用于特定场景 | 与 TBE 一致              | 与 TBE 参考源码保持一致 |

### 3.3 支持硬件

Atlas A2 训练系列产品。

### 3.4 算子约束限制

1. 支持 dtype：FLOAT16、FLOAT、INT32。
2. storage format 支持 ND，且输入输出必须一致。
3. indices dtype 支持 INT32、INT64。
4. indices 和 updates 的 shape 必须完全一致。

## 四、特性交叉分析

| 特性            | 当前实现                      |
|:------------- |:------------------------- |
| 动态 shape      | host 侧根据运行时 shape 推导输出    |
| 动态 format     | 当前支持 ND                   |
| dtype         | ND 支持 FLOAT16、FLOAT、INT32 |
| indices dtype | 支持 INT32、INT64            |

## 五、可维可测分析

### 5.1 精度标准/性能标准

| 标准   | 描述                       |
|:---- |:------------------------ |
| 精度标准 | 输出满足 AscendOpTest 默认精度阈值 |
| 性能标准 | 所有核参与计算场景下性能与 TBE 持平     |

### 5.2 兼容性分析

当前实现的参数顺序与 TBE 接口语义一致，输入输出、属性列表、dtype/format 注册和 shape 推导均以 TBE 实现为基准。
