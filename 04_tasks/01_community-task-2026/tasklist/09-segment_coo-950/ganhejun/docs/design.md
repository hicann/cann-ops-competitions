# 需求背景（required）

## 需求来源

本算子对应 2026 年 CANN 社区任务 `segment_coo`，参考
[torch_scatter.segment_coo](https://pytorch-scatter.readthedocs.io/en/latest/functions/segment_coo.html)
及任务书验收约束，目标硬件为 Ascend 950PR。

## 背景介绍

图神经网络的 COO 边索引通常已按目标节点排序，需要沿有序索引将消息分段归约。
通用 scatter 支持任意顺序但会产生随机访存和原子更新；segment_coo 利用连续段可
显著降低访存及同步开销。本实现提供 PyTorch/NPU 适配层，并保留 CSR 兼容包装。

# 需求分析（required）

## 需求描述

沿 `index.dim()-1` 维，将 `src` 中索引相同的元素执行 `sum/add/mean/min/max`
归约。支持 `out`、`dim_size`、空输入、空段、非连续张量、多维广播及
`min/max` 的 `arg_out`。

## 需求拆解

* 支持 FLOAT16、FLOAT32、BFLOAT16、INT8、UINT8、INT32、INT64；
* 支持 1～8 维 `src`，`index.dim() <= src.dim()` 和广播；
* `index` 最后一维须非降序，取值范围为 `[0, dim_size-1]`；
* 自动输出时空段为 0，`min/max` 返回最后一个相等元素的位置；
* PyTorch 层接口优先，性能不低于标杆 0.45 倍。

# 详细设计（required）

## 算子分析

### 数学公式

对归约轴上的位置 `j` 和 segment `g`：`out[g] = reduce({src[j] | index[j] = g})`。
`mean` 为段内和除以段长度；空段输出 0；`arg_out[g]` 为极值最后出现位置，空段
为归约轴长度。

### 输入输出规格

| 参数 | 说明 |
|---|---|
| `src` | 1～8 维数据张量 |
| `index` | PyTorch 层固定使用 INT64 COO 索引（适配层兼容 INT32 并转换），最后一维有序 |
| `dim_size` | 可选 segment 数；缺省为 `index.max()+1` |
| `reduce` | `sum/add/mean/min/max`，不支持 `mul` |
| `out` | 可选同 dtype、同设备输出缓冲区 |
| 返回值 | 普通 reduce 返回 Tensor；命名 min/max 返回 `(Tensor, INT64 arg_out)` |

## 算子实现

### 实现方案

#### host 侧设计

Host 适配层校验 rank、dtype、设备、index 可广播性、索引范围和 `dim_size`，并将
归约轴移动到第 0 维。dim=0 且一维 index 选择紧凑路径；其余情况使用广播后的
自然索引视图。按照任务约束，调用方必须保证 index 最后一维非降序；设置
`OPS_GNN_VALIDATE_INDEX=1` 时执行范围检查，测试层额外检查排序。NPU 默认不做
全量排序扫描，避免每次调用产生同步；不满足排序约束的输入行为按任务书定义为未定义。

#### kernel 侧设计

NPU 的 min/max 热路径复用仓库已有 Ascend C `scatter_forward` 内核：COO 的原始
紧凑 index 按 `dim=index.dim()-1` 直接映射到 scatter 的归约轴，避免为每个特征列
物化完整 index。内核按输入元素数和 UB 容量选择向量原子、热点目标或通用 lane-owner
路径，并同时维护值和位置以输出 arg_out。950 上的一维有序 COO 使用专用
ordered-segment 路径：规则等长段直接以段长寻址，不再初始化或扫描边界；不规则段
先并行扫描 begin/end。归约 kernel 使用 1024 个 SIMT 线程，并让单线程处理多个连续
feature，以复用段遍历和行地址；1～4 行的规则短段使用 reshape 后的设备原生归约。
INT64 也通过该路径在 NPU 原生执行。BF16 的 sum/mean 使用 FP32 中间精度；未构建
扩展、CPU 输入及不满足热路径条件的输入使用紧凑 PyTorch 参考路径。

#### 分核、分块和 tiling 策略

按输入元素数、segment 数和 feature 数动态计算 block 数，950 上最多调度三波 AIV
block 以隐藏 GM 访存等待；每个 block 启动 1024 个 SIMT 线程。规则段按 8-feature
tile 展开，尾部 feature 逐项保护；不规则段使用独立 bounds workspace。sum/mean 与
min/max 采用不同归约逻辑，min/max 同时写回最后一个相等元素的 INT64 arg_out。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
|---|---|
| Ascend 950PR | √ |

软件环境：CANN 9.1.0+、PyTorch 2.7+、匹配版本 torch_npu。

## 算子约束限制

1. `index` 最后一维必须非降序；无序输入行为未定义。
2. `src`、`index`、`out` 必须在同一设备；`out` dtype 必须与 `src` 相同。
3. 不支持 BOOL、COMPLEX、Float8 和 `reduce="mul"`。
4. NPU 默认跳过全量排序/范围扫描；调试时设置 `OPS_GNN_VALIDATE_INDEX=1`。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
|---|---|---|
| 精度标准 | 与 torch_scatter CPU 参考一致；FP16/BF16 按规定误差阈值，INT32/INT64 bit-wise 一致 | [任务书](https://gitcode.com/org/cann/discussions/285) |
| 性能标准 | 每组 benchmark 的标杆耗时/实测耗时 ≥ 0.45 | [任务书](https://gitcode.com/org/cann/discussions/285) |

## 可测用例

功能测试覆盖五种 reduce、七种 dtype、空输入/空段、dim_size、out、非连续 src/index、
多维广播和 CSR 一致性。性能测试覆盖 N=256K～4096K、C=32/64/128/256、不同
segment 密度，以及 float16/float32/int32/int64；四种 dtype 均按任务书参与性能验收。

### 多路径性能对比（950PR 实测）

| 场景 | 路径 | 典型耗时 | 说明 |
|---|---|---:|---|
| 规则等长、有序、短段 | reshape/pairwise device reduction | 0.04–0.28 ms | 直接按 `[segment, rows, feature]` 归约 |
| 规则等长、有序、一般段长 | ordered-segment SIMT tile | 0.05–2.36 ms | 1024 线程、8-feature tile |
| 不规则但有序 | bounds init + parallel scan + reduce | 需按输入实测 | 保证空段和稀疏段语义 |
| 任意广播/非连续/CPU | PyTorch reference/index_add fallback | 不纳入 NPU 热路径 | 优先保证接口和精度 |

与通用 scatter 的选型差异：scatter 允许任意顺序，通常需要原子更新或热点目标处理；
segment_coo 利用连续有序段，避免大规模 atomics，并可在规则段直接计算边界。因此
规则有序输入走 ordered-segment 路径，不满足形状条件时才复用 scatter 的通用实现，
避免为了单一 benchmark 牺牲广播、非连续和空段兼容性。

# 兼容性分析

Segment COO 为新增 PyTorch/NPU 适配接口，不修改既有算子 ABI；CSR 包装仅在 Python
层由 `indptr` 构造有序 COO 索引后复用同一实现。与 torch_scatter 的参数、返回值和
空段语义保持兼容。

## 提交参考链接

* [设计文档模板](https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md)
* [社区任务流程与讨论 285](https://gitcode.com/org/cann/discussions/285)
* [自测报告表格模板](https://docs.qq.com/sheet/DUmVWWndaUE12WGFB?tab=BB08J2)
