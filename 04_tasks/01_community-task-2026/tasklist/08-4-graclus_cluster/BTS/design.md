# graclus_cluster 算子设计文档

# 需求背景（required）

## 需求来源

本任务来自 CANN 社区 2026 年 8 月 GNN 算子开发任务，要求在 `ops-gnn` 仓中实现与 `torch_cluster.graclus_cluster` 接口和 PyTorch 层行为一致的 NPU 版本。任务适配硬件为 Ascend 950PR，CANN 版本以 `ops-gnn` 开源仓指定版本为准。

`graclus_cluster` 是图神经网络中常用的图粗化和池化前处理算子。输入为 COO 格式边 `row`、`col`，可选边权 `weight`，输出每个节点所属的 cluster ID。

## 背景介绍

### graclus_cluster 算子实现优化

Graclus 聚类采用贪心图匹配策略：按随机顺序遍历未标记节点，将当前节点与未标记邻居中权重最大的节点配对；无权重时选择当前 CSR 邻接表中的第一个未标记邻居。未找到可配对邻居时，节点单独成簇。

原始 `torch_cluster.graclus_cluster` 的 Python 层负责图预处理，底层 `torch.ops.torch_cluster.graclus` 接收 CSR 图并执行聚类。本任务要求在 NPU 版本中复现相同的 Python 预处理和输出语义，验收口径以 PyTorch 层 `graclus_cluster()` 为准。

### torch_cluster 实现现状分析

原版接口如下：

```python
def graclus_cluster(row, col, weight=None, num_nodes=None):
    ...
```

输入输出规格如下：

| 参数 | 参数含义 | 数据类型 | 支持数据类型 | 约束 | 形状 |
| --- | --- | --- | --- | --- | --- |
| row | COO 源节点 | Tensor | int64 | `0 <= row[i] < num_nodes` | `(E,)` |
| col | COO 目标节点 | Tensor | int64 | `0 <= col[i] < num_nodes` | `(E,)` |
| weight | 边权 | Optional Tensor | float16, bfloat16, float32, float64 | 与边一一对应 | `(E,)` |
| num_nodes | 节点数 | Optional int | int | 非负 | 标量 |
| cluster | 输出 cluster ID | Tensor | int64 | 每节点一个 cluster ID | `(num_nodes,)` |

### graclus_cluster 算子功能分析

功能流程为：

1. 若 `num_nodes is None`，由 `max(row.max(), col.max()) + 1` 推断节点数。
2. 去除自环边，即过滤 `row == col` 的边。
3. 无 `weight` 时随机打乱边顺序，使“第一个未标记邻居”的选择与随机边顺序相关。
4. 按 `row` 稳定排序，将 COO 转为 CSR：`rowptr`、`col`、`weight`。
5. 按随机节点顺序执行贪心聚类。
6. 返回 `cluster: LongTensor[num_nodes]`。

# 需求分析（required）

## 需求描述

实现 `ops_gnn.graclus_cluster`，接口与 `torch_cluster.graclus_cluster` 完全一致，满足 Ascend 950PR 上的 PyTorch 层调用和验收要求。

函数签名：

```python
def graclus_cluster(
    row: torch.Tensor,
    col: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    num_nodes: Optional[int] = None,
) -> torch.Tensor:
    ...
```

## 需求拆解

1. 支持 `row` / `col` 为 int64 COO 边索引。
2. 支持 `weight=None` 和有权重两种路径。
3. 支持 `weight` 为 float16、bfloat16、float32；float64 走 L2 CPU 回退语义。
4. `num_nodes` 未指定时按任务书规则推断。
5. Python 层必须去除自环。
6. 无 weight 时必须随机打乱边顺序。
7. 排序后构造 CSR `rowptr`。
8. 固定 RNG seed 时结果可复现。
9. 输出为 int64 cluster，shape 为 `[num_nodes]`。
10. 仅实现前向。

# 详细设计（required）

## 算子分析

### 数学公式

令图为 `G=(V,E)`，每个节点初始未标记。随机排列节点得到 `perm`。对 `perm` 中每个节点 `v`：

```text
if cluster[v] 已标记:
    continue
if weight 不存在:
    cluster[v] = v
    按 CSR 邻接顺序查找第一个未标记邻居 u
    若找到 u: cluster[v] = cluster[u] = min(v, u)
else:
    u* = v
    w_max = 0
    按 CSR 邻接顺序遍历未标记邻居 u
    若 weight(v, u) >= w_max: u* = u, w_max = weight(v, u)
    cluster[v] = cluster[u*] = min(v, u*)
```

说明：cluster 标号与 `torch_cluster` 参考实现保持一致，配对簇使用两端节点编号的较小值 `min(v, u*)`，未配对点标号为自身节点编号。

### 支持数据类型

| 数据 | 类型 | 路径 |
| --- | --- | --- |
| row | int64 | NPU / CPU |
| col | int64 | NPU / CPU |
| rowptr | int64 | Python 预处理生成 |
| weight | float16, bfloat16, float32 | NPU 输入支持 |
| weight | float64 | CPU 回退语义 |
| cluster | int64 | 输出 |

### 支持形状

```text
row:    (E,)
col:    (E,)
weight: (E,) or None
cluster:(num_nodes,)
```

其中 `E >= 0`，`num_nodes >= 0`。

## 算子实现

### 实现方案

当前实现以任务书 PyTorch 层接口为入口，Python 层完成原版 `torch_cluster.graclus_cluster` 对应预处理，L1 路径通过 pybind 调用 Ascend C NPU kernel 完成贪心匹配核心逻辑；float64 按任务书 L2 口径保留 CPU 回退语义。实现由 `python/ops_gnn/graclus_cluster.py`、`csrc/npu/host/graclus_cluster/`、`csrc/npu/kernel/graclus_cluster/` 和 `csrc/pybind.cpp` 组成。

工程新增或修改文件如下：

| 模块 | 新增/修改 | 说明 |
| --- | --- | --- |
| `python/ops_gnn/graclus_cluster.py` | 新增 | Python API、任务书预处理、NPU/CPU 路径分发 |
| `python/ops_gnn/__init__.py` | 修改 | 导出 `graclus_cluster` |
| `csrc/pybind.cpp` | 修改 | 导出 `graclus_cluster_npu` NPU 核心接口 |
| `csrc/npu/host/graclus_cluster/` | 新增 | Host 侧参数校验、输出申请和 kernel launch |
| `csrc/npu/kernel/graclus_cluster/` | 新增 | Ascend C greedy matching kernel |
| `test/test_graclus.py` | 新增 | 基础功能测试 |
| `test/graclus/test_graclus_precision.py` | 新增 | 任务书 TC-01 到 TC-06 精度测试，TC-02 覆盖 float16/bfloat16/float32 |
| `test/graclus/test_graclus_performance.py` | 新增 | 任务书性能 shape 测试 |
| `test/graclus/reference_graclus.py` | 新增 | 独立 CPU golden，不导入被测 `ops_gnn` |
| `test/graclus/graclus_selftest_report.md` | 新增 | 记录精度、性能自测结果 |
| `test/graclus/README.md` | 新增 | 测试步骤和复现说明 |
| `graclus/README.md` | 新增 | 算子目录说明 |
| `README.md` / `docs/zh/api_reference.md` | 修改 | 补充 API 文档 |

### 3.2.1 host侧设计：

Python host 侧完成任务书指定预处理：

1. 参数校验：
   - `row`、`col` 必须是一维 `torch.long` Tensor；
   - `row.numel() == col.numel()`；
   - `weight` 若存在，必须为一维浮点 Tensor 且长度为 `E`；
   - `row`、`col`、`weight` 必须在同一设备。
2. `num_nodes` 推断：
   - 指定时使用指定值；
   - 未指定且无边时为 0；
   - 未指定且有边时为 `max(row.max(), col.max()) + 1`。
3. 自环过滤：
   - 使用 `mask = row != col` 同步过滤 `row`、`col` 和 `weight`。
4. 无 weight 随机边打乱：
   - 使用 `torch.randperm(E)` 生成边随机顺序；
   - 固定 `torch.manual_seed` 时复现。
5. 按 `row` 稳定排序：
   - 使用 `torch.argsort(row, stable=True)`；
   - 同步重排 `col` 和 `weight`。
6. CSR 构造：
   - 使用 `torch.bucketize(torch.arange(num_nodes + 1), row)` 生成 `rowptr`。
7. 调用贪心匹配核心逻辑生成 cluster。

### 3.2.2 kernel侧设计：

NPU 核心接口为 `graclus_cluster_npu(rowptr, col, weight, node_perm, num_nodes, has_weight)`。Python 层将 COO 预处理为 CSR 后，把 `rowptr`、排序后的 `col`、可选 float32 `weight`、固定 seed 下生成的 `node_perm` 传入 Ascend C kernel。

Kernel 采用单 AIV block 顺序执行贪心匹配，原因是 Graclus 的 cluster 标记存在严格先后依赖，并行化会改变 `torch_cluster` 语义。kernel 在 GM 上维护 `cluster` 输出，流程如下：

1. 初始化 `cluster[0:num_nodes] = -1`。
2. 按 `node_perm` 遍历节点。
3. 若节点已标记则跳过。
4. 根据 `rowptr[v]` 与 `rowptr[v+1]` 遍历 CSR 邻接边。
5. 过滤越界邻居和已标记邻居。
6. 无 weight 时选择第一个未标记邻居。
7. 有 weight 时在 NPU kernel 内比较 float32 权重并选择最大权重邻居。
8. 将当前节点和匹配邻居写为同一个 cluster ID，未匹配节点单独成簇。

L1 `float16/bfloat16/float32` 权重输入在 NPU 路径执行，其中 `float16/bfloat16` 在 Python 层通过 NPU Tensor 转换为 float32 后传入 kernel 进行比较；`float64` 按任务书 L2 路径 CPU 回退，不做性能考核。weighted 路径使用带 tensor `_version` 的 CSR cache，重复图输入可复用预处理结果；输入原地修改后 cache 自动失效。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Ascend 950PR | √ |

## 算子约束限制

| 约束项 | 说明 |
| --- | --- |
| row/col dtype | 仅支持 int64 |
| weight dtype | 支持 float16、bfloat16、float32、float64 |
| num_nodes | 必须非负；未指定时由边推断 |
| 自环 | Python 层去除 |
| 随机性 | 固定 `torch.manual_seed` 后可复现 |
| 反向 | 不支持，仅前向 |

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述(不涉及说明原因) | 标准来源 |
| --- | --- | --- |
| 精度标准 | 固定 seed 下 cluster 输出与 CPU 标杆 bit-wise 一致 | 任务书 |
| 性能标准 | 所有性能用例性能不低于 A100 标杆的 0.6 倍 | 任务书 |

CPU golden 计算方式：

```python
torch.manual_seed(seed)
cluster_ref = torch_cluster.graclus_cluster(row.cpu(), col.cpu(), weight_cpu, num_nodes)
```

若测试环境未安装 `torch_cluster`，自测脚本可使用等价 PyTorch CPU reference 作为本地开发验证；正式验收以 CPU 版 `torch_cluster` 为标杆。

## 测试设计

精度测试覆盖以下场景：

| 编号 | 场景 | 说明 |
| --- | --- | --- |
| TC-01 | 小图无 weight | 覆盖随机边 shuffle 和首个未标记邻居选择 |
| TC-02 | 带 L1 weight | 覆盖 float16、bfloat16、float32 最大权重邻居选择 |
| TC-03 | 指定 num_nodes | 覆盖孤立节点 |
| TC-04 | 含自环 | 验证自环被去除 |
| TC-05 | float64 weight | 覆盖 L2 CPU 回退语义 |
| TC-06 | 空边 | 覆盖无边图，每个节点单独成簇 |

本地精度自测结果：`21 passed`。除任务书 TC-01 到 TC-06 外，额外覆盖随机无权图、随机 L1 weighted 图、全自环图、非法索引异常和 CSR cache 原地修改失效回归。测试环境未安装 `torch_cluster`，自测使用独立 CPU reference 生成 golden；reference 不导入 `ops_gnn`，避免自比较。

性能测试覆盖任务书完全图 shape：

| V | E | A100 耗时 | 阈值 A100/0.6 | 本地 median | 结果 |
| --- | --- | --- | --- | --- | --- |
| 4 | 12 | 0.400ms | 0.667ms | 0.463ms | passed |
| 8 | 56 | 0.507ms | 0.845ms | 0.459ms | passed |
| 16 | 240 | 0.705ms | 1.175ms | 0.437ms | passed |
| 32 | 992 | 1.234ms | 2.057ms | 0.448ms | passed |
| 64 | 4032 | 2.825ms | 4.708ms | 0.487ms | passed |

性能阈值按 `A100_ms / 0.6` 换算，测试使用 5 次 warmup、30 次 repeat，取 median latency。本地性能自测结果：`5 passed`。

## 兼容性分析

该算子为新增 Python API，不改变现有 `add_sample` 和 `segment_max_csr` 行为。输入不满足 dtype、shape、device 一致性或索引范围约束时，Python 层直接抛出异常，避免产生错误 cluster。

# 参考资料

- `graclus_cluster_task_doc.md`
- `/workspace/design.md`
- `torch_cluster/graclus.py`
- `csrc/cpu/graclus_cpu.cpp`
- `csrc/cuda/graclus_cuda.cu`
- ops-gnn 仓库：<https://gitcode.com/cann/ops-gnn>
- pytorch_cluster 仓库：<https://github.com/rusty1s/pytorch_cluster>
