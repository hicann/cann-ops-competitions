# 需求背景（required）

## 需求来源

2026 年 8 月 CANN 社区任务要求在 Ascend 950 系列上实现与
`torch_cluster` 1.6.0 及以上版本 `knn`、`knn_graph` 接口兼容的 NPU 算子，
验收以 PyTorch 层输出为准。

## 背景介绍

### KNN 算子实现

KNN 用于点云和图神经网络邻域构建。对查询集合 `y[M,F]` 中每个点，在候选
集合 `x[N,F]` 的同 batch 数据中找到 k 个最近邻，返回 int64 边索引。
`knn_graph` 在 KNN 结果上组合自环和消息流方向语义。

参考实现与 API 路径：

- `torch_cluster/knn.py`：接口、batch pointer 和 `knn_graph` 组合逻辑；
- `pytorch_cluster/csrc/cpu/knn_cpu.cpp`：CPU 单标杆；
- `pytorch_cluster/csrc/cuda/knn_cuda.cu`：GPU 暴力扫描与局部 top-k。

### KNN 算子实现现状分析

原 CUDA 实现以一个线程处理一个 query，扫描同 batch 候选并在局部数组内插入
top-k。CPU 版本不支持 cosine，而本任务明确要求 NPU 支持 `cosine=True`。
本设计沿用精确搜索语义，同时不物化 `M*N` 距离矩阵。

| 参数 | 参数含义 | 数据类型 | 支持数据类型 | 约束 | 形状 |
|---|---|---|---|---|---|
| x | 候选点集 | Tensor | float16/float32 | NPU | `[N,F]` |
| y | 查询点集 | Tensor | 与 x 相同 | F/dtype/device 与 x 相同 | `[M,F]` |
| batch_x | x 的 batch | 可选 Tensor | int64 | 非降序 | `[N]` |
| batch_y | y 的 batch | 可选 Tensor | int64 | 非降序 | `[M]` |
| k | 邻居数 | int | 正整数 | 不超过对应 batch 候选数 | 标量 |
| cosine | 距离类型 | bool | true/false | true 使用余弦距离 | 标量 |
| edge_index | 输出边 | Tensor | int64 | NPU | `[2,M*k]` |

计算公式：

```text
euclidean(i,j) = sum_d((x[i,d] - y[j,d])^2)
cosine(i,j) = 1 - dot(x[i],y[j]) / (norm(x[i]) * norm(y[j]))
```

### KNN 算子功能分析

- `knn` 输出 `(row,col)=(y_idx,x_neighbor_idx)`；
- 搜索只能发生在相同 batch 内；
- 一维输入自动转换为 `[-1,1]`；空 x 或 y 返回 `[2,0]`；
- `knn_graph(loop=False)` 搜索 `k+1` 后去除自环；
- `flow` 支持 `source_to_target` 与 `target_to_source`；
- 距离平局时边顺序可不同，但邻居集合必须与标杆一致。

# 需求分析（required）

## 需求描述

使用 Ascend C 为 Ascend 950 实现精确 KNN。L1 float16/float32 全部走 NPU
Kernel，欧氏和余弦距离均须支持；Python 层完整复现 `torch_cluster.knn` 和
`knn_graph` 的公开签名、batch pointer 构造、空输入及 graph 组合逻辑。

## 需求拆解

1. 接入 `ops-gnn` 的 Python、pybind、Host、Kernel 四层结构。
2. 支持 float16/float32、1D/2D、batch=None 和 sorted multi-batch。
3. 单 Kernel 融合距离计算和 top-k，输出 int64 NPU Tensor。
4. 实现 `cosine=True`；零范数采用零相似度，避免 NaN 污染排序。
5. 实现 `knn_graph` 的 loop/flow 语义。
6. 覆盖任务书 TC-01～TC-10，按单标杆比较每个 query 的邻居集合。
7. 覆盖任务书 32 组 shape/k 和两种 dtype，性能比不低于 0.45。
8. 提供算子 README、自测说明和自测报告底稿。

# 详细设计（required）

## 算子分析

### 数学公式

欧氏模式按平方距离排序，省略不影响次序的平方根。余弦模式按
`1-cosine_similarity` 升序。稳定比较键为 `(distance,index)`，距离相同时较小
候选下标优先；验收仍按集合比较，不依赖平局顺序。

### 支持数据类型

| 层级 | x/y | batch/输出 | 计算类型 |
|---|---|---|---|
| L1 NPU | float16、float32 | int64 | 距离使用 float32 累计 |

### 支持形状

- `x:[N,F]`、`y:[M,F]`，N/M/F 为运行时动态值；
- 1D 输入在 Python 层转换为 `[N,1]` / `[M,1]`；
- 输出固定为 `[2,M*k]`；
- 空输入输出 `[2,0]` 且不启动 Kernel。

## 算子实现

### 实现方案

```text
knn/knn_graph
  -> Python：空输入、1D reshape、contiguous、batch_size/ptr 构造
  -> torch.ops.torch_cluster.knn：PrivateUse1/NPU dispatcher
  -> pybind：knn_npu（传递当前 NPU stream）
  -> Host：设备/shape/dtype/k/ptr 校验，输出/工作区/Tiling 申请
  -> Ascend C SIMT：query 分核，batch 内扫描，稳定 top-k
  -> [2,M*k] int64 NPU Tensor
  -> knn_graph：按 flow 交换行，loop=False 时过滤 row==col
```

#### 3.2.1 host 侧设计

Python 严格保留任务书参数顺序和默认值。`batch_size>1` 时使用
`torch.bucketize(torch.arange(batch_size+1), batch)` 构造 `ptr_x/ptr_y`，随后按
任务书调用 `torch.ops.torch_cluster.knn`。扩展只为上游算子注册
`PrivateUse1` 实现，CPU 实现继续由官方 `torch-cluster` 提供。

Host 校验同设备、二维、同特征维、同 dtype、支持 dtype、`k>0`；batch pointer
必须 int64 一维、首值 0、尾值等于数据长度、单调非降且不越界。每个含 query 的
batch 候选数必须不少于 k，防止非法 GM 输出。

Tiling 结构为：

```cpp
struct KnnTilingData {
    int64_t n;
    int64_t m;
    int64_t feature_dim;
    int64_t batch_size;
    int64_t k;
    int64_t cosine;
};
```

Host 将 Tiling 作为 uint8 Tensor 上传 GM，并传入当前 PyTorch NPU stream。
返回前仅同步当前 stream，保证 GM Tiling/大 k 临时区生命周期并及时暴露设备
错误，不执行 device-wide synchronize。

输出申请为 `[2,M*k]` 连续 Tensor，两行通过基地址和 `M*k` 偏移传给 Kernel。
`k<=64` 无额外工作区；更大 k 申请 `M*k` 个 float32 距离作为通用回退工作区。

#### 3.2.2 kernel 侧设计

Kernel 使用 Ascend C SIMT `VF_CALL`。所有 AIV core 均分连续 query 区间，每个
线程处理一个 query。batch 模式通过 `ptr_y` 二分定位 query batch，再从 `ptr_x`
读取候选范围，保证不跨 batch。

欧氏路径在一次 feature 循环累计平方差；余弦路径在同一循环累计 dot、x norm
和 y norm。FP16 先转 FP32 累计。每产生一个距离即插入当前 top-k，不产生中间
距离矩阵。

top-k 分两条路径：

1. `k<=64`：每线程使用固定长局部距离/下标数组，按 `(distance,index)` 稳定插入，
   搜索后一次写回 GM；覆盖任务书全部性能用例（最大 k=32）及 graph 的 k+1。
2. `k>64`：距离使用 GM 工作区，下标复用输出列执行相同插入，保证接口不人为
   限制 k；该路径以功能泛化为主，不参与任务书性能验收。

两条路径均显式初始化为 `FLT_MAX/-1`。Host 的候选数校验保证合法输入填满 k 个
位置。

#### 3.2.3 knn_graph 组合设计

`loop=True` 调用 `knn(...,k)`；`loop=False` 调用 `knn(...,k+1)` 后删除
`row==col`。`source_to_target` 把 `(query,neighbor)` 交换为 `(neighbor,query)`；
`target_to_source` 保持原顺序，与 `torch_cluster/knn.py` 一致。

#### 3.2.4 分核和内存优化策略

- 分核：`queries_per_block=ceil(M/AIV_core_num)`，线程在本核区间按
  `thread_num` 步进，保证全部 AIV core 参与；
- 内存：不创建 `M*N` 距离矩阵；常用路径仅有 `O(M*k)` 输出和每线程 `O(k)`
  局部数组；
- 复杂度：时间 `O(M*N_b*(F+k))`，其中 `N_b` 为 query batch 的候选数；
- 融合：距离、选择、输出在单 Kernel 完成，无压缩 Kernel 和 Host 数据回读；
- 泛化：大 k 路径增加 `O(M*k)` float32 GM 工作区。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
|---|---|
| Ascend 950 系列（`dav-3510` / `arch35`） | √ |

## 算子约束限制

- x/y 必须位于同一 NPU，dtype 和特征维一致；
- 仅支持 float16、float32；
- batch_x/batch_y 必须为非降序 int64 向量；
- `k>0` 且不超过每个含 query 的 batch 候选数；
- `flow` 仅允许 `source_to_target`、`target_to_source`；
- `num_workers` 保留但在 NPU 路径忽略；
- 仅返回离散索引，不提供反向梯度。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述（不涉及说明原因） | 标准来源 |
|---|---|---|
| 精度标准 | FP16/FP32 每个 query 的 k 邻居集合与 CPU `torch_cluster` 单标杆一致；平局允许顺序不同；cosine、graph loop/flow 必测 | 任务书与生态算子开源精度标准 |
| 功能标准 | 覆盖 TC-01～TC-10，shape/dtype/device、batch 隔离与异常符合约束 | KNN 社区任务书 |
| 性能标准 | 任务书 32 组参数、float16/float32 的性能比均不低于 0.45 | KNN 社区任务书 |
| 内存自验证标准 | 与性能矩阵相同的 64 条用例逐项测量增量峰值；不超过理论输出基线加 2 MiB，释放输出后 allocated memory 残留为 0 | 项目交付自验证门槛（任务书未提供官方内存数值门槛） |

```bash
NPU_DEVICE_ID=0 python3 -m pytest test/knn/test_knn.py -v
NPU_DEVICE_ID=0 python3 test/knn/benchmark_knn.py --warmup 20 --repeat 100
NPU_DEVICE_ID=0 python3 test/knn/memory_knn.py --output logs/memory_knn.csv
```

功能测试覆盖基础 2D、1D、k=1、多 batch、cosine、graph loop/flow、空输入、
graph+batch、大 shape、非连续输入和非法参数。性能脚本使用 NPU Event 计时，
输出延迟、任务书标杆、性能比和 PASS/FAIL，低于门槛时非零退出。
内存脚本使用 NPU allocator 统计输入准备后的增量峰值，对照 `[2,N*k]` int64
输出的理论最低存储量，逐项检查额外占用和释放后残留，失败时非零退出。

## 兼容性分析

新增 `ops_gnn.knn` / `ops_gnn.knn_graph`、`torch_cluster::knn` 的
PrivateUse1 实现和底层 `_pybind.knn`，不修改既有签名或 CPU dispatcher。
CMake 的递归规则自动收集 `csrc/npu/knn/op_host/*.cpp` 与
`op_kernel/arch35/*.cpp`。Python 参数顺序、默认值、batch pointer、loop 和 flow
与 `torch_cluster` 对齐；NPU 额外实现 CPU C++ 路径不支持的 cosine。

后续可为 3D 欧氏场景加入空间网格剪枝，并将大 k 回退升级为分块局部 top-k 与
多级归并；这些优化不改变公开接口和输出集合语义。
