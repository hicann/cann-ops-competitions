# 需求背景（required）

## 需求来源

CANN 社区任务 2026 - 8 月社区任务，任务编号 08-1，算子名称 `random_walk`。
参考实现：`torch_cluster.random_walk`（pytorch_cluster，版本 ≥ 1.6.0）。
任务书路径：`docs/202608/random_walk_task_doc.md`。

## 背景介绍

### random_walk 算子功能

在 CSR 格式的图上，从一组起始节点 `start` 出发，采样长度为 `walk_length` 的随机游走序列。支持 node2vec 偏置参数 `p`/`q`：

- 当 `p=1, q=1` 时，退化为**均匀随机游走**（每步从当前节点的邻居中均匀采样一个）。
- 当 `p≠1` 或 `q≠1` 时，采用 **node2vec 拒绝采样**（rejection sampling）实现二阶有偏随机游走。

该算子是图神经网络（GNN）领域 node2vec、DeepWalk 等图表示学习算法的核心基础算子。

### 标杆实现分析

标杆仓库 `pytorch_cluster` 的实现路径与关键文件：

| 文件 | 作用 |
| --- | --- |
| `torch_cluster/rw.py` | Python 层接口，COO→CSR 预处理，调用 `torch.ops.torch_cluster.random_walk` |
| `csrc/rw.cpp` | C++ 注册算子 `torch_cluster::random_walk`，按设备分发 |
| `csrc/cpu/rw_cpu.cpp` | CPU 实现，含 `uniform_sampling` 与 `rejection_sampling` |
| `csrc/cuda/rw_cuda.cu` | CUDA 实现，每 start 一个线程，`curand` 生成随机数 |
| `test/test_rw.py` | 功能测试用例 |

### 算子输入输出

| 输入/输出 | 名称 | 类型 | 形状 | 说明 |
| --- | --- | --- | --- | --- |
| 输入 | `rowptr` | int64 | `[num_nodes+1]` | CSR 行指针（Host 预处理生成） |
| 输入 | `col` | int64 | `[E]` | CSR 列索引（目标节点） |
| 输入 | `start` | int64 | `[S]` | 起始节点序列 |
| 输入 | `walk_length` | int | 标量 | 游走步数，≥ 0 |
| 输入 | `p` | float | 标量 | 返回上一节点惩罚，> 0 |
| 输入 | `q` | float | 标量 | BFS/DFS 插值参数，> 0 |
| 输出 | `node_seq` | int64 | `[S, walk_length+1]` | 游走节点序列 |
| 输出 | `edge_seq` | int64 | `[S, walk_length]` | 游走边索引（无邻居为 -1） |

### Python 层预处理（须复现）

标杆 `rw.py` 在调用底层算子前完成以下预处理：

1. `num_nodes` 默认 `max(row.max(), col.max(), start.max()) + 1`；
2. `coalesced=True` 时，按 `row * num_nodes + col` 排序边；
3. 构造 CSR：`deg.scatter_add_(0, row, ones)` → `rowptr = cumsum(deg)`；
4. 调用 `torch.ops.torch_cluster.random_walk(rowptr, col, start, walk_length, p, q)`。

本算子仅实现底层 `random_walk(rowptr, col, start, walk_length, p, q)`，CSR 构造在 Python 层完成。

# 需求分析（required）

## 需求描述

使用 Ascend C 编程语言（SIMT 编程模型）在 Ascend 950PR 上实现 `random_walk` 算子，满足：

1. 接口与 `torch_cluster.random_walk` 完全一致；
2. 支持均匀随机游走（`p=1, q=1`）与 node2vec 拒绝采样（`p≠1` 或 `q≠1`）两条路径；
3. 全程 int64 计算，无 float64 回退；
4. 含随机性，验收固定 RNG seed，与 CPU 标杆 bit-wise 一致；
5. 性能 ≥ 0.6 倍 GPU（A100）标杆性能。

## 需求拆解

1. **PyTorch 层接口**：复现 `rw.py` 的预处理与封装，注册 `torch.ops.torch_cluster.random_walk`；
2. **底层算子**：实现 `aclnnRandomWalk`，输入 CSR `(rowptr, col, start)`，输出 `(node_seq, edge_seq)`；
3. **均匀采样路径**：`p=1, q=1` 时，每步从 `[rowptr[cur], rowptr[cur+1])` 均匀采样一条边；
4. **拒绝采样路径**：`p≠1` 或 `q≠1` 时，按 node2vec 拒绝采样逻辑实现；
5. **孤立节点处理**：度为 0 时 `edge_seq=-1`，节点序列保持当前节点；
6. **RNG 对齐**：固定 seed 下与 CPU 标杆采样路径一致，保证 bit-wise；
7. **性能优化**：SIMT 线程级并行，每 start 一个线程，充分利用 Ascend 950PR 多核与线程束。

# 详细设计（required）

## 算子分析

### 数学公式

**均匀随机游走**（`p=1, q=1`）：

对第 `n` 条游走的第 `l` 步（`l = 0, ..., walk_length-1`）：

```
row_start = rowptr[n_cur]
row_end   = rowptr[n_cur + 1]
if row_end - row_start == 0:
    e_cur = -1
else:
    idx   = floor(rand[n, l] * (row_end - row_start))   # rand ~ U[0, 1)
    e_cur = row_start + idx
    n_cur = col[e_cur]
node_seq[n, l+1] = n_cur
edge_seq[n, l]   = e_cur
```

**node2vec 拒绝采样**（`p≠1` 或 `q≠1`）：

参考 https://louisabraham.github.io/articles/node2vec-sampling.html ，归一化概率：

```
max_prob = max(1/p, 1, 1/q)
prob_0   = (1/p) / max_prob   # 返回上一节点 t
prob_1   = 1    / max_prob   # 走向 t 的邻居
prob_2   = (1/q) / max_prob   # 走向其他
```

第 0 步同均匀采样；第 `l ≥ 1` 步采用拒绝采样：

```
loop:
    e_cur = row_start + (rand() % (row_end - row_start))
    x     = col[e_cur]
    r     = rand() / RAND_MAX          # U[0, 1)
    if x == t and r < prob_0:      break
    elif is_neighbor(x, t) and r < prob_1:  break
    elif r < prob_2:               break
```

其中 `is_neighbor(x, t)` 通过在 `[rowptr[x], rowptr[x+1])` 区间线性扫描 `col` 判断 `t` 是否为 `x` 的邻居。

### 支持数据类型

| 参数 | 类型 |
| --- | --- |
| `rowptr` / `col` / `start` / `node_seq` / `edge_seq` | int64 |
| `walk_length` | int32/int64（标量） |
| `p` / `q` | float32（标量，kernel 内转 float 计算） |

全程 int64 图操作，不涉及 float64。

### 支持形状

- `rowptr`：一维 `[num_nodes+1]`
- `col`：一维 `[E]`
- `start`：一维 `[S]`
- `node_seq`：二维 `[S, walk_length+1]`
- `edge_seq`：二维 `[S, walk_length]`

## 算子实现

### 实现方案

本算子为**图遍历 + 随机采样**类型，具有以下特征：

1. **线程级并行**：每条游走（每个 `start`）相互独立，天然适合 SIMT 编程模型，每线程负责一条游走；
2. **不规则访存**：CSR 图的邻居区间长度不一，访存不规则，需依赖核内共享内存（DCache）缓存 `rowptr` 与 `col`；
3. **含随机性**：每步需 RNG 采样，均匀路径需 `S × walk_length` 个均匀随机数，拒绝路径需在线生成；
4. **分支密集**：孤立节点、单邻居、拒绝采样循环均含分支，SIMT 下需注意 warp 分支效率。

基于 Ascend 950PR（DAV_3510）的 SIMT 编程模型实现。

#### 3.2.1 host 侧设计

##### 1. Tiling 策略

本算子为 SIMT 线程级并行，Tiling 核心是**线程数规划**与**DCache/UB 空间分配**：

- **核数切分**：`num_starts = S`，按 `GetBlockNum()`（A5 平台物理核数）均分 `start`，每核处理 `S / coreNum` 条游走，余数分到前几个核；
- **线程数设置**：每核线程数 `threadNum = min(S_per_core, MAX_THREADS_PER_BLOCK)`，A5 SIMT 每核最大线程数参考平台规格（典型 1024）；
- **DCache 规划**：`rowptr`（`(num_nodes+1) * 8B`）与 `col`（`E * 8B`）为只读全局输入，通过 DCache 缓存加速不规则访存；`start`、`node_seq`、`edge_seq` 按线程切分；
- **UB 空间**：每线程仅需少量寄存器存储 `n_cur, e_cur, t, v, x, row_start, row_end` 等标量，UB 占用极小。

##### 2. TilingKey 规划

| TilingKey | 条件 | kernel 分支 |
| --- | --- | --- |
| 0 | `p == 1.0 && q == 1.0` | `uniform_sampling_kernel`（均匀采样） |
| 1 | `p != 1.0 \|\| q != 1.0` | `rejection_sampling_kernel`（拒绝采样） |

##### 3. RNG Seed 管理

- Host 侧接收 Python 层传入的 `seed`（由 `torch.manual_seed` 或专用接口设置）；
- Seed 通过 TilingData 或运行时参数传入 kernel，保证固定 seed 下结果可复现；
- 与 CPU 标杆对齐：CPU 均匀路径使用 `torch::rand({numel, walk_length})`（Philox/MT19937），NPU 需采用相同 RNG 算法或等价映射，保证 bit-wise。

##### 4. 输出内存分配

- `node_seq = empty({S, walk_length+1}, int64)`
- `edge_seq = empty({S, walk_length}, int64)`

#### 3.2.2 kernel 侧设计

采用 SIMT 编程模型，每线程处理一条游走。kernel 分两个：`UniformSamplingKernel` 与 `RejectionSamplingKernel`，由 TilingKey 分发。

##### 1. UniformSamplingKernel（TilingKey=0）

```cpp
__global__ void UniformSamplingKernel(
    const int64_t* rowptr, const int64_t* col, const int64_t* start,
    const float* rand,                         // 预生成 [S, walk_length] 均匀随机数
    int64_t* node_seq, int64_t* edge_seq,
    int64_t walk_length, int64_t numel) {
  int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= numel) return;
  int64_t n_cur = start[tid], e_cur, row_start, row_end, idx;
  node_seq[tid * (walk_length + 1)] = n_cur;
  for (int64_t l = 0; l < walk_length; l++) {
    row_start = rowptr[n_cur];
    row_end   = rowptr[n_cur + 1];
    if (row_end - row_start == 0) {
      e_cur = -1;
    } else {
      idx   = (int64_t)(rand[tid * walk_length + l] * (row_end - row_start));
      e_cur = row_start + idx;
      n_cur = col[e_cur];
    }
    node_seq[tid * (walk_length + 1) + l + 1] = n_cur;
    edge_seq[tid * walk_length + l]           = e_cur;
  }
}
```

随机数 `rand` 在 Host 侧通过 `aclnnRng` 或等价接口预生成 `[S, walk_length]` float 张量，保证与 CPU `torch::rand` 序列一致。

##### 2. RejectionSamplingKernel（TilingKey=1）

```cpp
__global__ void RejectionSamplingKernel(
    uint32_t seed,
    const int64_t* rowptr, const int64_t* col, const int64_t* start,
    int64_t* node_seq, int64_t* edge_seq,
    int64_t walk_length, int64_t numel, double p, double q) {
  int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= numel) return;

  // 每线程独立 RNG 状态
  RngState state;
  rng_init(&state, seed, tid);

  double max_prob = fmax(fmax(1.0/p, 1.0), 1.0/q);
  double prob_0 = (1.0/p) / max_prob;
  double prob_1 = 1.0      / max_prob;
  double prob_2 = (1.0/q) / max_prob;

  int64_t t = start[tid], v, x, e_cur, row_start, row_end;
  node_seq[tid * (walk_length + 1)] = t;

  // 第 0 步：均匀采样
  row_start = rowptr[t]; row_end = rowptr[t + 1];
  if (row_end - row_start == 0) { e_cur = -1; v = t; }
  else { e_cur = row_start + (rng_next(&state) % (row_end - row_start)); v = col[e_cur]; }
  node_seq[tid * (walk_length + 1) + 1] = v;
  edge_seq[tid * walk_length]           = e_cur;

  // 第 1..walk_length-1 步：拒绝采样
  for (int64_t l = 1; l < walk_length; l++) {
    row_start = rowptr[v]; row_end = rowptr[v + 1];
    if (row_end - row_start == 0) {
      e_cur = -1; x = v;
    } else if (row_end - row_start == 1) {
      e_cur = row_start; x = col[e_cur];
    } else {
      while (true) {
        e_cur = row_start + (rng_next(&state) % (row_end - row_start));
        x = col[e_cur];
        double r = rng_uniform(&state);          // (0, 1]
        if (x == t && r < prob_0) break;
        if (is_neighbor(rowptr, col, x, t) && r < prob_1) break;
        if (r < prob_2) break;
      }
    }
    node_seq[tid * (walk_length + 1) + l + 1] = x;
    edge_seq[tid * walk_length + l]           = e_cur;
    t = v; v = x;
  }
}
```

`is_neighbor` 通过线性扫描 `[rowptr[x], rowptr[x+1])` 判断 `t` 是否在 `col` 中。

##### 3. RNG 对齐策略

bit-wise 一致的关键是 RNG 序列对齐：

- **均匀路径**：CPU 使用 `torch::rand({numel, walk_length})`（一次性生成全部随机数，按 `[n, l]` 行主序排列）。NPU 侧在 Host 预生成相同分布的 `[S, walk_length]` float 张量传入 kernel，按相同索引顺序消费，保证 `idx = floor(rand * deg)` 一致；
- **拒绝路径**：CPU 使用 `rand()`（C 标准库，非线程安全，全局状态）。NPU 侧每线程独立 RNG 状态（Philox），通过 `seed + tid` 偏移保证并行可复现。由于 CPU `rand()` 全局状态在 `at::parallel_for` 下线程数不确定，**严格 bit-wise 需在 Python 层固定单线程执行 CPU 标杆**，或采用 ATK 双标杆比对。

##### 4. 孤立节点与边界处理

- 度为 0：`e_cur = -1`，`n_cur` 保持不变（与 CPU 一致）；
- `walk_length = 0`：仅输出 `node_seq[:, 0] = start`，`edge_seq` 为空；
- 单邻居（`row_end - row_start == 1`）：拒绝路径直接选该邻居，跳过拒绝循环。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Ascend 950PR（DAV_3510） | √ |

## 算子约束限制

| 约束项 | 说明 |
| --- | --- |
| `rowptr` | CSR 行指针，长度 `num_nodes+1`，非递减，int64 |
| `col` | CSR 列索引，长度 `E`，`0 ≤ col[i] < num_nodes`，int64 |
| `start` | 起始节点，`0 ≤ start[i] < num_nodes`，int64 |
| `walk_length` | `≥ 0` |
| `p` | `> 0` |
| `q` | `> 0` |
| 数据类型 | 仅支持 int64，不支持 float32/float16/bfloat16 |
| 随机性 | 含 RNG，验收固定 seed |
| 设备 | 全程 NPU，不涉及 CPU 回退 |

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | 固定 seed 下 `node_seq`/`edge_seq` 与 CPU 标杆 bit-wise 一致；单标杆不满足时采用 ATK 双标杆，最大相对误差比例 ≤ 2、平均 ≤ 1.2、均方根 ≤ 1.2 | 生态算子开源精度标准（experimental_standard.md） |
| 性能标准 | 所有用例性能 ≥ 0.6 倍 GPU（A100）标杆性能 | 任务书性能要求 |

### 精度验收用例

| 编号 | 场景 | 参考 |
| --- | --- | --- |
| TC-01 | 均匀游走 `p=q=1` | `test/test_rw.py::test_rw_large` |
| TC-02 | node2vec `p≠1` 或 `q≠1` | `test/test_rw.py` 扩展 |
| TC-03 | `return_edge_indices=True` | `test/test_rw.py::test_rw_large_with_edge_indices` |
| TC-04 | `coalesced=False` | 自行构造 |
| TC-05 | 孤立节点（度为 0） | `test/test_rw.py::test_rw_small`（节点 2） |
| TC-06 | `walk_length=0` | 自行构造 |
| TC-07 | 多 start 大规模 | 性能用例 |

### 性能验收用例

固定 seed，随机边 + 随机起点，与 GPU（A100）标杆对比：

| shape（边数, 步数, 起点数） | GPU A100 耗时 | 输出元素 | NPU 目标耗时 |
| --- | --- | --- | --- |
| E=64K, walk=128, start=8K | 0.576ms | 1.0M | ≤ 0.960ms |
| E=256K, walk=128, start=16K | 0.709ms | 2.1M | ≤ 1.182ms |
| E=512K, walk=128, start=32K | 1.139ms | 4.2M | ≤ 1.898ms |
| E=512K, walk=256, start=32K | 1.867ms | 8.4M | ≤ 3.112ms |

### 性能优化要点

1. **SIMT 满线程**：每 start 一线程，尽量打满每核线程束；
2. **DCache 缓存**：`rowptr`、`col` 为只读全局，命中 DCache 加速不规则访存；
3. **随机数预生成**：均匀路径在 Host 预生成 `[S, walk_length]` 随机数一次搬入，避免 kernel 内频繁 RNG；
4. **分支收敛**：孤立节点、单邻居等高频分支单独处理，减少 warp 分歧；
5. **输出布局**：按 `[S, walk_length+1]` 行主序写入，合并写事务。

## 兼容性分析

新算子，不涉及兼容性分析。接口与 `torch_cluster.random_walk`（≥ 1.6.0）完全对齐。

## 修订记录

| 版本 | 日期 | 修改人 | 修改内容 |
| --- | --- | --- | --- |
| 0.1 | 2026-08-04 | JaneCoann | 初版设计文档 |