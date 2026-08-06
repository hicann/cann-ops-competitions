# radius / radius_graph 算子设计文档（昇腾 NPU）

> 8月社区任务 · radius 算子开发 · 适配硬件：Ascend 950PR
> 对齐目标：`torch_cluster.radius`、`torch_cluster.radius_graph`（https://github.com/rusty1s/pytorch_cluster ，版本 ≥ 1.6.0）

---

# 需求背景（required）

## 需求来源

本任务来源于昇腾 CANN 生态社区任务《8月社区任务 - radius 算子开发任务书》，要求在 `ops-gnn` 开源仓（https://gitcode.com/cann/ops-gnn ）中以 Ascend C 编程语言实现与 `torch_cluster.radius` / `torch_cluster.radius_graph` **接口完全一致**的 NPU 版本算子，开发、测试硬件为 **Ascend 950PR**，CANN 版本为算子开源仓指定版本。

## 背景介绍

### 任务概述

`radius` 是图神经网络（GNN）中最常用的邻域查询原语之一：对查询点集 `y` 中的每一点，找出参考点集 `x` 中欧氏距离 **< r** 的所有邻居点（每点最多 `max_num_neighbors` 个），输出边索引 `edge_index [2, E]`。`radius_graph` 是 `radius(x, x, ...)` 的自环特例，用于 PointNet++、DGCNN 等点云/几何 GNN 模型的自监督建图。

该算子天然是**访存密集 + 分支判断**型算子：计算量随点数平方级增长（最坏 O(M·N) 次距离评估），无法用常规的张量算子（矩阵乘、逐元素）直接表达，必须依赖**并行度逐查询点**的 SIMT/标量多线程编程模型实现。

### torch_cluster 参考实现分析

| 实现 | 路径 | 核心算法 |
|------|------|----------|
| Python 宿主 | `torch_cluster/radius.py` | 空输入早退；1D→2D view；构造 `ptr_x`/`ptr_y`（batch > 1 时用 `bucketize`）；调用自定义算子 `torch.ops.torch_cluster.radius`；`radius_graph` 复用 `radius` 并按 `flow` 重组行列 |
| CPU | `csrc/cpu/radius_cpu.cpp` | 基于 nanoflann KD-tree 的 `radiusSearch`（`sorted=false`），逐 query 取回 `max_num_neighbors` 个命中；同一 batch 内搜索 |
| GPU | `csrc/cuda/radius_cuda.cu` | 每查询点一个 CUDA 线程（block=256）；线程遍历同一 example（`ptr_x[example]..ptr_x[example+1]`）内全部 x 点，逐 F 维累加平方距离 `dist < r*r` 判定；命中写入 `[M × max_num_neighbors]` 大小的 row/col 缓冲；kernel 后 `masked_select` 压缩并 `stack` 为 `[2, E]` |

关键语义（已对照参考源码与官方 `test/test_radius.py` 逐一核对）：

1. **输出布局**：`edge_index[0] = y 中查询点索引`，`edge_index[1] = x 中邻居索引`（官方用例断言集合如 `{(0,0),(0,1),(0,2),(0,3),(1,1),(1,2),(1,5),(1,6)}` 可验证）。
2. **距离判定**：`Σ_d (x[j,d]-y[i,d])² < r·r`（严格小于；直接比较平方距离，避免开方）。
3. **batch 语义**：仅搜索同一 example 内的 x（`ptr_x[example]..ptr_x[example+1]`）；batch 须 sorted。
4. **超限处理**：命中数超过 `max_num_neighbors` 时截断/随机采样（详见 3.1.4 节），允许非确定性。
5. **`ignore_same_index`**：为 `True` 时跳过 `x` 索引与 `y` 索引相同的点对；`radius_graph` 传 `not loop`。
6. **空输入**：直接返回 `empty(2, 0, dtype=long)`，不进入 kernel。

### 昇腾 NPU 现状与差距

昇腾 Ascend 950PR 提供 AI Vector 核（AIV），CANN 支持两类编程范式：

- **Vector 范式（Ascend C 矢量编程）**：基于 UB 的矢量指令流水（`DataCopy`/`Add`/`Mul` 等），适合规整的稠密张量计算；
- **SIMT 范式（AscendC::Simt）**：AIV 核上的标量多线程执行模型，提供 `GetThreadIdx`/`GetThreadNum`/`VF_CALL` 等接口（`ops-gnn` 仓内 `add_sample` 算子已示范 `__simt_vf__` + `VF_CALL` 用法），适合**逐元素分支密集型**算法。

`radius` 的逐查询点独立搜索、动态邻居数、随机采样等特性天然适配 **SIMT 范式**，这也是本设计的核心编程模型。

---

# 需求分析（required）

## 需求描述

实现与 `torch_cluster.radius` / `torch_cluster.radius_graph`（版本 ≥ 1.6.0）接口完全一致的 NPU 版本：

- 对 `y` 中每个查询点，找出 `x` 中欧氏距离 **< r** 的所有邻居（上限 `max_num_neighbors`）；
- 超出上限时随机采样邻居（与 GPU 语义一致，允许非确定性）；
- 输出 `edge_index [2, E]`；
- 验收口径：**PyTorch 层为准**，须覆盖 `radius` + `radius_graph`。

## 需求拆解

1. **PyTorch 层接口对齐**：`radius()` / `radius_graph()` 函数签名、默认值、host 逻辑（构造 `ptr_x`/`ptr_y` 后调用 `torch.ops.torch_cluster.radius`）与参考完全一致，支持 `torch.jit.script`。
2. **NPU 算子实现**：Ascend C（SIMT 范式）实现搜索核，支持 L1 数据类型 float16 / bfloat16 / float32。
3. **功能语义对齐**：欧氏距离判定、batch 内搜索、`ignore_same_index`/`loop`、超限随机采样（集合等价）、空输入早退。
4. **精度达标**：严格满足《生态算子开源精度标准》，以 CPU 版 torch_cluster 为标杆，采用 AscendOpTest 测试。
5. **性能达标**：所有用例 ≥ 0.6 倍 A100 标杆耗时（验收取最优实现：grid/hash cell、SIMT brute-force 等）。
6. **交付件完整**：算子设计文档（本文件）、自测用例及测试代码（含 README 复现说明）、自测报告、待验收代码仓库（含算子 README）。

---

# 详细设计（required）

## 算子分析

### 数学公式

对查询点 `y_i ∈ R^F`（i = 0..M-1），其 batch 为 `b = batch_y[i]`：

```
edge_index 中所有满足下式的点对 (i, j)：
    j ∈ [ptr_x[b], ptr_x[b+1])
    ‖x_j - y_i‖² = Σ_{d=0}^{F-1} (x[j,d] - y[i,d])²  <  r²
    且 (ignore_same_index == false 或 j != i)
```

每查询点邻居数上限 `max_num_neighbors`，超出时随机采样；输出 `edge_index[0]=i`（y 索引）、`edge_index[1]=j`（x 索引）。

### 支持数据类型

| 级别 | dtype | 实现路径 | 说明 |
|------|-------|----------|------|
| L1 | float16 / bfloat16 / float32 | NPU 核内计算（距离累加用 fp32 中间精度，见 3.2.5） | 主验收路径 |
| L2 | float64 | CPU 回退（Python 层转 CPU 调参考实现，结果回拷 NPU） | 不做性能考核；bit-wise 一致 |

### 支持形状与参数约束

| 参数 | 类型 | 约束 |
|------|------|------|
| `x` | Float `[N, F]` | L1：fp16/bf16/fp32；L2：fp64（CPU 回退）。`F ≥ 1`，2D（1D 由 Python 层 `view(-1,1)`） |
| `y` | Float `[M, F]` | 与 x 同 dtype、同 F |
| `r` | float | `> 0`，Python float 标量（非 Tensor） |
| `batch_x` / `batch_y` | Optional Long | 须 sorted；长度分别等于 N / M；`batch_size > 1` 时二者都须提供 |
| `max_num_neighbors` | int | 默认 32；≥ 1；每 y 点最多返回邻居数 |
| `num_workers` | int | NPU 忽略（接口保留，不参与调度） |
| `ignore_same_index` | bool | True 时跳过 `j == i` 点对（`radius_graph` 用 `not loop`） |
| `batch_size` | Optional[int] | 缺省自动计算 |

**空输入**：`x.numel()==0` 或 `y.numel()==0` 时返回 `torch.empty(2, 0, dtype=torch.long, device=x.device)`。

### 功能语义（与 torch_cluster 对齐的关键点）

#### 输出布局

```
edge_index = [ [i0, i1, ...],      ← 查询点 y 索引（row）
               [j0, j1, ...] ]     ← 邻居 x 索引（col）
```

`radius_graph` 按 `flow` 重组：`source_to_target` 时 `row, col = edge_index[1], edge_index[0]`；`target_to_source` 时保持原序。

#### batch 语义

同 batch 内搜索：查询点 `y_i` 只与 `[ptr_x[b], ptr_x[b+1])` 范围内的 x 点比较距离；`batch_x`/`batch_y` 须 sorted；允许跳过空 batch（`x_start==x_end` 或 `y_start==y_end` 的 batch 不产生边）。

#### ignore_same_index / loop

`ignore_same_index=True` 时跳过 `j == i`（x 绝对索引与 y 绝对索引相等）的点对。`radius_graph(x, r, loop=False)` 等价于 `radius(x, x, r, ..., ignore_same_index=not loop)` 并按 flow 重组。

#### 超限邻居采样语义

- 命中数 ≤ `max_num_neighbors`：全部返回（按 x 扫描序写入，顺序确定）。
- 命中数 > `max_num_neighbors`：**随机采样**（与 GPU 一致，非确定性）。设计为**蓄水池采样（Reservoir Sampling）**：
  - 前 `max_num_neighbors` 个命中直接写入槽位；
  - 第 k（k > max）个命中以概率 `max_num_neighbors / k` 随机替换已有槽位中的一个（均匀）；
  - 每个查询点独立 PRNG 状态，种子由 host 传入（默认派生自 `torch.manual_seed`，便于固定 seed 复现）。
- 验收判据为**集合等价**（超限场景不要求 edge 顺序一致；固定 seed 下要求 bit-wise 一致）。
- 说明：实现层通过 tiling 标志 `useReservoir` 可切换为"确定性截断"（与当前 master 分支 `break` 语义一致），默认按任务书要求启用随机采样；最终以验收环境实际安装参考版本的采样语义为对齐基准，该开关保证两种语义均可复现。

#### 空输入

Python 层早退，返回 `[2, 0]` LongTensor；不进入 NPU kernel。

## 算子实现

### 总体架构

#### 代码组织结构

```
ops-gnn/
├── python/ops_gnn/
│   ├── radius.py                      # Python 层 radius()/radius_graph()（host 逻辑与参考一致）
│   └── __init__.py                    # 导出 radius / radius_graph
├── csrc/
│   ├── pybind.cpp                     # 扩展绑定入口
│   └── npu/
│       ├── host/radius/
│       │   ├── radius.cpp             # Host 入口：校验、tiling、内存规划、Launch、压缩组装
│       │   ├── radius.h
│       │   └── radius_tiling.h        # TilingData 结构
│       └── kernel/radius/
│           ├── radius_kernel.cpp      # SIMT 搜索核 + 网格构建核 + 压缩核（kernel 侧）
│           └── radius_kernel.h
└── test/
    ├── test_radius.py                 # 功能/精度用例（TC-01~TC-07）
    └── test_radius_perf.py            # 性能用例
```

#### 算子注册与调度

复用 `torch_cluster` 的算子注册名，实现**零侵入替换**：

```cpp
// csrc/npu/host/radius/radius.cpp
TORCH_LIBRARY(torch_cluster, m) {
    m.def("radius(Tensor x, Tensor y, Tensor? ptr_x, Tensor? ptr_y, float r,"
          " int max_num_neighbors, int num_workers, bool ignore_same_index) -> Tensor");
}
TORCH_LIBRARY_IMPL(torch_cluster, PrivateUse1, m) {
    m.impl("radius", radius_npu);   // NPU 设备分发
}
```

- 当 `x`/`y` 位于 NPU（PrivateUse1，即 torch_npu 注册的 `npu` 设备）时，`torch_cluster.radius` 内部对 `torch.ops.torch_cluster.radius` 的调用自动分发到 `radius_npu`，PyTorch 层行为与参考完全一致；
- `ops_gnn.radius` / `ops_gnn.radius_graph` 提供与参考 `radius.py` 逐行一致的 host 逻辑（空输入早退、view、contiguous、batch_size 计算、`bucketize` 构造 ptr），保证不依赖 torch_cluster 包也可单独使用；
- 保持 `torch.jit.script` 兼容（Python 层不使用不可脚本化语法）。

#### 算法模式选择

依据 `F`（维度）、点数与半径相对密度，host 侧在 tiling 阶段二选一：

| 模式 | 适用场景 | 原理 |
|------|----------|------|
| **Tier 1：SIMT 暴力搜索** | `F > 4`、点云稀疏、`r` 与点云尺寸可比（网格剪枝失效） | 每查询点一线程，遍历同 batch 全部 x 点 |
| **Tier 2：均匀网格加速** | `F ∈ {2, 3}`（对标性能基准 3D 场景） | 空间网格（边长 = r）剪枝，只评估相邻 `3^F` 个 cell 内候选 |

选择启发式（host 预估两路径工作量取小者，`avg_cell_pts = N_b·(r^F)/vol_b`）：

```
est_brute  = M · N_b
est_grid   = M · min(3^F · avg_cell_pts, N_b)
```

`r` 较大或 `avg_cell_pts` 接近 `N_b` 时自动回退 Tier 1。两模式共用同一搜索/采样/压缩逻辑，仅候选集来源不同。

### host 侧设计

#### 入口流程

```
radius_npu(x, y, ptr_x, ptr_y, r, max_num_neighbors, num_workers, ignore_same_index)
  ├─ 1. 校验：device(PrivateUse1)/contiguous/2D/F 一致/dtype 合法/r>0
  ├─ 2. 空输入早退（x.numel()==0 || y.numel()==0 → [2,0] long）
  ├─ 3. L2 路径：dtype==float64 → Python 层已回退 CPU（host 断言兜底）
  ├─ 4. 构造 ptr（None → [0, N]/[0, M]；否则取传入 ptr，校验 sorted、numel 一致）
  ├─ 5. 计算 tiling：模式选择、分核、tile 大小、grid 参数、PRNG seed
  ├─ 6. 分配设备中间缓冲（见内存规划）
  ├─ 7. 按需启动"网格构建流水"（Tier 2）：
  │      cellID+直方图核 → 前缀和核 → x 按 cell 重排核
  ├─ 8. 启动"SIMT 搜索核"（计数 + 写入候选 + 蓄水池采样）
  ├─ 9. 启动"压缩核"（count 前缀和 → 按偏移 scatter → edge_index）
  └─ 10. 释放临时缓冲，返回 edge_index [2, E]
```

#### 参数校验（host）

| 校验项 | 规则 |
|--------|------|
| 设备 | `x.device.type == 'npu'`（PrivateUse1） |
| 连续 | `x.is_contiguous()` / `y.is_contiguous()`（Python 层已 `.contiguous()`） |
| 维度 | `x.dim()==2 && y.dim()==2 && x.size(1)==y.size(1)` |
| dtype | L1: fp16/bf16/fp32；L2: fp64（应已被 Python 层回退拦截） |
| 半径 | `r > 0`（NaN/负值 TORCH_CHECK 报错） |
| ptr | `ptr_x.numel()==ptr_y.numel()`；单调不减；首元素 0、尾元素 N/M |
| 邻居上限 | `max_num_neighbors ≥ 1` |

#### tiling 策略

1. **分核**：`coreNum = min(GetCoreNumAiv(), M)`（AIV 满核优先）；查询点按 batch 边界对齐均分到各核，避免跨 batch 拆分（保证核内 batch 局部性）；`yPerCore = ceil(M/coreNum)`，尾核 `yTail = M % coreNum`。
2. **块内 tile**：每核每次处理 `queriesPerBlock` 个查询点（UB 容量约束），x 候选按 `xTileSize` 分块流入（双缓冲）。
3. **模式参数**（Tier 2）：cell 边长 = `r`，各维 extents = `ceil((coord_max - coord_min) / r) + 1`，`invCellSize = 1/r`；网格按 **batch 独立**构建（每个 batch 一个网格，cell 索引自 batch 内局部坐标计算，天然满足同 batch 搜索约束）。
4. **PRNG seed**：`seed = host 从 c10 RNG 状态派生（torch.manual_seed 可复现）`；每查询点状态 = `hash(seed, i, b)`。
5. **tilingkey**：`useGrid`（模式）、`useReservoir`（采样模式）、`ignoreSameIndex` 等布尔位合并为 tilingkey，kernel 侧按位分支，避免重复实例化核。

#### 内存规划

| 缓冲 | 大小 | 说明 |
|------|------|------|
| `candRow` / `candCol` | `M × max_num_neighbors` × int64 | 每查询点候选邻居（x 索引）缓冲 |
| `counts` | `M` × int64 | 每查询点命中总数（≤ max 时即有效邻居数） |
| `offsets` | `M+1` × int64 | counts 前缀和（压缩阶段复用） |
| `edge_index` | `2 × E` × int64 | 最终输出 |
| Tier2: `cellIds`(N) / `cellOffsets` / `perm`(N) | 各 N 级 | 网格构建临时 |
| UB 内 | `queriesPerBlock×F` + `xTileSize×F`（双缓冲） | 数据搬运粒度 |

内存峰值约 `M·max_num_neighbors·16B`（候选缓冲），M=64K、max=32 时约 32MB，可接受；与 CUDA 参考（同规模 row/col 缓冲 + mask）量级一致。

#### TilingData 定义

```cpp
// csrc/npu/kernel/radius/radius_tiling.h
struct RadiusTilingData {
    uint32_t numX;             // N
    uint32_t numY;             // M
    uint32_t dim;              // F
    uint32_t numExamples;      // ptr 段数
    float    radiusSquared;    // r²（fp32）
    uint32_t maxNumNeighbors;  // 每查询点上限
    uint32_t ignoreSameIndex;  // 0/1
    uint32_t useGrid;          // 0: Tier1 暴力; 1: Tier2 网格
    uint32_t useReservoir;     // 0: 确定性截断; 1: 蓄水池采样
    uint32_t coreNum;          // 分核数
    uint32_t queriesPerCore;   // 每核查询点数
    uint32_t queriesTail;      // 尾核余量
    uint32_t queriesPerBlock;  // UB 内查询点批大小
    uint32_t xTileSize;        // x 候选分块大小
    uint32_t seed;             // PRNG 种子
    // Tier 2（useGrid=1）：
    int32_t  gridExtents[3];   // 各维 cell 数
    float    invCellSize;      // 1/r
    uint32_t cellsPerBatch;    // 单 batch 网格 cell 总数
    // GM 指针（运行时填充）：
    __gm__ uint8_t* x;         // 原始或按 cell 重排后的 x
    __gm__ uint8_t* y;
    __gm__ int64_t* ptrX;
    __gm__ int64_t* ptrY;
    __gm__ int64_t* cellOffsets; // [B × cellsPerBatch + 1]（Tier2）
    __gm__ int64_t* counts;
    __gm__ int64_t* candRow;
    __gm__ int64_t* candCol;
};
```

### kernel 侧设计

#### SIMT 编程模型说明

参考 `ops-gnn/csrc/npu/kernel/add_sample/add_sample_kernel.cpp` 的既有范式：

```cpp
__simt_vf__ __aicore__ LAUNCH_BOUND(1024) void RadiusSearchKernelSimt(...) {
    uint32_t blockIdx  = AscendC::Simt::GetBlockIdx();
    uint32_t blockNum  = AscendC::Simt::GetBlockNum();
    uint32_t threadIdx = AscendC::Simt::GetThreadIdx();
    uint32_t threadNum = AscendC::Simt::GetThreadNum();
    ...
}
__attribute__((aiv)) __global__ __aicore__ void RadiusSearchKernel(...) {
    AscendC::Simt::VF_CALL<RadiusSearchKernelSimt>(AscendC::Simt::Dim3{THREADS}, ...);
}
// host: RadiusSearchKernel<<<coreNum, nullptr, stream>>>(...)
```

线程映射：**每个 SIMT 线程处理一个查询点** `i = blockIdx*threadNum + threadIdx`（核内连续区间），与 CUDA 参考的"一 query 一线程"完全同构，可无缝迁移搜索逻辑。

#### 方案一：SIMT 暴力搜索核（Tier 1）

```text
RadiusSearchKernelSimt(x, y, ptrX, ptrY, r2, maxK, ignoreSame, counts, candRow, candCol, tiling)
  i = 全局线程号映射的查询点
  if i >= M: return
  b = 由 ptrY 二分/顺序定位 i 所属 example
  yPtr = y + i*F          // 查询点坐标
  count = 0
  prng  = InitPrng(hash(seed, i, b))          // 仅 useReservoir 时启用
  for j in [ptrX[b], ptrX[b+1]):
      xPtr = x + j*F
      dist2 = 0
      for d in 0..F-1:                        // 展开 + 向量化（fp32 累加）
          diff  = (fp32)xPtr[d] - (fp32)yPtr[d]
          dist2 += diff * diff
      if dist2 < r2 and !(ignoreSame && j == i):
          if count < maxK:
              candRow[count] = i;  candCol[count] = j;  count++
          else if useReservoir:
              k = count + 1
              if PrngFloat(prng) < (float)maxK / k:      // 概率 maxK/k 替换
                  s = PrngInt(prng) % maxK               // 均匀选槽位
                  candRow[s] = i;  candCol[s] = j
              count++
  counts[i] = count                          // 供压缩阶段使用
```

复杂度 O(M·N_b) 距离评估；每评估仅 3 次减法、3 次乘加 + 1 次比较（F=3），分支简单，SIMT 吞吐高。

#### 方案二：均匀网格加速（Tier 2，grid/hash cell）

三小核 + 搜索核流水（均按 batch 独立处理）：

```
① CellBuildKernel    对 x：cell = 由局部坐标 floor((coord - min) * invCellSize) 编码为
                     cellId（多维 → 一维，stride 乘积）；对每 batch 做 cell 计数（直方图）
② CellOffsetKernel   每 batch 内对 cell 计数做前缀和 → cellOffsets，并求总点数
③ CellReorderKernel  按 (batch, cellId) 稳定重排 x 坐标 → xSorted（perm 数组写入 candRow 暂存）
④ RadiusSearchKernel 对查询点 i：cell_q = 由 y_i 局部坐标计算
                      for 相邻 3^F 个 cell（含自身，坐标为 cell_q ± 1）：
                          for j in [cellOffsets[b][c], cellOffsets[b][c+1]):
                              恢复原始 x 索引（perm）→ 距离判定（同 Tier 1）
```

- 候选由 `N_b` 降至 `≤ 3^F × avg_cell_pts`，3D 场景约减 1~2 个数量级；
- 距离判定保持**严格 `< r²`**（cell 边长 = r 保证不遗漏 `dist < r` 的点，相邻 cell 覆盖足够）；
- `perm` 同时用于输出正确性：edge_index 中邻居索引必须为**原始 x 索引**。

#### 邻居写入与随机采样（蓄水池）

见 3.1.4 节语义。实现要点：

- 候选缓冲按查询点定长（`maxK`），超限替换只发生在蓄水池模式；
- PRNG 使用轻量级确定性算法（如 SplitMix32 / LCG，32 位状态即可），生成 `[0,1)` 浮点与 `[0,maxK)` 整数两个例程；
- 采样决策与 GPU 一致为**非确定性**；固定 seed 下每个查询点序列确定，可满足"固定 seed 下 bit-wise 复现"验收。

#### 压缩与输出组装

```
CompactKernel(counts, candRow, candCol, offsets, edgeIndex, tiling)
  ① 对 counts 求前缀和（SIMT 并行：块内扫 + 块间偏移链）→ offsets，E = offsets[M]
  ② 每查询点 i 的有效条数 k = min(counts[i], maxK)
     目标写入区间 [offsets[i-1], offsets[i-1] + k)
     逐条 edgeIndex[0][t] = candRow[i*maxK + s]
               edgeIndex[1][t] = candCol[i*maxK + s]   // 原始 x 索引
```

输出天然为 `[2, E]` 紧凑布局，无需 host 侧 `masked_select`（避免 torch 图外算子引入的性能与内存开销）。`radius_graph` 的行列重组在 Python 层完成（`torch.stack`）。

#### float64（L2）路径

Python 层检测 `x.dtype == torch.float64`：将 `x`/`y`/`batch` 拷回 CPU，调用参考 CPU 实现（或本仓自带的 CPU 兜底实现）得到 `edge_index`，再拷回 NPU 设备。结果与 CPU 标杆 **bit-wise 一致**；不做性能考核。替代方案（NPU 低精度拼接：fp64 高/低位拆分为两个 fp32 计算）仅作为 README 文档化备选，默认不启用。

### 性能优化策略

对标 A100 基准（fp32/fp16，3D 点云建图）并给出达标分析：

| 场景 (点数, r) | A100 fp32 基准 | NPU 目标（≤ 基准/0.6） |
|----------------|----------------|-------------------------|
| N=8K,  r=0.8 | 2.118 ms | 3.53 ms |
| N=16K, r=0.5 | 3.916 ms | 6.53 ms |
| N=32K, r=0.3 | 8.446 ms | 14.08 ms |
| N=64K, r=0.2 | 20.830 ms | 34.72 ms |
| N=64K, r=0.3 | 20.902 ms | 34.84 ms |

优化手段：

1. **SIMT 并行（每查询点一线程）**：M 可到 64K，天然高并行；AIV 核 SIMT 通道数 × 核数提供充足线程数；线程内避免原子操作（候选写入按查询点独立槽位，无竞争）。
2. **grid/cell 候选剪枝（Tier 2）**：3D 场景候选评估量从 `M·N_b` 降至 `M·27·avg_cell_pts`。以 N=64K、r=0.2、单位立方体为例：每 cell 约 512 点，27 cell 约 1.4 万候选/查询 → 总评估约 8.8e8 次（较暴力法约 4.3e9 次降低 ~5×），配合 fp32 向量化可落至约 10~20 ms，满足目标。
3. **F 维向量化**：距离累加用矢量指令（VSUB/VMUL/VADD）一次处理多维度；对 F=3 展开为 3 条独立乘加，消除依赖链。
4. **UB 双缓冲流水**：`xTileSize × F` 候选块与查询点块均双缓冲，`CopyIn → Compute → CopyIn` 重叠，隐藏 GM→UB 搬运延迟。
5. **核融合与减少往返**：计数与候选写入合一（单趟遍历 x，避免 CUDA 早期版本的两趟方案）；压缩核内完成前缀和与 scatter，避免 host 侧额外 kernel 与图外内存拷贝。
6. **数据重排与局部性**：Tier 2 将 x 按 (batch, cell) 连续化，候选访问呈顺序流；查询点按核连续切分，同核查询点共享 x 候选块（块级复用）。
7. **PRNG 轻量化**：仅超限命中时才推进 PRNG（32 位状态，单指令更新），正常路径零开销。
8. **精度友好的 fp16/bf16 路径**：元素差在输入精度计算，平方与累加提升至 fp32，避免半精度累加溢出/丢精度导致的边界误判（同时保证与 CPU 参考的集合一致性，见"可维可测"）。

**达标分析**：Tier 2 在基准 5 组场景下评估量均落在 SIMT 核可达吞吐区间（单 AIV SIMT 有效距离评估吞吐预估 ≥ 5e10 次/s），估计 8~20 ms 区间，满足 ≤34.7 ms 目标；r=0.8（N=8K）等大半径场景自动回退 Tier 1（M·N_b = 6.4e7 次，亚毫秒级）。最终以实测为准，预留 `useGrid` 阈值调参位。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Ascend 950PR（AIV/SIMT） | √ |
| Atlas 800I/T A2（兼容验证，非验收目标） | 可选 |

## 算子约束限制

1. `batch_x` / `batch_y` 必须 sorted；搜索限定在 batch 内；
2. `r` 为 Python float 标量（非 Tensor），`> 0`；
3. `x`/`y` 须 contiguous，`F` 一致，`F ≥ 1`；
4. 仅支持**前向**（与 torch_cluster 一致，无反向）；
5. 超限邻居采样允许非确定性（集合等价验收）；
6. L1 支持 fp16/bf16/fp32；fp64 走 CPU 回退（L2，不考核性能）；
7. 空输入返回 `[2, 0]` LongTensor，不进入 kernel；
8. 输出 dtype 固定 int64（Long），与参考一致。

## 接口定义

### Python 层接口（`python/ops_gnn/radius.py`）

与参考 `torch_cluster/radius.py` 签名一致（含 `torch.jit.script` 兼容）：

```python
def radius(
    x: torch.Tensor,
    y: torch.Tensor,
    r: float,
    batch_x: Optional[torch.Tensor] = None,
    batch_y: Optional[torch.Tensor] = None,
    max_num_neighbors: int = 32,
    num_workers: int = 1,
    batch_size: Optional[int] = None,
    ignore_same_index: bool = False,
) -> torch.Tensor: ...

def radius_graph(
    x: torch.Tensor,
    r: float,
    batch: Optional[torch.Tensor] = None,
    loop: bool = False,
    max_num_neighbors: int = 32,
    flow: str = 'source_to_target',
    num_workers: int = 1,
    batch_size: Optional[int] = None,
) -> torch.Tensor: ...
```

Host 逻辑（与参考一致）：空输入早退 → 1D view → contiguous → batch_size 计算 → `torch.bucketize` 构造 `ptr_x`/`ptr_y` → 调用 `torch.ops.torch_cluster.radius`；`radius_graph` 校验 `flow` 并重组行列。

### C++ 扩展接口（自定义算子）

```
radius(Tensor x, Tensor y, Tensor? ptr_x, Tensor? ptr_y, float r,
       int max_num_neighbors, int num_workers, bool ignore_same_index) -> Tensor
```

经 `TORCH_LIBRARY_IMPL(torch_cluster, PrivateUse1)` 注册，NPU 设备自动分发；`ptr_x`/`ptr_y` 为 `None` 时表示单 example。

### 数据类型分级

| 级别 | dtype | 路径 |
|------|-------|------|
| L1 | float16 / bfloat16 / float32 | NPU（Ascend C SIMT 核） |
| L2 | float64 | CPU 回退（Python 层），bit-wise 一致；低精度拼接仅 README 文档化 |

---

# 可维可测分析

## 精度标准 / 性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | 严格满足《生态算子开源精度标准》（experimental_standard）；以 CPU 版 torch_cluster 为标杆，AscendOpTest 测试 | 任务书 §8 |
| 精度判定 | L1：每查询点半径内邻居集合与 CPU 标杆一致；超限随机截断时集合等价（固定 seed 下 bit-wise）；`radius_graph` 各 loop/flow 组合 edge 集合与 CPU 一致；L2：bit-wise | 任务书 §8 精度策略表 |
| 性能标准 | 所有用例 ≥ 0.6 倍 A100 标杆耗时（基准见 §3.2.6 表） | 任务书 §7 |

单标杆不满足时：采用 ATK 双标杆比对（`cv_fused_double_benchmark`），以更高精度 CPU 实现为真值；满足 **NPU/同精度 CPU 最大相对误差比例 ≤ 2、平均相对误差比例 ≤ 1.2、均方根误差比例 ≤ 1.2**。

## 测试验证计划

### 功能测试用例（`test/test_radius.py`）

| 编号 | 场景 | 用例要点 |
|------|------|----------|
| TC-01 | 基础 radius | 参考 `test_radius.py::test_radius`：8 点 x / 2 点 y、r=2、max=4，无 batch 与 batch 两分支，`to_set` 集合断言；含 `torch.jit.script` 分支 |
| TC-02 | batch | `batch_x=[0,0,0,0,2,2,2,2]`、`batch_y=[0,2]` 跳过 batch；`radius_graph` 大图（1000×3 随机点，cKDTree 真值比对） |
| TC-03 | max_num_neighbors 截断 | 构造命中数 > max 场景（如 r=100 全连接、max=1）：断言每查询点度数 ≤ max、集合等价 |
| TC-04 | ignore_same_index | 与参考语义一致：x/y 同 tensor 时跳过 `j==i`；`loop=False` 无自环 |
| TC-05 | radius_graph loop/flow | 4 点正方形 r=2.5：`source_to_target` / `target_to_source` 两组方向断言（对照官方用例边集）；`loop=True/False` |
| TC-06 | float64 L2 路径 | fp64 输入 CPU 回退结果与 CPU 参考 bit-wise 一致 |
| TC-07 | 空输入 | `x`/`y` 任一空 → `[2,0]` long |
| TC-08 | dtype 遍历 | fp16/bf16/fp32 各跑 TC-01~TC-05 |
| TC-09 | 随机采样集合等价 | 固定 seed：NPU 超限采样结果与参考（同 seed）按集合等价比对；不同 seed 允许不同集合但均满足半径/上限约束 |

### 精度测试（AscendOpTest）

- 依据《生态算子开源精度标准》构建 op base 用例（dtype/shape/参数组合矩阵），真值由 CPU 版 torch_cluster 生成；
- 比对维度：每查询点邻居**集合**（超限场景）或全序（未超限场景）；`radius_graph` 按 loop/flow 组合校验 edge 集合；
- 边界规避：用例构造避免 `dist ≈ r²` 的边界点（避免精度舍入导致集合分歧）；如需覆盖边界，使用 ATK 双标杆策略兜底；
- 测试入口提供 README（复现步骤、环境、命令），保证验收人可复现。

### 性能测试（`test/test_radius_perf.py`）

- 5 组基准场景（§3.2.6 表）：随机 3D 点云，`torch.npu.synchronize()` 计时，多次取中位数；
- 断言：NPU 耗时 ≤ A100 基准 / 0.6；
- 记录实现模式（Tier 1/2）与关键 tiling 参数，便于性能回归分析。

### 双标杆兜底（ATK）

单标杆（CPU torch_cluster）不满足时启用 `cv_fused_double_benchmark`，按 §"精度标准"的比例阈值判定。

## 兼容性分析

- **算子本体**：ops-gnn 新增算子，不修改既有算子；`radius`/`radius_graph` 为新增接口，无历史兼容负担；
- **生态兼容**：复用 `torch_cluster` 命名空间的 `torch.ops.torch_cluster.radius` 注册，NPU tensor 调用时自动分发，`radius.py` host 逻辑、`torch.jit.script`、`flow` 语义均与参考一致；
- **接口兼容**：签名/默认值/输出 dtype（int64）与 torch_cluster ≥ 1.6.0 对齐；`num_workers` 保留但忽略；
- **平台兼容**：仅 NPU（PrivateUse1）注册分发；CPU/GPU 路径不受影响；
- **精度兼容**：超限随机采样允许非确定性（集合等价），与任务书验收口径一致。

---

# 参考资料

1. 8月社区任务任务书：`radius_task_doc.md`
2. 设计文档模板：https://gitcode.com/cann/cann-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md
3. pytorch_cluster 参考源码：`torch_cluster/radius.py`、`csrc/cpu/radius_cpu.cpp`、`csrc/cuda/radius_cuda.cu`、`test/test_radius.py`（https://github.com/rusty1s/pytorch_cluster ，6dabb04）
4. ops-gnn 仓既有范式：`csrc/npu/kernel/add_sample/add_sample_kernel.cpp`（SIMT）、`csrc/npu/kernel/segment_max_csr/*`（tiling 范式）
5. 《生态算子开源精度标准》：https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md
6. AscendOpTest：https://gitcode.com/HIT1920/AscendOpTest ；ATK 双标杆：https://gitcode.com/AscendTest/ATK
