# radius / radius_graph 算子设计文档

> 8月社区任务 · radius 算子开发 · 适配硬件：Ascend 950PR
> 对齐目标：`torch_cluster.radius`、`torch_cluster.radius_graph`（https://github.com/rusty1s/pytorch_cluster ，版本 ≥ 1.6.0）

---

# 需求背景（required）

## 需求来源

2026年8月CANN社区任务《8月社区任务 - radius 算子开发》，任务书要求以 Ascend C 编程语言在 `ops-gnn` 开源仓（https://gitcode.com/cann/ops-gnn ）中实现与 `torch_cluster.radius` / `torch_cluster.radius_graph`（版本建议 ≥ 1.6.0）**接口完全一致**的 NPU 版本算子。开发、测试硬件为 **Ascend 950PR**，CANN 版本为算子开源仓指定版本。

## 背景介绍

### radius 算子实现优化

`radius` 是图神经网络（PointNet++、DGCNN 等）中最常用的邻域查询原语之一：对查询点集 `y` 中的每一点，找出参考点集 `x` 中欧氏距离 **< r** 的所有邻居点（每点最多 `max_num_neighbors` 个），输出边索引 `edge_index [2, E]`。`radius_graph` 是 `radius(x, x, ...)` 的自环特例，用于点云/几何 GNN 模型的自监督建图。

该算子天然是**访存密集 + 分支判断**型算子：计算量随点数平方级增长（最坏 O(M·N) 次距离评估），无法用常规张量算子（矩阵乘、逐元素）直接表达，必须依赖**并行度逐查询点**的 SIMT/标量多线程编程模型实现。

torch_cluster 已有 CPU（nanoflann KD-tree）与 CUDA（每查询点一线程）后端，但缺少昇腾 NPU 后端。本任务使用 Ascend C SIMT 范式在 Ascend 950PR 上实现 NPU 版本，通过 Python/PyTorch 扩展提供接口完全一致的算子。验收以 PyTorch 层为准，覆盖 `radius` + `radius_graph`。

### radius 算子 TBE 实现现状分析

本算子为**全新算子，无历史 TBE 版本**。昇腾生态此前无 radius 的 TBE 实现，故本设计不包含"与 TBE 版本对比"章节，直接以 torch_cluster 的 CPU/CUDA 实现为对齐基准。

torch_cluster 参考实现分析：

| 实现 | 路径 | 核心算法 |
|------|------|----------|
| Python 宿主 | `torch_cluster/radius.py` | 空输入早退；1D→2D view；构造 `ptr_x`/`ptr_y`（batch > 1 时用 `bucketize`）；调用 `torch.ops.torch_cluster.radius`；`radius_graph` 复用 `radius` 并按 `flow` 重组行列 |
| CPU | `csrc/cpu/radius_cpu.cpp` | 基于 nanoflann KD-tree 的 `radiusSearch`，逐 query 取回命中；同一 batch 内搜索 |
| GPU | `csrc/cuda/radius_cuda.cu` | 每查询点一个 CUDA 线程；遍历同 example 内全部 x 点，逐 F 维累加平方距离 `dist < r*r` 判定；命中写入 `[M × max_num_neighbors]` 缓冲；kernel 后 `masked_select` 压缩为 `[2, E]` |

关键语义（已对照参考源码与官方 `test/test_radius.py` 逐一核对）：

1. **输出布局**：`edge_index[0] = y 中查询点索引`，`edge_index[1] = x 中邻居索引`。
2. **距离判定**：`Σ_d (x[j,d]-y[i,d])² <= r·r`（含边界；直接比较平方距离，避免开方）。
3. **batch 语义**：仅搜索同一 example 内的 x（`ptr_x[example]..ptr_x[example+1]`）；batch 须 sorted。
4. **超限处理**：命中数超过 `max_num_neighbors` 时，当前实现按**扫描顺序前 K 个**确定性截断（与 torch_cluster CPU/CUDA 参考一致），任务书允许非确定性随机采样，集合等价为验收判据。
5. **`ignore_same_index`**：为 `True` 时跳过 `x` 索引与 `y` 索引相同的点对；`radius_graph` 传 `not loop`。
6. **空输入**：直接返回 `empty(2, 0, dtype=long)`，不进入 kernel。

### radius 算子功能分析

- **功能**：对 `y` 中每点，找 `x` 中欧氏距离 `<= r` 的邻居（上限 `max_num_neighbors`），输出 `edge_index [2, E]`（int64）；仅同一 batch 内搜索。
- **输入**：`x` [N, F]、`y` [M, F]、`r`（float 标量）、`batch_x`/`batch_y`（可选）、`max_num_neighbors`（默认 32）、`ignore_same_index`、`num_workers`、`batch_size`。
- **输出**：`edge_index` [2, E] int64。
- **支持数据类型**：L1 float16 / bfloat16 / float32（NPU 路径）；L2 float64（CPU 回退，bit-wise 一致）。
- **`radius_graph`**：内部调用 `radius(x, x, r, batch, batch, ...)`，处理 `loop` 与 `flow` 语义。

---

# 需求分析（required）

## 需求描述

使用 Ascend C 编程语言（SIMT 范式）实现 `radius` / `radius_graph` 算子，支持 float16、bfloat16、float32（L1，NPU 路径）与 float64（L2，CPU 回退），接口与 `torch_cluster >= 1.6.0` 完全一致，验收以 PyTorch 层为准。性能不低于 A100 标杆耗时的 0.6 倍。

## 需求拆解

1. Python 层接口、参数名、默认值、返回类型与 torch_cluster 一致；支持 `torch.jit.script`。
2. 空输入返回 `[2, 0]`；1 维输入展平为 `[N,1]`；非连续输入自动 contiguous。
3. 支持 sorted batch；同 batch 内搜索；正确构造 `ptr_x`/`ptr_y`。
4. `max_num_neighbors` 截断按**邻居集合等价**验收。
5. 支持 `ignore_same_index` 与 `radius_graph` 的 `loop`/`flow`。
6. L1 float16/bfloat16/float32 走 NPU；L2 float64 走 CPU 回退（bit-wise 一致）。
7. 性能不低于 A100 标杆耗时的 0.6 倍（所有基准用例）。
8. 提供自测用例、自测报告与可复现的测试步骤。

---

# 详细设计（required）

## 算子分析

### 数学公式

查询点 `y_i ∈ R^F`（i = 0..M-1），其 batch 为 `b = batch_y[i]`：

```
edge_index 中所有满足下式的点对 (i, j)：
    j ∈ [ptr_x[b], ptr_x[b+1])
    dist2 = Σ_d (x[j,d] - y[i,d])²  <=  r²
    且 (ignore_same_index == false 或 j != i)
```

每查询点邻居数上限 `max_num_neighbors`；输出 `edge_index[0]=i`（y 索引）、`edge_index[1]=j`（x 索引）。float16/bfloat16 输入在设备端提升为 float32 累加，降低精度损失。

### 支持数据类型

| 级别 | dtype | 实现路径 | 说明 |
|------|-------|----------|------|
| L1 | float16 / bfloat16 / float32 | NPU Ascend C SIMT kernel | 距离累加用 fp32 中间精度；主验收路径 |
| L2 | float64 | CPU 回退（Python 层转 CPU 调参考实现，结果回拷 NPU） | 不做性能考核；bit-wise 一致 |

索引与输出：`ptr_x`/`ptr_y`、`edge_index` 均为 int64。

### 支持形状

| 参数 | 类型 | 约束 |
|------|------|------|
| `x` | Float `[N, F]` | L1：fp16/bf16/fp32；L2：fp64（CPU 回退）。`F >= 1`，2D（1D 由 Python 层 `view(-1,1)`） |
| `y` | Float `[M, F]` | 与 x 同 dtype、同 F |
| `r` | float | `> 0`，Python float 标量（非 Tensor） |
| `batch_x` / `batch_y` | Optional Long | 须 sorted；长度分别等于 N / M |
| `max_num_neighbors` | int | 默认 32；>= 1；每 y 点最多返回邻居数 |
| `num_workers` | int | NPU 忽略（接口保留，不参与调度） |
| `ignore_same_index` | bool | True 时跳过 `j == i` 点对（`radius_graph` 用 `not loop`） |
| `batch_size` | Optional[int] | 缺省自动计算 |

**空输入**：`x.numel()==0` 或 `y.numel()==0` 时返回 `torch.empty(2, 0, dtype=torch.long, device=x.device)`，不进入 kernel。

## 算子实现

### 实现方案

`radius` 采用 **SIMT（Ascend C 单指令多线程）+ 空间网格剪枝** 方案：查询点按 SIMT 线程并行分派，每个查询点由单一线程处理（无原子操作）；3D 输入构建空间网格，kernel 只探测 27 邻域 cell 的候选点，并用升序 top-K 数组 + early-break 保证输出为 CPU 参考顺序。`feature_dim==3` 且 `K<=64` 且网格密度适中时启用空间网格，否则回退暴力扫描。

#### 3.2.1 host 侧设计：

**tiling 策略**：

`RadiusTilingData` 结构体（n、m、feature_dim、batch_size、max_num_neighbors、r²、ignore_same_index、use_grid、core_num、workspace_pairs、grid_cell_count、config_ptr）以原始字节拷贝到 GM，kernel 逐字段读取。SIMT 启动路径不按值传聚合结构（已证实不可靠），改由 GM 指针传递。网格构建结果（reordered_x、orig_idx、cell_starts、config）作为独立 GM 数组传入，kernel 按 cell_offsets 稠密寻址，无需二次 tiling。

**1. 分核策略**：

按 `PlatformAscendCManager::GetCoreNumAiv()` 取全部 AIV 核（56 个）启动 kernel。查询分派：块 `b` 负责连续区间 `[b*queriesPerBlock, (b+1)*queriesPerBlock)`（`queriesPerBlock = ceil(m / blockNum)`），块内 1024 线程 stride 步进。相比全局 grid-stride（`q = globalTid`，只用到前 m/1024 个块），连续区间让全部 AIV 核参与，避免 7/8 核空闲（实测 kernel 4.4ms → 1.5ms）。

**2. 数据分块和内存优化策略**：

- **Host 侧 CPU 配额**：容器 cgroup CPU 配额 32 核，torch 默认 128 线程在 grid 构建的 min/max/sort 爆发时触发 ~78ms 周期冻结。构建期间将 intra-op 线程池限制到配额（`NumThreadsCap`），用后恢复。
- **Device 侧网格构建**：单 batch 路径 min/max、cell、sort、index_select、offsets 全部在 NPU 执行，消除 `x.cpu()` 往返和 CPU `torch::sort`（cgroup 下 N≥32K 退化至 ~8ms，device 上 ~0.07ms）。
- **输出压缩在 device 执行**：`repeat_interleave` + `masked_select` 在 NPU 上 将稠密 `m*K` 缓冲压缩为 `[2, E]`，只 D2H 拷贝压缩结果（避免 16MB 全量拷贝）。

**3. tilingkey 规划策略**：

算子内部按 `feature_dim==3` 且 `K<=64` 且网格密度适中时启用空间网格（`use_grid=1`），否则回退暴力扫描。无需外部 tiling 多档 key。

**入口流程**：

```
radius_npu(x, y, ptr_x, ptr_y, r, max_num_neighbors, num_workers, ignore_same_index)
  ├─ 1. 校验：device(PrivateUse1)/contiguous/2D/F 一致/dtype 合法/r>0
  ├─ 2. 空输入早退（x.numel()==0 || y.numel()==0 → [2,0] long）
  ├─ 3. L2 路径：dtype==float64 → Python 层已回退 CPU（host 断言兜底）
  ├─ 4. 构造 ptr（None → [0, N]/[0, M]；否则取传入 ptr，校验 sorted、numel 一致）
  ├─ 5. 计算 tiling：模式选择、分核、tile 大小、grid 参数
  ├─ 6. 分配设备中间缓冲（counts[m]、out_row0[m*K]、out_row1[m*K]）
  ├─ 7. 按需启动"网格构建流水"（Device 侧 BuildGridDevice：min/max、cell、sort、
  │      index_select、offsets 全在 NPU 执行）
  ├─ 8. 启动"SIMT 搜索核"
  ├─ 9. 输出压缩（device 侧 repeat_interleave + masked_select → [2, E]）
  └─ 10. D2H 拷贝压缩结果，返回 edge_index [2, E]
```

Stream 复用：kernel 在 torch 框架当前流上启动（`torch.npu.current_stream()` 传入），与 H2D/零填充顺序天然一致，无需每次 `aclrtCreateStream/Destroy`；kernel 后 `aclrtSynchronizeStream` 保持 host-sync 语义。

#### 3.2.2 kernel 侧设计：

kernel 进行 Init 和 Compute 两个阶段。

**Init 阶段**：每线程从 GM tiling 读取标量参数（n/m/feature_dim/r²/K/use_grid 等），读取网格配置（cell_size、origin、dims），确定本线程所属块与查询区间。

**Compute 阶段**（每个查询点）：
1. 定位查询点 cell（`(y-origin)/cell_size` 取整，clamp 到网格边界）。
2. 探测 27 邻域 cell，先 AABB 最小距离拒绝（整 cell 不可能含邻居则跳过）。
3. 扫描 cell 内候选：逐轴早拒绝 → `dist² <= r²` → 升序 top-K 线性插入，`idx >= top[K-1]` 时 early-break。
4. 写出 out_row0/out_row1（升序）+ counts。

**CopyOut**：kernel 直接写 GM 结果，无 UB 中间缓冲；host 后续用 device 侧 `repeat_interleave`/`masked_select` 压缩后 D2H。

**并行模型**：AIV 块按连续区间分派查询点（`qStart = blockIdx * queriesPerBlock`，块内 1024 线程 stride 步进），全部 AIV 核参与负载均衡。每个查询点由**恰好一个线程**处理，槽位分配**无需原子操作**。

**搜索**：查询点确定所在 batch（对 `ptr_y` 二分），扫描 `[ptr_x[b], ptr_x[b+1])`（batch_size==1 时扫描全量）。`dist² <= r²` 接受（含边界，与上游一致）；`ignore_same_index && i==q` 跳过。

**空间网格（3D 主路径，`use_grid=1`）**：点按 (batch, cell) 排序，`cell_starts[cell_id]` 稠密偏移表 O(1) 寻址。kernel 对每个查询点探测其所在 cell 的 **27 邻域**，先做 AABB 最小距离拒绝。

**排序数组 top-K + early-break**：每个查询维护**升序有序数组 top[64]**，用线性插入维护 `top[K-1]` 门槛；cell 内候选按 index 升序扫描，一旦 `idx >= top[K-1]` 即 break（早停，剪枝 4.8x）。数组天然升序 → 输出直接为 CPU 参考顺序（官方 `torch.equal` 通过）。

**逐轴早拒绝**：候选先做逐轴范围检查（`|dx|,|dy|,|dz| <= cell_size`）再算 dist²，减少无效 fma。

**回退暴力扫描**：feature_dim≠3、N/M<8、K>64 或网格过密（>4M cells）时，回退全量扫描 + early-exit 截断（扫描顺序前 K，确定性）。

**SIMT 编程模型**（参考 `ops-gnn/csrc/npu/kernel/add_sample/add_sample_kernel.cpp` 既有范式）：

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
```

线程映射：每个 SIMT 线程处理一个查询点，与 CUDA 参考的"一 query 一线程"完全同构。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Ascend 950PR（AIV/SIMT） | √ |

## 算子约束限制

1. `batch_x` / `batch_y` / `batch` 必须 sorted；搜索限定在 batch 内。
2. `r` 为 Python float 标量（非 Tensor），`> 0`。
3. `x`/`y` 须 contiguous，`F` 一致，`F >= 1`。
4. 仅支持**前向**（与 torch_cluster 一致，无反向）。
5. 超限邻居按扫描顺序前 K 个确定性截断（与 CPU/CUDA 参考一致）；任务书允许非确定性随机采样，集合等价为验收判据。
6. L1 支持 fp16/bf16/fp32；fp64 走 CPU 回退（L2，不考核性能）。
7. 空输入返回 `[2, 0]` LongTensor，不进入 kernel。
8. 输出 dtype 固定 int64（Long），与参考一致。
9. 网格路径（feature_dim==3 且 K<=64 且网格不超密）下邻居按 x-index 升序输出；回退暴力路径按扫描顺序输出。两者均与 CPU 参考顺序一致。
10. 不支持 `ptr_x`/`ptr_y` 手动传入的稀疏切分（仅通过 batch 推断）。

---

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述(不涉及说明原因) | 标准来源 |
| --- | --- | --- |
| 精度标准 | radius 输出 int64 索引（整型索引算子），《生态算子开源精度标准》明确"整型计算类算子需按各算子实际业务场景单独制定标准，不在本标准讨论范围内"（experimental_standard 第 0 节）。任务书 §8 自定义精度判据：**L1 邻居集合与 CPU 标杆一致，超限随机截断时集合等价（固定 seed 下 bit-wise）；L2 float64 CPU 回退 bit-wise**。本实现精度更严：非超限场景与 CPU 参考**逐位（bit-wise）一致**（matched_ratio 1.0、绝对误差 0） | 任务书 §8；experimental_standard |
| 性能标准 | 所有用例 ≥ 0.6 倍 A100 标杆耗时（基准表见下） | 任务书 §7 |

**性能基准与实测**（Ascend 950PR，2026-08-07，官方均值口径 warmup20/iters100/perf_counter，K=32，随机 3D 坐标 randn，与官方 benchmark 数据分布一致）：

预算 = A100 耗时 / 0.6；预算占用率 < 100% 即 PASS。

float32：

| 场景 | A100(us) | 预算(us) | NPU(us) | 预算占用 | 判定 |
| --- | ---: | ---: | ---: | ---: | ---: |
| N=8K, r=0.8 | 2118 | 3530 | 2644 | 74.9% | PASS |
| N=16K, r=0.5 | 3916 | 6527 | 3228 | 49.5% | PASS |
| N=32K, r=0.3 | 8446 | 14077 | 5125 | 36.4% | PASS |
| N=64K, r=0.2 | 20830 | 34717 | 7959 | 22.9% | PASS |
| N=64K, r=0.3 | 20902 | 34837 | 9481 | 27.2% | PASS |

float16：

| 场景 | A100(us) | 预算(us) | NPU(us) | 预算占用 | 判定 |
| --- | ---: | ---: | ---: | ---: | ---: |
| N=8K, r=0.8 | 2269 | 3782 | 2941 | 77.8% | PASS |
| N=16K, r=0.5 | 4424 | 7373 | 3427 | 46.5% | PASS |
| N=32K, r=0.3 | 10082 | 16803 | 5251 | 31.3% | PASS |
| N=64K, r=0.2 | 28418 | 47363 | 8636 | 18.2% | PASS |
| N=64K, r=0.3 | 28470 | 47450 | 10424 | 22.0% | PASS |

全部 10 行通过预算（NPU 耗时 <= A100耗时/0.6）。8K 为最薄余量行（ratio 0.78-0.94，多轮稳定），16K/32K/64K 均有充足余量。

**性能优化手段**：
1. **SIMT 并行（每查询点一线程）**：M 可到 64K，天然高并行；AIV 核 SIMT 通道 × 核数提供充足线程；线程内无原子操作（候选写入按查询点独立槽位）。
2. **空间网格剪枝**：3D 场景候选评估量从 `M·N_b` 降至 `M·27·avg_cell_pts`；针对 randn 分布（中心密集、边缘稀疏），kernel 用稠密 `cell_starts` 直接寻址（O(1) 定位，无 GM 二分查找）。
3. **AABB 拒绝**：先做整 cell 最小距离检查，不可能含邻居的 cell 直接跳过。
4. **排序数组 top-K + early-break**：`top[K-1]` 门槛使 cell 内扫描按 index 升序早停（剪枝 4.8x），输出天然为 CPU 参考顺序（`torch.equal` 精确通过）。
5. **Device 网格构建**：单 batch 路径 min/max、cell、sort、index_select 全在 NPU 执行，offsets 用 bincount+cumsum，消除 CPU `torch::sort`（cgroup 配额下 N≥32K 退化至 ~8ms）。
6. **F 维向量化**：距离累加用矢量指令（VSUB/VMUL/VADD）一次处理多维度；对 F=3 展开为 3 条独立乘加。
7. **输出压缩在 device**：`repeat_interleave` + `masked_select` 在 NPU 上压缩 `m*K` 缓冲为 `[2, E]`，只 D2H 拷贝压缩结果。
8. **Host 线程池配额限制**：grid 构建期间将 intra-op 线程数限制到 cgroup 配额（32），消除 ~78ms 周期冻结。
9. **网格自动回退**：网格在 cell 数超 4M 时自动回退暴力扫描（细网格无收益）。

## 兼容性分析

- **算子本体**：ops-gnn 新增算子，不修改既有算子；`radius`/`radius_graph` 为新增接口，无历史兼容负担。
- **生态兼容**：复用 `torch_cluster` 命名空间的 `torch.ops.torch_cluster.radius` 注册，NPU tensor 调用时自动分发；`radius.py` host 逻辑、`torch.jit.script`、`flow` 语义均与参考一致。
- **接口兼容**：签名/默认值/输出 dtype（int64）与 torch_cluster >= 1.6.0 对齐；`num_workers` 保留但忽略。
- **平台兼容**：仅 NPU（PrivateUse1）注册分发；CPU/GPU 路径不受影响。
- **精度兼容**：非超限场景与 CPU 参考 bit-wise 一致；超限截断按集合等价验收（与任务书口径一致）。

---

# 验收用例

| 编号 | 场景 | 说明 | 覆盖 |
| --- | --- | --- | --- |
| TC-01 | 基础 radius | 参考官方 `test_radius.py`：8 点 x / 2 点 y、r=2、max=4，无 batch 与 batch 两分支，`to_set` 集合断言；含 `torch.jit.script` 分支 | 25/25 PASS |
| TC-02 | batch | batch_x/batch_y 多 batch 隔离搜索；`radius_graph` 大图（1000×3 随机点，cKDTree 真值比对） | 25/25 PASS |
| TC-03 | max_num_neighbors 截断 | 构造命中数 > max 场景（r=100 全连接、max=1）：断言每查询点度数 <= max、集合等价 | 25/25 PASS |
| TC-04 | ignore_same_index | x/y 同 tensor 时跳过 `j==i`；`loop=False` 无自环 | 25/25 PASS |
| TC-05 | radius_graph loop/flow | 4 点正方形 r=2.5：`source_to_target` / `target_to_source` 两组方向断言；`loop=True/False` | 25/25 PASS |
| TC-06 | float64 L2 路径 | fp64 输入 CPU 回退结果与 CPU 参考 bit-wise 一致 | 25/25 PASS |
| TC-07 | 空输入 | `x`/`y` 任一空 → `[2,0]` long | 25/25 PASS |
| TC-08 | dtype 遍历 | fp16/bf16/fp32 各跑 TC-01~TC-05 | 25/25 PASS |
| TC-09 | 自研扩展 | 自研 `test_radius_custom.py` 15 例（边界 shape/dtype/截断） | 15/15 PASS |

**补充测试**（详见自测报告）：
- GT 管道（`scripts/test_npu.py`）：30/30 PASS（per-dtype set-equivalence）
- 完整矩阵（`scripts/test_matrix.py`）：33/33 PASS
- 位精确（`scripts/test_bitwise.py`）：14/14 PASS（非超限 bit-wise + 截断 bit-wise）
- AscendOpTest（`aot/radius/`）：3/3 PASS（FrameworkLaunch 包装，位精确 vs numpy golden）
- 性能 benchmark（`benchmark/run_benchmark.py`）：10/10 行通过预算

---

# 交付计划

1. 本设计文档以 PR 形式提交至 `cann-competitions` 任务列表，评审通过后合入。
2. 已完成 950PR 环境功能开发与性能优化（实测 10/10 达标）。
3. 完成精度与性能自测报告、测试步骤文档。
4. 提交代码至个人仓（https://gitcode.com/KuaaMU/ops-gnn，分支 master）并邀请 `Ascend-CANN`，提交验收。
5. 验收通过后合入 `ops-gnn/radius`。

---

# 参考资料

1. 8月社区任务任务书：`04_tasks/01_community-task-2026/docs/202608/radius_task_doc.md`
2. 设计文档模板：https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md
3. pytorch_cluster 参考源码：`torch_cluster/radius.py`、`csrc/cpu/radius_cpu.cpp`、`csrc/cuda/radius_cuda.cu`、`test/test_radius.py`（https://github.com/rusty1s/pytorch_cluster ，>= 1.6.0）
4. ops-gnn 仓既有范式：`csrc/npu/kernel/add_sample/add_sample_kernel.cpp`（SIMT）
5. 《生态算子开源精度标准》：https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md
6. AscendOpTest：https://gitcode.com/HIT1920/AscendOpTest
