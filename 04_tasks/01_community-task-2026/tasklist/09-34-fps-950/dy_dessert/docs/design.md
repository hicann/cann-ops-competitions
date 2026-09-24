# FPS 算子设计文档

# 需求背景（required）

## 需求来源

CANN 社区任务《CANN训练营北京邮电大学-fps算子开发(950)》要求参考
`torch_cluster.fps`（建议版本不低于 1.6.0），在 Ascend 950 上使用
Ascend C Kernel 和 PyTorch 适配层实现接口、功能、精度与性能均满足任务书的
最远点采样算子，最终合入 `cann/ops-gnn`。

## 背景介绍

FPS（Farthest Point Sampling，最远点采样）用于从点集里选出分布尽量均匀的
子集，常见于 PointNet++、3D 检测和点云下采样。它不是寻找离第一个点最远的
若干点，而是维护每个候选点到“全部已选点”的最小距离，每轮选择该最小距离
最大的候选点。

对 batch 内候选点 `x_i` 和已选集合 `S`，评分为：

```text
score(i) = min_{s in S} ||x_i - x_s||^2
next = argmax_i score(i)
```

因此第三个点既要远离第一个点，也要远离第二个点；更准确地说，它最大化到最近
已选点的距离，而不是最大化到已选点的平均距离。

### 支持能力

| 参数 | 含义 | 类型与约束 |
| --- | --- | --- |
| src | 点特征矩阵 | NPU Tensor，可 view 为 `[N, F]`，float16/float32；允许非连续（Host 照上游 `view({N,-1}).contiguous()` 的做法先物化） |
| batch | 点所属 batch | 可选 int64 Tensor `[N]` |
| ratio | 每个 batch 的采样比例 | float 或 float16/float32 Tensor，`(0, 1]` |
| random_start | 是否随机选择首点 | bool |
| batch_size | batch 数 | 可选 int |
| ptr | CSR 分段边界 | 可选 int64 Tensor 或 `List[int]` |
| 返回值 | `src` 中的全局点下标 | NPU int64 Tensor `[K]` |

# 需求分析（required）

## 需求描述

公开接口与 `torch_cluster.fps` 保持一致：

```python
fps(src, batch=None, ratio=None, random_start=True, batch_size=None, ptr=None)
```

Python 层须保留四个 TorchScript overload，支持 `ratio` 的 float/Tensor 形式和
`ptr` 的 Tensor/List 形式。每个 batch 输出
`ceil(float32(ratio_b) * float32(degree_b))` 个全局下标；非空 batch 至少输出
一个点。

## 需求拆解

1. Python 层完成默认参数、batch 到 CSR ptr 的转换和底层 dispatcher 调用；
2. Host 层校验 dtype、device、shape，计算输出 CSR 边界并生成首点；
3. Kernel 层按 batch 独立迭代，更新每点到已选集合的最小平方距离；
4. 距离归约与并列选择都对齐 torch_cluster 的 **CPU** 实现（任务书指定的唯一
   标杆），使输出下标逐位一致；
5. 针对 F=4 和 F=16/32/64 的验收主形态优化距离计算与块内归约，并把单 batch
   的点集拆到多个 AIV（任务书性能矩阵全部是单 batch）；
6. 覆盖任务书全部功能、精度和性能矩阵。

# 详细设计（required）

## 算子分析

### 数学公式

设 batch 内有 `N` 个点，目标采样数为 `M = ceil(float32(ratio) * float32(N))`。
选定首点后维护：

```text
d_i^(0) = ||x_i - x_selected(0)||^2
d_i^(t) = min(d_i^(t-1), ||x_i - x_selected(t-1)||^2)
selected(t) = argmax_i d_i^(t)
```

CPU 标杆没有 `50000` 预热下限：选定首点后直接令
`dist = (x - x[start]).pow_(2).sum(1)`，之后逐轮取最小。kernel 用 “+inf”
初值实现等价的“首轮赋值”，因此每一轮的距离值都与标杆逐位一致。逐轮更新顺序
为“先 argmax 选点，再把新距离并入最小值”；并列选择采用 CPU 的 `argmax()`
首索引规则（见下）。

### 支持数据类型和形状

- `src`：float16、float32，形状 `[N, *]`，按 `[N, F]` 解释；
- `batch`、`ptr` 和输出：int64；
- `ratio`：float 标量或 float16/float32 Tensor；
- 支持单 batch、多 batch、空 CSR 段、非均匀分段和空输入。

## 算子实现

### Host 与 Python 侧设计

- `ratio=None` 转换为 0.5；float ratio 转成与 `src` 同 device、**dtype 与 `src`
  一致的标量**——上游 `torch_cluster/fps.py` 即
  `torch.tensor(ratio, dtype=src.dtype, device=src.device)`，任务书要求该 Python
  层逻辑须复现，故该 dtype 属接口语义而非实现细节；用户传入的 `ratio` Tensor
  保持其自身 dtype 与值，不被再量化；
- `ptr` 给定时直接调用底层算子，List 先转成 NPU int64 Tensor；
- `batch` 给定时通过 `scatter_add_` 得到各 batch 点数，再 `cumsum` 构造 ptr；
- 两者均未给定时构造 `ptr=[0, N]`；
- Host 侧按 `ceil(float32(degree) * float32(ratio))` 得到每个 batch 的输出数和
  `out_ptr`：乘法在 float32 中进行（`degree` 先转 float32），而 `ratio` 的**值**
  取自上述 Python 层构造的张量。fp16 输入下这既避免把 `degree` 舍入到 fp16
  （`2049 × 0.5` 应为 1025，而不是 1024），也保留 `ratio` 已被舍入到 fp16 的值
  （`10000 × 0.3` 应为 3001，而不是 3000）。`random_start` 的随机首点同样按
  float32 计算；
- 非连续但可 view 为 `[N, F]` 的 `src` 由 Host 先 `contiguous()` 物化；
- `random_start=True` 时在 NPU 上生成每 batch 的随机首点；否则取局部下标 0；
- 只将最终输出长度这一个标量同步到 Host 以分配输出，点特征和 FPS 主计算不
  离开 NPU；
- 使用当前 NPU stream 发射 Kernel，不创建额外 stream；
- 入口即校验接口契约，且 Python 层与底层入口各自独立校验：`ptr` 须首项为 0、
  末项等于 `src.size(0)` 且单调非降，`ratio` 须有限并落在 `(0, 1]`，`pointCount`
  须小于 `2^31`，`len(batch)` 须等于 `src.size(0)`。底层校验折进 Host 已有的那次
  标量回读（校验标志与输出长度一次拷回，不额外增加同步），因此绕过 Python 封装
  直接调用 `torch.ops.torch_cluster.fps` 时，非法参数同样进不了 kernel。

### Kernel 并行划分

一个 AIV block 负责一个 batch；batch 数超过可用 AIV 核数时，block 按
`gridDim.x` 跨步继续处理。每个 block 启动 512 个编译期固定的物理 SIMT 线程，
线程 `t` 处理局部点 `t + k*512`。512 个物理 worker 按 256 个逻辑 lane 折叠，
每个逻辑 lane 合并 `t` 和 `t+256` 两个 worker。该折叠形状只影响块内归约的
效率，不影响结果：并列比较是（距离降序、全局下标升序）的全序，归约树形状
不改变胜者（并列规则见下）。

该划分同时利用了两个层次的并行：

1. AIV block 之间并行处理不同 batch；
2. block 内 512 个 SIMT 线程并行更新候选点距离。

FPS 的下一轮依赖上一轮选出的点，因此同一 batch 的采样轮次不能并行，只在每轮
内部并行遍历候选点和执行最大值归约。

#### 单 batch 的点集拆分（性能关键）

任务书性能矩阵全部是单 batch，按上面的划分**只会用到一个 AIV**，而标杆是多核
CPU。因此当 `batchSize == 1 && (featureDim == 16 || featureDim == 32) &&
pointCount >= 1024` 时，把该 batch 的点集拆到 `FPS_SPLIT_BLOCKS = 4` 个 block：

- 每个 block 拥有连续的一段点，只读写自己那段 `distance`；
- 唯一的跨核数据是“本 block 的 (距离, 下标)”，每轮写进 3 个 bank 中对应的一
  个（`bank[sample % 3]`），折叠读 `bank[(sample-1) % 3]`；同一个 bank 的下次
  写入在两轮之后，而 block 不可能越过了一次 rendezvous 跑到两轮之后，因此
  **每轮一次 rendezvous** 就足够；
- 交接的两个方向都**绕过 L1**（`asc_ldcg` 读 / `asc_stcg` 写，只落在 L2），
  用设备级原子计数器做 rendezvous。普通 GM 访问配 `DataCacheCleanAndInvalid`
  是不行的：`clean` 会把调用核自己那份含对端上一轮旧值的整条 cache line 写回，
  反而覆盖掉对端刚发布的新值；
- 拆分路径**不复用**单核路径的 512 线程，而是用 `FPS_SPLIT_THREADS = 256` 与
  免暂存归约（每个线程直接做 warp shuffle，只有 warp lead 写 UB）。一轮拆给
  4 个 block 后每 block 只剩 1/4 的点，单轮里剩下的主要是归约和同步；512 线程
  意味着一半线程没有点却仍要参与 512 项暂存、16 个 warp 的 barrier 与归约，
  实测这一项值 16%（`1024×16 fp16` 400 次逐调用中位数 4.69~4.76 ms →
  3.95~4.01 ms）。

不满足上述条件时行为与原来完全一致（`partial == nullptr` 即走单核路径）。
实测 `4096×16 fp16` 设备侧耗时由 64.39 ms 降到 26.82 ms（2.40×），msprof 显示
Block Num 由 1 变为 4；`1024×4` 等 F=4 形态仍为单 block。

### 距离更新与特化

CPU 标杆的度量是张量级表达式 `(x - x[idx]).pow_(2).sum(1)`，且保持输入
dtype。`sum(1)` 由 PyTorch CPU 的 `cascade_sum`（`vectorized_inner_sum` →
`row_sum` → `multi_row_sum`）实现，其求和顺序并非逐特征累加：从 F≥16 起，逐特征累加的结果与它
在低位就已不同，而 FPS 是迭代算法，这种差异最终会改变被选中的下标。因此
kernel 必须复现该顺序，而不是“更高精度”的累加。

- 距离按下标顺序逐点独立计算：特征向量切成 `vecSize` 个长度为 `vecNumel`
  的向量，每个累加 lane 以步长 `vecNumel` 遍历向量，按 4 个一组进行 4 层 ×
  4 行的分块累加，最后按 lane 顺序折叠；
- float32 每步都是 float32 运算，`vecNumel = Vectorized<float>::size()`；
- float16 的减法与平方保持 half（与 CPU 逐元素算子一致），累加在 float32 中
  进行并在最后舍入回 half，`vecNumel = 2 * Vectorized<float>::size()`；
- 维度不足一个向量时退化为 CPU 的 `scalar_inner_sum` 路径；
- F=4 使用专用的 float4 装载路径，其累加顺序与 CPU 的标量路径一致；四项各自经
  乘积隔离 helper（`FloatSquareTerm`）累加，避免 `d0*d0 + d1*d1 + d2*d2 + d3*d3`
  被后端整体当成可融合的乘加链（F=4 float32 的真库网格 120 例中曾有 118 例因
  1 ulp 差值选点分叉，隔离后 120/120 逐位一致）；
- F=16/32/64 在 `sumLanes == 8` 时使用按维展开的等价实现（性能矩阵形状）；
- 累加 lane 宽度由 Host 按 PyTorch 的 CPU capability 选择：CPU `sum` 只为
  DEFAULT 与 AVX2 注册，AVX512 机器仍走 AVX2 kernel，故宽度为 8（仅无 AVX
  的机器为 4），可用 `OPS_GNN_FPS_SUM_LANES` 覆盖；
- **必须阻止 FMA 合并**：CPU 先对乘积舍入再累加，而 Ascend 后端会把
  `acc + d*d` 合并成 FMA，少一次乘积舍入（实测每条距离差 1 ulp，在
  `8192×256` float32 上第 1896 个采样点起选点分叉）。`-ffp-contract=off` 与
  `#pragma STDC FP_CONTRACT OFF` 只作用于前端，后端仍会合并，因此在 kernel 里
  写成 `d * d + zeroGuard`（`zeroGuard` 是 Host 传入的运行时 0），迫使后端先
  产生 `fl(d*d)` 再累加，与 CPU 逐位一致；
- 每个物理线程只写自己负责点的 distance，无原子操作和写冲突。

### 块内归约与并列规则

每个物理线程产生一组 `(best_distance, best_global_index)`，写入 UB：

- `bestDistance[512]`：float32，共 2 KiB；
- `bestIndex[512]`：int32，共 2 KiB（候选是点下标，`pointCount < 2^31` 时
  int32 可精确表示；写回 `out` 时拓宽为 int64）。

归约分三步：

1. 前 256 个线程分别合并同一逻辑 lane 的两个物理 worker；
2. 8 个 warp 使用 `asc_shfl_down` 在寄存器中完成 32-lane 归约；
3. 首个 warp 再归约 8 个 warp 的结果，lane 0 写出下一采样点。

比较首先选择更大距离；距离相同时选择**全局下标更小**者。这与 CPU 标杆的
`dist.argmax()`（返回第一个最大值）一致。**不能采用 CUDA kernel 的
`(index % 256, index // 256)` 并列规则**：CUDA 在等距时会让 256 胜过 1，而 CPU
会选 1。该比较关系是 (distance 降序, index 升序) 的全序，因此归约树的形状不
影响结果。

### NaN / Inf 语义

按 CPU 标杆处理：`torch.minimum` 在两个操作数任一为 NaN 时保留 NaN；
`torch.argmax` 在存在 NaN 时返回**首个** NaN 的下标。Host 每次调用检测一次
`src` 是否含**非有限值**（`torch.logical_not(torch.isfinite(src)).any()`，NaN 与
±Inf 一并覆盖；与既有的一次标量回读合并，不新增同步），kernel 用
`template <bool kNaN>` 编译期二选一：不含非有限值的输入走原来的朴素比较，含非有限值
时走该语义。仅检测 NaN 是不够的：`src` 自身含 `±Inf` 时，`inf - inf` 仍会产生
NaN 进入距离，归约胜负必须按"首个 NaN"判定；只测 NaN 会让这类输入走朴素比较
分支，实测 12 例 Inf 用例中有 10 例与 CPU 标杆分叉。任务书性能矩阵的输入不含
非有限值，因此常规路径不付这份开销。

### 内存与同步

- GM：`src`、`ptr`、`out_ptr`、`start`、`distance`、`out`；
- workspace：每个输入点一个与 `src` 同 dtype 的最小距离；拆分路径另需一段
  64 word 的零初始化 scratch（3 个 bank + 一个轮次计数器）；
- UB：每 block 固定约 4.0625 KiB（2 KiB 距离 + 2 KiB 下标 + 64 B 中心点行），
  保留充足 DCache 空间；
- 距离初始化后同步一次；每轮再使用三次 `asc_syncthreads()`，分别保护 worker
  候选写入、warp 首项写入和下一轮读取；空 batch 由所有线程一致跳过，不会使
  部分线程滞留在屏障前；
- 拆分的 rendezvous 轮询设自旋上限，超时用 `__trap()` 响亮失败，不静默返回
  错误索引。

### SIMT 接口规范

Kernel 使用 arch35 C 风格 SIMT 接口：`threadIdx.x`、`blockIdx.x`、
`gridDim.x`、`asc_vf_call`、`dim3`、`asc_syncthreads` 和
`asc_shfl_down`。线程数常量同时用于 `LAUNCH_BOUND` 与 `dim3`，VF 参数总大小
不超过 28×32 bit。

## 支持硬件

| 支持的芯片版本 | 支持情况 |
| --- | --- |
| Ascend 950（arch35 / dav-3510） | 支持 |

验证环境为 CANN 9.1.0、配套 PyTorch/torch_npu，要求 PyTorch 版本不低于 2.7。

## 算子约束限制

1. `src` 首维表示点数；非连续但可 view 为 `[N, F]` 的输入（转置、切片等）
   是允许的，Host 会先物化，再交给按 `point * featureDim` 线性寻址的 kernel；
2. `ratio` 应在 `(0, 1]`，标量或每 batch 一个值；**标量 `ratio` 按 `src.dtype`
   构造**（与上游 `torch_cluster/fps.py` 一致：fp16 输入下其值即 fp16 舍入后的
   值），传入的 `ratio` Tensor 保持自身 dtype 与值；采样数为
   `ceil(float32(degree) * float32(ratio))`——乘法在 float32 中进行，`ratio` 的
   值取自上述张量——与 CPU 标杆（CPU 版 `torch_cluster` 的完整管线）及仓库
   `test/fps/golden.py` 一致；
3. `ptr` 为单调非降 CSR 边界，首项为 0、末项为 N；
4. `src` 含 NaN/±Inf 时按 CPU 标杆语义处理，并以编译期分支实现（见上）；
5. 单个 batch 的采样轮次具有算法级串行依赖；
6. `pointCount < 2^31`：kernel 内候选下标在 UB 中以 `int32_t` 存放（把每轮
   归约的 UB 搬运量减半），写回 `out` 时才拓宽为 `int64_t`；任务书矩阵最大
   16384，远低于该界。Host 已显式校验该上界，越界直接报错而不是静默截断。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 功能 | PyTorch 层签名和分支行为与 `torch_cluster.fps` 一致 | 任务书 |
| 精度 | float16/float32 输出 int64 下标与 CPU 单标杆 bit-wise 一致 | 任务书、生态算子开源精度标准 |
| 性能 | 任务书每个性能用例均不低于 0.45 倍标杆性能 | 任务书 |

自测覆盖单/多 batch、ptr Tensor/List、per-batch ratio、空输入（实际调用 NPU
算子）、随机首点、大规模形状、非连续输入、NaN/Inf 语义（单核路径与多 AIV 拆
分路径各一组，按 CPU 标杆语义比对）和距离平局。平局用例单独覆盖两件事：等距时返回最小全局
下标（CPU `argmax()` 语义，与 CUDA 的 256-lane 规则相反），以及 batch 起点非
256 对齐时结果不随偏移改变。距离归约另有专门用例覆盖 F=4、F≥16 的累加顺序与 CPU
逐特征累加的分歧。另有负向用例覆盖底层入口拒绝非法参数：`ptr` 非单调、首项非 0、
末项不等于 N，以及 `ratio` 为 0、大于 1、NaN、±Inf 均须报错，合法参数须正常
返回；`len(batch) != src.size(0)` 须报 `ValueError`。采样数另有针对性用例覆盖
`2049 × 0.5 = 1025`（fp16 装不下 degree）与 `10000 × 0.3 = 3001`（fp16 输入）/
`3000`（fp32 输入）（fp16 装不下 ratio 的值）两类场景。性能脚本覆盖任务书 21 个
shape/ratio 组合，并分别测试 float16 和 float32。

在 Ascend 950 + CANN 9.1.0 环境回归：任务书官方矩阵按 CPU 标杆语义重评
62/62（61 项 + 1 项平局守卫）、验收方重建矩阵 66/66、仓内扩展用例 68/68 通过；
差分门禁 920 例（900 例常规配置 + 20 例 NaN 注入配置）与 CPU golden 逐位一致；42 个
dtype/shape/ratio 性能项按官方协议（`warmup=20`、`iter=100`）在**最终二进制**上重测：
**全量 42 项 × 5 遍中位数 42/42 过门限，全矩阵最低 0.494**（`1024×16 f16`），
`1024×4` 家族 5 遍中位数落在 0.496~0.531；单遍官方协议同样 42/42。余量最小的是
`1024×4` 家族（该族按 gate 不拆核、只用一个 AIV，收益只来自距离与归约本身），故对它另做
10 遍独立测量：最紧的 `1024×4 r=1.0 fp16` 为 0.450，**8 档 × 10 遍共 80 次测量全部过门限**。
（"42 项 × 5 遍中位数、全矩阵最低 0.506（`1024×16 fp16`）"是 F=4 累加修复**之前**的二进制上
测得的基线，保留在交付包 `logs/perf_median_5pass.log`，不再作为最终结论。）
验收机为共享机器：100~400 ms 的调度停顿可把 `1024×4` 这类 1 ms 量级的用例抬高 10%~40%
——本轮出现过一次 5 遍窗口读到 0.434，同一天在同一二进制上空闲重测时该项 5 遍中位数为
0.520——故建议以多轮中位数或最小值作判据。

## 兼容性分析

- 通过 `torch.library` 注册 `torch_cluster::fps` 的 PrivateUse1 实现；
- 若进程已加载 torch_cluster 的 CPU/CUDA Kernel，不覆盖原有 dispatch；
- 保留四个 TorchScript overload 和全部公开参数；
- 输出始终位于输入所在 NPU，dtype 固定为 int64。
