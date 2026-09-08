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
| src | 点特征矩阵 | 连续 NPU Tensor，可 view 为 `[N, F]`，float16/float32 |
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
`ceil(ratio_b * degree_b)` 个全局下标；非空 batch 至少输出一个点。

## 需求拆解

1. Python 层完成默认参数、batch 到 CSR ptr 的转换和底层 dispatcher 调用；
2. Host 层校验 dtype、device、shape，计算输出 CSR 边界并生成首点；
3. Kernel 层按 batch 独立迭代，更新每点到已选集合的最小平方距离；
4. 保留 torch_cluster CUDA 的 256-lane 并列距离选择顺序，使输出下标逐位一致；
5. 针对 F=4 和 F=16/32/64 的验收主形态优化距离计算与块内归约；
6. 覆盖任务书全部功能、精度和性能矩阵。

# 详细设计（required）

## 算子分析

### 数学公式

设 batch 内有 `N` 个点，目标采样数为 `M=ceil(ratio*N)`。选定首点后维护：

```text
d_i^(0) = 50000
d_i^(t) = min(d_i^(t-1), ||x_i - x_selected(t-1)||^2)
selected(t) = argmax_i d_i^(t)
```

距离初值 `50000`、逐轮更新顺序和并列选择顺序均与参考实现对齐。

### 支持数据类型和形状

- `src`：float16、float32，形状 `[N, *]`，按 `[N, F]` 解释；
- `batch`、`ptr` 和输出：int64；
- `ratio`：float 标量或 float16/float32 Tensor；
- 支持单 batch、多 batch、空 CSR 段、非均匀分段和空输入。

## 算子实现

### Host 与 Python 侧设计

- `ratio=None` 转换为 0.5；float ratio 转成与 `src` 同 device/dtype 的标量；
- `ptr` 给定时直接调用底层算子，List 先转成 NPU int64 Tensor；
- `batch` 给定时通过 `scatter_add_` 得到各 batch 点数，再 `cumsum` 构造 ptr；
- 两者均未给定时构造 `ptr=[0, N]`；
- Host 侧由 `degree * ratio` 向上取整得到每个 batch 的输出数和 `out_ptr`；
- `random_start=True` 时在 NPU 上生成每 batch 的随机首点；否则取局部下标 0；
- 只将最终输出长度这一个标量同步到 Host 以分配输出，点特征和 FPS 主计算不离开 NPU；
- 使用当前 NPU stream 发射 Kernel，不创建额外 stream。

### Kernel 并行划分

一个 AIV block 负责一个 batch；batch 数超过可用 AIV 核数时，block 按
`gridDim.x` 跨步继续处理。每个 block 启动 512 个编译期固定的物理 SIMT 线程，
线程 `t` 处理局部点 `t + k*512`。512 个物理 worker 最后映射为参考 CUDA
实现的 256 个逻辑 lane，每个逻辑 lane 合并 `t` 和 `t+256` 两个 worker。

该划分同时利用了两个层次的并行：

1. AIV block 之间并行处理不同 batch；
2. block 内 512 个 SIMT 线程并行更新候选点距离。

FPS 的下一轮依赖上一轮选出的点，因此同一 batch 的采样轮次不能并行，只在每轮
内部并行遍历候选点和执行最大值归约。

### 距离更新与特化

- 通用路径逐特征计算平方距离并更新 `distance[i]=min(distance[i], current)`；
- F=4 路径缓存中心点四个分量，float32 使用 16B 装载，减少 GM 指令数；
- float16 的 F=4 路径使用展开后的原生 half 运算；F=16/32/64 路径每 8 个
  特征显式执行一次 round-to-nearest 检查点；其余维度每一步显式舍入，以匹配
  任务书标杆数值路径；
- 每个物理线程只写自己负责点的 distance，无原子操作和写冲突。

### 块内归约与并列规则

每个物理线程产生一组 `(best_distance, best_global_index)`，写入 UB：

- `bestDistance[512]`：float32，共 2 KiB；
- `bestIndex[512]`：int64，共 4 KiB。

归约分三步：

1. 前 256 个线程分别合并同一逻辑 lane 的两个物理 worker；
2. 8 个 warp 使用 `asc_shfl_down` 在寄存器中完成 32-lane 归约；
3. 首个 warp 再归约 8 个 warp 的结果，lane 0 写出下一采样点。

比较首先选择更大距离。距离相同时，按 batch 内局部索引的
`(local_index % 256, local_index // 256)` 升序选择。这一点必须使用局部索引，
不能直接对全局索引取模，否则 batch 起点不是 256 的倍数时会改变 CUDA 的并列顺序。

### 内存与同步

- GM：`src`、`ptr`、`out_ptr`、`start`、`distance`、`out`；
- workspace：每个输入点一个与 `src` 同 dtype 的最小距离；
- UB：每 block 固定约 6 KiB，保留充足 DCache 空间；
- 距离初始化后同步一次；每轮再使用三次 `asc_syncthreads()`，分别保护 worker
  候选写入、warp 首项写入和下一轮读取；空 batch 由所有线程一致跳过，不会使
  部分线程滞留在屏障前。

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

1. `src` 必须连续并且首维表示点数；
2. `ratio` 应在 `(0, 1]`，标量或每 batch 一个值；
3. `ptr` 为单调非降 CSR 边界，首项为 0、末项为 N；
4. 输入 NaN/Inf 不属于任务书随机测试数据域；
5. 单个 batch 的采样轮次具有算法级串行依赖。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 功能 | PyTorch 层签名和分支行为与 `torch_cluster.fps` 一致 | 任务书 |
| 精度 | float16/float32 输出 int64 下标与 CPU 单标杆 bit-wise 一致 | 任务书、生态算子开源精度标准 |
| 性能 | 任务书每个性能用例均不低于 0.45 倍标杆性能 | 任务书 |

自测覆盖单/多 batch、ptr Tensor/List、per-batch ratio、空输入、随机首点、
大规模形状和距离平局。除常规 256-lane 平局外，单独覆盖 batch 起点非 256 对齐
场景，以验证局部 lane 语义。性能脚本覆盖任务书 21 个 shape/ratio 组合，并分别
测试 float16 和 float32。

在 Ascend 950 + CANN 9.1.0 环境按任务书脚本回归：官方功能/精度用例
61/61 通过，仓内扩展用例 25/25 通过；42 个 dtype/shape/ratio 性能项均满足
0.45 门槛，本轮最小性能比（标杆耗时/实测耗时）为 0.481。

## 兼容性分析

- 通过 `torch.library` 注册 `torch_cluster::fps` 的 PrivateUse1 实现；
- 若进程已加载 torch_cluster 的 CPU/CUDA Kernel，不覆盖原有 dispatch；
- 保留四个 TorchScript overload 和全部公开参数；
- 输出始终位于输入所在 NPU，dtype 固定为 int64。
