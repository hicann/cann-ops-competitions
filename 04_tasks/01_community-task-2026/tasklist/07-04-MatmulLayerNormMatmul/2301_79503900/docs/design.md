# MatmulLayerNormMatmul 算子设计文档

# 需求背景

## 需求来源

本需求来源于 CANN 2026 年 7 月社区任务 `MatmulLayerNormMatmul`。任务要求在 Ascend 950 上使用 CATLASS 和 Ascend C 实现单 kernel 融合算子，在一次 kernel launch 中完成 `Matmul -> LayerNorm -> Matmul`，并提交样例代码、测试交付件、自验证报告、README 和设计文档。

任务文档：

<https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/docs/202607/MatmulLayerNormMatmul_task_doc.md>

目标仓库：

<https://gitcode.com/cann/catlass>

## 背景介绍

### MatmulLayerNormMatmul 算子实现优化

Transformer、MLP 等模型中存在连续的矩阵乘、行级归一化和矩阵乘计算：

```python
c0 = torch.mm(a0, b0)
c0_norm = torch.nn.functional.layer_norm(
    c0, normalized_shape=(n0,), weight=gamma, bias=beta, eps=1e-6
)
c1 = torch.mm(c0_norm, b1)
```

小算子拼接方案至少包含三次 kernel launch，并需要把 `C0` 和 `C0_norm` 写入 GM 后再读取。任务测试中 `N0` 最大为 `8192`，完整中间矩阵会带来明显的 GM 带宽、缓存和 launch 开销。

本设计使用 Ascend 950 MIX kernel：AIC 完成两个 Matmul，AIV 完成 LayerNorm；第一个 Matmul 的 FP16 结果通过 `L0C -> UB -> L1` 保存在片上，LayerNorm 统计和归一化均在 UB/L1 内完成，第二个 Matmul直接读取片上的归一化结果。主路径不在 GM 中保存或读取 `C0/C0_norm`。

### 实现路径和参考组件

CATLASS 定位为算子样例库，本任务按 example 组织，不采用传统自定义算子的 `op_host/op_kernel` 目录，也不新增运行时自动选择 kernel 的机制。参考 `examples/64_ascend950_matmul_evg` 的静态 TileShape 样例组织方式，在同一个 example 目录内提供多套 kernel 实现方案及对应 host 文件；不同方案通过不同 host 源文件或编译目标显式选择，TileShape 作为模板静态常量固化在对应实现中。性能自测时按 case 主动选择不同 host/kernel 并调整 TileShape，记录最终保留方案和实测依据。

新增目录编号以提交时 CATLASS 主干可用编号为准，以下用 `<id>` 表示：

```text
examples/<id>_matmul_layer_norm_matmul/
  CMakeLists.txt
  README.md
  gen_data.py                                                   # 如采用离线 golden
  matmul_layer_norm_matmul_kernel.hpp                           # 样例私有 kernel 组件
  matmul_layer_norm_matmul_host_base.cpp                        # 常规静态 TileShape
  matmul_layer_norm_matmul_host_small_m.cpp                     # 小 M0 并行优化
  matmul_layer_norm_matmul_host_large_n.cpp                     # 大 N0 资源优化
  matmul_layer_norm_matmul_host_tail.cpp                        # 尾块/非对齐保守方案
tests/optest/kernels/<id>_matmul_layer_norm_matmul/              # 如任务交付要求接入 optest
tests/optest/torch_catlass/ops/matmul_layer_norm_matmul.py
tests/optest/tests/test_<id>_matmul_layer_norm_matmul.py
```

主要参考：

| 参考项 | 用途 |
| --- | --- |
| `examples/64_ascend950_matmul_evg` | Ascend 950 静态 TileShape kernel、example host 组织和手动调参方式 |
| `examples/44_quant_matmul_full_loadA_tla` | CATLASS Device 组装、TileShape 和 workspace 接口 |
| `examples/73_ascend950_matmul_full_loadA` | Ascend 950 Matmul 和 `Arch::Ascend950` 配置 |
| `examples/80_grouped_matmul_slice_m_gelu` | `L0C -> UB`、AIC/AIV 跨核 flag、UB 双缓冲 |
| `docs/zh/1_Practice/others/FA_kernel_optimization.md` | `UB -> L1` 片上中转和多阶段 MIX 流水 |
| `tests/optest` | Torch 注册、Python wrapper 和 pytest 接入，不作为运行时 kernel 选择机制 |
| CATLASS PR #678 | 样例与 optest 同步交付的文件范围和测试证据 |

### 算子现状分析

该任务为新增融合样例，不是历史 TBE 算子的语义迁移。外部接口如下：

| 参数 | 输入/输出 | 含义 | dtype | 逻辑形状 | 物理布局/stride |
| --- | --- | --- | --- | --- | --- |
| `A0` | 输入 | 第一个 Matmul 左矩阵 | FP16 | `(M0, K0)` | RowMajor，stride `(K0, 1)` |
| `B0` | 输入 | 第一个 Matmul 右矩阵 | FP16 | `(K0, N0)` | ColumnMajor，stride `(1, K0)` |
| `B1` | 输入 | 第二个 Matmul 右矩阵 | FP16 | `(N0, M1)` | ColumnMajor，stride `(1, N0)` |
| `gamma` | 输入 | LayerNorm 缩放参数 | FP32 | `(N0,)` | 连续 |
| `beta` | 输入 | LayerNorm 偏置参数 | FP32 | `(N0,)` | 连续 |
| `C1` | 输出 | 第二个 Matmul 输出 | FP16 | `(M0, M1)` | RowMajor，stride `(M1, 1)` |

说明：从 Torch 视角，ColumnMajor 的 `B0/B1` 可以表现为具有列主 stride 的二维 Tensor，不能简单等同于“所有输入必须 contiguous”。Host 侧应校验任务要求的逻辑 shape 和实际 stride，不能把 B 矩阵错误地按 RowMajor 解释。

算子公式：

```text
C0 = A0 * B0
mean[m] = sum(C0[m, :]) / N0
var[m] = sum((C0[m, :] - mean[m])^2) / N0
C0_norm[m, n] = (C0[m, n] - mean[m]) * rsqrt(var[m] + 1e-6) * gamma[n] + beta[n]
C1 = C0_norm * B1
```

### 算子功能分析

- 输入：`A0`、`B0`、`B1`、`gamma`、`beta`。
- 输出：`C1`，不暴露 `C0`、`C0_norm`、`mean`、`variance` 或 `invstd`。
- 矩阵输入和输出 dtype：FP16。
- LayerNorm 参数和统计 dtype：FP32。
- LayerNorm 归一化轴：`C0` 最后一维，即每行长度 `N0`。
- `eps`：固定为 `1e-6`，不增加外部属性。
- 广播：仅 `gamma/beta` 按行应用，不支持其他广播语义。
- kernel launch：一次。

# 需求分析

## 需求描述

使用 Ascend C/CATLASS 在 Ascend 950 上实现 `MatmulLayerNormMatmul`，满足以下结果：

1. 一次 MIX kernel launch 完成两个 Matmul 和中间 LayerNorm。
2. `C0/C0_norm` 仅在片上 L1/UB 中存在，不进行中间 GM 读写。
3. `mean/variance/invstd` 由 kernel 内部 UB scratch 管理，不作为输出。
4. 功能和精度通过任务测试集及生态算子开源精度标准。
5. 相对 `torch.mm + F.layer_norm + torch.mm` 达到任务要求的 `1.1x` 性能。
6. 按 CATLASS 当前规范同时交付 example、多静态 host 自测入口、必要的 optest 接入、README、设计文档和自验证证据。

## 需求拆解

| 编号 | 子需求 | 设计响应 | 验证方式 |
| --- | --- | --- | --- |
| R1 | 固定 dtype 契约 | A0/B0/B1/C1 FP16；gamma/beta FP32 | Host 负向测试、pytest dtype 断言 |
| R2 | shape 和布局正确 | 校验四组矩阵维度关系及指定 stride | shape/stride 正反向测试 |
| R3 | 单 launch 融合 | 单个 `__mix__` kernel，AIC/AIV 配对执行 | msprof kernel 列表、代码审查 |
| R4 | 无 C0 中间显存读写 | `C0/C0_norm` 驻留 L1，UB 统计和归一化 | 代码审查、Memory/流水分析 |
| R5 | 行级 LayerNorm | 每个 row block 完整持有 N0，FP32 两遍统计 | golden 对比、常量行测试 |
| R6 | 任务 shape 覆盖 | 官方 CSV 116 个 case 全覆盖 | 参数化 pytest 和汇总日志 |
| R7 | 精度达标 | 中间 Matmul/LN 输出按 FP16 语义，统计 FP32 | MERE/MARE、NaN/Inf 检查 |
| R8 | 性能达标 | 片上复用、AIC/AIV 流水、shape 分档 | `msprof op` 逐 case 数据 |
| R9 | 可合入性 | example、静态 kernel/host 自测、必要的 optest 接入和文档同步 | clean build、pytest、PR checklist |

## 验收边界

任务 CSV 共 116 个 case：

| 维度 | 取值 |
| --- | --- |
| `M0` | `128, 512, 1024, 2048` |
| `K0` | `768, 2048, 4096, 8192` |
| `N0` | `2048, 3072, 4096, 8192` |
| `M1` | `768, 2048, 4096` |

基线耗时范围约为 `19.005 us` 到 `1188.58 us`，中位数约 `78.329 us`，P90 约 `240.268 us`。测试集的 `K0/N0/M1` 均能被 `256` 整除，但实现仍需要覆盖尾块，避免把测试集对齐特征写成外部接口限制。

# 详细设计

## 算子分析

### 数学公式

第一阶段：

```text
C0[m, n] = sum(A0[m, k] * B0[k, n]), k = 0..K0-1
```

第二阶段：

```text
mean[m] = (1 / N0) * sum(C0[m, n])
var[m] = (1 / N0) * sum((C0[m, n] - mean[m])^2)
invstd[m] = rsqrt(var[m] + 1e-6)
C0_norm[m, n] = (C0[m, n] - mean[m]) * invstd[m] * gamma[n] + beta[n]
```

第三阶段：

```text
C1[m, p] = sum(C0_norm[m, n] * B1[n, p]), n = 0..N0-1
```

### dtype 语义

| 阶段 | 输入/输出 dtype | 计算 dtype |
| --- | --- | --- |
| 第一个 Matmul | FP16 -> FP16 `C0` | Cube FP32 累加，FixPipe 转 FP16 |
| LayerNorm 均值/方差 | FP16 `C0` -> FP32 stats | FP32 |
| LayerNorm 输出 | FP16 `C0_norm` | FP32 归一化和仿射，写 L1 前转 FP16 |
| 第二个 Matmul | FP16 x FP16 -> FP16 `C1` | Cube FP32 累加，FixPipe 转 FP16 |

选择 FP16 `C0/C0_norm` 有两点原因：

1. 与 PyTorch 链路中 FP16 `torch.mm` 和 FP16 LayerNorm 输出的预期 dtype 传播一致。
2. 使完整 row block 能驻留 512 KiB L1，避免 GM 中间结果。

在 Ascend 950 环境开始编码前，必须用一个 smoke case 打印 `C0.dtype` 和 `C0_norm.dtype`，锁定 NPU PyTorch golden 的实际行为。若 golden 与上述语义不一致，先修订本节和精度路径，再实现 kernel，不能通过放宽阈值掩盖语义差异。

### 数值稳定性

主方案不使用 `E[x^2] - E[x]^2` 计算方差，因为当均值较大、方差较小时容易发生抵消。利用 `C0` 已保存在 L1 的条件，采用 FP32 两遍统计：

```text
Pass A: sum(C0) -> mean
Pass B: sum((C0 - mean)^2) -> var -> invstd
Pass C: normalize + affine，并把 FP16 C0_norm 原位写回 L1
```

方差除数使用总体方差定义 `N0`；在 `rsqrt` 前执行 `var = max(var, 0.0f)`，再加 `1e-6`。

## 算子实现

### 实现方案概述

主方案是“row block 完整片上驻留”的 Ascend 950 MIX kernel。每个 AIC/AIV 配对任务拥有一个 `M` 方向行块，并完整覆盖该行块的 `N0`：

```text
GM A0/B0
   -> AIC: Matmul0, FP32 accumulate
   -> FixPipe: FP16 C0 tile
   -> AIV UB: accumulate FP32 row sum
   -> AIV: UB -> L1, cache full C0 row block

L1 C0
   -> AIV: centered variance pass
   -> AIV: normalize + gamma/beta
   -> AIV: overwrite L1 with FP16 C0_norm

L1 C0_norm + GM B1
   -> AIC: Matmul1, FP32 accumulate
   -> FixPipe: FP16 C1
   -> GM C1, write once
```

该 ownership 保证：

- 每一行的统计量在一个任务内完成，不需要跨 AIC 的全局归约。
- `C0/C0_norm` 不落 GM。
- 归一化后的完整行块可被不同 `M1` 输出列 tile 重复读取，不重算 Matmul0。
- 不同任务写 `C1` 的区域互不重叠，不需要 atomic add。

### 备选方案及结论

| 方案 | 结论 | 原因 |
| --- | --- | --- |
| 完整/局部 `C0` 写 GM workspace | 不采用 | 与“无需中间显存读写”的任务目标冲突 |
| 不保存 `C0`，先统计后重算 | 仅保留为调试思路，不进入提交路径 | 至少重复一次 Matmul0；按 M1 分块时还可能重复更多次，性能模型不成立 |
| 一遍 `sum/sumsq` 方差 | 不作为默认路径 | FP32 仍存在抵消风险，精度风险高于两遍中心化方差 |
| 每个核只持有部分 N0 | 不采用 | 第二个 Matmul 需要跨核归约或 partial C1 GM workspace |

## 3.2.1 Host 侧设计（静态 TileShape）

### 接口和样例组织

该样例归类为 Ascend 950 matmul family 融合 example。Host 侧不实现运行时 kernel selector，也不通过 `PlanId` 或 JIT 宏在一次入口中自动选择方案。每个 host 文件绑定一组静态 kernel 类型，kernel 模板参数固化 TileShape、流水深度、scheduler 和 replica 策略；自测时通过选择不同可执行目标或 host 源文件来覆盖不同 case。

运行时参数保持轻量结构，只描述 problem shape 和 GM 地址：

```cpp
struct MatmulLayerNormMatmulProblemShape {
    uint32_t m0;
    uint32_t k0;
    uint32_t n0;
    uint32_t m1;
};

struct MatmulLayerNormMatmulParams {
    void* a0;
    void* b0;
    void* b1;
    void* gamma;
    void* beta;
    void* c1;
    MatmulLayerNormMatmulProblemShape shape;
};

template <class Kernel>
void RunMatmulLayerNormMatmul(
    uint32_t blockNum,
    aclrtStream stream,
    const MatmulLayerNormMatmulParams& params);
```

字段约定：

```text
shape.m0 = M0
shape.k0 = K0
shape.n0 = N0
shape.m1 = M1
a0/b0/b1/gamma/beta = 输入 GM 地址
c1                  = 输出 GM 地址
```

`eps` 固定为 kernel 编译期常量 `1e-6f`，不新增外部参数。矩阵和参数 dtype 固定，TileShape、scheduler 和流水配置均由当前 host 绑定的静态 kernel 类型决定。

如任务交付要求接入 optest，Python wrapper 只暴露固定功能入口，并在测试配置中显式绑定被测静态 kernel 目标，不在 wrapper 内增加运行时 kernel 自动选择逻辑：

```python
torch_catlass.matmul_layer_norm_matmul(a0, b0, b1, gamma, beta) -> c1
```

### Host 参数校验

Torch adapter 在分配输出和启动 kernel 前执行：

1. 五个输入均在 NPU，且位于同一 device。
2. `A0/B0/B1` 为 FP16，`gamma/beta` 为 FP32。
3. `A0/B0/B1` 为二维，`gamma/beta` 为一维。
4. `A0.shape[1] == B0.shape[0]`。
5. `B0.shape[1] == B1.shape[0] == gamma.numel() == beta.numel()`。
6. `A0` 为 RowMajor stride，`B0/B1` 为 ColumnMajor stride，`gamma/beta` 连续。
7. 所有维度为正，转换到 `uint32_t` 前检查范围。
8. 输出固定分配为 `(M0, M1)`、FP16、RowMajor。

输入地址使用 `tensor.data_ptr()` 获取逻辑首元素，不能直接使用 `tensor.storage().data()`，否则带 storage offset 的合法 view 会读取错误位置。

校验失败使用 `TORCH_CHECK` 返回明确错误，不启动 kernel。

### Tile 和资源预算

CATLASS 当前 `Arch::Ascend950` 资源常量：

| 存储 | 容量 |
| --- | --- |
| L1 | 512 KiB |
| UB | 248 KiB |
| L0A | 64 KiB |
| L0B | 64 KiB |
| L0C | 256 KiB |

定义：

```text
BM      = 每个 row task 的行数
BN0     = Matmul0/LN 的 N0 tile，初始候选 256
BK0     = Matmul0 的 K0 tile，候选 128/256
BN1     = Matmul1 的 M1 输出 tile，初始候选 256
BK1     = Matmul1 的 N0 reduction tile，候选 128/256
N0a     = RoundUp(N0, 16)  # FP16 32B 对齐
```

初始预算使用 `L1A_STAGES=1`、`L1B_STAGES=2`、`UB_STAGES=2`，并分别为 L1/UB 预留 `32 KiB/16 KiB` 的对齐和组件元数据空间。若具体 Block 组件的静态分配不同，必须使用组件真实常量重新计算，不能沿用该估算。

L1 必须满足分阶段最大占用：

```text
CACHE_BYTES = BM * N0a * sizeof(half)

L1_PHASE0 = CACHE_BYTES
          + L1A_STAGES * BM * BK0 * sizeof(half)
          + L1B_STAGES * BK0 * BN0 * sizeof(half)
          + ALIGN_GUARD

L1_PHASE1 = CACHE_BYTES
          + L1B_STAGES * BK1 * BN1 * sizeof(half)
          + ALIGN_GUARD

max(L1_PHASE0, L1_PHASE1) <= 512 KiB
```

UB 必须满足：

```text
UB_BYTES = UB_STAGES * BM_VEC * BN0 * sizeof(half)   # C0 tile ping-pong
         + BM_VEC * BN0 * sizeof(float)              # FP32 LN work
         + 2 * BN0 * sizeof(float)                   # gamma/beta
         + 4 * BM * sizeof(float)                    # sum/mean/var/invstd
         + ALIGN_GUARD

UB_BYTES <= 248 KiB
```

L0C 约束：

```text
max(BM * BN0, BM * BN1) * sizeof(float) <= 256 KiB
```

所有候选配置必须用 `static_assert` 检查静态资源，用 `CanImplement` 检查运行时 shape；不能只依赖文档估算。

初始静态 TileShape 候选分档如下，实际提交值由 116 case profiling 决定：

| 静态候选 | N0 范围 | BM 候选 | C0 cache | BK0/BK1 候选 | 目标 |
| --- | --- | --- | --- | --- | --- |
| S0 | `N0 <= 2048` | `64` | `<= 256 KiB` | `128` | 大行块、提高 Cube 效率 |
| S1 | `N0 == 3072` | `32` | `192 KiB` | `128/256` | 平衡 L1 和并行度 |
| S2 | `N0 == 4096` | `32` | `256 KiB` | `128` | 保证完整行块驻留 |
| S3 | `N0 == 8192` | `16` | `256 KiB` | `128` | 大 N0 专用路径 |
| S4 | 小 `M0` 并行档 | `S0-S3` 的一半 | `<= 128 KiB` | `128/256` | 增加 row task 数，降低空闲核数 |
| S5 | 通用尾块 | 固定保守 BM，按 16 对齐 | `<= 128 KiB` | `128` | 覆盖非测试集对齐 shape |

这里的 `BM` 是设计候选，不是已验证的最优值。PR 中必须附带候选对比数据后才能删除其他候选或固化默认值。

按上述初始 stage、`BN0=BN1=256` 和 L1 guard 复算，静态预算为：

| 候选 | Cache | L1 Phase 0 | L1 Phase 1 | L0C max | 结论 |
| --- | ---: | ---: | ---: | ---: | --- |
| S0，`BK=128` | 256 KiB | 432 KiB | 416 KiB | 64 KiB | 通过 |
| S1，`BK=128` | 192 KiB | 360 KiB | 352 KiB | 32 KiB | 通过 |
| S1，`BK=256` | 192 KiB | 496 KiB | 352 KiB | 32 KiB | 余量较小，仅在组件真实静态分配通过时保留 |
| S2，`BK=128` | 256 KiB | 424 KiB | 416 KiB | 32 KiB | 通过 |
| S3，`BK=128` | 256 KiB | 420 KiB | 416 KiB | 16 KiB | 通过 |

UB 按最保守的 `BM_VEC=BM=64` 估算约为 `147 KiB`，低于 `248 KiB`；实际 MIX kernel 若每个 AIV sub-block 处理 `BM/2`，占用会更低。该表只证明初始候选在容量上有实现空间，不代表寄存器、event 或流水性能已经通过。

### 分核和任务 ownership

基础分核只沿 `M0` 行块切分：

```text
row_blocks = CeilDiv(M0, BM)
row_task_id = task_id / replica
replica_id  = task_id % replica
```

当 `row_blocks >= aic_core_num` 时，`replica=1`，每个任务完成该行块的全部 `M1` 输出列。

当小 `M0` 导致 row task 不足时，可复制同一 row block，并把 `M1` 输出列块分配给不同 replica。各 replica 独立生成片上 `C0_norm`，但只写自己的 `C1` 列区间：

```text
col_blocks = CeilDiv(M1, BN1)
replica <= min(CeilDiv(aic_core_num, row_blocks), col_blocks)
replica r handles col block: r, r + replica, r + 2 * replica, ...
```

复制会增加 Matmul0 开销，不能仅按“满核”选择。Host 侧在不同静态方案的 `replica` 候选中比较以下估算，并用实测系数校准：

```text
F0 = 2 * BM * K0 * N0
F1 = 2 * BM * N0 * M1
estimated_work(replica) = replica * F0 + F1
```

原则：`K0` 大或 Matmul0 占主导时优先 `replica=1`；`M0` 小、`K0` 较小且 `M1` 大时才考虑复制。任何两个 task 的 `C1` 写区间必须互斥。

### 静态 kernel 方案组织

本设计不使用传统 `op_host` 的 `SetTilingKey`，也不引入 `PlanId`、JIT 宏或运行时算子选择器。不同优化路径以独立静态 kernel 类型和 host 文件组织，CMake 中形成可显式运行的 example 目标：

| 静态方案 | 对应 host/目标 | kernel 差异 | 自测选择原则 |
| --- | --- | --- |
| `base` | `matmul_layer_norm_matmul_host_base.cpp` | 256 KiB cache 档，`replica=1` | 常规 shape、吞吐优先 |
| `small_m` | `matmul_layer_norm_matmul_host_small_m.cpp` | 128 KiB cache 档，可启用 replica | 小 `M0`、row task 不足 |
| `large_n` | `matmul_layer_norm_matmul_host_large_n.cpp` | 更小 BM，降低 L1/UB 峰值 | `N0=8192` 或资源余量紧张 |
| `tail` | `matmul_layer_norm_matmul_host_tail.cpp` | 保守 TileShape，完整边界保护 | 非对齐或补充边界 shape |

示例 CMake 组织：

```cmake
catlass_example_add_executable(<id>_matmul_layer_norm_matmul_base mix matmul_layer_norm_matmul_host_base.cpp)
catlass_example_add_executable(<id>_matmul_layer_norm_matmul_small_m mix matmul_layer_norm_matmul_host_small_m.cpp)
catlass_example_add_executable(<id>_matmul_layer_norm_matmul_large_n mix matmul_layer_norm_matmul_host_large_n.cpp)
catlass_example_add_executable(<id>_matmul_layer_norm_matmul_tail mix matmul_layer_norm_matmul_host_tail.cpp)
```

方案数量保持最小化。只有 profiling 证明稳定收益且覆盖测试充分时才新增 host/kernel 组合，禁止为每个 CSV case 硬编码一个目标。

### GM workspace

主路径的用户 GM workspace 大小为 `0`：

- `C0/C0_norm`：L1。
- `sum/mean/var/invstd`：UB。
- AIC/AIV 同步：硬件 cross-core flag。
- 唯一 GM 写回：最终 `C1`。

如果 CATLASS Device/launcher 启动接口需要 workspace 参数，仍保留统一接口，但主路径用户 workspace 大小返回 `0`。禁止为了复用现有 multistage workspace kernel 而把 `C0` 写入 GM。

## 3.2.2 Kernel 侧设计

### Kernel 类型和组件

kernel 使用 Ascend 950 MIX 模式，参考 `MmadPreloadAsyncWithCallbackL0CToUB` 和 FAI 的 `CopyUb2L1Tla` 数据通路：

```text
ArchTag            = Arch::Ascend950
ElementA0/B0/B1    = half
ElementGamma/Beta  = float
ElementAccumulator = float
ElementC0/C0Norm   = half
ElementC1          = half
LayoutA0/C0/C1     = RowMajor
LayoutB0/B1        = ColumnMajor
```

example 的 CMake 目标类型配置为 `mix`；如接入 optest，其 wrapper 也必须绑定 MIX 类型的静态 kernel 目标，不能误注册为仅 AIC 的 `cube` kernel。

建议的组件边界：

| 组件 | 职责 |
| --- | --- |
| `MatmulLayerNormMatmul` kernel | 参数解析、任务调度、阶段编排和同步 |
| `BlockMmad0` | `A0 x B0`，L0C 输出回调到 UB |
| `BlockLayerNorm` | FP32 sum、centered variance、normalize、gamma/beta |
| `CopyUbToL1` | 把 FP16 `C0/C0_norm` 写入 AIC 可读 L1 区域 |
| `BlockMmad1` | 使用 L1-resident 左操作数路径读取 `C0_norm`，与 GM `B1` 相乘并写 `C1` |

若 LayerNorm 组件仅服务本样例，可先放在 kernel 私有实现中；确认可复用且接口稳定后再下沉到公共 `epilogue/block` 或 `epilogue/tile`，避免为单一场景过早扩展公共 API。

`BlockMmad1` 不能原样套用“左矩阵从 GM 搬入 L1”的通用 Matmul 路径。需要复用或抽取 FAI/PV 中 L1-resident A 操作数的 copy/迭代方式，使 AIC 直接消费 AIV 已写入 L1 的 `C0_norm`；该适配必须保持 B1 的 ColumnMajor 读取和 L0A/L0B/L0C 资源约束。

### Init 阶段

1. 解析 `M0/K0/N0/M1`、当前静态方案的 TileShape 和 task 映射。
2. 设置 A0/B0/B1/gamma/beta/C1 的 GM Tensor 和 TLA layout。
3. 从 `Arch::Resource<ArchTag>` 按固定 offset 划分 L1、UB、L0A/L0B/L0C。
4. 初始化 AIC->AIV 和 AIV->AIC 的 stage flag；AIC/AIV 两侧的 set/wait 次数必须严格配对。
5. 计算 row block、有效 `actualM` 和当前 replica 负责的 M1 列块范围。

### Phase 0：Matmul0、C0 缓存和均值

AIC：

1. 按 `BK0/BN0` 读取 A0/B0，执行 Matmul0。
2. FP32 累加完成后通过 FixPipe 转成 FP16 RowMajor C0 tile。
3. 使用 L0C-to-UB 回调写入当前 ping-pong stage。
4. `CrossCoreSetFlag` 通知配对 AIV，复用 stage 前等待 AIV idle。

AIV：

1. 等待 C0 tile ready。
2. C0 tile FP16 转 FP32，沿 N0 对每行做分块归约并累计 `rowSum`。
3. 使用 `CopyUb2L1Tla` 把 FP16 C0 tile 写到 `L1_C0_BASE + nOffset`。
4. 释放 stage，通知 AIC 继续。
5. 完成全部 N0 tile 后计算 `mean = rowSum / N0`。

尾块必须通过 actual shape/mask 参与归约，padding 元素不能计入分母。

### Phase 1：方差

AIV 从 L1 重新读取 C0：

```text
for nTile in N0:
    x = cast_fp32(load_l1_c0(nTile))
    centered = x - mean[row]
    rowVarSum += reduce_sum(centered * centered)
var = max(rowVarSum / N0, 0)
invstd = rsqrt(var + 1e-6)
```

`rowVarSum/mean/var/invstd` 均为 FP32。Vector API 的 repeat 次数若超过接口上限，必须分批执行，不能把大循环次数截断为 `uint8_t`。

### Phase 2：归一化和片上回写

AIV 按 BN0 tile 执行：

```text
x = cast_fp32(load_l1_c0(tile))
y = (x - mean[row]) * invstd[row]
y = y * gamma[col] + beta[col]
store_l1_c0(tile, cast_fp16(y))
```

复用同一 L1 区域，Phase 2 结束后该区域语义由 `C0` 变为 `C0_norm`。gamma/beta 每个 tile 搬入 UB 并在 BM 行间复用。

初版实现可以在 Phase 2 完成后统一通知 AIC；性能版本可让 AIV 归一化 tile `t+1` 时，AIC 消费已完成的 tile `t`。引入流水后必须证明：

- AIC 不读取尚未归一化的 tile。
- AIV 不覆盖 AIC 尚未消费的 stage。
- kernel 尾部 drain 完整，不遗留未配对 flag。

### Phase 3：Matmul1 和输出

AIC 对当前 replica 负责的每个 `M1` 列 tile：

1. 以 L1 中 `C0_norm[actualM, N0]` 为左矩阵。
2. 从 GM 读取 ColumnMajor `B1[N0, BN1]`。
3. 沿 N0 以 `BK1` 累加 FP32 L0C。
4. 通过 FixPipe 转 FP16，只向最终 `C1` 写回一次。

`C0_norm` 对多个 M1 tile 保持 L1 resident。M1 尾块使用 actual shape 写回，禁止越界。

### 同步与死锁约束

1. 配对 AIC/AIV 使用固定 flag 范围，不能与 CATLASS 组件内部 flag 冲突。
2. 每个 stage 的 ready/idle set 和 wait 次数在两侧完全一致。
3. 无有效 row task 的核不参与 launch；如果某静态方案使用全核 barrier，则所有已启动核必须无条件到达相同 barrier。
4. 调试构建记录 stage/task/flag，生产构建移除 device print。
5. kernel 退出前执行必要的 BlockMmad drain 和 `PipeBarrier<PIPE_ALL>`。

### 尾块和对齐

- 测试集 shape 均高度对齐，但 kernel 使用 `CeilDiv` 和 actual block shape。
- FP16 GM/L1 搬运按 32B 对齐，不能确定对齐时使用 CATLASS 的边界 copy 或 `DataCopyPad` 等安全路径。
- L1 区域和 stage offset 至少按 CATLASS 组件要求对齐；profiling 时同时检查 512B 对齐对 MTE/FixPipe 的影响。
- padding 只用于存储和计算对齐，不改变 LayerNorm 的 `N0` 分母。

## 支持硬件

| 芯片版本 | 支持 |
| --- | --- |
| Ascend 950 | 是 |
| Atlas A2/A3 | 否，本设计依赖 Ascend 950 的资源和 MIX 数据通路 |

## 算子约束限制

任务书给出的约束为“无”。本实现不额外声明 shape 白名单，但外部契约仍要求：

1. dtype、rank、shape 关系和布局符合参数表。
2. 所有逻辑维度为正且可由运行时参数类型表示。
3. `eps` 固定为 `1e-6`。
4. `gamma/beta` 长度严格等于 `N0`。
5. 功能路径处理尾块；性能只对任务 116 个 case 承诺验收目标。

# 可维可测分析

## 精度标准/性能标准

| 验收项 | 标准 | 来源 |
| --- | --- | --- |
| 功能 | 官方 116 case 全部通过，无 crash/timeout/NaN 异常 | 任务书和自测 CSV |
| 精度 | FP16 输出 MERE `< 2^-10`，MARE `< 10 * 2^-10`；同时检查 NaN/Inf 位置 | 生态算子开源精度标准 |
| 性能 | 相对 PyTorch 小算子拼接达到 `1.1x` | 任务书 |
| 融合 | 一次目标 kernel launch，`C0/C0_norm` 无 GM 中转 | 任务功能要求 |

误差定义：

```text
relative_error = abs(actual - golden) / (abs(golden) + 1e-7)
MERE = mean(relative_error)
MARE = max(relative_error)
```

对于接近零的 golden，同时记录最大绝对误差，避免只看相对误差造成误判。不得通过放宽阈值替代根因分析。

性能定义：

```text
speedup = torch_baseline_us / catlass_kernel_us
```

任务书使用“整体性能 1.1x”表述。内部自验采用更保守的逐 case `speedup >= 1.1`，并额外报告算术平均、几何平均、P50 和最差 case；若任务维护者明确采用其他聚合口径，再同步调整验收表。

## 测试设计

### Golden

主 golden 必须保持任务链路和 dtype，不用全 FP32 结果替代：

```python
c0 = torch.mm(a0, b0)
c0_norm = torch.nn.functional.layer_norm(
    c0, (n0,), weight=gamma, bias=beta, eps=1e-6
)
golden = torch.mm(c0_norm, b1)
```

测试同时断言 `c0/c0_norm/golden` 的 dtype，防止框架版本变化导致精度口径漂移。

### 功能和边界用例

| 类别 | 覆盖 |
| --- | --- |
| 官方主集 | CSV 116 case 全量参数化 |
| 静态候选边界 | 每个 S0-S5 至少 1 个 case |
| M/N/M1 尾块 | 不能被 BM/BN0/BN1 整除的补充 shape |
| 最小合法 shape | 小矩阵 smoke，用于接口和尾块 |
| 零方差 | A0/B0 构造常量行，验证 `eps` 和无 NaN |
| 数值范围 | 正负随机、近零、较大有限值 |
| gamma/beta | gamma=1 beta=0；gamma=0；随机 FP32 |
| 负向接口 | dtype、rank、shape、stride、device 不匹配 |

随机测试固定 seed，并在失败日志中打印 seed、shape、静态 kernel 方案、TileShape、MERE/MARE 和首个错误位置。

### optest 接入验证

```bash
cd tests/optest
bash build.sh
python -c "import torch_catlass; print(hasattr(torch_catlass, 'matmul_layer_norm_matmul'))"
pytest tests/test_<id>_matmul_layer_norm_matmul.py -v -s
```

无 Ascend 950 时仅允许通过 `@only_on_3510` 跳过运行测试；编译、导入和 ABI 问题不能用 skip 掩盖。

### 性能验证

小 shape 提高预热次数，避免未提频影响：

```bash
msprof op --application ./<id>_matmul_layer_norm_matmul_base \
  --warm-up 30 --output ./prof/<case_id>
```

逐 case 记录：

```text
idx,M0,K0,N0,M1,baseline_us,kernel_us,speedup,
kernel_variant,BM,BN0,BK0,BN1,BK1,replica,precision_pass
```

性能分析至少查看：

- AIC/AIV、MTE2/MTE3、FixPipe 占比。
- AIC 等待 AIV 和 AIV 等待 AIC 的空泡。
- 每核任务数和长尾。
- L1/UB 资源、bank conflict、L2 命中和 GM 带宽。
- Phase 0/1/2/3 的热点和首尾 flush。

任何 TileShape、replica 或流水深度调整都需要保留前后对比数据。切换静态 kernel 或修改 TileShape 后执行 clean rebuild，并核对运行日志中的目标名和参数，避免旧二进制污染结果。


## 兼容性分析

本算子是 CATLASS 新增融合样例，不替换已有 API。兼容性要求：

1. 不修改既有 Matmul/LayerNorm 行为。
2. 新增 example 私有 `MatmulLayerNormMatmulParams`，不改变既有 Matmul 参数结构语义。
3. Python API 如需新增导出，仅作为测试入口，不影响现有 wrapper。
4. 固定 dtype 和 Ascend 950 架构通过编译期/运行时检查隔离。
5. 若公共 LayerNorm tile/block 组件后续被其他 kernel 复用，其模板参数和资源约束必须有独立单测。

# 风险与应对

| 风险 | 触发信号 | 应对和退出条件 |
| --- | --- | --- |
| 小 BM 导致 Cube 利用率低 | AIC ArithmeticUtilization 低、M tile 大量 padding | 对比 128/256 KiB cache 档；调整 BM 和 replica，不破坏完整 N0 驻留 |
| L1 cache 挤压 A/B 流水 | MTE2 高、L1 stage 无法双缓冲 | 降低 BK/BN 或 BM；按 phase 复用 L1；以 `static_assert` 为硬门禁 |
| AIV LayerNorm 成为瓶颈 | AIC 长时间等待 AIV | 调整 BN0、UB stage、归约实现；流水化 Phase 2/3 |
| 两遍方差增加 Vector 耗时 | Phase 1 占比过高 | 先验证精度；只有一遍算法同时满足全部精度 case 才允许替换 |
| 小 M0 核利用率不足 | `row_blocks < core_num` | 使用成本模型选择 S4/replica；不盲目满核 |
| ColumnMajor 解释错误 | 与 golden 大面积不一致 | 固定 stride 测试，检查 TLA Layout 和 wrapper 是否误调用 contiguous |
| golden dtype 与假设不一致 | smoke case 中 `C0_norm.dtype != FP16` | 暂停 kernel 实现，先更新 dtype 路径和 L1 预算 |
| flag 死锁 | kernel timeout/卡死 | 最小 stage 测试，核对两侧 set/wait；逐步增加流水，不带问题提交 |
| 19 us 小 case不达标 | launch/flush 占比高 | 精简静态方案、同步和初始化；提高预热并确认测量口径 |


# 参考资料

1. [MatmulLayerNormMatmul 任务书](https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/docs/202607/MatmulLayerNormMatmul_task_doc.md)
2. [CATLASS 创新样例开发流程](https://gitcode.com/cann/catlass/blob/master/docs/zh/1_Practice/10_innovative_example_development_guide.md)
3. [64_ascend950_matmul_evg](https://gitcode.com/cann/catlass/tree/master/examples/64_ascend950_matmul_evg)
4. [CATLASS optest README](https://gitcode.com/cann/catlass/blob/master/tests/optest/README.md)
5. [CATLASS 性能工具](https://gitcode.com/cann/catlass/blob/master/docs/zh/1_Practice/evaluation/performance_tools.md)
6. [CATLASS 参考样例 44](https://gitcode.com/cann/catlass/tree/master/examples/44_quant_matmul_full_loadA_tla)
7. [CATLASS 参考 PR #678](https://gitcode.com/cann/catlass/pull/678)
8. [生态算子开源精度标准](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md)
