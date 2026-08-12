# BatchedMatmulSoftmax 算子设计文档

# 需求背景（required）

## 需求来源

本设计对应 CANN 社区任务“BatchedMatmulSoftmax 算子开发”，目标是在
Ascend 950 上实现高性能融合算子：

```text
S = row_softmax(A x B)
```

算子需要融合 Batched Matmul 与 Softmax，减少中间数据搬运和 kernel
launch 开销，并相对官方“小算子标杆耗时”达到不低于 1.2x 的加速。

## 背景介绍

非融合实现通常由 Matmul 和 Softmax 两个算子组成。Matmul 将完整中间结果
写入 GM，Softmax 再从 GM 读取，同时产生两次 kernel launch。对于
`N <= 128` 的小矩阵场景，额外访存与 launch 开销占比较高；当 N 扩展到
512、1024 和 2048 时，单行数据无法继续沿用相同的 UB epilogue 策略。

本方案将 Matmul 放在 Cube Core 执行，将数值稳定的行 Softmax 放在
Vector Core 执行，并封装为单个 `MIX_AIC` kernel。根据 shape 动态选择
UB epilogue 路径或 GM 原地整行 Softmax 路径。

## 复用基础

- 复用 CATLASS 的 TLA Matmul、KernelAdapter 和 JIT 框架。
- 复用 CATLASS epilogue visitor 组合方式。
- 使用 Ascend C 向量指令完成 max、exp、sum、normalize 和 FP16 输出。
- 通过 Torch adapter 暴露 Python API，并使用 pytest 对比 PyTorch golden。

# 需求分析（required）

## 需求描述

首版支持以下输入输出：

- 输入 A：FP16，RowMajor，逻辑形状 `[batch, M, K]`。
- 输入 B：FP16，ColumnMajor，逻辑形状 `[batch, K, N]`。
- 输出 S：FP16，RowMajor，逻辑形状 `[batch, M, N]`。
- Softmax 维度：最后一维 N。
- 支持 `N = 64/128/512/1024/2048`。
- 支持 batch、多 M tile 和多核调度。

## 需求拆解

1. 使用 Cube Core 完成 FP16 Matmul 和 FP32 累加。
2. 对小 N 使用 L0C 到 UB 的片上 epilogue，避免 GM 中间张量。
3. 对长 N 使用输出 GM 暂存 Matmul 结果，并在同一 MIX kernel 中由
   Vector Core 原地完成整行 Softmax。
4. 使用 AIC 全局屏障和 AIC 到 AIV 通知保证长 N 路径的数据依赖。
5. 根据 shape 动态选择 tile、swizzle、block 数和 N=128 执行路径。
6. 接入 CATLASS JIT、Torch adapter 和官方 pytest。
7. 使用官方 180 case 测试集完成正确性和性能验证。

## 范围边界

当前提交聚焦社区任务的主要测试范围：

- 仅支持 FP16 输入输出。
- 仅支持 A RowMajor、B ColumnMajor、S RowMajor。
- Softmax 仅作用于最后一维。
- 当前优化路径支持 `N = 64/128/512/1024/2048`。
- 长 N 路径使用 GM 原地整行 Softmax，不实现跨 tile 在线 Softmax。
- 不支持 NZ 输入、mask、scale、dropout 和反向计算。

# 详细设计（required）

## 算子分析

### 数学公式

对每个 batch 和输出行 `i`：

```text
C[b, i, j] = sum(A[b, i, k] * B[b, k, j])
m[b, i] = max(C[b, i, :])
e[b, i, j] = exp(C[b, i, j] - m[b, i])
S[b, i, j] = e[b, i, j] / sum(e[b, i, :])
```

先减行最大值可避免指数溢出。

### 支持数据类型

| 张量 | 数据类型 | 计算类型 |
| --- | --- | --- |
| A | FP16 | FP16 |
| B | FP16 | FP16 |
| Matmul 累加 | FP32 | FP32 |
| Softmax 中间值 | FP32 | FP32 |
| S | FP16 | FP16 |

### 支持形状

官方 180 case 全量验证范围：

- `batch = 2/5/9`
- `M = 128/512/1024/2048/4096`
- `K = 128/512/1024`
- `N = 128/512/1024/2048`

此外，pytest 覆盖 `N=64`、batch=1、N=128 动态分派边界以及三个长 N
取值。

## 算子实现

### 总体架构

整个流程由一个 `MIX_AIC` kernel 完成，并根据 shape 选择两条路径。

小 N 路径：

```text
GM(A/B)
  -> Cube: TLA Matmul
  -> L0C FP32 accumulator
  -> L0C-to-UB handoff
  -> Vector: row max / exp / row sum / normalize / cast
  -> GM(S)
```

长 N 路径：

```text
GM(A/B)
  -> Cube: TLA Matmul
  -> GM(S) temporary logits
  -> AIC global barrier
  -> AIC-to-AIV notification
  -> Vector: whole-row Softmax in place
  -> GM(S)
```

小 N 路径不生成 Matmul GM 中间张量。长 N 路径复用最终输出地址暂存
logits，但 Matmul 和 Softmax 仍处于同一个 MIX kernel 中。

### 模块拆分

- `ascend950_batched_matmul_softmax.cpp`
  - Torch/JIT 入口。
  - 缓存静态 JIT 参数。
  - 直接向给定 stream 异步下发 kernel。
- `ascend950_batched_matmul_softmax_impl.cpp`
  - 定义 TLA Matmul、tile、swizzle、epilogue visitor 和长 N kernel。
  - 根据输出 tile 数量选择长 N tile 和 swizzle。
- `batched_basic_matmul_tla_ub_visitor.hpp`
  - 扩展 batch 与 M tile 的工作映射。
  - 管理 Matmul 到 UB epilogue 的数据交接。
- `batched_matmul_softmax.h`
  - Torch adapter、长 N Softmax tiling 和 block 数选择。
- `ascend950_batched_matmul_softmax.py`
  - Python API 和输入检查。

### Tiling 设计

小 N 配置：

```text
L1 tile: (M, N, K) = (48, 128, 128)
L0 tile: (M, N, K) = (48, 128, 64)
```

长 N 候选配置：

```text
Main:  L1 (256, 256, 128), L0 (256, 256, 64)
M128:  L1 (128, 256, 128), L0 (128, 256, 64)
N128:  L1 (256, 128, 128), L0 (256, 128, 64)
Small: L1 (128, 128, 128), L0 (128, 128, 64)
```

小 N 配置在主形状上兼顾 Cube 利用率、L0/UB 容量和尾块开销。
长 N 路径根据输出 tile 数量以及 M/N 关系选择候选配置和 swizzle 方向。

### 执行策略

小 N 工作项按 `(batch_index, m_tile_index)` 展开。长 N 路径按
`(batch_index, m_tile_index, n_tile_index)` 调度，并根据 tile 数将
AIC block 数限制为 16、20 或设备 AIC 核数。

`N=128` 使用混合分派：

```text
batch * M <= 2560: UB epilogue path
batch * M >  2560: long-N in-place path
```

kernel 入口采用异步 dispatch：

- 不在算子 wrapper 内部调用 `aclrtSynchronizeStream`。
- 同步边界由调用者或性能测试统一控制。
- 静态类型、布局及 JIT 符号在首次解析后缓存。
- 运行时仅更新地址、形状、batch stride 和 block 数等参数。

### Softmax 设计

每个输出行在 UB 中完成：

1. 对 FP32 累加结果求行最大值。
2. 逐元素减最大值并计算指数。
3. 求指数行和及倒数。
4. 将归一化与 FP16 cast 融合。
5. 一次写回最终输出。

`N <= 128` 的小工作量使用 FP32 UB visitor。长 N 路径按
`rowsPerTile` 将 GM logits 搬入 UB，调用 Ascend C `SoftMax` 后原地写回
输出。支持 N=128/512/1024/2048 的预计算 Softmax tiling。

### AIC/AIV handoff 与同步

小 N 路径由 CATLASS packed tile copy 完成 L0C 到 UB 的交接。长 N
路径先由所有 AIC block 完成 Matmul 和 GM 写回，再执行 AIC 全局屏障；
随后每个 AIC 向对应 AIV 发出完成通知，AIV 才开始读取 GM 并执行
Softmax。wrapper 不增加 host 同步。

### Host 侧设计

Torch adapter 验证设备、dtype、布局和形状，构造输出后调用
`Ascend950BatchedMatmulSoftmax`。host 与 kernel 共用 N=128 分派阈值，
避免参数准备和设备执行路径不一致。长 N Softmax tiling 按 N 静态缓存。

### Workspace 与片上存储

- 不申请额外 GM workspace。
- 小 N 路径的 Matmul 中间结果保留在 L0C/UB。
- 长 N 路径复用最终输出地址暂存 Matmul logits。
- Softmax 中间值使用 FP32 UB buffer。
- 最终输出以 FP16 写回 GM。

### 资源约束

- 目标架构为 Ascend 950，`CATLASS_ARCH=3510`。
- Tile 和 Softmax buffer 必须满足 L0/UB 容量限制。
- 输入需具有与声明布局一致的连续 backing storage。
- M、N、K 和 batch stride 在当前接口中使用 32 位运行时参数。

### 数值稳定性设计

- Matmul 使用 FP32 累加。
- Softmax 在 FP32 中执行 max-shift、exp、sum 和 normalize。
- 最终阶段才转换为 FP16。
- pytest 使用 `torch.bmm + torch.softmax` 作为 golden。

### 编译与调用

在 CATLASS `tests/optest` 下配置 Ascend 950：

```bash
PYTHON=$(which python3)
TORCH_DIR=$($PYTHON -c \
  'import torch,os; print(os.path.join(os.path.dirname(torch.__file__),"share","cmake","Torch"))')

cmake -S . -B build-bmms \
  -DCMAKE_BUILD_TYPE=Release \
  -DCATLASS_ARCH_LIST=3510 \
  -DPython_EXECUTABLE="$PYTHON" \
  -DTorch_DIR="$TORCH_DIR"

cmake --build build-bmms \
  --target \
  jit_verify_ascend950_batched_matmul_softmax_ascend950_batched_matmul_softmax_impl_3510 \
  -j
```

## 支持硬件

- Ascend950PR_9579。
- Driver 25.7.rc1。
- CANN 9.0.0.beta2 实机验证通过。

## 算子约束限制

- 输入输出仅支持 FP16。
- A/B 必须位于同一 NPU，且 batch、K 维满足 Matmul 约束。
- 仅支持规定的 RowMajor/ColumnMajor/RowMajor 布局。
- 支持 `N = 64/128/512/1024/2048`。
- 不支持空 Tensor 和跨设备输入。

# 可维可测分析

## 精度标准/性能标准

验证环境：

```text
Device: Ascend950PR_9579
Driver: 25.7.rc1
CANN: 9.0.0.beta2
Python: 3.12.9
PyTorch: 2.7.1+cpu
torch_npu: 2.7.1.post4
```

仓库 pytest 共 8 项通过，官方任务集 180/180 正确性通过。

同一进程、相同外层同步策略下，五轮中位数结果：

| shape (B,M,N,K) | fused (us) | torch pair (us) | speedup |
| --- | ---: | ---: | ---: |
| (8,128,128,128) | 4.472 | 7.561 | 1.691x |
| (8,128,64,128) | 4.766 | 7.691 | 1.614x |
| (1,128,128,128) | 4.540 | 7.684 | 1.693x |
| (2,256,128,128) | 4.670 | 7.705 | 1.650x |
| (1,512,128,128) | 4.529 | 7.702 | 1.700x |
| (8,128,128,64) | 4.652 | 7.654 | 1.645x |

以上小 shape 结果用于验证 UB epilogue 快路径。完整官方任务集使用官方
CSV 中的“小算子标杆耗时”作为基线，参数为 `warmup=5`、
`repeat=30`：

```text
passed           = 180/180
average_speedup  = 1.439749x
weighted_speedup = 1.350159x
```

加权性能超过 1.2x 要求。

msprof 结果确认：

- 每次融合调用仅有一个 `MIX_AIC` task。
- 30 次 task 平均 4.892 us，最小 4.211 us，最大 7.370 us。
- 小 N 路径未观察到 Matmul 中间结果写回 GM 的独立算子。
- 长 N 路径在同一个 MIX task 内完成 GM logits 写回和原地 Softmax。

## 功能测试矩阵

测试覆盖：

- pytest 覆盖 batch=1、N=64、N=128 分派边界和 N=512/1024/2048。
- 官方任务集覆盖 batch=2/5/9、M=128/512/1024/2048/4096、
  K=128/512/1024、N=128/512/1024/2048。
- Torch 算子注册、JIT 首次编译及缓存命中。
- 数值误差、不同 batch 和尾 tile。
- 单 kernel task 数量与端到端性能。

## 可维护性分析

- Tiling、运行时参数和 visitor 分层，便于独立调整。
- Python API 负责边界检查，kernel 仅处理支持范围。
- pytest 固化主要 shape，防止性能修改引入精度回归。
- JIT 接入沿用 CATLASS 既有注册方式，避免引入独立构建系统。

## 兼容性分析

- 代码基于 `cann/catlass:master` 完成 rebase。
- JIT kernel 和 `libcatlass_torch.so` 均完成编译链接验证。
- torch_npu 2.7.1.post4 可识别 Ascend950PR_9579。
- 预装 torch_npu 2.7.1.post2 不支持该 SoC，验收需使用 post4 或更新版本。

# 风险与应对

| 风险 | 影响 | 应对 |
| --- | --- | --- |
| 长 N GM 中间访存 | 降低融合收益 | 单 MIX kernel、原地写回及 shape 专用 tile |
| 新 shape 的 tile 效率下降 | 性能波动 | 扩充 shape matrix 并按 N/K 选择 tile |
| JIT 冷启动影响测量 | 首次耗时偏高 | 预热并报告稳态中位数 |
| host 同步混入算子热路径 | 吞吐下降 | wrapper 保持异步，统一外层同步 |
| torch_npu/CANN 版本不匹配 | 无法初始化或运行 | 固化已验证版本组合并记录环境 |
| 多核流水同步错误 | 错误或挂起 | pytest、重复运行和 msprof task 检查共同门禁 |

# 参考资料

- CATLASS 源码与 `tests/optest` 示例。
- CANN Ascend C API 文档。
- PyTorch `torch.bmm` 与 `torch.softmax` 接口文档。
- CATLASS PR 935：<https://gitcode.com/cann/catlass/pull/935>
