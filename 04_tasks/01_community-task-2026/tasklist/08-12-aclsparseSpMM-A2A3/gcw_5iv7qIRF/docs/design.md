# aclsparseSpMM A2/A3 算子设计文档

# 需求背景（required）

## 需求来源

本设计对应 CANN 社区任务 2026 的 `08-12-aclsparseSpMM-A2A3`，目标是在
Ascend A2/A3 上补齐 PyTorch `torch.sparse.addmm` / `aten::_sparse_addmm`
到 aclsparseSpMM C++ 接口和 Ascend C Kernel 的完整链路。

## 背景介绍

`torch.sparse.addmm(input, mat1, mat2, beta, alpha)` 计算：

```text
output = alpha * mat1 * mat2 + beta * input
```

其中 `mat1` 为二维 CSR 稀疏矩阵，`mat2` 为二维稠密矩阵，`input` 可广播到
`[M,N]`。任务要求支持 FP16、BF16、FP32、Complex64，补齐 Python/ATen、
aclsparse Host、Ascend C Kernel、测试和 A3 性能量化，同时与 arch35/A5 路径共存。

# 需求分析（required）

## 需求描述

1. 注册 NPU Sparse CSR `aten::addmm` 和 `addmm.out`，不得 CPU fallback。
2. 复用 `aclsparseSpMMGetBufferSize`、`aclsparseSpMMPreprocess`、`aclsparseSpMM`。
3. 支持 `ACL_FLOAT16`、`ACL_BF16`、`ACL_FLOAT`、`ACL_COMPLEX64`。
4. CSR rowOffsets/colIndices 仅支持 int32，index base 支持 0/1。
5. 支持任务声明的 ROW/COL、NON_TRANSPOSE 和 ALG1/ALG2/ALG3 组合。
6. 覆盖 zero nnz、空行、长尾行、零维、broadcast、noncontiguous、普通/复数标量。
7. A2/A3 功能精度通过；A3 各性能场景倍率大于 0.25，平均不低于 0.35。

## 需求拆解

| 层次 | 交付内容 |
| --- | --- |
| Python/ATen | 参数校验、NPU dispatch、输出构造、out 语义、无 fallback |
| 公共 C++ 接口 | descriptor、workspace、preprocess、stream、生命周期、错误码 |
| arch22 Host | dtype/layout/algorithm 校验、CSR preprocess、tiling、typed dispatch |
| arch22 Kernel | 实数/低精度/复数计算、边界、ROW/COL、性能优化 |
| 共存 | arch35/A5 公共 Host 和 descriptor 回归 |
| 测试 | C++ smoke、Python contract、精度、性能、Profiler |

# 详细设计（required）

## 算子分析

### 数学定义

设 CSR `A` 为 `[M,K]`，稠密 `B` 为 `[K,N]`，可广播输入 `C` 为 `[M,N]`：

```text
Y[m,n] = alpha * sum(A[m,k] * B[k,n]) + beta * C[m,n]
```

Complex64 对实部和虚部分别执行复数乘加：

```text
(ar + i*ai) * (br + i*bi)
= (ar*br - ai*bi) + i*(ar*bi + ai*br)
```

### 数据类型

| A/B/C | computeType | 累加 | 输出 |
| --- | --- | --- | --- |
| FP16 | FP32 | FP32 | FP16 |
| BF16 | FP32 | FP32 | BF16 |
| FP32 | FP32 | FP32 | FP32 |
| Complex64 | Complex64 | FP32 实/虚平面 | Complex64 |

### 主要约束

- `mat1` 必须为二维 CSR；`mat2` 与输出为二维 dense。
- A/B/C dtype 一致，不做隐式 promotion。
- rowOffsets 与 colIndices 必须为连续 int32 NPU Tensor。
- ROW 要求 `ld >= cols`，COL 要求 `ld >= rows`。
- `beta=0` 时校验 input 规格但不读取数值，NaN/Inf 不传播。
- 零输出维度返回合法空 Tensor；`K=0` 在 adapter 内完成 beta 路径。

## 总体架构

```text
torch.sparse.addmm / aten::_sparse_addmm
                 |
                 v
SparseCsrPrivateUse1 ATen adapter
                 |
                 v
aclsparseSpMMGetBufferSize
        -> aclsparseSpMMPreprocess
        -> aclsparseSpMM
                 |
                 v
arch22 Host tiling / typed launcher
                 |
                 v
FP32 / FP16 / BF16 / Complex64 Ascend C Kernel
```

## Python/ATen adapter 设计

### 注册与校验

adapter 使用：

```cpp
TORCH_LIBRARY_IMPL(aten, SparseCsrPrivateUse1, m)
```

注册 `aten::addmm` 和 `aten::addmm.out`。入口检查 layout、dim、device、dtype、shape、
broadcast、CSR int32/contiguous 和 NPU stream，任何缺注册或 CPU 结果在严格门禁中失败。

### C/Y 与 broadcast 处理

完整连续 `[M,N]` self 直接作为只读 C，output 作为独立 Y，通过内部 descriptor
output pointer 保持单 SpMM kernel，避免 P-02 的 TensorMove。

broadcast 或 noncontiguous self 不使用长序列中不稳定的
`expand().contiguous()`；改为：

```cpp
output.copy_(self);
```

让 TorchNPU 按 broadcast/stride 语义写入 base-format output，再由 SpMM 原位读写。
noncontiguous mat2 在 NPU 上构造连续副本并记录当前 stream。

### 首次 typed launch 稳定性

CANN 9.0/TorchNPU 2.10 在部分新 typed 调用签名首次执行时可能返回 success 但不写
output。诊断确认首轮与次轮的 tiling、binEdge、alpha/beta 完全一致，Host stream
同步实验无效。

adapter 使用容量 64 的 thread-local 调用签名缓存。签名包含 device/stream、dtype、
M/K/N/nnz、alpha identity、beta-zero、self shape/contiguity、mat2 contiguity。
新签名以独立资源完整调用并同步一次，再执行真实调用；缓存循环替换，避免无界增长。
固定 shape 正式性能在 warmup 后不再进入该分支。

## 公共 descriptor 与接口设计

- CSR descriptor 校验 index base、value type 和 pointer 完整性。
- dense descriptor 允许零维；具体算子决定零维语义，SpSM 配套保留原拒绝合同。
- pointer 更新使 active workspace 失效，防止复用旧 preprocess。
- dense descriptor 内部支持 C input 与 Y output 指针分离，不改变公开 ABI。
- Handle 绑定调用方 stream；workspace 由调用方或 TorchNPU allocator 提供并保持至执行完成。

## arch22 Host 设计

### Workspace

workspace 包含：

```text
64B header
128B SpmmArch22TilingData slot
reorder[M]
binEdge[blockDim + 1]
optional Complex64 planar B
```

所有 offset 使用 64B 对齐并执行整数溢出检查。

### CSR preprocess

Host 读取 rowOffsets/colIndices，计算：

- 每行 nnz 与 maxRowNnz；
- 连续列标志；
- identity reorder 可行性；
- greedy row bin pack；
- 每个 AIV 核的 `[rowStart,rowEnd)`。

reorder 内部编码 row id、degree 与 contiguous flag，减少 Kernel 重复 CSR 扫描。

### blockDim

按当前设备 AIV core count 获取：A3 为 48，A2 为 40。`GetBufferSize`、Preprocess、
SpMM 每个公开阶段只查询一次，保证 workspace 布局一致。

### typed dispatch

Host 根据 dtype 选择 FP32、FP16、BF16、Complex64 Kernel；FP32 `alpha=1` 选择
identity-alpha 专用实例。未知 dtype/algorithm 返回明确错误，不允许静默 no-op。

## arch22 Kernel 设计

### 通用行/列路径

每个 AIV 核处理 binEdge 指定行区间，再按 N 切 chunk。ROW 使用连续 DataCopyPad，
COL 使用 staging + offset gather/scatter。窄行使用 byte-counted DataCopyExtParams，
避免不足 32B 时的错误搬运。

### FP32 row-block

对 ROW/ROW、identity reorder、N<=256 的热路径：

- 分块预取 rowOffset/reorder/col/value metadata；
- 连续 B-run 使用单块 DMA；
- degree<=7、N<=128 使用 108-row block；其他路径使用 84-row block；
- identity-alpha 模板在编译期删除 alpha 分支；
- C/beta 每 row-block 一次加载，输出 queue 深度 2 重叠 V 与 MTE3。

### FP16/BF16

- values 与 B 转换到 FP32；
- FP32 累加，输出阶段转换回 FP16/BF16；
- lowp row-block 使用双缓冲 B input；
- 混合容差全量精度 MatchedRatio=1.0、硬错误 0。

### Complex64

- interleaved Complex64 在 UB 中拆为实/虚平面；
- B 可在 workspace 中预打包 planar layout；
- 使用四实数平面完成复数乘加和复数 alpha/beta；
- ROW/COL 写回使用精确 8B complex element 语义与 V->MTE3 同步。

## A5/arch35 共存

公共 enum、descriptor、zero-dimension 和 pointer validation 对 arch35 同步补充回归。
arch35 保留自身 Kernel 和平台实现，A2/A3 arch22 优化不进入 arch35 热路径。

## 性能优化方案

### 目标场景

- P-01 FP32：小 nnz、中等 N。
- P-02 FP16/BF16/FP32：大 M、N=128、degree 约 6/7、beta=1。
- P-03 四 dtype：超大 M、N=256、degree 25/26、beta=0。

### 关键优化

1. 使用全部 AIV 核并做 nnz 负载分桶。
2. Host preprocess 携带 row degree/continuity，删除 Kernel 重复扫描。
3. FP32 row-block metadata staging、C/beta 块级处理和输出双缓冲。
4. 连续 B-run 单块 DMA、identity-alpha 专用化、degree 6/7 固定展开。
5. 108-row 小 degree block 降低 block 数。
6. 低精度 FP32 累加与 B 双缓冲。
7. Complex64 planar B pack 和实虚平面 vector path。
8. adapter 连续 C/Y 分离，删除 P-02 TensorMove。

## 支持硬件

| 支持的芯片版本 | 支持 |
| --- | --- |
| Atlas A2 / Ascend910B | 是 |
| Atlas A3 / Ascend910_93 | 是 |

## 算子约束限制

- 仅 CSR；不新增 COO/CSC SpMM。
- CSR index 仅 int32。
- 仅声明支持的 op/layout/algorithm 组合。
- Python adapter 当前要求 A/B/C 同 dtype。
- ATen adapter 每次调用重建 descriptor/workspace/preprocess；正式 Kernel 性能使用
  preprocess 复用的 C++/Profiler 口径。
- 公开 ATK 26.7.8 原生与任务附件 outputs/comparator/kwargs 协议不兼容；显式 runtime
  shim 可在原附件 SHA256 不变的前提下完成 200/200，相关兼容证据不进入生产代码 PR。

# 可维可测分析

## 功能与精度

| 平台 | Release | C++ smoke | 严格 NPU | 代表性精度 |
| --- | --- | --- | --- | --- |
| A2 | 通过 | 0 failed | 41 tests OK，skip 3 | P-01/P-02/P-03 四 dtype 0 错误 |
| A3 | 通过 | 0 failed | 41 tests OK，skip 2 | 117/117；原附件+runtime shim 200/200 |

FP16/BF16 CPU Golden 使用 FP32，FP32 使用 FP64，Complex64 使用 Complex128 并
分别比较实部/虚部。所有代表性 P-02/P-03 精度均无硬错误。

## A3 性能

| 场景 | NPU us | A100 us | 倍率 | 结果 |
| --- | ---: | ---: | ---: | --- |
| P-01 FP32 | 54.179 | 87.040 | 1.606513 | 通过 |
| P-02 FP16 | 1057.263 | 321.536 | 0.304121 | 通过 |
| P-02 BF16 | 1066.230 | 413.920 | 0.388209 | 通过 |
| P-02 FP32 | 734.587 | 204.576 | 0.278491 | 通过 |
| P-03 FP16 | 39541.109 | 25446.496 | 0.643545 | 通过 |
| P-03 BF16 | 39543.032 | 33704.096 | 0.852340 | 通过 |
| P-03 FP32 | 40464.388 | 16571.232 | 0.409526 | 通过 |
| P-03 Complex64 | 147286.362 | 38806.400 | 0.263476 | 通过 |

8/8 场景均严格大于 0.25，算术平均 0.593278，不低于 0.35。

## 维护与回归

- Host validation、workspace、layout、pointer、zero-dimension 有独立 C++/Python 合同。
- A2/A3 构建映射和 arch35 source selection 有 dependency-free 单测。
- Profiler 每 step 调用后设备同步，active Count 与调用次数一致。
- clean 代码 PR 不包含内部实验文档和 ATK 兼容诊断目录。
