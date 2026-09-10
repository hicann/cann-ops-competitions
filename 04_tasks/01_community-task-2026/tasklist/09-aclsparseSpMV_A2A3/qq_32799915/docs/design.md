# aclsparseSpMV（A2/A3）算子设计文档

# 需求背景（required）

## 需求来源

参考 cuSPARSE SpMV 接口语义，在 Atlas A2/A3（DAV_2201，`arch22`）上使用
C++ Host 与 Ascend C Kernel 实现 `aclsparseSpMV`，并补齐公开 C API、
Python/ATen 入口、C++ UT 和端到端测试。

算子计算公式：

```text
Y = alpha * op(A) * X + beta * Y
```

A 为 CSR 稀疏矩阵，X/Y 为连续稠密向量，op(A) 支持非转置、转置和复数共轭转置。

## 背景介绍

SpMV 是稀疏线性代数、图计算及稀疏模型中的基础操作。CSR 以长度 `M+1` 的
row offsets、长度 `nnz` 的 column indices 和 values 描述矩阵。现有稀疏库已有
Handle、SpMat、DnVec 与 ACL stream 基础设施，本设计在其上实现完整三阶段接口，
并保持 A2/A3 标准 SpMV 与 A5 `aclsparseSpMVOp` 路径解耦。

# 需求分析（required）

## 需求描述

| 项目 | 支持范围 |
| --- | --- |
| 稀疏格式 | CSR，I32 row offsets/column indices |
| index base | 0、1 |
| 操作 | NON_TRANSPOSE、TRANSPOSE、CONJUGATE_TRANSPOSE |
| shape | 动态 M/K/nnz，空行、长尾行、nnz=0/1 |
| computeType | INT32、FP32、complex64 |
| 算法 | DEFAULT、CSR_ALG1、CSR_ALG2 |
| 标量模式 | Host pointer、Device pointer |
| 执行位置 | NPU，不允许 CPU fallback |

声明类型组合：INT8→INT32（INT32 累加）、INT8→FP32、FP16→FP32、BF16→FP32、
FP16→FP16（FP32 累加）、BF16→BF16（FP32 累加）、FP32→FP32、
complex64→complex64 和 FP32→complex64。

## 需求拆解

1. 实现 GetBufferSize、Preprocess、Execute 三阶段 C API；
2. 校验 CSR、shape、dtype、index base、pointer mode、algorithm 和 workspace；
3. 为 N/T/H、混合精度和 complex64 提供 arch22 Ascend C Kernel；
4. 提供 `torch.mv`/`aten::mv` NPU 注册，禁止 CPU fallback；
5. 保证输入只读、Y 原地 beta 语义、stream 顺序、确定性及资源无泄漏；
6. 达到任务书精度、性能、内存、ATK 和 Profiler 验收要求。

# 详细设计（required）

## 算子分析

### 数学公式

非转置时：

```text
for i in [0, M):
    sum = Σ values[p] * X[colInd[p]]
    Y[i] = alpha * sum + beta * Y[i]
```

转置时输出长度为 K；共轭转置在乘法前对 complex value 取共轭。重复坐标按输入
顺序累加，实现不要求 column indices 有序，但拒绝越界索引。

### 接口设计

```c
aclsparseStatus_t aclsparseSpMVGetBufferSize(
    aclsparseHandle_t handle, aclsparseOperation_t opA, const void *alpha,
    aclsparseConstSpMatDescr_t matA, aclsparseConstDnVecDescr_t vecX,
    const void *beta, aclsparseDnVecDescr_t vecY, aclDataType computeType,
    aclsparseSpMVAlg_t alg, size_t *bufferSize);

aclsparseStatus_t aclsparseSpMVPreprocess(
    aclsparseHandle_t handle, aclsparseOperation_t opA, const void *alpha,
    aclsparseConstSpMatDescr_t matA, aclsparseConstDnVecDescr_t vecX,
    const void *beta, aclsparseDnVecDescr_t vecY, aclDataType computeType,
    aclsparseSpMVAlg_t alg, void *externalBuffer);

aclsparseStatus_t aclsparseSpMV(
    aclsparseHandle_t handle, aclsparseOperation_t opA, const void *alpha,
    aclsparseConstSpMatDescr_t matA, aclsparseConstDnVecDescr_t vecX,
    const void *beta, aclsparseDnVecDescr_t vecY, aclDataType computeType,
    aclsparseSpMVAlg_t alg, void *externalBuffer);
```

三阶段复用同一校验与 workspace 计算。Preprocess 状态绑定 CSR pattern、三个描述符、
external buffer、stream、pointer mode、operation、computeType 和 algorithm；任一项
改变即失效，防止跨输入误复用。

## 算子实现

### Host 侧设计

Host 在任何输出写入前完成如下检查：

1. Handle/stream、op 和 algorithm 合法；
2. SpMat/DnVec 描述符、CSR/I32、device pointer、shape 和长度一致；
3. alpha/beta 非空并符合 computeType/pointer mode；
4. row offsets 单调、首尾与 nnz 一致，column index 在有效范围内；
5. dtype 组合、base0/base1、整数和 workspace 大小无溢出；
6. external buffer 存在、32-byte 对齐且不与 A/X/Y 非法重叠。

平台信息提供 AIV core 数和缓存信息。非转置按行及 nnz 权重生成
`SpmvRowPartition`，长尾行在 kernel 内按 tile 处理。普通路径 workspace 为 0；
窄 FP16/BF16 转置使用对齐 staging；complex64 多核转置查询
`blockCount * K * sizeof(complex64)` private partial workspace。其大小受 A2/A3
最小 192 MiB L2 限制，超限时回退确定性单核零 workspace 路径。

### Kernel 侧设计

非转置按独立 row range 分核，CSR row offsets/indices/values 和 X 从 GM 分块读取，
UB 内完成 gather 与乘加，再执行 alpha/beta epilogue。FP16/BF16 以 FP32 累加；
INT8 和 complex64 按声明 computeType 计算；空行仍正确执行 beta 语义。

转置路径按输入行贡献输出列。窄类型使用对齐 staging；complex64 每个 AIV 写独立
partial slice，reducer 再按固定 block 顺序归约到 Y，避免不支持的 complex GM
atomic 并保证确定性。共轭转置在复数乘法前执行 conjugate。

所有 memset、DMA 和 kernel launch 使用调用方 stream。描述符及 Tensor 生命周期由
preprocess 绑定、retirement event 和 `record_stream` 保护；不持久化完整稠密矩阵，
异常路径释放临时资源。

### Python/ATen 设计

通过 TorchNPU `OpCommand::RunOpApiV2` 将底层调用排入当前 NPU stream。adapter 在
入队前复制标量并验证 ABI、native 函数地址和 stream；缺少 adapter 或版本不匹配
直接报错，不静默回退。

注册 `aten::mv`，公开入口 `torch.mv(csr_tensor, dense_vector)` 仅接受二维 CSR NPU
矩阵和一维连续 NPU vector，固定 alpha=1、beta=0。适配层检查 layout、dtype、
shape、stride、device 与 alias，并按 PyTorch 语义构造输出；其他组合显式报错。

## 支持硬件

| 支持芯片 | 状态 |
| --- | --- |
| Atlas A2 910B3/910B4（DAV_2201） | 支持 |
| Atlas A3 `ascend910_9391`（arch22） | 支持构建；运行由对应实机验收 |

## 算子约束限制

- 仅支持 CSR 与 I32 索引；
- X/Y 必须为一维连续向量且与矩阵位于同一 NPU device；
- 不支持的 dtype/layout/algorithm 返回明确错误；
- external buffer 必须按查询值分配并满足对齐要求；
- A5 `aclsparseSpMVOp` 为独立接口，不作为本实现的替代路径。

# 可维可测分析

## 精度标准

INT8→INT32 逐元素 exact match；INT8→FP32、FP16/BF16 使用 FP32 Golden；FP32
使用 FP64 Golden；complex64 使用 complex128 Golden。FP16 使用
`rtol/atol=2^-9`、`A=1e-1`，BF16 使用 `2^-6`、`A=1`，FP32 使用
`rtol=2^-10`、`atol=2^-16`、`A=1e-2`；complex 实虚部分别按 FP32 标准。

## 性能标准

GPU 和 NPU 使用相同 device Event 调用边界，每 case 至少 warmup 10 次、采样 30 次，
报告 median/p90，复用 descriptor、workspace 与 preprocess，目标 GPU/NPU 比值不低于
0.25，不删除慢样本。

## 内存标准

可映射 GPU 接口的 case，NPU 额外峰值内存不超过 GPU 总内存的 50%；无 GPU 对照的
固有 workspace 不超过目标硬件 L2。连续 Create/Execute/Destroy 后 Host/device
内存无泄漏，A/X 保持只读。

## 测试设计

| 层级 | 覆盖内容 |
| --- | --- |
| C++ UT/ST | 212 个功能、边界、dtype、N/T/H、base 和 pointer mode 用例 |
| Python/ATen | torch.mv、out/alias、CSR layout、shape/stride/device、错误与无 CPU fallback |
| 精度 | 308 个完整 performance cases，逐元素检查 |
| ATK | 200 个 accuracy cases |
| 内存 | 首次、稳态和题目原始 collector 三种口径 |
| Profiler | 910B2/910B4 kernel、runtime API 与 SQLite quick_check |

当前验证结果：910B2 完整回归精度 308/308、ATK 200/200、内存 308/308、性能
296/296、12,628 个输出全部通过；910B4 当前源码干净构建、Host/Device pointer ×
base0/base1 四组 C++ 回归均 212/212，Profiler SQLite `quick_check=ok`；A3
`ascend910_9391` 当前源码交叉构建 rc0。

## 兼容性分析

新增标准 aclsparseSpMV 能力，不改变已有描述符 ABI；三阶段接口与公开枚举保持兼容。
A2/A3 arch22 实现与 A5 SpMVOp 分离，未声明能力显式返回错误，避免静默行为变化。
