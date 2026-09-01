# aclsparseSpMM / torch.sparse.addmm A2/A3 设计文档

## 需求背景（required）

### 需求来源

本设计对应 CANN 社区 `aclsparseSpMM 算子开发（A2/A3）`任务。目标是在 Atlas A2/A3 上补齐
`torch.sparse.addmm`、`aten::_sparse_addmm`、aclsparse 三段式 C++ 接口和 Ascend C Kernel，且核心计算
不得回退到 CPU。

### 背景介绍

算子计算

\[
C \leftarrow \alpha\,op(A)\,op(B)+\beta C,
\]

其中 `A` 是 CSR 稀疏矩阵，`B`、`C` 是稠密矩阵。与稠密 GEMM 相比，SpMM 的访存由 CSR
列索引驱动，行长度不均匀，并且对 `B` 存在不规则 gather。实现既要覆盖转置、布局、dtype、空行和
长尾行，又要保证 workspace 可复用、stream 异步语义及动态输入值的正确性。

本仓已有 arch35 SpMM。本任务新增 `sparse/spmm/arch22/`，公共描述符和 API 保持共用，硬件相关的
Tiling、预处理及 Kernel 隔离在架构目录中，以便 A2/A3 与其他平台在同一主干共存。

## 需求分析（required）

### 需求描述

- Python：对齐 PyTorch 2.7 及以上的 `torch.sparse.addmm` 行为。
- C++：保持 `aclsparseSpMMGetBufferSize`、`aclsparseSpMMPreprocess`、`aclsparseSpMM` 的签名和
  三段式生命周期。
- 格式：`A` 仅 CSR，int32 行偏移和列索引，index base 支持 0/1；`B/C` 支持 row-major 和
  column-major 及合法 leading dimension。
- dtype：FP16、BF16、FP32、complex64；低精度路径采用 FP32 中间计算。
- 泛化：动态 `M/K/N/nnz`、空行、长尾行、转置/共轭转置、非连续 Python 输入及广播 input。
- 质量：混合容差 CPU 单标杆、确定性、输入只读、workspace 边界、资源生命周期和无 CPU fallback。
- 性能：A3 的 8 个量化场景逐项倍率严格大于 0.25，算术平均倍率不低于 0.35。

### 需求拆解

1. 扩展公共描述符，使 CSR 和稠密矩阵描述符能表达任务要求的 dtype、布局和 const 语义。
2. 在 Host 侧统一校验 shape、dtype、operation、algorithm、pointer mode、leading dimension 和容量溢出。
3. 在 preprocess 中校验 CSR 结构，并为转置、分桶或专用调度生成可复用的结构计划。
4. 实现覆盖六条 dtype 路径的通用 AIV Kernel，并以动态、失败关闭的分类器增加安全快路。
5. 注册 PrivateUse1/NPU 的 ATen 实现，完成广播、连续化、标量转换、输出构造和 stream 计划缓存。
6. 建立 C++ L0/L1、Python E2E、ATK 精度、Profiler 和性能回归体系。

### 冻结基线与验证边界

- 当前 PR 生产树身份为
  `f3bca25e5ef4e9e0d95434bbf224b2d198bc1c1496129e531f653ed46b40ab37`，覆盖 `CMakeLists.txt`、
  `build.sh`、`cmake/`、`include/` 和 `sparse/` 中的 182 个文件；测试资产清单校验 58/58 通过，
  清单自身哈希随交付索引一并提供。
- A3 主证据来自 CANN 9.0.0 clean build 和 npu47 全局设备 `/dev/davinci6`、
  `/dev/davinci7`：它们分别是 Card ID 3 的 Chip 0/1，字符设备 minor 为 6/7。设备 12/14
  的历史结果不进入最终结论。
- 当前生产代码在 global6/global7 的 C++ 回归分别为 900/900、Python/ATen 分别为 85/85；
  双卡跨设备合同 1/1，四 dtype CPU Golden 400/400，default/explicit stream destroy 20/20，
  Profiler 记录 `no_cpu_fallback=true`。
- A3 实测安装包 `torch_npu 2.10.0` 对应 TorchNPU 产品版本 26.0.0、代码分支
  `v2.10.0-26.0.0` 和 CANN 9.0.0，映射关系见同一份官方版本说明。
- Atlas A2 Ascend 910B3 实机证据绑定生产树
  `85fa1f3b43febb4aaa9005096edd1a644952eaa2886b26beb40b5afc4d766750`：C++ 900/900、
  Python/ATen 85/85、四 dtype CPU Golden 400/400、destroy 20/20 和无 CPU fallback 验证均通过。
  与当前树逐文件比较后，SpMM 生产范围仅 `spmm_host.cpp` 增加失败关闭的零值防御，arch22 Kernel
  字节一致；对比清单随交付证据提供。A2 性能按任务书不要求提交。
- A2 实测安装包 `torch_npu 2.7.1.post4` 对应[官方版本说明](https://gitcode.com/Ascend/pytorch/blob/v2.7.1-26.0.0/docs/en/release_notes/release_notes.md)
  中的 TorchNPU 产品版本 26.0.0 和代码分支 `v2.7.1-26.0.0`；`torch 2.7.1+cpu` 中的 `+cpu` 是 wheel 构建标签，不表示
  算子回退 CPU。运行证据另以设备映射、已加载 NPU 动态库和 Profiler `no_cpu_fallback` 三重校验。
- global7 的 8 个量化场景均完成 10 次预热和 30 次正式 Profiler 样本，逐项倍率严格大于
  0.25、算术平均不低于 0.35。CSR、values 和稠密输入均由交付 manifest 固定；A100 原始输入
  字节未随任务附件提供，验收方仍需确认其基准采用同一固定输入口径。

## 详细设计（required）

### 算子分析

#### 数学公式

设原始 CSR 矩阵 `A0` 的 shape 为 `R×S`。根据 `opA`，有效矩阵

\[
A=op(A_0)\in\mathbb{F}^{M\times K}.
\]

根据 `opB`，有效稠密矩阵

\[
B=op(B_0)\in\mathbb{F}^{K\times N}.
\]

输出逐元素为

\[
C_{ij}=\beta C^{old}_{ij}+\alpha\sum_{p=0}^{K-1}A_{ip}B_{pj}.
\]

CSR 使用 `rowOffsets[M+1]`、`colIndices[nnz]` 和 `values[nnz]`。index base 为 `b∈{0,1}` 时，
第 `i` 行边范围为

\[
[rowOffsets_i-b,\ rowOffsets_{i+1}-b).
\]

行主矩阵地址为 `base + row*ld + col`，列主矩阵地址为 `base + col*ld + row`。转置及
complex64 共轭在有效地址和值读取阶段处理。

#### dtype 路径

arch22 C++ 接口支持下列组合：

| 路径 | A | B | C | computeType | 中间累加 |
| --- | --- | --- | --- | --- | --- |
| D01 | FP16 | FP16 | FP16 | FP32 | FP32 |
| D02 | FP16 | FP16 | FP32 | FP32 | FP32 |
| D03 | BF16 | BF16 | BF16 | FP32 | FP32 |
| D04 | BF16 | BF16 | FP32 | FP32 | FP32 |
| D05 | FP32 | FP32 | FP32 | FP32 | FP32 |
| D06 | complex64 | complex64 | complex64 | complex64 | 两个 FP32 分量 |

Python/ATen 公开接口要求三个 Tensor dtype 相同，因此使用 D01、D03、D05、D06；D02/D04 由
C++ 接口测试覆盖。

### 软件分层

```text
torch.sparse.addmm / aten::_sparse_addmm
                 |
                 v
sparse/spmm/torch/spmm_torch.cpp
  参数校验、广播、连续化、描述符/plan缓存、当前NPU stream
                 |
                 v
aclsparseSpMMGetBufferSize -> Preprocess -> SpMM
                 |
                 v
sparse/spmm/arch22/spmm_host.cpp
  能力校验、Tiling、workspace规划、Kernel异步下发
                 |
                 v
sparse/spmm/arch22/spmm_kernel.cpp
  CSR校验/结构计划、动态分类器、通用或专用AIV计算
```

### Host 侧设计

#### 参数和能力校验

Host 在任何 Kernel 下发前完成以下可见错误校验：

- handle、stream、描述符、alpha/beta 和数据指针有效性；
- CSR format、两个索引 dtype 均为 int32、index base 为 0/1；
- `op(A)`、`op(B)` 和 `C` 的矩阵维度一致；
- row/column-major 的 `ld` 满足最小跨度；
- A/B/C/computeType 精确匹配支持表，不进行隐式 dtype 提升；
- algorithm/operation 组合合法；所有 size、offset、task 数和 workspace 加法/乘法均做溢出检查。

arch22 不支持 `ACL_SPARSE_SPMM_CSR_FP32_HIGH_PRECISION_ALG`。`CSR_ALG3` 要求 `opA=N` 且
`opB=N/T`；混合输出的 DEFAULT 路径使用同样的操作限制。其余已声明组合由 ALG1/ALG2 或
FP32/complex64 DEFAULT 通用路径处理。

#### Tiling 和多核切分

Host 从平台读取 AIV 数、AIC 数和 UB 容量。`tileN` 根据 dtype、布局和 UB 动态选择，通用 Kernel
把 `(row, column-tile)` 展开为 task，并以 `task % blockNum` 分配给 AIV。短行批处理、CSR chunk 和
双缓冲用于减少小 DMA 与流水线等待；尾块始终携带真实 valid length。

任务书的 P01-P03 量化性能输入是 row-major 连续 B/C；该布局支持块搬运，只有运行时 constant-A/B
classifier 命中时才进入专用快路。column-major B 仍属于功能支持范围，但逻辑 N 方向在 GM 中跨
`ldb`，通用兜底必须逐元素 gather；实现使用精确的
MTE2→Scalar 事件保证正确性，不把该布局的吞吐等同于 P01-P03 性能承诺。后续若需要列主序量化性能，
应增加执行期预打包 workspace 或布局专用 Kernel，而不是改变当前接口语义。

#### workspace 与计划复用

workspace 的各子区域以 32-byte 对齐规划；其中 16-word header 固定占 64 bytes。workspace 包含：

- 16-word header：magic、ABI、plan hash、结构校验状态和 prepared 状态；
- 每 AIV 动态分类 flag；
- `opA=T/H` 时的有效 CSR、cursor 和 permutation；
- complex64 的配对元数据；
- 可选的行分桶/列调度元数据。

plan fingerprint 包含 workspace 地址、CSR 结构指针、shape、nnz、布局、operation、algorithm、dtype、
tile 和平台核数。values、B、C 和标量不属于结构 fingerprint，因此同一 sparsity pattern 可以复用
preprocess；所有依赖当前数值的优化都必须在每次 `aclsparseSpMM` 调用中重新分类。

`Preprocess` 和 `SpMM` 都只向调用方 stream 异步提交任务。若调用者未显式 preprocess，执行阶段会
建立结构计划；已有匹配计划则直接复用。Host 不为性能分支读取 device 数据，也不引入同步。

arch22 的 `matA` 描述符只记录一份 active-plan identity，因此同一个 `matA` 描述符及其 active
workspace 不是跨线程或跨 stream 可重入资源。并发 stream 必须分别使用独立的 handle、`matA`
描述符和 workspace，并在各自 stream 上完成 preprocess/execute 链；底层只读 CSR 数据可以共享。

#### handle 销毁与 default stream 重试

`aclsparseDestroy` 只在 handle 绑定 stream 中已提交任务完成后才释放默认 workspace。
显式 stream 直接等待该 stream；`nullptr` 表示的 default stream 则通过
`aclrtCreateEventExWithFlag(..., ACL_EVENT_CAPTURE_STREAM_PROGRESS)` 创建可复用 Event，
将 marker 记录到 default stream 尾部并等待该 Event，以覆盖 marker 之前的所有任务，
不扩大为整卡同步。

Event create/record/synchronize/destroy 或 workspace free 任一步失败时，handle 和 workspace
保持可达；已创建但未成功回收的 Event 也保留在 handle 中。接口返回错误供调用者对同一
handle 重试。每次 default-stream 重试会在队尾重新 record；进入销毁重试状态后，
`aclsparseSetStream` 拒绝切换到不同 stream，避免遗漏原 stream 中仍可能访问 workspace
的任务。default/显式 stream 正常路径已上板回归；Runtime 错误注入覆盖
create/record/synchronize/destroy/free 的分支仍是 P2 测试债。

### Kernel 侧设计

#### CSR 校验和有效结构

prepare Kernel 验证首尾 row offset、单调性、nnz 范围和列索引边界。错误写入 workspace header；
compute 读取 header 并对非法或不匹配计划失败关闭。`opA=T/H` 时在 workspace 生成有效 CSR 和
permutation，complex64 的 H 路径在读 value 时取共轭。

#### 通用 SpMM

每个 task 处理一行和一个 `N` 方向 tile：

1. 读取该行的 CSR 边段；
2. 分块搬入列索引和 A values；
3. 根据列索引和 B 的 layout/opB gather 稠密向量；
4. 在 FP32 或 complex FP32 分量中累加；
5. 应用 alpha；beta 为 0 时不读取旧 C，否则读取并应用 beta；
6. 对 D01/D03 仅在最终写回前转换一次低精度。

空行直接执行 epilogue，因此 `beta=0` 得到数学空和 `+0`，`beta!=0` 保留缩放后的 C。complex64
显式实现复乘与共轭，实部/虚部分开比对精度。

#### 动态 constant-A/B 快路

图数据中可能出现 A values 全相等且 B 的所有元素全相等。若

\[
A_{ip}=a_0\quad((i,p)\in\operatorname{supp}(A)),\qquad B_{pj}=b_0,
\]

则

\[
(AB)_{ij}=degree(i)\,a_0b_0.
\]

该路径把复杂度从 `O(nnz*N)` 降为分类扫描和输出的 `O(nnz+K*N+M*N)`。它不是固定值替换：

- Host 只按 dtype、布局、operation、N tile 和标量选择候选；不包含数据常量；
- classifier 每次调用逐 bit 扫描全部 A values 和 B，任一元素不同即拒绝；
- NaN/Inf、零符号歧义或保守溢出检查不通过时拒绝，交给通用 Kernel；
- classifier、direct output、generic 按同一 stream 顺序下发；命中时 direct 写结果且 generic no-op，
  拒绝时 direct 零写且 generic 完整执行；
- n=128 的 FP16/BF16/FP32 路径读取旧 C，在 FP32 中执行 `C+degree*a0*b0` 并仅最终转换一次；
- n=256 的 FP32/complex64、`alpha=1,beta=0` 路径直接覆盖输出；degree=0 显式写数学空和 `+0`；
- workspace 复用后修改 A/B 的首、中、尾任一值，下一次调用都会动态拒绝快路。

分类器约占 99.1 KiB UB；n=128 direct 约 65.1 KiB；n=256 direct 约 129.1 KiB，均低于
DAV2201 的 192 KiB UB。所有 partial chunk 在向上对齐后先初始化尾区，并以明确的 V/MTE 事件消除
未初始化 UB 读取和流水线竞争。

#### 为什么主计算使用 AIV 而不是 AIC/Cube

通用 CSR SpMM 的每条边指向不连续的 B 行，行长度也不规则。直接送入 Cube 需要先把稀疏块扩展或
打包成规则 ND/NZ tile，并处理重复/空洞；在本任务的低密度图上，打包、补零和额外 GM 流量高于
Cube 乘加收益。constant-A/B 路径更是 degree 映射和连续输出，瓶颈是 HBM 扫描/写回而非矩阵乘。
因此生产路径选择 AIV 做 gather、归约、类型转换和连续写回。AIC 数仍进入平台/tiling 信息，并为未来
高块密度或可复用 packed-B 的混合算法保留接口，但不为使用率而强制引入负收益的 Cube 阶段。

### Python/ATen 适配

`spmm_torch.cpp` 注册 `_sparse_addmm` 的 NPU 实现：

- 校验三输入同 NPU、同 dtype，mat1 为 2-D CSR 且索引为 int32；
- 按 PyTorch 规则把 input 广播到 `[M,N]`，`beta=0` 时不传播 input 中的 NaN/Inf；
- 对支持的非连续 input、mat2 和 CSR 元数据生成 ND 连续副本；
- alpha/beta 转为目标 dtype，实数 dtype 拒绝非零虚部；
- 从当前 NPU stream 建立/复用结构计划和 workspace；graph capture 时禁用不安全缓存；
- 输出始终为同 device、同 dtype、shape `[M,N]` 的 Dense Tensor。

## 支持硬件

| 支持的芯片版本 | 状态 |
| --- | --- |
| Atlas A2 训练/推理系列（arch22，ascend910b） | 基线树 `85fa1f3b...` 在 Ascend 910B3 实机通过 C++ 900/900、Python 85/85、四 dtype CPU Golden 400/400、destroy default/explicit 20/20，Profiler `no_cpu_fallback=true`；当前 arch22 Kernel 与实测基线字节一致，Host 侧仅增加失败关闭防御；安装包 `torch_npu 2.7.1.post4` 对应产品版本 26.0.0 |
| Atlas A3 训练/推理系列（arch22，ascend910_93） | CANN 9.0.0 clean build 及 global6/global7 上板回归通过 |
| Ascend 950PR/950DT（arch35） | 保留既有独立实现 |

## 算子约束限制

- 稀疏格式仅 CSR；行偏移和列索引仅 int32，且类型必须一致；index base `b` 支持 0/1。所有原始/
  逻辑维度及 leading dimension 均不超过 `INT32_MAX`，`0<=nnz<=INT32_MAX-b`；row offsets 必须从
  `b` 单调增长到 `nnz+b`，列索引范围为 `[b, physicalACols+b)`。
- A/B 为二维矩阵；A/B/C dtype 组合必须精确命中支持表。
- 不支持跨设备 Tensor；Python 层不做 Tensor dtype 提升。
- `ACL_SPARSE_SPMM_CSR_FP32_HIGH_PRECISION_ALG` 在 arch22 返回 NOT_SUPPORTED。
- 稠密矩阵最后地址按 row-major `(rows-1)*ld+cols` 或 column-major
  `(cols-1)*ld+rows` 计算；元素数乘 dtype 字节数、CSR 数组及 workspace 的全部加法/乘法必须能由
  `size_t` 表示。Python workspace 还不得超过 Tensor 的 `INT64_MAX` 可寻址范围。
- workspace 必须使用 `GetBufferSize` 返回的完整容量，并保持到 stream 中相关任务完成；少 1 byte 的
  已注册 workspace 确定失败。裸 DEVICE 指针的实际容量由调用者保证；描述符和数据也必须满足同样
  生命周期。逻辑 `M==0` 或 `N==0` 时 workspace 为 0 且允许空指针。
- 专用快路只是优化；不满足守卫条件不会改变接口支持范围或结果语义。

## 可维可测分析

### 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| FP16 | `rtol=2^-9, atol=2^-9, A=1e-1`，匹配率不低于 0.99 | 生态算子开源精度标准 |
| BF16 | `rtol=2^-6, atol=2^-6, A=1`，匹配率不低于 0.99 | 生态算子开源精度标准 |
| FP32 | `rtol=2^-10, atol=2^-16, A=1e-2`，匹配率不低于 0.99 | 生态算子开源精度标准 |
| complex64 | 实部和虚部分别按 FP32 标准 | 任务书 |
| A3 性能 | 8 场景逐项倍率 `>0.25`，算术平均 `>=0.35` | 任务书 A100 NCU 基线 |

CPU Golden 分别用 FP32（低精度）、FP64（FP32）和 complex128（complex64）计算。性能主指标为
一次公开接口调用内所有 NPU Kernel 的 Profiler 总耗时；至少预热 10 次、正式采样 30 次，报告中位数
和 P90。Python 公开入口 wall-clock 和 Event 分别作为补充；C++ 另用 ACL Event 只包围复用
descriptor/workspace 后的 Execute 阶段。preprocess 一次性耗时单独报告，上述补充值都不替代 Profiler 主指标。

### 测试设计

- C++ 参数矩阵：6 dtype path、4 algorithm、N/T/H、B/C 两种 layout、base0/1、host/device scalar。
- 契约：三段式状态、workspace 少 1 byte、输入只读、guard bytes、plan 复用/失效、使用独立
  handle/descriptor/workspace 的双 stream、descriptor lifecycle、malformed CSR trap，以及 default/显式
  stream 的正常销毁等待。Event/workspace 失败注入分支仍列为 P2 测试债，不声明已动态覆盖。
- Python：默认/out/meta、广播、非连续、dtype/device/shape 异常、cache 身份/失效、四 dtype E2E。
  实现会在 graph capture 期间禁用不安全缓存；当前回归不包含 NPU graph capture E2E，不把该 guard
  声明为 graph capture 测试覆盖。
- 泛化精度：400 条 CPU 单标杆，覆盖空行、nnz=0/1、长尾、不同 shape、标量和特殊值。
- 快路守护：常量值、动态 A/B mutation、恢复、base0/1、chunk 边界、NaN/Inf/±0 fail-closed。
- 性能：锁定 manifest、CSR、runner 和两个动态库 SHA；每个 case 独立进程，Profiler 证明无 CPU fallback。
  global7 最终采集的 8/8 case 均完成 30/30 正式样本和采样后精度检查，逐项倍率严格大于
  0.25、算术平均不低于 0.35，数值门禁通过；逐项数值与输入哈希见交付 evidence。任务附件未提供
  A100 原始输入字节，验收方可按 manifest 中固定的 CSR、values 和稠密输入哈希核对基准口径。

## 兼容性分析

- 公共 `aclsparseSpMM*` 原型不变，新枚举只沿用既有声明。
- arch22 与 arch35 通过 CMake 的 SOC→arch 映射隔离，公共描述符校验为两个平台共享。
- workspace header 携带 ABI 和 plan hash；旧 workspace 不会被新 Kernel 静默解释。
- Python 注册只作用于 NPU/PrivateUse1 dispatch，不改变 CPU/CUDA 实现。
