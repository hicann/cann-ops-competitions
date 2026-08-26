# 需求背景（required）

## 需求来源

本设计对应 2026 年社区任务《aclsparseSpMM 算子开发（A2/A3）任务书》。任务要求参考
PyTorch `torch.sparse.addmm`、`aten::_sparse_addmm` 与 cuSPARSE SpMM，在昇腾 NPU 上完成
Python/ATen 适配，并复用和扩展 `ops-sparse` 已有 `aclsparseSpMM*` C++ 接口及 Ascend C
Kernel，补齐 Atlas A2、Atlas A3 上的功能、精度、泛化、性能、测试和文档。

目标软件版本为 PyTorch 2.7 及以上、torch_npu 26.0.0 及之后版本，CANN 版本采用
算子开源仓指定版本。核心计算必须在 NPU 上完成，不允许使用 CPU fallback 代替 NPU
实现。

## 背景介绍

### aclsparseSpMM 算子实现优化

SpMM（Sparse Matrix-Dense Matrix Multiplication）计算 CSR 稀疏矩阵与稠密矩阵的乘加：

$$
C = \alpha \cdot op(A) \cdot op(B) + \beta \cdot C.
$$

PyTorch 层对应：

$$
\mathrm{out}=\beta\cdot\mathrm{input}+\alpha\cdot(\mathrm{mat1}\times\mathrm{mat2}).
$$

其中 `input`、`mat2` 和输出为稠密 Tensor，`mat1` 为 CSR Tensor。任务要求打通
`torch.sparse.addmm` 到 `aten::_sparse_addmm`、aclsparse Host 和 Ascend C Kernel 的完整
NPU 数据流，同时保留已有公开 C++ 接口的命名和函数签名。

### aclsparseSpMM 现有实现现状分析

`ops-sparse` 已有 Handle、稀疏/稠密矩阵描述符、workspace 管理，以及
`aclsparseSpMMGetBufferSize`、`aclsparseSpMMPreprocess`、`aclsparseSpMM` 三阶段接口。
本任务不新增同名或同功能接口，而是在现有接口和 SpMM 目录中补齐 A2/A3 能力。

| 层次 | 复用对象 | 本任务补齐内容 |
| --- | --- | --- |
| Python/ATen | PyTorch `torch.sparse.addmm` 与 `aten::_sparse_addmm` schema | NPU 注册、参数校验、Tensor/CSR 元数据转换、输出构造 |
| Handle | `aclsparseCreate`、`aclsparseDestroy`、`aclsparseSetStream` | A2/A3 功能验证、调用方 stream 语义 |
| 稀疏描述符 | `aclsparseCreateCsr`、`aclsparseCreateConstCsr`、`aclsparseDestroySpMat` | 四种 dtype、int32 索引、base 0/1、异常处理 |
| 稠密描述符 | `aclsparseCreateDnMat`、`aclsparseCreateConstDnMat`、`aclsparseDestroyDnMat` | 四种 dtype、Row/Column-major、leading dimension、异常处理 |
| Workspace | `aclsparseSpMMGetBufferSize` | 各合法布局、算法和 dtype 组合的 workspace 计算 |
| Preprocess | `aclsparseSpMMPreprocess` | 复用 CSR 行重排和分桶，明确结果复用与失效条件 |
| 执行 | `aclsparseSpMM` 与已有 Ascend C Kernel | A2/A3 四种 dtype、布局、操作、精度和泛化能力 |

公开接口保持以下调用顺序：先查询 workspace，可选执行 preprocess，随后执行 SpMM。
描述符、workspace 和 preprocess 结果允许在相同输入元数据下复用。

# 需求分析（required）

## 需求描述

在 `ops-sparse` 现有 SpMM 实现上完成 Atlas A2、Atlas A3 能力补齐，支持
`float16`、`bfloat16`、`float32`、`complex64`，实现 Python/ATen 与 aclsparse C++
接口、Host 调度、preprocess 和 Ascend C Kernel 的统一 NPU 路径。接口行为、返回值、
dtype、shape、device 和异常行为对齐目标 PyTorch 与任务书约束；C++ 调用阶段、描述符、
workspace、算法和错误处理保持既有 `aclsparseSpMM*` 接口兼容。

## 需求拆解

1. **Python/ATen 闭环**：为 `aten::_sparse_addmm` 注册 NPU 实现，使
   `torch.sparse.addmm` 在 CSR 输入下命中该实现；完成广播、非连续 Tensor、标量、
   dtype、shape 和 device 校验。
2. **C++ 接口闭环**：复用三阶段 `aclsparseSpMM*` 接口和已有描述符/Handle，不修改
   函数原型，不重复增加接口。
3. **dtype 闭环**：A、B、C 支持 float16、bfloat16、float32、complex64；半精度与
   bfloat16 使用 float32 累加，float32 使用 float32 计算，complex64 使用复数乘加。
4. **格式与操作闭环**：A 仅支持 CSR；B、C 支持 Row-major 和 Column-major；覆盖
   任务书声明的 `opA`、`opB`、算法枚举、leading dimension 与合法组合。
5. **索引闭环**：CSR row offset 与 column index 均为 int32 且类型一致，支持 base 0
   和 base 1，拒绝 int64 与非法索引组合。
6. **资源闭环**：明确 Host/Device pointer mode、workspace、preprocess、stream 和
   描述符生命周期；更新 CSR 指针时使旧 preprocess 状态失效。
7. **边界与泛化闭环**：覆盖动态 M/K/N/nnz、nnz=0/1、空行、长尾行、非均匀行、
   零长度维度、最大 shape、leading dimension 和溢出检查。
8. **验证闭环**：提供 C++ UT、ATen 端到端 UT、NPU 精度/性能测试和 complex64
   专项用例；A2/A3 功能精度、A3 性能与 Profiler 证据按任务书验收。
9. **共存闭环**：公共 Host/描述符逻辑与 arch22 Kernel 分离，使 A2/A3 与 A5 实现
   能在同一主干共存，避免复制公共流程或改变其他架构行为。

# 详细设计（required）

## 算子分析

### 数学公式

设 CSR 矩阵 $A\in\mathbb{F}^{M\times K}$，稠密矩阵
$B\in\mathbb{F}^{K\times N}$，输出 $C\in\mathbb{F}^{M\times N}$。非转置路径为：

$$
C_{i,j}=\alpha\sum_{p=rowOffsets_i}^{rowOffsets_{i+1}-1}
values_p\cdot B_{colInd_p,j}+\beta C_{i,j}.
$$

`opA` 或 `opB` 为转置时，根据操作后的逻辑 shape 计算内维和输出 shape；
`CONJUGATE_TRANSPOSE` 对 complex64 的被操作元素取共轭。对实数 dtype，转置与共轭
转置的数值效果相同。`beta=0` 时不读取原 C/input 数值，因而其中的 NaN、Inf 不传播
到输出。

Python 接口将可广播的 `input` 扩展为 `[M,N]` 后作为 C 的初始值，输出始终为同 NPU
device 上的稠密 Tensor。

### 支持数据类型

| A/B/C value dtype | C++ 枚举 | computeType | 计算语义 |
| --- | --- | --- | --- |
| float16 | `ACL_FLOAT16` | `ACL_FLOAT` | float32 累加，输出转换为 float16 |
| bfloat16 | `ACL_BF16` | `ACL_FLOAT` | float32 累加，输出转换为 bfloat16 |
| float32 | `ACL_FLOAT` | `ACL_FLOAT` | float32 乘加；高精度算法使用补偿求和 |
| complex64 | `ACL_COMPLEX64` | `ACL_COMPLEX64` | complex64 乘加，支持复数 alpha/beta 与共轭 |

A、B、C 的 value dtype 必须相同。Python 层不执行 Tensor 间 dtype 提升；`input`、
`mat1.values()`、`mat2` 必须相同 dtype。实数 dtype 的 alpha/beta 接受 Python 整数或
浮点数，拒绝虚部非零的复数；complex64 接受实数或复数并转换为 complex64。

### 支持形状

- Python `mat1` 为 `[M,K]` CSR，`mat2` 为 `[K,N]` Dense，输出为 `[M,N]`。
- Python `input` 支持 `[M,N]`、`[N]`、`[1,N]`、`[M,1]` 等可广播至 `[M,N]` 的
  形状；`mat1` 和 `mat2` 不做矩阵维度广播。
- C++ 根据 `opA`、`opB` 计算逻辑 shape，并校验乘法内维与 C shape。
- B、C 支持 Row-major 和 Column-major。Row-major 要求 `ld>=cols`，Column-major
  要求 `ld>=rows`；支持大于最小值的 padding。
- 支持规格内 M/K/N/nnz 动态变化、nnz=0/1、空行和零长度维度。
- arch22 实现限制 M、K 和 nnz 不超过 int32 可表示范围；稠密元素数量和地址计算必须
  可由 int64 表示。超出限制时返回明确错误。

## 算子实现

### 实现方案

整体数据流如下：

```text
torch.sparse.addmm
  -> aten::_sparse_addmm / SparseCsrPrivateUse1
  -> Python/ATen 参数校验、input 广播与连续化、输出构造
  -> 创建或复用 CSR/Dense 描述符与执行计划
  -> aclsparseSpMMGetBufferSize
  -> 可选 aclsparseSpMMPreprocess（CSR 行重排、分桶）
  -> aclsparseSpMM（调用方 stream 上发射一个 SpMM Kernel）
  -> Dense NPU Tensor
```

整个计算路径不调用 CPU reference、框架 CPU 稀疏算子、其他后端或算子级 fallback。
CPU Golden 仅用于测试结果对比，不进入运行时。

#### 3.2.1 host侧设计：

**公共参数检查。** `aclsparseSpMMGetBufferSize`、`aclsparseSpMMPreprocess` 与
`aclsparseSpMM` 复用同一校验函数，依次检查：

1. Handle、描述符、alpha/beta 和输出指针；
2. A 为 CSR，row offset/column index 均为 int32 且类型一致，index base 为 0 或 1；
3. A/B/C dtype 与 computeType 组合；
4. `opA`、`opB`、algorithm、order、leading dimension 的合法组合；
5. 逻辑 shape、非空数据指针、M/K/N/nnz 和乘积溢出。

无效参数返回 `ACL_SPARSE_STATUS_INVALID_VALUE`，空 Handle 返回
`ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR`，不支持的格式、索引、dtype、operation 或
algorithm 返回相应 NOT_SUPPORTED 状态，workspace 为空返回
`ACL_SPARSE_STATUS_INSUFFICIENT_RESOURCES`。

**算法分发。** 支持 `ACL_SPARSE_SPMM_ALG_DEFAULT`、`CSR_ALG1`、`CSR_ALG2`、
`CSR_ALG3` 和 `CSR_FP32_HIGH_PRECISION_ALG`。CSR_ALG2 优先用于 Row-major 路径，
CSR_ALG1 优先用于 Column-major 路径；CSR_ALG3 仅接受 CSR、`opA=NON_TRANSPOSE`，
且 `opB` 不能为 `CONJUGATE_TRANSPOSE`；高精度算法仅接受 float32。其他未声明组合
直接返回错误，不做静默替换。

**Tiling 数据。** Host 将以下运行时元数据写入 tiling：

```text
A/B/C 逻辑 shape、nnz、ldb、ldc、indexBase
opA、opB、orderB、orderC、pointerMode、dtype、algorithm
Host 标量值或 Device 标量地址
workspace 内 rowReorder、binEdges 的偏移
```

这些字段全部来自当前接口参数和描述符，不依赖 case id、固定公开 shape 或输入数值。

**Workspace。** workspace 由行重排数组和分桶边界组成，所有区段按 64 byte 对齐：

```text
rowReorder: M * sizeof(int32_t)
binEdges:   (blockDim + 1) * sizeof(int32_t)
```

`GetBufferSize` 是唯一的大小计算入口。调用方按返回值分配 Device buffer，并保证其在
异步工作完成前有效。Python/ATen 适配使用 NPU Byte Tensor 管理 workspace；当前容量
不足时按不小于需求的几何增长策略扩展，并保留旧 buffer 至其异步使用结束。

**Preprocess。** 显式 preprocess 读取 CSR row offset，计算每行 nnz，按行负载降序进行
稳定排序，再采用最小当前负载的贪心策略分配到 AIV bin，生成 `rowReorder` 与
`binEdges`。稳定排序与固定 tie-break 保证相同输入得到确定性结果。结果与 CSR 描述符
及传入 workspace 绑定；修改 CSR 指针或更换 workspace 时使活动 preprocess 状态失效。
调用 `aclsparseSpMM` 时若没有匹配的活动结果，则在执行前完成一次 preprocess。

Preprocess 是显式准备阶段，允许为分析 CSR 元数据执行必要的数据传输；正式重复计算
复用描述符、workspace 和 preprocess 结果，计时阶段不包含该一次性成本。

**分核策略。** 从目标平台取得可用 AIV 核数，非转置 Row-major 快路径按预处理后的
bin 分核。每个核只处理自己的输出行集合，避免跨核写冲突。转置、共轭转置、列主和
高精度路径采用确定性遍历；在不能安全并行拆分输出所有权时使用单核，优先保证精度、
确定性和无原子竞争。

**Python/ATen 适配。** 在 `SparseCsrPrivateUse1` 注册 `aten::_sparse_addmm`：

- 校验三个 Tensor 均在同一 NPU，`mat1` 为 CSR、其索引为 int32，三者 dtype 一致；
- 校验二维矩阵乘法 shape 和 `input` 广播；
- 对不能直接描述的非连续 `input`、`mat2` 与 CSR 元数据生成连续 NPU 副本；
- `beta!=0` 时将广播后的 input 作为输出初值，`beta=0` 时直接创建输出且不读取 input；
- 从 torch_npu 获取当前 stream 并传给 `aclsparseSetStream`；
- 执行计划按设备、数据指针、M/K/N/nnz、dtype 和 alpha/beta 匹配，合法时复用描述符、
  workspace 和 preprocess，元数据变化时重新创建；
- 输出为同 device、同 dtype 的 Dense Tensor，不覆盖任何输入。

**生命周期。** A/B/C 数据、Device 标量、workspace 与描述符必须保持有效，直到调用方
确认 handle stream 上的工作完成。C++ 调用方在 stream 完成后销毁描述符和 Handle；
ATen 适配持有参与异步执行的 Tensor 和计划对象，避免提前释放。Handle 销毁只处理自身
资源，不销毁调用方 stream 或用户 workspace。

#### 3.2.2 kernel侧设计：

Kernel 入口根据 dtype 选择 float16、bfloat16、float32 或 complex64 模板实例；一次
`aclsparseSpMM` 只发射一个 SpMM 计算 Kernel。

**Row-major 快路径。** 当 `opA=N`、`opB=N` 且 B/C 为 Row-major 时，每个 AIV block
根据 `binEdges` 取得行区间，再由 `rowReorder` 找到源 CSR 行。按 N 维分 tile：

1. 从 GM 搬入一段 B 行和对应 C tile；
2. 对该 CSR 行的所有非零元素执行标量-向量乘加；
3. 应用 alpha/beta；
4. 将逻辑输出 tile 搬回 C，padding 区域不写入。

float32 与 complex64 走向量化 Row-major 路径。complex64 以相邻实部/虚部表示，乘法
使用 `(ar*br-ai*bi, ar*bi+ai*br)`，共轭路径在乘加前翻转虚部符号。

**通用路径。** float16、bfloat16、Column-major、`opA/opB=T/H` 及其他合法算法组合
使用统一地址计算函数，根据 order、ld 和 operation 将逻辑坐标映射到 GM。float16 与
bfloat16 转换为 float32 累加后再写回目标 dtype；float32 高精度算法使用 Kahan 补偿
求和。非法组合不进入 Kernel。

**确定性与边界。** 每个输出元素只有一个核负责，累加顺序由 CSR row offset、column
index 和确定性循环顺序固定，不使用跨核原子累加。nnz=0 时仅计算 beta*C；零输出
元素时 Host 不发射 Kernel；尾 tile、空行、base 1 和 leading-dimension padding 均由
显式边界处理。

**流水与内存。** 快路径使用 GM→UB、向量计算、UB→GM 的分块流水和双缓冲队列，tile
大小按 dtype 宽度、UB 容量和对齐要求确定。B tile 和 C tile 不在 GM 中间物化；同一
输出 tile 的累加留在 UB/寄存器中直到写回，避免无必要的 GM 往返。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas A2 训练系列产品 / Atlas A2 推理系列产品 | √ |
| Atlas A3 训练系列产品 / Atlas A3 推理系列产品 | √ |

A2/A3 使用 arch22 实现并分别通过开源仓支持的 SoC 配置构建。A5 使用独立架构目录；
公共描述符和 Host 接口修改不得改变 A5 现有行为，后合入的任务需完成相关冲突处理和
双架构回归。

## 算子约束限制

1. Python 仅支持 Dense `input`、CSR `mat1` 和 Dense `mat2`；`mat1`、`mat2` 均为
   二维，且 `mat1.size(1)==mat2.size(0)`。
2. `input` 必须可广播至 `[M,N]`；`mat1`、`mat2` 不支持矩阵维度广播。
3. 三个 Tensor 必须位于同一 NPU 且 dtype 一致，不执行 Tensor 间 dtype 提升。
4. C++ A 仅支持 CSR；row offset 与 column index 仅支持一致的 int32；index base 支持
   0 和 1，不支持 int64 索引。
5. B、C 支持 Row-major 与 Column-major，必须满足对应最小 `ld`；只接受任务书和
   cuSPARSE SpMM 类型表声明的 layout、operation、dtype、algorithm 组合。
6. float16、bfloat16、float32、complex64 为完整支持范围，不扩展其他 dtype。
7. `beta=0` 仍校验 input 的 shape、dtype 和 device，但不读取 input 数值。
8. 本算子不要求图融合；不得覆盖输入或执行未声明的 in-place/alias 行为。
9. 最大 M、K、nnz 受 int32 索引范围限制；稠密元素数、ld、workspace 和地址偏移必须
   通过溢出校验。
10. 不支持的参数组合在 Host 明确返回错误，不允许 CPU、框架、其他后端或其他算子
    fallback。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述(不涉及说明原因) | 标准来源 |
| --- | --- | --- |
| float16 精度 | CPU Golden 使用 float32；逐元素满足 `rtol=2^-9`、`atol=2^-9`，匹配率不低于 0.99，绝对误差不超过 `max(1e-1, 32*ULP(golden))` | 任务书、《生态算子开源精度标准》 |
| bfloat16 精度 | CPU Golden 使用 float32；逐元素满足 `rtol=2^-6`、`atol=2^-6`，匹配率不低于 0.99，绝对误差不超过 `max(1e0, 32*ULP(golden))` | 任务书、《生态算子开源精度标准》 |
| float32 精度 | CPU Golden 使用 float64；逐元素满足 `rtol=2^-10`、`atol=2^-16`，匹配率不低于 0.99，绝对误差不超过 `max(1e-2, 32*ULP(golden))` | 任务书、《生态算子开源精度标准》 |
| complex64 精度 | CPU Golden 使用 complex128；实部、虚部分别使用 float32 标准并同时满足匹配率和硬上限 | 任务书、《生态算子开源精度标准》 |
| 确定性 | 确定性算法重复执行按 bit-wise 规则验收；其他算法按混合容差验收 | 任务书 |
| A3 性能 | 每个 case×dtype 性能倍率均大于 A100 的 0.25 倍，全部量化场景算术平均不低于 0.35 倍 | 任务书 |
| NPU Dispatch | Profiler 证明核心计算在 NPU 执行，无 CPU fallback；报告 NPU Kernel 总耗时 | 任务书 |

功能与精度需分别在 Atlas A2、Atlas A3 执行；性能仅需在 Atlas A3 执行。C++ UT 覆盖
四种 dtype、Host/Device 标量、两种 order、合法 N/T/H 与算法组合、base 0/1、
nnz=0/1、空行、长尾行、padding、确定性、输入只读、workspace 和错误码。ATen
端到端 UT 覆盖 NPU 注册、广播、非连续 Tensor、beta=0、complex64 标量、动态 shape、
nnz/稀疏度和异常输入。

性能采集使用任务书固定的排序、去重 CSR 输入；每个 case 预热至少 10 次、正式采样
30 次，每次同步后计时，报告中位数和 P90。描述符、workspace 和 preprocess 结果在
正式采样期间复用；首次编译、数据生成、Host→Device 搬运、无关初始化和一次性
preprocess 不计入正式时间。

固定量化场景如下，NPU 时间在 A3 实测后填入自验证报告：

| 编号 | M×K×N | nnz | dtype | alpha/beta | A100 NCU Kernel 总耗时（μs） |
| --- | --- | ---: | --- | --- | ---: |
| P-01 | 2,708×2,708×1,433 | 10,556 | float32 | 1/0 | 87.040 |
| P-02 | 169,343×169,343×128 | 1,166,243 | float16 / bfloat16 / float32 | 1/1 | 321.536 / 413.920 / 204.576 |
| P-03 | 2,449,029×2,449,029×256 | 61,859,140 | float16 / bfloat16 / float32 / complex64 | 1/0 | 25,446.496 / 33,704.096 / 16,571.232 / 38,806.400 |

每个测试记录 CANN、PyTorch、torch_npu 版本，具体 A2/A3 型号、物理设备标识、case
参数、精度结果、性能数据、Profiler 证据和失败项。连续创建、执行、同步完成后销毁
描述符与 Handle，并使用资源检查工具验证无内存或资源泄漏。

## 兼容性分析

本设计不修改 `aclsparseSpMMGetBufferSize`、`aclsparseSpMMPreprocess`、
`aclsparseSpMM` 的函数原型和三阶段调用语义，不新增同名接口。既有 Handle、描述符、
workspace 和 stream API 继续复用；新增 dtype、布局、算法与错误检查只放行任务书声明
的合法组合，原有合法调用保持行为不变。

Python/ATen 适配是新增的 NPU 注册路径，不改变 CPU/CUDA 注册。A2/A3 特有 tiling 和
Kernel 位于 arch22，公共描述符逻辑保持架构无关；A5 继续使用自己的架构实现。公共
代码合入时以同一参数校验和生命周期规则为唯一来源，避免复制 Host 流程、双份 tiling
或运行时兼容分支。
