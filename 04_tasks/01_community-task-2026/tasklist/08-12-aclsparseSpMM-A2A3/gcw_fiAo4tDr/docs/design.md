# aclsparseSpMM A2/A3 算子设计文档

> 本文按社区 `design_template.md` 编写，描述目标能力、实现方案和验收方法。

# 需求背景（required）

## 需求来源

本设计对应 CANN 社区 2026 年 8 月任务
“aclsparseSpMM 算子开发（A2/A3）”，以任务书
`aclsparseSpMM_A2A3_task_doc.md` 为接口、功能和验收要求的唯一来源。

## 背景介绍

SpMM 用于计算 CSR 稀疏矩阵与稠密矩阵的乘加：

```text
out = beta * input + alpha * (mat1 * mat2)
```

目标入口为 PyTorch `torch.sparse.addmm` / `aten::_sparse_addmm`，底层复用
`ops-sparse` 的 `aclsparseSpMM*` 三阶段 C++ API，在 Atlas A2/A3 的 DAV-2201 /
arch22 上执行 Ascend C Kernel。实现不能通过 CPU 计算替代 NPU 核心路径。

与逐输出行、逐 term 搬运 B 的基础实现相比，通用性能路径采用“拓扑识别 + 模板选择”
的组织方式：Preprocess 只分析不变的 CSR topology；执行时按行模板或
pattern/term-major 模板复用 B row，并对不适用结构安全回退。

# 需求分析（required）

## 需求描述

在保持既有公开函数签名和描述符 ABI 的前提下，补齐：

```text
aclsparseSpMMGetBufferSize
        -> aclsparseSpMMPreprocess
        -> aclsparseSpMM
```

并将其接入 PyTorch NPU sparse dispatch。数学语义为
`C = alpha * op(A) * op(B) + beta * C`，其中 A 为 CSR，B/C 为稠密矩阵。

## 需求拆解

1. Python/ATen 支持 `torch.sparse.addmm`，遵循 PyTorch 2.7 及以上版本的 shape、
   dtype、广播、异常和 alias 语义。
2. C++ 层复用 handle、CSR/DnMat 描述符、workspace、stream 和销毁接口，三阶段调用
   可独立使用且支持重复执行。
3. 数据类型贯通 FP16、BF16、FP32、Complex64；CSR row offsets/column indices 使用
   int32，index base 支持 0/1。
4. B/C 支持合法 Row-major、Column-major 和 `ld` padding；`opA` 支持
   NON_TRANSPOSE、TRANSPOSE 及 Complex64 CONJUGATE_TRANSPOSE，`opB` 支持
   NON_TRANSPOSE、TRANSPOSE；未声明组合返回明确错误。
5. 覆盖动态 shape、动态 nnz、零 nnz、空行、长尾和非均匀 CSR，保持输入只读。
6. Preprocess 识别可复用的行拓扑、重复 pattern 和列 panel；values、alpha、beta
   等可变数据不得被缓存。
7. A3 性能按任务书 P-01/P-02/P-03 及附件 50 条门禁执行；A2/A3 分别进行功能和
   精度验证，提供 NPU dispatch 与 Profiler 证据。

# 详细设计（required）

## 算子分析

### 数学公式

CSR A 的逻辑形状为 `[M, K]`，B 为 `[K, N]`，D 为可广播到 `[M, N]` 的 input，
输出 C 为 `[M, N]`：

```text
C[i,j] = beta * D_broadcast[i,j]
       + alpha * sum(A_values[p] * B[col_indices[p],j])
```

`opA/opB` 通过逻辑索引变换参与维度检查和元素地址计算。Complex64 使用复数乘加，
CONJUGATE_TRANSPOSE 对 A 的非零值取共轭。

### 支持数据类型

| 输入/输出 | 类型 | 计算策略 |
| --- | --- | --- |
| `input`、A values、B、C | FP16 | FP32 累加，写回 FP16 |
| `input`、A values、B、C | BF16 | FP32 累加，写回 BF16 |
| `input`、A values、B、C | FP32 | FP32 累加；高精度算法可使用 Kahan 补偿 |
| `input`、A values、B、C | Complex64 | 交错实虚 FP32 复数乘加 |
| row offsets、column indices | int32 | `ACL_SPARSE_INDEX_32I`，base 0/1 |

Tensor 间不做 dtype promotion；三个数值 Tensor 必须同 dtype，`computeType` 由合法
组合决定。alpha/beta 在 Host 或 Device pointer mode 下均可用。

### 支持形状与布局

- `A` 是二维 CSR `[M,K]`，`B` 的逻辑形状由 `opA/opB` 决定，维度必须匹配。
- `input` 仅按 PyTorch 规则广播到 `[M,N]`；A、B 不做矩阵广播。
- `M/K/N` 可为非对齐正整数；支持 `nnz=0`、空行和行内非均匀 nnz。
- B/C 支持 Row-major、Column-major，分别检查 `ld >= cols` 或 `ld >= rows`。
- CSR 行内索引按列排序并合并重复坐标是性能验收输入约束；非排序输入仍走正确性
  回退路径或按接口约束返回错误。

## 算子实现

### 总体数据流

```text
ATen 参数与稀疏元数据
        |
        v
描述符校验 -> GetBufferSize -> Preprocess(topology)
        |                         |
        +---- 模板选择 <-----------+
                         |
                         v
       通用 CSR 行模板 / term-batched / pattern-major
                         |
                         v
               Ascend C Kernel -> C 写回
```

### Host 侧设计

#### 参数校验与三阶段语义

Host 按 handle、描述符、指针、CSR 类型、index base、shape、dtype、computeType、
layout、`ld`、op 和 alg 的顺序校验；非法组合返回明确的
`ACL_SPARSE_STATUS_INVALID_VALUE` 或 `ACL_SPARSE_STATUS_NOT_SUPPORTED`。
`INT32_MAX` 边界、workspace 对齐和 stream 归属也在 Host 检查。

`GetBufferSize` 只计算所需空间。`Preprocess` 仅写入不变 topology 元数据（行重排、
分桶、pattern 分组、panel 描述）；不保存 values、alpha、beta。每次 `SpMM` 都从
当前指针重新取得 alpha/beta 和 values，因此在 Preprocess 后修改这些数据仍然生效。
workspace、descriptor 和 preprocess 结果可被同一 stream 上的重复调用复用。

#### 模板选择

| 模板 | 工作单元 | 适用条件 | 不适用时 |
| --- | --- | --- | --- |
| 通用 CSR 行 | `{row, columnChunk}` | 任意合法 CSR、布局和标量模式 | 不回退，作为基线 |
| 均匀短行 | `{rowGroup, termPair, columnChunk}` | 行 nnz 近似均匀且 UB/stride 对齐 | 通用行模板 |
| topology/pattern-major | `{patternId, physicalRows, columnChunk}` | 精确重复列 pattern、可验证周期、输出写集互斥 | 通用行模板 |
| term-batched B panel | `{patternId, termBatch, columnChunk}` | pattern-major 的连续 B row 可合并搬运 | pattern-major 或通用行 |
| 高精度 FP32 | `{row, columnChunk}` | `CSR_FP32_HIGH_PRECISION_ALG` | 返回不支持或标准 FP32 |

公开 `CSR_ALG1/ALG2/ALG3` 只表达任务书规定的语义、布局和确定性约束；是否使用
上述内部模板由 topology、dtype、layout、op、标量和 UB 门禁共同决定，不能仅通过
更换枚举名制造性能差异。

#### topology/pattern-major 方案

Preprocess 对 row offsets/column indices 做一次扫描，识别：

1. 每行 nnz、空行、长尾和可均衡的行组；
2. 列索引是否构成稳定的重复 pattern 或周期 pattern；
3. pattern 行集合、degree、列 chunk、B panel 连续性和输出行互斥性。

识别结果只包含 row id、列索引/列 panel 的不变元数据。Kernel 执行时仍从 GM 读取
每个物理行的当前 values；同一 pattern 的每个 B row 在一个 column chunk 内只搬入
一次，再对多个物理行批量广播乘加。pattern 不完整、列回绕无法安全切分、stride
不满足对齐、layout/op/dtype 不在白名单或 workspace 不足时，整个 work item 回退
通用模板，不混用两种输出写集。

#### 分核与负载均衡

- 通用 CSR：按行 nnz 降序后采用小顶堆贪心分配到 AIV，生成 `rowReorder` 和
  `binEdges`，降低长尾造成的核间空闲。
- 均匀短行：在行 nnz 差异不超过 1 时按连续等行数分桶，减少重排和 Scalar 开销。
- pattern-major：按 pattern/column chunk 的估算 DMA 与向量工作量分配，保证每个
  物理输出行只归属一个核，避免跨核归约。
- 任何模板选择必须保留可验证的通用回退，不能因优化门禁失败而返回错误结果。

#### workspace 与 TilingData

workspace 采用 64B 对齐，逻辑布局为：

```text
header | TilingData | rowReorder | binEdges | pattern metadata | B panel metadata
```

可选区按模板申请，不缓存 values。TilingData 至少包含 `M/K/N/nnz`、`ld`、layout、
op、dtype、alg、index base、blockDim、N 方向 chunk、各区段 offset 和模板 key。
alpha/beta 的本次调用值作为执行参数更新，不固化到 preprocess 结果。Host 与 Kernel
共用同一对齐和 UB 容量公式，避免越界。

### Kernel 侧设计

#### 通用 CSR 行模板

每个 AIV block 处理一组行和一个 N 方向 tile：

```text
读取 rowOffsets/colIndices
  -> 搬入 B[col,:] 和（beta != 0 时）C/input
  -> 按 CSR term 顺序执行 alpha * value * B 的累加
  -> epilogue 合并 beta 分支并写回 C
```

FP16/BF16 在 FP32 accumulator 中计算；FP32 标准路径保持 CSR 顺序；高精度路径维护
补偿项；Complex64 在 AoS 交错实虚布局中完成四乘加。`beta=0` 时不搬入、不读取
原 C/input 的数值，但仍校验其地址、shape、dtype 和 device。

#### pattern-major 与 B panel 模板

一个 work item 为一个 pattern、一个物理行集合和一个 N column chunk。Kernel 先将
该 pattern 的 B rows 以 term batch 方式搬入 UB，再对多个物理行读取各自 values，
用 `Brcb`/`MulAddDst` 或等价向量指令完成广播乘加。每个物理行的 C 只在所属 block
写回一次；跨 pattern 不共享可变 values，也不需要跨核归约。

列索引跨 K 回绕时拆分 B DMA 段，但保持 term 顺序和输出 group 不变。尾 pattern、
非连续 B/C 和未对齐列 chunk 使用 DataCopyPad 或通用标量兜底；不能为了合并 DMA
改变 CSR 的数值顺序。

#### 内存、对齐与同步

- DAV-2201 UB 规划按 32B 对齐，B、accumulator、输出和 metadata 槽位分时复用；
  `DataCopyPad` 的 GM/UB stride 按接口单位填写，`blockCount <= 4095`。
- A2/A3 通过编译期 SoC 分支隔离完成依赖。A2 保留已验证的 MTE2→Vector、
  Vector→MTE3 和 MTE3→Scalar 完成事件；A3 使用对应的最小同步集合。
- 所有输出 DMA 在复用 UB 或 Kernel 退出前满足 producer/consumer 依赖；禁止以
  不匹配的 PipeBarrier 替代完成事件。
- 不使用 `GlobalTensor::GetValue/SetValue` 访问 GM；非对齐搬运统一使用
  `DataCopyPad`。不引入 DAV-2201 不支持的 arch35 SIMT/VF API。

### Python/ATen 适配

`aclsparse_spmm_aten.cpp` 注册 Sparse COO/CSR dispatch：

1. 检查 device、dtype、维度和 input 广播规则；COO 在 NPU 上转换为 CSR，CSR 复用
   crow/col/value 元数据，输入保持只读。
2. 将非连续 dense Tensor 按 stride 正确描述，无法直接描述时在 NPU 上生成连续副本。
3. 取得调用方 stream，创建/复用 aclsparse descriptor、workspace 和 preprocess plan，
   顺序调用三阶段 API，输出在同一 NPU device 创建。
4. 不把 CPU Golden 或 CPU sparse kernel 接入运行时；仅在测试程序中使用 CPU Golden。
5. plan cache 只允许缓存 topology 和 descriptor 生命周期信息；以 Tensor 身份、
   version、shape、layout 和索引快照使结构变化失效，values 更新不触发错误复用。

### 文件规划

| 目录/文件 | 责任 |
| --- | --- |
| `include/cann_ops_sparse.h` | 既有 SpMM API、算法枚举和约束说明 |
| `sparse/spmm/common` | 公共描述符、校验、workspace 与算法分派 |
| `sparse/spmm/arch22` | A2/A3 Host、Tiling、Kernel 及 SoC 条件分支 |
| `python/aclsparse_spmm_aten.cpp` | PyTorch NPU dispatch、广播和资源生命周期 |
| `test/spmm/arch22`、`python/test_*` | C++/ATen 功能、边界、精度和回归测试 |
| `docs`、任务书附件 | 接口限制、复现方式、自测报告和性能记录 |

公共 Host 逻辑必须与 arch35/A5 共存；硬件差异限制在 arch22/arch35 的 Tiling、
Kernel specialization 和编译期宏中。

## 支持硬件

| 产品 | 架构/编译目标 | 验证范围 |
| --- | --- | --- |
| Atlas A2 | DAV-2201/arch22，Ascend910B2/B3 | 功能、精度、接口、稳定性 |
| Atlas A3 | DAV-2201/arch22，Ascend910_93 系列 | 功能、精度、接口、稳定性、性能 |

软件基线遵循任务书指定的 CANN 版本；Python 适配面向 PyTorch 2.7+、torch_npu
26.0.0+。ATK 使用执行时公开的最新版，不在设计中固定旧版本号。

## 算子约束限制

| 约束 | 设计要求 |
| --- | --- |
| 稀疏格式 | 仅 CSR；row offsets/column indices 为同类型 int32 |
| index base | 支持 ZERO/ONE，Kernel 内部统一归一化为零基 |
| dtype | FP16、BF16、FP32、Complex64；数值 Tensor 必须同 dtype |
| op/layout | 仅实现任务书和 cuSPARSE 支持矩阵声明的组合 |
| workspace | 由 GetBufferSize 计算；Preprocess/SpMM 在同一 stream 使用并复用 |
| 可变数据 | 不缓存 values、alpha、beta；Preprocess 后更新必须生效 |
| beta=0 | 不读取原 C/input 数值，NaN/Inf 不得传播 |
| 异步与资源 | 使用调用方 stream；描述符/workspace 生命周期结束前保证设备完成 |
| 回退 | 专用模板门禁失败时回退通用 CSR，不牺牲正确性 |
| 不支持项 | I64 索引、非法维度/布局/op/dtype/算法组合返回明确错误 |

# 可维可测分析

## 精度标准/性能标准

| 验收项 | 标准 | 来源 |
| --- | --- | --- |
| 精度 | 采用生态算子开源精度标准，CPU Golden 单标杆混合容差；FP16/BF16 用 FP32，FP32 用 FP64，Complex64 用 complex128 Golden | 任务书 |
| A2/A3 功能精度 | 200 条附件精度用例分别在 A2、A3 执行；Complex64 实虚部均满足标准 | 任务书 |
| P 组性能 | A3 上 P-01/P-02/P-03 的每个 case×dtype 严格 `>0.25×A100`，8 项算术平均 `>=0.35×A100` | 任务书 |
| 泛化性能 | A3 上附件 50 条每项严格 `>0.25×A100`，50 条平均 `>=0.35×A100` | 任务书 |
| NPU 路径 | Dispatch、Profiler 和 `ldd` 证据证明核心计算未走 CPU fallback | 任务书 |

## 测试设计与接受方式

1. Release 构建后先运行 C++ feature/conformance，再运行 C++ 专项、异常路径、
   PyTorch 主测试和边界测试。
2. 精度覆盖四 dtype、COO/CSR、idxBase、转置/共轭转置、两种布局、padding、
   非连续 Tensor、空行、零 nnz、重复调用和非法参数；A2、A3 分开留存结果。
3. ATK 精度测试使用届时公开最新版原生执行器；附件自带执行器只能作为环境自检，
   不能替代 ATK 正式证据。
4. 性能每个 case 预热至少 10 次、正式采样至少 30 次，设备同步后采集 median/P90；
   descriptor、workspace 和 preprocess 结果在正式采样期间复用，preprocess 单独计时。
5. 计时排除首次编译、数据生成、Host→Device 搬运和无关初始化；按 case id 关联，
   缺失或非 `ok` case 不参与组平均。P 组和附件 50 条分别验收，不合并平均。
6. 通过 NPU dispatch、msprof/Profiler 的 Kernel 名称、耗时和设备信息证明没有 CPU
   fallback；连续创建、执行、销毁资源并检查无泄漏和非法同步。

## 可维护性、可观测性与风险

- 公开 API 与硬件实现解耦，A2/A3 只通过编译期 SoC 宏选择同步和 Kernel specialization。
- topology cache 的 key、失效规则和 workspace offset 由 Host/Kernel 共享定义，新增
  模板必须保留通用回退。
- 每个性能实验只改变一个变量，并将假设、命令、采样和归因写入性能记录；设计文档
  只维护稳定的方案和约束。
- 主要风险是 CSR 拓扑不规则、B row 不连续、pattern 跨 K 回绕、UB 不足以及 A2/A3
  完成事件差异。风险处理顺序为：门禁识别、独立模板、精确回退、功能/精度回归、
  再进行性能长测。

## 兼容性分析

保持 `include/cann_ops_sparse.h` 中既有 `aclsparseSpMM*` 函数签名和原有枚举语义，
新增能力仅通过兼容扩展和内部 TilingKey 表达。公共 Host 校验和描述符逻辑可与 A5/
arch35 共存；arch22 不依赖 arch35 的 SIMT/VF。Python 侧只注册 NPU sparse dispatch，
不改变 CPU/CUDA 后端行为。未声明的参数组合按既有错误码返回，不以隐式转换或 CPU
fallback 掩盖兼容性问题。

## 修订记录

| 版本 | 日期 | 修改内容 |
| --- | --- | --- |
| 0.1 | 2026-09-02 | 按社区模板重写 aclsparseSpMM A2/A3 需求、Host/Kernel、拓扑模板、回退和验收设计；移除实测状态与日志明细 |

## 参考资料

- `aclsparseSpMM_A2A3_task_doc.md`
- `aclsparseSpMM_testCase/README.md`
- `research/ops-sparse/sparse/spmm/arch22/DESIGN.md`
- PyTorch `torch.sparse.addmm` 与 `aten::_sparse_addmm`
- NVIDIA cuSPARSE SpMM 文档
- <https://gitcode.com/cann/ops-sparse>
- <https://gitcode.com/cann/cann-ops-competitions>
- <https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md>
