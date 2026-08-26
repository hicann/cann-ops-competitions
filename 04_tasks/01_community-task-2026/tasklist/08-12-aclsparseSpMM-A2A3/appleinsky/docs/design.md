# 需求背景（required）

## 需求来源

本任务来源于 CANN 社区任务"aclsparseSpMM 算子开发（A2/A3）"：为 ops-sparse
仓库中已有的 aclsparse SpMM 稀疏矩阵乘算子补齐 Atlas A2（Ascend 910B）与
Atlas A3（Ascend 910_93）系列产品上的能力，并向上打通
`torch.sparse.addmm` 的 NPU 端到端调用链路。

## 背景介绍

### SpMM 算子简介

SpMM（Sparse Matrix-Dense Matrix Multiplication）计算：

```
C = alpha · op(A) · op(B) + beta · C
```

其中 A 为 M×K 的 CSR 稀疏矩阵，B 为 K×N 稠密矩阵，C 为 M×N 稠密矩阵，
alpha/beta 为标量。SpMM 是 GNN、推荐系统、科学计算等场景的核心算子，
NVIDIA 侧对应 cuSPARSE 的 `cusparseSpMM`。

### ops-sparse 仓库 SpMM 实现现状分析

ops-sparse 仓库已声明 aclsparse SpMM 三步法公开接口
（`include/cann_ops_sparse.h`：`aclsparseSpMMGetBufferSize` /
`aclsparseSpMMPreprocess` / `aclsparseSpMM`），并按 SOC 架构分目录实现：

| 架构目录 | 面向 SOC | 现状 |
| --- | --- | --- |
| `sparse/spmm/arch22/` | ascend910b*（A2）、ascend910_93*（A3），DAV-2201 | 仅 fp32 基础路径，缺 fp16/bf16/complex64、B/C 布局、opB=T/C、预处理复用等能力 |
| `sparse/spmm/arch35/` | ascend950*，DAV-3510 | 独立实现（fp32/fp16/int8），与本任务无关 |
| `sparse/spmm/arch20/` | ascend310p* | 既有实现，与本任务无关 |

根 CMakeLists 的 SOC→arch 映射：`ascend910b*`→arch22、`ascend910_93*`→arch22、
`ascend950*`→arch35、`ascend310p*`→arch20。A2 与 A3 同属 DAV-2201，
本任务只需交付 arch22 一份实现即可双平台复用。

PyTorch 侧，`torch.sparse.addmm` 在 NPU 上没有 SparseCsr 后端的
`aten::_sparse_addmm` 注册，调用会报 "Could not run 'aten::addmm.out'
with arguments from the 'SparseCsrnpu' backend"，无任何 NPU 实现。

### SpMM 算子功能分析

C++ 三步法接口参数（对齐 cuSPARSE Generic API）：

| 参数 | 参数含义 | 类型 | 支持范围 | 约束 |
| --- | --- | --- | --- | --- |
| handle | aclsparse 句柄 | handle | 复用已有实现 | 非空 |
| opA / opB | A/B 的转置属性 | 枚举 | opA=N；opB=N/T/C（C 仅 complex64 且非 ALG3） | opA=T/H 明确报错 |
| alpha / beta | 标量乘子 | host/device 指针 | 实数 dtype 用实部；complex64 为复数 | 实数 dtype 虚部非零报错 |
| matA | 稀疏矩阵描述符 | CSR | M×K，索引 I32/I32，idxBase 0/1 | 仅 CSR；I64 明确报错 |
| matB | 稠密矩阵 B | tensor | K×N，ROW/COL 主序 + padding ld | 与 A 同 dtype |
| matC | 稠密矩阵 C | tensor | M×N，ROW/COL 主序 + padding ld | 与 A 同 dtype |
| computeType | 计算类型 | 枚举 | 见下"支持数据类型" | 与 dtype 不匹配报错 |
| alg | 算法枚举 | 枚举 | DEFAULT/CSR_ALG1/CSR_ALG2/CSR_ALG3/FP32_HIGH_PRECISION | 非法值明确报错 |

支持数据类型：float16、bfloat16、float32、complex64（必选，非扩展项）。
computeType 组合对齐 cuSPARSE SpMM 类型表：实数 dtype 对应同型或
float32 计算类型（fp16 允许 FLOAT16 或 FLOAT）；complex64 对应
ACL_COMPLEX64。

# 需求分析（required）

## 需求描述

1. 在 arch22（DAV-2201）上补齐 aclsparse SpMM 的 fp16/bf16/float32/complex64
   全 dtype 能力，覆盖 B/C ROW/COL 布局 + padding ld、opB=N/T/C、idxBase 0/1、
   五种算法枚举的校验与分发，非法组合确定性报错；
2. 具备泛化能力：覆盖不同 shape、nnz、稀疏度、空行、长尾行分布与合法边界输入；
3. 实现 ATen `_sparse_addmm` 的 NPU 注册，打通 `torch.sparse.addmm`
   （input 广播、非连续输入、beta=0 不传播 NaN/Inf），无 CPU fallback；
4. 精度满足《生态算子开源精度标准》混合容差单标杆；A3 上性能达到每个
   "case×dtype"场景 > 0.25×A100 cuSPARSE、算术平均 ≥ 0.35×A100。

## 需求拆解

1. Host 侧：三步法流程、能力校验、tiling 计算、workspace 规划、预处理
   （行重排分桶负载均衡）与预处理结果复用；
2. Kernel 侧：fast 路径（行主 B 直通）与 fallback 路径（全布局标量点积）
   双路径；complex64 的 split-plane 专用方案；fp16/bf16 的 cast 计算；
3. Python/torch 层：`torch.library` 注册、参数转换、预处理缓存、端到端 UT；
4. 测试：C++ 能力 UT、精度 200 用例（fp32/complex64 各 100）、性能 50 用例
   （含 A100 NCU 基线对照）、complex64 专项；
5. 可交付：A2/A3 功能精度自测报告、A3 性能自测报告、设计文档。

# 详细设计（required）

## 算子分析

### 数学公式

```
C = alpha · op(A) · op(B) + beta · C
```

- A：M×K CSR 稀疏矩阵（rowOffsets[M+1]、colIndices[nnz]、values[nnz]，
  列索引已排序且已合并重复坐标，idxBase∈{0,1}）；
- B：K×N 稠密矩阵，ldb ≥ N（ROW）或 ldb ≥ K（COL）；op(B) 为 N/T/C 之一；
- C：M×N 稠密矩阵，ldc 同理；beta=0 时 C 的输入值不参与计算
  （NaN/Inf 不传播）。

按 CSR 结构展开，输出第 i 行：

```
C[i, :] = beta · C[i, :] + alpha · Σ_{j=rs_i}^{re_i} values[j] · op(B)[col[j], :]
```

其中 `[rs_i, re_i)` 为第 i 行在 values/colIndices 中的区间。
即每行是一组"取 B 的行 → 标量乘 → 累加"的 Axpy 序列，天然适合矢量指令。

### 支持数据类型

| A/B/C dtype | computeType | 说明 |
| --- | --- | --- |
| float16 | ACL_FLOAT16 / ACL_FLOAT | kernel 内 cast 到 fp32 计算 |
| bfloat16 | ACL_FLOAT | cast 到 fp32 计算 |
| float32 | ACL_FLOAT | 原生计算 |
| complex64 | ACL_COMPLEX64 | split-plane 方案（见 kernel 设计） |

### 支持形状

M、K、N、nnz 均为 int32 动态形状（≤ INT32_MAX），支持 M=0/N=0/nnz=0、
空行、长尾行（单行 nnz 接近 K）、N 超过 UB 单块上限的多 chunk 场景；
B/C 支持 ROW/COL 主序与 padding ld。

## 算子实现

整体分层：

```
torch.sparse.addmm
   └─ ATen _sparse_addmm (SparseCsrPrivateUse1 注册, python/csrc/sparse_addmm_npu.cpp)
        └─ aclsparseSpMM 三步法 (sparse/spmm/arch22/spmm_host.cpp)
             ├─ Preprocess: 行重排分桶 + rowInfo + (complex64) B split prepass
             └─ SpMM: tiling 下发 + kernel launch (sparse/spmm/arch22/spmm_kernel.cpp)
                  ├─ fast 路径: spmm_arch22_fast_{f32,f16,bf16,c64}
                  └─ fallback 路径: spmm_arch22_fallback_{f32,f16,bf16,c64}
```

### 3.2.1 host 侧设计

#### 三步法职责划分

- `aclsparseSpMMGetBufferSize`：纯 host 计算。校验入参，经
  `aclrtGetDeviceInfo(ACL_DEV_ATTR_AICORE_CORE_NUM)` 取核数（失败兜底 24），
  计算 workspace 总字节数（见下）写回，不触碰 device。
- `aclsparseSpMMPreprocess`：执行昂贵且与 alpha/beta 无关的部分——CSR 行
  重排分桶与 rowInfo 生成（D2H 回读 rowOffsets → 按行 nnz 降序贪心装入当前
  负载最小的 bin，实现跨核负载均衡 → reorder/binEdge/rowInfo 三元组 H2D 写
  入 workspace）；complex64 fast 路径额外在同一 stream 上发射 B split
  prepass kernel；完成后置 `matAInner->activeBuffer = buffer`。
- `aclsparseSpMM`：轻量路径。每次调用重写 tiling（alpha/beta/opB/order 等
  逐次可变参数当次生效）；若 `activeBuffer != buffer`（未预处理或更换了
  workspace）则先就地重建预处理结果再发射 kernel，保证接口语义正确；
  M==0 或 N==0 直接返回成功。

#### workspace 分段布局

单块连续 workspace，64B 对齐分段（偏移量写入 tiling 供 kernel 使用）：

| 段 | 大小 | 内容 |
| --- | --- | --- |
| header | 64B | 预留 |
| tiling | ROUNDUP(sizeof(SpmmArch22TilingData), 64) | tiling 结构 |
| reorder | ROUNDUP(M×4B, 64) | 重排后行→原始行映射 |
| binEdge | ROUNDUP((blockDim+1)×4B, 64) | 各核负责的重排行区间 |
| rowInfo | ROUNDUP(M×3×4B, 64) | 每行三元组 [原始行, rs, re]（保留原始 idxBase） |
| splitB | 仅 complex64 fast：ROUNDUP(K×nChunks×2×nAlign×4B, 64) | B 的实/虚分离平面副本 |

#### tiling 策略（N 向 chunk 切分）

kernel 每行输出为 N 个元素的累加器，受 UB（192KB）限制，N 过大时按 chunk
切分列方向循环：

1. 可用 UB = 192KB − 8KB（系统预留）− 2KB（TPipe 元数据）− 固定开销
   （rowInfo 行块 64×3×4B + colIdx staging 2×2048×4B + values staging
   2×2048×dtype 字节）；
2. 每元素 UB 成本按路径与 dtype 计：fast 路径 f32=72B/元素（B 双槽×8 行×4B
   + 累加器），f16/bf16=42B，c64=112B/复数元素；fallback 按 16B/元素；
3. nMaxPerChunk = 可用 UB ÷ 每元素成本，向下对齐（f32/c64 对齐 8，
   f16/bf16 对齐 16），nChunks = ceil(N / nMaxPerChunk)；
4. tiling 结构仅含标量字段（M/K/N/nnz/blockDim、nMaxPerChunk、nChunks、
   ldb/ldc、orderPair、opB、idxBase、dtype、path、alpha/beta 实虚部、
   各 workspace 段偏移），每次 SpMM 调用整体重写。

#### 分核策略

预处理阶段按行 nnz 降序贪心装箱（GreedyRowBinPack）：把重排后的行依次放入
当前累计 nnz 最小的核，binEdge 记录各核行区间。相比均分行号，长尾行分布下
各核负载（累加次数）基本均衡；无有效 rowOffsets 时退化为均匀分块。
kernel 内重排只改变行的计算顺序，输出按 rowInfo 中的原始行号写回，
结果布局不受影响。

#### 路径选择（替代 tilingkey 的分发策略）

arch22 不使用数字 tilingkey，而是 host 侧 `SelectSpmmPath` 选择路径、
kernel launch 按 `path × dtype` 直接分发到 8 个核函数之一：

- FAST（orderC==ROW 且 (orderB,opB)∈{(ROW,N),(COL,T/C)}）：B 行可直通
  DataCopyPad，是主优化路径；
- FALLBACK：其余布局组合（如 orderC==COL），逐输出元素标量点积，保证
  全能力覆盖、正确性兜底；
- 环境变量 `SPMM_ARCH22_FORCE_FALLBACK` 可强制 fallback 便于对比调试。

算法枚举 alg 在 arch22 上仅做合法性校验（白名单五种，非法值
`NOT_SUPPORTED`），不改变 kernel 路径；`ACL_SPARSE_SPMM_CSR_ALG3` 叠加
"仅 CSR、opA=N、opB≠C"约束校验，与 cuSPARSE 语义对齐。

### 3.2.2 kernel 侧设计

所有 kernel 为 AIV（矢量核）。公共参数：UB 192KB；staging 段长
SEG_LEN=2048；rowInfo 行块 64 行；B 行分组 GROUP_ROWS=8（complex64 为 4）。

#### fast 路径主流水线（fp32/fp16/bf16）

每核处理 binEdge 划定的一段重排行，整体为四级流水：

```
rowInfo(64行/块, DataCopyPad) → colIdx/values 分段 staging(双缓冲队列)
   → B 行组加载(GROUP_ROWS 行/槽, 双槽预取) → Axpy 累加 → (beta) → 写回
```

1. **CopyIn**：rowInfo 按 64 行一块搬入 UB；行内 colIdx/values 按 2048
   分段，经独立双缓冲队列批量搬入（替代热循环内的标量 GM 读）；
2. **B 行加载**：一个 EnQue/DeQue 同步周期覆盖 GROUP_ROWS=8 行
   （2D DataCopyPad 单传输），双槽位保持一组在途预取，隐藏 GM 时延；
3. **Compute**：标量读 staged value（fp16/bf16 位级扩展为 fp32）乘 alpha
   后执行 `Axpy(acc, bRow, val, chunkAlign)` 融合乘加；fp16/bf16 先把
   B 行 `Cast` 到 fp32 再 Axpy，累加器恒为 fp32；
4. **beta**：beta≠0 时复用 B 槽位读入 C 行（cast 后）`Axpy(acc, c, beta)`；
   beta==0 整体跳过，天然满足 NaN/Inf 不传播语义；
5. **CopyOut**：fp32 直接 DataCopyPad 写回；fp16/bf16 `Cast(CAST_RINT)`
   后经输出队列写回原始行地址。

#### complex64 split-plane 方案（本任务的关键设计）

DAV-2201 无复数矢量指令，若直接在交错 [re,im] 布局上计算，每个 nnz 需要
"Gather 拆分 + 2 条 2N-lane Axpy"共 3 条矢量指令且 Gather 开销随 nnz 线性
增长。为此采用 **预处理拆分 + 平面计算** 方案：

1. **prepass kernel `spmm_arch22_c64_split`**（在 Preprocess 阶段发射一次）：
   用两次 Gather（偶/奇字节偏移）把交错 B 拆成按 (k × nChunks) 组织的
   [BR(nAlign) | BI(nAlign)] 平面副本，写入 workspace splitB 段，
   采样期间完全复用；
2. **主 kernel `spmm_arch22_fast_c64`**：累加器为单缓冲 [accR | accI]；
   B 行 [BR|BI] 一次 DataCopyPad 直通，无需逐 nnz Gather；colIdx 与
   values 合并进同一 staging 槽位（每行队列循环从 6 次降为 4 次）；
3. **内环每 nnz 仅 4 条 N-lane Axpy**：复数乘法
   `(er+i·ei)(br+i·bi)` 展开为
   `accR += er·BR + (conjSign·ei)·BI`、`accI += (−conjSign·er)·BI + ei·BR`，
   其中 `conjSign = +1 (opB=C) / −1 (否则)`——共轭转置只体现为标量取号，
   零矢量开销；
4. **beta**：C 行交错读入后用 even/odd Gather 拆分为 [CR|CI]，同样 4 条
   Axpy（C 不共轭）；
5. **行末输出**：一次 Gather 把 [accR|accI] 交错回 [re,im] 后
   DataCopyPad 写出，每行仅一次。

效果：complex64 相对 A100 的性能倍率从初版（逐 nnz Gather）的
0.01–0.08× 提升到 910B4 上算术平均 0.34×（25 case 中 23 个 ≥0.25×）。

#### fallback 路径

逐输出元素标量点积，覆盖全部 order/opB/idxBase 组合；不使用
reorder/binEdge（连续行块均分）。针对 DAV-2201 的两个平台约束做了专门
处理：

1. 标量 GM 写（GlobalTensor::SetValue）经 per-core 写缓冲，与后续读交错会
   丢写——所有输出一律 UB staging + 事件同步 + DataCopyPad 块写出；
2. 16bit 类型标量读写一律位级展开为 fp32，规避半精度标量指令的边界行为。

#### DAV-2201 平台适配要点（已在实现中固化）

- Gather 的 srcOffset 为**字节**语义（与文档元素语义不符），拆分/交错
  偏移均按 4B 倍数计算；小计数 Gather 行为异常，gatherLen 统一按 8 对齐；
- TQue 事件池每类上限 8，流水线深度 ≤2；UB 预算计入全部队列深度的缓冲，
  越界表现为随机错误/全零，host 侧 FixedBytes 已显式扣除；
- Debug/Release 指令开销相差 10–40 倍，交付与性能测试均固定 Release。

### 3.2.3 ATen / Python 层设计

`python/csrc/sparse_addmm_npu.cpp` 以
`TORCH_LIBRARY_IMPL(aten, SparseCsrPrivateUse1, m)` 注册 `_sparse_addmm`：

1. **参数转换与校验**：mat1 必须 sparse_csr 2-D，mat2 dense 2-D，三者同
   device 同 dtype（仅 fp16/bf16/fp32/complex64，不做类型提升）；int64
   索引转 int32 副本；实数 dtype 拒绝虚部非零的 alpha/beta；
2. **广播与 beta 语义**：self 支持标量/[N]/[1,N]/[M,1]/[M,N] 广播；
   beta=0 时跳过 `out.copy_(self)`，NaN/Inf 不传播；
3. **预处理缓存**：按 (crow/col/values 指针, 形状, dtype) 缓存 matA 描述符
   与 workspace（容量 8，FIFO，thread_local），命中时经 activeBuffer 直接
   复用预处理结果——与任务书"描述符、workspace 和 preprocess 结果在正式
   采样期间复用"的性能口径一致；`CANN_OPS_SPARSE_NPU_CACHE=0` 可关闭；
4. **AiCpu 异步竞争规避**：torch_npu 下 complex 的 `copy_`/`contiguous` 落
   AiCPU，与 AiCore kernel 异步竞争实测触发 device fault/结果随机，统一改
   用 `view_as_real`/`view(kFloat)` 的 float 视图完成拷贝与连续化；临时
   张量保活到下一次调用，避免与 AiCPU 内存复用的竞争；
5. 底层固定走 opA=opB=N、ROW 主序、ALG_DEFAULT、idxBase=0 的 fast 路径
   （torch 语义子集），复杂布局由 C++ 接口层覆盖。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas 800I/T A2（Ascend 910B，DAV-2201） | √ |
| Atlas A3（Ascend 910_93，DAV-2201） | √ |

A2/A3 同属 arch22，同一份源码按 SOC_VERSION 编译两份产物
（`--soc=ascend910b3` / `--soc=ascend910_93`），kernel 无硬件差异分支。

## 算子约束限制

1. 仅支持 CSR 稀疏格式、opA=N；索引类型仅 I32/I32，I64 明确报错；
2. A/B/C 必须同 dtype；fp16 的 computeType 仅允许 FLOAT16/FLOAT；
3. opB=C（共轭转置）仅 complex64 且 alg≠CSR_ALG3；fp32+opB=C 明确报错；
4. M/K/N/nnz ≤ INT32_MAX；ldb/ldc 需满足对应主序的最小值约束；
5. fast 路径覆盖 orderC=ROW 且 (orderB,opB)∈{(ROW,N),(COL,T/C)}；其余
   合法组合自动落 fallback（正确性保证，性能非最优）；
6. alg 枚举仅做合法性校验，不改变 kernel 执行路径；
7. Python 层仅暴露 torch 语义子集（opB=N、ROW、ALG_DEFAULT、idxBase=0），
   完整能力经 C++ 接口使用。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述(不涉及说明原因) | 标准来源 |
| --- | --- | --- |
| 精度标准 | 混合容差单标杆：fp32 以 fp64、complex64 以 complex128 CPU 结果为 golden，匹配率 ≥0.99；INF/NAN 按精度标准对应规则（beta=0 不传播）；精度 200 用例（fp32 100 + complex64 100）A2/A3 双平台全通过 | 《生态算子开源精度标准》（任务书引用） |
| 性能标准 | 每个"case×dtype"场景性能倍率 > 0.25×NVIDIA A100（cuSPARSE SpMM NCU kernel 总耗时基线），全部场景算术平均 ≥ 0.35×；每 case 预热 10 次、采样 30 次取中位数及 P90，采样期间复用描述符/workspace/preprocess 结果；仅 A3 提交性能 | 任务书"性能要求" |

自测结果（A2/910B4 参考值）：精度 200/200；性能 fp32 24/25 case ≥0.25×
（mean 0.435×）、complex64 23/25 case ≥0.25×（mean 0.342×）；A3 数据以
`SELF_TEST_REPORT_A3.md`（一键脚本生成）为准。

可维性设计：

- 调试开关：`SPMM_ARCH22_FORCE_FALLBACK`（强制 fallback 对照）、
  `SPMM_SKIP_SPLIT`（关闭 c64 split prepass）、`SPMM_SYNC_BETWEEN`
  （Python 层逐步同步定位）、`CANN_OPS_SPARSE_NPU_CACHE`（缓存开关）；
- 测试资产：C++ 能力 UT（49 功能 + 8 异常 + 100 次资源循环）、端到端
  pytest 26 项（含 dispatch 命中证据、无 CPU 回退）、精度 200、性能 50
  （含 A100 基线 CSV 与自动倍率判定脚本）。

## 兼容性分析

1. **接口兼容**：完全复用 `include/cann_ops_sparse.h` 已有
   `aclsparseSpMM*` 声明，未新增同名/同功能接口；算法枚举在头文件追加
   枚举值（向后兼容，不改变已有枚举数值）；
2. **架构共存**：arch22 改动局限于 `sparse/spmm/arch22/` 与公共头文件，
   与 arch35（ascend950）、arch20（ascend310p）实现解耦；A5 任务若改动
   Host 公共代码，按任务书要求基于已合入版本处理冲突、保留各自硬件分支；
3. **PyTorch 兼容**：通过 `SparseCsrPrivateUse1` dispatch key 注册，不侵入
   torch_npu 本体，不覆盖任何既有 NPU 算子；CPU/CUDA 行为不受影响；
4. **行为兼容**：beta=0 不传播 NaN/Inf、预处理复用语义与 cuSPARSE plan
   语义对齐（调用方原地修改 CSR 内容而指针不变时需重新 Preprocess，与
   cuSPARSE 一致，已在接口注释与代码注释中声明）。

# 附录：算子执行流程详解

## 附 1 总体调用链

```
用户代码  torch.sparse.addmm(input, mat1(CSR), mat2, beta=β, alpha=α)
   │
   ▼
ATen 层  _sparse_addmm  [dispatch: SparseCsrPrivateUse1]
   │      python/csrc/sparse_addmm_npu.cpp
   │      · 校验：2-D / 同 device / 同 dtype ∈ {fp16,bf16,fp32,c64}
   │      · self 广播到 [M,N]；int64 索引转 int32；非连续 mat2 连续化
   │      · β=0 → 跳过 out.copy_(self)（NaN/Inf 不传播）
   │      · c64 拷贝一律走 view_as_real 的 float 视图（规避 AiCpu 竞争）
   │      · 查预处理缓存（指针+形状为 key，容量 8 FIFO，thread_local）
   ▼
aclsparse 三步法  sparse/spmm/arch22/spmm_host.cpp
   │  ① aclsparseSpMMGetBufferSize ── 纯 host 算 workspace 大小
   │  ② aclsparseSpMMPreprocess ──── 行重排分桶 + rowInfo + (c64)B 拆分
   │  ③ aclsparseSpMM ────────────── 重写 tiling + 发射主 kernel
   ▼
Ascend C kernel  sparse/spmm/arch22/spmm_kernel.cpp  (AIV, 按 SOC 20/24 核)
   fast 路径：spmm_arch22_fast_{f32,f16,bf16,c64}
   fallback： spmm_arch22_fallback_{f32,f16,bf16,c64}（冷门布局标量点积）
```

## 附 2 三步法时序

```
GetBufferSize(...)                Preprocess(...)                      SpMM(...)
     │                                 │                                   │
     ▼                                 ▼                                   ▼
 校验入参                       校验 + buffer 非空                   校验入参
 (dtype/布局/opB/alg 白名单)         │                              activeBuffer==buffer?
 取核数 aclrtGetDeviceInfo           ▼                              ├─ 否 → 就地重建预处理
 (失败兜底 24)                 DoPreProcess（昂贵）                  │   （保证语义正确）
     │                          ├ D2H 回读 rowOffsets                ▼
     ▼                          ├ 逐行 nnz 统计                   WriteTiling
 ComputeWsOffsets               ├ 贪心装箱 GreedyRowBinPack      （α/β/opB/order 当次生效）
 header|tiling|reorder|          │  （行按 nnz 降序放入最闲核）        ▼
 binEdge|rowInfo|splitB          ├ 生成 reorder/binEdge/rowInfo   M==0 或 N==0？
     │                          ├ H2D 写入 workspace            ├─ 是 → 直接返回成功
     ▼                          ▼                                   ▼
 *bufferSize = 总字节          WriteTiling                    spmm_arch22_kernel_launch
                                 │                            （按 path×dtype 选核函数，
                                 ▼                             alg 不参与分发）
                          c64 且 fast？──是──► 发射 split prepass
                                 │              spmm_arch22_c64_split
                                 ▼
                          activeBuffer = buffer
                          （之后同 buffer 的 SpMM 不再重复预处理）
```

任务书"采样期间复用描述符/workspace/preprocess 结果"的口径即由
`activeBuffer` 机制 + ATen 层预处理缓存实现：计时循环内每次调用只执行
第三列（WriteTiling + kernel launch）。

## 附 3 主 kernel 单核执行流程（fast 路径，fp32 为例）

```
核 b 启动
 │
 ▼
从 workspace 读 tiling ──► 本核行区间 [binEdge[b], binEdge[b+1])
 │
 ▼
┌─ for 每个 rowInfo 块（64 行，DataCopyPad 入 UB）─────────────────────┐
│  └─ for 块内每一行：取三元组 (origRow, rs, re)                        │
│       └─ for 每个 N-chunk（N > UB 单块时切列，acc 长度 = chunkAlign） │
│            acc ← 0                                                    │
│            ┌─ for seg in [rs,re) 步进 2048 ────────────────────────┐  │
│            │  colIdx/values 分段 staging（深度 2 双缓冲队列）        │  │
│            │  └─ for 每组 8 行（GROUP_ROWS）─────────────────────┐ │  │
│            │     │ 预取下一组：2D DataCopyPad 8 行×chunk → bQueue │ │  │
│            │     │ 当前组：for t in 0..8                          │ │  │
│            │     │    val = stage[t] × α                         │ │  │
│            │     │    Axpy(acc, B_row_t, val)   ← 每 nnz 1 条指令 │ │  │
│            │     └────────────────────────────────────────────────┘ │  │
│            └─────────────────────────────────────────────────────────┘  │
│            β≠0？──是──► Axpy(acc, C行, β)   （复用 bQueue 槽位）       │
│            f16/bf16：Cast(acc→fp16/bf16, RINT)                         │
│            DataCopyPad 写回 Y[origRow, chunk列]                        │
└────────────────────────────────────────────────────────────────────────┘
```

流水线同时维持三级双缓冲（staging 队列、B 行组、输出队列），第 i 组的
Axpy 计算与第 i+1 组的 GM 搬运重叠；行重排只改变计算顺序，写回按
origRow，结果布局与未重排一致。

## 附 4 complex64 数据流（split-plane）

```
Preprocess 阶段（一次性，采样期复用）
  B (K×N, 交错 [re,im]) ──两次 Gather(偶/奇字节偏移)──► splitB：
  按 (k × nChunks) 展开的平面副本  [BR(nAlign) | BI(nAlign)]

主 kernel 每 nnz（无 Gather，4 条 N-lane Axpy）
  accR += er·BR + (conjSign·ei)·BI
  accI += (−conjSign·er)·BI + ei·BR
  其中 (er,ei) = α×value 的实/虚部，conjSign = +1(opB=C) / −1(否则)
  → 共轭转置只是标量取号，零矢量开销

行末输出（每行仅 1 次 Gather）
  [accR | accI] ──Gather(交错偏移)──► [re,im,re,im,...] ──► 写回 Y
```

对比初版"逐 nnz Gather 拆分交错 B"方案：每 nnz 省掉 1 条 2N-lane Gather，
50 case 中 complex64 算术平均倍率从 0.01–0.08×A100 提升到 0.34×（910B4）。

## 附 5 fallback 路径

orderC=COL 等冷门组合逐输出元素标量点积，不做行重排；因 DAV-2201 标量
GM 写经 per-core 写缓冲会丢写，输出一律 UB staging + 事件同步 +
DataCopyPad 块写出。该路径保证全布局正确性兜底，性能不在验收口径内。