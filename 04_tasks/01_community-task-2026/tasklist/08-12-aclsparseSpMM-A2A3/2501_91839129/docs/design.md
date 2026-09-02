# 【社区任务】aclsparseSpMM 算子（A2/A3）设计文档

> 对应社区任务「8月社区任务-aclsparseSpMM 算子开发(A2/A3)」（任务列表第 12 项）。
> 精度与性能指标以官方任务书原件 `aclsparseSpMM_A2A3_task_doc.md` 及配套自测用例包
> `aclsparseSpMM_testCase/` 为准。
>
> 配套代码 PR（`ops-sparse`）：#114（基线边界缺陷修复）、#123（`torch.sparse.addmm`
> NPU 适配层）、#124（fp16/bf16/complex64 + `idxBase=1` + 性能优化）。


# 需求背景（required）

## 需求来源

CANN 社区任务 2026 · 8 月批次「aclsparseSpMM 算子开发(A2/A3)」。

## 背景介绍

### 任务形态：不是从零实现，是扩展一份已合入的极小基线

`ops-sparse` 仓的 `aclsparseSpMM` **C++ 接口已在 `include/cann_ops_sparse.h`
中定义并冻结**，三阶段调用流程为：

```
aclsparseSpMMGetBufferSize  →  aclsparseSpMMPreprocess  →  aclsparseSpMM
```

仓内已有 `sparse/spmm/arch35/`（Ascend 950 专属）实现。**2026-08-17，`xssxyz` 通过
`ops-sparse!92` 向 `sparse/spmm/arch22/` 合入了一份极小的 A2/A3 实现**：仅支持
`fp32/fp32/fp32/fp32`、`opA/opB` 均为 `NON_TRANSPOSE`、仅紧凑 ROW-major（`ld==cols`）、
仅 `idxBase=ZERO`、仅 `ALG_DEFAULT/CSR_ALG1`——覆盖约 **10~15%** 的任务实际范围
（完整范围见下文与第 12.5 节）。

**因此本任务的实质是：把这份已合入的最小基线，扩展到任务书完整要求**，同时修复
基线自身携带的缺陷（见下文「已合入基线审计」）。这对我们是利好：不用从零啃
接口契约与 workspace 布局，且已有基线里现成的缺陷/性能问题是天然的差异化切入点。

目标交付目录：**`sparse/spmm/arch22/`**（在已合入代码基础上扩展，不是新建）。

### 已合入基线审计结论（详见勘查文档第 11 章，逐行核对过）

**3 个缺陷（须修复，各自独立 PR/commit）**：

| 编号 | 缺陷 | 影响 |
| --- | --- | --- |
| D1 | 关键指针（`values`/`colIndices`/`rowOffsets`/B/C 等）缺少空指针校验 | 传 `nullptr` 直接崩溃，而非返回 `ACL_SPARSE_STATUS_INVALID_VALUE` |
| D2 | `Preprocess` 之后再改 `alpha`/`beta` 不生效——tiling 数据在 `Preprocess` 时整体固化，`aclsparseSpMM` 执行时没有重新读取 host 侧最新的 alpha/beta | 违反"改 alpha 后结果应正确"这一基本契约；`sddmm` 已有"仅刷新可变字段"的范式可抄（`sparse/sddmm/README.md:197`、`sddmm_host.cpp:627`） |
| D3 | 缺少 `INT32_MAX` 边界校验 | 超大 shape/nnz 可能整数溢出而不报错，需在校验链里加界并返回 `NOT_SUPPORTED` |

**4 个性能改造点（P1~P4，按此优先级排序，别均摊；最终实现见 3.2.3~3.2.7）**：P1 内循环标量
读 GM（首要瓶颈，几乎所有场景都卡在这里）、P4 complex64 专项吞吐（AoS 交织域下复数
乘加没有现成的向量化路径，投入产出比最差，放最后）、P2 `Muls+Add` 合并为 `Axpy`、
P3 堆优化装箱（长行拆分本身收益≈0，见 13 章，不做）。

**8 处功能缺口**：dtype（仅 fp32，缺 fp16/bf16/complex64）、opA/opB 转置、
COL-major 与 ld padding、idxBase=ONE、`CSR_ALG2/ALG3`、I64 索引组合、
按 computeType 分派 alpha/beta、PyTorch 层完全没有。

### 关键技术判断：arch35 无法机械移植（基线已验证这一点）

`sparse/spmm/arch35/spmm_kernel.cpp` 采用 **Ascend 950 专有的 SIMT 编程模型**
（`asc_simt.h`、`__simt_vf__`、`threadIdx.x`/`blockDim.x` 语义），这套机制在
A2/A3 上不存在。已合入的 `arch22` 基线证实了这一点：它用的是**经典 Ascend C
向量编程模型**（`TPipe`/`TQue`/`GlobalTensor`/`LocalTensor`/`DataCopyPad`/
`Mul`/`ReduceSum`），后续所有扩展都必须遵循这套已定型的模型，不能引入 SIMT 概念。

---

# 需求分析（required）

## 需求描述

在已合入基线之上，把 `aclsparseSpMM` 的 A2/A3 后端扩展到与 `arch35` 完全对齐的
完整语义：`C = alpha * op(A) * op(B) + beta * C`，A 为 CSR 稀疏矩阵，B/C 为稠密
矩阵，支持完整 dtype/转置/布局/索引基址/算法组合，并接入 PyTorch
`torch.sparse.addmm`。接口签名不变，调用方无需感知硬件差异。

## 需求拆解（对照第 12.5 节完整范围表）

1. 修复 D1/D2/D3 三个基线缺陷（独立 PR，优先做，见 T-2）
2. dtype 从仅 `fp32` 扩展到 **`fp16`/`bf16`/`fp32`/`complex64`** 四型
   （`fp16`/`bf16` 用 fp32 累加，`fp32` 可选 Kahan 高精度，`complex64` 用
   complex64 累加、golden 用 complex128 验收）
3. `opA` 从仅 `NON_TRANSPOSE` 扩展到 **`NON_TRANSPOSE`/`TRANSPOSE`/`CONJUGATE_TRANSPOSE`**
   （共轭转置仅对 complex64 有意义）；`opB` 扩展到 `NON_TRANSPOSE`/`TRANSPOSE`
4. B/C 布局从仅紧凑 ROW-major 扩展到 **ROW/COL-major 任意组合 + 带 padding 的 `ld`**
5. `idxBase` 从仅 `ZERO` 扩展到 **`ZERO`/`ONE`**
6. 算法从 `DEFAULT`/`CSR_ALG1` 扩展到 **`DEFAULT`/`CSR_ALG1`/`CSR_ALG2`/`CSR_ALG3`/
   `CSR_FP32_HIGH_PRECISION_ALG`**（任务书 2026-08-19 原件确认：`CSR_ALG2` 对齐行主、
   `CSR_ALG1` 对齐列主、`CSR_ALG3` 仅 CSR+`opA=NON_TRANSPOSE`。`ALG2/ALG3` 需在
   `cann_ops_sparse.h` 新增公开枚举值，属于给已有 enum 追加新成员的**兼容扩展**，
   不改变任何已有枚举语义，无兼容性风险）
7. 接入 `torch.sparse.addmm` / `aten::_sparse_addmm`（PyTorch 2.7+，`torch_npu`）
8. 泛化能力：任意合法 shape、任意稀疏度（含 nnz=0）、非对齐尺寸
9. 精度满足《生态算子开源精度标准》（12.3 节混合容差单标杆）；性能达标
   8 个 case×dtype 逐项 > 0.25×A100，算术平均 ≥ 0.35×（12.2 节，仅 A3 量化）

---

# 详细设计（required）

## 算子分析

### 数学公式

$$
C = \alpha \cdot op(A) \cdot op(B) + \beta \cdot C
$$

- $A \in \mathbb{C}^{m \times k}$（或实数），**CSR 格式**，`opA` 支持
  `NON_TRANSPOSE`/`TRANSPOSE`/`CONJUGATE_TRANSPOSE`（共轭转置仅 complex64）
- $op(B)$ 逻辑形状为 $k \times n$（`opA=TRANSPOSE/CONJUGATE_TRANSPOSE` 时用
  $A$ 的行数 $m$ 作为 contract 维，需按 opA 重新核算 B/C 的期望 shape）
- $C \in \mathbb{R}^{m \times n}$ 或 $\mathbb{C}^{m \times n}$，稠密，既是输入也是输出
- $\alpha, \beta$ 为标量，内存位置由 `aclsparseSetPointerMode` 控制（Host 或 Device）

### 支持数据类型（12.3 节 + 已确认 int8 不在范围内）

| matA/matB/matC | computeType（累加精度） | golden（验收基准精度，仅自测用） |
| --- | --- | --- |
| `ACL_FLOAT16` | `ACL_FLOAT`（fp32 累加） | fp32 |
| `ACL_BFLOAT16` | `ACL_FLOAT`（fp32 累加） | fp32 |
| `ACL_FLOAT` | `ACL_FLOAT`（可选 Kahan 高精度） | **fp64** |
| `ACL_COMPLEX64` | `ACL_COMPLEX64` | **complex128** |

索引类型：行偏移与列索引须为 `ACL_SPARSE_INDEX_32I`（任务书 2026-08-19 原件明确
"不支持 `ACL_SPARSE_INDEX_64I`"——基线目前对 I64 直接拒绝的行为就是**最终正确行为**，
不是缺口，无需改动，标记解除）；索引基址支持 `ACL_SPARSE_INDEX_BASE_ZERO`/`ONE`，
但注意 PyTorch 原生 `torch.sparse_csr_tensor` 只产生 0 基索引，`idxBase=ONE` 在
官方 ATen 自动化评分路径上不可达，只在 C++ 接口层面需要支持（见文首核对纪要第 3 条）。

### 支持形状

- 维度匹配随 `opA` 变化：`NON_TRANSPOSE` 时 `A.cols == B.rows`、`A.rows == C.rows`；
  `TRANSPOSE`/`CONJUGATE_TRANSPOSE` 时用 `A.rows` 作为 contract 维、`A.cols` 作为
  输出行数，`B.cols == C.cols` 恒成立
- 支持 `m`、`k`、`n` 为任意正整数（含 1），不要求 32B 对齐
- 支持 `nnz = 0`（全零稀疏矩阵，退化为 $C \leftarrow \beta C$）
- 支持行内全空、尾部连续空行等非均匀结构
- B / C 支持 `ld > 逻辑宽度` 的 padding 布局（ROW/COL 均需支持）

## 算子实现

### 实现方案

#### 3.2.1 Host 侧设计

**（1）参数校验（在已合入基线校验链上扩展，不是重写）**

| 校验项 | 失败返回 | 状态 |
| --- | --- | --- |
| `handle`/`matA`/`matB`/`matC`/`values`/`colIndices`/`rowOffsets` 为 `nullptr` | `ACL_SPARSE_STATUS_INVALID_VALUE` | **补 D1** |
| `matA` 非 CSR 格式 | `ACL_SPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED` | 已有 |
| 索引类型非 `INDEX_32I` 或行偏移/列索引类型不一致 | `ACL_SPARSE_STATUS_NOT_SUPPORTED` | 已有 |
| `m`/`k`/`n`/`nnz`/`ld` 任一超过 `INT32_MAX` | `ACL_SPARSE_STATUS_NOT_SUPPORTED` | **补 D3** |
| dtype 组合不在四型之内 | `ACL_SPARSE_STATUS_NOT_SUPPORTED` | 需扩展 |
| `opA=CONJUGATE_TRANSPOSE` 且 dtype 非 complex64 | `ACL_SPARSE_STATUS_NOT_SUPPORTED` | 需新增 |
| `alg=CSR_ALG3` 且 `opB=CONJUGATE_TRANSPOSE` | `ACL_SPARSE_STATUS_NOT_SUPPORTED` | 需新增（12.4①表） |
| 维度不匹配（按 opA 核算后） | `ACL_SPARSE_STATUS_INVALID_VALUE` | 需按新 opA 语义调整 |

**（2）修复 D2——alpha/beta 生效方式**

参照 `sddmm` 的范式：`Preprocess` 只固化**不随每次调用变化**的部分（reorder、
bin_edge、UB 切分参数等）；`alpha`/`beta` 在**每次 `aclsparseSpMM` 调用时**从
host/device 指针重新读取并写入本次下发的 tiling，不在 `Preprocess` 阶段固化。
补一条"改 alpha 后结果应正确"的 UT，这条现在应该是红的，修完变绿。

**（3）Workspace 布局**

沿用已合入基线与 arch35 的布局约定（64B 对齐），保证与 arch35 行为一致、
便于交叉验证：

```
[0            , 64          )  64B header（保留 / 对齐）
[tilingOff    , reorderOff  )  SpmmTilingData
[reorderOff   , binEdgeOff  )  int32 reorder[m]           逻辑行 → 原始行
[binEdgeOff   , endOff      )  int32 bin_edge[blockDim+1] 各核行区间
```

`GetBufferSize` 返回 `endOff`，不实际分配。

**（4）分核与负载均衡（Preprocess）——只做堆优化装箱，不做长行拆分**

基线的 `GreedyRowBinPack` 是 `O(m × blockDim)` 线性扫描找负载最小的 bin。
改用**小顶堆**维护 `(负载, bin 号)` 降到 `O(m log blockDim)`，行为完全等价
（本地已用 Python 精确复现并验证，`tools/spmm_golden.py::lpt_bin_pack`；
C++ 原型 `prototype/spmm_row_binpack.h`，74 项宿主机单测通过，与 Python 参考
63 组 fixture 逐位一致）。

> **不做"长行拆分"**：原计划曾设想对超长行（hub 行）做跨核拆分+归约来突破
> 行粒度装箱的理论下界。**已用 P-01/P-02/P-03 官方性能规格实测量化**（勘查
> 文档第 13 章）：这个收益上限在三组规格、六档 blockDim 下全部 ≈1.00x——
> 结构性原因是 P-03 的单行度数理论天花板（`m-1≈244.9万`）本身就低于
> `blockDim≥26` 时的每核平均负载（`≥258万`），行不可分割在这个规模下不可能
>成为瓶颈，与真实数据集的具体 hub 度数无关。**该优化不值得做**，节省下来的
> 工时应投入 P4（complex64 吞吐，最终方案见 3.2.3）。

**（5）Tiling 数据**

在已合入基线的 `SpmmTilingData` 基础上扩展，新增字段支持新维度：

```cpp
struct SpmmArch22TilingData {
    int32_t m, n, k;
    int32_t ldb, ldc;
    int32_t reorderOffset, binEdgeOffset;
    int32_t highPrecision;      // 0=标准, 1=fp32 Kahan
    int32_t opA;                // 0=N, 1=T, 2=H(共轭，仅complex64)
    int32_t opB;                // 0=N, 1=T
    int32_t orderPair;          // RR / RC / CR / CC
    int32_t idxBase;            // 0 / 1（kernel 内统一归一化为 0 基后处理）
    int32_t algVariant;         // DEFAULT/ALG1/ALG2/ALG3/HIGH_PRECISION
    int32_t nTileLen;           // n 方向单次驻留 UB 的列数
    int32_t nTileCnt;           // n 方向分块数
    // alpha/beta 不放这里——每次调用时单独下发，见 D2 修复
};
```

**（6）UB 切分与容量核算——吸取 `ops-sparse!96`（SDDMM）评审教训**

Host 侧按 `GetCoreMemSize` 查得的 UB 容量反解 `nTileLen`：

```
nTileLen = floor( (UB_size × 安全系数) / (每列所需字节数 × BUFFER_NUM) )
nTileCnt = ceil(n / nTileLen)
```

`n` 较小时 `nTileLen = n`（单块装下）；`n` 极大时分多块。**关键教训**（详见
勘查文档第 14 章，SDDMM arch22 的评审因此挂了 17 天）：UB buffer 的分配容量
必须按"单次实际搬进 UB 的最大长度"核算，**不能只卡聚合维度**（如总 nnz、
总 shape）。凡是新增"按行/按 chunk 搬 UB"的路径（P1 的批量搬索引、任意长行
在单核内仍需分批处理其 nnz），都要保证：

- 分批大小严格 ≤ 已分配 buffer 的元素容量（`DataCopyParams.blockLen` 是
  `uint16_t`，当前 `nMax=11776`→47104 字节；做 P2 把 `buffersPerChunk` 降到 3
  后 `nMax=15696`→62784 字节，仍安全，别再往下降）
- 补"实际长度恰好等于 chunk 容量"和"超出一个 chunk"两条边界用例
- `opB`/布局转换（ROW↔COL、TRANSPOSE）**在 device 侧 kernel 内**用
  `DataCopy`/`Gather` 完成，**不做 Host 侧整块转置再搬运**（SDDMM 那条评审
  意见："matY 转置应在 device 侧执行，不需要来回拷贝"）

#### 3.2.2 Kernel 侧设计

**核心设计沿用已合入基线的思路**：按行组织，"标量 × 稠密行 AXPY 累加"，
而非 arch35 的逐元素标量 gather。对逻辑行 `r`（原始行 `row = reorder[r]`），
非零元区间 `[s, e)`：

```
acc[0..nTileLen)  ← 0
for p in [s, e):                          // P1: 分批 DataCopy 进 UB 后再取标量，不逐个 GetValue
    c   = colInd[p]
    v   = values[p] * alpha               // D2 修复后：alpha 每次调用重新读取
    bRow ← op(B) 的第 c 行的当前 n 分块
    acc += v * bRow                       // P2: Axpy 一趟，不拆 Muls+Add
C[row, 当前 n 分块] ← acc + beta * C[row, 当前 n 分块]
```

**dtype 分型扩展**（沿用 `spmv/arch22` 的模板参数风格，一个 dtype 组合一个 `.cpp`）：

| kernel 文件 | ValT | OutT | CompT | 备注 |
| --- | --- | --- | --- | --- |
| `spmm_kernel_f32.cpp` | `float` | `float` | `float` | 已合入 |
| `spmm_kernel_f16.cpp` | `half` | `half` | `float` | 新增，输出钳位 ±65504 |
| `spmm_kernel_bf16.cpp` | `bfloat16_t` | `bfloat16_t` | `float` | 新增 |
| `spmm_kernel_c64.cpp` | `complex64` | `complex64` | `complex64` | 新增，P4 专项优化对象 |

**complex64 专项（P4，最硬的一关）**：每元素 8 字节，UB 容量
减半、chunk 数翻倍。规划阶段设想的「实虚部分离存储」最终**未采用** —— 分离需要
`GatherMask` 反交织、且每个非零元 6 条向量指令；落地方案是全 AoS 交织域计算，详见
3.2.3；`opA=CONJUGATE_TRANSPOSE`
需要对 A 的值取共轭（`conj(v)`），`opB=CONJUGATE_TRANSPOSE`（若支持）同理对
B 取共轭——纯 kernel 内标量共轭，不引入额外搬运。

**`CSR_ALG3` 确定性要求**：固定次序执行、不用无序原子累加，同输入连续执行
≥3 次输出需逐字节一致（12.3 节 bit-wise 判据，`check_bitwise_repeatable()`
已在本地 golden 落地）。已有的按行处理模型天然满足"固定次序"，只需保证不
引入任何依赖调度顺序的原子操作。

**空行与空矩阵**：`s == e` 的行直接写 `C[row,:] = beta * C[row,:]`。
`nnz == 0` 时全部行走此路径。

#### 3.2.3 complex64：全 AoS 交织域计算，不做反交织

GM 上 complex64 是 `{float re; float im;}` 交织（AoS）存储。朴素做法是把实部虚部分离
（反交织）成两个平铺数组、按复数乘法展开、再交织写回，但反交织本身需要 `GatherMask`
这类算子，且每个非零元要 6 条向量指令。

本实现**全程在交织域上计算**。定义"配对换位取负"算子

```
J([x0, x1, x2, x3, …]) = [-x1, x0, -x3, x2, …]
```

即"乘以虚数单位 i"。对交织存储的复向量 `v` 与复标量 `s = sRe + i·sIm`：

```
s * v = sRe * v + sIm * J(v)
```

两项都是"实标量 × 交织向量"，可直接用 `Axpy` 完成。又因 `J` 是线性算子、可与行内求和
交换次序：

```
acc = Σ_k s_k * v_k = Σ_k (sRe_k · v_k) + J( Σ_k (sIm_k · v_k) )
```

因此只需维护两个交织累加器 `accA`/`accB`，**每个非零元只有 2 条向量指令**（两条
`Axpy`），而 `J` 每行末尾只算一次（`Gather` 按预计算的配对下标表换位 + `Mul` 一个固定
±1 符号向量 + `Add`）。`beta*C` 折进同一套分解（`accA += betaRe·C`、`accB += betaIm·C`），
不需要单独的复数乘法。累加器本身就是 AoS，写回零交织开销。

**UB 预算口径**：全 AoS 布局下每个复数元素占 16 个 float 的存储位 —— raw 双缓冲
`2×2=4`、outQueue `2`、两个累加器 `accA/accB` 各 `2`、`J` 的中间结果 `swapped` `2`、
符号表 `sign` `2`、配对下标表 `pairOffset`（`uint32_t`，同宽）`2`。host 侧
`SpmmArch22ComputeNChunksComplex64` 与 kernel 侧 `Init()` 必须用同一口径，否则两侧算出的
chunk 边界会错位。

#### 3.2.4 性能设计：跨行分组与连续行块装箱

真实图数据（ogbn-arxiv 平均行度数 6.9）下，瓶颈不是带宽也不是向量算力，而是**每行的
固定开销** —— `reorder`/`rowOffsets` 的散落标量读，加上 C 矩阵读入写出两条 DMA 与配套的
跨引擎同步事件，约 1392 ns/行。行度数低时这笔开销摊不下去。

据此的设计是**把固定开销按组摊薄**：

1. **跨行分组路径**（`ProcessChunkGrouped`）：一组行一起处理 —— 一次 `Duplicate`/`Muls`
   覆盖整组，C 的读入与写出各只需一组同步事件，非零元的整批 gather 可跨行凑满槽位。
   组内行数 `rowGroup` 由 UB 余量在 `[SPMM_ARCH22_ROW_GROUP_MIN=2,
   SPMM_ARCH22_ROW_GROUP_MAX=64]` 内自适应选取；凑不到 2 行时退回逐行路径（大 N 场景
   本来也不需要，固定开销已被大 N 摊薄）。
2. **host 侧连续行块装箱**：LPT（行划分表）改按连续行块划分，使一组内的行号必然连续，
   kernel 因此能用**单条 DMA** 批量取整组的 `rowOffsets` 与 C，而不是逐行 gather。

**连续性必须由 host 显式声明，不能由 kernel 推断。** kernel 侧曾用"首尾行号之差是否等于
`gCnt-1`"来判断一组是否连续，这是不安全的：当 M 较小、块数凑不到每核 4 块时 host 会退回
逐行装箱，而 `reorder` 按行度数排序，会产生 `[5, 9, 3, 8]` 这类**首尾恰好对上而中间乱序**
的排列，被误判为连续后 kernel 会把打散的行号当作连续区间寻址，结果错误（已在回归中以
5 个小 M 用例暴露）。

因此 tiling 增设 `rowBlockHost` 字段，由 host 写入**实际生效的装箱粒度**，kernel 仅在
`rowBlockHost == rowGroup` 时才信任连续性，否则一律逐行读取。装箱、tiling、kernel 三处的
粒度统一由 `SpmmArch22EffectiveRowBlock` 取值，避免三处各算一遍再对不上。

#### 3.2.5 性能设计：分组路径的软件流水化

3.2.4 摊掉的是**每行**的固定开销，但一批非零元内部 MTE2 与 Vector 仍然完全串行 ——
`AccumulateGroup` 的形态是「一批 B 行 DMA -> 一个 `MTE2_V` 事件等齐 -> 一批 `Axpy` ->
一个 `V_MTE2` 事件防覆写」，搬运的时间藏不进计算里。

设计上三项改动**互为前提，缺一件另两件都不成立**：

1. **列索引与非零值按段整块搬进 UB**（段长 `SPMM_ARCH22_NZ_TILE = 256`），热点里不再回
   GM 读标量。单独做这一步收益有限 —— 探针把两次 GM 标量读整个去掉，P-02 float32 也只
   从 5599 µs 到 4408 µs（1.27x）。它真正的作用是**解开第 2 步的前置条件**：`col` 还在
   GM 里时，要预取第 j+k 个 B 行必须先做一次阻塞式 GM 读拿 `colIndices[j+k]`，流水无从
   加深。这也解释了先前"把 B 行预取深度从 2 加到 8 反而全线慢约 10%"的反常结果。
   段缓冲复用逐行整批路径已有的三块定长 tile，占在 `SPMM_ARCH22_UB_OVERHEAD` 预留的
   4 KB 内（最宽情形为窄类型的 4+2+4 字节/非零元 × 256 = 2.5 KB）。
2. **B 行槽位分 `SPMM_ARCH22_PIPE_STAGES = 3` 个半区轮换**，下一组的 DMA 排在本组
   `Axpy` 之前，MTE2 因此藏进 Vector 的执行里。**取 3 而不是 2 的理由是等待链的位置**：
   一组填槽前必须挡住"上一次读这个半区"的 `Axpy`；两个半区时那次读正是紧邻的上一组，
   于是每组都要把 PIPE_V 排空一次，MTE2 与 V 仍然串行；三个半区时挡的是上上一组，
   它在发本组 DMA 时早已做完，等待为零。
3. **取 `col` 到发 DMA 的依赖链按 8 路展开**：先把 8 个列号读出来，再连发 8 发 DMA。
   读一个立刻发一个的写法会让每发 DMA 都串在一次 UB 标量读的延迟后面；8 次读背靠背
   发射时延迟互相重叠。流水打通之后这是剩下的主要标量开销。

**事件 ID 必须用 `TPipe::AllocEventID` 而不是 `FetchEventID`。** 后者只返回事件池里
第一个空闲位且**不置占用位**（见 CANN `kernel_tpipe_impl.h`：`AllocEventID` 会
`sbitset1`、`FetchEventID` 不会），连续两次调用会拿到同一个 ID，轮换退化成单 ID；而
旗标只是一个 bit，同一 ID 上连着置位两次而中间没人等，会吞掉一个令牌，表现为挂死或
读到未搬完的数据。预占排在所有 `InitBuffer` 之后，先让 `TQue` 从同一批事件池取完；
DAV-2201 的 `QUE_MAX_EVENT` 为 8，3 个 `MTE2_V` 加 2 个 `V_MTE2` 仍有余量。

**令牌配平**（n 组、S = `PIPE_STAGES`、`nVEvt` = S−1）：`MTE2->V` 第 0 组的令牌在序头
置于 ID 0，第 g 组（g ≥ 1）置于第 g−1 轮体内的 ID `g mod S`，第 p 轮等 ID `p mod S`，
共 n 置 n 等。这里**必须给满 S 个 ID 而不是 2 个** —— 第 g+S 组的置位位于第 g+S−1 轮
体内，其前面挡着 `V->MTE2` 的等待，而那面旗由第 g 轮 `Axpy` 之后置出，所以 V 必然已
吃掉第 g 组的令牌；只用 2 个 ID 时 MTE2 会冲过序头那两次"免检"等待并最终死锁。
`V->MTE2` 序头给每个 ID 预置一面旗，第 p 轮先等 ID `p mod nVEvt`、`Axpy` 之后置回同一
ID，退出时每个 ID 各余一面、收尾等掉。

**原实现完整保留，由 tiling 的 `pipeModeHost` 选择**，host 侧读环境变量
`SPMM_ARCH22_PIPE`（缺省走流水，设 `=0` 退回原实现）。这样做的理由有两条：两条路径的
槽位数与分组行数取自不同的 UB 预算函数（`SpmmArch22PipeSlots`/`PipeRowGroup` 对
`SpmmArch22BatchSlots`/`RowGroup`），行度数与 N 的组合又很多，需要能在同一个二进制上
逐场景 A/B 才说得清收益归因；出问题时也有一条不必重新编译就能退回的路。

**开关不影响列分块边界**：流水模式那对预算函数与原来那对一样，只花"按 N 伸缩的预算
之外剩下的 UB"，**不进 `SpmmArch22BytesPerElem` 的口径**，因此 `nChunks` 与 chunk 边界
在两种模式下逐值相同；host 决定装箱粒度时走的是与 `pipeModeHost` 配套的那组函数，
判定结果随 tiling 下发，两侧不会各算一套（若两侧口径漂移，组边界会跨过块边界，
3.2.4 的"行号连续"判定将全部落空）。

**已知代价**：`nAlign` 很宽的场景（如 P-01 的 1440）三个半区会把 UB 占满，导致槽位数
与分组行数都减半，实测 −6.8%。该场景本身是达标线的 6 倍，不影响验收；没有采用"按
`nAlign` 阈值静态关掉流水"的做法 —— CTL 场景的槽位同样从 32 掉到 8，却快了 56%，
说明槽位数不是可靠判据。

#### 3.2.6 起核数：标记 AIV_ONLY 并按 Vector Core 数起 block

arch22 的 SpMM kernel 只使用 Vector 引擎（`DataCopyPad` + `Duplicate`/`Axpy`/`Muls`/
`Add`/`Cast`/`Gather`），从不碰 Cube，因此标记 `KERNEL_TYPE_AIV_ONLY`（与仓库内其他算子
及 `spmm/arch35` 一致）。A2 上每个 AI Core 带 2 个 Vector Core（实测
`AICORE_CORE_NUM=20`、`VECTOR_CORE_NUM=40`），若按 AI Core 数起 block 会有一半向量算力
闲置。`GetSpmmBlockDim()` 改取 `ACL_DEV_ATTR_VECTOR_CORE_NUM`（取不到时回退 AI Core 数）。

下游的 workspace 尺寸、`binEdge` 数组、LPT 行划分都是从 `GetSpmmBlockDim()` 推导的，
无需改动别处；行数少于核数时，分不到行的核直接 return。

#### 3.2.7 同步原语：裸 TBuf 跨引擎访问必须用 SetFlag/WaitFlag

`PipeBarrier<PIPE_X>()` 与 `SetFlag/WaitFlag<HardEvent::A_B>` **不是强弱之分，而是管的
事情不同**：前者只保证单条流水线内部的顺序，后者才建立跨引擎的数据可见性。

具体到本算子：非 float dtype 的标量取值路径上有一块单元素 scratch，会被同一行的多个非零
元反复复用，涉及 MTE2 的 DMA、Vector 的 `Cast`、Scalar 的 `GetValue` 三个引擎。用
`PipeBarrier` 无法约束它们之间的四个危险（MTE2→V、V→S，以及下一次迭代 MTE2 写回踩掉本次
V 仍在读的 WAR），表现为"行度数 0/1 永远正确、≥2 时间歇性读到垃圾值、每次错得不一样"。
正确写法是 `SetFlag`/`WaitFlag` 配 `HardEvent::MTE2_V`、`V_S`、`V_MTE2` 三对事件，事件 ID
从框架事件池 `FetchEventID` 取（与 `TQue` 经 `AllocEventID` 预留的 ID 不冲突）。

**判据**：凡是裸 `TBuf`（没有 `TQue` 的 EnQue/DeQue 依赖跟踪）被两个不同引擎先后访问，
就必须用事件而不是 `PipeBarrier`。出现"小规模能过、压力上来就错、且每次错得不一样"的
现象时，应当优先怀疑同步原语用错，而不是去找数据逻辑的错。

### 目录结构（在已合入基线上扩展）

```
sparse/spmm/arch22/
├── spmm.h                       // 常量、dtype 编码、workspace 布局工具
├── spmm_tiling_data.h           // SpmmArch22TilingData（扩展）
├── spmm_host.cpp                // 三接口实现（补 D1/D2/D3 + 新维度校验）
├── spmm_csr_mat.cpp/.h          // 堆优化装箱（替换 O(m×blockDim) 扫描）
└── kernels/
    ├── spmm_kernel.h            // 模板 kernel 主体
    ├── spmm_kernel_group.h      // 跨行分组路径与其流水化 SpmmArch22GroupPath（见 3.2.4/3.2.5）
    ├── spmm_kernel_common.h     // 公共基类：列分块遍历、分组行加载、游标推进
    ├── spmm_kernel.cpp          // 已合入，仅 fp32
    ├── spmm_kernel_f16.cpp      // 新增
    ├── spmm_kernel_bf16.cpp     // 新增
    └── spmm_kernel_c64.cpp      // 新增，P4 专项
```

### PyTorch ATen 层（T-7，需 torch_npu 真机）

注册 `aten::_sparse_addmm` 与 `aten::addmm.out` 到 dispatch key
`SparseCsrPrivateUse1`，使 `torch.sparse.addmm` 在 NPU 上的接口、返回值、
dtype、shape、device、异常行为与 PyTorch 2.7+ 一致。底层直接调用上述三段式
`aclsparseSpMM` 接口。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas A2 训练系列产品 / Atlas A2 推理系列产品 | √ |
| Atlas A3 训练系列产品 / Atlas A3 推理系列产品 | √ |
| Ascend 950PR / Ascend 950DT | 不涉及（已由 arch35 覆盖） |

## 算子约束限制

下表按"本次实现是否覆盖"分两组。凡未覆盖的组合都在 host 入参校验里**显式返回
`ACL_SPARSE_STATUS_NOT_SUPPORTED`**，而不是放行后静默算错 —— 三个对外入口
（`GetBufferSize`/`Preprocess`/`SpMM`）共用同一个 `ValidateSpmmInputs`，实现见
`spmm_host.cpp` 的 `ValidateSpmmOperations`/`IsSupportedDnMatLayout`/
`IsSupportedSpmmAlg`/`IsSupportedSpmmDtypeCombo`。

**已实现并在真机验证**

1. `matA` 仅支持 **CSR** 格式（COO / CSC 返回 `MATRIX_TYPE_NOT_SUPPORTED`）
2. `matA` 行偏移与列索引类型须**均为 `ACL_SPARSE_INDEX_32I`** 且一致
3. 索引基址支持 **`ACL_SPARSE_INDEX_BASE_ZERO` 与 `ONE`**（1-based 已实现，kernel
   侧统一减 `idxBaseOffset` 换算，见各 `InitTiling`）
4. dtype 组合支持四型：**fp32 / fp16 / bf16 / complex64**。实数三型要求
   `computeType == ACL_FLOAT`（fp16/bf16 用 fp32 累加），complex64 要求
   `computeType == ACL_COMPLEX64`，且 A/B/C 三者 dtype 一致。**不支持 int8**
5. 算法支持 **`ACL_SPARSE_SPMM_ALG_DEFAULT`** 与 **`ACL_SPARSE_SPMM_CSR_ALG1`**
6. 调用 `aclsparseSpMM` 前**必须**先调用 `aclsparseSpMMPreprocess`
7. `buffer` 不可为 `nullptr`，须按 `GetBufferSize` 返回的大小分配
8. 维度须满足 `rows <= INT32_MAX`、`cols/nnz/matB->cols <= UINT32_MAX`（host 与
   kernel 内部把这些量收窄到 32 位，超限会静默回绕，故前置拒绝）

**任务书要求的完整范围中，本次尚未覆盖的部分（显式拒绝，非静默错算）**

9. `opA`/`opB` 仅支持 `NON_TRANSPOSE`。任务书要求的 `TRANSPOSE` 与
   `CONJUGATE_TRANSPOSE` 目前返回 `NOT_SUPPORTED`
10. 稠密矩阵 B/C 仅支持**紧凑 ROW-major**（`order == ACL_SPARSE_ORDER_ROW` 且
    `ld == cols`）。COL-major 与带 padding 的 ROW-major 返回 `NOT_SUPPORTED`：
    kernel 按 `B[col*N+j]`、`C[row*N+j]` 硬编码寻址，不读描述符的 `order`/`ld`，
    放行会得到静默错误的结果
11. `ACL_SPARSE_SPMM_CSR_ALG2`、`ACL_SPARSE_SPMM_CSR_ALG3`、
    `ACL_SPARSE_SPMM_CSR_FP32_HIGH_PRECISION_ALG` 返回 `NOT_SUPPORTED`
    （注：任务书原件对 `CSR_ALG3` 的约束是"仅支持 CSR、`opA=NON_TRANSPOSE`、
    不支持 `opB=CONJUGATE_TRANSPOSE`"，2026-08-19 逐字确认；`opA`/`opB`/布局/
    dtype/算法的完整合法组合表以 cuSPARSE 13.3 Update 1 官方 SpMM 支持矩阵为准）

## 已知约束与被否决的实现方案

以下方案在实现过程中评估或试过，结论是不采用。列出以免后续重复投入。

| 方案 | 结论 | 原因 |
| --- | --- | --- |
| complex64 用 `GatherMask` 反交织成实虚两个平铺数组再计算 | **否决** | count 型 Level-2 向量 API 存在未文档化的元素数上限（`GatherMask` 仅在 mask=1024 的场景被验证过），且每个非零元需 6 条向量指令。改用 3.2.3 的全 AoS 方案后不再调用 `GatherMask`，chunk 上限也随之从 1024 放宽到 2048 |
| 加深 B 行 DMA 预取深度（1 → 3、2 → 8） | **当时否决，后被 3.2.5 部分推翻** | 单独加深深度实测无收益甚至倒退（2 → 8 全线慢约 10%）：`col` 还在 GM 里时，预取第 j+k 个 B 行必须先做一次阻塞式 GM 读，深度加得越多阻塞读越多。当时据此归因为"瓶颈是每行固定开销而非 DMA 延迟"，只对了一半 —— 把列索引整段搬进 UB 之后加深流水是有效的，见 3.2.5 |
| `beta == 0` 时跳过读 C（`skipBeta`） | **否决** | 与评分 golden 冲突。CPU 侧 `torch.sparse.addmm` 会把 `0.0 * NaN` 算成 `NaN`，而比较器要求 NaN/Inf 位置与 golden 完全一致；跳过读 C 会得到 `0`。任务书正文允许该优化，但以 golden 为准 |
| B 行复用 / 列分块 | **未实施（已知缺口）** | N 较小时每个非零元都要 gather 一行 B，同一列被多行复用却未在 UB 中留存，这是 P-02/P-03 的 float32 与 complex64 仍不达标的主因。是继续突破 0.25 线的关键路径，工作量大于本轮全部改动之和 |

### 复现性提示：`build.sh` 默认 Debug 构建

`build.sh` 的构建类型默认为 Debug（`BUILD_TYPE="${CMAKE_BUILD_TYPE:-Debug}"`），
Debug 产物约 19 MB，实测比 `CMAKE_BUILD_TYPE=Release`（约 1.0 MB）慢一个数量级。
任何性能相关的验证都要显式带上 `CMAKE_BUILD_TYPE=Release`，判据是
`ls -la build_out/lib64/libops_sparse.so` 的产物大小；功能与精度结论不受构建
类型影响。

### UB 预算的记法约束

host 与 device 两侧的 UB 预算必须**逐项列举、每项注明对应的 buffer**，不能写成合并后的
常数。曾经把 `accBuf` 与 `bFloatBuf` 合并成一项，导致 half/bf16 每元素实占 18 字节却按
14 估算，`nMax` 被高估，`N ≥ 10917` 时 `aclrtSynchronizeStream` 返回设备错误 507035。

这个缺陷躲过了"两侧一致性"检查 —— host 侧用的是**同一个错误口径**，所以两侧算出的 chunk
边界始终一致、不会错位，而两侧是一致地错。**"两侧一致"不能替代"两侧都对"。**

### tiling 复用的快路径签名

复用同一 workspace 的快路径（不重新 Preprocess）除 buffer 指针外，还必须比对稀疏结构签名
`(rows, cols, nnz, N)`。`N` 必须在签名内：workspace 大小只与 `matA` 行数和核数有关、不随
`N` 变化，因此"同一块 workspace 配不同宽度的 B"是调用方合法用法，而 `N` 及由它推导的
`nChunks` 都写在设备侧 tiling 里 —— 漏掉 `N` 会让 kernel 按旧的行距寻址 B/C。

# 可维可测分析

## 精度标准/性能标准（12.2/12.3 节三角测量结果，★★★ 高置信度）

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | 混合容差单标杆：`\|actual-golden\|≤atol+rtol×\|golden\|` 且匹配率≥0.99 且单元素误差≤`max(A,32×ULP(golden))`；具体 rtol/atol/A 按 dtype 见勘查文档 12.3 表 | 三角测量（两份逐字一致） |
| 性能标准 | 8 个 case×dtype 逐项 > 0.25×A100，算术平均 ≥ 0.35×；**性能仅在 A3 量化**，功能精度需 A2+A3 双份 | 三角测量（三份逐字一致，含具体 A100 μs 数） |
| 用例规模 | 官方精度 200 条 + 性能 50 条自测包（见 U-1），另需自行按同口径扩展 fp16/bf16 用例 | #1056 |
| 确定性 | `CSR_ALG3` 需 bit-wise 可重复（同输入连续执行 ≥3 次逐字节一致） | 三角测量 |

### 自测方案

已在本地（无 NPU）完成 golden 参考与用例矩阵，可直接用于自测与交叉验证：

**CPU golden 参考实现**（`tools/spmm_golden.py`）：严格按接口契约实现，
覆盖 4 种 order_pair × 2 种 opB × 3 种 opA × 4 种 dtype 组合 × 2 种 idxBase ×
任意 alpha/beta × 带 padding 的 ld × Kahan 路径，并复现了 LPT 装箱。
device 精度与 golden 精度（fp64/complex128 累加）分离为两条独立路径，自带
一组针对 golden 自身实现的自校验断言（与 `scipy.sparse` 逐元素比对）。

**用例矩阵**（`cases/spmm_cases.json` / `.csv`）：3 组固定 + 200 组泛化，
覆盖 dtype×opA×opB×order_pair×idxBase×alg 全部轴，落盘并回读验证。

**P-01~P-03 性能等效用例**（`tools/gen_perf_cases.py` → `cases/perf_local/`）：
按官方性能规格合成统计特征吻合的 CSR，量化 LPT 装箱 makespan 与理论下界
（结论见上文"不做长行拆分"）。

精度验收使用 [ATK](https://gitcode.com/Ascend/ATK) 做单标杆评估。

**交叉验证策略**：A2/A3 结果同时与 CPU golden 和 arch35（950，功能重叠部分）
结果比对，多方一致才认为通过。

## 兼容性分析

本任务为**已有接口扩展硬件后端实现范围**，不改动 `include/cann_ops_sparse.h`
中任何既有函数签名或枚举语义；`CSR_ALG2`/`CSR_ALG3` 是新增的兼容性枚举扩展
（若任务书确认需要），不影响现有调用方。新增/修改代码全部位于
`sparse/spmm/arch22/`，与 `sparse/spmv` 的 arch22/arch35 并存方式一致。
