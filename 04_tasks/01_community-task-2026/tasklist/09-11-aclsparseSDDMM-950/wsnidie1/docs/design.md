# 需求背景（required）

## 需求来源

昇腾 CANN 社区任务 2026 年 9 月社区任务 —— aclsparseSDDMM 算子开发（950），任务列表编号 11。要求在 ops-sparse 工程（基线 master `3f0e150`）上，参考 cuSPARSE `cusparseSDDMM_bufferSize` / `cusparseSDDMM_preprocess` / `cusparseSDDMM` 三阶段接口语义与 PyTorch `torch.sparse.sampled_addmm`，完善 Ascend 950（DAV_3510，`arch35`，A5）实现，并将 Python/torch 层、ATen NPU 注册、aclsparse C++ 接口、Ascend C Kernel 和测试统一交付。

## 背景介绍

### aclsparseSDDMM 算子现状分析

基于 ops-sparse master `3f0e150` 分析，`aclsparseSDDMM` 现有能力与本任务缺口如下：

| 项目 | 既有能力 | 本任务缺口 |
| --- | --- | --- |
| 稀疏格式 | CSR（arch35）、CSR/COO 旧实现（arch22） | BSR（块稀疏）不支持 |
| values 数据类型 | FP32 / FP16 | BF16、complex64、FP16 输入 + FP32 values 混合组合不支持 |
| 索引 | I32，base 0/1 | BSR 块行/块列索引语义缺失 |
| Dense 布局 | ROW / COL，op(X)/op(Y) 全组合 | strided batch（batchStride/batchCount 校验）缺失 |
| Python/ATen | 旧 `python/ops_sparse_torch/` 布局（已被上游 Torch Extension 框架取代） | 需按 `sparse/<op>/torch_extension/` 新布局适配 `aten::sparse_sampled_addmm` |
| 性能 | FP16/FP32 CSR 快路径（SIMT/Cube） | BSR 与 complex64 走通用标量内核，性能与标杆差距大 |
| 分核一致性 | 内核按 `GetBlockNum()` 分核 | 与 host 写入的分核表可能不一致，导致部分行重复写/漏写 |

### aclsparseSDDMM 算子功能分析

算子按稀疏矩阵 `C` 的样点模式，计算两个稠密矩阵乘积在样点位置上的取样结果，并与 `C` 原值按 `alpha`/`beta` 组合，分三阶段：

1. `aclsparseSDDMMBufferSize`：校验参数组合并查询 workspace 字节数，校验含 strided batch、`batchStride` 覆盖性、BSR 块尺寸与块网格；
2. `aclsparseSDDMMPreprocess`：建立调用级状态——把 X/Y 重新打包为 kernel 友好布局、按组划分行并计算列窗口、生成 pattern 重排与分核表、缓存 pattern 指纹；
3. `aclsparseSDDMM`：执行取样乘加，`C.values` 为输入输出缓冲区（batch=1 原地更新；batch>1 返回逐 batch 输出）。

语义约束：`C.values` 参与 `beta` 项且为就地缓冲区；样点模式只读；同一输入重复执行的 nnz、结构与 values 位级一致；不支持的 dtype/format 组合必须显式返回错误，不得静默降级。

# 需求分析（required）

## 需求描述

使用 Ascend C 在 arch35 上完善 `aclsparseSDDMM`：values 支持 FP16、BF16、FP32、complex64 及 FP16 输入 + FP32 values 混合组合；支持 CSR 与 BSR（blockSize 2/4/8/16/32/64/128）；Device 索引 I32，index base 0/1；支持 op(X)/op(Y) 全部 transpose 组合、ROW/COL 布局与 strided batch；完成 Python/ATen 适配（PyTorch 2.7+，不得 CPU fallback）；性能达到任务书 GPU 同口径验收线。

## 需求拆解

1. 补齐 BSR 描述符与 strided batch 公共接口（`aclsparseCreateBsr` / `aclsparseCreateConstBsr` / `aclsparseDnMatSetStridedBatch` / `aclsparseDnMatGetStridedBatch` / `aclsparseSpMatSetStridedBatch` / `aclsparseBsrSetStridedBatch`）；
2. host 侧重构：6 组 dtype 组合、base 0/1、BSR/批校验、pattern cache（按 matC 绑定 + 128bit 内容哈希 + buffer 代数）；
3. kernel 侧三条执行路径：Cube 快路径（正式形状）、SIMT 快路径、通用标量内核（BSR / complex64 / 混合 dtype / batch 等功能路径）；
4. 分核一致性修复：tiling 携带 host 分核核数，内核按同值分核并与分核表（binEdge）一致；
5. Python/ATen：按上游 Torch Extension 框架交付 `sparse/sddmm/torch_extension/`，注册 `aten::sparse_sampled_addmm` 到 `SparseCsrPrivateUse1`，并提供 `torch_npu.sparse.sampled_addmm` façade；
6. 测试：CSV gtest harness 扩展 complex64/bf16 输入构造与复数 golden；smoke 覆盖 6 组 dtype、CSR/BSR、base 0/1、batch；ATen 端到端 UT；官方 200 条精度用例与性能锚点；
7. 公共 Host 交叉回归（A2/A3/A5）与跨架构编译。
8. 性能：保持 FP16/FP32 CSR 快路径达标，并给出各锚点实测倍率与已知差距。

# 详细设计（required）

## 算子分析

### 数学公式

对稀疏矩阵 `C ∈ K^{m×n}`（样点集 `spy(C) = {(i,j) : C_ij ≠ 0}`）与稠密矩阵 `X`、`Y`：

```text
C_ij ← alpha · Σ_t op(X)_it · op(Y)_tj + beta · C_ij ,   (i,j) ∈ spy(C)
```

其中 `op(·)` 由 `opX`/`opY` 决定（`NON_TRANSPOSE` 或 `TRANSPOSE`），`alpha`/`beta` 为标量（pointer mode HOST）。

- 实数类型：`K` 为 IEEE 半精度/单精度/bfloat16 域，累加在 FP32 进行，FP16 输出前按 ±65504 截断（与 `cuda` 语义一致）；
- complex64：`K` 为复数域，按复数乘加计算，标量以 `(re, im)` 两个 float 传入；
- strided batch：`X`/`Y`/`C` 的 batch 维可各自为 1 或 `batchCount`（广播语义由 stride=0 表达），逐 batch 独立计算。

### 支持数据类型

| `X` / `Y` | `C.values` | computeType | 组合标识 | 说明 |
| --- | --- | --- | --- | --- |
| FP32 | FP32 | FP32 | `SDDMM_DTYPE_FP32` | — |
| FP16 | FP16 | FP32 | `SDDMM_DTYPE_FP16` | 输入输出同 dtype，计算收敛到 FP32 |
| FP16 | FP32 | FP32 | `SDDMM_DTYPE_F16_C32` | FP16 输入 + FP32 values 混合组合（f16c32） |
| BF16 | BF16 | FP32 | `SDDMM_DTYPE_BF16` | 仅 BSR（官方清单中无 `csr×bfloat16`，故 CSR+bf16 返回不支持） |
| BF16 | FP32 | FP32 | `SDDMM_DTYPE_BF16_C32` | 仅 BSR |
| complex64 | complex64 | complex64 | `SDDMM_DTYPE_C64` | 复数乘加 |

`IsSupportedSddmmDtypeCombo` 的组合表另保留基线既有组合（FP16 输入输出 + `computeType=FP16`，判为同一 `SDDMM_DTYPE_FP16`）以向后兼容；任务书组合表已收敛为 FP32 computeType。

索引固定 I32（`ACL_SPARSE_INDEX_32I`），base 0/1。

### 支持形状

- CSR：`m×n`，`crow_indices` 长度 `m+1`，`col_indices`/`values` 长度 `nnz`；base 0 的首尾为 `0/nnz`，base 1 为 `1/nnz+1`；
- BSR：块行数 `blockRows`、块列数 `blockCols`，`rowBlockSize = colBlockSize = blockSize ∈ {2,4,8,16,32,64,128}`，`values` 形状 `(blockNnz, blockSize, blockSize)`，块内布局 ROW/COL 由 `direction` 指定；
- 稠密：`X` 为 `(batchX, m, k)`、`Y` 为 `(batchY, k, n)`（按 opX/opY 与 order 解释），合法 leading-dimension padding；
- batch：`1 ≤ batchCount ≤ 65535`；`batchX`/`batchY` 各自为 1 或 `batchCount`；
- 空样点（`nnz = 0`）为合法输入。

## 算子实现

### 实现方案

#### 3.2.1 host侧设计：

##### 1. 分核策略：

按执行路径分别取核数并写入 tiling，内核**按 tiling 中的核数分核**，不使用 `GetBlockNum()` 自行推导，避免与 host 写入的分核表（`binEdge`）越界不一致：

- **Cube 快路径**：AIC 核数取 `GetSddmmAicBlockDim()`；host 把行按列窗口切分为 chunk（`window = align16(lastCol - firstCol + 64)`），以 `cost = rows × window` 做**最长优先装箱**分配到各 AIC 核（`bins/baseGroups/extraGroups`），再按轮次轮转输出 `cubeGroups`，使各核负载均衡且相邻 chunk 的列窗口局部性最好；
- **SIMT / 通用路径**：AIV 核数取缓存值或 `GetCoreNumAiv()`；通用内核按核数跨步遍历（块）行，保证每行恰被处理一次；
- `Preprocess` 生成的 `reorder`、`binEdge`、`cubeGroups`、`patternHash` 等全部落在 workspace 内，`bufferGen` 代数与 pattern 内容哈希共同保证跨 buffer 复用的正确性。

##### 2. 数据分块和内存优化策略：

**三条执行路径的门控**（host 侧判定，全部条件满足才进入对应快路径，否则逐级回退到通用内核）：

| 路径 | 门控条件（要点） |
| --- | --- |
| Cube 快路径 | 先满足 SIMT 门控，再要求：`k`（`matX.cols`）`== 128`、`m×n` 属于正式锚点形状集合（8192×28672 / 4096×1536 / 7168×2048）、`cubeRowCount ≥ 16`；FP16 还要求 `alpha == 1`（Fixpipe 在乘 alpha 前先量化到 FP16，非 1 的 alpha 会二次舍入，故留在 SIMT 路径） |
| SIMT 快路径 | `uniformNnzPerRow == 64`（每行恰 64 个非零）、`k == 128`、CSR、`opX`/`opY` 均为 `NON_TRANSPOSE`、X/Y 均 `ROW`、X/Y/C 均单 batch、dtype 组合为 FP32 或 FP16；FP32 另要求 `nnz ≥ 8·m`。`alpha` 可任意；`beta != 0` 时 FP32 走分片 relaunch（每 launch 每核处理 bin 内一行，规避 β·C 项的确定性丢失），FP16 的 `beta != 0` 直接落通用内核 |
| 通用内核 | BSR、complex64、f16c32、bf16（含 bf16_c32）、batch>1、非 uniform 样点、非 128 的 `k`、非 N/N 的 op 组合、空样点等全部功能场景；`beta` 从 workspace 的不可变备份单遍读取 |

**Cube 快路径的内存预算与数据分块**（fp16，UB/L1 视角）：

| 缓冲 | 大小 | 用途 |
| --- | --- | --- |
| `a1_` / `a2_` | `kGroupRows(64) × 128 × 2B = 16 KB` × 2 | X 打包块（A1 → A2） |
| `b1_` / `b2_` | `128 × 256 × 2B = 64 KB` × 2 | Y 窗口（B1 → B2） |
| `c1_` | `64 × 272 × 4B ≈ 68 KB` | CO1 上的 FP32 累加结果 |

- 列窗口宽度 `kTempN = 272`（fp16）为**布局不变量**：它同时决定 `temp` 的行跨距、`colOffset` 取值范围与 wrap（跨 `n` 边界）行的偏移编码，因此不可调整；
- 每个 chunk 的窗口覆盖其行列跨度，Fixpipe 按 `dstStride = kTempN` 写入 `temp`；
- 跨 `n` 边界的行由 `packedYWrap` 与 `cubeWrapStart` 单独处理，窗口装不下的行进入离群（outlier）集合，由 scatter 阶段以标量路径补算；
- **workspace 组成**：batchedX / packedY(wrap) / pattern 重排与分核表 / `temp`（`cubeRowCount × kTempN`）/ scatter 元数据 / 离群位置表，全部由 `GetBufferSize` 一次性返回，`Analysis` 与 `Convert` 复用同一块。

**pattern cache**：按 `matC` 绑定，键为 `(ptrs, idxs, rows, cols, nnz, k, format, base, rowBlockSize, batchCount)` 加上 128bit 内容哈希与 buffer 代数；缓存不含调用级状态（不缓存裸指针/stream），符合公共层对句柄缓存的约束。

##### 3. tilingkey规划策略：

以 `dispatchId`（dtype 组合 × format × base × batch 维）作为派发键，kernel 侧按 `dataType`/`format`/`batchCount` 选择路径：

| dispatchId | 对应组合 | 执行路径 |
| --- | --- | --- |
| `SDDMM_DTYPE_FP32` | FP32 → FP32 | Cube（正式形状）／SIMT（`nnz ≥ 8·m`）／通用 |
| `SDDMM_DTYPE_FP16` | FP16 → FP16（computeType 收敛 FP32） | SIMT（`beta == 0`；`alpha == 1` 且正式形状时升 Cube）／通用 |
| `SDDMM_DTYPE_F16_C32` | FP16 输入 → FP32 values | 通用内核 |
| `SDDMM_DTYPE_BF16` | BF16 → BF16（仅 BSR） | 通用内核 |
| `SDDMM_DTYPE_BF16_C32` | BF16 输入 → FP32 values（仅 BSR） | 通用内核 |
| `SDDMM_DTYPE_C64` | complex64 → complex64 | 通用内核 |

#### 3.2.2 kernel侧设计：

`Preprocess` 之后 `Compute` 阶段按路径执行：

1. **Cube 快路径（fp16/fp32 CSR）**：三个内核串行——
   - `sddmm_cube_pack_*`（AIV/SIMT）：按 `reorder` 把 X 重新打包为 `packedX`（每线程 4 个 half），并生成 wrap 行的 `packedYWrap`；
   - `sddmm_cube_compute_*`（AIC）：按 `cubeGroups` 逐 chunk 搬入 A（`packedX`）与 B（Y 窗口），`Mmad` 计算 `64×128×curN`，`Fixpipe` 以 `ndNum/mSize/nSize/srcStride/dstStride` 写 `temp`（fp16 走 `QuantMode_t::F322F16`，含 ±65504 截断）；
   - `sddmm_cube_scatter_*`（AIV/SIMT）：按 `scatterMeta`（每行打包 `row | (rowColOffset << 32)`）把 `temp` 的窗口元素写回 `CSR values`，并对 wrap/离群行执行标量补算（`OutlierRange`）。

2. **通用内核（BSR / complex64 / f16c32 / bf16 / batch / 非 uniform）**：`sddmm_generic_kernel` 以标量实现全组合功能路径——按核数跨步处理各行，逐 `(i,j)` 做点积并按 `alpha/beta` 组合，BSR 按块内 ROW/COL 解码、batch 按 stride 偏移逐 batch 输出。

3. **同步与确定性**：内核内不依赖原子操作；每行恰由一个核处理一次，输出顺序由样点模式的确定性顺序决定；标量以 host 指针传入（pointer mode HOST），complex64 传 `(re, im)` 双 float。

4. **公共 Host 复用**：`Preprocess` 的 pattern 重排/分核/窗口计算与 `Compute` 的坐标计算共用 `aclsparse_descr*` 中的描述符访问器；BSR/批相关能力通过新增公共接口暴露，供其他算子复用。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Ascend 950PR（Atlas 350，DAV_3510 / arch35） | √ |
| Atlas 800I/T A2、A3 | 公共 Host 交叉回归通过（155 项 × 3 配置）；本任务不声明 A2/A3 真机算子回归 |

## 算子约束限制

- 仅支持 CSR 与 BSR；COO 等其他 layout 由 arch22 旧实现覆盖，不在本任务范围；
- `C.values` 必须连续（batch=1 时算子就地更新该缓冲区）；
- 不支持的 dtype 组合（如 CSR + bfloat16、CSR + 混合精度之外的组合）显式返回 `ACL_SPARSE_STATUS_NOT_SUPPORTED` / `INVALID_VALUE`，不静默降级；
- BSR 块网格必须整覆盖逻辑矩阵（`blockRows × blockSize == m`、`blockCols × blockSize == n`）；
- 大 K 采用分片处理；`batchCount > 65535` 报参数错误；
- 执行保持异步，不在每次调用末尾同步；workspace 由 `GetBufferSize` 查询、经公共层 `AclSparseWorkspace` 申请并记录到当前 stream。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述(不涉及说明原因) | 标准来源 |
| --- | --- | --- |
| 精度标准 | 以 CPU FP64/complex128 golden 为基准，nnz/结构/坐标 exact match；values 按官方判据 `SparseSddmmExactOrMixed`（exact `allclose(rtol=1e-5, atol=1e-8)`，否则框架 MIXED_TOLERANCE）逐条判定 | 任务书 §3.2 与 `test_cases/.../accuracy_sparse_ops.py` |
| 性能标准 | 性能倍率 = GPU 标杆 median_us / NPU 同调用范围耗时，≥ 0.3（口径与 cuSPARSE 标杆一致：预分配描述符/workspace/输出，设备 Event 计时） | 任务书 §3.3 |

**实测（Ascend 950PR / CANN 9.1.0 / torch 2.7.1+cpu / torch_npu 2.7.1.post8）**：

| 项 | 结果 |
| --- | --- |
| 官方 200 条精度用例 | 200/200 通过（exact 150 + 混合容差 50） |
| CSV gtest（arch35 套件） | 89/89，含 complex64 / f16c32 新增用例 |
| smoke（6 组 dtype、CSR/BSR、base 0/1、batch、opX/opY、ROW/COL） | 23/23 |
| ATen 端到端（`torch.sparse.sampled_addmm`） | 12/12（wheel 安装 → JIT 首次调用） |
| 公共 Host 交叉回归（A2/A3/A5） | `COMMON_DESCRIPTOR_PASS 155` × 3 |
| 性能（CSR FP16 快路径，三锚点 median） | P-01 61.9 µs（0.336）、P-02 35.3 µs（0.419）、P-03 41.7 µs（0.485），均过 0.3 验收线 |
| 已知差距 | BSR 与 complex64 由通用标量内核承担，当前实测单次调用 4.2–9.0 s，与标杆差距大，需要向量化优化（后续迭代） |

## 兼容性分析

- 新算子能力扩展，不改变既有 API 语义；BSR/strided batch 为新增公共接口，既有算子不受影响；
- 公共 Host（`sparse/common/aclsparse_descr.cpp`、`aclsparse_descr_internal.h`）的改动在 A2/A3/A5 三种构建配置下各 155 项描述符回归一致，且跨架构编译通过；
- 上游 `python/ops_sparse_torch/` 已被 Torch Extension 框架取代，本任务按新布局交付，不引入对旧布局的依赖。