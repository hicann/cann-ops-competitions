# aclblasSsbmv 算子设计文档（Ascend C / arch22）

> 任务书：[aclblasSsbmv_task_doc.md](./aclblasSsbmv_task_doc.md)
> 交付件模板：[cann-ops-competitions design_template.md](https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md)
> 目标仓库：[cann/ops-blas](https://gitcode.com/cann/ops-blas)，落盘路径 `blas/sbmv/arch22/`

---

# 需求背景（required）

## 需求来源

昇腾社区任务（2026 年第一批）——`aclblasSsbmv` 算子 A2/A3 开发。验收通过后合入昇腾算子开源仓 [cann/ops-blas](https://gitcode.com/cann/ops-blas) 的 `blas/sbmv/arch22` 目录。

## 背景介绍

### 算子现状分析

ops-blas 仓中 `blas/sbmv/` **当前仅有 arch35 实现**（`ssbmv_kernel.cpp` / `ssbmv_host.cpp` / `ssbmv_tiling_data.h`），面向 Ascend 950PR/950DT。其 `README.md` 明确标注：

| 产品 | 支持情况 |
|------|---------|
| Ascend 950PR / Ascend 950DT | 支持 |
| Atlas A3 训练系列产品 / Atlas A3 推理系列产品 | **不支持** |
| Atlas A2 训练系列产品 / Atlas A2 推理系列产品 | **不支持** |

因此本任务不是"从零新增算子"，而是**为已有算子补齐 arch22（A2/A3）实现**。arch35 版本采用 SIMT 编程模型（`__simt_vf__` / `asc_vf_call` / `threadIdx`，依赖 `simt_api/asc_simt.h`），该模型在 arch22 上不可用——经全仓检索确认，`blas/*/arch22/` 下无任何文件引用 SIMT 头或 `asc_vf_call`，arch22 统一使用经典 Ascend C SIMD（MEMBASE）模型（`TPipe` / `TQue` / `LocalTensor` / `DataCopy` / `Mul` 等向量指令）。**故 arch22 实现需按 SIMD 模型重新设计，不能复用 arch35 内核代码**，仅可复用其参数校验语义与带状索引公式。

### 对标接口

cuBLAS `cublasSsbmv`（语义参考 Netlib `ssbmv.f`）：`y = alpha * A * x + beta * y`。参数序列与 cuBLAS 一一对应，无需额外映射。

---

# 需求分析（required）

## 需求描述

使用 Ascend C 编程语言实现单精度实数对称带状矩阵-向量乘法算子 `aclblasSsbmv`，浮点数据类型 FLOAT32，支持 UPPER/LOWER 两种存储模式、正负步长、以及 n=0 空算子的合法 no-op。

## 需求拆解

1. 实现 `y = alpha * A * x + beta * y`，A 为 n×n 对称带状矩阵，列主序存储，仅存 `uplo` 指定三角的带状部分。
2. 接口签名与 `include/cann_ops_blas.h:156` 已有声明严格一致，禁止定义产品私有平行接口。
3. 参数语义、边界行为、异常返回值与 cuBLAS / Netlib `ssbmv` 对齐。
4. 产品支持表补齐 Atlas A2/A3 系列产品：支持。
5. 性能达到任务书 §3.3 标杆耗时；精度达到生态算子开源精度标准 FLOAT32 档。

---

# 详细设计（required）

## 算子分析

### 数学公式

```
y = alpha * A * x + beta * y
```

其中 A 为 n×n 对称带状矩阵（A = Aᵀ），alpha、beta 为 FLOAT32 标量，x、y 为长度 n 的 FLOAT32 向量。

### 对称带状存储语义

A 仅存储 `uplo` 指定三角的带状部分，含主对角线及 k 条副对角线，列主序存于 `lda × n` 数组中，`lda ≥ k+1`。

沿用 LAPACK 约定（与 arch35 实现、`test/sbmv/ssbmv/ssbmv_golden.h` 的 cblas golden 一致），0-based 下标下元素 `A(r, c)` 在带存储数组中的偏移为：

| uplo | 条件 | 数组偏移 |
|------|------|---------|
| UPPER | r ≤ c | `(k + r - c) + c * lda` |
| UPPER | r > c | `(k + c - r) + r * lda`（对称性，转置读取） |
| LOWER | r ≥ c | `(r - c) + c * lda` |
| LOWER | r < c | `(c - r) + r * lda`（对称性，转置读取） |

即：UPPER 主对角线位于带存储的第 `k` 行，LOWER 主对角线位于第 `0` 行，与 LAPACK `ssbmv` 参考实现逐项对应。

> **注意**：`blas/tbmv/arch22` 的旧版接口 `aclblasStbmv_legacy` 使用的是"转置行主序"带布局（`bandRow * lda + aCol`），与 LAPACK 不同。本算子**不使用**该布局，统一采用 LAPACK 列主序带布局，以保证与 goldens（cblas）及 arch35 语义一致。

### 支持数据类型

FLOAT32（单精度实数）。

### 支持形状

- A：`lda × n`，逻辑上为 n×n 对称带状矩阵
- x：逻辑长度 `1 + (n-1) * abs(incx)`
- y：逻辑长度 `1 + (n-1) * abs(incy)`

## 算子实现

### 实现方案

#### 3.2.1 算法选型

参考 Netlib `ssbmv.f` 的循环结构（按**列**扫掠，每列同时产生"散射累加"与"点积"两类贡献）。对 UPPER，LAPACK 参考实现的第 J 列为：

```fortran
TEMP1 = ALPHA*X(J)
DO I = MAX(1,J-K), J-1
    Y(I) = Y(I) + TEMP1*A(L+I,J)      ! 贡献 B：向量乘加
    TEMP2 = TEMP2 + A(L+I,J)*X(I)     ! 贡献 A：点积
END DO
Y(J) = Y(J) + TEMP1*A(KPLUS1,J) + ALPHA*TEMP2
```

对应到本设计的两类贡献（0-based）：

| 贡献 | 目标 | 形式 | 访存特征 |
|------|------|------|---------|
| **A** | `y[j] += alpha * Σ_{d} AB[d][j] * x[j-k+d]` | 长度 k+1 的点积 | AB 列与 x 窗口**均连续** |
| **B** | `y[j-k+d] += alpha * AB[d][j] * x[j]`（d ∈ [0, k-1]） | 标量广播的向量乘加 | AB 连续，y 目标连续 |

**关键结论**：按列扫掠时，A 的访存是**连续**的；而按行扫掠（逐行点积）时，A 的访存跨列、步长为 `lda-1`，是跨步访问。arch22 是 SIMD 模型，连续访存可被 `DataCopy` 批量搬运与向量指令直接处理，跨步访存则退化为逐元素操作（违反仓内 R1 规范）。**故本设计选择按列扫掠。**

#### 3.2.2 并行切分策略（分核）

采用**行块归属 + 列扫掠 + halo 裁剪**，行归属互不相交：

- `useCoreNum = min(aivCoreNum, max(1, ceil(n / MIN_ROWS_PER_CORE)))`
- 第 `blk` 个核负责行区间 `[r0, r1) = [blk*rowsPerCore, min(n, (blk+1)*rowsPerCore))`，`rowsPerCore = ceil(n / useCoreNum)`
- 该核需要扫掠的列区间：`c0 = max(0, r0-k)`，`c1 = min(n, r1+k)`
- 扫掠每列 j 时，**贡献 A、B 的目标行均裁剪到本核的 `[r0, r1)` 内**

由于行归属互不相交，**对 y 的写入天然无跨核冲突**，因此：

- **无需 `SetAtomicAdd`**（对比 `blas/symv/arch22`、`blas/spmv/arch22` 依赖 MTE3 原子加）
- **无需 workspace 暂存**
- **无核间同步**，单次 kernel 启动即可完成

同时，该结构**对任意 incx/incy（含负步长）都支持多核**——而仓内 symv/spmv 的 arch22 实现因采用原子散射，必须强制 `useCoreNum = 1`。这是本设计相对既有 arch22 算子的结构性改进。

**halo 开销分析**：每核额外扫掠 `2k` 列（列宽 `lda`），冗余数据量 `≈ useCoreNum * 2k * lda`。以性能 case 4（n=4096, k=64, lda=65）为例，基准 A 数据量 `4096*65 = 266,240` floats（1.06 MB），40 核 halo 约 332,800 floats，总访存约 2.3 MB。910B HBM 带宽下约 3 μs，相对 47.74 μs 标杆可忽略。**故 halo 冗余在本算子的规格下不构成瓶颈，设计以正确性与简洁性优先。**

#### 3.2.3 host 侧设计

严格遵循仓内 `agent/skills/repo-op-templates/references/simd-membase/blas/{op}/archxx/op_host.cpp` 模板：**`Validate{Ssbmv}Params` + `Cal{Ssbmv}TilingData` + `Launch{Ssbmv}Kernel` 三段式**，dlog 集成（`OP_LOGE`/`OP_LOGD`/`OP_LOGI`）。

**参数校验顺序**（与 `ssbmv_golden.h` 一致，保证正负向用例行为对齐）：

| 顺序 | 校验项 | 返回值 |
|------|--------|--------|
| 1 | `n < 0` | `ACLBLAS_STATUS_INVALID_VALUE` |
| 2 | `handle == nullptr` | `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| 3 | `n == 0` → 直接返回 | `ACLBLAS_STATUS_SUCCESS`（合法 no-op，不校验其余参数） |
| 4 | `uplo ∉ {ACLBLAS_UPPER, ACLBLAS_LOWER}` | `ACLBLAS_STATUS_INVALID_ENUM` |
| 5 | `k < 0` | `ACLBLAS_STATUS_INVALID_VALUE` |
| 6 | `lda < k+1` | `ACLBLAS_STATUS_INVALID_VALUE` |
| 7 | `incx == 0 \|\| incy == 0` | `ACLBLAS_STATUS_INVALID_VALUE` |
| 8 | `alpha/beta/A/x/y == nullptr` | `ACLBLAS_STATUS_INVALID_VALUE` |

> ⚠ **与现有 arch35 实现的差异**：arch35 的 `ValidateSbmvParams`（`blas/sbmv/arch35/ssbmv_host.cpp:29-32`）对非法 `uplo` 返回 `ACLBLAS_STATUS_INVALID_VALUE`，而任务书 §2.4 与该任务的测试用例（`TC_ED_156 invalid_uplo` → `ACLBLAS_STATUS_INVALID_ENUM`）均要求 **`INVALID_ENUM`**。arch22 实现按任务书执行；同时需同步修正 `test/sbmv/ssbmv/ssbmv_golden.h` 中的 golden 校验，否则该负向用例会出现"golden 返回 INVALID_VALUE、NPU 返回 INVALID_ENUM"的假失败。

**Tiling 计算**：仅做 O(1) 标量运算，无循环、无设备内存分配。

```cpp
struct SsbmvTilingData {
    uint32_t n;
    uint32_t k;
    uint32_t lda;
    uint32_t uplo;          // ACLBLAS_UPPER / ACLBLAS_LOWER
    uint32_t useCoreNum;
    uint32_t rowsPerCore;   // ceil(n / useCoreNum)
    uint32_t xTileLen;      // 本核 x 切片长度（含 halo）
    uint32_t aTileCols;     // 单次搬运的 band 列数
    float    alpha;
    float    beta;
    int64_t  incx;
    int64_t  incy;
};
```

**核数与 tile 尺寸**：

- `aivCoreNum` 经 `GetAivCoreCount()` 动态获取（`blas/common/helper/host_utils.h:75`，内部走 `PlatformAscendCManager::GetCoreNumAiv()`）——**禁止硬编码核数**（仓内 R2 规范）。当前环境 910B1 为 48 AIV，910B3 为 40 AIV，硬编码会导致跨型号行为不一致。
- `MIN_ROWS_PER_CORE` 取 32，避免小 n 场景起过多核。
- `aTileCols` 按 UB 预算推导：`aTileCols = floor(A_TILE_BYTES / (lda * sizeof(float)))`，并对齐到 8 元素（32B）边界。

**Tiling 传递**：以 `const SsbmvTilingData&` 传入 `ssbmv_kernel_do`，kernel launch 时按值拷贝为函数参数（仓内 arch22 标准做法，**不使用** GM tiling buffer，全仓无 `GET_TILING_DATA` 使用）。

**Workspace**：本算子不需要，`ssbmv_kernel_do` 传 `nullptr`。

#### 3.2.4 kernel 侧设计

Kernel 类 `SsbmvAIV`，`TPipe` 在 kernel entry 中创建并以指针传入（R3 规范，**不作为成员变量**）。

**UB 布局**（单核预算，`ub_size = 196,608 B` = 192 KB，见 `AscendC910B*.ini`）：

| 缓冲 | 用途 | 大小 |
|------|------|------|
| `yQue` (VECOUT, BUFFER_NUM=2) | y 行块输出 | `rowsPerCore * 4 B` |
| `xQue` (VECIN, BUFFER_NUM=2) | x 切片（含 halo） | `xTileLen * 4 B` |
| `aQue` (VECIN, BUFFER_NUM=2) | band 列块 | `lda * aTileCols * 4 B` |
| `accQue` (VECCALC) | y 累加器 | `rowsPerCore * 4 B` |

**Process 流程**：

```
Process():
  # ---- 空算子/special case（对齐 Netlib ssbmv 控制流）----
  if (alpha == 0):
      if (beta == 1) return                       # y 不变，位精确
      load y 行块 → (beta==0 ? 置 0 : Muls by beta) → store
      return
  # ---- 主路径：alpha != 0 ----
  Cop入 x 切片 [c0, c1) 到 UB（incx==1: DataCopy/DataCopyPad；否则 Gather/散列）
  acc ← 0
  for (colBlk = c0; colBlk < c1; colBlk += aTileCols):
      搬入 band 列块 A_local[lda][w]（连续 DataCopy，共 lda*w floats）
      for (j = colBlk; j < colBlk+w; ++j):
          # 贡献 A：点积，仅当 j ∈ [r0, r1)
          # 贡献 B：向量乘加，目标行裁剪到 [r0, r1)
  # ---- 收尾（epilogue）----
  load y 行块 → yLocal
  Muls(acc, acc, alpha)
  beta == 0 ? acc : Axpy(acc, yLocal, beta)       # 禁止读 y 当 beta==0
  store y 行块
```

**贡献 A（点积）的向量化**：对列 j，取 `A_local` 中该列的前 `k+1` 个元素与 x 切片窗口 `x[j-k .. j]`，两者均连续且索引同向递增（AB 的第 d 行对应矩阵行 `j-k+d`），可直接用 `Mul` + `ReduceSum`：

```
Mul(tmp, aCol, xWin, validLen);
ReduceSum(tmp, tmp, tmp, validLen);
accLocal[j - r0] += alpha * tmp[0];
```

**贡献 B（向量乘加）的向量化**：对列 j，`A_local[0 .. k-1][j]` 对应矩阵行 `max(0,j-k) .. j-1`，与 y 行索引**同向递增**，裁剪到本核行区间后为一段连续区间：

```
d0 = max(0, k - j)                 # 列 j 的首个有效 band 行
i0 = max(r0, j - k)                # 目标行下界
i1 = min(r1 - 1, j - 1)            # 目标行上界（不含对角线）
Muls(tmp, aCol[d0 + (i0 - (j-k))], xLocal[j - c0], i1 - i0 + 1);
Add(accLocal[i0 - r0], accLocal[i0 - r0], tmp, i1 - i0 + 1);
```

**边界裁剪**（覆盖 `k ≥ n` 退化场景）：

- UPPER 列 j：有效 band 行 `d ∈ [max(0, k-j), k]`，有效矩阵行 `i ∈ [max(0, j-k), j]`
- LOWER 列 j：有效 band 行 `d ∈ [0, min(k, n-1-j)]`，有效矩阵行 `i ∈ [j, min(n-1, j+k)]`

超出范围的行不参与贡献 A/B 的计算，与 LAPACK 参考实现的循环边界 `MAX(1,J-K)` / `MIN(N,J+K)` 完全一致。

**uplo 分支**：以编译期模板参数实现两个实例（`SsbmvAIV<true>` / `SsbmvAIV<false>`），避免运行期分支；仅 `alpha`/`beta`/`incx`/`incy` 等运行期值走分支。

**步长处理**：沿用仓内 arch22 统一约定（`blas/spmv/arch22/sspmv_kernel.cpp:45-52`）：

```cpp
__aicore__ inline uint32_t XPhysicalPos(uint32_t logical) {
    return (incx >= 0) ? (logical * absIncx) : ((n - 1U - logical) * absIncx);
}
```

- `incx == 1`：`DataCopy`/`DataCopyPad` 连续搬入（主路径，性能用例全覆盖）
- `incx == -1`：物理连续但逻辑逆序，按逆序索引映射后仍为连续块
- `|incx| > 1`：x 侧用 `Gather`（索引向量常驻 UB，向量化）收集
- `incy != ±1`：y 侧无向量 scatter 原语，回退 `SetValue`/`GetValue` 标量回写

> ⚠ **R1 规范偏差说明**：`incy != ±1` 路径为逐元素回写，违反仓内 R1「禁止逐元素操作」。该偏差是**已知且被限定**的：Ascend C 无向量 scatter 原语；且全部性能用例（`TC_PF_*`）均为 `incx=incy=1`，该路径不承载性能指标。若后续要求消除，可改为"收集到 UB 后按物理位置批量写回"或引入 workspace 中转。

#### 3.2.5 精度设计

**特例路径（位精确）**：任务书 §3.2 要求 `alpha = 0` 且 alpha 指针非空时跳过矩阵乘，`y = beta*y` 结果位精确。本设计的控制流与 Netlib `ssbmv.f` 完全一致：

| 条件 | 行为 | 位精确性 |
|------|------|---------|
| `alpha == 0 && beta == 1` | y 不变，直接返回 | 精确（不触碰 y） |
| `alpha == 0 && beta == 0` | y 置 0 | 精确 |
| `alpha == 0 && 其他 beta` | `y = beta * y`，单次 `Muls` | 精确（单次乘法，无中间累加） |
| `alpha != 0 && beta == 0` | `y = alpha * acc`，**不读 y** | 见下 |

**`beta == 0` 必须不读 y**：测试用例 `TC_ED_148 beta0_y_nan` 将 y 填为 NaN 且 `beta = 0`，若实现写成 `alpha*acc + beta*y`，按 IEEE 754 `0 * NaN = NaN`，结果将污染为 NaN，而精度校验器对 NaN 判定为**直接失败**（`test/frame/verify.h:298-302`）。Netlib `ssbmv.f` 对 `BETA.EQ.ZERO` 有独立的置零分支，golden 侧输出 0，故算子侧必须同样跳过 y 的读取。**arch35 实现未做此处理（`blas/sbmv/arch35/ssbmv_kernel.cpp:65` 无条件读 y），arch22 实现必须修正。**

**一般路径（容差判定）**：本设计的累加顺序（按列扫掠、双类贡献分别累加）与 golden 的 cblas 逐行点积顺序不同，属非 bit-exact 的正常浮点差异。量级估算：输入 `|A|,|x| ≤ 5`，k ≤ 64，部分和上界约 `65 × 25 ≈ 1.6e3`，FP32 相对误差 2⁻²⁴，129 项累加的绝对误差量级远小于 FLOAT32 档容差（见下）。**无需额外的高精度累加策略。**

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas 800I/T A2（Ascend 910B） | √ |
| Atlas A3 训练/推理系列产品 | √ |

编译目标：`ascend910b*` / `ascend910_93*` → `arch22` → `--npu-arch=dav-2201`（`CMakeLists.txt:21-31`）。

## 算子约束限制

| 约束项 | 内容 |
|--------|------|
| 参数合法性 | `n ≥ 0`、`k ≥ 0`、`lda ≥ k+1`、`incx ≠ 0`、`incy ≠ 0`、alpha/beta/A/x/y 非空 |
| 对称带状语义 | `uplo ∈ {LOWER, UPPER}`；A 仅存指定三角带状部分，未存半带按 A=Aᵀ 隐含引用 |
| 非连续 Tensor | 不支持超出 `incx`/`incy`/`lda` 语义的非连续内存访问 |
| broadcast | 不涉及 |
| dynamic shape | 不涉及，n/k 为运行时入参 |
| 原地与视图 | y 原地覆写，不返回视图 |
| 确定性计算 | 不要求；本设计为确定性实现（无原子加、无核间竞争） |
| 空 Tensor | `n = 0` 合法 no-op，返回成功且不执行计算 |
| 异步执行 | 依赖 `aclblasSetStream` 绑定 stream；host 侧 launch 后直接返回，**不调用 `aclrtSynchronizeStream`** |

---

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | FLOAT32 档混合容差：逐元素 `\|actual - golden\| ≤ atol + rtol × \|golden\|`；`matched_ratio ≥ 0.99` 且 `max_abs_error ≤ 1e-2 或 32×ULP` | 生态算子开源精度标准 |
| 精度 golden | cblas（Netlib BLAS 实数实现 `ssbmv`）单标杆比对，输出向量 y **全向量**逐元素验证 | 任务书 §3.2 |
| 性能标准 | 各性能 case 平均单次耗时 ≤ 任务书 §3.3 标杆；warmup 后有效采样 >50 次取平均 | 任务书 §3.3 |

### ⚠ 精度阈值版本漂移（需确认）

任务书 §3.2 引用的 `docs/zh/ops_precision_standard/experimental_standard.md` 在上游**已更名**为 `mixed_tolerance_standard.md`，且阈值已收紧。两版 FLOAT32 档对比如下：

| 项 | 任务书（旧版） | 上游当前 master（新版） |
|----|--------------|---------------------|
| rtol | 2⁻¹⁰ ≈ 9.77e-4 | **2⁻¹³ ≈ 1.22e-4** |
| atol | 2⁻¹⁶ ≈ 1.53e-5 | **2⁻¹³ ≈ 1.22e-4** |
| required_matched_ratio | 0.99 | 0.99 |
| max_abs_error_limit | 1e-2 或 32×ULP | 1e-2 或 32×ULP |

逐元素阈值 `atol + rtol×|golden|` 在 `|golden|` 较大时**新版显著更严**（如 `|golden| = 10` 时：旧版 9.79e-3，新版 1.34e-3）。**本设计按更严的新版（rtol = 2⁻¹³）做裕量评估**，理由：算子精度标准以合入时的仓库版本为准，且测试工程 `applyMixedTolerance` 对 FLOAT32 使用的默认值（`test/frame/verify.h:250`）即为 `rtol = 0.0009765625 (2⁻¹⁰)`、`atol = 1.52587890625e-5 (2⁻¹⁶)`，恰好是**旧版阈值**；CSV 中的 `mere_threshold = 2⁻¹³` 又对应新版。三处口径不一致，**建议在开发前与验收方确认最终判定口径**，并据此调整 `applyMixedTolerance` 的调用参数。

## 兼容性分析

1. **接口兼容性**：`aclblasSsbmv` 声明既存于 `include/cann_ops_blas.h:156`，arch22 实现不修改声明，与 arch35 实现通过 `SOC_ARCH_DIRS` 编译期隔离，**同一 SoC 下仅一份实现进入链接集，无符号冲突**（先例见 `aclblasStbmv` / `aclblasStbmv_legacy`）。
2. **产品支持表**：需将 `blas/sbmv/README.md:25-27` 中 A2/A3 的"不支持"更新为"支持"。
3. **既有行为不变**：arch35 实现与测试用例不受影响；本设计对 arch35 的**唯一**建议性修正是 `beta == 0` 不读 y（见 §3.2.5）与非法 uplo 返回 `INVALID_ENUM`，需单独评估是否同步回 arch35。
4. **测试工程改动**：`test/sbmv/ssbmv/CMakeLists.txt` 当前为 `if("arch35" IN_LIST SOC_ARCH_DIRS) ... else() ops_blas_add_tests`，在 arch22 下会走 legacy 分支（不链接 gtest），需改为无条件 `ops_blas_add_gtest_tests(${OPS_BLAS} ssbmv_test)`（参照 `test/trsv/strsv/CMakeLists.txt`）。

---

# 附：交付件清单与待办

## 代码交付件

| 路径 | 说明 |
|------|------|
| `blas/sbmv/arch22/ssbmv_tiling_data.h` | tiling 结构（host/kernel 共享） |
| `blas/sbmv/arch22/ssbmv_kernel.h` | `ssbmv_kernel_do` 声明 |
| `blas/sbmv/arch22/ssbmv_kernel.cpp` | `SsbmvAIV` + kernel entry + launcher |
| `blas/sbmv/arch22/ssbmv_host.cpp` | `aclblasSsbmv` + Validate/CalTiling/Launch |
| `blas/sbmv/README.md` | 产品支持表更新 |
| `test/sbmv/ssbmv/arch22/{ssbmv_test.cpp, ssbmv_npu_wrapper.h, ssbmv_test.csv}` | 测试交付件 |
| `test/sbmv/ssbmv/CMakeLists.txt` | 改无条件 gtest 注册 |

> 无需修改 `cmake/asc_devkit_version.cmake` 的 `TENSOR_API_OPS`（该表仅门控 arch35/tensor_api 算子）；`blas/CMakeLists.txt` 的 glob 会自动收集 `arch22/*.cpp`。

## ⚠ 测试用例 CSV 与仓库解析器不兼容（阻塞项）

任务随附的自测用例 `test_cases/ssbmv_test.csv`（1200 条）与仓内解析器 `test/sbmv/ssbmv/ssbmv_param.h` 存在**三处硬性不兼容**，直接使用会导致**全部用例失败**：

| # | 问题 | 任务 CSV | 仓内解析器 | 后果 |
|---|------|---------|-----------|------|
| 1 | `uplo` 枚举拼写 | `ACLBLAS_FILL_MODE_UPPER` | `parseFillMode` 仅识别 `ACLBLAS_UPPER`/`UPPER`/`LOWER`/整型（`csv_loader.h:250-259`） | 名字查表失败 → `std::stoi` 抛异常 → 回退默认值 `0xFF`（INVALID）→ **每条用例的 uplo 都非法**，全部返回 INVALID_ENUM |
| 2 | 填充列名 | `a_fill` / `x_fill` / `y_fill` | 读取 `a` / `x` / `y`（`ssbmv_param.h:41,43,47`） | 列名不匹配 → 回退默认 `"RANDOM"` → **`NULLPTR`、`VALUE_NORM_NAN`、`VALUE_NORM_INF` 等标记被静默忽略**，空指针与特殊值负向用例全部失效 |
| 3 | 标量空指针 | `alpha`/`beta` 列写 `null`（小写） | 无对应机制；`alpha_fill`/`beta_fill` 列在任务 CSV 中**不存在** | `parseFloat("null")` 抛异常回退 0.0 → alpha 变成 0.0 而非 nullptr → `TC_ED_151` 无法命中 INVALID_VALUE |

> 注：列顺序差异（任务 CSV 的 `expect_result` 在第 14 列）**不影响**解析——`csv_map` 按列名索引。`mere_threshold`/`mare_multiplier` 两列亦无影响：sbmv 驱动走 `MIXED_TOLERANCE` 而非 MERE/MARE（`ssbmv_test.cpp:74-77`），这两列不被读取。

**处理建议（二选一，推荐 A）**：

- **A. 改造 CSV 生成脚本**（`test_cases/gen_csv.py`）：输出列名改为 `alpha_fill,a,lda,x,incx,beta_fill,y,incy,random_seed,expect_result`；`uplo` 写 `ACLBLAS_UPPER`/`LOWER`；空指针改为 `alpha_fill=NULLPTR` / `beta_fill=NULLPTR`。1200 条用例的**语义不变**，仅列名与枚举拼写对齐仓内解析器。
- **B. 扩展解析器**：在 `ssbmv_param.h` 中兼容 `ACLBLAS_FILL_MODE_*` 写法与 `a_fill`/`x_fill`/`y_fill` 列名，并支持 `alpha=null`。改动落在测试框架侧，影响面更大，且与仓内其他算子风格不一致。

无论选哪种，开发前需**先用 `--run` 跑通少量用例**验证 CSV 解析链路（例如 `python verify_accuracy.py --filter TC_L0`），再展开全量。

## 环境信息（本次开发环境实测）

| 项 | 本次环境 | 任务书要求 | 影响 |
|----|---------|-----------|------|
| 芯片 | Atlas 800I A2（**910B1**），8 卡 | 性能测试设备 Atlas 800T A2（**910B3**） | 同为 arch22（`dav-2201`），指令集一致；算力指标不同 |
| AI Core | 24 AIC / **48 AIV**，UB 196,608 B | 910B3：20 AIC / **40 AIV**，UB 196,608 B | **核数不同**——进一步印证 R2：必须 `GetAivCoreCount()` 动态获取 |
| CANN | **9.1.1**（`/usr/local/Ascend/cann-9.1.1`） | 9.1.0 | 兼容；`cann-8.5.0` 亦在机 |
| 驱动 | 26.1.1 | — | — |
| OS / CPU | Ubuntu 22.04.5 / Kunpeng-920 aarch64，192 核 / 1.5 TB | — | — |
| 编译器 | `aarch64-linux/bin/ccec`、`bishengir-compile` | — | 支持 `dav-2201` |
| torch / torch_npu | 未安装（仅有 `/usr/local/python3.10.20`，无 conda/venv） | torch ≥ 2.1.0、torch_npu ≥ 2.1.0.post3 | **非阻塞**：测试工程为 C++ GTest + cblas，不依赖 torch；随任务提供的 `verify_accuracy.py` / `verify_performance.py` 仅用标准库（`subprocess/os/sys/re/argparse/csv`），可直接运行 |

> **性能数据可比性提示**：任务书 §3.3 的标杆耗时由 GPU（A100 / CUDA 12.2）实测值 ÷ 0.8 换算而来（如 case 1：`gpu_ms = 0.010857` → `13.57 μs`）。该量级已接近**单次 kernel 启动开销**，属于 launch-bound 而非 compute-bound。因此 arch22 实现的首要性能原则是：**单次 kernel 启动、host 侧零设备内存分配、零同步、最小化 launch 前标量计算**——本设计全部满足。开发时建议用 `npu-smi info -t usages` 与 profiler 剥离 launch 开销，避免误判为计算瓶颈。

## 参考

1. 任务书 §3.3 性能 case 与标杆耗时（4 条，对应 `test_cases/gpu_baseline.csv` 的 `ssbmv-base-000~003`）
2. Netlib `ssbmv.f`：https://www.netlib.org/blas/ssbmv.f
3. cuBLAS `cublasSsbmv`：https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-sbmv
4. 生态算子开源精度标准（当前路径已更名为 `mixed_tolerance_standard.md`）：https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/mixed_tolerance_standard.md
5. 仓内编码规范：`agent/skills/repo-coding-rules/`（R1 禁止逐元素操作 / R2 动态获取 CoreNum / R3 TPipe 禁止成员变量 / R4 TilingData 禁止数组）
6. 仓内算子模板：`agent/skills/repo-op-templates/references/simd-membase/blas/{op}/archxx/`
