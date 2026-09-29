# aclblasCtrmm 算子设计文档（A2/A3）

## 1 需求背景（required）

### 1.1 需求来源

CANN 社区任务 2026（9 月批次）：在昇腾 NPU（Atlas A2/A3 系列产品，性能验收设备为 Atlas 800T A2 (910B3)）上，基于 Ascend C / CATLASS 编程语言开发单精度复数（complex64）三角矩阵乘算子 `aclblasCtrmm`，对标 cuBLAS `cublasCtrmm`，完成算子设计、开发、测试全流程。验收通过后以 PR 形式合入昇腾算子开源仓 ops-blas（https://gitcode.com/cann/ops-blas ），实现代码位于 `blas/trmm/arch22/`，测试代码位于 `test/trmm/ctrmm/arch22/`。

任务书：`aclblasCtrmm_A2A3_task_doc.md`。

### 1.2 背景介绍

#### 1.2.1 aclblasCtrmm 算子功能

`aclblasCtrmm` 执行单精度复数三角矩阵-矩阵乘（BLAS Level 3），离席计算（结果写入 C）：

- `side = ACLBLAS_SIDE_LEFT`：`C = alpha * op(A) * B`
- `side = ACLBLAS_SIDE_RIGHT`：`C = alpha * B * op(A)`

其中 `op(A)` 由 `trans` 决定：`ACLBLAS_OP_N`（A）、`ACLBLAS_OP_T`（Aᵀ）、`ACLBLAS_OP_C`（Aᴴ，复共轭转置）；`A` 为三角矩阵，上/下三角由 `uplo` 决定，主对角是否读取由 `diag` 决定（`ACLBLAS_UNIT` 时对角视为 1 且不读取）。该算子广泛用于矩阵分解（QR/LU 后的三角回代相关计算）、线性方程组求解、协方差矩阵更新等科学计算与机器学习预处理场景，是 cuBLAS / Netlib BLAS 能力对齐的必备算子。

#### 1.2.2 aclblasCtrmm 算子现状分析

1. **开源仓现状**：ops-blas 仓中已有实数版本 `aclblasStrmm`（`blas/trmm/`，当前含 arch35 实现），尚无单精度复数版本 `aclblasCtrmm`。实数版本只涉及实数乘加；复数版本需要处理复数乘法（实虚部交叉项）与 `trans=ACLBLAS_OP_C` 的复共轭转置语义，不能直接复用实数实现。
2. **硬件能力**：目标平台 Atlas A2/A3 为 DAV_2201（arch22）架构，Cube 单元 MMAD 原生面向实数数据格式。复数矩阵乘需按"实虚部分解为实数 GEMM"的经典恒等式实现（§3.1.2），复共轭在数据准备阶段对虚部取负完成，Cube 阶段只跑实数 FP32 矩阵乘，精度路径与 Netlib BLAS `ctrmm` 语义严格对齐。
3. **工程模式**：本算子采用 kernel 直调（Ascend C）模式，基于 ops-blas 仓工程框架实现句柄式 BLAS 接口：通过 `aclblasHandle_t` 绑定 stream，host 侧完成参数校验与 tiling 后直调 NPU kernel。参照仓内 `strmm` 的多 kernel 编排模式（数据准备 → 矩阵乘 → 缩放写回），本算子采用三段式 kernel 流水线（§3.2.0），所有 24 组 (side, uplo, trans, diag) 组合通过 tiling 参数在 kernel 内运行时分支，不膨胀 kernel 实例数量。

#### 1.2.3 aclblasCtrmm 算子功能分析

| 项目 | 内容 |
| --- | --- |
| 输入 | `handle`、`side`、`uplo`、`trans`、`diag`、`m`、`n`、复数标量 `alpha`（Host 内存）、复数三角矩阵 `A`（Device 只读，列主序）、前导维 `lda`、复数一般矩阵 `B`（Device 只读，列主序）、前导维 `ldb` |
| 输出 | 复数一般矩阵 `C`（Device，离席写入，列主序）、前导维 `ldc` |
| 支持数据类型 | COMPLEX64（`aclblasComplex`，实部/虚部均为 float32，交错排布） |
| 维度关系 | `side=LEFT`：A 为 m×m，B/C 为 m×n；`side=RIGHT`：A 为 n×n，B/C 为 m×n |
| 数据排布 | 列主序（Column-Major），元素 (i,j) 位于 `i + j*ld` |
| 广播 | 不涉及（A/B/C 为独立矩阵） |
| 动态 shape | 不要求，m/n 为运行时入参，由 host tiling 支持 |
| 原地语义 | 允许 B≡C 原地写出（cuBLAS trmm 等价原地用法）；除此之外 C 与 A/B 不允许内存重叠 |
| 异步 | 依赖 `aclblasSetStream` 绑定 stream；读回结果前须同步 stream |
| 对标基线接口 | cuBLAS `cublasCtrmm`（参数序列一一对应，无需映射） |

## 2 需求分析（required）

### 2.1 需求描述

使用 Ascend C 编程语言实现 `aclblasCtrmm`，提供句柄式 BLAS 接口，通过 handle 绑定 stream 直调 NPU kernel，完成单精度复数三角矩阵乘计算。支持 `side=LEFT/RIGHT`、`uplo=UPPER/LOWER`、`trans=N/T/C`、`diag=NON_UNIT/UNIT` 共 2×2×3×2=24 组枚举组合，覆盖合法输入、边界与负向校验，满足任务书 §3.2 精度标准与 §3.3 性能标准。

### 2.2 需求拆解

1. **功能实现**：`side=LEFT` 时 `C = alpha * op(A) * B`；`side=RIGHT` 时 `C = alpha * B * op(A)`。`op(A)` 由 `trans` 决定（N/T/C，其中 C 为复共轭转置）。
2. **三角裁剪**：严格按 `uplo` 指定的三角区域读取 A；`diag=UNIT` 时主对角视为 1 且不读取，`diag=NON_UNIT` 时读取对角元素（乘法语义下对角允许为零）。`side` 决定 A 的阶数（LEFT 为 m×m，RIGHT 为 n×n），不得混淆。
3. **维度与排布**：列主序。`side=LEFT` 时 A 为 m×m（lda ≥ max(1,m)）；`side=RIGHT` 时 A 为 n×n（lda ≥ max(1,n)）；B、C 为 m×n，ldb ≥ max(1,m)，ldc ≥ max(1,m)。
4. **边界与 quick return**：`m=0` 或 `n=0` 为合法 no-op（返回成功，不执行计算）；`alpha=(0,0)` 时 A、B 不被引用，C 全部置零；负维度、非法枚举、非法前导维、空指针等负向场景返回对应状态码。
5. **接口与校验**：在 `include/cann_ops_blas.h` 新增声明，签名与 cuBLAS `cublasCtrmm` 逐参数对齐；host 侧完成全部参数合法性校验。
6. **精度达标**：输出 C 全矩阵与 cblas（Netlib BLAS `ctrmm`）golden 混合容差比对，COMPLEX64 标准下 matched_ratio ≥ 0.99 且 max_abs_error 达标（§4.1）。
7. **性能达标**：Atlas 800T A2 (910B3) 上 COMPLEX64 输入场景，先 warmup 再有效采样取平均，5 个基准 case 平均单次耗时无低于任务书标杆（§4.2）。

## 3 详细设计（required）

### 3.1 算子分析

#### 3.1.1 数学公式

三角矩阵乘定义为（列主序）：

```
C = alpha * op(A) * B      (side = LEFT)
C = alpha * B * op(A)      (side = RIGHT)
```

其中 `op(A)` 由 `trans` 决定：

| trans | op(A) | 元素访问（列主序，A 阶数为 a = side 对应阶数） |
| --- | --- | --- |
| ACLBLAS_OP_N | A | `op(A)[i][k] = A[i + k*lda]` |
| ACLBLAS_OP_T | Aᵀ | `op(A)[i][k] = A[k + i*lda]` |
| ACLBLAS_OP_C | Aᴴ（复共轭转置） | `op(A)[i][k] = conjg(A[k + i*lda])`，即虚部取负 |

**复数乘法语义**：设 `a = (ar, ai)`、`b = (br, bi)`，则

```
a * b = (ar*br - ai*bi) + i*(ar*bi + ai*br)
```

三角裁剪规则（以 A 的 (i,k) 为索引）：

| uplo | k 的访问范围 |
| --- | --- |
| ACLBLAS_UPPER | k ≥ i |
| ACLBLAS_LOWER | k ≤ i |

对角规则：`diag=NON_UNIT` 时对角元素从 A 读取（允许为零）；`diag=UNIT` 时对角视为 (1,0)，**不读取** A 的对角位置（越界或脏数据不影响结果）。

#### 3.1.2 复数-实数 GEMM 分解（实现核心恒等式）

DAV_2201（arch22）Cube MMAD 原生面向实数格式，复数矩阵乘按经典恒等式分解为实数 GEMM。以 `side=LEFT`、`op(A) = A`（M×M，实部 Re、虚部 Im）为例：

```
A * B = (Re(A) + i*Im(A)) * (Re(B) + i*Im(B))
      = [Re(A)Re(B) - Im(A)Im(B)]  +  i * [Re(A)Im(B) + Im(A)Re(B)]
```

即 4 次实数 GEMM 合成 1 次复数 GEMM。利用 GEMM 结果的线性性合并为 **2 次实数 GEMM**（每块 2 路）：

```
T1 = Re(A) * Re(B) - Im(A) * Im(B)      （输出实部）
T2 = Re(A) * Im(B) + Im(A) * Re(B)      （输出虚部）
C = T1 + i*T2
```

**实现上按输出块构造拼接 GEMM**：将 A 的实/虚部按块交错拼成 `A' = [Re(A) | -Im(A) | Re(A) | Im(A)]`（宽度 4×K，按 M×4K 排布），将 B 的实/虚部拼成 `B' = [Re(B) ; Im(B) ; Re(B) ; Im(B)]`（4K×N），则

```
C' = A' * B' = [ T1  T2 ]   （M × 2N）
```

一次实数 GEMM 同时算出实部 T1 与虚部 T2，Cube 利用率与实数 GEMM 相同。`trans=OP_C` 的共轭语义在 A 侧数据准备阶段完成：拼 `A'` 时对相应块取 `-Im(Aᴴ)` 即共轭转置的虚部，无需单独 kernel。

#### 3.1.3 side=RIGHT 的归约

`side=RIGHT` 时计算 `C = alpha * B * op(A)`。将等式两边转置：

```
Cᵀ = op(A)ᵀ * Bᵀ
```

`op(A)ᵀ` 与 `op(A)` 同族（N→N，T↔C 互换、uplo 互换、diag 不变），因此 RIGHT 路径可完全复用 LEFT 的 GEMM 流水：对 `op(A)ᵀ`（n×n）× `Bᵀ`（n×m）做 LEFT 式计算得到 `Cᵀ`（n×m），最后按列主序写回即得 C。转置不显式构造，通过数据准备阶段的访问下标映射完成。

#### 3.1.4 支持数据类型

- 输入/输出矩阵 A、B、C：`aclblasComplex`（COMPLEX64，实部/虚部均为 float32，内存交错排布）
- 标量 alpha：`aclblasComplex`（Host 内存）
- 中间计算：FP32 实数（T1/T2 及 alpha 乘均在 FP32 下完成，无额外舍入）

#### 3.1.5 支持形状

- A：`side=LEFT` 时 m×m；`side=RIGHT` 时 n×n
- B、C：m×n
- m、n 覆盖 0 ~ 大规模（含奇数、非对齐、2 的幂 ±1），由 host tiling 运行时确定，无固定 shape 特化
- 无广播；前导维 lda/ldb/ldc 支持紧凑与任意 padding

### 3.2 算子实现

#### 3.2.0 总体流水线

参照仓内 `strmm` 的多 kernel 编排模式，本算子采用三段式 kernel 流水线（host 侧顺序下发到同一 stream，天然异步衔接）：

| 阶段 | kernel | 计算单元 | 职责 |
| --- | --- | --- | --- |
| 1. 数据准备（Mirror/Prepare） | `ctrmm_prepare_kernel` | AIV（Vector） | 读取三角区 A（按 uplo/diag 裁剪，UNIT 对角补 1），按 §3.1.2/§3.1.3 规则把 A、B 的实/虚部拼成 GEMM 输入 `A'`/`B'`（GM workspace） |
| 2. 矩阵乘（GEMM） | `ctrmm_gemm_kernel` | AIC（Cube） | 实数 FP32 GEMM：`C' = A' × B'`（M×4K × 4K×2N），输出实/虚部拼接矩阵到 GM 临时区 |
| 3. 缩放写回（Scale/Epilogue） | `ctrmm_scale_kernel` | AIV（Vector） | `C = alpha * C'`（复数乘，Host 读入 alpha 值传入 tiling），按 ldc 写回 C |

设计要点：

1. **24 组枚举组合零分支膨胀**：side/uplo/trans/diag 全部编码进 tiling 参数，kernel 内运行时分支（§3.2.2），host 侧不膨胀 kernel 实例与二进制体积。
2. **三角性只体现在阶段 1**：A 只读取三角区（`diag=UNIT` 时对角补 (1,0) 且不读 A），GEMM 阶段按完整矩形 M×4K×4K×2N 计算——裁剪收益由阶段 1 的数据量与 GEMM 输入构造保证，Cube 阶段保持满矩形最高效率。
3. **workspace**：`A'`（M×4K×4B）、`B'`（4K×2N×4B）、`C'`（M×2N×4B），host 侧按 tiling 计算后通过 stream 申请/释放，kernel 结束即回收；不依赖 C 与 A/B 之外的重叠。
4. **小 shape 退化**：M×N 很小时（taskBlock < 可用 AIC 核数），GEMM 单核完成或退化为向量直算路径，避免 cube 发射开销主导延迟（§3.2.4 小 shape 策略）。
5. **B≡C 原地**：阶段 3 按行块读入-写回，读块与写块为同一缓冲区内的不同行区间且块内先读完再写，原地语义安全（与 cuBLAS trmm 等价用法一致）。

#### 3.2.1 host 侧设计

**接口设计**：在 `include/cann_ops_blas.h` 新增如下声明（参数序列与 cuBLAS 逐一对应）：

```cpp
aclblasStatus_t aclblasCtrmm(
    aclblasHandle_t handle,
    aclblasSideMode_t side,
    aclblasFillMode_t uplo,
    aclblasOperation_t trans,
    aclblasDiagType_t diag,
    int m, int n,
    const aclblasComplex* alpha,
    const aclblasComplex* A, int lda,
    const aclblasComplex* B, int ldb,
    aclblasComplex* C, int ldc);
```

**参数校验策略**（顺序执行，任一失败即返回，不发起 kernel）：

1. `handle == nullptr` → `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`。
2. `side`/`uplo`/`trans`/`diag` 不在合法枚举内 → `ACLBLAS_STATUS_INVALID_VALUE`。
3. `m < 0` 或 `n < 0` → `ACLBLAS_STATUS_INVALID_VALUE`。
4. `alpha == nullptr` → `ACLBLAS_STATUS_INVALID_VALUE`。
5. `m == 0` 或 `n == 0` → quick return `ACLBLAS_STATUS_SUCCESS`（合法 no-op，不读 A/B）。
6. 前导维：`lda >= max(1, dimA)`（dimA = side 对应阶数）、`ldb >= max(1, m)`、`ldc >= max(1, m)`，不满足 → `ACLBLAS_STATUS_INVALID_VALUE`。
7. `alpha=(0,0)`（host 读取）→ A、B 不被引用，C 置零（下发单核向量化置零 kernel 或复用 scale kernel 的 alpha=0 分支），返回成功。
8. `A`/`B`/`C` 为 nullptr（矩阵引用非空场景）→ `ACLBLAS_STATUS_INVALID_VALUE`。

**Tiling 策略**：

host tiling 分三层，与 Ascend C MatMul 高阶 API 的三级 tiling 架构（L2 级 → Block 级核间切分 → BaseBlock 级 cube 发射）对齐：

- **L2 级（M/N 方向）**：M×N 输出按 `l2M × l2N` 的 base 块网格组织，serpentine 遍历提升 L2 命中率（A' 的 M 行块与 B' 的 N 列块在相邻核间共享）。
- **核间切分（Block 级）**：总 base 块数 = ceil(M/singleCoreM) × ceil(2N/singleCoreN)（GEMM 输出为 M×2N 实/虚部拼接）。按 AIC 核数均分，余数块分给前几个核，保证负载均衡；核内 M 优先、N 次之的遍历顺序保证数据局部性。
- **核内切分（BaseBlock 级）**：`tileM × tileN × tileKChunk` 由 L1 容量（arch22：A1+B1 双缓冲约束）反推：

```
aSideL1  = align(tileM, baseM) * align(tileKChunk, baseK) * 4B
bSideL1  = align(tileKChunk, baseK) * align(tileN, baseN) * 4B
while (L1_BUF_NUM * (aSideL1 + bSideL1) > L1_SIZE && tileKChunk > baseK): tileKChunk /= 2
```

  baseM/baseN/baseK 取 Cube MMAD 最小发射粒度（16/16/8 量级），默认 tile 128×128×256，超 L1 容量时 K 维逐步对半降。

**tiling 数据结构**（`CtrmmTilingData`，POD，host 填充后随 kernel 下发）：

| 字段 | 说明 |
| --- | --- |
| sideMode / uploMode / transMode / diagMode | 4 个枚举编码（uint8），kernel 运行时分支依据 |
| m / n / dimA | 输出行/列数与 A 的阶数 |
| lda / ldb / ldc | 前导维 |
| usedAicCoreNum / usedAivCoreNum | GEMM/准备/缩放阶段使用的 AIC/AIV 核数 |
| singleCoreM / singleCoreN | 每核 M/N 维 base 块数（核间切分结果） |
| tileM / tileN / tileKChunk | 核内 cube 发射粒度（L1 约束反推） |
| totalBlock / mBlockCnt / nBlockCnt | base 块网格与总块数（serpentine 遍历） |
| alphaReal / alphaImag | host 读入的 alpha 实/虚部（FP32），scale 阶段使用 |
| wsOffsetA / wsOffsetB / wsOffsetC | GM workspace 三段偏移（A'、B'、C'） |

**kernel 下发与同步**：

1. host 计算 tiling → 按 tiling 申请 GM workspace（`aclrtMalloc`/stream 级分配）。
2. 依次下发 `ctrmm_prepare_kernel`、`ctrmm_gemm_kernel`、`ctrmm_scale_kernel`（`<<<gridSize, nullptr, workspace, stream>>>`），stream 取自 handle（`aclblasSetStream`）。
3. 释放 workspace。函数返回即入队完成；读回 C 前由调用方同步 stream（异步语义与 cuBLAS 一致）。

#### 3.2.2 kernel 侧设计

每个 kernel 遵循 Ascend C 的 Init/Process 两阶段；Process 内按 CopyIn → Compute → CopyOut 流水。

**1. ctrmm_prepare_kernel（AIV，Vector）**

- **CopyIn**：`DataCopyPad` 搬入本核负责的 A 行/列块（含 lda padding 跳读）与对应 B 块；A 按三角裁剪（`uplo=UPPER` 时仅 k≥i，`LOWER` 时仅 k≤i），越界三角区填充 0；`diag=UNIT` 时对角位置写 (1,0) 而非读 A。
- **Compute**：交错拆实/虚部（`DataPack`/手动交织），按 §3.1.2/§3.1.3 的拼接规则写入 A'、B' 的 GM 目标位置：
  - LEFT：`A'[m][4k] = {Re(A[m][k]), -Im(A[m][k]), Re(A[m][k]), Im(A[m][k])}`，`B'[4k][2n] = {Re(B[k][n]), Re(B[k][n]), Im(B[k][n]), Im(B[k][n])}`（OP_C 时 A 下标互换且虚部取负）。
  - RIGHT：转置映射（§3.1.3），`op(A)ᵀ` 与 `Bᵀ` 等价拼接。
- **CopyOut**：拼接块写回 GM workspace（`DataCopy`）。
- 核间切分：按 A 的行（dimA）维均分，AIV 核数 = min(核数, dimA 分块下限)。

**2. ctrmm_gemm_kernel（AIC，Cube）**

- **CopyIn**：`LocalTensor` 双缓冲搬运 A'、B' 的 L1 tile（`DataCopy`，GM→A1/B1），L0 预取（L1→L0A/L0B）。
- **Compute**：`Mmad` 循环 K 维（tileKChunk 步进），累加 C'（M×2N，FP32，L0C 累加）。实/虚部在输出宽度方向天然拼接，一次 MMAD 流水同时产出 T1、T2。
- **CopyOut**：L0C→UB（`DataCopy`，FP32 直搬，无格式转换）→GM 临时区（按 `tempRowStride` 对齐）。
- **同步**：MTE1/MTE2/Vector 三引擎 flag 事件（`SetFlag/WaitFlag<HardEvent::...>`），L1/L0C 双缓冲轮转。
- 核间遍历：serpentine（N 方向奇偶反向），相邻核共享 L2 中的 B' 列块。

**3. ctrmm_scale_kernel（AIV，Vector）**

- **CopyIn**：搬入 C'（M×2N 实/虚部拼接）本核行块。
- **Compute**：复数缩放 `C = alpha * C'`：

```
C.real = alphaReal * T1 - alphaImag * T2
C.imag = alphaReal * T2 + alphaImag * T1
```

  `alpha=(0,0)` 分支直接输出 0（覆盖 quick return 下沉路径）。
- **CopyOut**：按 ldc 写回 C（`DataCopy`，支持 padding 列宽）。B≡C 原地安全（块内先读后写）。

**核心算法伪代码**（LEFT、OP_N、NON_UNIT 语义参考，C 参考实现）：

```cpp
for (int i = 0; i < m; ++i) {
  for (int j = 0; j < n; ++j) {
    complex acc{0.0f, 0.0f};
    int kBegin = (uplo == UPPER) ? i : 0;
    int kEnd   = (uplo == UPPER) ? m : i + 1;
    for (int k = kBegin; k < kEnd; ++k) {
      complex aik = (trans == OP_N) ? A[i + k*lda]
                  : (trans == OP_T) ? A[k + i*lda]
                                    : conjg(A[k + i*lda]);
      if (diag == UNIT && k == i) aik = {1.0f, 0.0f};
      complex bkj = B[k + j*ldb];
      acc.real += aik.real * bkj.real - aik.imag * bkj.imag;
      acc.imag += aik.real * bkj.imag + aik.imag * bkj.real;
    }
    C[i + j*ldc] = complex{alpha->real * acc.real - alpha->imag * acc.imag,
                           alpha->real * acc.imag + alpha->imag * acc.real};
  }
}
```

本实现以 §3.1.2 的拼接 GEMM 等价替换内层 k 循环（k 循环被 Cube 的 K 维流水吸收），数值语义与伪代码一致（FP32 累加，累加顺序不要求逐位一致，任务书明确确定性不要求）。

**边界处理**：

- `m=0`/`n=0`：host quick return，不进 kernel。
- `alpha=(0,0)`：A/B 不引用，C 置零（scale kernel alpha=0 分支）。
- `diag=UNIT`：prepare 阶段对角补 (1,0)，不读 A 对角（脏数据/越界值不影响结果）。
- 非对齐 m/n/ld：`DataCopyPad` 尾块 padding + 计算边界截断，GEMM tile 尾块用 base 粒度向上取整并以 0 填充。

#### 3.2.3 分支条件与合法组合

| 分支维度 | 取值 | kernel 内分支行为 |
| --- | --- | --- |
| side | LEFT / RIGHT | prepare 拼接映射（§3.1.2 vs §3.1.3 转置归约） |
| uplo | UPPER / LOWER | prepare 三角裁剪的 k 范围 |
| trans | N / T / C | prepare 的 A 下标映射；C 额外虚部取负 |
| diag | NON_UNIT / UNIT | prepare 对角读 A vs 补 (1,0) |

24 组全部合法、运行时分支；无 host 侧 kernel 选择。dtype 固定 COMPLEX64，无 dtype 分支。

#### 3.2.4 小 shape 策略

当 GEMM 总 base 块数（ceil(M/baseM) × ceil(2N/baseN)）小于可用 AIC 核数时：

1. 优先增大 tileM/tileN（至 L1 容量上限），减少发射次数；
2. 块数仍 < 核数时单核/少数核完成 GEMM，prepare/scale 相应减少 AIV 核数，避免 cube 发射与核间同步开销主导延迟（对标 case1/2 的小尺寸延迟场景，见 §4.2）。

### 3.3 内存与步长

- A、B：Device 只读；C：Device 输出；均列主序，元素 (i,j) 位于 `i + j*ld`。
- 前导维约束：`lda >= max(1, dimA)`、`ldb >= max(1, m)`、`ldc >= max(1, m)`，支持任意 padding。
- 非连续 Tensor 支持：不要求（步长仅由 lda/ldb/ldc 表达）。
- 允许 B≡C 原地写出（scale kernel 块内先读后写保证安全）；除此之外 C 与 A/B 不允许内存重叠。
- GM workspace：prepare 输出的 A'、B' 与 GEMM 输出的 C'，host 按 tiling 计算大小、kernel 生命周期内有效。
- 确定性：不要求（FP32 累加顺序不保证逐位一致）。

### 3.4 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas 800T A2 (910B3) | √ |
| Atlas 800I A3 | √ |
| Atlas A2 系列其他款型 | √（同 DAV_2201 架构） |
| Atlas A3 系列其他款型 | √（同 DAV_2201 架构） |

架构目录：`blas/trmm/arch22/`（arch22 = DAV_2201，ascend910b* 系列）；CANN 版本 9.1.0。

### 3.5 算子约束限制

1. `handle`、`alpha` 不可为空：`ACLBLAS_STATUS_HANDLE_IS_NULLPTR` / `ACLBLAS_STATUS_INVALID_VALUE`。
2. `side`/`uplo`/`trans`/`diag` 必须为合法枚举，否则 `ACLBLAS_STATUS_INVALID_VALUE`。
3. `m >= 0`、`n >= 0`；负值 `ACLBLAS_STATUS_INVALID_VALUE`；`m=0` 或 `n=0` 为合法 no-op。
4. `alpha=(0,0)` 时 A、B 不被引用，C 置零。
5. 前导维约束不满足返回 `ACLBLAS_STATUS_INVALID_VALUE`。
6. 矩阵引用非空场景下 A/B/C 为 nullptr 返回 `ACLBLAS_STATUS_INVALID_VALUE`。
7. 结果离席写入 C；允许 B≡C 原地，除此之外 C 与 A/B 不允许重叠；不返回视图。
8. 不支持超出 lda/ldb/ldc 语义的非连续 Tensor；无广播；无 dynamic shape 特化。
9. 异步执行依赖 `aclblasSetStream`；读回 Device 结果前须同步 stream。
10. 接口声明放入 `include/cann_ops_blas.h` 供其他产品线共用，不另设产品私有平行接口。

## 4 可维可测分析

### 4.1 精度标准

遵循生态算子开源精度标准（COMPLEX64 档；复数按实部/虚部分别套用 FLOAT32 容差），golden 由 cblas（Netlib BLAS `ctrmm`，`cblas_ctrmm`）生成，对输出矩阵 C（m×n）全矩阵验证：

| 数据类型 | rtol | atol | required_matched_ratio | max_abs_error_limit |
| --- | --- | --- | --- | --- |
| COMPLEX64 | 2^-13 (1.22e-4) | 2^-13 (1.22e-4) | 0.99 | 1e-2 或 32 * ULP |

- 逐元素通过条件：`|actual - golden| ≤ atol + rtol × |golden|`。
- 整体通过条件：`matched_ratio ≥ 0.99` 且 `max_abs_error ≤ max_abs_error_limit`。
- 说明：本算子含 k 维规约（多个元素求和叠加），大数规约场景可能引入更大绝对误差导致误报，`max_abs_error_limit` 可酌情放宽至 2 ULP（以任务书 §3.2.4 为准）。
- 特殊值：Inf/NaN 输入按任务书 §3.5 用例规则覆盖，行为对齐 cuBLAS。

### 4.2 性能标准

测试设备：Atlas 800T A2 (910B3)；COMPLEX64 输入场景平均单次耗时（Avg time，us），先 warmup 再有效采样 10 次取平均（msprof，op 自带 5 次 warmup）。5 个基准 case 达标耗时如下：

| case | m | n | side | uplo | trans | diag | 达标耗时（Avg time，us） |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 256 | 256 | LEFT | UPPER | N | NON_UNIT | 104.6 |
| 2 | 512 | 512 | LEFT | LOWER | C | NON_UNIT | 235.2 |
| 3 | 1024 | 1024 | LEFT | UPPER | T | UNIT | 803.8 |
| 4 | 2048 | 2048 | RIGHT | LOWER | N | NON_UNIT | 3241 |
| 5 | 4096 | 4096 | LEFT | UPPER | C | NON_UNIT | 19931 |

性能设计手段：三段 kernel 流水（AIV/Cube/AIV 重叠）、L1 双缓冲 + L0 预取、serpentine 遍历提升 L2 命中、小 shape 退化策略（§3.2.4）、三角裁剪降低 prepare 阶段数据量。

### 4.3 自验用例覆盖

使用 ops-blas 仓 test 目录测试框架（CSV 描述用例，python 脚本调用 C++ GTest 加载 CSV 执行），覆盖：

| 类别 | 覆盖内容 |
| --- | --- |
| 小 shape 基础 | 24 组 side×uplo×trans×diag 全枚举 × 小尺寸（8×8 量级） |
| shape 扫描 | 0、1、小质数、2 的幂及 2 的幂 ±1、非对齐值，直至大规模（含 fat/thin 非方阵） |
| 填充模式 | A 均匀/正态（各 50%，实虚部分别采样）、全零、Inf/NaN；B 均匀/正态、全零、Inf/NaN |
| 标量 | alpha = 0、1、-1、1+i、纯虚、大值；alpha=(0,0) 置零路径 |
| 前导维 | 紧凑（ld = 最小约束）与多档 padding（+4/+8 等） |
| 边界与负向 | 零维 no-op、空指针（handle/alpha/A/B/C）、非法枚举（side/uplo/trans/diag）、非法前导维（lda/ldb/ldc）、负维度（m/n），期望状态码对齐 cuBLAS |
| 性能 | 5 个任务书基准 case + 扩展 shape 扫描 |

测试工程位置：`test/trmm/ctrmm/arch22/`（含 CSV 用例文件），列格式对齐仓内 `strmm` 测试惯例（复数标量拆 alpha_real/alpha_imag 列，大小写不敏感解析枚举简写/全名）。

### 4.4 兼容性分析

本算子为新增接口（`aclblasCtrmm`），不修改任何既有接口行为，不涉及存量代码兼容性改造；实现位于 `blas/trmm/arch22/`，与既有 arch 目录隔离。接口声明加入 `include/cann_ops_blas.h`，供其他产品线共用。

## 5 交付与合入

| 交付件 | 说明 |
| --- | --- |
| 算子设计文档 | 本文档（`docs/design.md`），经 PR 评审合入 |
| 自测用例及测试代码 | `test/trmm/ctrmm/arch22/`（CSV + GTest），readme 说明测试步骤 |
| 自测报告 | 精度/性能/内存数据（模板见任务书 §4） |
| 待验收代码 | 个人仓（邀请 Ascend-CANN 为开发者），算子 readme 标注 Atlas A2/A3 支持 |

验收通过后提交 PR 合入 ops-blas：实现代码 `blas/trmm/arch22/`，测试代码 `test/trmm/ctrmm/arch22/`，遵循仓贡献指南，无冗余文件。
