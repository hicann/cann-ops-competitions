# aclblasCtrmm 算子设计文档

## 1 需求背景（required）

### 1.1 需求来源
华为昇腾 CANN 社区任务——算子实操工坊。本任务要求在 Atlas A2/A3 系列产品上，使用 Ascend C/CATLASS 编程语言开发单精度复数（complex64）三角矩阵乘算子 `aclblasCtrmm`，完成算子设计、开发、测试全流程工作。验收通过后合入昇腾算子开源仓 ops-blas。

### 1.2 背景介绍

#### 1.2.1 aclblasCtrmm 算子实现优化
基于 BLAS 库历史 CUDA/cuBLAS 版本（`cublasCtrmm`）及 Netlib BLAS `ctrmm` 参考实现，使用 Ascend C/CATLASS 在昇腾 NPU 上进行算子原生开发与性能优化。该算子执行单精度复数三角矩阵乘 `C = alpha * op(A) * B`（或 `C = alpha * B * op(A)`），是 BLAS Level 3 基础算子，在矩阵分解、线性方程组求解、协方差计算等场景中广泛使用。

#### 1.2.2 aclblasCtrmm 算子参考实现现状分析
当前 `ops-blas` 开源仓中仅有同族实数算子 `aclblasStrmm`（位于 `blas/trmm/`），尚无复数版本实现。实数算子仅涉及实数乘加，而复数算子需要处理复数乘法（含实虚部交叉项）以及 `trans=ACLBLAS_OP_C` 的复共轭转置语义。本算子需参照 `strmm` 的工程框架，构建句柄式 BLAS 接口，通过 handle 绑定 stream 直调 NPU kernel，实现代码放在 `blas/trmm/arch22/`。

#### 1.2.3 aclblasCtrmm 算子功能分析
**输入**：`side`、`uplo`、`trans`、`diag`、`m`、`n`、复数标量 `alpha`、复数三角矩阵 `A`、前导维 `lda`、一般矩形矩阵 `B`、前导维 `ldb`。
**输出**：复数一般矩阵 `C`（离席写入），前导维 `ldc`。
**支持数据类型**：COMPLEX64（实部/虚部均为 float32）。
**对标基线接口**：cuBLAS `cublasCtrmm`。

## 2 需求分析（required）

### 2.1 需求描述
使用 Ascend C 编程语言实现 `aclblasCtrmm`，提供句柄式 BLAS 接口，通过 handle 绑定 stream 直调 NPU kernel，完成单精度复数三角矩阵乘计算。支持 `side=LEFT/RIGHT`、`uplo=UPPER/LOWER`、`trans=N/T/C`、`diag=NON_UNIT/UNIT` 共 24 组枚举组合，满足精度、性能与边界语义要求。

### 2.2 需求拆解
1. **功能实现**：`side=ACLBLAS_SIDE_LEFT` 时 `C = alpha * op(A) * B`；`side=ACLBLAS_SIDE_RIGHT` 时 `C = alpha * B * op(A)`。`op(A)` 由 `trans` 决定：`OP_N` — `op(A)=A`；`OP_T` — `op(A)=Aᵀ`；`OP_C` — `op(A)=Aᴴ`（复共轭转置）。
2. **三角裁剪**：仅引用 `uplo` 指定的三角区域；`diag=UNIT` 时主对角视为 1 且不读取，`diag=NON_UNIT` 时读取对角元素（乘法语义下允许为零）。
3. **维度关系**：列主序（Column-Major）。`side=LEFT` 时 A 为 m×m，`side=RIGHT` 时 A 为 n×n；B 和 C 为一般矩形矩阵 m×n，结果离席写入 C。
4. **边界与 quick return**：`m=0` 或 `n=0` 为合法 no-op；`alpha=0` 时 A、B 不被引用，C 全部置零。
5. **接口与约束**：在 `include/cann_ops_blas.h` 新增声明，签名与 cuBLAS `cublasCtrmm` 逐参数对齐；校验 handle、alpha、A、B、C 空指针，校验各 lda/ldb/ldc 前导维度约束。
6. **性能达标**：Atlas 800T A2 (910B3) 上 COMPLEX64 输入场景，平均单次 kernel 耗时满足任务书 §3.3 标杆。

## 3 详细设计（required）

### 3.1 算子分析

#### 3.1.1 数学公式
三角矩阵乘定义为：

$$
C = \alpha \cdot op(A) \cdot B \quad (\text{side} = \text{LEFT})
$$

$$
C = \alpha \cdot B \cdot op(A) \quad (\text{side} = \text{RIGHT})
$$

其中 op(A) 由 trans 决定：

$$
op(A) = \begin{cases}
A & \text{trans} = OP\_N \\
A^T & \text{trans} = OP\_T \\
A^H & \text{trans} = OP\_C \ (\text{复共轭转置})
\end{cases}
$$

**维度关系**（列主序）：
- `side=LEFT`：A 为 m×m（lda ≥ max(1,m)）；`side=RIGHT`：A 为 n×n（lda ≥ max(1,n)）
- B、C 为 m×n；ldb ≥ max(1,m)，ldc ≥ max(1,m)

**三角裁剪**：
- `uplo=UPPER`：仅访问 j ≥ i 区域；`uplo=LOWER`：仅访问 j ≤ i 区域
- `diag=NON_UNIT`：对角元素从 A 读取，允许为零
- `diag=UNIT`：对角元素视为 1，不读取 A 对角位置

**复数乘法语义**：复数乘加分解为实虚部运算，`(x+iy)(u+iv) = (xu - yv) + i(xv + yu)`；`OP_C` 的共轭转置为 `conjg(A^T)`，即 `op(A)[i][j] = conjg(A[j][i])`。

**边界语义**：`m=0` 或 `n=0` 为合法 no-op；`alpha=(0,0)` 时 A/B 不被引用，C 全部置零。

#### 3.1.2 支持数据类型
- 输入/输出矩阵 A、B、C：`aclblasComplex`（COMPLEX64，实部/虚部均为 float32）
- 标量 alpha：`aclblasComplex`

#### 3.1.3 支持形状
- A：`side=LEFT` 时 m×m；`side=RIGHT` 时 n×n
- B、C：m×n
- 无广播，无动态 shape 特化，m/n 为运行时入参由 host tiling 支持

### 3.2 算子实现

#### 3.2.1 Host 侧设计

**接口设计**：在 `include/cann_ops_blas.h` 中新增如下声明：

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

**参数校验策略**：
1. 校验 `handle` 是否为空，为空返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`。
2. 校验 `side`、`uplo`、`trans`、`diag` 枚举合法，非法返回 `ACLBLAS_STATUS_INVALID_VALUE`。
3. 校验 `m >= 0`、`n >= 0`，负值返回 `ACLBLAS_STATUS_INVALID_VALUE`。
4. 校验 `alpha` 非空，为空返回 `ACLBLAS_STATUS_INVALID_VALUE`。
5. `m=0` 或 `n=0` 或 `alpha=(0,0)` 时快速返回（no-op 或 C 置零）。
6. 校验前导维度：`side=LEFT` 时 `lda >= max(1,m)`，`side=RIGHT` 时 `lda >= max(1,n)`；`ldb >= max(1,m)`；`ldc >= max(1,m)`。
7. 校验 A、B、C 指针非空，为空返回 `ACLBLAS_STATUS_INVALID_VALUE`。

**Tiling 策略**：
- 按 (side, uplo, trans, diag) 组合特化 tiling 分支，共 24 种组合。
- **LEFT 路径**：对 C 的行分块（按 m 维切分），每个核处理若干行的 `C[i,:] = alpha * Σ_{k∈三角区} op(A)[i,k] * B[k,:]`。三角裁剪决定 k 的遍历范围。
- **RIGHT 路径**：对 C 的列分块（按 n 维切分），每个核处理若干列的 `C[:,j] = alpha * Σ_{k∈三角区} B[:,k] * op(A)[k,j]`。
- 大规模情形下进一步在 n（或 m）维上切分，多核并行。
- Tiling 参数包括：`m`、`n`、`lda`、`ldb`、`ldc`、`side`、`uplo`、`trans`、`diag`、`blockDim` 等。
- 不申请 workspace（与实数 strmm 对齐）。

**Kernel 下发与同步**：Device 路径通过 `<<<blockDim, nullptr, stream>>>` 下发 kernel。Host 侧返回后读回结果前须同步 `handle` 绑定的 stream。

#### 3.2.2 Kernel 侧设计

**整体流程**：采用多核并行，每个核处理 C 的一部分行（LEFT）或列（RIGHT）区间。

1. **CopyIn 阶段**：
   - 搬入 A 的三角区域对应片段（含 padding 跳读）。
   - 搬入 B 对应片段。
   - `diag=UNIT` 时跳过 A 主对角位置的读取。

2. **Compute 阶段**：
   - **LEFT 路径**：对每个输出行 i，按列序遍历 k ∈ 三角区域（`uplo=UPPER` 时 k ≥ i，`uplo=LOWER` 时 k ≤ i；`trans=T/C` 时 k 范围相应转置），累加 `op(A)[i,k] * B[k,j]`。
   - **RIGHT 路径**：对每个输出列 j，按列序遍历 k ∈ 三角区域，累加 `B[i,k] * op(A)[k,j]`。
   - **复数乘加**：`(ar + i·ai)(br + i·bi) = (ar·br - ai·bi) + i(ar·bi + ai·br)`。
   - **OP_C 共轭**：对 op(A) 的虚部取负。
   - **OP_T/OP_C 转置**：访问 A 时下标互换。
   - **对角 unit**：跳过 A 对角读取，直接使用 1。
   - **alpha 乘**：结果为 `alpha * acc`。
   - **写回 C**：离席写入（C 与 A/B 不允许内存重叠）。

3. **CopyOut 阶段**：
   - 将 C 的对应片段写回。

**核心算法伪代码**（以 LEFT, NON_UNIT 为例）：

```cpp
for (int i = row_start; i < row_end; ++i) {
    for (int j = 0; j < n; ++j) {
        complex acc = {0.0f, 0.0f};
        // 三角裁剪
        int k_start = (uplo == UPPER) ? i : 0;
        int k_end   = (uplo == UPPER) ? m : i + 1;
        for (int k = k_start; k < k_end; ++k) {
            complex aik;
            if (trans == OP_N) aik = A[i + k * lda];
            else if (trans == OP_T) aik = A[k + i * lda];
            else aik = conjg(A[k + i * lda]);   // OP_C
            if (diag == UNIT && k == i) aik = {1.0f, 0.0f};
            complex bkj = B[k + j * ldb];
            // 复数乘加
            acc.real += aik.real * bkj.real - aik.imag * bkj.imag;
            acc.imag += aik.real * bkj.imag + aik.imag * bkj.real;
        }
        // alpha 乘
        complex cij;
        cij.real = alpha->real * acc.real - alpha->imag * acc.imag;
        cij.imag = alpha->real * acc.imag + alpha->imag * acc.real;
        C[i + j * ldc] = cij;
    }
}
```

**边界处理**：
- `m=0` 或 `n=0`：Host 侧 quick return，不进入 Kernel。
- `alpha=(0,0)`：Host 侧 quick return，将 C 全部置零。
- `diag=UNIT`：Kernel 内跳过 A 主对角位置的读取，直接使用 (1, 0)。

#### 3.2.3 内存与步长
- A、B 为 Device 只读内存；C 为 Device 输出内存。
- 列主序：元素 `(i,j)` 位于 `i + j * ld`。
- 各前导维度满足约束：`lda >= max(1, side 对应阶数)`，`ldb >= max(1,m)`，`ldc >= max(1,m)`。
- 允许 B≡C 原地写出（cuBLAS trmm 的等价原地用法），除此之外 C 与 A/B 不允许内存重叠。
- 不要求非连续 Tensor 支持，步长由 lda/ldb/ldc 表达。

### 3.3 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas 800T A2 (910B3) | √ |
| Atlas 800I A3 | √ |

### 3.4 算子约束限制
1. `handle`、`alpha` 不可为空，否则返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` / `ACLBLAS_STATUS_INVALID_VALUE`。
2. `side`、`uplo`、`trans`、`diag` 必须为合法枚举，非法返回 `ACLBLAS_STATUS_INVALID_VALUE`。
3. `m >= 0`、`n >= 0`；负值返回 `ACLBLAS_STATUS_INVALID_VALUE`。
4. `m=0` 或 `n=0` 为合法 no-op；`alpha=(0,0)` 时 A/B 不被引用，C 置零。
5. 前导维度：`lda >= max(1, side 对应阶数)`，`ldb >= max(1,m)`，`ldc >= max(1,m)`；不满足返回 `ACLBLAS_STATUS_INVALID_VALUE`。
6. A/B/C 引用非空下不得为 nullptr；结果离席写入 C，允许 B≡C 原地。
7. 异步执行依赖 `aclblasSetStream` 绑定 stream；读回 Device 结果前须同步 stream。
8. 接口声明放入 `include/cann_ops_blas.h`，禁止定义产品私有平行接口。

## 4 可维可测分析

### 4.1 精度标准/性能标准

#### 4.1.1 精度验收标准
遵循生态算子开源精度标准（COMPLEX64 档，实部/虚部分别按 FLOAT32 判定）：

| 数据类型 | rtol | atol | required_matched_ratio | max_abs_error_limit |
| --- | --- | --- | --- | --- |
| COMPLEX64 | 2^-13 (1.22e-4) | 2^-13 (1.22e-4) | 0.99 | 1e-2 或 32 * ULP |

逐元素通过条件：`|actual - golden| ≤ atol + rtol × |golden|`。用例同时满足 matched_ratio ≥ 0.99 且 max_abs_error ≤ max_abs_error_limit 时判定通过。含大数规约的场景，max_abs_error_limit 可酌情放宽至 2 ULP。golden 由 Netlib BLAS `ctrmm`（`cblas_ctrmm`）生成，对输出矩阵 C（m×n）全矩阵验证。

#### 4.1.2 性能验收标准
测试设备：Atlas 800T A2 (910B3)。性能数据为 COMPLEX64 输入场景下的平均单次 kernel 耗时，须先 warmup 再有效采样 >10 次取平均。

| case | m | n | side | uplo | trans | diag | 达标耗时（us） |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 256 | 256 | LEFT | UPPER | N | NON_UNIT | 104.6 |
| 2 | 512 | 512 | LEFT | LOWER | C | NON_UNIT | 235.2 |
| 3 | 1024 | 1024 | LEFT | UPPER | T | UNIT | 803.8 |
| 4 | 2048 | 2048 | RIGHT | LOWER | N | NON_UNIT | 3241 |
| 5 | 4096 | 4096 | LEFT | UPPER | C | NON_UNIT | 19931 |

### 4.2 自验用例覆盖
基于 `trmm_test.csv` 的用例集（1200 条：精度 1000 + 性能 200），覆盖以下类别：

| 类别 | 前缀 | 条数 | 说明 |
| --- | --- | --- | --- |
| L0 基础 | TC_L0 | 24 | side×uplo×trans×diag 24 组枚举全组合 × 小尺寸 8×8 |
| L1 尺寸 | TC_SQ | 23 | 23 种尺寸（1→2048，含奇数/边界/2 的幂 ±1/非对齐）× 枚举组合轮转 |
| L2 标量 | TC_AB | 24 | alpha 特殊值（含 (0,0) 置零路径、(-1,0)、(1,1) 即 1+i、纯虚、大值）× 枚举组合 |
| L3 非方阵 | TC_WS/TH | 12+12 | fat（m≪n，side=LEFT）+ thin（m≫n，side=RIGHT） |
| L4 前导维 | TC_LD | 12 | lda/ldb/ldc padding（= 最小约束 + 4）× 尺寸 (16,32,64) |
| L5 填充 | TC_FL | 13 | A：均匀/交替/极端/全零（NON_UNIT 全零合法）/UNIT 全零；B：均匀/全零/交替/极端/Inf/NaN |
| L5b 覆盖 | TC_CV | 24 | 中等尺寸 × 枚举组合抽样覆盖 |
| L6 边界 | TC_ED | 23 | 零维 quick return / alpha=(0,0) 路径（含 A/B 空指针合法场景）/ 空指针 / 非法枚举 / 非法前导维 / 负维度 |
| EX 扩展 | TC_EX | 833 | 尺寸池 × 枚举组合 × 标量 × padding 确定性采样 |
| PF 性能 | TC_PF | 200 | 5 条任务书典型 case + 小尺寸延迟区 + 规模对数扫描（≤4096）+ 宽/窄网格 + 枚举×尺寸网格 + 预算内混合（全部纯连续访存） |

**负向覆盖**：`null_handle`（期望 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`）、`null_alpha`/`null_A`/`null_B`/`null_C`、非法枚举 `invalid_side/uplo/trans/diag`、非法前导维 `invalid_lda_left/right`/`invalid_ldb`/`invalid_ldc`、负维度 `neg_m`/`neg_n`。

**枚举全覆盖**：L0 覆盖 24 组 side×uplo×trans×diag 组合；trans 的 N/T/C 三个值全覆盖。

### 4.3 兼容性分析
本算子为新增算子，不涉及存量代码兼容性改造。接口签名严格对齐 `cublasCtrmm`，在 `include/cann_ops_blas.h` 中声明供其他产品线共用。测试工程参照 `test/trmm/strmm/` 模式新建，CSV 列格式对齐 `strmm_param.h`（复数标量按仓内复数工程惯例拆 `alpha_real`/`alpha_imag`），golden 由 Netlib BLAS `ctrmm`（`cblas_ctrmm`）生成。在 `blas/trmm/README.md` 产品支持表中新增 `aclblasCtrmm` 并标注 Atlas A2/A3 系列产品支持。实现代码合入 `blas/trmm/arch22/`，测试代码合入 `test/trmm/ctrmm/arch22/`。