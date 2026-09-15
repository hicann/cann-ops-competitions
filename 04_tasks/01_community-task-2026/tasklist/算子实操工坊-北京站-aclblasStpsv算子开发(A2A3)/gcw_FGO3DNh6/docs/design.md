# aclblasStpsv 算子设计文档（Atlas A2/A3）

| 文档版本 | 日期 | 作者/团队 | 说明 |
| --- | --- | --- | --- |
| V1.0 | 2026-09-14 | gcw_FGO3DNh6 | 按社区任务 CheckList 完整整理 A2/A3 评审稿 |

> 目标代码目录为 `ops-blas/blas/tpsv/arch22/`，测试目录为 `ops-blas/test/tpsv/stpsv/arch22/`。
> 公开接口由所有产品线共用，本文只设计 Atlas A2/A3（DAV_2201 / arch22）路径；`blas/tpsv/arch35/`（Ascend 950PR）已有实现不受影响。

# 需求背景（required）

## 需求来源

本需求来自 CANN 社区任务「算子实操工坊-北京站-aclblasStpsv算子开发(A2A3)」。

任务要求在昇腾 NPU（Atlas 800I A2 / Atlas 800I A3 系列产品，DAV_2201，CANN 9.1.0）上使用 Ascend C 开发**单精度实数三角方程组求解（打包存储）**算子 `aclblasStpsv`，功能与参数语义对标 cuBLAS `cublasStpsv` 与 Netlib `stpsv`；算子实现最终合入昇腾算子开源仓 ops-blas（https://gitcode.com/cann/ops-blas ）的 `blas/tpsv/arch22/` 目录，测试代码合入 `test/tpsv/stpsv/arch22/`。

## 背景介绍

### 算子定位与数学定义

STPSV 属于 BLAS Level-2，用于求解单右端三角线性系统（Packed Storage 版本）：

```
op(A) * x = b
```

其中 A 为 n×n 三角实数矩阵，**以打包存储（Packed Storage）格式**逐列堆叠 `uplo` 指定三角（含对角）于一维数组 AP（长度 n(n+1)/2，**无前导维 lda**）；x 为长度 n 的单精度实数向量，**入口存放右端项 b，出口原地覆写为解向量 x**；`op(A)` 由 `trans` 决定。

STPSV 是 TRSV（全存储，`blas/trsv/`）的压缩存储同族算子，二者数学语义完全相同，差别仅在于矩阵寻址：TRSV 用 `A[i + j*lda]`，STPSV 用打包下标（无 lda，AP 长度仅 n(n+1)/2）。相比于调用方先显式求逆再乘向量，三角回代在计算量与数值误差上都更优，是 Cholesky/LU 相关计算、协方差求解、隐式时间推进等场景的基础能力。

与 TRMV/TPSV 类"无依赖"算子不同，**STPSV 的每个新解分量都依赖前面已经求出的分量**，存在严格的前代（forward substitution）/回代（backward substitution）顺序，不能把所有输出元素无依赖地并行计算；同时 x 是原地输入输出，进一步限制了并行组织方式——这是本算子的核心难点。

本次任务的实现路径：

- 目标硬件：Atlas A2/A3 系列产品（DAV_2201，arch22），CANN 9.1.0
- 编程语言：Ascend C（经典 arch22 编程模型：`TPipe`/`TQue`/`GlobalTensor`，**AIV only**）
- 工程模式：**Ascend C kernel 直调**。Host 侧实现句柄式 BLAS 接口（`aclblasHandle_t` 携带 stream），通过 `<<<numBlocks, nullptr, stream>>>` 直接下发 NPU kernel，不经过 aclnn 图/原型注册链路
- 实现目录：`blas/tpsv/arch22/`（Host + Kernel + Tiling）
- 接口声明位置：`include/cann_ops_blas.h:160`（**已有声明，本次不新增、不修改签名**）

### packed 存储布局与一项必要的勘误

Netlib `stpsv.f` 与 cuBLAS 的实际内存布局如下（以下均为 **0-based** 下标，沿用仓内 `blas/tpsv/arch35/stpsv_kernel_utils.h` 的记法）：

| uplo | 元素 A(i,j) 在 AP 中的下标 | 第 j 列列首 colstart(j) | 第 j 列长度 |
| --- | --- | --- | --- |
| UPPER（i ≤ j） | `i + j*(j+1)/2` | `j*(j+1)/2` | `j+1` |
| LOWER（i ≥ j） | `i + j*(2n-j-1)/2` | `j*(2n-j+1)/2` | `n-j` |

**勘误说明**：任务书 §2.1 给出的 LOWER 公式为 `AP[i + ((2*n-j-1)*j)/2 + i]`（i ≥ j），其中末尾多出一个 `+ i`，与 Netlib/cuBLAS 实际布局不一致。以 n = 3 验证：

- `A(1,0)`：任务书公式给出 `1 + 0 + 1 = 2`，而 Netlib 布局为 `AP[1]`（LOWER 第 0 列为 `a(1,1) a(2,1) a(3,1)` → 下标 0,1,2）；
- `A(2,2)`（对角）：任务书公式给出 `2 + 3 + 2 = 7`，但 LOWER 的 AP 只有 n(n+1)/2 = 6 个元素（合法下标 0..5），**直接越界**；正确公式给出 `2 + 3 = 5`。

因此对标口径明确为：**以 Netlib `stpsv` 参考实现（golden 由 cblas 生成）的实际内存布局为准**，即

```
UPPER: A(i,j) -> AP[i + j*(j+1)/2]      (i <= j)
LOWER: A(i,j) -> AP[i + j*(2n-j-1)/2]   (i >= j)
```

该式与以下三处完全一致，可直接互证：

1. Netlib `stpsv.f` 中 `KK` 指针的推进（UPPER：`KK = KK - J`，对角在 `AP(KK)`；LOWER：`KK = KK + (N-J+1)`，对角在 `AP(KK)`）；
2. 本仓 Ascend 950PR 已合入实现 `blas/tpsv/arch35/stpsv_kernel_utils.h` 中的 `TpsvPackedUpperIdx / TpsvPackedLowerIdx`；
3. 本仓测试 golden 的 CPU 侧 `test/tpsv/stpsv/stpsv_golden.h` 中的 `TpsvPackedUpperIdxCpu / TpsvPackedLowerIdxCpu`。

> 该处已在「风险与待确认项 #1」登记，建议同步在任务讨论帖反馈，避免其他开发者按任务书公式实现导致 golden 比对全量失败。任务书 §2.1 给出的 **UPPER** 公式 `AP[i + j*(j+1)/2]` 是正确的，无需修改。

### 对标接口与现状分析

| 接口 / 实现 | 现状 | 与本设计的关系 |
| --- | --- | --- |
| `cublasStpsv`（cuBLAS） | 语义基线 | 参数序列 `handle, uplo, trans, diag, n, AP, x, incx`，无 lda、无 alpha/beta；支持 N/T/C、UPPER/LOWER、UNIT/NON_UNIT |
| `stpsv`（Netlib） | 参考实现 + golden | 与 cuBLAS 语义一致；含 `X(J).NE.ZERO` 提前跳过优化与 `INFO` 参数检查（见「风险与待确认项 #4」） |
| `aclblasStpsv`（`include/cann_ops_blas.h:160`） | **已有声明，本次直接实现** | `int n`、`const float* AP`、`float* x`、`int incx`，逐参数对齐 cuBLAS |
| `blas/tpsv/arch35/stpsv_*`（Ascend 950PR） | 仓内已有实现 | 同一算子的 950PR 版本，采用 **SIMT VF**（`asc_vf_call`）并行内积；arch22 无 SIMT 编程模型，**不能直接复用**，但 tiling 字段、参数校验顺序、packed 索引工具、测试 golden 全部可复用 |
| `blas/trsv/arch22/strsv_*`（A2/A3） | 仓内已有实现 | 全存储实数三角求解，是 STPSV 的数学同族；其 arch22 工程结构（Host 校验/launch、`strsv_kernel_do` 分发、`KERNEL_TYPE_AIV_ONLY`）直接可参考；但其 **单核标量逐元素 `GetValue/SetValue`** 实现方式性能不足以达标（见 §性能标杆与达标分析），本设计需要向量化重写 |
| `blas/tpmv/arch22/stpmv_*`（A2/A3） | 仓内已有实现 | packed 存储三角矩阵-向量乘，arch22 上的 **packed 索引 / incx 展开 / 分核 / DataCopyPad / ReduceSum** 工程范式参考 |
| `blas/tpsv/README.md` | 产品支持表当前标注 A2/A3「不支持」 | 本次需更新为「支持」，并补充 arch22 目录说明 |
| `test/tpsv/stpsv/` | 仅有 `arch35/`，`stpsv_param.h`、`stpsv_golden.h`、`CMakeLists.txt` 与架构无关 | `arch22/` 目录需新建；`param`/`golden` 直接复用，但需扩展 CSV 列解析（见 §测试设计） |

### aclblasStpsv 算子功能分析

- 计算 `op(A) * x = b`，`op(A) ∈ {A, Aᵀ, Aᴴ}`；**实数域下 `ACLBLAS_OP_C` 与 `ACLBLAS_OP_T` 语义等价**（实矩阵共轭为其自身），实现上二者合并为同一分支；
- A 以 packed 格式存储，**只引用 `uplo` 指定的三角**，另一三角不存于 AP、由三角性隐含；
- `diag = ACLBLAS_UNIT` 时主对角视为 1，**不读取 AP 中对应对角位置**（`diag = ACLBLAS_NON_UNIT` 时对角元从 AP 读取并参与除法）；
- 求解方向由 `trans` 与 `uplo` 联合决定（见 §数学与依赖方向）；
- `n = 0` 为合法 no-op（quick return，返回 `ACLBLAS_STATUS_SUCCESS`，不引用 AP、不更新 x）；
- `incx` 支持任意非零值（含负步长，负步长时按 Netlib 语义反向遍历逻辑下标）；
- 本算子不做奇异性或近奇异性检测（与 cuBLAS/Netlib 一致），调用方须保证 `diag = NON_UNIT` 时对角元非零；
- x 原地覆写（入口 b，出口解 x），不返回视图。

# 需求分析（required）

## 需求描述

使用 Ascend C 在 Atlas A2/A3（arch22）上实现 `aclblasStpsv`，接口签名与仓内 `include/cann_ops_blas.h:160` 的既有声明完全一致：

```cpp
aclblasStatus_t aclblasStpsv(
    aclblasHandle_t handle,
    aclblasFillMode_t uplo,
    aclblasOperation_t trans,
    aclblasDiagType_t diag,
    int n,
    const float* AP,
    float* x,
    int incx);
```

实现置于 `blas/tpsv/arch22/`，测试工程置于 `test/tpsv/stpsv/arch22/`。**不新增接口、不定义产品私有平行接口**。

## 需求拆解

1. **功能**：12 组枚举组合（uplo × trans × diag = 2×3×2）全部正确；`ACLBLAS_OP_C` 与 `ACLBLAS_OP_T` 实数等价；支持 `incx` 正/负步长；支持 `n = 0` quick return；支持 `diag = UNIT` 对角不读。
2. **接口与异常**：复用既有声明；异常返回码与仓内 arch35 同族实现、以及任务配套 CSV 用例口径保持一致（`INVALID_VALUE` / `HANDLE_IS_NULLPTR`）。
3. **精度**：FLOAT32 档，`atol = 2⁻¹⁶`、`rtol = 2⁻¹⁰`、`matched_ratio ≥ 0.99`、`max_abs_error ≤ 1e-2`（或 32×ULP）；golden 由 cblas（Netlib `stpsv`）生成，对输出向量 x 全量逐元素验证。
4. **性能**：4 条典型 case（n = 512/1024/2048/4096）平均单次耗时 ≤ 任务书 §3.3 标杆（432.71 / 941.23 / 1690.94 / 5185.2 us）。
5. **工程**：Host + Kernel + Tiling 三件套，kernel 直调；`blas/tpsv/README.md` 产品支持表标注 Atlas A2/A3：支持；测试工程含 CSV 用例与可复现的 readme。
6. **自验**：覆盖 L0 基础、尺寸扫描、步长、填充（含 Inf/NaN）、边界负向、性能用例，输出自测报告。

# 详细设计（required）

## 算子分析

### 数学公式

```
op(A) · x = b

        ⎧ A      trans = ACLBLAS_OP_N
op(A) = ⎨ Aᵀ     trans = ACLBLAS_OP_T
        ⎩ Aᴴ     trans = ACLBLAS_OP_C   （实数下 Aᴴ ≡ Aᵀ）

      ⎧ (b_i − Σ_{j<i} Aeff(i,j)·x_j) / Aeff(i,i)     Aeff 有效下三角（前代，i 递增）
x_i = ⎨
      ⎩ (b_i − Σ_{j>i} Aeff(i,j)·x_j) / Aeff(i,i)     Aeff 有效上三角（回代，i 递减）
```

其中 `Aeff = op(A)`，`diag = UNIT` 时上式不读取也不除以 `Aeff(i,i)`（等价于替换为 `+ 1·x_i`）。

### 数学与依赖方向

| uplo | trans | `Aeff = op(A)` 有效三角 | 求解方向 | 扫描索引 |
| --- | --- | --- | --- | --- |
| LOWER | N | LOWER | 前代 | k = 0 → n−1 递增 |
| UPPER | N | UPPER | 回代 | k = n−1 → 0 递减 |
| LOWER | T / C | UPPER | 回代 | k = n−1 → 0 递减 |
| UPPER | T / C | LOWER | 前代 | k = 0 → n−1 递增 |

即：`uploEff = (trans == N) ? uplo : opposite(uplo)`；`uploEff = LOWER` 时按 k 递增前代，`uploEff = UPPER` 时按 k 递减回代。

```mermaid
flowchart TD
    A[uplo / trans] --> B{trans == N?}
    B -- 是 --> C[uploEff = uplo]
    B -- 否 --> D[uploEff = opposite(uplo)]
    C --> E{uploEff}
    D --> E
    E -- LOWER --> F[前代 k = 0 -> n-1 递增]
    E -- UPPER --> G[回代 k = n-1 -> 0 递减]
    F --> H[原地覆写 x_k]
    G --> H
```

### 支持数据类型

| 张量/标量 | 类型 | 说明 |
| --- | --- | --- |
| AP | `const float*`（FP32） | Device 内存，只读，长度 n(n+1)/2 |
| x | `float*`（FP32） | Device 内存，输入/输出，物理长度 1+(n−1)·\|incx\| |
| n / incx | `int` | Host 标量 |
| uplo / trans / diag | 枚举（int） | Host 标量 |

### 支持形状

- AP：长度 `n(n+1)/2` 的一维数组（**无 lda**），按列主序堆叠 `uplo` 指定三角各列（含对角）
- x：长度 `1 + (n−1)·|incx|` 的一维数组；`incx < 0` 时逻辑下标 i 对应物理下标 `(n−1−i)·|incx|`
- 不涉及广播；n 为运行时入参（dynamic shape 编译期特化不作要求）；packed 族无前导维维度，故无 L3 非方阵与 L4 lda padding 类别

### 参数与异常行为矩阵

| 参数 | 合法值域 | 非法时返回 |
| --- | --- | --- |
| handle | 非 nullptr | `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| uplo | `{ACLBLAS_UPPER(121), ACLBLAS_LOWER(122)}` | `ACLBLAS_STATUS_INVALID_VALUE` |
| trans | `{ACLBLAS_OP_N(111), ACLBLAS_OP_T(112), ACLBLAS_OP_C(113)}` | `ACLBLAS_STATUS_INVALID_VALUE` |
| diag | `{ACLBLAS_NON_UNIT(131), ACLBLAS_UNIT(132)}` | `ACLBLAS_STATUS_INVALID_VALUE` |
| n | n ≥ 0（n = 0 为合法 no-op，直接返回 `SUCCESS`） | n < 0 → `ACLBLAS_STATUS_INVALID_VALUE` |
| AP | n > 0 时非 nullptr | `ACLBLAS_STATUS_INVALID_VALUE` |
| x | n > 0 时非 nullptr | `ACLBLAS_STATUS_INVALID_VALUE` |
| incx | incx ≠ 0 | `ACLBLAS_STATUS_INVALID_VALUE` |

**校验顺序**（与 `blas/tpsv/arch35/stpsv_host.cpp` 的 `aclblasStpsv` 完全一致，保证跨产品线行为一致）：
`handle` → `n >= 0` → `n == 0` quick return → 枚举（uplo/trans/diag） → `incx != 0` → `AP != nullptr` → `x != nullptr`。异常分支全部在 Host 侧返回，不下发 kernel。

> **注意**：任务书 §2.4 文字描述非法枚举返回 `ACLBLAS_STATUS_INVALID_ENUM`，但仓内 arch35 已合入实现与任务配套 CSV 用例（`TC_INV_001~003`）均为 `ACLBLAS_STATUS_INVALID_VALUE`。本设计**与仓内既有实现及配套用例保持一致**，使用 `INVALID_VALUE`；该口径差异已在「风险与待确认项 #2」登记。

## 算子实现

### 工程结构与文件清单

```
blas/tpsv/README.md                       # 更新产品支持表：Atlas A2/A3 → 支持；补充 arch22 路径说明
blas/tpsv/arch22/stpsv_host.cpp           # Host：校验、归一化、分块 tiling、workspace、launch 调度
blas/tpsv/arch22/stpsv_kernel.cpp         # Kernel：对角块求解 + 尾块更新 + 分发表（stpsv_kernel_do）
blas/tpsv/arch22/stpsv_kernel_utils.h     # packed 索引 / 扫描方向 / UB 工具（Host 与 Kernel 共享常量）
blas/tpsv/arch22/stpsv_tiling_data.h      # Tiling 结构体（Host/Kernel 共享）
blas/tpsv/arch22/stpsv_aux_kernel.cpp     # Kernel：x 规整（gather）/ 回写（scatter），仅 |incx| ≠ 1 时使用
test/tpsv/stpsv/stpsv_param.h             # 【已有】CSV 参数解析（需扩展可选列，见 §测试设计）
test/tpsv/stpsv/stpsv_golden.h            # 【已有】cblas stpsv golden + packed 索引 CPU 版
test/tpsv/stpsv/arch22/stpsv_test.cpp     # 【新建】GTest 用例（CSV 驱动 + 负向用例）
test/tpsv/stpsv/arch22/stpsv_test.csv     # 【新建】驱动用例
test/tpsv/stpsv/arch22/stpsv_npu_wrapper.h# 【新建】NPU 侧调用封装（负责 H2D/D2H 与 handle 构造）
test/tpsv/stpsv/CMakeLists.txt            # 【已有】ops_blas_add_gtest_tests(${OPS_BLAS})，自动收集 arch22
```

工程约束沿用仓内既有约定：编译期由 `get_soc_arch_dirs()` 按 `SOC_VERSION` 选择 `arch22`（`ascend910b*` / `ascend910_93*`）或 `arch35`（`ascend950*`），**同一时刻只编译一个 arch 目录**，因此 `arch22/stpsv_host.cpp` 与 `arch35/stpsv_host.cpp` 同名不会产生重复符号；kernel 入口统一标注 `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY)`，Host 侧不调用 `aclrtSynchronizeDevice`（异步语义交由 handle stream）。

### 实现方案

#### 3.2.1 Host 侧设计

##### 0. 核心难点

| 难点 | 分析 |
| --- | --- |
| **串行依赖** | x_i 依赖同一次求解中的前序分量，无法把所有输出元素无依赖并行；并行度只能来自"同一扫描步对多个未解分量的批量更新" |
| **原地语义** | x 既是输入又是输出：入口 b 与出口解共用一个缓冲。所幸求解是"单向扫描"（每个 k 只影响尚未求解的分量），因此原地更新天然安全，**无需 workspace 中转**（这一点与同族 TRMV/TPMV 的原地语义不同） |
| **跨核同步受限** | arch22 上同一 kernel 内没有可靠的通用 AIV 全局栅栏。跨核阶段序**必须**由同一 stream 上多次 kernel launch 天然提供（与仓内既有实践一致，禁止使用无保障的跨 AIV 自旋等待） |
| **packed 连续性** | 无 lda，且 packed 布局下"按行取数"会退化为跨列碎片读（n = 4096 时产生数倍带宽放大）。kernel 侧必须按 **列方向连续段**组织访存（见 §kernel 侧设计） |
| **负步长/非单位步长** | `\|incx\| ≠ 1` 时 x 的物理下标跨步，向量化 axpy/DataCopy 无法直接使用 |

##### 1. 归一化与访存模式选择

Host 侧先做两项归一化，使 kernel 侧的热路径尽可能单一：

**(a) 扫描方向归一化**：计算 `uploEff = (trans == N) ? uplo : opposite(uplo)`，由 `uploEff` 唯一决定扫描方向（LOWER → k 递增前代；UPPER → k 递减回代）。kernel 侧只感知 `uploEff`。

**(b) 访存模式归一化**：packed 存储下"连续段"出现在固定的一侧，由 `trans` 唯一决定，形成两类 kernel 变体：

| 模式 | 触发条件 | 连续段含义 | 组织方式 |
| --- | --- | --- | --- |
| **AXPY 模式** | `trans == N` | 物理列 k 的"对角之后"部分（LOWER：`AP[colstart(k)+1 .. colstart(k)+n−k−1]`，即行 k+1..n−1；UPPER：`AP[colstart(k) .. colstart(k)+k−1]`，即行 0..k−1） | 列扫描：`x_k /= diag` 后，对全部未解分量做一次向量化 axpy |
| **DOT 模式** | `trans != N`（T/C 合并） | 物理列 k 的"对角之后"部分（与 Aeff 的第 k 行同构：Aeff(k,j) = A(j,k)） | 行扫描：输出行 k 做一次内积，内积的两个操作数都是连续段 |

> 该划分与 Netlib `stpsv.f` 的两个分支严格对应：`LSAME(TRANS,'N')` 分支逐列做 `X(I) = X(I) − TEMP*AP(K)`（即 AXPY 列扫描）；`ELSE` 分支逐行做 `TEMP = TEMP − AP(K)*X(I)`（即 DOT 行扫描）。**本算子的 kernel 与 Netlib 执行完全相同的浮点运算序列，只是改变了并行调度**（见 §精度策略）。

**(c) incx 归一化**：若 `|incx| != 1`，先用一个轻量 kernel 把 x 按逻辑序 gather 到 workspace 连续缓冲 `xw`，主求解全程使用 `xw`（等价 `incx' = 1`），求解结束再用一个 kernel 把 `xw` scatter 回 x 的物理位置。收益：

- 主求解 kernel 完全不需要处理跨步访存，热路径唯一、可充分向量化；
- scatter 只写逻辑位置，x 的物理空洞（`|incx| > 1` 时的间隙）保持不变，满足"不得改写 padding"的约束；
- 开销 O(n)：n = 4096 时两趟共 32 KB，可忽略。

`incx == ±1`（含全部 4 条性能 case）时不触发该路径，零额外开销。

##### 2. 执行路径与分块策略

由于扫描是串行的，本设计采用**面板分块右看（panel-blocked right-looking）**方案，把"串行临界路径"压缩为面板内部的求解，把"数据量占绝对多数"的尾块更新交给全核并行：

```
将扫描序切分为宽度 PB 的面板： panel p 覆盖扫描索引 [p·PB, min((p+1)·PB, n))
对每个面板 p（顺序执行，由 stream 保序）：
  阶段 A  PanelSolve  —— 求解面板内 PB 个分量的解（单核，UB 常驻，向量化）
  阶段 B  TrailUpdate —— 用已解出的面板分量，更新面板之后全部未解分量的右端（全核，行区间并行）
```

- **PanelSolve（单核）**：面板内分量存在严格递推关系，必须串行。把面板窗口的 x（≤ PB 个 float，PB = 512 时仅 2 KB）常驻 UB，逐扫描索引 k 做：`x_k` 除以对角（非单位对角）→ 用 `AP` 的连续段做一次长度为 ≤ PB 的**向量化 axpy/内积**（`Axpy` / `Mul`+`ReduceSum`）。AP 段按需从 GM `DataCopy` 进 UB（AXPY 模式逐列取段，DOT 模式逐行取段，长度均 ≤ PB）。
- **TrailUpdate（全核）**：把面板之后的未解行按连续区间分给各核，各核只写自己拥有的行、无重叠、无需原子操作；每个核按扫描索引读取 `AP` 的连续子段（AXPY 模式：`x_rowrange -= aseg * x_k`；DOT 模式：`acc_row -= dot(aseg, xPanel)`）并做归约。AP 的每个元素在整个求解过程中**恰好被读取一次**，且每次读取都落在连续区间内。

**PB 取值**：由 Host 依 `n` 计算，默认 `PB = min(512, n)`，即

| n | 面板数 P | kernel launch 数 | 说明 |
| --- | --- | --- | --- |
| ≤ 512 | 1 | 1（尾块为空，阶段 B 跳过） | 退化为单核整解，延迟最优 |
| 1024 | 2 | 4 | |
| 2048 | 4 | 8 | |
| 4096 | 8 | 16 | |

> PB 是纯调优量：增大 PB 会减少 launch 次数但拉长单核临界路径，减小 PB 反之。默认 512 的选取依据见 §性能标杆与达标分析；实机用 msprof 标定后可在 [128, 1024] 区间回调。

##### 3. workspace 需求

| 用途 | 大小 | 触发条件 |
| --- | --- | --- |
| x 规整缓冲 `xw` | `n × sizeof(float)`（n = 4096 → 16 KB） | `\|incx\| != 1` |
| 合计最大需求 | 16 KB | 远低于 handle 默认 workspace（32 MiB） |

复用仓内既有机制（`blas/common/helper/aclblas_handle_internal.h`）：

```cpp
size_t need = (std::abs(incx) == 1) ? 0 : static_cast<size_t>(n) * sizeof(float);
if (need > 0 && !CheckEffectiveWorkspaceSize(h, need)) {
    if (EnsureDefaultWorkspace(h, need) != ACLBLAS_STATUS_SUCCESS) { return ACLBLAS_STATUS_ALLOC_FAILED; }
}
auto* ws = reinterpret_cast<uint8_t*>(GetEffectiveWorkspace(h));
```

`incx == ±1`（全部性能 case）时 `need == 0`，不申请、不校验、零开销。用户通过 `aclblasSetWorkspace` 注入自有 workspace 时不做扩容，容量不足直接返回 `ACLBLAS_STATUS_ALLOC_FAILED`。

##### 4. launch 调度

所有 kernel 顺序下发到 handle 绑定的同一 stream，由 stream 天然保证阶段序：

```
if |incx| != 1:  launch GatherX<<<numBlocks>>>(x, xw, n, incx)
for p in 0 .. P-1:
    launch PanelSolve<<<1>>>(tiling, p)
    if 存在未解行: launch TrailUpdate<<<numBlocks>>>(tiling, p)
if |incx| != 1:  launch ScatterX<<<numBlocks>>>(xw, x, n, incx)
```

- `PanelSolve` 恒定单 block（面板内递推不可并行）；
- `TrailUpdate` / `GatherX` / `ScatterX` 用满核（`GetBlockNum()`），按连续行区间均分，余数分给前几个核，不区分大小核；
- Host 不插入任何同步；读回 Device 结果须由调用方 `aclrtSynchronizeStream`。

##### 5. Tiling 数据

```cpp
struct StpsvTilingData {
    uint64_t ap;         // packed A（Device）
    uint64_t x;          // 求解用连续 x（Device）：|incx|==1 时为用户 x，否则为 workspace 缓冲
    uint32_t n;
    uint32_t uplo;       // ACLBLAS_UPPER / ACLBLAS_LOWER（原值）
    uint32_t trans;      // ACLBLAS_OP_N / _T / _C（决定 AXPY / DOT 模式）
    uint32_t diag;       // ACLBLAS_NON_UNIT / ACLBLAS_UNIT
    uint32_t useCoreNum; // TrailUpdate 实际用核数
    uint32_t panelSize;  // PB
    uint32_t panelIdx;   // 当前面板序号（逐面板下发时更新）
    uint32_t reserved;   // 8 字节对齐占位
};
```

辅助量（Host 侧计算，写入 tiling）：

```
uploEff   = (trans == ACLBLAS_OP_N) ? uplo : opposite(uplo)
scanFwd   = (uploEff == ACLBLAS_LOWER)          // true: k 递增前代
panelNum  = ceil(n / PB)
p0        = panelIdx * PB
p1        = min(p0 + PB, n)
```

##### 6. tilingkey 规划

tilingkey 用于让 kernel 侧在**编译期**展开 `uploEff × diag × 访存模式` 分支、消除内层判断。编码：

```
tilingKey = uploEff(1 bit) | diag(1 bit) << 1 | mode(1 bit) << 2      // mode: 0 = AXPY(trans==N), 1 = DOT(trans!=N)
          = 2 × 2 × 2 = 8 个编译期实例（PanelSolve / TrailUpdate 各 8 个入口）
```

Host 侧按 tilingKey 分发到对应的 `__global__ __aicore__` 入口；kernel 侧以模板参数 `<bool UPLO_IS_UPPER, bool DIAG_IS_UNIT, bool MODE_DOT>` 实例化。**`trans = T` 与 `trans = C` 合并为同一模板实例**（实数域 C ≡ T）。

#### 3.2.2 Kernel 侧设计

Kernel 分 Init / Process 两段；Process = CopyIn（按需把 AP 连续段搬入 UB）→ Compute（向量化 axpy / 内积）→ CopyOut（写回 x）。全部为 AIV。

##### 1. packed 连续段的统一索引

记扫描索引为 k，以下所有下标均以 **0-based** 表示，`int64_t` 计算以防大 n 溢出：

| 模式 | 段语义 | 段首 | 段长 | 对角位置 |
| --- | --- | --- | --- | --- |
| AXPY（trans = N） | 物理列 k 的"对角之后" | UPPER：`colstart_U(k)`；LOWER：`colstart_L(k) + 1` | UPPER：`k`；LOWER：`n−k−1` | UPPER：`colstart_U(k)+k`；LOWER：`colstart_L(k)` |
| DOT（trans != N） | 物理列 k 的"对角之后"（与 Aeff 第 k 行同构） | 同上 | 同上 | 同上 |

其中 `colstart_U(k) = k(k+1)/2`、`colstart_L(k) = k(2n−k+1)/2`。

**关键性质（两种模式共享）**：在任意扫描方向下，"对角之后"的那一段永远是连续区间，且该段所对应的**未解分量集合**正好是"同侧的全部未解分量"。因此：

- 段读取永远连续（无跨列碎片读）；
- 段写入的 x 目标索引为一个连续区间（AXPY 模式）或一次内积（DOT 模式）。

##### 2. PanelSolve kernel（单核，每面板一次）

```
Init:  xPanelUb ← 面板窗口的 x[p0 .. p1)（PB 个 float，UB 常驻）
Process（按扫描方向遍历 k ∈ [p0, p1)）:
    CopyIn:  DataCopy 从 GM 取 AP 的连续段到 UB（段定义见下表，长度 ≤ PB）
    Compute:
        # 对角与段地址（按 uploEff 编译期确定）
        diagOff   = colstart(k) + (LOWER ? 0 : k)
        offSegBeg = colstart(k) + (LOWER ? 1 : 0)          # 非对角连续段
        if MODE == AXPY:                                   # trans == N
            if DIAG == NON_UNIT: xPanelUb[k] = xPanelUb[k] / apUb[diagOff]   // if constexpr 剔除对角 load
            Axpy(xPanelUb[面板内 k 之后的未解窗口], apUb[offSegBeg ...], xPanelUb[k], 窗口长度)
        if MODE == DOT:                                    # trans == T / C
            acc = ReduceSum(apUb[面板内段] · xPanelUb[面板内 k 之后的已解部分])
            xPanelUb[k] = (xPanelUb[k] − acc) / (DIAG == UNIT ? 1 : apUb[diagOff])
    （面板内更新只作用于"尚未求解的面板内分量"，完全在 UB 内完成；尾块由阶段 B 处理）
CopyOut: 把面板窗口的最终解写回 x（连续区间，1 次 DataCopy）
```

- UB 预算：面板窗口 `PB×4 B`（PB = 512 → 2 KB）+ AP 段 `PB×4 B`（2 KB）+ 内积临时区，`BUFFER_NUM = 2` 双缓冲，合计 < 16 KB（UB 总量 248 KB，充裕）；
- `diag = UNIT` 时对角地址的 load 由 `if constexpr` 在**编译期剔除**——即使 AP 对角位置被填入 NaN/Inf，结果也不受影响，与 golden 语义一致；
- 面板内 x 的物理位置连续（`|incx| == 1` 归一化后保证），CopyIn/CopyOut 均为单次连续 `DataCopy`。

##### 3. TrailUpdate kernel（全核，每面板一次）

```
Init:  按 GetBlockIdx()/GetBlockNum() 计算本核负责的未解行连续区间 [t0, t1)
       xTrailUb ← 本核区间内的 x[t0 .. t1)（UB 常驻，长度 ≤ n / numBlocks）
       xPanelUb ← 面板解 x[p0 .. p1)（PB 个 float，UB 常驻一次）
Process（按扫描方向遍历 k ∈ [p0, p1)）:
    if MODE == AXPY:                                   # trans == N：按列做分段 axpy
        CopyIn:  DataCopy 取列 k 中本核行区间对应的连续子段 [colstart(k)+Δ, ...)，长度 t1 − t0
        Compute: Axpy(xTrailUb, asegUb, xPanelUb[k], t1 − t0)
    if MODE == DOT:                                    # trans != N：按行做内积
        for r in [t0, t1):                             # 各行独立，无跨行依赖
            CopyIn:  DataCopy 取存储列 r 中"对应面板行"的那一段（连续，长度 = 面板宽）
            Compute: xTrailUb[r] −= ReduceSum(asegUb · xPanelUb)
CopyOut: 写回 x 的连续区间（每核 1 次 DataCopy）
```

（Δ 为列首到本核首行的偏移：LOWER 为 `t0 − k`，UPPER 为 `t0`；两种模式下"行区间 → 列内偏移"的映射都保持连续。）

- **各核只写自己拥有的行区间，区间互不重叠**，不产生写冲突，不需要原子操作；
- 每个 AP 元素在这两个 kernel 合计恰好被读取一次（PanelSolve 读面板内三角块，TrailUpdate 读面板之外的三角块），无冗余读；
- 空核（`t0 >= t1`，n 小于核数时可能出现）直接 `return`。

##### 4. 精度策略

**核心结论：本算子的 kernel 与 Netlib `stpsv` 执行完全相同的浮点运算，累加顺序也保持一致，因此逐元素误差相对 cblas golden 为 ULP 级。**

| 维度 | 做法 | 与 golden 的一致性 |
| --- | --- | --- |
| 累加顺序 | AXPY 模式：按列序（= Netlib 的 J 序）逐次 `x_i -= a·x_k`；DOT 模式：按行序（= Netlib 的 I 序）累加 | Netlib 同一分支的运算序完全一致 |
| 对角处理 | 在扫描步开头做除法（与 Netlib `X(J) = X(J)/AP(KK)` 位置一致） | 一致 |
| 面板切分带来的重排 | DOT 模式下面板内部分与"已解部分"的相加被拆成两段（已解部分在面板求解前先行归约） | 存在浮点结合顺序差异，量级为 ULP；FLOAT32 阈值 `rtol = 2⁻¹⁰ ≈ 1000 ULP` 远宽于该量级 |
| 向量归约 | DOT 模式的内积使用树形归约（`ReduceSum`），而非标量顺序累加 | 同上，ULP 级 |
| **不做 `X(J) == 0` 跳过** | Netlib 参考实现含 `IF (X(J).NE.ZERO)` 提前跳过优化；本实现**不做跳过**（对齐 cuBLAS 语义） | 仅在 `x_j` 恰为精确 0 **且** AP 该列含 ±Inf/NaN 时才会发散；配套用例的 `VALUE_NORM_0` 填充搭配有限值 AP、Inf/NaN 填充搭配非零 x，两者不交叉（见「风险与待确认项 #4」） |
| Inf/NaN 传播 | 全程 FP32，无降精度中间类型；不做 FMA 融合改写 | 传播路径与 Netlib 一致 |
| 中间精度 | 所有中间量为 FP32，除法用 IEEE 单精度除法 | 一致 |

> 简化说明：`n ≤ PB`（退化单面板）时，PanelSolve 独占整个求解过程且无尾块更新，标量累加顺序与 Netlib **逐位同构**，可用于定位精度问题的基线对照。

##### 5. 边界与退化

| 场景 | 处理 |
| --- | --- |
| `n = 0` | Host 直接返回 `SUCCESS`，不下发 kernel |
| `n = 1` | 单面板，`PB = 1`；`diag = UNIT` → `x_0` 不变；否则 `x_0 /= AP[0]` |
| `PB >= n` | 面板数 P = 1，阶段 B 无未解行，**跳过 TrailUpdate launch**（性能 case 1 即走此路径，仅 1 次 launch） |
| 空核 | `t0 >= t1` 直接 return |
| 大 n 索引 | `colstart(k)`、`k·(k+1)/2` 一律用 `int64_t`/`uint64_t` 计算，避免 32 位溢出（n = 4096 时 AP 下标上界已达 8.4 M） |
| `|incx| != 1` | 走 Gather/Scatter 归一化路径；x 物理空洞不改写 |
| 非连续 Tensor | 不支持（超出 incx 语义的内存访问不要求支持） |
| 确定性 | 分核区间固定，结果确定；但确定性计算不作要求（不保证 bit-exact） |

### 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas A2 训练系列产品 / Atlas A2 推理系列产品（DAV_2201 / arch22） | √ |
| Atlas A3 训练系列产品 / Atlas A3 推理系列产品（DAV_2201 / arch22） | √ |
| Ascend 950PR / Ascend 950DT（arch35） | 由 `blas/tpsv/arch35/` 既有实现覆盖，本设计不修改 |

### 算子约束限制

- 仅支持 FLOAT32（FP32）；不支持 FP16/BF16/双精度与复数 dtype（复数族由 Ctpsv 另行实现）
- 不支持 `incx = 0`
- 不支持非连续 Tensor（超出 `incx` 语义的内存访问）
- 不支持广播；n 为运行时入参，不做 dynamic shape 编译期特化
- A 必须为方阵三角，packed 存储无 lda，不支持普通二维矩阵输入
- `diag = ACLBLAS_UNIT` 时实现不读取也不校验 AP 对角位的内容
- 不做奇异性/近奇异性检测，`diag = NON_UNIT` 时调用方须保证对角元非零
- 不保证 bit-exact 结果（确定性计算不作要求）
- 依赖 `aclblasSetStream` 绑定 stream；读回 Device 结果前须由调用方同步 stream

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | FLOAT32：`atol = 2⁻¹⁶ ≈ 1.5259e-5`、`rtol = 2⁻¹⁰ ≈ 9.7656e-4`；逐元素 `\|actual − golden\| ≤ atol + rtol·\|golden\|`；用例级 `matched_ratio ≥ 0.99` 且 `max_abs_error ≤ 1e-2`（或 32×ULP） | 任务书 §3.2 + 生态算子开源精度标准（FLOAT32 档） |
| golden | cblas（Netlib `stpsv`）单标杆，对输出 x 的 n 个元素（按 incx 步长）全量逐元素比对 | 任务书 §3.2 |
| 测试工程口径 | MERE = 2⁻¹³ ≈ 1.2207e-4（平均相对误差上限），MARE 离群倍率 10.0 | ops-blas test frame 惯例（与 `test/tpsv/stpsv/arch35/stpsv_test.csv` 一致） |
| 性能标准 | 4 条典型 case 平均单次耗时 ≤ 标杆（Avg time，us，warmup 后有效采样 > 50 次） | 任务书 §3.3 |
| 内存标准 | 不涉及固定上限；workspace 需求 ≤ 16 KB（`n = 4096` 且 `\|incx\| ≠ 1`） | 任务书 §3.4 |

### 性能标杆与达标分析

任务书 §3.3 标杆（= GPU A100 实测 `gpu_ms` / 0.8，与配套 `gpu_baseline.csv` 的 `stpsv-base-000~003` 精确对应）：

| case | n | uplo | trans | diag | incx | 标杆耗时（us） |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 512 | LOWER | N | NON_UNIT | 1 | 432.71 |
| 2 | 1024 | UPPER | N | NON_UNIT | 1 | 941.23 |
| 3 | 2048 | LOWER | T | NON_UNIT | 1 | 1690.94 |
| 4 | 4096 | UPPER | C | NON_UNIT | 1 | 5185.2 |

**可行性估算**（实施前量级估算，实机以 msprof 标定；假设 AIV 主频 ≈ 1.4 GHz，单核向量指令 ≈ 64 FP32 lane，HBM 聚合带宽按 400 GB/s 保守估计）：

三角求解的总浮点工作量仅 `n²/2` 次乘加，n = 4096 时也只有 8.4 M 次——**远未触及算力上限**，耗时几乎完全由"串行临界路径"与"AP 单次读取的访存量"决定：

| case | AP 数据量 | 访存下界（全核聚合） | 单核临界路径（P × PB 步） | launch 开销（2P−1 次） | 估算总计 | 标杆 | 余量 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.53 MB | < 2 us | 512 步 ≈ 22 us | 1 次 ≈ 5 us | ≈ 30 us | 432.71 us | ≈ 14× |
| 2 | 2.10 MB | ≈ 6 us | 2 × 512 步 ≈ 44 us | 3 次 ≈ 15 us | ≈ 70 us | 941.23 us | ≈ 13× |
| 3 | 8.40 MB | ≈ 21 us | 4 × 512 步 ≈ 88 us | 7 次 ≈ 35 us | ≈ 150 us | 1690.94 us | ≈ 11× |
| 4 | 33.6 MB | ≈ 84 us | 8 × 512 步 ≈ 176 us | 15 次 ≈ 75 us | ≈ 340 us | 5185.2 us | ≈ 15× |

（临界路径按"单核每扫描步 ≈ 60 cycles"估算，已含 DataCopy 与向量归约开销；launch 开销按 5 us/次估算。4 条 case 均为 `incx = 1`，不触发 Gather/Scatter，也不触发 workspace 扩容。）

同时需要说明：`blas/trsv/arch22/strsv_*` 当前采用**单核标量逐元素 `GetValue/SetValue`** 的实现方式，其每步开销为数十至数百 cycle 且完全串行，按同口径外推 n = 4096 需数秒量级，**无法达标**——这正是本设计必须走"面板分块 + 向量化 + 全核尾块更新"的原因。本设计比该实现多出的关键收益：

1. AP 访存全部落在连续区间，消除 packed 布局下的跨列碎片读；
2. 尾块更新（占总工作量 `≈ 1 − 1/P`）由全核并行，AP 的读取带宽得以聚合；
3. 串行临界路径被压缩到 `P × PB = n` 个"单核向量步"，与 n 线性而非二次相关。

> 上述估算为设计阶段量级判断，不作为验收依据。PB、`useCoreNum`、UB 分段策略均为调优量，实机用 msprof 采集 AIV/HBM 指标后回调。

## 测试设计

测试工程基于 ops-blas `test/frame`（CSV 驱动 + GTest），在 `test/tpsv/stpsv/` 下新增 `arch22/`，复用同目录已有的 `stpsv_param.h`（CSV 解析）与 `stpsv_golden.h`（cblas golden + packed 索引 CPU 版），golden 由 cblas（Netlib `stpsv`）生成。

### 需要一并完成的测试框架扩展

| 项 | 现状 | 本任务需做 |
| --- | --- | --- |
| `TpsvParam` 解析字段 | 仅 `uplo/trans/diag/n/incx`（+ 基类的 `expect_result`、`mere_threshold`、`mare_multiplier`、`random_seed`） | 扩展解析可选列 `ap_fill` / `x_fill`（默认 `"RANDOM"`，向后兼容），用于表达 `NULLPTR`、`VALUE_NORM_0`、`RANDOM_ALTER`、`RANDOM_EXTREME`、`VALUE_NORM_INF`、`VALUE_NORM_NAN` 等填充模式 |
| `arch22/stpsv_npu_wrapper.h` | 不存在 | 新建，提供 `aclblasStpsv_npu(...)`：负责 handle 构造、H2D/D2H、按 `\|incx\|` 计算物理长度、nullptr 透传 |
| `arch22/stpsv_test.cpp` | 不存在 | 新建，CSV 驱动 GTest + 负向用例（null handle、`diag=UNIT` 对角毒化） |
| `arch22/stpsv_test.csv` | 不存在 | 新建；列名与仓内 `TC_*` 惯例一致，可直接复用任务配套 CSV |

### 用例集

任务配套 `test_cases/stpsv_test.csv` 共 **1200 条**（`gen_csv.py` 固定种子生成，条数可扩展），本设计全量采纳：

| 类别 | 前缀 | 条数 | 覆盖点 |
| --- | --- | --- | --- |
| L0 基础 | TC_L0 | 24 | 12 组枚举全组合 × 小尺寸 (8, 32) |
| L1 尺寸 | TC_SQ | 92 | 尺寸池 1→2048（含 1、质数 3/5/7、2 的幂及 ±1、非对齐值 33/65/127/1025）× 枚举轮转 |
| L2 步长 | TC_INC | 24 | incx ∈ {±1, ±2, ±3} × uplo × diag（trans = N） |
| L5 填充 | TC_FL | 8 | AP：均匀/交替/极端；x：全零/交替/极端/Inf/NaN |
| L5b 覆盖 | TC_CV | 96 | 中尺寸（10~400）× 12 组枚举全组合 |
| L6 边界 | TC_ED | 9 | n = 0 quick return（SUCCESS）/ AP、x 空指针 / 三枚举非法 / 负维度 / 零步长 |
| EX 扩展 | TC_EX | 747 | 尺寸池 × 枚举组合 × 步长的确定性采样 |
| PF 性能 | TC_PF | 200 | 4 条任务书典型 case（逐参数一致）+ 小尺寸 + 阶数对数扫描 + 预算内混合，均为连续访存（incx = 1） |

CSV 列格式：

```
case_name,description,uplo,trans,diag,n,incx,ap_fill,x_fill,
random_seed,expect_result,mere_threshold,mare_multiplier
```

前 7 列与仓内 `TpsvParam` 的 ReadMap 逐一对应；`ap_fill` / `x_fill` 为本次新增可选列；非法枚举用字面量 `INVALID`（与 `arch35/stpsv_test.csv` 的 `TC_INV_001~003` 写法一致），空指针用 `NULLPTR`。

### 负向用例与异常行为对照

| 异常行为（任务书 §2.4） | 用例 | expect_result |
| --- | --- | --- |
| n = 0 合法 no-op | TC_ED `n0_quick_return_UPPER/LOWER` | `ACLBLAS_STATUS_SUCCESS` |
| n > 0 时 AP 为 nullptr | TC_ED `null_ap`（`ap_fill=NULLPTR`） | `ACLBLAS_STATUS_INVALID_VALUE` |
| n > 0 时 x 为 nullptr | TC_ED `null_x`（`x_fill=NULLPTR`） | `ACLBLAS_STATUS_INVALID_VALUE` |
| uplo / trans / diag 非法枚举 | TC_ED `invalid_uplo / invalid_trans / invalid_diag` | `ACLBLAS_STATUS_INVALID_VALUE` |
| n < 0 | TC_ED `neg_n` | `ACLBLAS_STATUS_INVALID_VALUE` |
| incx = 0 | TC_ED `incx0` | `ACLBLAS_STATUS_INVALID_VALUE` |
| handle 为 nullptr | 测试工程构造（不入 CSV） | `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |

### 补充用例（任务配套集合之外，本设计自行补齐）

1. **packed 布局交叉验证**：固定手算小矩阵（n = 1/2/3），UPPER 与 LOWER 各 6 种枚举组合，逐元素硬编码期望值——用于直接暴露任务书 LOWER 索引公式的偏差（勘误 #1）。
2. **对角位毒化**：`diag = UNIT` 时把 AP **全部对角位置填为 NaN/Inf**，验证结果不受影响（证明对角未被 load）。
3. **物理空洞完整性**：`incx = ±2/±3` 时用 canary 填充 x 的物理空洞，验证 Gather/Scatter 归一化路径**未改写**空隙。
4. **面板边界专项**：n = PB、PB±1（即 511/512/513/1023/1024/1025），覆盖单面板退化、面板数与核数非整除、空核 early-return 分支。
5. **核数边界**：n 小于 / 等于 / 略大于 AIV 核数，覆盖 TrailUpdate 空区间分支。
6. **面板等价性**：同一组输入分别以 `PB = n`（单面板）与 `PB = 512`（多面板）执行，两者输出差异应在 ULP 量级——用于证明面板切分未引入逻辑错误。
7. **负步长 + 非整除组合**：`incx = -1/-3` 且 n 为奇数，验证逻辑下标与物理下标映射。
8. **大规模**：n = 8192（如时间允许），验证 64 位 packed 索引与 workspace 在跨多面板时正确。

### 可复现执行口径

| 项 | 说明 |
| --- | --- |
| 精度 | `python verify_accuracy.py --repo <ops-blas> --soc ascend910b3 --csv ./stpsv_test.csv`（脚本自动安装 CSV、`build.sh --soc=ascend910b3 --ops=stpsv` 编译、运行 GTest 并解析逐条 PASS/FAIL，自动排除 `TC_PF_` 前缀用例） |
| 性能 | `python verify_performance.py --repo <ops-blas> --soc ascend910b3`，与 `gpu_baseline.csv` 关联比对，先 warmup 再有效采样 > 50 次取平均 |
| 记录项 | CANN / 驱动版本、SOC、CSV 随机种子、编译参数、逐条用例结果、精度对比截图、性能数据截图、workspace 占用数据 |
| 诊断项 | 正常 `NON_UNIT` 用例记录对角最小幅值，避免把病态系统误判为 kernel 错误 |

### 交付件

1. 设计文档 PR 合入截图；
2. 测试用例（`stpsv_test.csv` + CSV 驱动 GTest + 测试步骤 readme，保证验收人可复现）；
3. 用例结果自测报告（含逐条精度对比结果与截图、性能数据 > 50 次平均与截图、内存占用数据）；
4. 待验收代码地址（个人仓链接、分支、算子目录 `blas/tpsv/arch22/` 与 `test/tpsv/stpsv/arch22/`，并邀请 `Ascend-CANN` 为开发者）。

## 兼容性分析

- **不修改公开接口**：`aclblasStpsv` 声明已存在于 `include/cann_ops_blas.h:160`，签名与参数语义保持原样，无 ABI 破坏；未定义任何产品私有平行 API。
- **不影响既有实现**：新增 `blas/tpsv/arch22/` 与 `test/tpsv/stpsv/arch22/`，`arch35/`（Ascend 950PR）路径完全不动。编译期由 `get_soc_arch_dirs()` 按 SOC 只选择单一 arch 目录，同名文件（`stpsv_host.cpp`、`stpsv_tiling_data.h`）不会产生重复符号——与 `blas/trsv/`、`blas/tpmv/` 的既有 arch22/arch35 并存方式一致。
- **产品支持表**：`blas/tpsv/README.md` 中 `aclblasStpsv` 的 Atlas A2/A3 行由「不支持」改为「支持」，并补充 `arch22` 目录说明。
- **跨产品线一致**：与 arch35 实现共享 tiling 字段命名、参数校验顺序、packed 索引记法（`TpsvPackedUpperIdx / TpsvPackedLowerIdx`）与错误码口径，便于后续 Ctpsv/Dtpsv/Ztpsv 复用。
- **对同族算子的可推广性**：packed 扫描索引工具、面板分块右看框架、Gather/Scatter 归一化三处可下沉到 `blas/common/helper/`，供 `tbmv`、`spmv`、`stbsv` 等后续 packed 族求解算子复用。
- **数值兼容**：与 Netlib `stpsv` 的运算序列一致（仅并行调度不同），相对 cblas golden 的逐元素差异为 ULP 级，满足 `rtol = 2⁻¹⁰` 阈值。

## 风险与待确认项

| # | 事项 | 影响 | 处置 |
| --- | --- | --- | --- |
| 1 | 任务书 §2.1 的 LOWER packed 公式 `AP[i + ((2n−j−1)·j)/2 + i]` 末尾多一个 `+ i`，n = 3 时对角下标算出 7（合法上界 5），**越界** | 高：按任务书公式实现将导致全量 golden 比对失败 | 已用 Netlib `KK` 推进、仓内 arch35 `TpsvPackedLowerIdx`、测试 golden `TpsvPackedLowerIdxCpu` 三处交叉确认正确式为 `i + j(2n−j−1)/2`；**实现以 Netlib/仓内既有实现为准**，并在任务讨论帖反馈勘误 |
| 2 | 任务书 §2.4 文字称非法枚举返回 `ACLBLAS_STATUS_INVALID_ENUM`，与仓内 arch35 已合入实现、任务配套 CSV 用例（`TC_INV_001~003`）的 `ACLBLAS_STATUS_INVALID_VALUE` 不一致 | 中：口径不一致会导致用例判定歧义 | 本设计跟随**仓内既有实现 + 配套 CSV**（`INVALID_VALUE`），保证与 arch35 跨产品线一致；在任务讨论帖确认最终口径 |
| 3 | 配套 `gpu_baseline.csv` 的 `gpu_ms` 列已回填（`stpsv-base-000~003` 为 0.346172/0.752986/1.352750/4.148160 ms），÷0.8 后与任务书 §3.3 标杆一致 | 低 | 性能以任务书 §3.3 标杆为准；回填数据用于倍率复核 |
| 4 | golden 所用 cblas 实现带 Netlib 的 `X(J).NE.ZERO` 跳过优化，在 `x_j = 0 且 AP 该列含 Inf/NaN` 组合下会与"不跳过"发散 | 低 | 现有用例中零填充（`VALUE_NORM_0`）配有限值 AP、Inf/NaN 填充配非零 x，两者不交叉；另加补充用例 3 专项观察，如实测发散则改为与 golden 一致的跳过策略 |
| 5 | PB（默认 512）、`useCoreNum`、UB 分段/双缓冲阈值、`N_SINGLE` 类阈值均为经验值 | 低 | 首轮取 PB = 512，在 A2（910B3）上用 `TC_PF` 小尺寸与 4 条典型 case 实测 msprof 后回调至 [128, 1024] 区间 |
| 6 | arch22 单内核内无通用 AIV 全局栅栏，故采用"每面板 2 次 launch"的阶段序 | 低（已规避） | 阶段序由同一 stream 天然保证，不使用跨 AIV 自旋等待；若实测 launch 开销占比过高，再评估"单 kernel + `AscendC::SyncAll()`"方案（需先在 A2 上验证正确性与收益，作为后续优化项，不作为基线） |
| 7 | 任务配套测试 CSV 含 `description` / `ap_fill` / `x_fill` 三个 arch35 未使用的列，现有 `TpsvParam` 未解析后两列 | 中 | 本任务扩展 `TpsvParam`（ReadMap 带默认值 `"RANDOM"`，向后兼容 arch35 既有 CSV 写法）并在设计文档 §测试设计中登记，避免影响 arch35 用例 |
| 8 | 本算子不做奇异性检测，`NON_UNIT` 用例须保证对角远离零 | 低 | 沿用仓内 arch35 测试做法：填充后对被引用对角元加符号保持的偏移量 `boost = max(5, n)`，golden 侧使用同一偏移后的矩阵 |
