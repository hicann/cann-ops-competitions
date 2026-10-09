# aclblasStrmm 算子设计文档（Atlas A2 / A3 系列 · arch22 · 基于 CATLASS）

> 任务：9 月社区任务 #82 · `aclblasStrmm` 算子开发（A2/A3）　
> 提交团队：`gcw_NGVX66oU`
> 文档版本：v1.0
> 日期：2026-10-08
> 算子目录：`ops-blas/blas/trmm/arch22/`
> 测试目录：`ops-blas/test/trmm/strmm/arch22/`
> 任务书：`aclblasStrmm_A2A3_task_doc.md`（9 月社区任务 #82，下称“任务书”）
>
> **实现基座声明**：本设计基于 ops-blas 仓已引入的 `3rdparty/catlass` 子模块（CATLASS 提供了 `Trmm` kernel、`BlockMmadTla`、`PackedTileCopyTla`、`GemmIdentityBlockSwizzle`、`DeviceGemm` 等组件）实现，三角语义归一与数据规整由本算子自研 AIV kernel 完成。CATLASS 子模块原 pin（`db93081c`，v1.6.0）不含 `trmm.hpp`，本设计将 pin 升级到 `44e75248`（tag `v2.1.0`，含该 kernel；见 §3.2.5、§5.2 风险 R1）。

# 一、需求背景（required）

## 1.1 需求来源

通过昇腾社区任务完成开源仓算子贡献：在 Atlas A2 系列产品（性能测试设备 Atlas 800T A2 / 910B3）与 Atlas A3 系列产品（Atlas 800I A3）上，基于 Ascend C / CATLASS 编程语言开发单精度实数（float32）三角矩阵乘算子 `aclblasStrmm`，实现 `C = alpha * op(A) * B`（`side=LEFT`）或 `C = alpha * B * op(A)`（`side=RIGHT`），完成设计、开发、测试全流程，验收通过后合入昇腾算子开源仓 ops-blas：

- 算子实现：`blas/trmm/arch22/`
- 测试代码：`test/trmm/strmm/arch22/`

对标基线接口为 cuBLAS `cublasStrmm`，精度 golden 由 cblas（Netlib BLAS `strmm`）生成。

## 1.2 背景介绍

### 1.2.1 算子定位与实现现状

`aclblasStrmm` 是 ops-blas 库中的句柄式 BLAS Level-3 三角矩阵乘接口（**乘法类，非求解类**）。接口已在 `include/cann_ops_blas.h` 声明，参数序列与 `cublasStrmm` 一一对应：

```cpp
aclblasStatus_t aclblasStrmm(
    aclblasHandle_t handle, aclblasSideMode_t side, aclblasFillMode_t uplo, aclblasOperation_t trans,
    aclblasDiagType_t diag, int m, int n, const float* alpha, const float* A, int lda,
    const float* B, int ldb, float* C, int ldc);
```

仓内现状：

| 项 | 现状 |
|---|---|
| 接口声明 | `include/cann_ops_blas.h` 已有（不改动，供多产品线共用） |
| 算子实现 | `blas/trmm/arch35/` 已有 Ascend 950PR 实现（`strmm_host.cpp` / `strmm_kernel.cpp` / `strmm_tiling_data.h`）；**arch22（Atlas A2/A3）目录不存在，为本次新建** |
| 产品支持表 | `blas/trmm/README.md` 当前标注 A2/A3 “不支持”，交付时需更新为“支持” |
| 测试骨架 | `test/trmm/strmm/strmm_param.h`（CSV 参数解析）、`test/trmm/strmm/strmm_golden.h`（cblas golden）、`test/trmm/strmm/CMakeLists.txt`（`ops_blas_add_gtest_tests`，按 SOC 架构目录自动纳入用例）**已存在**；缺 `arch22/strmm_test.cpp` 与 `arch22/strmm_test.csv` |
| 测试用例 | 任务包已提供 `test_cases/strmm_test.csv`（1200 条：1184 SUCCESS、15 INVALID_VALUE、1 HANDLE_IS_NULLPTR）与 `gen_csv.py`、`README.md`（测试指导）、`gpu_baseline.csv` |

### 1.2.2 标杆算子（cuBLAS `cublasStrmm` / Netlib `strmm`）现状分析

#### 1.2.2.1 支持的数据类型和数据格式

| 参数 | 类型 | 数据格式 | 说明 |
|---|---|---|---|
| alpha | `const float*` | Host 内存标量 | 单精度实数乘数 |
| A | `const float*` | **列主序（Column-Major）** ND，只读 | 三角矩阵；`side=LEFT` 为 m×m，`side=RIGHT` 为 n×n；前导维 `lda` |
| B | `const float*` | 列主序 ND，只读 | 一般矩阵 m×n；前导维 `ldb` |
| C | `float*` | 列主序 ND，输出（离席写出） | 一般矩阵 m×n；前导维 `ldc` |
| side | `aclblasSideMode_t` | Host attr | `{LEFT, RIGHT}` |
| uplo | `aclblasFillMode_t` | Host attr | `{UPPER, LOWER}`：A 的存储三角；另一三角**不被引用** |
| trans | `aclblasOperation_t` | Host attr | **仅 `{OP_N, OP_T}`**；`OP_C` 在实数语义下无定义，按任务书须拒绝 |
| diag | `aclblasDiagType_t` | Host attr | `{NON_UNIT, UNIT}`；UNIT 时主对角视为 1 且不读取 |

不支持非连续访问（超出 `lda/ldb/ldc` 语义）、广播与 dynamic shape 机制（`m/n` 为运行时入参）。

#### 1.2.2.2 标杆实现描述（语义要点）

1. **参数校验**：非法参数返回错误码；`handle` 为空返回 `HANDLE_IS_NULLPTR`；非法枚举、空指针、非法前导维返回 `INVALID_VALUE`。
2. **quick-return**：`m==0` 或 `n==0` 为合法 no-op，返回成功、不执行计算；`alpha==0` 时 A/B 不被引用（可为 `nullptr`），C 置零。
3. **三角/对角裁剪**：只引用 `uplo` 指定的存储三角，另一三角为 don't-care、不得影响结果；`diag=UNIT` 时对角视为 1 且不读取，`diag=NON_UNIT` 时读取对角（乘法类语义，**对角允许为 0**，无 TRSM 类的非零强制）。
4. **alpha 折入**：Netlib `strmm` 先把 alpha 折入被乘元素（`TEMP = ALPHA*B(k,j)`）再三角乘加，等价于 `alpha * op(A) * B`。
5. **`op(A)` 取向**：转置只改变 `op(A)` 的取向，**不改变被读取的存储三角**（`uplo=UPPER, trans=T` 仍只读 A 的上三角存储，但它作为下三角矩阵参与乘法）——这是本算子最易错点。

#### 1.2.2.3 标杆算子实现流程（伪代码）

```text
// 标杆语义：side=LEFT → C = alpha * op(A) * B；side=RIGHT → C = alpha * B * op(A)
strmm(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, C, ldc):
    // ① 零维快速返回：合法 no-op，不校验其余参数
    if m == 0 or n == 0:
        return SUCCESS

    // ② 句柄校验
    if handle == nullptr:
        return HANDLE_IS_NULLPTR

    // ③ 枚举 / 维度 / 指针 / 前导维校验
    if side/uplo/trans/diag 越域, or m < 0 or n < 0,
       or alpha == nullptr or C == nullptr,
       or lda < max(1, side == LEFT ? m : n) or ldb < max(1, m) or ldc < max(1, m):
        return INVALID_VALUE

    // ④ alpha == 0：A、B 不被引用，C 的 ldc×n 区域整体置 0
    if *alpha == 0:
        zero(C, ldc × n)
        return SUCCESS

    // ⑤ 三角乘加：op(A) 取向由 trans 决定，读取范围只由 uplo / diag 决定
    //    diag=UNIT 时主对角视为 1 且不读取；diag=NON_UNIT 时读取对角（乘法类语义，允许为 0）
    opA = (trans == OP_T) ? Aᵀ : A
    for j in [0, n), i in [0, m):
        C[i][j] = 0
        for k in K(i, j, side, uplo, diag):                 // 有效 k 区间按 uplo 裁剪
            if side == LEFT:
                C[i][j] += opA[i][k] * (alpha * B[k][j])    // alpha 折入被乘元素 B(k,j)
            else:
                C[i][j] += (alpha * B[i][k]) * opA[k][j]    // alpha 折入被乘元素 B(i,k)
    return SUCCESS
```

# 二、需求分析（required）

## 2.1 需求描述

使用 Ascend C / CATLASS 在 arch22（Atlas A2/A3，NPU 架构 `dav-2201`）实现 `aclblasStrmm`：

- 功能语义、参数值域、异常行为对齐任务书 §2.4 与 cuBLAS `cublasStrmm`（唯一差异：`trans` 仅 N/T，`OP_C` 返回 `INVALID_VALUE`）；
- kernel 直调（句柄式接口 + `aclblasSetStream` 绑定的 stream），实现放在 `blas/trmm/arch22/`；
- 精度满足生态算子开源精度标准 FLOAT32 档；性能满足任务书 §3.3 的 5 条达标耗时；
- 自验覆盖任务包提供的 1200 条用例与负向/边界场景，交付设计文档、测试代码、自测报告。

## 2.2 需求拆解

| 编号 | 需求 | 验收口径 |
|---|---|---|
| R1 | 16 组枚举组合（`side×uplo×trans×diag`）功能正确 | 任务包 CSV 中 TC_L0/TC_CV 等成功用例 |
| R2 | 列主序 + 任意合法 `lda/ldb/ldc`（含 padding） | TC_LD（+4）与 TC_EX（+8）padding 用例 |
| R3 | `alpha` 特殊值（0/1/-1/0.5/100/0.001 等）与 `alpha=0` 置零快速路径 | TC_AB / TC_ED 用例 |
| R4 | 空指针、非法枚举、非法前导维、负维度负向语义 | TC_ED 用例（期望 `INVALID_VALUE` / `HANDLE_IS_NULLPTR`） |
| R5 | 零维 no-op（m=0 / n=0） | TC_ED_121~123 |
| R6 | 允许 `B ≡ C`（原地写出） | 用例/自测补充 |
| R7 | 性能：5 条标杆 case 达标（表见 §4.1） | msprof `Task Duration(us)` 平均 ≤ 达标耗时 |
| R8 | 精度：FLOAT32 混合容差 | `verify_accuracy.py` 逐条 PASS |
| R9 | 双款型：Atlas A2 与 Atlas A3 均验收 | 同 arch22 产物分别实跑 |

## 2.3 外部组件依赖

| 依赖 | 用途 | 说明 |
|---|---|---|
| CANN 9.1.0 + Ascend C（asc-devkit） | 编译/运行环境 | 任务书 §3.1；A2/A3 共 `arch22` / `dav-2201` |
| **CATLASS（`3rdparty/catlass` 子模块，头文件依赖）** | TRMM kernel / Block / Tile / 调度 / DeviceGemm 启动适配 | 仅编译期头文件，无动态库依赖；License 与 ops-blas 同源（CANN Open Software License v2.0）；本次升级 pin（§3.2.5） |
| cblas（Netlib BLAS `strmm`） | 精度 golden（**仅测试侧**） | `test/trmm/strmm/strmm_golden.h` 已接入 |
| msprof | 性能采集（**仅自验侧**） | `msprof op --application=...`，读 `OpBasicInfo.csv` 的 `Task Duration(us)` |

算子本体运行无第三方动态库依赖；CATLASS 为 header-only。

## 2.4 内部适配模块

| 模块 | 适配内容 |
|---|---|
| `include/cann_ops_blas.h` | 复用既有 `aclblasStrmm` 声明（不改动） |
| `include/cann_ops_blas_common.h` | 复用 `aclblasStatus_t` 与各枚举；**枚举越域返回 `ACLBLAS_STATUS_INVALID_VALUE`**（按任务书 §2.4 与配套 CSV 期望；不采用 `arch35` README 所述的 `INVALID_ENUM` 口径） |
| `blas/common/helper` | handle（stream 绑定）、工作空间（`EnsureDefaultWorkspace` / `GetEffectiveWorkspace` / `CheckEffectiveWorkspaceSize`，默认 32 MB、可扩容至 2 GB）、核数（`GetAicCoreCount`）、日志宏 |
| `blas/trmm/arch22/`（**本任务新增**） | `strmm_host.cpp`（校验 / quick-return / 工作空间 / tiling / 调度）、`strmm_kernel.cpp`（AIV 规整化 kernel + CATLASS `Trmm` 实例化与启动）、`strmm_kernel.h`（host→kernel launcher 声明）、`strmm_tiling_data.h`（tiling POD） |
| 参考实现（仓内只读） | `blas/trmm/arch35/`（同算子既有实现，仅作 host 组织/文件集参考）、`blas/symm/arch22/`（arch22 工程惯例） |
| 测试框架 | `test/frame/`（CSV 加载/填充/校验/混合容差）、`test/trmm/strmm/strmm_param.h`、`strmm_golden.h`（已存在，复用） |

## 2.5 需求模块设计（与标杆的对齐与差异）

```cpp
aclblasStatus_t aclblasStrmm(
    aclblasHandle_t handle, aclblasSideMode_t side, aclblasFillMode_t uplo, aclblasOperation_t trans,
    aclblasDiagType_t diag, int m, int n, const float* alpha, const float* A, int lda,
    const float* B, int ldb, float* C, int ldc);
```

| 项 | 标杆（cuBLAS / Netlib） | 本算子 | 依据 |
|---|---|---|---|
| 数据类型 | float32 | 仅 float32 | 任务书范围 |
| `trans` 值域 | 参考实现字面接受 `OP_C`（实数档等价 `OP_T`） | **仅 `{OP_N, OP_T}`；`OP_C` → `INVALID_VALUE`** | 任务书 §2.4；CSV `TC_ED_135` |
| 对角语义 | NON_UNIT 读对角（允许 0）；UNIT 视为 1 不读 | 同左 | 任务书 §7 第 3 条 |
| 输出 | Netlib 为原地（覆写 B） | 离席输出到 C；额外允许 `B ≡ C`；C 与 A 不允许重叠 | 任务书 §2.5 |
| 排布 | 列主序 ND | 同左；不支持超出 `ld*/` 语义的非连续访问 | 任务书 §2.5 |
| 零维 | no-op | no-op（优先于其它校验，与提供 golden 一致） | CSV `TC_ED_121~123` |
| 确定性 | 不要求 | 不要求（浮点累加顺序不保证） | 任务书 §2.5 |
| 枚举越域错误码 | — | `ACLBLAS_STATUS_INVALID_VALUE` | 任务书 §2.4 + CSV |

# 三、详细设计（required）

## 3.1 算子分析

### 3.1.1 数学公式

$$
\begin{aligned}
side = LEFT:&\quad C = \alpha \cdot op(A) \cdot B, \quad A \in \mathbb{R}^{m\times m},\ B \in \mathbb{R}^{m\times n},\ C \in \mathbb{R}^{m\times n}\\
side = RIGHT:&\quad C = \alpha \cdot B \cdot op(A), \quad A \in \mathbb{R}^{n\times n},\ B \in \mathbb{R}^{m\times n},\ C \in \mathbb{R}^{m\times n}
\end{aligned}
$$

其中 $op(A)=A$（`trans=N`）或 $A^{T}$（`trans=T`）；A 只存储 `uplo` 指定的三角（含对角），另一三角不被引用；`diag=UNIT` 时 $op(A)$ 的主对角视为 1。

### 3.1.2 支持数据类型

| 数据 | 类型 | 说明 |
|---|---|---|
| alpha / A / B / C | FLOAT32 | 无类型转换；Cube 中间量 FP32（L0C FP32 累加） |
| 计算模式 | **FP32（非 HF32）** | CATLASS `Trmm` 在 fp32 路径显式 `SetHF32Mode(false)`，满足 FLOAT32 精度档要求 |

### 3.1.3 支持形状

- `m, n` 为非负 `int` 运行时入参：`m=0` 或 `n=0` 为合法 no-op；`m<0`/`n<0` 返回 `INVALID_VALUE`；
- A 的阶数 $k_A = m$（`side=LEFT`）/ $n$（`side=RIGHT`）；
- 前导维：$lda \ge \max(1,k_A)$，$ldb \ge \max(1,m)$，$ldc \ge \max(1,m)$；
- 排布：全部列主序 ND；不涉及广播、dynamic shape 机制。

## 3.2 算子实现

### 3.2.1 CATLASS 现状与可复用能力（实现基座）

`ops-blas` 仓已通过 `.gitmodules` 引入 `3rdparty/catlass`（`https://gitcode.com/cann/catlass.git`）。原 pin `db93081c`（v1.6.0）不含 TRMM 实现；本设计将 pin 升级到 `44e75248`（tag `v2.1.0`），其中已包含 CATLASS 的 TRMM 实现（见 §5.2 R1）。CATLASS 侧可复用的能力如下（均已在工作区源码中核实）：

| 组件 | 位置 | 提供的 TRMM 能力 |
|---|---|---|
| TRMM kernel | `include/catlass/gemm/kernel/trmm.hpp` | 统一支持 `side/uplo/trans` 运行时参数；**按输出 tile 裁剪有效 K 范围**（`effectiveUplo = uplo ^ trans`），整块跳过恒零 K 区间；fp32 在 AIC 入口显式 `SetHF32Mode(false)`（全精度 FP32，非 HF32）；`GetWorkspaceSize() == 0` |
| Block / Tile / 调度 | `block_mmad_tla.hpp`、`tile_copy.hpp`（`PackedTileCopyTla`）、`block_swizzle.hpp`（`GemmIdentityBlockSwizzle`）、`dispatch_policy.hpp`（`MmadPingpong`） | AtlasA2（dav-2201）FP32 稠密 GEMM 的分块、流水、搬运与分核调度 |
| 启动适配 | `gemm/device/device_gemm.hpp` | `DeviceGemm<TrmmKernel>` 模板；`Run` 依赖编译期宏 `CATLASS_ARCH`（见 §3.2.5） |
| 参考样例 | `examples/76_trmm/`（`trmm.cpp` / `76_trmm.md`） | tile/调度配置、layout 与 `trans` 的组合方式、多档 tile 变体选择表 |

CATLASS `Trmm` 的两条关键契约（决定了本算子的预处理设计）：

1. **`CanImplement` 拒绝 `diag != 0`**（`trmm.hpp` §`CanImplement`）——`diag=UNIT` 必须由调用方预处理；
2. **kernel 只做 tile 级 K 裁剪，不做逐元素三角 mask，要求未引用三角为 0**（kernel 文件头注释）——A 的未引用三角必须显式置零。

此外，`Trmm` 的写入端在 AtlasA2 上由 `CopyL0CToGmTla` 支持 **RowMajor 与 zN** 两种排布（`gemm/tile/atlasa2/copy_l0c_to_gm.hpp`），因此 L0C→GM 直接写列主序 C 不可用；本设计通过“列主序 ↔ 行主序等价改写（`Cᵀ`）”解决（§3.2.2）。

### 3.2.2 总体方案（四步归一 + CATLASS 稠密 GEMM）

本设计把「三角裁剪 / 对角语义 / 转置取向 / 列主序」全部收敛到一个 AIV 预处理 kernel（`strmm_prepare`，内部按 A、B 两段处理）中，**CATLASS 侧只看到规范化的稠密行主序问题**，从而直接复用 CATLASS `Trmm` 的 K 裁剪与 A2 FP32 GEMM 流水。四步如下：

**第一步：A 侧语义归一（生成 $\tilde A$）**

在工作空间生成 $k_A \times k_A$ 列主序稠密矩阵 $\tilde A$（前导维 $a' = \mathrm{RoundUp}(k_A, 8)$）：

$$
\tilde A(i,j)=
\begin{cases}
0, & (i,j) \notin T(uplo) \quad \text{（未引用存储三角，显式置 0，不读取原 A）}\\
\alpha \cdot A(i,j), & (i,j) \in T(uplo),\ i \neq j \\
\alpha \cdot A(i,i), & i = j,\ diag=NON\_UNIT \\
\alpha, & i = j,\ diag=UNIT \text{（不读取原 A 对角）}
\end{cases}
$$

读取原 A 时按 `lda` 取列；写 $\tilde A$ 时按 $a'$ 紧凑排布（顺带完成前导维压缩与 32B 对齐）。此时恒有：

$$
op(\tilde A) = \alpha \cdot op(A)\ \text{（未引用三角贡献为 0）}
$$

alpha 在此折入，CATLASS 侧 `alpha=1`（CATLASS AIV 分支直接返回，无后处理）。

**第二步：B 侧规整暂存（生成 $\tilde B$）**

在工作空间生成 $m \times n$ 列主序稠密矩阵 $\tilde B$（前导维 $b' = \mathrm{RoundUp}(m, 8)$）：按 `ldb` 逐列拷贝 B 的前 m 个元素，行 $m..b'-1$ 补零。该步骤同时解决三件事：

1. 任意 `ldb` padding 与非 32B 对齐的首地址读入；
2. `B ≡ C` 原地语义（GEMM 全程只读 $\tilde B$，写 C；**无需任何指针检测或 D2D 回拷**）；
3. CATLASS 只需要处理规范紧凑排布。

**第三步：列主序等价改写（Cᵀ 视角）**

列主序矩阵与其“行主序读法”互为转置。取 CATLASS 的输出 $D := C$ 的行主序读法，即 $D = C^{T}$、$D \in \mathbb{R}^{n \times m}$、`RowMajor(ld=ldc)`（C 缓冲区内存不变，直接落盘，无物理转置）：

$$
side=LEFT:\quad D=C^{T}=\tilde B^{T}\cdot op(\tilde A)^{T}
\qquad\qquad
side=RIGHT:\quad D=C^{T}=op(\tilde A)^{T}\cdot \tilde B^{T}
$$

即 CATLASS 的矩阵乘为 $D = L \cdot R$：

| BLAS side | CATLASS `side`（三角在哪一侧） | CATLASS 问题尺寸 $(M',N',K')$ | $L$ | $R$ |
|---|---|---|---|---|
| LEFT | **RIGHT**（三角在右操作数） | $(n,\ m,\ m)$ | $\tilde B^{T}$ | $op(\tilde A)^{T}$ |
| RIGHT | **LEFT**（三角在左操作数） | $(n,\ m,\ n)$ | $op(\tilde A)^{T}$ | $\tilde B^{T}$ |

其中 $op(\tilde A)^{T}$ 始终是 **CATLASS 语义下的三角操作数**，其有效三角为：

$$
T_{cat}=
\begin{cases}
\mathrm{flip}(uplo), & trans=N \quad (\tilde A^{T}\text{ 的三角与存储三角相反})\\
uplo, & trans=T \quad (op(\tilde A)^{T}=\tilde A)
\end{cases}
$$

**第四步：K 裁剪（直接复用 CATLASS `Trmm`）**

$\tilde A$ 的未引用三角已为 0，因此乘积中恒零的 K 块可整块跳过。CATLASS `Trmm` 以 `effectiveUplo = uplo ^ trans` 计算每个输出 tile 的有效 K 区间，本设计传入 `uplo = T_{cat}`、`trans = 0`，即 `effectiveUplo = T_{cat}`，与三角操作数的有效三角一致，裁剪数学上严格等价。有效 MACs 由 $k_A^3$ 降至 $\approx k_A^3/2$。

**映射汇总表（16 组合 → CATLASS 参数，权威口径）**

| 参数 | 取值规则 |
|---|---|
| `problemShape` | $(M',N',K') = (n,\ m,\ k_A)$，其中 $k_A = (side=LEFT)\ ?\ m\ :\ n$ |
| `catlass.side` | `side=LEFT → RIGHT`；`side=RIGHT → LEFT` |
| `catlass.uplo` | `trans=N → flip(uplo)`；`trans=T → uplo` |
| `catlass.trans` | `0` |
| `catlass.diag` | `0`（对角已在 $\tilde A$ 中处理） |
| `catlass.alpha` | `1.0`（alpha 已折入 $\tilde A$） |
| `layoutL` | `side=LEFT`：`RowMajor(rows=n, cols=m, ld=b')`；`side=RIGHT`：`trans=N ? RowMajor(rows=n, cols=n, ld=a') : ColumnMajor(rows=n, cols=n, ld=a')` |
| `layoutR` | `side=LEFT`：`trans=N ? RowMajor(rows=m, cols=m, ld=a') : ColumnMajor(rows=m, cols=m, ld=a')`；`side=RIGHT`：`RowMajor(rows=n, cols=m, ld=b')` |
| `layoutD` | `RowMajor(rows=n, cols=m, ld=ldc)` |

**正确性推导（两个代表例）**

例 1：`side=LEFT, uplo=LOWER, trans=N`。$A$ 下三角有效，$C(i,j)=\sum_k A(i,k)B(k,j)$，对固定 $i$ 有效 $k \in [0, i]$。
映射：CATLASS `side=RIGHT`，$uplo_{cat}=\mathrm{flip}(LOWER)=UPPER$，三角操作数为 $\tilde A^{T}$（下三角有效）。
K 裁剪（CATLASS side=RIGHT & UPPER）：`kEnd = min(nEnd, kEnd)`，其中 $n$ 索引 $N'=m$ 即 BLAS 行号 $i$ → 输出行块 $[i_0, i_0+B_N)$ 的 `kEnd = i_0+B_N`，与 $k \in [0,i]$ 一致。✓

例 2：`side=LEFT, uplo=UPPER, trans=T`。$C(i,j)=\sum_k A^{T}(i,k)B(k,j)=\sum_k A(k,i)B(k,j)$，$A$ 上三角有效 → 有效 $k \in [0, i]$。
映射：CATLASS `side=RIGHT`，$uplo_{cat}=uplo=UPPER$，三角操作数 $op(\tilde A)^{T}=\tilde A$（上三角有效，`ColumnMajor` 读）。
K 裁剪（side=RIGHT & UPPER）：同上 `kEnd=min(nEnd,kEnd)` → 一致。✓

其余组合按同一规则对称成立；K 裁剪只跳过“整块落在零三角”的块，边界块依赖 $\tilde A$ 的零三角保证结果不受原 A 未引用三角数据影响。

### 3.2.3 kernel 侧设计

kernel 族共 **2 个 `__global__` 入口**（不含 CATLASS 内部实现），不按 `side/uplo/trans/diag` 拆分为多个 kernel 入口（主计算段仅按 layout / 三角位置做编译期模板分派，见 §3.2.4）：

| # | kernel | 归属 | 职责 |
|---|---|---|---|
| 1 | `strmm_prepare` | AIV（Vector） | 生成 $\tilde A$（有效三角读入 + alpha 折入 + UNIT 对角覆盖 + ld 压缩；零三角由 host 预清零）与 $\tilde B$（规整暂存） |
| 2 | CATLASS `Trmm`（经 `DeviceGemm` 模板实例化） | AIC + AIV（mix） | 稠密 GEMM `D = L·R`（含 K 裁剪），AIV 分支因 `alpha=1` 直接返回 |

**kernel 1：`strmm_prepare`**

- 纯 Vector kernel，下发块数取 $\min(\text{AIV 核数},\ k_A + n)$；核间按**列**轮转切分（列主序下每列连续）：索引 $0..k_A-1$ 对应 $\tilde A$ 的列、其余对应 $\tilde B$ 的列，同一列只由一个核写，无写竞争（同一 kernel 内两段，省一次下发）；
- host 在启动前对整段工作空间 `aclrtMemsetAsync` 清零；kernel 只写各列的**有效区间**，未引用三角与行尾 padding 由该预清零保证为 0；
- $\tilde A$ 单列处理（列 $j$）：
  - `uplo=UPPER`：有效行区间 $[0, j]$；`uplo=LOWER`：有效行区间 $[j, k_A)$（只由存储三角决定，与 `trans` 无关）；
  - 有效区间按 `lda` 读取原 A 列连续段，乘 alpha（`alpha==1` 时跳过 `Muls`）后写入 $\tilde A$ 的 $a'$ 排布；段首非对齐处使用 `DataCopyPad`；
  - 对角元素：`diag=UNIT` 时以标量写把对角位置覆盖为 `alpha`（结果**不依赖**原对角值，含 NaN）；`NON_UNIT` 时保留读入的 `A[j][j]`；
- $\tilde B$ 单列处理（列 $j$）：按 `ldb` 把前 m 个元素搬到 $\tilde B$ 的 $b'$ 排布。
- 复杂度 $O(k_A^2 + mn)$，全部为连续访存；不查表、不做逐元素三角判定。

**kernel 2：CATLASS `Trmm` 实例化与启动**

- 按 CATLASS 样例 76 的方式实例化：

```cpp
using ArchTag = Arch::AtlasA2;                     // dav-2201（A2/A3 共用）
using DispatchPolicy = Gemm::MmadPingpong<ArchTag, true /*unitFlag*/, false /*useHF32*/>;
using TileCopy = Gemm::Tile::PackedTileCopyTla<ArchTag, float, LayoutTagL, float, LayoutTagR, float, layout::RowMajor>;
using BlockMmad = Gemm::Block::BlockMmadTla<DispatchPolicy, L1TileShape, L0TileShape, float, float, float, void, TileCopy>;
using TrmmKernel = Gemm::Kernel::Trmm<BlockMmad, void, Gemm::Block::GemmIdentityBlockSwizzle<3, 0>>;
using TrmmAdapter = Gemm::Device::DeviceGemm<TrmmKernel>;
```

> 说明：`LayoutTagL`/`LayoutTagR` 为编译期 layout tag，共 4 种组合（`side∈{L,R} × trans∈{N,T}`，见 §3.2.2 映射表），由 host 侧以模板分派方式选取对应实例化；`L1TileShape`/`L0TileShape` 为编译期常量。

- 基线 tile 取 CATLASS 样例默认档：`L1TileShape = <128,128,256>`、`L0TileShape = <128,128,64>`、swizzle `<3,0>`；样例中的小 shape 变体（小 L1-K、`<3,1>` 等）作为后续调优项（§5.1），按“模板实例化 + host 门控”扩展；
- 启动：`trmmOp(stream, aicCoreNum)`，`blockDim = GetAicCoreCount()`（与样例一致，由 CATLASS mix kernel 内部派生 AIV）；
- 由于 `alpha=1`，CATLASS AIV 分支立即返回，无额外后处理 kernel。

**数据流与同步**

```text
A（列主序，ld=lda） ──AIV strmm_prepare──▶ Ã（工作空间，稠密列主序，ld=a'）
B（列主序，ld=ldb） ──AIV strmm_prepare──▶ B̃（工作空间，稠密列主序，ld=b'）
Ã、B̃ ──CATLASS Trmm（AIC，含 K 裁剪）──▶ C（列主序，ld=ldc，输出）
```

- `strmm_prepare` 与 CATLASS kernel 同 stream 顺序下发，GM 数据依赖由 stream 顺序保证（无显式跨 kernel event）；
- CATLASS kernel 内部为 MTE2→MTE1→M(Fixpipe) 的 pingpong 流水，unitFlag 开启（样例同档）；
- kernel 尾部 `PipeBarrier<PIPE_ALL>()`（CATLASS kernel 内已有）保证返回前流水排空。

**输出写入契约**

- 正常路径只写 C 的 $m \times n$ 有效区域（列主序，每列前 m 个元素），padding 行 $[m, ldc)$ 不改写；CATLASS 侧由 block scheduler 的 `actualBlockShape` 限制尾块边界；
- `alpha=0` 路径按提供 golden 的口径（`strmm_golden.h` 使用 `memset(ldc×n)`）在 host 以 `aclrtMemsetAsync` 将 **整个 `ldc×n` 区域** 置 0（与 golden 位可复现一致）；
- FP32 C 的 L0C fractal 为 C0=8；尾块尺寸由 CATLASS `actualBlockShape` 处理，m 非 8 倍数的小 shape 用例（3/5/7/33/65/127）需在自验中重点回归（§5.2 R3）。

### 3.2.4 host 侧设计

host 负责：参数校验 → quick-return → 工作空间规划 → tiling 组装 → 两次 kernel 下发（全部挂 `handle->stream`，host 不搬数、无 D2H/H2D）。

**参数校验（顺序固定，与提供 golden 及配套 CSV 期望一致）**

```
1) m == 0 || n == 0                          → ACLBLAS_STATUS_SUCCESS（合法 no-op，不校验其它参数）
2) handle == nullptr                         → ACLBLAS_STATUS_HANDLE_IS_NULLPTR
3) side/uplo/trans/diag 越域（trans==OP_C 亦在此拦截） → ACLBLAS_STATUS_INVALID_VALUE
4) m < 0 || n < 0                            → ACLBLAS_STATUS_INVALID_VALUE
5) alpha == nullptr || C == nullptr          → ACLBLAS_STATUS_INVALID_VALUE
6) lda < (side==LEFT ? max(1,m) : max(1,n))  → ACLBLAS_STATUS_INVALID_VALUE
   ldb < max(1,m) || ldc < max(1,m)         → ACLBLAS_STATUS_INVALID_VALUE
7) *alpha == 0                               → 置零 C（aclrtMemsetAsync，ldc×n）后返回 SUCCESS
                                               （不引用 A/B，允许 A/B 为 nullptr）
8) A == nullptr || B == nullptr              → ACLBLAS_STATUS_INVALID_VALUE（被引用时）
```

与配套 CSV 的对应：零维 `TC_ED_121~123`（第 1 步）；`TC_ED_127`（第 2 步）；`TC_ED_132~136`（第 3 步，含 `trans=OP_C`）；`TC_ED_141/142`（第 4 步）；`TC_ED_128/131`（第 5 步）；`TC_ED_137~140`（第 6 步）；`TC_ED_125/126`（第 7 步，其中 126 的 A/B 为空指针）；`TC_ED_129/130`（第 8 步）。

**工作空间规划（Device GM，经 handle Effective Workspace）**

| 缓冲 | 形状 / 前导维 | 字节数 | 启用条件 |
|---|---|---|---|
| $\tilde A$ | $a' \times k_A$，$a'=\mathrm{RoundUp}(k_A, 8)$ | $a' \cdot k_A \cdot 4$ | 恒启用 |
| $\tilde B$ | $b' \times n$，$b'=\mathrm{RoundUp}(m, 8)$ | $b' \cdot n \cdot 4$ | 恒启用 |

两段在工作空间内顺序排布、按 32B 对齐（以 float 元素计数）：$\tilde A$ 起始偏移 0，$\tilde B$ 起始偏移 $a' \cdot k_A$（由 host 启动时按该式计算指针，不额外下发偏移字段）。所需字节数由 `EnsureDefaultWorkspace` 申请并校验（默认 32 MB，不足按 2 倍扩容，上限 2 GB）；超过上限返回 `ACLBLAS_STATUS_INVALID_VALUE`，申请失败返回 `ACLBLAS_STATUS_ALLOC_FAILED`。

量级：4096² 用例 $a'=b'=4096$，合计 $2\times 4096^2\times 4\text{B} \approx 134$ MB；1024² 约 8.4 MB；256² 约 0.5 MB。

**tiling 数据（`StrmmTilingData`，单个 POD 随 kernel 入参下发）**

| 组 | 字段 |
|---|---|
| 问题尺寸 | `m`、`n`、`kA`、`lda`、`ldb`、`ldc`、`aStride`（$a'$）、`bStride`（$b'$） |
| 标量 | `alphaVal`（host 读值下发） |
| 预处理判定 | `lowerTriangle`（有效存储三角：1=下、0=上）、`unitDiag`（对角语义：1=UNIT） |
| CATLASS 视角 | `catlassSide`（0=三角在左操作数、1=在右）、`catlassUplo`（有效三角 $T_{cat}$，按 CATLASS 约定 0=lower、1=upper）、`triRowMajor`（三角操作数 layout：1=`RowMajor`、0=`ColumnMajor`） |
| 分核 | `aivCoreNum`、`aicCoreNum` |

**分核策略**

- `strmm_prepare`（AIV）：下发块数 $numBlocks = \min(\text{aivCoreNum},\ k_A + n)$，核 `b` 以步长 `numBlocks` 轮转处理列 `b, b+numBlocks, ...`（前 $k_A$ 列为 $\tilde A$、其余为 $\tilde B$）；同一列只由一个核写，无写竞争；
- CATLASS Trmm（AIC）：由 `GemmIdentityBlockSwizzle` 按输出 tile 分配，`blockDim = aicCoreNum`；有效 K 随输出块变化（三角裁剪）导致的核间不均衡在 v1 不做额外负载均衡，实测不达标再引入（§5.2 R5）。

**tilingKey 规划**

16 组枚举组合在预处理后收敛为**同一主计算形态**，且 CATLASS `Trmm` 的 `side/uplo/trans` 为运行时参数，因此 **v1 不设置运行时 tilingKey**；主计算段按三角操作数的 layout（`RowMajor`/`ColumnMajor`）× 三角位置（左/右）在编译期分派 **4 份 `Trmm` 实例化**，由 host 侧模板分派选取（见 §3.2.3）。若后续启用多档 tile 变体，按“模板实例化 + host 门控”扩展，并同步更新本文档。

### 3.2.5 集成与构建（ops-blas 侧改动）

| 改动 | 内容 |
|---|---|
| 子模块升级 | `3rdparty/catlass` pin 由 `db93081c`（v1.6.0，不含 `trmm.hpp`）升级到 `44e75248`（tag `v2.1.0`，已验证含该文件）；PR 中带出子模块指针变更（`git add 3rdparty/catlass`） |
| include 路径 | 根 `CMakeLists.txt` 的 `${OPS_BLAS}` 增加 `3rdparty/catlass/include`（CATLASS 头文件以 `#include "catlass/..."`、`#include "tla/..."` 引用） |
| 编译宏 | `${OPS_BLAS}` 的 ASC 编译增加 `-DCATLASS_ARCH=2201`。**注意**：`DeviceGemm::Run` 的启动体由 `#if (defined(CATLASS_ARCH) && CATLASS_ARCH == 2201)` 守卫，未定义该宏时函数体为空、kernel 静默不启动（`device_gemm.hpp` 已核实） |
| C++ 标准 | `${OPS_BLAS}` 增加 `cxx_std_17`（CATLASS 使用 `if constexpr` 等 C++17 特性；blasLt 目标已有该设置，blas 目标需补充） |
| 隔离性 | CATLASS 头文件只在 `blas/trmm/arch22/strmm_kernel.cpp` 中引入；host 文件（`strmm_host.cpp`）保持不依赖 CATLASS，避免宏/命名空间与 CANN 头文件相互污染 |
| README | 更新 `blas/trmm/README.md` 产品支持表：Atlas A2/A3 系列（含 Atlas 800I A2/A3）“支持”，并补充 arch22 的约束说明 |

## 3.3 支持硬件

| 支持的芯片版本 | 涉及勾选 |
|---|---|
| Atlas A2 训练系列产品（Atlas 800T A2，910B3） | √（性能测试设备，自验） |
| Atlas A3 训练/推理系列产品（Atlas 800I A3） | √（需实机验收） |

A2/A3 在 ops-blas 构建中同映射 `arch22` / `NPU_ARCH=dav-2201`，同一份 `ops_blas` 产物覆盖两款型；CATLASS 侧对应 `Arch::AtlasA2` + `CATLASS_ARCH=2201`。

## 3.4 算子约束限制

| 约束项 | 内容 |
|---|---|
| 数据类型 | 仅 float32 |
| 计算模式 | FP32 全精度（HF32 关闭）；不支持 HF32 加速档 |
| 排布 | 仅列主序 ND；不支持超出 `lda/ldb/ldc` 语义的非连续访问 |
| `trans` | 仅 `OP_N/OP_T`；`OP_C` 返回 `INVALID_VALUE` |
| `diag` | `UNIT` 时 A 对角视为 1、其值不影响结果（可为任意值，含 NaN） |
| 输出 | 离席写 C；允许 `B ≡ C`；C 与 A 不允许重叠；正常路径不改写 `ldc` padding 行 |
| 批量/stride | 不支持 batched / strided 变体 |
| beta | 不支持（TRMM 无 `beta*C` 项） |
| 零维 | `m=0` 或 `n=0` 合法 no-op（不校验后续参数） |
| 工作空间 | 依赖 handle 工作空间（4096² 约 134 MB；默认 32 MB 自动扩容） |
| CATLASS 依赖 | 算子编译依赖 `3rdparty/catlass`（含 `trmm.hpp` 的版本）；CATLASS 要求“未引用三角已置零、`diag=0`”，由本算子预处理满足 |

# 四、可维可测分析

## 4.1 精度标准 / 性能标准

| 验收标准 | 描述 | 标准来源 |
|---|---|---|
| 精度标准 | 与 cblas（Netlib `strmm`）golden 比对，FLOAT32 混合容差：$rtol=atol=2^{-13}$，逐元素 $\lvert actual-golden\rvert \le atol + rtol\cdot\lvert golden\rvert$；`matched_ratio ≥ 0.99` 且 `max_abs_error ≤ 1e-2`（或 32×ULP） | 任务书 §3.2；生态算子开源精度标准 |
| 性能标准 | 5 条标杆 case 的 NPU 平均单次 kernel 耗时 ≤ `gpu_ms / 0.8` | 任务书 §3.3；`gpu_baseline.csv` |
| 内存标准 | 任务书标注“不涉及”；自测报告按工作空间解析式填报实际占用 | 任务书 §3.4 |
| 确定性 | 不要求逐位一致 | 任务书 §2.1 第 6 条 |

标杆性能目标（`gpu_ms / 0.8`）：

| case | m | n | side | uplo | trans | diag | gpu_ms | 达标耗时（us） |
|---|---|---|---|---|---|---|---|---|
| 1 | 256 | 256 | LEFT | UPPER | N | NON_UNIT | 0.046743 | 58.43 |
| 2 | 512 | 512 | LEFT | LOWER | T | NON_UNIT | 0.110553 | 138.19 |
| 3 | 1024 | 1024 | LEFT | UPPER | N | UNIT | 0.273768 | 342.21 |
| 4 | 2048 | 2048 | RIGHT | LOWER | N | NON_UNIT | 0.945616 | 1182.02 |
| 5 | 4096 | 4096 | LEFT | UPPER | T | NON_UNIT | 5.345090 | 6681.36 |

## 4.2 兼容性分析

- 新算子（arch22 为本仓首版），不涉及既有接口兼容性变更；`include/cann_ops_blas.h` 声明不变；
- 唯一语义偏离（`trans=OP_C` 拒绝）依据任务书 §2.4，且不影响 `OP_N/OP_T` 调用方；
- 与 `blas/trmm/arch35`（Ascend 950PR）产品支持范围互补，同接口多产品线共用；
- 新增对 `3rdparty/catlass` 的编译期依赖：不影响 `ops_blas` 动态库的 ABI 与对外符号，仅影响本仓构建（需子模块升级 + include 路径 + `CATLASS_ARCH` 宏）。

## 4.3 测试设计（可测性）

- **测试工程**：复用仓内 `test/frame`（CSV 加载、填充、混合容差、状态码解析）与已存在的 `test/trmm/strmm/strmm_param.h` / `strmm_golden.h` / `CMakeLists.txt`（`ops_blas_add_gtest_tests`，按 SOC 架构目录自动纳入 arch22 用例）；新增 `test/trmm/strmm/arch22/strmm_test.cpp`，并将任务包 `strmm_test.csv` 落到 `test/trmm/strmm/arch22/strmm_test.csv`；
- **用例覆盖**（任务包 1200 条 + 必要补充，前缀与任务包分类一致）：
  - TC_L0 16 条：16 组枚举全组合（8×8）；TC_SQ 23 条：尺寸扫描（1→2048，含 1/3/5/7 等边界与非对齐值）；TC_AB 16 条：alpha 特殊值（0/1/−1/0.5/−0.5/2.5/100/0.001）；TC_WS/TH 24 条：非方阵（fat/thin）；TC_LD 12 条：前导维 +4 padding（尺寸 16/32/64）；TC_FL 13 条：填充（A：均匀/交替/极端/NON_UNIT 全零/UNIT 全零；B：均匀/全零/交替/极端/Inf/NaN）；TC_CV 16 条：中等尺寸枚举覆盖；TC_ED 23 条：边界负向（零维、alpha=0、空指针、非法枚举、非法前导维、负维度）；TC_EX 857 条：尺寸×枚举×标量×padding 的确定性扩展采样；TC_PF 200 条：性能用例（前 5 条与任务书 §3.3 标杆 case 逐参数一致）；
  - 补充用例：`B ≡ C` 原地写出（含 `ldb == ldc` 与 `ldb != ldc`）、`ldc` padding 行不被改写、A 未引用三角填 NaN/Inf（验证“不引用”语义）、m 非 8 倍数的小 shape 尾块；
- **性能**：`msprof op --application=./build/test/.../strmm_test --gtest_filter=*TC_PF*`，取 `OpBasicInfo.csv` 各 kernel `Task Duration(us)`，单用例独立进程/过滤，warmup 后平均 >10 次；
- **负向判定**：以 CSV `expect_result` 为准（1184 SUCCESS、15 INVALID_VALUE、1 HANDLE_IS_NULLPTR）。

# 五、性能预估与风险分析

## 5.1 性能设计要点与预估

（以下为模型估算与设计决策，**实测数据在 M2 阶段回填**。）

- **算力需求**：K 裁剪后有效 MACs ≈ $\frac{1}{2}k_A^2\cdot n$（side=LEFT 为 $\frac{1}{2}m^2n$；side=RIGHT 为 $\frac{1}{2}mn^2$）。case 5（4096²）≈ 34.4 GMACs（按 2 FLOP/MAC 折合约 68.7 GFLOP）；要在 6681 us 内完成主体计算，等效需要 ≥ $68.7/6.68 \approx 10.3$ TFLOP/s 的有效算力，达标余量以 910B3 实测有效算力为准（M2 回填），因此 v1 先以基线 tile 实测，并将三角裁剪下的核间负载均衡（§5.2 R5）列为达标必要项；
- **预处理开销**：$O(k_A^2 + mn)$ 连续访存。4096² 下 $\tilde A$（读有效半三角、写全量）与 $\tilde B$（全量读写）合计约 230 MB，另有整段工作空间预清零约 134 MB；按 HBM ~1.5 TB/s 估算合计约 240 us（< 4% 总预算）；256² 约 1 MB，主要由 kernel 下发开销决定（< 5~8 us），case 1 预算 58.4 us 充足；
- **kernel 数**：仅 2 次下发（prepare + GEMM），比“A/B/mirror/copyout 四段式”少 2 次启动开销，对小 shape（case 1）更有利；
- **tile 选型**：基线 `<128,128,256>`/`<128,128,64>`/`<3,0>`；若小 shape 或窄矩阵（M'=n 或 N'=m 很小时）出现填核不足，按 CATLASS 样例 76 的变体表（小 L1-K、`<3,1>`/`<4,1>` 等）增加档位，并按“模板实例化 + host 门控”扩展（§3.2.4）；
- **进一步优化项（按需）**：① 在 `ldb` 满足对齐且 `B != C` 时跳过 $\tilde B$ 暂存（省 O(mn) 带宽）；② 将 $\tilde A$ 归一化与 GEMM 做列块流水（提高异构重叠）。两者均以 5 条标杆实测余量不足为前提再启用。

## 5.2 风险与缓解

| 编号 | 风险 | 影响 | 缓解措施 |
|---|---|---|---|
| R1 | CATLASS 子模块原 pin（`db93081c`，v1.6.0）**不含** `trmm.hpp` | 阻塞实现与合入 | 本设计将 pin 升级到含该 kernel 的 `44e75248`（tag `v2.1.0`，已验证包含），PR 中带出子模块指针变更；设计文档与 README 同步声明依赖版本 |
| R2 | 构建集成：ASC 编译 `CATLASS_ARCH` 未定义时 `DeviceGemm::Run` 静默为空；catlass 与 CANN 头文件宏/命名空间潜在冲突 | kernel 不启动或编译失败（隐蔽） | ① 断点/单测校验 kernel 实际下发（下发计数或 msprof）；② CATLASS 仅在一个 TU 引入；③ 编译期 `static_assert`/`#ifndef CATLASS_ARCH #error` 防御 |
| R3 | FP32 L0C fractal（C0=8）与 m 非 8 倍数尾块的 Fixpipe 写回边界 | 尾块越界写/精度失败 | 以 CATLASS `actualBlockShape` 限制边界；自验重点回归 3/5/7/33/65/127 尺寸；若 Fixpipe 对非对齐 `nSize` 有限制，退化为“scratch tile + AIV 回写” |
| R4 | 任意 `lda/ldb/ldc`（含 padding）与非对齐首地址 | 结果错误/越界 | 通过 $\tilde A/\tilde B$ 规整暂存消除前缀维影响；C 仅按 `RowMajor(ld=ldc)` 落盘；TC_LD 与自测 padding 用例回归；`ldc` padding 行不写 |
| R5 | 三角裁剪导致核间负载不均（UPPER 情形有效 K 随行号线性递减） | 大 shape 性能不达标 | 先用 CATLASS swizzle 默认档实测；不达标依次引入多 tile 变体、swizzle 调优、（必要时）工作量配额分配/StreamK |
| R6 | `Inf/NaN` 填充与 `diag=UNIT` 对角语义 | 边界用例失败 | 归一化 kernel 对 UNIT 对角做覆盖写（不依赖原对角值）；未引用三角由 host 预清零保证为 0；Inf/NaN 用例按框架判定回归 |
| R7 | `B ≡ C` 原地语义 | 结果被覆盖 | 无条件规整暂存 $\tilde B$，GEMM 只读暂存；自测补原地用例 |
| R8 | 评审方对“为单算子引入子模块依赖”的接受度 | 合入受阻 | 依赖为仓内既有子模块（本次仅升级 pin + include 路径），License 同源；文档 §3.2.1/§3.2.5 明确复用边界；若被否决，回退方案为：保留预处理与 K 裁剪算法层，用 Ascend C 高阶 Matmul API（参考 `blas/symm/arch22`）替换 CATLASS 主计算段 |
| R9 | A3 款型仅同 `arch22` 产物验证、无独立实现 | 双款型验收 | 同一 `dav-2201` 产物在 A2/A3 分别实跑 5 条标杆与全量精度；性能在 910B3 采集，A3 记录参考值 |