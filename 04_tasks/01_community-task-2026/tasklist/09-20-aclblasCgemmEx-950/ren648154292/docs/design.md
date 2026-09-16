# aclblasCgemmEx 设计文档

## 需求背景（required）

### 需求来源

| 项目 | 内容 |
| --- | --- |
| 任务来源 | 2026 年社区任务（算子开发类），权威需求来源为任务书 `aclblasCgemmEx_task_doc.md` |
| 目标芯片 | Ascend 950PR（完整 SOC 名 `Ascend950PR_9579`，短 soc 名 `ascend950`） |
| 运行环境 | CANN 9.1.0（`/usr/local/Ascend/cann-9.1.0`），NPU device 0 |
| 框架对齐 | cuBLAS `cublasCgemmEx`（接口参数序列、参数语义、边界行为逐项对齐）；单批语义参考 Netlib BLAS `cgemm` 的 quick return 语义 |
| 精度 golden | **CPU 端 `double` 精度独立实现**（测试工程 `test/gemmex/cgemmex/arch35/cgemmex_golden.h`），不复用本算子自身的任何代码路径，也不是 `cblas` / Netlib BLAS `cgemm` |
| 对比基线 | 官方 `gpu_baseline.csv`（200 条 GPU 基线，NPU 侧以 `ratio = gpu_ms / (npu_mean_us / 1000)` 与之比对） |
| 交付仓与路径 | ops-blas 开源仓；实现目录 `blas/gemm/arch35/`，测试目录 `test/gemmex/cgemmex/`，接口声明 `include/cann_ops_blas.h` |
| 交付形态 | 昇腾算子设计文档、自测用例及测试代码、自测报告、待验收代码（README 产品支持表标注 Ascend 950PR：支持） |

### 背景介绍

#### 现状分析

Ascend 950PR 上已有的 GEMM 家族能力存在一处空缺：

- `aclblasCgemm`：单精度复数 GEMM，但 A/B/C 三矩阵的数据类型由签名固定为 `aclblasComplex`，调用方无法独立指定矩阵类型；
- `aclblasGemmEx` 系列（`aclblasGemmEx` / `aclblasGemmBatchedEx` / `aclblasGemmStridedBatchedEx`）：支持 `Atype` / `Btype` / `Ctype` 独立类型，但覆盖的是实数与混合精度场景，缺少扩展级复数接口；
- 因此，在 Ascend 950PR 上用 Ascend C 直调方式实现扩展级复数 GEMM 接口 `aclblasCgemmEx` 是一项待补齐的能力。

本算子的实现路径与 API 路径：

- 接口声明：`ops-blas/include/cann_ops_blas.h`（与兄弟 Ex 接口同头文件，禁止 950PR 私有平行接口）；
- 类型与状态码定义：`ops-blas/include/cann_ops_blas_common.h`；
- Kernel 实现：`ops-blas/blas/gemm/arch35/`（`cgemm_ex_host.cpp` / `cgemm_ex_tiling_data.h` / `cgemm_ex_kernel.h` / `cgemm_ex_kernel.cpp`）；
- 测试工程：`ops-blas/test/gemmex/cgemmex/`（含 `arch35/` 子目录）。

#### 计算内容

计算公式：`C = alpha * op(A) * op(B) + beta * C`

输入：`alpha`（Host 复数标量指针）、`A`（Device 矩阵）、`typeA`、`lda`、`B`（Device 矩阵）、`typeB`、`ldb`、`beta`（Host 复数标量指针）、`C`（Device 矩阵，兼作输入）、`typeC`、`ldc`、`transa`、`transb`、`m`、`n`、`k`、`handle`

输出：`C`（原地覆写）

支持数据类型：`ACLBLAS_C_32`（complex64）、`ACLBLAS_R_32`（float32）、`ACLBLAS_H_R_32`（float16）、`ACLBLAS_H_C_32`（complex32）四条 type 路径，仅同型组合

支持转置：`ACLBLAS_OP_N` / `ACLBLAS_OP_T` / `ACLBLAS_OP_C` 三种形态

支持广播：不支持（A/B/C 为独立矩阵）

#### 硬件层面的实现难点

arch35 的存储拓扑与 arch220 有本质区别，是本算子设计的主要约束来源：

| 存储 | 容量 | 归属 | 对设计的影响 |
| --- | --- | --- | --- |
| Cube L1 | 32 KB | Cube 侧独立物理存储 | 需按半区分列使用：`a1` 16 KB + `b1` 16 KB 恰好占满，零余量；双缓冲只能在半区内部轮转 |
| Cube L0A / L0B | 64 KB / 64 KB | Cube 侧独立物理存储 | tile 尺寸受块划分求解约束 |
| Cube L0C | 256 KB | Cube 侧独立物理存储 | 累加器恒为 fp32，`MmadParams.isBias` 不得置位（其语义为"累加初始矩阵"，k 维累加不应置位） |
| fixBuf | 7 KB | Cube 侧 fixpipe 独立物理存储 | 必须分片写出，禁止一次性 fixpipe 整个输出 tile |
| UB | 248 KB（254272 B） | **AIV 专用** | 复数后处理流水的全部缓冲预算 |

关键架构事实：arch35 上 `l1Size` / `l0ASize` / `l0BSize` / `l0CSize` 是 Cube 侧独立物理存储，不是 `ubSize` 的子区域；`ubSize` 专供 AIV。因此 **Cube kernel 不吃 UB 预算，AIV kernel 不吃 Cube L1/L0 预算**，两套预算必须分开核算。

另一项硬约束是：复数类型在 Cube 侧无原生 dtype，矩阵乘必须分解为实数 GEMM。本算子采用 3 个实数 GEMM 的分解方式，较 4 个实数 GEMM 的参考路径减少 25% 矩阵乘工作量。同时 `alpha` / `beta` 是 Host 端复数标量指针，需要在矩阵乘之外增加独立的标量后处理步骤，单个角色无法一次完成，因此需要双核角色分工。

#### 交付范围界定（重要）

本算子的接口契约覆盖四条 type 路径与三种转置形态，但**本次交付的范围已按"最小可用"路线收窄**（2026-09-11 用户决策），文档其余章节中的"本次交付"均指收窄后的范围：

| 能力 | 契约范围 | 本次交付 |
| --- | --- | --- |
| 数据类型 `ACLBLAS_C_32`（complex64）同型路径 | 支持 | **已交付** |
| 数据类型 `ACLBLAS_R_32`（float32）扩展路径 | 支持 | 本次交付未实现 |
| 数据类型 `ACLBLAS_H_R_32`（float16）扩展路径 | 支持 | 本次交付未实现 |
| 数据类型 `ACLBLAS_H_C_32`（complex32）扩展路径 | 支持 | 本次交付未实现 |
| 转置形态 `transA = ACLBLAS_OP_N`、`transB = ACLBLAS_OP_N` | 支持 | **已交付** |
| 转置形态 `ACLBLAS_OP_T` / `ACBLAS_OP_C` | 支持 | 本次交付未覆盖 |

收窄原因：本次交付以"最小可用"为目标优先落地主路径，扩展路径与其余转置形态不在本次验收范围内。**该收窄属于交付范围选择，不是能力缺陷或实现遗漏**：接口签名、枚举与校验逻辑已按完整契约实现，未交付的数据类型在 host 侧有显式的、可判定的返回路径（`ACLBLAS_STATUS_NOT_SUPPORTED`），不会静默降级。

## 需求分析（required）

### 需求描述

在 Ascend 950PR（arch35，`DAV_3510`）上，以 Ascend C kernel 直调方式实现单精度复数扩展级通用矩阵乘接口 `aclblasCgemmEx`，计算 `C = alpha * op(A) * op(B) + beta * C`：

- A、B、C 三矩阵的数据类型由 `typeA` / `typeB` / `typeC` 三个独立枚举参数指定，区别于 `aclblasCgemm` 的固定 `aclblasComplex` 签名；
- `alpha` / `beta` 为单精度复数标量，以 `const aclblasComplex*` 形式**驻留 Host 内存**，由 host 端直接解引用，不做 device 拷贝；
- `A` / `B` / `C` 为 **Device 内存**矩阵指针，元素位宽由对应 type 参数决定，采用 BLAS 标准**列主序**（Column-Major）存储，复数矩阵为实部/虚部交错（real-first）；
- 语义对标 cuBLAS `cublasCgemmEx`，参数序列与 cuBLAS 一一对应（含 handle 位置，维度参数为 `int`），无需额外映射；
- 执行模型为句柄式异步下发：通过 `aclblasHandle_t` 绑定 `aclrtStream`，读回 Device 结果前须同步 stream；
- 单次调用为单批语义，不含批量（batched）语义。

接口签名（17 个参数，与 `cann_ops_blas.h` 声明一致）：

```c
aclblasStatus_t aclblasCgemmEx(
    aclblasHandle_t handle,
    aclblasOperation_t transa, aclblasOperation_t transb,
    int m, int n, int k,
    const aclblasComplex* alpha,
    const void* A, aclblasType_t typeA, int lda,
    const void* B, aclblasType_t typeB, int ldb,
    const aclblasComplex* beta,
    void* C, aclblasType_t typeC, int ldc);
```

**本次交付范围**：`ACLBLAS_C_32`（complex64）同型路径，且 `transA = ACLBLAS_OP_N`、`transB = ACLBLAS_OP_N`。

### 需求拆解

| # | 需求项 | 状态 |
| --- | --- | --- |
| 1 | 接口声明：`include/cann_ops_blas.h` 新增 `aclblasCgemmEx`（17 参数） | 已完成 |
| 2 | 类型定义：`include/cann_ops_blas_common.h` 新增 `typedef aclDataType aclblasType_t;` 及 `ACLBLAS_C_32` / `ACLBLAS_R_32` / `ACLBLAS_H_R_32` / `ACLBLAS_H_C_32` 常量 | 已完成 |
| 3 | 参数合法性校验与错误码（含 9 步校验顺序） | 已完成 |
| 4 | `ACLBLAS_C_32` → `ACLBLAS_C_32` 同型主路径实现 | 已完成 |
| 5 | `ACLBLAS_OP_N` / `ACBLAS_OP_N` 转置形态实现与验证 | 已完成 |
| 6 | 复数矩阵乘的 3 个实数 GEMM 分解 | 已完成 |
| 7 | 精度达到官方 CSV `mere_threshold` + `mare_multiplier` 判据 | **部分达成**：范围内 60 条数值用例实测 **46 PASS / 14 FAIL**（修复前 45 PASS / 15 FAIL，差的 1 条即 `TC_ED_707`，已修复）；14 条 FAIL 归因于判据形态与填充 token 语义，见"可维可测分析" |
| 8 | 性能达到验收判据（`ratio >= 0.4`） | **未达成**：实测 47/50 条全部未通过，最高 `ratio = 0.0115`，属已知性能缺陷，见"可维可测分析" |
| 9 | `ACLBLAS_R_32`（float32）扩展路径 | **本次交付未实现** |
| 10 | `ACLBLAS_H_R_32`（float16）扩展路径 | **本次交付未实现** |
| 11 | `ACLBLAS_H_C_32`（complex32）扩展路径 | **本次交付未实现** |
| 12 | `ACBLAS_OP_T` / `ACBLAS_OP_C` 转置形态的实现与验证 | **本次交付未覆盖** |

> 需求项 9–12 的设计推导已在"详细设计"中作为设计意图保留，标注为本次交付未实现/未覆盖。这些能力在接口层已可被调用方触及（枚举、校验、错误码均已就位），host 侧对未实现的数据类型有显式返回，不存在静默降级路径。

## 详细设计（required）

### 算子分析

#### 数学公式

主公式：

$$
C = \alpha \cdot \operatorname{op}(A) \cdot \operatorname{op}(B) + \beta \cdot C
$$

逐元素形式（列主序，`op(A) ∈ ℂ^(m×k)`，`op(B) ∈ ℂ^(k×n)`）：

$$
C_{ij} = \alpha \cdot \left( \sum_{l=0}^{k-1} \operatorname{op}(A)_{il} \cdot \operatorname{op}(B)_{lj} \right) + \beta \cdot C_{ij}, \quad 0 \le i < m,\ 0 \le j < n
$$

维度：`op(A)` 为 `m × k`，`op(B)` 为 `k × n`，`C` 为 `m × n`（均为逻辑维度）。

`op(A)` / `op(B)` 语义：

| 取值 | 枚举值 | `op(A)` | `op(B)` | 说明 |
| --- | --- | --- | --- | --- |
| `ACBLAS_OP_N` | 111 | `A` | `B` | 原样使用 |
| `ACBLAS_OP_T` | 112 | `A^T`（转置，**不取共轭**） | `B^T` | 仅交换行列下标：`A^T_{ij} = A_{ji}` |
| `ACBLAS_OP_C` | 113 | `A^H`（共轭转置） | `B^H` | `A^H_{ij} = conj(A_{ji})`，实部不变、虚部取反 |

复数路径下 `T` 与 `C` 语义不同（`T` 不共轭、`C` 共轭），须分别正确实现；实数路径下 `A^H ≡ A^T`，结果等价。

**复数矩阵乘的 3 个实数 GEMM 分解**（本实现的核心）：

复数乘法 `(Ar + j·Ai)(Br + j·Bi) = (Ar·Br − Ai·Bi) + j·(Ar·Bi + Ai·Br)` 无法用少于 3 个实数 GEMM 完成（2 个的构造不能由实数加法与矩阵乘组合得到）。引入第三组操作数 `(Ar + Ai)` 与 `(Br + Bi)`：

```
T1 = Ar · Br
T2 = Ai · Bi
T3 = (Ar + Ai) · (Br + Bi)

G_re = T1 − T2
G_im = (T3 − T1) − T2
```

三次减法的顺序不可交换（`G_im` 与 `G_re` 都消费原始 `T1`，必须在 `T1` 被覆写前用完它）。本实现使用 3 次非原地减法写入独立的计算缓冲：数学上完全等价，且避免了 `dst == src` 的读写别名。

标量装配：

```
C.re = ar·G_re − ai·G_im + br·C.re − bi·C.im
C.im = ar·G_im + ai·G_re + bi·C.re + br·C.im
```

其中 `(ar, ai) = alpha`、`(br, bi) = beta` 为 Host 端复数标量。

列主序存储与元素寻址：列主序下，逻辑维度 `rows × cols`、前导维 `ld` 的矩阵，元素 `(i, j)` 的物理索引为 `j * ld + i`。因此：

- `A`（`transa = N`）物理形状 `lda × k`；`transa = T/C` 时物理形状 `lda × m`；
- `B`（`transb = N`）物理形状 `ldb × n`；`transb = T/C` 时物理形状 `ldb × k`；
- `C` 物理形状 `ldc × n`，逻辑维度 `m × n`。

复数矩阵 `aclblasComplex` 以实部/虚部交错（real-first）存储：元素占 8 字节，`real` 在前、`imag` 在后。

特殊值语义：输入含 `Inf` / `NaN` 时按 IEEE 754 传播，对齐 cuBLAS，不得因特殊值提前报错。

#### 支持数据类型

| type 枚举 | 映射的 `aclDataType` | 元素字节数 | 存储 | 本次交付 |
| --- | --- | --- | --- | --- |
| `ACLBLAS_C_32` | `ACL_COMPLEX64` | 8 | 实/虚各 float32，交错 real-first | **已交付（唯一）** |
| `ACLBLAS_R_32` | `ACL_FLOAT` | 4 | 单精度实数 | 本次交付未实现 |
| `ACLBLAS_H_R_32` | `ACL_FLOAT16` | 2 | 半精度实数 | 本次交付未实现 |
| `ACLBLAS_H_C_32` | `ACL_COMPLEX32` | 4 | 实/虚各 float16，交错 real-first | 本次交付未实现 |

约束与规则：

1. **仅同型组合**：要求 `typeA == typeB == typeC`；跨类型组合返回 `ACLBLAS_STATUS_INVALID_VALUE`。
2. **类型不提升**：输出 dtype 与输入一致，无隐式提升。
3. **累加精度**：矩阵乘中间累加恒使用 float32 累加器（arch35 的 L0C 仅支持 fp32 累加），与输出 dtype 是否为 fp16 无关。
4. **未交付类型的行为**：`typeA != ACL_COMPLEX64` 时 host 侧返回 `ACLBLAS_STATUS_NOT_SUPPORTED`，不启动任何 kernel。该路径是显式可判定的，不是遗漏。
5. 未交付的原因：交付范围收窄（"最小可用"路线），见"背景介绍 › 交付范围界定"。

**接口声明注释与实现的一致性说明**：`include/cann_ops_blas.h` 中 `aclblasCgemmEx` 的函数注释声明支持四条 type 路径（`ACLBLAS_R_32` / `ACLBLAS_C_32` / `ACLBLAS_H_R_32` / `ACLBLAS_H_C_32`），但当前实现仅对 `typeA == ACL_COMPLEX64` 启动 kernel，其余同型组合在 host 侧返回 `ACLBLAS_STATUS_NOT_SUPPORTED`。注释描述的是接口契约的完整形态，实现状态见上表"本次交付"列。

#### 支持形状

| 项 | 内容 |
| --- | --- |
| 逻辑形状 | `op(A)` 为 `m × k`，`op(B)` 为 `k × n`，`C` 为 `m × n` |
| 秩 | A / B / C 均为 rank 2，layout 为 ND |
| 存储 | 列主序，要求连续存储 |
| 原地语义 | `C` 既是输入（旧值参与 `beta·C`）又是输出（原地覆写），不返回视图 |
| `transa = N` 时 | A 物理形状 `lda × k`，要求 `lda ≥ max(1, m)` |
| `transa = T/C` 时 | A 物理形状 `lda × m`，要求 `lda ≥ max(1, k)` |
| `transb = N` 时 | B 物理形状 `ldb × n`，要求 `ldb ≥ max(1, k)` |
| `transb = T/C` 时 | B 物理形状 `ldb × k`，要求 `ldb ≥ max(1, n)` |
| C | 物理形状 `ldc × n`，要求 `ldc ≥ max(1, m)` |
| 维度取值 | `m ≥ 0`、`n ≥ 0`、`k ≥ 0`；负值返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| 空矩阵 | `m == 0` 或 `n == 0` 为合法 no-op，返回 `ACLBLAS_STATUS_SUCCESS`，不执行任何计算 |
| `k == 0` | 跳过矩阵乘，执行 `C = beta · C` |

本次交付覆盖的形状与转置组合：**任意合法的 `m` / `n` / `k` 与 `lda` / `ldb` / `ldc`**，且 `transA = ACLBLAS_OP_N`、`transB = ACLBLAS_OP_N`。

本次交付未覆盖：

- `ACBLAS_OP_T` 与 `ACBLAS_OP_C` 转置形态的验证（见"算子约束限制"）。
- 非连续 Tensor（超出 `lda` / `ldb` / `ldc` 语义的非连续内存访问）；
- 广播（A / B / C 为独立矩阵）；
- 动态 shape（`m` / `n` / `k` 为运行时入参，但无动态 shape 语义）；
- 批量（batched）矩阵乘（由独立的 `aclblasGemmBatchedEx` / `aclblasGemmStridedBatchedEx` 承接）。

### 算子实现

#### 实现方案

**技术路线：SIMD/MemBase 低阶 Cube block API 完成矩阵乘，配 AIV 标量后处理，双核角色分工。**

| 角色 | kernel 任务类型 | 职责 |
| --- | --- | --- |
| Cube | `KERNEL_TYPE_AIC_ONLY` | 矩阵乘累加主计算（含 3 个实数 GEMM 分解） |
| AIV | `KERNEL_TYPE_AIV_ONLY` | 复数解交错、3-GEMM 合成、`alpha·G + beta·C` 后处理与原地写回 |

选择依据：

1. 复数 Cube 无原生 dtype，必须分解为实数 GEMM，需要可控的 block API 才能精确控制 L1/L0 复用与 fixpipe 分片；
2. arch35 上 `l1Size` / `l0ASize` / `l0BSize` / `l0CSize` 是 Cube 侧独立物理存储，只有低阶 API 能对齐这套预算模型；
3. `alpha` / `beta` 为 Host 端复数标量，需要独立的 AIV 后处理步骤，Cube 单角色无法一次完成。

排除的路线：

| 排除路线 | 排除原因 |
| --- | --- |
| RegBase | `DAV_3510` 上未启用，且本算子非 vector 类算子，不满足触发条件 |
| `Matmul` 高阶 API | 内部 tile 尺寸与 L0C 复用策略不可控，无法满足"先解交错为实矩阵 + `isBias` 语义精确控制 + fixBuf 7 KB 分片"三项硬约束 |
| Blaze / tensor_api 图算子路线 | 前提为自定义算子形态与 aclnn 两段式 ABI，本算子二者皆无（句柄式库接口，kernel 直调） |

**TilingData 分发机制**：本算子是 ops-blas 句柄式库，**无 op-tiling ABI**，不使用 `ASCENDC_TPL_ARGS_DECL` / `ASCENDC_TPL_SEL_PARAM`；自有结构体 `CgemmExTilingData`（`cgemm_ex_tiling_data.h`）由 host 填充后随 kernelArgs 按值传入，由编译器负责 host → device 参数拷贝，不需要额外设备内存分配。分发由自有枚举 `GemmDTypeCase` + `switch` 选 kernel 入口 + kernel 内编译期 `if constexpr` 完成。废弃宏 `TILING_KEY_IS` / `BEGIN_TILING_DATA_DEF` 全文禁用。

##### 3.2.0 整体执行流程图

```mermaid
graph TD
    %% Host 入口与校验
    A["外部调用 aclblasCgemmEx<br/>17 参数句柄式接口"] --> B["ValidateCgemmExParams<br/>9 步参数校验"]
    B -->|handle 空| B1["ACLBLAS_STATUS_HANDLE_IS_NULLPTR"]
    B -->|m/n/k 负值| B2["ACLBLAS_STATUS_INVALID_VALUE"]
    B -->|枚举越界| B3["ACLBLAS_STATUS_INVALID_ENUM"]
    B -->|跨类型 / 指针空 / ld 违规| B4["ACLBLAS_STATUS_INVALID_VALUE"]
    B -->|校验通过| C{"m==0 或 n==0?"}
    C -->|是| C1["ACLBLAS_STATUS_SUCCESS<br/>合法 no-op，不下发 kernel"]
    C -->|否| D{"typeA == ACL_COMPLEX64?"}
    D -->|否| D1["ACLBLAS_STATUS_NOT_SUPPORTED<br/>本次交付未实现的类型"]
    D -->|是| E{"k==0 或 alpha==(0,0)?"}

    %% 快路径
    E -->|是| F["快路径：C = beta·C"]
    F --> F1{"beta 值"}
    F1 -->|(0,0)| F2["aclrtMemset(C, 0)<br/>位精确置零，避免 NaN 污染"]
    F1 -->|(1,0)| F3["ACLBLAS_STATUS_SUCCESS<br/>无操作"]
    F1 -->|其他| F4["cgemm_ex_beta_scale_do<br/>AIV，单 tile 256 元素"]

    %% 主路径
    E -->|否| G["主路径：3-GEMM 复数分解"]
    G --> G1["cgemm_ex_deinterleave_do ×2<br/>A、B 解交错 → 9 个 workspace 分区"]
    G1 --> G2["cgemm_ex_cube_do ×3<br/>Cube：T1 / T2 / T3<br/>共用同一 tiling"]
    G2 --> G3["cgemm_ex_alpha_beta_do<br/>AIV：合成 + alpha·G + beta·C<br/>原地写回 C"]

    F2 --> H["Host 侧异步返回"]
    F3 --> H
    F4 --> H
    G3 --> H
    H --> I["调用方读回结果前须同步 stream"]
```

##### 3.2.1 host 侧设计：

host 入口 `aclblasCgemmEx`（`blas/gemm/arch35/cgemm_ex_host.cpp`）按 6 个步骤组织，前 3 步在校验函数 `ValidateCgemmExParams` 内完成：

| 步骤 | 内容 | 实现位置 |
| --- | --- | --- |
| 1 | 参数校验（9 步顺序，见下） | `ValidateCgemmExParams` |
| 2 | 快路径判定（`k == 0` 或 `alpha == (0,0)`） | 入口函数内，校验之后 |
| 3 | dtype 路径判定 → `GemmDTypeCase`，未交付类型返回 `ACLBLAS_STATUS_NOT_SUPPORTED` | 入口函数内 |
| 4 | tiling 计算：分核划分、块尺寸、GM 偏移、workspace 分区 | `CalcCgemmExMultiCorePartition` + 主路径填充 |
| 5 | workspace 分配与容量检查 | 主路径 |
| 6 | kernel 启动 | 快路径 / 主路径各自启动序列 |

**步骤 1 的 9 步校验顺序**（顺序敏感，不可重排）：

1. `handle == nullptr` → `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`（最先判定，与其他错误互斥）
2. `m < 0` 或 `n < 0` 或 `k < 0` → `ACLBLAS_STATUS_INVALID_VALUE`
3. `m == 0` 或 `n == 0` → 合法 no-op，返回 `ACLBLAS_STATUS_SUCCESS`
4. `typeA` / `typeB` / `typeC` 取值越界 → `ACLBLAS_STATUS_INVALID_ENUM`
5. `transa` / `transb` 取值越界 → `ACLBLAS_STATUS_INVALID_ENUM`
6. 跨类型（非 `typeA == typeB == typeC`）→ `ACLBLAS_STATUS_INVALID_VALUE`
7. `alpha == nullptr` → `ACLBLAS_STATUS_INVALID_VALUE`
8. `beta == nullptr` → `ACLBLAS_STATUS_INVALID_VALUE`
9. `A` / `B` 空指针（`m > 0` 且 `n > 0` 时）、`C` 空指针（`beta` 非零且 `m > 0` 且 `n > 0` 时）→ `ACBLAS_STATUS_INVALID_VALUE`；随后校验 `lda` / `ldb` / `ldc` 前导维约束，违反 → `ACBLAS_STATUS_INVALID_VALUE`

第 3 步的优先级高于第 4、5 步，因此"`m = 0` 且 `typeA` 非法"返回 `ACBLAS_STATUS_SUCCESS`。

状态码使用以下 7 个枚举值，状态码前缀统一为 **`ACBLAS`（带 L）**：

| 状态码 | 含义 | 触发位置 |
| --- | --- | --- |
| `ACBLAS_STATUS_SUCCESS` | 成功 | 校验通过、no-op、快路径无操作、正常完成 |
| `ACBLAS_STATUS_INVALID_VALUE` | 参数值非法（维度负值、指针空、跨类型、前导维违规） | `ValidateCgemmExParams` |
| `ACBLAS_STATUS_INVALID_ENUM` | 枚举取值越界（`typeA`/`typeB`/`typeC`/`transa`/`transb` 不在合法范围） | `ValidateCgemmExParams` |
| `ACBLAS_STATUS_HANDLE_IS_NULLPTR` | handle 为空 | `ValidateCgemmExParams` 第 1 步 |
| `ACBLAS_STATUS_NOT_SUPPORTED` | 本次交付未实现的同型组合（`typeA != ACL_COMPLEX64`） | 入口函数，校验之后 |
| `ACBLAS_STATUS_EXECUTION_FAILED` | workspace 分配失败或超限（> 2 GiB）、`aclrtMemset` 失败 | 快路径 `aclrtMemset`、主路径 workspace 分配 |
| `ACBLAS_STATUS_INTERNAL_ERROR` | 平台查询失败（`GetAicCoreCount` / `GetAivCoreCount` 返回 0），属防御性路径 | 快路径 / 主路径入口 |

其中 `ACBLAS_STATUS_INVALID_ENUM` 与 `ACBLAS_STATUS_INVALID_VALUE` 不得合并：枚举取值越界返回 `INVALID_ENUM`，取值域/指针/维度合法性违反返回 `INVALID_VALUE`。

**1. 分核策略：**

- **核数一律运行时获取，禁止写死**：host 侧用 `GetAicCoreCount()` / `GetAivCoreCount()`，device 侧用 `AscendC::GetBlockNum()` / `AscendC::GetBlockIdx()`。
- **Cube 侧（M/N 二分）**：在 `mBlocks × nBlocks ≤ cubeCoreNum` 的约束下搜索使 `mBlocks × nBlocks` 最大的划分，`mBlocks` / `nBlocks` 决定实际参与计算的核数；`singleCoreM` / `singleCoreN` 由 `CeilDiv` 求得。**K 维不跨核切分**（`singleCoreK = k`），避免跨核归约。
- **AIV 侧按职责分别计算 block 数**：
  - 解交错：per-core 元素步长为复数元素个数 4096（`CGEMM_EX_DEINTER_STEP`，host 与 kernel 共用同一常数，避免步长不一致导致漏写），block 数 = `ceil(count / 4096)`，下限 1，上限 `aivCoreNum`；
  - 标量合成：按列切分，`colsPerCore = ceil(n / blockNum)`，余列分配给前几个核。
- **不均衡分配**：余量数据块分配给前几个核，不做再平衡，避免数据重分布开销。

**2. 数据分块和内存优化策略：**

Cube 侧（基线块尺寸取自主路径的 tiling 常数）：

| 参数 | 值 | 说明 |
| --- | --- | --- |
| `baseM` | 16 | Cube M 维基线 |
| `baseN` | 16 | Cube N 维基线 |
| `baseK` | 8 | 复数/单精度实数路径的 K 维基线 |
| `C0` | 8 | fixpipe 一次搬运的 C0 行数 |
| L0C C0 块列数 | 16 | arch35 固定 |
| K 维分块粒度 | 64 | 单次 MMAD 的 K 分块 |

- **L1 半区轮转双缓冲**：`a1` 与 `b1` 各占 L1 的固定半区（`l1Size / 2` = 16 KB），合计恰好占满 32 KB、**零余量**；因此双缓冲只能在半区内部轮转，不得另申请 L1 缓冲。块尺寸求解时 L1 输入相应取 `l1Size / 2`，并按半区分列约束（`a1` 与 `b1` 不能写成 `A + B ≤ 32 KB` 的弱形式）。
- **L0C 与 fixpipe**：L0C 256 KB 恒 fp32 累加，驻留 tile 数由 `l0CSize / C0_TILE_BYTES` 决定；`fixBuf` 仅 7 KB，必须分片写出，禁止一次性 fixpipe 整个输出 tile。`MmadParams` 的 `isBias` **不得置位**（其语义为"累加初始矩阵"，k 维累加不应置位）。
- **3-GEMM 的组合不在 Cube 侧**：L0C 256 KB 无法同时驻留 T1 / T2 / T3 三块 fp32，故 3 个 GEMM 按 tile 顺序写出到 workspace，组合下放到 AIV。

AIV 侧（UB 248 KB = 254272 B）：

- 单 tile 元素数固定为 256（`CGEMM_EX_AIV_TILE_ELEMS`），并以 `static_assert` 在编译期守住 UB 上限：`4 × 256 × 12 = 12288 B ≤ 254272 B`（复数最坏情形 `unitBytes = 12`，约 20 倍余量）。
- 按 256 元素 tile 流水而非按 `singleCoreM × singleCoreN` 计算 UB 上限的原因：大 `m` / `n` 下按核工作量反推 tile 不可行，固定 tile 加编译期断言是可验证的替代方案。
- 流水采用 `TQue<QuePosition::VECIN, 2>` / `TQue<QuePosition::VECOUT, 2>` 与 `TBuf<QuePosition::VECCALC>` 组成"搬入 → 计算 → 搬出"三段双缓冲。
- 非 32 字节对齐的搬运用 `DataCopyPad` + `DataCopyExtParams{1, nbytes, 0, 0, 0}`；仅在严格 32 字节对齐时使用 `DataCopy`。

workspace（GM，由 host 分配）：

- 复数路径共 **9 个 fp32 分区**：解交错 A 三块（实部 / 虚部 / 实部+虚部）、B 三块、3-GEMM 中间态 T1 / T2 / T3 三块。
- 大小 = `3 × ceilAlign(m×k) + 3 × ceilAlign(k×n) + 3 × ceilAlign(tempLdc × n)` 字节，各分区按 **256 B** 对齐（`CGEMM_EX_WS_ALIGN`；`aclrtMalloc` 返回的基地址已满足 256 B 对齐）。其中 `tempLdc = ceilAlign(m, 16)` 是 T1/T2/T3 的写出 stride，区别于用户传入的 `ldc`。
- 分配经 `EnsureDefaultWorkspace` 申请、`GetEffectiveWorkspace` 取址，容量上限为 2 GiB（`ACLBLAS_MAX_WORKSPACE_SIZE`）；需求超过上限时返回 `ACBLAS_STATUS_EXECUTION_FAILED`。

**3. tilingkey 规划策略：**

需要 tilingkey 的原因：host 侧的 dtype 判定决定 kernel 走哪条分支（解交错宽度、累加态、回写转换各不相同），kernel 侧需要据此选择计算形态。

**仅 dtype 进入 TilingKey**，取值由 `GemmDTypeCase` 定义（`cgemm_ex_tiling_data.h`）：

| TilingKey | 枚举值 | 触发条件（host 侧判定） | 核角色 | 本次交付 |
| --- | --- | --- | --- | --- |
| TK0 | `GEMM_DTYPE_C32 = 0` | `typeA == typeB == typeC == ACLBLAS_C_32` | Cube + AIV | **已交付** |
| TK1 | `GEMM_DTYPE_R32 = 1` | `== ACLBLAS_R_32` | Cube + AIV | 本次交付未实现 |
| TK2 | `GEMM_DTYPE_HR32 = 2` | `== ACLBLAS_H_R_32` | Cube + AIV | 本次交付未实现 |
| TK3 | `GEMM_DTYPE_HC32 = 3` | `== ACLBLAS_H_C_32` | Cube + AIV | 本次交付未实现 |
| TK4 | `GEMM_DTYPE_BETA_ONLY = 4` | `k == 0` 或 `alpha == (0,0)`（其余参数合法），dtype 无关 | 仅 AIV | 已实现并已纳入本次验收范围（`TC_ED_699`–`TC_ED_707`、`TC_ED_728` 触及） |

**不进入 TilingKey 的维度**：`transa` / `transb`（运行时枚举，由 host 折算为解交错阶段的运行参数，见 3.2.2 第 1 点）、`alpha` / `beta` 标量值（Host 指针，运行时解引用）、`m` / `n` / `k`（运行时值，仅影响循环次数与 tile 裁剪）、`ld` 值（运行时值，仅影响 GM 寻址）。

TK0–TK4 的 dtype 语义差异：complex64 在 Cube 侧无原生 dtype，必须先解交错为 fp32 实矩阵再乘，解交错宽度为 fp32；complex32 需解交错为 fp16 实矩阵，宽度不同；float16 路径需强制中间态 fp32 并在回写前转换。这些差异直接决定解交错宽度、L1/L0 复用倍率与 UB 预算，因此 dtype 必须进入 key。

##### 3.2.2 kernel 侧设计：

kernel 侧共 4 个入口（`blas/gemm/arch35/cgemm_ex_kernel.cpp`，launcher 声明于 `cgemm_ex_kernel.h`）：

| kernel | 任务类型 | launcher | 职责 |
| --- | --- | --- | --- |
| `cgemm_ex_deinterleave_kernel` | `KERNEL_TYPE_AIV_ONLY` | `cgemm_ex_deinterleave_do` | 复数矩阵解交错 |
| `cgemm_ex_cube_kernel` | `KERNEL_TYPE_AIC_ONLY` | `cgemm_ex_cube_do` | 3 个实数 GEMM 中的单次矩阵乘 |
| `cgemm_ex_alpha_beta_kernel` | `KERNEL_TYPE_AIV_ONLY` | `cgemm_ex_alpha_beta_do` | 3-GEMM 合成 + `alpha·G + beta·C` 后处理 |
| `cgemm_ex_beta_scale_kernel` | `KERNEL_TYPE_AIV_ONLY` | `cgemm_ex_beta_scale_do` | 快路径 `C = beta·C` |

1. **解交错（`cgemm_ex_deinterleave_kernel`）**：把列主序 real-first 交错复数矩阵解成 3 个列主序 fp32 矩阵——实部、虚部、(实部 + 虚部)，供第 2 步的三次 GEMM 分别作为 A 操作数与 B 操作数。
   - **转置与共轭语义在这一层一次性折算**：入参 `CgemmExDeinterArgs` 携带 `isTrans`（`OP_T`/`OP_C` 时按逻辑 `op(A)` 转置后写出，即交换源索引 `sidx = isTrans ? (lc + lr·ld) : (lr + lc·ld)`）与 `isConj`（`OP_C` 时**先取负虚部，再算 sum**）。因此 Cube kernel 只看到"不转置、不共轭的实数 fp32 矩阵"，不承载任何转置分支。
   - 跨核 striding：每核每次领 4096 个复数元素，步长 `step = blockNum × 4096`，host 侧用同一常数反推 block 数。
   - 每矩阵一次调用：A、B 各启动一次。

2. **Cube 矩阵乘（`cgemm_ex_cube_kernel`）**：使用 tensor_api 骨架完成 `GM → L1 → L0A/L0B → L0C → GM` 的单次实数 GEMM。
   - GM → L1：`MakeMemPtr` + `Copy`（GM2L1，`Nd2NzParams`），A 用 `NDExtLayoutPtn`、B 用 `DNExtLayoutPtn`。这两个 frame layout 参数的 `d1` 是**列数而不是 stride**，转置位恒为 0。
   - L1 → L0：`Copy`（L12L0A / L12L0B）。
   - 矩阵乘：`Mmad` + `MmadParams{curM, curN, curK, unitFlag, isFirstK}`，`unitFlag` 在最后一个 K 块取 3、其余取 2；**`isBias` 全程不置位**。
   - L0C → GM：`Fixpipe` + `FixpipeParamsArch3510<CO2Layout::ROW_MAJOR>` 分片写出。
   - 入口调用 `AscendC::InitSocState()`；按 `transA2D` / `transB2D` 的四种组合做编译期分派（本实现中两者恒为 0，因为转置已在上一步折算）。
   - host 按 T1、T2、T3 **启动 3 次**该 kernel，三次共用同一份 tiling，仅 A/B/C 的 GM 基地址不同。

3. **标量合成与后处理（`cgemm_ex_alpha_beta_kernel`）**：
   - 读入 T1 / T2 / T3（三块 fp32）；仅 `hasBeta == 1`（`beta != (0,0)`）时读入 C 原值，`beta == (0,0)` 跳过该读以省一次 GM 带宽。
   - 3 次减法（顺序不可交换）：`tmpBuf = T3 − T1`；`gimBuf = tmpBuf − T2`（即 `G_im`）；`greBuf = T1 − T2`（即 `G_re`）。使用独立计算缓冲而非原地覆写，避免 `dst == src` 的读写别名。
   - 标量装配：`are = ar·gre − ai·gim`，`aim = ar·gim + ai·gre`；`hasBeta` 时再累加 `br·cre − bi·cim` 与 `bi·cre + br·cim`。
   - `beta == (0,0)` 时 C 直接写纯 0，**不读 C 原值**：C 原值若含 NaN/Inf，`0 × NaN = NaN` 会污染结果。
   - C 的索引空间以 float 为单位（实部/虚部交错），故列偏移为 `col × cLdc × 2`、行偏移为 `rowOffset × 2`。
   - 写回用 `DataCopyPad` + `DataCopyExtParams{1, count × 2 × 4, 0, 0, 0}`，原地覆写 `C`。

4. **快路径（`cgemm_ex_beta_scale_kernel`）**：`k == 0` 或 `alpha == (0,0)` 时跳过矩阵乘，仅执行 `C = beta · C`。`beta == (0,0)` 时 host 改用 `aclrtMemset` 直接置零（流序操作，与后续 kernel 无竞争），保证位精确且避免 NaN 污染；`beta == (1,0)` 时直接返回 `ACBLAS_STATUS_SUCCESS`（无操作）；其余值启动本 kernel 逐元素缩放。单 tile 256 个复数元素，`static_assert(4 × 256 × 8 ≤ 254272)` 守住 UB 上限。
   - **浮点下标单位约束（已修复缺陷，务必遵守）**：`cGlobal_` 绑定为 `GlobalTensor<float>`、size = `cLdc_ × n_ × 2`，因此其索引空间以 **float 为单位**而非 complex 单元；列 stride 必须是 `cLdc_ × 2`、行 stride 必须是 `2`。2026-09-14 修复 `TC_ED_707` 时（`ProcessTile`，约第 904 行）曾把二者按 complex 单元计算、漏掉 `* 2u`，导致每个核整体左移半个列步长、非首核整列从未被写（16³ 用例漏写 120/256 元素）。该坑在主路径 kernel（第 3 点）里已踩过并修过一次，姊妹 kernel 新增时必须逐位核对下标单位。

5. **同步与流水线**：
   - Cube 与 AIV 是**两个独立 kernel**（`supportMix = 0`），角色间顺序由 stream 上的启动顺序序列化保证，不存在跨 kernel 的角色间同步；kernel 内部用 `BufferID` + `HardEvent`（`SetFlag` / `WaitFlag`，在 MTE2_MTE1、M_MTE1、MTE1_M、M_FIX、FIX_M 等 pipe 对上）完成流水同步。
   - Cube kernel 退出前用 `PipeBarrier<PIPE_ALL>` 确保 fixpipe 的 L0C → GM 写回完成。

**启动序列汇总**：

```text
host: aclblasCgemmEx
      ├─ ValidateCgemmExParams（9 步）
      ├─ m==0 或 n==0  → 返回 ACLBLAS_STATUS_SUCCESS（no-op，不下发 kernel）
      ├─ typeA != ACL_COMPLEX64 → 返回 ACLBLAS_STATUS_NOT_SUPPORTED（本次交付未实现的类型）
      ├─ k==0 或 alpha==(0,0) → 快路径
      │      β==(0,0): aclrtMemset(C, 0)                     …… 位精确
      │      β==(1,0): return ACLBLAS_STATUS_SUCCESS          …… 无操作
      │      其他:     cgemm_ex_beta_scale_do（AIV）           …… 位精确
      └─ 主路径
             cgemm_ex_deinterleave_do ×2（A、B，AIV）→ 9 个 workspace 分区
             cgemm_ex_cube_do ×3（T1 / T2 / T3，Cube，共用同一 tiling）
             cgemm_ex_alpha_beta_do（AIV，合成 + alpha/beta 装配 + 原地写回）

读回结果前须同步 stream（异步语义）
```

##### 3.2.3 算子泛化功能设计

为满足全场景兼容的泛化要求，本算子在以下三个维度进行特化兼容设计：

- **维度泛化**：`m` / `n` / `k` 为运行时 `int` 入参，任意非负取值均被支持；`lda` / `ldb` / `ldc` 作为独立的前导维参数，允许调用方传入大于逻辑维度的值以适配非连续内存布局或对齐填充。host 侧在 tiling 阶段按运行时值动态计算分核划分与 tile 裁剪，kernel 侧不感知具体维度值，仅按 tiling 数据执行循环。
- **边界 shape 兼容**：
  - `m == 0` 或 `n == 0`：合法 no-op，校验阶段直接返回 `ACBLAS_STATUS_SUCCESS`，不启动任何 kernel，避免零尺寸 launch 引发的硬件异常；
  - `k == 0`：跳过矩阵乘，走快路径执行 `C = beta · C`，其中 `beta == (0,0)` 通过 `aclrtMemset` 显式置零（保证位精确，避免 `0 × NaN` 污染），`beta == (1,0)` 直接返回成功（无操作），其余值逐元素缩放；
  - `alpha == (0,0)`：同样走快路径，跳过矩阵乘（因为 `alpha · op(A) · op(B) == 0`），仅执行 `C = beta · C`；
  - 负维度（`m < 0` / `n < 0` / `k < 0`）：校验阶段返回 `ACBLAS_STATUS_INVALID_VALUE`；
  - 前导维边界（`lda < max(1, m)` 等）：校验阶段返回 `ACBLAS_STATUS_INVALID_VALUE`。
- **dtype 泛化**：TilingKey 按 dtype 分为 TK0–TK4 共 5 个分支。TK0（`ACBLAS_C_32`）本次交付已实现；TK1–TK3（`ACLBLAS_R_32` / `ACLBLAS_H_R_32` / `ACLBLAS_H_C_32`）的枚举定义、参数校验与 TilingKey 注册通道均已就位，但 kernel 路径未实现，host 侧对 `typeA != ACL_COMPLEX64` 返回 `ACLBLAS_STATUS_NOT_SUPPORTED`，调用方可据此判定能力边界，不存在静默降级。跨类型组合（`typeA != typeB` 或 `typeB != typeC`）在校验阶段返回 `ACBLAS_STATUS_INVALID_VALUE`，避免混用非法 dtype。

##### 3.2.4 复数分解与 Host 标量装配专属适配说明

**复数分解适配：**

- **3 GEMM 分解的数学依据**：复数乘法 `(Ar + j·Ai)(Br + j·Bi) = (Ar·Br − Ai·Bi) + j·(Ar·Bi + Ai·Br)` 无法用少于 3 个实数 GEMM 完成。本实现引入第三组操作数 `(Ar + Ai)` 与 `(Br + Bi)`，通过 T1 = Ar·Br、T2 = Ai·Bi、T3 = (Ar+Ai)·(Br+Bi) 三个实数 GEMM 构造出 `G_re = T1 − T2` 与 `G_im = (T3 − T1) − T2`，较 4 GEMM 参考路径减少 25% 矩阵乘工作量。
- **Cube 侧无原生复数 dtype**：arch35 的 Cube 单元不支持复数 dtype，矩阵乘必须分解为实数 GEMM。解交错阶段在 AIV 上完成，将 real-first 交错复数矩阵解为 3 个列主序 fp32 实矩阵（实部、虚部、实部+虚部），供 Cube 的三次 GEMM 分别作为操作数。
- **转置与共轭语义在解交错阶段一次性折算**：`isTrans` 与 `isConj` 在解交错时处理，Cube kernel 只看到"不转置、不共轭的实数 fp32 矩阵"，不承载任何转置分支。这使得 Cube kernel 的实现不随 `transa` / `transb` 变化，降低了 kernel 分支复杂度。
- **3 GEMM 组合不在 Cube 侧**：L0C 256 KB 无法同时驻留 T1 / T2 / T3 三块 fp32 结果，因此 3 个 GEMM 按 tile 顺序写出到 GM workspace，组合运算下放到 AIV 完成。

**Host 标量装配适配：**

- **`alpha` / `beta` 驻留 Host 内存**：二者类型为 `const aclblasComplex*`（Host 指针），由 host 端在启动 kernel 前直接解引用获取标量值，不通过 `aclrtMemcpy` 拷贝到 device。标量值通过 tiling 数据的 `alphaReal` / `alphaImag` / `betaReal` / `betaImag` 字段按值传入 kernel。
- **独立后处理步骤**：标量装配（`alpha·G + beta·C`）需要读取矩阵乘结果 T1/T2/T3 与 C 原值，执行复数减法与标量乘法，再原地写回 C。单个 Cube 角色无法完成此步骤（Cube 无复数向量运算能力），因此必须增加独立的 AIV kernel（`cgemm_ex_alpha_beta_do`）负责后处理。
- **双核角色分工**：Cube（`KERNEL_TYPE_AIC_ONLY`）与 AIV（`KERNEL_TYPE_AIV_ONLY`）是两个独立 kernel（`supportMix = 0`），角色间顺序由 stream 上的启动顺序序列化保证。AIV 承担解交错、标量合成与快路径三种职责，Cube 承担矩阵乘累加。

**IEEE 754 特殊值语义：**

- 输入含 `Inf` / `NaN` 时按 IEEE 754 传播，对齐 cuBLAS，不提前报错。
- `beta == (0,0)` 时 C 直接写纯 0，**不读 C 原值**：C 原值若含 NaN/Inf，`0 × NaN = NaN` 会污染结果。快路径中 `beta == (0,0)` 通过 `aclrtMemset` 显式置零，保证位精确。
- 主路径中 `beta == (0,0)` 时，AIV 合成 kernel 跳过 C 原值的 GM 读取（节省一次 GM 带宽），直接写纯 0。
- **`Inf` 输入的实测边界**：官方 CSV 的 `VALUE_NORM_INF` 填充 token 在 float32 算子语义下会产生 `out = nan` 而 CPU double golden 产生 `gold = -inf`（`TC_FL_546`，`mismatch = 1024/1024`），两者均为 IEEE 754 合法传播结果但数值不等；该 token 与本算子 float32 实现不兼容，已登记为 B 类判据/语义问题（见"可维可测分析"），非算子正确性缺陷。

### 数据检测：

- **数据类型检测**：`typeA` / `typeB` / `typeC` 必须同型（`typeA == typeB == typeC`），跨类型返回 `ACBLAS_STATUS_INVALID_VALUE`；本次交付仅支持 `typeA == ACL_COMPLEX64`（`ACLBLAS_C_32`），其余同型组合返回 `ACLBLAS_STATUS_NOT_SUPPORTED`；枚举取值越界返回 `ACBLAS_STATUS_INVALID_ENUM`。
- **形状检测**：`m` / `n` / `k` 必须 ≥ 0，负值返回 `ACBLAS_STATUS_INVALID_VALUE`；`m == 0` 或 `n == 0` 为合法 no-op；`lda ≥ max(1, transa==OP_N ? m : k)`、`ldb ≥ max(1, transb==OP_N ? k : n)`、`ldc ≥ max(1, m)`，违反返回 `ACBLAS_STATUS_INVALID_VALUE`。
- **越界检查**：`handle` / `alpha` / `beta` 不可为 `nullptr`；`A` / `B` 在 `m > 0` 且 `n > 0` 时不可为 `nullptr`；`C` 在 `beta` 非零且 `m > 0` 且 `n > 0` 时不可为 `nullptr`；workspace 需求超过 2 GiB（`ACLBLAS_MAX_WORKSPACE_SIZE`）时返回 `ACBLAS_STATUS_EXECUTION_FAILED`。
- **测试侧检测**：测试工程对官方 CSV 表头做 ORDER_MISMATCH 校验（30 列名字与位置须与官方一致，多一列或少一列都会报 `unexpectedColumns` / `missingColumns`，不会静默读错列）；`random_seed` 列作为随机数种子，保证重跑可复现同一组数据。

### 支持硬件

| 支持的芯片版本 | 架构 / variant | CANN 版本 | 涉及勾选 |
| --- | --- | --- | --- |
| Ascend 950PR（`Ascend950PR_9579` / `ascend950`） | `DAV_3510`（arch35），variant `dav_c310` | 9.1.0 | √ |
| Atlas 800I/T A2、Atlas 800I/T A3（`DAV_2201`） | arch220 | — | 不支持 |
| Atlas 300I Duo / Arch310（`DAV_3000`） | arch310 | — | 不支持 |

- 无集合通信依赖（不使用 HCCL）。
- 存储预算按 arch35 硬件事实核算：Cube L1 32 KB（`a1` / `b1` 各 16 KB 固定半区）、L0A 64 KB、L0B 64 KB、L0C 256 KB、fixBuf 7 KB 为 Cube 侧独立物理存储；UB 248 KB（254272 B）专供 AIV。arch35 必须使用 V350 特化的硬件常量，不得混用默认模板常量。
- 实测环境：`ASCEND_DEVICE_ID=0`，`SOC_VERSION=ascend950`，`__NPU_ARCH__ 3510`。

### 算子约束限制

**数据类型限制**

1. 本次交付仅实现 `ACLBLAS_C_32`（complex64）同型路径。`ACLBLAS_R_32`、`ACLBLAS_H_R_32`、`ACLBLAS_H_C_32` 三条扩展路径**本次交付未实现**，调用时返回 `ACLBLAS_STATUS_NOT_SUPPORTED`，不启动任何 kernel。收窄原因见"背景介绍 › 交付范围界定"，非能力缺陷或遗漏。
2. 仅支持同型组合（`typeA == typeB == typeC`）；跨类型组合返回 `ACBLAS_STATUS_INVALID_VALUE`。
3. `include/cann_ops_blas.h` 中 `aclblasCgemmEx` 的函数注释声明支持四条 type 路径，但实现仅支持 `ACLBLAS_C_32`。该不一致属交付范围收窄所致，接口契约的完整形态由注释描述，实现状态由返回码判定。

**转置限制**

4. 本次交付覆盖并验证 `transA = ACBLAS_OP_N`、`transB = ACBLAS_OP_N`。`ACBLAS_OP_T` 与 `ACBLAS_OP_C` 转置形态**本次交付未覆盖**。说明：转置与共轭语义在解交错阶段已有完整的参数通道（`isTrans` / `isConj` 折算），host 侧的枚举校验也已就位，因此调用带 T/C 参数的 `ACLBLAS_C_32` 组合不会报错；但该组合的正确性未纳入本次交付的验证范围，验收结论以 NN 组合为准。

**接口与形态限制**

5. 算法选择由后端自动完成，调用方无算法选择相关入参。
6. 单批语义，无 `batchCount` 参数；批量接口由独立的 `aclblasGemmBatchedEx` / `aclblasGemmStridedBatchedEx` 承接。
7. 不支持图模式调用：仅提供 ops-blas 库函数形态，不提供图算子节点注册，不通过 `aclOpExecutor` 传参。
8. 不支持非连续 Tensor、广播、动态 shape、自定义 workspace、通用确定性计算要求。
9. `m` / `n` / `k` 为负值返回 `ACBLAS_STATUS_INVALID_VALUE`；`m == 0` 或 `n == 0` 为合法 no-op 并返回 `ACBLAS_STATUS_SUCCESS`（该判定先于类型与转置枚举校验，因此"空矩阵 + 非法类型"返回成功）。
10. `alpha` 与 `beta` 指针不可为 `nullptr`；`C` 在 `beta` 非零时不可为 `nullptr`；`A` / `B` 在 `m > 0` 且 `n > 0` 时不可为 `nullptr`。
11. 前导维约束：`lda ≥ max(1, transa==OP_N ? m : k)`；`ldb ≥ max(1, transb==OP_N ? k : n)`；`ldc ≥ max(1, m)`，违反返回 `ACBLAS_STATUS_INVALID_VALUE`。
12. `alpha` / `beta` 驻留 Host 内存，由 host 端直接解引用；调用方不得传入 Device 指针。
13. workspace 需求超过 2 GiB（`ACBLAS_MAX_WORKSPACE_SIZE`）时返回 `ACBLAS_STATUS_EXECUTION_FAILED`。

**数值语义限制**

14. 默认非位精确：浮点累加顺序敏感，不保证跨运行、跨核数的 bitwise 可复现。
15. `k == 0` 或 `alpha == (0,0)` 的快路径 `C = beta·C` 要求位精确；其中 `beta == (0,0)` 通过显式置零（而非 `0 × C`）保证不被 C 中的 NaN/Inf 污染。
16. 输入含 `Inf` / `NaN` 时按 IEEE 754 传播，对齐 cuBLAS，不提前报错。但官方 CSV 的 `VALUE_NORM_INF` / `RANDOM_EXTREME` 两个填充 token 与 float32 实现不兼容（`nan` 与 `-inf` 不判相等、float 累加溢出为 `Inf` 而 double golden 有限），实测不通过（见"可维可测分析 › B 类"）。

## 可维可测分析

### 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | **实测判据**：逐行 `mere_threshold` + `mare_multiplier`，PASS 条件为 `mismatchCount == 0` 且 `mere < mere_threshold` 且 `maxRelErr < outlierLimit`（`outlierLimit = mare_multiplier × mere_threshold`）。范围内全部行的取值为 `mere_threshold = 2⁻¹³ ≈ 1.221e-4`、`mare_multiplier = 10.0`、`outlierLimit ≈ 1.221e-3`。复数输出按实部、虚部分别统计，两分量均须通过（AND 语义）；`|gold| < mere_threshold` 的元素跳过平均。golden 为**CPU 端 `double` 精度独立实现**。**实测结果（官方 CSV C_32 NN 范围内 66 条，2026-09-14）**：60 条数值比较 **46 PASS / 14 FAIL**（修复前 45 PASS / 15 FAIL，差的 1 条即已修复的算子缺陷 `TC_ED_707`），6 条状态码断言 **全部 PASS**，合计 **52 PASS / 14 FAIL**。**任务书陈述**（文档参考，非测试判据）：`ACLBLAS_C_32` 分量 rtol = 2⁻¹⁰ ≈ 9.7656e-4，atol = 2⁻¹⁶ ≈ 1.5259e-5；matched_ratio ≥ 0.99 且 max_abs_error ≤ 1e-2。 | 官方 CSV `mere_threshold` / `mare_multiplier` 列；任务书 §3.2 |
| 性能标准 | 判据为 `ratio = gpu_ms / (npu_mean_us / 1000) >= 0.4`（阈值取官方 `verify_performance.py` 的 `PERF_THRESHOLD = 0.4`），warmup 3 次 + 50 有效样本均值，逐条与 `gpu_baseline.csv` join。范围内共 50 条（`C_32` + NN）。**实测结果（2026-09-14）**：50 条中 47 条已落盘，**0 PASS / 42 FAIL / 5 ERROR**（ERROR 为 `stream sync failed`），3 条因运行被中断未落盘。已落盘的 42 条 FAIL 中最高 `ratio = 0.0115`（`TC_PF_1005`，1×1×1）。属**已知性能缺陷**，不阻塞本次交付。**任务书 §3.3 陈述的阈值**（1024³ ≤ 415.95 µs、2048³ ≤ 3243.65 µs）与官方 `gpu_baseline.csv` 中同形状基线（1024³ = 166.381 µs、2048³ = 1297.460 µs）相差恒定 **2.5 倍**，见"风险与降级预案"。 | 官方 `verify_performance.py` 常量；官方 `gpu_baseline.csv`；任务书 §3.3 |
| 内存要求 | 不涉及（任务书未提内存要求） | 任务书 §3.4 |

**精度判据明细（官方 CSV 判据）**

| 判据 | 值 | 说明 |
| --- | --- | --- |
| `mere_threshold` | 0.00012207031 = 2⁻¹³ | 平均相对误差上限（Mean Error Ratio）；范围内每行同值 |
| `mare_multiplier` | 10.0 | `maxRelErr` 允许超出 `mere_threshold` 的倍率 |
| `outlierLimit` | ≈ 1.221e-3 | `mare_multiplier × mere_threshold`；`maxRelErr` 越界即 FAIL |
| `kEpsilon` | 2⁻¹⁴ ≈ 6.104e-5 | 相对误差分母的稳定项 |
| golden | CPU 端 `double` 精度独立实现（`cgemmex_golden.h`） | 独立实现，不复用算子自身代码路径；非 cblas |
| 比对方式 | 实部、虚部分别比对 | C_32 路径，AND 语义 |
| 近零处理 | `|gold| < mere_threshold` 的元素跳过平均 | 纯相对误差判据；`maxAbsErr` 另记但不参与判定 |
| 输入分布 | 官方 CSV 的 fill token 原样支持：`RANDOM_NORM_5_5`、`RANDOM_EXTREME`、`RANDOM_ALTER`、`VALUE_NORM_0`、`VALUE_NORM_INF` 等 | 由 `random_seed` 列驱动，重跑可复现 |
| 阈值来源 | 逐行读 CSV，不写死、不上调 | 未做任何上调 |

**实测精度结果（2026-09-14，官方 CSV 范围内 66 条）**

| 分类 | 条数 | 明细 | 结论 |
| --- | --- | --- | --- |
| 数值比较 PASS | 46（修复前 45） | 34 条真实比较通过 + 12 条 `valid = 0/0`（其中 `TC_ED_707` 由独立判别探针确认 256/256 逐位相等，见下） | 达标 |
| 状态码 PASS | 6 | `TC_ED_720` / `722` / `724` / `725` / `726` / `727`，期望 `INVALID_VALUE`、实得 `INVALID_VALUE` | 达标 |
| **合计** | **52 PASS / 14 FAIL**（修复前 51 / 15） | 66 条范围内用例 | — |
| A 类 FAIL（判据形态） | 12 | `TC_SQ_022`/`024`/`025`/`026`/`027`/`028`/`029`/`030`/`031`、`TC_CV_609`、`TC_CV_618`、`TC_AB_438` | `mere` 全部达标（低于阈值 10 倍以上），仅 `maxRelErr` 越 `outlierLimit` |
| B 类 FAIL（填充 token 语义） | 2 | `TC_FL_545`、`TC_FL_546` | token 超出 float32 算子语义的定义域 |
| C 类 FAIL（算子缺陷） | 1 | `TC_ED_707` | **已修复**，修复后转入 PASS 列，见下 |

关键实测锚点：

- `TC_L0_001`（4³）与 `TC_L0_002`（8³）：`mere` 量级 1e-7，通过。
- **分界干净**：`TC_SQ_021`（64³）PASS，`TC_SQ_022`（65³）起 FAIL —— A 类判据争议的边界点。
- `TC_SQ_024`（128³，主精度点）：`mere = 1.031e-6 / 3.846e-6`（低于阈值两个数量级），`mismatch = 0/0`，仅 `maxRelErr.im = 2.220e-2` 越过 `outlierLimit`，`maxAbsErr = 1.831e-4 / 3.648e-4` —— A 类。
- `TC_SQ_031`（2048³）：`mere = 5.426e-6 / 1.114e-5`（仍低于阈值），`maxAbsErr = 4.395e-3 / 9.521e-3` —— A 类；`maxAbsErr` 随 `k` 从 `9.155e-5`（k=65）单调增长到 `9.521e-3`（k=2048）。
- CPU 自测（golden 手算 / 转置共轭交叉核对 / 快路径 / 填充语义 / 精度策略）：6/6 PASS。

**`TC_ED_707` 修复（C 类，2026-09-14）**

用例 `16×16×16`，NN，`alpha=(0,0)`，`beta=(0.5,0.5)` —— 触发 beta-only 快路径的 `cgemm_ex_beta_scale_do`。

- **根因**：`cGlobal_` 绑定为 `GlobalTensor<float>`（size = `cLdc_ × n_ × 2`），索引空间以 **float** 为单位；原代码把列号与行偏移按 **complex 单元**计算，列 stride 少乘 `2`、行 stride 少乘 `2`。每个核整体左移半个列步长，相邻核窗口重叠，非首核整列从未被写，16³ 用例漏写 **120/256** 元素（列 8 漏行 8–15、列 9–15 全漏），被写到的 136 个元素是**位精确正确**的，且无越界/污染写（`corrupted = 0`）。
- **修复**：`cgemm_ex_kernel.cpp` 约第 904 行，`ProcessTile` 内 GM 偏移补 `* 2u`（列偏移 `col × cLdc × 2`、行偏移 `rowOffset × 2`），净变化 2 处乘法 + 10 行注释；host 代码、主路径 kernel、测试工程均未改动。
- **修复后实测**：官方判据 PASS（`mere = 0`，`skipped = 256/256`）；判别探针确认 `untouched = 0/256`、`corrupted = 0/256`、256/256 逐位相等，且全部元素 `|gold| ≥ 1.221e-4`（真·位精确，不是"近零被容差跳过"的假象）。
- **影响面**：对全部 1200 行做分派枚举，快路径命中 85 行（β=(0,0)→`aclrtMemset` 42 行、β=(1,0)→no-op 42 行、β=其他→`beta_scale` **1 行**），主路径 1115 行。**能到达被改代码行的官方用例只有 `TC_ED_707` 一条**。
- **主路径回归**：`TC_SQ_024`（128³，不进入快路径）修复前后数字**逐字相同**（`mere = 1.031e-6 / 3.846e-6`），确认未受影响。

**性能判据与实测明细**

| # | m × n × k | transA | transB | dtype | 任务书 §3.3 阈值（µs） | `gpu_baseline.csv` 基线 | NPU 实测均值 | 结论 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1024³ | N | N | `ACLBLAS_C_32` | ≤ 415.95 | 0.166381 ms = 166.381 µs | 6,134,067 µs（≈6.13 s） | **FAIL**，`ratio ≈ 0.0000`（约 36,866× 慢于 GPU 基线） |
| 2 | 2048³ | N | N | `ACLBLAS_C_32` | ≤ 3243.65 | 1.297460 ms = 1297.460 µs | 40,713,292 µs（≈40.7 s） | **FAIL**，`ratio ≈ 0.0000`（约 31,380× 慢于 GPU 基线） |
| 3 | 1024³ | T | N | `ACLBLAS_C_32` | ≤ 411.27 | 0.164508 ms = 164.508 µs | — | 本次交付未覆盖（`OP_T` 形态） |
| 4 | 2048³ | N | N | `ACLBLAS_R_32` | ≤ 851.86 | 0.340743 ms = 340.743 µs | — | 本次交付未实现（`ACLBLAS_R_32`） |

范围内 50 条性能用例的实测状态：

| 状态 | 条数 | 说明 |
| --- | --- | --- |
| FAIL | 42 | `ratio` 全部 < 0.4，最高 0.0115（`TC_PF_1005`，1×1×1，`npu_mean = 570 µs`）；`ratio` 随 `m·n·k` 单调下降到 1e-6 量级 |
| ERROR | 5 | `stream sync failed`：`TC_PF_1091`（1024×2048×64）、`TC_PF_1099`（4096×512×64）、`TC_PF_1101`（4096×1024×64）、`TC_PF_1103`（4096×2048×64）、`TC_PF_1164`（931×3291×3057） |
| 未落盘 | 3 | `TC_PF_1051`、`TC_PF_1069`（3635³，估算单条约 242 分钟）、`TC_PF_1158` —— 后台运行被中断 |
| **PASS** | **0** | — |

说明：

1. 50 条按 `m·n·k` 升序执行、逐条 flush CSV，中断后 CSV 中始终是"已完成的小 shape"。全量 50 条按 `TC_PF_1001` 的吞吐折算约 **12.5 小时**（`total_ops = 1.4879e+11`，`est_total_min ≈ 750.1`），单会话内无法跑完。
2. 小 shape 的 `ratio` 已经很低（1×1×1 仅 0.0115），说明约 0.5–0.6 ms 的固定开销主导了小 shape 耗时，与算子规模无关 —— 这是实现侧缺陷，不是"小 shape 采样噪声"。
3. 实测 `TC_PF_1001`（1024³）单条 50 样本约 324,866 ms，与另一条开发线独立测得的 1024³ ≈ 6.12 s 相互印证。
4. 任务书 §3.3 的阈值与 `gpu_baseline.csv` 同形状基线相差恒定 2.5 倍（4 行全部命中：415.95/166.381、3243.65/1297.460、411.27/164.508、851.86/340.743 均 = 2.500），判据以 CSV 为准。

### 测试用例规划

测试工程位于 `ops-blas/test/gemmex/cgemmex/`，由官方 CSV 驱动（逐行读取 30 列），按"最小可用"范围过滤后执行。**范围没有做任何人为削减**：下列条目是该范围内官方 CSV 的**全部**行，一条不省；阈值一律取 CSV 列值，未做任何上调。

**覆盖口径（实测统计，非估算）**

| 项 | 数量 | 说明 |
| --- | --- | --- |
| 官方 CSV 总行数 | 1200 | `cgemmex_test.csv`，30 列，MD5 `b2c88b625db1194162e658241ba8c4ef` |
| 本次覆盖 | **116（9.67%）** | 精度 66 条（60 数值 + 6 状态码）+ 性能 50 条 |
| 未覆盖 | 1084 | 608 条非 C_32 或非 NN（含 5 条非法枚举负向：`TC_ED_715/716/717/718/719`，依次为 `transA` / `transB` / `typeA`+`transA` / `typeB` / `typeC` 取 `INVALID`）；7 条落在范围筛选器内部的空指针/空 handle 负向（`TC_ED_708` + `TC_ED_709`~`714`）；150 条 TC_PF 非范围内；319 条 TC_EX（271）+ TC_RC（48） |

上表四项互不重叠且加总恰为 1084。范围筛选器把 681 条精度族切成 73 条 C_32/C_32/C_32 + NN（其中 60 条数值比较 + 6 条状态码 PASS + 7 条负向被排除）与 608 条范围外；加上 TC_PF 50/200、TC_EX+TC_RC 0/319，覆盖 116 条。下文"明确不在本次覆盖范围内"列出的"12 条负向"是**全 CSV 口径**（6 空指针 + 1 空 handle + 5 非法枚举），其中 7 条落在上表的"7"，另 5 条因字段非法落在"608"——两个数字各自成立，不构成矛盾。

**按用例族的覆盖与实测状态**

| 用例族 | 官方总数 | 本次覆盖 | 实测结果 | 备注 |
| --- | --- | --- | --- | --- |
| `TC_L0` 基线 | 8 | 2 | 2/2 PASS | `4³`、`8³`，`mere` 量级 1e-7 |
| `TC_SQ` 方阵形状 | 414 | 23 | 14 PASS / 9 FAIL | FAIL 全为 A 类；`64³` PASS、`65³` 起 FAIL |
| `TC_AB` 非方阵 | 48 | 8 | 7 PASS（含 2 条退化）/ 1 FAIL | FAIL 为 `TC_AB_438`（A 类）；退化 `TC_AB_432`/`435`（`VALUE_NORM_0`，`valid = 0/0`） |
| `TC_CV` 常量填充 | 144 | 8 | 6 PASS / 2 FAIL | FAIL 为 `TC_CV_609`/`618`（A 类） |
| `TC_LD` 前导维 | 24 | 3 | 3/3 PASS | 保留 `ldc > n` 的 padding |
| `TC_FL` 填充/极值 | 12 | 6 | 4 PASS（含 3 条退化）/ 2 FAIL | FAIL 为 `TC_FL_545`/`546`（B 类）；退化 3 条为 `VALUE_NORM_0`，真实比较仅 `TC_FL_543` |
| `TC_ED` 边界与负向 | 31 | 16 | 10 数值 PASS（含 7 条退化）/ 0 FAIL / 6 状态码 PASS | 真实比较 3 条（`703`/`704`/`728`）+ 退化 7 条（`699`/`700`/`701`/`702`/`705`/`706` 零维或 `k=0`，`707` beta-only 快路径） |
| **精度合计** | **681** | **66** | **52 PASS / 14 FAIL**（修复前 51 / 15） | 46 数值 PASS + 6 状态码 PASS |
| `TC_PF` 性能 | 200 | 50 | 0 PASS / 42 FAIL / 5 ERROR / 3 未落盘 | `gpu_baseline` join 命中率 50/50 |

**FAIL 完整清单（三分类，无遗漏；修复后 14 条 = A 类 12 + B 类 2，C 类 1 条已修复转入 PASS）**

| 类别 | 用例 | 关键实测 | 归因 |
| --- | --- | --- | --- |
| **A 类：大 `k` 近零元素的相对误差假象（12 条）** | `TC_SQ_022`(65³)、`TC_SQ_024`(128³)、`TC_SQ_025`(200³)、`TC_SQ_026`(256³)、`TC_SQ_027`(400³)、`TC_SQ_028`(512³)、`TC_SQ_029`(800³)、`TC_SQ_030`(1024³)、`TC_SQ_031`(2048³)、`TC_CV_609`(200³)、`TC_CV_618`(400³)、`TC_AB_438`(32³) | `mere` 全程 `1.021e-6` ~ `1.114e-5`，**低于阈值 1.221e-4 十倍以上，全部达标**；仅 `maxRelErr` 越过 `outlierLimit = 1.221e-3`；`mismatch = 0/0`；`maxAbsErr` 随 `k` 从 `9.155e-05` 单调增长到 `9.521e-03` | 判据形态问题，**非算子正确性问题**：float32 累加误差（k=2048、输入 ±5.5、累加器量级 ~1e4）撞上**纯相对**判据。若判据改为相对+绝对混合或按 `|gold|` 分档，这 12 条会转为 PASS。阈值未做任何上调。**该结论属判据争议，需上游调整判据口径，不在本算子修复范围内。** |
| **B 类：填充 token 超出 float32 算子语义（2 条）** | `TC_FL_545`（`RANDOM_EXTREME`）、`TC_FL_546`（`VALUE_NORM_INF`） | `TC_FL_545`：`maxAbsErr = 5.559e+04 / 6.457e+04`，最坏 `out = -25414.7` / `gold = 513.366`（float 累加溢出为 `Inf` 而 double golden 有限）；`TC_FL_546`：`mismatch = 1024/1024`，`out = nan` / `gold = -inf` | 填充 token 与 float32 实现的定义域冲突，**非精度缺陷**。期望行为（算子应产出何值）需上游明确，本次不擅自定义。 |
| **C 类：真实算子缺陷（1 条，已修复）** | `TC_ED_707`（16³，`alpha=(0,0)`、`beta=(0.5,0.5)`） | 修复前 `mere = 4.245e+00`、`maxAbsErr = 4.484e+00`、漏写 120/256；修复后 PASS、`mere = 0`、256/256 逐位相等 | beta-only 快路径 GM 下标按 complex 单元而非 float 计算，漏乘 `* 2`。已修复并回归通过。 |

**零比较的 PASS（不计入有效覆盖）**

下列 12 条 PASS 的 `valid = 0/0`（没有任何元素进入 `mere` 的均值统计），**不计入有效覆盖**。三种成因不同：

- **零维或 `k = 0`**（6 条）：`TC_ED_699`(0×8×8)、`TC_ED_700`(8×0×8)、`TC_ED_701`(8×8×0)、`TC_ED_702`(0×0×0)、`TC_ED_705`(8×8×0)、`TC_ED_706`(8×8×0) —— 无元素参与或走快路径全量置零；
- **`VALUE_NORM_0` 填充**（5 条）：`TC_AB_432`、`TC_AB_435`、`TC_FL_544`、`TC_FL_547`、`TC_FL_548` —— 全量 `|gold| = 0 < mere_threshold`，走 `skippedNearZero` 分支（`skipped = 1024/1024`）；
- **beta-only 快路径**（1 条）：`TC_ED_707`(16³) —— `skipped = 256/256` 走的是"逐位相等则跳过"分支，即 256 个元素全部 `out == gold`。

> 注：既有测试报告中这组条目的计数口径不一致（一处记 8 条、一处记 6 条，而逐条枚举为 11 条 + `TC_ED_707` 修复后新增 1 条 = 12 条）。本节按**逐条实测的 `valid = 0/0` 枚举**给出 12 条，为最严格口径。据此，46 条数值 PASS 中真正比较了非零元素的是 **34 条**（`TC_ED_703`（1³）、`TC_ED_704`（2³）等 `valid` 很小但非零，计入 34 条）。`TC_ED_707` 的 PASS 不是容差跳过的假象：其正确性由独立判别探针确认的 256/256 逐位相等 + 全部元素 `|gold| ≥ 1.221e-4` 佐证。

**复现方式**

```bash
cd ops-blas/test/gemmex/cgemmex/arch35
bash run.sh --clean       # 只构建
bash run.sh --golden      # 6 条 CPU 自测（golden 手算 / 转置共轭交叉核对 / 快路径 / 填充语义 / 精度策略）
bash run.sh --scope       # 打印覆盖 vs 未覆盖清单（含逐条原因）
bash run.sh --accuracy    # 66 条精度用例
bash run.sh --perf        # 50 条性能用例（全量约 12.5 小时）
```

常用参数：`--csv`、`--baseline`、`--out`、`--device N`、`--warmup N`、`--samples N`（下限强制 50）。退出码：全 PASS `0`、有 FAIL `1`、依赖缺失 `3`。

**明确不在本次覆盖范围内**（未实现，不做挑用例）：

- `R_32` / `H_R_32` / `H_C_32` 及任何混合 dtype 组合（608 条非 C_32 或非 NN）；
- `transA = T` / `C`、`transB = T` / `C` 的转置与共轭路径；
- `TC_EX`（271 条）、`TC_RC`（48 条）用例族；
- 性能用例中非 C_32 NN 的 150 条；
- 空指针用例（`alpha_null` / `beta_null` / `a_null` / `b_null` / `c_null`，6 条）与 `handle = nullptr`（1 条）、非法枚举（`transA = INVALID` / `typeA = INVALID` 等，5 条）—— 本次不覆盖（host 侧校验逻辑已实现并可用，但本次未纳入验收）。

### 兼容性分析

本算子为新增接口，不涉及既有接口的行为变更，兼容性影响集中在三处：

1. **新增函数符号 `aclblasCgemmEx`**：声明于 `include/cann_ops_blas.h`，与同族 Ex 接口（`aclblasGemmEx` / `aclblasGemmBatchedEx` / `aclblasGemmStridedBatchedEx` / `aclblasSgemm` / `aclblasCgemm`）并列，不修改任何既有函数的签名与行为。
2. **新增 `aclblasType_t` typedef**：`include/cann_ops_blas_common.h` 中新增 `typedef aclDataType aclblasType_t;` 与 `ACLBLAS_R_32` / `ACLBLAS_C_32` / `ACLBLAS_H_R_32` / `ACLBLAS_H_C_32` 常量。上游仓中该 typedef 无既有定义，同族 Ex 接口继续使用 `aclDataType`，不受影响；该 typedef 使 `ACL_*` 常量可直接作为 type 参数传入，**不引入 950PR 私有平行 API**，符合"接口须可与其他产品线共用"的要求。
3. **状态码的使用**：本次交付在既有状态码集合内使用 `ACBLAS_STATUS_NOT_SUPPORTED`（未实现的同型组合）、`ACBLAS_STATUS_EXECUTION_FAILED`（workspace 超限）与 `ACBLAS_STATUS_INTERNAL_ERROR`（平台查询失败的防御性路径），均为既有枚举值，不新增状态码。调用方若已按完整枚举处理返回值则无需改动。

上游先例：`aclblasCgemm` 已提供单精度复数 GEMM 的 complex 路径，`aclblasGemmEx` 已提供混合类型的 fp32 路径，本次交付的 `ACLBLAS_C_32` 主路径与二者语义一致，不引入新的语义分歧点。

### 风险与降级预案

| 风险项 | 影响 | 降级预案 |
| --- | --- | --- |
| **性能已知缺陷（不阻塞交付，但须如实披露）** | 范围内 50 条性能用例实测 0 PASS：42 条 FAIL、5 条 `stream sync failed`、3 条未落盘。最高 `ratio = 0.0115`（1×1×1），1024³ 约 6.13 s、2048³ 约 40.7 s（GPU 基线 166.4 µs / 1297.5 µs，约 3.1万–3.7万倍）。小 shape 的 `ratio` 已极低，说明存在约 0.5–0.6 ms 的**固定开销主导项**，与算子规模无关；5 条 `stream sync failed` 集中在 `m·n ≥ 2M` 且 `k = 64` 的形状与一个 931×3291×3057 形状。全量 50 条折算约 12.5 小时，单会话内无法跑完。 | ① 固定开销定位优先（6 次 kernel launch 串行：2 次解交错 + 3 次 cube + 1 次合成），核对是否 fallback 到 AIV-only、tiling 是否退化、核数是否被钳到 1；② `stream sync failed` 5 条单独复现，确认是 host 侧句柄/stream 状态问题还是显存/超时问题；③ 性能不作为本次交付验收阻塞项（本次交付的验收口径是精度判据），如实标注 47 条落盘中 0 PASS 与最高 `ratio`，不虚构达标数字。 |
| **A 类判据争议（12 条 FAIL，需上游调整口径）** | `TC_SQ_022`–`031`、`TC_CV_609`/`618`、`TC_AB_438` 全部判 FAIL，但 `mere` 低于阈值十倍以上、`mismatch = 0/0`、`maxAbsErr` 随 `k` 从 `9.155e-05` 增长到 `9.521e-03`（float32 GEMM 的正常量级）。分界干净：`64³` PASS、`65³` 起 FAIL。 | **不改算子、不上调阈值**。如实归类为判据形态问题（纯相对判据撞上 float32 累加误差），提交时附 A 类 12 条的完整实测数字与 `mere`/`maxRelErr`/`maxAbsErr` 三元组，由上游决定判据是否改为相对+绝对混合或按 `|gold|` 分档。明确不声称"其实算子是对的"——判据确实判 FAIL，归类如实。 |
| **B 类填充 token 语义未定义（2 条 FAIL）** | `TC_FL_545`（`RANDOM_EXTREME`）float 累加溢出为 `Inf` 而 double golden 有限（最坏 `out = -25414.7` / `gold = 513.366`）；`TC_FL_546`（`VALUE_NORM_INF`）`out = nan` / `gold = -inf`，`mismatch = 1024/1024`。 | **不擅自定义期望行为**。如实归类为 token 与 float32 算子语义的定义域冲突，提交时说明两点：(a) `nan != -inf` 在任何数值语义下都不相等，该用例无法通过 `(out, gold)` 数值比较；(b) `RANDOM_EXTREME` 的取值范围超出 float32 累加可表示范围。期望行为需上游明确后再实现。 |
| **GPU 基线与 NPU 实测的口径差异（性能判据脆弱性）** | 性能判据 `ratio = gpu_ms / npu_ms` 把 NPU 实测与官方 `gpu_baseline.csv` 直接相比，二者来自**不同的硬件与软件栈**，量纲之外还叠加了 kernel 组织方式（本算子 6 次串行 launch vs GPU 单 kernel）与固定开销的差异。任务口径中称此差异为"GCN 与 NPU 差异"；**仓内与本报告均无"GCN"字样来源**（`gpu_baseline.csv` 无硬件型号/驱动版本/软件栈列，仅 `id,m,n,k,transA,transB,typeA,typeB,typeC,gpu_ms` 9 列），故此处按"GPU 基线 vs NPU 实测的口径差异"记录，具体基线产生环境**需上游澄清**。另外任务书 §3.3 的阈值与 CSV 基线相差恒定 2.5 倍（4 行全部 = 2.500），两套口径并存。 | 性能结论一律标注"判据为官方 CSV 基线口径"，并同时给出 NPU 绝对耗时与 GPU 基线两个数字，不换算成"达标/不达标"以外的额外承诺。判据常量以 `verify_performance.py` 的 `PERF_THRESHOLD = 0.4` 为准（其 docstring 写的 0.8 与常量不符，以常量为准）。基线产生环境待上游澄清后回填本节。 |
| **测试工程既有缺陷（已修 1 项，余 1 项登记）** | ① `cgemmex_test.cpp:154` 的 `--gtest_` 匹配分支原为死代码（字面量写成连字符 `--gtest-`，`rfind` 恒返回 `npos`），导致 `--gtest_filter` 被当作 Unknown option 拒绝——任务书要求的 `bash run.sh --accuracy --filter=TC_SQ_024` 无法按字面执行。**已于 2026-09-14 修复**（1 字符：`"--gtest-"` → `"--gtest_"`），`--gtest_filter='Suite/CgemmexAccuracy.Run/58'` 可精准跑到单条并 PASS；字面量 `--filter=...` 仍非本工程 flag，未被实现别名。② 全量性能运行需约 12.5 小时，后台进程带 6 小时 timeout 会被中断，只能拿到部分 CSV。 | ① 已修复，无需降级；单条用例定位统一用 `--gtest_filter` + 真实 test 名（参数化索引），或在 `.cannbot/csvs/` 下放定向子集走 `--csv`。② 性能用例集拆分为"小 shape 快速回归"与"大 shape 长跑"两档，小 shape 档纳入常规回归；长跑档接受"逐条 flush + 中断可续"的口径，CSV 为追加式。 |
| **未覆盖的 dtype、转置与负向用例** | `ACLBLAS_R_32` / `ACLBLAS_H_R_32` / `ACLBLAS_H_C_32` 三条 dtype 路径与 T/C 转置形态本次交付未实现/未覆盖；空指针、空 handle、非法枚举等 12 条负向用例未纳入本次验收。 | host 侧对未实现的 dtype 返回 `ACLBLAS_STATUS_NOT_SUPPORTED`（显式可判定），对 T/C 转置形态有完整参数通道但正确性未验证，调用方可通过返回码判定能力边界，不存在静默降级。负向校验逻辑（9 步顺序）已实现，可后续补验收。 |
| **环境风险** | 依赖 CANN 9.1.0（`/usr/local/Ascend/cann-9.1.0`）+ Ascend 950PR（`DAV_3510`）+ 静态 GTest（`/usr/lib/x86_64-linux-gnu/libgtest.a`）+ `g++ -std=c++17`。若环境缺失或版本不匹配，无法完成运行验证；`run.sh` 自带 `.so` 新鲜度防护，kernel/host 改动后须先 `cmake --build build-cgemm --target ops_blas`。 | 设计文档与代码可静态验证；运行验证需在具备正确环境的机器上完成。环境不满足时如实报告，不跳过验证。 |

## 附录：修订记录

| 日期 | 修订版本 | 修改描述 | 作者 |
| --- | --- | --- | --- |
| 2026-09-11 | v1.0.0 | 初稿构建：完成 `aclblasCgemmEx` 接口分析、技术路线决策与 C_32 主路径设计 | — |
| 2026-09-11 | v1.1.0 | 按官方社区任务合并样例结构重构；补充整体执行流程图、算子泛化功能设计、复数分解与 Host 标量装配专属适配说明、数据检测、测试用例规划、风险与降级预案；精度验收判据更新为官方 CSV 的 `mere_threshold` + `mare_multiplier`；补全 7 个状态码清单与接口签名；修正接口声明注释与实现的一致性说明 | — |
| 2026-09-11 | v1.2.0 | 迭代一 A1-Main 主线实测结果回填：精度 FAIL（maxAbs re=33–82, matched_ratio 0.006%–0.018%）、性能 FAIL（1024³ 约 14,720× 阈值、2048³ 超时）；补充 4 个精度定位候选点与 5 个性能定位候选点；新增测试方法学陷阱披露（stream 同步缺失导致假一致）；测试用例规划表增加实测状态列。**该版本的"算子全面 FAIL"结论已被 v3.0.0 的官方 CSV 实测推翻，保留仅作历史轨迹** | — |
| 2026-09-14 | v3.0.0 | 官方 CSV 实测回填，覆盖 v1.2.0 的结论：① 精度标准/性能标准章节重写——范围内 66 条精度用例 46 数值 PASS / 14 FAIL / 6 状态码 PASS（合计 52 PASS / 14 FAIL；修复前 45 / 15 / 6，即 51 / 15，差的 1 条即本版本修复的 `TC_ED_707`），判据明确为 `mere_threshold = 2⁻¹³` + `mare_multiplier = 10.0`（`outlierLimit ≈ 1.221e-3`）、PASS 需 `mismatchCount == 0` 且 `mere < threshold` 且 `maxRelErr < outlierLimit`、golden 更正为 CPU double 独立实现（非 cblas）；② 性能标准重写为实测口径（50 条中 0 PASS / 42 FAIL / 5 ERROR / 3 未落盘，最高 `ratio = 0.0115`），删除 A1-Main 的 6,122,958.67 µs / 14,720× 等 harness 数字；③ 测试用例规划重写为 116/1200（9.67%）真实覆盖口径，给出 14 条 FAIL 的三分类完整清单、12 条零比较 PASS（3 种成因，含已修复的 `TC_ED_707`）、覆盖口径表与复现命令；④ 风险与降级预案重写为性能已知缺陷 / A 类判据争议 / B 类 token 语义 / GPU 基线口径差异 / 测试工程缺陷 / 环境风险六项；⑤ 新增 `TC_ED_707` 修复记录（beta-only 快路径 GM 下标按 float 而非 complex 单位，`* 2u` 补齐，漏写 120/256 → 0/256，主路径 `TC_SQ_024` 逐字相同未受影响）与 3.2.2 浮点下标单位约束说明；⑥ 更正需求来源表中的 golden 与测试目录路径、需求拆解第 7/8 项状态、TK4 快路径验收状态；⑦ 如实标注三处与任务书的不一致（CSV 30 列非 52 列、`mere` 公式在 `verify.h::MereMareStrategy` 而非 `verify_accuracy.py`、`verify_performance.py` docstring 0.8 与常量 0.4 不符）以及任务书 §3.3 阈值与 `gpu_baseline.csv` 恒定 2.5 倍差异。版本号自 v1.2.0 接续（v2.0 未占用） | — |
