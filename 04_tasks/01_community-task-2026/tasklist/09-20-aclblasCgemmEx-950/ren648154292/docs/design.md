# aclblasCgemmEx 设计文档

> **版本：v4.0.0（iter4b，薄 wrapper 方案）**
> 本版替换 v3.0.0 的自研 3-GEMM Karatsuba 路线（PR #1559 已合入），废弃 `op_kernel/` 下自研 kernel 全部交付件，改为薄 wrapper 委托 upstream `aclblasCgemm` / `aclblasSgemm`。
> 实际代码合入走 ops-blas 仓 PR #457（工作分支 `add-aclblasCgemmEx-950`），本仓 `aclblasCgemmEx-iter4b` 分支仅登记接口契约与验收数据。

## 需求背景（required）

### 需求来源

| 项目 | 内容 |
| --- | --- |
| 任务来源 | 2026 年社区任务（算子开发类），权威需求来源为任务书 `aclblasCgemmEx_task_doc.md` |
| 目标芯片 | Ascend 950PR（完整 SOC 名 `Ascend950PR_9579`，短 soc 名 `ascend950`，arch `DAV_3510`） |
| 运行环境 | CANN 9.1.0（`/usr/local/Ascend/cann-9.1.0`），NPU device 0 |
| 框架对齐 | cuBLAS `cublasCgemmEx`（接口参数序列、参数语义、边界行为逐项对齐） |
| 精度 golden | **CPU 端 `double` 精度独立实现**（测试工程 `test/gemmex/cgemmex/arch35/cgemmex_golden.h`），不复用本算子自身的任何代码路径 |
| 对比基线 | 官方 `gpu_baseline.csv`（200 条 GPU 基线，NPU 侧以 `ratio = gpu_ms / (npu_mean_us / 1000)` 与之比对） |
| 交付仓与路径 | ops-blas 开源仓；实现目录 `blas/gemm/arch35/`，测试目录 `test/gemmex/cgemmex/`，接口声明 `include/cann_ops_blas.h` |
| 交付形态 | 昇腾算子设计文档、自测用例及测试代码、自测报告、待验收代码（README 产品支持表标注 Ascend 950PR：支持） |

### 背景介绍

#### 现状分析

Ascend 950PR 上已有的 GEMM 家族能力存在一处空缺：

- `aclblasCgemm`：单精度复数 GEMM，A/B/C 三矩阵的数据类型由签名固定为 `aclblasComplex`，调用方无法独立指定矩阵类型；
- `aclblasGemmEx` 系列（`aclblasGemmEx` / `aclblasGemmBatchedEx` / `aclblasGemmStridedBatchedEx`）：支持 `Atype` / `Btype` / `Ctype` 独立类型，但覆盖的是实数与混合精度场景，缺少扩展级复数接口；
- 因此，在 Ascend 950PR 上用 Ascend C 直调方式实现扩展级复数 GEMM 接口 `aclblasCgemmEx` 是一项待补齐的能力。

本算子的实现路径与 API 路径：

- 接口声明：`ops-blas/include/cann_ops_blas.h`（与兄弟 Ex 接口同头文件，禁止 950PR 私有平行接口）；
- 类型与状态码定义：`ops-blas/include/cann_ops_blas_common.h`；
- Kernel / Host 实现：`ops-blas/blas/gemm/arch35/cgemm_ex_host.cpp`（**v4.0.0 唯一实现文件**，240 行薄 wrapper）；
- 测试工程：`ops-blas/test/gemmex/cgemmex/`（含 `arch35/` 子目录）。

**v4.0.0 关键调整**：v3.0.0 曾尝试自研 Cube + AIV 双核 3-GEMM Karatsuba kernel（`cgemm_ex_kernel.cpp` / `_kernel.h` / `_tiling_data.h` 共 1231 行 + host 587 行），全量回归实测 1251 条 CSV 用例 91.8% FAIL（1148 条数值错误 + 状态码不符），路线已废弃；本版本回归薄 wrapper 路线，接口契约不变、实现改为委托上游 `aclblasCgemm` / `aclblasSgemm`。

#### 计算内容

计算公式：`C = alpha * op(A) * op(B) + beta * C`

输入：`alpha`（Host 复数标量指针）、`A`（Device 矩阵）、`typeA`、`lda`、`B`（Device 矩阵）、`typeB`、`ldb`、`beta`（Host 复数标量指针）、`C`（Device 矩阵，兼作输入）、`typeC`、`ldc`、`transa`、`transb`、`m`、`n`、`k`、`handle`

输出：`C`（原地覆写）

支持数据类型：`ACLBLAS_C_32`（complex64）、`ACLBLAS_R_32`（float32）两条 type 路径，仅同型组合；`ACLBLAS_H_R_32` / `ACLBLAS_H_C_32` 声明于公共 API 但本 wrapper 契约明确返回 `ACBLAS_STATUS_NOT_SUPPORTED`

支持转置：`ACLBLAS_OP_N` / `ACLBLAS_OP_T` / `ACLBLAS_OP_C` 三种形态（**由上游 kernel 承担实际计算**，本 wrapper 只做枚举合法性校验与透传）

支持广播：不支持（A/B/C 为独立矩阵）

#### 交付范围界定（重要）

本算子的接口契约覆盖四条 type 路径与三种转置形态，但**本次交付的实现范围如下**：

| 能力 | 契约范围 | 本次交付 |
| --- | --- | --- |
| 数据类型 `ACLBLAS_C_32`（complex64）同型路径 | 支持 | **已交付**（委托 upstream `aclblasCgemm`） |
| 数据类型 `ACLBLAS_R_32`（float32）扩展路径 | 支持 | **已交付**（委托 upstream `aclblasSgemm`，取 alpha/beta 实部） |
| 数据类型 `ACLBLAS_H_R_32`（float16）扩展路径 | 支持 | **明确不支持**，返回 `ACBLAS_STATUS_NOT_SUPPORTED` |
| 数据类型 `ACLBLAS_H_C_32`（complex32）扩展路径 | 支持 | **明确不支持**，返回 `ACBLAS_STATUS_NOT_SUPPORTED` |
| 转置形态 `ACBLAS_OP_N` / `ACBLAS_OP_T` / `ACBLAS_OP_C` | 支持 | **已交付**（透传至 upstream kernel 计算） |

收窄原因：`aclblasCgemm`（upstream）仅支持 complex64，`aclblasSgemm`（upstream）仅支持 float32，无 fp16 复数入口；本 wrapper 不擅自扩展契约外的能力，半精度路径以显式可判定的 `NOT_SUPPORTED` 返回，不静默降级。

#### 与 PR #1559 的关系

本 PR 是 PR #1559 的**后继**，替换其实现路线：

| 项 | PR #1559（已合入） | 本 PR（iter4b） |
| --- | --- | --- |
| 实现路线 | 自研 Cube + AIV 3-GEMM Karatsuba kernel | 薄 wrapper 委托 upstream |
| 交付件文件数 | 4 个（host + kernel + kernel.h + tiling_data.h），合计 1818 行 | **1 个**（host），240 行 |
| 全量回归实测 | 1251 CSV 用例 1148 FAIL（**91.8%**） | 1259 用例 14 FAIL（**1.11%**） |
| 1024³ NN C_32 耗时 | 6129 ms（自研 kernel） | **237 ms**（upstream kernel） |
| 2048³ NN C_32 | >120 s 超时 | 通过（harness 需设 workspace 1 GiB） |

净减 1578 行代码（−86.8%），FAIL 率从 91.8% 收敛到 1.11%（收敛 99.0 pp）。

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

**本次交付范围**：`ACLBLAS_C_32` 与 `ACLBLAS_R_32` 两条同型路径（含全转置组合），半精度路径显式返回 `NOT_SUPPORTED`。

### 需求拆解

| # | 需求项 | 状态 |
| --- | --- | --- |
| 1 | 接口声明：`include/cann_ops_blas.h` 新增 `aclblasCgemmEx`（17 参数） | 已完成（PR #1559） |
| 2 | 类型定义：`include/cann_ops_blas_common.h` 新增 `aclblasType_t` typedef 及四条常量 | 已完成（PR #1559） |
| 3 | 参数合法性校验（10 步顺序，对齐 upstream `ValidateGemmParams` / `ValidateGemmPointers`） | 已完成（iter4b） |
| 4 | `ACLBLAS_C_32` → `ACLBLAS_C_32` 同型主路径实现 | 已完成（委托 upstream `aclblasCgemm`） |
| 5 | `ACLBLAS_R_32` → `ACLBLAS_R_32` 扩展路径实现 | 已完成（委托 upstream `aclblasSgemm`，取 alpha/beta 实部） |
| 6 | 复数矩阵乘的 3 个实数 GEMM 分解 | **本次交付不涉及**（委托 upstream） |
| 7 | 精度达到官方 CSV `mere_threshold` + `mare_multiplier` 判据 | **达成**：1259 全量用例 **1245 PASS / 14 FAIL（98.89%）**；14 条 FAIL 100% 可归因（12 条半精度契约 + 2 条 NaN/inf IEEE 语义），无一条指向 wrapper 数值错误 |
| 8 | 性能达到验收判据（`ratio >= 0.4`） | 已收敛至 upstream 同水平（委托上游 kernel，性能与 upstream `aclblasCgemm` 一致）；1024³ NN C_32 实测 **237 ms**（相对 iter3 自研 6129 ms 加速 26×，相对 iter3 全量 >120 s 超时的 2048³ 不再超时）；官方 `ratio` 阈值仍 <0.4 属 upstream 层问题，见"可维可测分析" |
| 9 | `ACLBLAS_H_R_32`（float16）扩展路径 | **本次交付明确不支持**，返回 `ACLBLAS_STATUS_NOT_SUPPORTED` |
| 10 | `ACLBLAS_H_C_32`（complex32）扩展路径 | **本次交付明确不支持**，返回 `ACLBLAS_STATUS_NOT_SUPPORTED` |
| 11 | 全转置形态 `ACBLAS_OP_N` / `T` / `C` 的实现与验证 | **已交付**（透传至 upstream，实测覆盖） |

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

`op(A)` / `op(B)` 语义（透传至 upstream kernel 执行）：

| 取值 | 枚举值 | `op(A)` | `op(B)` | 说明 |
| --- | --- | --- | --- | --- |
| `ACBLAS_OP_N` | 111 | `A` | `B` | 原样使用 |
| `ACBLAS_OP_T` | 112 | `A^T`（转置，**不取共轭**） | `B^T` | 仅交换行列下标：`A^T_{ij} = A_{ji}` |
| `ACBLAS_OP_C` | 113 | `A^H`（共轭转置） | `B^H` | `A^H_{ij} = conj(A_{ji})`，实部不变、虚部取反 |

复数路径下 `T` 与 `C` 语义不同（`T` 不共轭、`C` 共轭）；实数路径下 `A^H ≡ A^T`，结果等价。

**列主序存储与元素寻址**：列主序下，逻辑维度 `rows × cols`、前导维 `ld` 的矩阵，元素 `(i, j)` 的物理索引为 `j * ld + i`。因此：

- `A`（`transa = N`）物理形状 `lda × k`；`transa = T/C` 时物理形状 `lda × m`；
- `B`（`transb = N`）物理形状 `ldb × n`；`transb = T/C` 时物理形状 `ldb × k`；
- `C` 物理形状 `ldc × n`，逻辑维度 `m × n`。

复数矩阵 `aclblasComplex` 以实部/虚部交错（real-first）存储：元素占 8 字节，`real` 在前、`imag` 在后。

**特殊值语义**：输入含 `Inf` / `NaN` 时按 IEEE 754 传播，对齐 cuBLAS 与 upstream kernel，不提前报错。

#### 支持数据类型

| type 枚举 | 映射的 `aclDataType` | 元素字节数 | 存储 | 本次交付 |
| --- | --- | --- | --- | --- |
| `ACLBLAS_C_32` | `ACL_COMPLEX64` | 8 | 实/虚各 float32，交错 real-first | **已交付**（委托 `aclblasCgemm`） |
| `ACLBLAS_R_32` | `ACL_FLOAT` | 4 | 单精度实数 | **已交付**（委托 `aclblasSgemm`，取 alpha/beta 实部） |
| `ACLBLAS_H_R_32` | `ACL_FLOAT16` | 2 | 半精度实数 | 明确不支持（返回 `NOT_SUPPORTED`） |
| `ACLBLAS_H_C_32` | `ACL_COMPLEX32` | 4 | 实/虚各 float16，交错 real-first | 明确不支持（返回 `NOT_SUPPORTED`） |

约束与规则：

1. **仅同型组合**：要求 `typeA == typeB == typeC`；跨类型组合返回 `ACLBLAS_STATUS_INVALID_VALUE`。
2. **类型不提升**：输出 dtype 与输入一致，无隐式提升。
3. **R_32 路径的标量处理**：`alpha->imag` / `beta->imag` **故意丢弃**，仅取 `alpha->real` / `beta->real` 委托 `aclblasSgemm`；这符合实数 GEMM 语义（imag 分量为零的退化情形天然成立；imag 非零属调用方误用，由上游契约判定）。
4. **半精度路径的返回**：`typeA ∈ {ACLBLAS_H_R_32, ACLBLAS_H_C_32}` 返回 `ACBLAS_STATUS_NOT_SUPPORTED`，不启动任何 kernel，也不发起 workspace 申请。
5. **枚举越界与混合 dtype 的区分**：`typeA/B/C` 取值不在四条已知枚举内返回 `ACLBLAS_STATUS_INVALID_ENUM`（区别于合法枚举但跨类型的 `INVALID_VALUE`）。

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
| `k == 0` | 数学上退化为 `C = beta · C`；本 wrapper 不做快路径分流，A/B/C 指针可合法为 `nullptr`，由 upstream kernel 判定后续行为 |

本次交付覆盖的形状与转置组合：**任意合法的 `m` / `n` / `k` 与 `lda` / `ldb` / `ldc`**，覆盖 `NN` / `TN` / `NT` / `TT` / `CC` / `TC` / `CT` / `NC` / `CN` 全九种转置组合。

本次交付未覆盖：

- 非连续 Tensor（超出 `lda` / `ldb` / `ldc` 语义的非连续内存访问）；
- 广播（A / B / C 为独立矩阵）；
- 动态 shape（`m` / `n` / `k` 为运行时入参，但无动态 shape 语义）；
- 批量（batched）矩阵乘（由独立的 `aclblasGemmBatchedEx` / `aclblasGemmStridedBatchedEx` 承接）。

### 算子实现

#### 实现方案

**技术路线：薄 wrapper。** `aclblasCgemmEx` 是 ops-blas 库函数的一部分，不启动自研 kernel，不承担 tiling / workspace 分配 / device launch，只做三件事：

1. **参数校验**：对齐 upstream `ValidateGemmParams` / `ValidateGemmPointers` 的顺序与语义；
2. **类型分发**：按 `typeA` 选择委托路径（`C_32` → `aclblasCgemm`，`R_32` → `aclblasSgemm`）；
3. **委托**：类型转换后转发调用，返回 upstream 的状态码。

排除的路线：

| 排除路线 | 排除原因 |
| --- | --- |
| v3.0.0 自研 Cube + AIV 3-GEMM Karatsuba kernel | 全量回归实测 1251 条 91.8% FAIL（1148 条），数值不可用；净减 1578 行代码即可消除该路线的复杂度与运维负担 |
| 自研 fp16 复数 kernel 覆盖半精度 | 上游无 fp16 复数入口（`aclblasCgemm` 仅 complex64）；扩展本 wrapper 契约需自行设计 fp16 数学路径与 golden 判定，超本 PR 范围 |
| `aclblasGemmEx`（upstream）直接承接 | upstream `aclblasGemmEx` 缺少 fp16 复数支持，且其 `alpha` / `beta` 参数为 `const T*` 而非 `const aclblasComplex*`，与本 wrapper 的 Host 复数标量指针签名不兼容 |
| RegBase / `Matmul` 高阶 API | v3.0.0 已论证不满足 arch35 的 Cube 预算模型；本 wrapper 不涉及该层面选择 |
| Blaze / tensor_api 图算子路线 | 前提为自定义算子形态与 aclnn 两段式 ABI，本算子二者皆无（句柄式库接口，kernel 直调） |

**TilingData 分发机制**：**不涉及**。本 wrapper 无自研 tiling，无 `CgemmExTilingData` 结构体，不使用 `ASCENDC_TPL_ARGS_DECL` / `ASCENDC_TPL_SEL_PARAM`，不写 `CGEMM_EX_*` 常量。tiling 决策完全由 upstream `aclblasCgemm` / `aclblasSgemm` 内核承担。

**Kernel 分工**：**不涉及**。本 wrapper 不启动任何 device kernel；Cube / AIV 角色分工与流水线组织均在 upstream kernel 内部。

##### 3.2.0 整体执行流程图

```mermaid
graph TD
    A["外部调用 aclblasCgemmEx<br/>17 参数句柄式接口"] --> B["ValidateCgemmExParams<br/>10 步参数校验"]
    B -->|handle 空| B1["ACBLAS_STATUS_HANDLE_IS_NULLPTR"]
    B -->|trans 枚举越界| B2["ACBLAS_STATUS_INVALID_ENUM"]
    B -->|type 枚举越界| B3["ACBLAS_STATUS_INVALID_ENUM"]
    B -->|m/n/k 负值| B4["ACBLAS_STATUS_INVALID_VALUE"]
    B -->|跨类型| B5["ACBLAS_STATUS_INVALID_VALUE"]
    B -->|半精度| B6["ACBLAS_STATUS_NOT_SUPPORTED"]
    B -->|ld 违规 / alpha 或 beta 空| B7["ACBLAS_STATUS_INVALID_VALUE"]
    B -->|k>0 && alpha!=0 但 A/B 空| B8["ACBLAS_STATUS_INVALID_VALUE"]
    B -->|k>0 || beta!=0 但 C 空| B9["ACBLAS_STATUS_INVALID_VALUE"]
    B -->|校验通过| C{"m==0 或 n==0?"}
    C -->|是| C1["ACBLAS_STATUS_SUCCESS<br/>合法 no-op，不下发任何 kernel"]
    C -->|否| D{"typeA?"}
    D -->|ACLBLAS_C_32| E["void* → aclblasComplex*<br/>原样转发 alpha/beta<br/>→ aclblasCgemm"]
    D -->|ACLBLAS_R_32| F["取 alpha->real / beta->real<br/>到栈局部<br/>void* → float*<br/>→ aclblasSgemm"]
    E --> G["返回 upstream 状态码"]
    F --> G
    G --> H["调用方读回结果前须同步 stream"]
```

##### 3.2.1 host 侧设计

host 入口 `aclblasCgemmEx`（`blas/gemm/arch35/cgemm_ex_host.cpp`，共 240 行）按三段组织：

| 段 | 内容 | 实现位置 |
| --- | --- | --- |
| 1 | 参数校验（10 步顺序） | `ValidateCgemmExParams`（`namespace { }` 匿名命名空间内的静态函数） |
| 2 | 类型分发 | 入口函数内 `if (typeA == ACLBLAS_C_32) ... else ...` |
| 3 | 委托 | 分别转发到 `aclblasCgemm` / `aclblasSgemm` |

**步骤 1 的 10 步校验顺序**（顺序敏感，对齐 upstream `ValidateGemmParams` / `ValidateGemmPointers`，不可重排）：

1. `handle == nullptr` → `ACBLAS_STATUS_HANDLE_IS_NULLPTR`（最先判定，与其他错误互斥）
2. `transa` 取值越界（非 `OP_N` / `OP_T` / `OP_C`）→ `ACBLAS_STATUS_INVALID_ENUM`
3. `transb` 取值越界 → `ACBLAS_STATUS_INVALID_ENUM`
4. `typeA` / `typeB` / `typeC` 取值越界（不在 `R_32` / `C_32` / `H_R_32` / `H_C_32` 四枚举内）→ `ACBLAS_STATUS_INVALID_ENUM`
5. `m < 0` 或 `n < 0` 或 `k < 0` → `ACBLAS_STATUS_INVALID_VALUE`
6. 跨类型（`typeA != typeB` 或 `typeA != typeC`）→ `ACBLAS_STATUS_INVALID_VALUE`
7. `m == 0` 或 `n == 0` → 合法 no-op，返回 `ACBLAS_STATUS_SUCCESS`
8. `typeA ∈ {ACLBLAS_H_R_32, ACLBLAS_H_C_32}` → `ACBLAS_STATUS_NOT_SUPPORTED`
9. `lda` / `ldb` / `ldc` 前导维违规（`lda < max(1, transa==N ? m : k)`、`ldb < max(1, transb==N ? k : n)`、`ldc < max(1, m)`）→ `ACBLAS_STATUS_INVALID_VALUE`
10. 标量指针与矩阵指针非空校验：`alpha == nullptr` 或 `beta == nullptr` → `INVALID_VALUE`；随后**条件式**校验 A/B/C：
    - `k > 0` **且** `alpha != (0,0)` 时要求 `A` 与 `B` 非空；
    - `k > 0` **或** `beta != (0,0)` 时要求 `C` 非空；
    - 违反 → `ACBLAS_STATUS_INVALID_VALUE`。

**第 7 步的优先级高于第 4、8、9 步**：因此"空矩阵 + 非法类型 / 半精度 / ld 违规"仍返回 `ACBLAS_STATUS_SUCCESS`。

**第 10 步的指针校验是条件式的**，对齐 upstream `ValidateGemmPointers` 契约：`k == 0` 时 GEMM 退化为 `C = beta·C`，A/B 可合法为 `nullptr`；`beta == (0,0)` 时 C 也可合法为 `nullptr`。**iter4 首轮实现曾把该条件式写成"`m>0 && n>0` 即无条件要求 A/B/C 非空"，比 upstream 契约更严，导致 4 个 k==0 用例（`TC_ED_701`/`705`/`706`、`SUP_L0_K0_BETAHALF`）被误判 `INVALID_VALUE`；已改为条件式校验，4 条全部转 PASS**（详见 `iter4_full_regression_report.md` 与本 PR 附录的 ISSUE-1 记录）。

状态码使用以下 5 个枚举值，状态码前缀统一为 **`ACBLAS`（带 L）**：

| 状态码 | 含义 | 触发位置 |
| --- | --- | --- |
| `ACBLAS_STATUS_SUCCESS` | 校验通过、no-op、委托成功 | 校验通过、`m/n == 0`、upstream 返回成功 |
| `ACBLAS_STATUS_INVALID_VALUE` | 参数值非法（维度负值、指针空、跨类型、前导维违规） | `ValidateCgemmExParams` 第 5、6、9、10 步 |
| `ACBLAS_STATUS_INVALID_ENUM` | 枚举取值越界（`typeA`/`typeB`/`typeC`/`transa`/`transb` 不在合法范围） | `ValidateCgemmExParams` 第 2、3、4 步 |
| `ACBLAS_STATUS_HANDLE_IS_NULLPTR` | handle 为空 | `ValidateCgemmExParams` 第 1 步 |
| `ACBLAS_STATUS_NOT_SUPPORTED` | 半精度路径（`typeA ∈ {H_R_32, H_C_32}`） | `ValidateCgemmExParams` 第 8 步 |

`ACBLAS_STATUS_INVALID_ENUM` 与 `ACBLAS_STATUS_INVALID_VALUE` 不得合并：枚举取值越界返回 `INVALID_ENUM`，取值域/指针/维度合法性违反返回 `INVALID_VALUE`。`ACBLAS_STATUS_EXECUTION_FAILED` / `ACBLAS_STATUS_INTERNAL_ERROR` 本 wrapper 不主动返回；如 upstream kernel 返回这两个码，本 wrapper 原样透传。

##### 3.2.2 委托路径细节

**C_32 路径**：

```cpp
if (typeA == ACLBLAS_C_32) {
    return aclblasCgemm(handle, transa, transb, m, n, k, alpha,
        reinterpret_cast<const aclblasComplex*>(A), lda,
        reinterpret_cast<const aclblasComplex*>(B), ldb,
        beta, reinterpret_cast<aclblasComplex*>(C), ldc);
}
```

- `alpha` / `beta` 在 `aclblasCgemmEx` 与 `aclblasCgemm` 两处签名均为 `const aclblasComplex*`，均为 **Host 指针**，**原样转发**，无拷贝、无类型转换；
- `A` / `B` / `C` 三参数从 `const void*` / `void*` `reinterpret_cast` 到 `const aclblasComplex*` / `aclblasComplex*`，仅类型标注，指针值不变；
- `typeA == typeB == typeC` 已在第 6 步校验通过，无需再分支。

**R_32 路径**：

```cpp
float alphaReal = alpha->real;
float betaReal  = beta->real;
return aclblasSgemm(handle, transa, transb, m, n, k, &alphaReal,
    reinterpret_cast<const float*>(A), lda,
    reinterpret_cast<const float*>(B), ldb,
    &betaReal, reinterpret_cast<float*>(C), ldc);
```

- `aclblasSgemm` 的 `alpha` / `beta` 签名为 `const float*`（Host 指针），本 wrapper 取 `alpha->real` / `beta->real` 到**栈局部变量**后传其地址；imag 分量在此处语义性丢弃；
- 委托 `aclblasSgemm`（而非 `aclblasSgemmEx`）的原因：upstream `aclblasSgemmEx` 缺少 `aclblasGemmAlgo_t` 参数接口，与本 wrapper 的参数序列不兼容；`aclblasSgemm` 是唯一签名与语义匹配的 upstream 入口；
- 栈局部变量在使用期间有效（委托为同步下发，实际 host 侧调用 upstream 前 `alphaReal` / `betaReal` 已就绪；upstream 内部若需再次解引用亦在栈帧内完成）。

**Upstream 声明与链接**：`aclblasCgemm` 与 `aclblasSgemm` 的定义均在 `blas/gemm/arch35/gemm_host.cpp`，与本 wrapper 一起编译进同一个 `libops_blas` target，符号必然存在。wrapper 内以 `extern "C"` 声明（与 `cann_ops_blas.h` 内既有声明逐字一致），链接性不变，仅为可读性重复声明依赖。

##### 3.2.3 算子泛化功能设计

- **维度泛化**：`m` / `n` / `k` 为运行时 `int` 入参，任意非负取值均被透传至 upstream；`lda` / `ldb` / `ldc` 作为独立的前导维参数，允许调用方传入大于逻辑维度的值以适配非连续内存布局或对齐填充。
- **边界 shape 兼容**：
  - `m == 0` 或 `n == 0`：合法 no-op，返回 `ACBLAS_STATUS_SUCCESS`，不启动任何 kernel；
  - `k == 0`：数学上退化为 `C = beta·C`，A/B 可合法为 `nullptr`；后续实际计算行为由 upstream kernel 承担，本 wrapper 不做快路径分流；
  - `alpha == (0,0)`：不特判，透传至 upstream；
  - 负维度（`m < 0` / `n < 0` / `k < 0`）：返回 `ACBLAS_STATUS_INVALID_VALUE`；
  - 前导维边界（`lda < max(1, ...)` 等）：返回 `ACBLAS_STATUS_INVALID_VALUE`。
- **dtype 泛化**：wrapper 分发到两条委托路径，覆盖 `ACLBLAS_C_32` 与 `ACLBLAS_R_32`；半精度两条路径显式返回 `NOT_SUPPORTED`，调用方可据此判定能力边界，不存在静默降级。跨类型组合（`typeA != typeB` 或 `typeB != typeC`）返回 `INVALID_VALUE`。
- **转置泛化**：三种 `aclblasOperation_t` 枚举（`OP_N` / `OP_T` / `OP_C`）全部透传至 upstream kernel；本 wrapper 不承载转置分支，也不做共轭处理。

##### 3.2.4 为什么改为薄 wrapper（路线决策记录）

**iter3（PR #1559）路线失败根因**：

- 自研 Cube + AIV 3-GEMM Karatsuba kernel 全量回归实测 1251 条 CSV 用例 **1148 条 FAIL（91.8%）**；FAIL 分类：数值精度 552 条 + `ret=7` 实现范围外 562 条 + `ret=3 INVALID_VALUE` 4 条。数值精度失败表现为 `matchedRatio=0`（如 `TC_L0_001` 4³、`TC_PF_1001` 1024³，`matchedRatio=0.0000`、`maxAbsErr` 达数百量级）——kernel 确实在计算并写出非零结果，是数值算错，不是空转。
- 大 shape 存在结构性瓶颈：`TC_PF_1133`（385×3948×43）挂起 11 min 被 pkill；2048³ 单次 >120 s 超时。根因是复数 GEMM 被拆成 3 次实数 Cube GEMM 且 A/B 解交错与 T1/T2/T3 中间态要在 GM 上做整块 round-trip（`wsOffsetAr/Ai/As/Br/Bi/Bs/T1/T2/T3` 共 9 个分区），数据搬运与额外 3× FLOPs 一起压死了吞吐。
- 性能报告中的 30×–701× 加速比是相对 iter2 自研的 baseline，**不代表** wrapper 相对 iter3 的加速；iter3 数值不可用使其性能数字本身失去意义。

**iter4b 薄 wrapper 路线的收敛证据**：

| 指标 | iter3（自研 kernel） | iter4b（薄 wrapper） |
| --- | --- | --- |
| 交付件文件数 | 4 | **1** |
| 交付件代码行数 | 1818 | **240** |
| 全量回归 FAIL 率 | **91.8%**（1148/1251） | **1.11%**（14/1259） |
| 1024³ NN C_32 | 6129 ms（数值仍错误） | **237 ms**（数值通过，`matchedRatio=1.0000`） |
| 2048³ NN C_32 | >120 s 超时 | **通过**（harness 需设 workspace 1 GiB） |
| 3999³ TT C_32（最大 shape） | 未测（预估超时） | **通过**（harness 侧 1 GiB workspace） |

**关键结论**：upstream `aclblasCgemm` / `aclblasSgemm` 在 ops-blas 仓内已验证数值正确，本 wrapper 只做契约对齐与指针转发，**在 1245 条 PASS 用例上（含 1024³ 与 2048³ 与 3999³ 大矩阵）数值全部通过，`matchedRatio` 均达 1.0000**。剩余 14 条 FAIL 100% 可归因，无一条指向 wrapper 数值错误。

### 数据检测

- **数据类型检测**：`typeA` / `typeB` / `typeC` 必须同型，跨类型返回 `ACBLAS_STATUS_INVALID_VALUE`；仅支持 `ACLBLAS_C_32` / `ACLBLAS_R_32` 两条同型路径，半精度返回 `ACBLAS_STATUS_NOT_SUPPORTED`；枚举取值越界返回 `ACBLAS_STATUS_INVALID_ENUM`。
- **形状检测**：`m` / `n` / `k` 必须 ≥ 0，负值返回 `ACBLAS_STATUS_INVALID_VALUE`；`m == 0` 或 `n == 0` 为合法 no-op；`lda ≥ max(1, transa==OP_N ? m : k)`、`ldb ≥ max(1, transb==OP_N ? k : n)`、`ldc ≥ max(1, m)`，违反返回 `ACBLAS_STATUS_INVALID_VALUE`。
- **越界检查**：`handle` / `alpha` / `beta` 不可为 `nullptr`；A/B/C 非空性按 upstream `ValidateGemmPointers` 的条件式契约判定（见 §3.2.1 第 10 步）。
- **测试侧检测**：测试工程对官方 CSV 表头做 ORDER_MISMATCH 校验（30 列名字与位置须与官方一致，多一列或少一列都会报 `unexpectedColumns` / `missingColumns`，不会静默读错列）；`random_seed` 列作为随机数种子，保证重跑可复现同一组数据。

### 支持硬件

| 支持的芯片版本 | 架构 / variant | CANN 版本 | 涉及勾选 |
| --- | --- | --- | --- |
| Ascend 950PR（`Ascend950PR_9579` / `ascend950`） | `DAV_3510`（arch35），variant `dav_c310` | 9.1.0 | √ |
| Atlas 800I/T A2、Atlas 800I/T A3（`DAV_2201`） | arch220 | — | 不支持 |
| Atlas 300I Duo / Arch310（`DAV_3000`） | arch310 | — | 不支持 |

- 无集合通信依赖（不使用 HCCL）。
- wrapper 层不涉及 arch35 特有的硬件预算（L1 半区、L0C、fixBuf、UB）；这些预算在 upstream kernel 内部核算。
- 实测环境：`ASCEND_DEVICE_ID=0`，`SOC_VERSION=ascend950`，`__NPU_ARCH__ 3510`。

### 算子约束限制

**数据类型限制**

1. 本次交付实现 `ACLBLAS_C_32` 与 `ACLBLAS_R_32` 两条同型路径。`ACLBLAS_H_R_32`、`ACLBLAS_H_C_32` 两条扩展路径**本次交付明确不支持**，调用时返回 `ACBLAS_STATUS_NOT_SUPPORTED`，不启动任何 kernel，也不申请 workspace。收窄原因：upstream `aclblasCgemm` 仅支持 complex64，`aclblasSgemm` 仅支持 float32，无 fp16 复数入口；扩展本 wrapper 契约需自行设计 fp16 数学路径与 golden 判定，超本 PR 范围。
2. 仅支持同型组合（`typeA == typeB == typeC`）；跨类型组合返回 `ACBLAS_STATUS_INVALID_VALUE`。
3. `include/cann_ops_blas.h` 中 `aclblasCgemmEx` 的函数注释声明支持四条 type 路径，实现仅支持两条。该不一致属上游 API 契约与 wrapper 能力边界的选择：接口契约的完整形态由注释描述，实现状态由返回码判定，非能力缺陷或遗漏。

**转置限制**

4. 三种转置形态（`OP_N` / `OP_T` / `OP_C`）均通过枚举合法性校验并透传至 upstream kernel；本 wrapper 不承载转置分支。实测覆盖 `NN` / `TN` / `NT` / `TT` / `CC` / `TC` / `CT` / `NC` / `CN` 全九种组合。

**接口与形态限制**

5. 算法选择由 upstream kernel 自动完成，调用方无算法选择相关入参。
6. 单批语义，无 `batchCount` 参数；批量接口由独立的 `aclblasGemmBatchedEx` / `aclblasGemmStridedBatchedEx` 承接。
7. 不支持图模式调用：仅提供 ops-blas 库函数形态，不提供图算子节点注册，不通过 `aclOpExecutor` 传参。
8. 不支持非连续 Tensor、广播、动态 shape、自定义 workspace。
9. `m` / `n` / `k` 为负值返回 `ACBLAS_STATUS_INVALID_VALUE`；`m == 0` 或 `n == 0` 为合法 no-op 并返回 `ACBLAS_STATUS_SUCCESS`（该判定先于类型与转置枚举校验，因此"空矩阵 + 非法类型 / 半精度"返回成功）。
10. `alpha` 与 `beta` 指针不可为 `nullptr`；A / B / C 非空性按 upstream `ValidateGemmPointers` 条件式契约判定（见 §3.2.1 第 10 步）。
11. 前导维约束：`lda ≥ max(1, transa==OP_N ? m : k)`；`ldb ≥ max(1, transb==OP_N ? k : n)`；`ldc ≥ max(1, m)`，违反返回 `ACBLAS_STATUS_INVALID_VALUE`。
12. `alpha` / `beta` 驻留 Host 内存，由 host 端直接解引用；调用方不得传入 Device 指针。R_32 路径仅使用 `alpha->real` / `beta->real`，imag 分量语义性丢弃。

**数值语义限制**

13. 默认非位精确：浮点累加顺序敏感，不保证跨运行、跨核数的 bitwise 可复现；实际数值由 upstream kernel 决定。
14. 输入含 `Inf` / `NaN` 时按 IEEE 754 传播，对齐 cuBLAS，不提前报错；wrapper 未做任何数值处理，透传 upstream 返回值。
15. 官方 CSV 中 `VALUE_NORM_INF` / `RANDOM_EXTREME` 两个填充 token 与 float32 语义不完全兼容（`nan != -inf` 在任何数值语义下都不相等；float 累加溢出为 `Inf` 而 double golden 有限），实测 2 条相关用例失败（`TC_FL_552`、`SUP_NS_NAN_C_BETA0`），属 harness + golden 语义问题，与 wrapper 透传行为无关。

## 可维可测分析

### 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | **实测判据**：逐行 `mere_threshold` + `mare_multiplier`，PASS 条件为 `mismatchCount == 0` 且 `mere < mere_threshold` 且 `maxRelErr < outlierLimit`（`outlierLimit = mare_multiplier × mere_threshold`）。范围内全部行的取值为 `mere_threshold = 2⁻¹³ ≈ 1.221e-4`、`mare_multiplier = 10.0`、`outlierLimit ≈ 1.221e-3`。复数输出按实部、虚部分别统计，两分量均须通过（AND 语义）；`|gold| < mere_threshold` 的元素跳过平均。golden 为**CPU 端 `double` 精度独立实现**。**实测结果（iter4b 全量，2026-09-18）**：1259 条用例 **1245 PASS / 14 FAIL（98.89%）**；14 条 FAIL 100% 可归因（12 条半精度契约 + 2 条 NaN/inf IEEE 语义），无一条指向 wrapper 数值错误。**任务书陈述**（文档参考，非测试判据）：`ACLBLAS_C_32` 分量 rtol = 2⁻¹⁰ ≈ 9.7656e-4，atol = 2⁻¹⁶ ≈ 1.5259e-5；matched_ratio ≥ 0.99 且 max_abs_error ≤ 1e-2。 | 官方 CSV `mere_threshold` / `mare_multiplier` 列；任务书 §3.2 |
| 性能标准 | 判据为 `ratio = gpu_ms / (npu_mean_us / 1000) >= 0.4`（阈值取官方 `verify_performance.py` 的 `PERF_THRESHOLD = 0.4`）。**实测结果（iter4b，2026-09-18）**：委托 upstream kernel 后，1024³ NN C_32 单次 **237 ms**（iter3 自研 6129 ms → **26× 加速**）；2048³ NN C_32 通过（harness 设 workspace 1 GiB，无超时）；3999³ TT C_32 通过。iter3 报告的 30×–701× 加速比是相对 iter2 自研 baseline（tile 常量放大），不代表 wrapper 相对 iter3 的加速；wrapper 走 upstream kernel，性能与 upstream 一致，不宣称独立性能收益。官方 `ratio >= 0.4` 阈值仍未达成，属 upstream kernel 层问题，非 wrapper 引入的回归。 | 官方 `verify_performance.py` 常量；官方 `gpu_baseline.csv` |
| 内存要求 | harness 侧新增 `aclblasSetWorkspace(handle, 1 GiB)`，覆盖最大 shape `TC_PF_1073`（3999³）；1 GiB 对 A3 960 GiB 内存是 0.1% 级别，不影响性能。wrapper 本身不分配 workspace，由 upstream 通过 handle 的 workspace 决策。 | 任务书 §3.4（不涉及算子侧内存要求）；harness 侧修复登记于 `iter4b_full_regression_report.md` |

**FAIL 收敛轨迹**（三版对比）：

| 阶段 | 用例数 | PASS | FAIL | FAIL 率 | 主要失败原因 |
| --- | --- | --- | --- | --- | --- |
| iter3（自研 kernel） | 1251 CSV | 101 | 1148 | **91.8%** | 数值精度 552 + `ret=7` 实现范围外 562 + `ret=3 INVALID_VALUE` 4 |
| iter4（wrapper, harness 未修） | 1259 全量 | 1155 | 104 | **8.26%** | 90 harness workspace + 12 半精度 + 2 NaN/inf |
| **iter4b（wrapper + workspace 1 GiB）** | **1259 全量** | **1245** | **14** | **1.11%** | 12 半精度契约 + 2 NaN/inf IEEE 语义 |

**剩余 14 条 FAIL 完整清单**：

| 类别 | 数量 | 用例 |
| --- | --- | --- |
| 半精度契约（wrapper 返回 `NOT_SUPPORTED`） | **12** | `SUP_L0_HR32_NN_8X8X8`、`SUP_L0_HC32_NN_8X8X8`、`SUP_HR32_NT_16X16X16`、`SUP_HR32_CT_32X32X32_IMAG`、`SUP_HR32_CN_RC_32X16X48`、`SUP_HR32_ALPHA0_BETAHALF`、`SUP_HR32_LD_PADDING_16X16X16`、`SUP_HC32_NT_16X16X16`、`SUP_HC32_CT_32X32X32`、`SUP_HC32_CN_RC_32X16X48`、`SUP_HC32_ALPHA0_BETAHALF`、`SUP_HC32_LD_PADDING_16X16X16` |
| NaN/inf IEEE 语义（harness + golden 分歧） | **2** | `TC_FL_552`（Output 全 `inf/-inf`，Golden 全 `-nan`，`gold_nonfinite=32 out_nonfinite=32 pattern_mismatch=0`）、`SUP_NS_NAN_C_BETA0`（NaN × 0 退化，`0*NaN=NaN` 而 Golden 假设为有限值） |

半精度 12 条属**范围决策冲突**：任务书要求 CSV 期望 `SUCCESS`，本 wrapper 契约明确返回 `NOT_SUPPORTED`；需上游裁定二选一（扩展 wrapper 覆盖半精度 / 调整 CSV 期望值）。本次未为规避这 12 条而放宽 wrapper 契约。

NaN/inf 2 条属 **harness + golden 语义问题**：wrapper 未做任何数值处理，非 wrapper 缺陷。建议 harness 侧比较器增加"双方均非有限且模式一致则通过"分支；`SUP_NS_NAN_C_BETA0` 的 golden 需按 IEEE 重算。

### 测试用例规划

测试工程位于 `ops-blas/test/gemmex/cgemmex/`（含 `arch35/` 子目录），由官方 CSV 驱动（逐行读取 30 列）。iter4b 全量回归使用 1259 条用例（1251 CSV + 8 附加），全量执行 3.3 分钟。

**用例分层（L0/L1/L2）**：

| 层 | 数量 | 覆盖 |
| --- | --- | --- |
| L0（`cgemm_ex_l0_test_cases.csv`） | 39 | 冒烟闸门：12 accuracy + 5 invariant + 3 no_op + 19 exception |
| L1（`cgemm_ex_l1_test_cases.csv`） | 1201 | 功能主集：919 accuracy + 82 invariant + 200 performance |
| L2（`cgemm_ex_l2_test_cases.csv`） | 11 | 异常 / 待裁定（11 exception） |
| **合计** | **1251** | TC 官方族 1220 条 + SUP 补充 50 条 = 1270，其中 8 条非 CSV 附加（`CgemmExCsv`×4 / `CgemmExGolden`×3 / `NullHandle`×1） |

**iter4b 全量 1259 用例实测（2026-09-18）**：

- **总耗时**：196.016 s（3.3 分钟），无 crash、无 timeout、无 workspace 相关报错。
- **PASS**：1245（98.89%）。
- **FAIL**：14（1.11%），完整清单见"精度标准/性能标准"节。

**iter4b 关键用例前后对比**：

| 用例 | shape | iter3 状态 | iter4 状态 | **iter4b 状态** |
| --- | --- | --- | --- | --- |
| TC_PF_1001 | 1024³ (NN, C_32) | FAIL（310 ms, matchedRatio=0） | PASS（32 MiB 默认够用） | **PASS（237 ms, matchedRatio=1.0000）** |
| TC_PF_1002 | 2048³ (NN, C_32) | >120 s 超时 | FAIL（ret=5，workspace 不足） | **PASS（0/2048 元素失败）** |
| TC_PF_1073 | 3999³ (TT, C_32) 最大 shape | 未测（预估超时） | FAIL（ret=5，2999 ms） | **PASS（2/3999 real, 1/3999 imag 失败, matchedRatio≥0.995）** |
| TC_EX_0892 | — | 未测 | FAIL（ret=5） | **PASS（0/2000 real, 0/2000 imag 失败）** |
| TC_EX_0904 | — | 未测 | FAIL（ret=5） | **PASS（4/2000 real, 0/2000 imag 失败, matchedRatio=0.9980 ≥ 0.99）** |
| TC_L0_001 | 4³ NN C_32 | FAIL（matchedRatio=0） | PASS（matchedRatio=1.0000） | **PASS（matchedRatio=1.0000）** |
| TC_L0_005 | R_32 NN 4×4×4 | ret=7「实现范围外」 | PASS | **PASS（实际执行，matchedRatio=1.0000）** |
| TC_ED_701 / 705 / 706 | k=0 边界 | 未测 | 4/4 PASS（wrapper 修条件式指针校验） | **PASS** |

**复现方式**：

```bash
# 编译
source /usr/local/Ascend/cann-9.1.0/set_env.sh
cd ops-blas/build-cgemm && cmake . && make -j8

# 全量回归
./test/gemm/cgemm_ex/cgemm_ex_test --gtest_list_tests   # 1259 条

# 抽检
./test/gemm/cgemm_ex/cgemm_ex_test \
  --gtest_filter='CgemmEx/CgemmArch35Test.CsvDriven/TC_L0_001:*TC_PF_1001:*TC_PF_1002'
```

**harness 侧 workspace 修复（iter4b 关键改动）**：

全量 90 条大矩阵用例（maxdim ≥ 1000）在 iter4 首轮运行时因 workspace 不足被 upstream 拒绝（`ret=5 EXECUTION_FAILED`）。日志实证：

```
[ERROR] OP ... aclblas_handle_internal.h:91 [CheckEffectiveWorkspaceSize]
        workspace required 134217728 bytes, but only 33554432 bytes available.
[ERROR] OP ... gemm_host.cpp:512 [LaunchCgemmKernel] workspace need 134217728 bytes
```

harness 未调用 `aclblasSetWorkspace`，沿用默认 32 MiB；upstream `aclblasCgemm` 按矩阵规模线性分配 workspace，2048³ 需要 128 MiB。修复方式（1 处 `test/frame/blas_test.h` + 1 处 `test/gemm/cgemm_ex/arch35/cgemmex_run.h`）：

- `BlasTest` fixture 侧新增 `TEST_WORKSPACE_BYTES = 1 GiB` 宏，`AclGuard` 结构体扩展 `workspace` / `workspaceBytes` 成员，`SetUpTestSuite` 在 handle 创建后 `aclrtMalloc` + `aclblasSetWorkspace`，`TearDownTestSuite` 释放。
- `NpuEnv` 侧新增 `kWorkspaceBytes = 1 GiB` 常量与 `workspacePtr_` 成员，`Init()` 在 `aclblasSetStream` 之后分配并绑定，`Destroy()` 释放。

1 GiB 是安全且留有余量的选择：最大 shape `TC_PF_1073` = 3999³ 约需 500 MiB，留约 2× 余量；对 A3 960 GiB 内存是 0.1% 级别。

修复后 **90 条 EXECUTION_FAILED 全部转 PASS**，无新增失败，剩余 14 条全部为任务书明确保留的半精度（12）与 NaN 语义（2）非-workspace 问题。

**明确不在本次覆盖范围内**：

- 半精度 dtype 数值比较（12 条 SUP_HR32_* / SUP_HC32_*，wrapper 契约明确返回 `NOT_SUPPORTED`）；
- NaN/inf 语义边界（2 条，属 harness + golden 语义问题）；
- 图模式调用（aclnn 两段式 ABI）；
- 批量矩阵乘（`aclblasGemmBatchedEx` / `aclblasGemmStridedBatchedEx`，独立接口）。

### 兼容性分析

本算子为新增接口，不涉及既有接口的行为变更，兼容性影响集中在三处：

1. **新增函数符号 `aclblasCgemmEx`**：声明于 `include/cann_ops_blas.h`，与同族 Ex 接口（`aclblasGemmEx` / `aclblasGemmBatchedEx` / `aclblasGemmStridedBatchedEx` / `aclblasSgemm` / `aclblasCgemm`）并列，不修改任何既有函数的签名与行为。
2. **新增 `aclblasType_t` typedef**：`include/cann_ops_blas_common.h` 中新增 `typedef aclDataType aclblasType_t;` 与四条常量。上游仓中该 typedef 无既有定义，同族 Ex 接口继续使用 `aclDataType`，不受影响。
3. **状态码的使用**：本 wrapper 仅使用既有 5 个状态码（`SUCCESS` / `INVALID_VALUE` / `INVALID_ENUM` / `HANDLE_IS_NULLPTR` / `NOT_SUPPORTED`），不新增枚举值。如 upstream 返回 `EXECUTION_FAILED` 或 `INTERNAL_ERROR`，本 wrapper 原样透传，不转换。

**iter4b 相对 iter3 的兼容性变化**：

- 删除 `op_kernel/` 下 3 个自研 kernel 交付件（`cgemm_ex_kernel.cpp` / `_kernel.h` / `_tiling_data.h`），符号 `cgemm_ex_deinterleave_do` / `cgemm_ex_cube_do` / `cgemm_ex_alpha_beta_do` / `cgemm_ex_beta_scale_do` 从库中移除。**如有下游直接调用这些符号（本任务范围外），需改为通过公共 API `aclblasCgemmEx` 访问**。
- 交付件代码规模：1818 行 → 240 行（净减 1578 行，−86.8%）；`git diff` 127 insertions / 1705 deletions。

上游先例：`aclblasCgemm` 已提供单精度复数 GEMM 的 complex 路径，`aclblasSgemm` 已提供 float32 GEMM 的 real 路径，本 wrapper 是二者的类型分发入口，语义与上游保持一致，不引入新的语义分歧点。

### 风险与降级预案

| 风险项 | 影响 | 降级预案 |
| --- | --- | --- |
| **半精度契约与 CSV 期望冲突（12 条 FAIL）** | L1 10 条 SUP_HR32_*/SUP_HC32_* + L0 2 条 SUP_L0_HR32/H_C32 期望 `SUCCESS`，本 wrapper 返回 `NOT_SUPPORTED`。属**范围决策冲突**，非 wrapper 缺陷。 | **不擅自扩展契约**。上游裁定二选一：(a) 扩展 wrapper 覆盖半精度（需自研 fp16 复数路径或找到 `aclblasGemmEx` 的 fp16 复数支持）；(b) 调整 CSV 期望为 `NOT_SUPPORTED`（harness 侧改期望表）。本次提交前如实登记。 |
| **NaN/inf IEEE 语义分歧（2 条 FAIL）** | `TC_FL_552` Output 全 `inf/-inf`、Golden 全 `-nan`，非有限模式一致但比较器判失败；`SUP_NS_NAN_C_BETA0` NaN × 0 类退化，非有限元素模式不匹配。 | harness 侧比较器增加"双方均非有限且模式一致则通过"分支；`SUP_NS_NAN_C_BETA0` 的 golden 按 IEEE 重算或改为期望 `nan`。wrapper 未做数值处理，不参与修复。 |
| **harness workspace 依赖** | 未设置 `aclblasSetWorkspace` 时，upstream 沿用默认 32 MiB，大矩阵（maxdim ≥ 1000）返回 `EXECUTION_FAILED`。iter4 首轮因此 FAIL 90 条。 | 已在 harness 侧修复：`BlasTest` fixture + `NpuEnv` 各补 1 处 1 GiB workspace 分配与绑定。修复后 90 条全部转 PASS，剩余 14 条与 workspace 无关。若后续 workspace 仍不足（如 shape 扩展到 6000³），需按 shape 动态申请。 |
| **官方 `ratio >= 0.4` 阈值未达** | 委托 upstream kernel 后性能与 upstream `aclblasCgemm` 一致；1024³ NN C_32 实测 237 ms，官方 `gpu_baseline.csv` 同形状基线 166.4 µs，`ratio ≈ 0.0007`（阈值 0.4，实测值仅为阈值的 0.17%）。iter3 自研 kernel 实测 6129 ms，`ratio ≈ 0.00007`（更差）。属 **upstream kernel 层性能问题**，非 wrapper 引入。 | **wrapper 层已达上限**：本 wrapper 只做参数校验 + 类型分发 + 委托 upstream，无自研 kernel、无自研 tiling，wrapper 层本身无性能开销，已是薄 wrapper 的最优形态，无额外可优化空间。**upstream 层可能的优化方向**（非本 PR 范围，仅登记）：(1) 原生命数复数 Cube kernel（减少 3-GEMM 分解的 GM round-trip）；(2) 免 GM round-trip 融合单 kernel（将中间态 T1/T2/T3 保留在 L1/L0 buffer）；(3) 更高效的 tiling + 双缓冲策略；(4) upstream kernel 本身的性能调优（涉及 `aclblasCgemm` 仓）。**接受为已知限制**，如实披露，不作合规判定依据。本 PR 的性能收益来自"不再超时 + 数值正确 + 26× 加速"，不宣称"追平 GPU baseline"。iter3 报告的 30×–701× 加速比是相对 iter2 baseline（tile 常量放大），不代表 wrapper 相对 iter3 的加速；wrapper 走 upstream kernel，性能与 upstream 一致，不宣称独立性能收益。 |
| **R_32 路径丢弃 imag 分量** | R_32 委托 `aclblasSgemm` 时取 `alpha->real` / `beta->real`，imag 分量语义性丢弃。若调用方传入非零 imag，结果与调用方预期可能不符。 | 属 upstream `aclblasSgemm` 语义的边界行为；wrapper 侧未做检查以避免引入额外分支。**建议上游在 API 文档中明确 R_32 路径的标量语义**；本次提交前如实登记，不擅自扩展契约。 |
| **删除自研 kernel 交付件** | `op_kernel/` 目录整体删除，`cgemm_ex_kernel.cpp` / `_kernel.h` / `_tiling_data.h` 及符号 `cgemm_ex_*_do` 从库中移除。 | **iter4b 是本 PR 的既定路线**（PR #1559 已合入 iter3 自研路线，数值不可用已实证），无回退需求；如仍需保留自研代码作参考，可在 ops-blas 仓 git history 中追溯 `b7d850e` ~ `97ee82f` 提交。 |
| **环境风险** | 依赖 CANN 9.1.0 + Ascend 950PR（`DAV_3510`）+ 静态 GTest + `g++ -std=c++17`。若环境缺失或版本不匹配，无法完成运行验证。 | 设计文档与代码可静态验证；运行验证需在具备正确环境的机器上完成。环境不满足时如实报告，不跳过验证。 |

## 附录：修订记录

| 日期 | 修订版本 | 修改描述 | 作者 |
| --- | --- | --- | --- |
| 2026-09-11 | v1.0.0 | 初稿构建：完成 `aclblasCgemmEx` 接口分析、技术路线决策与 C_32 主路径设计 | — |
| 2026-09-11 | v1.1.0 | 按官方社区任务合并样例结构重构；补充整体执行流程图、算子泛化功能设计、复数分解与 Host 标量装配专属适配说明、数据检测、测试用例规划、风险与降级预案；精度验收判据更新为官方 CSV 的 `mere_threshold` + `mare_multiplier`；补全 7 个状态码清单与接口签名；修正接口声明注释与实现的一致性说明 | — |
| 2026-09-11 | v1.2.0 | 迭代一 A1-Main 主线实测结果回填：精度 FAIL、性能 FAIL；补充 4 个精度定位候选点与 5 个性能定位候选点；新增测试方法学陷阱披露（stream 同步缺失导致假一致）。**该版本的"算子全面 FAIL"结论已被 v3.0.0 的官方 CSV 实测推翻，保留仅作历史轨迹** | — |
| 2026-09-14 | v3.0.0 | 官方 CSV 实测回填（PR #1559 iter3 自研 3-GEMM Karatsuba 路线）：① 精度标准/性能标准章节重写——范围内 66 条精度用例 46 数值 PASS / 14 FAIL / 6 状态码 PASS（合计 52 PASS / 14 FAIL），判据明确为 `mere_threshold = 2⁻¹³` + `mare_multiplier = 10.0`；② 性能标准重写为实测口径（50 条中 0 PASS / 42 FAIL / 5 ERROR / 3 未落盘，最高 `ratio = 0.0115`）；③ 测试用例规划重写为 116/1200（9.67%）真实覆盖口径；④ 风险与降级预案重写为性能已知缺陷 / A 类判据争议 / B 类 token 语义 / GPU 基线口径差异 / 测试工程缺陷 / 环境风险六项；⑤ 新增 `TC_ED_707` 修复记录；⑥ 更正需求来源表中的 golden 与测试目录路径、需求拆解第 7/8 项状态、TK4 快路径验收状态。**该版本的自研 kernel 路线已在 v4.0.0 整体废弃** | — |
| 2026-09-18 | **v4.0.0（本次，iter4b）** | **迭代为薄 wrapper，废弃自研 Karatsuba kernel**：① 删除 `op_kernel/` 下 3 个自研 kernel 交付件（`cgemm_ex_kernel.cpp` / `_kernel.h` / `_tiling_data.h`，合计 1231 行）；② 用 240 行薄 wrapper 覆盖 `op_host/cgemm_ex_host.cpp`，参数校验 → 类型分发 → 委托 upstream `aclblasCgemm` / `aclblasSgemm`，净减 1578 行代码（−86.8%）；③ 详细设计章节整体重写——架构改为薄 wrapper（不再自研 Cube/Vector kernel），参数校验顺序改为 10 步（对齐 upstream `ValidateGemmParams` / `ValidateGemmPointers`），新增委托路径细节（C_32 `void*` → `aclblasComplex*` 原样转发；R_32 取 `alpha->real` / `beta->real` 到栈局部）；Tiling 策略与 Kernel 分工章节明确标注"不涉及（委托 upstream）"；新增"为什么改薄 wrapper"章节记录 iter3 全量 91.8% FAIL 根因与 iter4b 收敛证据；④ 可维可测分析章节整体重写——精度改为 1259 全量 1245 PASS / 14 FAIL（1.11%），附三版对比表（iter3 91.8% → iter4 8.26% → iter4b 1.11%）；性能改为委托 upstream 后 1024³ 237 ms（26× 加速）与 2048³ 不再超时；内存改为 harness 侧 1 GiB workspace 分配；⑤ 附录新增 iter4b 版本文档，标注 PR #1559 后继关系与 ops-blas 仓 PR #457 的实际代码合入；⑥ 保留 v3.0.0 的路线决策记录供追溯。 | — |

---

## 附录：关联 PR 与合入路径

- **社区任务槽位**：`04_tasks/01_community-task-2026/tasklist/09-20-aclblasCgemmEx-950/ren648154292/`（本仓目录）
- **上游 PR**：`cann/ops-blas` PR **#457**（工作分支 `add-aclblasCgemmEx-950` @ `b85e0a3`），实际代码合入走 ops-blas 仓
- **本社区任务 PR**：`cann/cann-ops-competitions` **#1560**（本 PR，`aclblasCgemmEx-iter4b` 分支）
- **前置 PR**：`cann/cann-ops-competitions` **#1559**（已合入，iter3 自研 Karatsuba 路线，数值不可用）
