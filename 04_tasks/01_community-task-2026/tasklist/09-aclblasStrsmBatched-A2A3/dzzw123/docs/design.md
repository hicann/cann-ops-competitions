# aclblasStrsmBatched 算子设计文档（Atlas A2/A3）

| 版本 | 日期 | 作者团队 | 说明 |
| --- | --- | --- | --- |
| v0.1 | 2026-09-23 | dzzw123 | 初版：A2/A3（arch22）批量三角求解算子设计 |
| v0.2 | 2026-09-25 | dzzw123 | 按模板补充算子分析与可维可测章节，明确接口与约束范围 |

> 目标算子目录：`blas/trsmbatched/arch22/`
> 目标测试目录：`test/trsmbatched/strsmbatched/arch22/`
> 对标接口：cuBLAS `cublasStrsmBatched`；CANN 9.1.0；Atlas A2/A3（arch22）

---

# 需求背景（required）

## 需求来源

9 月社区任务：在 Atlas A2/A3（arch22）平台使用 Ascend C 实现单精度实数批量三角求解算子 `aclblasStrsmBatched`，对 `batchCount` 个相互独立的三角线性系统批量求解，验收后向 ops-blas 提交 PR。

## 背景介绍

### 算子目标

对每个批次 `i = 0 .. batchCount-1` 求解一个三角线性系统，并把解写入独立输出缓冲：

- LEFT：`op(A_i) X_i = alpha · B_i`，其中 `A_i` 为 `m × m` 三角矩阵；
- RIGHT：`X_i op(A_i) = alpha · B_i`，其中 `A_i` 为 `n × n` 三角矩阵；
- 解 `X_i` 写入 `Carray[i]`（`m × n`），`Barray[i]` 只读，作为右手项参与运算。

其中 `op(A)` 由 `trans` 决定（N：A；T：Aᵀ；实数算子不涉及共轭），`uplo` 指定上/下三角存储的半边，`diag = UNIT` 时对角元素按 1 处理。矩阵均为列主序，`lda/ldb/ldc` 为各矩阵的前导维。

### 算子现状分析

| 项 | 现状 |
| --- | --- |
| 公共接口 | `include/cann_ops_blas.h` 中已声明三角批量求解接口，为原地覆写右手项的形式 |
| 其它架构实现 | 已存在基于向量指令扩展的批量求解实现，覆盖 panel 求解与转置等步骤 |
| arch22 实现 | 尚无同类批量三角求解算子，本任务为首次落地 |
| 相邻可参考实现 | arch22 已有批量矩阵乘、三角/对称矩阵-向量求解等算子，可作为 host 侧编排与向量核写法参考 |

因此本工作的主要内容是：在 arch22 上按统一接口约定实现求解内核，并把三角求解算法组织为适配 NPU 存储层级的分块流程。

### 算子功能分析

需要覆盖的功能维度：

- `side` ∈ {LEFT, RIGHT}
- `uplo` ∈ {LOWER, UPPER}
- `trans` ∈ {N, T}
- `diag` ∈ {NON_UNIT, UNIT}

四者组合共 16 种模式，各模式对应不同的遍历方向（前代 / 回代）与索引映射。此外需支持非法参数判定、`m/n/batchCount = 0` 的空操作语义，以及 `alpha = 0` 时不依赖右手项输入的直接置零。

---

# 需求分析（required）

## 需求描述

实现句柄式 `aclblasStrsmBatched`（批量、独立输出 C），基于 Ascend C kernel 直调方式在 arch22 上完成求解计算，支持 FLOAT32，接口行为与 cuBLAS 对应算子一致（本实现在其上增加独立输出参数）。计算通过 handle 关联的流异步下发。

## 需求拆解

1. **接口与参数校验**：枚举合法性、维度非负、前导维下界、指针数组与标量指针非空，以及空操作与 `alpha = 0` 的早退分支。
2. **批量定址**：`Aarray / Barray / Carray` 为设备端指针数组，host 侧取回各批次基址后按批次下发计算。
3. **求解算法组织**：三角求解沿三角维存在顺序依赖，采用分块（blocked）写法：先解对角面板，再用一次矩阵乘更新剩余部分，使大规模问题以 GEMM 为主。
4. **方向与索引映射**：按 `side / uplo / trans / diag` 统一推导遍历方向与元素索引，避免分支组合的重复实现。
5. **数据类型与内存布局**：仅 FLOAT32，列主序，按前导维访问，不假设连续打包。
6. **精度与性能验收**：建立与参考实现的对比口径，并对任务书给定形状完成性能验证。
7. **测试与文档**：补充 arch22 测试工程、用例集与 README 复现说明。

## 算子接口原型

```cpp
aclblasStatus_t aclblasStrsmBatched(
    aclblasHandle_t handle, aclblasSideMode_t side, aclblasFillMode_t uplo,
    aclblasOperation_t trans, aclblasDiagType_t diag, int m, int n, const float* alpha,
    const float* const Aarray[], int lda, const float* const Barray[], int ldb,
    float* const Carray[], int ldc, int batchCount);
```

接口在参考语义（解覆盖右手项）基础上引入独立输出：解写入 `Carray`，`Barray` 保持只读。若公共头中已有的同名声明与本签名不一致，需要统一到本签名，并同步适配仓内其它架构的实现与调用点。

### 参数说明与异常行为

| 参数 | 说明与异常 |
| --- | --- |
| `handle` | 为空返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| `side` | `{LEFT, RIGHT}`，其它值 `INVALID_VALUE` |
| `uplo` | `{LOWER, UPPER}`，其它值 `INVALID_VALUE` |
| `trans` | `{N, T}`；实数算子不接受 `C`，其它值 `INVALID_VALUE` |
| `diag` | `{NON_UNIT, UNIT}`，其它值 `INVALID_VALUE` |
| `m, n, batchCount` | 需 ≥ 0；为负返回 `INVALID_VALUE`；为 0 时为合法空操作 |
| `alpha` | Host 端标量；指针为空返回 `INVALID_VALUE` |
| `lda` | LEFT：`≥ max(1, m)`；RIGHT：`≥ max(1, n)` |
| `ldb`, `ldc` | `≥ max(1, m)` |
| `Aarray`, `Barray`, `Carray` | 设备端指针数组；数组本身为空或某批次地址为空返回 `INVALID_VALUE` |

校验按 handle → 枚举 → 维度 → 前导维 → 指针 → 空操作早退的顺序进行。`alpha = 0` 时不引用 `Aarray / Barray`，输出按零填充。

## 设计范围与约束

- 仅支持 FLOAT32；不支持其它实数/复数类型。
- 列主序存储，按前导维访问，支持带填充的 leading dimension。
- 不支持 `trans = C`；不支持批量步长（批量通过指针数组表达）。
- 输出缓冲 `Carray` 与其它输入缓冲不重叠。
- 计算按流异步下发，结果读取前由调用方同步。

---

# 详细设计（required）

## 算子分析

### 数学公式

以 LEFT、非转置、下三角为例，逐列独立求解 `L x = alpha · b`：

```
x_0 = (alpha · b_0) / L_00
x_i = (alpha · b_i − Σ_{j<i} L_ij · x_j) / L_ii        (i = 1 .. m-1)
```

`diag = UNIT` 时省略除法。上三角或转置情形对应回代形式：

```
x_i = (alpha · b_i − Σ_{j>i} U_ij · x_j) / U_ii        (i = m-1 .. 0)
```

LEFT 的问题按列彼此独立，RIGHT 的问题按行彼此独立，这构成并行划分的基本维度。

分块（blocked）写法：把 `op(A)` 分Panel，先解对角面板，再用一次矩阵乘把该面板的解对剩余右手项的影响扣除，即

```
C_tail ← C_tail − op(A_tail, panel) · X_panel
```

使主要计算量落到矩阵乘上，面板内的顺序回代规模被限制在面板宽度内。

### 支持数据类型

| 数据类型 | 支持情况 |
| --- | --- |
| FLOAT32 | 支持 |
| FLOAT64 / 复数 / 半精度 | 不在本任务范围 |

### 支持形状

- `m, n` 允许为 0（空操作），`batchCount` 允许为 0。
- 方阵部分 `A` 维数为 `m`（LEFT）或 `n`（RIGHT），右手项为 `m × n`。
- 大尺寸形状（数千量级）通过分块流程承担，中小尺寸可直接按单面板求解。

## 算子实现

### 整体架构

单个批次的计算组织为：

```
输入 B ──(乘 alpha 预置到输出缓冲)──► C
                                        │
        对每个面板 panel：               │
          1) 面板内三角求解（顺序回代）  │
          2) 面板解 × 三角子块 → 矩阵乘  ─► 更新 C 的剩余行/列
                                        │
输出 C（解）◄───────────────────────────┘
```

批次之间相互独立，可在 host 侧按批次下发，也可在核侧按批次与列（LEFT）/行（RIGHT）分派到多个计算单元。

#### host 侧设计

- 参数校验与空操作/`alpha = 0` 分支处理。
- 从设备指针数组取回各批次基址，逐批下发内核。
- 分块参数（面板宽度、矩阵乘分片）由 tiling 阶段按形状与可用存储层级容量推导。
- 中间面板与矩阵乘结果需要临时存储，统一由 workspace 承载，按批量与分块上界估算大小，做尺寸溢出保护与地址对齐。
- 计算通过 handle 绑定的流异步下发，host 侧不逐批同步。

#### kernel 侧设计

- **Init**：设置批次的基址、前导维、分块参数与枚举解析出的方向标志。
- **CopyIn**：右手项按列（或行）搬入并乘 `alpha`，使用带边界处理的搬运接口以适配非对齐尾部。
- **Compute**：面板内按顺序回代求解（受三角依赖约束，面板内不作并行假设）；面板外以矩阵乘完成尾块更新，累加长度按矩阵乘的划分组织。
- **CopyOut**：更新结果写回输出缓冲的对应列/行区间，注意前导维带来的步长。
- 转置与上/下三角通过索引映射与遍历方向表达，不额外物化转置矩阵；如某些模式需要转置形态的数据访问，则通过搬运阶段的步长参数完成。

#### 文件组织

```
blas/trsmbatched/arch22/
├── strsmbatched_host.cpp        # 参数校验、批量编排、分块与 workspace 规划
├── strsmbatched_kernel.cpp      # 求解内核与尾块更新内核
├── strsmbatched_kernel.h        # 内核入口声明
└── strsmbatched_tiling_data.h   # 传递给内核的参数结构
blas/trsmbatched/README.md       # 架构支持表与接口说明更新
test/trsmbatched/strsmbatched/arch22/
├── strsmbatched_test.cpp        # 测试用例入口
├── strsmbatched_test.csv        # 用例集
└── *_npu_wrapper.h              # 设备端数据搬运与调用封装
```

## 支持硬件

| 平台 | 支持情况 |
| --- | --- |
| Atlas A2 系列（arch22） | 支持（本任务目标平台） |
| Atlas A3 系列（arch22 同族） | 支持（按验收结果确认） |
| 其它架构已有实现 | 接口统一后需同步适配并回归 |

## 算子约束限制

- 仅 FLOAT32、列主序；`trans` 只支持 N/T。
- 要求满足各前导维下界；不保证重叠缓冲上的行为（输出不得与输入重叠）。
- 大规模尺寸下按分块流程执行，面板宽度与矩阵乘分片由 tiling 决定；接口层面对用户透明。
- 异步执行，调用方需在读取结果前同步对应流。

---

# 可维可测分析

## 精度标准 / 性能标准

- **精度**：以参考实现（BLAS 三角求解）为 golden，按逐批全元素比较；FLOAT32 采用混合容差（相对与绝对容差同量级，容差与维度平方相关），并规定匹配比例下限与最大绝对偏差上限。用例集以 `MERE / MARE` 双指标判定。
- **非有限值口径**：浮点溢出产生的非有限值不满足逐位可比前提。对参考值为非有限的位置，按仓内既有先例（同族 arch22 算子）在统计中剔除并显式记录剔除数量；剔除后无剩余可比样本的用例标注为不可验证，不记为普通通过。
- **性能**：以任务书给定的形状与达标时间为准，采样多次取统计值；对比时明确单次调用与批量整体的口径。

## 测试设计

- 用例覆盖：基础尺寸扫描、`side × uplo × trans × diag` 模式正交、非法枚举、负维度与负批量、空操作、`alpha = 0`、极端值（Inf/NaN）行为、性能形状。
- 参考实现按批次生成 golden，测试侧准备 A/B/C 三组独立缓冲并校验 B 不被改写。
- 断言分两层：返回值（成功 / `INVALID_VALUE` / handle 为空）与数值比较（容差、匹配比例、最大偏差）。
- README 记录构建开关、用例运行命令与结果采集方式，保证第三方可复现。

## 兼容性分析

- **前向兼容**：接口采用独立输出形式后，原“解覆盖右手项”的调用方式不再成立；统一到该签名时需同步适配仓内其它架构实现与所有调用点，并回归其用例。
- **行为差异**：与参考接口在非有限值场景、以及数值细节（分块顺序导致的舍入差异）上可能存在差异；有限值按容差判定，非有限值按上述口径处理。
- **扩展空间**：后续可在同一组织方式上支持更多数据类型与批量步长形式，接口与目录结构不需调整。
