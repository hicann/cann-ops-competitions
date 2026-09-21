# aclblasChemm 算子设计文档

> 任务：算子实操工坊-上海站-aclblasChemm 算子开发（950）
> 适配硬件：Ascend 950PR（arch35 / DAV_3510）　CANN 版本：9.1.0
> 对标接口：cuBLAS `cublasChemm`（语义参考 Netlib `chemm`）
> 目标仓库：https://gitcode.com/cann/ops-blas （算子 `blas/symm/arch35/`，测试 `test/symm/chemm/arch35/`）

# 需求背景（required）

## 需求来源

昇腾 CANN 社区任务"算子实操工坊-上海站"，要求在 Ascend 950PR 上使用 Ascend C（kernel 直调方式）开发单精度复数厄米特矩阵乘算子 `aclblasChemm`，完成设计、开发、测试全流程，验收通过后向 ops-blas 开源仓提交 PR。

## 背景介绍

### 对标接口现状分析

cuBLAS `cublasChemm`：

- `side = LEFT` 时 `C = alpha·A·B + beta·C`；`side = RIGHT` 时 `C = alpha·B·A + beta·C`；
- A 为厄米特（Hermitian）复数矩阵，满足 `A = A^H`：仅 `uplo` 指定的上/下三角被引用，未引用部分由共轭对称性隐含，对角元素虚部假定为 0；
- alpha/beta/A/B/C 均为单精度复数（COMPLEX64，实虚部交错），列主序存储；
- Netlib quick-return 语义：`m=0` 或 `n=0` 为合法 no-op。

### ops-blas 仓现状分析

| 现状 | 位置 | 与本任务关系 |
| --- | --- | --- |
| `aclblasChemm` 接口声明已存在（int64_t 全参数） | `include/cann_ops_blas.h` | 实现签名与之保持一致即可，公共头零改动 |
| arch22 已有 `chemm` 实现（三角展开 + 实虚拆分多路 Mmad） | `blas/hemm/arch22/` | 语义与参数校验参考；其 Cube 输出层使用 arch22 专有 API，不可平移到 arch35 |
| arch35 复数同族先例 | `blas/herk/arch35/cherk_*`、`blas/symm/arch35/ssymm_*` | 工程范式参考（前处理拆面 + GEMM + 合并的三段式；测试工程 wrapper 模式） |
| arch35 FP32 MMA 通路实测吞吐有限（Sgemm 2048³ 探针远超本任务复数标杆允许的单实数 GEMM 时间片） | `blas/gemm/arch35/` | 结论：复数路径不能建立在 FP32 直乘之上，需窄类型高分解力通路 |
| 测试工程框架已存在 | `test/chemm/`（CSV 参数解析 + cblas double golden） | 任务 CSV 列格式与其严格对齐，golden 直接复用 |

### 功能分析

核心计算是"复数 GEMM 的厄米特特例"：A 为方阵且 `A = A^H`（实部对称、虚部反对称、对角虚部为 0），side×uplo 共 4 种组合；B/C 为 m×n 全矩阵。相对通用复数 GEMM 的增量语义是三角引用与厄米特约束；相对 arch22 版本的增量目标是 950PR 上的达标性能。

### 硬件层面的实现难点

1. arch35 复数 MMA 不直生：复数乘需分解为实数 GEMM；朴素 4-GEMM 分解与低吞吐通路叠加后无法达到任务书 §3.3 标杆，需在"实数 GEMM 次数"和"单 GEMM 通路吞吐/精度"两层同时做分解设计；
2. 厄米特三角引用需要设备端前处理（三角展开、共轭镜像、对角虚部清零），并与主计算的数据布局（前导维、padding）协同；
3. 拆分式数值方案在极端输入（Inf/NaN、大幅值）下的精度长尾需要兜底策略。

### 交付范围界定

- 交付：`blas/symm/arch35/` arch35 host+kernel 实现、`test/symm/chemm/arch35/` CSV 驱动测试工程、随任务 1200 条用例真机自验、4 档性能标杆、README 支持表（Ascend 950PR：支持）、自测报告；
- 不交付：A2/A3（arch22 已有）、非连续内存 / broadcast / dynamic shape（任务书不要求）；不修改任何既有接口与内核。

# 需求分析（required）

## 需求描述

在 ops-blas 仓内以 Ascend C kernel 直调实现 `aclblasChemm`（声明与 `include/cann_ops_blas.h` 既有定义一致）：

```cpp
aclblasStatus_t aclblasChemm(
    aclblasHandle_t handle,
    aclblasSideMode_t side,
    aclblasFillMode_t uplo,
    int64_t m, int64_t n,
    const aclblasComplex* alpha,
    const aclblasComplex* A, int64_t lda,
    const aclblasComplex* B, int64_t ldb,
    const aclblasComplex* beta,
    aclblasComplex* C, int64_t ldc);
```

**参数校验与返回码口径**：

| 场景 | 返回码 | 依据 |
| --- | --- | --- |
| handle 为 nullptr | `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` | 任务书 §2.4 |
| side/uplo 非法枚举 | `ACLBLAS_STATUS_INVALID_VALUE` | 任务 CSV（TC_ED_212/213）期望值；仓内 issue #416 维护者裁定"非法枚举返回码以测试脚本为准，自测报告备注"（任务书 §2.4 字面为 INVALID_ENUM，按该裁定不采用） |
| m/n < 0；lda/ldb/ldc 违反下界；alpha/beta/A/B 为 nullptr；beta≠0 且 C 为 nullptr | `ACLBLAS_STATUS_INVALID_VALUE` | 任务书 §2.4/§2.5 |
| m=0 或 n=0 | `ACLBLAS_STATUS_SUCCESS`（no-op，不 launch） | 任务书 §2.1 |

lda 下界：side=LEFT 时 `lda ≥ max(1,m)`；side=RIGHT 时 `lda ≥ max(1,n)`；`ldb/ldc ≥ max(1,m)`。

精度（任务书 §3.2）：golden 由 cblas（Netlib `chemm`）生成，全矩阵 m×n 实/虚部分别按 FLOAT32 判定——rtol 2⁻¹⁰、atol 2⁻¹⁶、matched_ratio ≥ 0.99、max_abs ≤ 1e-2 或 32×ULP；测试工程另有 MERE/MARE 口径（沿用 `test/chemm` 框架列配置）。

性能（任务书 §3.3，warmup 后 >50 次有效采样均值 ≤ 标杆）：

| case | m | n | side | uplo | 标杆（µs） |
| --- | --- | --- | --- | --- | --- |
| 1 | 1024 | 1024 | LEFT | UPPER | 564.14 |
| 2 | 2048 | 2048 | LEFT | UPPER | 3824.68 |
| 3 | 1024 | 1024 | RIGHT | LOWER | 547.6 |
| 4 | 2048 | 2048 | RIGHT | LOWER | 3683.6 |

## 需求拆解

1. **接口与校验**：host 侧参数校验（顺序：handle → 枚举 → 维度符号 → 指针 → 前导维）、零维 quick-return、`alpha=(0,0)` 退化为 `C=beta·C` 的快速路径；
2. **厄米特前处理**：设备端三角展开——按 uplo 读取被引用三角，共轭镜像补全另一三角，对角虚部强制清零，拆分为实数平面；
3. **复数 GEMM 主计算**：实数 GEMM 分解（Gauss 型少乘法计数）+ 高分解力窄类型通路与精度补偿；
4. **后处理**：实/虚组装、alpha/beta 复标量乘加、beta·C 旧值读取、按 ldc 列主序写回（含 padding 跳步）；
5. **side/uplo 组合**：4 组合走同一管线，side 仅改变矩阵在乘积中的位置与 A 维度约束，uplo 仅改变前处理读取的半区；
6. **测试**：`test/symm/chemm/arch35/` 测试工程接入任务 CSV（1200 条），新增性能计时路径；
7. **交付**：README 支持表 + 自测报告。

# 详细设计（required）

## 算子分析

### 数学公式

A 厄米特 ⟺ `A = Ar + i·Ai`，其中 `Ar^T = Ar`、`Ai^T = −Ai`、`diag(Ai) = 0`。

- LEFT：`C = α·A·B + β·C`（A 为 m×m）；RIGHT：`C = α·B·A + β·C`（A 为 n×n）。
- 复数乘拆实：`(Ar+iAi)(Br+iBi) = (Ar·Br − Ai·Bi) + i(Ar·Bi + Ai·Br)`；本设计进一步采用 Gauss 型 3-GEMM 分解减少 cube 乘法次数（与 ops-blas 复数家族通用做法一致）。
- 每个实数 GEMM 的 FP32 精度目标通过输入平面的高分解力窄类型拆分与累加端 FP32 求和重构实现，舍去项相对量级远低于 FLOAT32 判定阈值；极端数值场景由精确重算分支兜底（具体数值方案与常数在实现代码与自测报告中展开）。
- side=RIGHT 数学上等价于把展开后的 A 置于右操作数位置，不引入额外转置。

### 支持数据类型

恒 COMPLEX64（A/B/C/alpha/beta，`aclblasComplex{float real, imag}`）。

### 支持形状

- m、n ∈ [0, 4096]（任务 CSV 值域），支持非方阵（fat/thin）；
- 前导维：lda/ldb/ldc ≥ 各自下界，支持 padding；
- 不涉及广播与超出 ld 语义的非连续访问。

## 算子实现

### 实现方案（总览）

单 stream 串行的多 kernel 流水线（全程设备侧执行，无 host 数据往返）：

```
K0 前处理（AIV）  ：读 uplo 三角 → 共轭镜像展开为全矩阵视图 → 对角虚部清零
                   → 拆实/虚平面 → 数值范围统计 → 主计算输入平面构造（含对齐补零）
K1 主计算（AIC）  ：高分解力通路的实数 GEMM × N（Gauss 分解），FP32 累加
K2 后处理（AIV）  ：平面结果组装 → 幅值还原 → α/β 复标量乘加 → 列主序写回 C
K3 精确重算（条件）：极端数值分类下的兜底路径，常规数据零开销
```

与 arch22 版本的关系：语义与校验口径一致，计算通路完全重写（arch22 Cube 输出层 API 在 arch35 不可用，且 FP32 直乘吞吐无法满足标杆）。

### host 侧设计

- 校验顺序与状态码见"需求分析"；非法枚举返回 `ACLBLAS_STATUS_INVALID_VALUE`（issue #416 口径）；
- quick-return：`m==0 || n==0` 直接 SUCCESS；`alpha==(0,0)` 走 `C = β·C` 缩放路径（不经主计算）；`beta==(0,0)` 时 C 免读；
- workspace：按平面与中间结果尺寸预算，超出库上限时按行分带循环复用临时缓冲（对外行为不变）；
- tiling：按值传递的结构体参数，无额外 Tiling API 依赖；接口置于 `include/cann_ops_blas.h` 既有声明之下，无 950PR 私有平行 API。

### 边界行为实现

- 三角引用：K0 仅读取 uplo 指定半区，未引用三角的脏数据不影响结果；对角虚部无条件置 0（不信任输入）；
- TC_ED 负向族 20 条：零维、空指针、非法枚举 999、非法前导维、负维度、特殊标量，期望码逐条对 CSV；
- Inf/NaN：与 cublas 对齐的 IEEE 传播语义，特殊值族由精确重算分支兜底。

### 支持硬件

| 芯片 | 支持 |
| --- | --- |
| Ascend 950PR（arch35 / DAV_3510） | √（本任务交付） |
| A2/A3（arch22） | ×（既有 arch22 实现不动） |

### 算子约束限制

单批；仅 lda/ldb/ldc 前导维语义的连续内存；无 dynamic shape（m/n 运行时入参）；不承诺跨运行 bit 确定性；dim 上限受 workspace 预算约束（4096 实测覆盖）。

## 落位

按任务书目录：算子实现 `blas/symm/arch35/chemm_*`，测试工程 `test/symm/chemm/arch35/`；构建侧 `blas/CMakeLists.txt` 按 `blas/*/<arch>/*.cpp` glob 自动收集，新目录零注册成本。

# 可维可测分析

## 精度标准/性能标准

### 精度标准

- 任务书口径：实/虚部分别按 FLOAT32（rtol 2⁻¹⁰ / atol 2⁻¹⁶ / matched_ratio 0.99 / max_abs 1e-2 或 32×ULP）；
- golden：`test/chemm/chemm_golden.h`（double 累加 CPU 参考，语义 = Netlib `chemm`）直接复用；
- 全量 1000 条精度用例（排除 TC_PF）真机通过为验收口径，结果在自测报告逐条登记。

### 性能标准

4 档标杆见需求分析表；测量口径：warmup 后有效采样 >50 次取平均（测试工程内计时输出）；TC_PF 其余 196 条做扫描记录。返回码口径备注：非法枚举按 issue #416 裁定统一 `ACLBLAS_STATUS_INVALID_VALUE`，自测报告相应位置引用该 issue。

### 内存要求

任务书 §3.4 不涉及；交付件主动提供 workspace 预算解析式与 npu-smi 峰值采样。

## 测试用例规划

- 随任务 `chemm_test.csv` 1200 条全量（列格式与 `test/chemm/chemm_param.h` 解析严格对齐）；
- arch35 测试工程适配点：①新目录/新测试类；②TC_PF 行 lda/ldb/ldc 为空的解析夹紧（空值按 0 处理会违反前导维下界，需在参数层取 max(1,·) 下界）；③新增性能计时路径（既有 arch22 测试无计时）；④wrapper 按 arch35 先例自管 H2D/D2H；
- 负向用例期望码逐条对表；官方 `verify_accuracy.py` / `verify_performance.py` 复现路径写入测试 README。

## 兼容性分析

纯增量：新增 `blas/symm/arch35/chemm_*` 与 `test/symm/chemm/arch35/`；M `blas/symm/README.md` 支持表（追加 aclblasChemm：Ascend 950PR 支持）；不改任何既有接口、枚举、公共头与 arch22 实现。

## 风险与降级预案

| # | 风险 | 预案 |
| --- | --- | --- |
| R1 | 与在途复数 GEMM 类 PR 的通路同构性 | 实现自包含、不依赖未合入代码；合入后可提重构 PR 抽公共管线 |
| R2 | 大 shape workspace 超限 | 按行分带（host 编排，对外透明） |
| R3 | 窄类型拆分在厄米特输入分布下的精度长尾 | 极端类精确重算兜底 + 全量真机回归 |
| R4 | 非法枚举返回码与任务书字面不一致 | issue #416 维护者裁定为据，自测报告备注 |

## 文件与接口落位清单

```
A  blas/symm/arch35/chemm_host.cpp
A  blas/symm/arch35/chemm_kernel.h
A  blas/symm/arch35/chemm_kernel.cpp
A  blas/symm/arch35/chemm_tiling_data.h
M  blas/symm/README.md
A  test/symm/chemm/arch35/CMakeLists.txt
A  test/symm/chemm/arch35/chemm_test.cpp
A  test/symm/chemm/arch35/chemm_npu_wrapper.h
A  test/symm/chemm/arch35/chemm_test.csv
```

# 附录

## 参考资料

1. cuBLAS cublasChemm：https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-hemm
2. Netlib chemm：https://www.netlib.org/blas/chemm.f
3. ops-blas 仓：https://gitcode.com/cann/ops-blas
4. 非法枚举返回码裁定：https://gitcode.com/cann/ops-blas/issues/416
5. 生态算子开源精度标准：https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md
6. 随任务测试指导：`test_cases/README.md`

## 修订记录

| 版本 | 日期 | 说明 |
| --- | --- | --- |
| v1.0 | 2026-09-20 | 评审公开版初稿（落位按任务书目录；返回码口径引 issue #416） |
