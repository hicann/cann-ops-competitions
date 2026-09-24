# aclblasStrmm A2/A3 算子设计文档

# 需求背景（required）

## 需求来源

本任务来自 CANN 社区任务广场（HiAscend）aclblasStrmm 开发任务，要求在 `ops-blas` 开源仓中为 **Atlas A2/A3 系列产品**（含 Atlas 800I A2 / Atlas 800I A3，架构 **arch22 / DAV_2201**）补齐单精度三角矩阵乘法算子 `aclblasStrmm` 的 arch22 实现、测试与文档。

对标接口：cuBLAS `cublasStrmm` / Netlib BLAS `strmm`。接口声明复用 `include/cann_ops_blas.h` 已有声明，与 950PR（arch35）等产品线共用同一 API，**禁止**新增产品私有平行接口。

> 注：官方任务书 / `test_cases` 以任务中心压缩包为准；本文精度门槛对齐仓内 `blas/trmm/README.md` 与生态开源精度标准；性能标杆 us 待官方 `gpu_baseline` 替换。

## 背景介绍

### aclblasStrmm 算子功能

`aclblasStrmm` 实现单精度三角矩阵与一般矩阵乘法，结果乘以标量 `alpha` 后**离席写入**输出矩阵 `C`（无 `beta*C` 项）：

```text
LEFT  : C := alpha * op(A) * B
RIGHT : C := alpha * B * op(A)
```

其中：

- `A` 为三角矩阵（仅存储上/下三角），`side=LEFT` 时为 `m×m`，`side=RIGHT` 时为 `n×n`
- `op(A)` ∈ {A, Aᵀ, Aᴴ}（FP32 下 C 与 T 等价）
- `B`、`C` 为 `m×n` 列主序矩阵
- `diag=UNIT` 时对角线视为 1，不读 `A` 对角线元素
- `alpha==0` 时将 `C` 的 `m×n` 有效区清零（`A`/`B` 可为无效指针）

### 仓内现状分析

| 能力 | 现状 |
| --- | --- |
| API 声明 | `include/cann_ops_blas.h` 已声明 `aclblasStrmm` |
| arch35（950PR） | `blas/trmm/arch35/` 已实现三阶段 Mirror(SIMT)→Gemm(Te::tensor_api)→Scale(SIMT) |
| 测试框架 | `test/trmm/strmm/` CSV + cblas golden |
| arch22（A2/A3） | **缺失**；README 产品表写「不支持」→ 本任务补齐 |
| 硬约束 | arch22 **无 SIMT**；arch35 的 `__simt_vf__` / `Te::` 不可照搬到 A2 |

# 需求分析（required）

## 需求描述

在 Atlas A2/A3（arch22）上使用 Ascend C 实现 `aclblasStrmm`，使功能、精度、性能满足社区任务验收标准，并合入 `cann/ops-blas` 的 `blas/trmm/arch22/` 与 `test/trmm/strmm/arch22/`。

接口：

```cpp
aclblasStatus_t aclblasStrmm(
    aclblasHandle_t handle,
    aclblasSideMode_t side,
    aclblasFillMode_t uplo,
    aclblasOperation_t trans,
    aclblasDiagType_t diag,
    int m, int n,
    const float *alpha,
    const float *A, int lda,
    const float *B, int ldb,
    float *C, int ldc);
```

## 需求拆解

1. 覆盖 LEFT/RIGHT × UPPER/LOWER × N/T/C × UNIT/NON_UNIT 全组合；列主序与 ld 约束对齐 cuBLAS/仓内 README。
2. quick return：`m==0 || n==0` 直接成功；`alpha==0` 清零 `C` 有效区。
3. `alpha` 支持 Host/Device；动态获取 AIV/AIC 核数，禁止硬编码 CoreNum。
4. 精度：cblas `cblas_strmm` golden；FLOAT32 MIXED_TOLERANCE（rtol 2^-10，atol 2^-16，matched_ratio 0.99）。
5. 性能：warmup 后有效采样 >50 次；平均耗时不高于官方标杆（待任务包）。
6. 更新 `blas/trmm/README.md` 产品支持表为 A2/A3 支持。
7. 新增 `test/trmm/strmm/arch22/` CSV 驱动精度与性能用例。

# 详细设计（required）

## 算子分析

### 数学公式

见上文 LEFT/RIGHT 公式。实现上将三角 `A` **补齐为稠密矩阵**（缺失三角填 0；UNIT 对角线填 1），再做稠密 GEMM，最后乘 `alpha` 写 `C`。

列主序适配（继承 arch35）：GEMM 内核按行主序语义工作时，Host 侧对 GEMM 做 `m↔n` 交换并翻转 `side`，使列主序内存布局下结果正确。

### 支持数据类型

| 参数 | 数据类型 |
| --- | --- |
| alpha / A / B / C | float32 |

### 支持形状

- `m≥0, n≥0`；`m==0` 或 `n==0` 不访问指针
- `side=LEFT`：`lda ≥ max(1,m)`；`side=RIGHT`：`lda ≥ max(1,n)`
- `ldb, ldc ≥ max(1,m)`
- 列主序；允许 `ld* > dim` 的 padding（padding 行不写）

## 算子实现

### 实现方案（选定）

三阶段流水线（与 arch35 同构，arch22 换算子模型）：

```text
Phase1 Mirror/Pack (AIV, Vector SIMD)
  GM(A 三角) ──DataCopyPad + 掩码/填零/UNIT──▶ GM(workspaceA 稠密)

Phase2 GEMM (AIC, MatmulImpl / Cube)
  GM(workspaceA), GM(B) ──Matmul──▶ GM(temp)

Phase3 Scale (AIV, Vector SIMD)
  GM(temp) ──Muls(alpha)──▶ GM(C)
```

方案来源：三模型辩论（mimo/deepseek/longcat）综合择优，见工作区 `docs/DEBATE_CHOICE.md`。

| 阶段 | arch35 | arch22（本设计） |
| --- | --- | --- |
| Mirror | SIMT `__simt_vf__` | Vector：`DataCopyPad` + `Duplicate`/`Select`/`Muls` |
| GEMM | `tensor_api` Te:: | `MatmulImpl`（参照 herk/symm arch22） |
| Scale | SIMT | Vector：`DataCopyPad` + `Muls` |

**不采用**：纯 Vector TRMM 主路径；Cube 三角 mask；照搬 Te::tensor_api。

性能迭代（可选）：若官方标杆压线，将 Phase1 改为按 K-chunk 在线 pack（仿 ssymm），降低完整 `workspaceA` 的 GM 往返。

### Host 侧设计

路径：`blas/trmm/arch22/strmm_host.cpp`

1. `handle==nullptr` → `HANDLE_IS_NULLPTR`；枚举非法 → `INVALID_ENUM`；`m/n/ld*/空指针` 校验对齐 README。
2. `m==0 || n==0` → 成功返回。
3. 解析 `alpha`（Host 直读 / Device D2H）；`alpha==0` → `aclrtMemsetAsync` 清零 `C` 有效区（`ldc==m` 整块，否则按列）。
4. `GetAivCoreCount` / `GetAicCoreCount` 动态取核数；组装 Mirror / Gemm / Scale 三份 TilingData。
5. `EnsureDefaultWorkspace`：`workspaceA`（对齐 32B）+ `temp`（`tempRowStride=CeilAlign(n,8)`）。
6. 顺序 launch：`strmm_mirror_kernel_do` → `strmm_gemm_kernel_do` → `strmm_scale_kernel_do`。

分核：

- Mirror：按 `dimA` 行均分到 AIV
- GEMM：按 `CeilDiv(m,BASE_M)*CeilDiv(n,BASE_N)` tile 数限制 AIC 核数；tileM/N/K 受 L1 512KB 双缓冲约束下调
- Scale：按 `m` 行均分到 AIV

### Kernel 侧设计

路径：`blas/trmm/arch22/strmm_kernel.cpp` + `strmm_kernel.h` + `strmm_tiling_data.h`

1. **Mirror**：模板参数覆盖 `UPLO × TRANS × DIAG`；按行 tile 读 A，三角外写 0，UNIT 对角线写 1；`trans=T/C` 时按转置索引读；结果写 `workspaceA`（列主序，供后续 GEMM 侧翻转语义使用）。TPipe 入口栈上创建，非成员（R3）。禁止逐元素 `SetValue` 热路径（R1），优先 Vector 批量 API。
2. **GEMM**：`MatmulType` + `MatmulImpl`，左右操作数来自 `workspaceA` / `B`（经 Host 侧 side 翻转后的逻辑左右）；结果写 `temp`。
3. **Scale**：按行从 `temp`（stride=`tempRowStride`）读到 UB，`Muls(alpha)` 后按 `ldc` 写 `C` 的前 `m` 行。

### 文件规划

| 文件 | 新增/修改 | 说明 |
| --- | --- | --- |
| `blas/trmm/arch22/strmm_host.cpp` | 新增 | arch22 Host |
| `blas/trmm/arch22/strmm_kernel.cpp` | 新增 | Mirror + Gemm + Scale |
| `blas/trmm/arch22/strmm_kernel.h` | 新增 | `*_kernel_do` 声明 |
| `blas/trmm/arch22/strmm_tiling_data.h` | 新增 | 三阶段 tiling（无数组字段，R4） |
| `blas/trmm/README.md` | 修改 | A2/A3 改为支持 |
| `test/trmm/strmm/arch22/strmm_test.cpp` | 新增 | GTest CSV |
| `test/trmm/strmm/arch22/strmm_test.csv` | 新增 | 精度 + 性能 |
| `test/trmm/strmm/arch22/strmm_npu_wrapper.h` | 新增 | NPU 封装 |
| `test/trmm/strmm/strmm_param.h` / `strmm_golden.h` | 复用 | 参数与 cblas golden |

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas A2 系列产品（含 Atlas 800I A2 / 800T A2） | √ |
| Atlas A3 系列产品（含 Atlas 800I A3） | √ |

## 算子约束限制

| 约束项 | 说明 |
| --- | --- |
| 空指针 | handle 空 → `HANDLE_IS_NULLPTR`；`alpha`/`C` 空 → `INVALID_VALUE`；`alpha≠0` 时 `A`/`B` 空 → `INVALID_VALUE` |
| 枚举 | side/uplo/trans/diag 非法 → `INVALID_ENUM` |
| dtype | 仅 float32 |
| 架构 | arch22 禁止 SIMT；禁止 950 私有平行 API |
| CoreNum | 动态获取，禁止硬编码 |
| 异步 | 依赖 `aclblasSetStream`；读回前须同步 |

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | FLOAT32 MIXED_TOLERANCE：rtol 2^-10，atol 2^-16，matched_ratio 0.99，max_abs 1e-2 或 32 ULP；`alpha==0` 时 EXACT | README / 生态精度标准 / 任务书 |
| 性能标准 | warmup 后 >50 次平均；不高于官方标杆 us | 任务包 `gpu_baseline`（待替换） |
| golden | cblas `cblas_strmm` | 仓内 `strmm_golden.h` |

精度覆盖重点：LEFT/RIGHT、UPPER/LOWER、N/T、UNIT/NON_UNIT、alpha∈{0,1,随机}、小/中/大 shape、ld padding、空指针与非法枚举负向、Inf/NaN 描述用例。

性能覆盖重点：任务包性能 case；补充中等方阵与瘦高/矮胖矩阵。

## 兼容性分析

- API 与 arch35 共用 `aclblasStrmm`，不新增平行接口。
- 新增 `arch22/` 目录，由 `SOC_ARCH_DIRS` 自动收集源文件；不影响 arch35 产品线。
- arch35 仍受 `TENSOR_API_OPS` / asc-devkit≥9.1 门禁；arch22 路径不依赖该门禁。
