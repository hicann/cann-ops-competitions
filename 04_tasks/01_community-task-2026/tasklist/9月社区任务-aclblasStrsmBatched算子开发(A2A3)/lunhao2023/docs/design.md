# 【社区任务】aclblasStrsmBatched 算子设计文档（A2/A3）

# 需求背景（required）

## 需求来源

本需求来自 9 月社区任务《aclblasStrsmBatched 算子开发（A2/A3）》。任务要求在 Atlas 800I A2 / Atlas 800I A3（arch22）上基于 Ascend C，于 `cann/ops-blas` 开源仓实现单精度实数批量三角求解接口 `aclblasStrsmBatched`，完成设计、开发、精度/性能自验，并以 PR 合入 `blas/trsmbatched/arch22/` 与 `test/trsmbatched/strsmbatched/arch22/`。

- **开发仓（个人 fork）**：https://gitcode.com/lunhao2023/ops-blas
- **上游仓**：https://gitcode.com/cann/ops-blas
- **设计文档仓（个人 fork）**：https://gitcode.com/lunhao2023/cann-ops-competitions
- **对标接口**：cuBLAS `cublasStrsmBatched`
- **Golden**：Netlib CBLAS `strsm` 逐 batch
- **验收环境**：CANN 9.1.0；精度建议 910B3 或 A3；性能建议 Atlas 800T A2（910B3）

## 背景介绍

### aclblasStrsmBatched 算子功能

对一批相互独立、尺寸一致的三角线性系统做批量求解（Batched Triangular Solve）。所有矩阵列主序存储。每个 batch `i`（`i = 0 .. batchCount-1`）独立求解，结果写入独立输出 `C[i]`（离席，不覆盖 `B[i]`）：

```text
side = LEFT :  op(A[i]) * X[i] = alpha * B[i] ,  C[i] = X[i]
side = RIGHT:  X[i] * op(A[i]) = alpha * B[i] ,  C[i] = X[i]
```

`op(A)` 由 `trans` 决定：`ACLBLAS_OP_N` → `A`，`ACLBLAS_OP_T` → `Aᵀ`。本接口为实数算子，**不支持** `ACLBLAS_OP_C`（非法枚举，返回 `ACLBLAS_STATUS_INVALID_VALUE`）。

### 与仓内 arch35 实现的关系

主仓已有 `blas/trsmbatched/arch35/` 的 950 实现，公共声明目前为 cuBLAS 风格的原地覆写（解 `X` 写回 `B`，无 `Carray/ldc`）：

```cpp
aclblasStatus_t aclblasStrsmBatched(
    aclblasHandle_t handle, aclblasSideMode_t side, aclblasFillMode_t uplo,
    aclblasOperation_t trans, aclblasDiagType_t diag,
    int m, int n, const float* alpha,
    const float* const A[], int lda,
    float* const B[], int ldb,
    int batchCount);
```

本任务书将语义扩展为 **B 只读、C 独立输出**，并收紧 `trans`（禁止 `OP_C`）。A2/A3 实现必须按任务书签名落地，不能直接复用 arch35 的原地接口与 950 Cube/SIMT 路径。

### 任务书接口（A2/A3 真值）

```cpp
aclblasStatus_t aclblasStrsmBatched(
    aclblasHandle_t handle,
    aclblasSideMode_t side,
    aclblasFillMode_t uplo,
    aclblasOperation_t trans,
    aclblasDiagType_t diag,
    int m, int n,
    const float* alpha,
    const float* const* Aarray, int lda,
    const float* const* Barray, int ldb,
    float* const* Carray, int ldc,
    int batchCount);
```

| 参数 | 输入/输出 | 说明 | 约束 / 异常 |
| --- | --- | --- | --- |
| handle | 输入 | ops-blas 句柄，携带 stream | nullptr → `HANDLE_IS_NULLPTR` |
| side | 输入 | LEFT：A 为 m×m；RIGHT：A 为 n×n | 非法枚举 → `INVALID_VALUE` |
| uplo | 输入 | UPPER / LOWER，只引用对应三角 | 非法枚举 → `INVALID_VALUE` |
| trans | 输入 | N / T（实数，禁止 C） | 非法枚举（含 OP_C）→ `INVALID_VALUE` |
| diag | 输入 | NON_UNIT 用实际对角；UNIT 对角视为 1，不读对角存储 | 非法枚举 → `INVALID_VALUE` |
| m, n | 输入 | C 的行/列 | <0 → `INVALID_VALUE`；=0 合法 no-op |
| alpha | 输入 | Host FLOAT32 标量指针 | nullptr → `INVALID_VALUE` |
| Aarray | 输入 | Device 三角矩阵指针数组 | nullptr 或其元素 nullptr → `INVALID_VALUE`（alpha=0 时不引用 A） |
| lda | 输入 | A 前导维 | LEFT：`lda ≥ max(1,m)`；RIGHT：`lda ≥ max(1,n)` |
| Barray | 输入 | Device 右端矩阵指针数组，只读 | nullptr 或其元素 nullptr → `INVALID_VALUE`（alpha=0 时不引用 B） |
| ldb | 输入 | B 前导维 | `ldb ≥ max(1,m)` |
| Carray | 输出 | Device 结果矩阵指针数组，独立输出 | nullptr 或其元素 nullptr → `INVALID_VALUE` |
| ldc | 输入 | C 前导维 | `ldc ≥ max(1,m)` |
| batchCount | 输入 | 批量个数 | <0 → `INVALID_VALUE`；=0 合法 no-op |

返回值与 `include/cann_ops_blas_common.h` 一致。

# 需求分析（required）

## 需求描述

在 ops-blas 的 arch22 路径上实现 `aclblasStrsmBatched`：句柄式直调 NPU Kernel，覆盖 LEFT/RIGHT × UPPER/LOWER × N/T × UNIT/NON_UNIT，支持运行时 `m/n/batchCount` 与 lda/ldb/ldc padding，离席写出 C，满足 FLOAT32 混合容差精度与任务书 5 条性能 case。

## 需求拆解

1. 公共头文件按任务书提供带 `Carray/ldc` 的声明，供产品线共用；arch35 原地签名以 C++ 重载保留，避免破坏 950。
2. Host 完成全部合法性校验与 no-op / `alpha=0` 快路径，不在 Kernel 内重复业务校验。
3. Kernel 每个 batch 独立求解，batch 间不得混淆尺寸与指针；`C` 与 `A`/`B` 不允许内存重叠。
4. `ACLBLAS_DIAG_UNIT` 跳过对角存储，对角视为 1。
5. `trans=OP_C` 在 Host 直接判非法，与 cuBLAS 实数档（OP_C 等价 OP_T）刻意收紧。
6. 小 shape 走 Vector 三角回代；大 shape 走分块 TRSM（panel 回代 + Cube GEMM 更新），以覆盖 2048/4096 性能 case。
7. 测试落入 `test/trsmbatched/strsmbatched/arch22/`，CSV + GTest，golden 为 cblas `strsm` 逐 batch。
8. README 产品支持表标注 Atlas A2/A3（含 800I A2/A3）：支持。

# 详细设计（required）

## 算子分析

### 数学公式

令 `op(A)` 为 `A` 或 `Aᵀ`。对每个 batch `i`：

- LEFT + NON_UNIT：前代/回代求解 `op(A[i]) X[i] = alpha B[i]`
- LEFT + UNIT：对角乘子恒为 1，不读取 `A[i]` 对角元
- RIGHT：等价于对 `op(A[i])ᵀ` 做 LEFT 求解后再按列主序写回，或直接按行方向回代

`alpha=0` 时不读 `A[i]`、`B[i]`，将 `C[i]` 的有效 `m×n` 区域（按 `ldc`）置零。

列主序地址：`A(row, col) = A + row + col * lda`（B/C 同理，使用 ldb/ldc）。

### 支持数据类型

仅 FLOAT32。alpha 仅 Host 指针。

### 支持形状

- `m ≥ 0`，`n ≥ 0`，`batchCount ≥ 0`
- LEFT：`A[i]` 为 `lda × m` 物理存储中的 `m×m` 三角；RIGHT：`A[i]` 为 `lda × n` 中的 `n×n` 三角
- `B[i]`、`C[i]` 为 `ldb/ldc × n` 中的 `m×n`
- 允许 `lda/ldb/ldc` padding，不支持超出前导维语义的任意非连续访问
- 无 broadcast、无 dynamic shape 约束（m/n/batchCount 为运行时 Host 标量）

## 算子实现

### 工程落点

```text
ops-blas/
├── include/cann_ops_blas.h                 # 增加 A2/A3 任务书签名（重载）
├── blas/trsmbatched/
│   ├── README.md                           # 产品支持表补 Atlas A2/A3
│   └── arch22/
│       ├── strsmbatched_host.cpp
│       ├── strsmbatched_kernel.cpp
│       ├── strsmbatched_kernel.h
│       └── strsmbatched_tiling_data.h
└── test/trsmbatched/strsmbatched/
    ├── strsmbatched_golden.h               # 复用 cblas 逐 batch
    └── arch22/
        ├── strsmbatched_npu_wrapper.h
        ├── strsmbatched_test.cpp
        └── strsmbatched_test.csv
```

开发在个人 fork `lunhao2023/ops-blas` 的 `feat/aclblas-strsmbatched-arch22` 分支进行，合入目录与任务书第 5 节一致。

### Host 侧设计

1. **参数校验顺序**（与任务书异常列对齐，先 handle 后枚举后维度）：
   - `handle == nullptr` → `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`
   - `side/uplo/trans/diag` 非法（`trans` 仅允许 N/T）→ `INVALID_VALUE`
   - `m < 0` 或 `n < 0` 或 `batchCount < 0` → `INVALID_VALUE`
   - `alpha == nullptr` → `INVALID_VALUE`
   - 前导维：LEFT 时 `lda ≥ max(1,m)`，RIGHT 时 `lda ≥ max(1,n)`；`ldb ≥ max(1,m)`；`ldc ≥ max(1,m)`
   - `Carray == nullptr` → `INVALID_VALUE`（no-op 前也校验 C 指针数组本身；`m/n/batchCount` 全 0 时不访问元素）
2. **快路径（不 launch kernel）**：
   - `m==0 || n==0 || batchCount==0`：合法 no-op，返回 SUCCESS
   - `alpha==0`：按 batch 将 C 有效区域置零（可走专用 memset/Vector fill kernel），不读 A/B
3. **指针数组**：`Aarray/Barray/Carray` 在 Device。Host 将指针数组与 tiling 一并下发；alpha=0 路径不要求 A/B 元素有效。
4. **异步**：通过 `aclblasSetStream` 绑定的 handle stream 下发，Host 不主动同步。
5. **分核**：
   - 优先按 batch 切分：`usedCores = min(aivCoreNum, batchCount)`，前 `batchCount % usedCores` 个核多 1 个 batch
   - 单 batch 超大（m 或 n ≥ 256）时，在 batch 内再按 RIGHT 的列 panel / LEFT 的 RHS 列块切分，避免 4096 方阵打不满多核
6. **路径选择（tilingKey）**：

| Key | 条件 | 路径 |
| --- | --- | --- |
| 0 | `max(m,n) ≤ 64` 或 batch 内矩阵无法填满 Cube 最小块 | AIV Vector 回代 |
| 1 | `max(m,n) > 64` 且 LEFT | 分块 TRSM：AIV panel 回代 + Cube GEMM 更新 RHS |
| 2 | `max(m,n) > 64` 且 RIGHT | 对 C 的行 panel 做 RIGHT 回代，或转置为 LEFT 等价形式 |

7. **Tiling 字段（下发 Kernel）**：side/uplo/trans/diag、m/n、lda/ldb/ldc、batchCount、本核 batch 起点/个数、panelSize、tileM/tileN、alpha。

### Kernel 侧设计

**Init / Process**：按 tiling 绑定 GM 指针数组与 UB Buffer，再按 tilingKey 进入 Vector 或分块路径。

**离席语义**：先将 `B[i]` 的 `m×n`（按 ldb）拷到 `C[i]`（按 ldc），后续求解只写 C。禁止 C 与 A/B 重叠。

**UNIT 对角**：回代乘子固定为 1，不 load 对角。

**Vector 路径（小 shape / 尾块）**：

- UB 规划：A 三角块、C 的 RHS 列块、临时向量；开启 double buffer
- LEFT+LOWER+N：对 C 的每一列做前代；LEFT+UPPER+N：回代；`trans=T` 时按对偶三角访问
- RIGHT：对 C 的每一行（列主序下为跨 ldc 的 stride）做对应三角求解
- 单次搬入按 32B 对齐，padding 列不写回

**分块路径（大 shape，覆盖性能 case 3–5）**：

标准 Left-looking / Right-looking TRSM：

```text
for panel k:
    1. 用对角 panel A_kk 求解当前 C_k           # AIV 三角回代
    2. 用 A_ik 对剩余 C_i 做 GEMM 更新           # Cube：C_i -= A_ik * C_k  (LEFT)
```

- panel 边长取 64/128（按 UB 与 Cube L0 对齐，FP32 C0=8）
- GEMM 复用仓内 arch22 实数 GEMM helper（与 `hemm/arch22` 相同的 Host 直调 + Cube 风格），`alpha` 在首次拷贝 B→C 时融合，后续更新系数为 -1
- 尾 panel 不足对齐时回退 Vector，不把整题打成纯标量

**多 batch**：各核持有连续 batch 子区间，核间无依赖；核内 batch 串行以复用 UB 中的 A panel。

### 存储与对齐

- 外部契约始终列主序 `row + col * ld`
- Kernel 内部可将一个 panel 转成行/列连续补齐块供 Cube 使用，写回时再按 ldc scatter
- lda/ldb/ldc 大于逻辑维时，padding 区域保持不写（C 仅写 `m×n`）

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas 800I/T A2（arch22） | √ |
| Atlas 800I/T A3（arch22） | √ |
| Ascend 950（arch35） | 已有仓内实现，不在本任务修改范围 |

自验证覆盖其中一种款型即可；性能以 910B3 为准。

## 算子约束限制

- 不支持 broadcast、不支持超出 lda/ldb/ldc 的非连续访问
- 不要求确定性逐位一致（浮点累加顺序不保证）
- C 与 A 或 B 不允许内存重叠
- alpha 仅 Host；A/B/C 仅 Device
- 读回 Device 结果前由调用方同步 stream
- `trans=ACLBLAS_OP_C` 非法

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度 | FLOAT32：rtol=atol=`2^-13`（1.22e-4），matched_ratio≥0.99，max_abs_error≤1e-2 或 32 ULP；逐元素 `\|actual-golden\| ≤ atol + rtol×\|golden\|`。含规约的大数场景可将 max_abs_error 放宽至 2 ULP | 任务书 §3.2 / 生态混合容差标准 |
| Golden | cblas `strsm` 逐 batch，全矩阵 C 比对 | 任务书 §3.2 |
| 性能 | 910B3，warmup 后有效采样 >10 次取平均（msprof op 自带 5 次 warmup），Avg time 不高于下表 | 任务书 §3.3 |

| case | batchCount | m | n | side | uplo | trans | diag | 达标耗时（us） |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 64 | 256 | 256 | LEFT | LOWER | N | NON_UNIT | 630.7 |
| 2 | 128 | 384 | 512 | LEFT | UPPER | T | NON_UNIT | 3062 |
| 3 | 32 | 1024 | 1024 | RIGHT | LOWER | N | UNIT | 5134 |
| 4 | 8 | 2048 | 2048 | LEFT | UPPER | T | NON_UNIT | 8680 |
| 5 | 2 | 4096 | 4096 | RIGHT | UPPER | N | UNIT | 14361 |

## 测试计划

- 用例来源：任务配套 `test_cases/strsmbatched_test.csv` + 缺口自补（零维、空指针、非法枚举含 OP_C、非法前导维、alpha∈{0,1,-1}、Inf/NaN、lda padding）
- 框架：ops-blas test 目录 CSV → C++ GTest → `aclblasStrsmBatched`
- 负向对齐任务书错误码，不按 cuBLAS 实数 OP_C 等价 T 放行

## 兼容性分析

- 新增 arch22 实现与测试目录，不删除 arch35
- 公共头文件以重载形式增加 `Carray/ldc` 签名，950 原地接口保持可编译
- README 产品支持表：Atlas A2/A3 改为支持；950 保持原状
- 个人仓已邀请 `Ascend-CANN` 作为开发者，供验收拉取
