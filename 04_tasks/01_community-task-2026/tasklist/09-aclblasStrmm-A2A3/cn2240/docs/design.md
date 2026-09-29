# aclblasStrmm 算子设计文档

## 1 概述

### 1.1 算子名称

aclblasStrmm（Single-precision Triangular Matrix Multiplication）

### 1.2 功能描述

在昇腾 NPU 上实现单精度实数三角矩阵乘算子，对齐 cuBLAS `cublasStrmm` 语义：

- `side=LEFT`：`C = alpha * op(A) * B`
- `side=RIGHT`：`C = alpha * B * op(A)`

其中 A 为三角矩阵（上/下三角由 `uplo` 决定），`op(A)` 为 A 或 A^T（由 `trans` 决定），B/C 为一般矩形矩阵（m×n），alpha 为标量。输出 C 为标准一般矩阵（非三角输出），支持离席计算与 B≡C 原地写出。

### 1.3 数学公式

```
side=LEFT :  C := alpha * op(A) * B      A 为 m×m 三角矩阵
side=RIGHT:  C := alpha * B * op(A)      A 为 n×n 三角矩阵

op(A) = A      当 trans = ACLBLAS_OP_N
op(A) = A^T    当 trans = ACLBLAS_OP_T

uplo=UPPER: 引用 A 上三角（含对角）
uplo=LOWER: 引用 A 下三角（含对角）
diag=UNIT:  对角视为 1，不读取 A 对角元素
diag=NON_UNIT: 正常读取对角元素
```

### 1.4 算法描述

将 TRMM 分解为三段管线，通过三角零填充把三角矩阵乘归约为标准稠密 GEMM：

1. **Mirror（三角裁剪+零填充）**：将三角矩阵 A 按 uplo/diag 裁剪，非引用区填零，产出稠密方阵 Ã 存入 workspace
2. **GEMM（矩阵乘）**：利用 Matmul 高阶模板计算 Ã 与 B 的矩阵乘，结果存入临时缓冲区 temp
3. **Scale（缩放写出）**：将 temp 乘以 alpha 后按列主序写入输出矩阵 C

### 1.5 数据流图

```
A(三角,Device) ──[K1:Mirror AIV]──→ workspaceA(稠密,GM)
                                          │
B(Device) ─────────────────────────→ [K2:GEMM AIC] ──→ temp(GM)
                                                           │
alpha(Host/Device) ────────────→ [K3:Scale AIV] ──→ C(Device)
```

## 2 适配硬件与环境

| 项目 | 内容 |
|------|------|
| 适配硬件 | Atlas A2 系列产品（Atlas 800T A2, Ascend 910B3）、Atlas A3 系列产品 |
| 架构目录 | arch22（DAV_C220：25 AIC + 48 AIV，UB 192KB，L1 512KB） |
| CANN 版本 | CANN 9.1.0 |
| 编程语言 | Ascend C（kernel 直调，ops-blas 仓工程框架） |
| 数据类型 | FLOAT32（A/B/C/alpha 全为 float32） |

## 3 接口设计

### 3.1 函数原型

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

接口声明位于 `include/cann_ops_blas.h`，与 cuBLAS `cublasStrmm` 参数序列一一对应。差异点：`trans` 仅接受 N/T，`ACLBLAS_OP_C` 视为非法枚举返回 `ACLBLAS_STATUS_INVALID_VALUE`。

### 3.2 参数说明

| 参数 | 方向 | 内存 | 类型 | 约束 |
|------|------|------|------|------|
| handle | 输入 | Host | aclblasHandle_t | 非 nullptr，否则返回 HANDLE_IS_NULLPTR |
| side | 输入 | Host | aclblasSideMode_t | LEFT/RIGHT，非法返回 INVALID_VALUE |
| uplo | 输入 | Host | aclblasFillMode_t | UPPER/LOWER，非法返回 INVALID_VALUE |
| trans | 输入 | Host | aclblasOperation_t | 仅 N/T，OP_C 及非法值返回 INVALID_VALUE |
| diag | 输入 | Host | aclblasDiagType_t | UNIT/NON_UNIT，非法返回 INVALID_VALUE |
| m | 输入 | Host | int | m >= 0，m < 0 返回 INVALID_VALUE，m=0 为 no-op |
| n | 输入 | Host | int | n >= 0，n < 0 返回 INVALID_VALUE，n=0 为 no-op |
| alpha | 输入 | Host | const float* | 非 nullptr；alpha=0 时 A/B 不被引用 |
| A | 输入 | Device | const float* | 三角矩阵，列主序；alpha!=0 时非 nullptr |
| lda | 输入 | Host | int | side=LEFT: lda>=max(1,m)；RIGHT: lda>=max(1,n) |
| B | 输入 | Device | const float* | m×n 矩阵；alpha!=0 时非 nullptr |
| ldb | 输入 | Host | int | ldb >= max(1, m) |
| C | 输出 | Device | float* | m×n 输出矩阵；非 nullptr |
| ldc | 输入 | Host | int | ldc >= max(1, m) |

### 3.3 参数校验顺序

```
handle==nullptr → HANDLE_IS_NULLPTR
side/uplo/trans/diag 非法 → INVALID_VALUE（trans 仅 N/T，OP_C 非法）
m<0 || n<0 → INVALID_VALUE
m==0 || n==0 → SUCCESS（no-op，不校验指针/ld）
lda/ldb/ldc 约束违反 → INVALID_VALUE
alpha==nullptr → INVALID_VALUE
C==nullptr → INVALID_VALUE
alpha==0 → memset C 全零 → SUCCESS（A/B 不被引用）
A==nullptr || B==nullptr → INVALID_VALUE
正常计算路径
```

## 4 实现方案

### 4.1 总体架构：三段式管线

采用 AIV-AIC-AIV 三段式管线，同 stream 串行 launch，kernel 间依赖由 stream 顺序保证，无需 device 侧跨核同步原语：

```
aclblasStrmm (host)
  ├─ 参数校验 → quick return → alpha 解析（dev_alpha 时 D2H 回读）
  ├─ alpha==0 → aclrtMemsetAsync 清零 ldc×n（快路径，不 launch kernel）
  └─ 正常路径（同 stream 串行 launch）：
       [K1] strmm_mirror_kernel  (AIV_ONLY, ≤48 核)  A → workspaceA
       [K2] strmm_gemm_kernel    (AIC_ONLY, ≤24 核)  workspaceA × B → temp
       [K3] strmm_scale_kernel   (AIV_ONLY, ≤48 核)  temp × alpha → C
```

### 4.2 文件组织

| 文件 | 职责 |
|------|------|
| `blas/trmm/arch22/strmm_host.cpp` | API 入口：校验、tiling 计算、workspace 管理、三段 kernel launch |
| `blas/trmm/arch22/strmm_kernel.cpp` | 三个 kernel 实现：mirror(AIV) + gemm(AIC MatmulImpl) + scale(AIV) |
| `blas/trmm/arch22/strmm_tiling_data.h` | tiling 结构体定义 + arch22 常量 |

### 4.3 列主序归一化

列主序矩阵内存布局等价行主序转置视图。利用此性质，host 侧做 m↔n 交换 + side 翻转后，kernel 代码不感知列主序：

- 列主序 C(m×n) == 行主序 C^T(n×m)，令 GEMM 计算 M'=n, N'=m
- side=LEFT → 翻转为 kernel 侧 side=RIGHT，反之亦然
- B 以 ND 非转置读即得 B^T，无需额外搬运

16 组枚举组合（side×uplo×trans×diag）归一化后，GEMM 段仅需 3 个模板实例（plain / transA / transB）。

### 4.4 K1 Mirror 段（AIV）

产出 workspaceA = Ã（A 的三角区 + 非引用区全零），列主序，stride=ldw。

- **mirror 恒产出非转置 Ã**：trans 语义交由 K2 的 Matmul isTrans 消化（arch22 独有简化）
- uplo=UPPER：列 j 有效段为行 [0, j]，其余填零
- uplo=LOWER：列 j 有效段为行 [j, dimA)，其余填零
- diag=UNIT：对角位置强制 1.0f，不读取 A 对角（支持 NaN 对角判别）
- 每核按列分配（col = blockIdx; col += blockNum），段内 DataCopy + DataCopyPad 向量化搬运

### 4.5 K2 GEMM 段（AIC）

基于 `MatmulImpl` 高阶模板，静态 tiling：

- Tile 尺寸：128×128×64（fp32 下 L0 双缓冲最优，对齐仓内 complex_blas3 实测结论）
- 配置：`CONFIG_NORM`（equal-or-better at every shape）
- `SetOrgShape` 使用 5 参重载按转置模式填 stride（3 参重载无法表达三操作数互异 stride）
- K > 8192 时 host 分段 launch，首段覆写 + 后续段原子累加（enAtomic=0/1）
- 核数：`clamp(totalTiles, 1, 24)`

trans 分派组合表（翻转后视角）：

| 原始 side | trans=N | trans=T |
|-----------|---------|---------|
| LEFT | plain 实例（wsA 落 B 槽） | transB 实例（wsA 落 B 槽，isTrans=true） |
| RIGHT | plain 实例（wsA 落 A 槽） | transA 实例（wsA 落 A 槽，isTrans=true） |

### 4.6 K3 Scale 段（AIV）

temp 第 j 行 == C 第 j 列，行序一一对应，无需转置：

- 每核按列处理：DataCopyPad 读 temp 行 → Muls(alpha) → DataCopyPad 写 C 列
- 只写 m 个活动行，padding 行不触碰
- alpha==1.0f 时跳过 Muls（纯搬运直通）

### 4.7 Workspace 规划

```
wsBase ─┬─ workspaceA : dimA × ldw × 4B      (ldw = CeilAlign(dimA, 8))
        └─ temp       : n × tempRowStride × 4B (tempRowStride = CeilAlign(m, 8))
总需求 = CeilAlign(workspaceA_size, 32) + temp_size
```

使用 handle workspace 复用机制（EnsureDefaultWorkspace），峰值约 128MB（4096² 时），上限 2GB。

### 4.8 Tiling 结构

```cpp
struct StrmmMirrorTilingData {       // K1
    uint32_t uploMode, diagMode;
    uint32_t usedAivCoreNum, dimA, lda, ldw;
};
struct StrmmGemmTilingData {         // K2（翻转后视角）
    uint32_t m, n, k;
    uint32_t usedAicCoreNum, mBlocks, nBlocks;
    uint32_t ldw, ldb, tempRowStride, transMode;
};
struct StrmmScaleTilingData {        // K3
    uint32_t m, n, ldc, tempRowStride;
    uint32_t usedAivCoreNum;
    float alphaVal;
};
```

## 5 精度方案

### 5.1 精度标准

对齐生态算子开源精度标准，采用混合容差（Mixed Tolerance）逐元素比对：

| 数据类型 | rtol | atol | required_matched_ratio | max_abs_error_limit |
|----------|------|------|------------------------|---------------------|
| FLOAT32 | 2^-13 (1.22e-4) | 2^-13 (1.22e-4) | 0.99 | 1e-2 或 32×ULP |

- 逐元素通过条件：`|actual - golden| <= atol + rtol * |golden|`
- 整体通过条件：matched_ratio >= 0.99 且 max_abs_error <= max_abs_error_limit
- alpha==0 时使用 EXACT 位精确校验（结果应为全零）
- golden 由 Netlib cblas `cblas_strmm` 生成

### 5.2 精度保证措施

- 全链路 fp32 计算，禁止 SetHF32（HF32 的 10-bit 尾数精度约 1e-3，超出 2^-13 判定阈值）
- 三角裁剪 + 零填充将 TRMM 精确归约为稠密 GEMM，无近似
- alpha 缩放独立于 GEMM 累加，避免中间结果溢出

## 6 性能方案

### 6.1 性能目标

测试设备 Atlas 800T A2 (910B3)，有效采样 10 次取平均：

| case | m | n | 达标耗时(us) |
|------|---|---|-------------|
| 1 | 256 | 256 | 58.429 |
| 2 | 512 | 512 | 138.2 |
| 3 | 1024 | 1024 | 342.2 |
| 4 | 2048 | 2048 | 1182 |
| 5 | 4096 | 4096 | 6681 |

### 6.2 性能优化措施

| 优化项 | 说明 |
|--------|------|
| mirror 分段向量化 | 列段 DataCopy + 批量填零，替代逐元素操作 |
| scale 按列切分 | temp 行与 C 列双连续访存，替代跨步访问 |
| trans 由 Matmul isTrans 消化 | mirror 保持纯连续搬运，消除跨步读 |
| alpha=1 跳过 Muls | scale 段纯搬运直通 |
| 三角 K 裁剪 | GEMM 按输出块仅计算三角带内 K 区间，方阵类 shape 降低 58%~69% 计算量 |
| alpha=0 快路径 | 单次整块 memset，不 launch kernel |

### 6.3 性能实测

200 条性能用例中 192 条达标（96%），8 条未达标为 1×1 微小 shape（kernel launch 固定开销主导）和极限窄形（65×3259），5 个任务书关键 case 全部达标，余量中位 67.2%。

## 7 测试方案

### 7.1 测试框架

基于 ops-blas 仓 test/ 目录的 CSV 驱动 GTest 框架，测试工程位于 `test/trmm/strmm/arch22/`。

### 7.2 用例覆盖

| 类别 | 前缀 | 数量 | 覆盖内容 |
|------|------|------|----------|
| 基础功能 | TC_L0 | 16 | side×uplo×trans×diag 全 16 组合 |
| Shape 扫描 | TC_SS | ~900+ | 0/1/质数/2 幂/2 幂±1/非对齐/大规模 |
| LD Padding | TC_LD | ~50+ | lda/ldb/ldc 紧凑及多种 padding 场景 |
| 特殊值 | TC_SP | ~20+ | INF/NaN 输入、alpha 特殊值（0/1/-1） |
| 边界/负向 | TC_ED | 23 | null handle/alpha/A/B/C、非法枚举、OP_C 拒绝、neg m/n、非法 ld |
| 性能 | TC_PF | 200 | 全量性能基线用例 |
| 白盒守卫 | TC_WB | 9 | K 分段、三角裁剪空窗、mirror 尾部边界 |
| 直接 TEST_F | G1~G5 | 6 | NullHandle、AlphaZeroLdcPadding、AlphaZeroCompactLd、BInPlaceAliased、DevAlpha、NonTriangleAndUnitDiagNotReferenced |

总计 1220 用例（1214 CSV 参数化 + 6 TEST_F），全量通过。

### 7.3 自验结果

| 验证项 | 结果 |
|--------|------|
| 精度 | 1220/1220 全 PASS（matched_ratio >= 0.99，max_abs_error <= 阈值） |
| 性能 | 192/200 达标，5 个关键 case 全部达标 |
| 内存 | HBM 占用稳定（+1% ≈ 655MB），退出回落无泄漏，上限 2GB 防御 |
| 编译 | build.sh --soc=ascend910b3 --ops=strmm，RC=0 |

## 8 风险与应对

| 风险 | 影响 | 应对 |
|------|------|------|
| fp32 Cube 有效吞吐不足，case5(4096²) 逼近红线 | 性能 | 三角 K 裁剪降低 58%~69% 计算量；直写 C 优化可进一步省 0.3~0.6ms |
| Matmul isTrans 的 shape/stride 语义 | 精度 | 4 组合逐参填表 + 2×2/17×17 手算单验，回退路径 = mirror 内转置 |
| 非对齐 ld/尺寸的 DataCopy 边界 | 精度 | 全段 DataCopyPad 化 + TC_LD/TC_SP 专项回归 |
| K > 8192 分段静默限制 | 极大尺寸 | host 分段防御 + 白盒用例覆盖上界 |
| 3 kernel launch 固定开销致小 shape 紧张 | 性能 | 预估 15-30us 余量充足；极端情况可合并 K1/K3 |
