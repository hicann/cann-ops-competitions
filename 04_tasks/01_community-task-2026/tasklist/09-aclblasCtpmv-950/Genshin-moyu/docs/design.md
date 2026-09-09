# aclblasCtpmv 算子设计文档

# 需求背景（required）

## 需求来源

本需求来自算子实操工坊-上海站社区任务"aclblasCtpmv 算子开发(950)"（任务页：https://www.hiascend.com/activities/task-center/details/87b85654a9444c119727ee5339f149df ）。

目标是在 **Ascend 950PR** 产品上，基于 `cann/ops-blas` 开源仓工程框架，以 Ascend C（SIMT VF 编程模型）Kernel 直调方式实现单精度复数三角压缩存储矩阵-向量乘接口 `aclblasCtpmv`。

接口参数序列与 cuBLAS `cublasCtpmv` 逐参数对齐；packed 存储、转置/共轭、单位对角、负步长与 quick return 语义参考 Netlib `ctpmv.f`（参考实现已精读全部四段循环）。开发完成后，公共接口声明放入 `include/cann_ops_blas.h`（紧随同族实数接口 `aclblasStpmv` 之后，不定义 950PR 私有平行接口），实现放入 `blas/tpmv/arch35/`，测试放入 `test/tpmv/ctpmv/arch35/`。

## 背景介绍

### 算子功能

`aclblasCtpmv` 属于 BLAS Level-2，计算：

```text
x := op(A) * x
```

其中 A 为 n×n 单精度复数（complex64，实部/虚部各 FP32 交错存储）三角矩阵，以 packed（列优先压缩）格式存于向量 AP（长度 n(n+1)/2，无前导维参数 lda）；x 为 n 元复数向量，原地覆写输出。

- `trans = ACLBLAS_OP_N`：op(A) = A
- `trans = ACLBLAS_OP_T`：op(A) = A^T（转置不共轭）
- `trans = ACLBLAS_OP_C`：op(A) = A^H（共轭转置）

### 接口现状

ops-blas 仓 `include/cann_ops_blas.h` 当前尚无 `aclblasCtpmv` 声明，需新增；同族实数接口 `aclblasStpmv`（arch35 已有实现，SIMT VF 路线）为本算子的工程母版。

# 需求分析（required）

## 需求描述

使用 Ascend C 在 Ascend 950PR（arch35，CANN 9.1.0）上实现 `aclblasCtpmv` 句柄式 BLAS 接口：通过 handle 绑定 stream 直调 NPU kernel，支持 12 组枚举组合（uplo 2 × trans 3 × diag 2）全覆盖、正/负步长、n=0 快速返回与全量参数校验，精度满足生态算子开源精度标准，性能不高于任务书标杆。

## 需求拆解

1. 12 组枚举组合（uplo × trans × diag）功能全覆盖
2. packed 列优先存储正确索引（UPPER: `AP[i+j*(j+1)/2]`；LOWER: `AP[i+(2n-j-1)*j/2]`，与 cblas/LAPACK 0-based 约定一致）
3. diag=UNIT 时主对角假定为 1 且**不读取** AP 对角位置
4. 原地覆写语义：x 输入旧值、输出新值（12 组合均为纯矩阵-向量乘，无解算型链式依赖）
5. 参数校验与异常返回码（INVALID_ENUM / INVALID_VALUE / HANDLE_IS_NULLPTR / n=0 no-op）
6. 精度：实部/虚部分别按 FLOAT32 判定（rtol 2^-10、atol 2^-16、ratio≥0.99、max_abs≤1e-2 或 32×ULP）
7. 性能：4 条典型 case 不高于标杆（44.89/88.46/184.2/1842.81 μs）

# 详细设计（required）

## 算子分析

### 数学公式

y(i) = Σ_j op(A)(i,j) · x(j)，其中 j 取该 uplo/trans 组合下 op(A) 第 i 行的非零列。

12 组合求和范围与元素取值：

| # | uplo | trans | diag | y(i) 求和范围 | op(A)(i,j) 取值 |
|---|---|---|---|---|---|
| 1 | UPPER | N | NON_UNIT | j ∈ [i, n) | AP[i+j(j+1)/2] |
| 2 | UPPER | N | UNIT | 同上，对角=1（不读） | 同上 |
| 3 | LOWER | N | NON_UNIT | j ∈ [0, i] | AP[i+(2n-j-1)j/2] |
| 4 | LOWER | N | UNIT | 同上，对角=1（不读） | 同上 |
| 5 | UPPER | T | NON_UNIT | j ∈ [0, i] | AP[j+i(j+1)/2] |
| 6 | UPPER | T | UNIT | 同上，对角=1（不读） | 同上 |
| 7 | LOWER | T | NON_UNIT | j ∈ [i, n) | AP[j+(2n-i-1)i/2] |
| 8 | LOWER | T | UNIT | 同上，对角=1（不读） | 同上 |
| 9 | UPPER | C | NON_UNIT | j ∈ [0, i] | conj(AP[j+i(j+1)/2]) |
| 10 | UPPER | C | UNIT | 同上，对角=1（不读） | 同上 |
| 11 | LOWER | C | NON_UNIT | j ∈ [i, n) | conj(AP[j+(2n-i-1)i/2]) |
| 12 | LOWER | C | UNIT | 同上，对角=1（不读） | 同上 |

语义要点（源自 Netlib ctpmv.f 精读）：

- **12 组合均为纯 matvec**：每个输出只依赖全部 x 旧值，Netlib 的原地遍历顺序经过设计保证读旧写新，不存在 tpsv 式链式依赖
- **全程无除法**（tpmv 为纯乘加，除法属 tpsv），无病态放大风险
- **逐组合累加顺序**：对角项先乘入，N+UPPER 与 T/C+LOWER 非对角项按列升序累加，N+LOWER 与 T/C+UPPER 按列降序累加——与 Netlib 参考实现的求值顺序一致，最小化浮点舍入差异

### 支持数据类型

COMPLEX64（aclblasComplex：实部/虚部各 FLOAT32 交错），AP 长度 n(n+1)/2，x 长度 1+(n-1)*|incx|。

### 支持形状

n 为运行时入参（0 ≤ n，int），无 shape 编译期约束；incx 支持正/负步长（负步长按 Netlib 语义从物理尾部反向遍历），incx=0 非法。

## 算子实现

### 实现方案

采用 **workspace-y 并行方案** + **SIMT VF kernel 直调**：

```text
x_old ──┐
        ├─► [kernel] y = op(A)·x_old（行并行，写独立 workspace）
AP ─────┘        │
                 ├─ incx == 1 → aclrtMemcpy D2D 拷回 x
                 └─ incx != 1 → [scatter kernel] 按步长散写回 x
```

选择依据：
- workspace-y 消除原地读写依赖：kernel 全部输入读取均为 x_old，输出写独立缓冲 y（n 个 complex64，n≤4096 时仅 32KB），与 Netlib 原地遍历数学等价，天然规避多线程冲突
- 行并行：每行一个归约，行间零同步
- 与仓内同族 `stpmv`（arch35，SIMT VF）工程结构同构，复用已验证的 launch/编排链路

#### host 侧设计

**校验策略**（顺序对齐 Netlib：参数检查先于 quick-return，n=0 不引用 AP/x）：

| 条件 | 返回码 |
|---|---|
| handle 为 nullptr | ACLBLAS_STATUS_HANDLE_IS_NULLPTR |
| uplo / trans / diag 非法枚举 | ACLBLAS_STATUS_INVALID_ENUM |
| n < 0 或 incx == 0 | ACLBLAS_STATUS_INVALID_VALUE |
| n == 0 | 合法 no-op，返回 SUCCESS |
| n > 0 且 AP/x 为 nullptr | ACLBLAS_STATUS_INVALID_VALUE |

**网格策略**：查询平台 AIV 核数 → `numBlocks = min(n, AIV核数)`，`numThreads = clamp(CeilAlign(CeilDiv(n, numBlocks), SIMT_MIN_THREAD_NUM), …, SIMT_MAX_THREAD_NUM)`——小 n 退化为少线程单块，n=4096 满载。

**编排**：分配 workspace（n×8B，ACL_MEM_MALLOC_HUGE_FIRST）→ 发射 kernel + stream 同步 → 回写（incx=1 走 D2D memcpy，否则 scatter kernel）→ 释放。kernel 与回写在 handle 绑定的同一 stream 上保序。

#### kernel 侧设计

**模板分发**：`CtpmvSimt<UPPER, TRANS_MODE, UNIT>`，TRANS_MODE ∈ {0=N, 1=T, 2=C}，共 2×3×2=12 份模板实例，编译期消分支；入口按 tiling 字段静态分发（dim3{numThreads}，grid-stride 行循环）。

**行负载均衡**：三角形负载不均（行长度 0~n-1），采用交错行映射（k 偶→前半行，k 奇→尾部对称行）摊平线程间总负载。

**累加顺序**（每组合与 Netlib 对齐）：sum 以对角项初始化（UNIT 时为 1.0×x_old(i) 不读 AP），非对角项按 §数学公式 表的方向累加。

**复数展开**：AP/x 以 `__gm__ float*` 视图按 2*idx（实）/2*idx+1（虚）寻址；复数乘加展开为 4 mul + 2 add；`#pragma clang fp contract(off)` + volatile 中间量阻止 FMA 融合，保持与 CPU BLAS 一致的求值舍入（该开关经真机 A/B 验证对结果无影响，按同族 stpmv 惯例保留 off）。

**转置/共轭**：trans=T/C 时元素下标交换 (i,j)→(j,i)（三角方向随之翻转），C 组合对 A 元素虚部取负（含对角，UNIT 跳过）。

**UNIT 对角**：`if constexpr (UNIT)` 分支不发射对 AP 对角位置的读指令（编译期消除，满足"不得读取"约束）。

**特殊值行为**：Inf/NaN 按浮点规则自然传播；已知差异：ccec 编译的 SIMT kernel 对 NaN 输入参与乘加时可能产生 ±Inf 载荷（CPU cblas 为 ±NaN 载荷；torch_npu 探针证实非平台级行为，系 SIMT 代码生成特性）。测试对特殊值用例采用非有限性等价判定（见可维可测分析）。

**scatter kernel**：incx≠1 时把连续 workspace 按 `incx ≥ 0 ? i·|incx| : (n-1-i)·|incx|` 散写回 x（每元素 2 float）。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas A5 推理系列（Ascend 950PR） | √ |

## 算子约束限制

- 仅支持 arch35（950PR/950DT）；非 arch35 构建跳过本算子及测试目录（CMake 防护）
- 不支持超出 incx 语义的非连续内存访问（与任务书一致）
- n 上限受 device 内存约束（packed 存储 n=4096 时 AP≈67MB）

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述(不涉及说明原因) | 标准来源 |
| --- | --- | --- |
| 精度标准 | 实/虚分别按 FLOAT32：rtol 2^-10、atol 2^-16、matched_ratio≥0.99、max_abs≤1e-2 或 32×ULP（逐元素：\|actual-golden\| ≤ atol + rtol×\|golden\|） | 生态算子开源精度标准 experimental_standard.md（任务书 §3.2） |
| 性能标准 | n=512/1024/2048/4096 四条典型 case 平均单次耗时 ≤ 44.89/88.46/184.2/1842.81 μs（warmup 后采样 >50 次均值） | 任务书 §3.3 |

**真机实测（Ascend 950PR，CANN 9.1.0，HiDevLab）**：

- 精度：随任务提供的 1000 条精度用例 **1000/1000 全部通过**（ALL PASS）；MERE 典型值 1e-6 量级（阈值 2^-13 的 1/100）
- 验证策略：golden 由 cblas（Netlib 复数实现）生成；有限值用例按上表混合容差判定（测试框架内置 `applyMixedTolerance(ACL_FLOAT)`）；golden 含非有限值的用例（Inf/NaN 填充）采用非有限性等价判定（both-nan / both-inf 视为等价，仅验证传播性，依据任务测试指导的特殊值用例调整条款）

## 兼容性分析

新算子，不涉及兼容性分析。公共声明进入 `include/cann_ops_blas.h` 供各产品线共用，不定义 950PR 私有平行 API；`blas/tpmv/README.md` 产品支持表标注 Ascend 950PR：支持。
