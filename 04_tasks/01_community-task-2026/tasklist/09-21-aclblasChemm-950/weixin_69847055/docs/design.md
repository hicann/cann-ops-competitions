# aclblasChemm 算子设计文档（Atlas 950PR）

# 需求背景（required）

## 需求来源

算子实操工坊-上海站-aclblasChemm算子开发(950) 社区任务任务书；对标 cuBLAS `cublasChemm`（语义参考 Netlib BLAS `chemm.f`）。

## 背景介绍

### aclblasChemm 算子定位

`aclblasChemm` 为单精度复数（complex64）厄米特矩阵乘算子：A 为厄米特矩阵（`A = A^H`，仅 `uplo` 指定三角被引用，对角虚部假定为 0），计算

- `side = LEFT`：`C = alpha * A * B + beta * C`（A 为 m×m）
- `side = RIGHT`：`C = alpha * B * A + beta * C`（A 为 n×n）

B、C 均为 m×n 列主序，alpha/beta 为复数标量。接口为 ops-blas 仓 `include/cann_ops_blas.h` 已有声明的句柄式 BLAS 接口（handle 绑定 stream 直调 NPU kernel），实现代码落 `blas/symm/arch35/`，测试代码落 `test/symm/chemm/arch35/`。

### 实现基础分析

ops-blas 仓 master 已具备同族可复用资产，本设计在其骨架上做复数化扩展：

| 资产 | 位置 | 可复用点 |
| --- | --- | --- |
| aclblasSsymm（已合入 arch35） | `blas/symm/arch35/ssymm_*` | 三 kernel 流水线（mirror → Cube GEMM → scale）、L1 分块 tiling、列主序行主序适配、alpha/beta host/device 指针解析、alpha==0/beta==0 快路径 |
| aclblasCherk（已合入 arch35） | `blas/herk/arch35/cherk_*` | 复数 split-real（解交织 → 实数域 GEMM → 交错合并）的成熟范式 |
| 实数 GEMM Cube kernel | `ssymm_kernel.cpp` | 直接以实/虚平面为操作数复用，无需新写 Cube 主循环 |

# 需求分析（required）

## 需求描述

使用 Ascend C 在 Ascend 950PR 上实现 `aclblasChemm`：功能/参数语义同 `cublasChemm`，精度满足生态算子开源精度标准（COMPLEX64 实部/虚部分别按 FLOAT32 判定），四个性能 case 达标，负向与边界行为对齐 cuBLAS。

## 需求拆解

1. 功能：side（LEFT/RIGHT）× uplo（UPPER/LOWER）全组合；m/n=0 合法 no-op；对角虚部按 0 处理；C 原地覆写。
2. 接口：签名与 `cann_ops_blas.h` 一致（`int64_t` 维度、`aclblasComplex*` 标量/矩阵）；禁止 950PR 私有平行接口。
3. 精度：实部/虚部分别比对，rtol=2^-10、atol=2^-16、matched_ratio≥0.99、max_abs_error≤1e-2（或 32×ULP）；golden 由 cblas（Netlib chemm）生成。
4. 性能（COMPLEX64，warmup 后有效采样 >50 次平均）：

| case | m | n | side | uplo | 标杆（Avg time, us） |
| --- | --- | --- | --- | --- | --- |
| 1 | 1024 | 1024 | LEFT | UPPER | 564.14 |
| 2 | 2048 | 2048 | LEFT | UPPER | 3824.68 |
| 3 | 1024 | 1024 | RIGHT | LOWER | 547.6 |
| 4 | 2048 | 2048 | RIGHT | LOWER | 3683.6 |

5. 负向与边界：handle nullptr / 非法枚举 / m,n<0 / lda、ldb、 ldc 越界 / alpha、beta nullptr / beta≠0 时 C nullptr 等，返回约定状态码；B 含 Inf/NaN 用例行为对齐 cublas。
6. 异步执行：依赖 `aclblasSetStream` 绑定的 stream，读回前同步。

# 详细设计（required）

## 算子分析

### 数学公式

记 A 的厄米特完全展开为 A_full：`A_full[i][j] = A[i][j]`（i、j 同在引用三角内）；`A_full[i][j] = conj(A[j][i])`（镜像填充）；对角元素取实部（虚部假定为 0）。则两种 side 统一为一次复数矩阵乘 `T = A_full · B`（LEFT）或 `T = B · A_full`（RIGHT），最终 `C = α·T + β·C_old`。

复数乘 split-real 展开（4 实 GEMM）：设 A_full 平面 (Ar, Ai)、B 平面 (Br, Bi)，则

- `Tr = Ar·Br − Ai·Bi`
- `Ti = Ar·Bi + Ai·Br`

### 支持数据类型与形状

- 数据类型：COMPLEX64（实部/虚部各 float32），存储 ND 列主序。
- 形状：A 为 dimA×dimA（side=LEFT 时 dimA=m，RIGHT 时 dimA=n），B/C 为 m×n；m、n ≥ 0。

## 算子实现

### 实现方案总览

沿用 ssymm 的「预处理 → Cube GEMM → 收尾」三段式，扩展为四类 kernel 的流水线（同 stream 顺序发射，天然保序）：

```
A(m×m complex, 三角) ──mirror(AIV)──▶ A_full(complex, workspace)
A_full ──deinterleave(AIV)──▶ Ar, Ai（实数平面, 行距 32B 对齐）
B ──────deinterleave(AIV)──▶ Br, Bi
(Ar,Ai,Br,Bi) ──gemm×4(AIC Cube)──▶ T1,T2,T3,T4（实数平面）
T1..T4 ──combine_scale(AIV)──▶ C = α·(T1−T2 + i(T3+T4)) + β·C_old
```

### 3.2.1 host 侧设计

**参数校验链**（与 ssymm 同序）：

1. early check：handle 非空（否则 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`）、side/uplo 枚举合法（否则 `INVALID_ENUM`）；
2. quick return：m==0 或 n==0 直接返回 SUCCESS（不触碰指针，BLAS 标准）；
3. 维度校验：m,n ≥ 0；side=LEFT 时 lda ≥ max(1,m)、RIGHT 时 lda ≥ max(1,n)；ldb ≥ max(1,m)；ldc ≥ max(1,m)，否则 `INVALID_VALUE`；
4. 复数标量解析：`aclrtPointerGetAttributes` 判定 alpha/beta 位于 host 还是 device，二者 mode 必须一致（混合拒绝）；device 模式经 `ReadAlphaBetaFromDevice` 同款 D2H（8 字节复数）+ stream 同步后取值；
5. BLAS 语义：alpha≠(0,0) 时 A/B 不可为 nullptr；beta≠(0,0) 时 C 不可为 nullptr；beta==(0,0) 且 C==nullptr 直接返回 SUCCESS（避免向空地址发射 kernel 污染 stream）。

**快路径**（host 判读标量值后分流，省全部 GEMM）：

- alpha==(0,0)：beta==(0,0) → `aclrtMemsetAsync(C)`；beta==(1,0) → 原样返回；其余 → 仅发射复数 scale kernel（C = β·C）。
- beta==(0,0)：combine 阶段跳过 C_old 读入（C = α·T）。

**workspace 布局**（`EnsureDefaultWorkspace` 一次分配）：

```
[A_full | Ar | Ai | Br | Bi | T1 | T2 | T3 | T4]
  dimA²     m×n 各平面 ×8B，平面行距 planeLd = CeilAlign(行元素数, 8) float（32B 对齐）
```

总字节数 = 8B × [dimA² + 5·m·n（含 4 temp）]（按对齐行距上取），2048² 约 336MB，量级与 ssymm 同阶。

**tiling 结构**（三份，对应三类 kernel）：

1. mirror tiling：dimA、lda、side、uplo、rowsPerCore=CeilDiv(dimA, usedAivCoreNum)、nthreads；
2. gemm tiling：沿 ssymm——tileM/tileN/tileKChunk + L1 预算收敛循环（`2×(aSideL1+bSideL1) > L1_SIZE` 时 tileKChunk 减半）、tempRowStride、usedAicCoreNum = min(CeilDiv(m,BASE_M)×CeilDiv(n,BASE_N), aicCoreNum)；
3. combine tiling：m、n、ldc、rowsPerCore、复数 α/β 值、betaIsZero 标志。

**列主序适配**（沿用 ssymm 已验证方案，实数平面同样成立）：gemm kernel 内部按行主序计算 Cᵀ——host 侧交换 m↔n 并翻转 side，使 LEFT/RIGHT 互换；平面 GEMM 的四个操作数指针按 side 选择：

- LEFT：`(A侧=Ar/Ai, B侧=Br/Bi)`，即 T1=Ar·Br、T2=Ai·Bi、T3=Ar·Bi、T4=Ai·Br；
- RIGHT：操作数角色对调（T1=Br·Ar 等），lda/ldb 不交换（gemm kernel 对 aGlobal/bGlobal 各自用其前导维度访问）。

四个平面 GEMM 复用同一 `ssymm_gemm` 主循环，仅指针与 temp 落点不同，4 次顺序发射。

**不设 tilingkey**：side/uplo/快路径差异全部由 tiling 结构体字段与发射编排表达，kernel 二元唯一，避免编译膨胀（单 GEMM 模板实例）。

### 3.2.2 kernel 侧设计

**1）mirror kernel（AIV，SIMT 标量）**：每核认领 dimA 的连续行段，行内扫描引用三角：

- 引用三角元素：从 A 原位读出写 A_full；
- 镜像位置：读 `A[j·lda + i]` 写 `conj`（虚部取负）到 `A_full[i·dimA + j]`；
- 对角：写实部、虚部写 0（对齐「对角虚部假定为 0」语义，输出确定值）。

纯标量 GM 读写，2048² 约 4.2M 元素、33.5MB 流量，微秒级（<50μs），不构成瓶颈。

**2）deinterleave kernel（AIV）**：A_full 与 B 的复数交错数据拆为实/虚平面。Bᵀ 行（即 B 的列）在内存中连续，按段读入后奇偶分离写两平面；写侧两平面基址均 32B 对齐、行距 8 float 对齐（规避 MTE DataCopyPad 非对齐 dst 的静默脏数据问题，该问题在 ctpmv LOWER 变体已有实证）。带宽型操作，2048² 约 100MB 流量，<100μs。

**3）gemm kernel（AIC，Cube）**：完整复用 `ssymm_gemm` 主循环——L1 内 tileM×tileKChunk / tileKChunk×tileN 双缓冲搬运、主循环 FIXPIPE 输出、K 维 chunk 循环；输入输出均为 float32 实数平面，无复数逻辑。四平面 GEMM 顺序发射，共享同一 tiling。

**4）combine_scale kernel（AIV，SIMT 标量/向量混合）**：每核认领 C 的行段，逐元素：

```
Cr = αr·(T1−T2) − αi·(T3+T4) + βr·Cold_r − βi·Cold_i
Ci = αr·(T3+T4) + αi·(T1−T2) + βr·Cold_i + βi·Cold_r
```

betaIsZero 时跳过 C_old 项；同元素读后写、无跨元素依赖，无需核间同步。O(m·n) 带宽型，<100μs。

**arch35 工程约束**（设计内建，均为仓内已实证条款）：

- 全 kernel 不用 set_flag/wait_flag 管道配对（arch35 编译期拒绝），同步只用 `PipeBarrier`，kernel 间依赖由同 stream 顺序保证；
- MTE 搬运 dst 32B 对齐；索引算术全程 u32/u64 显式宽度，避免 64 位索引算术打满向量 ALU；
- 负向路径（参数校验）全部 host 侧完成，kernel 侧无分支负担。

### 性能预算

| 阶段 | 1024² 流量/算力 | 2048² | 说明 |
| --- | --- | --- | --- |
| mirror + deinterleave | ~75MB | ~600MB | 带宽型，@1.6TB/s ≈ 47/375μs…实测预期减半（纯单向流） |
| 4×GEMM | 8.6 GFLOP | 68.7 GFLOP | 需持续 15.2/17.9 TFLOPS FP32，对照 cherk arch35 同族已达标余量 |
| combine | ~50MB | ~200MB | 带宽型 |

标杆折算：case1 需 564μs 内完成 8.6 GFLOP = 15.2 TFLOPS；case2 需 3824μs 内完成 68.7 GFLOP = 17.9 TFLOPS。参照同族实证——cherk/cher2k（arch35，split-real 实数 GEMM 同路子）1024² 全案 642μs（≈2× chemm 计算量）——本方案预期 case1 ≈ 350~480μs、case2 ≈ 2400~3300μs，标杆余量 15%~40%。预处理/收尾三段 O(m²+n²) 带宽开销随 n³ 计算量摊薄，不构成瓶颈。

后续优化档（首版不启用，留性能迭代空间）：① 4 GEMM 合并为 Karusat 3 GEMM；② T1/T2 在 FIXPIPE 侧融合减法省 1 个 temp 平面；③ mirror 与 A 平面 deinterleave 融合为单遍。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas 950PR（DAV_3510，CANN 9.1.0） | √ |

## 算子约束限制

- 仅支持 COMPLEX64；非连续 Tensor（超出 lda/ldb/ldc 语义）不支持；
- 不要求 dynamic shape、不涉及广播；确定性计算不要求；
- 对角虚部不校验，按 0 语义输出；引用三角外的输入值不参与计算。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度 | COMPLEX64 实/虚部分别按 FLOAT32：rtol 2^-10、atol 2^-16、matched_ratio ≥ 0.99、max_abs_error ≤ 1e-2 或 32×ULP；golden = cblas chemm | 任务书 §3.2 / opbase 精度标准 |
| 性能 | 4 个标杆 case Avg time 不高于 §需求拆解.4 表值；warmup 后有效采样 >50 次平均；msprof kernel Duration 口径出具报告 | 任务书 §3.3 |
| 内存 | workspace 用量实测并出报告（公式+实测双口径） | 任务书 §3.4 |

## 测试设计

1. 测试工程落 `test/symm/chemm/arch35/`，CSV 驱动 GTest（结构同 test/symm/ssymm）：
   - 功能/精度：side×uplo 全组合 × m/n ∈ {0,1,2,3,7,8,64,255,256,1024,2048} × lda/ldb/ldc 紧凑与 padding × alpha/beta 特殊值（(0,0)、(1,0)、纯虚、大值）× 均匀/正态分布 50/50；
   - 负向：非法枚举、负维度、lda/ldb/ldc 越界、alpha/beta nullptr、beta≠0 且 C nullptr、handle nullptr；
   - 特殊值：B 含 Inf/NaN（行为对齐 cublas）。
2. 自测报告：用例参数、实/虚部精度截图、性能数据截图（msprof）、内存数据，按任务书 §4 交付件清单组织。

## 兼容性分析

新算子，接口声明使用 `include/cann_ops_blas.h` 既有 `aclblasChemm` 原型，不引入平行接口，不涉及兼容性问题。
