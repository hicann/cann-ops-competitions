# 需求背景（required）

## 需求来源

CANN 社区任务（2026 年 9 月）：aclblasCher2k 算子开发（Ascend 950PR）。任务要求基于 ops-blas 开源仓（https://gitcode.com/cann/ops-blas）工程框架，使用 Ascend C 在 Ascend 950PR（arch35，CANN 9.1.0）上实现 BLAS Level 3 单精度复数 Hermitian 秩-2k 更新算子 `aclblasCher2k`，完成算子设计、开发、测试全流程。验收通过后算子代码合入 `blas/herk/arch35/`，测试代码合入 `test/herk/cher2k/arch35/`，接口声明新增至公共头文件 `include/cann_ops_blas.h`（供其他产品线共用，不定义 950PR 私有平行接口）。

## 背景介绍

### 算子功能定位

`aclblasCher2k` 为 BLAS Level 3 单精度复数（complex64）Hermitian 秩-2k 更新算子：以两个一般复矩阵 A、B 的乘积项及其共轭项对 n×n Hermitian 矩阵 C 做**原地**更新。C 仅 `uplo` 指定的上三角（ACLBLAS_UPPER）或下三角（ACLBLAS_LOWER）被引用和更新，另一三角不被引用；输出 C 保持 Hermitian 性质 `C[i][j] = conj(C[j][i])`，对角元素虚部假定并强制置 0。矩阵均按列主序（Column-Major）存储；`trans` 仅支持 OP_N 与 OP_C（Hermitian 秩-2k 语义不定义普通转置，OP_T 为合法枚举但不支持）。

### 业界对标

- cuBLAS `cublasCher2k`（https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-her2k）：核心功能与参数序列与本算子一一对应（handle 及参数顺序逐项对齐，无额外映射）；
- Netlib BLAS `cher2k` 参考实现（https://www.netlib.org/blas/cher2k.f）：quick return、TRANS='T' 报错、对角虚部置 0 等行为依据；精度比对 golden 亦由 cblas（Netlib BLAS 复数实现）生成。

### 工程形态

采用 ops-blas 仓 Ascend C kernel 直调模式：`aclblasCher2k` 为句柄式 BLAS 接口，通过 `aclblasHandle_t` 绑定 stream 直调 NPU kernel，异步执行，读回 Device 结果前须同步 stream。实现与同族算子 `aclblasCherk` 同目录（`blas/herk/arch35/`），复用仓内 arch35 已有的 GEMM kernel、复数解交织与合成编排资产。

### 接口规格

```cpp
aclblasStatus_t aclblasCher2k(
    aclblasHandle_t handle, aclblasFillMode_t uplo, aclblasOperation_t trans,
    int n, int k, const aclblasComplex* alpha,
    const aclblasComplex* A, int lda, const aclblasComplex* B, int ldb,
    const float* beta, aclblasComplex* C, int ldc);
```

| 参数 | 含义 | 类型 | 约束与异常行为 |
| --- | --- | --- | --- |
| handle | 库上下文句柄，携带 stream | aclblasHandle_t，Host | 指向已创建的有效句柄；为 nullptr 返回 HANDLE_IS_NULLPTR |
| uplo | C 的三角引用模式 | 枚举，Host | {UPPER(121), LOWER(122)}；越界值返回 INVALID_ENUM |
| trans | 对 A、B 的操作 | 枚举，Host | {OP_N(111), OP_C(113)}；OP_T(112) 返回 INVALID_VALUE；越界值返回 INVALID_ENUM |
| n | C 的阶数 | int，Host | n ≥ 0；n < 0 返回 INVALID_VALUE；n = 0 合法 quick return |
| k | op(A)/op(B) 的列数(N)/行数(C) | int，Host | k ≥ 0；k < 0 返回 INVALID_VALUE；k = 0 且 beta = 1 合法 quick return |
| alpha | 复数标量乘数指针 | complex64 标量，Device | nullptr 返回 INVALID_VALUE；实/虚部取值于 FLOAT32 全集 |
| A | 矩阵 A，只读，列主序 | complex64，Device | trans=N 时 n×k，trans=C 时 k×n；n>0 且 k>0 时 nullptr 返回 INVALID_VALUE |
| lda | A 的前导维度 | int，Host | trans=N 时 ≥ max(1,n)，trans=C 时 ≥ max(1,k)；否则 INVALID_VALUE |
| B | 矩阵 B，只读，列主序 | complex64，Device | 形状同 A；n>0 且 k>0 时 nullptr 返回 INVALID_VALUE |
| ldb | B 的前导维度 | int，Host | 同 lda 规则；否则 INVALID_VALUE |
| beta | 实数标量乘数指针 | float32 标量，Device | nullptr 返回 INVALID_VALUE；beta = 0 时 C 不必是有效输入 |
| C | n×n Hermitian 矩阵，原地输入输出 | complex64，Device | 仅 uplo 三角更新，对角虚部强制置 0；beta 非零且 n>0 时 nullptr 返回 INVALID_VALUE |
| ldc | C 的前导维度 | int，Host | ≥ max(1,n)；否则 INVALID_VALUE |

返回值 `aclblasStatus_t`，状态码语义与仓内 `include/cann_ops_blas_common.h` 定义一致。

### 数学公式

```
trans = OP_N:  C = alpha * A * B^H + conj(alpha) * B * A^H + beta * C    （A、B 为 n×k）
trans = OP_C:  C = alpha * A^H * B + conj(alpha) * B^H * A + beta * C    （A、B 为 k×n）
```

其中 alpha 为单精度复数标量，beta 为单精度实数标量；H 表示共轭转置。

# 需求分析（required）

## 需求描述

在 Ascend 950PR 上实现 `aclblasCher2k`，支持 complex64 输入（alpha 复数、beta 实数）、uplo（UPPER/LOWER）× trans（N/C）全组合、任意 n ≥ 0 / k ≥ 0（含前导维 padding 与 quick return 场景）；精度满足生态算子开源精度标准（golden 为 Netlib cblas，实部/虚部按 FLOAT32 分量判定）；4 个典型性能 case 的平均单次耗时不高于标杆（641.43 / 4157.94 / 680.47 / 4258.12 us）；以 ops-blas 仓测试框架完成 1200 条随任务用例的自验并交付自测报告。

## 需求拆解

1. **功能**：实现接口签名与 cuBLAS `cublasCher2k` 一致的句柄式接口；全量参数校验（含 INVALID_ENUM / INVALID_VALUE 区分、C 指针按 beta 分支的口径）；quick return（n=0，或 (alpha=(0,0) 或 k=0) 且 beta=1）；beta=0 时不读旧 C；仅更新 uplo 三角且对角虚部置 0，非 uplo 三角与 ldc padding 字节保持原样。
2. **精度**：golden 由 cblas（Netlib cher2k）生成，仅验证 uplo 三角（含对角），实部/虚部分别按 FLOAT32 判定：rtol 2^-10、atol 2^-16、matched_ratio ≥ 0.99、max_abs_error ≤ 1e-2 或 32 ULP。
3. **性能**：Ascend 950PR 实测，COMPLEX64 场景平均单次耗时（warmup 后有效采样 >50 次取平均）不高于 4 case 标杆。
4. **交付**：设计文档、自测用例及测试代码（README 含复现步骤）、自测报告（实/虚部分别截图、性能、内存数据）、个人代码仓与算子 README（产品支持表标注 Ascend 950PR：支持）。

# 详细设计（required）

## 算子分析

### 数学公式

复数 GEMM 拆为实数 4M 分解。记 A = Ar + i·Ai，B = Br + i·Bi，产生 4 个 n×n 实矩阵 t1..t4（列主序，行宽 tempLdc = CeilAlign(n,16)）：

| 路径 | t1 | t2 | t3 | t4 |
| --- | --- | --- | --- | --- |
| OP_N | Ar·Br^T | Ai·Bi^T | Ai·Br^T | Ar·Bi^T |
| OP_C | Ar^T·Br | Ai^T·Bi | Ar^T·Bi | Ai^T·Br |

记 `M = alpha·P`（P = A·B^H 或 A^H·B），则 `M_R = t1 + t2`，`M_I = t3 − t4`。第二项满足共轭恒等式 `conj(alpha)·B·A^H = M^H`，故：

```
C = M + M^H + beta·C，M = M_R + i·M_I
CR[i][j] = ar*(u+w) - ai*(v+x) + beta*C_R[i][j]
CI[i][j] = ar*(v-x) + ai*(u-w) + beta*C_I[i][j]
（u=M_R[i][j], v=M_I[i][j], w=M_R[j][i], x=M_I[j][i], alpha=ar+i*ai）
```

M^H 项通过转置偏移读 t1..t4 获得（列主序下坐标互换即转置），无需额外临时矩阵。合计 **4 次实数 GEMM**。对角元素满足 u=w、v=x，使 `CI[i][i] = ar*(v−v) + ai*(u−u) = 0`，即对角虚部**代数相消精确为 0**，无需额外置零指令。

### 支持数据类型

| 数据类型 | 说明 |
| --- | --- |
| COMPLEX64（aclblasComplex = {float real; float imag}） | A、B、C、alpha |
| FLOAT32 | beta |

### 支持形状

| 矩阵 | OP_N | OP_C | 布局 |
| --- | --- | --- | --- |
| A、B | n×k | k×n | 列主序，前导维 lda/ldb 可含 padding |
| C | n×n | n×n | 列主序，原地输出，仅 uplo 三角更新 |

n、k 为运行时入参，n/k 覆盖 0、1、奇数、2 的幂及 ±1、非对齐值直至 2048（精度）/ 4096（性能）；不要求 broadcast、非连续 Tensor（超出 ld 语义）与确定性计算。

## 算子实现

### 实现方案

#### 总体架构与数据流

采用 **Cube+AIV mix 三阶段** 主路径（OP_N/OP_C 通用），辅以 **SIMT 小规格路径**（n ≤ 16）：

```
A(n×k|k×n, cplx, GM) ─┐ Phase0（AIV SIMD，1 次发射，64×64 tile，DeInterleave）
B（同形，cplx, GM）──┘→ Ar、Ai、Br、Bi（rows×cols FP32，GM，紧凑 ld=rows）
                        │ Phase1（AIC Cube，4 次实 GEMM，同一份 tiling）
                        ├→ t1、t2（M_R = t1 + t2）
                        └→ t3、t4（M_I = t3 − t4）
                        │ Phase2（AIV SIMD，1 次发射，64×64 tile，读 (i,j) 块 + (j,i) 转置块）
C(n×n, cplx, GM, 原地) ←┘  C_tri = M_tri + M^H_tri + beta*C；非 uplo 三角与 padding 不写
```

alpha=(0,0) 或 k=0 时跳过 Phase0/1（不分配临时空间），仅由 Phase2 做 beta 缩放或清零；n ≤ 16 时走 SIMT 融合直算路径，规避 6 次发射与临时空间开销。三阶段全部 enqueue 到同一 stream，由 stream 顺序保证依赖，Host 不做额外同步。

#### host 侧设计

**校验顺序**（先完成全量校验，再 quick return；序号即实现顺序）：

| # | 校验项 | 失败返回 |
| --- | --- | --- |
| 1 | handle 为 nullptr | HANDLE_IS_NULLPTR |
| 2 | uplo ∈ {UPPER, LOWER} | INVALID_ENUM |
| 3 | trans ∈ {OP_N, OP_T, OP_C} | INVALID_ENUM |
| 4 | trans = OP_T（合法枚举但不支持） | INVALID_VALUE |
| 5 | n ≥ 0 且 k ≥ 0 | INVALID_VALUE |
| 6 | lda ≥ (OP_N ? max(1,n) : max(1,k)) | INVALID_VALUE |
| 7 | ldb 同 lda 规则 | INVALID_VALUE |
| 8 | ldc ≥ max(1,n) | INVALID_VALUE |
| 9 | alpha ≠ nullptr 且 beta ≠ nullptr | INVALID_VALUE |
| 10 | n>0 且 k>0 时 A、B ≠ nullptr | INVALID_VALUE |
| 11 | C = nullptr 且 n>0：读 beta 判定（beta≠0 报错；beta=0 返回 SUCCESS 且无写出） | INVALID_VALUE / SUCCESS |

alpha（2 个 float）与 beta（1 个 float）为 Device 标量，经 staging 读回仅用于分支判定与合成阶段标量；#11 的 C 空指针分支只读 beta。

**分支表**（校验全过后）：

| 条件 | 行为 | kernel |
| --- | --- | --- |
| n = 0 | SUCCESS | 无 |
| (alpha=(0,0) 或 k=0) 且 beta=1 | SUCCESS，不访问 A/B/C 数据 | 无 |
| (alpha=(0,0) 或 k=0) 且 beta=0 | 选定三角写零，不读 A/旧 C | K3（跳过临时矩阵） |
| (alpha=(0,0) 或 k=0) 且其他 beta | 仅按 beta 缩放选定三角 | K3（跳过临时矩阵） |
| 一般乘积，beta=0 | 计算 M，不读旧 C | K1 + 4×K2 + K3 |
| 一般乘积，beta≠0 | 计算 M 并合成旧 C | K1 + 4×K2 + K3 |
| n ≤ 16（非上述场景） | SIMT 融合直算，无临时空间 | K4 |

**workspace 规划**（内部实现细节）：`arBytes = rows×cols×4B`（Ar/Ai/Br/Bi 四份同大小），`tempBytes = tempLdc×n×4B`（t1..t4 四份），总量 = 4×arBytes + 4×tempBytes，512B 对齐后经默认 workspace 申请；跳过临时矩阵的场景为 0。典型值：n=k=2048 约 128MiB；n=4096 低秩场景上界约 260MiB，HBM 可容纳。

#### kernel 侧设计

| # | kernel | 类型 | 发射次数 | 职责 |
| --- | --- | --- | --- | --- |
| K1 | cher2k_deinterleave | AIV SIMD | 1 | 64×64 tile 循环，2D strided 搬运 + DeInterleave，将 A、B 拆为 Ar/Ai/Br/Bi 紧凑实数矩阵；核间二分（前半核处理 A、后半核处理 B），A/B 同形同工作量 |
| K2 | 实 GEMM kernel（复用仓内 arch35 gemm 模块，无改动） | AIC Cube | 4 | 以 m=n=n、k=k、ldc=tempLdc、alpha=1、无 beta 的参数四发，落盘 t1..t4；OP_C 在参数构造中映射为等价转置形态 |
| K3 | cher2k_combine | AIV SIMD | 1 | 按 uplo 三角 64×64 tile 分核；每 tile 双加载（(i,j) 块与 (j,i) 转置块），向量原语合成 M 与 M^H、完成复数 alpha 乘与实数 beta 累加，对角虚部代数归零；beta=0 不加载旧 C；完全 uplo 块整块写出，对角块逐列按三角掩码写出，非 uplo 段与 padding 回写旧值 |
| K4 | cher2k_simt_small | AIV SIMT | ≤1 | n ≤ 16 融合直算：逐元素 k 长度双实点积 + 同款合成公式，无临时空间、无 Phase0/1 |

**UB 预算**（950PR UB 248KiB/核）：

| kernel | 主要缓冲 | 预算 |
| --- | --- | --- |
| K1 | 输入复数 tile 32KiB + 实/虚输出各 16KiB（A、B 串行复用） | 64KiB |
| K3 | t1..t4 双 tile 8×64×64×4B = 128KiB + C 输入 32KiB + C 输出 32KiB | 192KiB |
| K4 | 行缓冲 + 累加器 | < 64KiB |

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Ascend 950PR（arch35） | √ |
| Atlas 800I/T A2 |  |
| Atlas 900 A3 PoD |  |

## 算子约束限制

- `trans` 仅支持 OP_N / OP_C；OP_T 为合法枚举但不支持，返回 INVALID_VALUE；uplo/trans 越界值返回 INVALID_ENUM。
- 仅引用/更新 uplo 指定三角；对角元素虚部强制置 0；非 uplo 三角与 ldc padding 字节不被读写（保持原样）。
- C 原地覆写，不返回视图；beta = 0 时 C 旧值不必有效（不读取，避免 NaN×0 传播）。
- n = 0，或 (alpha=(0,0) 或 k=0) 且 beta=1 时为合法 quick return，不执行计算。
- 异步执行：依赖 `aclblasSetStream` 绑定 stream，读回 Device 结果前须同步 stream；Host 不直接解引用 alpha/beta。
- 不支持超出 lda/ldb/ldc 语义的非连续内存访问；无 broadcast；n/k 为运行时入参，无 dynamic shape 要求。
- 大 k 场景（至 4096）4M 分解改变归约顺序，若纯 FP32 实测超限，则在乘加阶段启用 HF32 三项残差补偿，其余阶段不变。

# 可维可测分析（required）

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | golden 由 cblas（Netlib cher2k）生成，仅验证 C 的 uplo 三角（含对角），实部/虚部分别按 FLOAT32 分量判定；对角虚部按强制 0 口径参与比对（golden 同口径） | 生态算子开源精度标准（https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md） |
| 性能标准 | Ascend 950PR 实测，COMPLEX64 场景平均单次耗时不高于 4 case 标杆（下表），warmup 后有效采样 >50 次取平均 | 社区任务性能要求 |

| 数据类型 | rtol | atol | required_matched_ratio | max_abs_error_limit |
| --- | --- | --- | --- | --- |
| COMPLEX64（实部/虚部按 FLOAT32 分量） | 2^-10 (9.77e-4) | 2^-16 (1.53e-5) | 0.99 | 1e-2 或 32 ULP |

逐元素通过条件 `|actual − golden| ≤ atol + rtol×|golden|`；测试工程并行启用 MERE/MARE 统计（mere_threshold = 2^-13、mare_multiplier = 10.0，实/虚分别统计）。

| case | uplo | trans | n | k | 标杆耗时（Avg time, us） |
| --- | --- | --- | --- | --- | --- |
| 1 | UPPER | N | 1024 | 1024 | 641.43 |
| 2 | UPPER | N | 2048 | 2048 | 4157.94 |
| 3 | LOWER | C | 1024 | 1024 | 680.47 |
| 4 | LOWER | C | 2048 | 2048 | 4258.12 |

4 条标杆由随任务 GPU 基线（H100）前 4 条 gpu_ms ÷ 0.4 换算而来（如 0.256574 ms ÷ 0.4 = 641.43 us），判定口径为 NPU 平均单次耗时 ≤ gpu_ms/0.4；其余性能用例基线未回填前仅采集不判定。

## 测试概览

测试代码落位 `test/herk/cher2k/`（golden/param 头文件）与 `test/herk/cher2k/arch35/`（wrapper、GTest、CSV、CMakeLists），由仓内脚本驱动 CSV 加载 GTest 执行。随任务提供 1200 条用例（1000 精度 + 200 性能，固定种子可复现）：

| 类别 | 前缀 | 条数 | 覆盖内容 |
| --- | --- | --- | --- |
| L0 冒烟 | TC_L0 | 8 | uplo×trans 四组合 × 小尺寸主路径连通 |
| L1 尺寸 | TC_SQ | 92 | 23 档尺寸（1→2048，含奇数/边界/非对齐）× 4 组合 |
| L2 标量 | TC_AB | 32 | alpha（0/(1,0)/纯虚/复/大值）× beta（0/1/负/分数）× 4 组合 |
| L3 k 非对称 | TC_TK/BK | 24 | k≪n（N 路径）与 k≫n（C 路径） |
| L4 前导维 | TC_LD | 6 | 紧凑与 padding 场景 |
| L5 填充 | TC_FL / TC_CV | 38 | 均匀/全零/交替/极端/Inf/NaN；中尺寸全覆盖 |
| L6 边界负向 | TC_ED | 23 | 零维、k=0、alpha=0 quick return、空指针（含 C 指针新口径）、非法枚举、OP_T、非法/负维度 |
| EX 扩展 | TC_EX | 777 | 尺寸池 × 组合 × 标量 × padding 确定性采样（大 k 精度主力） |
| 性能 | TC_PF | 200 | 4 条典型 case + 小尺寸 + 方阵扫描 + k 网格 + 低秩大 n + 预算内混合 |

验证策略：仅验证 uplo 三角（含对角），实部/虚部分别比对；非 uplo 三角与 ldc padding 以 canary 填充后 bit-exact 断言不被污染；对角虚部另加 `max|CI_ii| ≤ 2^-16` 硬断言；负向用例只断言返回码。golden 侧实现与 NPU 接口一致的参数镜像校验，保证两侧异常返回码逐条相等。

任务要求输入含正态分布 50%（μ∈[-5,5]、σ∈[0.1,2]，实/虚独立），随任务用例以均匀分布为主，故自补 **60 条正态 + 60 条均匀** 补充用例（固定种子生成，覆盖 uplo×trans、n∈{7,65,400,1024,2048}、k∈{64,1024,2048,4096}、beta∈{0,1,0.5} 及 pad 场景），条数与分布写入自测报告。

性能双口径采集：GTest 端到端（含 host 准备、golden、比对，保守上界）与 msprof kernel 纯耗时，两口径分开报告；TC_PF 分支 warmup 10 次 + 有效采样 60 次取平均，aclrtEvent 计时；n=1..16 小尺寸用例同时验证 SIMT 回退收益。精度回归脚本自动排除 TC_PF 前缀。

## 兼容性分析

新算子，`include/cann_ops_blas.h` 中当前无 `aclblasCher2k` 声明，新增声明紧邻同族 `aclblasCherk`，不改动既有接口与既有算子行为，无兼容性影响；声明供其他产品线共用，不引入 950PR 私有平行 API。
