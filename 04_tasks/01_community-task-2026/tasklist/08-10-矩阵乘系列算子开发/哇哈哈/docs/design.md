# 矩阵乘系列算子开发设计文档

> 任务编号：08-10 矩阵乘系列算子开发
> 团队：哇哈哈
> 开发语言：Ascend C（CANN 9.0.0 及以上）
> 适配硬件：Atlas A2 训练系列产品（dav-c220 / arch22；部分算子覆盖 dav-3510 / arch35）
> 对应任务书：[matmul_series_task_doc.md](https://gitcode.com/cann/cann-competitions/blob/master/04_tasks/01_community-task-2026/tasklist/08-10-矩阵乘系列算子开发)
> 设计文档模板：[design_template.md](https://gitcode.com/cann/cann-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md)
> 社区任务流程：[discussions/39](https://gitcode.com/org/cann/discussions/39)
> PR 链接：[#1143 feat(docs): 添加矩阵乘系列算子设计文档（5个算子）](https://gitcode.com/cann/cann-ops-competitions/pull/1143)

# 需求背景

## 需求来源

- 社区任务总入口：https://www.hiascend.com/activities/task-center/details/4dde422dc8c64747a94f44da0a9ce773
- 任务书（算子级）：`04_tasks/01_community-task-2026/tasklist/08-10-矩阵乘系列算子开发`
- 社区任务流程与注意事项：https://gitcode.com/org/cann/discussions/39
- 精度标准（生态算子开源精度标准）：https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md
- 参考接口：标准 BLAS（https://netlib.org/blas/blasqr.pdf）与 cuBLAS（https://docs.nvidia.com/cuda/cublas/index.html）
- 验收脚本（精度/性能）：任务书 `test_script/chemm/`（`verify_accuracy.py` / `verify_performance.py` / `gpu_baseline.csv`）

## 背景介绍

本任务要求在昇腾 NPU（Atlas A2 训练系列，CANN 9.0.0+）上用 Ascend C 实现 5 个
complex64 BLAS Level-3 算子：`chemm`、`cher2k`、`cherk`、`csymm`、`csyrk`。
算子保持与标准 BLAS / cuBLAS 一致的列主序（Column-Major）语义。验收时对性能的要求为：
**NPU 单算子耗时不得高于 GPU A100 耗时的 1.25 倍**（即不低于 0.8 倍 A100 性能，
见任务书 `test_script/matmul_series_gpu_perf.csv` 基准）。

5 个算子的核心公共点：复数矩阵乘可经 **4M 分解** 转化为 4 个实数 GEMM。因此本设计
统一采用「**AIV 解交织（complex64 → 实部/虚部实数矩阵）→ AIC 实数 GEMM → AIV 组合写回**」
的三阶段流水线，复用 ops-blas 中 `blas/gemm` 与 `blas/herk` 已验证的 GEMM 调度与
多核切分模式。各算子之间的差异集中在：三角补齐规则、GEMM 操作数映射、组合公式的
加减号、标量是否为复数、以及 C 是否仅写 uplo 三角。

# 需求分析

## 需求描述

实现 5 个 complex64 BLAS Level-3 算子，列主序语义对齐标准 BLAS：

| 算子 | 数学公式 | 关键参数 | C 形状 |
| --- | --- | --- | --- |
| chemm | C = α·A·B + β·C（side=L/R，A Hermitian） | side, uplo, α/β 复数 | 稠密 m×n |
| cher2k | C = α·A·Bᴴ + conj(α)·B·Aᴴ + β·C（trans=N/C） | uplo, trans, α 复数, β 实数 | Hermitian 三角 |
| cherk | C = α·A·Aᴴ + β·C（trans=N/C） | uplo, trans, α/β 实数 | Hermitian 三角 |
| csymm | C = α·A·B + β·C（side=L/R，A 对称） | side, uplo, α/β 复数 | 稠密 m×n |
| csyrk | C = α·A·Aᵀ + β·C（trans=N/T） | uplo, trans, α/β 复数 | 对称三角 |

共同约束：输入/输出均为 complex64，Column-Major，不支持 broadcast；A 参与三角运算时
仅引用 `uplo` 指定三角。Hermitian 类算子（cherk/cher2k）对角线虚部强制清零；对称类
（csymm/csyrk）不做共轭，对角线虚部保留。

## 需求拆解

1. 5 个算子均实现功能等价、参数校验、快速路径（alpha=0 / k=0 / m,n=0）与泛化形状支持。
2. 统一复用实数 GEMM：每个算子经 4M 分解执行 4 个实数 GEMM（cher2k 利用 Hermitian
   对称性把第二乘积项合入组合阶段，无需第 2 组 4 GEMM）。
3. 三角类算子（cherk/cher2k/csyrk）只写 `uplo` 三角；Hermitian 类对角线虚部强制清零。
4. 精度满足生态算子开源精度标准；性能以任务书 GPU A100 数据为基准（NPU ≤ GPU/0.8）。

# 详细设计

## 算子分析

### 数学公式

设复数矩阵 A = Ar + i·Ai，B = Br + i·Bi（Ar/Ai/Br/Bi 为实数矩阵）。

**chemm（side=L：C = α·A·B + β·C；side=R：C = α·B·A + β·C）**

```
side='L': t1=Ar·Br, t2=Ai·Bi, t3=Ar·Bi, t4=Ai·Br
side='R': t1=Br·Ar, t2=Bi·Ai, t3=Br·Ai, t4=Bi·Ar
Cr = t1 - t2,  Ci = t3 + t4
C  = α·(Cr + i·Ci) + β·C
```

**cherk（trans=N：C = α·A·Aᴴ + β·C；trans=C：C = α·Aᴴ·A + β·C，α/β 实数）**

```
trans='N': t1=Ar·Arᵀ, t2=Ai·Aiᵀ, t3=Ai·Arᵀ, t4=Ar·Aiᵀ
trans='C': t1=Arᵀ·Ar, t2=Aiᵀ·Ai, t3=Arᵀ·Ai, t4=Aiᵀ·Ar
Cr = t1 + t2,  Ci = t3 - t4
C  = α·(Cr + i·Ci) + β·C
```

**cher2k（trans=N：C = α·A·Bᴴ + conj(α)·B·Aᴴ + β·C；α 复数, β 实数）**

只计算 S = α·A·Bᴴ 方向的 4 个实数 GEMM（同 chemm 的 side=L 映射，B 取共轭转置）：

```
trans='N': t1=Ar·Brᵀ, t2=Ai·Biᵀ, t3=Ai·Brᵀ, t4=Ar·Biᵀ
S_re = t1 + t2,  S_im = t3 - t4
P = α 缩放后的 S
```

第二项满足 `conj(α)·B·Aᴴ = conj(Pᵀ)`，输出元素取 `P[i][j] + conj(P[j][i]) + β·C_old[i][j]`，
因此无需执行第 2 组 4 GEMM。C 仅写 uplo 三角，对角线虚部清零。

**csymm（side=L/R：C = α·A·B + β·C；A 对称，α/β 复数）**

4M 分解与 chemm 相同（t1−t2 为实部、t3+t4 为虚部）。A 为对称矩阵，三角补齐时实部、
虚部均直接镜像（不取共轭），且不对对角线做虚部清零。C 为稠密 m×n，全量写回。

**csyrk（trans=N：C = α·A·Aᵀ + β·C；trans=T：C = α·Aᵀ·A + β·C；α/β 复数）**

对称转置不共轭，虚部符号与 cherk 相反：

```
trans='N': t1=Ar·Arᵀ, t2=Ai·Aiᵀ, t3=Ar·Aiᵀ, t4=Ai·Arᵀ
trans='T': t1=Arᵀ·Ar, t2=Aiᵀ·Ai, t3=Arᵀ·Ai, t4=Aiᵀ·Ar
Cr = t1 - t2,  Ci = t3 + t4
C = α·(Cr + i·Ci) + β·C
```

C 为对称三角矩阵，只写 uplo 三角，无 Hermitian 对角线约束。

### 支持数据类型

- 5 个算子均仅支持 **complex64**（单精度复数，实部/虚部各 32-bit）。
- alpha/beta：chemm/csymm/csyrk 为 complex64 标量；cherk/cher2k 的 beta 为实数
  （complex64 但虚部为 0），cherk 的 alpha 为实数、cher2k 的 alpha 为复数。

### 支持形状

- 存储：全部 Column-Major；不支持 broadcast；不支持 batch。
- 维度：chemm/csymm 的 m、n 必须 > 0；cherk/cher2k/csyrk 的 n、k 必须 ≥ 0。
- 前导维度满足 `lda ≥ max(1, dim)`、`ldb ≥ max(1, m)`、`ldc ≥ max(1, m)`，支持 padding 布局。
- A 参与三角运算时仅引用 `uplo` 指定三角（上三角 U / 下三角 L）。

## 算子实现

### 实现方案

统一三阶段流水线（AIV 解交织 → AIC 实数 GEMM → AIV 组合写回），以 chemm 为典型，
其余算子复用同一骨架，差异仅在三角补齐与组合公式。

#### 3.2.1 host侧设计：

**参数校验**：枚举值（side/uplo/trans）、m/n/k 符号与下界、空指针、前导维度下界、
u32 溢出（lda/ldb/ldc/m/n 超过 UINT32_MAX 判非法）。校验失败返回对应
`ACLBLAS_STATUS_*` 错误码，不启动 kernel。

**标量读取**：实数标量直接读 1 个 float；复数标量读实部/虚部 2 个 float。通过
`aclrtMemcpy` + `aclrtSynchronizeStream`（或 handle 内的 device→host 读取工具）取回 host。

**快速路径**：
- m == 0 或 n == 0（乘类）/ n == 0（秩更新类）：直接返回成功，C 不变。
- alpha == 0 或 k == 0 且 beta == 1：跳过 GEMM，C 不变（三角类算子仍需在 combine 阶段
  恢复非 uplo 区域或零化 uplo）。
- alpha == 0 且 beta == 0：跳过 GEMM 并零化（三角场景恢复非 uplo 区域）。

**workspace 布局**（device GM，512B 对齐）：依次存放解交织后的实数输入矩阵
（dAr/dAi 及 dBr/dBi）+ 4 个 temp 矩阵（dT1..dT4）+ 每核 GEMM A 操作数紧凑化临时区。
以 chemm SIDE=L、m=n=1024 为例：

| 缓冲区 | 维度（float） | 字节 | 说明 |
| --- | --- | --- | --- |
| dAr / dAi | 1024 × 1024 | 4 MB ×2 | Hermitian A 解交织后实部/虚部（含共轭补齐） |
| dBr / dBi | 1024 × 1024 | 4 MB ×2 | B 解交织后实部/虚部 |
| dT1..dT4 | 1024 × 1024（×kChunks） | 16 MB ×4 | 4 个实数 GEMM 的部分和（K 分块累加） |
| pack scratch | 每核 compact A | ~数 MB | GEMM A 操作数紧凑化，按 AIC 核数分配 |

总 workspace 约 80 MB 级，远小于 host 侧 4GB 内存预算，单用例不溢出。

**tiling 策略**：
- AIV 核数由 `GetAivCoreCount()` 获取，按行均分（满核优先，行尾余量分给前几个核）。
- AIC 核数由 `GetAicCoreCount()` 获取；GEMM 阶段采用 M 维带状切分（每核负责连续若干
  行，`numBlocks = min(AIC核数, M)`），K 维按 256 分块；各 K 块的部分和由 AIV 累加核做
  K 维归约（dav-c220 上对同一 GM 区域重复 atomic IterateAll 不可靠，故显式 K 分块累加）。
- 条带宽度权衡：M 维条带过宽时打包开销随 M 增长，Cube tile 效率提升不足以补偿；设计上
  以满核均分为主（如 20 核、每核约 52 行 @m1024），权衡打包开销与 Cube 效率的平衡。

**数据通路图**：

```
                 A,B (complex64, GM, Column-Major)
                          │  H2D
                          ▼
   Phase 0 (AIV)  DeInterleave: complex64 → [Ar,Ai] / [Br,Bi]
                  (Hermitian/对称三角在此补齐：实部镜像、虚部取反/镜像、对角虚部清零)
                          │
   Phase 1 (AIC)  4 × gemm_kernel_do (MatmulImpl)
                  t1..t4 = 4 个实数 GEMM（K 分块，部分和写入 dT1..dT4）
                          │
                  (K 分块累加: AIV 将 dT1..dT4 的 K 维部分和求和)
                          │
   Phase 2 (AIV)  Combine: α/β 缩放 + 共轭 + Interleave 回 complex64
                  (三角算子只写 uplo；Hermitian 对角虚部清零)
                          │  D2H
                          ▼
                      C (complex64, GM)
```

#### 3.2.2 kernel侧设计：

**Phase 0（AIV 解交织）**：complex64 输入按 [re, im] 交织读入 UB，`DeInterleave`
拆出实部/虚部，按行块处理并写回 GM。三角 A 在此阶段补齐镜像三角：Hermitian 实部镜像、
虚部取负（共轭），对角线虚部写 0；对称矩阵实部/虚部均直接镜像。行块按 AIV 核数均分，
核间写不同行区间，无竞争。

**Phase 1（AIC 实数 GEMM）**：4 次 `gemm_kernel_do`（MatmulImpl），输出 4 个 temp 矩阵
dT1..dT4，行步长 `tempLdc = CeilAlign(输出行数, 16)`。`gemm_kernel_do` 存储结果为
`(A·B)ᵀ`（fixpipe 转置），cherk/csyrk 等通过交换 t3/t4 操作数顺序补偿该转置。
K 维按 256 分块，每块写独立 GM 区域，避免原子累加冲突；多 K 块时由 AIV 累加核求和。

**Phase 2（AIV 组合写回）**：按行块加载 dT1..dT4 与 C_old，按算子组合公式做缩放与共轭，
`Interleave` 回 complex64，按列主序写回 C。三角算子只写 `uplo` 区域，非 uplo 区域保持
输入 C 不变；Hermitian 算子对角线虚部强制清零。行块按 AIV 核数均分，行互斥无竞争。

**同步协议**：host 侧每阶段 `aclrtSynchronizeStream` 保证阶段间数据就绪（解交织完成→
GEMM→累加→组合）。kernel 内部遵循 Ascend C 流水线同步（PIPE_MTE2/PIPE_MTE3/PIPE_M/
PIPE_FIX 事件），避免读写冒险。

**异常处理与内存安全**：
- host 侧 malloc/H2D/D2H/同步均检查返回码，失败按逆序释放已分配资源并返回错误码。
- GM 搬运只覆盖逻辑有效区域；UB/L1/L0 缓冲按对齐尺寸分配，附编译期/运行期边界检查。
- workspace 一次分配、512B 对齐；temp 矩阵部分和不依赖清零（各 K 块写独立区域，累加核
  再求和）。

### 各算子设计

#### 1. chemm

**公式**：side='L' 时 C = α·A·B + β·C；side='R' 时 C = α·B·A + β·C。
**4M 分解**：见上「数学公式」chemm 段（t1−t2 实部、t3+t4 虚部）。
**三角补齐**：A 为 Hermitian，补齐时实部镜像、虚部取反，对角线虚部写 0。
**写回**：C 为稠密 m×n，全量写回，无三角掩码。
**A100 基准（性能验收目标）**：

| M | N | side | uplo | A100(ms) | NPU 目标(≤ms) |
| --- | --- | --- | --- | --- | --- |
| 1024 | 1024 | L | U | 0.609 | 0.761 |
| 2048 | 2048 | L | U | 4.951 | 6.189 |
| 1024 | 1024 | R | L | 0.490 | 0.613 |
| 2048 | 2048 | R | L | 4.337 | 5.421 |

#### 2. cherk

**公式**：trans='N' 时 C = α·A·Aᴴ + β·C；trans='C' 时 C = α·Aᴴ·A + β·C。
**4M 分解**：见上「数学公式」cherk 段（Cr = t1+t2, Ci = t3−t4）。
**写回**：C 为 Hermitian 三角矩阵，只写 uplo 三角，对角线虚部强制清零。
**A100 基准**：

| N | K | uplo | trans | A100(ms) | NPU 目标(≤ms) |
| --- | --- | --- | --- | --- | --- |
| 1024 | 1024 | U | N | 0.314 | 0.392 |
| 2048 | 2048 | U | N | 1.929 | 2.411 |
| 1024 | 1024 | L | C | 0.250 | 0.312 |
| 2048 | 2048 | L | C | 2.024 | 2.530 |

#### 3. cher2k

**公式**：trans='N' 时 C = α·A·Bᴴ + conj(α)·B·Aᴴ + β·C。
**4M 分解**：仅计算 S = α·A·Bᴴ 的 4 个实数 GEMM（见上），第二项由 `conj(Pᵀ)` 在组合阶段
加回，省去第 2 组 GEMM。
**写回**：C 为 Hermitian 三角，只写 uplo，对角线虚部清零。
**A100 基准**：

| N | K | uplo | trans | A100(ms) | NPU 目标(≤ms) |
| --- | --- | --- | --- | --- | --- |
| 1024 | 1024 | U | N | 0.654 | 0.818 |
| 2048 | 2048 | U | N | 4.055 | 5.069 |
| 1024 | 1024 | L | C | 0.548 | 0.685 |
| 2048 | 2048 | L | C | 4.156 | 5.195 |

#### 4. csymm

**公式**：side='L' 时 C = α·A·B + β·C；side='R' 时 C = α·B·A + β·C。
**4M 分解**：与 chemm 相同。A 对称，三角补齐实部/虚部均镜像，不做共轭、不清零对角虚部。
**写回**：C 为稠密 m×n，全量写回。
**A100 基准**：

| M | N | side | uplo | A100(ms) | NPU 目标(≤ms) |
| --- | --- | --- | --- | --- | --- |
| 1024 | 1024 | L | U | 0.615 | 0.769 |
| 2048 | 2048 | L | U | 4.902 | 6.128 |
| 1024 | 1024 | R | L | 0.494 | 0.618 |
| 2048 | 2048 | R | L | 4.363 | 5.454 |

#### 5. csyrk

**公式**：trans='N' 时 C = α·A·Aᵀ + β·C；trans='T' 时 C = α·Aᵀ·A + β·C。
**4M 分解**：见上「数学公式」csyrk 段（对称转置不共轭，虚部符号与 cherk 相反）。
alpha/beta 为复数，组合阶段做交叉缩放。
**写回**：C 为对称三角，只写 uplo 三角，无 Hermitian 对角线约束。
**A100 基准**：

| N | K | uplo | trans | A100(ms) | NPU 目标(≤ms) |
| --- | --- | --- | --- | --- | --- |
| 1024 | 1024 | U | N | 0.307 | 0.384 |
| 2048 | 2048 | U | N | 1.945 | 2.431 |
| 1024 | 1024 | L | T | 0.249 | 0.311 |
| 2048 | 2048 | L | T | 2.009 | 2.511 |

### 性能达标路径

任务书要求 NPU ≤ GPU A100 / 0.8（即不慢于 A100 的 1.25 倍）。本设计以方案正确性与精度
为首要验收项，性能以任务书 GPU A100 基准为目标。性能差距可能来自三方面，对应设计上的
优化路径如下：

1. **复用 Hermitian/对称对称性**：Ar 对称、Ai 反对称，标准 HEMM/SYMM 实现只需约 2 个
   GEMM 的量（而非 4 个全量），计算量可砍半。
2. **提升 Cube 利用率**：GEMM 部分直接经 MatmulImpl 调度时 Cube 利用率不足，需定制
   MatmulImpl 切分或手写 Mmad 流水线，将 AIC 多核利用率拉满。
3. **AIV 全向量化**：解交织/组合采用 Gather/DataCopy 向量化替代标量循环，
   压缩 Phase 0/Phase 2 开销。
4. **减少中间量**：合并解交织与 GEMM 操作数紧凑化，降低 workspace 搬运。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas A2 训练系列产品 | √ |

注：chemm 实现面向 arch22（dav-c220，20 AIC / 20 AIV）；cherk 面向 arch35（dav-3510）。
其余算子复用同一套 arch22/arch35 实现骨架。

## 算子约束限制

- 5 个算子仅支持 complex64 数据；alpha/beta 按任务书区分实数或复数。
- 全部采用 Column-Major 存储，不支持 broadcast，不支持 batch。
- 乘类算子（chemm/csymm）m、n 必须 > 0；秩更新算子（cherk/cher2k/csyrk）n、k 必须 ≥ 0。
- 前导维度满足 `lda ≥ max(1,dim)`、`ldb ≥ max(1,m)`、`ldc ≥ max(1,m)`，支持 padding 布局。
- A 参与三角运算时仅引用 `uplo` 指定三角；Hermitian 类算子对角线虚部强制清零。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | 逐元素 `|NPU − golden| ≤ atol + rtol·max(|golden|)`，且 `max(|NPU−golden|/|golden|) ≤ 10·rtol`；atol=2⁻¹⁶≈1.5259e-5，rtol=2⁻¹⁰≈9.7656e-4 | 生态算子开源精度标准（experimental_standard） |
| 性能标准 | NPU 耗时 ≤ GPU A100 耗时 / 0.8（即不慢于 A100 的 1.25 倍） | 任务书 `test_script/matmul_series_gpu_perf.csv` 基准 |
| 三角一致性 | 三角算子非 uplo 区域与输入 C 精确一致；Hermitian 对角线虚部按容差验证 | 标准 BLAS 语义 + 精度标准 |

**自测用例**：任务书 `test_script/chemm/` 提供 318 条 CSV 驱动用例（精度 218 + 性能 100），
覆盖全参数组合 × 多尺寸扫描（含奇数 3/5/7/65、边界 1、大尺寸 4096/8000/16000），由
`verify_accuracy.py`（编译+执行 C++ GTest，逐条比对）与 `verify_performance.py`
（执行 TC_PF_ 用例，关联 `gpu_baseline.csv` 判定 NPU ≤ GPU/0.8）自动验收。

## 兼容性分析

5 个算子均为新增算子，不涉及既有接口兼容性迁移。功能语义与 cuBLAS / 标准 BLAS 对齐，
错误码与参数校验返回路径遵循 ops-blas 现有约定。代码按任务书要求在 `cann-competitions`
仓库 `tasklist/08-10-矩阵乘系列算子开发` 目录以 PR 形式提交，通过评审后合入。
