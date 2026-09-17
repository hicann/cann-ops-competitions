# 需求背景（required）

## 需求来源

本算子来源于 ops-blas 开源仓社区任务（仓库：https://gitcode.com/cann/ops-blas ），要求在 Atlas A2/A3 系列产品上，使用 Ascend C 编程语言开发句柄式（handle-based BLAS）批量单精度复数矩阵乘算子 `aclblasCgemmBatched`，接口语义、参数顺序与对标接口 cuBLAS `cublasCgemmBatched`（NVIDIA cuBLAS）及 Netlib `cgemm`（Netlib BLAS Fortran）保持一致，精度满足生态算子开源精度标准、性能满足给定标杆。

## 背景介绍

### aclblasCgemmBatched 算子实现现状分析

ops-blas 仓中批量矩阵乘目录 `blas/gemm_batched/` 已存在 arch35 版本实现，但缺少面向 Atlas A2/A3 的 arch22（dav-2201 指令集）版本，需要基于 Ascend C 新增 arch22 实现。当前对标接口（cuBLAS cublasCgemmBatched）支持的能力如下：

| 参数 | 参数含义 | 数据类型 | 支持数据类型 | 约束 | 形状 |
| --- | --- | --- | --- | --- | --- |
| transa | A 的操作类型 | 枚举 | N / T / C | 枚举值合法 | 标量（Host 内存） |
| transb | B 的操作类型 | 枚举 | N / T / C | 枚举值合法 | 标量（Host 内存） |
| m | op(A) 与 C 的行数 | int | int | m ≥ 0 | 标量 |
| n | op(B) 与 C 的列数 | int | int | n ≥ 0 | 标量 |
| k | op(A) 的列数 / op(B) 的行数 | int | int | k ≥ 0 | 标量 |
| alpha | 复数乘性系数 | aclblasComplex* | Complex FP32 | 非空指针 | 标量（Host 内存） |
| Aarray | A 矩阵指针数组 | const aclblasComplex* const[] | Complex FP32 | batchCount>0 时非空；逐批首址 | 见维度约束 |
| lda | A 的主维（复数元素） | int | int | T/C 时 lda≥k，N 时 lda≥m | 标量 |
| Barray | B 矩阵指针数组 | const aclblasComplex* const[] | Complex FP32 | batchCount>0 时非空；逐批首址 | 见维度约束 |
| ldb | B 的主维（复数元素） | int | int | T/C 时 ldb≥n，N 时 ldb≥k | 标量 |
| beta | 复数加性系数 | aclblasComplex* | Complex FP32 | 非空指针 | 标量（Host 内存） |
| Carray | C 矩阵指针数组（原地更新） | aclblasComplex* const[] | Complex FP32 | batchCount>0 时非空；逐批首址 | ldc×n 复数矩阵 |
| ldc | C 的主维（复数元素） | int | int | ldc ≥ max(1, m) | 标量 |
| batchCount | 批次数量 | int | int | batchCount ≥ 0 | 标量 |

其中维度约束（列主序，物理行列）：

- transa = N：op(A) 为 m×k，A 物理大小 lda×k；transa = T/C：op(A) 为 m×k，A 物理大小 lda×m。
- transb = N：op(B) 为 k×n，B 物理大小 ldb×n；transb = T/C：op(B) 为 k×n，B 物理大小 ldb×k。
- C 物理大小 ldc×n。
- 复数元素 aclblasComplex 为实部、虚部交错（interleaved）存储，每元素 8 字节（real float + imag float）。

### aclblasCgemmBatched 算子功能分析

算子功能：对 batchCount 组同形状复数矩阵逐批执行

```
C[i] = alpha * op(A[i]) * op(B[i]) + beta * C[i],  i = 0 .. batchCount-1
```

- 输入：Aarray、Barray、alpha、beta、m、n、k、lda、ldb、ldc、transa、transb、batchCount。
- 输出：Carray（原地覆盖）。
- 数据类型：单精度复数 Complex FP32（FP32 实部 + FP32 虚部）。
- 转置：op(X) 支持不转置 N、转置 T、共轭转置 C（C 在 T 基础上对虚部取反）。
- 批量：uniform batch，所有批次共享 m/n/k/lda/ldb/ldc/transa/transb，逐批矩阵首址由设备侧指针数组独立指定，允许各批次位于不同设备地址，不要求批次间内存连续。
- 边界语义：
  - m = 0、n = 0、batchCount = 0：合法空操作，直接返回成功。
  - k = 0 或 alpha = (0,0)：跳过矩阵乘，执行 C = beta * C；其中 beta = (1,0) 时 C 保持不变。
  - beta = (0,0) 时输出无需依赖 C 的初值。

# 需求分析（required）

## 需求描述

使用 Ascend C 编程语言在 arch22（Atlas A2/A3）上实现 `aclblasCgemmBatched`：

1. 支持 Complex FP32（FP32 实部/虚部）输入输出，矩阵列主序、实虚部交错存储。
2. 支持 transa/transb 取 N、T、C 三种操作，支持 alpha、beta 任意复数值与 C 原地更新。
3. 支持 uniform batch（1～1024），逐批矩阵由设备侧指针数组寻址。
4. 覆盖 m/n/k 为任意非负值（含非 64 对齐、非 8 对齐的尾块），精度满足生态算子开源精度标准，性能不低于给定标杆。
5. 接口、参数顺序、边界返回码与 cuBLAS/Netlib 语义对齐，可直接被 ops-blas 上层和测试框架复用。

## 需求拆解

1. 参数合法性校验（handle、alpha/beta 指针、维度非负、枚举合法、主维约束、批量指针数组非空）与对应返回码。
2. 复数矩阵乘的实数分解：采用 Karatsuba 3M，将一次复数 GEMM 降为 3 次实数 GEMM。
3. 交错复数输入到实数 SoA（Split of Array）工作矩阵的高效搬运，支持连续列（N）与转置/共轭（T/C）。
4. 基于 Cube 单元的实数 GEMM 计算（FP32 MMAD），含分块、尾块 padding、K 维搬运与计算流水重叠。
5. 实数结果重组、alpha/beta 复数缩放叠加，并重新交错写回列主序 C。
6. 强消去场景的精度补偿（双精度重算）与边界场景（k=0、alpha=0、beta=0/1）正确处理。
7. 多核负载划分、workspace 布局与跨核缓存一致性。
8. 满足精度（rtol/atol/matchedRatio/maxAbsErr）与性能标杆（4 个规格，warmup 后 50 次以上采样平均）。

# 详细设计（required）

## 算子分析

### 数学公式

逐批计算 `C = alpha * op(A) * op(B) + beta * C`。设

```
A = Ar + j*Ai,  B = Br + j*Bi
```

直接展开（4M）需要 4 次实数矩阵乘：

```
real = Ar*Br - Ai*Bi
imag = Ar*Bi + Ai*Br
```

实现采用 Karatsuba 3M 分解，仅需 3 次实数矩阵乘：

```
P1 = (Ar + Ai) * (Br + Bi)
P2 = Ar * Br
P3 = Ai * Bi
real = P2 - P3
imag = P1 - P2 - P3
```

最终：

```
out_real = alpha_real*real - alpha_imag*imag + beta_real*C_real - beta_imag*C_imag
out_imag = alpha_real*imag + alpha_imag*real + beta_real*C_imag + beta_imag*C_real
```

列主序输入通过"零数据搬运的转置等价"映射为行主序 Cube GEMM：令 `X = op(B)^T`、`Y = op(A)^T`、`D = C^T`，则 `D = X * Y` 为行主序实数 GEMM，N 情形下列主序存储可直接重解释，不需要物理转置。

### 支持数据类型

- 输入/输出矩阵：aclblasComplex（Complex FP32，实部/虚部均为 float32，交错存储）。
- 标量 alpha、beta：aclblasComplex（Host 内存）。
- Cube 内部实数 GEMM：FP32（不使用 HF32/FP16，保证全精度尾数）。
- 精度补偿路径：使用两数和/乘积的 double-float 风格标量补偿运算。

### 支持形状

- m、n、k 任意非负整数，batchCount 1～1024（及 0 的空操作）。
- N/T/C 全组合；主维可大于逻辑维（允许矩阵内 padding 跨步）。
- 内部统一将维度向上对齐：mP = align(m,64)、nP = align(n,64)、kP = align(k,64)，按 64×64×64 的 Cube 分块计算，逻辑尾块（padding 区）写零，输出只取 m×n 逻辑区。

## 算子实现

### 实现方案

整体为一次 API 调用内、同一条 stream 上顺序提交的 3 类设备 kernel，外加一个早退 kernel：

1. **pack（AIV，packB+packA 合并为单 kernel）**：把交错复数 A/B 解交错为 workspace 中按 64/32 对齐的实/虚部行主序矩阵 Xr/Xi/Yr/Yi，并融合 Karatsuba 预算 AddX=Xr+Xi、AddY=Yr+Yi。
2. **gemm（AIC）**：对每个批次、每个 64×64 输出 tile 执行 P2、P3、P1 三次 FP32 实数 MMAD，结果 T1、T2、P1 写回 workspace。
3. **combine（AIV）**：读取 T1/T2/P1，按 Karatsuba 公式重组实/虚部，做 alpha 复数缩放与 beta*C 叠加，重新交错写回列主序 C；强消去元素走双精度补偿重算。
4. **scale（AIV，早退）**：k=0 或 alpha=0 时只执行 C = beta*C。

workspace 仅在一次调用内使用，由句柄统一管理，容量随 m/n/k/batchCount 自动增长（上限 8 GiB）。Gather 交错用的字节偏移表是编译期常量，存放于进程级常驻设备缓冲区，首次调用上传一次，之后在 kernel stream 上做设备到设备拷贝，避免每次调用的同步 H2D 开销。

#### 3.2.1 host侧设计：

**（1）参数校验与早退**

- handle 为空返回 ACLBLAS_STATUS_HANDLE_IS_NULLPTR；alpha/beta 为空、维度为负、枚举非法、主维不满足约束、batchCount>0 时指针数组为空，均返回 ACLBLAS_STATUS_INVALID_VALUE。
- m=0/n=0/batchCount=0：直接成功返回。
- k=0 或 alpha=(0,0)：构造 CgbtEpilogueTiling，提交 scale kernel 后返回；beta=(1,0) 时直接返回（C 不变）。

**（2）对齐维度与 workspace 布局**

- mP=CeilAlign(m,64)、nP=CeilAlign(n,64)、kP=CeilAlign(k,64)；Cube tile 固定 CUBE_M=CUBE_N=CUBE_K=64；AIV 向量块 AIV_CHUNK=1024。
- 记 xFloats=nP*kP、yFloats=kP*mP、tFloats=nP*mP，单批次 perBatch = 3*xFloats + 3*yFloats + 3*tFloats（float）。每批次布局：

```
[Xr nP*kP][Xi nP*kP][Yr kP*mP][Yi kP*mP][AddX nP*kP][AddY kP*mP][T1 nP*mP][T2 nP*mP][P1 nP*mP]
```

- 全部批次数据区之后是 per-core 的 C0 快照区（仅 combine 补偿使用），按 64 字节对齐。
- Gather 偏移表不放入每次调用的 workspace，而由进程级常驻缓冲区提供，保证同 stream 上先于 combine 可见。

##### 1. 分核策略：

遵循"满核优先 + 不浪费核"原则，通过 ClampCores(total, cores) 把核数限制为 min(任务并行度, 物理核数)：

- packB 段任务并行度 = batchCount * nP（X 区行数），AIV 物理核 40。
- packA 段任务并行度 = batchCount * kP（Y 区行数）。
- gemm 任务并行度 = batchCount * (nP/64) * (mP/64)，AIC 物理核 20。
- combine 任务并行度 = batchCount * n（按批次×输出列组划分），AIV 40。
- scale 任务并行度 = batchCount * n。

packB 与 packA 在一个合并 kernel 内表示为两段连续的 flat 行空间（B 段 [0,rowsB)、A 段 [rowsB,rowsB+rowsA)），每核通过 BlockRange 在合并空间上取 [begin,end)，再分别与两段求交，绝不使用 BlockRange 跨两个逻辑区分块（避免跨区覆盖问题）；合并 kernel 的核数取两段并行度的较大者，落在较小段上的核自然得到空区间。

##### 2. 数据分块和内存优化策略：

- Cube：输出固定 64×64 tile，K 维以 64 为步长；ND2NZ 参数按 nValue/dValue=64、srcDValue=kP(或 mP) 配置，Fixpipe 按 dstStride=mP 写回。
- AIV（pack/combine）：CHUNK=1024 float，单次 DMA 突发达到 8～16KB 量级；Gather pattern 表覆盖 s=1..1024，CHUNK 与 pattern 最大长度同步配置。
- pack 紧凑非转置路径采用跨行成组：当 isTrans=0、rowStride=cols、srcLd=cols 且 cols 为 8 对齐时，连续若干行在源和 workspace 上都连续，合并为一次 AoS 突发读、一次解交错、三次连续写；中小列宽进一步使用每个平面 2048 float 的双槽（aos 槽为其两倍），组成 MTE2(取数)/V(解交错与加法)/MTE3(写回) 三级流水。
- combine 对齐路径使用 8 列一组的 fast8 批量 ND 搬运与整块 Gather（要求 m%8==0、m≤256、beta=0、无需快照），其余路径按 CHUNK 走通用 DataCopyPad + Gather。
- workspace 与 C 均按缓存行对齐访问；非对齐尾块统一走 DataCopyPad（32B 对齐 burst + 标量 lead/tail）。
- 跨核数据依赖通过 DataCacheCleanAndInvalid 维护：pack 末尾对 workspace 做 clean 使 AIC 可见，gemm 末尾 clean 使 AIV combine 可见，combine 开头对 workspace invalidate、对 C 的输出按批次做 clean；不依赖隐式跨核缓存一致性。

##### 3. tilingkey规划策略：

实现不使用编译期 TilingKey 分支，而通过运行时 tiling 字段在同一 kernel 内选择路径（避免多分支二进制体积、保持小 shape 低启动开销）：

- CgbtPackTiling：isTrans（0=N，1=T/C）、isConj（C 时虚部取反）、doAdd（融合实+虚）、rows/cols/rowStride/rowsPadded/srcLd/offReal/offImag/offAdd、usedAivCores。
- CgbtMergedPackTiling：内含 pb、pa 两段 CgbtPackTiling 与 rowsB/rowsA 段边界。
- CgbtGemmTiling：各区域偏移、mP/nP/kP、mTiles/nTiles、totalTasks、usedAicCores。
- CgbtEpilogueTiling：T1/T2/P1 偏移、alpha/beta、hasBeta、usedAivCores、enableCorrection、m/n/mP/ldc、trans/lda/ldb、C0 快照区 snapOff/c0Slot/snapPerGroup、pattern 偏移、fast8 pattern 偏移。
- 路径选择：host 根据 beta 是否为 0 与 m*n 是否 ≤128 决定 enableCorrection；kernel 根据 fast8 条件、isTrans、对齐情况选择 fast8 / 通用 / 转置标量分支；k=0 或 alpha=0 由 host 直接路由到 scale kernel。

#### 3.2.2 kernel侧设计：

**（1）pack kernel（AIV，合并）**

- 每个 UB 持有 aos、real、imag、addOut 四块缓冲；三级流水时重解释为两个等大 ping-pong 槽。
- N（连续列）路径：源列在 GM 上连续，DataCopy 读交错 AoS → GatherMask 虚实分离为 real/imag → 可选 Muls 做共轭虚部取反 → 可选 Add 得 addOut → 连续写 real/imag/addOut 三区；非 8 对齐的列首/列尾用标量 GetValue/SetValue 兜底。
- T/C（转置）路径：输出[r][c] 取源 (r,c) = 源复下标 r + c*srcLd，逐元素标量搬运并处理虚部取反；该路径面向非紧凑转置，正确性优先。
- padding：r≥rows 的整行为零填充；每行 [cols,rowStride) 列按"标量 lead → 8 对齐 bulk Duplicate/DataCopy → 标量 tail"处理。
- 三级流水双缓冲事件严格配平（每次调用内信号量收支平衡，可在每个 batch 重复调用而不累积）：MTE2→V 与 V→MTE2 复用每槽事件 ID0/ID1，V→MTE3 与 MTE3→V 复用每槽 ID2/ID3；首个槽预取不等"空闲"令牌，循环中"写回(g-1)、预取(g+1)、计算(g)"交叠，末尾回收两个方向的尾令牌，单组（groups=1）时只回收实际使用的槽，避免跨调用死锁。
- 同一批次的源指针只在批次切换时 LoadDevicePtr 一次；连续 padding 行复用一次零模板。

**（2）gemm kernel（AIC）**

- 每核以 task=bid 起步、步长 bnum 遍历 totalTasks，解析 (batch, tileM, tileN)。
- 每个输出 tile 顺序执行 P2=Xr*Yr、P3=Xi*Yi、P1=AddX*AddY 三次 Product。
- Product 内 L1 为共享 SRAM，四个槽连续布局 A0@0、A1@8KB、B0@16KB、B1@24KB（A1/B1 是同一块 L1 的视图，必须避免地址重叠）；K 循环每轮在一个槽上 LoadData/MMAD，同时向另一个槽 ND2NZ 预取 k+1，事件按槽配对，保证"MTE1 读完旧槽 → MTE2 才可覆写"和"MTE2 搬完 → MTE1 才可读"双向都有握手，从根本上消除同址 workspace 复用、几何翻转时的踩槽竞态。
- 首个 K 块 cmatrixInitVal=true 清零累加，后续块累加；MMAD 关闭 HF32（SetHF32Mode(DISABLE)）保证 FP32 精度；Fixpipe 经 M/FIX 事件与下一 Product 的 CO1 复用串行。
- kernel 末尾对 workspace 做 ENTIRE_DATA_CACHE clean，保证 combine 可见。

**（3）combine kernel（AIV）**

- 按 (batch, 列组) 分核，列组宽度 groupCols = 4 / gcd(ldc,4)，保证组内列的 32B 写出块长与地址都对齐。
- 每个 CHUNK：批量 GmToUbPad 读 T1/T2/P1（及 beta≠0 时的 C0）与 Gather 偏移表 → 向量计算 pR=T1-T2、pI=P1-T1-T2 → alpha 复数缩放（4 次 Muls、2 次 Add/Sub）→ beta 路径 AosToSoa C0 并叠加 → merge 出 [oR;oI] → Gather 按字节偏移交错 → DataCopyPad 连续写回列主序 C。
- fast8 路径：8 列、m 为 8 对齐且 ≤256、无需快照时，用 DataCopyParams 二维参数一次搬 8 列，整块向量化后一次 Gather、一次 ND 写，显著降低小/中规格开销。
- 精度补偿：enableCorrection（beta≠0 或 m*n≤128）时先快照原始 C0；对每个元素用 double-float 标量运算估计消去程度，当 |q| ≤ 5%*(|alpha*p|+|beta*c0|) 判定为强消去，调用 CgbtCorrectElement 直接从 A/B 原始复数做位精确重算；非强消去元素仍走向量快路径。
- 批次切换时按设备指针数组 LoadDevicePtr 获取 C（及补偿所需 A/B）地址，并维护 cG 的 clean。

**（4）scale kernel（AIV，早退）**

- k=0 或 alpha=0 时执行 C=beta*C：beta=0 直接写零，否则读 C0 → 复数缩放 → Gather 交错写回；事件/分块方式与 combine 常规路径一致。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas 800I A2 推理产品 | √ |
| Atlas 800T A2 训练产品（及同 arch22/dav-2201 的 Atlas A2/A3 系列） | √ |

实现基于 arch22（dav-2201）指令集，使用 AIC Cube 单元（FP32 MMAD，20 核）与 AIV Vector 单元（40 核，与 AIC 为 1:2）；CANN 9.1.0 工具链编译。

## 算子约束限制

- 仅支持 Complex FP32（FP32 实部/虚部），不支持 FP16/BF16/整数复数类型。
- 矩阵为列主序、实虚部交错存储；不支持非交错（独立实/虚数组）布局输入。
- batch 为 uniform：各批次共享 m/n/k/lda/ldb/ldc/transa/transb，仅矩阵首址可不同；不支持逐批不同形状或不同转置。
- 不支持操作数之间的重叠（overlap）语义；各批次 C 矩阵之间不得重叠。
- 维度与主维约束如上表；非法参数返回对应错误码，不做静默纠正。
- 不涉及广播（BLAS GEMM 语义，区别于 element-wise 算子）。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述(不涉及说明原因) | 标准来源 |
| --- | --- | --- |
| 精度标准 | golden 由 cblas（Netlib BLAS 复数实现）逐批生成，实部、虚部分别独立比对；逐元素满足 \|actual-golden\| ≤ atol + rtol*\|golden\|，其中 rtol = 2^-10（9.7656e-4）、atol = 2^-16（1.5259e-5）；同时 max_abs_error ≤ max(1e-2, 32*ULP_at_each_element)，matched_ratio ≥ 0.99（实部、虚部各自判定）。alpha=0 且 alpha 非空的跳过位精确特例按位一致（EXACT）校验。 | 生态算子开源精度标准（experimental_standard）、任务书 |
| 性能标准 | 设备 aclrtEvent 计时，先 warmup 再有效采样 100 次取平均，4 个规格平均单次耗时分别不高于：256/256/256/b32 ≤ 342.44us；512/512/512/b16 ≤ 1462.32us；1024/1024/1024/b8 ≤ 5018.64us；2048/2048/2048/b4 ≤ 18032.75us（transA=transB=N，alpha=(1,0)，beta=(0,0)）。 | 任务书性能标杆 |

可维可测手段：

- 精度：gtest 参数化用例（1200 例 CSV 驱动）覆盖小 shape、质数/非对齐、N/T/C 全组合、padding、边界维度、负向参数（空指针/非法枚举/负维度）、alpha/beta 特殊值（(0,0)、(1,0)、纯虚数）、Inf/NaN、batchCount 1～1024；先小用例冒烟再全量回归。
- 性能：独立 cgemm_batched_benchmark（warmup=20、repeats=100、aclrtEvent 区间平均）与 host 侧分阶段设备事件计时（CG_STAGE_EVENTS，分别测 pack/gemm/combine 纯设备耗时），用于瓶颈定位。
- 调试：host 侧保留 CG_BENCH/CG_DEBUG/CG_PIPELINE 环境变量开关输出分阶段 wall time，默认关闭不影响正常路径。

## 兼容性分析

- 新算子在 ops-blas 仓 arch22 目录新增，接口声明置于 cann_ops_blas.h 既有 aclblasCgemmBatched 名下，与 arch35 已有同名实现并存，按产品/架构分别编译，不修改公共接口签名，不影响其他 BLAS 算子。
- 语义对标 cuBLAS cublasCgemmBatched 与 Netlib cgemm，返回码、边界（空维度、k=0、alpha=0、beta 取值）行为与对标接口对齐，上层可无缝切换。
- 仅复用句柄已有 workspace 机制（按需增长，不缩小、不跨调用释放），并新增进程级只读常量缓冲区（Gather 偏移表），不改变用户可见内存模型。
