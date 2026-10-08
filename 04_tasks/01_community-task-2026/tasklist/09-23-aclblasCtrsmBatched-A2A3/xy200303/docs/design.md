# 需求背景（required）

## 需求来源

昇腾 9 月社区任务：`aclblasCtrsmBatched` 算子开发（A2/A3），任务书 `aclblasCtrsmBatched_A2A3_task_doc.md`。

## 背景介绍

### aclblasCtrsmBatched 算子实现

基于 ops-blas 开源仓（https://gitcode.com/cann/ops-blas ）使用 Ascend C 为 Atlas A2/A3（arch22，DAV C220）实现 BLAS Level-3 单精度复数批量三角求解算子 `aclblasCtrsmBatched`。对每个 batch i（`i = 0 .. batchCount-1`）独立求解，解原地覆写右端矩阵 B[i]：

- `side = LEFT`：`op(A[i]) * X[i] = alpha * B[i]`，X[i] 写回 B[i]
- `side = RIGHT`：`X[i] * op(A[i]) = alpha * B[i]`，X[i] 写回 B[i]

`op(A)` 由 `trans` 决定：`OP_N` 为 A，`OP_T` 为 Aᵀ，`OP_C` 为 Aᴴ（共轭转置）。矩阵列主序。仅 `uplo` 指定三角被引用；`diag = UNIT` 时对角视为 1+0i 且不读取对角存储。

对标接口：cuBLAS `cublasCtrsmBatched`。精度 golden 为测试工程里的 Netlib `cblas_ctrsm`，内层复数乘对齐 libgcc `__mulsc3`。

实现现状：

- 仓内 `blas/trsmbatched/arch35/` 已有实数 `aclblasStrsmBatched`，面向 arch35（950），不能直接用于本任务的 arch22。
- `include/cann_ops_blas.h` 增加 `aclblasCtrsmBatched` 声明，不另起平行 API。
- `aclblasComplex`（`{float real; float imag;}`）定义于 `include/cann_ops_blas_common.h`。设备上按交织 float 存放，一个复数两个 float。
- 任务书 §2.2 写落点 `blas/trsm/`，§5 写落点 `blas/trsmbatched/arch22/`。实现按 §5 与现有 batched 目录放在 `blas/trsmbatched/arch22/`。

### aclblasCtrsmBatched 算子功能分析

| 参数 | 参数含义 | 输入/输出 | 数据类型 | 支持数据类型 | 约束 | 形状 |
| --- | --- | --- | --- | --- | --- | --- |
| handle | 上下文句柄，携带 stream | 输入 | scalar | - | 空指针返回 HANDLE_IS_NULLPTR | - |
| side | LEFT：解 op(A)X=alpha·B；RIGHT：解 X·op(A)=alpha·B | 输入 | attr | int | 仅 LEFT/RIGHT，否则 INVALID_VALUE | - |
| uplo | 引用上三角或下三角 | 输入 | attr | int | 仅 UPPER/LOWER | - |
| trans | op(A)：N / T / C（C 为共轭转置） | 输入 | attr | int | 仅 OP_N/OP_T/OP_C | - |
| diag | NON_UNIT 读对角；UNIT 对角视为 1 | 输入 | attr | int | 仅 NON_UNIT/UNIT | - |
| m | B 的行数；LEFT 时也是 A 的阶 | 输入 | scalar | int | m≥0；m=0 为 no-op | - |
| n | B 的列数；RIGHT 时也是 A 的阶 | 输入 | scalar | int | n≥0；n=0 为 no-op | - |
| alpha | 复数标量，仅 Host 指针 | 输入 | scalar | COMPLEX64 | 空指针 INVALID_VALUE；alpha=(0,0) 时把各 B[i] 置零 | - |
| A | 设备侧指针数组，A[i] 只读三角矩阵 | 输入 | tensor | COMPLEX64 | 数组长度 batchCount；alpha≠0 时数组及元素不可空 | LEFT：m×m，lda≥max(1,m)；RIGHT：n×n，lda≥max(1,n) |
| lda | A 的前导维 | 输入 | scalar | int | 见上 | - |
| B | 设备侧指针数组，输入右端、输出解，原地覆写 | 输入/输出 | tensor | COMPLEX64 | 各 B[i] 不得互相重叠 | 各矩阵 m×n，ldb≥max(1,m) |
| ldb | B 的前导维 | 输入 | scalar | int | ldb≥max(1,m) | - |
| batchCount | 批次数 | 输入 | scalar | int | batchCount=0 为 no-op；&lt;0 返回 INVALID_VALUE | - |

计算公式（列主序，k = side==LEFT ? m : n）：

```
对每个 batch i：
  B[i] ← alpha * B[i]
  求解 op(A[i]) * X = B[i]   （LEFT）
  或   X * op(A[i]) = B[i]   （RIGHT）
  解 X 写回 B[i]
```

复数乘法按 libgcc `__mulsc3`：有限值走 Dekker 误差补偿，两个结果分量都是 NaN 时才按无穷的符号把结果收成有符号无穷。`OP_C` 在准备 A 时对虚部取反。`diag=UNIT` 时不读 A 的对角，代入时不再除对角。`diag=NON_UNIT` 时对角须非零，本算子不做奇异性检测。`alpha` 恰好为 (1,0) 时跳过列缩放，避免 `1 * inf` 变成 NaN。

# 需求分析（required）

## 需求描述

使用 Ascend C 为 Atlas A2/A3（arch22）实现 `aclblasCtrsmBatched`，支持 COMPLEX64、LEFT/RIGHT、UPPER/LOWER、OP_N/T/C、UNIT/NON_UNIT、lda/ldb padding、原地覆写 B，满足任务书异常返回码、生态算子混合容差精度标准，以及任务书 §3.3 五条性能标杆。

## 需求拆解

1. 功能语义对齐 cuBLAS `cublasCtrsmBatched`：批量、均匀尺寸、列主序、解原地写回 B。
2. 返回码：handle 空 → `HANDLE_IS_NULLPTR`；非法枚举、负维度、前导维不足、alpha/A/B 空、指针数组元素空 → `INVALID_VALUE`。
3. `m=0`、`n=0`、`batchCount=0` 返回 SUCCESS，不读 alpha、A、B。`batchCount<0` 返回 `INVALID_VALUE`。验收 CSV 的 `TC_ED_184` 是 `batchCount=0` 且指针数组可空，实现按 CSV 走 no-op。
4. `alpha=(0,0)` 时不引用 A，对每个 B[i] 做 `aclrtMemset` 置零后返回 SUCCESS。
5. `OP_C` 与 `OP_T` 分开：C 是共轭转置，T 是普通转置。
6. 精度按 COMPLEX64 混合容差：rtol=atol=2^-13，matched_ratio≥0.99，并且 max_abs_error ≤ max(1e-2, 32×ULP)。golden 为 Netlib `cblas_ctrsm`。
7. 性能：任务书 §3.3 五条 case 的平均 kernel 耗时不超过 895.1 / 5551 / 13884 / 21816 / 40969 us。设备是 Atlas 800T A2（910B3），`msprof op` 自带 5 次 warmup，有效采样大于 10 次取平均。
8. 自测工程放在 `test/trsmbatched/ctrsmbatched/`，覆盖随任务提供的 CSV。接口没有 C/ldc，`TC_ED_193`、`TC_ED_194`、`TC_ED_202` 由 GTest 跳过。

# 详细设计（required）

## 算子分析

### 数学公式

记复数对角元 `d = dr+di·i`。代入法对一行或一列：

```
x = (b − Σ a·x_known) / d          （NON_UNIT）
x =  b − Σ a·x_known               （UNIT，不读 d）
```

复数除法：`b/d = (br·dr+bi·di)/(|d|²) + (bi·dr−br·di)/(|d|²) i`。

列主序元素 `(i,j)` 位于 `i + j*ld`。内核按行主序读同一块内存时，看到的是转置：列主序 `op(A)*X = αB` 变成 `Xᵀ * op(A)ᵀ = αBᵀ`。因此 NON_UNIT 进入 MIX 核之前交换 `m/n`、交换 LEFT/RIGHT、翻转 uplo，`trans`、`diag`、`lda`、`ldb` 和指针保持不动。UNIT 不走这次重映射，按调用方原始 side/uplo/trans 做逐元代入。

`needTA = ((T 或 C) 且 LEFT) 或 (N 且 RIGHT)`，按重映射之后的 side 计算。需要转置时再把 uplo 翻一次，`effUplo == LOWER` 则前代，否则回代。四条模板是 `CtrsmLowerLeft`、`CtrsmLowerRight`、`CtrsmUpperLeft`、`CtrsmUpperRight`。用户侧 `LEFT + LOWER + N + NON_UNIT` 进内核后是 `RIGHT + UPPER + N`，`needTA` 为真，模板为 `CtrsmLowerRight`。

### 支持数据类型

COMPLEX64。GM 上是交织 `aclblasComplex`。MIX 工作区把一行收成 `[实部半行 | 虚部半行]`，一次实数 GEMM 的 N 覆盖两半。32 字节对齐单元为 8 个 float。

### 支持形状

- A[i]：LEFT 时 m×m，RIGHT 时 n×n，列主序，lda 可大于紧凑前导维。
- B[i]：m×n，列主序，ldb 可大于紧凑前导维，原地覆写。
- batch 为 uniform：所有 batch 共享 m、n、lda、ldb 和四个枚举。
- 不广播。各 B[i] 之间不得重叠。

## 算子实现

### 实现方案

文件在 `blas/trsmbatched/arch22/`：

```
ctrsmbatched_host.cpp
ctrsmbatched_kernel.cpp
ctrsm_batched_tiling_data.h
ctrsm_batched_kernel_do.h
ctrsm_batched_kernel_common.h
ctrsm_batched_kernel_aic.h
ctrsm_batched_kernel_aiv.h
ctrsm_batched_kernel_aiv_cfg.h
ctrsm_batched_kernel_aiv_convert.h
ctrsm_batched_kernel_aiv_canon_a.h
ctrsm_batched_kernel_aiv_canon_b.h
ctrsm_batched_kernel_aiv_canon_b_deinterleave.h
ctrsm_batched_kernel_aiv_solver.h
ctrsm_batched_unit_kernel.h
trsm_mix_common.h
include/cann_ops_blas.h
```

计算路径按 `diag` 与形状分流，不按用例编号分流：

- `diag = UNIT`（默认）：`LaunchCtrsmUnitKernel`。20 个 mix block 只跑 AIV，AIC 直接返回。保持调用方列主序，逐行或逐列代入。复数乘用 Dekker（分裂常数 4097，volatile float，目标指数 177），两个分量都是 NaN 时才调用与 `__mulsc3` 相同的无穷恢复。`alpha` 为 (1,0) 时 `ScaleAlpha` 直接返回原值。
- `diag = UNIT` 且命中以下任一条件（cubePerf）时，改走与 NON_UNIT 相同的列主序重映射 + MIX 核：三角阶 ≥4096；或 m、n 均 ≥1024 且 batchCount ≥32；或 RIGHT 且（m 或 ldb 非 8 对齐）且 ceil(batchCount/40)·m·n² ≥ 8×10⁹。前两类形状 MIX 吞吐更高；第三类下 UNIT 核的多核按行切分失效，退化为单 AIV 串行，单核工作量超出设备单任务时限。工作量乘积按 int64 计算，避免 int32 溢出导致改道条件失效。
- `diag = NON_UNIT`：列主序重映射之后启动 `ctrsm_batched_mix12_kernel`（`KERNEL_TYPE_MIX_AIC_1_2`，1 个 AIC + 2 个 AIV）。对角 panel 在 AIV 上求解，拖尾用一次宽实数 GEMM 原子加回 B 工作区。

panel 宽度：三角阶 `kDim > 1024` 时 `nb = 32`，否则 `nb = 16`。

#### host 侧设计：

参数校验先于任何设备访问：

1. handle 为空 → `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`。
2. side/uplo/trans/diag 不在枚举内 → `ACLBLAS_STATUS_INVALID_VALUE`。
3. `m<0`、`n<0` 或 `batchCount<0` → `INVALID_VALUE`。
4. `m=0`、`n=0` 或 `batchCount=0` → SUCCESS，不读 alpha、A、B。
5. alpha 或 B 为空 → `INVALID_VALUE`。alpha 非零且 A 为空，或某个 `A[i]`/`B[i]` 为空 → `INVALID_VALUE`。
6. LEFT 时 `lda<max(1,m)`，RIGHT 时 `lda<max(1,n)`，或 `ldb<max(1,m)` → `INVALID_VALUE`。

quick return 与主路径：

- `alpha=(0,0)`：对每个 B[i] 按 `n * ldb` 个复数做 `aclrtMemset`，不读 A，不启动 kernel。
- `diag=UNIT`：默认在列主序重映射之前返回 UNIT 核；命中上述 cubePerf 条件的与 NON_UNIT 同样重映射后启动 MIX 核。
- `diag=NON_UNIT`：交换 m/n、LEFT↔RIGHT、翻转 uplo 后填 tiling 并启动 MIX 核。
- `kDim<128` 且 `batchCount>1`、未分列时，按 batch 逐个启动 MIX 核，避免多个小 batch 在同一 kernel 里写坏 B 的列。

接口在 `aclrtSynchronizeStream` 成功后返回。同步失败返回 `INTERNAL_ERROR`。workspace 和 tiling 缓冲在同步之后释放。

##### 1. 分核策略：

arch22 按 20 个 AIC、40 个 AIV。一个 mix block 占用 1 个 AIC 和 2 个 AIV。

- `useDualAiv = (kDim≥128 且 nColsAligned≥128)`。满足时 `numBlocks = batchCount`，一个 block 处理一个 batch，两个 AIV 按列中点切开，中点向上对齐到 8。
- `useDualAiv` 且 `batchCount≤10` 时按列再切。`maxSplits = 20/batchCount`，每段至少 `max(nb, 64)` 列，段宽按 16 对齐，`numBlocks = batchCount * numSplits`。最后一段吃余数。AIC 的 slot 与 AIV 展开后的 `blockIdx/2` 对齐。
- 不满足双 AIV 时仍是一个 batch 一个 block。不把两个 batch 塞进同一个 Cube。
- UNIT 核固定启动 20 个 block，只使用 AIV。

##### 2. 数据分块和内存优化策略：

每个 AIV 的 UB 预算 192KB（49152 个 float）。

- A 的准备：`useOrigA` 仅当不需要转置、没有 padding、不是共轭、且 `lda==kDim` 并且 `kDim*2` 是 16 的倍数。否则 AIV 把 A 转置或打包进对齐 workspace。半宽不超过 128、行数是 16 的倍数时，加载和转置双缓冲；转置与 scatter 写出之间用 `PIPE_ALL`，窄同步对不齐 scatter。
- B 的准备：解交织成 SoA。RIGHT 先把 AoS 转成工作区行。`alpha` 不是 (1,0) 时再做复数列缩放。
- 拖尾 GEMM：B 工作区一行是 `[实部 | 虚部]`，`ldb = 2*nColsAligned`。Cube 做一次 `IterateAll`，K 方向长度是 `2*actualK`，因为复数在 K 上交织。结果原子加回实部平面。`Matmul` 类型是 ND float，不改模板、不设 `SetFixSplit`。
- 组宽 `LIM_GROUP = 4`。一组内先做当前 panel 的 Direct GEMM，再预更新同组后续 panel，组末做 BigGroupGemm。Xneg 按两组 slot 双缓冲，host 仍按 16 个 slot 分配。`doSplit` 保持关闭，FarGemm 不走到。
- 每个 batch 在 panel 循环前对有效 A 做一次整数据缓存刷新。首波 block（AIC `blockIdx<20`，AIV `blockIdx<40`）额外刷 workspace。MTE3 写出和 MTE2 读入不共缓存，不刷会读到上一轮的行。
- RIGHT 且两半列段都能被快速回写覆盖时，求解完的 panel 直接从 UB 交织写回用户 B。判断只用两边共享的宽度：`nColsOrig==nColsAligned`，两半都为正、都是 8 的倍数、且都不超过 `colTile`。任一条件不满足，两核都不走快速回写，一起过 `DualAivBarrier` 后做补齐回写。单 AIV 路径没有这道屏障，条件不满足时直接补齐回写。
- lda/ldb padding 用 `DataCopyPad`。padding 行在 workspace 里写成 0，避免 NON_UNIT 除到 0 后再把 NaN 带回有效行。

每个 batch 的 workspace：对齐后的 A 平面，加上 B 的 SoA 平面和 Xneg 区。分列时 B/GEMM 区按 split 份数重复，A 平面不重复。

##### 3. tilingkey 规划策略：

host 传 `CtrsmBatchedTilingData` 字段和一份 Cube tiling，不按用例编译多套二进制。kernel 分支：

| 条件 | 路径 |
| --- | --- |
| alpha=(0,0) | Host `aclrtMemset`，不启动 kernel |
| m=0 或 n=0 或 batchCount=0 | SUCCESS，不启动 |
| diag=UNIT | 默认 UNIT 代入核（列主序，不分配 Cube workspace）；cubePerf 形状（大阶/大批量/RIGHT 串行退化）改走 MIX |
| diag=NON_UNIT | 列主序重映射后走 MIX |
| kDim>1024 | nb=32，否则 nb=16 |
| kDim≥128 且列对齐宽度≥128 | 双 AIV |
| 双 AIV 且 batchCount≤10 | 按列切分，铺满 20 个 AIC |
| alpha=(1,0) | 跳过列缩放 |
| needTA 且非 useOrigA | AIV 转置或打包 A；共轭在这次转置里对虚部取反 |
| RIGHT 且两半宽度都可快速回写 | UB 直接写回用户 B |

#### kernel 侧设计：

1. **UNIT 核**（仅 AIV）：按 batch 绑定 A、B。LEFT 按列装入、代入、写回；RIGHT 按行。`ScaleAlpha` 在 alpha 为 (1,0) 时不做乘法。写回后按缓存行作废 B，避免下一 batch 读到旧行。
2. **MIX 的 AIV**：`PrepareA` 与 `PrepareB` 之后进入 panel 循环。每个 panel 读对角块、做三角代入、把解写入 Xneg，再 `CrossCoreSetFlag(FLAG_TRSV)`。有拖尾时预取下一块 A，然后 `CrossCoreWaitFlag(FLAG_GEMM)`。RIGHT 快速回写夹在设旗和等旗之间，与 Cube 重叠。两个 AIV 用 `FLAG_DUAL_AIV` 在 MTE3 上互等。
3. **MIX 的 AIC**：等待 `FLAG_TRSV` 后，对当前 panel 的列做宽 GEMM，`PIPE_FIX` 之后置 `FLAG_GEMM`。组内还有后续 panel 时接着做预更新，组末做跨组更新。AIV 不能在 AIC 仍等待 `FLAG_TRSV` 时返回，否则死锁。

# 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas A2 系列（含 Atlas 800I/T A2，910B3） | √ |
| Atlas A3 系列 | √ |

精度自测可以在 910B3 或 A3 上做一种卡型。性能标杆的设备是 Atlas 800T A2（910B3）。当前开发容器是 Ubuntu 22.04、CANN 9.1.0 的 Ascend910 双 die、单 die HBM 64GB，用于精度和开发期单次 `msprof --task-time=on`。这台机器的单次耗时不是 §3.3 的验收数。

# 算子约束限制

- 不支持广播。batch 内尺寸必须一致。
- 不支持超出 lda/ldb 语义的非连续 Tensor。
- 各 B[i] 不得互相重叠，也不与对应 A[i] 重叠。结果原地写在 B 上，没有独立的 C/ldc。
- `batchCount=0` 是 SUCCESS。`batchCount<0` 是 `INVALID_VALUE`。
- `diag=NON_UNIT` 时对角为零属于未定义输入，算子不检测。`TC_FL_147`、`TC_FL_153`、`TC_FL_154` 是 NON_UNIT 极端值，超过混合容差，实现不按用例编号改写。
- 复数乘只在两个分量都是 NaN 时恢复无穷。有限值的舍入按 Dekker，不使用 double，也不使用 `fmaf`。
- 浮点累加顺序不要求与 golden 逐位一致。MIX 的拖尾是原子加，UNIT 是标量代入。
- 分发只看 diag、形状和转置，不看用例名。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | COMPLEX64：rtol=atol=2^-13，matched_ratio≥0.99，max_abs_error≤max(1e-2, 32×ULP)。实部、虚部分开比对。golden 为 Netlib `cblas_ctrsm`。除 TC_PF 外的 CSV 在 A3 上已跑完：994 通过，失败为 TC_FL_147、TC_FL_153、TC_FL_154，跳过为 TC_ED_193、TC_ED_194、TC_ED_202 | 任务书 §3.2，验收 CSV |
| 性能标准 | §3.3 五条门槛 895.1 / 5551 / 13884 / 21816 / 40969 us。设备 Atlas 800T A2（910B3）。`msprof op`，5 次 warmup，有效样本大于 10 次的 Task Duration 平均 | 任务书 §3.3 |
| 内存标准 | 任务书 §3.4 写明不涉及。设备侧为指针表、Cube tiling、每 batch 一份 A/B/Xneg workspace，以及 matmul 系统 workspace | 任务书 §3.4 |

随任务 CSV 与任务书有三处差异，实现按下表处理：

1. 任务书签名没有 C、ldc，结果原地写 B。说明文档写成独立输出 C。CSV 没有 C/ldc 列。`TC_ED_193`、`TC_ED_194`、`TC_ED_202` 期望对 C/ldc 返回 `INVALID_VALUE`，GTest 跳过，不在算子里补一个不存在的参数。
2. 任务书写 `batchCount<1` 返回 `INVALID_VALUE`。CSV 的 `TC_ED_184` 是 `batchCount=0` 且期望 SUCCESS。实现把 0 当作 no-op，只把负数判为 `INVALID_VALUE`。
3. `alpha=(0,0)` 时把 B 原地置零且不读 A，与任务书 §2.5 一致。

## 兼容性分析

- 只在 `cann_ops_blas.h` 增加 `aclblasCtrsmBatched`，不改 `aclblasStrsmBatched` 和其他已有声明。
- 新文件只放在 `blas/trsmbatched/arch22/`，不改 `arch35` 实数实现。
- 测试放在 `test/trsmbatched/ctrsmbatched/`。
- 参数顺序与任务书 §2.3、cuBLAS `cublasCtrsmBatched` 一致。
