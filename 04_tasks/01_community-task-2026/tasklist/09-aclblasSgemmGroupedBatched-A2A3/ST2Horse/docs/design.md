# aclblasSgemmGroupedBatched 算子设计文档（Atlas A2/A3）

版本：v1.1（设计评审稿），2026-09-23  
代码仓：cann/ops-blas，新增 `blas/gemm_grouped_batched/arch22/`  
目标环境：CANN 9.1.0；Atlas A2 与 Atlas A3

## 1. 需求背景（required）

本设计对应 [2026 年 9 月社区任务 aclblasSgemmGroupedBatched 算子开发](https://www.hiascend.com/activities/task-center/details/e531ef0ad20a4896a31654798c933d43?menu=tasks)。接口分组、列主序 GEMM 数学语义参考 [NVIDIA cuBLAS Grouped Batched GEMM 文档](https://docs.nvidia.com/cuda/cublas/#cublas-t-gemmgroupedbatched)。具体参数约束和指针数组驻留位置以本任务书为准。主仓已有公共 API 和 arch35 实现，但 arch35 kernel 使用 Ascend 950 的 MicroAPI 与 `__VEC_SCOPE__`，无法在 A2/A3 的 arch22 编译。新增 arch22 Host 与 Ascend C kernel，保持已有公共接口签名和 arch35 实现不变。

分组批量 GEMM 允许每组分别设置转置、m/n/k、alpha/beta、前导维和 batch 数；不同组的形状完全独立。组内每个矩阵仍有独立的 A/B/C Device 地址，语义参考 cuBLAS cublasSgemmGroupedBatched。

## 2. 需求分析（required）

### 2.1 计算与数据布局

组 g 的第 j 个 batch 使用扁平下标 `idx = sum(groupSize[0:g]) + j`，计算

`C[idx] = alpha[g] * op(A[idx]) * op(B[idx]) + beta[g] * C[idx]`。

A、B、C 为 FP32 列主序矩阵。支持 N/T 四种转置组合 NN、NT、TN、TT；实数算子不接受 OP_C。Aarray、Barray、Carray 是 Host 内存中的扁平指针数组，数组元素为 Device 矩阵地址。接口不含 batch stride，也不对矩阵做广播；每组只共享本组形状、标量和前导维。

### 2.2 参数与边界

Host 校验 handle、groupCount、必需数组、各组维度、groupSize、转置和前导维。非法转置或数值参数返回 `ACLBLAS_STATUS_INVALID_VALUE`，空 handle 返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`。每组满足 `lda >= max(1, transa=N ? m : k)`、`ldb >= max(1, transb=N ? k : n)`、`ldc >= max(1,m)`。批次总数使用宽整数检查溢出。

`groupCount=0`、`groupSize=0`、`m=0` 或 `n=0` 是合法空操作。`k=0` 或 `alpha=0` 时仅执行 `C=beta*C`，不读取 A/B；`beta=0` 时不读取旧 C。实际需要读写的矩阵地址不能为空；零维 no-op 不访问矩阵，`k=0` 或 `alpha=0` 不读取 A/B。任务书参数表对空地址的总体约束与部分合法退化用例存在冲突，处理优先级列于第 2.4 节待评审。C 的列间 padding 不得被改写。A/B 与 C 重叠不在接口支持范围内。


### 2.3 接口数据与异构组示例

公开接口沿用 `include/cann_ops_blas.h` 中的 `aclblasSgemmGroupedBatched`，由 handle 绑定 stream，不增加产品专有参数。`transaArray`、`transbArray`、`mArray`、`nArray`、`kArray`、`alphaArray`、`betaArray`、`ldaArray`、`ldbArray`、`ldcArray` 和 `groupSizeArray` 均按组索引。本任务要求指针数组本身位于 Host、元素为 Device 矩阵地址；这与 cuBLAS 原接口的指针数组驻留位置不同，适配时以任务书为准。A/B/C 指针数组按所有组的 batch 顺序展平；第 g 组对应的起点为 `groupOffset[g] = sum(groupSizeArray[0:g])`。

例如组 0 使用 NN、`m/n/k=128/256/64`、batch 2；组 1 使用 TN、`m/n/k=512/96/128`、batch 3。两组可分别设置前导维与 alpha/beta，设备端以 `groupOffset=[0,2]` 找到各自的矩阵地址。组与组的形状、转置、batch 数和矩阵数据彼此独立。

### 2.4 契约边界与设计约束

| 项目 | 设计约束 |
| --- | --- |
| 指针数组 | 顶层 Aarray/Barray/Carray 按任务书参数表必须存在；有效计算所需的矩阵元素必须为非空 Device 地址。零维与不读取 A/B 的退化用例允许未使用地址为空，需与任务方确认其优先级。 |
| 别名 | A/B 与 C 重叠不在支持范围内；组与组的矩阵彼此独立。 |
| 连续异步调用 | 工作区属于 handle；连续调用须保证前次 kernel 完成读取后才能复用对应元数据。通过同 stream 顺序、必要同步或多缓冲实现，并在设计评审后固定生命周期约束。 |
| 任务书冲突 | 参数表把任一空矩阵地址列为非法，但附件中的零维及 `k=0` 合法用例可能给未使用矩阵传空地址。本方案按 no-op / 不读取的语义优先处理；若评审明确要求全部非空，应同步调整附件用例与实现。 |

## 3. 详细设计（required）

### 3.1 工程结构与 Host 流程

新增 `blas/gemm_grouped_batched/arch22/gemm_grouped_batched_host.cpp`、`gemm_grouped_batched_kernel.cpp` 和 `gemm_grouped_batched_tiling_data.h`；测试放在 `test/gemm_grouped_batched/arch22/`。Host 顺序执行：

1. 校验参数与实际参与计算的矩阵指针，计算各组 batch 偏移和总 batch 数。
2. 将每组参数、tiling 和 Host 指针数组中保存的 Device 地址值打包到 handle 工作区。
3. 在 handle 绑定的 stream 上下发 AIV 或 Cube kernel；需要两种路径时分别下发。结果读回由调用者同步。

Host 上传的是 Device 地址值，不把 Host 指针数组本身交给设备。工作区依次保存 tiling、按 256 字节对齐的 GroupParam 表，以及 A/B/C 三组 64 位 Device 地址表；空间随组数 G 和总 batch 数 B 按 O(G+B) 增长。Host 在写入前检查工作区容量，不足则返回错误并提示扩容。元数据通过 H2D 拷贝上传，kernel 沿 handle stream 发射；大批次元数据可按工作区容量分块下发。连续异步调用须按第 2.4 节的生命周期约束处理。

### 3.2 分组调度与 Ascend C 计算

设备端 `GroupParam` 保存转置、m/n/k、lda/ldb/ldc、alpha/beta、groupSize、batchOffset。总 batch 数为 B = Σ groupSize[g]。AIV 核心 core 处理 batch=core, core+coreNum, ...，再由组前缀和查找所属组；一个 batch 由一个 AIV 独占，避免相邻核心细粒度写入同一 GM 32B 块。它处理退化计算、一般 alpha/beta、未满足 Cube 直写条件的布局以及极小形状。极小形状 `m,n,k <= 16` 使用顺序累加，以覆盖极值输入中不同归约顺序导致的 Inf 符号差异。

大矩阵默认直写路径使用 Ascend C `MatmulImpl`。列主序运算等价于行主序 `C^T = op(B)^T * op(A)^T`，因此 Cube 左输入为 B、右输入为 A，逻辑输出为 `n x m`，仍按原 `ldc` 写回。NN/NT/TN/TT 分别实例化编译期 Matmul 类型，并按组的转置标志传入运行时参数。实际输出 tile 为 256 x 256，内部基础块 128 x 128、K 基础块 64；尾块按真实剩余尺寸处理。每组 Cube 任务数为 groupSize[g] × ceil(n/256) × ceil(m/256)。AIC 核心 core 处理任务 core, core+coreNum, ...；任务号反解为组内 batch、输出行块和列块。每个任务只写 C 的一个不重叠 tile，尾块使用真实剩余维度，组间形状互不混用。

Cube 直写条件设为 `k>0`、`alpha=1`、`beta=0`、`ldc` 按 8 个 FP32 元素对齐且不属于上述极小形状。其它合法输入由 AIV 基线处理。对一般 alpha/beta，可引入 Cube 中间结果和 AIV 融合，避免大矩阵落到标量路径；中间结果分块受工作区容量约束。

### 3.3 性能优化方案与风险

| 路径 | 设计选择与优化方向 | 需验证的风险 |
| --- | --- | --- |
| 小形状、退化计算 | AIV 基线按 batch 分核，避免 Cube 初始化开销；仅在需要时读 A/B 或旧 C | 特殊值与归约顺序、C padding、零维与非法指针 |
| 通用 FP32 GEMM | Cube 按组、batch 与 256×256 输出 tile 分任务，K 维按基础块迭代 | 四种转置的地址映射、尾块、前导维、GM 写入对齐 |
| 非默认 alpha/beta | 先算 Cube 中间结果，再以 AIV 完成缩放和 C 的融合更新 | 中间结果占用、额外 H2D/GM 流量、beta=0 不读旧 C |
| 极大批次与连续调用 | 元数据分块、多缓冲或受控复用，减少重复初始化 | 工作区容量、stream 顺序及异步生命周期 |

优化过程保留稳定实现，每次只改一个主要因素；先做精度与边界回归，再按任务书规定的预热和有效采样方法比较性能。性能瓶颈以实际 profiling 数据定位，不把未经验证的原因写成结论。

## 4. 支持硬件与算子约束（required）

| 产品 | 目标 SoC 示例 | 实现目录 |
| --- | --- | --- |
| Atlas A2 | Ascend910B3 | arch22 |
| Atlas A3 | Ascend910_9382 | arch22 |
| Ascend 950 | 主仓既有 arch35 | 保持既有实现 |

支持 FP32、列主序、N/T 转置、Host 指针数组及 handle stream。A/B 与 C 重叠、额外 stride、广播及任务书未要求的其他数据类型不在本方案范围。小形状和退化计算使用 AIV；大矩阵优先使用 Cube，非默认 alpha/beta 的融合路径按第 3.3 节完善。

## 5. 可维可测分析（required）

### 5.1 精度与功能验证计划

以仓库 GTest/CBLAS 逐组逐 batch 生成 golden，输出 C 的所有有效元素按任务书混合容差判定：`atol=rtol=2^-13`、匹配率至少 0.99、最大绝对误差阈值 0.01 或 32 ULP。保留任务附件 CSV 作为验收输入来源，同时为任务书未覆盖的边界补充单测。

验证维度包括：NN/NT/TN/TT、各组不同 m/n/k 与 batch 数、非单位 alpha/beta、lda/ldb/ldc 和 C padding、零维、k=0、alpha=0、非法参数、矩阵指针空值、Inf/NaN、对齐偏移与不完整输出 tile。A2/A3 分别构建并执行同一组测试，记录源码版本、CANN、SoC 和原始日志。

构建命令分别为 `bash build.sh --ops=gemm_grouped_batched --soc=ascend910b3` 与 `bash build.sh --ops=gemm_grouped_batched --soc=ascend910_9382`；测试程序为 `build/test/gemm_grouped_batched/gemm_grouped_batched_test`。性能使用 msprof 的逐 kernel `Task Duration(us)`，不以包含 Host 准备和 CPU golden 的 GTest 总耗时替代。

### 5.2 910B3 性能指标

| case | groupSizeArray | 每组 M=N=K | 转置 | 目标 Avg kernel 耗时 |
| --- | --- | --- | --- | --- |
| 1 | [64,64] | 256 | NN | 351.3 μs |
| 2 | [128,64] | 512 | NN | 3714 μs |
| 3 | [64,128,64] | 1024 | NT | 37652 μs |
| 4 | [128,128] | 2048 | TN | 298380 μs |
| 5 | [64,64] | 4096 | NN | 1188440 μs |

任务书规定 5 次预热、超过 10 次有效采样取均值；每个 case 分别保存原始性能日志、均值和设备信息。内存验证记录工作区分配、峰值和越界检查。

### 5.3 交付与复现

最终交付包括 arch22 Host/kernel/tiling、测试 CSV 与 GTest、测试步骤、精度/性能/内存报告及设计文档。设计文档放在 cann-competitions 的 `04_tasks/01_community-task-2026/tasklist/` 对应任务目录；算子代码另按社区流程提交至 cann/ops-blas。A2/A3 使用同一份 arch22 源码分别构建验收。

## 6. 兼容性分析

保持现有 `aclblasSgemmGroupedBatched` 公共 ABI，不增加产品私有接口。构建系统按 SoC 选择 arch22 或 arch35。Cube 调度改变 FP32 累加顺序，按任务书混合容差验收，不要求逐位一致。极小形状使用顺序 AIV 路径覆盖特殊值语义。
