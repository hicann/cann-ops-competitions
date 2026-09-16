# aclblasCgemmBatched 算子设计文档（Atlas A2/A3）

版本：v028；更新日期：2026-09-15；贡献者：weixin_47449709。

# 需求背景（required）

## 需求来源

CANN 社区任务 2026「算子实操工坊（广州站）— aclblasCgemmBatched 算子开发（A2A3）」，需求依据随任务提供的 `aclblasCgemmBatched_Atlas800IA3_task_doc.md`。实现目标仓库为 [cann/ops-blas](https://gitcode.com/cann/ops-blas)，实现目录 `blas/gemm_batched/arch22/`，测试目录 `test/gemm_batched/cgemm_batched/arch22/`。

本文按本任务目录的 [设计模板](../../../../resources/design_template.md) 组织。设计 PR 交付本文；实现代码、自测工程及原始报告作为独立交付件，不将文档合入等同于实现代码已经合入。

## 背景介绍


目标是补充 ops-blas 的 arch22 复数批量 GEMM。使用公共头文件 `include/cann_ops_blas.h` 中已有接口，不增加产品私有接口。矩阵采用列主序、实虚交错 complex64；alpha、beta 在主机，Aarray/Barray/Carray 是设备指针数组。

对每个 batch 执行：

```text
C = alpha * op(A) * op(B) + beta * C
op ∈ {N, T, C}，C 表示共轭转置
```

原始测试为 1000 条非 PF CSV 用例、200 条 PF CSV 用例，以及额外的 NullHandle，共 1201 个 GTest。CSV 和精度判据保持原样。PF 用例在现有 GTest 中执行的是功能/精度检查，四个性能目标另行计时。


开发基线为 ops-blas 提交 `7eae2328a65753bf55cffc489253eb434ea3317e`。本任务补齐 arch22 的复数批量 GEMM 路径，通过 handle 绑定 stream 直接调用 Ascend C kernel，共用已有公共接口，不采用自定义 OPP 安装包交付方式。

# 需求分析（required）

## 需求描述

支持 uniform batch 的列主序 complex64 GEMM，数据实虚交错，alpha/beta 位于 Host，Aarray/Barray/Carray 为 Device 指针数组；所有 batch 共享形状、前导维和转置标志。支持 N、T、C 共九种组合，不支持 batch 间不同 shape 的广播。

```cpp
aclblasStatus_t aclblasCgemmBatched(
    aclblasHandle_t handle, aclblasOperation_t transa, aclblasOperation_t transb,
    int m, int n, int k, const aclblasComplex* alpha,
    const aclblasComplex* const Aarray[], int lda,
    const aclblasComplex* const Barray[], int ldb,
    const aclblasComplex* beta, aclblasComplex* const Carray[], int ldc,
    int batchCount);
```

## 需求拆解与接口行为


- 检查 handle、标量指针、枚举、非负维度、batchCount、leading dimension 和设备指针数组。
- m/n/batchCount 为零时合法快速返回；k=0 或 alpha=(0,0) 走 C=beta*C。
- beta=0 不读取原 C；beta=1 的纯缩放路径可直接返回。
- N/T/C 共九种组合，共轭操作折叠为虚部符号。
- 保留 ldc 的 padding；只覆写逻辑 m×n 输出。
- 工作在调用方 handle 的 stream 上；操作之间依靠同一 stream 顺序执行。
- 从设备平台查询 AIC/AIV 核数，按任务数限制启动核数。

当前实现适配 arch22；双平台实测结果见本文“性能标准与实测结果”。


## 支持形状与约束

| 对象 | 逻辑形状 / 要求 |
| --- | --- |
| op(A[i]) | m×k；N 时 lda>=max(1,m)，T/C 时 lda>=max(1,k) |
| op(B[i]) | k×n；N 时 ldb>=max(1,k)，T/C 时 ldb>=max(1,n) |
| C[i] | m×n，ldc>=max(1,m)，只覆写逻辑输出，保留 padding |
| 标量 | alpha/beta 为 Host complex64；beta=0 时原 C 无需初始化 |
| 维度与 batch | m/n/k、batchCount 非负；所有 batch 共享参数 |
| 地址 | 各 C[i] 不得重叠，调用者保证设备指针指向足量可访问内存 |

# 详细设计（required）

## 算子分析与 Host 设计

计算公式为 `C[i]=alpha*op(A[i])*op(B[i])+beta*C[i]`。Host 校验参数、处理快速返回和缩放条件、查询平台核数、计算对齐尺寸及工作区预算，再按小矩阵/通用 Cube/直接输出条件选择 kernel。工作区不足时沿 batch 切片；单批无法放入可用工作区时返回分配错误。数值路径选择由参数决定，不依赖外部配置文件或环境开关。

## 算子实现与路径选择

### 缩放路径

k=0 或 alpha=0 时使用 AIV 完成复数 beta*C；不进入 Cube，也不读取 A/B 内容。beta=0 输出零。该路径保持原测试中 alpha=0 的严格比较要求。

### 小输出矩阵路径

m*n<=256 时使用 `gemm_batched_small.h` 的 AIV 路径。每个核处理若干 batch：

1. 以 K=16 的 panel 搬入 A/B。
2. 根据转置和 leading dimension 构造 Gather 索引，使一个向量指令同时处理所有输出位置。
3. 沿 K 顺序累加复数外积，避免先计算两个大和再相减造成额外消减误差。
4. 按 Netlib 参考实现的运算次序处理 alpha、beta 和复数乘法。
5. Gather 交织结果并用 DataCopyPad 写回逻辑输出，保留 ldc padding。

A2 向量 MulAddDst 的舍入与主机参考 BLAS 使用的融合乘加不同。当前实现以乘积拆分和 TwoSum 误差补偿恢复相应的舍入结果，并针对非有限数保留普通结果或无穷加数的传播语义。该实现的正确性证据是原始极值与小矩阵测试，不宣称为任意 float32 输入的完整 IEEE FMA 替代。

比较掩码使用向上对齐到 64 元素的长度，缓冲区预留至 256 元素；实际运算与选择仍只处理逻辑元素数。此前不足 64 元素时的掩码尾块是小矩阵失败的重要原因。

只有索引、batch 和 K 归约使用标量循环；矩阵数值计算使用向量指令，不逐元素搬运到标量寄存器计算。

### 通用 Cube 路径

默认采用两个实数 GEMM，每个 GEMM 的归约维扩展到 2K：

```text
P_real = [Br, -Bi] * [Ar; Ai]
P_imag = [Bi,  Br] * [Ar; Ai]
```

这里按输出转置的行主序视图计算，即把原列主序输出看成 n×m 行主序；预处理根据原转置标志决定交错位于行方向还是归约方向。共轭符号在预处理时折叠。

流水为：

```text
工作区指针初始化 → A/B 预处理 → 两个 2K 的 FP32 Cube GEMM → alpha/beta 组合写回
```

Cube 使用 DAV_2201 的 DataCopy/Nd2Nz、LoadData、Mmad、Fixpipe；每核初始化时关闭 HF32。当前 tile 为 M=128、N=128、K=64。沿 batch 和输出 tile 分配任务；L1 和 L0 使用双缓冲，通过显式 HardEvent 保护读写与重用顺序。

源码保留了此前四实数 GEMM 的内部备用分支。useCombined=1 表示通用两个实数 GEMM，2 表示下面的单 GEMM 直接输出路径，0 是旧备用分支。最终验收不依赖外部文件、调试开关或环境变量改变算法。

### 对齐 NN 快速路径

当 N/N、输入无需 padding、lda=m、ldb=k、m<=4096 且 4096 可整除 m 时：

- 使用 `gemm_batched_dense.h`，每个任务搬运 4096 个复数。
- 单个预处理 kernel 处理 A/B，直接生成 combined 视图，避免额外写出 Ar/Ai/Br/Bi。
- GatherMask 分离 A 的实虚部，Gather 和向量符号乘生成 B 的两份视图。
- 若 ldc=m、alpha=(1,0)、beta=(0,0)，直接连续搬入两份结果并 Gather 交织回 C。

所有不满足条件的输入仍走通用预处理/后处理。四个指定性能 shape 均满足这些条件，并进一步命中下面的直接输出路径。

### v028 直接输出与双缓冲展开

在 3.4 条件外，还要求 ldc=m、alpha=(1,0)、beta=(0,0)、m<=2*n。输入物理维度必须满足无 padding 条件。以原列主序矩阵的转置行主序视图理解，B 原始交错数据是 n×2k 的实数矩阵；A 的每一行按以下形式扩展为两行：

```text
[ Ar0, Ai0, Ar1, Ai1, ...]
[-Ai0, Ar0,-Ai1, Ar1, ...]
```

A 展开后为 2k×2m 的实数矩阵，单次 FP32 实数 GEMM 输出 n×2m，恰好是原 C 的实虚交错存储。无需 B 预处理、两份结果临时写回或结果组合 kernel。beta=0 时不读原 C。

展开 kernel 使用两次 float32 TransDataTo5HD 完成成对重排，再通过向量掩码只对偶数元素取负。输入双缓冲 64 KiB、输出双缓冲 64 KiB、转置临时区 32 KiB，共 160 KiB UB；该路径不分配普通路径的辅助区或初始化 Gather 索引。

每个缓冲槽由 MTE3_MTE2 事件保护重用，输入搬入结束由 MTE2_V 交给向量计算，V_MTE3 保护写回。等待当前输入后预取另一槽，计算和写回与下一次输入搬运重叠；退出前等待两个槽的写回结束。

展开 kernel 同时发布唯一需要的 CombA 指针表，按每核独占 8 个指针（64 字节）使用 DMA 写入，省去独立工作区初始化 kernel。标量循环只生成地址和任务元数据，数值变换使用向量指令。


## 工作区与执行顺序

工作区包含 11 个指针槽，分别定位 Ar、Ai、Br、Bi、T1..T4、CombA、CombB1、CombB2。每个槽按完整 batch 数计算，并对齐到 64 字节。

设单批填充后的 A/B 元素数为 aF/bF，输出实数元素数为 tF：

```text
单批工作区 float 数 = 4*aF + 6*bF + 4*tF
指针区字节数 = 11 * align64(batchCount * 8)
```

即使快速路径不使用所有区域，当前布局仍保留这些区域，便于统一指针与切片管理。直接输出路径从 CombA 起使用 4*aF 个 float，其后可复用容量为 2*aF+4*bF；m<=2*n 保证该容量足够。起点为单批工作区内 2*aF+2*bF+4*tF 个 float 的偏移。可继续降低总工作区占用，但不属于此次通过全部测试所必需的改动。

默认数据切片预算 1 GiB；用户提供工作区时以其实际大小限制切片。调用库的 EnsureDefaultWorkspace 管理增长，超出可用空间返回错误。不同切片复用同一工作区。

关键顺序保证：

- padding 清零使用 aclrtMemsetAsync，绑定 handle stream，防止与前一片未完成的 kernel 交叉。
- 工作区指针由设备 kernel 用 UB+64B DMA 写入；同一个 64B 块归一个核负责，避免多个核标量写 GM 指针造成可见性和缓存行问题。
- 输入完全覆盖填充区时省去清零，降低大矩阵 API 开销。
- 指针生产、预处理、Cube 和写回均在同一 stream 上排队；公共 API 不额外执行全设备同步。

任务书 3.4 未规定内存门限。原设计提及的主机 512MB 不是任务书的算子内存验收条件。报告分别给出测试进程 RSS、性能测试矩阵字节数和设备内存观测口径，避免混淆测试框架内存与算子工作区。


## 文件职责

| 文件 | 职责 |
| --- | --- |
| gemm_batched_host.cpp | 参数检查、路径选择、工作区、切片、kernel 调度 |
| gemm_batched_kernel.cpp | 工作区指针、通用预处理、Cube、通用组合和缩放 kernel |
| gemm_batched_kernel.h | Host 调用包装声明 |
| gemm_batched_tiling_data.h | Host/Device POD 参数、tile 常量 |
| gemm_batched_small.h | 小矩阵向量顺序归约与舍入补偿 |
| gemm_batched_dense.h | 对齐 NN 的连续预处理和写回，以及直接输出路径的双缓冲 A 展开与指针发布 |


## 支持硬件

| 产品 | 实测配置 | 支持情况 |
| --- | --- | --- |
| Atlas A2 训练/推理系列，含 Atlas 800I/T A2 | Ascend 910B3，64 GB HBM，CANN 9.1.0 | 支持 |
| Atlas A3 训练/推理系列，含 Atlas 800I/T A3 | Ascend910_9382，64 GB HBM，CANN 9.1.0 | 支持 |
| Ascend 950PR/950DT | 既有 arch35 实现 | 本次不修改、不外推验证结果 |

当前实机验证针对 DAV_2201 / arch22、CANN 9.1.0，不声明已在其他 CANN 版本完整验证。

# 可维可测分析

## 精度标准与测试覆盖

当前 arch22 测试对每个 batch 的整个 ldc×n 输出分离实部和虚部，分别调用 `applyMixedTolerance` 和 `Verifier::verifyVector`，使用 FLOAT32 混合精度策略：

- atol=2^-16，rtol=2^-10，匹配比例至少 0.99。
- 所有有限元素还须满足误差不超过 `max(0.01,32×该 golden 元素处的 ULP)`。
- Inf/NaN 按既有测试框架规则比较。CSV 中保留的 MERE/MARE 字段不参与这条 MIXED_TOLERANCE 路径。
- alpha=0 的 CPU golden 使用 beta*C 专用路径，仍采用上述混合精度判据，不描述为单独的位精确验收。
- 输出日志的 `x/N failures` 是超出逐元素 atol/rtol 的数量，不等于 GTest 失败数；最终仍须同时满足匹配比例和全部元素 ULP 上限。

golden 使用 Reference-LAPACK 3.12.0 参考 CBLAS/BLAS，逐批调用 cblas_cgemm。原始用例为 1000 条非 PF + 200 条 PF，再加 NullHandle，共 1201 个 GTest，不过滤 PF、不排除 TC_FL_142、不修改比较阈值。覆盖九种转置组合、尺寸边界、不同 alpha/beta、batch 扫描、leading dimension/padding、Inf/NaN、零维、k=0 和非法参数。

## 性能标准与实测结果

每组 NN，alpha=(1,0)，beta=(0,0)；预热 5 次，有效样本 60 次。完整功能回归结束后串行计时，区间为公共 API 到 stream 同步完成，分配、上传和 CPU golden 位于计时区间之外，不剔除慢样本。

| 规格 | A2 平均 us | A3 平均 us | 门限 us | 结果 |
| --- | ---: | ---: | ---: | --- |
| 256³ × 32 | 104.598 | 337.184 | 342.44 | 通过 |
| 512³ × 16 | 350.845 | 1079.343 | 1462.32 | 通过 |
| 1024³ × 8 | 1648.956 | 3843.946 | 5018.64 | 通过 |
| 2048³ × 4 | 5858.931 | 14702.019 | 18032.75 | 通过 |

任务书性能设备写明 A2 910B3；A3 也直接采用同样四项门限，不另套倍率。A3 最小规格余量约 1.53%，环境和并发负载变化后需重新测量。

同一 v028 源码及独立构建的动态库在 A2、A3 均完成 1201/1201 功能精度和 4/4 性能。每个平台使用 GTest 原生 16 分片，全部退出 0；完整 XML 汇总检查无缺漏、重复、失败、错误或跳过。PF GTest 整例时间不作为独立性能结果。

## 内存与资源

任务书 3.4 未规定内存门限。A2 最大单测试进程 RSS 1209040 KiB，A3 为 1229988 KiB，包含 CPU golden 和框架副本，不等于算子设备工作区。矩阵/指针分配字节、设备总已用内存和 RSS 分别记录，避免混用口径。

## 证据与复现

自测材料保存逐项参数 CSV、全部 1201 项 XML、16 分片元数据、240 个原始性能样本/平台、关键实虚精度日志与产物哈希。相关完整报告作为独立交付件，本文不使用不可访问的本地文件链接。

- CSV SHA256：`e1681879ae1a6d17fb7c104c05beb5b28141348247252bbb2f9074f7772030b0`。
- 双平台动态库 SHA256：`d4de6c85c1f2976ab6a8005ae9c14e16eb5e74f62aac4050e318621ae23c761d`。
- A2 完整 XML SHA256：`0f7346002775def9fe72775441f45e266d82184263237b625bd96636c5a67189`。
- A3 完整 XML SHA256：`e89381ac712c0383184db513ace21f2f029b4610317f07189371c0b1b340d06e`。

复现顺序：在目标 CANN 9.1.0 环境设置 SOC_VERSION（A2 ascend910b3，A3 ascend910_9382），构建 ops-blas 和 cgemm_batched_test；绑定冻结库与 Reference-LAPACK，运行完整 CSV；核对 1201 个唯一结果；所有分片结束后执行四组 60 次基准，独立重算均值并比较门限。异机重建需保留 CSV 与编译源码路径的关系，避免 __FILE__ 记录的旧路径导致用例未加载。

## 兼容性与维护

保持 `include/cann_ops_blas.h` 已有签名与列主序 BLAS 语义；arch22 代码仅在对应 SOC 构建中收集，既有 arch35 实现保持不变。当前说明反映 v028：通用两个 2K GEMM 与受保护的单 GEMM 直接输出路径。旧内部四 GEMM 备用分支、部分长函数和少量历史注释属于后续代码规范维护项；重排数值代码后应重新进行完整验证。

本文的“通过”指提供用例与门限的实机自测，不代表设计 PR 已获批准、实现代码已合入或官方平台验收已完成。
