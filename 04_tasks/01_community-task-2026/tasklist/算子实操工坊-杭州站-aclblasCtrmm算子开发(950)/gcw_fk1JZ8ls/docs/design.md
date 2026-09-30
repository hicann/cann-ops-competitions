# 需求背景（required）

## 需求来源

昇腾 CANN 社区任务（2026）：在 ops-blas 开源仓（https://gitcode.com/cann/ops-blas）为 Ascend 950PR（arch35）补齐单精度复数三角矩阵-矩阵乘算子 `aclblasCtrmm`，以 Ascend C 编程语言开发，验收后合入仓库（算子目录 `blas/trmm/arch35/`，测试目录 `test/trmm/ctrmm/arch35/`）。

## 背景介绍

### aclblasCtrmm 算子实现补齐

基于 ops-blas 仓已有 BLAS 工程框架与同族实数算子 `aclblasStrmm` 的设计范式，使用 Ascend C 编程语言实现 `aclblasCtrmm` 句柄式 BLAS 接口。

参考实现与相关路径：

- 同族参考算子（仓内实数版 TRMM）：`blas/trmm/`（strmm 系）及其测试工程 `test/trmm/strmm/`
- 对标基线接口：cuBLAS `cublasCtrmm`（离席语义）；golden 参考实现：Netlib BLAS `ctrmm`（https://www.netlib.org/blas/ctrmm.f ，仓内测试工程以 cblas_ctrmm 单标杆生成 golden）
- 接口声明：`include/cann_ops_blas.h`（此前无 `aclblasCtrmm` 声明，仅有实数版 `aclblasStrmm`，本任务新增声明，与其他产品线共用）

### 仓内现状分析

通过对 ops-blas 仓的功能分析，TRMM 族当前支持能力如下：

| 参数 | 参数含义 | 数据类型 | 仓内支持情况 | 约束 | 形状 |
| --- | --- | --- | --- | --- | --- |
| side | A 矩阵位置 | 枚举 | 仅实数 strmm 支持 | LEFT/RIGHT | - |
| uplo | 三角存储模式 | 枚举 | 仅实数 strmm 支持 | UPPER/LOWER | - |
| trans | op(A) 选择 | 枚举 | 仅实数 strmm 支持 | N/T/C | - |
| diag | 对角类型 | 枚举 | 仅实数 strmm 支持 | NON_UNIT/UNIT | - |
| A/B/C | 三角阵与矩阵 | tensor | 仅 float32（strmm） | 列主序 | A 为方阵，B/C 为 m×n |
| alpha | 复数标量 | scalar | strmm 为实数 | Host/Device 内存 | - |

**complex64（单精度复数）TRMM 在仓内完全缺失**。复数引入的核心增量难点：①复数乘加需实/虚分解（4 次实乘），长内积（A 的阶数最大 4096）下 fp32 部分和舍入游走必须控制在容差内；②大形状（2048² 档）纯矢量路径实测持续吞吐 3.76 TFLOPS，而门槛需求 11.7 TFLOPS 以上，架构性不可达，必须引入 Cube（矩阵核）路径。

### aclblasCtrmm 算子功能分析

计算公式（列主序，离席写入 C）：

- `side = ACLBLAS_SIDE_LEFT` 时：`C = alpha * op(A) * B`
- `side = ACLBLAS_SIDE_RIGHT` 时：`C = alpha * B * op(A)`
- `op(A)`：N → A，T → A^T，C → A^H（共轭转置）

输入：handle、side/uplo/trans/diag 枚举、m/n、复数标量 alpha（Host 或 Device 内存）、三角矩阵 A（仅 uplo 指定三角被引用；diag=UNIT 时对角视为 1 且不读取）、B（m×n）、前导维 lda/ldb/ldc。

输出：C（m×n，离席；允许 C 与 B 传同一指针实现原地计算，除此之外不支持参数重叠）。

支持数据类型：COMPLEX64（实部/虚部各 float32）。

支持广播：不涉及（A/B/C 为独立矩阵）。

特殊语义：m=0 或 n=0 为合法 no-op；alpha=(0,0) 时 A 不被引用、C 置零返回；本算子为乘法（非求解类），对角元允许为零。

# 需求分析（required）

## 需求描述

使用 Ascend C 编程语言实现 `aclblasCtrmm` 算子：complex64 数据类型，side×uplo×trans×diag 共 24 组枚举全组合，任意 m/n（0~4096，含奇数/非对齐）、紧凑与 padding 前导维、B==C 原地、Host/Device alpha、alpha=(0,0) 置零、零维 no-op 及全量非法参数校验（INVALID_VALUE 口径对齐 cuBLAS）。精度满足生态算子开源精度标准（COMPLEX64 实部/虚部分别按 FLOAT32 判定）；性能满足任务书 5 硬门槛（Avg time 直比）与参考用例 0.4 倍率判据。

## 需求拆解

1. 支持 complex64 单一数据类型、24 组枚举全组合、任意前导维与零维/置零/原地等边界语义
2. 精度满足生态开源精度标准（rtol 2^-10、atol 2^-16、matched_ratio ≥ 0.99、max_abs_error ≤ 1e-2 或 32 ULP）
3. 性能满足 5 硬门槛（256²/512²/1024²/2048²/2048² 档 avg time ≤ 120.18/268.37/690.54/2945.91/2963.64 us）与 195 条参考用例（NPU ≤ GPU 耗时/0.4）
4. 大形状走 Cube（矩阵核）路径：复数 GEMM 分解为 4 次 fp32 实 GEMM，以 SIMD+Cube 混合架构逼近门槛；纯矢量路径保留为小形状与精度兜底

# 详细设计（required）

## 算子分析

### 数学公式

- LEFT：`C(i,j) = alpha * Σ_k op(A)(i,k)·B(k,j)`，k 遍历 A 的有效三角带
- RIGHT：`C(i,j) = alpha * Σ_k B(i,k)·op(A)(k,j)`
- 复数乘加按实/虚分解：`(a+ai·i)(b+bi·i) = (a·b − ai·bi) + (a·bi + ai·b)·i`，即每个复数 MAC = 4 次实乘

### 支持数据类型

COMPLEX64（float32 实部 + float32 虚部）

### 支持形状

A 为 max(1,q)×max(1,q)（q = m 或 n），B/C 为 m×n，列主序，lda/ldb/ldc 支持紧凑与任意 padding；不支持除 B==C 外的参数重叠。

## 算子实现

### 实现方案

#### host侧设计：

**路径分派（tiling 策略总纲）**：host 侧依据形状与标量做双路径分派——

- 矢量路径（AIV）：`macElems = m·n·(q+1)/2 < 2M` 或 q < 64 或 alpha 非有限时启用（3 次 kernel 发射开销占优的小形状与数值兜底场景）；
- Cube 路径（精度包络内启用）：`alpha 有限 && |αr|+|αi| ≤ 1 && q ≤ 2048 && macElems ≥ 2M && 存在 AIC 核`。包络是实测的 fp32 精度能力边界：q > 2048 时 fp32 长链部分和游走超出 1e-2 绝对元素上限，|α| > 1 时绝对误差随 alpha 放大——包络外形状走矢量路径（全量精度全绿），不做容差放宽。

tiling 参数（Cube 路径）：

- 分段分裂累加器数 SP：按 q 分段（q≤512→1，≤1024→2，≤1536→4，其余→7）。机理：fp32 顺序累加的舍入随机游走误差按 √(链长) 增长，将 k 链按块奇偶分配到 SP 个独立 L0C 累加器再合并，误差按 √SP 缩减（实测 q=2048 档误差 ~2e-2 → ~7e-3，进入 1e-2 容差内）；
- K 深 chunk：chunk = q/SP（对齐到 8），受 L1 双缓冲预算（512KB×9/10）约束；
- 瓦片与分核：内层瓦片 64(gemmM)×64(gemmN)（MMAD 16 分形的整数倍）；核块 singleCoreM/N = 64（每核块恰一瓦片，块数 = divM×divN 为数百~上千的细粒度网格，消除粗块尾核空闲——该项为性能收官关键，2048² 档 −14%）；
- 工作区布局（handle 工作区，32B 对齐）：`[B==C 快照(可选)][非有限旗标 64×32B][Ar|Ai|An 三张 A' 平面 qPad×qPad][Br|Bi 平面 nPad×bpLd][Cr|Ci 各 SP 张部分平面 nPad×crStride]`，其中 An=−Ai 使 GEMM 各段全加号累加，qPad/bpLd/crStride/nPad 均 pad16（MMAD 分形对齐，尾分形越界落入零垫区）。

##### 1. 分核策略：

- 矢量路径：输出瓦片 64×32，P = min(瓦片总数, AIV 核数)，瓦片跨核轮转（t = blockIdx; t += numBlocks）；
- K1 预处理（AIV）：列任务 q+n 个跨核轮转，每核独占一 32B 旗标槽（单写者）；
- K2 GEMM（AIC）：核块 (r, N) 按扁平索引 tileIdx += blockNum 轮转 + 奇数 M 行 N 序蛇形反转（L2 面板局部性实测最优，三种替代调度方案均为负结果）；
- K3 合并（AIV）：输出瓦片跨核轮转，瓦片独占无跨核同步。

##### 2. 数据分块和内存优化策略：

- UB 预算：矢量路径 244,768B（X/YS/ACC/WB/SCR/YBC/AR/AI/C0/DGR/DGI/EDG 分区，边缘表 16384 项/AIV）；K1 约 28KB（4096B 头防护 + 双列窗口 + 4 组列暂存平面）；K3 借用边缘表尾部 64KB 作 8 平面暂存（立方模式边缘表上限降为 5000 项，溢出走内联 RMW 路径）；
- L1/L0：K2 四面板（两段×A/B）双缓冲驻留 L1；L0 装载按半 chunk 大块搬运（E1'，仅双轴 64 对齐的无尾分形网格启用，带尾网格回退逐块路径——带尾分形与 L0 大块装载的交互曾触发 AIC 异常，已实测定因并门控）；
- 批量搬运：K1 单列窗口一次 MTE2 取整列（chunk=1024 复数行）；K3 每张部分平面一次 2D DataCopyPad（nt_×mt_ 块、crStride 行距）替代逐列 256B 小搬运——该项为 K3 最大单项收益（−60%）。

##### 3. tilingkey规划策略：

本算子为句柄式 BLAS 单 kernel 组，无需 TilingKey 模板分支；路径分派（矢量/Cube、E1' 门控 l0StageMode、分裂数 SP）全部由 host 计算后经 POD tiling 结构按值传入 kernel（CtrmmTilingData / CtrmmCubePreTiling / CtrmmCubeGemmTiling 三级），kernel 侧无运行时分支推导。

数据检测：alpha 经 CheckPtrLocation 判定 Host/Device（Device 时 8B D2H 同步读取）；A/B 全量数据由 K1 设备侧逐块扫描非有限值（mul-by-0 自等 trick），任一命中则 K3 整调用回退矢量 netlib 慢路径（非有限语义按 Annex-G 保持）。

#### kernel侧设计：

矢量路径与 Cube 路径均按 Init → Process 两阶段组织；Cube 路径 Process 为三核流水线（K1 搬入/变换 → K2 矩阵计算 → K3 合并/搬出），流式串行发射到绑定 stream。

**矢量路径（AIV RegBase，小形状与兜底）**：

1. 逐瓦片 OP4 4 列组 VF 累加（k 循环驻留 VF 寄存器，X 平面按三角裁剪预清零，Y 段 alpha 折叠位精确对齐 netlib 舍入序）；
2. 两阶段 MTE 存储契约：中段（32B 对齐部分）MTE3 整段搬出，边缘元素栅栏后整 32B 块 RMW 写（有界校验重试，杜绝标量 GM 写的 DCache 丢失类缺陷）。

**Cube 路径（K1/K2/K3，大形状主力）**：

1. **K1 预处理（AIV）**：一趟读 A/B，RMW 窗口范式（整块 MTE2 取入 → UB 内实部对齐/无效三角清零/UNIT 对角注入/OP_C 共轭 → VF 位置解交织）产出 5 张 fp32 实平面（Ar/Ai/An/Br/Bi），垫列清零；逐块非有限扫描并写核旗标；
2. **K2 GEMM（AIC）**：复数乘加分解为 4 次 fp32 实 GEMM（Cr=(Br·Ar)+(Bi·An)，Ci=(Bi·Ar)+(Br·Ai)，te:: 张量 API），输出写为转置平面 Cr/Ci（n×m 行主序，K3 读列即连续）；逐瓦片三角带裁剪（跳过带外 k 块：跳过的是 exact ±0 加法，数值中性）；k 链按块奇偶分配到 SP 个 L0C 累加器（交替段配对累加保持部分和平衡，it3c2 精度修复），每累加器独立 fixpipe 到各自部分平面；带内块数不足 SP 的瓦片仅写已用平面；
3. **K3 合并（AIV）**：核旗标全查——任一命中则整调用矢量 netlib 慢路径重算（非有限语义保持）；否则每输出瓦片按 2D 搬运批量读入该瓦片的 SP 张 Cr/SP 张 Ci 部分平面，按原累加序 VF 逐平面求和（零播种，0+x≡x 数值不变），合并值单次非有限检查（Inf/NaN 吸收性），复数 alpha 缩放后走矢量路径同款两阶段 MTE 存储契约写出 C。

aclblasCtrmm 的 Cube 路径流程：

```
host: 校验→alpha 解析→路径分派(包络)→tiling→工作区布局
  └─ K1(AIV): A,B ──RMW窗口+VF解交织──> Ar,Ai,An,Br,Bi 平面 + 非有限旗标
  └─ K2(AIC): 5 平面 ──三角带内 4×fp32 GEMM(奇偶分裂SP累加器)──> Cr/Ci × SP 部分平面
  └─ K3(AIV): 旗标检查→SP 平面 2D 批量读+VF 合并→alpha 缩放→两阶段 MTE 写 C
              (旗标命中 → 矢量 netlib 慢路径整调用重算)
```

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas 950PR（Ascend 950PR，arch35） | √ |

## 算子约束限制

- 数据类型仅 COMPLEX64；不支持除 B==C（原地）外的任何参数重叠；
- Cube 路径精度包络：alpha 有限且 |αr|+|αi| ≤ 1 且 q ≤ 2048（包络外自动走矢量路径，精度不受影响，性能为矢量路径水平）；
- 不支持超出 lda/ldb/ldc 语义的非连续内存访问；不涉及广播；m/n 为运行时入参，不支持动态 shape 特化；
- 异步执行：依赖 aclblasSetStream 绑定 stream，读回 Device 结果前须同步 stream。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述(不涉及说明原因) | 标准来源 |
| --- | --- | --- |
| 精度标准 | COMPLEX64 实部/虚部分别按 FLOAT32 判定：rtol 2^-10、atol 2^-16、matched_ratio ≥ 0.99、max_abs_error ≤ max(1e-2, 32 ULP)；golden 为 Netlib cblas_ctrmm 单标杆。实测：全量 1359/1359 PASS（1200 条 CSV 驱动 + 159 条代码内补充）、白盒 82/82 PASS、官方 verify_accuracy 脚本 1159/1159 ALL PASS | 生态算子开源精度标准（https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md） |
| 性能标准 | 5 硬门槛（warmup 10 + 采样 100 设备事件平均）全部达标：256²=62.9/120.18、512²=138.3/268.37、1024²=516.3/690.54、2048²(LEFT/L/C)=2836.3/2945.91、2048²(RIGHT/L/N)=2816.0/2963.64 us；200 条参考用例 83 条达 0.4 倍率（其余 117 条为小形状 launch 地板与包络外矢量天花板所限，已逐例定性与留证）。相对社区任务起点：1004/1005 档分别优化 −80.7%/−81.1% | 任务书 §3.3 性能要求与 gpu_baseline.csv（H100 基线，倍率 0.4） |

## 兼容性分析

新算子（仓内原无 `aclblasCtrmm`），仅在 `include/cann_ops_blas.h` 新增声明供各产品线共用，不修改任何既有接口与算子行为，不涉及兼容性分析。
