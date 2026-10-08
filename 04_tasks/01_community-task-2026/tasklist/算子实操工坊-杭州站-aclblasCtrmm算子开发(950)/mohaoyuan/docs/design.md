# 需求背景(required)

## 需求来源

通过【算子实操工坊-杭州站】社区任务,向昇腾算子开源仓 ops-blas 贡献单精度复数三角矩阵-矩阵乘算子
`aclblasCtrmm`(Ascend 950PR / arch35)。任务书:《aclblasCtrmm 950 算子开发任务书》,
目标合入目录:代码 `blas/trmm/arch35/`,测试 `test/trmm/ctrmm/arch35/`。

## 背景介绍

### aclblasCtrmm 算子实现

ops-blas 仓已有实数版 `aclblasStrmm`(`blas/trmm/arch35/` 下 `strmm_host.cpp` /
`strmm_kernel.cpp` / `strmm_tiling_data.h` 三件套,三阶段 AIV/AIC 流水线),
公共头文件 `include/cann_ops_blas.h` 原先无 `aclblasCtrmm` 声明。
本任务在同一目录族新增 complex64 接口,参数序列与 cuBLAS `cublasCtrmm` 一一对应,
语义对齐 Netlib `ctrmm`(参考 https://www.netlib.org/blas/ctrmm.f)。

本算子不是 TBE 迁移,而是 ops-blas 句柄式 BLAS + Ascend C kernel 直调(arch35 / DAV_3510)。

### 标杆现状分析

| 参数 | 参数含义 | 数据类型 | 支持数据类型 | 约束 | 形状 |
| --- | --- | --- | --- | --- | --- |
| handle | 库句柄 | scalar | aclblasHandle_t | 非空,携带 stream | - |
| side | A 在左/右侧 | attr | LEFT / RIGHT | 非法返回 INVALID_VALUE | - |
| uplo | 引用上/下三角 | attr | UPPER / LOWER | 仅引用指定三角 | - |
| trans | op(A) 选择 | attr | N / T / C | C 为共轭转置 | - |
| diag | 对角类型 | attr | UNIT / NON_UNIT | UNIT 时对角视为 1+0i 且不读取 | - |
| m, n | B/C 行列数 | scalar | int | ≥0;0 为合法 no-op | - |
| alpha | 复数标量乘数 | COMPLEX64 | Host 或 Device 指针 | (0,0) 时 C 置零返回 | - |
| A | 三角矩阵 | COMPLEX64 | 列主序 ND | 仅引用 uplo 三角 | LEFT: m×m;RIGHT: n×n |
| lda | A 前导维度 | scalar | int | LEFT ≥max(1,m);RIGHT ≥max(1,n) | - |
| B | 输入矩阵 | COMPLEX64 | 列主序 ND | 只读 | m×n |
| ldb | B 前导维度 | scalar | int | ≥max(1,m) | - |
| C | 输出矩阵 | COMPLEX64 | 列主序 ND | 允许 C==B 原地 | m×n |
| ldc | C 前导维度 | scalar | int | ≥max(1,m) | - |

计算公式:

- LEFT:`C = alpha · op(A) · B`
- RIGHT:`C = alpha · B · op(A)`
- `op(A) ∈ {A, Aᵀ, Aᴴ}`

标杆(cuBLAS/Netlib)语义要点:三角矩阵-矩阵**乘法**(非 TRSM 求解类,对角元允许为 0,
无奇异性检测);cuBLAS 为离席实现(结果写 C,区别于 BLAS 原地覆写 B);alpha=(0,0) 时
A/B 不被引用、C 置零;m=0 或 n=0 直接返回成功。

# 需求分析(required)

## 需求描述

使用 Ascend C(SIMT 向量编程 + tensor_api 立方编程)在 Ascend 950PR(arch35/DAV_3510)
实现 `aclblasCtrmm`:COMPLEX64 全枚举组合(side×uplo×trans×diag = 2×2×3×2 = 24 组)、
列主序、离席计算、支持 C==B 原地,精度满足生态算子开源精度标准,性能不高于标杆耗时。

接口签名(新增于公共头文件 `include/cann_ops_blas.h`,紧随 `aclblasStrmm` 之后,
与 cuBLAS `cublasCtrmm` 逐参数对应):

```cpp
aclblasStatus_t aclblasCtrmm(
    aclblasHandle_t handle,
    aclblasSideMode_t side,
    aclblasFillMode_t uplo,
    aclblasOperation_t trans,
    aclblasDiagType_t diag,
    int m, int n,
    const aclblasComplex* alpha,
    const aclblasComplex* A, int lda,
    const aclblasComplex* B, int ldb,
    aclblasComplex* C, int ldc);
```

## 需求拆解

1. 公共头文件 `include/cann_ops_blas.h` 新增 `aclblasCtrmm` 声明(供产品线共用,禁止私有平行接口)
2. host 侧:参数校验(cuBLAS 口径返回码)、复数 alpha Host/Device 双位置解析、
   alpha=(0,0) 与 m/n=0 快路径、workspace 管理、kernel 编排
3. kernel 侧:复数三角矩阵稠密化(mirror)、复数实/虚平面分解(deinterleave)、
   复数矩阵乘(3M 方法分解为实数 GEMM)、复数标量合并输出(scale)
4. 测试工程:CSV 驱动 GTest(任务包 1200 条用例),golden 由 cblas_ctrmm 生成,
   复数 MERE/MARE 精度判定
5. 精度达标:rtol=2⁻¹⁰、atol=2⁻¹⁶、matched_ratio≥0.99、max_abs_error≤max(1e-2, 32ULP)
6. 性能达标:5 个标杆 case(256²→2048²)平均单次耗时不高于标杆值

# 详细设计(required)

## 算子分析

### 数学公式与复数分解

```
C = alpha · op(A) · B        (side = LEFT,  A 为 m×m)
C = alpha · B · op(A)        (side = RIGHT, A 为 n×n)
op(A) = A | Aᵀ | Aᴴ
```

复数乘法分解为 4 次实数 GEMM(4M,与 cblas golden 同构的误差轮廓):

```
平面:Ar = Re(op(A)), Ai = Im(op(A));  Br = Re(B), Bi = Im(B)

t1 = Ar·Br    t2 = Ai·Bi    t3 = Ar·Bi    t4 = Ai·Br
Cr = t1 − t2                         (展开:Ar·Br − Ai·Bi ✓)
Ci = t3 + t4                         (展开:Ar·Bi + Ai·Br ✓)
C  = (αr+iαi) · (Cr + i·Ci)
```

> 3M(Karatsuba)方案已实测评估并放弃:K ≥ 2048 时 P3−P1−P2 的消退使个别元素
> max_abs_error 微超 1e-2 生态上限(matchedRatio 0.9994 健康),而 4M 与参考
> 实现(cblas)的累加结构一致,精度完全达标。3M 保留为性能不达标时的备选路径
> (届时可配合分段累加控制误差)。

### 支持数据类型

COMPLEX64(float 实部 + float 虚部,`aclblasComplex`,实平面分解后内部按 FP32 计算)。

### 支持形状

m ≥ 0、n ≥ 0 任意形状(m/n 含 1、奇数、非 2 幂对齐值直至 4096 量级);
前导维支持紧凑(=最小约束)与 padding(> 最小约束)。

## 算子实现

### 实现方案

三段式复数化流水线(预处理 2 个 AIV kernel → 立方计算 4 个 AIC kernel → 后处理 1 个 AIV kernel,共 7 个,全部在同一 stream 顺序执行):

```
[K1] Mirror   (AIV·SIMT)  复数三角 A → 稠密实平面 Ar / Ai(列主序,步长 lda)
[K2] Deint    (AIV·SIMT)  复数 B(m×n,交织存储)→ 实平面 Br / Bi(步长 ldb)
[K3] GEMM ×4  (AIC·tensor_api)  t1=Ar·Br, t2=Ai·Bi, t3=Ar·Bi, t4=Ai·Br(复用 strmm 同构 GEMM 内核)
[K4] Scale    (AIV·SIMT)  C = α·((t1−t2) + i·(t3+t4)) 交织写回 C 活动区
```

workspace 布局(8 段,32B 对齐):`[Ar | Ai | Br | Bi | t1 | t2 | t3 | t4]`,
总大小 = 2·(lda·dimA) + 2·(ldb·n) + 4·(n·tempRowStride) 个 FP32 元素
(2048² 最坏情形约 128MB,设备内存充裕)。

#### host 侧设计

**(1) 参数校验顺序**(保证非法值在快路径之前被拦截,与 golden 同口径):

```
handle 非空(否则 HANDLE_IS_NULLPTR)
→ side/uplo/trans/diag 枚举合法(否则 INVALID_VALUE,cuBLAS 口径)
→ m ≥ 0、n ≥ 0(否则 INVALID_VALUE)
→ m==0 或 n==0:直接 SUCCESS(Netlib 快速返回,不触碰任何指针)
→ lda/ldb/ldc 下界 + alpha 非空(否则 INVALID_VALUE)
→ alpha 位置解析(见 (2))→ C 非空 → alpha==(0,0) 快路径(见 (3))
→ A、B 非空 → kernel 编排
```

**(2) 复数 alpha 双位置支持**:`CheckPtrLocation` 判定 Host/Device 指针;Device 时
8 字节 D2H 异步拷回 + stream 同步后取值(复数实/虚部都要参与零判断)。

**(3) alpha=(0,0) 快路径**:`aclrtMemsetAsync` 将 C 的 m×n 活动区置零——ldc==m 时整块
连续 memset,ldc>m 时逐列 memset 保留 padding 行原值(padding 行内容由调用方预置;
golden 侧同样只保证活动区逐元素为零,测试用例的 alpha=(0,0) 场景均采用紧凑 ldc)。

**(4) 列主序适配(swap trick,继承 strmm 已验证方案)**:AIC GEMM 内核按行主序工作,
host 传参时 m↔n 互换且 side 翻转,使内核实际计算行主序 Cᵀ;lda/ldb 不交换
(gmLeft 恒指向 A 平面用 lda、gmRight 恒指向 B 平面用 ldb)。mirror 产出的平面保持列主序,
恰好被行主序内核读作转置矩阵,无需额外搬运。

**(5) tiling 策略**(与 strmm 完全一致——实平面即普通 FP32 矩阵,无需为复数额外换算):

- Mirror/Deint/Scale(AIV):按行均分到向量核,`mirrorRowsPerCore = CeilDiv(行数, 核数)`,
  线程数 = clamp(CeilAlign(rowsPerCore, 128), 128, 2048)
- GEMM(AIC):tile 128×128、K-chunk 256 起步;L1(512KB)双缓冲约束下自动折半 K-chunk
  至满足 `2×(A侧 + B侧) ≤ L1`;tempRowStride = CeilAlign(m, 8)(fixpipe 对齐)
- 4 次 GEMM 形状完全一致 → tiling 仅计算一次,4 次调用只换平面与 temp 指针

#### kernel 侧设计

**(K1) Mirror 内核(AIV·SIMT,模板分发 uplo × transT × conj × diag,trans=N 时 conj 恒为 false,有效路径 12 组)**:

- 读取:列主序交织复数 `aGm[2·(col·lda+row)]` / `+1`
- 三角判定与转置索引与 strmm 逐行同构:`aGm[X·lda+Y] = A[Y][X]`;trans=T/C 时读
  `A[col][row]` 实现读取期转置
- **trans=C 共轭**:虚部取负(实数版没有的复数语义)
- diag=UNIT:对角写 (1, 0),不读取 A 对角位置
- 缺失三角写 (0, 0);两平面同址写:`planeAr=Re, planeAi=Im`

**(K2) Deint 内核(AIV·SIMT)**:B 交织复数拆为三平面,仅写 m 活动行
(GEMM 的 K 范围不会越界读 padding 行)。

**(K3) GEMM 内核(AIC·tensor_api,与 strmm 逐行同构)**:
GM→L1(`Te::CopyGM2L1`)→L0A/L0B(`Te::CopyL12L0A/L0B`)→`Te::Mmad`(L0C 累加,
首块/末块 unitFlag 控制)→L0C→GM(`Te::CopyL0C2GM`)。L1/L0 乒乓缓冲,
fixpipe 事件同步;tile 遍历按核数网格 stride 分配,奇数行 M 反向遍历 N 平衡访存。

**(K4) Scale 内核(AIV·SIMT)**:读 P1/P2/P3(行主序 Cᵀ 布局),
`Cr=P1−P2、Ci=P3−P1−P2`,与复数 alpha 相乘后交织写入列主序 C 的活动行;
padding 行不触碰(由上层 B→C 预拷贝语义保证);C==B 原地安全(读 temp 不读 B)。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Ascend 950PR(arch35 / DAV_3510) | √ |
| Ascend 950DT(arch35 / DAV_3510) | √ |
| Atlas A2 训练/推理系列(arch22) | 不支持 |
| Atlas A3 训练/推理系列 | 不支持 |

依赖 CANN asc-devkit ≥ 9.1(低于该版本编译与运行跳过)。

## 算子约束限制

- 不支持除 C==B 外的参数内存重叠(与 cuBLAS 口径一致)
- 不支持超出 lda/ldb/ldc 语义的非连续内存(本批次约束)
- diag=UNIT 时实现与 golden 均不读取 A 对角位置;alpha=(0,0) 时实现与 golden 均不引用 A/B
- 无 broadcast、无动态 shape 要求(m/n 为运行时入参)、无确定性计算要求

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | golden = Netlib cblas_ctrmm;实部/虚部分别按 FLOAT32 判定:逐元素 \|actual−golden\| ≤ 2⁻¹⁶ + 2⁻¹⁰·\|golden\|,matched_ratio ≥ 0.99,max_abs_error ≤ max(1e-2, 32ULP);测试工程口径 MERE < 2⁻¹³ 且 MARE ≤ 10× | 生态算子开源精度标准(https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md) |
| 性能标准 | 5 个标杆 case 平均单次耗时(warmup 后采样 >50 次取平均)不高于:256²L/U/N=120.18us;512²L/L/N=268.37us;1024²R/U/T=690.54us;2048²L/L/C=2945.91us;2048²R/L/N/UNIT=2963.64us | 任务书 §3.3(标杆耗时 = GPU 基线 ÷ 0.4,即 H100 实测 ×2.5 宽限) |

测试工程:`test/trmm/ctrmm/`(CSV 驱动 GTest,1200 条用例 = 1000 精度 + 200 性能),
覆盖 24 组枚举全组合、尺寸扫描(1→2048,含奇数/非对齐)、前导维 padding、
alpha 特殊值((0,0)/(1,0)/(−1,0)/纯虚/大值)、非方阵(fat/thin)、Inf/NaN 填充、
零维/空指针/非法枚举/非法前导维/负维度等边界负向用例;`verify_accuracy.py` /
`verify_performance.py` 一键复现,readme 说明步骤。

## 兼容性分析

新算子,不涉及兼容性分析;接口声明放入公共头文件 `include/cann_ops_blas.h`,
与其他产品线共用,不定义 950PR 私有平行接口。

## 风险与应对

| 风险 | 分析 | 应对 |
| --- | --- | --- |
| Inf/NaN 特殊值输入 | 复数乘法对 Inf/NaN 的传播路径与参考实现的运算顺序相关,极端填充用例可能出现 NaN 位模式或 Inf/NaN 分歧 | 4M 与 cblas 参考实现同构,Inf 传播天然对齐;测试侧对 NaN 位模式做规整化(±NaN 视为相等,IEEE 语义);确有分歧的极端场景按仓内测试规范过滤并说明口径 |
| 2048² 性能贴线 | 4M 立方量为 3M 的 1.33 倍,标杆余量取决于 950PR FP32 MMAD 持续吞吐 | tile/K-chunk 调优;AIC 核内对象复用;必要时启用 3M(Karatsuba)+ 分段累加控误差的备选路径 |
