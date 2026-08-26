# 矩阵乘系列算子 arch22 设计文档

# 需求背景（required）

## 需求来源

本文对应 2026 年 8 月社区任务“矩阵乘系列算子开发”，设计范围为
`chemm`、`cher2k`、`cherk`、`csymm`、`csyrk` 五个 BLAS Level-3
接口。目标代码仓为 `cann/ops-blas`，目标硬件为 Atlas A2 训练系列产品，
实现架构为 `arch22`，开发和验证基于 CANN 9.0.0。

需求依据按以下顺序执行：任务书正文及随附测试数据、`ops-blas` 仓库规范、
开发工具建议。任务书测试 CSV 包含零维和 `k=0` 用例；这部分作为标准 BLAS
兼容扩展纳入设计，不改变任务书对正维输入的主约束。

## 背景介绍

五个接口均采用 Column-Major `complex<float>` 矩阵，但结构约束不同：
CHEMM/CHERK/CHER2K 使用 Hermitian 语义，CSYMM/CSYRK 使用复对称语义；
矩阵乘接口需要支持左右乘，秩更新接口需要区分普通转置和共轭转置。

直接使用 AIV 完成大矩阵复数乘加无法满足性能要求。本设计将复数输入拆成
FP32 实部、虚部平面，由 AIC `MatmulImpl` 承担主计算，AIV 完成拆分、
转置/共轭、结构恢复和结果合并。小矩阵保留直接计算路径，避免固定流水的
启动成本。

# 需求分析（required）

## 需求描述

### 运算语义

| 接口 | 运算定义 | 模式 |
| --- | --- | --- |
| CHEMM | `C=alpha*A*B+beta*C` 或 `C=alpha*B*A+beta*C`，`A=A^H` | `side=L/R`，`uplo=U/L` |
| CHER2K | `C=alpha*A*B^H+conj(alpha)*B*A^H+beta*C` 及共轭转置形式 | `trans=N/C`，`uplo=U/L` |
| CHERK | `C=alpha*A*A^H+beta*C` 或 `C=alpha*A^H*A+beta*C` | `trans=N/C`，`uplo=U/L` |
| CSYMM | `C=alpha*A*B+beta*C` 或 `C=alpha*B*A+beta*C`，`A=A^T` | `side=L/R`，`uplo=U/L` |
| CSYRK | `C=alpha*A*A^T+beta*C` 或 `C=alpha*A^T*A+beta*C` | `trans=N/T`，`uplo=U/L` |

CHEMM、CSYMM、CSYRK 的 `alpha/beta` 为复数；CHERK 的两项标量均为
实数；CHER2K 的 `alpha` 为复数、`beta` 为实数。HERK/HER2K 只更新
目标三角，并将输出对角虚部写为 `+0.0f`。

### 公共接口

公开声明放在 `include/cann_ops_blas.h`，接口保持 C ABI：

```cpp
aclblasStatus_t aclblasChemm(
    aclblasHandle_t handle, aclblasSideMode_t side, aclblasFillMode_t uplo,
    int m, int n, const aclblasComplex* alpha,
    const aclblasComplex* A, int lda, const aclblasComplex* B, int ldb,
    const aclblasComplex* beta, aclblasComplex* C, int ldc);

aclblasStatus_t aclblasCher2k(
    aclblasHandle_t handle, aclblasFillMode_t uplo, aclblasOperation_t trans,
    int n, int k, const aclblasComplex* alpha,
    const aclblasComplex* A, int lda, const aclblasComplex* B, int ldb,
    const float* beta, aclblasComplex* C, int ldc);

aclblasStatus_t aclblasCherk(
    aclblasHandle_t handle, aclblasFillMode_t uplo, aclblasOperation_t trans,
    int n, int k, const float* alpha,
    const aclblasComplex* A, int lda,
    const float* beta, aclblasComplex* C, int ldc);

aclblasStatus_t aclblasCsymm(
    aclblasHandle_t handle, aclblasSideMode_t side, aclblasFillMode_t uplo,
    int m, int n, const aclblasComplex* alpha,
    const aclblasComplex* A, int lda, const aclblasComplex* B, int ldb,
    const aclblasComplex* beta, aclblasComplex* C, int ldc);

aclblasStatus_t aclblasCsyrk(
    aclblasHandle_t handle, aclblasFillMode_t uplo, aclblasOperation_t trans,
    int n, int k, const aclblasComplex* alpha,
    const aclblasComplex* A, int lda,
    const aclblasComplex* beta, aclblasComplex* C, int ldc);
```

标量指针和矩阵指针均指向 Device 内存。Host 侧在 handle 对应 stream 上
异步回读标量并同步，用于选择 `alpha=0`、`beta=0/1` 等分支；矩阵数据
不回传 Host。

### 参数约束

| 接口 | 合法枚举 | leading dimension |
| --- | --- | --- |
| CHEMM/CSYMM | `side=L/R`，`uplo=U/L` | `lda>=max(1, side==L?m:n)`；`ldb,ldc>=max(1,m)` |
| CHER2K | `uplo=U/L`，`trans=N/C` | `lda,ldb>=max(1, trans==N?n:k)`；`ldc>=max(1,n)` |
| CHERK | `uplo=U/L`，`trans=N/C` | `lda>=max(1, trans==N?n:k)`；`ldc>=max(1,n)` |
| CSYRK | `uplo=U/L`，`trans=N/T` | `lda>=max(1, trans==N?n:k)`；`ldc>=max(1,n)` |

空 handle 返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`。非法枚举、负维度、
过小 leading dimension 或计算所需的空指针返回
`ACLBLAS_STATUS_INVALID_VALUE`。`m=0` 或 `n=0` 直接成功返回；秩更新中
`k=0` 不读取 A，只对目标三角执行 `beta*C`。

## 需求拆解

1. 增加五个公开 ABI，并按接口独立完成枚举、维度、指针和前导维校验。
2. 在 `arch22` 下设计 Host 调度和 Ascend C Kernel，不影响其他架构路径。
3. 提供复数拆平面、转置/共轭、Hermitian/对称恢复、FP32 GEMM 和结果合并能力。
4. 对 1024、2048 基准形状设置专用 GEMM 分支，对奇数、非方、padding 和大 K
   形状保留统一回退路径。
5. 保护非目标三角，处理 Hermitian 对角、特殊标量、NaN/Inf 和零维语义。
6. 为五个算子分别设置构建、功能精度、任务书性能和泛化稳定性门禁。

# 详细设计（required）

## 公共架构

### 设计边界

公共层负责可复用的复数数据变换、实数 GEMM 和结果合并。接口特有的数学
定义、参数限制及调度条件仍放在各算子 Host 文件中，避免将 CHER2K 的双项
更新或 CSYRK 的非共轭语义隐藏在通用入口中。

整体流水如下：

```text
Device complex64 input
        |
        v
AIV: deinterleave / transpose / conjugate / symmetric expansion
        |
        v
AIC: 3 x FP32 MatmulImpl (Gauss 3M)
        |
        v
AIV: reconstruct complex result / alpha,beta / triangle mask
        |
        v
Device complex64 output
```

所有 Kernel 按顺序提交到同一 handle stream。中间平面允许原地复用的前提
是前序消费者已在该 stream 中排在复用写入之前，不引入跨 stream 隐式依赖。

### 复数平面与 3M 分解

输入 `X=Xr+iXi`、`Y=Yr+iYi` 拆成连续 FP32 平面。一般复乘采用：

```text
P1 = Xr * Yr
P2 = Xi * Yi
P3 = (Xr + Xi) * (Yr + Yi)
Re(XY) = P1 - P2
Im(XY) = P3 - P1 - P2
```

Hermitian 路径将右操作数虚部符号折叠到拆分或求和平面中。合并阶段根据
实际平面方向恢复符号，不依赖上层对输出做二次共轭。CHER2K 先形成
`alpha*A*B^H`，再利用其共轭转置得到第二项。

该方案把一般复数矩阵乘从四次实 GEMM 降为三次。AIV 的逐元素加减和
交错写回相对 AIC 主计算开销较小，同时仍保留 FP32 累加路径。

### 数据变换

复数输入在 GM 中按 `[real, imag]` 交错存储。AIV 拆分 Kernel 使用
`vreducev2` 分离实部、虚部；需要转置时，由平面转置 Kernel 生成逻辑
Row-Major GEMM 输入。共轭只改变虚部符号。

CHEMM/CSYMM 的结构矩阵只允许读取 `uplo` 指定三角。设计先拆分存储三角，
再恢复完整平面：

- 非对角块按 128×128 tile 镜像；
- 对齐形状的对角块拆成 16×16 micro-tile，减少单核串行工作；
- 非 128 对齐形状先处理非对角块，再启动 diagonal-row Kernel 修复对角块，
  由 Kernel 边界形成全局顺序；
- Hermitian 镜像时虚部取反且对角虚部清零；对称镜像仅交换行列；
- `order<=128` 的对角修复使用一个 AIV，避免任务数小于核数时漏行。

所有物理输入地址均按 Column-Major `col*ld+row` 计算，padding 区不参与
逻辑矩阵计算。

### GEMM 分块与路由

内部实 GEMM 基本块为 `128×256×64`，静态配置采用 CANN 9.0
`MatmulImpl` 的 `CONFIG_NORM`。Cube 显式执行
`SetHF32Mode(HF32Mode::DISABLE)`，保证任务书 FP32 精度门限。

路由顺序设计为：

1. `m/n` 均为 256 对齐且 `k=1024`：使用字面量 `256×256×1024` 单块形状；
2. `m/n` 均为 256 对齐且 `k=2048`：使用字面量 `256×256×2048` 单块形状；
3. 大输出且 `k<=1024/2048`：使用对应静态桶的 256×256 输出 tile，尾块通过
   `SetSingleShape(rowCount,colCount,k)` 处理；
4. 其他形状按最大维度选择 1024、8192 或 16384 静态桶，并使用二维 AIC 分区。

二维分区枚举 `mParts*nParts=usedCoreNum` 的因子组合，以 padding 后的总计算
面积最小为主要目标，同面积时优先增加 N 向分区。CHERK/CSYRK 的输出块
网格超过 4×4 时，仅调度目标三角对应块；4×4 及以下保留全网格，以免减少
活跃 AIC 数量。边缘 tile 仍在合并阶段执行逐元素三角掩码。

### Workspace 设计

所有分区按 512 B 对齐，尺寸使用 `size_t` 计算。handle 初始库管理空间为
32 MiB，不足时在同步当前 stream 后扩容；用户通过 `aclblasSetWorkspace`
提供空间时不执行内部扩容。库管理 workspace 硬上限为 8 GiB。

记 `P=align512(n*k*sizeof(float))`，`R=align512(n*n*sizeof(float))`，
`A=align512(order*order*sizeof(float))`，`O=align512(m*n*sizeof(float))`：

| 算子 | 通用路径 workspace | 用途 |
| --- | --- | --- |
| CHEMM/CSYMM | 常规 `3A+6O`；超过 2 GiB 优选阈值时 `2A+5O` | A 实/虚/和、B 实/虚/和、三个结果 |
| CHER2K | `4P+2*max(P,R)+2R` | A/B 逻辑平面、复用平面、四个结果/转置结果 |
| CHERK | `4P+4R` | A 与其转置的四平面、四个结果平面 |
| CSYRK | `4P+4R` | A 与其转置的四平面、四个结果平面 |

第四结果平面在部分 3M 路径中仅用于统一合并 ABI、跳过乘积清零或
CHER2K 的共轭转置项。大 workspace 用例不得降级为 Host 计算。

### 小矩阵与特殊标量

矩阵乘两外维和压缩维均不超过 32 的有效计算进入单 AIV 直接路径，直接按
结构读取输入并执行复数累加。该路径避免拆分、三次 GEMM 和合并共多个
launch，同时便于保持极值输入的有限值、Inf、NaN 分类。

`alpha=0` 或 `k=0` 时跳过主乘积。若 CHERK 同时满足 `beta=1`，C 无需
写回；其他接口仍按目标区域执行 beta 缩放。`beta=0` 与极值输入作为独立
精度用例，不能用普通有限值误差比较替代数值分类检查。

### 公共实现落点

下表用于设计评审时核对实现边界，不替代各算子方案描述：

| 设计模块 | 源码落点 |
| --- | --- |
| 复数 Host 公共工具、GEMM tiling | `blas/common/helper/complex_level3_host.h` |
| CHEMM/CSYMM 公共调度 | `blas/common/helper/complex_symm_host.h::RunComplexSymm` |
| 拆分、转置、镜像、求和、合并、小矩阵 Kernel | `blas/common/arch22/complex_level3_kernel.cpp` |
| 公共 tiling 数据 | `blas/common/arch22/complex_level3_kernel.h` |
| AIC FP32 GEMM 路由 | `blas/gemm/arch22/gemm_kernel.cpp::gemm_kernel_do` |
| workspace 生命周期和上限 | `blas/common/helper/aclblas_handle_internal.h` |

## CHEMM 设计

### 接口契约与数学定义

CHEMM 的结构矩阵 `A` 为 Hermitian。`side=L` 时 `order=m`，计算
`C=alpha*A*B+beta*C`；`side=R` 时 `order=n`，计算
`C=alpha*B*A+beta*C`。`A` 只读取指定三角，未存储三角中的数据即使为
NaN 也不得影响输出。A 的逻辑对角只取实部。

参数校验覆盖 `side`、`uplo`、`m/n`、三组 leading dimension 及必要
指针。`m=0` 或 `n=0` 直接返回。`alpha/beta` 按两个 FP32 组成的复数从
Device 回读。

### 数据流与分块

1. 拆分 A 的存储区域为 `Ar/Ai`，按 `uplo` 恢复完整 Hermitian 平面。
2. 拆分 B 为 `Br/Bi`；常规 workspace 同时产生 `Bs=Br+Bi`，紧凑模式
   在 GEMM 消费原平面后原地生成和。
3. 左乘以 `B*A` 的内部 Row-Major 方向提交三次 GEMM，右乘以 `A*B`
   的内部方向提交，二者最终都映射回公开的 Column-Major C。
4. 合并三项结果并计算复数 `alpha*product+beta*C`，写完整 `m×n` 输出。

GEMM 逻辑形状统一构造成 `(n,m,order)`，左右乘通过交换输入平面选择实现，
无需复制输出矩阵。`order=1024/2048` 且输出对齐时进入固定 K 路径；奇数
`m/n/order` 进入静态桶尾块路径。

### Workspace 与边界处理

常规布局为 `Ar,Ai,As,Br,Bi,Bs,P1,P2,P3`，即 `3A+6O`。超过 2 GiB
优选阈值后，`As` 复用 `Ar`，`Bs` 复用 `Br`，布局缩减为 `2A+5O`。
当 `alpha=0+0i` 时不拆分 A/B，三个结果平面清零后只执行 beta 合并。

小矩阵使用 `ComplexDirectSymmTiling`，每次访问 A 时现场判断元素位于存储
三角还是镜像三角；镜像元素虚部取反，对角虚部置零。

### 精度与性能关注点

- 结构恢复必须覆盖 UPPER/LOWER、LEFT/RIGHT 和非 128 对齐 order；
- padding 的 `lda/ldb/ldc` 不得被当作逻辑列长；
- 全矩阵 C 均参与误差比较，无非目标三角豁免；
- 性能重点为 `1024/2048` 的 L/U 与 R/L 四个任务书基准，同时覆盖
  cross、padded、极宽/极高矩阵的 NPU 稳定性。

实现落点：公开入口位于 `blas/hemm/arch22/chemm_host.cpp`，以
`hermitian=true` 调用 `RunComplexSymm`；镜像由
`complex_symmetric_mirror_do` 完成。

## CHER2K 设计

### 接口契约与数学定义

`trans=N` 时：

```text
C = alpha*A*B^H + conj(alpha)*B*A^H + beta*C
```

`trans=C` 时：

```text
C = alpha*A^H*B + conj(alpha)*B^H*A + beta*C
```

`alpha` 为复数，`beta` 为实数。A、B 在 `trans=N` 时为 `n×k`，在
`trans=C` 时为 `k×n`。只更新 `uplo` 指定三角，并强制 C 对角虚部为零。

参数校验仅接受 `N/C`，A、B 分别校验 `lda/ldb`。`n=0` 快速返回；
`k=0` 或 `alpha=0` 时不读取 A/B，但仍按实数 beta 更新目标三角。

### 数据流与分块

1. 根据 `trans` 选择物理拆分方向。`N` 路径转置 A 平面，B 的物理
   Column-Major 排布直接作为右操作数；`C` 路径对 A 拆分时取共轭，
   并对 B 平面执行一次转置。
2. 计算 `P1=Ar*Br^T` 和 `P2=Ai*Bi^T`。
3. 在同一 stream 上将输入平面原地改写为 `Ar+Ai` 与 `Br^T-Bi^T`，计算 P3。
4. 由 P1/P2/P3 恢复第一项复数结果并乘 alpha。
5. 对第一项执行平面转置，生成其共轭对应项；合并两项和 `beta*C`。
6. 合并 Kernel 仅遍历目标三角，对角虚部固定写 `+0.0f`。

CHER2K 不采用 CHERK 的三角 GEMM 裁剪：第二项来自第一项的结构转置，
需要保留形成目标三角所需的完整中间结果。GEMM 为 `n×n×k`，共享公共
1024/2048 固定 K 路由和通用尾块路由。

### Workspace 与边界处理

布局包含四个 `P` 大小的逻辑输入平面、两个 `max(P,R)` 可复用平面和
两个独立 `R` 平面，总量为 `4P+2*max(P,R)+2R`。可复用平面先承担
拆分/转置，GEMM 完成后再作为结果平面；顺序由同一 stream 保证。

`n,k<=32` 时使用 `ComplexDirectRankKTiling(mode=2)`，在一次内层循环中
同时累加 `alpha*A*B^H` 和 `conj(alpha)*B*A^H`。该路径同样只写目标三角。

### 精度与性能关注点

- `alpha` 的实部、虚部以及第二项的 `conj(alpha)` 符号分别覆盖；
- `trans=C` 的两次共轭不得重复或遗漏；
- 非目标三角逐位保持，Hermitian 对角虚部为正零；
- 除 1024/2048 基准外，覆盖 `n=500,k=13377` 极宽形状和 padded LD。

实现落点：Host 调度位于 `blas/her2k/arch22/cher2k_host.cpp`；第一项缩放
使用 `complex_her2k_product_scale_do`，结果转置使用
`complex_planar_transpose_pair_do`，最终由
`complex_physical_product_combine_do(productMode=3)` 合并。

## CHERK 设计

### 接口契约与数学定义

`trans=N` 计算 `C=alpha*A*A^H+beta*C`；`trans=C` 计算
`C=alpha*A^H*A+beta*C`。`alpha/beta` 均为实数，公开接口不接受普通
转置 T。C 只更新指定三角，对角虚部写零。

参数校验根据 `trans` 选择 `lda>=max(1,n)` 或 `lda>=max(1,k)`。
`n=0` 快速返回。Host 内部将 `C` 映射为转置标识 T，用于平面数据布局，
但不放宽公开枚举范围。

### 数据流与分块

1. 将 A 拆成物理实/虚平面。`trans=C` 时在拆分阶段完成虚部取反。
2. 生成与逻辑操作数对应的转置平面，形成 `Ar/Ai/Atr/Ati`。
3. 计算 `P1=Ar*Atr`、`P2=Ai*Ati`。
4. 原地生成 `Ar+Ai` 与 `Atr-Ati`，计算 P3。
5. 合并 P1/P2/P3，乘实数 alpha，再与实数 `beta*C` 相加。
6. 仅写目标三角，并清零对角虚部。

GEMM 逻辑形状为 `n×n×k`。输出平面使用 Row-Major，而公开 C 为
Column-Major，因此 Host 将公开三角映射到内部相反三角。块网格大于
4×4 时 GEMM 只调度这一内部三角；小网格仍计算全块，最终由合并 Kernel
执行精确元素掩码。

### Workspace 与特殊路径

通用布局为四个输入/转置平面加四个 `n×n` 结果平面，即 `4P+4R`。
`alpha=0` 或 `k=0` 时不申请乘积平面；若 `beta=1` 直接返回，否则由
CHERK 合并 Kernel 完成目标三角缩放。`n,k<=32` 时使用
`ComplexDirectRankKTiling(mode=1)`。

设计固定关闭 HF32。Hermitian 乘积在消去项附近对尾数敏感，HF32 路径
不作为可接受的性能换精度选项。

### 精度与性能关注点

- `N/C` 两条路径都校验实数 alpha/beta 和对角虚部；
- `alpha=0`、`beta=0/1/-1` 与 `k=0` 分别验证是否误读 A/C；
- 非目标三角逐位不变；
- 覆盖 `n=16384,k=500` 大 workspace 和 `n=500,k=16384` 极宽泛化形状。

实现落点：Host 入口、参数准备、workspace 和 GEMM 阶段位于
`blas/herk/arch22/cherk_host.cpp`；特殊 beta 路径使用
`blas/herk/arch22/cherk_kernel.cpp`，通用结果使用
`complex_physical_product_combine_do(productMode=4)`。

## CSYMM 设计

### 接口契约与数学定义

CSYMM 与 CHEMM 具有相同的尺寸、side、uplo 和标量接口，但 A 满足
`A=A^T` 而不是 `A=A^H`。因此镜像元素保持原虚部符号，A 的对角虚部
也是有效输入，不得清零。

`side=L` 时 `order=m`，计算 `alpha*A*B+beta*C`；`side=R` 时
`order=n`，计算 `alpha*B*A+beta*C`。参数校验、零维处理和 Device 标量
读取与 CHEMM 共用同一设计。

### 数据流与分块

1. 拆分 A 的指定三角并恢复完整复对称平面，镜像时只交换行列。
2. 拆分 B，并按 workspace 模式生成或原地复用 B 的求和平面。
3. 根据 side 选择 `B*A` 或 `A*B` 的三次实 GEMM 输入顺序。
4. 按一般复数 3M 公式恢复结果，应用复数 alpha/beta，写完整 C。

GEMM 形状、固定 K 路由、二维核心划分和尾块策略与 CHEMM 相同。两者只
共享调度骨架，不共享结构语义标志：CSYMM 必须始终设置
`hermitian=false`。

### Workspace 与边界处理

常规布局使用 `3A+6O`，超过 2 GiB 优选阈值后使用 `2A+5O`。小矩阵
进入直接路径，`LoadSymmetric` 对非存储三角只做转置索引，不执行共轭。
`alpha=0` 时跳过 A/B 展开和 GEMM，仅保留 beta 更新。

### 精度与性能关注点

- 必须用包含非零对角虚部的数据区分 CSYMM 与 CHEMM；
- UPPER/LOWER 的未存储半区填入 NaN 时，输出仍只由存储半区决定；
- cross、padded LD 和非 128 对齐 order 验证镜像边界；
- 1024/2048 的 L/U 与 R/L 四个任务书点独立验收，不能沿用 CHEMM 结果。

实现落点：公开入口位于 `blas/symm/arch22/csymm_host.cpp`，以
`hermitian=false` 调用 `RunComplexSymm`；其余公共落点与 CHEMM 相同。

## CSYRK 设计

### 接口契约与数学定义

`trans=N` 计算 `C=alpha*A*A^T+beta*C`；`trans=T` 计算
`C=alpha*A^T*A+beta*C`。alpha/beta 均为复数。公开接口只接受普通
转置 T，不接受共轭转置 C；C 是复对称矩阵，对角虚部可以非零。

参数校验按 `N/T` 选择 lda 下限。`n=0` 快速返回；`k=0` 或复数
`alpha=0+0i` 时跳过 A，只更新目标三角。

### 数据流与分块

1. 按物理形状拆分 A，不对虚部取反。
2. 生成配对转置平面 `Ar/Ai/Atr/Ati`。
3. 计算 `P1=Ar*Atr`、`P2=Ai*Ati`。
4. 原地生成 `Ar+Ai`、`Atr+Ati` 并计算 P3。
5. 以一般复数 3M 规则恢复乘积，应用复数 alpha/beta，仅写目标三角。

内部 Row-Major 三角与公开 Column-Major 三角方向相反，GEMM 的
`outputTriangle` 在 Host 设置。块网格超过 4×4 时压缩三角任务，小网格
保持完整调度。

### 长 K 与 Workspace

通用 workspace 为 `4P+4R`。当 `k>=8000` 时，每个实 GEMM 沿 K 维
固定拆成四段：第一段 `enAtomic=0` 覆盖输出，后续段
`enAtomic=1` 通过 `MatmulImpl::IterateAll(...,1)` 累加。各段 A/B 偏移
分别按逻辑左、右平面步长计算，最后仍只进行一次复数合并。

`n,k<=32` 时使用 `ComplexDirectRankKTiling(mode=0)`；该模式的右操作数
不共轭，且不强制对角虚部为零。

### 精度与性能关注点

- 使用纯虚输入区分 `A*A^T` 和 `A*A^H`；
- `T` 路径不允许在拆分阶段取反虚部；
- 非目标三角逐位不变，但目标对角虚部按普通复数结果比较；
- `k>=8000` 专门覆盖 split-K 累加精度；
- 重点验证任务书最紧的 1024 LOWER/T 性能点及 padded LD 稳定性。

实现落点：Host 校验、平面准备和四路 split-K 位于
`blas/syrk/arch22/csyrk_host.cpp`；通用合并使用
`complex_physical_product_combine_do(productMode=1)`。

## 支持硬件

| 芯片版本 | CANN | 架构目录 | 支持情况 |
| --- | --- | --- | --- |
| Atlas A2 训练系列产品（910B2） | 9.0.0 | `arch22` / `dav-2201` | 支持 |

本设计不声明 arch20、arch35 或其他 SOC 已完成适配。构建时以
`--soc=ascend910b3` 选择本实现。

## 算子约束限制

- 数据类型仅支持单精度复数矩阵；标量类型按各接口定义。
- 矩阵仅支持 Column-Major，不支持 broadcast。
- CHEMM/CSYMM 只引用 A 的指定三角；秩更新只修改 C 的指定三角。
- 非紧凑 leading dimension 受支持，但调用方必须满足最小值约束。
- Kernel 不跨 stream 共享临时平面；调用方负责遵守 handle 和 stream 生命周期。
- 库管理 workspace 最大 8 GiB；用户 workspace 不足时返回分配失败，不隐式换用库空间。
- 设计不采用 HF32、Host 回退或以牺牲非目标三角语义换取性能的路径。

# 可维可测分析

## 可维护性与定位手段

Host 的参数错误返回统一状态码，并保留算子 tag。workspace 扩容失败记录
所需和可用字节数。AIV/AIC 核数由运行环境查询，不在接口代码中写死。

源码职责保持分层：接口特有逻辑位于对应 `*_host.cpp`；共享变换位于
`common/arch22`；实 GEMM 优化位于 `gemm/arch22`。后续修改固定 K 路径
时，需要同时回归五个接口，因为它们共用 `gemm_kernel_do`。

实现核对基线为 `ops-blas` 个人 fork 分支 `feat/arch22-matmul-series` 的
commit `fa2fd391be04d711b10c03cdadc278e967ce71f1`。设计中的文件名和函数名
按该版本核对；验收提交若产生新 commit，应在自验证快照中记录新版本及 diff。

## 分算子验证门禁

每个算子按独立 blueprint 推进，前一门禁未通过时不登记该算子完成：

| 门禁 | 判定内容 | 记录 |
| --- | --- | --- |
| G1 构建 | Release、`ascend910b3` 单算子及五算子联合构建 | commit、构建命令、完整日志 |
| G2 功能 | 枚举组合、零维、特殊标量、padding、结构语义 | GTest 用例和失败明细 |
| G3 精度 | 官方 CSV 非性能集、非目标区、HER 对角、大 workspace | 逐例结果、随机种子、设备号 |
| G4 性能 | 任务书四点、独立轮次、NPU 稳定性 | 原始样本、统计 CSV、门限 |
| G5 泛化 | 奇数、非方、cross/padded、极宽/极高 | NPU 多轮结果及异常复测 |

对应测试入口为：

| 算子 | 实现测试 | 任务书脚本 |
| --- | --- | --- |
| CHEMM | `test/hemm/chemm/` | `test_script/chemm/verify_accuracy.py`、`verify_performance.py` |
| CHER2K | `test/her2k/cher2k/` | `test_script/cher2k/verify_accuracy.py`、`verify_performance.py` |
| CHERK | `test/herk/cherk/` | `test_script/cherk/verify_accuracy.py`、`verify_performance.py` |
| CSYMM | `test/symm/csymm/` | `test_script/csymm/verify_accuracy.py`、`verify_performance.py` |
| CSYRK | `test/syrk/csyrk/` | `test_script/csyrk/verify_accuracy.py`、`verify_performance.py` |

## 精度标准

有限值按生态算子标准比较，记录 `atol=2^-16`、`rtol=2^-10`，并要求至少
99% 元素满足任务书采用的误差判定。NaN/Inf 用例比较数值分类，不将两个
NaN 当作普通浮点误差。

除数值容差外，结构条件单独判定：

- CHEMM/CSYMM：输出完整矩阵参与比较；
- CHERK/CHER2K/CSYRK：非目标三角与输入 C 逐位一致；
- CHERK/CHER2K：目标对角虚部必须为 `+0.0f`；
- padding、A 的未存储三角不得被读入有效结果。

## 性能与稳定性标准

任务书性能判定为：

```text
NPU median <= gpu_baseline.csv latency / 0.8
```

正式任务书集对每个用例执行 5 个独立进程轮次，每轮预热 10 次、采样 50
次。泛化集执行 5 个独立进程轮次，每轮预热 20 次、采样 100 次。统计中位数、
最小值、最大值、总体 CV 和五轮 median CV。稳定性只评价 NPU，门限为总体
`CV<=5%` 且轮间 median `CV<=3%`。首次失败保留原始记录，只有增加样本后的
独立复测满足相同门限才可关闭问题。

## 已归档验证结果

以下数据属于当前实现的验证结果，不作为设计假设：

| 项目 | 结果 |
| --- | --- |
| 五算子官方非性能集 | 1092/1092；三轮累计 3276/3276 |
| 大 workspace 精度 | CHER2K 13377×500、CHERK/CSYRK 16384×500，9/9 |
| 任务书性能 | 20/20；5000 个有效 NPU 样本；最大总体 CV 0.906% |
| 最紧任务书点 | CSYRK 1024 LOWER/T：0.288053 ms，门限 0.311250 ms |
| 最终泛化集 | 30/30 NPU 稳定；15000 样本；最大总体/轮间 CV 1.043365%/0.788186% |
| 任务书基线比值 | 20/20；最大 `median/target=0.782654` |
| 当前 A100 对照 | 30/30 满足 1.25 倍关系；最大 `NPU/A100=0.995711` |

完整日志、原始样本、阶段状态、代码版本、tracked diff、未跟踪文件清单及
SHA-256 位于实现仓 `verification/matmul_series_arch22/`。A100 自身稳定性
不作为 NPU 门禁结果。

## 兼容性与风险控制

- **公共 GEMM 回归风险**：固定 K 和静态桶由五个接口共享，任何修改均执行
  五算子联合精度及性能门禁。
- **三角方向风险**：Column-Major 输出映射到内部 Row-Major 后方向相反，
  通过非目标三角逐位检查和 upper/lower 成对用例约束。
- **镜像竞态风险**：非对齐对角修复使用独立 Kernel 建立提交顺序，并以
  4、8、65、1001、1025 等 order 做逐元素镜像检查。
- **workspace 风险**：所有尺寸按 `size_t` 和 512 B 对齐计算；超出 8 GiB
  明确返回失败，不截断、不静默回退。
- **精度换性能风险**：HF32 探针未达到精度门限，最终路径固定 FP32；候选
  优化只有同时通过精度门禁和性能筛选后才能进入正式版本。
- **泛化波动风险**：极宽、极高、padded 和非对齐形状使用独立进程多轮采样，
  不用单次最好值作为结论。
