# aclblasCsymm 算子设计文档

# 需求背景

## 需求来源

本需求来源于 2026 年 8 月社区任务《aclblasCsymm 算子开发任务书》。任务要求基于
`ops-blas` 工程，使用 Ascend C Kernel 直调方式为 Ascend 950PR 实现单精度复数对称矩阵乘
接口 `aclblasCsymm`，并完成接口、功能、精度、性能和边界场景验证。

## 背景介绍

### 算子功能

`aclblasCsymm` 是 BLAS Level 3 单精度复数对称矩阵乘算子。根据 `side` 的取值，计算：

$$
\begin{aligned}
side=LEFT &: C \leftarrow \alpha A B + \beta C \\
side=RIGHT &: C \leftarrow \alpha B A + \beta C
\end{aligned}
$$

其中 $A$ 为复数对称矩阵，满足 $A=A^T$。这里使用普通转置，不使用共轭转置；对角元素
的虚部按输入值参与计算，不能按 Hermitian 矩阵规则清零。

矩阵采用列主序存储。`uplo` 指定 $A$ 的有效三角区域，另一半矩阵通过对称关系得到：

- `ACLBLAS_UPPER`：只读取上三角；
- `ACLBLAS_LOWER`：只读取下三角。

`aclblasComplex` 由相邻的两个 FP32 分量组成，设备内存布局为
`[real0, imag0, real1, imag1, ...]`。

### 设计目标

1. 公共接口和参数语义与任务书、cuBLAS `cublasCsymm`、Netlib `csymm` 对齐。
2. 覆盖 LEFT/RIGHT、UPPER/LOWER、矩形、非对齐尺寸及前导维 padding。
3. 小尺寸和一般复数系数优先保证参考顺序与精度；常用的大尺寸
   `alpha=(1,0), beta=(0,0)` 使用矩阵计算单元提高吞吐。
4. 利用 Ascend 950PR 的 Vector Core、Cube Core、L1/L0 分级存储和异步搬运能力。

# 需求分析

## 需求描述

新增以下公共接口，声明位于 `include/cann_ops_blas.h`：

```cpp
aclblasStatus_t aclblasCsymm(
    aclblasHandle_t handle,
    aclblasSideMode_t side,
    aclblasFillMode_t uplo,
    int m, int n,
    const aclblasComplex* alpha,
    const aclblasComplex* A, int lda,
    const aclblasComplex* B, int ldb,
    const aclblasComplex* beta,
    aclblasComplex* C, int ldc);
```

接口通过 `handle` 获取已绑定的 stream，并在该 stream 上异步下发。调用方读取 Device 输出
之前负责同步 stream。

## 参数与约束

| 参数 | 位置 | 含义与约束 |
| --- | --- | --- |
| `handle` | Host | 必须是有效句柄；空指针返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| `side` | Host | 仅支持 `ACLBLAS_SIDE_LEFT`、`ACLBLAS_SIDE_RIGHT` |
| `uplo` | Host | 仅支持 `ACLBLAS_UPPER`、`ACLBLAS_LOWER` |
| `m`、`n` | Host | 必须大于等于 0；任一为 0 时合法返回成功 |
| `alpha`、`beta` | Host | COMPLEX64 标量指针；非零维场景不可为空 |
| `A` | Device | LEFT 时为 `m×m`，RIGHT 时为 `n×n`；只读取指定三角 |
| `B` | Device | `m×n`，列主序，只读 |
| `C` | Device | `m×n`，列主序，原地更新 |
| `lda` | Host | LEFT 时 `lda>=max(1,m)`，RIGHT 时 `lda>=max(1,n)` |
| `ldb`、`ldc` | Host | 分别满足 `ldb>=max(1,m)`、`ldc>=max(1,m)` |

本任务不支持超出 `lda/ldb/ldc` 语义的任意 stride，不涉及 broadcast，不返回视图。
$A$ 的对称性由调用方保证，算子不比较输入的两个三角。

## 特殊值和边界语义

Host 侧按以下顺序处理参数和快速路径：

1. 检查 `handle`、`side`、`uplo`、`m`、`n`。
2. `m==0 || n==0` 时返回成功，不访问标量或矩阵。
3. 非零维场景检查 `alpha`、`beta`、`A`、`B` 和三个前导维。
4. 按任务书约定，`C==nullptr && beta==(0,0)` 时返回成功；`beta` 非零时 C 不能为空。
5. `alpha==(0,0) && beta==(1,0)` 时返回成功，C 保持不变。
6. `alpha==(0,0)` 的其他情况只计算 $C\leftarrow\beta C$，不读取 A、B 的数据内容。
7. 非法枚举返回 `ACLBLAS_STATUS_INVALID_ENUM`，其他非法值返回
   `ACLBLAS_STATUS_INVALID_VALUE`。

# 详细设计

## 算子分析

### 数学展开

记参与乘法的两个复矩阵为：

$$
X=X_r+iX_i,\qquad Y=Y_r+iY_i
$$

LEFT 时 $X=A,Y=B$，RIGHT 时 $X=B,Y=A$。复矩阵乘积为：

$$
XY=(X_rY_r-X_iY_i)+i(X_rY_i+X_iY_r)
$$

最终输出为：

$$
\begin{aligned}
C'_r &= \alpha_rP_r-\alpha_iP_i+\beta_rC_r-\beta_iC_i \\
C'_i &= \alpha_rP_i+\alpha_iP_r+\beta_rC_i+\beta_iC_r
\end{aligned}
$$

### 三次实数矩阵乘

当归约维度 `k<=2048` 时，性能主路径使用三次实数矩阵乘计算一个复数矩阵乘：

$$
\begin{aligned}
P_1 &= X_r(Y_r+Y_i) \\
P_2 &= (X_r+X_i)Y_i \\
P_3 &= (X_i-X_r)Y_r \\
P_r &= P_1-P_2 \\
P_i &= P_1+P_3
\end{aligned}
$$

该变换相对于常规四次实数矩阵乘少一次 AIC 计算和一次结果写回，但额外的输入加减会改变
浮点舍入。因此它只用于当前已经建立性能路径的范围，并保留全量精度门禁。

当 `k>2048` 时改用四次实数矩阵乘：

$$
P_r=X_rY_r-X_iY_i,\qquad P_i=X_rY_i+X_iY_r
$$

避免三次乘法中的预加减在长归约下继续放大消减误差。

### 对称矩阵寻址

列主序元素 `(row,col)` 的物理索引为 `col*ld+row`。读取 $A(row,col)$ 时：

```text
UPPER: row <= col 时读取 col*lda+row，否则读取 row*lda+col
LOWER: row >= col 时读取 col*lda+row，否则读取 row*lda+col
```

镜像只交换行列位置，实部和虚部均保持原值。未指定三角允许放入任意数据，不能被读取。

## 总体方案

当前实现先完成参数检查和快速返回，再按输入特征选择执行路径。`alpha==0` 时只处理
$\beta C$；小尺寸或一般复数系数使用 SIMT 直接计算；大尺寸且
`alpha=(1,0), beta=(0,0)` 时，先由 AIV 打包并补全对称矩阵，再由 AIC 完成三次或四次
FP32 矩阵乘，最后由 AIV 合并实部和虚部并写回 C。多阶段 Kernel 全部在同一 stream 上
按序下发，不进行 Host 数据往返或 Host 同步。

## Host 侧设计

### 路由策略

设 `k = (side==LEFT ? m : n)`：

| 条件 | 执行路径 | 目的 |
| --- | --- | --- |
| `m==0` 或 `n==0` | Host 直接返回 | 避免无效下发 |
| `alpha==0 && beta==1` | Host 直接返回 | C 无变化 |
| `alpha==0` | AIV scale-only | 不进行矩阵乘 |
| `k<=32` | AIV SIMT 直接路径 | 小尺寸减少启动和工作区开销 |
| alpha/beta 不是 `(1,0)/(0,0)` | AIV SIMT 直接路径 | 保留任务参考的计算顺序 |
| `k>32 && alpha==1 && beta==0` | AIV 打包 + AIC 矩阵乘 + AIV 合并 | 大尺寸吞吐路径 |

### 核数获取

- AIV 阶段通过 `GetAivCoreCount()` 获取可用 Vector Core 数，采用网格步进处理打包、合并
  或直接计算任务。
- AIC 阶段通过 `GetAicCoreCount()` 获取可用 Cube Core 数，输出块按核心编号循环领取。
- 当输出块数不少于 AIC 核数时可让全部 AIC 核心持续领取任务；输出块较少时会存在空闲核。

### Tiling 数据

Host 侧生成 `CsymmTilingData`，主要字段如下：

```text
m, n, k
lda, ldb, ldc
packedLda, packedLdb, tempLd
side, uplo
aPlane, bPlane, tempPlane
alpha.real, alpha.imag, beta.real, beta.imag
```

内部布局按以下规则对齐：

```text
packedLda = AlignUp(k, 16)
packedLdb = AlignUp(m, 16)
tempLd    = packedLdb
aPlane    = AlignUp(packedLda*k, 128) 个 float
bPlane    = AlignUp(packedLdb*n, 128) 个 float
tempPlane = AlignUp(tempLd*n, 128) 个 float
```

16 元素对齐满足 Cube 基础分块要求，128 个 FP32 元素对应 512B 的工作区分段对齐。

### Workspace 规划

三次实数矩阵乘路径包含 3 份 A 平面、3 份 B 平面和 3 份结果平面：

$$
workspace_{3M}=4\times(3aPlane+3bPlane+3tempPlane)\ bytes
$$

四次实数矩阵乘路径包含 2 份 A 平面、2 份 B 平面和 4 份结果平面：

$$
workspace_{4M}=4\times(2aPlane+2bPlane+4tempPlane)\ bytes
$$

Host 使用 64 位整数逐项检查工作区大小，再通过句柄的默认工作区一次性获取。分配器实际保留
的工作区可能按更大的容量档位增长，因此测试报告同时记录公式需求和设备实际分配值。

## Kernel 侧设计

### AIV 打包 Kernel

打包 Kernel 使用 SIMT 线程处理线性元素区间，完成：

1. 只从 `uplo` 指定三角读取 A，并在工作区补全完整对称矩阵。
2. 将交错 COMPLEX64 的 A、B 拆成 FP32 平面。
3. 对三次乘法路径直接生成 `Xr`、`Xr+Xi`、`Xi-Xr` 和
   `Yr+Yi`、`Yi`、`Yr`，使后续 AIC 直接读取。
4. 对四次乘法路径生成 `Xr`、`Xi`、`Yr`、`Yi`。
5. 将内部对齐产生的 padding 置零，保证边界块不会引入无效值。

线性布局中 B 的相邻行和 A 的有效半区可形成连续访问；A 的镜像半区存在跨列访问，是当前
打包阶段的主要离散访问来源。

### AIC 矩阵乘 Kernel

列主序 C 在内存中等价于行主序 $C^T$。Host 在 AIC Tiling 中交换 `m/n` 并翻转 `side`，
Kernel 直接计算行主序 $C^T$，从而复用统一的实数矩阵乘流程。

当前默认分块为：

```text
tileM = 64
tileN = 64
tileKChunk = 256
```

每个输出 tile 在 FP32 L0C 中完成 K 方向累加。数据路径为 GM→L1→L0A/L0B→L0C→GM：

- L1 预算为 512 KiB，A、B 的 K 分块分别使用两个缓冲区；
- L0A、L0B 各 64 KiB，使用双缓冲交替装载；
- L0C 为 256 KiB，保存当前输出块的 FP32 累加结果；
- 使用 `Nd2Nz` 搬运到 Cube 布局，使用 `Mmad` 完成 FP32 块矩阵乘；
- HF32 模式显式关闭，避免降低尾数精度；
- 完整 16 对齐块的 L0 K 步长为 32，非对齐尾块使用 8，保证边界可处理。

输出 tile 采用网格步进分配。奇数 M 块反向遍历 N 块，形成蛇形顺序，减少相邻任务反复跳转
到相距较远的输出区域。各搬运、矩阵乘和写回阶段使用事件同步，L1/L0 双缓冲用于覆盖部分
搬运等待。

对于 1024×1024 的指定示例，64×64 输出分块产生 256 个 tile，能够覆盖 28 个 AIC 核心。
对于 256×256 示例只有 16 个 tile，因此虽然启动全部 AIC 核心，实际最多 16 个核心承担
输出任务。

### AIV 合并 Kernel

三次乘法路径按 `real=P1-P2`、`imag=P1+P3` 合并；四次乘法路径按
`real=rr-ii`、`imag=ri+ir` 合并。随后融合 alpha、beta 并写回 C。

仅当 beta 非零时读取旧 C。每个线程负责独立输出元素，不使用原子操作，不写 `ldc` padding。

### AIV 直接 Kernel

直接路径由一个 SIMT 线程负责一个 C 元素，并沿 k 顺序累加。它直接使用用户提供的
`lda/ldb/ldc`，无需工作区，天然覆盖非对齐尺寸和 padding。

通用 alpha/beta 路径按 Netlib 的三角遍历顺序组织对角项、显式三角项和镜像三角项。单次
FP32 乘积先物化，再进行复数加减，避免编译器收缩操作改变任务参考所需的舍入行为。

# 可维可测分析

## 功能与精度标准

Golden 使用任务包提供的 Netlib CBLAS `cblas_csymm` 缓存结果。C 的实部、虚部分别按 FP32
标准检查：

| 项目 | 标准 |
| --- | ---: |
| `rtol` | $2^{-10}$ |
| `atol` | $2^{-16}$ |
| `required_matched_ratio` | 0.99 |
| `max_abs_error_limit` | `1e-2` 或任务标准规定的 `32*ULP` |

逐分量匹配条件为：

$$
|actual-golden|\le atol+rtol\times|golden|
$$

用例还需同时通过最大绝对误差、NaN/Inf 一致性、输出保护区和输入未修改检查。测试覆盖：

- LEFT/RIGHT 与 UPPER/LOWER 四种组合；
- 零维、1、质数、2 的幂、2 的幂±1、非对齐值和大尺寸；
- alpha/beta 的 0、1、纯虚数、一般复数和较大值；
- 方阵、宽矩形、窄矩形；
- `lda/ldb/ldc` 紧凑布局与 padding；
- 未引用三角填入不同值或非有限值，验证没有误读；
- 对角虚部非零，验证没有按 Hermitian 规则处理；
- 空指针、非法枚举、非法前导维和负维度的返回值。

三次实数矩阵乘、四次实数矩阵乘和直接路径都必须单独覆盖。
