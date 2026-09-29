# 需求背景（required）

## 需求来源

`aclblasCtpsv` 对齐 cuBLAS `cublasCtpsv`，面向 Atlas A2/A3 系列产品提供 COMPLEX64 三角打包线性方程求解能力。算子采用 Ascend C kernel 直调方式，接口属于 ops-blas 句柄式 BLAS 接口。

## 背景介绍

### aclblasCtpsv 算子功能

算子求解

$$
\operatorname{op}(A)x=b,
$$

其中右端向量 $b$ 初始存放在 `x` 中，解原地写回 `x`。$A$ 为 $n\times n$ 上三角或下三角复数矩阵，`AP` 按列主序仅保存有效三角部分。

| 参数 | 含义 | 类型 | 约束 |
| --- | --- | --- | --- |
| `handle` | 携带执行 stream 的上下文句柄 | `aclblasHandle_t` | 非空 |
| `uplo` | `AP` 保存上三角或下三角 | `aclblasFillMode_t` | `ACLBLAS_UPPER`、`ACLBLAS_LOWER` |
| `trans` | $\operatorname{op}(A)$ 的形式 | `aclblasOperation_t` | `ACLBLAS_OP_N`、`ACLBLAS_OP_T`、`ACLBLAS_OP_C` |
| `diag` | 单位或非单位对角线 | `aclblasDiagType_t` | `ACLBLAS_UNIT`、`ACLBLAS_NON_UNIT` |
| `n` | 矩阵阶数 | `int` | $n\ge 0$ |
| `AP` | 三角矩阵打包数据 | `const aclblasComplex *` | `n>0` 时非空，长度 $n(n+1)/2$ |
| `x` | 输入右端项及原地输出解 | `aclblasComplex *` | `n>0` 时非空 |
| `incx` | `x` 的元素步长 | `int` | `incx != 0`，支持负值 |

# 需求分析（required）

## 需求描述

实现 COMPLEX64 三角打包求解，覆盖上下三角、非转置、转置、共轭转置、单位对角、非单位对角及正负步长。接口在 `handle` 绑定的 stream 上异步执行；`n=0` 直接返回成功。

## 需求拆解

1. 正确换算上、下三角列主序打包下标，并以 64 位无符号整数完成长度及地址计算。
2. 根据 `uplo` 和 `trans` 选择前向或后向替换顺序，保证每个未知量只依赖已经求出的元素。
3. `ACLBLAS_OP_C` 对参与计算的矩阵元素取共轭；`ACLBLAS_UNIT` 不访问 `AP` 中的对角元素。
4. 将带步长向量映射为逻辑连续向量。`incx<0` 时，逻辑元素 $x_i$ 位于物理偏移 $(n-1-i)|incx|$。
5. 对 `n<=4096` 使用整向量驻留 UB 的连续访存路径；更大规模使用 GM 驻留的分块路径，保持输入规模泛化。
6. Host 侧只做参数检查、Tiling 计算和异步 kernel 下发，不分配临时 Device 内存，也不执行 stream 同步。

# 详细设计（required）

## 算子分析

### 数学公式

复数元素记为 $z=z_r+i z_i$。复数乘法和共轭乘法分别为

$$
(a_r+i a_i)(x_r+i x_i)=(a_rx_r-a_ix_i)+i(a_rx_i+a_ix_r),
$$

$$
\overline{(a_r+i a_i)}(x_r+i x_i)=(a_rx_r+a_ix_i)+i(a_rx_i-a_ix_r).
$$

上三角元素 $A_{i,j}$（$0\le i\le j<n$）在 `AP` 中的下标为

$$
P_U(i,j)=i+\frac{j(j+1)}{2}.
$$

下三角元素 $A_{i,j}$（$0\le j\le i<n$）在 `AP` 中的下标为

$$
P_L(i,j)=i+\frac{j(2n-j-1)}{2}.
$$

所有乘法及加法使用 FP32 实部和虚部计算。非单位对角采用缩放形式的复数除法：根据 $|a_r|$ 与 $|a_i|$ 选择比例计算式，避免直接计算 $a_r^2+a_i^2$ 引起不必要的上溢或下溢。零、Inf 和 NaN 按 FP32 运算规则自然传播，不执行奇异性检测。

### 支持数据类型

`AP`、`x` 均为 COMPLEX64，每个元素由相邻的 FP32 实部和虚部组成。输出仍为 COMPLEX64。

### 支持形状

- `AP` 逻辑长度为 $n(n+1)/2$。
- `x` 的逻辑长度为 $n$，物理跨度为 $1+(n-1)|incx|$ 个复数元素。
- 支持 $n=0$、非对齐规模、正步长和负步长。

## 算子实现

### 实现方案

#### 3.2.1 host 侧设计

参数检查顺序如下：

1. `handle` 为空时返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`。
2. 检查 `uplo`、`trans`、`diag`、`n>=0` 和 `incx!=0`，非法值返回 `ACLBLAS_STATUS_INVALID_VALUE`。
3. `n=0` 时返回 `ACLBLAS_STATUS_SUCCESS`，不访问 `AP` 和 `x`。
4. `n>0` 时检查 `AP` 和 `x` 非空。

Host 侧从平台信息获取 Vector Core 数量和 UB 容量。三角替换存在逐元素真依赖，kernel 使用一个 Vector Core；增加分核会引入逐行核间同步并降低小、中规模性能。Tiling 数据包含 `n`、`incx`、`uplo`、`trans`、`diag`、UB 容量、`tileComplex` 和执行路径标识。

`n<=4096` 时采用整向量驻留路径。设 UB 可用容量为 $U$，AP 单次处理 $C$ 个复数，固定保留区为 $S=16\text{ KiB}$，则保守资源约束为

$$
8n+32C+S\le U.
$$

其中 $8n$ 字节保存 `xReal/xImag`，$16C$ 字节保存两份交替搬入的交织 AP tile，$8C$ 字节保存拆分后的 `aReal/aImag`，另 $8C$ 字节用于乘积和归约临时区。`tileComplex` 取 `min(n, floor((U-8n-S)/32))` 向下对齐到 4 个复数，并限制在 4096 以内。DAV_2201 的 UB 为 192 KiB；当 `n=4096`、`C=4096` 时分配量为 176 KiB，UB 利用率约 91.7%。

当整向量不满足 UB 约束时，采用 GM 驻留分块路径。该路径按同一替换顺序分块搬入已求出的 `x` 和连续 AP 片段，在 UB 内形成部分和或完成向量更新，再把当前结果写回 GM，不限制合法 `n`。

#### 3.2.2 kernel 侧设计

kernel 以 `KERNEL_TYPE_AIV_ONLY` 在一个 Vector Core 上执行。整向量驻留路径先把逻辑 `x` 搬入 UB 并拆成 `xReal`、`xImag` 两个 FP32 平面：`incx=1` 使用连续搬运和向量解交织，其他步长按逻辑下标聚集；完成求解后执行相反的交织及散写。

四类替换均将 AP 访问转换为列内连续访问：

| `uplo` | `trans` | 替换方向 | 连续计算方式 |
| --- | --- | --- | --- |
| LOWER | N | $j=0\rightarrow n-1$ | 求出 $x_j$ 后，用下三角第 $j$ 列的对角线以下连续片段更新 $x_{j+1:n}$ |
| UPPER | N | $j=n-1\rightarrow 0$ | 求出 $x_j$ 后，用上三角第 $j$ 列的对角线以上连续片段更新 $x_{0:j}$ |
| UPPER | T/C | $j=0\rightarrow n-1$ | 用上三角第 $j$ 列对角线以前的连续片段与 $x_{0:j}$ 做复数点积 |
| LOWER | T/C | $j=n-1\rightarrow 0$ | 用下三角第 $j$ 列对角线以后的连续片段与 $x_{j+1:n}$ 做复数点积 |

`N` 分支的列更新以四次 FP32 标量 AXPY 组成复数向量更新：

$$
x_r\leftarrow x_r-x_{j,r}a_r+x_{j,i}a_i,
$$

$$
x_i\leftarrow x_i-x_{j,i}a_r-x_{j,r}a_i.
$$

`T/C` 分支分别形成实部和虚部乘积向量，再执行 FP32 向量归约。`C` 分支通过虚部符号调整实现共轭。不同 AP tile 的部分和以固定的逻辑顺序累加到当前标量结果。

AP tile 使用双缓冲：MTE2 搬入下一片时，Vector 单元处理当前片。完整块使用对齐搬运，末块使用带 Padding 的搬运并按真实元素数设置向量长度。每一列只读取有效的非对角片段；仅在 `diag=ACLBLAS_NON_UNIT` 时单独读取对角元素，因此单位对角路径不会触碰对应 AP 地址。

`n=4096` 时，AP 总数据量约为 64 MiB，每个有效三角元素只从 GM 读取一次；`x` 在主计算期间驻留 UB，只在入口和出口各搬运一次。计算量为 $n(n-1)/2$ 次复数乘加及至多 $n$ 次复数除法。该路径消除逐元素 GM 读取、Host 侧内存分配和同步开销。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas 800I/T A2 | √ |
| Atlas 800I/T A3 | √ |

Atlas A2/A3 均对应 DAV_2201，代码位于 `arch22`，以运行时平台接口返回的 UB 容量为 Tiling 依据。

## 算子约束限制

- `n>=0`，`incx!=0`。
- `uplo`、`trans`、`diag` 仅支持接口定义中的合法枚举。
- `n>0` 时 `AP` 和 `x` 必须为有效 Device 指针。
- `AP` 与 `x` 的内存区域不得重叠。
- 除 `incx` 表达的向量步长外，不支持其他非连续布局；不涉及广播。
- `ACLBLAS_UNIT` 将对角线视为 $1+0i$，不读取 AP 对角元素。
- 接口异步返回；调用方在读取输出前负责同步 `handle` 所绑定的 stream。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | 以 Netlib CBLAS `ctpsv` 为单标杆，对复数模误差执行混合容差判定：`rtol=atol=2^-13`、`required_matched_ratio=0.99`，并满足 `max_abs_error <= max(1e-2, 32*ULP)` | aclblasCtpsv A2/A3 任务书 |
| 性能标准 | Atlas 800T A2 (910B3)，msprof 预热后有效采样取平均；5 个指定用例的 Avg time 分别不高于 252.2、573.9、1228、1211、7327 us | aclblasCtpsv A2/A3 任务书 |

功能覆盖包含两种 `uplo`、三种 `trans`、两种 `diag`、`n=0/1`、非对齐规模、`incx=1/2/3/-1/-2/-3`、非法枚举、负 `n`、零步长、空指针以及 Inf/NaN 输入。性能关键规模覆盖 256、512、1024、2048 和 4096。

## 兼容性分析

该接口为新增 COMPLEX64 TPSV 能力，参数顺序和语义与 `cublasCtpsv` 对齐。实现复用 ops-blas 的句柄、stream、状态码及 kernel 直调约定，不改变已有 `aclblasStpsv` 接口和行为。
