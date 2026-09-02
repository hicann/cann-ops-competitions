# aclblasCsymv 算子设计文档

## 需求背景（required）

### 需求来源

本算子来源于 2026 年 CANN 社区任务“8 月社区任务-aclblasCsymv 算子开发（950）”，
目标仓库为 `cann/ops-blas`，适配 Ascend 950PR（arch35）和 CANN 9.1.0。

### 背景介绍

`aclblasCsymv` 执行单精度复数对称矩阵-向量乘：

```text
y = alpha * A * x + beta * y
```

其中 A 为列主序 `complex64` 方阵，仅存储 `uplo` 指定的上三角或下三角。
未存储的另一半按 `A = A^T` 推导，不对虚部取共轭，因此不是 Hermitian `chemv` 语义。

ops-blas 已有实数 `aclblasSsymv`，但公共头文件和 arch35 目录中尚无 `aclblasCsymv`。
本实现新增公共 API，不定义 950PR 私有平行接口。

## 需求分析（required）

### 需求描述

- 实现 `COMPLEX64` 对称矩阵-向量乘，支持 `ACLBLAS_UPPER` 和 `ACLBLAS_LOWER`。
- 支持 Host/Device 内存中的 `alpha` 和 `beta`。
- 支持正负 `incx`/`incy`，负步长按 Netlib 反向起点语义访问。
- 满足 quick return、scale-only 和 `beta=0` 不读旧 y 的契约。
- 非法 `uplo` 返回 `ACLBLAS_STATUS_INVALID_ENUM`，其他非法数值或必要空指针返回 `ACLBLAS_STATUS_INVALID_VALUE`。
- 使用 handle 绑定的 stream 异步直调 Ascend C kernel。

### 需求拆解

1. 在 `include/cann_ops_blas.h` 增加接口声明，并补充 `blas/symv/README.md`。
2. 在 `blas/symv/arch35/` 实现 Host 校验、tiling 和 SIMT kernel。
3. 为复数对称语义实现独立 CPU golden，防止误用共轭。
4. 导入 1200 条任务 CSV，补齐空指针、quick return、Host/Device 标量和性能门槛测试。
5. 在 Ascend 950PR/CANN 9.1.0 上完成编译、精度和性能验证。

## 详细设计（required）

### 算子分析

#### 数学公式

对每个输出行 `i`：

```text
sum(i) = Σ A(i,j) * x(j), j = 0..n-1
y(i)   = alpha * sum(i) + beta * y(i)
```

复数乘法显式拆分为：

```text
(ar + i*ai) * (xr + i*xi)
  = (ar*xr - ai*xi) + i*(ar*xi + ai*xr)
```

矩阵元素索引（列主序）：

```text
UPPER: idx = i + j*lda  (i <= j), else j + i*lda
LOWER: idx = i + j*lda  (i >= j), else j + i*lda
```

交换 `i`/`j` 时只做转置，不修改虚部符号。

#### 支持数据类型和形状

| 项目 | 设计 |
| --- | --- |
| A/x/y | `aclblasComplex`，即两个 float32 交错存储 |
| alpha/beta | `aclblasComplex*`，Host 或 Device 内存 |
| A | `lda * n` 列主序存储，有效形状 `n * n` |
| x/y | 逻辑长度 n，存储长度至少 `1 + (n-1)*abs(inc)` |
| n | 运行时参数，`n >= 0` |

### 算子实现

#### Host 侧设计

1. 先检查 `n < 0`、handle 和 `n == 0` quick return，再检查其余固定参数。
2. 使用 `aclrtPointerGetAttributes` 分别判断 `alpha`/`beta` 的内存位置。
3. Host 标量直接写入 tiling；Device 标量指针直接传给通用 kernel，普通路径不做 D2H 或 stream 同步。
4. 只有 A/x/y 存在空指针时，才需要将 Device 标量拷回 Host，以判断 quick return 或 scale-only 是否允许该空指针。
5. 通用路径使用 `min(ceil(n / SIMT_MIN_THREAD_NUM), aivCoreNum)` 个 AIV，线程数按每核输出行数对齐并限制到 `SIMT_MAX_THREAD_NUM`。
6. Host 标量、`beta=0`、`alpha!=0`、`incx=incy=1` 且形状为 512/2048 时进入性能快路径；2048 路径还会确认 handle workspace 至少有 512 KiB。

#### Kernel 侧设计

- 通用路径使用 AIV-only SIMT kernel，一个逻辑线程计算一个输出行，并通过 grid-stride loop 覆盖 n 行；该路径支持全部步长、标量位置和特殊行为。
- 512 快路径使用 warp-per-row 的协同 SIMT 计算，warp 内 32 个 lane 分担列累加并归约，不使用 workspace。
- 2048 快路径把存储三角划分为 256x64 矩形任务。每个 AIV 将矩形载入 UB，使用 64-lane 寄存器向量同时形成直接项和对称项，写入按输出 tile/source tile 排列的 partial workspace，再由第二个 AIV kernel 完成 32 路归约和 alpha 融合。
- 对角 tile 使用寄存器 mask，只读取 `uplo` 指定的三角；非对角 tile 同时产生 `A*x` 的行贡献和转置后的列贡献，全程不取共轭。
- `UPPER`/`LOWER` 使用模板实例化，避免热循环中的动态 uplo 分支。
- 负步长时，逻辑元素 `k` 对应存储索引 `(n-1-k)*abs(inc)`。
- `alpha=(0,0)` 时跳过 A/x 读取；`beta=(0,0)` 时不读取 y 旧值。
- `alpha=(0,0), beta=(1,0)` 在 kernel 中也保留无写回保护，覆盖 Device 标量且 Host 不可知的路径。
- 累加和最终 alpha/beta 组合均显式拆分实部和虚部，不使用共轭操作。

#### 性能优化方案

- 512 形状采用 warp 协同累加，将单行串行循环缩短为 32 路并行，并通过 shuffle 完成寄存器归约。
- 2048 形状复用每个 256x64 UB 矩形，同时计算直接项和对称项；矩阵只搬运一次，不生成完整对称矩阵，也不使用原子加。
- 2048 partial workspace 固定为 `32*32*64*sizeof(complex64) = 512 KiB`；按输出 tile 连续布局，使第二阶段可整段搬入 UB 并向量归约。
- 寄存器热循环使用 post-update load，减少 Aux Scalar 地址计算；仅 `csymv_kernel.cpp` 在项目默认 Debug 构建中追加 `-O3 -DNDEBUG`，不改变其他算子的构建方式。
- Device 标量在普通路径直接由 kernel 读取，避免每次 D2H 和同步；Host 标量通过 tiling 传值。
- `beta=0` 和 `alpha=0` 分支避免无效内存读取。

### 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Ascend 950PR（arch35） | √ |
| 其他产品 | 不支持 |

### 算子约束限制

- 仅支持 `ACLBLAS_UPPER` 和 `ACLBLAS_LOWER`。
- A 必须为列主序，`lda >= max(1,n)`。
- `incx`/`incy` 可为正数或负数，不能为 0。
- 除 `n=0` quick return 外，`alpha`/`beta` 不能为空。
- A/x 只在 `alpha!=0` 时必须有效；y 只在需要写回时必须有效。
- 运行为异步调度，读取结果前由调用方同步 handle 绑定的 stream。

## 可维可测分析

### 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | 实部/虚部分别按 FLOAT32：`rtol=2^-10`、`atol=2^-16`、匹配率 >=0.99，且最大绝对误差 <= `max(1e-2,32*ULP)` | opbase 生态算子开源精度标准 |
| 性能标准 | UPPER/512 <=22.75 us；UPPER/2048 <=55.62 us；LOWER/2048 <=40.25 us；先预热，有效采样 60 次 | Csymv 任务书 |

测试工程包含 1000 条功能/精度 CSV 和 200 条性能参考 CSV。均匀/正态输入各 50%，
覆盖 uplo、shape、padding lda、正负步长、Host/Device 标量、NaN/Inf 以及全部边界状态。
未使用的矩阵三角填入 NaN，用于检测误读和错误共轭。

### 兼容性分析

本次新增统一公共 API，不改变现有 `aclblasSsymv` 的签名和行为。
新文件仅在 arch35 编译集合中生效，其他产品的现有算子实现不受影响。

## 自测结果

在 HiDevLab Ascend 950PR、CANN 9.1.0 环境中以工程默认 Debug 配置完成干净构建，
其中仅 Csymv kernel 使用源文件级 O3 优化。快速集 26/26、全量功能/精度集 1004/1004、
性能门槛测试 1/1 通过。代码审阅修正后的最终平均耗时为 UPPER/512 `18.3722 us`、
UPPER/2048 `16.7261 us`、LOWER/2048 `16.8605 us`。
