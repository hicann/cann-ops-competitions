# aclblasStrmm 算子设计文档

任务：9 月社区任务，`aclblasStrmm` 算子开发（A2/A3）

目标开源仓：<https://gitcode.com/cann/ops-blas>

适配硬件：Atlas A2/A3 系列产品（arch22 / DAV_2201）

开发语言：C++ / Ascend C

实现目录：`blas/trmm/arch22/`，测试目录：`test/trmm/strmm/arch22/`

# 需求背景（required）

## 需求来源

本设计来源于 CANN 2026 年 9 月社区任务 `aclblasStrmm` 算子开发（A2/A3）。目标是在
`ops-blas` 开源仓中补充 Atlas A2/A3 的单精度三角矩阵乘，并复用仓内已有公共接口：

```cpp
aclblasStatus_t aclblasStrmm(
    aclblasHandle_t handle, aclblasSideMode_t side, aclblasFillMode_t uplo,
    aclblasOperation_t trans, aclblasDiagType_t diag, int m, int n,
    const float* alpha, const float* A, int lda, const float* B, int ldb,
    float* C, int ldc);
```

接口以 handle 绑定的 stream 异步提交任务。`alpha` 为 Host 或 Device 标量指针，A、B、C 为 Device 指针。
声明位于 `include/cann_ops_blas.h`，本设计不新增产品私有平行接口。

## 背景介绍

`aclblasStrmm` 对齐 cuBLAS `cublasStrmm`，计算列主序单精度三角矩阵乘：

```text
side = LEFT : C = alpha * op(A) * B    A 为 m×m，B、C 为 m×n
side = RIGHT: C = alpha * B * op(A)    A 为 n×n，B、C 为 m×n
```

`op(A)` 只允许 `ACLBLAS_OP_N` 与 `ACLBLAS_OP_T`。实数 TRMM 下 `ACLBLAS_OP_C` 无定义，返回
`ACLBLAS_STATUS_INVALID_VALUE`。cuBLAS 参考实现在实数档把 `OP_C` 当作 `OP_T`，本接口按任务书收紧。

A 只读取 `uplo` 指定的上三角或下三角。`diag = UNIT` 时对角视为 1 且不读取；`diag = NON_UNIT` 时读取对角。
C 是一般矩阵，离席写出。允许 `B` 与 `C` 指向同一块设备内存，其余重叠未定义。

### 现状分析

公共 API 已声明。Atlas A2/A3 对应的 `blas/trmm/arch22/` 需要补齐 Host、Kernel 和测试。本算子的关键问题如下：

1. 用户矩阵是列主序，Cube GEMM 按行主序组织 tile。把列主序缓冲解释成转置后的行主序，可以避免整块转置。
2. 三角部分之外必须写成 0，`UNIT` 对角必须写成 1。直接把 A 送进 GEMM 会把无效三角算进结果。
3. `alpha` 可以折进打包后的稠密三角矩阵，主路径 GEMM 不再单独做标量后处理。
4. FP32 Cube 累加顺序与 Netlib `strmm` 不同。大 `|alpha|` 且 `LOWER + T + NON_UNIT` 时，绝对误差会碰到
   `max(1e-2, 32*ULP)`。这条窄路径改为与参考 BLAS 相同的标量累加顺序。
5. `m`、`n` 都很小时代价主要在 launch，不值得走打包加 Cube。

# 需求分析（required）

## 需求描述

使用 Ascend C 实现 `aclblasStrmm`，数据类型为 FLOAT32，列主序，覆盖 LEFT/RIGHT、UPPER/LOWER、
N/T、UNIT/NON_UNIT 的合法组合，以及任务书规定的零维、`alpha=0`、空指针和非法枚举。

## 需求拆解

| 编号 | 子需求 | 设计响应 |
| --- | --- | --- |
| F1 | 公共 API | 沿用 `include/cann_ops_blas.h` 已有签名和状态码 |
| F2 | 左右乘 | LEFT 时 A 为 m×m，RIGHT 时 A 为 n×n |
| F3 | 三角裁剪 | 按 uplo 只保留一侧，对侧写 0；UNIT 对角写 1 且不读 A 的对角 |
| F4 | 转置 | 只接受 N/T；OP_C 返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| F5 | 列主序 | 打包仍是列主序；GEMM 把列主序缓冲看成转置后的行主序 |
| F6 | alpha | 主路径把 alpha 乘进打包矩阵；alpha=0 只清零 C |
| F7 | 前导维 | 校验并按 lda/ldb/ldc 寻址，不改写 padding 以外的要求范围 |
| F8 | 零维 | m=0 或 n=0 返回成功，不启动 Kernel |
| F9 | 原地 | B 与 C 同址时先把 B 拷到 workspace |
| F10 | 小形状 | m、n 都不大于 48 时走 AIV 参考 kernel |
| F11 | 大形状 | AIV 打包三角矩阵，再由 AIC FP32 GEMM 写出 C |
| F12 | 窄精度路径 | LOWER、T、NON_UNIT 且 `|alpha|>=50` 时走与参考 BLAS 同序的 AIV 标量累加 |
| F13 | 异步 | Kernel 提交到 handle 的 stream，读回前由调用方同步 |

## 范围边界

本设计支持：

- FP32 输入、输出与累加；
- side、uplo、trans(N/T)、diag 的合法正交组合；
- 运行时 m、n 和 lda/ldb/ldc padding；
- alpha 为 0、1、-1 及一般 FLOAT32 值，含 Host 指针和 Device 指针；
- B 与 C 同址；
- Inf、NaN 按 IEEE 运算传播。

本设计不支持或不定义：

- 非 FP32；
- `ACLBLAS_OP_C`；
- 超出 lda/ldb/ldc 语义的非连续 Tensor 和广播；
- 除 B≡C 以外的 A/B/C 内存重叠；
- 跨设备、跨卡拆分同一次调用。

# 详细设计（required）

## 算子分析

### 数学公式

列主序地址为 `matrix[col * ld + row]`。

```text
LEFT,  trans=N: C(i,j) = alpha * sum_k tri(A)(i,k) * B(k,j)
LEFT,  trans=T: C(i,j) = alpha * sum_k tri(A)(k,i) * B(k,j)
RIGHT, trans=N: C(i,j) = alpha * sum_k B(i,k) * tri(A)(k,j)
RIGHT, trans=T: C(i,j) = alpha * sum_k B(i,k) * tri(A)(j,k)
```

`tri(A)` 在 uplo 指定的三角之外为 0。`diag=UNIT` 时对角为 1。

主路径先构造列主序稠密矩阵 `P = alpha * tri(A)`，`ldp = k`，其中 LEFT 时 `k=m`，RIGHT 时 `k=n`。
随后：

```text
LEFT : C = P * B
RIGHT: C = B * P
```

转置已经吸收进 `tri(A)` 的读取下标，P 本身不再额外转置。列主序乘法等价于把缓冲看成转置后的行主序：

```text
C^T = B^T * P^T     (LEFT)
C^T = P^T * B^T     (RIGHT)
```

因此 GEMM 的逻辑 M、N 分别取 n、m。LEFT 时左矩阵是 B、右矩阵是 P；RIGHT 时左矩阵是 P、右矩阵是 B。
`trans=T` 时，参与转置的是打包后的那一侧。

### 支持数据类型

| 数据 | 类型 | 存储位置 | 说明 |
| --- | --- | --- | --- |
| alpha | FP32 | Host 或 Device | 调用前读入标量 |
| A | FP32 | GM | 三角矩阵，只读 |
| B | FP32 | GM | 一般矩阵，只读；与 C 同址时先拷贝 |
| C | FP32 | GM | 一般矩阵，写出 |
| P | FP32 | workspace | `alpha * tri(A)`，列主序，ldp=k |
| Cube 累加 | FP32 | L0C | `MatrixMadType::NORMAL`，Matmul 类型为 float |

### 支持形状

| 参数 | 物理范围 | 约束 |
| --- | --- | --- |
| A，LEFT | lda × m | lda ≥ max(1, m) |
| A，RIGHT | lda × n | lda ≥ max(1, n) |
| B、C | ldb × n、ldc × n | ldb ≥ max(1, m)，ldc ≥ max(1, m) |

m、n ≥ 0。维度由运行时参数给出。尾块由 tile 余数和带 pad 的搬运处理。

## 算子实现

### 总体架构

```text
aclblasStrmm
   |
   +-- 参数校验
   |
   +-- m=0 或 n=0 -----------------> 返回成功，不启动 Kernel
   |
   +-- alpha=0 --------------------> 异步清零 ldc×n 个元素，不读 A、B
   |
   +-- m≤48 且 n≤48 ---------------> AIV strmm_ref_kernel
   |
   +-- 其余
         |
         +-- B≡C 时先把 B 拷到 workspace
         |
         +-- LOWER + T + NON_UNIT
         |   且 |alpha|≥50
         |   且 m、n≤8192 ----------> AIV strmm_lt_kernel（参考 BLAS 累加顺序）
         |
         `-- 默认 ------------------> AIV 打包 P，再 AIC FP32 GEMM 写 C
```

三条计算路径都提交到同一个 stream。打包与 GEMM 按提交顺序在该 stream 上串行，Host 不在两次 launch 之间同步。

### Host 侧设计

代码在 `blas/trmm/arch22/strmm_host.cpp`。

#### 参数校验

1. handle 为空返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`。
2. side 只能是 LEFT/RIGHT，uplo 只能是 UPPER/LOWER，trans 只能是 N/T，diag 只能是 UNIT/NON_UNIT。
3. m、n 小于 0，或 lda/ldb/ldc 不满足上表，返回 `ACLBLAS_STATUS_INVALID_VALUE`。
4. m=0 或 n=0 直接成功。
5. alpha 为空，或需要读矩阵时 A、B、C 为空，返回 `ACLBLAS_STATUS_INVALID_VALUE`。
6. alpha 在 Device 上时，先异步拷回 Host 并同步该 stream，再选择后续路径。

`alpha=0` 时用 `aclrtMemsetAsync` 清零 `ldc * n` 个 FP32，覆盖列主序下 C 的全部已声明存储，包含 leading dimension 的 padding。

#### 路径选择

| 条件 | 路径 |
| --- | --- |
| m≤48 且 n≤48 | `strmm_ref_kernel`，单 block AIV |
| uplo=LOWER，trans=T，diag=NON_UNIT，`|alpha|≥50`，m≤8192，n≤8192 | `strmm_lt_kernel` |
| 其余非零计算 | `strmm_pack_kernel` 然后 `strmm_gemm_kernel` |

小形状阈值取 48，是因为官方性能用例从 256 起，都不进入这条路径。窄精度路径只包住实测会越出绝对误差上限的模式，五条性能用例的 alpha 都是 1，仍走打包加 GEMM。

#### Tiling

GEMM 使用 `MultiCoreMatmulTiling`，`SetDim(1)`，数据类型 DT_FLOAT，`SetMadType(NORMAL)`。基础块：

| 条件 | baseM × baseN × baseK |
| --- | --- |
| 交换后的两边都 ≥ 512 | 128 × 128 × 64 |
| 交换后的两边都 ≥ 128 | 64 × 128 × 64 |
| 更小 | 由 tiling 库自动选择 |

Host 按 baseM、baseN 数出 tile 个数，launch 的 AIC 数取 tile 数和物理 AIC 数的较小值。每个 AIC 以 grid-stride 领取 tile。K 维固定拆分请求不会覆盖库给出的 baseK，因此不再依赖 `SetFixSplit` 的 K 参数做精度修正。

打包按 A 的阶数分到 AIV，核数不超过 AIV 数量。`strmm_lt_kernel` 按 C 的列分到 AIV。

#### Workspace

默认 workspace 按 512 字节对齐串联：

```text
P            ldp * k * sizeof(float)          ldp = k
B 别名       B≡C 时 ldb * n * sizeof(float)
紧凑 B       RIGHT 且 ldb≠m 时 m * n * sizeof(float)
tiling       sizeof(TCubeTiling)
系统区       GetLibApiWorkSpaceSize() * AIC 数
```

RIGHT 且 `ldb≠m` 时，GEMM 右矩阵按紧凑列主序解释，所以先把 B 的每一列有效 m 行拷成紧凑缓冲。小形状路径只在 B≡C 时申请别名缓冲。

### Kernel 侧设计

#### 小形状 AIV

`strmm_ref_kernel` 在单 block 上按 Netlib 循环顺序直接写 C。对角、上/下三角和转置都在读 A 时裁剪。B≡C 时输入来自 workspace 中的副本，写出不会覆盖尚未读取的 B。

#### 打包

`strmm_pack_kernel` 把三角 A 展开成稠密列主序 P：

- 保留三角内元素，乘上 alpha；
- 对侧写 0；
- UNIT 对角先写成 1，再乘 alpha；
- 列内按最多 2048 个 FP32 分段，8 对齐的内部用向量清零，尾部标量补 0。

P 的前导维等于阶数，后续 GEMM 不再回读 A 的 padding。

#### FP32 GEMM

`strmm_gemm_kernel` 是 AIC-only。每个核从 GM 读 tiling，用自己的 workspace 切片构造 `Matmul<ND, ND, float>`。
tile 坐标 `(m0, n0)` 换算列主序偏移后 `SetSingleShape` 并 `IterateAll` 写到 C。尾块把 baseM、baseN 收到剩余行列。
写完后对 C 做数据缓存清理。

交换后的 orgM 是 n，orgN 是 m。LEFT 时 `transB` 表示 P 是否按转置读取；RIGHT 时 `transA` 表示 P 是否按转置读取。B 侧不转置。

#### 窄精度 AIV

`strmm_lt_kernel` 只服务 `LOWER + T + NON_UNIT`。UB 放两段长度 8192 的 FP32：累加列和 B 的一列。按列分核，只写每列的 m 个有效元素，不改 ldc padding。

LEFT 与参考 BLAS 一致，alpha 放在列累加之外：

```text
temp = B(i,j) * A(i,i)
temp += A(k,i) * B(k,j)    k = i+1 .. m-1
C(i,j) = alpha * temp
```

RIGHT 与参考 BLAS 一致，alpha 放进每一项，k 从对角向下递减：

```text
temp = (alpha * B(i,j)) * A(j,j)
temp += (alpha * A(j,k)) * B(i,k)    k = j-1 .. 0
```

RIGHT 在重新装入下一列 B 之前使用 `PipeBarrier<PIPE_ALL>`，避免向量路径上的缓冲复用和标量读交错。这条路径用标量乘加，不用向量 Axpy。向量 Axpy 的舍入与参考 BLAS 不一致，曾经把同一组用例的绝对误差放大到 10^3 量级。

## 支持硬件

| 支持的芯片版本 | 架构目录 | 涉及勾选 |
| --- | --- | --- |
| Atlas A2 系列产品（含 Atlas 800I/T A2） | arch22 / DAV_2201 | √ |
| Atlas A3 系列产品（含 Atlas 800I A3） | arch22 / DAV_2201 | √ |

构建与验收环境按任务书使用 CANN 9.1.0。自验证设备为 Ascend910_9382。任务书允许自验覆盖一种款型；正式验收仍需覆盖 Atlas 800T A2 (910B3) 与 Atlas 800I A3。

## 算子约束限制

1. 仅支持 FP32。
2. A、B、C 按列主序解释。
3. trans 只有 N/T。`ACLBLAS_OP_C` 返回非法参数。
4. 不支持 lda/ldb/ldc 语义之外的非连续访问，不支持广播。
5. 只允许 B 与 C 同址。A 与 B、A 与 C 重叠的结果未定义。
6. 调用方保证指针、shape 和前导维对应的显存有效。
7. 一次调用只使用 handle 所在设备。
8. API 异步返回。读回 C 之前须同步 handle 绑定的 stream。
9. 浮点累加顺序不保证与参考 BLAS 逐位一致。窄精度路径为了对齐绝对误差上限，才使用参考 BLAS 的标量顺序。

# 可维可测分析

## 可维护性分析

- 实现放在 `blas/trmm/arch22/`，不改其他架构目录和公共签名。
- Host 负责校验、路由和 workspace；打包、GEMM、小形状、窄精度各在独立 kernel。
- 窄精度路径的入口条件写在 Host 一处。不满足条件的输入仍走打包加 GEMM。
- tiling 只传维度、转置、前导维和核数，不依赖用例编号。

## 可测试性分析

测试在 `test/trmm/strmm/arch22/`，CSV 驱动 GTest，golden 为 cblas `strmm`。任务书 1200 条保持原样，另补 A 为 Inf、A 为 NaN 两条，以及三个 GTest 固定用例：空 handle、alpha=0 的 padding、B≡C。

| 类别 | 前缀 | 覆盖内容 |
| --- | --- | --- |
| 基础 | TC_L0 | 小形状与四种属性的基本组合 |
| 尺寸 | TC_SQ、TC_WS、TC_TH | 方阵、宽矩阵、高矩阵 |
| 标量 | TC_AB | 0、1、-1 和一般 alpha |
| 前导维 | TC_LD | 紧凑与 padding |
| 填充 | TC_FL | Inf、NaN、0 填充 |
| 覆盖与边界 | TC_CV、TC_ED | 枚举组合、零维、空指针、非法枚举、负维度、非法前导维 |
| 扩展 | TC_EX | 大规模采样，含窄精度路径 |
| 性能 | TC_PF | 性能形状的精度 |
| 补充 | TC_SP | A 为 Inf、A 为 NaN |

测试框架的数据填充名是均匀分布 `RANDOM_NORM_5_5`。任务书写的正态 50% 在该框架里没有独立生成器，自测报告中单独说明。

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | 逐元素 `|actual-golden| ≤ atol + rtol×|golden|`；整体 `matched_ratio ≥ 0.99`，且没有元素超过 `max(1e-2, 32*ULP)`。alpha=0 时 C 逐位为 0 | 任务书混合容差；判定实现采用 ops-blas 测试框架 |
| 性能标准 | 五条官方形状的截尾均值不高于 `gpu_ms / 0.8` | 任务书 §3.3 |

任务书表格中的 FLOAT32 rtol、atol 写为 `2^-13`。仓内测试框架实际打印并执行的 rtol 是 `2^-10`，atol 约为 `1.53e-5`，匹配比例和绝对误差上限与任务书一致。自测没有改宽松 `verify.h`。

性能计时在本机不能使用 `msprof op`：profiling 通道拿到的是物理 id，而运行时只接受逻辑 id。计时改为系统时间线，70 次 launch，去掉前 10 次，再去掉剩余样本中的 10 个最大值，对 50 次取平均。有效样本数大于任务书要求的 10 次。耗时是同一次调用里 pack kernel 与 gemm kernel 之和。

任务书五条门槛与 Ascend910_9382 上的自测截尾均值：

| 规模 | 配置 | 门槛 us | 自测 trim_us |
| --- | --- | --- | --- |
| 256 | LEFT UPPER N NON_UNIT | 58.429 | 9.342 |
| 512 | LEFT LOWER T NON_UNIT | 138.2 | 15.716 |
| 1024 | LEFT UPPER N UNIT | 342.2 | 44.689 |
| 2048 | RIGHT LOWER N NON_UNIT | 1182 | 227.195 |
| 4096 | LEFT UPPER T NON_UNIT | 6681 | 1838.302 |

同一设备上全量 GTest 为 1205 条，1205 条通过。其中精度 CSV（TC_L0 至 TC_EX）1000/1000，TC_PF 200/200，TC_SP 2/2。

## 兼容性分析

- 函数签名、枚举和状态码保持 `cann_ops_blas.h` 的现有定义。
- 新增实现只进入 arch22 构建，不替换其他产品的 TRMM。
- 与 cuBLAS 的差异只有一项：实数 `OP_C` 返回非法参数，而不是当成转置。这一点按任务书 §2.4 实现。
- B≡C、m/n 为 0、alpha=0 不引用 A/B，与任务书和 cblas 边界一致。
- 依赖 CANN 9.1.0 的 Ascend C 与 Matmul tiling。

## 参考资料

1. [CANN 社区任务流程及注意事项](https://gitcode.com/org/cann/discussions/39)
2. [ops-blas 开源仓](https://gitcode.com/cann/ops-blas)
3. [cuBLAS cublasStrmm](https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-trmm)
4. [Netlib STRMM](https://www.netlib.org/blas/strmm.f)
5. [生态算子开源精度标准](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/mixed_tolerance_standard.md)
