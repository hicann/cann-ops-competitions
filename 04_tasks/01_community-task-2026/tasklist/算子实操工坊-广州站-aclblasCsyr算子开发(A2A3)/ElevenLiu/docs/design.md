# 需求背景(required)

## 需求来源

本需求来源于 gitcode 竞赛社区任务《算子实操工坊-广州站-aclblasCsyr算子开发(A2A3)》，要求在昇腾 NPU（Atlas A2/A3 系列产品）上使用 Ascend C/CATLASS 编程语言开发单精度复数对称秩-1 更新算子 `aclblasCsyr`，完成算子设计、开发、测试全流程工作，验收通过后合入昇腾算子开源仓 ops-blas（`blas/syr/arch22/` 目录）。

## 背景介绍

对称秩-1 更新（symmetric rank-1 update）是 BLAS 二级运算中的基础算子，完成 `A = alpha * x * x^T + A` 的原地更新，广泛应用于迭代法求解线性方程组、特征值分解、矩阵修正等数值计算场景。对标基线接口为 cuBLAS 的 `cublasCsyr`：其中 x 为 n 元素单精度复数向量，A 为 n×n 对称复数矩阵（列主序存储），alpha 为复数标量。需要特别强调两点语义：

1. **对称（symmetric）而非厄米特（Hermitian）**：更新使用 x 的普通转置 `x^T`，**不共轭**，区别于厄米特秩-1 更新 `cublasCher` 的 `x^H`；对角元素为普通复数元素，虚部不作任何假定或置零处理。复数 `syr` 在 Netlib BLAS 中无参考实现（仅有实数 `ssyr`），复数语义以 cuBLAS 文档为准。
2. **三角引用**：A 仅 `uplo` 指定的上三角（ACLBLAS_UPPER）或下三角（ACLBLAS_LOWER）被引用和更新，未引用三角不读不写，由对称性隐含。

# 需求分析(required)

## 需求描述

在 Atlas A2/A3 产品上基于 ops-blas 开源仓工程框架，以 Ascend C **kernel 直调**方式实现 `aclblasCsyr` 句柄式 BLAS 接口，通过 handle 绑定 stream 直调 NPU kernel。接口声明新增于 `include/cann_ops_blas.h`（仓内已有实数同族算子 `aclblasSsyr`，本算子将 alpha/x/A 扩展为 `aclblasComplex`，即实部/虚部各 float32），签名与 cuBLAS `cublasCsyr` 逐参数对应：

```cpp
aclblasStatus_t aclblasCsyr(
    aclblasHandle_t handle, aclblasFillMode_t uplo, const int n,
    const aclblasComplex* alpha,
    const aclblasComplex* x, const int incx,
    aclblasComplex* A, const int lda);
```

## 需求拆解

1. **功能语义**：`A(i,j) += alpha * x(i) * x(j)`（不共轭），仅更新 uplo 指定三角；支持任意非零 incx（含负步长，按 Netlib `ssyr` 语义反向遍历 x）；`n = 0` 或 `alpha = (0,0)` 时为合法 no-op（quick return，返回 `ACLBLAS_STATUS_SUCCESS`，不更新 A）。
2. **参数校验与错误码**：handle 为 nullptr 返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`；n < 0、incx = 0、lda < max(1,n)、alpha 为 nullptr、n > 0 且 alpha ≠ (0,0) 时 x/A 为 nullptr 返回 `ACLBLAS_STATUS_INVALID_VALUE`；uplo 取值不在 {ACLBLAS_UPPER, ACLBLAS_LOWER} 时返回 `ACLBLAS_STATUS_INVALID_ENUM`。状态码语义与仓 `include/cann_ops_blas_common.h` 定义一致。
3. **精度**：按生态算子开源精度标准（任务书 §3.2），实部/虚部按 FLOAT32 分量分别判定。
4. **性能**：在 Atlas 800I A2 (910B3) 上，各性能 case 的平均单次耗时不高于任务书 §3.3 标杆耗时。
5. **工程化**：接口声明放入公共头文件供多产品线共用，禁止产品私有平行接口；按 ops-blas 测试框架（CSV 用例 + GTest + 自实现复数 golden）完成自测并输出自测报告；README 产品支持表标注 Atlas A2/A3 支持。

# 详细设计(required)

## 算子分析

### 数学公式

对标 cuBLAS `cublasCsyr`，对 uplo 指定三角（含对角）逐元素执行：

```
A(i,j) = alpha * x(i) * x(j) + A(i,j)
```

其中 uplo = UPPER 时 0 ≤ i ≤ j < n，uplo = LOWER 时 0 ≤ j ≤ i < n；共更新 n(n+1)/2 个复数元素。按复数运算展开（alpha = a_r + a_i·i，x(i) = p + q·i，x(j) = s + t·i）：

```
实部: A_r(i,j) += (a_r·p − a_i·q)·s − (a_r·q + a_i·p)·t
虚部: A_i(i,j) += (a_r·q + a_i·p)·s + (a_r·p − a_i·q)·t
```

可先对每个 x(i) 预计算 w(i) = alpha·x(i)（一次复数乘法），将更新简化为 `A(i,j) += w(i)·x(j)`（其中 w = alpha·x），把 alpha 复数乘的开销从每元素降为每向量元素一次。

### 支持数据类型

| 数据 | 类型 |
| --- | --- |
| x、A、alpha | COMPLEX64（aclblasComplex，实部/虚部各 float32） |
| n、incx、lda | int32 |
| uplo | 枚举 int（ACLBLAS_UPPER / ACLBLAS_LOWER） |

### 支持形状

- x：ND，逻辑长度 n，物理长度至少 `1 + (n-1) × |incx|`；
- A：ND（列主序），形状 (lda, n)，原地更新；
- 约束：n ≥ 0，incx ≠ 0，lda ≥ max(1, n)。

## 算子实现

### 实现方案

总体思路：计算访存比低（每更新一个元素仅需一次复数乘加，却需读写一次 A），属于访存受限算子。设计上以"**按列分片 + 三角形负载均衡 + UB 内列缓冲 + 向量化复数乘加**"为主线，核间无写冲突、核内顺序流水。

#### host侧设计：

##### 1. 分核策略：

目标硬件为 40 核 AIV 并行。若按列区间连续切分（core k 负责 [k·n/N, (k+1)·n/N) 列），因三角形分布下列长随列号递减/递增，各核工作量严重不均。设计采用**按列成本均衡的贪心分片**：按列处理成本（UPPER 列长 j+1 递增、LOWER 列长 n-j 递减）将列依序分配给当前累计成本最小的核，使每核持有"长列与短列混合"的列集合，总元素数近似均衡（偏差不超过单列元素数级别）。对角元素（i == j 的自更新项）随所属列一并处理，无需单独分核。incx/负步长不影响分核，仅影响 x 元素到 `x(i)` 的地址映射，在核内统一处理。

##### 2. 数据分块和内存优化策略：

- **x 向量缓冲**：kernel 启动后将本核所需的 x 元素（以 incx 步长 Gather 或按连续段搬移）载入 UB，一次载入、多列复用，避免每列重复访存；n 较大时按 UB 容量对 x 分块，外层循环按块推进。
- **列粒度分块**：每次处理 `colsPerTile` 列（由 UB 容量与列长上限共同确定，保证 `colsPerTile × n_max × sizeof(complex64)` 不超 UB 预算），每列循环内完成"计算该列乘积增量 → 原子累加写回 A 列"。
- **复数乘加向量化**：complex64 按实部/虚部交织存储，向量运算时用向量乘/加指令成对处理实虚部（预计算 w = alpha·x 后，每元素退化为一次复数乘与一次复数加，即 4 次实数乘 + 2 次实数加），充分利用 128B 对齐与双发射向量单元；对角元素的虚部按普通复数元素处理，不置零。
- **写回**：按列独占分核使每列只被一个核处理，列内多个输出组以**向量原子加（SetAtomicAdd）写回**，仅写 uplo 指定三角区域，未引用三角不读不写；原子加消除核内写序敏感，无需额外同步。

##### 3. 路径分派策略（kernel 直调，无 TilingKey）

本算子采用 kernel 直调模式，host 侧不经过 TilingKey/框架二进制匹配流程，而由 **tiling 标量字段承载全部分派信息**：host 侧根据 (uplo, n, incx, alpha 是否为零) 计算 tiling 结构（uplo 方向、实际启用核数、列块大小、x 分块参数、quick return 标志等），序列化后随启动参数下发；kernel 侧读取标量字段完成分支分派。uplo 上下三角通过列内行区间的起止表达式统一（以对角线为界的下标比较方向翻转），单一 kernel 模板即可覆盖两种填充模式，避免维护两份代码。`n = 0`、`alpha = (0,0)` 的 quick return 在 host 侧短路：完成参数校验后直接返回 `ACLBLAS_STATUS_SUCCESS`，不启动 kernel。

#### kernel侧设计：

- **入口**：解析 tiling 标量，按 `block_idx` 获取本核列集合，quick return 标志置位时直接退出。
- **主循环**：① 载入 alpha 并校验；② 预计算 w = alpha·x（本核 x 分块，向量化复数乘）；③ 逐列：按 uplo 确定该列行区间 [lo, hi]，执行 `A(i,j) += w(i)·x(j)` 的向量化复数乘加，乘积增量以原子加写回 GM（无需读入 A 旧值）；负步长时 x 地址按 `x_base − k·|incx|·sizeof(complex64)` 反向映射。
- **边界处理**：列长不足向量宽度时按尾部标量/短向量处理；n = 1 时退化为单元素对角更新。
- **确定性**：无跨核归约，结果仅依赖单核内的顺序乘加，浮点结果确定。

## 支持硬件

Atlas A2 系列产品（性能测试设备：Atlas 800T A2 (910B3)）与 Atlas A3 系列产品，对应架构目录 `blas/syr/arch22/`；CANN 9.1.0。

## 算子约束限制

- n ≥ 0；本设计支持规模上限取 n ≤ 4864（覆盖任务书全部性能 case 与自测大规模用例），超出该规模的合法性由调用方保证；
- incx ≠ 0，支持任意非零步长（含负步长）；lda ≥ max(1, n)；
- alpha 为复数标量且指针不可为 nullptr；`alpha = (0,0)` 或 `n = 0` 为 no-op；
- 仅支持 ND 列主序存储与 incx/lda 语义内的访问，不支持超出该语义的非连续 Tensor；不涉及广播；A 原地更新，确定性计算不作要求；
- 仅引用/更新 uplo 指定三角，另一三角不读不写；更新不共轭，对角虚部不作特殊处理。

# 可维可测分析

## 精度标准/性能标准

**精度标准**：按任务书 §3.2 执行。golden 由测试工程内自实现的复数对称秩-1 更新参考代码生成（`A(i,j) += alpha·x(i)·x(j)` 逐元素更新 uplo 指定三角，不共轭，对角虚部不作特殊处理），单标杆比对，验证范围为 uplo 指定三角区域（含对角），实部、虚部分别按 FLOAT32 标准判定：

| 数据类型 | rtol | atol | required_matched_ratio | max_abs_error_limit |
|----------|------|------|------------------------|---------------------|
| COMPLEX64（实部/虚部按 FLOAT32 分量） | 2^-10 (9.77e-4) | 2^-16 (1.53e-5) | 0.99 | 1e-2 或 32 * ULP |

逐元素通过条件：`|actual - golden| ≤ atol + rtol × |golden|`；用例同时满足 matched_ratio ≥ required_matched_ratio 且 max_abs_error ≤ max_abs_error_limit 时判定通过。实测结果见自测报告。

**性能标准**：按任务书 §3.3 执行，测试设备 Atlas 800I A2 (910B3)，COMPLEX64 输入，先 warmup 再有效采样 >50 次取平均，各 case 平均单次耗时不高于下表标杆耗时（实测见自测报告）：

| case | n | uplo | incx | alpha | 标杆耗时（Avg time，us） |
|---|---|---|---|---|---|
| 1 | 512 | UPPER | 1 | (1.0, 0.0) | 6.76 |
| 2 | 1024 | LOWER | 1 | (1.0, 0.0) | 8.91 |
| 3 | 2048 | UPPER | 1 | (1.0, 0.0) | 17.57 |
| 4 | 4096 | LOWER | 1 | (1.0, 0.0) | 116.93 |

**可维可测手段**：自测基于 ops-blas 仓 test 框架（CSV 用例描述 + Python 脚本调用 C++ GTest 工程执行），用例覆盖 uplo 双模式全切换、n ∈ {0, 1, 小质数, 2 的幂及 ±1, 非对齐值, 大规模}、incx ∈ {±1, ±2, ±3} 组合、lda 紧凑与 padding 场景、alpha/x/A 的均匀与正态分布及全零/交替/Inf/NaN 特殊值、quick return 与全部负向错误码用例（n < 0、incx = 0、lda 越界、非法枚举、空指针）。

## 兼容性分析

1. **接口兼容**：参数序列与 cuBLAS `cublasCsyr` 一一对应，无需映射说明；与仓内实数同族算子 `aclblasSsyr` 同构（alpha/x/A 扩展为 `aclblasComplex`），命名与状态码语义沿用 `cann_ops_blas_common.h`，不引入产品私有平行 API，接口声明位于公共头 `include/cann_ops_blas.h` 供多产品线共用。
2. **工程兼容**：本算子为新增接口，不修改既有算子声明与行为，对仓内已有功能无前向/后向影响；实现目录与实数同族算子同目录族（`blas/syr/arch22/`），测试代码合入 `test/syr/csyr/arch22/`，文件结构对齐主仓 BLAS 算子测试代码。
3. **环境兼容**：基于 CANN 9.1.0 与 Ascend C kernel 直调模式开发，无新增三方软件依赖（golden 随测试工程自实现提供）；异步执行依赖 `aclblasSetStream` 绑定 stream，读回 Device 结果前须同步 stream，与仓内句柄式 BLAS 接口的既定使用方式一致。
