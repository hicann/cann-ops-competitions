# 需求背景（required）

## 需求来源

昇腾社区任务 2026 年社区任务 - aclblasCtpsv 算子开发（Ascend 950PR）。

## 背景介绍

### aclblasCtpsv 算子实现

基于 ops-blas 开源仓既有实数同族算子 `aclblasStpsv`（`blas/tpsv/arch35/`）的工程模式，使用 Ascend C 编程语言实现单精度复数（complex64）三角压缩存储（packed）求解算子 `aclblasCtpsv`，通过 handle 绑定 stream 直调 NPU kernel，适配 Ascend 950PR。

aclblasCtpsv 对标 cuBLAS `cublasCtpsv` 与 Netlib BLAS `ctpsv.f`，为昇腾 NPU 提供复数三角线性系统求解能力。三角求解是 LU/Cholesky 分解回代、最小二乘、稀疏直接法等算法的基本构建块。

ops-blas 仓接口声明位于 `include/cann_ops_blas.h`，与实数 `aclblasStpsv` 逐参数对应，元素类型扩展为 `aclblasComplex`，不引入 950PR 私有平行接口。

### aclblasCtpsv 算子功能分析

求解 `op(A) * x = b`，其中 A 为 n×n 三角矩阵，以 packed 格式存储（仅存三角部分，共 `n(n+1)/2` 个复数元素，逐列无间隙）。入口时 x 存放右端向量 b，出口时解向量**原地覆写** x。

| 参数 | 参数含义 | 数据类型 | 支持数据类型 | 约束 | 形状 |
| --- | --- | --- | --- | --- | --- |
| handle | 库上下文句柄，携带 stream | aclblasHandle_t | - | 必须为有效句柄 | - |
| uplo | A 的三角存储模式 | aclblasFillMode_t | {UPPER, LOWER} | 枚举合法 | - |
| trans | 方程形态 | aclblasOperation_t | {OP_N, OP_T, OP_C} | 枚举合法 | - |
| diag | 对角处理方式 | aclblasDiagType_t | {NON_UNIT, UNIT} | 枚举合法 | - |
| n | 矩阵阶数 | int | - | n >= 0；n = 0 为合法 no-op | - |
| AP | packed 三角矩阵 A，只读 | const aclblasComplex* | complex64 | n > 0 时非空 | n(n+1)/2 个元素 |
| x | 右端 b / 解 x（原地覆写） | aclblasComplex* | complex64 | n > 0 时非空 | 逻辑 n，物理 1+(n-1)·\|incx\| |
| incx | x 的存储增量 | int | - | incx != 0，可正可负 | - |

数学语义分支：

- `trans = ACLBLAS_OP_N`：`op(A) = A`
- `trans = ACLBLAS_OP_T`：`op(A) = A^T`（转置）
- `trans = ACLBLAS_OP_C`：`op(A) = A^H`（共轭转置）

packed 存储格式（0 基索引）：

- `uplo = ACLBLAS_UPPER`：A(i, j)（i <= j）存放于 `AP[i + j*(j+1)/2]`
- `uplo = ACLBLAS_LOWER`：A(i, j)（i >= j）存放于 `AP[i + (2*n - j - 1)*j/2]`

未引用的另一半三角不读取，由三角性隐含。`diag = ACLBLAS_UNIT` 时对角元素不被访问、假定恒为 (1, 0)；`diag = ACLBLAS_NON_UNIT` 时对角元素从 AP 读取。本算子**不做奇异或近奇异检测**（与 cuBLAS 一致），`NON_UNIT` 时要求调用方保证对角元素非零。

> 说明：任务书 §2.2 给出的 LOWER 索引写法 `AP[i + ((2*n-j+1)*j)/2]` 与 Netlib/JIS packed 约定不符（n=3、j=1、i=2 时代入得 5，实际应为 4）。本实现按 Netlib `ctpsv.f` 与仓内 `aclblasStpsv` 的 `i + (2*n-j-1)*j/2` 实现，并由 cblas（Netlib）单标杆逐元素验证一致。

# 需求分析（required）

## 需求描述

使用 Ascend C 编程语言实现 aclblasCtpsv 算子，支持 complex64 数据类型与 uplo×trans×diag 共 12 组枚举组合，支持正负 incx 步长，完成算子设计、开发、测试全流程。性能不低于任务书 §3.3 给出的 4 个典型 case 标杆耗时。

## 需求拆解

1. 支持 complex64（实部/虚部各 float32），覆盖 uplo×trans×diag 共 12 组枚举组合
2. packed 存储无前导维（lda）参数，仅引用 uplo 指定三角
3. 参数校验口径与运行库一致：非法枚举 / 负维度 / 零步长 / 空指针返回 `ACLBLAS_STATUS_INVALID_VALUE`，handle 为空返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`，n = 0 为合法 no-op
4. x 原地覆写，支持 incx 正负步长（incx < 0 反向存储）
5. 复数除法数值稳定：极端量级输入下不出现中间积溢出
6. 性能不低于 cuBLAS 对标版本（任务书 §3.3 四个 case）

# 详细设计（required）

## 算子分析

### 数学公式

求解 `op(A) * x = b`，三种 trans 分别对应：

- `OP_N`：A·x = b
- `OP_T`：A^T·x = b
- `OP_C`：A^H·x = b，其中 A^H(i, j) = conj(A(j, i))

三角求解为代入过程。以 LOWER + OP_N 的前向代入为例：

```
x[0] = b[0] / A(0,0)
x[i] = (b[i] - Σ_{j<i} A(i,j)·x[j]) / A(i,i),  i = 1..n-1
```

复数域除法（缩放式，见 3.2.2 数值稳定性）：

```
(a + bi) / (c + di),  |c| >= |d|:  r = d/c, t = 1/(c + d·r)
  Re = a·t + (b·t)·r,  Im = b·t - (a·t)·r
```

### 支持数据类型

complex64（`aclblasComplex`，实部/虚部各 float32）。

### 支持形状

- AP：packed 一维向量，长度 n(n+1)/2，无前导维
- x：逻辑长度 n，物理长度 `1 + (n-1)·|incx|`；incx < 0 时逻辑元素 i 位于物理位置 `(n-1-i)·|incx|`（与 Netlib 步长约定一致）
- 不涉及广播；非连续 Tensor 仅支持 incx 描述的步长访问；n 为运行时入参，不要求 dynamic shape

## 算子实现

### 实现方案

#### 3.2.1 host 侧设计：

**1. 参数校验与启动边界**

校验顺序与 `aclblasStpsv` 保持一致：handle 非空 → `n >= 0` → `n == 0` 直接返回成功（合法 no-op，不校验 AP/x）→ uplo/trans/diag 枚举 → `incx != 0` → `n > 0` 时 AP/x 非空。所有异常返回 `ACLBLAS_STATUS_INVALID_VALUE`，handle 为空返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`。

**2. tiling 策略**

本算子无需分块与多核切分：三角求解的代入过程存在严格串行依赖（第 s 步依赖前 s-1 步的结果），因此采用单 block、SIMT VF 多线程按步推进的结构。tiling 字段如下（`ctpsv_tiling_data.h`）：

| 字段 | 含义 |
| --- | --- |
| ap / x | AP 与 x 的 device 指针 |
| n | 矩阵阶数 |
| uplo / trans / diag | 分派用属性（复刻 cuBLAS 语义） |
| incx | x 步长（int64） |
| numThreads | SIMT 线程数；0 表示走 scalar 路径 |

**3. 执行路径选择**

| 条件 | 路径 |
| --- | --- |
| n < 128 | scalar 路径：单线程顺序代入，避免 SIMT 启动开销 |
| n >= 128 | SIMT VF 路径：`asc_vf_call` 多线程按步推进 |

**4. 线程数策略（`SelectThreadCount`）**

单个求解步覆盖的元素数不超过 n，因此可用并行度受步长而非 n 约束，且两类步型的伸缩性不同：

- `trans = OP_N`：步内为流式向量更新 `x[r] -= A(r,s)·x[s]`，每线程一个复数乘加，取线程数 `min(align(n,128), 2048)`（与 `aclblasStpsv` 一致）
- `trans = OP_T/OP_C`：步内为点积，每步都需要一次跨线程归约，线程数越多归约开销越大，按平均步长 n/2 上每线程约 3 个元素取 `round128(n/6)`，上限 1024

实测（Ascend 950PR）该策略使 n = 512/1024/2048/4096 四个验收 case 分别落在 512/1024/384/640 线程的最优区间。

**5. kernel 分派**

按 (uplo, trans, diag) 分派到 12 个 kernel 入口（trans 的 T 与 C 共用转置步型、以 CONJ 模板参数区分共轭），scalar 与 SIMT 路径各一套。分派为编译期模板实例化，步内无运行期分支。

#### 3.2.2 kernel 侧设计：

**1. 步型抽象（`ctpsv_kernel_utils.h`）**

packed 布局与三角结构使得每一步都落在 AP 中一段**连续**的区间上，且该区间对应的 x 下标同样连续。四种 (uplo, trans) 组合的几何关系只由 uplo 决定：

| uplo | 第 s 步的 AP 区间 | x 下标区间 |
| --- | --- | --- |
| UPPER | `[s*(s+1)/2, +s)` | `[0, s)` |
| LOWER | `[idxL(s+1,s), +n-1-s)` | `[s+1, n)` |

由此每步归为两种形态：

- **scatter（trans = OP_N，右看式）**：解出列 s 的 `t = x[s]/A(s,s)`，再对区间内所有 r 执行 `x[r] -= A(r,s)·t`。连续访存、无需归约。
- **gather（trans = OP_T/OP_C，左看式）**：`x[s] = (x[s] - Σ conj(A(j,s))·x[j]) / conj(A(s,s))`，区间内为连续访存，但需要逐步归约。

扫描方向 `kForward = (LOWER && OP_N) || (UPPER && OP_T/OP_C)`，与 `aclblasStpsv` 一致。

**2. scalar 路径（n < 128）**

单线程按上述两种形态顺序代入，代码与 SIMT 路径共用同一套几何与复数运算辅助函数，避免两条路径语义漂移。

**3. SIMT 路径（n >= 128）**

线程按 `l = threadIdx.x; l += blockDim.x` 在步内区间上分段，天然合并访存。

- **scatter：每步 1 次 barrier**。pivot 的写回采用软件流水——`t` 保存在寄存器中，于**下一步开头**（barrier 之后）由 0 号线程写回 x[s]。因上一步的下标不在本步区间内，该写回与任何读改写均不冲突。
- **gather：每步 1 次 barrier + 两级归约**。上一拍的 pivot 位于本步区间的固定槽位（UPPER 前向为末槽、LOWER 后向为首槽），各线程从**寄存器**携带该值并自行折算，因此不需要"解出 pivot → 广播 → 下一步使用"之间的 barrier。

归约分两级：warp 内 `asc_shfl_down` 蝶形（5 轮）→ 各 warp 由 lane 0 写入 UB → barrier → 跨 warp 归约后再用 `asc_shfl(..., 0)` 广播 lane 0 的结果。**跨 warp 阶段必须广播**：蝶形归约逐 lane 舍入不同，若各线程直接把"自己那份"总和的舍入结果折回下一步点积，结果会随线程调度而变（实现过程中曾出现 1000 条用例中 69~77 条随机失败的竞态）。

**4. x 暂存 UB 优化**

x 每步被访问一次，整体 O(n²) 次；AP 仅流式读取一遍。因此 n <= 4096 时将 x（按逻辑序，步长映射在装载/回写时完成）暂存于 UB，步内只从 GM 流式读 AP，内层循环去掉一次 GM 往返与 incx 乘加。n > 4096 时回退为直接访问 GM（几何已含步长）。

**5. 数值稳定性（复数除法）**

复数除法采用缩放式 `t = 1/(c + d·r)`，先缩放被除数再合并：

- 朴素式 `(a·c + b·d)/(c² + d²)`：被除数量级接近 fp32 上限时，`a·c` 本身就溢出为 Inf，即使商可表示（例如 b 含 FLT_MAX、对角约 30 时商为 1e37，朴素式得 Inf/NaN）
- 仓内 BLAS 的 Netlib `cdiv` 形式 `((a + b·r)·t, (b - a·r)·t)`：`b - a·r` 在 a、b 同量级时同样会溢出（本例中为 3.78e38 > FLT_MAX）
- 本实现先按 t 缩放：`((a·t) + (b·t)·r, (b·t) - (a·t)·r)`，中间量被 t 压回量程，与 Netlib cblas 结果一致（经双精度 ground truth 复核）

该缺陷由 CSV 中 x 含 ±FLT_MAX 的用例（TC_FL_146）暴露，修复后该用例通过。

**6. 数据流**

```
host: aclblasCtpsv 校验 → 填 tiling → ctpsv_kernel_do
        ├─ n < 128:  scalar kernel（模板实例化，按 uplo×trans×diag 分派）
        └─ n >= 128: ctpsv_simt_kernel → asc_vf_call<CtpsvSimt<...>>
                        ├─ n <= 4096: x 载入 UB → 逐步求解 → 回写 GM
                        └─ n >  4096: 直接按 incx 访问 GM
```

## 支持硬件

| 支持的芯片版本 | 涉及勾选 | 说明 |
| --- | --- | --- |
| Ascend 950PR | √ | 本任务适配与验证平台（arch35，CANN 9.1.0） |
| Ascend 950DT | √ | 与 950PR 同属 arch35，本期未单独上机验证 |
| Atlas A3 训练/推理系列产品 | 不支持 | |
| Atlas A2 训练/推理系列产品 | 不支持 | |

## 算子约束限制

- 仅支持 complex64；不做奇异/近奇异检测，`NON_UNIT` 时调用方须保证对角元素非零
- 仅支持 incx 描述的一维步长访问，不支持其它非连续内存布局；不涉及广播
- 不做确定性要求（本实现实际为确定性实现：归约顺序固定）
- n = 0 为合法 no-op

# 可维可测分析

## 精度标准/性能标准

### 精度标准

golden 由 cblas（Netlib BLAS 复数实现 `cblas_ctpsv`）单标杆生成，输出向量 x 的实部、虚部分别按 FLOAT32 判定（与仓内 `stpsv`/`strsv` 测试口径一致）：

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 逐元素 | `\|actual - golden\| <= atol + rtol·\|golden\|`，atol = 2⁻¹⁶，rtol = 2⁻¹⁰ | 生态算子开源精度标准（任务书 §3.2） |
| 用例 | matched_ratio >= 0.99 且逐元素绝对误差 <= max(1e-2, 32·ULP) | 同上 |

测试工程对求解类做如下前置保障（两侧输入严格一致）：

- `NON_UNIT`：填充 AP 后对被引用的对角元加偏移 `boost = max(5, n)`，保证求解良态（沿用仓内 `stpsv_test.cpp` 做法）
- `UNIT`：对角不被引用，将 AP 按 `min(1, 5/n)` 缩放并归一化到最大幅值 1（ALTER/EXTREME 填充会达到数百乃至 FLT_MAX，不归一化会使 fp32 标杆本身溢出）
- 特殊值：实部/虚部分别比较，两侧同为非有限值（NaN/±Inf）时该元素跳过——kernel 与 cblas 的复数除法公式与求和顺序不同，退化输入落入 {NaN, ±Inf} 中的哪一个不属于算子契约；仅一侧非有限则判失败

自验结果：**精度用例 1000/1000 通过（连续 3 轮）**，性能/内存用例 200/200 通过。

### 性能标准

采样口径：先 warmup 5 次，再有效采样 51 次取平均（`avg_us` 含 kernel 与单次下发开销）。

| case | n | uplo | trans | diag | incx | 标杆耗时 (us) | 实测 (us) | 结果 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 512 | LOWER | N | NON_UNIT | 1 | 971.85 | 404.5 | 2.40× |
| 2 | 1024 | UPPER | N | NON_UNIT | 1 | 2074.42 | 866.5 | 2.39× |
| 3 | 2048 | LOWER | T | NON_UNIT | 1 | 4012.85 | 3522.0 | 1.14× |
| 4 | 4096 | UPPER | C | NON_UNIT | 1 | 12481.55 | 10744.8 | 1.16× |

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | 同 4.1 精度标准表 | 生态算子开源精度标准 |
| 性能标准 | 平均单次耗时 <= 任务书 §3.3 标杆耗时（等价于 gpu_baseline 的 ratio >= 0.4） | 任务书 §3.3 |

> 性能用例耗时以测试工程打印的 `[PERF] ... avg_us=...` 为准。GTest 自身打印的用例耗时覆盖 warmup 与全部采样，不能直接作为单次耗时；配套 `verify_performance.py` 以 GTest 耗时解析，两种口径均无法与其对齐，已在测试目录 README 中说明。

## 兼容性分析

新算子，不新增/修改既有接口，不涉及兼容性分析。接口声明位于 `include/cann_ops_blas.h`，与同族 `aclblasStpsv` 逐参数对应，可供其它产品线共用。
