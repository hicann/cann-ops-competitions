# aclblasStbsv 算子设计文档

> 任务：2026 年 9 月社区任务 aclblasStbsv（Atlas A2/A3）
> 目标仓：`cann/ops-blas`，个人开发仓：`Helloyouth_git/ops-blas`
> 目标架构：Ascend 910B / Atlas A2、Atlas A3，ops-blas 架构目录 `arch22`
> 软件环境：CANN 9.1.0，Ascend C

## 一、需求背景

### 1.1 需求来源

本设计依据《9月社区任务-aclblasStbsv算子开发(A2A3)/aclblasStbsv_A2A3_task_doc.md》，在
ops-blas 中实现单精度实数三角带状线性方程组求解接口 `aclblasStbsv`。目标是与 cuBLAS
`cublasStbsv` 的列主序带状矩阵、转置、单位对角和向量步长语义对齐，并满足任务规定的功能、精度和性能验收。

### 1.2 算子背景与参考实现

`stbsv` 求解 `op(A) * x = b`，右端向量 `b` 初始存放在 `x` 中，求得的解原位写回 `x`。矩阵为
`n×n` 三角带状矩阵，只存储所选三角侧主对角及其 `k` 条相邻对角，使用列主序带状格式；`k=0` 时退化为
对角求解，`n=0` 时为成功 no-op。

仓库检查得到的参考实现和适配边界如下：

| 参考 | 文件/路径 | 对本设计的用途 |
| --- | --- | --- |
| ops-blas 三角带求解实现 | `blas/stbsv/arch35/stbsv_kernel.cpp`、`stbsv_kernel_simt.cpp` | 带状下标、负 `incx` 逻辑、前代/回代语义；SIMT 核作为协作式求解的实现参考。 |
| ops-blas A2 三角求解实现 | `blas/trsv/arch22/strsv_host.cpp`、`strsv_kernel.cpp` | arch22 Host/Kernel 接口与 ASC 构建布局参考；该实现是稠密矩阵，不直接用于带状访存。 |
| CBLAS golden | `test/stbsv/stbsv_golden.h` | Netlib `cblas_stbsv` 作为精度对照。 |
| 社区任务测试数据 | 任务附件 `test_cases/stbsv_test.csv`、`gpu_baseline.csv` | CSV 驱动的精度、边界和性能验收输入。 |

arch35 的现有 kernel 有 SIMT 协作规约路径，但不能直接视作 arch22 已适配实现。本设计将其带状寻址与
求解语义作为基线，在 arch22 独立实现并验证目标芯片可用的并行执行原语和构建流程。

### 1.3 任务要求到设计落点

| 编号 | 要求 | 设计落点 |
| --- | --- | --- |
| RQ-01 | 支持 `uplo` 上/下三角、`trans` N/T/C、`diag` UNIT/NON_UNIT | Host 参数检查和 Kernel 编译期分支；实数 OP_C 等价 OP_T。 |
| RQ-02 | 列主序带状矩阵，`lda >= max(1,k+1)` | 上/下三角分别使用 cuBLAS 带状存储公式寻址。 |
| RQ-03 | `incx` 可正可负且不为零 | 统一逻辑向量下标到物理地址的映射；拒绝 `INT_MIN` 以避免绝对值溢出。 |
| RQ-04 | 前代/回代依赖顺序正确 | 根据 `uplo × trans` 决定前向或反向求解，依赖结果按序可见。 |
| RQ-05 | 空维、参数错误和空指针行为符合任务定义 | Host 先验证 handle、标量/枚举和指针；`n=0` 验证通过后直接成功返回。 |
| RQ-06 | FP32 精度满足 mixed tolerance | CBLAS golden、逐元素 atol/rtol、matched ratio 与最大绝对误差联合验收。 |
| RQ-07 | 5 个规定性能 case 达标 | arch22 协作式 SIMT/块内并行归约、无临时张量/同步拷贝；用 msprof 实测迭代。 |
| RQ-08 | 以 `aclblasSetStream` 绑定流，调用保持异步 | Kernel 直接在 handle stream 上下发；Host 不做流同步。 |

### 1.4 交付范围

本设计覆盖 ops-blas 公共头文件声明、arch22 Host 与 Ascend C Kernel、算子 README、CSV 驱动测试、精度与性能自验。
不实现批处理、多 RHS、复数类型、奇异性检测或 A/x 重叠处理；这些不属于任务接口要求。性能结果只在目标设备
上实际采样后报告，不以设计估算代替验收数据。

## 二、需求分析

### 2.1 数学语义与存储格式

对于逻辑行 `i` 和逻辑列 `j`，上三角带仅当 `i <= j` 且 `j-i <= k` 时有效；下三角带仅当 `j <= i` 且
`i-j <= k` 时有效。矩阵按列主序带状格式存放，带外内存和未选择的三角侧都不得读取。

不转置时带状数组下标为：

```text
UPPER: A_band(k + i - j + j * lda)
LOWER: A_band(i - j + j * lda)
```

转置/共轭转置（实数按转置）时读取原矩阵交换行列，即对应 `A_band(j,i)`。对角地址分别为
`UPPER: k + i*lda`、`LOWER: i*lda`；`diag=UNIT` 时不得读取该位置。

逻辑向量 `x[i]` 的物理偏移为：

```text
incx > 0: i * incx
incx < 0: (n - 1 - i) * (-incx)
```

该映射使 Kernel 内的逻辑顺序始终为 `0..n-1`，负步长只影响物理地址，不改变代数求解方向。

### 2.2 依赖方向

| 原矩阵 | trans | `op(A)` 三角方向 | 求解顺序 |
| --- | --- | --- | --- |
| LOWER | N | 下三角 | 前代，`i=0..n-1` |
| UPPER | N | 上三角 | 回代，`i=n-1..0` |
| LOWER | T/C | 上三角 | 回代，`i=n-1..0` |
| UPPER | T/C | 下三角 | 前代，`i=0..n-1` |

每个分量按下式计算，其中 `J(i)` 是 `op(A)` 在该行中除对角外有效且已求解的至多 `k` 个列索引：

```text
sum = x[i] - Σ(op(A)[i,j] * x[j]), j ∈ J(i)
x[i] = diag == UNIT ? sum : sum / op(A)[i,i]
```

除法遵循 FP32 运算；不对非单位对角的零值做奇异性检查，和 BLAS 求解接口一致。

### 2.3 性能约束与设计目标

性能验收设备为 Atlas 800T A2 / 910B3；目标对照 case 为：

| n | k | uplo | trans | diag | 上限（μs） |
| ---: | ---: | --- | --- | --- | ---: |
| 256 | 8 | UPPER | N | NON_UNIT | 198.0 |
| 512 | 32 | LOWER | N | NON_UNIT | 379.8 |
| 1024 | 16 | UPPER | T | UNIT | 659.5 |
| 2048 | 64 | LOWER | T | NON_UNIT | 1725 |
| 4096 | 128 | UPPER | N | UNIT | 2365 |

该算子存在三角依赖，不能把各行当作彼此独立的任务并发计算。性能路径采用一个协作 SIMT block 处理一个向量：
行与行仍严格按依赖方向推进，每行的带内点积由 block 内线程分担，并通过 block reduction 汇总；因此并行度用于
加速每行最多 `k` 项的带内更新，而不是违反前代/回代依赖。对 `n` 可放入 UB 的场景，将工作向量搬入片上存储，
依赖值从片上读取并减少对全局内存的重复访问；超出片上容量时采用分段向量访问的 TILED 路径。带宽较小时另设
低开销窄带路径，避免为很短的归约支付过多线程同步成本。

核内策略按实测选择阈值：小 `k` 使用窄带展开/少线程协作，中大 `k` 使用满 block 并行规约；`n` 能否放入 UB
只由实际可用 UB 与保留空间共同决定，不能使用固定容量假设。通过 msprof 分离 Kernel 时延，重点观测 256/8、
512/32 的启动与同步占比，以及 4096/128 的访存、规约和 UB 容量影响。所有调度分支须经精度用例覆盖且不得改变
数学顺序。性能门槛为任务验收标准，本设计不预先声称已达标。

### 2.4 参数与异常契约

| 参数 | 合法条件 | 错误行为/说明 |
| --- | --- | --- |
| `handle` | 非空句柄 | 空指针返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`；句柄有效性由 ops-blas 句柄生命周期管理保证。 |
| `uplo` | `ACLBLAS_UPPER` 或 `ACLBLAS_LOWER` | 其他值返回 `ACLBLAS_STATUS_INVALID_VALUE`。 |
| `trans` | `ACLBLAS_OP_N/T/C` | 其他值返回 `ACLBLAS_STATUS_INVALID_VALUE`。 |
| `diag` | `ACLBLAS_UNIT` 或 `ACLBLAS_NON_UNIT` | 其他值返回 `ACLBLAS_STATUS_INVALID_VALUE`。 |
| `n`, `k` | `n>=0`, `k>=0` | 负值返回 `ACLBLAS_STATUS_INVALID_VALUE`。 |
| `lda` | `lda >= max(1,k+1)` | 不满足返回 `ACLBLAS_STATUS_INVALID_VALUE`。 |
| `incx` | 非零且不为 `INT_MIN` | 非法步长返回 `ACLBLAS_STATUS_INVALID_VALUE`。 |
| `A`, `x` | `n>0` 时非空 | 任一为空返回 `ACLBLAS_STATUS_INVALID_VALUE`。 |
| `n=0` | 标量、枚举、handle 校验通过 | 返回成功，不访问 A/x，不发射 Kernel。 |

`k` 不强制小于 `n`：当 `k>=n` 时，实际参与计算的列范围裁剪到矩阵边界；带状存储和 `lda` 仍按传入 `k`
定义。所有地址和跨度中间计算使用足够宽的有符号整数，避免 `n*lda`、`(n-1)*abs(incx)` 的 32 位溢出。

## 三、详细设计

### 3.1 工程结构

实现合入 ops-blas 仓的路径规划如下：

```text
ops-blas/
├── include/cann_ops_blas.h                 # 核对并保留公共 aclblasStbsv ABI 声明
├── blas/tbsv/README.md                     # A2/A3 产品支持信息与接口说明
├── blas/tbsv/arch22/stbsv_host.cpp         # 校验、tiling 与 stream launch
├── blas/tbsv/arch22/stbsv_kernel.cpp       # arch22 Ascend C 协作式求解入口
├── blas/tbsv/arch22/stbsv_simt.cpp         # 若工程拆分需要，放置 SIMT block 内点积/规约实现
├── blas/tbsv/arch22/stbsv_tiling_data.h    # Host/Kernel 共享参数
└── test/tbsv/stbsv/arch22/                # A2/A3 CSV 驱动测试
```

当前仓库已有通用声明、`blas/stbsv/README.md`、测试参数/golden 和 arch35 实现；按任务约定新增 A2/A3 实现放在
`blas/tbsv/arch22/`，不复制公共头文件声明。架构目录源文件由仓库 `SOC_ARCH_DIRS` 自动筛选。测试工程按任务要求放在
`test/tbsv/stbsv/arch22/`；接入时核实当前分支 CMake 的家族路径识别，并只增加必要的构建发现配置。

### 3.2 Host 侧设计

Host 接口原型与公共头文件一致：

```cpp
aclblasStatus_t aclblasStbsv(
    aclblasHandle_t handle, aclblasFillMode_t uplo, aclblasOperation_t trans,
    aclblasDiagType_t diag, int n, int k, const float* A, int lda,
    float* x, int incx);
```

执行顺序：

1. 检查 handle，优先保证空 handle 返回约定状态；随后检查 `n/k`、枚举、`lda`、`incx`。
2. 当 `n>0` 时检查 `A/x` 非空；`n=0` 在其余标量约束校验后直接成功返回。
3. 依据 `n/k/lda/incx/uplo/trans/diag` 填充轻量 tiling 参数。Host 不分配设备临时矩阵，不复制 A/x，不同步流。
4. 根据 `n`、`k`、UB 使用上限和测得的分支阈值选择窄带、UB 驻留或 TILED 核内路径；阈值仅控制实现路径，语义不变。
5. 在 `handle` 携带的 stream 上发射 kernel，立即返回成功；运行时异步错误按仓库现有 BLAS 接口约定处理。

Host 不将 `n` 个行分配给多个独立 block，因为三角方程之间有递归依赖，普通 block 间没有 kernel 内全局同步。
当前实现不需要额外 workspace；若后续 profile 证明分阶段 wavefront 可抵消额外 launch 成本，再另行评估，并须在设计变更中说明。

### 3.3 Kernel 侧设计

#### 3.3.1 并行组织

Kernel 使用一个 AIV/SIMT 协作 block 求解一个向量，避免跨 block 依赖和全局同步。每次迭代对应一个逻辑行：

1. 根据 `uplo/trans` 计算当前行的已求解列区间，区间长度不超过 `k`，并对矩阵端点裁剪。
2. block 内线程以 lane-stride 分配这些系数，每线程累加局部乘积。
3. 使用 block 内规约得到点积；同步后由负责线程减去点积、（非 UNIT 时）除以对角元，并回写 `x[i]`。
4. 通过 block 同步保证该结果对下一行迭代可见。

当 `n` 和 `x` 跨度满足 UB 预算时，Kernel 并行搬入逻辑向量到 UB，保持一个片上解向量副本；每个解出分量同步
更新片上副本并按 `incx` 写回全局 `x`。对较大 `n` 选择分段/TILED 模式，保留全局向量作为解值来源，不创建与 n 成正比
的全局临时区。UB 占用由运行时架构常量和编译后资源报告校验；block reduction scratch 按最大协作线程数静态规划。

#### 3.3.2 分支特化与访存优化

- `uplo/trans/diag` 组合在 Host dispatch 到编译期特化入口，避免每个带内元素重复执行枚举判断。
- OP_C 与 OP_T 共用转置实现；UNIT 路径从不加载对角元。
- 带状地址按列主序公式直接计算，不读取带外 padding；`lda` padding 只影响列间距。
- 窄带路径以有效长度 `min(k, row distance, n-1-row)` 为界，避免扫描无效列；小 `k` 的短循环静态展开须以编译器报告和 profile 为依据。
- 向量连续（`incx=1`）的性能主用例采用连续加载/存储；一般正负步长使用同一逻辑索引映射保证完整功能。
- reduction 顺序不要求逐位确定；需满足混合容差。禁用会破坏 `diag=UNIT` 对 NaN/Inf 对角“不读取”保证的投机读。

#### 3.3.3 SIMT/arch22 兼容性门禁

arch35 已有的 SIMT API 作为算法参考，不直接复用其 arch35 kernel 文件或假设其所有接口均可在 arch22 编译。正式实现前先在
CANN 9.1.0 的 `dav-2201` 目标上验证 block launch、线程索引、同步、UB 指针和规约所需 API。若目标编译器不支持相同 SIMT
接口，则在 arch22 Ascend C vector kernel 中实现等价的 block 内协作规约。无论采用哪套接口，性能方案仍保持“行依赖串行、
带内更新并行、单 block 全局同步”的执行契约，并通过各性能 case 的 msprof 结果决定窄带/协作实现切换阈值。

### 3.4 精度与特殊值

精度基准使用随任务提供的 CBLAS/Netlib `cblas_stbsv`。A 与 RHS 的常规随机数据覆盖均匀分布 `[-5,5]` 和正态分布
（`μ∈[-5,5]`、`σ∈[0.1,2]`），两类各占约 50%；若仓内 CSV 测试框架当前不能生成正态分布，需扩展测试填充器或在任务附件
生成器中提供确定性正态填充值，并保持 seed 可复现，不能将仅支持均匀分布当作覆盖完成。A 的非单位对角在测试侧强化为远离零的值，以排除奇异系统未定义行为；
UNIT 用例把对角位置设为 NaN/Inf，验证 kernel 不读取对角。向量 Inf/NaN 按传播结果与测试工程参考规则比对；不把包含未定义
除零的 NON_UNIT 奇异矩阵作为正确性承诺。

FLOAT32 采用任务规定的 mixed tolerance：

```text
element_match: abs(actual - golden) <= 2^-13 + 2^-13 * abs(golden)
whole_case:    matched_ratio >= 0.99 && max_abs_error <= 1e-2 (或按验收确认的 ULP 限值)
```

CPU golden 与 NPU 使用完全相同的 A/x 输入、`uplo/trans/diag/n/k/lda/incx` 参数。误差报告需保留逐 case matched ratio、
max absolute error、首个不匹配逻辑下标和实际/期望值。

### 3.5 测试与验收设计

以任务附件 `test_cases/README.md` 和 CSV 为准，使用 ops-blas CSV 驱动 GTest 工程。测试分类包括：

| 类别 | 覆盖 |
| --- | --- |
| 基础枚举 | UPPER/LOWER × N/T/C × UNIT/NON_UNIT；小 n、多种 k。 |
| Shape/带宽 | n=0/1/质数/非对齐/2 的幂及邻值至 2048；k=0/1/中间/满带。 |
| 地址语义 | `lda` 紧凑与 padding；`incx=±1/±2/±3`；上下三角、转置组合。 |
| 对角与数值 | UNIT 对角 NaN/Inf 不读取；A 与 RHS 均匀/正态（各约 50%）、交替/极端输入；NON_UNIT 对角非零。 |
| 错误与边界 | 空 handle、空 A/x、非法 enum、负 n/k、非法 lda、incx=0/INT_MIN、n=0 quick return。 |
| 精度 | Netlib CBLAS golden；FLOAT32 atol/rtol、matched ratio、最大绝对误差。 |
| 性能 | 任务书 5 个 case 与附件全部 TC_PF；连续 `incx=1`、紧凑 `lda=k+1`；warmup 后有效采样并以 msprof 核时为准。 |

附件生成器默认产生 1000 条精度和 200 条性能用例，随机种子固定时输出可复现。当前附件说明仓内填充器主要提供均匀分布，
与任务书 50% 正态分布要求之间存在测试工具缺口；自验交付前必须补上正态生成能力，并对固定 seed 做可复现性检查。测试接入前
核对参数列、错误状态码、空维 quick-return 顺序和各测试构建目标与仓库现行版本一致，不通过删减 case 掩盖失败。

性能验收流程：先编译 `--soc=ascend910b3 --ops=stbsv`，运行 TC_PF 用例并预热；对代表 case 分别运行 msprof，统计
`OpBasicInfo.csv` 中 `Task Duration(us)` 的有效样本平均值。逐 case 对照任务书时延上限；如未达标，按 kernel 阶段 profile
优化窄带阈值、block 规约线程数、UB 驻留界限和加载/同步方式，再重跑全部相关精度与性能测试。性能结果应记录芯片型号、CANN
版本、编译命令、设备号、warmup/sample 数、各 case 平均值及 profiler 原始文件。

### 3.6 风险和验证门禁

| 风险 | 应对与退出条件 |
| --- | --- |
| arch35 SIMT API 在 arch22 上不可用或行为不同 | 先以 `dav-2201` 编译探针验证；改用 arch22 可用的 Ascend C 协作规约实现，不将未编译 API 带入提交。 |
| 三角依赖限制多核并行 | 不做无同步的跨 block 行并行；通过带内乘加并行、片上 x 缓存与窄带低同步路径优化。 |
| reduction 同步开销导致小 k 延迟过高 | 对窄带建立少线程/展开路径，用 256/8、512/32 case profile 决定阈值。 |
| n 大于 UB 可驻留容量 | 切换 TILED 全局向量访问；容量边界两侧均纳入 shape 与精度测试。 |
| `incx<0` 地址基准不一致 | 与 Netlib CBLAS 对照负步长 case，按逻辑向量下标映射物理偏移。 |
| 社区附件和仓内 harness 有列/状态码差异 | 先检查并适配 CSV loader、golden 和 GTest，不改变任务书接口语义。 |
| 性能受启动开销或目标设备影响 | 在指定 910B3 上报告真实 msprof 数据；设计阶段不宣称达标。 |

## 四、兼容性与交付

该接口为 ops-blas 新增的 `aclblasStbsv` A2/A3 kernel 实现，保持现有 API 原型和枚举 ABI 不变。架构选择由仓库
`SOC_ARCH_DIRS` 控制，910B/A2/A3 构建选择 arch22；950/arch35 实现不受本设计修改影响。实现需更新 README 产品支持表，
将 Atlas A2/A3 训练与推理系列标为支持，并同步提交 `blas/tbsv/arch22/`、`test/tbsv/stbsv/arch22/` 及所需公共声明变更。

最终交付包含：设计文档、算子 README、实现与 CSV 测试、可复现的自验步骤、精度报告/日志、性能报告/日志及内存报告/日志。
提交社区任务验收前，A2 与 A3 均需完成任务要求的验证；本设计阶段不包含未执行的测试结果或截图。
