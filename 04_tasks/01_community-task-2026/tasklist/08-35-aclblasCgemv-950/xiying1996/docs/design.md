# 需求背景（required）

## 需求来源

8月社区任务 — aclblasCgemv 算子开发（Ascend 950PR）。在昇腾 NPU（Ascend 950PR）上使用 Ascend C 编程语言开发单精度复数（complex64）矩阵-向量乘算子 `aclblasCgemv`，完成算子设计、开发、测试全流程，验收通过后合入昇腾算子开源仓 ops-blas（gitcode.com/cann/ops-blas）。

## 背景介绍

### aclblasCgemv 算子实现现状分析

ops-blas 仓 `include/cann_ops_blas.h` 已声明 `aclblasCgemv` 接口（与 cuBLAS `cublasCgemv` 参数序列一致），供各产品线共用，禁止定义 950PR 私有平行接口。

当前实现现状：

| 架构            | 现状   | 说明                                                                                                                            |
| :------------ | :--- | :---------------------------------------------------------------------------------------------------------------------------- |
| arch22（A2/A3） | 已有实现 | `blas/gemv/arch22/`，AIV vector 模型；存在缺陷：OP\_C 未实现共轭转置（T/C 共用内核按普通转置计算）、不支持负步长、参数校验不完整、no-op/quick return 语义缺失，不能直接作为 arch35 复用 |
| arch35（950PR） | 空缺   | `blas/gemv/arch35/` 仅有实数 `aclblasSgemv`（SIMT 模型），复数 `aclblasCgemv` 待开发                                                        |

### 基线来源说明

本设计以 ops-blas 仓已有实现为工程基线，不引入新的外部依赖：

| 基线层次             | 直接路径                                          | 作用                                                                     |
| :--------------- | :-------------------------------------------- | :--------------------------------------------------------------------- |
| arch35 SIMT gemv 框架 | `blas/gemv/arch35/sgemv_{host,kernel}.cpp`    | 复用实数 sgemv 的 host 校验/tiling/SIMT kernel 结构（grid-stride、正负步长寻址、`asc_vf_call` 启动骨架） |
| arch35 复数算子惯例     | `blas/geam/arch35/cgeam_{host,kernel}.cpp`    | 复用复数交织存储（float 实部/虚部成对）、alpha/beta 判零、alpha=0 时 A/x 空指针替换等复数处理惯例          |
| 复数语义参考           | `blas/gemv/arch22/cgemv_*`、Netlib `cgemv.f`   | 核对 OP_N/OP_T/OP_C 与 quick-return 语义（arch22 的缺陷项按 Netlib 修正，不直接复用）        |
| 测试框架             | `test/frame/*`、`test/gemv/sgemv/arch35/*`     | CSV 驱动 GTest、cblas golden、MIXED_TOLERANCE 精度判定（实部/虚部分离比对）               |

# 需求分析（required）

## 需求描述

使用 Ascend C 编程语言（kernel 直调方式，handle 绑定 stream）实现 `aclblasCgemv` 算子，支持 complex64 数据类型，功能语义与 cuBLAS `cublasCgemv`/Netlib `cgemv` 对齐，精度满足生态算子开源精度标准，性能满足任务书 §3.3 标杆耗时，并完成 README、测试代码与自测报告等交付件。

## 需求拆解

1. 实现 `aclblasCgemv` 接口（`include/cann_ops_blas.h` 已有声明，逐参数一致），代码位于 `blas/gemv/arch35/`；
2. 支持 op(A) 三种模式，其中 OP\_C 共轭转置必须实现（不得退化为普通转置）；
3. 支持正负步长 incx/incy（负步长按 Netlib 反向遍历）；
4. 实现全部参数校验与 no-op/quick return 语义（错误码与任务书 §2.4 一致）；
5. 精度：COMPLEX64 实部/虚部分别按 FLOAT32 判定，满足 atol=2⁻¹⁶、rtol=2⁻¹⁰、matched\_ratio≥0.99、max\_abs\_error≤1e-2 或 32×ULP；
6. 性能：OP\_N 512×512 ≤ 2.18us、OP\_N 2048×2048 ≤ 3.86us、OP\_T 2048×2048 ≤ 3.51us（warmup 后有效采样 >50 次取平均）；
7. 新增 `blas/gemv/arch35/README_cgemv.md`（接口语义/执行路径/三级耗时统计/性能分析与优化建议），`blas/gemv/README.md` 产品支持表同步（Ascend 950PR：支持）随 PR 提交；
8. 测试代码合入 `test/gemv/cgemv/arch35/`（CSV 驱动 GTest 工程，结构参考主仓 blas 算子测试代码）；精度套件覆盖任务书 §3.5 全部类别，性能套件（TC\_PF，200 用例）覆盖任务书 §3.3 三个 case 及扩展规模；自验结果与性能数据记录于自测报告。

## 外部组件依赖

不引入新的第三方组件。复用 ops-blas 已有的：

| 依赖                     | 来源                                | 用途                                       |
| :--------------------- | :-------------------------------- | :--------------------------------------- |
| handle 机制              | `common/helper/aclblas_handle_internal.h` | handle 携带 stream、常驻 workspace 复用          |
| host 工具                | `common/helper/host_utils.h`      | `CHECK_RET`/`CeilDiv`/`CeilAlign`/`GetAivCoreCount` |
| 硬件/常量                  | `common/helper/kernel_constant.h` | `UB_SIZE`、`SIMT_MIN_THREAD_NUM`、`SIMT_MAX_THREAD_NUM` |
| 接口声明                   | `include/cann_ops_blas.h`         | `aclblasCgemv`/`aclblasComplex` 既有声明（不改）  |
| 测试公共件                  | `test/frame/*`、`test/utils/*`     | GTest 注册、CSV 加载、精度判定                     |
| 精度 golden（仅测试工程）       | 系统 cblas（Netlib `cgemv` 复数实现）     | 生成 golden，随测试工程提供                        |

## 内部适配模块

| 模块        | 计划文件                                             | 设计职责                                                        |
| :-------- | :----------------------------------------------- | :---------------------------------------------------------- |
| Host 实现   | `blas/gemv/arch35/cgemv_host.cpp`                | `aclblasCgemv` 入口：参数校验、quick return、路径选择（gating）、tiling 计算、workspace 获取、kernel 直调、四阶段计时 |
| Kernel 实现 | `blas/gemv/arch35/cgemv_kernel.cpp`              | SIMT 三核（N / TC / Scale）+ AIV 快速路（FastTc 列拆分 / FastN 行拆分）      |
| Tiling 结构 | `blas/gemv/arch35/cgemv_tiling_data.h`           | host/kernel 共享 `CgemvTilingData`（POD）、phaseMask 诊断位、`CgemvBuildFastTiling`/`CgemvHostTimingGet` 导出 |
| 精度测试      | `test/gemv/cgemv/arch35/cgemv_test.cpp` + `cgemv_test.csv` | CSV 驱动 GTest（1200 用例）：cblas golden、实部/虚部分离 MIXED\_TOLERANCE 比对、负向用例 |
| 性能测试      | `test/gemv/cgemv/arch35/cgemv_test_perf.cpp`     | 性能套件（200 用例）：phaseMask 差分阶段分解、host 四阶段导出、CSV 报告落盘            |
| 测试辅件      | `cgemv_param.h`、`cgemv_npu_wrapper.h`、`cgemv_golden.h` | CSV 参数解析（alpha\_real/alpha\_imag 等列）、device 内存搬运包装、cblas\_cgemv golden 包装 |
| 运行入口      | `run_cgemv_test.sh [smoke\|accuracy\|perf\|all]` | 一键构建 + 运行 + 报告；`CGEMV_BENCH_WARMUP`/`CGEMV_BENCH_SAMPLES` 可调 |
| 构建接入      | 顶层 `CMakeLists.txt`                              | 性能验收要求优化编译：`CMAKE_BUILD_TYPE` 默认 Release（-O3），Debug 仅供调试      |
| 文档        | `blas/gemv/arch35/README_cgemv.md`、`blas/gemv/README.md` | 接口/架构/三级计时/性能分析文档；产品支持表更新（Ascend 950PR：支持）              |

`blas/CMakeLists.txt` 按 `SOC_ARCH_DIRS` 自动收集 `blas/*/arch35/*.cpp`，算子源码无需改构建脚本。

## 接口原型

与 ops-blas 仓 `include/cann_ops_blas.h` 已有声明逐参数一致（复数类型 `aclblasComplex` 以 `include/cann_ops_blas_common.h` 定义为准，实部/虚部各 float32）：

```cpp
aclblasStatus_t aclblasCgemv(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n, const aclblasComplex* alpha,
    const aclblasComplex* A, int lda, const aclblasComplex* x, int incx, const aclblasComplex* beta,
    aclblasComplex* y, int incy);
```

### 参数说明

| 参数名    | 输入/输出 | 描述                                                         | dtype     | 维度/值域                                   | 异常行为                                                              |
| :----- | :---- | :--------------------------------------------------------- | :-------- | :-------------------------------------- | :---------------------------------------------------------------- |
| handle | 输入    | ops-blas 库上下文句柄，携带 stream，Host 内存                           | -         | 指向已创建的有效句柄                               | nullptr → `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`                      |
| trans  | 输入    | 矩阵操作类型，Host 内存                                             | 枚举        | {OP\_N, OP\_T, OP\_C}                   | 非法枚举 → `ACLBLAS_STATUS_INVALID_ENUM`                              |
| m      | 输入    | A 的行数，Host 内存                                              | int       | m ≥ 0                                   | m < 0 → `ACLBLAS_STATUS_INVALID_VALUE`；m = 0 合法 quick return       |
| n      | 输入    | A 的列数，Host 内存                                              | int       | n ≥ 0                                   | n < 0 → `ACLBLAS_STATUS_INVALID_VALUE`；n = 0 合法 quick return       |
| alpha  | 输入    | 复数标量指针，Host 内存                                             | COMPLEX64 | FLOAT32 全集                               | nullptr → `ACLBLAS_STATUS_INVALID_VALUE`                          |
| A      | 输入    | 列主序复数矩阵，数组维度 lda×n（有效 m×n），Device 内存，只读                     | COMPLEX64 | FLOAT32 全集                               | m>0 且 n>0 且 alpha≠(0,0) 时 nullptr → `ACLBLAS_STATUS_INVALID_VALUE` |
| lda    | 输入    | A 前导维度，Host 内存                                             | int       | lda ≥ max(1, m)                          | 违反 → `ACLBLAS_STATUS_INVALID_VALUE`                               |
| x      | 输入    | 输入向量，Device 内存，只读；逻辑长度 lenx（N:n，T/C:m），存储元素数 ≥ 1+(lenx−1)×\|incx\| | COMPLEX64 | FLOAT32 全集                               | 需读取（lenx>0 且 alpha≠(0,0)）时 nullptr → `ACLBLAS_STATUS_INVALID_VALUE` |
| incx   | 输入    | x 步长，支持正负，Host 内存                                          | int       | incx ≠ 0 且 ≠ INT32\_MIN                  | =0 或 =INT32\_MIN → `ACLBLAS_STATUS_INVALID_VALUE`                 |
| beta   | 输入    | 复数标量指针，Host 内存；beta=(0,0) 时 y 不必是有效输入                      | COMPLEX64 | FLOAT32 全集                               | nullptr → `ACLBLAS_STATUS_INVALID_VALUE`                          |
| y      | 输入/输出 | 输入/输出向量，Device 内存，原地覆写；逻辑长度 leny（N:m，T/C:n）                 | COMPLEX64 | FLOAT32 全集                               | 需写回（leny>0 且非 quick-return）时 nullptr → `ACLBLAS_STATUS_INVALID_VALUE` |
| incy   | 输入    | y 步长，支持正负，Host 内存                                          | int       | incy ≠ 0 且 ≠ INT32\_MIN                  | =0 或 =INT32\_MIN → `ACLBLAS_STATUS_INVALID_VALUE`                 |

**返回值**：`aclblasStatus_t`，状态码语义与 `include/cann_ops_blas_common.h` 一致。

### no-op / 退化语义（依据 Netlib cgemv）

| 场景                        | 行为                                                       |
| :------------------------ | :------------------------------------------------------- |
| m = 0 或 n = 0             | 合法 quick return，返回 `ACLBLAS_STATUS_SUCCESS`，不执行计算、不写 y    |
| alpha = (0,0) 且 beta = (1,0) | quick return，不读 A/x、不写 y                                 |
| alpha = (0,0) 且 beta ≠ (1,0) | 退化为 `y = beta * y`（mode=1，不读 A、x；A/x 允许为 nullptr）        |
| beta = (0,0)              | y 的输入值不被读取（y 可为 NaN/未初始化），仅写回 `alpha*op(A)*x`（防 0×NaN 污染） |
| beta = (1,0)              | 正常路径，y 旧值直通相加（不走复数乘，防 1×NaN 传播差异）                        |

### 设计范围与约束

| 类别        | 约束项     | 约束内容                                                   |
| :-------- | :------ | :----------------------------------------------------- |
| 参数合法性     | 校验序     | handle → trans 枚举 → m/n → lda → incx/incy → alpha/beta 指针 → A/x/y 指针（按需） |
| 共轭转置      | OP\_C   | 必须对 A 取共轭转置（A^H），不得退化为普通转置                             |
| 非连续 Tensor | 不要求    | 不支持超出 lda/incx/incy 语义的非连续内存访问                          |
| broadcast | 不涉及     | A/x/y 独立，无广播                                           |
| dynamic shape | 不要求 | m/n 为运行时入参，tiling 每次调用即时计算                             |
| 原地与视图     | y 原地覆写  | 不返回视图                                                  |
| 确定性计算     | 不要求     | 浮点乘累加顺序与 golden 可在容差内不同                                 |
| 异步执行      | 依赖 handle 绑定的 stream | kernel 异步下发；读回 Device 结果前须同步 stream         |

# 详细设计（required）

## 算子分析

### 数学公式

```
y = alpha * op(A) * x + beta * y
```

复数乘法展开（实部/虚部分别为 FLOAT32 累加，非 bit-exact，允许舍入顺序差异）：

```
非共轭（N/T）:  accR += aR*xR - aI*xI;   accI += aR*xI + aI*xR
共轭（C）:      accR += aR*xR + aI*xI;   accI += aR*xI - aI*xR   （A 逐元素取共轭 → 交叉项反号）
写出:          yR = alphaR*accR - alphaI*accI + betaR*yR_in - betaI*yI_in
               yI = alphaR*accI + alphaI*accR + betaR*yI_in + betaI*yR_in
```

### 支持数据类型

| 参数         | 数据类型                                   | 存储               |
| :--------- | :------------------------------------- | :--------------- |
| A          | COMPLEX64（aclblasComplex，实/虚各 float32） | Device，列主序，lda×n |
| x、y        | COMPLEX64                              | Device，步长存储      |
| alpha、beta | COMPLEX64                              | Host 标量指针        |

### 支持形状

- m、n 为任意非负整数（int 范围均支持）：A 有效形状 \[m, n]、存储形状 \[lda, n]；trans=N 时 x/y 逻辑长度 n/m，trans=T/C 时 x/y 逻辑长度 m/n；m=0 或 n=0 为合法 quick return；
- 自验覆盖范围：精度用例 m、n 单边至 16384，性能用例至 4096（受测试侧 host 内存预算约束，非算子能力上限）；
- lda ≥ max(1, m)，支持 padding（lda > m）；
- incx、incy ∈ ℤ \ {0, INT32\_MIN}，支持负值（负步长按 Netlib 语义反向遍历）；
- 不涉及广播；不要求 dynamic shape；不要求非连续 Tensor 支持。

## 算子实现

### 实现方案

采用 **双路径架构**：SIMT 通用兜底 + AIV 向量快速路，kernel 直调方式（host 侧完成参数校验、no-op 分支、路径选择与 tiling 计算后 `cgemv_kernel<<<coreNum, nullptr, stream>>>` 启动，handle 绑定 stream 异步执行）。

```
aclblasCgemv（host）
 ├─ 参数校验 / quick return（m·n==0；alpha=(0,0)∧beta=(1,0)）
 ├─ mode = alpha==(0,0) ? 1 : 0        // 1 = 退化 y=beta*y（不读 A/x）
 ├─ 路径选择（CgemvBuildFastTiling gating，§3.2.1）：
 │     lda==m ∧ incx==1 ∧ incy==1 ∧ m%32==0 ∧ UB 预算可容纳
 │       → path=1 AIV 快速路（T/C 列拆分；N 行拆分，均无跨核归约）
 │     否则 → path=0 SIMT 兜底（任意 incx/incy/lda/小尺寸）
 └─ cgemv_kernel<<<coreNum, nullptr, stream>>>(tiling)   // 异步
```

选择依据：

- **SIMT 标量点积**实现简单、语义全覆盖，但带宽受限算子下逐元素标量访存效率低，大尺寸距屋顶线 2~3 个数量级；
- **AIV 向量快速路**（MicroAPI + MTE2/MTE3 大块搬运 + 3 缓冲流水）可将有效带宽推向屋顶线；
- 快速路 gating（lda==m、incx/incy==1、m%32==0）不满足时**自动回落 SIMT**，对用户透明，语义无损。

### host侧设计（`cgemv_host.cpp`）

1. **参数校验**（任务书 §2.4 顺序）：
   - handle == nullptr → `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`（入口处校验）；
   - trans 不在 {N, T, C} → `ACLBLAS_STATUS_INVALID_ENUM`；
   - m<0 / n<0 / lda\<max(1,m) / incx==0 / incx==INT32\_MIN / incy==0 / incy==INT32\_MIN → `ACLBLAS_STATUS_INVALID_VALUE`；
   - alpha/beta == nullptr → `ACLBLAS_STATUS_INVALID_VALUE`；
   - A：m>0 且 n>0 且 alpha≠(0,0) 时 A==nullptr → `ACLBLAS_STATUS_INVALID_VALUE`；x 同条件（xLen>0 且 alpha≠(0,0)）；y：需写回（yLen>0 且非 quick-return）时 y==nullptr → `ACLBLAS_STATUS_INVALID_VALUE`。
2. **no-op / quick return 分支**：
   - m==0 \|\| n==0 → 返回 SUCCESS，不启 kernel；
   - alpha==(0,0) 且 beta==(1,0) → 返回 SUCCESS，不写 y、不启 kernel；
   - alpha==(0,0) 且 beta≠(1,0) → **mode=1 退化分支**：kernel 仅执行 y=beta\*y，不读 A、x（A/x 此时允许为 nullptr）。
3. **路径选择与 tiling 计算（`CgemvBuildFastTiling`，host 与性能测试共用）**：
   - **gating**：`lda==m && incx==1 && incy==1 && (m%32)==0 && m>0 && n>0`，且 UB 预算可容纳，任一不满足返回 0 → SIMT 兜底；
   - 核切分（base/rem 精确均分）：
     - T/C：`coreNum = min(aivCoreNum, n)`，按 A 的列（=输出维）拆核，每核独立产出 y，**无跨核归约**；
     - N：`coreNum = min(aivCoreNum, n, m/32)`，按 32 行粒度行段拆核（保证每核 ≥32 行），每核独立产出 y，**无跨核归约**；
   - **groupCols G**（T/C 每 MAC tile 列数，模板仅实例化 1..4）：host 按 UB 预算（`CGEMV_UB_BUDGET = UB_SIZE − 8KB 保留`）从 4 向下试到 1：
     - T/C 需求 = x 段 2m + 3-slot A 流 3·G·2m + rs nTiles×32 + y 段；
     - N 需求 = partial 2m + 3-slot aTile 3×(2·32·128) + x 段；
     - 放不下（m 过大）→ 返回 0 走 SIMT。
4. **workspace**：N 快速路按 `coreNum×2m` floats 预留（优先复用 handle 常驻 workspace，保持异步；不足时按调用 `aclrtMalloc`，launch 后 sync+free，回退为同步语义并 `OP_LOGW`）。当前 N 实现为行拆分、每核独立产出 y，kernel 侧未消费 workspace（`(void)wsGm`），预留保留为后续跨核归约变体余量。
5. **kernel 启动**：`cgemv_kernel_do(A, x, y, workSpace, tiling, tiling.coreNum, h->stream)`，异步入队。
6. **host 四阶段计时**（详见 §3.2.4）：validate / tiling / wsAlloc / launch / total，`steady_clock` 写 thread\_local `CgemvHostTiming`，经 `CgemvHostTimingGet` 导出。

**host 侧流程**：

```
aclblasCgemv(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy)
   │
   ├─ handle == nullptr ───────────────────────────────→ HANDLE_IS_NULLPTR
   ├─ 参数校验（trans/m/n/lda/incx/incy/alpha/beta/A/x/y）→ INVALID_ENUM / INVALID_VALUE
   ├─ m==0 ∨ n==0 ─────────────────→ SUCCESS（quick return，不启 kernel）
   ├─ alpha==(0,0) ∧ beta==(1,0) ──→ SUCCESS（不写 y）
   ├─ mode = alpha==(0,0) ? 1 : 0
   ├─ CgemvBuildFastTiling（gating：lda==m ∧ incx==1 ∧ incy==1 ∧ m%32==0 ∧ UB 预算）
   │     ├─ 通过 → path=1：T/C coreNum=min(aiv,n)；N coreNum=min(aiv,n,m/32)；G∈{4..1} 按 UB 预算
   │     └─ 不通过 → BuildSimtTiling：path=0，coreNum=min(ceil(outDim/128), aivCoreNum)
   ├─ workspace（N 快速路预留 coreNum×2m floats；常驻复用优先）
   └─ cgemv_kernel_do<<<coreNum, stream>>> → SUCCESS（异步）
```

**tiling 数据结构**（host/kernel 共享 POD，按值直传 kernel，8B 对齐）：

```cpp
struct CgemvTilingData {
    // ---- 问题描述 ----
    uint32_t m;          // A 行数
    uint32_t n;          // A 列数
    uint32_t lda;        // A 前导维（列主序）
    uint32_t trans;      // 0=N, 1=T, 2=C
    uint32_t mode;       // 0=正常计算; 1=退化 y=beta*y（不读 A/x）; 2=空 kernel（launch 地板，测试用）
    float alphaR;
    float alphaI;
    float betaR;
    float betaI;
    int64_t incx;        // 支持正负步长
    int64_t incy;

    // ---- 路径与核切分 ----
    uint32_t path;       // 0=SIMT 兜底; 1=AIV 快速路
    uint32_t coreNum;    // launch block 数
    uint32_t colPerCore; // 快速路：每核基础份额（base 向上取整后均分）
    uint32_t colRem;     // 前 colRem 个核多 1 份
    uint32_t groupCols;  // T/C：每个 MAC tile 的列数 G（host 按 UB 预算计算）
    uint32_t phaseMask;  // 诊断位掩码（生产恒 0，见 CGEMV_PM_*）
};
static_assert(sizeof(CgemvTilingData) % 8 == 0, "CgemvTilingData must be 8B aligned for kernel param");
```

### kernel侧设计（`cgemv_kernel.cpp`）

1. **数据视图**：A/x/y 按 aclblasComplex 交错 float 存储访问（real=gm\[2i]，imag=gm\[2i+1]）。
2. **入口分发**（`KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY)`，块数 = tiling.coreNum）：

```
cgemv_kernel(a, x, y, workSpace, tiling)
   │
   ├─ mode == 2 → return                     // 空 kernel：launch 地板测量（测试用）
   ├─ mode == 1 → CgemvSimtScale             // y = beta*y（beta=(0,0) 写 0 不读 y；(1,0) 直通）
   ├─ path == 1（AIV 快速路）
   │     ├─ trans == N → CgemvFastN          // 行拆分，outTile×dotTile 三缓冲流水
   │     └─ trans == T/C → CgemvFastTc       // 列拆分，3-slot ping-pong 流水
   └─ path == 0（SIMT 兜底）
         ├─ trans == N   → CgemvSimtN        // 行 grid-stride 标量点积
         └─ trans == T/C → CgemvSimtTC       // 列 grid-stride（conj 参数区分 T/C）
```

3. **T/C 快速路（`CgemvFastTc`）——列拆分点积，无跨核归约**：
   - 每核列段 \[start, count)（base/rem 均分）；G 列一个 MAC tile；
   - **装载**：lda==m → 列段在 GM 连续，1D DataCopy 整块 `cnt·2m` floats（128KB 级大块，3-slot ping-pong，slot t 占 `t%3`）→ `SetFlag(MTE2_V)`；
   - **流水序**：主循环先发 tile t+2 的 MTE2 描述符、再等 tile t 就绪——MTE2 队列在 V 计算 MAC(t) 期间恒持有 2 个在途描述符（若先等后发则 MTE2 在途仅 1 个且与 V 强串行，MAC 无法隐藏）；
   - **MAC（`CgemvTcMacTile<G>`）**：逐 32 行 chunk（64 floats）DINTLV 装载 xR/xI 与 aR/aI，每列 4 条 `MulAddDst`：
     `sPR += aR·xR`、`sPI += aI·xI`、`sQR += aR·xI`、`sQI += aI·xR`；
   - **列尾归约**：`ReduceSum` → rs 分量连续布局 \[PR|PI|QR|QI]（各 8-float 步长，每 tile 32 floats，保 32B 对齐）；
   - **epilogue**：y 段先整块 DataCopyPad 入 UB（向量核标量单元不能直访 GM）；标量合成——
     T：`accR = PR−PI, accI = QR+QI`；C（共轭）：`accR = PR+PI, accI = QR−QI`（**共轭仅合成符号差，零额外访存/计算**）；alpha/beta 缩放（beta==(1,0) 直通、==(0,0) 不读 y）后一次 DataCopyPad 写回 y。
4. **N 快速路（`CgemvFastN`）——行拆分 + 全列点积，无跨核归约**：
   - 每核行段 \[rsStart, rsStart+rsCount)（32 行粒度均分，count 恒为 32 倍数），独立产出 y 直接写 GM；
   - **tile 常量**：`N_OUT_TILE=32` 行 × `N_DOT_TILE=128` 列，aTile 三缓冲（Prime-Pump-Drain 事件协议：MTE2\_V 装载就绪 / V\_MTE2 缓冲回收，跨 outTile 轮次残留 free 事件在 drain 后统一 wait 清理）；
   - **A tile 装载**：2D DataCopyPad（blockCount=128 列、blockLen=32 复数=256B、srcStride=(lda−32)×8B、dst 连续）；
   - **x 驻留 UB（stageX）**：`incx==1 && n≤4096 && n%4==0` 时 x 一次 DataCopy 入 UB，MAC 内 `DIST_BRC_B32` 单元素广播（替代逐列 GM 标量读——否则每核重复 GM 读 x n 次成为 N 路径瓶颈）；不满足条件走 `CgemvVecNMacGm` 标量兜底（保正确性）；
   - **MAC（`CgemvVecNMac<CHUNKS,PITCH>`）**：**4 路列展开**独立累加器（同列对同一 acc 的 FMA 构成串行链，vmula 延迟 ~7 cycle，4 列一组把链长降为 cols/4）；`acc1 += A·xR`、`acc2 += A·xI`（平面交错布局：每 32 行 chunk 64f = (ΣaR·xR, ΣaI·xR) 交错对）；u0 从 UB 载入跨 tile 部分和、u1..u3 清零、末尾合并写回；
   - **向量化 epilogue（`CgemvVecNEpilogue<BETA_ZERO,BETA_ONE>`）**：y 整块 MTE2 入 UB → DINTLR 去交错 acc1/acc2 → 寄存器内复数合成（`accR = ΣaR·xR − ΣaI·xI`，`accI = ΣaR·xI + ΣaI·xR`）→ alpha 复数乘 → beta 分支（BETA\_ZERO 跳过 y 读 / BETA\_ONE 直通加 / 一般复数乘）→ `DIST_INTLV` 交错存回 → MTE3 整块写 y；
   - incy≠1 或行段尾（构造上恒 32 对齐，标量补丁为防御性保留）逐元素处理。
5. **SIMT 兜底（`CgemvSimtN` / `CgemvSimtTC` / `CgemvSimtScale`）**：`asc_vf_call` 启动，`dim3{SIMT_MAX_THREAD_NUM}` 线程、grid-stride 循环，逐输出元素标量点积；负步长 `idx = (inc>=0) ? i*inc : (len-1-i)*|inc|` 对 x/y 通用；beta 语义镜像 cblas。覆盖任意 incx/incy/lda 与小尺寸。
6. **关键正确性设计点**：

| 设计点              | 处理方式                                                                 |
| :--------------- | :------------------------------------------------------------------- |
| 复数 MAC           | T/C：4 分量（PR/PI/QR/QI）+ 列尾 ReduceSum；N：平面交错双累加器 + DINTLR 去交错合成        |
| 共轭转置（OP\_C）      | 与 OP\_T 共用 MAC 数据通路，仅列尾合成符号差（accR=PR±PI / accI=QR∓QI），零成本，杜绝退化        |
| 负步长              | `idx = (inc>=0) ? i*inc : (len-1-i)*|inc|`，x/y 通用（AIV：`IndexWithStrideAiv`；SIMT：内联） |
| beta==(0,0) 不读 y | 各路径写回前判 `betaR!=0 || betaI!=0` 才读 y（防 0\*Inf=NaN）                    |
| beta==(1,0) 直通   | y 直通加，不走复数乘（防 1\*NaN 传播差异，cblas 语义）                                  |
| 累加器/缓冲清零         | `Duplicate` 只写不读（`Muls(acc,acc,0)` 先读 UB 残留值 → 0\*Inf=NaN）            |
| 多核并行             | T/C 按列、N 按 32 行段 base/rem 精确均分；无跨核归约、无 SyncAll                        |
| 索引溢出             | 全部索引 int64 计算                                                        |
| bisheng 编译器约束    | RegTensor 必须显式寄存器变量 + `if constexpr` 手工展开（数组+循环索引报 "Unsupported Inst must be hoisted"）；`__VEC_SCOPE__` 内循环变量须 uint16\_t |

7. **特殊情况与边界处理**：

| 特殊情况                     | 处理方式                                                                       |
| :----------------------- | :------------------------------------------------------------------------- |
| m=0 或 n=0                | host quick return SUCCESS，不下发 kernel                                        |
| alpha=(0,0) 且 beta=(1,0) | host quick return SUCCESS，不写 y                                              |
| alpha=(0,0) 且 beta≠(1,0) | `CgemvSimtScale` 核：y=beta\*y，不读 A/x（A/x 可为 nullptr，下发时以 y 地址替换形参，惯例同 cgeam） |
| beta=(0,0)               | kernel 不读 y 旧值（y 可为 NaN/未初始化，不发生 0×NaN）                                     |
| beta=(1,0)               | y 直通相加，βw 乘法保持精确恒等                                                          |
| 负步长 incx/incy            | 反向起点寻址 `idx=(len-1-i)*|inc|`，与 Netlib 一致                                    |
| incx/incy = INT32\_MIN   | host 校验拒绝（防止取绝对值溢出）                                                         |
| incx/incy = 0            | host 校验拒绝                                                                   |
| lda > m（padding）         | 寻址按 lda，pad 区不读取；lda==m 是快速路条件，padding 自动走 SIMT                              |
| m=1 / n=1 / 1×1          | gating（m%32==0 等）自然不满足 → SIMT grid-stride 覆盖（单线程有效工作，其余线程空转退出）              |
| A/x/y 含 Inf/NaN          | 按 IEEE754 传播；beta=(1,0) 直通、beta=(0,0) 跳读防 0×Inf=NaN；累加器 Duplicate 只写清零防 UB 残留污染 |
| 大尺寸                      | 全路径索引 int64；UB 预算不足时 `CgemvBuildFastTiling` 返回 0 自动回 SIMT                    |
| trans 非法（如 999/0xFF）     | host 返回 `INVALID_ENUM`                                                      |
| m%32≠0 / incx≠1 / incy≠1 | 自动回落 SIMT（语义等价，内部路径选择，非用户约束）                                                |

8. **实现约束（性能/精度回归锚定，改动前须回归对应用例）**：
   - `N_OUT_TILE=32`：64 行 tile 会实例化 `CgemvVecNMac<CHUNKS=2>`（26 个向量寄存器），-O3 下产生个别元素大误差；32 行方案回归全通过且大尺寸耗时仅小幅增加；
   - GlobalTensor 缓存 hint 保持默认 NORMAL（`CACHE_MODE_DISABLE` 显著劣化）；
   - **必须 Release -O3 编译**（CMake 默认）：Debug -O0 下标量胶水未优化，V 流水被译码节流，且 MTE2/V 无法重叠，kernel 耗时劣化数倍；
   - 诊断位 `tiling.phaseMask` 生产恒 0（仅性能测试直启使用，见 §3.2.4）。

### 性能调优设计

> 目标：回答"性能从哪来、瓶颈在哪、怎么验证、达标前提是什么"。
> 结论先行：**cgemv 是带宽受限（memory-bound）算子**，调优主线是"减少访存量 + 把有效带宽用满"；
> 计算侧（复数 MAC 展开）优化是第二位的，仅在 2048² 这类计算与访存同量级的场景才与访存同等重要。

#### 1) 访存量模型

单次调用总访存量（性能用例恒为 alpha=(1,0)、beta=(0,0)、incx=incy=1、lda=m，见测试 CSV perf 段）：

```
V = 8·m·n          （A 只读一次，8B/复数元素）
  + 8·xLen         （x 读一次；N 时 xLen=n，T/C 时 xLen=m）
  + 8·yLen         （y 写一次；N 时 yLen=m，T/C 时 yLen=n）
```

- beta≠(0,0) 时 y 需额外读 8·yLen；alpha=(0,0) 退化分支（mode=1）V = 8·yLen（不读 A、x）；
- 算术强度 I = 8·m·n / V ≈ **1 flop/B**（m、n 同量级时），远低于昇腾向量单元数十 flop/B 的平衡点
  → 带宽受限成立；其中 **A 的流读占 V 的 99.9%**（2048² 时 A 占 99.90%），是唯一需要认真对待的流量。

#### 2) roofline 分析（设计期，任务书标杆）

平台参数（arch35，`blas/common/arch/hardware.h`）：HBM 峰值带宽 4 TB/s（白皮书）、L2 峰值 20 TB/s / 容量 192 MB、UB 248 KB/AIV、AIV 核数 `GetAivCoreCount()`。

| case | trans | m    | n    | 总访存量 V     | T(HBM 峰值 4T/s) | T(HBM 60% 2.4T/s) | T(L2 峰值 20T/s) | T(L2 60% 12T/s) | 任务书标杆   | 标杆隐含有效带宽 |
| :--- | :---- | :--- | :--- | :--------- | :------------- | :---------------- | :------------- | :-------------- | :------ | :------- |
| 1    | N     | 512  | 512  | 2.105 MB   | 0.53 us        | 0.88 us           | 0.11 us        | 0.18 us         | 2.18 us | 0.97 TB/s |
| 2    | N     | 2048 | 2048 | 33.587 MB  | 8.40 us        | 14.0 us           | 1.68 us        | 2.80 us         | 3.86 us | 8.70 TB/s |
| 3    | T     | 2048 | 2048 | 33.587 MB  | 8.40 us        | 14.0 us           | 1.68 us        | 2.80 us         | 3.51 us | 9.57 TB/s |
| 4    | C     | 2048 | 2048 | 33.587 MB  | 8.40 us        | 14.0 us           | 1.68 us        | 2.80 us         | （无标杆，与 T 同量级） | — |
| 5    | N     | 4096 | 4096 | 134.283 MB | 33.6 us        | 56.0 us           | 6.71 us        | 11.2 us         | （无标杆，记录基线） | — |

**关键结论（设计主线由此导出）**：

1. **512×512（标杆 2.18us）是开销受限场景**：数据搬运理论时间 <1µs，标杆时间主要覆盖 tile 启动、x 装载、
   同步等固定开销；优化重点是控制小尺寸固定开销（快速路常驻路径、避免多余同步）。
2. **2048² 名义标杆（3.86/3.51us）隐含有效带宽 ≥8.7 TB/s**，高于 HBM 峰值屋顶线（4 TB/s 下纯冷读下限 8.40us）
   → 达标**必须依赖 L2 缓存驻留**：A 工作集 33.5 MB ≪ L2 192 MB，L2 60% 时理论 2.80us < 3.86us。
   设计主线由此确定为：**列拆分使每核列段 GM 连续（1D 大块流读）+ 3-slot 流水保 MTE2 满带 + tile 级数据复用提升 L2 命中率**。
3. T/C 与 N 访存量相同，但 T/C 每核整列流读（GM 连续）更易打满带宽；N 需 2D DataCopyPad 跨列取行片，
   通过 x 驻留 UB + 三缓冲弥补访存模式差异。



### 三级耗时统计设计

三级计时互相独立、口径互补，用于定位"耗时花在哪一级"：

**第 1 级：kernel 阶段（phaseMask 差分，仅测试直启使用，生产恒 0）**

`cgemv_tiling_data.h` 定义诊断位，host 以不同 mask 多次直启 `cgemv_kernel_do`（warmup+samples，墙钟取中位数），差分得设备侧各阶段耗时；mode=2 空 kernel 给出 launch 地板（floor）：

| 位 | 常量 | 含义 |
| :-- | :-- | :-- |
| bit0 | `CGEMV_PM_SKIP_MAC` | 跳过 MAC（保留 load 流水）→ 与 floor 差 = load；与 full 差 = epi 边际 |
| bit1 | `CGEMV_PM_SKIP_EPILOGUE` | 跳过 epilogue（y 写回）→ 与 floor 差 = load+MAC；与 load 差 = mac 边际 |
| bit2 | `CGEMV_PM_DBL_MAC` | MAC 执行两遍（纯计算成本测量，重叠能力判定） |
| bit3 | `CGEMV_PM_SPIN_V` | MAC 换纯寄存器 FMA 自旋（MTE2/V 并行能力判定：边际≈0=可并行） |

**第 2 级：host 四阶段（每次 API 调用自动记录）**

`cgemv_host.cpp` 以 `steady_clock` 记录 validate / tiling / wsAlloc / launch / total 五项墙钟写入 thread\_local `CgemvHostTiming`，经 `CgemvHostTimingGet` 导出；kernel 设备耗时不含在内（异步提交）。

**第 3 级：测试套件**

- 精度：`*CsvDriven*`（1200 用例，CSV 驱动），逐 real/imag 通道报 matchedRatio / maxAbsErr / failed indices；
- 性能：`*CgemvPerf*`（200 用例），每用例输出 `[PF]`（wall/floor/带宽/roofline/verdict）、
  `[PF-PHASE]`（load/mac\_marg/epi\_marg 分解）、`[PF-HOST]`（host 四阶段），并落盘 CSV 报告；
- 运行入口：`./run_cgemv_test.sh [smoke|accuracy|perf|all]`；环境变量 `CGEMV_BENCH_WARMUP`（默认 10）、
  `CGEMV_BENCH_SAMPLES`（默认 55，<50 断言）。

## 特性交叉分析

| 交叉维度                    | 设计关注点                                              | 应对策略                                                                     |
| :---------------------- | :------------------------------------------------- | :----------------------------------------------------------------------- |
| trans × 步长              | 三分支 × 正负步长的寻址正交性                                   | 寻址公式仅依赖（len, inc），与 trans 解耦；N 快速路要求 incx==1，其余组合自动回落 SIMT；用例覆盖 N/T/C × incx/incy ∈ ±1/±2/±3 全组合 |
| trans × 形状              | 宽（m≪n）/窄（m≫n）矩阵下 outDim 与内积维互换                     | T/C 按列拆核（coreNum≤n）、N 按行段拆核（coreNum≤m/32），base/rem 均分；n 小于核数时部分核空转退出；宽窄用例覆盖 |
| alpha/beta 特殊值 × 指针合法性  | alpha=0 时 A/x 可为空、beta=0 时 y 可为 NaN                | 校验条件带 alpha/beta 判零；kernel 按 mode/beta 分支跳读；空指针下发时以 y 地址替换形参防空描述符          |
| OP\_C × 复数运算            | 共轭仅作用于 A，不作用于 x/alpha/beta                         | 与 OP\_T 共用 MAC 数据通路，列尾合成符号差（accR=PR±PI），写回缩放不变；纯虚 alpha + OP\_C 专项用例覆盖     |
| gating × 语义等价           | 快速路条件（lda==m、incx/incy==1、m%32==0、UB 预算）与 SIMT 回落   | 两路径语义等价、共用同一 golden 验证；边界尺寸（m=32 倍数±1）专项用例覆盖双路径切换                          |
| Inf/NaN × 累加器管理         | 特殊值传播行为与 golden 一致性；UB 残留污染                         | beta=(0,0) 跳读、beta=(1,0) 直通防 0×Inf/1×NaN；累加器 Duplicate 只写清零；TC\_FL 专项（A/x/y 分别含 Inf/NaN） |
| 与 sgemv/其他产品线共存         | 同名 API 多架构实现、符号唯一性                                  | arch35 源文件仅在 ascend950 构建收集；接口签名以 include 头为准，无平行 API                     |
| 异步 × 测试读回               | stream 未同步读回脏数据                                    | 测试 wrapper 在读回前同步 stream/device；文档明确异步语义；workspace 常驻复用靠 stream 顺序保证安全     |
| 三级计时 × 生产路径             | 诊断设施不得影响生产性能                                        | phaseMask 生产恒 0（分支可被编译器消除）；host 计时为 thread\_local 少量写内存；mode=2 空 kernel 仅测试使用 |

## 支持硬件

| 支持的芯片版本              | 涉及勾选 |
| :------------------- | :--- |
| Ascend 950PR（arch35） | √    |

（A2/A3 的 arch22 已有实现，不在本次改动范围。）

## 算子约束限制

- m ≥ 0、n ≥ 0、lda ≥ max(1, m)、incx/incy ≠ 0 且 ≠ INT32\_MIN；alpha/beta 不可为 nullptr；非法数值参数返回 `ACLBLAS_STATUS_INVALID_VALUE`，非法枚举返回 `ACLBLAS_STATUS_INVALID_ENUM`；
- A/x/y 为 Device 内存，alpha/beta 及标量参数为 Host 内存；y 原地覆写；
- 复数类型为 `aclblasComplex`（float 实部+虚部交织），与 cuComplex 内存布局一致；
- 异步语义：依赖 `aclblasSetStream` 绑定 stream，读回结果前须同步 stream；
- 不支持超出 lda/incx/incy 语义的非连续访问；无广播、无确定性计算要求；
- N 快速路预留 handle 常驻 workspace（coreNum×2m floats），常驻不足时按调用分配并同步回退（打告警日志），不新增 device 常驻内存；
- 快速路 gating（lda==m、incx=incy=1、m%32==0）为**内部路径选择条件**，不满足时自动回落 SIMT，不构成用户约束；
- 编译约束：性能验收须 Release -O3（CMake 默认），Debug 构建仅供调试。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述                                                                                                                        | 标准来源          |
| :--- | :------------------------------------------------------------------------------------------------------------------------ | :------------ |
| 精度标准 | COMPLEX64 实部/虚部分别按 FLOAT32：atol=2⁻¹⁶≈1.526e-5、rtol=2⁻¹⁰≈9.766e-4、matched\_ratio≥0.99、max\_abs\_error≤1e-2 或 32×ULP；逐元素 `\|actual-golden\| ≤ atol + rtol×\|golden\|`；golden 由 cblas（Netlib cgemv 复数实现）生成，输出 y 全量验证 | 任务书 §3.2 + 生态算子开源精度标准 |
| 性能标准 | **OP\_N 512×512 ≤ 2.18us、OP\_N 2048×2048 ≤ 3.86us、OP\_T 2048×2048 ≤ 3.51us**（warmup 后有效采样 >50 次取平均） | 任务书 §3.3 |
| 内存标准 | 不新增 device 常驻内存：N 快速路优先复用 handle 自带常驻 workspace（coreNum×2m floats），常驻不足时按调用分配并在 launch 后释放（同步回退，打告警）；热路径（常驻复用命中时）零 `aclrtMalloc` | 任务书 §3.3 + 仓内 workspace 管理惯例 |
| 自验标准 | 自测用例（任务随附 CSV，覆盖下述全部类别）全量通过；性能 case 按任务书 §3.3 标杆方法采集并记录，输出自测报告                                                                | 任务书 §3.5      |

## 自测用例设计

测试用例以 CSV 描述，由 C++ GTest 工程加载后调用 `aclblasCgemv` 执行，精度 golden 由 cblas（Netlib 复数实现）生成；覆盖类别与任务书 §3.5 一致（用例以 `TC_XX_NNN` 前缀编号，如 TC\_SQ\_029、TC\_RC\_118、TC\_FL\_231、TC\_CV\_250、TC\_PF\_1003，全量清单见 `cgemv_test.csv`）：

| 类别           | 覆盖内容                                                            | 示例前缀      |
| :----------- | :-------------------------------------------------------------- | :-------- |
| 基础功能         | trans=N/T/C 全覆盖，小尺寸与常规方阵                                        | TC\_L0    |
| 尺寸扫描         | m、n：0、1、小质数、2 的幂及 2 的幂±1、非对齐值、直至大规模                             | TC\_SQ / TC\_CV |
| 标量特殊值        | alpha/beta：均匀分布与正态分布各 50%，含 (0,0)、(1,0)、纯虚数、大值                  | TC\_AB / TC\_ED |
| 非方阵          | m\<n 宽矩阵、m>n 窄矩阵（与 m、n 正交组合）                                    | TC\_RC    |
| 前导维          | lda=max(1,m) 紧凑场景与多个 padding 场景                                 | TC\_LD    |
| 步长           | incx/incy 覆盖 ±1、±2、±3（含 OP\_C 组合）                               | TC\_INC   |
| quick return | m=0、n=0、alpha=(0,0) 且 beta=(1,0)                                | TC\_ED    |
| 退化计算         | alpha=(0,0) 且 beta≠(1,0)（y=beta\*y）、beta=(0,0)（y 输入不被使用，可为未初始化） | TC\_ED / TC\_CV |
| 负向用例         | 空指针、非法枚举、负维度、非法 lda、零步长                                         | TC\_ED    |
| 填充特殊值        | 全零/交替/极值/Inf/NaN（A、x、y 分别）                                      | TC\_FL    |
| 性能用例         | 任务书 §3.3 三个 case 及扩展规模（TC\_PF 200 用例，含阶段分解与 host 四阶段采集）         | TC\_PF    |

负向用例期望返回：

| 场景                                                                                            | 期望返回                                 |
| :-------------------------------------------------------------------------------------------- | :----------------------------------- |
| handle == nullptr                                                                             | ACLBLAS\_STATUS\_HANDLE\_IS\_NULLPTR |
| trans 不在 {N,T,C}                                                                              | ACLBLAS\_STATUS\_INVALID\_ENUM       |
| m<0 / n<0 / lda\<max(1,m) / incx==0 / incy==0 / alpha 或 beta 为 nullptr / 计算所需 A、x、y 为 nullptr | ACLBLAS\_STATUS\_INVALID\_VALUE      |

## 可维护性分析

1. 代码按 ops-blas 仓既有架构落位：host 侧 `blas/gemv/arch35/cgemv_host.cpp`、kernel 侧 `cgemv_kernel.cpp`、tiling 结构 `cgemv_tiling_data.h`，与同目录 sgemv 文件命名与风格一致；
2. 双路径收敛于统一 tiling 结构（`CgemvTilingData`）：host 只产出 tiling，kernel 只消费 tiling，路径选择/gating 集中在 `CgemvBuildFastTiling` 单点，便于定位与单测；
3. 关键实现约束（N\_OUT\_TILE=32、Duplicate 只写清零、NORMAL 缓存、Release -O3）均以注释锚定在代码常量处，并注明对应回归用例编号，改动前可追溯；
4. 三级耗时统计为常驻能力（诊断位生产路径零开销）：性能问题可按"测试级 → host 级 → kernel 阶段"逐级下钻，诊断方法沉淀于 `blas/gemv/arch35/README_cgemv.md`；
5. 同步新增 `blas/gemv/arch35/README_cgemv.md`（接口/架构/计时机制/性能分析）并随 PR 提交测试 CSV 与 GTest 工程，保证验收人可复现。

## 交付件清单

1. **算子实现**：`blas/gemv/arch35/cgemv_host.cpp`、`cgemv_kernel.cpp`、`cgemv_tiling_data.h`，提交 ops-blas fork（邀请 Ascend-CANN 为开发者）；
2. **测试工程**：`test/gemv/cgemv/arch35/`（`cgemv_test.csv` 全量用例 + GTest 代码 `cgemv_test.cpp`/`cgemv_test_perf.cpp` + 测试辅件 `cgemv_param.h`/`cgemv_npu_wrapper.h`/`cgemv_golden.h` + `README.md`，明确精度/性能 case 与复现步骤）；
3. **三级耗时统计**：kernel phaseMask 差分（含 DBL\_MAC/SPIN\_V 诊断位）+ host 四阶段计时 + 测试套件 `[PF]`/`[PF-PHASE]`/`[PF-HOST]` 输出与 CSV 报告；
4. **文档**：`blas/gemv/arch35/README_cgemv.md`（接口/架构/三级计时/性能分析）、`blas/gemv/README.md` 产品支持表更新（Ascend 950PR：支持）；
5. **自测报告**：用例参数、精度对比结果（实部/虚部分别）、性能数据（按任务书 §3.3 标杆方法采集）、复现步骤。

## 兼容性分析

- 接口复用 ops-blas 仓 `include/cann_ops_blas.h` 已有 `aclblasCgemv` 声明，与其他产品线共用，未新增 950PR 私有平行接口；
- 参数序列与 `cublasCgemv` 一一对应（handle 及参数顺序一致，复数类型由 cuComplex 对应为 aclblasComplex），无需额外映射说明；
- 构建按 `SOC_ARCH_DIRS` 自动收集 arch35 源文件，不修改 sgemv 及 arch22 cgemv 既有实现与行为，对仓内其它算子无侵入；
- 顶层 CMake 的 Release 默认仅改变构建类型默认值，可用 `-DCMAKE_BUILD_TYPE=Debug` 显式覆盖，不影响既有调试流程；
- `blas/gemv/README.md` 补充 aclblasCgemv 参数说明章节与产品支持表（Ascend 950PR：支持）。
