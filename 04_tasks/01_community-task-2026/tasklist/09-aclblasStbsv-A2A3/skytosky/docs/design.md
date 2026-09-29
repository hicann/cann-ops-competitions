# 需求背景（required）

## 需求来源

- 任务：9月社区任务-aclblasStbsv算子开发(A2A3)，在 Atlas A2/A3（arch22）上开发单精度实数三角带状方程组求解算子 `aclblasStbsv`。
- 团队及提交账号：skytosky。
- 依据：任务方提供的《aclblasStbsv_A2A3 任务书》及配套 `test_cases/`（1200 条 CSV 用例、200 条 GPU 基线、生成与校验脚本）。
- 目标代码仓：[cann/ops-blas](https://gitcode.com/cann/ops-blas)，算子代码入 `blas/tbsv/arch22/`，测试代码入 `test/tbsv/stbsv/arch22/`；硬件 Atlas A2/A3，CANN 9.1.0。
- 本 PR 为设计文档评审稿；实现代码在评审通过后按 ops-blas 流程单独提交。

## 背景介绍

### aclblasStbsv 功能与工程现状

Stbsv 求解三角带状线性方程组 `op(A) * x = b`，A 为 n×n 上/下三角带状矩阵（半带宽 k），列主序束带存储于 lda×n 数组；输入向量 x 存放右端项 b，调用后解原地写回 x。对标接口为 cuBLAS `cublasStbsv`，计算语义参考 Netlib BLAS `stbsv.f`。

以本设计核对时的 ops-blas 提交 `4c3dfa98a78342e9672f2663441e7a883023ff89`（2026-09-29）为工程基线，现状如下：

- 公共接口已就绪：`include/cann_ops_blas.h` 已声明 `aclblasStbsv`（handle, uplo, trans, diag, n, k, A, lda, x, incx），状态码与枚举定义在 `include/cann_ops_blas_common.h`。**本任务无需修改公共头文件**。
- 同族实现可参照：`blas/tbsv/arch35/` 已有面向 Ascend 950 的 Stbsv 实现（host/kernel/tiling_data 三件套），其带状寻址、Netlib 语义（零主元跳过、INT_MIN 步长拒绝）与测试框架（`test/tbsv/stbsv/` 的 param/golden/CMake 注册）可直接复用与对齐。
- arch22 工程范式已验证：`blas/trsv/arch22/` 已有 AIV-only 向量核实现（模板类 + `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY)` + `<<<blocks, nullptr, stream>>>` 启动），`blas/CMakeLists.txt` 对 `blas/*/arch22/*.cpp` 自动收集，无需新增算子侧 CMake。
- 已知差距：`blas/trsv/arch22/strsv_host.cpp` 现实现将 A/x 视为 Host 指针做 H2D/D2H 拷贝并中途同步 stream，与本任务书"A/x 为 Device 指针、异步执行依赖 `aclblasSetStream`"的语义不符；本设计遵循任务书语义（Device 指针直通 kernel、仅入队不等待），不沿用它。

本任务采用 Ascend C kernel 直调（句柄式 BLAS 接口），不涉及 TBE 原型、ACLNN 两阶段接口或算子注册 JSON。

# 需求分析（required）

## 需求描述

在 CANN 9.1.0 上实现符合任务书与 cuBLAS `cublasStbsv` 语义的 arch22 版本：覆盖 uplo×trans×diag 全部 12 组枚举组合、k≥0 全带宽、正负步长、原地写回；参数校验返回值与任务书 §2.4 一致；精度满足生态算子 FLOAT32 混合容差标准；任务书 5 条典型性能 case 及 200 条 GPU 基线 case 达标。

## 需求拆解

1. 功能：op(A) 由 trans 决定（N/T/C，实数档 C 与 T 等价）；uplo 取上/下三角带；diag=UNIT 时对角视为 1 且不读写 A 的对角位置；n=0 合法 no-op；k=0 退化为纯对角三角阵。
2. 布局：列主序束带寻址（UPPER 对角位于带内第 k 行、LOWER 位于第 0 行，0 基），lda≥max(1,k+1) 支持列尾 padding；x 支持正负步长（incx≠0），仅写逻辑元素、不动步长间隙；带外与填充区不读取。
3. 边界与负向：handle 空 → `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`；非法枚举、n<0、k<0、lda<max(1,k+1)、incx=0、incx=INT_MIN、n>0 时 A/x 空指针 → `ACLBLAS_STATUS_INVALID_VALUE`；n=0 在全部标量校验通过后返回 SUCCESS。
4. 计算：按三角依赖顺序求解，FLOAT32 全程；还原 Netlib N 路径"右端项为零跳过本列更新"的语义以对齐 Inf/NaN 行为。
5. 精度：golden 由 Netlib `stbsv` 生成，逐元素混合容差判定（见可维可测分析）。
6. 性能：warmup 后有效采样>10 次取平均 kernel 耗时（msprof），5 条任务书 case 不超上限，其余 200 条逐条≤`gpu_ms/0.8`。
7. 交付：按第 4 节交付件清单完成自测报告与 `task_submission/` 七件套，私仓邀请 Ascend-CANN。

# 详细设计（required）

## 算子分析

### 数学公式

以下标 0 基。`A[p, q]` 表示带状存储数组第 p 行第 q 列元素，`a(i,j)` 表示数学矩阵元素：

```text
LOWER：a(i,j) = A[i-j, j]      ，j ≤ i ≤ min(n-1, j+k)     （对角位于带内第 0 行）
UPPER：a(i,j) = A[k+i-j, j]    ，max(0, j-k) ≤ i ≤ j       （对角位于带内第 k 行）
```

带外元素恒为 0，不参与计算，即使 k>n-1（UPPER 存储偏移仍按入参 k，不用有效带宽替换）。令 M=op(A)，d(i)=1（UNIT）或 M(i,i)（NON_UNIT）：

```text
M 为下三角：x(i) = ( b(i) - Σ_{j=max(0,i-k)}^{i-1}  M(i,j)·x(j) ) / d(i) ，i = 0..n-1     前代
M 为上三角：x(i) = ( b(i) - Σ_{j=i+1}^{min(n-1,i+k)} M(i,j)·x(j) ) / d(i) ，i = n-1..0   回代
```

| 原矩阵 uplo | trans | op(A) 三角 | 求解方向 |
| --- | --- | --- | --- |
| LOWER | N | 下三角 | 前代（i 递增） |
| UPPER | N | 上三角 | 回代（i 递减） |
| UPPER | T / C | 下三角 | 前代 |
| LOWER | T / C | 上三角 | 回代 |

Netlib `stbsv` 的 N（更新）路径中，若当前求得主元 `x(j)==0` 则跳过该列对剩余分量的更新（避免 0×Inf 产生额外 NaN），本设计保留该语义；T 路径无此跳过。除法一律用 FP32 除法，不用倒数乘法近似。对角元素奇异时结果无定义，不做奇异性检查。

### 支持数据类型与形状

- 仅 FLOAT32；中间量 FP32；不降精度、不引入 TF32。
- n≥0、k≥0、lda≥max(1,k+1)；n 为运行时标量，无 dynamic shape 要求。
- x 物理跨度 `1+(n-1)×|incx|`（n>0），incx 可正可负、不可为 0 或 INT_MIN（|−INT_MIN| 溢出，与配套测试期望一致返回 INVALID_VALUE）。
- 不涉及广播；不支持超出 lda/incx 语义的非连续布局；A 与 x 不允许内存重叠（调用方保证）。
- 测试规模 n≤4096、k≤n-1（满带）；设计不据此设置接口上限，更大规模由分段窗口模式覆盖（见 kernel 侧设计）。

### 地址计算

以元素为单位、64 位中间量计算，防溢出：

```text
A 元素：offset = int64(col) × lda + row_in_band
x 元素：start = incx > 0 ? 0 : int64(n-1) × (-int64(incx))
        offset = start + int64(i) × incx        （i 为逻辑序号）
op 系数：M(i,j) = uplo/trans 组合按上式映射到 A 存储（T/C 交换行列索引，C 不取共轭）
```

## 算子实现

### 实现方案

核心思路：**把三角带求解改写为"逐主元的连续段向量运算"**。列主序束带存储下，无论 N/T，任一主元步骤涉及的系数都连续（按列存），N 路径的更新区间与 T 路径的点积窗口也都是 x 的连续段（incx=1 时），因此每步只需 4~6 条向量指令即可完成，串行依赖链的每步开销可压进性能预算（见性能优化方案）。整体流程：参数校验 → tiling 计算 → 启动 kernel（异步）→ 返回，读回前由调用方同步 stream。

#### Host 侧设计

参数校验顺序（与 `blas/tbsv/arch35/stbsv_host.cpp` 现行实现一致）：

1. handle 为空 → `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`；
2. n<0、k<0 → `ACLBLAS_STATUS_INVALID_VALUE`；
3. 枚举 uplo∈{UPPER,LOWER}、trans∈{N,T,C}、diag∈{NON_UNIT,UNIT}、lda≥max(1,k+1)、incx≠0 且 ≠INT_MIN → 违反返回 `ACLBLAS_STATUS_INVALID_VALUE`；
4. n>0 时 A/x 非空检查（含 k=0 且 UNIT 的情形，与任务书"n>0 时 nullptr 即 INVALID_VALUE"一致）；
5. n=0 → 返回 SUCCESS（不启动 kernel、不访问 A/x）；k=0 且 diag=UNIT → 返回 SUCCESS（x 恒等）；
6. 填充 tiling POD，经 handle 取 stream，`kernel_do` 分派启动，**不做任何同步等待**（异步语义）。

不采用 `blas/trsv/arch22/strsv_host.cpp` 的"n<=0 先于参数校验返回"与"Host 指针 H2D 拷贝"两种与任务书相悖的写法。

Tiling 为 POD 结构体随 kernel 参数按值传递（同 arch35 stbsv 现行方式，不额外申请 device 缓冲、不产生 H2D 拷贝）：

```cpp
struct StbsvTilingData {
    uint64_t a;        // A 设备地址
    uint64_t x;        // x 设备地址
    uint32_t n;
    uint32_t k;
    uint32_t lda;
    uint32_t uplo;     // 冗余存放，供调试与 GM 回退路径使用
    uint32_t trans;
    uint32_t diag;
    int32_t  incx;     // 原值（可负）
    uint32_t winLen;   // 滑窗长度 W = min(k, n-1)（k>0 时）
};
```

分核策略：

- **k=0 且 NON_UNIT**（纯对角三角阵）：x(i)=b(i)/A[0,i]，各元素独立，按 40 个 AIV 核均分（每核连续段），专用对角 kernel 多核启动；
- **k=0 且 UNIT**：host 早退（见上）；
- **k>0**：串行依赖链决定主求解使用单 AIV 核（0 号），核内以向量指令完成每步；不将行间独立任务跨核拆分（依赖链上无合法并行点），也不使用无保障的跨核自旋同步。

数据分块与内存优化策略（单核，k>0 路径）：

- UB 预算（FLOAT32，按满带 k=n-1=4095 估算，910B 单 AIV 可用 UB 约 192KB，代码经平台接口实取）：

| 缓冲 | 用途 | 容量 |
| --- | --- | --- |
| bandBuf[2] | 列带系数双缓冲预取 | 2 × W × 4B |
| pendBuf | 未求解分量滑窗（pending b）N 路径用 | (W+1) × 4B |
| solBuf | 已求解分量滑窗 T 路径用 | W × 4B |
| bSeg[2] | b 段双缓冲预取 | 2 × SEG × 4B |
| xStage | 写回聚合栈 | SEG × 4B |

  合计 (4W+3SEG)×4B ≈ 112KB（W=SEG=4096 满带极限值），实际性能用例 k≤128 时 < 5KB；测试范围（k≤4095）内单窗口即可容纳，不触发 GM 分段回退（N/T 两路径的滑窗分别独立使用，此处取并集保守预算）。窗口超出 UB 的极端大 k 场景降级为分段窗口模式：窗口只持有 [j, j+SEG) 段，段首求解、段间经 GM 往返，正确性不变、仅吞吐下降。
- 双缓冲：列带系数与 b 段各两份，`DataCopyPad`（处理 lda≠k+1 的列尾与首尾非 32B 对齐）预取下一步所需数据，与当前步计算重叠，把 GM 访存延迟移出串行关键路径；
- 负步长 x：逻辑序号到物理偏移的映射在 tiling 内换算（见地址计算），UB 滑窗始终按逻辑序维护，写回按物理偏移逐元素 `DataCopyPad`/SetValue 散写（incx≠1 为正确性路径，性能 case 均为 incx=1 连续写）。

tiling key 规划：本算子为 kernel 直调，不注册图算子 tiling key。路径分派完全由 host 依据公开入参完成：k=0 → 对角 kernel（多核）；k>0 → 8 个模板入口之一（`<UPLO, TRANS, DIAG>` 编译期展开，N/T 与 UNIT/NON_UNIT 分支零运行时开销）。

#### Kernel 侧设计

入口组织沿用 arch35 stbsv 的宏展开模式（8 个 `__global__ __aicore__` 模板实例 + 1 个对角入口），`KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY)`，`<<<blocks, nullptr, stream>>>` 启动：

**N 路径（更新式，Netlib 同构）**——以 LOWER/N 前代为例，UB 滑窗 `pend[0..W]` 维护尚未求解分量的当前值：

1. 预取：`DataCopyPad` 装载 b 段与列带 bandBuf；
2. 求解：`x(j) = pend[0]`（NON_UNIT 时标量除以对角，对角即列 j 第 0 元素）；
3. Netlib 跳过语义：若 `x(j)==0`，不更新、只滑窗；
4. 更新：`Muls(tmp, bandBuf[0..L), x(j))`；`Sub(pend[1..L], pend[1..L], tmp)`，L=min(k, n-1-j)；滑窗 `DataCopy` UB 内左移一位并装载新 b；
5. 写回：x(j) 计入写回栈，连续段满后整段 `DataCopy` 回 GM（incx=1）。

每步 2 条向量运算 + 1 次 UB 滑移 + 标量除法，列带与 b 段由双缓冲预取隐藏访存。

**T/C 路径（点积式）**——以 UPPER/T 前代为例（已求解滑窗 `sol[0..W)` 维护最近 W 个解）：

1. 预取列 i 带内系数段（k 个连续元素，单位对角时不读第 k 行）；
2. `Mul(tmp, coef, sol[0..k))` → `ReduceSum` 得 s；
3. `x(i) = b(i) - s`（NON_UNIT 再除以列 i 第 k 行对角元素）；
4. `sol` 滑窗右移纳入 x(i)，写回同 N 路径。

关键正确性约束：

- 严格三角访问：循环范围先排除未引用三角；UNIT 在 load 前排除对角，不做"整段加载后掩码替换"；`DataCopyPad` 搬运段完全位于允许区域，斜边与尾部用精确长度，不为对齐扩大读取；padding 区不参与运算；
- A 只读、x 仅写逻辑位置、步长间隙不动；
- 特殊值：除法与加减按 IEEE FP32 传播；N 路径保留零主元跳过；不引入 FMA 收缩改变舍入顺序；
- 尾块与极小 n（n≤32 或 k=1 等退化形态）走同一模板的短循环分支，不单设正确性存疑的特例。

### 性能优化方案

串行依赖链是本算子的固有约束，优化目标是把每步开销压进任务书预算。按任务书 5 条 case 折算单步预算（Avg time / n）：

| case | n | k | uplo/trans/diag | 上限 us | 单步预算 ns |
| --- | --- | --- | --- | --- | --- |
| 1 | 256 | 8 | UPPER/N/NON_UNIT | 198.0 | 773 |
| 2 | 512 | 32 | LOWER/N/NON_UNIT | 379.8 | 742 |
| 3 | 1024 | 16 | UPPER/T/UNIT | 659.5 | 644 |
| 4 | 2048 | 64 | LOWER/T/NON_UNIT | 1725 | 842 |
| 5 | 4096 | 128 | UPPER/N/UNIT | 2365 | 577 |

本设计每步关键路径为 2~3 条 UB 向量指令 + 1 次标量除法（UB 指令延迟约数十 ns 量级），列带/b 段 GM 访存由双缓冲移出关键路径，预期每步 200~400ns，对最紧的 case 5（577ns）保留约 30% 余量；总访存量 ≤ 3×(k+1)×4B×n（case 5 约 6MB，远低于带宽上限）。

调优与验证手段：

- msprof 对单条 `TC_PF` 用例采集 `OpBasicInfo.csv`，按 kernel 名统计 `Task Duration(us)` 均值（口径同 `test_cases/README.md`）；分析运行与正式计时分离；
- 参数扫描：SEG 段长、双缓冲深度、（若需要）滑窗与 GM 直读写模式的切换阈值；
- 预留优化（仅当实测不达标时启用）：对 k 较大的长链采用"对角块内串行 + 块间流水"的两级划分（前 k 行的解不阻塞更远行的 b 预取），不引入跨核同步语义。

### 工程组织与构建安装

```text
include/cann_ops_blas.h                 # 已有声明，不改
blas/tbsv/README.md                     # 产品支持表更新：Atlas A2/A3 支持
blas/tbsv/arch22/                       # 新增（blas/CMakeLists.txt 自动收集，无需 CMakeLists）
    stbsv_host.cpp                      # 参数校验、tiling、启动
    stbsv_kernel.cpp                    # AIV 向量核：8 模板入口 + 对角入口
    stbsv_tiling_data.h                 # POD tiling（与 kernel 侧共用）
test/tbsv/stbsv/arch22/                 # 新增
    stbsv_npu_wrapper.h                 # 直调 wrapper（对齐 arch35 命名）
    stbsv_test.cpp                      # CSV 驱动 GTest（frame: blas_test/csv_loader/fill）
    stbsv_test.csv                      # 随任务 1200 条用例
    gpu_baseline.csv                    # 200 条性能基线
test/tbsv/stbsv/CMakeLists.txt          # 如需注册 arch22 目标则最小修改
docs/zh/api_list.md                     # 公共接口索引补充（如缺）
```

复用 `test/tbsv/stbsv/stbsv_param.h`、`stbsv_golden.h`（arch35 已建，字段与浮点 golden 对 arch22 通用；golden 为 Netlib `stbsv` 自实现）。构建按仓内 `build.sh --soc ascend910b3`（Release）；`ops_blas_add_gtest_tests()` 注册测试；自验环境为 Atlas 800T A2（910B3），A3 环境在验收阶段复跑同 arch22 产物。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas A2 系列产品（含 Atlas 800T/I A2，910B3） | √（本次实现与自验） |
| Atlas A3 系列产品（含 Atlas 800I A3） | √（同 arch22 代码，验收阶段复验） |

## 算子约束限制

- 仅 FLOAT32、单矩阵单右端项；无 alpha/beta、无批量、无广播；A 与 x 不得重叠；
- NON_UNIT 下对角非零由调用方保证，不做奇异性检测，奇异问题结果无定义；
- x 原地读写；异步执行，读回前须同步 handle 绑定的 stream；
- 不要求逐位确定性，浮点顺序差异须落在任务书精度标准内；
- 带外、填充区、UNIT 对角与 x 步长间隙均不读写（毒化 + 哨兵用例验证）。

# 可维可测分析

## 精度标准/性能标准

| 项目 | 验收标准 | 来源 |
| --- | --- | --- |
| 精度 golden | Netlib BLAS `stbsv`，两侧输入严格一致 | 任务书 §3.1 |
| 逐元素 | abs(actual-golden) ≤ atol + rtol×abs(golden)，rtol=atol=2⁻¹³ | 任务书 §3.2 |
| 整体 | matched_ratio ≥ 0.99 且 max_abs_error ≤ 1e-2（或 32×ULP；大数规约误报可放宽至 2ULP，单独报告） | 任务书 §3.2 |
| 性能 | warmup（msprof 自带 5 次）+ 有效采样 >10 次平均；5 条典型 case 不超任务书上限；其余 200 条 ≤ gpu_ms/0.8 | 任务书 §3.3、test_cases/README |
| 内存 | 无额外 workspace（tiling 按值传递）；记录 UB 用量与 NPU 峰值显存，无越界泄漏 | 任务书 §3.4 |

测试组织（沿用配套 1200 条 CSV，前缀分类见 `test_cases/README.md`）：

- 枚举组合 12 组全覆盖；n 扫描（0/1/质数/2 的幂±1/非对齐/2048）、k 扫描（0/1/小值/半带/满带）、lda padding、±步长；
- 负向：空 handle、非法枚举、n/k<0、lda 非法、incx=0/INT_MIN、n>0 时 A/x 空指针；n=0 与非法标量组合的返回顺序；
- 特殊值：Inf/NaN 填充（NON_UNIT 对角强化远离零，UNIT 对角 NaN/Inf 不读验证）、全零/交替/极值；N 路径零主元跳过与 golden 对齐专项用例；
- 只读语义毒化：未引用三角、padding、x 间隙填哨兵值，调用后校验未被改写；
- 性能/内存：TC_PF 200 条，msprof kernel 级计时，逐条关联 `gpu_baseline.csv`（关联键 n/k/uplo/trans/diag/incx）；
- 每次真机自测留存日志（sha256 绑定提交号），自测报告按官方模板给出参数、精度对比、性能均值与内存数据。

## 兼容性分析

新增算子实现，不改公共接口签名、枚举值、状态码语义与其他算子行为；`aclblasStbsv` 声明已存在，本任务仅补 arch22 实现与测试，与 arch35（950）实现互不影响；产品支持表由"不支持"更新为"支持"（Atlas A2/A3），与实际验证范围一致。

## 参考资料

- 任务书及配套用例：`aclblasStbsv_A2A3_task_doc.md`、`test_cases/`（gen_csv.py / stbsv_test.csv / gpu_baseline.csv / verify_accuracy.py）。
- ops-blas 工程基线 `4c3dfa98a78342e9672f2663441e7a883023ff89`：`blas/tbsv/arch35/`（同族实现与 Netlib 语义）、`blas/trsv/arch22/`（arch22 kernel 直调范式）、`test/tbsv/stbsv/`（测试框架）、`agent/skills/repo-knowledge/`（BLAS 命名与参数顺序规范）。
- [Netlib stbsv.f](https://www.netlib.org/blas/stbsv.f)、[cuBLAS stbsv](https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-bsv)：接口语义、带状存储与特殊值行为参考。
- [生态算子开源精度标准（混合容差）](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/mixed_tolerance_standard.md)。
- Ascend C 官方文档：数据搬运（DataCopyPad 非对齐/尾块）、矢量计算 API（Muls/Sub/ReduceSum）、`KERNEL_TASK_TYPE_DEFAULT`（www.hiascend.com/document）。
- [社区任务流程及注意事项](https://gitcode.com/org/cann/discussions/39)、[cann-competitions 设计文档模板](https://gitcode.com/cann/cann-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md)。
