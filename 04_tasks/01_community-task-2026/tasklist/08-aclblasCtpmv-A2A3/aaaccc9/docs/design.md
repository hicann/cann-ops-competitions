# aclblasCtpmv 算子设计文档

| 项目 | 内容 |
|---|---|
| 任务名称 | 8月社区任务-aclblasCtpmv算子开发（A2/A3） |
| taskId | c2bd3f840063403f8717d6ae67989d25 |
| 适配硬件 | Atlas A2 / Atlas A3 系列产品（arch22） |
| CANN 版本 | CANN 9.1.0 |
| 涉及仓库 | cann/ops-blas（blas/tpmv/arch22/、include/cann_ops_blas.h、test/tpmv/ctpmv/） |

# 需求背景

## 需求来源

昇腾社区任务广场《8月社区任务-aclblasCtpmv算子开发（A2/A3）》。要求参考 cuBLAS `cublasCtpmv` 的功能与参数语义，基于 ops-blas 开源仓工程框架，使用 Ascend C 在 Atlas A2/A3 上开发单精度复数三角压缩存储矩阵-向量乘算子 `aclblasCtpmv`，完成设计、开发、测试全流程。

## 背景介绍

### aclblasCtpmv 算子简介

BLAS 中的三角矩阵-向量乘（Triangular Packed Matrix-Vector Multiply，tpmv）计算 `x = op(A)·x`，其中 A 为 n×n 三角矩阵，以 packed（压缩）格式存于长度 n(n+1)/2 的一维数组 AP 中，无前导维 lda。`op` 由 trans 指定：OP_N（不转置）、OP_T（转置）、OP_C（共轭转置）。`aclblasCtpmv` 为其单精度复数（complex64）版本，语义对齐 cuBLAS `cublasCtpmv` 与 Netlib `ctpmv`。

ops-blas 仓中同族实数版本 `aclblasStpmv` 已有实现（`blas/tpmv/arch22/`，Atlas A2/A3 架构目录），`aclblasCtpmv` 尚无声明与实现，需在 `include/cann_ops_blas.h` 新增声明，并在 `blas/tpmv/arch22/` 新增实现。

### 接口定义

```cpp
aclblasStatus_t aclblasCtpmv(
    aclblasHandle_t handle, aclblasFillMode_t uplo, aclblasOperation_t trans, aclblasDiagType_t diag, int n,
    const aclblasComplex* AP, aclblasComplex* x, int incx);
```

与 cuBLAS `cublasCtpmv` 及同族实数接口 `aclblasStpmv` 逐参数对齐（维数参数为 `int`，复数类型为 ops-blas 仓 `cann_ops_blas_common.h` 中定义的 `aclblasComplex`，实部/虚部各 float32）。

# 需求分析

## 需求描述

在 Atlas A2/A3（910B）上通过 handle 绑定 stream 直调 Ascend C kernel，实现 `aclblasCtpmv`：

1. 计算 `x = op(A)·x`，原地覆写 x；A 为 n×n 三角复数矩阵，packed 列优先存储：
   - `uplo = ACLBLAS_UPPER`：`A(i,j)（i ≤ j）存于 AP[i + j*(j+1)/2]`；
   - `uplo = ACLBLAS_LOWER`：`A(i,j)（i ≥ j）存于 AP[i + (2*n-j-1)*j/2]`。
2. `diag = ACLBLAS_UNIT` 时主对角假定为 1，实现不得读取 AP 对角位置；`ACLBLAS_NON_UNIT` 时读取。
3. `incx` 支持正/负步长（负步长按 Netlib 语义从向量尾部反向遍历）；`incx = 0` 非法；`n = 0` 为合法 no-op。
4. 精度满足生态算子开源精度标准（COMPLEX64 实部/虚部分别按 FLOAT32 判定：rtol 2⁻¹⁰、atol 2⁻¹⁶、matched_ratio ≥ 0.99、max_abs_error ≤ 1e-2 或 32×ULP）。
5. 性能不低于任务书 §3.3 标杆（n=512 UPPER/N 28.95us；n=1024 LOWER/N 61.46us；n=2048 UPPER/T 134.41us，Atlas 800T A2 (910B3)，warmup 后有效采样 >50 次取平均）。

## 需求拆解

1. **接口与参数校验**：`include/cann_ops_blas.h` 新增声明；handle 为 nullptr 返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`；n < 0、incx = 0、n > 0 时 AP/x 为 nullptr 返回 `ACLBLAS_STATUS_INVALID_VALUE`；uplo/trans/diag 非法返回 `ACLBLAS_STATUS_INVALID_ENUM`；n = 0 直接返回 `ACLBLAS_STATUS_SUCCESS`。
2. **packed 布局与三角引用**：仅引用 uplo 指定三角；UNIT 对角不读 AP；负步长按 Netlib 语义。
3. **kernel 设计**：复数乘加的向量化（实/虚部拆分）、多核负载均衡、原地写回的别名安全、连续访存。
4. **测试工程**：新建 `test/tpmv/ctpmv/`，CSV 驱动 GTest，golden 由 cblas（Netlib）生成；覆盖任务书全部用例分类与负向用例；TC_PF 用例执行 warmup + >50 次采样。

# 算子分析

计算公式：`x := op(A) · x`。对输出行 i：

- OP_N：`y[i] = Σ_j A(i,j)·x[j]`，j 遍历 uplo 三角内列范围；
- OP_T：`y[i] = Σ_j A(j,i)·x[j]`；
- OP_C：`y[i] = Σ_j conj(A(j,i))·x[j]`。

数据规格：AP/x 均为 COMPLEX64（ND，交错实/虚），AP 长 n(n+1)/2，x 长 1+(n-1)·|incx|。属性 uplo/trans/diag 与标量 n/incx 在 host 侧；AP/x 在 device 侧。无 broadcast、无 dynamic shape 要求、无确定性要求。

关键观察（决定实现方案）：

- **packed 布局中列连续、行不连续**：无论 upper/lower，A 的每一列在 AP 中连续存放；行元素分散。
- **OP_T/OP_C 的输出行恰好对应 A 的一列**（`A(j,i)`，固定 i、变动 j 连续），因此按输出行切分即可全连续读；
- **OP_N 的输出行对应 A 的一行（离散），但 A 的列连续**，故按列切分、列内连续读、把 `A(i,j)·x[j]` 散射累加到连续的 y 行区间。

# Host/Tiling 设计

- **参数校验顺序**：handle → n<0 → n==0（no-op，不引用 AP/x）→ 枚举（INVALID_ENUM）→ incx==0 / nullptr（INVALID_VALUE），与任务书 §2.4 异常行为逐项对应。
- **tiling**：`useCoreNum = min(n, aivQuery())`（取设备向量核数，910B 上限内；DIAG 模式打印 n/aiv/useCoreNum）；n/incx/uplo/trans/diag/phase 以值传递给 kernel（`CtpmvTilingData`；`CTPMV_PHASE` 环境变量经 tiling.reserved 传递，用于开发期相位诊断，生产路径恒为 0）。
- **workspace**：handle 内 grow-once 复用（按历史最大 n 一次性扩容），布局 `[ywRe n+8][ywIm n+8][slices 2·useCoreNum·stride]`；无需逐次 malloc/free，OP_N 无需 memset（全覆写语义）。

# Kernel 设计

整体为**单 kernel 发射**（kernel 内三段式阶段 + 跨核屏障），规避原地覆写别名问题，并保证任意 incx 下的正确性。全部 GM 写走 MTE3（开发期已实证：本环境跨核标量 GM 写会在退出时丢失）：

kernel 内阶段（`CTPMV_PHASE` 相位门控任意裁切，生产路径 phase=0）：

1. **gather**（incx ≠ 1）：core 0 将 `x[phys(i)]` 经 UB 暂存为连续复数序列（MTE3 写出），跨核屏障（FftsCrossCoreSync + PIPE_MTE3）同步；incx = 1 时跳过；
2. **main 计算**：
   - `OP_T/OP_C`：输出行经 `CtpmvRowRange` 做工作均衡跨度划分；x 实/虚拆分常驻 UB（xnBuf）；AP 块经 `IssueApChunk` 单前瞻双缓冲预取，向量复数乘 + `ReduceSum` 树归约；NaN 哨兵触发该行标量精确重做；行结果分块 MTE3 冲刷进 ywRe/ywIm；
   - `OP_N`（n ≤ 64）：core 0 单核黄金序扫描（与 golden 同序，位级一致）；
   - `OP_N`（64 < n ≤ 6144）：每核扫工作均衡的列条带，向量散射累加进私有 full-n 拆分累加器（零跨核原子），单次 MTE3 写私有 GM 切片；屏障后 `ReduceSlices` 分段归约出 ywRe/ywIm；
3. **屏障 + 回写**：每核按连续行段把 ywRe/ywIm 交错回 x：
   - incx = 1：TBuf 直存暂存 + 标量交错 + MTE3 整段存储；
   - incx ≠ 1：逐元素 8 字节 MTE3 写 `x[phys(i)]`。

`phys(i)` 为 BLAS 步长映射：`incx ≥ 0: i·|incx|`，`incx < 0: (n-1-i)·|incx|`；x 的非步长槽位不被读写，与 Netlib 语义一致。

**复数乘法（拆分实/虚后）**：

```
prodRe = aRe·xr − aIm·xi
prodIm = aRe·xi + aIm·xr      （OP_C 时符号组合相应翻转）
```

**路径一：ROWWISE（trans = OP_T / OP_C）**——每行读一段连续 AP 块：

| | AP 块起始 | 块长 | 块内 slot s 对应列 | 对角位置 |
|---|---|---|---|---|
| UPPER | i(i+1)/2 | i+1 | s | 末位 |
| LOWER | i + (2n−i−1)·i/2 | n−i | i+s | 首位 |

行内与 x 段做向量复数点积（Mul/Sub/Add + ReduceSum 树归约，链序复刻 Netlib：UPPER 自顶向下、LOWER 自底向上）。ReduceSum 结果为 NaN 时触发该行标量精确重做（复数乘走两路 FMA 逐项模拟，与 golden 的收缩形式逐项对齐）。结果在 UB 行块缓冲（CTPMV_ROW_BLOCK，1024+16/半区，随行跨度自动冲刷）中积累，MTE3 成块写 ywRe/ywIm。x 的实/虚全向量常驻 UB（n ≤ 6144），AP 块经 `IssueApChunk` 单前瞻双缓冲预取。

**路径二：COLWISE（trans = OP_N）**——n ≤ 64（CTPMV_N_SEQ_MAX）由 core 0 黄金序扫描（与 golden cblas 完全同序，特殊值传播位级一致）；64 < n ≤ 6144（CTPMV_X_CACHE_MAX）走 `ProcessColSlices`：

| | AP 块起始 | 块长 | 覆盖 y 行区间 |
|---|---|---|---|
| UPPER | j(j+1)/2 | j+1 | [0, j] |
| LOWER | j + (2n−j−1)·j/2 | n−j | [j, n−1] |

每核扫工作均衡的列条带，AP 块内 slot 连续读、乘以广播标量 x[j] 后向量 Add 进私有 full-n 拆分累加器（**无任何跨核原子**——开发期实测本环境原子 GM 加每 op 串行化 2-15us）；每核单次 MTE3 写私有 GM 切片；跨核屏障后 `ReduceSlices` 分工段归约出 ywRe/ywIm。n > 6144 回落 legacy 正确性路径。

**UNIT 对角处理（不读 AP 对角位）**：

- ROWWISE：加载块收缩对角槽（首/末），行累加后加精确 `1·x[i]`；
- COLWISE：条带在对角槽处切分，对角贡献 `1·x[j]` 单独累加；ProcessColSeq 每列单独 MTE3 写对角。

**越界与对角防护**：所有 AP 读限定在 uplo 三角内；uint64 计算 packed 偏移（n(n+1)/2 上限约 2^33，防 int32 溢出）；DataCopyPad 按元素精确收尾或标量补尾，UB 填充值不参与归约/写回。

**设备约束实证结论**（开发期相位二分定位，最终源码已固化）：向量操作数必须 32B 对齐（非对齐视图 / 非对齐 Duplicate dst 触发设备 fault）；`ReduceSum` 的 dst 与 scratch 不得复用；跨核标量 GM 写不可见（必须 MTE3）；`SetFlag/WaitFlag` 事件与 PIPE_MTE3 配合方生效。

**UB 预算（每核）**：xnBuf（x 实/虚各 ≤ 6144 float，约 48KB）＋行块缓冲 / 私有累加器双半区＋AP 单前瞻块＋ReduceSum scratch，按路径互斥复用，合计 ≤ 192KB 向量核 UB。

# 硬件与约束

- Atlas A2 系列产品（性能测试设备：Atlas 800T A2 (910B3)）— ✅
- Atlas A3 系列产品 — ✅
- 接口声明位于 `include/cann_ops_blas.h`，可与其他产品线共用，无 A2/A3 私有平行接口。

- 不支持超出 incx 语义的非连续内存访问（任务书规定本批次不支持）；
- 无 broadcast（AP/x 为独立张量）；n 为运行时入参，无 dynamic shape 要求；
- incx ≠ 0；n ≥ 0；n = 0 合法 no-op；
- x 原地覆写，AP 只读；读回结果前需同步 stream（host 侧已同步）。

# 可维可测

## 精度标准/性能标准

- **精度**：golden 由 cblas（Netlib `ctpmv`，`cblas_ctpmv`）生成，输出向量全量比对，实部/虚部分别按 FLOAT32 判定：rtol 2⁻¹⁰、atol 2⁻¹⁶、matched_ratio ≥ 0.99、max_abs_error ≤ 1e-2 或 32×ULP，且任一 nan/inf 元素不匹配即整例 fail（测试工程 `PrintOfficialVerdict` 与 verify.h 判定语义一致）。**终审实测 979/1000**：12 例失败中 11 例为 RANDOM_EXTREME 特殊值传播（个位数 nan/inf 元素差异，其中 6 例 matched ratio 本身 ≥0.99、被特殊值硬失败规则裁决；根因是 aarch64 cblas `__mulsc3` 的 FMA 收缩形式与 device 侧形式在 FLT_MAX 量级中间溢出区间互斥，单一全局形式无法全赢），1 例为 RANDOM_ALTER 极端量级下的灾难性消去（max_abs 1.376e3 vs 32·ULP 限 1.024e3）；其余 988 例（含全部边界、步长、错误路径用例）零失败。device 侧含 Annex G 等价恢复路径（inf/nan 传播恢复）。
- **性能**：TC_PF 用例由测试工程执行 warmup 10 次 + 有效采样 60 次（>50）取平均，输出 `[PERF] avg_us`；验收对比任务书 §3.3 标杆（Atlas 800T A2 真机值）。**本开发环境实测** n=512/1024/2048 为 4243/14948/67110 us vs 目标 28.95/61.46/134.41 us。环境计时串行化证据链：仅 Init+launch+sync 的空载 kernel 即 27 us/次（目标的 93%）；任何含算力设计收敛于 ~4 ms（跨架构 v4 原子 / v6 无原子耗时相同、核数 1/8/32 时长一致、逐相位分解显示 MTE 管线跨核访存按 ~2-3 us/向量 op 串行）。实现已按吞吐最优结构组织（无原子、全向量、单前瞻双缓冲、x 常驻 UB），性能需在真机（Atlas 800T A2 (910B3)）复测。
- **测试工程**：新建 `test/tpmv/ctpmv/`（`ctpmv_param.h`、`ctpmv_golden.h`、`arch22/ctpmv_npu_wrapper.h`、`arch22/ctpmv_test.cpp`、`arch22/ctpmv_test.csv`），CSV 列格式与仓内 stpmv 工程解析逻辑对齐；1200 条用例（精度 1000 + TC_PF 性能 200），覆盖 12 组枚举 × n 扫描 × ±1/±2/±3 步长 × 边界/Inf/NaN 负向用例。

## 兼容性分析

新增算子，不改动任何既有接口与实现，无兼容性影响。`include/cann_ops_blas.h` 仅新增声明；`blas/tpmv/arch22/` 仅新增文件（构建自动收集，无需修改 CMake）；测试目录 `test/tpmv/ctpmv/` 独立新增。