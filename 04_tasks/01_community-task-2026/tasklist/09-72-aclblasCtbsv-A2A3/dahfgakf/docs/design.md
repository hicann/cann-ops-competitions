# 需求背景（required）

## 需求来源

CANN 社区任务 09-72：aclblasCtbsv 算子开发（A2/A3）。依据官方任务书《aclblasCtbsv_A2A3_task_doc.md》，本文结构对应官方模板 `04_tasks/01_community-task-2026/resources/design_template.md`。提交目录为 `04_tasks/01_community-task-2026/tasklist/09-72-aclblasCtbsv-A2A3/dahfgakf/docs/design.md`。本文记录实现方案与开发者自验证结果，不代表社区或华为已完成验收。

- 代码仓库：[dahfgakf/ops-blas](https://gitcode.com/dahfgakf/ops-blas)
- 代码分支：`feat/ctbsv-arch22`
- 生产代码：`blas/tbsv/arch22/`；公共接口声明：`include/cann_ops_blas.h`
- 测试代码：`test/tbsv/ctbsv/arch22/`（含测试用例 CSV）

## 背景介绍

### aclblasCtbsv 算子实现优化

在 ops-blas 仓中新增 Ascend C kernel 直调接口 `aclblasCtbsv`，原地求解单精度复数（complex64）三角带状线性方程组，覆盖 Atlas A2/A3（对应架构目录 arch22）。

### aclblasCtbsv 算子实现现状分析

本任务为新增 BLAS 直调算子，非替换既有实现。数值 golden 参考为 Netlib BLAS `ctbsv`（经 cblas 调用），性能基线为任务提供的 GPU 时延。

### aclblasCtbsv 算子功能分析

求解 op(A)x = b，其中 A 为 n 阶三角带状矩阵（complex64，列主序带存储），x 保存右端项 b，返回后保存解向量。

| 参数 | 参数含义 | 数据类型 | 约束 |
| --- | --- | --- | --- |
| handle | BLAS 句柄，绑定 stream | aclblasHandle_t | 非空，否则返回 HANDLE_IS_NULLPTR |
| uplo | 上/下三角 | aclblasFillMode_t | UPPER 或 LOWER |
| trans | N / T / C | aclblasOperation_t | OP_N、OP_T、OP_C |
| diag | 是否单位对角 | aclblasDiagType_t | UNIT 或 NON_UNIT |
| n | 矩阵阶数 | int | n ≥ 0；n=0 为合法 no-op |
| k | 上/下带宽 | int | k ≥ 0 |
| A | 带存储矩阵 | const aclblasComplex* | (k+1)×n 列主序，分配 lda×n，不修改 |
| lda | 带存储列距 | int | lda ≥ k+1 |
| x | 右端/解向量 | aclblasComplex* | 原地更新 |
| incx | x 步长 | int | 非零、非 INT_MIN，支持负步长 |

不支持广播。UNIT 不读取对角元素；k=0 且 UNIT 退化为恒等，直接返回成功。

# 需求分析（required）

## 需求描述

实现与任务规定 API、列主序带存储布局、异常返回码和精度要求一致的 `aclblasCtbsv`；精度按 FLOAT32 单精度标准逐分量比对；性能满足 NPU 平均单次耗时不超过 GPU 基线/0.8；完成实机自验证并输出自测报告。

## 需求拆解

1. 参数校验覆盖枚举、尺寸、步长与空指针，返回任务规定错误码。
2. 覆盖 uplo×trans×diag 全部 12 种组合、正/负步长与 n=0 等边界输入。
3. 利用 UB 缓存与向量指令降低带矩阵访问成本；超出 UB 预算的输入走标量回退，保证任意合法形状可算。
4. 完成精度 1000 条、性能 200 条自测并输出报告与原始日志。

# 详细设计（required）

## 算子分析

### 数学公式

求解 op(A)x = b，op(A) ∈ {A, A^T, A^H}。带存储布局（lda ≥ k+1）：

- 上三角：A_band[k+i-j + j·lda] = a_ij，max(0, j-k) ≤ i ≤ j
- 下三角：A_band[i-j + j·lda] = a_ij，j ≤ i ≤ min(n-1, j+k)

仅访问对应三角带内元素。UNIT 时对角按 1 处理，不读取存储的对角元素。

### 支持数据类型

`aclblasComplex`：实部、虚部均为 float32（complex64），不做精度转换。

### 支持形状

运行时由 host 生成 tiling，不依赖固定编译形状。n、k 非负，lda ≥ k+1，incx 非零且支持负步长（按 BLAS 约定从逻辑首元素开始访问）。超出 UB 预算的大带宽输入由标量回退路径承接。

## 算子实现

### 实现方案

#### 3.2.1 host侧设计：

##### 1. 分核策略：

三角求解为逐元素递推，存在严格顺序依赖，使用单个 AIV 核；核内对带宽窗口内的乘加/归约做向量并行，不跨核拆解依赖链。

##### 2. 数据分块和内存优化策略：

- x 全量搬入 UB：arch22 无 DeInterleave 指令，实部/虚部用分块 stride-2 Gather 解交织为连续向量（避免逐元素标量搬运），求解完成后整体搬出，避免递推期间反复访问 GM。
- A 按列 ping-pong 分块搬入 UB，每列仅取带内 k+1 个元素，列首预留 16 floats 对齐余量。
- host 侧按 UB 预算守卫值（176KB，计入 x 缓冲、A 列 ping-pong 缓冲与窗口 scratch 共 7 组带长向量）预估占用；超预算输入置 `useScalarFallback`，kernel 走 GM 直访华标量回退路径。n<8 的极小形状同样走标量回退（其固定开销低于向量 kernel 的 TPipe 初始化与 DMA 往返，实测 2.3-2.6us）。

##### 3. tilingkey规划策略：

直调 tiling 结构 `CtbsvTilingData` 传递 n/k/lda/incx/uplo/trans/diag、指针地址与回退标志；kernel 以模板参数分发 12 种 (uplo, trans, diag) 组合，无图模式 tilingkey 注册。

数据检测：handle 为 nullptr 返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`；uplo/trans/diag 非法枚举、n<0、k<0、lda≤k、incx=0 或 incx=INT_MIN 返回 `ACLBLAS_STATUS_INVALID_VALUE`；n=0 或（k=0 且 UNIT）为合法 no-op；n>0 时 A/x 空指针返回 `ACLBLAS_STATUS_INVALID_VALUE`。

#### 3.2.2 kernel侧设计：

Init / Process 两阶段，Process 含 x 搬入（DataCopy + DataCopyPad 尾块对齐）、逐列求解、x 搬出。

1. NoTrans（列更新）路径：解出当前列 t = x[j]（NON_UNIT 时除以对角）后，对带宽窗口内向量扣减 x[i] -= a_ij·t。向量化 ScatterWindowVec：Gather 基址整体回移 pad 对齐到 32B，头部 pad 元素先复制原值再清零，随后用对齐 Sub 完成窗口更新，避免非对齐访问与窗口外误写。
2. Trans/Conj（点积）路径：每行计算 A 列带内段与已解出 x 窗口的复点积并经 ReduceSum 归约。x 在 UB 中已解交织为连续向量，直接以 32B 对齐窗口参与 Mul；仅 A 侧做 2 次 stride-2 Gather（从交织列缓冲取实部/虚部，基址随 x 对齐回移 pad，列缓冲 16 floats 头区在 Init 时预清零覆盖 pad 空槽）。LOWER+UNIT 下 pad 会读到存储对角（带内第 0 行），用一条 V 管道 Duplicate 清零（UNIT 不读对角，且避免 NaN/Inf 毒化对角污染归约）；LOWER+NON_UNIT 保留 pad 通道显式清零。Conj 的共轭通过复乘 Add/Sub 组合实现，无额外向量遍历。
3. 复数除法采用 Smith 算法（`ctbsv_complex_arith.h`），按分子分母量级选择直接式或缩放式，抑制中间溢出。
4. 负 incx 按 BLAS 约定映射物理偏移；UNIT 对角跳过除法且不读对角。
5. 标量回退 kernel（`ctbsv_kernel_scalar_gm`）直接按 GM 地址顺序递推，功能与向量路径等价，仅承接 UB 预算外形状。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 | 实测环境 |
| --- | --- | --- |
| Atlas A2 系列（含 Atlas 800I A2） | √ | Ascend910B4，CANN 9.1.0 |
| Atlas A3 系列（含 Atlas 800I A3） | √（与 A2 共用 arch22 实现） | 开发环境暂无 A3 实机，未实测 |

## 算子约束限制

- A 与 x 不允许重叠；调用后需同步 handle 的 stream 再读取结果。
- 奇异矩阵行为与 Netlib 参考一致，不做额外检查。
- 超出 UB 预算的形状走标量回退，功能正确但性能非最优。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述（不涉及说明原因） | 标准来源 |
| --- | --- | --- |
| 精度 | 实部/虚部逐分量比对，rtol=atol=2⁻¹³，合格比例 ≥ 0.99；全量 1201 条（1000 精度 + 200 性能 + 空 handle 负向）全部通过。TC_EX_0990（n=1024, k=511, UPPER/N/UNIT）与 TC_PF_1107（n=2676, k=160, UPPER/T/UNIT）为固有数值混沌用例（独立的 GM 标量顺序实现对 Netlib golden 同样超阈值，非实现缺陷），按测试用例 README 允许条款更换随机种子（TC_EX_0990: 20261004，TC_PF_1107: 20262107）并在报告中注明 | 任务书 §3.2 与生态算子开源精度标准 |
| 性能 | NPU 平均单次耗时 ≤ GPU 基线/0.8；msprof 采集 kernel Task Duration，每例 warmup 后重复 >10 次取均值；自测 200 条全部达标（明细见自测报告） | 任务书 §3.3 |
| 内存 | 任务书标注“不涉及”，未做峰值测量 | 任务书 §3.4 |

## 兼容性分析

新增公共 API `aclblasCtbsv`，不修改既有算子接口与行为，无兼容性影响。