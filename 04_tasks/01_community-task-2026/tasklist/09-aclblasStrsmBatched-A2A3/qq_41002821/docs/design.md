# aclblasStrsmBatched 算子开发设计文档（A2/A3）

| 项目 | 内容 |
| --- | --- |
| 算子 | `aclblasStrsmBatched`：单精度实数批量三角求解，独立输出 |
| 任务 | 9月社区任务——aclblasStrsmBatched 算子开发（A2/A3） |
| 目标硬件 | Atlas A2（性能基准 Atlas 800T A2 / 910B3）、Atlas A3 |
| CANN | 9.1.0 |
| 实现目录 | `ops-blas/blas/trsmbatched/arch22/` |
| 测试目录 | `ops-blas/test/trsmbatched/strsmbatched/arch22/` |
| 接口基线 | cuBLAS `cublasStrsmBatched`，参考实现为 Netlib CBLAS `strsm` |
| 文档版本 | v2.0 |

> 本文只设计 Atlas A2/A3 的 arch22 实现。具体硬件参数、MatMul 模板接口和 tile 阈值在 CANN 9.1.0 目标环境中以最小编译探针和真实性能数据为准，不把未经真机验证的数值写成既定事实。

# 1. 需求背景（required）

## 1.1 需求来源

本设计依据任务书《aclblasStrsmBatched_A2A3 算子开发》、配套测试用例以及 ops-blas 工程规范编写。目标是在 Atlas A2/A3 上实现 FLOAT32 批量三角求解，并通过精度、性能、边界和异步语义验收。

相关资料：

- 任务书：`aclblasStrsmBatched_A2A3_task_doc.md`
- 算子仓：<https://gitcode.com/cann/ops-blas>
- 任务仓：<https://gitcode.com/cann/cann-ops-competitions>
- cuBLAS 语义：<https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-trsmbatched>
- 精度标准：<https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/mixed_tolerance_standard.md>

## 1.2 背景介绍

`aclblasStrsmBatched` 属于 BLAS Level 3。对于每个 batch 独立求解：

```text
side = LEFT :  op(A[i]) · X[i] = alpha · B[i]
side = RIGHT:  X[i] · op(A[i]) = alpha · B[i]
C[i] = X[i]
```

其中 `A[i]` 为三角矩阵，`B[i]` 为只读右端矩阵，`C[i]` 为独立输出矩阵。与传统 cuBLAS 原地接口相比，本任务明确增加 `Carray/ldc`，因此实现不得修改 `Barray`。

三角求解沿主维度存在递推依赖，但每个 batch、每个右端项以及块更新中的矩阵乘均具有并行性。设计采用“**小规模 AIV 通用路径 + 大规模分块 TRSM：AIV 面板求解、Cube trailing GEMM、AIV 更新**”的混合方案，避免把全部 O(n³) 计算压在向量核上。

## 1.3 现有实现分析与本设计取舍

- `arch35` 的同名 13 参数接口是原地写回版本，不能直接复制到 arch22；本任务使用 15 参数独立输出签名。
- `arch22` 的 `strsv` 可作为 AIV 三角递推参考，但其单向量模型不能直接满足批量矩阵的性能需求。
- `ssymm` 以及同族 TRSM 原型使用 `lib/matmul_intf.h` 高层 MatMul 模板。Trailing 更新采用该高层接口，并为非对齐尾块增加 padding，避免因尺寸不满足 Cube 对齐而整条路径退化为 AIV。
- 不采用逐 batch 的 Host 端多次 kernel 发射作为主方案：指针数组保持 Device 侧，单次 grid 覆盖 batch，减少 batchCount 较大时的调度开销。
- 不把手工底层 Cube API、未经验证的 event 链或跨核全局同步写成前置依赖；必要时先以标准高层模板和显式 kernel 边界保证可验证性，再通过 profiling 做定向优化。

# 2. 需求分析（required）

## 2.1 公共接口

```cpp
aclblasStatus_t aclblasStrsmBatched(
    aclblasHandle_t handle,
    aclblasSideMode_t side,
    aclblasFillMode_t uplo,
    aclblasOperation_t trans,
    aclblasDiagType_t diag,
    int m, int n,
    const float* alpha,
    const float* const* Aarray, int lda,
    const float* const* Barray, int ldb,
    float* const* Carray, int ldc,
    int batchCount);
```

参数语义：

- `LEFT` 时 `A[i]` 为 `m×m`，`RIGHT` 时 `A[i]` 为 `n×n`。
- `B[i]`、`C[i]` 均为列主序 `m×n`，前导维分别为 `ldb`、`ldc`。
- `Aarray/Barray/Carray` 为 Device 上的指针数组；指向的矩阵在异步操作完成前必须保持有效。
- `OP_N` 使用 A，`OP_T` 使用转置 A；实数接口不接受 `OP_C`，返回 `ACLBLAS_STATUS_INVALID_VALUE`。
- `UNIT` 对角按 1 处理且不得读取存储的对角元素。

## 2.2 参数校验和边界行为

采用稳定、可测试的校验顺序：

1. `handle == nullptr` 返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`。
2. 校验 `side/uplo/trans/diag`、`m/n`、`batchCount`、`alpha` 和前导维；不合法返回 `ACLBLAS_STATUS_INVALID_VALUE`。
3. `m == 0 || n == 0 || batchCount == 0` 为合法 no-op，返回 SUCCESS 且不发射 kernel。
4. `alpha == 0` 时只要求 `Carray/ldc` 可写；A、B 不被读取，C 的有效逻辑区域逐元素写零。
5. 非零 alpha 时，A/B/C 顶层数组以及每个元素地址均需有效；实现通过 Device 指针数组的异步拷贝/校验机制完成必要检查，不在接口内部同步调用方 stream。

调用方保证 A、B、C 不发生不支持的重叠。算子不进行代价高昂的运行时区间重叠检测。NON_UNIT 对角为零或接近零时不返回奇异性错误，按 BLAS 约定由调用方负责输入合法性。

## 2.3 需求拆解

| 子项 | 设计结论 | 验收重点 |
| --- | --- | --- |
| 数学语义 | 16 组 `side×uplo×trans×diag` 合法组合 | LEFT/RIGHT、转置和三角方向不混淆 |
| 独立输出 | 先执行 `C=alpha·B`，后续只读 A、读写 C | B 调用后保持不变，alpha 只应用一次 |
| 小规模计算 | AIV 单 kernel 通用递推 | 非对齐、小 m/n、padding 正确 |
| 大规模计算 | 分块面板 TRSM + Cube GEMM 更新 | 计算量主要落到 Cube，尾块不退化 |
| 批量并行 | Device 侧读取指针数组，grid 映射 batch | 避免 Host 逐 batch 发射 |
| 异步 | 使用 handle stream，不内部同步 | 调用方同步后读取 C |
| 精度 | CBLAS/Netlib `strsm` golden | 满足任务书 FLOAT32 混合容差 |

# 3. 详细设计（required）

## 3.1 算子分析

### 3.1.1 统一数学形式

LEFT 侧按列求解。令 `K=m`，每个右端列 `r` 满足：

```text
T · x_r = alpha · b_r,  T = op(A)
```

RIGHT 侧利用转置等价关系统一到 LEFT：

```text
X · T = alpha · B
等价于
Tᵀ · Xᵀ = alpha · Bᵀ
```

实现中优先使用逻辑转置视图（交换索引和前导维），只有当访存不规则且 profiling 证明收益明显时，才分配 workspace 做物理转置。这样可避免所有 RIGHT 用例都产生额外的转置 kernel 和显存流量。

### 3.1.2 求解方向

对 `op(A)` 的有效三角方向进行统一判断：

| 原始 `uplo` | `trans` | 有效方向 | 递推方向 |
| --- | --- | --- | --- |
| LOWER | N | LOWER | 正向 |
| UPPER | N | UPPER | 反向 |
| LOWER | T | UPPER | 反向 |
| UPPER | T | LOWER | 正向 |

每个面板内采用前代或回代。`UNIT` 路径在读取对角前就分支，确保对角存储值即使为 NaN/Inf 也不会影响结果。

### 3.1.3 分块 TRSM 正确性

以 LEFT、LOWER、N 为例，将 A 和 C 按主维度切成两块：

```text
[A11   0 ] [X1] = [B1]
[A21 A22] [X2]   [B2]
```

先求 `A11·X1=B1`，再执行：

```text
R2 = B2 - A21·X1
A22·X2 = R2
```

对每个 panel 重复该过程即为标准 blocked TRSM。Upper、Trans 和 Right 只改变遍历方向及索引，不改变数学等价性。Trailing 更新使用 FP32 累加，最终结果保持 FP32。

### 3.1.4 复杂度与数据类型

- 数据类型：A、B、C、alpha 均为 FLOAT32。
- 计算复杂度：`O(batchCount · m · n · k)`，其中 `k = LEFT ? m : n`。
- 额外 workspace：仅在物理转置或 GEMM 临时结果确有需要时申请，优先复用 handle workspace。
- 不支持广播、任意 stride 或超出 lda/ldb/ldc 语义的视图。

## 3.2 Host 侧设计

### 3.2.1 入口与快速路径

Host 入口只负责校验、构造 tiling、确定路径和发射 kernel，不搬运矩阵数据、不在内部同步：

```text
Validate(handle, attrs, dims, pointers)
  ├─ zero-size/batchCount==0  -> return SUCCESS
  ├─ alpha==0                 -> ZeroC kernel
  └─ alpha!=0
       ├─ small-shape          -> FusedAivTrsm kernel
       └─ large-shape          -> CopyScale + BlockedTrsm pipeline
```

`CopyScale` 只执行一次。Blocked 路径中的 panel kernel、GEMM kernel 和 update kernel 禁止再次乘 alpha，防止 `alpha²·B` 错误。

### 3.2.2 Tiling 与路径门控

Tiling 字段至少包含：逻辑 m/n/k、lda/ldb/ldc、batchCount、枚举、alpha、panel 起止、panel 大小、batch 映射信息、尾块 padding 信息和 workspace 偏移。所有地址/元素计数在 Host 侧使用 64 位计算后再写入受控字段。

路径门控采用运行时形状和硬件能力共同决定：

- 当 `k`、RHS 数量或 batch 较小，使用融合 AIV 路径，减少 kernel 发射和 workspace。
- 当 `k` 达到分块收益阈值且 trailing 区域足够大，使用 Cube 更新路径。
- `panelSize` 不写死为单一值：按 k 分档，通过性能 case 做 A/B 测试；候选值必须同时满足 UB、L1/L0、MatMul 对齐和尾块开销约束。
- 非对齐 panel 通过 padded tile 进入 MatMul，padding lane 置零并在写回时裁剪，禁止因为一个尾块使整个矩阵退回 AIV。

### 3.2.3 批量映射

将三维工作映射到一次 grid：

```text
job = blockId * tileJobs + localJob
batch = job / jobsPerBatch
panel_or_rhs = job % jobsPerBatch
A = Aarray[batch]
B = Barray[batch]
C = Carray[batch]
```

指针数组在 Device 侧读取，避免 Host 端为每个 batch 复制地址并发射 kernel。各 batch 之间没有数据依赖，grid 内可独立调度；同一 batch 的面板迭代通过 kernel 间 stream 顺序保证依赖。

## 3.3 Kernel 侧设计

### 3.3.1 Fused AIV 小规模路径

单个 AIV kernel 覆盖多个 batch/RHS 工作项。每个工作项负责一个 RHS 或一个小矩阵，按递推顺序在 UB 中缓存当前面板或 RHS tile：

```text
for i in solve_order:
    v = alpha * B(i, rhs)
    v -= dot(A(i, solved), X(solved, rhs))
    if diag == NON_UNIT:
        v /= A(i, i)
    X(i, rhs) = v
```

通过模板或 tiling key 固定常见的 `uplo/trans/diag` 组合，减少主循环中的分支；通用尾块仍使用 mask。所有写回只覆盖逻辑 m×n，不触碰 leading dimension 的 padding。

### 3.3.2 CopyScale/Zero

- `CopyScale`：`C[i] = alpha * B[i]`，alpha 为 1 时退化为拷贝；按列块搬运，双缓冲重叠 MTE 与 Vector。
- `ZeroC`：使用 Duplicate/填零后写 C；不读取 A、B，不能用“读取 NaN 后乘零”替代。

两者都在 batch 维度上并行，且只写 C。

### 3.3.3 PanelSolve AIV

每轮 panel 由 AIV 完成：

1. 将 A 的对角 panel 和 C 对应 RHS tile 搬入 UB。
2. 按前代/回代顺序求解 panel 内小三角系统。
3. 将已解 panel 写回 C 或写入后续更新可见的 GM 区域。
4. 对 `UNIT` 路径跳过对角加载；对尾行/尾列使用 mask。

panel 计算采用列块并行。由于同一列的递推存在顺序依赖，不使用未经验证的跨核全局同步拆分一条依赖链；并行度来自 RHS、batch 和 panel 外围的独立工作。

### 3.3.4 TrailingGemm AIC

对未求解区域计算：

```text
Temp = A_trailing,panel · C_panel
```

使用 `lib/matmul_intf.h` 高层 MatMul 模板，参照 `ssymm` 和同族 TRSM 原型的工程范式：

- A/B/C 使用 GM ND 描述；
- K 维按 panelSize 迭代；
- 通过 padded M/N/K 处理非 16/32/64 对齐的尾块；
- 首次 K 迭代清零，后续迭代累加；
- 临时结果先落到 workspace，再由 AIV 做减法，避免 Cube 与 C 的并发写冲突。

不在设计阶段硬编码 A2/A3 的 L1/L0 数值或单一 tile 形状。实现前用最小 MatMul 探针确认 CANN 9.1.0 模板参数和 A2/A3 支持情况；随后以五个性能基准和 TC_PF 扫描调参。

### 3.3.5 TrailingUpdate AIV

```text
C_trailing = C_trailing - Temp
```

Update kernel 采用向量 Sub 和双缓冲。当前 panel 的 GEMM 结果完成并写入 workspace 后才启动对应更新，panel 之间由同一 stream 的顺序保证依赖。完成更新后进入下一 panel。

### 3.3.6 RIGHT 侧访存优化

RIGHT 侧优先以转置视图访问，避免全量物理转置：

- 连续维度可直接按 C 的列块组织；
- 不连续维度使用小 tile 搬入 UB 后转置；
- 只有当 m/n 组合使重复跨步访存明显超过转置成本时，才启用 workspace 转置分支。

该策略减少常见 RIGHT case 的额外 GM 往返，同时保持一个统一的 LEFT blocked solver。

## 3.4 内存与同步

- A、B、C 指向的矩阵均由调用方提供并保持有效。
- UB 用于 panel、RHS tile 和双缓冲；workspace 用于 padded GEMM 临时结果及必要的 RIGHT 转置。
- workspace 通过已有 handle 机制复用，按最大路径一次申请，不在每个 panel 内申请/释放。
- kernel 使用 `aclblasSetStream` 绑定的 stream；算子内部不调用 `aclrtSynchronizeStream`。
- 调用方在读取 C 前负责同步。异步错误由运行时同步或事件查询接口报告。

# 4. 支持硬件

| 产品 | 支持 |
| --- | --- |
| Atlas A2 系列（含 Ascend 910B3、Atlas 800I/T A2） | √ |
| Atlas A3 系列（含 Atlas 800I A3） | √ |

编译和运行时通过 arch22 目标选择同一套公共接口。A2/A3 若在实际工具链中呈现不同的 MatMul 能力或片上资源，使用独立 tiling 参数表，不改变 API 和数学路径。

# 5. 算子约束限制

- 仅支持 FLOAT32。
- `trans` 仅支持 `OP_N`、`OP_T`；`OP_C` 返回 INVALID_VALUE。
- `m/n/batchCount` 不得为负；零维或零 batch 为合法 no-op。
- `lda ≥ max(1, k)`、`ldb ≥ max(1,m)`、`ldc ≥ max(1,m)`。
- `alpha==0` 时 A、B 不被读取，C 必须可写。
- `diag=UNIT` 不读取 A 的对角元素。
- 不支持广播、任意 stride、视图和不符合接口契约的重叠内存。
- 不进行奇异性检测；不承诺逐位确定性。
- 结果为异步写入，调用方需在读取 C 前同步 stream。

# 6. 可维可测分析（required）

## 6.1 精度标准

golden 由 Netlib CBLAS `strsm` 逐 batch 生成，比较 C 的全部有效 m×n 元素。按照任务书 FLOAT32 标准：

```text
|actual - golden| <= atol + rtol * |golden|
rtol = 2^-13
atol = 2^-13
matched_ratio >= 0.99
max_abs_error <= 1e-2 或 32 * ULP
```

包含以下专门用例：

- 16 组 side/uplo/trans/diag 组合；
- m/n 为 0、1、奇数、2 的幂、2 的幂±1 和非对齐值；
- lda/ldb/ldc padding；
- alpha 为 0、1、-1 和一般非零值；
- `UNIT` 对角存储 NaN/Inf，验证实现不读取对角；
- A/B 中 Inf/NaN 的按约定传播；
- alpha=0 配合无效 A/B，验证只写零 C；
- 调用后逐 batch 比较 B，验证 B 未被修改。

## 6.2 性能标准

测试设备为 Atlas 800T A2（910B3），Release 构建，先 warmup，再有效采样不少于 10 次取平均。典型门限如下：

| case | batchCount | m | n | side | uplo | trans | diag | Avg time 上限（us） |
| --- | ---: | ---: | ---: | --- | --- | --- | --- | ---: |
| 1 | 64 | 256 | 256 | LEFT | LOWER | N | NON_UNIT | 630.7 |
| 2 | 128 | 384 | 512 | LEFT | UPPER | T | NON_UNIT | 3062 |
| 3 | 32 | 1024 | 1024 | RIGHT | LOWER | N | UNIT | 5134 |
| 4 | 8 | 2048 | 2048 | LEFT | UPPER | T | NON_UNIT | 8680 |
| 5 | 2 | 4096 | 4096 | RIGHT | UPPER | N | UNIT | 14361 |

性能优化顺序：先确认精度和异步语义，再比较 AIV-only、blocked Cube、panelSize、tile 形状、batch 融合和 RIGHT 物理/逻辑转置。每次只改变一个因素，使用 msprof 的 kernel duration 而不是包含 Host 准备和 golden 比对的端到端时间作为最终依据。

## 6.3 自测矩阵

测试工程采用 CSV 驱动 GTest，覆盖：

1. 基础小 shape 和 16 组合法组合；
2. 尺寸、batch、前导维扫描；
3. 非方阵 LEFT/RIGHT；
4. 零维、零 batch、非法枚举、负维、非法 leading dimension；
5. 空指针和 alpha=0 特殊规则；
6. Inf/NaN、单位对角、一般 alpha；
7. 五个性能 case 及扩展性能扫描。

最终验收前分别在 A2 和 A3 上执行精度与性能回归；开发期可先使用 910B3 或 A3 完成单平台自验。

# 7. 预期文件变更

| 文件 | 变更 |
| --- | --- |
| `include/cann_ops_blas.h` | 增加 15 参数独立输出声明，保留原有 13 参数接口 |
| `blas/trsmbatched/arch22/strsmbatched_host.cpp` | 参数校验、路径门控、tiling、workspace 和 kernel 发射 |
| `blas/trsmbatched/arch22/strsmbatched_kernel.{h,cpp}` | Zero/CopyScale/Panel/GEMM/Update 及小规模融合路径 |
| `blas/trsmbatched/arch22/strsmbatched_tiling_data.h` | tiling 和 workspace 描述 |
| `test/trsmbatched/strsmbatched/arch22/*` | CSV、GTest、C 输出缓冲区管理和 README |
| `blas/trsmbatched/README.md` | A2/A3 产品支持说明 |
| `docs/zh/api_list.md` | 新接口文档条目 |

`blas/CMakeLists.txt` 若已通过 arch22 通配规则收集目录则不重复修改；提交前必须通过构建日志确认新目录实际被编译。

# 8. 兼容性分析

- 新增 15 参数接口，不改变已有 13 参数原地接口的 ABI、语义和 arch35 构建产物。
- 新实现只进入 arch22，不影响 950/arch35。
- 与 cuBLAS 的差异——独立输出、`OP_C` 拒绝、`batchCount==0` no-op——均来自本任务书，不是实现擅自扩展。
- 浮点累加顺序因 Cube/AIV 分块而不同，不要求逐位一致，按任务书混合容差验收。

# 9. 实施与验收门禁

1. 先检查公共头文件、handle/workspace API 和 arch22 编译收集规则。
2. 编写最小 `lib/matmul_intf.h` 探针，确认 CANN 9.1.0 下模板参数、padding 和 A2/A3 编译运行方式。
3. 先实现并验证 Zero/CopyScale 和 AIV 小规模路径。
4. 用全组合、边界、padding、alpha=0 和 UNIT 对角用例验证接口语义。
5. 接入 blocked Panel/GEMM/Update，确保 C 只被正确路径写入、B 始终只读。
6. 用 Address/哨兵数据检查 workspace 偏移、尾块 mask 和 RIGHT 路径。
7. 在 Release 模式运行完整精度 CSV，再执行五个基准和扩展性能扫描。
8. 分别在 A2、A3 完成回归后，提交自测步骤、精度/性能/内存交付件。

# 10. 参考资料

- 任务书《aclblasStrsmBatched_A2A3 算子开发》及配套 `test_cases`。
- ops-blas 中 `blas/trsmbatched/arch35/`、`blas/trsv/arch22/`、`blas/symm/arch22/ssymm_kernel.cpp`、`blas/gemv_batched/arch22/`。
- Ascend C 编程指南和 CANN 9.1.0 API 文档。
- cuBLAS `cublasStrsmBatched` 文档。
- Netlib BLAS `strsm` 参考实现。
- Ascend 生态 FLOAT32 混合容差标准。
