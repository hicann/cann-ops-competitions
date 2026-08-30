# aclblasCgetriBatched 算子设计文档

本文档对应《8 月社区任务——`aclblasCgetriBatched` 算子开发任务书》。目标产品为 Ascend 950PR，软件环境为 CANN 9.1.0。

> GitCode 贡献者：`yhh-max`
>
> 设计文档提交路径：`04_tasks/01_community-task-2026/tasklist/08-aclblasCgetriBatched/yhh-max/docs/design.md`

# 需求背景（required）

## 需求来源

任务要求在 `ops-blas` 仓新增 Complex64 批量矩阵求逆接口，功能和参数语义对齐
`cublasCgetriBatched`，计算语义对齐 LAPACK `cgetri`。接口使用 Ascend C Kernel 直调模式，通过
`aclblasHandle_t` 绑定的 stream 异步执行。

## 背景介绍

`Aarray[i]` 保存上游 `aclblasCgetrfBatched` 产生的列主序 LU 因子，满足 `P*A=L*U`。本算子计算：

```text
Carray[i] = inv(A[i]) = inv(U[i]) * inv(L[i]) * P[i]
```

L 的对角线隐含为 1，U 包含对角线。`PivotArray` 每个 batch 含 n 个 1-based 主元索引；为
`nullptr` 时按无主元分解处理，即 P=I。求逆为 out-of-place，`Aarray[i]` 与 `Carray[i]` 重叠时行为未定义。

仓库已有同族实数接口 `aclblasSgetriBatched`，但没有 C 版公开声明、Complex64 Kernel 和对应测试。本设计只增加统一 C 版 API，不改变 S 版语义。

# 需求分析（required）

## 需求描述

公开声明新增至 `include/cann_ops_blas.h`，禁止定义 950PR 私有平行接口。函数原型严格为：

```cpp
aclblasStatus_t aclblasCgetriBatched(
    aclblasHandle_t handle, int n,
    const aclblasComplex* const Aarray[], int lda,
    const int* PivotArray,
    aclblasComplex* const Carray[], int ldc,
    int* infoArray, int batchSize);
```

`aclblasComplex` 采用仓内定义的交错 Complex64 布局，实部和虚部均为 FP32。

## 需求拆解

| 维度 | 设计要求 |
| --- | --- |
| 功能 | 对每个 batch 根据 LU 和可选 pivot 计算独立逆矩阵 |
| 数据 | Complex64 输入/输出，INT32 pivot/info，列主序 |
| 形状 | n、batchSize 为运行时标量；逻辑矩阵固定为 n×n 方阵 |
| 工程 | arch35 Ascend C Kernel 直调，共用 ops-blas 公开 API 和 handle stream |
| 正确性 | 覆盖 pivot/no-pivot、奇异、padding、no-op、负向和 Inf/NaN |
| 精度 | 实部、虚部分别按任务书 FLOAT32 混合容差判定 |
| 性能 | 三个固定规模均不超过任务书 Avg time 上限，采样次数大于 50 |
| 交付 | 设计 PR、算子/测试代码、测试 README、自测报告和代码地址 |

## 参数与返回值

| 参数 | 方向/位置 | 类型与布局 | 形状或值域 | 约束及异常行为 |
| --- | --- | --- | --- | --- |
| `handle` | 输入，Host | `aclblasHandle_t` | 有效句柄 | `nullptr` 返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| `n` | 输入，Host | `int` | `n>=0` | `n<0` 返回 `ACLBLAS_STATUS_INVALID_VALUE`；`n=0` 为合法 no-op |
| `Aarray` | 输入，Device | Complex64 指针数组，矩阵列主序 | 数组长度 batchSize；每矩阵物理空间 `lda*n`、逻辑形状 `n*n` | 非 no-op 时为空返回 `ACLBLAS_STATUS_INVALID_VALUE`；内容须为有效 LU 因子 |
| `lda` | 输入，Host | `int` | `lda>=max(1,n)` | 不满足约束返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| `PivotArray` | 输入，Device | `const int*`，按 batch 线性排列 | `n*batchSize`，每项在 `[1,n]` | 可为 `nullptr`，表示 P=I；不在 Kernel 内校验每个主元值 |
| `Carray` | 输出，Device | Complex64 指针数组，矩阵列主序 | 数组长度 batchSize；每矩阵物理空间 `ldc*n`、逻辑形状 `n*n` | 非 no-op 时为空返回 `ACLBLAS_STATUS_INVALID_VALUE`；不得与对应 A 重叠 |
| `ldc` | 输入，Host | `int` | `ldc>=max(1,n)` | 不满足约束返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| `infoArray` | 输出，Device | `int*` | batchSize | 非 no-op 时为空返回 `ACLBLAS_STATUS_INVALID_VALUE`；0 表示成功，k>0 表示首个 `U(k,k)==0` |
| `batchSize` | 输入，Host | `int` | `batchSize>=0` | 小于 0 返回 `ACLBLAS_STATUS_INVALID_VALUE`；等于 0 为合法 no-op |

返回值遵循 `cann_ops_blas_common.h`：合法计算返回 `ACLBLAS_STATUS_SUCCESS`；handle 为空返回
`ACLBLAS_STATUS_HANDLE_IS_NULLPTR`；参数非法返回 `ACLBLAS_STATUS_INVALID_VALUE`；平台核心数异常返回
`ACLBLAS_STATUS_INTERNAL_ERROR`；内部 workspace 不足、超限或申请失败返回
`ACLBLAS_STATUS_ALLOC_FAILED`，workspace 管理过程中发生的其他错误按公共管理器状态透传。`infoArray[i]>0`
是合法计算结果，接口仍返回 SUCCESS。

## 边界与兼容性判定

校验顺序固定为：handle → n/batchSize → lda/ldc → no-op → 数据指针。因而 `n==0` 或
`batchSize==0` 仍要求标量约束合法，但允许 `Aarray/Carray/infoArray/PivotArray` 全为空，且不申请 workspace、不启动 Kernel。

任务书 §3.5 的通用测试描述中出现了本接口无法表达的条目，按接口定义处理如下：

| 条目 | 本算子判定 |
| --- | --- |
| `alpha=(0,0)` | 接口无 alpha 参数，不适用 |
| 非方阵、宽/窄矩阵 | 接口只有方阵阶数 n，不适用 |
| stride/零步长 | 本接口是指针数组模式，无 stride 参数，不适用 |
| Inf/NaN | 适用；除精确复零对角判奇异外，按 FP32 规则传播 |

该判定不扩展或改变任务书 §2.3 规定的公开签名。

# 详细设计（required）

## 算子分析

### 数学推导

由 `P*A=L*U` 可得 `A^-1=U^-1*L^-1*P`。实现不显式形成 L、U 的逆，而是依次求解：

```text
L * X = P
U * C = X
```

前代中 L 的单位对角线不参与除法；回代中使用 U 的显式复数对角线。复数 `a+bi` 除以
`c+di` 时，根据 `|c|` 与 `|d|` 选择 Smith scaled division 分支，避免直接计算 `c*c+d*d`
造成可避免的上溢或下溢。中间计算保持 FP32，不降精度到 FP16。

### 支持范围

| 项目 | 支持范围 |
| --- | --- |
| 输入/输出 dtype | COMPLEX64；实部、虚部均为 FP32 |
| 矩阵 | n×n 方阵，列主序，支持 `lda>n` 和 `ldc>n` |
| batch | Device 指针数组，不同 batch 的矩阵地址可以不连续 |
| pivot | 每 batch n 个 1-based INT32；允许整体为空表示无主元 |
| dynamic shape | 不要求；n 和 batchSize 作为运行时 Host 入参处理 |
| broadcast/stride | 不支持，也无对应接口参数 |
| 确定性 | 任务书不要求确定性模式 |

### 奇异与主元语义

每个 batch 按 k 从 0 到 n-1 扫描 U 对角线，首个实部、虚部均精确等于 0 的位置写为
`infoArray[i]=k+1`；非奇异写 0。奇异 batch 的 C 内容不定义，也不参与数值精度比较。

P 通过按 getrf 顺序执行行交换构造。`PivotArray==nullptr` 时跳过交换。Host 不回读或校验 Device
主元内容，非法主元属于调用者违反前置条件。

## 算子实现

### 工程结构

```text
include/cann_ops_blas.h
blas/getri_batched/
├── README.md
├── docs/
└── arch35/
    ├── cgetri_batched_host.cpp
    ├── cgetri_batched_kernel.cpp
    ├── cgetri_batched_blocked_kernel.cpp
    ├── cgetri_batched_kernel.h
    └── cgetri_batched_tiling_data.h
test/getri_batched/cgetri_batched/
├── CMakeLists.txt
├── README.md
├── cgetri_batched_param.h
├── cgetri_batched_golden.h
└── arch35/
    ├── cgetri_batched_npu_wrapper.h
    ├── cgetri_batched_test.cpp
    └── cgetri_batched_test.csv
```

### Host 侧设计

1. 完成参数校验和 no-op 判断。
2. `n<64` 进入单 Kernel 小矩阵路径；`n>=64` 进入分块 AIV+Cube 路径。
3. 小矩阵路径按 AIV 核数均分 batch。分块路径查询 AIV/AIC 核数，为 prepare、panel、update、finalize 和共享 FP32 batched GEMM 生成 tiling。
4. 所有 Kernel 与 GEMM 均按调用顺序提交至 `handle->stream`；Host 不为读取 A、C、pivot 或 info
   主动同步、不回读 Device 数据。若库管理的默认 workspace 容量不足，公共 workspace 管理器可在扩容前同步
   stream；容量充足时不产生该同步。计算结果仍为异步，调用者读取 C 或 info 前必须同步 stream。

### 分路与 Tiling 策略

| 范围 | 并行模型 | Tiling 策略 |
| --- | --- | --- |
| `1<=n<=32` | AIV SIMT | 单 block 704 个 VF 线程，即 22 个 32-thread group；每组处理一个 batch，每个 lane 常驻一列的至多 32 个复数。704 由 950PR profiling 在 batch 并发与寄存器占用之间选取 |
| `33<=n<64` | AIV SIMT | 每个 block 顺序处理分配到的 batch，列在线程间并行；输出作为 GM 工作区完成前代和回代 |
| `n>=64` | AIV+AIC Cube | `n<=128` 面板宽 16，否则宽 32；GEMM 采用 128×128 tile、K chunk 最大 256，并以 16×16 基本单元在 AIC 间切分 |

小矩阵 tiling 传递 n、lda、ldc、使用核数、每核 batch 数、尾核 batch 数和 `usePivot`。分块 tiling
额外包含 panel 范围、更新行范围、临时矩阵行跨、指针数组对齐字节数及 workspace 偏移。

### Kernel 侧设计

#### 小矩阵路径

对 `n<=32`，32-thread group 先并行检查矩阵结构和主元，再由每个 lane 负责一个输出列：构造 P 的对应列，执行 `L*X=P` 前代和 `U*C=X` 回代，最后列主序写入 C。列值常驻寄存器，避免求解过程反复访问 C。

对 `33<=n<64`，先并行初始化 P，再以列并行方式在 GM 输出上进行两次三角求解。每个 batch 只启动一次 Kernel，避免按列或按消元步产生 Host launch 开销。

#### 分块路径

1. prepare Kernel 将交错 Complex64 LU 拆分为 `Ar/Ai`，按 pivot 构造 `Cr/Ci=P`，扫描 U 对角线并写 info，同时建立后续 GEMM 所需的 Device 指针数组。
2. panel Kernel 依序完成当前对角块内的前代或回代。
3. 面板外更新转化为复数矩阵乘。复数乘法拆为四次 FP32 Cube GEMM：

   ```text
   T1 = Ar*Cr, T2 = Ai*Ci, T3 = Ar*Ci, T4 = Ai*Cr
   C  = C - (T1-T2) - i*(T3+T4)
   ```

4. 下三角阶段按 panel 从前向后处理；上三角阶段按 panel 从后向前处理。finalize Kernel 将 `Cr/Ci` 重新交错写入各 `Carray[i]`，只写逻辑 n 行，保留 ldc padding。
5. 对确认为对角 LU 的 batch，可直接对 P 的非零元素执行对角除法并设置 skip flag，使其跳过后续 panel/GEMM；该优化保持一般 pivot 语义。三个规定性能用例使用对角占优矩阵，性能验收不依赖此特殊结构快捷路径。

### Workspace 设计

`n<64` 不申请 workspace。`n>=64` 通过 handle 默认 workspace 管理接口申请，布局为：

```text
[12 组 64B 对齐指针数组][batch 个 diagonal flag][64B 对齐填充]
[Ar][Ai][Cr][Ci][T1][T2][T3][T4]
```

令 `M=n*n`、`R=align_up(n,16)*n`、`B=batchSize`，数据区上界为
`4*(M+R)*B*sizeof(float)`。Ar/Ai/Cr/Ci 各占 `M*B` 个 FP32；T1–T4 各占 `R*B`
个 FP32并由所有 panel 复用。所有乘加和对齐计算先提升至 64 位，并在转换为 `size_t` 前检查溢出；申请失败返回分配错误。
任务书 §3.4 未规定额外内存上限，但自测报告仍记录三个必验 case 的 workspace 及测试总峰值内存。

### 性能优化方案

- 小矩阵将 batch 并行和列并行组合到单次 Kernel，寄存器保存列向量，减少 launch、同步和 GM 读写。
- 大矩阵用 AIV 完成串行依赖较强的面板求解，用 Cube 承担面板外 O(n³) 更新；按 AIC 数切分 m/n tile。
- workspace、指针数组和 Device buffer 在单次 API 内复用；性能采样时复用测试侧已分配输入输出，不计 H2D/D2H、内存申请和 golden。
- 对角结构检测仅作为合法输入的附加优化；一般对角占优矩阵必须走通用求解路径并独立达到任务书门槛。
- 以 msprof 分解 Host launch、AIV panel、AIC GEMM 和 finalize 耗时；任何一项未达门槛时以三个固定性能 case 为调优依据，不以其他输入替代。

### 关键风险与防护

| 风险 | 设计防护 | 验收方法 |
| --- | --- | --- |
| pivot 方向或顺序错误 | 严格按 getrf 的 1-based 序列正向施加行交换，直接构造 P | 与 LAPACK 语义 golden 全矩阵比较，同时覆盖 PIVOT/NO_PIVOT |
| 复数除法溢出/下溢 | FP32 Smith scaled division，不形成平方和 | 大小差异显著的实虚部、Inf/NaN 和良态随机用例 |
| 奇异误报或错误返回码 | 只将实部和虚部同时精确为零判奇异，info 与 API status 分离 | singular/mixed 用例逐 batch 精确比较 info |
| workspace 整数溢出 | 偏移和乘积采用 64 位，申请前验证 `size_t` 可表示 | 大 n/大 batch 边界检查和内存记录 |
| 特殊结构优化掩盖通用性能 | 三个性能 case 固定使用对角占优输入，禁止替换为 identity/diagonal | 保存输入参数、51 次平均日志和 msprof 截图 |
| 共享 GEMM 改动影响其他算子 | skip flag 默认 0，保持原 ABI 行为 | `gemm_batched` 与 Sgetri 构建、功能回归 |

## 支持硬件

| 支持的芯片版本 | 支持情况 |
| --- | --- |
| Ascend 950PR（arch35） | 支持 |

## 算子约束限制

- 输入必须是相同 n、lda、batchSize 的 getrf 有效 LU 结果；主元值必须位于 `[1,n]`。
- `Carray[i]` 不得与 `Aarray[i]` 重叠；Host 不同步回读指针数组做 alias 检查。
- 不支持非方阵、broadcast、stride-batched 或超出 lda/ldc 表达能力的任意非连续布局。
- Aarray/Carray 是 Device 指针数组，且每个数组元素必须指向合法 Device 内存；Host 只校验数组首地址是否为空。
- 奇异 batch 只保证 info 精确，不保证 C 数值；奇异不是 API 错误。
- 算子异步执行，调用者读取 C 或 info 前必须同步 handle 所绑定的 stream。
- 任务书未规定 n 和 batchSize 的额外上限；实际可执行规模受 64 位 workspace 计算及设备可用内存约束。

# 可维可测分析

## 精度标准

Host golden 按 LAPACK `cgetrf/cgetri` 语义自实现：先从原始良态矩阵得到 LU 与 pivot，再根据 LU 独立计算逆矩阵和 info；可使用 LAPACK 结果进行交叉校验，但测试不依赖 cblas 求逆接口。

| 分量 | rtol | atol | required matched ratio | max abs error |
| --- | ---: | ---: | ---: | ---: |
| Complex64 实部 | `2^-10` | `2^-16` | `>=0.99` | `<=1e-2` 或框架等价的 `32*ULP` |
| Complex64 虚部 | `2^-10` | `2^-16` | `>=0.99` | `<=1e-2` 或框架等价的 `32*ULP` |

逐元素条件为 `abs(actual-golden) <= atol + rtol*abs(golden)`。每个非奇异 batch 的实部、虚部分别同时满足 matched ratio 与 max abs error 才通过；info 必须逐 batch 完全一致。奇异 batch 只比较 info。

## 性能标准

在 Ascend 950PR、CANN 9.1.0、目标设备卡空闲状态下，使用任务书 CSV 指定的 PIVOT、紧凑前导维、
`DIAGONALLY_DOMINANT` Complex64 输入。预热 5 次后用 ACL Event 对同一 stream 连续采样 51 次，计算平均单次耗时：

| case | n | batchSize | lda/ldc | Avg time 上限 |
| --- | ---: | ---: | ---: | ---: |
| `TC_PF_1001` | 32 | 1024 | 32/32 | 26.26 us |
| `TC_PF_1002` | 512 | 32 | 512/512 | 13579.32 us |
| `TC_PF_1003` | 1024 | 16 | 1024/1024 | 118585.00 us |

计时窗口只包含 API 提交及其 NPU 执行，不包含输入生成、golden、显存分配、H2D 和 D2H。报告同时保留原始日志和 msprof 截图，不能以 identity/diagonal 输入替代规定的对角占优输入。

## 测试覆盖与复现

随任务提供的 CSV 共 1200 条，其中功能/精度 1000 条、性能 200 条，原文件不删减。测试使用 GoogleTest 参数化框架，覆盖：

- n 为 0、1、小质数、2 的幂及 ±1、非对齐值和大规模；batch 为 0、1、小质数、2 的幂和大 batch。
- PIVOT/NO_PIVOT、紧凑与 lda/ldc padding、对角/三角、混合与奇异矩阵。一般良态矩阵的实部和虚部分别独立采样：50% 使用 `[-5,5]` 均匀分布，50% 使用 `mu∈[-5,5]`、`sigma∈[0.1,2]` 的正态分布，并做对角加强/对角占优以控制条件数。
- handle、n、batchSize、lda、ldc、Aarray、Carray、infoArray 的负向用例；quick return 数据指针为空；info>0 仍 SUCCESS。
- Inf/NaN 专项测试；ldc padding 使用哨兵值验证逻辑区域外不被写入。
- 三个任务书必验性能点及其余 TC_PF 规模采集。

测试 README 必须给出构建、过滤运行、设备选择和结果判定命令，并按功能/精度、性能分类列出
case。自测报告记录环境、用例参数、实部/虚部误差、info、三项性能和内存占用；实部、虚部精度
结果分别提供截图，性能数据保留 51 次采样的原始日志及截图。

## 兼容性与回归

新增 API 不改变已有公开接口。共享 FP32 batched GEMM tiling 增加的 skip flag 对其他调用者必须默认置 0；回归至少包含现有 `aclblasSgetriBatched`、`gemm_batched` 的构建和测试。新增文件遵循仓库 `.clang-format`、版权头和 arch35 目录规则。

# 需求追踪与送审检查

| 任务书条款 | 设计落点 | 必须提交的验收证据 |
| --- | --- | --- |
| §2.1.1–3 LU、调用链、out-of-place | 背景、数学推导、约束 | pivot/no-pivot、padding 精度日志 |
| §2.1.4–5 info、nullable pivot | 参数表、奇异与主元语义 | singular、mixed、no-pivot 日志 |
| §2.1.6–9 dtype、no-op、统一 API | 需求描述、参数表、边界判定 | 头文件编译、no-op/负向日志 |
| §2.2–2.5 工程模式及约束 | 工程结构、Host/Kernel、workspace | arch35 构建日志、CSV 参数测试 |
| §3.2 精度 | 精度标准、测试覆盖 | 实/虚误差、matched ratio、info |
| §3.3 性能 | 性能优化方案、性能标准 | 三个 DD 固定 case 的 51 次平均与截图 |
| §3.4 内存 | workspace 设计 | 64 位溢出检查、峰值内存记录 |
| §3.5 自验 | 测试覆盖与复现 | 完整 1200 条 CSV、GTest、README |
| §4 交付件 | 工程结构、追踪表 | 设计 PR、代码地址、README、自测报告 |

送审前检查项：

- [x] 官方模板必填章节及性能优化方案完整。
- [x] 接口原型、参数、内存位置、列主序、错误码和校验顺序与任务书一致。
- [x] LU/pivot、out-of-place、singular、no-op、padding 和异步语义均有明确设计。
- [x] 算法、tiling、workspace、精度阈值和三个性能门槛均已定义对应的验收方法及证据类型。
- [x] 不可表达的通用模板条目逐项说明，未修改公开 API。
- [ ] 在 `cann-ops-competitions` 提交设计 PR，并经评审合入。

前五项表示文档内容具备送审条件，不表示代码精度、性能或回归已经全部通过；最后一项只有在社区
PR 评审通过并合入后才能勾选，不能以本地自检代替官方验收。
