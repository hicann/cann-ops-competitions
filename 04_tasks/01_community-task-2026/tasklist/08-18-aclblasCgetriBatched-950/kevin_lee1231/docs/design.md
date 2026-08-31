# aclblasCgetriBatched 算子设计文档

更新日期：2026-08-28

目标平台：Ascend 950PR，arch35，CANN 9.1.0

本文描述 `aclblasCgetriBatched` 的接口语义、实现方案和测试方案，不记录开发分支、测试结果或验收状态。

# 需求背景（required）

## 需求来源

任务要求在 `ops-blas` 中新增单精度复数批量矩阵求逆接口 `aclblasCgetriBatched`，输入为一批已经完成 LU 分解的列主序 COMPLEX64 方阵，输出各矩阵的逆矩阵，并满足以下要求：

- 支持 `PIVOT` 和 `NO_PIVOT` 两种模式；
- 对奇异矩阵逐 batch 返回 LAPACK 风格 `infoArray`；
- 支持 `lda/ldc >= n`，输出 padding 不被修改；
- 接口异步执行，继承 handle 绑定的 stream；
- 在 Ascend 950PR 上通过 API、精度和性能门禁；
- 补齐公共声明、算子 README、测试代码、自测报告和待验收代码地址。

## 验收边界

本任务实现的是 `getri`，输入必须是与本接口 `n/lda/batchSize` 一致的 LU 分解结果，不负责在接口内部执行 `getrf`。单算子测试先用 CPU LAPACK `cgetrf_` 生成 LU 和 1-based pivot，再调用 NPU `aclblasCgetriBatched`，最后用 LAPACK `cgetri_` 生成逆矩阵 golden。

求逆采用 out-of-place 语义，`Carray[i]` 不得与 `Aarray[i]` 重叠。`PivotArray == nullptr` 表示输入由无主元 LU 分解产生；奇异矩阵通过 `infoArray[i] > 0` 报告，接口仍返回 `ACLBLAS_STATUS_SUCCESS`。

任务附件包含 1000 条 API/精度用例和 200 条 PF 用例。`TC_PF_1001`～`TC_PF_1003` 按任务书绝对阈值判定，其余 PF 用例执行并记录参考数据，不作为正式性能门禁。附件未覆盖的 Inf/NaN 特殊值场景在仓内测试中补充。

# 需求分析（required）

## 数学定义

对第 `b` 个矩阵，LU 分解语义为：

```text
P_b A_b = L_b U_b
```

因此：

```text
A_b^(-1) = U_b^(-1) L_b^(-1) P_b
```

实现不显式构造 `L^-1` 和 `U^-1`，而是从经主元置换的单位阵出发，依次求解：

```text
L_b Y_b = P_b
U_b C_b = Y_b
```

最终 `C_b=A_b^-1`。当 `PivotArray == nullptr` 时，`P_b=I`。

## 对外接口

```cpp
aclblasStatus_t aclblasCgetriBatched(
    aclblasHandle_t handle,
    int n,
    const aclblasComplex* const Aarray[],
    int lda,
    const int* PivotArray,
    aclblasComplex* const Carray[],
    int ldc,
    int* infoArray,
    int batchSize);
```

参数语义：

| 参数 | 位置 | 说明 |
| --- | --- | --- |
| `handle` | Host | ops-blas 句柄，提供 stream 和默认 workspace |
| `n` | Host | 方阵阶数，`n >= 0` |
| `Aarray` | Device | Device 指针数组；每项指向列主序 COMPLEX64 LU 因子 |
| `lda` | Host | 输入 leading dimension，`lda >= max(1,n)` |
| `PivotArray` | Device/可空 | 每个 batch 连续存放 `n` 个 1-based pivot；空指针表示无主元模式 |
| `Carray` | Device | Device 指针数组；每项指向独立的列主序输出矩阵 |
| `ldc` | Host | 输出 leading dimension，`ldc >= max(1,n)` |
| `infoArray` | Device | `0` 表示成功；`k>0` 表示第 `k` 个 U 对角元为零 |
| `batchSize` | Host | batch 数，`batchSize >= 0` |

## 参数检查和返回规则

1. `handle == nullptr` 返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`；
2. `n < 0`、`batchSize < 0`、`lda < max(1,n)` 或 `ldc < max(1,n)` 返回 `ACLBLAS_STATUS_INVALID_VALUE`；
3. `n == 0` 或 `batchSize == 0` 直接返回成功，不检查 Device 指针、不分配 workspace、不启动 Kernel；
4. 非 quick return 时，`Aarray/Carray/infoArray` 为空返回 `ACLBLAS_STATUS_INVALID_VALUE`；
5. `PivotArray == nullptr` 合法，表示无主元 LU；
6. blocked 路径的步长或字节数溢出、默认 workspace 分配失败时返回分配错误；
7. U 对角元为零属于逐矩阵数值状态：Host API 仍返回成功，由 `infoArray[b]` 报告首个 1-based 零对角位置；
8. 所有 Kernel 都进入 handle 的同一 stream，接口内部不做同步。

# 详细设计（required）

## 总体架构

实现按 `n` 路由为两条路径：

| 路径 | 条件 | 计算单元 | 目的 |
| --- | --- | --- | --- |
| SMALL/GENERAL SIMT | `n < 128` | AIV/SIMT | 避免工作区和多阶段 launch，覆盖小矩阵高 batch |
| BLOCKED 4M | `n >= 128` | AIV + AIC Cube | 把主要 `O(n^3)` 更新交给 FP32 Cube，满足中大矩阵性能 |

公共 API 只做参数检查、路径选择和异步下发，不读取 Device 矩阵内容。

## 工程结构

```text
include/cann_ops_blas.h
blas/getri_batched/
├── README.md
└── arch35/
    ├── cgetri_batched_host.cpp
    ├── cgetri_batched_kernel.cpp
    ├── cgetri_batched_kernel.h
    └── cgetri_batched_tiling_data.h
blas/gemm_batched/arch35/gemm_batched_kernel.cpp
test/getri_batched/cgetri_batched/
├── CMakeLists.txt
├── README.md
├── cgetri_batched_golden.h
├── cgetri_batched_param.h
└── arch35/
    ├── cgetri_batched_npu_wrapper.h
    ├── cgetri_batched_test.cpp
    └── cgetri_batched_test.csv
```

`gemm_batched_kernel.cpp` 新增的是供 cgetri blocked 路径内部调用的 FP32 strided GEMM launcher；原有 `aclblasSgemmBatched` 公共入口和语义未改。

## 小矩阵与一般 SIMT 路径

### `n <= 32`

- 每个 32-lane group 负责一个 batch 矩阵；
- LU 的实部、虚部复制到 UB SoA 平面，减少重复 GM 读取；
- 每个 lane 负责一个 RHS 列，列向量保存在寄存器数组中；
- `n <= 8` 的基础过程使用更小的固定寄存器数组；
- 若 LU 为对角阵且 pivot 为单位置换，直接计算对角倒数；
- `n=32 && lda==ldc==32` 使用 512 线程 speculative 路径，提升 `32×1024` 正式性能 case 的 batch 并行度；
- 一般情形依次应用 pivot、L 前代和 U 回代。

### `33 <= n < 128`

每个 AIV block 处理分配到的若干 batch，使用 GM 上的单位阵、主元置换、L 前代和 U 回代完成求逆。这一路径不申请 blocked workspace。

## BLOCKED 4M 路径

blocked 阈值和 tile 均固定为 128：

```cpp
CGETRI_BLOCKED_THRESHOLD_N = 128;
CGETRI_BLOCKED_TILE_N = 128;
```

### 阶段 1：初始化

`cgetri_blocked_init_kernel` 完成：

- 输入 AoS COMPLEX64 拆分为 compact SoA `aReal/aImag`；
- 扫描 U 对角，记录首个零对角位置；
- 将 `cReal/cImag` 初始化为应用 pivot 后的单位阵；
- 对“对角 LU + 单位 pivot”设置内部 fast-path 标记并直接写倒数。

### 阶段 2：分块 L 前代

按 `rowStart=0,128,...` 从上到下处理：

1. AIV 对每个 batch、每个 RHS 列求解当前 128 行的单位下三角块；
2. 若下方仍有行，以当前解块更新尾部：

```text
C_tail -= L_tail,block × C_block
```

### 阶段 3：分块 U 回代

按最后一个 block 向前处理：

1. AIV 对每个 batch、每个 RHS 列求解当前上三角块，复数除法使用缩放算法；
2. 若上方仍有行，更新前部：

```text
C_head -= U_head,block × C_block
```

### 阶段 4：4M 复数更新

每次复数 GEMM 更新分解为四次实数 FP32 Cube GEMM：

```text
real(A×B) = Ar×Br - Ai×Bi
imag(A×B) = Ar×Bi + Ai×Br
```

四个乘积使用内部 `cgetri_strided_gemm_kernel_do`，随后由 AIV combine Kernel 完成符号合并和 `C -= A×B`。GEMM 按 batch 和二维 M/N block 分配 AIC 任务，保持每个物理 AIC 在一次 launch 中执行一个逻辑任务。

### 阶段 5：输出

`cgetri_blocked_finalize_kernel` 把 compact `cReal/cImag` 合并为 `Carray[b]` 的 AoS COMPLEX64，仅写 `n×n` 有效区域，不修改 `ldc` padding；内部 diagonal fast-path 标记在此恢复为公开的 `info=0`。

## Workspace 设计

blocked 路径通过 handle 的 `EnsureDefaultWorkspace` 一次取得六段 256 字节对齐的 FP32 工作区：

```text
aReal | aImag | cReal | cImag | temp1 | temp2
```

- `aReal/aImag/cReal/cImag` 每段大小为 `batchSize*n*n*sizeof(float)`；
- `temp1/temp2` 每段大小为 `batchSize*align(n,C0)*n*sizeof(float)`；
- 所有乘加先经过 checked arithmetic；步长超过 `uint32_t` 或字节数超过 `size_t` 时失败；
- workspace 由 handle 管理，没有新增公共 workspace 参数；warmup 可完成首次扩容，正式计时不包含初始化和数据准备。

正式性能规模的理论内部 workspace（未计每段最多 255 字节对齐间隙）为：

| case | 路径 | 理论内部 workspace |
| --- | --- | ---: |
| `n=32,batch=1024` | SMALL | 0 |
| `n=512,batch=32` | BLOCKED | 192 MiB |
| `n=1024,batch=16` | BLOCKED | 384 MiB |

该数据是代码公式推导值，不等于进程总显存峰值；进程总占用还包含输入、输出、pivot、info、runtime 和测试框架分配，必须另以现场观测记录。

## 数值稳定性

复数除法采用 Smith 风格缩放形式，根据 `|real|` 与 `|imag|` 选择分支，避免直接构造 `real²+imag²` 导致不必要的上溢或下溢。

非奇异精度用例使用 CPU LAPACK 逆矩阵作为 golden，并同时执行以下门禁：

- 实部、虚部分别按 `rtol=2^-10`、`atol=2^-16` 比较；
- matched ratio 不低于 0.99；
- 最大绝对误差不超过 `max(1e-2,32 ULP)`；
- 原矩阵和 LAPACK 逆矩阵估算的无穷范数条件数不超过 `1e6`；
- 奇异用例只比较 `infoArray`，不比较未定义逆矩阵值；
- `ldc > n` 时检查 padding 哨兵未被改写。

随机通用矩阵以及上下三角矩阵都使用 `max(10,10*n)` 强化实对角线。该生成规则用于满足任务书“条件受控”的输入边界，不改变算子精度阈值。

## 异步与生命周期

- Host 不读取 `Aarray/PivotArray/Carray/infoArray` 的 Device 内容；
- 所有 AIV/AIC Kernel 按算法依赖顺序进入 handle stream；
- API 返回只表示参数与 launch 流程成功；读取输出前由调用者同步 stream；
- 默认 workspace 归 handle 所有，调用者不得在同一 handle 的未完成算子之间破坏其生命周期。

# 可维可测分析

## 测试资产与分层

参数化 GTest 从 `arch35/cgetri_batched_test.csv` 加载任务附件全部用例，并补充附件未覆盖的 Inf/NaN 特殊值用例：

| 集合 | 规模 | 测试方案 |
| --- | ---: | --- |
| API 与精度 | 1000 条任务附件用例 | 覆盖小 shape、shape/batch 扫描、leading dimension padding、PIVOT/NO_PIVOT、奇异矩阵、quick return、空指针和非法参数 |
| 特殊值 | 2 条补充用例 | 构造含 Inf/NaN 的合法 LU 因子，检查接口状态、`infoArray` 和非有限值传播 |
| PF | 200 条任务附件用例 | 全部执行 warmup 与有效采样；3 条正式门禁按绝对阈值判定，其余仅记录参考数据 |

PIVOT 用例调用 LAPACK `cgetrf_` 生成 LU 和 1-based pivot；NO_PIVOT 用例使用测试工程内的复数无主元 LU。非奇异 golden 调用 LAPACK `cgetri_`，奇异用例只比较 `infoArray`。

## 功能与异常测试方案

| 场景 | 预期行为 |
| --- | --- |
| 正常 PIVOT/NO_PIVOT | 返回 SUCCESS，逆矩阵与 LAPACK golden 一致 |
| `n == 0` 或 `batchSize == 0` | 返回 SUCCESS，不访问 Device 指针、不启动 Kernel |
| `n < 0`、`batchSize < 0`、非法 `lda/ldc` | 返回 INVALID_VALUE |
| 空 handle | 返回 HANDLE_IS_NULLPTR |
| 非 quick return 时空 `Aarray/Carray/infoArray` | 返回 INVALID_VALUE |
| 奇异 U | 接口返回 SUCCESS，`infoArray` 与 golden 逐 batch 相等 |
| `ldc > n` | 只写逻辑矩阵，padding 哨兵保持不变 |

## 精度测试方案

可逆且条件数受控的输入按实部、虚部分别执行 FLOAT32 门禁：`rtol=2^-10`、`atol=2^-16`、matched ratio 不低于 0.99，最大绝对误差不超过 `1e-2` 或 `32 ULP`。均匀分布和正态分布分别生成实部与虚部，通用随机矩阵通过对角加强控制条件数。

Inf/NaN 用例不参与有限数 matched ratio 统计。该类输入不作为参数错误；测试检查调用成功、stream 可正常同步、非零 U 对角对应 `infoArray == 0`，并检查非有限值传播到输出。

## 性能测试方案

每个 PF 用例先 warmup 5 次，再有效采样 51 次；数据准备、Host/Device 拷贝和 CPU golden 不计入设备执行区间。3 个正式性能门禁如下：

| case | n | batchSize | lda | ldc | 平均耗时上限 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `TC_PF_1001` | 32 | 1024 | 32 | 32 | 164.14 us |
| `TC_PF_1002` | 512 | 32 | 512 | 512 | 84870.75 us |
| `TC_PF_1003` | 1024 | 16 | 1024 | 1024 | 741156.25 us |

其余 PF 用例使用相同计时口径输出平均耗时，但没有任务书绝对阈值，不据此宣称正式性能通过。测试报告必须同时记录硬件、CANN 版本、分支、完整提交、命令、有效样本数和原始输出。

## 内存与回归测试方案

任务书未设置额外内存上限；自测报告仍记录最大正式性能用例的进程峰值和按 workspace 公式计算的内部需求，二者分开表述。构建后执行 Cgetri 全量用例、同族 `aclblasSgetriBatched` 回归，以及受内部 FP32 GEMM launcher 影响的 GEMM Batched 定向回归。

# 兼容性与约束

- 新接口只在 Ascend 950PR/arch35 标记支持；
- 不修改 `aclblasSgetriBatched` 的接口和实现；
- 不修改现有 `aclblasSgemmBatched` 对外语义；
- 输入与输出必须是互不重叠的 Device 缓冲区；
- `PivotArray` 使用 LAPACK 风格 1-based pivot；
- 最大可处理规模受步长整数范围、默认 workspace 和设备内存共同限制；
- Inf/NaN 不作为参数错误，按 FP32 运算传播；特殊值测试不套用有限数 matched ratio；
- 当前没有公共复数 getrf，纯 NPU 端到端链路是外部依赖。

# 风险与剩余项

| 风险 | 影响 | 设计处理 |
| --- | --- | --- |
| blocked 路径 workspace 随 `batchSize*n^2` 增长 | 大规模用例可能分配失败 | checked arithmetic、256 字节对齐、统一返回分配错误，并在自测报告记录峰值 |
| 内部复用 FP32 GEMM launcher | 可能影响 GEMM Batched 同族代码 | 保持 launcher 为内部符号，并加入 GEMM Batched 定向回归 |
| 缺少公共 `aclblasCgetrfBatched` | 不能提供纯 NPU LU→求逆链路 | 在交付材料中明确单算子边界 |
| Inf/NaN 结果不适用有限数误差判定 | matched ratio 无意义 | 单独检查接口状态、info 和非有限值传播 |
| 补充 PF 用例没有任务书绝对阈值 | 不能据此给出正式性能结论 | 执行并记录参考数据，只对 3 个正式 case 判定门禁 |

# 交付件

1. 算子设计文档；
2. 覆盖任务附件全部用例及补充特殊值用例的测试代码与复现 README；
3. 包含参数、实部/虚部精度、性能和内存数据的自测报告；
4. 个人代码仓、开发分支、算子目录和公共接口信息。

# 参考资料

- 任务书：`aclblasCgetriBatched_Atlas950PR_task_doc.md`
- 仓库：`https://gitcode.com/cann/ops-blas`
- 个人 fork：`https://gitcode.com/kevin_lee1231/ops-blas.git`
- LAPACK：CGETRF/CGETRI 语义
- Ascend C/CANN 9.1.0 目标环境
