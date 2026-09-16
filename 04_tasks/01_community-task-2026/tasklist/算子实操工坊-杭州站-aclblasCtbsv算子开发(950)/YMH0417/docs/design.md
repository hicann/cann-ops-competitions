# aclblasCtbsv 算子设计文档

> 任务：算子实操工坊-杭州站-aclblasCtbsv 算子开发（950）  
> 目标仓：https://gitcode.com/cann/ops-blas  
> 实现目录：`blas/stbsv/arch35/`  
> 测试目录：`test/tbsv/ctbsv/`  
> 适配硬件：Ascend 950PR（arch35）  
> CANN 版本：9.1.0  
> 代码：`4b3e5f4`（`feat/aclblas-ctbsv-950`）  
> 合入 PR：https://gitcode.com/cann/ops-blas/pull/445  
> 设计文档 PR：https://gitcode.com/cann/cann-ops-competitions/pull/1543（已合入；本文档按实现提交同步更新）  
> 模板：https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md

| 项 | 内容 |
|----|------|
| 算子 | aclblasCtbsv |
| 硬件 / CANN | Ascend 950PR / 9.1.0 |
| 文档日期 | 2026-09-15 |
| 个人仓 | https://gitcode.com/YMH0417/ops-blas |

---

## 一、需求背景

### 1.1 需求来源

本任务在 ops-blas 中新增单精度复数三角带状线性方程组求解接口 `aclblasCtbsv`，补齐同族实数接口 `aclblasStbsv` 的 COMPLEX64 覆盖。语义对齐 cuBLAS `cublasCtbsv` 与 Netlib `ctbsv`。

该接口不属于神经网络图算子，仓库中无对应 TBE 实现，亦不走 ACLNN。按任务书要求，以 ops-blas 句柄式 BLAS 合入：Host 完成参数校验后直调 Ascend C Kernel。

### 1.2 背景介绍

#### 1.2.1 aclblasCtbsv 算子实现优化

本算子为 BLAS Level-2 求解类接口。昇腾生态中不存在对应 TBE 实现，因此不以 `/usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_impl/ai_core/tbe/impl` 或算子信息库 `tbe.dsl` 作为标杆。参考来源如下。

| 角色 | 路径 / 链接 | 说明 |
|------|-------------|------|
| 语义标杆 | [cuBLAS `cublasCtbsv`](https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-tbsv) | 接口签名、带状存储、uplo/trans/diag；不做奇异性检查 |
| 算法标杆 | [Netlib `ctbsv.f`](https://www.netlib.org/blas/ctbsv.f) | 前向 / 回代顺序、负步长、`N.EQ.0` 直接返回 |
| 仓内同族 | `ops-blas/blas/stbsv/arch35/stbsv_host.cpp` | Host 校验、quick return、SIMT 门槛 |
| 仓内同族 | `ops-blas/blas/stbsv/arch35/stbsv_kernel.cpp` | 标量回代 Kernel |
| 仓内同族 | `ops-blas/blas/stbsv/arch35/stbsv_kernel_simt.cpp` | SIMT 并行点积与归约 |
| 公共类型 | `ops-blas/include/cann_ops_blas_common.h` | `aclblasComplex`、枚举与状态码 |
| 接口声明 | `ops-blas/include/cann_ops_blas.h` | `aclblasCtbsv` 公共头声明，不另设 950PR 私有平行接口 |

相对 `aclblasStbsv` 的主要差异：元素类型由 `float` 扩展为 `aclblasComplex`；`trans = ACLBLAS_OP_C` 须走共轭转置（实数路径下 C 与 T 等价）；Kernel 对 A 的虚部取反，对角除法改为复数除法。

#### 1.2.2 aclblasCtbsv 算子现状分析

##### 1.2.2.1 标杆算子支持的数据类型和数据格式

与 cuBLAS `cublasCtbsv`、Netlib `ctbsv` 及任务书 §2.4 一致：

| 参数 | 含义 | 数据类型 | 支持 dtype | 内存 | 排布 / 形状 | 约束 |
|------|------|----------|------------|------|-------------|------|
| handle | 库上下文，绑定 stream | `aclblasHandle_t` | — | Host | scalar | 空指针返回 `HANDLE_IS_NULLPTR`（优先校验） |
| uplo | 上 / 下三角带 | enum | `{UPPER=121, LOWER=122}` | Host | attr | 非法取值返回 `INVALID_ENUM` |
| trans | `op(A)` | enum | `{N=111, T=112, C=113}` | Host | attr | 非法取值返回 `INVALID_ENUM` |
| diag | 单位对角 / 非单位对角 | enum | `{NON_UNIT=131, UNIT=132}` | Host | attr | 非法取值返回 `INVALID_ENUM` |
| n | 矩阵阶 | int | n ≥ 0 | Host | scalar | n < 0 非法；n = 0 为空操作 |
| k | 超对角 / 次对角数 | int | k ≥ 0 | Host | scalar | k < 0 非法 |
| A | 三角带状矩阵，只读 | COMPLEX64 | 实部 / 虚部 FLOAT32 | Device | 列主序带状，数组 `lda × n`，有效 `(k+1) × n` | n > 0 时非空；lda ≥ k+1 |
| lda | A 的 leading dimension | int | lda ≥ k+1 | Host | scalar | lda < k+1 非法 |
| x | 入口为右端 b，出口为解 | COMPLEX64 | 实部 / 虚部 FLOAT32 | Device | 步长 incx，逻辑长 n | n > 0 时非空；原地覆写 |
| incx | x 元素步长 | int | incx ≠ 0 且 ≠ INT_MIN | Host | scalar | 负步长按 Netlib 反向遍历 |

不支持广播，不要求 dynamic shape，不支持超出 lda / incx 语义的非连续 Tensor。

##### 1.2.2.2 标杆算子实现描述

Netlib `ctbsv` 与 cuBLAS `cublasCtbsv` 求解：

```text
op(A) * x = b
```

入口时 `x` 存放右端 `b`，解原地写回 `x`。`op(A)` 由 trans 决定：

- `N`：使用 A
- `T`：使用 Aᵀ，不取共轭
- `C`：使用 Aᴴ，元素取共轭

实现要点与源码逻辑一致：

1. 参数检查通过后 quick return：`N.EQ.0` 直接返回。单位对角且带宽为 0 时，A 退化为单位阵，x 保持不变。
2. 带状列主序寻址（文档为 1-based，实现转为 0-based）：
   - LOWER：主对角位于数组第 1 行，`A(i,j)` 存于 `A(1+i-j, j)`，右下 k×k 区域不引用
   - UPPER：主对角位于数组第 k+1 行，`A(i,j)` 存于 `A(1+k+i-j, j)`，左上 k×k 区域不引用
3. 回代方向：
   - 前向（j = 0 → n-1）：`(LOWER && N)` 或 `(UPPER && (T 或 C))`
   - 后向（j = n-1 → 0）：其余组合
4. 逐行更新：对已求解的带内邻元做复乘累减，再按 diag 决定是否除以对角元。`UNIT` 时不读取对角，视为 1；`NON_UNIT` 时做复数除法。`C` 路径对非对角元虚部取反，对角除以 `conj(diag)`。
5. 负步长：`incx < 0` 时，第 `idx` 个逻辑元的物理偏移为 `(n-1-idx)*(-incx)`，对应 Netlib 起始点 `1-(n-1)*incx`。
6. 不做奇异性检查。对角为零时行为未定义，由调用方保证 `NON_UNIT` 对角远离零。

##### 1.2.2.3 标杆算子实现流程图

```mermaid
flowchart TD
    A[进入 cublasCtbsv / ctbsv] --> H0{handle 非空?}
    H0 -->|否| E0[返回 HANDLE_IS_NULLPTR]
    H0 -->|是| B1{uplo/trans/diag 合法?}
    B1 -->|否| E1[返回 INVALID_ENUM]
    B1 -->|是| B2{n/k/lda/incx 合法?}
    B2 -->|否| E3[返回 INVALID_VALUE]
    B2 -->|是| C{n == 0 ?}
    C -->|是| E2[返回 SUCCESS]
    C -->|否| D{k == 0 且 UNIT ?}
    D -->|是| E2
    D -->|否| F[按 incx 确定 x 起始偏移]
    F --> G{前向回代?}
    G -->|是| H[row = 0 .. n-1]
    G -->|否| I[row = n-1 .. 0]
    H --> J[带内已求解邻元复乘累减]
    I --> J
    J --> K{diag == UNIT?}
    K -->|是| L[不读对角]
    K -->|否| M[x_row = x_row / op对角]
    L --> N[写回 x_row]
    M --> N
    N --> O{还有下一行?}
    O -->|是| G
    O -->|否| E2
```

---

## 二、需求分析

### 2.1 外部组件依赖

| 依赖 | 是否适配 | 说明 |
|------|----------|------|
| CANN Runtime / ACL | 是 | `aclrtStream`、Device 内存、异步下发 |
| Ascend C | 是 | 标量 Kernel（`TPipe` / `GlobalTensor`）与 SIMT（`asc_simt` / `asc_vf_call`） |
| ops-blas handle | 是 | `aclblasCreate` / `aclblasSetStream` |
| cblas / OpenBLAS | 仅测试 | golden 使用 `cblas_ctbsv`，运行时不依赖 |
| TBE / tbe.dsl | 否 | 非 NN 图算子，无 TBE 标杆 |
| ACLNN | 否 | 任务书要求 Kernel 直调 |
| PyTorch / torch_npu | 否 | 不走框架适配 |

### 2.2 内部适配模块

| 模块 | 路径 | 作用 |
|------|------|------|
| 公共头 | `include/cann_ops_blas.h` | 新增 `aclblasCtbsv` 声明 |
| 公共类型 | `include/cann_ops_blas_common.h` | 复数类型、枚举、状态码 |
| Host | `blas/stbsv/arch35/ctbsv_host.cpp` | `ValidateCtbsvParams`、`ChooseCtbsvNumThreads`、`FillCtbsvTiling` 及入口编排 |
| 标量 Kernel | `blas/stbsv/arch35/ctbsv_kernel.cpp` | 单核串行回代 |
| SIMT Kernel | `blas/stbsv/arch35/ctbsv_kernel_simt.cpp` | 行内并行点积，warp 归约使用 `asc_reduce_add` |
| Tiling 结构 | `blas/stbsv/arch35/ctbsv_tiling_data.h` | Host 至 Device 参数块 |
| 常量 | `blas/common/helper/kernel_constant.h` | `SIMT_MIN_THREAD_NUM=128`，`SIMT_MAX_THREAD_NUM=2048` |
| ST 框架 | `test/tbsv/ctbsv/` | CSV、GTest 与 `cblas_ctbsv` golden |

实现与 `aclblasStbsv` 同目录族，由 `blas/CMakeLists.txt` 自动收集，不单独增加算子 CMake。

Host 入口将校验、线程选择与 tiling 填写拆分为上述三个辅助函数，以满足函数长度与圈复杂度门禁；SIMT 转置热路径同样拆分，避免单函数过长。

### 2.3 需求模块设计

#### 2.3.1 Ascend C 算子原型

除任务书明确不要求的能力外，接口与标杆逐参数对齐，仅将元素类型由 `float` 扩展为 `aclblasComplex`：

```cpp
aclblasStatus_t aclblasCtbsv(
    aclblasHandle_t handle,
    aclblasFillMode_t uplo,
    aclblasOperation_t trans,
    aclblasDiagType_t diag,
    int n, int k,
    const aclblasComplex* A, int lda,
    aclblasComplex* x, int incx);
```

计算公式：

```text
op(A) * x = b
op(A) = A      当 trans = ACLBLAS_OP_N
op(A) = A^T    当 trans = ACLBLAS_OP_T
op(A) = A^H    当 trans = ACLBLAS_OP_C
```

#### 2.3.2 Ascend C 算子相关约束（相对标杆缺失 / 收窄的功能）

| 标杆能力 | 本实现 | 说明 |
|----------|--------|------|
| cuBLAS 另提供 `cublasZtbsv`（COMPLEX128） | 不支持 | 任务书仅要求 COMPLEX64 |
| 多右端 / 批量 | 不支持 | 接口为单向量 TBSV |
| 多核按行并行 | 不支持 | 回代行间存在串行依赖 |
| 超出 lda/incx 的非连续 Tensor | 不支持 | 任务书明确不要求 |
| 奇异性检测 | 不支持 | 与 cuBLAS 一致 |
| ACLNN / PyTorch 包装 | 不提供 | 任务书要求 Kernel 直调 |

---

## 三、需求详细设计

### 3.1 调用方式

Kernel 直调（ops-blas 句柄式 BLAS）：

```text
应用
  → aclInit / aclrtSetDevice / aclrtCreateStream
  → aclblasCreate + aclblasSetStream
  → aclblasCtbsv(...)          // Host 校验、组 tiling、下发
  → aclrtSynchronizeStream     // 由调用方同步，Host 侧不做流同步
  → 读回 x
```

不接入 ACLNN，不注册 PyTorch schema。

### 3.2 需求总体设计

整体为 Host 校验与路径选择，加单 AiCore Kernel。TBSV 行间存在依赖，不分多核；仅在行内使用 SIMT 并行点积。

```mermaid
flowchart LR
    APP[调用方] --> API[aclblasCtbsv]
    API --> VAL[参数校验 / quick return]
    VAL --> TIL[填写 CtbsvTilingData]
    TIL --> DISP{numThreads?}
    DISP -->|0| SC[标量 Kernel 12 路特化]
    DISP -->|>0| SM[SIMT Kernel 12 路特化]
    SC --> STREAM[handle->stream]
    SM --> STREAM
```

#### 3.2.1 Host 侧设计

校验顺序遵循仓内 interface-spec 及 PR #445 评审要求：

1. `handle == nullptr` → `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`（优先校验）
2. `uplo` / `trans` / `diag` 取值非法 → `ACLBLAS_STATUS_INVALID_ENUM`
3. `n < 0`、`k < 0`、`lda ≤ k`、`incx == 0` 或 `incx == INT_MIN` → `ACLBLAS_STATUS_INVALID_VALUE`  
   带状存储仅要求 `lda > k`，不与 `n` 比较；负步长合法，不属于空操作。
4. 上述校验通过后，`n == 0` → `ACLBLAS_STATUS_SUCCESS`（空操作，不访问 A/x）
5. `n > 0` 且 `A` 或 `x` 为空 → `ACLBLAS_STATUS_INVALID_VALUE`
6. `k == 0 && diag == UNIT` → `ACLBLAS_STATUS_SUCCESS`（x 保持不变）
7. 按 n、带宽、trans 选择标量或 SIMT，填写 `CtbsvTilingData`，调用 `ctbsv_kernel_do`

`CtbsvTilingData` 字段：`a, x, n, k, uplo, trans, diag, incx, lda, numThreads`。

##### 3.2.1.1 分核策略

第 `i` 行依赖已求解邻元，不能按行划分到多个 AiCore。策略固定为：

- `blockDim = 1`（一个 Vector 核）
- 标量路径：核内单线程串行扫行
- SIMT 路径：核内 `numThreads` 个 SIMT 线程并行计算当前行带内点积，行与行之间 `syncthreads` 后继续

不采用满核均分或大小核划分。

##### 3.2.1.2 数据分块和内存优化策略

950PR 每核 UB 为 253952 B。本算子不将 A 按 tile 划分到多核，仅在核内用 UB 缓存 x 与归约缓冲。

**有效带宽**

```text
maxBand = min(k, n - 1) + 1
```

**标量路径 LocalMemory**

```text
xUbRe  = n * sizeof(float)
xUbIm  = n * sizeof(float)
UB_scalar ≈ 8n 字节
```

`incx == 1` 时先写入 UB，整行结束后一次性 Flush 至 GM，以减少逐元 GM 写。`incx != 1` 时每行直接写 GM，避免按逻辑下标 Flush 错位。

A 不整块搬入 UB。带状列主序访问跨度为 `lda`，带宽变化较大，标量路径按元素 `GetValue`。

**SIMT 路径 LocalMemory**

```text
partialReSums = SIMT_MAX_THREAD_NUM * 4 = 8192 B
partialImSums = SIMT_MAX_THREAD_NUM * 4 = 8192 B
```

当 `n ≤ CTBSV_UB_X_COMPLEX(4096)` 时额外缓存 x：

```text
xUbRe = 4096 * 4 = 16384 B
xUbIm = 4096 * 4 = 16384 B
UB_simt_ub ≈ 16 KiB + 32 KiB = 48 KiB  < 248 KiB
```

当 `n > 4096` 时走 Tiled 路径：x 不进入 UB，点积直接读 GM，UB 仅保留 16 KiB 归约缓冲。

**SIMT 线程数**

```text
minBand        = trans∈{T,C} ? 48 : 24
threadsForBand = CeilAlign(max(maxBand, 128), 128)
numThreads     = min(n, 2048, threadsForBand)
trans 路径再封顶 256
```

转置路径线程数过多时，UNIT T/C 同步开销显著，故将线程数上限设为 256。仅当 `n ≥ 128` 且 `maxBand ≥ minBand` 时走 SIMT。线程按 `j = jStart + threadIdx.x; j += blockDim.x` 划分当前行带内列；行内以 `asc_reduce_add` 做 warp 归约，再跨 warp 合并。

任务书用例 TC_PF_1004（n=4096, k=32, C, UNIT）带宽未达到转置门槛，保持标量路径。早期门槛过低时该用例会误入 SIMT 并超出标杆，因此将 T/C 门槛调整为 48。

##### 3.2.1.3 tilingKey 规划策略

不使用传统 `tilingKey` 整型分支表，采用两级开关：

| 条件 | 取值 | Kernel 分支 |
|------|------|-------------|
| `n < 128` 或带宽低于门槛 | `numThreads = 0` | 标量 12 路 |
| `n ≥ 128` 且 `maxBand ≥ 24`（N）或 `≥ 48`（T/C） | `numThreads > 0` | SIMT 12 路 |
| `uplo × trans × diag` | 编译期模板 | 12 个特化（2×3×2） |

`trans=C` 单独特化，避免热循环中判断共轭。Host 按枚举分发，不将 12 路合并为单一大函数。

相对 `aclblasStbsv`：实数路径主要以 `n ≥ 128` 切换 SIMT。复数点积计算量更大，非转置要求 `maxBand ≥ 24`，转置/共轭要求 `maxBand ≥ 48`，以避免窄带或转置同步开销超过收益。

#### 3.2.2 Kernel 侧设计

##### 3.2.2.1 Kernel 侧实现描述

**共同语义**

- 寻址与标杆一致。UPPER 非转置：`idx = k + row - col + col*lda`；转置时交换 row/col。LOWER 非转置：`idx = row - col + col*lda`。
- x 偏移：`incx ≥ 0` 为 `idx*incx`，否则 `(n-1-idx)*(-incx)`。
- 复数按两个连续 float 存放，标量路径以 `complexIdx * 2` 访问实部 / 虚部。
- `CONJ`：非对角 `aIm = -aIm`；`NON_UNIT` 时  
  `x = (sum * conj(diag)) / |diag|²`。

**标量 Kernel（`CtbsvKernel<UPLO,TRANS,DIAG>`）**

1. `Init`：绑定 A/x 的 `GlobalTensor<float>`
2. `Process` → `ProcessRows`：申请 x 的实部 / 虚部 UB
3. 按 `kForward` 正向或反向扫行
4. `ComputeRow`：读取 `x[row]`，带内邻元复乘累减，按 diag 收尾并写回

**SIMT Kernel（`CtbsvSimt`）**

1. `n ≤ 4096`：先将 x 协同载入 UB，再逐行执行  
   `CtbsvSimtDotPartial`（FROM_GM=false）与 warp `asc_reduce_add`。  
   转置列点积拆分为 `CtbsvSimtTransMulPair` / `CtbsvSimtTransMulOne`，对角收尾使用 `CtbsvSimtTransScaleCol`。
2. `n > 4096`：x 保留在 GM（FROM_GM=true）
3. thread 0 完成对角除法并写回 `x[row]`，`asc_syncthreads()` 后进入下一行

转置 SIMT 初版按线程顺序累加，耗时可达 GPU 的百倍量级。改为 warp `asc_reduce_add` 后，TC_PF_1003 等转置用例低于标杆。

##### 3.2.2.2 Ascend C 实现流程图

```mermaid
flowchart TD
    S[aclblasCtbsv] --> V{校验}
    V -->|失败| R1[HANDLE_IS_NULLPTR / INVALID_ENUM / INVALID_VALUE]
    V -->|n==0 或 k==0且UNIT| R0[SUCCESS]
    V -->|通过| T[计算 numThreads 并组 tiling]
    T --> P{numThreads > 0?}
    P -->|否| SC[标量: Init + 申请 x UB]
    P -->|是| SM{n ≤ 4096?}
    SM -->|是| UB[SIMT-UB: 协同 LoadX]
    SM -->|否| GM[SIMT-Tiled: x 走 GM]
    SC --> LOOP
    UB --> LOOP
    GM --> LOOP
    LOOP[按前向/后向取下一行] --> DOT[带内邻元复乘累减]
    DOT --> DIAG{UNIT?}
    DIAG -->|是| WR[写回 x_row]
    DIAG -->|否| DIV[复数除对角 / conj对角]
    DIV --> WR
    WR --> MORE{还有行?}
    MORE -->|是| LOOP
    MORE -->|否| R0
```

##### 3.2.2.3 与标杆流程图的差异

| 差异点 | 标杆（cuBLAS / Netlib） | 本实现 | 原因 |
|--------|-------------------------|--------|------|
| 执行后端 | CUDA kernel / CPU Fortran | Ascend C AIV Kernel 直调 | 任务书要求合入 ops-blas，目标芯片为 950PR |
| 并行 | 实现私有，对外不暴露 | 行间串行；行内 SIMT 点积 | 回代依赖限制多核；宽带时行内点积可并行 |
| 双路径 | 通常单一实现 | 标量与 SIMT 按 n/带宽切换 | 对齐仓内 `stbsv`，并避免窄带 SIMT 亏性能 |
| 复数存储 | `cuComplex` / `COMPLEX` | `aclblasComplex`（两个 float） | 仓内公共类型 |
| 同步 | CUDA stream | `aclblasSetStream`，Host 不同步 | 与 ops-blas 其它接口一致 |
| 数据类型 | 另有 Ztbsv | 仅 COMPLEX64 | 任务书范围 |
| 模板分派 | 运行时或宏 | 12 路编译期特化 | 去掉热路径枚举分支 |

数学回代顺序、带状寻址、共轭语义、负步长与空操作与标杆保持一致。

### 3.3 支持硬件

| 芯片 | 是否支持 | 依据 |
|------|----------|------|
| Ascend 950PR | 支持 | 任务书唯一适配硬件，arch35 |
| Ascend 950DT | 不支持 | 任务书未要求 |
| Atlas A2 / A3（arch22） | 不支持 | 任务书未要求 |

README 产品支持表中，Ascend 950PR / 950DT 仅将 950PR 标注为支持。接口位于公共头，可供其它产品线后续复用同一原型。

### 3.4 算子约束限制

- 仅 COMPLEX64，不支持 FP32 / FP64 / COMPLEX128
- 仅单右端向量，无 batched / strided-batched
- 仅单核；n、k 为运行时标量，无动态 shape 框架
- 无广播
- `handle == nullptr` 返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`
- `uplo` / `trans` / `diag` 非法返回 `ACLBLAS_STATUS_INVALID_ENUM`
- `lda ≥ k+1`（实现按仓内 stbsv 写成 `lda > k`，不与 n 比较）
- `incx ≠ 0` 且 `incx ≠ INT_MIN`；负步长合法
- 不做奇异性检查；`NON_UNIT` 时对角必须非零
- 仅引用 uplo 指定的三角带
- Host 不做 stream 同步

---

## 四、特性交叉分析

| 交叉项 | 结论 |
|--------|------|
| 与 `aclblasStbsv` | 同目录、tiling 形状一致；Ctbsv 增加共轭与复数除法，非法枚举返回 `INVALID_ENUM`，SIMT 按 trans 使用 24/48 带宽门槛 |
| 与其它 BLAS 接口 | 仅共享 handle / stream / 公共头，不改变现有算子控制流 |
| 与 ACLNN / 图编译 | 不注册 Op，无图融合，无 workspace 与其它算子复用 |
| 与测试框架 | 复用 `test/frame` 的 CSV 与 GTest；golden 为 cblas，不依赖 GPU |
| 内存别名 | A 只读，x 原地写；A 与 x 重叠时行为与 BLAS 同样未定义 |
| 线程安全 | 同一 handle / 同一 stream 上的调用由调用方串行化 |

与卷积、归一化等 NN 算子无精度或格式交叉。

---

## 五、可维护性分析

### 5.1 精度标准 / 性能标准

| 验收标准 | 描述 | 来源 |
|----------|------|------|
| 精度 | COMPLEX64 按 FLOAT32 分量：rtol = 2⁻¹⁰，atol = 2⁻¹⁶，matched_ratio ≥ 0.99；max_abs_error ≤ 1e-2 或 32 ULP。golden 为 `cblas_ctbsv`，实部 / 虚部分开比对 | 任务书 §3.2，[生态算子开源精度标准](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md) |
| 性能 | Ascend 950PR，warmup 后有效采样 >50 次取平均；4 个典型 case 平均耗时不高于 917.5 / 1831.11 / 3753.09 / 4898.68 us | 任务书 §3.3 |
| 测试规模 | CSV 1200 条（精度 1000 + 性能 200），覆盖 uplo×trans×diag、尺寸 / 带宽、lda padding、±1/±2/±3 步长、零维 / 空指针 / 非法枚举（`INVALID_ENUM`）/ handle 空指针 | 任务书 §3.5 |

自测典型性能（`ctbsv_bench`，warmup=20，repeats=100，提交 `4b3e5f4`，2026-09-15 复测）：179.51 / 639.70 / 1322.20 / 4242.46 us，四条均低于标杆；200 条 PF 用例 GPU/NPU 均不低于 0.4。

### 5.2 兼容性分析

- **接口兼容**：新增 API，不改变已有 `aclblasStbsv` 及其它符号。
- **ABI**：声明位于 `cann_ops_blas.h`，无 950PR 私有平行接口；后续 arch22 或其它精度可复用同一原型。
- **行为兼容**：带状布局、负步长、空操作与 cuBLAS / Netlib 对齐。非法枚举按 ops-blas interface-spec 返回 `INVALID_ENUM`（同族 `aclblasStbsv` 仍为 `INVALID_VALUE`）。
- **版本**：针对 CANN 9.1.0 与 Ascend 950PR；不承诺更低 CANN 或其它芯片。
- **文档**：`blas/stbsv/README.md` 已登记 `aclblasCtbsv` 及 950PR 支持情况。
