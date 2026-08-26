# aclsparseSpMM 算子设计

## 需求背景

### 需求来源

本任务补齐 Ascend 950PR 上的 CSR SpMM，并通过同一套底层实现支持
`torch.sparse.addmm`。规格以《aclsparseSpMM 算子开发（950）任务书》为准，接口
放置和构建方式遵循 ops-sparse 仓库约定。需求与实现、测试证据的对应关系记录在
`ops-sparse/docs/aclsparseSpMM_traceability.md`。

### 背景介绍

SpMM 是图神经网络和稀疏线性代数中的基础操作。本任务使用 CSR 存储稀疏矩阵，
输出为稠密矩阵。上层需要对齐 PyTorch 2.7 及以上的 `torch.sparse.addmm`
行为，底层继续使用 ops-sparse 已有的 aclsparse 描述符和三阶段调用流程。

### 现有实现

仓库已经包含 aclsparse 描述符、handle、stream 管理和 SpMM 目录，Ascend 950PR
分支原先只覆盖部分数据类型。此次开发沿用已有公开 API，不再增加同功能接口；
新增 BF16、complex64、Python/ATen 分派，并补齐参数校验、预处理复用和测试。

## 需求分析

### 计算定义

```text
C = alpha * op(A) * op(B) + beta * C
```

`A` 为二维 CSR 矩阵，`B`、`C` 为二维稠密矩阵。`op` 支持非转置、转置和共轭
转置。Python 接口对应：

```python
torch.sparse.addmm(input, mat1, mat2, *, beta=1, alpha=1) -> Tensor
```

其中 `mat1` 对应 `A`，`mat2` 对应 `B`，`input` 对应公式右侧原始 `C`。Python
接口返回新 Tensor；公开 C API 保持原地更新 `C` 的语义。`input` 可以是
`[M,N]`、`[N]`、`[1,N]`、`[M,1]` 或其他能广播到 `[M,N]` 的 shape。

### 能力范围

| A values / B / C | computeType | 累加类型 |
|---|---|---|
| `ACL_FLOAT16` | `ACL_FLOAT` | FP32 |
| `ACL_BF16` | `ACL_FLOAT` | FP32 |
| `ACL_FLOAT` | `ACL_FLOAT` | FP32 |
| `ACL_COMPLEX64` | `ACL_COMPLEX64` | complex64 |

兼容保留原有 INT8×INT8→INT32 组合，不纳入本任务验收。CSR row offsets 和 column
indices 必须同时为 `ACL_SPARSE_INDEX_32I`，index base 支持 0 和 1。B、C 支持
row-major 和 column-major；row-major 要求 `ld >= cols`，column-major 要求
`ld >= rows`。

### 需求拆解

1. 保持 `GetBufferSize -> Preprocess -> SpMM` 三阶段 C API 和既有 ABI。
2. 支持四种必选 dtype、N/T/H、两种稠密布局和 Host/Device scalar pointer mode。
3. 实现 `SparseCsrPrivateUse1` 分派，不允许回退到 CPU。
4. 处理 input 广播、非连续稠密 Tensor、空矩阵和 `beta=0` 特殊语义。
5. 使用任务书精度标准验收；性能采集只统计规定调用范围内的 Device Kernel。
6. Ascend 950PR 实现不得破坏 `arch22` 的 Ascend 910B 编译和行为。

## 详细设计

### 算子分析

CSR 的非零元素按行连续存放。在 N/N 主路上，一个计算单元负责一个稀疏行和
一段输出列：遍历该行的非零元素，用 column index 定位 B 的对应行，在 FP32
或 complex64 累加器中完成乘加。行之间没有数据依赖，适合在 AIV 核间分配；行内的
输出列用 SIMT 线程并行。T/H 和列主布局使用通用索引路径保证功能完整性。

### 算子实现

#### 软件分层

| 层次 | 文件 | 职责 |
|---|---|---|
| 公开接口 | `include/cann_ops_sparse.h` | 算法枚举及三阶段 API 声明 |
| 描述符公共层 | `sparse/common/` | CSR/Dense 描述符、handle、pointer mode、stream |
| Host | `sparse/spmm/arch35/spmm_host.cpp` | 参数校验、workspace、tiling、Kernel launch |
| CSR 预处理 | `sparse/spmm/arch35/spmm_csr_mat.cpp` | 行负载统计、重排和分核边界 |
| Device | `sparse/spmm/arch35/spmm_kernel.cpp` | 各 dtype 的 SpMM 计算 |
| Python/ATen | `python/ops_sparse_npu/registration.py` | 参数适配、plan 复用和 NPU 分派注册 |

#### C API

```c
aclsparseStatus_t aclsparseSpMMGetBufferSize(
    aclsparseHandle_t handle, aclsparseOperation_t opA,
    aclsparseOperation_t opB, const void *alpha,
    aclsparseConstSpMatDescr_t matA,
    aclsparseConstDnMatDescr_t matB, const void *beta,
    aclsparseDnMatDescr_t matC, aclDataType computeType,
    aclsparseSpMMAlg_t alg, size_t *size);

aclsparseStatus_t aclsparseSpMMPreprocess(
    aclsparseHandle_t handle, aclsparseOperation_t opA,
    aclsparseOperation_t opB, const void *alpha,
    aclsparseConstSpMatDescr_t matA,
    aclsparseConstDnMatDescr_t matB, const void *beta,
    aclsparseDnMatDescr_t matC, aclDataType computeType,
    aclsparseSpMMAlg_t alg, void *buffer);

aclsparseStatus_t aclsparseSpMM(
    aclsparseHandle_t handle, aclsparseOperation_t opA,
    aclsparseOperation_t opB, const void *alpha,
    aclsparseConstSpMatDescr_t matA,
    aclsparseConstDnMatDescr_t matB, const void *beta,
    aclsparseDnMatDescr_t matC, aclDataType computeType,
    aclsparseSpMMAlg_t alg, void *buffer);
```

`GetBufferSize` 只返回 workspace 字节数。`Preprocess` 根据 CSR pattern 生成行重排
和分核边界，并把 buffer 记录为该稀疏描述符的 active buffer。`SpMM` 读取 handle
中的当前 stream 异步下发；当 buffer、`opA` 或算法与 active 记录不一致时重新执行
预处理。调用方在相关 stream 完成前必须保持描述符、输入、workspace 和 Device
scalar 有效。

#### 参数校验

Host 侧统一完成以下检查：

- handle、alpha、beta、描述符和 workspace 指针有效；
- A 为 CSR，row offsets/column indices 均为 INT32，index base 合法；
- 根据 `opA/opB` 推导 M、K、N，校验矩阵 shape 和 leading dimension；
- A/B/C dtype、computeType、算法和布局组合受支持；
- M、K、N、nnz 和 ld 不超过有符号 INT32 范围；
- ALG3 仅接受 `opA=N`，且拒绝 `opB=H`。

Python 适配层在创建描述符前转换 alpha/beta。实数 dtype 接受整数和浮点
scalar，拒绝虚部非零的复数；complex64 接受实数或复数。`beta=0` 时仍然
检查 input 的 shape、dtype 和 device，但 Kernel 不读取 input 数值。

非法参数返回 `ACL_SPARSE_STATUS_INVALID_VALUE`；不支持的格式、dtype 或算法组合
返回对应的 `NOT_SUPPORTED` 状态；预处理阶段的 Runtime copy 失败返回
`ACL_SPARSE_STATUS_EXECUTION_FAILED`。Kernel 异步执行错误按 Runtime stream 同步规则上报。

#### Workspace 与预处理

workspace 按 64 Byte 对齐，头部为 tiling 保留区，后续存放 M 个 row reorder
和 `blockDim + 1` 个 bin edge。当前 tiling 通过 Kernel 形参传值，不从保留区读取。
blockDim 从平台 AIV 核数取得，平台信息不可用时使用仓库兼容默认值。M 被
限制在 INT32 范围内，workspace 尺寸使用 64-bit 中间量计算。调用方必须按
`GetBufferSize` 返回值分配 buffer。

非转置 CSR 的预处理先将 row offsets 拷回 Host，计算每行 nnz，再决定分核方式：

- 一般 CSR：按行 nnz 降序，将整行放入当前累计负载最小的核；
- `M >= 131072` 且行度差不超过 1：保留原始行序，按累计 nnz 分位点切连续区间；
- 转置路径：使用连续行区间，不做 CSR 行模式优化。

预处理还记录均匀 degree，并对受支持的 degree 检查列模式周期，用于选择 B tile
复用路径。该分析只读取 CSR 内容，不依赖 case 名称或固定 shape。由于当前实现需要
D2H 读取 row offsets，`Preprocess` 是一次性同步步骤；重复 `SpMM` 不再做该同步。

#### Tiling 与 Kernel

tiling 保存 M/K/N、ldb/ldc、op、layout、index base、pointer mode、行模式信息和
workspace offset。Host pointer mode 下 alpha/beta 按值写入 tiling；Device pointer
mode 下由 Kernel 从 Device 地址读取。

非转置 row-major 是性能主路径。输出按 `(row, column tile)` 切分，Kernel 根据 dtype
和 CSR 特征选择实现：

| 路径 | 使用条件 | 处理方式 |
|---|---|---|
| FP32 vector | row-major，规则 CSR 或宽 N | UB 内按列分块，连续搬运 B/C，FP32 FMA |
| FP32 periodic | 预处理确认列模式周期 | 同一 B tile 在重复行间复用 |
| FP16/BF16 fast | row-major，ldb/ldc 对齐，`N >= 2` | 每线程计算 16 列，FP32 累加；FP16 对齐整块使用 128-bit load |
| complex64 vector | row-major 宽列 | 实部、虚部分离计算后交错写回 |
| generic SIMT | 其他布局、非规则 shape、T/H | 按 CSR 行和列 tile 直接计算 |
| FP32 high precision | 高精度算法 | Kahan 补偿求和 |

`beta=0` 时所有路径都跳过原 C 读取，避免 input 中的 NaN/Inf 传播。FP16 写回前按
有限表示范围处理；BF16 保留自身动态范围。complex64 的 H 路径在乘法前执行共轭。
空输出（M=0 或 N=0）直接成功返回；K=0 或 nnz=0 仍执行 `C=beta*C`。

行重排以整行为最小分核单位，行内再按输出列 tile 分配 SIMT 线程。这一划分保持
CSR 连续访存，同时让宽输出行在核内并行。

#### Python / ATen 适配

适配层注册 `aten::_sparse_addmm` 和 `aten::addmm.out` 的
`SparseCsrPrivateUse1` 实现。进入 C API 前完成：

1. 校验 mat1 为二维 CSR、mat2 为二维 Dense，三者 dtype 和 NPU device 一致；
2. 校验 crow/col 为 INT32，并按 PyTorch 规则检查 input 是否可广播到 `[M,N]`；
3. 将无法由 Dense 描述符表达的 mat2/input 转为连续 Tensor；
4. 创建输出和描述符，在当前 NPU stream 上调用 ALG2；
5. 以 CSR crow/col/values 和连续 B 的指针、shape、dtype、device 作为 key，
   复用最近一次 plan 和 workspace。

完整连续 input 使用仓库内部 `aclsparseSpMMOutOfPlace`，让 Kernel 分别读取 input、
写入新输出，避免预先复制整块 C。该入口不写入公开头文件，不改变公开 C API 的原地
语义。广播或 strided input 先物化后调用公开接口。

## 支持硬件

| 硬件 | 代码目录 | 状态 |
|---|---|---|
| Ascend 950PR | `sparse/spmm/arch35/` | 本任务实现与验收目标 |
| Ascend 910B / 910_93 | `sparse/spmm/arch22/` | 保持原实现并做共存编译回归 |

## 算子约束

- 仅支持 CSR，不新增 COO、CSC、BSR 或 Blocked-ELL SpMM。
- Tensor 间不做 dtype 提升；Python 三个输入必须为相同 dtype。
- Python 输入必须位于同一 NPU device；跨设备输入报错。
- DEFAULT 使用通用选路；ALG1 为 column-major 优先，ALG2 为 row-major 优先，
  ALG3 为确定性路径；`CSR_FP32_HIGH_PRECISION_ALG` 仅用于 FP32 Kahan 累加。
- `opA=T/H` 为功能路径，不作为本任务性能主路径。
- active workspace 只绑定 CSR pattern；复用期间不得修改 row offsets 或 column
  indices。A values、B/C values 和 alpha/beta 可在执行时更新。

## 可维可测分析

### 精度标准

CPU Golden 按任务书规定计算：FP16/BF16 使用 FP32，FP32 使用 FP64，complex64
使用 complex128。实数分量按
`abs(actual-golden) <= atol + rtol*abs(golden)` 判断，匹配率不低于 0.99，并检查
`max(A, 32*ULP(golden))` 绝对误差上限。complex64 实部、虚部分别按 FP32 标准
验收；NaN/Inf 的位置和 Inf 符号必须一致。ALG3 另做重复执行 bit-wise 检查。

| dtype | rtol | atol | A |
|---|---:|---:|---:|
| FP16 | `2^-9` | `2^-9` | `1e-1` |
| BF16 | `2^-6` | `2^-6` | `1e0` |
| FP32 | `2^-10` | `2^-16` | `1e-2` |
| complex64 | 实、虚部均按 FP32 | 实、虚部均按 FP32 | 实、虚部均按 FP32 |

### 测试范围

| 类别 | 覆盖内容 |
|---|---|
| C++ 接口 | 四种 dtype、N/T/H、base 0/1、两种布局、两种 pointer mode |
| Python | 广播、非连续 Tensor、out 变体、dtype/device/shape 拒绝路径 |
| 边界 | 空维、nnz=0/1、空行、长尾行、padded ld、`beta=0` + NaN/Inf |
| 安全 | A/B 只读、C 和 workspace 边界哨兵、重复创建/执行/销毁 |
| 精度 | 任务书 200 条用例、四 dtype 和混合容差 |
| 性能 | 附带 50-case manifest，以及 P-01/P-02/P-03，10 次预热 + 30 次采样 |
| 分派 | Profiler 确认命中自研 Kernel，无 CPU fallback |

### 性能标准

NPU 侧复用描述符、workspace 和预处理结果，采集 Kernel 总耗时的 median 和 p90。
数据生成、首次编译、Host 到 Device 搬运和预处理耗时不计入正式采样。FP16、BF16、
FP32 目标为不低于 A100 1.0 倍，complex64 目标为不低于 A100 0.8 倍。

| 用例 | M×K×N / nnz | dtype 与 A100 Kernel 基线（μs） |
|---|---|---|
| P-01 | 2,708×2,708×1,433 / 10,556 | FP32 87.040 |
| P-02 | 169,343×169,343×128 / 1,166,243 | FP16 321.536；BF16 413.920；FP32 204.576 |
| P-03 | 2,449,029×2,449,029×256 / 61,859,140 | FP16 25,446.496；BF16 33,704.096；FP32 16,571.232；complex64 38,806.400 |

## 兼容性分析

本次修改复用已有 API 名称和描述符，不改变现有调用方 ABI。新增算法枚举值按现有
枚举顺序保留，arch22/arch35 分目录维护。公开头文件、算子 README、Host、Kernel
和测试需同步修改；新增 dtype 或布局组合时，先扩展校验矩阵，再补充 Kernel 和正反
用例，不能只放宽 Host 检查。
