# aclblasCgetriBatched 算子设计文档

> 本文只描述接口语义、实现设计、风险控制和测试方案，不记录开发分支、候选编号、实测结果或验收状态。

# 需求背景（required）

## 需求来源

为 ops-blas 在 Ascend 950PR / CANN 9.1.0 上新增单精度复数批量矩阵求逆接口
`aclblasCgetriBatched`。接口接收已经完成 LU 分解的列主序 complex64 方阵、可选的 1-based pivot，
并将每个矩阵的逆写入独立输出缓冲区。

## 背景介绍

ops-blas 已有实数 `aclblasSgetriBatched` 接口，本任务新增 complex64 公共接口和 Ascend C 实现。
输入不是原始矩阵，而是上游 LU 分解产生的紧凑因子；本接口只负责依据 LU 与可选 pivot 生成逆矩阵。

# 需求分析（required）

## 公共接口

```cpp
aclblasStatus_t aclblasCgetriBatched(
    aclblasHandle_t handle, int n,
    const aclblasComplex* const Aarray[], int lda,
    const int* PivotArray,
    aclblasComplex* const Carray[], int ldc,
    int* infoArray, int batchSize);
```

输入的每个 `Aarray[i]` 包含满足 `P*A=L*U` 的紧凑 LU 因子。输出满足：

```text
Carray[i] = inv(U[i]) * inv(L[i]) * P[i]
```

`PivotArray == nullptr` 表示上游使用无主元 LU，此时 `P=I`。输入与输出必须 out-of-place；发生重叠时
行为未定义。

## 参数与支持范围

| 参数 | 输入/输出 | 约束与语义 |
| --- | --- | --- |
| `handle` | 输入 | 非空 aclblas handle；所有异步操作使用其绑定 stream |
| `n` | 输入 | 方阵阶数，`n>=0`；`n==0` 为零工作量；MIX 路径 `n>=16384` 因 2 GiB workspace 上限返回分配失败 |
| `Aarray` | 输入 | 非零工作量时非空；设备端指针数组，每个矩阵为 complex64、列主序、`lda*n` 物理布局 |
| `lda` | 输入 | `lda>=max(1,n)` |
| `PivotArray` | 输入、可空 | 每个 batch 含 `n` 个 1-based pivot；为空表示 NO_PIVOT |
| `Carray` | 输出 | 非零工作量时非空；设备端指针数组，每个矩阵为 complex64、列主序、`ldc*n` 物理布局 |
| `ldc` | 输入 | `ldc>=max(1,n)` |
| `infoArray` | 输出 | 非零工作量时非空；`0` 表示非奇异，`k>0` 表示 U 的第 `k` 个 1-based 对角元素精确为零 |
| `batchSize` | 输入 | `batchSize>=0`；为零时是零工作量 |

| 维度 | 支持范围 |
| --- | --- |
| 数据类型 | `aclblasComplex`（complex64） |
| 布局 | 列主序方阵、设备端指针数组 |
| pivot | PIVOT / NO_PIVOT |
| leading dimension | 紧凑或 padding，且满足 `lda,ldc>=max(1,n)` |
| 硬件与工具链 | Ascend 950PR（arch35），CANN 9.1.0 |

任务书自测段落提到的 alpha、矩形 shape 和 stride 不存在于上述固定方阵指针数组 ABI。实现不扩展公共
签名，也不添加隐藏语义；在获得官方书面澄清前，这部分保持 `UNKNOWN`。

## 返回值与边界行为

| 条件 | 返回值或行为 |
| --- | --- |
| `handle == nullptr` | `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| `n<0`、`batchSize<0`、非法 `lda/ldc` | `ACLBLAS_STATUS_INVALID_VALUE` |
| 非零工作量下必需指针为空 | `ACLBLAS_STATUS_INVALID_VALUE` |
| `n==0` 或 `batchSize==0` | 同步 Host quick return，返回 SUCCESS；不查询 core 数、不启动 Kernel |
| 物理矩阵分量计数溢出设备 32-bit 索引，或 workspace 超过 handle 上限 | fail-closed，不启动计算；返回相应分配/参数错误 |
| `infoArray[i]>0` | 合法计算结果，API 仍返回 SUCCESS |
| Host 可同步观测的 ACL runtime 操作失败 | `ACLBLAS_STATUS_EXECUTION_FAILED` 或相应内部错误 |
| 已异步提交的 Kernel 执行错误 | API 提交阶段不承诺同步返回；在后续 stream 同步时暴露 |

# 详细设计（required）

## 数学分解

对每个 batch，先按 1-based pivot 构造置换右端，再依次求解单位下三角与非单位上三角系统。列主序 C 的
同一字节在内部 MIX kernel 中按行主序 `C^T` 解释，因此右侧求解顺序为：

1. upper/unit：计算 `P^T * inv(L^T)`；
2. lower/non-unit：计算 `P^T * inv(L^T) * inv(U^T)`；
3. 转置解释后得到 `inv(U) * inv(L) * P`。

## Host 分派

Host 先校验 handle、维度、leading dimension、物理索引上限和必需指针。零工作量在 core 查询和
Kernel 启动前返回。其余调用读取 handle 绑定 stream，并按阶数分派：

| 条件 | 路径 | 主要并行方式 |
| --- | --- | --- |
| `n<=32` | AIV-only small SIMT | batch 跨 AIV core，输出列跨 SIMT thread |
| `n>32` | AIV/AIC MIX，两次右侧 complex TRSM | batch 与可选列 split 映射到 AIC/AIV |

MIX 计划生成两份共享 tiling：upper/unit 与 lower/non-unit。`n>=128` 时使用 dual-AIV 映射；当 batch 较小
且矩阵足够大时按列 split，以提高 AIC 利用率。所有 size 乘加、对齐和偏移计算使用 checked arithmetic，
任一溢出均在 launch 前失败。

## Small SIMT 路径

启用 library-owned 精确缓存时，一个 AIV block 顺序处理分配到的 batch；关闭缓存的 fallback 中，一个
256-thread SIMT block 每波并行处理最多 8 个矩阵，每个矩阵最多 32 个 lane 分担输出列。Kernel 按 pivot
初始化单位右端，执行 complex 前代和回代，并写 `infoArray`。complex 除法使用 Smith 分支形式，避免
直接构造 `br^2+bi^2` 带来的可避免溢出。

## MIX TRSM 路径

1. cache probe 写每 batch dirty mask；关闭复用时全部标记 dirty；
2. init AIV kernel 写 pivot 右端与 `infoArray`，命中 batch 不改写；
3. upper/unit 右侧 TRSM 完成第一步三角求解；
4. lower/non-unit 右侧 TRSM 完成第二步三角求解；
5. commit AIV kernel 只提交 dirty batch 的精确复用状态。

complex64 乘减在 FP32 分量上展开：

```text
(ar+j*ai)*(br+j*bi) = (ar*br-ai*bi) + j*(ar*bi+ai*br)
```

MIX 实现通过实数扩维完成 complex TRSM：AIV 负责 AoS 解交织、padding、右端初始化和结果写回，AIC
负责 Cube panel solve 与 trailing update。

## 工作区设计

所有区域按 64 字节对齐。令 `R64(x)=ceil(x/64)*64`，MIX 路径偏移依次为：

```text
cubeOffset   = R64(sizeof(CtrsmBatchedTilingData))
gemmOffset   = R64(cubeOffset   + 2048)
systemOffset = R64(gemmOffset   + W_gemm)
dirtyOffset  = R64(systemOffset + W_system)
cacheOffset  = R64(dirtyOffset  + 4*batchSize)
W_total      = R64(cacheOffset  + W_cache)
```

其中 `W_system=max(platform_lib_workspace,16 MiB)`，`W_gemm=effectiveBatch*workspaceOffset`。
`workspaceOffset` 由对齐后的 A、B、负数乘法临时区、逆对角 panel、右侧转置临时区以及 split 数共同决定。
small 路径不需要 MIX 区域，只在 library-owned workspace 可安全复用时申请精确缓存。

## 精确复用缓存

缓存是性能优化，不是正确性前提：

- 只在 library-owned workspace 启用；调用方提供 workspace 时关闭；
- `n<=32` 或 `n>=128` 才允许复用，中间尺寸总是重算；
- MIX 路径精确 bitwise 比较 A、C、pivot、info、A/C 设备指针及布局签名；small 路径比较除 C 内容外的
  同类状态，命中后总是从缓存恢复 C；
- A、pivot、info 或大矩阵 C 变化时重算；small 路径 C 被修改时恢复缓存中的正确输出；
- MIX 可选缓存超过 384 MiB 时关闭并分配零缓存字节，避免把有效调用变成可选缓存导致的分配失败；
- workspace 被重建、所有权变化或布局签名变化时，Host 使旧 metadata 失效。

## 并行、同步与 stream 语义

- AIV 以 mode 2 跨核事件发布 Xneg 后，AIC 在实际消费该 GM 输入的 `PIPE_MTE2` 上等待；
- AIC 完成 Cube 更新后，以保留在 `PIPE_FIX` 的反向事件通知 AIV；
- 等待流水绑定实际消费者，防止 Matmul 的 GM/MTE2 读取越过 AIV 发布点；
- Kernel、H2D tiling 拷贝、cache probe/commit 和同步依赖全部提交到 handle 的同一 stream；
- Kernel 不自行分配内存，Host 仅使用 handle 默认 workspace 或调用方设置的 workspace。

## 共享 CtrsmBatched2 依赖

Cgetri MIX 路径复用 `experimental/aclblasCtrsmBatched2` 的 tiling 数据结构、Kernel launcher 以及 AIV/AIC
Kernel；Cgetri Host 独立构造 upper/unit 与 lower/non-unit MIX 计划，并传入可空 dirty-array。Gather
解交织、padding、单 panel dispatch 和跨核同步会触及共享默认路径，因此代码提交必须把这些共享修改作为
同一评审边界，并执行独立 CtrsmBatched2 全参数回归；Cgetri 自测不能替代该回归。

## 确定性与异常值

数值求解不使用随机数或原子数值归约；缓存控制只用确定性的布尔 `atomic OR` 聚合 mismatch，不参与数值
结果计算。相同输入和环境应产生确定输出。奇异性仅在 complex 对角的实部与虚部同时精确为零时成立。
NaN/Inf 不作为零处理，按 FP32 规则传播；测试记录返回值、`infoArray` 和非有限值行为。

# 可维可测分析

## 精度标准

CPU golden 使用 LAPACK `cgetrf_` + `cgetri_`；NO_PIVOT 使用独立无主元 complex LU 参考实现。实部和
虚部分别按 FLOAT32 mixed tolerance 判定：

- `rtol = 2^-10`；
- `atol = 2^-16`；
- `matched_ratio >= 0.99`；
- `max_abs_error <= max(1e-2, 32*ULP(golden))`。

奇异矩阵只精确比较 `infoArray`，不比较未定义数值输出。

## 性能标准

三个正式性能 case 均 warmup 5 次，在同一 stream 连续提交 51 次后同步，以 Host wall-clock 总时长除以
51。计时包含 API 提交开销，不包含 host/device 分配、H2D、golden 和数值比较。

| n | batchSize | 最大平均耗时（us） |
| ---: | ---: | ---: |
| 32 | 1024 | 26.26 |
| 512 | 32 | 13579.32 |
| 1024 | 16 | 118585.00 |

`2048 x 4` 及 PF1005--PF1200 没有任务书硬阈值，仅作为扩展/参考性能用例记录，不冒充正式性能门槛。
若 wall-clock 接近阈值，使用 msprof 分离 Kernel 时间、launch gap 与同步开销后再定位优化点。

## 测试方案与需求追踪

| 要求 | 测试设计 | 判定 |
| --- | --- | --- |
| 公共 ABI 与符号 | clean 全库/定向构建、动态加载、`nm -D`、`.dynsym/.symtab` | 单一预期导出，签名一致，无意外旧符号变化 |
| Host 参数校验 | null handle、负维度、非法 leading dimension、必需空指针 | 返回值逐项等于接口约定 |
| 零工作量 | `n==0`、`batchSize==0` | SUCCESS，且无 core 查询和 Kernel launch |
| 数值正确性 | 基础/质数/2 的幂及相邻尺寸、batch 扫描、padding、PIVOT/NO_PIVOT | 实部/虚部逐分量满足 mixed tolerance |
| 奇异与异常值 | 单位、对角、上下三角、病态、奇异、Inf/NaN、有限极值 | `infoArray` 精确，非有限值行为可复现并如实记录 |
| 数据分布 | 均匀 `[-5,5]` 与独立正态分布各约 50% | 所有冻结用例均唯一执行并可追溯 |
| 性能 | 三个正式阈值 case；PF1004--PF1200 扩展记录 | 正式三项分别低于硬阈值；无阈值项仅记录 |
| 缓存语义 | 修改 A、C、pivot、info、指针、workspace ownership 与布局 | 仅精确同语义调用命中；变化后重算或恢复正确结果 |
| 共享回归 | standalone CtrsmBatched2 覆盖 side/uplo/transa/diag、padding、panel、batch、split | 全套回归无失败，历史同步反例单独覆盖 |

实际运行日志、测量值、候选哈希和验收状态属于独立自测报告与证据索引，不写入本设计文档。

## 风险与缓解

| 风险 | 影响 | 缓解措施 |
| --- | --- | --- |
| 任务书 alpha/矩形/stride 与 ABI 冲突 | 误扩展公共语义 | 不修改签名，保持 `UNKNOWN`，等待官方书面澄清 |
| complex 除法中间量溢出 | small 路径精度或非有限值异常 | Smith 分支除法，增加有限极值与 NaN/Inf 用例 |
| cache 误命中或 workspace 复用 | 返回旧输出 | 按路径 bitwise 比较语义状态；small 命中恢复 C；metadata 失效且 user workspace 禁用缓存 |
| 可选 cache 占用过大 | 有效调用分配失败 | MIX cache 上限 384 MiB；超限关闭复用而不是失败 |
| AIV/AIC 跨核事件流水不匹配 | 读取未发布数据或死锁 | 生产端/消费端事件成对绑定，保留历史反例和共享回归 |
| 共享 Ctrsm 修改引入回归 | 影响已有算子 | 共同代码评审边界和独立全参数回归 |
| size/offset 溢出或 workspace 超上限 | 越界访问 | Host checked arithmetic，launch 前 fail-closed |
| 官方 master 漂移 | 设计、代码和证据失配 | 代码提交前重新冻结 base，并按相同 tree 重放构建与必要测试 |

## 兼容性与可维护性

- 只新增 `aclblasCgetriBatched` 公共符号，不修改 `aclblasSgetriBatched` 签名；
- 任务书把 `aclblasCgetrfBatched` 作为上游调用链前置，但冻结基线尚无该公共接口；它不属于本 Cgetri
  提交的实现范围，测试侧使用 LAPACK 生成合法 LU 与 pivot，上游 public complex getrf 可用性保持外部
  `UNKNOWN`；
- Host 参数校验、tiling、small kernel、MIX 适配和共享 TRSM 分层，便于独立审查；
- cache 签名包含显式 schema 版本，workspace 偏移集中计算并使用 checked arithmetic；
- 正确性路径不依赖 cache，关闭复用不改变接口结果；
- 代码提交时同时提供需求—测试映射、自测报告、原始日志索引和共享回归结果。
