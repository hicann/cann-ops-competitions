# Atlas 950 单精度实数 Cholesky 接口族设计文档

| 项目 | 内容 |
| --- | --- |
| 任务 | 9月社区任务-单精度实数 Cholesky 分解、求解和批量接口（950） |
| 目标仓库 | `cann/ops-solver` |
| 目标硬件 | Atlas 950PR（Ascend 950，dav-3510） |
| 数据类型 | FLOAT32 |
| 接口 | `aclsolverSpotrf`、`aclsolverSpotrs`、`aclsolverSpotri`、`aclsolverSpotrfBatched`、`aclsolverSpotrsBatched` |
| 贡献者目录 | `XYz-dev-max` |

## 1. 需求背景

### 1.1 需求来源

`ops-solver` 已提供 LU 分解、矩阵求逆和特征值等能力，但尚缺少面向实对称正定矩阵的 Cholesky
接口族。本任务要求在 Atlas 950PR 上补齐五个计算接口及 `Spotrf`、`Spotri` 的 workspace 查询接口，
接口语义、参数顺序和异常约定对齐 cuSolver DN legacy API。

五个接口必须作为一个整体交付，核心计算使用 AscendC/CATLASS 在 AI Core 上执行，不允许以 CPU
fallback 代替。实现还需满足列主序、`lda/ldb` padding、Device 指针、调用方 stream、确定性和
真实 `info` 写回等契约。

### 1.2 目标

1. 支持 `LOWER` 和 `UPPER` 两种存储侧。
2. 支持单矩阵阶数 `n=1..4096`，`Spotrs` 的 `nrhs=1..128`。
3. 支持批量接口至少覆盖 `batchSize=1..30000`，且不设置小于任务性能用例规模的人为上限；其中
   `SpotrsBatched` 仅支持 `nrhs=1`。
4. 对所有合法 `lda/ldb >= max(1,n)` 保持正确的列主序访问，不读指定三角之外的输入。
5. 参数错误返回 `ACLSOLVER_STATUS_INVALID_VALUE`，并在输出指针可写时按参数序号写入负值。
6. 非正定或奇异输入按 LAPACK 约定写入首个失败位置。
7. 合法输入在相同 stream 上重复执行时结果 bit-wise 一致。
8. 在任务书全部性能用例上达到不低于 GPU 参考数据 0.35 倍的门禁。

### 1.3 非目标

- 不支持广播。
- 不扩展到 FP16、FP64、复数或稀疏矩阵。
- 不改变现有 handle/stream 生命周期接口。
- 不把 Device 数据复制到 Host 侧完成数值计算。

## 2. 接口与语义分析

### 2.1 公共枚举

在 `cann_ops_solver_common.h` 中定义：

```cpp
typedef enum {
    ACLSOLVER_FILL_MODE_LOWER = 0,
    ACLSOLVER_FILL_MODE_UPPER = 1
} aclsolverFillMode_t;
```

该枚举与 `cublasFillMode_t` 数值对齐。若仓库已有同名定义，则迁移到公共头文件并避免重复定义，
保持既有接口的源码兼容性。

### 2.2 公开函数

```cpp
aclsolverStatus_t aclsolverSpotrf_bufferSize(
    aclsolverHandle_t handle, aclsolverFillMode_t uplo,
    int n, float *A, int lda, int *Lwork);

aclsolverStatus_t aclsolverSpotrf(
    aclsolverHandle_t handle, aclsolverFillMode_t uplo,
    int n, float *A, int lda,
    float *Workspace, int Lwork, int *devInfo);

aclsolverStatus_t aclsolverSpotrs(
    aclsolverHandle_t handle, aclsolverFillMode_t uplo,
    int n, int nrhs, const float *A, int lda,
    float *B, int ldb, int *devInfo);

aclsolverStatus_t aclsolverSpotri_bufferSize(
    aclsolverHandle_t handle, aclsolverFillMode_t uplo,
    int n, float *A, int lda, int *Lwork);

aclsolverStatus_t aclsolverSpotri(
    aclsolverHandle_t handle, aclsolverFillMode_t uplo,
    int n, float *A, int lda,
    float *Workspace, int Lwork, int *devInfo);

aclsolverStatus_t aclsolverSpotrfBatched(
    aclsolverHandle_t handle, aclsolverFillMode_t uplo,
    int n, float *Aarray[], int lda,
    int *infoArray, int batchSize);

aclsolverStatus_t aclsolverSpotrsBatched(
    aclsolverHandle_t handle, aclsolverFillMode_t uplo,
    int n, int nrhs, float *Aarray[], int lda,
    float *Barray[], int ldb, int *info, int batchSize);
```

`A`、`B`、workspace、`devInfo/info` 以及批量指针数组均位于 Device。`Lwork` 的单位为 FLOAT32
元素个数。Host 只读取标量参数、handle 和 stream，不解引用 Device 指针。

### 2.3 数学语义

- `Spotrf(LOWER)`：计算 `A=L*L^T`，将 `L` 写回 A 的下三角。
- `Spotrf(UPPER)`：计算 `A=U^T*U`，将 `U` 写回 A 的上三角。
- `Spotrs`：输入为对应侧 Cholesky 因子，通过两次三角求解将 `X` 原地写回 B。
- `Spotri`：输入为 Cholesky 因子，将对称逆矩阵的指定三角原地写回 A。
- `SpotrfBatched`：对 Device 指针数组中每个矩阵独立执行分解并写 `infoArray[i]`。
- `SpotrsBatched`：对每个因子和单列右端执行求解，标量 `info` 只报告参数错误。

另一半三角不参与输入语义，可以被实现用作临时空间；测试和验收只比较 `uplo` 指定侧。

### 2.4 参数检查和 info 优先级

Host 按公开参数顺序执行检查。handle 不计入 `info` 的参数序号；handle 为空返回
`ACLSOLVER_STATUS_HANDLE_IS_NULLPTR`。其余非法参数返回 `ACLSOLVER_STATUS_INVALID_VALUE`，并在
对应 info 指针非空且存在可用 stream 时，通过轻量 Device kernel 写入 `-i`。

检查原则：

1. 先检查枚举、维数和 leading dimension，再检查问题规模所需的数据指针。
2. `n=0`、`nrhs=0` 或 `batchSize=0` 按任务书定义为空问题，成功返回且不访问数据矩阵。
3. `SpotrsBatched` 在非空问题下只接受 `nrhs=1`；`nrhs=2` 返回非法参数。
4. `SpotrfBatched` 参数错误只写 `infoArray[0]=-i`；数值计算时逐矩阵写完整数组。
5. `Spotrf` 遇到首个非正主元时写 `k>0` 并停止该矩阵，保留前 `k-1` 列已经得到的因子。
6. `Spotri` 遇到首个零对角因子时写 `k>0`。
7. `Spotrs` 和 `SpotrsBatched` 的数值路径不重新报告正定性，成功时 info 为 0。

## 3. 总体设计

### 3.1 工程结构

计划按任务书建议保留五个公开算子目录，并把真正可复用的分块原语放入公共工具目录：

```text
include/
├── cann_ops_solver.h
└── cann_ops_solver_common.h
src/
├── spotrf/
├── spotrs/
├── spotri/
├── spotrf_batched/
├── spotrs_batched/
└── utils/
    ├── cholesky_host_common.h
    └── kernel/cholesky/
test/
├── spotrf/
├── spotrs/
├── spotri/
├── spotrf_batched/
└── spotrs_batched/
docs/zh/
├── spotrf.md
├── spotrs.md
├── spotri.md
├── spotrf_batched.md
└── spotrs_batched.md
```

每个公开目录只保留 Host 入口、kernel launcher 和该接口特有逻辑，避免复制 POTF2、TRSM、SYRK、
指针数组解析以及参数检查代码。

### 3.2 Host 侧职责

Host 侧完成：

1. 参数校验和空问题处理。
2. 从 handle 取得调用方 stream。
3. 查询 950PR 的 AI Core 数量和片上存储能力。
4. 根据 `n/nrhs/batchSize/uplo/lda/ldb` 选择算法路径、块大小、使用核数和 workspace 布局。
5. 将定长 tiling 结构复制到 Device 并异步下发 kernel。
6. 将 ACL/launch 错误映射为 `aclsolverStatus_t`。

正常计算路径不调用 `aclrtSynchronizeStream`，不执行 Device-to-Host 数据回传。

### 3.3 Tiling 数据

公共 tiling 至少包含：

```text
opKind, uplo, n, nrhs, lda, ldb, batchSize,
panelSize, matrixTileM, matrixTileN,
vectorCoreNum, cubeCoreNum,
workspaceStride, pointerChunk, algorithmKey
```

所有尺寸乘法在 Host 使用 64-bit 或 `size_t` 检查溢出，确认可由公开的 32-bit 维数和 workspace
接口表达后再写入 tiling。

### 3.4 Kernel 分工

- AI Vector Core：POTF2 对角块、对角有效性检查、三角小块求解、padding/布局转换、info 写入。
- AI Cube Core：大块 GEMM/SYRK 更新和多右端 TRSM 的矩阵乘主路径。
- 大矩阵采用面板级确定性调度；同一输出 tile 只由一个确定的 core 写入，归约顺序固定。
- 批量小矩阵采用“矩阵/矩阵组到 core”的静态映射，相同输入内容走完全相同的指令路径。

## 4. 核心算法

### 4.1 Spotrf：分块右看 Cholesky

以 LOWER 为例，对面板 `k` 执行：

1. 对角块更新：`Akk := Akk - A(k,0:k)*A(k,0:k)^T`。
2. POTF2：在 UB 中对 `Akk` 做确定性的非分块 Cholesky，逐列检查主元。
3. 面板更新：`A(k+nb:n,k) := A(k+nb:n,k) - A(k+nb:n,0:k)*A(k,0:k)^T`。
4. TRSM：`A(k+nb:n,k) := A(k+nb:n,k)*inv(Lkk^T)`。
5. 后续面板在下一轮通过固定顺序继续更新。

UPPER 使用对称的转置访问和左侧 TRSM，不读取下三角输入。大块更新使用 Cube 矩阵乘路径；非 16/32
对齐的边界 tile 在 workspace 中补零，回写时只覆盖真实范围。对角块在进入 POTF2 前按固定顺序归约，
保证主元判断与输出确定性。

### 4.2 Spotrs：两阶段分块 TRSM

- LOWER：先解 `L*Y=B`，再解 `L^T*X=Y`。
- UPPER：先解 `U^T*Y=B`，再解 `U*X=Y`。

每个阶段按依赖方向遍历对角块。对角块由 Vector Core 求解，剩余 RHS 块使用 Cube GEMM 更新。
`nrhs` 较小时使用向量路径降低启动和格式转换开销；`nrhs` 较大时使用 Cube 路径提高利用率。

### 4.3 Spotri：三角逆与对称乘积

采用 LAPACK 风格的两阶段算法，避免逐列调用完整求解：

1. 计算三角因子的逆 `inv(L)` 或 `inv(U)`，对角块在 UB 中求逆，非对角块由 TRSM/GEMM 更新。
2. LOWER 计算 `inv(A)=inv(L)^T*inv(L)`，UPPER 计算 `inv(A)=inv(U)*inv(U)^T`。
3. 只将指定三角写回 A，另一侧不作为有效输出。

进入数值计算前按对角顺序检查因子；首个零或非法对角写 `devInfo=k`。中间三角逆和布局转换区由
`Spotri_bufferSize` 返回的 workspace 承载。

### 4.4 批量接口

批量参数 `Aarray/Barray` 是 Device 上的指针数组。kernel 按 `uintptr_t` 宽度分块读取地址，检查空地址后构造
对应的 `GlobalTensor`，不把指针数组复制到 Host。

按规模选择路径：

- 小矩阵：一个 Vector Core 同时驻留一个或多个矩阵，矩阵在 UB 中完成 POTF2/TRSV，减少 GM 往返。
- 中等矩阵：一个矩阵映射到一组 AI Core，批次之间并行，矩阵内部使用分块算法。
- 大矩阵或小 batch：复用单矩阵分块内核，以矩阵为最外层静态任务。

`SpotrfBatched` 的每个矩阵独立停止和写 `infoArray[i]`，某一矩阵失败不影响其他矩阵。
`SpotrsBatched` 固定 `nrhs=1`，使用两次批量 TRSV，最终只写一个标量 info。

## 5. Workspace 设计

### 5.1 Spotrf

workspace 用于：

- Cube 友好的对齐 tile 或临时布局；
- 面板更新临时区；
- 多核阶段同步和 kernel 控制字。

`Spotrf_bufferSize` 根据 `n` 和平台块大小返回元素数。计算接口校验 `Lwork` 不小于查询值；若查询值为
0，则允许 `Workspace=nullptr`。

### 5.2 Spotri

workspace 包含三角逆中间结果、乘积 tile 和同步区。布局按固定偏移计算并进行溢出检查，不在计算接口
内部申请额外大块 Device 内存。

### 5.3 Batched 与 Spotrs

公开接口没有 workspace 参数，因此使用 UB 和未使用三角区完成临时计算；必要的少量控制数据通过
固定大小 tiling/控制区传入，不把批量矩阵改成连续张量接口。

## 6. 精度与确定性设计

1. 所有乘加保持 FP32，不使用降低任务书精度的隐式 FP16 路径。
2. 对角主元在 FP32 固定顺序累加后判断，非正定位置采用 1-based 下标。
3. 指定三角外的 NaN/INF 不参与读取；指定三角内的异常值按生态精度规则处理并记录测试结果。
4. 每个输出 tile 的生产 core、K 维遍历顺序和同步顺序固定，不使用不确定性的跨核原子浮点累加。
5. 批量调度使用静态 stride 映射，不使用运行时抢占队列。
6. 相同输入和 stream 连续运行至少两次，以 `memcmp` 检查输出和 info。

精度首先执行任务包的逐元素混合容差判定；不通过时，再执行任务书规定的 LAPACK 归一化残差复核。

## 7. 性能设计

性能目标不是以串行标量 kernel 完成功能后再补救，而是在首版中采用块算法：

1. POTRF 的主要 FLOP 由 Cube 上的 GEMM/SYRK 承担。
2. 单矩阵大规格优先增大连续 Cube 计算比例，减少每面板 kernel 启动和全局同步次数。
3. `Spotrs` 按 `nrhs` 选择 Vector/Cube 路径，覆盖 1、8、32、128。
4. batched 性能用例 `n=32/128` 采用 UB 驻留和矩阵级并行，指针数组分块预取。
5. LOWER/UPPER 共用计算原语，避免一侧通过完整矩阵转置造成额外 O(n^2) 往返。
6. `lda/ldb` padding 只影响地址步长，不触发整矩阵重排。

最终使用 `msprof op` 对 P-01 至 P-12 逐例单独采集 `Task Duration(us)`，预热后多次运行取平均，要求
`T_NPU <= T_GPU/0.35`。任何未在 950PR 上实际采集的数据不得写成 PASS。

## 8. 测试方案

### 8.1 Host/API 契约

- null handle、非法 uplo、负维数、非法 lda/ldb、空指针、workspace 不足。
- `SpotrsBatched(nrhs=2)` 非空问题必须失败。
- `n=0`、`nrhs=0`、`batchSize=0` 成功且不访问数据。
- `_bufferSize` 与计算接口对 workspace 的要求一致。
- 参数错误的负 info 序号与公开签名一致。

### 8.2 功能和精度

- LOWER/UPPER。
- `lda/ldb=n`、`n+8`、`n+32`。
- `n=1`、2 的幂、2 的幂减 1、最大 `n=4096`。
- `Spotrs` 的 `nrhs=1/8/32/128`。
- batched 的 `batchSize=1/8/128/1024/3000` 以及任务包性能大 batch。
- 对角占优 SPD、均匀/正态 `B^T*B+nI`、非正定矩阵和奇异因子。
- 非默认 stream、重复执行确定性。
- 批量 A0 抽样：同一代表内容对应的所有槽位输出和 info bit-wise 相同。

执行器从任务包生成的 `.npz` 或 A0 现场构造结果读取输入，分配 Device 矩阵和 Device 指针数组，调用
接口后输出任务包要求的 `out32/info/status`。正式结论由 `verify_accuracy.py` 生成。

### 8.3 性能和内存

- P-01 至 P-12 每例独立 profiling，输出原始 `OpBasicInfo.csv` 和汇总 JSON。
- 记录 CANN 版本、驱动版本、950PR 型号、预热次数、计时次数和 stream 设置。
- 使用任务模板记录 Device workspace、输入输出和峰值内存；检查失败路径无资源泄漏。

## 9. 交付与提交

`ops-solver` PR 同时包含五个接口的源码、公开头文件、五组测试、五篇接口文档、README/接口列表更新。
不把任务包的大型生成数据或模拟 DUT 输出提交到源码仓。

个人仓另创建 `task_submission/`：

```text
task_submission/
├── 1 自验证步骤说明.md
├── 2.1 精度自验证报告.xlsx
├── 2.2 精度自验证日志.log
├── 3.1 性能自验证报告.xlsx
├── 3.2 性能自验证日志.log
├── 4.1 内存自验证报告.xlsx
└── 4.2 内存自验证日志.log
```

代码在 950PR 完成全量验证后，从个人 fork 的功能分支向 `cann/ops-solver` 提交一个整体 PR；PR 评论
`compile` 触发门禁。设计文档先从个人竞赛仓分支提交并完成评审。

## 10. 风险与应对

| 风险 | 应对 |
| --- | --- |
| 大矩阵面板串行部分限制性能 | 调整 panel size；融合对角更新与 POTF2；提高 trailing update 的 Cube 占比 |
| batched 指针间接访问降低带宽 | 地址分块预取；矩阵级静态并行；小矩阵 UB 驻留 |
| LOWER/UPPER 路径精度或地址不一致 | 公共索引原语；分别覆盖 padding 和毒化非存储侧测试 |
| 跨核归约导致不确定性 | 单写者 tile 和固定 K 顺序；不使用浮点原子累加 |
| workspace 整数溢出 | Host 端 64-bit checked arithmetic；超出 `int Lwork` 表达范围时报非法值 |
| 缺少本地 950PR/CANN 环境 | 本地只做静态检查；功能、精度、性能结论必须在目标环境补齐，禁止伪造报告 |

## 11. 参考资料

1. 本任务《Atlas 950 单精度实数 Cholesky 分解、求解和批量接口任务书》。
2. NVIDIA cuSolver DN legacy API：POTRF/POTRS/POTRI 及 batched 接口。
3. `cann/ops-solver` 公开头文件、构建工程、handle/stream 实现及现有 Solver 算子。
4. CANN AscendC 算子开发文档与 CATLASS 文档。
5. CANN 生态算子开源精度标准。
6. 任务包 `spotrf/spotrs/spotri/spotrfbatched/spotrsbatched` 的生成与检查说明。
