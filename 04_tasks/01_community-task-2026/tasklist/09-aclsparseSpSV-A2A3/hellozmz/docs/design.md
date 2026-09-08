# 需求背景（required）

## 需求来源

- 任务：9 月社区任务—aclsparseSpSV 算子开发（A2/A3）。
- 官方任务页：[昇腾社区任务详情](https://www.hiascend.com/activities/task-center/details/5248532a9b7f4f16b245183ac0bd3a0e)。
- 任务书：`aclsparseSpSV_A2A3_task_doc.md`，随本材料包放在 `task_book/`。
- 参考实现：cuSPARSE SpSV 语义和 CANN Legacy Sparse API；代码仓为
  `https://gitcode.com/hellozmz/ops-sparse`，实现分支为
  `feature/spsv-a2a3-impl`，代码 MR 为 !174。

## 背景介绍

任务要求通过 ATen Dispatcher/ACL Sparse 接口完成 A2/A3 平台稀疏三角矩阵向量求解
（Sparse Triangular Solve, Vector，简称 SpSV）的 NPU 适配，支持 FP32 和
complex64，并提供可复现的功能、精度、性能和内存证据。本设计文档对应当前
`arch22` 实现；A2/A3 的公共接口和算法边界统一描述，910B2（DAV-2201/arch22）
是本轮实际验证环境。

当前上游仓库没有可直接复用的完整 `aclsparseSpSV` A2/A3 实现。现有实现需要解决：

1. CSR、CSC、COO、SLICED_ELL 等输入格式的统一访问；
2. LOWER/UPPER、UNIT/NON_UNIT 以及 N/T/H 操作的依赖关系；
3. 长依赖链、多 block 调度、重复/未排序坐标和更新矩阵后的确定性；
4. C++ Legacy API 生命周期、workspace、pointer mode 和错误码语义；
5. A2 上有限片上缓存下的 workspace 和并行度平衡。

# 需求分析（required）

## 需求描述

实现 `aclsparseSpSV` C++ Legacy API，使其在 Atlas 800I/T A2（arch22）上完成
`op(A) * Y = alpha * X`，并为 A3 保留兼容的公共接口和架构扩展边界。需要支持：

- FP32（`ACL_FLOAT`）和 complex64（交错实虚布局）；
- CSR、CSC、COO、SLICED_ELL 输入格式，index base 0/1，I32/I64 设备索引；
- `NON_TRANSPOSE`、`TRANSPOSE`、`CONJUGATE_TRANSPOSE`（N/T/H）；
- LOWER/UPPER、UNIT/NON_UNIT，对复数 H 正确执行共轭转置；
- Host/Device pointer mode 的 alpha；X/Y 原地地址；GENERAL/DIAGONAL
  `UpdateMatrix`；
- create、bufferSize、analysis、solve、updateMatrix、destroy 完整生命周期；
- stream 异步下发，求解主路径不回退到 CPU。

## 需求拆解

1. 与公开 Legacy API 原型和参数校验规则保持一致。
2. 在 analysis 阶段规范化稀疏格式并构建依赖 level-set。
3. 在 solve 阶段按 level 前代/回代，level 间使用设备同步。
4. 对重复 COO、未排序坐标、空行、缺失/零非单位对角给出确定语义。
5. 控制 workspace，并验证最大 workspace 不超过任务书允许的设备 L2 路径。
6. 提供 UT/ST、异常、CPU Golden 精度、生命周期、性能和 Profiler 证据。
7. 以 GPU baseline median 为分子、NPU 同范围 `UpdateMatrix + Solve`
   median 为分母计算性能倍率，目标不低于任务书的 `0.25x`。

# 详细设计（required）

## 算子分析

### 数学公式

给定稀疏三角方阵 `A`、向量 `X` 和标量 `alpha`，计算：

```text
op(A) * Y = alpha * X
```

对第 `i` 行，非单位对角的前代/回代为：

```text
Y[i] = (alpha * X[i] - sum(A[i,j] * Y[j], j != i)) / A[i,i]
```

`op(A)` 由 N/T/H 决定；H 对复数矩阵元素先取共轭再转置。三角区域之外的元素
不参与计算。NON_UNIT 的缺失或零对角遵循 IEEE `INF/NAN` 传播语义，UNIT 忽略
对角 values 并使用隐式 1。

### 接口、输入输出和支持范围

公共声明在 `include/cann_ops_sparse.h`，原型与任务书一致：

```cpp
aclsparseStatus_t aclsparseSpSV_createDescr(aclsparseSpSVDescr_t *spsvDescr);
aclsparseStatus_t aclsparseSpSV_destroyDescr(aclsparseSpSVDescr_t spsvDescr);
aclsparseStatus_t aclsparseSpSV_bufferSize(
    aclsparseHandle_t handle, aclsparseOperation_t opA, const void *alpha,
    aclsparseConstSpMatDescr_t matA, aclsparseConstDnVecDescr_t vecX,
    aclsparseDnVecDescr_t vecY, aclDataType computeType,
    aclsparseSpSVAlg_t alg, aclsparseSpSVDescr_t spsvDescr,
    size_t *bufferSize);
aclsparseStatus_t aclsparseSpSV_analysis(
    aclsparseHandle_t handle, aclsparseOperation_t opA, const void *alpha,
    aclsparseConstSpMatDescr_t matA, aclsparseConstDnVecDescr_t vecX,
    aclsparseDnVecDescr_t vecY, aclDataType computeType,
    aclsparseSpSVAlg_t alg, aclsparseSpSVDescr_t spsvDescr,
    void *externalBuffer);
aclsparseStatus_t aclsparseSpSV_solve(
    aclsparseHandle_t handle, aclsparseOperation_t opA, const void *alpha,
    aclsparseConstSpMatDescr_t matA, aclsparseConstDnVecDescr_t vecX,
    aclsparseDnVecDescr_t vecY, aclDataType computeType,
    aclsparseSpSVAlg_t alg, aclsparseSpSVDescr_t spsvDescr);
aclsparseStatus_t aclsparseSpSV_updateMatrix(
    aclsparseHandle_t handle, aclsparseSpSVDescr_t spsvDescr,
    void *newValues, aclsparseSpSVUpdate_t updatePart);
```

| 维度 | 设计支持 |
|---|---|
| 稀疏格式 | CSR、CSC、COO、SLICED_ELL |
| 数据类型 | FP32、complex64 |
| 索引 | Host 元数据 int64；Device I32/I64；base 0/1 |
| 操作 | N、T、H（complex64 H 含共轭） |
| 三角属性 | LOWER/UPPER、UNIT/NON_UNIT |
| alpha | Host pointer mode、Device pointer mode |
| 向量 | 连续 Device values；X/Y 可同址 |
| 更新 | GENERAL 替换全部 values；DIAGONAL 替换对角 values |
| 算法 | `ACL_SPARSE_SPSV_ALG_DEFAULT` |

### 功能和错误语义

Host 侧检查 NULL handle/descriptor、非法枚举、非方阵、向量维度不一致、dtype/
computeType 不一致、索引类型/元数据范围、pointer mode、workspace 对齐和
updatePart。Analysis 会对 CSR/CSC/COO/SLICED_ELL 的 Device 索引元数据按固定
4096 元素分块读取并检查范围、单调性和末端偏移；原始 Device 指针仍不携带分配
长度，Update 数组长度及底层内存有效期不能据此宣称已由 Host 完整校验，调用者
须满足接口约束。
`bufferSize`/`analysis` 允许 vecX/vecY 描述符为 NULL；solve 时要求
有效的 Device values，非法参数返回稳定错误码。Analysis 后 descriptor、结构、
操作属性、索引和 workspace 必须保持一致；改变 values 时使用 UpdateMatrix。

## 算子实现

### Host 侧设计

Host 模块为 `sparse/spsv/arch22/spsv_host.cpp`，共享 descriptor 位于
`sparse/common/aclsparse_spsv_descr.h`。生命周期如下：

```text
createDescr -> bufferSize -> 分配 Device externalBuffer -> analysis
             -> solve -> updateMatrix(GENERAL/DIAGONAL) -> solve -> destroyDescr
```

Analysis 根据 shape、nnz、index base/width、fill/diag、op、dtype 计算 TilingData，
把 CSR/CSC/COO/SLICED_ELL 映射为内部行访问表示，并记录 level offsets、workspace
布局、values 绑定和 pointer mode。下发 Kernel 前执行固定分块索引元数据校验，
随后所有 Kernel 使用调用 handle 的 stream 异步下发。

### 格式规范化和 level-set

COO/未排序坐标按 `(row, column, original_position)` 稳定遍历，重复坐标按固定顺序
累加，从而避免跨 block 竞争写和非确定结果。CSC 在 N/T/H 下转换为行访问，
SLICED_ELL 按切片宽度展开。对目标三角区域建立依赖边：同一 level 内的行互不
依赖，level 之间用 `SyncAll` 分隔；深依赖图在同步成本超过并行收益时收敛到单
block 调度。

### Kernel 侧设计

Kernel 文件为 `sparse/spsv/arch22/spsv_kernel.cpp/.h`，执行三个阶段：

1. **Init/Load**：读取 TilingData、workspace、alpha 和输入向量地址；Device pointer
   mode 从 Device scalar 取 alpha。
2. **Compute**：每个 level 内由 AIV block 处理互不依赖的行，执行标量/复数乘加、
   前代或回代；complex64 H 在读取 A 时完成转置和共轭。
3. **Store**：把 Y 写回 Device；X/Y 同址时先读取当前行所需 X，再写 Y。

DAV-2201 标量 GlobalTensor 访问按 512B cache-line 归属将同一 cache line 的
标量写分配给唯一 block，修复多 block 随机缺行问题。Kernel 只通过 Device
Global Memory 访问数据，不依赖 CPU fallback。

### Tiling、workspace 和内存

TilingData 至少包含矩阵维度/nnz、索引类型、index base、fill/diag/op、dtype、
level 数与 offsets、workspace 偏移和 block 调度参数。Workspace 用于规范化索引、
level-set、临时 values/向量和同步状态，不复制同规模 Host 中间结果。

910B2 六个 P case 的最大 workspace 为 `135,267,008 B`，设备 L2 为
`201,326,592 B`，占比约 67.19%，满足任务书允许的 workspace≤L2 路径。输入输出
超过 500 MB 的 GPU extra-peak 路径属于本轮用户明确延期项，不将其写成已通过。

### UpdateMatrix

- `GENERAL` 替换全部 A values，刷新受影响的规范化 values、对角和 tiling 状态；
- `DIAGONAL` 把 m 个对角 values 异步拷贝到专用 workspace；Solve 覆盖已存储的
  非单位对角值，H 模式在读取时共轭，A.values 保持只读。GENERAL 或重新 Analysis
  清除该覆盖状态；
- `CONJUGATE_TRANSPOSE` 的显式转置 workspace 更新时对 complex64 新值执行
  共轭；CSC-as-CSR(A^T) 路径保留原始 values，由 Solve 统一共轭，
  避免 base1 双重共轭；
- 更新在同一 handle stream 异步执行，完成后复用 descriptor 进行 Solve。

## 支持硬件

| 支持的芯片版本 | 当前状态 | 说明 |
|---|---|---|
| Atlas 800I/T A2（910B2，DAV-2201/arch22） | 已验证 | 物理 2 号卡，CANN 9.1.0-beta.1 |
| Atlas A2（910B3/910B4）、Atlas A3 | 实机延期 | 按当前用户范围暂不启动跨型号矩阵 |

## 算子约束限制

1. 本轮实际证据仅覆盖 910B2/arch22；不能把它外推为 910B3、910B4 或 A3 已验证。
2. `ACL_SPARSE_SPSV_ALG_DEFAULT` 是当前实现支持的算法枚举。
3. X/Y 需要连续 Device values；Host 侧仅保存描述符和标量元数据。
4. Analysis 到 Solve/Update 完成前，descriptor、结构属性、workspace 和 stream
   生命周期必须有效；改变结构需重新 Analysis。
5. Python hook `spsv_analysis_npu`、`spsv_update_npu`、`spsv_npu` 不在当前镜像中
   注册，Python runner 不作为本轮已完成项；C++ ACL NPU 路径是真实验证入口。
6. GPU baseline 重采集和输入输出超过 500 MB 的 extra-peak 对比按用户要求延期；
   当前倍率使用任务书已有 baseline 区间计算。

# 可维可测分析

## 可维护性设计

公共接口、共享 descriptor、arch22 Host、Kernel、TilingData 和测试目录分层，后续
架构只需替换 `sparse/spsv/<arch>/` 与构建选择，不改 Legacy API。Host 校验和
规范化逻辑集中，Kernel 仅依赖稳定的 TilingData/workspace 契约；每轮修改记录在
`implementation/history/round3.md`～`round13.md`。

## 可测试性设计

构建和核心验证命令：

```bash
bash build.sh --ops=spsv --soc=ascend910b
ctest --test-dir build -R spsv
```

真实 910B2 证据位于 `evidence/`：165/165 全量 GTest（146 功能 + 19 异常）、
FP32 12/12、complex64 扩展回归通过、生命周期 31/31、70/20/10 分布匹配率 1.000，
以及六个官方规模计时输出全部有限且逐分量 100% 匹配，最大误差 6.11e-8。性能每组
10 次预热+30 次采样。Profiler 数据库为
`evidence/profiler/msprof_20260903072802.db`，含
`spsv_analysis_kernel`、`spsv_solve_kernel`、AI_VECTOR_CORE 和 stream/task 记录。

## 精度标准/性能标准

| 验收标准 | 任务要求 | 当前实测 |
|---|---|---|
| 精度 | FP32/complex64 结果不低于参考精度，覆盖任务书功能矩阵 | FP32 12/12、complex64 14/14；匹配率 1.000；最大绝对误差分别 `5.54e-8`、`7.03e-7` |
| 功能/生命周期 | 参数、异常、Update 和 descriptor 生命周期正确 | 全量 GTest 165/165；生命周期 31/31 |
| 性能 | GPU baseline median / NPU (Update+Solve) median ≥ `0.25x` | 六个 P case 为 0.639x～2.339x |
| 内存 | 允许 workspace≤L2 路径；GPU extra-peak 为另一条路径 | 最大 workspace 135,267,008B / L2 201,326,592B；extra-peak 延期 |

性能明细（10 次预热、30 次采样）。Update 和 Solve 在同一 caller stream
的连续 ACL Event 区间内计时，倍率使用 Total median：

| Case | dtype | m/nnz | Analysis ms | Update median ms | Solve median ms | Total median / P90 ms | workspace B | 倍率 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| P-01 base0 | FP32 | 65,536 / 524,288 | 232.883 | 0.004 | 219.847 | 219.850 / 219.869 | 1,573,440 | 2.339x |
| P-01 base1 | FP32 | 65,536 / 524,288 | 739.095 | 143.805 | 219.448 | 363.230 / 363.870 | 8,127,104 | 1.415x |
| P-02 base0 | complex64 | 131,072 / 1,572,864 | 2,331.271 | 670.931 | 1,046.625 | 1,717.948 / 1,720.184 | 29,360,768 | 0.716x |
| P-02 base1 | complex64 | 131,072 / 1,572,864 | 3,504.281 | 670.896 | 1,046.474 | 1,717.098 / 1,720.114 | 55,050,944 | 0.715x |
| P-03 base0 | complex64 | 262,144 / 3,932,160 | 5,632.604 | 2,169.440 | 2,644.881 | 4,813.967 / 4,821.547 | 71,303,808 | 0.639x |
| P-03 base1 | complex64 | 262,144 / 3,932,160 | 8,462.561 | 2,167.754 | 2,643.495 | 4,812.716 / 4,819.261 | 135,267,008 | 0.640x |

倍率按原始微秒值计算后四舍五入到三位小数。最终源码 `32cd9fa` 已重跑
GTest 和生命周期；性能及扩展复数日志对应 `31361c5`。后续仅增加 Analysis
字节/地址溢出保护和空矩阵索引检查，Update/Solve Kernel 与性能驱动未改动。
完整日志来源记录于 `evidence/contract/README.md`，不将历史采样冒充最终库重采。

## 兼容性分析

- API：采用公开 `aclsparseSpSV_*` Legacy API 命名、参数和生命周期，保留与
  cuSPARSE SpSV 对齐的 N/T/H、fill/diag、pointer mode 和 UpdateMatrix 语义。
- 软件：实测 CANN 9.1.0-beta.1；构建脚本通过 `--soc=ascend910b` 选择 arch22。
- 硬件：当前真实结果对应 Atlas 910B2/DAV-2201；A3 由同一接口预留，但未将
  910B2 结果冒充 A3 结果。
- 数据：FP32 使用 FP64 Golden，complex64 使用 complex128 Golden；重复 COO、
  未排序、空行、缺失/零对角、原地 X/Y 和 Host/Device alpha 均有专项回归。

## 风险和未闭环项

计时输入在预热前按稳定已知解固定，计时后直接核验输出，不换输入、不额外 Solve。
GPU 标杆采用任务包已有结果，只作同规模/同调用范围对照，不声称数值逐 bit 一致。

此前驱动属性、计时范围和 H 更新共轭问题已修复。后续又修复 Analysis 绑定、
失败调用状态污染和 DIAGONAL 更新改写输入的问题。旧随机计时输入出现大范围
NaN，因此新版在预热前准备有限数值已知解输入，计时后不换输入直接验算输出。
旧数据全部归档，不再作为当前验收结论。

Device 索引元数据已在 Analysis 阶段按固定 4096 元素分块完成 Host 范围扫描；
裸指针分配长度、同址内容篡改和提前释放不能由当前状态绑定检查完整识别。外部范围仍包括 910B3/910B4/A3、
GPU extra-peak、Python hook 注册、设计评审/合入和门户附件替换。

## 设计文档修订记录

| 版本 | 日期 | 修改内容 |
|---|---|---|
| v1.1 | 2026-09-03 | 按官方模板补齐需求背景、需求分析、详细设计和可维可测章节；纳入 arch22 实现、146/146 回归、精度/性能/Profiler 实测结果 |
| v1.2 | 2026-09-04 | 统一 Update+Solve 计时范围；修正 fill/diag 驱动与 H 更新共轭缺陷；刷新 165/165、六个官方规模正确性及 0.646x～2.366x 实测性能 |
| v1.3 | 2026-09-04 | 修复 Analysis 绑定与 DIAGONAL 输入只读；生命周期 29/29、扩展复数回归通过；计时输出直接验算，六组全有限且 100% 匹配，倍率 0.639x～2.341x；明确尚未覆盖的校验边界 |
| v1.4 | 2026-09-04 | 增加四种格式 Device 索引元数据固定分块范围校验；生命周期 30/30、165/165 和复数双模式重新通过；刷新六组性能为 0.639x～2.339x |
| v1.5 | 2026-09-04 | 强制 CSR/CSC 首项偏移等于 index base；最终提交 `32cd9fa`，GTest 165/165、生命周期 31/31 重新通过 |
