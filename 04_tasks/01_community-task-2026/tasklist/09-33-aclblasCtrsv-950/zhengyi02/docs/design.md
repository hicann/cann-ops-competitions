# aclblasCtrsv（Ascend 950PR）算子设计

提交人：`zhengyi02`；社区任务：`09-33-aclblasCtrsv-950`。

本次提交用于设计方案评审，内容依据已实现版本整理；实现与自测证据保存在[个人工程仓库](https://gitcode.com/zhengyi02/aclblasCtrsv)。

## 需求背景（required）

### 需求来源

本工程实现社区任务《aclblasCtrsv 950 算子开发任务书》。正式接口为 `aclblasCtrsv`，
基于 ops-blas 主仓快照 `fe54d86f00a4d449f55da1e8d144b898953d97c5` 开发。
任务要求见[任务书](https://gitcode.com/zhengyi02/aclblasCtrsv/blob/8a83fb645015b23bf84f0ae4e2ee91cf3dfc31a3/aclblasCtrsv_Atlas950PR_task_doc.md)，原始 1200 条 CSV 见[原始用例](https://gitcode.com/zhengyi02/aclblasCtrsv/blob/8a83fb645015b23bf84f0ae4e2ee91cf3dfc31a3/test_cases/ctrsv_test.csv)。
本设计沿用社区 design_template.md 的需求背景、需求分析、详细设计、可维可测分析结构。

### 背景介绍

TRSV 求解一个右端项的三角线性系统 `op(A) x = b`，结果原地覆盖 x。
复数转置和共轭转置具有不同语义，不能直接复用实数 Strsv 的转置分支。
逐主元依赖是主要性能限制：不能把每一行当成互不相关的任务同时求解。

## 需求分析（required）

| 要求 | 实现方式 |
|---|---|
| complex64 | 复用 `cann_ops_blas_common.h` 的 `aclblasComplex {float real, imag;}` |
| 12 种枚举组合 | upper/lower × N/T/C × unit/non-unit 模板分派 |
| 列主序 | A(i,j) 位于 `i + int64_t(j)*lda` |
| incx 正负 | 负步长首逻辑元素位于 `(n-1)*(-int64_t(incx))` |
| 单位对角 | 编译期 UNIT 分支，不读取主对角 |
| 不读取另一三角 | 所有矩阵加载仅覆盖实际依赖区 |
| 空矩阵 | 校验 handle 和标量后 n=0 返回成功，不检查数据指针、不发核 |
| 非法参数 | handle 空返回 HANDLE_IS_NULLPTR；其他无效参数返回 INVALID_VALUE |
| 非奇异保证 | 由调用方提供，不检测奇异性，不正则化用户矩阵 |
| 异步执行 | 所有核提交至 handle 绑定的 stream，无内部同步或额外设备分配 |

## 详细设计（required）

### 输入输出与约束

| 参数 | 类型/形状 | 约束与语义 |
|---|---|---|
| A | complex64，逻辑 n×n，列主序存储 | lda≥max(1,n)，存储跨度 lda×n；仅访问所选三角，UNIT 不读取对角 |
| x | complex64，逻辑长度 n | 输入 b，原地写回解；n>0 时至少 1+(n-1)×abs(incx) 个元素 |
| n / lda / incx | int | n≥0，incx≠0；偏移运算提升到 int64_t |
| uplo / trans / diag | BLAS 枚举 | UPPER/LOWER、N/T/C、UNIT/NON_UNIT |
| handle | aclblasHandle | 有效句柄，使用关联 stream 异步提交 |

不支持 batch 或广播。调用者保证有效指针与存储容量、非奇异矩阵以及异步任务完成前输入输出内存有效。

### 数学公式与逻辑顺序

令 `forward = (uplo == UPPER) == (trans != N)`。
在统一的求解顺序中，逻辑位置 s 对应原矩阵行 `s`（forward）或 `n-1-s`（backward）。
该顺序下方程总是下三角：

```
x[j] = (b[j] - sum(k<j, op(A)[j,k] * x[k])) / op(A)[j,j]
```

UNIT 分支省略除法。共轭转置只对加载的矩阵虚部取负。
复数乘法使用 FP32 分量，保留逐步舍入并恢复 Inf 引起的双 NaN 乘积。常规除法先缩放分母；极大/极小有限分母使用位级二进制缩放、双 float 补偿乘积和商修正，避免直接平方溢出或近似缩放损失次正规数。
无穷大分子采用显式分类处理，使对应测试的特殊值行为与 Netlib 一致。
该实现不宣称所有浮点输入、所有病态系统均可达到有限误差阈值。

### Host 侧设计

1. 按 handle、枚举/维度/步长、n=0、非空指针的顺序校验。
2. 填充 `CtrsvTilingData`，仅携带 n、lda、incx、uplo、trans、diag。
3. 通过内部 handle 的 stream 调用核启动封装；禁止同步取回计算结果。
4. 公共声明放入 `include/cann_ops_blas.h`，不存在产品私有平行 API。

### Kernel 侧设计

**小规模（n<128）**：单 AIV 上执行 SIMT。向量缓存在 UB。
N 模式按列更新未解元素；T/C 模式按 Netlib 顺序逐项扣除复数乘积，避免改变舍入顺序。
原生向量路径与 SIMT 路径分别保留独立的乘法、复数加减和 RHS 更新舍入。

**分块路径（n≥128）**：以 64 为对角分块大小，在同一个 stream 顺序提交两类核。

1. PrepareNative 使用 512 个 SIMT 线程把三角块拆为实部/虚部平面，并预计算常规对角倒数。NativeSolve 用原生 64 lane 向量完成块内回代，StoreNative 写回本块解。非有限矩阵及不适合倒数路径的非单位对角输入使用 SolvePanel 回退。
2. 多个 AIV 更新剩余 RHS。NON_UNIT 保留 UpdatePanel 分组累加；UNIT 使用 PrepareUpdate + OrderedUpdate，逐主元扣除乘积，保持 CPU 舍入顺序。非有限矩阵使用通用回退。
3. 重复处理下一块，最后不足 64 的尾块使用实际 count。

NON_UNIT 更新中，N 模式相邻线程读取相邻矩阵行，T/C 模式一个 warp 读取连续矩阵列。UNIT 更新将 64×64 子块装入 UB，再按主元顺序执行向量乘减。
每个更新块只写自己的输出行。stream 顺序保证下一块求解看到已完成的更新，
不存在跨核自旋等待、跨核全局锁或未受保证的调度依赖。
同一 panel kernel 内所有 SIMT VF 使用相同的 512 线程宽度；混合宽度导致的实测回归已有对角矩阵补充用例覆盖。

为避免设备地址空间隐式结构体复制的编译问题，GM/UB 的复数读写使用分量封装。
地址乘法、负 incx 的取反首先提升到 int64_t，覆盖 `incx=INT_MIN, n=1`。

### 空间与复杂度

计算复杂度 O(n²)，不生成转置副本，不分配 n×n 的辅助矩阵。
原生路径显式 UB 数组约 35 KiB，回退路径另有约 33.25 KiB 数组，编译器可能复用存储；NON_UNIT 更新核显式数组约 4.5 KiB，UNIT 顺序向量更新复用约 35 KiB 的原生缓冲区。这些是源码数组大小，不包含编译器额外开销。
算子新增 GM workspace 为 0；ops-blas handle 自带的默认 workspace 属于库公共开销。
测试报告的 host/device buffer 字节是测试显式分配量，不冒充设备进程峰值。

### 支持硬件

| 产品 | 实现支持 | 验收说明 |
|---|---|---|
| Ascend 950PR | 支持 | CANN 9.1.0；功能和性能实测结论分别见自测报告 |
| 其他产品 | 未实现 | 不向 arch22 注入占位实现 |

### 约束与性能风险

API 不新增任务书以外的 n 上限。分块启动次数约 `2*ceil(n/64)-1`，
小块回代依赖和多次核启动仍限制扩展性，当前五项性能结果见自测报告。
测试覆盖范围不等于已验证全部 int 维度；调用者仍需提供足够设备内存。
当前性能是否达标以[归档设备事件测量](https://gitcode.com/zhengyi02/aclblasCtrsv/blob/8a83fb645015b23bf84f0ae4e2ee91cf3dfc31a3/docs/ctrsv/evidence/revised-original/results.csv)为准，不以功能 PASS 替代。

## 可维可测分析

### 精度标准/性能标准

#### 精度标准

CBLAS/Netlib 为唯一数值 golden，使用相同物理内存布局和负步长。
实部和虚部分别统计：`|actual-golden| <= 2^-16 + 2^-10*|golden|`，匹配比例至少 0.99。
每个有限分量还需满足绝对误差 ≤ 1e-2 或 ≤ 32 ULP；NaN 与 NaN、同号 Inf 视为对应特殊值，
特殊值分类不匹配会失败。MERE、MARE 作为补充诊断输出，主判据为任务书双阈值。

unused triangle、lda padding、UNIT diagonal 填入 NaN；x 前后守卫和步长间隙必须原样保留；
A 全部字节在调用后保持不变。该测试验证语义与越界写，不将 NaN poisoning 夸大为访存审计器。
设备地址偏移一个 complex64，覆盖不满足 32 字节对齐的有效接口输入。

### 数据治理

原始用例不覆盖；工程 CSV 显式增加 `matrix_policy` 字段，调整清单见[测试数据调整记录](https://gitcode.com/zhengyi02/aclblasCtrsv/blob/8a83fb645015b23bf84f0ae4e2ee91cf3dfc31a3/docs/ctrsv/test_adjustments.md)。
测试工程以种子奇偶交替均匀/正态采样，实部虚部独立抽样。
NON_UNIT 对角加符号保持的 `max(5,n)` 偏移。工程主集不再对 UNIT 缩放，也不替换密集极值矩阵；补充集的策略单独标注。
算子不会修改用户矩阵或隐式改变对角。

#### 性能标准

10 次 warmup，60 次有效采样；每次在 event start 之前恢复初始 b，防止反复对解向量求解。
ACL event 仅包围 API 发起的设备工作，排除 golden、数据生成、H2D 重置和 D2H 验证。
这是完整 API 设备时间，包含该 API 的多核启动与可能的提交间隙，不称为某单个 kernel 的时间。
五个任务书上限分别是 146.33/272.53/526.22/1177.84/1030.61 µs，直接比较，不再除以 0.4。
其余 195 条基线未提供，标记 NO_REF，仅报告耗时。

### 当前验证结果与待确认项

以下数据来源于已归档的[自测报告](https://gitcode.com/zhengyi02/aclblasCtrsv/blob/8a83fb645015b23bf84f0ae4e2ee91cf3dfc31a3/docs/ctrsv/self_test_report.md)，测试源码提交为 `df4d929d99cdba8fe818f2b7dc242ecac35c8baa`。
工程主集 1202/1202、补充集 175/175 通过；原始 CSV 1201/1202 通过。
原始 TC_ED_095 的描述为 lda=0，但实际字段为 n=1、lda=1，仍预期 INVALID_VALUE；工程主集只将该行 lda 修正为 0，该差异须由社区确认。

| 原始性能用例 | 实测 µs | 任务上限 µs |
|---|---:|---:|
| TC_PF_1001 | 72.49 | 146.33 |
| TC_PF_1002 | 146.94 | 272.53 |
| TC_PF_1003 | 384.54 | 526.22 |
| TC_PF_1004 | 919.78 | 1177.84 |
| TC_PF_1005 | 812.38 | 1030.61 |

五项已提供基线的性能用例通过，其余 195 项没有基线，不能据此宣称全部性能验收完成。
随机输入采用种子奇偶交替均匀/正态分布，不保证整个数据集严格各占 50%；当前特殊值比较以 Netlib 为依据，尚未提供 cuBLAS 特殊值行为的交叉验证证据。
社区设计评审、完整验收材料、上游 CI 与代码 PR 仍需独立完成。

### 兼容性、复现与交付

新增 API，不修改既有 Strsv 实现。源文件由上游构建规则自动收集，
`test/trsv/ctrsv/CMakeLists.txt` 使用 `ops_blas_add_gtest_tests` 注册正式测试。
复现步骤见[测试 README](https://gitcode.com/zhengyi02/aclblasCtrsv/blob/8a83fb645015b23bf84f0ae4e2ee91cf3dfc31a3/test/trsv/ctrsv/README.md)和[自测报告](https://gitcode.com/zhengyi02/aclblasCtrsv/blob/8a83fb645015b23bf84f0ae4e2ee91cf3dfc31a3/docs/ctrsv/self_test_report.md)。
CI/社区评审、邀请 Ascend-CANN、向外部仓库提交 PR 不属于本地测试通过即可自动完成的状态。

参考：
- https://www.netlib.org/blas/ctrsv.f
- https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-trsv
- https://gitcode.com/cann/ops-blas
- https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md
