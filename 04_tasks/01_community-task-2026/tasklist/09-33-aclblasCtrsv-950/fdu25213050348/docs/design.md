# aclblasCtrsv 950 设计文档

## 需求背景（required）

来源为 9 月社区任务 aclblasCtrsv 算子开发（950），目标平台为 Ascend 950PR / CANN 9.1.0。
任务编号：`09-33-aclblasCtrsv-950`。开发者及个人目录：`fdu25213050348`。
基准仓库提交：`fe54d86f00a4d449f55da1e8d144b898953d97c5`。

## 需求分析（required）

求解 \(\operatorname{op}(A)x=b\)，其中 \(\operatorname{op}(A)\in\{A,A^T,A^H\}\)。输入 x 存放 b，输出原地覆盖为解。
A 为列主序，lda >= max(1,n)。向量物理长度为 \(1+(n-1)|incx|\)，incx 可正可负但不为零。

```cpp
aclblasStatus_t aclblasCtrsv(
    aclblasHandle_t handle, aclblasFillMode_t uplo, aclblasOperation_t trans,
    aclblasDiagType_t diag, int n, const aclblasComplex* A, int lda,
    aclblasComplex* x, int incx);
```

覆盖 12 组枚举组合。使用公共 `aclblasComplex` 类型，只引用指定三角，UNIT 不读对角。
矩阵只读，向量步长间隙不写。不做奇异性或近奇异性检测，不涉及广播。

## 详细设计（required）

### Host

公共声明加入 `include/cann_ops_blas.h`，实现位于 `blas/trsv/arch35/`。
空 handle 优先返回 HANDLE_IS_NULLPTR；负 n 返回 INVALID_VALUE；有效 handle 且 n=0 时沿用同族 quick-return 顺序直接成功。
n>0 时检查枚举、lda、incx、A/x 非空。通过 handle 的 stream 异步下发，不在 host 同步或读回输入。

### 分块求解

按 32 阶对角块推进。有效下三角前代，上三角回代，转置后翻转方向。
每块由单个 warp 求解，每个 lane 持有一行，将需要的系数预加载到寄存器。
逐 pivot 计算解分量，通过 shuffle 发布，然后更新未求解行。循环静态展开，减少 GM 访问。
UNIT 不加载对角、不做除法。非单位对角使用缩放复数除法，降低模平方溢出风险。
共轭转置在加载时对虚部取负。

### 多核更新

每个对角块完成后，使用 32 个 AIV block、每 block 128 个线程更新剩余向量。
非转置按列连续读取；转置/共轭转置由 warp 计算一个逻辑行并归约。
同 stream 的 kernel 顺序保证块间依赖，不使用跨 block 自旋屏障，不申请额外 GM workspace。
尾块由 n 保护，偏移采用 int64_t，负步长子块重新定位到其物理存储起点。

### 兼容性与限制

新增公共 API 与 arch35 实现，保留 Strsv。无 CUDA 类型或运行时依赖。
不要求 bit-exact，分块与并行归约顺序不同于串行 CBLAS。
调用方保证非单位对角元素非零；算子不对系统条件数作出保证。

| 适配硬件 | 软件环境 |
| --- | --- |
| Ascend 950PR | CANN 9.1.0 |

## 可维可测分析

### 精度

Netlib `cblas_ctrsv` 作为 golden。实部、虚部分别检查 rtol=2^-10、atol=2^-16、匹配比例至少 99%，误差不超过 max(1e-2, 32 ULP)。
NaN/Inf 分类独立检查，不能被匹配比例掩盖；相同 NaN 不代表有意义的有限解。

以任务配套 CSV 为基础设计测试，覆盖全部枚举组合、尺寸扫描、lda padding、正负步长、特殊值及非法参数。
补充分块边界及 4097 阶用例，以及空句柄、零维、负维度和 n=1/INT_MIN 步长检查。
计划在未引用三角与 UNIT 对角填充 NaN，验证这些位置不参与计算；矩阵整块回读检查只读性，向量边界和间隙使用哨兵检查越界写入。

随机输入按任务要求覆盖均匀分布与正态分布，使用固定 seed 保证可复现；正态均值在 [-5,5]、标准差在 [0.1,2]，实部/虚部独立采样。
NON_UNIT 对角按符号添加 max(5,n) 偏移，golden 与 NPU 使用完全相同的输入。

### 性能

ACL event 包围完整调用，预热 10 次、正式 60 次。每次恢复 RHS，恢复和传输不计入设备时间。
不使用 GTest 总耗时。任务书 5 项阈值直接比较，不重复折算倍率。

| n | uplo | trans | diag | incx | 上限（us） |
| --- | --- | --- | --- | --- | --- |
| 512 | UPPER | N | NON_UNIT | 1 | 146.33 |
| 1024 | LOWER | N | NON_UNIT | 1 | 272.53 |
| 2048 | UPPER | T | NON_UNIT | 1 | 526.22 |
| 4096 | LOWER | C | NON_UNIT | 1 | 1177.84 |
| 4096 | UPPER | N | UNIT | 1 | 1030.61 |

### 性能优化策略与内存验证

对角块大小和更新阶段线程配置作为调优参数，结合寄存器占用、矩阵访问效率和 kernel 下发开销选择。
使用 profiling 分别分析对角求解与剩余向量更新，优先减少串行依赖路径的内存访问和小 kernel 调度开销。
验证时记录设备缓冲区分配量及运行峰值内存，区分输入输出存储、算子临时空间与测试框架开销。
