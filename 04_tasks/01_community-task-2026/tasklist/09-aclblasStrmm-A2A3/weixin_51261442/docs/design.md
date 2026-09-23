# 需求背景

## 需求来源

面向 Atlas A2/A3 产品，使用 Ascend C 开发单精度实数三角矩阵乘算子 aclblasStrmm。

## 背景介绍

### aclblasStrmm 算子实现优化

Strmm 是 BLAS Level 3 单精度实数三角矩阵乘，执行三角矩阵与一般矩阵的乘法。设计基于 Ascend C，采用 ops-blas 的 handle/stream 直调模式，代码归属 `blas/trmm/arch22/`，测试归属 `test/trmm/strmm/arch22/`。

接口沿用 ops-blas 的 `aclblasStrmm` 声明。转置模式仅接受 N/T，传入 `ACLBLAS_OP_C` 返回非法参数。

### 接口参数

| 参数 | 参数含义 | 数据类型 | 支持数据类型 | 约束 | 形状 |
| --- | --- | --- | --- | --- | --- |
| A | 输入三角矩阵 | tensor | float32 | 只引用指定三角；单位对角不读取原对角值 | LEFT：m×m；RIGHT：n×n |
| B | 输入一般矩阵 | tensor | float32 | 列主序，ldb≥max(1,m) | m×n |
| C | 输出一般矩阵 | tensor | float32 | 列主序，ldc≥max(1,m)；允许 B=C | m×n |
| alpha | 主机标量指针 | scalar | float32 | 不可为空；值为 0 时不引用 A/B | 标量 |
| side/uplo/trans/diag | 模式选择 | attr | 枚举 | LEFT/RIGHT、UPPER/LOWER、N/T、UNIT/NON_UNIT | 标量 |
| m/n、lda/ldb/ldc | 尺寸及前导维度 | scalar | int | 非负尺寸；lda≥max(1,K)，ldb/ldc≥max(1,m) | 标量 |

### aclblasStrmm 算子功能分析

输入是 A、B、alpha 及矩阵描述参数，输出是完整的一般矩阵 C，不是三角矩阵。只支持 float32，无广播和批处理。矩阵物理寻址采用 `X[col*ld + row]`，前导维度可包含填充。

# 需求分析

## 需求描述

```cpp
aclblasStatus_t aclblasStrmm(
    aclblasHandle_t handle, aclblasSideMode_t side,
    aclblasFillMode_t uplo, aclblasOperation_t trans,
    aclblasDiagType_t diag, int m, int n, const float* alpha,
    const float* A, int lda, const float* B, int ldb,
    float* C, int ldc);
```

有效输出满足 `C=alpha*op(A)*B`（LEFT）或 `C=alpha*B*op(A)`（RIGHT）。支持合法运行时尺寸和前导维度。

## 需求拆解

1. 完成枚举、尺寸、前导维度、句柄和指针校验，保持 ops-blas 错误码约定。
2. 覆盖左右乘、上下三角、转置和单位对角的 16 种组合。
3. 支持零维快速返回、alpha=0、B=C、非对齐尺寸和列间填充；不改写输出有效区域之外的内容。
4. 小矩阵减少启动和搬运成本；中大矩阵使用 Cube 执行 FP32 乘加，并逐步消除三角零区域的无效计算。
5. 覆盖功能、精度、边界和性能用例，保证测试结果可复现。
6. 适配 Atlas A2/A3，软件环境使用 CANN 9.1.0。

# 详细设计

## 算子分析

### 数学公式

令 K=m（LEFT）或 K=n（RIGHT）。先定义逻辑三角矩阵 T：指定三角区域取 A，区域外为 0；diag=UNIT 时主对角为 1，且不读取 A 的主对角。

令 P=T（trans=N）或 P=Tᵀ（trans=T）：

- LEFT：`C[i,j] = alpha * Σ(k=0..m-1) P[i,k] * B[k,j]`。
- RIGHT：`C[i,j] = alpha * Σ(k=0..n-1) B[i,k] * P[k,j]`。

计算时跳过不属于三角矩阵的项，避免无效项与 Inf/NaN 相乘影响结果。

### 支持数据类型

输入、输出和累加均为 float32。Cube 路径使用原生 FP32 乘加，完成点积后乘 alpha。

### 支持形状

A 为 K×K，B/C 为 m×n；m、n≥0。`lda≥max(1,K)`，`ldb、ldc≥max(1,m)`。支持小尺寸、矩形、非对齐尺寸及带 padding 的列主序矩阵，不支持超出前导维度语义的任意非连续视图。

## 算子实现

### 实现方案

#### 3.2.1 host侧设计：

主机侧负责校验、分支选择、workspace 管理和同一 stream 上的有序启动，不搬回矩阵执行 CPU 计算。

参数校验顺序：

1. handle 为空返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`。
2. 非法枚举、负尺寸、alpha 为空或非法前导维度返回 `ACLBLAS_STATUS_INVALID_VALUE`。
3. m=0 或 n=0 返回成功，不启动计算。
4. 非空输出要求 C 非空；alpha≠0 时要求 A/B 非空。
5. alpha=0 进入专用清零路径，只覆盖 C 的 m×n 有效元素，不读取 A/B。

B=C 时先使用独立结果 workspace，所有输入读取完成后再回写 C，避免多个核心覆盖尚未读取的 B。除 B=C 外不支持 C 与 A/B 重叠。alpha 为主机标量指针。

##### 1. 分核策略：

| 路径 | 选择条件 | 工作划分 |
| --- | --- | --- |
| 零维 | m=0 或 n=0 | 不启动设备计算 |
| 清零 | alpha=0 且输出非空 | 按输出连续片段分配 Vector 核 |
| 小矩阵 | m≤64 且 n≤64 | 按输出列分配 Vector 核 |
| 通用矩阵 | 其他合法尺寸 | 按输出二维分块分配 Cube 核，并裁剪有效 K 范围 |

核心数由平台能力和有效任务数共同决定。三角矩阵各输出块的有效计算量不同，按有效 K 长度估算工作量并均衡分配。

##### 2. 数据分块和内存优化策略：

小矩阵以 64 为局部存储行跨度，使用 Vector Gather、乘法和规约完成点积。根据实际尺寸缩小初始化范围，减少小用例的固定开销。

通用路径以 32×32 作为 A 预处理的基础分块，以 64×64×64 作为 Cube 基础计算块。A 工作区前导维度 `LA=align_up(K,16)`；中间输出前导维度 `LC=align_up(m,8)`。需要同时保留两个工作区时，空间上界为 `4*LA*LA + 4*LC*n` 字节。申请前检查尺寸乘法溢出，并按路径分配所需空间。

若 C 的前导维度满足输出路径要求且 B≠C，可直接写 C；否则写入临时结果，再只拷贝有效 m×n 元素。

Cube 的基础缓存预算为 A1/B1 各 16 KiB、L0A/L0B 各 16 KiB、L0C 16 KiB。分块大小和缓冲数量结合平台容量选择，通过数据复用与双缓冲减少搬运等待。

##### 3. tilingkey规划策略：

本接口采用 kernel 直调，无图模式算子注册的 tilingkey。host 使用参数结构和不同启动入口区分小矩阵、清零、A 预处理、Cube 计算和结果回写。传递字段包括 m/n、lda/ldb/ldc、side/uplo/trans/diag、alpha、工作区跨度和分块信息。

分支依据尺寸、布局和模式选择，覆盖相同条件下的任意合法输入。

#### 3.2.2 kernel侧设计：

**小矩阵路径。** 搬入所需数据，在局部空间形成点积排列，按有效三角范围计算，对输出列执行乘法与规约，最后乘 alpha 并写回。读取时按有效片段搬运，跳过无效三角和 UNIT 模式的对角元素。

**通用路径。** 列主序 C 在物理上等价于行主序 Cᵀ，因此将输出逻辑尺寸交换为 n×m，并交换乘法左右顺序。LEFT 对应 `Cᵀ=Bᵀ*Pᵀ`；RIGHT 对应 `Cᵀ=Pᵀ*Bᵀ`。这只是索引转换，不修改外部接口语义。

执行流程为 A 预处理 → Cube 矩阵乘 → alpha 缩放与结果写回。预处理完成三角语义和 N/T 转换；Cube 通过 GM→L1→L0A/L0B 搬运，FP32 乘加到 L0C，再经 Fixpipe 写结果；跨阶段使用同一 stream 顺序保证依赖。

**三角有效区裁剪。** 用转置后的 P 上下三角属性计算有效 K：

| 模式 | 有效范围 |
| --- | --- |
| LEFT、P 上三角 | k≥i |
| LEFT、P 下三角 | k≤i |
| RIGHT、P 上三角 | k≤j |
| RIGHT、P 下三角 | k≥j |

远离对角的完整块直接跳过零区域；对角交叉块按行/列裁剪，保证 Inf/NaN 场景下无效项不参与计算。

**同步与尾块。** 缓存复用前等待此前消费者结束，分别建立 MTE2→Vector、Vector→MTE3、MTE2→MTE1、MTE1→Cube、Cube→Fixpipe 的依赖。跨迭代复用输入缓存时同样需要反向释放依赖。尾块搬运长度以有效行列数计算；内部 padding 初始化为零，不越界读取输入或覆盖输出 padding。

**性能优化。** 小矩阵减少启动、初始化和搬运开销；大矩阵通过三角裁剪、数据复用、分核均衡及流水重叠降低耗时。对优化前后版本使用相同输入和采样方式进行比较，并检查相邻尺寸与不同模式。

## 支持硬件

| 适配产品 | 设计范围 |
| --- | --- |
| Atlas 800I/T A2 | arch22 |
| Atlas A3 系列 | arch22 |

软件环境：CANN 9.1.0。

## 算子约束限制

- 仅支持 float32、列主序、单矩阵输入；无广播、无批处理。
- trans 仅接受 N/T；所有非法枚举返回非法参数。
- UNIT 对角及无效三角不读取；alpha=0 不读取 A/B。
- B=C 允许，其余输出重叠不支持；padding 保持不变。
- 遵循 handle 绑定 stream 的异步执行方式，调用方读回前同步。
- 不要求逐位确定性，输出满足 float32 精度标准。

# 可维可测分析

## 精度标准/性能标准

| 标准 | 要求 |
| --- | --- |
| 精度 | 对比 Netlib BLAS strmm，采用 float32 混合容差，rtol=atol=2^-13，匹配率≥99%，并检查最大绝对误差。 |
| 性能 | 在 910B3 上预热后取平均，完整算子耗时不高于对应 GPU 基线耗时的 1/0.8。 |

## 兼容性分析

在公共头文件复用既有 `aclblasStrmm` 声明，为 arch22 增加实现，不更改接口参数、错误码或 arch35 行为。构建通过架构选择对应实现，避免同名符号冲突。沿用 ops-blas handle、stream 和 workspace 生命周期约定；跨机型参数通过平台查询获得。
