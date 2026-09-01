# 【社区任务】SpMV算子设计文档

| 项目 | 内容 |
| --- | --- |
| 任务 | 8月社区任务 08-16：SPMV算子开发 |
| 提交者 | amelia_2026 |
| 目标代码仓 | cann/ops-sparse |
| 目标硬件 | Ascend 950PR（DAV-3510，arch35） |
| 开发语言 | Ascend C / C++ |
| 文档状态 | 设计阶段，精度与性能结果待实现后在950PR实测 |

# 一、需求背景

## 1.1 需求来源

本需求来自“8月社区任务-SPMV算子开发”。任务要求参考cuSPARSE SpMV接口语义，在Ascend 950PR上使用Ascend C实现CSR稀疏矩阵与稠密向量乘法，并以aclsparse接口风格提供缓冲区查询、可选预处理和计算三个阶段。

目标计算为：

$$
Y = \alpha \cdot op(A) \cdot X + \beta \cdot Y
$$

其中，$A$为CSR稀疏矩阵，$op(A)$支持$A$和$A^T$；$X$、$Y$为一维稠密向量；$Y$同时作为输入和输出。

任务要求支持以下能力：

1. Ascend 950PR上的CSR SpMV非转置与转置计算；
2. 任务书规定的7种输入、计算和输出数据类型组合；
3. Host/Device两种标量指针模式；
4. GetBufferSize、Preprocess、SpMV三阶段接口；
5. 相同输入多次执行结果一致；
6. 不同矩阵规模、稀疏度、alpha/beta组合及边界场景；
7. 950PR实机精度、确定性与性能验证。

## 1.2 背景介绍

### 1.2.1 SpMV标杆实现与接口位置

本任务是aclsparse库算子适配，不是aclnn或TBE算子，因此仓内没有对应的算子信息库JSON和TBE实现文件。评审时用于确认接口、参数和实现语义的文件如下：

| 类型 | 仓内路径 | 用途 |
| --- | --- | --- |
| 公共接口声明 | include/cann_ops_sparse.h | GetBufferSize、Preprocess、SpMV原型及参数语义 |
| 算子使用说明 | sparse/spmv/README.md | 产品、格式、数据类型和约束说明 |
| 既有Host实现 | sparse/spmv/arch22/spmv_host.cpp | 参数校验、平台信息、tiling与Kernel启动参考 |
| 既有tiling结构 | sparse/spmv/arch22/spmv_tiling_data.h | arch22 Host/Kernel传参参考 |
| 既有Kernel实现 | sparse/spmv/arch22/kernels/spmv_kernel.h | 非转置与转置CopyIn/Compute/CopyOut流程参考 |
| 架构选择 | CMakeLists.txt | ascend950到arch35（DAV-3510）的构建路由 |

接口参数以include/cann_ops_sparse.h为准，数据类型和约束同时对照任务书与sparse/spmv/README.md。由于没有TBE源码，本设计中的“标杆算子”特指仓内arch22实现；950PR方案与其保持API和数学语义一致，不要求内部实现机制完全相同。

### 1.2.2 SpMV现状分析

#### 1.2.2.1 标杆算子支持的数据类型和数据格式

标杆实现使用CSR格式，row offsets和column indices均为int32，index base为zero-based。任务要求的目标类型组合如下：

| CSR values / X | computeType | Y | 目标累加类型 |
| --- | --- | --- | --- |
| float32 | float32 | float32 | float32 |
| int8 | int32 | int32 | int64临时累加，写回int32 |
| int8 | float32 | float32 | float32 |
| float16 | float32 | float32 | float32 |
| float16 | float32 | float16 | float32，写回转换 |
| bfloat16 | float32 | float32 | float32 |
| bfloat16 | float32 | bfloat16 | float32，写回转换 |

arch22现有代码中的整数路径为INT32输入/INT32输出，且只支持DEFAULT算法；任务书要求的是INT8输入，因此950PR适配不能直接照搬该数据类型分发。

#### 1.2.2.2 标杆算子实现描述

arch22 Host侧先校验描述符、CSR格式、index类型、opA、computeType、X/Y长度及stream，然后把row offsets从Device拷回Host，计算最大行长度并检查UB容量。Host侧生成rows、cols等tiling数据，根据数据类型和transpose标记启动对应模板Kernel。

非转置Kernel按CSR行分核，每行执行CopyIn、Compute、CopyOut。转置Kernel先按beta缩放Y，再逐CSR行生成贡献，并对Y的列位置执行原子累加。

#### 1.2.2.3 标杆算子实现流程图

~~~mermaid
flowchart TD
    A[调用aclsparseSpMV] --> B[Host参数与描述符校验]
    B --> C[查询AIV核数]
    C --> D[Device到Host读取row offsets]
    D --> E[计算最大行长度并校验UB容量]
    E --> F[生成rows和cols等tiling数据]
    F --> G{opA}
    G -->|NON_TRANSPOSE| H[按CSR行分核]
    H --> I[CopyIn]
    I --> J[Compute行点积]
    J --> K[CopyOut写Y]
    G -->|TRANSPOSE| L[先执行beta乘Y]
    L --> M[按CSR行计算贡献]
    M --> N[AtomicAdd到对应Y列]
~~~

#### 1.2.2.4 标杆实现与任务要求的差距

| 差距 | 标杆实现 | 950PR任务要求 |
| --- | --- | --- |
| 架构 | arch22，DAV-2201 | arch35，DAV-3510 |
| 整数输入 | INT32 | INT8，computeType为INT32或FLOAT32 |
| 算法枚举 | 仅DEFAULT | DEFAULT、CSR_ALG1、CSR_ALG2 |
| 三阶段接口 |执行入口为主，workspace未实际使用 | GetBufferSize、Preprocess、SpMV完整流程 |
| 转置确定性 | 多核AtomicAdd，浮点累加顺序可能变化 | 相同输入重复执行结果一致 |
| Host同步 |为计算最大行长执行D2H及stream同步 | 热路径避免依赖整份row offsets的Host同步 |
| 950PR适配 | 不支持 | 必须支持 |

# 二、需求分析

## 2.1 外部组件依赖

| 依赖 | 使用位置 | 说明 |
| --- | --- | --- |
| ACL Runtime | Host侧 | stream、Device内存及异步Kernel执行 |
| platform_ascendc | Host侧 | 查询950PR AIV核数 |
| Ascend C Kernel API | Kernel侧 | AIV Kernel、SIMT VF调用、shuffle与数据类型 |
| aclsparse公共接口 | API层 | handle、CSR/DnVec描述符、状态码和枚举 |
| CMake架构路由 | 构建层 | ascend950选择arch35目录 |

方案不引入新的第三方运行时依赖。

## 2.2 内部适配模块

| 模块 | 适配内容 |
| --- | --- |
| sparse/common | 为SpMV维护独立的预处理workspace状态，避免与其他稀疏算子混用 |
| sparse/spmv/arch35/spmv_host.cpp | 参数校验、workspace、tiling、预处理和执行入口 |
| sparse/spmv/arch35/spmv.h | arch35数据类型编号、tiling结构及workspace布局 |
| sparse/spmv/arch35/spmv_kernel.cpp | 转置预处理、非转置/转置计算及数据类型分发 |
| sparse/spmv/README.md | 950PR产品支持、数据类型、算法与workspace说明 |
| test/spmv/arch35 | API、精度、异常、确定性和性能测试 |

## 2.3 需求模块设计

### 2.3.1 Ascend C算子原型

对外接口保持include/cann_ops_sparse.h中的三阶段原型，不新增参数：

~~~cpp
aclsparseStatus_t aclsparseSpMVGetBufferSize(
    aclsparseHandle_t handle,
    aclsparseOperation_t opA,
    const void *alpha,
    aclsparseConstSpMatDescr_t matA,
    aclsparseConstDnVecDescr_t vecX,
    const void *beta,
    aclsparseDnVecDescr_t vecY,
    aclDataType computeType,
    aclsparseSpMVAlg_t alg,
    size_t *bufferSize);

aclsparseStatus_t aclsparseSpMVPreprocess(
    aclsparseHandle_t handle,
    aclsparseOperation_t opA,
    const void *alpha,
    aclsparseConstSpMatDescr_t matA,
    aclsparseConstDnVecDescr_t vecX,
    const void *beta,
    aclsparseDnVecDescr_t vecY,
    aclDataType computeType,
    aclsparseSpMVAlg_t alg,
    void *externalBuffer);

aclsparseStatus_t aclsparseSpMV(
    aclsparseHandle_t handle,
    aclsparseOperation_t opA,
    const void *alpha,
    aclsparseConstSpMatDescr_t matA,
    aclsparseConstDnVecDescr_t vecX,
    const void *beta,
    aclsparseDnVecDescr_t vecY,
    aclDataType computeType,
    aclsparseSpMVAlg_t alg,
    void *externalBuffer);
~~~

### 2.3.2 功能语义

- opA为NON_TRANSPOSE：$A$形状为$M \times K$，X长度至少为$K$，Y长度至少为$M$；
- opA为TRANSPOSE：计算$A^T$，X长度至少为$M$，Y长度至少为$K$；
- beta为0时不读取旧Y，直接覆盖输出；
- NNZ为0时乘积项为0，仍执行$Y=\beta Y$；
- Y原位更新，不创建完整输出副本；
- alpha和beta支持Host pointer mode与Device pointer mode。

### 2.3.3 相对标杆需补齐的约束

1. 新增INT8输入到INT32、INT8输入到FLOAT32两条路径；
2. 新增GetBufferSize和Preprocess的有效实现；
3. 转置路径不使用对同一浮点输出的跨核原子加；
4. DEFAULT、CSR_ALG1、CSR_ALG2首版共用正确性路径，后续仅在实测证明收益后分化；
5. rows、cols、NNZ不得超过INT32_MAX；
6. CSR row offsets与column indices均为int32，且仅支持zero-based；
7. 不支持共轭转置；
8. 修改CSR结构后必须重新Preprocess，仅更新values时可以复用。

# 三、需求详细设计

## 3.1 调用方式

本算子通过aclsparse C接口调用，不属于aclnn、PyTorch算子注册或Kernel直调接口。推荐调用顺序如下：

~~~mermaid
sequenceDiagram
    participant App as 调用方
    participant API as aclsparse Host API
    participant Pre as 预处理Kernel
    participant Compute as SpMV Kernel
    App->>API: aclsparseSpMVGetBufferSize
    API-->>App: bufferSize
    App->>App: 分配externalBuffer
    App->>API: aclsparseSpMVPreprocess（可选）
    API->>Pre: 构建转置索引
    App->>API: aclsparseSpMV
    API->>Compute: 异步执行
    Compute-->>App: stream同步后读取Y
~~~

非转置路径bufferSize为0且Preprocess为空操作。转置且NNZ大于0时需要workspace；调用方可以显式Preprocess并复用，也允许SpMV发现当前buffer未激活时执行一次预处理。

## 3.2 需求总体设计

### 3.2.1 Host侧设计

#### 3.2.1.1 参数校验

三个入口共享校验逻辑：

1. handle、矩阵/向量描述符、alpha、beta及必要数据指针非空；
2. opA、alg、CSR格式、index base和index type受支持；
3. rows、cols、NNZ在int32范围内；
4. CSR values与X类型一致，数据类型组合位于支持表；
5. X、Y逻辑长度满足转置或非转置维度；
6.执行和预处理时stream有效；
7. 转置且NNZ大于0时externalBuffer存在。

校验失败返回与aclsparse约定一致的HANDLE_IS_NULLPTR、INVALID_VALUE、NOT_SUPPORTED或INSUFFICIENT_RESOURCES。

#### 3.2.1.2 分核策略

输出长度记为：

$$
L =
\begin{cases}
M, & opA = NON\_TRANSPOSE \\
K, & opA = TRANSPOSE
\end{cases}
$$

Host侧查询AIV核数$C_{AIV}$。可用外层block上限为$3C_{AIV}$，实际block数为：

$$
B = \min(L, 3C_{AIV})
$$

若$L=0$，直接返回成功。否则每个block处理连续输出区间，区间长度为：

$$
W = \left\lceil \frac{L}{B} \right\rceil
$$

第$b$个block处理：

$$
[bW, \min((b+1)W, L))
$$

该分核方式保证输出区间互不重叠，不需要跨block同步或原子写Y。

#### 3.2.1.3 数据分块和内存优化策略

非转置路径直接访问CSR和向量，不申请workspace。转置路径将结构信息转换为稳定CSC索引，values仍保留在原CSR数组中。

workspace按64字节对齐：

| 区域 | 元素数 | 用途 |
| --- | ---: | --- |
| header/reserved | 256 bytes | 对齐及后续元数据扩展 |
| colOffsets | K + 1 | CSC列偏移 |
| next | K | 预处理写入游标 |
| rowIndices | NNZ | CSC行索引 |
| permutation | NNZ | CSC位置到原CSR values下标 |

布局递推为：

$$
O_{col}=256
$$

$$
O_{next}=align_{64}(O_{col}+4(K+1))
$$

$$
O_{row}=align_{64}(O_{next}+4K)
$$

$$
O_{perm}=align_{64}(O_{row}+4NNZ)
$$

$$
S_{workspace}=align_{64}(O_{perm}+4NNZ)
$$

非转置或NNZ为0时，GetBufferSize返回0。

#### 3.2.1.4 tilingKey规划策略

arch35首版不使用单一整数tilingKey，而是在tiling数据中用dataType和transpose共同决定模板分支，条件如下：

| dataType分支 | 输入/输出 | computeType |
| --- | --- | --- |
| FP32_FP32 | FP32 -> FP32 | FP32 |
| INT8_INT32 | INT8 -> INT32 | INT32 |
| INT8_FP32 | INT8 -> FP32 | FP32 |
| FP16_FP32 | FP16 -> FP32 | FP32 |
| FP16_FP16 | FP16 -> FP16 | FP32 |
| BF16_FP32 | BF16 -> FP32 | FP32 |
| BF16_BF16 | BF16 -> BF16 | FP32 |

transpose为0时读取CSR row offsets和column indices；transpose为1时读取预处理得到的CSC元数据。该组合等价于14条可判定路径，避免运行中依赖不透明的动态类型判断。

#### 3.2.1.5 Host侧流程

~~~mermaid
flowchart TD
    A[进入三阶段接口之一] --> B[统一参数校验]
    B --> C{接口类型}
    C -->|GetBufferSize| D{转置且NNZ大于0}
    D -->|否| E[返回0]
    D -->|是| F[计算对齐workspace布局]
    C -->|Preprocess| G{是否转置}
    G -->|否| H[成功空操作]
    G -->|是| I[启动稳定CSR-to-CSC预处理]
    C -->|SpMV| J[计算输出长度与block数]
    J --> K{转置索引已激活}
    K -->|否| I
    K -->|是| L[生成tiling数据]
    I --> L
    L --> M[异步启动arch35 SpMV Kernel]
~~~

### 3.2.2 Kernel侧设计

#### 3.2.2.1 非转置实现

CSR第$i$行输出为：

$$
Y_i = \alpha \sum_{p=rowPtr_i}^{rowPtr_{i+1}-1}
csrVal_p \cdot X_{csrColInd_p} + \beta Y_i
$$

每个线程组负责一个CSR行。浮点路径使用4个线程协作一行：

1. 每个线程跨步读取非零元素；
2. 主循环按4次FMA展开；
3. 使用固定宽度shuffle完成组内归约；
4. lane 0融合alpha/beta并写回。

线程数按32向上对齐并限制在512以内。INT8到INT32路径使用int64临时累加和缩放，写回时处理int32范围；其余低精度输入使用float32累加。

#### 3.2.2.2 转置预处理与计算

转置输出第$j$个元素为：

$$
Y_j = \alpha \sum_{i=0}^{M-1} A_{ij}X_i + \beta Y_j
$$

预处理按固定顺序执行：

1. 统计每列NNZ得到colOffsets；
2. 对列计数执行前缀和；
3. 按CSR行顺序和行内元素顺序写rowIndices；
4. 写permutation，记录CSC位置对应的原CSR values下标。

计算时每个输出列只由一个线程组负责，按colOffsets范围读取rowIndices和permutation。values不复制，通过permutation读取原CSR values。

#### 3.2.2.3 Ascend C实现流程图

~~~mermaid
flowchart TD
    A[arch35 Kernel入口] --> B[按dataType选择模板]
    B --> C[计算当前block输出区间]
    C --> D[按32对齐SIMT线程数]
    D --> E{transpose}
    E -->|0| F[读取CSR行范围]
    E -->|1| G[读取CSC列范围]
    F --> H[4线程组分段FMA]
    G --> H
    H --> I[固定shuffle归约]
    I --> J[计算alpha乘sum加beta乘oldY]
    J --> K[lane 0单写者写回Y]
~~~

转置预处理流程：

~~~mermaid
flowchart LR
    A[清零colOffsets] --> B[按colInd统计列计数]
    B --> C[前缀和生成列偏移]
    C --> D[复制列起点到next]
    D --> E[按CSR行序遍历NNZ]
    E --> F[写rowIndices和permutation]
~~~

#### 3.2.2.4 与标杆算子流程的差异及原因

| 设计点 | arch22标杆 | arch35方案 | 原因 |
| --- | --- | --- | --- |
| 执行模型 | Ascend C队列，CopyIn/Compute/CopyOut | AIV + SIMT VF线程组 | DAV-3510执行模型不同，稀疏短行适合细粒度SIMT |
| Host读取row offsets | D2H并同步求最大行长 | 不依赖整表D2H | 避免每次调用的Host同步和PCIe/总线开销 |
| 非转置并行 | 每核多个CSR行 | 连续输出区间 + 4线程组/行 | 提高高稀疏短行利用率 |
| 转置写回 | 多核AtomicAdd到Y | 稳定CSR-to-CSC + 每列单写者 | 消除写竞争并满足确定性 |
| 转置workspace | 忽略 | 保存CSC结构元数据 | 支持预处理复用及固定累加顺序 |
| INT8路径 | 标杆整数输入为INT32 | INT8输入，INT32/FP32计算 | 对齐任务书类型组合 |
| alpha/beta | Host值为主 | Host值或Device地址 | 对齐pointer mode语义 |
| 算法枚举 | DEFAULT | 三种枚举首版共用正确性路径 | 先保证完整语义，再以950PR数据选择策略 |

#### 3.2.2.5 确定性保证

1. 每个输出元素只由一个固定线程组写入；
2. 非转置按CSR存储顺序累加；
3. 转置预处理稳定保留CSR行顺序和行内元素顺序；
4. 线程组使用固定宽度、固定顺序shuffle归约；
5. 不对同一浮点输出执行跨核AtomicAdd；
6. 测试对同一输入重复执行至少20次并逐字节比较。

#### 3.2.2.6 性能优化策略

1. 按输出划分，避免跨block写竞争；
2. 4线程协作一行并展开FMA；
3. 寄存器归约，减少中间GM写回；
4. beta为0时跳过旧Y读取；
5. 转置只构建索引，不复制values；
6. 相同sparsity pattern复用预处理结果；
7. block数依据AIV核数和输出长度动态限制；
8. 后续依据950PR实测评估单线程、4线程、宽线程组和长行分段归约，不在设计阶段虚构收益。

## 3.3 支持硬件

| 支持的芯片版本 | 是否支持 | 实现目录 |
| --- | --- | --- |
| Ascend 950PR / ascend950系列 | 是 | sparse/spmv/arch35 |
| Atlas A2/A3 | 保持既有实现 | sparse/spmv/arch22 |

本次任务只对Ascend 950PR结果负责，不修改arch22 Kernel行为。

## 3.4 算子约束限制

| 类别 | 约束 |
| --- | --- |
| 稀疏格式 | 仅CSR |
| 索引 | row offsets和column indices均为int32 |
| index base | 仅zero-based |
| 操作 | NON_TRANSPOSE、TRANSPOSE；不支持CONJUGATE_TRANSPOSE |
| 规模 | rows、cols、NNZ不超过INT32_MAX |
| 数据类型 | 仅支持2.3节列出的7种组合 |
| 算法 | DEFAULT、CSR_ALG1、CSR_ALG2 |
| workspace | 转置且NNZ大于0时必须满足GetBufferSize结果 |
| 预处理复用 | CSR结构不变时可复用；修改row offsets/column indices后必须重做 |
| 稠密向量布局 | DnVec描述符只有长度和指针，无stride；非连续Tensor需由上层转换为连续逻辑向量 |
| 标量 | alpha/beta类型必须与computeType一致 |
| 异步语义 | Preprocess和SpMV在handle所绑定stream上执行 |

## 3.5 异常与边界处理

| 场景 | 预期行为 |
| --- | --- |
| handle为空 | HANDLE_IS_NULLPTR |
| 描述符、alpha或beta为空 | INVALID_VALUE |
| 非CSR、非zero-based、非int32 index | NOT_SUPPORTED |
| 不支持的数据类型组合或算法枚举 | NOT_SUPPORTED |
| X/Y长度不足 | INVALID_VALUE |
| 转置需要workspace但未提供 | INSUFFICIENT_RESOURCES |
| 输出长度为0 | 成功返回，不启动计算Kernel |
| NNZ为0 | 执行Y = beta * Y |
| alpha为0 | 结果只由beta * Y决定 |
| beta为0 | 不读取旧Y，直接覆盖 |

# 四、特性交互分析

## 4.1 三阶段接口交互

- GetBufferSize只计算workspace字节数，不启动计算；
- Preprocess只生成与CSR结构相关的转置索引，不依赖values内容；
- SpMV复用同一个externalBuffer时检查SpMV专用激活状态；
- 修改values不使预处理失效，修改row offsets或column indices必须重新预处理。

## 4.2 与公共描述符的交互

SpMV转置workspace布局与SpMM、SDDMM等算子不同，因此矩阵描述符使用SpMV独立的active-buffer标记。这样可以防止同一个指针曾被其他算子使用后被误判为有效SpMV元数据。

## 4.3 pointer mode与stream交互

- Host pointer mode在Host侧读取alpha/beta并写入tiling；
- Device pointer mode由Kernel在执行时读取标量；
- Preprocess和SpMV均使用handle中的stream，不在API内部执行无必要的同步；
- 调用方负责保证异步执行完成前描述符、输入、输出和workspace生命周期有效。

## 4.4 架构隔离

CMake根据SOC选择架构目录：ascend950路由到arch35，ascend910b/ascend910_93路由到arch22。arch35新增代码不替换arch22 Kernel；公共描述符变更需通过其他稀疏算子回归，避免兼容性退化。

# 五、可维可测分析

## 5.1 精度标准/性能标准

| 验收标准 | 目标 | 证据 |
| --- | --- | --- |
| 功能 | 三阶段接口、转置/非转置、全部类型和alpha/beta组合通过 | 950PR逐case日志 |
| 精度 | 满足生态算子开源精度标准 | CPU对比结果、最大误差和失败元素数 |
| 确定性 | 相同输入重复执行输出逐字节一致 | 至少20次重复执行compare/hash |
| 性能 | 达到任务书规定的0.5倍标杆水平 | 四个官方性能case原始日志 |
| 可复现 | 验收人可按测试README独立复现 | 环境、命令、SHA、输入和输出记录 |

### 5.1.1 静态与构建测试

- 许可证头、格式、编译告警和git diff --check；
- 950PR构建命中arch35；
- Host与Kernel目标完整编译、链接；
- 公共描述符变更不影响其他稀疏算子构建。

### 5.1.2 接口与异常测试

覆盖三个接口、三种alg、Host/Device pointer mode、转置/非转置，以及空指针、类型、维度、索引类型和workspace异常。

### 5.1.3 精度与功能测试

CPU参考实现使用相同CSR输入生成期望结果，覆盖：

- 全部7种数据类型组合；
- trans为false和true；
- alpha、beta分别取0、0.5、1.0、2.0；
- 稀疏度50%、75%、90%、95%、97.5%、99%、99.9%；
- 方阵、长矩阵、宽矩阵；
- 空矩阵、空行、单元素、重复列索引、极长行和行长度严重不均；
- NNZ为0、M为0、K为0；
- 显式预处理、隐式预处理和同一buffer复用；
- 只更新values后复用预处理结果。

精度门限按输出dtype和生态算子开源精度标准执行。报告逐case记录最大误差、失败元素数及参考实现版本，不以平均值代替全部case结论。

### 5.1.4 确定性测试

同一输入、stream、workspace和算法连续运行至少20次，保存每次输出并逐字节比较。转置、非转置和全部dtype组合均覆盖；bitwise一致后再执行数值精度门禁。

### 5.1.5 性能测试

| 编号 | 矩阵规模 | 稀疏度 | 标杆时延 |
| ---: | ---: | ---: | ---: |
| 1 | 128 x 128 | 95% | 43.9 us |
| 2 | 1024 x 1024 | 99% | 46.3 us |
| 3 | 2048 x 4096 | 97.5% | 45.4 us |
| 4 | 160220 x 68750 | 99.9% | 193.2 us |

测试要求：

1. 固定CANN、驱动、固件、设备ID、代码SHA和输入；
2. 预热后执行足够次数，报告median、P90、最小值和样本数；
3. 非转置报告端到端SpMV时延；
4. 转置分别报告Preprocess、首次调用和复用workspace的稳态时延；
5. 与标杆使用相同输入布局和同步边界；
6. 每项性能结果先通过精度和确定性门禁；
7. 按验收方对“达到0.5倍标杆水平”的统一口径判定，不混用吞吐和时延倍数。

## 5.2 兼容性分析

- 对外API原型不变；
- arch35通过现有SOC目录选择机制新增，不改变arch22/arch20 Kernel；
- 非转置GetBufferSize返回0，保持普通CSR SpMV调用习惯；
- Device pointer mode和stream异步语义与aclsparse接口一致；
- 公共描述符只增加SpMV专用预处理状态，需要执行SpMM、SpGEMM、SDDMM等回归；
- 设计文档只描述方案与验证方法，不把尚未完成的950PR编译、精度或性能测试写成已通过。
