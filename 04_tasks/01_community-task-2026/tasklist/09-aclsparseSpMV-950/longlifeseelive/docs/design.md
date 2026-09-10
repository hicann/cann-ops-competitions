# 需求背景（required）

## 需求来源

本设计来源于《aclsparseSpMV 算子开发任务书（A5）》。任务面向 ops-sparse 仓库，目标是在 Ascend 950PR（A5，DAV_3510，arch35）上提供 `aclsparseSpMV` 的 C++ 三阶段接口、Host 调度、Ascend C Kernel、Python/ATen 适配及配套测试。

## 背景介绍

### aclsparseSpMV 算子适配背景

`aclsparseSpMV` 用于计算 CSR 稀疏矩阵与稠密向量的乘法，是稀疏线性代数中的基础算子。典型应用场景包括大模型稀疏权重、稀疏激活、图计算和科学计算中的矩阵向量乘。

本任务要求面向 Ascend 950PR（A5）补齐 `aclsparseSpMV` 在 arch35 上的功能、性能、内存和框架适配能力。C++ 接口语义参考 cuSPARSE SpMV，采用 `GetBufferSize -> Preprocess -> SpMV` 三阶段接口。Python/ATen 侧需要适配 `torch.mv(mat, vec)` 与 `aten::mv`，使 CSR 二维稀疏矩阵乘一维稠密向量能够通过 NPU Dispatcher 调用到底层 aclsparse 接口与 Ascend C Kernel。

### CSR SpMV 功能分析

CSR 格式使用 `rowPtr`、`colInd`、`values` 表示稀疏矩阵 A。对矩阵 A 的每一行，Kernel 根据 `rowPtr` 给出的非零区间读取列索引和值，并从稠密向量 X 中 Gather 对应元素完成乘加，最后按 `alpha` 和 `beta` 写入输出向量 Y。

设计需覆盖动态 M、K、nnz、空行、长尾行、零 nnz、转置、base 0/1、I32 索引、任务书声明的 dtype/computeType 组合以及 complex64 复数乘加语义。核心计算在 NPU 上完成，不使用 CPU 作为计算回退。

# 需求分析（required）

## 需求描述

使用 aclsparse C++ 算子工程模式，为 Ascend 950PR（A5，DAV_3510，arch35）设计 `aclsparseSpMV`。设计范围包含：

1. `aclsparseSpMVGetBufferSize`
2. `aclsparseSpMVPreprocess`
3. `aclsparseSpMV`
4. Host 参数校验、workspace 管理和 stream 调度
5. Ascend C Kernel 计算
6. C++ UT、端到端 UT、Python/ATen `torch.mv` / `aten::mv` 适配

算子公式为：

`Y = alpha * op(A) * X + beta * Y`

A 为 CSR 稀疏矩阵，X/Y 为稠密向量。支持 `NON_TRANSPOSE` 和 `TRANSPOSE`，I32 索引，index base 0/1。核心计算必须在 Ascend 950PR NPU 上完成，不允许通过 CPU fallback 替代。

## 需求拆解

1. CSR SpMV 三阶段 C++ 接口：提供 workspace 查询、可选预处理和执行阶段，并保证阶段语义与状态生命周期一致。
2. 操作类型：支持 `NON_TRANSPOSE` 和 `TRANSPOSE`；未声明的操作类型返回明确错误。
3. 索引类型：CSR `rowPtr` 和 `colInd` 使用 I32。
4. index base：支持 base 0 和 base 1，并在 Host/Preprocess 侧统一索引语义。
5. alpha/beta：按 computeType 解释缩放标量，支持 `Y = alpha * op(A) * X + beta * Y`。
6. workspace/preprocess/pattern reuse：`GetBufferSize` 给出后续阶段所需 workspace，`Preprocess` 对 CSR pattern 相关数据进行预处理，执行阶段复用合法预处理结果。
7. dtype/computeType：覆盖任务书规定的九类组合。
8. complex64：支持 complex64 乘加、complex alpha/beta，以及 FP32 输入生成 complex64 输出的 computeType 语义。
9. Python/ATen：适配 `torch.mv(mat, vec)`、`aten::mv` 和 `aten::mv.out`，映射 CSR 二维矩阵与一维向量。
10. current stream：Host 调度、预处理与 Kernel launch 使用调用方 stream。
11. 参数及异常校验：覆盖 handle、descriptor、shape、dtype、layout、base、operation、algorithm、workspace 和指针合法性。
12. 精度要求：按任务书指定 CPU Golden、容差和匹配率验收。
13. 性能要求：按 GPU 设备 Event median 与 NPU 同调用范围耗时计算倍率，声明 dtype 的性能倍率目标不低于 0.3。
14. 内存要求：workspace 与额外内存满足任务书定义的内存验收规则。
15. 无 CPU fallback：C++ 接口和 Python/ATen 路径均不得使用 CPU 计算替代 NPU Kernel。

# 详细设计（required）

## 算子分析

### 数学公式

`Y = alpha * op(A) * X + beta * Y`

其中：

- A：CSR 稀疏矩阵，逻辑形状为 `[M, K]`
- X：稠密输入向量
- Y：稠密输出向量，同时作为 `beta * Y` 的输入
- `alpha`、`beta`：按 computeType 解释的缩放标量

当 `op(A) = NON_TRANSPOSE` 时：

- A 的逻辑形状为 `[M, K]`
- X 长度为 K
- Y 长度为 M

当 `op(A) = TRANSPOSE` 时：

- `op(A)` 的逻辑形状为 `[K, M]`
- X 长度为 M
- Y 长度为 K

本设计范围中的 `TRANSPOSE` 表示普通转置，不把未声明的共轭转置静默映射为普通转置。

### 支持数据类型

| matA/vecX | vecY | computeType | 计算策略 |
|---|---|---|---|
| INT8 | INT32 | INT32 | INT32 整数乘加与输出 |
| INT8 | FP32 | FP32 | INT8 输入转换为 FP32 后累加 |
| FP16 | FP32 | FP32 | FP16 输入转换为 FP32 后累加 |
| BF16 | FP32 | FP32 | BF16 输入转换为 FP32 后累加 |
| FP16 | FP16 | FP32 | FP32 累加后按输出 dtype 写回 |
| BF16 | BF16 | FP32 | FP32 累加后按输出 dtype 写回 |
| FP32 | FP32 | FP32 | FP32 乘加与输出 |
| complex64 | complex64 | complex64 | complex64 复数乘加与输出 |
| FP32 | complex64 | complex64 | FP32 实数 dot 后按 complex alpha/beta 生成 complex64 输出 |

### 支持形状

设计支持动态：

- M
- K
- nnz

CSR pattern 覆盖：

- general CSR
- empty row
- long-tail row
- irregular row distribution
- `nnz = 0`
- `nnz = 1`
- `NON_TRANSPOSE`
- `TRANSPOSE`
- base 0
- base 1

Host 侧根据 M、K、nnz、向量长度和 CSR 元数据完成合法性检查；Kernel 侧根据 Host 传入的 tiling 和辅助元数据处理尾块、空行和非均匀分布。

## 算子实现

### 实现方案

#### 3.2.1 host侧设计：

##### 1. 参数校验与接口语义

Host 侧设计负责统一校验：

- handle 是否有效
- `opA` 是否为声明范围内的 `NON_TRANSPOSE` 或 `TRANSPOSE`
- SpMat/DnVec descriptor 是否有效
- CSR layout、shape、nnz、rowPtr、colInd、values 是否合法
- CSR index 类型是否为 I32
- index base 是否为 base 0 或 base 1
- X/Y 长度是否与 `opA` 和矩阵形状匹配
- matA/vecX/vecY dtype 与 computeType 是否属于任务书声明组合
- `alpha`、`beta` 是否符合 pointer mode 和 computeType 语义
- algorithm 枚举是否合法并受支持
- `bufferSize`、`externalBuffer`、workspace 对齐和生命周期是否满足接口要求
- 调用方 stream 是否可用于 Host 调度与 Kernel launch

非法参数按接口语义返回对应错误；未声明 dtype、layout、operation 或 algorithm 返回不支持类错误；空指针、非法 shape、索引越界、溢出等返回参数错误或执行错误。

##### 2. 三阶段接口设计

###### GetBufferSize

`aclsparseSpMVGetBufferSize` 设计职责：

- 根据 dtype、computeType、operation、shape、nnz、index base、CSR pattern 和 algorithm 计算 workspace 字节数
- 为 Preprocess 所需的索引转换、格式化辅助数据、调度元数据和 Kernel 参数预留空间
- 对所有 offset、alignment 和 size 加法进行溢出检查
- 对无需 workspace 的场景返回零
- 保证返回值能够满足后续 Preprocess 与 Execute 在相同输入条件下使用

###### Preprocess

`aclsparseSpMVPreprocess` 设计职责：

- 对 CSR pattern 相关、可跨多次执行复用的信息进行预处理
- 对 base 1 输入建立统一索引语义，避免 Kernel 内重复处理不同 base
- 为转置访问构造适合目标输出维度并行计算的辅助索引或布局
- 为性能路径构造 Kernel 所需调度元数据和压缩/重排数据
- 将 preprocess 状态绑定到 matrix pattern、descriptor 元数据、operation、computeType、algorithm、workspace 和生命周期
- 当 descriptor pattern、values 指针、operation、computeType、algorithm 或 workspace 不兼容时要求重新 Preprocess
- Preprocess 失败时保持状态无效，防止执行阶段复用不完整数据

Preprocess 只处理可复用的结构与辅助数据。X、运行期 Y 和依赖本次调用的输入内容不作为可跨 pattern 固化的数据。

###### SpMV Execute

`aclsparseSpMV` 设计职责：

- 校验输入 descriptor、operation、computeType、algorithm、workspace 与 preprocess 状态是否兼容
- 对需要 preprocess 的路径使用已验证的 workspace 和辅助元数据
- 对无需 preprocess 的路径直接生成 tiling 并 launch Kernel
- 在调用方 stream 上执行 NPU Kernel
- 完成 `alpha * op(A) * X + beta * Y`
- 保证输入 A/X 只读，Y 按接口语义更新

##### 3. NON_TRANSPOSE设计

CSR 天然按行组织，`NON_TRANSPOSE` 设计以输出行为主要并行单元。Host 侧根据 M、K、nnz、平均行 nnz、行分布、dtype、computeType 和 UB 容量选择多核任务划分方式。

设计原则：

- 每个输出元素由单一工作单元负责，避免多个核无保护写同一 Y 元素
- 对连续 CSR values 和 colInd 进行批量搬运
- 对 X 的随机访问采用局部 staging、Gather 或等价局部访问策略
- 对长行、空行和非均匀稀疏分布使用负载均衡策略
- 对小规模和零 nnz 场景使用低开销路径
- 对 complex64 使用复数实部/虚部分离或等价表示完成乘加，保证输出写回的 ownership

##### 4. TRANSPOSE设计

CSR 输入在 `TRANSPOSE` 下的输出维度为原矩阵列维度。若直接从原始 CSR 行遍历写 Y 的列位置，多个工作单元可能竞争同一输出元素。因此 Host/Preprocess 侧设计构造适合转置访问的辅助索引或布局，使 Execute 阶段能够按目标输出区间并行计算。

设计原则：

- Preprocess 根据 CSR 构造转置访问需要的结构化索引
- base 0/base 1 在 Preprocess 阶段统一为 Kernel 可直接使用的索引语义
- Execute 阶段不重复进行昂贵的全量格式转换
- 每个工作单元独占一段输出区间，避免无保护写冲突
- complex64 的转置只进行普通转置语义，不进行未声明的共轭转换

##### 5. 数据类型与累加策略

Host 侧根据 dtype/computeType 选择 Kernel 计算策略：

- INT8→INT32：使用 INT32 整数累加并写回 INT32
- INT8→FP32：输入转换为 FP32 后累加并写回 FP32
- FP16→FP32：输入转换为 FP32 后累加并写回 FP32
- BF16→FP32：输入转换为 FP32 后累加并写回 FP32
- FP16→FP16：使用 FP32 累加，最终按 FP16 输出
- BF16→BF16：使用 FP32 累加，最终按 BF16 输出
- FP32→FP32：使用 FP32 乘加
- complex64→complex64：执行复数乘加
- FP32→complex64：先按 FP32 实数 dot 计算 `A * X`，再按 complex alpha/beta 生成 complex64 输出

complex alpha/beta 按 `{real, imag}` 解释。对 `dot = op(A) * X`，输出满足：

- `out.real = alpha.real * dot.real - alpha.imag * dot.imag + beta.real * oldY.real - beta.imag * oldY.imag`
- `out.imag = alpha.real * dot.imag + alpha.imag * dot.real + beta.real * oldY.imag + beta.imag * oldY.real`

对 FP32→complex64，`dot.imag = 0`。

##### 6. index base设计

设计统一支持 base 0 和 base 1。Host/Preprocess 侧对 CSR rowPtr 起点/终点、单调性、nnz 一致性和 colInd 范围进行检查。

Kernel 内部使用统一、明确的索引表示，避免为 base 0 和 base 1 重复实现核心计算逻辑。用户输入 CSR 保持只读；需要统一索引语义时，在 workspace 中构造辅助表示。

##### 7. workspace与内存设计

workspace 由 `GetBufferSize` 统一管理，并由 Preprocess 与 Execute 复用。

设计原则：

- 不持久化完整稠密矩阵
- 输入 A/X 只读
- Y 按接口语义原地更新
- workspace 与 descriptor、pattern、operation、computeType、algorithm 和 buffer 生命周期绑定
- 对 base 1 规范化、转置辅助索引、调度元数据、packed values 或 packed indices 进行统一 offset 和 alignment 管理
- workspace 大小受目标硬件 L2 Cache 约束或任务书额外内存规则约束
- 对零 workspace 场景兼容空 externalBuffer
- 对需要 workspace 的场景，空 externalBuffer 返回错误
- 对资源申请、销毁和异常路径进行生命周期管理，避免泄漏

##### 8. stream设计

Host 侧使用调用方 stream 完成 Preprocess 所需数据搬运和 Kernel launch。Python/ATen 适配、C++ 三阶段接口和 Kernel 调度均遵循 handle 或框架传入的 current stream 语义。

核心计算不得转 CPU。Host 侧如存在必要的参数检查、预处理搬运或接口同步，应与任务书和目标框架语义保持一致，并在测试中覆盖 current stream 调用行为。

##### 9. pointer mode

`alpha` 和 `beta` 按任务书接口参数说明作为标量指针传入。Host pointer mode 下，Host 侧读取标量并转换为 computeType 对应表示。

若按 cuSPARSE 对齐启用 Device pointer mode，设计需在 handle 中区分 pointer mode，并在 Device mode 下校验标量指针位于设备侧，避免 Host 直接解引用。Device mode 的标量读取和应用应由 Kernel 或设备侧辅助流程完成，并通过专项用例覆盖。

##### 10. Python / ATen 设计

Python/ATen 调用链设计为：

`torch.mv(mat, vec)` → `aten::mv` → NPU Dispatcher → CSR descriptor / DnVec descriptor → `aclsparseSpMV*` → Ascend C Kernel

`aten::mv.out` 复用相同底层接口，并遵循 out Tensor 的 dtype、shape、device、alias 和写回语义。

ATen 侧职责：

- 识别 CSR 二维稀疏矩阵与一维稠密向量
- 校验 dtype、shape、layout、stride 和 device
- 构造输出 Tensor 或校验 out Tensor
- 按 `torch.mv` 语义使用 `alpha=1`、`beta=0`
- 将 CSR 元数据映射为 SpMat descriptor
- 将 dense vector 映射为 DnVec descriptor
- 使用 NPU Dispatcher 和调用方 current stream
- 对声明范围外的 dtype、layout、device 或 requires-grad 场景返回明确错误
- 禁止 CPU fallback

#### 3.2.2 kernel侧设计：

Kernel 侧采用 Init 和 Process 两阶段，其中 Process 包含 CopyIn、Compute、CopyOut。

##### Init

Kernel Init 阶段设计负责：

- 获取 A values、rowPtr、colInd、X、Y 和 workspace 指针
- 解析 Host 传递的 tiling 或辅助元数据
- 初始化 GlobalTensor 和 LocalTensor
- 根据 dtype、operation 和分核策略初始化 UB 缓冲
- 确定本核负责的输出行或输出区间

##### Process

Process 阶段按 CopyIn、Compute、CopyOut 组织，使数据搬运、计算和写回边界清晰。

##### CopyIn

CopyIn 阶段设计负责：

- 按 tile 搬入 CSR values 和 indices
- 根据当前计算区间搬入或 Gather 稠密 X 的相关元素
- 对连续数据尽量批量搬运
- 根据 UB 容量切分长行、列窗口或输出区间
- 处理尾块、空行和零 nnz

##### Compute

Compute 阶段设计负责：

- 根据 colInd 或预处理后的索引 Gather 对应 X 元素
- 按 dtype/computeType 执行乘加
- FP16/BF16 混合场景使用 FP32 累加
- INT8→INT32 使用整数累加
- complex64 执行实部/虚部复数乘加
- 针对长行和非均匀分布进行合理任务切分
- 保证同一输出元素在同一执行单元内完成累加

##### CopyOut

CopyOut 阶段设计负责：

- 应用 alpha 和 beta
- 对 `beta != 0` 场景读取旧 Y 并按公式合并
- 将结果写回 Y
- 保证输出 ownership，避免核间写冲突
- 处理尾元素和 dtype 转换

##### tiling / 分核策略

Host 侧根据 M、K、nnz、平均行 nnz、行分布、dtype、computeType、operation 和 UB 大小确定分核策略。

设计原则：

- 优先利用多核并行
- 按输出行或输出区间划分任务，确保写回无冲突
- 长尾行或非均匀分布下避免单核负载过重
- 单核内根据 UB 容量进行 tile
- 对转置场景利用 Preprocess 辅助减少 Execute 阶段重复工作
- 对尾块、空行、零 nnz 和小规模场景设置低开销处理策略

##### tiling key / algorithm分类

Host 侧可根据以下任务书层面的属性选择不同 tiling key 或 algorithm 分支：

- dtype/computeType
- `opA`
- M、K、nnz
- 平均行 nnz 和行分布特征
- index base
- 是否需要 workspace/preprocess
- algorithm 枚举

内部分类只服务于已声明接口语义。未声明 layout、dtype 组合或 algorithm 不进入 Kernel 执行路径。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Ascend 950PR（A5，DAV_3510，arch35） | √ |

## 算子约束限制

- 稀疏矩阵格式：CSR
- 索引类型：I32
- index base：base 0、base 1
- 操作类型：`NON_TRANSPOSE`、`TRANSPOSE`
- X/Y：一维连续 dense vector
- dtype/computeType：仅支持任务书声明的九类组合
- alpha/beta：按 computeType 解释，Host pointer mode 为基础语义；Device pointer mode 按任务书条件性要求设计专项分支
- workspace：调用方按 `GetBufferSize` 查询结果提供 externalBuffer；需要 workspace 的场景不接受空 buffer
- preprocess：状态与 pattern、descriptor、operation、computeType、algorithm、workspace 和生命周期绑定；pattern 改变后重新 Preprocess
- 未声明 layout、algorithm、dtype 组合、共轭转置或非法枚举返回明确错误
- 核心计算不得 CPU fallback
- 标准 `aclsparseSpMV` 与 `aclsparseSpMVOp` 接口职责分离
- A5 Host 逻辑与 A2/A3 硬件分支解耦

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述(不涉及说明原因) | 标准来源 |
| --- | --- | --- |
| 精度标准 | 采用单标杆方法，NPU 结果与 CPU Golden 比较，整体匹配率和误差上限满足任务书要求 | 《aclsparseSpMV 算子开发任务书（A5）》3.2 |
| 性能标准 | 性能倍率 = GPU 设备 Event 调用耗时 / NPU 同调用范围总耗时；声明 dtype 在 P-01/P-02/P-03 上目标不低于 0.3 | 《aclsparseSpMV 算子开发任务书（A5）》3.3 |
| 内存标准 | 满足输入输出总量场景的额外内存规则，或方案固有 workspace 不超过目标硬件 L2 Cache 容量 | 《aclsparseSpMV 算子开发任务书（A5）》3.4 |

### 精度设计

精度测试采用 CPU Golden：

- INT8→INT32：整数精确计算，逐元素 exact match
- INT8→FP32：按 FP32 Golden 验收
- FP16：按任务书 float16 容差验收
- BF16：按任务书 bfloat16 容差验收
- FP32：CPU 侧采用 float64 计算 Golden，按任务书 float32 容差验收
- complex64：CPU 侧采用 complex128 计算 Golden，实部和虚部分别按 float32 混合容差验收

浮点组合按：

`|actual - golden| ≤ atol + rtol × |golden|`

并满足任务书规定的整体匹配率和绝对误差上限。float16、bfloat16、float32 的 rtol、atol 和绝对误差上限按任务书表述执行。INF/NAN 按生态算子开源精度标准对应规则验收。

### 性能设计

性能设计遵循任务书口径：

- GPU 设备 Event median 作为性能标杆
- NPU 侧统计与标杆接口相同调用范围内所有 Kernel 的总耗时
- warmup 不少于 10 次
- samples 不少于 30 次
- 报告 median 和 P90
- 描述符、workspace 和 preprocess 结果在正式采样期间复用
- 不计入首次编译、数据生成、Host 到 Device 搬运和无关初始化
- P-01、P-02、P-03 使用任务书指定的大模型维度锚点和固定随机生成规则
- 所有声明 dtype 的性能倍率目标不低于 0.3

### 内存设计

内存设计遵循任务书两类验收规则：

- 输入输出总量超过任务书阈值时，对比同 Torch API 调用下 NPU 较 GPU 使用的额外内存
- 方案固有 workspace 的绝对值不超过目标硬件 L2 Cache 容量

workspace 通过 `GetBufferSize` 查询，Preprocess 与 Execute 复用。设计不额外持久化完整稠密矩阵，不泄漏描述符或设备内存，并通过 memory runner 汇总输入基线、峰值内存和额外峰值内存。

## 兼容性分析

设计兼容目标：

- CANN 9.1.0 及后续配套版本
- PyTorch 2.7 及以上
- torch_npu 26.0.0 及之后
- Ascend 950PR（A5，DAV_3510，arch35）
- ops-sparse 公开头文件与描述符体系
- 标准 `aclsparseSpMV` 三阶段接口
- Python/ATen `torch.mv`、`aten::mv`、`aten::mv.out`

A5 Host 逻辑与 A2/A3 硬件分支解耦。标准 `aclsparseSpMV` 与 `aclsparseSpMVOp` 职责分离，避免不同接口语义混用。
