# aclsparseSpSV算子（arch22）Ascend C实现设计说明

## 需求背景（required）
### 需求来源
本需求来源于CANN 2026社区算子开发任务，面向`ops-sparse`稀疏算子库能力补齐，基于Atlas A2/A3（arch22/DAV_2201）平台，使用Ascend C编程语言实现稀疏三角向量求解算子`aclsparseSpSV`，对标cuSPARSE SpSV接口语义，主计算全程运行于NPU，无CPU fallback，纳入ops-sparse算子库统一开源交付。

### 背景介绍
#### aclsparseSpSV算子实现优化
基于稀疏三角向量求解的标准接口语义，使用Ascend C编程语言完成arch22平台的全路径实现优化，覆盖Host侧生命周期管理、参数校验与Kernel侧核心计算，充分挖掘NPU向量算力与流水调度能力，实现高性能稀疏三角求解。
- 算子交付路径：`ops-sparse`仓库`sparse/spsv/arch22/`目录
- 公开接口定义路径：`include/cann_ops_sparse.h`

#### aclsparseSpSV算子现状分析
当前`ops-sparse`库已实现稀疏矩阵稠密矩阵乘法（SpMM）等算子，稀疏三角向量求解（SpSV）尚未在arch22平台落地，无对应Ascend C版本。本次需完整搭建Host/Kernel全链路，支持多稀疏格式、多操作语义与完整算子生命周期。

#### aclsparseSpSV算子功能分析
aclsparseSpSV算子完成稀疏三角线性方程组求解，核心语义为`op(A)Y = alpha × X`：
- 输入：稀疏三角方阵A、稠密向量X、缩放系数alpha
- 输出：稠密向量Y
- 支持稀疏格式：CSR、CSC、COO、SLICED_ELL
- 支持数据类型：ACL_FLOAT（FP32）、ACL_COMPLEX64
- 支持操作：非转置、转置、共轭转置；上/下三角、单位/非单位对角
- 支持矩阵更新：GENERAL全量更新、DIAGONAL对角更新

## 需求分析（required）
### 需求描述
在Atlas A2/A3（arch22/DAV_2201）平台使用Ascend C实现aclsparseSpSV算子，支持CSR/CSC/COO/SLICED_ELL四种稀疏三角矩阵格式，支持FP32与complex64数据类型，支持完整的算子生命周期管理与矩阵更新，精度符合生态算子开源精度标准，性能达到0.25倍GPU标杆以上。

### 需求拆解
1. 支持CSR、CSC、COO、SLICED_ELL四种稀疏矩阵描述符格式
2. 支持FP32、complex64数据类型，Device端索引采用I32，支持base 0/1
3. 支持LOWER/UPPER三角、UNIT/NON_UNIT对角，支持非转置、转置、共轭转置操作
4. 支持`BufferSize → Analysis → Solve → UpdateMatrix`完整算子生命周期
5. 支持GENERAL全量值更新、DIAGONAL对角值更新两种矩阵更新模式
6. 精度不低于生态算子开源精度标准，性能不低于0.25倍GPU标杆
7. 主计算全程运行于NPU，不提供CPU fallback

## 详细设计（required）
### 算子分析
#### 数学公式
算子完成稀疏三角线性方程组求解，数学表达式为：
$$ op(A) \times Y = \alpha \times X $$
其中：
- $A$ 为 $m \times m$ 稀疏三角方阵（上三角/下三角）
- $X$、$Y$ 为 $m$ 维稠密向量
- $op(A)$ 可取 $A$（非转置）、$A^T$（转置）、$A^H$（共轭转置）
- $\alpha$ 为缩放系数，与计算数据类型一致

#### 支持数据类型
- 计算类型：ACL_FLOAT（FP32）、ACL_COMPLEX64
- 矩阵索引：Device端统一使用I32类型，Host侧元数据使用int64_t
- alpha缩放系数：与computeType类型一致，支持Host/Device pointer mode读取
- complex64场景下，实部、虚部分别执行对应运算，共轭转置时对矩阵元素取共轭

#### 支持形状
- 稀疏矩阵A：m×m三角方阵，支持任意nnz规模，兼容空行、长尾分布、重复索引、未排序索引等场景
- 稠密向量X/Y：长度为m的连续内存向量，支持原地求解（Y与X共用同一Device values指针）
- 支持base 0与base 1两种索引基准

### 算子实现
#### 实现方案
##### 3.2.1 host侧设计
tiling策略：
算子计算强依赖稀疏矩阵的结构分布，host侧完成稀疏格式解析、维度校验、三角属性校验、分块tiling规划与生命周期状态管理。针对不同稀疏格式设计对应分块策略，结合arch22的L2 Cache大小与AI Core数量，将三角求解任务按行/列维度分块，分配到多个AI Core并行执行；Analysis阶段缓存格式、tiling参数与状态，Solve阶段直接复用分析结果，降低重复开销。

任务均分：根据矩阵规模、非零元分布与核心数量动态调整分块大小，优先保证满核运行；非均匀稀疏分布场景下通过负载均衡策略调整各核心任务量，避免核心空闲。

批量搬运：按tile粒度批量搬运稀疏矩阵的values、indices与向量数据，结合double buffer机制实现数据预取与计算重叠，掩盖访存延迟；尾块处理逻辑确保不完整分块正确融入计算流程。

###### 1. 分核策略
优先使用满核的原则。
- 若分块总数可被核心数均分，所有核心处理的数据块数一致，无大小核区分；
- 若无法均分，将剩余的数据块分配到前几个核心上，保证核心间任务量差异最小。

输入数据大小计算：通过矩阵描述符获取维度m、非零元数nnz、稀疏格式与数据类型，结合单元素字节数计算总数据量；向量数据按维度m与类型长度计算总长度。
UB内存大小和核心数量获取：通过平台信息接口获取UB内存大小与AI Core数量，并根据数据规模动态调整实际使用的核心数量。

###### 2. 数据分块和内存优化策略
充分使用UB空间的原则。
综合考虑arch22的UB大小、是否开启double buffer、kernel侧临时工作区与索引缓存需求，计算单核内可承载的分块规模。
UB内存大小获取：通过`GetCoreMemSize`函数获取UB内存大小，用于后续分块大小计算。
Tile块计算：结合L2 Cache容量与稀疏矩阵平均行/列密度，计算每个tile覆盖的行/列数与非零元规模，保证热点数据常驻Cache。
数据切分：按行/列维度对稀疏矩阵进行切分，对应向量同步分块，计算每个core需要处理的tile块数与最后一个块的剩余数据量。
设置切分参数：将分块参数、格式信息、三角属性、操作类型、workspace地址等写入tiling结构体，传递至kernel侧。

###### 3. tilingkey规划策略
需要感知host侧信息对kernel侧走不同分支的场景，通过tilingkey标记分支，消除运行时判断开销。
- 按稀疏格式分为CSR、CSC、COO、SLICED_ELL四个分支
- 按数据类型分为FP32、complex64分支
- 按操作类型分为非转置、转置、共轭转置分支
host侧解析输入描述符的属性，组合生成对应tilingkey值。

##### 3.2.2 kernel侧设计
进行Init和Process两个阶段，其中Process包括数据搬入（CopyIn）、计算（Compute）、搬出（CopyOut）三个阶段。
1. 数据类型适配：FP32直接参与向量计算；complex64的实部与虚部分别执行对应运算，共轭转置场景下对矩阵元素执行共轭操作。
2. CopyIn阶段：按tile粒度搬运稀疏矩阵的values、indices与向量数据到UB；完成未排序索引规范化、重复坐标合并等预处理；支持double buffer双缓冲预取，实现数据搬运与计算重叠。
3. Compute阶段：根据tilingkey执行对应分支的三角求解逻辑，实现向前/向后替换算法；支持单位对角语义（UNIT模式下对角按1计算）、原地求解；采用向量化指令，通过固定遍历顺序保证bitwise确定性结果。
4. CopyOut阶段：将求解结果向量搬移至Global Memory的输出地址；支持尾块非对齐场景的正确搬运，不产生越界访问。
5. 流水线设计：采用CopyIn→Compute→CopyOut三级流水架构，结合双缓冲机制最大化AI Core利用率。

### 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| :--- | :--- |
| Atlas A2（910B3/910B4） | √ |
| Atlas A3 系列 | √ |

### 算子约束限制
1. A必须为二维三角方阵，fill mode之外的非目标三角项按三角语义忽略
2. NON_UNIT模式下缺失或零对角的数值结果按接口定义传播INF/NAN
3. Analysis到Solve期间，matA/vecX/vecY描述符、参数与externalBuffer必须保持一致，参数变更需重新执行Analysis
4. 主求解和格式相关计算由NPU Kernel执行，禁止CPU fallback代替NPU实现
5. 仅支持稠密向量X/Y，不支持稀疏向量输入输出
6. Device端索引使用I32，维度与索引范围在Host侧校验并返回对应错误码

## 可维可测分析
### 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| :--- | :--- | :--- |
| 精度标准 | 符合《生态算子开源精度标准》；FP32 rtol=2^-10、atol=2^-16，整体元素匹配率≥0.99；complex64实部虚部分别执行同等标准 | 生态算子开源精度标准 |
| 性能标准 | A2/A3全场景性能倍率达到0.25倍GPU标杆以上 | 任务书性能要求 |
| 内存标准 | 输入输出总量超500MB时，NPU较GPU额外内存不超过GPU总内存的50%；或workspace不超过目标硬件L2 Cache容量 | 任务书内存要求 |

### 兼容性分析
算子公开接口与`ops-sparse`库现有接口规范完全一致，与`include/cann_ops_sparse.h`中定义的函数原型逐字对齐，上层调用无需修改即可无缝兼容；Host公共逻辑与arch22差异层解耦，可与arch35等其他平台路径共同维护，具备良好的平台可扩展性。