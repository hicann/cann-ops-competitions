# 需求背景（required）

## 需求来源

- 社区任务：9月社区任务-aclsparseSparseToDense算子开发(A2/A3)（任务书随社区任务 `04_tasks/01_community-task-2026` 配套发布）。
- 交付目标：参考 cuSPARSE SparseToDense 接口语义，在 ops-sparse 仓以 aclsparse Legacy API 工程模式新增 `aclsparseSparseToDense` 的 Atlas A2/A3（DAV_2201，arch22）实现，代码合入 `sparse/sparse2dense/arch22/`、`test/sparse2dense/arch22/`。
- 文档版本：v1.1（2026-09-05，按社区任务设计模板 `resources/design_template.md` 重组；v1.0 为 2026-09-03 spec 驱动初版，修订人 2403_89192208）。

## 背景介绍

### aclsparseSparseToDense 算子 arch22 实现新增

基于上游 ops-sparse 仓 aclsparse Legacy API 模式（公开接口声明 + Host 校验/分发 + Ascend C Kernel，非 aclnn 两段式），使用 Ascend C 通用 SIMD/MemBase 编程路线新增 DAV_2201（arch22）实现：

- **目标架构**：DAV_2201（arch22），覆盖 Atlas A2 训练系列（Ascend 910B3、910B4）与 Atlas A3（任务环境型号，报告期记录实际型号）；A2/A3 共享同一份 Host 分发与 Kernel 实现。
- **上游现状**：`sparse/sparse2dense/arch35/` 已有 A5（DAV_3510）的 SIMT（`__simt_vf__`）实现。DAV_2201 无 SIMT 硬件，arch22 Kernel 须按 SIMD/MemBase 重写，仅复用 arch35 的 Host 结构与「清零 + scatter」双 Kernel 协议及 tiling 数据结构（m/n/indexBase/valueType/isColMajor/ld/perBlock/format/nnz）。
- **路线排除依据**：SIMT 线程级随机写硬件自 DAV_3510 起才有；RegBase 为 DAV_3510 专属能力；Memory 矢量 `Scatter`（ISASI）官方支持表对 A2/A3 训练系列为 ×（由 `DataCopyPad` 手法替代）。

### aclsparseSparseToDense 公开接口现状分析

公开接口在 `include/cann_ops_sparse.h` L633–679 为既有声明，本任务**零签名修改**复用（命名以头文件 `_bufferSize` 为准，任务书伪代码 `aclsparseSparseToDenseGetBufferSize` 的命名差异以头文件为准）：

```C
typedef enum aclsparseSparseToDenseAlg_t {
    ACL_SPARSE_SPARSETODENSE_ALG_DEFAULT = 0
} aclsparseSparseToDenseAlg_t;

aclsparseStatus_t aclsparseSparseToDense_bufferSize(
    aclsparseHandle_t handle, aclsparseConstSpMatDescr_t matA,
    aclsparseDnMatDescr_t matB, aclsparseSparseToDenseAlg_t alg,
    size_t *bufferSize);
aclsparseStatus_t aclsparseSparseToDense(
    aclsparseHandle_t handle, aclsparseConstSpMatDescr_t matA,
    aclsparseDnMatDescr_t matB, aclsparseSparseToDenseAlg_t alg,
    void *buffer);
```

接口参数（两接口共用同一参数表）：

| 参数 | 属性 | 类型 | 说明 |
|---|---|---|---|
| `handle` | 输入 | `aclsparseHandle_t` | 上下文及执行 stream；空 handle 返回参数错误 |
| `matA` | 输入 | `aclsparseConstSpMatDescr_t` | 只读稀疏源描述符（CSR/CSC/COO，I32 索引，base 0/1） |
| `matB` | 独立输出 | `aclsparseDnMatDescr_t` | 稠密输出描述符（ROW/COL 布局，dtype 与 matA 一致） |
| `alg` | 属性 | `aclsparseSparseToDenseAlg_t` | 仅 `ACL_SPARSE_SPARSETODENSE_ALG_DEFAULT`，非法枚举返回参数错误 |
| `bufferSize` | 独立输出（bufferSize 接口） | `size_t*` | 执行所需 Device workspace 字节数，非负且不溢出（arch22 恒返回 0） |
| `buffer` | 输入 workspace（执行接口） | `void*` | 按查询值分配的 Device workspace；查询为 0 时可为空 |

张量层输入输出（数量与名称与 spec.yaml 一致）：输入 3 个——`values`（len=NNZ）、`primary`（CSR rowOffsets / CSC colOffsets / COO rowInd，I32）、`secondary`（CSR colInd / CSC rowInd / COO colInd，I32）；输出 1 个——`dense`（shape=(rows, cols)，dtype=values.dtype，ND）。`values`/`primary`/`secondary` 与稀疏结构只读，未声明的输入输出 alias 返回参数错误。

### aclsparseSparseToDense 功能分析

本算子将只读稀疏矩阵 `matA`（CSR / CSC / COO 格式，I32 索引，index base 0/1）转换为同 shape（`rows×cols`）的 ROW/COL 布局稠密矩阵 `matB`：

1. **清零（正零填充）**：对 `matB` 全部逻辑元素写对应 dtype 的**正零** bit 模式（INT8 为 `0x00`，浮点为符号位 0 的全零，complex64 实/虚两分量均为 +0f）；
2. **scatter（唯一坐标写）**：对每个稀疏坐标 `(row,col)`（存储坐标减 `index_base` 得 0-based 逻辑坐标）写 `B[row,col] = values[p]`，**纯 bit 搬运**，无算术、无类型转换；
3. 未被坐标覆盖的稠密逻辑位置保持正零；ld padding 区域按 DnMat 的 ld 语义保持边界安全，不写逻辑元素（ROW 布局存储寻址 `B[r,c]=dense[r*ld+c]`，COL 布局 `B[r,c]=dense[c*ld+r]`）。

关键语义边界：输入稀疏结构（offsets/indices）与 values **只读**；输入坐标可**乱序**但经 base 换算后必须**唯一**（重复坐标返回参数错误）；转换**确定性**（同输入重复执行 bit-wise 一致）；`alg` 仅支持 `ACL_SPARSE_SPARSETODENSE_ALG_DEFAULT`。

调用形态为 aclsparse C++ **双阶段接口**（对标 cuSPARSE `cusparseSparseToDense_bufferSize` / `cusparseSparseToDense`）：第一阶段查询 workspace 字节数，第二阶段在 `handle` 绑定的调用方 stream 上**异步**执行清零 + scatter（核心路径在 A2/A3 NPU 完成，禁止 CPU fallback）。

用户可见调用入口仅两类：① aclsparse C++ 双阶段调用（核心交付）；② Python/ATen 适配入口（必选交付）——公开入口 `Tensor.to_dense`，内部映射 ATen Dispatcher `aten::_to_dense`，适配层完成参数/dtype/shape/layout/stride/device 校验与 sparse tensor 元数据转换后调用 `aclsparseSparseToDense` 并构造 dense 输出；支持 PyTorch 2.7+、torch_npu 26.0.0+；不支持组合显式报错，不得 CPU fallback。本算子不提供 ACLNN 单算子调用入口，也不支持图模式调用。

# 需求分析（required）

## 需求描述

使用 Ascend C 编程语言在 arch22（DAV_2201）实现 `aclsparseSparseToDense` 算子：支持 int8、float16、bfloat16、float32、complex64 五种 values/dense 数据类型，支持 CSR/CSC/COO 三种稀疏格式、ROW/COL 两种稠密布局、index base 0/1，提供 `aclsparseSparseToDense_bufferSize` / `aclsparseSparseToDense` 双阶段接口（bufferSize 恒 0 协议），精度对 CPU Golden 逐元素 exact match（bit-wise），性能达到 0.25 倍性能标杆以上，并完成 Python/ATen NPU 适配。

## 需求拆解

1. 复用公开头文件既有接口声明，零签名修改；
2. 支持 5 dtype（int8/float16/bfloat16/float32/complex64，dense.dtype = values.dtype，无类型转换）× 3 格式（CSR/CSC/COO）× 2 布局（ROW/COL）× 2 base（0/1）全交叉；
3. Host 校验链：空 handle/描述符/指针、非法 format/index base/layout/ld/alg、dtype 不一致、重复坐标、offsets 非单调、越界、溢出防护、未声明 alias 防护，非法输入返回参数错误且不发起 Kernel；
4. Kernel：清零（K1，全 buffer 含 ld padding 写正零）+ scatter（K2，唯一坐标 bit 搬运写 values）双 Kernel；CSC×ROW 与 CSR×COL 转置组合采用「桶转置 / 批量散布直写」分层加速；
5. 精度：bit-wise exact（rtol=0/atol=0），complex64 实/虚部分别 bit-wise 一致，正零填充，NaN/±INF/±0 原样搬运转发；
6. 确定性：坐标唯一 ⇒ 每输出元素至多写一次，无写冲突、无归约，同输入重复执行 bit-wise 一致；输入只读；
7. 内存：`aclsparseSparseToDense_bufferSize` 恒返回 0，无与完整稠密输出成比例的 Device 临时副本，满足任务书内存双规则；
8. 性能：P-01/P-02/P-03 每个有效 case 性能倍率 ≥ 0.25（预热 ≥10 次、采样 ≥30 次，中位数口径）；
9. Python/ATen 适配：PrivateUse1 注册，无 CPU fallback，int64 索引 Device 侧显式 I32 cast。

# 详细设计（required）

## 算子分析

### 数学公式

以 spec.yaml `math_semantics.formula` 为唯一真值源。

**记号**：稀疏矩阵 $A \in T^{m \times n}$（$T$ 为支持 dtype），非零元集合 $S=\{(r_p,c_p,v_p)\}_{p=0}^{nnz-1}$，$v_p=\text{values}[p]$；`primary` = CSR rowOffsets / CSC colOffsets / COO rowInd，`secondary` = CSR colInd / CSC rowInd / COO colInd（均 I32）；Host 尺寸元数据 `rows/cols/ld` 为 int64。0-based 逻辑坐标与存储坐标的 **base 换算**：$\tilde r = r^{stored}-base,\ \tilde c = c^{stored}-base$（offsets 同减 base）。目标稠密矩阵 $B \in T^{m\times n}$，DnMat 寻址：ROW 布局 $B[r,c]=\text{dense}[r\cdot ld+c]$，COL 布局 $B[r,c]=\text{dense}[c\cdot ld+r]$。

**（1）CSR/CSC/COO 坐标展开语义**：

$$\text{CSR: } \forall \tilde r\in[0,m),\ \forall k\in[\text{rowOffsets}[\tilde r]-base,\ \text{rowOffsets}[\tilde r+1]-base):\ (\tilde r,\ \text{colInd}[k]-base,\ \text{values}[k])\in S$$

$$\text{CSC: } \forall \tilde c\in[0,n),\ \forall k\in[\text{colOffsets}[\tilde c]-base,\ \text{colOffsets}[\tilde c+1]-base):\ (\text{rowInd}[k]-base,\ \tilde c,\ \text{values}[k])\in S$$

$$\text{COO: } \forall p\in[0,nnz):\ (\text{rowInd}[p]-base,\ \text{colInd}[p]-base,\ \text{values}[p])\in S$$

坐标可乱序但唯一：$(r,c)$ 经 base 换算后互不相同（重复坐标返回参数错误）；CSR/CSC offsets 单调不减。

**（2）写入与正零填充**：

$$\forall (r,c)\in[0,m)\times[0,n):\quad B[r,c]=\begin{cases}\text{values}[p] & \exists p,\ (r_p,c_p)=(r,c)\\ +0_T & \text{否则}\end{cases}$$

仅做 bit 搬运，不执行数值计算或类型转换；$+0_T$ 为对应 dtype 的**正零** bit 模式（INT8 为 0x00，浮点为符号位 0 的全零），complex64 实/虚两分量均为 +0f。

**（3）确定性**：数学语义为对每个稀疏坐标独立写入目标稠密位置；坐标唯一 ⇒ 每个输出元素至多被写一次，无归约、无写冲突，同一输入重复执行结果 bit-wise 一致（默认确定性实现）。

> 说明：spec 公式中的 `np.clip / np.resize` 仅保证 spec 生成阶段随机 smoke 输入下公式可执行，对合法输入是 no-op；真实输入由 Host 校验保证（offsets 单调不减、坐标经 base 换算后唯一且在界内，越界/重复由 Host 拒绝并返回参数错误）。

### 支持数据类型

- values / dense：`int8`、`float16`、`bfloat16`、`float32`、`complex64`（dtype_policy.fixed，dense.dtype = values.dtype，无类型转换，纯 bit 搬运）；
- 索引/offsets：仅 `int32`（I32，`ACL_SPARSE_INDEX_32I`）；Host 尺寸元数据 rows/cols/nnz/ld 为 `int64`；
- complex64 无原生复数类型，按 **2×float32（8 字节，实/虚相邻）** 搬运与清零（float/uint32 通道）。

### 支持形状

- `rows/cols ∈ [1, 2147483647]`；`ld`：ROW 布局 `ld ≥ cols`、COL 布局 `ld ≥ rows`；`nnz ≤ INT32_MAX`（I32 索引契约）；
- `len(primary)`：CSR = rows+1 / CSC = cols+1 / COO = nnz；
- broadcast：none（无广播分支）；所有输入张量 rank = 1（ND）；
- NNZ=0 合法：输出全 +0（由清零 Kernel 承接输出全定义）。

## 算子实现

### 实现方案

#### 技术路线

最终路线为**通用 SIMD/MemBase**（Ascend C 向量核，TPipe/TQue 队列框架，MTE 搬运主导，纯访存无矢量算术），**双 Kernel**：K1 清零（+0 填充）+ K2 scatter（唯一坐标写 values）。本算子为 Sparse/ScatterUpdate 范式的纯访存型算子（无算术、无归约），主计算形态决定 MemBase 路线最优；SIMT/RegBase 路线因硬件不适用被排除（见背景介绍），Memory 矢量 `Scatter` 指令官方支持表为 ×，以 `DataCopyPad` 手法替代。

| 备选路线 | 结论 |
|---|---|
| RegBase | 不适用：DAV_3510 专属，DAV_2201 无此硬件 |
| SIMT（`__simt_vf__` 线程级随机写） | 不可用：SIMT 硬件 DAV_3510 起才有 |
| Memory 矢量 `Scatter`（ISASI） | 不可用：官方支持表 A2/A3 训练系列为 ×，由 `DataCopyPad` 手法替代 |
| Host 清零（`aclrtMemset`）+ Device scatter | 备选不采用：基线选 Kernel 清零，利于 NPU stream 语义统一 |

架构约束：UB/核 192 KB（`GetCoreMemSize` 运行时确认，禁止硬编码）；UB 起始地址 32B 对齐，`DataCopy` 主体 32B 对齐最优，`DataCopyPad` 支持字节粒度；核数一律 `GetBlockIdx()` / `GetBlockNum()` 动态获取。清零与 scatter 均在 handle 绑定 stream 上 NPU 执行，禁止 CPU fallback。

#### host侧设计

**Host 分层**（执行接口内部顺序）：`CheckMeta`（元数据静态校验）→ `ValidateF3`（坐标校验链）→ `FillTiling`（TilingData/TilingKey 计算）→ `GetBufferSize`（恒 0 协议）→ 同 stream 先 K1 后 K2 异步 launch。

1. **CheckMeta**：空 handle / 空描述符 / 空指针、非法 format/index base/layout/alg 枚举、rows/cols/ld 越域（rows/cols ∈ [1, 2³¹−1]、ROW ld≥cols / COL ld≥rows）、dtype 不在支持组合或 matA 与 matB dtype 不一致、len 匹配（lenPrimary = rows+1 / cols+1 / nnz）、未声明 alias 防护（matB.dense 与 matA.values 同 buffer → 参数错误）。错误码映射：len 匹配 / offsets 非单调 / 坐标越界 / 重复坐标 / ld 违反 → shape_mismatch；dtype 非法或不一致 → dtype_not_supported；format/base/layout/alg 非法枚举或维度越域 → attribute_value_out_of_range；空指针类 → null_input；统一返回 `ACL_SPARSE_STATUS_INVALID_VALUE` 语义的参数错误。
2. **ValidateF3（坐标校验链，发起任何 Kernel 之前）**：将 primary/secondary（I32）经 `aclrtMemcpyAsync`（D2H，handle 绑定 stream，pinned buffer）拷至 Host 校验缓冲（规模 (lenPrimary + nnz) × 4B）并同步该 stream，随后校验 offsets 单调不减、坐标经 base 换算后唯一（排序/哈希）且落在 [0,rows)×[0,cols) 内，失败返回参数错误且**不发起 K1/K2**。该机制与两条约束不冲突：① 非 CPU fallback——清零与 scatter 仍全部在 NPU 执行，Host 仅做输入校验（工程模式「Host 校验/分发」明示职责）；② stream 异步语义不变——同步点位于执行接口内部、先于 K1/K2 发起，接口返回后 K1/K2 仍在 stream 上异步执行。备选（不采用）：Device 侧预检 kernel + 4B 标志回传可降 D2H 开销，但唯一性校验需 O(nnz) 额外 workspace，与 bufferSize 恒 0 决策冲突。NNZ=0 时按守卫跳过空 vector 的 D2H（values/secondary data 可空）。PyTorch 适配层另引入 F3 两级记忆化（设备侧 FNV-1a 64 指纹守卫内核 + TensorImpl 身份缓存，未命中回退全量 F3，校验语义不降级）。
3. **溢出防护**：zeroTotalBytes 等地址/字节计算用无符号 64 位中间量 + 显式判溢出（rows=cols=ld=2³¹−1 且 elemSize=8 时 ≈3.69×10¹⁹ > int64 上界，溢出返回参数错误）；COO blockNum 显式校验 nnz ≤ UINT32_MAX × tileElems。

##### 1. 分核策略：

优先使用满核的原则，核数一律 `GetBlockIdx()` / `GetBlockNum()` 动态获取（910B3/910B4 为 20 核、A3 型号运行时确认，禁止硬编码）。

- **K1 清零**：各核按 `GetBlockIdx()` 对 GM 清零区间 [begin, end)（字节）做动态分片，**各核边界按 32B 粒度向上取整对齐**（不足 32B 的余量并入尾侧），保证每核 begin 32B 对齐、对齐主体走 `DataCopy` 最优路径；区间不足以整分时部分核区间为空。
- **K2 scatter**：块（段块/元素块）→ 核由 grid-stride 动态认领，Host 不下发 per-core/per-block 映射，也**不预扫描 offsets**（无 D2H 预扫描）；块划分仅依赖 segNum/segsPerBlock（CSR/CSC）或 nnz/tileElems（COO），**与元素分布/空段分布无关**——空块快速跳过，负载经多块摊薄（P-01 常规场景约 128 块 ≫ 20 核）。

##### 2. 数据分块和内存优化策略：

充分使用 UB 空间的原则，双缓冲 BUFFER_NUM=2。

- **两级分片（CSR/CSC）**：主轴为**段块**——blockNum = ceil(segNum / segsPerBlock)（TilingData 下发），块 b 覆盖段 [b×segsPerBlock, min((b+1)×segsPerBlock, segNum))，块内元素区间 [eBegin, eEnd) 由块端点 offsets 标量读定位（各 2 次 4B GM 标量读，与上游 arch35 同构）；二级为**段内元素 tile**（tileElems=1024 基准），单段 nnz 可远超 tileElems（极端：单行全密集 nnz=cols）由该段所在块的内层 tile 循环承载。**COO** 无段头，两级退化为元素块一级分片：blockNum = ceil(nnz/tileElems)。
- **offBuf 确定上界**：块 offsets 区间项数 = 段块段数 + 1 ≤ segsPerBlock + 1，**与 nnz/空段分布无关**（spec 允许任意空行/空列：相邻 offsets 相等只压缩块内元素区间、不扩大 offsets 读取规模），消除「maxSegsPerTile = min(tileElems+1, segNum)」在空段密集时的越界风险。
- **UB 缓冲构成**（NNZ_TILE=1024、BUFFER_NUM=2）：K1 零块 2×8KB=16KB；K2 基线实例 values/secondary/offBuf/packBuf 分片双缓冲合计 ≈56KB（complex64 8B 通道复核 ≈72KB）；**三实例运行时三选一互斥分配**（K2 Init 依 tiling 元数据判定，TPipe 按分支分配，物理上不并存）——基线 ≈56KB、CSC×ROW 桶转置实例最坏 ≈168.5KB（fp32，含 16 窗 bucketTile 128KB + rowMin/occList 16KB）、V1 批量散布直写实例 ≈88.5KB（含 packSlot 64KB）；单 launch 峰值 = max(…) = 168.5KB ≤ 184KB 保守线 ≤ 192KB（`GetCoreMemSize` 运行时复核）。运行时护栏：实测可用 < 184KB 时 W_CO 降档（16→15→12）并由联合成本门自动收窄激活域，被收窄 case 落入 V1 层，零劣化保证。
- **L1 不使用**（纯访存搬运、无数据复用需求）；Device workspace 固有申请为 0（bufferSize 恒 0 协议，见下）。

##### 3. tilingkey规划策略：

**TilingKey 数量 = 5，维度 = dtype**，与 spec `dtype_policy.supported_combinations`（5 条组合）一一对应；TilingKey 分发一律用模板参数 `ASCENDC_TPL_ARGS_DECL` / `ASCENDC_TPL_SEL_PARAM`（编译期 `if constexpr`）选择 K1/K2 的 dtype 实例（elemSize 1/2/4/8B 与造零/搬运通道），禁用废弃宏 `TILING_KEY_IS` / `BEGIN_TILING_DATA_DEF`。

| TilingKey | dtype 组合（values=dense，primary/secondary=int32） | 搬运通道（elemSize） | 说明 |
|---|---|---|---|
| TK0 | int8 | 1B；清零走 **uint16 通道造零**（一次 uint16 零块覆盖 2×int8） | `Duplicate` 不含 int8；bit 模式仍为 0x00，正零语义不变 |
| TK1 | float16 | 2B（half） | `Duplicate` 支持 half |
| TK2 | bfloat16 | 2B（bfloat16_t） | `Duplicate` 支持 bfloat16_t |
| TK3 | float32 | 4B（float/uint32） | — |
| TK4 | complex64 | 8B（2×float32，float/uint32 通道） | 与 fp32 分列，便于校验与测试矩阵对齐 |

**TilingKey 不编码的维度**：format（CSR/CSC/COO）为 K2 内部运行时分支；layout（ROW/COL）、index_base（0/1）为 TilingData 寻址公式参数与坐标减法；rows/cols/ld/nnz/lenPrimary 为 TilingData；坐标唯一/越界/offsets 单调由 Host 校验保证；broadcast none 无广播分支；dtype promotion fixed 无 cast 步骤。若后续实测 format 分支开销在 P 场景敏感，可提升进 TilingKey（则 15 个），基线不采用。

**TilingData 字段**（与 arch35 tiling 数据同构，新增 zero*/tileElems/segsPerBlock/blockNum/segNum 支撑 MemBase 分片）：

| 字段 | 类型 | 语义 |
|---|---|---|
| rows / cols / ld / nnz / lenPrimary | int64_t | 尺寸元数据（Host 侧 int64；lenPrimary：CSR=rows+1 / CSC=cols+1 / COO=nnz） |
| format / indexBase / isColMajor / valueType | uint8_t | 0=CSR/1=CSC/2=COO；0/1；0=ROW/1=COL；dtype 枚举 |
| zeroTotalBytes / zeroTileBytes | int64_t | K1 清零总字节数（（ROW ? rows : cols）× ld × elemSize，含 padding，Host 无符号 64 位判溢出）/ 每块零块字节数 |
| tileElems / segsPerBlock / blockNum / segNum | uint32_t | K2 二级 tile（1024）/ 段块粒度（64；R3+ 起按目标块数 ≥128 在 [8,64] 自适应收缩）/ 段块数 / 段数（COO=0） |
| workspaceSize | uint64_t | 恒 0 |

**bufferSize 恒 0 协议**：`aclsparseSparseToDense_bufferSize` 恒返回 0（与 arch35 语义一致、交叉回归安全）——清零直写 matB、无 Device 临时副本，不存在与完整稠密输出成比例的临时副本；查询为 0 时执行接口 `buffer` 可空，非零需求时空指针/无效地址由 Host 校验返回参数错误。固有 workspace 绝对值 0 ≤ L2 Cache 容量，任务书内存规则之二平凡满足。

#### kernel侧设计

进行 Init 和 Process 两个阶段；Process 数据面为「GM↔UB 搬入（MTE2）→ Scalar 索引处理 → UB 打包 → 搬出（MTE3）」。两个模板均以 TilingKey 模板参数实例化（elemSize 1/2/4/8B 与通道），K2 内 format/isColMajor/indexBase 为运行时参数。

**运行期数据流**（同 stream 先 K1 后 K2，异步执行；读取结果前调用方自行同步 stream）：

```text
【清零 kernel K1（先执行）】
  UB：造 +0 零块（Duplicate，按 dtype 位宽通道 1/2/4/8B 参数化，双缓冲）
   │  MTE3：DataCopy 大块写（32B 对齐主体）+ DataCopyPad 尾块
   ▼
  GM dense：matB 全 buffer（按 DnMat 布局、含 ld padding 区）动态分片写正零

【scatter kernel K2（后执行）】
  GM values(NNZ) / primary(LEN_PRIMARY) / secondary(NNZ)   ← 只读输入
   │  MTE2：DataCopyPad 分片读入 → UB（VECIN，双缓冲）
   ▼
  UB：base 换算（坐标减 indexBase）→ 连续 run 识别与合并（Copy/SetValue 打包）
      → 寻址换算（ROW：dense[r*ld+c]；COL：dense[c*ld+r]）→ VECOUT 打包分片
   │  MTE3：DataCopyPad（DataCopyExtParams）随机地址段写出
   ▼
  GM dense：唯一坐标位置写 values（bit 搬运）
```

**K1 清零模板**：

1. **Init**：TPipe 分配零块缓冲（双缓冲 BUFFER_NUM=2，每缓冲 zeroTileBytes）；
2. **分片**：`GetBlockIdx()` / `GetBlockDim()` 动态划分 GM 清零区间（字节），各核边界按 32B 粒度向上取整对齐；区间不足以整分时部分核区间为空；
3. **造零（按模板参数通道）**：float32 → `Duplicate<float>`；float16/bfloat16 → `Duplicate<half>` / `Duplicate<bfloat16_t>`；int8 → uint16 通道造零（一次覆盖 2×int8）；complex64 → float/uint32 通道（8 字节实/虚均 +0f）；
4. **写出**：对齐主体 `DataCopy` 大块写（32B 对齐最优路径），尾块（非 32B 对齐/字节余数）`DataCopyPad` 字节粒度写；奇字节尾块（如 rows=cols=ld=1 的 int8）由 `DataCopyPad` 承接；
5. **输出写回**：直接写 matB GM（含 ld padding 区），无 Device 临时副本；
6. **边界**：rows/cols ≥ 1 → zeroTotalBytes ≥ elemSize，不存在空清零；NNZ=0 时仅 K1 执行即满足输出全定义（全 +0）。

**K2 scatter 模板（CSR/CSC/COO 运行时分支）**：

1. **Init**：TPipe + `TQue<VECIN/VECOUT>` 双缓冲，分配 values/secondary 分片缓冲、CSR/CSC 块 offsets 区间缓冲（offBuf）或 COO primary 逐点分片缓冲与打包缓冲；CSR/CSC 另以 TBuf 持有锚点 anchor = offsets[0]（kernel 启动后一次 4B GM 标量读）；
2. **分片**：段块主轴 × 段内元素 tile 两级（CSR/CSC）/ 元素块一级（COO），块→核 grid-stride 动态认领（见 host 侧设计）；
3. **读入（MTE2）**：`DataCopyPad` 读 values/secondary 内层 tile 分片 → VECIN；CSR/CSC 每块在首个内层 tile 前一次性读入块 offsets 区间（≤ segsPerBlock+1 项 → offBuf，块内复用）；
4. **坐标解析（Scalar pipe）**：段定位/展开以 **anchor = offsets[0] 为锚点的相对差值**承接——元素 j 属于段 s ⇔ offsets[s] − anchor ≤ j < offsets[s+1] − anchor（**禁止用 raw offsets 与元素下标直接比较**，base=1 时 rowOffsets[0]=1 直接比较会整体 off-by-one）；CSR：行号 = 段号、列 = secondary − indexBase；CSC：列号 = 段号、行 = secondary − indexBase；COO：行/列 = primary/secondary − indexBase；rowOffsets/colOffsets 为长度语义**不参与 base 换算**（不减 base）；
5. **寻址**：ROW 布局 GM 偏移 = (r×ld+c)×elemSize；COL 布局 GM 偏移 = (c×ld+r)×elemSize；ld padding 区不参与逻辑比对；
6. **run 识别与打包**：相邻元素目标 GM 地址连续 → run；run 内元素 UB→UB `Copy` 打包（32B 对齐快速路径，非对齐退 `SetValue` 逐元素）；随机孤立点按 run=1 承接；**run 不跨内层 tile**（块边界天然隔离），拆分不改变写入内容与次数，bit-wise 确定性不受影响；
7. **写出（MTE3）**：每 run 一次 `DataCopyPad`（DataCopyExtParams 字节粒度）写 matB GM，随后 `PipeBarrier<PIPE_MTE3>`；
8. **尾块与边界**：块内尾 tile/非整块由 `DataCopyPad` 字节粒度承接；NNZ=0 时 K2 Init 后直接返回；空段（相邻 offsets 差值 0）跳过、不产生 run；空块整体跳过；lenPrimary 按 format 语义核对（CSR=rows+1 / CSC=cols+1 / COO=nnz）；
9. **确定性**：坐标唯一（Host 保证）→ 每输出元素至多写一次、无写冲突；run 合并仅改变搬运粒度、不改变写入内容与次数；块/内层 tile 划分仅依赖 segNum/segsPerBlock 或 nnz/tileElems，与元素分布无关 → 同输入重复执行 bit-wise 一致；
10. **转置组合分层加速**：CSC×ROW（及对称的 CSR×COL）在基线路径下同列相邻元素目标 GM 地址步长 ld 完全非连续 → run 恒为 1、退化为逐元素 Scalar SetValue + 逐元素 DataCopyPad（实测 ~270–293ns/元素/核，K2 恒 2.9–3.9ms 主导）。K2 对该组合按**三层机制 + 元数据门**分发（判定顺序，K2 Init 单点判定，Host FillTiling 同式复算，纯元数据、不改 TilingKey）：

```text
S2D_K2_Path(tiling 元数据, GetBlockNum()):
  if nnz == 0:                     → K2 直接返回（K1 承接全定义，先于一切路径判定）
  if not transposed                → P0 基线（CSR/COO run 合并、CSC×COL colreg；零改动）
     （transposed = (CSC×ROW) ∪ (CSR×COL)，R3+ 对称扩展）
  elif valueType == complex64:     → P0 基线（complex64 逐元素）
  elif est ≤ 238ns:                → P1 桶转置层（单遍多窗聚合）
  else:                            → P2 V1 批量散布直写层
```

**P1 桶转置层（CSC×ROW / CSR×COL，联合成本门 est ≤ 238ns 激活）**——门公式（Host FillTiling 与 K2 Init 共用 `S2DBucketGateActive` 同式纯元数据判定，单一公式两处一致）：

```text
C       = 32 / elemSize                                  # 列子块宽 = 对齐粒度 A（fp32=8 / 2B=16 / int8=32）
W_total = ceil(winDim / 256)                             # 窗口总数，rowTile=256
W_eff   = ceil(W_total / 16)                             # W_CO=16 窗共存 → 批次数（=扫描次数）
seq     = ceil(blockNum / coresUsed) × ceil(64 / C) × W_eff   # 每核批序列数
elems   = nnz / coresUsed                                # 每核元素数
est     = W_eff×33ns + 70ns + seq×7μs/elems + 20ns       # S/H/F/M 实测中值
bucketActive = transposed && (valueType != complex64) && (est ≤ 0.85×280ns = 238ns)
```

机制（单遍多窗聚合）：① 列子块 C=A=32/elemSize（行宽恒 32B、窗口 tile 统一 8KB）；② 行窗口多窗共存 rowTile=256、W_CO=16（16×9KB=144KB）；③ 单遍扫描聚合——每（子块×批次）16 窗 tile 清零（+0 与 K1 幂等）后重扫子块元素一遍，每元素解码行号 → `SetValue` 直写对应窗口 tile，u16 rowMin 首触哨兵（0xFFFF）去重 + occList 紧凑追加；④ 每批一次 `PipeBarrier<PIPE_ALL>` 后按 occList 对每占用行一次**恒 32B** `DataCopyPad` 段写（gap 为 +0，与 K1 清零幂等）+ `PipeBarrier<PIPE_MTE3>`。确定性：门与参数只依赖 tiling 元数据（同输入同路径）；元素→（子块,窗口,行内偏移）映射与 occList 追加序（存储序）均为输入的确定函数；跨核写区间两两不交；段写 gap 恒 +0 幂等。P-01 fp32 桶机制上界 ≈0.20–0.23 < 0.25（窗口状态下界 ≈295KB ≫ K2 可用 159.5KB，W_eff≥2 不可消除），如实判不可达 → 由 P2 层承接。

**P2 V1 批量散布直写层（承接桶门排除域 est > 238ns，探针 NPU 实证合入）**——块内两阶段循环：**阶段 A（打包）**：元素 tile（1024）循环，MTE2 双缓冲读入 (secondary, values) 分片；**列号按段摊销解码**（列游标沿 offBuf 每段解码一次，P-01 ≈18 元素/段）；每元素 Scalar：r ← secondary−indexBase、gmOff ← (r×ld+c)×elemSize（64 位），向槽写入——槽布局 32B/元素：[value@w0（对应 dtype 通道 SetValue）, gmOff 低/高 32 位@w1–w2]；**屏障对（每 tile 一次）**：`PipeBarrier<PIPE_ALL>`；**阶段 B（发射）**：按槽序每元素一次单 burst `DataCopyPad` 直写 dense[r×ld+c]（blockLen=elemSize，GM 任意字节地址、UB 槽基址 32B 对齐）→ `PipeBarrier<PIPE_MTE3>`。不引入合并/多 burst（探针 V2/V3 裁决：合并簿记 +71ns/元素负收益；多 burst 段写被 API 实证证伪）。写域恰覆盖坐标位置、不触碰 ld padding，gap 由 K1 清零承接。complex64 CSC×ROW 从逐元素基线并入 V1（K7 增补：8B 值通道槽位扩展为 [valLo@u0, valHi@u1, offLo@u2, offHi@u3]，槽距恒 32B 不变；P-02-csc-complex64 端到端倍率 0.243/0.247 → 0.369/0.362）。确定性：三层判定纯元数据互斥完备；槽位序 = 存储序；坐标唯一 → 每元素恰一次 pad；跨核写区间两两不交；探针两次运行全量 memcmp EQ。

**探针实证（probe_csc_batched_scatter，NPU 910B3 / CANN 9.1.0 / 40 核）**：

| 变体 | 机制 | ns/元素/核 | 裁决 |
|---|---|---:|---|
| V0 anchor | 每元素 pad + 逐元素 PIPE_ALL（基线同款） | 170.9 | 对照锚点：V0−V1=4.1ns → 屏障归因证伪 |
| **V1** | 列段分组摊销 + 32B 槽位 + tile 级屏障对 + 每元素 4B pad | **166.8** | **✅ 达标 → 合入主线** |
| V2 | 相邻列同行 8B 段合并 | 238.3 | uniform 负收益 → 不并入 |
| V3 | 列段内纵向 run 多 burst 段写 | — | API 证伪（blockLen=4B 仅首 burst 落盘）→ 判死 |

**R3+ 对称扩展（泛化用例收口）**：转置组合 CSR×COL 与 CSC×ROW 结构对偶，桶/V1 两层路径对称扩展至 CSR×COL——门条件泛化为 `transposed`、窗口批维 winDim 按 layout 取 rows/cols、GM 寻址泛化为 `(secIdx×ld+segIdx)×elemSize` 单点表达式；host tiling 段块粒度自适应（目标块数 ≥128，2 的幂，[8,64]）修复「段数小 + 行分布偏斜」下元素压进个别块的单核包揽慢点。复测：泛化 12 例慢点 0.022–0.185 → 0.251–1.775 全达标（最高 36×），P-01/02/03 共 90 例无回退。

**API 映射与验证记录**（全部条目以官方文档 + 上游生产代码为证据源，不虚标状态）：

| 计算步骤 | Ascend C API | 验证状态 | 约束说明 |
|---|---|---|---|
| 动态核数/UB 容量 | `GetBlockIdx`/`GetBlockNum`/`GetCoreMemSize` | 已验证 | 禁止硬编码；DAV_2201 典型 UB 192KB |
| K1 造零块 | `Duplicate` | 已验证 | **不含 int8** → int8 走 uint16 通道；complex64 走 float/uint32 通道 |
| K1 大块写 | `DataCopy` | 已验证 | 主体 32B 对齐最快；尾块交 `DataCopyPad` |
| K1 尾块 / K2 读入 / run 写出 | `DataCopyPad`（DataCopyExtParams） | 已验证 | 字节粒度；GM 偏移按 dtype 位宽对齐（上游 scatter/arch22 生产实证任意元素偏移） |
| K2 run 打包（UB→UB） | `Copy` | 已验证 | 32B 对齐快速路径；非对齐退 `SetValue` 逐元素；**`Copy<int8_t>` 8 位通道静默失效（穿刺实证）** → 8 位打包必须 SetValue |
| K2 标量访问 | `LocalTensor::SetValue/GetValue` | 已验证 | 标量路径仅索引/打包，不承载数据面 |
| 双缓冲与同步 | TPipe/TQue/TBuf、`PipeBarrier` | 已验证 | VECIN/VECOUT；Scalar 写→MTE3 读跨流水需 `PipeBarrier<PIPE_ALL>`（L0_004 竞争实证不可移除） |
| 随机分散写 | `Scatter`（ISASI） | 已验证不支持（排除） | A2/A3 支持表 ×，由 `DataCopyPad` 替代 |
| GM 初始化 / Host 清零 | `InitGlobalMemory` / `aclrtMemset(Async)` | 待验证（不纳入基线） | 仅备选路线；启用前须容器实测补记录 |

#### 并行策略与流水线

- **核间切分**：核数由 `GetBlockIdx()` / `GetBlockNum()` 动态获取，Host 侧 tiling 按 `GetBlockNum()` 计算分片量；清零按 GM 地址区间分片贴近 HBM 写带宽，scatter 按段块/元素块分片、空块快速跳过、负载经 grid-stride 摊薄。
- **双缓冲**：清零与 scatter 的 UB 分片均采用 TPipe/TQue 双缓冲（BUFFER_NUM=2），VECIN/VECOUT position。
- **事件同步与分 pipe**：scatter 读侧走 MTE2、写侧走 MTE3，分 pipe 并以 `PipeBarrier<PIPE_MTE3>` 保证写序，与上游 scatter/arch22 手法一致；UB 内 run 打包用 `Copy`（对齐 32B 快速路径）。
- **Kernel 拆分决策**：基线按「清零 + scatter」两 kernel 设计（与 arch35 协议对齐、复杂度最低）；融合单 kernel 仅作 P-02/P-03 小矩阵性能备选，实测后择一，不阻塞设计。
- 输出写主导路径直接 UB→GM，不经过中间缓存；不创建与完整稠密输出成比例的临时副本。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas 800I/T A2 训练系列（Ascend 910B3、910B4，DAV_2201 / arch22） | √ |
| Atlas A3（任务环境型号，DAV_2201 / arch22，与 A2 同一 Host/Kernel 二进制） | √ |
| Atlas 950 系列（A5，DAV_3510 / arch35） | ×（非本任务目标；A2/A3–A5 交叉回归待 A5 算力可用后补测） |

## 算子约束限制

1. `alg` 仅支持 `ACL_SPARSE_SPARSETODENSE_ALG_DEFAULT`，非法枚举返回参数错误；
2. 索引固定 I32（int32）；rows/cols ∈ [1, INT32_MAX]；nnz ≤ INT32_MAX（int64 输入在 PyTorch 适配层做 Device 侧显式 I32 cast，C++ 接口直接拒绝）；
3. 输入坐标可乱序但经 base 换算后必须唯一，CSR/CSC offsets 单调不减；坐标越界/重复/非单调/锚定破坏（offsets[0]≠base 或 offsets 末端−base≠nnz）由 Host 校验链拒绝（`ACL_SPARSE_STATUS_INVALID_VALUE`）；
4. 输入稀疏结构（offsets/indices）与 values 只读，执行前后 bit 不变；输出与输入同 buffer（alias）返回参数错误；
5. 零 nnz 时 values/secondary 允许空张量（DataPtr 可空），由 K1 全量清零承接输出定义；
6. `aclsparseSparseToDense_bufferSize` 恒返回 0；查询为 0 时执行接口 `buffer` 可空；workspace 仅按查询值分配；
7. 不提供 ACLNN 单算子调用入口、不支持图模式（GE/torch.compile）；Python/ATen 适配无 CPU fallback；
8. A3 型号运行期核数与 UB 容量以 `GetBlockNum()` / `GetCoreMemSize()` 实测为准，报告期记录实际型号；A2/A3–A5 交叉回归待 A5 算力可用后补测（交付分支为纯增量改动、arch35 构建输入零触碰，详见自测报告 §7.1）。

# 可维可测分析（required）

## 精度标准/性能标准

| 验收标准 | 描述(不涉及说明原因) | 标准来源 |
| --- | --- | --- |
| 精度标准 | 对 CPU Golden 逐元素 exact match：全 dtype bit-wise equal（rtol=0/atol=0），complex64 实/虚部分别 bit-wise 一致；未覆盖元素为正零；NaN/±INF/±0 原样 bit 搬运；覆盖 CSR/CSC/COO、ROW/COL、base 0/1、ld padding、乱序唯一坐标、重复执行确定性 | 任务书 §3.2；opbase 精度标准 experimental_standard.md |
| 性能标准 | P-01/P-02/P-03 每个有效 case 性能倍率 ≥ 0.25（GPU 设备 Event 标杆 / NPU 同调用范围总耗时）；预热 ≥10、采样 ≥30，报告中位数/p90/清零 scatter 分段/workspace 峰值/有效带宽/端到端耗时；C++ 标杆与 PyTorch GPU 标杆分别统计 | 任务书 §3.3 |
| 内存标准 | >500 MB 场景额外内存 ≤ GPU 内存总量 50%，或固有 workspace ≤ L2 Cache；bufferSize 恒 0 + 清零直写 matB，无成比例临时副本 | 任务书 §3.4 |

### 测试规划

| 测试体系 | 用例数 | 判定口径 |
|---|---|---|
| C++ UT（Host 校验链/tiling/溢出/编译链接/分片边界） | 74 | gtest 断言 |
| C++ ST Real（真实内核 NPU 执行） | 125（L0 12 / L1 85 / L2 28） | bit-wise exact（golden = CPU canonical 同 dtype 构造） |
| PyTorch 适配层冒烟 | 61 + 3 负例 | torch.equal（bit-wise）+ RuntimeError 命中 |
| 官方精度用例（ATK） | 200 | bitwise_equal（官方 accuracy_sparse_ops.py 原样） |
| 官方性能/内存用例 | 290（P-01/02/03 × 90 + 泛化 200） | 倍率 ≥0.25；内存双规则 |

覆盖场景：全 dtype × 格式 × 布局 × base 交叉、零 nnz、空行/列、极端稀疏度、高密度、最大 shape、溢出、空描述符/指针、非法 shape/dtype/format/index/base/ld/alg、输入只读、stream 异步语义、重复执行确定性、数值边界（正负零/±INF/NaN/complex64 实虚独立）。

## 兼容性分析

新算子新增 arch22 实现，不涉及兼容性分析：

- 公开头文件 `include/cann_ops_sparse.h` 零修改（复用既有 L633–679 声明），无签名/枚举/语义变更；
- 交付分支相对上游 master 为纯增量改动（305 文件 +129,921 行、0 删除，其中算子本体 15 文件 +3,194 行、测试代码与随测试引入的 googletest 源码占其余；arch22 目录外仅新增仓根交付说明），共享 CMake 与上游既有代码零触碰；
- arch35（A5）实现与构建输入零改动，共存破坏风险趋零；`bufferSize` 恒 0 语义与 arch35 一致，交叉回归待 A5 算力可用后补测。
