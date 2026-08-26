# 需求背景（required）

## 需求来源

8月社区任务-aclsparseSpGemm 算子开发(A2/A3)。

## 背景介绍

### 现状

PyTorch 的稀疏矩阵乘法入口 `torch.sparse.mm(mat1, mat2)` 在两个乘数均为稀疏张量时分派到
`aten::_sparse_sparse_matmul`。该算子在 CPU 与 CUDA 后端均有实现，在昇腾 NPU 后端为空白，
稀疏×稀疏乘法无法在 NPU 上执行。

`ops-sparse` 仓已按现有 SpGEMM 社区任务规划出 `aclsparseSpGEMM*` 多阶段 C++ 接口及对应
Ascend C Kernel，数据类型范围为 `float16`、`bfloat16` 和 `float32`。

### 差距

| 层次 | 现状 | 本任务需要补齐 |
| --- | --- | --- |
| Python / ATen | NPU 上无 `aten::_sparse_sparse_matmul` 实现 | 注册 NPU 实现，完成稀疏 layout 转换与稀疏输出构造 |
| aclsparse C++ | 多阶段接口按规划交付，覆盖三种实数类型 | 新增 `complex64`，补齐 A2/A3 上四种类型的异常处理与测试能力 |
| Ascend C Kernel | 实数路径已有实现 | 新增 `complex64` 的乘加、抵消与排序归并 |

`complex64` 是本任务的必选能力，需打通描述符、多阶段接口、Kernel、输出组装与测试全流程。

# 需求分析（required）

## 需求描述

在昇腾 NPU 上实现 `aten::_sparse_sparse_matmul`，使 `torch.sparse.mm` 在稀疏×稀疏输入下的参数、
返回值、dtype、shape、device、稀疏 layout 与异常行为与 PyTorch 2.7 及以上版本保持一致；复用现有
SpGEMM 社区任务规划的 `aclsparseSpGEMM*` 接口与 Ascend C Kernel，新增 `complex64` 支持并补齐
Atlas A2 系列产品与 Atlas A3 系列产品上四种数据类型的功能、异常处理与测试能力。核心计算在 NPU 上完成，
不使用 CPU fallback。

## 需求拆解

1. 注册 `aten::_sparse_sparse_matmul` 的 NPU 实现，覆盖 COO 与 CSR 稀疏 layout 输入。
2. 复用规划交付的七个 `aclsparseSpGEMM*` 接口，不新增同名或同功能接口。
3. 新增 `complex64`（`ACL_COMPLEX64`）全链路支持。
4. `float16`、`bfloat16`、`float32`、`complex64` 打通 Python、ATen、aclsparse 与 Kernel 四层。
5. 输出 C 为规范化 CSR：`rowOffsets` 单调非降、每行列索引严格升序、重复坐标累加合并、显式零保留并计入
   `nnz(C)`；Python 层返回 COO 时为 coalesced 状态。
6. 支持规格内的 `M`、`K`、`N`、`nnz(A)`、`nnz(B)` 与中间乘积数量动态变化，覆盖空行、空列、无交集乘积、
   中间乘积膨胀与长尾行分布。
7. 精度按《生态算子开源精度标准》的混合容差单标杆方法验收，同时校验稀疏结构与 values。
8. 与同期分发的 A5 任务共存：公共 Host 逻辑与硬件差异解耦，两套代码在同一主干并行。

# 详细设计（required）

## 算子分析

### 数学公式

$$
C = A \times B
$$

A 为 `[M,K]` 稀疏矩阵，B 为 `[K,N]` 稀疏矩阵，C 为 `[M,N]` 稀疏输出矩阵。输出稀疏结构、`nnz(C)` 与
values 由乘法结果确定。

C++ 接口按 cuSPARSE SpGEMM 语义携带 alpha/beta，完整形式为 $C = \alpha \cdot A \times B + \beta \cdot C$；
ATen 入口固定 `alpha = 1`、`beta = 0`。

按行展开的计算形式决定了实现结构：C 的第 $i$ 行是 A 第 $i$ 行的每个非零元 $a_{ik}$ 与 B 第 $k$ 行整行的
乘积之和。

$$
C_{i,:} = \sum_{k \in \mathrm{nz}(A_{i,:})} a_{ik} \cdot B_{k,:}
$$

因此 C 的第 $i$ 行由 $\mathrm{nnz}(A_{i,:})$ 个有序段归并而成，段的总长度即该行的中间乘积数。

### 支持数据类型

| 数据类型 | `aclDataType` | Golden 计算类型 |
| --- | --- | --- |
| float16 | `ACL_FLOAT16` | float32 |
| bfloat16 | `ACL_BF16` | float32 |
| float32 | `ACL_FLOAT` | float64 |
| complex64 | `ACL_COMPLEX64` | complex128 |

A、B、C 与 `computeType` 采用同一类型。索引类型：`csrRowOffsetsType` 与 `csrColIndType` 均仅支持
`ACL_SPARSE_INDEX_32I` 且必须相同。

cuSPARSE 文档将 float16 与 bfloat16 同精度路径标记为 deprecated，本任务仍对齐并支持这两条路径。

### 支持形状与边界

| 项 | 范围 |
| --- | --- |
| 维度 | A、B 均为二维，`A.size(1) == B.size(0)` |
| 广播 | A、B 不做矩阵维度广播 |
| 空输入 | `nnz(A)=0`、`nnz(B)=0`、空行、空列、无交集乘积、`nnz(C)=0` 均为合法输入 |
| 列索引 | 输入 A/B 每行列索引严格升序，取值位于 `[base, cols + base)` |
| 规模上限 | `M`、`K`、`N`、`nnz` 及中间乘积数量以 `INT32_MAX - 1` 为界；列号判据在 $2^{24}$ 以内使用向量路径，超出时退回标量路径 |

## 算子原型

### C++ 多阶段接口

接口命名、签名与调用顺序沿用现有 SpGEMM 社区任务规划及 `include/cann_ops_sparse.h` 的声明，本任务不新增
同名或同功能接口。

```C
/* 描述符 */
aclsparseStatus_t aclsparseSpGEMMCreateDescr(aclsparseSpGEMMDescr_t *descr);
aclsparseStatus_t aclsparseSpGEMMDestroyDescr(aclsparseSpGEMMDescr_t descr);

/* 阶段 1：工作估算 */
aclsparseStatus_t aclsparseSpGEMMWorkEstimation(
    aclsparseHandle_t handle,
    aclsparseOperation_t opA, aclsparseOperation_t opB,
    const void *alpha,
    aclsparseConstSpMatDescr_t matA,
    aclsparseConstSpMatDescr_t matB,
    const void *beta,
    aclsparseSpMatDescr_t matC,
    aclDataType computeType,
    aclsparseSpGEMMAlg_t alg,
    aclsparseSpGEMMDescr_t spgemmDescr,
    size_t *bufferSize1, void *externalBuffer1);

aclsparseStatus_t aclsparseSpGEMMGetNumProducts(
    aclsparseSpGEMMDescr_t spgemmDescr, int64_t *numProds);

/* 阶段 2：内存估算 */
aclsparseStatus_t aclsparseSpGEMMEstimateMemory(
    aclsparseHandle_t handle,
    aclsparseOperation_t opA, aclsparseOperation_t opB,
    const void *alpha,
    aclsparseConstSpMatDescr_t matA,
    aclsparseConstSpMatDescr_t matB,
    const void *beta,
    aclsparseSpMatDescr_t matC,
    aclDataType computeType,
    aclsparseSpGEMMAlg_t alg,
    aclsparseSpGEMMDescr_t spgemmDescr,
    float chunkFraction,
    size_t *bufferSize3, void *externalBuffer3,
    size_t *bufferSize2);

/* 阶段 3：计算 */
aclsparseStatus_t aclsparseSpGEMMCompute(
    aclsparseHandle_t handle,
    aclsparseOperation_t opA, aclsparseOperation_t opB,
    const void *alpha,
    aclsparseConstSpMatDescr_t matA,
    aclsparseConstSpMatDescr_t matB,
    const void *beta,
    aclsparseSpMatDescr_t matC,
    aclDataType computeType,
    aclsparseSpGEMMAlg_t alg,
    aclsparseSpGEMMDescr_t spgemmDescr,
    size_t *bufferSize2, void *externalBuffer2);

/* 阶段 4：结果拷贝 */
aclsparseStatus_t aclsparseSpGEMMCopy(
    aclsparseHandle_t handle,
    aclsparseOperation_t opA, aclsparseOperation_t opB,
    const void *alpha,
    aclsparseConstSpMatDescr_t matA,
    aclsparseConstSpMatDescr_t matB,
    const void *beta,
    aclsparseSpMatDescr_t matC,
    aclDataType computeType,
    aclsparseSpGEMMAlg_t alg,
    aclsparseSpGEMMDescr_t spgemmDescr);
```

各阶段职责与本任务补齐范围：

| 阶段 | 接口 | 本任务补齐范围 |
| --- | --- | --- |
| 描述符创建 | `aclsparseSpGEMMCreateDescr` | `complex64` 对应的描述符状态与生命周期验证 |
| 工作量估算 | `aclsparseSpGEMMWorkEstimation` | `complex64` 在 A2/A3 及各声明算法下的工作量与 workspace 估算 |
| 中间乘积数量查询 | `aclsparseSpGEMMGetNumProducts` | 四种类型下的查询与边界行为验证 |
| 内存估算 | `aclsparseSpGEMMEstimateMemory` | `complex64` 在 ALG2/ALG3 及其他声明算法下的内存估算 |
| 结构及数值计算 | `aclsparseSpGEMMCompute` | `complex64` 的 Kernel、精度与泛化能力 |
| 结果拷贝及组装 | `aclsparseSpGEMMCopy` | `complex64` 的 values 拷贝与 CSR 结构组装 |
| 描述符销毁 | `aclsparseSpGEMMDestroyDescr` | 完整调用链无资源泄漏 |

关键参数语义：

| 参数 | 说明 |
| --- | --- |
| `opA` / `opB` | 仅支持 `ACL_SPARSE_OP_NON_TRANSPOSE`；传入 TRANSPOSE 或 CONJUGATE_TRANSPOSE 返回明确错误 |
| `alpha` / `beta` | 指针内存位置由 `aclsparseSetPointerMode` 控制；`complex64` 按 `aclsparseComplex` 读 8 字节 |
| `alg` | `ACL_SPARSE_SPGEMM_ALG_DEFAULT`、`ALG1`、`ALG2`、`ALG3` |
| `bufferSize1` / `externalBuffer1` | 阶段 1 的 workspace，保存 B 行段表与各核统计，需存活到 Copy 结束 |
| `bufferSize2` / `externalBuffer2` | 阶段 3 的中间结果 workspace，容量由中间乘积数量决定 |
| `bufferSize3` / `externalBuffer3` | 阶段 2 的估算辅助 workspace |
| `spgemmDescr` | 跨阶段状态载体，记录已完成阶段、`numProducts` 与各 buffer 绑定 |

`nnz(C)` 在 Compute 之后由 `aclsparseSpMatGetSize` 取得，调用方据此为 C 分配输出存储并通过
`aclsparseCsrSetPointers` 写回描述符，然后调用 Copy。

### ATen 接口

```text
aten::_sparse_sparse_matmul(
    Tensor self,
    Tensor other
) -> Tensor
```

Python 公开入口 `torch.sparse.mm(mat1, mat2)`。

| 参数名 | 输入/输出 | 描述 | shape | 数据类型 | 稀疏 layout |
| --- | --- | --- | --- | --- | --- |
| self / mat1 | 输入 | 稀疏矩阵 A | `[M,K]` | float16、bfloat16、float32、complex64 | COO 或 CSR，C++ 层转为 CSR |
| other / mat2 | 输入 | 稀疏矩阵 B | `[K,N]` | 同 self | COO 或 CSR，C++ 层转为 CSR |
| 返回值 | 输出 | 稀疏矩阵 C | `[M,N]` | 同输入 | self 为 COO 时返回 coalesced COO，为 CSR 时返回 CSR |

两个输入须同 device、同 dtype，不做 dtype 提升。

## 算子实现

### 总体架构

一次完整计算由三个 Kernel 与两个 Host 同步点组成：

```
WorkEstimation ──> work kernel     ──> buffer1: B 行段表 + 各核 products
                   work_reduce     ──> numProducts
      [Host 同步]  GetNumProducts
EstimateMemory ──> 按 numProducts 计算 bufferSize2
Compute        ──> compute kernel  ──> buffer2: 各核私有区 (col, value) + 局部 rowOffsets
                   compute_reduce  ──> nnz(C) + 各核输出基址
      [Host 同步]  SpMatGetSize，调用方分配 C 存储
Copy           ──> copy kernel     ──> C.rowOffsets / colIndices / values
```

两处 Host 同步由接口语义决定：`numProducts` 与 `nnz(C)` 都必须返回给调用方用于分配内存。其余阶段
全部异步提交到调用方 stream，Host 侧不额外同步。

### 分核与 Tiling

按 A 的行划分：每个 AIV 核处理一段连续行区间，使各核的输出区互不重叠，无需跨核归约或原子操作。
行区间按中间乘积数均衡，而非按行数均衡——SpGEMM 的每行代价由该行的段总长度决定。

TilingData 随 kernel 按值下发，不经过 workspace。它的内容是 Host 侧整数运算结果加上本次调用的
alpha/beta。走 workspace 需要在每次下发前插一次同步的小包 Host-to-Device 拷贝，该拷贝时延与问题规模
无关，在小规模用例上会成为主要开销；按值下发同时省掉每个核 Init 阶段对 TilingData 各字段的全局内存
标量读。

TilingData 携带：`m`、`k`、`n`、`nnzA`、`nnzB`、`nnzCInput`、`nnzC`、`valueType`、`pointerMode`、
`blockDim`、alpha/beta 的值或 Device 地址，以及 workspace 内各段的字节偏移。

### Workspace 设计

**buffer1（WorkEstimation 产出，存活至 Copy 结束）**

| 段 | 容量 | 用途 |
| --- | --- | --- |
| DeviceResult | 64 B | `numProducts` 与状态码 |
| WorkStat | 64 B × blockDim | 每核的 products、状态、最长 A 行、buffer2 内起始下标 |
| Segment 表 | 8 B × nnz(A) | 每个 A 非零元对应的 B 行段 `(start, length)` |

段表把「按列号查 B 的行区间」这一随机访问从 Compute 阶段前移到 Work 阶段，Compute 顺序读取即可。
每核的统计记录独占一条 64 B 缓存行，避免标量写回时的伪共享。

**buffer2（EstimateMemory 定容，Compute 写入）**

| 段 | 容量 |
| --- | --- |
| 局部 rowOffsets | `int32 × (m + 1)` |
| ComputeStat | 64 B × blockDim |
| DeviceResult | 64 B |
| 中间结果列索引区 | `int32 × numProducts` |
| 中间结果值区 | `elemBytes × numProducts` |

各核在中间结果区按其 products 前缀获得一段私有区（64 B 对齐），容量为本核的中间乘积数，是本核
`nnz(C)` 的上界，因此写入不会越界，也不需要核间协调。`beta != 0` 时容量还需计入参与累加的 C 输入 nnz。

### Host 侧设计

**参数校验**：句柄与描述符非空、格式为 CSR、索引类型均为 `ACL_SPARSE_INDEX_32I` 且相同、`opA`/`opB`
为 NON_TRANSPOSE、A/B/C 与 `computeType` 同类型、维度满足 `A.cols == B.rows`、`A.rows == C.rows`、
`B.cols == C.cols`。

**阶段状态机**：描述符记录已完成阶段。乱序调用（未 WorkEstimation 即 Compute、未 Compute 即 Copy）、
workspace 与登记不符、`bufferSize` 小于所需值，均返回明确错误码而不下发 kernel。

**错误传播**：CSR 结构错误只有 Device 侧能发现。Kernel 把状态码写入 workspace 的统计记录，Host 在
下一个需要同步的阶段（`GetNumProducts`、`SpMatGetSize`）读取并转换为接口错误码返回。状态码区分
结构非法、规模溢出与未实现路径。

### Kernel 侧设计

**阶段 1：WorkEstimation**

每核流式校验其行块的 CSR 结构，并写出该行块每个 A 非零元的 B 行段。

- 结构校验：行偏移单调非降、首尾与 nnz 相符、列索引落在 `[0, cols)` 且行内严格升序。逐块向量化判定——
  块内列号的最小值与最大值管范围，相邻差的最小值管升序，每块只回读三个标量，取代每个非零元一次的
  标量读与比较。行首处的相邻差跨行，逐行改写为合法值；行数远少于非零元数，这一处保持标量。判据用
  fp32 表达，故要求 `cols` 不超过 $2^{24}$，超出时走标量版本。
- 段查找：按 A 的列号读 `rowOffsets(B)` 的相邻两项得到 `(start, length)`。`rowOffsets(B)` 常驻 UB 一段
  固定窗口（96 KiB），`k + 2` 不超过该容量时整表常驻，否则每核检查其行块能否由一个连续窗口覆盖；窗口
  外的行直接读 GM。查找结果的整理走向量路径：`(start, end)` 用 Gather 抽成两个连续向量，一次 Sub 得
  长度，一次 Gather 交织成 `(start, len)` 写入段表。乘积数与边界校验由向量累加器承接，每若干组才回收
  一次，把每个非零元的 UB 标量读写降到常数分之一。
- `work_reduce` 单核归约得到 `numProducts`，并按 products 前缀为各核算出 buffer2 内的起始下标。

**阶段 3：Compute**

每核对本核行块逐行做符号与数值归并，写入本核私有区。

主路径为批量向量排序归并：每批取若干乘积 slot，按 32 slot 一组（Sort32 粒度），每行占整数个组，段起点
按组边界对齐。排序键取一个固定偏置减去列号，按键降序即列升序，补齐用的占位列键为负自然排到行尾；
键值域限制在 fp32 精确整数区间内。排序后按索引 Gather 取 B 值、按段号查表取 A 值，向量乘法得到全部
乘积；相同列的乘积用分段前缀和归并，再压缩掉非行末重复项。合并行的段号由一条按段长拉伸的斜坡算出：
段长相等是合并的前提，因此段号等于本行的段号基址加上「槽位序号整除段长」，三条向量指令即可写出整行，
与段数无关。

单行乘积数超出批容量的长行走游标流式归并：每个 A 非零元一个游标位于 UB，线性扫描取最小列，逐列
输出。该路径与主路径产生相同结果，作为长行的慢路径保留。

两条路径都：同列乘积累加合并为一个条目；计算产生的显式零保留并计入 `nnz(C)`；输出列索引严格升序。
`beta != 0` 时 C 的输入行作为一路额外的有序段参与同一次归并。

`compute_reduce` 单核归约得到 `nnz(C)` 并写各核输出基址。

**阶段 4：Copy**

每核把局部 rowOffsets 加上本核输出基址写到 `C.rowOffsets`，把私有区的列索引与 values 连续搬到 C 的
输出数组。私有区与输出区都按核有序，搬运是连续块拷贝。

**GM 写约定**：所有 GM 写经 UB staging 后由 MTE3（`DataCopyPad`）发出，各核写入区域互不重叠。

### 数值策略

- 累加一律在 fp32 上进行（`complex64` 为两路 fp32），与 `computeType` 的存储宽度无关；fp16/bf16 在写回
  前按 RNE 舍入并保留 NaN。
- `complex64` 在交织域上计算，实部与虚部分别累加；乘加、抵消与排序归并按复数语义处理。
- 相同输入重复执行时，分核方式、行内段顺序与归并顺序均由 CSR 结构唯一确定，不依赖运行期调度，因此
  结果按位一致，满足确定性要求。
- 长行累加提供固定顺序的 Kahan 补偿求和以抑制舍入与抵消误差；任一操作数非有限时退化为直接相加，
  保证 INF/NaN 的传播语义不被补偿项改变。

### PyTorch / ATen 设计

适配层把 aclsparse 接口注册为 `PrivateUse1` 后端实现，`import` 即完成注册，不导出额外 Python API。

| ATen 算子 | dispatch key |
| --- | --- |
| `aten::_sparse_sparse_matmul` | `SparsePrivateUse1`、`SparseCsrPrivateUse1` |

- **输入归一**：未 coalesce 的 COO 先 coalesce；`complex64` 不在 NPU 原生 coalesce 的支持类型内，改用
  key 排序 + `unique_consecutive` + 实虚分量上的 `index_add_` 归并。COO 到 CSR 的 rowOffsets 由
  `index_add_` 与 `cumsum` 在 NPU 上生成，不回落 Host。
- **调用序列**：`alpha = 1`、`beta = 0`、`opA = opB = N`，按 WorkEstimation → GetNumProducts →
  EstimateMemory → Compute → SpMatGetSize → Copy 执行，全部提交到当前 NPU stream；两处按接口语义
  同步 Host。
- **输出构造**：第一个输入为 COO 时返回 coalesced COO，行索引由 int32 rowOffsets 经 `searchsorted`
  展开为 int64；为 CSR 时返回 CSR。输出结构与 CPU 后端一致：列升序、无重复列、显式零保留。
- **共享入口路由**：当前 PyTorch 版本下 `torch.sparse.mm(CSR, CSR)` 先进入 `aten::_sparse_addmm`，
  该入口由同一扩展中的 SpMM 适配注册。因此在该入口按第二个乘数的 layout 分派：稠密乘数走 SpMM，
  稀疏乘数转交 SpGEMM。转交路径要求 `beta = 0`，否则返回明确错误。
- **workspace**：用 `at::empty(uint8)` 在 NPU 上分配，由 PyTorch 缓存分配器按 stream 顺序回收。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas A2 系列产品 | √ |
| Atlas A3 系列产品 | √ |

两系列均映射到 arch22，共用同一份 Kernel 实现。本任务与同期分发的 A5 任务共同修改 SpGEMM 的 Host 侧
公共代码，公共逻辑与硬件差异按目录与分支解耦，保证两个硬件范围的代码在同一主干共存。

## 算子约束限制

| 项 | 约束 |
| --- | --- |
| 稀疏格式 | A、B、C 均仅支持 CSR，用 `aclsparseCreateCsr` 创建；Python 层在调用前完成 layout 转换 |
| 布局 | SpGEMM 三个矩阵均为 CSR，不适用稠密矩阵的 Row-major / Column-major 参数 |
| 操作类型 | `opA`、`opB` 仅支持 NON_TRANSPOSE；其他取值返回明确错误 |
| 索引 | 仅 `ACL_SPARSE_INDEX_32I` 且两者相同；输入列索引必须有序，非法索引返回确定错误 |
| 维度 | A、B 为二维且 `A.size(1) == B.size(0)`；不做广播 |
| 数据类型 | 四种类型打通全链路，A/B/C 与 `computeType` 同类型 |
| 输出结构 | 规范化 CSR，重复坐标累加合并，显式零保留，C++ 层输出无重复坐标 |
| 非连续 Tensor | 稀疏 values 与元数据按上述规格验收，未支持场景返回明确错误 |
| 融合 | 作为独立稀疏算子实现，不要求图融合 |
| 异步执行 | C++ 接口使用调用方 stream，除接口语义要求的两处外不做 Host 同步 |
| 确定性 | 满足确定性计算要求 |
| 规模限制 | 最大 shape、nnz、中间乘积数量、workspace、输出存储与索引溢出边界在接口文档中明确 |

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | 混合容差单标杆；同时校验稀疏结构与 values | 《生态算子开源精度标准》、任务书精度要求 |
| 性能标准 | 每个 case×dtype 场景倍率 > 0.25×A100，全部场景算术平均 ≥ 0.35×A100 | 任务书性能要求 |

**精度判据**：以 CPU Golden 为单标杆，fp16/bf16 用 float32 计算 Golden，fp32 用 float64，`complex64` 用
complex128。`rowOffsets`、`colIndices` 与 `nnz(C)` 须精确一致；values 逐元素按
`|actual - golden| ≤ atol + rtol × |golden|` 判定，整体匹配率不低于 0.99，且每个元素的绝对误差不超过
`max(A, 32 × ULP(golden))`。

| dtype | rtol | atol | A |
| --- | --- | --- | --- |
| float16 | $2^{-9}$ | $2^{-9}$ | 1e-1 |
| bfloat16 | $2^{-6}$ | $2^{-6}$ | 1e0 |
| float32 | $2^{-10}$ | $2^{-16}$ | 1e-2 |
| complex64 | 实部与虚部分别按 float32 参数 | | |

确定性算法重复执行按结构与 values 的 bit-wise 规则验收。INF/NAN 按精度标准文档的对应规则验收。

**性能口径**：标杆为 NVIDIA A100 cuSPARSE SpGEMM 的 NCU Kernel 总耗时；NPU 侧采集相同调用范围内所有
Kernel 的总耗时并计算倍率。每个 case 至少预热 10 次、正式采样 30 次，报告中位数与 90% 分位，每轮计时前
执行设备同步。描述符与已申请 workspace 在采样期间复用，计时不含首次编译、数据生成、Host 到 Device 搬运
与无关初始化。另需分别报告 work estimation、memory estimation、compute、copy、C++ 完整流程与 Python
端到端耗时，以及峰值 workspace、中间乘积数量、`nnz(C)` 与输出存储量。

性能输入使用任务书给定的确定性生成规则：规模 $n \times n$、每行 $d$ 个非零元时，A 的第 $i$ 行列索引为
$(i+a) \bmod n$，B 的第 $i$ 行列索引为 $(i + b \cdot d) \bmod n$，其中 $a, b \in [0, d)$；$d^2 < n$ 时
`nnz(A) = nnz(B) = n·d`、`nnz(C) = n·d²`。values 全取 1，`alpha = 1`、`beta = 0`。

按任务书要求，功能与精度提交 Ascend A2 与 Ascend A3 上的测试结果，性能提交 Ascend A3 上的测试结果；
无法全量执行的组合在验收前说明并取得确认。

## 测试覆盖

测试同时覆盖 Python/ATen 端到端、aclsparse C++ 多阶段接口、Ascend C Kernel 与输出稀疏结构。

| 类别 | 覆盖设计 |
| --- | --- |
| 基础功能 | CSR 方阵、长矩阵、宽矩阵 × 四种 dtype |
| 稀疏边界与输出 | `nnz=0`、`nnz=1`、空行、空列、无交集乘积、多项归并；逐项校验 `nnz(C)`、rowOffsets、colIndices、values |
| 布局限制 | A/B/C 均为 CSR；仅验证 cuSPARSE 13.3 Update 1 的 SpGEMM 官方规格所支持的 CSR 与 operation 组合 |
| 多阶段流程 | CreateDescr → WorkEstimation → EstimateMemory → Compute → Copy → DestroyDescr 主路径，按声明算法分别执行 |
| Workspace 与异常 | 一个精确 workspace 用例、一个 workspace 不足用例；维度、dtype、索引不匹配用例 |
| 阶段状态机 | 乱序调用、重复调用、workspace 与登记不符 |
| C++ 接口契约 | 每条错误路径的返回码逐项核对；输入 A、B 在完整调用链前后逐字节不变；输出缓冲区与 workspace 末尾设哨兵，验证无越界写 |
| 数值场景 | 普通值、小值、正负混合、零值、数值抵消、离群值、规格允许的 INF/NAN |
| 确定性 | 同一输入重复执行，结构与 values 按位比对 |
| 资源 | 连续创建、执行、销毁描述符，检查内存与资源泄漏 |
| ATen 与泛化 | COO 与 CSR 输入、未 coalesced COO；对 shape、nnz、稀疏度做代表性组合抽样 |
| Dispatch 与 Profiler | 核对 ATen 分派命中 NPU 注册而非 CPU fallback，并由 Profiler 采集确认核心计算 Kernel 在 NPU 上执行 |

Kernel 侧的分核、段表与归并路径通过构造性用例区分：长尾行分布触发长行慢路径，规则行分布触发向量排序
归并路径，段长相等与不等的行分别覆盖段号构造的两条分支。

**测试环境**：Ascend A2 / Ascend A3；CANN 版本以算子开源仓指定版本为准，实际使用版本记录在自测报告中；
PyTorch 2.7 及以上；torch_npu 26.0.0 及之后。

## 兼容性分析

复用现有 SpGEMM 社区任务规划的接口声明与调用流程，不新增、不修改公开接口签名，对已有 float16、
bfloat16、float32 调用方保持源代码与行为兼容。新增的 `complex64` 是既有类型分支的扩展，不改变其他类型的
执行路径。若设计评审需要调整原型，调整结果与现有 SpGEMM 社区任务同步，保持源代码兼容或给出明确的兼容
方案，并同步写入公开头文件与接口文档。
