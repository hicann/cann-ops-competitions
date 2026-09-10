# aclsparseSparseToDense 算子设计文档

本文定义 <code>aclsparseSparseToDense</code> 在 Ascend 950PR A5
平台上的接口语义、Python/ATen 适配、Host 校验、Ascend C Kernel、
workspace、性能优化和测试判定标准。设计遵循任务书、cuSPARSE
SparseToDense 语义和 ops-sparse 公共接口约束。

| 项目 | 内容 |
| --- | --- |
| 任务名称 | 9 月社区任务：aclsparseSparseToDense 算子开发（950） |
| 算子接口 | <code>aclsparseSparseToDenseGetBufferSize</code>、<code>aclsparseSparseToDense</code> |
| 目标硬件 | Ascend 950PR，DAV_3510，<code>arch35</code>，A5 |
| 目标软件 | CANN 9.1.0 及配套版本 |
| Python 适配 | PyTorch 2.7 及以上、torch_npu 26.0.0 及以后 |
| 输入格式 | CSR、CSC、COO |
| 输出布局 | ROW、COL |
| 数据类型 | INT8、FP16、BF16、FP32、complex64 |
| 索引类型 | Device I32，index base 0 或 1 |
| 算子目录 | <code>sparse/sparse2dense/arch35/</code> |
| 测试目录 | <code>test/sparse2dense/arch35/</code> |
| 文档版本 | V1.0 |
| 提交作者 | 待提交人补充 |
| 设计日期 | 2026 年 9 月 8 日 |

## 一、需求描述

本节说明稀疏到稠密转换的数学语义、接口边界和公开使用方式。实现必须
同时满足 C++ 接口和 Python/ATen 入口的语义要求。

### 1.1 需求来源

<code>aclsparseSparseToDense</code> 将 CSR、CSC 或 COO 稀疏矩阵转换为
指定 ROW 或 COL 布局的稠密矩阵。算子参考 cuSPARSE
SparseToDense 的 BufferSize/Convert 接口能力，采用
<code>aclsparse</code> C++ 接口、Host 校验/分发和 Ascend C Kernel
的工程模式。

本任务额外要求完成 PyTorch NPU 适配。公开入口为
<code>Tensor.to_dense()</code>，内部映射到 <code>aten::_to_dense</code>，
再由 NPU Dispatcher 调用 ops-sparse 的实现。稀疏结构、values、清零和
scatter 均在 NPU 路径完成，不允许 CPU fallback 或构造额外 CPU 中间结果。

### 1.2 功能定义

转换先将输出稠密矩阵的所有逻辑元素定义为对应 dtype 的正零，再将每个
唯一稀疏坐标对应的 values 写入输出：

~~~text
B[row, col] = A.values[p]
~~~

其中坐标由 sparse 格式和 index base 决定：

| 输入格式 | primary 数组 | secondary 数组 | 第 p 个 values 的逻辑坐标 |
| --- | --- | --- | --- |
| CSR | row offsets，长度为 rows + 1 | column indices | row 由 offsets 区间确定，col = secondary[p] - base |
| CSC | column offsets，长度为 cols + 1 | row indices | col 由 offsets 区间确定，row = secondary[p] - base |
| COO | row indices，长度为 nnz | column indices | row = primary[p] - base，col = secondary[p] - base |

输入坐标可以按任意顺序排列，但同一逻辑坐标必须唯一。所有写入均为
直接赋值，不执行累加。输入 sparse 结构和 values 只读，输出的每个逻辑
位置都必须有定义。

### 1.3 布局和地址语义

输出矩阵逻辑形状为 <code>rows × cols</code>，<code>ld</code> 是输出
描述符中的 leading dimension。物理地址按下式计算：

~~~text
ROW: B[row * ld + col]   ，ld >= cols
COL: B[row + col * ld]   ，ld >= rows
~~~

ROW 和 COL 只改变输出地址计算，不改变 sparse 坐标的逻辑含义。输出
padding 区域不属于逻辑矩阵，Kernel 不写入该区域。Python/ATen 层创建
COL 输出时使用等价于 shape <code>(rows, cols)</code>、stride
<code>(1, rows)</code> 的 NPU dense tensor；C++ 接口按描述符中的
<code>ld</code> 精确寻址。

### 1.4 数据类型和索引约束

本次实现支持以下 value dtype：

- INT8；
- FP16；
- BF16；
- FP32；
- complex64，实部和虚部均为 FP32。

Device 上的 offsets、indices 和 COO 坐标统一使用 I32。Host 侧
<code>rows</code>、<code>cols</code>、<code>nnz</code> 和
<code>ld</code> 使用 <code>int64_t</code> 保存并在计算字节数、地址和
workspace 时执行溢出检查。index base 只接受 0 或 1，Kernel 在执行地址
计算前减去 base。

输入 values 按原 dtype 写入输出，不发生 dtype 转换或数值重排。对
complex64，已写入的正负零、Inf 和 NaN 按输入 bit pattern 保持；未覆盖
位置写入正零复数。

### 1.5 Python/ATen 接口目标

Python/ATen 适配负责公开入口、Dispatcher 注册、输入检查、描述符构造、
输出构造和异常映射。适配层的调用路径如下：

~~~text
Tensor.to_dense()
    -> aten::_to_dense
    -> NPU Dispatcher
    -> sparse descriptor + dense descriptor
    -> aclsparseSparseToDenseGetBufferSize
    -> aclsparseSparseToDense
    -> NPU dense Tensor
~~~

适配层必须满足以下约束：

- 仅接受 NPU sparse tensor 和声明支持的 CSR、CSC、COO layout；
- 将 sparse tensor 的 values、primary、secondary 指针直接映射到
  Device 描述符，不在 CPU 创建稠密副本；
- 检查 dtype、index dtype、shape、stride、device、layout 和
  index base；
- 输出 dense tensor 与输入 shape 一致，ROW/COL stride 与请求布局一致；
- 不支持的 layout、dtype、index 类型或 device 返回明确的参数错误；
- 输入 sparse tensor 只读，输出不与 sparse storage alias；
- 保持调用 stream 和异步生命周期，不在 ATen 层插入 CPU 同步或 fallback。

### 1.6 C++ 接口定义

公开接口声明统一放入 ops-sparse 的
<code>include/cann_ops_sparse.h</code>，接口签名如下：

~~~cpp
typedef enum aclsparseSparseToDenseAlg_t {
    ACL_SPARSE_SPARSETODENSE_ALG_DEFAULT = 0
} aclsparseSparseToDenseAlg_t;

aclsparseStatus_t aclsparseSparseToDenseGetBufferSize(
    aclsparseHandle_t handle,
    aclsparseConstSpMatDescr_t matA,
    aclsparseDnMatDescr_t matB,
    aclsparseSparseToDenseAlg_t alg,
    size_t *bufferSize);

aclsparseStatus_t aclsparseSparseToDense(
    aclsparseHandle_t handle,
    aclsparseConstSpMatDescr_t matA,
    aclsparseDnMatDescr_t matB,
    aclsparseSparseToDenseAlg_t alg,
    void *buffer);
~~~

<code>GetBufferSize</code> 只查询执行所需 Device workspace 字节数，不
执行清零或 scatter。执行接口使用与查询阶段一致的 descriptor、algorithm
和 shape 参数，并在 handle 绑定的 stream 上完成 NPU 转换。

### 1.7 参数契约

下表定义两个接口的参数方向、数据位置、布局和异常要求。

| 参数 | 方向 | 类型和位置 | 语义 | 合法性及异常 |
| --- | --- | --- | --- | --- |
| <code>handle</code> | 输入；两个接口 | <code>aclsparseHandle_t</code>，Host | 上下文和执行 stream | 空句柄返回参数错误 |
| <code>matA</code> | 输入；两个接口 | <code>aclsparseConstSpMatDescr_t</code> | 只读 sparse 描述符 | 空描述符、非法 format/index/base/shape/dtype 或不满足坐标约束时返回参数错误 |
| <code>matB</code> | 输入；两个接口 | <code>aclsparseDnMatDescr_t</code> | 独立 dense 输出描述符 | 空描述符、shape/dtype/layout/ld 不匹配时返回参数错误 |
| <code>alg</code> | 输入；两个接口 | <code>aclsparseSparseToDenseAlg_t</code>，Host | 转换算法 | 仅接受 DEFAULT，其他值返回参数错误 |
| <code>bufferSize</code> | 输出；查询接口 | <code>size_t*</code>，Host | workspace 字节数 | 空指针、计算溢出返回参数错误 |
| <code>buffer</code> | 输入；执行接口 | Device 指针 | 查询所得 workspace | 需求为零时可为空；需求非零时必须有效且容量足够 |

<code>matA</code> 的 descriptor 必须完整描述 format、rows、cols、nnz、
value dtype、index dtype、index base 以及 values/indices 的 Device 地址。
CSR 的 primary 长度为 <code>rows + 1</code>，CSC 的 primary 长度为
<code>cols + 1</code>，COO 的 primary 和 secondary 长度均为
<code>nnz</code>。

<code>matB</code> 的 value dtype 必须与 <code>matA</code> 一致，shape
必须为 <code>rows × cols</code>。ROW 布局要求
<code>ld >= cols</code>，COL 布局要求 <code>ld >= rows</code>。输出
存储必须与输入 sparse arrays 独立，避免输入只读约束被破坏。

### 1.8 参数校验顺序

Host 端使用统一校验顺序，在提交 Kernel 前拒绝可确定的非法参数。

1. 检查 handle 和两个 descriptor 是否为空。
2. 检查 algorithm 是否为 DEFAULT。
3. 检查 format、value dtype、index dtype、index base 和 layout。
4. 检查 rows、cols、nnz、ld 的非负性、形状关系和整数溢出。
5. 检查 sparse arrays、dense output 和 workspace 指针的必要性。
6. 检查 matA 与 matB 的 shape、dtype 和输出布局是否匹配。
7. 检查 sparse offset 的长度约束、单调性、坐标范围和唯一性约束。
8. 通过检查后计算 workspace 布局和 tiling 参数，并在 handle stream
   上提交清零和 scatter Kernel。

Device index arrays 属于 sparse descriptor 的输入契约。Python/ATen
适配和测试数据构造必须保证 offsets 单调、坐标在 shape 内、坐标唯一；
C++ 接口对能够在提交前确定的非法内容返回参数错误。实现不将 Device
稀疏数组复制到 CPU，也不通过 CPU 扫描替代 NPU 转换。

## 二、方案调研与设计取舍

本节比较 cuSPARSE、PyTorch sparse-to-dense 和候选 NPU 实现方式，说明
为什么采用格式专用的清零加 scatter 方案。

### 2.1 cuSPARSE 语义参考

cuSPARSE SparseToDense 将 sparse descriptor 转换为 dense descriptor，
并通过 BufferSize/Convert 两阶段接口让调用者管理 workspace。本设计
保留这两个阶段，使用 descriptor 表达 sparse format、index base、value
dtype、dense layout 和 leading dimension。

本设计与参考语义保持以下一致：

- CSR、CSC、COO 都表示唯一的二维坐标集合；
- values 直接写入对应 dense 坐标；
- 未覆盖元素为对应 dtype 的正零；
- input arrays 只读；
- BufferSize 阶段不改变输出，Convert 阶段在绑定 stream 上执行。

### 2.2 PyTorch 现状和适配边界

PyTorch 公开的 <code>Tensor.to_dense()</code> 和内部
<code>aten::_to_dense</code> 为 Python/ATen 适配提供入口语义。NPU
适配不重新实现 sparse tensor 的 CPU 逻辑，而是从 Dispatcher 接收 sparse
tensor 的 Device storage，构造 aclsparse descriptors，并调用 NPU 算子。

不支持的 sparse layout、非 I32 索引、非声明 dtype、CPU tensor 或不兼容
stride 必须在 ATen 层明确报错。适配层不把 unsupported case 静默转到
CPU。

### 2.3 候选方案对比

候选方案的比较如下，最终采用“格式专用 Kernel 加共享输出路径”的设计。

| 方案 | 优点 | 风险或代价 | 结论 |
| --- | --- | --- | --- |
| CPU 构造 dense 后搬到 NPU | 参考实现简单 | 违反无 CPU fallback，搬运和内存开销大 | 不采用 |
| 先把所有输入转换为 COO | Kernel 结构统一 | 额外索引副本，破坏大规模场景内存目标 | 不采用 |
| 一个通用 Kernel 处理三种格式 | 代码入口少 | 分支多，CSR/CSC 长尾负载难以平衡 | 不采用 |
| CSR/CSC/COO 分格式 dispatch | 地址规律清晰，便于负载均衡 | 需要维护三种索引遍历路径 | 采用 |
| 清零后直接 scatter | 语义简单，无输出临时副本 | 稠密输出写带宽成为主要成本 | 采用 |
| 全 dense 临时副本再修正 | 方便处理部分路径 | 额外显存与带宽随输出规模增长 | 不采用 |
| 统一 atomic scatter | 可处理潜在重复坐标 | 重复坐标本身不在输入契约内，atomic 影响性能和确定性 | 不采用 |

### 2.4 设计原则

实现遵循以下原则：

1. 用 descriptor 和参数校验保证接口语义，Kernel 只处理通过校验的数据。
2. 把输出清零和 sparse scatter 拆成可独立调度的 NPU 阶段。
3. 只在 Device 上使用输入、输出和查询所得 workspace，不创建 CPU
   中间张量。
4. 使用 format 专用的索引遍历和负载均衡，避免所有格式强行走 COO。
5. 将 dtype、layout、base 和 ld 放入 tiling，保持公开接口稳定。
6. 以输入只读、输出独立和同一 stream 生命周期作为 Python/ATen 与
   C++ 接口的共同约束。

## 三、总体设计与实现方案

本节描述 Python/ATen、C++ Host、workspace 和 Ascend C Kernel 的模块
边界，以及一次转换的执行顺序。

### 3.1 总体架构

一次 NPU 调用经过以下路径：

~~~mermaid
flowchart LR
    P[Tensor.to_dense] --> D[aten::_to_dense / NPU Dispatcher]
    D --> S[构造 sparse descriptor]
    S --> O[构造 dense descriptor]
    O --> Q[GetBufferSize]
    Q --> A[分配或复用 Device workspace]
    A --> H[aclsparseSparseToDense Host]
    H --> V[参数和 descriptor 校验]
    V --> Z[Zero-fill Kernel]
    Z --> K[Format-specific Scatter Kernel]
    K --> W[ROW/COL dense output]
~~~

Python/ATen 层只负责框架对象和 descriptor 的转换。清零、索引解析和
values 写入均在 NPU 上完成。C++ 公开接口直接复用相同的 Host 和 Kernel
实现。

### 3.2 Descriptor 和 workspace 设计

Host 将 descriptor 中的元数据编码为内部 tiling，包含以下字段：

~~~cpp
struct SparseToDenseTilingData {
    int64_t rows;
    int64_t cols;
    int64_t nnz;
    int64_t dense_ld;
    int32_t format;
    int32_t value_dtype;
    int32_t index_base;
    int32_t dense_layout;
    int32_t index_dtype;
    int32_t zero_mode;
    int32_t scatter_mode;
    uint64_t dense_elements;
    uint64_t workspace_bytes;
};
~~~

workspace 只承载控制信息和必要的格式调度元数据，不承载完整 dense
输出。建议的内部区域如下：

| 区域 | 用途 | 规模约束 |
| --- | --- | --- |
| 状态区 | Kernel 阶段状态和错误标志 | 固定小尺寸 |
| 调度区 | CSR/CSC 长行或长列的任务边界 | 与 rows、cols 或 task 数成正比 |
| 对齐区 | 供不同 dtype 和 Kernel 参数对齐 | 固定小尺寸 |

<code>GetBufferSize</code> 根据 format、shape、nnz 和调度策略计算
workspace 字节数，并执行加法、对齐和乘法溢出检查。workspace 的固有
申请量必须不超过目标硬件 L2 Cache 容量；不得将完整 dense 输出或与
<code>rows × cols</code> 同阶的副本放入 workspace。

查询和执行必须使用一致的 descriptor 参数。执行接口重新计算所需字节
数并检查传入 workspace 容量，防止使用过小或与 descriptor 不匹配的
workspace。

### 3.3 执行阶段

有效输入的执行分为两个主阶段：

1. Zero-fill Kernel 按 ROW 或 COL 的物理布局将输出逻辑区域写成对应
   dtype 的正零。
2. Scatter Kernel 根据 CSR、CSC 或 COO 的索引结构，把 values 写入
   对应的输出坐标。

当 <code>nnz == 0</code> 时跳过 scatter，只执行 zero-fill。rows 或 cols
为零时输出逻辑区域为空，Host 返回成功且不读取 sparse values。

### 3.4 Zero-fill Kernel

Zero-fill Kernel 以 dense 输出的物理存储为调度单位，使用连续向量写入
和尾部 mask。ROW 方向优先沿列内的连续地址写入，COL 方向优先沿物理
连续列写入；ld 大于逻辑最小值时，padding 不写入。

每种 value dtype 使用对应的零值：

| dtype | 写入的未覆盖值 |
| --- | --- |
| INT8 | 整数 0 |
| FP16 | 正零 |
| BF16 | 正零 |
| FP32 | 正零 |
| complex64 | 实部和虚部均为正零 |

zero-fill 和 scatter 使用同一 handle stream。scatter 必须在 zero-fill
完成后启动，避免未覆盖位置被后续 sparse 写入流程覆盖。

### 3.5 CSR Scatter Kernel

CSR 的每个 row 由 primary offsets 定义一个连续的 values 区间。Kernel
读取 row offset，遍历该行的 column indices，并按下式写入：

~~~text
start = row_ptr[row] - base
end   = row_ptr[row + 1] - base
col   = col_ind[p] - base
B[row, col] = values[p]
~~~

短行采用一个 task 处理多个 row，长行按 values 区间切分为多个 task。
调度区记录 row 段边界，使长尾 row 不阻塞其他 core。由于坐标唯一，
scatter 采用直接 store，不使用 atomic。

### 3.6 CSC Scatter Kernel

CSC 的每个 column 由 primary offsets 定义一个连续的 values 区间。Kernel
读取 column offset，遍历 row indices，并按下式写入：

~~~text
start = col_ptr[col] - base
end   = col_ptr[col + 1] - base
row   = row_ind[p] - base
B[row, col] = values[p]
~~~

短列使用批量 task，长列按 values 区间切分。ROW 输出仍按照
<code>row * ld + col</code> 写入，COL 输出按照
<code>row + col * ld</code> 写入；输入格式和输出布局彼此独立。

### 3.7 COO Scatter Kernel

COO 的每个 values 元素对应一个 row index 和一个 column index，适合
按 p 直接并行：

~~~text
row = row_ind[p] - base
col = col_ind[p] - base
B[row, col] = values[p]
~~~

Kernel 使用一条 values 记录对应一个输出 store。索引和 values 采用相同
的 p，保证乱序 COO 的坐标和值保持配对。坐标唯一时不需要 atomic，也不
需要排序或临时重排。

### 3.8 确定性和特殊值

输入坐标唯一时，每个输出逻辑位置最多被一个 values 写入，执行顺序不会
改变输出，因此同一输入重复执行必须 bit-wise 一致。Kernel 不对
values 做数学运算，正负零、Inf、NaN 和 complex64 的实虚部分量直接
复制。

未覆盖位置只由 zero-fill 定义为正零。padding 不参与输出语义，测试
可以预填充 padding 后检查其未被写入。

### 3.9 dtype Kernel 设计

Kernel 使用 dtype 模板或等价的编译期分派复用索引逻辑，value store
使用实际 dtype 的向量类型：

- INT8 采用字节向量写入；
- FP16 和 BF16 采用半精度向量写入；
- FP32 采用单精度向量写入；
- complex64 按两个 FP32 分量整体搬运，避免拆分后改变 bit pattern。

所有 dtype 的尾部元素使用 mask。values 只读，不进行隐式类型提升、
截断或饱和转换。

### 3.10 ATen Dispatcher 设计

ATen 适配分为输入解析、descriptor 构造、输出构造和错误映射四个部分：

| 模块 | 设计职责 |
| --- | --- |
| 输入解析 | 判断 sparse layout、dtype、index dtype、device、shape 和 stride |
| descriptor 构造 | 将 CSR/CSC/COO storage 映射为 matA，填入 format/base/nnz |
| 输出构造 | 在 NPU 创建独立 dense tensor，设置 ROW 或 COL stride |
| workspace 管理 | 调用 GetBufferSize，按查询值申请或复用 Device workspace |
| 执行分发 | 调用 aclsparseSparseToDense，并沿用当前 stream |
| 错误映射 | 将参数、dtype、layout、device 和 workspace 错误映射为 ATen 异常 |

输入 sparse tensor 不与输出 dense tensor 共享 storage。若公开入口收到
不支持的 layout 或 dtype，适配层直接报错，不转发到 CPU 实现。

### 3.11 Stream 和生命周期

所有清零和 scatter Kernel 使用 handle 绑定的 stream。以下对象在 stream
完成前必须保持有效：

- matA 的 values、offsets、indices Device arrays；
- matB 的输出 storage；
- workspace Device storage；
- descriptor 所引用的 Host 元数据；
- tiling 和 Kernel 参数。

Host 不在正常有效路径上调用全局同步。Python/ATen 调用返回后，用户在
读取输出或释放相关 Device storage 前，必须遵循当前 NPU stream 的同步
生命周期。

## 四、性能和内存优化方案

本节定义性能测试口径、目标和优化方向。

### 4.1 性能目标和测试范围

性能倍率定义为：

~~~text
性能倍率 = GPU 参考接口设备 Event 调用耗时
         / NPU 同调用范围总耗时
~~~

目标场景如下。每个场景覆盖 3 种 sparse format、5 种 value dtype 和
base 0/1，共 30 个核心组合：

| 场景 | dense shape | sparse pattern | 目标 |
| --- | --- | --- | --- |
| P-01 | 8192 × 28672 | 每行 64 个非零元素 | 所有声明 dtype、format 和 base 的性能倍率至少达到 0.3 |
| P-02 | 4096 × 1536 | 每行 64 个非零元素 | 所有声明 dtype、format 和 base 的性能倍率至少达到 0.3 |
| P-03 | 7168 × 2048 | 每行 64 个非零元素 | 所有声明 dtype、format 和 base 的性能倍率至少达到 0.3 |

每个 case 至少 warmup 10 次、有效采样 30 次，统计 median、p90、清零
耗时、scatter 耗时、端到端耗时、workspace 峰值和有效带宽。性能测试
排除首次编译和输入搬运，复用 descriptor 和 workspace。

### 4.2 Zero-fill 优化

Zero-fill 通常涉及完整 dense 输出，是带宽敏感阶段。优化采用以下策略：

- 根据 ROW/COL 选择连续物理方向；
- 使用向量化 zero store 和尾部 mask；
- 按 dense storage 范围划分任务，保持各 core 工作量均衡；
- 将 zero-fill 与 scatter 分成顺序明确的两个 Kernel，避免 sparse
  写入和清零写入产生竞争；
- 对 <code>nnz == 0</code> 使用仅清零路径，跳过索引解析。

### 4.3 Scatter 访存和负载均衡

CSR/CSC 使用 offset 区间分配任务，避免把所有输入先复制为 COO。对于
超长 row 或 column，将单个 offset 区间切片；对于大量空 row 或 column，
跳过空区间并把有效区间合并到任务队列。

COO 使用 p 维度直接并行，并按输出地址计算写入。由于输入坐标唯一，
不使用 atomic；若检测到重复坐标，按参数错误路径处理，不让重复写入
进入性能 Kernel。

### 4.4 dtype 和布局优化

同一 format 下，索引遍历逻辑与 value dtype 解耦。通过 dtype 分派减少
运行时分支，并让 value load/store 使用匹配向量宽度。ROW 和 COL 只在
最终地址计算处分派，避免为每个 dtype 再复制一套索引逻辑。

对于 complex64，采用 8 字节元素粒度的连续搬运，保证实虚部同时写入，
避免为了计算而产生临时复数转换。

### 4.5 内存优化

workspace 只存放固定状态和小规模调度元数据，不存放完整 dense output。
输出 storage 只由 matB 的 shape 和 ld 决定，values、indices 和 offsets
均复用输入 storage。

内存测试统一采集以下指标：

- input baseline allocated/reserved bytes；
- output storage bytes；
- input plus output storage bytes；
- peak allocated/reserved bytes；
- extra peak allocated/reserved bytes；
- GetBufferSize 查询值和实际 workspace 分配值。

当输入输出总量超过 500 MB 时，NPU 额外峰值相对 GPU 的比例不得超过
5%。对于不适用 GPU 对比的 case，workspace 绝对值不得超过目标硬件
L2 Cache 容量。任一条件满足即可作为该 case 的内存判定路径。

## 五、测试设计与判定标准

本节只定义测试内容、执行方法和验收标准。

### 5.1 测试环境要求

测试使用以下目标软件和硬件组合：

- Ascend 950PR，A5，<code>arch35</code>；
- CANN 9.1.0 及配套驱动、固件和编译器；
- PyTorch 2.7 及以上；
- torch_npu 26.0.0 及以后；
- ops-sparse 对应提交版本；
- GPU 参考侧使用 PyTorch 公共 sparse CSR/CSC/COO
  <code>Tensor.to_dense()</code> 路径；
- CPU Golden 使用确定性 sparse-to-dense 参考实现。

### 5.2 精度判定标准

精度测试对 dense 输出的全部逻辑元素执行 exact match，包含以下规则：

- 五种 value dtype 的已覆盖元素与 Golden bit-wise 一致；
- complex64 的实部和虚部均 bit-wise 一致；
- 未覆盖逻辑位置为对应 dtype 的正零；
- 正负零、Inf、NaN 必须按照输入 values 和 zero-fill 语义处理；
- 输出 padding 不参与逻辑比较，但必须通过边界写检查；
- 相同输入重复执行的 dense 输出必须 bit-wise 一致。

### 5.3 精度用例覆盖

精度执行使用任务包中的 <code>accuracy_cases.json</code>，覆盖 200 条
泛化用例和以下参数组合：

| 维度 | 覆盖范围 |
| --- | --- |
| format | CSR、CSC、COO |
| value dtype | INT8、FP16、BF16、FP32、complex64 |
| index base | 0、1 |
| dense layout | ROW、COL |
| nnz | 0、极小、低密度、中密度、高密度、近稠密 |
| shape | 小 shape、非方阵、长宽矩形和大 shape |
| row/column pattern | uniform、highly imbalanced、one long row、many empty rows、skewed、diagonal、banded、random、block-like、power-law |

精度用例必须覆盖空 row、空 column、零 nnz、长尾 row/column、乱序唯一
坐标、最大 shape、padding、正负零、Inf、NaN 和重复执行。

### 5.4 C++ 接口测试

C++ UT 覆盖 BufferSize 和执行接口的成对调用，并验证以下行为：

- 有效 handle、descriptor、DEFAULT algorithm；
- CSR/CSC/COO 的 format dispatch；
- I32 index 和 base 0/1；
- 五种 value dtype；
- ROW/COL dense layout 和最小/带 padding 的 ld；
- zero nnz、空 row、空 column 和不规则稀疏度；
- input arrays 只读，输出所有逻辑元素定义；
- workspace 查询值、workspace 容量和 stream 生命周期；
- 空描述符、空指针、非法 format/index/base/dtype/shape/ld/algorithm；
- shape 或 dtype 不匹配；
- 坐标越界、offset 非单调和重复坐标的参数错误路径；
- 非法 alias/in-place 场景的错误路径。

### 5.5 Python/ATen 测试

Python 端到端 UT 必须从公开入口验证 NPU Dispatcher，而不是只调用
底层 C++ 函数：

1. 构造 NPU CSR、CSC、COO sparse tensor。
2. 分别调用 <code>Tensor.to_dense()</code> 和对应的
   <code>aten::_to_dense</code> Dispatcher 路径。
3. 检查输出 device、dtype、shape、stride、layout 和数值。
4. 检查五种 dtype、index base、空结构、异常参数和 stream 行为。
5. 检查 input sparse storage 只读，输出不与输入 alias。
6. 对不支持的 layout、dtype、index type、device 和 stride 验证明确
   异常。
7. 通过 NPU dispatch 和 profiler 检查核心转换未回退到 CPU。

ROW 输出使用标准 row-major dense stride；COL 输出使用等价
<code>(1, rows)</code> stride。适配层不得通过 CPU tensor 构造输出后再
搬运到 NPU。

### 5.6 精度测试步骤

从任务包 <code>test_cases</code> 目录执行以下测试流程：

1. 生成或校验 <code>accuracy_cases.json</code>，确认用例包含三种格式、
   五种 dtype、base 0/1 和 ROW/COL。
2. 准备 NPU operator registration、ATen NPU registration 和 CPU Golden。
3. 执行 <code>run_accuracy_atk.sh</code>，由 ATK 同时调用 NPU 被测入口
   和 CPU Golden。
4. 对每个用例按 exact match、正零、padding、只读和无 CPU fallback
   标准判定。

示例命令如下：

~~~bash
cd 9月社区任务-aclsparseSparseToDense算子开发(950)/test_cases
bash run_accuracy_atk.sh
~~~

### 5.7 性能测试步骤

性能执行使用 <code>performance_cases.json</code>，包含 P-01、P-02、
P-03 的核心组合和泛化性能用例：

1. 使用同一 case file 分别准备 GPU 参考入口和 NPU 被测入口。
2. 复用 sparse descriptor、dense descriptor 和 workspace。
3. 输入构造和搬运不计入设备 Event 的调用范围。
4. 每个 case warmup 至少 10 次，采样至少 30 次。
5. 统计 median、p90、清零、scatter、端到端耗时和有效带宽。
6. 以 GPU 设备 Event 与 NPU 同范围耗时计算性能倍率。
7. 每个 P-01/P-02/P-03 的所有 dtype、format 和 base 组合均达到
   0.3 倍性能目标。

示例命令如下：

~~~bash
cd 9月社区任务-aclsparseSparseToDense算子开发(950)/test_cases
(cd aclsparseSparseToDense_testCase && \
  python3 benchmark_sparse_ops_npu.py \
  --case-file performance_cases.json --device 0 --output results)
~~~

### 5.8 内存测试步骤

内存测试对 GPU/NPU 使用相同 case file、相同输入结构和相同 dtype：

1. 运行 GPU memory collector，采集 input baseline、peak 和 extra peak。
2. 运行 NPU memory collector，采集同名字段以及 workspace 查询/分配值。
3. 按相同 case id 和 case fingerprint 对齐两侧输入。
4. 输入输出总量超过 500 MB 时，计算 NPU 相对 GPU 的 extra peak ratio。
5. 没有等价 GPU 接口时，检查 workspace bytes 是否不超过 L2 Cache。
6. 同时检查实现没有创建与完整 dense output 同阶的临时副本。

示例命令如下：

~~~bash
cd 9月社区任务-aclsparseSparseToDense算子开发(950)/test_cases
(cd aclsparseSparseToDense_testCase && \
  python3 collect_sparse_ops_gpu_memory.py \
  --case-file performance_cases.json --device 0 --output memory_gpu)
(cd aclsparseSparseToDense_testCase && \
  python3 collect_sparse_ops_npu_memory.py \
  --case-file performance_cases.json --device 0 --output memory_npu)
~~~

内存判定标准为：超过 500 MB 的 case 满足额外峰值不超过 GPU 峰值的
5%，或无等价 GPU 接口的 case 满足 workspace 不超过目标硬件 L2
Cache 容量。

### 5.9 无 CPU fallback 判定

无 CPU fallback 通过以下三类检查确认：

- Python/ATen Dispatcher 将 NPU sparse input 分发到 NPU 实现；
- profiler 显示 zero-fill 和 scatter 在 NPU stream 上执行；
- NPU 入口不调用 CPU reference、CPU sparse constructor 或 CPU dense
  materialization。

任意不支持的输入必须明确抛出异常，不能静默切换到 CPU。

## 六、风险和关键决策

本节记录设计阶段需要重点验证的风险和对应控制方式。

### 6.1 Device index 校验与异步语义

offsets、indices 和 COO 坐标位于 Device，Host 不能通过 CPU 扫描替代
转换。descriptor 元数据在 Host 侧校验，输入构造层保证 Device index
满足单调、范围和唯一性约束；对可在提交前确定的非法参数返回参数错误。
实现不得为了普通有效输入把整个 sparse 结构搬回 CPU。

### 6.2 长尾稀疏结构

一个超长 row 或 column 可能阻塞单个 task。CSR/CSC 使用 offset 区间
切片和调度元数据拆分长段；空 row/column 不生成无效 work；COO 直接按
nnz 并行。该策略同时覆盖规则稀疏和高度不均衡稀疏结构。

### 6.3 ROW/COL stride 和 padding

ROW/COL 地址公式不同，且 ld 可以大于逻辑最小值。所有 Kernel 使用
统一的逻辑坐标到物理地址函数，并对尾块使用 mask。测试对 padding
进行 guard 检查，确保只写逻辑区域。

### 6.4 大 dense 输出的内存压力

SparseToDense 的输出可能远大于输入 sparse storage。方案不创建第二份
完整 dense 输出，workspace 只保存固定状态和小规模调度信息；zero-fill
按输出布局进行带宽优化，内存测试按任务书的两条规则判定。

### 6.5 dtype 与 bit pattern

该算子不应对 values 做算术运算。Kernel 采用 dtype 对应的 load/store，
complex64 整体搬运，确保负零、Inf 和 NaN 不因转换而改变。未覆盖位置
由专用 zero-fill 路径生成正零。

### 6.6 Python/ATen 适配边界

ATen 层必须保持 sparse layout、dtype、index type、device、shape 和
stride 的明确约束。适配不支持的输入时返回稳定异常；输入输出 storage
独立；公开入口、Dispatcher、C++ Host 和 Kernel 使用同一 stream。

## 七、名词解释

下表统一本文中的接口、布局和稀疏结构术语。

| 名词 | 解释 |
| --- | --- |
| CSR | Compressed Sparse Row，按行压缩的稀疏矩阵格式 |
| CSC | Compressed Sparse Column，按列压缩的稀疏矩阵格式 |
| COO | Coordinate，使用 row index 和 column index 表示坐标 |
| primary | CSR 的 row offsets、CSC 的 column offsets 或 COO 的 row indices |
| secondary | CSR 的 column indices、CSC 的 row indices 或 COO 的 column indices |
| nnz | 非零 values 的数量 |
| index base | sparse index 的起始基，支持 0 或 1 |
| ROW | dense 矩阵按 row-major 物理布局存储 |
| COL | dense 矩阵按 column-major 物理布局存储 |
| ld | dense 输出的 leading dimension |
| zero-fill | 将 dense 逻辑区域写为对应 dtype 正零的阶段 |
| scatter | 按 sparse 坐标将 values 写入 dense 输出的阶段 |

## 八、参考资料

参考资料和工作区测试入口如下：

1. [aclsparseSparseToDense 任务书](../aclsparseSparseToDense_A5_task_doc.md)。
2. [测试包说明](../test_cases/README.md)。
3. [专项测试说明](../test_cases/aclsparseSparseToDense_testCase/README.md)。
4. [cuSPARSE SparseToDense 文档](https://docs.nvidia.com/cuda/cusparse/)。
5. [ops-sparse 开源仓](https://gitcode.com/cann/ops-sparse)。
6. [ops-sparse 公共头文件](https://gitcode.com/cann/ops-sparse/blob/master/include/cann_ops_sparse.h)。
7. [生态算子精度标准](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md)。
8. [PyTorch Tensor.to_dense](https://pytorch.org/docs/stable/generated/torch.Tensor.to_dense.html)。
9. [PyTorch native_functions.yaml](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/native_functions.yaml)。
10. [社区任务设计文档模板](https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md)。
