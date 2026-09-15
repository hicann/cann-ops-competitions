# aclsparseSDDMM 算子设计方案（Atlas A2/A3，arch22）

- 设计者：`lzc12138`
- 任务：2026 年 9 月社区任务《aclsparseSDDMM 算子开发（A2/A3）》
- 目标硬件：Atlas A2 训练系列（910B3、910B4）和 Atlas A3（DAV_2201，`arch22`）
- 目标仓库：[`cann/ops-sparse`](https://gitcode.com/cann/ops-sparse)
- 依据：任务书 `aclsparseSDDMM_A2A3_task_doc.md`，归档包 SHA256
  `5d64fef0b312ca8a28c8708635db0832b0b2d1fea0c507e05adcf52b89cc1efd`

本文是设计文档，不宣称实现、性能或验收已经完成。实现完成后，必须用任务书规定的
用例、设备和计时口径补充独立自测报告。

## 需求背景（required）

### 需求来源

任务要求在 `ops-sparse` 的 `arch22` 路径中补齐 `aclsparseSDDMM` 的 C++ 接口、Host
校验与调度、Ascend C Kernel、C++ 测试和 Python/ATen 适配。接口语义参考 cuSPARSE
SDDMM，但实现只允许使用 aclsparse 描述符、Ascend C 和目标 NPU；cuSPARSE 只作为
接口和 GPU 设备事件标杆。

Python 交付必须覆盖 PyTorch 2.7 及以上、`torch_npu` 26.0.0 及之后版本的
`torch.sparse.sampled_addmm` 和 `aten::sparse_sampled_addmm` NPU 路径。任何不支持的
组合都显式返回错误，不转 CPU、dense、参考实现、其他后端或 peer workspace。

### 背景介绍

SDDMM 只计算稠密矩阵乘积在稀疏结构已有位置上的值：

$$
C_{out} = (\alpha \cdot op(X) \cdot op(Y) + \beta \cdot C_{in}) \circ spy(C)
$$

CSR 中的单个值满足：

$$
C_{ij} \leftarrow \alpha \sum_{k=0}^{K-1} op(X)_{ik}op(Y)_{kj} + \beta C_{ij}
$$

其中 `spy(C)` 只保留原有 CSR 非零位置或 BSR 非零块内的标量位置。`rowOffsets`、
`colIndices`、BSR block pattern 和描述符结构在执行前后不变，只有 `values` 原地更新。
`TRANSPOSE` 只改变逻辑坐标映射，complex64 不执行共轭。

本任务的增量集中在四个边界：

| 维度 | 必须覆盖的能力 |
| --- | --- |
| 稀疏格式 | CSR 与方块 BSR；BSR 块内 ROW/COL 布局 |
| 类型与索引 | FP16、FP32、complex64；BSR 增加 BF16 混合路径；I32、base 0/1 |
| 生命周期 | `BufferSize -> Preprocess -> Execute`，状态绑定 `matC`，支持可复用 workspace |
| 调用接口 | C++ 三阶段接口、BSR/DnMat strided-batch API、PyTorch/ATen NPU 入口 |

## 需求分析（required）

### 需求描述

1. 在不改变既有三阶段 ABI 的前提下，为 A2/A3 `arch22` 提供 CSR/BSR SDDMM。
2. 对所有声明组合执行严格的 Host 参数校验、索引校验、workspace 校验和错误返回。
3. 让 `Preprocess` 产生可验证、可复用且与 `matC` pattern 绑定的状态，禁止以 workspace
   指针为 key 的进程全局非线程安全缓存。
4. 以稀疏 value/block 为并行单位完成 NPU 计算，不生成完整 `M x N` dense 结果。
5. 交付 ATen Dispatcher、Python E2E、C++ UT/ST 和 NPU Dispatch/Profiler 证据。

### 算子原型

保留以下三阶段函数的参数顺序和错误语义：

```cpp
aclsparseStatus_t aclsparseSDDMMBufferSize(
    aclsparseHandle_t handle, aclsparseOperation_t opX, aclsparseOperation_t opY,
    const void *alpha, aclsparseConstDnMatDescr_t matX,
    aclsparseConstDnMatDescr_t matY, const void *beta,
    aclsparseSpMatDescr_t matC, aclDataType computeType,
    aclsparseSDDMMAlg_t alg, size_t *size);

aclsparseStatus_t aclsparseSDDMMPreprocess(
    aclsparseHandle_t handle, aclsparseOperation_t opX, aclsparseOperation_t opY,
    const void *alpha, aclsparseConstDnMatDescr_t matX,
    aclsparseConstDnMatDescr_t matY, const void *beta,
    aclsparseSpMatDescr_t matC, aclDataType computeType,
    aclsparseSDDMMAlg_t alg, void *buffer);

aclsparseStatus_t aclsparseSDDMM(
    aclsparseHandle_t handle, aclsparseOperation_t opX, aclsparseOperation_t opY,
    const void *alpha, aclsparseConstDnMatDescr_t matX,
    aclsparseConstDnMatDescr_t matY, const void *beta,
    aclsparseSpMatDescr_t matC, aclDataType computeType,
    aclsparseSDDMMAlg_t alg, void *buffer);
```

本任务新增的描述符接口如下：

```cpp
aclsparseStatus_t aclsparseCreateBsr(
    aclsparseSpMatDescr_t *spMatDescr, int64_t blockRows,
    int64_t blockCols, int64_t blockNnz, int64_t rowBlockSize,
    int64_t colBlockSize, void *bsrRowOffsets, void *bsrColInd,
    void *bsrValues, aclsparseIndexType_t bsrRowOffsetsType,
    aclsparseIndexType_t bsrColIndType, aclsparseIndexBase_t idxBase,
    aclDataType valueType, aclsparseOrder_t order);

aclsparseStatus_t aclsparseCreateConstBsr(
    aclsparseConstSpMatDescr_t *spMatDescr, int64_t blockRows,
    int64_t blockCols, int64_t blockNnz, int64_t rowBlockSize,
    int64_t colBlockSize, const void *bsrRowOffsets,
    const void *bsrColInd, const void *bsrValues,
    aclsparseIndexType_t bsrRowOffsetsType,
    aclsparseIndexType_t bsrColIndType, aclsparseIndexBase_t idxBase,
    aclDataType valueType, aclsparseOrder_t order);

aclsparseStatus_t aclsparseDnMatGetStridedBatch(
    aclsparseConstDnMatDescr_t dnMatDescr, int *batchCount,
    int64_t *batchStride);

aclsparseStatus_t aclsparseDnMatSetStridedBatch(
    aclsparseDnMatDescr_t dnMatDescr, int batchCount,
    int64_t batchStride);

aclsparseStatus_t aclsparseBsrSetStridedBatch(
    aclsparseSpMatDescr_t spMatDescr, int batchCount,
    int64_t offsetsBatchStride, int64_t columnsBatchStride,
    int64_t valuesBatchStride);
```

### 支持矩阵

| 稀疏格式 | `matX/matY` | `matC.values` | `computeType` | 设备端累加 |
| --- | --- | --- | --- | --- |
| CSR/BSR | FP32 | FP32 | FP32 | FP32 |
| CSR/BSR | complex64 | complex64 | complex64 | 两路 FP32 复数乘加 |
| CSR/BSR | FP16 | FP16 | FP32 | FP32，写回 FP16 |
| CSR/BSR | FP16 | FP32 | FP32 | FP32 |
| BSR | BF16 | BF16 | FP32 | FP32，写回 BF16 |
| BSR | BF16 | FP32 | FP32 | FP32 |

CSR 不扩展 BF16；BSR 只支持方块，`rowBlockSize == colBlockSize`，块大小为
`2/4/8/16/32/64/128`。所有稀疏索引为 I32，index base 为 0 或 1。

### 参数说明

| 参数 | 作用 | 取值与布局 | 关键校验 |
| --- | --- | --- | --- |
| `handle` | 上下文和调用方 stream | 有效 aclsparse handle | 非空、stream 有效 |
| `opX/opY` | 稠密逻辑变换 | `NON_TRANSPOSE` 或 `TRANSPOSE` | 不接受共轭转置和非法枚举 |
| `alpha/beta` | 缩放标量 | compute type；Host/Device pointer mode | 指针、类型和可读性合法 |
| `matX/matY` | 稠密输入 | FP16/BF16/FP32/complex64，ROW/COL，合法 `ld` | op 后形状可乘，batch/stride 不溢出 |
| `matC` | 固定 pattern 与可写 values | CSR 或方块 BSR，I32，base 0/1 | pattern、shape、dtype、指针一致 |
| `computeType` | 点积和标量计算类型 | FP32 或 complex64 | 必须匹配上表 |
| `alg` | 算法选择 | `DEFAULT` | 其他枚举返回 `NOT_SUPPORTED` |
| `size` | workspace 查询输出 | `size_t*` | 空指针和对齐/加法溢出返回错误 |
| `buffer` | Preprocess/Execute workspace | Device 连续内存 | 按查询大小分配并满足对齐；未 preprocess 拒绝执行 |
| BSR block 属性 | 块维度和块数量 | 非负 `int64`，方块约束 | 乘法不溢出，值数量匹配 |
| `order` | BSR 块内物理布局 | ROW 或 COL | 非法枚举返回 `INVALID_VALUE` |
| batch 属性 | DnMat/BSR 批量跨度 | `batchCount` 为 1..65535，stride 以元素计 | 非负、覆盖单批存储、总跨度不溢出 |

逻辑形状要求为 `op(X) @ op(Y) == [M, N]`。DnMat 的 `batchCount=1` 可在 BSR 输出批次
中广播；否则必须等于 BSR 的 `batchCount`。CSR 的 `matC` 为单批，因而 CSR 场景的
有效 batch 为 1。四种 BSR 组合均合法：X/Y 同时广播、仅 X 批量、仅 Y 批量、X/Y
同时批量。

## 详细设计（required）

### 分层与目录

实现按以下边界放置，避免 A2/A3 和 A5 的公共 Host 逻辑互相污染：

| 层 | 位置 | 责任 |
| --- | --- | --- |
| 公共声明 | `include/cann_ops_sparse.h` | 三阶段 API、BSR 创建和 strided-batch 声明 |
| arch22 Host | `sparse/sddmm/arch22/` | 描述符扩展、参数校验、workspace、dispatch |
| Kernel | `sparse/sddmm/arch22/` 的 Ascend C 源文件 | CSR 快路径、CSR/BSR 泛化路径、dtype 转换 |
| 测试 | `test/sddmm/arch22/` 和任务专项目录 | C++ UT/ST、Python/ATen E2E、性能和内存采集 |

公共辅助函数只承载格式无关的检查和状态编码；SOC-specific tiling、Kernel 属性和
编译入口留在 `arch22`，不通过运行时读取 `arch35` 或 `arch22` 之外的实现。

### 描述符与生命周期

`SpMat` 描述符增加 format、base、I32 index 类型、逻辑维度、nnz、BSR block 元数据、
块内 order、batch strides，以及 SDDMM preprocess 绑定。`DnMat` 描述符增加
`batchCount` 和 `batchStride`。设置指针、shape、layout、dtype、batch 或 BSR 属性时，
清除旧的 preprocess 绑定。

`BufferSize` 的执行顺序为：

1. 校验 handle、算法、op、描述符、dtype、shape、索引和 batch。
2. 计算 pattern 元数据、K 维和所需临时状态，使用显式无符号溢出检查。
3. 按 256 字节对齐返回精确 workspace 字节数；零 nnz 也返回合法的最小 header 空间。

workspace 以版本化 `SddmmWorkspaceHeader` 开头，至少包含 magic/version、format/base、
M/N/K/nnz、block、batch、dtype、tiling key、pattern 指针身份和 pattern signature。可选
的尾部只保存压缩元数据或状态字，不保存完整 dense 结果。workspace 只由调用方分配，
库不以 workspace 地址建立全局缓存。

`Preprocess` 在调用方 stream 上完成一次必要的 pattern 读取和校验：检查 row offsets
端点与单调性、column 范围、base 编码、BSR block 数量和 batch stride；随后把 header
和
签名写入 device workspace，并把完整签名绑定到对应 `matC`。同一 `matC` 的指针或属性
改变会立即失效；同地址 pattern 内容改变时，调用方必须重新 `Preprocess`，执行路径
不得静默使用旧状态。不同 `matC` 和独立 workspace 可以并发，同一描述符的阶段顺序由
调用方串行保证。

`Execute` 只接受已绑定的 workspace，先检查 header magic/version、format/base、shape、
block、batch、dtype、pattern 指针和签名，再进入 Kernel。由于 ABI 没有传入 buffer
大小，
接口可以拒绝空指针、未对齐和未 preprocess 状态，但不能承诺可靠检测调用方提供的
欠配字节数；测试按 `BufferSize` 精确分配。

### Host 参数校验与调度

Host 校验顺序固定为“句柄与枚举 -> 描述符 -> dtype -> 形状 -> 存储布局 -> batch/stride
-> workspace”，这样错误在进入设备队列前可定位。所有元素数量、字节数、`ld`、batch
偏移、BSR value 数量和 256 字节对齐计算都使用显式 `int64/uint64` 检查，拒绝负值、
乘加溢出和超过 arch22 GlobalTensor 可寻址范围的输入。

调度只使用运行时元数据，不读取公开 case id、seed 或固定 shape：

1. CSR、FP16/FP32 同类型、batch=1、K 和稀疏行分布满足向量化条件时进入 CSR 快路径。
2. 其他合法组合进入同一个参数化泛化 Kernel，覆盖 CSR/BSR、base、order、transpose、
   混合精度、complex64 和 strided batch。
3. 非法组合返回明确的 `INVALID_VALUE` 或 `NOT_SUPPORTED`，不通过另一个后端补算。

### CSR/BSR 坐标恢复

CSR 的 `rowOffsets` 长度为 `M+1`，value index 通过分段二分或分块索引表找到逻辑行，
column index 减去 `idxBase` 得到逻辑列。base 0 时端点为 `0/nnz`，base 1 时端点为
`1/nnz+1`；空行和 zero-nnz 直接产生空工作段。

BSR 先把展平的 value index 拆成 `blockIndex` 与块内线性位置，再由 block row 的
rowOffsets 恢复 block row；块列为 `colInd - idxBase`。块内 ROW 布局使用
`inner = linear / blockDim, linear % blockDim`，COL 布局交换两者，最后组合成逻辑
`(row, col)`。所有 batch 的 offsets、columns 和 values 均以各自 stride 定位，base
只
影响逻辑坐标，不修改原始索引。

### Kernel 与 tiling

#### CSR 快路径

快路径按非空行分配任务，沿 K 维分块搬运 X/Y，在 UB 中做 FP32 累加，减少每个 value
重复查找 row offset 的开销。长尾行采用 work-stealing 式的固定 chunk，避免单个大行
拖慢尾部；K 不对齐时由尾块掩码处理。该路径仍逐 value 写回 `C.values`，不构造 dense
输出。

#### 泛化路径

泛化路径把 `batch * stored_value_count` 展平后按 core 进行 strided 分配。每个任务：

1. 解码 batch/value 或 batch/block/inner 坐标；
2. 按 op、order、leading dimension 和 batch stride 读取逻辑 X/Y；
3. FP16/BF16/FP32 提升到 FP32 累加，complex64 以实部/虚部两路 FP32 累加；
4. 计算 `alpha * dot + beta * old_value`，按 `matC.values` dtype 转换并原地写回。

K tile 大小由 UB、double buffer、dtype 和当前 K 的运行时元数据共同决定。tile、core
数和尾块信息写入 workspace header；不以 P-01/P-02/P-03 的 shape 建立专用分支。BSR
块内连续值优先合并搬运，块大小变化只影响坐标解码和 tile 选择。

#### 复数与标量

complex64 的乘加按

`(ar + i*ai) * (br + i*bi) = (ar*br-ai*bi) + i*(ar*bi+ai*br)`

执行，`alpha/beta` 使用同一 compute type。FP16/BF16 的 alpha、beta 在 Host 校验后
转换为 FP32；Device pointer mode 的标量只在调度需要时读取一次。Kernel 不执行隐式
共轭，输入 X/Y 和稀疏结构保持只读。

### Stream、同步与错误语义

所有 device copy、preprocess header 写入和计算 Kernel 都排入调用方 stream。Preprocess
允许为验证 device pattern 做一次明确的同步读取；Execute 不做无关的 Host 同步，也不
改变当前 stream。错误 descriptor、非法 index、shape/dtype/layout、未绑定 workspace、
不支持算法或溢出在 Host 侧返回对应状态码；Kernel 只消费已经通过 Host 校验的元数据。

### Python/ATen 适配

适配层注册 NPU PrivateUse1 的 `aten::sparse_sampled_addmm`，公开入口为
`torch.sparse.sampled_addmm`。调用过程为：

1. 检查所有 dense tensor、CSR crow/col/value 位于同一 NPU，layout、dtype、shape、stride
   和 alias/in-place 语义符合 PyTorch 约定。
2. 从 CSR tensor 构造 const/mutable `DnMat` 与 `SpMat` 描述符；BSR 专用路径通过任务
   新增的 BSR API 表达 block 和 batch 属性。
3. 在当前 NPU stream 上依次调用 `BufferSize`、workspace 分配、`Preprocess` 和
   `aclsparseSDDMM`，只构造 values 更新后的稀疏输出，保留原 crow/col。
4. 对未支持的 BSR/类型/stride/设备组合用 `TORCH_CHECK` 报错；禁止捕获异常后转 CPU。

Python E2E 同时检查 dispatcher 命中、输出 pattern 不变、异步 stream 语义、异常类型
和无 CPU fallback。C++ 入口与 Python 入口共享同一 Host 校验和 Kernel，不维护第二份
计算逻辑。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas A2 训练系列（910B3/910B4） | √ |
| Atlas A3（DAV_2201，`arch22`） | √ |

## 支持软件版本

| 软件 | 版本 |
| --- | --- |
| CANN | 9.1.0 及后续配套版本 |
| PyTorch | 2.7 及以上 |
| torch_npu | 26.0.0 及之后 |

每个 SOC 都必须单独记录驱动、固件、CANN、编译提交和实际设备；一个 SOC 的结果不能
替代另一个 SOC 的验收证据。

## 算子约束限制

- 仅支持 `NON_TRANSPOSE`/`TRANSPOSE`，不支持共轭转置。
- CSR/BSR 索引均为 I32，base 只能为 0 或 1；BSR 只支持方块和规定的七种 block size。
- I32 的 offset/index、维度乘积和 batch 地址必须分别落在 `INT32_MAX`、`INT64_MAX`
  和 `SIZE_MAX` 可表示范围内；单个 arch22 GlobalTensor 元素偏移不得超过 `UINT32_MAX`，
  超出范围返回 `NOT_SUPPORTED`。
- CSR 支持 FP16、FP32、complex64；BSR 另支持 BF16->BF16/FP32；compute type 必须与
  支持矩阵一致。
- X/Y 的 op 后形状必须匹配 C 的 `[M,N]`；ROW/COL、合法 leading dimension 和动态
  M/N/K/nnz 均须校验。
- BSR batchCount 为 1..65535；DnMat batchCount 为 1 或 BSR batchCount，stride 不得
  覆盖越界。CSR matC 只接受单 batch。
- `C` 的结构、X/Y、row offsets 和 column indices 只读，执行只更新已有 values。
- 必须先查询并按结果分配 workspace，再调用 Preprocess；pattern、描述符属性或指针
  改变后必须重新 Preprocess。
- 不允许 CPU、dense、参考、peer、跨后端或静默错误值 fallback；不支持组合必须显式
  返回错误。
- 不得把公共任务 shape、case id、seed 或 density 作为运行时 dispatch 条件。

## 可维可测分析

### 精度标准

CPU golden 使用任务书规定的更高精度累加：FP16/BF16 使用 FP32，FP32 使用 FP64，
complex64 使用 complex128。NPU 结果只与 CPU golden 比较，不以另一个候选实现作为
正确性标杆。

| dtype | rtol | atol | 匹配率 | 绝对误差硬上限 |
| --- | ---: | ---: | ---: | ---: |
| FP16 | `2^-9` | `2^-9` | >= 0.99 | `max(1e-1, 32 * ULP)` |
| BF16 | `2^-6` | `2^-6` | >= 0.99 | `max(1, 32 * ULP)` |
| FP32 | `2^-10` | `2^-16` | >= 0.99 | `max(1e-2, 32 * ULP)` |
| complex64 | 按 FP32 分量 | 按 FP32 分量 | 实部、虚部均 >= 0.99 | 实部、虚部均按 FP32 上限 |

测试覆盖普通值、正负混合、零、小值、离群值及任务书允许的 INF/NAN，并同时验证
稠密输入、稀疏 pattern、alpha/beta、workspace 和输出结构不变性。

### 性能标准

性能倍率严格定义为：

`GPU device-event median / NPU 同调用范围的全部 Kernel 总耗时`。

每个 case 预热 10 次、正式采样 30 次，报告 median 和 p90；描述符、workspace、
preprocess 状态在正式采样期间复用，不计入编译、数据生成、Host/Device 搬运或无关
初始化。cuSPARSE 与 PyTorch GPU 标杆是独立调用范围，不能混合。

以下数值仅是任务书提供的标杆记录，不是本设计的实测声明：

| Case | 固定维度和稀疏结构 | GPU 标杆 median_us（任务书范围） | 目标 |
| --- | --- | --- | --- |
| P-01 | `8192 x 28672`, `K=128`；CSR 每行 64 nnz；BSR block 16 | cuSPARSE 49.504--49.568（2 条）；PyTorch 423.120--656.480（4 条） | 每个声明组合 >= 同范围标杆的 0.25x |
| P-02 | `4096 x 1536`, `K=128`；CSR 每行 64 nnz；BSR block 32 | cuSPARSE 33.312--33.600（2 条）；PyTorch 291.808--408.096（4 条） | 每个声明组合 >= 同范围标杆的 0.25x |
| P-03 | `7168 x 2048`, `K=128`；CSR 每行 64 nnz；BSR block 64 | cuSPARSE 45.920--46.016（2 条）；PyTorch 386.592--604.144（4 条） | 每个声明组合 >= 同范围标杆的 0.25x |

P-01/P-02/P-03 的每个声明 dtype、base、格式和目标 SOC 都必须有独立结果；缺少某个
组合不能用其他 case 或其他设备补齐。

### 内存标准

当输入输出总量超过 500 MB 且存在等价 GPU Torch 调用时，NPU 额外峰值内存不得超过
GPU 总使用量的 50%。没有等价 GPU 调用时，方案固有 workspace 绝对值不得超过目标
硬件 L2 Cache。实现不得保存完整 dense `M x N` 结果，不得泄漏 Host/Device 资源，
并且 BufferSize、Preprocess 和 Execute 必须复用同一 workspace。

### 自测覆盖

交付测试至少包括：

1. CSR/BSR、base 0/1、空行、zero-nnz、长尾 pattern 和动态 M/N/K/nnz；
2. FP16、FP32、complex64、BSR BF16 两种 values 组合；四种 opX/opY、ROW/COL dense
   order、leading dimension padding、alpha/beta 边界和复数标量；
3. BSR block 2/4/8/16/32/64/128、块内 ROW/COL、四种 batch 广播组合、batch 边界和
   非法 stride；
4. BufferSize 精确值、空 buffer、未 preprocess、workspace 复用、pattern/descriptor
   变化后的重新 preprocess、描述符创建/销毁和资源泄漏；
5. C++ 返回码、输入只读性、NPU Dispatch、Profiler Kernel 证据、ATen dispatcher、
   Python E2E 输出及明确的 no-fallback 断言；
6. 三个固定性能 case 的 10/30 设备事件采样、median/p90、workspace 峰值和逐 SOC
   结果。

测试数据按任务书规则固定种子生成：X/Y/C values 的 70% 来自 `[-1,1]` 均匀分布，
20% 来自 `N(0,1)`，10% 覆盖零值、边界值及规格允许的特殊值；CSR/BSR pattern 同时
覆盖均匀、长尾和空块行。确定性算法的重复执行还需通过任务书规定的 bit-wise 检查，
其他算法使用本节的逐元素容差。

## 兼容性分析

既有三阶段函数签名保持不变；新增 BSR 和 strided-batch API 是向前扩展。描述符的新
字段位于不透明结构内部，旧 CSR FP16/FP32 base-0 单 batch 调用继续使用兼容路径。
公共 Host 校验与 A5 通过独立 arch 适配层解耦，避免一个 SOC 的 tiling 或 workspace
状态泄漏到另一个 SOC。

本设计不引入全局缓存、legacy alias、双写状态或 fallback 兼容层。任何未通过任务书
声明组合的调用都应在 Host 侧给出稳定错误码；实现完成后的 PR 还必须附上实际代码
提交、测试日志、设备信息和性能/内存报告，才能声称任务验收完成。

## 实施顺序与风险控制

1. 先更新公共声明和不透明描述符字段，补齐创建/设置/销毁的生命周期 UT。
2. 实现 arch22 Host 校验、BufferSize/Header/Preprocess 绑定，再用零 nnz 和小 CSR
   验证
   状态机。
3. 接入 CSR 快路径，随后加入 BSR 坐标恢复、混合精度、complex64 和 batch 泛化路径。
4. 完成 ATen/Python dispatcher 与 no-fallback 测试，再进行三 SOC 的精度、性能和内存
   采集。
5. 只有当全部 required 用例和报告齐全后，才在 `ops-sparse` 提交实现 PR；本设计 PR
   本身不替代实现 PR 或自测报告。

主要风险及对应控制：

- BSR base/order 解码错误：用每种 block size、布局和 base 的逐元素坐标 golden 反查。
- preprocess 状态误复用：将完整 descriptor/pattern 签名写入 header，并在所有 setter
  上清除绑定；pattern 生命周期由测试显式覆盖。
- batch stride 溢出或广播错位：Host 使用 checked arithmetic，四种广播模式分别验证。
- 计时范围不一致：GPU 和 NPU 分别记录调用边界、设备事件和所有 Kernel，总耗时不以
  Host wall-clock 替代。
- 不支持路径隐式回退：Python 和 C++ 测试检查 dispatcher、设备归属和 NPU profiler，
  并把不支持组合断言为显式异常。

## 参考资料

1. 社区任务设计模板：`04_tasks/01_community-task-2026/resources/design_template.md`。
2. 任务书：`aclsparseSDDMM_A2A3_task_doc.md`（本 PR 对应的归档包）。
3. [ops-sparse](https://gitcode.com/cann/ops-sparse)。
4. [PyTorch ATen native functions](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/native_functions.yaml)。
5. [生态算子开源精度标准](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md)。
6. [社区任务流程](https://gitcode.com/org/cann/discussions/39)。
