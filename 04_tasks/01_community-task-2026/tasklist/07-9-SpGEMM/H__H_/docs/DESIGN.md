# aclsparseSpGEMM A2/A3 算子设计

## 1. 需求背景

### 1.1 任务目标

本任务面向 Atlas A2、Atlas A3 系列产品，实现 PyTorch 2.7 及以上版本的稀疏矩阵乘法 NPU 能力，并复用 `ops-sparse` 中既有的 `aclsparseSpGEMM*` 多阶段接口。目标公开入口为：

```python
torch.sparse.mm(mat1, mat2) -> Tensor
```

其内部对应：

```text
aten::_sparse_sparse_matmul(Tensor self, Tensor other) -> Tensor
```

底层计算为：

$$
C = \alpha \cdot A \cdot B + \beta \cdot C
$$

其中 $A$ 为 $M\times K$ CSR 矩阵，$B$ 为 $K\times N$ CSR 矩阵，输出 $C$ 为 $M\times N$ CSR 矩阵。PyTorch 的 `sparse.mm` 主路径取 $\alpha=1,\beta=0$；C++ 多阶段接口仍需保持 `alpha`、`beta*C` 语义。

核心计算、符号分析和数值计算的目标实现均位于 NPU，不以 CPU fallback 代替。Host 只允许进行参数检查、workspace/输出分配和读取 API 分配所必需的少量标量元数据。

### 1.2 输出规范

输出采用规范化 CSR：

- `rowOffsets` 单调非降，长度为 `M + 1`；
- 每行 `colIndices` 严格升序；
- 同一坐标的重复乘积必须归并；
- 计算产生的显式零值保留并计入 `nnz(C)`；
- `rowOffsets`、`colIndices` 和 `nnz(C)` 必须与 Golden 精确一致，values 按任务书混合容差验收；
- Python 层如果需要返回 COO，则必须构造 coalesced 输出。

## 2. 需求分析与约束

### 2.1 支持范围

| 项目 | 支持范围 |
| --- | --- |
| 产品 | Atlas A2、Atlas A3 |
| C++ 稀疏格式 | A、B、C 均为 CSR |
| Python 稀疏格式 | 按目标 PyTorch 语义接收，进入原生接口前规范化为 CSR |
| values dtype | `float16`、`bfloat16`、`float32`、`complex64` |
| ACL dtype | `ACL_FLOAT16`、`ACL_BF16`、`ACL_FLOAT`、`ACL_COMPLEX64` |
| 索引 | `ACL_SPARSE_INDEX_32I`，A/B/C 类型一致，zero-based |
| operation | 仅 `ACL_SPARSE_OP_NON_TRANSPOSE` |
| shape | A=`[M,K]`、B=`[K,N]`、C=`[M,N]`，支持规格内动态 shape/nnz |
| 广播 | 不支持矩阵维度广播 |

A、B、C 的 values dtype 与 `computeType` 必须一致。输入 CSR 的列索引必须有序、坐标合法。转置、共轭转置、非 int32 索引、非 zero-based、维度不匹配及未支持 dtype 均返回明确错误，不做隐式降级。

### 2.2 数据类型与精度策略

- FP16/BF16：加载后转换为 FP32 累加，最终转换回输出 dtype；
- FP32：采用 FP32 设备计算，长乘积链通过分段或补偿求和控制误差；
- Complex64：以连续的实部/虚部 FP32 表示，执行
  `real += ar*br-ai*bi`、`imag += ar*bi+ai*br`；
- 相同输入、算法和调度配置应产生确定的 CSR 结构；数值按任务书规定的 dtype 容差验收；
- INF/NAN、抵消、小值和显式零按照精度标准逐元素处理。

### 2.3 边界与溢出

需要覆盖零维、A/B 零 nnz、空行/空列、无相交乘积、`alpha=0`、`beta=0`、长尾行和中间乘积膨胀。计算 workspace 前使用 64 位整数完成乘加和对齐检查：

- `nnz(C)` 超出 int32 表示范围时返回不支持；
- 中间乘积总数超出 int64 或任一 workspace 字节数发生溢出时返回不支持；
- 空结果仍生成合法的全零 `rowOffsets`；
- 任何输出缓冲区不足、空指针或非法 CSR 元数据均不得越界写入。

## 3. 总体架构

### 3.1 正式目标数据流

正式架构采用完全设备侧的符号与数值流水线：

```text
A/B/C CSR (Device)
        |
        v
NPU symbolic count
  - 统计每行唯一输出列数
  - 统计每行中间乘积数
        |
        v
NPU prefix scan
  - rowCounts -> rowOffsetsC
  - 归约 nnz(C) 与 numProducts
        |
        +--> Host 仅读取 nnz(C)、numProducts 标量用于 API 分配
        |
        v
NPU symbolic fill
  - 生成每行严格升序的 colIndicesC
        |
        v
NPU numeric
  - alpha*A*B + beta*C
  - 重复坐标确定性归并
        |
        v
规范化 CSR C (Device)
```

`symbolic count -> prefix scan -> symbolic fill -> numeric` 是正式目标顺序。A/B 的行指针、列索引和值不整体搬回 Host，Host 不保存逐行列集合。

`prefix scan` 后需要一次受控同步或等价事件依赖，读取 `nnz(C)` 和 `numProducts` 两个标量，以便调用方分配 C 的 `colIndices`/`values`。该标量回读属于动态输出分配协议，不属于 CPU 符号计算。完成分配后，symbolic fill 和 numeric 继续在 handle 绑定的 stream 上执行。

### 3.2 旧基线与当前差距

早期正确性基线会将 A/B CSR 结构 D2H，在 Host 逐行使用 `std::set` 去重、排序，生成 `rowOffsetsC`/`colIndicesC` 后再 H2D。该路径曾用于验证数值 Kernel，并通过旧基线官方精度 200/200，但它是 CPU symbolic fallback，不能作为目标架构或最终验收实现。

当前 A2 device-symbolic 候选仅在 `beta=0` 主路径启用 count/scan/fill。`beta*C` 仍使用上述 Host symbolic 基线，因此现阶段不能宣称所有 API 路径已经消除 CPU fallback。

## 4. 多阶段 API 与状态机

### 4.1 调用顺序

```text
CreateDescr
  -> WorkEstimation(query bufferSize1)
  -> WorkEstimation(execute symbolic count + scan)
  -> GetNumProducts
  -> 调用方读取 nnz(C)，分配/更新 C 的 colIndices 与 values
  -> [ALG2/ALG3] EstimateMemory(query bufferSize2/bufferSize3)
  -> [ALG2/ALG3] EstimateMemory(execute planning)
  -> Compute(query bufferSize2)
  -> Compute(execute symbolic fill + numeric)
  -> Copy
  -> DestroyDescr
```

`aclsparseSpGEMMGetNumProducts(spgemmDescr, &numProds)` 必须在成功执行 WorkEstimation 后调用，返回所有行的真实中间乘积总数。它用于内存估算、算法选择、性能报告和溢出检查；在 WorkEstimation 前调用返回未初始化错误。

### 4.2 描述符状态

描述符维护以下逻辑状态：

| 状态 | 可进入操作 | 保存内容 |
| --- | --- | --- |
| Created | WorkEstimation 查询/执行 | 初始空状态 |
| WorkDone | GetNumProducts、EstimateMemory、Compute 查询 | M/K/N、dtype、alg、`nnz(C)`、`numProducts`、symbolic 模式 |
| MemoryEstimated | ALG2/ALG3 Compute | chunk 计划、buffer 需求、算法元数据 |
| Computed | Copy | C 的结构与 values 已生成 |
| Copied | Destroy 或按协议复用 | 完整输出可见 |

查询调用只返回字节数，不应错误地推进执行状态。重新执行 WorkEstimation 时必须清理旧的 symbolic/compute 状态。后续阶段需校验矩阵 shape、dtype、算法枚举与 WorkEstimation 时一致；跨描述符、跨 shape 或跳阶段调用返回 `NOT_INITIALIZED` 或 `INVALID_VALUE`。

同一描述符不是多 stream 并发安全对象。调用方需要串行复用，或者为不同 stream 创建独立描述符。

### 4.3 算法枚举

| 算法 | 目标用途 | 当前状态 |
| --- | --- | --- |
| `DEFAULT` | 根据 shape、行分布、`numProducts/nnz(C)` 和产品能力选择实现 | 当前路由到确定性基线 |
| `ALG1` | 低 workspace、确定性逐行/分行实现，适合一般及中小乘积行 | 已有功能基线；继续迁移为全 Device symbolic |
| `ALG2` | 由 `chunkFraction` 控制的分块算法，限制中间乘积膨胀时的峰值 workspace | 枚举和阶段校验存在，优化实现待完成，当前共享 ALG1 基线 |
| `ALG3` | 面向重行/高复用场景的高并行 hash/merge 方案，使用更大 scratch 换取吞吐 | 枚举和阶段校验存在，优化实现待完成，当前共享 ALG1 基线 |

ALG2/ALG3 的 `EstimateMemory` 必须根据 `numProducts`、`nnz(C)`、dtype、行分布和 `chunkFraction` 给出真实需求，不能在最终版本中仅返回固定占位值。不同算法必须保持相同的规范化 CSR 语义与错误行为。

## 5. Workspace 与生命周期

### 5.1 分区设计

目标 workspace 按对齐后的字节区间切分，不通过整数 tiling 字段间接恢复 GM 指针：

| Buffer | 主要分区 | 生命周期 |
| --- | --- | --- |
| `externalBuffer1` | `rowCounts[M]`、`productCounts[M+1]`、scan 临时区、symbolic tiling | WorkEstimation 执行至 count/scan 完成 |
| `externalBuffer3` | ALG2/ALG3 的分块边界、桶/merge 计划、算法元数据 | EstimateMemory 执行及其计划被 Compute 消费期间 |
| `externalBuffer2` | symbolic fill scratch、numeric tiling、分块中间值/归并区 | Compute 执行至 stream 上相关 Kernel 完成 |

所有 offset 与总大小先对齐再计算，并检查 `size_t`、int32/int64 溢出。用户拥有 external buffer；算子不释放用户内存。调用方必须保证 buffer 大小不小于查询值，并在对应 stream 工作完成前保持其地址有效且不得复用。

`rowOffsetsC`、`colIndicesC` 和 `valuesC` 属于调用方输出，不是临时 workspace。WorkEstimation 完成 count/scan 后，调用方依据 `nnz(C)` 更新 C 描述符的输出指针；Compute 再写入 `colIndicesC` 和 `valuesC`。

### 5.2 Stream 语义

- 所有 Device Kernel 和异步拷贝均提交到 `aclsparseHandle_t` 绑定的 stream；
- count、scan、fill、numeric 依靠同一 stream 的顺序或显式事件建立依赖；
- 除动态输出分配所需的两个标量回读外，不做无必要 Host 同步；
- Compute/Copy 返回后，调用方在消费输出前负责同步 stream；
- Host 栈上 tiling 数据不得成为函数返回后的异步拷贝源，需同步完成拷贝或使用生命周期足够长的存储；
- runtime/kernel 启动错误映射为 `EXECUTION_FAILED`，异步错误在调用方同步点报告。

## 6. Host 与 Kernel 设计

### 6.1 Host 侧职责

Host 仅负责：

1. 校验 handle、描述符、shape、CSR 格式、索引、dtype、operation、算法和阶段状态；
2. 根据平台信息、行分布、`numProducts` 与 `nnz(C)` 选择算法、block 数和 workspace；
3. 组织 query/execute 两次调用协议并写入 tiling；
4. 在同一 stream 上依次发射 count、scan、fill、numeric；
5. 回读动态输出分配所需的 `nnz(C)`、`numProducts` 标量；
6. 将底层 ACL/runtime 错误转换为稳定的 aclsparse 状态码。

Host 不遍历 A/B 的全部 CSR 数据，不维护逐行 `std::set`，不执行符号去重或数值乘加。

### 6.2 Device symbolic

**Count：** 每行遍历 A 的非零项及其引用的 B 行，统计中间乘积数，并使用局部有序 merge、位图或 hash scratch 统计唯一输出列数。对于 `beta*C`，还需在 Device 合并初始 C 的列集合。

**Prefix scan：** 对 `rowCounts` 做前缀和，生成 `rowOffsetsC`；同时归约 `nnz(C)` 和 `numProducts`。空行自然得到相同的相邻 row offset。

**Fill：** 按 `rowOffsetsC` 为每行分配的区间，重新遍历或复用 symbolic scratch，写出严格升序、无重复的 `colIndicesC`。任何 count/fill 数量不一致均作为内部错误处理。

### 6.3 Numeric

numeric 阶段以 C 的 `(row, col)` 结构为目标，遍历对应 A 行和 B 行，确定性归并同坐标乘积，执行 `alpha*A*B + beta*C`。短行批处理以降低启动和控制开销；普通行按连续行块分核；重行允许行内协作或分块归并，避免单个长行拖尾。values 写入必须受 `rowOffsetsC` 边界约束。

## 7. A2/A3 差异化调度

A2 与 A3 共用公共 API、CSR 语义及 arch22 源码框架，但不能假设同一 block 数、UB 切分和 shape 阈值在两个产品上都最优。Host 从运行时平台信息获取可用 AIV 核数和片上存储能力，并使用分别标定的调度表。

| 调度维度 | A2 设计 | A3 设计 |
| --- | --- | --- |
| symbolic 写入 | 优先保证 cache-line 独占；按连续行块分配 rowCounts/输出区间 | 在保持 cache-line 安全前提下提高并行行组和流水重叠 |
| 短行 | 多行合批，减少控制开销 | 扩大批量并根据实测提高并发 |
| 普通行 | 连续 row-block，静态/加权均分 | 根据核数、带宽和行工作量使用更细粒度分块 |
| 重行 | 行内分段并在 Device 归并，避免跨核标量覆盖 | ALG2/ALG3 增加协作 hash/merge 并行度 |
| Numeric | 控制 UB 占用和重复扫描次数 | 依据 Profiler 调整 tile、预取和核间负载 |

DAV-2201 上曾发现多个 AIV 对循环分配的标量 GM 地址写入会造成 cache-line clobber。当前正确性候选因此将 symbolic count/fill 暂时限制为单 AIV。正式实现需要按 cache line 对齐 scratch，并使每个 cache line 由唯一核心拥有，或采用独立局部区加后续 Device merge；不得把单 AIV 方案作为性能最终架构。

A3 必须单独完成官方性能采样后再确定阈值和调度参数，不能使用 A2 结果推断 A3 已达标。

## 8. PyTorch/ATen 接入

目标是在 `ops-sparse` 正式交付树中完成：

```text
torch.sparse.mm
  -> aten::_sparse_sparse_matmul
  -> SparsePrivateUse1 / SparseCsrPrivateUse1 dispatch
  -> layout/shape/dtype/device 校验
  -> 输入规范化为 CSR
  -> aclsparseSpGEMM 多阶段调用
  -> 根据 nnz(C) 分配 NPU 输出
  -> 构造 PyTorch sparse CSR/COO 结果
```

目前完成的是隔离的 PyTorch C++ bridge 验证：它已证明四种 dtype 可以命中 NPU dispatch 并调用当前 `libops_sparse.so`，不是正式 `op-plugin`/`torch-npu` 集成。正式注册、构建依赖、错误语义、layout 转换和端到端 UT 仍需按 `ops-sparse` 交付规范合入，不得以隔离 bridge 代替正式集成。

## 9. 错误处理

| 场景 | 处理 |
| --- | --- |
| handle 为空 | `HANDLE_IS_NULLPTR` |
| 必需指针、描述符或 bufferSize 为空 | `INVALID_VALUE` |
| dtype、operation、格式或索引类型不支持 | `NOT_SUPPORTED` 或对应矩阵类型错误 |
| shape、CSR 边界、排序或描述符前后不一致 | `INVALID_VALUE` |
| GetNumProducts/Compute/Copy 跳过前置阶段 | `NOT_INITIALIZED` |
| workspace 不足、chunkFraction 越界 | `INVALID_VALUE` |
| nnz/乘积数/workspace 算术溢出 | `NOT_SUPPORTED` |
| Device 分配失败 | `ALLOC_FAILED` |
| memcpy、Kernel launch、stream 执行失败 | `EXECUTION_FAILED` |
| count/fill 数量不一致等内部不变量破坏 | `INTERNAL_ERROR` |

同步校验失败时不启动 Kernel、不修改已有效输出。异步执行错误由 runtime 在同步点返回。DestroyDescr 需允许安全释放已创建对象，完整调用链不得泄漏 Host/Device 资源。

## 10. 测试与验收方案

### 10.1 正确性与精度

1. C++ direct-core 回归：FP32、FP16、BF16、Complex64，覆盖方阵、长/宽矩阵、空矩阵、空行、单 nnz、重复归并、显式零和边界；
2. 多阶段 API：query/execute、`GetNumProducts`、ALG2/ALG3 EstimateMemory、workspace 精确值/不足、错误阶段和重复使用；
3. 官方精度：完整 200 case，逐项精确比较 `nnz(C)`、rowOffsets、colIndices，并按 dtype 容差比较 values；
4. PyTorch 端到端：确认 NPU dispatch 命中、返回 layout/shape/dtype/device 正确且无 CPU fallback；
5. A2 与 A3 分别提交功能、精度和 NPU Dispatch/Profiler 证据。

### 10.2 性能

性能范围以随任务提供的官方 performance manifest 为准，共 **50 个 case×dtype 场景**，manifest 中每一行代表一个固定 shape、稀疏度、seed 和 dtype 的验收场景。不得用少量自定义样例代替该范围。

每个场景至少预热 10 次、正式采样 30 次，设备同步后计时，报告中位数、P90、workspace 峰值、`numProducts`、`nnz(C)` 和输出存储量，并分别记录：

- symbolic count；
- prefix scan；
- symbolic fill；
- numeric；
- memory estimation / Copy；
- C++ 多阶段完整流程；
- Python 端到端。

NPU Kernel 总耗时与官方 A100 NCU Kernel 总耗时按同一调用范围比较。A3 每个 case×dtype 性能倍率必须大于 0.25，全部 50 个场景算术平均值不低于 0.35。Profiler 用于区分 symbolic 串行、重行拖尾、重复扫描、UB 容量、访存带宽和启动开销，再驱动 ALG/shape dispatch 优化。

## 11. 当前验证进展

以下结果严格区分旧基线和新候选：

| 项目 | 当前结果 | 说明 |
| --- | --- | --- |
| A2 direct-core 旧稳定基线 | 249/249 通过：FP32 109/109、FP16 70/70、BF16 70/70 | complex64 基础验证通过 |
| Host-symbolic 旧基线官方精度 | 200/200 通过 | 仅证明功能基线；包含 D2H + Host `std::set`，不是目标架构 |
| 隔离 PyTorch bridge | FP16/BF16/FP32/Complex64 通过 | 仅隔离验证，正式集成待完成 |
| A2 device-symbolic 构建 | 通过 | `beta=0` 候选路径 |
| device-symbolic bounded core | 代表性 FP32/FP16/BF16 shape 的 CSR 结构和值通过 | 不是完整官方集合 |
| device-symbolic 官方精度 pilot | **5/5 FP32 通过，matched=1.0** | 尚未运行完整 200 case，不得写成 200/200 |
| A2/A3 正式性能验收 | 未完成 | 当前没有最终达标结论 |

## 12. 遗留项与实施顺序

1. 在 A2 device-symbolic 路径运行完整官方 200 case，并补齐 BF16/FP16/Complex64 证据；
2. 将单 AIV symbolic count/fill 替换为 cache-line 安全的多核实现；
3. 将 `beta*C` 的 D2H + Host `std::set` 迁移为 Device symbolic union；
4. 实现 ALG2/ALG3 的真实 workspace 估算、chunk/hash/merge 路径和自动 dispatch；
5. 将隔离 bridge 正式集成到 `ops-sparse` 规定目录，补齐 PyTorch/ATen 注册和端到端 UT；
6. 在 A2、A3 分别完成完整功能/精度验证，在 A3 跑完官方 50 个 case×dtype 性能场景；
7. 优化 A3 的 symbolic/numeric 调度，解决重行、负载不均和 workspace 峰值问题；
8. 收集自测报告所需的版本、返回码、内存、Profiler、NPU Dispatch 和复现命令证据。

在上述项目完成前，本设计 PR 仅用于方案评审，不代表 A2/A3 算子已经完成最终验收。
