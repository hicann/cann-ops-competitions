# Gather COO 算子设计文档

# 需求背景（required）

## 需求来源

本需求来源于“8 月社区任务 - gather_coo 算子开发”。目标是在 Ascend 950PR 上，基于 Ascend C Kernel 与 Python PyTorch 适配层实现与 `torch_scatter.gather_coo` 功能、接口和结果一致的 Gather COO 算子。

参考接口：

```python
def gather_coo(
    src: torch.Tensor,
    index: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    ...
```

Gather COO 是 Segment COO 的逆向数据变换，典型用途包括图神经网络中的节点特征到边特征广播、消息扩展和 segment 结果恢复。

## 背景介绍

设 `dim = index.dim() - 1`，对于 `index` 中的每个位置 `e`：

```text
idx = index[..., e]
out[..., e, ...] = src[..., idx, ...]
```

`index` 沿最后一维非降序排列。同一个索引值可能连续出现，此时对应的源行或源切片需要复制到多个连续输出位置。

与 Segment COO 的关系如下：

| 算子 | 数据方向 | 输入输出规模变化 |
| --- | --- | --- |
| Segment COO | 多个元素聚合为一个分组 | 压缩 |
| Gather COO | 一个分组广播到多个位置 | 扩展 |

Gather COO 不涉及浮点计算、归约或原子操作，本质为“索引读取 + 数据搬运”。因此精度目标是与 CPU `torch_scatter.gather_coo` bit-wise 一致，性能优化重点是减少重复索引解析、地址计算、GM 访问和同步开销。

# 需求分析（required）

## 需求描述

在 Ascend 950PR 上实现 Gather COO，满足以下要求：

1. PyTorch 接口严格保持 `gather_coo(src, index, out=None)`，不增加额外参数。
2. `index` 固定为 INT64，并沿最后一维非降序排列。
3. 支持 1～8 维、空张量、多 batch、非连续 `src/index/out`。
4. 未提供 `out` 时创建输出；提供 `out` 时复用传入 Tensor 的存储。
5. L1 原生支持 FLOAT16、BFLOAT16、FLOAT32、INT8、INT16、INT32、UINT8。
6. FLOAT64、INT64 作为 L2 可选路径，不参与性能验收。
7. 输出 dtype 和 device 与 `src` 保持一致。
8. 使用当前 PyTorch NPU stream 异步调度，不在算子内部创建、同步或销毁私有 stream。

## 需求拆解

### 功能与形状

设：

```text
dim          = index.dim() - 1
outerSize    = product(src.shape[0:dim])
srcDimSize   = src.shape[dim]
indexCount   = index.shape[dim]
innerSize    = product(src.shape[dim + 1:])
totalElements = outerSize * indexCount * innerSize
```

输出 shape 与 `src` 相同，仅将第 `dim` 维替换为 `indexCount`。

### 数据类型

| 输入 | 支持类型 | 实现方式 |
| --- | --- | --- |
| src/out | FP16、BF16、FP32、INT8、INT16、INT32、UINT8 | Ascend C 原生 Kernel |
| index | INT64 | Ascend C 原生读取 |
| src/out L2 | FP64、INT64 | 可选回退路径 |

Gather COO 为纯比特搬运。FP16/BF16 使用 16-bit 有符号载体，UINT8 使用 8-bit 有符号载体进行 bit-copy，不改变数据位模式。

### 正确性与性能路径拆分

算子采用双路径设计：

1. 通用 SIMT 路径负责完整语义覆盖和复杂 shape 兜底。
2. 行搬运快路径负责连续二维或可展平为二维的主流性能场景。

Python 适配层对 dtype、device、shape、stride 和 `out` 进行检查。当前 Kernel 快路径只接收满足其布局条件的 Tensor；非连续 Tensor 进入通用参考路径，从而保证接口完整性。

# 详细设计（required）

## 算子分析

### 数学公式

对展平后的三维逻辑视图：

```text
src   : [outerSize, srcDimSize, innerSize]
index : [outerSize, indexCount]
out   : [outerSize, indexCount, innerSize]
```

计算公式为：

```text
out[outer, e, inner] =
    src[outer, index[outer, e], inner]
```

### 计算特征

- 无算术计算和类型转换。
- 无归约和原子操作。
- 输出规模通常大于源数据规模。
- 索引有序，同一个 `idx` 常连续出现。
- 性能主要受索引读取、地址计算、GM 读取和 GM 写回影响。
- 官方性能场景中 `innerSize=128`，FP16 每行 256 bytes，FP32 每行 512 bytes，均满足 32-byte 对齐搬运要求。

## 算子实现

### Python 适配层设计

Python 层负责：

1. 检查 `src`、`index` 和 `out` 是否位于同一设备。
2. 检查 `index.dtype == torch.int64`。
3. 推导并校验输出 shape。
4. 检查 `out` 的 dtype、device 和 shape。
5. 空张量直接返回正确 shape，并在提供 `out` 时保持 TensorImpl 复用。
6. 连续且 dtype 支持时调用原生 NPU Kernel。
7. 非连续输入或非连续 `out` 使用 PyTorch 参考路径，保证 stride 语义和结果正确。
8. L2 dtype 使用可选回退路径。

### Host 侧设计

Host 侧将任意 1～8 维输入展平为 `outerSize/srcDimSize/indexCount/innerSize` 四个逻辑参数，并生成 Tiling 数据：

```cpp
struct GatherCooTilingData {
    uint64_t totalElements;
    uint64_t outerSize;
    uint64_t indexCount;
    uint64_t innerSize;
    uint64_t srcDimSize;
    uint64_t rowCopyMode;
};
```

Host 根据 shape 与布局选择路径：

| 条件 | 路径 |
| --- | --- |
| 连续、`outerSize=1`、`innerSize=128` | 行搬运快路径 |
| 其他连续通用 shape | SIMT 通用路径 |
| 非连续 Tensor | Python 参考路径 |

Kernel 使用 `c10_npu::getCurrentNPUStream()` 获取当前 stream，保持与上下游 PyTorch 算子的异步顺序，不执行 host 侧强制同步。

### Kernel 通用 SIMT 路径

通用路径将输出视为一维数组，每个 SIMT thread 处理若干输出元素：

```text
inner = linearOut % innerSize
e     = (linearOut / innerSize) % indexCount
outer = linearOut / (indexCount * innerSize)
idx   = index[outer * indexCount + e]
```

随后计算源地址并复制一个元素。

该路径优势是逻辑简单、shape 泛化能力强，适用于小规模或不满足快路径条件的输入；缺点是每个元素都存在整数除法、取模和索引读取，不适合作为大规模性能主路径。

### Kernel 行搬运快路径

快路径按完整输出行分核。每个 AIV Core 负责若干 `index` 行：

```text
row = blockIdx + k * blockNum
idx = index[row]
srcOffset = idx * innerSize
outOffset = row * innerSize
```

每行只读取一次 INT64 索引，通过以下流程完成搬运：

```text
GM source row -> UB buffer -> GM output row
```

相比逐元素 SIMT 路径，该路径消除每个元素上的除法、取模和重复 index 读取，使主要工作转移到 MTE2/MTE3 搬运流水线。

### 有序索引分段缓存

由于 `index` 非降序，相同索引通常连续出现。最终优化版本按 segment 扫描：

1. 读取当前 `idx`。
2. 识别连续相同 `idx` 的 run length。
3. 将 `src[idx]` 只搬入 UB 一次。
4. 将 UB 中缓存的整行写入多个连续输出位置。

该策略减少重复源行 GM 读取，尤其适用于输出行数显著大于源行数的扩展场景。

### 多行分块与双缓冲

单行原型会产生较多逐行事件同步。最终快路径将多行组成一个 tile，并采用 ping-pong 双缓冲：

```text
buffer 0: CopyIn(tile n)   -> CopyOut(tile n)
buffer 1: CopyIn(tile n+1) -> CopyOut(tile n+1)
```

通过交替使用两组 UB buffer，使 MTE2 读取与 MTE3 写回尽可能重叠。Tile 行数根据 dtype、UB 容量、`indexCount` 和 AIV Core 数动态选择。

### 分核策略

1. 获取可用 AIV Core 数量。
2. 小规模输入减少 block 数，避免空核和启动开销。
3. 大规模输入优先满核。
4. 以输出行作为基本调度单位，保证不同核心写入区域互不重叠。
5. 对有序索引分段路径，避免将同一连续 segment 过度拆分到多个核心。

### 对齐与尾块处理

- `innerSize=128` 的所有 L1 dtype 行字节数均为 32 bytes 的整数倍，使用 `DataCopy`。
- 通用非对齐行宽使用 `DataCopyPad` 或 SIMT fallback。
- 空输入不启动 Kernel。
- 尾行由最后一轮 tile 单独处理，不越界读取或写入。

### 可选 out 处理

提供 `out` 时：

- 校验 shape、dtype 和 device。
- 直接将 Kernel 输出写入传入 Tensor。
- 返回值与传入 `out` 共享同一个 TensorImpl。

未提供 `out` 时，由适配层创建连续输出 Tensor。

## 支持硬件

| 支持芯片 | 状态 |
| --- | --- |
| Ascend 950PR | 支持 |

## 算子约束限制

1. `index` 必须为 INT64。
2. `index` 沿最后一维非降序。
3. `index` 取值范围为 `[0, src.size(dim)-1]`。
4. `index.dim() <= src.dim()`。
5. `index` 前 `index.dim()-1` 维与 `src` 对齐。
6. 仅支持前向，不验收梯度。
7. 接口无 `reduce` 和 `dim_size` 参数。

# 可维可测分析

## 精度标准/性能标准

| 验收项 | 标准 | 验证方式 |
| --- | --- | --- |
| L1 精度 | 与 CPU torch_scatter bit-wise 一致 | `torch.equal` 全量比较 |
| out 语义 | 复用传入 out | TensorImpl identity 检查 |
| 非连续 Tensor | 结果与 CPU 标杆一致 | 构造非连续 src/index/out |
| 空张量 | shape、dtype、device 正确 | 零维长度用例 |
| 性能 | 所有官方用例不低于 0.6 倍标杆 | NPU Event 与端到端计时 |

## 测试设计

正确性测试覆盖：

1. 标准一维、二维和多维输入。
2. 1～8 维 shape。
3. 多 batch。
4. 空 `src/index/out`。
5. 非连续 `src/index/out`。
6. 提供和不提供 `out`。
7. 全部 L1 dtype。
8. 重复、有序、边界索引。
9. 与 Segment COO 的可选互逆验证。
10. 非默认 NPU stream 与下游依赖顺序。

性能测试覆盖任务书规定的 5 个 shape，分别测试 FP32 和 FP16，共 10 个用例。测试排除首次编译和冷启动影响，使用 warmup、多轮重复与中位数统计，并记录 NPU Event 时间和 PyTorch API 端到端时间。

## 阶段验证状态

当前已完成：

- 7 个 L1 dtype 原生 NPU Kernel 路径。
- Python API、可选 `out`、空张量和非连续 Tensor 语义验证。
- 当前 NPU stream 异步接入。
- 通用 SIMT baseline。
- 单行 GM→UB→GM DataCopy 原型。
- 仓库 Gather COO 回归测试通过。
- msprof 确认逐元素地址计算为 baseline 主要瓶颈。

单行 DataCopy 原型已验证优化方向，但仍存在逐行事件同步开销。后续通过有序段缓存、多行 tile 和双缓冲完成性能收敛。最终官方性能矩阵将在实现稳定后补充至自测报告。

## 兼容性分析

Gather COO 为新增算子，不修改已有算子接口。Python 适配层只新增公开 API 和对应 pybind 注册。通用路径保留完整语义兜底，快路径仅在满足明确条件时启用，因此不会影响非标准 shape 的正确性。

# 风险分析与规避措施

| 风险 | 规避措施 |
| --- | --- |
| 非连续 Tensor 地址计算复杂 | 适配层走参考路径，连续场景进入 Kernel 快路径 |
| INT64 index 读取开销 | 每行只读取一次，并利用有序 segment 缓存 |
| 逐行事件同步开销 | 多行 tile 与双缓冲流水 |
| 小 shape 快路径初始化收益不足 | Host 动态选择 SIMT 路径 |
| 不同 dtype 搬运实现重复 | 使用相同 bit-copy 模板和 signed carrier |
| stream 顺序错误 | 使用当前 NPU stream，并增加非默认 stream 压力测试 |

# 开发计划

1. 完成 L1 dtype、API 和通用正确性路径。
2. 完成当前 stream 接入和稳定性回归。
3. 完成单行 DataCopy 原型及 profiling。
4. 实现有序段缓存、多行分块和双缓冲。
5. 跑通官方 10 个性能用例并完成参数调优。
6. 补充完整自测报告、复现步骤和性能证据。
7. 提交代码 PR 和社区任务验收。

# 参考资料

1. `torch_scatter.gather_coo`
2. `torch_scatter/csrc/cpu/segment_coo_cpu.cpp`
3. Gather COO 社区任务书
4. Ascend C `DataCopy`、`DataCopyPad`、`TPipe` 和 `TBuf` 接口文档
