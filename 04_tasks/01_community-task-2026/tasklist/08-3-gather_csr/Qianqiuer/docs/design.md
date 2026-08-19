# GatherCsr 算子设计文档

| 项目 | 内容 |
| --- | --- |
| 任务 | 2026 年 8 月社区任务 `gather_csr` |
| 团队 | Qianqiuer |
| 目标仓库 | `cann/ops-gnn` |
| 目标硬件 | Ascend 950PR（NPU Arch `dav-3510`） |
| 对标版本 | `torch_scatter` 2.1.2 |
| 文档版本 | V1.0 |

本文档依据社区任务书和社区任务设计模板编写。验收以 PyTorch 层接口的功能、精度和性能为准，本文不把可选的 aclnn 接口列入首期实现范围。

# 1. 需求背景（required）

## 1.1 需求来源

本任务要求参考 `torch_scatter.gather_csr`，在 Ascend 950PR 上使用 Ascend C Kernel 和 Python/PyTorch 适配层实现接口、功能和数值行为一致的 Gather CSR 算子，验收通过后贡献至 `ops-gnn` 仓库。

参考来源：

1. 社区任务书：`04_tasks/01_community-task-2026/docs/202608/gather_csr_task_doc.md`；
2. `torch_scatter` 2.1.2：`torch_scatter/segment_csr.py`、`csrc/cpu/segment_csr_cpu.cpp`、`csrc/cuda/segment_csr_cuda.cu`；
3. `torch_scatter` 官方测试：`test/test_gather.py`、`test/test_zero_tensors.py`；
4. `ops-gnn` 开发指南和现有 `segment_max_csr` 分层实现。

## 1.2 背景介绍

CSR（Compressed Sparse Row）用 `indptr` 描述每一段在展开序列中的起止位置。Gather CSR 将每段对应的源特征复制到该段覆盖的全部输出位置，常用于将节点或分段特征下发到边列表。

令：

- `dim = indptr.dim() - 1`；
- 第 `i` 段起止位置为 `rowStart = indptr[..., i]`、`rowEnd = indptr[..., i + 1]`；
- `src[..., i, ...]` 为该段的特征切片。

则：

$$
out[..., rowStart:rowEnd, ...] = src[..., i, ...]
$$

例如：

```text
src    = [10, 20, 30, 40]
indptr = [0, 2, 5, 5, 6]
out    = [10, 10, 20, 20, 20, 40]
```

第三段为空段，因此 `src[2]` 不产生输出。

## 1.3 与相关算子的关系

| 算子 | 索引形式 | 数据方向 | 主要操作 |
| --- | --- | --- | --- |
| `segment_csr` | CSR 指针 | `nnz -> segment` | 分段归约 |
| `gather_csr` | CSR 指针 | `segment -> nnz` | 分段扩展 |
| `gather_coo` | 逐元素 COO 索引 | `segment -> nnz` | 按索引扩展 |

对合法且等价的 CSR/COO 索引，`gather_csr(src, indptr)` 与 `gather_coo(src, index)` 的输出应逐元素一致。Gather CSR 不包含归约、浮点运算或原子操作，理论结果应与 CPU 标杆逐位一致并保持完全确定性。

## 1.4 现有实现分析

### 1.4.1 torch_scatter 行为

`torch_scatter` 的 CPU/CUDA 实现执行以下步骤：

1. 要求 `src.dim() >= indptr.dim()`；
2. 将 `indptr` 的前缀维扩展到 `src` 对应维度；
3. 令 `dim = indptr.dim() - 1`；
4. 将输入转换为连续布局；
5. 将 `src` 归一化为 `[B, S, F]`，将输出归一化为 `[B, E, F]`；
6. 对每个 `(batch, segment)` 读取一次源特征，并写入 `[rowStart, rowEnd)`。

其中：

$$
B=\prod_{d=0}^{dim-1}src.size(d),\quad
S=src.size(dim),\quad
F=\prod_{d=dim+1}^{rank-1}src.size(d),\quad
E=out.size(dim)
$$

官方空张量用例规定：当 `src.numel() == 0` 且未提供 `out` 时，输出形状与 `src` 相同，并将 `dim` 维长度置为 0。

### 1.4.2 ops-gnn 工程现状

`ops-gnn` 当前采用以下四层结构：

```text
Python API -> pybind11 -> C++ Host/Tiling -> Ascend C Kernel
```

仓库已经包含 `segment_max_csr`，可复用其目录组织、PyTorch 扩展构建、平台信息查询和 Kernel 启动方式。Gather CSR 与其在以下方面不同：

- 本任务规定 `indptr` 必须为 `INT64`，不能沿用现有样例的 `INT32`；
- 本任务要求 L1 的 7 种 dtype、非连续 Tensor、可选 `out` 原位写入和严格参数校验；
- Gather 是只读源数据并连续写输出，不需要规约和原子操作；
- 性能验收为 PyTorch 公共接口端到端性能，Host 不应为每次调用创建并同步独立流。

# 2. 需求分析（required）

## 2.1 功能需求

公开接口必须与任务书保持一致：

```python
def gather_csr(
    src: torch.Tensor,
    indptr: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor: ...
```

接口只接受 `src`、`indptr`、`out` 三个参数，不增加 `reduce`、`dim` 或 `dim_size`。本任务仅实现前向，不实现 backward。

## 2.2 输入输出规格

| 参数 | 类型 | 方向 | dtype | Shape | 非连续 |
| --- | --- | --- | --- | --- | --- |
| `src` | Tensor | 输入 | L1/L2，见下表 | 1～8 维 | 支持 |
| `indptr` | Tensor | 输入 | `torch.int64` | 1～`src.dim()` 维 | 支持 |
| `out` | Optional[Tensor] | 输入/输出 | 与 `src` 相同 | 与推导结果一致 | 支持 |
| 返回值 | Tensor | 输出 | 与 `src` 相同 | 与推导结果一致 | 连续，或原 `out` 布局 |

数据类型分级如下：

| 级别 | `src/out` dtype | 实现路径 | 性能验收 |
| --- | --- | --- | --- |
| L1 NPU 原生 | `float16`、`bfloat16`、`float32`、`int8`、`int16`、`int32`、`uint8` | Ascend C 段内复制 | 纳入基线 |
| L2 | `float64`、`int64` | NPU 位级拆分复制 | 不考核 |

不支持 `bool`、复数、Float8 及任务书未列出的 dtype。

## 2.3 Shape 与广播规则

令 `rank = src.dim()`、`ptrRank = indptr.dim()`、`dim = ptrRank - 1`，合法输入满足：

1. `1 <= ptrRank <= rank <= 8`；
2. 对 `0 <= d < dim`，`indptr.size(d)` 必须等于 1 或 `src.size(d)`；
3. `src.numel() != 0` 时，`src.size(dim) == indptr.size(-1) - 1`；
4. `src.numel() == 0` 时允许空 `indptr`，以兼容 `torch_scatter` 的零张量用例；
5. `indptr` 的前缀维按 PyTorch `expand` 规则扩展至 `src.shape[:dim]`；
6. `src` 在 `dim` 之后的维度均为 feature 维，不参与 CSR 索引。

广播后的逻辑形状为：

```text
src:    [B, S, F]
indptr: [B, S + 1]
out:    [B, E, F]
```

这里的三维只是连续内存上的逻辑视图，不改变用户可见 rank。

## 2.4 indptr 合法性

Host 完成元数据校验，设备端校验 Kernel 完成取值校验。每个广播后的 CSR 行必须满足：

1. `indptr[b, i] <= indptr[b, i + 1]`，允许相等以表示空段；
2. 所有指针值位于 `[0, E]`；
3. 每个 batch 的末指针相同，均等于公共输出长度 `E`；
4. 地址和元素数乘积不发生 64 位溢出。

要求公共末指针是因为 PyTorch Tensor 的同一维不能在不同 batch 具有不同长度。非法输入通过 `TORCH_CHECK` 抛出包含参数名和失败原因的异常，不允许越界访问或返回部分结果。

标准 CSR 的首指针为 0。任务书未将首指针为 0 列为硬约束，因此实现不因 `indptr[b, 0] > 0` 拒绝输入；此时先将未被任何段覆盖的前缀 `out[b, 0:indptr[b, 0], ...]` 置零，再执行段内复制，以满足完全确定性。官方精度与性能用例使用首指针为 0 的标准 CSR，不产生额外清零开销。

## 2.5 输出形状与 out 语义

未提供 `out` 且 `src.numel() != 0` 时：

```text
out.shape = src.shape
out.size(dim) = E = indptr.flatten()[-1]
```

未提供 `out` 且 `src.numel() == 0` 时，按 `torch_scatter` 官方零张量行为将 `out.size(dim)` 置为 0。

提供 `out` 且 `src.numel() == 0` 时，完成 dtype、device、rank 和 shape 检查后不启动 Gather Kernel，保持 `out` 内容不变并返回原对象；若 `indptr` 非空，`out.size(dim)` 仍须等于其公共末指针。

提供 `out` 时必须满足：

- 与 `src` rank、dtype、device 相同；
- 除 `dim` 外的所有维度与 `src` 相同；
- 非空输入时 `out.size(dim) == E`；
- 函数写入并返回用户传入的同一个 `out` 对象，而不是替换对象；
- `out` 非连续时通过连续临时 Tensor 执行 Kernel，完成后使用当前 NPU 流上的 `copy_` 写回原始 strided Tensor；
- `out` 与 `src` 存储区重叠时先对 `src` 建立快照，避免写后读风险。

## 2.6 确定性与精度

每个输出元素只有唯一写入者，不使用归约、原子操作或类型转换。L1 dtype 直接复制其原始位模式；L2 的 64 位元素拆成两个 `uint32_t` 搬运并按原顺序写回，同样不执行数值转换。因此所有支持 dtype 均以 `torch.equal`/逐位比较为验收标准。

# 3. 详细设计（required）

## 3.1 总体架构

```text
ops_gnn.gather_csr(src, indptr, out=None)
                 |
                 v
Python 参数透传与公开接口
                 |
                 v
pybind11: gather_csr
                 |
                 v
C++ Host
  - dtype/device/rank/shape/out 校验
  - indptr 广播、非连续输入规整
  - ValidateIndptr Kernel + 动态输出申请
  - Tiling 与当前 PyTorch NPU stream
                 |
                 v
Ascend C Gather Kernel
  - SegmentMajor: 正常 CSR 分布
  - OutputMajor: 段数少或长度严重倾斜
                 |
                 v
连续输出，或 copy_ 写回非连续 out
```

计划新增和修改的文件：

```text
ops-gnn/
├── csrc/pybind.cpp                                  # 注册 gather_csr
├── csrc/npu/host/gather_csr/
│   ├── gather_csr.h
│   └── gather_csr.cpp                               # 校验、规整、Tiling、Launch
├── csrc/npu/kernel/gather_csr/
│   ├── gather_csr_kernel.h
│   ├── gather_csr_kernel.cpp                        # Kernel Launch 与模板实例化
│   ├── gather_csr_kernel_impl.h                     # 两种调度路径
│   ├── gather_csr_validate_kernel.cpp               # indptr 设备校验
│   └── gather_csr_tiling.h
├── python/ops_gnn/
│   ├── gather_csr.py
│   └── __init__.py                                  # 导出 gather_csr
├── test/test_gather_csr.py
└── docs/zh/api_reference.md                         # 接口说明
```

## 3.2 Python 与 PyBind 设计

Python 层只保留类型标注、docstring 和调用转发，不改变参数名和默认值：

```python
def gather_csr(
    src: Tensor,
    indptr: Tensor,
    out: Optional[Tensor] = None,
) -> Tensor:
    return _pybind.gather_csr(src, indptr, out)
```

PyBind 注册原型为：

```cpp
m.def(
    "gather_csr",
    &gather_csr,
    py::arg("src"),
    py::arg("indptr"),
    py::arg("out") = py::none());
```

Python 包在 `ops_gnn.__init__` 中导出 `gather_csr`。测试以 `ops_gnn.gather_csr` 作为 NPU 被测接口，以 CPU `torch_scatter.gather_csr` 作为 Golden。

## 3.3 Host 侧设计

### 3.3.1 参数检查顺序

Host 依次执行以下检查：

1. `src`、`indptr` 已定义且位于同一 NPU；
2. `src.dtype` 属于 L1/L2，`indptr.dtype == int64`；
3. rank、前缀广播、段数关系合法；
4. `out` 存在时检查 device、dtype、rank 和静态 shape；
5. 使用带溢出检查的 `int64_t/uint64_t` 计算 `B/S/F` 和元素数；
6. 对非连续 `src` 执行 `contiguous()`；
7. 将 `indptr` 按前缀维 `expand` 后执行 `contiguous()`，得到 `[B, S+1]`；
8. 启动设备端 `ValidateIndptr`，取得公共 `E`、最大段长和错误状态；
9. 申请或选择 Kernel 输出，计算 Tiling 并在当前 NPU stream 启动 Gather Kernel；
10. 非连续 `out` 执行 `out.copy_(contiguousOut)`，返回原始 `out`。

所有 NPU 操作使用调用方当前 PyTorch NPU stream，保持与前后 PyTorch 算子的执行顺序。除动态 shape/严格校验需要取得小量状态外，不创建、同步和销毁私有 stream。

### 3.3.2 indptr 设备校验

直接将完整 `indptr` 拷回 CPU 会引入与段数成正比的 D2H 开销。设计使用轻量设备校验 Kernel：

- 每个 AIV 分配若干完整 CSR 行，避免跨核边界状态；
- 顺序检查首指针、相邻单调性、范围和公共末指针；
- 同时统计本核最大段长和是否存在未覆盖前缀；
- 每个 AIV 写一个独立状态记录，不使用不必要的原子操作；
- Host 仅取回 `coreNum` 个固定长度状态记录并汇总；
- 未提供 `out` 时，动态申请输出本就需要取得末指针，因此把合法性检查合并进同一次同步。

状态记录包含 `errorCode`、`batchIndex`、`segmentIndex`、`badValue`、`endpoint` 和 `maxSegmentLength`。错误码用于生成稳定、可定位的异常信息。

### 3.3.3 非连续张量处理

首期 Kernel 使用连续 `[B,S,F]`/`[B,E,F]` 布局，以保证连续读写带宽：

| 场景 | 处理方式 |
| --- | --- |
| 连续 `src` | 直接使用 |
| 非连续 `src` | `src.contiguous()` |
| 连续 `indptr` 且无需广播 | 直接使用 |
| 非连续或广播 `indptr` | `indptr.expand(...).contiguous()` |
| 未提供 `out` | 创建连续输出 |
| 连续 `out` | Kernel 直接写入 |
| 非连续 `out` | Kernel 写临时连续输出，再 `out.copy_()` |

非连续转换属于公共接口端到端耗时。性能基线用例为连续 Tensor，非连续路径以功能和精度为验收重点。

### 3.3.4 L2 数据类型策略

Gather CSR 只有复制，没有 FP64/INT64 算术限制。对 `float64/int64`，Host 将每个逻辑元素的拷贝宽度设置为 8 字节，Kernel 把元素视为两个连续 `uint32_t`：

```text
[high/low 32-bit word] -> 原样搬入 UB -> 原样搬出
```

该方案不进行 `float64 -> float32` 或 `int64 -> int32` 转换，不损失 NaN payload、无穷、符号位或整数高位，结果逐位一致。L2 路径仍不参加性能考核。

## 3.4 Tiling 设计

### 3.4.1 Tiling 数据

`GatherCsrTilingData` 至少包含：

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `batchCount` | `uint64_t` | $B$ |
| `segmentCount` | `uint64_t` | $S$ |
| `outputRows` | `uint64_t` | $E$ |
| `featureElements` | `uint64_t` | $F$ |
| `featureBytes` | `uint64_t` | 单个特征切片字节数 |
| `srcElements` | `uint64_t` | `src.numel()` |
| `outElements` | `uint64_t` | `out.numel()` |
| `tileBytes` | `uint32_t` | 单次 UB 特征块大小 |
| `elementBytes` | `uint32_t` | 1、2、4 或 8 |
| `activeCoreNum` | `uint32_t` | 实际启用 AIV 数 |
| `scheduleMode` | `uint32_t` | SegmentMajor/OutputMajor |

地址偏移和总元素数使用 64 位计算，避免大输出下 32 位乘法溢出。`tileBytes` 按 32 字节对齐，并由平台接口查询到的可用 UB 容量动态计算，不硬编码 950PR 的全部 UB。

### 3.4.2 Tiling Key

| Tiling Key | 条件 | 处理方式 |
| --- | --- | --- |
| `EMPTY` | `out.numel()==0` | 不启动 Gather Kernel |
| `SEGMENT_MAJOR` | 段数充足且长度分布正常 | 按 `(batch, segment)` 分核 |
| `OUTPUT_MAJOR` | `B*S < coreNum` 或最长段明显大于平均段长 | 按输出行范围分核 |

建议初始选择条件为：

```text
OUTPUT_MAJOR if B*S < activeCoreNum
             or maxSegmentLength > 4 * ceil(E / max(S, 1))
otherwise SEGMENT_MAJOR
```

阈值只影响性能，不影响结果；后续根据 950PR 实测调整。

## 3.5 Kernel 侧设计

### 3.5.1 SegmentMajor 路径

逻辑工作单元为 `(batch, segment)`。每个 AIV 处理一段连续工作单元，执行：

```text
rowStart = indptr[b, s]
rowEnd   = indptr[b, s + 1]

for featureTile in [0, F):
    src[b, s, featureTile] GM -> UB          # 每个特征块只读一次
    for row in [rowStart, rowEnd):
        UB -> out[b, row, featureTile] GM    # 段内连续写
```

空段满足 `rowStart == rowEnd`，直接跳过。不同工作单元写入区间不重叠，不需要核间同步或原子操作。

### 3.5.2 OutputMajor 路径

当段数少或分布严重倾斜时，按 `B*E` 个输出行均匀切分：

1. 每个 AIV 得到连续的全局输出行范围；
2. 解码出 batch 和 batch 内输出位置；
3. 对本核首个位置在 `indptr[b]` 中二分定位 segment；
4. 随后顺序推进 segment 边界，不对每个输出行重复二分；
5. 对同一 segment 的连续输出 run，仍采用“源特征块只加载一次、连续写多个输出行”。

该路径保证超长段可以被多个 AIV 拆分，同时每个 AIV 的输出地址仍连续。不同 AIV 的输出区间互不重叠。

### 3.5.3 UB 与搬运流水

Kernel 使用 AIV 和 `TPipe/TQue` 管理 UB：

- 从平台信息取得可用 UB；
- 预留事件和尾块处理空间后，将其划分为两个源特征缓冲；
- 相邻 segment/feature tile 使用双缓冲隐藏 GM 到 UB 的读延迟；
- 单个源特征 tile 可向多个连续输出行重复发起 UB 到 GM 搬运；
- 复用缓冲前通过对应 MTE2/MTE3 事件保证前序读写完成。

对 32 字节对齐的完整块使用 `DataCopy`；首尾不足 32 字节的有效范围使用 `DataCopyPad` 或等价的安全尾块接口，禁止越过 Tensor 边界读写。8 字节 L2 元素按两个 4 字节 word 搬运，但 tile 边界始终保持完整逻辑元素。

### 3.5.4 dtype 分发

Kernel 不执行数值计算，按元素字节宽度复用实现：

| 模板存储类型 | 对应 PyTorch dtype |
| --- | --- |
| `uint8_t` | `int8`、`uint8` |
| `uint16_t` | `float16`、`bfloat16`、`int16` |
| `uint32_t` | `float32`、`int32` |
| `uint32_t[2]` 位搬运 | `float64`、`int64` |

这种分发方式避免把 BF16 当数值参与转换，也减少模板实例数量。

## 3.6 边界与异常处理

| 场景 | 设计行为 |
| --- | --- |
| `src.numel()==0` | 未提供 `out` 时返回 `dim` 维为 0 的空 Tensor；不启动 Gather Kernel |
| `indptr.size(-1)==0` | 仅在零张量兼容路径允许，否则报错 |
| `indptr.size(-1)==1` | 0 个 segment；合法输出长度为 0 |
| 空段 | 跳过，不读取对应 `src` 特征 |
| 所有段为空 | 返回长度 0 的输出 |
| 首指针大于 0 | 将未覆盖输出前缀置零，再执行段内复制 |
| 单个超长段 | 选择 OutputMajor，多核拆分输出区间 |
| feature 维乘积为 0 | 返回空输出，不启动 Gather Kernel |
| 非连续输入/out | 连续 staging，结果写回原 `out` |
| batch 广播 | 展开并物化紧凑 `indptr` 后执行 |
| batch 末指针不同 | 报错，避免无法表示的变长 batch 输出 |
| shape/offset 溢出 | Host 报错，不启动 Kernel |
| `out` 与输入别名 | 快照源输入后写出 |
| 不支持 dtype/device | Host 报错 |

## 3.7 支持硬件

| 芯片版本 | NPU Arch | 支持 |
| --- | --- | --- |
| Ascend 950PR | `dav-3510` | 是 |
| Ascend 910B/A2 | `dav-2201` | 不在本任务验收范围 |
| Ascend 910_93/A3 | - | 不在本任务验收范围 |

构建使用 `ops-gnn` 指定的 CANN、PyTorch 和 `torch_npu` 版本。当前仓库要求 PyTorch 2.7+、`torch_npu` 26.0.0+，CANN 具体版本以开发和合入时 `ops-gnn` 主干要求为准。

## 3.8 算子约束限制

1. `src`、`indptr` 和可选 `out` 必须位于同一 NPU；
2. `indptr` 固定为 `INT64`，按最后一维表示 CSR 指针；
3. `indptr.dim() <= src.dim()`，`dim = indptr.dim() - 1`；
4. 非空输入的 segment 数必须与 `src.size(dim)` 一致；
5. 每个 CSR 行非降序、范围合法并具有相同末指针；首指针大于 0 时确定性补零；
6. 只支持前向 Gather，不支持 reduce 和变长 batch 输出；
7. 支持的 dtype、rank 和硬件范围以第 2、3 节为准；
8. 非连续 Tensor 通过公共接口内部规整，Kernel 本体只处理连续布局。

## 3.9 明确不在本期范围的能力

1. backward/自动求导；
2. `reduce`、`dim`、`dim_size` 参数；
3. aclnn C 接口；
4. CPU 设备实现；
5. 非任务书 dtype。

# 4. 性能设计

## 4.1 性能模型

Gather CSR 的主要流量为：

$$
Bytes \approx B\times S\times F\times sizeof(T)
+B\times(S+1)\times8
+B\times E\times F\times sizeof(T)
$$

当平均段长大于 1 时，输出写流量占主导，因此算子属于内存带宽受限操作。优化目标不是减少算术，而是减少源特征重复读取、确保输出连续写并保持各 AIV 负载均衡。

## 4.2 优化措施

1. **段内连续写**：一个工作单元处理连续输出区间，避免 COO 式逐元素索引开销；
2. **源特征复用**：源 feature tile 只搬入 UB 一次，服务该段的多次输出；
3. **双调度路径**：正常分布使用 SegmentMajor 降低索引处理开销，长尾分布使用 OutputMajor 保证多核利用率；
4. **连续布局快路径**：性能基线不引入 stride 地址计算；
5. **动态 UB tile**：根据平台 UB 容量和 dtype 设置 tile，减少小块 DMA；
6. **双缓冲**：重叠下一特征块的 GM 读取和当前特征块写出；
7. **减少 Host 往返**：取回固定大小校验结果，不复制完整 `indptr`；
8. **当前流异步执行**：不为每次调用创建私有 stream，不做多余全流同步；
9. **位宽复用**：相同字节宽度共享 Kernel，避免数值转换；
10. **零 workspace 主计算**：除校验状态和必要的非连续 staging 外，不申请与输出规模相关的中间 Tensor。

## 4.3 性能验收阈值

任务书要求 NPU 性能不低于 A100 标杆的 0.6 倍。以吞吐率定义：

$$
PerformanceRatio=\frac{1/t_{NPU}}{1/t_{A100}}=\frac{t_{A100}}{t_{NPU}}\ge0.6
$$

因此端到端 NPU 耗时上限为 `A100 时间 / 0.6`：

| 输出 Shape，segment 数 | A100 FP32 | NPU FP32 上限 | A100 FP16 | NPU FP16 上限 |
| --- | ---: | ---: | ---: | ---: |
| `256K x 128`，512 | 0.100 ms | 0.167 ms | 0.060 ms | 0.100 ms |
| `512K x 128`，1024 | 0.176 ms | 0.293 ms | 0.096 ms | 0.160 ms |
| `1024K x 128`，1024 | 0.330 ms | 0.550 ms | 0.174 ms | 0.290 ms |
| `1024K x 128`，2048 | 0.333 ms | 0.555 ms | 0.175 ms | 0.292 ms |
| `2048K x 128`，4096 | 0.638 ms | 1.063 ms | 0.326 ms | 0.543 ms |
| `4096K x 128`，4096 | 1.246 ms | 2.077 ms | 0.627 ms | 1.045 ms |

性能测试以公共 PyTorch API 为验收口径，包含参数处理、动态输出申请和 Kernel。预热后使用 NPU 同步边界测量至少 100 次，报告中记录中位数、P90、输出有效带宽和性能比例；Kernel-only 数据仅用于定位瓶颈，不替代端到端结果。

# 5. 可维可测分析

## 5.1 精度标准

Gather CSR 是位复制操作，所有支持 dtype 的预期结果均为 bit-wise 一致：

| 路径 | 标准 |
| --- | --- |
| L1 7 种 dtype | 与 CPU `torch_scatter.gather_csr` 逐位一致 |
| L2 `float64/int64` | 与 CPU 标杆逐位一致 |
| CSR/COO 等价性 | 与 CPU/NPU 等价 `gather_coo` 逐位一致 |
| 确定性 | 同一输入重复执行结果完全相同 |

使用 AscendOpTest 生成执行日志。由于算子无数值误差，首选 `torch.equal` 和字节级比较，不使用宽松的 `rtol/atol`。仅当单标杆流程无法满足生态精度工具要求时，按任务书使用 ATK `cv_fused_double_benchmark` 补充双标杆报告。

## 5.2 功能测试设计

| 编号 | 场景 | 主要检查 |
| --- | --- | --- |
| TC-01～TC-06 | 移植 `torch_scatter/test/test_gather.py` 标准数据 | 数值、shape、dtype、device |
| TC-07 | CSR/COO 等价索引 | `gather_csr == gather_coo` |
| TC-08 | 提供连续 `out` | 返回对象身份、原位写入、结果 |
| TC-09 | 非连续 `src/indptr/out` | stride 规整和写回 |
| TC-10 | 官方零张量 `(0,0,0,16)` | 输出 shape、无非法访问 |
| TC-11 | 空段、连续多个空段、全空段 | 区间边界 |
| TC-12 | `float64` L2 | 特殊值和随机位模式逐位一致 |
| TC-13 | `int64` L2 | 包含 32 位范围外值逐位一致 |
| TC-14 | L1 全 dtype | 7 种 dtype 全覆盖 |
| TC-15 | rank 1～8 | 不同 `dim` 和 feature 组合 |
| TC-16 | batch 广播 | 1D/2D/多前缀 `indptr` |
| TC-17 | 大 feature 与非 32B 对齐尾块 | `F=1,3,7,31,33,127,129` |
| TC-18 | 极端段长分布 | 单超长段、长短交替、段数少于核数 |
| TC-19 | 可复现随机泛化 | 随机合法 CSR、空段比例和 batch |
| TC-20 | 重复执行 | 确定性和内存越界检查 |

测试维度进行笛卡尔和分层抽样，确保每种 L1 dtype 至少覆盖：rank 边界、空段、非连续、batch 广播、对齐/非对齐 feature 和提供/不提供 `out`。

## 5.3 参数校验测试

下列非法输入均应在 Host/校验阶段稳定报错：

1. `indptr` 非 `int64`；
2. CPU/NPU 混用或设备不一致；
3. `indptr.dim() > src.dim()` 或 rank 超范围；
4. 前缀维不可广播；
5. segment 数与 `src.size(dim)` 不一致；
6. `indptr` 下降、负数或越界；
7. batch 末指针不一致；
8. `out` dtype/device/rank/shape 不一致；
9. 不支持的数据类型；
10. shape 乘积或地址计算溢出。

## 5.4 性能测试设计

1. 完整覆盖任务书 6 组 shape 的 FP16、FP32，共 12 组必测基线；
2. 对 L1 其余 dtype 做性能回归，确认无异常慢路径；
3. 分别构造均匀、Zipf 长尾、单超长段三类 `indptr`；
4. 对比 SegmentMajor 与 OutputMajor，记录调度选择和有效带宽；
5. 分离记录公共 API、Host/校验、连续化、Kernel 时间；
6. 性能测试前固定 NPU、锁定相同软件版本、预热并清除首次编译影响；
7. 对等价 CSR/COO 索引补充与 `gather_coo` 的同机对照，目标为 Gather CSR 不慢于 Gather COO；
8. 每个 case 校验输出后再计时，避免错误 Kernel 获得虚假性能结果。

## 5.5 自验证产物

自验证报告至少包含：

- 硬件型号、CANN/PyTorch/torch_npu/ops-gnn commit；
- 完整测试命令和环境变量；
- 每个功能 case 的 shape、dtype、stride、结果；
- AscendOpTest 日志，必要时附 ATK 日志；
- 12 组必测性能的原始耗时、统计值和性能比例；
- 非连续和 L2 路径说明；
- 失败用例、修复记录和复测结果；
- 可复现 README。

# 6. 兼容性与风险分析

## 6.1 兼容性

本算子是 `ops-gnn` 新增接口，不修改已有算子行为。公共函数签名与 `torch_scatter.gather_csr` 一致；L1/L2 dtype、输出 shape、`out` 写入和空张量行为均以任务书及 `torch_scatter` 2.1.2 为基准。

## 6.2 风险与应对

| 风险 | 影响 | 应对措施 |
| --- | --- | --- |
| 动态输出长度需要读取 `indptr[-1]` | Host 同步可能影响小 case | 与完整合法性检查合并为一次固定小状态 D2H；提供 `out` 时避免输出申请 |
| 段长严重倾斜 | SegmentMajor 核间不均衡 | 校验阶段统计最大段长，选择 OutputMajor |
| 小 feature 导致 DMA 指令占比高 | 带宽利用率下降 | 小宽度专项实测，必要时增加 950PR SIMT 小包写路径 |
| 非连续输入复制开销 | 端到端时间增加 | 连续快路径零额外 staging；非连续首先保证语义正确 |
| 64 位类型 API 支持差异 | 编译或搬运受限 | 使用 `uint32_t` 双 word 位搬运，不实例化 FP64/INT64 向量算术 |
| 大 shape 的 32 位偏移溢出 | 错误地址 | Host 和 Kernel 偏移统一使用 64 位并做乘法溢出检查 |
| 非连续 `out` 被临时 Tensor 替换 | 破坏原位语义 | 显式 `copy_` 回原对象并测试 `data_ptr`/对象身份 |
| 当前 stream 接入不正确 | 数据竞争或多余同步 | 使用 torch_npu 当前流并加入前后算子链式测试 |

# 7. 交付与评审检查项

| 检查项 | 设计结论 |
| --- | --- |
| PyTorch 三参数接口 | 必选并严格保持参数名/default |
| Ascend C Kernel | 必选，AIV 段内连续写 |
| aclnn | 首期不实现，任务书允许可选 |
| 950PR | 唯一验收硬件，`dav-3510` |
| L1 7 dtype | NPU 原生位复制，全部覆盖 |
| L2 2 dtype | NPU 双 `uint32` 位复制，不考核性能 |
| `indptr` INT64 | Host 和 Kernel 全链路保持 INT64 |
| batch 广播 | `expand + contiguous` 后归一化为 `[B,S+1]` |
| 非连续 Tensor | staging + `out.copy_` 写回 |
| 空张量/空段 | 专门分支和测试 |
| 确定性 | 每个输出唯一写入者，无原子 |
| 精度 | 全 dtype bit-wise 一致 |
| 性能 | 12 组官方基线，要求 `>=0.6x` A100 |
| 自测材料 | pytest、AscendOpTest/必要时 ATK、性能日志和可复现 README |

# 8. 参考资料

1. `cann-ops-competitions`：2026 年 8 月 `gather_csr` 任务书；
2. `cann-ops-competitions`：社区任务设计文档模板；
3. `rusty1s/pytorch_scatter` 2.1.2：`segment_csr.py`；
4. `rusty1s/pytorch_scatter` 2.1.2：`segment_csr_cpu.cpp`、`segment_csr_cuda.cu`；
5. `rusty1s/pytorch_scatter` 2.1.2：`test_gather.py`、`test_zero_tensors.py`；
6. `cann/ops-gnn`：README、开发指南及 `segment_max_csr` 示例；
7. 《生态算子开源精度标准》；
8. AscendOpTest、ATK 使用文档。
