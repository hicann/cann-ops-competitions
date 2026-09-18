# aclsparseScatter 算子开发设计文档（A2/A3）

# 需求背景（required）

## 需求来源

本任务来源于 CANN 社区任务 2026“9 月社区任务-aclsparseScatter 算子开发（A2/A3）”。目标是在 Atlas A2/A3（DAV_2201，arch22）补齐 `aclsparseScatter` 的完整工程能力，并通过标准 PyTorch API `Tensor.index_copy_` 提供 NPU 调用链。

## 背景介绍

Scatter 将稀疏向量的 values 按 indices 原地写入稠密向量：

```text
Y[X.indices[i] - idxBase] = X.values[i], i in [0, nnz)
```

主线 arch22 基线仅覆盖 FP32、I32、base 0，且 Python/ATen 入口、五种 dtype、base 1、异常语义和完整验收测试尚未形成闭环。本任务要求与 cuSPARSE Scatter 的描述符接口语义以及 PyTorch `index_copy_` 的原地、stream 和异常语义对齐，核心计算必须在 NPU 完成，不允许 CPU fallback。

# 需求分析（required）

## 需求描述

在不改变公开 C API 原型的前提下，完成以下能力：

1. Atlas A2/A3 Ascend C Kernel 与 Host 调用链；
2. values 支持 INT8、FP16、BF16、FP32、Complex64；
3. indices 支持 I32，index base 支持 0/1；
4. 支持乱序、重复索引、尾块、`nnz=0/1` 和动态长度；
5. 无重复索引时 bit-wise exact，重复索引为非确定性 last-write-wins；
6. 公开 PyTorch 入口为 `Tensor.index_copy_(0, index, source)`，通过 `aten::index_copy_` 的 PrivateUse1 注册调用 `aclsparseScatter`；
7. 沿用调用方 stream，接口异步返回，无 workspace、无 Host 同步；
8. 提供 C++、ATen/Python、性能、内存及 Profiler 可复现测试。

## 需求拆解

| 模块 | 工作内容 | 验收关注点 |
| --- | --- | --- |
| Host | 描述符、签名、dtype、index type/base、shape、指针、别名、规模及 stream 校验；生成 tiling 并启动 Kernel | 错误码明确；`nnz=0` 无操作；不读取 Device indices |
| Kernel | 多 AIV 分核，indices/values 双缓冲搬入，按索引进行字节级原地写入 | 五种 dtype 逐位复制；base 0/1；尾块；越界不产生 GM 写 |
| ATen | 输入校验、NPU dispatcher 注册、描述符 RAII、I64 index 安全转换、stream 生命周期记录 | 原地返回；同 device；无 CPU fallback；异步异常 |
| 测试 | C++ API、Python e2e、异常、重复索引、性能、内存、Profiler | 结果真实可复现，环境和提交 SHA 可追溯 |
| 文档 | 算子 README、PyTorch 中文接口文档、设计文档、自测报告 | 能力、限制、命令与测试证据一致 |

# 详细设计（required）

## 算子分析

### 数学公式与语义

对每个非零元素执行：

```text
target = indices[i] - idxBase
Y[target] = values[i]
```

该算子不进行数值计算，只搬运元素原始字节。因此 INT8、FP16、BF16、FP32、Complex64 共用同一 Kernel 主流程，元素字节数分别为 1、2、2、4、8。无重复索引时输出必须与 values 的 bit pattern 完全一致；重复索引由不同 AIV 竞争写同一地址，最终结果必须来自对应 source values 之一，但不承诺写入顺序。

### 支持范围

| 项目 | A2/A3 支持范围 |
| --- | --- |
| 硬件 | Atlas 训练/推理 A2、A3，arch22 |
| CANN | 9.1.0 及后续配套版本 |
| PyTorch / torch_npu | PyTorch 2.7+ / torch_npu 26.0.0+ |
| values dtype | ACL_INT8、ACL_FLOAT16、ACL_BF16、ACL_FLOAT、ACL_COMPLEX64 |
| index | ACL_SPARSE_INDEX_32I；base zero/one |
| shape | 一维 SpVec `[nnz]` 写入一维 DnVec `[size]`，`size == vecY.nums` |
| workspace | 0 |

## Host 侧设计

### 参数校验

Host 按以下顺序校验并尽早返回：

1. `handle`、`vecX`、`vecY` 非空，描述符签名有效；
2. index type 为 I32，base 为 zero/one；
3. values dtype 属于五种支持类型，且 X/Y dtype 相同；
4. `vecX.size == vecY.nums`、`nnz <= size`；
5. `size <= INT32_MAX`、`nnz <= UINT32_MAX`；
6. `nnz > 0` 时 indices、values、Y 指针非空，且输入输出无直接别名；
7. handle 已绑定 stream；
8. `nnz == 0` 直接成功返回，不启动 Kernel。

indices 位于 Device，C API 不做 D2H 范围扫描，合法索引是调用方前置条件。Kernel 仍设置最后一道边界保护，非法 target 直接跳过，避免越界 GM 写。

### 分核与 tiling

```text
blockNum = min(nnz, vectorCoreCount)
baseCount = nnz / blockNum
remainder = nnz % blockNum
coreCount = baseCount + (blockIdx < remainder)
coreOffset = blockIdx * baseCount + min(blockIdx, remainder)
```

每核最多以 4096 个元素为一 tile。tiling 下发 `nnz`、`blockNum`、`tileElements`、`valueSize`、`ySize` 和 `idxBase`。4096×最大 8 字节为 32 KiB，indices 为 16 KiB；双缓冲总量适配 A2/A3 UB，并保持输入块长度不超过 `DataCopyPad` 参数范围。

## Kernel 侧设计

### 流水结构

Kernel 仅使用 AIV：

1. `CopyIn`：indices 和 values 分别通过双缓冲队列从 GM 搬入 UB；
2. `ScatterTile`：逐元素读取 I32 index，减去 base，计算目标字节偏移；
3. `CopyOut`：使用 `DataCopyPad` 将 1/2/4/8 字节原始值写入目标 GM；
4. tile 之间预取下一块，末尾以 `PIPE_MTE3` barrier 保证本核写请求提交完成。

values 统一映射为 `int8_t` 字节流，避免为五种 dtype 维护重复模板，也避免浮点/复数转换改变 NaN payload、符号零或 Complex64 位模式。

### 内存安全与并发

- 目标地址使用 64 位字节偏移计算，Host 已限制元素索引范围；
- Kernel 在写入前检查 `0 <= target < ySize`，非法索引不触发 GM 写；
- 不修改 indices/values；未命中的 Y 元素不被访问；
- 重复 index 允许多核写竞争，符合任务规定的非确定性 last-write-wins；
- 不分配 workspace，不执行 Host 同步或 D2H。

## Python / ATen 设计

Python 层使用：

```python
torch.library.impl("aten::index_copy_", "PrivateUse1", index_copy_)
```

公开入口仍是标准 `Tensor.index_copy_`，不 monkey-patch `torch.Tensor`，也不创建同名私有 façade。C++ wrapper 完成：

1. `self/index/source` 为同一 NPU device 的一维连续 Strided Tensor；
2. 仅支持 `dim=0`，index 为 int32/int64，self/source dtype 相同且属于五种支持类型；
3. source 长度与 index 一致，self 不与 index/source 重叠；
4. 空 index 原地返回；
5. 使用 NPU `aminmax` 与 `_assert_async` 在当前 stream 上排入上下界断言；
6. int64 index 在 NPU 上转换为 I32；
7. RAII 创建/销毁 SpVec、DnVec 描述符，调用公共 `RunAclSparse` 复用当前 device/stream handle；
8. 通过 `RecordTensors` 记录 self、source、原 index 和转换后的 index 生命周期；
9. 返回同一 `self` 对象，无 CPU fallback、无主动同步。

## 异常与边界

| 场景 | 行为 |
| --- | --- |
| 空 handle | `ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR` |
| 空/损坏描述符、非法 base、shape/dtype 不一致、空数据指针或别名 | `ACL_SPARSE_STATUS_INVALID_VALUE` |
| 非 I32 C index、未支持 values dtype、超过实现规模 | `ACL_SPARSE_STATUS_NOT_SUPPORTED` |
| Python 非 NPU、异 device、非一维/非连续/非 Strided、dtype/shape 不符 | `TORCH_CHECK` 抛出 RuntimeError |
| Python index 越界 | 当前 NPU stream 上异步断言报错，scatter 写入排在断言之后 |
| `nnz=0` / 空 index | 成功且 Y/self 不变 |

# 可维可测分析

## 精度标准/性能标准

| 验收项 | 标准 |
| --- | --- |
| 精度 | 无重复索引逐元素 bit-wise exact；重复索引结果来自对应 values 之一；未写入 Y 与输入 X 保持不变；覆盖五 dtype、base 0/1、正负零、INF/NAN、Complex64 |
| 性能 | P-01/P-02/P-03 每个有效 case：GPU Event median / NPU 同调用范围总 Kernel 耗时 ≥ 0.25；预热≥10，采样≥30，报告 median/p90 |
| 内存 | workspace 为 0；不得产生不必要的线性 Device/Host 临时内存，I64→I32 转换仅用于 Python I64 index 适配 |
| 异步 | 沿用调用方 stream，算子层不调用 synchronize；Profiler 证明核心计算在 NPU |

## 测试方案

1. C++ API：五 dtype × base 0/1 原始字节矩阵；`nnz=0/1`、尾块、随机乱序、连续、逆序、步长、大规模；固定随机种子；对结果执行 bit-wise 比较。
2. Python e2e：五 dtype × int32/int64 index；原地返回；边界长度；空 index；重复 index；越界、dim、dtype、shape、layout、device、alias 异常。
3. 官方任务包：运行 accuracy、performance、memory 脚本，保留原始 JSON/CSV、终端日志及截图。
4. 性能：P-01 `128256/8192`、P-02 `151936/4096`、P-03 `129280/7168`，五 dtype、base 0/1；记录 910B3/910B4/A3 实际硬件、CANN/驱动/固件、提交 SHA。
5. Profiler：确认调用链只包含 NPU 断言/转换（需要时）及 `scatter_custom`，不存在 CPU 计算 fallback；确认 workspace 为 0。
6. 资源稳定性：循环创建、调用、销毁描述符，检查无泄漏、非法同步和 stream 生命周期错误。

## 兼容性分析

- C API 原型不变；既有 FP32/base0 调用继续工作，新增能力不影响 arch35；
- arch22 tiling 仅在本架构 Host/Kernel 内共同变更，不暴露 ABI；
- 公共 descriptor、handle 和 torch extension 基础设施直接复用，不新增全局状态；
- Python 注册不修改 `torch.Tensor`，减少与 torch_npu 其他模块的导入副作用；
- A5/arch35 后续变更若触及公共 README 或 Host 分发，合入前执行交叉回归。

# 交付件与计划

| 交付件 | 路径/说明 |
| --- | --- |
| 设计文档 | 本文档，提交 cann-ops-competitions PR |
| 算子实现 | `sparse/scatter/arch22/` |
| PyTorch/ATen | `sparse/scatter/torch_extension/`、`torch_extension/cann_ops_sparse/docs/zh/scatter.md` |
| C++ 测试 | `test/scatter/arch22/scatter_test.cpp` |
| Python e2e | `test/scatter/test_torch_extension.py` |
| 用户文档 | `sparse/scatter/README.md` |
| 自测报告 | 按官方模板记录实际 NPU 精度、性能、内存及 Profiler 证据 |

开发优先在官方 910B3 环境完成；再申请/使用 910B4 与 A3 环境做交叉验证。所有未实际运行的测试明确标记为“待验证”，不以静态检查替代 NPU 验收结果。

## AI 辅助披露

本任务使用 OpenAI Codex（当前实际模型由提交时会话信息如实填写）辅助需求梳理、代码生成、代码审查、测试设计、文档撰写与问题排查。全部代码、测试结果和提交内容由参赛者检查；PR 模板中的 AI 参与项将如实选择“是”，且不会声称未实际执行的测试已经通过。
