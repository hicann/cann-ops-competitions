# aclsparseGather 算子设计说明

## 需求背景（required）

### 需求来源

- 通过 CANN 社区 9 月任务完成 `aclsparseGather` 算子在 Ascend 950（DAV_3510，`arch35`，下文简称 A5）上的开发与能力补齐，参考 cuSPARSE `cusparseGather` 接口语义，面向稀疏向量访问场景，从稠密向量按稀疏向量索引读取元素并原地写入稀疏 values。
- 任务要求支持 `float16`、`bfloat16`、`float32`、`complex64` 四种 values dtype，I32 索引和 index base 0/1；核心计算在 NPU 完成，禁止 CPU fallback。
- Python/ATen 适配为必选交付：参考公开 PyTorch 入口 `torch.index_select(input, 0, index)` 及 `aten::index_select` Dispatcher 接口，在 NPU 完成适配；适配 PyTorch 2.7+、torch_npu 26.0.0+。
- 性能要求：任务书 P-01/P-02/P-03 场景下，所有声明 dtype 达到 0.3 倍性能标杆以上。

### 背景介绍

#### 算子功能

`aclsparseGather` 是稀疏向量聚集（Gather）算子。从稠密源向量 `Y` 中，按稀疏向量 `X` 的索引数组读取元素，并原地写入 `X.values`：

$$
X.values[i] = Y[X.indices[i] - idxBase],\quad i\in[0,nnz)
$$

- 输入：稠密向量 `vecY`（shape `[size]`，只读）、稀疏向量 `vecX` 的 `indices`（shape `[nnz]`，只读）
- 输出：`vecX.values`（shape `[nnz]`，原地覆写）
- 只更新 `vecX.values`，不修改 `vecY` 和 `vecX.indices`；`indices` 允许乱序与重复
- 无 workspace、无 preprocess，单步异步 Kernel 完成

#### aclsparseGather 算子实现路径

对标参考实现：

- Python 层：`torch.index_select(input, 0, index)`
- ATen Schema：`aten::index_select(Tensor self, int dim, Tensor index) -> Tensor`
- cuSPARSE 参考：CUDA Toolkit 13.3 Update 1 所含 cuSPARSE 13.3 Update 1 的 `cusparseGather` 接口

当前 AscendC 工程路径：

```text
ops-sparse/sparse/gather/arch35/
```

#### aclsparseGather 算子现状分析

设计基线为 `ops-sparse` `master`（`e0015bb`）。该基线已存在 `sparse/gather/arch35/` 路径，包含公开接口声明、Host 参数校验与 launch、SIMT Kernel 及 CSV 驱动的 C++ 测试。本任务采用"复用并补齐"的增量方案：

| 模块 | 基线现状（`e0015bb`） | 本任务差距 |
| --- | --- | --- |
| 公开接口 | `include/cann_ops_sparse.h` 已声明 `aclsparseGather`，原型保持源代码兼容 | 补充 complex64 支持说明、边界、重叠、别名与错误行为文档 |
| Host | 基础校验（handle/描述符非空、valueType 一致、`vecY.nums >= vecX.size`）、`nnz=0` 快速返回、AIV 核数与 numBlocks 计算 | 补描述符签名、数据指针、index type/base 合法性、指针区间重叠、规模溢出校验；新增 complex64 分派 |
| Kernel | arch35 SIMT VF，grid-stride loop，支持 FP32/FP16/BF16/FP64 × I32/I64 × base 0/1 模板实例 | 新增 `complex64` 8 字节位拷贝实例（I32、base 0/1），保持纯拷贝无数值运算 |
| dtype | values 支持 FP32/FP16/BF16/FP64 | 新增任务必选 `complex64`；既有 FP64 能力保留不回退 |
| index | I32/I64、base 0/1 | 任务验收矩阵为 I32、base 0/1；既有 I64 能力保留但不纳入本任务验收声明 |
| Python/ATen | 仓内尚无 Gather 的 `aten::index_select` NPU 适配 | 新增 NPU Dispatcher 注册、Tensor 校验、描述符桥接、输出构造与端到端 UT |
| 测试 | C++ CSV 用例 98 条，dtype 以 FP16/FP32/FP64 为主 | 补 BF16/complex64、异常、重叠、输入只读性、bit-wise 一致性、Python/ATen 与无 CPU fallback 证据 |

Python/ATen 适配与 Kernel 分工：

| 阶段 | 执行位置 | 职责 |
| --- | --- | --- |
| Tensor 元数据校验 | ATen 适配层 | 校验 dtype、shape、layout、stride、dim、index、device |
| 输出 Tensor 分配 | ATen 适配层 | 按 PyTorch functional 语义分配 `[nnz]` 连续输出 |
| 描述符创建与管理 | C++ Host 层 | 创建/销毁 DnVec/SpVec 描述符（RAII 管理） |
| 参数校验 | C++ Host 层 | 校验描述符签名、指针、重叠、规模并生成 TilingData |
| Gather 计算 | Ascend C Kernel（NPU） | 按 indices 间接寻址，从 GM 读 `Y[pos]` 写 `X.values[i]` |

#### 总体流程图

![image.png](https://raw.gitcode.com/user-images/assets/10331120/c67d8499-c00b-45ee-8841-22ebc804df7e/image.png 'image.png')

## 需求分析

### 外部组件依赖

- PyTorch 2.7 及以上版本
- torch_npu 26.0.0 及之后版本
- CANN 9.1.0 及后续配套版本（自测报告记录实际 CANN、驱动、固件、硬件型号和代码提交版本）

### 内部适配模块

- 适配 `aten::index_select` NPU 注册（torch_npu 自定义 dispatch），仅覆盖 `dim=0` 的一维索引读取子集。
- 复用现有 `aclsparseHandle_t`、`aclsparseConstDnVecDescr_t`、`aclsparseSpVecDescr_t` 描述符和资源管理接口。
- 复用 `sparse/gather/arch35/` 现有 Host 与 SIMT Kernel，新增 `complex64` 全链路支持（描述符、Host 分派、Kernel 模板实例、输出组装、测试）。
- 覆盖不同向量长度、nnz、稀疏度、乱序/重复索引、尾块及 `nnz=0/1` 等泛化场景。

### 需求模块设计

#### 算子原型

**Python 公开入口：**

```python
torch.index_select(input, 0, index) -> Tensor
```

**ATen Schema：**

```text
aten::index_select(Tensor self, int dim, Tensor index) -> Tensor
```

本任务仅注册 functional overload；`index_select.out`、Dimname overload 不在范围内。PyTorch 索引固定 0-based，adapter 创建 SpVec 时使用 `ACL_SPARSE_INDEX_BASE_ZERO`；base 1 仅通过 C++ 接口和测试覆盖。

**C++ 接口：**

| 接口 | 功能 |
| --- | --- |
| `aclsparseGather` | 从稠密向量 `vecY` 按稀疏向量 `vecX` 的 indices 读取元素，原地写入 `vecX.values` |

```c
aclsparseStatus_t aclsparseGather(
    aclsparseHandle_t handle,
    aclsparseConstDnVecDescr_t vecY,
    aclsparseSpVecDescr_t vecX);
```

接口原型以 `include/cann_ops_sparse.h` 为准，不改变参数顺序、类型或符号名，保持源代码兼容。

**参数说明：**

| 名称 | 类别 | dtype | format | shape | 说明 |
| --- | --- | --- | --- | --- | --- |
| handle | 输入 | 不涉及 | Handle | 标量 | aclsparse 句柄及调用方 stream，空句柄返回参数错误 |
| vecY | 输入 | float16/bfloat16/float32/complex64 | 连续一维向量 | `[size]` | 稠密源向量 Y，`size>=0`；当 `nnz>0` 时 `size>0` |
| vecX | 输入输出 | values 同 vecY；indices 为 I32 | 一维 SpVec，base 0/1 | indices/values `[nnz]` | 读 indices 与 idxBase，原地写 values；`nnz>=0` |

描述符和数据指针在接口异步执行完成前由调用方保持有效。`vecY.values` 与 `vecX.indices` 只读；未声明的 `vecY`/`vecX.values` 重叠返回参数错误。

相关约束：

- 仅支持 `dim=0` 的一维 gather；input 为一维连续稠密向量，index 为一维 I32 向量。
- `vecY.nums >= vecX.size`；换算 base 后 indices 必须位于 `[0, size)`，Device 索引越界按异步错误协议或调用方前置条件处理。
- `nnz=0` 成功返回且不启动 Kernel；零长度 `vecY` 仅允许 `nnz=0`。
- 相同输入和索引的输出必须 bit-wise 一致；禁止 CPU fallback。
- 不新增无必要 workspace 或 Host 同步，沿用调用方 stream 异步执行。

## 需求详细设计

### 使能方式

本任务的接口分层与使能路径如下：

```
Python 层：torch.index_select(input, 0, index)     ← 本任务新增适配
    └── ATen 层：aten::index_select                ← 本任务新增 NPU 注册
        └── aclsparse C++ 层：aclsparseGather      ← 复用现有接口，补齐 complex64
            └── Ascend C Kernel（arch35 SIMT）     ← 复用现有 Kernel，新增 complex64 模板实例
```

**与仓库现有 Gather 实现（基线 `e0015bb`）的关系：**

| 层级 | 仓库现有实现 | 本任务（A5） |
| --- | --- | --- |
| Python/torch 接口 | 不涉及 | **新增**：`torch.index_select` NPU 适配 |
| ATen NPU 注册 | 不涉及 | **新增**：`aten::index_select` dispatch |
| aclsparse C++ 接口 | 已交付，原型不变 | **复用**并补齐 complex64、校验与文档 |
| Ascend C Kernel | arch35 SIMT VF，FP32/FP16/BF16/FP64 | **复用**并新增 complex64 模板实例 |
| C++ 测试 | CSV 驱动 98 条 | **补齐** BF16/complex64/异常/一致性用例并新增 Python 端到端 UT |

适配 PyTorch 2.7+ / torch_npu 26.0.0+，核心计算必须在 NPU 上完成，不允许 CPU fallback。

### 需求总体设计

#### Python/ATen 适配层设计

代码位置：`ops-sparse/` Python 及 ATen 适配目录（随 PR 交付，落位遵循仓库 adapter 框架规范）

职责：

1. 注册 `aten::index_select` 的 NPU dispatch，仅处理任务约束内组合；未支持组合返回带参数信息的明确错误，不 redispatch 到 CPU。
2. 校验 Tensor 元数据：

| 项目 | 支持条件 | 不满足时 |
| --- | --- | --- |
| input rank/dim | 一维，规范化后 `dim=0` | `TORCH_CHECK` 明确提示仅支持 1-D |
| index rank/dtype | 一维、I32 | 返回参数错误，不做隐式转换 |
| input dtype | FP16/BF16/FP32/complex64 | 返回不支持，不做隐式 cast |
| layout/stride | `torch.strided` 且 input/index 连续 | 返回不支持，不隐式 materialize |
| device | input/index 均为 NPU 且同 device | 返回错误 |

3. 在 input 的 device guard 下获取当前 NPU stream 并设置到 aclsparse handle，保持异步 stream 语义与 PyTorch 一致。
4. 按 PyTorch functional 语义分配 `[nnz]` 连续输出 Tensor（dtype/device 与 input 相同，不与 input/index 共享 storage），创建 `aclsparseCreateConstDnVec`（vecY）与 `aclsparseCreateSpVec`（vecX：indices、values=output、I32、base 0）描述符，调用 `aclsparseGather` 后按 RAII 销毁描述符并异步返回输出。
5. `nnz=0`（index 为空）时返回独立空 Tensor，不调用 Kernel 路径之外的分支；`index_select.out` 不在本任务范围。

异常、alias、异步语义对齐 PyTorch 2.7+：输出为新建 Tensor，input/index 不被原地修改；Kernel 异常通过 stream 异步错误协议上报。

#### AscendC host 侧设计

代码位置：`ops-sparse/sparse/gather/arch35/gather_host.cpp`

Host 文件职责拆分：

| 模块 | 主要职责 | 关键点 |
| --- | --- | --- |
| 参数校验 | 校验 handle/描述符/指针/重叠/规模 | 校验顺序从廉价到昂贵，全程不访问 Device 数据 |
| 快速路径 | `nnz=0` 直接返回 SUCCESS | 不启动 Kernel |
| Tiling | 计算 `numBlocks`，组装 `GatherTilingData` | 线程数固定 256，block 数随 nnz 动态 |
| Kernel launch | `gather_kernel_do` 按 (valType, idxType, idxBase) 分派模板实例 | 沿用调用方 stream，无同步 |

Host 核心执行链路：

1. 校验 handle、vecY/vecX 非空；校验描述符签名（复用 `aclsparse_descr_internal.h` 的 signature 机制），拒绝悬垂/失效描述符。
2. 校验 `vecX.valueType == vecY.valueType` 且在支持矩阵内（任务矩阵新增 `ACL_COMPLEX64`）；校验 idxType/idxBase 合法（I32/I64、base 0/1，保持既有兼容）。
3. 校验规模：`vecY.nums >= vecX.size`、`nnz >= 0`；`nnz>0` 时校验 `vecY.values`、`vecX.indices`、`vecX.values` 非空指针；字节数计算使用 checked 乘加避免 Host 侧整数回绕。
4. 校验 `vecY.values` 与 `vecX.values`、`vecX.indices` 与 `vecX.values` 的字节区间不重叠，未声明重叠返回参数错误。
5. `nnz=0` 快速返回；否则计算 `numBlocks = min(ceilDiv(nnz, kGatherMaxThreadsPerBlock), GetAivCoreCount())`，`numBlocks=0` 视为内部错误。
6. 组装 `GatherTilingData{nnz, numBlocks, valType, idxType, idxBase}` 并异步 launch。

Host 校验与 launch 流程图：

![image.png](https://raw.gitcode.com/user-images/assets/10331120/8f19af85-550c-4b80-aa6d-294fa2478eca/image.png 'image.png')

Tiling 策略要点：

- Gather 为"随机读、连续写"的 memory-bound 算子，每个输出元素独立、无归约，采用 SIMT 线程级并行 + grid-stride loop 覆盖任意 `nnz`（含尾块与非 256 倍数规模），无需按 shape 维度切分。
- `kGatherMaxThreadsPerBlock = 256` 与 Kernel `__launch_bounds__` 一致；AIV 核数通过 `PlatformAscendCManager::GetCoreNumAiv()` 获取，不写死芯片核数。

模板实例规划（`gather_kernel_do` 分派矩阵，任务新增部分加粗）：

| valType | IdxT | idxBase | Kernel 模板实例 | 状态 |
| --- | --- | --- | --- | --- |
| ACL_FLOAT16 | int32_t | 0/1 | `gather_kernel<__fp16, int32_t, base>` | 已有 |
| ACL_BF16 | int32_t | 0/1 | `gather_kernel<__fp16, int32_t, base>`（2 字节位拷贝，不做数值计算） | 已有 |
| ACL_FLOAT | int32_t | 0/1 | `gather_kernel<float, int32_t, base>` | 已有 |
| **ACL_COMPLEX64** | **int32_t** | **0/1** | **`gather_kernel<uint64_t, int32_t, base>`（8 字节整体位拷贝）** | **本任务新增** |
| ACL_DOUBLE / I64 组合 | int64_t 等 | 0/1 | 既有模板实例 | 保留兼容 |

错误码映射：

| 状态 | 场景 |
| --- | --- |
| `ACL_SPARSE_STATUS_SUCCESS` | 参数合法；`nnz=0` 快速返回或 Kernel 成功下发 |
| `ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR` | `handle == nullptr` |
| `ACL_SPARSE_STATUS_INVALID_VALUE` | 描述符/签名/指针/shape/base/重叠/规模不合法 |
| `ACL_SPARSE_STATUS_NOT_SUPPORTED` | dtype 或 index type 不在支持矩阵内 |
| `ACL_SPARSE_STATUS_INTERNAL_ERROR` | AIV 核数查询失败或内部状态错误 |

#### AscendC kernel 侧设计

代码位置：`ops-sparse/sparse/gather/arch35/gather_kernel.cpp`

核心计算逻辑：

```cpp
template <typename ValT, typename IdxT, aclsparseIndexBase_t idxBase>
__simt_vf__ __aicore__ void GatherSimtCompute(
    __gm__ const IdxT* indices, __gm__ const ValT* yValues,
    __gm__ ValT* xValues, int64_t nnz)
{
    int64_t globalTid = threadIdx.x + blockIdx.x * blockDim.x;
    int64_t gridStride = blockDim.x * gridDim.x;
    for (int64_t i = globalTid; i < nnz; i += gridStride) {
        int64_t pos = indices[i] - idxBase;
        xValues[i] = yValues[pos];
    }
}
```

Kernel 设计要点：

- 遵循 SIMT VF 两层结构：Layer 1 `__simt_vf__` 计算函数（线程级并行），Layer 2 `__global__` kernel 经 `asc_vf_call` 启动，`KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY)` 绑定 AIV。
- 技术路线选择：Gather 访存模式由 indices 分布决定（乱序/重复均合法），将 tile 预搬入 UB 不会提高复用率，反而增加搬运与容量开销；因此不使用 Vector 大 tile 与 `DataCopy`/UB staging，直接以 `__gm__` 指针完成标量 load/store。
- base 0/1 与 dtype/idxType 作为编译期模板参数，消除循环内分支（DCE 优化）。
- `complex64` 以 `uint64_t` 为 payload 整体 8 字节位拷贝，不解释实部/虚部、不做复数算术，与 FP64 共享 8 字节路径的索引逻辑；正负零、INF、NAN 及 NAN payload 位模式均不改变。
- BF16 沿用基线做法：仅按 2 字节位拷贝（`__fp16` payload），不做浮点计算。
- 确定性与尾块：`i < nnz` 为唯一循环条件，任意 `nnz`（含 `nnz=1`、非 256 倍数尾块）不越界；每个输出地址仅被唯一线程写一次，无原子、无归约、无跨核通信，重复索引只产生重复读 `Y[pos]`，不产生写冲突，同一输入 bit-wise 可复现。
- 内存占用：无 TPipe/TQue/LocalTensor，UB 占用 0 字节，无 workspace，输出原地复用 `vecX.values`。

Kernel 执行流程图：
![image.png](https://raw.gitcode.com/user-images/assets/10331120/a0e962f6-ed49-4eea-bd1d-524b408ddc12/image.png 'image.png')

### 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Ascend 950（DAV_3510，arch35） | √ |

### 算子约束限制

- values dtype：任务验收矩阵为 `float16`、`bfloat16`、`float32`、`complex64`；既有 `float64` C++ 能力保持兼容但不作为本任务交付声明。
- index dtype：任务验收矩阵为 I32；既有 I64 能力保持兼容，Python adapter 仅接受 I32。
- index base：C++ 支持 0/1；PyTorch 公开入口固定为 0。
- `vecY` 为连续一维向量；`vecX.indices`/`vecX.values` 为长度 `nnz` 的连续一维数组；高维输入、其他 `dim`、非连续 Tensor 返回明确不支持。
- 索引可以乱序或重复，但换算 base 后必须位于 `[0, size)`；Kernel 不做逐索引越界检查，越界按异步错误协议或调用方前置条件处理。
- 不支持 `vecY.values` 与 `vecX.values`、`vecX.indices` 与 `vecX.values` 重叠。
- 不使用 workspace/preprocess，不做 CPU fallback，不插入无必要 Host 同步。
- 最大 `size`、`nnz` 及索引溢出边界在接口文档（README 与公开头文件注释）中明确：I32/base 0 下 `size <= 2^31`；`nnz` 满足 indices/values 字节数可由 `size_t` 表示。

## 特性交叉分析、可维可测分析

### 精度标准 / 性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | 以 CPU Golden 为基准，四种 dtype 输出逐元素 bit-wise exact match；complex64 实部和虚部均 exact match；校验 `vecY`、indices 和非目标数据未被修改 | 《生态算子开源精度标准》及任务书 §3.2 |
| 性能标准 | 倍率 = GPU 设备 Event 调用耗时（median_us） / NPU 同调用范围全部 Kernel 总耗时；P-01/P-02/P-03 所有声明 dtype 达到 0.3 倍以上 | 任务书 §3.3 |
| 内存标准 | 无 workspace、无 preprocess；输出原地复用 `vecX.values`；方案固有 workspace 为 0（不超过目标硬件 L2 Cache 容量） | 任务书 §3.4 |

精度说明：Gather 为纯拷贝算子，Kernel 不做浮点计算、舍入或类型转换，因此不使用 rtol/atol 放宽，直接要求 bit-wise exact。Golden 生成规则：`float16`/`bfloat16` 使用 float32 Golden，`float32` 使用 float64 Golden，`complex64` 使用 complex128 Golden，再转换为目标 dtype 后按 index 选择。

性能参考基准（PyTorch GPU 设备 Event 调用耗时，`test_cases/baseline_results/gpu_full_results.tsv`）：

| 编号 | 场景 | size | nnz | dtype/base 组合 | GPU median（μs） | NPU 目标 |
| --- | --- | ---: | ---: | --- | --- | --- |
| P-01 | Llama 3.1 70B 词表索引 | 128256 | 8192 | 4 dtype × base 0/1（8 场景） | 27.488–32.736 | 每组合倍率 ≥ 0.3 |
| P-02 | Qwen3-235B-A22B 词表索引 | 151936 | 4096 | 4 dtype × base 0/1（8 场景） | 31.056–31.264 | 每组合倍率 ≥ 0.3 |
| P-03 | DeepSeek-V3 词表索引 | 129280 | 7168 | 4 dtype × base 0/1（8 场景） | 30.832–31.072 | 每组合倍率 ≥ 0.3 |

采集口径：每个 case 预热至少 10 次、采样 30 次，报告中位数与 p90；不计首次编译、数据生成、Host-to-Device 搬运和无关初始化；固定随机种子并复用描述符；同时给出 aclsparse Kernel 与 Python/ATen 端到端耗时；NPU 实测数据和倍率在自测报告中记录，本文档不预填。

内存测试：使用 `test_cases/aclsparseGather_testCase/performance_cases.json` 的 P-01/P-02/P-03 及泛化用例，等价 GPU 接口执行 `collect_sparse_ops_gpu_memory.py` / `collect_sparse_ops_npu_memory.py`，以 `compare_sparse_ops_memory.py` 按相同 case id 对比 `input_baseline_*_bytes`、`peak_*_bytes`、`extra_peak_*_bytes`。

### 可测性分析

- 测试目录：`test/gather/`（公共 golden/param）与 `test/gather/arch35/`（NPU wrapper、CSV 用例）；Python/ATen 端到端 UT 随适配层交付。
- 精度测试：CPU Golden 对比，CSV 驱动 GTest，覆盖任务包 `accuracy_cases.json` 200 条泛化用例。
- 性能测试：预热 10 次、采样 30 次，报告中位数与 p90；覆盖 `performance_cases.json` 的 P-01/P-02/P-03 共 24 个 dtype × base 组合及泛化用例。
- 结构验证：输出长度恒为 `nnz`，第 `i` 个输出仅由第 `i` 个 index 决定；输入只读性以调用前后逐字节比对校验。

用例生成规则（任务书 §3.5）：DnVec values 中 70% 用 `[-1,1]` 均匀分布、20% 用正态分布 `μ=0、σ=1`、10% 覆盖零值/边界值/规格允许特殊值（含 INF/NAN）；SpVec indices 在合法范围均匀生成并专项覆盖重复、首尾、乱序、越界异常；属性覆盖 values dtype 四种、I32、base 0/1、`nnz=0/1` 及动态长度组合。

建议覆盖场景：

| 编号 | 场景 | 覆盖点 |
| --- | --- | --- |
| TC-01 | 基础功能 - 规模泛化 | 不同 `size`/`nnz`/稀疏度，四种 dtype，base 0/1，动态长度 |
| TC-02 | 基础功能 - 索引形态 | 顺序、乱序、重复、首尾元素、`nnz=0/1`、非 256 倍数尾块 |
| TC-03 | 数值专项 | 正负混合、正负零、小值、离群值、INF/NAN；complex64 实/虚部及 NAN payload 位模式 |
| TC-04 | 一致性 | 相同输入重复执行 bit-wise 一致；`vecY` 与 indices 调用前后逐字节不变 |
| TC-05 | 接口与异常 | 空 handle/描述符/指针、dtype 或 device 不一致、非法索引类型/base、重叠区间、规模溢出、输出边界 |
| TC-06 | Python/ATen 端到端 | `torch.index_select(input, 0, index)` 四 dtype、alias/storage、异常语义与 CPU 对齐 |
| TC-07 | NPU Dispatch 证据 | Dispatch 日志与 Profiler 证明无 CPU fallback，仅出现 NPU Gather 路径 |
| TC-08 | 资源鲁棒性 | 连续创建、执行、销毁 handle/描述符无泄漏或非法同步；workspace 恒为 0 |

### 兼容性分析

- 公开 `aclsparseGather` 原型不变，新增 complex64 属能力扩展，不新增同名或同功能接口。
- 保留基线 FP64/I64 模板实例与行为，避免现有调用方回归；任务验收矩阵（I32、四 dtype、base 0/1）在 README 中单独声明。
- arch35 代码继续位于 `sparse/gather/arch35/`，与 A2/A3 公共描述符和 Host 逻辑可共存；后续 PR 基于已合入版本处理公共 Host 冲突并完成交叉回归。
- Python 只注册任务约束内的 functional `aten::index_select`，不覆盖 `out`/Dimname overload；接口语义以 PyTorch 2.7+ 为准，C++ 语义以 cuSPARSE 13.3 Update 1 为准。
- 合入路径 `ops-sparse` `master`，接口声明与限制说明同步至 `include/cann_ops_sparse.h` 与 `sparse/gather/README.md`，C++ UT/ST 合入 `test/gather/arch35/` 及 `test/gather/` 公共目录，任务验收脚本与用例随任务包 `test_cases/` 交付。
