# aclsparseGather（A2/A3）算子设计文档

# 需求背景（required）

## 需求来源

本设计面向 CANN 社区任务“aclsparseGather 算子开发（A2/A3）”。目标是在 `ops-sparse` 中补齐 Atlas A2/A3 的 Gather 能力，并提供 PyTorch/ATen 调用入口。

算子语义参考 cuSPARSE `cusparseGather`：

```text
X.values[i] = Y[X.indices[i] - indexBase],  i ∈ [0, nnz)
```

`vecY` 是稠密源向量，`vecX` 保存稀疏索引和输出 values。计算只更新 `vecX.values`，索引的顺序和重复关系原样保留。

## 背景介绍

Gather没有数值计算，主要开销来自按索引读取 `Y`。输出是连续的，但源地址可能连续、等步长、集中在一个小窗口，也可能完全离散，因此数据搬运方式决定了性能。

仓库已有 `sparse/gather/arch35/`，其 Kernel 使用 DAV_3510 的 SIMT 能力直接访问 GM。A2/A3 对应 DAV_2201，无法直接复用该实现。仓内 `sparse/spvv/arch22/` 已有“先把数据搬入 UB，再用 Gather 重排”的实现基础，本任务据此增加 `arch22` 路径。

# 需求分析（required）

## 需求描述

公开 C++ 接口保持不变：

```cpp
aclsparseStatus_t aclsparseGather(
    aclsparseHandle_t handle,
    aclsparseConstDnVecDescr_t vecY,
    aclsparseSpVecDescr_t vecX);
```

同时提供两类 Tensor 入口：

- `torch.index_select(input, 0, index)` / `aten::index_select`：公开 PyTorch 接口，索引按 base 0 解释。
- `torch.ops.ops_sparse_test.gather_npu(values, indices, base)`：任务测试入口，用于覆盖 base 0 和 base 1。

两个入口共用同一套 Tensor bridge，最终都调用 `aclsparseGather`。

## 接口范围

| 项目 | 支持范围 |
| --- | --- |
| values dtype | FP16、BF16、FP32、complex64 |
| index dtype | I32 |
| index base | C++ 接口和任务测试入口支持 0/1；PyTorch 公开入口为 0 |
| shape | 一维连续向量，`sizeY` 和 `nnz` 动态变化，`0 <= nnz <= sizeY` |
| 索引形态 | 连续、等步长、乱序、重复、首尾索引 |
| 空输入 | `nnz=0` 成功返回；`sizeY=0` 时只接受 `nnz=0` |
| 软件基线 | CANN 9.1.0 及后续配套版本、PyTorch 2.7 及以上、torch_npu 26.0.0 及以后版本 |
| 执行方式 | 使用调用方当前 NPU stream 异步执行 |
| 输出语义 | C++ 原地更新 `vecX.values`；PyTorch 返回新的连续 Tensor |

Gather 按位复制 payload。FP16/BF16 不发生升精度计算，complex64 的实部和虚部作为一条 8 字节记录处理，因此结果应与输入位模式完全一致。

## 需求拆解

1. 在 `sparse/gather/arch22/` 实现 Host、Tiling 和 Ascend C Kernel。
2. 根据索引局部性选择数据搬运路径，在保持输出顺序的前提下降低 DMA 次数。
3. 补齐四种 values dtype、I32、base 0/1、动态规模和边界场景。
4. 接入 Python/ATen，并完成 C++、Tensor 入口、精度、性能和内存测试。

# 详细设计（required）

## 算子分析

### 数据访问特点

每个输出元素需要读取 4 字节 index、读取一个 `Y` 元素并写回一个输出元素，算法复杂度为 `O(nnz)`。输出地址连续且互不重叠，适合按输出区间分核；源地址由 index 决定，需要在核内根据局部性选择搬运方式。

任务包的三个核心性能场景如下：

| 场景 | `sizeY` | `nnz` | 索引生成 |
| --- | ---: | ---: | --- |
| P-01 | 128256 | 8192 | `(arange(nnz) * 17 % sizeY) + base` |
| P-02 | 151936 | 4096 | 同上 |
| P-03 | 129280 | 7168 | 同上 |

这些索引以步长 17 为主，P-01 包含一次回绕。等步长搬运是核心优化路径，任意乱序路径用于保证泛化能力。

### 平台能力

| 平台 | NpuArch | 实现 |
| --- | --- | --- |
| Atlas A2（910B3/910B4） | DAV_2201 | 本任务新增 `arch22` SIMD/UB Kernel |
| Atlas A3 | DAV_2201 | 复用 `arch22`，按实际型号回归 |
| Ascend 950/A5 | DAV_3510 | 保留已有 `arch35` SIMT Kernel |

Host 通过 `PlatformAscendC` 获取 NpuArch、AIV 核数和 UB 容量。910B4/CANN 9.1.0 的设计校准值为 40 个 AIV、196352 字节 UB；这些数值只用于计算初始 Tiling 候选，正式实现以运行平台返回值为准。

DAV_2201 的旧式 Sort 不支持 I32，UB Scatter 在该架构也不可用。因此 Kernel 按原始 index 顺序组织 staging，使用 `DataCopyPad + Gather` 得到连续输出，不引入排序和逆置换。

## 算子实现

### 总体流程

```text
Python/ATen 或 C++ 调用
        │
        ▼
参数检查与描述符桥接
        │
        ▼
Host 计算分核和 UB Tiling
        │
        ▼
arch22 Kernel
  indices CopyIn
  → 模式识别
  → Y 数据搬运
  → UB Gather
  → 连续 CopyOut
```

### Host 侧设计

Host 依次完成以下工作：

1. 检查 handle、描述符、数据指针、dtype、index type 和 base。
2. 校验 `sizeY`、`nnz` 及相关字节数计算，`nnz=0` 时直接返回。
3. 检查输入输出区间没有非法重叠；数据指针由调用方保证可被 handle 所属 stream 访问。
4. 从平台信息取得 AIV 核数和 UB 容量，计算 `blockDim`、`tileElems` 和 `windowElems`。
5. 按 values dtype 选择 Kernel 模板，base 作为 TilingData 字段下发，并提交到 handle 保存的 stream。

`sizeY`、`nnz`、地址偏移和乘法使用 64 位 checked arithmetic。Ascend C API 的字段范围较小时，由 Tiling 将任务拆成多个 tile。

索引内容保留在 Device 侧。Kernel 在形成 `Y` 地址前完成 base 换算和上下界判断；PyTorch 入口通过 torch_npu 的 Device 异常机制对齐 `index_select` 的越界行为。C++ 接口仍按稀疏库惯例把合法 index 作为调用前提。

### Tiling 设计

TilingData 保存执行所需的最小信息：

| 字段 | 含义 |
| --- | --- |
| `sizeY` / `nnz` | 源向量长度和输出元素数 |
| `elemBytes` | values 元素字节数 |
| `indexBase` | 0 或 1 |
| `blockDim` | 实际使用的 AIV 数 |
| `tileElems` | 单核一次处理的输出元素数 |
| `tileAlignedElems` | `tileElems` 按 32 个元素向上对齐后的 buffer 尺寸 |
| `windowElems` | 小窗口路径允许搬入的最大元素数 |
| `sourceBufferBytes` | Y window 或 staging 共用 buffer 的物理字节数 |
| `valueType` | Host 启动时的 dtype 分派依据 |

输出区间按核均分。令

```text
q = nnz / blockDim
r = nnz % blockDim
```

第 `c` 个核处理：

```text
count = q + (c < r)
begin = c * q + min(c, r)
end   = begin + count
```

这种切分使各核负载最多相差一个元素，输出区间保持连续。大规模用例优先尝试全部有效 AIV；小 `nnz` 根据每核最小工作量裁减核数，最终参数由 P-01～P-03 的 Profiling 结果确定。

### Kernel 数据路径

每个 tile 先将连续的 I32 indices 搬入 UB，完成 base 换算并识别索引形态，然后按以下顺序选择路径。

#### 1. 连续路径

当 index 严格连续时，一次把对应的 `Y` 区间搬入 source UB。offset 从 0 开始按元素字节数递增，随后复用公共 Gather 和 CopyOut 流程。这使四条路径共享同一套输出同步与尾块处理。

#### 2. 等步长路径

当相邻 index 具有相同正步长 `d` 时，使用 `DataCopyPad` Ext 多 block 搬运：

```text
blockCount = tileCount
blockLen   = elemBytes
srcStride  = (d - 1) * elemBytes
dstStride  = 0
```

每个元素落到一个 32 字节对齐的 UB slot，随后由 UB `Gather` 压紧为连续输出。`blockCount` 最大为 4095，因此该路径的 `tileElems <= 4095`。

#### 3. 小窗口路径

令当前 tile 的索引跨度为：

```text
span = maxIndex - minIndex + 1
```

当 `span <= windowElems` 时，一次搬入 `Y[minIndex : minIndex + span]`。Gather offset 为 `(index - minIndex) * elemBytes`，乱序和重复 index 都可以直接保持。

#### 4. 通用路径

当索引跨度较大且没有统一步长时，Kernel 按原顺序扫描 indices，把相邻、同正步长的片段合并为 run。每个输出位置仍对应一个 32 字节 staging slot：

- 可表达的 run 用一次多 block `DataCopyPad` 搬入。
- 单个元素、重复 index、降序片段使用单 block 搬入。
- staging 完成后按 slot 顺序执行一次 UB `Gather`。

该路径没有额外的排序过程，输出顺序天然与 indices 一致。完全随机索引最坏会退化为每个元素一次短 DMA，后续根据 Profiler 决定是否增加更细的窗口切分。

### dtype 处理

| values dtype | Kernel payload | Gather 方式 |
| --- | --- | --- |
| FP16 | `uint16_t` | 每个输出一个 u16 lane |
| BF16 | `uint16_t` | 每个输出一个 u16 lane |
| FP32 | `uint32_t` | 每个输出一个 u32 lane |
| complex64 | 两个相邻 `uint32_t` | offset 为 `recordOffset` 和 `recordOffset + 4` |

complex64 不做复数运算。Gather 后的 lane 顺序为 `real0, imag0, real1, imag1, ...`，可直接作为连续的 8 字节记录写回。

### UB 与 Buffer 规划

设：

```text
U = PlatformAscendC 返回的单核 UB 字节数
R = 对齐、bank padding 和实现调整预留，初值 4096
T = tileElems
A = alignUp(T, 32)
S = elemBytes
L = 1（S=2/4），2（S=8）
O = 4 * L * A          // Gather offset
P = 32 * A             // 32B staging slots
```

物理 buffer 分配为：

```text
B_fixed  = 4A + O + SA
B_source = alignDown(U - R - B_fixed, 32)
B_total  = B_fixed + B_source
```

`4A` 是 indices buffer，`SA` 是输出 buffer。`B_source` 在 Y window 和 32B staging slots 之间复用，并保证至少容纳 `P=32A`。Host 搜索满足 `B_total + R <= U`且 `T <= 4095` 的候选。

以 910B4 的 `U=196352`、`R=4096` 计算，首轮参数为：

| dtype | `T` | `B_fixed` | `B_source` | 预留 | `windowElems` 上界 |
| --- | ---: | ---: | ---: | ---: | ---: |
| FP16/BF16 | 4095 | 40960B | 151296B | 4096B | 75648 |
| FP32 | 4095 | 49152B | 143104B | 4096B | 35776 |
| complex64 | 3584 | 71680B | 120576B | 4096B | 15072 |

这些数值用于启动实现和调优。最终 tile、window、预留空间以及是否采用双缓冲，以正式 Kernel 的 UB 分配和 Profiling 数据为准。

### 流水与性能优化

indices CopyIn、Y CopyIn、Gather 和 CopyOut 通过 `TQue` 或事件建立 MTE2、Scalar、Vector、MTE3 之间的依赖。初始实现使用单 buffer，先确保四条路径和尾块正确；Profiling 后再根据 MTE2/Vector/MTE3 时间占比决定是否缩小 tile 并引入双缓冲。

性能优化按以下顺序进行：

1. 优先保证核心用例命中等步长路径，P-01 的回绕 tile 拆成两个 run。
2. 调整 `blockDim` 和 `tileElems`，平衡并行度与 DMA 启动开销。
3. 比较等步长 staging 与小窗口搬运的耗时，选择更合适的阈值。
4. 统计通用路径的 run 数和单块 DMA 数，再优化随机索引场景。
5. 根据 UB bank 冲突数据调整 buffer 起始地址和 padding。

Kernel 只使用每核 UB，接口不申请随 `sizeY` 或 `nnz` 增长的 workspace。

### Python/ATen 适配

Tensor 层使用一个共享的 `GatherTensorBridge`：

```text
torch.index_select / aten::index_select
                 ┐
                 ├─ GatherTensorBridge
任务测试 Hook ───┘      │
                        ▼
                aclsparseGather
```

Bridge 负责 Tensor 参数检查、输出分配、当前 stream 获取和 DnVec/SpVec 描述符生命周期。公开 PyTorch 入口创建 base 0 描述符；任务 Hook 接收 base 参数，用于覆盖 C++ 的 base 0/1 能力。

实现时先确认 torch_npu 26.0.0 中 `aten::index_select` 的现有 NPU 注册方式，再按仓库认可的扩展机制接入。两类入口都调用同一个 bridge，使测试 Hook 与公开接口不会形成两套实现。

### 工程落位

| 目录 | 内容 |
| --- | --- |
| `sparse/gather/arch22/` | Host、Tiling、Kernel 和构建配置 |
| `sparse/gather/README.md` | 接口能力、使用方式和约束 |
| `sparse/common/` | 必要的公共描述符或设备信息补充 |
| `test/gather/arch22/` | C++ UT/ST |
| `python/npu_sparse_gather/` | Tensor bridge、ATen 注册和任务 Hook |
| `python/npu_sparse_gather/tests/` | Python/ATen 端到端测试 |

公共描述符或头文件的修改同时回归已有 `arch35` Gather。

## 支持硬件

| 支持的芯片版本 | 支持状态 |
| --- | --- |
| Atlas A2 训练系列 910B3/910B4 | 支持 |
| 任务平台提供的 Atlas A3 型号 | 支持 |
| Ascend 950/A5 | 保留仓库已有实现 |

## 算子约束限制

- `vecY`、indices 和 values 使用一维连续存储，values dtype 一致。
- 受 SpVec 描述符约束，`0 <= nnz <= sizeY`。
- index 为 I32；换算 base 后位于 `[0, sizeY)`。
- `vecY` 和 indices 保持只读，输出写入 `vecX.values`。
- 输入与输出指针由调用方保证可被 handle 所属 stream 访问，数据区间不发生非法重叠。
- PyTorch functional 入口返回独立 Tensor，使用当前 stream 执行，不引入 CPU fallback。

# 可维可测分析

## 精度和正确性标准

Gather 只搬运数据，四种 dtype 均采用 bit-wise exact match：

| dtype | Golden 生成 | 比较方式 |
| --- | --- | --- |
| FP16 | FP32 数据转为 FP16 后在 CPU 按 index 选择 | 比较 16 位模式 |
| BF16 | FP32 数据转为 BF16 后在 CPU 按 index 选择 | 比较 16 位模式 |
| FP32 | FP64 数据转为 FP32 后在 CPU 按 index 选择 | 比较 32 位模式 |
| complex64 | complex128 数据转为 complex64 后在 CPU 按 index 选择 | 分别比较实部、虚部及完整 8 字节记录 |

输入 `Y` 和 indices 在调用前后保持一致；相同输入重复执行得到相同输出。

## 测试设计

| 类别 | 主要场景 |
| --- | --- |
| 基础功能 | 四种 dtype、base 0/1、动态 `sizeY/nnz`、`nnz=0/1` |
| 索引形态 | 连续、步长 17、回绕、乱序、重复、降序、首尾索引 |
| 边界 | 非 32B 对齐、尾 tile、`blockCount=4095` 附近和大长度算术 |
| 特殊值 | 正负零、小值、离群值、INF/NAN、complex64 实虚部 |
| 参数异常 | 空/失效描述符、dtype 不一致、非法 index type/base、越界 index |
| Python/ATen | Dispatcher 命中、输出 shape/dtype/device、默认与非默认 stream |
| 兼容性 | A2 910B3/910B4、A3，以及公共改动后的 arch35 回归 |

C++ UT/ST 验证公开接口和 Kernel；Python 测试验证 `torch.index_select` 与任务 Hook；Profiler 用于确认调用进入 `aclsparseGather -> arch22 Kernel`。

## 性能和内存

P-01、P-02、P-03 共覆盖四种 dtype 和 base 0/1，每个 case 的目标倍率为：

```text
性能倍率 = GPU Device Event 中位耗时 / NPU 同调用范围中位耗时 >= 0.25
```

每个 case 预热至少 10 次，采样 30 次，记录 median 和 p90。任务 Hook、公开 PyTorch 接口和 C++ 接口按各自调用范围分别统计。

算子 C++ 路径的额外内存由每核 UB 构成，GM workspace 为 0。Python functional 接口只分配语义所需的输出 Tensor。内存测试使用任务包提供的 GPU/NPU 采集脚本；当输入输出总量超过 500 MB 时，NPU 额外内存不超过 GPU 内存总量的 50%。

正式功能、精度、性能、内存和 Profiler 数据在实现完成后写入独立自测报告，不在设计文档中预填。

## 兼容性分析

`aclsparseGather` 的公开函数原型保持不变，A2/A3 通过独立 `arch22` 目录接入。公共描述符如需增加 dtype 或设备元数据，仅修改内部不透明结构，并对 Create/Get/Set/Destroy 和 arch35 Gather 做回归。

Python 适配只覆盖任务定义的一维 functional `index_select`，不改变其他 overload。后续如果仓库形成统一 Tensor adapter 目录，只调整代码落位，共享 bridge 和 C++ 调用链保持不变。

## 风险与应对

| 风险 | 处理思路 |
| --- | --- |
| 完全随机索引产生较多短 DMA | 通过 run 统计和窗口阈值调优，保留正确性通用路径 |
| torch_npu 已有 `index_select` 注册与扩展入口冲突 | 按 torch_npu 26.0.0 和社区评审认可的注册方式接入 |
| 不同 A2/A3 型号的 AIV/UB 参数不同 | Host 动态读取平台资源，分别执行功能和性能回归 |

## 参考依据

1. 本任务书：`aclsparseGather_A2A3_task_doc.md`。
2. CANN 社区任务算子设计文档模板：`resources/design_template.md`。
3. `ops-sparse/sparse/gather/arch35/`：已有接口和 A5 实现。
4. `ops-sparse/sparse/spvv/arch22/`：DAV_2201 的 `DataCopyPad + Gather` 参考。
5. CANN 9.1.0 Ascend C API 文档：`DataCopyPad`、`Gather`、TQue/TBuf 和平台信息接口。
6. 任务测试包：精度用例、性能用例、内存脚本和 `ops_sparse_test::gather_npu` 适配约定。
