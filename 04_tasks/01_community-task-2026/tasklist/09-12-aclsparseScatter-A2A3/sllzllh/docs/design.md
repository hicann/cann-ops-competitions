# aclsparseScatter 算子（Atlas A2/A3）设计文档

> 社区任务：9 月社区任务——aclsparseScatter 算子开发（A2/A3）  
> 目标代码仓：[`cann/ops-sparse`](https://gitcode.com/cann/ops-sparse)  
> 设计文档目标路径：`04_tasks/01_community-task-2026/tasklist/09-12-aclsparseScatter-A2A3/sllzllh/docs/design.md`  
> 目标平台：Atlas A2/A3，DAV_2201（`arch22`）
> 当前 C++ 实现基线：`cann/ops-sparse` 分支 `feat/scatter-arch22-p2`，提交 `1a26f5152d91170cfde9eeeaf0480bef7608435d`

# 需求背景（required）

## 需求来源

本设计来源于 CANN 社区任务 2026 的 `aclsparseScatter` A2/A3 算子开发任务。任务要求参考
cuSPARSE `cusparseScatter` 的接口语义，在 `ops-sparse` 中完善公开 C++ API、Host 参数校验、
Ascend C Kernel、C++ UT/ST、Python/ATen NPU 适配及端到端测试，并满足精度、性能和内存验收
要求。

设计文档按照 CANN 社区任务官方
[算子设计文档模板](https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md)
编写，后续以独立文档 PR 提交至 `cann-ops-competitions` 对应任务目录；算子实现以独立代码 PR
提交至 `cann/ops-sparse` 的 `master` 分支。

## 背景介绍

### 算子功能

`aclsparseScatter` 将稀疏向量 `X` 的 values 按 indices 原地写入稠密向量 `Y`：

$$
Y[X.indices[i]-idxBase] = X.values[i],\quad i\in[0,nnz)
$$

其中：

- `vecX` 是只读 SpVec 描述符，承载逻辑长度、`nnz`、Device indices、Device values、索引类型、
  index base 和 value dtype；
- `vecY` 是 DnVec 描述符，承载原地更新的 Device 稠密向量；
- 未命中的 `Y` 元素保持原值；`vecX.indices` 和 `vecX.values` 不得被修改；
- 无重复索引时，各目标位置只写一次，输出必须逐 bit 一致；
- 有重复索引时，多个更新竞争同一位置，结果是非确定性 last-write-wins，不做累加，也不承诺
  固定获胜者；
- `nnz=0` 时完成参数与元数据校验后直接成功返回，不启动 Kernel，不改变 `Y`。

该能力可用于稀疏特征回填、图计算、词表局部写入和稀疏格式转换的数据整理。任务书中的三个
重点性能场景分别来自 Llama 3.1 70B、Qwen3-235B-A22B 和 DeepSeek-V3 的词表规模。

### 现有实现与差距

公开接口已在 `include/cann_ops_sparse.h` 中声明，SpVec/DnVec 与 handle/stream 基础设施也已存在。
`sparse/scatter/` 当前采用按架构分目录的实现方式。arch22 C++ 核心已经推进到 P2，Python/ATen
入口及跨型号验收仍待补齐：

| 项目 | arch22（A2/A3）当前实现 | arch35（A5）既有实现 | 任务最终要求 |
| --- | --- | --- | --- |
| C++ API | 已存在 | 已存在 | 保持 ABI 不变 |
| value dtype | int8/FP16/BF16/FP32/complex64 | FP32/FP16/BF16 | int8/FP16/BF16/FP32/complex64 |
| index type | I32 | I32/I64 | I32 |
| index base | base 0/1 | base 0/1 | base 0/1 |
| indices | 顺序、倒序、乱序、重复、尾块 | 支持乱序 | 支持乱序/重复 |
| Python/ATen | 尚未实现正式入口 | 未形成任务要求的完整调用链 | `Tensor.index_copy_(0, index, source)` |
| 资源 | 无线性 workspace | 无线性 workspace | 无线性 Device/Host 临时内存 |

P0 原型消除了固定 64 核 tiling 数组，将 `TPipe` 移至 Kernel 入口局部；P1 补齐五种 dtype、
base 0/1、完整元素写和 200 条 C++ 精度用例；P2 将连续 run 打包到 32-byte 对齐 VECOUT slot，
一次提交当前 tile 的写命令并统一等待 MTE3 完成。提交 `1a26f51` 已在 910B3 上通过 14/14 API、
200/200 精度、30/30 P 门禁、130/130 无重复性能用例以及 130 用例每例追加 1000 次的稳定性运行。
该结果证明 910B3 C++ 核心路径成立，但不替代 Python/ATen、正式内存/Profiler、910B4/A3 和
arch35/A5 回归证据。

# 需求分析（required）

## 需求描述

在 Atlas A2/A3（DAV_2201，`arch22`）上完善 `aclsparseScatter`，实现以下完整能力：

1. 支持 `int8`、`float16`、`bfloat16`、`float32`、`complex64` 五种 value dtype；
2. indices 仅支持 I32，index base 支持 0 和 1；
3. 支持动态 `size/nnz`、`nnz=0/1`、尾块、顺序、倒序、乱序和重复索引；
4. 保证无重复索引结果 bit-wise exact，重复索引结果属于对应输入 values 的候选集合；
5. 保持 `vecX` 只读、`vecY` 原地更新、未命中位置不变；
6. 沿用调用方 stream 异步执行，不在 Host 常规路径 D2H 或同步；
7. 不新增 preprocess，不申请随 `size` 或 `nnz` 线性增长的 Device/Host 临时内存；
8. 建立 `Tensor.index_copy_(0, index, source)` 到 `aclsparseScatter` 的 ATen/NPU 调用链，禁止
   CPU fallback；
9. 按任务书完成精度、性能、峰值内存、Profiler、跨 SoC 和 arch35 回归证据。

## 需求拆解

| 编号 | 需求 | 设计落点 | 验证方式 |
| --- | --- | --- | --- |
| R-API | 保持公开 C++ 接口兼容 | 不修改函数原型；统一 Host 合同和状态码 | Host/API UT |
| R-DTYPE | 五种 value dtype | Host 分发、元素宽度模板、完整元素写 | 五 dtype C++/ATen/E2E |
| R-INDEX | I32、base 0/1、乱序/重复 | tiling 携带 base；Kernel 原位归一化 | directed + 随机用例 |
| R-INPLACE | Y 原地、X 只读 | 直接写调用方 Y；不修改输入 GM | storage identity + 回读 |
| R-ASYNC | 调用方 stream、无 Host 同步 | 单次异步 Kernel launch | 多 stream/Profiler |
| R-TILING | 动态核数、均衡尾块 | 标量 `perCoreN/remainder` | tiling UT + 尾块 ST |
| R-ACC | bit-wise 与重复候选语义 | 纯搬运、不做数值转换 | CPU Golden/raw bits |
| R-PYTORCH | `aten::index_copy_` NPU 路径 | Dispatcher adapter + RAII 描述符 | ATen/Python/Profiler |
| R-PERF | 每个 P case ≥0.25× | 连续 run 批量写、低开销通用路径 | Event median/p90 |
| R-MEM | 无线性额外内存 | 仅常量 Host tiling 和单核 UB | memory 脚本/Profiler |
| R-COMPAT | arch22/arch35 共存 | 架构目录隔离、公共 ABI 不变 | 双架构构建回归 |

### 支持矩阵

| 维度 | 最终支持范围 |
| --- | --- |
| value dtype | `ACL_INT8`、`ACL_FLOAT16`、`ACL_BF16`、`ACL_FLOAT`、`ACL_COMPLEX64` |
| index type | `ACL_SPARSE_INDEX_32I` |
| index base | `ACL_SPARSE_INDEX_BASE_ZERO`、`ACL_SPARSE_INDEX_BASE_ONE` |
| shape | 一维连续 SpVec/DnVec，动态 `size/nnz` |
| index pattern | 顺序、倒序、乱序、重复；性能用例不含重复 |
| output | `vecY` 原地更新，未命中元素保持原值 |
| workspace | 0；无 preprocess |
| Python/ATen | 一维 `Tensor.index_copy_(0, index, source)`，index 为 I32 |

# 详细设计（required）

## 算子分析

### 数学公式与执行语义

设归一化目标下标：

$$
j_i=X.indices[i]-idxBase
$$

调用方须保证 $0\le j_i<size$。算子执行：

$$
Y_{out}[j_i]=X.values[i]
$$

且对所有未被任何 $j_i$ 命中的位置 $k$：

$$
Y_{out}[k]=Y_{in}[k]
$$

当 indices 互异时，不同更新之间不存在写冲突，可按 `i` 维度多核并行。当 indices 重复时，
不同核或同一核中的写入顺序共同决定最终值；最终值必须是写入该位置的某个完整输入元素，不能
出现累加、混合或 complex64 实虚部撕裂。

### 数据类型与存储语义

Scatter 不进行数值计算，所有 dtype 均按原始存储位搬运：

| dtype | 元素字节数 | Kernel 存储视图 | 精度语义 |
| --- | ---: | --- | --- |
| int8 | 1 | `int8_t`/原始字节 | 8 bit 原样复制 |
| float16 | 2 | `half`/16 bit 原始值 | 保留符号、NaN payload 等全部位 |
| bfloat16 | 2 | 16 bit 原始值 | 不转 FP32 计算，不改写位模式 |
| float32 | 4 | `float`/32 bit 原始值 | 32 bit 原样复制 |
| complex64 | 8 | 单个完整 8 字节元素 | 实部和虚部作为一个元素搬运 |

虽然任务书规定 FP16/BF16、FP32、complex64 的 CPU Golden 分别使用更高精度类型生成，但
Scatter 最终比较标准是原始元素 bit-wise exact；这是比数值误差阈值更严格且更符合纯搬运
算子的判断方式。

### 支持形状

- `vecX.indices`：`[nnz]`，I32；
- `vecX.values`：`[nnz]`，与 `vecY` dtype 相同；
- `vecY.values`：`[size]`，一维连续；
- `0 <= nnz <= size <= INT32_MAX`，且 `vecX.size == vecY.nums`；
- `nnz=0` 允许 `vecX.indices/values == nullptr`；`size>0` 时 `vecY.values` 必须有效。

选择 `INT32_MAX` 作为共同上限，可以同时覆盖 I32 base 0/1，避免 base 1 的最大合法索引和
Kernel 地址换算发生有符号溢出。所有字节数和地址区间计算使用受检 `uint64_t/size_t`。

### 性能特征

该算子的读侧为连续 indices/values，写侧可能完全随机。主要开销是：

1. Host API、描述符和 Kernel launch 的固定开销；
2. indices/values 从 GM 连续搬入 UB；
3. 乱序目标导致的非合并 GM 写；
4. 小 `nnz` 下核启动和多核调度占比偏高；
5. 连续或局部连续 indices 下，逐元素写会浪费可合并带宽。

因此选择“通用直接 scatter + 连续 run 批量写”的双路径，不在 Host 或 Device 侧全量排序。

## 算子实现

### 总体架构

```text
Tensor.index_copy_（Python/ATen，任务最终必选入口，当前待实现）
  └─ NPU Dispatcher adapter
       ├─ shape/dtype/layout/device/alias 校验
       ├─ 当前 NPU stream
       └─ SpVec/DnVec RAII 描述符
            ↓
aclsparseScatter（公开 C++ API）
  ├─ Host 合同与溢出校验
  ├─ 动态核数和标量 tiling
  └─ 异步下发 arch22 AIV Kernel
       ├─ indices/values 连续 CopyIn
       ├─ base 归一化与连续 run 检测
       ├─ 连续 run 批量 CopyOut
       └─ 一般乱序/重复索引完整元素间接写
            ↓
调用方 vecY 原地更新
```

C++ 正常执行路径只有一次 Host tiling 和一次异步 Kernel launch。输入数据不回 Host，不创建
Device workspace。最终 Python/ATen 适配也不得创建 `index.long()`、`contiguous()` 或临时输出。

### 方案选型

| 方案 | 优点 | 问题 | 结论 |
| --- | --- | --- | --- |
| 直接多核 scatter | 无 workspace、低延迟、天然支持乱序/重复 | 随机 GM 写难合并 | 作为通用路径 |
| 先排序再批量写 | 连续写比例高 | 需要 O(nnz) 临时内存和额外 Kernel，改变重复竞争顺序 | 不采用 |
| Host 回读 indices | 易做边界/连续性分析 | 引入 D2H 和 Host 同步，破坏异步语义 | 禁止 |
| 单核顺序写 | 重复索引结果稳定 | 大规模性能差且错误承诺确定性 | 不采用 |
| 直接路径 + run 检测 | 无全量临时内存，连续场景可批量写 | 需维护通用尾块路径 | 选定方案 |

### 公开接口设计

接口原型保持不变：

```cpp
aclsparseStatus_t aclsparseScatter(
    aclsparseHandle_t handle,
    aclsparseConstSpVecDescr_t vecX,
    aclsparseDnVecDescr_t vecY);
```

参数合同：

| 参数 | 输入/输出 | 内存域 | 约束 |
| --- | --- | --- | --- |
| `handle` | 输入 | Host | 有效句柄，已通过 `aclsparseSetStream` 绑定 stream |
| `vecX` | 输入、只读 | Host 描述符/Device 数据 | size/nnz 合法；I32；base 0/1；支持 dtype；数据指针在异步完成前有效 |
| `vecY` | 输入输出 | Host 描述符/Device 数据 | 长度与 `vecX.size` 相等；dtype 相同；原地写入 |

### Host 侧设计

Host 实现拆分为参数校验、tiling 构造和 Kernel launch，公开入口只负责组织调用和返回状态。
校验顺序固定，防止 `nnz=0` 绕过非法元数据：

1. `handle` 非空；
2. `vecX/vecY` 非空且描述符签名有效；
3. `size/nnz`、dtype、index type/base 合法，且 X/Y 兼容；
4. handle 已绑定 stream；
5. 必需数据指针、对齐和 alias 合法；
6. `size*elementBytes`、`nnz*elementBytes` 和地址区间无溢出；
7. `nnz=0` 成功早退；
8. 获取动态 AIV 核数，构造 tiling，并在调用方 stream 异步下发 Kernel。

状态码合同：

| 场景 | 状态码 |
| --- | --- |
| handle 为空 | `ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR` |
| 描述符为空/签名错误 | `ACL_SPARSE_STATUS_INVALID_VALUE` |
| shape、nnz、字节数或地址区间非法 | `ACL_SPARSE_STATUS_INVALID_VALUE` |
| X/Y dtype 不一致 | `ACL_SPARSE_STATUS_INVALID_VALUE` |
| 非 I32 index 或不支持 dtype | `ACL_SPARSE_STATUS_NOT_SUPPORTED` |
| idxBase 非法枚举 | `ACL_SPARSE_STATUS_INVALID_VALUE` |
| stream 未绑定、必需指针为空、错设备或非法 alias | `ACL_SPARSE_STATUS_INVALID_VALUE` |
| 获取设备核数失败 | `ACL_SPARSE_STATUS_INTERNAL_ERROR` |

`vecX.indices/values` 按对标接口要求 16 字节对齐；`vecY.values` 至少满足 dtype 自然对齐。
三段有效地址区间不得重叠。C++ Host 当前只校验指针值、对齐和地址区间，不通过 D2H 或同步查询
Device indices；索引值范围及指针所属 Device 属于 C++ 调用方前置条件。Python/ATen 层必须先
保证三个 Tensor 位于同一 NPU Device，并对值域异常采用 torch_npu 可用的 NPU 侧异步错误机制，
不得回退到 CPU 扫描。

### Tiling 设计

TilingData 只保存标量，禁止按“每核一个字段”保存固定数组。Host 通过仓库公共
`GetAivCoreCount()` 获取实际 AIV 核数，不直接调用平台私有接口，也不硬编码某一型号的核数。
当前 P2 使用经过 arch22 编译和实机验证的固定 tile 上限 `SCATTER_TILE_NN_MAX=2048`，而不是在
Host 侧调用 `GetUbSize()` 动态推导 UB 容量：

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `nnz` | `uint32_t` | 更新元素总数 |
| `blockNum` | `uint32_t` | 实际使用 AIV 核数 |
| `tileNn` | `uint32_t` | 单次 UB tile 元素数 |
| `perCoreN` | `uint32_t` | 每核基础元素数 |
| `remainder` | `uint32_t` | 前 `remainder` 个核各多处理一个元素 |
| `idxBase` | `uint32_t` | 0 或 1 |
| `elementBytes` | `uint32_t` | 1、2、4 或 8 |
| `reserved` | `uint32_t` | ABI 对齐预留，当前置 0 |
| `ySize` | `uint64_t` | Y 的逻辑元素数 |

最终分核公式为：

$$
blockNum=\min(aivCoreNum, nnz)
$$

令最大单核元素数为 `maxPerCore=perCoreN+(remainder>0)`。`tileNn` 从 1 开始按 2 的幂增长，
直到覆盖 `maxPerCore` 或达到 2048：

$$
tileNn=\min\left(2^{\lceil\log_2(maxPerCore)\rceil},2048\right)
$$

当前单核 UB 预算为：

```text
2 × Align32(tileNn × sizeof(int32))   # indices VECIN 双缓冲
+ 2 × Align32(tileNn × elementBytes)  # values VECIN 双缓冲
+ tileNn × 32                         # VECOUT：最坏每个元素一个 32B 对齐 run slot
+ 对齐与控制预留
```

VECOUT 之所以按 `tileNn×32` 预留，是因为完全随机索引时每个元素都会形成独立 run，而
`DataCopyPad` 的 LocalTensor 起始地址必须按完整 data block 对齐。2048 上限保证两组 VECIN
双缓冲与最坏 VECOUT slot 能落入 arch22 UB 预算；该常量若调整，必须重新完成编译资源检查、
五 dtype 精度和三类硬件性能回归。

### Kernel 侧设计

#### 分核

对 `blockIdx`：

$$
coreCount=perCoreN+[blockIdx<remainder]
$$

$$
coreOffset=blockIdx\times perCoreN+\min(blockIdx,remainder)
$$

该切分覆盖 `[0,nnz)` 且无遗漏、无重叠，并可适配不同 A2/A3 型号的实际 AIV 核数。`TPipe`
在 `extern "C" __global__ __aicore__` Kernel 入口中定义，再以指针传给算子对象，避免将 `TPipe`
作为成员导致资源生命周期不符合规范。入口通过 `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY)`
显式声明 AIV-only 任务类型。

#### 核内流水

1. `CopyIn`：使用 `DataCopyPad` 将当前 tile 的连续 indices 和 values 从 GM 搬入 VECIN；
2. `MTE2→S`：等待当前 tile 搬入完成，Scalar 读取 indices；
3. 第一遍 run 扫描：识别相邻递增 run，将每个 run 的原始 values 打包到独立的 32-byte 对齐
   VECOUT slot；
4. `S→MTE3`：保证 Scalar 打包对 MTE3 可见；
5. 第二遍 run 扫描：按 `index[runStart]-idxBase` 计算目标位置，为每个 run 提交一次
   `DataCopyPad`；单元素随机写是长度为一个完整元素的 run；
6. `PipeBarrier<PIPE_MTE3>()`：在归还并复用 VECOUT 前统一等待本 tile 的所有写命令完成；
7. 尾块使用实际 `count` 控制有效字节，UB padding 不写入 Y 合法范围外，然后顺序处理下一 tile。

输入队列按双缓冲资源配置，但当前 `Process` 循环按 tile 顺序执行 CopyIn、run 打包和 CopyOut；
设计不宣称已经实现“下一 tile MTE2 与当前 tile MTE3”的流水重叠。当前优化收益来自减少每个 run
的队列申请/释放和同步次数，以及连续 run 的批量写。

#### 通用乱序与连续 run

对 tile 内相邻元素，如果：

$$
index[i+1]=index[i]+1
$$

则二者属于同一 run。run 长度大于 1 时，将对应 values 作为连续字节区间批量写入 Y；否则走
单元素间接写。该策略只利用输入原有顺序，不排序、不重排，不依赖全局单调性，因此真正乱序的
唯一索引仍然正确。

#### dtype 分发与完整元素写

Kernel 使用统一的原始字节视图，不按 dtype 编译五份模板实例：

- Host 将五种 dtype 映射为 `elementBytes=1/2/4/8`，Kernel 只按元素宽度计算字节偏移；
- indices 保持 I32 视图；values、VECOUT 和 Y 使用 `uint8_t` 原始存储视图；
- 连续 run 的搬运长度统一为 `runLen*elementBytes`，单元素路径严格提交一个完整元素长度；
- complex64 的 8 字节先整体打包到同一个对齐 slot，再由一次 8 字节 `DataCopyPad` 写请求提交，
  不拆分实部和虚部；
- 特殊值不进入算术指令，NaN payload、正负零和 complex64 两分量均按位保留。

重复索引跨核写入的完成顺序不确定，符合任务定义。验证时对每个重复目标位置构造候选 values
集合，要求输出等于其中某个完整元素；complex64 同时检查实部和虚部来自同一个候选项。

### Python/ATen 适配设计（任务必选，当前待实现）

公开入口为：

```python
self.index_copy_(0, index, source)
```

适配层在 NPU DispatchKey 注册 `aten::index_copy_` 对应实现，并限定到本任务声明的一维路径：

1. `self`、`source`、`index` 均在同一 NPU Device；
2. `dim == 0`，三者均为一维连续 Tensor；
3. `source.numel() == index.numel()`，`self.numel() <= INT32_MAX`；
4. `self.dtype == source.dtype`，且属于五种 dtype；
5. `index.dtype == torch.int32`，不隐式创建 `index.long()` 或 I64→I32 临时 Tensor；
6. storage offset 对应的数据地址满足对齐，三者不存在未声明的 storage overlap；
7. 复用 adapter/runtime 管理的长生命周期 aclsparse handle，并在每次调用时绑定当前 NPU
   stream；禁止每次算子调用创建和销毁 handle，因为现有 handle 销毁流程会同步 stream；
8. 将 `source+index` 组装为只读 SpVec，将 `self` 组装为 DnVec。调用
   `aclsparseScatter` 后可立即销毁仅位于 Host 的临时描述符并返回同一 `self`；Device 数据生命
   周期由 stream 语义和 Tensor 持有关系保证，不得 Host synchronize；
9. 不支持的 dim、shape、dtype、stride、device 和 alias 显式抛错，Profiler 中不得出现 CPU
   fallback 或 D2H 索引回读。

PyTorch 通用文档常见 index 类型为 LongTensor，但本社区任务明确规定 I32；因此正式 NPU 路径
只接受 int32，并在接口文档中明确差异。当前提交 `1a26f51` 尚未包含本节代码、Dispatcher UT
和 Python E2E；这三项属于代码 PR 和最终验收的阻塞项，不能以 C++ 性能结果替代。

### 内存与资源设计

Device 常驻数据仅为调用方提供的：

```text
indices = nnz × sizeof(int32)
values  = nnz × elementBytes
Y       = size × elementBytes（原地）
```

额外资源只包含常量大小的 Host 描述符/tiling 和每个 AIV 的 UB 队列，不随全局 `size/nnz`
线性增长。单核 UB 由两组 VECIN 双缓冲和一个 `tileNn×32B` VECOUT 组成，`tileNn<=2048`。
接口不提供 bufferSize/preprocess，不申请 Device workspace，Host 不建立 indices 副本。ATen adapter
复用已有 handle，不把 handle 的公共默认 workspace 作为 Scatter 的逐调用申请或计入算子固有
workspace。最终仍须通过任务包内存脚本记录 GPU/NPU 的 baseline、peak 和 extra_peak；当前
`workspace_bytes=0` 只证明算子接口未申请 workspace，不能替代峰值内存报告。

### 工程文件规划

| 路径 | 设计职责 | 当前状态 |
| --- | --- | --- |
| `include/cann_ops_sparse.h` | 保持 API 原型，修订平台能力、异步、对齐、上限和错误说明 | 已更新 |
| `sparse/scatter/arch22/scatter.h` | 标量 tiling、2048 tile 上限和双缓冲常量 | 已实现 |
| `sparse/scatter/arch22/scatter_host.cpp` | 合同校验、元素宽度映射、动态核数、异步 launch | 已实现 |
| `sparse/scatter/arch22/scatter_kernel.h/.cpp` | base 0/1、尾块、run 打包和随机写路径 | 已实现 |
| `test/scatter/arch22/scatter_test.cpp` | C++ Host/API 与 NPU directed/random 测试 | 已实现 |
| `test/scatter/arch22/scatter_perf.cpp` | P30、unique130、原始样本和稳定性入口 | 已实现 |
| `test/scatter/` 公共测试文件 | 参数、Golden、CMake 和复现 README | 已更新 |
| ATen/Python 适配与测试目录 | Dispatcher 注册、adapter、ATen UT、Python E2E | 待与 maintainer 固化目录后实现 |
| `sparse/scatter/README.md` | 能力矩阵、接口合同、限制和调用示例 | 已更新 |

### 分阶段实施

| 阶段 | 开发内容 | 退出条件 | 当前状态 |
| --- | --- | --- | --- |
| P0 | FP32/I32/base 0 基础闭环；标量 tiling；Host 合同；规范化 Kernel；确定性测试 | arch22 构建通过，P0 directed/API 用例 100% bit-wise 通过 | 已完成 |
| P1 | 五 dtype、base 1、重复索引完整元素写、Device/alias 合同、ATen/Python 正式入口 | C++/ATen/Python 能力矩阵和 200 条 accuracy 全通过，无 CPU fallback | C++ 部分完成；ATen/Python 待完成 |
| P2 | 核数/tile/run 优化，30 个 P case、130 个无重复泛化 case、内存与跨 SoC 收口 | 每个 P case ≥0.25×，跨 910B3/910B4/A3 与 arch35 回归完成 | 910B3 C++ 性能与稳定性完成；其余验收项待完成 |

## 支持硬件

| 支持的芯片版本 | 架构 | 本任务范围 |
| --- | --- | --- |
| Atlas A2 训练系列 910B3 | DAV_2201 / `arch22` | 功能、精度、性能、内存 |
| Atlas A2 训练系列 910B4 | DAV_2201 / `arch22` | 功能、精度、性能、内存 |
| 任务环境提供的 Atlas A3 型号 | DAV_2201 / `arch22` | 功能、精度、性能、内存 |
| Atlas A5 / Ascend 950 | DAV_3510 / `arch35` | 既有实现回归，不在本任务新增范围 |

软件环境按任务书记录实际版本，验收基线为 CANN 9.1.0 及后续配套版本、PyTorch 2.7 及以上、
torch_npu 26.0.0 及之后版本。

## 算子约束限制

1. `vecX.values` 与 `vecY.values` dtype 必须一致，且只支持五种声明 dtype；
2. C++ 与正式 NPU adapter 的 indices 仅支持 I32，index base 仅支持 0/1；
3. `vecX.size == vecY.nums`，`0 <= nnz <= size <= INT32_MAX`；
4. 换算 base 后的每个 index 必须位于 `[0,size)`。C++ 快路径不扫描 Device indices；越界
   按公开前置条件或 NPU 异步错误协议处理；
5. `vecX.indices/values` 只读，`vecY` 原地更新；除正式的 Y 原地语义外，不允许输入输出
   alias；
6. 重复索引为非确定性 last-write-wins，不提供累加语义，不承诺固定结果；
7. 输入描述符和 Device 内存在关联 stream 完成前保持有效；接口本身不负责同步；
8. 无 workspace、无 preprocess、无 CPU fallback、无 Host D2H 索引检查；
9. Python/ATen 仅声明一维连续、`dim=0`、int32 index 路径，其他组合显式拒绝；
10. 性能验收用例必须使用无重复索引，重复索引用例只进入功能/精度集合。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | 无重复索引逐元素 bit-wise exact；重复索引输出属于对应输入完整元素集合；X 只读，Y 未命中位置不变；覆盖零、正负、边界、INF/NAN 和 complex64 完整性 | 任务书 §3.2、生态算子开源精度标准 |
| 性能标准 | P-01/P-02/P-03 的每个有效 dtype/base case 均达到 `GPU Event median / NPU 同调用范围 median >= 0.25`；预热≥10，采样≥30，报告 median/p90/raw samples | 任务书 §3.3 |
| 内存标准 | 固有 Device workspace 为 0，不超过目标硬件 L2 Cache；不新增随 size/nnz 线性增长的临时内存 | 任务书 §3.4 |

任务性能场景：

| 编号 | size | nnz | GPU Event baseline median_us | 有效组合 |
| --- | ---: | ---: | ---: | ---: |
| P-01 | 128256 | 8192 | 84.304–92.752 μs | 5 dtype × 2 base |
| P-02 | 151936 | 4096 | 83.008–86.064 μs | 5 dtype × 2 base |
| P-03 | 129280 | 7168 | 83.648–86.080 μs | 5 dtype × 2 base |

GPU C++ 标杆和 PyTorch GPU 标杆属于不同调用范围，分别与相同范围的 NPU 结果比较，不混用。

## 功能与精度测试设计

1. **接口合同**：空 handle/描述符/必需指针，签名错误，stream 未绑定，shape/nnz/溢出，
   dtype/index/base，错设备、对齐和 alias；
2. **基础功能**：`size=0`、`nnz=0/1`、`nnz=size`、小/中/大规模，32B 对齐和非对齐尾块；
3. **索引分布**：连续、升序不连续、倒序、stride、真正乱序唯一、跨核边界；
4. **重复索引**：核内、跨 tile、跨核、全重复；按候选集合判断，不把某个固定 winner 写入 Golden；
5. **dtype**：五 dtype × 两 base 全矩阵，特殊值覆盖原始 bit、±0、极值、INF/NAN；
6. **complex64**：检查实虚部来自同一候选元素，不允许撕裂；
7. **只读/原地**：执行前后回读 indices/values，检查未命中 Y、storage pointer 和返回 self；
8. **stream/生命周期**：双 stream、同 stream 排队读回、连续创建执行销毁、异步资源生命周期；
9. **ATen/Python**：Dispatcher 命中、异常、原地返回、无 CPU fallback；
10. **规模集合**：任务包 200 条 accuracy case 全量执行，并保留逐 case 可解析结果。

随机数据使用固定 seed。values 按任务书要求包含 70% `[-1,1]` 均匀分布、20% 正态分布和
10% 零/边界/允许的特殊值；indices 在合法范围内均匀生成，并单独构造重复/OOB 用例。

## 性能测试设计

1. 执行 P-01/P-02/P-03 共 30 个 `dtype×base` hard-gate case，性能集合排除重复索引；
2. 每个 case 预热至少 10 次，采样至少 30 次，固定数据与描述符；
3. NPU 统计同一 API 范围内全部 Kernel 总耗时，报告 median、p90 和原始 samples；
4. 分别记录 aclsparse C++ Kernel 范围和 Python/ATen 端到端范围；
5. 执行 130 条无重复泛化性能用例，防止只针对三种 shape 过拟合；
6. 使用 `scatter_perf --all --stability 1000` 在每条正式采样后追加 1000 次调用，检查长时间执行；
7. profiler 分析 Host launch、小 `nnz` 核启动、MTE2 读、随机 MTE3 写、run 命中率和核间
   负载；
8. 所有优化必须回归 P0/P1 正确性，不允许以排序、Host 预处理或临时 Device 数组换性能。

## 内存与无 fallback 验证

- 用任务包 GPU/NPU memory 脚本按同一 case id 比较输入基线、峰值和额外峰值；
- profiler/运行时记录显示 workspace 为 0，采样期间无新增 `aclrtMalloc`；
- Python profiler 必须出现 `aclsparseScatter`/对应 arch22 Kernel，且无 CPU scatter、D2H
  indices、隐式 cast/contiguous；
- 正常调用在 Host API 返回前不得出现 `aclrtSynchronizeStream`；使用调用方 stream 上的后续
  操作验证异步有序性。

## 分阶段验证记录

### P0 技术路径

| 项目 | 结果 |
| --- | --- |
| 分支/提交 | `feat/scatter-arch22-p0` / `1d9c9b8` |
| 环境 | ARM64 Ubuntu 22.04，`ascend910b` 构建目标 |
| 命令 | `bash build.sh --ops=scatter --soc=ascend910b --run` |
| 总结果 | 69/69 PASS，退出码 0 |
| API 异常 | 12/12 PASS |
| 功能 | 57/57 PASS，含 30 随机、2 个百万 nnz 用例 |
| 精度 | bit-wise 判定通过，所示 MARE/MERE 均为 0 |
| 结论 | FP32/I32/base 0 的 Host→tiling→Kernel 技术路径成立 |
| 未覆盖 | 五 dtype/base 1/重复/ATen/Python/性能/OAT/跨 SoC/arch35 回归 |

### P2 当前 C++ 基线

以下结果均绑定 `cann/ops-sparse` 提交
`1a26f5152d91170cfde9eeeaf0480bef7608435d`，运行于 910B3、Release 构建；正式自测报告还须补齐
CANN、驱动、固件和完整硬件型号信息。

| 项目 | 结果 |
| --- | --- |
| 构建与 API | Release 构建通过；Host/API 14/14 |
| 精度 | 200/200，覆盖五 dtype、base 0/1、唯一/重复、尾块和 complex64 完整候选判断 |
| 性能 smoke | 3/3，通过 |
| P-01/P-02/P-03 | 30/30，每条 GPU/NPU 参考倍率均高于 0.25；一次正式运行最低 2.101× |
| 无重复泛化 | 130/130，其中 30 条 P 用例 `PASS`、100 条泛化用例 `RECORDED`，Failures 0 |
| 稳定性 | `--all --stability 1000`：130/130，每例追加 1000 次，共 130000 次；最低 P 倍率 2.072×，Failures 0 |
| 计时协议 | Device Event；warmup=10、samples=30；保存 median/p90/raw samples |
| Workspace | `workspace_bytes=0` |

证据边界：GPU 参考来自任务包 PyTorch Device Event，而当前 NPU 数值来自 C++
`aclsparseScatter` Device Event；二者用于当前核心性能预验收。最终报告仍须补同调用范围的
Python/ATen NPU 结果、峰值内存、Profiler、910B4/A3 和 arch35/A5 回归，不能将本表解释为
任务最终验收已完成。

## 兼容性分析

- **公开 ABI**：不修改 `aclsparseScatter(handle, vecX, vecY)` 原型，不新增产品私有平行 API；
- **既有 arch22 调用**：FP32/I32/base 0 是最终能力子集，合法调用语义保持不变；更严格的描述符、
  shape、stream、对齐和 alias 校验只拒绝原本未定义或不安全的调用；
- **arch35 共存**：构建系统按 SoC 选择 `arch22/arch35`，A2/A3 PR 不改写 arch35 Kernel；公共
  头文件和 README 以平台支持矩阵描述差异；
- **合入顺序**：若 A5 同名算子 PR 先合入，A2/A3 PR 基于最新 master 解决公共 Host/README
  冲突，并执行双架构构建和既有用例回归；
- **Python/ATen**：仓内尚无可直接复用的正式 torch 适配骨架，物理目录、构建 target 和注册
  方式在实现前与 maintainer 固化；逻辑合同、公开入口和无 fallback 要求不因目录调整而改变；
- **行为差异**：PyTorch 常用 I64 index，而本任务正式 NPU 路径仅支持 I32。该限制通过明确
  异常和文档暴露，不做隐式转换。

### 风险与应对

| 风险 | 影响 | 应对 |
| --- | --- | --- |
| arch22 随机 GM 写原语吞吐不足 | P case 不达标 | 保留正确通用路径，按 profiler 调整核数/tile，强化连续 run 批量写 |
| int8/FP16/BF16 尾块误写相邻元素 | 正确性/内存破坏 | 按字节计算有效长度，UB padding 不出界，guard 区专项测试 |
| complex64 重复写发生撕裂 | 输出不是候选完整元素 | 使用完整 8B 元素写，跨核重复专项与 raw-bit membership 验证 |
| Device index 越界难以同步报错 | 异步错误语义不清 | C++ 文档固定前置条件；ATen 使用 NPU 异步错误机制，OOB 独立进程验证 |
| torch 适配基础设施缺失 | 集成路径延迟 | 先与 maintainer 冻结目录/target/DispatchKey，再实现公共最小骨架 |
| A2/A3 与 A5 公共文件冲突 | 回归或合入阻塞 | 架构逻辑隔离，rebase 后双架构编译测试 |

# 参考资料

1. [CANN 社区任务 2026 算子设计文档模板](https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md)
2. [CANN 社区任务 2026 提交流程](https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/README.md)
3. [ops-sparse 代码仓](https://gitcode.com/cann/ops-sparse)
4. [生态算子开源精度标准](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md)
5. [cuSPARSE Scatter](https://docs.nvidia.com/cuda/cusparse/#cusparsescatter)
6. [PyTorch `Tensor.index_copy_`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.index_copy_.html)
