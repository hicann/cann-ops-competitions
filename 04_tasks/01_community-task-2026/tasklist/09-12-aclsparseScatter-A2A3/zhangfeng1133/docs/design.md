# aclsparseScatter 算子设计文档（Atlas A2/A3）

> 社区任务：9 月社区任务——aclsparseScatter 算子开发（A2/A3）
> 开发仓库：`gitcode.com/zhangfeng1133/my_aclsparseScatter`（分支 `v1-cann9.1.0-torch2.10.0`）
> 代码提交：`8cf6915 docs(selftest): 统一自测报告与设计文档中的证据路径`
> 目标平台：Atlas A2（910B3/910B4）/ Atlas A3，DAV_2201（`arch22`）
> 软件环境：CANN 9.1.0 + torch 2.10.0 + torch_npu 2.10.0.post4

# 需求背景（required）

## 需求来源

- 来源：aclsparseScatter 社区开源算子开发任务（A2/A3 验收口径，硬件 arch22），
  任务书《aclsparseScatter 算子开发任务书（A2/A3）》，开发仓库
  `gitcode.com/zhangfeng1133/my_aclsparseScatter`（分支 `v1-cann9.1.0-torch2.10.0`）。
- 软件环境：CANN 9.1.0 + torch 2.10.0 + torch_npu 2.10.0.post4。
- 设计文档按社区任务
  [设计文档模板](https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md)
  编写，以 PR 形式提交至 `cann-ops-competitions` 任务目录；算子实现以独立代码 PR
  提交至 [`cann/ops-sparse`](https://gitcode.com/cann/ops-sparse) 的 `master` 分支。

## 背景介绍

### aclsparseScatter 算子实现优化

本算子为 sparse 库新增算子（稀疏向量 scatter 到稠密向量），CANN 内置 TBE 算子库中
无对应实现，无历史 TBE 版本可作基线；本任务使用 Ascend C 编程语言完成实现与优化，
功能与性能以 GPU 侧 cuSPARSE scatter（`cusparseSpVec2Dn` 稀疏转稠密语义）及
PyTorch `Tensor.index_copy_` CUDA 实现作为对照标杆（任务书 3.3 定义性能倍率的标杆）。

#### 参考实现路径

- 语义参考：cuSPARSE `cusparseSpVec2Dn`（稀疏向量 → 稠密向量按索引写入）；
- 行为对照：PyTorch ATen `index_copy_` CUDA 实现（`aten/src/ATen/native/cuda/IndexKernel.cu`，
  重复索引语义为 last-write-wins，不累加）；
- 本仓 Ascend C 实现路径：`sparse/scatter/arch22/`（scatter.h / scatter_host.cpp /
  scatter_kernel.cpp / scatter_kernel.h）；
- 本仓 API 绑定路径：`python/aclsparse_torch/`（_lib.py ctypes 绑定 C API，
  _aten.py 适配 `Tensor.index_copy_(0, index, source)`，注册
  `torch.ops.ops_sparse_test.scatter_npu`）。

#### aclsparseScatter 算子实现现状分析

C API：`aclsparseScatter(handle, vecX, vecY)`，参数与支持能力如下：

| 参数 | 参数含义 | 数据类型 | 支持数据类型 | 约束 | 形状 |
| --- | --- | --- | --- | --- | --- |
| handle | 库句柄（含执行流） | aclsparseHandle_t | - | 须先 aclsparseSetStream | - |
| vecX | 输入稀疏向量描述符 | aclsparseConstSpVecDescr_t | values: int8, float16, bfloat16, float32, complex64；indices: int32（ACL_SPARSE_INDEX_32I） | idxBase 取 ZERO/ONE；nnz ≤ size；nnz ≤ UINT32_MAX | (size, nnz) |
| vecY | 输出稠密向量描述符 | aclsparseDnVecDescr_t | 同 vecX.values | nums ≥ vecX.size | (nums) |

计算公式：`Y[vecX.indices[i] - idxBase] = vecX.values[i]`（i ∈ [0, nnz)）

### aclsparseScatter 算子功能分析

算子功能：将稀疏向量的非零元按索引一次性写入稠密向量（纯数据搬移，不参与数值运算）。

- 输入：vecX（indices + values）、vecY（初始稠密向量）
- 输出：vecY（原地写）
- 支持数据类型：values 为 int8 / float16 / bfloat16 / float32 / complex64（complex64 按
  2×float = 8B 纯字节搬移）；indices 为 int32；idxBase 支持 0 基 / 1 基
- 重复索引语义：非确定性 last-write-wins（结果为该位置对应输入 values 之一，
  不累加），与 ATen `index_copy_` 对齐
- 不支持广播（稀疏/稠密向量逐位置写，无广播概念）

# 需求分析（required）

## 需求描述

使用 Ascend C 编程语言实现 aclsparseScatter 算子：支持 int8 / float16 / bfloat16 /
float32 / complex64 五种 value 类型、int32 索引、0/1 基索引；重复索引满足
last-write-wins 语义且不崩溃；性能不低于任务书规定倍率（每个有效性能 case ≥ 0.25×
GPU 标杆）；全流程禁 CPU fallback、不引入 workspace 与 Host 同步。

## 需求拆解

1. 支持 int8 / float16 / bfloat16 / float32 / complex64 数据类型（统一字节宽度搬运，
   bit-exact）
1. 支持 int32 索引与 ZERO/ONE 两种 idxBase
1. 重复索引 last-write-wins（输出 ∈ 对应输入 values 之一，不得实现为累加），必测不崩溃
1. 无重复索引时输出与 Golden bit-wise 一致；未写入位置保持初值；输入只读
1. 性能：P-01/02/03（词表索引场景）每个有效 case ≥ 0.25× GPU 标杆（预热 ≥10、
   采样 30，报告 median/p90；性能 case 不得使用重复索引）
1. 内存：不引入 workspace/preprocess，峰值内存不超过 GPU 基线 +50%（50% 规则）

## 支持矩阵

| 维度 | 支持范围 |
| --- | --- |
| value dtype | int8、float16、bfloat16、float32、complex64 |
| index type | ACL_SPARSE_INDEX_32I（int32） |
| index base | ZERO（0 基）、ONE（1 基） |
| shape | 一维连续 SpVec/DnVec，动态 size/nnz，`0 ≤ nnz ≤ size` |
| index pattern | 顺序、倒序、乱序、重复；性能用例不含重复索引 |
| 输出语义 | vecY 原地更新，未命中位置保持原值，vecX 只读 |
| workspace | 0；无 preprocess |
| Python/ATen | `Tensor.index_copy_(0, index, source)`，PrivateUse1 Dispatcher 注册 |

# 详细设计（required）

## 算子分析

### 数学公式

`Y[vecX.indices[i] - idxBase] = vecX.values[i]`，i ∈ [0, nnz)

- 无重复索引：逐位置唯一写，结果确定；
- 有重复索引：同一位置被多次写，最终值为写序最后一次（last-write-wins，非确定，
  属输入 values 集合之一）；
- nnz = 0：合法早退，Y 保持不变。

### 支持数据类型

- values：int8（1B）/ float16（2B）/ bfloat16（2B）/ float32（4B）/ complex64（8B）；
- indices：int32；idxBase：0 / 1

Scatter 不进行数值计算，所有 dtype 均按原始存储位搬运，NaN payload、±0、INF 等
特殊值 bit 级保留；complex64 的实虚部作为单个 8B 元素整体搬移，不允许撕裂。

### 支持形状

稀疏向量 (size, nnz) → 稠密向量 (nums)，nums ≥ size；一维向量语义，不支持广播。

## 算子实现

### 实现方案

总体结构：host 侧完成参数校验与 tiling 划分，kernel 侧按 tiling 下发的模式
（FAST / SAFE 双路径，v10）执行。全部 value dtype 统一为字节宽度 elemSize ∈
{1,2,2,4,8} 的纯字节搬移，kernel 不感知具体 dtype，保证 bit-exact。

```text
Tensor.index_copy_（Python，公开入口）
  └─ _aten.py 适配（PrivateUse1 Dispatcher kernel / Tensor shim 双机制）
       ├─ device/dim/shape/dtype/长度校验，不支持组合显式抛错（无 CPU fallback）
       └─ ctypes 绑定 C API（_lib.py）
            ↓
aclsparseScatter（公开 C++ API）
  ├─ Host：参数合同校验 → MapValueElemSize → FAST/SAFE 双路径 tiling
  └─ 异步下发 arch22 AIV Kernel（调用方 stream）
       ├─ CopyIn：idx 连续装载 + val 32B 槽位装载（MTE2）
       ├─ Compute：FAST 直写 / SAFE 输出区间命中扫描
       └─ CopyOut：DataCopyPad 直写 GM Y（MTE3）
            ↓
调用方 vecY 原地更新
```

#### 3.2.1 host 侧设计：

参数校验（scatter_host.cpp）：handle/vecX/vecY 非空；stream 非空；idxType 仅
ACL_SPARSE_INDEX_32I；valueType ∈ 支持集且 vecX/vecY 一致；idxBase ∈ {ZERO, ONE}；
nnz=0 早返 SUCCESS（此时 indices/values 允许为空）；nnz/nums 非负、size ≥ nnz、
nnz ≤ UINT32_MAX、indices/values/Y 指针非空；校验失败按错误码返回并 OP_LOGE，
不做任何 CPU fallback。

tiling 策略（v10 自适应双路径，按 nnz 选择执行模式）：

- 背景约束：按输入切核直写最快，但重复索引对跨核分裂时会并发 partial 写同一
  32B sector，实测偶发 L2 合并故障（"MTE accesses an invalid GM address"）；
  按输出切核单核独占绝对安全，但每核需全量扫描索引。任务书规定性能 case 不得
  含重复索引、功能场景重复索引只需"结果为输入值之一、不崩溃、不累加"，故按
  nnz 自适应：
  - SAFE 模式（nnz ≤ SCATTER_SAFE_MAX_NNZ=1024，覆盖全部小规模重复索引用例）：
    按输出 Y 区间切核。wantBlocks = ceil(yNums / SCATTER_CORE_MIN_SIZE=4096)，
    blockNum = min(wantBlocks, aivNum)，coreYn = ceil(yNums / blockNum)，
    coreNnz = 0；每核独占输出区间 [blockIdx·coreYn, min((blockIdx+1)·coreYn, yNums))。
  - FAST 模式（nnz > 1024，性能主路径）：按输入 nnz 切连续切片。
    wantBlocks = ceil(nnz / SCATTER_CORE_MIN_SLICE=128)，blockNum =
    min(wantBlocks, aivNum)，coreNnz = ceil(nnz / blockNum)，coreYn = 0；
    切片 ≥128 将重复索引跨核分裂概率压至 ~2/切片量级。
- aivNum 通过 `aclrtGetDeviceInfo(deviceId, ACL_DEV_ATTR_VECTOR_CORE_NUM)` 获取。
- tiling 数据结构 `ScatterTilingData`（无 64 核定长数组，瘦身为标量字段）：
  nnz / blockNum / mode / coreNnz / coreYn / tileNn / idxBase / elemSize / yNums。

##### 1. 分核策略：

优先使用满核原则：blockNum = min(ceil(总量/单核下限), aivNum)。

- 核间均分（ceil 划分后各核等量）：FAST 各核 coreNnz 等量，尾核数据不足时由
  `coreOffset_ + coreNnz_ > nnz` 截断；超出 blockNum 的核直接 inactive 返回。
- 不能均分时：ceil 分配天然将余量摊入各核（最后一核为余块），不单独区分大小核。
- SAFE 模式各核输出区间互不重叠（单核独占），从结构上消除跨核同地址写。

##### 2. 数据分块和内存优化策略：

充分使用 UB 空间并保证 MTE 对齐约束：

- UB 预算：idx tile 2048×4B = 8KB + val tile 2048×32B = 64KB，合计 72KB ≪ 192KB UB；
- tile 大小 SCATTER_TILE_NN=2048：P-01/02/03（nnz 4096~8192，每核切片
  ceil(nnz/blockNum) ≤ 2048）单核仅 1 个 tile，降低头开销；
- val 按"每元素 1 个 32B 槽位"装载（MTE2 `DataCopyPad` blockCount=n、blockLen=
  elemSize 逐 block 补齐 32B）：MTE3 的 UB 源必须 32B 对齐（ONE_BLK_SIZE），
  直写循环从槽位取源，写出侧无 V 指令、无 staging、无逐写 barrier（实测
  MTE3 散布小写 ~0.47μs/次、槽位装载 ~75ns/元素）；
- 单缓冲 SCATTER_BUFFER_NUM=1 + 每 tile 一次 `PipeBarrier<PIPE_MTE3>`：MTE3
  读源排空后才允许下一 tile 的 MTE2 覆写，从根上消除缓冲复用竞争；
- 无 workspace、无 Host 同步，沿用调用方 stream 异步执行。

##### 3. tilingkey 规划策略：

本实现将路径选择经 `ScatterTilingData.mode` 字段下发（SCATTER_MODE_FAST=0 /
SCATTER_MODE_SAFE=1），作用等价于 tiling key：kernel 侧据其走不同分支，host 侧
感知 nnz 与 yNums 完成决策。由于 kernel 全程按字节宽度搬运、不感知 dtype，
无需按数据类型规划 tiling key；重复索引是否存在无法在 host 侧 O(1) 判定
（索引在 device 上），故模式只按 nnz 划分而非按"有无重复"划分。

#### 3.2.2 kernel 侧设计：

`KernelScatterDual`（scatter_kernel.cpp）分 Init 与 Process 两阶段，Process 含
搬入（CopyIn / LoadTile）、计算（Compute / ScatterTileFast|Safe）、搬出
（CopyOut / DataCopyPad 直写 GM）三段；入口 `scatter_custom`，
`KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY)`。

1. 字节视角统一：所有 value dtype 以 elemSize 字节搬移，complex64 按 8B
   （2×float），MTE2 槽位装载 + MTE3 直写全程不解释数据，bit-exact；
2. CopyIn（LoadTile，FAST/SAFE 共用）：idx 连续装载（blockCount=1，n×4B）；
   val 槽位装载（blockCount=n，blockLen=es，逐元素补齐 32B 对齐槽）；
3. FAST 计算段：逐元素读 idx（raw `__ubuf__` 指针访问，规避 GetValue 抽象层
   ~113ns/元素开销，实测 raw 读 ~32ns/元素），逻辑索引 = raw − idxBase，
   `DataCopyPad(yGm[idx*es], valLocal[i*32], {1, es})` 直写；
4. SAFE 计算段：每核扫描全量 tile（64bit 成对读 + 4 对展开），仅写本核输出区间
   [yStart+idxBase, yStop+idxBase) 内的命中（比较在 raw 域、写偏移用逻辑域）；
   单核独占输出区间 ⇒ 跨核无同地址写，重复索引在核内按扫描序 last-write-wins
   （确定性）；奇数尾块走单元素路径（规避把 32bit 索引符号扩展为 64bit 后
   高位字被填 0/全 1 而产生的伪命中，该缺陷曾致 base=0 小规模用例首元素被
   误写，已修复并用 UT 覆盖）；
5. 按 tiling.mode 分支执行 ProcessFast / ProcessSafe；nnz=0 或区间为空的核
   inactive 直接返回。

数据流（文字流程图）：

```text
host: 校验 → MapValueElemSize → 双路径 tiling → scatter_kernel_do(stream)
kernel(FAST, nnz>1024): GM idx/val --MTE2(连续/32B槽位)--> UB
                        --> raw 标量读 idx --MTE3(1×es 直写)--> GM Y
kernel(SAFE, nnz<=1024): GM idx/val(全量) --MTE2--> UB
                        --> 输出区间命中扫描(64bit成对+展开) --MTE3--> GM Y(本核区间)
每 tile: WaitCopyIn → ScatterTile → PipeBarrier<MTE3> → FreeTile → 预取下一 tile
```

### Python/ATen 适配设计

公开入口 `Tensor.index_copy_(0, index, source)`，映射 `aten::index_copy_`，实现于
`python/aclsparse_torch/_aten.py`，采用双注册机制：

1. **PrivateUse1 Dispatcher kernel**（正式集成机制）：
   `lib.impl("index_copy_", index_copy_, "PrivateUse1")` 注册 ATen kernel；
2. **Tensor shim**（NPU-only 补丁，默认 `ACLSPARSE_PATCH_TENSOR_METHOD=1`）：
   包装 `torch.Tensor.index_copy_`，NPU tensor 走本实现、CPU tensor 透传原生实现。
   因 torch_npu 已以更高优先级抢占原生 NPU `index_copy_`，双机制保证本适配在
   现网 torch_npu 环境下确定生效。

适配层校验链（任一不满足显式抛 RuntimeError，无 CPU fallback）：

| 校验项 | 要求 |
| --- | --- |
| device | self/index/source 同一 NPU Device |
| dim | 仅 dim=0（index_copy_ 0-based 语义） |
| shape | 三者一维；index 与 source 长度一致；self.numel 为逻辑规模 |
| dtype | self/source ∈ 五种支持 dtype 且一致；index 为 int64/int32 |
| fallback | source 必须为 NPU tensor，禁止 CPU 回退 |

执行链：`self/source/index` 组装 SpVec（values+indices）与 DnVec（self）描述符 →
ctypes 调用 `aclsparseScatter` → 原地返回同一 `self`（保持原位语义与
strides/offset）。索引统一按 0 基逻辑索引处理；描述符生命周期由 Python 层 RAII
管理，Device 数据生命周期由 stream 语义保证，不做 Host synchronize。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas 800I/T A2（910B3/910B4，arch22；A3 型号同源适配，性能自验证优先 910B3） | √ |

## 算子约束限制

- 不支持广播；
- 索引仅支持 int32（ACL_SPARSE_INDEX_32I），I64 暂不支持；
- nnz ≤ UINT32_MAX；
- 重复索引结果非确定（last-write-wins，任一输入值），不保证核间写序；
  nnz > 1024 且重复索引跨核分裂时依赖切片下限（≥128）压低并发写冲突概率，
  极端构造下存在残余风险（任务书性能/功能用例已全部覆盖验证）；
- values 与 Y 的 dtype 必须一致。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | 无重复索引位置与 CPU Golden bit-wise 一致；重复索引位置输出 ∈ 对应输入 values（不累加）；未写位置保持初值；输入 indices/values bit 级只读。实测 200/200 通过（含重复索引 100 条、5 dtype × base0/1 全覆盖） | 任务书 3.2 |
| 性能标准 | P-01/02/03 每个有效 case ≥ 0.25× GPU 标杆，口径：GPU Event median / NPU 同调用范围 Kernel 总耗时 median，预热 10 采样 30。实测 P-01 0.475~0.519×、P-02 0.703~0.729×、P-03 0.519~0.538×，30/30 PASS | 任务书 3.3 |
| 内存标准 | 无 workspace/preprocess；NPU 峰值不超 GPU 基线 +50%。实测 230 case 中 225 例低于 GPU 峰值，最大超出 12.5%，workspace=0 | 任务书 3.4 |

可维测配套：

- UT：C++ 设备侧用例 88 项（build/test/scatter/scatter_test）；ATen 适配 16 项；
  端到端 18 项，全部通过；
- 自测工具链（test_cases/aclsparseScatter_testCase/）：run_accuracy_standalone.py
  （CPU Golden 判定）、benchmark_npu_kernel_time.py（kernel 口径）、
  benchmark_sparse_ops_npu.py（Event 口径）、collect_sparse_ops_npu_memory.py
  （峰值内存）、compare_perf_baseline.py / compare_memory_standalone.py（倍率对比）、
  make_selfcheck_evidence.py（自检报告与截图生成）；
- 证据归档：截图 `pic_my/`（00~08 汇总 + 09~12 逐 dtype），明细
  `test_cases/aclsparseScatter_testCase/results/`，报告
  `test_cases/aclsparseScatter_testCase/selftest_report.md`；
- 可维护性：host 校验失败路径统一 OP_LOGE（含参数值）并返回
  ACL_SPARSE_STATUS_INVALID_VALUE / NOT_SUPPORTED 等语义化错误码；kernel 常量
  （模式、tile、切片下限等）集中在 scatter.h，调参不散落。

## 兼容性分析

sparse 库新增算子，无存量版本，不涉及兼容性分析；对外仅暴露 C API
（aclsparseScatter）与既有关键字/描述符体系（aclsparseSpVecDescr /
aclsparseDnVecDescr / aclsparseHandle）复用，ABI 稳定；Python 侧
`torch.ops.ops_sparse_test.scatter_npu` 与 `Tensor.index_copy_` 适配层为增量注册。

# 参考资料

1. [CANN 社区任务 2026 算子设计文档模板](https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md)
2. [CANN 社区任务 2026 提交流程](https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/README.md)
3. [ops-sparse 代码仓](https://gitcode.com/cann/ops-sparse)
4. [生态算子开源精度标准](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md)
5. [cuSPARSE Scatter 官方文档](https://docs.nvidia.com/cuda/cusparse/)
6. [PyTorch Tensor.index_copy_](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.index_copy_.html)
