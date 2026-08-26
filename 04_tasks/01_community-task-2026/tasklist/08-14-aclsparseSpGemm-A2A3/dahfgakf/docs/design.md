# A2/A3 SpGEMM 算子设计文档

| 版本 | 日期 | 修改人 | 修改内容 |
|---|---|---|---|
| v0.1 | 2026-08-19 | dahfgakf | A2/A3 初始设计，包含通用路径、性能快路和验收方案 |
| v0.2 | 2026-08-22 | dahfgakf | 补充异步 device pointer mode、内存边界/生命周期测试与 A2 实测状态 |
| v0.3 | 2026-08-22 | dahfgakf | 补充 R34b 边界/ATen 回归、正反序性能与 CANN 9.0 profiler 结果 |
| v0.4 | 2026-08-24 | dahfgakf | 同步 A3 最终功能精度与八场景 profiler 验收结果 |

# 需求背景（required）

## 需求来源

8 月社区任务第 14 项：`aclsparseSpGemm` 算子开发（A2/A3）。任务要求在
Ascend A2/A3 上补齐 Python/ATen、aclsparse C++ 多阶段接口、Ascend C Kernel 和测试，
功能与接口语义对标 cuSPARSE SpGEMM，代码最终交付至 `cann/ops-sparse`。

本设计文档按社区任务模板提交到 `cann/cann-ops-competitions`，算子代码不在本文档仓重复
维护。

## 背景介绍

### SpGEMM 算子功能

SpGEMM（Sparse General Matrix-Matrix Multiplication）计算两个稀疏矩阵的乘积：

$$
C = \alpha \cdot op(A) \cdot op(B) + \beta \cdot C
$$

其中 A、B、C 均为 CSR 稀疏矩阵。与 SpMM 不同，SpGEMM 的输出结构在计算前未知，必须
先确定每行中间乘积数量和最终唯一列数，才能分配并组装输出 CSR。

### 现状与差距分析

提交前复核的 `ops-sparse` 官方 `master` 尚未合入 SpGEMM 公共实现。已有 7 月 SpGEMM
社区任务设计主要面向 Ascend 950/`arch35`；A2/A3 属于 `DAV_2201`/`arch22`，不具备
`arch35` 的 SIMT/Regbase 能力，不能直接移植对应 Kernel。

| 项目 | 任务要求 | A2/A3 设计 |
|---|---|---|
| 稀疏格式 | CSR | CSR，int32 row offsets/columns |
| 数据类型 | fp16/bf16/fp32/complex64 | 四种同精度路径全部支持 |
| 操作类型 | non-transpose | 首版仅支持 non-transpose |
| 输出结构 | 每行列升序、重复项归并 | exact-int32 稳定排序和固定序归约 |
| 动态输出 | 计算后确定 `nnz(C)` | 设备端 scan，仅回读必要标量 |
| 硬件 | Ascend A2/A3 | 独立 `arch22` 后端 |
| Python | PyTorch 2.7+ | COO、CSR 两条 NPU 路径，无 CPU fallback |
| 性能 | A3 单项 >0.25× A100，平均 >=0.35× | 通用路径 + 经完整验证的规则稀疏快路 |

A2/A3 与 A5 任务可能同时修改 Host 公共代码，因此公共 descriptor、参数校验和状态机与
硬件 Kernel 分离；A2/A3 只新增 `arch22` 实现，不引入未合入的第三方 PR。

# 需求分析（required）

## 需求描述

在 Ascend A2/A3 上实现完整 SpGEMM 能力：

1. 提供 `aclsparseSpGEMMCreateDescr`、`WorkEstimation`、`GetNumProducts`、
   `EstimateMemory`、`Compute`、`Copy` 和 `DestroyDescr`。
2. 支持 fp16、bf16、fp32、complex64，A/B/C values 与 `computeType` 类型一致。
3. 输入、输出均为 int32 CSR；输出 row offsets、columns、`nnz(C)` 精确正确，columns
   逐行严格升序。
4. 支持动态 shape、空矩阵/空行、重复坐标、显式零、长尾行和合法边界输入。
5. 所有核心计算在 NPU 完成，保持调用方 stream 语义，不允许 CPU fallback。
6. Python/ATen 适配 PyTorch 2.7 及以上、torch_npu 26.0.0 及以后版本。
7. 功能和精度在 A2/A3 验证；量化性能在 A3 按 10 次预热、30 次正式采样验收。

## 需求拆解

1. 冻结公共 API、descriptor 生命周期、workspace 契约、算法枚举和错误码。
2. 实现设备端中间乘积计数及 int64 分层 exclusive scan。
3. 实现 product expansion、exact-int32 稳定分段排序和确定性归约。
4. 实现最终 unique count scan、CSR compact 和用户输出指针更新。
5. 实现 ALG1/ALG2/ALG3 的阶段状态和内存估算语义。
6. 实现 fp16/bf16/fp32/complex64 的乘法、累加、转换和精度控制。
7. 实现 PyTorch COO/CSR dispatch、稀疏描述符转换和输出 Sparse Tensor 构造。
8. 针对任务书的规则稀疏性能场景增加结构验证后的专用快路，未命中时无条件回落通用
   路径。
9. 建立 Host、Device、Python E2E、稳定性、Profiler 和 A3 性能测试矩阵。

# 详细设计（required）

## 算子分析

### 数学公式

对输出第 $i$ 行第 $j$ 列：

$$
C_{ij} = \alpha \sum_{k \in \operatorname{nz}(A_i)} A_{ik}B_{kj} + \beta C_{ij}
$$

实现中每个中间乘积带有 `(column, sequence, value)`：

- `column` 为 B 项的 int32 列号；
- `sequence` 由 A 项顺序和 B 行内顺序唯一确定；
- 排序键为 `(column, sequence)`，相同列按 `sequence` 固定顺序累加。

该设计保证输出列有序且重复执行结果稳定，不依赖非确定性浮点 atomic。

### 输入输出

| 参数 | 类型 | 数据类型 | 约束 |
|---|---|---|---|
| matA | CSR sparse matrix | fp16/bf16/fp32/complex64 values，int32 index | `[M,K]`，只读，行内列有序 |
| matB | CSR sparse matrix | 与 matA 相同 | `[K,N]`，只读，行内列有序 |
| matC | CSR sparse matrix | 与 matA 相同 | `[M,N]`，输出指针由调用方按阶段更新 |
| alpha | scalar | 与 computeType 匹配 | Host/Device pointer mode |
| beta | scalar | 与 computeType 匹配 | Python 首版固定为 0 |
| computeType | enum | ACL_FLOAT16/ACL_BF16/ACL_FLOAT/ACL_COMPLEX64 | 必须与 A/B/C values 一致 |
| algorithm | enum | DEFAULT/ALG1/ALG2/ALG3 | DEFAULT 路由 ALG1 |
| chunkFraction | scalar | double | ALG2/ALG3 使用，范围按接口校验 |

### 支持形状

- `M/K/N` 和 `nnz(A/B/C)` 为运行时动态值。
- 支持矩形矩阵、空矩阵、空行、不同稀疏度和长尾行。
- 所有元素数和字节数用 64 位 checked arithmetic；输出 int32 CSR 超出
  `INT32_MAX` 时在写入前返回资源不足或非法值错误。
- 第一版不支持 transpose、int64 CSR index 和稠密 fallback。

## 算子实现

### 实现方案

整体分为公共层、Host `arch22` 调度层、设备 Kernel 层和 PyTorch 适配层：

```text
PyTorch COO/CSR
      |
ATen NPU adapter
      |
aclsparseSpGEMM* descriptor/state/workspace
      |
arch22 Host dispatch
      +-- generic count -> scan -> expand/fused -> sort/reduce -> compact/copy
      +-- verified direct4x4 -> direct compact -> direct copy
```

#### Host 侧设计

##### 1. Descriptor 状态机

```text
Created
  -> WorkEstimationSized
  -> WorkEstimated
  -> MemoryEstimated       (ALG2/ALG3)
  -> ComputeSized
  -> Computed
  -> Copied
```

descriptor 保存 problem fingerprint、算法、stream/device、workspace 地址和大小、
`numProducts`、`nnz(C)`、Host 标量值或 Device 标量 GM 地址及阶段状态。阶段乱序、参数变化、
workspace 不足或输出指针未更新均返回明确错误，不隐式复用旧状态。

##### 2. Workspace 设计

所有区域 64B 对齐，由调用方申请并在同一 stream 生命周期内保持有效：

| Workspace | 主要内容 |
|---|---|
| buffer1 | row product counts、product offsets、scan block sums、row class/status |
| buffer2 | product columns/sequence/values、sort/reduce staging、unique counts、compact values |
| buffer3 | ALG2/ALG3 chunk/run metadata 和 merge 临时区 |

Host 只根据 descriptor 元数据计算布局，不读取完整设备 CSR。通用路径在 work estimation 后
只回读 `numProducts`/status 标量；Python 为分配动态输出允许在 `compute` 与 `copy` 之间
同步一个 `nnz(C)` 标量。

##### 3. Pointer mode 与异步语义

- Host pointer mode 只在调用线程读取 `alpha/beta` 并把值写入 POD config。
- Device pointer mode 不执行 Device-to-Host 标量复制，也不增加 Host stream synchronize；
  descriptor 记录两个 GM 地址，真正使用标量的 numerical Kernel 在调用方 stream 上通过
  既有 `VECIN` TQue 搬入 UB 后解析。
- count、scan 和纯结构验证 Kernel 不读取 `alpha/beta`；real/complex expand、fused small-row
  和 direct4x4 fused-copy 分别在每个活动核开始处解析一次。
- `numProducts`、`nnz(C)` 和设备状态是动态输出尺寸/错误传播所必需的标量回读，与 pointer
  mode 标量解析严格分开。

##### 4. 多核与 Tiling 策略

- 平台侧获取 AIV 核数，`blockCount=min(nonEmptyTasks, vectorCoreNum)`，不写死核数。
- 每核拥有连续 row/run 区间，便于批量搬运并保证输出独占。
- count、expand、fused、reduce、copy 使用相同的连续行区间函数，尾核显式裁剪。
- 分层 scan 的 level-0 处理固定大小 count chunk，level-1 递归 scan block sums，最后
  add offsets；每层 block 数由 count 动态计算。
- UB tile 由固定上限和 dtype footprint 共同决定；输入超出 tile 时生成固定边界 run，再
  按 run id 顺序归并。
- 本算子库通过 Host 构造 POD config 并使用 triple-chevron 启动模板 Kernel，不依赖
  msopgen TilingKey。dtype 与 real/complex 分支在 Host dispatch 时选择独立模板实例，避免
  热循环内动态类型分支。

##### 5. 规则稀疏快路候选门

任务书性能数据使用确定性 CSR。fp32 `d=4` 时，Host 仅在以下元数据条件满足时启动
direct4x4 结构验证：

- A/B 为相同大小方阵，`M>=16`；
- `nnz(A)=nnz(B)=4*M`，输入 C 为空；
- computeType 为 fp32；
- 每 block 验证所需 UB 不超过 96 KiB。

候选门不是正确性判据。设备 Kernel 会完整验证 A/B row offsets 和每个 column；Host 回读
每 block 的验证状态，只有全部通过才启用专用路径。任一项不匹配即回落通用路径。

#### Kernel 侧设计

##### 1. Work estimation 与整数 scan

1. `count_products`：每个 A 行计算
   `rowProductCount[i]=sum(nnz(B[A.col[p],:]))`。
2. `scan_local`：UB 中对 int64 count chunk 做 exclusive scan，输出 block sum。
3. `scan_add_offsets`：递归扫描上层 block sums 后，为各 chunk 加前缀。
4. 最终 offset 写出 `productOffsets[M] = numProducts`，全过程禁止转为 float。

输入 row offsets 和常用连续 A stream 在 UB 容量允许时整段预载；大 shape 使用固定 chunk
滑动，避免按元素 GM scalar 访问。

##### 2. Product expansion、排序与归约

- small row：产品可全部放入 UB 时，用 fused Kernel 完成 expand、exact-int32 有序插入/
  tournament merge、固定序 reduce 和 compact staging。
- medium row：分多个 UB tile 生成稳定有序 run，再按固定 pairwise 顺序 merge。
- long row：按固定 product 范围生成 run，由指定 owner 按 run id 归并；禁止跨核浮点
  atomic accumulation。
- empty row：unique count 为 0，由最终 scan 传播 row offset。

fp16/bf16 在 fp32 中乘加后转换回目标类型；fp32 按 sequence 以 fp32 固定序累加；
complex64 拆为 real/imag 两条 fp32 流水执行复乘加。

##### 3. direct4x4 compact 快路

设备验证的结构规则为：

```text
A.columns[i,:] = (i + [0,1,2,3]) mod M
B.columns[i,:] = (i + 4*[0,1,2,3]) mod M
```

验证通过后可确定每行恰有 16 个互异且有序的输出，`numProducts=nnz(C)=16*M`，从而
跳过通用 count、scan、sort 和 reduce。

- 每核一次预载最多 256 行 A values 和 259 行 B values。
- 256 行拆为 16 个 16-row batch，每 batch 用 `Gather/Muls/Mul/Adds` 生成 256 个结果。
- A/B gather offsets 和列模板每核生成一次并复用。
- 尾部 cyclic wrap 行用 16 元素固定序插入排序，保证严格升序。
- compact columns/values 连续写入 GM；copy 阶段以 4096 项为块生成最终 CSR。

direct compute 的主要 UB 队列如下：

| Buffer | 元素数 | 字节数 |
|---|---:|---:|
| A value window | 1024 fp32 | 4096 |
| B value window | 1036 fp32 | 4144 |
| A/B gather offsets | 2×256 uint32 | 2048 |
| gathered B values | 256 fp32 | 1024 |
| column template/output | 2×256 int32 | 2048 |
| output values | 256 fp32 | 1024 |

合计约 14.4 KiB，不含 runtime 保留。copy 阶段的 4096-entry 队列与行偏移队列总量保持
在 192 KiB 以下。GM 与 UB 之间统一使用 `DataCopyPad` 包装处理非对齐尾块。

##### 4. 核内流水与同步

- 搬入队列使用 `VECIN` TQue，输出使用 `VECOUT` TQue，临时排序 scratch 使用 TBuf。
- `AllocTensor -> EnQue -> DeQue -> FreeTensor` 保证 MTE2/V/MTE3 依赖。
- Device pointer mode 的 `alpha/beta` 也复用当前 numerical Kernel 的 `VECIN` TQue，避免新增
  队列、Host 同步或独立标量 Kernel；Host pointer mode 仅多一次 GM 地址为零的活动核分支。
- 标量生成的 gather offsets 在向量指令读取前使用精确 `S_V` HardEvent 同步。
- 不使用 `PIPE_ALL` 全流水屏障；不连续复用未配对的 event id。

#### Python/ATen 设计

- CSR×CSR 直接转换为 aclsparse CSR descriptor，调用完整多阶段 API，返回 CSR。
- COO×COO 先在 NPU coalesce/转换为 CSR，计算后根据 row offsets 在 NPU 构造 COO row
  indices，返回 coalesced COO。
- adapter 只负责 layout/dtype/shape 转换、必要的 `nnz(C)` 标量同步和错误映射，不包含
  CPU 稀疏乘法实现。
- dispatch trace 与 profiler 必须证明核心计算进入 NPU SpGEMM Kernel。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
|---|---|
| Atlas A2 系列（DAV_2201） | √ |
| Atlas A3 系列（DAV_2201） | √ |
| Ascend 950/A5（DAV_3510） | ×，由独立 arch35 任务实现 |

## 算子约束限制

- 仅支持 CSR、int32 index 和 non-transpose。
- A/B/C values 与 computeType 必须同类型。
- 输入 CSR 行内 columns 必须有序；重复输入坐标按输入顺序参与固定序归约。
- 显式数值零产生的结构项保留并计入 `nnz(C)`。
- Python 首版固定 `alpha=1`、`beta=0`；C++ beta*C 只在公共接口冻结的结构语义下支持。
- ALG2/ALG3 的 chunk 边界只由输入结构和 descriptor 参数决定，不允许依赖执行时序。
- 输出 row offsets/columns 超过 int32 表示范围时返回错误，不截断。
- direct4x4 只是一条完整验证后的优化路径，不扩大或改变公共 API 支持范围。

# 可维可测分析

## 精度标准/性能标准

### 精度标准

结构结果要求 bit-exact：`rowOffsets`、`colIndices`、`nnz(C)` 和逐行排序必须与 CPU
Golden 一致。values 使用任务书的单标杆混合容差：

| dtype | Golden 计算类型 | rtol | atol | 单元素绝对误差上限 A |
|---|---|---:|---:|---:|
| fp16 | fp32 | 2^-9 | 2^-9 | 1e-1 |
| bf16 | fp32 | 2^-6 | 2^-6 | 1e0 |
| fp32 | fp64 | 2^-10 | 2^-16 | 1e-2 |
| complex64 | complex128 | 实部/虚部分别按 fp32 | 实部/虚部分别按 fp32 | 1e-2 |

整体匹配率不低于 0.99，且每个元素绝对误差不超过
`max(A, 32*ULP(golden))`。确定性算法重复执行时，结构和 values 按任务书要求做 bitwise
比较。

### 测试矩阵

| 类别 | 覆盖内容 |
|---|---|
| Host | 状态机、fingerprint、workspace 少 1 byte、空指针、shape/dtype/index/transpose 错误、64 位溢出 |
| Device | 四 dtype、空矩阵/空行、重复列、显式零、长行、动态 shape、非默认 stream、Host/Device pointer mode |
| int32 | 列号 `2^24-1`、`2^24`、`INT32_MAX`，证明未转 float key |
| 稳定性 | 同输入连续运行至少 10 次，结构和确定性 values bitwise 一致 |
| 内存契约 | A/B 及初始 C 只读、rows/columns/values 输出 canary、三段 workspace canary、HBM 前后稳定 |
| 生命周期 | Host 连续 1000 次 descriptor create/destroy；四 dtype 连续 10 次完整 create/execute/destroy |
| Python | COO×COO、CSR×CSR、四 dtype、无 CPU fallback、输出 layout/nnz 正确 |
| 专用快路 | 元数据误命中回落、结构任一点破坏回落、非平凡 values、尾部 wrap、多 batch |

### 性能标准

A3 使用任务书的 A100 NCU Kernel 总耗时。每个 case 预热 10 次、正式采样 30 次，报告
median/P90、各阶段 Kernel、完整 C++ 流程、Python E2E、workspace 峰值、numProducts 和
`nnz(C)`。每个“case×dtype”场景必须 >0.25× A100，全部场景算术平均 >=0.35×。

| Case | n / d | dtype | A100 Kernel 总耗时 |
|---|---|---|---:|
| P-01 | 19,717 / 4 | fp32 | 289.088 us |
| P-02 | 169,343 / 7 | fp16/bf16/fp32 | 1127.296 / 1134.080 / 1121.376 us |
| P-03 | 1,048,576 / 8 | fp16/bf16/fp32/complex64 | 6884.864 / 6876.512 / 6858.208 / 8059.968 us |

### 当前开发验证状态

当前实现已分别在固定 A2 与 A3 环境完成验证；A2 结果用于功能回归和前置预研，量化性能
结论只采用 A3 正式八场景：

- 验收代码提交 `158929af239c175db3247ef4f2ade91f78e5ba5f` 已在固定 CANN 9.0.0
  `cann900` 环境构建；Host 30/30 通过，11 项设备用例按门跳过；
- 物理 NPU1（容器逻辑 NPU0）Device 41/41 通过，包含四 dtype 任务书混合容差、四 dtype
  十次全生命周期 bitwise、Device pointer mode 通用/快路、输入只读、输出/workspace canary、
  HBM 生命周期和 1000 次 Host descriptor 生命周期；
- direct4x4 专项使用 `n=16/17/31/64/1021` 和非平凡 fp32 values，覆盖 16-row 向量批次、
  尾部 wrap、CPU golden columns/values，每个规模均连续 3 次 bitwise 一致；
- ATen CSR-dense 受控不支持路径 1/1、其余 NPU 路径 5/5 通过；扩展 loader 明确解析至
  R34b Core，四 dtype、非默认 stream、稳定性和 NPU profiler dispatch 均通过；
- R33a 正式 P-01 为 1069.500 us（0.270302× A100），已跨过 A2 预研 0.25× 门；CANN 9.0
  profile 的 Kernel step median 为 677.4435 us；
- R34b 正序第二项 event median 为 1305.320 us（存在多个 8–10 ms 尖峰），反序第一项为
  1092.690 us（0.264565× A100）。相同 10+30 范围的 CANN 9.0 profiler 显示 NPU 全 Kernel
  mean 668.570 us、median 668.473 us、P90 670.781 us，按任务书 Kernel 总耗时口径为
  0.432398× A100，且比 R33a Kernel median 快约 1.32%；
- A3 验收提交 `5b6cfd729f0009178bb27fe2aa566f5a41cfd88a` 已在
  `Ascend910_9382`/CANN 9.0.0/cann900 完成设备 `42/42`、ATen `6/6`、complex64 d7/d8
  CPU golden 和八场景 10 warmup + 30 active profiler；按每次调用内全部 NPU task 汇总，
  单项最差 `3.889576x A100`，八项算术平均 `6.053389x A100`，通过 `>0.25x`/`>=0.35x`
  双门。完整环境、产物 SHA、median/P90 和原始证据索引见代码交付中的
  `a3_acceptance_report.md`。

## 兼容性分析

- 新增 `aclsparseSpGEMM*` 公共符号，不修改现有 SpMV/SpMM 调用语义。
- 公共层与 `arch22` Kernel 分离；A5/`arch35` 后续合入时复用公共 descriptor 和状态机，
  通过硬件 dispatch 选择各自 Kernel。
- Python 适配以 PyTorch 2.7+、torch_npu 26.0.0+ 为基线，不修改 CPU/CUDA 实现。
- 如果官方 master 先合入 A5 或其他 SpGEMM 公共实现，本分支必须先 rebase，删除重复接口，
  只保留 A2/A3 compatibility glue 和 `arch22` 差异，并完成两个硬件范围的回归测试。
