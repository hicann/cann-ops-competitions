# 需求背景（required）

## 需求来源

社区任务：`aclsparseSpGEMM` 算子开发（Atlas A2 / Atlas A3）。目标是复用
SpGEMM 社区任务规划的 `aclsparseSpGEMM*` 多阶段接口，在 DAV-2201 上补齐
float16、bfloat16、float32、complex64，并将 PyTorch 2.7+
`aten::_sparse_sparse_matmul` 接入 torch_npu 26.0.0+。

## 背景介绍

### SpGEMM 算子实现

计算公式为：

```text
C = alpha * A * B + beta * C
```

其中 `A[M,K]`、`B[K,N]` 和 `C[M,N]` 均为 zero-based CSR。C++ 接口的
调用阶段、workspace 与错误处理对齐 cuSPARSE SpGEMM 的多阶段模型；Python
公开入口为 `torch.sparse.mm(mat1, mat2)`，内部命中
`aten::_sparse_sparse_matmul`。

并行 A5 社区任务方案基于 DAV-3510 SIMT，符号分析路径与 A2/A3 不同。本设计
新增独立 `arch22` 路径：输入校验、工作量估算、符号分析、输出组装和数值计算
均在 NPU 上完成，Host 只在 API 明确要求查询中间乘积数或 `nnz(C)` 时读取
小型结果标量。

### 对标与差异

| 项目 | 目标规格 | A2/A3 设计 |
|---|---|---|
| 调用阶段 | WorkEstimation → EstimateMemory → Compute → Copy | 保持同名、同签名；ALG2/ALG3 执行 EstimateMemory，其他算法也兼容该阶段 |
| 稀疏格式 | CSR | zero-based CSR |
| 索引 | int32 | rowOffsets / colIndices 均为 int32 |
| operation | NON_TRANSPOSE | 仅 NON_TRANSPOSE，其他操作返回 NOT_SUPPORTED |
| values | fp16 / bf16 / fp32 / complex64 | 四种同精度全链路 |
| 输出 | canonical CSR | 严格升序、重复合并、显式数值零保留 |
| 异步 | 调用方 stream | Compute / Copy 无无条件 Host 同步 |
| 确定性 | 算法确定性 | 互斥输出分段、固定 k 路归并和固定求和顺序 |

# 需求分析（required）

## 需求描述

1. 在 Atlas A2（Ascend 910B）和 Atlas A3（Ascend 910_93）实现 CSR
   SpGEMM Ascend C Kernel。
2. 复用并补齐公开 `aclsparseSpGEMM*` 多阶段接口，不新增重复接口。
3. 支持 `ACL_FLOAT16`、`ACL_BF16`、`ACL_FLOAT`、`ACL_COMPLEX64`，且
   A/B/C/computeType 一致。
4. 注册 PyTorch 2.7+ `aten::_sparse_sparse_matmul` 的 NPU sparse dispatch，
   支持二维 COO/CSR 输入并在 NPU 上完成 CSR 转换和输出构造。
5. 对输出 `nnz`、rowOffsets、colIndices、values、确定性、workspace 边界、
   错误码和输入只读性进行验证；核心计算不允许 CPU fallback。

## 需求拆解

- 公共层：算法枚举、不透明描述符、七个公开 API、动态 nnz 查询状态。
- Host 层：参数与阶段状态机、workspace 计算、stream 与 pointer mode、Kernel
  launch、动态输出指针绑定。
- Device 层：CSR 校验、中间乘积计数、符号 k 路归并、前缀和、数值归并。
- PyTorch 层：dispatch 注册、layout/dtype 转换、描述符构造、输出 Sparse
  Tensor 构造和 task queue / ACL stream 桥接。
- 测试层：C++ 多阶段 UT、任务包 ATen/ST、A2/A3 精度、A3 Profiler 性能。

# 详细设计（required）

## 算子分析

### 数学公式

对于输出行 `i` 和列 `j`：

```text
C[i,j] = alpha * sum_k(A[i,k] * B[k,j]) + beta * C_in[i,j]
```

输出结构是 `A*B` 的结构并集；`beta != 0` 时再与输入 C 的结构取并集。同一
坐标的所有乘积按确定顺序累加为一个条目。即使最终 values 为数值零，该结构
条目仍保留。

### 支持数据类型

| 存储类型 | Kernel 累加 | 输出转换 |
|---|---|---|
| float16 | float32 | float16 |
| bfloat16 | float32 | RNE bfloat16 |
| float32 | float32 补偿求和 | float32 |
| complex64 | 实/虚 float32 补偿求和 | complex64 |

Host / Device pointer mode 均支持，`alpha`、`beta` 的类型必须与
`computeType` 一致。complex64 使用 `{real, imag}` 二元 float 布局。

### 支持形状与边界

- `M`、`K`、`N`、输入 nnz、输出 nnz 动态变化，均受 int32 CSR 上限约束。
- 支持零维、零 nnz、空行、空列、无交集乘积、长尾行和矩形矩阵。
- 输入每行列索引必须严格升序且无重复；Device Work 阶段验证行偏移、末端
  nnz、列范围和排序。
- 输出前缀和超过 `INT32_MAX` 时返回资源不足，不产生部分有效输出。

## 算子实现

### 实现方案

### 总体架构

```text
torch.sparse.mm
  └─ aten::_sparse_sparse_matmul (SparsePrivateUse1 / SparseCsrPrivateUse1)
      └─ COO/CSR → int32 CSR (NPU)
          └─ aclsparseSpGEMM* multi-stage API
              ├─ arch22 Host state machine
              └─ DAV-2201 Ascend C kernels
                  ├─ work + reduce
                  ├─ symbolic + prefix
                  └─ copy / numeric merge
```

A2、A3 均通过根 CMake 的 SoC 映射选择 `arch22`。A5 保留 `arch35`，公共
接口和矩阵描述符位于架构无关目录，避免两个任务在主干上互相覆盖硬件实现。

### Tiling 与 Kernel 参数

本算子采用 Ascend C `<<<>>>` 直接调用，不经过算子注册框架生成 `TilingData`。
Host 将动态参数和 workspace 偏移分别封装为 `SpgemmArch22Params`、
`SpgemmArch22WorkLayout` 和 `SpgemmArch22ComputeLayout`，按值传入 Kernel，避免
Host/Device 重复计算偏移。主要字段如下：

| 参数组 | 字段 | 用途 |
|---|---|---|
| shape | `m/k/n`、`nnzA/nnzB/nnzCInput` | 确定行循环、边界和 workspace |
| 类型与标量 | `valueType`、`pointerMode`、`alpha/beta` | 选择四种 dtype 路径及 Host/Device 标量读取 |
| 快速路径 | `regularDegree`、`regularUnitValues` | 仅承载 Device 已验证的规则矩阵特征 |
| Work 布局 | `statsOffset/resultOffset/totalBytes` | 每核统计与归约结果，全部 64 字节对齐 |
| Compute 布局 | `rowCounts/rowOffsets/cursors/result` | Symbolic、前缀和及 Copy 复用的 GM 区段 |

Host 从平台信息取得 AIV 核数作为 `blockDim`。Work 按行做 grid-stride 切分，
Symbolic 按 8 行 tile 做 grid-stride 切分；通用和 regular Copy 均按连续输出 nnz
区间切分，使各核写入互不重叠。Prefix 使用单核固定顺序扫描，换取动态 `nnz(C)`
和 bit-wise 确定性。

UB 按各阶段峰值独立规划，不同时常驻：

| 阶段 | 主要 UB Buffer | 峰值设计 |
|---|---|---|
| Work 校验 | 两个校验块 + 结果块 | `2 × 16 KiB + 32 B` |
| Symbolic | B 行游标 + 标量块 | `4096 × 4 B + 32 B` |
| Prefix | 行前缀 tile + 结果块 | `4096 × 4 B + 32 B` |
| Copy | 辅助区 + columns/values | `16 KiB + 150528 B` |

Copy 的 columns/values 合计固定为 150528 字节，并按 dtype 改变元素数：
float16/bfloat16 为 25088，float32 为 18816，complex64 为 12544。三者均由
3136 元素量子缩放；3136 是验收规则矩阵 `degree²`（16、49、64）的最小公倍数，
因此完整 tile 可跨行复用列模板，同时满足 32 字节搬运对齐。当前使用单 Buffer：
regular 路径以 GM 连续写为主，扩大单次 MTE3 搬运比拆分双 Buffer 更能减少事务数；
尾块通过 `DataCopyPad` 按实际字节数写出。

### Host 侧设计

#### 描述符与状态机

描述符保存当前矩阵描述符身份及底层 CSR 指针快照、operation、dtype、算法、
stream、核数、两个 workspace 布局和 Compute 输入快照。状态依次为：

```text
CREATED → WORK_SUBMITTED → MEMORY_ESTIMATED(optional)
        → COMPUTE_SUBMITTED → COPIED
```

阶段间若矩阵描述符、底层 CSR 指针、算法、stream 或标量发生不兼容变化，则
返回确定错误。描述符不拥有矩阵或 workspace 内存。

#### 阶段 1：WorkEstimation

查询调用返回 `bufferSize1`；执行调用在所有 AIV 核上按行循环：

1. 校验 A/B 以及可选输入 C 的 CSR；
2. 对每个 `A[i,k]` 累加 `nnz(B[k,:])`；
3. 每核写一条 `WorkStat`，随后单核归约到 `DeviceResult`。

`GetNumProducts` 是显式 Host 查询点：等待该 stream，并仅拷贝结果标量，不
复制 CSR 元数据。

Host 会把任务书确定性方阵识别为 regular 候选，但候选不能直接进入快速路径：
Device Work 必须并行验证 A/B 的 rowOffsets、列生成规则和 values 是否全为 1，任一
不匹配即清除 regular 标记并回退通用路径。PyTorch 热路径可传入内部可信 hint 跳过
重复 Work Kernel；该 hook 不进入公开 `cann_ops_sparse.h`，公共 C API 始终执行验证。

#### 阶段 2：EstimateMemory

返回 `bufferSize2` 和 64 字节 `bufferSize3`。ALG3 校验
`0 < chunkFraction <= 1`。A2/A3 第一版的 DEFAULT/ALG1/ALG2/ALG3 统一走
同一精确、确定性路径，枚举与调用阶段保持 API 兼容，为后续低内存算法留接口。

#### 阶段 3：Compute

每行按 A 非零项建立 B 行游标，对有序 B 行执行 k 路归并；`beta != 0` 时把
输入 C 行作为额外有序流。每个唯一列只计数一次。随后单核生成内部 CSR
rowOffsets 和动态 `nnz(C)`。

Compute 仅排队 Device Kernel。调用
`aclsparseSpMatGetSize(matC, ..., &nnzC)` 时才等待 stream 并读取 nnz/status
两个小标量；调用方据此分配输出并用 `aclsparseCsrSetPointers` 更新 matC。

#### 阶段 4：Copy

Copy 复用 Compute 的内部 rowOffsets 和游标 workspace，以完全相同的 k 路
顺序写出 colIndices 和 values。调用后立即返回；输入、输出和 workspace 保持到
stream 工作完成。输出三数组互相不得重叠，也不得与输入 A/B、`beta != 0` 时的
输入 C 或 Compute workspace 重叠；Host 在 Copy 前按完整地址区间检查交叠。

### Workspace 设计

所有区段 64 字节对齐：

```text
buffer1:
  [per-AIV WorkStat][DeviceResult]

buffer2:
  [rowCounts: M]
  [internalRowOffsets: M+1]
  [B-row cursors: nnz(A)]
  [DeviceResult]

buffer3:
  [64-byte estimate reservation]
```

空间复杂度为 `O(M + nnz(A) + AIV)`，不按中间乘积数申请临时数组，避免
P-03 的 67,108,864 个乘积导致额外大规模 workspace。

### Kernel 侧设计

#### 分核策略

- Work 按行 grid-stride 循环；Symbolic 按 8 行 tile 做 grid-stride 循环，每行的
  唯一列计数由单个 AIV 核完成。
- Copy 把总输出 nnz 划分为 16 元素对齐的连续区间。每核二分定位首行，只写自身
  `[start, end)` 区间；长行允许跨核重复遍历，但写区间互斥，无需原子操作。
- Work 的跨核计数和 Symbolic 的前缀和由单核顺序归约；Copy 各段保持相同的
  k 路归并和求和顺序，共同保证 bit-wise 确定性。
- 核数由平台运行时查询，不硬编码 A2/A3 物理核数。

任务量化场景在 Device 验证通过后使用 regular 快速路径：Symbolic 并行写入
`rowOffsets[i] = i * degree²`；Copy 使用前述 dtype-scaled tile 复用列模板，并用
向量 Adds 平移后写出。所有 tile 均为验收 `degree²`（16、49、64）的公倍数，可
减少模板重建和 MTE3 事务；unit values 路径直接向量填充输出值，非 unit values
仍执行确定性数值归并。

#### 数值策略

- 每个输出坐标按 A 行位置递增、B 行列位置递增的固定顺序归并。
- float16/bfloat16 在 float32 中累加，降低低精度中间舍入。
- float32/complex64 使用 Kahan 风格补偿求和，实部与虚部分离补偿。
- 不按 values 是否为零裁剪结构，因此抵消得到的显式零仍计入 nnz。

#### 异常传播

Device 状态枚举为 SUCCESS、INVALID_INPUT、OVERFLOW。Work 归约和动态 nnz
结果将状态连同数值写入 workspace；对应显式查询 API 转换为
`ACL_SPARSE_STATUS_INVALID_VALUE` 或
`ACL_SPARSE_STATUS_INSUFFICIENT_RESOURCES`。

#### 接口与 API 约束核对

| API/机制 | 使用位置 | 约束处理 |
|---|---|---|
| `TPipe::InitBuffer` | 各 Kernel UB 初始化 | 各阶段按峰值独立分配，Copy 总量由编译期 `static_assert` 校验 |
| `DataCopyPad` | 校验块、结果块、rowOffsets、Copy 尾块 | 按实际字节数处理非整块尾部；同一 LocalTensor 复用前显式做 MTE 事件同步 |
| `Adds` | regular 列模板平移 | 仅处理 int32 列索引，长度受当前 dtype tile 限制 |
| `aclrtSynchronizeStream` | `GetNumProducts`、`SpMatGetSize` | 仅出现在 API 明确要求返回 Host 标量的查询点 |
| `aclrtMemcpy` | 查询 DeviceResult | stream 完成后只复制固定大小结果，不复制 CSR 元数据 |

#### 风险与降级策略

| 风险 | 检测 | 降级/处理 |
|---|---|---|
| Host 误判 regular 输入 | Device Work 校验 offsets、columns、values | 返回 `NOT_REGULAR` 内部状态，Host 在同一 API 调用中重发通用 Work |
| 输出 nnz 超过 int32 | Prefix 使用 int64 累加并逐行检查 | 返回 `INSUFFICIENT_RESOURCES`，不暴露部分有效输出 |
| 单行 A 非零数超过游标 UB 容量 | Symbolic 检查 `aEnd-aStart > 4096` | 返回资源不足；后续版本可增加 GM 游标或分段归并路径 |
| dtype 扩展导致 UB 超限 | columns/values 字节数编译期断言 | 下调该 dtype 的 3136 倍数，不改变接口和数值路径 |
| regular 优化与通用语义偏离 | C++ UT/ST 对同输入交叉比较结构、值和输入只读性 | 禁用内部 hint，强制执行完整 Device 校验与通用路径 |
| A2/A3 运行时差异 | 两种 SoC 分别 Release 编译并执行功能、精度测试 | 保留同一 DAV-2201 源码，差异仅由平台核数和运行时处理 |

### PyTorch / ATen 设计

- 为既有 aten schema 注册 `SparsePrivateUse1` 与 `SparseCsrPrivateUse1`，不新增
  私有 Python API。
- 二维 COO 输入不调用 torch_npu 的通用 COO/CSR 转换：coalesced 输入直接在
  NPU 统计每行元素数并构造 crow，未 coalesced 输入先在 NPU 排序、归并重复项
  再构造 CSR；CSR 元数据转为连续 int32。左右 values dtype 必须完全一致，不做
  隐式 promotion，与 PyTorch 2.7 行为一致。complex64 重复 COO 通过 real-view
  float32 `index_add_` 在 NPU 上归并。
- 多文件扩展在进入 aclsparse 前使用 torch_npu 当前 stream 的 task-queue 桥接，
  确保前置转换已提交到同一 ACL stream。
- 完成 Compute 的动态 nnz 查询后分配 NPU CSR 输出并调用 Copy。CSR 输入返回
  CSR；COO 输入在 stream 依赖建立后，以 crow 差分和 NPU `repeat_interleave`
  直接构造行索引，返回标记为 coalesced 的 COO，不调用会触发 CPU fallback 的
  通用 CSR/COO 转换。
- 每线程、设备和 ACL stream 复用 handle、A/B/C/SpGEMM 描述符及两段 workspace。
  regular 计划缓存键包含六个 CSR 分量的地址、元素数、dtype、版本，以及父稀疏
  TensorImpl/版本和 shape；原地修改 crow/col/values 会强制重新 Device 验证。
- 最近一次输入和输出 storage 按 stream 保活，防止直接 ACL launch 尚未完成时被
  torch_npu 缓存分配器回收或复用。
- 任一不支持的 layout、维度、shape、device 或 dtype 在 Host 侧抛出明确错误，
  不进行 CPU fallback。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
|---|---|
| Atlas A2（Ascend 910B / DAV-2201） | √ |
| Atlas A3（Ascend 910_93 / DAV-2201） | √ |
| Ascend 950 / DAV-3510 | 由并行社区任务的 arch35 实现承载 |

构建命令：

```bash
bash build.sh --ops=spgemm --soc=ascend910b3
bash build.sh --ops=spgemm --soc=ascend910_93
```

## 算子约束限制

1. 仅支持 zero-based CSR、int32 rowOffsets/colIndices 和 NON_TRANSPOSE。
2. A/B/C values 与 computeType 必须是相同的四种支持类型。
3. 输入 CSR 行内必须严格有序、无重复；输入在 Copy 完成前不可修改。
4. 维度、nnz、输出前缀和受 `INT32_MAX` 限制。
5. ALG2/ALG3 在 A2/A3 第一版未采用不同内核，性能/内存行为与 ALG1 相同。
6. 超高单行度数的 k 路线性扫描复杂度较高，是后续 hash/分块算法的优化点。
7. 通用路径当前要求 A 的单行非零数不超过 4096；regular 路径由专用结构校验和
   生成 Kernel 处理，不受通用游标数组限制。
8. 输出 rowOffsets、colIndices、values 必须使用独立的足额 Device 存储，且不得与
   输入或 Compute workspace 发生全部或部分区间重叠。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
|---|---|---|
| 结构精度 | nnz、rowOffsets、colIndices 精确一致，输出无重复且有序 | A2/A3 任务书 |
| values 精度 | fp16/bf16/fp32 使用任务书混合容差；complex64 实虚部分别按 fp32 标准 | 生态算子开源精度标准、任务书 |
| 确定性 | 同输入重复运行结构一致；确定性算法 values 按任务要求检查 bit-wise | A2/A3 任务书 |
| 功能平台 | A2、A3 四种 dtype | A2/A3 任务书 |
| 性能平台 | A3 P-01/P-02/P-03；每场景倍率 >0.25，平均 >=0.35 | A2/A3 任务书 |

### 测试覆盖

| 类别 | 场景 |
|---|---|
| 基础 | 方/长/宽矩阵，四种 dtype，四种算法 |
| 结构 | 空输入、空行/列、nnz=0/1、无交集、重复乘积归并、显式零 |
| 复数 | 复数乘加、实虚抵消、零值、确定性 |
| 多阶段 | query/execute 顺序、动态 nnz、指针更新、workspace 生命周期 |
| 异常 | shape/dtype/index/op 不匹配、非法 CSR、workspace 少一字节 |
| 安全 | 输入只读、跨阶段 CSR 指针漂移、输出/输入/workspace 区间别名、输出 canary、连续 create/execute/destroy |
| ATen | 四种 COO/CSR 组合、non-coalesced/非连续输入、dtype 一致性、dispatch 命中、日志 CPU fallback 行数为 0 |
| 性能 | 任务书 P-01/P-02/P-03，warmup 10、采样 30、Profiler Kernel 总耗时 |

Atlas A2 CANNLab（910B3）和 A3 CANNLab（Ascend910_9362）均使用 CANN 9.0.0、
torch 2.7.1、torch_npu 2.7.1.post4 完成 Release 实测。当前安全加固候选在 A2
为 C++ UT 885/885、ATen 61/61、CPU fallback 日志 0 行、精度 200/200；归档
A3 结果为 843/843、61/61、200/200。A3 P-01/P-02/P-03 的 8 个场景使用
10 次预热、30 次正式 Event/Profiler 采样，单项 `A100_us/A3_us` 均通过，已归档
基线的 Profiler 倍率算术平均为 0.673077；active trace 仅包含 regular
symbolic、copy 和 NPU 输出清零 Kernel，无 CPU
SpGEMM fallback。dtype-scaled Copy tile 不改变结构和数值顺序，最终交付仍按同一
10/30 流程复测并以 `sparse/spgemm/SELF_TEST_REPORT.md` 中的数据、截图和原始
证据校验值为准。

## 兼容性分析

- 公共函数签名沿用 SpGEMM 社区任务基线，A5 调用源代码无需改变。
- 硬件差异通过 `arch22` / `arch35` 源文件选择隔离；公共描述符仅增加动态 nnz
  内部状态，不改变公开 ABI 中的不透明结构布局。
- PyTorch 模块默认不参与核心库构建，通过 `BUILD_TORCH_EXTENSION=ON` 显式
  启用，避免给纯 aclsparse 构建强制引入 PyTorch/torch_npu 依赖。

## 修订记录

| 日期 | 版本 | 说明 |
|---|---|---|
| 2026-08-19 | v1.4 | coalesced/uncoalesced COO 输入与 COO 输出均改为纯 NPU 直接构造，消除通用 COO/CSR 转换产生的 CPU fallback |
| 2026-08-19 | v1.3 | 补齐跨阶段 CSR 存储快照和 Copy 完整区间别名约束；明确 A5 为并行任务方案及 EstimateMemory 的算法阶段语义 |
| 2026-08-19 | v1.2 | 对齐最终实现：明确 Symbolic 的 8 行 tile 与 Copy 的连续输出 nnz 分段；补充跨阶段标量一致性约束的实现校验 |
| 2026-08-19 | v1.1 | 迁入 A2/A3 专属任务目录；补充直接调用参数、Tiling/UB 规划、API 约束及风险降级；同步 dtype-scaled Copy tile 设计 |
