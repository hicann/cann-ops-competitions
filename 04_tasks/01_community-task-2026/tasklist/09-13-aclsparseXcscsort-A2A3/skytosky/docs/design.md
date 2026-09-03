# aclsparseXcscsort 算子设计文档（Atlas A2 / A3）

---

# 一、需求背景（required）

## 1.1 需求来源

2026 年 9 月社区任务——**aclsparseXcscsort 算子开发（A2/A3）**。

面向 Atlas A2 训练系列产品与 Atlas A3 系列产品（DAV_2201，`arch22`）完善
`aclsparseXcscsort` CSC 纯索引排序接口并合入 ops-sparse 仓库
（https://gitcode.com/cann/ops-sparse ，`master` 分支）。

- 功能及 C++ 接口语义参考 cuSPARSE `cusparseXcscsort`：bufferSizeExt、执行阶段、CSC
  描述符、index base、workspace、stream 和错误处理保持接口语义一致，统一使用
  `aclsparseXcscsort*` 命名；
- 本算子仅处理 **I32 索引**，不存在 values 或 compute dtype；
- 核心计算必须在 Atlas A2/A3 的 NPU 上完成，**不允许 CPU fallback**。

## 1.2 背景介绍

`ops-sparse` 仓库已存在 arch35（950 系列）的 cscsort 实现
（`sparse/cscsort/arch35/`），本任务以其为复用基线，新增 `arch22` 实现。两个硬件
范围的 Kernel 能力存在本质差异，决定了 arch22 不能简单照搬：

| 能力 | arch35（950） | arch22（A2/A3，910B/910C） | 对本算子的影响 |
|------|--------------|---------------------------|----------------|
| SIMT（`__simt_vf__`/`asc_vf_call`） | 支持 | **不支持**（编译期报 `attribute __simt_callee__/__simt_vf__ is not supported ... on current architecture`） | arch35 长列 SIMT merge-path 不可移植，长列路径须重写为纯矢量/标量方案 |
| `AscendC::Sort` / `MrgSort` | 支持 int32 RADIX_SORT | **仅支持 half/float**，无 int32 实例 | int32 索引排序无法在 UB 内调用高层 Sort，排序须完全自实现 |
| UB 容量 | 大于 910B/910C（192 KB） | 192 KB | run 容量、tiling 常数须按 arch22 重算 |

因此 arch22 的核心设计命题是：**在没有 SIMT、没有 int32 高层 Sort 的硬件上，自实现
一个带 payload（置换向量 P）的 int32 稳定排序 Kernel**，并满足 0.25×GPU 标杆的性能
门槛。

### 1.2.1 标杆算子支持的数据类型和数据格式

标杆为 cuSPARSE `cusparseXcscsort`（CUDA Toolkit 13.3）：

- 数据类型：纯 int32 索引（`cscColPtr`/`cscRowInd`/`P`），无 values dtype；
- 数据格式：CSC（Compressed Sparse Column），列指针单调非降，index base 0 或 1；
- 语义：逐列稳定升序排序 `cscRowInd`，同步重排 `P`，满足 `sortedVal[i] =
  origVal[P[i]]`。

### 1.2.2 标杆算子实现描述

cuSPARSE cscsort 在 GPU 上的经典实现为：按列分段（segment = 列区间）→ 段内排序
（短段用分段排序/位网络，长段走全局基数排序或归并）→ 置换同步应用到 P。关键特征：
**段（列）之间完全独立、段内要求稳定**，与 NPU 上"按累计 nnz 切完整列区间到各 AIV
核、核内自实现稳定归并"的设计一一对应。

### 1.2.3 标杆算子实现流程

```
cusparseXcscsort_bufferSizeExt(nnz)          # 查询 workspace（2*nnz*4B）
        │
        ▼
cusparseXcscsort(handle, m, n, nnz, descrA, colPtr, rowInd, P, pBuffer)
        │  参数校验（维度/base/对齐）
        ▼
   按列分段 ──► 段内稳定排序（key=rowInd，payload=P）
        │
        ▼
   rowInd/P 原地更新（调用方 stream，异步）
```

本设计在 NPU 侧的对应实现见 §3.2/§3.3（Host 校验与 tiling、Kernel 分核与双路径
排序）。

## 1.3 交付目标

| 交付物 | 交付位置 |
|--------|----------|
| **本设计文档** | `cann-ops-competitions` 仓对应社区任务目录，PR 评审后合入 |
| `aclsparseXcscsort` 公开接口、Host 实现、Ascend C Kernel、C++ UT、端到端适配 | `ops-sparse` 仓 `master` 分支；A2/A3 实现放入 `sparse/cscsort/arch22/`，测试放入 `test/cscsort/arch22/`，公共声明沿用 `include/cann_ops_sparse.h` |
| 任务包专项测试（精度/性能/内存三脚本 + baseline） | PR 中以 `test_cases/aclsparseXcscsort_testCase/` 等价结构提交 |
| 自测用例及测试代码、自测报告 | 社区任务 IT 系统提交验收 |

A2/A3 与 A5 任务的 PR 可能先后合入，后合入方须基于已合入版本更新、处理公共 Host
代码冲突并完成双向回归（见 §3.6）。

## 1.4 算子功能

对 CSC 稀疏矩阵每一列区间 `[cscColPtr[j]-base, cscColPtr[j+1]-base)` 内的行索引
`cscRowInd` 执行**原地稳定升序排序**，并同步重排调用方提供的置换向量 `P`，满足
`sortedVal[i] = origVal[P[i]]`。`cscColPtr` 只读；`cscRowInd`、`P` 原地更新。

---

# 二、需求分析（required）

## 2.1 需求描述

使用 Ascend C 在 Atlas A2/A3（arch22）上实现 `aclsparseXcscsort_bufferSizeExt` 与
`aclsparseXcscsort` 两个公开接口：对 CSC 每列行索引做原地稳定升序排序并同步重排
置换向量 P，精度与 CPU Golden **exact match**，性能达到 GPU 标杆（cuSPARSE）的
0.25 倍以上，核心计算全部在 NPU 完成。

## 2.2 需求拆解

| 编号 | 需求项 | 来源（任务书章节） | 验收方式 |
|------|--------|-------------------|----------|
| R1 | 复用 ops-sparse 现有 Legacy API 与矩阵描述符，实现 CSC 逐列原地稳定排序 + P 同步重排 | 功能实现要求 1、2 | C++ UT + ATK 用例 |
| R2 | `aclsparseXcscsort_bufferSizeExt` 按 nnz 精确计算 workspace 并检查溢出 | 功能实现要求 3 | C++ UT（精确值/溢出） |
| R3 | `aclsparseXcscsort` 复用现有 arch35 能力并新增 arch22 实现，补齐长列、多 run 归并和边界能力 | 功能实现要求 3 | C++ UT（长列/多 run/边界） |
| R4 | 复用 `aclsparseCreateMatDescr` 的 index base（0/1），完整验证描述符生命周期 | 功能实现要求 3 | base 0/1 专项用例 |
| R5 | P 是调用方初始化的输入输出置换向量（非 workspace）；pBuffer 来自 bufferSizeExt 且 128B 对齐 | 功能实现要求 4 | 非 identity P 用例 + 对齐校验用例 |
| R6 | 纯 I32 索引接口，不涉及 FP16/BF16/FP32/complex64 values dtype | 功能实现要求 5 | 接口评审 |
| R7 | 覆盖空矩阵/空列、nnz=0/1、重复索引稳定性、单长列、多核与多 run 边界 | 功能实现要求 6 | 功能/边界用例 |
| R8 | Host、Kernel、UT/ST、文档统一交付，不得 CPU fallback | 功能实现要求 7、其他要求 | Dispatch/Profiler 证据 |
| R9 | A2/A3 与 A5 公共排序逻辑解耦，支持交叉回归 | 功能实现要求 8 | 目录结构 + 回归 |
| R10 | 精度：`cscColPtr`/排序后 `cscRowInd`/`P` 与 CPU Golden exact match，含稳定性与逐列边界校验；重复执行 bit-wise 一致 | 精度要求 1–8 | ATK exact 比对 |
| R11 | 性能：P-01/P-02/P-03 三个锚点 case 均 ≥ 0.25×GPU 标杆；预热 10 次、采样 30 次取中位数 | 性能要求 1–8 | 官方 benchmark 脚本 |
| R12 | 内存：输入输出 <500MB 场景下，方案固有 workspace 不超过目标硬件 L2 Cache 容量 | 内存要求 | 内存采集脚本 |

## 2.3 输入输出规格

| 名称 | 角色 | dtype | 形状/长度 | 约束 |
|------|------|-------|-----------|------|
| `m`/`n`/`nnz` | 属性 | int | 标量 | 非负；m=0 或 n=0 时 nnz 必须为 0 |
| `descrA` | 输入 | aclsparseMatDescr_t | - | 提供 index base（0 或 1） |
| `cscColPtr` | 输入（只读） | int32 | n+1 | 单调非降；base=0 端点为 0/nnz，base=1 端点为 1/nnz+1 |
| `cscRowInd` | 输入输出（原地） | int32 | nnz | 元素换算后位于 `[0,m)`；按列稳定升序排序 |
| `P` | 输入输出（原地） | int32 | nnz | 调用方通常初始化为 `0..nnz-1`，同步重排 |
| `pBuffer` | workspace | void* | bufferSizeExt 查询值 | 128 字节对齐，不得与输入输出重叠 |

## 2.4 外部组件依赖

| 依赖 | 用途 | 来源 |
|------|------|------|
| CANN 工具链（Ascend C 编译器 ccec/bisheng、ascendc 库） | Kernel 编译 | 随 CANN 包安装（实测 CANN 9.0.0） |
| ops-sparse 公共基础设施（`aclsparse_handle_internal.h`、`log/log.h`、`PlatformAscendCManager`） | handle/stream、日志、平台信息（核数/UB 容量） | 仓内已有，直接复用 |
| AscendC 基础 API（`DataCopyPad`、`TPipe`、硬事件屏障） | GM↔UB 搬运与流水线同步 | CANN Ascend C 库 |

无第三方新增依赖；不使用 AscendC `Sort`/`MrgSort`（arch22 无 int32 实例）。

## 2.5 内部适配模块

| 模块 | 文件 | 职责 |
|------|------|------|
| 公开接口声明 | `include/cann_ops_sparse.h`（已有，零改动） | `aclsparseXcscsort_bufferSizeExt` / `aclsparseXcscsort` |
| Host 实现 | `sparse/cscsort/arch22/cscsort_host.cpp` | 参数校验、workspace 计算、tiling、Kernel 启动 |
| Kernel 实现 | `sparse/cscsort/arch22/cscsort_kernel.cpp` | 分核、短列批量 UB 归并、长列 GM 归并 |
| Tiling 定义/计算 | `sparse/cscsort/arch22/cscsort_tiling_{data,utils}.h` | TilingData 结构、runSize 反推 |
| C++ UT | `test/cscsort/arch22/cscsort_test.cpp` + golden | 49 用例（功能/边界/异常/PERF 锚点） |
| 端到端适配验证 | `torch_plugin/`（torch_npu 钩子） | 官方 benchmark/精度脚本对接 |

## 2.6 调用方式

```c
aclsparseCreate(&handle);
aclsparseSetStream(handle, stream);
aclsparseCreateMatDescr(&descrA);
aclsparseSetMatIndexBase(descrA, ACL_SPARSE_INDEX_BASE_ZERO);  // 或 ONE

// 1) 查询 workspace
aclsparseXcscsort_bufferSizeExt(handle, m, n, nnz, cscColPtr, cscRowInd, &bufferSize);
aclrtMalloc(&pBuffer, bufferSize, ...);   // 128B 对齐

// 2) 原地稳定排序 + P 重排（异步，使用 handle 的 stream）
aclsparseXcscsort(handle, m, n, nnz, descrA, cscColPtr, cscRowInd, P, pBuffer);

aclsparseDestroyMatDescr(descrA);
aclsparseDestroy(handle);
```

与 cuSPARSE 调用形态一致：两阶段（bufferSizeExt → 执行），无 preprocess 阶段；
P 由调用方初始化（通常 identity），cscRowInd/P 原地更新。

---

# 三、详细设计（required）

## 3.1 算子分析

### 3.1.1 数学定义

对每列 `j ∈ [0,n)`，记列区间 `I_j = [cscColPtr[j]-base, cscColPtr[j+1]-base)`，
求该区间上的稳定排序置换 `σ`（相等键保持原相对顺序），使得：

- `cscRowInd'[I_j[k]] = cscRowInd[I_j[σ(k)]]`（升序、稳定）；
- `P'[I_j[k]] = P[I_j[σ(k)]]`；
- 恒等 P 初始化时满足 `sortedVal[i] = origVal[P[i]]`。

无浮点数值参与，无近似误差概念，精度语义为与 CPU 稳定排序 Golden 的 **exact
match**（索引序列、置换序列、列边界逐项一致）。

### 3.1.2 支持数据类型

| 数据 | 类型 | 说明 |
|------|------|------|
| cscColPtr / cscRowInd / P | int32 | 纯索引算子，无 values dtype |

### 3.1.3 关键设计约束（arch22 硬件实测）

以下三条为开发期在 910C 上实测确认的硬件行为，直接决定了 Kernel 结构：

1. **无 SIMT、无 int32 高层 Sort**（见 §1.2）：排序由 Kernel 内自实现的稳定归并完成，
   key 与 payload P 始终同步搬运。
2. **DataCopyPad 的 UB 侧地址必须 32B 对齐**：非对齐地址配合小 blockLen（≤56B）的
   UB→GM 拷贝会触发 vector core exception（507035）。对策：批内每列独占一个按
   32B（8 个 int32）对齐的 UB 槽位；GM 侧的不对齐由 DataCopyPad 自身处理。
3. **标量写入 UB 的数据只能由 MTE3 引擎读出**：归并（标量引擎）写入 UB 后，同核
   后续标量读 UB 拿到的是旧数据（volatile、编译屏障、事件 flag 均无效），只有
   MTE3（DataCopyPad 写出）路径能读到新值。对策：归并结果**一律经 DataCopyPad
   写回 GM**，禁止标量回读 UB 后再标量写 GM 的"优化"。

## 3.2 Host 侧设计

Host 实现位于 `sparse/cscsort/arch22/cscsort_host.cpp`，两个 `extern "C"` 接口，
参数校验逻辑与 arch35 保持一致（硬件无关），tiling 常数按 arch22 重算。

### 3.2.1 参数校验

`aclsparseXcscsort_bufferSizeExt`：

- handle/pBufferSizeInBytes 非空；m/n/nnz 非负；空矩阵（m=0 或 n=0）必须 nnz=0；
  n>0 时 cscColPtr 非空，nnz>0 时 cscRowInd 非空；
- nnz=0 返回 workspace 0 字节；否则返回 `2*nnz*sizeof(int32_t)`，并对 size_t 溢出
  做显式检查（`ACL_SPARSE_STATUS_INSUFFICIENT_RESOURCES`）。

`aclsparseXcscsort`：在公共校验之上追加——descrA 非空且 indexBase ∈ {0,1}；
nnz=0 直接成功返回（早退，不启动 Kernel）；P、pBuffer 非空；pBuffer 128B 对齐。

### 3.2.2 tiling 策略

- **核数选择**：`coreNum = min(GetAivCoreCount(), min(n, nnz))`。每个非空列最多形成
  一个独立任务，限制启动核数不超过 nnz，避免大量空列矩阵启动无实际工作的 AIV。
  `GetAivCoreCount()`/`GetUbSize()` 走 PlatformAscendCManager，**硬件无关**。
- **runSize 反推**（`CscsortTiling::FindMaxRunSize`）：UB 需要 4 个 run 级 buffer
  （keyIn/pIn/keyOut/pOut，归并 ping-pong），另预留 8 KB；同时受 DataCopyPad 单次
  blockLen 上限 `2^21-1` 字节约束：

  ```
  runSize = min( (ubSize - 8KB) / (4 buffer × 4B),  (2^21-1)/4 )
  ```

  910C（UB 192 KB、48 AIV）实测 runSize=11760。UB 不足最小 run（<2 元素）时返回
  `ACL_SPARSE_STATUS_INSUFFICIENT_RESOURCES`。
- **tiling 出参**（`CscsortTilingData`）：`n / nnz / indexBase / runSize / coreNum`，
  定长结构直接随 Kernel 启动下发，无额外 H2D 拷贝。

### 3.2.3 tilingKey 规划

**本算子不需要 tilingKey**。排序路径（短列批量化 / 长列 GM 归并）由 Kernel 按每列
实际长度在运行时自行选择，Host 侧无需感知列长分布；index base 差异经 tiling 出参
`indexBase` 传入 Kernel 统一处理。单 Kernel 全场景覆盖，避免多 tilingKey 带来的
二进制膨胀与维护成本。

### 3.2.4 Kernel 启动

按 `tiling.coreNum` 个 AIV block 在调用方 stream 上启动单个 Kernel
（`KERNEL_TYPE_AIV_ONLY`），全程无 Host 同步。

## 3.3 Kernel 侧设计

Kernel 位于 `sparse/cscsort/arch22/cscsort_kernel.cpp`。整体流程为
Init（tiling 解析 + GM 指针绑定）→ Process（列区间排序）。

### 3.3.1 分核策略

以**累计 nnz 为权重**把完整列区间切给各核：核 `i` 负责元素区间
`[nnz*i/coreNum, nnz*(i+1)/coreNum)` 覆盖的列，列边界通过在 `cscColPtr` 上二分定位
（`FindColBoundary`），**单列不拆分**。各核只访问自己列区间对应的
rowInd/P/workspace，核间无任何同步。

### 3.3.2 核心排序原语：UB 内自底向上两路稳定归并

`MergeSortUb` 在 UB 内对单段做自底向上两路归并（宽度 1,2,4,... 倍增，ping-pong 双
缓冲），key 与 P 同步搬运；相等键优先取左 run（`aKey <= bKey` 取左），保证稳定排序
语义。两个关键实现决策：

- **`__ubuf__`/`__gm__` 裸指针访问**：绕开 `GetValue/SetValue` 的封装开销（实测
  较初始实现提升约 2.2×）；
- **归并头寄存器缓存**：左右归并头缓存在标量寄存器，每元素仅 1 次 UB 读（前进
  侧）+ 2 次 UB 写（key、P）。

### 3.3.3 短列路径（len ≤ runSize）：多列批量化

短列是性能主战场（三个性能锚点 case 的平均列长均远小于 runSize）。逐列独立处理的
固定开销（pipe Reset、InitBuffer、各阶段屏障）在短列场景占比极高，因此采用**批量化**：

1. **聚批**：列循环中将 `2 ≤ len ≤ runSize` 的列累积进批，直到 UB 槽位预算
   （`batchSlots + slot > runSize`）或批规模上限（`kMaxBatchSegs = 256`）触顶后
   `FlushBatch`；`len ≤ 1` 的列天然有序，直接跳过不进批；遇到长列先冲刷当前批。
2. **批内搬入**：每列独占一个 32B 对齐的 UB 槽位（`slot = align_up(len, 8)`），
   逐列 `DataCopyPad` 从 GM 搬入 key/P；一次 `MTE2_S` 屏障后进入计算。
3. **批内归并**：逐列调用 `MergeSortUb`，记录每列结果所在的 ping-pong 侧
   （`segInOut`）；一次 `S_MTE3` 屏障后写回。
4. **批内写回**：按各列结果侧逐列 `DataCopyPad` 写回 GM；`MTE3_MTE2` 屏障保证下
   一批 copy-in 不会复写未读完的 UB。

批量化把每列的 pipe 初始化与三次屏障摊薄到**每批一次**，是 P-01 场景从 10.31ms
降到 7.93ms（kernel 口径）的关键优化。

### 3.3.4 长列路径（len > runSize）：UB 分块排序 + GM 多趟归并

长列为低频路径（仅单列长度超过 UB run 容量时命中），以正确性优先：

1. 按 runSize 把长列分块，逐块走 `SortSingleRun`（DataCopyPad 进 UB → MergeSortUb
   → DataCopyPad 写入 GM workspace），形成有序 run；
2. 在 GM 上做自底向上两路稳定归并（`MergeRoundGm`，`__gm__` 裸指针标量循环，
   归并头寄存器缓存，每元素 1 次 GM 读 + 2 次 GM 写），src/dst 在 workspace 与
   原数组间 ping-pong；
3. 奇数趟后结果位于 workspace 时回拷到原数组（`CopyColGm`）。

workspace 布局：`[0, nnz)` 为 scratchRowInd，`[nnz, 2*nnz)` 为 scratchP，与
bufferSizeExt 的 `2*nnz*4` 字节精确对应。

### 3.3.5 流水线屏障

核内仅使用三类硬事件屏障，均按批/按 run 摊薄：`MTE2_S`（搬入→归并读）、
`S_MTE3`（归并写→搬出读）、`MTE3_MTE2`（写回→下一批搬入复写）、长列另有
`MTE3_S`（run 落盘→GM 归并读）。

## 3.4 精度设计

- **Golden**：CPU 侧稳定的逐列 int32 升序排序并同步重排 P（gtest golden 与官方
  精度脚本两套独立实现互为印证）；
- **比对口径**：`cscColPtr`（只读性）、排序后 `cscRowInd`、`P` 全序列 exact match，
  含逐列边界校验；重复键场景校验稳定语义（相等键原相对顺序不变）；
- **确定性**：归并比较为纯整数比较，无浮点、无随机、无核间交互，相同输入重复执行
  bit-wise 一致；
- 覆盖已排序、逆序、随机、重复、空列、单长列、多核边界的索引分布。

## 3.5 性能设计与实测

### 3.5.1 优化点

| 优化 | 收益 |
|------|------|
| `__ubuf__`/`__gm__` 裸指针替代 GetValue/SetValue | 约 2.2×（kernel 口径） |
| 归并头寄存器缓存 | 每元素 UB/GM 读次数减半 |
| 短列批量化（槽位对齐 + 屏障/初始化按批摊薄 + len≤1 跳批） | P-01：10.31ms → 7.93ms |
| 按累计 nnz 均衡切列、空核不启动 | 长尾/空列矩阵不浪费算力 |

### 3.5.2 实测数据（910C，CANN 9.0.0）

官方口径（torch Event 计时、含分发开销、预热 10/采样 30 取中位数，官方
`benchmark_sparse_ops_npu.py` + `performance_cases.json`）：

| 锚点 | 场景 | NPU 实测 | GPU 标杆 median_us | 性能倍率 | 门槛 |
|------|------|----------|--------------------|----------|------|
| P-01 | Llama 3.1 70B MLP：8192×28672，nnz=524288 | 7.86 ms | 2961.7–2969.3 μs | **0.378×** | ≥0.25× |
| P-02 | Qwen3-235B-A22B MoE：4096×1536，nnz=262144 | 5.01 ms | 2595.9–2610.4 μs | **0.521×** | ≥0.25× |
| P-03 | DeepSeek-V3 MoE：7168×2048，nnz=458752 | 8.53 ms | 2780.8–2783.1 μs | **0.326×** | ≥0.25× |

三锚点全部达标，最小余量 30%（P-03）。结果导出
`xcscsort_npu_performance.json/csv` 存档。

## 3.6 兼容性与共存设计

- arch22 实现全部位于 `sparse/cscsort/arch22/`、`test/cscsort/arch22/`，与 arch35
  目录平级隔离；公共接口声明沿用 `include/cann_ops_sparse.h`（已存在，零改动）；
- Host 公共逻辑（校验、bufferSizeExt 语义）两 arch 各自实现、语义一致，无共享
  状态；
- A2/A3 与 A5 任务 PR 先后合入时，后合入方 rebase 处理公共 Host 冲突并完成双向
  回归（任务书 PR 要求 3）。

## 3.7 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas A2 训练系列产品 | √ |
| Atlas A3 系列产品 | √ |

## 3.8 算子约束限制

- 仅支持 int32 索引；index base ∈ {0,1}；无 values dtype、无布局/转置/算法/标量参数；
- m/n/nnz 非负；cscColPtr 单调且端点与 base、nnz 一致；换算后 row index ∈ [0,m)；
- pBuffer 必须按 bufferSizeExt 查询值足量分配并 128B 对齐，不得与输入输出重叠；
  执行 ABI 不传实际 size，不做欠配检测；
- 越界行索引属接口前置条件违反，行为不做保证。

---

# 四、可维可测分析

## 4.1 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | 与 CPU 稳定排序 Golden exact match（cscRowInd、P、列边界逐项一致；重复执行 bit-wise 一致）；不适用浮点混合容差 | 任务书精度要求 + 《生态算子开源精度标准》单标杆方法 |
| 性能标准 | P-01/P-02/P-03 各 case 均 ≥ 0.25×GPU 标杆（torch Event 口径，预热 10/采样 30 取中位数） | 任务书性能要求 |
| 内存标准 | 方案固有 workspace `2*nnz*4` B（P-01 为 4 MB）不超过目标硬件 L2 Cache 容量 | 任务书内存要求条件 2 |

## 4.2 测试设计

`test/cscsort/arch22/cscsort_test.cpp`（gtest，**49 个用例全绿**）：

| 类别 | 用例（代表） | 覆盖任务书场景 |
|------|--------------|----------------|
| 基础功能 | Basic / AlreadySorted / Reverse / DuplicateRows / Base1 / EmptyCols / SingleElemCols / EmptyMatrix | 空矩阵/空列、重复索引稳定性、index base 0/1 |
| P 语义 | NonIdentityP（非 identity 置换输入） | P 为调用方输入输出 |
| 多核/边界 | ManySmallCols / MultiCoreSkewedCols / LargeNnzMultiCore / RunSizeBoundaries / LastColEmpty / SingleRow | 多核切分与 nnz=0/1 |
| 长列/多 run | LongCol / LongColMultiRun / WbMergeTwoRunsNoCopyBack / WbMergeThreeRunsCopyBack / WbMergeFiveRunsNoCopyBack / WbRunSizeExactMultiple | 单长列、多 run 归并、回拷奇偶 |
| 对齐/空闲核 | NonAlignedBuffer / WbAllSingleElemMultiCore / WbEmptyMatrixZeroDims / WbIdleCoresBetweenGaps / WbTailCoreNonAligned | 128B 对齐、空列矩阵、核间隙 |
| 参数异常 | NullHandle / NullBufferSize / InvalidM/N/Nnz / EmptyMatrixWithNnz / NullColPtr / NullRowInd / NullDescr / NullP / NullBuffer | 返回码与输入校验 |
| 性能锚点 | PERF_P01_Llama70B / PERF_P02_Qwen235B / PERF_P03_DeepSeekV3 + 6 个规模/分布用例 | 三锚点 + 泛化性能 |

端到端：torch_npu 钩子（`torch.ops.ops_sparse_test.xcscsort_npu`）跑通官方
`benchmark_sparse_ops_npu.py`（200 case）与精度脚本（200/200 exact match），作为
NPU Dispatch 无 CPU fallback 的证据来源之一；Profiler 证据随自测报告提交。

## 4.3 兼容性分析

既有 arch35 能力保持不动，arch22 为新增目录；公共头文件接口声明已存在，无 API 语义
变更，不影响存量用户。
