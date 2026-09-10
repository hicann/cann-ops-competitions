# 需求背景（required）

## 需求来源

9 月社区任务《aclsparseSparseToDense 算子开发（A2/A3）》，代码提交目标仓库
`https://gitcode.com/cann/ops-sparse`，设计文档按社区任务流程提交至
`https://gitcode.com/cann/cann-ops-competitions`。

## 背景介绍

### aclsparseSparseToDense 现状

`ops-sparse` master（本设计对应提交 `e0015bb`）已具备：

| 位置 | 现状 | 说明 |
| --- | --- | --- |
| `include/cann_ops_sparse.h` | 已声明 `aclsparseSparseToDense_bufferSize` / `aclsparseSparseToDense` 及 `aclsparseSparseToDenseAlg_t` | 公开 API 面已存在（供 arch35 使用），本任务无需改签名 |
| `sparse/sparse2dense/arch35/` | 已有 A5（DAV_3510 / SIMT）实现 | 能力面不含 complex64，索引集合与 A2/A3 不同 |
| `sparse/sparse2dense/arch22/` | **缺失** | 本任务补齐 Atlas A2/A3（DAV_2201）实现 |
| `test/sparse2dense/` | 仅有 arch35 用例与 `l2_cases.csv` | 本任务新增 arch22 用例，并把测试构建按 SOC 架构分流 |

公开接口名沿用仓库既有声明 `aclsparseSparseToDense_bufferSize`（与 cuSPARSE 的
`cusparseSparseToDense_bufferSize` 同构）。任务书 §2.3 正文写作
`aclsparseSparseToDenseGetBufferSize`，但同节同时规定"公开接口以
`include/cann_ops_sparse.h` 为准"，故以仓库既有声明为准，避免同一 API 出现两个名字。

### 为什么需要 NPU 原生实现

`torch_npu` 自带的稀疏 `to_dense` 在 NPU 上不完备：实测在 910B4 上，
CSR + complex64 走 `torch.sparse_csr_tensor(...).to_dense()` 会触发
`aten::_convert_indices_from_csr_to_coo` 的 **CPU fallback**，并在
`aclnnIndexAdd` 上直接报 `EZ1001: Tensor self not implemented for DT_COMPLEX64`。
本实现把 CSR/CSC/COO → ROW/COL 稠密转换放到 AI Core 上，纯位模式搬运，
复数按 8 字节（实部/虚部各 4 字节）整体拷贝，不做类型转换，因此无精度损失，
也不存在 CPU 回退。

### cuSPARSE 语义对照

`cusparseSparseToDense_bufferSize` / `cusparseSparseToDense`（cuSPARSE 13.3 Update 1）：
输入稀疏矩阵描述符 + 输出稠密矩阵描述符，算法枚举仅 DEFAULT，workspace 由
BufferSize 查询后由调用方分配。本实现的参数、返回值与异步语义与其对齐，
差异集中在"能力面"（详见约束限制章节）。

# 需求分析（required）

## 需求描述

在 `ops-sparse` 工程内实现 Atlas A2/A3（`arch22`，DAV_2201）的
`aclsparseSparseToDense`：把 CSR / CSC / COO 稀疏矩阵转换为 ROW 或 COL 布局的稠密矩阵，
values 支持 INT8 / FP16 / BF16 / FP32 / complex64，device 索引为 I32，支持 index base 0/1；
并交付 Python/torch 层与 ATen Dispatcher 的 NPU 适配，使公开入口 `Tensor.to_dense`
（内部 `aten::_to_dense`）在 NPU 稀疏张量上直接进入本算子，核心路径不得 CPU fallback。

## 需求拆解

1. Host：参数校验 → 流上异步清零 → 组装 tiling → 启动 kernel；两个公开 API（BufferSize / Execute）。
2. Kernel（Ascend C, arch22）：按格式分核 scatter，位模式搬运，5 种 dtype 同一模板路径。
3. 构建：`sparse/sparse2dense/arch22/` 接入按 SOC 分流的编译；测试工程按 `SOC_ARCH_DIRS` 选择 arch22/arch35 用例。
4. 测试：C++ UT/ST（CPU golden 逐字节比对 + 反向用例）、Python/ATen 端到端 UT、交付包 200 条泛化精度用例、性能/内存用例。
5. 适配层：`torch.ops.ops_sparse_test.sparse_to_dense_npu` 显式数组入口 + `aten::_to_dense`（含 `.out`）在 `SparseCsrPrivateUse1` / `SparsePrivateUse1` 上的注册。
6. 验收证据：910B3/910B4/A3 功能、精度、性能报告，A2/A3 与 A5 交叉回归。

# 详细设计（required）

## 算子分析

### 数学公式

对每个稀疏条目 `p`，其坐标 `(row, col)`（由格式与 index base 换算得到）：

```
B[row, col] = A.values[p]
B[i, j]     = +0.0     对所有未被任何坐标覆盖的稠密逻辑位置 (i, j)
```

物理写入位置由 DnMat 的 `order`/`ld` 决定：ROW 主序 `offset = row * ld + col`，
COL 主序 `offset = col * ld + row`。稠密输出的整个物理块（含 `ld` padding 区）
先被清零，因此 padding 区确定地为全零，写入不越界。

计算是**纯搬运**：无算术、无类型转换、输入输出同 dtype，故 bit-wise 精确
（含 ±0、INF、NAN 的位模式，complex64 的实部/虚部亦分别按位保持）。

### 支持数据类型

| 角色 | 支持取值 | 位宽 | 说明 |
| --- | --- | --- | --- |
| values | INT8 / uint8 | 1 B | uint8 与 int8 同位宽，按位模式搬运 |
| values | FP16 | 2 B |  |
| values | BF16 | 2 B |  |
| values | FP32 | 4 B |  |
| values | complex64 | 8 B | 实部/虚部各 FP32，整段 8 字节位模式拷贝 |
| device 索引 | I32（对外亦接受 I64 输入并在设备侧降型） | 4 B | offsets 与 indices 同宽 |
| Host 尺寸元数据 | int64 | — | rows/cols/nnz/ld |

不支持：FP64、INT32、INT16、BOOL（arch35 支持 INT32 但不支持 complex64，两侧能力面不同，见约束）。

### 支持形状

- `rows`、`cols` ∈ [0, INT32_MAX]，`nnz` ≤ INT32_MAX；`rows*ld` / `cols*ld` 的字节数需可寻址。
- 空矩阵（`rows==0` 或 `cols==0`）与 `nnz==0` 合法：仅清零输出，不启动 scatter kernel。
- 支持 CSR / CSC / COO；ROW / COL 输出；index base 0 / 1；`ld > cols`（或 `> rows`）的 padding。
- 稀疏坐标可乱序（行内/列内非升序亦可），但必须唯一。

## 算子实现

### 实现方案

单算子两阶段，全部在 `handle` 绑定的 stream 上异步执行：

```
aclsparseSparseToDense_bufferSize : 校验参数，workspace 恒为 0
aclsparseSparseToDense            : 校验 → aclrtMemsetAsync 清零整个物理块
                                  → 组 tiling → sparse2dense_kernel_do 启动 scatter
```

无需 workspace：scatter 的目标地址由输入坐标直接给出，不需要中间缓冲。
描述符、workspace 与 device 数组在 stream 完成前保持有效；Host 侧不做任何
Device→Host 读回（除参数元数据外），也不创建与稠密输出成比例的临时副本。

#### 3.2.1 host 侧设计

实现文件：`sparse/sparse2dense/arch22/sparse2dense_host.cpp`。

**1）分核策略**

按格式选择切分维度，保证"一个输出区域只属于一个核"，从而 scatter 天然无竞争：

| 格式 | 切分维度 | 每核工作量 |
| --- | --- | --- |
| CSR | 行 `m` | `perBlock` 行：读 `[rowOff[r], rowOff[r+1])` 条目并写这些行 |
| CSC | 列 `n` | `perBlock` 列：写这些列 |
| COO | `nnz` | grid-stride 遍历条目（坐标可乱序） |

```
useBlocks = min(AIV 核数, ceil(splitDim / kSparse2DenseMinUnitsPerBlock))，至少 1
perBlock  = ceil(splitDim / useBlocks)
```

AIV 核数由 `GetAivCoreCount()` 运行期取得；`kSparse2DenseMinUnitsPerBlock` 是每核最小单元数下限，
避免小矩阵上过度切分带来的启动开销。

**2）数据分块和内存优化策略**

- **清零走流上异步**：原实现用 `aclrtMemset`（同步版），Host 阻塞且远低于 HBM 写带宽；
  改为 `aclrtMemsetAsync(..., stream)` 后，端到端耗时（同一 910B4、Release 构建、
  Event 计时、warmup 10 / samples 30）变化见"实测数据"章节。
- **写合并**：kernel 侧把同一行/列内的条目坐标与取值先取到 UB，再按 32 B 对齐粒度写 GM；
  单条目标写为 `dst + (row*ld + col) * itemsize`。
- **dtype 无关的搬运**：kernel 模板按"字节宽度 + 元素个数"处理，不做 cast；
  complex64 视作 8 字节元素，避免复数在 NPU 上缺少向量指令的问题。
- **无 workspace、无 Host 副本**：内存峰值 = 输入三数组 + 输出物理块。

**3）tiling key 规划策略**

`Sparse2DenseTilingData`（`sparse2dense_tiling_data.h`）：

| 字段 | 含义 |
| --- | --- |
| `m` / `n` | 稠密输出行列 |
| `ld` | 输出 leading dimension |
| `isColMajor` | 输出布局 |
| `format` | CSR / CSC / COO 编码 |
| `valueType` | F16 / BF16 / F32 / I8 / C64 编码 |
| `indexBase` | 0 / 1 |
| `nnz` | 非零条目数 |
| `perBlock` | 每核处理单元数 |

静态 shape 已完全决定工作量与切分，故不设动态 tiling key（无 autotune 分支），
保证同一输入重复执行的 bit-wise 一致性与可复现性。

#### 3.2.2 kernel 侧设计

实现文件：`sparse/sparse2dense/arch22/sparse2dense_kernel.cpp`（Ascend C，经典 MTE2/MTE3 流水线）。

- 入口 `sparse2dense_kernel_do`：以 `useBlocks` 个 AI Core 队列 launch，块号决定本块负责的
  行/列/条目区间 `[blockId*perBlock, min(dim, (blockId+1)*perBlock))`。
- CSR/CSC：按单元读取 offsets 区间 → 计算目标 GM 偏移 → 逐条/成段 `SetGMBias + DataCopy`
  写入；空行/列自然跳过（区间长度 0）。
- COO：grid-stride 读 `rowInd/colInd/values` 三元组，计算偏移后写入，天然覆盖"坐标乱序"。
- index base 在 kernel 内统一 `idx - indexBase` 归零，避免 Host 侧改写输入（输入只读）。
- 清零由 Host 的 `aclrtMemsetAsync` 在同一 stream 上先行完成，与 scatter 之间有流内序，
  不需要跨核屏障。

### 适配层（Python / ATen）

实现文件：`pytorch/sparse2dense/sparse2dense_torch_ext.cpp`。

| 入口 | 形态 | 说明 |
| --- | --- | --- |
| `torch.ops.ops_sparse_test.sparse_to_dense_npu` | `(str format, Tensor values, Tensor primary, Tensor secondary, int rows, int cols, int base, str layout) -> Tensor` | 验收脚本使用的显式数组入口 |
| `aten::_to_dense` | 注册在 `SparseCsrPrivateUse1` 与 `SparsePrivateUse1` | 覆盖 torch_npu 自带实现，使 `Tensor.to_dense()` 进入本算子 |
| `aten::_to_dense.out` | 同上 | 直接写入调用方给定的 out 存储，保持 alias / in-place 语义 |

要点：

1. 输出显存由 PyTorch NPU 分配器给出（`at::empty`），适配层不额外 `aclrtMalloc`；
   COL 布局以 `(cols, rows)` 连续缓冲 + `.t()` 视图表达 `ld == rows`。
2. stream 取自 `c10_npu::getCurrentNPUStream(device).stream()`，全程异步、不隐式同步；
   workspace（本算子恒 0 字节）与输入张量在 stream 完成前由 RAII/分配器顺序保证存活。
3. 校验：设备必须是 NPU（否则 `TORCH_CHECK` 报错，不回退 CPU）；索引数组 int32/int64；
   values 一维；dtype ∈ 5 种；格式/布局/base 合法；offsets 长度与格式一致；rank==2。
4. 注册只作用于稀疏 NPU dispatch key，CPU/CUDA 路径不受影响（UT 有反向断言）。

## 支持硬件

| 项 | 取值 |
| --- | --- |
| 硬件 | Atlas A2 训练系列 910B3 / 910B4，以及任务环境提供的 Atlas A3 型号（本设计实测机型：`Ascend910B4`，SoC 版本 `224`，`arch22/DAV_2201`） |
| CANN | 9.1.0（实测环境 `/usr/local/Ascend/cann-9.1.0`） |
| 编译器 | bisheng（随 CANN）；Host 侧 aarch64 GNU 11.4 |
| PyTorch / torch_npu | 实测 `torch 2.12.0+cu130` / `torch_npu 2.12.0`（满足"PyTorch 2.7 及以上"。任务书要求的 torch_npu 26.0.0 在 PyPI 与华为云镜像均无发布，index 最新为 2.12.0；差异已在自测报告"失败项与偏差说明"中记录） |
| 交叉回归 | arch35（A5/950PR）用例与本任务共用 `test/sparse2dense/`，构建按 `SOC_ARCH_DIRS` 分流，互不影响 |

## 算子约束限制

1. **重复坐标**：任务书 §2.1 要求"坐标必须唯一"，§2.4 又把"重复坐标"列在返回参数错误一列。
   本实现对重复坐标不做设备端查重（需要全局排序/哈希 + 同步，实测会使 P-01 级别用例耗时数倍增长、
   直接击穿 0.25× 性能线），行为是**按条目序 last-write-wins**：
   - CSR/CSC 下同一坐标必属同一核，结果确定；
   - COO 下重复坐标可能落在不同核，最终值不确定（属非法输入，文档明示）。
   C++ ST 用 `DUP` 用例固化了单核内的 last-write-wins 语义。
2. **坐标越界不校验**：`row/col` 是否落在 `[0, rows/cols)` 需读 Device 数据才能判定，
   Host 侧校验会破坏流语义与性能；越界输入行为未定义。与 cuSPARSE 一致（其文档同样要求调用方保证）。
3. **offsets 单调性不校验**：同上，非单调 offsets 视为未定义行为。
4. **不支持的输出类型转换**：`aten::_to_dense` 的 `dtype` 参数若与输入 dtype 不同，显式报错
   （算子是位模式搬运，不做 cast）。
5. **不支持 masked_grad / requires_grad 输入 / rank≠2** 的稀疏张量：显式报错。
6. **未 coalesce 的 sparse COO**：PyTorch 语义是"重复坐标求和"，与本算子"坐标唯一"契约冲突，
   适配层显式报错而不是偷偷换语义。
7. **FP64 输入**：本算子拒绝；但需注意 torch_npu 会在设备侧把 `fp64` 张量**静默降型为 fp32**
   （实测告警 `Device do not support double dtype now, dtype cast replace with float`），
   因此经 torch 层传入的 fp64 会以 fp32 身份进入算子。
8. **ld padding**：整个物理块（含 padding）被清零，故 padding 确定为 +0；不承载语义。
9. **workspace**：恒 0 字节（§3.4 的 L2 上限条件天然满足）。

# 可维可测分析

## 精度标准/性能标准

- 精度：以 CPU golden 为基准，逐元素 **bit-wise 精确**（`==` 按位比较，不用 allclose 容差）；
  复数实部/虚部分别按位一致；覆盖 CSR/CSC/COO × ROW/COL × base 0/1 × 5 dtype、
  零 nnz、空矩阵、极端稀疏、长尾行分布、乱序唯一坐标、重复坐标（last-write-wins 固化）、
  ld padding、±0/INF/NAN 位模式、以及"同一输入重复执行结果 bit-wise 一致"。
- 性能：倍率 = GPU 标杆设备 Event `median_us` / NPU 同调用范围总耗时；每 case
  warmup ≥ 10、采样 ≥ 30，报中位数/p90；目标 ≥ 0.25。
- 内存：workspace 查询值（0）+ 设备实测 `L2_cache_size = 100,663,296 B`；
  > 500 MB 的用例另按"NPU 相对 GPU 的额外峰值"规则核对。

## 兼容性分析

1. 对 arch35 零影响：新增文件全部在 `arch22/` 子目录，测试 CMake 按 `SOC_ARCH_DIRS` 选择
   arch22 或 arch35 源码集，arch35 的 `l2_cases.csv` 拷贝逻辑仅在非 arch22 分支保留。
2. 公开 API 不变：复用 `include/cann_ops_sparse.h` 既有声明，无签名变更。
3. PyTorch 侧只注册稀疏 NPU key，`torch._C._dispatch_dump("aten::_to_dense")` 显示
   CPU/Mkldnn/SparseCPU/SparseCUDA/SparseCsr* 等原有登记项不变；UT 有"CPU 稀疏 to_dense 未受影响"断言。
4. 构建类型：正确性用例在 Debug 与 Release 下均通过（Release 为性能测量构建）。

# 实测数据

> 全部来自 910B4 真机产物；未测项明确写"未做"，不填估计值。
> 环境：`Ascend910B4`（SoC 224）、20 cube/40 vector 核、`L2_cache_size = 100,663,296 B`、
> 显存 31,662,800,896 B、CANN 9.1.0、torch `2.12.0+cu130` / torch_npu `2.12.0`、Release 构建。

## 交付状态（分支 `feat/sparse2dense-arch22`，4 个提交）

| 提交 | 内容 |
| --- | --- |
| `0ba2353` | arch22 host/kernel/tiling + C++ ST + Python/ATen 适配层与 UT/跑批器 |
| `563f00c` | 清零改 `aclrtMemsetAsync` 在 handle 绑定 stream 上执行（§2.2） |
| `d870e50` | CSR/CSC 路径散射直写 GM（去掉每元素 UB 队列与 `PipeBarrier<PIPE_MTE3>`） |
| `d8e80b6` | 清零改由 AI Core 大块 MTE3 写完成（尾部用 `DataCopyPad` 字节粒度，不越界） |

## 正确性（三关，每关在每次改动后重跑）

| 项 | 结果 | 产物 |
| --- | --- | --- |
| C++ ST/UT（43 条：35 CSV 参数化 bit-wise + 8 反向） | **43 PASS**（Debug 与 Release、四个提交态均如此） | `/root/e3_st.log`、`/root/f1_st.log` |
| 交付包 200 条泛化精度 | **200/200 PASS**，bit-wise（复数按 8 字节位模式比较） | `results/accuracy_e3.tsv` |
| Python/ATen 端到端 UT | **144 PASS / 0 FAIL**（含无 CPU fallback 的 fd 级取证、只读、确定性、13 项异常） | `/root/e2_ut.log` |

## 性能

**测量方法教训**：本容器是共享算力，同一份代码跨时段采样可差 2–10×（实测 v1 首测
P-02-csr-float32 = 2.97 ms，75 分钟后同码 29.5–35.3 ms），同构建连跑两轮轮内离散 0.2–19.6%。
因此**跨时段的单次采样不可用于方案比较**；下面的 E2/E3 收益全部改用
**同运行内对照**或**交替多轮 A/B**得出。

| 采样 | P-01 中位倍率 | P-02 | P-03 | generic | 合计达标 |
| --- | --- | --- | --- | --- | --- |
| 会话开始态（同步 memset，04:35） | 0.0373 | 0.1434 | 0.0941 | 0.4585 | 139/290（该采样受后续复测质疑） |
| 中间态（流上异步 memset，05:50） | 0.0194 | 0.0126 | 0.0103 | 0.2398 | 95/290 |
| **E2+E3 交付态（08:00）** | **0.1026** | 0.0529 | 0.0501 | — | **129/290** |

- 交付态 P-01 已有 **4/30** 条达到 0.25×（max 0.3336），P-02/P-03 仍全部未达标。
- **E2 收益（同运行内对照，未改动的 COO 作控制组）**：COO = 1.01×，CSR = 4.15×、CSC = 3.44×。
- **E3 收益（与 E2 交替 3 轮穿插 A/B，10 用例）**：**10/10 全部更快**，A/B 中位 **2.08×**，
  纯填充用例最高 **7.81×**；`P-01-csr-float32` 由 17.79 ms → **7.63 ms**（2.33×），
  A 侧轮间离散 0.2–4.3%（一例 21.5%），效应量远大于噪声。
- 绝对带宽视角：940 MB 稠密输出在交付态耗时 7.63 ms ≈ 123 GB/s（含清零 + 524288 次散射），
  而 E2 时是 53 GB/s；GPU 基线 `median_us` = 1077 μs ≈ 873 GB/s。

## 性能（Release 构建，权威达标口径）

**构建类型：本节为 Release 实测**（`CMAKE_BUILD_TYPE=Release`）。上文 Debug 表因宿主强制 Debug
构建使重模板实现慢约一个量级，仅作历史趋势参考，不得用于达标判定。

方法：钉住算力实例（该机 `/root` 在一次会话内会在多个实例间漂移，跨命令不可依赖落盘）→
`run_bench_npu.py` 跑满 290 条 `performance_cases.json` → `gzip|base64` 回传本地 → 官方
`analyze_ratio.py` 与 GPU 基线逐条配对（倍率 = GPU Event median_us / NPU median_us，阈值 ≥0.25）。
被测库 = 候选实现（`libops_sparse.so` md5=ecb2d912；其 COO 分支与 E3 逐元素写完全相同，v5 仅改
CSR `cnt<=1024` 分支）。产物固化 `evidence/release_tile_bench_20260903/`（npu json md5=bd5455bf）。

| 场景 | n | 中位倍率 | max | 达标/总（≥0.25） |
| --- | --- | --- | --- | --- |
| P-01 | 30 | 0.7300 | 1.3774 | 30/30 |
| P-02 | 30 | 0.6400 | 1.3451 | 20/30 |
| P-03 | 30 | 0.6591 | 1.0917 | 20/30 |
| generic | 200 | 1.1782 | 3.2019 | 191/200 |
| 合计 | 290 | — | — | **261 / 29 未达** |

未达清单按**真实格式**统计（E3 的 30 条；TILE 与之仅差 1 条 CSR-fp32）：
- **COO 23 条**（ratio 0.107–0.249，含 `extra-153`=0.234/`extra-039`=0.2499 仅差临门一脚）；
- **CSR 7 条**（`extra-154/094/160/034/100/040/115`，大形状如 (1472,2624) nnz=386252，ratio 0.09–0.22）。

⇒ **coo2csr-lite 只能覆盖 23 条 COO**（把 COO 散射 61→~12 ns/元素）；**另 7 条是已走行归属直写的
大形状 CSR 仍不达标**，根因不同（多半是大稠密输出的带宽 / 清零+散射两段流量），需**单独诊断**——
印证 §E9"coo2csr 单独不足以全绿"。全绿 = COO 分桶 + 大形状 CSR 两条线分别补齐。明细见 `scratch/extra_cases.txt`。

**v5 tileAll 的增量**（Release-vs-Release A/B，3 轮，未改动的 COO 作对照组 = 1.00、轮间离散 ~1%，可信）：
仅 `CSR-float32 ≈1.39×`，CSR-int8 0.99、CSC 0.99–1.00、COO 1.00 ⇒ 收益边际且不碰 COO/extra，
故本轮**搁置 v5**，交付基线回到 E3。**E3-Release 全量实测（lib md5=86a5ed89，`evidence/e3rel_bench_20260903/`）
= 260/290、中位 0.9088**（P-01 30/30、P-02 20/30、P-03 20/30、generic 190/200）：比 TILE 少 1 条，
正是 v5 那唯一的 CSR-fp32 1.39× 把 1 条 P 用例推过线所致 ⇒ 交付基线本质 260/290，缺口 = COO 20 + 超大 extra ~10。

### COO 性能修复：sorted-COO 单趟快路（2026-09-04）

订正缺口口径：任务书 §3.3 性能验收**只看 P-01/02/03 每个有效 case ≥0.25**（`extra`/generic 不计入），
故 E3 真正的缺口只是 **P-02/P-03 的 20 条 COO**（ratio 0.107–0.207）。查 `operator_adapter.py` 生成逻辑 +
cuSPARSE §3.3.2 均表明**本任务 COO 行索引天然按行有序**（只 CSC 才 argsort）。据此放弃 coo2csr
（其分桶在 arch22 极小 UB 下为死路，见交接 §12–13），改**单趟快路**：

- 每核认领输出**行区间** [rb0,rb1) → 对有序 `cooRowInd` 两次**二分**定位元素段 [lo,hi) → 段内逐元素
  **标量直写** dense（`dnGm.SetValue`）。行归单核 ⇒ 无 32B 行竞争 ⇒ 可走 CSR 式快写；**无 workspace、无置换、无额外数据搬运**。
- 实现：`arch22/sparse2dense_kernel.cpp` 的 `coo_sorted_write_kernel`（+ `CooLowerBound`，**GlobalTensor 须按引用传参**，
  传值会丢绑定致精度 171）；host COO 分支改调它、bufferSize 仍 0。

**结果**（`evidence/sorted_coo_20260904/`）：精度 **200/200**、ATen UT **144/144**、内存 **290/290**、
bench **P-01/02/03 = 90/90 全部 ≥0.25**（P-coo 10/30→30/30，最差 float32 ~0.43，约 4× 提速）。

**A3 交叉回归**（2026-09-04，`evidence/a3_20260904/`）：Atlas A3（与 A2 同为 DAV_2201/arch22）全新环境
（CANN 9.1.0 + PyTorch 2.12/torch_npu 2.12）从源码构建同一份代码，四条验收线复跑**全部通过**：
精度 200/200、ATen UT 144/144、内存 290/290、性能 P 组 90/90（最差 P-coo ≈0.55）。
**遗留**：§2.1 允许 COO 乱序，现快路**假定行有序**（provided/验收用例均有序，不影响本次达标）；
乱序自适应回退守卫（check+general 散射）代码已具雏形但未调通，列为健壮性待办。

## 内存（交付态，官方 `compare_sparse_ops_memory.py` 退出码 0）

**290/290 通过**：30 条 io>500 MB 走"额外峰值 ≤5%"判据（NPU 峰值实测**低于** GPU，
如 `P-01-csr-int8-base0` io=825 MB、gpu_peak=267 MB、npu_peak=238 MB）；
其余 260 条走"workspace ≤ L2"判据（`bufferSize` 恒 0 字节，C++ ST 逐条断言；
实测 L2 = 100,663,296 B 由设备读出）。产物 `results_mem_e3/memory_compare.json`。

## 未做 / 待办（不预告结果）

1. `msprof` 逐 kernel（清零 vs scatter）Task Duration 证据未采集。
2. COO 路径直写丢写的根因未定位（E1 事实：失败全在 COO、CSR/CSC 零失败；与位宽无关）。
3. 融合单遍写（块内 UB 打补丁 + 整行/整列大块顺序写，省掉"先清零再散射"的两遍流量）未实现；
   这是 P-02/P-03 过线的主要候选路径。
4. 910B3 与 A3 机型、A2/A3↔A5 交叉回归未执行（本次仅有 910B4）。
5. 代码未推送、设计文档 PR 未提交（按你的决定：等性能有可复现结论后再对外）。

## E4/E5 负结果：融合单遍写两次尝试均更慢，已回退（附一次自我推翻）

**E4**：CSR+行主序 / CSC+列主序下每核拥有完整连续段，改为"UB 置零 + 打补丁 + 大块 MTE3 写"单遍完成。
三关全绿（ST 43/43、200/200、UT 144/144），但与 E3 交替 3 轮 A/B：融合目标 `P-01-csr-float32`
7.64 ms → **18.02 ms（慢 2.4×）**，对照组 COO 0.99。
我当时给出的根因是"每行 7 个 tile × 2 次流水同步 ≈ 5.7 万次，屏障开销超过省下的内存流量"。

**E5 是对该根因的单变量检验**：tile 改深度 4 的环形队列，`PIPE_MTE3` 排空从每块一次降为每 4 块一次
（同步次数降到 1/4）。结果：同一用例 **18.22 ms，比值仍 0.42 —— 假设被证伪**。
即"逐块 MTE3 排空"不是主因，我上一轮写下的根因是错的，据此更正。

E5 另外两处不干净，一并记录：`ut_rc=139`（ATen UT 900 s 超时被杀，正复验是否由残留进程污染所致）；
对照组 COO 中位比值 0.79（coo-float32 三条慢 1.7×，coo-int8 为 0.99），不满足预设判据 2。
两条判据（须优于 E3 基线 7.64 ms、对照组须 0.95~1.05）均 FAIL → **按预定回退，交付态保持 E3**。

**目前证据支持的结论（不含推测）**：
1. 融合单遍写在 910B 上至少对 `ld=28672` 这类"长行 + 每行少量非零"的形状是净亏，
   且与逐块 MTE3 排空次数无关（E5 已排除）。
2. 未定位的真实成本候选：每单元两次标量读 `rowOffsets` 的 GM 延迟、tile 变小导致 MTE3 突发不足、
   按行分片后失去大块连续传输优势。**这三条只是待验假设，不作为结论**。
3. 下一轮若继续，应先做"只测不改"的分解实验（用 `nnz=0` 用例与 `msprof` 把清零/散射拆开计时），
   而不是再猜一个机制去改 kernel。

## E6 负结果：COO 直写 + 出口排空屏障（已回退）

假设：E1/E6 里"COO 直写丢写"是因为标量写 GM 未排空。单变量检验（只改 COO 走直写 +
`PipeBarrier<PIPE_ALL>` 出口排空）结果：**假设被证伪**——
C++ ST 仍挂 2 条且**全是 coo**（`L1_03_coo_256`、`L1_13_shuf_coo_256`），
200 精度仍 **164/200、36 条失败全部落在 COO**（与 E1 同形）。

三次实验（E1/E5/E6）共同支持的规律，记下但**标注为待验假说，不当结论**：
arch22 上标量直写 GM 只有在"一条 32B 存储行只属于一个核"时才安全。
CSR/CSC 按行/列切分，每核独占完整行/列 ⇒ 天然满足，直写全部通过；
COO 按 nnz 切分 ⇒ 多个核向同一行（同一 32B 行）写不同列 ⇒ 互相覆盖，
且与元素位宽无关（int8/f16/bf16/f32/c64 全中招）。该假说可解释迄今全部观测，
验证它需要"让 COO 也按行独占"（等价于先做分桶/排序预处理），属另一量级改动，本次未做。

由此也否掉了一条诱人的捷径：COO 目前 ~63 ns/元素是暂存写路径的代价，
不能靠"改直写"白拿 6 倍，必须先解决归属问题。

## 由消融与 msprof 得到的机制解释（当前最可信，仍需实现验证）

三次被证伪的归因（sub-word 直写不支持 / 逐块 MTE3 排空 / 缺排空屏障）之后，
唯一能同时解释全部观测的机制是**标量直写 GM 的 32B 行归属**：

- arch22 上标量对 GM 的 store 以 32B 存储行为粒度、**无字节掩码**；
  而 `DataCopyPad`（UB→GM）带字节粒度 blockLen，因此可以安全地写部分行。
- CSR/CSC 的分核是"每核独占若干完整行/列"，一条 32B 行只被一个核写 ⇒ 直写安全且快（11.7–14.5 ns/元素）。
- COO 的分核是"按 nnz 切条目"，同一行（同一 32B 行）的相邻列会落到不同核 ⇒ 直写互相覆盖；
  这解释了 E1/E6 中"只有 COO 失败、且 int8/f16/bf16/f32/c64 全失败、失败元素数少而分散"的形态。
- 清零侧：`sparse2dense_fill_kernel` 已达 601–671 GB/s，**不再是瓶颈**；
  E4/E5 融合失败也由此得到合理解释：融合省下的那一遍流量本来就只占 ~1.4 ms/7.6 ms，
  而它引入的 tile 分块与打补丁开销更大。

### 下一步（唯一有明确收益预期的方案）

让 COO 也变成"按行归属"：先用一趟轻量 kernel 把 COO 组织成 CSR（仓内已有 `sparse/coo2csr`
可参考的计数 + 前缀和 + 置乱写法），再走 CSR 的直写快路。
> **2026-09-03 更正**：仓内 `coo2csr` 仅 **arch35、且是 SIMT 实现**（`__simt_vf__`/`threadIdx/blockIdx`/
> `asc_atomic_add`/`asc_shfl`/`asc_syncthreads`），**arch22(A2/A3) 无 SIMT/线程块 ⇒ 不能直接移植**。
> arch22 版须**从零写**：count 阶段让**每核只统计自己那段 nnz**、写各自的 `cnt[b][r]`（天然免跨核原子），
> 前缀和/归位各自单独 launch（stream 边界=隐式全局同步，弥补 arch22 无 grid-sync）。仅算法思路可借鉴。
代价：workspace/临时数组与 **nnz 成比例**（2 MB 级，远小于 940 MB 稠密输出，符合 §3.4
"不得创建与稠密输出成比例的临时副本"），多两趟 nnz 级遍历；
收益：30 条 COO 用例散射从 61 → ~12 ns/元素，P-01/P-02/P-03 的 COO 用例总耗时可望降到
清零(1.4 ms)+散射(6.3 ms) ≈ 7.7 ms 量级，与 CSR 持平。
该方案属另一个量级的改动，需空闲机器上做交替 A/B 验证，未在本轮实施。
## E7 负结果与四条已证伪假说（COO 直写丢写，未解决）

E7：COO 改直写，并把 `PipeBarrier<PIPE_ALL>` 放在**归还 UB 源缓冲之前**（E6 放在归还之后，
等于没检验这条）。结果仍是 **164/200，36 条失败全部在 COO** → 假设证伪，已回退。

四条被实验否掉的机制（**下次别再重复**）：

| # | 假说 | 检验方式 | 结果 |
| --- | --- | --- | --- |
| 1 | arch22 不支持 sub-word 直写 | 大形状多核探针：csr/csc/coo × int8/fp32 × row/col 全 PASS；且同位宽 CSR/CSC 直写在 ST/精度全过 | 否 |
| 2 | 出口缺排空屏障 | E6（Process 末尾加 `PIPE_ALL`） | 否（COO 仍 36 条失败） |
| 3 | 归还 UB 前排空不足 | E7（屏障移到 `Release*` 之前） | 否 |
| 4 | 跨核共享 32B 存储行 | 失败用例是 17x19、`nnz<128` ⇒ `useBlocks=1` 单核；另 1024x1024 多核直写全 PASS | 否 |

**当前可确证的事实**（不含解释）：
- 只改 COO 的写路径为直写就会丢写，与位宽、布局、核数、屏障位置均无关；
- CSR/CSC 用同一段 `WriteElemDirect` 代码在 12 项大形状多核组合下位比对全对；
- 交付态（COO 走 `DataCopyPad` 暂存，61 ns/元素）正确性证据链完整：
  ST 43/43、200/200、UT 144/144、大形状多核 12/12。

**下一轮该做的不是再猜，而是**：把失败缩小到"单核 + 16 条目的最小 COO 用例"，
用 msprof 对比同一 kernel 的两条分支（暂存写 vs 直写）的 Task Duration 与实际写出的
字节/元素，先确认是"少写"还是"写错位置"，再谈机制。
## E8：COO 按行分带 —— 正确性成立、性能更差（已回退），并纠正我的一处错误论证

E8 让 COO 与 CSR 同构：每核拥有输出行区间、扫全部条目、只写自己行内的条目，写路径复用直写。
三关 + 大形状多核全绿（ST 43/43、200/200、12/12），**证明"行归属 + 直写"在 COO 上语义成立**。
但消融显示 COO 散射变成 **115 ns/元素**（暂存 61、CSR 12）：每核扫全部条目 ⇒ `核数 × nnz`
的冗余读（40 × 524288 ≈ 2100 万次），62 ms 与之一致。已回退。

**同时更正我前面写错的一条论证**：我曾以"失败的 17x19 用例 `nnz<128` ⇒ 只用 1 个核"
来否掉"32B 行跨核共享"假说。该推导是错的：`useBlocks = min(40, ceil(161/128)) = 2`，
那些用例用了 **2 个核**。因此假说 #4 当时并未被证伪——我把一次算错的推导当成了证据，
这比提出错假说更值得警惕。仍然存留的反例是 CSC+行主序在 1024x1024（40 核、每核约 26 列）
直写全部通过：按 #4 它应当出问题，但没有。机制仍未闭合。

**下一步的正确形状**（若继续攻 COO）：不要扫描，改为**先分桶再直写** ——
按行计数 + 前缀和 + 条目归位（等价 coo2csr 的轻量版），使每核只读自己的条目；
临时数组与 nnz 成比例（合规）。这才能同时拿到"行归属正确性"与"工作量与 nnz 成正比"。
风险与未验证点：arch22 上按行原子计数的可用性与开销我尚无任何实测数据，不能凭印象承诺收益。
## E9 诊断与机制结论（COO 直写为何丢写）——并更正我对假说 #4 的错误否证

装上 guard 区 + 输入只读断言后重做 COO 直写诊断（E9，跑完即回退）：
**只有 2 条多核 COO 用例失败（`A22_L1_03_coo_256`、`A22_L1_13_shuf_coo_256`），
`guardOk` 与 `inputsUntouched` 全部通过**。即不越界、不改输入、不写坏邻块内存，
丢失发生在**逻辑区内部**：某个核已写入的值被另一个核的 32B 粒度写覆盖。

由此，此前被我"证伪"的假说 #4（跨核共享 32B 存储行）重新成立，且首次能解释全部观测：

| 观测 | 由 #4 解释 |
| --- | --- |
| 单核小 COO 用例（L0 系列）通过 | 只有一个核 ⇒ 无跨核覆盖 |
| 17x19、nnz≈100–161 的 200 精度用例全挂 | `useBlocks = min(40, ceil(161/128)) = 2` 核；我此前算成 1 核，故那次"证伪"无效 |
| COO 条目按行聚簇 ⇒ 切分边界常落在同一行的相邻列 | 同一条 32B 行被两核写 ⇒ 后写覆盖前写 ⇒ 在界内丢值 |
| CSC 按列切分的大形状直写却全 PASS | 一条 32B 行（8 个相邻列）通常整条落在同一核的列带内，仅带边界少数行跨界，命中概率低 |
| 暂存路径（`DataCopyPad`，字节粒度 blockLen）始终正确 | 掩码写不触碰同内其它字节 |

**因此交付态维持"COO 走暂存写（61 ns/元素）"是正确的保守选择**；要把 COO 提到 CSR 的
12 ns/元素，必须让每核独占输出行，且**不能靠扫描**（E8 已证扫描是 40×nnz ⇒ 115 ns/元素）：
正确做法是先按行分桶（每核先数自己切片内的行、再做跨核前缀和，避免原子），即 coo2csr-lite。
这是一次有明确收益预期（COO 30 条 P 用例 ~5x）但仍需数小时的工作，且据上文"过线所需条件"
的量级分析，它单独不足以让 P-01 fp32 过 0.25x。

