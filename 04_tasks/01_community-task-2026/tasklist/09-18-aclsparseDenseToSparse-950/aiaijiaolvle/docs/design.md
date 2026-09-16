# 需求背景（required）

## 需求来源

昇腾 CANN 社区任务 2026 年 9 月社区任务 —— aclsparseDenseToSparse 算子开发（950），任务列表编号 18。要求在 ops-sparse 工程（基线 `d86aa93`）上，参考 cuSPARSE `cusparseDenseToSparse_bufferSize` / `cusparseDenseToSparse_analysis` / `cusparseDenseToSparse_convert` 三阶段接口语义，补齐 Ascend 950（DAV_3510，`arch35`，A5）实现，并将 Python/torch 层、ATen NPU 注册、aclsparse C++ 接口、Ascend C Kernel 和测试统一交付。

## 背景介绍

### aclsparseDenseToSparse 算子现状分析

基于 ops-sparse master `d86aa93` 分析，`aclsparseDenseToSparse` 现有 arch35 能力如下：

| 项目 | 既有能力 | 本任务前缺口 |
| --- | --- | --- |
| 稀疏格式 | CSR / CSC / COO / Blocked-ELL | Blocked-ELL 尾块容量语义与 cuSPARSE 13.3 不一致 |
| values 数据类型 | INT8 / FP16 / BF16 / FP32 | 不支持 complex64 |
| 索引 | I32，base 0/1（保留 I64 兼容） | — |
| Dense 布局 | ROW / COL，合法 leading-dimension padding | — |
| Python/ATen | 无 | `to_sparse*` 系列入口未适配 NPU |
| 性能 | 逐元素串行计数/提取 | P-01 大形状转换耗时显著高于 GPU 标杆 |
| sanitizer | count 缓存复用序列存在 2010 条 memcheck 越界告警（工具报告的区间元数据与缓存复用交互问题） | 需根因修复（本任务已完成：工作区改正向布局后 2010 → 0） |

### aclsparseDenseToSparse 算子功能分析

算子将 `matA`（稠密矩阵）按 `matB`（稀疏矩阵描述符）预设格式转换为稀疏矩阵，分三阶段：

1. `aclsparseDenseToSparseGetBufferSize`：查询 Analysis/Convert 所需 workspace；
2. `aclsparseDenseToSparseAnalysis`：扫描非零分布，得到每行/列 nnz 与 offsets，调用方按实际 nnz 分配输出并经 `aclsparseCsrSetPointers` / `aclsparseCscSetPointers` / `aclsparseCooSetPointers` 绑定；
3. `aclsparseDenseToSparseConvert`：执行实际转换，写出 values/indices。

语义约束：正零、负零不形成稀疏结构；NaN、Inf、subnormal 保留；complex64 任一分量非零即保留。CSR 按行、CSC 按列、COO 按 row-major 生成确定性顺序。输入 Dense 与 BELL pattern 只读；BELL 无效块与超出逻辑行列的元素写正零。Convert 重扫当前输入结构，不依赖过期 Analysis 快照。

# 需求分析（required）

## 需求描述

使用 Ascend C 在 arch35 上完善 `aclsparseDenseToSparse`：values 支持 INT8、FP16、BF16、FP32、complex64，Device 索引 I32，index base 0/1；支持 CSR/CSC/COO/Blocked-ELL 与 ROW/COL 布局；完成 Python/ATen 适配（PyTorch 2.7+、torch_npu 26.0.0+，不得 CPU fallback）；性能达到任务书 GPU 同口径验收线。

## 需求拆解

1. 补齐 complex64 values 全链路（host/kernel/golden/wrapper/测试）；
2. Blocked-ELL 非整除尾块（ceil 几何、写正零、descr 放宽，对齐 cuSPARSE 13.3）；
3. 计数/前缀和/提取向量化与并行化，GPU 同口径性能达标；
4. Python/ATen `to_sparse` / `to_sparse_csr` / `to_sparse_csc` / `to_sparse_bsr` 及 Dispatcher 适配；
5. sanitizer memcheck/racecheck/synccheck 0 告警（根因修复，不规避工具）；
6. workspace 有界（不超过目标硬件 L2 容量）；
7. 精度全量通过并完成 A2/A3/A5 公共 Host 交叉回归。

# 详细设计（required）

## 算子分析

### 数学公式

DenseToSparse 将稠密矩阵 `A ∈ K^{M×N}`（K 为 values 类型域）转换为稀疏表示：

- CSR：`values(k)`、`col_indices(k)`（k = 0..nnz-1，按行序），`row_ptr(i)`（i = 0..M）；
- CSC：按列序对称定义；
- COO：`row_indices(k)`、`col_indices(k)` 按 row-major 序；
- Blocked-ELL：`blockRows = ceil(M/b)`、`slots = ceil(ellCols/b)`；pattern 元素数 `blockRows × slots`，values 元素数 `blockRows × slots × b × b`，每块 values 块内 column-major。整除时退化为 `rows × ellCols`。零判定为位级：complex64 同时屏蔽实虚两个 FP32 符号位，任一分量非零即保留；不使用浮点比较。

### 支持数据类型

| matA values | matB values | 索引 |
| --- | --- | --- |
| INT8 / FP16 / BF16 / FP32 / complex64 | 同 matA | I32（base 0/1） |

### 支持形状

任意 M×N 二维稠密（Host 尺寸 int64），ROW/COL 布局及合法 leading-dimension padding；Blocked-ELL blockSize ∈ {16, 32, 64} 且支持非整除尾块；`slots ≤ blockCols`。空矩阵（nnz=0）为合法输入。不支持任意 strides、batch/hybrid sparse（显式报错）。

## 算子实现

### 实现方案

#### 3.2.1 host侧设计：

host 侧完成 workspace 布局与容量计算、分核与批参数计算、路径派发，全部容量乘法经过溢出检查；ceil 用除法与余数表达（避免 `n+b-1` 溢出）。

##### 1. 分核策略：

优先使用满核原则。Analysis 按 Dense 实际扫描元素量（而非较小的行/列数）选择核数；BELL 按 padded payload 元素量分核，线程独立负责输出元素，避免单线程串行搬完整大块。核间均分时无大小核区分；不能均分时余量块分给前几个核。无输出线程提前返回，避免空 slot 除法开销。

##### 2. 数据分块和内存优化策略：

输入沿确定性输出顺序切分为 256 元素 count unit。每批 INT8/FP16/BF16 为 128 units、FP32 为 64、complex64 为 32。连续计数采用双输入缓冲：最大输入 UB 128KiB、结果缓冲约 5KiB，低于实际 248KiB UB；当前块就绪后预取另一 slot 再执行 VF。workspace 存储 status、nnz、bitmap、unit counts 与多级 prefix，随扫描 units 增长，不分配 Dense 大小的 Host 中间缓冲；Analysis 与 Convert 复用同一查询所得 workspace；BELL 预置 pattern 的核心 workspace 为 0。

工作区布局为 level0 count 在前、bitmap 在后、其余 scan level 再后：三个区域均为 level0 kernel 参数的正向子区间，count kernel 显式接收 bitmap 指针并为各 GM tensor 设置精确可达长度，保证 torch_npu 缓存分配复用时 sanitizer 边界元数据正确（memcheck 越界告警 2010 → 0）。

##### 3. tilingkey规划策略：

本算子未使用数值 tilingkey。host 侧感知输入特征后直接选择 kernel 与路径参数，等价实现"host 信息感知 kernel 分支"：

| 判据（host 计算） | 派发路径 |
| --- | --- |
| 连续物理方向 且 256 对齐 且 unitCount ≥ 1024 | VF 向量计数（双缓冲 + 预取） |
| 连续但不满足上述条件 | 32-lane warp 协作扫描 |
| 跨步方向（ROW-CSC / COL-CSR） | SIMT strided ILP4 计数 + warp 协作提取 |
| Blocked-ELL | 混合进制坐标递增 kernel（blockSize 与 ellCols 均 ≤ 2^31-1 时块内坐标 uint32，否则 uint64） |

#### 3.2.2 kernel侧设计：

kernel 分 count、scan（多级前缀和）、convert 三类，另含 BELL 专用内核。count 阶段在 UB 用整数位掩码识别非零，经 StoreUnAlign/StoreUnAlignPost 直接写出每 unit 32 字节 bitmap，避免按位加权归约；complex64 对实虚位载体合并，保持 NaN/Inf/次正规数语义。跨步方向四路独立加载/累加减少计数依赖。scan 内部使用 warp shuffle 实现 inclusive prefix，各级独立 kernel 建立跨核阶段顺序，不使用不确定的跨核原子顺序。convert 每次重扫当前输入构造 prefix/bitmap：warp 先合并查找 32 个 unit 中的非空 unit，广播 prefix 后 32 lane 按 bitmap 协作提取，写入位置由 prefix 决定、不使用原子追加，保证重复运行顺序一致。流水屏障：PIPE_MTE2 顺序化搬运，V→MTE2 保护输入 slot 重用，V→MTE3 / MTE3→V 保护输出读写，事件仅在实际存在后继依赖时配对。

Python/ATen 侧：BSR 走 NPU block-mask kernel → INT8 mask CSR 转换 → NPU block-gather → SparseBSR 构造；BELL hook 走 block-mask → CSR block 结构 → 最大行宽归约（仅回传 width 标量）→ pattern kernel → aclsparse BELL Analysis/Convert，核心数据与输出 payload 不经 CPU。设备侧使用 NPU device guard 与输入设备当前 stream，直调 ACL kernel 前 flush torch_npu 任务队列保证执行顺序。适配层以上游 Torch Extension 框架交付（`sparse/densetosparse/torch_extension/`，cann_ops_sparse JIT wheel，导入即注册标准 ATen 实现）。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas 800I/T A2 | √（公共 Host 描述符路径） |
| Atlas A3 训练系列产品 | √（公共 Host 描述符路径） |
| Atlas 350 / Ascend 950PR（A5） | √（完整 Host + Kernel） |

## 算子约束限制

不支持任意 strides 输入、batch/hybrid sparse 输出（显式报错）；workspace 仅按查询值分配，Analysis 与 Convert 可复用，峰值不超过目标硬件 L2 容量（A5 实测最大 36,758,016 bytes < L2 134,217,728 bytes）；核心路径不创建与输入 Dense 或输出 payload 成比例的额外 Host 缓冲；仅声明二维 sparse conversion 与 dense_dim=0、sparse_dim=2 的 Python 适配。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | C++ gtest 121/121；ATen 126/126；任务包 200/200（未修改任务包 CPU Golden，values 按位检查、I32 结构检查）；vector 90/90；guard 40/40；strided tails 200/200；random strided 150/150；large guard 3/3；BELL 专项 1000/1000 | 任务书 §3.2 |
| 性能标准 | GPU 同口径对比 120/120 达 ≥0.3 验收线（GPU time / NPU time；交付环境 DevEnv_271283 实测 120/120 通过，19 项达到或超过 GPU，最弱项 P-03 CSC complex64 base0 ratio = 0.4672）；每 case 预热 10 次、采样 30 次，设备 Event，报告 median/p90；标杆取任务包 `baseline_results/gpu_full_results.tsv` | 任务书 §3.3 |
| 内存标准 | 峰值 workspace 36,758,016 bytes < 128MiB L2；Python 按 allocator baseline/peak/extra peak 分别统计 | 任务书内存条款 |
| sanitizer | memcheck 越界告警 2010 → 0（根因修复后，默认缓存序列 int8→fp32→complex64 验证）、racecheck 0、synccheck 0；交付环境 BELL 定向 memcheck 6/6 `No error detected` | 任务书 §3 |

CPU-fallback 断言：5 dtype × 4 入口（`to_sparse` / `_csr` / `_csc` / `_bsr`）共 20 项 + 未支持 dtype（int32）负例，合计 21 项全部通过——受支持 dtype 输出均在 NPU 且无 torch_npu fallback 告警，未支持 dtype 显式报错不静默降级（脚本 `test/densetosparse/python/nofallback_check.py`）。

测试脚本：`test/densetosparse/python/run_release_validation.sh`（输出独立证据目录，不覆盖历史记录）；公共 Host 交叉回归 `test_common_descriptors.cpp` 在 A2/A3/A5 构建配置下各 135 项通过（`COMMON_DESCRIPTOR_PASS 135`；A5 为真机 A5 库运行，A2/A3 为对应构建配置下的 Host 描述符回归，不声明 A2/A3 真机算子回归）。

交付验证环境与证据：Ascend 950PR（Atlas 350）/ CANN 9.1.0 / driver 25.7.rc1 / torch 2.7.1+cpu / torch_npu 2.7.1.post8，DevEnv_271283（容器 98c44e3e51d5），从零部署（`d86aa93` 基线 + 本分支单 commit 变更）实测：构建 rc=0，18 项子测试全 rc=0，精度合计 1930 项全部通过；完整证据包 SHA256 `3556f5e0336c8d6490bb0c1a13a21d75d5950e8ee95b73bf37d69c329a1982c3`。

## 兼容性分析

公共 Dense 描述符接受空矩阵（非空矩阵仍要求合法 values/ld）；公共 BELL 描述符增加尾块容量语义。两项公共 Host 修改已通过 A2/A3/A5 跨架构回归。公开 C++ API 签名不变；Python 输出为新建 NPU payload，不与输入 alias，COO indices 按 Torch 语义构造 int64（C++ 底层保持 I32）。未采用的实验（全 Dense 转置中间缓冲、header 融合、ILP8、过大 batch、WAW 预取、BELL 重排/二次幂特化）不在最终库中。
