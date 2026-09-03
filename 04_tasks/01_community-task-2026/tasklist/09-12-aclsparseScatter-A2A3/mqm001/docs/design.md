# aclsparseScatter A2/A3 算子设计文档

# 需求背景（required）

## 需求来源

2026 年 9 月社区任务「aclsparseScatter 算子开发（A2/A3）」（任务编号 09-12）。任务书对标 cuSPARSE `cusparseScatter`（cuSPARSE 13.3 Update 1）接口语义，面向 Atlas A2/A3（DAV_2201，`arch22`）稀疏向量写入场景，使用 C++ Host 与 Ascend C Kernel 开发 `aclsparseScatter` 接口并合入 ops-sparse 仓。

## 背景介绍

### 上游实现现状

ops-sparse 仓 `sparse/scatter/arch22/` 已有 aclsparseScatter 的 AIV 实现（多核 nnz 分片 + tile 双缓冲 + run 连续段检测 + UB 暂存 MTE3 写出），但仅支持 **FP32 / index base 0 / I32**。

### 本任务定位

**能力扩展而非重写**：在既有实现基线上补齐五 dtype、index base 0/1，并打通 Python/ATen 适配层（`Tensor.index_copy_(0, index, source)` → `aclsparseScatter`），同步交付 C++ UT/ST 与 README。

# 需求分析（required）

## 需求描述

实现稀疏向量原地散布：按稀疏输入向量 `vecX` 的索引将其 values 原地写入稠密输入输出向量 `vecY`，接口不修改 `vecX`，核心计算在 NPU 完成（禁止 CPU fallback）。

## 需求拆解

1. values dtype 支持 `int8`、`float16`、`bfloat16`、`float32`、`complex64`（complex64 为必选能力），indices 仅 I32，index base 0/1；
2. 重复索引为非确定性 last-write-wins（不得实现为累加）；无重复索引时输出 bit-wise 一致；`nnz=0` 成功返回且不修改 `vecY`；
3. 复用 aclsparse Handle / SpVec / DnVec 描述符与资源管理接口，沿用调用方 stream 异步执行，不新增无必要 workspace 或 Host 同步；
4. Python/ATen 必选适配：公开入口 `Tensor.index_copy_(0, index, source)`，Dispatcher NPU 注册，不支持组合显式报错；
5. 性能：P-01（size=128256/nnz=8192）/P-02（151936/4096）/P-03（129280/7168）每个有效 case 性能倍率 ≥ 0.25（标杆 PyTorch GPU 设备 Event median 约 83–93μs）；
6. 内存：接口无 workspace，满足内存验收条件二（方案固有 workspace ≤ L2）。

# 详细设计（required）

## 算子分析

### 数学公式

$$Y[X.indices[i]-idxBase] = X.values[i],\quad i\in[0,nnz)$$

换算 base 后 indices 须位于 `[0, size)`（调用方契约）；重复索引时多写竞态，最终值为对应输入 values 之一。

### 支持数据类型

| 参数 | dtype | 说明 |
| --- | --- | --- |
| vecX.values / vecY | int8 / float16 / bfloat16 / float32 / complex64 | 两者必须一致；complex64 建模为 2×float32 交错连续存储，纯字节搬运 |
| vecX.indices | int32（I32） | 不支持 64I |

### 支持形状

一维：`indices[nnz]`、`values[nnz]`（SpVec，`nnz ≤ SpVec.size ≤ vecY.nums`）；`y[size]`（DnVec，连续）。`nnz ≥ 0`、`size ≥ 0` 动态变化，`nnz > 0` 时 `size > 0`；规模上界 INT32_MAX（同时受 Device 内存限制）。

## Host 侧设计

### 参数校验

校验链按 spec 错误码映射（空句柄→`ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR`；空描述符、非法 index type/base、五 dtype 外、dtype 不一致、`nnz>0 且 size=0`、`size>nums`、`nnz/size>INT32_MAX`、空数据指针、未声明 alias 重叠→`ACL_SPARSE_STATUS_INVALID_VALUE`）。非法 idxType/dtype 由上游的 NOT_SUPPORTED 修正为参数错误（以 spec 为准）。`nnz=0` 早退：不下发 kernel，vecY 不被触碰。共享句柄非线程安全，SetStream/Scatter 需调用方串行或加锁。

### Tiling 设计

`ScatterTilingData` 扩展：`idxBase`（运行时标量减法）、按 dtype 的对齐元素数合并为 `tileNnAligned`（int8:32 / fp16·bf16:16 / fp32:8 / complex64:4）、`yLen` 字节口径、多核均分 `coreNnzOffset/coreNnzCount`。核数动态获取（`GetCoreNumAiv`，clamp 64，禁止写死）；**单 TilingKey**（dtype 经 Host 侧模板分发、idxBase 运行量化、尾块统一处理，均不进 TilingKey 编码）。0 workspace。

### dtype 分发

`scatter_kernel_do` 按 `aclDataType` 分发 5 个 `KernelScatter<T, ELEM_BYTES>` 模板实例化（complex64 = `KernelScatter<float, 8>`，T=float、ELEM_BYTES=8）。

## Kernel 侧设计

### 基础执行路径

AIV 单 kernel（`KERNEL_TYPE_AIV_ONLY`，DAV_2201 通用 SIMD/MemBase 路线）：

1. **双缓冲 tile 循环**：MTE2 预取下一 tile 与当前 tile 的计算/写出重叠；尾块 `DataCopyPad` 按实际长度 blockLen（可非 32B 对齐）；
2. **run 连续段检测**：在换算 base 后的索引上做连续性判断，连续段走 `Copy<T>` UB→UB 段拷贝快路径（分块 ≤64 + `PipeBarrier<PIPE_V>`），单元素/断段走单元素写路径；
3. **GM 写**：一律 UB 暂存 + MTE3 `DataCopyPad` 任意偏移写出（禁止 `GlobalTensor::SetValue` 标量写 GM——对后续 MTE2 不可见的架构坑）；写后 `PipeBarrier<PIPE_ALL>`（修复上游 V→MTE3 可见性竞态）；
4. **小 dtype 处理**：`Copy<int8_t>` 1B mask 语义错误（910B3 实测），int8 一律 `SetValue` 标量兜底；complex64 的 Copy 路径即 fp32 mask（T=float），无需适配。

### 关键口径

- complex64 GM 绑定与偏移统一 **T 视角**（`yLen/sizeof(T)`、偏移 ×2），防绑定过短；
- 重复索引：多写竞态天然 last-write-wins，无原子无归约，不承诺确定输出；
- Device 侧索引内容越界不做运行时校验（调用方前置条件，与 cuSPARSE/upstream 一致）；确定报错由 ATen 适配层承接。

## Python/ATen 适配

`aten::index_copy_` PrivateUse1 Dispatcher NPU 注册（`libaclsparse_scatter_aten.so`），适配层校验 dim=0、dtype、shape、stride/layout、device（`c10_npu::NPUGuard` + 同 device 一致性校验，越界确定 RuntimeError），越界/base 换算结果记忆化（version-counter 键 + 持源张量防地址复用），无 CPU fallback。

## 支持硬件

| 型号 | 支持 | 架构 |
| --- | --- | --- |
| Atlas A2 训练系列（910B3/910B4） | √ | DAV_2201 / arch22 |
| Atlas A3 | √ | DAV_2201 / arch22 |
| A5（950 系列） | ×（独立 PR 合入 arch35） | — |

CANN 9.1.0 及后续配套版本。

## 算子约束限制

- values 与 y dtype 必须一致；indices 仅 I32；`nnz ≤ SpVec.size ≤ vecY.nums`；
- `nnz`/`size` 上界 INT32_MAX（同时受 Device 内存限制）；
- 重复索引输出非确定（last-write-wins）；无重复时 bit-wise 一致；
- 未声明 alias 的输入输出重叠返回参数错误；`vecX.values/indices` 只读；
- 共享句柄非线程安全；描述符与数据指针在异步执行完成前由调用方保持有效。

# 可维可测分析

## 精度测试方案

以 CPU Golden 为基准（参考 cuSPARSE 语义与《生态算子开源精度标准》实验性标准）。判定口径按用例 indices 是否含重复**静态分派**：

- 无重复 → oracle 对比 bit-wise exact（golden dtype：int8 按 bit / fp16·bf16→fp32 / fp32→fp64 / complex64→complex128）；
- 有重复 → invariant 三断言（冲突值 ∈ 对应 values 集合 + 未写入位置 bit 保持 + vecX 只读）；
- 覆盖五 dtype × base 0/1 × 乱序/重复/尾块/nnz=0/1、INF/NAN/subnormal、complex64 专项；
- 自验结果：官方 accuracy_cases 200/200 全过（ATK 评委口径原样命令执行，含 2251 个冲突位集合校验）；黑盒 ST 79/79、UT 84、白盒 658、torch 端到端 69/69。

## 性能测试方案

- 正式采样集：官方 performance_cases 230 条中 **130 条 duplicates=false**（30 P 场景 + 100 extra；dup=true 100 条不用于倍率统计——重复索引计时引入非确定竞态）；
- 口径：预热 ≥10 / 采样 30，NPU 设备 Event 同调用范围（PyTorch GPU 设备 Event 为独立来源，分别统计不混用），报告中位数与 p90，不计首次编译/H2D/初始化；固定随机种子、复用描述符；
- 结果：130/130 倍率 ≥ 0.25（min 0.2540 / 中位 0.3963 / max 0.5785；P-01/02/03 min 0.433–0.448）。

## 内存测试方案

接口无 workspace（无 pBuffer 概念），内存验收条件二平凡满足；实测 extra_peak_reserved 130/130 = 0；与 GPU 官方基线对比 extra_peak NPU≤GPU 125/130（最大 +12.5% ≪ 50%）。

## 无 CPU fallback 验证

流阻塞法（大流量 D2D + 事件阻塞 handle 流：阻塞期 vecY 不变、提交仅 ~0.03ms 入队）+ NPU Profiler（P-01 稳态每迭代仅 2 kernel：ZerosLike + scatter_custom，AICore 利用 36.7%，无 CPU 计算路径）+ dispatch dump（适配层接管 `aten::index_copy_` 后原 kernel 转 inactive）。

## 兼容性分析

- **源代码兼容**：公开接口 `include/cann_ops_sparse.h` 仅注释更新，A5（arch35）路径不受影响；A2/A3 与 A5 公共逻辑可共存，先后合入时后续 PR 处理公共 Host 冲突并交叉回归；
- **上游既有用例回归**：上游 56 个 FP32 用例 96/96 全过（含修复上游 V→MTE3 可见性竞态后原 large-1M-contiguous 非确定失败用例转稳）；
- PyTorch 2.7+ / torch_npu 26.0.0+ 适配验证通过（torch 2.7.1 + torch_npu 2.7.1.post8）。
