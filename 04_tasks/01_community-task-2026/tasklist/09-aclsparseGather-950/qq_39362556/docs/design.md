# aclsparseGather 算子设计文档

# 需求背景（required）

## 需求来源

本文档对应昇腾社区任务“9 月社区任务-aclsparseGather 算子开发（950）”。任务要求参考
cuSPARSE `cusparseGather` 的接口语义，在 `cann/ops-sparse` 中完善
`aclsparseGather`，并统一交付 PyTorch/ATen NPU 适配、C++ Host、Ascend C Kernel、
测试与说明文档。

任务链接：<https://www.hiascend.com/activities/task-center/details/48de5591b88c4c8ca6cea9343313b871?menu=tasks>

目标环境如下：

| 项目 | 要求 |
| --- | --- |
| 硬件 | Ascend 950，DAV_3510，`arch35` |
| CANN | 9.1.0 及后续配套版本 |
| PyTorch | 2.7 及以上 |
| torch_npu | 26.0.0 及以后版本 |

## 背景介绍

Gather 从稠密向量 `Y` 中按照稀疏向量 `X.indices` 读取数据，并原地写入
`X.values`：

$$
X.values[i] = Y[X.indices[i] - idxBase],\quad i\in[0,nnz)
$$

该操作常用于词表、Embedding 或稀疏索引访问。任务给出的三个典型场景分别来自
Llama 3.1 70B、Qwen3-235B-A22B 和 DeepSeek-V3 的词表规模，特点是源向量较长、
索引数量较小、索引访问乱序且允许重复。

上游 `ops-sparse` 已有 `aclsparseGather` 的基础实现，可复用 Handle、DnVec/SpVec
描述符和 `arch35` SIMT 下发框架，但任务范围仍需补齐：

- `complex64` 数据类型；
- Ascend 950 的完整参数、边界、只读及 alias 语义；
- `torch.index_select(input, 0, index)` 到 `aten::index_select` 的 NPU Dispatcher 路径；
- 四种必选 dtype、I32、base 0/1 的 C++ 与 Python 测试；
- NPU Dispatch/Profiler、性能和内存验收材料。

# 需求分析（required）

## 需求描述

### C++ 接口

保持公开接口原型不变：

```cpp
aclsparseStatus_t aclsparseGather(
    aclsparseHandle_t handle,
    aclsparseConstDnVecDescr_t vecY,
    aclsparseSpVecDescr_t vecX);
```

只更新 `vecX.values`，`vecY.values` 和 `vecX.indices` 均为只读。调用沿用
`handle` 中的 stream 异步下发；`nnz=0` 直接成功返回，不启动 Kernel；不申请
workspace，也不执行 Host 同步。

### Python/ATen 接口

公开入口及映射为：

```text
torch.index_select(input, 0, index)
        -> aten::index_select
        -> PrivateUse1/NPU implementation
        -> aclsparseGather
        -> Ascend C Kernel
```

本任务只覆盖一维、`dim=0`（一维场景下接受等价的 `dim=-1`）、连续 Strided Tensor、
I32 index。输入和 index 必须位于同一 NPU Device。未支持的 dtype、shape、layout、
stride、index 类型或 device 组合返回明确错误，不进行 CPU fallback。

验收脚本还需要公开测试钩子：

```text
torch.ops.ops_sparse_test.gather_npu(values, indices, base)
```

该钩子与生产路径复用同一 NPU 实现，用于覆盖 base 0/1；公开
`torch.index_select` 按 PyTorch 语义使用 base 0。

## 需求拆解

1. 支持 `float16`、`bfloat16`、`float32`、`complex64`，输出逐 bit 精确一致。
2. 必选索引为 I32，支持 base 0 和 base 1；换算后 index 必须落在 `[0, size)`。
3. 支持不同 `size/nnz`、乱序、重复、首尾、尾块、`nnz=0/1` 和动态长度。
4. Host 校验空句柄/描述符/指针、描述符签名、dtype、index type/base、长度、地址范围、
   设备内存及未声明重叠。
5. Kernel 在 A5 Vector Core 上完成全部数据读取和写回，不做数值计算或类型转换。
6. PyTorch 层校验 Tensor 元数据、构造新输出并保持输入/index 不变；沿当前 NPU stream
   异步执行。
7. C++ UT/ST、ATen Dispatcher UT、Python 端到端 UT 与官方验收脚本均可复现。
8. 性能以任务书 GPU Event 基线为标杆，24 个必测组合均达到 GPU/NPU 不低于 0.3；
   同时报告 aclsparse Kernel 和 Python/ATen 端到端的 median/p90。
9. 不新增与输入规模线性相关的临时 Host/Device 内存；Gather 本身 workspace 为 0。

# 详细设计（required）

## 算子分析

### 数学公式

令 `base` 为 0 或 1，则每个输出元素互相独立：

$$
o_i = X.indices[i] - base
$$

$$
X.values[i] = Y[o_i]
$$

允许多个 `i` 对应同一个 `o_i`，因此重复索引只产生重复读取，不存在写冲突。

### 支持数据类型

| 层级 | values dtype | index dtype | index base |
| --- | --- | --- | --- |
| 任务必选 C++ 能力 | FP16、BF16、FP32、COMPLEX64 | I32 | 0、1 |
| PyTorch/ATen 能力 | Half、BFloat16、Float、ComplexFloat | Int | 0（公开入口） |
| 验收测试钩子 | Half、BFloat16、Float、ComplexFloat | Int | 0、1 |

Gather 仅复制 bit payload。Kernel 分别以 `uint16_t`、`uint32_t`、`uint64_t` 承载
2/4/8 字节元素，从而避免 BF16 或 complex64 的隐式转换，并原样保留 NaN payload、
INF、次正规数和正负零。

### 支持形状

- `vecY`：连续一维 `[size]`；
- `vecX.indices`、`vecX.values`：连续一维 `[nnz]`；
- `size >= 0`、`nnz >= 0`；C++ SpVec 约束 `nnz <= size`；
- PyTorch 允许重复索引导致 `index.numel() > input.numel()`，适配层将索引分成多个独立
  chunk 调用同一 Gather，不分配中间 Tensor；
- I32/base 0 的最大可寻址 `size` 为 `2^31`，I32/base 1 为 `2^31-1`。

## 算子实现

### Host 侧设计

#### 参数校验

Host 按以下顺序做同步、常数复杂度校验：

1. 校验 `handle`、`vecY`、`vecX` 非空及描述符签名；
2. 校验调用方已向 Handle 设置有效 stream；
3. 校验 values dtype 相同且属于声明集合；校验 index type/base；
4. 校验 `vecY.size >= vecX.size`、`nnz <= size` 及零长度语义；
5. 校验非空输入对应数据指针不为空；
6. 校验 I32/base 的最大寻址范围及字节数、地址末端不溢出；
7. 校验三个有效数据区位于当前 NPU Device；
8. 校验输出 `vecX.values` 不与 `vecY.values` 或 `vecX.indices` 重叠。

索引内容位于 Device，接口保持异步，因此逐元素越界检查作为调用方前置条件；非法索引
由异步错误协议报告，不为检查索引新增 D2H、同步或检查 Kernel。

#### Tiling 与下发

TilingData 只携带 `nnz`、dtype、index type、base 和 block 数。线程块大小固定为 256：

```text
numBlocks = min(ceil(nnz / 256), AIV core count)
```

`nnz=0` 在 Host 返回。其他场景将 indices、Y values、X values 和轻量 TilingData 直接
下发到调用方 stream。该路径不使用默认 workspace，不做 stream synchronize。

### Kernel 侧设计

使用 Ascend 950 SIMT Vector Function。每个线程以 grid-stride loop 处理若干输出：

```cpp
globalTid = threadIdx.x + blockIdx.x * blockDim.x;
gridStride = blockDim.x * gridDim.x;
for (i = globalTid; i < nnz; i += gridStride) {
    pos = int64(indices[i]) - int64(base);
    output[i] = input[pos];
}
```

实现使用无符号 payload 类型进行原始位复制。索引先提升到 `int64_t` 再减 base，避免 I32
算术边界问题。循环增量在最后一次迭代前判断剩余范围，避免极大 `nnz` 下的有符号溢出。

该访问模式每个输出只执行一次 index 读取、一次随机 values 读取和一次连续写出。任务给定
场景规模可以直接从 GM 访问；额外搬入 UB 会增加一次复制且随机地址难以合并，因此不申请
UB workspace 或全局临时缓冲。

### PyTorch/ATen 侧设计

1. 通过 `TORCH_LIBRARY_IMPL(aten, PrivateUse1, ...)` 注册 `aten::index_select`。
2. 校验一维、Strided、连续、同一 NPU、I32 index、支持 dtype 和 `dim=0/-1`。
3. 使用 `at::empty` 在相同 Device/dtype 上创建 `[index.numel()]` 的独立连续输出；输出不与
   input/index alias。
4. 通过 `torch_npu` Device Guard 选择输入 Device，并取得当前 NPU stream。
5. 每个 Device 缓存一个 aclsparse Handle，避免每次公开调用重复创建 Handle 及其默认资源；
   每个 Handle 以 mutex 保护跨线程设置当前 stream 和下发。
6. 每次调用用 RAII 创建/销毁 DnVec、SpVec 描述符，调用 `aclsparseGather`；仅下发到当前
   stream，不显式同步。
7. 为官方脚本注册 `ops_sparse_test::gather_npu` 的 PrivateUse1 和 Meta 实现。

PyTorch 输出本身属于公开 API 必需结果，不计作中间 workspace；除输出外不分配与输入规模
线性相关的 Device 或 Host 临时空间。

### 构建与加载

在根 CMake 中增加默认关闭的 `BUILD_TORCH_ADAPTER` 选项。开启时发现当前 Python 环境中的
PyTorch、torch_npu 头文件和动态库，构建 `libops_sparse_torch.so` 并链接
`ops_sparse`、`ascendcl`、PyTorch 和 torch_npu。Python 包仅负责定位并通过
`torch.ops.load_library` 加载该动态库，不实现任何计算 fallback。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Ascend 950（DAV_3510，`arch35`） | √ |

现有 A2/A3 公共描述符和其他架构实现保持共存。此次 A5 Kernel 与 Torch 适配只在对应构建
选项和目标 SoC 下启用。

## 算子约束限制

- 仅声明任务矩阵中的四种 values dtype、I32 和 base 0/1；上游已有的兼容能力不作为本任务
  验收声明。
- Python/ATen 仅覆盖一维连续 Strided Tensor 和 `dim=0`（或一维等价 `-1`）。
- 输入与 index 必须位于同一 NPU Device；CPU、跨 Device 和非连续输入明确报错。
- 换算 base 后的所有 index 必须位于 `[0, size)`；C++ 接口不为内容检查引入同步。
- `vecY`、indices、output 的生命周期必须覆盖异步执行完成时刻。
- `vecY.values`/`vecX.values` 和 `vecX.indices`/`vecX.values` 不得重叠。
- Gather 无 workspace、无 preprocess、无隐式 Host/Device 同步、无 CPU fallback。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度 | 四种 dtype 逐元素 bit-wise exact match；输入 Y、indices 和非目标数据不变 | 社区任务书 3.2 |
| 性能 | 24 个 P-01/P-02/P-03 × dtype × base 场景，GPU Event median / NPU 同范围耗时均不低于 0.3 | 社区任务书 3.3 |
| 内存 | Gather workspace 为 0；不申请线性临时内存；按官方脚本比较峰值 | 社区任务书 3.4 |
| 执行位置 | NPU Dispatch/Profiler 中无 CPU fallback | 社区任务书 3.5 |

### 测试设计

#### C++ UT/ST

- 四种任务 dtype、I32、base 0/1；
- `nnz=0/1`、不同动态规模、非整块尾部；
- 顺序、逆序、乱序、重复、首尾索引；
- 原始 bit payload，覆盖普通值、小值、正负、零、离群值、INF/NAN 和 complex64 实虚部；
- 相同输入重复运行，逐 bit 对比 CPU Golden；
- 回读 Y 和 indices 验证只读；
- 空对象/指针、签名、stream、dtype、index/base、长度、Device 和 alias 异常。

#### ATen 与 Python UT

- 分别调用 `torch.index_select`、`torch.ops.aten.index_select.default` 和官方测试钩子；
- 四种 dtype exact match、base 0/1、重复索引、空输出、index 长于 input；
- shape、layout、stride、dtype、index dtype、CPU/跨 Device、非法 dim/base 错误；
- 输出 shape/dtype/device/连续性及不与输入 alias；
- 输入/index 调用前后逐 bit 不变；
- 检查 PrivateUse1 注册，并用 NPU Profiler 核验调用栈中无 CPU 计算回填。

#### 性能与内存

使用任务包原始 `performance_cases.json`、固定 seed、预热 10 次、采样 30 次，记录 median
和 p90。官方测试钩子使用 NPU Event 测 Python/ATen 端到端；Profiler 导出的 Kernel 记录
用于报告 aclsparse Kernel 耗时。内存按任务包 GPU/NPU 收集与比较脚本执行，报告输入输出、
peak、extra peak 和 workspace。

实测数值只在 Ascend 950 配套环境执行后写入自测报告，不在设计阶段预填。

## 兼容性分析

- `aclsparseGather` 公开 C 函数原型保持不变，现有调用方源代码兼容。
- 复用公共 Handle、DnVec/SpVec 及 stream 约定，不改变描述符对外布局。
- 新增 complex64 分支，不影响已有 FP16/BF16/FP32 分支；保留上游既有兼容类型路径，但本任务
  只声明任务书矩阵。
- Torch 适配由独立、默认关闭的 CMake 选项构建；未安装 PyTorch/torch_npu 时，基础
  `ops_sparse` 构建不受影响。
- A2/A3 与 A5 代码先后合入时，以最新 `master` rebase 并完成 Gather 及公共描述符交叉回归。
