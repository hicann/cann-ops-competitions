# 【社区任务】torch_sparse 工具类接口（non_diag_mask / random_walk）算子设计文档

> 任务：CANN 训练营北京邮电大学 - torch_sparse 工具类接口开发（950）
> 提交目录：`04_tasks/01_community-task-2026/tasklist/09-41-torch_sparse_utils-950/vanerin/docs/design.md`
> 硬件：Ascend 950PR（arch35，NPU_ARCH=dav-3510）
> 文档版本：v1.0（2026-09-08）

# 一、需求背景

## 1.1 需求来源

本设计来源于 CANN 社区任务《CANN训练营北京邮电大学 - torch_sparse 工具类接口开发（950）》（社区任务清单第 41 项，见
`04_tasks/01_community-task-2026/docs/README.md`）。任务目标：在 Ascend 950 系列（arch35）上，使用 Ascend C、C++ 与 PyTorch 扩展
实现 PyG（PyTorch Geometric）稀疏图工具链依赖的两个 `torch_sparse` 工具类接口，功能与接口对齐开源对标库
[rusty1s/pytorch_sparse](https://github.com/rusty1s/pytorch_sparse)，验收通过后提交至昇腾算子开源仓
[cann/ops-gnn](https://gitcode.com/cann/ops-gnn)。

| 接口 | 用途 |
| --- | --- |
| `torch_sparse::non_diag_mask(row, col, M, N, k)` | 为 COO 边生成 bool 掩码，服务于稀疏图对角线设置（`SparseTensor.set_diag` / `fill_diag`） |
| `torch_sparse::random_walk(rowptr, col, start, walk_length)` | 在 CSR 图上执行均匀随机游走，服务于图采样（`SparseTensor.random_walk`） |

本算子为**新增生态算子**：ops-gnn 仓库与 CANN 内置算子中均不存在这两个接口的历史 TBE 实现，不涉及 TBE 算子路径迁移。
对标库信息以 pytorch_sparse 开源实现为准（本次核对版本 0.6.18，本地源码 commit `2be752e` "Add PyTorch 2.11 support (#409)"），
涉及文件名如下：

| 标杆文件 | 内容 |
| --- | --- |
| `torch_sparse/diag.py` | `set_diag` / `fill_diag` / `remove_diag` Python 层实现，内部调用 `torch.ops.torch_sparse.non_diag_mask` |
| `torch_sparse/rw.py` | `SparseTensor.random_walk` Python 层实现，内部调用 `torch.ops.torch_sparse.random_walk` |
| `csrc/cpu/diag_cpu.cpp` | `non_diag_mask` CPU 标杆实现（按边写入掩码槽位） |
| `csrc/cuda/diag_cuda.cu` | `non_diag_mask` CUDA 标杆实现（逐线程按边写入掩码槽位） |
| `csrc/cpu/rw_cpu.cpp` | `random_walk` CPU 标杆实现（先生成 CPU 随机数矩阵，再顺序游走） |
| `csrc/cuda/rw_cuda.cu` | `random_walk` CUDA 标杆实现（先生成显存随机数矩阵，再逐线程游走） |

术语说明（本章及后续章节统一使用）：

- **COO**：坐标格式（Coordinate format），用 `row`/`col` 两个等长 int64 一维数组表示稀疏矩阵的非零元位置；`E` 为边（非零元）个数。
- **CSR**：压缩稀疏行格式（Compressed Sparse Row），用 `rowptr`（长度 `num_nodes+1` 的行偏移数组）与 `col`（列索引数组）表示稀疏图；`nnz` 为非零元个数。
- **arch35 / dav-3510**：Ascend 950PR 对应的芯片架构标识与 NPU 架构编译参数。
- **SIMT**：单指令多线程（Single Instruction Multiple Threads）执行模型，Ascend C 在该架构上提供的 kernel 编程形式，同一 block 内多个线程执行同一段代码但处理不同数据元素。
- **PrivateUse1**：PyTorch 为第三方设备后端保留的分发键（dispatch key）；torch_npu 将 NPU 设备映射到该键。
- **dispatcher（分发器）**：PyTorch 按算子 schema 与输入张量设备/类型选择具体实现的机制。
- **D2H**：设备到主机（Device to Host）的内存拷贝方向。
- **UB**：统一缓冲区（Unified Buffer），AI Core 上用于数据搬运中转的片上存储。
- **L2 Cache**：芯片上的二级缓存，任务书用它作为固有 workspace 容量的上界参考。

验收口径：以 **PyTorch 层接口**（`torch.ops.torch_sparse.*` 与 `SparseTensor` 前端）的功能、精度、性能为唯一验收基准；
交付形态采用任务书 §2.2 规定的 **torch 接口工程化开发模式**，落在 `ops-gnn/experimental/torch_sparse_utils/`。
边界约束：仅新增 ops-gnn 的 NPU 路径，与 ops-sparse 的 `aclsparse*` 接口解耦，不修改 ops-sparse，NPU 输入不得回退 CPU 计算。

其他参考来源：

- 精度标准：《[生态算子开源精度标准（experimental）](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md)》
- PyG 稀疏张量文档：<https://pytorch-geometric.readthedocs.io/en/stable/advanced/sparse_tensor.html>
- ops-gnn 仓库既有算子目录规范（`csrc/npu/`、`experimental/` 等）

## 1.2 背景介绍

### 1.2.1 torch_sparse 工具类接口现状分析

#### 1.2.1.1 标杆算子支持的数据类型和数据格式

对标库 `pytorch_sparse` 的两个接口均为**离散输出算子**（bool 掩码、int64 路径下标），参数、数据类型与约束如下。

`non_diag_mask(row, col, M, N, k)`：

| 参数 | 参数含义 | 数据类型 | 支持数据类型 | 约束 | 形状 |
| --- | --- | --- | --- | --- | --- |
| row | COO 行索引 | tensor | int64 | 与 col 等长 | `[E]` |
| col | COO 列索引 | tensor | int64 | 与 row 等长 | `[E]` |
| M | 矩阵行数 | int | int64 | ≥0 | 标量 |
| N | 矩阵列数 | int | int64 | ≥0 | 标量 |
| k | 对角线偏移 | int | int64 | 任意整数，`k=0` 为主对角 | 标量 |
| 返回值 | 非目标对角掩码 | tensor | bool | 与 `set_diag` 的槽位语义配套 | `[E+num_diag]` |

`random_walk(rowptr, col, start, walk_length)`：

| 参数 | 参数含义 | 数据类型 | 支持数据类型 | 约束 | 形状 |
| --- | --- | --- | --- | --- | --- |
| rowptr | CSR 行指针 | tensor | int64 | 首项 0、末项 `nnz`、单调不减 | `[num_nodes+1]` |
| col | CSR 列索引 | tensor | int64 | 值域 `[0, num_nodes)` | `[nnz]` |
| start | 游走起点 | tensor | int64 | 值域 `[0, num_nodes)` | `[num_starts]` |
| walk_length | 游走步数 | int | int64 | ≥0 | 标量 |
| 返回值 | 游走路径 | tensor | int64 | 首列等于 start | `[num_starts, walk_length+1]` |

数据格式说明：两接口只处理一维索引张量，不涉及 NCHW/NHWC 等图像数据排布格式；输入须为连续（contiguous）内存布局。
输入设备为 NPU 时由本设计的 PrivateUse1 实现承接，CPU/CUDA 前端保持上游实现不变。

#### 1.2.1.2 标杆算子全面描述

**（1）`non_diag_mask` 的标杆语义**

标杆的掩码不是"逐边布尔比较"，而是**插入槽位编码**：CUDA/CPU 实现先把长度 `E+num_diag` 的输出清零，再对每条边按分支把 `true`
写入某个位置（`idx`、`idx+r`、`idx+r+1` 或 `idx+num_diag`）：

```cpp
// csrc/cpu/diag_cpu.cpp（k >= 0 分支，CUDA 实现同构）
if (r + k >= N)          mask[i + num_diag] = true;
else if (r + k > c)      mask[i + r]        = true;
else if (r + k < c)      mask[i + r + 1]    = true;
```

该编码只在 `row` 非降序、且目标对角线上的已有边已被 `remove_diag` 移除时成立：此时 `mask` 中 `true` 的个数恰为 `E`，
`false` 的个数恰为 `num_diag`，`false` 的位置即新插入的目标对角元在拼接后数组中的位置，
`diag.py` 据此用 `new_row[mask] = row`、`new_row[inv_mask] = diag` 重建 COO。

任务书 §2.1 对该接口给出的输出定义是**逐边布尔比较 + 尾部补 False**：

```text
num_diag = max(0, k >= 0 ? min(M, N-k) : min(M+k, N))
mask[i]  = (row[i] + k != col[i])     , 0 <= i < E
mask[i]  = False                      , E <= i < E + num_diag
```

两种编码的真值个数相同（`E` 个 true、`num_diag` 个 false），区别在于 true 的位置排列：任务书语义把 `E` 个 true 放在前 `E` 位、
`num_diag` 个 false 放在尾部。本设计以任务书 §2.1 的逐边语义为**公共接口验收基准**（验收 golden 与之一致），
上游 `diag.py` 的位置重建语义差异在 3.2.2.3 与 3.4 中单列说明。

`num_diag` 的边界处理也存在差异：标杆公式 `k < 0 ? min(M+k, N) : min(M, N-k)` 未做下界截断，`k` 越出矩阵范围时为负值；
本设计按任务书取 `max(0, ...)`，负值按 0 处理（此时输出长度为 `E`）。

**（2）`random_walk` 的标杆语义**

标杆 CPU 与 CUDA 实现均为"先生成随机数矩阵，再逐起点顺序游走"：

```cpp
// csrc/cpu/rw_cpu.cpp
auto rand = torch::rand({start.size(0), walk_length}, float_options);   // CPU 随机数矩阵
...
cur = col_data[row_start + int64_t(rand_data[n*walk_length + l] * (row_end - row_start))];
```

```cpp
// csrc/cuda/rw_cuda.cu
auto rand = torch::rand({walk_length, start.size(0)}, float_options);   // 显存随机数矩阵
auto out  = torch::full({walk_length + 1, start.size(0)}, -1, start.options());
... out[(l+1)*numel + thread_idx] = cur;
return out.t().contiguous();                                            // 转置回 [S, L+1]
```

要点：

- 采样下标由 float32 随机数与出度的乘积截断得到（`int64(rand * degree)`），随机数来自框架默认随机数发生器的一次性张量填充。
- 死端（出度为 0 的节点）没有专门处理：`row_end - row_start == 0` 时下标仍为 `row_start`，会读取该行之外的 `col` 元素，
  标杆未对该行为给出定义；任务书要求"未写入位置初值为 -1"。
- CUDA 路径输出先按 `[L+1, S]` 布局生成再转置连续化，产生一次额外的显存读写。

任务书 §2.1 对随机性的要求是：均匀选择邻边；单邻居图必须逐元素确定性一致；多邻居图须记录 RNG 算法、seed 与消费顺序，
只有三端 RNG 对齐时才做逐元素比较；NPU 输入不得回退 CPU。

**（3）两个接口与其它 torch_sparse 算子的关系**

`non_diag_mask` 与 `random_walk` 都是 `SparseTensor` 前端的**被调算子**（分别被 `diag.py`、`rw.py` 调用），
本身不做稀疏结构变换，输出为稠密索引/掩码张量；因此本设计不涉及 `scatter`/`gather`/`SpMM` 等算子的选型问题，
与 ops-sparse 的 `aclsparse*` 接口也没有调用关系。

#### 1.2.1.3 标杆算子实现流程图

本算子无 TBE 历史版本，标杆以 pytorch_sparse 的 CPU/CUDA 实现为准。标杆整体流程如下：

```text
┌──────────────────────────────────────────────────────────────────────────┐
│                  pytorch_sparse：non_diag_mask                            │
├──────────────────────────────────────────────────────────────────────────┤
│ 1. Python 层（diag.py）：remove_diag 先移除目标对角线上的已有边             │
│ 2. 计算 num_diag = k<0 ? min(M+k,N) : min(M,N-k)（未做下界截断）           │
│ 3. 分配 mask = zeros([E+num_diag], bool)                                  │
│ 4. 逐边分支写 true（槽位编码）：idx / idx+r / idx+r+1 / idx+num_diag        │
│ 5. 返回 mask；diag.py 用 mask 与 inv_mask 重建 new_row / new_col / value    │
└──────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────┐
│                  pytorch_sparse：random_walk                              │
├──────────────────────────────────────────────────────────────────────────┤
│ 1. 结构检查：rowptr/col/start 均为一维                                   │
│ 2. 随机数矩阵：CPU -> rand([S, L])；CUDA -> rand([L, S]) 显存张量          │
│ 3. out = full([...], -1)                                                  │
│ 4. 逐起点顺序（CPU）/ 逐线程（CUDA，THREADS=1024）游走：                    │
│      begin=rowptr[cur], end=rowptr[cur+1]                                 │
│      idx = int64(rand * (end-begin))，cur = col[begin+idx]                 │
│      写 out；出度为 0 时无专门分支（读越界 col，行为未定义）                 │
│ 5. CUDA：out 由 [L+1,S] 转置为 [S,L+1] 并连续化                            │
└──────────────────────────────────────────────────────────────────────────┘
```

# 二、需求分析

## 2.1 外部依赖分析

| 依赖 | 用途 | 是否运行时依赖 |
| --- | --- | --- |
| PyTorch + torch_npu | 提供张量、设备内存、`PrivateUse1` 分发与默认随机数发生器；PyTorch 层接口为验收基准 | 是 |
| CANN Toolkit（Ascend C / bisheng 编译器） | 编译并运行 arch35 SIMT kernel | 是 |
| pytorch_sparse（上游） | 仅作标杆对拍、CPU golden 参考与 `SparseTensor` 前端集成验证 | 否（仅验证期） |
| Python 3.12 | 执行测试与基准脚本 | 否（仅验证期） |

无其他第三方运行时依赖；测试脚本不依赖 ATK 或本地数据文件，全部输入在内存中构造。

## 2.2 内部耦合模块

本设计在 ops-gnn 仓库内新增 `experimental/torch_sparse_utils/` 目录，除目录规范外与仓库既有算子无代码耦合：

| 模块 | 耦合内容 | 影响 |
| --- | --- | --- |
| `experimental/torch_sparse_utils/`（新增） | `op_kernel/`、`op_host/`、`torch_extension/`、`tests/`、`CMakeLists.txt`、`setup.py`、`README.md` | 新增目录，自包含构建 |
| `experimental/torch_sparse_utils/op_kernel/arch35/aiv_launch_utils.h` | AIV 核数解析工具（与 `csrc/npu/common/aiv_launch_utils.h` 逻辑一致） | `experimental` 目录要求自包含，保留本地副本，不改动原文件 |
| `torch_sparse` 算子命名空间 | 本库以 `TORCH_LIBRARY` 定义 schema，并注册 `PrivateUse1`（NPU）实现与 `CPU` 拒绝内核 | schema 与上游 `torch::RegisterOperators` 推断出的签名一致；上游同名算子的不分设备注册可能与 NPU 分发冲突，处理方式见 5.2 |
| ops-gnn `csrc/`、`python/`、`csrc/pybind.cpp` | 无改动 | 不影响既有算子入口与构建 |
| ops-sparse（`aclsparse*`） | 无调用、无签名依赖 | 完全解耦 |

## 2.3 需求模块设计

### 2.3.1 Ascend C 算子原型

PyTorch 层接口（`torch_extension/torch_sparse_npu/api.py`）：

```python
def non_diag_mask(row: Tensor, col: Tensor, M: int, N: int, k: int = 0) -> Tensor: ...
def random_walk(rowptr: Tensor, col: Tensor, start: Tensor, walk_length: int) -> Tensor: ...
```

torch 算子 schema（`torch_extension/torch_sparse_ops.cpp`）：

```text
torch_sparse::non_diag_mask(Tensor row, Tensor col, int M, int N, int k) -> Tensor
torch_sparse::random_walk(Tensor rowptr, Tensor col, Tensor start, int walk_length) -> Tensor
```

Host 侧入口（`op_host/`）：

```cpp
torch::Tensor non_diag_mask_npu(torch::Tensor row, torch::Tensor col,
                                int64_t M, int64_t N, int64_t k);
torch::Tensor random_walk_npu(torch::Tensor rowptr, torch::Tensor col,
                              torch::Tensor start, int64_t walkLength);
```

Kernel 侧入口（`op_kernel/arch35/`，由 host 侧以 SIMT 方式启动）：

```cpp
void LaunchNonDiagMaskKernel(const int64_t* row, const int64_t* col, uint8_t* mask,
                             int32_t* violation, int64_t E, int64_t numDiag,
                             int64_t M, int64_t N, int64_t k, aclrtStream stream);
void LaunchRandomWalkKernel(const int64_t* rowptr, const int64_t* col, const int64_t* start,
                            int64_t* out, uint64_t startCount, uint32_t walkLength,
                            uint64_t seed, uint64_t offset, aclrtStream stream);
```

张量参数与属性约束同 1.2.1.1。两接口的输出形状分别为 `[E+num_diag]`（bool）与 `[num_starts, walk_length+1]`（int64）。

### 2.3.2 Ascend C 算子相关约束

1. 输入索引张量固定 INT64；`mask` 输出固定 BOOL（kernel 按 `uint8_t` 写入 0/1），`out` 输出固定 INT64。
2. 索引张量限定一维；`row` 与 `col` 等长，`rowptr` 长度至少为 1；同一调用的全部输入须位于同一 NPU 设备。
3. `M`、`N`、`walk_length` 必须非负；`k` 可为任意 int64。
4. `walk_length` 与 `nnz` 的启动参数上限为 `2**32-1`（超出时抛出明确错误，不静默截断）。
5. 索引张量须连续（contiguous）；非连续输入在 host 侧明确报错，不做隐式拷贝（测试脚本在构造输入时保证连续）。
6. 不支持 BOOL/COMPLEX/Float8 等索引 dtype；不支持 batch 维与广播。
7. CPU 输入明确报错"仅支持 NPU"，不做 CPU 回退。
8. 非法输入必须报错，不得静默把错误结果伪装成合法输出（-1 仅表示合法死端之后的未写入位置）。

# 三、需求详细设计

## 3.1 调用方式

本算子采用 **PyTorch 框架调用**方式，调用链自上而下：

```text
用户 Python 调用
  torch.ops.torch_sparse.non_diag_mask / random_walk
  torch_sparse_npu.non_diag_mask / random_walk        （便捷封装）
  SparseTensor.set_diag / fill_diag / random_walk     （上游 Python 前端）
  ▼
PyTorch dispatcher（按设备选择分发键）
  PrivateUse1（NPU） → 本库 NPU 实现
  CPU               → 本库拒绝内核（明确报错，不回退）
  CUDA              → 上游 pytorch_sparse 实现（如已安装）
  ▼
Host 侧（op_host/）
  non_diag_mask.cpp：结构校验 → num_diag 计算 → 输出分配 → 越界缓冲 → 启动 kernel → 单次 D2H 校验
  random_walk.cpp  ：结构校验 → CSR/值域校验 → 快速路径 → 取 Philox 状态 → 启动 kernel
  ▼
Kernel 侧（op_kernel/arch35/）
  non_diag_mask_kernel.cpp：SIMT 逐元素比较 + 尾部补 false + 融合值域校验
  random_walk_kernel.cpp  ：SIMT 每线程一条游走 + kernel 内 Philox4x32-10 采样 + 死端 -1 填充
```

## 3.2 需求总体设计

### 3.2.1 host 侧设计

两接口的 host 侧职责是**参数与结构校验、形状推导、输出分配、kernel 启动与错误上报**；计算全部在 kernel 内完成，
host 侧不复制输入数据、不做全量规约检查。

按 checker 位置与错误时机，校验项分布如下：

| 校验项 | 实现位置 | 错误方式 |
| --- | --- | --- |
| 张量已定义、NPU 张量、INT64、一维、连续 | host（`TORCH_CHECK`） | 调用时立即抛出 `RuntimeError` |
| `row`/`col` 等长、`rowptr` 至少一项、`M`/`N`/`walk_length` 非负 | host（`TORCH_CHECK`） | 调用时立即抛出 `RuntimeError` |
| `walk_length`、`nnz` 的 32 位上限 | host（`TORCH_CHECK`） | 调用时立即抛出 `RuntimeError` |
| COO `row`/`col` 值域（`row∈[0,M)`、`col∈[0,N)`） | mask kernel 内逐元素融合，写 `violation` 标记 | host 单次 D2H 回读后抛出 `RuntimeError` |
| CSR 首项为 0、末项等于 `nnz`、单调不减 | host（NPU 张量算子 + `item()`） | 调用时（同步读取时）抛出 `RuntimeError` |
| `col`、`start` 值域 `[0, num_nodes)` | host（NPU 张量算子 + `item()`） | 调用时（同步读取时）抛出 `RuntimeError` |

`non_diag_mask` 的 host 侧流程：

1. **结构校验**：`row`/`col` 已定义、为 NPU 张量、同一设备、INT64、一维、等长、连续，`M`、`N` 非负。
2. **形状推导**：`numDiag = max(0, k >= 0 ? min(M, N-k) : min(M+k, N))`；输出长度 `E + numDiag`。
3. **空边快速路径**：`E == 0` 时直接返回 `zeros([numDiag], bool)`（此时不存在越界可能，无需启动 kernel）。
4. **越界标记缓冲**：`violation` 为 `int32[1]`，使用静态张量缓存复用（首次调用分配），每次调用以 `aclrtMemsetAsync` 驱动级清零。
5. **启动 kernel**：以 `aclrtStream` 为当前 NPU 流，网格步长方式遍历 `E + numDiag` 个元素。
6. **错误上报**：kernel 完成后以**单次 `aclrtMemcpy` D2H** 回读 `violation`；非 0 时抛出越界错误（错误信息包含合法值域）。

`random_walk` 的 host 侧流程：

1. **结构校验**：`rowptr`/`col`/`start` 已定义、NPU 张量、同一设备、INT64、一维、连续；`rowptr.numel() >= 1`；
   `walk_length >= 0` 且不超过 `2**32-1`；`col.numel()` 不超过 `2**32-1`。
2. **CSR 与值域校验**：`rowptr[0] == 0`、`rowptr[-1] == col.numel()`、`rowptr` 单调不减（`narrow` + `ge` + `all`）、
   `col ∈ [0, num_nodes)`、`start ∈ [0, num_nodes)`。该校验由 NPU 张量算子完成，每项引入一次同步读取（`item()`）。
   与 `non_diag_mask` 不同，随机游走 kernel 会以 `col`/`start` 的值作地址下标，越界索引可能造成非法访存，
   因此校验放在 host 侧、在任何启动之前完成。
3. **快速路径**：`start_count == 0` 时返回 `empty([0, walk_length+1])`（图内容仍需先校验）；
   `walk_length == 0` 时返回 `start.reshape([S, 1]).clone()`（仅首列）。
4. **随机数状态获取**：`walk_length > 0` 且 `S > 0` 时，持锁从 torch_npu 默认 NPU 随机数发生器取 `philox_engine_inputs(4)`，
   得到 `(seed, offset)`。
5. **启动 kernel**：输出 `[S, walk_length+1]`，每线程处理一条完整游走。
6. **错误上报**：非法 CSR/起点/步长在执行任何 kernel 之前报错；不存在"把非法输入静默转成 -1"的路径。

#### 3.2.1.1 分核策略

优先使用满核的原则，采用"每 block 固定线程数 + 网格步长遍历"的 SIMT 映射：

```text
threadsPerBlock = 256
total_mask      = E + numDiag
total_walk      = startCount
needCoreNum     = ceil(total / threadsPerBlock)
coreNum         = min(needCoreNum, GetCoreNumAiv())      # 平台 API 获取，不硬编码
globalIdx       = blockIdx * threadsPerBlock + threadIdx
stride          = coreNum * threadsPerBlock
```

- 一个启动 block 对应一个 AIV 核，block 内固定 256 个线程（`LAUNCH_BOUND(256)`，`Dim3{256}`）。
- 需要处理的元素数超过物理核数时，由网格步长循环承担剩余迭代，不需要大小核区分。
- 核数由 `torch_sparse_utils::ResolveAivCoreNum()` 通过平台信息 API（`GetCoreNumAiv()`）获取，返回 0 时退化到 1，
  不写死硬件核数。
- `non_diag_mask` 在 `E + numDiag == 0`（空输入）时不启动 kernel；`random_walk` 在 `S == 0` 或 `walk_length == 0` 时走 host 快速路径。
- `random_walk` 的并行粒度是一条游走（`start` 的一行），不是一步采样：一条路径内部存在数据依赖（下一步的当前节点是上一步的输出），
  因此核间无需同步。

#### 3.2.1.2 数据分块和内存优化策略

充分使用片上资源的原则，两个 kernel 的资源占用如下：

| 项 | non_diag_mask | random_walk |
| --- | --- | --- |
| 动态 workspace | 无（launch 传 `nullptr`） | 无（launch 传 `nullptr`） |
| UB 使用 | 不使用（GM 直接读写） | 不使用（GM 直接读写；Philox 状态保存在寄存器） |
| 寄存器状态 | 循环变量与标量参数 | `PhiloxState`（key/counter/values/lane） |
| 输出预清零 | 不需要（每个输出位置都由 kernel 写入） | 不需要（首列、每步结果、死端后缀均显式写入） |
| 额外常驻缓冲 | `violation`：`int32[1]`（静态缓存，4 字节） | 无 |

不使用 UB 中转与 double buffer 的原因：两接口的每个输出元素都只依赖少量标量/索引输入，不存在需要在片上重复使用的
规整数据块；GM 直接读写避免了搬运与容量计算的复杂度。由于没有 UB 分块，也不存在"按 UB 预算推导单核内分块行数"的需求。

已实施的三项内存/时延优化（`op_host/non_diag_mask.cpp`）：

1. **越界标记清零**：由 `torch::zeros`（额外一次填充 kernel）改为 `aclrtMemsetAsync`（驱动级 memset），省去一次 kernel 下发。
2. **越界回读**：由多次 `tensor.item()`（每次附加一次默认流同步）改为**单次 `aclrtMemcpy` D2H**，把同步次数压缩到 1 次。
3. **越界缓冲复用**：`violation` 张量静态缓存，避免每次调用重新分配。

上述改动把固定性能 case（`M=N=10000, E=100000, k=0`）的接口 Event wall time 从约 36 µs 降到约 25 µs。

`random_walk` 的输出张量为 `8 * S * (L+1)` 字节（固定 case 约 270 KB）；host 侧的 CSR/值域校验会产生若干临时布尔张量
（`narrow`/`ge`/`lt`/`any` 等），规模为 `O(num_nodes)` 或 `O(nnz)`。固定 case 实测的额外峰值分配约 17 MB，
包含输入搬运与校验临时张量的分配器开销，详见 5.1。

#### 3.2.1.3 tilingkey 规划策略

**本设计不使用 tiling key。** 理由：

1. 两个接口的索引 dtype 唯一（INT64），输出 dtype 唯一（BOOL / INT64），不存在按 dtype 分派的 kernel 变体；
2. 没有广播、没有多 batch 维，不存在按 shape 组合分派的变体；
3. 所有形状标量（`E`、`numDiag`、`M`、`N`、`k`、`S`、`L`、`seed`、`offset`）都以 kernel 实参直接下发，
   kernel 内的分支只由运行期数据决定（例如"当前线程下标是否小于 `E`"、"出度是否为 0"、"当前节点是否为死端"），
   不改变指令路径结构，因此也不需要 tiling key 选择。
4. 核数、`threadsPerBlock` 等启动参数在 host 侧计算后通过启动配置下发，不写入 TilingData 结构。

### 3.2.2 kernel 侧设计

两接口的 kernel 均采用 SIMT 形式实现：`__attribute__((aiv)) __global__ __aicore__` 函数通过
`AscendC::Simt::VF_CALL<...>(Dim3{256}, ...)` 调用 `__simt_vf__` 核函数，核函数内用
`GetBlockIdx()`/`GetBlockNum()`/`GetThreadIdx()`/`GetThreadNum()` 计算全局线程号。

#### 3.2.2.1 kernel 伪代码描述

```text
NonDiagMaskKernelSimt(row, col, mask, violation, E, numDiag, M, N, k):
    total = E + numDiag
    for i in grid_stride(globalThreadIdx, totalThreadNum, total):
        if i < E:
            r = row[i]; c = col[i]
            if r < 0 or r >= M or c < 0 or c >= N:
                violation[0] = 1                 # 所有写者写入同一值，普通 32 位 store 即可，无需原子
            mask[i] = (r + k != c) ? 1 : 0       # 逐边比较，bool 以 uint8 表示
        else:
            mask[i] = 0                          # 尾部 numDiag 位固定为 false

RandomWalkKernelSimt(rowptr, col, start, out, startCount, walkLength, seed, offset):
    for walk in grid_stride(globalThread, globalStride, startCount):
        rng = MakePhiloxState(seed, offset, walk)      # key = seed^offset 拆 32 位；counter = (0,0,walk低32,walk高32)
        current = start[walk]
        base = walk * (walkLength + 1)
        out[base] = current
        for step in [0, walkLength):
            begin = rowptr[current]
            end   = rowptr[current + 1]
            if end > begin:                            # 出度 > 0：均匀采样一条邻边
                current = col[begin + umulhi(PhiloxNext(rng), end - begin)]
                out[base + step + 1] = current
            else:                                      # 出度 == 0：死端，剩余位置全部写 -1
                for l in [step + 1, walkLength]:       # 闭区间
                    out[base + l] = -1
                break
```

`PhiloxNext` 的实现要点（`random_walk_kernel.cpp`）：

- 采用 Philox4x32-10 计数器式伪随机数发生器：10 轮迭代，乘子常量 `0xD2511F53` / `0xCD9E8D57`，
  轮常量 `0x9E3779B9` / `0xBB67AE85`，每轮用 `__umulhi` 取 32 位乘法的进位部分。
- 每次 refill 生成 4 个 uint32（lane 0..3），组内取尽后 `counter[0]` 自增、溢出时 `counter[1]` 进位。
- **每条游走一条独立流**：`counter[2:3]` 编码游走序号 `walk`。因此每条路径的随机序列只由 `(seed, offset, walk)` 决定，
  与线程到路径的映射、网格步长的具体遍历顺序无关，结果对并行度不敏感。
- **无偏采样映射**：`idx = __umulhi(u32, degree)`，即把均匀 uint32 乘出度后取高 32 位，得到 `[0, degree)` 上的均匀下标；
  这里不使用浮点乘法，避免 `int64(rand * degree)` 在 `rand` 接近 1 时的浮点舍入偏差。
- **消费顺序**：每条游走在每一步采样前消费 1 个 uint32（包括走到死端的那一步）；进入死端后缀填充后不再消费。
  `S == 0` 或 `L == 0` 时不从默认发生器取状态。
- `seed`/`offset` 由 host 侧从 torch_npu 默认 NPU 随机数发生器获取，因此 `torch.npu.manual_seed(seed)` 可以复现结果。

`ascendc_assert` / 设备侧断言：两 kernel 不使用 `__trap` 等设备侧中断，全部错误通过 host 侧校验（`random_walk`）
或 `violation` 标记（`non_diag_mask`）显式上报，保证错误信息在 Python 层可见。

#### 3.2.2.2 Ascend C 主流程图

```text
【non_diag_mask】
┌──────────────────────────────────────────────┐
│host：结构校验 / num_diag 推导 / 输出分配     │
│      │                                       │
│      ▼                                       │
│E == 0 ?                                      │
│  ├─ 是 → 返回 zeros([numDiag], bool)         │
│  └─ 否 → violation 清零（aclrtMemsetAsync）  │
│            │                                 │
│            ▼                                 │
│SIMT kernel：grid-stride 遍历 E + numDiag     │
│  i < E ：校验值域并写 (row + k != col)       │
│  i >= E：写 false                            │
│            │                                 │
│            ▼                                 │
│host：单次 aclrtMemcpy D2H 回读 violation     │
│  violation != 0 → 抛出越界错误               │
└──────────────────────────────────────────────┘

【random_walk】
┌──────────────────────────────────────────────────────┐
│host：结构校验 / 快速路径判断                         │
│      │                                               │
│      ▼                                               │
│S == 0 或 L == 0 ?                                    │
│  ├─ 是 → 返回空输出 [0, L+1] / 仅首列 [S, 1]         │
│  └─ 否 → CSR/col/start 值域校验（NPU 算子 + item()） │
│            │                                         │
│            ▼                                         │
│取 philox(seed, offset)（默认 NPU 发生器，持锁）      │
│            │                                         │
│            ▼                                         │
│SIMT kernel：grid-stride 遍历 startCount              │
│  每线程处理一条游走：                                │
│    写首列 out[base] = start                          │
│    逐步采样并写 out[base + step + 1]                 │
│    出度为 0 → 剩余位置写 -1 并终止该条游走           │
│            │                                         │
│            ▼                                         │
│返回 out [S, L+1]                                     │
└──────────────────────────────────────────────────────┘
```

#### 3.2.2.3 与标杆算子实现流程的差异点说明

| # | 差异点 | 标杆实现 | 本设计实现 | 原因 |
| --- | --- | --- | --- | --- |
| 1 | 掩码编码 | 插入槽位散布：按分支把 `true` 写到 `idx+r`、`idx+r+1`、`idx+num_diag` 等位置，要求输入按 `row` 非降序且目标对角线边已移除 | 逐边比较：`mask[i] = (row[i]+k != col[i])`，尾部 `num_diag` 位恒为 false | 任务书 §2.1 的输出定义与验收 golden 一致；两种编码 true/false 个数相同，仅排列不同，位置重建差异在 3.4 第 1 条单列 |
| 2 | `num_diag` 边界 | `k<0 ? min(M+k,N) : min(M,N-k)`，未截断，`k` 越界时为负 | `max(0, ...)`，负值按 0 处理 | 任务书明确"负值按 0 处理"；避免负长度导致输出长度小于 `E` |
| 3 | 输入值域校验 | 无校验，越界索引为未定义行为 | `row`/`col` 值域在 kernel 内融合检查并写 `violation`，host 单次 D2H 回读后报错 | 任务书 §2.4 要求"类型、长度、越界错误"必须报错；融合检查只增加一次同步，不新增 NPU kernel |
| 4 | 随机数来源 | CPU：`torch::rand([S,L])` CPU 张量；CUDA：`torch::rand([L,S])` 显存张量 | torch_npu 默认发生器的 Philox 状态 + kernel 内计数器式 Philox4x32-10（每条游走一条流） | 避免一次性生成并读写 `S×L` 随机数张量（固定 case 为 131 KB，大 case 可达数百 MB）；随机数在寄存器中生成，不占显存带宽 |
| 5 | 采样映射 | `int64(rand * degree)`（float32 乘法后截断） | `umulhi(u32, degree)`（整数乘高位映射） | 整数映射在数学上无偏，且避免 `rand → degree` 的浮点舍入越界 |
| 6 | 死端处理 | CUDA 路径无专门分支，出度为 0 时仍按下标读取 `col`（读该行之外的元素，行为未定义）；任务书要求未写入位置为 -1 | `end == begin` 时把该行剩余位置写 -1 并终止该条游走 | 任务书 §2.1 明确"未写入位置初值为 -1"，且必须先判定死端再决定是否读取 `col` |
| 7 | 输出布局 | CUDA 先生成 `[L+1, S]` 再 `t().contiguous()` 转置（一次额外读写） | 按 `[S, L+1]` 直接布局写 `out[walk*(L+1)+step]` | 省掉一次转置连续化拷贝，输出布局与任务书一致 |
| 8 | 并行映射与确定性 | 每线程一条游走（CUDA `THREADS=1024`），随机数取自预生成矩阵的固定下标，结果与线程映射无关 | 每线程一条游走（`THREADS=256`），随机序列由 `walk` 序号定流 | 两者对线程映射都不敏感；本设计把"路径 ↔ 随机流"绑定显式写进 kernel，便于复现与对齐 |
| 9 | 设备支持面 | CPU 与 CUDA 两套实现，按输入设备分发 | 仅 NPU（PrivateUse1）；CPU 输入注册拒绝内核并明确报错 | 任务书 §7 要求"NPU 不得回退 CPU"；CPU 侧继续由上游 pytorch_sparse 承担 |

差异总览附图：

```text
    标杆 pytorch_sparse                     本设计 Ascend 950PR
┌──────────────────────────┐          ┌────────────────────────────────┐
│ mask：槽位散布 true       │          │ mask：逐边比较 + 尾部 false      │
│  依赖 row 有序与 remove_diag│ ──────► │  融合值域校验，violation + D2H  │
└──────────────────────────┘  编码替换  └────────────────────────────────┘
┌──────────────────────────┐          ┌────────────────────────────────┐
│ walk：预生成 rand[S,L]    │          │ walk：kernel 内 Philox4x32-10   │
│  float 乘法截断，无死端分支 │ ──────► │  umulhi 整数映射，死端填 -1      │
│  CUDA 输出转置            │  范式替换 │  直接 [S, L+1] 布局              │
└──────────────────────────┘          └────────────────────────────────┘
┌──────────────────────────┐          ┌────────────────────────────────┐
│ CPU / CUDA 双实现         │  ──────►  │ 仅 NPU；CPU 明确拒绝，不回退      │
└──────────────────────────┘  范围收窄  └────────────────────────────────┘
```

## 3.3 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Ascend 950 系列（arch35，`NPU_ARCH=dav-3510`） | √ |

构建期 `CMakeLists.txt` 对 `NPU_ARCH` 做显式校验：不等于 `dav-3510` 时直接报错，不宣称支持其他芯片架构。
GPU 环境仅作为性能与精度的对标环境使用，不属于本设计的交付范围。

## 3.4 算子限制

1. **掩码排列与 `SparseTensor` 前端适配**：公共 `non_diag_mask` 采用任务书 §2.1 的逐边布尔语义；上游
   `torch_sparse/diag.py` 的 `new_row[mask] = row` / `new_row[inv_mask] = diag` 位置重建按上游槽位编码编写。
   因此 `SparseTensor.set_diag/fill_diag` 在 NPU 上使用时，前端需要保证重建后的 COO 排列（按 `(row, col)` 稳定排序）
   与存储的 `is_sorted` 声明一致。本次交付的 `torch_extension/torch_sparse_npu/sparse_tensor.py` 提供
   `SparseTensor.random_walk` 的挂载与集成自检函数 `check_integration()`；`set_diag/fill_diag` 的 COO 排序适配属前端配套项，
   与上游 `torch_sparse` 安装环境的集成验证一并列入待补验证（见 5.3），本设计不声称已通过该项集成测试。
2. **索引 dtype 与维度**：`row`/`col`/`rowptr`/`start` 仅支持 INT64 一维连续张量；无 batch 维、无广播。
3. **32 位启动参数上限**：`walk_length` 与 `nnz` 不超过 `2**32-1`（超出报错）。输入输出总字节数仍受分配器与 `int64` 维度限制。
4. **CPU 输入**：明确报错，不回退 CPU 计算。
5. **随机性口径**：本设计的多邻居路径使用自有 Philox 流，未与 GPU 标杆的 float32 采样建立逐位对齐；
   只对单邻居（出度恒为 1）/确定性图承诺与标杆逐元素一致，多邻居场景按 5.1 的随机场景校验方式验收。
6. **校验开销**：`random_walk` 的 CSR/值域校验在 host 侧以 NPU 张量算子 + 同步读取完成，引入固定同步开销，
   该开销包含在被测的公共调用内，不作为可关闭的调试开关。
7. **`non_diag_mask` 的 host 同步**：越界 `violation` 回读是一次 D2H 阻塞拷贝，是小规模 case 下接口时延的主要组成部分；
   该同步属于错误上报契约的一部分，不提供绕过路径。

# 四、特性交叉分析

| 交叉项 | 分析 |
| --- | --- |
| `SparseTensor.set_diag` / `fill_diag` | 上游 `diag.py` 通过 `torch.ops.torch_sparse.non_diag_mask` 调用本实现；NPU 张量经 `PrivateUse1` 分发进入本 kernel，上游 Python 代码无需修改；COO 排列差异见 3.4 第 1 条 |
| `SparseTensor.random_walk` | 上游 `rw.py` 通过 `torch.ops.torch_sparse.random_walk` 调用本实现；本包在方法缺失时提供等价挂载（`sparse_tensor.py`），并暴露 `check_integration()` 供测试与验收脚本调用 |
| 与上游 pytorch_sparse 的共存 | 本库 schema 与上游保持一致；`torch_sparse_npu/__init__.py` 先尝试导入上游（复用其 Python 前端与既有分发），再由本库补充 `PrivateUse1` 实现与 `CPU` 拒绝内核。上游以 `torch::RegisterOperators` 注册的同名算子是**不分设备键**的注册，实测会抢占 NPU 分发，因此本设计采用自包含注册并在验证环境移除冲突包，详见 5.2 |
| torch_npu 默认随机数发生器 | `random_walk` 每次成功调用从默认发生器取 4 个 counter 增量；`torch.npu.manual_seed` 可复现结果，同 seed 两次调用结果一致，不同 seed 结果不同（已由扩展测试覆盖） |
| ops-sparse（`aclsparse*`） | 无调用、无数据类型/签名依赖；互不影响 |
| ops-gnn 既有算子（gather_coo / gather_csr / scatter 等） | 仅遵循同一目录与代码规范，无代码级耦合；本设计不改动 `csrc/`、`python/`、`csrc/pybind.cpp` |
| 非连续输入 | 本设计要求输入连续并显式报错，不做隐式拷贝；不影响其它算子的内存布局约定 |

# 五、可维可测分析

## 5.1 精度标准 / 性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 功能标准 | 覆盖空 COO、方阵/矩形、正负及越界 `k`、空 `start`、`walk_length=0`、单边、单邻居、孤立节点、hub、≥1e5 节点图；合法输入结果正确，非法输入必须报错 | 任务书 §2.4 / §3.5 |
| 精度标准 | 输出为 bool/int64 离散结果，与 CPU golden 及标杆按**逐元素 EXACT** 比对（`torch.testing.assert_close` 对离散类型为全等比较） | 任务书 §3.2、《生态算子开源精度标准（experimental）》 |
| 随机场景精度 | 多邻居图按"转移合法性 + 首列/shape/dtype/值域 + 死端后缀 + 同 seed 可复现"校验；仅单邻居/确定性图做逐元素 EXACT（三端 RNG 未对齐时不声称逐元素一致） | 任务书 §3.2 |
| 性能标准 | 固定性能集与两个接口各自的泛化性能集，分别计算**接口性能比**（GPU Event wall time ÷ NPU Event wall time）与 **Kernel 性能比**（GPU Kernel duration 总和 ÷ NPU Kernel duration 总和）的算术平均，四项均须 ≥0.3×；未执行/不支持/失败的 case 保留在集合与分母中 | 任务书 §3.3 |
| 内存标准 | 满足任务书 §3.4 两项条件之一：输入输出总量超过 500 MB 时额外内存不超过 GPU 的 50%；或固有 workspace 绝对值不超过目标硬件 L2 Cache 容量 | 任务书 §3.4 |
| 交付标准 | 源码、构建说明、功能/精度脚本、200 条 case 文件、执行日志、性能 CSV/JSON、GPU/NPU 对比结论、README、设计文档 | 任务书 §4 |

固定性能 case（任务书 §3.3）：

| 接口 | Case | 图/矩阵参数 | 其他参数 | GPU 接口 (ms) | GPU Kernel 总和 (ms) |
| --- | --- | --- | --- | ---: | ---: |
| `non_diag_mask` | fixed_mask_01 | M=10000, N=10000, E=100000 | k=0 | 0.009904 | 0.004800 |
| `random_walk` | fixed_walk_01 | num_nodes=10000, nnz=100000 | num_starts=1024, walk_length=32, regular CSR | 0.135451 | 0.058784 |

泛化 case 集的验收要求为每接口 200 条（任务书 §3.3 的形状/结构组合）。本次可获得的 GPU 标杆性能基线为每接口 80 条
（固定 case 单列，泛化 79 条），因此泛化集均值按 79 条逐 case 对齐计算，缺口在 5.3 中如实登记。

### 5.1.1 已执行验证与测量结果

| 项 | 结果 | 证据 |
| --- | --- | --- |
| 功能验收（400 例） | PASS：`non_diag_mask` 200 例 + `random_walk` 200 例，逐元素 EXACT | `tests/test_torch_sparse_utils.py`、`verification/execution_logs/functional_test_torch_sparse_utils.log` |
| 扩展正确性（边界/异常/可复现） | PASS：空 COO、`E=0`、方阵/矩形、正负及越界 `k`、空 `start`、`L=0`、单邻居确定性、孤立节点 -1、hub 值域、同/异 seed；越界、dtype、维度、长度、非法 CSR、负步长均正确报错 | `tests/test_correctness.py`、`verification/execution_logs/test_correctness.log` |
| `SparseTensor` 集成 | SKIP：验证环境未安装上游 `torch_sparse`，集成用例未执行 | 同 `test_correctness.py`（打印 SKIP）；`verification/验证结果报告.md` |
| 固定 case 接口性能比 | `mask_001` 0.333×（GPU 0.009904 / NPU 0.029771 ms）；`walk_001` 0.396×（GPU 0.135451 / NPU 0.341652 ms） | `verification/perf_comparison.csv` |
| 固定 case Kernel 性能比 | `mask_001` 0.866×（GPU 0.004800 / NPU 0.005541 ms）；`walk_001` 0.757×（GPU 0.058784 / NPU 0.077674 ms） | `verification/perf_comparison.csv` |
| 泛化集接口性能比（79 条/接口，算术平均） | `non_diag_mask` 0.420×；`random_walk` 0.506× | `verification/perf_comparison.csv`、`GPU_NPU_对比结论.md` |
| 泛化集 Kernel 性能比（79 条/接口，算术平均） | `non_diag_mask` 1.225×；`random_walk` 3.483× | 同上 |
| 内存 | 固定 case 额外峰值分配：`mask_001` 110,592 B（≈0.1 MB）；`walk_001` 17,982,464 B（≈17.2 MB），两者均远低于 L2 约束与 500 MB 门槛 | `verification/验证结果报告.md` |

测量方法：接口时延由 Event 计时（warmup=10、iterations=30，取单次调用均值），Kernel 总和由 profiler
`kernel_details.csv` 的 `Duration(us)` 累加后除以 5 个 active step 得到；GPU 数值取自任务方提供的标杆 CSV
（`gpu_mask_perf.json` / `gpu_walk_perf.json` 合并结果），未在本地重测 GPU。环境：Ascend 950PR 单卡 NPU0，
CANN 9.1.0，Python 3.12.13，PyTorch / torch_npu 2.12.0。

## 5.2 兼容性分析

- ops-gnn 中新增 `experimental/torch_sparse_utils/` 模块，属新算子目录，不涉及对历史 TBE 算子或 aclnn 接口的兼容；
  PyTorch 层接口与 `pytorch_sparse` 对齐。
- 与上游 pytorch_sparse 共存：本库 schema 与上游一致，仅补充 `PrivateUse1`（NPU）实现与 `CPU` 拒绝内核；
  CPU/CUDA 路径仍由上游实现承担，不覆盖、不修改上游 `diag.py` / `rw.py`。
- 同名算子的注册冲突：上游 0.6.18 使用 `torch::RegisterOperators().op(...)`，该注册不带设备键限制；
  验证过程中观察到该包（研发机上的改造版）会抢占 `non_diag_mask` / `random_walk` 的 NPU 分发，使 NPU 输入回落到 CPU 行为。
  本设计的处理方式是**自包含注册**：由本库以 `TORCH_LIBRARY` 定义 schema、以 `TORCH_LIBRARY_IMPL(PrivateUse1)` 提供 NPU 内核，
  并在验证环境移除冲突包。`SparseTensor` 前端集成验证须在仅安装**官方版** pytorch_sparse（其 Python 前端调用
  `torch.ops.torch_sparse.*`，由本库承接 NPU 分发）的环境中进行，该项集成测试当前未执行（见 5.3）。
- 共享文件无改动：不修改 ops-gnn 的 `csrc/`、`python/`、`csrc/pybind.cpp`、`CMakeLists.txt`；
  `experimental/` 目录自包含构建（`CMakeLists.txt` + `setup.py` + `build.sh`）。
- 硬件适配范围限定 Ascend 950PR（arch35 / `dav-3510`）；其他架构不在支持范围，构建期显式报错。
- 随机性行为与上游不一致属于设计选择（见 3.2.2.3 差异点 4/5/6），不影响"接口签名与功能语义对齐"这一验收口径。

## 5.3 交付件与覆盖缺口登记

交付件清单：

| 交付项 | 位置 |
| --- | --- |
| 算子实现源码 | `op_kernel/arch35/`、`op_host/`、`torch_extension/` |
| 构建说明与配置 | `README.md`、`README_EN.md`、`CMakeLists.txt`、`setup.py`、`build.sh` |
| 功能/精度脚本 | `tests/test_torch_sparse_utils.py`、`tests/test_correctness.py`、`tests/golden.py` |
| 200 条 case 文件 | `verification/cases/non_diag_mask_validation_200.json`、`random_walk_validation_200.json` |
| 执行日志 | `verification/execution_logs/*.log` |
| 性能脚本与结果 | `tests/benchmark_torch_sparse_utils.py`、`verification/perf_comparison.csv`、`perf_comparison.json` |
| GPU/NPU 对比结论与结果报告 | `verification/GPU_NPU_对比结论.md`、`verification/验证结果报告.md` |

书面覆盖与本次 case 附件的差异（如实登记，未从集合或分母中剔除任何 case）：

| 接口 | 任务书书面要求（§3.3） | 本次附件实际覆盖 | 补充方式 |
| --- | --- | --- | --- |
| `non_diag_mask` | M/N = 1e2~1e5，E = 0~1e6，k = 0/±1/±3/±10 | N ∈ {1, 100, 1000, 10000, 100000}；M ∈ {1, 10000}；E ∈ {0, 1, 100, 1000, 10000, 100000}；k ∈ {0, 1, 3, 10} | 负 k 与越界 k 由 `tests/test_torch_sparse_utils.py`（k ∈ {0, ±1, ±3, ±10, 4096}）与 `tests/test_correctness.py`（k = ±2/±4/±100 等越界值）覆盖；M=1e2~1e5 的中间档位待补充 case 与 GPU 基线 |
| `random_walk` | V = 1e3~1e5，nnz = 1e3~1e6，S = 0/1/64/1024/4096，L = 0/1/8/32/64，regular/hub/isolated | V ∈ {1000, 10000}；nnz ∈ {1000, 10000, 100000, 1000000}；S ∈ {1, 16, 64, 256, 1024}；L ∈ {1, 8, 32, 64}；结构含 regular/hub/isolated | S=0、L=0、孤立节点与 hub（1000 节点）由 `tests/test_correctness.py` 覆盖；S=4096、V=1e5（任务书 §3.5 的 ≥1e5 节点图）档位待补充 case 与 GPU 基线 |
| 性能泛化集 | 每接口 200 条 | 每接口 80 条 GPU 基线（泛化 79 条 + 固定 1 条） | 泛化集均值按 79 条逐 case 计算结果；200 条规模的基线与 case 集需与任务方确认后补充 |
| `SparseTensor` 集成 | 覆盖 `set_diag/fill_diag/random_walk` 的 NPU 使用 | 未执行（环境缺上游 `torch_sparse`） | 需在装有上游 `torch_sparse` 的环境补测；本设计已提供 `check_integration()` 与前端挂载代码 |

上游 RNG 对齐状态：本设计未与 GPU 标杆建立逐位 RNG 对齐，因此 400 例功能验收中 `random_walk` 的 200 例使用单后继
（每条边唯一后继）图构造确定性路径；多邻居场景的校验采用扩展测试中的"转移合法性 + 可复现性"判据，不声称跨设备逐元素一致。
