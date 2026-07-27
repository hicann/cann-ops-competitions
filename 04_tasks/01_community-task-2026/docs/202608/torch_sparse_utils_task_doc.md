# 8月社区任务 - torch_sparse 工具类接口开发任务书

## 任务概述

参考 pytorch_sparse（https://github.com/rusty1s/pytorch_sparse ） 中图采样及稀疏对角操作相关工具算子，在昇腾 NPU 上于 **ops-gnn 仓**实现功能一致的算子集合，并通过 PyTorch 自定义算子机制对外提供与 CUDA 侧一致的调用语义。

| 接口名 | 对标算子 | 主要用途 |
|--------|----------|----------|
| `non_diag_mask` | `torch.ops.torch_sparse.non_diag_mask` | 为 COO 稀疏矩阵生成「非第 k 条对角线元素」布尔掩码，供 `SparseTensor.set_diag` / `fill_diag` 合并对角与非对角边 |
| `random_walk` | `torch.ops.torch_sparse.random_walk` | 在 CSR 邻接图上做**均匀随机游走**，输出多条游走路径，供图采样 / 子图构造等使用 |

上述接口为 PyG 生态中 `SparseTensor` 对角线设置及图采样等功能的底层依赖，须在 NPU 设备张量上提供与 CUDA 参考实现一致的功能与数值行为。

**Python 对外入口**（上游 pytorch_sparse（https://github.com/rusty1s/pytorch_sparse ））：

```python
from torch_sparse import SparseTensor

# non_diag_mask：由 diag.py 内部调用，典型用户路径为 set_diag / fill_diag
out = src.set_diag(values, k=0)      # 设置第 k 条对角线（主对角 k=0）
out = src.fill_diag(fill_value, k=0) # 用常数填充对角线

# random_walk：SparseTensor 方法
paths = adj.random_walk(start, walk_length)
# paths.shape == [num_starts, walk_length + 1]；paths[:, 0] == start
```

## 核心开发要求及验收标准

开发/测试硬件和软件要求：
- **适配硬件**：Ascend 950PR
- **CANN 版本**：算子开源仓（https://gitcode.com/cann/ops-gnn ）指定版本

### 功能实现要求

1. **接口语义对齐**：输入输出及边界行为须与 pytorch_sparse（https://github.com/rusty1s/pytorch_sparse ） 一致。参考实现：

   | 算子 | Host | CPU | CUDA |
   |------|------|-----|------|
   | `non_diag_mask` | `csrc/diag.cpp` | `csrc/cpu/diag_cpu.cpp` | `csrc/cuda/diag_cuda.cu` |
   | `random_walk` | `csrc/rw.cpp` | `csrc/cpu/rw_cpu.cpp` | `csrc/cuda/rw_cuda.cu` |

2. **CPU 实现范围（本任务不要求）**：**仅新增 NPU 路径**；CPU/CUDA 由 pytorch_sparse 提供，NPU 输入不得回退 CPU。

3. **PyTorch 集成**：建议路径 `experimental/torch_sparse_utils/`，注册至 `torch_sparse` 命名空间；design doc 说明与上游 wheel 共存方案。

4. **设备与数据类型**：

| 算子 | 输入 dtype | 输出 dtype | 设备 |
|------|------------|------------|------|
| `non_diag_mask` | int64（row/col） | bool | NPU |
| `random_walk` | int64（rowptr/col/start） | int64 | NPU |

5. **算子功能规格**：

#### 5.1 non_diag_mask

**功能**：给定 COO 边列表 `(row, col)`、矩阵规模 `(M, N)` 及对角线偏移 `k`，生成长度 `E + num_diag` 的布尔掩码 `mask`，用于将原 COO 边与「第 k 条对角线上的缺失元素」合并为新的 COO 表示（见上游 `torch_sparse/diag.py` 中 `set_diag`）。

**对角线语义**（与 PyTorch `torch.diagonal` 偏移一致）：

- `k = 0`：主对角线，满足 `row == col`；
- `k > 0`：主对角线上方第 `k` 条，满足 `row + k == col`；
- `k < 0`：主对角线下方第 `|k|` 条，满足 `row == col + |k|`。

其中 `num_diag = min(M, N - k)`（`k ≥ 0`）或 `min(M + k, N)`（`k < 0`），表示该偏移对角线在 `(M, N)` 矩阵中的有效对角元个数。

**掩码含义**（对标 `csrc/cpu/diag_cpu.cpp` / `diag_cuda.cu`）：

| `mask[i]` | 含义 |
|-----------|------|
| `True` | 输出 COO 第 `i` 个位置保留**原非对角边**（来自输入 `row/col` 中不在第 k 条对角线上的边） |
| `False` | 输出 COO 第 `i` 个位置用于**插入对角元**（原 COO 中该对角位置缺失，由 `set_diag` 填入 `values` 或默认值） |

**典型用法**：`SparseTensor.set_diag(values, k)` 先 `remove_diag` 去掉已有第 k 对角边，再调用本算子得到 `mask`，按 `new_row[mask]=row`、`new_row[~mask]=diag_index` 合并出完整 COO。

**输出**：`bool [E + num_diag]`，与参考实现逐元素一致。

#### 5.2 random_walk

**功能**：在 **无向/有向图** 的 CSR 邻接表上，从多个起点并行执行**均匀随机游走**（每一步在当前节点的邻居中等概率选取下一跳），对标 `csrc/cpu/rw_cpu.cpp` / `rw_cuda.cu`。

**算法**（每条游走独立）：

1. 初始化 `out[n, 0] = start[n]`；
2. 对 `l = 0 … walk_length-1`：令 `cur = out[n, l]`，在 `[rowptr[cur], rowptr[cur+1])` 范围内按均匀分布随机选取一条出边，令 `out[n, l+1] = col[选中边]`；
3. 随机数来源与 CUDA 参考一致（`torch.rand` 浮点 × 度数取整）；须在 design doc 说明 NPU 侧 RNG 与 seed 复现方案。

**输入约束**：`rowptr` 为 CSR 行指针 `[num_nodes+1]`，`col` 为列索引 `[nnz]`，`start` 为起点节点 id `[num_starts]`；节点 id 须在 `[0, num_nodes)` 内。

**输出**：`int64 [num_starts, walk_length + 1]`；第 0 列为起点，后续列为各步到达的节点 id。未写入位置初值为 `-1`（参考实现分配时填充；正常完整游走后前 `walk_length+1` 列均有效）。

**典型用途**：图神经网络中的子图采样、Deep Graph Infomax、负采样路径生成等；Python 入口为 `SparseTensor.random_walk(start, walk_length)`（`torch_sparse/rw.py`）。

6. **工程隔离**：本任务合入 **ops-gnn**，与 ops-sparse 仓 `aclsparse*` C API 解耦，不得修改 ops-sparse 仓代码。

7. **泛化能力**：覆盖空图、单边、大规模（≥10^5 节点）及 hub 节点等场景。

### 参数说明

以下 C++ 接口签名与 pytorch_sparse/csrc/sparse.h（https://github.com/rusty1s/pytorch_sparse/blob/master/csrc/sparse.h ） 保持一致。

#### non_diag_mask

| 参数名 | 输入/输出 | 描述 | 数据类型 | 维度 Shape |
|--------|-----------|------|----------|------------|
| row | 输入 | COO 行索引 | int64 | [E] |
| col | 输入 | COO 列索引 | int64 | [E] |
| M | 属性 | 矩阵行数 | int64 | 标量 |
| N | 属性 | 矩阵列数 | int64 | 标量 |
| k | 属性 | 对角线偏移（`k≥0` 为主对角线上方第 k 条，`k<0` 为下方） | int64 | 标量 |
| mask | 输出 | 布尔掩码，长度 `E + num_diag`；`True` 表示该槽位放原 COO 非对角边，`False` 表示该槽位放第 k 对角元（见 §5.1） | bool | [E + num_diag] |

#### random_walk

| 参数名 | 输入/输出 | 描述 | 数据类型 | 维度 Shape |
|--------|-----------|------|----------|------------|
| rowptr | 输入 | CSR 行指针 | int64 | [num_nodes+1] |
| col | 输入 | CSR 列索引 | int64 | [nnz] |
| start | 输入 | 游走起始节点 | int64 | [num_starts] |
| walk_length | 属性 | 游走步数（不含起始节点） | int64 | 标量 |
| out | 输出 | 游走路径；`out[:,0]=start`，`out[:,l+1]` 为第 `l` 步到达的节点（均匀随机选邻居） | int64 | [num_starts, walk_length+1] |

### 接口设计建议

```cpp
torch::Tensor non_diag_mask(torch::Tensor row, torch::Tensor col,
                            int64_t M, int64_t N, int64_t k);
torch::Tensor random_walk(torch::Tensor rowptr, torch::Tensor col,
                          torch::Tensor start, int64_t walk_length);
// 注册至 torch_sparse 命名空间，见 sparse.h
```

### 测试标准

1. **功能对齐测试**：以 pytorch_sparse CUDA 实现为参考标杆，在相同输入下对比 NPU 输出，要求逐元素完全一致（整数/布尔类型）或在规定 seed 下 random_walk 输出一致。
2. **覆盖场景**：
   - 常规图：Cora、Citeseer 规模（节点 10^3~10^4）
   - 边界：E=0、M=1、walk_length=0、k 正负偏移、空 start
   - 大规模：节点 ≥ 10^5
3. **PyG 集成冒烟**：`non_diag_mask` 对角操作、`random_walk` 采样路径在 NPU 上可正常运行。
4. 自验证报告完整、可复现，所有测试用例执行通过。

### 性能要求

对标 **pytorch_sparse CUDA（A100）**，NPU 性能须达到 GPU 参考实现的 **0.5 倍**及以上（即 ≥0.5×）。

**性能验收说明**：本任务若有多支队伍提交实现，同规格用例下横向对比各队 NPU 实测性能，**取性能最优的提交进行验收**。

**固定参考用例**（必测）：

| 编号 | 算子 | 输入规模 | 达标要求 |
|------|------|----------|----------|
| 1 | non_diag_mask | E=10^5, M=N=10^4 | ≥0.5× |
| 2 | random_walk | num_starts=1024, walk_length=32, nnz=10^5 | ≥0.5× |

**泛化覆盖范围**（在固定参考用例之外**另抽 200 组**用例；由下列维度组合抽样或网格扫描生成，须附完整用例列表；达标 ≥0.5×，且不得出现某一子范围内性能断崖）：

| 维度 | `non_diag_mask` | `random_walk` |
|------|-----------------|---------------|
| 规模 | `M,N`：10²～10⁵；`E`：0～10⁶ | 节点数：10³～10⁵；`nnz`：10³～10⁶ |
| 结构 | `k`：主对角及偏移 ±1、±3、±10 | 平均度 1～10²；含 hub 节点（单点出度 ≥10³） |
| 批量 | — | `num_starts`：1、64、1024、4096 |
| 步长 | — | `walk_length`：1、8、32、64 |
| 边界 | `E=0`；`M=1`；`|k|≥min(M,N)` | 孤立节点（度=0）；空 `start` |

### 精度要求

- `non_diag_mask`：与参考实现**逐元素完全一致**（布尔类型）。
- `random_walk`：固定 seed 下与 CUDA 参考完全一致。

## 验收交付件

在社区任务IT系统中提交验收时， 需要提交以下交付件：

| 序号 | 交付件名称 | 交付件要求 |
|------|-----------|------------|
| 1 | 算子设计文档 | 1. 设计文档模板：https://gitcode.com/cann/cann-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md ；<br> 2. 在cann-competitions 仓库（https://gitcode.com/cann/cann-competitions/tree/master/04_tasks/01_community-task-2026/tasklist ）以PR形式提交设计文档，通过评审后合入仓库，详细说明见：https://gitcode.com/cann/cann-competitions/blob/master/04_tasks/01_community-task-2026/README.md|
| 2 | 自测用例及测试代码 |1. 需要清晰列出精度测试case和性能测试case；<br> 2. 测试代码中的readme文件需要说明测试步骤，保证验收人可以复现测试结果|
| 3 | 自测报告 | 1. 自测报告模板：https://docs.qq.com/sheet/DUmVWWndaUE12WGFB?tab=BB08J2 ； <br> 2. 需要包含用例参数、精度对比结果及截图、性能数据及截图 |
| 4 | 待验收代码地址 | 1. 个人代码仓链接、分支、算子目录；需要在个人仓邀请账号Ascend-CANN作为开发者，如下图所示； <br> 2. 算子目录下需要提供readme文件，readme 须说明编译方式、Python 调用示例及与 pytorch_sparse 的对应关系 |

![邀请示意](./pics/invite.jpeg)

### PR 申请合入

验收通过后，在 **ops-gnn** 开源仓提交 PR，建议合入路径：

```
experimental/torch_sparse_utils/
├── op_kernel/              # non_diag_mask / random_walk Kernel
├── op_host/                # Host Launch
├── torch_extension/        # PyTorch 算子注册
├── tests/                  # NPU/CUDA 对比测试
└── README.md
```

## 参考资料

1. pytorch_sparse 源码仓库（https://github.com/rusty1s/pytorch_sparse ）
2. PyG SparseTensor 文档（https://pytorch-geometric.readthedocs.io/en/stable/advanced/sparse_tensor.html ）
3. Ascend C 算子开发文档、算子开发接口文档
4. ops-sparse 仓（https://gitcode.com/cann/ops-sparse ） SpMM/SpMV 实现（`src/spmm/`、`src/spmv/`），供 Stream 管理与 Host 调度参考（**只读参考，不合入 ops-sparse**）

## 环境获取

1. 使用 hidevlab webIDE 算力：https://hidevlab.huawei.com/online-develop-intro?from=hiascend ；
   - 如果是新用户，在申请权限的时候需要备注使用的算力类型（A2、A3或者950）；
   - 如果是老用户且需要使用950算力，需要向昇腾CANN小助手反馈账号名（个人中心->基本信息，如下图所示），后台会添加账号至950使用白名单。

   ![环境截图](./pics/zaixiankaifa1.png)  
   ![账号名](./pics/account.png)  

2. 开源仓提供100小时免费时长，请不使用时及时关闭，用时耗尽前请务必保存相关资料，建议及时提交备份。

   ![环境截图](./pics/yunkaifa.png)

3. 如需额外环境资源，请联系昇腾CANN小助手。


## 特别注意事项

1. 本任务仅合入 **ops-gnn**，不得修改 ops-sparse 仓代码。
2. 算子注册至 `torch_sparse` 命名空间，`import torch_sparse_npu` 调用;
3. 所有交付件须提前完成自验证，确认符合验收标准后再提交；
4. 开发前务必阅读【社区任务】流程及注意事项（https://gitcode.com/org/cann/discussions/39 ）。