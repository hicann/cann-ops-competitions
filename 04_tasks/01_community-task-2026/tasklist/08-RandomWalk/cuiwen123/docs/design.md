# random_walk 算子设计文档

# 需求背景（required）

## 需求来源

CANN 社区 2026 年 8 月 RandomWalk 任务要求在 `cann/ops-gnn` 中提供 Ascend 950PR 版本
`ops_gnn.random_walk()`。公开接口、Python 预处理、随机游走语义、精度和性能标准以任务书为准，
参考接口为 `torch_cluster.random_walk` 1.6.0 及以上版本。

## 背景介绍

RandomWalk 从一组起点出发，在图上生成定长节点序列，是 DeepWalk、node2vec 等图学习算法的
采样基础。公开接口接收 COO 边 `row/col`，在 Python 层完成排序和 CSR 构造，再调用 NPU
采样扩展。`p=q=1` 时执行均匀随机游走；其余情况执行 node2vec 拒绝采样。

# 需求分析（required）

## 需求描述

设计与任务书 Python 原型一致的 PyTorch/PyBind 扩展，只支持 Ascend 950PR 前向计算。
输入、图索引和输出均为 INT64，图预处理和采样保持在 NPU 链路内，不增加 aclnn/OPP 入口。
算子不产生 float64 Tensor，也不执行浮点图数据的 CPU fallback；`p/q` 仅作为标量概率参数。

## 需求拆解

- 复现任务书规定的九参数 Python 接口和四步 CSR 预处理；
- 提供均匀随机游走和 node2vec 拒绝采样；
- 支持边索引返回、孤立节点、空输入和 `walk_length=0`；
- 通过 PyTorch dispatcher 连接 Python、PyBind 和 Ascend C Kernel；
- 按公开 Python API 完成功能、随机性和性能验收。

| 任务书条款 | 设计落点 |
| --- | --- |
| §1 Python 接口 | “算子原型”中的 Python 签名、参数表和返回规则 |
| §2 Python 预处理 | “Python 预处理”中的节点数推导、排序、CSR 和六参数调用 |
| §3 参数与约束 | “算子原型”和“算子约束限制” |
| §4 随机性、分支和孤立节点 | “采样语义”和“Device 侧设计” |
| §5 TC-01～TC-07 | “功能与精度测试” |
| §6 性能要求 | “性能标准”中的四个官方 case 和公开 API 计时口径 |
| §7 精度要求 | “精度标准”中的 CPU reference、固定 seed 和统计判据 |
| 特别注意事项 | Python CSR、CPU 拒绝控制流、随机性待裁决项和 950PR 支持范围 |

## 算子语义

设输入 COO 图为 `(row, col)`，边数为 `E`，起点数为 `S`，游走长度为 `L`。Python 层生成
CSR `rowptr`，采样结果为：

- `node_seq`：形状 `[S, L+1]`；
- `edge_seq`：形状 `[S, L]`，仅在 `return_edge_indices=True` 时向用户返回。

第 `i` 条游走满足：

```text
node_seq[i, 0] = start[i]

for step in [0, L):
    v = node_seq[i, step]
    if out_degree(v) == 0:
        node_seq[i, step + 1] = v
        edge_seq[i, step] = -1
    else:
        e = sample_out_edge(v)
        node_seq[i, step + 1] = col[e]
        edge_seq[i, step] = e
```

`e` 是排序后工作 COO/CSR 中的边位置。`coalesced=True` 时 Python 层先排序，因此
`edge_seq` 不映射回原始输入次序；`coalesced=False` 时工作次序就是输入次序。重边分别作为独立
候选，自环不删除，输入图不隐式转为无向图。

### 均匀随机游走

当 `p=q=1` 时，每一步从当前节点的全部出边中等概率选择一条。孤立节点保持不动，出度为 1 时
选择唯一边。该分支也是 node2vec 的第一步采样规则。

### node2vec 拒绝采样

从第二步开始，记上一节点为 `t`、当前节点为 `v`、候选下一节点为 `x`：

```text
M = max(1/p, 1, 1/q)
prob_0 = (1/p) / M
prob_1 = 1 / M
prob_2 = (1/q) / M

repeat:
    从 v 的出边中均匀抽取候选边 e，x = col[e]
    生成 r ∈ [0, 1)
    if x == t and r < prob_0: accept e
    else if 存在有向边 x→t and r < prob_1: accept e
    else if r < prob_2: accept e
```

谓词顺序与 `torch_cluster` CPU reference 一致。`x==t`、自环和邻接关系可能重叠，不能改写成
三个互斥类别；`is_neighbor(x,t)` 表示查询有向边 `x→t`。拒绝循环不得以固定次数后强制接受
候选，否则会改变目标分布。

## 任务书冲突与设计裁决

| 条目 | 任务书/参考行为 | 设计口径 |
| --- | --- | --- |
| 固定 seed bit-wise 与统计等价 | 任务书同时要求 CPU bit-wise，并写明允许统计等价；CPU node2vec 使用不受 `torch.manual_seed()` 控制的 C `rand()` | 设计以结构合法、目标分布和 NPU 自身固定 seed 可复现作为可执行方案；是否必须复刻 CPU 随机序列，提交验收前由任务方裁决 |
| `walk_length=0` | 任务书明确要求覆盖；`torch_cluster` 1.6.3 CPU 路径不能提供稳定合法输出 | 服从任务书输出定义，返回 `[S,1]` 节点序列和 `[S,0]` 边序列 |
| `edge_seq` | 任务书写 COO index；参考 Python 层排序后未做逆映射 | 返回排序后工作 COO/CSR 位置 |
| `p/q` 取值域 | 任务书要求 `>0`，未单列 IEEE 特殊值 | 按可计算概率的有限正实数解释；NaN 和无穷值作为参数错误 |
| 随机输出精度 | 任务书列出浮点误差比例；输出是离散 INT64 索引 | 确定性场景逐元素比较；随机场景使用结构、分布和可复现性，不对节点 ID 计算相对误差 |

## 算子原型

### Python 公开接口

```python
def random_walk(
    row: Tensor,
    col: Tensor,
    start: Tensor,
    walk_length: int,
    p: float = 1,
    q: float = 1,
    coalesced: bool = True,
    num_nodes: Optional[int] = None,
    return_edge_indices: bool = False,
) -> Union[Tensor, Tuple[Tensor, Tensor]]: ...
```

| 名称 | 类别 | 类型 / dtype | 形状与存储 | 默认值 / 取值 | 语义与约束 |
| --- | --- | --- | --- | --- | --- |
| `row` | 输入 | Tensor / INT64 | 一维 `[E]`，NPU | 必选 | COO 源节点，元素位于 `[0,num_nodes)` |
| `col` | 输入 | Tensor / INT64 | 一维 `[E]`，与 `row` 同 device | 必选 | COO 目标节点，与 `row` 等长 |
| `start` | 输入 | Tensor / INT64 | 一维 `[S]`，与图同 device | 必选 | 起点，元素位于 `[0,num_nodes)` |
| `walk_length` | 属性 | Python int | 标量 | `≥0` | 游走步数 `L`，须可安全形成输出 shape |
| `p` | 属性 | Python float | 标量 | 默认 `1`，有限正数 | 返回上一节点的权重参数 |
| `q` | 属性 | Python float | 标量 | 默认 `1`，有限正数 | BFS/DFS 插值参数 |
| `coalesced` | 属性 | Python bool | 标量 | 默认 `True` | 是否在 Python 层按 `(row,col)` 排序 |
| `num_nodes` | 属性 | Optional[int] | 标量 | 默认 `None`，显式值 `≥0` | 图节点数；缺省时按任务书公式推导 |
| `return_edge_indices` | 属性 | Python bool | 标量 | 默认 `False` | 是否同时返回 `edge_seq` |
| `node_seq` | 返回值 | Tensor / INT64 | NPU `[S,L+1]`，新分配 | 始终返回 | 游走节点序列 |
| `edge_seq` | 条件返回值 | Tensor / INT64 | NPU `[S,L]`，新分配 | 属性为 `True` 时返回 | 工作 COO/CSR 边位置；孤立步为 `-1` |

输入只读。非连续输入在 Python 层规范化为连续 Tensor；输出不与输入或彼此别名。
`return_edge_indices=False` 返回单个 Tensor，`True` 返回 `(node_seq, edge_seq)`。公开接口保持
`torch_cluster.random_walk` 的 TorchScript 可编译形态。

### PyTorch dispatcher 与 C++ binding

Python 预处理后调用任务书指定的六参数内部算子：

```text
torch_cluster::random_walk(
    Tensor rowptr, Tensor col, Tensor start,
    int walk_length, float p, float q
) -> (Tensor, Tensor)
```

对应 C++ binding 原型：

```cpp
std::tuple<at::Tensor, at::Tensor> random_walk(
    const at::Tensor& rowptr,
    const at::Tensor& col,
    const at::Tensor& start,
    int64_t walk_length,
    double p,
    double q);
```

| 名称 | 类别 | 类型 / shape | 约束 |
| --- | --- | --- | --- |
| `rowptr` | 输入 | NPU INT64 `[N+1]` | 连续、非递减，`rowptr[0]=0`、`rowptr[N]=E` |
| `col` | 输入 | NPU INT64 `[E]` | 连续，元素位于 `[0,N)` |
| `start` | 输入 | NPU INT64 `[S]` | 连续，元素位于 `[0,N)` |
| `walk_length` | 属性 | INT64 | `≥0` |
| `p/q` | 属性 | double | 与公开属性一致 |
| 返回值 | 输出 | 两个 NPU INT64 Tensor | `[S,L+1]` 和 `[S,L]`，非原地 |

binding 在当前 PyTorch NPU stream 上异步下发采样 Kernel。CPU Tensor 仍由已安装的
`torch_cluster` CPU dispatch 处理；NPU 扩展只注册对应 device dispatch，不定义第二套公开 API。
参数、shape、分配和运行时错误转换为 PyTorch 可见异常。

# 详细设计（required）

## 算子分析

数据流如下：

```text
COO row/col + start
        │
        ▼
Python：校验、num_nodes、排序、CSR rowptr
        │
        ▼
PyTorch dispatcher / PyBind：输出分配、stream、Kernel 下发
        │
        ▼
Ascend C：uniform / node2vec 采样
        │
        ▼
node_seq + edge_seq ──► Python 按属性选择返回对象
```

不同起点之间独立，可沿 `S` 维并行；同一条游走的 `L` 步有严格前后依赖。性能瓶颈主要来自
Python 预处理、CSR 随机访存、node2vec 邻接查询和拒绝重试。

## 算子实现

### Python 预处理

1. `num_nodes=None` 时执行
   `max(row.max(), col.max(), start.max()) + 1`；无法执行该公式的空输入要求显式传入
   `num_nodes`。
2. `coalesced=True` 时按 `row*num_nodes+col` 的数学顺序排序 `row/col`；实现可采用等价的
   `(row,col)` 字典序以避免整数乘加溢出。
3. 用 `deg.scatter_add` 和 `cumsum` 构造 `rowptr`。
4. 调用六参数 `torch.ops.torch_cluster.random_walk`，再按
   `return_edge_indices` 选择公开返回值。

上述步骤属于公开 API 语义和性能口径。`coalesced=False` 不排序，调用方须保证同一源节点的边连续
存放，否则 CSR 区间与 `col` 次序不一致。预处理在输入所在 NPU device 上完成，不执行图数据的
CPU fallback。

### Host 与 binding

Host 侧负责：

- 校验 dtype、rank、device、属性和值域；
- 对输出元素数、字节数和临时空间执行溢出检查；
- 分配 `[S,L+1]` 与 `[S,L]` 两个输出；
- 准备图规模、采样参数和随机数状态；
- 在当前 NPU stream 上下发 Kernel，并保持与 Python 预处理的依赖关系。

`S=0` 时直接返回空 batch；`L=0` 时写入起点列并返回空边序列，不进入采样 Kernel。

### Device 侧设计

每个逻辑 worker 维护一条游走状态，起点多于并行容量时分批覆盖。均匀路径每步只进行一次出边
选择；node2vec 路径保存上一节点，并按参考短路顺序执行候选选择、邻接查询和拒绝判定。

`coalesced=False` 不保证邻接表内 `col` 有序，因此 `is_neighbor(x,t)` 不能无条件使用二分查找；
基础方案在线性扫描与按图规模启用的等价优化之间选择。长游走或低接受率场景采用有界执行片保存
未完成状态并继续下发，执行片边界不得强制接受候选或改变随机数域。

### RNG 与可复现性

NPU 路径使用 PyTorch 当前 NPU device 的默认生成器建立调用级随机数域，并采用与线程调度无关的
counter-based 随机数映射。固定 seed、输入和调用顺序相同时，输出应逐元素可复现；并行度、分批和
执行片划分不得改变结果。`S=0` 或 `L=0` 不消费随机数。

候选边选择必须均匀，node2vec 接受判定必须使用同一 trial 的随机量和固定短路顺序。RNG 算法、
有界整数映射和概率阈值均通过独立已知答案或统计测试验证，不能用输出结构正确替代分布正确。

### 性能方案

- 以起点维作为主要并行维度，同一游走内部顺序执行；
- 为 `p=q=1` 提供无邻接查询的均匀快路径；
- 合并连续状态读写，减少每步 Host 交互；
- node2vec 邻接查询根据度数选择直接扫描或等价索引策略；
- Python 排序、CSR 构造、采样和输出形成均纳入公开 API 优化，不以预构造 CSR 的内部时间替代。

任何性能优化都必须保持工作边索引、拒绝采样分布和 RNG 可复现性。

## 支持硬件

| 支持项 | 设计范围 |
| --- | --- |
| 芯片 | Ascend 950PR |
| CANN | 目标 `cann/ops-gnn` 仓指定版本 |
| 接口 | PyTorch/PyBind NPU extension |
| 计算方向 | 前向 |

其它 SoC、反向传播和 aclnn/OPP 不在本任务范围内。

## 算子约束限制

| 场景 | 设计行为 |
| --- | --- |
| `row/col/start` 非 INT64、一维 NPU Tensor，或不在同一 device | 参数错误 |
| `row/col` 长度不同 | 参数错误 |
| `walk_length<0`，或输出 shape/字节数溢出 | 参数错误 |
| `p/q` 非有限正数 | 参数错误 |
| 显式 `num_nodes<0`，或节点值越过 `[0,num_nodes)` | 参数错误 |
| `num_nodes=None` 且任务书最大值公式无法执行 | 参数错误；调用方应显式提供节点数 |
| `coalesced=False` 但边未按源节点连续分组 | 不满足接口前置条件 |
| 孤立节点 | 后续节点保持当前节点，边索引为 `-1` |
| `walk_length=0` | 返回起点列和空边序列 |
| 自环与重边 | 保留并按工作 COO 独立处理 |
| 拒绝采样长时间未接受 | 保存状态并继续执行，不以固定上限改变分布 |

# 可维可测分析

## 精度标准/性能标准

### 精度标准

验收对象始终是公开 `ops_gnn.random_walk()`，测试分为四层：

| 层级 | 入口 | 判据 |
| --- | --- | --- |
| 公开功能 | 官方 pytest/PyTorch API | import、device dispatch、返回类型、shape、边界与错误行为 |
| 语义与随机性 | PyTorch API + 独立 oracle | 图结构不变量、确定性场景逐元素比较、采样分布和自身可复现性 |
| 内部单元 | binding/RNG/采样组件 | schema、RNG 已知答案、有界整数映射、拒绝控制流和执行片延续 |
| 交付重放 | clean checkout + wheel | 构建、安装后按同一公开入口复现 |

任务书指定 AscendOpTest 组织精度测试。若 AscendOpTest 不能直接调用 PyBind extension，则增加只负责
装载 `ops_gnn` 和转发参数的薄适配层，不能改写 Python 预处理、seed 或返回语义。官方公开用例
`case_M8/ops-gnn/test/test_random_walk.py` 作为入口兼容性门禁，但其 device、shape 和返回类型断言
不能替代下面的结构与分布判据。

单 CPU 标杆不能形成有效判据时，按任务书进入 ATK 双标杆流程。确定性场景在指标有定义时使用
最大相对误差比例 `≤2`、平均相对误差比例 `≤1.2`、均方根误差比例 `≤1.2`；随机 INT64 路径
不把节点 ID 数值距离当作误差，仍按结构和分布判定并保留任务方裁决。

功能与精度 case：

| 编号 | 场景 | 主要判据 |
| --- | --- | --- |
| TC-01 | `p=q=1` | uniform 分支、合法边、等概率分布、固定 seed 可复现 |
| TC-02 | `p≠1` 或 `q≠1` | CPU 短路谓词、候选边理论分布、固定 seed 可复现 |
| TC-03 | `return_edge_indices=True/False` | 返回类型、shape、节点与工作边位置一致、输入只读 |
| TC-04 | `coalesced=False` | 已按源节点分组的无序邻接表，工作边位置和分布正确 |
| TC-05 | 孤立节点 | 节点保持，`edge_seq=-1` |
| TC-06 | `walk_length=0` | `[S,1]`、`[S,0]` 和起点值 |
| TC-07 | 多 start 大规模 | 公开 API 功能与性能 |
| TC-08 | 自环、重边、出度 1 | 重叠谓词、独立候选边和确定性分支 |
| TC-09 | 空边、空起点、显式/缺省 `num_nodes` | 合法空输出或明确参数错误 |
| TC-10 | dtype、rank、device、范围和属性错误 | 在排序、分配或采样前报错 |
| TC-11 | TorchScript 与 dispatcher 共存 | Python 签名、双 Tensor binding、CPU/NPU dispatch 不冲突 |

随机输出不使用节点 ID 或边 ID 的相对误差。uniform 和 node2vec 分别根据独立推导的候选边概率
执行拟合优度检验；样本量保证每个有效类别有足够期望频数，并对多 case 显著性做校正。

任务书要求的 CPU fixed-seed bit-wise 对 node2vec 不是稳定可执行 oracle。若任务方裁决必须逐位
一致，则需同时冻结 CPU reference 版本、libc、线程数、进程内调用顺序和 RNG 状态，并据此重新
设计 NPU 随机数流；在此之前不把统计等价写成 CPU bit-wise 已满足。

### 性能标准

官方性能入口调用完整 `ops_gnn.random_walk(row, col, start, walk_length)`，使用默认
`p=q=1`、`coalesced=True`、`num_nodes=None`、`return_edge_indices=False`。输入驻留 NPU，
计时包括 Python 预处理、binding、采样和输出形成。

| `E` | `L` | `S` | 随机节点上界 | A100 标杆 ms | 最大允许时间 `A100/0.6` ms |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 65,536 | 128 | 8,192 | 16,384 | 0.576 | 0.960000 |
| 262,144 | 128 | 16,384 | 65,536 | 0.709 | 1.181667 |
| 524,288 | 128 | 32,768 | 131,072 | 1.139 | 1.898333 |
| 524,288 | 256 | 32,768 | 131,072 | 1.867 | 3.111667 |

按官方 benchmark 先预热 20 次，在计时前后同步 NPU，连续调用 100 次并取算术平均。四个 case
分别判定，均须不超过对应上限；不使用预构造 CSR、裸 Kernel 时间或跨 case 平均值替代。
设备号等环境映射可以调整，但不得修改 shape、输入生成、公开调用、预热、迭代和同步边界。

## 兼容性分析

- Python 参数名、顺序、默认值和返回联合类型与任务书一致；
- CSR 仍在 Python 层构造，六参数 dispatcher schema 不变；
- NPU 扩展只增加 RandomWalk device dispatch，不改变其它 `ops-gnn` API；
- 已存在 CPU `torch_cluster` schema 时复用同一 schema，CPU/NPU Tensor 命中各自实现；
- TorchScript、eager 和两种公开返回形式保持同一预处理与采样语义；
- 设计变更涉及 RNG、边索引、拒绝控制流或 Python 预处理时，必须回归对应功能、统计和性能 case。
