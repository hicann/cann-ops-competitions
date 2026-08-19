# GraclusCluster 算子设计说明

## 需求背景（required）

### 需求来源

- 通过 CANN 社区 8 月任务完成 `graclus_cluster` 算子的 Ascend C NPU 开发，实现与 `torch_cluster.graclus_cluster`（≥1.6.0）接口完全一致的 NPU 版本。
- 任务要求适配 Ascend 950PR，性能不低于 GPU（A100）参考耗时的 60%。

### 背景介绍

#### 算子功能

`graclus_cluster` 是图贪心聚类算子。按随机顺序遍历未标记顶点，将其与未标记邻居中边权最大者配对（无权重时取首个未标记邻居）。每个节点最终被分配一个聚类 ID，配对的两个节点共享相同 ID，未配对节点以自身为聚类。

- 输入：`row`、`col`、`weight`（可选）、`num_nodes`（可选）
- 输出：`cluster`
- Kernel 输入（预处理后）：`rowptr`、`col`、`weight`（可选）、`perm`

计算规则可表示为：

$$
\text{cluster}[v] = \text{cluster}[u^*] = \min(v, u^*), \quad u^* = \arg\max_{u \in \mathcal{N}(v),\ \text{unmarked}(u)} w(v,u)
$$

无 weight 时：

$$
u^* = \text{first unmarked neighbor of } v \text{ in CSR order}
$$

未配对节点：$\text{cluster}[v] = v$

#### GraclusCluster 算子实现路径

对标参考实现：
- Python 层：`torch_cluster/graclus.py`
- CPU 实现：`torch_cluster/csrc/cpu/graclus_cpu.cpp`
- CUDA 实现：`torch_cluster/csrc/cuda/graclus_cuda.cu`

当前 AscendC 工程路径：

```text
ops-gnn/graclus/
```

#### GraclusCluster 算子现状分析

基于 `torch_cluster` 源码可归纳：

- Python 预处理步骤：
  1. `num_nodes` 默认 `max(row.max(), col.max()) + 1`；
  2. 去除自环：`mask = row != col`；
  3. 无 weight 时：`torch.randperm` 随机打乱边顺序；
  4. 按 `row` 排序转 CSR：得到 `rowptr[N+1]`、排序后的 `col[E]`（及 `weight[E]`）；
  5. 调用底层 C++/CUDA kernel。
- CPU kernel 实现（`graclus_cpu.cpp`）：
  - 生成节点遍历顺序；
  - 对每个未标记节点按策略选择邻居配对；
  - 配对后更新聚类标记。
- CUDA kernel 实现（`graclus_cuda.cu`）：
  - 采用并行贪心匹配方案，多轮迭代完成配对。

Python 预处理与 Kernel 分工：

| 阶段 | 执行位置 | 职责 |
| --- | --- | --- |
| 去自环 | Python 层 | 过滤 `row == col` 的边 |
| 边 shuffle（无 weight） | Python 层 | 打乱边顺序 |
| COO → CSR 转换 | Python 层 | 排序得 `rowptr`、`col`、`weight` |
| 节点遍历序生成 | Python 层 | 确定遍历顺序 |
| 贪心匹配 | Kernel（NPU） | 核心配对逻辑 |

#### 总体流程图

![image.png](https://raw.gitcode.com/user-images/assets/10331120/1cef0bdd-0fc5-4963-8617-b154b13a783b/image.png 'image.png')

## 需求分析

### 外部组件依赖

- 无新增外部组件依赖。

### 内部适配模块

- 适配 PyTorch 自定义算子接口（`torch.ops` 注册）。
- 覆盖有权/无权、L1/L2 dtype、空边、孤立节点等场景。
- 保证固定 seed 下与 CPU 版本 bit-wise 一致。

### 需求模块设计

#### 算子原型

| 名称 | 类别 | dtype | format | shape | 说明 |
| --- | --- | --- | --- | --- | --- |
| rowptr | 输入 | int64 | ND | `[N+1]` | CSR 行指针，单调递增 |
| col | 输入 | int64 | ND | `[E]` | 排序后的列索引 |
| weight | 输入（可选） | float16/bfloat16/float32 | ND | `[E]` | 边权重 |
| perm | 输入 | int64 | ND | `[N]` | 节点遍历序 |
| cluster | 输出 | int64 | ND | `[N]` | 聚类分配结果 |

相关约束：

- `rowptr` 长度为 `num_nodes + 1`，单调非递减。
- `col` 所有值在 `[0, num_nodes)` 范围内。
- `weight` 若存在，长度与 `col` 一致。
- `perm` 为 `[0, num_nodes)` 的一个排列。
- `cluster` 输出满足：每个聚类最多 2 个节点，同聚类节点在图中相邻。
- 有 weight 时选 weight 最大的未标记邻居配对。
- 无 weight 时选 CSR 中第一个未标记邻居。

## 需求详细设计

### 使能方式

| 上层框架 | 涉及的框架勾选 |
| --- | --- |
| TF 训练/推理 |  |
| Pytorch 训练/推理 | √ |
| ATC 推理 |  |
| Aclnn 直调 |  |
| OPAT 调优 |  |
| SGAT 子图切分 |  |

### 需求总体设计

#### AscendC host 侧设计

代码位置：`ops-gnn/graclus/op_host/`

Host 文件职责拆分：

| 文件 | 主要职责 | 关键点 |
| --- | --- | --- |
| `graclus_cluster_def.cpp` | 算子注册与配置 | 注册输入/输出、硬件配置 |
| `graclus_cluster_tiling.cpp` | tiling 生成 | 解析图规模、dtype，生成搬运策略参数 |
| `graclus_cluster_tiling_data.h` | tiling 数据结构 | host/kernel 共享参数定义 |

Host 核心执行链路：

1. 读取输入 shape 推断图规模（N、E）。
2. 判断 `weight` 是否存在及其 dtype，确定 tiling key。
3. 获取平台信息（UB 容量等），评估数据搬运策略。
4. 根据图规模与 UB 容量的关系选择合适的执行策略。
5. 填充 TilingData，设置 blockDim。

`GraclusClusterTilingData` 关键字段：

| 字段 | 用途 |
| --- | --- |
| `numNodes` | 节点数 N |
| `numEdges` | 边数 E |
| `hasWeight` | 是否有边权重 |
| `strategyFlag` | 执行策略标记（根据图规模与 UB 容量选择） |

Tiling key 规划：

| Tiling Key | kernel 模板 | 用途 |
| --- | --- | --- |
| 0 | `GraclusCluster<NoWeight>` | 无权图 |
| 1 | `GraclusCluster<float16_t>` | float16 权重 |
| 2 | `GraclusCluster<bfloat16_t>` | bfloat16 权重 |
| 3 | `GraclusCluster<float>` | float32 权重 |

Host 侧流程图：

![image.png](https://raw.gitcode.com/user-images/assets/10331120/e3019649-3f8b-45d2-919e-bb377439899f/image.png 'image.png')

#### AscendC kernel 侧设计

代码位置：`ops-gnn/graclus/op_kernel/`

入口分发文件：`op_kernel/graclus_cluster.cpp`。
根据模板 tiling key 实例化不同 kernel 模板：

| Tiling Key | 实例化类型 | 说明 |
| --- | --- | --- |
| 0 | `GraclusCluster<NoWeight>` | 无权图，纯 int64 逻辑 |
| 1 | `GraclusCluster<float16_t>` | fp16 权重 |
| 2 | `GraclusCluster<bfloat16_t>` | bf16 权重 |
| 3 | `GraclusCluster<float>` | fp32 权重 |

Kernel 共性策略：

- 遵循 `Init → Process` 生命周期。
- `Init` 阶段绑定 GM 地址、读取 tiling 参数、初始化 UB buffer。
- `Process` 阶段根据 `strategyFlag` 选择对应处理路径。
- 核心匹配逻辑保证与 CPU 版本语义一致。
- 搬运使用 `DataCopy` / `DataCopyPad` 处理对齐。

Kernel 设计要点：

- 根据图规模选择数据常驻或按需搬运策略，平衡 UB 利用率与搬运开销。
- 匹配过程中需维护节点状态标记，确保已配对节点不被重复访问。
- 有权模式需在邻居集合中进行比较选择；无权模式仅需定位首个可用邻居。
- 输出初始化为自身索引，配对成功时更新为 `min(v, u*)`。

Kernel 执行流程图：

![image.png](https://raw.gitcode.com/user-images/assets/10331120/7b50ff6a-44d5-4ddb-9215-5b4bccc2c76d/image.png 'image.png')

### 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Ascend 950PR | √ |

### 算子约束限制

- 当前工程支持 `weight` 为 `float32/bfloat16/float16` 或空（无权图）。
- 当前工程支持 `rowptr/col/perm/cluster` 为 `int64`。
- 算法含随机性（节点遍历顺序由 `perm` 决定），固定 seed 可复现。
- 有 weight 时选 weight 最大的未标记邻居配对。
- 无 weight 时选第一个未标记邻居（CSR 顺序）。
- 仅验收前向，不涉及反向传播。
- 输入图须为无向图（边需成对出现）。
- 每个聚类最多 2 个节点。
- 接口名必须为 `graclus_cluster`。
- L2 float64 weight 通过 CPU 回退处理，不进入 NPU kernel。

## 特性交叉分析、可维可测分析

### 精度标准 / 性能标准

| 验收标准 | 描述（不涉及说明原因） | 标准来源 |
| --- | --- | --- |
| 精度标准 | 固定 seed 下 cluster 输出与 CPU 标杆 bit-wise 一致 | 任务书要求 |
| 性能标准 | 所有用例性能 ≥ 0.6× GPU（A100）参考耗时 | 任务书要求 |

性能参考基准（GPU A100，完全图，权重全 1）：

| shape (节点数, 边数) | GPU 耗时 |
| --- | --- |
| V=4 E=12 | 0.400ms |
| V=8 E=56 | 0.507ms |
| V=16 E=240 | 0.705ms |
| V=32 E=992 | 1.234ms |
| V=64 E=4032 | 2.825ms |

### 可测性分析

- 测试目录：`ops-gnn/graclus/tests/`
- 精度测试：固定 seed 对比 CPU `torch_cluster.graclus_cluster` 结果
- 性能测试：完全图场景计时对比 GPU 基准
- 真值生成：`torch.manual_seed(seed)` + CPU 版 `graclus_cluster()`

建议覆盖场景：

| 编号 | 场景 | 覆盖点 |
| --- | --- | --- |
| TC-01 | 小图无 weight | V=4, E=12，无权取首个邻居 |
| TC-02 | 带 weight | V=4, E=12，最大权配对 |
| TC-03 | 指定 num_nodes | 含孤立节点 |
| TC-04 | 含自环 | Python 层过滤验证 |
| TC-05 | float64 weight | L2 CPU 回退 |
| TC-06 | 空边 | E=0，所有节点自成聚类 |
| TC-07 | float16 weight | L1 NPU 路径 |
| TC-08 | bfloat16 weight | L1 NPU 路径 |
| TC-09 | 中等规模 | V=32, E=992 性能验证 |
| TC-10 | 较大规模 | V=64, E=4032 性能验证 |

### 兼容性分析

- 新算子开发，不涉及旧接口破坏；
- 接口名 `graclus_cluster` 与 torch_cluster 保持一致；
- 合入路径 `ops-gnn/graclus`，导出 API 名 `graclus_cluster`。
