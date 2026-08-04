# 8月社区任务 - random_walk 算子开发任务书

## 任务概述

实现与 `torch_cluster.random_walk`（https://github.com/rusty1s/pytorch_cluster ，版本建议≥ 1.6.0） **接口完全一致** 的 NPU 版本：在 CSR 图上从 `start` 节点出发，采样长度为 `walk_length` 的随机游走（支持 node2vec 偏置 `p`/`q`）。

**验收口径**：PyTorch 层 `random_walk()` 为准。

### 功能定义

- 输入：COO 边 `(row, col)`，经 Python 预处理为 CSR `rowptr`
- 输出：
  - `node_seq`：`[num_starts, walk_length+1]` 节点序列
  - 可选 `edge_seq`：`[num_starts, walk_length]` 边索引（`return_edge_indices=True`）

当 `p=1, q=1` 时为**均匀随机游走**；否则为 **node2vec 拒绝采样**。

---

## 核心开发要求及验收标准

开发/测试硬件和软件要求：
- **适配硬件**：Ascend 950PR
- **CANN 版本**：算子开源仓（https://gitcode.com/cann/ops-gnn ）指定版本

### 1. PyTorch 层接口（**必选**）

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

### 2. Python 预处理（须复现）

1. `num_nodes` 默认 `max(row.max(), col.max(), start.max())+1`
2. `coalesced=True`：按 `row*num_nodes+col` 排序边
3. 构造 CSR：`deg.scatter_add` → `rowptr`
4. 调用 `torch.ops.torch_cluster.random_walk(rowptr, col, start, walk_length, p, q)`

### 3. 参数与约束

| 参数 | 类型 | 约束 |
|------|------|------|
| `row` | Long `[E]` | 源节点 |
| `col` | Long `[E]` | 目标节点 |
| `start` | Long `[S]` | 起始节点，`0 ≤ start[i] < num_nodes` |
| `walk_length` | int | `≥ 0` |
| `p` | float | `> 0`；返回上一节点（t→x→t）惩罚 |
| `q` | float | `> 0`；BFS/DFS 插值（node2vec） |
| `coalesced` | bool | 默认 True，排序边 |
| `num_nodes` | Optional int | 图节点数 |
| `return_edge_indices` | bool | 是否返回边序列 |

| 输出 | 形状 | 说明 |
|------|------|------|
| `node_seq` | `[S, walk_length+1]` | 游走节点 ID；孤立点后续步重复或 -1（与 CPU 一致） |
| `edge_seq` | `[S, walk_length]` | 边在 COO 中的 index；无邻居时为 -1 |

**数据类型**：row/col/start/输出均为 **int64**，**无 float64 特征**；全程 NPU，**不涉及 float CPU 回退**。

### 4. 算子约束

1. **含随机性**：均匀采样与拒绝采样均依赖 RNG；验收固定 seed；
2. `p=1, q=1` 须走均匀采样路径（`uniform_sampling`）；
3. 孤立节点（度为 0）：后续 `edge_seq=-1`，节点序列保持当前节点（见 CPU 实现）；
4. 仅验收前向。

### 5. 功能验收用例

| 编号 | 场景 | 参考 |
|------|------|------|
| TC-01 | 均匀游走 p=q=1 | `test/test_rw.py` |
| TC-02 | node2vec p≠1 或 q≠1 | 同上 |
| TC-03 | return_edge_indices=True | 同上 |
| TC-04 | coalesced=False | 自行构造 |
| TC-05 | 孤立节点 | 自行构造 |
| TC-06 | walk_length=0 | 自行构造 |
| TC-07 | 多 start 大规模 | 性能用例 |

### 6. 性能要求

- 算子所有用例的性能需大于等于0.6倍标杆性能。
- GPU（A100）的耗时如下：

图上随机游走。数据：随机边 + 随机起点。

| shape (边数, 步数, 起点数) | 耗时 | 输出元素 |
|-------|------|----------|
| E=64K walk=128 start=8K | 0.576ms | 1.0M |
| E=256K walk=128 start=16K | 0.709ms | 2.1M |
| E=512K walk=128 start=32K | 1.139ms | 4.2M |
| E=512K walk=256 start=32K | 1.867ms | 8.4M |

验收取最优实现。全部为 int64 图操作，无 float64 回退项。

### 7. 精度要求

满足《生态算子开源精度标准（https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md ）》；仿照已有用例自行编写Python用例进行自测试，使用单标杆精度进行验收。

**真值生成方式**：以 CPU 版 torch_cluster 为标杆；NPU 结果通过 **PyTorch 层 `random_walk()`** 调用获取并与标杆比对。须 **固定 RNG seed**。

| 数据类型 / 场景 | 精度策略 |
|-----------------|----------|
| node_seq / edge_seq（int64） | 固定 seed 下与 CPU 标杆 **bit-wise 一致** |
| 均匀游走（p=q=1） | 与 CPU 采样路径一致 |
| node2vec（p≠1 或 q≠1） | 拒绝采样逻辑与 CPU 一致；固定 seed bit-wise |
| 孤立节点 | `edge_seq=-1`、节点序列行为与 CPU 一致 |

---


## 验收交付件

在社区任务IT系统中提交验收时， 需要提交以下交付件：

| 序号 | 交付件名称 | 交付件要求 |
|------|-----------|------------|
| 1 | 算子设计文档 | 1. 设计文档模板：https://gitcode.com/cann/cann-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md ；<br> 2. 在cann-competitions 仓库（https://gitcode.com/cann/cann-competitions/tree/master/04_tasks/01_community-task-2026/tasklist ）以PR形式提交设计文档，通过评审后合入仓库，详细说明见：https://gitcode.com/cann/cann-competitions/blob/master/04_tasks/01_community-task-2026/README.md|
| 2 | 自测用例及测试代码 |1. 需要清晰列出精度测试case和性能测试case；<br> 2. 测试代码中的readme文件需要说明测试步骤，保证验收人可以复现测试结果|
| 3 | 自测报告 | 1. 自测报告模板：https://docs.qq.com/sheet/DUmVWWndaUE12WGFB?tab=BB08J2 ；<br> 2. 需要包含用例参数、精度对比结果及截图、性能数据及截图 |
| 4 | 待验收代码地址 | 1. 个人代码仓链接、分支、算子目录；需要在个人仓邀请账号Ascend-CANN作为开发者，如下图所示； <br> 2. 算子目录下需要提供readme文件，参考：https://gitcode.com/cann/ops-transformer/blob/master/attention/chunk_gated_delta_rule/README.md <br> |

![邀请示意](./pics/invite.jpeg)


### PR 申请合入

- PR 路径建议：`ops-gnn/random_walk`（导出 API 名：**`random_walk`**）

## 参考资料

- `torch_cluster/rw.py`、`csrc/cpu/rw_cpu.cpp`、`csrc/cuda/rw_cuda.cu`
- node2vec 采样说明：https://louisabraham.github.io/articles/node2vec-sampling.html

## 环境获取

1. 使用 hidevlab webIDE 算力：https://hidevlab.huawei.com/online-develop-intro?from=hiascend ，点击 "体验 WebIDE"；
   - 如果是新用户，在申请权限的时候需要备注使用的算力类型（A2、A3或者950）。申请内容示例：本人gitcode账号是 yolo，现在参与社区任务"7月社区任务-aclnnRoll算子开发"，需要申请A2和950算力进行任务开发。
   - 如果是老用户且需要使用950算力，需要向昇腾CANN小助手反馈账号名（个人中心->基本信息，如下图所示），后台会添加账号至950使用白名单。

   ![环境截图](./pics/zaixiankaifa1.png)  
   ![账号名](./pics/account.png)  

2. 开源仓提供100小时免费时长，请不使用时及时关闭，用时耗尽前请务必保存相关资料，建议及时提交备份。

   ![环境截图](./pics/yunkaifa.png)

3. 申请算力资源后一般在1-2个工作日完成审批，如果没有审批完成，请及时在任务对应的讨论帖留言或者联系昇腾CANN小助手。

## 特别注意事项

1. **CSR 预处理在 Python 层**，与 torch_cluster 原版一致；
2. `p/q` 拒绝采样逻辑须与 CPU 参考实现一致；
3. 随机游走结果允许统计等价验收（固定 seed 下 bit-wise）；
4. 开发须严格遵循 Ascend C 编程规范、Python/PyTorch 扩展开发规范及 Ascend 950PR 算子开发相关要求；
5. 所有交付件须提前完成自验证，确认符合验收标准后再提交；
6. 开发前务必阅读【社区任务】流程及注意事项（https://gitcode.com/org/cann/discussions/39 ）。
