# 8月社区任务 - radius 算子开发任务书

## 任务概述

实现与 `torch_cluster.radius`、`radius_graph`（https://github.com/rusty1s/pytorch_cluster ，版本建议≥ 1.6.0） **接口完全一致** 的 NPU 版本：对 `y` 中每点，找 `x` 中距离 **< r** 的所有邻居（上限 `max_num_neighbors`）。

**验收口径**：PyTorch 层为准；须覆盖 `radius` + `radius_graph`。

### 功能定义

- 距离：**欧氏距离**（`‖x-y‖ < r`）
- 超出 `max_num_neighbors` 时 **随机** 采样邻居（与 GPU 一致，非确定性）
- 输出：`edge_index [2, E]`

---

## 核心开发要求及验收标准

开发/测试硬件和软件要求：
- **适配硬件**：Ascend 950PR
- **CANN 版本**：算子开源仓（https://gitcode.com/cann/ops-gnn ）指定版本

### 1. PyTorch 层接口（**必选**）

```python
def radius(
    x: torch.Tensor,
    y: torch.Tensor,
    r: float,
    batch_x: Optional[torch.Tensor] = None,
    batch_y: Optional[torch.Tensor] = None,
    max_num_neighbors: int = 32,
    num_workers: int = 1,
    batch_size: Optional[int] = None,
    ignore_same_index: bool = False,
) -> torch.Tensor: ...

def radius_graph(
    x: torch.Tensor,
    r: float,
    batch: Optional[torch.Tensor] = None,
    loop: bool = False,
    max_num_neighbors: int = 32,
    flow: str = 'source_to_target',
    num_workers: int = 1,
    batch_size: Optional[int] = None,
) -> torch.Tensor: ...
```

### 2. 参数与约束

| 参数 | 类型 | 约束 |
|------|------|------|
| `x` | Float `[N, F]` | L1 float16/32；L2 float64 可选 CPU 回退或低精度拼接 |
| `y` | Float `[M, F]` | 同 F |
| `r` | float | `> 0`；搜索半径 |
| `batch_x` / `batch_y` | Optional Long | **须 sorted** |
| `max_num_neighbors` | int | 默认 32；每 y 点最多返回邻居数 |
| `num_workers` | int | NPU 可忽略 |
| `ignore_same_index` | bool | True 时忽略 `x[i]` 与 `y[i]` 同 index（`radius_graph` 用 `not loop`） |

**空输入**：返回 `[2, 0]` LongTensor。

**Host 逻辑**：同 knn，构造 `ptr_x`/`ptr_y` 后调用 `torch.ops.torch_cluster.radius(...)`。

### 3. `radius_graph` 约束

- 内部调用 `radius(x, x, r, batch, batch, max_num_neighbors, ..., ignore_same_index=not loop)`
- `flow` 同 knn_graph

### 4. 数据类型分级

| 级别 | dtype | 路径 |
|------|-------|------|
| L1 | float16/bfloat16/float32 | NPU |
| L2 | float64 | CPU 回退 或 NPU 低精度拼接（**不做性能考核**） |

### 5. 算子约束

1. batch sorted；同 batch 内搜索；
2. 邻居数超限时的 **随机子采样** 须与 torch_cluster 语义一致（允许非确定性）；
3. `r` 为 float 标量（非 Tensor）；
4. 仅验收前向。

### 6. 功能验收用例

| 编号 | 场景 | 参考 |
|------|------|------|
| TC-01 | 基础 radius | `test/test_radius.py` |
| TC-02 | batch | 同上 |
| TC-03 | max_num_neighbors 截断 | 同上 |
| TC-04 | ignore_same_index | 同上 |
| TC-05 | radius_graph loop/flow | 同上 |
| TC-06 | float64 L2 路径 | 自行构造 |
| TC-07 | 空输入 | 自行构造 |

### 7. 性能要求

- 算子所有用例的性能需大于等于0.6倍标杆性能。
- GPU（A100）的耗时如下：

3D 空间里按距离建邻接图。数据：随机 3D 坐标。

| shape (点数, 搜索半径) | float32 | float16 |
|-------|---------|---------|
| N=8K r=0.8 | 2.118ms | 2.269ms |
| N=16K r=0.5 | 3.916ms | 4.424ms |
| N=32K r=0.3 | 8.446ms | 10.082ms |
| N=64K r=0.2 | 20.830ms | 28.418ms |
| N=64K r=0.3 | 20.902ms | 28.470ms |


验收取最优实现（grid/hash cell、SIMT brute-force 等）。

### 8. 精度要求

算子计算精度需严格满足《生态算子开源精度标准（https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md ）》，采用 AscendOpTest（https://gitcode.com/HIT1920/AscendOpTest ） 测试。

**真值生成方式**：以 CPU 版 torch_cluster 为标杆；NPU 结果通过 **PyTorch 层 `radius()` / `radius_graph()`** 调用获取并与标杆比对。

**单标杆不满足时**：采用 ATK（https://gitcode.com/AscendTest/ATK ） 双标杆比对（`cv_fused_double_benchmark`），以更高精度的 CPU 实现为真值，同时评估同精度 CPU 与 NPU 算子实现相对于该真值的误差；满足条件为 NPU/同精度 CPU 的**最大相对误差比例 ≤ 2**、**平均相对误差比例 ≤ 1.2**、**均方根误差比例 ≤ 1.2**。

| 数据类型 / 场景 | 精度策略 |
|-----------------|----------|
| L1 float16/bfloat16/float32 | 每个查询点半径内 **邻居集合** 与 CPU 标杆一致；超限随机截断时 **集合等价**（固定 seed 下 bit-wise） |
| L2 float64 | CPU 回退 bit-wise 一致；低精度拼接 README 文档化精度策略 |
| `radius_graph` | loop/flow 组合后 edge 集合与 CPU 一致 |

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

- PR 路径建议：`ops-gnn/radius`

## 参考资料

- `torch_cluster/radius.py`、`csrc/cpu/radius_cpu.cpp`、`csrc/cuda/radius_cuda.cu`

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

1. PointNet++/DGCNN 等常用 **radius_graph**，须与 `radius` 一并验收；
2. 超限随机采样导致浮点验收用 **集合等价** 而非 edge 顺序一致；
3. `ignore_same_index` 与 `loop` 参数关系须与原版一致；
4. 开发须严格遵循 Ascend C 编程规范、Python/PyTorch 扩展开发规范及 Ascend 950PR 算子开发相关要求；
5. 所有交付件须提前完成自验证，确认符合验收标准后再提交；
6. 开发前务必阅读【社区任务】流程及注意事项（https://gitcode.com/org/cann/discussions/39 ）。
