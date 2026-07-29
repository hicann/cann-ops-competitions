# 8月社区任务 - knn 算子开发任务书

## 任务概述

实现与 `torch_cluster.knn`、`knn_graph`（https://github.com/rusty1s/pytorch_cluster ，版本建议≥ 1.6.0） **接口完全一致** 的 NPU 版本：对 `y` 中每个点，在 `x` 中找 **k 个最近邻**（欧氏或余弦距离）。

**验收口径**：PyTorch 层为准；须同时覆盖 `knn` 与 `knn_graph`（后者为 Python 组合逻辑，须复现）。

### 功能定义

- 距离默认：**欧氏距离**；`cosine=True` 时用余弦距离（**NPU 须支持**，CPU C++ 不支持 cosine）
- 输出 `knn`：`edge_index` 形状 `[2, M×k]`，`(row, col) = (y_idx, x_neighbor_idx)` 语义与原版一致

---

## 核心开发要求及验收标准

开发/测试硬件和软件要求：
- **适配硬件**：Ascend 950PR
- **CANN 版本**：算子开源仓（https://gitcode.com/cann/ops-gnn ）指定版本

### 1. PyTorch 层接口（**必选**）

```python
def knn(
    x: torch.Tensor,
    y: torch.Tensor,
    k: int,
    batch_x: Optional[torch.Tensor] = None,
    batch_y: Optional[torch.Tensor] = None,
    cosine: bool = False,
    num_workers: int = 1,
    batch_size: Optional[int] = None,
) -> torch.Tensor: ...

def knn_graph(
    x: torch.Tensor,
    k: int,
    batch: Optional[torch.Tensor] = None,
    loop: bool = False,
    flow: str = 'source_to_target',
    cosine: bool = False,
    num_workers: int = 1,
    batch_size: Optional[int] = None,
) -> torch.Tensor: ...
```

### 2. `knn` 参数与约束

| 参数 | 类型 | 约束 |
|------|------|------|
| `x` | Float `[N, F]` | 候选点集；L1 float16/32；L2 float64 可选 CPU 回退或低精度拼接 |
| `y` | Float `[M, F]` | 查询点集；`F` 与 `x` 相同 |
| `k` | int | `> 0`；`k ≤ N`（batch 内） |
| `batch_x` | Optional Long `[N]` | **须 sorted**（非降序） |
| `batch_y` | Optional Long `[M]` | **须 sorted** |
| `cosine` | bool | NPU 须支持；CPU 回退路径可报错或 scipy 替代 |
| `num_workers` | int | NPU 路径可忽略（保留参数） |
| `batch_size` | Optional int | 多 batch 时由 batch 向量推断 |

**空输入**：`x.numel()==0` 或 `y.numel()==0` → 返回 `[2, 0]` LongTensor。

**Host 逻辑（须复现）**：

- 1D 输入 auto `view(-1, 1)`；
- `batch_size > 1` 时用 `bucketize` 构造 `ptr_x`、`ptr_y`；
- 调用 `torch.ops.torch_cluster.knn(x, y, ptr_x, ptr_y, k, cosine, num_workers)`。

### 3. `knn_graph` 约束

| 参数 | 约束 |
|------|------|
| `flow` | `'source_to_target'` 或 `'target_to_source'` |
| `loop` | False 时调用 `knn(..., k+1)` 并去除自环 |
| 其余 | 同 `knn` |

### 4. 数据类型分级

| 级别 | dtype | 路径 |
|------|-------|------|
| L1 | float16/bfloat16/float32 | NPU（含 cosine=True） |
| L2 | float64 | CPU 回退 或 NPU 低精度拼接（**不做性能考核**） |
| 输出/ batch | int64 | NPU |

### 5. 算子约束

1. **batch 必须 sorted**，否则行为未定义（与文档一致）；
2. 同 batch 内 `x`、`y` 配对搜索，禁止跨 batch；
3. `cosine=True` 时 NPU 必须实现（GNN 常用）；
4. 仅验收前向。

### 6. 功能验收用例

| 编号 | 场景 | 参考 |
|------|------|------|
| TC-01 | 基础 2D knn | `test/test_knn.py` |
| TC-02 | batch 多图 | 同上 |
| TC-03 | cosine=True | 同上（NPU 必测） |
| TC-04 | knn_graph loop/flow | 同上 |
| TC-05 | 空输入 | 自行构造 |
| TC-06 | float64 L2 路径 | 自行构造 |
| TC-07 | k=1 | 自行构造 |

### 7. 性能要求

- 算子所有用例的性能需大于等于0.6倍标杆性能。
- GPU（A100）的耗时如下：

每个点找 k 个最近邻。数据：随机坐标，在同一个点集里找。

| shape (点数 × 维度, k) | float32 | float16 |
|-------|---------|---------|
| N=1K F=3 k=4 | 0.815ms | 0.884ms |
| N=1K F=16 k=8 | 2.677ms | 1.998ms |
| N=4K F=32 k=8 | 29.138ms | 16.150ms |
| N=4K F=64 k=16 | 59.881ms | 59.079ms |
| N=16K F=32 k=16 | 127.436ms | 71.591ms |

### 8. 精度要求

算子计算精度需严格满足《生态算子开源精度标准（https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md ）》，采用 AscendOpTest（https://gitcode.com/HIT1920/AscendOpTest ） 测试。

**真值生成方式**：以 CPU 版 torch_cluster 为标杆；NPU 结果通过 **PyTorch 层 `knn()` / `knn_graph()`** 调用获取并与标杆比对。

**单标杆不满足时**：采用 ATK（https://gitcode.com/AscendTest/ATK ） 双标杆比对（`cv_fused_double_benchmark`），以更高精度的 CPU 实现为真值，同时评估同精度 CPU 与 NPU 算子实现相对于该真值的误差；满足条件为 NPU/同精度 CPU 的**最大相对误差比例 ≤ 2**、**平均相对误差比例 ≤ 1.2**、**均方根误差比例 ≤ 1.2**。

| 数据类型 / 场景 | 精度策略 |
|-----------------|----------|
| L1 float16/bfloat16/float32 | 每个查询点的 **k 邻居集合** 与 CPU 标杆一致；距离平局时 edge 顺序可不同 |
| L2 float64 | CPU 回退 bit-wise 一致；低精度拼接 README 文档化精度策略 |
| `cosine=True` | 邻居集合与 CPU/GPU 标杆一致（NPU 必测） |
| `knn_graph` | 自环剔除、flow 转置后 edge 集合与 CPU 一致 |

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

- PR 路径建议：`ops-gnn/knn`
- 须同时交付 `knn` + `knn_graph` Python 层

## 参考资料

- `torch_cluster/knn.py`、`csrc/cpu/knn_cpu.cpp`（nanoflann）、`csrc/cuda/knn_cuda.cu`

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

1. **cosine 为 NPU 必实现项**（与 CPU C++ 能力不对齐，但 GPU 有）；
2. `num_workers` 参数保留但不影响 NPU 语义；
3. 输出 edge_index 顺序允许与 GPU 不同（若距离平局），但邻居集合须一致。
4. 开发须严格遵循 Ascend C 编程规范、Python/PyTorch 扩展开发规范及 Ascend 950PR 算子开发相关要求；
5. 所有交付件须提前完成自验证，确认符合验收标准后再提交；
6. 开发前务必阅读【社区任务】流程及注意事项（https://gitcode.com/org/cann/discussions/39 ）。