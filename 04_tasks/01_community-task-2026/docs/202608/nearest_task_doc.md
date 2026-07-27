# 8月社区任务 - nearest 算子开发任务书

## 任务概述

实现与 `torch_cluster.nearest`（https://github.com/rusty1s/pytorch_cluster ，版本建议≥ 1.6.0） **接口完全一致** 的 NPU 版本：对 `x` 中每个点，分配 **最近** 的 `y` 中 query 点索引（1-NN 聚类）。

**验收口径**：PyTorch 层 `nearest()` 为准。

> **重要**：原版 C++ **无 CPU Kernel**（CPU 走 scipy.cluster.vq）；**NPU 必须实现完整 Kernel**，不可长期依赖 scipy 回退。

### 功能定义

$$
\text{cluster}_i = \arg\min_j \| x_i - y_j \|
$$

batch 模式下按图实例独立计算，且 `batch_x` 与 `batch_y` 的 **非空 batch 集合须一致**。

---

## 核心开发要求及验收标准

开发/测试硬件和软件要求：
- **适配硬件**：Ascend 950PR
- **CANN 版本**：算子开源仓（https://gitcode.com/cann/ops-gnn ）指定版本

### 1. PyTorch 层接口（**必选**）

```python
def nearest(
    x: torch.Tensor,
    y: torch.Tensor,
    batch_x: Optional[torch.Tensor] = None,
    batch_y: Optional[torch.Tensor] = None,
) -> torch.Tensor: ...
```

### 2. 参数与约束

| 参数 | 类型 | 约束 |
|------|------|------|
| `x` | Float `[N, F]` | L1 float16/32；L2 float64 可选 CPU 回退或低精度拼接 |
| `y` | Float `[M, F]` | `F == x.size(1)` |
| `batch_x` | Optional Long `[N]` | **须 sorted**；未 sorted 抛 `ValueError` |
| `batch_y` | Optional Long `[M]` | **须 sorted**；同上 |

| 输出 | 类型 | 约束 |
|------|------|------|
| cluster | Long `[N]` | `cluster[i] ∈ [0, M-1]`（batch 内局部或全局 y 下标，与 torch_cluster 一致） |

### 3. Python 层逻辑（NPU 路径须复现 GPU 分支）

当 `x` 在 NPU 上时（对标原版 `x.is_cuda` 分支）：

1. 由 `batch_x`/`batch_y` 构造 `ptr_x`、`ptr_y`（同 nearest.py）
2. 校验非空 batch 集合一致，否则 `ValueError`
3. 调用 `torch.ops.torch_cluster.nearest(x, y, ptr_x, ptr_y)`

**L2 路径**（float64 或显式 CPU）：可 CPU 回退（scipy 或自研 CPU）或 NPU 低精度拼接；**NPU float32 路径不得长期依赖 scipy**。**不做 L2 性能考核**。

### 4. 数据类型分级

| 级别 | dtype | 路径 |
|------|-------|------|
| L1 | float16/bfloat16/float32 | **NPU Kernel（必选）** |
| L2 | float64 | CPU 回退 或 NPU 低精度拼接（**不做性能考核**） |
| batch | int64 | NPU |

### 5. 算子约束

1. batch sorted 校验须在 Python 层执行（与原版一致）；
2. `batch_x`/`batch_y` 非空实例集合必须匹配；
3. 1D 输入 auto `view(-1, 1)`；
4. 确定性：同输入同输出（平局可文档化 tie-break）；
5. 仅验收前向。

### 6. 功能验收用例

| 编号 | 场景 | 参考 |
|------|------|------|
| TC-01 | 基础 nearest | `test/test_nearest` 或 knn 类测试 |
| TC-02 | batch 对齐 | `nearest.py` 文档示例 |
| TC-03 | batch 不一致 | 须 ValueError |
| TC-04 | batch 未 sorted | 须 ValueError |
| TC-05 | float64 L2 路径 | 自行构造 |
| TC-06 | N=10⁵ 大规模 | 性能用例 |

> 注：仓库无独立 `test_nearest.py`，须依据 `nearest.py`  docstring 与 knn 测试模式自建用例。

### 7. 性能要求

Nearest 为 1-NN，较 knn 简单。

| 分档 | 场景 | **达标基线（≥ A100）** |
|------|------|------------------------|
| S1 | N=M=10⁴, F=32 | **0.55×** |
| S2 | N=10⁵, M=10³, F=64 | **0.50×** |
| S3 | float16 | **0.60×** |

验收取最优实现。

### 8. 精度要求

算子计算精度需严格满足《生态算子开源精度标准（https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md ）》，采用 AscendOpTest（https://gitcode.com/HIT1920/AscendOpTest ） 测试。

**真值生成方式**：以 CPU 版 torch_cluster（或 scipy `vq` 参考路径）为标杆；NPU 结果通过 **PyTorch 层 `nearest()`** 调用获取并与标杆比对。

**单标杆不满足时**：采用 ATK（https://gitcode.com/AscendTest/ATK ） 双标杆比对（`cv_fused_double_benchmark`），以更高精度的 CPU 实现为真值，同时评估同精度 CPU 与 NPU 算子实现相对于该真值的误差；满足条件为 NPU/同精度 CPU 的**最大相对误差比例 ≤ 2**、**平均相对误差比例 ≤ 1.2**、**均方根误差比例 ≤ 1.2**。

| 数据类型 / 场景 | 精度策略 |
|-----------------|----------|
| L1 float16/bfloat16/float32 | cluster 下标（int64）与 CPU 标杆 **bit-wise 一致** |
| L2 float64 | CPU 回退 bit-wise 一致；低精度拼接 README 文档化精度策略 |
| 距离平局 | 须与 CPU 标杆 tie-break 规则一致（文档化） |

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

- PR 路径建议：`ops-gnn/nearest`

## 参考资料

- `torch_cluster/nearest.py`、`csrc/nearest.cpp`、`csrc/cuda/nearest_cuda.cu`

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

1. 原版 **CPU 无 C++ 实现**，NPU 不能以此为借口缺 Kernel；
2. batch 校验逻辑必须在 Python 层与原版一致；
3. float64 L2 路径可选 CPU 回退或低精度拼接，**不做性能考核**；NPU float32 性能须达标；
4. 开发须严格遵循 Ascend C 编程规范、Python/PyTorch 扩展开发规范及 Ascend 950PR 算子开发相关要求；
5. 所有交付件须提前完成自验证，确认符合验收标准后再提交；
6. 开发前务必阅读【社区任务】流程及注意事项（https://gitcode.com/org/cann/discussions/39 ）。
