# 8月社区任务 - grid_cluster 算子开发任务书

## 任务概述

实现与 `torch_cluster.grid_cluster`（https://github.com/rusty1s/pytorch_cluster ，版本建议≥ 1.6.0） **接口完全一致** 的 NPU 版本：在点云上叠加 **D 维规则网格**，同 voxel 内点共享 cluster ID。

**验收口径**：PyTorch 层 `grid_cluster()` 为准。

### 功能定义

$$
\text{cluster}_i = \sum_d \left\lfloor \frac{pos_{i,d} - start_d}{size_d} \right\rfloor \times \prod_{k < d} \text{num\_voxels}_k
$$

- `start`/`end` 缺省时分别为 `pos.min(0)` / `pos.max(0)`

---

## 核心开发要求及验收标准

开发/测试硬件和软件要求：
- **适配硬件**：Ascend 950PR
- **CANN 版本**：算子开源仓（https://gitcode.com/cann/ops-gnn ）指定版本

### 1. PyTorch 层接口（**必选**）

```python
def grid_cluster(
    pos: torch.Tensor,
    size: torch.Tensor,
    start: Optional[torch.Tensor] = None,
    end: Optional[torch.Tensor] = None,
) -> torch.Tensor: ...
```

底层调用 `torch.ops.torch_cluster.grid(pos, size, start, end)`。

### 2. 参数与约束

| 参数 | 类型 | 约束 |
|------|------|------|
| `pos` | Float `[N, D]` | D 维坐标；L1 float16/32；L2 float64 可选 CPU 回退或低精度拼接 |
| `size` | Float `[D]` | 每维 voxel 边长，`> 0` |
| `start` | Optional Float `[D]` | 网格起点；默认 `pos.min(0)` |
| `end` | Optional Float `[D]` | 网格终点；默认 `pos.max(0)` |

| 输出 | 类型 | 约束 |
|------|------|------|
| cluster | Long `[N]` | 线性化 voxel ID，非负 |

### 3. 数据类型分级

| 级别 | dtype | 路径 | 性能考核 |
|------|-------|------|----------|
| L1 | float16/bfloat16/float32 | NPU | 是 |
| L2 | float64 | CPU 回退 或 NPU 低精度拼接 | 否 |

### 4. 算子约束

1. `size.numel() == pos.size(1)`；
2. 除法与取整语义与 CPU 一致（`true_divide` 后转 long）；
3. 空 `pos` 返回空 LongTensor；
4. 确定性算子，须 bit-wise 一致（整数 cluster ID）；
5. 仅验收前向。

### 5. 功能验收用例

| 编号 | 场景 | 参考 |
|------|------|------|
| TC-01 | 2D 网格 | `test/test_grid.py` |
| TC-02 | 指定 start/end | 同上 |
| TC-03 | 3D 点云 | 自行构造 |
| TC-04 | float64 L2 路径 | 自行构造 |
| TC-05 | 空 pos | 自行构造 |

### 6. 性能要求

Grid 为轻量哈希，预期优于邻域搜索算子。

| 分档 | 场景 | **达标基线（≥ A100）** |
|------|------|------------------------|
| S1 | N=10⁵, D=3 | **0.60×** |
| S2 | N=10⁶, D=3 | **0.55×** |
| S3 | float16 | **0.65×** |

验收取最优实现。

### 7. 精度要求

算子计算精度需严格满足《生态算子开源精度标准（https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md ）》，采用 AscendOpTest（https://gitcode.com/HIT1920/AscendOpTest ） 测试。

**真值生成方式**：以 CPU 版 torch_cluster 为标杆；NPU 结果通过 **PyTorch 层 `grid_cluster()`** 调用获取并与标杆比对。

**单标杆不满足时**：采用 ATK（https://gitcode.com/AscendTest/ATK ） 双标杆比对（`cv_fused_double_benchmark`），以更高精度的 CPU 实现为真值，同时评估同精度 CPU 与 NPU 算子实现相对于该真值的误差；满足条件为 NPU/同精度 CPU 的**最大相对误差比例 ≤ 2**、**平均相对误差比例 ≤ 1.2**、**均方根误差比例 ≤ 1.2**。

| 数据类型 / 场景 | 精度策略 |
|-----------------|----------|
| L1 float16/bfloat16/float32 | cluster ID（int64）与 CPU 标杆 **bit-wise 一致**（确定性算子） |
| L2 float64 | CPU 回退 bit-wise 一致；低精度拼接 README 文档化精度策略 |

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

- PR 路径建议：`ops-gnn/grid` （导出 API 名：**`grid_cluster`**）

## 参考资料

- `torch_cluster/grid.py`、`csrc/cpu/grid_cpu.cpp`、`csrc/cuda/grid_cuda.cu`

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

1. 算术简单，优先保证 **正确性与确定性**；
2. `start`/`end` 可选逻辑须在 Host 层与原版一致；
3. 开发须严格遵循 Ascend C 编程规范、Python/PyTorch 扩展开发规范及 Ascend 950PR 算子开发相关要求；
4. 所有交付件须提前完成自验证，确认符合验收标准后再提交；
5. 开发前务必阅读【社区任务】流程及注意事项（https://gitcode.com/org/cann/discussions/39 ）.
