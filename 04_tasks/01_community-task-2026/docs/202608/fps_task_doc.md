# 8月社区任务 - fps 算子开发任务书

## 任务概述

参考 torch_cluster.fps（https://github.com/rusty1s/pytorch_cluster ，版本建议 ≥ 1.6.0）（Farthest Point Sampling，PointNet++ 采样算法），在昇腾 NPU 上实现 **Ascend C Kernel + PyTorch 适配层**，接口与原版 **完全相同**。

**验收口径**：以 PyTorch 层 `fps()` 的功能、精度、性能为唯一验收基准；aclnn 为可选。

### 功能定义

迭代采样：在每个 batch 内，反复选择距已选点集**最远**的点，直至达到 $\lceil \text{ratio} \times N \rceil$ 个样本。

- **输入**：点特征矩阵 $src \in \mathbb{R}^{N \times F}$
- **输出**：被采样点在 $src$ 中的 **全局下标**（`LongTensor`）

### 典型场景

点云下采样、3D 检测 backbone、PyG 几何深度学习。

---

## 核心开发要求及验收标准

开发/测试硬件和软件要求：
- **适配硬件**：Ascend 950PR
- **CANN 版本**：算子开源仓（https://gitcode.com/cann/ops-gnn ）指定版本

### 1. PyTorch 层接口（**必选，逐字对齐**）

```python
def fps(
    src: torch.Tensor,
    batch: Optional[Tensor] = None,
    ratio: Optional[Union[Tensor, float]] = None,
    random_start: bool = True,
    batch_size: Optional[int] = None,
    ptr: Optional[Union[Tensor, List[int]]] = None,
) -> torch.Tensor: ...
```

含 `@torch.jit._overload` 四种签名（`ratio` 为 float/Tensor，`ptr` 为 Tensor/List），须与 `torch_cluster/fps.py` 一致。

**Python 层逻辑（须复现）**：

| 分支 | 行为 |
|------|------|
| `ratio is None` | 等价 `ratio=0.5` |
| `ptr` 给定 | 走 CSR 边界；支持 `List[int]` / `Tensor` |
| `batch` 给定 | 由 batch 构造 `ptr`，调用底层 `torch.ops.torch_cluster.fps` |
| 均无 | `ptr=[0, N]` 单 batch |

### 2. 参数与约束

| 参数 | 类型 | 约束 |
|------|------|------|
| `src` | Float Tensor `[N, F]` 或可 view 为 `[N, *]` | L1：float16/bfloat16/float32；L2：float64 可选 CPU 回退或低精度拼接 |
| `batch` | Optional Long `[N]` | 与 `src.size(0)` 一致；`batch[i] ∈ [0, B-1]` |
| `ratio` | float / Float Tensor `[B]` | `(0, 1]`；每 batch 采样数 = `ceil(ratio × deg)` |
| `random_start` | bool | True 时首点随机；False 时取段内 index 0 |
| `batch_size` | Optional int | 有 batch 时默认 `max(batch)+1` |
| `ptr` | Long `[B+1]` 或 List | CSR，单调非降，`ptr[0]=0`，`ptr[B]=N` |

| 输出 | 类型 | 约束 |
|------|------|------|
| 返回值 | Long `[K]` | `K = sum(ceil(ratio_b × deg_b))`；值为 **全局** 点索引 |

### 3. 数据类型分级

| 级别 | dtype | 路径 | 性能考核 |
|------|-------|------|----------|
| L1 | float16/bfloat16/float32 | NPU Kernel | 是 |
| L2 | float64 | CPU 回退 或 NPU 低精度拼接 | 否 |
| 索引 | int64（batch/ptr/输出） | NPU | — |

### 4. 算子约束

1. 距离度量：欧氏距离平方（与 `fps_cpu` 一致：`pow(2).sum(1)`）；
2. `src.numel()==0` 时返回空 LongTensor；
3. 单 batch 内 `ratio×N < 1` 时至少采样 1 点（ceil 语义）；
4. **仅验收前向**；
5. float64 **L2 路径**：可 CPU 回退或 NPU 低精度拼接，**不做性能考核**。

### 5. 功能验收用例

测试通过 PyTorch 层 `fps()` 执行；

| 编号 | 场景 | 参考 |
|------|------|------|
| TC-01 | 2D 点云 ratio=0.5 | `test/test_fps.py` |
| TC-02 | batch 多图 | 同上 |
| TC-03 | ptr 路径 | 同上 |
| TC-04 | random_start=False | 同上 |
| TC-05 | per-batch ratio Tensor | 自行构造 |
| TC-06 | float64 L2 路径 | 自行构造 |
| TC-07 | 空输入 | 自行构造 |

### 6. 性能要求

- 分档为 **达标基线**；**验收取最优实现**；float64 L2 路径**不做性能考核**。
- 参照 **torch_cluster GPU（A100）** `fps` 耗时。

| 分档 | 场景 | dtype | **达标基线（≥ A100）** |
|------|------|-------|------------------------|
| S1 | N=10⁴, F=3, ratio=0.5 | float32 | **0.50×** |
| S2 | N=10⁵, F=32, batch=8 | float32 | **0.45×** |
| S3 | float16 点云 | float16 | **0.55×** |
| S4 | L2 路径 | float64 | 不做性能考核 |

### 7. 精度要求

算子计算精度需严格满足《生态算子开源精度标准（https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md ）》，采用 AscendOpTest（https://gitcode.com/HIT1920/AscendOpTest ） 测试。

**真值生成方式**：以 CPU 版 torch_cluster 为标杆；NPU 结果通过 **PyTorch 层 `fps()`** 调用获取并与标杆比对。`random_start=True` 时须 **固定 RNG seed**。

**单标杆不满足时**：采用 ATK（https://gitcode.com/AscendTest/ATK ） 双标杆比对（`cv_fused_double_benchmark`），以更高精度的 CPU 实现为真值，同时评估同精度 CPU 与 NPU 算子实现相对于该真值的误差；满足条件为 NPU/同精度 CPU 的**最大相对误差比例 ≤ 2**、**平均相对误差比例 ≤ 1.2**、**均方根误差比例 ≤ 1.2**。

| 数据类型 / 场景 | 精度策略 |
|-----------------|----------|
| L1 float16/bfloat16/float32 | 输出采样下标（int64）与 CPU 标杆 **bit-wise 一致**（固定 seed；`random_start=False` 必测） |
| L2 float64 | CPU 回退 bit-wise 一致；低精度拼接 README 文档化精度策略 |
| 距离平局 | 须与 torch_cluster CPU 采用相同 tie-break 规则（文档化） |

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

- PR 路径建议：`ops-gnn/fps`；

## 参考资料

- `pytorch_cluster/torch_cluster/fps.py`、`csrc/cpu/fps_cpu.cpp`、`csrc/cuda/fps_cuda.cu`

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

1. FPS 为**迭代**算法，NPU 实现需权衡并行度与迭代依赖；
2. `ratio` 可为 **per-batch Tensor**，Kernel 须支持；
3. 接口不得增删参数。
