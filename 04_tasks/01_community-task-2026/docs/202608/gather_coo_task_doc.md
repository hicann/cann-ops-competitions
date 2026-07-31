# 8月社区任务 - gather_coo 算子开发任务书

## 任务概述

参考 torch_scatter.gather_coo（https://github.com/rusty1s/pytorch_scatter ） 在昇腾 NPU 上基于 **Ascend C（Kernel）+ Python（PyTorch 适配层）** 实现功能与接口完全对齐的 Gather COO 算子。Gather COO 是 **segment_coo 的逆操作**：将已分组的 `src` 按有序 COO `index` **广播/扩展**到与 `index` 同形的输出。

**验收口径**：以 **PyTorch 层接口** 的功能、精度、性能为唯一验收基准；aclnn 为可选。

### 功能定义

设 `index` 沿最后一维有序，对每个输出位置 `e`，设 `idx = index[..., e]`，则：


$$ 
\text{out}[..., e, \dots] = \text{src}[..., \text{idx}, \dots] 
$$

同一 `idx` 连续出现时，`out` 中对应段内各位置均复制 **同一** `src[idx]` 行/切片（见 `gather_coo_cpu.cpp` 缓存 `vals` 逻辑）。

**与 segment_coo 关系**：

| 算子 | 方向 | 输入→输出规模 |
|------|------|---------------|
| segment_coo | 压缩（多→一） | 大 index 维 → 小分组维 |
| **gather_coo** | **扩展（一→多）** | 小分组维 → 大 index 维 |

### 典型应用场景

消息广播到边、segment 逆变换、PyG 特征从节点复制到边。

---

## 核心开发要求及验收标准

开发/测试硬件和软件要求：
- **适配硬件**：Ascend 950PR
- **CANN 版本**：算子开源仓（https://gitcode.com/cann/ops-gnn ）指定版本

### 功能实现要求

#### 1. 与 torch_scatter 完全对齐

- 仅 **gather** 语义，**无 reduce 参数**
- 输出 dtype/device 与 `src` 一致
- **无梯度验收**（本任务仅前向；接口仍与原版一致）

#### 2. index 约束

| 约束 | 说明 |
|------|------|
| 排序 | 沿 **`index.dim()-1` 非降序**（与 segment_coo 相同） |
| 取值 | `index[i] ∈ [0, src.size(dim)-1]`，`dim = index.dim()-1` |
| 维度 | `index.dim() <= src.dim()`；前 `index.dim()-1` 维与 `src` 对齐 |

#### 3. 输出形状

未提供 `out` 时：

```
out.shape 与 src 相同，但 out.size(dim) = index.size(dim)
```

即输出在归约维长度等于 `index` 最后一维长度。

提供 `out` 时：`out.size(dim) == index.size(dim)`，其余维与 `src` 一致。

#### 4. 支持的数据类型与实现路径分级

| 级别 | src/out dtype | 实现路径 | 性能验收 |
|------|---------------|----------|----------|
| **L1 NPU 原生** | FLOAT16/BFLOAT16/FLOAT32、INT8/16/32、UINT8 | Ascend C Kernel（段内复制/广播） | **纳入 NPU 性能基线** |
| **L2 可选路径** | **FLOAT64、INT64** | **CPU 回退** 或 **NPU 低精度拼接** | **不做性能考核** |

**index**：固定 **INT64**。

Gather 为**纯数据搬运 + 索引读**，无归约原子；float64/int64 走 **L2 可选路径**（CPU 回退或低精度拼接），**不做性能考核**。

#### 5. 接口分层要求

**PyTorch 层（必选，须逐字对齐）**：

```python
def gather_coo(src: torch.Tensor, index: torch.Tensor,
               out: Optional[torch.Tensor] = None) -> torch.Tensor: ...
```

**aclnn 层（可选）**：`aclnnGatherCoo(src, index, out)`。

**Kernel 建议**：有序扫描 `index`，段内 `idx` 不变时复用已加载的 `src[idx]` 向量（参考 CPU 实现），减少重复 GM 读。

#### 6. 算子泛化

- 1～8 维；空张量；非连续 Tensor；多 batch

---

### 参数说明

| 参数名 | 输入/输出 | 描述 | 数据类型 | 非连续 |
|--------|-----------|------|----------|--------|
| src | 输入 | 分组后的源张量 | L1/L2 | 支持 |
| index | 输入 | 有序 COO 索引 | INT64 | 支持 |
| out | 输入/输出 | 可选；shape 见 §3 | 同 src | 支持 |

---

### 算子约束限制

1. **index 有序**（非降序）；
2. **无 reduce / dim_size 参数**；
3. **float64/int64 L2 路径**：可 CPU 回退或低精度拼接，**不做性能考核**；
4. **确定性**：同输入输出确定（纯复制）；
5. **仅验收前向**。

---

### 功能验收用例

| 编号 | 场景 | 参考 |
|------|------|------|
| TC-01～TC-06 | 标准数值 | `test/test_gather.py` tests[0]～[5] |
| TC-07 | gather_csr 对照 | 同 tests 的 indptr 路径 |
| TC-08 | out 原地 | `test_out` |
| TC-09 | 非连续 | `test_non_contiguous` |
| TC-10 | 空张量 | `test_zero_tensors.py` |
| TC-11 | 全 L1 dtype | `testing.py dtypes` |
| TC-12 | float64 L2 路径 | 自行构造 |
| TC-13 | int64 L2 路径 | 自行构造 |
| TC-14 | 与 segment_coo 互逆 | segment(gather(x)) 等价性（可选增强） |

---

### 性能要求

#### 性能基线说明

- 算子所有用例的性能需大于等于0.6倍标杆性能。
- GPU（A100）的耗时如下：

按索引列表抓取矩阵行。数据：随机源矩阵（N行 × 128特征）+ 随机索引列表。

| shape (源 N × 128, 索引数) | float32 | float16 | 输出元素 |
|------|---------|---------|----------|
| 16384 × 128, 65536 | 0.077ms | 0.067ms | 8.4M |
| 65536 × 128, 65536 | 0.083ms | 0.074ms | 8.4M |
| 65536 × 128, 262144 | 0.315ms | 0.295ms | 33.6M |
| 262144 × 128, 524288 | 0.542ms | 0.516ms | 67M |
| 262144 × 128, 1048576 | 1.079ms | 1.029ms | 134M |

---

### 精度要求

算子计算精度需严格满足《生态算子开源精度标准（https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md ）》，采用 AscendOpTest（https://gitcode.com/HIT1920/AscendOpTest ） 测试。

**真值生成方式**：以 CPU 版 torch_scatter 为标杆；NPU 结果通过 **PyTorch 层接口** 调用获取并与标杆比对。

**单标杆不满足时**：采用 ATK（https://gitcode.com/AscendTest/ATK ） 双标杆比对（`cv_fused_double_benchmark`），以更高精度的 CPU 实现为真值，同时评估同精度 CPU 与 NPU 算子实现相对于该真值的误差；满足条件为 NPU/同精度 CPU 的**最大相对误差比例 ≤ 2**、**平均相对误差比例 ≤ 1.2**、**均方根误差比例 ≤ 1.2**。

| 数据类型 / 场景 | 精度策略 |
|-----------------|----------|
| L1 全 dtype | 纯索引搬运，无浮点累加；与 torch_scatter CPU **bit-wise 一致** |
| L2 float64/int64 | CPU 回退 bit-wise 一致；低精度拼接 README 文档化精度策略 |

---

### 文档规范要求

同 Scatter 任务书；重点：**有序段缓存、与 segment_coo 对偶关系、带宽优化**；自验证报告须含 AscendOpTest/ATK 执行日志。

---

## 验收交付件

在社区任务IT系统中提交验收时， 需要提交以下交付件：

| 序号 | 交付件名称 | 交付件要求 |
|------|-----------|------------|
| 1 | 算子设计文档 | 1. 设计文档模板：https://gitcode.com/cann/cann-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md ,重点：**有序段缓存、与 segment_coo 对偶关系、带宽优化**；<br> 2. 在cann-competitions 仓库（https://gitcode.com/cann/cann-competitions/tree/master/04_tasks/01_community-task-2026/tasklist ）以PR形式提交设计文档，通过评审后合入仓库，详细说明见：https://gitcode.com/cann/cann-competitions/blob/master/04_tasks/01_community-task-2026/README.md|
| 2 | 自测用例及测试代码 |1. 需要清晰列出精度测试case和性能测试case；<br> 2. 测试代码中的readme文件需要说明测试步骤，保证验收人可以复现测试结果|
| 3 | 自测报告 | 1. 自测报告模板：https://docs.qq.com/sheet/DUmVWWndaUE12WGFB?tab=BB08J2 ；<br> 2. 需要包含用例参数、精度对比结果及截图、性能数据及截图 |
| 4 | 待验收代码地址 | 1. 个人代码仓链接、分支、算子目录；需要在个人仓邀请账号Ascend-CANN作为开发者，如下图所示； <br> 2. 算子目录下需要提供readme文件，参考：https://gitcode.com/cann/ops-transformer/blob/master/attention/chunk_gated_delta_rule/README.md <br> |

![邀请示意](./pics/invite.jpeg)


**PR 合入路径（建议）**：

```
https://gitcode.com/cann/ops-gnn/tree/master/gather_coo
```

---

## 参考资料

1. `torch_scatter/segment_coo.py`（`gather_coo`）、`csrc/cpu/segment_coo_cpu.cpp`（`gather_coo_cpu`）

---

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

1. Gather 与 Segment COO **共享 index 有序约束**；
2. **接口仅 3 个参数**（src, index, out），不得扩展；
3. **float64/int64 L2 路径**：可 CPU 回退或低精度拼接，**不做性能考核**；
4. **性能验收取最优实现**。
