# 8月社区任务 - segment_csr 算子开发任务书

## 任务概述

参考 torch_scatter.segment_csr（https://pytorch-scatter.readthedocs.io/en/latest/functions/segment_csr.html ，版本建议 ≥ 2.1.0） 及其子算子（`segment_sum_csr` / `segment_add_csr` / `segment_mean_csr` / `segment_min_csr` / `segment_max_csr`），在昇腾 NPU 上基于 **Ascend C（Kernel）+ Python（PyTorch 适配层）** 实现功能与接口完全对齐的 Segment CSR 系列算子，完成算子设计、开发、测试全流程工作，验收通过后将算子提交至昇腾算子开源仓 **ops-gnn**。

**验收口径**：以 **PyTorch 层接口** 的功能、精度、性能为唯一验收基准；C++ 层（aclnn）接口为可选交付项，不作为验收必要条件。

### 功能定义

Segment CSR 是一种**基于 CSR 压缩行指针（indptr）的分组归约**操作：第 \(i\) 段对应 `src[indptr[i] : indptr[i+1])` 的元素，归约结果写入 `out[i]`。

一维 `reduce="sum"` 时：

$$ 
\text{out}_i = \sum_{j = \text{indptr}[i]}^{\text{indptr}[i+1]-1} \text{src}_j 
$$

**与其他算子对比**：

| 特性 | segment_csr | segment_coo | scatter |
|------|-------------|-------------|---------|
| 索引格式 | CSR `indptr` | 有序 COO `index` | 任意 `index` |
| 确定性 | **完全确定** | GPU 上可能非确定 | 非确定 |
| 性能 | **最快**（torch_scatter 官方） | 次之 | 最通用 |
| dim_size | **无** | 有 | 有 |

### 典型应用场景

图神经网络 CSR 邻接聚合、推荐系统稀疏特征池化、已知行指针的批量段归约。

---

## 核心开发要求及验收标准

开发/测试硬件和软件要求：
- **适配硬件**：Ascend 950PR
- **CANN 版本**：算子开源仓（https://gitcode.com/cann/ops-gnn ）指定版本

### 功能实现要求

#### 1. 与 torch_scatter 核心功能完全对齐

须实现 5 种归约模式（**不含 mul**）：

| reduce | Python API | 语义 |
|--------|------------|------|
| `sum` / `add` | `segment_sum_csr` / `segment_add_csr` | 段内求和 |
| `mean` | `segment_mean_csr` | 段内均值 |
| `min` / `max` | `segment_min_csr` / `segment_max_csr` | 段内极值 + arg_out |

#### 2. indptr 约束（**硬性要求**）

| 约束 | 说明 |
|------|------|
| 形状 | `indptr.dim() <= src.dim()`；归约维 `dim = indptr.dim() - 1` |
| 长度 | `indptr.size(dim) = out.size(dim) + 1` |
| 单调性 | `indptr` 沿最后一维**非降序** |
| 取值范围 | `0 <= indptr[i] <= src.size(dim)`；`indptr[0]=0` 且 `indptr[-1]=src.size(dim)` 为常见合法形态 |
| 空段 | `indptr[i]==indptr[i+1]` 时段长为 0，输出按 reduce 初值规则 |

#### 3. 输出形状推断

未提供 `out` 时：

```
out.size(dim) = indptr.size(dim) - 1
```

**无 `dim_size` 参数**（与 segment_coo / scatter 不同，接口不得添加）。

#### 4. 广播

`indptr` 前 `indptr.dim()-1` 维可广播至 `src`；参考 torch_scatter 官方示例 `indptr.view(1, -1)`。

#### 5. min/max 与 arg_out

与 segment_coo 相同；`segment_csr(..., reduce="min"/"max")` 仅返回 `out`。

#### 6. 支持的数据类型与实现路径分级

| 级别 | src/out dtype | 实现路径 | 性能验收 |
|------|---------------|----------|----------|
| **L1 NPU 原生** | FLOAT16/BFLOAT16/FLOAT32、INT8/16/32、UINT8 | Ascend C Kernel（段内归约，**无全局原子**） | **纳入 NPU 性能基线** |
| **L2 可选路径** | **FLOAT64、INT64** | **CPU 回退** 或 **NPU 低精度拼接** | **不做性能考核** |

**indptr**：固定 **INT64**。

##### float64 / int64 实现路径（L2，可选）

PyTorch 层检测 `src.dtype ∈ {float64, int64}` 时，可 **CPU 回退**（bit-wise 一致）或 **NPU 低精度拼接**（README 说明精度策略）。**不做性能考核**。

#### 7. 接口分层要求

**PyTorch 层接口（须与原版逐字对齐）**：

```python
def segment_sum_csr(src: torch.Tensor, indptr: torch.Tensor,
                    out: Optional[torch.Tensor] = None) -> torch.Tensor: ...

def segment_add_csr(src: torch.Tensor, indptr: torch.Tensor,
                    out: Optional[torch.Tensor] = None) -> torch.Tensor: ...

def segment_mean_csr(src: torch.Tensor, indptr: torch.Tensor,
                     out: Optional[torch.Tensor] = None) -> torch.Tensor: ...

def segment_min_csr(src: torch.Tensor, indptr: torch.Tensor,
                    out: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]: ...

def segment_max_csr(src: torch.Tensor, indptr: torch.Tensor,
                    out: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]: ...

def segment_csr(src: torch.Tensor, indptr: torch.Tensor,
                out: Optional[torch.Tensor] = None,
                reduce: str = "sum") -> torch.Tensor: ...
```

**aclnn 层（可选）**：`aclnnSegmentCsr` / `aclnnSegmentSumCsr` 等。

**Kernel 建议**：每段独立向量归约（参考 CUDA warp reduction）；**须保持完全确定性**。

---

### 参数说明

| 参数名 | 输入/输出 | 描述 | 数据类型 | 非连续 |
|--------|-----------|------|----------|--------|
| src | 输入 | 源张量 | L1/L2 见 §6 | 支持 |
| indptr | 输入 | CSR 行指针，长度 = 段数+1 | INT64 | 支持 |
| out | 输入/输出 | 输出；`size(dim)=indptr.size(dim)-1` | 同 src | 支持 |
| reduce | 属性 | sum/add/mean/min/max | STRING | - |
| arg_out | 输出 | min/max 索引 | INT64 | 支持 |

---

### 算子约束限制

1. **完全确定性**：同输入须 bit-level 一致（浮点 sum 除外可按误差阈值）；
2. **无 dim_size 参数**：接口不得增删；
3. **无 mul 模式**；
4. **float64/int64 L2 路径**：可 CPU 回退或低精度拼接，**不做性能考核**；
5. **仅验收前向**。

---

### 功能验收用例

| 编号 | 场景 | 参考 |
|------|------|------|
| TC-01～TC-06 | 标准数值 | `test/test_segment.py` |
| TC-07 | out 原地 | `test_out` |
| TC-08 | 非连续 | `test_non_contiguous` |
| TC-09 | 空张量 | `test_zero_tensors.py` |
| TC-10 | 空段 indptr | `indptr[i]==indptr[i+1]` |
| TC-11 | 与 segment_coo 一致 | 同一 tests |
| TC-12～13 | float64/int64 L2 路径 | 自行构造 |

---

### 性能要求

- 算子所有用例的性能需大于等于0.6倍标杆性能。
- GPU（A100）的耗时如下：

#### 测试数据

按行指针分段归约，通过 indptr 标记每段起止。数据：随机矩阵（N行 × 128特征），均匀分段为 seg 段。

#### sum

| shape (N × 128, seg段) | float32 | float16 | 总元素 |
|-------|---------|---------|--------|
| N=256K seg=16K | 0.094ms | 0.052ms | 33M |
| N=512K seg=32K | 0.181ms | 0.099ms | 66M |
| N=1024K seg=32K | 0.341ms | 0.182ms | 131M |
| N=2048K seg=64K | 0.675ms | 0.358ms | 262M |
| N=4096K seg=64K | 1.330ms | 0.703ms | 524M |

#### mean

| shape (N × 128, seg段) | float32 | float16 | 总元素 |
|-------|---------|---------|--------|
| N=256K seg=16K | 0.094ms | 0.053ms | 33M |
| N=512K seg=32K | 0.181ms | 0.100ms | 66M |
| N=1024K seg=32K | 0.340ms | 0.183ms | 131M |
| N=2048K seg=64K | 0.674ms | 0.360ms | 262M |
| N=4096K seg=64K | 1.328ms | 0.706ms | 524M |

#### min

| shape (N × 128, seg段) | float32 | float16 | 总元素 |
|-------|---------|---------|--------|
| N=256K seg=16K | 0.127ms | 0.121ms | 33M |
| N=512K seg=32K | 0.240ms | 0.231ms | 66M |
| N=1024K seg=32K | 0.404ms | 0.385ms | 131M |
| N=2048K seg=64K | 0.790ms | 0.753ms | 262M |
| N=4096K seg=64K | 1.478ms | 1.334ms | 524M |

#### max

| shape (N × 128, seg段) | float32 | float16 | 总元素 |
|-------|---------|---------|--------|
| N=256K seg=16K | 0.127ms | 0.121ms | 33M |
| N=512K seg=32K | 0.240ms | 0.231ms | 66M |
| N=1024K seg=32K | 0.404ms | 0.385ms | 131M |
| N=2048K seg=64K | 0.790ms | 0.752ms | 262M |
| N=4096K seg=64K | 1.479ms | 1.334ms | 524M |


---

### 精度要求

满足《生态算子开源精度标准（https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md ）》；仿照已有用例自行编写Python用例进行自测试，使用单标杆精度进行验收。

**真值生成方式**：以 CPU 版 torch_scatter 为标杆；NPU 结果通过 **PyTorch 层接口** 调用获取并与标杆比对。

| 数据类型 / 场景 | 精度策略 |
|-----------------|----------|
| L1 浮点 | 与 torch_scatter CPU 比对；段内顺序归约须 **确定性**（推荐实现），浮点 sum 对标 CPU |
| L1 整数 | **bit-wise 一致** |
| L2 float64/int64 | CPU 回退 bit-wise 一致；低精度拼接 README 文档化精度策略 |
| `min/max` | `out` + `arg_out` 与标杆一致 |


## 验收交付件

在社区任务IT系统中提交验收时， 需要提交以下交付件：

| 序号 | 交付件名称 | 交付件要求 |
|------|-----------|------------|
| 1 | 算子设计文档 | 1. 设计文档模板：https://gitcode.com/cann/cann-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md ，重点描述：**CSR 段并行策略、确定性保证、indptr 校验**；<br> 2. 在cann-competitions 仓库（https://gitcode.com/cann/cann-competitions/tree/master/04_tasks/01_community-task-2026/tasklist ）以PR形式提交设计文档，通过评审后合入仓库，详细说明见：https://gitcode.com/cann/cann-competitions/blob/master/04_tasks/01_community-task-2026/README.md|
| 2 | 自测用例及测试代码 |1. 需要清晰列出精度测试case和性能测试case；<br> 2. 测试代码中的readme文件需要说明测试步骤，保证验收人可以复现测试结果|
| 3 | 自测报告 | 1. 自测报告模板：https://docs.qq.com/sheet/DUmVWWndaUE12WGFB?tab=BB08J2 ；<br> 2. 需要包含用例参数、精度对比结果及截图、性能数据及截图 |
| 4 | 待验收代码地址 | 1. 个人代码仓链接、分支、算子目录；需要在个人仓邀请账号Ascend-CANN作为开发者，如下图所示； <br> 2. 算子目录下需要提供readme文件，参考：https://gitcode.com/cann/ops-transformer/blob/master/attention/chunk_gated_delta_rule/README.md <br> |

![邀请示意](./pics/invite.jpeg)


### PR 申请合入

```
https://gitcode.com/cann/ops-gnn/tree/master/segment_csr
```

---

## 参考资料

1. `torch_scatter/segment_csr.py`、`csrc/cpu/segment_csr_cpu.cpp`、`csrc/cuda/segment_csr_cuda.cu`

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

1. Segment CSR 是 GNN **性能关键路径**，优先投入优化；
2. **确定性**是区别于 scatter 的硬性优势，Kernel 不得引入非确定原子争用；
3. **接口无 dim_size**，严禁擅自扩展参数；
4. **float64/int64 L2 路径**可选 CPU 回退或低精度拼接，不做性能考核；**性能验收取最优实现**（仅 L1）；
5. 开发须严格遵循 Ascend C 编程规范、Python/PyTorch 扩展开发规范及 Ascend 950PR 算子开发相关要求；
6. 所有交付件须提前完成自验证，确认符合验收标准后再提交；
7. 开发前务必阅读【社区任务】流程及注意事项（https://gitcode.com/org/cann/discussions/39 ）。
