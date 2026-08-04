# 8月社区任务 - gather_csr 算子开发任务书

## 任务概述

参考 torch_scatter.gather_csr（https://github.com/rusty1s/pytorch_scatter ，版本建议 ≥ 2.1.0） 在昇腾 NPU 上基于 **Ascend C（Kernel）+ Python（PyTorch 适配层）** 实现功能与接口完全对齐的 Gather CSR 算子。Gather CSR 是 **segment_csr 的逆操作**：将 CSR 压缩的 `src`（段数 = `indptr.size-1`）按 `indptr` **扩展**为扁平输出。

**验收口径**：以 **PyTorch 层接口** 的功能、精度、性能为唯一验收基准；aclnn 为可选。

### 功能定义

对第 \(i\) 段，令 `row_start = indptr[i]`，`row_end = indptr[i+1]`，输出长度为 `row_end - row_start`，且：

$$
\text{out}[row\_start : row\_end) = \text{src}[i]
$$

（逐 feature 维复制；见 `gather_csr_cpu.cpp`。）

**与 segment_csr 关系**：

| 算子 | 方向 | src 段维 | out 维 |
|------|------|----------|--------|
| segment_csr | 压缩 | `src.size(dim)=nnz` | `out.size(dim)=段数` |
| **gather_csr** | **扩展** | `src.size(dim)=段数` | `out.size(dim)=nnz` |

### 典型应用场景

节点特征广播到 CSR 边列表、segment_csr 逆变换、稀疏图消息下发。

---

## 核心开发要求及验收标准

开发/测试硬件和软件要求：
- **适配硬件**：Ascend 950PR
- **CANN 版本**：算子开源仓（https://gitcode.com/cann/ops-gnn ）指定版本

### 功能实现要求

#### 1. 与 torch_scatter 完全对齐

- 仅 gather 语义，**无 reduce / dim_size**
- 与 `gather_coo` 在相同 `test_gather.py` 用例上 **数值一致**（CSR/COO 等价索引）

#### 2. indptr 约束

| 约束 | 说明 |
|------|------|
| 形状 | `indptr.dim() <= src.dim()`；`dim = indptr.dim()-1` |
| 一致性 | `src.size(dim) == indptr.size(dim) - 1`（或 `src.numel()==0`） |
| 单调性 | `indptr` 最后一维非降序 |
| 范围 | `0 <= indptr[i] <= indptr[-1]`；输出长度 = `indptr[-1]`（flatten 后末元素） |

#### 3. 输出形状

未提供 `out` 时：

```
out.size(dim) = indptr[..., -1]   # 即 nnz / 扁平长度
```

提供 `out` 时：维度与 `src` 一致，且 `out.size(dim)` 满足上式。

#### 4. 支持的数据类型与实现路径分级

| 级别 | src/out dtype | 实现路径 | 性能验收 |
|------|---------------|----------|----------|
| **L1 NPU 原生** | FLOAT16/BFLOAT16/FLOAT32、INT8/16/32、UINT8 | Ascend C Kernel（段内 broadcast 复制） | **纳入 NPU 性能基线** |
| **L2 可选路径** | **FLOAT64、INT64** | **CPU 回退** 或 **NPU 低精度拼接** | **不做性能考核** |

**indptr**：固定 **INT64**。

#### 5. 接口分层要求

**PyTorch 层（必选）**：

```python
def gather_csr(src: torch.Tensor, indptr: torch.Tensor,
               out: Optional[torch.Tensor] = None) -> torch.Tensor: ...
```

**aclnn 层（可选）**：`aclnnGatherCsr`。

**Kernel 建议**：按段并行，段内连续写 out（高带宽顺序写）；参考 CUDA/CPU 每段复制 `src[i]` 到 `[row_start, row_end)`。

#### 6. 算子泛化

空张量、非连续、batch 广播、空段（输出对应区间长度 0）

---

### 参数说明

| 参数名 | 输入/输出 | 描述 | 数据类型 | 非连续 |
|--------|-----------|------|----------|--------|
| src | 输入 | 段特征，`size(dim)=indptr.size(dim)-1` | L1/L2 | 支持 |
| indptr | 输入 | CSR 指针 | INT64 | 支持 |
| out | 输入/输出 | 扩展后输出 | 同 src | 支持 |

---

### 算子约束限制

1. **indptr 单调**且与 `src` 维数匹配；
2. **无 reduce / dim_size**；
3. **float64/int64 L2 路径**：可 CPU 回退或低精度拼接，**不做性能考核**；
4. **完全确定性**；
5. **仅验收前向**。

---

### 功能验收用例

| 编号 | 场景 | 参考 |
|------|------|------|
| TC-01～TC-06 | 标准数值 | `test/test_gather.py` |
| TC-07 | 与 gather_coo 一致 | 同 tests |
| TC-08 | out 原地 | `test_out` |
| TC-09 | 非连续 | `test_non_contiguous` |
| TC-10 | 空张量 | `test_zero_tensors.py` |
| TC-11 | 空段 | `indptr[i]==indptr[i+1]` |
| TC-12～13 | float64/int64 L2 路径 | 自行构造 |

---

### 性能要求

#### 性能基线说明

- 算子所有用例的性能需大于等于0.6倍标杆性能。
- GPU（A100）的耗时如下：

用分段偏移指针抓取矩阵。数据：随机源矩阵（seg段 × 128特征）+ 偏移指针。

| shape (输出 N × 128, seg段) | float32 | float16 | 输出元素 |
|-------|---------|---------|----------|
| 256K × 128, 512 | 0.100ms | 0.060ms | 33.6M |
| 512K × 128, 1024 | 0.176ms | 0.096ms | 67M |
| 1024K × 128, 1024 | 0.330ms | 0.174ms | 134M |
| 1024K × 128, 2048 | 0.333ms | 0.175ms | 134M |
| 2048K × 128, 4096 | 0.638ms | 0.326ms | 268M |
| 4096K × 128, 4096 | 1.246ms | 0.627ms | 537M |

---

### 精度要求

满足《生态算子开源精度标准（https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md ）》；仿照已有用例自行编写Python用例进行自测试，使用单标杆精度进行验收。

**真值生成方式**：以 CPU 版 torch_scatter 为标杆；NPU 结果通过 **PyTorch 层接口** 调用获取并与标杆比对。

| 数据类型 / 场景 | 精度策略 |
|-----------------|----------|
| L1 全 dtype | 段内 broadcast 复制，无浮点累加；与 torch_scatter CPU **bit-wise 一致** |
| L2 float64/int64 | CPU 回退 bit-wise 一致；低精度拼接 README 文档化精度策略 |

---

### 文档规范要求

同 Scatter 任务书；重点：**段内连续写、与 segment_csr 对偶、indptr 校验**；自验证报告须含 AscendOpTest/ATK 执行日志。

---

## 验收交付件

在社区任务IT系统中提交验收时， 需要提交以下交付件：

| 序号 | 交付件名称 | 交付件要求 |
|------|-----------|------------|
| 1 | 算子设计文档 | 1. 设计文档模板：https://gitcode.com/cann/cann-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md ,重点：**段内连续写、与 segment_csr 对偶、indptr 校验**；<br> 2. 在cann-competitions 仓库（https://gitcode.com/cann/cann-competitions/tree/master/04_tasks/01_community-task-2026/tasklist ）以PR形式提交设计文档，通过评审后合入仓库，详细说明见：https://gitcode.com/cann/cann-competitions/blob/master/04_tasks/01_community-task-2026/README.md|
| 2 | 自测用例及测试代码 |1. 需要清晰列出精度测试case和性能测试case；<br> 2. 测试代码中的readme文件需要说明测试步骤，保证验收人可以复现测试结果|
| 3 | 自测报告 | 1. 自测报告模板：https://docs.qq.com/sheet/DUmVWWndaUE12WGFB?tab=BB08J2 ；<br> 2. 需要包含用例参数、精度对比结果及截图、性能数据及截图 |
| 4 | 待验收代码地址 | 1. 个人代码仓链接、分支、算子目录；需要在个人仓邀请账号Ascend-CANN作为开发者，如下图所示； <br> 2. 算子目录下需要提供readme文件，参考：https://gitcode.com/cann/ops-transformer/blob/master/attention/chunk_gated_delta_rule/README.md <br> |

![邀请示意](./pics/invite.jpeg)


**PR 合入路径（建议）**：

```
https://gitcode.com/cann/ops-gnn/tree/master/gather_csr
```

---

## 参考资料

1. `torch_scatter/segment_csr.py`（`gather_csr`）、`csrc/cpu/segment_csr_cpu.cpp`（`gather_csr_cpu`）

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

1. **`src.size(dim) == indptr.size(dim)-1`** 是常见硬约束，Host 须校验；
2. **接口仅 3 参数**，与 torch_scatter 完全一致；
3. **float64/int64 L2 路径**：可 CPU 回退或低精度拼接，**不做性能考核**；
4. **性能验收取最优实现**；Gather CSR 通常应 **不慢于 Gather COO**。
