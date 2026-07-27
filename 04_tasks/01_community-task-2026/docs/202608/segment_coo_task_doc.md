# 8月社区任务 - segment_coo 算子开发任务书

## 任务概述

参考 torch_scatter.segment_coo（https://pytorch-scatter.readthedocs.io/en/latest/functions/segment_coo.html ，版本建议 ≥ 2.1.0） 及其子算子（`segment_sum_coo` / `segment_add_coo` / `segment_mean_coo` / `segment_min_coo` / `segment_max_coo`），在昇腾 NPU 上基于 **Ascend C（Kernel）+ Python（PyTorch 适配层）** 实现功能与接口完全对齐的 Segment COO 系列算子，完成算子设计、开发、测试全流程工作，验收通过后将算子提交至昇腾算子开源仓 **ops-gnn**。

**验收口径**：以 **PyTorch 层接口** 的功能、精度、性能为唯一验收基准；C++ 层（aclnn）接口为可选交付项，不作为验收必要条件。

### 功能定义

Segment COO 是一种**基于 COO 格式排序索引的分组归约**操作：沿 `index` 的**最后一维**，将 `src` 中属于同一分组索引的元素归约写入 `out`。

设 `src` 为 $n$ 维、`index` 为 $m$ 维（$m \leq n$），归约维为 `dim = index.dim() - 1`，则 `out` 形状与 `src` 相同，仅 `out.size(dim)` 为分组数 $y$。

一维 `reduce="sum"` 时：

$$ 
\text{out}_i = \sum_{j \,:\, \text{index}_j = i} \text{src}_j 
$$


**与 scatter 的关键区别**：

| 特性 | segment_coo | scatter |
|------|-------------|---------|
| index 顺序 | **须沿 `index.dim()-1` 非降序排列** | 任意顺序 |
| reduce 模式 | sum/add/mean/min/max | 另含 mul |
| 典型性能 | 通常快于 scatter（可利用有序性） | 通用但慢 |

### 典型应用场景

图邻接 COO 格式边聚合、有序分组统计、PyG 稀疏消息传递（index 已排序时优先选用 segment_coo）。

---

## 核心开发要求及验收标准

开发/测试硬件和软件要求：
- **适配硬件**：Ascend 950PR
- **CANN 版本**：算子开源仓（https://gitcode.com/cann/ops-gnn ）指定版本

### 功能实现要求

#### 1. 与 torch_scatter 核心功能完全对齐

须实现以下 5 种归约模式（**不含 mul**，与原版一致）：

| reduce 取值 | 对应 Python API | 语义 | 空桶输出 |
|-------------|-----------------|------|----------|
| `sum` | `segment_coo(..., reduce="sum")` / `segment_sum_coo` | 求和 | `0` |
| `add` | `segment_coo(..., reduce="add")` / `segment_add_coo` | 同 `sum` | `0` |
| `mean` | `segment_coo(..., reduce="mean")` / `segment_mean_coo` | 段内均值 | `0` |
| `min` | `segment_coo(..., reduce="min")` / `segment_min_coo` | 段内最小 | `0` |
| `max` | `segment_coo(..., reduce="max")` / `segment_max_coo` | 段内最大 | `0` |

**各 reduce 初值与更新规则**（自动创建 `out` 时）与 torch_scatter 一致：sum/add→0，mean→先 sum 再除，min→max 初值后空桶置 0。

#### 2. index 排序约束（**硬性要求**）

| 约束 | 说明 |
|------|------|
| 排序维 | `index` 沿 **`index.dim() - 1`** 须**非降序**（`index[..., i] <= index[..., i+1]`） |
| 取值范围 | `index` 值 ∈ `[0, out.size(dim)-1]` |
| 违反处理 | 建议 Host 侧检测并返回错误；不检测时行为未定义（与 torch_scatter 一致） |

#### 3. 广播与形状推断

- `index.dim() <= src.dim()`；`index` 前 `index.dim()-1` 维广播对齐 `src`
- 归约维 `dim = index.dim() - 1`
- 未提供 `out` 时：

```
out.size(dim) =
  dim_size                        # 若指定
  0                               # index 为空
  index[..., -1].max() + 1        # 否则（取最后一列/行 max）
```

#### 4. 提供 out 时的原地语义

| reduce | 行为 |
|--------|------|
| sum/add | 在现有 out 上**不**额外累加（segment 路径为段内写回；提供 out 时按 torch_scatter `test_out` 语义验证） |
| mean/min/max | 与 torch_scatter 官方测试一致 |

#### 5. min/max 额外输出 arg_out

`segment_min_coo` / `segment_max_coo` 返回 `(out, arg_out)`；`segment_coo(..., reduce="min"/"max")` 仅返回 `out`。

| 属性 | 约束 |
|------|------|
| dtype | INT64 |
| shape | 与 out 相同 |
| 含义 | 极值在 `src` 沿 `dim` 维的下标 |

#### 6. 支持的数据类型与实现路径分级

##### 6.1 数据类型分级总表

| 级别 | src/out dtype | 实现路径 | 支持 reduce | 性能验收 |
|------|---------------|----------|-------------|----------|
| **L1 NPU 原生（必选）** | FLOAT16、BFLOAT16、FLOAT32 | Ascend C Kernel | sum/add/mean/min/max | **纳入 NPU 性能基线** |
| **L1 NPU 原生（必选）** | INT8、INT16、INT32、UINT8 | Ascend C Kernel | sum/add/mean/min/max | **纳入 NPU 性能基线** |
| **L2 可选路径** | **FLOAT64** | **CPU 回退** 或 **NPU 低精度拼接** | 全部 | **不做性能考核** |
| **L2 可选路径** | **INT64** | **CPU 回退** 或 **NPU 低精度拼接** | 全部 | **不做性能考核** |

**index / indptr 相关索引张量**：固定 **INT64**。

**不支持**：BOOL、COMPLEX、Float8 等。

##### 6.2 float64 / int64 实现路径（L2，可选）

当 `src.dtype ∈ {float64, int64}` 时，PyTorch 层须保证接口兼容，可采用 **CPU 回退**（结果与 torch_scatter CPU **bit-wise 一致**）或 **NPU 低精度拼接**（README 说明精度策略，功能验收通过即可）。**不做性能考核**。首次触发 L2 路径建议打印 `warning`。

#### 7. 算子泛化

- `src` 维度 1～8；`index.dim() <= src.dim()`
- 空张量、非连续 Tensor、多 batch 广播
- 段长度 0（空段）输出对应 reduce 初值规则

#### 8. 接口分层要求

```
PyTorch 层（必选，验收基准）→ aclnn 层（可选）→ Ascend C Kernel（必选）
```

**PyTorch 层接口（须与原版逐字对齐）**：

```python
def segment_sum_coo(src: torch.Tensor, index: torch.Tensor,
                    out: Optional[torch.Tensor] = None,
                    dim_size: Optional[int] = None) -> torch.Tensor: ...

def segment_add_coo(src: torch.Tensor, index: torch.Tensor,
                    out: Optional[torch.Tensor] = None,
                    dim_size: Optional[int] = None) -> torch.Tensor: ...

def segment_mean_coo(src: torch.Tensor, index: torch.Tensor,
                     out: Optional[torch.Tensor] = None,
                     dim_size: Optional[int] = None) -> torch.Tensor: ...

def segment_min_coo(src: torch.Tensor, index: torch.Tensor,
                    out: Optional[torch.Tensor] = None,
                    dim_size: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]: ...

def segment_max_coo(src: torch.Tensor, index: torch.Tensor,
                    out: Optional[torch.Tensor] = None,
                    dim_size: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]: ...

def segment_coo(src: torch.Tensor, index: torch.Tensor,
                out: Optional[torch.Tensor] = None,
                dim_size: Optional[int] = None,
                reduce: str = "sum") -> torch.Tensor: ...
```

**aclnn 层（可选）**：建议提供 `aclnnSegmentCoo` / `aclnnSegmentSumCoo` 等，参数含 `src`、`index`、`out`、`dimSize`、`reduce`。

**Kernel 实现建议**：优先采用**有序段扫描**（参考 `segment_coo_cpu.cpp`），避免全局原子；GPU 版 warp 归约 + 段末原子写可改为 NPU 段内向量归约 + 单次写回。

---

### 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 非连续 Tensor |
|--------|----------------|------|----------|---------------|
| src | 输入 | 源张量；L2 dtype 可选 CPU 回退或低精度拼接 | L1 见 §6.1 | 支持 |
| index | 输入 | COO 分组索引，**最后一维有序** | INT64 | 支持 |
| dim_size | 属性 | 输出分组数；未指定则 `index.max()+1` | INT64 标量 | - |
| reduce | 属性 | sum/add/mean/min/max | STRING | - |
| out | 输入/输出 | 可选输出缓冲区 | 同 src | 支持 |
| arg_out | 输出 | min/max 专用索引 | INT64 | 支持 |

---

### 算子约束限制

1. **index 必须有序**：沿 `index.dim()-1` 非降序，否则结果未定义。
2. **无 mul 模式**：`reduce="mul"` 须抛 `ValueError`（与原版一致）。
3. **浮点非确定性**：若 Kernel 使用段末原子写，float sum 允许微小顺序误差；推荐扫描实现以获得确定性。
4. **float64/int64 L2 路径**：可 CPU 回退或 NPU 低精度拼接，**不做性能考核**。
5. **验收仅前向**：backward 为后续扩展。

---

### 功能验收用例（须全部通过）

| 编号 | 场景 | 参考来源 |
|------|------|----------|
| TC-01～TC-06 | 与 segment_csr 共用数值用例 | `test/test_segment.py` tests[0]～[5] |
| TC-07 | 提供 out | `test_segment.py::test_out` |
| TC-08 | 非连续 Tensor | `test_segment.py::test_non_contiguous` |
| TC-09 | 空张量 | `test_zero_tensors.py` |
| TC-10 | index 无序（负向） | 自行构造，须报错或文档声明 UB |
| TC-11 | 全 L1 dtype | `testing.py dtypes` |
| TC-12 | float64 L2 路径 | 自行构造 |
| TC-13 | int64 L2 路径 | 自行构造 |
| TC-14 | 与 segment_csr 结果一致 | 同一 tests 数据 index/indptr 对齐 |

---

### 测试标准

**验收测试须通过 PyTorch 层接口执行**（pytest），自验证报告完整可复现。aclnn 测试为可选项。

---

### 性能要求

#### 性能基线说明

- 以下分档为 **达标基线**；以 **GPU A100** 上 torch_scatter 为 100% 参照；
- **验收取各实现路径中最优方案**合入；
- **float64/int64（L2 路径）不做性能考核**。

Segment COO 利用 index 有序性，**预期优于 scatter**；相对 A100 segment_coo 分档基线：

| 场景分档 | 特征 | reduce | dtype | **达标基线（≥）** |
|----------|------|--------|-------|-------------------|
| S1 GNN COO 典型 | 有序 index，中等段长 | sum | float32 | **0.55×** |
| S2 长段 / 低冲突 | 段内元素多 | sum | float32 | **0.60×** |
| S3 mean | 两次扫描或融合 | mean | float32 | **0.50×** |
| S4 min/max | 含 arg_out | min/max | float32 | **0.50×** |
| S5 float16 | GNN 常规 | sum | float16 | **0.60×** |
| S6 L2 路径 | float64/int64 | 全部 | L2 | 不做性能考核 |

**验收基准用例**：

| 编号 | 场景 | src shape | index | 分档 | 达标基线 |
|------|------|-----------|-------|------|----------|
| P-01 | GNN COO sum | `[E, F]`，E=10⁵ | 有序 COO，长度 E | S1 | ≥ 0.55× |
| P-02 | 多维广播 | `[10, 6, 64]` | `[1,6]` 有序 | S1 | ≥ 0.55× |
| P-03 | mean | `[10⁵, 32]` | 有序 | S3 | ≥ 0.50× |
| P-04 | vs scatter 同数据 | 同 P-01 | 同 | - | **须快于 NPU scatter** |

---

### 精度要求

算子计算精度需严格满足《生态算子开源精度标准（https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md ）》，采用 AscendOpTest（https://gitcode.com/HIT1920/AscendOpTest ） 测试。

**真值生成方式**：以 CPU 版 torch_scatter 为标杆；NPU 结果通过 **PyTorch 层接口** 调用获取并与标杆比对。

**单标杆不满足时**：采用 ATK（https://gitcode.com/AscendTest/ATK ） 双标杆比对（`cv_fused_double_benchmark`），以更高精度的 CPU 实现为真值，同时评估同精度 CPU 与 NPU 算子实现相对于该真值的误差；满足条件为 NPU/同精度 CPU 的**最大相对误差比例 ≤ 2**、**平均相对误差比例 ≤ 1.2**、**均方根误差比例 ≤ 1.2**。

| 数据类型 / 场景 | 精度策略 |
|-----------------|----------|
| L1 浮点（float16/bfloat16/float32） | 与 torch_scatter CPU 比对；`sum/add/mean` 允许非确定性微小误差；推荐段扫描实现以获得确定性 |
| L1 整数（int8/16/32, uint8） | **bit-wise 一致** |
| L2 float64/int64 | CPU 回退 bit-wise 一致；低精度拼接 README 文档化精度策略 |
| `min/max` | `out` + `arg_out` 与标杆一致（平局按后写者优先） |
| 全部 reduce | sum/add/mean/min/max 均须覆盖（无 mul） |


## 验收交付件

在社区任务IT系统中提交验收时， 需要提交以下交付件：

| 序号 | 交付件名称 | 交付件要求 |
|------|-----------|------------|
| 1 | 算子设计文档 | 1. 设计文档模板：https://gitcode.com/cann/cann-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md ，重点描述：**有序段扫描算法、与 scatter 选型差异、index 有序性校验、多路径性能对比**；<br> 2. 在cann-competitions 仓库（https://gitcode.com/cann/cann-competitions/tree/master/04_tasks/01_community-task-2026/tasklist ）以PR形式提交设计文档，通过评审后合入仓库，详细说明见：https://gitcode.com/cann/cann-competitions/blob/master/04_tasks/01_community-task-2026/README.md|
| 2 | 自测用例及测试代码 |1. 需要清晰列出精度测试case和性能测试case；<br> 2. 测试代码中的readme文件需要说明测试步骤，保证验收人可以复现测试结果|
| 3 | 自测报告 | 1. 自测报告模板：https://docs.qq.com/sheet/DUmVWWndaUE12WGFB?tab=BB08J2 ；<br> 2. 需要包含用例参数、精度对比结果及截图、性能数据及截图 |
| 4 | 待验收代码地址 | 1. 个人代码仓链接、分支、算子目录；需要在个人仓邀请账号Ascend-CANN作为开发者，如下图所示； <br> 2. 算子目录下需要提供readme文件，包含含 PyTorch 调用示例、L1/L2 分级、性能分档基线、**精度策略说明** <br> |

![邀请示意](./pics/invite.jpeg)


### PR 申请合入

```
https://gitcode.com/cann/ops-gnn/tree/master/segment_coo
```

---

## 参考资料

1. torch_scatter：`torch_scatter/segment_coo.py`、`csrc/cpu/segment_coo_cpu.cpp`、`csrc/cuda/segment_coo_cuda.cu`
2. Ascend C 算子开发文档（https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/opdevg/Ascendcopdevg/atlas_ascendc_map_10_0002.html ）

---

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

1. **index 有序**是正确性前提，须在 Host 或测试层明确约束；
2. **验收以 PyTorch 层为准**，接口与 torch_scatter 完全相同；
3. **float64/int64 L2 路径**可选 CPU 回退或低精度拼接，不做性能考核；
4. **性能验收取最优实现**，分档指标为达标基线；
5. 同图数据应 **segment_coo 不慢于 scatter**（有序路径优势）；
6. 开发须严格遵循 Ascend C 编程规范、Python/PyTorch 扩展开发规范及 Ascend 950PR 算子开发相关要求；
7. 所有交付件须提前完成自验证，确认符合验收标准后再提交；
8. 开发前务必阅读【社区任务】流程及注意事项（https://gitcode.com/org/cann/discussions/39 ）。
