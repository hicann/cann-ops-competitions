# Gather CSR 算子设计文档

本文档对应 [2026 年 8 月社区任务 gather_csr 算子开发任务书](https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/docs/202608/gather_csr_task_doc.md)，并按照[算子设计文档模板](https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md)编写。

# 需求背景

## 需求来源

Gather CSR 是 PyTorch Geometric 生态中 `torch_scatter.gather_csr`（https://github.com/rusty1s/pytorch_scatter ，版本 ≥ 2.1.0）对应的 NPU 实现缺口。当前 ops-gnn 开源仓（https://gitcode.com/cann/ops-gnn ）已具备 `segment_csr` 系列算子（SegmentSumCsr / SegmentMeanCsr / SegmentMinCsr / SegmentMaxCsr），但缺少其逆操作 `gather_csr`，导致「节点特征广播到 CSR 边列表」「segment_csr 逆变换」「稀疏图消息下发」等典型场景只能回退 CPU 执行。

本需求要求基于 **Ascend C（Kernel）+ Python（PyTorch 适配层）** 在昇腾 NPU（Ascend 950PR）上实现功能与接口完全对齐的 Gather CSR 算子。**验收口径以 PyTorch 层接口的功能、精度、性能为唯一验收基准；aclnn 为可选交付项。**

## 背景介绍

### 功能定义

Gather CSR 是 **segment_csr 的逆操作**：将 CSR 压缩的 `src`（段数 = `indptr.size(dim)-1`）按 `indptr` **扩展**为扁平输出。

对第 `i` 段，令 `row_start = indptr[i]`，`row_end = indptr[i+1]`，输出长度为 `row_end - row_start`，且：

$$
\text{out}[row\_start : row\_end) = \text{src}[i]
$$

（逐 feature 维复制；参考 `gather_csr_cpu.cpp`。）

**与 segment_csr 关系**：

| 算子 | 方向 | src 段维 | out 维 |
|------|------|----------|--------|
| segment_csr | 压缩 | `src.size(dim)=nnz` | `out.size(dim)=段数` |
| **gather_csr** | **扩展** | `src.size(dim)=段数` | `out.size(dim)=nnz` |

### 对标实现现状分析

对标实现位于 torch_scatter 仓库：

| 文件 | 内容 |
|------|------|
| `torch_scatter/segment_csr.py` | `gather_csr` Python 入口，函数签名 `gather_csr(src, indptr, out=None)`，仅 gather 语义，无 reduce / dim_size |
| `csrc/cpu/segment_csr_cpu.cpp` | `gather_csr_cpu` 内核：逐段读取 `src[i]`，按 `row_start..row_end` 写入输出 |

torch_scatter 的 `gather_csr` 约束如下：

| 约束 | 说明 |
|------|------|
| 形状 | `indptr.dim() <= src.dim()`；`dim = indptr.dim()-1` |
| 一致性 | `src.size(dim) == indptr.size(dim) - 1`（或 `src.numel()==0`） |
| 单调性 | `indptr` 最后一维非降序 |
| 范围 | `0 <= indptr[i] <= indptr[-1]`；输出长度 = `indptr[-1]`（flatten 后末元素） |

### 典型应用场景

- 节点特征广播到 CSR 边列表（GNN 消息下发）；
- `segment_csr` 逆变换（压缩 → 解压往返）；
- 稀疏图按段复制特征。

# 需求分析

## 需求描述

使用 Ascend C 编程语言实现 Gather CSR 算子的 NPU 原生 kernel，并在 Python 层提供与 `torch_scatter.gather_csr` **函数名、参数列表、默认值、返回值类型完全一致** 的公开接口，可直接作为 torch_scatter 的 NPU 后端替换使用。输出与 torch_scatter CPU 标杆 **bit-wise 一致**。

## 需求拆解

1. **接口对齐**：PyTorch 层仅 3 参数 `gather_csr(src, indptr, out=None)`，与 torch_scatter 完全一致，不得增删改；
2. **L1 NPU 原生路径**：FLOAT16 / BFLOAT16 / FLOAT32、INT8 / INT16 / INT32 / UINT8 走 Ascend C Kernel（段内 broadcast 复制），**纳入 NPU 性能基线**；
3. **L2 可选路径**：FLOAT64、INT64 走 **CPU 回退** 或 **NPU 低精度拼接**，**不做性能考核**；
4. **indptr 固定 INT64**；
5. **输出形状推断**：未提供 `out` 时 `out.size(dim) = indptr[..., -1]`；
6. **算子泛化**：空张量、非连续 Tensor、batch 广播、空段（输出对应区间长度 0）；
7. **性能要求**：所有用例性能 ≥ 0.6 倍 A100 标杆，且通常应**不慢于 Gather COO**。

## 接口定义

### PyTorch 层（必选）

```python
def gather_csr(src: torch.Tensor, indptr: torch.Tensor,
               out: Optional[torch.Tensor] = None) -> torch.Tensor: ...
```

| 参数 | 输入/输出 | 描述 | 数据类型 | 非连续 |
|------|-----------|------|----------|--------|
| src | 输入 | 段特征，`size(dim)=indptr.size(dim)-1` | L1/L2 | 支持 |
| indptr | 输入 | CSR 指针 | INT64 | 支持 |
| out | 输入/输出 | 扩展后输出 | 同 src | 支持 |

### aclnn 层（可选）

`aclnnGatherCsr(src, indptr, out)`，按 CANN 标准两段式接口封装（`GetWorkspaceSize` + 执行入口）。实现与否不影响验收。

### Kernel 层（必选）

Ascend C 实现段内 broadcast 复制核心逻辑，由 PyTorch 层（必选）直接调用或通过 aclnn 层（可选）间接调用。

## 算子支持型号

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Ascend 950PR | √ |

# 详细设计

## 算子分析

### 数学公式

$$
\forall i \in [0, N_{seg}),\ \text{out}[e][row\_start : row\_end][f] = \text{src}[e][i][f]
$$

其中 `row_start = indptr[i]`，`row_end = indptr[i+1]`，`N_seg = indptr.size(dim)-1`，`e` 遍历 batch（`indptr` 前导维），`f` 遍历 feature 维。

### 支持数据类型

| 级别 | src/out dtype | 实现路径 | 性能验收 |
|------|---------------|----------|----------|
| **L1 NPU 原生** | FLOAT16/BFLOAT16/FLOAT32、INT8/16/32、UINT8 | Ascend C Kernel（段内 broadcast 复制） | **纳入 NPU 性能基线** |
| **L2 可选路径** | **FLOAT64、INT64** | **CPU 回退** 或 **NPU 低精度拼接** | **不做性能考核** |

**indptr**：固定 **INT64**。

### 支持形状

- `indptr.dim() <= src.dim()`，`dim = indptr.dim()-1`；
- `src.size(dim) == indptr.size(dim)-1`（或 `src.numel()==0`）；
- 支持 batch 广播（`indptr` 前导维与 `src` 前导维对齐或为 1）；
- 支持空段（`indptr[i] == indptr[i+1]`，输出区间长度 0）；
- 支持非连续 Tensor（strided）与空张量。

## 算子实现

### 实现方案

总体采用三层架构：

```
┌─────────────────────────────────────────┐
│  PyTorch 层（必选）— Python              │  ← 验收基准，接口与 torch_scatter 完全相同
├─────────────────────────────────────────┤
│  aclnn 层（可选）— C++                   │  ← 建议实现，便于 CANN 生态集成
├─────────────────────────────────────────┤
│  Kernel 层（必选）— Ascend C             │  ← 段内连续写核心实现
└─────────────────────────────────────────┘
```

**核心设计思路：按段并行，段内连续写 out。**

Gather CSR 是纯数据搬运算子（无归约、无原子操作），性能关键在 **GM 写带宽**。将输出按「段」划分：每个段对应输出一个**连续区间** `[row_start, row_end)`（沿 feature 维展开后为连续地址），因此段内可进行大块顺序写，充分利用 MTE3 带宽。src 侧每个段只需读一次 `src[i]` 行（`K` 个 feature），读少写多。

#### Host 侧设计

##### 1. 参数校验（InferShape / InferDataType）

| 校验项 | 规则 | 非法处理 |
|--------|------|----------|
| `indptr.dim()` | `>= 1` 且 `<= src.dim()` | 返回参数错误 |
| 维度一致性 | `src.size(dim) == indptr.size(dim)-1` 或 `src.numel()==0` | 返回参数错误 |
| `indptr` 单调性 | 最后一维非降序（Host 侧抽样校验 + 可选全量校验开关） | 返回参数错误 |
| `indptr` 取值范围 | `0 <= indptr[i] <= indptr[-1]` | 返回参数错误 |
| dtype | src/out 属 L1 或 L2 集合；indptr 仅 INT64 | 返回参数错误 |
| `out`（提供时） | 维度与 `src` 一致，且 `out.size(dim) = indptr[..., -1]` | 返回参数错误 |

输出 shape 推断：

```
out.shape = src.shape
out.size(dim) = indptr[..., -1]   # nnz / 扁平长度（flatten 后末元素）
```

`out` 未提供时按上式分配；提供时校验 `out.size(dim)` 满足上式。

##### 2. 分核策略

沿用 ops-gnn 仓 segment_csr 的分核模型（参考 `segment_sum_csr.cpp`），按 **batch 维（E_1）** 划分核心：

```
E_1  = Π src.size(d)[0, indptrDimNum-1)   # batch 总数
aivNum = min(aivCoreNum, E_1)              # 满核优先，小 shape 动态收敛核数
rowsPerAiv   = ceil(E_1 / aivNum)
rowsLastAiv  = E_1 - (aivNum-1) * rowsPerAiv
```

每核处理 `rowsPerAiv` 个 batch 行；batch 内逐段处理。**小 shape 减少参与核数**，避免调度与同步开销。

##### 3. 数据分块与 UB 优化

- feature 维 `K` 按 `MAX_DEAL_NUM`（如 2048 元素）分 tile，`KloopTime = ceil(K / MAX_DEAL_NUM)`；
- 段内连续写沿 feature 维展开：`outBase = e * nSegments * K + seg * K` 处为段起点，段内写入区间 `[seg*K, (seg+1)*K)` 是连续地址；
- UB 队列：`inQueueSrc`（src 行缓存）+ `outQueueOut`（输出缓冲），双缓冲重叠 MTE2 与 Vector/MTE3；
- 非 32B 对齐尾块使用 `DataCopyPad`，仅写有效元素；
- **不落中间张量**：src 行搬入 UB 后直接批量复制到输出缓冲再写回 GM，无片上归约中间结果。

##### 4. TilingKey 规划

按 dtype 分流（dtype 决定 `elemPerBlock` 与计算/搬运动作；Gather 无浮点计算，T 模板实例化即可，TilingKey 主要服务 host 侧分支）：

| TilingKey | 含义 |
|-----------|------|
| 0 | FLOAT32 / FLOAT16 / BFLOAT16（浮点路径） |
| 1 | INT8 / INT16 / INT32 / UINT8（整型路径） |
| 2 | 提供 out 的原地覆盖路径（与无 out 路径共用计算，仅 host 分支不同） |

L2（FLOAT64 / INT64）不进入 NPU kernel，在 PyTorch 层直接路由到 CPU 回退或低精度拼接。

##### 5. TilingData 参数

| 字段 | 类型 | 含义 |
|------|------|------|
| srcLength | uint32 | src 元素总数 |
| E_1 | uint32 | batch 总数 |
| nSegments | uint32 | 段数（indptr 最后一维 - 1） |
| M | uint32 | `src.size(dim)`，段特征数 |
| K | uint32 | feature 维长度 |
| indptrPhysicalNum | uint32 | indptr 元素总数 |
| strideIndptr | uint32 | batch 内 indptr 偏移（支持 indptr 前导维广播时为 0） |
| indptrDimNum | uint32 | indptr 维数 |
| aivNum / rowsPerAiv / rowsLastAiv | uint32 | 分核参数 |
| KloopTime / coreDataNum / coreTailDataNum | uint32 | feature 维分块参数 |
| ALIGN_NUM | uint32 | 32B 对齐元素数 |
| hasOptionalOut | uint32 | 是否提供 out |

#### Kernel 侧设计

##### 1. 初始化

- 绑定 `srcGm`、`indptrGm`、`outGm`（及可选 `optionalOutGm`）GlobalTensor；
- 按分核参数计算本核 batch 区间 `[coreRowBeg, coreRowEnd)`；
- 分配 `inQueueSrc`、`outQueueOut` UB 缓冲及 MTE2/V/MTE3 事件。

##### 2. Process 主流程（段内连续写）

```
for e in [coreRowBeg, coreRowEnd):
    for seg in [0, nSegments):
        row_start = indptrGm.GetValue(e*strideIndptr + seg)
        row_end   = indptrGm.GetValue(e*strideIndptr + seg + 1)
        if row_start >= row_end:   # 空段：输出区间长度 0，跳过
            continue
        # 读一次 src 行，段内连续写
        for f_tile in [0, KloopTime):
            DataCopy(srcLocal, srcGm[e*M*K + seg*K + f_tile*MAX_DEAL_NUM], tileLen)
            DataCopy(outLocal, srcLocal, tileLen)          # broadcast 复制，无计算
            ComputeDataCopy<T>(outGm[e*nSegments*K + row_start*K + f_tile*MAX_DEAL_NUM],
                               outLocal, tileLen)          # 连续区间顺序写
```

##### 3. 空段处理

`row_start == row_end` 时输出区间长度 0，不触发任何读写（无输出写）。

##### 4. 提供 out 的原地语义

提供 `out` 时语义为 **覆盖写**（`out[row_start:row_end) = src[i]`），与 torch_scatter 一致（非累加）。kernel 路径与无 out 相同，仅 host 侧校验 out 形状、复用已有 GM 地址。

##### 5. 与 segment_csr 的对偶性设计

| 维度 | segment_csr（已有） | gather_csr（本设计） |
|------|---------------------|----------------------|
| 方向 | 压缩（多→一） | 扩展（一→多） |
| src 段维长度 | `nnz` | `段数` |
| out 段维长度 | `段数` | `nnz = indptr[-1]` |
| 核心操作 | 段内 ReduceSum/Mean/Min/Max | 段内 broadcast 复制 |
| 数据依赖 | 每段读 `[row_start, row_end)` 段数据 | 每段读 1 个 `src[i]` 行 |

- 复用 ops-gnn `segment_csr` 的 host 分核框架（E_1 / K / aivNum 模型）与 tiling 结构，降低开发与评审成本；
- kernel 地址计算与其镜像（`srcBase = e*M*K + j*K` ↔ `outBase = e*nSegments*K + row*K`），便于代码审查与对偶验证；
- 提供可选的自验证增强用例：`segment_csr(gather_csr(x))` 往返等价性检查。

##### 6. indptr 校验落地

- Host 侧：形状/单调性/取值范围静态校验（`src.size(dim) == indptr.size(dim)-1` 为**硬约束**）；
- 单调性全量校验成本为 `O(indptr.numel())`，为单次 kernel 启动额外开销；设计提供「默认开启全量校验、大 indptr 时可关闭（开发者自验证时评估）」的开关，验收默认开启。

#### L2 路径设计（FLOAT64 / INT64）

| 路径 | 说明 | 精度 |
|------|------|------|
| **CPU 回退**（首选） | PyTorch 层透明回退 CPU 完成计算（`src`/`indptr`/`out` 经 D2H/H2D），直接调用 torch_scatter 等价逻辑或自实现 CPU 内核 | 与 torch_scatter CPU **bit-wise 一致** |
| **NPU 低精度拼接** | 备选：float64 拆高位/低位 float32 或 int64 拆 32 位高低半，在 NPU 上分段计算后拼接还原 | README 文档化精度策略 |

选择 **CPU 回退为首选路径**：Gather 为纯复制操作，无浮点累加，回退 CPU 结果天然 bit-wise 一致，且该路径不做性能考核；低精度拼接复杂度和出错风险更高，仅作为可选的工程探索。首次触发 L2 路径时打印 `warning`（含 dtype 与实现路径）。

#### PyTorch 层设计

| 要求项 | 说明 |
|--------|------|
| 接口一致性 | 函数签名 `gather_csr(src, indptr, out=None)` 与 torch_scatter 逐字对齐 |
| L1 分发 | 走 NPU Kernel（通过 aclnn 或直接 kernel launch） |
| L2 分发 | 按 `src.dtype ∈ {float64, int64}` 路由 CPU 回退/低精度拼接，与 device 无关 |
| 非连续/广播 | 非连续 src/indptr 由 torch 层 `contiguous()` 或 strided 寻址处理；batch 广播在 host 侧解析 |
| 自动求导 | 验收仅前向，backward 作为后续扩展 |

#### aclnn 层设计（可选）

按 CANN 标准封装：

```c
aclnnStatus aclnnGatherCsrGetWorkspaceSize(const aclTensor *src, const aclTensor *indptr,
                                           aclTensor *out, uint64_t *workspaceSize,
                                           aclOpExecutor **executor);
aclnnStatus aclnnGatherCsr(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                           aclrtStream stream);
```

内部调用 GatherCsr 算子，保证与 PyTorch 层前向结果一致。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Ascend 950PR | √ |

## 算子约束限制

1. **indptr 单调**（最后一维非降序）且与 `src` 维数匹配（`src.size(dim) == indptr.size(dim)-1` 为硬约束）；
2. **无 reduce / dim_size**，仅 gather 语义；
3. **float64/int64 L2 路径**：可 CPU 回退或低精度拼接，**不做性能考核**；
4. **完全确定性**：纯复制操作，同输入输出确定，无浮点累加顺序问题；
5. **仅验收前向**，backward 不在本任务范围；
6. 不启动 kernel 的空张量（`src.numel()==0`）直接返回空输出。

# 可维可测分析

## 功能测试设计

验收测试须通过 **PyTorch 层接口** 执行（pytest 直接调用 `gather_csr`），自验证报告完整、可复现。

| 编号 | 场景 | 参考 | 说明 |
|------|------|------|------|
| TC-01～TC-06 | 标准数值 | `test/test_gather.py` tests[0]～[5] | 覆盖不同维数/段数/feature 组合 |
| TC-07 | 与 gather_coo 一致 | 同 tests | CSR/COO 等价索引下数值一致 |
| TC-08 | out 原地 | `test_out` | 提供 out 时覆盖写 |
| TC-09 | 非连续 | `test_non_contiguous` | strided src/indptr |
| TC-10 | 空张量 | `test_zero_tensors.py` | `src.numel()==0` |
| TC-11 | 空段 | `indptr[i]==indptr[i+1]` | 输出对应区间长度 0 |
| TC-12 | float64 L2 路径 | 自行构造 | 功能/精度通过，不做性能考核 |
| TC-13 | int64 L2 路径 | 自行构造 | 功能/精度通过，不做性能考核 |

## 精度标准

真值生成方式：以 CPU 版 torch_scatter 为标杆；NPU 结果通过 **PyTorch 层接口** 调用获取并与标杆比对，采用 AscendOpTest（https://gitcode.com/HIT1920/AscendOpTest ）测试。

单标杆不满足时采用 ATK（https://gitcode.com/AscendTest/ATK ）双标杆比对（`cv_fused_double_benchmark`），以更高精度的 CPU 实现为真值；满足条件为 NPU/同精度 CPU 的**最大相对误差比例 ≤ 2**、**平均相对误差比例 ≤ 1.2**、**均方根误差比例 ≤ 1.2**。

| 数据类型 / 场景 | 精度策略 |
|-----------------|----------|
| L1 全 dtype | 段内 broadcast 复制，无浮点累加；与 torch_scatter CPU **bit-wise 一致** |
| L2 float64/int64 | CPU 回退 bit-wise 一致；低精度拼接 README 文档化精度策略 |

## 性能标准

- 算子所有用例性能 ≥ 0.6 倍标杆性能；**Gather CSR 通常应不慢于 Gather COO**；
- GPU（A100）标杆耗时（随机源矩阵 seg段 × 128特征 + 偏移指针）：

| shape (输出 N × 128, seg段) | float32 | float16 | 输出元素 |
|-------|---------|---------|----------|
| 256K × 128, 512 | 0.100ms | 0.060ms | 33.6M |
| 512K × 128, 1024 | 0.176ms | 0.096ms | 67M |
| 1024K × 128, 1024 | 0.330ms | 0.174ms | 134M |
| 1024K × 128, 2048 | 0.333ms | 0.175ms | 134M |
| 2048K × 128, 4096 | 0.638ms | 0.326ms | 268M |
| 4096K × 128, 4096 | 1.246ms | 0.627ms | 537M |

性能优化关注点：

1. **段内连续写**：段内输出区间连续，MTE3 大块顺序写，避免离散写；
2. **src 行只读一次**：每段仅一次 `src[i]` 行读取，减少 GM 读流量；
3. **双缓冲**：MTE2（src 行读）与 MTE3（out 段写）重叠；
4. **小 shape 收敛核数**：按工作量动态减少 aivNum，避免调度开销；
5. **带宽分析**：以输出元素总量估算理论带宽上限，实测 ≥ 0.6 倍 A100 标杆即为达标基线，另对比 Gather COO 实测数据。

## 兼容性分析

新算子，不涉及历史兼容性迁移。与 ops-gnn 仓既有 `segment_csr` 系列共享 host 分核框架与 tiling 结构，PR 合入路径建议：

```
https://gitcode.com/cann/ops-gnn/tree/master/gather_csr
```

## 风险分析

| 风险 | 应对方案 |
|------|----------|
| indptr 非法值（非单调/越界） | Host 侧全量校验 + 参数错误返回；验收用例覆盖 |
| 大 batch 下分核不均 | E_1 大于核数时满核均分；E_1 小于核数时收敛核数 |
| 非 32B 对齐段写 | DataCopyPad 尾块处理，仅写有效元素 |
| 提供 out 形状不一致 | Host 校验 `out.size(dim) == indptr[..., -1]` |
| L2 低精度拼接精度损失 | 首选 CPU 回退保证 bit-wise；低精度拼接须 README 文档化精度策略 |

## 关联的 Issue

暂无。

## 文档更新

本文档。

## 类型标签

- [ ] Bug 修复
- [ ] 新特性
- [ ] 性能优化
- [ ] 文档更新
- [x] 其他：社区任务算子设计文档
