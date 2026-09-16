# MaxPool2dWithMaskBackward 算子设计文档（Ascend C）

> 算子：`MaxPool2dWithMaskBackward`（MaxPool2dWithMask 正向算子的反向传播，消费正向 mask 语义 `indices`）
> 接口基准：`aclnnMaxPool2dWithMaskBackward` / `aclnnMaxPool2dWithMaskBackwardGetWorkspaceSize`
> 仓库与落点：`ops-nn` → `experimental/pooling/max_pool2d_with_mask_backward/`（`op_host/`、`op_kernel/`、`op_graph/`）
> 硬件：Atlas 800T A2、Atlas 300V Pro；CANN 9.0.0 或 9.1.0；Ascend C；团队 `zhangfeng1133`
> 任务：2026 社区任务 09-46-MaxPool2dWithMaskBackward（反向）；配套正向文档：09-45-MaxPool2dWithMask

# 需求背景（required）

## 需求来源

社区任务「MaxPool2dWithMaskBackward 算子开发」任务书：参考 `aclnnMaxPool2dWithMaskBackward` 接口说明，用 Ascend C 实现
正向算子 MaxPool2dWithMask 的反向传播，与正向严格配套，完成设计、开发、测试全流程，
验收通过后合入昇腾算子开源仓 `ops-nn` 的 `experimental/pooling`。

## 背景介绍

### MaxPool2dWithMaskBackward 算子实现优化

反向的唯一定位信息来自正向产出的 `indices`（INT8、4 维位掩码），因此实现要点不是重新搜索最大值，
而是**正确解码 mask 并按位回填梯度**，正确性完全依赖正反向对 `indices` 编码口径的一致性。
口径来源：`aclnnMaxPool2dWithMaskBackward` 官方 API 文档（签名 `(gradOutput, self, indices, kernelSize, stride, padding, dilation, ceilMode, gradInput)`，
`indices` 定义为"最大值在求 mask 的 kernel 位置的 bit 值组成的 Tensor"）；实现参考 ops-nn
`pooling/max_pool3d_grad_with_argmax`（含 max_pool2d_with_mask_backward 用例）；性能/精度基线为 CANN `opp/built-in/op_impl/ai_core/tbe/` 下 pooling 实现。

### MaxPool2dWithMaskBackward 算子实现现状分析（TBE 支持能力）

| 参数名 | 输入/输出 | 描述 | 使用说明 | 数据类型 | 数据格式 | 维度(shape) | 非连续Tensor |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gradOutput | 输入 | 反向传播过程中上一步输出的梯度 | 和正向的输出 shape 一致，数据格式和 self 一致 | FLOAT32、FLOAT16、BFLOAT16 | NCHW | 4 | √ |
| self | 输入 | 正向的输入张量（`gradInput` 的 shape 权威来源） | aclnn 官方签名中位于 `gradOutput` 之后；任务书参数表未单列，实现按官方签名要求必传 | FLOAT32、FLOAT16、BFLOAT16 | NCHW | 4 | √ |
| indices | 输入 | 最大值的索引位置组成的 Tensor（mask 语义） | mask 语义为 int32 argmax，每通道独立 | INT8 | NCHW | 4 | √ |
| kernelSize | 输入 | 池化操作中使用的滑动窗口大小 | 长度仅支持 1、2 | INT64 | — | — | — |
| stride | 输入 | 窗口移动的步长 | 长度仅支持 0、1、2。stride 长度为 0 时，stride 数值等于 kernelSize 的值 | INT64 | — | — | — |
| padding | 输入 | 输入数据的填充，表示输入每个维度上的填充量 | 长度仅支持 1、2；补充位置填写"负无穷" | INT64 | — | — | — |
| dilation | 输入 | 控制窗口中元素的步幅 | 长度仅支持 1、2，值仅支持 1 | INT64 | — | — | — |
| ceilMode | 输入 | 计算输出形状时取整的方法 | True 表示向上取整，False 表示向下取整 | BOOL | — | — | — |
| gradInput | 输出 | 反向传播输出的梯度 | shape 和数据格式与 self 保持一致 | BFLOAT16、FLOAT16、FLOAT32 | NCHW | 4 | √ |

> 仅 Atlas 800T A2 支持 BFLOAT16。

### MaxPool2dWithMaskBackward 算子功能分析

按正向记录的 `indices`（mask 语义）把 `gradOutput` 的梯度"回填"到 `gradInput` 对应位置；池化窗口中未产生最大值的位置梯度为 0。
公式：设输出位置 `(n,c,h,w)` 的最大值位置由 mask 解码为 `(n,c,h_s,w_s)`，则 `gradInput(n,c,h_s,w_s) += gradOutput(n,c,h,w)`。
**需要累加的原因**：`stride < kernelSize` 时窗口重叠（如 yolov11 的 SPPF 用 `kernelSize=5, stride=1, padding=2`），
同一输入位置可能是多个输出窗口的最大值位置；`gradOutput` 同正向 `out`、`indices` 同正向 `indices`、`gradInput` 同 `self`。

# 需求分析（required）

## 需求描述

用 Ascend C 在 Atlas 800T A2 与 Atlas 300V Pro 上实现 `MaxPool2dWithMaskBackward`：dtype FLOAT32/FLOAT16/BFLOAT16
（BFLOAT16 仅 A2），format NCHW、rank 4，四类张量支持非连续；经 mask 解码 + 回填实现反向语义
（命中位置累加 `gradOutput`，未命中位置置 0）；提供 ACLNN 两段式接口、Host 校验与 Tiling、Kernel、原型与 shape/dtype 推导；
默认非确定性实现、支持 `aclrtCtxSetSysParamOpt` 开启确定性；性能不低于原 TBE 的 95%；
在 Atlas 300V Pro 上以 yolov11 + DOTAv1 验证 mAP50 误差 ≤ 0.01；交付自测用例、可复现测试 README、自测报告与算子 README。

## 需求拆解

1. 原型与注册：`op_graph` 原型、dtype/format 注册、shape/dtype 推导（`gradInput` 跟随 `self`，`gradOutput` 对齐正向 `out`）；
2. Host：正反向一致性校验、workspace 与清零规划、分核与 UB 切分、确定性开关读取、tilingKey 规划；
3. Kernel：mask 解码、命中回填与重叠累加（UB 内累加 / `SetAtomicAdd`）、未命中位置置 0；
4. 确定性：默认非确定，开关开启后 bit-wise 确定；5. 测试与交付：精度/性能/负向用例、确定性验证、模型级验证与交付材料。

# 详细设计（required）

## 算子分析

### 数学公式

```
gradInput(n,c,h_s,w_s) += gradOutput(n,c,h,w)          # (h,w) 遍历所有输出位置
mask 解码：由 indices 得知 out(n,c,h,w) 的最大值来自窗口偏移 (m,n)，m∈[0,k_h), n∈[0,k_w)
h_s = stride[0]*h + m - padding[0]；w_s = stride[1]*w + n - padding[1]
(h_s,w_s) 落在 [0,H_in)×[0,W_in) 之外（padding 区）→ 丢弃该梯度；其余未命中位置 = 0
```

**shape 推导**（与正向完全一致，保证正反向可配套；`ceilMode=False` 取 floor，`=True` 取 ceil）：

```
H_out = floor|ceil( (H_in + 2*padding[0] - dilation[0]*(kernelSize[0]-1) - 1) / stride[0] ) + 1
W_out = floor|ceil( (W_in + 2*padding[1] - dilation[1]*(kernelSize[1]-1) - 1) / stride[1] ) + 1
gradInput: [N,C,H_in,W_in]；gradOutput: [N,C,H_out,W_out]
indices  : [N, C, k_h*k_w, (ceil(H_out*W_out/16) + 1) * 2 * 16]        # 与正向同公式，INT8
```

**indices 的 mask 语义与解码**（与正向设计逐条对齐，记 `BW = ⌈H_out×W_out/16⌉+1`）：

| 维度/因子 | 取值 | 语义 |
| --- | --- | --- |
| 第 1、2 维 | `N, C` | 掩码与回填按 `(n,c)` 独立 |
| 第 3 维 | `k_h × k_w` | 窗口内相对偏移枚举，下标 `p = m*k_w + n` |
| 第 4 维 | `BW × 32` int8 | 第 `p` 个偏移的位平面，每 16 个输出位置一个分块；块内第 `b` 位（`b = l mod 16`，`l = h*W_out+w`）为 1 表示 `out(h,w)` 的最大值取自偏移 `(m,n)` |
| 尾部 `+1` / 每块 `2×16` | 1 个 32 字节尾块 / 32 字节 | 对齐哨兵块与分块存储颗粒度，保证任意 `H_out×W_out` 可按 32 字节块访问 |

**解码算法**：对输出位置 `l` 取唯一置位的平面 `p`，换算 `(m,n) = (p/k_w, p%k_w)`、`(h,w) = (l/W_out, l%W_out)`，
得 `(h_s,w_s)` 并判断是否落在有效输入域；mask 信息等价于逐输出位置的 int32 argmax。
**正向保证的不变量**：每输出位置恰好 1 位置位（并列取最小索引），故解码唯一、回填不重复计数。

### 支持数据类型与形状

- `gradOutput` / `self` / `gradInput`：FLOAT32、FLOAT16、BFLOAT16（BFLOAT16 仅 Atlas 800T A2），三者 dtype 必须一致；
  `indices`：INT8；属性 `kernelSize/stride/padding/dilation` 为 INT64 数组（长度 1 或 2，长度为 1 时 H/W 同值），`ceilMode` 为 BOOL；
- dtype 命名口径：`FLOAT32` 即正向任务书中的 FLOAT，本设计与配套正向文档视为同一 dtype；
- `self`/`gradInput`：`[N,C,H_in,W_in]`；`gradOutput`：`[N,C,H_out,W_out]`（须等于由 `self`+属性推导的尺寸）；
  `indices`：`[N,C,k_h*k_w,(⌈H_out*W_out/16⌉+1)*2*16]`，允许非连续；均 rank 4；
- 泛化覆盖：`kernelSize ∈ {1,2,3,5}`（含 `(2,3)`）、`stride ∈ {0(缺省),1,2,3}`、`padding ∈ {0,1,2}`、`ceilMode ∈ {False,True}`、
  窗口重叠（`stride < kernelSize`）与非重叠（`stride ≥ kernelSize`）两类、`H_out*W_out` 非 16 倍数、`W_in` 非 32B 对齐、
  `N*C` 小于/大于/等于 AI Core 数、四类张量非连续。

## 算子实现

### 实现方案

#### Host 侧设计

沿用 ops-nn pooling 反向算子结构：`_def.cpp`（注册）、`_infershape.cpp`、`_tiling.cpp`。校验全部在 Host 完成、不触碰 NPU 内存，
任一失败即返回参数错误（aclnn 层 `ACLNN_ERR_PARAM_INVALID`，口径与既有 pooling 反向算子一致）。

| # | 校验项 | 判定条件 | 处置 |
| --- | --- | --- | --- |
| 1 | dtype/rank | 三者 dtype 不一致或不在 {FLOAT32, FLOAT16, BFLOAT16}；300V Pro 上 BFLOAT16；任一 rank ≠ 4 | 参数错误 / 不支持 |
| 2 | `self`/`gradInput` | shape 不一致（N/C/H_in/W_in 任一维） | 参数错误 |
| 3 | `gradOutput` | shape ≠ 由 `self`+属性推导的 `[N,C,H_out,W_out]` | 参数错误 |
| 4 | `indices` | dtype ≠ INT8，或 shape ≠ `[N,C,k_h*k_w,(⌈H_out*W_out/16⌉+1)*2*16]` | 参数错误 |
| 5 | `kernelSize` / `stride` | `kernelSize` 长度 ∉ {1,2} 或元素 ≤ 0；`stride` 长度 ∉ {0,1,2} 或元素 ≤ 0；长度 0 时归一化 `stride[i]=kernelSize[i]` | 参数错误 / 归一化 |
| 6 | `padding` / `dilation` | `padding` 长度 ∉ {1,2} 或 `padding[i] < 0` 或 `> kernelSize[i]`；`dilation` 长度 ∉ {1,2} 或元素 ≠ 1 | 参数错误 / 不支持 |
| 7 | 尺寸与 ceilMode | `H_out ≤ 0` 或 `W_out ≤ 0`；`ceilMode=True` 且 `(H_out-1)*stride[0] ≥ H_in + padding[0]` 或 `(W_out-1)*stride[1] ≥ W_in + padding[1]` | 参数错误 / 不支持 |
| 8 | 输入取值 | 含 `NaN`、`-Inf` | 不在支持域（Host 不读 device 数据，结果未定义） |

`self` 是 `gradInput` shape 的**唯一权威来源**：仅凭 `H_out` 反推 `H_in` 在多解区间 `[(H_out-1)*s-2p+k, H_out*s-2p+k-1]` 内不唯一，
故按官方签名要求 `self` 必传，并以它校验 `gradInput`/`gradOutput`。第 7 条与正向文档同口径，保证可消费正向支持域内全部合法组合。
**shape 推导与清零规划**：`gradInput` 的 shape/dtype/format 跟随 `self`；`H_out/W_out` 由 `self`+属性按正向公式推导以校验
`gradOutput`/`indices`；未命中任何窗口的输入位置必须为 0，故需保证 `gradInput` 全量覆盖写：

| 路径 | 分核所有权 | 清零方式 | workspace | 原子累加 |
| --- | --- | --- | --- | --- |
| 确定路径（开关开启） | 每核独占若干 `(n,c)` 行的输入位置 | UB 内 tile 先 `Duplicate(0)` 再累加，写回即覆盖 | 0 | 不需要 |
| 默认路径（性能优先，非确定） | 按输出块并行，`(n,c)` 行可跨核 | Host 侧对 `gradInput` 触发 `aclrtMemsetAsync` | 0（清零由框架 memset 完成，不额外占用 workspace） | 重叠场景需要 `SetAtomicAdd` |

第二段接口沿调用方 stream 异步执行，除清零外无额外 Host 同步。

##### 分核策略

```
确定路径：rowTask = N*C；blockDim = min(aivCoreNum, rowTask)；每核独占整数个 (n,c) 行
默认路径：outTask = ceil(N*C*H_out / OUT_ROWS_PER_TASK)；blockDim = min(aivCoreNum, outTask)
每核任务量 = ceil(总任务数 / blockDim)，余量分到前若干核（尾核收尾）
```

- 确定路径：核间 `gradInput` 写集互不相交 → 无竞争、无顺序依赖、结果 bit-wise 确定；代价是需重新枚举会命中本核输入位置的输出窗口，mask 位被重复检查；
  若 `N*C < aivCoreNum`，在 `(n,c)` 行内继续按 H_in（必要时再按 W_in）划分子任务并保持写集互斥，仍无需原子累加；
- 默认路径：分核更细、负载更均衡，但同一 `(n,c)` 行可能被多核写，重叠时浮点累加顺序随并行度变化 → 非确定；
- 窗口不重叠（`stride[0] ≥ k_h` 且 `stride[1] ≥ k_w`）时同一输入位置最多被一个窗口命中，两路径均无需原子累加。

##### UB 切分与 Buffer 规划

```
确定路径：TILE_IN 覆盖若干输入行，行数按 UB 预算取最大；TILE_IN_bytes = align32(TILE_IN_rows * W_in * dtypeSize)
默认路径：TILE_OUT = min(剩余输出量, floor(UB_BUDGET / (gradOutBytes + maskBytes)))
UB_BUDGET = GetCoreMemSize(UB) - RESERVE，再按 BUFFER_NUM 折半（double buffer）
maskBytes（每 tile）= k_h*k_w * align32(ceil(TILE_OUT/16)*2)     # 按 32 字节块读取，取有效 2 字节
```

| Local Memory | 类型 | 容量口径 | 用途 |
| --- | --- | --- | --- |
| `gradOutQue` | `TQue<VECIN,2>` | `TILE_OUT * dtypeSize`（对齐后） | `gradOutput` 搬入 |
| `maskQue` | `TQue<VECIN,2>` | `k_h*k_w * align32(ceil(TILE_OUT/16)*2)` | `indices` 位平面搬入（按 32 字节块，取有效字节） |
| `gradInBuf` | `TBuf<VECCALC>` | 确定路径 `TILE_IN * dtypeSize`；默认路径为 tile 的 UB 累加区 | 累加缓冲 |
| `gradInQue` | `TQue<VECOUT,2>` | 同 `gradInBuf`（对齐后） | 确定路径 `gradInput` 回写 |

容量全部由 `GetCoreMemSize` 运行期计算、不硬编码；Atlas 300V Pro 的 UB 小于 A2，其 tile 更小、分核更细属预期。

##### tilingKey 规划

```
tilingKey = dtypeCode*10000 + pathCode*1000 + ceilModeCode*100 + overlapFlag*10 + maskAlignFlag
```

| 位段 | 含义 | 取值 |
| --- | --- | --- |
| `dtypeCode` | dtype | 0: FLOAT16 / 1: FLOAT32 / 2: BFLOAT16 |
| `pathCode` | 累加路径 | 0: 非重叠直接写 / 1: 重叠 + UB 内累加（确定） / 2: 重叠 + `SetAtomicAdd`（非确定默认） / 3: 输入位置独占（确定路径） |
| `ceilModeCode` | 取整方式 | 0: False / 1: True |
| `overlapFlag` | 是否重叠 | `stride[0] < kernelSize[0] \|\| stride[1] < kernelSize[1]` → 1，否则 0 |
| `maskAlignFlag` | 掩码尾块 | 0: `H_out*W_out` 为 16 倍数 / 1: 非 16 倍数 |

`pathCode` 由 Host 综合 `overlapFlag` 与确定性开关（`aclrtCtxSetSysParamOpt` 相关标志，tiling 阶段读取）计算，Kernel 只做分支分发。

#### Kernel 侧设计

入口读取 `MaxPool2dWithMaskBackwardTilingData`，执行 `Init` 与 `Process`（CopyIn/Compute/CopyOut 三段），由 tilingKey 选定路径。

**mask 解码**：按 32 字节块粒度把 `indices` 第 `p` 个位平面的一段搬入 UB，取出块内有效 16 位掩码字；
对 16 位字逐位扫描（或先 `Select`/`And` 提取非零位）得到命中的输出位置线性索引 `l`；
由 `p = m*k_w+n` 与 `l` 换算 `h_s = stride[0]*(l/W_out) + m - padding[0]`、`w_s = stride[1]*(l%W_out) + n - padding[1]`；
`h_s ∉ [0,H_in)` 或 `w_s ∉ [0,W_in)` 时命中 padding 区、丢弃该梯度；`indices` 允许非连续，CopyIn 按 stride 逐段寻址（块内连续），
或 Host 侧先物化为连续 workspace，按实测择优；`gradOutput`/`gradInput` 非连续时同样按各自 stride 逐段搬入/写回。
**越界处理说明**：反向不做窗口取值与最大值比较，因此不涉及正向那份"越界位置用 `-inf` 填充"的搬入处理；
越界只表现为"解码出的输入坐标落在 padding 区"，此时直接丢弃该梯度，`gradInput` 的 padding 区位置保持 0。

**回填与累加（何时用 `SetAtomicAdd`）**：

| 场景 | 判定 | 实现 | 原子累加 |
| --- | --- | --- | --- |
| 窗口不重叠 | `stride[0] ≥ k_h && stride[1] ≥ k_w` | 直接写：命中位置写 `gradOutput`，其余位置由核内零初始化覆盖 | 否 |
| 窗口重叠 + `(n,c)` 行独占（确定路径） | 开关开启 | UB 的 `gradInBuf` 按输出位置升序累加，tile 起始 `Duplicate(0)` 后一次性写回 | 否 |
| 窗口重叠 + 输出块跨核（默认路径） | 开关关闭且需细分核 | `SetAtomicAdd` 累加写回 GM（写入前必须已清零） | 是 |

结论：**原子累加只在"窗口重叠且同一 `(n,c)` 行被多个核写"时必需**，分核保证 `(n,c)` 行独占时重叠场景也不需原子。
使用要点：只对 `gradInput` 的融合写生效、必须与清零配对、原子模式下不可与非原子写混用同一地址区间、写完恢复非原子模式；
FLOAT16/BFLOAT16 必要时在 FLOAT32 域累加，口径与 TBE 对齐。

**非最大值位置置 0**：确定路径/非重叠直接写在核内 tile 先 `Duplicate(0)` 再叠加命中值，写回即最终结果（天然覆盖写 0）；
默认路径 + 原子累加由 Host 侧 `aclrtMemsetAsync` 清零，Kernel 只对命中位置原子加。
**确定性边界**：非确定来源是重叠场景下同一位置多次浮点累加的顺序随并行度/核数变化；默认（开关关闭）走 `pathCode=2`，
以非确定换取更细分核与均衡（与任务书"默认非确定性实现"一致）；开关开启后切到 `pathCode=1/3`，每个 `gradInput` 位置仅由一个核写、
核内累加顺序固定（输出位置升序、tile 升序），结果 bit-wise 确定；两条路径的 0/非 0 位置集合一致，差异仅在浮点末位。

#### 与正向算子 MaxPool2dWithMask 的一致性契约

| # | 契约项 | 正向提供 | 反向依赖 |
| --- | --- | --- | --- |
| 1 | `indices` shape/语义 | `[N,C,k_h*k_w,(⌈H_out*W_out/16⌉+1)*2*16]`，INT8；第 3 维＝偏移 `p=m*k_w+n`，第 4 维＝16 输出位置一组的位平面（等价逐输出位置 int32 argmax） | 按同一公式校验、同一语义解码 |
| 2 | 唯一性 | 每输出位置恰好 1 位置位（并列取最小索引） | 解码唯一、不重复计数 |
| 3 | 越界 | 解码坐标可能落在 padding 区 | 丢弃该梯度 |
| 4 | 属性口径 | `dilation` 仅支持 1；输入不含 `NaN`/`-Inf`；`ceilMode=True` 且 `(H_out-1)*s ≥ H_in + padding` 不支持 | 采用完全相同判定，可消费正向支持域内全部合法组合 |
| 5 | padding 上界 | 正向校验 `padding[i] ≤ kernelSize[i]/2` | 反向按任务书校验 `padding[i] ≤ kernelSize[i]`（正向支持域是其子集），无冲突 |
| 6 | dtype/format | `out` 与 `self` 一致；rank 4；NCHW/ND | `gradOutput` 对齐 `out`、`gradInput` 对齐 `self`；本算子以 NCHW 验收 |

### 测试设计

- **精度用例矩阵**（与 TBE 的 `gradInput` 逐元素比对）：dtype ∈ {FLOAT16, FLOAT32, BFLOAT16(仅 A2)}；
  shape ∈ {`(1,1,4,4)`,`(1,3,16,16)`,`(8,32,64,64)`,`(2,64,224,224)`,`(1,8,1,1)`}；
  `kernelSize` ∈ {`(1,1)`,`(2,2)`,`(3,3)`,`(2,3)`,`(5,5)`}；`stride` ∈ {0,1,2,3}（覆盖重叠与非重叠）；
  `padding` ∈ {0,1,2}（覆盖命中 padding 被丢弃）；`ceilMode` ∈ {False,True}；非连续张量（单独与组合）；
  确定性开关开/关各跑一遍并比对 0/非 0 位置集合一致。
- **负向用例**：dtype 不一致、rank≠4、`self` 与 `gradInput` shape 不一致、`indices` dtype/形状不匹配、`kernelSize` 长度 3 或元素≤0、
  `padding` 超 `kernelSize`、`dilation=2`、`ceilMode=True` 命中第 7 条限制、`H_out≤0` → 断言参数错误且无 device 访问。
- **性能用例**：与 TBE 相同环境、shape、dtype、核数下采集算子级耗时（NPU Event，预热后采样取中位数与 p90），
  分小 shape（<100us）与大 shape 两组；300V Pro 上以 A2(910B3) 耗时的 5 倍为基准对比。**确定性验证**：开关开启时同输入连续运行 N 次、
  变更分核/切分参数后再运行，`gradInput` bit-wise 一致。**复现要求**：测试代码含 README（环境、编译运行命令、用例 CSV 结构、
  golden 生成方式 = 以 TBE 正向产出的 `indices` 作为反向输入以复现正反向链路、结果判读）；自测报告含参数、精度对比与性能数据及截图。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas A2 训练系列产品（Atlas 800T A2） | √ |
| Atlas 推理系列产品（Atlas 300V Pro） | √ |

> BFLOAT16 仅 Atlas 800T A2 支持；FLOAT 在 Atlas 训练系列产品上内部转 FLOAT16 计算。

## 算子约束限制

1. **确定性计算**：`aclnnMaxPool2dWithMaskBackward` 默认**非确定性实现**，支持通过 `aclrtCtxSetSysParamOpt` 开启确定性计算。
2. **NaN/-Inf 限制**：输入数据暂不支持 NaN、-Inf。
3. **FLOAT 类型精度**：Atlas 训练系列产品中，FLOAT 输入会转换为 FLOAT16 计算，存在一定精度损失。
4. **dilation 限制**：dilation 值仅支持 1。
5. **padding 约束**：padding 值必须大于等于 0 且小于等于 kernelSize。
6. **ceilMode 限制**：继承正向口径——`ceilMode=True` 时暂不支持 `s_h >= (H_in + padding_size)/(H_out - 1)`、
   `s_w >= (W_in + padding_size)/(W_out - 1)` 的 stride 场景。
7. **`indices` 依赖**：`indices` 必须由正向 MaxPool2dWithMask 产出（每输出位置恰好 1 位置位）；多位置位等非法输入行为未定义。
8. **rank/format**：rank 4、format NCHW（验收范围）；`self` 与 `gradInput`、`gradOutput` 与正向 `out` 严格对齐。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | `gradInput` 满足生态算子开源精度标准的混合容差（atol/rtol）；以原 TBE 的 `gradInput` 为黄金参考逐元素比对，0/非 0 位置集合完全一致 | [生态算子开源精度标准](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md) |
| 性能标准（A2） | 整体性能与现有实现持平，不低于原 TBE 算子的 95% | 任务书性能要求 1 |
| 小 shape 口径（A2） | 100us 以下场景若超过 TBE 耗时 30% 以上，须提供性能仿真图与分析结论，证明与 TBE 一致或更优 | 任务书性能要求 2 |
| 性能标准（300V Pro） | 验收功能与精度；耗时基准参考 A2(910B3) 的 5 倍，部分 shape 超过基准 30% 须提供性能仿真图 | 任务书性能要求 3 |
| 确定性 | 开关开启时多次运行 bit-wise 一致；关闭时差异仅允许出现在浮点累加末位 | 任务书约束说明 1 |
| 模型级精度 | yolov11 + DOTAv1 上 mAP50 与 Atlas 800T A2 的误差 ≤ 0.01 | 任务书模型验证章节 |

> 上表为**验收口径**。本文件为开发前设计文档，不含任何实测数值（含 mAP50）；结论以开发完成后的自测报告为准。

## Atlas 300V Pro 模型接入验证（yolov11 + DOTAv1）

| 项 | 方案 |
| --- | --- |
| 硬件/软件 | Atlas 300V Pro（对照 A2/910B3）；CANN 9.0.0 或 9.1.0；两端同版本、同模型代码，仅目标设备不同 |
| 模型 | yolov11：SPPF 用 `MaxPool2d(kernelSize=5, stride=1, padding=2)`，属窗口重叠场景、命中重叠累加路径；PyTorch 不返回 indices 的 `MaxPool2d` 反向时内部即调用 `max_pool2d_with_indices_backward`，故本算子位于 yolov11 反向图中 |
| 数据集与流程 | DOTAv1（YOLO 格式，链接见任务书）；固定随机种子与超参、同一初始权重，两端各跑一次相同训练/微调流程，结束后在同一验证集以 ultralytics 官方流程评测 mAP50（IoU=0.5） |

**通过判据**：

1. 主判据：`|mAP50(300V Pro) − mAP50(A2)| ≤ 0.01`（绝对误差）；
2. 稳定性：至少重复 2 次（不同种子或轮次检查点）结论一致；首次不达标先排查累加顺序（非确定性路径）、FLOAT 精度损失、
   padding 区越界丢弃逻辑，定位后重跑再判定；3. 链路有效性：反向链路的 `indices` 必须由正向算子实际产出（不允许 CPU/其他后端替代），
   报告中给出算子调用计数或 profiling 证据；
4. 算子级证据：`gradInput` 与 TBE golden 在混合容差下一致，作为 mAP50 结论的技术支撑；
5. 不达标处置：给出训练曲线对比、算子级精度与性能数据，定位偏差来源（累加顺序/精度损失/越界丢弃）与修复计划后再提交验收。

验证流程与结论按要求贡献到
[modelzoo-GPL 目标检测目录](https://gitcode.com/Ascend/modelzoo-GPL/tree/master/contrib/PyTorch/Research/cv/image_object_detection)。

## 兼容性分析

- **接口兼容**：ACLNN 签名与参数顺序（`gradOutput, self, indices, kernelSize, stride, padding, dilation, ceilMode, gradInput`）与官方一致；
  可与现网 TBE 正向/反向互换配套，这也是 mask 编码必须逐字节与 TBE 对齐的原因。
- **算子仓兼容**：新增算子目录落在 `experimental/pooling/`，不改动既有算子文件。
- **确定性开关兼容**：默认（非确定）与开启（确定）由同一份 Kernel 经 tilingKey 分支承载，不引入两套二进制，性能影响在自测报告中给数据。
- **硬件/版本兼容**：tiling 由 `GetCoreMemSize`/`GetAivCoreCount` 运行期计算，同一源码适配 A2 与 300V Pro，差异仅在 tile 与分核粒度；
  CANN 9.0.0 与 9.1.0 接口与 tiling 结构一致，不依赖版本私有接口。
