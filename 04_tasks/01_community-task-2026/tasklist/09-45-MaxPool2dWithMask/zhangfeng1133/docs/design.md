# MaxPool2dWithMask 算子设计文档（Ascend C）

> 算子：`MaxPool2dWithMask`（二维最大池化，输出池化值 `out` 与 mask 语义 `indices`）
> 接口基准：`aclnnMaxPool2dWithMask` / `aclnnMaxPool2dWithMaskGetWorkspaceSize`
> 仓库与落点：`ops-nn` → `experimental/pooling/max_pool2d_with_mask/`（`op_host/`、`op_kernel/`、`op_graph/`）
> 硬件：Atlas 800T A2、Atlas 300V Pro；CANN 9.0.0 或 9.1.0；Ascend C；团队 `zhangfeng1133`
> 任务：2026 社区任务 09-45-MaxPool2dWithMask（正向）；配套反向文档：09-46-MaxPool2dWithMaskBackward

# 需求背景（required）

## 需求来源

社区任务「MaxPool2dWithMask 算子开发」任务书：参考 `aclnnMaxPool2dWithMask` 接口说明，用 Ascend C 实现与原 TBE 功能一致的
MaxPool2dWithMask，完成设计、开发、测试全流程，验收通过后合入昇腾算子开源仓 `ops-nn` 的 `experimental/pooling`。

## 背景介绍

### MaxPool2dWithMask 算子实现优化

`indices` 不是扁平化 int64 argmax，而是 **Ascend mask 语义**：INT8、4 维
`[N, C, k_h×k_w, (⌈H_out×W_out/16⌉+1)×2×16]` 位掩码，是反向算子 MaxPool2dWithMaskBackward 的必需输入，
其**字节级编码口径必须正反向严格一致**（见「反向衔接契约」）。
口径来源：`aclnnMaxPool2dWithMask` 官方 API 文档；op-plugin `opapi/MaxPool2dWithIndicesKernelNpuOpApi.cpp`
（`BLOCKSIZE=16`、`mask_W=CeilDiv(H_out×W_out,16)+1`、dtype=int8、末维 `mask_W×32`）；
实现参考 ops-nn `pooling/max_pool3d_with_argmax_v2`；性能/精度基线为 CANN `opp/built-in/op_impl/ai_core/tbe/` 下 pooling 实现。

### MaxPool2dWithMask 算子实现现状分析（TBE 支持能力）

| 参数名 | 输入/输出 | 描述 | 使用说明 | 数据类型 | 数据格式 | 维度(shape) | 非连续Tensor |
| --- | --- | --- | --- | --- | --- | --- | --- |
| self | 输入 | 待池化输入（公式中的 `input`） | 4 维 NCHW；长度 1 的 `kernelSize/stride/padding/dilation` 在 H/W 取同值 | BFLOAT16、FLOAT16、FLOAT | NCHW、ND | 4 | √ |
| kernelSize | 输入 | 最大池化的窗口大小 | 长度为 1 或 2，且数组元素必须都大于 0 | INT64 | -- | -- | -- |
| stride | 输入 | 窗口移动的步长 | stride 长度为 0 时，stride 数值等于 kernelSize 的值 | INT64 | -- | -- | -- |
| padding | 输入 | 每一条边补充的层数，补充位置填写"负无穷" | 数组长度必须为 1 或 2，且数组元素必须都大于等于 0 或者小于等于 kernelSize/2 | INT64 | -- | -- | -- |
| dilation | 输入 | 控制窗口中元素的步幅 | 值仅支持 1 | INT64 | -- | -- | -- |
| ceilMode | 输入 | 计算输出形状时取整的方法 | True 表示向上取整，False 表示向下取整 | BOOL | -- | -- | -- |
| out | 输出 | 池化后的结果 tensor | shape 由公式推导出 | BFLOAT16、FLOAT16、FLOAT | NCHW、ND | 4 | √ |
| indices | 输出 | 最大值的索引位置组成的 Tensor（mask 语义为 int32 argmax，每通道独立） | shape 由公式推导出 | INT8 | NCHW、ND | 4 | -- |

### MaxPool2dWithMask 算子功能分析

对输入每个通道做二维最大池化，窗口不跨 batch/channel。`indices` 记录"每个输出位置的最大值来自窗口内哪个相对偏移"，
信息等价于逐输出位置的 int32 argmax，以位掩码承载，按 `(n,c)` 独立计算与存储，供反向算子消费。

# 需求分析（required）

## 需求描述

用 Ascend C 在 Atlas 800T A2 与 Atlas 300V Pro 上实现 `MaxPool2dWithMask`：dtype BFLOAT16（仅 A2）/FLOAT16/FLOAT，
format NCHW/ND，rank 4，支持非连续 Tensor；输出 `out` 与 mask 语义 `indices`，shape 由 Host 运行期推导，
`indices` 编码与 TBE 产物逐字节一致；提供 ACLNN 两段式接口、Host 校验与 Tiling、Kernel、原型与 shape/dtype 推导；
泛化支持任意合法 shape 与属性组合（动态 shape、无编译期常量）；默认确定性计算（并列取最小索引）；
性能不低于原 TBE 的 95%；交付自测用例、可复现测试 README、自测报告与算子 README。

## 需求拆解

1. 原型与注册：`op_graph` 原型（属性 `kernelSize/stride/padding/dilation/ceilMode`）、dtype/format 注册、shape/dtype 推导；
2. Host：参数校验、`out`/`indices` shape 推导、workspace 规划、分核与 UB 切分、tilingKey 规划；
3. Kernel：分核落地、搬入（越界 padding 用 `-inf` 填充）、窗口最大值与并列取最小索引、mask 编码与 32 字节块回写；
4. 一致性：并列规则确定、每输出位置恰好 1 位置位、结果可复现；
5. 测试与交付：精度/性能/负向用例、README 与自测报告。

# 详细设计（required）

## 算子分析

### 数学公式

```
out(n, c, h, w) = max over m in [0,k_h), n in [0,k_w) of input(n, c, stride[0]*h + m, stride[1]*w + n)
```

**out 的 shape 推导公式**（`ceilMode=False` 取 floor，`=True` 取 ceil）：

```
H_out = floor|ceil( (H_in + 2*padding[0] - dilation[0]*(kernelSize[0]-1) - 1) / stride[0] ) + 1
W_out = floor|ceil( (W_in + 2*padding[1] - dilation[1]*(kernelSize[1]-1) - 1) / stride[1] ) + 1
[N, C, H_out, W_out]
```

**indices 的 shape 推导公式（原样保留任务书口径）**：

```
[N, C, H_indices, W_indices] = [N, C, k_h * k_w, (ceil(H_out * W_out / 16) + 1) * 2 * 16]
```

**mask 语义解释**（记 `BW = ⌈H_out×W_out/16⌉ + 1`，末维 `= BW × 2 × 16 = BW × 32`）：

| 维度/因子 | 取值 | 语义 |
| --- | --- | --- |
| 第 1、2 维 | `N, C` | 与 `out` 对齐；按 `(n,c)` 通道独立 |
| 第 3 维 | `k_h × k_w` | 窗口内相对偏移枚举，下标 `p = m*k_w + n`（`m∈[0,k_h)`, `n∈[0,k_w)`） |
| 第 4 维 | `BW × 32` int8 | 第 `p` 个偏移的**掩码位平面**，每 16 个输出位置一个分块（`BLOCKSIZE=16`）；块内第 `b` 位（`b = l mod 16`，`l = h*W_out + w`）为 1 表示 `out(h,w)` 的最大值取自偏移 `(m,n)` |
| 尾部 `+1` | 1 个 32 字节尾块 | 对齐/哨兵块，保证任意 `H_out×W_out` 都有完整尾块，位图不跨块 |
| 每块 `2×16` | 32 字节 | 16 位掩码分块的存储颗粒度（32 字节对齐单元，便于 DMA/向量化）；未占用字节写 0 |

**与 int32 argmax 的等价性**：由 `(p,l)` 唯一还原 `(m,n)`、`(h,w)`，得输入坐标
`(stride[0]*h + m - padding[0], stride[1]*w + n - padding[1])`。
**不变量（反向依赖）**：任意 `l` 满足 `Σ_p bit(p,l) == 1`。
**padding 语义**：越界位置按"负无穷"参与比较；因输入不含 `NaN`/`-Inf`，"跳过越界位置"与"以 `-inf` 取 max"等价。

### 支持数据类型

- `self` / `out`：BFLOAT16（仅 Atlas 800T A2）、FLOAT16、FLOAT；`indices`：INT8（固定，mask 语义）；
- 属性：`kernelSize/stride/padding/dilation` 为 INT64 数组（长度 1 或 2，长度为 1 时 H/W 同值），`ceilMode` 为 BOOL；
- dtype 命名口径：同一类型在正向任务书记为 FLOAT、在反向任务书记为 FLOAT32，本设计与配套反向文档视为同一 dtype；
- 中间计算按 `self` dtype；FLOAT 在 Atlas 训练系列产品上内部转 FLOAT16 计算（有精度损失，与 TBE 一致）。

### 支持形状

- `self`：`[N,C,H_in,W_in]`；`out`：`[N,C,H_out,W_out]`（`H_out ≥ 1`、`W_out ≥ 1`）；
  `indices`：`[N,C,k_h*k_w,(⌈H_out*W_out/16⌉+1)*2*16]`，INT8；均 rank 4；
- 泛化覆盖：`kernelSize ∈ {1,2,3,5}`（含 `(2,3)`）、`stride ∈ {0(缺省),1,2,3}`、`padding ∈ {0,1,2}`、
  `ceilMode ∈ {False,True}`、`W_in` 非 32B 对齐、`H_out*W_out` 非 16 倍数、`N*C` 小于/大于/等于 AI Core 数、非连续输入。

## 算子实现

### 实现方案

#### Host 侧设计

沿用 ops-nn pooling 通用结构：`_def.cpp`（注册）、`_infershape.cpp`、`_tiling.cpp`。

##### 1. 参数校验

全部在 Host 完成、不触碰 NPU 内存；任一失败即返回参数错误（aclnn 层 `ACLNN_ERR_PARAM_INVALID`，
图模式对应 GE 参数错误，口径与 ops-nn 既有 pooling 算子一致）。

| # | 校验项 | 判定条件 | 处置 |
| --- | --- | --- | --- |
| 1 | dtype/rank | `self`/`out` 不在 {FLOAT16, FLOAT, BFLOAT16} 或不一致；300V Pro 上为 BFLOAT16；`self` rank ≠ 4 或第 1/2 维 ≤ 0 | 参数错误 / 不支持 |
| 2 | `kernelSize` | 长度 ∉ {1,2}；任一元素 ≤ 0 | 参数错误 |
| 3 | `stride` | 长度 ∉ {0,1,2}；长度 > 0 时元素 ≤ 0；长度 0 时归一化 `stride[i]=kernelSize[i]` | 参数错误 / 归一化 |
| 4 | `padding` | 长度 ∉ {1,2}；`padding[i] < 0` 或 `padding[i] > kernelSize[i]/2` | 参数错误 |
| 5 | `dilation` | 长度 ∉ {1,2}；任一元素 ≠ 1（仅支持 dilation=1） | 不支持 |
| 6 | 推导尺寸 | `H_out ≤ 0` 或 `W_out ≤ 0` | 参数错误 |
| 7 | ceilMode 组合 | `ceilMode=True` 且 `(H_out-1)*stride[0] ≥ H_in + padding[0]`，或 `(W_out-1)*stride[1] ≥ W_in + padding[1]` | 不支持 |
| 8 | `out` | dtype 或 shape 与推导结果不一致 | 参数错误 |
| 9 | `indices` | dtype ≠ INT8，或 shape ≠ `[N,C,k_h*k_w,(⌈H_out*W_out/16⌉+1)*2*16]` | 参数错误 |
| 10 | 输入取值 | 含 `NaN`、`-Inf` | 不在支持域（Host 不读 device 数据，结果未定义） |

第 7 条与任务书「ceilMode 限制」同义；据此可证支持域内每个输出窗口至少覆盖 1 个有效输入元素，
"整窗落在 padding 内"不可达，Kernel 无需该分支。

##### 2. shape 推导与 workspace 规划

- `out` 按公式推导（`ceilMode` 二选一），dtype 跟随 `self`；`indices` 按 mask 公式推导，dtype INT8；
- `indices` 中未被任何输出位置覆盖的位必须为 0（否则反向会误回填）。实现为：由框架对 `indices` 触发 `aclrtMemsetAsync` 清零，
  Kernel 仅对有效位置 1（先读后写合并），workspace 保持 0；只有当 Kernel 能按 32 字节块整体重建掩码（含未覆盖块）时，
  才可省略清零。该取舍不影响外部接口；
- 第二段接口沿调用方 stream 异步执行，除清零外无额外 Host 同步。

##### 3. 分核策略

```
rowTask  = N * C                              # 一阶任务：通道行
blockDim = min(aivCoreNum, rowTask)           # 每核独占整数个 (n,c) 行 → 核间零竞争
若 rowTask < aivCoreNum：blockDim = min(aivCoreNum, rowTask * H_out)     # 二阶：按输出行 (n,c,h) 切
若仍不足：按输出行内 W 维继续切，分片起点必须按 16 对齐                   # 三阶：W 分片
每核任务量 = ceil(总任务数 / blockDim)，余量分到前若干核（尾核收尾）
```

三阶 W 分片按 16 对齐是硬性要求：保证任一 16 位 mask 分块只归属一个核。

##### 4. UB 切分与 Buffer 规划

以"`TILE_H` 个输出行 × `W_out`"为一个 tile；`UB_BUDGET = GetCoreMemSize(UB) - RESERVE`，再按 `BUFFER_NUM` 折半（double buffer）：

```
输入行数 = (TILE_H - 1) * stride[0] + k_h        # dilation = 1
rowInBytes = align32((W_in + 2*padding[1]) * dtypeSize)；rowOutBytes = align32(W_out * dtypeSize)
rowMaskBytes = align32(ceil(W_out * TILE_H / 16) * 2)
TILE_H = min(H_out, max(1, floor(UB_BUDGET / (rowInBytes + rowOutBytes + rowMaskBytes))))
若单行放不下：退化为逐输出行 + 行内 W 分片，保证 TILE_H ≥ 1、TILE_W ≥ 16
```

| Local Memory | 类型 | 容量口径 | 用途 |
| --- | --- | --- | --- |
| `inQue` | `TQue<VECIN,2>` | `((TILE_H-1)*stride[0]+k_h)` 行 padded 行字节 | 输入行块，行内已含左右 `-inf` padding |
| `outQue` | `TQue<VECOUT,2>` | `TILE_H * rowOutBytes` | `out` 结果，算完即搬出 |
| `maskBuf` | `TBuf<VECCALC>` | `k_h*k_w * align32(ceil(TILE_H*W_out/16) * 2)` 字节 | mask 位平面聚合，按 32 字节块回写 |
| `idxBuf` | `TBuf<VECCALC>` | `TILE_H * W_out * sizeof(uint16)` | 记录 `p = m*k_w+n`（与位设置解耦，便于调试） |

容量全部由 `GetCoreMemSize` 运行期计算、不硬编码；Atlas 300V Pro 的 UB 小于 A2，其 `TILE_H` 更小属预期。

##### 5. tilingKey 规划

```
tilingKey = dtypeCode*10000 + pathCode*1000 + ceilModeCode*100 + kernelClass*10 + alignFlag
```

| 位段 | 含义 | 取值 |
| --- | --- | --- |
| `dtypeCode` | dtype | 0: FLOAT16 / 1: FLOAT / 2: BFLOAT16 |
| `pathCode` | 实现路径 | 0: 通用（标量+向量混合） / 1: 向量化（padded 行按 32B 对齐搬入 + 窗口 32B 对齐、dilation=1） |
| `ceilModeCode` | 取整方式 | 0: False / 1: True（影响尾窗有效性判定） |
| `kernelClass` | 窗口规模 | 0: `k_h*k_w ≤ 16` / 1: `> 16`（影响循环展开与 `maskBuf` 布局） |
| `alignFlag` | 尾块对齐 | 0: `H_out*W_out` 为 16 倍数 / 1: 非 16 倍数 |

#### Kernel 侧设计

入口读取 `MaxPool2dWithMaskTilingData`，实例化模板类，执行 `Init` 与 `Process`（CopyIn/Compute/CopyOut 三段），
由 tilingKey 选定路径。按 tilingData 的 `(核起始任务, 核任务数, 切分阶数, TILE_H)` 推进：一阶按 `(n,c)` 行 +
`TILE_H` 输出行分 tile，二阶/三阶按 `(n,c,h)` 与行内 W 分片；分片边界与 16 对齐的 mask 块边界一致；
索引统一用 int64/uint32 计算，避免千万级 shape 溢出。

##### 1. 数据搬入/搬出与越界 padding 处理

- **W 方向**：先在 UB 目标行区域整体 `Duplicate(-inf)`，再按 `padding[1]` 偏移用 `DataCopyPad` 搬入 `W_in` 个有效元素，
  尾部非对齐用 `-inf` 补足 → 窗口取值无需边界判断，与任务书"补充位置填写负无穷"一致；
- **H 方向**：窗口行 `row = stride[0]*h + m - padding[0]`，`row ∉ [0,H_in)` 时该行不参与比较（等价 `-inf` 行）；
- **搬出**：`out` 用 `DataCopyPad` 写回（覆盖非 32B 对齐与非连续输出）；`maskBuf` 按 32 字节块写回 `indices`；
- **非连续 `self`**：CopyIn 时按 stride 逐行寻址（行内连续），或 Host 侧先物化到连续 workspace，按实测择优并打标。

##### 2. 窗口最大值与并列取最小索引

遍历顺序固定为 `(m,n)` 行优先升序（`p = m*k_w+n` 递增），比较用**严格大于**：

```
best = -inf; bestP = 0
for m in [0,k_h): for n in [0,k_w):
    v = inRow(stride[0]*h + m, stride[1]*w + n)
    if (v > best) { best = v; bestP = m*k_w + n; }
```

遍历顺序即偏移索引升序，`v > best` 保留**最先出现**的最大值 → 即"并列取最小索引"，`out(h,w)` 与 `bestP` 唯一确定。
该规则与核数、切分、并行顺序无关，**因此默认即为确定性计算**。`kernelClass=large` 时用向量 `Max` 求 `best`，
再用 `Compare`+`Select` 定位首个相等位置取最小索引，语义一致。

##### 3. indices mask 编码与写回

1. `bw = l >> 4`（第几个 16 位分块）、`b = l & 15`（块内位序），`l = h*W_out + w`；
2. 在 `maskBuf` 第 `bestP` 个位平面的第 `bw` 个 16 位字上置位：`word[bestP][bw] |= (1 << b)`；
3. tile 完成后按"每个 16 位字占 32 字节块"展开写回 `indices[n,c,bestP,bw*32 ...]`，无效位写 0；
4. 第 3 维是 `bestP` 维度，访存天然"按偏移平面分散写"，故以 `(n,c)` 行 + `TILE_H` 输出行聚合以减少 GM 写次数。

块内 16 位字在 32 字节块内的落位以 **TBE 产物逐字节校准**为准：开发期用同一组输入比对 TBE 输出 `indices` 的每个字节；
该校准只影响"块内偏移"常量与 `<<` 方向，不影响 shape 公式、算法结构与反向契约。

#### 反向衔接契约（供 MaxPool2dWithMaskBackward 消费）

| # | 契约项 | 正向保证 |
| --- | --- | --- |
| 1 | `indices` shape/语义 | `[N,C,k_h*k_w,(⌈H_out*W_out/16⌉+1)*2*16]`，INT8；第 3 维＝偏移 `p=m*k_w+n`，第 4 维＝16 输出位置一组的位平面，等价于逐输出位置 int32 argmax |
| 2 | 唯一性 | 每输出位置恰好 1 位置位（并列取最小索引） |
| 3 | 越界 | 掩码解码出的输入坐标可能落在 padding 区，反向须丢弃该梯度 |
| 4 | 数据对齐 | 反向 `gradOutput` 对齐正向 `out`，反向 `gradInput` 对齐正向 `self`（shape/dtype/format 完全一致） |
| 5 | 约束继承 | `dilation` 仅支持 1；输入不含 `NaN`/`-Inf`；`ceilMode` 可用组合与正向同口径 |

### 测试设计

- **精度用例矩阵**（与 TBE 逐元素比对，`indices` 逐字节比对）：dtype ∈ {FLOAT16, FLOAT, BFLOAT16(仅 A2)}；
  shape ∈ {`(1,1,4,4)`,`(1,3,16,16)`,`(8,32,64,64)`,`(2,64,224,224)`,`(1,8,1,1)`,`W_in` 非 32B 对齐,`H_out*W_out` 非 16 倍数}；
  `kernelSize` ∈ {`(1,1)`,`(2,2)`,`(3,3)`,`(2,3)`,`(5,5)`}；`stride` ∈ {0,1,2,3}；`padding` ∈ {0,1,`kernelSize/2`}；
  `ceilMode` ∈ {False,True}；非连续输入/输出；另构造窗口内并列最大值验证"取最小索引"。
- **负向用例**：`kernelSize` 长度 3 / 元素 ≤0、`stride` 元素负、`padding` 超上界、`dilation=2`、
  `ceilMode=True` 命中第 7 条限制、`H_out≤0`、`out`/`indices` dtype 与 shape 不匹配 → 断言参数错误且无 device 访问。
- **性能用例**：与 TBE 相同的环境、shape、dtype、核数下采集算子级耗时（NPU Event，预热后采样取中位数与 p90），
  分小 shape（<100us）与大 shape 两组，核对 95% 与 +30% 两条口径。复现要求：测试代码含 README（环境、编译运行命令、
  用例 CSV 结构、golden 生成方式 = TBE 实现 + CPU golden 双路、结果判读）；自测报告含参数、精度对比与性能数据及截图。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas A2 训练系列产品（Atlas 800T A2） | √ |
| Atlas 推理系列产品（Atlas 300V Pro） | √ |

> BFLOAT16 仅 Atlas 800T A2 支持；FLOAT 在 Atlas 训练系列产品上内部转 FLOAT16 计算。

## 算子约束限制

1. **确定性计算**：MaxPool2dWithMask 默认实现确定性计算。
2. **NaN/-Inf 限制**：输入数据暂不支持 NaN、-Inf。
3. **FLOAT 类型精度**：Atlas 训练系列产品中，FLOAT 输入会转换为 FLOAT16 计算，存在一定精度损失。
4. **ceilMode 限制**：`ceilMode=True` 时暂不支持 `s_h >= (H_in + padding_size)/(H_out - 1)`、
   `s_w >= (W_in + padding_size)/(W_out - 1)` 的 stride 场景（等价判定见 Host 校验第 7 条）。
5. **dilation 限制**：dilation 值仅支持 1。
6. **padding 约束**：`0 ≤ padding[i] ≤ kernelSize[i]/2`，长度 1 或 2。
7. **dtype/format**：`out` 与 `self` dtype 必须一致；`indices` 固定 INT8；format NCHW/ND，rank 4。
8. **3 维输入**：aclnn 文档另声明 `self` 支持 3 维（无 batch），本设计以 4 维为验收范围；若扩展则按同一公式以 C 维承载、`N=1` 退化处理。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | `out` 满足生态算子开源精度标准的混合容差（atol/rtol）；`indices` 为位掩码，按整数逐字节精确一致 | [生态算子开源精度标准](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md) |
| 精度基线 | 以原 TBE 实现的 `out`/`indices` 为黄金参考逐元素比对，`indices` 逐字节 equal | 任务书功能实现要求 |
| 性能标准（A2） | 整体性能与现有实现持平，不低于原 TBE 算子的 95% | 任务书性能要求 1 |
| 小 shape 口径（A2） | 100us 以下场景若超过 TBE 耗时 30% 以上，须提供性能仿真图与分析结论，证明与 TBE 一致或更优 | 任务书性能要求 2 |
| 性能标准（300V Pro） | 验收功能与精度；耗时基准参考 A2(910B3) 的 5 倍，部分 shape 超过基准 30% 须提供性能仿真图 | 任务书性能要求 3 |
| 确定性 | 同输入多次运行 bit-wise 一致；换核数/切分参数结果不变 | 任务书约束说明 1 |

> 上表为**验收口径**。本文件为开发前设计文档，不含任何实测数值；结论以开发完成后的自测报告为准。
> 模型级验证（yolov11 + DOTAv1 的 mAP50）仅反向任务书要求，正向不涉及。

## 兼容性分析

- **接口兼容**：ACLNN 签名、参数顺序、属性语义与 `aclnnMaxPool2dWithMask` 一致；`indices` 保持 mask 语义，
  正向产物可直接喂给反向，与现网 TBE 正向/反向可互换配套。
- **算子仓兼容**：新增算子目录落在 `experimental/pooling/`，不改动既有算子文件；`indices` 编码一经发布即为契约，
  开发期若与 TBE 的块内字节口径有差异，以 TBE 为准修正编码常量，不改变对外 shape 公式与语义。
- **硬件/版本兼容**：tiling 由 `GetCoreMemSize`/`GetAivCoreCount` 运行期计算，同一份 Kernel 源码在 A2 与 300V Pro 上
  按 UB 与核数自适应，差异仅在 tile 大小与分核粒度；CANN 9.0.0 与 9.1.0 接口与 tiling 结构一致，不依赖版本私有接口。
