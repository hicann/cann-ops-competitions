# 需求背景（required）

## 需求来源

segment_coo 算子来源于图神经网络（GNN）消息传递中的分段归约（segment reduction）需求，对应 torch_scatter.segment_coo 的语义，用于将属于同一 segment 的特征按指定归约方式聚合，在昇腾 NPU 上基于 Ascend C 实现。

## 背景介绍

segment_coo 沿 index 的最后一维，将 src 中属于同一分组索引的元素归约写入 out。与 scatter 类算子相比，segment_coo 的关键前提是 index 沿归约维非降序排列，因此同一 segment 的元素在内存中连续出现，可以利用有序段扫描避免全局原子和跨核冲突。

当前 torch_npu 生态中，segment_coo 通常由 index_add_ / scatter_reduce_ / index_reduce_ 等 ATen 算子组合实现。该组合路径存在两个问题：

1. 组合路径需要多次中间张量分配与搬移，访存开销较大。
2. min/max 依赖的 index_reduce_ 在部分 NPU 后端存在 CPU 回退，成为 min/max 路径的主要性能瓶颈。

本设计以该组合路径作为性能基线，使用 Ascend C 实现有序段扫描，并在密集均匀的 segment 分布场景下进一步使用按 run 的逐通道并行段内归约，使 min/max 性能显著优于组合路径。

## segment_coo 与 scatter 选型差异

| 特性 | segment_coo | scatter |
|------|-------------|---------|
| index 顺序 | 沿 index.dim()-1 非降序 | 任意顺序 |
| 归约方式 | sum/add/mean/min/max | 另含 mul 等 |
| 并行策略 | 有序段扫描，段内归约，无需全局原子 | 依赖原子或散射写 |
| 典型性能 | 有序场景下通常更快 | 通用但访存模式更复杂 |

由于 segment_coo 对 index 有序性有硬性约束，本文在 host 侧对 index 进行非降序校验；不满足约束时直接报错，避免 device-side assert 污染运行环境。

# 需求分析（required）

## 需求描述

使用 Ascend C 编程语言实现 segment_coo 算子，支持 float16、bfloat16、float32、int8、uint8、int32、int64 数据类型，支持 sum/add/mean/min/max 五种归约方式。`segment_min_coo` / `segment_max_coo` 返回 `(out, arg_out)`；`segment_coo(..., reduce="min"/"max")` 仅返回 `out`。

## 需求拆解

1. 支持 float16、bfloat16、float32、int8、uint8、int32、int64 数据类型。
2. 支持 sum/add/mean/min/max 五种归约方式；`segment_min_coo` / `segment_max_coo` 额外返回 arg_out，通用 `segment_coo` 的 min/max 不返回 arg_out。
3. 支持 index 前缀维广播，归约维为 index.dim()-1。
4. 空 segment 输出 0；min/max 并列时 arg_out 返回末位位置。
5. 所有用例性能满足 0.45 倍标杆要求，同图数据 segment_coo 不慢于 scatter。

# 详细设计（required）

## 算子分析

### 数学公式

设归约维为 dim，dim_size 为 segment 数。以下公式在归约维的一维视图上描述，前缀维与后缀维经广播展开后逐行独立处理。记 I_s = {i | index[i] = s}：

- sum/add：out[s] = Σ_{i∈I_s} src[i]，空段输出 0
- mean：out[s] = (Σ_{i∈I_s} src[i]) / |I_s|，空段输出 0
- min：out[s] = min_{i∈I_s} src[i]，arg_out[s] = max{i∈I_s | src[i] = out[s]}；空段 out[s]=0 且 arg_out[s]=0
- max：out[s] = max_{i∈I_s} src[i]，arg_out[s] = max{i∈I_s | src[i] = out[s]}；空段 out[s]=0 且 arg_out[s]=0

浮点 sum/add/mean 采用有序段扫描，累加顺序与输入顺序一致，保证确定性。整数 sum/add/mean 统一使用 int64 累加器，计算完成后按 PyTorch 整数 dtype cast 规则回写：int32/int64 按补码截断，int8 按有符号补码截断，uint8 按模 2^8 环绕。整数 mean 的取整规则为向零截断（truncation toward zero），与 torch_scatter CPU 标杆一致。min/max 在并列时按“后写者优先”持续更新 arg_out，从而返回段内末位位置。

### 支持数据类型与精度路径

| 数据类型 | 归约路径 | 说明 |
|----------|----------|------|
| float16 | 原生参与归约；密集均匀 min/max 路径直接使用 half | 精度与 torch_scatter 对齐，允许浮点误差 |
| float32 | 原生参与归约 | 精度与 torch_scatter 对齐 |
| bfloat16 | host 侧先转 float32 参与归约，回写前转回 bfloat16 | 仅功能验收 |
| int32 | 原生或 count 版归约；min/max 支持向量归约 | 参与功能与性能验收 |
| int64 | sum/add/mean 使用 int64 累加；密集均匀 value-only min/max 采用 int32/uint32 高低位拆分的 Ascend C 快速路径；通用场景仍走 Ascend C 有序段扫描 | 参与功能与性能验收 |
| int8 / uint8 | sum/add/mean 使用 int64 累加后按上述 dtype cast 规则回写；min/max 走标量路径 | 仅功能验收 |

index 固定为 int64，arg_out 固定为 int64。所有整数类型的 sum/add/mean 均以 int64 累加器计算，以保证 int32/int64 的 bit-wise 一致，并统一 int8/uint8 的溢出回写规则。

### 支持形状

src 维度为 1~8 维，index.dim() <= src.dim()，index.size(-1) = src.size(index.dim()-1)，输出将归约维替换为 dim_size。

## 算子实现

### 总体架构

算子按三层组织：

1. PyTorch 适配层：提供与 torch_scatter 一致的 Python API，完成 dtype 校验与设备判断。
2. host 侧：完成 index 有序性校验、广播展开、形状推导、输出分配、tiling 计算与 kernel 启动。
3. kernel 侧：完成有序段扫描与段内归约，输出最终 out，min/max 按需输出 arg_out。

### host 侧设计

#### 形状归一化

host 侧将归约维通过 movedim 移到最后一维并转为连续布局：

- src 归一化为 [rows, seq_len]
- index 按前缀维广播后归一化为 [prefix_rows, seq_len]
- out 归一化为 [rows, dim_size]

其中：

- prefix_rows = prod(src.shape[:dim])
- suffix_rows = prod(src.shape[dim+1:])
- rows = prefix_rows × suffix_rows
- seq_len = src.shape[dim]

当 suffix_rows > 1 时，index 的每个 prefix 行被 rows 中的多个行共享：第 r 行使用 prefix_row = r / suffix_rows 对应的 index 行，src 行与 out 行按 r 直接寻址。

#### 参数校验与 dim_size 推导

host 侧完成以下校验：

1. src/index 设备与 dtype 校验。
2. index.dim() <= src.dim()，且 index.size(-1) 等于归约维长度。
3. index 沿归约维非降序。
4. index 取值位于 [0, dim_size) 区间。
5. 提供 out 时，out dtype、设备与形状校验。

dim_size 推导优先级为：

1. 显式传入 dim_size
2. 提供 out 时取 out.size(dim)
3. index 为空时取 0
4. 否则取 index[..., -1].max() + 1

dim_size 与 out 同时提供时以 dim_size 为准，并校验 out.size(dim) 与 dim_size 一致，不一致则报错。

#### 分核策略

优先满核使用：usedCoreNum = min(rows, coreNum)。rows 不能均分时，余数行分配给前几个核，形成大核（多处理一行）和小核。每个核处理的行区间由 formerNum/formerLength/tailLength 描述。

#### tiling 策略

通用路径按行处理，每个核内将 seq_len 按 srcTileLen 切分，逐 tile 搬运与计算。srcTileLen 由 UB 大小、double buffer 系数、out tile、arg tile 及归约临时缓冲共同决定，保证不越界且尾块按剩余长度处理。

密集均匀路径按 run 切分，并联合选择 srcTileLen 与 batchRuns：

- srcTileLen 必须是 runLen 的整数倍，且至少覆盖若干完整 run，避免逐 run 发起 DMA。
- batchRuns 决定单次 WaitVDone 前可流水执行的 run 数量。
- 二者在 UB 预算下联合求解，保证 runVal 缓冲不会过度挤压 src tile。

### kernel 侧设计

kernel 侧分为通用有序段扫描路径和密集均匀快速路径。

#### 通用有序段扫描路径

通用路径面向任意维度和广播场景，按行独立处理。对于单行内的有序 index，执行线性段扫描：

1. 相同 segment 的元素连续出现，扫描得到每个 run 的 [start, end) 区间。
2. 对每个 run 执行段内归约，将完整 run 的结果写入输出 tile。
3. 位于 tile 边界的 segment 视为部分段，携带跨 tile 状态，与下一 tile 的同 segment 续段合并后再写回。

各 reduce 的处理规则：

- sum/add：对完整 run 求段内和；浮点使用向量 ReduceSum，无法向量化的 dtype 使用标量累加。
- mean：在段内和的基础上除以段内元素个数，空段输出 0。
- min/max：维护当前极值与极值下标；比较时使用 <= / >= 更新，保证并列时返回末位下标；arg_out 只在需要时输出。
- 对向量归约指令不支持或 run 长度过小的场景，回退为标量比较/累加，保证正确性。

#### 密集均匀快速路径

当输入满足以下条件时，启用按 run 的逐通道并行段内归约路径：

- src 为二维 [seq_len, C]，归约维为 0。
- index 为一维有序张量，且可表示为 0..dim_size-1 按 runLen 连续重复：每个 segment 均非空，长度均为 runLen = seq_len / dim_size。
- 通道数 C 为偶数，且 runLen 满足向量归约的 32 字节对齐约束。
- runLen 不超过向量归约的 mask 上限，且单个 [C, runLen] 块可在 UB 内完整处理；否则回退通用路径。
- min/max value-only 场景；需要 arg_out 时仍走通用路径。

该路径将 src 布局为 [C, seq_len]，每个 run 覆盖 [C, runLen] 区域。对 float16、float32、int32 的每个 run 使用 WholeReduceMin/WholeReduceMax，以 repeatTime=C 让 C 个通道并行完成段内归约，消除逐通道 index 扫描和逐通道归约。int64 的密集均匀路径见下一小节。

为得到 [C, seq_len] 布局，host 侧在启动前完成一次转置/重排；该拷贝成本计入算子总耗时。该路径用这一次转置换取 index 只扫描一次和通道维向量化，在 C 较大、run 数量较多的场景下收益大于转置开销。对于 float16，转置后的 half 直接进入 kernel，不再额外做 half→float32 拷贝。

计算流程如下：

1. 按 run 将 segment 区间划分到多个 AIV 核，各核写互不重叠的 out 行。
2. 每个核按 tile 搬入 [C, srcTileLen] 的 src 块与对应 index 块。
3. 对 tile 内的每个 run 调用 WholeReduceMin/Max，将结果写入 runVal 缓冲。
4. 攒满 batchRuns 个 run 后统一 WaitVDone，再将结果批量写回 out。
5. 输入搬运采用双缓冲预取：在处理当前 tile 的同时预取下一 tile，使 MTE2 搬入与 Vector 计算、MTE3 写回重叠。

该路径的适用 dtype 为 float16、float32、int32；int64 采用后续描述的高低位拆分快速路径。对于 float16，直接以 half 参与归约，避免 host 侧 half→float32 的整张拷贝。

#### int64 密集均匀 min/max 快速路径

int64 的 WholeReduce 指令在当前硬件上不可用，因此 int64 密集均匀 value-only 场景采用高低位拆分的纯 Ascend C 路径。

host 侧将 int64 src 转置为 [C, seq_len] 布局后，按如下方式拆成两个 32 位张量：

- 先将有符号 int64 映射为可无符号比较的排序键：u = v ^ 0x8000000000000000。
- high = (uint32_t)(u >> 32)，low = (uint32_t)(u & 0xFFFFFFFF)。

kernel 对每个 run 执行两级归约：

1. 使用 WholeReduceMax/Min<uint32_t> 对 high 逐通道并行做段内归约，得到 highBest。
2. 将 low 中 high != highBest 的通道替换为无影响的填充值；min 填充 0xFFFFFFFF，max 填充 0。该步骤使用向量 Compare/Select 完成 masked select，避免逐元素处理。
3. 使用 WholeReduceMin/Max<uint32_t> 对处理后的 low 逐通道并行做段内归约，得到 lowBest。
4. 以 highBest、lowBest 合成 u，再通过 u ^ 0x8000000000000000 还原为 int64 结果。

该路径复用密集均匀路径的 [C, seq_len] 布局、多核按 run 切分、srcTileLen/batchRuns 联合 tiling 与输入双缓冲预取。高低位两个张量由 host 一次性准备，kernel 内不再处理 int64 标量。int64 快速路径的 srcTileLen/batchRuns 求解需同时计入 high/low 两个 32 位缓冲与 runVal 缓冲；UB 不足时回退通用路径。该路径仅用于 value-only min/max；需要 arg_out 的 int64 场景仍走通用有序段扫描路径。

若目标 CANN 工具链的 WholeReduceMin/Max 不支持 uint32_t，则可将 high/low 先与 0x80000000 异或映射到 int32_t，再执行有符号归约；该映射等价于无符号比较，仍保持纯 Ascend C 实现。

#### 多路径对比

| 路径 | 适用场景 | 核心策略 | 预期收益 | 主要代价 |
|------|----------|----------|----------|----------|
| 通用有序段扫描 | 任意维度、广播、suffix、arg_out | 单行内扫描 run，段内向量/标量归约 | 功能完备，确定性好 | 每个 suffix 行重复扫描 index |
| 密集均匀快速路径 | 二维、dim=0、一维 index、等长 run、min/max value-only | [C, seq_len] 布局 + WholeReduce 逐通道并行段内归约 | 消除逐通道 index 扫描与逐通道归约 | host 侧一次转置，条件受限 |
| int64 高低位拆分快速路径 | 二维、dim=0、一维 index、等长 run、int64 min/max value-only | uint32 high/low 拆分 + 两级 WholeReduce | 规避 int64 WholeReduce 缺失，避免标量扫描，且保持纯 Ascend C 归约 | 需要额外 high/low 张量与两级归约 |

路径选择在 host 侧完成：优先判断密集均匀条件，满足则进入快速路径；否则进入通用有序段扫描路径。所有路径共享相同的输入校验、形状推导和 out 语义。

#### dtype 路径与 tilingkey

host 侧通过 dtypeTag、reduceTag、needArgOut 和布局标志告知 kernel 走不同分支：

- float16 / float32：优先向量归约。
- bfloat16：host 侧转 float32 后进入 float 路径，回写前转回。
- int32：优先 count 版向量归约。
- int64：sum/add/mean 使用 int64 累加；min/max value-only 密集均匀场景按高低位拆分快速路径处理，其余场景走标量扫描。
- int8 / uint8：sum/add/mean 使用 int64 累加后按上述 dtype cast 规则回写；min/max 走标量路径。

#### 双缓冲与同步

输入队列使用深度为 2 的 TQue。密集均匀路径在循环中先发起下一 tile 的 CopyIn，再处理当前 tile，实现搬入与计算重叠。输出侧 out tile 在 UB 内缓冲后整 tile 回写，回写前通过 WaitScalarToMte3 / WaitVDone / WaitMte3Done 保证标量写、向量计算与 MTE3 搬出之间的依赖关系。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Ascend 950 系列（arch35） | √ |

## 算子约束限制

- index 必须为 int64，且沿归约维非降序；不满足时 host 侧直接报错，不进入 kernel。
- index 值必须落在 [0, dim_size) 区间。
- 不支持 reduce="mul"，传入 "mul" 须抛出 ValueError。
- index.dim() <= src.dim()，且 index.size(-1) == src.size(index.dim()-1)。
- min/max 并列时 arg_out 返回末位位置。
- 空 segment：out 输出 0，min/max 的 arg_out 输出 0。
- 提供 out 时：sum/add 不在现有 out 上额外累加，按段内结果写回；mean/min/max 忽略 out 原内容，按归约结果写回，min/max 的 arg_out 由算子生成。
- int8/uint8 的 min/max 因 Ascend C 归约指令不支持该 dtype，走标量实现。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | float16/float32 与 torch_scatter CPU 比对，sum/add/mean 采用有序段扫描实现确定性；int32/int64 与 torch_scatter CPU bit-wise 一致；bfloat16/int8/uint8 仅功能验收；min/max 的 out 与 arg_out 与标杆一致（平局后写者优先） | 任务书精度要求 |
| 性能标准 | 性能不低于标杆性能的 0.45 倍，即耗时不超过标杆耗时的约 2.22 倍；同图数据 segment_coo 不慢于 scatter | 任务书性能要求 |

## 兼容性分析

新算子，不涉及兼容性分析。
