# SquaredDifference 算子设计文档

## 1. 需求背景

### 1.1 需求来源

根据 CANN 社区任务 2026 及“CANN 训练营 2026 暑期季—西北工业大学专场”
任务书，参考 CANN 内置 SquaredDifference 箠子的 TBE 实现，在昇腾 NPU 上
使用 Ascend C 实现功能一致的 AI Core Kernel，完成算子定义、InferShape、
Tiling、Kernel、调用样例与测试，并在验收后向 `ops-math` 仓提交代码。

目标硬件为 Atlas A2 训练系列产品和 Atlas A3 系列产品。CANN 版本以
`ops-math` 目标分支的配套版本为准。

### 1.2 参考实现

TBE 与算子元信息参考路径：

```text
/usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_impl/ai_core/tbe/impl/dynamic/squared_difference.py
/usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_proto/inc/
/usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_impl/ai_core/tbe/config/ascend910b/
```

本设计同时参考以下资料：

- CANN 内置 TBE 源码 `squared_difference.py`
- 本地 Ascend C 原型工程中的 `op_host/`、`op_kernel/`、`examples/` 和 `tests/`
- `ops-math/math/squared_difference` 中的算子定义、README 和广播 InferShape 语义
- CANN 社区设计模板及 `ops-math` 单算子目录规范

### 1.3 TBE 实现现状分析

TBE 计算图首先按右对齐广播规则推导输出形状，再分别广播 `x1` 和 `x2`，
最后执行一次向量减法和一次向量乘法：

```python
x1_broadcast = broadcast(x1, output_shape)
x2_broadcast = broadcast(x2, output_shape)
difference = vsub(x1_broadcast, x2_broadcast)
y = vmul(difference, difference)
```

其数学表达式为：

$$
y_i = (x1_i - x2_i)^2
$$

TBE 使用 `ELEWISE_WITH_BROADCAST` 模式进行动态 Shape 分类，并由自动调度器
完成调度与构建。现有能力如下：

| 参数 | 输入/输出 | 支持数据类型 | 格式 | Shape 约束 |
| --- | --- | --- | --- | --- |
| x1 | 输入 | bfloat16、float16、float32、int32、int64 | ND | 与 x2 可广播 |
| x2 | 输入 | bfloat16、float16、float32、int32、int64 | ND | 与 x1 可广播 |
| y | 输出 | 与输入一致 | ND | 为 x1、x2 的广播结果 Shape |

两个输入的数据类型必须相同，输出数据类型与输入相同。

## 2. 需求分析

### 2.1 功能需求

Ascend C 实现需要满足：

1. 结果语义与 TBE `SquaredDifference` 一致。
2. 支持 `bfloat16`、`float16`、`float32`、`int32`、`int64`。
3. 支持同 Shape、任一输入为标量、单轴广播和一般多维广播。
4. 输出 Shape 按右对齐广播规则推导。
5. 支持非 32 字节对齐的尾块，不能读写越界。
6. 支持动态实际 Shape；Tiling 阶段根据运行时实际 Shape 生成切分参数。

### 2.2 广播规则

从两个输入的最后一维向前比较。每一维必须满足以下条件之一：

- 两个维度相等；
- `x1` 对应维度为 1；
- `x2` 对应维度为 1。

维数较少的输入在高维侧补 1。输出对应维度取可广播后的维度值。例如：

```text
x1: [2, 3, 1, 5]
x2: [   1, 4, 5]
y : [2, 3, 4, 5]
```

若某一维同时不相等且均不为 1，则 Host 侧返回失败。

### 2.3 外部接口

使用自动生成的两段式 aclnn 接口：

```cpp
aclnnSquaredDifferenceGetWorkspaceSize(...)
aclnnSquaredDifference(...)
```

接口详细说明见 [aclnnSquaredDifference.md](aclnnSquaredDifference.md)。

### 2.4 约束与边界

- 输入与输出仅支持 `ND` 格式。
- `x1`、`x2` 的数据类型必须一致，`y` 与输入类型一致。
- 当前广播 TilingData 最多保存 16 维；输入 Rank 范围为 0～16，Rank 0
  按单元素标量处理。
- 当前版本要求实际维度为正数；含 0 维度的空 Tensor 场景需在交付前结合
  目标 CANN 版本的广播语义补充验证。
- 整数运算遵循目标数据类型的溢出语义，不做饱和处理。
- 非连续输入由 OpDef 的 `AutoContiguous` 能力转换为连续存储后进入 Kernel。

## 3. 详细设计

### 3.1 工程结构

最终向 `ops-math` 交付时使用小写下划线算子目录：

```text
squared_difference/
├── docs/
│   ├── aclnnSquaredDifference.md
│   └── design.md
├── examples/
│   └── test_aclnn_squared_difference.cpp
├── op_api/                         # 可选，由工程自动生成时可不提交
├── op_host/
│   ├── squared_difference_def.cpp
│   ├── squared_difference_infershape.cpp
│   ├── squared_difference_tiling.cpp
│   └── CMakeLists.txt
├── op_kernel/
│   ├── squared_difference.cpp
│   ├── squared_difference.h
│   ├── squared_difference_tiling_data.h
│   ├── squared_difference_tiling_key.h
│   └── CMakeLists.txt
├── tests/
│   └── ut/
│       ├── op_host/
│       └── op_kernel/
├── CMakeLists.txt
└── README.md
```

### 3.2 Host 侧设计

#### 3.2.1 算子定义

通过 `OP_ADD(SquaredDifference)` 注册算子：

- 输入：`x1`、`x2`，均为必选 Tensor；
- 输出：`y`，必选 Tensor；
- 数据类型：BF16、FP16、FP32、INT32、INT64；
- 格式：ND；
- 使用 `AutoContiguous()` 保证 Kernel 接收连续 Tensor；
- 为目标 SoC 注册 AI Core 配置，交付前分别完成 A2 与 A3 编译验证。

#### 3.2.2 InferShape

InferShape 将两个输入 Shape 高维补 1 后逐维比较，输出广播 Shape。伪代码为：

```text
out_rank = max(rank(x1), rank(x2))
for axis in [0, out_rank):
    d1 = aligned_x1[axis]
    d2 = aligned_x2[axis]
    require d1 == d2 or d1 == 1 or d2 == 1
    out[axis] = broadcast_dim(d1, d2)
```

不满足广播条件时返回 `GRAPH_FAILED`，避免 Kernel 侧产生非法地址映射。

#### 3.2.3 广播场景分类

Host 侧将输入场景划分为四类，并通过 `broadcastMode` 传入 Kernel：

| broadcastMode | 场景 | Kernel 搬入策略 |
| ---: | --- | --- |
| 0 | x1、x2 元素数均等于输出元素数 | 两路连续 DataCopy |
| 1 | x1 为单元素，x2 与输出同形 | x1 Duplicate，x2 DataCopy |
| 2 | x2 为单元素，x1 与输出同形 | x1 DataCopy，x2 Duplicate |
| 3 | 一般广播 | 广播描述符快速路径或坐标映射兜底 |

一般广播进一步分析每个输入的连续模式：

- `SimpleBroadcast`：连续内块在若干广播维度上重复；
- `AxisBroadcast`：中间连续轴块在尾部维度上重复；
- `TailCopy`：输出当前位置后存在可连续搬入的尾块；
- `TailRepeat`：输出当前位置后存在同一值的连续重复段；
- 通用兜底：按输出坐标和输入 stride 计算输入索引。

该分类将多数常见广播转换为批量 `DataCopyPad` 或 `Duplicate`，减少逐元素
`GetValue/SetValue` 的使用。

#### 3.2.4 分核策略

Host 侧读取 AIV 核数和 UB 容量。输出总元素数记为 `totalNum`：

1. 小 Shape（`totalNum <= 2048`）默认单核，避免多核启动开销。
2. 大 Shape 根据数据类型和广播模式设置目标单核元素数。
3. 最终核数不超过平台 AIV 核数，并至少为 1。
4. 不能整除时，余数分配给前 `tailBlockNum` 个大核。

```text
smallCoreDataNum = totalNum / coreNum
tailBlockNum     = totalNum % coreNum
bigCoreDataNum   = smallCoreDataNum + (tailBlockNum > 0 ? 1 : 0)
```

前 `tailBlockNum` 个核处理 `bigCoreDataNum`，其余核处理
`smallCoreDataNum`，保证核间负载差最多为一个元素。

#### 3.2.5 UB 切分策略

Kernel 为两个输入队列和一个输出队列启用双缓冲。普通类型的单元素 UB
预算为：

$$
bytesPerElem = BUFFER\_NUM \times 3 \times sizeof(T)
$$

BF16 路径还需要两个 FP32 临时 Tensor：

$$
bytesPerElem_{BF16} = BUFFER\_NUM \times 3 \times 2 + 2 \times 4
$$

`tileDataNum` 由 UB 容量除以上述预算得到，再按 32 字节块对齐，并限制在
`uint32_t` 可表达范围内。每核根据自身元素数计算 Tile 数及尾 Tile 元素数。

#### 3.2.6 TilingKey 规划

数据类型在编译期决定 Ascend C 模板实例，因此使用五个调度模式：

| schMode | 数据类型 | Kernel 类型 |
| ---: | --- | --- |
| 0 | BF16 | `bfloat16_t` |
| 1 | FP16 | `half` |
| 2 | FP32 | `float` |
| 3 | INT32 | `int32_t` |
| 4 | INT64 | `int64_t` |

广播模式保存在 TilingData 中，不额外增加 Kernel 二进制数量。

#### 3.2.7 TilingData

主要字段如下：

| 字段 | 含义 |
| --- | --- |
| smallCoreDataNum / bigCoreDataNum | 小核/大核处理元素数 |
| finalSmallTileNum / finalBigTileNum | 小核/大核 Tile 数 |
| tileDataNum | 单 Tile 最大元素数 |
| smallTailDataNum / bigTailDataNum | 尾 Tile 元素数 |
| tailBlockNum | 大核数量 |
| x1Num / x2Num | 两个输入实际元素数 |
| broadcastMode | 同形、标量或一般广播模式 |
| outRank | 输出 Rank |
| outShape | 输出各维度 |
| x1Shape / x2Shape | 高维补 1 后的输入 Shape |
| x1Stride / x2Stride | 输入连续存储 stride |
| Simple/Axis/Tail 描述字段 | 广播快速路径参数 |

Workspace 当前设置为 0，不使用额外 GM 临时空间。

### 3.3 Kernel 侧设计

#### 3.3.1 总体流程

Kernel 使用 `Init` 和 `Process` 两阶段。每个 Tile 采用：

```text
GM(x1, x2) -> CopyIn/广播展开 -> UB
UB         -> Sub -> Mul -> UB(y)
UB(y)      -> CopyOut -> GM(y)
```

输入、输出使用 `TQue` 管理，`BUFFER_NUM=2`。计算流水为
`CopyIn -> Compute -> CopyOut`，尾 Tile 使用 `DataCopyPad` 处理非对齐长度。

#### 3.3.2 地址与分核

每核根据 `GetBlockIdx()`、大小核参数和 `tailBlockNum` 计算其输出起点。
一般广播下，输出线性索引先转换为多维坐标，再使用输入 stride 得到对应
输入线性索引。广播维的输入坐标固定为 0。

为避免通用广播逐元素重复执行除法和取模，Kernel 缓存当前输出坐标与输入
索引，并通过坐标进位增量更新。

#### 3.3.3 数据搬入

- 同形：两个输入均按连续块搬入。
- 标量广播：标量只从 GM 读取一次，使用 `Duplicate` 写入当前 Tile。
- 简单广播：连续片段使用 `DataCopyPad`，重复值使用 `Duplicate`。
- 一般广播：优先按广播边界切分连续片段；无法识别的模式使用坐标映射兜底。
- INT64 在目标指令能力受限的填充场景使用循环 `SetValue`。

对于一般广播，小核数据量不超过 512 或广播边界较小时，按完整 Tile 搬入并
在 Tile 内展开；大数据且连续边界较长时，按两个输入的最小广播边界分段，
减少无效地址计算并扩大批量搬运长度。

#### 3.3.4 计算路径

| 数据类型 | 计算方案 |
| --- | --- |
| FP16、FP32、INT32 | `Sub` 后复用输出 LocalTensor 执行 `Mul` |
| BF16 | 转 FP32 做减法，按 BF16 舍入差值，再转 FP32 平方，最后舍入到 BF16 |
| INT64 | 逐元素计算并做四路循环展开，使用无符号中间值保证模 $2^{64}$ 运算可定义 |

BF16 路径在减法后先回写一次 BF16，再执行平方，用于对齐 TBE 在 BF16
中间结果上的舍入行为。`Sub` 与 `Mul` 之间插入向量流水同步屏障，避免读写
相关冲突。

#### 3.3.5 数据搬出

每个 Tile 计算完成后，使用 `DataCopyPad` 将有效元素写回输出 GM。输出地址
以本核起点为基址，保证各核写区间互不重叠。最后一个 Tile 仅写
`tailDataNum` 个元素。

## 4. 性能优化方案

### 4.1 多核与自适应切分

- 根据 Shape、数据类型、是否一般广播动态选择核数；
- 大小核负载差控制为一个元素；
- 小 Shape 避免盲目满核造成启动开销；
- Tile 大小根据实际 UB 容量计算，不硬编码固定容量。

### 4.2 搬运与 UB 优化

- 两输入一输出全部双缓冲；
- 同形场景完全使用连续 DataCopy；
- 标量广播只读取一次标量；
- 常见广播合并为连续复制或批量填充；
- 输出 LocalTensor 复用为差值和平方结果，减少 UB 占用；
- BF16 临时 FP32 Buffer 仅在 BF16 模板中分配。

### 4.3 通用广播优化

- Host 预计算对齐 Shape、stride 和广播描述符；
- Kernel 增量维护坐标，减少每个元素的除法、取模；
- 依据广播边界分段，使 DataCopy/Duplicate 尽可能覆盖长连续区间；
- 复杂非连续模式保留正确性兜底路径，避免为性能牺牲泛化能力。

### 4.4 性能验收方法

在 Atlas A2/A3 目标环境中对 Ascend C 与 TBE 使用相同输入、相同 Stream 和
相同预热/重复次数测量。重点覆盖：

- 所有核参与的同 Shape 大 Tensor；
- x1/x2 标量广播；
- 尾维、首维、中间维和多轴广播；
- 32 字节对齐与非对齐尾块；
- 五种数据类型。

验收标准为所有核参与计算场景下 Ascend C 性能不低于 TBE 的 95%。对于
10 μs 以下小 Shape，若绝对差异不超过任务书允许范围，则附性能仿真图与
流水分析。本文档不预填尚未实测的性能数据。

## 5. 可维可测分析

### 5.1 Host 单元测试

至少覆盖：

1. 五种数据类型对应的 TilingKey；
2. 同 Shape、Rank 不同但可广播、标量广播和一般广播；
3. 不可广播 Shape 返回失败；
4. x1/x2 类型不一致返回失败；
5. 单核、多核、大小核和尾 Tile 参数；
6. Rank 0、Rank 1、Rank 16 边界；
7. A2/A3 平台信息下的 UB 切分。

### 5.2 Kernel/端到端测试矩阵

| 类别 | 代表 Shape |
| --- | --- |
| 同形 | `[1]`、`[19]`、`[1024, 1024]` |
| x1 标量 | `[]` 与 `[2, 3, 4]` |
| x2 标量 | `[2, 3, 4]` 与 `[1]` |
| 尾维广播 | `[2, 3, 1]` 与 `[2, 3, 17]` |
| 首维广播 | `[1, 3, 17]` 与 `[8, 3, 17]` |
| 中间维广播 | `[2, 1, 17]` 与 `[1, 5, 1]` |
| Rank 不同 | `[3, 1]` 与 `[2, 1, 17]` |
| 多轴复杂广播 | `[2, 1, 4, 1]` 与 `[1, 3, 1, 5]` |
| 非对齐尾块 | 输出元素数为 3、7、19、33、257 |
| 大 Shape | 足以启用全部 AIV 核的 Shape |

随机数据需覆盖正数、负数、零、极值、Inf/NaN（浮点）以及整数溢出边界。

### 5.3 Golden 与精度标准

Golden 按以下方式计算：

```python
golden = (numpy.broadcast_to(x1, out_shape) -
          numpy.broadcast_to(x2, out_shape)) ** 2
```

BF16/FP16 Golden 需要按目标 TBE 的中间舍入顺序生成，避免仅用 FP64 表达式
掩盖中间精度差异。

- INT32/INT64：逐元素二进制一致。
- BF16/FP16/FP32：满足 CANN Judge 对应题目的默认精度阈值。
- 特殊值：NaN 按位置一致，Inf 符号和位置一致。

### 5.4 自验证报告

最终交付的自验证报告应包含：

- 环境信息、CANN 版本、SoC 型号与代码 Commit；
- 完整用例清单及运行命令；
- Host UT、Kernel UT、端到端样例的执行日志；
- CANN Judge 全部用例通过截图；
- 各测试点 Ascend C/TBE 性能、比值和截图；
- 未达标小 Shape 的仿真流水图及原因分析。

## 6. 支持硬件

| 硬件 | 设计目标 | 交付要求 |
| --- | :---: | --- |
| Atlas A2 训练系列产品 | √ | 编译、功能、精度、性能实测 |
| Atlas A3 系列产品 | √ | 编译、功能、精度、性能实测 |

## 7. 风险与处理

| 风险 | 影响 | 处理方案 |
| --- | --- | --- |
| 一般广播逐元素寻址开销大 | 性能低于 TBE | 扩充广播模式识别，按边界批量搬运 |
| INT64 缺少完整向量指令能力 | Kernel 吞吐较低 | 循环展开，并以多核和大 Tile 降低控制开销 |
| BF16 中间舍入顺序不同 | 精度与 TBE 不一致 | 按 TBE 行为保留差值 BF16 舍入点 |
| A2/A3 注册配置差异 | 某一 SoC 无法构建或加载 | 分别生成并验证 SoC 配置与二进制 |
| 空 Tensor 广播语义差异 | Shape 推导异常 | 对照目标 CANN 版本补充 0 维度规则和测试 |
| 现有原型测试覆盖不足 | 泛化缺陷未暴露 | 按测试矩阵重构参数化 UT，不沿用单一固定用例 |

## 8. 兼容性分析

该工作为 SquaredDifference 的 A2/A3 Ascend C AI Core 实现补充，不改变算子
名称、输入输出顺序、数据类型集合和广播语义。上层图调用保持兼容；aclnn
接口由算子定义生成，调用方仍使用标准两段式接口。合入前需要在目标
`ops-math` 分支中确认是否复用已有 op_host/op_graph，避免重复注册。

## 9. 验收标准

| 验收项 | 标准 | 来源 |
| --- | --- | --- |
| 功能 | 五种类型和合法广播场景结果正确 | SquaredDifference 任务书/TBE |
| 精度 | 满足 CANN Judge 默认阈值 | SquaredDifference 任务书 |
| 性能 | 全核场景不低于 TBE 的 95% | SquaredDifference 任务书 |
| 泛化 | 验收泛化用例全部通过 | SquaredDifference 任务书 |
| 文档 | 设计、API、README、自验证报告完整 | 社区任务提交规范 |
