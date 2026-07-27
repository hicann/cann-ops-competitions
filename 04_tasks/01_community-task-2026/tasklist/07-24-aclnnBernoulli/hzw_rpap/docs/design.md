# aclnnBernoulli 算子内存优化设计文档

## 1. 需求背景（required）

### 1.1 需求来源

- 任务名称：`7月社区任务-aclnnBernoulli算子开发`
- 任务依据：[aclnnBernoulli 任务书](../../../../docs/202607/aclnnBernoulli_task_doc.md)
- 目标产品：Atlas A2/A3 训练系列产品
- 软件版本：CANN 8.5.0 及以上
- 开发语言：Ascend C

本任务要求在不改变 `aclnnBernoulli` 对外功能的前提下，优化 A2/A3
标量概率路径的设备内存占用，使其与等价 GPU 调用的内存差距小于 5%，
同时保证性能不低于原实现。

### 1.2 算子功能

`aclnnBernoulli` 根据 `self` 的 shape、dtype 和布局生成独立 Bernoulli
样本。`self` 的数值不参与计算，随机流由 `seed` 和 `offset` 决定。

对每个逻辑位置 `i`：

```text
out[i] ∈ {0, 1}
P(out[i] = 1) = prob
```

接口同时支持 out-of-place 和 inplace 调用。输出为浮点、整数或 `BOOL`
时，均写入对应 dtype 可精确表示的 0 或 1。

### 1.3 接口能力与约束

| 参数 | 类型 | 支持范围 | 约束 |
| --- | --- | --- | --- |
| `self` | `aclTensor*` | FP16、FP32、FP64、BF16、UINT8、INT8、INT16、INT32、INT64、BOOL | ND；0～8 维；可为空、连续或非连续 |
| `prob` | `aclScalar*` | FP16、FP32、FP64、BF16 | 有限值，且 `0 <= prob <= 1` |
| `seed` | `int64_t` | INT64 | 用于确定随机流 |
| `offset` | `int64_t` | INT64 | `offset % 4 == 0` |
| `out` | `aclTensor*` | 与 `self` 相同 | shape、dtype 与 `self` 一致；可为非连续 Tensor |

本次优化针对标量 `prob` 接口。Tensor-probability 接口不进入自定义融合
路径，继续沿用原有实现。

### 1.4 原实现与问题分析

CANN 8.5.0 中 A2/A3 标量概率的一般路径位于：

```text
random/dsa_gen_bit_mask/op_host/op_api/aclnn_bernoulli.cpp
```

其执行链路为：

```text
Contiguous(self)
  -> DSAGenBitMask
  -> Fill(ones)
  -> DropoutDoMask
  -> Cast(target dtype)
  -> ViewCopy(out)
```

设输出元素数为 `N`，目标 dtype 字节数为 `B`，DSA 生成的 packed mask
大小为：

```text
M = ceil(N / 128) * 16 = align_up(N, 128) / 8
```

原链路除 packed mask 外，还会产生全 shape 的 ones、mask 展开结果以及
部分 dtype 下的 Cast 结果。主要逻辑中间值如下：

| 中间值 | shape / dtype | 逻辑字节数 |
| --- | --- | ---: |
| packed mask | `uint8[M]` | `M` |
| `inputOnes` | 全 shape，FP16 或 FP32 | `N×2` 或 `N×4` |
| `doMaskOut` | 与 `inputOnes` 相同 | `N×2` 或 `N×4` |
| Cast 结果 | 目标 dtype | 最多 `N×B` |
| 非连续输入的连续副本 | 目标 dtype | `N×B` |

执行器可能复用不同逻辑节点的存储，因此不能把表中各项直接相加作为峰值；
但这些随 `N` 线性增长的中间 Tensor 会显著增加 workspace 和 GM 流量。
以 A2/A3、shape `[4096,4096]`、`prob=0.5` 为例，原 ACLNN workspace
在两个平台上的结果一致：

| dtype | 原 workspace |
| --- | ---: |
| FP16 / BF16 | 35,652,608 B |
| FP32 | 69,207,040 B |
| INT64 / FP64 | 69,207,552 B |

因此，本任务的核心不是更换随机数算法，而是在保留 DSA 随机流的同时，
消除 `Fill + DropoutDoMask + Cast` 引入的全 shape 中间值。

## 2. 需求分析（required）

### 2.1 设计目标

1. 保持公共 ACLNN 签名、两阶段调用方式和返回语义不变。
2. 覆盖任务书要求的全部 dtype、0～8 维、空 Tensor、连续与非连续布局。
3. 保留 `prob=0`、`prob=1` 快路径及 `seed/offset` 随机状态语义。
4. 连续 dense 输出不再申请随 shape 增长的全量中间 Tensor。
5. 非连续 Tensor、非零 storage offset 和特殊 storage/view 关系不得越界写。
6. A2 与 A3 使用同一份 Ascend C 源码，硬件资源通过 Tiling 运行时查询。
7. 新实现性能不低于原实现，并按等价调用协议完成 NPU/GPU 内存对比。

### 2.2 外部依赖

本方案不引入新的第三方组件，复用 CANN 已有能力：

| 依赖 | 用途 |
| --- | --- |
| `DSAGenBitMask` | 生成与原实现一致的 packed 随机 mask |
| Ascend C Vector API | 在 UB 内完成 mask 展开、Select、Cast 和 Gather |
| `ZerosLike` / `OnesLike` | 处理 `prob=0/1` 精确快路径 |
| `ViewCopy` | 将连续临时结果写回非连续逻辑 view |
| `PlatformAscendC` | 查询 UB 容量和 Vector Core 数量 |

### 2.3 方案比较

| 方案 | 内存效果 | 随机语义风险 | 实现风险 | 结论 |
| --- | --- | --- | --- | --- |
| 仅对旧 `Fill/DropoutDoMask` 做存储别名 | 仍保留多 Kernel 和部分全量中间值 | 低 | dtype、别名和非连续布局约束复杂 | 不采用 |
| 单 Kernel 自行实现 RNG 和写出 | 可完全融合 | 高，需要重新证明 seed/offset 消耗和统计质量 | A2/A3 无可直接复用的 Philox API | 不采用 |
| 保留 DSA，新增 packed mask 展开 Kernel | 连续路径无全 shape 临时值 | 低，可保持原 DSA 随机流 | 需要证明同址展开安全 | 采用 |

最终方案为：

```text
DSAGenBitMask(mask aliases dense out)
  -> BernoulliMask(high-to-low waves)
  -> out
```

对于不满足同址条件的输出，采用独立 packed mask 和连续输出临时值作为
fallback，以保证通用布局语义。

## 3. 详细设计（required）

### 3.1 总体设计

`BernoulliMask` 是新增的 Ascend C L0 算子。它不读取 `self` 的数据，
只读取 DSA 生成的 packed mask，并在 UB 内将每个 bit 展开为目标 dtype
的 0/1。

根据输出布局和容量，Host 侧选择三条路径：

```text
                         +--> prob=0/1: ZerosLike / OnesLike
                         |
aclnnBernoulli ----------+--> 一般概率 + 可安全同址
                         |      DSA(mask写入out起始区)
                         |        -> BernoulliMask原地展开
                         |
                         +--> 一般概率 + 不可同址
                                独立mask -> 连续结果
                                  -> ViewCopy(out)
```

### 3.2 Host 侧设计

#### 3.2.1 参数校验与特殊路径

Phase 1 按以下顺序处理：

1. 校验 `self`、`prob`、`out`、`workspaceSize` 和 `executor` 非空。
2. 校验 `self/out` 的 dtype、shape 和 format，并限制 rank 不超过 8。
3. 校验 `prob` 为有限值且位于 `[0,1]`。
4. 校验 `offset % 4 == 0`。
5. 空 Tensor 返回 `workspaceSize=0`，不下发 Kernel。
6. `prob=0` 调用 `ZerosLike`，`prob=1` 调用 `OnesLike`。
7. 其他概率进入 DSA + `BernoulliMask` 路径。

#### 3.2.2 Dense alias 判定

仅当以下条件全部满足时，packed mask 才允许与用户 `out` 共用存储：

1. `out` 为标准行主序连续布局，stride 与 shape 完全匹配；
2. view offset 和 storage offset 均为 0；
3. 逻辑 view 恰好覆盖整个底层 storage，即二者元素数相等；
4. 输出容量满足 `N×B >= M`；
5. Tensor format 不是私有格式。

第 3 条用于保护“连续小 view + 更大 backing storage”的场景。即使该
view 本身连续，也不能让原地 Kernel 写入 view 之外的 guard 区域。

满足条件时，Host 在 `out` 起始地址上建立 `UINT8[M]` view，作为
`DSAGenBitMask` 的输出。由于 mask 和最终输出属于同一 GM allocation，
连续 dense 路径无需独立 packed-mask Tensor。

由于 DSA mask 按 128 bit 对齐，极小 Tensor 可能不满足容量条件。对于
1/2/4/8 字节输出，允许同址的最小元素数分别为 16/8/4/2；更小输入走
独立 mask fallback。

#### 3.2.3 Fallback 与 FP64 非连续写回

下列场景使用保守 fallback：

- 非连续输出；
- view/storage offset 非零；
- 逻辑 view 未覆盖完整 storage；
- 极小 Tensor 容量不足；
- 不支持直接解释的私有格式直接在参数校验阶段拒绝。

fallback 路径为：

```text
DSAGenBitMask(independent mask)
  -> BernoulliMask(continuous temporary)
  -> ViewCopy(out)
```

A2 的 `ViewCopy` 路径未直接声明 FP64 支持。对于非连续 FP64 输出，Host
为源和目标建立保持 shape、stride 和 offset 的 INT64 bit-view，再执行
`ViewCopy`。该过程只搬运 64-bit 位模式，不进行数值转换。

#### 3.2.4 架构分流

A2/A3 对应的 DAV_2201 路径进入本次 DSA 融合实现。DAV_3510 及其他
非目标架构继续沿用 `StatelessBernoulli` 路径，避免改变非目标产品行为。

### 3.3 Tiling 设计

#### 3.3.1 TilingData

TilingData 只传递 Kernel 必需的信息：

```text
totalElements
elementsPerCore
tileElements
maskAliasesOut
```

Host 通过 `PlatformAscendC` 动态获取 UB 大小和 Vector Core 数，不硬编码
A2/A3 的资源参数。

#### 3.3.2 分块与分核

设输出元素宽度为 `B`，预留 UB 为 8 KiB。Host 以 4-byte work buffer
和保守的 1-byte mask 开销估算单元素空间：

```text
bytesPerElement = B + 4 + 1
tileElements = align_down((UB - 8 KiB) / bytesPerElement, 256)
tileElements = min(tileElements, 16384)
blockDim = min(vectorCoreNum, ceil(totalElements / tileElements))
elementsPerCore = align_up(ceil(totalElements / blockDim), 256)
```

小 shape 自动减少核数；大 shape 在 A2 上使用 40 个 Vector Core。
fallback 路径中，各核处理互不重叠的连续元素区间。

当 `maskAliasesOut=1` 时，Host 设置 `SetScheduleMode(1)`。该模式保证
所有已启动核心以一致的同步轮次执行高到低 wave，避免核间提前覆盖尚未
读取的 mask。

#### 3.3.3 TilingKey

| Key | 输出 dtype | Kernel 实例 |
| ---: | --- | --- |
| 1 | FP16 | `half` |
| 2 | FP32 | `float` |
| 3 | FP64 | `uint64_t` 位模式，Select + Gather |
| 4 | UINT8 / BOOL | `uint8_t` |
| 5 | INT8 | `int8_t` |
| 6 | INT16 | `int16_t` |
| 7 | INT32 | `int32_t` |
| 8 | INT64 | `int64_t` |
| 9 | BF16 | `bfloat16_t` |

### 3.4 Kernel 侧设计

#### 3.4.1 单 tile 执行流程

每个 tile 依次执行：

1. 使用 `DataCopyPad` 将 `ceil(count/8)` 字节 packed mask 搬入 UB；
2. 使用 `Select` 将 mask bit 展开为 FP16 或 FP32 的 0/1；
3. 浮点结果直接写出，整数和 BF16 在 UB 内 Cast；
4. FP64 使用 Select + Gather 组装 IEEE-754 64-bit 位模式；
5. 仅搬出 `count×sizeof(T)` 个有效字节，不写对齐 padding。

各 dtype 的计算路径如下：

| 类别 | dtype | UB 中间类型 | 写出策略 |
| --- | --- | --- | --- |
| 16-bit 浮点 | FP16 | FP16 | Select 后直接写出 |
| 32-bit 浮点 | FP32 | FP32 | Select 后直接写出 |
| 64-bit 浮点 | FP64 | 两个 FP32 word | Select high word，Gather 交织 |
| 8-bit / BOOL | INT8、UINT8、BOOL | FP16 | Select 后 Cast |
| 16-bit 整数 | INT16 | FP16 | `CAST_RINT` |
| 32/64-bit 整数 | INT32、INT64 | FP32 | `CAST_TRUNC` |
| BF16 | BF16 | FP32 | `CAST_RINT` |

#### 3.4.2 同址展开安全性

packed mask 初始位于 `out` 的低地址区域。如果从低地址开始展开，输出
可能覆盖后续尚未读取的 mask。为此，alias 路径按高地址到低地址分 wave
处理。

设尚未展开区间为 `[0, end)`，输出元素宽度为 `d` 字节，本轮起点为：

```text
start = align_up(ceil(ceil(end / 8) / d), 256)
```

此时满足：

```text
d * start >= ceil(end / 8)
```

因此 `[start,end)` 的输出写区域不会覆盖 `[0,ceil(end/8))` 中仍需读取
的 mask。所有核心完成当前 wave 后执行 `SyncAll`，再处理更低地址区间。
最后不超过 256 个元素的前缀由 core 0 先完整搬入 UB，再写回输出。

所有已启动核心必须参与相同次数的 `SyncAll`；即使某个核心在当前 wave
没有有效输出，也不能提前返回。

#### 3.4.3 UB 预算

Kernel 采用单缓冲：

```text
maskQueue = align_up(ceil(tileElements / 8), 32)
outQueue  = align_up(tileElements * B, 32)
workBuf   = align_up(tileElements * 4, 32)
UB_total  = maskQueue + outQueue + workBuf
```

以 A2 的 192 KiB UB 为例：

| dtype 宽度 | tileElements | maskQueue | outQueue | workBuf | 总计 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 B | 16,384 | 2,048 B | 16,384 B | 65,536 B | 83,968 B |
| 2 B | 16,384 | 2,048 B | 32,768 B | 65,536 B | 100,352 B |
| 4 B | 16,384 | 2,048 B | 65,536 B | 65,536 B | 133,120 B |
| 8 B | 14,336 | 1,792 B | 114,688 B | 57,344 B | 173,824 B |

各路径均低于 UB 容量，并保留 8 KiB 预算。A3 使用相同公式按运行时资源
重新计算。当前版本不启用双缓冲，因为其会缩小单 tile，且现有 Profile
未显示足以抵消该开销的收益。

### 3.5 内存与性能分析

#### 3.5.1 内存模型

| 指标 | 原实现 | 新实现（连续 dense） |
| --- | --- | --- |
| 主要链路 | DSA + Fill + DropoutDoMask + Cast + ViewCopy | DSA + BernoulliMask |
| 全 shape GM 临时值 | 至少一个，随 dtype 和规划变化 | 0 |
| packed mask | 独立分配 | 复用 `out` 起始存储 |
| 全 shape 写出 | 多次 | 一次 `N×B` |
| 非连续输出 | 连续化后多级中间值 | 一个连续结果 + ViewCopy |

A2 和 A3 在 `[4096,4096]` 一般概率路径上的 workspace 结果一致：

| dtype | 原 workspace | 新 workspace | 减少比例 |
| --- | ---: | ---: | ---: |
| FP16 / BF16 | 35,652,608 B | 1,024 B | 99.997% |
| FP32 | 69,207,040 B | 1,024 B | 99.999% |
| INT64 / FP64 | 69,207,552 B | 1,024 B | 99.999% |

workspace 结果用于证明算子内部随 shape 增长的临时值已被消除；最终
NPU/GPU 内存一致性仍以等价框架调用的进程峰值为准，不能用 workspace
降幅替代。

#### 3.5.2 测试对象与证据绑定

A2/A3 实测使用同一源码版本，结果分别绑定目标平台包和实际加载的
custom DSO：

| 对象 | 环境 | 版本标识 | 主证据 |
| --- | --- | --- | --- |
| 源码 | `experimental/random/bernoulli_mask` | commit `295d1e7fe793d3e5477d00cd1af86ebcea770f1d` | 算子 tree `db92e5b6d2999518723f231804d8c8a6202bacad` |
| A2 | Ascend 910B4，CANN 8.5.2 | package SHA-256 `06e71eca975f1ac8735352071e9984a35891ebceb1b6a3304b0d442ea111419a` | `results/*/20260725-p0-guard-hardening-final-a2-*` |
| A3 | Ascend910_9362，CANN 8.5.2，构建目标 `ascend910_93` | package SHA-256 `05040ef871a62bff8d66a43a098460fbfcd0990a5a24586dcd8e7b50d1d42029` | `results/a3/a3-final-007/` |
| GPU | NVIDIA H800 MIG | B011～B014，各 20 个隔离进程 | `results/gpu/20260725-h800-mig-torch-bernoulli/` |

#### 3.5.3 A2/A3 与 GPU 内存对比

三个平台采用相同 `[8192,8192]` shape、dtype、`prob=0.5`、inplace
`Tensor.bernoulli_` 调用层级和隔离进程协议。A2、A3 各完成 direct
ACLNN old/new 160 条和 torch-npu old/new 160 条记录；GPU 完成 4 个
case、每 case 20 条记录。下表为中位 total peak：

| Case | dtype | 指标 | H800 MIG | A2 | A2/GPU 差距 | A3 | A3/GPU 差距 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| B011 | FP32 | allocated | 268,435,456 B | 268,437,504 B | 0.000763% | 268,437,504 B | 0.000763% |
| B011 | FP32 | reserved | 268,435,456 B | 272,629,760 B | 1.5625% | 272,629,760 B | 1.5625% |
| B012 | FP16 | allocated | 134,217,728 B | 134,219,776 B | 0.001526% | 134,219,776 B | 0.001526% |
| B012 | FP16 | reserved | 134,217,728 B | 138,412,032 B | 3.1250% | 138,412,032 B | 3.1250% |
| B013 | INT64 | allocated | 536,870,912 B | 536,872,960 B | 0.000381% | 536,872,960 B | 0.000381% |
| B013 | INT64 | reserved | 536,870,912 B | 541,065,216 B | 0.7813% | 541,065,216 B | 0.7813% |
| B014 | BF16 | allocated | 134,217,728 B | 134,219,776 B | 0.001526% | 134,219,776 B | 0.001526% |
| B014 | BF16 | reserved | 134,217,728 B | 138,412,032 B | 3.1250% | 138,412,032 B | 3.1250% |

A2、A3 的四个 case 在 total peak allocated 和 reserved 两种可定义
口径下均小于 5%。同一 inplace 协议下，A2/A3 的 incremental peak
allocated 中位数为 1,536 B、reserved 为 2,097,152 B，GPU 两项均为
0 B，因此增量百分比的分母为 0。报告保留原始字节数并将百分比标记为
`undefined`，不将其改写为 0%。

#### 3.5.4 A2/A3 性能结果

两个平台均对 5 种 dtype 的 old/new 各执行 20 个隔离进程；每个进程
预热 10 次，再进行 100 次 device event 计时。每个平台 200/200 条记录
成功，且每组 old/new 输出 hash 一致。

**A2 性能：**

| dtype | 原/新 P50 (ms) | 原/新 P90 (ms) | P50 改善 |
| --- | ---: | ---: | ---: |
| FP16 | 0.236425 / 0.221500 | 0.241940 / 0.227110 | 6.313% |
| FP32 | 0.396570 / 0.229040 | 0.405220 / 0.233360 | 42.245% |
| BF16 | 0.256160 / 0.228795 | 0.264550 / 0.234860 | 10.683% |
| INT64 | 0.517965 / 0.379865 | 0.524110 / 0.385890 | 26.662% |
| FP64 | 34.954804 / 0.393395 | 35.015261 / 0.400220 | 98.875% |

**A3 性能：**

| dtype | 原/新 P50 (ms) | 原/新 P90 (ms) | P50 改善 |
| --- | ---: | ---: | ---: |
| FP16 | 0.210415 / 0.191985 | 0.213310 / 0.197960 | 8.759% |
| FP32 | 0.287395 / 0.200915 | 0.290620 / 0.207110 | 30.091% |
| BF16 | 0.226400 / 0.201440 | 0.229470 / 0.207100 | 11.025% |
| INT64 | 0.341265 / 0.264750 | 0.343050 / 0.271180 | 22.421% |
| FP64 | 34.992384 / 0.341385 | 35.031490 / 0.346200 | 99.024% |

A2/A3 的 P50、P90 均未出现性能回退。FP64 的主要收益来自 UB 内
Select + Gather，避免原路径的低效 64-bit 标量构造和多级中间值。

### 3.6 支持硬件

| 芯片版本 | 支持情况 | 验证情况 |
| --- | :---: | --- |
| Atlas 800I/T A2，`ascend910b` | √ | 已完成核心验收矩阵、TTK 和 Sanitizer |
| Atlas A3 训练系列，`ascend910_93` | √ | 已完成同等的核心验收矩阵、TTK 和 Sanitizer |

### 3.7 算子约束与设计边界

1. 不新增用户可见约束，公共 shape、dtype、layout 和随机参数语义保持不变。
2. 本次融合仅替换 A2/A3 标量概率的一般概率路径。
3. Tensor-probability 接口继续沿用原路径。
4. 非连续或不满足完整 storage 覆盖条件的输出需要连续临时值和 ViewCopy。
5. `offset` 按任务书要求校验为 4 的倍数。

## 4. 特性交叉分析

| 交叉场景 | 风险 | 设计处理 |
| --- | --- | --- |
| out-of-place × 连续 dense | packed mask 额外分配导致峰值增长 | mask 复用 `out` 起始存储 |
| inplace × 连续 dense | mask 与最终输出同址覆盖 | 高到低 wave + `SyncAll` |
| 连续 view × 更大 backing storage | Kernel 可能写到逻辑 view 外 | 仅当 view 元素数等于 storage 元素数时允许 alias |
| 非连续 × 非零 offset | 物理地址与逻辑顺序不一致 | 连续临时结果 + ViewCopy |
| FP64 × 非连续 | ViewCopy 不直接声明 DOUBLE 支持 | INT64 bit-view 搬运位模式 |
| 极小 Tensor × mask 对齐 | 输出容量小于 16-byte mask | 独立 mask fallback |
| 空 Tensor × Kernel 调度 | 无效下发或除零 | Phase 1 直接返回 workspace 0 |
| `prob=0/1` × 随机路径 | 不必要的 DSA 和同步开销 | ZerosLike / OnesLike 快路径 |
| A2 × A3 | UB 和核数不同 | `PlatformAscendC` 运行时查询 |

## 5. 可维可测分析

### 5.1 验收标准

| 验收项 | 标准 | 验证方法 |
| --- | --- | --- |
| 功能 | 覆盖全部 dtype、rank 0～8、空/非连续/inplace 和边界参数 | ACLNN 功能矩阵、ST、Host UT、TTK |
| 随机性 | 固定 seed/offset 可重现；改变状态后随机流改变；分布无退化 | checksum、Wilson 区间、双侧二项检验和 Holm 校正 |
| 内存 | 与等价 GPU 调用的内存差距小于 5% | 同 shape/dtype/API 层级的隔离进程峰值对比 |
| 性能 | 不低于原 A2/A3 实现 | 预热后 device event P50/P90 对比 |
| 安全 | 不发生越界、竞争或同步错误 | guard-region 与 mem/race/init/sync Sanitizer |

随机算子的一般概率输出不要求与 GPU 逐元素相同。固定 seed/offset 用于
验证 NPU 新旧实现的可重现性；跨平台正确性通过值域和统计检验判断。

### 5.2 覆盖矩阵与实测证据

#### 5.2.1 用例覆盖

| 验证维度 | 覆盖内容 |
| --- | --- |
| dtype | 10 种 `self/out` dtype，4 种 `prob` dtype |
| shape | 0～8 维、空 Tensor、标量、对齐边界、小 shape 和大 shape |
| layout | 连续、转置、切片、非零 offset、oversized storage guard |
| 概率 | `0`、`1`、接近边界及一般概率 |
| RNG | 相同/不同 seed，合法/不同/非法 offset |
| 接口 | out-of-place、inplace、标量概率及 Tensor-probability 回归 |
| 平台 | A2 与 A3 构建和实机验证 |

#### 5.2.2 A2/A3 核心验收结果

核心验收项在 A2/A3 上采用对称矩阵和相同判据：

| 验证项 | A2 | A3 | 结论 |
| --- | ---: | ---: | --- |
| ACLNN 功能矩阵 | 77/77 | 77/77 | 覆盖 10 dtype、rank 0～8、布局、概率、offset、inplace 和 guard |
| ACLNN ST | 97/97 | 97/97 | 覆盖接口校验、重现性、alias/fallback 边界和异常参数 |
| 固定状态重现与敏感性 | 通过 | 通过 | 相同 seed/offset checksum 相同，改变 seed 或 offset 后不同 |
| 正式分布统计 | 15/15 | 15/15 | 99% Wilson、双侧精确二项检验和 Holm 校正均通过 |
| 最大绝对 z 值 / 最小原始 p-value | 1.661061 / 0.097172 | 1.661061 / 0.097172 | 未发现随机分布退化 |
| old/new 正式性能 | 200/200 | 200/200 | 5 种 dtype 的 P50/P90 均无回退，输出 hash 一致 |
| direct ACLNN 内存 | 160/160 | 160/160 | B011～B014、old/new、每组 20 个隔离进程 |
| torch-npu 内存 | 160/160 | 160/160 | 与 GPU 使用等价框架调用协议 |
| NPU/GPU total peak | 4/4 `<5%` | 4/4 `<5%` | allocated 和 reserved 均满足阈值 |
| Tensor-probability 兼容回归 | 8/8 | 8/8 | 验证系统 Tensor-probability API 未被自定义标量接口回归 |

#### 5.2.3 工程与安全验证

TTK 与 Sanitizer 在 A2/A3 上使用相同用例和判定标准。Sanitizer
单独使用与 Release 包同源的插桩包，不替代功能、性能和内存测试所用的
Release 包。

| 工具层级 | 结果 | 说明 |
| --- | --- | --- |
| A2 TTK Kernel | 通用 26/26 + 存储复用 8/8 | 性能、精度和越界检查均为 PASS |
| A2 Sanitizer | 115 PASS、13 `LIMITED_MASKED_VSEL`、0 FAIL | 完整执行 128 项 memcheck/racecheck/initcheck/synccheck |
| A3 TTK Kernel | 通用 26/26 + 存储复用 8/8 | 性能、精度和越界检查均为 PASS |
| A3 Sanitizer | 115 PASS、13 `LIMITED_MASKED_VSEL`、0 FAIL | 完整执行 128 项 memcheck/racecheck/initcheck/synccheck |

CANN 8.5 对 masked `Select` 的部分 initcheck 用例存在工具识别限制，
因此 A2/A3 各有 13 项标记为 `LIMITED_MASKED_VSEL`。这些项目未计入 PASS；
其余 115 项通过，未发现代码故障。

平台专项工程验证中，A2 AscendOpTest 使用默认阈值完成 2 个固定随机状态
用例，结果为 2/2 PASS；A3 Host UT 为 36/36 PASS，并完成 Basic Profile
采集。A2/A3 的 TTK Kernel 精度检查均为 34/34 PASS。

### 5.3 兼容性分析

- 保留 `aclnnBernoulli` / `aclnnInplaceBernoulli` 的公共签名和两阶段执行方式。
- `self/out` 的 dtype、shape、format 和非连续 view 语义保持不变。
- DSA 的 seed、offset 和 `1-prob` 传参方式与原实现一致。
- `prob=0/1` 继续使用原有精确快路径。
- FP64 bit-view 仅改变搬运解释，不改变输出数值。
- 非目标架构继续沿用原 `StatelessBernoulli` 路径。
- 若特定平台或 dtype 的融合路径出现兼容性问题，可在 Host 侧回退到原链路；
  回退后需重新评估内存指标。
