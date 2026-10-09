# TurboQuantAttentionScore 算子设计文档

# 一、需求背景（required）

## 1.1 需求来源

通过社区任务完成面向量化Paged KV cache的attention score算子贡献。算子采用Ascend C实现，通过统一ACLNN接口供标准MHA/GQA调用。

## 1.2 背景介绍

### 1.2.1 TurboQuantAttentionScore算子实现优化

标准attention score需要读取历史Key并计算`Q@K^T`。当Key以TurboQuant格式存储时，score阶段需要同时完成页表寻址、主编码解包、centroid查表、metadata缩放和QJL残差修正。先恢复完整Key会产生大规模中间Tensor；逐query重复恢复Key则会重复消耗带宽。

该算子为新增量化score算子，CANN内置OPP中没有同名TBE实现。功能标杆采用包内PyTorch参考实现；质量参考采用原始BF16 Q/K的PyTorch FP32 Full-QK；性能标杆采用BF16 Paged Full-QK设备实现。


### 1.2.2 TurboQuantAttentionScore标杆实现现状分析

#### 1.2.2.1 标杆算子支持的数据类型和数据格式

令`T`为query token数、`B`为batch、`QH/KH`为query/KV head数、`P`为物理页数、`L`为输出KV容量，block size固定128，`D=J=128`。

| 参数 | 方向 | 数据类型 | 数据格式 | 形状 |
|---|---|---|---|---|
| `query_rotated` | 输入 | BF16 | ND | `[T,QH,128]` |
| `query_qjl` | 输入 | BF16 | ND | `[T,QH,128]` |
| `key_cache_idx` | 输入 | UINT8 | ND | `[P,128,KH,16*mse_bits]` |
| `key_cache_qjl` | 输入 | UINT8 | ND | `[P,128,KH,16]` |
| `key_cache_norm`、`key_cache_gamma` | 输入 | BF16/FP16 | ND | 各`[P,128,KH]` |
| `block_table` | 输入 | INT32 | ND | `[B,max_blocks_per_seq]` |
| `seq_lens` | 输入 | INT32 | ND | `[B]` |
| `attn_scores` | 输出 | FP32 | ND | `[T,QH,L]` |

属性`mse_bits`支持2、3、4，`max_kv_len=-1`时采用页表容量，否则采用指定输出容量。

#### 1.2.2.2 标杆算子实现描述

标杆实现依据`block_table`将逻辑Key位置映射到物理页，解包主编码并映射到centroid，同时解包QJL符号。对每个query head和Key位置计算：

```text
score[i] = norm[i] * dot(query_rotated, centroid[unpack(idx[i])])
         + sqrt(pi/2)/128 * gamma[i]
           * dot(query_qjl, unpack_sign(qjl[i]))

unpack_sign(bit) = 2*bit - 1
```

GQA按`kv_head=floor(q_head/(QH/KH))`映射。有效长度外输出`-inf`。softmax和Value聚合不属于该算子。

#### 1.2.2.3 标杆算子实现流程

标杆实现先通过页表将逻辑Key位置映射到物理页，然后分别解包主量化索引和QJL符号。主分支执行centroid查表、query内积和norm缩放；QJL分支执行实数query投影与符号向量内积，并乘gamma及固定系数。两路结果相加后写入有效score，输出容量尾部填`-inf`。

# 二、需求分析（required）

## 2.1 外部组件依赖

| 组件 | 用途 |
|---|---|
| CANN 9.1.0及以上 | ACLNN executor、算子注册与运行时调用 |
| Ascend C | Query Pack和Score Kernel实现 |
| Ascend Matmul API | BF16输入、FP32累加的Cube矩阵乘 |

PyTorch、NumPy和ml_dtypes仅用于测试数据、Full-QK和KL参考计算，不属于算子运行依赖。

## 2.2 内部适配模块

| 模块 | 作用 |
|---|---|
| Public ACLNN接口 | 参数检查、临时Tensor分配、内部节点编排 |
| TqQueryPack | Prefill query预排布，仅Prefill启用 |
| TqScoreCompute | 页表寻址、Key解码、Cube计算及输出写回 |
| Tiling模块 | 计算任务数、分核、Cube tiling、offset表及workspace |

## 2.3 需求模块设计

### 2.3.1 Ascend C算子原型

```text
TurboQuantAttentionScore(
    query_rotated, query_qjl,
    key_cache_idx, key_cache_qjl,
    key_cache_norm, key_cache_gamma,
    block_table, seq_lens,
    mse_bits=3, max_kv_len=-1)
  -> attn_scores
```

### 2.3.2 Ascend C算子相关约束

- `head_dim=qjl_dim=128`，block size固定128。
- query为BF16，metadata为同一种BF16或FP16，输出为FP32。
- `QH/KH`必须为正整数；支持标准MHA/GQA，不支持MLA。
- Decode要求`T=B`；Prefill要求`B=1`。
- 输入为连续ND Tensor，不支持广播。

# 三、需求详细设计（required）

## 3.1 调用方式

对外采用ACLNN两阶段调用：

```cpp
aclnnTurboQuantAttentionScoreGetWorkspaceSize(..., &workspaceSize, &executor);
aclnnTurboQuantAttentionScore(workspace, workspaceSize, executor, stream);
```

Decode时executor仅加入`TqScoreCompute`。Prefill时先分配packed query临时Tensor，再顺序加入`TqQueryPack`和`TqScoreCompute`。调用方仍只调用一个public算子接口。

## 3.2 需求总体设计

主Kernel采用AIV/AIC协同：AIV负责分页读取、bit unpack、centroid查表、metadata缩放和BF16 high/low表示生成；AIC负责Cube矩阵乘并输出FP32 score。Prefill通过预排布query，使同一份query跨所有Key页复用。

### 3.2.1 Host侧设计

#### 3.2.1.1 分核策略

设：

```text
groupSize = QH / KH
groupTiles = ceil(groupSize / 4)
keyTiles = ceil(L / 128)
taskCount = B * KH * groupTiles * keyTiles
coreNum = max(1, min(platformAicNum, taskCount))
```

每个主计算任务处理一个batch、一个KV head、最多4个query head和一个128-Key页。各core按照`task=coreId; task<taskCount; task+=coreNum`循环领取任务。

Prefill Query Pack以32个query token为一块：

```text
packTaskCount = QH * ceil(T / 32)
packCoreNum = max(1, min(platformAivNum, packTaskCount))
```

每个AIV任务负责一个head的一个32行query块。

#### 3.2.1.2 数据分块和内存优化策略

Key方向固定以128条为一个Cube N tile，每个AIV子核解码64条Key。query方向：Decode的有效M不超过4；Prefill采用`M=128`。主计算统一构造成`K=512`的BF16 high/low表示：主量化项和QJL项分别拆成高低两部分，在Cube中一次累加得到FP32输出。

Prefill packed query Tensor形状及大小：

```text
packedShape = [QH, ceil(T/128)*128, 512]
packedBytes = QH * ceil(T/128)*128 * 512 * sizeof(BF16)
```

Score workspace：

```text
Decode query双缓冲 = coreNum * 2 * 4 * 512 * sizeof(BF16)
Key双缓冲          = coreNum * 2 * 128 * 512 * sizeof(BF16)
总workspace         = CANN库workspace + 上述两项
```

Prefill query使用executor管理的临时Tensor，因此Score节点不再申请Decode query双缓冲；Key双缓冲仍保留。

单个AIV Decoder按`W=64`个Key处理，`E=W*128`。LocalMemory分别分配packed原始字节、半字/整型解包区、offset、FP32工作区、BF16输出、metadata、query和centroid LUT。3-bit路径增加pair offset缓冲，以批量处理跨字节的3-bit编码。

#### 3.2.1.3 tilingKey规划策略

`TqQueryPack`和`TqScoreCompute`当前均使用`tilingKey=0`。Decode/Prefill、2/3/4-bit、BF16/FP16 metadata、任务规模及offset表通过tiling data和模板实例选择，不为每种组合分配独立tilingKey。

#### 3.2.1.4 数据检测

Public ACLNN接口检查query、cache、metadata、页表和输出的rank、dtype及shape关系，检查`mse_bits`、head整除关系和`max_kv_len`容量。`block_table`页号与`seq_lens`具体值位于Device内，Host不回读逐值校验，调用方须保证有效页号和序列长度合法。

### 3.2.2 Kernel侧设计

#### 3.2.2.1 Kernel侧实现描述

Query Pack Kernel仅在Prefill启用：读取32行`query_rotated/query_qjl`，根据Host生成的offset执行Gather，写成按head连续的512维BF16 high/low布局。

Score Kernel处理流程：

1. 根据任务编号解析batch、KV head、query group和Key页。
2. 通过`block_table`取得物理页，读取idx、QJL、norm和gamma。
3. AIV按位宽解包idx，查centroid并结合metadata生成主分支；同时解包QJL符号生成残差分支。
4. AIV把两路数据转换为Cube使用的K=512 BF16 high/low布局，写入双缓冲slot。
5. AIC等待slot就绪，执行`M×512`和`128×512`矩阵乘，FP32结果直接写入`attn_scores`。
6. 尾页只写有效Key，输出容量中超出`seq_lens`的位置写`-inf`。

AIV/AIC通过ready/consumed事件管理双缓冲生命周期，防止生产端覆盖AIC仍在读取的Key tile。

#### 3.2.2.2 Ascend C实现流程

Decode直接进入Score mixed Kernel；Prefill先由AIV以32行query为单位完成Query Pack，再将packed query传给Score节点。主Kernel中的两个AIV子任务各准备64条Key并写入BF16 high/low双缓冲，AIC等待tile就绪后执行Cube计算，将主量化项和QJL项累加为FP32 score，最后处理尾页和`-inf`屏蔽。

#### 3.2.2.3 Ascend C实现与标杆实现的差异点和原因

| 差异点 | Ascend C实现 | 原因 |
|---|---|---|
| Query处理 | Prefill先做一次Query Pack并跨页复用 | 避免每个Key页重复Gather query |
| Key恢复 | 不生成完整FP32 Key Tensor | 降低GM写回及中间显存占用 |
| 内积 | high/low BF16输入、Cube FP32累加 | 使用Cube吞吐并保持输出精度 |
| 页级复用 | 一个Key tile服务同组最多4个query head | 减少相同KV head的重复解码 |
| QJL计算 | 保留实数query_qjl与符号向量内积 | 保持算法语义，不能用两个bit-vector popcount替代 |
| 对外接口 | Prefill内部两节点、外部一个ACLNN算子 | 隐藏内部调度并统一调用方式 |

## 3.3 支持硬件

| 芯片版本 | 是否支持 |
|---|---|
| Atlas 800T A2（Ascend 910B系列） | 是 |

## 3.4 算子约束限制

- `D=J=128`、block size=128。
- query固定BF16；norm/gamma必须同为BF16或同为FP16。
- 仅支持标准MHA/GQA，不支持MLA和多序列Prefill。
- `max_kv_len`不得超过页表容量，调用方须保证有效物理页号合法。
- softmax和Value聚合不属于该算子。

# 四、特性交叉分析

| 特性交叉项 | 处理方式 |
|---|---|
| Decode/Prefill × Query Pack | Decode跳过Pack；Prefill在同一executor内先Pack后Score |
| `mse_bits × cache宽度` | `key_cache_idx`末维必须等于`16*mse_bits` |
| GQA比 × query group | 每个任务最多处理4个query head，较大GQA比拆成多个group tile |
| metadata dtype × 解码 | 模板选择BF16或FP16读取，内部统一形成FP32缩放结果 |
| `seq_lens × 尾页` | 只读取有效页，输出容量尾部填`-inf` |
| KV长度 × 分核 | Key页数进入taskCount，任务按页分配到AIC核 |

# 五、可维可测分析

## 5.1 精度标准/性能标准

| 验收标准 | 要求 |
|---|---|
| Score实现精度 | 有限输出最大绝对误差小于`1e-3`，无效位置屏蔽一致 |
| Score质量 | 相对Full-QK的relative L2小于0.12，softmax后KL mean和P95均小于0.01 |
| Decode性能 | 整体score计算时间不超过BF16 Paged Full-QK参考实现的1.2倍 |
| 存储 | 持久化Key cache压缩比不低于3倍 |

## 5.2 兼容性分析

算子保留Paged KV cache、`block_table`和`seq_lens`语义，输出为标准FP32 raw score，可接入后续softmax及Value聚合。调用方需提供与cache编码一致的query投影、位宽和metadata类型。
