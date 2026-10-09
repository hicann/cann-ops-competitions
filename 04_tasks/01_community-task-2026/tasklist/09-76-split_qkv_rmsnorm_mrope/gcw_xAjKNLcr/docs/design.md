# SplitQkvRmsnormMrope 算子设计文档

# 一、需求背景（required）

## 1.1 需求来源

通过社区任务完成面向Qwen3-VL注意力输入处理的融合算子贡献。算子采用Ascend C实现，通过ACLNN工程化接口调用。

## 1.2 背景介绍

### 1.2.1 SplitQkvRmsnormMrope算子实现优化

模型在attention计算前需要从融合QKV/Gate张量中拆分四路数据，并对Q/K执行RMSNorm、仿射和三轴MRoPE。分步实现会产生多次Kernel启动和中间Tensor。本算子在一次设备调用内完成拆分、归一化和旋转，V/Gate按原始BF16位表示直通。

该融合算子在CANN内置OPP中没有同名TBE实现。功能标杆采用包内PyTorch参考实现，性能标杆为功能等价的Triton实现。


### 1.2.2 SplitQkvRmsnormMrope标杆实现现状分析

#### 1.2.2.1 标杆算子支持的数据类型和数据格式

令`T`为token数、`QH`为query head数、`KH`为KV head数，head维度固定256，MRoPE维度固定64。

| 参数 | 方向 | 数据类型 | 数据格式 | 形状 |
|---|---|---|---|---|
| `qkv` | 输入 | BF16 | ND | `[T,2*QH*256+2*KH*256]` |
| `q_weight`、`k_weight` | 输入 | BF16 | ND | 各`[256]` |
| `cos_sin` | 输入 | BF16 | ND | `[3,T,64]` |
| `q_bias`、`k_bias` | 可选输入 | BF16 | ND | 各`[256]` |
| `q_output` | 输出 | BF16 | ND | `[T,QH*256]` |
| `k_output`、`v_output` | 输出 | BF16 | ND | 各`[T,KH*256]` |
| `gate_output` | 输出 | BF16 | ND | `[T,QH*256]`或`[T,0]` |

属性包括`num_q_heads`、`num_kv_heads`、`eps`、`interleaved`和`has_gate`。

#### 1.2.2.2 标杆算子实现描述

输入按`[Q0,Gate0,Q1,Gate1,...,K,V]`排列。标杆先拆分Q/K/V/Gate，再对每个Q/K head计算RMSNorm，乘以对应weight并加可选bias，最后对前64维应用三轴MRoPE。V和Gate不参与数值计算。

对单个Q/K head向量`x`：

```text
u = x / sqrt(sum(x*x)/256 + eps)
a = u * weight + bias               # 无bias时省略
out[0:32]  = a[0:32]*cos - a[32:64]*sin
out[32:64] = a[32:64]*cos + a[0:32]*sin
out[64:256] = a[64:256]
```

MRoPE根据`interleaved`属性选择三轴频率映射；section为`[11,11,10]`。

#### 1.2.2.3 标杆算子实现流程

标杆实现先从融合输入中拆分Q、K、V和Gate。Q/K按head执行RMSNorm、weight及可选bias，再对前64维执行三轴MRoPE并转换为BF16输出；V和Gate不参与数值计算，直接保持原始BF16数据输出。

# 二、需求分析（required）

## 2.1 外部组件依赖

| 组件 | 用途 |
|---|---|
| CANN 9.1.0及以上 | ACLNN工程构建、算子注册与运行时调用 |
| Ascend C | AI Core Kernel实现 |

PyTorch和NumPy仅用于测试数据及参考结果生成，不属于算子运行依赖。

## 2.2 内部适配模块

| 模块 | 作用 |
|---|---|
| OpDef与Shape/Dtype推导 | 定义可选输入、属性和四路输出形状 |
| Tiling模块 | 选择短序列/通用路径并计算任务核数 |
| Q/K计算模块 | RMSNorm、仿射、MRoPE及BF16写回 |
| V/Gate直通模块 | 保持原始BF16位表示完成输出搬运 |

## 2.3 需求模块设计

### 2.3.1 Ascend C算子原型

```text
SplitQkvRmsnormMrope(
    qkv, q_weight, k_weight, cos_sin,
    q_bias=None, k_bias=None,
    num_q_heads=16, num_kv_heads=4,
    eps=1e-6, interleaved=true, has_gate=true)
  -> q_output, k_output, v_output, gate_output
```

### 2.3.2 Ascend C算子相关约束

- 输入输出为连续ND BF16 Tensor，内部Q/K计算使用FP32。
- head维度固定256，MRoPE维度固定64。
- 推荐`num_q_heads=16`、`num_kv_heads=2/4`。
- 支持可选Q/K bias及interleaved/non-interleaved模式。
- 不支持广播和动态head维度。

# 三、需求详细设计（required）

## 3.1 调用方式

算子采用ACLNN两阶段调用。GetWorkspaceSize阶段完成输出推导和tiling，执行阶段在调用方stream中启动一个Ascend C Kernel。当前用户workspace为0。

## 3.2 需求总体设计

Kernel包含短序列路径和通用路径。短序列路径针对单token固定head配置按head并行；通用路径按token分核。Q/K统一在FP32工作区计算，V/Gate使用独立BF16缓冲直通。

### 3.2.1 Host侧设计

#### 3.2.1.1 分核策略

Host读取`T`、`QH`、`KH`、bias、Gate和interleaved属性。当满足以下条件时选择短序列路径：

```text
T=1, QH=16, KH∈{2,4}, 无Q/K bias, has_gate=true, interleaved=true
```

短路径任务数为：

```text
taskCount = (QH + KH) / 2
```

其余场景按token分核：

```text
taskCount = T
blockDim = max(1, min(taskCount, platformAivNum))
```

通用路径中，第`block`个核处理的token数和起点为：

```text
count = floor(T/blockDim) + (block < T%blockDim)
start = block*floor(T/blockDim) + min(block, T%blockDim)
```

该划分保证不同核的任务数最多相差1。

#### 3.2.1.2 数据分块和内存优化策略

通用路径每次处理一个token。输入行宽为：

```text
qSize = QH * 256
kvSize = KH * 256
rowWidth = qSize + hasGate*qSize + 2*kvSize
```

LocalMemory划分如下：

```text
bfBuf      = 5120 * 2 bytes
outBfBuf   = 5120 * 2 bytes
passBuf    = 5120 * 2 bytes
rowBuf     = 10240 * 2 bytes
fpBuf      = 13056 * 4 bytes
合计       = 103424 bytes
```

`bfBuf/outBfBuf`保存Q/K转换数据，`passBuf`保存V/Gate直通数据，`rowBuf`用于固定布局整行读取，`fpBuf`复用为输入、平方、归约、weight/bias、cos/sin和offset工作区。

短序列路径按head对处理，仅分配：

```text
BF16 LocalMemory = 1024 * 2 bytes
FP32 LocalMemory = 1664 * 4 bytes
合计 = 8704 bytes
```

#### 3.2.1.3 tilingKey规划策略

当前使用`tilingKey=0`。短序列/通用路径、bias、Gate和MRoPE模式通过tiling字段传入，在同一Kernel入口内选择，减少二进制分支数量。

#### 3.2.1.4 数据检测

OpDef限定所有输入输出为ND BF16，并声明Q/K bias为可选输入。Host检查`qkv`为二维Tensor并读取全部属性；输入末维、weight、bias和`cos_sin`形状须符合接口布局，由调用方保证。`eps`应为可用于RMSNorm的非负有限值。

### 3.2.2 Kernel侧设计

#### 3.2.2.1 Kernel侧实现描述

通用路径按token执行：

1. 读取当前token的三轴`cos_sin`，构造32组旋转位置对应的Gather offset。
2. 从交错输入中抽取Q和K并转FP32，同时将V/Gate搬入直通缓冲。
3. 对每个Q/K head计算平方和、RMS倒数、weight及可选bias。
4. Gather三轴cos/sin，对前64维执行旋转，其余192维保持仿射结果。
5. Q/K转BF16写回，V/Gate按原始BF16数据写回。

短序列路径读取完整输入行，按head对并行执行同一数学语义。MTE2、Vector和MTE3之间使用硬件事件同步，保证搬入、计算和搬出顺序。

#### 3.2.2.2 Ascend C实现流程

Ascend C实现先由MTE2从融合输入中抽取Q/K及V/Gate。Q/K转为FP32后依次完成平方归约、RMSNorm、weight/bias和三轴MRoPE，再转回BF16写出；V/Gate通过独立BF16缓冲直接写出。各阶段使用MTE2、Vector和MTE3事件维持数据依赖。

#### 3.2.2.3 Ascend C实现与标杆实现的差异点和原因

| 差异点 | Ascend C实现 | 原因 |
|---|---|---|
| 算子融合 | 拆分、RMSNorm、仿射、MRoPE在一个Kernel完成 | 减少Kernel启动和中间Tensor |
| 输入拆分 | 使用带步长DataCopy抽取交错Q/Gate | 避免先生成完整拆分Tensor |
| V/Gate | BF16原始位直通 | 避免不必要Cast并满足bitwise一致 |
| 短序列 | 单token按head对并行 | 提高token并行度不足时的核利用率 |
| 计算精度 | Q/K转FP32计算，最终转BF16 | 控制归约和旋转误差 |

## 3.3 支持硬件

| 芯片版本 | 是否支持 |
|---|---|
| Atlas 800T A2（Ascend 910B系列） | 是 |

## 3.4 算子约束限制

- head维度固定256，MRoPE维度固定64，section固定`[11,11,10]`。
- 输入输出仅支持连续ND BF16。
- 不支持广播。
- 输入融合布局、Gate属性和head属性必须保持一致。

# 四、特性交叉分析

| 特性交叉项 | 处理方式 |
|---|---|
| `has_gate × qkv宽度` | Host按Gate属性推导输入布局及`gate_output`宽度 |
| `bias × 路径` | bias存在时使用通用路径并加载对应参数 |
| `interleaved × MRoPE` | 根据模式构造不同的三轴Gather offset |
| `token数 × 分核` | 单token固定配置按head对并行，其余按token均分 |
| `KH × 输出shape` | K/V输出宽度按`KH*256`推导 |

# 五、可维可测分析

## 5.1 精度标准/性能标准

| 验收标准 | 要求 |
|---|---|
| Q/K精度 | 相对参考实现最大绝对误差小于`1e-3` |
| V/Gate精度 | 输出与输入BF16位表示一致 |
| 性能 | 相对功能等价参考实现达到2倍及以上加速 |

## 5.2 兼容性分析

算子输出保持标准Q/K/V/Gate二维布局，可接入后续attention计算。调用方需按接口提供融合输入、head属性和三轴位置编码；softmax及Value聚合不在本算子范围内。
