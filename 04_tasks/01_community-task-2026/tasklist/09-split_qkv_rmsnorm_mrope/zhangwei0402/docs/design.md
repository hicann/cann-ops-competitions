# SplitQkvRmsnormMrope 算子设计文档

> 文档状态：设计评审稿（2026-09-23）。目标硬件按任务书限定为 Atlas 800T A2。本文为方案设计，不宣称代码已实现、编译或通过真机精度与性能验证。

# 需求背景（required）

## 需求来源

本设计以本题任务书、随题 `op.json`、`case.json`、`golden.py` 为功能与验收基线；以当期社区任务设计模板为章节结构。vLLM-Ascend 的 `split_qkv_rmsnorm_mrope.py` 是 Triton 语义和性能参考，实际对拍时必须固定源码 commit、依赖和运行环境。通用 Ascend C 资料只用于设计方法，不替代本题验收要求。

| 来源 | 本设计使用范围 |
| --- | --- |
| 本题任务书 §1–§5 | 功能、硬件、精度、性能、工程形态和交付 |
| 随题 `op.json` | 算子名、输入/输出顺序、可选输入和属性默认值 |
| 随题 `case.json` | 七组官方自测场景及输入/输出形状 |
| 随题 `golden.py` | Q/Gate 排列、逐 head RMSNorm、三轴索引和无 Gate 行为 |
| [当期设计模板](https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md) | 章节组织；其中 Addcdiv 示例不作为本题事实 |
| [vLLM-Ascend Triton 实现](https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/ops/triton/linearnorm/split_qkv_rmsnorm_mrope.py) | 待冻结的正式精度/性能参考；当前 `main` 不是固定版本 |

随题四文件 SHA-256 依次为：任务书 `f732d407e4dc87680fe4be18297743e0fb88d376c6c3dce824030e940b008fc8`，`op.json` `212f844fd7f9bdf774fc3c0d7758c9672ee6125483325db1dc44860dfbee5ac6`，`case.json` `aad5a56813d74db2187c0f9cac72522b7718efde6ce24611cae3ae7f25abe466`，`golden.py` `85e0b2dbc7d100ed7837e31a0c6f6109a8f1c747ef8c24c73255d71169a33fe4`。

## 背景介绍

Qwen3-VL 的融合线性层一次产生 Q、Gate、K、V。现有 Triton 实现负责拆分、对 Q/K 做逐 head RMSNorm、对前 64 维做三轴 MRoPE，并输出 BF16 的 Q/K/V/Gate。本题要求以 Ascend C 建立 aclnn 工程化算子，利用 Vector Core 上的数据搬入—计算—搬出流水缩短整体时延。由于没有矩阵乘法，本设计选择单次启动的 Vector Core 自定义 kernel；不在 GM 中物化拆分后的中间 Q/K 或归一化结果。

任务书目标硬件为 Atlas 800T A2，CANN 9.1.0+，未细分到 910B3 或 910B4，也未要求两种卡同时验证。按用户最新指令，仅按任务书的 A2 范围执行；实际受测设备的 SKU、版本和环境需在测试时如实记录。网页简介与任务书硬件口径有差异时，采用任务书 A2。

# 需求分析（required）

## 需求描述

定义 `SplitQkvRmsnormMrope`，处理 BF16、ND 的 `qkv`、Q/K 权重、三轴 `cos_sin` 及可选 Q/K bias，返回四个 BF16、ND 输出。`num_q_heads`、`num_kv_heads` 可由属性指定；`head_size=256`、`rope_dim=64`、`mrope_section=(11,11,10)` 固定。`eps` 默认 `1e-6`，`interleaved` 和 `has_gate` 默认 `true`。支持有/无 Gate 和两种轴映射模式；可选 Q/K bias 分别判断是否存在。

接口以随题 `op.json` 的**位置顺序**为准，不按任务书介绍表重新排序：

```text
SplitQkvRmsnormMrope(
    qkv, q_weight, k_weight, cos_sin,
    q_bias?, k_bias?;
    num_q_heads=16, num_kv_heads=4, eps=1e-6,
    interleaved=true, has_gate=true
) -> (q_output, k_output, v_output, gate_output)
```

这是逻辑签名，不是已生成的 aclnn C/C++ 函数声明；最终函数名、可选输入表示和返回类型以目标仓工程生成接口为准。本题未指定 PyTorch 同名原生接口；本题指定的 `op.json` 与上游 Triton 函数是本算子的接口对照。

## 需求拆解

1. 依据 `has_gate` 确定行宽和 Q/Gate 偏移：有 Gate 时每个 Q head 后紧跟一个 Gate head；无 Gate 时 Q 连续，Gate 输出形状为 `[T,0]`，不访问 Gate 内存。
2. 每个 Q/K head 的 256 个 BF16 元素升为 FP32，按 head 独立计算平方均值、`rsqrt(mean+eps)`、权重和独立可选 bias；不得跨 head 或跨 token 归约。
3. 对归一化后的前 64 维按指定的 temporal/height/width 轴选择 cos/sin，并以相隔 32 的成对维度旋转；64–255 维维持归一化值。
4. Q/K 最终只在写回时转 BF16；V/Gate 在 BF16 域逐值直通，避免任何升精度再降精度造成位模式变化。
5. 七组官方场景逐一满足 Q/K 与**冻结的 Triton** 输出最大绝对误差严格 `<1e-3`，V/Gate 完全一致；每组时延严格 `<` 任务书该组 baseline 的一半。附件 Torch Golden 是独立诊断参考，不能替代 Triton 正式对拍。
6. 在符合任务书要求的 Atlas 800T A2 环境覆盖七组官方精度与性能场景，记录实际设备、构建产物、原始日志和报告；不把 B3/B4 双机型验证列为本题门槛。

# 详细设计（required）

## 算子分析

### 数学公式

记 `T=num_tokens`、`Hq=num_q_heads`、`Hkv=num_kv_heads`、`D=256`、`R=64`、`Q=Hq·D`、`K=Hkv·D`、`G=Q`（`has_gate=true`）或 `0`（否则），`W=Q+G+2K`。`qkv` 形状为 `[T,W]`。

每个 token `t` 的输入行内：

```text
q_base(t,h)    = t·W + h·(has_gate ? 2D : D)     , 0≤h<Hq
gate_base(t,h) = q_base(t,h) + D                   , has_gate=true
k_base(t,h)    = t·W + Q + G + h·D                 , 0≤h<Hkv
v_base(t,h)    = t·W + Q + G + K + h·D
```

Q/Gate 不是先放全部 Q、再放全部 Gate；此处按 head 交错。输出中 Q、Gate、K、V 各自按 `[T,head数·D]` 连续排列。

对 `x`（某一 Q 或 K head）、对应 `weight` 与可选 `bias`：

```text
x32[j] = FP32(x[j]), w32[j] = FP32(weight[j]), 0≤j<256
mean2  = (Σ_{j=0}^{255} x32[j]·x32[j]) / 256
r      = rsqrt(mean2 + eps)
n[j]   = (x32[j]·r)·w32[j] + (bias存在 ? FP32(bias[j]) : 0)
```

全部中间计算保持 FP32。实现时保持参考实现的运算顺序，并单独验证 `rsqrt` 的实现路径、融合乘加及 BF16 舍入差异；不能只因为公式等价便宣称满足 `1e-3`。

令 `d∈[0,31]`，轴选择 `a(d)`：

```text
non-interleaved: a=0 (d<11); a=1 (11≤d<22); a=2 (22≤d<32)
interleaved:     a=1 if d%3==1 and d≤33;
                 else a=2 if d%3==2 and d≤30;
                 else a=0
c = FP32(cos_sin[a(d),t,d])
s = FP32(cos_sin[a(d),t,d+32])
y[d]    = n[d]·c - n[d+32]·s
y[d+32] = n[d+32]·c + n[d]·s
y[j]    = n[j], 64≤j<256
```

`cos_sin` 的最后一维前 32 个是 cos、后 32 个是 sin。这里的 `interleaved` 仅改变三轴分配，**不**把旋转配对改为相邻元素。`y` 转 BF16 后写 Q/K。V/Gate 的输出直接复制源 BF16。若 `has_gate=false`，`gate_output` 是 `[T,0]`；任务书参数表默认有 Gate 的 `[T,Q]`，无 Gate 形状以随题 Golden 和上游实现为准。

### 支持数据类型

| 对象 | dtype / format | 约束 |
| --- | --- | --- |
| 六个输入（其中 Q/K bias 可选） | BF16 / ND | 四个必选输入顺序固定；可选输入各自独立 |
| 四个输出 | BF16 / ND | Q/K 转回 BF16；V/Gate BF16 直通 |
| Q/K 内部算术 | FP32 | 归约、缩放、权重、bias、RoPE 均按 FP32 路径 |

不扩展到 FP16/FP32 输入；数据类型泛化不是本题要求。`eps` 是 FP32 属性，要求非负且有限；实现需要防止输入中的非有限值引发未定义操作，但任务附件只生成 `[-1,1]` 范围的 QKV/角度数据。

### 支持形状

| 对象 | 形状 | 说明 |
| --- | --- | --- |
| `qkv` | `[T, Q+G+2K]` | `T≥1` 为本次已定义测试范围；两 head 数为正整数 |
| `q_weight`,`k_weight` | `[256]` | 每个 head 共享各自 256 元素权重 |
| `q_bias`,`k_bias` | `[256]`（可缺省） | Q/K 单独选择有无 |
| `cos_sin` | `[3,T,64]` | 三轴，token 维等于 QKV 的 `T` |
| `q_output`,`gate_output` | `[T,Q]`,`[T,G]` | 无 Gate 时第二维为 0 |
| `k_output`,`v_output` | `[T,K]`,`[T,K]` | K/V head 数相同 |

官方自测只包含 `Hq=16`、`Hkv∈{2,4}`、`T∈{1,16,128,2048,8192}` 的七种组合（不是 5×2 全笛卡尔积），均无 bias、`has_gate=true`、`interleaved=true`。其它合法属性组合是接口与补充回归范围，不冒充七组官方 case。

## 算子实现

### 实现方案

选 Ascend C Vector Core 单 kernel、单次算子启动。每个逻辑工作项处理 `(token, Q-head)` 或 `(token, K-head)`：Q 工作项产生该 head 的 Q，并在有 Gate 时拷贝对应 Gate；K 工作项产生 K 并拷贝对应 V。四输出的写入区域互不重叠；无跨 core 归约、无需中间 GM 缓冲。这样 `T=1` 也至少能暴露 `Hq+Hkv` 个独立计算任务，而非把整行都串行压在一个 core 上；是否足以达到性能目标仍须实测。

#### Host 侧设计

`op_host` 原型与 shape 推导遵从 `op.json`，`op_api` 提供目标仓规范要求的 aclnn 接口与调用示例。Host 在访问维度和计算字节数前验证：输入/输出数及顺序、BF16/ND、rank、权重与 bias 长度、`cos_sin=[3,T,64]`、`Hq/Hkv>0`、行宽公式、`eps` 的有效性以及尺寸乘法溢出。`has_gate=false` 时推导 `[T,0]` 输出；具体零元素张量的分配/传参按目标仓 aclnn 工程约定实现并测试。是否接收非连续物理存储由目标仓适配层规范确定；本 kernel 只按连续 ND 物理布局寻址，绝不把非连续张量误当连续访问。

Host 查询**当前**芯片可用 Vector Core 数和 UB 大小，计算 `N=T·(Hq+Hkv)`、`C=min(N, availableAivCores)`，并把实际 `C` 设置为 kernel 启动核数。第 `c` 个 core 负责半开区间 `[floor(N·c/C), floor(N·(c+1)/C))`，整数运算采用足够宽的类型并做溢出校验，最大工作量差 1。若后续 profiling 证明长序列连续块造成某类工作项倾斜，可在不改变语义的情况下调整映射，但所有分支都须跑完七组和补充用例。

TilingData 的逻辑字段与来源如下（实际代码字段类型/注册宏在目标仓版本上核对）：

| 字段 | Host 计算/读取 | Kernel 用途 |
| --- | --- | --- |
| `tokens, qHeads, kvHeads` | 输入 shape 和属性 | 工作项解码、输出行距 |
| `qSize=256·qHeads, kvSize=256·kvHeads, gateSize` | 校验后计算 | Q/K/V/Gate 输入输出偏移 |
| `inputRowElems=qSize+gateSize+2·kvSize` | 校验后计算 | token 行首寻址 |
| `taskCount=tokens·(qHeads+kvHeads), coreCount=C` | 资源查询与整除 | 核间切分、尾工作项保护 |
| `eps, interleaved, hasGate, hasQBias, hasKBias` | 属性和可选输入存在性 | 计算分支；不把两个 bias 绑成同一个开关 |
| `headTile=256, pipeDepth` | UB 预算与硬件能力 | 核内整 head 归约、队列深度 |

五个属性按原型声明顺序读取，索引不可与参数表输入顺序混淆。TilingData 只保存 kernel 实际需要的标量；`mrope_section` 是本题固定常量，不增加未声明的运行时属性。系统 workspace 按最终使用的 Ascend C API/工程接口查询；本方案不需要**自定义的 GM 中间缓冲**，但不据此写死系统 workspace 为 0。

#### Kernel 侧设计

逻辑工作项 `i` 解为 `t=floor(i/(Hq+Hkv))`、`h=i%(Hq+Hkv)`。`h<Hq` 走 Q/Gate 路径，否则以 `kh=h-Hq` 走 K/V 路径。每项完整处理一个 256 元素 head，归约不跨 UB tile；偏移使用上文公式，并在访问前由 Host 保证总长度、可选指针状态合法。若用两级 tiling 描述，本方案的 Block 级是工作项区间，UB 级是固定 256 元素 head；`headTile=256`，没有跨 tile 累加误差。若某机型 UB 预算不足，应退化为分段归约加 FP32 累加并重做精度与性能验证，不能无声改变公式。

下面是**语义级 Ascend C API 伪代码**；具体 API 重载、scratch 大小和同步要求在目标 CANN 9.1.0+ SDK 上核实后编码，不把它当作已编译代码：

```text
for i in coreTaskRange:
    type, t, h = decode(i)
    src = Q或K的qkv输入偏移; dst = Q或K输出偏移
    CopyIn: DataCopy(src[0:256] -> x_bf16)
            DataCopy(相应weight -> w_bf16)
            if 相应bias存在: DataCopy(bias -> b_bf16)
            取token t三轴cos_sin的各64个BF16值到UB
            Q且hasGate: DataCopy(Gate源BF16 -> pass_bf16)
            K:          DataCopy(V源BF16 -> pass_bf16)
    Compute: Cast(x_bf16 -> x_fp32); Cast(w_bf16 -> w_fp32)
             Mul(square, x_fp32, x_fp32, 256)
             ReduceSum(sum, square, 256); sum *= 1/256
             sum += eps; r = rsqrt(sum)  # 初版核对与Triton的1/sqrt舍入
             Muls(norm, x_fp32, r, 256)
             Mul(norm, norm, w_fp32, 256)
             if 相应bias存在: Cast(b_bf16 -> b_fp32); Add(norm, norm, b_fp32, 256)
             选a(d), Cast对应cos/sin -> FP32, d=0..31
             保留norm[0:64]原值后按分半配对公式写入旋转结果
             norm[64:256]不变; Cast(norm -> out_bf16, BF16)
    CopyOut: DataCopy(out_bf16 -> Q或K输出)
             如存在直通目标: DataCopy(pass_bf16 -> Gate或V输出)
```

归约标量至少按 SDK 对齐要求分配；双缓冲时遵守 CopyIn/Compute/CopyOut 的队列拥有权及同步，禁止输入尚在 DMA 时读、输出尚未完成时复用。`DataCopy`、`Cast`、`ReduceSum`、`Rsqrt` 等具体签名、BF16 转换舍入模式和 32B 对齐边界在编码前按已安装 SDK 文档确认。Q/K 旋转时应保留 0–63 原值以避免原地写入污染另一个半区。

#### UB 分配与流水

初版单项以一个完整 head 为计算粒度，按 32 字节对齐计。下表为**设计预算**，不是已测 UB 占用；`R_reduce` 表示所选归约 API 的临时空间，须在编译/运行前补上 SDK 实际要求。

| UB 区域 | 每份字节 | 份数（双缓冲方案） | 合计字节 | 作用 |
| --- | ---: | ---: | ---: | --- |
| `xInBF16` | 512 | 2 | 1024 | Q/K head 输入 |
| `passInBF16` | 512 | 2 | 1024 | Gate/V 位保真输入；无 Gate 的 Q 项不使用 |
| `axisCosSinBF16` | 384 | 2 | 768 | 当前 token 的三轴 `3×64×2B` |
| `qkOutBF16` | 512 | 2 | 1024 | Q/K 输出 |
| `passOutBF16` | 512 | 2 | 1024 | Gate/V 直通输出；可与搬运链路优化 |
| `weightBF16` | 512 | 1 | 512 | Q 或 K 权重 |
| `biasBF16` | 512 | 1 | 512 | 可选 Q 或 K bias，保守预留 |
| `xFP32`,`squareFP32`,`normFP32` | 1024 | 各 1 | 3072 | 升精度、归约输入、最终归一化向量 |
| `pairScratchFP32`,`selectedCosSinFP32` | 256 | 各 1 | 512 | 保留两半区与选轴值；具体别名/大小编码时核对 |
| `reduceScalar` | 32 | 1 | 32 | 标量与对齐 |
| 设计上界小计 |  |  | **9504** | 另加 `R_reduce` 与管线/SDK 开销 |

启用 `pipeDepth=2` 的条件为 `9504+R_reduce+SDK额外开销 ≤ 当前设备可用UB`，且双缓冲确有多项工作可重叠。`T=1` 等少任务场景可选 `pipeDepth=1` 以减少队列/同步开销；此选择需与单核多项任务数一起由 profiling 验证。这里不套用通用 elementwise 的 bufferCoefficient，也不把未核实的 UB 容量写成目标 A2 设备的固定值。Weight、bias 与三轴表的跨工作项复用属于后续可测优化，必须在保证等价和尾部安全的前提下实施。

### 性能分析与优化顺序

单 kernel 避免将 Split、RMSNorm、MRoPE 和直通拆成多次启动或 GM 中间结果；每个 head 的输入与输出恰好 512B，官方两种行宽也都是 512B 整数倍，便于按 head 定位。主要风险是小 `T` 时只有 18/20 个 Q/K head 工作项，Vector Core 可能未满；大 `T` 时重复装载权重和三轴表、BF16↔FP32 转换及 RoPE 选轴可能成为瓶颈。不能仅凭启动数或理论搬运量承诺 2×。

优化顺序：先单缓冲、全精度且四输出正确；再测短序列的启动/核占用和长序列的搬运/计算比；然后按证据选择双缓冲、权重/角度复用、工作项分配或专门的短 `T` 分支。每次优化均回归七组及补充模式，并保留候选源码 commit、实际 A2 设备的构建身份和原始时延样本。若为了吞吐改变数学运算顺序，首先重新验证 Q/K 的严格误差门槛。

## 支持硬件

| 硬件/软件 | 设计范围 | 当前证据 |
| --- | --- | --- |
| Atlas 800T A2 | 任务书指定；CANN 9.1.0+，未指定 B3/B4 双卡 | 机器已准备（用户反馈）；实际 SKU、CANN、编译器、驱动、固件、构建与运行未核 |

Host 应根据实际 A2 设备查询 AIV 核数和 UB 容量，不依赖未经确认的 SKU 常量。报告记录实际设备、工具链、受测代码版本与结果。

## 算子约束限制

- 固定 `D=256`、`R=64`、`mrope_section=(11,11,10)`；只有 BF16/ND 与本题定义的连续物理布局进入本 kernel。
- `Hq/Hkv` 必须为正；`qkv` 宽度严格等于 `Q+G+2K`；`cos_sin` 的 token 维必须与 `qkv` 相等；非法 shape/属性在 Host 侧拒绝。
- `has_gate=false` 的零宽输出和 Q/K 单独 bias 是功能分支，不因官方七组仅测有 Gate、无 bias 而删掉。
- 七组官方输入均为 Q/K 权重 1，不能由此推断其它权重值已验证；需补充非全 1 权重、bias 组合、无 Gate、non-interleaved、极值和非法形状用例。
- `T=0`、非连续物理布局、未声明的 dtype/广播不在当前已证明范围；如仓库要求支持，需补充对应接口方案与测试，不可默默按当前偏移实现。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 判定方式 | 标准来源 |
| --- | --- | --- |
| Q/K 精度 | 各输出逐元素 `max(abs(candidate − frozen Triton)) < 1e-3`，严格小于；检查 BF16 与形状 | 任务书 §3.2 |
| V/Gate 直通 | 与原始输入相应切片逐值/位模式一致；无 Gate 时 `[T,0]` | 任务书 §3.2、`golden.py` |
| 性能 | **每一组**实测 NPU 时延 `<` 下表该组 baseline/2，不用平均达标抵消失败组 | 任务书 §3.3/§3.5 |
| 目标硬件 | 在符合任务书要求的 A2 环境执行七组精度与七组性能；如实登记实际 SKU | 任务书 §3.1；用户最新范围指令 |
| 内存 | 任务书写“不涉及”；材料中按模板如实说明，不伪造“通过”数据 | 任务书 §3.4/§4 |

| 官方 case | `T,Hq,Hkv` | QKV 宽 | Triton baseline (μs) | 严格上限 `<` (μs) |
| --- | --- | ---: | ---: | ---: |
| `decode_t1_q16_kv4` | 1,16,4 | 10240 | 636.240 | 318.1200 |
| `small_t16_q16_kv4` | 16,16,4 | 10240 | 643.246 | 321.6230 |
| `medium_t128_q16_kv4` | 128,16,4 | 10240 | 678.923 | 339.4615 |
| `prefill_t2048_q16_kv4` | 2048,16,4 | 10240 | 641.700 | 320.8500 |
| `prefill_t8192_q16_kv4` | 8192,16,4 | 10240 | 669.499 | 334.7495 |
| `decode_t1_q16_kv2` | 1,16,2 | 9216 | 640.793 | 320.3965 |
| `prefill_t2048_q16_kv2` | 2048,16,2 | 9216 | 669.931 | 334.9655 |

`case.json` 中七组均无 bias，`interleaved=true`、`has_gate=true`；Q/Gate 为 `[T,4096]`，K/V 对 kv4 为 `[T,1024]`、对 kv2 为 `[T,512]`。官方共 7 个场景，不按 B3/B4 翻倍计数。用例输入由测试工具按 `case.json` 生成时，要保存输入文件/生成器哈希与随机种子；其绝对 `case_path` 和 Golden 路径需在受测环境适配。正式性能计时协议（Triton 版本、原始基线机型、计时边界、预热/采样统计）当前尚未由任务材料给出，不能把任一自选测法冒充官方协议；可先用明确记录的方法做本地优化，再按官方答复复核可比性。

验证矩阵：在实际 A2 环境先核硬件、工具链与可调用性，跑七组附件 Golden 诊断、七组冻结 Triton 正式对拍及七组性能，再跑有/无 Gate、两种 RoPE、Q/K bias 四种存在性组合、非全 1 权重和错误形状补充测试。测试 README 写明依赖版本、数据准备、构建、调用、精度/性能命令、判定公式及日志位置。任一输出错误先定位拆分偏移、归约/舍入、轴映射、配对维度；只在实际执行后报告结果。交付材料按任务书和单套七文件模板准备，记录实际受测环境，不追加机型目录。

## 兼容性分析

该算子为新 Ascend C aclnn 工程，目标是在本题接口和七组场景中替代相应 Triton 路径，并不自动替换已有 Python 调用。接口兼容性以 `op.json` 的位置、属性默认值、四输出顺序为准；数值兼容性以冻结 Triton 的 Q/K 比对和 BF16 V/Gate 完全直通判定。上游 Triton 当前 `main` 中是否有 bias 的开关由 Q bias 决定，不能据此弱化本题两个 bias 独立可选的接口；混合 bias 组合应以任务附件 Golden 作功能回归，并在正式对拍脚本中注明参考实现支持边界。CANN 9.1.0+ 的实际 API 支持情况在所用 A2 设备上构建及真机测试时确认。
