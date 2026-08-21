# SageAttention2 算子设计文档

# 需求背景

## 需求来源

CANN 社区任务：在 Atlas 950PR 上基于 AscendC 与 CATLASS 实现 SageAttention2 的 INT8 QK^T + FP8 PV
路径，交付 PyTorch 与 aclnn 两层接口，合入 `cann/ops-transformer` 的
`experimental/attention/sageattention2`。

## 背景介绍

### SageAttention2 算法与应用场景

SageAttention2（ICML 2025，arXiv:2411.10958）是一种推理期量化注意力算法。它不改变注意力的数学定义，
只把注意力内部的两个矩阵乘替换为低比特运算：`QK^T` 用 INT8，`PV` 用 FP8，从而利用低精度 Cube 算力。
算法不需要重新训练或微调权重，可直接替换推理框架中的注意力实现。

适用场景是序列长、注意力占比高的推理负载：视频与图像生成（CogVideoX、HunyuanVideo、Flux）、
DiT 类模型、大模型长上下文推理。

注意力的两个矩阵乘具备不同的数值特性，这是该算法可以量化到 INT8/FP8 的前提：

| 矩阵乘 | 数值特性 | 量化选择 |
| --- | --- | --- |
| `QK^T` | Q/K 分布相对集中，K 在通道维存在离群值 | 先沿序列维减去 K 的均值，再做 INT8 分块量化 |
| `PV` | `P = exp(S - rowmax)` 上界已知 | FP8 E4M3，按固定偏置缩放后直接转换，无需标定 |

### 开源实现现状分析

对标实现为 `github.com/thu-ml/SageAttention`（main 分支 / PyPI `sageattention==2.2.0`）中的
`sageattn_qk_int8_pv_fp8_cuda`。其数据流为：

1. `smooth_k=True` 时沿序列维计算 K 的均值 `km` 并在量化阶段减去；
2. Q/K 按 `qk_quant_gran` 指定的粒度量化为 INT8，V 按通道量化为 FP8；
3. 调用 CUDA 内核完成注意力主循环，KV 步长为 64，V 的 scale 在内核内融合；
4. `return_lse=True` 时对内核返回的 LSE 做底数换算并叠加 smooth_k 的修正项。

量化在独立于注意力主循环的算子中完成，Q/K/V 的量化结果与 scale 经全局内存传入主循环。

对标实现的数值细节（本设计逐项对齐）：

| 项 | 对标实现取值 |
| --- | --- |
| K 均值 `km` | 沿序列维在 FP32 中累加求均值，结果以输入 dtype 存储 |
| INT8 scale 与舍入 | `per_thread`：`scale = absmax / 127 + 1e-7`，`trunc(x/scale + 0.5·sign)`（round-half-away-from-zero）；`per_warp`：`scale = absmax / 127`，round-to-nearest-even 并饱和到 INT8 范围 |
| `K - km` 的形成 | `per_thread`：差值在输入 dtype 中形成后转 FP32；`per_warp`：K 与 km 分别转 FP32 后相减 |
| V 的 FP8 scale | 每通道 `scale = absmax / scale_max`；`pv_accum_dtype="fp32+fp16"` 时 `scale_max = 2.25`，其余为 448.0；`v_f8 = rne_e4m3(v · scale_max / absmax)` |
| P 的 FP8 偏置 | 行最大值处的 P 取 `2^8.807`（≈447.9，E4M3 最大值以下），对应源码常量 `S_FP8_OFFSET = 8.807` |
| V 的均值 `vm`（`smooth_v` 生效时） | 序列长度补齐到 16 的倍数、补齐位置为 0：`vm = Σ V / ceil16(Sk)`，`absmax` 在补齐后的序列上取 |
| softmax 分母 | 对**未量化**的 FP32 概率求行和 |
| PV 累加 | 见“两级累加”节 |
| LSE | 内核内以 2 为底，对外除以 `1.44269504`；`smooth_k=True` 时加 `(Q · km^T) · sm_scale`，`Q · km^T` 以输入 dtype 矩阵乘（FP32 累加，结果为输入 dtype）后转 FP32 |

### 算子功能分析

输入：Q、K、V，dtype 为 FP16 或 BF16。
输出：注意力结果 O，dtype 与 Q 相同；`return_lse=True` 时附加 LSE。
支持 GQA、非因果的不等长 Q/KV、因果掩码、head_dim 补齐。
仅前向，不含反向。

# 需求分析

## 需求描述

实现 INT8 QK^T + FP8 PV 单一路径的量化注意力算子，Kernel 基于 AscendC 与 CATLASS 联合开发，
对外提供 PyTorch 与 aclnn 两层接口，功能与数值语义对齐开源 `sageattn_qk_int8_pv_fp8_cuda`。

## 需求拆解

1. PyTorch 层提供 `sageattn` 与 `sageattn_qk_int8_pv_fp8_asc`，参数名、类型、默认值与返回值逐项
   对齐开源对应接口；
2. aclnn 层提供统一的 FP8 PV 入口，shape、LSE 与错误语义与 PyTorch 层一致，且不依赖 PyTorch 前处理；
3. 支持 `qk_quant_gran ∈ {per_warp, per_thread}` 两种量化粒度，语义对齐开源；
4. 支持 `pv_accum_dtype ∈ {fp32, fp32+fp32, fp32+fp16}`，默认 `fp32+fp16`，三种模式分别对齐同配置 GPU 结果；
5. 支持 `smooth_k`，`smooth_v` 在两级累加配置下按开源行为告警并忽略；
6. 支持 `return_lse`，数值语义对齐开源；
7. 支持 head_dim 补齐与输出裁剪、GQA、因果掩码、HND/NHD 两种布局；
8. 非法参数按约定报错；
9. 注意力矩阵不得整体物化。

## 接口定义

### PyTorch 接口

```python
def sageattn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tensor_layout: str = "HND",
    is_causal: bool = False,
    sm_scale: Optional[float] = None,
    return_lse: bool = False,
    **kwargs: Any,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]

def sageattn_qk_int8_pv_fp8_asc(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tensor_layout: str = "HND",
    is_causal: bool = False,
    qk_quant_gran: str = "per_thread",
    sm_scale: Optional[float] = None,
    pv_accum_dtype: str = "fp32+fp16",
    smooth_k: bool = True,
    smooth_v: bool = False,
    return_lse: bool = False,
    **kwargs: Any,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
```

参数语义（两函数共有参数含义相同）：

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `q` | `torch.Tensor` | — | Query。HND `[B, Hq, Sq, D]`，NHD `[B, Sq, Hq, D]`；FP16 或 BF16；末维 `stride(-1) == 1` |
| `k` | `torch.Tensor` | — | Key。HND `[B, Hkv, Sk, D]`，NHD `[B, Sk, Hkv, D]`；dtype、device、D 与 `q` 相同；`Hq % Hkv == 0` |
| `v` | `torch.Tensor` | — | Value。shape、dtype、device 约束与 `k` 相同 |
| `tensor_layout` | `str` | `"HND"` | 取 `"HND"` 或 `"NHD"` |
| `is_causal` | `bool` | `False` | 因果掩码，仅 `Sq == Sk` 时合法 |
| `qk_quant_gran` | `str` | `"per_thread"` | 取 `"per_warp"` 或 `"per_thread"` |
| `sm_scale` | `Optional[float]` | `None` | 为 `None` 时取 `1/sqrt(D)`，`D` 为补齐前维度 |
| `pv_accum_dtype` | `str` | `"fp32+fp16"` | 取 `"fp32"`、`"fp32+fp32"` 或 `"fp32+fp16"` |
| `smooth_k` | `bool` | `True` | 沿序列维对 K 减均值 |
| `smooth_v` | `bool` | `False` | 沿序列维对 V 减均值；`pv_accum_dtype` 非 `"fp32"` 时发出 `UserWarning` 并按 `False` 处理 |
| `return_lse` | `bool` | `False` | 是否返回 LSE |
| `**kwargs` | `Any` | — | 见下 |

`**kwargs` 的处理：

| 函数 | 识别的键 | 其它键 |
| --- | --- | --- |
| `sageattn` | `pv_accum_dtype`、`qk_quant_gran`、`smooth_k`、`smooth_v`：透传给 `sageattn_qk_int8_pv_fp8_asc`；`attn_mask`：非 `None` 时报错 | 忽略 |
| `sageattn_qk_int8_pv_fp8_asc` | `attn_mask`：非 `None` 时报错 | 忽略 |

`sageattn` 在 NPU 上固定分发到 `sageattn_qk_int8_pv_fp8_asc`。`pv_accum_dtype` 的取值优先级为
kwargs > 环境变量 `SAGEATTN_PV_ACCUM_DTYPE` > 默认值 `"fp32+fp16"`，只有当前生效的那一个来源被解析
与校验，合法性由 `sageattn_qk_int8_pv_fp8_asc` 统一判定。

返回值：

| 返回项 | 说明 |
| --- | --- |
| `o` | 与 `q` 同 dtype、同布局、同 shape，最后一维为原始 `D` |
| `lse` | 仅 `return_lse=True` 时返回 `(o, lse)`；FP32，shape `[B, Hq, Sq]`，与布局无关；数值为 `QK^T · sm_scale` 各行的自然对数 logsumexp |

错误行为：以下条件在调用 aclnn 前校验，不满足时抛出 `ValueError`：
dtype 不是 FP16/BF16、三者 dtype 或 device 不一致、末维不连续、`D = 0` 或 `D > 128`、
`Hq % Hkv != 0`、`is_causal=True` 且 `Sq != Sk`、`tensor_layout`/`qk_quant_gran`/`pv_accum_dtype`
取值非法、`attn_mask` 非 `None`。

### aclnn 接口

两段式接口，量化、head_dim 补齐与输出裁剪均在算子内部完成，调用方传入原始 FP16/BF16 张量：

```c
aclnnStatus aclnnSageAttention2GetWorkspaceSize(
    const aclTensor *query,
    const aclTensor *key,
    const aclTensor *value,
    double           scaleValue,
    bool             isCausal,
    char            *tensorLayout,
    char            *qkQuantGran,
    char            *pvAccumDtype,
    bool             smoothK,
    bool             smoothV,
    bool             returnLse,
    const aclTensor *attentionOut,
    const aclTensor *softmaxLseOut,
    uint64_t        *workspaceSize,
    aclOpExecutor  **executor);

aclnnStatus aclnnSageAttention2(
    void             *workspace,
    uint64_t          workspaceSize,
    aclOpExecutor    *executor,
    const aclrtStream stream);
```

| 参数 | 类别 | 类型 / dtype | 形状 | 取值与可选性 | 语义与约束 |
| --- | --- | --- | --- | --- | --- |
| `query` | 输入 | `aclTensor*`，FLOAT16 或 BFLOAT16 | HND `[B, Hq, Sq, D]`；NHD `[B, Sq, Hq, D]` | 必选 | ND 格式；末维连续，其余维允许任意步长；`0 < D ≤ 128` |
| `key` | 输入 | `aclTensor*`，与 `query` 同 dtype | HND `[B, Hkv, Sk, D]`；NHD `[B, Sk, Hkv, D]` | 必选 | `Hq % Hkv == 0`；`B`、`D` 与 `query` 相同 |
| `value` | 输入 | `aclTensor*`，与 `query` 同 dtype | 与 `key` 相同 | 必选 | 同 `key` |
| `scaleValue` | 属性 | `double` | — | 必选 | softmax 缩放系数；PyTorch 层在 `sm_scale=None` 时填 `1/sqrt(D)` |
| `isCausal` | 属性 | `bool` | — | 必选 | `true` 时要求 `Sq == Sk` |
| `tensorLayout` | 属性 | `char*` | — | `"HND"` 或 `"NHD"` | 决定 `query`/`key`/`value`/`attentionOut` 的维度含义 |
| `qkQuantGran` | 属性 | `char*` | — | `"per_warp"` 或 `"per_thread"` | Q/K 的 INT8 量化粒度 |
| `pvAccumDtype` | 属性 | `char*` | — | `"fp32"`、`"fp32+fp32"` 或 `"fp32+fp16"` | PV 累加模式，见“两级累加”节 |
| `smoothK` | 属性 | `bool` | — | 必选 | K 减序列维均值 |
| `smoothV` | 属性 | `bool` | — | 必选 | `pvAccumDtype` 非 `"fp32"` 时按 `false` 处理，记录 warning 级算子日志，返回码不变 |
| `returnLse` | 属性 | `bool` | — | 必选 | 是否写 `softmaxLseOut` |
| `attentionOut` | 输出 | `aclTensor*`，与 `query` 同 dtype | 与 `query` 同 shape，末维为原始 `D` | 必选，调用方分配 | ND 格式，末维连续；算子写满全部元素 |
| `softmaxLseOut` | 输出 | `aclTensor*`，FLOAT32 | `[B, Hq, Sq]` | `returnLse=true` 时必选，否则可为空指针且被忽略 | 自然对数 logsumexp，含 `smoothK` 修正 |
| `workspaceSize` | 输出 | `uint64_t*` | — | 必选 | 第二段所需 device 内存字节数 |
| `executor` | 输出 | `aclOpExecutor**` | — | 必选 | 一次性执行器 |

执行语义：第一段仅在 Host 完成校验、tiling 与 workspace 计算，不下发任务；第二段在 `stream` 上
异步下发，调用方同步流后结果可用，`workspace` 与输出张量在任务完成前必须保持有效；`executor` 只能
被第二段使用一次。

返回码：

| 返回码 | 触发条件 |
| --- | --- |
| `ACLNN_SUCCESS` | 参数合法，任务成功下发 |
| `ACLNN_ERR_PARAM_NULLPTR` | `query`/`key`/`value`/`attentionOut`/`workspaceSize`/`executor` 或三个字符串属性为空指针；`returnLse=true` 且 `softmaxLseOut` 为空指针 |
| `ACLNN_ERR_PARAM_INVALID` | dtype 不是 FLOAT16/BFLOAT16 或三者不一致；rank 不为 4；`B`、`D` 不一致；`Hq % Hkv != 0`；`D = 0` 或 `D > 128`；末维不连续；`isCausal=true` 且 `Sq != Sk`；字符串属性取值非法；`attentionOut`/`softmaxLseOut` 的 shape 或 dtype 与约定不符 |
| `ACLNN_ERR_RUNTIME_ERROR` | 第二段任务下发失败 |

PyTorch 层与 aclnn 层的校验条件一一对应；PyTorch 层的输出张量按原始 `D` 分配，直接作为
`attentionOut` 传入。两条路径的一致性边界是：相同输入下输出值、shape、LSE 逐位相同，报错触发条件
相同（PyTorch 抛 `ValueError`，aclnn 返回对应错误码）；`smooth_v` 被忽略时 PyTorch 层向调用方发出
`UserWarning`，aclnn 层只写算子日志，这是两层唯一的行为差异。

# 详细设计

## 算子分析

### 数学公式

未量化的注意力定义：

```
O = softmax(Q · K^T · sm_scale) · V
```

本算子的量化计算链路。记 `c = 2^8.807`（≈447.9），KV 按 64 键为一块（=`BLKK`，与对标实现的 CTA_K 一致），`j` 为块序号：

```
K'      = K - km,  km = mean(K, dim=seq)                        (smooth_k=True)
V'      = V - vm,  vm = sum(V, dim=seq) / ceil16(Sk)             (smooth_v=True 且 pv_accum_dtype="fp32")
Q_i8    = quant_int8(Q,  gran)         → q_scale
K_i8    = quant_int8(K', gran)         → k_scale
V_f8    = quant_fp8 (V', per-channel)  → v_scale

S_j     = (Q_i8 · K_i8[j]^T) · q_scale[m] · k_scale[n] · sm_scale      (FP32；掩码位置置为极大负数)
          掩码位置：is_causal=True 时列号 n > 行号 m 的位置，以及序列尾部越界的列
m_j     = max(m_{j-1}, rowmax(S_j))                                   m_0 = -∞（用极大负数表示）
alpha_j = exp(m_{j-1} - m_j)
P_j     = exp(S_j - m_j) · c                                          ∈ (0, c]，FP32
P_f8    = rne_e4m3(P_j)
l_j     = l_{j-1} · alpha_j + rowsum(P_j)                              l_0 = 0，对未量化、含倍率 c 的 P 求和
Acc_j   = Acc_{j-1} · alpha_j + flush(P_f8 · V_f8)                     Acc_0 = 0，flush 见“两级累加”节
O       = Acc_last / l_last · v_scale        (+ vm，当 smooth_v 生效)
lse     = ln(l_last) - ln c + m_last         (+ (Q · km^T) · sm_scale，当 smooth_k=True)
```

不变量：

1. `m` 是不含偏置的行最大值，倍率 `c` 只作用一次（实现上以指数偏置 `ln c` 并入 exp 的参数，
   `exp(S - m + ln c)`）；`P_j ≤ c < 448`，FP8 转换不溢出。行最大值
   处的概率映射到 E4M3 的最大量级，可表示的相对动态范围从无偏置时的 `2^-9` 扩展到 `2^-17.8`，小概率
   不再大面积下溢。对标实现把偏置并入以 2 为底的最大值（`m2 = rowmax · log2e - 8.807`），
   `2^(S · log2e - m2) = exp(S - rowmax) · 2^8.807`，与本设计的 `P` 相同。
2. `Acc` 与 `l` 在同一步用同一 `alpha_j` 重标定，`O = Acc / l` 与 `m` 的取值无关（至 FP32 舍入）；
   分子与分母都含倍率 `c`，相消后输出归一化不受偏置影响。
3. 分母对未量化的 `P` 求和，与对标实现一致；`l` 与 LSE 因此不受 FP8 舍入影响。
4. `l` 含倍率 `c`，LSE 中减去 `ln c` 一次，`lse = ln Σ exp(S)`；对标实现内部以 2 为底、对外除以
   `1.44269504`，二者相差 `log2(e)/1.44269504 - 1 ≈ 6e-10`，可忽略。本设计内部即自然底，对外不做换算。
5. 本方案按对标实现的 CTA_K 每 64 键更新一次行最大值 `m_j`，使 `P_j` 的分组与 FP8 舍入位置和
   GPU 同配置路径一致。64 键是本方案的实现选择，不是接口约束；若改用其他内部调度，仍须保持公开
   语义并通过同配置 L-A/L-B 判据。位级结果不要求一致：对标实现使用 `fmaf` 与 `exp2`，本设计使用
   FP32 乘法链与自然指数（`exp(t - (m - ln c))`），归约次序与指数近似不同，落在 E4M3 舍入边界附近
   的元素可能得到不同的 FP8 值；最终输出按 L-A/L-B 判据验证。

`smooth_v` 仅在 `pv_accum_dtype="fp32"` 下生效：V 沿序列维减去 `vm` 后再量化，输出阶段加回 `vm`。
`vm` 与该模式下的每通道 `absmax` 按对标实现的规则计算：序列长度补齐到 16 的倍数、补齐位置按 0
参与——`vm = Σ V / ceil16(Sk)`，`absmax = max(|max(V ∪ {0}) - vm|, |min(V ∪ {0}) - vm|)`；
`Sk % 16 == 0` 时退化为普通均值与 `absmax(V - vm)`。`vm` 在减法与加回中相消，该规则只影响 FP8 舍入
位置，采用它是为了与同配置 GPU 结果逐元素对齐。`pv_accum_dtype` 为 `"fp32+fp32"` 或 `"fp32+fp16"` 时，按对标实现的行为告警并按 `False` 处理。

`smooth_k=True` 时的 LSE 修正项 `Q · km^T`：`km` 以输入 dtype 存储，乘加在 FP32 中完成，结果先舍入到
输入 dtype 再转 FP32 参与相加，与对标实现的计算路径一致。GQA 下 `km` 按 `Hq/Hkv` 广播到各 Q head。

### 量化粒度定义

`qk_quant_gran` 的两种取值对应不同的分组方式。分块常量取开源默认值：
`BLKQ=128`、`WARPQ=32`、`BLKK=64`、`WARPK=64`。

| 粒度 | Q 分组 | Q scale 数量 | K 分组 | K scale 数量 |
| --- | --- | --- | --- | --- |
| `per_warp` | 每 `WARPQ`=32 行连续一组 | `⌈Sq/BLKQ⌉ · 4` | 每 `BLKK`=64 行连续一组 | `⌈Sk/BLKK⌉` |
| `per_thread` | 每 32 行内按行号模 8 分 8 组，每组 4 行 | `⌈Sq/BLKQ⌉ · 4 · 8` | 每 64 行内分 4 组，第 `t` 组为行号模 8 余 `2t`、`2t+1` 的 16 行 | `⌈Sk/BLKK⌉ · 4` |

`per_thread` 的组内行号不连续，源于对标实现中矩阵乘指令的寄存器分片布局。本设计保留该分组关系：
量化时先按行计算 `absmax`，再按上表的行号映射合并为组 scale。逐行 `absmax` 是两种粒度共同需要的
中间量，保留非连续分组不引入额外的数据搬运。scale 与舍入公式按粒度分别对齐对标实现（见“背景介绍”
数值细节表）。

V 采用 per-channel 量化，每个 `head_dim` 通道一个 scale，沿序列维规约得到。`absmax` 为 0 的通道
（含补齐列）scale 记为 1、量化值为 0。

### 支持数据类型

| 张量 | 数据类型 |
| --- | --- |
| Q / K / V 输入 | FP16、BF16，三者一致 |
| Q / K 量化后 | INT8 |
| V、P 量化后 | FP8 E4M3 |
| QK^T 累加 | INT32 |
| S、softmax 中间量、`m`、`l` | FP32 |
| PV 段和（64 键） | L0C FP32；`fp32+fp16` 下经 Fixpipe 以 FP16 搬出（一次 RNE 舍入） |
| PV 跨段累加缓冲 | FP32 |
| O 输出 | 与 Q 相同 |
| LSE 输出 | FP32 |

### 支持形状与布局

| 项 | 约束 |
| --- | --- |
| 布局 | `HND` 为 `[B, H, S, D]`；`NHD` 为 `[B, S, H, D]` |
| head_dim | 原始 `D ∈ (0, 128]`；`D < 64` 补齐到 64，`D = 64` 不补齐，`64 < D < 128` 补齐到 128，`D = 128` 不补齐；`D > 128` 报错 |
| 末维连续 | `stride(-1) == 1`，其余维允许任意步长 |
| GQA | `Hq % Hkv == 0` |
| 序列长度 | 非因果下允许 `Sq ≠ Sk`；因果要求 `Sq == Sk` |
| 输出 | 最后一维为原始 `D` |

## 算子实现

### 模块边界：AscendC 与 CATLASS 分工

| 模块 | 实现方式 | 职责 |
| --- | --- | --- |
| Quant | AscendC Vector | Q/K 的 INT8 分块量化、V 的 per-channel FP8 量化、K/V 的序列维均值、head_dim 逻辑补零、LSE 修正项 |
| QK^T | CATLASS Block Mmad | INT8 矩阵乘，L0C 累加为 INT32 |
| Softmax epilogue | AscendC Vector | S 的反量化、掩码、online softmax、P 的 FP8 量化、行和 |
| PV | CATLASS Block Mmad | FP8 矩阵乘，L0C 累加为 FP32 |
| RescaleO epilogue | AscendC Vector | 段和转 FP32、Acc 重标定与累加、归一化、v_scale 还原、输出类型转换与裁剪写回 |
| Tiling 与调度 | Host + Scalar | 分核切分、循环索引、地址计算、AIC/AIV 同步 |

### 所用 CATLASS 组件

| 组件 | 用途 |
| --- | --- |
| `Gemm::Block::BlockMmadTla` | QK^T 与 PV 两个矩阵乘的 Block 层封装 |
| `Gemm::MmadFAIQK` / `Gemm::MmadFAIPV` | 注意力场景的 Mmad 调度策略 |
| `Gemm::Tile::PackedTileCopyTlaToUB` | GM/L1/L0 与 UB 之间的分片搬运 |
| `Gemm::Tile::TileMmadTla` | Tile 层矩阵乘指令封装 |
| `Epilogue::EpilogueAscend950FASoftmax` / `EpilogueAscend950FARescaleO` | online softmax 与 O 更新两个 Epilogue 的分发策略与 UB 缓冲组织；本算子的两个 Epilogue 类沿用其接口与 AIC/AIV 同步约定，Vector 计算体（反量化、掩码、exp、乘 `c`、行和、FP8 转换、段和舍入、重标定）按本设计的定义用 AscendC 微 API 实现 |
| `Arch::CrossCoreSetFlag` / `CrossCoreWaitFlag`、`Arch::Resource` | AIC/AIV 跨核同步与片上资源分配 |
| `Tile::CopyUb2L1Tla` | 量化后的 P 由 UB 直送 L1，避免经全局内存回流 |
| `tla` 布局与 `layout::RowMajor` / `ColumnMajor` / `zN` | 各级存储的分片布局描述 |

### Host 侧设计

**分核策略**：以 `batch × head × ⌈Sq/BLKQ⌉` 为并行单元展开，按 AI Core 数量贪心切分，
使各核承担的 KV 循环总量尽量均衡。因果场景下各 Q 块的有效 KV 长度不同，切分按有效工作量而非
块数计算。

**Tiling 数据**：传递 batch、head 数、GQA 比例、Sq、Sk、原始 head_dim、补齐后 head_dim、
`sm_scale`、量化粒度标识、累加模式标识、smooth_k/smooth_v/return_lse 标志、各 scale 张量的地址与
步长、掩码类型与多核切分索引。

**Workspace 布局**：分 stage 物化时保存补齐后的 `Q_i8`、`K_i8`、`V_f8`（均按
`[B, H, S, Dpad]` head 优先存放，`Dpad∈{64,128}`；`K_i8`、`V_f8`
的序列维补齐到 128 的倍数，补齐行为零、对应 `k_scale` 为 1，使主循环每个 KV 分片都是完整的 128 键）、
对应 scale 张量、`km`（`smooth_k=True` 时）、`vm` 与 V 的 per-channel scale、LSE 修正项
（`return_lse=True` 且 `smooth_k=True` 时）、K/V 序列维规约的分块部分和。Host 侧只计算 tiling
参数与 workspace 划分，不参与量化数值计算。

**布局处理**：`HND` 与 `NHD` 的差别是序列维与 head 维的次序。Host 侧按布局计算两个方向的步长
并下发，Kernel 按步长寻址，不做显式转置。

**head_dim 补齐与裁剪**：补齐在量化阶段以逻辑补零实现——量化 kernel 只读取原始 `D` 列，向 workspace
或片上生产缓冲写出补齐后的 `Dpad` 列，补齐列填 INT8/FP8 零值；补零列对 `QK^T`、`PV`、`km`、
`Q · km^T` 的贡献均为零。主循环按 `Dpad=64/128` 选择实例；输出在 RescaleO 写回时只写前 `D` 列。
输入与输出都不物化补齐后的 FP16/BF16 张量。性能基准用例均为 `D = 128`，不受 64 列实例影响。

**kernel 实例规划**：按输入 dtype（2 种）× `Dpad`（64/128）× `fp32+fp16` 段和舍入有无（2 种）
实例化注意力 kernel；因果掩码、量化粒度、`smooth_k`、`smooth_v`、`return_lse` 为运行时标志。
量化 kernel 同样覆盖两种 `Dpad`。

### Kernel 侧设计

算子由量化与注意力两个逻辑阶段构成。Q/K/V 的量化对每个元素只执行一次，而注意力主循环会多次读取
K/V。两阶段可采用全量 workspace、按 head/tile 的生产者-消费者流水或融合 kernel；选择标准是同一
公开语义下的片上容量、同步合法性与性能。分 stage 物化不是任务契约，任何融合方案也不得让同一输入
被重复量化或物化全量 attention 矩阵。

**量化阶段**：

1. `smooth_k=True` 时沿序列维在 FP32 中规约得到 `km`，舍入到输入 dtype；K 在量化前按粒度对应的路径
   减去 `km`；
2. 按行计算 `absmax`，按量化粒度定义的行号映射合并为组 `absmax`，按粒度对应的公式得到 scale，用
   scale 量化输入，量化值按粒度对应的舍入方式舍入并饱和为 INT8；
3. V 沿序列维规约得到每通道 `absmax`（`smooth_v` 生效时先按 `ceil16(Sk)` 规则求 `vm` 并减去），按
   `pv_accum_dtype` 选取 `scale_max`（`fp32+fp16` 取 2.25，其余取 448.0），量化为 FP8 E4M3；
4. `return_lse=True` 且 `smooth_k=True` 时计算修正项 `Q · km^T`，按 GQA 比例广播 `km`。

**注意力主循环**：沿 batch-head、Q 序列、KV 序列三层循环组织。Q 分片 128 行（=`BLKQ`），KV 分片
128 键（一次 `Q_i8 · K_i8^T` Mmad），分片内按两个 64 键块（=`BLKK`）依次做 online softmax 与 PV：
`per_warp` 下每 64 键块共用一个 `k_scale`，`per_thread` 下列 `n` 使用其所在 64 键块内第 `(n mod 8) / 2`
组的 `k_scale`；Q 侧按行号取 `q_scale`。`k_scale` 按列向量在每个分片搬入 UB，反量化时按列相乘。
128 键分片采用 CATLASS Ascend950 FA 的 AIC/AIV 软件流水与 S/P/段和多缓冲，两个 64 键块共享一次
S 搬出与一次 `k_scale` 搬入。该分片是本方案的 tiling 选择，不是公开接口约束。单个分片的处理顺序为：

1. Cube 执行 `Q_i8 · K_i8^T`（128 键），L0C 得到 INT32 的 S 分片，经 Fixpipe 按行分给两个 Vector 核；
2. Vector 将 S 转为 FP32 并乘以 `q_scale[m] · k_scale[n] · sm_scale`，应用因果掩码与越界掩码
   （置极大负数），分别求两个 64 键块的行最大值；
3. 依次用两个块的行最大值更新 `m`，得到各块的 `alpha`；
4. 对每个块计算 `P = exp(t - (m - ln c))`（= `exp(t - m) · c`，倍率并入指数偏置），对 FP32 值累加行和
   更新 `l`，再转换为 FP8 E4M3，两块的 P 分别写入 L1；
5. Cube 对每个块执行 `P_f8 · V_f8`（K=64），L0C 得到该块的 FP32 段和；
6. 段和按累加模式搬出到 UB（`fp32+fp16` 下 Fixpipe 直接以 FP16 搬出，段和 ≤ `c · 2.25 · 64 < 65504`），
   Vector 对两个块依次以 `alpha` 重标定 `Acc` 并累加段和。

S 与 P 始终以分片形式存在于 L0C 与 UB，不写回全局内存。单步的数据流与计算单元分工如下：

```
GM ─MTE2─> L1 ─> L0A/L0B ─Cube(INT8)─> L0C(INT32)
                                          │ Fixpipe，不做格式转换
                                          v
                                        UB(INT32)
                                          │ Vector: 转 FP32、乘 q/k scale 与 sm_scale、
                                          │         掩码、行最大值 → m/alpha、exp、
                                          │         乘 c、行和 → l、转 FP8 E4M3
                                          v
                                       UB(FP8 P) ─CopyUb2L1─> L1
                                                               │
                                          L0A/L0B <────────────┘
                                             │ Cube(FP8)
                                             v
                                          L0C(FP32 段和) ─Fixpipe(FP32，或 F322F16)─> UB
                                                                │ Vector: Acc = Acc·alpha + 段和、
                                                                │         归一化、乘 v_scale、
                                                                │         转输出类型、裁剪写回
                                                                v
                                                               GM
```

**S 的 INT32 通路**：Ascend950 的 Fixpipe 在 L0C 搬出路径上不支持 INT32 到 FP32 的格式转换，
INT32 源只能落到 INT32、FP16、BF16、INT8 或 UINT8。将 S 转换为 FP16 会损失有效位并使
反量化 scale 受量化模式约束，因此 S 以 INT32 原样搬出到 UB，由 Vector 完成类型转换与反量化。
INT32 与 FP32 位宽相同，转换在同一块 UB 上原位完成，不额外占用空间。

**两级累加**：Ascend950 的 Mmad 对 FP8 E4M3 输入只提供 FP32 结果类型（L0C），不存在 FP16 累加器；
Fixpipe 支持把 L0C 的 FP32 以 `F322F16`（round-to-nearest-even）搬出到 UB。`F322F16` 与双目的地址
（`SPLIT_M`）搬出不能同时使用，因此 `fp32+fp16` 下每个 64 键块的段和用两条单目的地址 Fixpipe 分别
搬到两个 Vector 核（各自的行）。三种 `pv_accum_dtype` 按下表实现，段长统一为 64 键，重标定统一在
每块累加段和之前进行：

| `pv_accum_dtype` | 对标实现（GPU）行为 | 本设计（NPU） | 数值关系 |
| --- | --- | --- | --- |
| `fp32` | FP8 MMA 累加器（有效 22 位）贯穿全部 KV，无中间缓冲 | L0C FP32 累加 64 键，段和以 FP32 搬出，累加到 UB 的 FP32 缓冲 | NPU 全程 FP32，无 22 位限制 |
| `fp32+fp32` | MMA 累加器（有效 22 位）累加 64 键，归约到 FP32 寄存器缓冲 | 与 `fp32` 相同数据流 | 与 NPU `fp32` 数值等价，仅 `smooth_v` 处理不同 |
| `fp32+fp16` | 每 64 键做两次 K=32 的 FP8 MMA，累加器为 FP16，累加器内部的舍入次序与精度未公开，源码只能确认两次 MMA 各以 FP16 输出 | L0C FP32 累加 64 键，段和经 Fixpipe 以 FP16 搬出（一次 RNE 舍入），Vector 转 FP32 后累加到 FP32 缓冲 | NPU 近似路径：段和只舍入一次；有效性按 TC-02 的同配置 GPU L-A/L-B 判据验证 |

`fp32+fp16` 在 NPU 上不复现 GPU 的 FP16 累加器：Cube 内无 FP16 累加，且 GPU MMA 累加器的内部舍入
次序与精度未公开。本设计取“段和一次 FP16 舍入”作为该模式的定义。该定义与 GPU 结果的差异是本设计
的已知风险，必须由 TC-02 中 `fp32+fp16` 对同配置 GPU 的 L-A/L-B 判据证伪；不预设其达标。若 TC-02
不通过，回退方案为：PV 按 K=32 分两次 Mmad，第一段以 FP16 搬出，第二段以 FP32 搬出，Vector 中按
`fp16(fp32(r1) + r2)` 合并后再累加到 FP32 缓冲，代价是每块多一次 L0C 搬出与一遍 Vector 合并；该回退
同样是近似，其有效性仍以 TC-02 判定。V 的 `scale_max = 2.25` 保证 `c · 2.25 · 64 < 65504`，段和不
超出 FP16 范围，与对标实现的取值来源一致。`fp32` 与 `fp32+fp32` 在实现上可在 `alpha` 全为 1 的块中
省略重标定并让相邻段和在 L0C 内连续累加，这只改变 FP32 加法次序，不改变上述契约；`fp32+fp16` 下段和
必须逐块搬出。

**LSE**：`return_lse=True` 时，主循环结束后由 `l` 与 `m` 得到 `ln(l) - ln c + m`，`smooth_k=True` 时叠加
量化阶段算得的 `(Q · km^T) · sm_scale`。

### PyTorch 与 aclnn 适配层

PyTorch 层负责参数校验、`sm_scale` 缺省值计算、`smooth_v` 在两级累加下的告警、`sageattn` 的分发与
kwargs/环境变量解析、按原始 `D` 分配输出，随后调用 aclnn 两段式接口。head_dim 补齐、裁剪、量化与
LSE 修正全部在 aclnn 内部完成，PyTorch 层不做张量预处理。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas 950PR | √ |

依赖 Ascend950 的 INT8 与 FP8 矩阵乘能力，不支持不具备原生 FP8 的型号。

## 算子约束限制

1. 仅前向，不提供反向；
2. 仅 INT8 QK^T + FP8 PV 路径，不提供 FP16 PV、变长序列与 Triton 路径；
3. `attn_mask` 不支持，传入非 `None` 时报错；
4. 输入 dtype 仅 FP16/BF16，Q/K/V 须同 dtype、同 device；
5. `head_dim = 0` 或 `head_dim > 128` 报错；
6. `qk_quant_gran`、`pv_accum_dtype`、`tensor_layout` 取值超出约定集合时报错；
7. `is_causal=True` 且 `Sq != Sk` 时报错；
8. `Hq % Hkv != 0` 时报错；
9. Q/K/V 末维不连续时报错。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | 双层策略，见下表 | 《生态算子开源精度标准》与任务书 |
| 性能标准 | PyTorch 与 aclnn 两条路径相对同仓 FIA 均达到 Speedup ≥ 1.6× | 任务书 |

精度双层策略：

| 层级 | 标杆 | 判据 |
| --- | --- | --- |
| L-A（主） | GPU `sageattn_qk_int8_pv_fp8_cuda`，相同 `pv_accum_dtype` 与 `qk_quant_gran` | 混合容差：FP16 `rtol = atol = 2^-9`，BF16 `2^-6`；`matched_ratio ≥ 0.99`；`max_abs_error ≤ 1e-1`（FP16）／`1e-0`（BF16）或 32×ULP |
| L-B（辅） | FP32 SDPA | NPU 与 GPU 对同一真值的误差比：`max ≤ 2`、`mean ≤ 1.2`、`RMSE ≤ 1.2` |

`per_warp` 与 `per_thread`、以及三种 `pv_accum_dtype` 分别对标同配置的 GPU 结果，不跨配置比较。
`return_lse` 的 LSE 采用 FP32 容差或相对 GPU LSE 的混合容差。

性能门禁：

| 项 | 约定 |
| --- | --- |
| 环境 | Atlas 950PR，CANN 9.2.0-beta.1，同机同 shape 同 dtype |
| 基线 | `cann/ops-transformer` 的 FusedInferAttentionScore；PyTorch 路径用 `torch_npu.npu_fused_infer_attention_score`（同一报告内固定 v1 或 `_v2`，不混用），aclnn 路径用 `aclnnFusedInferAttentionScoreV5` |
| 本算子 | PyTorch 路径 `sageattn_qk_int8_pv_fp8_asc`（P-06 经 `sageattn` 分发），aclnn 路径 `aclnnSageAttention2`；均为 `pv_accum_dtype="fp32+fp16"`、`qk_quant_gran="per_thread"`、`smooth_k=True` |
| FIA 配置 | `input_layout="BNSD"`（与 `HND` 同形，不引入布局转换）、`num_heads`/`num_key_value_heads`/`scale` 与本算子一致，因果按 FIA 文档的合法因果模式配置并在报告中记录 |
| 计时口径 | 排除首次编译，充分预热后取稳定耗时（PyTorch 用 `torch.npu.Event`，aclnn 用等价 device 计时），同步后统计，两路径分列 |
| 判据 | P-01～P-07 每条用例在两条路径上各自 Speedup ≥ 1.6×，且两路径的全用例几何平均各自 ≥ 1.6×；任一项不满足即不达标 |

## 测试用例设计

功能与精度用例按参数维度组合：

| 因子 | 取值 |
| --- | --- |
| dtype | FP16、BF16 |
| `tensor_layout` | HND、NHD |
| `qk_quant_gran` | per_warp、per_thread |
| `pv_accum_dtype` | fp32、fp32+fp32、fp32+fp16 |
| `smooth_k` / `smooth_v` | True/False 组合，覆盖两级累加下的忽略行为 |
| `return_lse` | True、False |
| head_dim | 16、32、48、63、64、65、96、127、128，覆盖补齐边界 |
| GQA | `Hq/Hkv` 为 1、4、8 |
| 序列 | `Sq = Sk` 与 `Sq ≠ Sk`；含非 64 倍数与非 16 倍数长度；短序列至 8K/16K 长序列 |
| 因果 | True、False |
| 接口 | PyTorch、aclnn 直调，两者对同一输入输出逐位一致 |

重点用例：

| 编号 | 场景 |
| --- | --- |
| TC-01 | `sageattn` 分发至 FP8 PV 路径，覆盖 FP16/BF16 与 HND/NHD |
| TC-02 | `per_thread`/`per_warp` × 三种 `pv_accum_dtype`，各自对标同配置 GPU |
| TC-03 | `smooth_k`/`smooth_v` 开关与两级累加下的告警忽略行为 |
| TC-04 | `return_lse=True`，LSE 形状为 `[B, Hq, Sq]` 且数值对齐同路径 GPU 结果；aclnn `returnLse` 两种取值 |
| TC-05 | head_dim 补齐：48/63→64，64 不补齐，96/127→128，输出裁剪正确；aclnn 直调与 PyTorch 路径结果一致 |
| TC-06 | GQA 且 `Sq ≠ Sk` |
| TC-07 | 因果掩码 |
| TC-08 | 异常用例，见下表 |
| TC-09 | 长序列 `S ≥ 8K/16K` |
| TC-10 | 替换 `F.scaled_dot_product_attention` 的冒烟用例 |
| TC-11 | 配置入口：`sageattn` 的 `pv_accum_dtype` 取值按 kwargs > `SAGEATTN_PV_ACCUM_DTYPE` > 默认值生效并可由输出与同配置 `sageattn_qk_int8_pv_fp8_asc` 逐位一致证明；当前生效来源取非法值时抛 `ValueError`（kwargs 非法；或 kwargs 缺省而环境变量非法）；kwargs 合法且环境变量非法时 kwargs 胜出、不报错；未识别 kwargs 被忽略；`attn_mask=None` 合法；两级累加下 `smooth_v=True` 触发一次 `UserWarning` 且结果与 `smooth_v=False` 逐位一致 |
| TC-12 | `smooth_v=True` 且 `pv_accum_dtype="fp32"`，`Sk % 16 != 0`（如 `Sk = 1000`），对标同配置 GPU 结果 |

TC-08 异常用例：

| 输入 | 期望 |
| --- | --- |
| `head_dim = 0`、`head_dim = 256` | PyTorch `ValueError`；aclnn `ACLNN_ERR_PARAM_INVALID` |
| 末维不连续（如 `q.transpose(-1, -2)`） | 同上 |
| `tensor_layout="BNSD"`、`qk_quant_gran="per_block"`、`pv_accum_dtype="fp16"` | 同上 |
| `is_causal=True` 且 `Sq ≠ Sk` | 同上 |
| `Hq = 7, Hkv = 4` | 同上 |
| Q/K/V dtype 不一致、dtype 为 FP32、device 不一致 | 同上 |
| `attn_mask` 非 `None` | PyTorch `ValueError` |
| aclnn 必选张量或字符串属性为空指针；`returnLse=true` 且 `softmaxLseOut` 为空 | `ACLNN_ERR_PARAM_NULLPTR` |
| aclnn `attentionOut`/`softmaxLseOut` shape 或 dtype 不符 | `ACLNN_ERR_PARAM_INVALID` |

性能用例与判定：

| 编号 | shape（HND，`[B, Hq, S, D]`） | dtype | 因果 | GQA | 分档 | 判定 |
| --- | --- | --- | --- | --- | --- | --- |
| P-01 | `[1, 32, 4096, 128]` | FP16 | 否 | 否 | S1 | 两路径各自 ≥ 1.6× |
| P-02 | `[1, 32, 8192, 128]` | FP16 | 否 | 否 | S1 | 两路径各自 ≥ 1.6× |
| P-03 | `[1, 32, 8192, 128]` | FP16 | 是 | 否 | S2 | 两路径各自 ≥ 1.6× |
| P-04 | `[2, 32, 8192, 128]` | FP16 | 否 | 否 | S1 | 两路径各自 ≥ 1.6× |
| P-05 | `[1, 32, 4096, 128]`，`Hkv = 8` | FP16 | 否 | 是 | S3 | 两路径各自 ≥ 1.6× |
| P-06 | `[1, 32, 4096, 128]` | BF16 | 否 | 否 | S4 | 两路径各自 ≥ 1.6× |
| P-07 | `[1, 32, 16384, 128]` | FP16 | 否 | 否 | S5 | 两路径各自 ≥ 1.6× |
| 汇总 | — | — | — | — | — | PyTorch 路径与 aclnn 路径的几何平均各自 ≥ 1.6× |

每条用例分别在 PyTorch 与 aclnn 两条路径上与 FIA 对比，报告分列 `T_FIA`、`T_本算子`、Speedup 与
所用 FIA 接口名。测试入口与复现步骤随测试代码提供，说明环境依赖、数据生成方式与判据。

## 兼容性分析

新增算子，位于 `experimental/attention/sageattention2` 独立目录，不修改既有算子实现与公共接口，
不涉及兼容性变更。与同仓其他 SageAttention 系列算子按子目录隔离，避免符号与包名冲突。
