# SageAttention2 算子设计文档

# 需求背景

## 需求来源

本设计对应 CANN 社区 SageAttention2 算子开发任务。目标是在 Atlas 950PR 上，使用
AscendC 与 CATLASS 实现 SageAttention2 的 INT8 `QK^T` + FP8 `PV` 前向路径，并向上
提供 PyTorch 和 aclnn 两层接口。最终代码计划合入 `cann/ops-transformer` 仓库的
`experimental/attention/sageattention2` 目录。

任务验收环境为 CANN 9.2.0-beta.1。功能与精度对标开源
`thu-ml/SageAttention` 的 `sageattn_qk_int8_pv_fp8_cuda`，性能基线为同仓
`FusedInferAttentionScore`（FIA）。

## 背景介绍

### 算法与应用场景

标准注意力为：

```text
O = softmax(QK^T * scale) V
```

SageAttention2 保持注意力的数学语义不变，通过低比特量化降低两个矩阵乘的计算和访存
成本：Q、K 量化为 INT8，V 与 softmax 概率量化为 FP8 E4M3。K 的序列维离群值通过
`smooth_k` 缓解，PV 通过分段累加和高精度缓冲限制低精度累积误差。

该路径面向长序列推理，典型场景包括视频/图像生成、DiT、LLM 长上下文推理和需要
`return_lse` 的 Ring Attention 组件。本任务仅交付前向 INT8 QK + FP8 PV，不包含 FP16
PV、varlen、反向和 Triton 路径。

### 开源实现现状

对标实现为 SageAttention 2.2.0 的 FP8 PV 路径。其主要过程如下：

1. 可选地沿序列维计算 K 的均值并执行 `K - mean(K)`；
2. 按 `per_thread` 或 `per_warp` 语义对 Q、K 分组并量化为 INT8；
3. 按通道统计 V 的 scale，将 V 量化为 FP8 E4M3；
4. 以分块方式完成 INT8 QK、online softmax 和 FP8 PV；
5. 按 `pv_accum_dtype` 选择跨段累加方式；
6. 在 `return_lse=True` 时返回自然对数域 LSE，并叠加 `smooth_k` 修正项。

AscendC 没有与开源算法同名的 `per_thread`/`per_warp` 量化开关，因此这两种粒度必须由
Vector 侧显式实现，并以同配置 GPU 输出验证语义。CATLASS 用于组织 Cube 矩阵乘和
Block/Tile 流水，AscendC 用于量化、softmax、低精度转换、归约和输出后处理。

### 功能边界

| 项 | 本设计范围 |
| --- | --- |
| 输入 | Q、K、V，FP16 或 BF16，同 dtype、同 NPU device |
| 输出 | 与 Q 同 shape、同 dtype 的 O；可选 FP32 LSE |
| 布局 | HND、NHD |
| 注意力 | 非因果与因果；因果仅允许 `Sq == Sk` |
| Head 关系 | MHA 与 GQA，要求 `Hq % Hkv == 0` |
| 量化 | Q/K INT8，V/P FP8 E4M3 |
| 累加 | `fp32`、`fp32+fp32`、`fp32+fp16` |
| 不交付 | FP16 PV、varlen、反向、CUDA/Triton 同名实现 |

# 需求分析

## 需求描述

实现一个自包含的 SageAttention2 NPU 路径。PyTorch API 对齐开源参数与返回值语义，
aclnn API 完成相同参数校验、内部量化、head_dim 补齐、注意力计算和输出裁剪。Kernel
必须使用 AscendC + CATLASS，不能调用 FIA、其他整算子或 Host/CPU 路径生成输出。

## 需求拆解

1. 提供 `sageattn` 自动分发入口和 `sageattn_qk_int8_pv_fp8_asc` 显式入口；
2. 提供必选的 aclnn 两段式接口；
3. 实现 `per_thread`、`per_warp` 两种 INT8 量化粒度；
4. 实现三种 `pv_accum_dtype`，默认覆盖 `fp32+fp16`；
5. 对齐 `smooth_k`、`smooth_v` 和 `return_lse` 行为；
6. 支持 HND/NHD、MHA/GQA、非因果不等长序列和因果等长序列；
7. 按原始 head_dim 选择 64 或 128 的内部计算维度，输出裁剪回原始 D；
8. attention 矩阵仅以片上分块存在，不在 GM 中整体物化；
9. PyTorch 与 aclnn 两条路径分别完成精度和 FIA 性能验收。

## PyTorch 接口

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

| 参数 | 约束与语义 |
| --- | --- |
| `q` | HND `[B,Hq,Sq,D]` 或 NHD `[B,Sq,Hq,D]`；FP16/BF16；末维连续 |
| `k`/`v` | HND `[B,Hkv,Sk,D]` 或 NHD `[B,Sk,Hkv,D]`；与 q 同 dtype/device |
| `tensor_layout` | `"HND"` 或 `"NHD"` |
| `is_causal` | 为 True 时要求 `Sq == Sk` |
| `qk_quant_gran` | `"per_thread"` 或 `"per_warp"` |
| `sm_scale` | None 时取 `1/sqrt(D)`，D 为补齐前维度 |
| `pv_accum_dtype` | `"fp32"`、`"fp32+fp32"` 或 `"fp32+fp16"` |
| `smooth_k` | 沿 K 的序列维减均值，并在 LSE 中加入修正项 |
| `smooth_v` | 仅 `fp32` 模式生效；另两种模式告警并按 False 处理 |
| `return_lse` | False 返回 O；True 返回 `(O, LSE)` |
| `attn_mask` | 仅接受 None；非 None 明确报错 |

`sageattn` 在 NPU 上固定分发到 `sageattn_qk_int8_pv_fp8_asc`。累加模式的覆盖优先级为
kwargs、环境变量 `SAGEATTN_PV_ACCUM_DTYPE`、默认值。只校验最终生效的值，未识别 kwargs
按开源兼容行为忽略。

输出 O 保持输入布局和原始 D；LSE 为 FP32 `[B,Hq,Sq]`，表示
`QK^T * sm_scale` 每行的自然对数 logsumexp。

## aclnn 接口

aclnn 接口采用 GetWorkspaceSize + Execute 两段式设计。接口名在实现阶段随仓库注册规范
最终确认，本设计使用 `aclnnSageAttention2` 表示统一 FP8 PV 入口：

```c
aclnnStatus aclnnSageAttention2GetWorkspaceSize(
    const aclTensor *query,
    const aclTensor *key,
    const aclTensor *value,
    double scaleValue,
    bool isCausal,
    const char *tensorLayout,
    const char *qkQuantGran,
    const char *pvAccumDtype,
    bool smoothK,
    bool smoothV,
    bool returnLse,
    const aclTensor *attentionOut,
    const aclTensor *softmaxLseOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

aclnnStatus aclnnSageAttention2(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    const aclrtStream stream);
```

第一段完成参数检查、shape 推导、tiling 选择和 workspace 规划，不执行数值计算；第二段在
传入 stream 上异步下发。量化、pad、attention 和裁剪均在 device 侧完成，Host 不读取输入
数据，也不参与结果计算。

| 错误类别 | 条件 |
| --- | --- |
| 空指针 | 必选输入、输出、属性、workspaceSize 或 executor 为空；要求 LSE 时 LSE 输出为空 |
| 参数非法 | rank/dtype/device/shape 不匹配；D 越界；GQA 不整除；末维不连续；枚举非法 |
| 因果非法 | `isCausal=true` 且 `Sq != Sk` |
| 运行失败 | Kernel 构建、任务下发或 runtime 执行失败 |

# 详细设计

## 算子分析

### 数学链路

令 K 的平滑均值为 `km`，V 的可选平滑均值为 `vm`，Q/K/V 的量化结果和 scale 分别为
`Q_i8/q_scale`、`K_i8/k_scale`、`V_f8/v_scale`：

```text
K_s     = K - km                                      (smooth_k=True)
V_s     = V - vm                                      (smooth_v 生效时)
Q_i8    = quant_int8(Q, qk_quant_gran)
K_i8    = quant_int8(K_s, qk_quant_gran)
V_f8    = quant_fp8_e4m3(V_s, per_channel)

S       = dequant(Q_i8 * K_i8^T) * sm_scale
P       = softmax(S)
O       = dequant(quant_fp8_e4m3(P) * V_f8)
O       = O + vm                                      (smooth_v 生效时)
LSE     = logsumexp(S) + smooth_k_correction          (return_lse=True)
```

实际实现使用 online softmax。对每个 KV 分块 j 保持行最大值 `m`、分母 `l` 和输出累加器
`acc`：

```text
m_new = max(m, rowmax(S_j))
alpha = exp(m - m_new)
P_j   = exp(S_j - m_new)
l     = l * alpha + rowsum(P_j)
acc   = acc * alpha + quant_fp8(P_j) * V_f8_j
m     = m_new
O     = acc / l * v_scale
```

为扩大 E4M3 对小概率的表示范围，P 的 FP8 转换采用开源路径的指数偏置常量
`S_FP8_OFFSET = 8.807`。实现时分子和分母使用同一倍率并在 LSE 中抵消，保证偏置不改变
归一化结果。掩码位置在求 rowmax 和 exp 前置为无效值，尾部列不参与分母。

### Q/K 量化粒度

`per_warp` 与 `per_thread` 是算法分组语义，不直接等同于 NPU 线程模型。实现采用“先逐行统计，
再按组规约 scale”的方式，将开源分组稳定映射到 AscendC Vector 计算：

| 模式 | Q 分组 | K 分组 | 验证要求 |
| --- | --- | --- | --- |
| `per_warp` | 每 32 个 Q 行一组 | 每 64 个 K 行一组 | 对标 GPU 同模式 |
| `per_thread` | 32 行内按开源 lane 映射分为 8 组 | 64 行内按开源 lane 映射分为 4 组 | 对标 GPU 同模式 |

每组先以 FP32 统计 absmax，再按对应 scale 和舍入规则生成 INT8。零 absmax 组写入安全 scale
并输出全零，避免除零。实现不能用不同粒度的输出互相作为 golden。

### V/P FP8 量化与累加

V 沿序列维按通道统计 scale。`fp32+fp16` 使用任务书指定的低范围 scale 配置，其他累加
模式使用对应开源配置。P 在 online softmax 分块内完成 FP8 转换后直接送入 PV，不写回 GM。

| 模式 | 分段结果 | 跨段缓冲 | 设计目的 |
| --- | --- | --- | --- |
| `fp32` | FP32 | FP32 | 保留最高累加精度 |
| `fp32+fp32` | 低精度 MMA 分段，段结果转 FP32 | FP32 | 对齐定期归约语义 |
| `fp32+fp16` | 低精度 MMA 分段，段结果经 FP16 舍入 | FP32 | 对齐默认两级累加路径 |

Atlas 950PR 上 FP8 Cube 的实际累加与 Fixpipe 转换能力须通过 CANN 9.2.0-beta.1 编译和设备
测试确认。若硬件原语与 GPU 累加器的内部舍入不完全同构，以同配置 L-A/L-B 精度门禁判断
方案是否可接受，不用其他累加模式替代默认模式的 golden。

### head_dim 补齐

head_dim 按任务书分为两个计算桶：

| 原始 D | 内部 Dpad | 行为 |
| --- | --- | --- |
| `0 < D < 64` | 64 | 量化输出逻辑补零，结果裁剪为 D |
| `D == 64` | 64 | 不补齐 |
| `64 < D < 128` | 128 | 量化输出逻辑补零，结果裁剪为 D |
| `D == 128` | 128 | 不补齐 |
| `D == 0` 或 `D > 128` | 不支持 | 明确报错 |

输入不物化为补齐后的 FP16/BF16 张量。量化 Kernel 只读取原始 D 列，在量化 workspace 中
补零；补齐列的 Q/K/V 量化值为 0，scale 采用安全值。输出 epilogue 仅写原始 D 列。

## 算子实现

### 模块划分

| 模块 | 主要实现 | 职责 |
| --- | --- | --- |
| PyTorch 适配层 | Python/C++ extension | 参数解析、输出分配、调用 aclnn |
| aclnn Host | C++ | 校验、shape 推导、tiling、workspace、executor |
| 统计与量化 | AscendC Vector | K/V 统计、smooth、Q/K INT8、V FP8、scale |
| QK | CATLASS Cube | INT8 Block/Tile Mmad，输出 INT32/支持类型分片 |
| Softmax | AscendC Vector | 反量化、mask、rowmax、exp、rowsum、P FP8 |
| PV | CATLASS Cube | FP8 Block/Tile Mmad，生成分段输出 |
| 输出后处理 | AscendC Vector | 跨段重标定、归一化、v_scale、vm、裁剪、LSE |

CATLASS 选用仓库和 CANN 9.2.0-beta.1 实际提供的 BlockMmad、TileMmad、TileCopy、布局描述
和跨核同步组件。具体类型名在实现时以所固定 CATLASS 版本为准，设计文档不假定未验证的内部
符号。AscendC 与 CATLASS 的边界是：CATLASS 负责可复用的 Cube 数据通路，AscendC 负责本算法
特有的 scale、softmax、量化和累加状态。

### Host 侧设计

Host 从输入描述符提取 B、Hq、Hkv、Sq、Sk、D、布局与 stride，并完成以下工作：

1. 验证 dtype、rank、device、shape、末维连续、GQA 和枚举值；
2. 计算 Dpad、GQA 比例、默认 `sm_scale` 和各 workspace offset；
3. 根据 dtype、Dpad 与累加模式选择 Kernel 变体；
4. 以 `B * Hq * ceil(Sq / Br)` 为主并行域切分工作；
5. 因果场景按每个 Q block 的有效 KV 长度估算工作量，减少核间长尾；
6. 将 tail、layout stride、量化粒度和功能标志写入 tiling data；
7. 返回 workspace 大小和一次性 executor。

Tiling 不读取张量值，不按公开测试 shape 或 case id 分发。桶只由 dtype、布局、Dpad、运行时
shape、因果标志和累加模式等合法元数据确定。

### Workspace 规划

workspace 保存跨 Kernel 阶段必须保留的数据：

```text
Q_i8 + q_scale
K_i8 + k_scale
V_f8 + v_scale
km / vm（按开关分配）
LSE correction（按 return_lse 和 smooth_k 分配）
必要的分块规约中间量
```

Q/K/V 量化结果各写一次并由后续主循环复用。S、P 和 PV 分段结果保持在 L1/L0/UB 流水中，
不进入 workspace。所有 offset 按硬件要求对齐，Host 采用溢出安全的 size 计算；workspace 生命周期
由 aclnn executor 管理，第二段完成前不得释放。

### Kernel 侧数据流

Kernel 分为统计量化阶段和 attention 主循环阶段：

```text
GM Q/K/V
  -> MTE2/Vector: smooth + scale + INT8/FP8 quant
  -> GM workspace: Q_i8/K_i8/V_f8 + scales
  -> MTE2/L1/L0A/L0B
  -> Cube: INT8 QK
  -> Fixpipe/UB
  -> Vector: dequant + mask + online softmax + P FP8
  -> UB/L1
  -> Cube: FP8 PV
  -> Fixpipe/UB
  -> Vector: rescale + accumulate + normalize + crop
  -> GM O/LSE
```

主循环采用 Q block 与 KV block 双层遍历。Q block 在多个 KV block 之间保留 `m/l/acc`，KV block
采用双缓冲或多缓冲覆盖搬运、Cube 和 Vector 计算。AIC/AIV 通过 CATLASS/AscendC 提供的事件与跨核
同步原语交换分片所有权，任一缓冲在生产者发出完成信号前不得被消费者读取，也不得在消费者释放前
复用。

### 边界与掩码

- Sq、Sk 非 tile 整倍数时，load、rowmax、exp、rowsum 和 store 都使用边界 mask；
- 因果 mask 与 KV 尾 mask 在 softmax 前统一生效；
- 完全无有效 KV 的内部分片不更新 `m/l/acc`；
- NHD/HND 通过运行时 stride 寻址，不额外转置完整张量；
- GQA 通过 `q_head / group_size` 映射到 KV head；
- 所有输出写回都以原始 D 为上界，避免补齐列越界。

### Kernel 变体与分发

计划只实例化会改变数据类型或 Cube/Fixpipe 数据通路的维度：

| 维度 | 处理方式 |
| --- | --- |
| 输入 dtype | FP16、BF16 编译变体 |
| Dpad | 64、128 编译变体 |
| PV 段输出 | FP32、FP16 舍入编译变体 |
| 因果、布局、gran、smooth、LSE | 尽量作为运行时 tiling 标志 |

该设计避免为公开 case 生成逐 shape Kernel，也避免重复实现同一机制。实际编译变体数量以资源占用、
编译时间和设备证据为准，新增变体必须对应明确的数据通路差异。

### 性能优化策略

1. Q/K/V 只量化一次，K/V 的低比特结果在多个 Q block 间复用；
2. S 与 P 不落 GM，QK、softmax、PV 通过片上流水衔接；
3. QK 与 PV 使用 CATLASS Cube 模板，Vector 同时处理 scale、softmax 和 epilogue；
4. HND/NHD 通过 stride 访问，避免完整布局转换；
5. 因果任务按有效 KV 工作量分核，降低尾核拖延；
6. KV 双缓冲重叠 GM/L1 搬运、Cube Mmad 与 Vector 计算；
7. scale 与状态向量驻留 UB，在可行范围内减少重复搬运；
8. 以 profiler 判断 Cube、Vector、MTE 和同步瓶颈，再调整 Br/Bc 与流水级数。

性能优化不得改变接口语义、精度配置或 FIA 对比口径，也不得通过调用基线整算子完成未覆盖路径。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas 950PR | √ |

依赖 Atlas 950PR 的原生 FP8 与 INT8 Cube 能力。编译、功能和性能验收固定在任务书指定的
CANN 9.2.0-beta.1 环境完成。

## 算子约束限制

1. 仅支持前向 INT8 QK + FP8 PV；
2. Q/K/V 仅支持 FP16 或 BF16，且必须同 dtype、同 NPU device；
3. 原始 `0 < D <= 128`，末维必须连续；
4. HND/NHD 之外的布局报错；
5. `Hq % Hkv != 0` 报错；
6. 因果模式要求 `Sq == Sk`；
7. 非空 `attn_mask` 不支持并明确报错；
8. `smooth_v` 在非 `fp32` 模式告警并忽略；
9. 不支持 FP16 PV、varlen、反向或 CUDA/Triton 路径；
10. 不允许 CPU、PyTorch、FIA、其他官方整算子或其他后端 fallback。

# 可维可测分析

## 精度标准与性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 功能 | PyTorch/aclnn 契约、shape、异常、分发和 LSE 行为完整 | SageAttention2 任务书 |
| 精度 L-A | NPU 对 GPU 同 gran、同 accum 路径，满足混合容差 | 任务书/生态算子精度标准 |
| 精度 L-B | NPU 与 GPU-Sage 相对 FP32 SDPA 的误差比满足门禁 | 任务书 |
| 性能 | P-01 至 P-07 每例、PyTorch/aclnn 两路径均相对 FIA ≥1.6x | 任务书 |

本文处于设计阶段，不记录尚未执行的功能、精度或性能结果。实测数据、设备信息、CANN 版本、
日志和 profiler 证据在代码完成后写入自验证报告。

### 精度判据

| dtype | rtol | atol | matched_ratio | max_abs_error_limit |
| --- | --- | --- | --- | --- |
| FP16 | `2^-9` | `2^-9` | ≥0.99 | `1e-1` 或 32 ULP |
| BF16 | `2^-6` | `2^-6` | ≥0.99 | `1e0` 或 32 ULP |

L-B 对 FP32 SDPA 的误差比例要求为 max ≤2、mean ≤1.2、RMSE ≤1.2。`per_thread` 与
`per_warp`、三种 accum 模式分别使用 GPU 同配置结果，不交叉替代。LSE 使用 FP32 容差或相对
GPU LSE 的混合容差。

## 功能测试设计

| 编号 | 验证范围 |
| --- | --- |
| TC-01 | `sageattn` 固定分发到 FP8 PV；FP16/BF16；HND/NHD |
| TC-02 | 两种 qk gran × 三种 PV accum 的组合 |
| TC-03 | smooth_k/smooth_v 开关及告警忽略行为 |
| TC-04 | return_lse=False/True，shape、dtype、数值与修正项 |
| TC-05 | D=16/48/63/64/65/96/127/128 的 pad 与裁剪边界 |
| TC-06 | MHA/GQA，`Hq/Hkv` 为 1/4/8，非因果 `Sq != Sk` |
| TC-07 | 因果模式与非 64 整倍数序列尾块 |
| TC-08 | dtype、device、shape、枚举、stride、mask 和空指针异常 |
| TC-09 | 8K/16K 长序列与 workspace 上界 |
| TC-10 | 替换 `F.scaled_dot_product_attention` 的 example 级冒烟 |
| TC-11 | kwargs/环境变量/默认值的覆盖优先级 |
| TC-12 | PyTorch 与 aclnn 对同一输入的输出一致性 |

除固定用例外，增加随机 B/H/S/D、非整块尾部、非因果不等长和合法非末维 stride 组合。异常
用例必须证明在 Kernel 下发前拒绝非法合同；设备测试必须覆盖实际选中的每个编译变体。

## 性能测试设计

环境固定为 Atlas 950PR + CANN 9.2.0-beta.1。同一设备、同 shape、同 dtype、同输入和同等
warmup 下分别测量：

- PyTorch：`sageattn_qk_int8_pv_fp8_asc` 对
  `torch_npu.npu_fused_infer_attention_score`；
- aclnn：`aclnnSageAttention2` 对 `aclnnFusedInferAttentionScoreV5`。

FIA 使用 `BNSD` 对齐 SageAttention HND，因果/GQA 参数保持语义一致。同一报告内固定 FIA
PyTorch API 版本，不混用 v1/v2。排除首次编译，充分预热后用 device event 或等价设备计时，
每次统计前后同步；报告原始耗时、统计方法和重复次数。

| 编号 | shape / 场景 | dtype | 配置 | 两路径门禁 |
| --- | --- | --- | --- | --- |
| P-01 | `[1,32,4096,128]` | FP16 | 非因果、MHA | 各自 ≥1.6x FIA |
| P-02 | `[1,32,8192,128]` | FP16 | 非因果、MHA | 各自 ≥1.6x FIA |
| P-03 | `[1,32,8192,128]` | FP16 | 因果、MHA | 各自 ≥1.6x FIA |
| P-04 | `[2,32,8192,128]` | FP16 | 非因果、MHA | 各自 ≥1.6x FIA |
| P-05 | Hq/Hkv=32/8，S=4096，D=128 | FP16 | 非因果、GQA | 各自 ≥1.6x FIA |
| P-06 | `[1,32,4096,128]` | BF16 | sageattn 默认 | 各自 ≥1.6x FIA |
| P-07 | `[1,32,16384,128]` | FP16 | 非因果、MHA | 各自 ≥1.6x FIA |

每个 case 的 PyTorch 与 aclnn 路径均独立判定，且两条路径各自的全用例几何平均也必须
≥1.6x。任一 case、任一路径未达标即不声明性能验收通过。

## 性能定位指标

profiler 至少记录：

1. Quant、QK、softmax、PV、epilogue 各阶段 device 时间；
2. Cube/Vector/MTE 利用率及主要 stall；
3. GM 读写量与是否出现 S/P 的意外 GM 回流；
4. AIC/AIV 同步等待与流水气泡；
5. workspace 大小、Kernel 下发数与 PyTorch/aclnn Host 开销；
6. 因果长尾下各核工作量分布。

若性能未达标，优先根据上述证据处理最大瓶颈，不通过降低 FIA 配置、减少精度覆盖或更换
golden 获得表面加速。

## 兼容性分析

本算子位于 `experimental/attention/sageattention2` 独立目录，不修改现有 FIA 或其他
SageAttention 算子接口。PyTorch 与 aclnn 使用同一 Kernel 合同，避免两套数值实现分叉。
不支持项直接报错，不回退到框架、CPU、FIA 或其他后端，因此不会静默改变现有算子的执行路径。

## 风险与验证闭环

| 风险 | 最小验证 |
| --- | --- |
| GPU/NPU FP8 舍入与累加次序不同 | TC-02 按 gran/accum 分组做 L-A/L-B 对照 |
| `smooth_k` LSE 修正与 GQA 广播错误 | return_lse + GQA 的逐行修正单测 |
| D=64/128 分桶和尾列越界 | D 边界集 + sanitizer/边界 case |
| online softmax 尾块或因果全 mask 错误 | 非整块 Sq/Sk 与因果对角边界测试 |
| AIC/AIV 缓冲所有权错误 | 小 shape 单步验证、同步检查和设备 sanitizer |
| 双接口口径不一致 | 同输入 PyTorch/aclnn 输出与错误行为对照 |
| 性能结论受 Host/JIT 干扰 | 预热、device 计时、同步与 profiler 交叉验证 |
