# KvCacheTurboQuant 算子设计文档

> 任务来源：昇腾社区 2026 年 9 月社区任务「kv_cache_turbo_quant 算子开发」；目标仓库：https://gitcode.com/cann/ops-transformer （合入 experimental/attention）
> 版本说明（2026-09-24 修订）：本版按**已交付实现**重写 Host/Tiling 设计与 Kernel 设计两章、精度对齐策略与可维可测章节，并按官方设计文档模板重构章名层级，补充 **需求背景 / 需求分析 / 硬件与约束 / 可维可测** 四章内容。
> 首版（PR #1825）中「矩阵乘走纯矢量单元、无需引入 cube」的选型经真机实测无法满足 10× 性能要求，实际交付实现为 **AIC/AIV 混合 Cube 路径**（tilingkey 202/203/204），本版与之对齐。
> 文中实现口径以交付代码 `experimental/attention/kv_cache_turbo_quant/` 为准；实测数字取自《KvCacheTurboQuant 算子自测报告_20260924》与 `results/kvtq_selftest_377a0591_20260924.log`。

---

## 1. 需求背景（required）

### 1.1 需求来源

- **任务书**：随任务下发的 `kv_cache_turbo_quant算子开发任务书.md`（下文简称「任务书 §X」）。任务定义、接口、验收标准、交付件与合入要求均以该文件为准；本文档对任务书条款的引用逐条落在 2.1 的依据列。
- **任务页**：昇腾社区任务中心（hiascend）2026 年 9 月社区任务列表中的「kv_cache_turbo_quant 算子开发」条目。任务列表与设计文档提交规则见 `https://gitcode.com/cann/cann-competitions/tree/master/04_tasks/01_community-task-2026/tasklist` 及同目录 `README.md`；本设计文档按 `tasklist/{任务目录}/{用户名}/docs/design.md` 路径以 PR 形式提交。
- **随任务提供的附件**（均在任务目录内，自测全程使用）：

  | 附件 | 内容 | 在设计中的作用 |
  |---|---|---|
  | `golden.py` | PyTorch/torch_npu 参考实现 `calc_expect_func`：内置 2/3/4 bit 质心表 `CENTROIDS`、边界中点、`_pack_bits` 位打包 | 精度标杆，语义定义见 1.2.3 |
  | `op.json` | 算子原型：3 个输入（bfloat16 + float32×2）、4 个输出（uint8×2 + bfloat16×2）、attr `mse_bits`（默认 3） | 接口与 dtype 的第一手约束（2.2） |
  | `case.json` | 5 个官方用例（shape、`value_range`、`data_path`、`err_threshold`），`expect_func` 指向 `golden.py:calc_expect_func` | 验收用例集合与自动比对判据（7.5、7.1） |
  | `test_data/` | 各用例输入/期望 `*.npz`、`matrices.npz`，以及 `rotation_matrix.bin` / `qjl_matrix.bin`（各 65536 B = 128×128 float32） | 精度对拍输入与矩阵 fixtures（7.3） |

- **上游现状与痛点**：vllm-ascend 在长序列场景下 KV cache 显存占用是主要瓶颈，框架内既有的压缩手段在压缩率或功能上均不满足需求（1.2.1、1.2.2）。
- **参考实现（TBE）说明**：**本任务未提供同名的 TBE 算子实现，也未给出任何 TBE 侧或算子库侧的实现路径、API 路径**。因此本文档不填写参考实现路径，也不据任何外部实现反推仓内既有实现的形态；算子功能与数值语义以任务书 §2.3 的量化（Encode）步骤定义和 `golden.py` 为准。

### 1.2 背景介绍

#### 1.2.1 业务场景

- **KV cache 是长上下文推理的常驻显存**：decode 每生成一个 token 都要为其保存 K、V 两份向量，cache 大小随已生成 token 数**线性增长**，而不像权重那样一次性占用；上下文越长，cache 越占主导。
- **bf16 下的基数是每 head 每 token 256 B（K 或 V 之一）**：`head_dim = 128` 时单个向量为 128 × 2 B = **256 B/head**，K + V 合计 512 B/head/token。GQA 只减少 KV head 数量，不改变 token 维的增长规律。
- **压缩必须在线完成**：任务书 §2.1 要求量化在 KV 生成路径上就地完成，不允许"先全精度落盘、离线再压"的两段式流程，因此编码算子本身的时延与带宽开销必须远小于它节省的显存与后续访存收益——这直接决定了 3.7 的矩阵乘选型与 5.6 的性能结论。
- **量化对象是"向量"而非"张量"**：TurboQuant 对每个 128 维向量独立编码，不需要跨向量统计量（对比 per-tensor/per-channel scale 类量化），因此没有标定阶段、没有跨 token 数据依赖——这是按向量粒度并行（4.1）的前提。

#### 1.2.2 参考实现现状分析

vllm-ascend 当前框架内已有的 KV cache 压缩方案：

- **INT8 静态量化（C8）**：per-channel scale + offset，压缩率仅 2x；
- **LSH 哈希（KVComp）**：用于稀疏选择，不压缩存储。

两者都不满足"在 bf16 精度语义下把 KV cache 压到 4 bit/channel 量级、且编码完全在线"的需求，故需要参考 TurboQuant 论文（ICLR 2026 投稿，"TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate"）设计并实现面向标准 MHA/GQA KV cache 的**在线向量量化压缩算子**：首版采用统一 mse_bits（2/3/4 bit）MSE 主编码 + 1-bit QJL 残差编码。默认 3-bit 配置下，计入向量范数与残差范数后，head_dim=128 时物理存储 4.25 bit/channel，理论显存压缩比约 **3.76x**（vs BF16，口径见 1.2.5）。

**参数能力表**（接口的能力边界；形状与约束与任务书 §2.3、`op.json`、交付实现一致）：

| 参数 | 参数含义 | 数据类型 | 支持数据类型 | 约束 | 形状 |
|---|---|---|---|---|---|
| kv_vectors | 待量化的 K 或 V 向量 | bfloat16 | bfloat16 | 三维、ND 连续；末维 head_dim 固定 128；不参与广播 | [num_tokens, num_kv_heads, 128] |
| rotation_matrix | 正交旋转矩阵 H | float32 | float32、bfloat16 | [128,128]；推理期固定输入，算子不生成随机矩阵；bf16 时两矩阵需同 dtype | [128,128] |
| qjl_matrix | 残差高斯投影矩阵 S | float32 | float32、bfloat16 | [128,128]（qjl_dim = head_dim）；与 rotation_matrix 同 dtype（运行时校验） | [128,128] |
| quant_idx | 主量化编码（bit-packed） | uint8 | uint8 | 末维 = ceil(128·mse_bits/8) = 32/48/64；字节序与位序与 golden 逐字节一致 | [num_tokens, num_kv_heads, ceil(128·mse_bits/8)] |
| quant_qjl | 残差 QJL 符号编码（bit-packed） | uint8 | uint8 | 末维固定 16（128 个符号位） | [num_tokens, num_kv_heads, 16] |
| quant_norm | 输入向量 L2 范数 ‖x‖₂ | bfloat16 | bfloat16、float16（由 attr `out_dtype` 选择） | 每个 (token, head) 一个标量，bf16/fp16 舍入按 RNE | [num_tokens, num_kv_heads] |
| quant_gamma | 原尺度残差范数 ‖x‖·‖r‖ | bfloat16 | bfloat16、float16（由 attr `out_dtype` 选择） | 同上；已含 ‖x‖，消费端不再二次缩放 | [num_tokens, num_kv_heads] |
| mse_bits | MSE 主量化位宽（attr） | int | int，取值 {2, 3, 4} | 默认 3；其余取值 host 侧报错（2.3） | — |
| out_dtype | quant_norm/quant_gamma 元素类型（attr） | int | int，取值 {0, 1} | 默认 0 = bfloat16，1 = float16；为交付版覆盖任务书"norm/gamma 支持 float16/bfloat16"而新增的可选属性 | — |

注：`op.json` 原型把 rotation_matrix / qjl_matrix 登记为 float32（与 `case.json` 的 fixtures 一致），任务书 §2.3 允许 float32 / bfloat16；交付实现按任务书放宽并在运行时校验两矩阵同 dtype。

#### 1.2.3 标杆（golden）与依赖边界

- **标杆来源**：`golden.py` 的 `calc_expect_func(kv_vectors, rotation_matrix, qjl_matrix, mse_bits)`，在 torch_npu 上以 PyTorch eager 表达任务书 §2.3 的步骤 1~8；它是**精度标杆**，本算子输出必须与其对齐。
- **标杆中决定实现口径的四条语义**（每条都必须在 kernel 内被严格复现）：① 归一化保护 `unit = x / clamp_min(norm, 1e-30)`，且 `norm == 0` 时按 `where` 语义直接置零（不产生 inf/nan）；② 量化边界取相邻质心中点 `(centroids[:-1] + centroids[1:]) * 0.5`，判定为**严格大于**（`rotated > boundaries` 计数）；③ QJL 先对残差归一化得 `r̂`（`residual_norm > 0` 保护）再投影 `S @ r̂`，按 **`>= 0` 记 1**（不是 sign 的 0），判定对正缩放不变、故实现可省去这次归一化（3.1）；④ quant_norm / quant_gamma 返回前转 bfloat16（RNE）。
- **`case.json` 与标杆的关系**：每个用例通过 `expect_func` 指向 `golden.py:calc_expect_func`，并用 `err_threshold` 给出自动比对阈值（idx `[0, 1e-5]`、qjl `[0, 1.1e-5]`、norm/gamma `[0, 0]`，即要求逐位）。该阈值是**比对判据**，与任务书 §3.2 的统计判据（相对 MSE、内积误差）是两套不同口径，对照与实测见 7.1、7.3。
- **依赖边界**：`torch` / `torch_npu` / `ml_dtypes` 只出现在标杆与测试侧，**不属于算子运行期依赖**；算子运行期只依赖 CANN 工具链与 ACL runtime（2.6）。`op.json` 用于核对输入/输出 dtype 与 attr 默认值，`case.json` 用于穷举验收 shape 与属性组合；二者覆盖范围有限，不得据此收窄任务书更宽的能力范围（2.7）。

#### 1.2.4 TurboQuant 两阶段量化原理

对输入向量 x ∈ R^d（d = head_dim = 128）：

1. **MSE 主量化器**：先归一化到单位球面，再做随机正交旋转 y = H·u（旋转后各坐标近似服从单位球面坐标分布），随后逐维做标量量化——量化码本为该分布下的 Lloyd-Max 最优质心（mse_bits 决定 4/8/16 个质心），质心表离线预计算、编译期固化；
2. **残差 QJL 修正**：主重构残差 r = y − ŷ 做高斯随机投影 S·r 并取 1-bit 符号，用于推理侧内积估计的无偏修正（decode/attention 侧使用，本算子仅负责编码）。

编码输出四元组：bit-packed 主编码 idx、bit-packed QJL 符号 qjl、向量范数 x_norm、原尺度残差范数 γ = ‖x‖·‖r‖。

#### 1.2.5 收益量化

存储格式（mse_bits=3，head_dim=128）：

| 组件 | 摊销 bit/channel | 总 bit | 说明 |
|---|---:|---:|---|
| idx（主量化） | 3.000 | 384 | 48 bytes（bit-packed） |
| qjl（残差符号） | 1.000 | 128 | 16 bytes（bit-packed） |
| x_norm（向量范数） | 0.125 | 16 | 2 bytes（bf16 scalar） |
| gamma（残差范数） | 0.125 | 16 | 2 bytes（bf16 scalar） |
| **合计** | **4.250** | **544** | **68 bytes/head vs 原始 256 bytes/head** |

压缩比 256 / 68 ≈ **3.76x**（bit 打包格式见 3.4，交付实现输出的字节数与该口径一致，见 7.3.1）。

**收益口径限定**：上表的 3.76x 是**编码 payload 的比值**，不含下列三项开销，实际端到端收益只会更低——① **共享矩阵 H/S**：推理期全体 token/head 共用一份，不计入 per-head 摊销，但调用方需为每层（或每组）常驻一份；② **分配器对齐与分页元数据**：KV cache 通常按 block/page 分配，单行字节数会被向上对齐（idx 末维 32/48/64、qjl 16 均为整字节，不跨行合并 bit）；③ **编码 workspace**：混合 Cube 路径的中间量中转区（声明公式见 4.8），属运行期峰值占用而非 cache 常驻。因此"理论压缩比 3.76x"只能作为编码格式的容量指标，端到端显存收益须按调用方的实际分配粒度与 cache 元数据设计重新核算。

### 1.3 功能分析

- **算子功能（一句话）**：对 KV cache 中每个 128 维向量做在线两阶段量化编码（MSE 主量化 + QJL 残差符号），一次性输出 bit-packed 主编码、bit-packed QJL 编码与两个范数标量。
- **输入清单**：`kv_vectors`（bfloat16，[T, H, 128]）、`rotation_matrix`（float32/bfloat16，[128,128]）、`qjl_matrix`（float32/bfloat16，[128,128]）；三者均为 ND 连续、必需。**输出清单**：`quant_idx`（uint8）、`quant_qjl`（uint8）、`quant_norm`（bfloat16/float16）、`quant_gamma`（bfloat16/float16），形状见 1.2.2；属性 `mse_bits`（默认 3）、`out_dtype`（默认 0），非法值行为见 2.3。
- **不支持广播**：三个输入的 shape 必须是各自声明的形状，输入之间不存在广播语义（H/S 是每个向量都要用的常量矩阵，不对 batch 维广播）；输出 shape 完全由输入 shape 与 `mse_bits` 推导（4.7）。
- **向量间完全独立**：每个 (token, head) 的编码只依赖它自己的 128 个元素与两份固定矩阵，**无跨向量归约、无跨向量统计、无跨 token 依赖**，故向量粒度可自由并行（4.1）。
- **算子职责边界与精度**：只负责 encode，不含量化参数标定、不含反量化、不含 attention/内积计算（decode 侧消费方式只在 3.6 约定）；算子内部按 fp32 语义完成归一化、量化与归约（3.1 的实现口径），仅矩阵乘因硬件限制使用 bf16 拆分（3.7、5.5），输出标量按 bf16/fp16 存储。

### 1.4 范围与边界

- **仅支持标准 MHA/GQA KV cache，不支持 MLA latent cache**：DeepSeek MLA 缓存的是低秩联合压缩 latent `cKV` 与独立 RoPE key `kR`，并通过权重吸收避免显式生成普通 K/V，不能把 `[cKV, kR]` 简化为一个 576 维普通 KV head 后套用本算子；TurboQuant 若要适配 MLA，需要围绕 latent cache、RoPE 分支与 MLA attention 公式单独设计接口与融合路径（任务书 §2.1）。
- **论文中 3.5 bit/channel 的混合精度模式（outlier channel 分组）不纳入首版范围**：3.5 bit/channel 是总编码平均位宽，依赖 outlier / non-outlier channel 分组与混合精度分配，并不等价于"统一 3-bit 主编码 + 1-bit QJL"；如后续支持，需要增加 channel 分组、不同主量化位宽与独立量化参数（任务书 §2.1）。
- **版本口径澄清**：任务页简介曾出现 A3 与论文 3.5 bit/channel 的描述，而任务书 §3.1 与 `op.json` 的口径为 **Atlas 800T A2 + 统一主编码位宽**（计入 1-bit QJL 与两个标量后 4.25 bit/channel）。硬件口径经任务方答复按任务书 Atlas 800T A2 执行（7.10），本文档与该口径一致。
- **首版固定 head_dim = 128、编码位宽范围**：质心表为 d=128 单位球面坐标分布的 Lloyd-Max 解，换 head_dim 需重算质心（3.8、6.5）；mse_bits ∈ {2,3,4}，加上 QJL 后编码位宽分别为 3/4/5 bit/channel。

### 1.5 交付范围声明

本文档描述的是**已交付实现与验证结论**（区别于纯设计稿）：交付实现走 tilingkey 202/203/204 的 AIC/AIV 混合 Cube 路径，首版原型中的纯矢量路径已从交付代码中移除（3.7）；实现状态、实测结论与证据链见第 7 章，凡涉及"实测"的数字均来自真机自测日志，未执行项一律显式标注（7.8）。

---

## 2. 需求分析（required）

### 2.1 需求描述（含依据引用）

使用 Ascend C 编程语言，基于 aclnn 算子工程化模式，在 Atlas 800T A2（CANN 9.1.0+）上实现 KvCacheTurboQuant 量化（encode）算子：输入 KV 向量与两个固定矩阵（正交旋转矩阵 H、高斯投影矩阵 S），在线输出 bit-packed 量化编码与两个范数标量；功能与数值步骤完全对齐任务书 §2.3 的量化（Encode）定义。本版交付代码的真机验证环境为 CANN 9.1.0（7.11）；9.1.0 的 opbuild 要求 def 层使用 `DataTypeList/FormatList` API，交付代码已按此适配。需求条目与依据逐条对应如下：

| # | 需求 | 依据 |
|---|---|---|
| R1 | 面向标准 MHA/GQA KV cache 的**在线**向量量化（encode）算子；不支持 MLA | 任务书 §2.1（核心要求与适用范围说明） |
| R2 | 首版采用统一 mse_bits（2/3/4）MSE 主编码 + 1-bit QJL 残差编码；计入两个范数后 4.25 bit/channel、理论压缩比 3.76x | 任务书 §2.1、§2.3 存储格式设计 |
| R3 | 采用 aclnn 算子工程化模式（标准两段式接口 + 算子工程骨架） | 任务书 §2.2 |
| R4 | 接口：3 输入 / 4 输出，attr `mse_bits` 默认 3；输出形状按 `ceil(head_dim·mse_bits/8)` 与 `ceil(qjl_dim/8)` 推导 | 任务书 §2.3 接口定义与参数说明、`op.json` |
| R5 | 约束：head_dim 固定 128；num_kv_heads 4~32；mse_bits 支持 2/3/4；矩阵推理期固定；量化在线完成 | 任务书 §2.3 算子约束 |
| R6 | 精度：`MSE/‖x‖² < 0.05`；内积估计相对误差 95th percentile `< 0.1` | 任务书 §3.2 |
| R7 | 性能：吞吐不低于 prefill 阶段 KV 生成速率，即相对功能等价 PyTorch eager reference 加速 ≥ 10× | 任务书 §3.3 |
| R8 | 内存指标不涉及；自验须覆盖任务书 §3.5 的 5 个测试 shape | 任务书 §3.4、§3.5 |
| R9 | 交付件：设计文档、自测用例及测试代码、自测报告、待验收代码地址 | 任务书 §4 |
| R10 | 合入代码不含冗余内容、遵循仓规范与贡献指南 | 任务书 §5 |

### 2.2 参数与接口定义

| 参数 | 方向 | 类型 | 形状 / 约束 | 依据 |
|---|---|---|---|---|
| kv_vectors | 输入 | bfloat16 | [T, H, 128]，ND 连续；三维且末维 = head_dim | 任务书 §2.3、`op.json` |
| rotation_matrix | 输入 | float32 / bfloat16 | [128,128]；正交矩阵（Hadamard 或随机正交），推理期固定 | 任务书 §2.3 |
| qjl_matrix | 输入 | float32 / bfloat16 | [128,128]；高斯随机投影，元素方差由 fixtures 给出，推理期固定 | 任务书 §2.3、`case.json` |
| quant_idx | 输出 | uint8 | [T, H, ceil(128·mse_bits/8)] | 任务书 §2.3、`case.json` |
| quant_qjl | 输出 | uint8 | [T, H, 16]（qjl_dim = 128，ceil(128/8) = 16） | 任务书 §2.3、`case.json` |
| quant_norm / quant_gamma | 输出 | bfloat16（默认）/ float16 | [T, H]；元素类型由 `out_dtype` 统一控制 | 任务书 §2.3；`out_dtype` 为交付版新增可选属性 |
| mse_bits（attr） | 属性 | int | 取值 {2, 3, 4}，**默认 3**；非法值 host 报错（2.3） | 任务书 §2.3、`op.json` |
| out_dtype（attr） | 属性 | int | 取值 {0, 1}，**默认 0**（0 = bfloat16，1 = float16） | 任务书 §2.3「float16 / bfloat16」的落地方式；交付版新增 |

属性与 ABI 说明：除上表两项外**不新增任何其它 ABI 属性**；参数顺序与 `op.json` 原型一致，`out_dtype` 追加在 `mse_bits` 之后，旧调用形式（仅传 `mse_bits`）按默认值 0 行为等价。输入输出均为 ND 格式连续 tensor、无格式互转属性；shape 推导、dtype 推导、校验（2.3）与 tiling 计算全部发生在 host 侧的 aclnn 两段式接口中（4.8），kernel 内不再做参数校验。

### 2.3 属性与非法值行为

非法输入一律在 **host 侧**被拦截并返回错误、**不进 kernel**（与 6.5 的约束清单一致）；错误码按仓内 aclnn 规范返回，不自定义错误码体系。

| 非法场景 | 判定位置 | 行为 |
|---|---|---|
| `mse_bits` ∉ {2, 3, 4}；`out_dtype` ∉ {0, 1} | host（tiling） | 返回参数错误，不进入 kernel |
| `num_kv_heads` ∉ [4, 32] | host（tiling） | 返回参数错误（该校验不放 infershape：9.1.0 的 opbuild 在构建期以占位 shape 执行 infer，不能依赖真实维度值） |
| `head_dim` ≠ 128 | host（shape 校验） | 返回参数错误（质心表仅对 d=128 有效） |
| kv_vectors 非三维 / 元素类型非 bfloat16 | host（input 校验） | 返回参数错误 |
| 两个矩阵非 [128,128]、dtype 不在 {float32, bfloat16}、或二者 dtype 不一致 | host（input 校验） | 返回参数错误 |
| 任一输入/输出非 ND 连续 | host（format 校验） | 返回参数错误；不在算子内隐式转连续（4.8） |
| 输出 shape 与推导值不一致 | host（output 校验） | 返回参数错误 |
| 空指针 / workspace 不足 | host / 框架层 | 按仓内规范返回错误（workspace 声明公式见 4.8） |

### 2.4 需求拆解与验收指标

需求拆解（实现范围）：

1. 支持标准 MHA/GQA KV cache 量化：输入 kv_vectors [num_tokens, num_kv_heads, head_dim]，bfloat16；head_dim 首版固定 128，num_kv_heads 4~32；
2. 支持 mse_bits = 2/3/4 三种主量化位宽（attr，默认 3）；
3. 输出 4 个 tensor：quant_idx（uint8，[T, H, ceil(128·mse_bits/8)]）、quant_qjl（uint8，[T, H, 16]）、quant_norm（bfloat16，[T, H]）、quant_gamma（bfloat16，[T, H]）；
4. 精度：量化-反量化重构相对 MSE < 0.05（默认 3-bit 配置，判定口径与实测见 7.3）；内积估计相对误差 95th percentile < 0.1；输出与 golden（torch_npu 参考实现）对齐，case.json err_threshold 为 idx [0, 1e-5]、qjl [0, 1.1e-5]、norm/gamma [0, 0]。实现口径：保持「逐行 fp32、k 升序累加」语义的纯矢量路径与 golden.py 同 bit（校核方法与结论见 5.5）；交付默认走的混合 Cube 路径因 GEMM 累加顺序不同，在大 shape 上存在 bf16 末位级差异（idx ≤25 byte/786432、qjl ≤194 byte/262144、gamma ≤8/16384 行差 1 ULP、norm 逐位一致），decode 与边界 shape 仍逐位一致；差异溯源与取舍见 5.5、7.3.3、7.10；
5. 性能：相对功能等价 PyTorch eager TurboQuant reference 加速 ≥ 10×，不成为 prefill 瓶颈；真机实测官方 5 个 shape 全部达标（7.3.4）。

验收指标与出处：

| 验收指标 | 阈值 | 出处 | 实现状态（实测结果） |
|---|---|---|---|
| 量化-反量化相对 MSE（主重构，样本均值口径） | < 0.05 | 任务书 §3.2 | 3-bit 0.0404~0.0407、4-bit 0.0110 达标；2-bit 档 0.05 数学不可达（Lloyd-Max 下限 0.1175），实测 0.1298 按理论下限判定（7.3.2） |
| 内积估计相对误差（95th percentile） | < 0.1 | 任务书 §3.2 | 0.030~0.091，全部达标（7.3.2） |
| 性能相对 PyTorch eager | ≥ 10× | 任务书 §3.3 | 官方 5 shape 全部达标，10.2×~41.5×（7.2） |
| prefill 不被编码拖慢 | 定性（不小于 KV 生成速率） | 任务书 §3.3 | prefill_t2048 实测 369.59 µs，10× 目标 385.5 µs（7.2） |
| 与 golden 的输出一致性（自动比对口径） | idx/qjl 阈值、norm/gamma 逐位 | `case.json` `err_threshold` | 位打包与小 shape、边界 shape 逐位一致；大 shape 剩 bf16 末位差（7.1、7.3.3） |

### 2.5 需求 → 设计承接对照表

| 需求项 | 设计承接（章节号） |
|---|---|
| 在线编码，向量间完全独立、无跨向量归约 | 3.1（逐向量公式）、4.1（按向量粒度并行）、5.4（prep 相位） |
| H/S 作为推理期固定矩阵输入，算子不生成随机矩阵 | 1.2.3（标杆语义）、3.1、5.3（权重一次拆分后 L1 常驻） |
| MSE 主量化 + QJL 残差修正两阶段编码 | 3.1、3.8（质心表）、5.4（量化链与 QJL 判定） |
| mse_bits = 2/3/4 三种位宽（含位打包口径） | 3.4（位打包与物理布局）、4.3（tilingkey 规划）、5.4（打包实现） |
| dtype 能力：矩阵 f32/bf16、标量输出 bf16/fp16 | 2.2（属性）、3.2（支持数据类型）、5.3/5.4、6.6 |
| 多核并行与片上资源复用（性能前提） | 4.1（分核）、4.2（UB/L1 预算）、5.1/5.2（双流水 + 双缓冲） |
| 小 shape（decode b1/b64、H=4/32）可用 | 4.1（核数取下界）、5.3（统一按 m=64 装载）、5.7 |
| prefill 吞吐 ≥ 10×，编码不成为瓶颈 | 3.7（矩阵乘选型）、5.3/5.6（AIC 承担 GEMM）、7.2 |
| 精度边界：小 shape 逐位、大 shape 容差 | 5.5（bf16x3 与 bit 级校核）、7.1（验收分层）、7.9（风险） |
| 矩阵乘走 Cube（性能下界决定） | 3.7（选型结论）、6.3（A2 Cube 约束）、5.6（性能结论） |

### 2.6 依赖与边界

- **外部依赖与测试期依赖**：运行期只依赖 CANN 工具链（任务书 §3.1 要求 9.1.0+）——Ascend C 编译器与 opbuild（算子包构建）、ACL runtime（stream/workspace/launch）与构建产物算子包，不依赖 Python；PyTorch + torch_npu + ml_dtypes 仅用于 `golden.py` 标杆，自测 harness 走 aclnn 两段式接口 + aclrt、不引入 torch（7.11）。
- **内部模块划分**：① 算子注册与 shape/dtype 推导（op_host 算子原型与 infershape）；② host 侧输入校验与 tiling 计算（op_host tiling，含分核、tilingkey、workspace 声明）；③ kernel（op_kernel，AIC/AIV 混合 MIX 内核）；④ aclnn 两段式对外接口（`aclnnKvCacheTurboQuantGetWorkspaceSize` / `aclnnKvCacheTurboQuant`）；⑤ 公共常量头 `kv_cache_turbo_quant_common.h`（HEAD_DIM / QJL_BYTES / 质心表，host 与 kernel 单一来源）；⑥ 测试调用器与复现文档（`examples/test_aclnn_kv_cache_turbo_quant.cpp`、`examples/README.md`）。
- **边界**：不依赖仓内其它算子；不与他人共享 workspace；不改变任何既有算子的注册或行为（7.7）。

### 2.7 附件范围说明

随任务下发的 `op.json` 与 `case.json` 只覆盖能力范围的一个子集：矩阵 fixtures 为 float32（`op.json` 也仅登记 float32）、num_kv_heads = 8（边界用例另取 4/32）、head_dim = 128、属性仅 `mse_bits`（2/3/4 各一例），且不带 dtype 变体用例。因此：**不得据附件收窄任务书 §2.3 更宽的 dtype 与属性范围**——交付实现据此按任务书放宽为矩阵 f32/bf16、标量输出 bf16/fp16（2.2），并补齐 num_kv_heads 4~32 的硬校验（2.3）；附件未覆盖的组合（矩阵 bf16、输出 fp16、num_kv_heads = 4/32、零向量）由自测用例补齐并单独给出结论（7.3）。附件通过 `data_path` 引用 `data/rotation_matrix.bin`、`data/qjl_matrix.bin`；自测已取得任务方 fixtures 并全程使用（矩阵身份与日志见 7.8，相关开放问题见 7.10）。

---

## 3. 算子分析（required）

### 3.1 数学公式与符号

对每一个 128 维向量 x（共 N = num_tokens × num_kv_heads 个，向量间完全独立）：

```
1. x_norm = ||x||_2                        # fp32 累加求平方和再开方，输出转 bf16
2. u = x / max(x_norm, 1e-30)              # 归一化到单位球面（x_norm=0 时 u=0，见 golden 的 where 语义）
3. y = H @ u                               # 正交旋转，H: [128,128] fp32，推理期固定
4. idx[j] = Σ_k (y[j] > b_k)               # b_k 为相邻质心中点（mse_bits=3 时 7 个边界）
5. ŷ = centroids[idx]                      # 主重构（旋转空间），质心表编译期固化
6. r = y − ŷ                               # 单位向量的量化残差
7. γ = x_norm · ||r||_2                    # 原尺度残差范数，输出转 bf16
8. qjl[j] = ( (S @ r)[j] >= 0 )            # S: [128,128] fp32 高斯投影；golden 语义为「≥0 记 1」
                                           # （零残差时投影为 0、bit 记 1，非 sign 函数的 0）；
                                           # 判定对正缩放不变，等价于 golden 先归一化 r̂ 再投影，
                                           # 省一次归一化
```

符号与取值：x 为输入向量（R^128）；N = num_tokens × num_kv_heads 为待编码向量总数；u = x/‖x‖ 为单位球面向量（x = 0 时为零向量）；H、S 为两个推理期固定矩阵（[128,128]，H 正交、S 为高斯投影）；y = H·u 为旋转空间向量；b_k 为相邻质心中点（mse_bits−1 个，2/3/4 bit → 1/3/7 个）；idx ∈ [0, 2^mse_bits − 1] 为每维量化索引；ŷ = centroids[idx] 为主重构、r = y − ŷ 为量化残差；qjl 为残差符号编码（128 bit → 16 B）；x_norm、γ 为两个标量输出（bf16/fp16）。

实现口径：步骤 1/2/4/5/6/7 及 bit 打包在 **AIV** 上以 fp32 完成；步骤 3 与步骤 8 的两个矩阵乘在 **AIC** 上以 bf16x3 拆分 GEMM（L0C fp32 累加）完成，数学上等价于 fp32 矩阵乘，差异只在末位（见 5.3、5.5）。

### 3.2 支持数据类型

| 参数 | 方向 | 数据类型 | shape |
|---|---|---|---|
| kv_vectors | 输入 | bfloat16 | [num_tokens, num_kv_heads, 128] |
| rotation_matrix | 输入 | float32、bfloat16 | [128, 128] |
| qjl_matrix | 输入 | float32、bfloat16（与 rotation_matrix 同 dtype） | [128, 128] |
| quant_idx | 输出 | uint8 | [num_tokens, num_kv_heads, ceil(128·mse_bits/8)] |
| quant_qjl | 输出 | uint8 | [num_tokens, num_kv_heads, 16] |
| quant_norm | 输出 | bfloat16、float16 | [num_tokens, num_kv_heads] |
| quant_gamma | 输出 | bfloat16、float16 | [num_tokens, num_kv_heads] |
| mse_bits | attr | int，默认 3，支持 2/3/4 | — |
| out_dtype | attr | int，默认 0：quant_norm/quant_gamma 元素类型（0=bfloat16，1=float16） | — |

矩阵以 bfloat16 输入时，kernel 侧 hi 分量即原值、lo 分量补零（bf16 矩阵天然无低位分量），数学语义与"以该 bf16 值为精确矩阵"的参考一致（6.5）。

### 3.3 支持形状

num_tokens ≥ 1（decode 单 token 到 prefill 数千 token），num_kv_heads ∈ [4, 32]，head_dim 固定 128。num_tokens 与 num_kv_heads 在 kernel 内展平为向量总数 N，不涉及维度语义。N 的取值只影响分核与 tile 划分（见 4.1），不改变任何计算路径的数学语义。

### 3.4 位打包与物理布局

bit 打包格式（必须与 golden 逐字节一致，无自由发挥空间）：**quant_idx** 每 8 个维度为一组，组内第 lane 维的索引值左移 `lane × mse_bits` 位拼入一个 `8 × mse_bits` 位的字，按**小端字节序**拆成 mse_bits 个字节（128 维 → 16 组 × mse_bits 字节 = 32/48/64 bytes）；**quant_qjl** 为 128 个符号位（projected ≥ 0 记 1），每 8 维一组、lane 位移、1 字节/组，共 16 bytes。

| 输出 | 元素类型 | 逻辑形状 | 每 (token, head) 字节数 | 布局说明 |
|---|---|---|---:|---|
| quant_idx | uint8 | [T, H, ceil(128·mse_bits/8)] | 32 / 48 / 64 | 16 组 × mse_bits 字节；组内 lane 0 占最低 mse_bits 位 |
| quant_qjl | uint8 | [T, H, 16] | 16 | 16 组 × 1 字节；组内 lane 0 占最低位 |
| quant_norm | bf16 / fp16 | [T, H] | 2 | 标量，原尺度 ‖x‖ |
| quant_gamma | bf16 / fp16 | [T, H] | 2 | 标量，原尺度 ‖x‖·‖r‖ |
| **合计（mse_bits=3，bf16 标量）** | — | — | **68** | 对比原始 256 B/head |

输出张量为 ND 连续，**逐行独立打包**：行间不做跨行 bit 合并、不加对齐填充，行步长即末维字节数（idx 48 / qjl 16），与 `golden.py` 的 `_pack_bits` 逐行输出一致。mse_bits = 4 时单个字段字为 32 bit，超出 fp32 的 24 bit 整数精度，打包在实现上拆成双掩码分别归并低/高半字（见 5.4）。

### 3.5 边界行为

| 边界场景 | 行为 | 说明 / 依据 |
|---|---|---|
| 零向量（x = 0） | x_norm = 0 ⇒ u = 0 ⇒ y = 0。量化按**严格大于**边界计数（3.1 步骤 4），0 恰落在中间边界（b = 0）之下侧，故 idx 取固定中间索引 `2^(mse_bits−1) − 1`（3-bit 为 3，2-bit 为 1，4-bit 为 7）；ŷ = 该质心（负值），r = −ŷ ≠ 0；γ = x_norm · ‖r‖ = 0 | 与 `golden.py` 的 `where` 保护语义一致；自测用例 `edge_zero_vec`（含全零与置零）**逐位一致** |
| 极小范数 | `invNorm = 1/max(norm, 1e-30)`，与 golden 的 `clamp_min(1e-30)` 一致；x_norm 低于 1e-30 时不产生 inf/nan（bf16 最小非零量级 ×1e30 仍远小于 fp32 上限），u 按 x·invNorm 计算 | 5.4；该保护同时覆盖 x_norm = 0 的分支 |
| 零残差（y 恰等于某质心） | r = 0 ⇒ ‖r‖ = 0 ⇒ γ = 0。QJL 判定为 **`>= 0` 记 1**，故投影为 0 时该组 8 位记 **全 1**（不是 sign 函数的 0）；golden 对 `residual_norm > 0` 的归一化保护在此不改变判定结果 | 3.1 步骤 8；5.4 的 `Compare(proj, 0, GE)` |
| num_tokens = 1 的 decode | N = num_kv_heads ≤ 32 个向量 < CUBE_TILE_V(64)，落成单 tile、单核；AIC 仍按满 64 行装载计算，Fixpipe 只写真实行数 | 4.1、5.3、5.7；实测 `gqa_decode_b1` 逐位一致 |
| 尾部不足一个 tile / 一个子批 | 尾 tile 行数 < 64 时 AIC 多读相邻有效数据（落在 GM 内、不影响结果），出图按真实行数；尾子批行数 < 32 时归约第二级 `repeatTime` 必须**向上取整** `(elems/8 + 63)/64`，否则尾部行不写出、残留上一轮值，表现为整行垃圾 | 5.3、5.5(4)；该口径同时覆盖 4 字节对齐与 32 B datablock 的对齐要求 |
| N 不能整除核数 | 前 `N % 核数` 个核多处理一个向量（tilingdata `formerCoreNum` / `vecsPerCore`） | 4.1 |
| mse_bits = 4 的字段字 | 4 bit × 8 = 32 bit > fp32 24 bit 整数精度，打包改双掩码分别归并低/高半字 | 3.4、5.4 |

### 3.6 编码消费约定

本算子不含反量化与 attention 计算，仅对下游（decode / 内积估计）约定以下接口语义：① **反量化路径**（任务书 §2.3 步骤 3~8 的逆过程）：`x̂_mse = ‖x‖·Hᵀŷ`，`x̂_qjl = ‖x‖·Hᵀ(ŷ + (√(π/2)/d)·‖r‖·Sᵀs)`，其中 s = 2b − 1（b 为 qjl 位），消费端用同一份推理期固定的 H / S；② **标度约定**：`quant_norm` 是原尺度 ‖x‖，`quant_gamma` 已乘入 ‖x‖（γ = ‖x‖·‖r‖），消费端不再二次乘范数；③ **bit order 与字节序**：idx 与 qjl 均为"每 8 维一组、lane 0 占最低位"、组内按小端字节序落字节（3.4），解码侧必须按同一口径拆位，跨行不合并；④ **QJL 的用途**：只用于内积估计的无偏修正，不参与 MSE 重构判定（含 QJL 的 relMSE 反而升高，实测见 7.3.2）；⑤ **依赖版本标识**：4 个输出只有在与"head_dim = 128、mse_bits、H/S 矩阵身份、质心表版本"绑定的前提下才有意义（7.7）。

### 3.7 矩阵乘实现选型（结论先行）

向量内两次 128×128 矩阵乘（y = H·u、proj = S·r）是全算子唯二的重计算项，选型决定了整体架构：

- **评估后否决的选型：纯矢量单元**（每输出维一次 128 维 fp32 点积）。优点是归约顺序完全可控、与 golden 的一致性风险最低；原型实现（v3）**真机实测 prefill_t2048 = 4462 µs，而该 shape 的 10× 目标是 385.5 µs，差距约 11.6×，无法达标**，已从交付代码中移除（移除前精度数据保留在开发记录中）；
- **交付选型：AIC/AIV 混合 Cube 路径**（tilingkey 202/203/204）。AIC 承担两次 GEMM，AIV 承担归一化与量化链，两侧以 CrossCore 标志双向流水。同一 shape prefill_t2048 降到 **369.2 µs（达标）**，相对纯矢量原型约 12× 提速；
- **精度代价与未采用的增强**：Cube 的 L0C fp32 累加顺序与「k 升序逐项累加」不同源，大 shape 上出现 bf16 末位级差异（idx ≤25 byte/786432 等，见 7.3.3），decode 与边界 shape 仍逐位一致，任务书 §3.2 为容差型判据、实测全部达标（7.3）；bf16x4（补 lo@lo 第四次 Mmad）只把末位差再压小一点，却挤掉近一半性能余量，判定不划算（见 5.5）。

### 3.8 质心表（编译期常量）

```
mse_bits=2: ±0.04002048075, ±0.1335033178                     (4 质心)
mse_bits=3: ±0.02166347019, ±0.06682205945, ±0.1187859178, ±0.19020693   (8 质心)
mse_bits=4: ±0.01134236995, ±0.03428063914, ±0.05802082643, ±0.08325428516,
            ±0.1109927073, ±0.1429702938, ±0.1828317791, ±0.2414890379   (16 质心)
```

质心为 d=128 单位球面坐标分布的 Lloyd-Max 最优解，**仅适用 head_dim=128**，与算子约束一致。kernel 侧以与 golden.py 逐字一致的常量表形式固化（`CentTable<BITS>::CENT`），边界在 fp32 域按 `(c[k]+c[k+1])*0.5` 现算（乘 0.5 为精确运算，与 golden 的 `(centroids[:-1]+centroids[1:])*0.5` 逐 bit 一致），不做任何近似或重算优化。质心表与打包规则集中在公共头 `kv_cache_turbo_quant_common.h`，host 与 kernel 共用同一份常量（7.6）。

---

## 4. Host 与 Tiling 设计（required）

### 4.1 并行维度与分核策略

**并行维度选取**：N = num_tokens × num_kv_heads 个 128 维向量彼此完全独立，无跨向量依赖，直接按向量粒度并行，不需要感知 num_tokens / num_kv_heads 的维度语义。

**分核策略**：

- **核映射**：MIX_AIC_1_2 核映射，`SetBlockDim` = AIC 数，AIV 数 = 2 × blockDim。核数取 `min(ceil(N / CUBE_TILE_V), AIC 数)`（A2 为 20），N 不能整除时前 `N % 核数` 个核多处理向量（tilingdata 的 `formerCoreNum` / `vecsPerCore`）；
- **核内两级切分**：**tile = CUBE_TILE_V = 64 个向量**（AIC 每次 Mmad 的 m 为 `CUBE_MMAD_M = 64`，即一个 tile 一次 Mmad）；**子批 = CUBE_SUBBATCH = 32 个向量**（每 tile 2 个子批，AIV 对偶 `GetBlockIdx()%2` 各承担一半子批（偶/奇））；
- **分核的代价模型（实测）**：时间基本由**每 tile 的串行相位**（AIC GEMM1 → AIV 量化打包 → AIC GEMM2 → AIV qjl）决定，而不是每 tile 的握手开销，因此子批越少（每子批的固定屏障越少）越快。`CUBE_SUBBATCH` 经 8→16→32 扫描后停在 32，前提是通过缓冲别名复用腾出 UB（见 4.2、5.4）；tile 尺寸取 `CUBE_TILE_V = 64`（128 无收益且显著拖慢小 case，属开发期扫描结论：32→64 的收益既不是"每 tile 固定开销"也不是握手粒度，tile 大小不是杠杆）；
- **规模自适应**：核数上界由 N 决定，因此 decode（N ≤ 32）只启用 1 个核的 1 个 tile，prefill（N = 16384）每核约 13 个 tile；同一份 kernel 通过 tiling 参数覆盖两个极端（5.7）。

### 4.2 数据分块与内存优化

**UB 切分（AIV 侧）**：所有子批缓冲按子批宽度（`SUB_ELEMS = 32 × 128 = 4096` 元素）分配，单个 fp32 缓冲 **16 KB**；常驻项只有 1024 宽的打包幂次表与若干标量小缓冲，H/S 权重不占 UB。

| 层级 | 缓冲（按用途归类） | 规模 | 生命周期 | 复用关系 |
|---|---|---|---|---|
| UB（AIV） | 子批级 fp32 工作缓冲（prep 的 x/u、quant 的 y/ŷ/r、P 的 proj 等） | 每份 `SUB_ELEMS = 4096` 元素 = 16 KB | 子批内，按相位使用 | `xf → y`、`u → qjl`（prep 与 quant/P 相位不重叠）；另四处别名：`xin → uHi`、`proj → y`、`tmpFp → ŷ`、`stBk → ŷ` |
| UB（AIV） | 打包幂次表 | 1024 元素常驻 | 全程 | `PackIdx` 与 QJL 打包按 1024 元素分块调用（共用一张 1024 宽的 2^(BITS·(i%8)) / 2^(i%8) 表） |
| UB（AIV） | H / S 权重 | **不占 UB** | — | 权重在 AIC 侧 L1 常驻（6.3） |
| L1（AIC） | H/S 的 hi/lo 四份 NZ 矩阵 | 各 128×128×2 B = 32 KB，共 **128 KB** | 全程常驻 | kernel Init 一次性 Nd2Nz，循环内不再搬权重 |
| L1（AIC） | A/B 装载缓冲 a1 / a2 / b2 | a1、a2 各 64×128×2 B = 16 KB；b2 为 128×128×2 B = 32 KB | 每轮 Mmad | 复用前后成对插入 MTE1_MTE2 / MTE2_MTE1 同步（5.2） |
| L0C（AIC） | C1 累加区 | 64×128×4 B = 32 KB | 每轮 Mmad 链 | fp32 累加，Fixpipe 直出 GM（6.3） |

**容量校核结论**：① UB 侧用量 ≈ 子批级 fp32 缓冲份数 × 16 KB + 常驻小项（1024 宽幂次表 + 标量），`CUBE_SUBBATCH = 32`（`SUB_ELEMS = 4096`）只有在别名腾挪生效后才容纳得下——开发期实测未腾挪时 9 个用例全挂（UB 溢出），这正是"缓冲别名复用是子批提升前提"的由来；② H / S 权重完全不占 UB，是"权重常驻 + 输入流水"能同时成立的原因；③ AIC 侧 L1 固定占用 128 KB（四份 NZ 权重），剩余空间给 A/B 装载缓冲；L0C 的 C1 为 32 KB fp32 累加区，与 `CUBE_MMAD_M = 64`、n = 128 的形状严格对应（6.3）。

### 4.3 tilingkey 规划

- **key 取值**：`tilingkey = 200 + mse_bits`，共 3 个：**202 = 2-bit、203 = 3-bit、204 = 4-bit**。位宽是唯一进入 key 的变量（shape、dtype、核数差异全部由 tiling 参数吸收）；
- **为什么编译期固化**：mse_bits 固化后可消除 kernel 内的运行期分支，并允许按位宽生成"无浪费"的打包代码——例如 3-bit 走单掩码单次归并、4-bit 走双掩码两半字归并（5.4），质心表也按位宽实例化（3.8）；
- **如何分派**：host 侧在 tiling 时由 attr `mse_bits` 计算出 key，框架据此选择对应的 kernel 实例；kernel 内不再出现"按 bits 分支"的代码路径，故三条路径的差异只体现在编译期常量上；
- **覆盖情况**：官方 5 个用例分别落在 202（mse2_t2048）、203（decode b1/b64、prefill_t2048）、204（mse4_t2048），三条 key 均有真机精度与性能数据（7.2、7.3.1）。

### 4.4 数据检测与输入校验

host 侧校验点为（逐项与 2.3 的非法值表对应，错误码按仓内 aclnn 规范返回，不自定义错误码体系）：① `kv_vectors` 为 bf16 三维且末维 = 128；② `rotation_matrix` / `qjl_matrix` 为 [128,128]、dtype ∈ {float32, bfloat16} 且二者同 dtype；③ `head_dim` = 128、`num_kv_heads` ∈ [4, 32]；④ attr `mse_bits` ∈ {2, 3, 4}、`out_dtype` ∈ {0, 1}；⑤ 输入/输出均为 ND 连续；⑥ 输出 shape 与推导值一致（4.7）。任一不满足即返回对应错误码、不进 kernel。校验集中在 host 侧的意义是：kernel 内无需重复做防御性分支（对 AIV 关键路径是净收益），同时保证非法输入不以"设备异常"的形式暴露（6.5）。其中 `num_kv_heads` 的范围校验只能放在 tiling：9.1.0 的 opbuild 在构建期以占位 shape 执行 infershape，不能在该阶段依赖真实维度取值。

### 4.5 tiling 参数与结构体

tiling data 结构定义在 op_host 的 tiling data 头（`kv_cache_turbo_quant_tiling_data.h`）；分核与容量相关的编译期常量定义在公共头，供 host 与 kernel 共用。

| 名称 | 类别 | 含义 / 取值 |
|---|---|---|
| `formerCoreNum` | tilingdata 字段 | N 不能整除核数时，前若干个核各多处理 1 个向量（4.1） |
| `vecsPerCore` | tilingdata 字段 | 每核负责的向量数（4.1） |
| `matDtype` | tilingdata 字段 | rotation_matrix / qjl_matrix 的元素类型（float32 / bfloat16），dtype 扩展新增 |
| `outDtype` | tilingdata 字段 | quant_norm / quant_gamma 的元素类型（0 = bfloat16，1 = float16），dtype 扩展新增 |
| `CUBE_TILE_V` / `CUBE_MMAD_M` | 编译期常量 | 64 / 64（每 tile 向量数与 AIC 单次 Mmad 的 m，A2 上 m 上限为 64；拆成两个常量是为了 tile 尺寸扫描时不改 Mmad 形状） |
| `CUBE_SUBBATCH` / 幂次表宽度 | 编译期常量 | 32（每 tile 子批向量数）/ 1024 元素（4.2） |
| `tilingkey` / 平台信息缓存 | 派生量 / 静态变量、compileInfo | 200 + mse_bits（4.3）/ AIC 核数与各级容量（4.6） |

### 4.6 platform 查询与缓存

`PlatformAscendC` / `GetCoreNumAic` / `GetCoreMemSize` 在本环境为 ms 级开销，而 tiling 与 TilingPrepare 每次下发都会执行，故只查一次并缓存在静态变量与 compileInfo 中。分核与容量校核**一律使用查询结果**，不在代码中硬编码具体型号的核数；平台信息拿不到时不得用猜测值继续（属环境不满足前提）。查询项与用法见 6.2。

### 4.7 输出 shape 推导

`quant_idx` 末维 = `ceil(128 × mse_bits / 8)`（mse_bits = 2/3/4 时分别为 32 / 48 / 64）；`quant_qjl` 末维固定 16（`ceil(qjl_dim / 8)`，qjl_dim = head_dim = 128）；`quant_norm` / `quant_gamma` 为 [num_tokens, num_kv_heads]，元素类型由 attr `out_dtype` 决定（0 = bfloat16、1 = float16，均 2 B/element）。推导结果参与 host 侧的输出校验（4.4 第 ⑥ 项），也是 workspace 大小的计算依据（4.8）。

### 4.8 aclnn 工程化接口与 workspace

按 ops-transformer 仓 experimental/attention 目录既有算子规范：

- 对外暴露标准两段式接口 `aclnnKvCacheTurboQuantGetWorkspaceSize(...)`（完成输入校验、输出 shape/dtype 推导、tiling 计算，返回 workspace 大小与 executor）与 `aclnnKvCacheTurboQuant(workspace, workspaceSize, executor, stream)`；
- **workspace 不为 0**：混合路径的 MIX 内核 codegen 会在 AIC 上执行 `matmul::clearWorkspace`，workspace 必须 > 0；另有中间量中转需求。声明量为 `WS_HEAD(64 MB) + 4 × 128 × 128 × 2 B（H/S 的 hi/lo，共 128 KB） + N × 128 × (4×2 B + 2×4 B)（u/r 的 hi/lo 与 y/proj，共 N×2048 B） + 32 MB 余量`。64 MB 头部保留区用于避开框架在 workspace 头部的系统区/清零区；32 MB 余量来自交付 host tiling 代码注释记录的实测约束（框架把 kernel 侧 workspace 指针垫在用户区之后，而 host 拿到的总量只比声明量多一部分，声明量必须大于布局尺寸，否则布局尾部会越界写并踩坏输出张量）；
- 输入校验：kv_vectors 为 bf16 三维且末维 = 128，rotation_matrix / qjl_matrix 为 fp32 [128,128]，attr mse_bits ∈ {2,3,4}，输出 shape 与推导值一致；任一不满足按仓内 aclnn 规范返回对应错误码，不自定义错误码体系；
- 仅支持 ND 格式连续 tensor；非连续输入的转连续由调用方完成，不在算子内隐式拷贝。

---

## 5. Kernel 设计（required）

### 5.1 总架构与数据流（双流水 + 双缓冲）

**总架构**：交付实现为 AIC/AIV 混合（MIX）内核，AIV 负责「归一化 + 量化链 + 打包 + 输出」，AIC 负责两次 GEMM，两侧通过 GM workspace 中转数据、CrossCore 标志握手；流程不再是「CopyIn → Compute → CopyOut 单流水」，而是**双流水 + 双缓冲十字同步**。各相位与 3.1 步骤的对应关系：prep = 步骤 1/2、GEMM1 = 步骤 3、quant = 步骤 4/5/6/7、GEMM2 = 步骤 8 的投影、qjl = 步骤 8 的符号判定与打包；四个相位两两之间都有跨核数据依赖，因此"AIC 与 AIV 的相位交错"是并行的唯一可利用维度。

**数据流（每核，tile t，双缓冲通道 d = t % 2）**：

```
Init（一次性）：
  AIV(subIdx=0)：H/S 逐 16 行块读入 → hi = RNE(x), lo = RNE(x − hi) → GM（hHi/hLo/sHi/sLo）
  AIC：hHi/hLo/sHi/sLo Nd2Nz → L1 常驻；AIV 侧同时建好打包幂次表
  ── 之后 AIV→AIC SetFlag(HS)，AIC 开始循环 ──

tile t：AIV 的 prep(t+1) 与 AIC 的 GEMM2(t) 重叠执行
  AIV  prep：kv →(DataCopy+Cast)→ fp32 → 平方和两级归约 → norm → u = x·invNorm
             → u 拆 hi/lo(bf16) → GM ubuf                    → SetFlag(U_READY + d)
  AIC  GEMM1：WaitFlag(U_READY+d) → Nd2Nz(u_hi) → LoadA/LoadB →
             Mmad(hi@hi, init) + Mmad(hi@lo) → Nd2Nz(u_lo) → Mmad(lo@hi)
             （三次 Mmad 链式累加，L0C 保持 fp32）→ Fixpipe float→float ND 直出 GM y
                                                              → SetFlag(Y_READY + d)
  AIV  quant：WaitFlag(Y_READY+d) → 读 y → 量化 idx/ŷ → r = y − ŷ → ‖r‖ 归约 → γ
             → r 拆 hi/lo(bf16) → GM rbuf                     → SetFlag(R_READY + d)
  AIC  GEMM2：WaitFlag(R_READY+d) → 同构 GEMM（权重换成 S，输入换成 r）→ GM proj
                                                              → SetFlag(P_READY + d)
  AIV  qjl：WaitFlag(P_READY+d) → proj ≥ 0 → qjl bit → 打包输出
```

### 5.2 同步与双缓冲机制

- 跨核标志用 `CrossCoreSetFlag/CrossCoreWaitFlag`（MODE2），flag ID 取 4~8：`HS`（AIV0→AIC：H/S 拆分完成）、`U/Y/R/P`（四个阶段）。`U/Y/R/P` 均带 dbIdx 后缀 `+ d`（d = t % 2），即**每阶段两级 ping-pong**，使 tile t 与 t+1 的不同相位可以并行。方向语义：AIC→AIV 为广播，AIV→AIC 按 sub-block 计数（两个 AIV 都 set 才算完成），因此 H/S 拆分只由 AIV0 做、仍需两个 sub-block 联合 notify；
- 跨核 wait 挂在真正消费该 GM 数据的流水上（**PIPE_MTE2**）：AIC 侧由 Nd2NzTile/Nd2NzWeights 消费 u/H/S/r，AIV 侧由 QuantSubbatch 消费 y、P 阶段消费 proj。若把 wait 挂在 PIPE_FIX 之类的空流水上，等待形同虚设，会把对端尚未写完的 y/proj 读进来，表现为部分子批结果错且跨轮次不确定；
- 核内流水复用同步（`HardEvent`）同样是正确性关键：UB 的 MTE2 覆写必须等上一轮 V 读完（V_MTE2）、L1 的 a1 覆写必须等上一轮 LoadA(MTE1) 读完（MTE1_MTE2）等，缺失时会出现「数据只落地一部分」的隐蔽错（历史上表现为 H 暂存只装了第一个 16 行块、y 从第 16 列起错）；
- 预准备：tile 0 的 u 在循环外先算好；循环内 tile t 的量化/flush 完成后再准备 tile t+1 的 u（与 AIC 的 GEMM2(t) 重叠）。prep 必须放在本轮 flush 之后——prep 会写 norm 缓存，提前会覆盖本轮待 flush 的 norm。

### 5.3 AIC 侧 GEMM 细节

- 权重常驻：H/S 在 kernel Init 由 AIV0 一次性 fp32→bf16 cast 并拆 hi/lo 写 GM，AIC 端 Nd2Nz 进 L1 后全程常驻，循环内不再搬权重；
- 输入装载：每轮按**满 CUBE_MMAD_M = 64 行**装载与计算（尾部不足 64 行的 tile 会多读相邻有效数据，落在 GM 内、不影响结果），只有 Fixpipe 出图用真实行数。这是「小 n 结果错」的修复手段：让 Cube 侧永远走已验证过的 m=64 形状，而不是 m=16 这类未验证的形状；
- Mmad 参数与出图：m = 64、n = 128、k = 128，bf16 输入 / fp32 累加，`k` 向按 LoadData2D（compressor 式）分块装载，A2 上单次 Mmad 的 m 上限为 64；L0C→UB 在 A2 不支持，出图改用 `Fixpipe<float, float, CFG_NZ>(gm, l0c, params)` 以 float→float 直出 GM ND 格式；
- 每轮 A1/B2 复用前后都插入 MTE1_MTE2 / MTE2_MTE1 同步，避免 A 面读到新旧混合数据。

### 5.4 AIV 侧计算链（fp32）

- **prep（kv → norm → u）**：DataCopy 输入 → Cast 到 fp32 → 平方 → 行内平方和用**两级 `BlockReduceSum`**（stage1 每 32 B datablock 内 8 元素求和、`repeatTime = elems/64`；stage2 再把 8 个 block 和折成每行 2 个半行和，`repeatTime = (elems/8 + 63)/64`，**必须向上取整**，见 5.5(4)）→ 每行 1 次标量读 + 加法 + Sqrt 得 norm，`invNorm = 1/max(norm, 1e-30)`，`u = x · invNorm`（Muls，与 golden 的 `x / max(norm,1e-30)` 等价，x_norm = 0 时输出全零）→ hi/lo 拆分 → GM；
- **量化链**（子批宽度一次算完，指令数 ÷ 子批行数）：`Duplicate(idx = 0)`、`Duplicate(ŷ = cent0)`；对每个边界 k：`CompareScalar(mask, y, b_k, GT)`（边界值用标量比较，省掉一次广播）+ `Duplicate(dup, k+1)` + `Select(idx, mask, dup, idx)`，**复用同一 mask** 再 `Duplicate(dup, cent[k+1])` + `Select(ŷ, mask, dup, ŷ)`（即 ŷ = cent0 + Σ Δ_k·[y > b_k]，省掉整条按索引查表的比较链）；随后 `r = y − ŷ`；‖r‖ 用与 norm 相同的两级归约；γ = norm × ‖r‖；
- **QJL**：`Duplicate(0)` + `Duplicate(1)` + `Compare(proj, 0, GE)` + `Select`，一次子批宽度完成（取代原逐行判定）；
- **bit 打包**：idx 用「向量乘幂次表 + `BlockReduceSum`」实现——`Mul(idx, 2^(BITS·(i%8)))` 后按 32 B datablock 归并，恰好等于「8 个元素各占 BITS 位的字段合成一个字」，再用标量按小端字节序写出；QJL 同理（表为 2^(i%8)）。mse_bits=4 时 4 bit × 8 = 32 bit 超出 fp32 的 24 bit 整数精度，改用双掩码（0x0F…/0xF0…）分别归并低/高半字。幂次表固定 1024 宽常驻 UB，打包按 1024 元素分块调用。注意：**不能用 4/2/1 元素偏移的 `Add` 树做 8 元素组内归约**——那些偏移是 16 B/8 B/4 B，违反矢量指令的 UB 32 B 对齐要求（整核崩溃）；`BlockReduceSum` 的 32 B datablock 语义正好绕开该限制；
- **输出与 UB 别名**：idx/norm/γ 每子批 flush（`DataCopyPad` 单块搬出），qjl 每子批 flush；norm/γ 搬出前 Cast 为 bf16（round-to-nearest-even，与 golden 的 bf16 转换一致；`out_dtype = 1` 时改为 fp16 分支，复用同一暂存缓冲）；每个 AIV 只写自己那一半子批的位置。UB 别名：prep 相位的 xf 与 u 分别别名到 y 与 qjl 缓冲（prep 相位与 quant/qjl 相位不重叠），另有 xin→uHi、proj→y、tmpFp→ŷ、stBk→ŷ 四处别名；这些复用是 `CUBE_SUBBATCH` 从 16 提到 32 的前提（容量校核见 4.2）。

### 5.5 精度对齐策略（bf16x3 / bf16x4 实验）

**（1）为什么用 bf16x3 拆分，而不是 bf16 直用**

- Cube 只吃 bf16 输入（本机 fp32 Mmad 不可用，真机证据见 6.3），bf16 只有 8 bit 尾数。若直接 bf16 直用（只做 hi@hi 一次 Mmad），`y` 与 `proj` 只保留 8 bit 有效尾数，量化边界与 QJL 符号的翻转率会显著放大到不可接受（交付代码注释与开发记录中记录了这一组对照结论；该组数据未收入自测报告，正式稿如引用需先复核）。
- **bf16x3 拆分**：把 fp32 加速器输入按 `hi = RNE(x)`、`lo = RNE(x − hi)` 拆成两个 bf16，GEMM 展开为 `hi@hi + hi@lo + lo@hi`（三次 Mmad 链式累加，L0C 保持 fp32），有效尾数恢复到约 16 bit。代价是 AIC 时间约 3 倍，但 AIC 不是关键路径，**精度换不到性能损失**，故为交付选择。
- 若只把 proj 换成 bf16x3、y 用 bf16 直用也无改善：qjl 位由 y 的精度决定，不由 proj 决定。因此 y 与 proj 两条 GEMM 都保持 bf16x3。

**（2）bf16x4 实验结论（已评估，未采用）**

给 GEMM 再补一项 `lo@lo`（四次 Mmad）后实测：`y` 与 fp32 参考的最大偏差 2.3e-6 → 1.7e-6，idx 差异 25 → 16 byte/786432；但 prefill/mse2 的 10× 余量从 4.3%/2.5% 掉到 3.0%/1.1%（余量本就只有个位数百分比），收益与风险不成比例，**最终保留 bf16x3**。剩余误差的主要来源也不是拆分精度，而是 Cube 内部 fp32 累加顺序与「k 升序逐项累加」不同源。

**（3）与 golden.py 的 bit 级校核方法与结论**

- **方法**：先把测试程序的 host 参考实现（与 golden.py 同算法：逐行 fp32、`k` 升序顺序累加、`Compare + Select` 量化、残差归一化后 QJL 投影、`_pack_bits` 同款位打包）在真机上与任务方 `golden.py`（torch 2.7.1 + torch_npu 2.7.1.post4 + ml_dtypes 0.6.0）逐 bit 比对：使用任务方 `rotation_matrix.bin` / `qjl_matrix.bin`，输入 bf16（value_range=[-2,2] 往返），覆盖 T=512 × 8 heads × {2,3,4} bit × 3 seed。再用该参考实现对交付 kernel 做 9 case 对拍（7.3.1）。
- **结论一**：host 参考实现与 `golden.py` **同 bit**——quant_norm 0/4096 不一致、quant_gamma 0/4096 不一致、quant_idx 0~1 byte/262144、quant_qjl 0~12 byte/65536（qjl 的极少数差异是 `S·r̂` 恰好取 0 的 fp32 末位掷点）。⇒ 只要 kernel 保持「逐行 fp32、k 升序累加」的语义（纯矢量路径即如此），就与 golden.py 同 bit。
- **结论二**：交付走的混合路径为了 10× 性能放弃了「k 升序」这一累加顺序，代价是与 fp32 参考在**末位**上不再逐位一致；量级为 idx ≤25 byte/786432、qjl ≤194 byte/262144、gamma ≤8/16384 行差 1 ULP，**norm 不经过 GEMM，逐位一致 0/16384**；decode 与边界 shape 仍逐位一致（说明混合路径的数据通路、staging、workspace、打包均正确）。

**（4）关键缺陷与修复：`BlockReduceSum` 第二级 repeatTime 整除截断**

- 现象：官方三个大 shape 各出现 48 行坏数据（16 个核各自末尾 3 行），`y` 误差达 24（y 本身只有 ±0.25）⇒ 整行垃圾而非精度问题，`idx_diff ≈ 2308/786432`、`norm`/`gamma` 各约 48/53 行错，而同 shape 里行数能整除的子批完全干净。
- 根因：`norm`/`γ` 的行内归约第二级 `repeatTime` 原写作 `elems/8/64`（整除截断）。当子批行数 n 不是 4 的倍数时（尾子批 19 行 → elems = 2432 → `2432/512 = 4`，应为 5），尾部若干行的平方和没有被写出，残留上一轮的值 ⇒ 该行 norm/γ 全错 → u 错 → y 错 → idx/qjl 连带错，且只在尾子批触发（典型尾块缺陷）。
- 修法：第二级 `repeatTime` 改为 `(elems/8 + 63)/64` 向上取整（prep 的 norm 与 quant 的 γ 两处）。修复后 prefill 的 **idx 差异 2308 → 10、qjl 821 → 72、norm 48 → 0、gamma 53 → 5，relMSE 0.043 → 0.040**，`y` 误差回到 2.3e-6 量级。

### 5.6 性能结论：为何必须用 Cube

- 纯矢量路径的真机实测：prefill_t2048 = 4462 µs（10× 目标 385.5 µs，差 11.6×）。矢量单元无法在预算内完成 2 × 16 K MAC/向量；
- 混合路径把两次 GEMM 搬到 AIC 后：prefill_t2048 = 369.2 µs，官方 5 shape 全部 ≥ 10×（7.3.4）；
- 关键路径在 **AIV 的量化相位**而不是 AIC：把 GEMM 从 bf16x3 降为 bf16 直用换不到可观收益且精度不可接受（5.5），因此 Cube 侧精度不做减法；反过来把量化链向量化（边界标量比较 + mask 复用 + BlockReduceSum 打包）才是主要收益来源；
- 结论：选型由性能下界决定，Cube 为必需项；Cube 路径引入的代价是「与 fp32 参考不再逐位一致」，以任务书容差判据衡量可接受。

### 5.7 大规模与小规模优化

- **小规模（decode / 小 head 数）**：N ≤ 32 时只启动 `min(ceil(N/64), 20) = 1` 个核，单 tile 单 Mmad 链；AIC 与 AIV 都按满 64 行装载、由 Fixpipe 只写真实行数（5.3），使小 shape 不必引入独立的形状分支；核数取整、tile 覆盖全 N，因此不存在"空核需特判"的路径。
- **大规模（prefill，N = 16384）**：每核约 13 个 tile，收益全部来自相位重叠（5.1/5.2）与子批宽度（5.4）——"每核处理更多 tile"并不改变代价模型，每 tile 的串行相位长度才是决定项（4.1 的代价模型）。
- **参数扫描结论（真机实测）与实现取舍**：tile 尺寸 32 → 64 → 128 的扫描显示 tile 大小不是杠杆（128 无收益且显著拖慢小 case），`CUBE_SUBBATCH` 8 → 16 → 32 才是有效方向（每子批固定屏障数减半），故最终配置为 `CUBE_TILE_V = 64`、`CUBE_SUBBATCH = 32`，大 shape 与小 shape 共用同一配置；不保留纯矢量 kernel（首版原型 v3 已从交付代码移除，tilingkey 仅剩 202/203/204），原因一是任务书 §5 要求不含冗余内容，二是两条实现会让"逐位一致"与"性能达标"两套判据分裂（3.7、7.9）。

### 5.8 调试打印纪律

kernel 编译单元内**禁止出现 `AscendC::printf` / `DumpTensor`**：CANN 只要看到 printf，就会为每次下发分配并初始化 dump-tensor workspace（每核 1 MB 块），实测使 host 侧 `aclnnKvCacheTurboQuant` 调用被固定阻塞约 **14 ms/launch**（与负载、声明量、kernel 类型均无关）。交付代码用 `KVPRINT` 宏承载调试打印，`KV_TQ_ENABLE_PRINT` 未定义时展开为空实现。

---

## 6. 硬件与约束（required）

### 6.1 支持硬件

| 支持的芯片版本 | 涉及勾选 |
|---|---|
| Atlas 800T A2 | √ |

注：任务简介曾写 A3，任务方已答复按任务书 Atlas 800T A2 执行（见 7.10），本文档与该口径一致。

**实测环境表**（真机自测与验收复现所用环境；固件等未记录项以现场 `npu-smi info` 输出为准）：

| 项目 | 取值 |
|---|---|
| 芯片型号 / 架构 | Atlas 800T A2（本机 SoC 标识 `910B3`），`dav_c220` / NPU 架构 2201 |
| 单卡核数 | AIC 20（`GetCoreNumAic` 查询值，见 6.2）；AIV 40（MIX_AIC_1_2 映射：AIV 数 = 2 × blockDim，blockDim = AIC 数 = 20，见 4.1） |
| CANN / 驱动 | CANN 9.1.0（`~/Ascend/ascend-toolkit`，任务书 §3.1 要求 9.1.0+）、驱动 25.5.0；固件版本本次自测未记录 |
| 开发环境 / 工具链 | 远端开发机 DevEnvC_z1k5x（单卡，32 vCPU / 245 GB 内存）；Ascend C 编译器与 opbuild（随 CANN 9.1.0）、cmake 3.20.5、gcc 9.4 |

单卡形态即验收形态：本算子的分核、UB/L1 容量校核都在单卡单进程内完成，不涉及多卡通信（6.2 的核数与容量全部来自平台查询）。

### 6.2 平台资源获取方式

- **不硬编码型号资源、只查一次**：AIC 核数与 UB/L1/L0C 容量一律通过 `PlatformAscendC` 查询（`GetCoreNumAic` 取核数、`GetCoreMemSize` 取各级片上容量），不写死具体型号的核数常量；因 tiling 与 TilingPrepare 每次下发都会执行、而平台查询在本环境为 ms 级开销，故只在首次查询并把结果缓存在静态变量与 compileInfo 中（4.6），后续下发复用缓存值；
- **用途与失败处理**：核数用于分核上界（4.1），容量用于 4.2 的 UB/L1/L0C 预算校核与 6.3 的约束核对；平台信息不可用时不得用猜测值继续（属环境不满足前提，按仓内规范报错返回），避免以"看起来能跑但资源假设错误"的方式静默降级。

### 6.3 A2 Cube 与搬运资源约束

| 约束项 | 实测/查证结论 | 对设计的影响 |
|---|---|---|
| fp32 Mmad | **910B 上不可用**。真机 plog 报 `InitInstMMadSocSpecAbility convert failed, inst str: f32f32f32`（配套 `CQE errType=0x20 / sqSwStatus=0x23` 设备异常中止） | Cube 输入只能 bf16；fp32 语义靠 bf16x3 拆分 + L0C fp32 累加恢复（5.5） |
| 单次 Mmad 的 m | 上限 64（`CUBE_MMAD_M = 64`） | tile 内一次 Mmad 对应 64 个向量；tile 尺寸超过 64 时需内部分两轮（4.1、5.3） |
| L1 常驻量 | H/S 的 hi/lo 四份 NZ 矩阵各 32 KB，共 128 KB 全程常驻 | 权重常驻换掉每轮搬运；剩余 L1 给 A/B 装载缓冲（4.2） |
| L0C 容量与出图路径 | C1 = 64×128×4 B = 32 KB fp32 累加区，与 m=64 / n=128 严格对应；**L0C → UB 在 A2 不支持** | 出图改 `Fixpipe<float, float, CFG_NZ>` float→float 直出 GM ND 格式（5.3） |
| LoadData2D 分块 | fract 恒为 512 B；bf16 的 K 向 `srcStride = mAlign/16`（通用式 `mAlign/16·(sizeof(T)/2)`） | `k` 向分块装载的 stride 换算依据（5.3） |
| UB 访问对齐与容量 | 矢量指令的 UB 地址必须 32 B 对齐；UB 用量需 ≤ 可用容量——`CUBE_SUBBATCH = 32`（`SUB_ELEMS = 4096`）在未做别名复用时 9 个用例全挂（溢出） | 组内归约必须用 `BlockReduceSum` 的 32 B datablock 语义、禁用 4/2/1 元素偏移的 `Add` 树（5.4）；子批尺寸受 UB 硬约束，必须靠别名腾挪（4.2、5.4） |
| MIX 内核运行守卫与 workspace | MIX 内核必须在 AIV 侧提前 return（`if ASCEND_IS_AIV return` 类守卫），否则 AIV 执行 Cube 指令导致设备崩溃（`CQE 0x23`）；codegen 会在 AIC 执行 `matmul::clearWorkspace`，要求 workspace > 0 且声明量必须大于实际布局尺寸 | kernel 入口的核角色分流（5.1）；workspace 声明公式见 4.8 |

### 6.4 API 使用依据表

| 操作 | 采用的 Ascend C 接口 | 必须满足的约束 |
|---|---|---|
| 平台资源查询 | `PlatformAscendC` / `GetCoreNumAic` / `GetCoreMemSize` | 只查一次并缓存（4.6、6.2） |
| 类型提升与输出转换 | `Cast`（bf16 → fp32；fp32 → bf16 RNE；`out_dtype = 1` 时 fp32 → fp16） | 舍入方式须与 golden 的 bf16 转换一致（3.2、5.4） |
| 行内平方和 / 残差范数归约 | `BlockReduceSum`（两级） | stage2 `repeatTime` 必须向上取整 `(elems/8 + 63)/64`（5.5(4)） |
| 量化边界比较与选择 | `CompareScalar` + `Duplicate` + `Select` | 边界用标量比较省广播；同一 mask 复用于 idx 与 ŷ 两条链（5.4） |
| 位打包（idx / qjl） | `Mul`（乘 `2^(BITS·(i%8))` / `2^(i%8)` 幂次表）+ `BlockReduceSum` | 依赖 32 B datablock 语义；mse_bits=4 用双掩码分半字（3.4、5.4） |
| 结果搬出 | `DataCopyPad` | 每子批一次单块搬出；每 AIV 只写自己的半批（5.4） |
| GEMM 计算与权重装载 | `Matmul` 相关接口（`Nd2Nz` 转权重/输入、`LoadData2D`、`Mmad`、`Fixpipe`） | m ≤ 64、bf16 输入 + L0C fp32 累加、L0C 不支持到 UB（5.3、6.3）；H/S 一进 L1 即常驻，每轮 A/B 复用前后插 MTE1_MTE2 / MTE2_MTE1（5.2、5.3） |
| 跨核与核内同步 | `CrossCoreSetFlag` / `CrossCoreWaitFlag`（MODE2）、`HardEvent`（V_MTE2、MTE1_MTE2、MTE2_MTE1 等） | AIC→AIV 广播、AIV→AIC 按 sub-block 计数、wait 须挂在真正消费数据的流水上（5.2）；UB/L1 复用的覆写—读取必须成对插入 |

### 6.5 算子约束限制

- head_dim 首版固定 128，不支持其它 head_dim（质心表维度相关）；
- num_kv_heads 取值范围 [4, 32]（host 侧硬校验）；
- attr 取值约束：mse_bits 仅支持 2/3/4（默认 3）、out_dtype 仅支持 0/1（默认 0），其余取值 host 侧直接返回参数错误、不进 kernel；
- 仅支持标准 MHA/GQA KV cache，不支持 MLA latent cache（1.4）；仅实现 encode（量化），decode/内积估计在推理侧由调用方使用编码结果完成（3.6）；
- rotation_matrix / qjl_matrix 为推理期固定输入，算子不生成随机矩阵；两者必须是**正交矩阵 / 固定高斯投影分布**，这是调用方契约，kernel 不做正交性或分布校验；
- rotation_matrix / qjl_matrix 支持 float32 与 bfloat16 两种元素类型（运行时校验两者同 dtype）。官方 op.json/case.json 的 fixtures 为 float32，为默认口径；bfloat16 输入时 kernel 侧 hi 分量即原值、lo 分量补零（bf16 矩阵天然无低位分量），数学语义与「以该 bf16 值为精确矩阵」的参考一致；
- 仅支持 ND 格式连续 tensor（见 4.8 aclnn 工程化接口）；无输入输出别名、无原地（in-place）写、不做广播（1.3）；
- 混合路径依赖非零 workspace（见 4.8），调用方必须按 `GetWorkspaceSize` 返回值申请；
- kernel 内不得使用 `printf` / `DumpTensor`（见 5.8），交付代码以 `KVPRINT` 宏在编译期关闭。

### 6.6 特性交叉分析

| 场景 / 特性 | 影响与处理 |
|---|---|
| A2 + CANN 9.1.0 组合 | 9.1.0 的 opbuild 要求 def 层使用 `DataTypeList/FormatList` 新 API（旧 `DataType/Format` 写法读不到 dtype、编不过）；运行期与 9.0.0 兼容（7.11） |
| MHA / GQA 与 num_kv_heads 范围 | num_kv_heads 4~32 由 host 校验；kernel 内展平为 N，不感知 head 维语义，因此 MHA（H = num_heads）与 GQA（H = num_kv_heads）走同一条路径（4.1、6.5） |
| dtype 组合与标量输出类型 | 矩阵 {f32, bf16} × 输出 {bf16, fp16} 共 4 种组合共用同一 kernel：矩阵 dtype 只影响 hi/lo 拆分（bf16 时 lo 补零），输出 dtype 只影响搬出前的 Cast 分支；norm/gamma 由同一个 `out_dtype` 统一控制、类型始终一致，搬出前按 RNE 转换（3.2、4.7、5.3、5.4） |
| 固定 H / S 矩阵 | 权重在 Init 阶段一次性拆分与转 NZ，循环内常驻 L1；矩阵身份不进入计算逻辑，只影响数值结果（5.3、7.7） |
| mse_bits 与可变 Q（qjl_dim） | 位宽只进入 tilingkey 与打包路径（202/203/204）；qjl_dim 在本算子内固定等于 head_dim = 128（末维 16 B），不随 bits 变化（3.4、4.3） |
| MLA 与 3.5 bit/channel | 均不在首版范围：MLA 需要独立接口与融合路径，3.5 bit 需要 channel 分组与混合精度分配（1.4） |
| 动态 shape、零向量与极小范数 | 核数与 tile 数随 N 变化，尾 tile/尾子批按 5.3、5.5(4) 的口径处理，不存在跨 token 依赖（任意 T 都不需要重算标定，4.1）；归一化保护 `max(norm, 1e-30)` + `where` 语义，零向量的 idx 落在固定中间索引、γ = 0（3.5） |
| workspace 与框架系统区 | 声明量含 64 MB 头部保留区与 32 MB 余量，避免越界写踩坏输出张量（4.8） |

---

## 7. 可维可测分析

### 7.1 精度标准 / 性能标准

| 验收标准 | 描述(不涉及说明原因) | 标准来源 | 当前状态 |
|---|---|---|---|
| 位布局往返一致 | idx / qjl 的 bit order（每 8 维一组、lane 0 占最低位）与小端字节序须与 `golden.py` 的 `_pack_bits` 逐字节一致；跨行不合并 | 任务书 §2.3 存储格式设计、`golden.py`、`case.json` | 已通过：9/9 用例的打包口径一致；大 shape 的字节差异来自 GEMM 末位而非打包（7.3.1、7.3.3） |
| 数值对拍容差 | idx 差异率 ≤ 2e-2、qjl ≤ 1e-2、norm/gamma 按 bf16 ULP ≤ 2（本机自测判据） | 自测判据；`case.json` 的 `err_threshold` 为更严的逐位口径（idx `[0,1e-5]`、qjl `[0,1.1e-5]`、norm/gamma `[0,0]`） | 已通过：9/9 用例（7.3.1）；两个口径的差异与取舍见 7.3.3、7.9 |
| 统计质量（相对 MSE） | 量化-反量化相对 MSE（主重构）< 0.05 | 任务书 §3.2 | 已实测：3-bit 0.0404~0.0407、4-bit 0.0110 达标；2-bit 档 0.05 数学不可达，按 Lloyd-Max 下限判定（0.1298 vs 下限 0.1175，7.3.2） |
| 统计质量（内积） | 内积估计相对误差 95th percentile < 0.1 | 任务书 §3.2 | 已实测：0.030~0.091 全部达标（7.3.2） |
| 性能 | 相对功能等价 PyTorch eager TurboQuant reference 加速 ≥ 10×，编码不成为 prefill 瓶颈 | 任务书 §3.3 | 已通过：官方 5 shape 全部达标，10.2×~41.5×（7.2、7.3.4） |
| 内存安全 | workspace 不越界写、不踩输出张量；无输入输出别名、无原地写；任务书 §3.4 的内存指标不涉及 | 任务书 §3.4（不涉及）；4.8 的 workspace 声明约束 | 已通过：声明公式含 64 MB 头部保留区与 32 MB 余量，9/9 用例输出正确（4.8、7.3.1） |

### 7.2 官方用例与性能目标换算表

口径：50 次连续下发 + 末尾一次同步（与任务书 §3.5 的 PyTorch eager 基线同口径），取每次下发平均时间；10× 目标 = 官方基线 / 10。

| case | num_tokens | num_kv_heads | head_dim | mse_bits | eager baseline(µs) | 10× 目标(µs) | 实测(µs) | 倍率 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| gqa_decode_b1 | 1 | 8 | 128 | 3 | 1491.676 | 149.2 | 35.95 | ≈41.5× |
| gqa_decode_b64 | 64 | 8 | 128 | 3 | 1552.784 | 155.3 | 55.71 | ≈27.9× |
| gqa_prefill_t2048 | 2048 | 8 | 128 | 3 | 3855.422 | 385.5 | 369.59 | ≈10.4× |
| gqa_mse2_t2048 | 2048 | 8 | 128 | 2 | 3607.963 | 360.8 | 352.85 | ≈10.2× |
| gqa_mse4_t2048 | 2048 | 8 | 128 | 4 | 5286.502 | 528.7 | 476.45 | ≈11.1× |

### 7.3 真机实测结果与证据

测试环境见 7.11；精度输入用任务方 `rotation_matrix.bin` / `qjl_matrix.bin`，其余输入按 case.json 的 `value_range=[-2,2]` 确定性生成并做 bf16 往返。完整日志：`results/kvtq_selftest_377a0591_20260924.log`（`9 passed, 0 failed`，退出码 0）；交付主口径日志与证据清单见 7.8。

**7.3.1 精度：9/9 用例通过**。判据：idx 差异率 ≤ 2e-2、qjl ≤ 1e-2、norm/gamma 按 bf16 ULP ≤ 2；差异数为「字节数」（1 个元素翻转只影响 1 字节）。

| case | T | H | mse_bits | idx 差异 | qjl 差异 | norm 差异 | gamma 差异 (ULP) | 判定 |
|---|---|---|---|---|---|---|---|---|
| gqa_decode_b1 | 1 | 8 | 3 | 0 | 0 | 0 | 0 | 逐位一致 |
| gqa_decode_b64 | 64 | 8 | 3 | 0 | 2 byte | 0 | 0 | 通过 |
| gqa_prefill_t2048 | 2048 | 8 | 3 | 10 byte | 72 byte | 0 | 5 行（2 ULP） | 通过 |
| gqa_mse2_t2048 | 2048 | 8 | 2 | 10 byte | 82 byte | 0 | 2 行（2 ULP） | 通过 |
| gqa_mse4_t2048 | 2048 | 8 | 4 | 25 byte | 194 byte | 0 | 8 行（2 ULP） | 通过 |
| perf_v4_small | 128 | 8 | 3 | 1 byte | 6 byte | 0 | 0 | 通过 |
| edge_zero_vec | 4 | 4 | 3 | 0 | 0 | 0 | 0 | 逐位一致 |
| edge_heads32 | 8 | 32 | 3 | 0 | 0 | 0 | 0 | 逐位一致 |
| edge_heads4 | 8 | 4 | 3 | 0 | 0 | 0 | 0 | 逐位一致 |

（三个大 shape 的 idx 差异率 10/786432 ≈ 1.3e-5、25/786432 ≈ 3.2e-5；qjl 82/262144 ≈ 3.1e-4、194/262144 ≈ 7.4e-4。`edge_zero_vec` 为全零/置零向量保护用例。）

**7.3.2 任务书 §3.2 官方精度指标**。解码按任务书 §2.3 步骤 3~8：`x̂_mse = ‖x‖·Hᵀŷ`，`x̂_qjl = ‖x‖·Hᵀ(ŷ + (√(π/2)/d)·‖r‖·Sᵀs)`（s = 2b−1）；内积误差取 200 个确定性随机**单位** q 的 95th percentile（官方 fixtures 与 QJL 常数）。

| case | mse_bits | relMSE（主重构） | 任务书阈值 0.05 | Lloyd-Max 理论下限 | relMSE（含 QJL） | 内积 ip95 | 内积 ipRms | 阈值 0.1 |
|---|---|---|---|---|---|---|---|---|
| gqa_decode_b1 | 3 | 0.0407 | 达标 | 0.03454 | 0.0663 | 0.043 | 0.026 | 达标 |
| gqa_decode_b64 | 3 | 0.0404 | 达标 | 0.03454 | 0.0636 | 0.059 | 0.028 | 达标 |
| gqa_prefill_t2048 | 3 | 0.0404 | 达标 | 0.03454 | 0.0638 | 0.055 | 0.028 | 达标 |
| gqa_mse2_t2048 | 2 | 0.1298 | 见下（不可达） | 0.1175 | 0.2079 | 0.091 | 0.045 | 达标 |
| gqa_mse4_t2048 | 4 | 0.0110 | 达标 | 0.009497 | 0.0174 | 0.030 | 0.014 | 达标 |

- **bits=2 档 0.05 不可达（数学结论）**：2-bit 标量量化的 Lloyd-Max 均方误差下限为 0.1175·σ²，而单位球面向量逐维 σ² = 1/d ⇒ 相对 MSE 下限 0.1175 > 0.05，**任何 2-bit 标量量化器都达不到 0.05**。本算子实测 0.1298 为该编码形式（固定质心表）的近最优值，故按「达到理论下限量级」判定。
- **内积**：用官方 `qjl_matrix`（元素方差 ≈ 1.0）时 QJL 解码常数 √(π/2)/d 成立，实测 95th percentile 0.030~0.091 **全部 < 0.1**，RMS 相对误差 0.014~0.045。（若换用自造 `U(-4,4)` 矩阵，同一常数会偏大 2.3×，指标不可比——自测必须用任务方 fixtures。）
- QJL 项按论文语义只用于**内积估计**：把它加进重构不会降低 MSE（含 QJL 的 relMSE 反而升高，见上表），故 MSE 用主重构判定。

**7.3.3 与参考实现的差异溯源（均已定位，非缺陷）**：① **Cube 的 fp32 累加顺序**——`y = u·Hᵀ`、`proj = r·Sᵀ` 在 AIC 上以 bf16x3 + L0C fp32 累加实现，与 host 参考的「k 升序逐项累加」不是同一加法顺序，`y` 最大偏差 **2.3e-6**（y 量级 0.1，约 2e-5 相对），由此引起的量化边界翻转即 idx 10~25 byte/786432、qjl 72~194 byte/262144；② **gamma 的 1~2 ULP**——γ = ‖x‖·‖r‖，‖r‖ 由 r 的两级归约得到，bf16 输出在极少数行跨过舍入边界（8/16384 行），**norm 不经过 GEMM、逐位一致（0/16384）**；③ **decode 与边界 case 逐位一致**（含全零向量、num_kv_heads = 4/32），说明混合路径的数据通路（staging、workspace 中转、打包）正确，差异只来自 GEMM 数值路径；④ 大 shape 与 `golden.py` 不是逐位一致是**已知取舍**——若验收坚持逐位比对，需切回纯矢量 fp32 路径，而该路径 prefill 为 4462 µs、达不到 10×（见 7.11 已知限制、7.9）。

**7.3.4 性能：官方 5 shape 全部达标**。口径：50 次连续下发 + 末尾一次同步（与任务书 §3.5 的 PyTorch eager 基线同口径），取每次下发平均时间；基线取任务书表格，10× 目标 = 基线/10。

| case | 官方基线 (µs) | 10× 目标 (µs) | 实测 BATCH (µs) | 加速比 | 判定 |
|---|---|---|---|---|---|
| gqa_decode_b1 | 1491.676 | 149.2 | 35.95 | ≈41.5× | 达标 |
| gqa_decode_b64 | 1552.784 | 155.3 | 55.71 | ≈27.9× | 达标 |
| gqa_prefill_t2048 | 3855.422 | 385.5 | 369.59 | ≈10.4× | 达标 |
| gqa_mse2_t2048 | 3607.963 | 360.8 | 352.85 | ≈10.2× | 达标 |
| gqa_mse4_t2048 | 5286.502 | 528.7 | 476.45 | ≈11.1× | 达标 |

- prefill/mse2 贴近阈值（余量约 4%/2%）；同一台机器多次运行有 **±1~3% 漂移**（本任务 CANN 9.1.0 多轮实测：prefill 369.06/369.21/369.34/369.59 µs、mse2 352.43/352.68/352.73/352.85 µs、mse4 476.10/476.25/476.45/476.53 µs），验收机上建议重复 2~3 次取中位；
- 补充数据（非官方 shape，仅作过程记录）：`perf_v4_small`（T=128, H=8, 3 bit）57.8 µs；纯矢量参考路径的 prefill 为 4462 µs（已被混合路径取代，仅用于说明 Cube 路径的收益）；
- 从纯矢量到交付版的优化节点（依次）：① 消除 CANN dump-workspace 的 14 ms/launch 固定开销（kernel 内不出现 `printf`）；② 量化链向量化（`CompareScalar` 边界比较、掩码复用、`BlockReduceSum` 位打包）；③ AIV 的 prep/量化与 AIC 的 GEMM 流水重叠（双缓冲十字同步）；④ `CUBE_SUBBATCH` 8→16→32（**每子批的固定屏障数**才是真正的瓶颈）；⑤ UB 别名腾挪以容纳 32 子批。

### 7.4 计时口径

- **同源比较**：性能数字必须在**同一台设备、同一份输入、同一组 H/S 矩阵、同一个 mse_bits** 下测量，禁止跨环境或跨 fixtures 拼接加速比（`qjl_matrix` 的来源会改变内积指标，见 7.3.2）；
- **预热、迭代与同步**：按任务书 §3.5 的口径，50 次连续下发 + 末尾一次同步，取每次下发的平均时间；基线（1491.676 / 1552.784 / 3855.422 / 3607.963 / 5286.502 µs）取自任务书表格，10× 目标 = 基线 / 10；计时区间以显式同步结尾，确保测量包含设备端完成时间；harness 同时输出单次下发与 BATCH 平均值，用于区分"冷启动/L2 未热"与稳态；
- **口径边界与漂移处理**：本文报告的数字为**设备端 encode**（50 次下发平均、含 launch 与 host 侧接口开销的稳态值），**不含**输入数据生成、矩阵回拷与结果比对的开销；若要单列"含 host 与 launch 的端到端"，须使用同一 harness 的对应输出，不与设备端数字混用。同机多次运行有 ±1~3% 漂移（7.3.4），贴近阈值的 shape（prefill、mse2）应重复 2~3 次取中位后再判定，不取最好值。

### 7.5 测试用例规划矩阵

按测试场景分类的规划矩阵（每行含本轮已执行用例与规划用例；"规划，未执行"表示本轮交付未覆盖，需在验收或后续版本补测；Shape 列形如 [T,H,128]）：

| 测试场景分类 | 用例描述 | Shape 及属性 | 数据类型 | 预期结果 |
|---|---|---|---|---|
| 主码本 / 位宽边界 | 已执行：三种位宽各一例 `gqa_mse2_t2048` / `gqa_prefill_t2048` / `gqa_mse4_t2048`。规划：decode 小 shape 走 2-bit 与 4-bit | [2048,8,128]，mse_bits = 2/3/4；规划项 [1,8,128] 与 [64,8,128] | bf16 输入、u8 输出、bf16 标量 | 已执行：三例通过（7.3.1），relMSE 与内积达标（7.3.2）。规划项用于确认小 shape 下打包与末维 32/64 的正确性（未执行） |
| 范数与符号边界 | 已执行：`edge_zero_vec`（全零/置零向量）。规划：极小范数（1e-30 量级）与零残差（y 恰落质心）构造用例 | [4,4,128] / 规划项 [8,8,128]，mse_bits = 3 | bf16 | 已执行：逐位一致（3.5 的零向量语义）。规划项期望：`max(norm,1e-30)` 保护下无 inf/nan；零残差时 QJL 组按"≥0 记 1"记全 1（未执行） |
| Shape 与尾部 | 已执行：H 上下限与 T = 1/64 —— `edge_heads4` / `edge_heads32` / `gqa_decode_b1` / `gqa_decode_b64`。规划：行数不均分核、"尾子批行数 %4 ≠ 0"、head_dim 扫描 Q = 1/7/8/9/127/128/129/257 | [8,4,128] / [8,32,128] / [1,8,128] / [64,8,128]；规划项 N 取核数非整数倍、尾子批 19 行、Q 按上列取值 | bf16 | 已执行：四例基本逐位一致（decode_b64 有 2 字节 qjl 差，7.3.1）。规划项：尾子批场景已被三个大 shape 间接覆盖（5.5(4)）；Q ≠ 128 期望 host 侧参数错误（6.5、3.4）（未执行） |
| 矩阵语义 | 已执行：官方 fixtures（float32）与矩阵 bf16 变体。规划：Hadamard 与非对称随机正交矩阵对照，验证旋转不依赖 H 元素分布 | [128,128]，f32 / bf16 | f32、bf16 | 已执行：默认 f32 口径 9/9，`KVTQ_MAT_DTYPE=bf16` 变体 9/9（7.11 已知限制③）。规划项未执行 |
| 输出安全 | 已执行：实际输出字节长度（idx 末维 32/48/64、qjl 16）。规划：输出守护区（越界写检测）与跨核相邻行边界 | 三种 mse_bits；规划项在每个输出张量尾部留守护区 | u8 / bf16 | 已执行：shape 与字节率与 4.7 的推导一致。规划项未执行 |
| 接口负例 | 非法 rank / dtype / mse_bits / out_dtype / num_kv_heads / head_dim / 两矩阵 dtype 不一致 / 非连续输入 / 空指针 / workspace 不足 | 逐项构造 | 各类型混合 | 规划，未执行：期望 host 侧返回对应错误码、不进 kernel（2.3、4.4） |
| 路由等价 | 三个 tilingkey（202/203/204）分别与 golden 对拍 | mse_bits = 2/3/4 | bf16 | 已执行：202 与 204 由 `gqa_mse2/mse4_t2048` 覆盖，203 由 decode b1/b64 与 prefill 覆盖（4.3、7.3.1） |

（`perf_v4_small`（T=128, H=8, 3bit，每核 1 tile）为过程用用例，见 7.3.4。）

测试代码与交付件组织：交付件按 ops-transformer 仓 experimental/attention 目录规范组织——算子代码目录含 `README.md`（产品支持情况 / 功能说明 / 参数说明 / 调用说明 / 贡献说明五个必备章节）及 `docs/aclnnKvCacheTurboQuant.md` 接口说明文档；测试代码与复现步骤见 `examples/test_aclnn_kv_cache_turbo_quant.cpp` 与 `examples/README.md`，满足验收人可复现要求（任务书交付件 2、4）。

### 7.6 可维护性与 DFX

- **分层与单一职责、常量单一来源**：host 侧（算子原型与 shape 推导、输入校验、tiling、aclnn 两段式接口）与 kernel 侧（AIC/AIV 相位）分离，校验只在一处（host）、kernel 不做重复防御分支（4.4）；质心表、`HEAD_DIM`、`QJL_BYTES` 与位打包规则集中在公共头 `kv_cache_turbo_quant_common.h`，host 与 kernel 共享同一份定义，避免常量漂移（3.8）；
- **不为 case 复制 kernel、阶段可追踪**：5 个官方用例的差异全部由 tiling 参数与 tilingkey 吸收（唯一变化维度是位宽），不存在"一个 shape 一份实现"的维护负担（4.3、4.5）；kernel 的六个计算阶段与跨核标志一一对应（`HS` / `U` / `Y` / `R` / `P`，见 5.1、5.2），定位问题时可从"哪个 GM 中间量不一致"直接映射到相位；
- **可诊断、日志可控、文档规范**：workspace 中的中间量（u/r 的 hi/lo、y、proj）按布局可回读，用于把"整行缺陷"（如 5.5(4)）与"末位差异"（7.3.3）区分开，自测 harness 保留矩阵来源、y 误差量级与坏行清单等诊断输出（7.8）；kernel 内调试打印统一走 `KVPRINT` 宏、默认编译期关闭（5.8），避免误引入 14 ms/launch 的固定开销；README 五章节 + `docs/` 接口文档 + `examples/` 复现说明齐全，tiling 字段集中在 4.5 说明，新增位宽只需加一条 tilingkey 与一个质心表实例。

### 7.7 兼容性分析

**结论：本算子为新算子，不改动任何既有算子的接口、注册与行为，不涉及既有算子的兼容性问题**（首版文档的结论）。但作为 KV cache 编码格式的生产者，它与下游消费端之间存在必须显式约定的格式契约：

1. **ABI 层面**：新增算子与新增 aclnn 两段式接口，不修改仓内既有算子的原型、tiling 或 kernel；除 `out_dtype`（可选、有默认值）外不新增 ABI 属性（2.2），旧调用形式（仅传 `mse_bits`）行为等价，属**向后兼容的新增**。构建期存在版本差异（9.1.0 的 opbuild 只认 def 层 `DataTypeList/FormatList`），运行期兼容（7.11）。
2. **与下游缓存元数据的契约**：4 个输出只有在与元数据绑定的前提下才能被解码——元数据至少须记录 `head_dim = 128`、`mse_bits`、H/S 矩阵版本（同一层内 K 与 V 可以使用不同矩阵，须分别记录）、质心表版本（位宽对应 3.8 的三张表）。任何一项不一致都会让解码结果系统性错误，且不会以"报错"的形式暴露。
3. **与既有 INT8 cache 布局的关系**：本算子输出的 `quant_idx` / `quant_qjl`（bit-packed uint8）**不自动兼容**框架既有 INT8 cache（C8：per-channel scale + offset）的分页布局与 attention kernel。接入 vllm-ascend 需要为该编码格式单独定义分页/块布局与配套的解码（反量化 + 内积估计）路径，属后续工作，不在本算子范围内。
4. **与其它 TurboQuant 系算子的互通**：与下游消费端（含 TurboQuant 系的 attention / score 类算子）互通时，需要独立约定三件事——idx 与 qjl 的 bit order 与小端字节序（3.4）、QJL 的符号约定（`>= 0` 记 1，而非 sign 的 0）、以及 `quant_gamma` 的标度（原尺度，已含 ‖x‖）。这三项没有"从格式自解释"的可能，必须写进接口约定（3.6）。

### 7.8 证据链与日志

每轮自测应逐 case 记录：输入 dtype 与 shape、矩阵文件身份（路径 + SHA256）、四个输出的 shape 与 dtype、与 golden 的比对结论、退出码、代码 commit、脚本版本与日志路径；未执行项一律显式标注"未执行"，不以推测补全（本版此类项见下表与 7.5）。

| 证据类别 | 应记录内容 | 本次交付的实际情况 |
|---|---|---|
| 精度 + 性能主日志 | 每 case 的输入 dtype/shape 与四输出 shape/dtype、`[ACC]` 指标（relMSE / ip95 / ipRms）、`[PERF]` BATCH 均值、PASS/FAIL、汇总 `9 passed, 0 failed`、退出码 | 已记录：7.3 引用的 `results/kvtq_selftest_377a0591_20260924.log`（输出 shape 亦可由 4.7 推导公式核对）；CANN 9.1.0 真机主口径日志及其 SHA256 见《KvCacheTurboQuant 算子自测报告_20260924》与 `results/` 目录 |
| 与 golden.py 的位级校核 | host 参考实现 vs 真机 golden.py 的逐位差异计数（norm/gamma bit 位、idx/qjl 字节数） | 已记录：`results/goldenpy_bitcheck_20260924.log`（结论见 5.5(3)） |
| 冷克隆端到端复现 | 从交付分支 clone → build.sh → 装包 → 编 harness → 跑，得到 9 passed / 0 failed | 已记录：`results/kvtq_coldclone_377a0591_20260924.log` |
| 矩阵文件身份与 SHA256 | H/S fixtures 的路径与哈希 | **未执行**：日志中记录了 fixtures 目录来源与"使用任务方 fixtures"标记，但未逐文件记录 SHA256；若验收要求归档，需补记 |
| 代码 commit 与脚本版本 | 交付 commit、自测脚本与复现脚本标识 | 已记录：私仓分支 `feature/kv-cache-turbo-quant`：算子目录 @ `377a0591`（clang-format 规范化 + 许可证头统一后的交付版本），分支 tip `d8ddff99`（含 `task_submission/` 七件套）；详见《自测报告_20260924》；测试代码随交付分支提供（7.5 末） |

### 7.9 风险与降级预案

| # | 风险 | 影响 | 降级/应对预案 |
|---|---|---|---|
| 1 | 大 shape 与 `golden.py` 不是逐位一致（Cube 的 fp32 累加顺序与「k 升序」不同源，idx ≤25 byte/786432、qjl ≤194 byte/262144、gamma ≤8/16384 行 1 ULP），而 `case.json` 的 `err_threshold` 对 norm/gamma 是 `[0,0]` | 若验收按逐位自动比对，大 shape 不满足 | 可切回"逐行 fp32、k 升序累加"的纯矢量路径（与 golden 同 bit），但该路径 prefill 实测 4462 µs，达不成 10×（3.7、5.6）——两个目标不可同时满足，需按 7.10 第 2 条明确判据优先级；另一条未验证的方向是改用"k 升序外积累加"的矢量 GEMM 同时满足同序与性能。追加 `lo@lo`（bf16x4）已验证无效：idx 25→16 byte 但性能余量掉到 3.0%/1.1%（5.5(2)），剩余误差来源是累加顺序，需换累加顺序而非加项 |
| 2 | 性能余量偏小（prefill 约 4%、mse2 约 2%）且同机有 ±1~3% 漂移 | 验收机上可能贴线或略微越线 | 用同一脚本重复 2~3 次取中位（7.4）；若仍贴线，继续压缩每子批固定屏障数或提高子批宽度（受 UB 硬约束，见 4.2），不改变数值路径 |
| 3 | 零向量/零残差语义与消费端不一致 | 消费端按 sign 语义解码会出现系统性偏差（bit 反相） | 已在 3.5、3.6 显式约定「严格大于边界」与「≥0 记 1」两条规则；解码侧须按同一口径实现 |
| 4 | 平台/版本差异（非 A2 硬件、不同 CANN 版本） | Cube 约束（fp32 Mmad 可用性、L0C→UB 支持、UB 容量）与 opbuild API 可能不同 | 资源与能力一律走平台查询、不硬编码（6.2）；换平台需重新核对 6.3 的约束表与 6.6 的交叉分析；平台查询不可用时按规范报错，不用猜测值继续 |
| 5 | workspace 声明量不足 | 布局尾部越界写并踩坏输出张量（历史上曾导致 norm 大面积错） | 按 4.8 的公式声明（含 64 MB 头部保留区与 32 MB 余量）；调用方必须使用 `GetWorkspaceSize` 返回值 |

### 7.10 与任务方确认的开放问题

硬件口径已确认：任务简介写 A3、任务书写 A2，**任务方答复按 A2 执行**，本文档 6.1 节与之保持一致。以下为开放式问题及当前状态：

| # | 任务书缺失项 | 影响 | 当前状态 / 建议口径 |
|---|---|---|---|
| 1 | case.json 引用的 rotation_matrix.bin / qjl_matrix.bin 未随任务下发，生成规则（种子、正交化方式、分布缩放）未说明 | 正式验收环境数据无法本地复现 | **已部分解决**：自测已取得任务方 `rotation_matrix.bin` / `qjl_matrix.bin` 并全程使用（日志首行 `[MAT] 使用任务方 fixtures`），全部精度指标基于同源 fixtures；仍需任务方在验收机上提供 case_path 下的 fixtures 目录 |
| 2 | 重构误差 MSE 未定义是逐维均方还是整向量平方误差、未说明重构是否计入 QJL 残差，且未说明判定粒度是样本均值还是逐向量判定 | 分子口径相差 128 倍（直接决定 2-bit 配置是否达标）；判定粒度不同会让同一实现得出完全不同的结论 | 建议明确为「整向量平方误差 / ‖x‖² 的样本均值」（与任务书公式记号一致，3-bit 实测 0.0404 与 Lloyd-Max 理论下限 0.03454 同量级）；按论文，MSE 重构仅用主编码、QJL 只用于内积无偏修正（计入 QJL 的 relMSE 反而升高，见 7.3.2）；bits=2 档阈值请按数学下限 0.1175 修正。**逐向量最坏值本地未测**，若验收按逐向量判定，建议放宽为达标率（如 ≥95% 向量 < 0.05） |
| 3 | 内积误差 P95 < 0.1 未提供 decode 测试程序与 query 生成规则 | 随机独立 query 真值近零、相对误差分母失真 | 我方已自建 decode + 内积估计参考实现（随交付件提交）：单位 q 口径下实测 ip95 = 0.030~0.091（mse_bits 2/3/4），全部达标；请任务方确认 query 生成规则与归一化口径 |
| 4 | 性能验收未提供 PyTorch eager baseline 的原始测速脚本 | 计时口径（warmup、同步方式、是否含 launch 开销）不明，10× 加速比无法对齐复现 | 已用 case.json 附的基线微秒数（1491.676 / 1552.784 / 3855.422 / 3607.963 / 5286.502）按任务书 §3.5 口径（50 次连续下发 + 末尾一次同步）自写脚本复现，官方 5 shape 全部达标；正式口径仍需索取原始脚本 |
| 5 | 任务书 §3.1 要求 CANN 9.1.0+ | 验收环境版本差异 | **已解决**：自测与冷克隆复现均已在 CANN 9.1.0 真机完成（9/9 精度 + 5/5 性能达标）；9.1.0 的 opbuild 只认 def 层 `DataTypeList/FormatList` 新 API（旧 `DataType/Format` 写法读不到 dtype、编不过），交付代码已按新 API 适配 |

### 7.11 复现方式与环境

- **硬件/软件**：Atlas 800T A2（本机 `910B3`，dav_c220 / NPU 架构 2201，单卡 20 AICore），CANN 9.1.0，驱动 25.5.0；远端开发机 DevEnvC_z1k5x。算子包构建与安装：

  ```bash
  source ~/Ascend/ascend-toolkit/set_env.sh
  bash build.sh --pkg --experimental --soc=ascend910b --ops=kv_cache_turbo_quant
  ./build_out/cann-ops-transformer-custom_linux-aarch64.run --install-path=<install-path> --quiet
  ```

- **自测程序**：`examples/test_aclnn_kv_cache_turbo_quant.cpp`（编译命令行、`KVTQ_MAT_DIR` 指向任务方 fixtures 的方式见 `examples/README.md`）；一次运行即完成 9 个 case 的精度对拍与官方 5 shape 的性能计时，输出 `[PASS]/[FAIL] ... [ACC] ... [PERF] ... BATCH avg`，末尾汇总 `9 passed, 0 failed`。
- **已知限制**：① 大 shape 与 `golden.py` 非逐位一致（末位量级，判据为任务书容差，全部达标）；② prefill/mse2 性能余量偏小（约 4%/2%），受机器漂移影响，建议多轮取中位；③ dtype 变体另测通过：rotation/qjl_matrix 支持 float32/bfloat16、quant_norm/gamma 支持 bfloat16/float16（新增可选属性 `out_dtype`，默认 bf16），`KVTQ_MAT_DTYPE=bf16` 与 `KVTQ_OUT_DTYPE=fp16` 两组各 9/9 通过。

---

## 8. 参考资料

1. **论文**：TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate（ICLR 2026 投稿），arXiv:2504.19874 —— 本文两阶段量化（MSE 主量化 + QJL 残差修正）与质心表来源。
2. **设计文档模板**（任务书 §4 交付件 1 指定）：https://gitcode.com/cann/cann-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md
3. **任务与提交规则**：社区任务流程及注意事项 https://gitcode.com/org/cann/discussions/39 ；任务列表与设计文档提交路径 https://gitcode.com/cann/cann-competitions/tree/master/04_tasks/01_community-task-2026/tasklist
4. **目标仓库与合入规范**：ops-transformer 仓 https://gitcode.com/cann/ops-transformer （合入 `experimental/attention`）；贡献指南 `https://gitcode.com/cann/ops-transformer/blob/master/CONTRIBUTING.md`；仓内同类算子样例见 `experimental/` 目录。
5. **Ascend C 文档与接口参考**：Ascend C 算子开发文档、算子开发接口文档、Ascend C 在线课程（链接见任务书 §6 参考资料）。
6. **本任务配套材料**：`kv_cache_turbo_quant算子开发任务书.md`、`golden.py`、`op.json`、`case.json`、`test_data/`；《KvCacheTurboQuant 算子自测报告_20260924》与 `results/` 下的自测日志（证据链见 7.8）。

---

## 附录：修订记录

| 日期 | 版本 | 修改描述 |
|---|---|---|
| 2026-09-23 | v1 | 首版（PR #1825） |
| 2026-09-24 | v2 | 适配 CANN 9.1.0 与 dtype 扩展（矩阵 float32/bfloat16、标量输出 bfloat16/float16，新增可选属性 `out_dtype`；num_kv_heads 4~32 硬校验；删除纯矢量原型路径，交付走 tilingkey 202/203/204 混合 Cube 路径）；按官方检视意见重构章名层级（需求背景 / 需求分析 / 算子分析 / Host 与 Tiling 设计 / Kernel 设计 / 硬件与约束 / 可维可测分析），并补充 需求背景、需求分析、硬件与约束、可维可测 四章内容（含参数能力表、支持硬件表、精度/性能标准表） |
