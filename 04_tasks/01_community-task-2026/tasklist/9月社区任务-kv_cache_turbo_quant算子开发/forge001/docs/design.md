# kv_cache_turbo_quant 算子设计文档

本文说明 kv_cache_turbo_quant 算子在 Atlas 800T A2（Ascend 910B3）上的设计与验证方法。算子语义、精度与性能要求以任务书及任务附件 golden 为准。

# 需求背景

## 需求来源

依据 2026 年 9 月社区任务「kv_cache_turbo_quant 算子开发」任务书（[任务详情](https://www.hiascend.com/activities/task-center/details/6d3df013343845608677c85f3c58d4d6)及任务附件），在 Atlas 800T A2、CANN 9.1.0+ 上使用 Ascend C 实现面向标准 MHA/GQA KV cache 的在线向量量化压缩算子，以 aclnn 接口工程化交付至 `cann/ops-transformer/experimental/attention`；设计文档按 cann-competitions 设计文档模板提交至 tasklist 评审合入。

## 背景介绍

### kv_cache_turbo_quant 算子功能现状分析

当前 vllm-ascend 中 KV cache 压缩方案现状：

| 方案 | 原理 | 局限 |
| --- | --- | --- |
| INT8 静态量化（C8） | per-channel scale + offset | 压缩率仅 2x |
| LSH 哈希（KVComp） | 局部敏感哈希做稀疏选择 | 不压缩存储 |

参考 TurboQuant 论文，本算子以「随机正交旋转 + MSE 最优标量量化」为主编码、「高斯随机投影 + 1-bit 符号量化」为残差修正，实现在线向量量化压缩：head_dim=128、mse_bits=3 时物理存储 4.25 bit/channel，相对 BF16 约 3.76x 压缩。

算子接口（任务书接口定义）：

| 参数名 | 输入/输出 | 描述 | 数据类型 | 维度 |
| --- | --- | --- | --- | --- |
| kv_vectors | 输入 | 待量化的 K 或 V 向量 | bfloat16 | [T, H, 128] |
| rotation_matrix | 输入 | 正交旋转矩阵 | float32 | [128, 128] |
| qjl_matrix | 输入 | 残差投影矩阵 | float32 | [128, 128] |
| quant_idx | 输出 | 主量化编码（bit-packed） | uint8 | [T, H, 16*mse_bits] |
| quant_qjl | 输出 | 残差 QJL 符号编码（bit-packed） | uint8 | [T, H, 16] |
| quant_norm | 输出 | 输入向量 L2 范数 | bfloat16 | [T, H] |
| quant_gamma | 输出 | 原尺度残差范数 | bfloat16 | [T, H] |

存储格式（head_dim=128，mse_bits = b）：

| 组件 | 摊销 bit/channel | 总 bit | 说明 |
| --- | ---: | ---: | --- |
| idx（主量化） | 3.000 | 384 | b-bit lane 交错 bit-packed，16b 字节/head |
| qjl（残差符号） | 1.000 | 128 | 8 符号/字节，16 字节/head |
| x_norm | 0.125 | 16 | bf16 标量 |
| gamma | 0.125 | 16 | bf16 标量 |
| 合计 | 4.250 | 544 | 68 bytes/head vs 原始 256 bytes/head |

# 需求分析

## 需求描述

使用 Ascend C 实现 kv_cache_turbo_quant 算子：对 [T, H, 128] 输入逐向量执行 TurboQuant 两阶段量化编码，输出主量化编码、QJL 残差符号编码与两个范数标量；提供 aclnn 两段式接口（GetWorkspaceSize + 执行）；精度满足任务书重构误差与内积估计误差要求；性能相对功能等价 PyTorch eager reference 加速 ≥10×，量化在线完成、不成为 prefill 瓶颈。

## 需求拆解

1. 功能语义：范数归一化 → 正交旋转 → b-bit 标量量化 → 残差 QJL 符号编码 → bit-packed 输出，全链与任务附件 golden 逐位一致（uint8 输出二进制一致、bf16 输出逐位一致）。
2. 参数校验与错误码：rank / dtype / shape / 属性 / 连续性 / 空指针校验全部前置在 GetWorkspaceSize 阶段，非法输入返回明确错误码（ACLNN_ERR_PARAM_NULLPTR / ACLNN_ERR_PARAM_INVALID）。
3. 精度：量化-反量化重构相对 MSE < 0.05；内积估计相对误差 p95 < 0.1；四输出对 golden 逐位对拍。
4. 性能：5 个官方 case 相对 eager baseline 加速 ≥10×。
5. 工程化：op_api / op_host / op_kernel / tests 标准目录结构；自测步骤可复现。

# 详细设计

## 算子分析

### 数学公式

对每个输入向量 x ∈ R^128（b = mse_bits ∈ {2,3,4}），全链 fp32 计算：

```text
norm = sqrt(sum_j x_j^2)
u = x / max(norm, 1e-30)              # norm = 0 时 u = 0（除零防护）
y = H u                                # 正交旋转（golden 口径 u @ H^T）
idx_j = sum_k 1[y_j > m_k]             # 严格大于：等号落低端码字
                                        # 边界 m_k = (c_k + c_{k+1}) / 2
y_hat_j = c_idx_j                      # C^(b) = {c_0..c_(2^b-1)} 为 MSE 最优码本
r = y - y_hat
residual_norm = sqrt(sum_j r_j^2)
gamma = norm * residual_norm
r_unit = r / max(residual_norm, 1e-30) # 残差范数为 0 时取 0
projected = S r_unit                   # 高斯随机投影（r_unit @ S^T）
qjl_i = 1[projected_i >= 0]            # tie 记 1
```

码本常量 C^(b)（b=2/3/4 分别 4/8/16 个码字，升序）与边界 m_k 以 fp32 逐位同式 `(c_k + c_{k+1}) * 0.5` 预计算后作为编译期常量内嵌于 kernel；数值以任务附件 golden 为唯一真值源，禁止重新推导或算术近似。

打包格式（golden 契约）：

- quant_idx：每 8 个连续维一组，word = sum_i(idx_i * 2^(i*b))；word 按小端序拆 b 个连续 uint8 字节；head_dim=128 → 每 head 16b 字节。
- quant_qjl：8 个符号/字节（lane i → bit i，正/零为 1），每 head 16 字节。
- quant_norm / quant_gamma：fp32 计算结果直接 cast bfloat16 输出。

### 支持数据类型

| 参数 | 数据类型 |
| --- | --- |
| kv_vectors | bfloat16 |
| rotation_matrix / qjl_matrix | float32 |
| quant_idx / quant_qjl | uint8 |
| quant_norm / quant_gamma | bfloat16 |

为唯一合法 dtype 组合（算子定义锁定，输出 dtype 不随输入提升）；全部中间量以 float32 计算。

### 支持形状

| 参数 | 形状 |
| --- | --- |
| kv_vectors | [T, H, 128] |
| rotation_matrix / qjl_matrix | [128, 128] |
| quant_idx | [T, H, 16*mse_bits]（尾维 32/48/64） |
| quant_qjl | [T, H, 16] |
| quant_norm / quant_gamma | [T, H] |

T ≥ 1 动态，H ∈ [4, 32]，head_dim / qjl_dim 固定 128；ND 连续布局，三输入间无广播关系（shape 精确匹配）。

## 算子实现

### 实现方案

**总体架构**：单次 aclnn 调用内部按「预处理 → 旋转 GEMM → 编码 → 投影 GEMM → 符号打包」五阶段 kernel 流水执行，阶段间以流序天然串行，无跨 kernel 同步开销。该拆分由目标平台执行模型决定：Ascend 910B3 AI Core 为 Cube/Vector 分离核（AIC 与 AIV 分置），Matmul 类 Contraction 须由 AIC 承担、归约/量化/打包类向量计算须由 AIV 承担，故按计算类型分段，各段在所在核型上取全量核数并行。

| 阶段 | 核型 | 输入 | 输出 | 职责 |
| --- | --- | --- | --- | --- |
| 1 预处理 | AIV | kv_vectors, rotation_matrix, qjl_matrix | unit, norm(fp32), quant_norm, 预转置矩阵 ×2 | bf16→fp32；逐行 L2 范数（固定顺序配对树归约）与归一化（除零防护）；两个 128×128 固定矩阵预转置为连续行主序 |
| 2 旋转 GEMM | AIC | unit, 预转置 H | rotated | rotated = unit @ H^T（fp32） |
| 3 编码 | AIV | rotated, norm | quant_idx, quant_gamma, r_unit | 边界比较（严格大于）得码字索引；bit-packed 打包；码本查表重构 → 残差范数 → gamma / r_unit |
| 4 投影 GEMM | AIC | r_unit, 预转置 S | projected | projected = r_unit @ S^T（fp32） |
| 5 符号打包 | AIV | projected | quant_qjl | 符号比较（≥0 记 1）→ 8 符号/字节打包 |

设计要点：

- **B 侧矩阵预转置**：两个固定矩阵在预处理阶段一次性转置为连续 ND 形态，GEMM 以无转置形态消费——既规避 B 侧转置形态在目标平台的正确性风险，又使推理期间固定的矩阵全调用复用，转置本身按块分布式并行。
- **中间量托管**：unit / norm / 预转置矩阵 / rotated / r_unit / projected 由 executor 统一分配托管；GEMM 消费侧使用 M 补齐形状、编码/符号段消费侧使用有效形状视图，pad 行不外写。
- **M 侧对齐**：GEMM 的 M = T×H 补齐到 16 对齐（分形对齐要求），pad 行为垃圾数据，仅用于对齐消费。

#### host侧设计：

host 侧完成参数校验、InferShape、各阶段 tiling 计算与 executor 构建；GetWorkspaceSize 阶段仅做校验与 tiling 计算，无数据搬运。

##### 1. 分核策略：
- 向量类阶段（预处理/编码/符号）：以 (t,h) 行向量为最小独立单元，总行数 N = T×H 均分到核，尾核尾块以有效行数收尾；行内 128 维归约在单核内完成、不跨核交换部分和（确定性要求）。
- GEMM 阶段：M = T×H 补齐 16 对齐、N = K = 128，blockIdx → (m 块, n 块) 标准分核 + 尾块；K=128 不做 splitK，保持累加顺序确定。
- 核数一律经平台信息运行时查询，禁止硬编码。

##### 2. 数据分块和内存优化策略：
- 行分 tile 处理，tile 行数按运行时查询的 UB 容量约束选取；fp32 工作平面在 UB 内按阶段原地复用，降低峰值占用。
- 输入队列开双缓冲，使 GM 搬入与计算重叠；输出按阶段完成节奏回写。
- 搬运按 32B 对齐区分：整块对齐部分用批量搬运，尾部不满 32B 的行（如 b=3 的 48 字节行宽、2 字节标量）用带 padding 的搬运接口。
- 小规模 case 的向量算子按有效行数截断发射，不计算无效行。

##### 3. tilingkey规划策略：
- 编码阶段按 mse_bits 属性做编译期模板特化（{2,3,4}，经模板参数机制 + if constexpr 分发），特化内容仅限码本常量表 / 边界数 / 打包位宽 / quant_idx 尾维；不使用已废弃的 tiling key 宏。
- 其余阶段无按属性的编译期分支，单一模板。
- dtype 组合、head_dim、qjl_dim 由算子定义锁定为唯一值，无组合分支可编；T 维动态由运行期 tiling 参数覆盖，无需编译期分支。

#### kernel侧设计：

各阶段 kernel 均为 Init + Process 结构（搬入 / 计算 / 搬出）。数值契约全阶段统一：

- 全部中间量 fp32（bf16 输入入 kernel 即升 fp32）；quant_norm / quant_gamma 以 fp32 结果直接 cast bfloat16（round-to-nearest-even），与 golden 逐位一致。
- 范数归约采用固定顺序的配对平衡树累加，归约顺序与 golden 对齐，保证 fp32 逐位一致且同输入逐次运行可复现。
- 边界比较严格大于（等号落低端码字）；QJL 符号 ≥0 记 1；NaN 参与比较恒为 false，与 golden 防护路径一致。
- GEMM 采用 fp32 全精度配置（host 侧与 kernel 侧配对设置，禁用 HF32 截断），与 golden 的 fp32 matmul 同精度口径。

打包设计：32-bit 打包字在 16-bit 数据域分半组装后按小端拆字节，规避 b=4 时打包字满占 32-bit 符号位的溢出问题；码本重构与输出布局转置经查表（Gather）完成。符号打包利用符号比较结果的位布局与 golden 字节格式逐位一致，直接形成输出字节流。

边界 case（与 golden 一致，不报错）：全零向量（norm=0 → u=0，自然传播为全低端码字、符号取 1）；残差恰为零（r_unit=0）；NaN/Inf 输入按 IEEE 比较语义传播。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas 800T A2 | √ |

## 算子约束限制

- 仅支持标准 MHA/GQA KV cache，不支持 MLA latent cache。
- head_dim / qjl_dim 固定 128；H ∈ [4, 32]；T ≥ 1（不支持空 Tensor）。
- mse_bits ∈ {2, 3, 4}，默认 3；论文的混合 3.5-bit 模式不在首版范围。
- 输入须 ND 连续，非连续输入由调用方先转连续。
- 三输入间无广播关系，shape 精确匹配。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | 量化-反量化重构相对 MSE < 0.05；内积估计相对误差 p95 < 0.1；四输出与任务附件 golden 逐位对拍（uint8 二进制一致 + bf16 逐位一致） | 任务书 3.2 |
| 性能标准 | 5 个官方 case 相对功能等价 PyTorch eager reference 加速 ≥10×；encode 在线完成不成为 prefill 瓶颈 | 任务书 3.3 |

精度测试覆盖：官方 5 case × 多种子采样；合法边界（H=4/32、mse_bits=2/3/4、T 维边界）；极端输入（全零 / NaN / Inf / 大值上溢 / 量化边界 tie / 完美量化）；确定性（同输入重复运行四输出逐位一致）；异常路径（非法 shape / dtype / 属性 / 空指针返回对应错误码）。

标杆 case（下表数值来自任务书，非本实现实测结果）：

| case | num_tokens | num_kv_heads | head_dim | mse_bits | PyTorch eager baseline (µs) | 10× 折算目标 (µs) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| gqa_decode_b1 | 1 | 8 | 128 | 3 | 1491.676 | 149.2 |
| gqa_decode_b64 | 64 | 8 | 128 | 3 | 1552.784 | 155.3 |
| gqa_prefill_t2048 | 2048 | 8 | 128 | 3 | 3855.422 | 385.5 |
| gqa_mse2_t2048 | 2048 | 8 | 128 | 2 | 3607.963 | 360.8 |
| gqa_mse4_t2048 | 2048 | 8 | 128 | 4 | 5286.502 | 528.7 |

性能验收口径：同卡独占采样、预热后多轮计时取中位、kernel-only 与端到端双口径并列披露、计时前先过正确性门；自测数据与复现步骤随代码交付件提交。

## 兼容性分析

新增算子，不涉及兼容性分析。
