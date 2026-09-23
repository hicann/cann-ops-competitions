# KvCacheTurboQuant 算子设计文档

> 状态：设计阶段，算子尚未实现或在 A2 上执行精度与性能测试。本文以随任务提供的 `kv_cache_turbo_quant算子开发任务书.md`、`op.json`、`case.json`、`golden.py` 为接口和自测依据。
>
> 基线：2026-09-23 核对 `cann/ops-transformer` `357ab966258e948cc3f7f5e0b6097df7634c12d2`，`cann/cann-ops-competitions` `147b47c40325065e550eb91250e99af836e46333`。上游后续变化需要重新核对。本文的性能数字均为任务目标或由其换算的上限，不是实测。

# 需求背景（required）

## 需求来源

本任务要求在 `cann/ops-transformer` 的 `experimental/attention` 类目增加 Ascend C / aclnn 算子 `KvCacheTurboQuant`，在线压缩标准 MHA/GQA 的 K 或 V 向量。输入为 `[num_tokens, num_kv_heads, 128]` 的 BF16 张量，独立预生成的正交旋转矩阵和高斯 QJL 投影矩阵；输出为主编码、残差符号编码、原向量范数及原尺度残差范数。首版覆盖 2/3/4-bit MSE 主编码，默认 3-bit。验收硬件为 Atlas 800T A2，CANN 9.1.0+。

任务书指定的旧 `cann/cann-competitions` 模板链接目前跳转至 [`cann/cann-ops-competitions` 的 2026 设计模板](https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md)。[2026 社区任务 README](https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/README.md) 要求将 `docs/design.md` 放入任务与队名目录。[本任务详情页](https://www.hiascend.com/activities/task-center/details/6d3df013343845608677c85f3c58d4d6)没有展示独立数字编号，因此目录沿用任务书标题 `9月社区任务-kv_cache_turbo_quant算子开发/PzeYui/docs/design.md`。2026-09-23 的[活动页](https://www.hiascend.com/developer/activities/details/11014a50a8794171a4a08688fd398774#tab2)列出本任务最晚提交验收日为 11 月 10 日，并要求设计文档 PR 审核通过且合入后再提交验收。进展登记与评论通知需分别按活动流程执行。

任务详情页的简述写有 **A3、3.5 bit/channel**，但同一页面下载的任务书 §2.1、§3.1、`op.json` 和五个 `case.json` 用例明确为 **A2、统一 3-bit 主编码加 1-bit QJL、含两个 BF16 标量共 4.25 bit/channel**。本文以可执行的任务书和测试附件为准；该页面简介差异列入待确认项，不能将其解释为接口已经改为 A3/3.5-bit。

## 背景介绍与上游现状

TurboQuant 论文将正交旋转后的向量做 MSE 标量量化，再用 1-bit QJL 编码主重构的残差，以改进内积估计。论文的 `TurboQuant_prod` 在总位宽 `b` 下使用 `(b-1)` bit 主编码和 1-bit QJL，并保存残差范数；对非单位向量还需保存原向量范数。[原论文第 2.2、3.1、3.2 节](https://arxiv.org/html/2504.19874)是数学依据。本任务指定的是统一 3-bit 主编码加 1-bit QJL，两个 BF16 标量后为 4.25 bit/channel；论文实验中 3.5 bit/channel 的混合配置不是本算子的接口。

在上述 `ops-transformer` 基线中，未发现 `KvCacheTurboQuant`。已有 [`TurboQuantSparseFlashAttention`](https://gitcode.com/cann/ops-transformer/blob/master/experimental/attention/turbo_quant_sparse_flash_attention/README.md) 和 [`TurboQuantSparseAttnSharedkv`](https://gitcode.com/cann/ops-transformer/blob/master/experimental/attention/turbo_quant_sparse_attn_sharedkv/README.md) 面向 MLA / shared-KV 的 512 维 latent、4-bit 码本及不同存储格式，不能作为本任务的 128 维 MHA/GQA Encode 实现或其解码器。它们仅为目录规范和 TurboQuant 命名的参考。

接入方式参考同一基线的 `experimental/attention/gen_position_ids_from_mask`：`op_host/*_def.cpp` 注册 `OpDef` 和 A2 配置，`*_infershape.cpp` 通过 `IMPL_OP` 推形，`*_tiling.cpp` 通过 `IMPL_OP_OPTILING` 配置核数、tiling key 与 workspace，`op_kernel/*.cpp` 根据 tiling key 分派；算子目录和 `op_host/CMakeLists.txt` 接入构建。关键符号分别在该算子的 `GenPositionIdsFromMask`、`InferShapeGenPositionIdsFromMask`、`TilingGenPositionIdsFromMask` 与 `gen_position_ids_from_mask` 中。本算子拟新增独立目录，不修改现有 MLA 代码。

# 需求分析（required）

## 需求描述

| 项目 | 首版要求 | 依据 |
| --- | --- | --- |
| 计算 | 对每个 token/head 独立完成归一化、旋转、MSE 编码、残差 QJL 编码 | 任务书 §2.1、`golden.py` |
| 输入 | `kv_vectors` BF16 `[T,N,128]`；`rotation_matrix` FP32 `[128,128]`；`qjl_matrix` FP32 `[M,128]` | `op.json`、任务书 §2.3 |
| 输出 | `quant_idx` UINT8 `[T,N,ceil(128*b/8)]`；`quant_qjl` UINT8 `[T,N,ceil(M/8)]`；`quant_norm`、`quant_gamma` BF16 `[T,N]` | `op.json`、`case.json` |
| 属性 | `mse_bits ∈ {2,3,4}`，默认 3 | `op.json` |
| 场景 | 标准 MHA/GQA，`N=4…32`，`head_dim=128`；不支持 MLA latent | 任务书 §2.3 |
| 精度 | 重构相对 MSE `<0.05`；内积相对误差的第 95 百分位 `<0.1`；另外须通过官方编码输出对照 | 任务书 §3.2、`case.json` |
| 性能 | Encode 相对功能等价 PyTorch eager reference 加速 `≥10×`，并不成为 prefill 瓶颈 | 任务书 §3.3、§3.5 |
| 目标环境 | Atlas 800T A2、CANN 9.1.0+、Ascend C、aclnn 工程 | 任务书 §3.1 |

`M` 由 `qjl_matrix` 首维推导，任务书写明通常等于 128，五个官方用例均为 `M=128`。设计将输出推形与按字节打包推广到正整数 `M`；实际设备资源边界和非 128 的官方验收范围仍需确认。任务书参数表允许旋转/QJL 矩阵 BF16、范数 FP16，但 `op.json` 与所有 `case.json` 用例固定为矩阵 FP32、范数 BF16；首版注册按后者，不自行扩展 ABI。

## 需求拆解

1. 确立与 `golden.py` 一致的码本、边界比较、矩阵方向、零向量及位打包协议。
2. 在 Host 侧校验 dtype、ND、rank、静态维度和属性，推导四个输出 shape，并根据行数及 UB 大小分核。
3. 在 A2 Kernel 中以 FP32 中间结果完成两个矩阵向量投影和范数归约，在单次 launch 中直接写四个输出，不将中间向量写回 GM。
4. 通过随任务提供的五个形状及非法参数、数值边界用例验证编码输出；另建解码/内积验证脚本验证任务书统计指标。
5. 用同一输入、相同矩阵和相同计时口径测 eager baseline 与算子耗时；记录原始日志，不将任务书给出的 baseline 当作本机实测。

# 详细设计（required）

## 算子分析

### 数学公式与编码协议

设一行输入为列向量 `x∈R^128`，`H∈R^(128×128)`，`S∈R^(M×128)`，`b=mse_bits`：

```text
n = ||float32(x)||₂
u = x / max(n, 1e-30)                  (n=0 时 u=0)
y = H u
idx_j = Σ_k 1[y_j > boundary[b][k]]   (严格大于；等于边界取较小编号)
c_j = centroid[b][idx_j]
r = y - c
rho = ||r||₂
gamma = n * rho
z_i = 1[(S (r / max(rho,1e-30)))_i >= 0]
```

码本是 `golden.py:CENTROIDS` 中 2/3/4-bit 各自的常量；边界是相邻码本值在 FP32 下的均值。`H` 和 `S` 作为推理期间固定、由调用方生成和维护的输入张量；Host 校验形状但不逐元素验证正交性、高斯性或有限值。矩阵方向按 `golden.py` 的行向量表达式 `unit @ rotation.T`、`residual_unit @ qjl.T`，不能擅自转置输入矩阵。归一化、旋转、范数和投影均按 FP32 中间精度设计，末尾两个标量转 BF16。

对每连续 8 个主编码 `idx[8g+j]`，形成 `word=Σ_(j=0..7) idx[8g+j] << (j*b)`，依次输出 `word` 的低 `b` 个字节。因此每组 8 维占 `b` 字节，字节内低位对应较小维号。QJL 第 `i` 位存 `z_i`，每 8 位一个字节，`M` 不是 8 的倍数时末字节高位补 0。`>=0` 的符号约定使精确零投影编码为 1。输出张量行间紧密排列；实际设备分配的对齐开销不计入任务书的 payload 压缩率。

零向量不能直接将四个输出全清零：参考实现对 `u=0` 仍执行码本选择和 QJL 投影，但 `quant_norm=0`、`quant_gamma=0`。Kernel 必须执行同一逻辑，确保打包输出与参考实现一致。`kv_vectors` 非有限值、矩阵非正交/非高斯的行为任务书未定义，先按调用方前置条件处理；不能以此宣称输出有效。

### 重构和内积统计的验证定义

Encode 本身只返回四个压缩张量。为自验另写离线解码器，解包得 `c` 和 `s_i=2z_i-1`，计算：

```text
y_hat = n*c + gamma * sqrt(pi/2)/M * S.T @ s
x_hat = H.T @ y_hat
dot_hat(q,x) = q.T @ x_hat
```

这是把论文的 QJL 逆映射用于旋转空间残差后再做逆旋转；当 `M=128` 时与论文的等维式一致。`n`、`gamma` 使用实际存储的 BF16 值进行解码。该解码器仅用于精度自验，不是首版算子输出或额外 ABI。若要与后续 `turbo_quant_attention_score` 直接互通，需独立约定矩阵、bit order、符号和范数布局；当前任务没有提供它的接口，不能预设已经互通。

### 支持形状、存储与数据类型

五个官方用例覆盖 `(T,N,b)=(1,8,3),(64,8,3),(2048,8,3),(2048,8,2),(2048,8,4)`，均有 `M=128`。`T` 为正整数、`N=4…32`、`head_dim=128`，输出 shape 从输入推导，`T*N` 的地址计算使用足够宽的整数避免溢出。当前官方用例没有覆盖 `N=4/32`、非 128 的 `M`、零向量和非 8 整除的 `M`，这些加入开发阶段补充测试。

| `b` | 主码 | QJL（`M=128`） | 两个 BF16 标量 | 总 payload/head | 相对 BF16 128 维 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 32 B | 16 B | 4 B | 52 B | 约 4.92× |
| 3 | 48 B | 16 B | 4 B | 68 B | 约 3.76× |
| 4 | 64 B | 16 B | 4 B | 84 B | 约 3.05× |

以上只比较单个向量的压缩 payload；不包括固定矩阵、分配器对齐、KV page 管理或 attention 中间量。

## 算子实现

### Host 侧设计

拟新增 `experimental/attention/kv_cache_turbo_quant/`，包含 `CMakeLists.txt`、`README.md`、`op_host/kv_cache_turbo_quant_def.cpp`、`op_host/kv_cache_turbo_quant_infershape.cpp`、`op_host/kv_cache_turbo_quant_tiling.cpp/.h`、`op_kernel/kv_cache_turbo_quant.cpp/.h`、`tests/`。参考上游 `gen_position_ids_from_mask` 的 OpDef、推形、tiling 和自动 aclnn 接入，不预设需要手写一套 `op_api`。

`OpDef` 依 `op.json` 注册 3 入 4 出及可选属性 `mse_bits=3`，仅注册 A2 对应 `ascend910b` 产品配置，并启用 aclnn。首版 Kernel 的地址公式以连续 ND 存储为前提；实现时核验生成的 aclnn 接口是否提供连续化，若不提供，则明确要求调用方传入连续张量并对不满足条件的输入报错。Host 校验：

1. 输入、输出描述均存在，`kv_vectors` 为 rank 3，`rotation_matrix`、`qjl_matrix` 为 rank 2；dtype 和 format 与上表一致。
2. `T>0`、`4≤N≤32`、`head_dim=128`，`H` 为 `[128,128]`，`S` 为 `[M,128]` 且 `M>0`；`b∈{2,3,4}`。
3. 输出 shape 分别为 `[T,N,16*b]`、`[T,N,(M+7)/8]`、`[T,N]`、`[T,N]`，所有字节长度及总元素数量经溢出检查。动态形状只在运行时具体化后 tiling。

Tiling 将 `rows=T*N` 视为互不依赖的向量。按平台可用核数和 UB 容量选择每核处理的行组 `B`：decode 小行数优先一行一核；prefill 用多行一组复用矩阵分片。`blockDim=min(coreNum,ceil(rows/B))`，各核按 tile 编号跨步处理，尾组带有效行数。`mse_bits` 可编码到 tiling key，以编译时固定码本长度和打包位宽；`M`、行组大小、矩阵分片行数及输出 stride 写入 tiling data。配置 `workspace=0`；若设备实际 UB 约束使单 kernel 不可行，先缩小矩阵分片/行组，而非默默增设全局中间缓冲。Host 不对 `H`/`S` 的数值逐元素扫描。

### Kernel 侧设计与分核

每个核只写自己负责的完整向量行，避免跨核归约和输出字节写冲突。单次 launch 中依次：

1. 从 GM 将一组 BF16 `x` 搬入 UB，转 FP32，对 128 维做平方和与开方，生成 `n`、`u`。
2. 将 `H` 按行分片搬入 UB；对组内多行复用同一分片，完成 `H@u`。`H` 总量为 64 KiB；是否整块驻留由实测 UB 容量决定，基本方案按分片工作。
3. 对 `y` 按固定 FP32 码本阈值做严格 `>` 比较，生成 `idx`、`c`、`r`；归约出 `rho`，计算 `gamma=n*rho`，归一化残差。
4. 将 `S` 按投影行分片搬入 UB，对组内多行复用，完成 `S@residual_unit`；按 `>=0` 得符号位。
5. 在 UB 内以 8 维为单位打包，按每行独占的 GM 区间写 `quant_idx`、`quant_qjl`；`n`、`gamma` 转 BF16 后写出。

矩阵只读且推理期间保持不变；同一核的搬运、计算和写回按 Ascend C 队列/事件同步，覆盖缓冲前等待前序计算完成。组内保留 FP32 `u/y/r` 和位打包暂存；按 UB 容量缩放 `B` 与矩阵分片，避免在 GM 产生 `y`、`r`。浮点归约树、矩阵乘加顺序及 BF16 舍入需要与 NPU PyTorch `golden.py` 对照验证：整数码字的边界样本可能因 FP32 运算顺序变化而改变，不能仅凭公式相同就宣称逐字节一致。首版优先 FP32 向量计算；若 prefill 性能不足，再在保持编码一致性的前提下评估更高吞吐的矩阵计算路径。

### 性能分析与优化验证

在 `M=128` 时每向量需两个 `128×128` 矩阵向量乘，约 32,768 次乘加；prefill `T=2048,N=8` 共 16,384 向量，约 5.37 亿次乘加。矩阵分片在一核的行组间复用是主要优化点；小 decode 主要受 launch/Host 开销影响，prefill 主要受 FP32 投影吞吐和矩阵从 GM/L2 重复搬运影响。应通过 A2 上的 profiler 分开观察 Host、device、GM 与计算时间，再调 `B`、分片大小和核数。上述工作量是静态估算，不代表可达到 10×。

## 支持硬件

| 产品 | 首版状态 |
| --- | --- |
| Atlas 800T A2（实测 Ascend 910B3） | `DevEnv_004092` 环境已核验，待实机编译、精度与性能验证 |
| A3、MLA 专用路径、其他产品 | 不在本任务首版验收范围 |

2026-09-23 对 HiDevLab A2 环境进行只读核验：`npu-smi 25.2.0` 报告芯片 `910B3`、健康状态 `OK`；`compiler/version.info` 和 `opp/version.info` 均为 `9.1.0`。该检查只确认环境可达及版本，尚未编译、运行算子或测试精度/性能。

## 算子约束限制

- 仅对标准 MHA/GQA 的一个 K 或 V 向量张量编码；K、V 分别调用。不包括 page scatter、Attention 解码、MLA latent、RoPE 分支或混合 3.5-bit 配置。
- 调用方须提供在整个推理生命周期一致的 `H`、`S`，`H` 正交、`S` 来自固定高斯投影；矩阵及码本版本是压缩数据协议的一部分。更换矩阵后旧缓存不可直接复用。
- 首版计算要求连续 ND 输入；非连续张量是由生成的 aclnn 接口连续化还是由调用方先处理，须在实现时核验并写入算子 README。输出必须按推形分配。矩阵内容、输入有限值、跨算子矩阵一致性无法由 Host 静态校验。
- `M` 非 128 时，输出形状和 bit packing 仍有定义，但精度与性能验收需另行确认，不把五个 `M=128` 用例的结果外推。

# 可维可测分析

## 精度标准与验证计划

| 层次 | 方法 | 判定/记录 |
| --- | --- | --- |
| ABI/功能 | 用任务包五个 `case.json` case 对 `golden.py:calc_expect_func` 比较四输出；另测错误 rank/dtype/属性/shape、零向量、边界值、`N=4/32`、`M` 非 8 倍数 | 保留每输出比较结果和官方工具原始日志；`quant_norm`/`quant_gamma` 的任务包阈值为 `[0,0]`，先按严格一致排查，工具对阈值的解释待实测确认 |
| 重构误差 | 用本文的离线解码器，对原始输入与压缩数据解码值比较 | 报告每 bit 宽的 `||x−x_hat||₂²/||x||₂²` 与逐元素 MSE 版本，明确实际验收采用哪一定义；零向量单列 |
| 内积误差 | 固定并记录 query 生成方法、矩阵 seed、样本数，以 `q·x_hat` 对 `q·x` 统计 | 报告任务书相对误差 P95；`q·x≈0` 的分母处理必须经官方确认，另报归一化绝对误差辅助诊断，不自行替代门槛 |
| 量化存储 | 检查 `b=2/3/4` 的 52/68/84 B payload、各打包字节及尾 bit | 以 `golden.py` 的字节顺序为准；记录矩阵和分配器开销，避免把 payload 比率称作端到端显存收益 |

任务包只提供 Encode golden，没有解码器或 query 数据；`case.json` 指向的 `data/rotation_matrix.bin`、`data/qjl_matrix.bin` 也不在当前压缩包中。开发前需取得官方矩阵文件或确认生成方法与 seed，不能用随机区间数代替正交/高斯矩阵。论文给出的 2-bit MSE 主量化理论平均失真约 0.117（单位向量、主码重构），因此任务书统一 `<0.05` 的门槛若覆盖 2-bit 且分子为整向量平方误差，存在明显风险；还需确认该门槛针对的位宽和 `MSE` 定义。QJL 全重构的误差与主码单独重构也不同，测试报告会分列。

## 性能标准与测量口径

| 官方 case | PyTorch eager baseline（任务书，µs） | 10× 加速对应算子上限（µs） |
| --- | ---: | ---: |
| `gqa_decode_b1` | 1491.676 | 149.168 |
| `gqa_decode_b64` | 1552.784 | 155.278 |
| `gqa_prefill_t2048` | 3855.422 | 385.542 |
| `gqa_mse2_t2048` | 3607.963 | 360.796 |
| `gqa_mse4_t2048` | 5286.502 | 528.650 |

上限只是 `baseline/10` 的算术换算。正式性能报告应在同一 A2/CANN 环境、同一张量和矩阵、足够 warmup 与重复次数下分别记录 eager reference 与 aclnn 算子时延、计时范围（含/不含 Host 和 launch）、统计量及 `speedup=t_eager/t_op`；补测 prefill KV 生成流水线是否受阻。任务书没有提供 baseline 测量脚本、重复次数和计时范围，先不宣称已达标。内存验收条目写为“不涉及”，但仍报告输出 payload、workspace=0 的设计目标和实际峰值以排查隐性缓冲。

## 兼容性分析与待确认项

这是新算子，不改变现有 TurboQuant MLA 或传统 KV cache 的 ABI。与后续 Attention 消费算子的互通依赖相同的矩阵、码本、bit order、`gamma` 标度和 QJL 逆映射，需单独集成验证。

1. **任务编号与页面简介**：任务详情页只显示任务标题和 URL 标识，未见独立数字编号；本次目录沿用任务书标题，队名为 `PzeYui`。详情页简介的 A3/3.5-bit 与附件的 A2/4.25-bit 不一致，需由任务方确认；当前设计和自验计划按附件执行。
2. **官方测试矩阵**：`case.json` 引用的两个二进制矩阵文件缺失；需确认获取途径、布局和 seed。
3. **精度口径**：`MSE/||x||²` 的 `MSE` 是逐元素均方还是整向量误差？`<0.05` 是否对 2/3/4-bit 全部适用？重构是否包含 QJL？内积 query 分布及近零分母如何定义？
4. **接口扩展**：`qjl_dim≠128`、矩阵 BF16 或范数 FP16 是否纳入首版验收？当前设计按 `op.json` 的 FP32/BF16 和动态 `M` 处理。
5. **性能口径**：官方 eager baseline 是否含矩阵构造、Host 同步和四输出分配？若仅提供任务书数字，正式报告需将本机复测和任务书数字分列。

## 参考资料

- [任务详情页](https://www.hiascend.com/activities/task-center/details/6d3df013343845608677c85f3c58d4d6)提供的[任务附件压缩包](https://www.hiascend.com/p/resource/202609/25e71624009c46969b4db01afca60aa8.zip)：`kv_cache_turbo_quant算子开发任务书.md`、`op.json`、`case.json`、`golden.py`。
- [TurboQuant 原论文](https://arxiv.org/html/2504.19874)，第 2.2、3.1、3.2 节。
- [2026 官方设计模板](https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md)、[任务目录说明](https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/README.md)、[社区任务流程讨论](https://gitcode.com/org/cann/discussions/39)，2026-09-23 核对。
- [`ops-transformer` 贡献指南](https://gitcode.com/cann/ops-transformer/blob/master/CONTRIBUTING.md)与同仓 `experimental/attention/gen_position_ids_from_mask` 的 Host、tiling、Kernel、CMake 实现，基线 `357ab966258e948cc3f7f5e0b6097df7634c12d2`。
