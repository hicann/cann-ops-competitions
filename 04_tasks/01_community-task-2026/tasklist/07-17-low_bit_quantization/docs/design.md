# MR-GPTQ：面向 MXFP4 的 LLM 低比特量化算法设计文档

| 项                | 内容                                                                      |
| ----------------- | ------------------------------------------------------------------------- |
| 任务              | 7月社区任务 — 低比特量化算法开发                                         |
| 数据类型          | MXFP4（微缩浮点）；**W4A16 主线 + W4A4 补充，均已达标**（见 §4.4） |
| 目标算法          | MR-GPTQ（Micro-Rotated GPTQ），AMCT 仓未适配算法（进阶项）                |
| 测试模型          | Qwen3-8B（dense）、Qwen3.6-35B-A3B（MoE）                                 |
| 测试集 / 指标     | WikiText2 / PPL（seq_len=4096, block 粒度）                               |
| 精度目标          | delta = ppl_quant − ppl_bf16 ≤ 0.4；量化层(nn.Linear)占比 ≥ 70%        |
| 作者 / gitcode_id | 程逸雷 / asp1r1n1                                                         |
| 版本              | v0.7（W4A4 达标；§3.3 / §4.4 / §4.5 / §8 按实测与最终实现校正）       |

> 交付目录 `experiment/task-book/mr-gptq-mxfp4_asp1r1n1/`；gitcode 账号 `asp1r1n1`，私仓 fork：https://gitcode.com/asp1r1n1/amct 。最终 PR 目标 `cann/amct` 的 `feature/community-tasks` 分支。

---

## 1. 背景与目标

### 1.1 背景

大模型推理的显存与带宽瓶颈主要来自权重与激活的搬运。将 `nn.Linear` 的权重/激活从 BF16 压到 4bit，可把显存占用降低 50% 以上，并在昇腾 NPU 上使能低比特运算。**MXFP4（Microscaling FP4）** 是业界主推的 4bit 浮点格式：每 32 个元素共享一个 2 的幂次（power-of-two, PoT）缩放因子，单元素为 E2M1（1 符号 + 2 指数 + 1 尾数）。相比 INT4，MXFP4 的分段浮点表示对权重的长尾分布更友好，且硬件可原生支持。

但 MXFP4 有一个结构性难点：**共享 scale 被限制为 2 的幂次**，直接量化时 scale 的舍入误差会被整块放大，导致精度显著劣化（这一点是 MR-GPTQ 论文的核心动机）。因此简单的 min-max / round-to-nearest 直转在 MXFP4 上往往不达标，需要专门的算法补偿。

### 1.2 目标

在 AMCT 中实现 **MR-GPTQ** 算法，用于 MXFP4 低比特量化，并在两个测试模型上满足：

- **绝对精度门**：`delta ≤ 0.4`（相对原始 BF16 网络）；
- **量化覆盖门**：被量化的 `torch.nn.Linear` 占比 ≥ 70%；
- **相对收益（进阶项要求"精度优化"）**：在相同 MXFP4 配置下，MR-GPTQ 的 PPL 明显优于仓内已有的 **GPTQ-MXFP4** baseline；
- **可复现**：提供独立评测脚本与完整日志（实施中未走 AMCT 标准 CLI，偏差原因见 §4.5）。

### 1.3 非目标

- 不做 hifloat8（≤0.1 门槛过严，非本任务主线）；hifloat8 仅作为可选附加场景，直接复用仓内 OFMR 现成结果（Qwen3-8B delta≈0.093），不额外开发。
- 不做自定义 NPU 算子；MXFP4 的 cast/量化算子复用仓内 `quantization/dtypes/mxfp_impl.py` 与已有 deploy 算子。
- 不改动 workflow / solver / quant module 主干；算法作为可插拔单元接入。

---

## 2. 现状分析（复用面）

AMCT 仓已具备接入 MR-GPTQ 所需的大部分积木，本方案是"改造 + 组合"，而非从零：

| 已有能力       | 位置                                                                                                            | 在本方案中的角色                                                                                                                                     |
| -------------- | --------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| 算法注册框架   | `amct_pytorch/algorithms/quant/__init__.py` 的 `register_algorithms()` + `registry_factory.ALGO_REGISTRY` | MR-GPTQ 按同样方式注册为`--algos mr_gptq`                                                                                                          |
| 算法接入范本   | `amct_pytorch/algorithms/quant/flatquant.py`                                                                  | 复用其`_kronecker_matmul` / `_random_orthogonal`（→ 改为 Hadamard），及 `forward/trainable_params/export_ptq_params/load_ptq_params` 接口契约 |
| GPTQ 误差补偿  | `amct_pytorch/classic/quantize_op/gptq_module.py`（已支持 MXFP4，见算法矩阵）                                 | 作为 baseline**对照组**，并提供误差补偿实现参考                                                                                                |
| MXFP4 数据类型 | `amct_pytorch/quantization/dtypes/mxfp.py` / `mxfp_impl.py`                                                 | 提供 MXFP4 的 fake-quant / cast                                                                                                                      |
| 全链路 CLI     | `amct_pytorch.eval / extract_ptq_data / ptq / deploy`                                                         | 直接复用，样例见`examples/models/qwen3.6/Qwen3.6-Moe.md`                                                                                           |
| 模型适配       | `common/models/llm/qwen/`（qwen3、qwen3_6_moe 已注册）                                                        | 两个测试模型均已适配，无需新写 adapter                                                                                                               |

**算法支持矩阵佐证**（`docs/zh/algorithm_brief.md`）：GPTQ 已支持 MXFP4 权重量化 → baseline 对照组现成可用；这是本方案"证明有精度提升"的关键前提。

### 2.1 baseline 代码级验证结论（本地无 NPU，已从代码确认链路）

> 目的：在上 NPU 环境前，先从代码确认 `--algos gptq --quant_dtype mxfp` 这条 baseline 链路真实可跑，避免"有提升"缺对照。以下均为代码实证：

- **CLI 合法**：`cli/llm/args.py` 中 `--quant_dtype` 的 `choices=['int','mxfp']`、`--quant_target` 的 `choices=['mlp','moe','attn-linear','attn-cache']`，与本方案命令行一致 ✅。
- **GPTQ 原生支持 MXFP4**：`gptq_module.py:26` 导入 `MXFP4_E2M1`；`:192` 对 MXFP4 走特殊分支（`cal_scale_offset_static` 返回 `None,None`，即 scale 由 per-block（group_size=32）动态计算，不走静态 min-max）✅。
- **MXFP4 微块大小 = 32**：`common/utils/vars.py:50` `MXFP4_E2M1: [32]`，`:64` mxfp4 可用算法为 `['awq','gptq','mxquant']` → GPTQ-MXFP4 坐实为合法 baseline ✅。
- **fake-quant / 部署路径齐全**：`common/utils/quant_util.py` 的 `quant_dequant_weight/tensor` 已处理 MXFP4_E2M1；部署 `classic/deploy_op/weight_npu_quant_module.py`、`npu_mx_quantization_linear.py` 已支持 MXFP4 导出 ✅。
- **两套注册系统（关键架构认知）**：
  - `AlgorithmRegistry`（`algorithms/__init__.py`）= **算子级权重量化算法**（minmax/awq/**gptq**/mxquant/ofmr/...），签名 `register(name, module_type, quant_module, deploy_module)`。GPTQ 在此：`register('gptq','Linear',GPTQuant,NpuWeightQuantizedLinear)`。
  - `ALGO_REGISTRY`（`algorithms/quant/`）= **可学习结构变换**（flatquant/lwc/lac/omniquant/autoround）。
  - **MR-GPTQ 属前者**（GPTQ 变体），不走 flatquant 那套注册（v0.1 曾误置于此，v0.2 修正）。
- **本地不可执行确认**：`gptq_module.py:132` 硬 `import torch_npu` → 数值实验必须在昇腾 NPU 环境完成。

---

## 3. 算法原理

MR-GPTQ = GPTQ 骨架 + 三个专为 FP4 微缩格式设计的增量（论文 §4.1，**全文已读，本节据论文校正**）。

### 3.1 MXFP4 格式与难点

- MXFP4 = block size **32** + 元素 E2M1（1符1指2尾，7个正值 {0.5,1,1.5,2,3,4,6}）+ 共享 scale 用 **E8M0**（纯指数、无尾数 → scale 被逼成 2 的幂次 PoT）。
- 元素量化：`q = fp4_e2m1(x / s)`，反量化 `x̂ = s · dequant(q)`。
- **难点（论文 §3 定量证明）**：native 权重/激活是 Laplace 重尾分布；absmax scaling 下块内 outlier 会挤压其余元素精度。更要命的是 E8M0 的 PoT scale：能表示 2⁻¹²⁷~2¹²⁸ 但实际数据范围很窄，网格太粗 → scale 量化误差大。这是 MXFP4 掉点（RTN 下 ~10% 相对）的主因。

### 3.2 GPTQ 误差补偿（基线，直接复用）

逐层 GPTQ：用校准激活构造 Hessian `H = 2XᵀX + λI`（λ=1% 均值阻尼，与仓内 `perc_damp=0.01` 一致），固定列序贪心量化，每量化一列用 `H⁻¹` 的 Cholesky 把误差补偿到未量化列，最小化 `‖XW − XŴ‖²`。仓内 `gptq_module.py` 已实现（含 MXFP4 分支）。这既是 baseline，也是 MR-GPTQ 的骨架。

### 3.3 MR-GPTQ 的三个增量（论文 §4.1）

**① Block-wise Hadamard 微旋转（Ingredient 3）**
对线性层 `Y = XWᵀ`，用 **block 对角 Hadamard** `H_k`（k×k 分块，k 为 2 的幂）：`Q(WH_k)·Q(XH_k)ᵀ`。

- 原理（论文 §3 证明）：Laplace 重尾经 Hadamard 后趋于 Normal，把块内 outlier 能量摊到整个 block → 旋转后 MXFP4 的 RTN 误差更小。
- **与 QuaRot/FlatQuant 的"全 hidden 大旋转"不同**：大旋转会把 outlier 跨 32-block 扩散、反破坏微缩块结构，故用 **micro（块内）** 旋转，k 对齐 block size。
- **k 取值有讲究**：主表用 Had32（对齐 block），但论文 Platinum bench（更低噪）显示 **MXFP4 上 Had128 明显优于 Had32**（跨 4 个 block 的更大块旋转）。→ 我们 8B 上 32 和 128 都要扫。
- **fuse 方式**：`WH_k` **离线**融进权重；`XH_k` **在线**（见 §4.3 我们的处理，精度评测无需快核）。
- **⚠️ 实测修正（本项目结论与上条"micro 优于大旋转"相反）**：我们在 8B 上把 k 从 128 一路加到 4096
  （= 全 hidden 宽），再把 `down_proj` 补到 12288（全宽），**精度单调改善**（W4A4 上分别再收 0.065
  与 0.0305）。论文取小 k 的真实约束是**在线旋转算子的推理开销**，而非"大旋转破坏微缩块结构"——
  本任务只做 fake-quant 精度评估、旋转不计入推理成本，故可取满。**这是本项目相对论文的主要发现之一**，
  数据见自验证报告 §4.3。

**② 静态激活重排（Static Activation Reordering, Ingredient 2）**
GPTQ 原生 dynamic act-order（按 Hessian 对角降序重排）精度好但推理要动态 reshuffle（慢 10-20%）。MR-GPTQ 改**静态**：先按原始列序定 scale/grid，再 shuffle 列跑 GPTQ，跑完 shuffle 回去，保持微缩 group 结构。**同等精度、零推理开销**，实现也简单。

**③ MXFP scale fitting（Ingredient 1 / 附录 H）—— 需与 Hadamard 配合，不能单独用于 GPTQ**
针对 3.1 的 PoT scale 误差，把 E8M0 的 256 个码位从 [2⁻¹²⁷,2¹²⁸] 重映射到数据实际范围，记作 **MXFP4†**。**确切公式（附录 H 式(1)(2)(3)）**：

- 标准 E8M0（含 4/3 去偏）：`s_E8M0 = (4/3)·2^clamp(round(log₂s),−128,127)`（式1）。
- 拟合网格：码位 `q = clamp(round(255·(log₂s − log₂s_min)/(log₂s_max − log₂s_min)), 0, 255)`；拟合后 `s_fit = 2^(α·q + β)`，其中 **α = (log₂s_max − log₂s_min)/255**（斜率 <1 → 次幂次、比幂次细）、**β = log₂s_min**（式2/3）。`s_min/s_max` 取该张量所有块 scale 的最小/最大值。
- 效果（论文表 11）：RTN 下 Qwen3-8B **93.7→96.3**、Llama3-8B 87.8→94.3；MR-GPTQ 下 Qwen3-8B **95.2→98.5**。
- **⚠️ 关键修正（原 v0.3 判断错误）**：scale fitting **不是**"单独最先做"。论文表 11 GPTQ 行：**Qwen3-8B 94.1→92.3（−1.8，变差）**；只有叠在 **Hadamard 之上**（MR-GPTQ）才发力（95.2→98.5）。RTN 下才单独有效。**我们实测复现了这点**：GPTQ-mxfp4 10.13 → 加 scale fitting 10.26（变差）。故 scale fitting 须与 Hadamard 配合、排在其后。

> 校准设置：论文用 1024 条 FineWeb 序列、λ=1e-2，与仓内 GPTQ 默认基本一致。
> **论文自证我们的靶子成立**：Qwen3-8B 上 MXFP4 MR-GPTQ(95.2%) > GPTQ(94.1%) > RTN(93.7%)（表10）——"相比 baseline 有提升"在测试模型上已被原作者验证，我们是在 AMCT 里复现这个已知正结果。

---

## 4. AMCT 接入设计

### 4.1 新增/修改文件

| 文件                                                   | 改动                                                   | 说明                                                                                                                                                      |
| ------------------------------------------------------ | ------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `amct_pytorch/classic/quantize_op/mr_gptq_module.py` | **新增**                                         | `class MRGPTQuant(GPTQuant)`：在 GPTQ 误差补偿前插入 block-Hadamard 旋转，量化目标改为旋转后 + MXFP4 微块 scale 特化。预估 200~400 行（大量复用父类）。 |
| `amct_pytorch/algorithms/__init__.py`                | 改（+import，+`BUILT_IN_ALGORITHM`，+register 一行） | 见 4.2                                                                                                                                                    |
| `amct_pytorch/configs/w4a4.yaml`                     | **复用**（无需新建）                             | MXFP4 =`w4a4.yaml`（w_bits/a_bits=4）+ CLI `--quant_dtype mxfp`，已验证                                                                               |
| `experiment/task-book/mr-gptq-mxfp4_<id>/`           | 新增                                                   | 交付目录：脚本、README、结果、日志                                                                                                                        |
| （复用，仅读）flatquant 的 Hadamard/Kronecker 工具     | `algorithms/quant/flatquant.py`                      | 拷贝`_kronecker_matmul`，`_random_orthogonal` → 换成确定性 Sylvester-Hadamard 构造                                                                   |

### 4.2 注册（经典算子系统，非 flatquant 那套）

```python
# amct_pytorch/algorithms/__init__.py
from amct_pytorch.classic.quantize_op.mr_gptq_module import MRGPTQuant
BUILT_IN_ALGORITHM = ['minmax', 'awq', 'gptq', 'smoothquant', 'mxquant', 'ofmr', 'cast', 'quantile', 'mr_gptq']
AlgorithmRegistry.register('mr_gptq', 'Linear', MRGPTQuant, NpuWeightQuantizedLinear)
# 部署模块：MXFP4 权重量化沿用 NpuWeightQuantizedLinear（已支持 MXFP4_E2M1）；
# 若最终走 W4A4 全量化(激活也 mxfp)，改用 NpuMXQuantizationLinear，实施时按导出验证选定。
```

### 4.3 MRGPTQuant 实现要点（复用父类 GPTQuant，增量分步落地）

父类 `GPTQuant` 已提供：Hessian 累积（`update_hessian`）、求逆（`cal_hessian_inverse`，Cholesky）、贪心量化+误差补偿主循环（`get_opt_weight_and_quant_factor`）、MXFP4 分支（`cal_scale_offset_static` 返回 `None,None`）、fake-quant 缓存。

按论文各增量分步实现（**实现顺序 = 投入产出比排序**；第三项静态激活重排最终有意未采用，理由见 §3.3 与 §4.4）：

> **顺序修正（v0.4，基于实测 + 论文表 11）**：原计划"scale fitting 先做/单独用"是错的——它单独会让 GPTQ 在 Qwen3-8B 变差（实测 10.13→10.26，与论文表 11 的 −1.8 一致）。MXFP4 的头号杠杆是 **Hadamard**（论文标题即此），scale fitting 须叠在其上才发力。故新顺序：

1. **【先做】Block-Hadamard 微旋转（§3.3①）**：`override get_opt_weight_and_quant_factor()`，进 GPTQ 列循环前 `W' = W·H_k`（H_k 确定性 Sylvester-Hadamard，k∈{32,128} 可配，论文示 MXFP4 上 Had128 更优），补偿在 `W'` 上做。weight-only 下激活侧只需前向乘一次 `H_kᵀ`（数值精确、无需快核）。
2. **【再做】MXFP scale fitting（§3.3③）**：叠在 Hadamard 之上，override MXFP4 的 per-block scale，用式(2)/(3) 重映射 E8M0 网格。落点：`cal_scale_offset_static` 的 MXFP4 分支 / `quant_util.py` 的 mxfp4 量化路径。
3. **【后做】静态激活重排（§3.3②）**：先定 scale/grid，再按 Hessian 对角序 shuffle 列跑 GPTQ、跑完 shuffle 回，保持 32-group 结构。

**关于激活侧旋转 `X·H_k`（原头号难点，现已明确）**：

- 本任务目标是**精度（PPL）**，不是推理速度 → **无需实现 QuTLASS 那种 fused NPU 快核**。精度评测阶段，`X·H_k` 直接在 fake-quant 前向里用 **朴素 PyTorch `matmul` 块对角 Hadamard** 实现即可（慢但数值正确）。落点：量化模块 forward 里、激活量化之前插一步旋转。
- 若最终做 **weight-only（W4A16，见 §4.4）**：激活是 FP16，`X·H_k` 无量化误差，仅需在前向乘一次（或 fuse 进相邻层），复杂度进一步降低。
- QuTLASS 快核 = 论文的**推理加速**贡献，是本任务的**可选加分**，不阻塞精度达标。

**校准型、无梯度训练**：与 GPTQ 一样只需 `extract_ptq_data` + 一遍前向求解，`ptq` 不跑训练循环——相对 BATQuant（可学习）在 100h 算力预算上的核心优势。

### 4.4 目标位宽决策：W4A16 为主线，W4A4 按评审要求补齐（均已达标）

论文数据给了明确指引（表2 vs 表1，Llama3-8B）：**weight-only MXFP `GPTQ` recovery 96.76% ≫ W4A4 的 89.47%**——激活量化贡献了约一半误差。对本任务：

| 方案                           | 权重  | 激活  | 满足显存≥50%?              | 满足量化层≥70%? | PPL 达标难度                              | 定位                |
| ------------------------------ | ----- | ----- | --------------------------- | ---------------- | ----------------------------------------- | ------------------- |
| **W4A16（weight-only）** | mxfp4 | bf16  | ✅ 权重 16→4bit，整体 >50% | ✅               | **低**（无激活量化+无在线激活旋转） | **首选/保底** |
| W4A4（全量化）                 | mxfp4 | mxfp4 | ✅ 更省                     | ✅               | 高（激活敏感，需在线旋转）                | 进阶/加分           |

任务书硬指标（delta≤0.4 / 量化层≥70% / 显存≥50%）**未限定位宽组合**，W4A16 已全部满足且风险最低 → **8B/35B 主线走 W4A16**（已达标：8B delta 0.273、35B 0.067）。

**评审补充要求**：weight-only 仅得存储收益，需以 **W4A4 精度**评估最终效果 → 本设计据此补齐 W4A4 实现与优化（下文）。**W4A4 现已在 Qwen3-8B 达标：delta 0.3897 ≤ 0.4、真 4-bit 激活层占比 71.4% ≥ 70%**（由初始实现的 0.763 降低 49%）。35B 未列 W4A4——其路由专家为融合 3D 张量、激活量化通路在该结构上不存在，详见自验证报告 §4.3。

**W4A4 精度评估（按评审意见补充）**：weight-only 仅得存储收益，matmul 时权重需反量化回 bf16、推理反而变慢；W4A4（激活也 4-bit）才能让 matmul 在低比特上算、拿到计算收益。为此在**已实现的 Hadamard 在线旋转**基础上补齐激活量化：

- **方法**：`fake_quant_forward` 内，对旋转后的激活 `X·H_k` 做**逐 token、逐 32 通道块的动态 MXFP4**（E8M0 shared exponent 每次前向重算）——旋转把激活离群值摊平，是激活能 4-bit 的前提；
- **实现**：`--increments hadamard scale_fitting act_quant` 启用；amct 配置仍为 weight-only 组合（放行），激活量化在自实现 fake-quant 路径完成，用于**精度评估**（真部署需低比特激活算子，属另一课题）；
- **消融**：W4A4「旋转开 vs 关」对照，量化 Hadamard 对激活量化的关键作用；
- W4A4 精度数据见自验证报告。预期 delta 明显大于 W4A16（激活敏感），作进阶精度补充、不改 W4A16 保底结论。

**W4A4 精度优化设计**：在上述基础实现之上，针对 4-bit 激活的主要误差源（离群值 + scale 粗糙 +
补偿输入失配）设计 4 项优化，均为可独立开关的增量，便于消融：

| 优化                    | 开关                           | 设计要点                                                                                                                                                                                                                                                                                                              |
| ----------------------- | ------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 旋转块加强              | `--hadamard_k K`             | 论文取 k=128 是**在线旋转算子**的开销权衡；本任务只做 fake-quant 精度评估、旋转无推理开销，故把 k 提升为可调（512/1024/4096），块越大离群值摊得越开。注意 `in_features % k == 0` 才旋转，且大 k 下旋转 matmul 须在 fp32 做（fp16 累加 4096 项误差会污染消融）                                                 |
| 激活 scale-fitting      | `--increments act_scale_fit` | 把权重侧已验证的附录 H 网格重映射（式2/3）复用到激活块 scale：以次幂次细网格替代 E8M0 幂次网格，减小激活 scale 的舍入误差                                                                                                                                                                                             |
| 联合补偿                | `--increments joint_comp`    | `update_hessian` 中先对（旋转后）激活做 MXFP4 量化再累积 Hessian。GPTQ 补偿的最优性依赖 Hessian 与推理期输入分布一致；W4A4 下推理输入是量化激活，用 bf16 输入建的 Hessian 会系统性失配                                                                                                                              |
| 混合精度保护            | `--act_protect <子串...>`    | 按层名子串把指定层的激活保留高精度（权重仍全程 MXFP4）。默认候选为`down_proj`（输入 = 门控后中间激活）与 `o_proj`（输入 = 注意力输出）——两处离群值最重。需同时报告"仍跑真 4-bit 激活的层占比"，避免用保护面换数字                                                                                               |
| **全宽旋转**      | `--hadamard_full`            | 每层旋转块取满`in_features`。`down_proj` 的 in=12288 在 k=4096 下只能切 3 个对角块，是全网**唯一未获全宽旋转**的层、而它最大。12288 非 2 的幂，且 `3×4096` 不可行（3 阶 Hadamard 不存在），走 `12×1024`：Paley H₁₂ ⊗ Sylvester H₁₀₂₄——正交矩阵的 Kronecker 积仍正交。**单项 −0.0305** |
| **GPTQ 阻尼系数** | `--perc_damp`                | `damp = perc_damp × mean(diag(H))`。最大层 `down_proj` 的样本/维度比仅约 1.1:1、Hessian 近奇异，该项实际撑住整个求逆，默认值按正常条件数选取、在此偏小。0.01→0.2 **单项 −0.0261**。此二项是达标关键                                                                                                      |

另实现并消融了 **5 项经实测排除的手段**（激活 scale 裁剪搜索、加大校准集、权重 scale 裁剪、
增加保护层、randomized Hadamard），开关默认关闭、代码保留作消融记录。其中两条给出与**量化粒度
相关**的一般性结论：① 大激活承担 attention sink 等功能、并非噪声，逐块 MSE 最优 ≠ 端到端最优，
故不可裁剪；② randomized Hadamard 在 block-32 粒度下**有害**——DC 分量由集中于一个坐标变为摊到
全部块，而 scale 是逐块的，**集中优于摊匀**，与其在 per-channel 粒度下的效果相反。
逐项数据与分析见自验证报告 §4.3 ④。

各优化的边际收益消融、最终精度、以及与公开工作的同协议对比见自验证报告 §4.3。W4A4 承担
**评审要求的计算收益路线评估**；三项硬指标的达标结论由 W4A16 主线承担，不依赖于 W4A4。

### 4.5 全链路调用（实际实现：独立评测脚本）

**与设计初稿的偏差及原因**：初稿计划复用 AMCT 标准 CLI（`eval / extract_ptq_data / ptq / deploy`）。
实施中改为提供独立脚本 `src/run_eval.py`，原因有二：① MR-GPTQ 采用**运行时注册**（`import` 即注册、
零改 amct 源码），标准 CLI 的算法白名单在包内，不改源码则无法识别 `mr_gptq`；② W4A4 的激活量化需
在 `fake_quant_forward` 内介入，且需按增量开关做消融，标准 CLI 无对应参数面。
**量化本身仍完全走 amct 的 `amct.quantize()` 与 `GPTQuant` 状态机，未绕过工具链。**

```bash
cd src
export ASCEND_RT_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1

# 1. BF16 基准
python3 -u run_eval.py --model_path <M> --mode bf16 --seq_len 4096

# 2. baseline 对照：GPTQ-MXFP4 直转（仓内已有算法）
python3 -u run_eval.py --model_path <M> --mode gptq_mxfp4 --seq_len 4096

# 3. MR-GPTQ 主线（W4A16）
python3 -u run_eval.py --model_path <M> --mode mr_gptq --seq_len 4096 \
    --increments hadamard scale_fitting

# 4. MR-GPTQ W4A4 达标配置（Qwen3-8B）
python3 -u run_eval.py --model_path <M> --mode mr_gptq --seq_len 4096 \
    --hadamard_k 4096 --hadamard_full --perc_damp 0.2 \
    --act_protect down_proj o_proj \
    --increments hadamard scale_fitting act_quant act_scale_fit joint_comp

# 35B 双卡：追加 --device_map auto，并 export ASCEND_RT_VISIBLE_DEVICES=0,1
```

**未做导出部署**：本任务验收指标为精度（PPL）与覆盖率 / 显存，均由 fake-quant 确定性数学决定；
`deploy` 导出属推理落地环节，不在本任务范围（W4A4 的真实部署另需低比特激活算子）。

---

## 5. 量化范围设计（满足 ≥70% 覆盖）

量化层占比 = 被量化 `nn.Linear` 数 / 全网 `nn.Linear` 数。

| 模型            | 结构  | quant_target 设计     | 覆盖分析                                                                                                                                     |
| --------------- | ----- | --------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| Qwen3-8B        | dense | `attn-linear + mlp` | attn(q/k/v/o) + mlp(gate/up/down) 覆盖每个 decoder layer 的全部 Linear，占比可 >90% ✅                                                       |
| Qwen3.6-35B-A3B | MoE   | `attn-linear + moe` | ⚠️**必须含 MoE experts**：experts 的 gate/up/down 占全网 Linear 绝大多数，只量化 attn-linear 远低于 70%。纳入 experts 后占比 >90% ✅ |

**MoE expert 是精度风险点**（casebook qwen3_moe 记录：expert layout / packed 权重坑）。实施时先 attn-linear 打通，再逐步纳入 experts，观察 expert 量化对 PPL 的影响。linear_attention（若有）保持 BF16，不计入量化闭环。

---

## 6. 实验设计与验收

### 6.1 对照实验矩阵

| 实验 | 配置            | 目的                          |
| ---- | --------------- | ----------------------------- |
| E0   | BF16 baseline   | 得`ppl_bf16`（精度门分母）  |
| E1   | GPTQ-MXFP4 直转 | baseline 对照，得`ppl_gptq` |
| E2   | MR-GPTQ MXFP4   | 主方案，得`ppl_mrgptq`      |

**判定**：

- 绝对达标：`ppl_mrgptq − ppl_bf16 ≤ 0.4`
- 相对提升（进阶项）：`ppl_mrgptq < ppl_gptq`（越低越好）
- 覆盖达标：Linear 占比 ≥ 70%（脚本统计并写入报告）

### 6.2 分阶段计划（对齐 100h NPU 预算）—— 已执行

1. **P1 — Qwen3-8B 打通**：调通 MR-GPTQ 实现、跑齐 E0/E1/E2，确认相对提升成立 ✅
2. **P2 — Qwen3.6-35B-A3B（含 MoE）**：另需处理融合路由专家（非 nn.Linear）才能达显存指标 ✅
3. **P3 — 自验证报告 + README** ✅；按评审意见追加 **P4 — W4A4 补齐与优化** ✅（见 §4.4）

统一口径：`seq_len=4096`、block 粒度、WikiText2、全量分块评测。

> **测量口径的自我修正**：项目中期曾以测试集前 20 块做快速筛选，后经两组同配置对照发现该子集的
> 偏移随配置摆动达 0.053、大于多数待测增量，**连排序都不可靠**。此后所有定量结论均基于全量评测，
> 详见自验证报告 §4.3 ⑤。

### 6.3 性能数据

由评测脚本直接统计并打印：量化层覆盖率（`[coverage]` / `[actual-coverage]`）、权重显存降低
（`[memory]` Linear 口径 / `[memory-full]` 全模型口径）、校准与补偿耗时、PPL 评测耗时，
连同原始日志写入自验证报告 §6。（初稿设想用 `cann-perf` 采集算子级耗时；因本任务验收指标为
精度与显存、且量化为 fake-quant，脚本内建统计已足够，未引入额外采集工具。）

---

## 7. 风险与规避

| 风险                                               | 影响                 | 规避                                                                                                                                                                                                                                             |
| -------------------------------------------------- | -------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| MoE expert 量化敏感，35B 掉点超 0.4                | 覆盖达标但精度不达标 | 先 attn-linear 定位，逐步纳入 experts；必要时对最敏感 expert 层放宽/保护；读 casebook qwen3_moe 坑                                                                                                                                               |
| baseline GPTQ-MXFP4 跑不通                         | "有提升"无对照       | P1 第一步即验证 GPTQ-MXFP4 能产出有效数；跑不通则改用直转 RTN-MXFP4 作对照并说明                                                                                                                                                                 |
| ~~激活侧旋转落点（原头号难点）~~ **已降级** | 曾担心等价性/快核    | 全文已明确：`WH_k` 离线 fuse、`XH_k` 在线。**本任务只要精度→用朴素 PyTorch matmul 做块 Hadamard 即可，不需 QuTLASS 快核**；再叠加 W4A16（激活 FP16）进一步免除激活量化。快核仅为加分。仍需在 8B 上先验证旋转前后 `bits=16` 数学等价 |
| MXFP scale fitting 是否吃到增益                    | 拿不到论文 +5pt      | 论文式(2)明确、实现简单，**最先做且单测**（对同一层比 E8M0-default vs fitted 的块重构 MSE）；先只做 scale-fitting 看 delta，再叠旋转                                                                                                       |
| 100h 算力不足                                      | 35B 跑不完           | MR-GPTQ 校准型（无训练循环）成本可控；35B 用多卡脚本`examples/ptq_multi_npu.sh`；优先保 8B 完整 + 35B 关键场景                                                                                                                                 |
| 官方设计文档模板不同                               | 评审格式返工         | 已向组织者确认模板（用户侧）；本文结构可平移                                                                                                                                                                                                     |

---

## 8. 交付件

> 以下为**实际交付位置**（初稿设想把算法落进 `amct_pytorch/` 并改 `algorithms/__init__.py` 注册；
> 最终改为**运行时注册、零改 amct 源码**，故代码全部位于交付目录内）。
> 交付目录：`experiment/task-book/mr-gptq-mxfp4_asp1r1n1/`

| 交付件     | 位置                                                                                                                |
| ---------- | ------------------------------------------------------------------------------------------------------------------- |
| 算法代码   | `src/mr_gptq_module.py`（子类化 `GPTQuant`，`import` 即注册，**零改 amct 源码**）                       |
| 评测脚本   | `src/run_eval.py`（入口）、`src/eval_utils.py`（加载/校准/PPL/覆盖率与显存统计）、`src/data.py`（数据集加载） |
| 实验日志   | `src/logs/`（bf16 基准、W4A16 主线、W4A4 优化链与各负结果，全部原始输出）                                         |
| 自验证报告 | `docs/self-verification-report.md`（结果 + 截图 + §3 可复现步骤 + §4.3 W4A4 与负结果）                          |
| 实验记录   | `docs/experiment-log.md`（环境搭建与踩坑流水）                                                                    |
| 设计文档   | 本文`docs/design.md`                                                                                              |
| README     | `README.md`（目标、结果、目录、运行方式、安全提示）                                                               |
| PR         | `feat/mr-gptq-impl` → `cann/amct:feature/community-tasks`（设计 + 代码 + 报告，**单 PR**）               |

---

## 附录 A：参考

- MR-GPTQ: *Bridging the Gap Between Promise and Performance for Microscaling FP4 Quantization*, arXiv:2509.23202
- GPTQ: arXiv:2210.17323
- MXFP4 微缩格式: arXiv:2310.10537
- FlatQuant: arXiv:2410.09426
- BATQuant（备选路线）: arXiv:2603.16590
- AMCT 算法矩阵: `docs/zh/algorithm_brief.md`
- Qwen3.6-MoE 量化样例: `examples/models/qwen3.6/Qwen3.6-Moe.md`