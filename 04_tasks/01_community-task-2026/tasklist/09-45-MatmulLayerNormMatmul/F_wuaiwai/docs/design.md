# MatmulLayerNormMatmul 算子设计文档

> 任务编号：`09-45-MatmulLayerNormMatmul`｜开源仓：[cann/catlass](https://gitcode.com/cann/catlass)｜适配硬件：Ascend 950（本任务在 Ascend 950PR 验证）
> 依据版本：catlass 主干 v2.1.0（参考 commit `165c7a1`）、CANN 9.0.0（C++ 路线）、`-DCATLASS_ARCH=3510`
> 文中引用标注：**[仓]** = catlass 仓内代码/文档（附文件:行）；**[算]** = 由任务测试集 116 例计算；**[书]** = 任务书；**[待实测]** = 需在 950PR 上实测确认

---

# 需求背景（required）

## 需求来源

[书] CANN 2026 社区任务「MatmulLayerNormMatmul 算子开发」（任务编号 09-45）：在 `cann/catlass` 仓新增融合算子样例与必要的 kernel/block/tile 分层组件，适配 Ascend 950，开发语言 Ascend C，最终以 PR 形式合入主干（合入范式参考 PR #678）。

## 背景介绍

### 算子组合现状分析

`Matmul → LayerNorm → Matmul` 是 Transformer / 推荐类网络中的常见串联（投影 → 行级归一 → 再投影）。以 PyTorch 三算子拼接实现时，强制产生 **3 次 kernel launch** 与 **4 遍 M0×N0 中间量显存访问**：

```text
torch.mm      : A0,B0            → 写 C0              （1 遍 M0·N0）
F.layer_norm  : 读 C0 → 统计 μ/σ² → 写 C0_norm         （2 遍 M0·N0）
torch.mm      : 读 C0_norm → 写 C1                    （1 遍 M0·N0）
```

任务书给出的 116 例标杆时延为 19.005us ~ 1188.58us、均值 132.7us [算]。其中 **M0=128 的 30 例集中在 19~66us**，这段区间的时延主要由 launch 开销与 LN kernel 的纯访存开销构成 —— 也就是说，**融合带来的收益在小 M0 段主要来自"少 2 次 launch + LN kernel 整体消失"，在大 M0 段主要来自"中间量访存削减"**。本设计的方案分层即围绕这一区分展开（见「实现方案」）。

### catlass 生态现状分析 [仓]

| 现状 | 对本任务的影响 |
| --- | --- |
| 全量检索 `include/catlass/` 的 `layernorm\|rms\|norm_` | **仓内不存在任何 LayerNorm/RMSNorm 组件**（仅命中 FA 的 `block_mmad_fai_*_normal.hpp`，语义无关）→ 归一化组件必须新写，这是本任务的主要工作量 |
| 「双 Matmul + 中间行级归一化」的结构 | 仓内已有三处同构实现：`examples/49_ascend950_flash_attention_infer/fai_kernel.h`（QK → 行 softmax → PV，单 kernel）、`examples/83_ascend950_hstu_infer/kernel/hstu_infer.hpp` 的 `HstuInfer<BlockMmadQK, BlockMmadPV, EpilogueSiluScale, EpilogueCast, ...>`、`block_mmad_mla_{qk,pv}.hpp` → **可沿用其结构与 AIC/AIV 分工范式，且这两例证明"kernel 层代码可先放样例目录"** |
| 行归约原语 | `epilogue/block/block_epilogue_amla_tp1_softmax.hpp:163-186` 使用 `AscendC::BlockReduceSum<float,false>`（`RowsumSPECTILE512/256` + `SetBlockReduceMask` 尾块）→ μ 与 Σx² 的直接改写出入口 |
| 列向量广播仿射 | `epilogue/tile/tile_broadcast_mul.hpp:28 TileRowBroadcastMul`（与 shape (1,n) 向量广播相乘，方向正是 γ/β 所需）、`tile_broadcast_add.hpp`、`tile_cast.hpp`；EVG 侧有 `examples/64_ascend950_matmul_evg_bias`（TreeVisitor + RowBroadcast）与 **`_add_ub`（L0C→UB 通路）** 实证 |
| 950 专用件 | `gemm/tile/ascend950/copy_l0c_to_ub.hpp`（L0C→UB，LN 必需）；`gemm/dispatch_policy.hpp:250 MmadAscend950FullLoadA`、`:354 MmadPingpongMutex`（`static_assert` 限定 `Arch::Ascend950`）；`examples/43_ascend950_basic_matmul` 实证 950 可用参数（见「组件选型」） |
| 精度判据 | `examples/common/golden/compare_data.hpp:30-88` 已有 MARE/MERE/RMSE 比值实现，可作辅助诊断；**验收判据以生态算子开源精度标准（混合容差）为准**，见「可维可测分析」 |

### 功能分析

| 参数 | 方向 | 形状 | 类型 | 布局 | 说明 |
| --- | --- | --- | --- | --- | --- |
| A0 | 输入 | (M0, K0) | FP16 | RowMajor / ND | 第一个 Matmul 左矩阵 |
| B0 | 输入 | (K0, N0) | FP16 | **ColumnMajor** / ND | 第一个 Matmul 右矩阵 |
| B1 | 输入 | (N0, M1) | FP16 | **ColumnMajor** / ND | 第二个 Matmul 右矩阵（其 K 维 = N0） |
| gamma | 输入 | (N0,) | **FP32** | ND | LayerNorm 缩放，沿 mm1 的 N 轴广播 |
| beta | 输入 | (N0,) | **FP32** | ND | LayerNorm 偏置，同上 |
| C1 | 输出 | (M0, M1) | FP16 | RowMajor / ND | 唯一对外输出 |
| ε | 内置常量 | 标量 | FP32 | — | 固定 `1e-6`，加在 sqrt 内 |
| （内部） | — | — | FP32/FP16 | — | `C0`、`μ`、`σ²`、`P` 累加器与 workspace，**不出现在对外接口** |

---

# 需求分析（required）

## 需求描述

用 Ascend C / catlass 模板在 Ascend 950 实现 `Matmul + LayerNorm + Matmul` 全融合算子：

1. **单次 kernel launch** 完成三段计算；
2. 均值/方差等中间结果由**算子内部 workspace 管理，不对外暴露**；
3. 允许采用多种 matmul 方案组合取优（任务书明确要求，本文给出三档方案 + 分档选型）；
4. 精度满足[生态算子开源精度标准](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md)；
5. 性能达到 `torch.mm + F.layer_norm + torch.mm` 拼接基线的 **1.1 倍以上**（`msprof op` 采集，116 例）；
6. 交付 optest 测试件 + ATK ≥200 例泛化 + 设计文档/自验证报告/README，并合入 catlass。

## 需求拆解

| 编号 | 子需求 | 交付形态 | 可验证判据 |
| --- | --- | --- | --- |
| F1 | 三段全融合、单 kernel launch | 一个 device kernel（内部多阶段，含 AIC/AIV 双 role） | `msprof op` 结果中本算子对应 1 个 kernel |
| F2 | 中间量内部 workspace、不外露 | host 侧 `GetWorkspaceSize` 分配；接口仅 6 个张量 | 对外接口签名无 C0/μ/σ² |
| F3 | 多方案取优 | 档 1（主）/ 档 2（难例）/ 档 3（快路径）+ 分档决策表 | 本文「实现方案」+ 实测数据对比表 |
| F4 | 精度达标 | 混合容差判定脚本 + golden | 116 例 + ATK ≥200 例全通过 |
| F5 | 性能 ≥1.1× | `msprof op` 逐例采集 | 平均(标杆/实测) > 1.1，并按方案/TileShape 备注 |
| F6 | optest 交付件 | `tests/optest/{include,kernels,src,torch_catlass,tests}` 五层接入 | `pytest tests/test_*.py` 通过 |
| F7 | ATK 交付件 | `NN_name.yaml` + generator + execute + node.yaml | ≥200 例精度通过（测试集仅 116 → 必须扩样） |
| F8 | 文档 | design.md（本档）、自验证报告、README | 评审通过并合入 |
| F9 | 规范 | Ascend C 编程规范、License 头、`.clang-format` | 代码检视无规范类意见 |

---

# 详细设计（required）

## 算子分析

### 数学公式

$$\mathbf{C_0}=\mathbf{A_0}\mathbf{B_0}\in\mathbb{R}^{M_0\times N_0},\qquad
\mu_m=\frac{1}{N_0}\sum_{n}C_0[m,n],\qquad
\sigma^2_m=\frac{1}{N_0}\sum_{n}\big(C_0[m,n]-\mu_m\big)^2$$

$$C_{\text{norm}}[m,n]=\frac{C_0[m,n]-\mu_m}{\sqrt{\sigma^2_m+\epsilon}}\,\gamma_n+\beta_n,\qquad
\mathbf{C_1}=\mathbf{C_{\text{norm}}}\,\mathbf{B_1}\in\mathbb{R}^{M_0\times M_1}$$

与 `torch.nn.functional.layer_norm` 必须逐条对齐的语义：

1. **population 统计**：分母为 `N0`（不是 `N0-1`）；
2. ε 加在 **sqrt 内**：`rstd = 1/sqrt(var + 1e-6)`；
3. γ/β 沿 mm1 的 **N 轴**（每行的列方向）广播，长度 `N0`，**FP32**；归一化在 FP32 域完成，最后一步才 cast FP16；
4. 标杆链路中 `C0` 与 `C0_norm` 均以 **FP16 存储**（mm 输出为 fp16）→ 本设计的中间量存储精度选择见「精度设计」。
5. **结构性约束**：LN 需要整行 `N0` 全部到齐才能归一。这是本算子唯一的关键约束，所有方案分歧都由它产生。

### 支持数据类型

| 张量 | 支持类型 | 计算类型 |
| --- | --- | --- |
| A0 / B0 / B1 | FP16 | mmad 累加 **FP32**（`CType=float`） |
| gamma / beta | FP32 | LN 统计与仿射全程 **FP32** |
| C1 | FP16 | 由 FP32 累加器 `P` 在最后一步 `TileCast` 得到 |

本期不支持 BF16/FP32 输入与 NZ 等非 ND 格式（见「算子约束限制」）。

### 支持形状

任务测试集 116 例的形状空间 [算]：`M0∈{128,512,1024,2048}`、`K0∈{768,2048,4096,8192}`、`N0∈{2048,3072,4096,8192}`、`M1∈{768,2048,4096}`。

设计需要覆盖的非测试集形态（ATK 扩样阶段验证）：奇数维、非 16 倍数维、`N0`/`K0` = 8192 的极端组合、`σ²=0`（常量行）、`γ≡1/β≡0`。

### ★ 操作数拓扑判定（先于任何组件选型）

| 判据 | 结论 |
| --- | --- |
| 中间处理的操作数是否跨 N-block？ | **是** —— μ、σ² 是对 `C0` 的**整行（全 N0）归约**；γ/β 是整行广播 |
| 直接推论 | 「每个 stage 只放一个 `[TileM,TileN]` tile」型的轮转 workspace **在 `N0 > L1TileShape::N` 时必然算错**。mm1 的 N-tile 最大 256，而本任务 `N0 ≥ 2048` [算] → **恒不满足**，不可照搬常规 fused-epilogue 结构 |
| 设计要求 | 必须在**调度层面**保证「行完整性先于归一」：① 同一处理单元持有该行带的完整 N0（档 1）；或 ② 拆成「局部统计 + 跨核合并 + 代数校正」（档 2） |

### 片上容量与并行度推演（定量依据）

每 AI Core 片上容量（真源 [仓 `include/catlass/arch/arch.hpp:29-37`]）：`L1 = 512KB`、`L0A = L0B = 64KB`、`L0C = 256KB`、`UB = 248KB`、`BIAS = 4KB`、`FIXBUF = 16KB`。
容量约束式：`m1·k1·szA·L1A_STAGES + n1·k1·szB·L1B_STAGES ≤ 512KB`；`m0·n0·4·L0C_STAGES ≤ 256KB`。

**(a)「整行带常驻片上」的 TileM 上限**（行带 = `TileM×N0×2B`，FP16）

| N0 | 2048 | 3072 | 4096 | 8192 |
| --- | --- | --- | --- | --- |
| TileM @ 独占全 L1(512KB) | 128 | 85 | 64 | 32 |
| TileM @ L1 余量 ~256KB（留给 B 面板） | 64 | 42 | 32 | **16** |

**(b) 行带切细 ⇒ B 面板重读爆炸（否决"纯片上常驻"作主方案的依据）** [算]
mm2 每核的 K 循环须完整扫过 `N0`，故 `B1` 重读次数 = `bands = ⌈M0/TileM⌉`：

| 用例 | 形状 | TileM=16（(a) 上限） | TileM=256（常规 GEMM） |
| --- | --- | --- | --- |
| idx116 | 2048,8192,**8192**,4096 | 128 bands × 64MB = **8GB** ≈ 5.3ms @1.5TB/s（标杆 1.19ms → 出局） | 8 bands × 64MB = 512MB ≈ 0.34ms（L2 可吸收大部分） |
| idx107 | 128,8192,8192,4096 | 1 band → **仅 1 个 AIC 工作**，其余核空转 | 同（M0=128 天然 1 band）→ 需沿 N0 再切核，即档 2 |

**(c) L0C 争用 ⇒ 两阶段基本不可回避**
mm1 的 `C0` tile（`TileM×TileN×4B`）与 mm2 的 `P` 累加器（`TileM×TileM1×4B`，须跨整个 K=N0 循环驻留）都要占用 L0C（单份 256KB）：`TileM=128` 时两者各 128KB，**理论上可时分复用同一 L0C 做真单遍流水**（列为优化目标，非 v1 承诺）；但当 `M1 > TileM1`（测试集 M1 最大 4096 → 16 个 256 列片 [算]）时 `P` 需按 M1 片外扩 → mm1/mm2 无法并发共享 L0C → **`C0` 必须先落一次存储**。

**(d) 中间量体量** [算]：`M0×N0×2B` 最大 **32MB**、平均 6.3MB → 落在 950PR L2 量级（112/128MB **[待实测]**）内 ⇒ 「workspace 写 GM」在多数 case 实为 **L2 命中**，故"两阶段 + 小型行带 workspace"是可行的主方案，而"整块 [M0,N0] FP32 反复往返"是必须避免的反模式。

## 算子实现

### 实现方案

#### 关键变换：LN 的仿射与统计量可解耦

把 LN 代入 mm2 并按 n 展开：

$$\mathbf{C_1}=\underbrace{\mathrm{diag}(\mathbf{rstd})\,\big(\mathbf{C_0}\!\oslash\!(\mathbf{1}\otimes\boldsymbol\gamma)\,\mathbf{B_1}\big)}_{P\ (\text{一次普通 matmul})}\;-\;\underbrace{\mathrm{diag}(\mathbf{rstd}\!\odot\!\boldsymbol\mu)\,\mathbf{g}}_{\text{秩 1 校正}}\;+\;\underbrace{\mathbf{1}\otimes\mathbf{b}}_{\text{行偏置}},\quad
\mathbf{g}=\boldsymbol\gamma^{\!\top}\mathbf{B_1},\ \ \mathbf{b}=\boldsymbol\beta^{\!\top}\mathbf{B_1}$$

收益：**① `C0_norm` 不必物化**（γ 折进 mm2 的 A 装载路径，μ/σ²/β 折进 C1 的 per-row 仿射 epilogue）；**② 消除"归一 barrier"**（mm1 tile 一产生即可作为 mm2 的 A 参与 K-累加）；**③ 让"沿 N0 跨核切"变便宜**（跨核只需合并 4 个小向量：`Σx`、`Σx²` 长 M0，`g`、`b` 长 M1）。该式与 FA 的「先累加 O、最后除 rowsum」`rescale_o` [仓 `block_epilogue_fa_rescale_o_ascend950.hpp`] 同构，不引入新的数学风险。

**适用边界（重要，已用探针实测）**：档 1 的两阶段结构**天然保证阶段 2 开始时整行已到齐**，因此档 1 直接在 **A 装载路径上完成完整 LN 仿射**（舍入点与标杆一致、也**不需要 `g/b` 与阶段 0**）。而校正路径参与 mm2 的是**未归一**的 `C0·γ`（量级 ≈ `1/rstd` 倍），实测结论是**它的 FP16 形态不可用**：

- 正常量级下，档 1 与档 2 在 **M0=128 整段 30 例均 `matched_ratio = 1.0` 通过**，最大绝对误差 ≈ 1~2 个 FP16 ULP（对 0.1 的硬上限有约 50× 裕度）；
- 抬高 `C0` 量级后，**档 2 在 `max|A|` 仅达 FP16 上限 10%（约 6652）时即不达标**（`matched_ratio = 0.973 < 0.99`），40% 仍 0.973，更大则出 `inf`；同条件下**档 1 始终通过**；
- 机理**不是溢出而是精度**：归一后置使 FP16 舍入发生在未归一的大数上，误差被 `rstd` 反比放大 `|C0|·rstd` 倍 → 精度随 `C0` 动态范围恶化（把 γ 折到 B1 侧可消除溢出但消不掉该精度退化）。

**因此本设计把难例方案改为「档 2′」**：阶段 1 沿 **N0 分核**算 `C0` 写 FP16 workspace 并累加局部 `Σx/Σx²`，一次跨核 barrier；阶段 2 按 **(band × M1 片)** 分核，仍**在装载路径做完整归一**（与档 1 同精度）。纯代数延迟校正仅在 `A` 以 **FP32** 存储（workspace 32MB→64MB）时才精度安全，作为备选记录。验证脚本与数据：`tools/precision_probe.py`、`test_case/probe_hard_m128.csv`、`probe_m128_all.csv`。

#### 三档方案对比

| 维度 | **档 1：行带 CTA 独占（v1 主方案）** | 档 2：沿 N0 跨核切 + 小向量归约 | 档 3：整行带常驻片上（快路径） |
| --- | --- | --- | --- |
| 结构 | CTA 拥有 `TileM × 全 N0`。**阶段1**：mm1 逐 N-tile 产出 `C0`（FP32→FP16，顺带累加 `Σx/Σx²`，**不折 γ**）→ 写 workspace；**阶段2**：从 workspace 读 A，在 **A 装载路径（UB, FP32）完成 LN 仿射** → 喂 cube，K=N0 累加 `P` → cast FP16 写 C1 | **档 2′（修正）**：阶段1 沿 N0 分核算 FP16 `C0` 写 workspace + 局部 `Σx/Σx²` → 一次跨核 barrier；阶段2 按 (band × M1 片) 分核，仍在**装载路径做完整归一**（与档 1 同精度）。纯代数延迟校正的 FP16 形态已被实测否定 | 阶段1 的 `C0` 行带留在 L1/UB，阶段2 直接取片上 A，零 GM |
| 跨核同步 | **无**（行带内自洽）→ 结果确定、易调试 | **有**（`arch/cross_core_sync.hpp` / 950 的 `MmadPingpongMutex` [仓]）；`P` 跨核相加引入归约顺序不确定性 | 无 |
| `C0` GM 往返 | 2 遍（写+读；多数 case 命中 L2） | 2 遍 + 4 个小向量 | **0 遍** |
| 并行度 | `bands=⌈M0/TileM⌉`，TileM 不受行约束（A 来自存储）→ 正常 | 高（M0=128 也能填满核） | `bands=⌈M0/TileM_max⌉`，TileM_max 16~64 → **大 N0 时崩**（见推演 (b)） |
| 主要风险 | 两阶段之间的空泡；workspace 分配与对齐 | 死锁（事件 ID 上限，见「kernel侧设计」）、非确定性、调试成本 | B 面板重读爆炸 → idx116 直接出局 |
| 定位 | **默认路径，116 例全覆盖** | **专治 M0=128 整段 30 例**（该段按档 1 口径的融合上界均值仅 1.06，其中 14 例连零往返上界都 <1.1 [算]） | 仅当 `TileM×N0×2B ≤ L1 余量` **且** `bands ≥ AIC 数` 时启用（如 M0≥512、N0≤2048），作为增量优化 |

**选型结论**：v1 = **档 1（两阶段 + 装载路径归一）**，单实例先打全 116 例；档 2 作为 M0=128 段的 v1.1 增量（不改接口，只改 scheduler 与归约方式）；档 3 视实测决定是否值得。理由：① 无跨核同步 → 正确性、可复现性与可维可测最稳；② 行完整性靠"两阶段"而非"整行常驻"满足 → 绕开推演 (a) 的容量死结；③ 舍入点与标杆一致，精度裕度最好解释；④ 剩余 GM 往返只有 2 遍且大概率 L2 命中。

#### catlass 组件选型表

**（1）参考 example 与选型理由**

| 项 | 内容 |
| --- | --- |
| 参考 example | `examples/43_ascend950_basic_matmul`（950 组件参数实证来源）、`examples/83_ascend950_hstu_infer`（双 mmad + 中间行处理的 kernel 编排范式）、`examples/44_quant_matmul_full_loadA_tla`（任务书指定参考） |
| 选型理由 | 43 给 950 上真实可用的 DispatchPolicy / stage / TileShape 取值；83 给"两个 BlockMmad + 中间归一化"的单 kernel 结构 |
| 变通点 | ① example 用 `main()` 直调 → 本算子需拆为 host tiling + Device 调用；② example 只有一段 mmad → 本算子两段并共享 workspace；③ example 无行归约 → 需新增 LN 组件；④ mm2 的 A 操作数来自 workspace 而非用户输入 → 需自定义 A 装载路径 |

**（2）BlockMmad（两段各自的 mmad 块）**

| 组件 | mm1（A0×B0→C0） | mm2（归一后 A ×B1→P） | 依据 |
| --- | --- | --- | --- |
| DispatchPolicy | `Gemm::MmadPingpong<ArchTag, enableUnitFlag=true, useHF32=false, l0CStages, enableL1Resident=false, l1AStages=2, l1BStages=2, l0AStages=2, l0BStages=2>`；性能阶段换 `Gemm::MmadPreloadAsyncWithCallback<..., preloadStages, ..., enableShuffleK=false, ...>` | 同 | 参数形状与取值实证自 [仓 `examples/43_ascend950_basic_matmul/basic_matmul_tla.cpp:108-131`]，其中注明 950 上 `enableShuffleK=false` 更优 |
| L1TileShape | `Shape<256,256,128>`（M,N,K）起步 | `Shape<128,256,256>` 起步 | 43 实证 `<256,256,128>`；mm2 的 K=N0 需大 K-tile，但 FP16 下 K ≤ 256（受 L1 约束） |
| L0TileShape | `Shape<256,256,32>` | `Shape<128,256,64>` | 43 实证 `<256,256,32>`；L0A/L0B 64KB 约束 |
| AType | `half, RowMajor` | `half, RowMajor`（**来自 workspace**） | 任务书布局 |
| BType | `half, ColumnMajor` (K0,N0) | `half, ColumnMajor` (N0,M1) | 任务书布局 |
| CType | `float, RowMajor` | `float, RowMajor` | 累加恒 FP32 |
| 容量自检 | `ΣL1 ≤ 512KB`、`L0C: 128×256×4×stages ≤ 256KB` → `l0CStages=1` 时 128KB | | 超限时自动降 TileM/TileN |

**（3）BlockEpilogue —— 槽位清单**（本算子有两个自定义出口）

E1｜mm1 出口 `EpilogueAscend950LayerNormPrep`（新增，**粒度 B**）

| 槽 | 运算 | 决策 | 组件 |
| --- | --- | --- | --- |
| 1 | L0C→UB 搬运 | ✅ 现成 | [仓] `gemm/tile/ascend950/copy_l0c_to_ub.hpp` |
| 2 | FP32→FP16 落 workspace | ✅ 现成 | `epilogue/tile/tile_cast.hpp` + `copy_ub_to_gm(_tla).hpp` |
| 3 | **行归约 `Σx`、`Σx²`** | 🔧 **自定义 Tile** `TileRowSumAndSumSq` | 接口参照 [仓] `block_epilogue_amla_tp1_softmax.hpp:163-186` 的 `AscendC::BlockReduceSum<float,false>` |
| 4 | μ、σ²、rstd 求值（行带末尾） | ✅ 组合 | [仓] `epilogue/fusion/operations.hpp` 的 `Sqrt/Rsqrt/RsqrtFast/Reciprocal/Muls/Add` |
| 5 | （仅档 2）γ 折叠与 `g/b` 校正 | 🔧 | 见 E2，档 1 不启用 |

E2｜mm2 入口/出口 `TileLayerNormAffineOnLoad` + `EpilogueAscend950LayerNormApply`

| 槽 | 运算 | 决策 | 组件 |
| --- | --- | --- | --- |
| 1 | 阶段2 的 A 装载路径：`(x-μ)·rstd·γ+β`（UB, FP32，γ/β tile 常驻） | 🔧 **自定义 Tile** | 复用 `TileRowBroadcastMul` / `TileBroadcastAdd` 的签名范式 [仓 `tile_broadcast_mul.hpp:28`] |
| 2 | （档 2）行标量乘 `rstd[m]` | ✅ | `TileBroadcastInplaceByColumn`（方向需按头文件注释确认） |
| 3 | （档 2）秩 1 减 `(rstd⊙μ)⊗g` | 🔧 **自定义 Tile** `TileRowScalarOuterSub` | 长度 `TileM1` 的向量 op |
| 4 | 行加 `b[n']` + cast + 搬出 | ✅ | `TileBroadcastAdd` + `TileCast` + `CopyUBToGM` |

> **粒度声明**：E1/E2 的 Block 主体为**粒度 B**（新建 `EpilogueDispatchPolicy` 特化 + 完整 Tile 槽序列，预估各 250~450 行），原因是「操作数跨 N-block + 需跨 tile 携带行归约状态」，任何现有单 tile 特化的槽位签名都无法容纳（见「操作数拓扑判定」）；而其中 3 个自定义 Tile 本身是**粒度 A**（各 ~80 行，严格对齐现有 Tile 的 `operator()` 签名与 `TileShape/COMPUTE_LENGTH/ElementCompute` typedef）。

**（4）BlockScheduler**

| 项 | 选型 | 说明 |
| --- | --- | --- |
| 调度器 | `GemmIdentityBlockSwizzle<3, 0>`（M0 ≥ M1 段）/ `<3, 1>`（M0 < M1 段） | 按 M/N 相对大小选 direction；[仓] 43 号样例实证两分支 |
| 分核维度 | 档 1 沿 **M0** 切 band；档 2 追加沿 **N0** 切（`nSplit`） | |
| L1 常驻 | `enableL1Resident=false` 起步；仅档 3 开启 | 43 实证 false |

**（5）Kernel**

| 项 | 选型 |
| --- | --- |
| Kernel 类型 | **新增 `Gemm::Kernel::MatmulLayerNormMatmul`**，模板形参 `<BlockMmad1, EpiloguePrep, BlockMmad2, EpilogueApply, BlockScheduler, WorkspacePolicy>`（对照 [仓] `HstuInfer<BlockMmadQK, BlockMmadPV, EpilogueSiluScale, EpilogueCast, ...>`） |
| 落位 | 先在样例目录 `experimental/matmul/matmul_layernorm_matmul/kernel/` 成型（`examples/83`、`examples/49` 为同类先例），稳定后上提 `include/catlass/gemm/kernel/` |
| 调用 | 样例入口走 host 组装；optest 的 JIT 模板统一 `Catlass::RunKernel<Kernel>()`，**不使用 `device_gemm.hpp`**；TileShape 经 `CatlassKernel::TileShapeScaler<ElementA, half, BaseShape>` 缩放，不硬编码 |

### host侧设计

1. **参数校验**：A0/B0/B1/C1 rank==2、gamma/beta rank==1 且长度恰为 `N0`；`A0.cols==B0.rows==K0`、`B0.cols==B1.rows==N0`、`B1.cols==M1`；dtype 固定（A0/B0/B1/C1 = FP16，gamma/beta = FP32）；布局按任务书（A0 RowMajor，B0/B1 ColumnMajor）；非 ND / 非连续 → 返回不支持，不 launch。
2. **TileShape 决策（表驱动）**：按 `(M0,N0,M1)` 查分档表取 `L1/L0TileShape`，并做容量自检（L1 ≤ 512KB、L0C ≤ 256KB、UB ≤ 248KB）；不满足逐级降 `TileM/TileN`，最劣退到 `128×128×128`（仍保证正确）。
3. **分核**：`bands = ⌈M0/TileM⌉`；`bands ≥ aicCoreNum` → 直接按 band 切；`bands < aicCoreNum`（M0=128 段典型情形）→ v1 接受并行度损失但保持无跨核同步，同时保留 `nSplit` 开关供档 2 调优对比。`aicCoreNum` **一律运行时取**（`PlatformAscendC::PlatformAscendCManager::GetInstance()->GetCoreNumAic()` [仓 examples/43]），禁止硬编码。
4. **分支实例化条件**：

| 条件 | 取值 | 影响的模板/分支 |
| --- | --- | --- |
| M0 与 TileM 整除性 | 整除 / 尾带 | 尾块 mask 分支（不做 padding kernel） |
| bands vs AIC 数 | band-major / nSplit | BlockScheduler + 是否启用档 2 归约 |
| Swizzle 方向 | `<3,0>` / `<3,1>` | BlockScheduler |
| 快路径开关 | 档 3 on/off | `WorkspacePolicy` |

**v1 只实例化「档 1 + band-major + 单一 swizzle」一组**，先用它打通 116 例，再按性能结论扩实例（避免组合爆炸与 JIT 时间浪费）。

5. **tiling 数据**：`{M0,K0,N0,M1, TileShape 三元组, bands, aicCoreNum, nSplit, 档3 开关, workspace 偏移表}`；kernel 侧不反推形状。

### kernel侧设计

**数据流（档 1，单次 launch）**

```text
┌──────────── 单次 launch（一个 grid，AIC + AIV 双 role）────────────┐
[阶段1] mm1：对每个 band(TileM×N0)，沿 K0 循环 mmad → L0C(FP32)
        │  E1：L0C→UB → cast FP16 → 累加 Σx/Σx²（不折 γ）→ 写 workspace[C0 band]
        │  统计量仅 TileM×2 float，留在本核 UB（不外露）
        │  band 完成 → 本核算 μ[m]、rstd[m]（TileM 个）→ 无跨核 barrier
[阶段2] mm2：沿 K=N0 循环
        │  A：workspace[C0] → UB → TileLayerNormAffineOnLoad：(x-μ)·rstd·γ+β（FP32）
        │      → L1 → L0A ；  B：B1 tile → L1 → L0B（不改动）
        │  → L0C 累加 P(FP32) → cast FP16 → C1 → GM
└──────────────────────────────────────────────────────────────────┘
[阶段0] 仅档 2 需要：g = γᵀB1、b = βᵀB1（长 M1，FP32）+ 一次跨核 barrier
```

**关键设计点**

- **AIC/AIV 分工**：AIC 发两段 mmad；AIV 做 E1 的行归约与 E2 的 LN 仿射。950 Cube:Vector = 1:2，LN 的向量负载有对等算力可用；目标是让 AIV 的归一完全落在 AIC 的 mmad 影子里。
- **同步**：950 由 set/wait 强配对切换为 **BufferID 同步**；`enableUnitFlag=true` 允许 L0C 边算边搬出 [仓 examples/43]。档 1 无跨核数据依赖（除档 2 的阶段 0）。
- **事件预算（防死锁）**：同一 `HardEvent` 类型（如 `V_MTE2`）的**事件 ID 上限为 8**，超限会**运行期挂死** → 阶段 1/2 的事件 ID 必须在写码前数一遍；v1 保守取 `UB_STAGES=1`，实测无冲突再升。
- **UB 预算**：`C0 tile(half) TileM×TileN×2` + `FP32 工作区 TileM×TileN×4` + `γ/β tile 8×TileN` + `Σx/Σx² 8×TileM` ≤ **248KB**。
  - `TileM=128, TileN=256` → 64+128+2+1 ≈ **195KB** ✓，但余量不足以再开一组双缓冲 → 只能 `UB_STAGES=1`（与事件 ID 结论一致）；
  - `TileN=128` → ≈ **99KB** → 可升 `UB_STAGES=2`。**`TileN ↔ 流水深度` 的交换比是性能调优的第一个单变量实验。**
- **Quick return / 边界**：`M0==0 || M1==0` 直接返回成功；`N0==0`、`K0==0` 视为非法参数返回错误码。

### workspace 设计

| 区 | 大小 | 对齐 | 说明 |
| --- | --- | --- | --- |
| `C0`（mm1 输出，**未归一、未折 γ**） | `M0×N0×sizeof(half)` | 512B | 主体中间量；最大 32MB、平均 6.3MB [算]。归一在阶段2 的装载路径完成 → **不存在 `C0_norm` 区** |
| `Σx`、`Σx²` | `2×M0×sizeof(float)` | 512B | 内部统计量，不对外暴露 |
| `g`、`b`（仅档 2） | `2×M1×sizeof(float)` | 512B | 阶段 0 产出 |
| 档 2 归约暂存 | `nSplit×M1×sizeof(float)×2` | 512B | 可与 `g/b` 复用 |

`GetWorkspaceSize(...)` 与 host 分配**必须同式**（不得按 `M*N` 粗估）；optest 侧统一走 `tests/optest/kernels/common/workspace_alloc.h` 的 `g_catlassWorkspaceAlloc`。

### 精度设计

1. **累加**：两段 mmad 一律 `CType=float`；optest 侧 **`#define CATLASS_JIT_ELEMENT_C float` 写死**（非 `#ifndef`），否则被 adapter 的 `outDType` 覆盖成 `half` → workspace 步长错乱出 inf。
2. **统计域**：μ/σ² 全程 FP32；`var = Σx²/N0 − μ²`（与 Σx² 同序累加），**不做 Bessel 修正**。
3. **rsqrt**：优先 `RsqrtFast/Rsqrt`；`rstd` 的相对误差对整行是共模的，且同时作用于主积缩放与 μ 校正 → 误差不会抵消，需实测决定是否补一步 Newton 精修。
4. **仿射域**：γ/β 保持 FP32 乘加，最后一步 `TileCast` → FP16；**golden 必须镜像"先 FP32 计算、最后 cast"的顺序**。
5. **与标杆的偏差点（分方案写进自验证报告）**：
   - **档 1**：A 在 UB 内 FP32 算完 LN 后**仍舍 FP16 再喂 cube** → 舍入点与标杆一致，偏差仅来自归约顺序与 rsqrt 实现 → 精度风险最低；
   - **档 2′（难例方案）**：与档 1 同样的装载路径归一 ⇒ 舍入点仍与标杆一致，只额外承担一次跨核 barrier 与 `C0` 的 L2 重读；
   - **纯代数延迟校正（备选）**：实测在 `C0` 量级抬升后会失配（`matched_ratio` 掉到 0.973），**必须**把 `A` 以 FP32 存储才可用；若采用需在自验报告中含大值/小方差输入的回归例。

## 支持硬件

| 支持的芯片 | 说明 |
| --- | --- |
| **Ascend 950PR / 950DT（`Arch::Ascend950`，arch `3510`）** | 本任务目标平台，样例编译需 `-DCATLASS_ARCH=3510` |
| Atlas A2 / A3（arch `2201`） | 本期不做。950 专用件（`copy_l0c_to_ub`、`MmadPingpongMutex`、`EpilogueAscend950*` 等）在 A2 上无对应实现，跨代需另立 DispatchPolicy 特化 |

环境配套：catlass 主干 v2.1.0（commit `165c7a1`）+ CANN 9.0.0（950 C++ 路线最低版本）+ Ubuntu 22.04.5 / gcc 11.3 / cmake 3.22 / python 3.10（optest 需 python 3.11+ 与 TorchNPU）。

## 算子约束限制

任务书「算子约束限制」为**无**；以下为设计推导出的**实现期自设约束**，将在样例 README 与 optest 参数校验中同步声明：

1. 数据类型固定：A0/B0/B1/C1 = FP16，gamma/beta = FP32；
2. 仅支持 ND 与任务书指定布局（A0 RowMajor；B0/B1 ColumnMajor），不支持 NZ/非连续/转置入参；
3. `M0 ≥ 1, K0 ≥ 1, N0 ≥ 1, M1 ≥ 1`；gamma/beta 长度必须恰为 `N0`（不接受 broadcast 形态）；
4. workspace 需求（`M0×N0×2B` + 小量）超过设备可分配量 → 返回 NOT_SUPPORTED；
5. 档 3（片上零往返快路径）仅在 `TileM×N0×2B ≤ L1 余量` 且 `bands ≥ AIC 数` 时启用，默认关闭；
6. 档 2（跨核切 N0）默认关闭：涉及跨核 barrier 与归约顺序不确定性，仅在难例上开启并单独验证。

---

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | 满足生态算子开源精度标准（**单标杆 + 混合容差**逐元素：`|actual-golden| ≤ atol + rtol·|golden|`）。FP16 输出：`rtol = atol = 2^-9 ≈ 1.95e-3`，`matched_ratio ≥ 0.99`，且 `max_abs_error ≤ max(1e-1, 32·ULP)`（FP16 `ULP@1.0 = 2^-10` → 上限 0.1）。**不允许在实现中放宽阈值** | [生态算子开源精度标准](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md) |
| 精度标杆 | `torch.mm → F.layer_norm(eps=1e-6) → torch.mm`（TorchNPU / CPU 同精度实现）为参考；另备 FP32/FP64 高精度实现用于定位误差来源；catlass 仓内 `golden/compare_data.hpp` 的 MARE/MERE/RMSE 比值法作辅助诊断 | 任务书测试标准 |
| 性能标准 | **平均(标杆时延 / 本算子时延) > 1.1**；工具 `msprof op`（小 shape 建议 `--warm-up 30`），warmup 后多次采样取均值；不同实现方案 / TileShape 需分别备注 | 任务书性能要求 + [catlass 性能调测文档](https://gitcode.com/cann/catlass/blob/master/docs/zh/1_Practice/evaluation/performance_tools.md) |
| 内存标准 | 记录 workspace 峰值占用与对外存分配（仅输出 C1）；与标杆的显存占用对比（融合应显著少一份 `M0×N0` FP16 中间量） | 任务书自验证报告要求 |

**性能预判与达成路径**

流量模型（每例，`sz=2B`）：
- 必需流量（两方案相同）：`2·(M0K0 + K0N0 + N0M1 + M0M1) + 4·N0` 字节；
- 标杆额外：`4·M0·N0·sz`（写 C0 / 读 C0 / 写 C0_norm / 读 C0_norm）+ 3 次 launch；
- 本设计档 1 额外：`2·M0·N0·sz`（写+读 workspace，大概率 L2 命中）+ 1 次 launch；
- FLOPs 两者相同：`2·M0·K0·N0 + 2·M0·N0·M1`。

116 例统计 [算]：零往返口径 `mean(上界)=1.645`（时延加权 1.676）；档 1 口径 `mean(上界)=1.32`。**M0=128 段 30 例（含 14 例零往返都达不到 1.1×：idx 3,12,15,18,19,20,21,23,24,26,27,105,106,107）是达标风险集中区**，其收益主要来自"LN kernel 整体消失 + 3 launch→1 launch"而非流量；该段需靠档 2（提并行度）或档 3（消除往返）补足。

达成路径（按优先级）：① 档 1 打全量，拿"少 2 遍往返 + 少 2 次 launch + 无阶段 0"的确定收益；② 阶段 1/2 的 L0C/UB 双缓冲与 `MmadPreloadAsyncWithCallback` 重叠 AIC/AIV；③ `TileN↔UB_STAGES` 交换比、L1/L0 TileShape、swizzle 方向按**单变量**轮换调优；④ M0=128 段上档 2 或档 3。

## 测试方案

| 层次 | 内容 | 通过判据 |
| --- | --- | --- |
| Tile 级 UT | `TileRowSumAndSumSq`、`TileLayerNormAffineOnLoad`、（档 2）`TileRowScalarOuterSub`、`copy_l0c_to_ub` 950 通路 | 单 tile 与 CPU 参考逐元素在混合容差内通过 |
| 组件级 | E1/E2 组合、L0C/UB 预算、事件 ID 数 | 无死锁；UB 峰值 ≤ 248KB |
| 冒烟 | 分层用例集 5 例：idx `1, 45, 54, 90, 116` | 精度通过 + 单 launch 确认 |
| optest 全量 | 任务测试集 **116 例**（`pytest tests/test_matmul_layer_norm_matmul.py`） | 全部通过（混合容差） |
| ATK 泛化 | **≥200 例**：116 全留 + 边界/非对齐/随机扩样 ≥84 | `atk task --task accuracy` 全通过 |
| 性能 | 116 例 `msprof op` 全量采集 + 标杆同条件复测（偏差 <5%） | 平均比值 > 1.1；逐例数据 + 方案/TileShape 备注表 |
| 边界/负向 | 常量行（σ²=0）、γ≡1/β≡0、奇数维、非 16 倍数、8192 极端维、非法参数（rank/dtype/长度/零尺寸） | 精度通过 / 正确返回错误码且不 launch |

**可维可测性设计**：阶段 1/2 可分别用 dump 点与 `AscendC::PipeBarrier` 单独验证（`ascendc_dump` / `print` / `msdebug`）；μ、σ²、`g`、`b`、`P` 在 **debug 构建**下可选落到 workspace 调试区（release 不分配）；TileM/TileN/TileK、各 stage 数、`UB_STAGES`、`nSplit`、档 3 开关、swizzle 全部走 tiling 可调，便于逐参数归因。无卡期用 `bash scripts/build.sh --simulator <target>` + `msprof op simulator` 取流水图与热点（注意：仿真只能 0 卡）。

## 风险与规避措施

| # | 风险 | 影响 | 规避 |
| --- | --- | --- | --- |
| 1 | 仓内无 LayerNorm 组件，需新写 Block/Tile 两级 | 工期主要在 epilogue 组件 | 自底向上：先 Tile 级 UT，再 E1/E2，最后 kernel 编排；结构照 `examples/83 HstuInfer` |
| 2 | 「每 slot 单 tile」的轮转 workspace 在跨 N 归约下会整片算错 | 精度大面积失败 | 「操作数拓扑判定」已显式否决该结构；行完整性由档 1 两阶段保证 |
| 3 | 跨核 barrier / 事件 ID 超 8 | 运行期挂死 | v1 主方案**无跨核同步**；档 2 默认关闭，开启前先数事件 ID 并做死锁压测 |
| 4 | 档 2 的未归一中间量：FP16 溢出 + **精度随动态范围退化**（已实测：`\|A\|` 达 FP16 上限 10% 即 `matched_ratio=0.973` 不达标） | 整 band 变 inf 或大面积超差 | 难例改走**档 2′（装载路径归一）**；纯代数形式必须 FP32 存 `A`；探针脚本已入交付工具链，量级实验可复跑 |
| 5 | M0=128 段 30 例融合收益不足 | 平均 1.1× 不达标 | 档 2 提并行度 / 档 3 消往返；同时确认性能统计口径（见待确认事项）；实测空 kernel launch 开销以判断"少 2 次 launch"的真实收益 |
| 6 | optest JIT 的 CType 被宏覆盖、JIT 缓存命中旧 `.so` | 假故障难查 | `#define CATLASS_JIT_ELEMENT_C float` 写死；模板变更后清 `~/.cache/catlass/jit_cache/` |

## 待确认事项（请评审在 PR 评论中反馈）

| # | 问题 | 对设计的影响 | 本文暂按 |
| --- | --- | --- | --- |
| Q1 | 性能「平均标杆时延/测试时延 > 1.1」是 `mean(bench_i/ours_i)`，还是 `Σbench/Σours`？是否要求逐例下限？ | 决定主攻 14 难例还是 44 个大时延例 | 两者都保（从严），先攻均值再补难例 |
| Q2 | 「无需中间显存读写」是否允许内部 GM workspace 承载 `C0` 本体（任务书仅明确 workspace 管 μ/σ²）？ | 决定档 1 主方案能否成立 | 允许（档 3 作为满足严格解读的兜底快路径） |
| Q3 | 「单次 kernel launch」的判定：是否允许一次 launch 内含多阶段 + AIC/AIV 双 role？ | 档 1/档 2 的合法性 | 允许一个 grid 派生多阶段 |
| Q4 | 精度标杆是否**只**以昇腾小算子拼接为准（还是需 CPU 高精度交叉）？γ/β 可否以属性/常量形态传入？ | 影响精度实现与自验口径 | 以 NPU 拼接为验收标杆 + CPU 高精度交叉校 |
| Q5 | 样例编号与落位：`experimental/matmul/matmul_layernorm_matmul/` 单点，还是同时新增 `examples/NN_*`？主干样例编号已用至 83（`experimental` 下测试件已见 85/86），本算子建议 87+ | PR 目录结构 | 仅进 `experimental/matmul/`，编号待评审指定 |
| Q6 | 目标硬件写作 "Ascend 950"，性能标杆数据在 950PR 采集：是否要求同时在 950DT 验证？ | 是否需双 SKU 回归 | 先在 950PR 交付，DT 作为后续回归 |

## 兼容性分析

- **接口兼容**：全新算子，无历史行为变更；对 catlass 的改动全部为**新增文件 + 新增 ABI 声明**（`catlass_kernel_jit.h` 追加函数签名与 `MatmulLayerNormMatmulParams : MatmulParams`），不修改任何既有签名 → 既有样例与测试件不受影响。
- **组件兼容**：新增 `EpilogueAscend950LayerNorm*` 两个 BlockEpilogue 特化与 3 个自定义 Tile，均为新 DispatchPolicy 分支，不与现有 policy 冲突；若评审要求，可全部先落在样例目录，仅把稳定件上提到 `include/catlass/`。
- **版本兼容**：CANN 9.0.0 起（950 C++ 路线）；不引入 CATLASS-DSL 依赖（避免 9.1.0 + wheel 的额外约束与 JIT 启动开销）。
- **架构兼容**：`Arch::Ascend950` 专属；A2/A3 需另做特化，本期不承诺。
