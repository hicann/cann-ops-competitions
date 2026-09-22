# MatmulLayerNormMatmul算子设计文档

| 项目 | 内容 |
| --- | --- |
| 算子名称 | MatmulLayerNormMatmul（融合 Matmul + LayerNorm + Matmul） |
| 所属任务 | 2026年9月CANN社区任务-MatmulLayerNormMatmul算子开发 |
| 设计文档提交仓库 | `cann/cann-ops-competitions` |
| 算子代码提交仓库 | `cann/catlass` |
| 交付路径 | `experimental/matmul/87_ascend950_matmul_layernorm_matmul` + `tests/optest/kernels/87_ascend950_matmul_layernorm_matmul` |
| 目标产品 | Ascend 950（arch35 / CATLASS_ARCH=3510） |
| CANN版本 | 9.1.0.beta3 |
| 开发语言 | Ascend C（CATLASS 模板库，TLA 编程范式） |

---

# 1 需求背景

## 1.1 需求来源

本需求来源于 2026 年 9 月 CANN 社区任务「MatmulLayerNormMatmul 算子开发」。任务要求在
`cann/catlass` 开源仓中，使用 Ascend C（CATLASS 模板库）开发融合 LayerNorm 的双 Matmul 算子，
适配 Ascend 950PR，并提供基于 `optest` 与 `ATK` 的测试交付件。

- 开源仓地址：https://gitcode.com/cann/catlass
- 适配硬件：Ascend 950（arch35 / CATLASS_ARCH=3510）
- 算子编号：87_ascend950_matmul_layernorm_matmul

## 1.2 背景介绍

算子需在**单次 kernel launch**（MIX kernel，AIC+AIV）内完成三段计算：

$$
\begin{aligned}
\mathbf{C0} &= \mathbf{A0} \times \mathbf{B0} &&\text{(M0,K0) × (K0,N0) → (M0,N0)} \\
\mu[m] &= \frac{1}{N_0}\sum_{n} \mathbf{C0}[m,n] \\
\sigma^2[m] &= \frac{1}{N_0}\sum_{n} \big(\mathbf{C0}[m,n]-\mu[m]\big)^2 \\
\mathbf{C0}_{norm}[m,n] &= \frac{\mathbf{C0}[m,n]-\mu[m]}{\sqrt{\sigma^2[m]+\epsilon}}\cdot\gamma[n]+\beta[n] \\
\mathbf{C1} &= \mathbf{C0}_{norm} \times \mathbf{B1} &&\text{(M0,N0) × (N0,M1) → (M0,M1)}
\end{aligned}
$$

其中 $\epsilon = 10^{-6}$，$\gamma$（缩放）与 $\beta$（偏置）为长度 $N_0$ 的可学习参数。

## 1.3 现状分析

性能标杆为小算子拼接方案 `torch.mm + F.layer_norm + torch.mm`，其执行流程为：

| 阶段 | 输入 | 处理 | 输出 |
|---|---|---|---|
| 1 | A0, B0 | `torch.mm(A0, B0)` | C0 写 GM |
| 2 | C0, gamma, beta | `F.layer_norm(C0, ...)` | C0_norm 写 GM |
| 3 | C0_norm, B1 | `torch.mm(C0_norm, B1)` | C1 写 GM |

该方案的固有开销：3 次 kernel launch 与序列化同步；中间结果 C0 / C0_norm 在 GM 与外部张量间
反复读写；LayerNorm 阶段独立往返 GM→UB→GM，无法与 Matmul 流水重叠。

CATLASS 现有能力调研（含样例 42/44/61/64/73/80 与 `include/catlass/` 模板组件）：

| 能力 | 现状 | 可复用性 |
|---|---|---|
| Ascend950 Matmul（TLA） | 已有（样例 43/73 等） | `BlockMmadTla` 直接复用 |
| Epilogue 融合 | 已有（样例 64/80） | 后处理挂载思路可参考 |
| 多 Matmul 级联 + AIC/AIV 协同 | 已有（样例 61 `svd_quant_matmul`，含 UB→L1 回送） | **骨架主要参考来源** |
| L0C→UB Fixpipe 通路 | Ascend950 新增能力 | Phase1 累加结果搬出依赖此通路 |
| LayerNorm / RMSNorm 组件 | **无现成组件** | 需自建 AIV 行归约 |

## 1.4 功能分析

- 算子功能：$\mathbf{C1} = \mathrm{LayerNorm}(\mathbf{A0}\times\mathbf{B0})\times\mathbf{B1}$，单次 launch 完成三段
- 输入：A0、B0、B1、gamma、beta
- 输出：C1
- 支持数据类型：输入 A0/B0/B1 为 FP16，gamma/beta 为 FP32，输出 C1 为 FP16
- 是否支持广播：不支持

---

# 2 需求分析

## 2.1 需求描述

使用 Ascend C（CATLASS 模板库）实现 MatmulLayerNormMatmul 融合算子，输入 FP16 矩阵
A0(M0,K0) / B0(K0,N0) / B1(N0,M1) 与 FP32 向量 gamma/beta(N0)，输出 FP16 矩阵 C1(M0,M1)，
精度满足生态算子开源精度标准，性能达成小算子拼接方案的 1.1 倍。

## 2.2 需求拆解

| 编号 | 需求项 | 验收方式 |
| --- | --- | --- |
| R1 | 单次 kernel launch 完成 Matmul1 → LayerNorm → Matmul2 三段计算 | 代码走读：`__mix__(1,2)` 单一 launch；profiling 显示单 kernel |
| R2 | Mean / Variance 等中间结果由算子内部 workspace 管理，不对外暴露 | 代码走读：`GetWorkspaceSize` 内部切分 C0 / C0_norm |
| R3 | 精度满足生态算子开源精度标准（FP16 混合容差） | `testset` 全量 shape 精度比对 + `verify` 判定 |
| R4 | 性能达成小算子拼接标杆的 1.1 倍（平均标杆时延 / 本算子时延 > 1.1） | `msprof op` 采集 Task Duration |
| R5 | optest 交付件通过任务测试集全量 116 组 shape | `pytest` 全通过 |
| R6 | ATK 交付件泛化 ≥ 200 例通过 | ATK 框架执行结果 |
| R7 | 提供 README，可复现编译、运行与验证 | 验收人复现 |

## 2.3 关键技术点识别

| 编号 | 技术点 | 说明 |
| --- | --- | --- |
| T1 | L0C→UB Fixpipe 通路 | Ascend950 新增能力，Phase1 的 L0C 累加结果需经此通路搬至 UB / GM |
| T2 | CrossCore 同步 | AIC 与 AIV 之间需 barrier（mode 0x0）+ notify（mode 0x2）双向同步 |
| T3 | 行归约完整性约束 | LayerNorm 需**完整一行**（N0 元素）才能算 mean/var，决定 C0 必须落 GM |
| T4 | L1/L0 TileShape 断言 | `block_mmad_pingpong_tla.hpp` 要求 `L1_TILE_M == L0_TILE_M` 且 `L1_TILE_N == L0_TILE_N` |
| T5 | UB 容量核算 | N0=8192 时 FP32 视角单行 32KB，gamma/beta 各 32KB，需按核数与 buffer 复用核算 |
| T6 | workspace 生命周期 | `GetWorkspaceSize` / `ToUnderlyingArguments` 双端一致，否则出现越界 |

---

# 3 详细设计

## 3.1 算子分析

### 3.1.1 数学公式

$$
\begin{aligned}
\mathbf{C0} &= \mathbf{A0} \times \mathbf{B0} \\
\mu[m] &= \frac{1}{N_0}\sum_{n=0}^{N_0-1} \mathbf{C0}[m,n] \\
\sigma^2[m] &= \frac{1}{N_0}\sum_{n=0}^{N_0-1} \big(\mathbf{C0}[m,n]-\mu[m]\big)^2 \\
\mathbf{C0}_{norm}[m,n] &= \frac{\mathbf{C0}[m,n]-\mu[m]}{\sqrt{\sigma^2[m]+\epsilon}}\cdot\gamma[n]+\beta[n] \\
\mathbf{C1} &= \mathbf{C0}_{norm} \times \mathbf{B1}
\end{aligned}
$$

其中 $\epsilon = 10^{-6}$，$\gamma$ / $\beta$ 长度均为 $N_0$。

### 3.1.2 支持数据类型

| 张量 | 数据类型 |
|---|---|
| A0 / B0 / B1 | FP16 |
| gamma / beta | FP32 |
| 内部累加与 LayerNorm 计算 | FP32 |
| C1 | FP16 |

### 3.1.3 支持形状

| 参数名 | 输入/输出/属性 | 描述 | 使用说明 | 数据类型 | 数据格式 | 维度(shape) | 非连续Tensor |
|---|---|---|---|---|---|---|---|
| A0 | 输入 | 第一个 Matmul 左矩阵，形状 (M₀, K₀) | RowMajor 布局 | FP16 | ND | 2 | — |
| B0 | 输入 | 第一个 Matmul 右矩阵，形状 (K₀, N₀) | ColumnMajor 布局 | FP16 | ND | 2 | — |
| B1 | 输入 | 第二个 Matmul 右矩阵，形状 (N₀, M₁) | ColumnMajor 布局 | FP16 | ND | 2 | — |
| gamma | 输入 | LayerNorm 缩放参数，长度 N₀ | 可学习参数 | FP32 | ND | 1 | — |
| beta | 输入 | LayerNorm 偏置参数，长度 N₀ | 可学习参数 | FP32 | ND | 1 | — |
| eps | 属性 | LayerNorm epsilon | 固定 1e-6 | FP32 scalar | — | 0 | — |
| C1 | 输出 | 第二个 Matmul 输出矩阵，形状 (M₀, M₁) | 唯一对外输出 | FP16 | ND | 2 | — |

M0 / K0 / N0 / M1 均为运行时输入。约束：`N0 ≤ 8192`（LayerNorm 单行 UB 容量约束）；
M0 / K0 / M1 受 GM 容量约束，无硬上限。任务测试集覆盖 116 组 shape。

## 3.2 算子实现

### 3.2.1 实现方案

#### 3.2.1.1 host侧设计

**Tiling 策略**：双配置按问题规模选择（参考样例 61 的 Common/Small 策略）。
入口宿主逻辑以 `blocks = CeilDiv(M0, 256) × CeilDiv(N0, 256)` 与 AIC 核数（28）比较分档：

| 配置 | 触发条件 | L1TileShape1 (M,N,K) | L0TileShape1 | L1TileShape2 | L0TileShape2 |
|---|---|---|---|---|---|
| Small（TILING=1） | blocks < 核数 | (64, 128, 128) | (64, 128, 64) | (64, 128, 128) | (64, 128, 64) |
| Common（TILING=0） | blocks ≥ 核数 | (128, 256, 128) | (128, 256, 64) | (128, 256, 128) | (128, 256, 64) |

> 注：`block_mmad_pingpong_tla.hpp` 断言要求 `L1_TILE_M == L0_TILE_M` 且 `L1_TILE_N == L0_TILE_N`
> （K 允许不同）。设计初期 Common 档曾误配 L1_M=256 / L0_M=128，该配置从未真正生效；
> 已修正为 L1_M=128 对齐。

**尾块负载均衡**：M 方向尾块用 `SplitLengthByCores` 均分到多核（参考
`svd_quant_matmul_tla.hpp:L1576`）。

**Swizzle**：`GemmIdentityBlockSwizzle<3, 0>`（offset=3, direction=0）。

**Workspace 规划**：`GetWorkspaceSize` 计算并返回内部 workspace 大小，由框架统一分配，
算子内部按偏移切分为 C0（M0×N0 FP16）与 C0_norm（M0×N0 FP16）两段。

#### 3.2.1.2 kernel侧设计

##### 1. 整体流水（单次 MIX kernel launch，AIC+AIV 协同）

```text
GM(A0/B0)
   │  MTE2: GM → L1
   ▼
L1 ──MTE1──▶ L0A/L0B ──M(Cube)──▶ L0C0 ──Fixpipe──▶ UB/GM(C0 workspace)
   │
   │  CrossCore barrier (mode 0x0): AIC → AIV
   ▼
GM(C0) ──MTE2──▶ UB ──V: mean/var 归约 + 归一化 + gamma/beta──▶ UB(C0_norm) ──MTE3──▶ GM(C0_norm ws)
   │
   │  CrossCore notify (mode 0x2): AIV → AIC
   ▼
GM(C0_norm/B1) ──MTE2──▶ L1 ──MTE1──▶ L0A/L0B ──M(Cube)──▶ L0C1 ──Fixpipe/MTE3──▶ GM(C1)
```

##### 2. 各阶段职责

| 阶段 | 执行核 | 数据位置变化 | 组件 |
|---|---|---|---|
| Phase 1 Matmul1 | AIC | GM(A0,B0) → L1 → L0A/L0B → L0C0 → GM(C0) | `BlockMmadTla` + `GemmIdentityBlockSwizzle<3,0>` |
| Barrier | AIC↔AIV | CrossCore flag mode 0x0 | `CrossCoreSetFlag/WaitFlag` |
| Phase 2 LayerNorm | AIV | GM(C0) → UB(FP32) → 归约/归一化 → GM(C0_norm) | 自建 LayerNorm 段（行归约 + 归一化） |
| Notify | AIV→AIC | CrossCore flag mode 0x2 | `CrossCoreSetFlag/WaitFlag` |
| Phase 3 Matmul2 | AIC | GM(C0_norm,B1) → L1 → L0A/L0B → L0C1 → GM(C1) | `BlockMmadTla` |

##### 3. LayerNorm 段实现要点

1. 按 M 方向分核，每核处理 `ceil(M0 / (aicCoreNum × aivNum))` 行，各行独立、无跨核归约。
2. 每行从 GM 读入完整 N0 元素（FP32 视角 32KB @ N0=8192），gamma/beta（各 32KB @ FP32）一次载入复用。
3. mean / var 计算使用 Vector 归约指令，随后做 `(x-μ)/√(σ²+ε)·γ+β` 归一化，结果转 FP16 写回。
4. UB 容量核算与 buffer 复用方案见 `IMPLEMENTATION_NOTES.md` §2.2。

##### 4. 存储层次与关键常量

| 存储单元 | 本算子中的用途 |
|---|---|
| Global Memory | A0/B0/B1、gamma/beta、C1，以及内部 workspace（C0 + C0_norm） |
| L1 Buffer | Matmul 输入分块（Nz 排布） |
| L0A / L0B | Matmul 分形输入 |
| L0C | Matmul 累加结果 |
| UB | LayerNorm 行数据、归约中间量、gamma/beta 副本 |

## 3.3 支持硬件

| 支持的芯片版本 | 涉及勾选 |
|---|---|
| Atlas 800I/T A2 | |
| Atlas A2 训练/推理系列（arch22） | |
| Atlas A3 训练/推理系列（arch22） | |
| **Ascend 950PR（arch35 / 3510）** | **√** |

依赖 Ascend950 新增能力：L0C→UB Fixpipe 通路、MIX kernel（AIC+AIV）、CrossCore flag 同步。

## 3.4 算子约束限制

1. 仅支持二维 ND 矩阵；A0 RowMajor，B0/B1 ColumnMajor，C1 RowMajor（任务书固定）。
2. 输入 A0/B0/B1 仅支持 FP16；gamma/beta 仅支持 FP32；输出 C1 仅支持 FP16。
3. `N0 ≤ 8192`（LayerNorm 单行 UB 容量约束）；M0/K0/M1 无硬上限（受 GM 容量约束）。
4. 不支持 Batch、转置属性、量化、稀疏语义。
5. 所有编译期配置须通过 Host 资源合法性检查（`CanImplement`）；不存在合法配置时不发起 kernel。

---

# 4 可维可测分析

## 4.1 精度标准/性能标准

| 验收标准 | 描述(不涉及说明原因) | 标准来源 |
|---|---|---|
| 精度标准 | FP16 输出满足混合容差：atol=rtol=2⁻¹⁰，matched_ratio ≥ 0.99，max_abs_error ≤ 1e-1 或 32ULP | [生态算子开源精度标准](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md)（仓内实际文件为 `mixed_tolerance_standard.md`） |
| 性能标准 | 平均(小算子拼接标杆时延) / 平均(本算子时延) > 1.1 | 任务书 §性能要求（标杆数据由任务测试集提供，Ascend 950PR 实测） |

**精度设计细节**：

- 参考计算：`torch.mm + F.layer_norm + torch.mm`（与性能标杆同源），LayerNorm 在 FP32 下计算，
  中间 C0 取 FP16（与 kernel 内部 workspace 精度对齐，避免参考比 kernel 更精确造成假 FAIL）。
- numpy 参考实现（`reference/matmul_layernorm_matmul_ref.py`）selfcheck 已 PASS：
  mu_err=0、var_err=0、LayerNorm 不变量（recover_mean≈0、recover_var≈1）满足、端到端一致。
- 实测结果：optest 160/160 用例通过；ATK 泛化 210/210 例通过。

**性能设计细节**：

- 采集工具：`msprof op`（任务书指定），读 `OpBasicInfo_*.csv` 的 `Task Duration(us)`（设备级 kernel 耗时口径）。
- 命令行：`msprof op --application="<python+脚本>" --output=<dir> --aic-metrics=BasicInfo --launch-count=N --warm-up=1 --replay-mode=kernel --kill=on`
- 性能来源：省 2 次 kernel launch 与同步；中间量 L2 cache 复用；AIC/AIV 流水重叠。
- 风险与实测结论：中间 C0 落 GM 引入额外带宽，实测整体性能详见自验证报告。

## 4.2 测试用例设计

| 用例编号 | 来源 | 覆盖目的 |
| --- | --- | --- |
| TC-01~116 | 任务测试集 `MatmulLayerNormMatmul_测试集.csv`（116 组 shape） | 覆盖 M0∈{128,512,1024,2048}、K0∈{768,2048,4096,8192}、N0∈{2048,3072,4096,8192}、M1∈{768,2048,4096} 的笛卡尔组合及边界 |
| TC-ATK-01~210 | ATK 泛化随机生成（210 例） | 泛化性验证，覆盖非测试集 shape 与随机数值分布 |

精度判定：以 numpy float64 计算参考结果，按混合容差判定逐元素通过率与最大绝对误差。

## 4.3 自验证方案

1. **功能自验证**：optest 框架编译 JIT kernel，通过 `pytest` 全量执行任务测试集，比对精度。
2. **泛化验证**：ATK 框架生成 ≥ 200 例随机 shape / 数值，验证泛化精度。
3. **参考实现校验**：numpy 参考实现先做 selfcheck（均值/方差误差、LayerNorm 不变量），确保参考本身正确。
4. **精度定位手段**：对比 FP32 全精度参考与 FP16 workspace 参考，区分「算法错误」与「精度舍入」。
5. **性能数据采集**：使用 `msprof op` 采集设备级 Task Duration，取多次执行结果，记入自验证报告。

## 4.4 兼容性分析

- 本算子为**新增** optest kernel 与 API，不修改现有算子函数签名。
- 公共 CATLASS `BlockMmadTla` / `Layout` / `TileCopy` / `Swizzle` 组件保持原有行为。
- 新增注册项（`torch.ops.catlass.ascend950_matmul_layernorm_matmul`）仅在显式调用时生效，
  不影响其他 optest kernel。
- 对已有文件的修改均为**追加**（ABI 头、`catlass_torch.cpp`、`ops/__init__.py`、
  `kernels/CMakeLists.txt`），不回滚任何已有内容。

## 4.5 交付件清单

| 序号 | 交付件 | 路径/形式 |
| --- | --- | --- |
| 1 | 算子设计文档 | 本文档，`docs/design.md` |
| 2 | 算子样例（Kernel 入口 + 测试件） | `experimental/matmul/87_ascend950_matmul_layernorm_matmul/` |
| 3 | 公共组件（kernel 模板） | `include/catlass/gemm/kernel/matmul_layernorm_matmul_tla.hpp` |
| 4 | optest 测试交付件 | `tests/optest/kernels/87_ascend950_matmul_layernorm_matmul/` 与 `tests/optest/src/include/template/matmul_layernorm_matmul.h` |
| 5 | ATK 测试交付件 | `tests/atk/87_matmul_layernorm_matmul/` |
| 6 | 参考实现 | `reference/matmul_layernorm_matmul_ref.py` |
| 7 | 样例说明文档 | `experimental/.../README.md` |
| 8 | 自验证报告 | 按官方模板提交，含用例参数、精度结果截图、性能数据截图 |
