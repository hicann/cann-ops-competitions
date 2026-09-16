# MatmulLayerNormMatmul 算子设计文档

# 需求背景（required）

## 需求来源

CANN 社区任务「MatmulLayerNormMatmul 算子开发」（任务编号 09-45），开源仓 [cann/catlass](https://gitcode.com/cann/catlass)，适配硬件 Ascend 950。

## 背景介绍

### 算子实现优化

Transformer 类模型中的 `Matmul → LayerNorm → Matmul` 组合（如 attention 输出投影后接归一化、FFN 中间层）在框架侧通常由三个独立算子拼接完成：
`torch.mm` → `F.layer_norm` → `torch.mm`。三算子独立下发意味着中间结果 C0、C0_norm 都要完整落回显存，再由下一个算子读回，
带来两次额外的 GM 往返（写 C0、读 C0 做 LayerNorm、写 C0_norm、读 C0_norm 做第二个 Matmul），
在 950 上这部分带宽开销无法被计算掩盖，成为端到端时延的主要来源之一。

### 现状分析

- catlass 已有的 Matmul 类样例（`examples/43_ascend950_basic_matmul` 等）只覆盖单一 Matmul，没有把逐行归一化融进 kernel 的先例；
- LayerNorm 需要"整行可见"（行和、行平方和），与 Matmul 的 tile 级流水天然冲突：Cube 侧按 `(M, N)` 切 tile，而 LayerNorm 的归约维度是整行 N0；
- catlass 的 `CrossCoreBarrier / CrossCoreSetFlag / CrossCoreWaitFlag` 提供 AIC↔AIV 的栅栏与握手能力，可以在同一次 kernel launch 内把三段计算串起来，并把中间结果只放在 device workspace 上。

因此本任务选择"融合实现"：一次 kernel launch 完成全部三段计算，C0 与 LayerNorm 的均值/方差只存在于算子内部 workspace / UB，对外只暴露 C1。

# 需求分析（required）

## 需求描述

实现融合算子 Matmul → LayerNorm → Matmul：

- 单次 kernel launch 完成三段计算；
- 中间结果（C0、均值、方差）由算子内部 workspace 管理，不对外暴露；
- 输入输出与任务书一致：A0/B0/B1 为 FP16，gamma/beta 为 FP32，C1 为 FP16；
- 精度满足[生态算子开源精度标准](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md)；
- 性能达到"小算子拼接"基线（`torch.mm + F.layer_norm + torch.mm`）的 1.1 倍以上（平均标杆时延 / 平均测试时延 > 1.1）。

## 需求拆解

| 编号 | 子需求 | 交付形态 |
| --- | --- | --- |
| R1 | 第一段 Matmul（Cube） | `include/catlass/gemm/kernel/matmul_layer_norm_matmul_tla.hpp` 中的 stage 1 |
| R2 | 行级 LayerNorm（Vector，fp32 计算） | 同文件的 stage 2（`LayerNormWorkspace`） |
| R3 | 第二段 Matmul（Cube） | 同文件的 stage 3 |
| R4 | 段间同步（AIC↔AIV 握手 + 全栅栏） | `Arch::CrossCoreBarrier` / `CrossCoreSetFlag` / `CrossCoreWaitFlag` |
| R5 | workspace 规划与对齐 | `GetWorkspaceSize` / `ToUnderlyingArguments` |
| R6 | 外部样例与自测接入 | `experimental/matmul/matmul_layer_norm_matmul/`（样例 + README + optest 测试件） |
| R7 | optest 框架接入与算子注册 | `tests/optest/**`（kernel 组装、JIT 参数传递、torch_catlass 注册） |
| R8 | ATK 泛化用例 | 基于已接入 optest 的测试件生成 ≥200 例 |

# 详细设计（required）

## 算子分析

### 数学公式

阶段一（第一个 Matmul）：

$$ \mathbf{C0} = \mathbf{A0} \times \mathbf{B0}, \quad \mathbf{C0} \in \mathbb{R}^{M_0 \times N_0} $$

阶段二（LayerNorm，对 $\mathbf{C0}$ 的每一行独立计算，$\epsilon = 10^{-6}$）：

$$ \mu[m] = \frac{1}{N_0}\sum_{n=0}^{N_0-1} \mathbf{C0}[m,n] $$

$$ \sigma^2[m] = \frac{1}{N_0}\sum_{n=0}^{N_0-1}\left(\mathbf{C0}[m,n] - \mu[m]\right)^2 $$

$$ \mathbf{C0}_{\text{norm}}[m,n] = \frac{\mathbf{C0}[m,n] - \mu[m]}{\sqrt{\sigma^2[m] + \epsilon}} \cdot \gamma[n] + \beta[n] $$

阶段三（第二个 Matmul）：

$$ \mathbf{C1} = \mathbf{C0}_{\text{norm}} \times \mathbf{B1}, \quad \mathbf{C1} \in \mathbb{R}^{M_0 \times M_1} $$

### 支持数据类型

| 张量 | 数据类型 | 说明 |
| --- | --- | --- |
| A0 | FP16 | 第一个 Matmul 左矩阵 |
| B0 | FP16 | 第一个 Matmul 右矩阵 |
| B1 | FP16 | 第二个 Matmul 右矩阵 |
| gamma / beta | FP32 | LayerNorm 逐通道缩放 / 偏置 |
| C0（内部） | FP16 | 中间结果，仅存在于 workspace |
| 统计量（内部） | FP32 | 行和、行平方和、均值、方差，只落在 UB |
| C1 | FP16 | 唯一对外输出 |

### 支持形状

| 张量 | 形状 | 布局 |
| --- | --- | --- |
| A0 | $(M_0, K_0)$ | RowMajor |
| B0 | $(K_0, N_0)$ | ColumnMajor |
| B1 | $(N_0, M_1)$ | ColumnMajor |
| gamma / beta | $(N_0)$ | ND |
| C1 | $(M_0, M_1)$ | RowMajor |

- 支持任意合法 `M0 / K0 / N0 / M1`，包括非对齐（如 `N0 = 260 / 5 / 33 / 129` 等不能整除 16/32 的取值）；
- 非对齐由 tile 尾块与 mask 处理，不要求调用方补齐。

## 算子实现

### 实现方案

#### host 侧设计

- **Tiling**：沿用 catlass 的 `BlockScheduler`（`GemmIdentityBlockSwizzle<3, k>`）做 block 级切分；两段 Matmul 各用一套 L1/L0 TileShape，样例与 JIT 模板中给出默认值：

| 阶段 | L1 TileShape (M, N, K) | L0 TileShape (M, N, K) |
| --- | --- | --- |
| stage 1（C0 = A0×B0） | (128, 256, 256) | (128, 256, 64) |
| stage 3（C1 = C0_norm×B1） | (128, 256, 256) | (128, 256, 64) |

- **workspace**：`GetWorkspaceSize` 返回 `M0 × N0 × sizeof(fp16) + 对齐余量(512B) + 余量(512B)`；
  `ToUnderlyingArguments` 把起始地址按 512B 对齐后构造 C0 的 RowMajor 布局，**调用方只申请、不感知布局**；
- **参数**：`Arguments` 只包含 A0/B0/B1/gamma/beta/C1 的指针、布局与 problem shape，加上 `epsilon`；
  `Params` 内部再补一个 workspace 上的 C0 描述，宿主侧看不到；
- **ACL 流**：样例按 catlass 统一方式（`aclrtStream` + `<<<>>>` 风格 launch）在 host 侧完成 workspace 申请、kernel 下发与同步。

#### kernel 侧设计

一次 kernel launch 内，AIC（Cube）与 AIV（Vector）分工如下：

1. **stage 1（Cube）**：`C0 = A0 × B0`，经 fixpipe 直接写入 workspace（RowMajor FP16）。
2. **同步 1**：所有 Cube 核先做一次同类型核的全栅栏 `Arch::CrossCoreBarrier<0x0, PIPE_MTE3>`，确保整个 C0 已落 workspace；
   再以 mode 4（AIC 与其配对的两个 AIV 子核）唤醒配对子核：
   `Arch::CrossCoreSetFlag<4, PIPE_FIX>(flagId)` 与 `<4, PIPE_FIX>(flagId + 16)`，两个子核各自等待 `flagId`，由硬件区分具体子核。
3. **stage 2（Vector）**：每个 Vector 核按行切分 workspace 上的 C0，对每行分三趟完成 LayerNorm（**全 fp32**）：
   - 第 1 趟：行和 → 均值 $\mu$；
   - 第 2 趟：平方偏差和 → 方差 $\sigma^2$；
   - 第 3 趟：$\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}\gamma + \beta$ 归一化后写回 workspace（FP16）。
   均值/方差只落在 UB 的临时 scratch 上，不写回 GM。
4. **同步 2**：两个 Vector 子核各自完成后，所有 Vector 核再做一次栅栏（stage 3 可能读到由其它 Vector 核归一化的行），
   随后每个子核以 mode 4 置位 `flagId`。
5. **stage 3（Cube）**：Cube 核按 mode 4 语义同时等待 `flagId` 与 `flagId + 16`，随后执行 `C1 = C0_norm × B1`，直接写对外输出。

要点：
- 第二段 Matmul 的 A 操作数就是 workspace 上的归一化结果，因此中间结果**不需要额外 GM 读写**，也不暴露给调用者；
- 归一化的向量计算在 950 上使用寄存器级向量 API（`AscendC::MicroAPI`/`AscendC::Reg` 的 `RegTensor / MaskReg / Cast / CastTrait / DataCopy / StoreAlign`），
  由 `#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510` 守卫，非 950 架构下走标量回退路径；
- **非对齐形状**：Vector 侧按 mask（`UpdateMask`）处理行尾，Cube 侧由 L0 tile 的尾块处理，不对齐不额外拆 kernel。

## 支持硬件

| 项 | 要求 |
| --- | --- |
| 硬件 | Ascend 950PR（本任务在单卡 1 NPU 环境验证） |
| 编译架构 | `-DCATLASS_ARCH=3510` |
| CANN | catlass 当前发行版要求的最低 CANN 版本（本任务环境：CANN 9.1.0） |

## 算子约束限制

- 无（任务书「算子约束限制」为无）；实现上额外说明：
  - `M0 / K0 / N0 / M1` 需为合法正数，`N0` 决定 LayerNorm 的行长；
  - workspace 由调用方按 `GetWorkspaceSize` 申请，容量与 `M0 × N0` 成正比（C0 是必须的中间态）；
  - 算子只在 AIC/AIV 协同时序下正确，调用方不要在同一 stream 上并发改写同一 workspace。

# 可维可测分析

## 精度标准/性能标准

- **精度**：满足[生态算子开源精度标准](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md)；
  参考实现为 TorchNPU 上的 `torch.mm + F.layer_norm + torch.mm` 三算子拼接；
  为降低误差，LayerNorm 的均值/方差/归一化全部在 **fp32** 下计算，写回时才转回 FP16（转换使用 `CAST_RINT`，与参考实现舍入一致）。
- **性能**：达标线为「平均基线时延 / 平均测试时延 > 1.1」，
  基线 = `torch.mm + F.layer_norm + torch.mm` 拼接（任务测试集已给出 950PR 上的基线时延），
  采集工具为 `msprof op`；性能测量需在 NPU 空闲状态下进行。

## 测试方案

| 层次 | 内容 | 通过判据 |
| --- | --- | --- |
| 样例冒烟 | `examples/matmul_layer_norm_matmul`，含对齐与非对齐形状 | 精度比对 `Compare success.` |
| optest 精度 | 任务测试集 116 个形状 + 若干非对齐形状，`pytest test_matmul_layer_norm_matmul.py` | 全部通过 |
| ATK 泛化 | 基于 optest 测试件生成 ≥200 例 | 全部通过 |
| 性能 | 任务测试集逐 shape 采集，与基线逐项对比 | 平均 基线/测试 > 1.1 |

## 性能优化策略

1. **消除两次中间结果的 GM 往返**：C0/C0_norm 只存在于 workspace，且 stage 3 直接消费 workspace，不额外搬运；
2. **Cube/Vector 任务级并行**：stage 2 的 LayerNorm 由 AIV 承担，Cube 侧在等待期间不占用 Cube 计算单元；
3. **TileShape 可调**：L1/L0 TileShape 通过样例与 JIT 模板参数暴露，便于按 shape 分布调优；
4. **Pingpong 流水**：两段 Matmul 均使用 `MmadPingpong` 策略，隐藏 MTE2/MTE1 与 MMAD 的往返；
5. **fp32 统计 + 向量寄存器计算**：LayerNorm 的三趟计算全部在 UB/寄存器完成，避免反复落 GM。

## 风险与规避措施

| 风险 | 说明 | 规避 |
| --- | --- | --- |
| 跨核同步时序 | AIC/AIV 的握手若少一次栅栏，stage 3 可能读到未归一化的行 | 采用 mode 4 + 双重 flagId（`flagId` / `flagId+16`）显式配对；用非对齐大 shape 压测 |
| 行长为 0 / 极小 | `N0` 很小时 mask 与规约退化 | 单测覆盖 `N0 = 5 / 33 / 129` 等小与非对齐值 |
| 非对齐尾块 | 尾块处理错误会导致精度超差 | optest 测试集含非对齐用例，逐例比对 |
| 向量 API 架构依赖 | 寄存器级 API 只在 950（`__NPU_ARCH__ == 3510`）可用 | 用 `#if` 守卫并提供标量回退，保证其它架构可编译 |
| 性能不达标 | 小 shape 收益不足 | TileShape 可配 + 记录调参过程说明 |

## 兼容性分析

- 仅新增 `experimental/matmul/matmul_layer_norm_matmul/` 与 `include/catlass/gemm/kernel/matmul_layer_norm_matmul_tla.hpp`，
  对 optest 的接入是对既有框架的**增量注册**（kernel 列表、JIT 签名、torch_catlass 注册），不改动既有算子行为；
- 不引入新的第三方依赖；不改变 catlass 的构建流程，样例遵循 `experimental/` 的既有约定（需拷到 `examples/` 并注册后方可编译）；
- 非 950 架构下包含该头文件仍可编译（向量路径被守卫），不破坏其它平台的构建。
