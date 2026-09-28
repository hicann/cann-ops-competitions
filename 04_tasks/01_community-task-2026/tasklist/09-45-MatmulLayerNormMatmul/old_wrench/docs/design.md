# MatmulLayerNormMatmul 算子设计文档

# 需求背景

## 需求来源

本需求来源于 CANN 2026 年 9 月社区任务 `MatmulLayerNormMatmul`（任务编号 45）。任务要求在 Ascend 950 上使用 CATLASS 与 Ascend C 实现单 kernel 融合算子，在一次 kernel launch 内完成 `Matmul -> LayerNorm -> Matmul` 三段计算，并提交样例代码、测试交付件、自验证报告、README 与设计文档。

社区任务列表页（其中登记了本任务编号 45）：

<https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/docs/README.md>

目标仓库：

<https://gitcode.com/cann/catlass>

## 背景介绍

### MatmulLayerNormMatmul 算子实现优化

Transformer/MLP 结构中存在"矩阵乘 → 行级归一化 → 矩阵乘"的连续计算：

```python
c0 = torch.mm(a0, b0)
c0_norm = torch.nn.functional.layer_norm(c0, normalized_shape=(n0,), weight=gamma, bias=beta, eps=1e-6)
c1 = torch.mm(c0_norm, b1)
```

小算子拼接方案至少需要三次 kernel launch，且必须把 `C0` 与 `C0_norm` 写入 GM 后再读回。任务测试集中 `N0` 最大为 8192，中间矩阵的 GM 往返带来明显的带宽与 launch 开销。本方案拟把三段计算融合进单次 launch，并尽量减少中间张量的显存往返。

### 实现路径与参考组件

CATLASS 定位为算子样例库，本任务按融合样例组织，不采用传统自定义算子的 `op_host/op_kernel` 目录结构。任务书规定的落仓路径为 `experimental/matmul/matmul_layer_norm_matmul/`，与 `examples/` 下的编号样例并列。

新增目录结构（设计规划）：

```text
experimental/matmul/matmul_layer_norm_matmul/
  CMakeLists.txt
  README.md
  matmul_layer_norm_matmul.cpp                    # 样例入口
  test_matmul_layer_norm_matmul.py                # 接入 Optest 的测试件
include/catlass/epilogue/block/
  block_epilogue_layernorm_stats.hpp              # 第一阶段 epilogue（统计 + 落 workspace）
  block_epilogue_layernorm_apply.hpp              # 归一化分支（UB 归一化，出口可选 L1 或 GM）
include/catlass/epilogue/tile/
  tile_row_reduce_square.hpp                      # 行级 Σx / Σx² 归约原语
  tile_layernorm_affine.hpp                       # 归一化仿射原语
include/catlass/gemm/kernel/
  matmul_layer_norm_matmul.hpp                    # kernel 编排层
  matmul_layer_norm_matmul_impl.hpp               # 三阶段实现
tests/optest/kernels/matmul_layer_norm_matmul/    # JIT 入口与模板实现
tests/optest/src/include/template/
  matmul_layer_norm_matmul.h                      # torch 适配器
tests/optest/torch_catlass/ops/
  matmul_layer_norm_matmul.py                     # Python 封装
tests/optest/tests/
  test_matmul_layer_norm_matmul.py                # pytest
```

主要参考：

| 参考项 | 用途 |
| --- | --- |
| `examples/43_ascend950_basic_matmul` | Ascend 950 基础 matmul 的 tile shape 取值与 kernel 装配方式 |
| `examples/44_quant_matmul_full_loadA_tla` | Optest 接入链路（ABI / JIT / adapter / python / pytest） |
| `block_epilogue_fa_softmax_ascend950.hpp` | AIV 在 UB 算完后经 `CopyUb2L1` 回写 L1、供后续 Cube 消费的既有范式 |

# 需求分析

## 需求描述

使用 CATLASS 与 Ascend C 在 Ascend 950 上实现融合算子 `Matmul -> LayerNorm -> Matmul`：

$$
\begin{aligned}
\mathbf{C0} &= \mathbf{A0}\times\mathbf{B0} & &(M_0, N_0)\\
\mu[m] &= \frac{1}{N_0}\sum_{n=0}^{N_0-1}\mathbf{C0}[m,n]\\
\sigma^2[m] &= \frac{1}{N_0}\sum_{n=0}^{N_0-1}\bigl(\mathbf{C0}[m,n]-\mu[m]\bigr)^2\\
\mathbf{C0}_{\text{norm}}[m,n] &= \frac{\mathbf{C0}[m,n]-\mu[m]}{\sqrt{\sigma^2[m]+\epsilon}}\cdot\gamma[n]+\beta[n]\\
\mathbf{C1} &= \mathbf{C0}_{\text{norm}}\times\mathbf{B1} & &(M_0, M_1)
\end{aligned}
$$

其中 $\epsilon = 10^{-6}$。`C0` 与 `mean/variance` 均为算子内部中间量，不对外暴露。

参数表：

| 参数名 | 方向 | 描述 | 数据类型 | 数据格式 | 维度 | 布局 |
| --- | --- | --- | --- | --- | --- | --- |
| A0 | 输入 | 第一段 matmul 左矩阵 | FP16 | ND | (M₀, K₀) | RowMajor |
| B0 | 输入 | 第一段 matmul 右矩阵 | FP16 | ND | (K₀, N₀) | ColumnMajor |
| B1 | 输入 | 第二段 matmul 右矩阵 | FP16 | ND | (N₀, M₁) | ColumnMajor |
| gamma | 输入 | LayerNorm 缩放参数 | FP32 | ND | (N₀,) | — |
| beta | 输入 | LayerNorm 偏置参数 | FP32 | ND | (N₀,) | — |
| C1 | 输出 | 第二段 matmul 输出 | FP16 | ND | (M₀, M₁) | RowMajor |

## 需求拆解

1. 三段计算在**单次 kernel launch** 内完成，`C0` 不落 GM（或仅落一次并只读一次，见 §详细设计）。
2. `mean` / `variance` 由算子内部 workspace 管理，不对外暴露。
3. 精度满足生态算子开源精度标准。
4. 性能不低于 `torch.mm + F.layer_norm + torch.mm` 小算子拼接方案，整体达到 1.1×。
5. 通过 Optest 任务测试集（116 例）与 ATK 泛化测试（≥200 例）。

# 详细设计

## 算子分析

### 数学公式

见 §需求描述。

### 支持数据类型

输入 A0/B0/B1 为 FP16，gamma/beta 为 FP32，输出 C1 为 FP16；中间累加与统计全程 FP32。

### 支持形状

M₀ ∈ {128, 512, 1024, 2048}，K₀ ∈ {768, 2048, 4096, 8192}，N₀ ∈ {2048, 3072, 4096, 8192}，M₁ ∈ {768, 2048, 4096}。

## 算子实现

### 实现方案

#### 3.2.1 关键设计取舍：中间结果能否常驻片上

LayerNorm 的硬约束是**归一化需要 `C0` 的完整行**（归约长度 = N₀），因此任何 tile 方案都必须让"整行"同时可见。Ascend 950 的片上容量为：

| 缓冲 | 容量 |
| --- | --- |
| L1 | 512 KB |
| L0A / L0B | 64 KB / 64 KB |
| L0C | 256 KB |
| UB | 248 KB |

各方案的可行边界：

- **整行留 UB**：单行 FP32 = `N₀ × 4B`，N₀ = 8192 时为 32 KB，UB 最多容纳 7 行 → 不足以支撑 Cube 的最小分形高度。
- **整行留 L1**：单行 FP16 = `N₀ × 2B` = 16 KB；若 M 方向 tile 取 16 行，则 `16 × 8192 × 2B = 256 KB`，可放入 L1（512 KB）。**这是本方案"零中间显存往返"路线的可行边界**，代价是 M 方向 tile 只能取到 Cube 的最小分形高度（16），M 方向并行度受限。
- **整块留 L0C**：`BM × N₀ × 4B ≤ 256 KB` 要求 N₀ = 8192 时 BM ≤ 8，低于 Cube 最小分形高度，不可行。
- **落 GM workspace**：`C0` 以 FP16 落 workspace，代价是 `2·M₀N₀` 字节写入 + `2·M₀N₀` 字节读回。若不落 workspace 而选择"重算 `C0`"，需重读 `B0`（`2·K₀N₀` 字节，为回写代价的 2 倍）并重跑一遍 Cube，故重算不如回写。

**本方案设计两条路径**，通过编译期开关 `USE_L1_RESIDENT_A`（JIT 宏 `CATLASS_JIT_LN_STAGE3_L1_RESIDENT_A`）选择：

| 路径 | 中间张量显存往返 | 依赖 |
| --- | --- | --- |
| 融合（`true`） | 仅写 `C0` 一次、读回一次（归一化后直接进 L1） | 需通用 `BlockMmadTla` 接受 L1 常驻左矩阵（组件改造项，见 §3.2.3） |
| 退路（`false`，**默认**） | `C0` 落 workspace，归一化结果再落第二块 workspace | 无，走通用 `GM → L1 → L0A` 通路 |

退路路径相对基线（4 次往返：写 `C0` / 读 `C0` / 写 `C0_norm` / 读 `C0_norm`）的收益主要来自 **launch 次数由 3 次降为 1 次**，以及省掉 `C0_norm` 的一次写与一次读。

#### 3.2.2 host 侧设计

Tiling 策略：

| 层级 | 取值 | 依据 |
| --- | --- | --- |
| L1 Tile (M×N×K) | `256 × 256 × 128` | 与 `examples/43_ascend950_basic_matmul` 对齐；K 方向经 `TileShapeScalerTLA` 按元素字节宽度等比缩放 |
| L0 Tile (M×N×K) | `256 × 256 × 32` | L0C 容量：`256×256×4B = 256 KB`，用满 L0C |
| 调度粒度 (`ScheduleTileShape`) | `256 × 256 × 128` | 独立于 `BlockMmad::L1TileShape` 声明为 `GemmShape`，避免 TLA `tla::Shape` 无 `::M` 成员的问题 |
| 第一阶段 UB Tile | `64 × 256` | 见下方 UB 预算 |
| DispatchPolicy | `MmadPingpong<Arch::Ascend950, true, ...>` | `enableUnitFlag = true` 省一次 `M_FIX` 同步 |
| BlockScheduler | `GemmIdentityBlockSwizzle<3, 0/1>` | 按 m>n 选 swizzle 方向以提升 L2 命中 |

第一阶段 epilogue 的 UB 预算（FP32，除注明外）：

| 缓冲区 | 元素数 | 字节 | 说明 |
| --- | --- | --- | --- |
| C0 tile | 64×256 | 64 KB | 从 L0C fixpipe 过来 |
| 平方暂存 | 64×256 | 64 KB | Σx² 的中间量 |
| 行归约暂存 | 64 | 0.25 KB | 分块归约用 |
| FP16 输出暂存 | 64×256 | 32 KB | Cast 后经 DataCopy 回 workspace |
| **合计** | | **≈ 160.25 KB** | < 248 KB，安全 |

说明：若 M 方向 tile 取 128，则合计约 258.5 KB > 248 KB，会超预算，故取 64。

任务分核：第一阶段按 (M-tile × N-chunk) 二维划分，第三阶段按 (M-tile × M1-chunk) 划分，各核以 `blockNum` 为步长轮转调度器的 loop index；`blockNum` 由 host 侧按 `GetCoreNumAic()` 注入。

Workspace 布局（`GetWorkspaceSize` 内部申请，不进入算子参数表）：

| 偏移 | 内容 | 大小 |
| --- | --- | --- |
| 0 | `C0`（FP16） | `M₀ × RoundUp(N₀, 32B) × 2B` |
| `sizeC0` | Σx 部分和（FP32） | `mTileCount × nTileCount × 64 × 4B` |
| `sizeC0 + sizePartial` | Σx² 部分和（FP32） | 同上 |
| `...` | mean / rstd（FP32） | `2 × M₀ × 4B` |
| `...` | 归一化结果（FP16，仅退路模式） | 同 `sizeC0` |

`C0` 的行跨距按 32B 向上填充：UB → GM 的 `DataCopy` 长度必须 32B 对齐，若不填充，尾块的对齐写会越过本行尾部侵入下一行，而下一行可能归属另一个核，构成写竞争。

#### 3.2.3 kernel 侧设计

采用 Ascend 950 MIX kernel（`__mix__(1,2)`），AIC 负责两段 Cube 计算，AIV 负责全部向量计算。阶段划分：

```text
P1  第一段 matmul + 行统计 + 落 workspace         并行域 (M-tile × N-chunk)
      AIC: GM[A0]→L1→L0A, GM[B0]→L1→L0B, Cube → L0C(FP32)
      AIV: L0C --fixpipe--> UB(FP32)
             ├─ 逐行归约 Σx 与块内离差平方和 Σ(x−x̄_块)² → partials workspace（按 (mIdx,nIdx) 分槽）
             └─ Cast(FP16) → C0 workspace
    ── AscendC::SyncAll<false>() ──
P2  部分和归约 → mean / rstd                       并行域：按 M 行
    ── AscendC::SyncAll<false>() ──
P3  归一化 + 第二段 matmul                         并行域 (M-tile × M1-chunk)
      退路：AIV 归一化 → C0n workspace；AIC 执行标准 matmul（左矩阵来自 GM）
      融合：AIV 归一化 → L1；AIC 直接从 L1 消费左矩阵
      ── （退路模式在两者之间再插一次 SyncAll） ──
```

方差口径采用**两级无抵消设计**（设计决定）：P1 的第二趟先按本块行均值中心化再求 `Σ(x−x̄_块)²`，P2 以 `var = ( Σ_i M2_i + Σ_i n_i·(μ_i − μ)² ) / N₀` 合并，全程只有非负数相加，避免朴素口径 `Σx²/N₀ − μ²` 在 `C0` 带大直流偏置时的大数相减。两级需同时采用：若 P1 仍落未中心化的 `Σx²`，块内项 `Σx²_i − (Σx_i)²/n_i` 依然是两大数相减。代价仅为每块多一趟 `Adds + Mul`，无额外 GM 流量（中间流量仍是 `4·M₀N₀`，低于基线的 `8·M₀N₀`）。

**竞争条件控制**：全部 workspace 槽位做到"唯一写入者"——partials 按 `(mIdx, nIdx)` 分槽、`mean/rstd` 按行分槽、`C0` 与 `C0n` 按 tile 分槽，且 `C0n` 的读写在两次 `SyncAll` 之间。因此**不使用任何原子操作**，仅依赖阶段间的全核屏障保证可见性。

**融合路径的 L1 分区（设计约定）**：退路模式下两个 `BlockMmad` 无时间重叠，均从 L1 offset 0 起即可；融合路径需把 AIV 产出的常驻 A 区显式偏移到 Stage3 自身 L1B stages 之上：

| 阶段 | L1 布局（offset 起） | 偏移来源 |
| --- | --- | --- |
| P1 | `[L1A×2 = 128KB | L1B×2 = 256KB]` 共 384KB @ 0 | 构造参数 0 |
| P3b 退路 | 同上 384KB @ 0（时序复用） | 构造参数 0 |
| P3 融合 | `[Stage3 L1B×2 = 256KB @0 | A 常驻区 2×64KB @256K..384K]` | `l1BufAddrStart = L1B_TILE_SIZE*L1B_STAGES` |

融合路径还需 AIC 侧绕过 `BlockMmadTla` 的 `GM→L1` 装载、直接消费既有 L1A slot（组件改造项，保留为编译期开关下的可选路径）。

### 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Ascend 950 / 950PR / 950DT | √ |

### 算子约束限制

- `N₀` 需为 8 的倍数（FP32 参数按 32B 对齐搬运的要求；任务测试集取值均满足）。
- `K₀` 无对齐要求（K 尾块由通用 `BlockMmadTla` 处理）。
- 本方案 `A0` 仅支持 RowMajor（`transA0 = false`）。
- `eps` 在 kernel 侧为编译期常量 `1e-6`，不通过运行期参数传递；Python 与 C++ 适配器均对其做显式校验，避免"传了不生效"。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | 满足生态算子开源精度标准：`rtol = atol = 2⁻⁹`、`matched_ratio ≥ 0.99`、`max_abs_error ≤ max(0.1, 32·ULP@1.0)`；golden 取"昇腾小算子拼接"标杆（`torch.mm → F.layer_norm → torch.mm`，FP16 进 FP16 出） | [生态算子开源精度标准](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md) §2.3（允许"CPU 高精度实现"或"昇腾小算子拼接"二选一） |
| 性能标准 | 达到 `torch.mm + F.layer_norm + torch.mm` 小算子拼接的 1.1×（平均的标杆时延 / 测试时延 > 1.1，按 116 例计） | 任务书测试集提供的标杆数据（Ascend 950PR） |

精度口径以「昇腾小算子拼接」为标杆，与任务书一致；若后续评审要求按生态精度标准 §3 换用"CPU 高精度实现"作单一标杆，则本算子中间量同步提升到 FP32 即可对齐该口径。

## 兼容性分析

新算子，不涉及兼容性分析。

## 可维可测性

- **可测**：通过 `tests/optest` 接入 PyTorch，注册为 `torch.ops.catlass.matmul_layer_norm_matmul`，pytest 覆盖任务测试集代表性切片。参考实现刻意不使用 `F.layer_norm`，而是按公式展开，避免归约口径差异掩盖问题。
- **负向用例**：另设一条用例专门验证 `gamma/beta` 真的生效（若仿射分支被绕过则必然失败），用于拦截"能跑但结果不对"的实现缺陷。
- **可维**：kernel 拆为 `RunStage1` / `RunStatsReduce` / `RunStage3` 三个阶段方法，每个阶段内的向量处理下沉到独立的 tile 原语；参数多且强相关的调用点用具名结构（`TileContext` / `SourceParams` / `TileOrigin`）封装。
