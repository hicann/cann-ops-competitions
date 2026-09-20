# MatmulLayerNormMatmul 算子设计文档（Ascend 950 / CATLASS）

> 融合算子 `Matmul → 行级 LayerNorm → Matmul`，**单次 MIX kernel launch** 完成三段
> 仓库：[cann/catlass](https://gitcode.com/cann/catlass)，代码落 `examples/106_ascend950_matmul_layer_norm_matmul/`、`include/catlass/gemm/kernel/matmul_layer_norm_matmul.hpp` 与 `tests/optest/`（对齐 [PR #678](https://gitcode.com/cann/catlass/pull/678) 交付结构）
> 硬件：Ascend 950（性能采集平台 Ascend 950PR，SoC `dav_3510`）；语言：Ascend C / CATLASS TLA
> 性能对标 `torch.mm + F.layer_norm + torch.mm` 小算子拼接，判定式「平均标杆时延 / 测试时延 > 1.1」
> 团队 `zhangfeng1133`。本文为 **r3 定稿件**：架构按最终实现定稿（GM-staging 中转），补充关键工程决策依据、同步协议表与 camodel 仿真约束附录；修订点见各节标注与文末「修订记录」
> 投稿版：9月社区任务-MatmulLayerNormMatmul算子开发（tasklist `09-MatmulLayerNormMatmul-950`）；章节结构对齐官方 `design_template.md`（需求背景 / 需求分析 / 详细设计[算子分析+算子实现(host侧/kernel侧)] / 支持硬件 / 算子约束限制 / 可维可测分析[精度性能标准+兼容性分析]），模板要求章节一节不缺，另有测试设计、策略映射、附录 A 等增补章节

# 需求背景（required）

## 需求来源

| 项 | 内容 |
| --- | --- |
| 任务 / 标签 / 硬件 | 9月社区任务-MatmulLayerNormMatmul算子开发 / 算子开发 / Ascend 950 |
| 开源仓 / CANN / 语言 | https://gitcode.com/cann/catlass / 算子开源仓指定版本 / Ascend C |
| 精度标准 | [生态算子开源精度标准](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md) |
| 合入方式 | catlass 仓提交 PR（[参考 PR #678](https://gitcode.com/cann/catlass/pull/678)）；设计文档 PR 至 cann-competitions 并通过评审 |
| 验收交付件 | ① 自测用例、测试结果报告、测试步骤指导；② 算子代码私仓邀请链接、代码仓路径、分支、算子目录 |

## 背景介绍

### MatmulLayerNormMatmul算子实现优化

Transformer / MLP 中普遍存在 `mm → 行级归一化 → mm` 链：

```python
c0      = torch.mm(a0, b0)                                 # (M0,K0)x(K0,N0) -> (M0,N0)
c0_norm = torch.nn.functional.layer_norm(c0, (n0,), weight=gamma, bias=beta, eps=1e-6)
c1      = torch.mm(c0_norm, b1)                            # (M0,N0)x(N0,M1) -> (M0,M1)
```

小算子拼接至少 3 次 launch，中间矩阵 `C0`/`C0_norm` 须写回 GM 再读，launch 与 flush 开销在小 shape 上占主导。本设计融合进**单次 MIX kernel launch**（1 AIC : 2 AIV）：AIC 执行两个 Matmul，AIV0 执行行级 LayerNorm，中间结果经**算子内部 GM workspace（W0/W1）中转**并与计算流水重叠；`Mean`/`Variance`（含 invstd）全程驻留 UB，不落 GM、不对外暴露；唯一对外输出为 `C1`。

### MatmulLayerNormMatmul算子功能分析

本任务为**新增融合样例**，不是历史 TBE 算子的语义迁移：catlass 仓无与该三段链路等价的既有实现（grep 确认无 LayerNorm/RMSNorm 既有实现），外部接口、dtype 与 eps 语义均由任务书定义，不存在与旧实现的性能/精度对比基线，唯一基线是小算子拼接方案。

参数表（逐列对齐任务书）：

| 参数名 | 输入/输出/属性 | 描述 | 使用说明 | 数据类型 | 数据格式 | 维度(shape) |
| --- | --- | --- | --- | --- | --- | --- |
| A0 | 输入 | 第一个 Matmul 左矩阵，形状 (M0, K0) | RowMajor 布局 | FP16 | ND | 2 |
| B0 | 输入 | 第一个 Matmul 右矩阵，形状 (K0, N0) | 见「布局契约符合性」 | FP16 | ND | 2 |
| B1 | 输入 | 第二个 Matmul 右矩阵，形状 (N0, M1) | 见「布局契约符合性」 | FP16 | ND | 2 |
| gamma | 输入 | LayerNorm 缩放参数，长度 N0 | 可学习参数 | FP32 | ND | 1 |
| beta | 输入 | LayerNorm 偏置参数，长度 N0 | 可学习参数 | FP32 | ND | 1 |
| C1 | 输出 | 第二个 Matmul 输出矩阵，形状 (M0, M1) | 唯一对外输出 | FP16 | ND | 2 |

其余契约：**单次 launch** 完成三段；中间结果由算子内部 workspace 管理且不对外暴露；`eps` 固定 `1e-6` 无属性；无广播，gamma/beta 仅沿 N0 作用；允许用多种 Matmul 方案达成更好性能。

**布局契约符合性（r3 定稿）**：任务书参数表标注 B0/B1 为 ColumnMajor 布局。本实现**定稿采用 RowMajor**：`b0[k*N0+n]`、`b1[n*M1+p]`，即 torch 侧传连续 (K0,N0)/(N0,M1) 张量，`C0=A0×B0`、`C1=LN(C0)×B1` 数学语义不变。定稿依据：① 官方 `43_ascend950_basic_matmul` 样例的 B 路径即 RowMajor（GM RowMajor → L1 zN → L1→L0 随路 nZ 转置）；② CATLASS `TileCopyTla` 无 GM ColumnMajor→L1 拷贝特化；③ ColumnMajor 组合（GM ColumnMajor + L1 nZ）经 camodel 端到端验证结果不正确。该差异已在样例 README「布局约定说明」中显式声明，供评审知悉，见「算子约束限制」C8。

# 需求分析（required）

## 需求描述

用 Ascend C / CATLASS 在 Ascend 950 上实现 `MatmulLayerNormMatmul`：单次 launch 完成 `Matmul → LayerNorm(行级, eps=1e-6) → Matmul`；中间结果不对外暴露；功能与精度通过任务测试集（116 条）及生态算子开源精度标准；平均标杆时延 / 测试时延 > 1.1；按 catlass 规范交付样例、`include/catlass` 组件、optest 接入件、README 与自验证据。

## 需求拆解

| 编号 | 子需求 | 设计响应 | 验证方式 |
| --- | --- | --- | --- |
| R1 | dtype 契约 | A0/B0/B1/C1=FP16，gamma/beta=FP32 | Host 校验 + 负向用例 |
| R2 | 布局契约 | B0/B1 按 RowMajor stride 组装 TLA tile（见布局契约符合性声明） | stride 专项用例、逐 case 比对 |
| R3 | 单次 launch | 单个 MIX kernel（1:2），cross-core flag 配对 | `msprof op` kernel 列表、代码审查 |
| R4 | 中间结果不对外暴露 | `C0`/`C0_norm` 经算子内部 GM workspace W0/W1 中转（launcher 内部申请/释放），统计量驻 UB | workspace 由 launcher 私有分配、代码审查 |
| R5 | 行级 LayerNorm | 每行块完整持有 N0，FP32 两遍中心化统计（mean→var） | 与 golden 比对、常量行用例 |
| R6 | 测试集覆盖 | 116 条全量参数化 | pytest 全量执行日志 |
| R7 | 精度达标 | Matmul/LN 按 FP16 语义，统计 FP32 | 精度标准判定 + NaN/Inf 位置检查 |
| R8 | 性能达标 | 融合消除 2 次 launch、流水重叠、AIC/AIV 双流 | `msprof op` 逐 case 倍率 |
| R9 | 泛化与可合入 | ATK ≥200 例；目录/README/PR 内容合规 | ATK 执行结果、clean build、PR 自检 |

# 详细设计（required）

## 算子分析

### 数学公式

阶段一（`C0` 形状 (M0,N0)）：`C0[m,n] = sum_{k=0}^{K0-1} A0[m,k] * B0[k,n]`

阶段二（LayerNorm，对 `C0` 每行独立，`epsilon = 10^-6`）：

$$
\begin{aligned}
\mu[m] &= \frac{1}{N_0}\sum_{n=0}^{N_0-1}\mathbf{C0}[m,n] \\
\sigma^2[m] &= \frac{1}{N_0}\sum_{n=0}^{N_0-1}\big(\mathbf{C0}[m,n]-\mu[m]\big)^2 \\
\mathbf{C0}_{\text{norm}}[m,n] &= \frac{\mathbf{C0}[m,n]-\mu[m]}{\sqrt{\sigma^2[m]+\epsilon}}\cdot\gamma[n]+\beta[n]
\end{aligned}
$$

阶段三（`C1` 形状 (M0,M1)）：`C1[m,p] = sum_{n=0}^{N0-1} C0_norm[m,n] * B1[n,p]`

kernel 内实际形式：`invstd[m] = rsqrt(max(var[m],0.0f) + 1e-6f)`；`C0_norm = (C0-mean[m])*invstd[m]*gamma[n]+beta[n]`。方差取总体方差 `N0`（不做 Bessel 校正）与 `F.layer_norm(normalized_shape=(N0,))` 口径一致；`rsqrt` 前对 `var` 取 0 下界，避免负数开方产生 NaN。方差必须**两遍**完成（先 mean 后 var）：单遍 `E[x²]-E[x]²` 在均值大方差小时存在抵消（r2 已论证，r3 实现保持两遍）。

### 支持数据类型

| 阶段 | 输入 dtype | 输出/中间 dtype | 计算 dtype |
| --- | --- | --- | --- |
| Matmul0 | A0/B0 FP16 | `C0`(W0) FP16 | Cube FP32 累加，FixPipe 随路转 FP16 |
| LN 统计 | W0 FP16 | mean/var/invstd FP32（UB） | FP32 |
| LN 归一化 | W0 FP16 + gamma/beta FP32 | `C0_norm`(W1) FP16 | FP32 计算后转 FP16 |
| Matmul1 | W1 FP16、B1 FP16 | `C1` FP16 | Cube FP32 累加，FixPipe 随路转 FP16 |

`C0`/`C0_norm` 取 FP16：与 PyTorch 链路 FP16 `torch.mm` 及 FP16 LayerNorm 输出的 dtype 传播一致。

### 支持形状

任务测试集共 116 条：M0 ∈ {128,512,1024,2048}，K0 ∈ {768,2048,4096,8192}，N0 ∈ {2048,3072,4096,8192}，M1 ∈ {768,2048,4096}。

测试集内 K0/N0/M1 均可被 128/256 整除，但实现按 `CeilDiv` + actual shape 处理尾块（KL0=32 尾片、BN0/BN1 尾 tile、BM 尾行块均支持，非对齐 case 已在 camodel 通过），**不把对齐特征写成接口限制**；最小合法单位 M0/K0/N0/M1 ≥ 1。

## 算子实现

### 实现方案（r3 架构定稿：GM-staging 中转）

主方案：Ascend 950 MIX kernel（1 AIC : 2 AIV，`GetSubBlockIdx()==0` 的 AIV0 承担 LN），行块为任务粒度，`row_blocks = CeilDiv(M0,BM)`，grid-stride 分发（`rb += blockNum/2`）。**中间结果经算子内部 GM workspace 两级中转**：W0（Matmul0 原始结果，行填充至 `n0Aligned=RoundUp(N0,16)`）与 W1（归一化结果，同形状），由 launcher 内部申请/释放，不对外暴露。

```text
                                    ┌──────────── AIV0 (subBlock 0) ────────────┐
GM A0 ─MTE2─► L1 A0(zN) ─┐          │  pass0: W0 chunk ─MTE2─► UB ─V─► rowSum   │
GM B0 ─MTE2─► L1 B0(zN) ─┼─MTE1─►   │          └────────────► mean              │
        (KL0=32 切片)    │  L0A/L0B │  passA: W0 chunk ─► UB ─V─► (x-µ)² ─► var │
AIC:  Mmad0 (FP32 L0C) ──┘          │          └────────────► invstd            │
  └─FIX─► FixPipe W0 (nz2nd,        │  passB: W0 chunk ─► UB ─V─► normalize     │
           FP32→FP16 随路) ─MTE3────┼─► W1 (GM) ─┬──────────────────────────────┘
                                     │            │
GM B1 ─MTE2─► L1 B1(zN) ─┐           │   AIC ─MTE2─┴─► L1 cache (zN, BM×n0Aligned)
        (KL0=32 切片)    ├─MTE1─► L0A/L0B
AIC:  Mmad1 (FP32 L0C) ──┘  左矩阵 = L1 cache 切片（局部坐标）
  └─FIX─► FixPipe C1 (nz2nd) ──► GM C1（唯一对外输出）
```

三遍 LN 的分块握手：AIV0 按 N0-tile（=chunk，`BN0` 宽）逐块 `StageChunkFromGm`（GM→UB，MTE2，逐行紧凑拷贝）→ Vector 段计算；pass0/passA/passB 共用同一 UB chunk 区，块间天然串行，无需额外 stage flag。

**为什么 GM-staging 而非 r2 的「L1 常驻 + UB 中转」**（r3 关键修订）：

1. **仿真可行性**：r2 的 Phase1 读回依赖 950 专属 `DataCopyL1ToUB`（AIC 发起、L1→AIV UB）。开发期 8 轮 camodel（dav_3510）实验证实该指令在本仿真快照为**静默 no-op**（附录 A），L1 常驻方案无法端到端验证；GM-staging 全链（Mmad0→W0→AIV 直读→W1→AIC 回填 L1→Mmad1→C1）每一段均已单独实证（`experimental/matmul/minimal_mlnm/` 诊断样例，C0 直转储与 golden 全量一致 maxErr=0.0039）。
2. **真机等价优势**：GM-staging 消除了 r2 方案中 AIC→AIV 的逐 tile UB stage 握手（FLAG_TILE_READY/STAGE_FREE×UB_STAGES）与 L1 cache 的跨行块 WAR 协议（CACHE_FREE/CACHE_ACK）——每个行块的 W0/W1 区间互斥，无核间数据竞争；跨核同步收敛为每行块 2 个 flag。AIC 的 W0 fixpipe 与 AIV0 的 W0 读取通过「行块内先产后耗」天然流水，对真机同样成立。
3. **代价（如实声明）**：相比 L1 常驻，W0/W1 各引入写+读 2 趟 GM 流量（上界 shape 约 4×32 MiB）。融合收益来源重新表述为：**消除 2 次 launch、统计量不出 UB、四趟中转流量与 Cube/LN 计算流水重叠、行填充对齐保证 32B 整块搬运**。真机迭代方向：`DataCopyL1ToUB` 真机可用后，W0 读取可切回 L1 直通（`CopyL0CToL1Tla` 已验证可写 L1 zN），恢复 r2 的零中间 GM 流量目标。

| 备选方案 | 结论 | 原因 |
| --- | --- | --- |
| L1 常驻 + `DataCopyL1ToUB` 读回（r2 主方案） | 真机迭代方向 | 本仿真快照 `DataCopyL1ToUB` no-op 无法验证；且需逐 tile 双向 stage flag + cache WAR 协议 |
| 不缓存 `C0`，先统计后重算 Matmul0 | 不采用 | 至少重复一次 Matmul0，分块后重复更多 |
| 一遍 `sum`/`sumsq` 求方差 | 不采用 | 抵消风险（见数学公式节） |
| 代数变形推迟归一化 | 不采用 | γ 沿归约轴缩放 B1 无法硬件随路；AIV 需额外整遍读 B1（流量翻倍） |
| FixPipe 直写 L1（`CopyL0CToL1Tla`）+ AIC gather | 部分验证 | 写向已验证；读向依赖 `DataCopyL1ToUB`（本快照 no-op），列为真机迭代路径 |

#### 3.2.1 Host 侧设计

**1. 样例组织**（实际交付，对齐 PR #678 文件清单）：

```text
catlass/
├── examples/106_ascend950_matmul_layer_norm_matmul/
│       ├── matmul_layer_norm_matmul.cpp        # host 入口 + golden 比对（Compare success 约定）
│       ├── CMakeLists.txt                      # catlass_example_add_executable(106_... mix ...)
│       ├── README.md                           # 算子说明/用法/约束
│       └── matmul_layer_norm_matmul.md         # 本文
├── include/catlass/gemm/kernel/matmul_layer_norm_matmul.hpp
└── tests/optest/
    ├── include/catlass_kernel_jit.h            # MatmulLayerNormMatmulParams + ABI 声明
    ├── kernels/106_ascend950_matmul_layer_norm_matmul/
    │       ├── ascend950_matmul_layer_norm_matmul.cpp       # JIT entry（JitKernelType::MIX）
    │       ├── ascend950_matmul_layer_norm_matmul_impl.cpp  # 模板 impl
    │       └── CMakeLists.txt
    ├── kernels/CMakeLists.txt                  # add_subdirectory 注册
    ├── src/include/template/matmul_layer_norm_matmul.h      # 独立 adapter
    ├── src/catlass_torch.cpp                   # REGISTER_TORCH_FUNC
    ├── torch_catlass/{__init__.py, ops/__init__.py, ops/ascend950_matmul_layer_norm_matmul.py}
    └── tests/test_106_ascend950_matmul_layer_norm_matmul.py # pytest（含非对齐 shape）
```

**2. 运行时参数**（kernel `Arguments`，与实现一致）：

```cpp
struct Arguments {
    GemmCoord problemShape;        // (M0, N0, K0)
    uint32_t m1{1};
    GM_ADDR a0GmAddr; GM_ADDR b0GmAddr; GM_ADDR b1GmAddr;
    GM_ADDR gammaGmAddr; GM_ADDR betaGmAddr;   // FP32
    GM_ADDR c1GmAddr;
    GM_ADDR c0StagingGmAddr{nullptr};  // W0: Matmul0 中转，行填充 FP16（launcher 内部申请）
    GM_ADDR c0NormGmAddr{nullptr};     // W1: 归一化中转，同形状
    GM_ADDR debugC0NormGmAddr{nullptr};   // 仅仿真诊断用，可空
    GM_ADDR debugC0DirectGmAddr{nullptr}; // 仅仿真诊断用，可空
    uint32_t debugFill{0};                // 仅仿真诊断用（device 侧填充）
};
```

`eps` 为编译期常量 `1e-6f`，不进入参数结构；TileShape / 流水深度由 host 绑定的静态 kernel 类型承载。W0/W1 尺寸均为 `M0 × RoundUp(N0,16) × 2B`，由 launcher（样例/optest adapter）申请、传入、用后释放——这是「中间结果由算子内部 workspace 管理且不对外暴露」（R4）的落位：workspace 生命周期完全在算子调用链内部，不作为公开契约。

**3. 参数校验与错误处理**（分配输出与下发 kernel 之前，任一失败即返回、不触碰 NPU）：

| 序号 | 校验项 | 判定条件 |
| --- | --- | --- |
| V1 | 指针与设备 | 五个输入及输出地址非空，且同一 NPU device |
| V2 | dtype | A0/B0/B1=FP16，gamma/beta=FP32 |
| V3 | rank 与取正 | A0/B0/B1 二维、gamma/beta 一维；四维均 >0 且可安全转 `uint32_t` |
| V4 | 归约维 | `A0.shape[1] == B0.shape[0]` (=K0) |
| V5 | 融合维 | `B0.shape[1] == B1.shape[0] == gamma.numel() == beta.numel()` (=N0) |
| V6 | stride | A0 `(K0,1)`；B0/B1 连续（RowMajor 自洽形态，见布局契约符合性）；gamma/beta 连续 |
| V7 | 输出 | `C1` 固定分配 `(M0,M1)` FP16 RowMajor |

地址取 `tensor.data_ptr()`（逻辑首元素）；校验失败以 `TORCH_CHECK` 报错并不启动 kernel。

**4. TileShape 静态档位**（106 样例实际分派；BN0=BN1=256、BK0=BK1=128、UB_STAGES=2、L1B_STAGES=2 固定）：

| 档位 | 适用 N0 | BM | 行块 cache（zN FP16） | 目标 |
| --- | --- | ---: | ---: | --- |
| S0 | ≤2048 | 64 | ≤256 KiB | 大行块、Cube 效率 |
| S1 | 3072~4096 | 32 | ≤256 KiB | 平衡 L1 与并行度 |
| S2 | 8192 | 16 | 256 KiB | 大 N0 专用 |

BM 随 N0 收缩的原因：行块 cache `BM×n0Aligned×2B` 必须与 B tile 槽位共存于 L1（预算式见下）。尾块/非对齐 shape 走 `CeilDiv` + actual shape，不设单独档位。

**预算算式**（真实常量 `include/catlass/arch/arch.hpp`：`L1_SIZE=512KiB`、`UB_SIZE=248KiB`、`L0A/L0B=64KiB`、`L0C=256KiB`）：

```text
CACHE    = BM * RoundUp(N0,16) * 2                      # 行块 zN cache（L1 offset 0）
L1_PHASE0 = CACHE_MAX(256K) + L1B_STAGES*BK0*BN0*2      # B0 槽位 2×64KiB
            + BM*BK0*2                                  # A0 tile（offset 在 B0 之后）
L1_PHASE1 = CACHE_MAX(256K) + L1B_STAGES*BK1*BN1*2      # B1 复用 B0 槽位区间
L0A/L0B  = 单缓冲（各 64KiB 内）；L0C = max(BM*BN0, BM*BN1)*4 ≤ 64KiB
UB       = UB_STAGES*BM*RoundUp(BN0,16)*2 + BM*LN_CHUNK_COLS*(2+4)
           + BM*BN0*4 + (BM*5 + 2*LN_CHUNK_COLS)*4      # LN_CHUNK_COLS 由 64KiB fp32 上限推出
```

运行时 `CanImplement` 按 actual shape 复核 phase0/phase1 ≤ `L1_SIZE`；静态 `static_assert` 锁 UB 总量 ≤ `UB_SIZE`。

**5. 分核与任务 ownership**：`row_blocks = CeilDiv(M0,BM)`；mix(1,2) 下 `coreNum = blockNum/2`，行块 grid-stride 分发；AIV1（subBlockIdx==1）空闲。每个行块任务内部完成全部 N0 chunk 与全部 M1 列 tile（`col_blocks` 复制策略列为迭代项，见策略映射①）。各任务 `C1`、W0、W1 的行区间互斥，无核间写冲突，无需 atomic。

#### 3.2.2 Kernel 侧设计

类型契约：`ArchTag = Arch::Ascend950`；`ElementIn = half`（A0/B0/B1/C0/C0_norm/C1）、`ElementParam = float`（gamma/beta/统计量）、`ElementAccumulator = float`。CMake 目标类型 `mix`；optest JIT 侧 `JitKernelType::MIX`。

**L1 布局图**（offset 单位 Byte，静态划分）：

```text
0                    CACHE_MAX(256K)              +2*L1B0_TILE(128K)      +L1A0
┌──────────────────────────────┬───────────────┬───────────────┬───────────────┐
│  L1 cache（行块归一化结果 zN） │ L1 B0 slot 0  │ L1 B0 slot 1  │  L1 A0 tile   │
│  BM × n0Aligned，Mmad1 左矩阵 │ (BK0×BN0 zN)  │ (BK1 复用同区) │ (BM×BK0 zN)   │
└──────────────────────────────┴───────────────┴───────────────┴───────────────┘
```

Matmul0（B0）与 Matmul1（B1）分相复用同一 B 槽位区间（同核内分相执行，无并发冲突）；phase0/phase1 各自独立做 `CanImplement` 预算。

**UB 布局图**：

```text
UB_TILE（UB_STAGES 保留区）│ UB_LN16（chunk fp16）│ UB_LN32（chunk fp32）│ UB_ACC（rowSum/rowVar）│ UB_STATS（mean/var/invstd/gamma/beta）
```

**跨核 flag 协议表**（每行块 2 个 flag，`LN_SYNC_MODE=4`；两侧 set/wait 严格配对，pipe = 生产者最后访问 / 消费者首次访问）：

| flag | 方向 | set（pipe） | wait（pipe） | 语义 |
| --- | --- | --- | --- | --- |
| `FLAG_CHUNK_GATHERED(5)` | AIC → AIV0 | AIC：W0 fixpipe 全部完成并 `PipeBarrier<PIPE_FIX>` 后，`PIPE_MTE1` | AIV0：`PIPE_V`（pass0 前） | 本行块 W0 就绪 |
| `FLAG_CHUNK_READY(4)` | AIV0 → AIC | AIV0：W1 写完并 `PipeBarrier<PIPE_ALL>` 后，`PIPE_MTE3` | AIC：`PIPE_MTE1`（`CopyA1TileFromGm` 前） | 本行块 W1 就绪 |

**核内事件表**（HardEvent + event id，ctor 对首次等待做 prime、循环内用后重置，两侧严格配对）：

| 事件 | id | set 侧 | wait 侧 | 语义 |
| --- | --- | --- | --- | --- |
| `MTE2_MTE1` | 0 | Mmad0/1 切片循环前 | 同处 | GM→L1（MTE2）先于 L1→L0（MTE1） |
| `M_MTE1` | 2 | 每片 mmad 后（ctor prime） | 每片 L1→L0 拷贝前 | 单缓冲 L0A/L0B 不被下一片覆盖 |
| `MTE1_M` | 1 | 每片拷贝后 | 每片 mmad 前 | L0 数据就绪才发射 Mmad |
| `M_FIX` / `FIX_M` | 0（L0C 域） | mmad 后 / fixpipe 后（ctor prime） | fixpipe 前 / 下一 tile initC 前 | L0C 产出→消费→复用闭环 |
| `MTE1_MTE2` | 3/4/6（A0/B0+slot/CACHE） | L1 读毕释放（ctor prime） | 下一 MTE2 覆写前 | L1 槽位 WAR |

**阶段流水表**：

| 阶段 | AIC | AIV0（subBlockIdx==0；AIV1 空闲） |
| --- | --- | --- |
| Init | 解析 shape；建立 GM/TLA layout；`SetMMLayoutTransform(true)`（Cube 按 N 分形优先写 L0C，fixpipe 源步进依赖此序）；prime 核内事件 | 配对初始化 |
| Phase 0 | 逐 N0-tile：`Matmul0Tile`（KL0=32 切片 Mmad0，FP32 L0C 累加）→ `Fixpipe0ToGm`（`CopyL0CToGmTla`，nz2nd + `SetFixpipeNz2ndFlag(1,1,1)`，FP32→FP16 随路，写 W0 行填充区）| — |
| W0 交接 | `PipeBarrier<PIPE_FIX>` → `SetFlag(GATHERED)` | `WaitFlag(GATHERED)`（PIPE_V） |
| LN 三遍 | — | pass0：逐 chunk `StageChunkFromGm`（GM→UB 紧凑行，MTE2）+ `AccumulateRowSumGathered` → `FinalizeMean`；passA：重取 chunk + `CenterSquareChunk` → `FinalizeVar`（invstd=rsqrt）；passB：重取 chunk + `NormalizeChunkInPlace`（乘加 gamma/beta，FP32 计算）→ MTE3 写 W1 |
| W1 交接 | `WaitFlag(READY)`（PIPE_MTE1）| `PipeBarrier<PIPE_ALL>` → `SetFlag(READY)`（PIPE_MTE3） |
| Phase 3 | `WaitFlag<MTE1_MTE2>(EVENT_L1_CACHE)` → `CopyA1TileFromGm`（W1→L1 cache zN，MTE2）→ 逐 M1-tile `Matmul1Tile`（左矩阵 = L1 cache **局部坐标** zN 切片；B1 从 GM 流水搬入 zN 槽位）→ `Fixpipe1ToGm` 写 C1 → `PipeBarrier<PIPE_MTE1>` → 释放 CACHE 槽位 | — |

关键实现要点（均有 camodel 实证，见附录 A）：

- **KL0=32**：950 L0 TileShape 的 K 粒度；`CeilDiv(K,KL0)` 切片，尾片 actualK。`TileMmadTla` 7 参形态（无 unitFlag）；3510 下 zN 路径自动 `disableGemv=true`。
- **B 路径**：GM RowMajor + L1 zN（43 号样例 RowMajor-B 同款）；L1→L0B 拷贝随路完成 zN→nZ 转置。**Mmad1 的 L1 cache 切片必须用槽位局部坐标**（`GetTile(cacheTla, (0, kStart+kl*KL0), ...)`，cache 为整行块布局；B1 槽位则用槽内局部坐标 `(kl*KL0, 0)`）——坐标基准错位是开发期实测出的两类典型错误。
- **FixPipe**：`M_FIX`/`FIX_M` 握手 + nz2nd（`SetFixpipeNz2ndFlag(1,1,1)`，持久 LOOP3_PARA）+ 随路 FP32→FP16；fixpipe 后接 `PipeBarrier<管道>()` 保证后级队列（MTE2/MTE1/V）读到落定数据——camodel 要求 SET 先于 WAIT 到达，所有生产者 set 前必须排空源队列（附录 A-L4）。
- **尾块**：actualM/actualN/curK 全程显式传递；padding 行列不进统计分母、不写 C1 有效区外。
- **工作区调试探针**：`debugC0Norm/debugC0Direct/debugFill` 仅仿真诊断使用（`MLNM_PROBE` 等宏门控），交付形态默认关闭。

### 测试设计

| 口径 | 工程 | 通过条件 |
| --- | --- | --- |
| ① | `tests/optest/` | 基于任务测试集（116 条）精度通过 |
| ② | [ATK](https://gitcode.com/Ascend/ATK) | 泛化 ≥200 例测试条目精度通过 |
| ③ | 样例 smoke | `./106_ascend950_matmul_layer_norm_matmul M0 N0 K0 M1` 输出 `Compare success`（camodel 与真机同口径） |

Golden 保持任务链路与 dtype：`c0 = mm(a0,b0)` → `F.layer_norm(c0,(n0,),gamma,beta,eps=1e-6)` → `mm(c0_norm,b1)`，固定 seed 确定性填充，FP32 累加比对，断言各段 dtype。

| 用例类别 | 覆盖内容 |
| --- | --- |
| 官方主集 | CSV 116 条全量参数化（四维扫描） |
| 尾块/非对齐 | M0/K0/N0/M1 不被 BM/BK/BN 整除（如 64/80/64/32、48/96/128/96，camodel 已过） |
| 最小 shape / 零方差 | 常量行验证 eps 与无 NaN |
| 数值与特值 | 正负随机、近零、较大有限值；gamma=1/beta=0、随机 FP32 |
| 负向接口 | dtype、rank、shape 关系、stride、device 不匹配 |

精度方法：`relative_error = |actual-golden| / (|golden|+1e-7)`，记录 MERE / MARE 与最大绝对误差；不得用放宽阈值替代根因分析。失败日志打印 case id、shape、档位、seed 与首个错误位置。

性能采集用 `msprof op`，逐 case 记录 `idx,M0,K0,N0,M1,baseline_us,kernel_us,speedup,...`；判定 `mean(speedup) > 1.1`，逐 case 报表并单独报告小 shape（标杆 19~30 us）区间。小 shape 提高 warmup（≥30）排除未提频；任何 TileShape 调整后 clean rebuild 并在自验报告备注。

# 性能优化策略映射（r3 修订）

8 大优化策略逐条映射（方法论同 r2；r3 按最终架构修正 ⑤⑥ 的落地形态）：

| # | 策略 | r3 结论 |
| --- | --- | --- |
| ① | 自适应 Tiling | **已内置**：BM 三档（N0 分档）静态实例 + host 分派；`col_blocks` 复制策略列迭代项（当前每行块任务覆盖全部 M1 列） |
| ② | 多级缓冲流水 | **已内置**：L1B_STAGES=2（B 槽位 ping-pong）、UB chunk 串行化由 LN 块握手承担 |
| ③ | Stride 搬运重排 | **读侧已内置**：B0/B1 GM→L1 zN 分形搬运；W0/W1 行填充（`n0Aligned`）保证 32B 整块 GM 搬运 |
| ④ | Cube 伪计算 | **已内置**：FixPipe 随路 FP32→FP16（两段 Matmul + W0/W1 写出共 4 处随路 cast，V 段零 cast 负担） |
| ⑤ | L1 Buffer 长驻 | **部分内置**：归一化行块 cache 长驻 L1 供 Matmul1 全部 M1-tile 复用（左矩阵零 GM 重读）；原始 C0 不再常驻（GM-staging 决策） |
| ⑥ | UB 融合 + L0C 暂存 | **已内置**：统计量（mean/var/invstd）全程 UB；LN 三遍 chunk 级 UB 融合（cast/center/square/normalize 单遍完成）；C0/C0_norm 不对外暴露（内部 workspace） |
| ⑦ | Bank Conflict 规避 | **可选迭代**：L1 cache zN 分形访问天然规避行冲突；shuffleK 思想迁移候选 |
| ⑧ | Dual-Stream Pipeline | **已内置**：MIX 双流；行块粒度 AIC(Mmad0→fixpipe) ∥ AIV0(上一行块 LN) 经 grid-stride 自然形成跨行块流水；SplitK 不采用（破坏行内统计闭合） |

主瓶颈预判：大 shape 瓶颈为 AIC Cube 吞吐 + W0/W1 四趟 GM 流量（叠加写带宽），收益主要来自消除 2 次 launch 与统计量不出 UB；小 shape（M0=128，27 条，标杆 19~66 us）瓶颈为任务欠载（BM=64 时仅 2 行块任务）+ launch 地板，迭代方向为更小 BM 档位与 col_blocks 复制。校准手段：`msprof op` 分相耗时与 AIC/AIV 空泡占比；实测后回修本表。

## 优化项落地清单

| 优化项 | 落地标注 | 证伪条件 |
| --- | --- | --- |
| BM 三档静态分派（①） | 必落地（已实现） | T4 区间 speedup<1 → 更小 BM/col_blocks 复测 |
| L1B 双缓冲（②） | 必落地（已实现） | static_assert 失败（静态可证） |
| W0/W1 行填充 + 32B 整块搬运（③） | 必落地（已实现） | profiling 显示 GM 中转占比 >30% → 切 L1 直通迭代 |
| FixPipe 随路 cast ×4（④） | 必落地（已实现） | — |
| 归一化 cache L1 长驻（⑤） | 必落地（已实现） | cache 档位放不下 → BM 降档 |
| LN chunk UB 融合（⑥） | 必落地（已实现） | Vector 段占比过高 → 调 LN_CHUNK_COLS |
| col_blocks 复制（①） | 可选迭代 | 代价模型 `work(c)=c·F0+F1` 实测校准 |
| W0 L1 直通（真机 `DataCopyL1ToUB`） | 真机迭代项 | 真机 profiling 证明 GM 中转为主要瓶颈时切换 |
| SplitK / 代数变形 | 不采用 | 破坏行内统计闭合 / B1 流量翻倍 |

## 参考仓证据清单（代码级）

| 证据点 | 位置 | 支撑结论 |
| --- | --- | --- |
| Ascend950 资源常量 | `include/catlass/arch/arch.hpp` | L1=512K/UB=248K/L0A=L0B=64K/L0C=256K；预算基准 |
| 43 号样例 Cube 流水 | `examples/43_ascend950_basic_matmul/` + `block_mmad_pingpong_tla.hpp` | KL0=32 切片、B RowMajor→L1 zN→L0 nZ 路径、M_FIX/unit-flag 纪律（实验证明两者均可用） |
| `CopyL0CToGmTla` nz2nd | `gemm/tile/ascend950/copy_l0c_to_gm.hpp` | `SetFixpipeNz2ndFlag(1,1,1)` + `unitFlag` 参数语义 |
| `CopyL0CToL1Tla` | `gemm/tile/ascend950/copy_l0c_to_l1.hpp` | L0C→L1 zN 直写（FP32→FP16 在途），真机迭代路径组件 |
| MIX 编排范式 | `kernel/matmul_mix_fixpipe_opti.hpp` | 1:2 发射比、`GetSubBlockIdx`、cross-core flag 管道归属 |
| optest 接入模式 | PR #678 文件清单 + `tests/optest/docs/design.md` | 3.2.1 接入文件点 |
| camodel 并行仿真方法 | 附录 A | dav_3510 快照的可用原语集合与验证结论 |

# 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Ascend 950（性能采集平台 Ascend 950PR，dav_3510） | √ |

依赖 Ascend 950 的 MIX 数据通路（L0C FixPipe nz2nd、cross-core flag、AIC/AIV 分相）与片上缓存容量，不承诺 Atlas A2/A3 可用。

# 算子约束限制

任务书给出的约束为**无**；本设计不额外声明 shape 白名单，对外契约仍约束：

| 序号 | 约束 | 可判定条件 |
| --- | --- | --- |
| C1 | dtype / rank | A0/B0/B1/C1=FP16、gamma/beta=FP32；A0/B0/B1 二维、gamma/beta 一维、C1 二维 |
| C2 | shape 关系 | `A0.sh[1]==B0.sh[0]`；`B0.sh[1]==B1.sh[0]==gamma.numel()==beta.numel()` |
| C3 | 维度取正 | M0/K0/N0/M1 ≥ 1 且可由运行时参数类型表示 |
| C4 | eps 固定 | `1e-6`，不可配置 |
| C5 | 归一化轴 | 仅 `C0` 最后一维（N0），无其他广播 |
| C6 | 尾块 | 功能路径支持任意 shape；性能只对 116 条承诺 |
| C7 | 中间结果不可见 | W0/W1/mean/variance 不出现在对外输出与公开参数结构（W0/W1 由 launcher 内部申请释放） |
| C8 | B0/B1 布局符合性 | **定稿 RowMajor**（`b0[k*N0+n]`/`b1[n*M1+p]`）：与官方 43 号样例 B 路径一致、TileCopyTla 无 GM ColumnMajor→L1 特化、ColumnMajor 组合端到端验证不正确；与任务书 ColumnMajor 标注的差异已在样例 README「布局约定说明」显式声明 |

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 功能 | 116 条全部通过、无 crash/timeout/NaN；ATK 泛化 ≥200 例通过 | 任务书"测试标准" |
| 精度 | 满足生态算子开源精度标准中 FP16 输出档；MERE/MARE 与最大绝对误差，逐条检查 NaN/Inf 位置 | [生态算子开源精度标准](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md) |
| 性能 | `平均(标杆时延) / 平均(测试时延) > 1.1`，标杆为小算子拼接 | 任务书"性能要求" |
| 采集口径 | `msprof op`；涉及不同实现方案与 TileShape 调整须备注说明 | 任务书"性能要求" |
| 交付 | 设计文档过评审并合入 cann-competitions；自验报告覆盖全部功能场景；README 完整 | 任务书"文档规范要求" |

性能标杆基准（来源：任务测试集「小算子标杆耗时(us)」列，116 条，采集平台 Ascend 950PR）：最小 19.005 / 中位 79.460 / 均值 132.721 / P90 242.849 / 最大 1188.580 us。标杆随 shape 组合非单调，一律**逐 case 比对**；小 shape（19~30 us）受 launch 与首尾 flush 影响最大，提高 warmup 并单独报告。

## 可维护性：诊断资产与仿真约束

开发期沉淀的维护资产（交付仓 `experimental/matmul/minimal_mlnm/`，不在 PR 主交付内）：

- 单段诊断样例：Mmad0 直转储、FixPipe 忠实性、GM-staging 各段独立验证（C0 直转储与 golden 全量一致 maxErr=0.0039）；
- 最小 MIX 基线 `minimal_mix_mmad`（4 组 shape 精度通过）——回归"Cube 通路是否健康"的首查工具。

**附录 A：dav_3510 camodel（CANN 9.2.0 快照）约束目录**（8 轮实验结论，真机不适用、升级快照需复验）：

| # | 约束 | 工程含义 |
| --- | --- | --- |
| L1 | AIC `DataCopyL1ToUB` 静默 no-op | L1→UB 读回不可用 → 本设计改 GM-staging |
| L2 | SET 必须先于 WAIT 到达 | 所有生产者 set 前排空源队列（`PipeBarrier<源管道>()`） |
| L3 | unit flag 单发 | 同代多个 0b11 fixpipe 会永等；整代只发一次 |
| L4 | ctor 重复 prime 同一 flag 挂死 | prime 一次性、循环内重置 |
| L5 | ASC 构建不跟踪 hpp 依赖 | **改 .hpp 后必须 touch 对应 .cpp 再编译**，否则跑陈旧二进制（开发期 15 轮结论作废的根因） |
| L6 | 并行模式必需 | 默认串行模式下 MIX flag 自旋分钟级且 watchdog 崩溃；用 `parsim` 并行配置 |
| L7 | v3 仓构建开关 | camodel 需 `cmake -B build -DCATLASS_ARCH=3510 -DENABLE_SIMULATOR=ON -DNPU_MODEL=dav_3510 -DENABLE_PRINT=ON`（默认链真机 HAL） |

## 兼容性分析

| 维度 | 分析 |
| --- | --- |
| 既有接口 | 新增 example 私有参数结构，不改动既有 Matmul/LayerNorm 行为与参数语义；不修改既有 kernel 模板签名 |
| 公共组件 | 仅新增 kernel 组件（`matmul_layer_norm_matmul.hpp`）；复用既有 tile 组件（`CopyL0CToGmTla`/`TileCopyTla`/`TileMmadTla` 等），无公共 API 变更 |
| Python API | `torch_catlass` 仅新增测试入口，不影响既有 wrapper |
| 架构隔离 | 950 特性由 `__NPU_ARCH__ == 3510` 门卫，非 950 平台不编译该目标 |

必要提交结构（对齐 PR #678 实际交付）：`examples/106_ascend950_matmul_layer_norm_matmul/` + `include/catlass/gemm/kernel/matmul_layer_norm_matmul.hpp` + `tests/optest/`（`catlass_kernel_jit.h`、`kernels/106_*/`、`src/`、`torch_catlass/`、`tests/`）。

| 类别 | 内容 |
| --- | --- |
| 允许 | 算子样例、`include/catlass` 组件、optest 测试件、README、设计文档；精度与性能证据以 PR 描述形式补充 |
| 不允许 | 个人敏感信息；PR 提交（文件）中附注精度/性能测试报告；编译中间文件与二进制；辅助测试临时脚本 |

# 修订记录

| 版本 | 修订内容 |
| --- | --- |
| r2 | ① 新增「性能优化策略映射」章节；② 预算算式以仓内真实常量重算；③ 明确 L1 zN 布局契约与 `DataCopyL1ToUB` 读回路径；④ 补充 mix(1,2) 发射比契约与 flag 管道归属；⑤ optest 接入文件点核对成表；⑥ 新增 shape 分档表 T0~T4；⑦ 备选方案表扩充；⑧ 测试设计新增片上布局往返用例 |
| r3 | ① **架构定稿为 GM-staging 中转**（W0/W1 行填充 GM workspace）：r2 的「L1 常驻 + `DataCopyL1ToUB` 读回」在本仿真快照不可验证（附录 A-L1），GM-staging 全链各段已单独实证，且消除逐 tile stage 握手与 cache WAR 协议；数据流图、阶段表、flag 协议表、L1/UB 布局图全部按最终代码重写；② **收益表述修正**：GM-staging 存在 W0/W1 四趟中转流量，r2「省 128 MiB GM 流量」的表述不再成立，收益重述为消除 2 次 launch + 统计量不出 UB + 流水重叠；「W0 L1 直通」列为真机迭代项；③ **布局契约符合性定稿**：实现按 B0/B1 RowMajor（golden `b0[k*N0+n]`/`b1[n*M1+p]`），与任务书 ColumnMajor 标注的差异定为 C8 已声明偏差（依据：43 号样例 RowMajor-B、TileCopyTla 无 CM→L1 特化、CM 组合端到端验证不正确）；④ 策略映射按最终架构修订（⑤ 改为归一化 cache 长驻、⑥ 补 GM workspace 中转、①col_blocks 列迭代）；⑤ TileShape 档位收敛为 BM 三档（64/32/16 按 N0），L1 预算按最终 L1 地图（CACHE_MAX 256K + B 槽位 + A0）重算；⑥ 新增核内事件表、Mmad1 局部坐标要点、`PipeBarrier` 排空纪律；⑦ 新增附录 A（camodel 约束目录，含 L5 陈旧二进制纪律）与可维护性资产清单；⑧ 交付落位对齐 PR #678（examples/106 + include + tests/optest） |
