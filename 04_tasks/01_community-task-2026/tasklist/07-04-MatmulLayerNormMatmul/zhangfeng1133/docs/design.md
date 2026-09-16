# MatmulLayerNormMatmul 算子设计文档（Ascend 950 / CATLASS）

> 融合算子 `Matmul → 行级 LayerNorm → Matmul`，**单次 kernel launch** 完成三段，中间结果不落 GM
> 仓库：[cann/catlass](https://gitcode.com/cann/catlass)，代码落 `experimental/matmul/matmul_layer_norm_matmul/` 与 `tests/optest/`
> 硬件：Ascend 950（性能采集平台 Ascend 950PR）；CANN 版本：算子开源仓指定版本；语言：Ascend C
> 性能对标 `torch.mm + F.layer_norm + torch.mm` 小算子拼接，判定式「平均标杆时延 / 测试时延 > 1.1」
> 团队 `zhangfeng1133`。本文为**开发前设计文档**，只含设计方案、验收标准与验证方法，不含实测结果

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

小算子拼接至少 3 次 launch，中间矩阵须写回 GM 再读。按测试集上界（M0=2048、N0=8192、FP16）估算，`C0` 与 `C0_norm` 各约 32 MiB，写+读约 128 MiB 额外 GM 流量且与计算无重叠收益。本设计融合进**单次 MIX kernel launch**：AIC 执行两个 Matmul，AIV 执行 LayerNorm；`C0`/`C0_norm` 驻留 L1，统计量驻留 UB 并按内部 workspace 管理，唯一 GM 写回为 `C1`。

### MatmulLayerNormMatmul算子功能分析

本任务为**新增融合样例**，不是历史 TBE 算子的语义迁移：catlass 仓无与该三段链路等价的既有实现，外部接口、dtype 与 eps 语义均由任务书定义，因此不存在与旧实现的性能/精度对比基线，唯一基线是小算子拼接方案。

参数表（逐列对齐任务书）：

| 参数名 | 输入/输出/属性 | 描述 | 使用说明 | 数据类型 | 数据格式 | 维度(shape) | 非连续Tensor | 内存 stride / Host 判定式 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A0 | 输入 | 第一个 Matmul 左矩阵，形状 (M0, K0) | RowMajor 布局 | FP16 | ND | 2 | — | `(K0,1)`：`stride(0)==K0 && stride(1)==1` |
| B0 | 输入 | 第一个 Matmul 右矩阵，形状 (K0, N0) | ColumnMajor 布局 | FP16 | ND | 2 | — | `(1,K0)`：**禁止按 RowMajor 解释** |
| B1 | 输入 | 第二个 Matmul 右矩阵，形状 (N0, M1) | ColumnMajor 布局 | FP16 | ND | 2 | — | `(1,N0)`：**禁止按 RowMajor 解释** |
| gamma | 输入 | LayerNorm 缩放参数，长度 N0 | 可学习参数 | FP32 | ND | 1 | — | `(1,)`：`is_contiguous()` |
| beta | 输入 | LayerNorm 偏置参数，长度 N0 | 可学习参数 | FP32 | ND | 1 | — | `(1,)`：`is_contiguous()` |
| C1 | 输出 | 第二个 Matmul 输出矩阵，形状 (M0, M1) | 唯一对外输出 | FP16 | ND | 2 | — | 算子分配，RowMajor `(M1,1)` |

ColumnMajor 的 `B0`/`B1` 在 Torch 视角是"列主 stride 的二维 Tensor"，不等于"所有输入 contiguous"；Host 不得因 `stride(0)!=1` 误判布局或直接拒收。其余契约：**单次 launch** 完成三段（任务书功能要求 1）；`C0`/`C0_norm` 片内、`Mean`/`Variance`（含 invstd）由算子内部 workspace 管理且不对外暴露（要求 2）；`eps` 固定 `1e-6` 无属性；无广播，gamma/beta 仅沿 N0 作用；`C1` 为唯一对外输出；允许用多种 Matmul 方案达成更好性能（要求 3）。

# 需求分析（required）

## 需求描述

用 Ascend C / CATLASS 在 Ascend 950 上实现 `MatmulLayerNormMatmul`：单次 launch 完成 `Matmul → LayerNorm(行级, eps=1e-6) → Matmul`；无中间矩阵 GM 读写；功能与精度通过任务测试集（116 条）及生态算子开源精度标准；平均标杆时延 / 测试时延 > 1.1；按 catlass 规范交付样例、`include/catlass` 组件、optest 接入件、README 与自验证据。

## 需求拆解

| 编号 | 子需求 | 设计响应 | 验证方式 |
| --- | --- | --- | --- |
| R1 | dtype 契约 | A0/B0/B1/C1=FP16，gamma/beta=FP32 | Host 校验 + 负向用例 |
| R2 | 布局契约 | B0/B1 按 ColumnMajor stride 组装 Tile | stride 专项用例、逐 case 比对 |
| R3 | 单次 launch | 单个 `__mix__` kernel，cross-core flag 配对 | `msprof op` kernel 列表、代码审查 |
| R4 | 无中间 GM 读写 | `C0`/`C0_norm` 驻留 L1，统计量驻留 UB | workspace=0、内存分析、代码审查 |
| R5 | 行级 LayerNorm | 每任务完整持有行块 N0，FP32 两遍中心化统计 | 与 golden 比对、常量行用例 |
| R6 | 测试集覆盖 | 116 条全量参数化 | pytest 全量执行日志 |
| R7 | 精度达标 | 中间 Matmul/LN 按 FP16 语义，统计 FP32 | 精度标准判定 + NaN/Inf 位置检查 |
| R8 | 性能达标 | 片上复用、AIC/AIV 流水、shape 分档 | `msprof op` 逐 case 倍率 |
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

kernel 内实际形式：`invstd[m] = rsqrt(max(var[m],0.0f) + 1e-6f)`；`C0_norm = (C0-mean[m])*invstd[m]*gamma[n]+beta[n]`。方差取总体方差 `N0`（不做 Bessel 校正）与 `F.layer_norm(normalized_shape=(N0,))` 口径一致；`rsqrt` 前对 `var` 取 0 下界，避免负数开方产生 NaN。

### 支持数据类型

| 阶段 | 输入 dtype | 输出/中间 dtype | 计算 dtype |
| --- | --- | --- | --- |
| Matmul0 | A0/B0 FP16 | `C0` FP16 | Cube FP32 累加，FixPipe 转 FP16 |
| LN 统计 | `C0` FP16 | mean/var/invstd FP32 | FP32 |
| LN 归一化 | `C0` FP16 + gamma/beta FP32 | `C0_norm` FP16 | FP32 计算后转 FP16 |
| Matmul1 | `C0_norm` FP16、B1 FP16 | `C1` FP16 | Cube FP32 累加，FixPipe 转 FP16 |

`C0`/`C0_norm` 取 FP16：与 PyTorch 链路 FP16 `torch.mm` 及 FP16 LayerNorm 输出的 dtype 传播一致，并使完整行块可片上驻留（FP32 令缓存需求翻倍）。**待锁定项**（实现前置动作）：编码前用 smoke case 打印 golden 链路 `c0.dtype`/`c0_norm.dtype`；若不符先修订本节与精度路径再实现 kernel，不允许用放宽阈值掩盖 dtype 语义差异。

### 支持形状

任务测试集 `test_case/MatmulLayerNormMatmul_测试集.csv` 共 116 条：M0 ∈ {128,512,1024,2048}，K0 ∈ {768,2048,4096,8192}，N0 ∈ {2048,3072,4096,8192}，M1 ∈ {768,2048,4096}。

测试集内 K0/N0/M1 均可被 256 整除，但实现按 `CeilDiv` + 有效 shape 处理尾块，**不得把对齐特征写成接口限制**；最小合法单位 K0/N0/M1 ≥ 1、M0 ≥ 1。

## 算子实现

### 实现方案

主方案：「行块完整片上驻留」的 Ascend 950 MIX kernel，每个 AIC/AIV 配对任务持有一个行块并完整覆盖其 N0。

```text
GM A0/B0 ─► AIC: Matmul0 (FP32 L0C) ─► FixPipe FP16 C0 tile ─► AIV UB rowSum ─► mean ─► L1_C0(FP16)
L1_C0 ─► AIV: 中心化方差 ─► var ─► invstd ─► normalize+gamma/beta ─► 原位覆写 L1_C0_norm
GM B1 ──┴─► AIC: Matmul1 (L1-resident 左矩阵) ─► FixPipe FP16 ─► GM C1 (唯一写回)
```

自洽性：① 每行统计量在单任务内闭合、无跨 AIC 全局归约，故不需要第二次 launch 或额外归约 kernel；② `C0`/`C0_norm` 不落 GM，"单次 launch"与"无中间显存读写"同时成立；③ 归一化后的行块对该任务的多个 M1 列 tile 复用，不重算 Matmul0；④ 各任务 `C1` 写区间互斥，无需 atomic。

| 备选方案 | 结论 | 原因 |
| --- | --- | --- |
| `C0` 写 GM workspace | 不采用 | 与"无中间显存读写"目标冲突 |
| 不缓存 `C0`，先统计后重算 | 仅调试 | 至少重复一次 Matmul0，分块后重复更多 |
| 一遍 `sum`/`sumsq` 求方差 | 非默认 | `E[x²]-E[x]²` 在均值大方差小时存在抵消 |

#### 3.2.1 Host 侧设计

**1. 样例组织**（`op_name = matmul_layer_norm_matmul`）：

```text
catlass/
├── experimental/matmul/matmul_layer_norm_matmul/{CMakeLists.txt(mix), matmul_layer_norm_matmul.cpp,
│       README.md, test_matmul_layer_norm_matmul.py}          # 样例入口 + optest 测试件
├── include/catlass/...                                        # kernel / block / tile 分层组件
└── tests/optest/{include/catlass_kernel_jit.h,                 # 算子函数签名注册
        kernels/matmul_layer_norm_matmul/{*.cpp, CMakeLists.txt},
        kernels/{CMakeLists.txt, common/kernel_runner.h},
        src/catlass_matmul_layer_norm_matmul.cpp,                # 中间层与 JIT 参数传递
        src/include/template/matmul_layer_norm_matmul.h,
        torch_catlass/{__init__.py, ops/__init__.py, ops/matmul_layer_norm_matmul.py}, README.md}
```

Python 入口：`torch_catlass.matmul_layer_norm_matmul(a0, b0, b1, gamma, beta) -> c1`（`(M0,M1)`、FP16、RowMajor）。

**2. 运行时参数**：

```cpp
struct MatmulLayerNormMatmulProblemShape { uint32_t m0, k0, n0, m1; };
struct MatmulLayerNormMatmulParams {
    void* a0; void* b0; void* b1; void* gamma; void* beta; void* c1;
    MatmulLayerNormMatmulProblemShape shape;
};
template <class Kernel>
void RunMatmulLayerNormMatmul(uint32_t blockNum, aclrtStream stream,
                              const MatmulLayerNormMatmulParams& params);
```

`eps` 为编译期常量 `1e-6f`，不进入参数结构；TileShape / 流水深度 / scheduler 由 host 绑定的静态 kernel 类型承载。

**3. 参数校验与错误处理**（分配输出与下发 kernel 之前，任一失败即返回、不触碰 NPU）：

| 序号 | 校验项 | 判定条件 |
| --- | --- | --- |
| V1 | 指针与设备 | 五个输入及输出地址非空，且同一 NPU device |
| V2 | dtype | A0/B0/B1=FP16，gamma/beta=FP32 |
| V3 | rank 与取正 | A0/B0/B1 二维、gamma/beta 一维；四维均 >0 且可安全转 `uint32_t` |
| V4 | 归约维 | `A0.shape[1] == B0.shape[0]` (=K0) |
| V5 | 融合维 | `B0.shape[1] == B1.shape[0] == gamma.numel() == beta.numel()` (=N0) |
| V6 | stride | A0 `(K0,1)`；B0 `(1,K0)`；B1 `(1,N0)`；gamma/beta 连续 |
| V7 | 输出 | `C1` 固定分配 `(M0,M1)` FP16 RowMajor |

地址取 `tensor.data_ptr()`（逻辑首元素），不得用 `tensor.storage().data()`，否则带 storage offset 的合法 view 会读错位置；校验失败以 `TORCH_CHECK` 报错并不启动 kernel。

**4. TileShape 静态化与档位**（不引入运行时 selector，一个 host 源文件绑定一组静态模板参数）：

| 档位 | 适用 N0 | BM | C0 cache 预算 | BK0/BK1 | 目标 |
| --- | --- | ---: | ---: | --- | --- |
| S0 | ≤2048 | 64 | 256 KiB | 128/128 | 大行块、Cube 效率 |
| S1 | 3072 | 32 | 192 KiB | 128/128 | 平衡 L1 与并行度 |
| S2 | 4096 | 32 | 256 KiB | 128/128 | 完整行块驻留 |
| S3 | 8192 | 16 | 256 KiB | 128/128 | 大 N0 专用 |
| S4 / S5 | 小 M0 / 尾块非对齐 | S0~S3 减半 / 16 | ≤128 KiB | 128/128 | 增加任务数、覆盖非对齐 |

预算算式（`BN0=BN1=256`，`L1A_STAGES=1`、`L1B_STAGES=2`、`UB_STAGES=2`，L1 留 32 KiB、UB 留 16 KiB 余量。容量按 L1 512 KiB / UB 248 KiB / L0C 256 KiB 假设；**该三项与各组件真实静态分配必须以仓内 `Arch::Ascend950` 提交时常量为准重算，本表不作为容量结论**）：

```text
CACHE = BM*RoundUp(N0,16)*2;  L0C_MAX = max(BM*BN0, BM*BN1)*4
L1_PHASE0 = CACHE + L1A_STAGES*BM*BK0*2 + L1B_STAGES*BK0*BN0*2 + 32KiB
L1_PHASE1 = CACHE + L1B_STAGES*BK1*BN1*2 + 32KiB
UB_BYTES  = UB_STAGES*BM_VEC*BN0*2 + BM_VEC*BN0*4 + 2*BN0*4 + 4*BM*4 + 16KiB   # BM_VEC=BM/2
```

| 档位 | CACHE | L1 PHASE0 | L1 PHASE1 | L0C_MAX | 预算结论 |
| --- | ---: | ---: | ---: | ---: | --- |
| S0 (BM=64) | 256 KiB | 432 KiB | 416 KiB | 64 KiB | 预算内（≤512/256 KiB 假设） |
| S1 (BM=32) | 192 KiB | 360 KiB | 352 KiB | 32 KiB | 预算内 |
| S2 (BM=32) | 256 KiB | 424 KiB | 416 KiB | 32 KiB | 预算内 |
| S3 (BM=16) | 256 KiB | 420 KiB | 416 KiB | 16 KiB | 预算内 |

UB 以 S0 最保守的 `BM_VEC=32` 估算约 83 KiB。上表只证明候选**在容量上有实现空间**，不代表寄存器、event 与流水性能已通过；候选须由 `static_assert` 检查静态资源、`CanImplement` 检查运行时 shape，不允许只依赖文档估算。

**5. 分核与任务 ownership**：`row_blocks = CeilDiv(M0,BM)`，`col_blocks = CeilDiv(M1,BN1)`；`task(r,c)` 负责行块 r 的全部 N0 与 `C1` 列区间 `[c*BN1, min((c+1)*BN1,M1))`。

| 条件 | 策略 |
| --- | --- |
| `row_blocks ≥ aic_core_num` | `col_blocks=1`，每任务负责行块的全部 M1 列 |
| `row_blocks < aic_core_num` 且 `row_blocks×col_blocks ≤ aic_core_num` | 二维网格，各列块任务独立完成本行块 Matmul0 + LayerNorm |
| 任务数超核数 | grid-stride 顺序分发；同一 `C1` 区间不得分给两个任务 |

列块复制会重复 Matmul0 与 LN，代价模型（系数需实现阶段实测校准）：`F0 = 2*BM*K0*N0`，`F1 = 2*BM*N0*M1`，`work(col_blocks) = col_blocks*F0 + F1`；K0 大或 Matmul0 占主导时优先 `col_blocks=1`，仅小 M0、K0 较小且 M1 大时才复制。**已识别的逻辑缺口**：测试集最小 M0=128、M1=768，此时 `row_blocks×col_blocks` 可能仍小于 `aic_core_num`，必然存在空闲核——小 shape 档（标杆 19~30 us）能否达标只能依赖单任务内流水效率与低 launch/flush 开销，属本项目最高风险项，自验阶段须单独采集该区间的 AIC/AIV 利用率与空泡并给出结论。

**6. workspace 契约**：用户可见 GM workspace = 0（`C0`/`C0_norm`→L1，统计量→UB，同步→cross-core flag，唯一写回→`C1`）。若某档位在真实常量下无法完整驻留行块，可退化为"分片 + 内部 workspace"路径：该 workspace 由 launcher 内部申请、不作为公开契约暴露，`Mean`/`Variance` 仍不出现在任何对外输出中；**禁止**为复用既有 multistage workspace kernel 而把 `C0` 写入外部可感知的 GM 缓冲。

#### 3.2.2 Kernel 侧设计

类型契约：`ArchTag = Arch::Ascend950`；`ElementA0/B0/B1 = half`、`ElementGamma/Beta = float`、`ElementAccumulator = float`、`ElementC0/C0Norm = half`、`ElementC1 = half`；`LayoutA0/C0/C1 = RowMajor`、`LayoutB0/B1 = ColumnMajor`。CMake 目标类型为 `mix`；若误注册为仅 AIC 的 cube 类型，AIV 侧 LayerNorm 不会执行，须在 optest 接入时校验。

| 组件 | 层级 | 职责 |
| --- | --- | --- |
| `MatmulLayerNormMatmul` | kernel | 参数解析、任务调度、阶段编排、跨核同步 |
| `BlockMmad0` / `BlockMmad1` | block | 两个 Matmul（Mmad1 以 L1-resident A 读 `C0_norm`，与 GM `B1` 相乘） |
| `BlockLayerNorm` | block | FP32 rowSum、中心化方差、normalize、gamma/beta 仿射 |
| `CopyUbToL1` | block | FP16 `C0`/`C0_norm` 写入 AIC 可读 L1 |

仅服务本样例的组件（如 `BlockLayerNorm`）先置于 kernel 私有实现，确认接口稳定且具复用价值后再下沉公共 `block/`、`tile/`，避免过早扩展公共 API。`BlockMmad1` 不能套用"左矩阵从 GM 搬入 L1"的通用路径，需复用/抽取 L1-resident A 的 copy 与迭代方式，同时保持 B1 的 ColumnMajor 读取与 L0A/L0B/L0C 约束。

| 阶段 | AIC | AIV |
| --- | --- | --- |
| Init | 解析 M0/K0/N0/M1 与档位 TileShape；建立五输入及输出的 GM Tensor 与 TLA layout（B0/B1 按 ColumnMajor stride）；按固定 offset 划分 L1/UB/L0A/L0B/L0C；初始化 AIC→AIV 与 AIV→AIC 的 stage flag（两侧 set/wait 次数严格配对）；计算 `actualM` 与负责列区间 | 同左（配对初始化） |
| Phase 0 | 按 `BK0/BN0` 读 A0/B0 执行 Matmul0，FP32 累加于 L0C；FixPipe 转 FP16 RowMajor tile；经 L0C→UB 回调写入 ping-pong stage；`CrossCoreSetFlag` 通知 AIV，复用 stage 前等 AIV 空闲 | 等 tile ready → FP16→FP32 分块归约累加 `rowSum[row]` → `CopyUb2L1Tla` 写 `L1_C0_BASE + nOffset` → 释放 stage 通知 AIC → 全部 tile 完成后 `mean[row]=rowSum[row]/N0` |
| Phase 1 | 不参与 | 从 L1 重读 `C0`（复用缓存、不重算 Matmul0）：`centered = cast_fp32(load_l1_c0(tile)) - mean[row]`，`rowVarSum += reduce_sum(centered*centered)`；`var=max(rowVarSum/N0,0)`，`invstd=rsqrt(var+1e-6f)`；统计量全程 FP32 |
| Phase 2 | 不参与 | `y=(x-mean[row])*invstd[row]`，`y=y*gamma[col]+beta[col]`，`store_l1_c0(tile, cast_fp16(y))` 原位覆写（语义由 `C0` 变 `C0_norm`）；gamma/beta 每 tile 搬入 UB 后在该任务所有行间复用；Phase 1→2 需 L1 可见性同步 |
| Phase 3 | 对任务负责的每个 M1 列 tile，以 L1 中 `C0_norm[actualM,N0]` 为左矩阵，从 GM 读 ColumnMajor `B1[N0,BN1]`，沿 N0 以 `BK1` 累加 FP32 L0C，FixPipe 转 FP16 后**只向 `C1` 写回一次** | 不参与 |

尾块以 actual shape/mask 参与归约，padding 不计入分母 `N0` 也不写回 `C1` 有效区间之外；M1 尾块按 actual shape 写回，禁止越界。Vector API repeat 超上限必须分批，禁止把大循环次数截断到窄类型。

阶段交接的 flag 配对（三个方向都必须在两侧等量 set/wait，否则死锁或读到半成品）：AIC→AIV（每个 `C0` tile ready，Phase 0）；AIV→AIC（每个 stage 释放，Phase 0 回环）；**AIV→AIC（整行块 `C0_norm` 完成，Phase 2→3）**。Phase 2→3 初版按整行块完成粒度通知 AIC；若改为 tile 级流水，须额外证明：AIC 只消费已完成归一化的 tile、AIV 不覆盖 AIC 尚未消费的 stage、kernel 尾部 drain 无残留未配对 flag。

**特化与 dispatch**：

| 维度 | 处理 |
| --- | --- |
| 计算通路特化 | 主通路为 L1-resident A 的 Matmul1；无法驻留的档位退回"分片 + 内部 workspace"通路 |
| shape 档位 | Host 选择静态目标（S0~S5 对应不同模板实例），kernel 内不做事后分支 |
| 尾块 | 统一走 `CeilDiv` + actual shape，不做单独 kernel 特化 |
| 禁止项 | 不为单条 case 硬编码目标；不引入运行时 kernel 选择器 / PlanId |

同步与死锁约束：flag 区间不与 CATLASS 组件内部 flag 冲突；每 stage 的 ready/idle set 与 wait 两侧完全一致；无有效任务的核不参与 launch，使用全核 barrier 的档位须所有已启动核无条件到达；退出前 drain BlockMmad 并 `PipeBarrier<PIPE_ALL>`。对齐：FP16 搬运按 32 B 对齐，无法保证时用边界 copy 或 `DataCopyPad`；L1 区域与 stage offset 至少按组件要求对齐，profiling 时检查 512 B 对齐对 MTE/FixPipe 的影响。

### 测试设计

| 口径 | 工程 | 通过条件 | 辅助 skill |
| --- | --- | --- | --- |
| ① | `tests/optest/` | 基于任务测试集（116 条）精度通过 | `catlass-example-to-pytest` 由样例代码自动生成测试件 |
| ② | [ATK](https://gitcode.com/Ascend/ATK) | 泛化 **≥200 例** 测试条目精度通过 | `catlass-atk-support` 由已接入 optest 的测试件自动生成 |

Golden 保持任务链路与 dtype（不用全 FP32 结果替代）：`c0 = mm(a0,b0)` → `F.layer_norm(c0,(n0,),gamma,beta,eps=1e-6)` → `mm(c0_norm,b1)`，并断言三者 dtype，防止框架版本变化造成口径漂移。

| 用例类别 | 覆盖内容 | 来源 |
| --- | --- | --- |
| 官方主集 | CSV 116 条全量参数化（四维扫描） | 任务测试集 |
| 档位边界 | S0~S5 每档 ≥1 条 | 自建 |
| 尾块/非对齐 | M0/K0/N0/M1 不被 BM/BK/BN 整除 | ATK 泛化 |
| 最小 shape / 零方差 | 小矩阵 smoke；常量行验证 eps 与无 NaN | 自建 |
| 数值与特值 | 正负随机、近零、较大有限值（固定 seed）；gamma=1/beta=0、gamma=0、随机 FP32；NaN/Inf 位置逐条说明 | 自建 |
| 负向接口 | dtype、rank、shape 关系、stride、device 不匹配 | 自建 |

精度方法：`relative_error = |actual-golden| / (|golden|+1e-7)`，记录 MERE / MARE，对近零 golden 同步记录最大绝对误差；不得用放宽阈值替代根因分析。失败日志须打印 case id、shape、档位、TileShape、seed、MERE/MARE 与首个错误位置。

性能采集用 `msprof op`（参考 [CATLASS 样例性能调试](https://gitcode.com/cann/catlass/blob/master/docs/zh/1_Practice/evaluation/performance_tools.md)），逐 case 记录 `idx,M0,K0,N0,M1,baseline_us,kernel_us,speedup,kernel_variant,BM,BN0,BK0,BN1,BK1,col_blocks,precision_pass`。小 shape 提高 warmup 次数排除未提频影响；切换静态目标或改 TileShape 后必须 clean rebuild 并核对日志中的目标名与参数；任何 TileShape / 列块复制 / 流水深度调整须保留前后对比并在自验报告中**备注说明**（任务书要求）。分析项：AIC/AIV、MTE2/MTE3、FixPipe 占比；AIC↔AIV 空泡；每核任务数与长尾；L1/UB 占用、bank conflict、L2 命中、GM 带宽与 Phase 0~3 热点。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Ascend 950（性能采集平台 Ascend 950PR） | √ |

依赖 Ascend 950 的 MIX 数据通路（L0C→UB 回调、UB→L1 片上中转、AIC/AIV cross-core flag）与片上缓存容量，不承诺 Atlas A2/A3 可用。

## 算子约束限制

任务书给出的约束为**无**；本设计不额外声明 shape 白名单，对外契约仍约束：

| 序号 | 约束 | 可判定条件 |
| --- | --- | --- |
| C1 | dtype / rank | A0/B0/B1/C1=FP16、gamma/beta=FP32；A0/B0/B1 二维、gamma/beta 一维、C1 二维 |
| C2 | shape 关系 | `A0.sh[1]==B0.sh[0]`；`B0.sh[1]==B1.sh[0]==gamma.numel()==beta.numel()` |
| C3 | 维度取正 | M0/K0/N0/M1 ≥ 1 且可由运行时参数类型表示 |
| C4 | eps 固定 | `1e-6`，不可配置 |
| C5 | 归一化轴 | 仅 `C0` 最后一维（N0），无其他广播 |
| C6 | 尾块 | 功能路径支持任意 shape；性能只对 116 条承诺 |
| C7 | 中间结果不可见 | `C0`/`C0_norm`/`mean`/`variance` 不出现在对外输出与公开参数结构 |

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 功能 | 116 条全部通过、无 crash/timeout/NaN；ATK 泛化 ≥200 例通过 | 任务书"测试标准" |
| 精度 | 满足生态算子开源精度标准中 FP16 输出档；比对 MERE/MARE 与最大绝对误差，逐条检查 NaN/Inf 位置 | [生态算子开源精度标准](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md) |
| 性能 | `平均(标杆时延) / 平均(测试时延) > 1.1`，标杆为小算子拼接 | 任务书"性能要求" |
| 采集口径 | `msprof op`；涉及不同实现方案与 TileShape 调整须备注说明 | 任务书"性能要求" |
| 交付 | 设计文档过评审并合入 cann-competitions；自验报告覆盖全部功能场景（日志/截图、整体通过截图、性能截图）；README 完整 | 任务书"文档规范要求" |

阈值引用说明：本文不预置 FP16 档 rtol/atol 数值，实施阶段在自验报告中逐条引用标准原文表格；未引用原文的阈值不作为验收依据。

性能标杆基准（来源：随任务测试集 `test_case/MatmulLayerNormMatmul_测试集.csv` 的「小算子标杆耗时(us)」列，116 条，采集平台 Ascend 950PR，任务书注明）。按该 CSV 复算：

| 统计量 | 最小值 | 中位数 | 均值 | P90 | 最大值 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 标杆耗时(us) | 19.005 | 79.460 | 132.721 | 242.849 | 1188.580 |

| idx | M0 | K0 | N0 | M1 | 标杆(us) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 128 | 768 | 2048 | 768 | 19.005 |
| 54 | 512 | 4096 | 4096 | 4096 | 121.857 |
| 82 | 2048 | 768 | 2048 | 768 | 110.457 |
| 83 | 2048 | 768 | 2048 | 2048 | 91.305 |
| 116 | 2048 | 8192 | 8192 | 4096 | 1188.580 |

标杆随 shape 组合非单调（idx 82 为 110.457 us 而 idx 83 为 91.305 us），故一律**逐 case 比对**，不得用拟合曲线或单一代表值替代。判定式 `speedup = baseline_us / kernel_us`，验收 `mean(speedup) > 1.1`，自验另报逐 case speedup、算术/几何平均、P50 与最差 case。小 shape（19~30 us）受 launch 与首尾 flush 影响最大，属风险最高区间，需提高 warmup 并单独报告。

## 兼容性分析

| 维度 | 分析 |
| --- | --- |
| 既有接口 | 新增 example 私有参数结构，不改动既有 Matmul/LayerNorm 行为与参数语义；不修改既有 kernel 模板签名，新增目标独立编译 |
| 公共组件 | 仅新增 kernel/block/tile 组件；下沉公共组件时其模板参数与资源约束需独立单测 |
| Python API | `torch_catlass` 仅新增测试入口，不影响既有 wrapper |
| 架构隔离 | FP16/FP32 契约与 950 特性由编译期/运行时检查隔离，非 950 平台不编译该目标 |
| 版本与 ABI | CANN 版本跟随 catlass 仓指定版本，改造既有组件时基于合入主干版本以避开并行 PR 冲突；不修改既有 kernel 模板签名，新增目标独立编译 |

必要提交结构（任务书规定）：`experimental/matmul/matmul_layer_norm_matmul/`（CMakeLists.txt、`${op_name}.cpp`、README.md、`test_${op_name}.py`）+ `include/catlass` 分层组件 + `tests/optest/`（`include/catlass_kernel_jit.h` 签名注册、`kernels/${op_name}/`、`kernels/common/kernel_runner.h`、`src/catlass_${op_name}.cpp`、`src/include/template/${op_name}.h`、`torch_catlass/ops/${op_name}.py`、README.md）。

| 类别 | 内容 |
| --- | --- |
| 允许 | 算子样例、`include/catlass` 组件、optest 测试件、README、设计文档；精度与性能证据以 PR 描述形式补充 |
| 不允许 | 个人敏感信息；PR 提交（文件）中附注精度/性能测试报告；编译中间文件与二进制；辅助精度/性能测试的临时脚本；非必要内容（如 Tiling 调优用的模型权重） |
