# MatmulLayerNormMatmul 算子设计文档

## 版本记录

| 版本 | 日期 | 作者 | 说明 |
| --- | --- | --- | --- |
| v1.0 | 2026-09-17 | hor1Zzz | 首版设计：需求分析、三阶段融合方案、Host/Kernel 分层设计、片上资源与 workspace 规划、精度路径设计、测试与性能验收方案 |

**提交路径**

```text
04_tasks/01_community-task-2026/tasklist/09-45-MatmulLayerNormMatmul/m0_56676577/docs/design.md
```

**目标代码仓与分册目录**（任务书《PR 申请合入》规定，参考 PR#678 目录树）

```text
https://gitcode.com/cann/catlass
  experimental/matmul/matmul_layernorm_matmul/   # 算子样例目录
  include/catlass/                               # 必要组件（kernel/block/tile 分层）
  tests/optest/                                  # optest 测试框架接入
```

---

# 一、需求背景（required）

## 1.1 需求来源

本需求来自 CANN 2026 年 **9 月社区任务「MatmulLayerNormMatmul 算子开发」**，任务编号 `09-45-MatmulLayerNormMatmul`。

| 项目 | 内容 |
| --- | --- |
| 任务书 | `MatmulLayerNormMatmul_task_doc.md`（9 月社区任务下发） |
| 任务测试集 | `MatmulLayerNormMatmul_测试集.csv`，116 条 `(M0, K0, N0, M1)` 组合，附 Ascend 950PR 实测标杆时延 |
| 适配硬件 | Ascend 950（性能验收使用 Ascend 950PR，标杆数据为 950PR 真机采集） |
| 开发语言 | Ascend C，基于 CATLASS 模板库 |
| 开源仓 | <https://gitcode.com/cann/catlass> |
| 算子目录 | `experimental/matmul/matmul_layernorm_matmul/` |
| 设计文档模板 | <https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md> |
| 参考合入 PR | <https://gitcode.com/cann/catlass/pull/678> |

## 1.2 背景介绍

### 1.2.1 算子定位与数学定义

`Matmul → LayerNorm → Matmul` 是 Transformer FFN / Adapter 结构中的典型片段：第一次投影得到隐层激活，对隐层做**行级** LayerNorm，再投影回输出维度。三个阶段的数学定义如下。

阶段一，第一个 Matmul（`A0` 为 `(M0, K0)`，`B0` 为 `(K0, N0)`）：

```text
C0[m, n] = Σk A0[m, k] · B0[k, n]，    C0 形状 (M0, N0)，RowMajor
```

阶段二，LayerNorm（对 `C0` 每一行独立计算，归一化维为 `N0`，`eps = 1e-6` 固定）：

```text
mean[m]       = (1/N0) · Σn C0[m, n]
var[m]        = (1/N0) · Σn (C0[m, n] − mean[m])²              （有偏方差）
C0_norm[m, n] = (C0[m, n] − mean[m]) / sqrt(var[m] + eps) · gamma[n] + beta[n]
```

阶段三，第二个 Matmul（`B1` 为 `(N0, M1)`）：

```text
C1[m, p] = Σn C0_norm[m, n] · B1[n, p]，    C1 形状 (M0, M1)，唯一对外输出
```

其中 `gamma`/`beta` 为长度 `N0` 的 FP32 可学习参数，沿行方向逐列作用。

### 1.2.2 小算子拼接基线的开销结构

性能标杆是 PyTorch 小算子拼接方案：

```python
c0      = torch.mm(a0, b0)                                              # launch 1
c0_norm = torch.nn.functional.layer_norm(c0, (n0,), gamma, beta, 1e-6)  # launch 2
c1      = torch.mm(c0_norm, b1)                                         # launch 3
```

该链路的开销分为三类，随 shape 变化此消彼长：

| 开销类别 | 来源 | 随 shape 的变化趋势 |
| --- | --- | --- |
| 固定 launch 开销 | 3 次 kernel launch 的调度、下发与首尾 flush | 小 shape 时占比高 |
| 中间激活的显存往返 | `C0` 写 1 读 1（LayerNorm），`C0_norm` 写 1 读 1（Matmul1） | 正比于 `M0 × N0`，大 `N0` 时占比高 |
| LayerNorm 纯带宽时间 | LayerNorm 无 Cube 计算，是暴露在关键路径上的搬运+归约 | 正比于 `M0 × N0`，三段串行无法重叠 |

**融合算子的收益来源**：① 3 次 launch 合并为 1 次（小 shape 主收益）；② LayerNorm 由 AIV 向量核承担，不再作为独立 kernel 占用端到端时间；③ 中间量全部由算子内部管理、单遍访问，消除对外中间 tensor 的多趟读写。融合后相对拼接基线的净收益为正（收益主项是 launch 合并与读趟数减少）。

### 1.2.3 标杆基线数据

测试集 116 例的标杆时延统计（`msprof op`、950PR 真机采集，数据来自任务测试集）：

| 指标 | 数值 |
| --- | --- |
| 标杆时延合计 | 15395.599 µs |
| 达标要求的测试时延合计上限 | 13996.0 µs（= 15395.599 / 1.1） |
| 标杆时延算术平均 | 132.721 µs |
| 最小 / 最大标杆时延 | 19.005 µs（idx 1）/ 1188.58 µs（idx 116） |

按 `M0` 分组：

| M0 | 例数 | 标杆均值（µs） | 标杆合计（µs） |
| ---: | ---: | ---: | ---: |
| 128 | 30 | 48.6 | 1458.7 |
| 512 | 30 | 83.2 | 2497.1 |
| 1024 | 30 | 150.6 | 4516.8 |
| 2048 | 26 | 266.3 | 6923.0 |

两个特征分组决定优化重心：`M0=128` 小 shape 组由 launch 与流水填充开销主导，融合收益空间最大；`K0=N0=8192` 大形状组（12 例）占标杆总时延约 38.6%，是性能达标的主战场，要求融合后 Cube 效率不劣化。

### 1.2.4 目标仓库现状核查

设计前对 CATLASS master 做了组件级核查，结论决定本设计的复用与新增边界：

| 核查项 | 结论 | 对本设计的影响 |
| --- | --- | --- |
| LayerNorm / RMSNorm 组件 | **不存在**，全仓关键字零命中 | 行统计 + 归一化组件为本算子核心新增点 |
| Cube+Vector 融合骨架先例 | `Gemm::Kernel::SvdQuantMatmulTla`（两段 Matmul 经 GM workspace 衔接 + 全局跨核同步的单 kernel 编排） | 三阶段编排骨架直接参考该先例 |
| Cube MMAD 组件 | `Gemm::Block::BlockMmadTla`（pingpong 双缓冲，L0C FP32 累加） | 两段 Matmul 复用 |
| A0 全载 L1 策略 | `MmadAscend950FullLoadA` DispatchPolicy（`CanImplement` 检查 A tile 驻留 L1） | K0 较小的形状组可复用 |
| 跨核同步原语 | `AscendC::CrossCoreSetFlag/WaitFlag` + `Arch::BarrierFlag/CrossCoreFlag`（FlagID 预算有限） | 阶段切换同步复用 |
| FP32 C 写 GM workspace | 既有样例 `ElementC = float` 先例 | C0 以 FP32 落 workspace 成立 |
| optest / torch 接入 | `tests/optest/torch_catlass/ops/` 30+ 样例注册先例 | 精度与泛化测试走既有框架 |

遵循 CATLASS「最大化复用、最小化创新」原则：新增收敛在融合 kernel 骨架与 LayerNorm 行统计/归一化 AIV 组件，其余复用或按样例先例组装。

---

# 二、需求分析（required）

## 2.1 需求描述

在 Ascend 950 上实现融合 LayerNorm 的双 Matmul 算子：**单次 kernel launch** 内完成 `Matmul → LayerNorm → Matmul` 三段计算；Mean/Variance 等中间结果由算子内部 workspace 管理，**不对外暴露**；允许多种 Matmul 实现方案分派以达成更好性能。

### 2.1.1 接口契约

| 参数 | 方向 | 数据类型 | 逻辑形状 | 物理布局 |
| --- | --- | --- | --- | --- |
| `A0` | 输入 | FP16 | `(M0, K0)` | ND，RowMajor 连续 |
| `B0` | 输入 | FP16 | `(K0, N0)` | ND，ColumnMajor 连续 |
| `B1` | 输入 | FP16 | `(N0, M1)` | ND，ColumnMajor 连续 |
| `gamma` | 输入 | FP32 | `(N0,)` | ND，一维连续 |
| `beta` | 输入 | FP32 | `(N0,)` | ND，一维连续 |
| `C1` | **唯一输出** | FP16 | `(M0, M1)` | ND，RowMajor |

`eps` 固定为 `1e-6`（任务书规定，不开放为入参）。ColumnMajor 为内存排布约定，不改变数学形状：`B0` 元素地址 `base + k + n·K0`，`B1` 元素地址 `base + n + p·N0`。

### 2.1.2 调用形态

CATLASS example 形态（样例 CLI，自带随机输入生成与 golden 比对，成功输出 `Compare success.`）：

```bash
./matmul_layernorm_matmul <M0> <K0> <N0> <M1> [DeviceID]
```

optest / PyTorch 侧（`torch.ops.catlass` 注册，optest 任务集精度与 ATK 泛化精度走此通道）：

```python
c1 = torch_catlass.ops.matmul_layernorm_matmul(a0, b0, b1, gamma, beta)
# a0: (M0, K0) fp16；b0: (K0, N0) fp16 列主；b1: (N0, M1) fp16 列主
# gamma/beta: (N0,) fp32；返回 (M0, M1) fp16
```

`C0`/`C0_norm`/`mean`/`var` 不出现在任何对外接口上。

## 2.2 需求拆解

| 编号 | 子需求 | 设计响应 | 验证方式 |
| --- | --- | --- | --- |
| R1 | 单次 kernel launch 完成三段计算 | AIC+AIV 混合单 kernel，内部三阶段编排（§3.2.2） | profiler 核实单次 launch、代码审查 |
| R2 | 中间量由内部 workspace 管理、不对外暴露 | C0/C0_norm/mean/var 全部走算子私有 workspace（§3.2.1） | 接口审查、内存规划核查 |
| R3 | dtype 契约固定（单组合） | FP16 矩阵 + FP32 γ/β → FP16 输出，FP32 累加（§3.1.2） | 负向用例（非法 dtype 报错） |
| R4 | shape/布局校验 | host 侧乘法链相容与布局校验、四类错误拒绝（§2.3） | 正反向用例 |
| R5 | 任务测试集 116 例精度通过 | optest 测试件按 CSV 逐例参数化（§4.2） | optest 全量执行 |
| R6 | ATK 泛化 ≥ 200 例精度通过 | 泛化矩阵设计（§4.2） | ATK 四件套 |
| R7 | 整体性能 > 1.1× 拼接标杆 | 融合结构 + 按 shape 分派多种 Matmul 方案（§3.2.1/§4.1） | `msprof op` 逐例采集 + 平均比值 |
| R8 | 满足生态算子开源精度标准 | FP32 全流程 golden + FP16 档容差（§3.1.2/§4.1） | 逐元素混合容差判定 |
| R9 | 交付件合规 | 目录树对齐参考 PR#678，PR 不带报告附件/中间产物 | PR checklist |

## 2.3 输入校验与错误处理

host 侧在 kernel launch 前完成全部校验，非法输入直接拒绝、不进 kernel：

| 校验项 | 规则 | 错误类别 |
| --- | --- | --- |
| 维度数 | `A0/B0/B1` 均为 2 维 | shape_mismatch |
| 乘法链相容 | `A0.shape[1] == B0.shape[0]`（K0）、`B0.shape[1] == B1.shape[0] == gamma.numel == beta.numel`（N0） | shape_mismatch |
| 空 tensor | 任一维度为 0（`M0/K0/N0/M1 ≥ 1`） | null_input |
| dtype | 仅支持 §2.1.1 单组合 | dtype_not_supported |
| 数值属性 | eps 属性固定 1e-6，host 侧防御性校验数值合法（≤ 0 拒绝，恒定行 σ²=0 会除零） | attribute_value_out_of_range |
| 布局 | A0/C1 RowMajor、B0/B1 ColumnMajor 连续 | shape_mismatch |

---

# 三、详细设计（required）

## 3.1 算子分析

### 3.1.1 数学公式

三段公式（ε = 1e-6，γ/β 为可学习参数）：

```text
阶段一：  C0 = A0 × B0，                        C0 ∈ R^(M0×N0)，RowMajor

阶段二（对 C0 每一行 m 独立计算，归约轴 = 行内 N0 方向）：
          mean[m] = (1/N0) · Σn C0[m, n]
          var[m]  = (1/N0) · Σn (C0[m, n] − mean[m])²          （有偏方差，两遍法）
          Z[m, n] = (C0[m, n] − mean[m]) / sqrt(var[m] + ε) · gamma[n] + beta[n]

阶段三：  C1 = Z × B1，                          C1 ∈ R^(M0×M1)，RowMajor，唯一对外输出
```

golden 参考实现（FP32 全流程，最终截断 FP16）：

```python
c0 = np.matmul(a0.astype(np.float32), b0.astype(np.float32))
mu = c0.mean(axis=-1, keepdims=True)
centered = c0 - mu
var = (centered * centered).mean(axis=-1, keepdims=True)
z = centered / np.sqrt(var + eps) * gamma + beta
c1 = np.matmul(z, b1.astype(np.float32)).astype(np.float16)
```

### 3.1.2 支持数据类型与精度路径

**外部 dtype**：单组合固定（`A0/B0/B1` FP16、`gamma/beta` FP32、`C1` FP16），无多 dtype 分支。

**内部精度设计**（本算子精度达标的关键决策）：

| 环节 | 输入 | 累加/计算 | 存储/输出 |
| --- | --- | --- | --- |
| Matmul0 | FP16 | **FP32（L0C 硬件累加）** | **C0 以 FP32 写入 workspace** |
| LayerNorm 统计/归一化 | FP32（C0） | FP32 | Z 先在 FP32 域完成归一化与仿射 |
| Matmul1 A 操作数 | FP16（hi/lo 双矩阵，见下） | **FP32（L0C）** | C1 fixpipe cast FP16 |

三项决策及依据：

1. **两段 Matmul 全程 FP32 累加**：K0/N0 可达 8192，FP16 部分和必然超差或溢出，累加器必须 FP32（硬件 L0C 原生支持）。
2. **C0 以 FP32 存储（不降精度）**：C0 的数值动态范围不受控——泛化测试中大数值输入、深 K0 累加下 C0 元素可达 1e8 量级，超出 FP16 上限（65504），FP16 存储会溢出为 inf 并污染整行归一化。C0 保持 FP32 从根上消除该风险。
3. **第二段 MatMul A 操作数采用 hi/lo 残差补偿双矩阵**：归一化结果 Z 若单次量化为 FP16 进入 Matmul1，量化舍入噪声（~2⁻¹¹ 相对量级）会在长度 N0（2048~8192）的点积中累积放大——数值分析表明单次 FP16 量化无法稳定满足 FP16 档 2⁻⁹ 容差的 0.99 匹配率要求，这是本设计选择残差补偿的直接依据。本设计将 FP32 域的 Z 分解为 `hi = round16(Z)` 与 `lo = round16(Z − hi)` 两个 FP16 矩阵（残差在 FP32 中精确，为 FP16 ULP 整倍数），Matmul1 按 `C1 = hi×B1 + lo×B1` 计算：两个 A tile 共用同一次加载的 B1 tile，两条 MMAD 累加进**同一 L0C FP32 累加器**，固定先 hi 后 lo 顺序。A 操作数有效精度恢复到 ~2⁻²² 量级，单次量化的精度损失被消除；代价是 A 侧加载与 MMAD 指令数约 ×2（B 侧不变），属可接受的精度换性能设计。

**方差算法**：采用**两遍法**（先 mean，再 Σ(x−mean)²），拒绝 `E[x²]−E[x]²` 单遍法——后者在 |mean| ≫ σ 时发生灾难性抵消，恒定行场景下方差可为负、开方产生 NaN。两遍法的第二遍在行数据驻留 UB 期间完成（见 §3.2.2），不产生额外显存读。归一化分母 `sqrt(var + eps)`，eps 在开方前相加；FP16→FP32 为精确转换无舍入，FP32→FP16 采用 RNE 舍入（与 golden 终截断同方向）。

### 3.1.3 支持形状与布局

- 任务测试集取值域：`M0 ∈ {128, 512, 1024, 2048}`、`K0 ∈ {768, 2048, 4096, 8192}`、`N0 ∈ {2048, 3072, 4096, 8192}`、`M1 ∈ {768, 2048, 4096}`（116 例，非完整笛卡尔积，按 CSV 逐例执行）。
- 不将测试集取值域写成接口限制：非 tile 倍数维度由尾块路径承接（§3.2.2）；归一化维超出行驻留能力的大 N0 由分块统计回退路径承接；小 M0 由行批收缩承接。上界由设备内存与索引范围决定。
- 退化 case：`N0 = 1` 时 var = 0，Z = beta，输出退化为 beta 行的线性组合，路径天然正确；全零/恒定行输入由 eps 分母保护；NaN/inf 输入按行传播（与 FP32 golden 同口径比对）。

## 3.2 算子实现

### 3.2.1 host 侧设计

host 侧（样例 CLI 与 torch 注册两通道共用同一实现）负责：参数校验（§2.3）→ 分派决策 → tile/行批规划 → workspace 申请 → 发起单个混合 kernel。

#### 1. 分核策略

单个 **AIC(Cube) + AIV(Vector) 混合 kernel**（每 AIC 核配 2 个 AIV 子核，各 AIV 子核独立 248KB UB），三阶段内部分工：

| 阶段 | 执行核 | 切分方式 |
| --- | --- | --- |
| Matmul0 / Matmul1 | AIC | (M-tile, N-tile) 二维 tile 切分，BlockScheduler 遍历多核并行 |
| LayerNorm | AIV | 按 M 方向**行批**切分，strided 循环分配到全部 AIV worker 并行 |

核数、UB/L1 容量一律运行时查询（`GetCoreNumAic()` / `GetBlockNum()` / `GetSubBlockNum()` 等），禁止写死。行归约经 workspace 中转，天然回避「同一 M 行的 N-tiles 分布在不同 Cube 核」的跨核归约同步问题。

**按 shape 的方案分派（多种 Matmul 方案，任务书功能要求第 3 条）**，host 侧按优先级判定，四分支互斥：

| 分支 | 触发条件 | 方案 | 动机 |
| --- | --- | --- | --- |
| full-loadA | `K0 == 768` | A0 tile 全载 L1（`MmadAscend950FullLoadA` DispatchPolicy），跨 N-tile 复用 | K0=768 时 A tile 驻留 L1 后对全部 N-tile 免重复加载 |
| split-k | `M0 == 128 且 K0 ≥ 2048` | K 维切分多核 + 固定序部分和合并 | M0=128 时 (M,N) tile 任务块数低于核数，K 维切分补齐并行度 |
| 小 M 均衡 | `M0 < 512`（非上述命中） | tileN 收缩 + 尾块多核均分 | 泛化小 M 域的负载均衡兜底 |
| 标准 | 其余 | BlockMmad pingpong 标准分档 | M/N 方向任务块数充足 |

任务集 116 例由前三分支 + 标准分支全覆盖（K0=768 → full-loadA；M0=128 且 K0≥2048 → split-k；其余 → 标准）。Phase 2 LayerNorm 对全部分支采用同一行批方案，行批参数随 M0/N0 自适应。

#### 2. 数据分块和内存优化策略

**片上资源**（`Arch::Ascend950`，DAV_3510）：L1 512KB、UB 248KB（每 AIV 子核独立）、L0A/L0B 64KB、L0C 256KB。

**UB（Phase 2 工作集，按行批规划）**：LayerNorm 以「整行联合驻留」的行批为处理单元——单行 C0（FP32）+ γ/β + 行级临时 + hi/lo 双 FP16 输出行 + 归约临时缓冲联合驻留 UB，行批行数 `lnRows` 由 host 按 UB 预算公式自适应计算，并叠加核利用率反馈（行批数不足 AIV worker 数时收缩 lnRows 增并行度）。预算公式：

```text
lnRows × N0a × 8B（C0 行批 FP32 4B + hi FP16 2B + lo FP16 2B）
+ 3 × N0a × 4B（行级临时 + γ + β）
+ 归约临时 4KB + 标量槽 256B  ≤  UB 可用上限
```

（`N0a` 为 32B 对齐后的 N0。）按该公式规划，任务集全部形状的行批工作集均满足单 AIV UB 上限；C0 行批**单次读入**，行统计与归一化在同一驻留内完成，无第二遍显存读。γ/β 每 worker 载入一次、跨自身全部行批复用。

**L1/L0C**：Matmul0/1 的 A/B 分块缓冲走 TileCopy 分级加载（pingpong 双缓冲内建）；L0C 256KB 承载 FP32 累加器 tile；full-loadA 分支以 `CanImplement` 校验 A tile 驻留 L1 的容量资格。

**GM workspace（算子私有，不对外暴露）**：

| 区域 | dtype/布局 | 规模 |
| --- | --- | --- |
| C0 | FP32 RowMajor (M0, N0) | M0×N0×4B |
| C0_norm（hi） | FP16 RowMajor (M0, N0) | M0×N0×2B |
| C0_norm_lo（lo 残差） | FP16 RowMajor (M0, N0) | M0×N0×2B |
| mean / var | FP32 (M0,) × 2 | 2×M0×4B |

合计 `M0×N0×8B + 8×M0` 字节，最大测试用例（2048, 8192, 8192, 4096）约 128 MiB。各区域均为**单遍流式访问**（Phase 2 单读 C0，Phase 3 A 侧 hi/lo 各单读一遍），无重读依赖，容量压力表现为带宽占用而非缓存颠簸。

**关于「无需中间显存读写」的解读**：任务书功能要求第 2 条明示中间结果由算子内部 workspace 管理。本设计将「无需中间显存读写」理解为**不经过对外可见的中间 tensor**——用户无需分配、管理任何中间显存，接口上不存在中间量，拼接基线中 C0/C0_norm 对外的 4 趟读写与 3 次 launch 全部消除；算子内部以单遍流式 workspace 中转中间量是该条款明示允许的手段，融合净收益为正（收益主项是 launch 合并与读趟数减少）。

#### 3. tilingkey 规划策略（host 侧模板分派）

CATLASS 无 `TILING_KEY` 宏机制（ops-* 框架机制不适用），分派的等价实现为 **host 侧按 shape 实例化不同 kernel 模板**（编译期类型分派，§3.2.1 分派表），kernel 侧由 `Params` 结构体携带分支标记与 tile/行批参数走静态分支。`Params` 承载：shape 四元组、eps、分支标记、两段 tile 形状、Phase 2 行批参数（lnRows/批数/尾批行数/有效与对齐双长度口径）、循环边界与 workspace 各区偏移。

### 3.2.2 kernel 侧设计

#### 三阶段编排（单 kernel）

```text
GM 输入            Phase 1 [AIC]               workspace (GM, 算子私有)      Phase 2 [AIV]                    Phase 3 [AIC]              GM 输出
──────    ┌─────────────────────────┐    ┌──────────────────────┐    ┌────────────────────────────┐    ┌────────────────────────┐    ───────
A0 ──────▶│ Matmul0: C0 = A0 × B0   │    │ C0 FP32 (M0,N0)      │    │ 行批单次读入 C0            │    │ Matmul1:               │
B0 ──────▶│ L0C FP32 累加 → fixpipe │──▶ │ C0_norm hi FP16      │◀──▶│ 两遍法统计 mean/var       │──▶ │ C1 = hi×B1 + lo×B1     │──▶ C1
          └─────────────────────────┘    │ C0_norm lo FP16      │    │ FP32 归一化+仿射           │    │ 双 MMAD 同一 L0C 累加  │    (M0,M1)
                    │                    │ mean/var FP32 ×2     │    │ 产出 hi/lo 双 FP16 写出   │    │ fixpipe cast FP16      │
                    └──── 全局跨核 barrier ─┴──────────▲─────────┘    └────────────┬───────────────┘    └───────────▲────────────┘
                                                  全局跨核 barrier ───────────────┴───────────────────────────────────┘
```

**Phase 1 [AIC]：Matmul0**。BlockScheduler 遍历 (M-tile, N-tile)；A0 GM→L1→L0A（RowMajor）、B0 GM→L1→L0B（ColumnMajor）；BlockMmad 沿 K0 循环 MMAD，L0C FP32 累加（无降精度）；tile 完成后 fixpipe 将 FP32 结果写 workspace 的 C0 区。split-k 分支下 K 维切分、各核 FP32 部分和写独立槽位、按核号固定序合并——无原子加。

**barrier 1**：全体核 CrossCore 同步，等待 Phase 1 全部 tile 落 workspace。

**Phase 2 [AIV]：LayerNorm（整行驻留行批，单读）**。行批 strided 分配到全部 AIV worker，每个 worker 处理若干「lnRows 连续行」的行批，批内逐行流水：

```text
载入行批 C0（FP32，单次搬运；尾批/非对齐维度走 DataCopyPad 有效长度口径）
对批内每行 r：
  ① mean[r]  = ReduceSum(行) / N0                          （第一遍）
  ② 中心化 x' = x − mean[r]，var[r] = ReduceSum(x'²) / N0   （第二遍，驻留内完成，免重读）
  ③ invStd = 1 / sqrt(var[r] + eps)
  ④ y = x' × invStd × gamma + beta                          （FP32 域；gamma/beta 与行同长逐元素）
  ⑤ hi = round16(y)；lo = round16(y − hi)                    （RNE 舍入；y−hi 在 FP32 中精确）
写出行批：hi / lo 分别以 FP16 写 workspace（两次搬运）；mean/var 写 workspace
```

行统计（mean/var）为「每行一个标量」的行级归约结果，归一化时以逐行标量广播指令作用整行，一次指令完成，无需物化广播矩阵；gamma/beta 与行同长，直接逐元素运算。

**barrier 2**：同 barrier 1，等待 Phase 2 全部行批完成。

**Phase 3 [AIC]：Matmul1（残差补偿双 MMAD）**。BlockScheduler 遍历 (M-tile, M1-tile)；hi 与 lo 两个 FP16 RowMajor workspace 作 A 操作数走标准 GM→L1→L0A 路径，B1（ColumnMajor）作 B 操作数；沿 N0（第二段 K 向）循环，**每个 k-step 对同一 L0C FP32 累加器发两条 MMAD**（`L0C += hi_tile×B1_tile` 后 `L0C += lo_tile×B1_tile`），B1 tile 每 k-step 仅加载一次供两条 MMAD 共享；fixpipe FP32→FP16 直出 GM 的 C1。

#### 同步设计

- 阶段切换使用全局跨核 barrier（`AscendC::CrossCoreSetFlag/WaitFlag` + `BarrierFlag`，全体核参与），两道 barrier 是已验证融合样例的最小编排，排障面小、行为确定。
- FlagID 资源有限（≤8 量级），阶段化设计只用少量 flag，预算充足；后续若引入行块粒度流水需先核算 FlagID 预算。
- 全体核 barrier 要求所有核（含尾块空计算核）都走完各阶段，避免「部分核不参与」导致的等待不配对。

#### 尾块与边界处理

| 场景 | 处理 |
| --- | --- |
| 非 tile 倍数维度（泛化） | Phase 1/3 尾块按核均分工具切分（len=0 核退出）；GM↔UB 搬运用 DataCopyPad 尾块屏蔽 |
| 有效长度与对齐长度双口径 | 归约/向量指令 count 用有效长度 N0；UB 内行偏移与缓冲分配用 32B 对齐长度，两口径分离传递 |
| 大 N0（超出单行驻留能力） | 分块统计回退路径：行内按 N 分块、分块稳定单遍统计（chunk 均值/M2 + 并行合并公式，抗灾难性抵消）+ 固定序合并 |
| 小 M0（行数少于 worker 数） | lnRows 收缩至 1，行批数 = M0，接受残余闲置（总工作量小，无害） |
| 恒定行 / 全零输入 | 两遍法 var 精确为 0，分母由 eps 保护，与 golden 同路径 |
| NaN/inf 输入 | 按行传播（该行输出全 NaN），与 FP32 golden 同口径比对 |

#### 确定性

默认确定性执行：MMAD K 循环固定序、split-k 固定序部分和合并、行批/chunk 固定分块序，**全链路无原子加**——相同输入多次执行结果逐位一致。任何引入非确定累加的优化（如原子流式归约）不得替换默认确定性路径。

## 支持硬件

| 支持的芯片版本 | 是否支持 |
| --- | :--: |
| Ascend 950PR | √（任务标杆与性能验收平台） |
| Ascend 950（同 DAV_3510 架构） | 代码路径一致，本任务以 950PR 验收为准 |

## 算子约束限制

- 仅支持 `A0/B0/B1` FP16、`gamma/beta` FP32、`C1` FP16 的单 dtype 组合；
- 仅支持 2 维输入（无 batch 维）；`B0`/`B1` ColumnMajor、`A0`/`C1` RowMajor 连续；
- `eps` 固定 `1e-6`；
- `mean`/`var`/`C0`/`C0_norm` 不对外输出，由算子内部 workspace 管理；
- 不将测试集维度取值写为接口限制（尾块与大 N0 回退路径承接泛化 shape）。

---

# 四、可维可测分析（required）

## 4.1 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | 满足生态算子开源精度标准（实验标准）FP16 档：逐元素混合容差 `abs(actual−golden) ≤ atol + rtol×abs(golden)`，`atol = rtol = 2⁻⁹`；matched_ratio ≥ 0.99 且 max_abs_error ≤ max(1e-1, 32×ULP)；golden 为 FP32 全流程参考（§3.1.1） | <https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md> |
| 性能标准 | 整体性能达成 1.1 倍小算子拼接（平均标杆时延 / 测试时延 > 1.1）：116 例总和口径，标杆合计 15395.599 µs → 测试合计 < 13996.0 µs | 任务书 + 任务测试集 CSV |

**性能测量口径**：`msprof op` 采集（参考 CATLASS 样例性能调试文档），在 Ascend 950PR 真机执行，标杆与测试同机同口径；涉及不同实现方案与 TileShape 的调整在结果中备注说明（多方案分派属任务书允许项）。任务书「平均的标杆时延 / 测试时延」按 116 例总和口径执行；自验证报告同时给出逐例比值与两种统计量，不以单一统计量替代另一种宣称达标。

## 4.2 测试设计

**optest（任务测试集精度）**：`experimental/matmul/matmul_layernorm_matmul/test_matmul_layernorm_matmul.py` 接入 optest 框架，按 CSV 逐例参数化执行 116 例，golden 为 FP32 全流程参考，判据按 §4.1 精度标准。

**ATK 泛化（≥ 200 例）**：基于 optest 测试件生成，泛化矩阵在任务集取值域邻域扩展：

| 维度 | 泛化覆盖 |
| --- | --- |
| shape | 非 128 倍数维度（尾块路径）、小 M0（< 128）、N0 超行驻留能力（分块统计回退）、各维度取 1 / 极小值 |
| 数值分布 | 标准正态、大均值小方差（两遍法判别）、恒定行/近恒定行、全零输入、大动态范围（C0 逼近/超出 FP16 上限，检验 FP32 workspace）、`gamma=0`/`beta=0`/负 gamma |
| 异常传播 | NaN/inf 输入的行级传播边界 |
| 非法输入 | 空 tensor、乘法链不匹配、非法 dtype、非法布局——校验报错路径，不计入精度通过数 |
| 确定性 | 同输入重复执行逐位一致断言 |

**性能用例**：116 例逐例 `msprof op` 采集，与 CSV 标杆逐例对比并汇总平均比值；不同分派方案/TileShape 的对比数据一并归档。

**自验证报告**：覆盖全部功能场景，含测试用例执行日志、整体通过截图、性能数据截图，随代码仓交付件提交。

## 4.3 兼容性分析

- 新增算子：不修改 CATLASS 已有组件的对外行为，无存量兼容性影响；
- LayerNorm 行统计/归一化 AIV 组件按通用行归约形态实现，后续 RMSNorm、Matmul+LayerNorm 等融合算子可直接复用；
- 资源约束双保险：静态资源约束以 `static_assert` 固化，运行期 shape 约束由 `CanImplement` 拦截；
- 单 commit、目录树对齐参考 PR#678；PR 不携带个人敏感信息、测试报告附件、编译中间产物与临时脚本（自验报告补在 PR 描述）。

## 当前状态与后续计划

- 设计核心路径（三阶段编排、行批 LayerNorm、残差补偿双 MMAD）已在仿真环境完成功能与精度验证（标准与 full-loadA 分派分支、退化与异常 case）；split-k 与小 M 分支随代码交付全量验证；
- 全量 116 例 optest 精度、ATK ≥ 200 例泛化与性能验收在 Ascend 950PR 真机执行，数据随代码仓交付件提交；
- 代码 PR：`experimental/matmul/matmul_layernorm_matmul/` + `include/catlass/` 组件 + `tests/optest/` 接入，按任务书目录树组织。

## 参考资料

1. 任务书：`MatmulLayerNormMatmul_task_doc.md`（9 月社区任务）
2. CATLASS 创新样例开发流程指南：`docs/zh/1_Practice/10_innovative_example_development_guide.md`
3. CATLASS 参考样例：`examples/44_quant_matmul_full_loadA_tla`
4. 参考合入 PR：<https://gitcode.com/cann/catlass/pull/678>
5. 生态算子开源精度标准（实验标准）：<https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md>
6. `msprof op` 使用说明与 CATLASS 性能调试：<https://gitcode.com/cann/catlass/blob/master/docs/zh/1_Practice/evaluation/performance_tools.md>
7. optest 测试框架：<https://gitcode.com/cann/catlass/blob/master/tests/optest/README.md>
