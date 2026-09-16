# grid_cluster 算子设计文档

> **⚠️ 状态说明（请评审人先读这一段）**
> 本文件是 CANN 社区任务 2026「CANN训练营北京邮电大学-grid_cluster算子开发(950)」的**设计文档**，
> 于 2026-09-11 完成设计决策定稿。
>
> - 本文**不含本项目自有的 NPU 实测数据**：本项目**尚未在 Ascend 950 上构建或运行过本算子**，
>   因此「本项目实测结果」一节按项目红线保持为空，不以估算值填充。
> - 所有**关于 `torch_cluster` 语义的陈述都是实测**，证据为
>   `F:\Ascend\tasks\grid_cluster\probes_20260911\p1.sh … p9c.sh` 在远端
>   `openlibing-DevEnv-576875`（CPU `torch 2.7.1+cpu` + `torch_cluster 1.6.3`）上的真实输出，
>   汇总见 `BASE_FORMULA_DERIVATION_20260911.md`（1106 组逐位比对，0 反例）。
> - 所有**关于实现方案的选择都是设计决策**，逐条给出理由；每条决策的「实测 / 设计判断」分类
>   另见同目录交付的决策台账 `F:\Ascend\tasks\grid_cluster\DESIGN_DECISIONS_20260911.md`。
> - **设计阶段无未决项**：本文不保留任何开放式待办。凡涉及「必须在目标环境实测后确认」的事实，
>   都写在「验证计划与未关闭风险」一节，作为实现阶段的验证门禁（gate），而不是设计缺口。
>
> 来源任务书：`CANN训练营北京邮电大学-grid_cluster算子开发(950)/grid_cluster_task_doc.md`
> （下载自 `https://www.hiascend.com/p/resource/202609/1a0ab9213c4e400889ac5e26822794a6.zip`，
> ZIP SHA256 `60D45F037A934B33C7ECE2F28FA4B84C601541E7A01F903CA2FC532D21459083`，
> 任务书 md SHA256 `B5A1AFA2C6E2CC83543651E1897C1CDFC52B0A20AB6EC5C7B0A03EE06EA62C96`，9748 字节）
>
> 模板依据：`04_tasks/01_community-task-2026/resources/design_template.md`
> （blob SHA256 `A7B6D9E54146EB8FEB0F79D6312A79E601627C6AC6584E538745EF009EAED065`）
>
> 本设计所依据的 oracle 闭式与逐位证据：`F:\Ascend\tasks\grid_cluster\BASE_FORMULA_DERIVATION_20260911.md`；
> cluster-ID 语义裁决：`F:\Ascend\tasks\grid_cluster\CLUSTER_ID_RULING_20260911.md`。

---

# 需求背景（required）

## 需求来源

CANN 社区任务 2026 · 9 月任务 · 「CANN训练营北京邮电大学-grid_cluster算子开发(950)」。

- HiAscend taskId：`adb411543daa47518e26419b702a4afb`
- 归属账号：GitCode `2403_87496454`（HiAscend userId `10086000900693595`）
- 截止时间：HiAscend 接口返回 `2026-09-25T16:00:00Z`（= 北京时间 2026-09-26 00:00）
- 目标任务仓：`cann/ops-gnn`（实现合入）；设计文档按任务书要求以 PR 形式提交至
  `cann/cann-ops-competitions`（任务书原文写作 `cann-competitions`）的 tasklist 任务目录。

## 背景介绍

### grid_cluster 算子实现优化

在点云上叠加 D 维规则网格，把每个点映射到其所属 voxel，同 voxel 内点共享同一 cluster ID。
需实现与 `torch_cluster.grid_cluster`（https://github.com/rusty1s/pytorch_cluster ，版本建议 ≥ 1.6.0）
**接口完全一致** 的 NPU 版本。

任务书原文（`grid_cluster_task_doc.md:5-7`）：

> 实现与 `torch_cluster.grid_cluster` … **接口完全一致** 的 NPU 版本：在点云上叠加 **D 维规则网格**，同 voxel 内点共享 cluster ID。
> **验收口径**：PyTorch 层 `grid_cluster()` 为准。

### grid_cluster 算子实现现状分析

任务书给出的功能定义与参数约束（`grid_cluster_task_doc.md:11-15, 40-67`）：

$$
\text{cluster}_i = \sum_d \left\lfloor \frac{pos_{i,d} - start_d}{size_d} \right\rfloor \times \prod_{k < d} \text{num\_voxels}_k
$$

- `start`/`end` 缺省时分别为 `pos.min(0)` / `pos.max(0)`

| 参数 | 类型 | 约束 |
|------|------|------|
| `pos` | Tensor `[N, D]` | D 维坐标；支持 float16、bfloat16、float32、int8、int16、int32、uint8 |
| `size` | Float `[D]` | 每维 voxel 边长，`> 0` |
| `start` | Optional Float `[D]` | 网格起点；默认 `pos.min(0)` |
| `end` | Optional Float `[D]` | 网格终点；默认 `pos.max(0)` |

| 输出 | 类型 | 约束 |
|------|------|------|
| cluster | Long `[N]` | 线性化 voxel ID，非负 |

### grid_cluster 算子功能分析

- 输入：`pos`、`size`（可选 `start`、`end`）。
- 输出：`cluster`（int64 `[N]`）。
- 计算语义：逐元素独立 —— 每个点的 cluster ID 只依赖该点坐标、`size` 与 `start`，
  不依赖其它点，因此天然可按 N 维完全并行；**无跨点归约、无排序、无 unique**。
- **`num_voxels_k` 与 `end` 的精确语义已由实测钉死**（本节原为待澄清项，现给出结论与证据）：

  ```
  start_d = start[d] if 给定 else pos[:,d].min()          # 逐维，dtype = pos.dtype
  end_d   = end[d]   if 给定 else pos[:,d].max()          # 逐维，dtype = pos.dtype
  n_d     = trunc_toward_zero( (end_d - start_d) / size_d ) + 1     # 逐维体素数
  base_0  = 1 ;  base_d = Π_{k<d} n_k                     # cumprod 前缀（base_0 ≡ 1）
  i_{p,d} = trunc_toward_zero( (pos_{p,d} - start_d) / size_d )     # 截断，不是 floor
  id_p    = Σ_d i_{p,d} · base_d                          # int64，补码回绕
  ```

  **证据（全部为远端 oracle 实测）**：

  | 命题 | 结论 | 证据 |
  |---|---|---|
  | `n_d` 是 `trunc(span/size)+1`，**不是** `ceil(span/size)` | 成立，16/16 命中 | `BASE_FORMULA_DERIVATION_20260911.md:155-182`（D2.2，`X=3.0` 是 `ceil` 的关键否决票；`X=2.5/2.4999/2.5001` 三连同值排除任何「先四舍五入/先取整跨度」写法） |
  | `end` **参与** `n_d`（即参与 base），**不是**死参数 | 成立 | 同文件 `:184-213`（D2.3：同一组 5 点、`size=1`、`start=None`，`end=None→[0,0,364,455,728]`；`end=2→[0,0,28,35,56]`；`end=10→[0,0,444,555,888]`） |
  | base **也不是** `max_i i_d + 1` | 成立 | 同文件 `:212`（`end=2` 时观测最大下标为 8、若取观测值则 ID 应不变，实测 ID 变了） |
  | base 是**逐维前缀积**，不是行主序 | 成立 | 同文件 `:215-229`（D2.4；3-D 例 `n=(5,2,2)⇒base=[1,5,10]`，行主序 `[81,9,1]` 被否决） |
  | 除法必须在**输入 dtype 精度内**做（f32 输入即 f32 运算），不是提升到 double | 成立，6/6 判给 f32；随机套件 f32 模型 0 失配、f64 模型 22 失配 | 同文件 `:113-139`（D1.7） |
  | 取整是**向零截断**，不是 floor | 成立 | 同文件 `:57-67`（D1.1：`pos=[-1.5,0,1.5,-0.5,0.5,-2]`、`start=0`、`size=1 → [-1,0,1,0,0,-2]`；floor 应为 `[-2,0,1,-1,0,-2]`） |
  | 空输入 `N=0`：oracle **抛异常**，不返回空张量 | 成立，42 组 N=0 实例全部抛同一异常 | 同文件 `:97-107`（D1.5） |

- **聚簇语义的完整含义**：`cluster` 是**非负的稀疏线性化 voxel ID**（`base_d = Π_{k<d} n_k` 为基数的
  按维线性化），**不是**重编号为 `0..K-1` 的簇序号。这一点由 oracle 直接实测确认
  （`CLUSTER_ID_RULING_20260911.md:26-35`：5 点输入输出 `[0, 0, 364, 455, 728]`，
  `unique = [0,364,455,728]`，「重编号为连续序号」判为 `False`），并与任务书 `:51` 完全一致。

---

# 需求分析（required）

## 需求描述

使用 Ascend C 编程语言实现 `grid_cluster` 算子，接口与 `torch_cluster.grid_cluster` 一致，
在 Ascend 950 系列上完成 NPU 实现（不得回退 CPU），支持任务书规定的 7 种输入 dtype，
输出 int64 cluster，满足 bit-wise 一致精度要求与 ≥ 0.45× 标杆性能要求。

## 需求拆解

1. PyTorch 层接口 `grid_cluster(pos, size, start=None, end=None) -> Tensor`，底层调用
   `torch.ops.torch_cluster.grid(pos, size, start, end)`（任务书 `:30-38`）。
2. 数据类型分级（任务书 `:53-59`）：

| 级别 | dtype | 路径 | 性能考核 |
|------|-------|------|----------|
| L1 | float16/float32 | NPU；参与功能与 NPU 性能验收 | 是 |
| L2 | bfloat16/int8/uint8 | NPU；仅做功能验收 | 否 |
| L2 | int16/int32 | NPU；参与功能与性能测试 | 是 |

3. 算子约束（任务书 `:61-66`）：
   1. `size.numel() == pos.size(1)`；
   2. 除法与取整语义与 CPU 一致（`true_divide` 后转 long）；
   3. 空 `pos` 返回空 LongTensor；
   4. 确定性算子，须 bit-wise 一致（整数 cluster ID）。
4. 功能验收用例：TC-01 … TC-09（任务书 `:68-80`），参考 `test/test_grid.py`。
5. 性能要求：所有用例性能 ≥ 0.45× 标杆（任务书 `:84`）。
6. 精度要求：cluster ID（int64）与 CPU 标杆 **bit-wise 一致**（任务书 `:148-157`）。

## 设计输入（三份前置证据）

| 文件 | 性质 | 本设计如何使用 |
|---|---|---|
| `BASE_FORMULA_DERIVATION_20260911.md` | 远端 oracle **只读实测 + 闭式反推**（1106 组逐位一致、0 反例） | 作为「输出定义」的唯一规范来源；实现按该闭式逐 op 复刻 |
| `CLUSTER_ID_RULING_20260911.md` | 语义**裁决**（含两次自我更正） | 裁定验收口径为 bit-wise 一致、输出为线性化 voxel ID |
| `probes_20260911/p1.sh … p9c.sh` | 上述两份文件的可复现脚本 | 评审人可原样重跑核对 |

**本设计的边界声明**：以上证据全部是 **CPU oracle 行为**，不是 NPU 行为。
NPU 侧是否复刻得到，属于实现阶段的验证门禁，见「验证计划与未关闭风险」。

---

# 详细设计（required）

## 算子分析

### 数学公式

见上文「背景介绍」中的线性化 voxel ID 公式（任务书 `:11-15`）及其实测修正闭式
（`BASE_FORMULA_DERIVATION_20260911.md` D4.1）。

### 支持数据类型

- 输入 `pos`：float16、bfloat16、float32、int8、int16、int32、uint8（任务书 `:44`）。
- `size`：任务书 `:45` 声明为 `Float [D]`；本设计同时接受与 `pos` 同 dtype 的张量，
  以兼容官方基准脚本 `benchmark_grid_cluster.py:89`（`size = torch.full((dim,), v, dtype=dtype)`，
  整数用例下 `size` 是 int32/int16 张量）。
- 输出：int64。

### 支持形状

- `pos`：`[N, D]`，`D = size.numel()`；支持 `N = 0`（空张量，见「空 `pos` 早退路径」）。
- `D` 的取值上界、`N` 的规模上界**任务书未规定**。
  本设计取：**N 上界只受 int64 索引与 GM 容量限制；D 的分支覆盖由下述两条计算路径共同保证**——
  - D ≤ 63：走「行内多体素 packed-lane 归约」路径，与 D 的具体值无关（该路径按列块循环，D 任意）；
  - D ≥ 64：走「列块循环 + 行分块归约」路径，同样与 D 的具体值无关。
  - 依据：官方性能矩阵覆盖 `D ∈ {3, 16, 32, 64}`（任务书 `:89-144`），
    官方功能用例覆盖 `D ∈ {1, 2, 3, 64}`（`test_grid.py:14-20` 的 `TEST_SHAPES`/`GENERAL_SHAPES`，
    其中 `(256, 64)`、`(8, 1)` 给出了 D=64 与 D=1 两个端点）。
    本设计按「任意 D ≥ 1」设计，不设 D 上界。

## 算子实现

### 实现方案

本节给出**已定稿的实现方案**。所有选择均为设计决策；每条决策的理由与「实测 / 判断」分类
见 `DESIGN_DECISIONS_20260911.md`。凡任务书未规定的点，均按
「任务书未规定；本设计取 X，理由是 Y」的形式显式记录。

#### 0. 方案总览与数据流

```
Python 层  ops_gnn.grid_cluster(pos, size, start=None, end=None)
   │  (1) 只做「None 直通」与最小校验；start/end 的**缺省推导不在这里做**（见 §15）
   ▼
Host 层  grid_cluster_npu(pos, size, start, end, stream)
   │  (2) 解析 dtype → kind / TilingKey；（3）算 n_d 与 base_d（前缀积）；
   │  (4) 定行宽形态（D ≤ 64 单组 / D > 64 多组）与 dtype 模板实例；
   │  (5) 查平台（GetCoreNumAiv / GetCoreMemSize(UB)）；(6) 分核；(7) 算 tile；(8) 启动
   ▼
Kernel（arch35 / AIV）
   Init：绑定 GM、解 tiling、InitBuffer、预生成 base/start/size 的 UB 常量与掩码
   Process：for tile ∈ [begin, end)  { CopyIn → Compute → CopyOut }，TQue 双缓冲
   ▼
GM 输出 cluster（int64 [N]，线性化 voxel ID，可为负，补码回绕，不做重编号/裁剪）
```

**核心式（本设计逐 op 复刻 oracle 的运算顺序，不做任何代数改写）**：

```
shift_{p,d} = pos_{p,d} - start_d                 # 在 promote(pos.dtype, start.dtype) 内
i_{p,d}     = trunc_to_int32( shift_{p,d} / size_d )      # 除法在 promote(shift,size) 内；
                                                          # 整型÷整型 ⇒ 提升到 f32（见 §4）
id_p        = Σ_d i_{p,d} · base_d                # int64 域，逐元素乘 + 行内归约
```

> **禁止改写（bit-wise 硬约束）**：`(pos-start)/size` 不得改写为 `*(1/size)`，
> 不得改写为 `pos/size - start/size`，不得用 FMA 合并，累加不得在浮点域做。
> 依据：`BASE_FORMULA_DERIVATION_20260911.md:526-528`（D4.2）——任何代数等价改写都会改变最后一次舍入。
> 冗余除法按 bit-wise 硬约束优先，不追求指令数最优（`torch.true_divide(a,b)` 是 `a/b`，
> 不是 `a*recip(b)`）。

#### 1. 分核策略

**决策**：**按 N 一维均分，每核一段连续的整点区间**；切分粒度取 tile，不做 (N, D) 二维切分。

- 核数：`cores = GetCoreNumAiv()`；`cores == 0` 时规范化为 1（防御，与仓内既有先例一致）。
- 每核区间：`per = ceil(N / cores)`；`begin = min(per·blockIdx, N)`，`end = min(per·(blockIdx+1), N)`。
  即**整块均分 + 尾块留给最后一个核**，不设大小核区分。
- 若 `ceil(N / cores) < 64`，则把 `launchBlocks` 收缩为 `max(1, ceil(N / 64))`，
  避免为极短输入启动空核、也避免退化 tile。
- **理由**：
  1. 本算子无跨点依赖、无跨核通信，按 N 均分即可达到满载，无需二维切分；
  2. 每个点的输出是**连续 int64**，按 N 切分使每核的 CopyOut 成为一段连续 DMA，写合并最优；
  3. 每核只读自己区间内的 `pos` 行，读也是连续的（在「按行搬运」路径下）；
  4. `per` 用 `ceil` 上取整 + `min(...)` 收尾，是仓内既有实现（`segment_csr_vec_kernel.cpp:268-269`）
     同一模式，评审人可对照。
- **依据（平台参数实测）**：Ascend 950PR 实测 `AIV/AIC = 56/28`、`UB = 253952 B (248 KiB)`，
  查询 API 为 `GetCoreNumAiv()` / `GetCoreMemSize(CoreMemType::UB)`
  （`tasks/nearest/delivery/reports/environment.md:8-9,22`；`tasks/segment_csr/CSR_GAP_DECISION_20260911.md:363`
  独立确认 `GetCoreNumAiv() = 56`）。**核数与 UB 必须运行时查询，不得硬编码** ——
  这样 D=3 时 N=32M 的单核负载与 D=64 时相同，核数只随平台变化。

#### 2. UB 切分与 Buffer 规划

**决策**：单核内以 `tilePoints`（点数）为 tile 级 UB 切分粒度，**2 个 TQue（tile 级）+ 4 块 TBuf
（其中中间量按 64-lane 组复用）**；`pos` 读入双缓冲，输出与中间量单份。

§7 的「Host 统一规整行 pitch」在 UB 侧的落点就是上面的 `Dp`，两种取值：
`Dp = 64 · ceil(D/64)`（必为 64 的整数倍，见 §7 与 §10）。
`rb = Dp × sizeof(T)`，`coefQue` 单份字节数 = `Dp·sizeof(T) + Dp·4 + Dp·8 + 256`。
UB 账本（`ubBytes` = 平台查询值，950PR 实测 253952 B = 248 KiB；`T` = `pos` 的 dtype）：

| # | 名称 | 类型 | 字节数 | 生命周期 |
|---|---|---|---|---|
| 1 | `inQue` | TQue，depth **2** | `2 × tilePoints × rb` | tile 级（整张 tile） |
| 2 | `coefQue` | TQue，depth **2** | `2 × (Dp·sizeof(T) + Dp·4 + Dp·8 + 256)` | tile 级（`size`/`start`/`base` 三向量） |
| 3 | `midBuf` | TBuf | `64 × (4 + 8) = 768 B`（对齐取 1 KiB） | **64-lane 组级**：`i_d`(int32) 与 `i_d·base_d`(int64) 复用同一块 |
| 4 | `redBuf` | TBuf | `tilePoints · groupsPerRow × 8` | tile 级（部分和 / 精确 int64 暂存） |
| 5 | `maskBuf` | TBuf | 256 B | 常量（D ≤ 63 时为编译期常量） |

**关键设计点：`i_d` 与 `i_d·base_d` 不按整张 tile 常驻 UB，而是按 64-lane 组复用**
（`768 B`，与 `tilePoints`、`D` 均无关）。若让它们对整张 tile 常驻，
`D ≤ 64`/f32/`tilePoints=256` 时单是 int64 中间量就是 `256×64×8 = 128 KiB`，UB 立刻见底。
按组推进同时让「行宽 > 64 的列循环」与「行内 64-lane 组循环」成为**同一个循环**。

**字节账本（实测 `ubBytes = 253952 B`）**：

| 场景 | `rb` | `Dp` | `groupsPerRow` | `tilePoints` | ①`inQue` | ②`coefQue` | ③`midBuf` | ④`redBuf` | ⑤`maskBuf` | 小计 | +⑥保留 | **合计** | 余量 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| f32, D=3 | 256 B | 64 | 1 | 256 | 128 KiB | 2.5 KiB | 1 KiB | 2 KiB | 0.25 KiB | 133.75 KiB | 32 KiB | **165.75 KiB** | 82 KiB |
| f16, D=3 | 128 B | 64 | 1 | 512 | 128 KiB | 2.5 KiB | 1 KiB | 4 KiB | 0.25 KiB | 135.75 KiB | 32 KiB | **167.75 KiB** | 80 KiB |
| f32, D=16 | 256 B | 64 | 1 | 256 | 128 KiB | 2.5 KiB | 1 KiB | 2 KiB | 0.25 KiB | 133.75 KiB | 32 KiB | **165.75 KiB** | 82 KiB |
| i16, D=16 | 128 B | 64 | 1 | 512 | 128 KiB | 2.5 KiB | 1 KiB | 4 KiB | 0.25 KiB | 135.75 KiB | 32 KiB | **167.75 KiB** | 80 KiB |
| f32, D=32 | 256 B | 64 | 1 | 256 | 128 KiB | 2.5 KiB | 1 KiB | 2 KiB | 0.25 KiB | 133.75 KiB | 32 KiB | **165.75 KiB** | 82 KiB |
| f32, D=64 | 256 B | 64 | 1 | 256 | 128 KiB | 2.5 KiB | 1 KiB | 2 KiB | 0.25 KiB | 133.75 KiB | 32 KiB | **165.75 KiB** | 82 KiB |
| i16, D=64 | 128 B | 64 | 1 | 512 | 128 KiB | 2.5 KiB | 1 KiB | 4 KiB | 0.25 KiB | 135.75 KiB | 32 KiB | **167.75 KiB** | 80 KiB |
| f32, D=128 | 512 B | 128 | 2 | 128 | 128 KiB | 4.5 KiB | 1 KiB | 2 KiB | 0.25 KiB | 135.75 KiB | 32 KiB | **167.75 KiB** | 80 KiB |

> 「⑥保留」= ①..⑤ 之外的 32 KiB 安全余量（§2 取值规则里的 `reserved`），
> 它同时充当 UB 对齐（TBuf 起始地址按 32 B 对齐）与 TBuf 合并后的余量。

> 注意 `rb` 取决于 `Dp` 而**不与 D 成正比**：`D ≤ 64` 时 `Dp ≡ 64`，因此
> D=3/16/32/64 的 `rb` 完全相同（f32 下均为 256 B）。这使 D≤64 的全部场景共用一套 tile 参数，
> kernel 侧只需 `D ≤ 64` / `D > 64` 两个实例化形态；`groupsPerRow` 只在 `D > 64` 时变化。

`tilePoints` 的取值规则（Host 侧一次算定，写进 kernel 参数）：

```
1. 需要同时容纳 2 份输入 tile（inQue depth2）+ 1 份 redBuf + 固定开销：
      perRow     = 2·rb + 8·groupsPerRow          # inQue 两份 + redBuf 每行的部分和
      reserved   = 32 KiB                          # coefQue + midBuf + maskBuf + 安全余量
      tilePoints = 2^floor( log2( (ubBytes − reserved) / perRow ) )
2. cap 到 [64, 4096]；再 cap 到不超过「每核点数」与「单张 GM 连续块 ≤ 4 GiB / rb」。
3. 若 tilePoints < 64（即 rb 过大，约 D > 430 的 f32）⇒ 直接取 tilePoints = 64；
   若连 64 点都放不下 ⇒ Host 报错（能力不足），不启动半成品 kernel。
```

以实测 `ubBytes = 253952` 代入（`ubBytes − 32 KiB = 226304 B`）：

| 场景 | `T` | `D` | `Dp` | `rb` | `perRow` | 未取整商 | `tilePoints` | 单核 tile 数 @N=32M |
|---|---|---|---|---|---|---|---|---|
| 性能矩阵 D=3 | f32 | 3 | 64 | 256 B | 520 B | 435 | **256** | 31250 |
| 性能矩阵 D=3 | f16 | 3 | 64 | 128 B | 264 B | 857 | **512** | 15625 |
| 性能矩阵 D=16 | f32 | 16 | 64 | 256 B | 520 B | 435 | **256** | 31250 |
| 性能矩阵 D=32 | f32 | 32 | 64 | 256 B | 520 B | 435 | **256** | 31250 |
| 性能矩阵 D=64 | f32 | 64 | 64 | 256 B | 520 B | 435 | **256** | 31250 |
| 性能矩阵 D=64 | i16 | 64 | 64 | 128 B | 264 B | 857 | **512** | 15625 |
| 功能用例 D=1 | f32 | 1 | 64 | 256 B | 520 B | 435 | **256** | — |
| 功能用例 D=2 | f32 | 2 | 64 | 256 B | 520 B | 435 | **256** | — |

**这张表本身就是一条应当写明的结论**：在 `Dp ≡ 64` 的统一布局下，
`D ≤ 64` 的全部 dtype/维度组合**收敛到同一套 tile 参数**（f32 全部 `tilePoints=256`，
f16/i16 全部 `512`）。因此 kernel 侧只需 `D ≤ 64` / `D > 64` 两个实例化形态，
而不是「每种 D 一套参数」。这是选择 `Dp = 64·ceil(D/64)` 而非 `Dp = D` 的主要收益。

**理由与取舍**：
- **为什么双缓冲**：本算子 MTE2（读 `pos`）与 V（算术）时间量级相当，单缓冲会串行化
  「搬运→计算→搬运」，直接损失约 2× 吞吐；双缓冲是达标的必要条件。
- **为什么 `redBuf` 只给 1 份**：每点的 CopyOut 量固定为 8 B，而每点的读入量是
  `D·sizeof(T)`——f32/D=3 时 12 B（写:读 = 2:3，写不是瓶颈），f32/D=64 时 256 B（写:读 = 1:32）；
  单份 + `V→MTE3` 事件即可让写回被下一张 tile 的读入与计算掩盖；
  给输出做双缓冲的收益低于其 UB 成本。
- **为什么 `tilePoints` 取 2 的幂**：使 UB 地址与 32 B block、64-lane repeat 同时整除，
  省掉所有运行期对齐分支；代价是 UB 利用率略低（可接受）。
- **为什么保留 32 KiB**：`coefQue` 在 D=64 时单份为 `256+256+512+256 = 1280 B`、depth2 = 2.5 KiB，
  `midBuf` 1 KiB、`maskBuf` 256 B，其余约 28 KiB 为安全余量与 TBuf 合并后的余量。
  该保留量是**保守设计判断**，不是实测最优值；实现阶段用 UB 账本断言
  （见验证计划 V-7）证明 `实际占用 ≤ ubBytes`，并可在实测后收紧。
- **为什么中间量用 TBuf 而不是 TQue**：`midBuf` 是核内 `V→V` 数据、不经 DMA，
  用 TBuf + `PipeBarrier<PIPE_V>` 比 TQue 更省事件资源；
  只有真正的跨 pipe 边界（MTE2→V、V→MTE3）才用 `SetFlag/WaitFlag`。
- **TBuf 合并风险**：`midBuf`/`redBuf` 生命周期不重叠的区段可能被编译器合并，
  仓内已有先例被评审标注过 TBuf 合并风险（`four_tasks_review_summary.md` 中
  「TBuf compiler merge」类条目）。本设计在验证计划 V-7 中要求「合并后仍满足容量断言」。

#### 3. TilingKey 与模板实例化规划

**决策**：沿用 ops-gnn 仓的实际调用链——**Host 侧按 dtype 模板分发 + `<<<>>>` 直接启动**
（仓内既有形态：`tasks/nearest/.../nearest_kernel.cpp:156-187`、`nearest_vec_kernel.cpp:491`），
同时把 `tilingKey` 作为 `TilingData` 的一个显式字段写入，用于：
(a) profiler / 日志可读性；(b) 实现阶段按 key 逐项点检「7 dtype × 2 行宽形态」的覆盖；
(c) 自测报告里机械核对「每种 dtype 都跑过」。

`TilingKey` 编码（`kind = tilingKey & 0xFF`；`bit8 = wide`，即 `D > 64`）：

| kind | dtype | 除法域 | 说明 |
|---|---|---|---|
| 0 | float32 | f32 | L1，性能考核 |
| 1 | float16 | f32 算 + 每步舍回 f16 | L1，性能考核 |
| 2 | bfloat16 | f32 算 + 每步舍回 bf16 | L2，仅功能 |
| 3 | int8 | 提升 f32 除 | L2，仅功能 |
| 4 | uint8 | 提升 f32 除 | L2，仅功能 |
| 5 | int16 | 提升 f32 除 | L2，性能考核 |
| 6 | int32 | 提升 f32 除 | L2，性能考核 |

**实例化矩阵（7 dtype × {`D ≤ 64`, `D > 64`}）＝ 14 个实例**，由一个宏表显式实例化，
不依赖运行期 key 分发。

**理由**：
1. **仓内没有运行期 tilingkey 分发设施**：`development_guide.md:288` 描述的实际机制是
   「`GetCoreNumAiv()` 取核数 → `<<<>>>` 启动 → 显式模板实例化」。
   引入一套与仓内不一致的 tilingkey 运行期分发会显著增加评审与编译风险。
2. **dtype 必须在编译期确定**：7 种 dtype 的算术语义不同（见 §4、§8），
   运行期分支会把 7 条路径的指令都编进同一个 kernel，既损性能又损可读性。
3. 之所以只需要 2 个行宽形态（而不是每个 D 一个）：§2 已证明 `D ≤ 64` 时
   `Dp ≡ 64`、tile 参数在全部 dtype 上收敛，`groupsPerRow` 只在 `D > 64` 时变化。
4. `tilingKey` 字段仍保留，使「7 种 dtype 是否都被覆盖」可以在自测报告中机械核对。

#### 4. dtype 分支与逐 op 提升模型

**决策**：以**三个算术域**覆盖 7 种 dtype，提升规则**逐 op** 复刻实测模型
（`BASE_FORMULA_DERIVATION_20260911.md` D4.3，证据 `p6.sh`/`p7.sh`/`p9c.sh`）：

| 类别 | dtype | `shift = pos − start` 的域 | `div` 的域 | 截断域 |
|---|---|---|---|---|
| 宽浮点 | float32 | f32 | f32 | f32 → int32 |
| 窄浮点 | float16、bfloat16 | **f32 中算，结果 CAST_RINT 舍回窄 dtype** | f32 中算，舍回窄 dtype | 窄 dtype → int32 |
| 整型 | int8、uint8、int16、int32 | **在 promote(pos,start) 的整型内做模 2^位宽回绕** | 整型⇒提升到 **f32**（`torch.true_divide` 整型⇒`get_default_dtype()`） | f32 → int32 |

- **窄浮点为什么「每步舍回」**：实测三模型判别（`p5.sh` C3，`BASE_FORMULA_DERIVATION:407-422`）
  显示 `f16/bf16` 的正确模型是 **M3＝每一步都降回该 dtype**，而
  「全 f32 算」（M1）与「只把输入降到窄 dtype」（M2）在 `bf16 + 0.1 步长`、`f16 + size 0.7`
  等判别样本上**失配**。因此 NPU 上必须显式复刻双重舍入：
  `cvt_f32(shift) → cvt_f16(·) → cvt_f32(·) → div_f32 → cvt_f16(·) → cvt_f32(·) → CAST_TRUNC → int32`。
- **整型为什么必须在位宽内回绕**：实测 `p4.sh` S4（`:424-430`）——
  `int8` 输入 `pos=[-128,-1,0,127]`、`size=1` 输出 `[0,127,-128,-1]`（`128→-128`、`255→-1` 回绕），
  `uint8` 输出 `[128,255,0,127]`，而 `int16` **不回绕**（`[0,127,128,255]`）。
  实现做法：减法在 int32 里做，再用 `cvt_int32→int8→int32`（或标准位运算）还原到源位宽，
  保证回绕语义与 oracle 逐位相同。
- **整型除法的域**：`p8.sh` Q3（`:432-438`）用 `2^24+1` 判别证明 `int/int` 的 `true_divide`
  提升到**进程级 `get_default_dtype()`**（默认 float32；f32 默认→`16777216`，f64 默认→`16777217`）。
  本设计取 **f32**，理由是验收 harness 未显式设置 default dtype 时 PyTorch 默认即 float32；
  该假设已登记为待确认项（见验证计划 V-6）。
- **`size` 为整数张量**：`p7.sh` 的 54 组 `pos dtype × size dtype` 全部 MATCH（`:440`），
  按上表逐 op 提升即可覆盖；整数 `size` 与整数 `pos` 的组合落到「提升 f32 除」那一行。

#### 5. TilingData 字段定义

**决策**：不使用 GE 注册式 TilingData，采用**结构体直传 kernel 参数**（仓内既有形态）。
字段与来源如下（`kMaxDim = 512` 为设计常量，见理由 3）：

| 字段 | 类型 | 来源 / 算法 |
|---|---|---|
| `nPoints` | uint64 | `pos.size(0)` |
| `dim` | uint32 | `D = pos.size(1)`（`pos` 为 1-D 时视作 `D=1`） |
| `pitchElems` | uint32 | `Dp = 64 · ceil(D/64)`，UB 侧行 pitch（元素数）。取值只有 64、128、192、256…（必为 64 的整数倍） |
| `tilePoints` | uint32 | §2 的 `tilePoints` |
| `groupsPerRow` | uint32 | `Dp / 64`，每行占几个 64-lane 组（`D ≤ 64` 时恒为 1） |
| `usedCores` | uint32 | 实际启动核数（已按「每核至少 64 点」收缩） |
| `pointsPerCore` | uint64 | `ceil(nPoints / usedCores)` |
| `kind` | uint8 | §3 的 dtype kind |
| `wide` | uint8 | 0 = `D ≤ 64`（每组即一行，归约结果直接可写）；1 = `D > 64`（需按行跨组再求和） |
| `tailPoints` | uint32 | 最后一张 tile 的有效点数（便于内核零分支收尾） |
| **`numVoxels[kMaxDim]`** | int64 | **Host 预计算的逐维体素数 `n_d`**（§6） |
| **`base[kMaxDim]`** | int64 | **Host 预计算的线性化基数 `base_d`**（§6，`base[0]=1`） |
| **`rowWeight[kMaxDim]`** | int64 | **Host 预计算的行权重**：`rowWeight[d] = max(1, base[d]·n_d)`。用于判定 `i_d·base_d` 是否可能超出 int32 域（§11） |
| `startRaw` / `sizeRaw` | by-value `[kMaxDim]` | 仅 `D ≤ 8` 时的快速通道；否则由 GM 读取 |
| `hasEnd` | uint8 | 仅用于日志与点检（数学上 `end` 已折叠进 `base`） |
| `tilingKey` | uint32 | §3 |

**理由**：
1. `n_d` 与 `base_d` 在 host 上各算一次 O(D)，kernel 每 tile 只需读 D 个常量，
   避免每点重算前缀积 —— 这是「num_voxels / 前缀积在 host 侧预计算」的具体落点。
2. `base` 由 `cumprod` 生成，`base[0] ≡ 1`（实测 `BASE_FORMULA_DERIVATION:150-153`），
   host 侧严格按 `cat([1], n.cumprod(0))[:D]` 生成，**不得**改成从 `n` 直接累乘（差一位）。
3. `kMaxDim = 512`：任务书与官方用例给出 `D ∈ {1,2,3,16,32,64}`，
   本设计取 `512` 作为元数据数组的静态上界（`512 × 3 × 8 B = 12 KiB` 的 host 栈/参数开销），
   是**保守设计判断**；若 `D > 512`，Host 走「`n`/`base` 也放 GM scratch」的备用分支，
   不因此报错（分支已列入实现范围）。

#### 6. `num_voxels` / 前缀积的 Host 侧预计算

**决策**：`n_d` 与 `base_d` **全部在 Host 侧按 oracle 的 op 顺序**预计算，用与输入同精度的
张量运算复刻（对应闭式见「背景介绍」）：

```
// （语义与顺序逐字对应 BASE_FORMULA_DERIVATION D4.1 的 Host 侧参考实现）
shift = P − start                                  # 域 = promote(pos.dtype, start.dtype)
n     = ((end − start).true_divide(size)).to(torch.int64) + 1        # 向零截断 + 1
base  = torch.cat([torch.ones(1, dtype=torch.int64), n.cumprod(0)])[:D]
idx   = shift.true_divide(size.reshape(1, -1)).to(torch.int64)       # 向零截断
out   = (idx * base.reshape(1, -1)).sum(1)                           # int64
```

- 上述代码是**参考实现**（也直接用作本项目自测的 oracle 对拍入口）；
  NPU 实现只把它拆成「host 算 `n`/`base`」+「kernel 算 `idx` 并加权求和」两段，
  **运算顺序不变**（这是 bit-wise 的前提）。
- 三个关键点，都必须与参考实现一致：
  1. `+1` 在**截断之后**加（`trunc(span/size)+1`，不是 `floor`、不是 `ceil`）；
  2. `base` 是 **`[1] ++ cumprod(n)[:D-1]`**，即 `base[0] = 1`；
  3. `n`、`base` 全程 **int64**，溢出按二进制补码回绕（实测 `BASE_FORMULA_DERIVATION:241-252`
     给出 `exact 27000027000009000000 → int64 8553282926299448384` 的回绕实例）。

#### 7. 搬运 API 选型与 D=3/16/32/64 的对齐处理

**决策**：**GM↔UB 统一使用 `DataCopyPad` + `DataCopyExtParams`**，不使用 `DataCopy`；
UB 侧行 pitch 由 **Host 统一规整为 `Dp = 64 · ceil(D/64)`**，kernel 内不出现运行期对齐分支。

| `D` | dtype | 真实行宽 `D·sizeof(T)` | Host 规整 | 每次 DMA 的 `blockLen` | `srcStride` | UB 行占（lane 数） |
|---|---|---|---|---|---|---|
| 3 | f32 | 12 B | `Dp=64`，3 有效 lane + 61 恒零 lane | 12 B | 12 B | 64 |
| 3 | f16 | 6 B | `Dp=64`，同上 | 6 B | 6 B | 64 |
| 3 | i16/i32 | 6 B / 12 B | `Dp=64`，同上 | 6 B / 12 B | 同左 | 64 |
| 16 | f32 | 64 B | `Dp=64`（真实数据正好 64 B） | 64 B | 64 B | 64 |
| 16 | i16 | 32 B | `Dp=64`，32 有效 lane + 32 恒零 | 32 B | 32 B | 64 |
| 32 | f32 | 128 B | `Dp=64`，分 2 个 64-lane 组 | 128 B | 128 B | 64×2 |
| 32 | i32 | 128 B | 同上 | 128 B | 128 B | 64×2 |
| 64 | f32 | 256 B | `Dp=64`，分 4 个 64-lane 组 | 256 B | 256 B | 64×4 |
| 64 | i16 | 128 B | `Dp=64`，分 2 组 | 128 B | 128 B | 64×2 |
| 64 | i32 | 256 B | `Dp=64`，分 4 组 | 256 B | 256 B | 64×4 |

> `blockLen` 是**字节数**，不是元素数（仓内 `DataCopyPad` 调用点均按 `sizeof(T)` 构造，
> 见 `03_remediation/review_20260908/api_prestudy.md:88-95`）。`srcStride` 为 0 表示
> 源侧行间无间隔（`pos` 的 D 维是连续的），**规整只发生在 UB 侧**——
> 这一点很重要：Host 不需要为对齐复制任何数据，避免了额外的一次全量 GM 往返。

- **为什么统一用 `DataCopyPad`**：`DataCopy` 要求 `blockLen` 为 32 B 整数倍，
  列宽 12 B（D=3, f32）、6 B（D=3, f16/i16）、32 B（D=16, i16 恰为边界）里前两者不满足；
  用 `DataCopy` 就必须在 host 侧预先把 `pos` 复制成行对齐的副本，多一次全量读写
  —— 这与 §「标杆量下界推算」给出的「不要引入多余 GM 往返」直接冲突。
  `DataCopyPad` 支持任意字节 `blockLen` + 显式 `srcStride`，一次覆盖全部 D；
  UB 侧尾部 padding 由硬件产生，**不产生对 GM 的额外读流量**。
  仓内先例：`segment_csr_vec_kernel.cpp:295-297,362-365,392-398` 与
  `nearest_vec_kernel.cpp:415-418` 均为 `DataCopyPad` 形态；`01_state/pr_1402/design_current.md:143`
  亦记「GM 与 UB 之间统一使用 `DataCopyPad`」。
- **为什么 `Dp` 一律取 64 的整数倍**：使「一行 = 整数个 64-lane 组」恒成立，
  从而行归约恒为「每组的组内归约」，**不需要任何跨 repeat 的地址重排或 gather**。
  `D ≤ 64` 时一行恰好 1 组；`D > 64` 时一行 `Dp/64` 组，按 §11 做跨组求和。
- **为什么不做「按 D 压缩 lane」的变体**（例如让 4 行共享一个 64-lane 组的 16-lane 子段）：
  那样归约要落在 16-lane 子段边界上，需要额外的 `Duplicate`/`Select` 位移处理，
  而收益只是省下 `inQue` 的 UB —— 本设计在 D=16 时 `inQue` 占用 32 KiB（见 §2 账本），
  UB 并不紧张。**任务书未规定 layout**；本设计取「一行一组」这个能一句话讲清楚的布局。
- **CopyOut 用精确字节 `DataCopyPad`**：`cluster` 是 `[N]` int64，连续；
  尾块只写 `validPoints × 8` 字节，避免越界写。
- **不选 `DataCopy` 做快路径**：对 D=16/32/64 用 `DataCopy` 确实等价于 `DataCopyPad`（都是整块对齐），
  但保留两条路径会让「对齐处理」分裂成两套代码。**任务书未规定必须用哪个 API**；
  本设计取单一 API，理由是**可验证性优先**（一套参数表、一套边界用例）。
  性能上的差异由实测裁决（验证计划 V-5：若微基准显示 `DataCopy` 快路径收益 > 5%，
  再在整块对齐场景增设快路径，并把该变更作为独立 commit 重新取证）。

#### 8. 取整语义（显式裁决：向零截断，不是 floor）

**决策**：**全程使用向零截断（truncate toward zero）**，实现为
**`CAST_TRUNC` 舍入模式的窄化 Cast（f32→int32）**，不使用 floor、不使用 round、不使用 saturate。

- **任务书自身存在冲突**：`:12` 写 $\lfloor \cdot \rfloor$（floor），`:64` 写「除法与取整语义与 CPU 一致
  （`true_divide` 后转 long）」（向零截断）。二者仅在 `start ≤ min(pos)` 时等价。
- **裁决依据（实测）**：`BASE_FORMULA_DERIVATION_20260911.md:57-67`（D1.1）——
  1-D、`start=0`、`size=1`、`pos=[-1.5, 0.0, 1.5, -0.5, 0.5, -2.0]`，oracle 输出 **`[-1, 0, 1, 0, 0, -2]`**；
  floor 应为 `[-2, 0, 1, -1, 0, -2]`。逐位一致 ⇒ **oracle 用 trunc**。
  同一结论与 `:64` 相符，与 `:12` 的 floor 记号不符。
- **负坐标行为（必须显式定义）**：
  1. `pos_{p,d} − start_d < 0` 时 `i_{p,d}` **为负**，`id_p` 也可为负；
     实测复现：`p3.sh` D3b 的 `start covers data (negative ids) → MATCH n=[-15,-15]`
     （`BASE_FORMULA_DERIVATION:323`）。
  2. **不做任何非负性裁剪、不做饱和、不做范围检查**。任务书 `:51` 写的「非负」
     只对默认参数（`start = pos.min(0)`）成立；显式给出 `start > min(pos)` 时 ID 可为负。
  3. 负零、`-0.0` 与 `0.0` 截断后同为 `0`，无需特判。
  4. `end < start` 时 `n_d ≤ 0` 合法（实测 `end<start → MATCH n=[0,0]`、
     `end==start → MATCH n=[1,1]`，`:317-318`），此时 `base` 可为 0 或负，
     **不同体素可能碰撞成同一 ID** —— 这是 oracle 的行为，本实现照实复刻，不做「去碰撞」。
- **实现落点**：仓内已有可复用的截断 CastTrait 先例
  （`segment_csr_vec_kernel.cpp:31-32`：`RegLayout::ZERO, SatMode::NO_SAT, MaskMergeMode::ZEROING,
  RoundMode::CAST_TRUNC`，注释即「trunc」），本设计沿用同一 trait；
  `SatMode::NO_SAT` 是必需的（饱和会把大值夹到 `INT32_MAX`，与 oracle 的 C 语义不同）。

#### 9. float16 / bfloat16 精度风险与硬约束

**决策**：**f16/bf16 一律「在 f32 中算、每步舍回本 dtype」，并禁止一切重结合**
（不用 FMA、不用倒数近似、不改写除法、不用 `*(1/size)`）。**不采用**「NPU 直接做 f16 除法」。

- **风险来源（实测）**：CPU 侧窄浮点模型是 M3＝每步降回本 dtype（`p5.sh` C3，
  `BASE_FORMULA_DERIVATION:407-422`）；且 CPU 的 f16/bf16 除法是
  **「float32 除完再舍回窄 dtype」**（`p5.sh` C2，20000 样本 0 差异，`:400-405`）。
  Ascend 的窄浮点除法若使用倒数近似或不同舍入路径，**f16 的 bit-wise 一致会破**
  —— 该风险已被登记为「最大落地风险」（`CLUSTER_ID_RULING_20260911.md:149`）。
- **本设计的处理**：
  1. **算法上消除对 NPU 窄浮点除法的依赖**：所有除法在 **f32** 上执行，
     窄 dtype 只承担「两次舍入」（`cvt_f32→cvt_narrow` 与 `cvt_narrow→cvt_f32`），
     舍入用 `RoundMode::CAST_RINT`（RNE），与 CPU 一致。
  2. 因此残留假设只有一个：**NPU 的 f32 除法对 f32 操作数给出正确舍入的 f32 商**
     （IEEE-754 单精度 `/`，不是 `a·recip(b)` 近似）。这是**必须在目标环境实测确认**的点，
     列入验证计划 **V-1**（唯一一条「设计依赖但尚未实测」的算术假设）。

     > **2026-09-11 收窄：该假设必须按「层 + 指令模式」限定（实测驱动）。**
     > *（本块为 2026-09-11 追记，**含 NPU 实测**；正文他处「本文不含本项目自有的 NPU 实测数据」
     > 等声明指的是本文**原稿**（当日更早的冻结版），追记块不在其范围内。）*
     > §9 的算术全部发生在 **kernel 内**（见 §4/§10：`trunc_to_int32(shift/size)`），
     > 因此 V-1 的取证对象必须是**寄存器层的除法**，而不是 aclnn 层的 `torch.div`——两者不是同一条代码路径。
     >
     > **实测**：`Reg::Div` 在**默认模式**下，在 §9/`BASE_FORMULA_DERIVATION:113-139` 自己点名的判别对
     > `(X=3.0, size=0.1)` 上给出与 oracle 不同的**截断结果**——**29** vs oracle **30**
     > （`X=-3.0` 对称为 **-29** vs **-30**）。同一张对拍表里，aclnn 列 15/15 与 CPU 正确舍入值一致。
     > ⚠️ **证据边界**：回执只记录了**截断后的整数**，**未记录误差幅度**，
     > 故本节**不主张**「误差恰为 1 ulp」这类量化表述；可主张的是
     > **「默认模式下该判别对不满足本假设」**。
     >
     > **实现规范（强制）**：kernel 内的除法必须走**补偿（高精度）路径**。
     > 在本仓头的实际布局下，**正确且可编译**的写法是置**补偿开关**：
     > `static constexpr DivSpecificMode kDivPrecise = {MaskMergeMode::ZEROING, true};`（第二成员即 `precisionMode`）；
     > 而**不是**按名指定 `DivAlgo::PRECISION_0ULP_FTZ_TRUE`——后者所属的 4 成员 `DivSpecificMode` 布局
     > **在非 3510 分支不存在**，按名书写**编译不过**。两种写法都落到同一个 `DivPrecisionImpl`
     > （先算 `x1 = a/b`，再用精确 FMA 求残差并补偿）。
     > 这**不违反**本节的禁令——它仍是一次 `shift/size`，**没有**把 `1/size` 提出循环、
     > **没有**对 `shift/size` 做重结合、**没有**使用倒数近似，只是除法**指令序列**的选择。
     > （§9 上文「不用 FMA」针对的是**对算式的重结合**；补偿序列是**除法指令自身的内部实现**，
     > 不是对算式的改写——据此澄清措辞，避免与本节字面冲突。）
     >
     > **取证（寄存器层 + 补偿开关）**：96/96 用例与真 CPU `torch_cluster` **逐位相等**
     > （`worst_max_abs_diff = 0`，覆盖 66 973 个 cluster ID，每例跑 2 遍）；
     > 同一套 96 例在**默认模式**下为 **94/96**，失败的两例正是上述判别对。
     > ⚠️ **本项的信息量边界**：96 例中 **94 例在两种模式下都通过**，对「补偿模式是否恢复正确舍入」
     > **不提供信息**；真正有判别力的只有那两个（及同族）判别对。
     > ⇒ **V-1 的结论限于：在「寄存器层 + 补偿开关」下，判别对与 oracle 逐位一致；默认模式下不成立。**
  3. 若 V-1 不成立，本设计的退路是：f16 路径改走「f32 除 + 与 oracle 对拍定位舍入差异」，
     并在 f16 上用软件舍入修正（f16 只有 10 位尾数，可用整数域精确复刻双舍入）。
     退路不改变 host/分核/tiling 设计。
- **为什么这不违反「f16 是 L1 性能考核 dtype」**：官方计时是
  `ops_gnn.grid_cluster(...)` 的**端到端 wall-clock**（`benchmark_grid_cluster.py:71-80`），
  不含算子内部的位级约束；把 f16 提升到 f32 计算只影响单核算术量，不影响正确性口径。
  性能是否达标由实测裁决（验证计划 V-4）。
- **代价的诚实陈述**：f16/bf16 路径的向量算术量约为同 D 下 f32 路径的 2×
  （多两次窄化转换往返）。这是为 bit-wise 一致性付出的确定成本，**不通过牺牲一致性来换性能**。

#### 10. 两条计算路径（`D ≤ 64` 与 `D > 64`）

**决策**：**只有一条 lane 布局**（每行固定占 `Dp/64` 个 64-lane 组，`Dp = 64·ceil(D/64)`），
但按 `groupsPerRow` 分两种归约收尾方式。两者**数学式完全相同**，差别只在「归约结果要不要跨组再求和」。

**`D ≤ 64`（`groupsPerRow = 1`）—— 每组即一行，组内归约直接就是该点的 ID**

- UB 行布局：每行占 **恰好 64 个 lane**，其中 `l < D` 为该点的第 l 维，`l ≥ D` 为恒零 lane。
- 向量算术在 64-lane repeat 粒度上完成，不跨界：
  `sub → [窄化→宽化] → div → [窄化→宽化] → cvt_int32(CAST_TRUNC) → cvt_int64 → mul_int64(base)`
  然后做**组内归约求和**（`ReduceSum` 的带 mask 归约形态，
  仓内先例 `nearest_vec_kernel.cpp:356,369` 使用 `ReduceMin<float/int32_t>(dst, src, work, count)`），
  得到「每行一个 int64」，直接进 `redBuf`。
- 恒零 lane 的安全性：`l ≥ D` 的 lane 由 Host 写入 `base[l] = 0`、`size[l] = 1.0`、`start[l] = 1.0`
  （`size`/`start` 的占位值**取 1.0 而不是 0.0**：0.0 会让 `0/0` 产生 NaN，而 NaN 走 `CAST_TRUNC`
  的结果依赖平台；取 1.0 使该 lane 的商恒为有限值，再乘 `base = 0` 得 0）。
  这与 oracle 无关（oracle 没有 padding 概念），是一条**纯实现性决策**，只为消除平台 UB。
- 这是性能矩阵 D=3 与功能用例 D=1/2/3/16/32/64 中 `D ≤ 64` 部分的唯一路径。

**`D > 64`（`groupsPerRow = Dp/64 ≥ 2`）—— 组内归约得到「部分和」，再按行跨组求和**

- 一行被拆成 `groupsPerRow` 个 64-lane 组；第 g 组承载第 `g·64 .. g·64+63` 维。
- 每组的组内归约得到该行的**部分和**，写入 `redBuf[(g mod groupsPerRow)·tilePoints + (g / groupsPerRow)]`；
  全部组算完后，对 `redBuf` 按行做一次跨组求和（归约长度 `groupsPerRow`、元素数 `tilePoints`，
  这是一次规整的 stride 归约），结果原地写回 `redBuf[0..n)`。
- **证据支撑「跨组求和安全」**：oracle 的 `out.sum(1)` 是 **int64** 求和
  （源码逐行见 `BASE_FORMULA_DERIVATION:456-470`），整数加法满足结合律，
  因此把 `Σ_d` 拆成「先组内、再跨组」**不改变结果**，不破坏 bit-wise。
  （这也是本设计**允许**在此处重新结合、而在除法处严禁改写的分界线：
   整数域可结合，浮点域不可。）
- **官方用例里 D > 64 不在测试矩阵内**（性能矩阵最大 `D=64`，功能用例最大 `D=64`），
  但任务书未给出 D 上界，故本设计把该路径一并实现并纳入自测（不属于「未决项」，
  属于「合同未覆盖但按通用性实现」）。

**为什么不做「按 D 压缩 lane」的第二套布局**（例如让 4 行共享一个 64-lane 组的 16-lane 子段）：
那样归约要落在子段边界上，需要额外的 `Duplicate`/`Select` 位移处理，
而收益只是省下 `inQue` 的 UB —— 本设计在 D=16 时 `inQue` 占 128 KiB（见 §2 账本），
而 `Dp=64` 的统一布局让 `D=3/16/32/64` 共用同一套 tile 参数，
省下的复杂度比省下的 UB 更值钱。**任务书未规定 layout**；本设计取「一行一组」，
理由是它能用一句话讲清、且归约不需要任何地址重排。

#### 11. 线性化累加与 int64 输出的写出

**决策**：累加**全程在 int64 域**，逐元素乘 `i_d·base_d` 后做行内归约，结果直接写 int64；
**不做重编号、不做 unique、不做裁剪、不做溢出检查**（补码回绕）。

- **int64 寄存器布局**：int64 在向量寄存器里占 2 个子通道（`RegTraitNumTwo`，
  仓内先例 `segment_csr_vec_kernel.cpp:44-45` 的 `RegisterType<int64_t>` 与
  `segment_csr_short_sum.h:37-40` 的 `RegTensor<int64_t, R::RegTraitNumTwo>`）。
  本设计把 int32 的 `i_d` 放在**偶数子通道**、奇数子通道置零，
  64-lane 组内的求和因此仍等于 `Σ_d i_d·base_d`，与 lane 排布无关。
- **`rowWeight[d] = max(1, base[d]·n_d)` 字段的用途**：判定某个 `i_d·base_d` 是否**可能**
  超出 int32 域。若 `rowWeight` 全为 1（即 `|base_d·(n_d−1)| ≤ INT32_MAX`），
  可把逐元素乘与累加压到 **int32 域**，最后一次性 int32→int64（省一半寄存器与带宽）；
  否则走全 int64 路径。**该优化是可选分支**；无论走哪支，**归约完的那一次转换必须在 int64 域**
  以保证回绕语义与 oracle 一致。
- **回绕的实测依据**：`BASE_FORMULA_DERIVATION:241-252`（D2.6）——
  `pos=[0,3e6]^3, size=1, start=0`，`exact = 27000027000009000000`，
  oracle 输出 `8553282926299448384` = `exact mod 2^64` 的有符号解释。
  实现若用 `long long`/`int64_t` 自然满足；若中间用高精度整数，必须显式模 2^64。
- **CopyOut**：`D ≤ 64` 时 `redBuf` 已是精确 int64（每行恰好一个元素），
  直接 `DataCopyPad(clusterGm + tileStart, redBuf, {1, n·8, 0, 0})`；
  `D > 64` 时先按行跨组求和（长度 `groupsPerRow`），再同样精确写 `n·8` 字节。
  尾块只写有效字节，不写 padding 字节。
- **`redBuf` 在两处的用途**：`D ≤ 64` 时它就是最终输出暂存；`D > 64` 时它先存部分和、
  跨组求和后再作为输出暂存（原地覆盖）。两种情形占用同一块 `tilePoints·groupsPerRow×8`。

#### 12. 空 `pos` 早退路径

**决策**：**Host 侧早退**。`N == 0` 时：

1. Python/Host 层直接返回 `torch.empty(0, dtype=torch.int64, device=pos.device)`；
2. **不查询平台、不启动 kernel、不分配 UB、不做任何 GM 写**；
3. `size`/`start`/`end` 的合法性校验**仍然执行**（`size.numel() == pos.size(1)` 等），
   使非法参数不会被空输入掩盖。

**理由与依据**：
- 任务书 `:65` 明确要求「空 `pos` 返回空 LongTensor」；
- 官方样例 `test_grid.py:34-35` 的手写 CPU 代理也是 `if p.size(0) == 0: return torch.zeros(0, dtype=torch.long)`，
  与本决策一致；
- **oracle 无法作为这一例的真值**：CPU `torch_cluster` 对任意 `(0,D)`/`(0,)`、无论是否给 `start/end`
  **一律抛 `RuntimeError: cannot reshape tensor of 0 elements into shape [0, -1]`**
  （`BASE_FORMULA_DERIVATION:97-107`，42 组实例）。
  本设计**取任务书口径**（返回空张量），理由是：任务书是合同、`:65` 明文规定；
  oracle 在这一例上没有输出、无法「逐位比对」，因此不存在与 bit-wise 口径的冲突。
  该口径冲突已登记为需上游确认项（见「需上游/官方维护者回答的问题」Q-2）。
- 该早退使 kernel 内**不存在 N=0 分支**，核内所有循环至少执行一次 —— 这是刻意的：
  仓内已有「inactive 分支在 DataCopyPad/event 之前 return，不留悬空 wait」的评审先例
  （`tasks/nearest/remediation_20260909/NEAREST_TRANSFER_DESIGN.md:68`），
  把空路径放在 host 是风险最低的落点。

#### 13. Kernel 的 Init / Process 划分与 CopyIn/Compute/CopyOut

**决策**：单 kernel、`Init()` + `Process()` 两段；`Process()` 内 `for tile` 循环由
`TQue` 的双缓冲自动流水；**核间无同步、核内无跨核 flag**（`SyncAll`/`CrossCoreSetFlag` 一律不用）。

```
Init(gmPos, gmSize, gmStartOrValue, gmBase, gmOut, tiling, pipe)
  ├─ GetBlockIdx() / GetBlockNum() → begin_, end_（按 tiling.pointsPerCore 切片）
  ├─ pipe.InitBuffer(inQue,   2, tilePoints·rb + 512)   # +512：DMA 尾部写 block 的 padding 余量
  ├─ pipe.InitBuffer(coefQue, 2, Dp·(sizeof(T)+4+8) + 256)
  ├─ pipe.InitBuffer(midBuf,  1024)                     # 64-lane 组级中间量（i_d 与 i_d·base_d 复用）
  ├─ pipe.InitBuffer(redBuf,  tilePoints·8)
  ├─ pipe.InitBuffer(maskBuf, 256)
  └─ 预生成有效 lane 掩码（`groupsPerRow == 1` 时为编译期常量；否则按行宽算一次）

Process()
  for t = begin_; t < end_; t += tilePoints:
      n  = min(tilePoints, end_ − t)
      ── CopyIn（MTE2）──────────────────────────────────────────
      inQue:  DataCopyPad(posTile, gmPos + t·D,
                          { blockCount = n,
                            blockLen   = D · sizeof(T),      // 字节，允许非 32B 倍数
                            srcStride  = 0,                  // 源行间无间隔（D 是连续行宽）
                            dstStride  = (Dp − D) · sizeof(T) / 32 },   // UB 侧补到 Dp 宽
                          { isPad = false, leftPad = 0, rightPad = 0, padValue = 0 })
      coefQue:DataCopyPad(sizeVec, gmSize,  {1, D·sizeof(T), 0, 0})   // Host 已补齐到 Dp
              DataCopyPad(startVec, gmStart, {1, D·sizeof(T), 0, 0})  // D ≤ 8 时可改为 by-value 参数
              DataCopyPad(baseVec, gmBase,  {1, D·8, 0, 0})           // int64，Host 已补 0
      ── Compute（V）── 按 64-lane 组循环；每组算完立即归约 ──────
      for g in 0 .. (n · groupsPerRow − 1):
          V1  shift  = sub(posTileG, startBroadcastG)           # 域按 §4
          V2  (窄浮点) cvt_narrow → cvt_f32（舍回再扩宽，两次）  # §9
          V3  (整型)  cvt 回源位宽 → cvt_f32                     # §4 回绕
          V4  quot   = div(shift_as_f32, sizeBroadcastG)        # 禁止改写为乘法；**必须启用补偿开关**（见 V-1）
                                                                # 本仓布局下：DivSpecificMode{MaskMergeMode::ZEROING, true}
          V5  i32    = cvt_int32(quot, CAST_TRUNC, NO_SAT)      # §8
          V6  i64    = cvt_int64(i32)                           # §11
          V7  prod   = mul_int64(i64, baseBroadcastG)           # §11
          V8  partial = ReduceSum<组内 64 lane>(prod)           # 每组产 1 个 int64
          V9  redBuf[g] = partial        # 布局：redBuf[(g mod groupsPerRow)·tilePoints + (g / groupsPerRow)]
          V10 PipeBarrier<PIPE_V>（midBuf 组间复用）
      V11 (仅 groupsPerRow > 1) 按行跨组求和 → redBuf[0..n)       # §10，int64 加法可结合
      ── CopyOut（MTE3）─────────────────────────────────────────
      DataCopyPad(gmOut + t, redBuf, {1, n·8, 0, 0})
```

> 说明：**`end` 不作为 kernel 参数**——它已在 Host 侧折叠进 `base`（§6），
> kernel 只需要 `size`、`start`、`base` 三个向量。这既减少了 kernel 参数，
> 也保证「`end` 的语义只有一个落点」，不会出现 host/kernel 两处各理解一次的风险。

- **同步事件（最小集）**：`MTE2→V`（消费输入 tile）、`V→MTE3`（消费归约结果）、
  `MTE3→V` 与 `MTE2→V` 的 buffer 复用配对由 `TQue` 的 `EnQue/DeQue` 自动管理；
  `V→S` 仅在需要把标量（如 `usedCores`/`tailPoints`）读回时使用。
  不引入 `PipeBarrier<PIPE_ALL>`（仓内评审明确反对「为消除启发式告警而加 PIPE_ALL」，
  见 `NEAREST_TRANSFER_BATCH.md:70`）。
- **为什么不用 SIMT**：本算子是**规整的向量算术 + 每点一次短归约**，
  没有分支发散、没有数据依赖链、没有 gather 型访存；
  仓内 SIMT 先例（nearest/knn/utils）用于「逐查询搜索树/逐线程走路径」这类发散场景。
  本设计取 AIV 向量路径，理由是访存规整、可双缓冲流水，且 int64 归约可用向量归约直接表达。
  这是一个**设计判断**（无实测对比支撑），登记为 decision-ledger 的 judgement call。
- **CopyIn 的 start 处理**：`start` 逐元素参与减法，因此**必须进 UB**。
  当 `D ≤ 8` 且 dtype 为 f32 时，直接把 `start` 以 **by-value 数组**形式放进 kernel 参数
  （省一次 DMA 与一次同步）；否则走 `coefQue` 搬入 `Dp` 元素并广播。

#### 14. 分核与 tile 的边界情形（穷举）

| 情形 | 行为 |
|---|---|
| `N < launchBlocks × 64` | `launchBlocks = max(1, ceil(N / 64))`，不启动无工作的空核 |
| 尾块 `n < tilePoints` | `n = min(tilePoints, end_ − t)`；CopyOut 只写 `n·8` 字节 |
| 某核 `begin_ == end_` | `Process()` 立即返回，不执行任何 CopyIn/CopyOut/事件 |
| `D = 1` | `Dp = 64`，`groupsPerRow = 1`；`base = [1]`，`id = i_0`（官方用例 `(8,1)` 覆盖，`test_grid.py:17`） |
| `D = 2` | `Dp = 64`，`groupsPerRow = 1`（官方用例 `(128,2)`、`(1024,3)` 等，`test_grid.py:14-20`） |
| `1 ≤ D ≤ 64` | `Dp ≡ 64`，`groupsPerRow = 1`，掩码为编译期常量 |
| `D > 64` | `Dp = 64·ceil(D/64)`，`groupsPerRow ≥ 2`，按行跨组求和 |
| `D > kMaxDim`（512） | Host 走 GM scratch 分支放 `n`/`base`（见 §5 理由 3） |
| `tilePoints` 算得 < 64 | 先取 64；若 64 点仍放不下 UB ⇒ Host 报错（能力不足），不启动半成品 kernel |

#### 15. 关于「`start`/`end` 缺省逻辑落点」的官方硬性要求

任务书 `:226`（特别注意事项 2）原文：

> `start`/`end` 可选逻辑须在 Host 层与原版一致

**决策**：**缺省推导放在 Host 层（C++ `op_host`），Python 层不做补齐**。

- Python 层只做 `def grid_cluster(pos, size, start=None, end=None)` 的直通与校验，
  **不把 `None` 预先替换成张量**；`None` 原样传到 Host。
- Host 层推导（严格按任务书 `:15` 与实测闭式）：
  `start_d = pos[:,d].min()`、`end_d = pos[:,d].max()`，**逐维**、**dtype = `pos.dtype`**。
- **理由**：
  1. 任务书明文要求「在 Host 层」；
  2. `min`/`max` 是 O(N·D) 归约，放在 Python 层会多一次 kernel 启动 + 一次 device→host→device 往返，
     直接吃掉 D=3、N=1M 档位的全部时间预算（该档标杆仅 0.014 ms）；
  3. 与官方参考实现的位置一致：`csrc/cpu/grid_cpu.cpp` 在 host 端做同一推导
     （逐行源码见 `BASE_FORMULA_DERIVATION:456-470`）；
  4. **dtype 必须继承 `pos.dtype`**，不是浮点：实测依据
     `BASE_FORMULA_DERIVATION:533`（D4.3 第 2 条）与 D1.3（`:73-81`）。
     若 Host 侧把 `min/max` 提升到 f32，整型输入的 `pos − start` 回绕语义就会改变。
- **`end` 的使用说明（本设计明确回答 §「背景介绍」中的问题）**：
  `end` **只用于推导每维体素数 `n_d`，从而只影响 `base_d`；它不裁剪点坐标**。
  超网格的点不报错、不被截断，只是 `i_d` 取到更大的整数（实测 `p1.sh` E2：
  `end=100` 时 `[9,9,9]` 仍映射到合法 ID `80808`，`BASE_FORMULA_DERIVATION:202`）。
  「网格边界」在本算子中**不是约束，只是基数**——这是本设计对该问题的最终回答。
- **基准脚本的调用形态要求 Host 支持显式 `start`/`end` 且不得做归一化**：
  `benchmark_grid_cluster.py:71-90` 以 `ops_gnn.grid_cluster(pos, size, pos.min(0).values, pos.max(0).values)`
  四个**位置参数**调用（即 `start = pos.min(0)`、`end = pos.max(0)`，均非 `None`）。
  此时 `(end−start)/size` 与缺省路径**逐位相同**（逐维 min/max 定义相同），
  故性能矩阵与功能矩阵走的是同一条 Host 代码路径，不存在「基准走特化路径」的风险。

#### 16. 正式数据（Host 层对 bit-wise 一致性无影响）

Host 侧只产出 `n`/`base` 两个**整数**向量与分核/tiling 参数；
所有浮点算术（减法、除法、截断）都发生在 kernel 内，且已按 §4 的域固定。
因此 Host 与 Device 之间的接口**不引入新的舍入点**。
唯一的例外是 `start`/`size` 的搬入方式（DMA 或 by-value 参数），
两者都不改变数值（同 dtype、同比特）。

### 算子伪代码（Host + Kernel，供实现直接对照）

```cpp
// ============ Host ============
// 0. 形状与校验（失败即抛，不静默修复）
int64_t n = pos.size(0);
int64_t dim = (pos.dim() == 2) ? pos.size(1) : 1;      // (N,) 视作 D=1
TORCH_CHECK(pos.dim() == 1 || pos.dim() == 2, ...);
TORCH_CHECK(size.numel() == dim, "size.numel() == pos.size(1)");
// 1. 空输入早退（§12）—— 早退前先完成上面的校验，避免空输入掩盖非法参数
if (n == 0) return torch::empty({0}, pos.options().dtype(torch::kLong));
// 2. start/end 缺省推导（§15）—— 逐维；dtype 继承 pos.dtype，不提升到 f32
Tensor startT = start.has_value() ? *start : std::get<0>(pos.min(0));
Tensor endT   = end.has_value()   ? *end   : std::get<0>(pos.max(0));
// 3. n_d 与 base_d（§6）—— 「逐维、先 trunc 后 +1、int64、补码回绕」
Tensor nv    = (endT - startT).true_divide(size).to(torch::kLong) + 1;
Tensor base  = torch::cat({torch::ones({1}, nv.options()), nv.cumprod(0)}, 0).narrow(0, 0, dim);
// 4. 平台与分核（§1）—— 必须运行时查询，不硬编码
auto platform = platform_ascendc::PlatformAscendCManager::GetInstance();
uint32_t cores = platform->GetCoreNumAiv(); if (cores == 0) cores = 1;
// 5. tile 与行宽形态（§2/§7/§10）→ 填 TilingData（§5）
//    → LaunchGridCluster<T, WIDE><<<blocks, nullptr, stream>>>(...)，WIDE = (D > 64)

// ============ Kernel（AIV）============
__aicore__ inline void Process() {
  for (uint64_t t = begin_; t < end_; t += tilePoints_) {
    const uint32_t n = static_cast<uint32_t>(min<uint64_t>(tilePoints_, end_ - t));
    CopyIn(t, n);                       // MTE2：DataCopyPad pos / size / start（§7）
    Compute(n);                         // V   ：sub→[narrow round-trip]→div→CAST_TRUNC→int64 mul→ReduceSum
    CopyOut(t, n);                      // MTE3：DataCopyPad 精确 n*8 字节（§11）
  }
}
```

### 验证计划与未关闭风险

以下各项**不是设计缺口**，而是「设计已定、须在目标环境取证」的验证门禁。
全部结论将在自测报告中逐项给出实测数据。

| ID | 验证项 | 为什么必须实测 | 判定 |
|---|---|---|---|
| **V-1** | **NPU f32 除法是否为正确舍入的 IEEE-754 单精度商**（非 `a·recip(b)` 近似）——**取证对象必须是寄存器层的 `Reg::Div`，且必须启用补偿开关** | 这是 §9 唯一残留的算术假设；f16/bf16 的 bit-wise 完全依赖它。**aclnn 层的 `torch.div` 不是同一代码路径，其通过不能作为本项证据** | 构造使 f32 与 f64 商截断结果分叉的 `(span,size)` 对（`BASE_FORMULA_DERIVATION:113-139` 已给出 45 组候选，如 `X=3.0,size=0.1`），**在 kernel 的寄存器除法路径上**与 oracle 对拍；并**记录该路径实际使用的 `DivSpecificMode` 取值**（本仓布局下为 `{MaskMergeMode::ZEROING, true}`，即 `precisionMode=true`）。**实测（2026-09-11）**：默认模式失败（`3.0/0.1 → 29`，oracle `30`）；启用补偿开关后 96/96 逐位相等（**其中仅 2 例具判别力，94 例两种模式都通过**） |
| **V-2** | NPU 侧 `CAST_TRUNC` + `NO_SAT` 的 f32→int32 行为与 oracle 一致（含负数、大值） | 截断方向是 bit-wise 分水岭（§8） | 用 `pos=[-1.5,0,1.5,-0.5,0.5,-2.0],start=0,size=1` 复现 `[-1,0,1,0,0,-2]` |
| **V-3** | 整型回绕（i8/u8/i16/i32）与 `true_divide` 的 f32 提升 | 三项均在 CPU 实测有判别样本（`p4.sh` S4、`p8.sh` Q3） | 复现 `int8 [0,127,-128,-1]`、`uint8 [128,255,0,127]`、`int16 [0,127,128,255]` |
| **V-4** | 全量性能 ≥ 0.45× 标杆（任务书 `:89-144` 的 47 行矩阵，逐例） | 本设计尚未在 NPU 上运行 | 按 `benchmark_grid_cluster.py` 的 `--iter/--warmup` 口径逐例采集；任一例不达标即按 §2 的 `tilePoints`/`reserved` 与 D-21/D-22 的双缓冲取舍迭代，并对新版本重新全量取证 |
| **V-5** | `DataCopyPad` 单一 API 是否满足性能（§7 的取舍） | 「统一 API」是设计判断，非实测结论 | 微基准：对 D=16/32/64 比较 `DataCopyPad` 与 `DataCopy` 快路径；差值 > 5% 才增设快路径 |
| **V-6** | 验收进程的 `torch.get_default_dtype()` 是否为 float32 | 整型输入 ID 依赖它（`2^24+1` 判别：f32 默认→`16777216`，f64 默认→`16777217`，`BASE_FORMULA_DERIVATION:432-438`） | 在验收 harness 上打印默认 dtype；若不是 f32，整型路径需改按实际默认 dtype 提升（对应问题 Q-5） |
| **V-7** | UB 账本实测占用 ≤ `ubBytes`（含 TBuf 合并后） | 32 KiB 保留量与 TBuf 合并都是设计判断 | kernel 内断言 + 编译期静态检查双保险 |
| **V-8** | 与 oracle 的直接对拍（bit-wise） | 官方样例只比分组，测不出线性化错误（`BASE_FORMULA_DERIVATION:538-571`） | 复用 `p3/p6/p9c` 的 1106 组配置作为回归基线，NPU vs CPU oracle 逐位比对 |

### 需上游/官方维护者回答的问题（不影响本设计先行实现）

以下两项**本设计已给出并可执行的口径**，但官方材料本身存在冲突，故同时登记请官方确认；
即使不回答，本设计的实现与自测也照常推进。

| ID | 问题 | 官方材料冲突点 | 本设计当前口径 | 若官方改判的影响面 |
|---|---|---|---|---|
| **Q-2** | `pos` 为空（`N = 0`）时应当返回空张量，还是复刻 CPU oracle 的异常？ | 任务书 `:65` 要求「空 `pos` 返回空 LongTensor」；而实测 CPU `torch_cluster` 对任意空输入一律抛 `RuntimeError: cannot reshape tensor of 0 elements into shape [0, -1]`（`BASE_FORMULA_DERIVATION:97-107`） | **取任务书口径**：返回 `int64` 空张量（§12）。理由：oracle 在该例无输出、不存在「逐位比对」，且官方样例 `test_grid.py:34-35` 亦返回空张量 | 只影响空输入用例的期望值；不影响任何非空用例 |
| **Q-3** | 取整语义以任务书 `:12` 的 `⌊·⌋` 为准，还是以 `:64`「`true_divide` 后转 long」+ CPU oracle 的向零截断为准？ | 任务书 `:12`（floor）与 `:64`（trunc）在 `start > min(pos)` 时不等价；实测 oracle 用 trunc（`BASE_FORMULA_DERIVATION:57-67`） | **取向零截断**（§8），与 `:64` 及 oracle 一致 | 若官方改判 floor，则所有 `start > min(pos)` 的用例（含负 ID 场景）整体不一致，需改 `CAST_TRUNC` → floor 语义 |
| **Q-4** | 官方样例的 `test_with_end` / `test_with_start_end` 是否应当更新为与真 oracle 对拍？ | 样例的 CPU 代理声明 `end` 却从不使用（`test_grid.py:30` vs `:36-37`），且比对函数只比分组（`:49-55`），因此 `end` 的正确性在样例里零判别力 | 本设计自建与 oracle 的逐位对拍（V-8），不以样例作为 `end` 的证据 | 只影响官方样例本身的可信度，不影响本设计 |
| **Q-5** | 整数 `pos` + 整数 `size` 时，`true_divide` 提升到进程 `get_default_dtype()` 的语义是否属于验收范围？ | `p8.sh` Q3（`BASE_FORMULA_DERIVATION:432-438`）实测该行为依赖进程默认 dtype；任务书 `:45` 只把 `size` 声明为 `Float [D]`，未规定整数 `size` | 本设计按 **f32** 提升（§4），并把它登记为验证门禁 V-6 | 若验收进程默认 dtype 不是 f32，整型输入在 `2^24` 量级附近的 ID 会变 |

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Ascend 950PR（arch35） | √ |

（依据任务书 `:22-23`「适配硬件：Ascend 950系列」、`:202`「Ascend 950 对应 `arch35`」。
`arch35` 即 `dav-3510`，编译目标实测值见
`tasks/nearest/delivery/reports/environment.md:19`「编译目标 dav-3510」。）

## 算子约束限制

1. `size.numel() == pos.size(1)`；
2. `size` 每维 `> 0`；
3. 空 `pos` 返回空 LongTensor；
4. 确定性算子，bit-wise 一致；
5. **合同外输入**（`size ≤ 0`、`N = 0` 的 oracle 行为）：按 §8、§12 的决策处理，
   即 `size ≤ 0` 不报错、按闭式照算（与 oracle 的 `trunc` 语义一致，但 `NaN/±inf→int64`
   转换是平台相关 UB，实测 `BASE_FORMULA_DERIVATION:444-452`），`N = 0` 返回空张量。

### 官方材料的内部不一致：本设计的裁决与依据

官方任务书与官方样例对「cluster ID 到底是什么」表面给出两种不相容的说法。
**本设计明确取任务书口径，理由如下（有实测证据，不是取舍）：**

| 来源 | 主张 | 本设计的判定 |
|---|---|---|
| 任务书 `:51` | 输出是**线性化 voxel ID**，非负 | ✅ **采信** |
| 任务书 `:156-157` | 必须与 CPU 标杆 **bit-wise 一致** | ✅ **采信** |
| 官方样例 `test_cases/test_grid.py:30-46`、`:58-69` | 手写 CPU 代理用 `hash(tuple(idx)) & ((1<<62)-1)` + `torch.unique(..., return_inverse=True)`，输出是**重编号簇序号** | ❌ **不采信**（该代理与真 oracle 数值不同） |

**裁决证据（`CLUSTER_ID_RULING_20260911.md`）**：

1. 任务书 `:152` 点名真值源是 **CPU 版 `torch_cluster`**，而实测该 oracle 返回
   `[0, 0, 364, 455, 728]`——稀疏、非负、**不连续**的线性化 voxel ID，
   「重编号为 `0..K-1`」判为 `False`（裁决书 `:26-35`）。
   ⇒ **任务书 `:51`、`:156-157` 与真实 oracle 三者互相吻合，并不矛盾。**
2. 表面矛盾**只来自官方样例里那个手写代理**，而任务是 `:150` 明说
   「**仿照已有用例自行编写 Python 用例进行自测试**」——
   即 `test_grid.py` 是**样例**，不是 oracle。样例的代理写错了，不改变合同。
3. 样例的比对函数 `_assert_same_groups`（`test_grid.py:49-55`）**只比分组集合、不比 ID 数值**，
   因此它**测不出**任何线性化改写：实测证据
   （`BASE_FORMULA_DERIVATION:542-556`，`p4.sh` S7）——
   同一配置下 radix ID `[943,429,499,294,531]`、行主序 ID `[1159,75,544,1106,448]`、
   哈希代理 ID `[24,111,6,35,91]` 三者**数值完全不同**，但
   `PARTITION identical` 对三者**全部为 True**。
   ⇒ **一个行主序实现能通过官方样例的全部功能用例**。

**推论（对本实现的硬要求）**：必须输出与 `torch_cluster` **逐位相同**的 voxel ID；
**只保证分组正确是不够的**。因此本项目的自测必须包含与 CPU oracle 的直接对拍（验证计划 V-8），
不能以官方 `test_grid.py` 全绿作为正确性证据。

**同一条款下另外两处需要如实说明的样例缺陷（均有行号证据）**：

- `test_with_end`（`test_grid.py:142-153`）**确实把 `end` 传给了 NPU**（`:150`），
  但它的比对基准 `grid_cluster_cpu`（`:30-46`）**声明了 `end` 却从不使用**
  （`:36-37` 只用 `start`），且其输出已被重新编号。
  ⇒ **该用例对 `end` 的正确性零判别力**，不能作为 `end` 语义已覆盖的证据。
- `test_empty_pos` / `test_empty_pos_2d` / `test_empty_pos_with_start_end`
  （`test_grid.py:197-227`）**只调用手写 CPU 代理，从不调用 NPU**
  （`:205`、`:214`、`:225` 三处调用点；全文件 11 处 `ops_gnn.grid_cluster(` 均不在这三个用例内）。
  ⇒ **空输入路径在官方样例里没有任何 NPU 覆盖**，本项目必须自建该用例（验证计划 V-8 的补充项）。

---

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | cluster ID（int64）与 CPU 版 torch_cluster 标杆 **bit-wise 一致**（确定性算子） | 任务书 `:148-157`；《生态算子开源精度标准》https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md |
| 性能标准 | 算子所有用例的性能 ≥ **0.45 倍**标杆性能；验收取最优实现 | 任务书 `:82-84, 146` |

### 标杆耗时（官方数据，逐字转录自任务书 `:89-116, 120-144`）

> **以下数字是官方给出的标杆基线，不是本项目 NPU 实测值。** 本项目尚未在 NPU 上运行过该算子。
> 数据为随机坐标；`N` 为输入点数，`D` 为坐标维度，`s` 为网格边长。

| N | D | s | float32 | float16 |
|---|---|---|---|---|
| 1M | 3 | 0.005 | 0.014ms | 0.015ms |
| 4M | 3 | 0.005 | 0.062ms | 0.064ms |
| 8M | 3 | 0.005 | 0.119ms | 0.124ms |
| 16M | 3 | 0.005 | 0.233ms | 0.243ms |
| 32M | 3 | 0.005 | 0.461ms | 0.482ms |
| 1M | 16 | 0.010 | 0.056ms | 0.048ms |
| 1M | 32 | 0.010 | 0.166ms | 0.099ms |
| 1M | 64 | 0.010 | 0.321ms | 0.322ms |
| 4M | 16 | 0.010 | 0.197ms | 0.172ms |
| 4M | 32 | 0.010 | 0.636ms | 0.366ms |
| 4M | 64 | 0.010 | 1.251ms | 1.257ms |
| 8M | 16 | 0.010 | 0.384ms | 0.336ms |
| 8M | 32 | 0.010 | 1.249ms | 0.717ms |
| 8M | 64 | 0.010 | 2.459ms | 2.471ms |
| 32M | 16 | 0.010 | 1.503ms | 1.315ms |
| 32M | 32 | 0.010 | 4.926ms | 2.827ms |
| 32M | 64 | 0.010 | 9.710ms | 9.755ms |
| 1M | 3 | 0.001 | 0.014ms | 0.015ms |
| 1M | 3 | 0.002 | 0.014ms | 0.015ms |
| 1M | 3 | 0.010 | 0.014ms | 0.015ms |
| 8M | 3 | 0.001 | 0.119ms | 0.124ms |
| 8M | 3 | 0.002 | 0.119ms | 0.124ms |
| 8M | 3 | 0.010 | 0.119ms | 0.124ms |
| 32M | 3 | 0.001 | 0.460ms | 0.483ms |
| 32M | 3 | 0.002 | 0.460ms | 0.483ms |
| 32M | 3 | 0.010 | 0.460ms | 0.483ms |

整数类型使用整数网格边长；`int16/int32` 表示输入数据类型。

| N | D | s | int16 | int32 |
|---|---|---|---|---|
| 1M | 3 | 4 | 0.014ms | 0.014ms |
| 4M | 3 | 4 | 0.053ms | 0.050ms |
| 8M | 3 | 4 | 0.101ms | 0.095ms |
| 16M | 3 | 4 | 0.198ms | 0.184ms |
| 32M | 3 | 4 | 0.392ms | 0.364ms |
| 1M | 16 | 8 | 0.047ms | 0.049ms |
| 1M | 32 | 8 | 0.091ms | 0.163ms |
| 1M | 64 | 8 | 0.313ms | 0.321ms |
| 4M | 16 | 8 | 0.174ms | 0.174ms |
| 4M | 32 | 8 | 0.343ms | 0.619ms |
| 4M | 64 | 8 | 1.221ms | 1.236ms |
| 8M | 16 | 8 | 0.340ms | 0.336ms |
| 8M | 32 | 8 | 0.672ms | 1.211ms |
| 8M | 64 | 8 | 2.398ms | 2.425ms |
| 32M | 16 | 8 | 1.330ms | 1.308ms |
| 32M | 32 | 8 | 2.644ms | 4.768ms |
| 32M | 64 | 8 | 9.457ms | 9.557ms |
| 1M | 3 | 1 | 0.014ms | 0.014ms |
| 1M | 3 | 16 | 0.014ms | 0.014ms |
| 8M | 3 | 1 | 0.101ms | 0.094ms |
| 8M | 3 | 16 | 0.101ms | 0.094ms |
| 32M | 3 | 1 | 0.391ms | 0.363ms |
| 32M | 3 | 16 | 0.391ms | 0.363ms |

### 标杆量的下界推算（用于校验设计是否可能达标）

对 D=3、f32、N=32M：输入 `32M×3×4 B = 402 MB`，输出 `32M×8 B = 268 MB`，
标称耗时 `0.461 ms`（任务书 `:114`）⇒ 隐含端到端带宽 ≈ **1.45 TB/s**。
本设计的读流量与写流量与该推算一致（每点只读一遍 `pos`、只写一次 `cluster`，
`base`/`size`/`start` 每 tile 各读一次可忽略），
因此**能否达标的决定因素是「不要引入多余的 GM 往返」**，而不是向量算力。
这是本设计选择「host 预计算 `base`」「原地复用 UB」「双缓冲」的直接理由。

### 本项目实测结果

**空。** 本项目尚未在 Ascend 950 上构建或运行过 `grid_cluster`，
因此**没有任何 NPU 耗时、精度比对或内存数据可以填写**。
按项目红线，在取得全量逐例实测证据之前，本节保持为空，不以估算值或他人数据填充。

## 兼容性分析

新算子，不涉及兼容性分析。

### 上游共存

本算子与上游 `torch_cluster` 的 Python schema/名称关系：

- Python 层暴露 `ops_gnn.grid_cluster(pos, size, start=None, end=None)`，与任务书 `:30-35` 一致；
- 不注册到 `torch_cluster::grid` schema 上、不覆盖上游 CPU 实现；
- 上游 CPU `torch_cluster` 仍可作为对拍 oracle 共存（仓内已有共存验证先例：
  `tasks/nearest/delivery/reports/environment.md:20`
  「上游共存库 torch_cluster 1.6.3+pt29cpu，PyG 官方 wheel」，未替换 `ops_gnn` 的动态库）。

---

## 附：已就绪的官方测试资产（来自任务书，未运行）

任务书附带 `test_cases/test_grid.py`（8770 字节，SHA256
`615DC180FF5CBD720696F0C441A9C3A8E4372355F6330181F449646B092167AC`）与
`test_cases/benchmark_grid_cluster.py`（3895 字节，SHA256
`038C185BB2A0C149A002DF3456FE5F4BB1E2AC0EBBDE6E6D698479388466C015`）。
两者均**未在本项目任何环境上运行过**。

### 官方资产的覆盖缺口（本设计已逐条对应补测用例）

| # | 缺口 | 证据 | 本设计的补测 |
|---|---|---|---|
| 1 | `test_grid.py` 只比分组，任何线性化改写都能通过 | `test_grid.py:49-55`；实测 `BASE_FORMULA_DERIVATION:542-556` | V-8：1106 组与 oracle 逐位对拍 |
| 2 | `end` 的比对基准是死参数（代理声明不用） | `test_grid.py:30` vs `:36-37` | V-8 覆盖显式 `end` 的逐位比对 |
| 3 | 三个空输入用例从不调用 NPU | `test_grid.py:205,214,225` | 自建 NPU 空输入用例（`numel==0` 且 `dtype==int64`） |
| 4 | `test_general_shapes`/`test_large` 只断言 device | `test_grid.py:93-105,178-185` | 上述形状纳入 V-8 对拍集合 |
| 5 | bfloat16 被提前 `.float()`，bf16 算术路径未覆盖 | `test_grid.py:83-84,101-102,202-203` | V-3/V-8 的 7-dtype 套件含真 bf16 输入 |
| 6 | `grid_cpu`（`:58-69`）是死代码，零调用点 | 全文件调用点枚举 | 不在本设计的参考链中 |
| 7 | 代理的 `hash()` 62 位掩码有碰撞风险，且非跨版本稳定契约 | `test_grid.py:44,67` | 本设计不以哈希代理为真值 |
