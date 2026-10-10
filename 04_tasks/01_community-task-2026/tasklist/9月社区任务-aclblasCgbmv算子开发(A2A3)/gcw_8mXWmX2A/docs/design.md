# aclblasCgbmv 算子设计文档（A2/A3，arch22）

# 需求背景（required）

## 需求来源

9月社区任务：aclblasCgbmv 算子开发（A2/A3）。在昇腾 NPU（Atlas A2/A3 系列产品，性能基准设备 Atlas 800T A2（910B3））上基于 Ascend C 以 kernel 直调方式实现单精度复数（COMPLEX64）一般带状矩阵向量乘 `aclblasCgbmv`，执行 `y = alpha * op(A) * x + beta * y`，参数与语义全面对齐 cuBLAS `cublasCgbmv`，精度 golden 由 Netlib CBLAS `cblas_cgbmv` 生成。

## 背景介绍

### ops-blas 仓现状分析

经代码走读，仓库现状与任务要求的差距如下：

| 项目 | 现状 | 任务要求 | 差距 |
| --- | --- | --- | --- |
| 接口声明 | `include/cann_ops_blas.h` 仅有 `aclblasSgbmv`（实数） | 新增 `aclblasCgbmv`（COMPLEX64）声明，供各产品线共用 | 缺接口声明 |
| 算子实现 | `blas/gbmv/` 仅 arch35 的 Sgbmv（实数）；arch22 无 gbmv | arch22 新增 Cgbmv 实现，放 `blas/gbmv/arch22/` | 缺 arch22 复数实现 |
| 可复用资产 | `blas/gemv/arch22/` 已有完整的 Cgemv 复数实现（复数乘加、共轭、 Gather 交织写回均已在 arch22 验证）；arch35 `sgbmv` 提供带状压缩索引（`ku+i-j + j*lda`）与正负步长处理范例 | — | 以 Cgemv arch22 工程范式为骨架，带状索引语义对齐 sgbmv |
| 测试 | `test/gbmv/` 仅 sgbmv（arch35） | 新增 `test/gbmv/cgbmv/arch22/`：C++ GTest + CSV 驱动 + Netlib golden | 参照仓内既有 BLAS 测试结构补齐 |
| README | `blas/gbmv/README.md` 产品支持表仅 950 | 补充 A2/A3（含 Atlas 800I A2/A3）支持与 Cgbmv 接口说明 | 需更新 |

### 算子功能分析

`A` 恒为 `m×n` 一般带状矩阵，列主序打包存储于 `lda×n` 数组（`lda ≥ kl+ku+1`），`kl`/`ku` 为下/上对角带数：

```text
y = alpha * op(A) * x + beta * y
op(A) = A      (ACLBLAS_OP_N)，op(A) 为 m×n
op(A) = A^T    (ACLBLAS_OP_T)，op(A) 为 n×m
op(A) = A^H    (ACLBLAS_OP_C)，op(A) 为 n×m（复共轭转置）
```

- `m`/`n` 恒为 A 的行数/列数，不随 trans 翻转；`N` 时 `x`/`y` 逻辑长度为 `n`/`m`，`T/C` 时为 `m`/`n`。
- `x`、`y` 按步长 `incx`/`incy` 存取，步长可为负（负步长从末尾反向取值）。
- `m=0` 或 `n=0` 为合法 no-op；`beta=0` 时 `y` 不必为有效输入（cuBLAS 语义）；`alpha=0` 时不引用 `A`/`x`，退化为 `y = beta*y`。
- 结果原地写回 `y`；异步语义依赖 `aclblasSetStream` 绑定的 stream。

目标公共接口：

```cpp
aclblasStatus_t aclblasCgbmv(
    aclblasHandle_t handle, aclblasOperation_t trans,
    int m, int n, int kl, int ku,
    const aclblasComplex* alpha, const aclblasComplex* A, int lda,
    const aclblasComplex* x, int incx,
    const aclblasComplex* beta, aclblasComplex* y, int incy);
```

| 参数 | 参数含义 | 数据类型 | 支持数据类型 | 约束 | 形状 |
| --- | --- | --- | --- | --- | --- |
| handle | 库上下文句柄，携带 stream | scalar | — | 非空 | — |
| trans | op(A) 选择 | attr | enum | N/T/C 三枚举 | — |
| m | A 的行数（不随 trans 翻转） | scalar | int | ≥0 | — |
| n | A 的列数（不随 trans 翻转） | scalar | int | ≥0 | — |
| kl | 下对角带数 | scalar | int | 0 ≤ kl ≤ m-1 | — |
| ku | 上对角带数 | scalar | int | 0 ≤ ku ≤ n-1 | — |
| alpha | 标量乘数 | scalar | COMPLEX64 | 非空指针 | — |
| A | 带状矩阵（打包存储） | tensor | COMPLEX64 | lda ≥ kl+ku+1，只读 | 存储 lda×n |
| lda | 带状存储主维 | scalar | int | ≥ kl+ku+1 | — |
| x | 输入向量 | tensor | COMPLEX64 | incx ≠ 0，只读 | n（N）/ m（T/C） |
| incx | x 的元素步长 | scalar | int | ≠0，可负 | — |
| beta | 标量乘数 | scalar | COMPLEX64 | 非空指针；beta=0 时 y 不必有效 | — |
| y | 输入/输出向量 | tensor | COMPLEX64 | 原地写回 | m（N）/ n（T/C） |
| incy | y 的元素步长 | scalar | int | ≠0，可负 | — |

`aclblasComplex` 为仓内公共复数类型（实部/虚部两个 `float` 交错存储）。

# 需求分析（required）

## 需求描述

实现 COMPLEX64 `aclblasCgbmv`，覆盖 `N/T/C` 三种变换、`kl/ku` 非对称与退化带宽（含 `kl=0`/`ku=0`）、紧凑与 padding 两种 `lda`、正负 `incx/incy`、`alpha/beta` 特殊值（0、1、-1）、Inf/NaN 传播语义、`m=0/n=0` no-op 及全部非法参数/非法枚举负向校验。行为对齐 cuBLAS `cublasCgbmv`。

## 需求拆解

1. **接口与工程**：`include/cann_ops_blas.h` 新增声明；`blas/gbmv/arch22/` 新增 Host（校验/tiling/workspace/launch）与 Ascend C kernel；更新 `blas/gbmv/README.md` 产品支持表。
2. **正确性**：带状压缩索引换算、trans 翻转后的索引域、`OP_C` 共轭语义、正负步长映射，逐一与 Netlib golden 对齐。
3. **精度**：实部/虚部全程 FP32 分量累加；满足 COMPLEX64 混合容差标准（rtol=atol=2^-13，matched_ratio≥0.99），Inf/NaN 模式与 golden 逐位一致。
4. **性能**：满足任务书 5 个性能用例的达标耗时（最大 m=n=4096、kl=64、ku=256、trans=C，达标 52.487us）。
5. **测试**：`test/gbmv/cgbmv/arch22/` 覆盖任务包全部精度/性能用例（CSV 1200 条）及边界、负向、特殊值用例。

# 详细设计（required）

## 算子分析

### 数学公式

```text
y = alpha * op(A) * x + beta * y
```

0-based 索引下，仅 `max(0, j-ku) ≤ i ≤ min(m-1, j+kl)` 的 `A(i,j)` 有效，物理偏移：

```text
band_row = ku + i - j          // 0 <= band_row <= kl+ku
A_offset = j * lda + band_row  // 64 位整数计算，防大维度溢出
```

- **N 模式**（按输出行）：`y[i] = alpha * Σ_{j∈[max(0,i-kl), min(n-1,i+ku)]} A(i,j)·x[j] + beta·y[i]`
- **T 模式**（按输出列）：`y[j] = alpha * Σ_{i∈[max(0,j-ku), min(m-1,j+kl)]} A(i,j)·x[i] + beta·y[j]`
- **C 模式**：同 T 的索引域，仅对 `A` 取复共轭（`x`、`alpha` 不共轭）

复数乘法 `(ar+i·ai)(xr+i·xi) = (ar·xr − ai·xi) + i·(ar·xi + ai·xr)`，实部/虚部以 FP32 分量分别累加。

### 支持数据类型

COMPLEX64（`aclblasComplex`，实部/虚部各一个 float）；计算全程 FP32。

### 支持形状

`m、n ≥ 0`（`m=0` 或 `n=0` 为合法 no-op）；`0 ≤ kl ≤ m-1`、`0 ≤ ku ≤ n-1`；`lda ≥ kl+ku+1`（紧凑 packing 与任意 padding 均支持）；`incx/incy` 任意非零整数（含负步长）。不涉及广播。

## 算子实现

**设计思路**：N 路径采用"列分核两阶段"结构——Phase 1 按列把矩阵切给各核，每核在自己拥有的列区间上做带段乘加，把部分和累加进本核私有的 y 窗口（UB 驻留）并写入 GM workspace；经全局同步后，Phase 2 按行重新分核，每个核把与本核行段相交的各核窗口片段按核号升序归约，得到每行的完整加和，再统一施加 alpha/beta 并写回 y。T/C 路径按输出列分核，每列的带内元素在 GM 上是连续区间，天然适合向量载入与归约。复数在 GM 为交错 float2，kernel 内拆成实/虚两个平面处理，写回时再以 Gather 交织。

### host 侧设计

**参数校验**（对齐任务书异常行为，按优先级）：handle 空 → `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`；`m<0 || n<0` → `INVALID_VALUE`；`m=0 || n=0` quick-return 成功（不解引用 Device 指针）；`trans` 非 N/T/C、`kl∉[0,m-1]`、`ku∉[0,n-1]`、`lda<kl+ku+1`、`incx=0`、`incy=0`、`alpha/beta` 空指针 → `INVALID_VALUE`；`alpha≠0` 时 `A/x` 空、`y` 空 → `INVALID_VALUE`；`alpha=0 && beta=(1,0)` 直接返回成功（不引用 A/x/y）。

**tiling 策略**：全部标量参数（m/n/kl/ku/lda/incx/incy/trans/alpha/beta 及 packed 批处理列数 K）经纯 POD tiling 结构体按值传入 kernel；Host 侧零 D2H 回拷、零同步。

**分核策略（自适应三档）**：以 `min(outDim, 设备 AIV 核数, 档位上限)` 确定 launch 核数，N 路径按 `n` 分档（n≤256 上限 16 核、256<n≤1024 上限 32 核、更大 48 核），T/C 路径恒 48 核——小 shape 降核可显著减少启动与同步开销（实测 256² N 用例 16 核最优、512² N 用例 32 核最优）。留 `CGBMV_FORCE_BLOCKS` 环境变量覆盖入口供调参。Phase 1 按列均分（`rpcC = CeilDiv(n, 核数)`），Phase 2 按行均分（`rpc = CeilDiv(m, 核数)`），首尾核处理余数。

**workspace 策略**：N 两阶段需要跨核交换部分和，使用 handle 工作区（`EnsureDefaultWorkspace` 按需增长，用户注入工作区不足时自动回退单阶段路径）。布局：头部 512B 为 64 核风险标志位（8B pitch），其后每核一对实/虚 y 窗口平面，平面步长 `rpcC + kl + ku + 8`（float）。带宽 `kl+ku+1 > 1024`（单列带段超过 UB 队列槽）或工作区申请失败时，整体回退单阶段行分核路径。

### kernel 侧设计（Ascend C，arch22，AIV）

每核 `Init → Process`。UB 总量约 159KB（192KB 预算内）：累加窗/旧 y/临时平面 2×1024 float 各若干、x 平面与窗口暂存 2×2048 float、DMA 队列槽 2×1024 复数（深度 2）、Gather 偏移表等。

**N 路径（两阶段）**

- **Phase 1（列分核）**：核 k 拥有列 `[k·rpcC, min(n, (k+1)·rpcC))`。该列块的贡献行范围（y 窗口）为 `[max(0, c0-ku), min(m-1, c1-1+kl)]`，窗长 ≤ 列数+带宽；超过 1024 行时按子块切分（每子块列数 ≤ 1024−带宽），子块间通过 workspace 高水位线做"重叠段读回-加-写回、新段直写"的增量刷写。窗口在 UB 中按列升序累加（与参考实现的求和顺序一致）。
  - **多列批处理 DMA**：紧凑 packing（`lda = kl+ku+1`）时相邻列带段在 GM 首尾相接，K 列（K 由槽容量自适应，3~4 列）共享一个队列槽——一次 EnQue/DeQue 握手搬 K 列，事件与发射次数 ÷K；每列带段头部向下扩展至 8 行边界（保证累加目的地 32B 对齐），扩展头行在 UB 内补零。
  - 每列计算：对交织带段做 Abs+ReduceMax 风险追踪，split 拆实/虚平面，标量 `x` 参与 4 条 Axpy 乘加（`accR += aR·xr − aI·xi`，`accI += aR·xi + aI·xr`）。
  - `incx=1` 时 x 窗口一次 DMA + split 进 UB 平面；否则按步长标量读。
- **全局同步**：`SyncAll()` 为顶层直线语句，全核（含空核）恰好到达一次。
- **Phase 2（行分核）**：先读全部核风险标志——任一核风险值 ≥ 2^126 则整矩阵回退单阶段路径重算；正常时按核号升序读取各贡献核的窗口片段（行段与窗口交集，头部 8 对齐扩展 + 补零），矢量加进本核行段累加器（升序 = 列块升序 = 列升序，与参考求和顺序一致），再统一合成 `alpha·acc + beta·y_old` 写回 y（`beta=0` 不读 y；`alpha` 零分量跳过对应乘法，避免 0·Inf=NaN 污染）。

**T/C 路径（列点积分核）**

- 每核拥有连续输出列区间，按 256 列分组；组内 K 列共享一个队列槽做带段 DMA（与 N 路径同款批处理）。
- **x 驻留复用**：`incx=1` 且组列范围的 x 跨度 ≤ 2048 复数时，整段 x 一次 DMA + split 驻留 UB 平面，组内各列按 `(iS−xWin0)&7` 头扩展对齐切片（A 段同步头扩展、扩展行补零），彻底消除逐列 x 的重复搬运；跨度超限或非单位步长时回退逐列/逐元素路径。空列组（带段整体落空）整体跳过，不发 DMA。
- 每列点积：split 后 4 条 Mul + Add/Sub 合成乘积（T 减共轭项、C 加共轭项），pad 区标量清零后 `ReduceSum` 归约，标量读出进 sums 平面；满组后统一施加 alpha/beta（`beta=0` 不读 y），`incy=1` 时 Gather 交织 + 一次 DataCopyPad 批量写出整组 y，非单位步长走 UB 8-float 间距暂存 + 逐元素 DataCopyPad。

**单阶段回退路径（V1 行分核）**

行分核、每行段独立完成的通用路径：x 窗口按行段覆盖区间 UB 复用，A 按列连续段双缓冲 DMA，行内 FP32 分量累加。该路径同时承担三种职责：宽带宽（`kl+ku+1 > 1024`）回退、workspace 不足回退、以及**风险重算**——快 pass 逐列追踪 `max|a|·max|x|`，超过 2^126（逼近 FP32 乘积上溢的 fma 边界）时，该 tile 在 2^−129 缩放域重算（幂次缩放与舍入可交换，精确复现参考实现的融合乘加语义，含 Inf/NaN 溢出模式），保证极端数据下与 golden 逐位一致。

**负步长与通用访存**：逻辑下标 k 的物理偏移 `inc>0 ? k·inc : (len−1−k)·|inc|`（64 位计算，`|inc|` 先转 64 位防 INT_MIN 溢出）；x/y 的标量 GM 读、负步长 y 的暂存写出均按已验证的可靠路径（UB 暂存 + DataCopyPad，不使用不可靠的标量 GM 写）。

**复数 fma 边界处理**：参考 BLAS 以融合乘加计算（乘积不预先圆整），而矢量核无 fma 指令，普通乘加在 `|a·x|` 接近 FLT_MAX 时与参考的舍入/上溢边界可能不同。采用"风险追踪 + 缩放域重算"两级方案：正常数据走快 pass（乘积先圆整，参考顺序累加）；仅当 tile 内 `max|a|·max|x| ≥ 2^126` 时触发整矩阵回退重算，兼顾精度与性能。

### 性能优化方案

历轮保留的有效优化（全部为不改变同步语义的结构性优化）：

| 优化 | 机制 | 效果 |
| --- | --- | --- |
| 多列批处理 DMA | packed 时 K 列共享队列槽，握手/发射次数 ÷K | C 路径主要收益来源（4096² C 用例较单列 −10.5%） |
| T/C x 驻留 | 组级 x 一次载入 UB 平面，切片对齐复用 | 消除逐列 x 重复搬运 |
| 自适应核数 | N 路径 16/32/48 三档 | 256² N：16 核较 48 核 −28%；512² N：32 核较 24 核再 −1us |
| 窗口增量刷写 | workspace 高水位线，重叠段加、新段直写 | 避免子块间全窗读回 |

实测（910B2 事件计时，kernel 口径，3 次取末次）：256² N 23.6us、512² N 29.8us、1024² T 27.6us、2048² N 53.6us、4096² C 60.2us。910B3 官方 msprof Task Duration 口径复测为提交前剩余事项。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas 800I/T A2 | √ |
| Atlas A3 系列（含 Atlas 800I A3） | √ |

## 算子约束限制

- 仅支持 COMPLEX64；不支持超出 `lda/incx/incy` 语义的非连续内存访问；不涉及广播。
- `lda ≥ kl+ku+1`、`incx ≠ 0`、`incy ≠ 0`、`0 ≤ kl ≤ m-1`、`0 ≤ ku ≤ n-1`（m/n>0 时）。
- `y` 与 `A`、`x` 不允许内存重叠；结果原地写回，不返回视图。
- N 两阶段路径要求 `kl+ku+1 ≤ 1024` 且工作区可用，不满足时自动回退单阶段路径（语义不变）。
- 确定性计算不要求（浮点累加顺序不保证逐位一致，由混合容差标准覆盖）。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述(不涉及说明原因) | 标准来源 |
| --- | --- | --- |
| 精度标准 | golden 为 Netlib CBLAS `cblas_cgbmv` 单标杆；COMPLEX64 混合容差逐元素比对：rtol = atol = 2^-13（1.22e-4），matched_ratio ≥ 0.99，max_abs_error ≤ 1e-2（或 32·ULP，大数规约可放宽至 2ULP）；Inf/NaN 模式与 golden 逐位一致 | 任务书 §3.2 + 生态算子开源精度标准（mixed_tolerance_standard） |
| 性能标准 | Atlas 800T A2（910B3），msprof 口径，warmup 后有效采样 10 次取平均；5 个性能用例达标耗时：256²k16 N 10.665us、512²k32 N 17.328us、1024²k64 T 25.875us、2048²k128 N 30.640us、4096²k64,256 C 52.487us | 任务书 §3.3 |
| 内存标准 | 不涉及（工作区经 handle 工作区机制按需申请） | 任务书 §3.4 |

## 测试设计

| 层级 | 内容 |
| --- | --- |
| C++ UT（`test/gbmv/cgbmv/arch22/`） | 参数校验全量负向用例（空 handle/指针、非法枚举、越界 kl/ku/lda、零步长、负维度）；`m=0/n=0` no-op；`alpha/beta` 特殊值与 null 语义 |
| CSV 精度用例 | 任务包 `cgbmv_test.csv` 全量 1200 条：N/T/C × 带宽扫描（0、1、质数、2 的幂 ±1、非对齐）× lda padding × incx/incy ∈ {1,2,−1,−2,多值} × 均匀/正态分布 × Inf/NaN 特殊值，对齐 Netlib golden 全向量验证 |
| 性能用例 | 任务书 5 个 case + 任务包 TC_PF 用例 200 条，有效采样 10 次取平均 |
| 回归 | 形状扫描/矩形/对齐偏移/边界组合；行为与 cuBLAS 对齐 |

## 兼容性分析

`aclblasCgbmv` 为新增接口，不影响既有算子；`blas/gbmv/README.md` 产品支持表新增 A2/A3 支持项与 Cgbmv 接口说明，既有 arch35 Sgbmv 路径不变。
