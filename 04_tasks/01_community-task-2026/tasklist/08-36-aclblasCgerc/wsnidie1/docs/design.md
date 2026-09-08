# 需求背景（required）

## 需求来源

昇腾 CANN 社区任务 2026 年 8 月社区任务 —— aclblasCgerc 算子开发（950），任务列表编号 36。

## 背景介绍

### aclblasCgerc 算子实现优化

本算子为 ops-blas 仓 gerc 算子新增 arch35（Ascend 950PR）实现，基于 Ascend C 编程语言。gerc 此前仅有 arch22 实现（`blas/gerc/arch22/`），本设计面向 950PR 的 AIV 向量核特性全新实现连续访存场景的共轭秩-1 更新。

### 实现路径与 API 路径

- 算子实现路径：`blas/gerc/arch35/cgerc_host.cpp`、`blas/gerc/arch35/cgerc_kernel.cpp`、`blas/gerc/arch35/cgerc_tiling_data.h`
- API 路径：`include/cann_ops_blas.h` 中已有 `aclblasCgerc` 声明（禁止定义 950PR 私有平行接口）
- 测试路径：`test/gerc/cgerc/arch35/`（CSV 驱动 GTest 框架）

### aclblasCgerc 算子现状分析

| 参数 | 参数含义 | 数据类型 | 支持数据类型 | 约束 | 形状 |
| --- | --- | --- | --- | --- | --- |
| handle | ops-blas 库上下文句柄 | aclblasHandle_t | - | 指向已创建的有效句柄；nullptr 返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` | - |
| m | 矩阵 A 的行数 | int | - | m ≥ 0；m=0 为合法 no-op | - |
| n | 矩阵 A 的列数 | int | - | n ≥ 0；n=0 为合法 no-op | - |
| alpha | 复数标量乘数（只读） | const aclblasComplex* | COMPLEX64 | 不可为 nullptr；=(0,0) 为合法 no-op | 标量 |
| x | m 元复数向量（只读） | const aclblasComplex* | COMPLEX64 | m>0 且 n>0 时不可为 nullptr | 1+(m-1)·\|incx\| |
| incx | x 元素步长 | int | - | ≠ 0 且 ≠ INT_MIN（有意收紧，见§算子约束限制） | - |
| y | n 元复数向量（只读，计算取共轭） | const aclblasComplex* | COMPLEX64 | m>0 且 n>0 时不可为 nullptr | 1+(n-1)·\|incy\| |
| incy | y 元素步长 | int | - | ≠ 0 且 ≠ INT_MIN（有意收紧） | - |
| A | m×n 复数矩阵（原地更新） | aclblasComplex* | COMPLEX64 | m>0 且 n>0 时不可为 nullptr | lda×n（列主序） |
| lda | A 的主维度 | int | - | ≥ max(1, m) | - |

计算公式（Fortran 1-based 索引）：`A(I,J) ← A(I,J) + ALPHA · X(I) · CONJG(Y(J))`，I = 1..m，J = 1..n，A 列主序存储，原地更新。

### aclblasCgerc 算子功能分析

复数共轭秩-1 更新：以向量 x 与共轭后的向量 y 的外积乘 alpha 叠加进矩阵 A。与 geru 的唯一差异是对 y 取共轭。

- 输入：复数标量 alpha、复数向量 x/y（只读）、标量 m/n/incx/incy/lda
- 输出：复数矩阵 A（原地覆写，lda>m 时的 padding 区不修改）
- 支持数据类型：COMPLEX64（`aclblasComplex = {float real; float imag;}`，8 字节/元素）
- 支持广播：不涉及（x/y/A 为独立操作数）
- incx/incy 语义：任意非零整数（含负步长），负步长按 Netlib 从向量末端反向遍历，即第 col 个逻辑元素位于物理下标 `(n-1-col)*(-incy)`

# 需求分析（required）

## 需求描述

在昇腾 Ascend 950PR NPU 上使用 Ascend C 开发单精度复数共轭秩-1 更新算子 `aclblasCgerc`，实现与 cuBLAS `cublasCgerc` / Netlib BLAS `cgerc` 接口完全对齐的功能，精度按生态算子开源标准（`task_doc §3.2`：rtol=2⁻¹⁰、atol=2⁻¹⁶、matched_ratio≥0.99），性能如实报告。

## 需求拆解

1. 支持 COMPLEX64；语义严格对齐 Netlib cgerc：共轭只作用于 y，负步长从向量末端反向遍历
2. 支持 m/n ≥ 0 任意规格（含 m=0/n=0/alpha=(0,0) 的合法 no-op）、任意非零步长（INT_MIN 除外）、lda ≥ max(1,m) 的 padding 布局
3. 接口签名与 `include/cann_ops_blas.h` 已有声明一致，禁止定义 950PR 私有平行接口
4. 参数校验严格先于 quick return：handle → m/n ≥ 0 → alpha/incx/incy/lda → 指针（仅 m>0 且 n>0）→ no-op（`cgerc_host.cpp:210` 注释 `// Full parameter validation first, then quick returns (task book §2.5).`）
5. 性能如实报告：4 case 计时（口径见§性能标准），与 GPU 基线比较 ratio
6. 测试工程基于 ops-blas 仓 CSV 驱动 GTest 框架（1200 条用例），golden 由 cblas_cgerc 生成

# 详细设计（required）

## 算子分析

### 数学公式

**Fortran 1-based 索引**：

```
A(I,J) ← A(I,J) + ALPHA · X(I) · CONJG(Y(J))
  I = 1..m, J = 1..n
  ALPHA = (aR, aI)   复数标量
  X(I)  = (xR, xI)   复数向量，物理下标 = (I-1)*incx（incx<0 时从末端反向：(m-I)*(-incx)）
  Y(J)  = (yR, yI)   复数向量，物理下标 = (J-1)*incy（incy<0 时同理反向）
```

复数展开（记 w = alpha · x，即 wRe = aR·xR − aI·xI，wIm = aR·xI + aI·xR）：

```
A.re(I,J) += yR(J) · wRe(I) + yI(J) · wIm(I)
A.im(I,J) += yR(J) · wIm(I) − yI(J) · wRe(I)
```

本实现将 w 预折叠为两个交织布局更新向量 U/V（每行块构造一次，按列摊销）：

```
U[2i] = wRe[i], U[2i+1] = wIm[i]       （w 原序交织）
V[2i] = wIm[i], V[2i+1] = -wRe[i]      （w 旋转交织，共轭号在此）
A_交织(2cnt) += yr_j · U + yi_j · V     （2 条连续 Axpy 同时更新实部/虚部）
```

### 支持数据类型

| 数据类型 | 说明 |
| --- | --- |
| COMPLEX64 | 单精度复数，实部/虚部各 float32，内存布局 `{float real; float imag;}`，等价于交织 float 对，8 字节/元素 |

### 支持形状

- A 为 m×n 列主序矩阵（物理存储 lda×n，lda ≥ max(1,m)）
- x/y 为一维向量，物理长度 1+(len−1)·|inc|
- 不涉及广播
- 原地更新：A 既是输入也是输出，lda padding 区不修改

## 算子实现

### 实现方案概述

**核心思路**：A 列主序 → 每列在内存中连续，MTE 友好。为使 A 热路径上零 GM 侧 Gather/Scatter，将 alpha 与共轭全部折进 x 侧的两个**交织布局更新向量 U/V**（每行块构造一次，按列摊销），每列只需 2 条连续 `Axpy` 完成全长度更新。

**两种 A 访问模式**（host 决策，kernel 服从 `tiling.tilingKey`）：

| tilingKey | 模式 | 适用条件 | DMA 策略 |
| --- | --- | --- | --- |
| 1 | strip 多段 DMA | `lda==m && n>1 && m%4==0 && UB 装得下` | 每组一次多段 `DataCopyPad`（blockCount=gc, blockLen=cnt×8B） |
| 0 | legacy 逐列 | 其余所有形状 | 每列一次单段 `DataCopyPad` |

### host 侧设计（`cgerc_host.cpp`，261 行）

#### 参数校验与 no-op（L28-217）

校验链（严格先于 quick return）：
1. `handle != nullptr`（L206）→ `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`
2. `m >= 0`、`n >= 0`（L207-208）→ `ACLBLAS_STATUS_INVALID_VALUE`
3. `ValidateCgercParams`（L211，内部 L28-45）：alpha 非空 → incx/incy ∉ {0, INT_MIN} → lda ≥ max(1,m) → x/y/A 非空（仅 m>0 且 n>0）
4. Quick return（L215-217）：`m==0 || n==0 || alpha==(0,0)` → 返回 SUCCESS，**不读 x/y**

`INT_MIN` 收紧理由：`-INT_MIN` 是 C/C++ 未定义行为（signed overflow），kernel 用 `-incx` 做反向索引依赖此约束。已在 `blas/gerc/README.md` 登记。

#### Tiling 策略（`CalCgercRowTile`，L87-201）

**UB 容量查询**：`GetCachedUbSize()`（L59-70）通过 `PlatformAscendCManager::GetCoreMemSize(CoreMemType::UB, ...)` 查询实际 UB 大小，fallback 为仓内具名常量 `UB_SIZE = 248*1024 = 253952`（`blas/common/helper/kernel_constant.h:16`）。

**UB 上限**：`ubUsable = std::min<uint64_t>(ubSize, static_cast<uint64_t>(UB_SIZE))`（L138），确保 UB 预算不超过仓内常量上限。

**UB 预算不等式**（strip 模式，`ubSafetyMargin = 2*1024 = 2048 B`）：

```
52t + 8·sg·t + (128 + 64 + yCapBytes) ≤ min(ubSize, UB_SIZE) − 2048
```

其中 t = rowTile，sg = stripCols，yCapBytes = yCapFloats × 4。各项来源：

| 组成 | 字节公式 | 说明 |
| --- | --- | --- |
| cplxInQue（TQue<VECIN,2>） | 2×(8t+32) = 16t+64 | x 行块交织缓冲（2 槽） |
| planeBuf | 36t+32+yCapBytes+64 | wRe/wIm/negWRe 各 4t + U/V 各 8t + evenOff/oddOff 各 4t = 36t，加 yCache |
| stripBuf | 8·sg·t+32 | strip 列分组缓冲（sg 列 × t 行 × 2 floats × 4B） |
| 合计 | 52t + 8·sg·t + (128+64+yCapBytes) | — |

**实测数值验证**：2048²（t=2048, sg=7, yCapBytes=16384）：
`52×2048 + 8×7×2048 + (128+64+16384) = 106496 + 114688 + 16576 = 237760 B ≤ 253952 B`（余量 16192 B）

**rowTile 求解**（L144-162）：先取 `tStrip = min(mRounded, maxRowTile=2048)` 向下 64 对齐，搜索直到 `60·tStrip ≤ availBytes`（保证 sg≥1），再求 `sgStrip = (availBytes − 52·tStrip) / (8·tStrip)`，clamp `sgStrip ≤ scMax`。

**maxRowTile = 2048**（L112）：Gather 指令 count 上限为 4096，Axpy 已验证包络内取 2048 为安全上限。

**Tiling 结构体零初始化**（L237）：`CgercTilingData tiling{}` — POD 按值传入 kernel，未赋値字段否则会携带不确定值到达设备端。

**Tiling 决策日志**（L251-254）：`OP_LOGD("aclblasCgerc", "tiling: key=%d m=%d n=%d lda=%d rowTile=%u stripRows=%u stripCols=%u numBlocks=%u path=%s", ...)` — 提供 tiling 决策链的可观测性。

#### §3.2.1.1 分核策略

**AIV 侧核数确定**（L229）：`numBlocks = min(n, aivCoreNum)`，其中 `aivCoreNum = GetCachedAivCoreCount()`（L49-56，进程内缓存，首次调用 `GetAivCoreCount()`）。**零值防护**：若平台查询失败返回 0，fallback 为 1 并打印 OP_LOGE，避免 `CalCgercRowTile` 中 `scMax = ceil(n/numBlocks)` 除零。

**Kernel 侧列分配**（L115-128）：

```
colsPerCore = ceil(n / blockNum)              // L117，向上取整
start = blockIdx × colsPerCore                // L118
end = min(start + colsPerCore, n)             // L119-122，越界截断
if (start >= n) { start = 0; end = 0; }       // L123-126，闲置核
```

**4 case 实际取值**（aivCoreNum = 56）：

| case | m×n | numBlocks | colsPerCore | 闲置核 |
| --- | --- | --- | --- | --- |
| 1 (512²) | 512×512 | 56 | 10 | 8（56×10=560>512，末核分 2 列） |
| 2 (1024²) | 1024×1024 | 56 | 19 | 21（56×19=1064>1024） |
| 3 (2048²) | 2048×2048 | 56 | 37 | 19（56×37=2072>2048） |
| 4 (4096²) | 4096×4096 | 56 | 74 | 8（56×74=4144>4096） |

#### §3.2.1.1 尾块处理逻辑

**行块尾**（kernel L323/L381）：`cnt = (m − r0 < t) ? (m − r0) : t`

**列段尾**（kernel L339）：`gc = (colEnd_ − g0 < sg) ? (colEnd_ − g0) : sg`

**yCache 标量尾补齐**（kernel L181-196）：DMA 段按 32B 向下取整（`dmaFloats = yFloats / 8 * 8`），剩余不足 8 floats 的尾部逐元素 `SetValue` 补齐。

**strip 门控要求 `m % 4 == 0` 的原因**（host L164-168）：保证任意行块（含尾块 `cnt = m % rowTile`）的 DMA 段长 `cnt×8B` 恒为 32B 倍数。当 `m%4==0` 且 `t` 是 64 的倍数时，尾块 `cnt = m − floor(m/t)×t` 也满足 `cnt%4==0` → `cnt×8B` 为 32B 倍数。

**`m%4 ≠ 0` 的形状**（CSV 中 119 个，占 59.5%）：退回 legacy 逐列路径，正确性不受影响。Legacy 路径对 `cnt` 无 32B 对齐约束（单段 DMA 起始地址天然对齐）。

#### §3.2.1.2 BLOCK_SIZE（UB 容量取值来源）

代码中无名为 `BLOCK_SIZE` 的常量。UB 容量由以下两级确定：

1. **运行时查询**：`GetCachedUbSize()`（`cgerc_host.cpp:59-70`）→ `PlatformAscendCManager::GetCoreMemSize(CoreMemType::UB, cached)`
2. **仓内具名常量上限**：`UB_SIZE = 248*1024 = 253952`（`blas/common/helper/kernel_constant.h:16`）
3. **最终取值**：`ubUsable = std::min<uint64_t>(ubSize, static_cast<uint64_t>(UB_SIZE))`（L138）

安全余量：`ubSafetyMargin = 2*1024 = 2048 B`（L110），从 ubUsable 中扣除后作为可分配预算。

#### §3.2.1.2 BUFFER_NUM 与 UB 数据分块

**BUFFER_NUM = 2**：唯一的声明是 `TQue<TPosition::VECIN, 2> cplxInQue_`（kernel L77）。

- 承载内容：x 行块交织数据（BuildUV）与 legacy A 列块
- 生命周期：在**同一次迭代内** `AllocTensor → EnQue → DeQue → FreeTensor` 闭合（kernel L222-245），**不跨迭代流水**
- EnQue/DeQue 提供 MTE2→V 同步语义

**A 矩阵主缓冲 `stripBuf_`**（kernel L87）：单 `TBuf<TPosition::VECCALC>`，非 TQue，无双缓冲。

**完整 UB 布局表**（t = rowTile，sg = stripCols，yCapB = yCapFloats×4）：

| Buffer | 用途 | 尺寸公式 | 2048² 实际 | 4096² 实际 |
| --- | --- | --- | --- | --- |
| cplxInQue（2 槽） | x 行块交织 / legacy A 列块 | 2×(8t+32) = 16t+64 B | 32832 B | 32832 B |
| planeBuf.wRe | w 实部平面 | 4t B | 8192 B | 8192 B |
| planeBuf.wIm | w 虚部平面 | 4t B | 8192 B | 8192 B |
| planeBuf.negWRe | −wRe 暂存 | 4t B | 8192 B | 8192 B |
| planeBuf.uInt (U) | 交织更新向量 U | 8t B | 16384 B | 16384 B |
| planeBuf.vInt (V) | 交织更新向量 V | 8t B | 16384 B | 16384 B |
| planeBuf.evenOff | Gather 偶位字节偏移表 | 4t B | 8192 B | 8192 B |
| planeBuf.oddOff | Gather 奇位字节偏移表 | 4t B | 8192 B | 8192 B |
| planeBuf.yCache | y 整体缓存（2n floats） | yCapB+64 B | 16448 B | 32832 B |
| **planeBuf 小计** | — | 36t+32+yCapB+64 B | 90208 B | 106592 B |
| stripBuf | strip 列分组 A 缓冲 | 8·sg·t+32 B | 114720 B | 98336 B |
| **合计** | — | 52t+8·sg·t+(192+yCapB) B | **237760 B** | **237760 B** |
| UB 上限 | — | 253952 B | 余量 16192 B | 余量 16192 B |

> 2048²：t=2048, sg=7, yCapB=16384; 4096²：t=2048, sg=6, yCapB=32768

#### §3.2.1.2 double buffer

**strip 热路径是串行的，不存在 MTE2/V/MTE3 跨迭代重叠。**

strip 热路径每组内由两处 `PipeBarrier<PIPE_ALL>` 强制串行（kernel L350、L375）：

```
CopyIn（多段 DataCopyPad 读）→ PIPE_ALL → Compute（列循环 2×Axpy）→ PIPE_V → V_MTE3 事件 → CopyOut（多段 DataCopyPad 写）→ PIPE_ALL
```

`PIPE_ALL` 是**承载性**的：
- L350：保证 DMA 读完成后才开始 Axpy（MTE2→V 依赖）
- L375：保证写回排空后下一组读 DMA 才能复用 `stripBuf_`（WAR 防护）

降级为 `PIPE_V` 曾导致 **897 例 ret=5 EXECUTION_FAILED**（真机实测），因为 Gather 与 Axpy 非同管道，需要全管道屏障。

**合规例外**：strip 列循环后的 `PipeBarrier<PIPE_V>`（L362）仅覆盖纯 V→V 段（Axpy 序列），其后紧跟 `V_MTE3` 事件 Set/Wait（L365-367）才发 DMA 写，属合规的管道级同步。

`TQue<VECIN,2>` 的双槽仅用于 EnQue/DeQue 的 MTE2→V 同步语义，不构成跨迭代流水重叠。

#### §3.2.1.3 tilingKey 规划策略

`a95c644` 实现显式 `tilingKey` 字段（`cgerc_tiling_data.h:42-46`），照仓内 arch35 先例 `blas/rot/arch35/srot_tiling_data.h:16-17` / `srot_host.cpp:205,209` / `srot_kernel.cpp:330`。

| tilingKey | 路径 | 判定条件（host L168） |
| --- | --- | --- |
| 0 | legacy 逐列 | `lda≠m` 或 `n≤1` 或 `tStrip<64` 或 `sgStrip<1` 或 `m%4≠0` |
| 1 | strip 多段 DMA | `lda==m && n>1 && tStrip>=64 && sgStrip>=1 && m%4==0` |

Host 侧赋值（L250）：`tiling.tilingKey = (tiling.stripRows > 0U) ? 1 : 0`

Kernel 侧使用（L167-168）：`stripMode_ = (tiling_.tilingKey == 1) && (tiling_.stripRows > 0U) && (sc > 0U) && (tiling_.stripCols > 0U) && (tiling_.lda == tiling_.m)`

其中 `tiling_.lda == tiling_.m` 是**防御式自校验**：防止 `colStride = (uint64_t(tiling_.lda) - cnt) * 8` 在 `cnt > lda` 时 uint64 下溢。

### kernel 侧设计（`cgerc_kernel.cpp`，404 行，类 `CgercAIV`）

#### Init（L105-199）

1. **GM 地址绑定**（L109-112）：xGM_/yGM_/aGM_ 设为 `GlobalTensor<float>`
2. **列区间计算**（L115-128）：`GetBlockNum()`/`GetBlockIdx()` → colsPerCore → [colStart_, colEnd_)
3. **cplxInQue 初始化**（L133）：`InitBuffer(cplxInQue_, 2, cplxSlotBytes+32)`
4. **planeBuf 布局**（L138-151）：wRe/wIm/negWRe/U/V/evenOff/oddOff 七区按偏移分配
5. **偏移表生成**（L148-151）：`ArithProgression<int32_t>(evenOff, 0, 8, t)` / `ArithProgression<int32_t>(oddOff, 4, 8, t)`，单指令替代逐元素 SetValue（节省 ~4.7μs/launch）
6. **strip 模式判定**（L153-173）：解析 tilingKey/stripRows/stripCols → `stripMode_` → `InitBuffer(stripBuf_, ...)`
7. **yCache 加载**（L175-198）：`incy==1` 走 DMA+标量尾补齐；否则逐元素标量收集（正确性路径）→ `PipeBarrier<PIPE_ALL>`

#### Process — CopyIn → Compute → CopyOut

**Strip 模式**（L318-377）：

```
for each row block [r0, r0+cnt):
  1. BuildUV(r0, cnt)                      // 构造 U/V（每行块一次）
  for each column group [g0, g0+gc):
    2. 多段 DataCopyPad 读（L346-349）      // CopyIn：gc 段 × segBytes
    3. PipeBarrier<PIPE_ALL>（L350）
    4. 列循环 2×Axpy（L353-361）            // Compute：纯 Vector
    5. PipeBarrier<PIPE_V>（L362）
    6. V_MTE3 事件 Set/Wait（L365-367）
    7. 多段 DataCopyPad 写（L370-372）      // CopyOut
    8. PipeBarrier<PIPE_ALL>（L375）        // WAR 防护
```

**Legacy 模式**（L378-387）：

```
for each row block [r0, r0+cnt):
  1. BuildUV(r0, cnt)
  for each column col in [colStart_, colEnd_):
    2. ProcessTileLegacy(col, r0, cnt)      // 单列 DataCopyPad 读 → 2×Axpy → V_MTE3 → 写
```

**BuildUV**（L220-264）— 每行块执行一次，被所有列/列分组共享：
1. DMA 装载 x 交织块（incx==1 走 `DataCopyPad`，L224-228；跨步走标量收集，L230-239）
2. `Gather(wRe_, cplx, evenOff_, 0, cnt)` / `Gather(wIm_, cplx, oddOff_, 0, cnt)`（L242-243）— UB→UB
3. `PipeBarrier<PIPE_ALL>`（L244）— Gather 与后续向量操作非同管道
4. 复数乘法折叠 alpha：Muls/Sub/Add（L248-255）
5. `Scatter(uInt_, wRe_, evenOff_, ...)` / `Scatter(uInt_, wIm_, oddOff_, ...)` / `Scatter(vInt_, wIm_, evenOff_, ...)` / `Scatter(vInt_, negWRe_, oddOff_, ...)`（L258-262）— UB→UB 交织组装
6. `PipeBarrier<PIPE_ALL>`（L263）

#### 类型转换

**不涉及数据类型转换**，仅交织↔平面布局转换。全程 float32（COMPLEX64 = 2×float32 交织），无 cast、无精度升降。

### strip 列分组（stripCols）—— 本 PR 核心性能改动

**问题**（旧 tiling `00a9281`）：要求「本核全部 sc 列一次装进 UB」，大 n 时 `sc = ceil(4096/56) = 74` 把 rowTile 挤到 320，DMA 段长仅 `320×8 = 2560 B`，地址密度（segBytes / (segBytes + srcStride)）仅 **7.8%**（srcStride = (4096−320)×8 = 30208 B）。

**新方案**（`066cd60`，+108/−48 行）：将「段长」与「列分组宽度」**解耦**：
1. **先取尽量大的 rowTile**（决定段长 = rowTile×8B，最大化 DMA 连续传输量）
2. **再用剩余 UB 决定 stripCols**（每组搬多少列）
3. Kernel 按 stripCols 分组，每组一次多段 `DataCopyPad`

**效果**：

| case | rowTile | 段长 | srcStride | 地址密度 | 加速 |
| --- | --- | --- | --- | --- | --- |
| 3 (2048²) | 640→2048 | 5120→16384 B | 0（lda==cnt → **全连续**） | 100% | **2.22×** |
| 4 (4096²) | 320→2048 | 2560→16384 B | (4096−2048)×8 = 16384 B | 7.8%→**50%** | **3.31×** |
| 1/2 (512²/1024²) | 不变（已是单行块） | 不变 | stripCols==sc，行为等价 | — | 无变化 |

**收益归因**：访问连续性与地址密度的改善（`srcStride` 归零 / 密度 7.8%→50%）。

> ⚠️ **关于「段长↔带宽」因果关系**：后续 `f4b732e` 的对照组实验已证伪「DMA 突发长度本身带来收益」这一因果——该实验把 DMA 突发从 512 B 拉到 96 KB，用 `m%4!=0` 的 119 个形状（代码路径一行未变）作天然对照组，实测实验组 median 1.0203× vs 对照组 median 1.0300×（max 1.0560×）——**对照组更快**，故 2-3% 是跨运行系统性漂移。结论：**DMA 突发长度不改变有效带宽，「段长↔带宽」是相关性不是因果**。因此本文档**不将收益归因于“段长变长”本身**，而归因于可观测的访问连续性与地址密度改善（`srcStride` 归零 / 密度 7.8%→50%）。（依据：Cindy P4 注释改写 commit `7fe1d24`，已同步更新 host/kernel/tiling_data 三处注释）

**DMA 参数**（kernel L346-347，strip 读）：
```cpp
DataCopyExtParams inParams{gc, segBytes, colStride, 0, 0};
// gc=组内列数, segBytes=cnt*8, colStride=(lda-cnt)*8（块间空隙）, dstStride=0（UB 内连续）
```

**DMA 参数**（kernel L370-371，strip 写）：
```cpp
DataCopyExtParams outParams{gc, segBytes, 0, colStride, 0};
// srcStride=0（UB 内连续）, dstStride=colStride（GM 列间空隙）
```

### GM 侧与 UB 侧 Gather/Scatter 使用界限

**GM 侧 Scatter 已禁用**：两次独立实验证实 GM 侧 Scatter 在 arch35 上静默失败（写入数据丢失、无错误码），故所有 GM 侧块数据传输**统一走 `DataCopyPad`**（5 处：kernel L226(x读)/L276(A读,legacy)/L293(A写,legacy)/L349(A读,strip)/L372(A写,strip)）。定向 grep `Scatter(...aGM_/xGM_/yGM_)` **零命中**。

**UB 内部 Gather/Scatter 是可靠原语**：本实现在 `BuildUV` 内用它们在 UB 内完成交织↔平面布局转换：
- `Gather`（L242-243）：将 x 交织块拆为 wRe/wIm 平面，dst/src 均为 `planeBuf_.GetWithOffset` 所得 UB LocalTensor
- `Scatter`（L258-262）：将 w 平面交织组装为 U/V 更新向量，dst/src 同为 UB LocalTensor

这与 kernel 侧 BuildUV 中的 Gather/Scatter + 字节偏移表操作是**同一方案**，全程在 `planeBuf_` UB 内、不涉及 GM、不在 A 数据热路径。

### yCache 设计

**整体缓存**（非分块）：kernel L175-198 在 Init 阶段一次性将 y 向量全部 2n floats 加载到 UB（`planeBuf_` 内偏移 `36t+32` 处）。

- 启用条件：`n ≤ 8192`（host L232：`yCapFloats = (nUint <= 8192U) ? nUint*2U : 0U`）
- 关闭条件：`n > 8192` 时 `yCapFloats=0`，ReadY 走 GM 标量读
- 安全性：DMA 段按 32B 向下取整（`dmaFloats = yFloats/8*8`），尾部逐元素标量补齐

> **历史死路说明**：早期曾尝试「y 整体 UB 缓存」但因 `DataCopy` 8200B（非 32B 倍数）破坏 UB 对齐，精度从 997 跌至 556。当前版本修复了该问题（DMA 段 32B 向下取整 + 标量尾补齐），已真机验证通过全部 1200 用例。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas 950T A2 | |
| Atlas 950T A3 | |
| Ascend 950PR | √ |

## 算子约束限制

| 约束项 | 内容 |
| --- | --- |
| 参数合法性 | m ≥ 0、n ≥ 0；incx/incy ≠ 0 且 ≠ INT_MIN（`-INT_MIN` 是未定义行为，kernel 依赖 `-incx` 做反向索引）；lda ≥ max(1,m)；alpha 不可为 nullptr；m>0 且 n>0 时 x/y/A 不可为 nullptr |
| 非连续 Tensor 支持 | 向量通过 incx/incy 支持任意非零步长（含负步长，Netlib 反向语义）；不支持超出 inc/lda 语义的非连续内存访问 |
| broadcast 规则 | 不涉及 |
| dynamic shape 要求 | 不要求，m/n 为运行时入参 |
| 原地与视图语义 | A 原地覆写，不返回视图；lda padding 区不修改 |
| 确定性计算要求 | 不要求（`task_doc §2.5` 明文），但本实现核间列段 disjoint、无 atomic 竞争，结果实际确定 |
| 空 Tensor 与 0 维处理 | m=0 或 n=0 或 alpha=(0,0) 为合法 no-op，返回 SUCCESS 且不读 x/y |
| 异步执行 | 依赖 `aclblasSetStream` 绑定 stream；读回 Device 结果前须同步 stream |
| <32B GM 写 | `DataCopy`（非 Pad）不支持 <32B GM 写；`DataCopyPad` 会按 32B 补齐，legacy 逐列路径依赖此行为，已由 TC_SQ_007/008/009（m=1/2/3）与 58 条 `m%4≠0` 的 TC_EX/TC_PF 用例真机验证通过。strip 多段 DMA 因 UB 段间对齐要求，额外用 `m%4==0` 门控保证 `cnt*8B` 恒为 32B 倍数 |

# 可维可测分析

## 精度标准

### 验收口径（`task_doc §3.2`，以此为验收依据）

| 参数 | 值 |
| --- | --- |
| 逐元素判据 | \|actual − golden\| ≤ atol + rtol × \|golden\| |
| rtol | 2⁻¹⁰ = 9.765625e-4 |
| atol | 2⁻¹⁶ = 1.52587890625e-05 |
| required_matched_ratio | 0.99 |
| max_abs_error_limit | 1e-2 或 32×ULP（逐元素取 max） |
| golden | cblas_cgerc（Netlib BLAS 复数实现，列主序） |

**实测结果（`c1f8095`，含 TC_FL_104/105 修复）**：

| 测试集 | 结果 | 说明 |
| --- | --- | --- |
| 非 TC_PF（1001 条） | **1001/1001 PASS** | c1f8095 修复特殊值顺序对齐路径，TC_FL_104/105 已闭合 |
| TC_PF（200 条） | **200/200 PASS** | matched_ratio=0.99999999、mismatches=0 |

### 仓内 MERE/MARE 严格口径（非验收口径，仅作对照）

| 参数 | 值 |
| --- | --- |
| mere_threshold | 2⁻¹³ = 0.001221 |
| mare_multiplier | 10.0 |
| outlier_limit | 0.001221 |

**实测结果**：非 TC_PF **78 FAIL**（76 TC_EX + 2 TC_FL）+ TC_PF **11 FAIL** = **89 FAIL**

- 76 TC_EX：全部 `mismatches=0`，属 1-ulp 继承性离群（outliers 1-49 个/用例）
- 11 TC_PF：全部 1-2 个 outlier，MARE 0.0013-0.0034 擦着 outlier_limit 的边缘效应
- TC_FL_104/105：**Inf 传播顺序差异**（非 1-ulp，见下）

### TC_FL_104/105 归因与修复（`c1f8095` 已闭合）

**根因**：Inf 传播**顺序**差异（非 FMA 融合本身）。`task_doc §3.5.4` 要求「特殊值（Inf/NaN）行为对齐 cublas」，而 cblas 走 FMA 融合求值序、我方走两条独立 `Axpy`（先乘后加），`Inf×0` 与 `Inf−Inf` 的产生次序不同 → 范畴差异（`Inf−Inf=NaN` vs `Inf+Inf=Inf`），容差吸收不了。

**关键洞察**：真正的差异是运算「顺序」而非 FMA「融合」——在 Inf 算术里融合与否不改变结果，改变的是中间 NaN 的产生时刻。

**修复方案（`c1f8095`）**：BuildUV 后对 w 平面做非有限值归约置 flag，命中时走按 Netlib 括号化顺序的特殊值对齐路径。修复后 1001/1001 全通过。

### 89 条 MERE 口径失败「非本分支引入」证据

离线全量差分（`CalCgercRowTile` 精确复刻，1200 形状 × 6 种硬件参数组合）：

| 归属 | 计数 | 论证 |
| --- | --- | --- |
| S1（路径迁移） | **0** | 全 1200 形状仅 1 个发生迁移（TC_ED_120，n=−1 非法输入，验证阶段即拒绝，永不进 tiling，且不在 89 条内） |
| S3（bit-identical） | 48 | path+rowTile+stripRows+stripCols 逐位相同 |
| S2b（仅列分组变化） | 36 | rowTile 不变，仅 stripCols 0→N，算术序列不变 |
| S2a（rowTile 真变） | 5 | 其中 **4 条已由真机 A/B 闭合**：rowTile 从 640/1152/1856→2048 后 outlier/mismatch/元素总数逐条一字不变 |

唯一残留：`TC_PF_1119`（m=632, n=4064, rowTile 320→640）待一次定向真机基线确认（NEW 侧两次测量均为 1 outlier / 0 mismatches）。

5 份历史日志的 78 条失败用例名集合排序后 md5 全部相同（`2566a18b3d68fa3e87f4ad1e19ab5e55`），证实失败集跨版本稳定。

> ⚠️ **措辞边界**：已证明的是「89 条在本分支 OLD→NEW 之间指纹不变、0 条路径迁移」即**非本分支引入**；至于「与 master 分支外既有失败同源」需要 pre-branch master 的对比日志，磁盘上没有、离线证不了。

## 性能标准

### 达标线与判定式

```
ratio = gpu_ms / npu_ms ≥ 0.4        （verify_performance.py PERF_THRESHOLD = 0.4）
等价于 npu_us ≤ gpu_us / 0.4
```

| Case | m×n | gpu_ms（H100） | 达标线 (μs) | A 矩阵字节 | A 是否驻留 H100 L2(50MB) |
| --- | --- | --- | --- | --- | --- |
| 1 (TC_PF_1001) | 512×512 | 0.003470 | **8.675** | 2.0 MiB | ✅ 完全驻留 |
| 2 (TC_PF_1002) | 1024×1024 | 0.005140 | **12.850** | 8.0 MiB | ✅ 完全驻留 |
| 3 (TC_PF_1003) | 2048×2048 | 0.017853 | **44.633** | 32.0 MiB | ✅ 完全驻留 |
| 4 (TC_PF_1004) | 4096×4096 | 0.096434 | **241.085** | 128 MiB | ❌ 不驻留 |

出处四方一致：`gpu_baseline.csv:2-5`、`verify_performance.py:16,113-114`、`task_doc.md §3.3`、`FEEDBACK.md:67-68`。

### 实测性能（`c1f8095`，DevEnv_135409，strip 路径，perf4 三独立样本 median）

| Case | 实测 median (μs) | 达标线 gpu_us/0.4 (μs) | ratio | 判定 |
| --- | --- | --- | --- | --- |
| 1 (512²) | **43.963** | 8.675 | 0.197 | **FAIL** |
| 2 (1024²) | **62.960** | 12.850 | 0.204 | **FAIL** |
| 3 (2048²) | **129.599** | 44.633 | 0.344 | **FAIL** |
| 4 (4096²) | **452.551** | 241.085 | 0.533 | **FAIL** |

**结论：0/4 达标（官方口径）。** 但 roofline 分析证明该达标线物理不可达（见下文），且官方 `verify_performance.py` 口径本身存在数学缺陷（整毫秒解析，PASS 不可达）。

**与任务书 §3.3 标杆的直接对比**（非官方判定口径，仅供参考）：

| Case | 实测 (μs) | 标杆 (μs) | 比值 |
| --- | --- | --- | --- |
| 1 (512²) | 43.963 | 8.68 | 5.06× |
| 2 (1024²) | 62.960 | 12.85 | 4.90× |
| 3 (2048²) | 129.599 | 44.63 | 2.90× |

### 计时口径

`PERF avg_us` 是 **host `steady_clock` 计时**（`test/gerc/cgerc/arch35/cgerc_npu_wrapper.h` 约 L142-152），**不是** aclrtEvent 设备侧计时。

> ⚠️ 此前所有材料中「设备侧 aclrtEvent」的表述均为错误。确切的计时区间（是否每次调用同步、warmup=3/iters=51 如何处理、avg_us 是均值/中位数/最小值、含不含 golden 计算）<<待 Terry 核实后补完>>。

### 有效带宽与 roofline 占用

copy roofline 基准：**689.6 GB/s**（本轮在 DevEnv_910241 上用 ccopy 测 128 MiB 形状得到）。

| case | 有效带宽 (GB/s) | roofline 占用 | 是否触顶 |
| --- | --- | --- | --- |
| 1 (512²) | 296.9 | **43.0%** | 否 |
| 2 (1024²) | 515.2 | **74.7%** | 否 |
| 3 (2048²) | 706.3 | **102.4%** | **是（已超纯拷贝 roofline）** |
| 4 (4096²) | 659.4 | **95.6%** | **是** |

case3 已超过纯拷贝 roofline（部分数据被 UB/L2 复用）；case4 达 95.6%。两者在物理上已触顶，无优化空间。

### launch/调用底座

**F = 28.962 μs**（8 个极小形状实测中位数：8×8=28.297、4×4=28.334、1×1=28.417、16×16=28.787、2×2=29.136、32×32=29.332、64×64=31.388、39×63=38.256；median=(28.787+29.136)/2=28.9615）。

**F 的含义**：一次完整 `aclblasCgerc` 调用在小形状下的总耗时，含 host tiling 计算 + workspace 获取 + kernel launch + 设备执行 + sync，**不是单纯的 kernel launch 固定开销**。

case1 的时间 **67% 是 F**、case2 **47% 是 F**。即使 F=0，case1 也只到 ~14.1 μs、仍差 1.6×。case1 的达标线 8.675 μs 仅为 F 的 **30%** → 物理不可达。

### 官方 `verify_performance.py` 口径问题

脚本第 99-100 行解析 GTest 的 `[ OK ] ... (N ms)` **整毫秒墙钟**，而四个 case 的 `base_ms/0.4` = 0.00868/0.01285/0.04463/0.241 ms **全部 < 1 ms**，GTest 整数 ms 最小正值是 1 → `1 > 0.241` 恒成立 → **对任何实现该口径下 PASS 在数学上不可达**。

脚本自身第 137-138 行注脚承认：「GTest 输出耗时含 host 准备+kernel+golden 计算+比对，为保守上界；精确 kernel 耗时可配合 msprof 采集。」

实证：`evidence_20260905/msprof_m1.log:11-12` 同一 m=n=1 用例设备侧 32.203 μs 而 GTest 报 2 ms，**62× 偏差**。

### GPU 基线公平性分析

case1-3 的 A 矩阵 2.0/8.0/32.0 MiB **全部完全驻留在 H100 的 50 MB L2**。按「流量=2×A ÷ gpu_ms」反推 GPU 隐含有效带宽：

| Case | 隐含带宽 | 是否超 H100 HBM3 峰值 3.35 TB/s |
| --- | --- | --- |
| 1 | 1.21 TB/s | 否 |
| 2 | 3.26 TB/s | 否（逼近） |
| 3 | **3.76 TB/s** | **是** |
| 4 | 2.78 TB/s | 否 |

case3 超过 HBM3 物理峰值 → 基线测的是 **L2 带宽**（~12 TB/s、命中率 98-99%）。0.4 系数套在 L2 驻留基线上等价于要求 NPU 达 4.8 TB/s = 其 1.6 TB/s HBM 峰值的 **3 倍**；若公平地 HBM-to-HBM 则只需 1.34 TB/s。

### 相对改进（`066cd60` 列分组带来的收益）

相对 `2c7138d` 的 strip 基线（281.642 / 1481.177 μs），`066cd60` 的列分组多段 DMA 带来：

| case | 改进前 (μs) | 改进后 (μs) | 加速比 |
| --- | --- | --- | --- |
| 3 (2048²) | 281.642 | 126.593 | **≈2.22×** |
| 4 (4096²) | 1481.177 | 447.972 | **3.31×** |
| 1/2 | ~42-43 | ~43-63 | 无显著变化（launch 开销主导） |

> 注：后续 `f4b732e` 测得 123.978/436.028 μs，相对 `066cd60` 仅差 2.1%/2.7%，落在跨运行漂移带内，**不作为收益声称**。`c1f8095` 在 DevEnv_135409 的 perf4 median 为 **43.963/62.960/129.599/452.551 μs**（见§实测性能）。

### 已排除的方案（死路黑名单）

| 方案 | 结果 | 教训 |
| --- | --- | --- |
| `aclblasSgemm` 复用（K=2/ldb=2） | 7.8-8.2 ms（黑盒慢路径） | 库 GEMM 对极小 K 无优化 |
| strip-atomic（列分段+原子累加） | 52/80/442/2573 μs，慢 20-75% | 列分段丧失连续性 |
| TQue 队列深度 2→4 | 122.0/215.7/415.6 μs 劣化 | ⚠️ 此实验被误归档在 `archive_scnrm2/真机验证日志_R19_队列深度4.txt`，实为 cgerc 失败实验 |
| `PIPE_ALL`→`PIPE_V` 降级 | 897 例 ret=5 EXECUTION_FAILED | Gather/Axpy 非同管道 |
| 多列合并 tile B=9-16 | 劣化 2×、精度 287 失败 | UB 溢出 |
| 张量域替代标量 Axpy | 劣化 34% | 小张量域指令启动开销大 |
| y 整体 UB 缓存（早期尝试） | DataCopy 8200B 非 32B 倍数，精度 997→556 | 对齐问题（已修复，现版本安全） |
| OP_T 转置 | case1 8241 μs 且部分形状结果错 | 转置不适合此访存模式 |
| GM 侧 Scatter | 静默失败（两次独立实验证实） | arch35 硬件约束 |
| <32B GM 写（DataCopy 非 Pad） | 不支持 | 必须用 DataCopyPad |
| 2D DataCopy gap 语义 | gap 单位与预期不符 | 不采用 |

### 未纳入本 PR 的后续探索：CUBE GEMM 路径

**探索内容**：`construct(AIV) → 自写 CUBE GEMM(AIC) → 原子合并(AIV)`，试图利用 AIC 矩阵乘法单元加速大规格场景。代码曾合入（`cgerc_construct_lr_kernel`、`cgerc_cube_gemm_kernel`、`cgerc_atomic_add_kernel`、`namespace cgerc_cube`、`struct CgercGemmTiling`、`CgercTilingData` 的 `useCube`/`wsROff` 字段、`getenv("CGERC_CUBE")` 门控）。

**摘除理由**（本 PR `a95c644` 已将上述全部代码删除）：
1. **零测试覆盖**：1200 条 CSV 用例全部只跑 AIV，`grep -rn "CGERC_CUBE\|getenv" test/gerc/` 零命中
2. **零精度证据**：无任何版本的 CUBE 路径通过全量精度验证
3. **大 shape 因 workspace 门控静默回退**：`wsAvail` 实测 32 MiB，2048² 需 33,603,584 B（仅超 48 KiB）→ 从未真正执行 CUBE
4. **小 shape 比 strip 慢 16.6-43.5×**：CUBE case1=708 μs vs strip 42.7 μs
5. **带有 1 个严重 + 3 个重要确定性缺陷**（workspace 越界写穿共享 handle workspace 等）
6. **blockDim 与物理核数不一致**：host 侧 `cgerc_cube_gemm_do(..., 32U, ...)` 硬编码 blockDim=32，而 950PR 的物理 AIC 核数为 **28**（`npu-smi` 的 `Aicore Count`，DevEnv_135409 实测）。kernel 内 `GetBlockNum()` 返回的是**启动时传入的 blockDim（32）而非物理核数**，两者不可混用。后果：CUBE 的 grid-stride `totalTiles=32`、每 block 恰好 1 次迭代，而 32 个 block 落在 28 个物理核上会形成**两波调度**——第 1 波 28 个 block 并行、第 2 波仅 4 个 block 而其余 24 核空闲，墙钟约为单 block 时间的 **2 倍**。仓内既有 `GetAicCoreCount()`（`blas/common/helper/host_utils.h:89`，`cgerc_host.cpp:22` 已 include）与规范惯例（`blas/gemm/arch35/gemm_host.cpp:255-258`：查询 + 0 校验 + 返回 `ACLBLAS_STATUS_INTERNAL_ERROR`），同文件的 **AIV 路径本身是规范的**（用 `GetCachedAivCoreCount()`），故这是 CUBE 分支的遗漏而非设计意图。⚠️ 上述“两波调度 ≈2× 损失”是**基于 totalTiles=32、blockDim=32、物理核 28 的推算，尚未经 A/B 实测确认**（判别方法：把 blockDim 改为 28 后重测 512²/1024²，若耗时接近减半则成立）。

**若要重启需先解决**：(a) 1 个严重 + 3 个重要缺陷 + blockDim 硬编码；(b) 全量精度覆盖；(c) workspace 门控使大 shape 真正执行 CUBE；(d) 证明相对 strip 有实质收益。

### 硬件发现（950PR 实机调试经验）

| 发现 | 说明 |
| --- | --- |
| Gather 偏移单位为字节 | evenOff[i] = i×8（复数 re/im 各 4B） |
| GM 侧 Scatter 静默失败 | 两次独立实验证实；A 的 GM 写入统一走 `DataCopyPad` |
| UB 内 Scatter/Gather 可靠 | BuildUV 内 4 次 Scatter + 2 次 Gather 全部 UB→UB，已验证正确 |
| AIV 核数 | 由 `GetCachedAivCoreCount()` 查询，无硬编码；实测返回 56。**本 PR 交付路径（strip/legacy）仅用 AIV，分核规范** |
| AIC 物理核数 | **28**（`npu-smi` Aicore Count，DevEnv_135409 实测）。`GetBlockNum()` 返回的是 blockDim 而非物理核数，两者不可混用。CUBE 分支硬编码 blockDim=32 导致两波调度（已随 CUBE 摘除） |
| `DataCopyPad` 支持 <32B | 按 32B 补齐写入，legacy 路径依赖此行为 |
| `PipeBarrier<PIPE_ALL>` 承载性 | Gather 与 Axpy 非同管道，降级致 897 例崩溃 |
| `ArithProgression` | 单指令生成等差偏移表，替代逐元素 SetValue，节省 ~4.7μs/launch |

## 兼容性分析

新算子（gerc 族新增 arch35 分支），不涉及兼容性分析。接口声明复用 `include/cann_ops_blas.h` 中已有 `aclblasCgerc` 声明，可与其他产品线共用。

**测试方案说明**：基于 ops-blas 仓 CSV 驱动 GTest 框架（1200 条用例 + 1 条独立 NullHandle 测试），golden 使用 cblas_cgerc（Netlib BLAS 复数实现，列主序，`OPENBLAS_NUM_THREADS=1` 单线程）。测试类别覆盖：L0 基础(6)、SQ 尺寸(23)、AB alpha 特殊值(10)、RC 矩形(12)、LD lda padding(3)、INC 步长(36)、FL 填充含 Inf/NaN(18)、ED 边界负向(14)、EX 扩展(878)、PF 性能(200)。

**范围说明**：本 PR 仅含 AIV strip/legacy 路径（`tilingKey` 0/1）。整条 CUBE 路径已在 `a95c644` 中完全摘除（见上节理由），不在本 PR 交付范围内。
