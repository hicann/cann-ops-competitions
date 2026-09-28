# 需求背景（required）

## 需求来源

CANN 社区任务（2026）：aclblasStrsmBatched（A2A3）算子开发。基于 ops-blas 开源仓（https://gitcode.com/cann/ops-blas）工程框架，使用 Ascend C 在 Atlas A2/A3 系列（arch22，dav-2201，性能设备 Atlas 800T A2 / 910B3，CANN 9.1.0）上实现 BLAS Level 3 单精度批量三角求解算子 `aclblasStrsmBatched`。验收通过后算子代码合入 `blas/trsmbatched/arch22/`，测试代码合入 `test/trsmbatched/strsmbatched/arch22/`，接口声明新增至公共头文件 `include/cann_ops_blas.h`（供其他产品线共用）。

## 背景介绍

### 算子功能定位

`aclblasStrsmBatched` 为单精度实数（FLOAT32）批量三角线性方程组求解算子：对 batchCount 个相互独立、同规格的三角系统批量求解。每个 batch i 独立求解

```
side=LEFT:   op(A[i]) · X[i] = alpha · B[i]     （A[i] 为 m×m 三角阵）
side=RIGHT:  X[i] · op(A[i]) = alpha · B[i]     （A[i] 为 n×n 三角阵）
```

其中 `op(A)` 由 trans 决定：`OP_N → A`、`OP_T → A^T`（实数算子不支持 `OP_C`，传入返回 INVALID_VALUE）；A 为 uplo 指定的上/下三角阵，`DIAG_UNIT` 时对角视为 1（不读存储值）。

与 cuBLAS `cublasStrsmBatched` 的关键差异（任务书 §3.5-2）：**离席计算语义**——参数 1~11（handle 至 ldb）与 cuBLAS 一一对应，在 ldb 之后新增 `Carray/ldc`，cuBLAS 的原地覆写（解 X 覆写 B）映射为 B→C 离席写出（B 为只读输入）。batchCount==0 为合法 no-op（返回 SUCCESS）。

### 业界对标

- cuBLAS `cublasStrsmBatched`（https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-strsm-batched）：参数序列前 11 个一一对应；本算子扩展独立输出 C；
- Netlib BLAS `strsm` 参考实现（https://www.netlib.org/blas/strsm.f）：quick return、参数校验等行为依据；精度 golden 亦由 cblas（Netlib 实数实现）逐 batch 生成。

### 工程形态

ops-blas 仓 Ascend C kernel 直调模式：`aclblasStrsmBatched` 为句柄式 BLAS 接口，经 `aclblasHandle_t` 绑定 stream 直调 NPU kernel，异步执行。同目录已有 arch35（Ascend 950，就地 13 参旧语义）实现——本任务新增 arch22 目录，并将公共头文件中旧接口改名 `aclblasStrsmBatchedInplace` 保留向后兼容，15 参离席签名使用正名 `aclblasStrsmBatched`。

### 接口规格

```cpp
aclblasStatus_t aclblasStrsmBatched(
    aclblasHandle_t handle, aclblasSideMode_t side, aclblasFillMode_t uplo,
    aclblasOperation_t trans, aclblasDiagType_t diag,
    int m, int n, const float* alpha,
    const float* const Aarray[], int lda,
    const float* const Barray[], int ldb,
    float* const Carray[], int ldc,
    int batchCount);
```

| 参数 | 含义 | 类型 | 约束与异常行为 |
| --- | --- | --- | --- |
| handle | 库上下文句柄，携带 stream | aclblasHandle_t，Host | nullptr 返回 HANDLE_IS_NULLPTR |
| side | A 在左/右 | 枚举 | {LEFT(141), RIGHT(142)}；越界返回 INVALID_ENUM |
| uplo | A 的三角存储 | 枚举 | {UPPER(121), LOWER(122)}；越界返回 INVALID_ENUM |
| trans | 对 A 的操作 | 枚举 | {OP_N(111), OP_T(112)}；OP_C 返回 INVALID_VALUE；越界 INVALID_ENUM |
| diag | 对角类型 | 枚举 | {NON_UNIT(131), UNIT(132)}；越界返回 INVALID_ENUM |
| m / n | B 的行/列数 | int | ≥0；负值返回 INVALID_VALUE；任一为 0 → no-op SUCCESS |
| alpha | 标量乘数指针 | float32，Device | nullptr 返回 INVALID_VALUE；alpha==0 走 P0 置零快路径（不读 A/B） |
| Aarray | 三角阵指针数组，只读 | float32*[]，Device | 元素 nullptr（alpha≠0 且引用时）返回 INVALID_VALUE |
| lda | A 前导维 | int | ≥ max(1, side==LEFT?m:n)；否则 INVALID_VALUE |
| Barray | RHS 指针数组，只读 | float32*[]，Device | 元素 nullptr 返回 INVALID_VALUE |
| ldb | B 前导维 | int | ≥ max(1,m)；否则 INVALID_VALUE |
| Carray | 输出矩阵指针数组 | float32*[]，Device | 元素 nullptr 返回 INVALID_VALUE |
| ldc | C 前导维 | int | ≥ max(1,m)；否则 INVALID_VALUE |
| batchCount | 批数 | int | ≥0；==0 no-op SUCCESS；负值 INVALID_VALUE；上限 2^31-1 |

矩阵均列主序（Column-Major）。校验分级（CP1-Q6 口径）：handle/枚举/符号/alpha/指针数组级校验先于 no-op 判定；前导维校验后置（no-op 不引用矩阵，ld 不校验）。

### 数学公式

对每个 batch i（i = 0..batchCount-1）独立求解：

```
side=LEFT, trans=N:  A[i]   · X[i] = alpha·B[i]      X[i] = alpha·A[i]⁻¹·B[i]
side=LEFT, trans=T:  A[i]ᵀ  · X[i] = alpha·B[i]      X[i] = alpha·A[i]⁻ᵀ·B[i]
side=RIGHT,trans=N:  X[i] · A[i]   = alpha·B[i]      X[i] = alpha·B[i]·A[i]⁻¹
side=RIGHT,trans=T:  X[i] · A[i]ᵀ  = alpha·B[i]      X[i] = alpha·B[i]·A[i]⁻ᵀ
```

解 X[i] 写入 C[i]（m×n 列主，行距 ldc），B[i] 保持只读，C[i] 的 ldc padding 列不被写（canary 校验覆盖）。

# 需求分析（required）

## 需求描述

在 Atlas A2/A3 系列（arch22）上实现 `aclblasStrsmBatched`：支持 FLOAT32、side（LEFT/RIGHT）× uplo（UPPER/LOWER）× trans（N/T）× diag（UNIT/NON_UNIT）全组合、任意 m≥0/n≥0/batchCount≥0（含前导维 padding、no-op、alpha==0 置零快路径、OP_C 非法等边界负向）；精度满足生态算子开源精度标准（FLOAT32 混合容差，golden 为 Netlib cblas 逐 batch）；五典型性能 case 平均单次耗时不高于标杆（630.7/3062/5134/8680/14361 us，Atlas 800T A2 实测，warmup 后有效采样 >10 次取平均）；以 ops-blas 仓 CSV 驱动测试框架完成 1226 条随任务用例 + 26 条 SP/负向补充用例自验并交付报告。

## 需求拆解

1. **功能**：15 参离席签名（对齐 cuBLAS 前 11 参 + Carray/ldc 扩展）；全量参数校验分级（handle → 枚举 → 符号/alpha/指针 → no-op → 前导维）；alpha==0 P0 置零快路径（不引用 A/B）；OP_C 非法；batchCount==0 no-op；C 的 ldc padding 不写。
2. **精度**：golden = Netlib cblas strsm 逐 batch；FLOAT32 混合容差 rtol=atol=2^-13、matched_ratio≥0.99、max_abs_error≤max(1.2207e-3, 32×ULP)；覆盖小 shape/shape 扫描/填充模式/对齐偏移/边界负向/Inf-NaN 规格场景。
3. **性能**：五典型 case（bc×m×n = 64×256×256 L/L/N/NU、128×384×512 L/U/T/NU、32×1024×1024 R/L/N/U、8×2048×2048 L/U/T/NU、2×4096×4096 R/U/N/U）avg ≤ 标杆；aclrtEvent 计时（含 host 编排的保守上界）。
4. **交付**：设计文档（cann-competitions）、自测用例与测试代码（含 readme 复现步骤）、自测报告（参数+精度+性能+内存）、个人仓分支与算子 README（产品支持表标注 Atlas A2/A3：支持）。

## 关键挑战（本算子特有）

- **吞吐重心**：TRSM 是典型访存受限算子（三角依赖导致串行递推），batch 维并行是主要吞吐来源；LEFT/N 的大宽 RHS 与 RIGHT 的转置输出均存在布局适配问题（Cube GEMM 要求行主 ND）。
- **精度风险**：显式求逆（inv(A_ii)）+ GEMM 的 blocked 方案较逐步代入法引入额外规约误差，需验证满足混合容差。
- **边界复杂度**：uplo×trans×side×diag 16 组合的等价归一（forward/backward、effTransT）、非 8 对齐 lda/ldb/ldc 的搬运对齐、batch 间 workspace 独立。

# 详细设计（required）

## 算子分析

### 数学公式

blocked panel 分块视角（panelBs 记 bs，步 s 对应对角块行区间 [k, k+bs)，forward 型示例）：

```
第 s 步：  Y_panel(s) = Y_trail(s-1) 的面板列段
           X_panel(s) = −( −inv(op(A_kk)) · Y_panel(s) )        （INV_ALL 已前置产出 −inv）
           Y_trail(s) = Y_trail(s-1) − op(A_trail) · X_panel(s)   （GEMM TRAIL 累加）
终局：     X 的第 r 行 = −slot[s(r)] 对应行                      （FOLD_ALL 直写 C）
```

其中 Y=Xᵀ（LEFT 转置域；RIGHT 天然此布局）。INV_ALL 产出 −invᵀ（求 op(A_kk)ᵀ·M=I 后取负），APPLY 即 slot = Y_panel·(−invᵀ) = −X_panelᵀ。等价 LEFT 归一下四个 side×trans 组合的求根公式：

```
side=LEFT, trans=N:  X[i] = alpha·A[i]⁻¹·B[i]        side=LEFT, trans=T:  X[i] = alpha·A[i]⁻ᵀ·B[i]
side=RIGHT,trans=N:  X[i] = alpha·B[i]·A[i]⁻¹        side=RIGHT,trans=T:  X[i] = alpha·B[i]·A[i]⁻ᵀ
```

### 支持数据类型

| 类型 | 支持 |
| --- | --- |
| FLOAT32（本实现） | 是（任务书要求） |
| FLOAT16/BF16/复数 | 否 |

### 支持形状

m、n ≥ 0 任意（含 0/1/质数/2 的幂±1/非对齐）；batchCount ≥ 0（受 workspace 预算分批）；lda ≥ max(1,k)、ldb/ldc ≥ max(1,m)（含 padding 场景）；单 batch workspace 超 2GiB 返回 INVALID_VALUE，bc×perBatch 超 2GiB 自动 subBatch 分批。

## 算子实现

### 实现方案

#### 总体架构与数据流

host 侧按形状选 5 条路径（P0~P3），全程在 device workspace 上多 kernel 编排（同一 stream 串行保证依赖）：

```
P0 (alpha==0):   [ZERO]   C[i] = 0（不引用 A/B）
P1 (S=k*w≤4096): [SMALL]  单 AIV kernel 整阵列递推求解（零搬运冗余）
P2 (k≤64):       [SCALE→PANEL→AXPY]*steps → [OUT]
P3 (else):       [SCALE] → [INV_ALL] → { [GEMM APPLY] → [GEMM TRAIL] }*steps → [FOLD_ALL]
                  （LEFT+P3 且 lda 对齐门禁通过时走 Y=Xᵀ 转置域；否则 X 域 PACK 路径）
```

等价归一（host 完成，kernel 只见归一后参数）：RIGHT 通过 kDim/wCols 翻转 + SCALE/OUT 内联转置归一为 LEFT 同构；uplo×trans 组合归一为 forward（前代/回代方向）与 effTransT（trailing 读转置标志）。

#### 核心数据结构（P3 主路径，Y 域）

| 区域 | 布局 | 用途 |
| --- | --- | --- |
| wsB（Y） | wCols×kDim 行主 | RHS 工作域：Y = Xᵀ（转置域核心——SCALE 直拷入、GEMM 行主消费） |
| wsInv[s] | bs×bs × numSteps | 各步对角块逆的负阵（−invᵀ，按步开槽） |
| slot[s]（wsXneg） | wCols×bs × numSteps | APPLY 产物 = −Xᵀ 面板列段（按步开槽，FOLD_ALL 终局直读） |

#### 关键机制

1. **panel 分块**：kDim 按 panelBs（32/64/128 自适应档）分步，每步解一个 bs×bs 对角块 + 更新 trailing。
2. **INV_ALL 前置**（iter14i）：求逆仅依赖输入 A 对角块 → 全部步合并为单次发射（任务 = bc×numSteps 铺满 AIV），消除逐步发射的波次的占用率损失。
3. **GEMM 双模式**（AIC Cube，运行时 tiling）：
   - APPLY：slot[s] = Y_panel · (−invᵀ)（enAtomic=0 覆写，产出 −Xᵀ 面板列段）
   - TRAIL：Y_trail += slot[s] · op(A)_trailᵀ（enAtomic=1 累加；B 操作数直读 raw A——effT=1 经 SetTensorB(isTrans=true) 转置读，省 PACK）
4. **FOLD_ALL 直写**（iter14i）：X 的每行 r 属于且仅属于一个 panel 步 s(r)，解值 = −slot[s(r)] 对应行 → 终局单发射从各步 slot 取负直写 C，OUT/COMMIT 链全部消除。
5. **Y=Xᵀ 转置域门禁**（iter14/14j）：
   - effT=1（trans=T）：要求 lda 8-float（32B）对齐（Cube LoadData 转置读的物理行距约束）；
   - effT=0（trans=N）：额外要求 lda==kDim（B 操作数物理行距 = lda 而 tiling 按 kDim 寻址）；
   - 不满足 → 回退 X 域老路径（wsA PACK + COMMIT，iter13 实证配置），`needWsA` 与门禁两处同步计算。

#### host 侧设计

- `CalcStrsmbatchedPlan`：参数校验分级、路径分派、tiling 计算（panelBs 档位/colTile/numSteps）、workspace 布局与 subBatch 分批；
- `GenerateStrsmCubeTiling/T`：AIC 运行时 tiling 两份（主 tiling + B 转置读 tiling，orgShape 取 max(kDim,wCols) 覆盖所有 SetSingleShape 收缩）；
- kernel 发射序（Y 域 P3）：SCALE → INV_ALL → {APPLY, TRAIL(非末步)}×numSteps → FOLD_ALL；各步 blocks = min(核数, 任务数)，kernel 内 grid-stride 兜底。

#### kernel 侧设计（AIV）

- 统一 `strsmbatched_aiv_kernel` 按 stepType 分派：SCALE（B→Y 直拷+alpha）/ SMALL / PANEL+AXPY（P2）/ INV_ALL / FOLD_ALL / PACK_A+COMMIT（X 域回退路径）；
- 搬运规范（硬件实证规则）：GM 写全部走 MTE（DataCopyPad 2D/列式 stage 重排），禁 scalar store（多核丢写 ~50%）；4B 小块的 2D 搬运按 32B 槽打包；KbCopyPad2D/KbWritePad2D 的 UB 侧 stride 以 32B 为单位（cols 须 8 倍数，余数列走小 MTE 补拷）；
- workspace 写读跨 kernel 依赖以 kernel 边界（stream 序）+ KbSyncWriteToNextLoad（MTE3→MTE2 事件）保证。

#### 性能优化要点（终态，全部实测驱动）

| 优化 | 收益（实测） |
| --- | --- |
| Y=Xᵀ 转置域同构（消除 LEFT 标量跨步 I/O 与 PACK） | case2 23.2→4.5ms、case4 33.0→16.7ms |
| bs 自适应档（k≤256→32 / LEFT+T、RIGHT→64） | case4 16575→6807us |
| INV_ALL 单发射前置 | case1 658→581us（消波次占用损失） |
| FOLD_ALL 直写 C（OUT/COMMIT 链消除） | case4 6807→5299us（−22%） |
| 2D 块搬运（64 列带 MTE）SCALE/OUT | case2 首达 2954→2485us |

## 支持硬件

| 产品 | 支持情况 |
| --- | --- |
| Atlas A2 训练/推理系列（含 800I/800T A2） | 支持（910B3 实测：精度 1226/1226、五 case 性能全达标） |
| Atlas A3 训练/推理系列（含 800I A3） | 支持（ascend910_93* 同归 arch22/dav-2201） |
| Ascend 950PR/DT（arch35） | 不支持（该架构请用 aclblasStrsmBatchedInplace） |

## 算子约束限制

1. 实数算子：trans 不支持 OP_C（返回 INVALID_VALUE，无共轭语义）；
2. 单精度 FLOAT32 专属实现（精度按 FLOAT32 混合容差判定）；
3. workspace 单 batch 超 2GiB 返回 INVALID_VALUE（P3 Y 域 slot 按步开槽的开销上限场景：kDim=4096/wCols=4096/bs=128 时 slot 区 32×128×4096×4B≈64MB/batch）；
4. kernel 侧固定假设 TaskRation=1、SubBlockIdx=0（910B3 实测），AIV grid 按 blockNum 任务分发。

# 可维可测分析（required）

## 精度标准/性能标准

**精度标准**（生态算子开源精度标准，FLOAT32 混合容差；golden = Netlib cblas strsm 逐 batch）：

| 项 | 值 |
| --- | --- |
| rtol / atol | 2^-13（1.2207e-4） |
| required_matched_ratio | 0.99 |
| max_abs_error_limit | max(1.2207e-3, 32×ULP) |
| 逐元素条件 | \|actual−golden\| ≤ atol + rtol×\|golden\| |

**测试框架**：ops-blas 仓 CSV 驱动 GTest（1226 条随任务用例 + 26 条自补 SP/边界）；ldc padding canary 校验（0xCAFEBABE 哨兵不被覆写）；覆盖面：side/uplo/trans/diag 全正交、m/n/bc 的 0/1/质数/2 幂±1/非对齐、lda/ldb/ldc 紧凑与 padding、alpha 均匀/正态/特殊值（0/1/−1/null）、Inf/NaN 传播、全部负向（null 指针/非法枚举/负维度/非法前导维）、A/B/C 缓冲重叠（OverlapCA/CB）。

**性能标准**（任务书 §3.3）：Atlas 800T A2 实测五典型 case 平均单次耗时不高于标杆；采集口径 warmup 5 + 有效采样 >10 次（env STRSM_PERF_SAMPLES，本次报告取 20）取平均，aclrtEvent 事件夹逼（含 host 编排开销的保守上界）。

**实测结果**（Atlas 800T A2，CANN 9.1.0，2026-09-25）：

| 项 | 结果 |
| --- | --- |
| 全量 CSV 精度 | **1226/1226 PASS（覆盖率 100%，FAIL 0）** |
| SP/负向补充批 | 37/37 PASS |
| 五典型性能 | 全部达标（565.3/2484.8/3638.9/5216.5/9141.4 us，比值 0.60×~0.90×） |
| Host 峰值 RSS（VmHWM） | 五 case 175~975 MiB（随 bc×m×n 线性） |

## 兼容性分析

**接口兼容**：

| 接口 | 语义 | 兼容策略 |
| --- | --- | --- |
| aclblasStrsmBatched（15 参，arch22 新增） | 离席：B 只读、解写 C（对齐 cuBLAS 前 11 参 + Carray/ldc 扩展） | 声明入公共头 `include/cann_ops_blas.h`，供其他产品线共用 |
| aclblasStrsmBatchedInplace（13 参，arch35 旧） | 就地覆写 B（Ascend 950 旧语义） | 公共头中原名改 Inplace 保留，arch35 实现同步改名，向后兼容 |

**数据类型/形态兼容**：仅 FLOAT32；列主序；ldc padding 列与 B 全程不被写（canary 实证）；Inf/NaN 按 IEEE 传播（规格场景验证）。

**回退路径兼容**：LEFT+P3 在 lda 不满足转置域门禁（非 8 对齐，或 trans=N 时 lda≠kDim）时自动回退 X 域 PACK 老路径——对外接口行为不变，仅性能差异；两条路径均被 1226 条全量回归覆盖。

**跨产品兼容**：Atlas A2/A3 同归 arch22（dav-2201）共用本实现；A3 侧 SOC_VERSION（ascend910_93*）经仓内 CMake 映射直达 arch22 目录；Ascend 950（arch35）不适用本目录（README 产品支持表已标注）。

## 日志与可观测

OP_LOGI 记录 plan（路径/形状/档位）与 launched（subBatches/steps）；OP_LOGE 覆盖全部分级校验失败分支（带非法值回显）；性能路径输出 StrsmPerf 标准行（case/参数/warmup/samples/avg/min/max/cv）。

## 维护性说明

- 路径分派与档位选择集中在 host `CalcStrsmbatchedPlan`（含实测依据注释）；kernel 按 stepType 单入口分派，Y 域与 X 域回退路径隔离清晰；
- 已知不采纳方案留档：同 kernel 双 IterateAll 合并（mode3，EX_0510/0642 竞争）、INV 列条带切分（iter6b）、GEMM N 块并行 Y 域版（iter14c）——均有实证数据，防止回退复踩。
