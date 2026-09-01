# aclblasCherk 算子设计文档（Atlas A2/A3）

| 文档版本 | 日期 | 作者/团队 | 说明 |
| --- | --- | --- | --- |
| v1.0 | 2026-08-28 | m0_56676577 | 初始方案：三段式流水线 + 4M 分解 + uplo 三角 tile 裁剪 + MatmulImpl 主干 |
| v1.1 | 2026-08-28 | m0_56676577 | 内部评审修复：trans=N 同样需要转置副本、SCALE 上限统一 76、workspace 体量算式修正 |
| v1.2 | 2026-08-31 | m0_56676577 | 实现演进收口：KFC 高吞吐 GEMM 双路径、交错 4M 分解（虚部精度根因修复）、编排压缩 |

> **实现位置**：ops-blas 仓 `blas/herk/arch22/`（与上游 950PR `blas/herk/arch35/` 同目录共存）；测试 `test/herk/cherk/arch22/`。

# 需求背景（required）

## 需求来源

昇腾 CANN 社区任务 2026——「8月社区任务-aclblasCherk算子开发（A2/A3）」（任务编号 29）。要求在 Atlas A2/A3 系列产品（arch22 架构，Ascend 910B3）上使用 Ascend C 编程语言开发单精度复数（complex64）厄米特秩-k 更新算子 `aclblasCherk`，与 cuBLAS `cublasCherk` 功能对齐、精度对标 Netlib `cherk` 参考实现，验收通过后合入昇腾算子开源仓 https://gitcode.com/cann/ops-blas （算子目录 `blas/herk/`，测试目录 `test/herk/cherk/arch22/`）。

## 背景介绍

### cherk算子现状分析

`cherk`（Complex Hermitian Rank-K Update）是 BLAS Level 3 标准接口。上游 ops-blas 仓现状：

- Ascend 950PR / 950DT（arch35）：已有实现（`blas/herk/arch35/`，AIV 拆实虚 → 4×实数 GEMM（4M 分解）→ AIV 合并的三段式）；
- Atlas A2 / A3 系列（arch22 / DAV_2201）：**无实现**，`blas/herk/README.md` 产品支持表标注「不支持」。

arch22 无现成 GEMM 积木可复用，`blas/symm/arch22/ssymm_kernel.cpp`（1406 行）为 arch22 Cube GEMM 的关键参考。

### cherk算子功能分析

对标 cuBLAS `cublasCherk`（参数序列一一对应，无额外映射）与 Netlib `cherk.f`：

| 参数 | 参数含义 | 数据类型 | 约束 |
| --- | --- | --- | --- |
| handle | BLAS 上下文句柄 | aclblasHandle_t | 不可为 nullptr |
| uplo | C 的引用/更新三角 | aclblasFillMode_t | UPPER(121) / LOWER(122)，非法值报错 |
| trans | A 的操作类型 | aclblasOperation_t | OP_N(111) / OP_C(113)；OP_T(112) 不支持 |
| n | C 的阶 | int | n ≥ 0；n=0 quick return |
| k | 秩-k 更新的秩 | int | k ≥ 0；k=0 且 beta=1 quick return |
| alpha | 实数缩放标量 | float（Device 指针） | 不可为 nullptr |
| A | 输入矩阵 | complex64（列主序） | trans=N 时 lda ≥ max(1,n)；trans=C 时 lda ≥ max(1,k) |
| beta | C 缩放标量 | float（Device 指针） | 不可为 nullptr；beta=0 时 C 不必为有效输入 |
| C | 输入/输出矩阵（原地） | complex64（列主序） | ldc ≥ max(1,n) |

计算公式：`C = alpha·op(A)·op(A)^H + beta·C`（op 由 trans 决定）。C 原地覆写：仅 uplo 指定三角（含对角）被引用与更新，另一三角 bit 级不变；输出满足厄米特性，**对角元素虚部强制置 0**（cuBLAS/Netlib 口径）。

# 需求分析（required）

## 需求描述

使用 Ascend C 编程语言在 arch22（DAV_2201，AIC Cube + AIV Vector 混合核）上实现 `aclblasCherk`：complex64 输入输出、实/虚部分别 float32 精度判定、仅 uplo 三角更新、列主序原地、支持 lda/ldc 前导维 padding、完整边界与错误码语义；精度满足生态算子开源精度标准（FLOAT32 分量 rtol 2⁻¹⁰ / atol 2⁻¹⁶）；性能在任务书 3 条硬指标 case 上不高于标杆耗时（472.38 / 1674.42 / 311.00 us）。

## 需求拆解

1. **功能**：uplo(2) × trans(2) 全组合；n=0 / (alpha=0 或 k=0) 且 beta=1 quick return；alpha=0 或 k=0 且 beta≠1 仅按 beta 缩放；Inf/NaN 特殊值传播与 golden 一致。
2. **接口**：复用 `include/cann_ops_blas.h` 已有 `aclblasCherk` 声明（与其他产品线共用，禁止 A2/A3 私有平行 API）；错误码按任务书口径（非法 uplo/trans 枚举→`INVALID_ENUM`；OP_T、n/k<0、前导维违例、空指针→`INVALID_VALUE`；handle 空→`HANDLE_IS_NULLPTR`）。
3. **精度**：实部/虚部分别按 FLOAT32 标准判定（rtol=2⁻¹⁰=9.77e-4，atol=2⁻¹⁶=1.53e-5，matched_ratio ≥ 0.99 且 max_abs_error ≤ max(1e-2, 32×ULP)）；golden 为 cblas（Netlib cherk，对角虚部置零同口径）。
4. **性能**：case1 UPPER/N/1024/1024 ≤ 472.38us；case2 UPPER/N/2048/2048 ≤ 1674.42us；case3 LOWER/C/1024/1024 ≤ 311.00us（warmup 后有效采样 >50 次取平均）。
5. **测试交付**：ST（CSV 驱动，覆盖任务书全部自验要求）+ UT + 白盒 + PyTorch ST + 200 条性能回归 + 自测报告 + 测试 README。

# 详细设计（required）

## 算子分析

### 数学公式

$$
\text{trans}=N:\ C \leftarrow \alpha A A^{H} + \beta C\ (A\in\mathbb{C}^{n\times k}),\qquad
\text{trans}=C:\ C \leftarrow \alpha A^{H} A + \beta C\ (A\in\mathbb{C}^{k\times n})
$$

输出仅保留 uplo 三角，对角虚部置零：`Im(C_out[i][i]) = 0, ∀i`。

**4M 分解**（复数 GEMM 拆为实 fp32 GEMM；规避 3M 分解在大数相消场景的精度风险）：

- trans=N：`T1 = Ar·Ar^T`，`T2 = Ai·Ai^T`，虚部 `T3` 走交错归约（见下）；
- trans=C：`T1 = Ar^T·Ar`，`T2 = Ai^T·Ai`（仅 operand 指针换序）；
- 合并：`Cr = alpha·(T1 + T2) + beta·Re(C)`，`Ci = alpha·T3 + beta·Im(C)`。

**交错 4M 分解（虚部精度方案）**：经典 4M 分解 `Ci = t3 − t4` 在全负数据等场景存在同号块累加消减——k 项部分和 ~k·E[a]²，两路独立 GEMM 末端相减时灾难性消减（白盒 negative 档实测 matched_ratio 0.9722~0.9849，不达标）。修复为虚部单 GEMM 2k 交错归约：A2 交错复数行 × BCI 交错带符号行对，±项按 K 粒度 1 交替进入同一累加链，消减在累加内部抵消而非末端相减。实测 k=2048 虚部绝对误差 1.0e-3 → 2.2e-5（KFC）/ 6.9e-6（gemm4），matched_ratio 0.9848 → 0.9998+（47~148x 改善）。实部保持两路独立 GEMM，与经典 t1+t2 **bit 级一致**（0/2098176 差异），避免实部误差翻倍破坏 32×ULP 上限。

### 支持数据类型

| 参数 | 数据类型 |
| --- | --- |
| A / C | complex64（实/虚部各 float32），列主序 |
| alpha / beta | float32（实数标量，Device 内存指针） |
| 累加精度 | float32（UB/累加器均 fp32） |

### 支持形状

n、k 为运行时入参（不要求 dynamic shape 机制）：n ≥ 0、k ≥ 0，覆盖 0、1、小质数、2 的幂及 ±1、非对齐值直至大规模（测试覆盖至 n=k=2048）；A 支持 lda、C 支持 ldc 前导维 padding（紧凑及多档 padding 场景）。

## 算子实现

### 实现方案

handle 式 BLAS 算子（不走 aclnn 图算子框架），L2 接口层 host 编排 + L0 kernel 三段式流水线，GM workspace 中转，同一 stream 保序：

```text
GM 输入                AIV（前处理）              GM workspace              AIC（Cube 主计算）           GM workspace           AIV（后处理）        GM 输出
┌────────┐  Phase 0  ┌──────────────┐  Phase 1  ┌────────────────┐  Phase 2  ┌──────────────┐
│ A(cplx) │ ───────> │ deinterleave │ ───────> │ P/Q/A2/BCI(fp32)│ ───────> │ combine      │ ──> C(原地)
└────────┘  拆实虚    │ 拆分+交错布局 │  GEMM     │ T1/T2/T3(fp32)  │  合并缩放  │ 三角写回      │     仅 uplo 三角
                     └──────────────┘  双路径    └────────────────┘           └──────────────┘     对角虚部=0
```

- **Phase 0（AIV，deinterleave kernel）**：从 GM 按 (lda, 列主序) 2D strided 拷贝 A 到 UB，`DeInterleave` 拆实/虚部，产出 GEMM 所需行主序 operand：P/Q（Ar/Ai 平面）+ A2（交错复数行，虚部交错归约用）+ BCI（交错带符号行对）。trans=C 且行 32B 对齐时走 A-direct 路由（GEMM 直读 A 本身，跳过 A2 物化）。
- **Phase 1（AIC，GEMM，KFC/gemm4 双路径）**：实部两路独立 GEMM（T1/T2，bit 级等价经典 t1+t2）；虚部单 GEMM 2k 交错归约（T3，±项按 K 粒度 1 交替）。双路径路由见下表。
- **Phase 2（AIV，combine kernel）**：读 T1/T2/T3 → `Cr = alpha·(T1+T2) + beta·Re(C)`、`Ci = alpha·T3 + beta·Im(C)` → 对角虚部 `SetValue(0)` 强制置零 → 仅 uplo 三角按 (ldc, 列主序) 写回 C；LOWER 对角块从 C_old 回填非 uplo 部分（另一三角 bit 级不变）。兼走仅缩放（skipTemp）路径。
- **特殊值路径**：host 检测（非有限/可溢出标记经 FLAGS 段传递）→ kernel 参考仿真路径逐序 float32 累加 + `±0.0f×acc` epilogue，保证 Inf/NaN 放置与 golden（cblas）一致。
- **编排压缩**：alpha/beta Device 读回经独立 stream2 与 GEMM 重叠（隐藏 ~30us 同步延迟）；tileList 设备缓存 + 多级缓存，launch 编排开销 ~465us→~50us。

**GEMM 双路径（KFC / gemm4）**：

| 路径 | 实现 | 吞吐 | 路由条件 |
| --- | --- | --- | --- |
| KFC（高吞吐） | `cherk_gemm_kfc.cpp`，`REGIST_MATMUL_OBJ`/KfcServer 厂商混合编译 TU，收敛为虚部 Ci GEMM（tiling (n,n,2k)，dense 门控）+ 实部两路 | 47–60 TFLOPS | n ≥ 1536 且 -O3 且 tile 全对齐（n%baseM==0、n%baseN==0、k%32==0 等），36 块分批 launch（规避镜像 tile 列表单 launch 死锁） |
| gemm4（通用回退） | `cherk_kernel.cpp`，AscendC `MatmulImpl`（ND fp32）合并单 launch：g0/g1 实部两路（转置形式）+ g2 虚部 2k 交错归约（单 tile 形状 n≤256 时细分块 64） | ~1.5 TFLOPS（-O0）/ 数 TFLOPS（-O3） | 其余全部 shape（含非 16 对齐、小 n、低 k） |

任意 shape 均有正确路径可达（gemm4 兜底）；KFC 仅是性能加速路由，门控外 shape 自动回退。该取舍使非对齐 shape 相对 GPU 基线偏慢（见「可维可测分析」性能数据），属已知设计边界，正确性不受影响。

#### 3.2.1 host侧设计

`cherk_host.cpp`（L2 接口层）职责链：参数校验（任务书错误码口径）→ quick-return / 仅缩放分派 → tiling 计算 → workspace 规划 → KFC/gemm4 路由 → launch 编排。

##### 1. 分核策略：

- **AIC（GEMM）**：输出按 baseM×baseN tile 网格划分，只发射 uplo 三角覆盖的 tile 对（利用 Hermitian 对称性，有效算力需求减半：case2 约 20.5→10.3 TFLOPS）；对角带 tile 整 tile 计算，由 combine 只读 uplo 部分吸收；tile 列表（TILELIST 段，uplo 压缩镜像编码）设备缓存，各核 round-robin 认领，核间 tile 数差 ≤ 1；AIC/AIV 核数运行时 `GetAicCoreCount()/GetAivCoreCount()` 动态获取。
- **AIV（deinterleave / combine）**：deinterleave 按行切分；combine 为 tile 粒度 round-robin 分发。
- **多 launch 编排**：同一 stream 保序多次 launch（deint → gemm → combine）；GEMM 双路径由 host 按 shape 对齐性运行时路由。

##### 2. 数据分块和内存优化策略：

GM workspace 布局（每段 512B 对齐，TryMul/TryAdd 溢出防护累加；k=0 时不分配）：

```text
wsBase → [ T1 | T2 | T3 | P | Q | A2? | BCI | FLAGS | TILELIST ]
  T1/T2/T3: tempLdc × rowPad × 4B each     (tempLdc = rowPad = CeilAlign(n,16))
  P/Q:      n × kLdc × 4B each             (kLdc = CeilAlign(k,16))
  A2:       n × a2Ld × 4B                  (a2Ld = CeilAlign(2k,16)；A-direct 时为 0)
  BCI:      2k × tempLdc × 4B
  FLAGS:    aivCoreNum × 4B                (每 AIV 核一个非有限标记 float 槽)
  TILELIST: tilesPerDim² × 4B              (tilesPerDim = CeilDiv(n, tileMN))
```

UB 容量验证（DAV_2201，保守 184KB 口径）：deinterleave 64KB、combine 128KB（S=64），均 < 184KB。combine SCALE_BLOCK=64，UB 峰值实算上限 76（`8·S²·4B ≤ 184KB`）。

##### 3. tilingkey规划策略：

handle 式 BLAS 无设备侧 TilingKey 机制：host 计算 tiling struct 按值下发（deint/gemm/combine 各持独立轻量 struct，`cherk_tiling_data.h`）。分派维度 = uplo(2) × trans(2) × 路径(主 GEMM / 仅缩放) = 8 种组合，等价于 tilingkey 感知的分支分派；GEMM 路径（KFC/gemm4）由 host 运行时按 shape 对齐性路由（n≥1536 且对齐走 KFC，否则 gemm4）。分块参数：AIC 输出 tile baseM/baseN ∈ {128,128}/{256,256} 档（受 `kRightCubeUnifiedShape{256,256,256,128,128,64}` 约束）、K chunk ≤ 256；尾块经 `SetSingleShape` 精确下发实际行列数；kernel 入口 blockIdx 越界守卫。

#### 3.2.2 kernel侧设计

三类 kernel（`cherk_kernel.cpp` / `cherk_gemm_kfc.cpp`），各走 Init / Process（CopyIn → Compute → CopyOut）三阶段：

1. **deinterleave（AIV）**：2D strided CopyIn（A 按 lda 列主序）→ `DeInterleave` 拆实虚 + 交错布局 → 产出 P/Q/A2/BCI 四类 fp32 operand（A-direct 场景跳过 A2 物化）。
2. **gemm4（AIC，MatmulImpl 合并单 launch）**：Init-once/End-once 跨 4 组 operand 复用；g0/g1 实部两路转置形式 GEMM（T1/T2）+ g2 虚部 2k 交错归约 GEMM（T3）；单 tile 形状 n≤256 时细分块 64。
3. **KFC（AIC，混合编译 TU）**：`REGIST_MATMUL_OBJ` 收敛为虚部 Ci GEMM（tiling (n,n,2k)，dense 门控）+ 实部两路；36 块分批 launch。
4. **combine（AIV）**：读 T1/T2/T3 → 合并缩放公式 → 对角虚部置零 → uplo 三角 (ldc, 列主序) 写回；LOWER 对角块 C_old 回填保证非 uplo 三角 bit 级不变；skipTemp 路径仅按 beta 缩放。
5. **特殊值路径**：FLAGS 段非有限标记 → 参考仿真逐序 float32 累加 + `±0.0f×acc` epilogue，Inf/NaN 放置与 cblas golden 一致。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas 800I/T A2（Ascend 910B3，arch22/DAV_2201） | √ |
| Atlas A3 系列（arch22） | √ |

> 依赖 CANN 9.1.0；性能验收环境 Atlas 800T A2。

## 算子约束限制

- `trans = ACLBLAS_OP_T` 不支持（返回 `ACLBLAS_STATUS_INVALID_VALUE`），仅支持 OP_N / OP_C；
- C 为原地覆写（不返回视图），仅 uplo 指定三角（含对角）被更新，另一三角 bit 级不变；输出对角虚部强制为 0；
- A/C 列主序；lda/ldc 须满足最小约束（trans=N 时 lda≥max(1,n)、trans=C 时 lda≥max(1,k)、ldc≥max(1,n)），违例返回 `INVALID_VALUE`；
- alpha/beta 以 Device 内存指针传入（不可为 nullptr）；读回 Device 结果前须同步 stream（`aclblasSetStream` 绑定语义）；
- 接口声明复用 `include/cann_ops_blas.h` 已有 `aclblasCherk` 声明，与其他产品线共用，未新增平行 API。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | complex64 实部/虚部分别按 FLOAT32 分量判定：逐元素 `|actual−golden| ≤ atol + rtol×|golden|`，rtol=2⁻¹⁰(9.77e-4)、atol=2⁻¹⁶(1.53e-5)；用例通过 = matched_ratio ≥ 0.99 且 max_abs_error ≤ max(1e-2, 32×ULP)。golden 为 cblas（Netlib cherk，对角虚部置零同口径）。实测：ST 1220/1220（含实/虚分别比对 + 非 uplo 三角 bit 级断言）、UT 37/37、白盒 1044/1044、PyTorch ST 76/76 全部通过 | 社区任务任务书 §3.2；生态算子开源精度标准（opbase experimental_standard） |
| 性能标准 | 任务书 §3.3 三条硬指标（Atlas 800T A2 / CANN 9.1.0 / Release(-O3)，warmup 3 + 采样 60 取平均）：case1 UPPER/N/1024/1024 ≤ 472.38us → 实测 303–330us ✅；case2 UPPER/N/2048/2048 ≤ 1674.42us → 实测 1345us ✅；case3 LOWER/C/1024/1024 ≤ 311.00us → 实测 259.9–297.5us ✅（13 轮独立 bench 全 PASS）。200 条 TC_PF 参考性能回归（对照 A100 GPU 基线）：PASS=36 / SLOWER=164，SLOWER 集中于非 16 对齐 shape 回退 gemm4 路径与小 shape 固定开销（~100–130us）；16 对齐大方阵 NPU 反超 GPU 1.3–2.6× | 社区任务任务书 §3.3 |

## 兼容性分析

- 新增算子实现（arch22 目录级新增），不改动既有产品线行为；与 `blas/herk/arch35/`（950PR）同目录共存，构建按芯片架构分目录隔离；
- 接口层零改动：复用 `include/cann_ops_blas.h` 已有声明（第 542-546 行），与 Ascend 950PR 等产品线共用同一 API；
- `blas/herk/README.md` 产品支持表由「A2/A3 不支持」更新为「支持（arch22）」；
- 仓库根 `CMakeLists.txt` 新增 `OPS_BLAS_BUILD_TYPE` 环境变量开关（默认保持原 Debug 行为不变，性能基线需 Release/-O3），对既有构建无影响。
