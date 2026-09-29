# aclblasSsbmv 算子设计文档（社区任务 PR 提交版）

> **算子**：`aclblasSsbmv`（单精度实数对称带状矩阵-向量乘，BLAS-2）
> **任务**：昇腾社区任务——aclblasSsbmv 算子开发（Atlas A2/A3）
> **目标架构**：arch22 / DAV_2201（Ascend 910B3，Atlas A2/A3），实现位于 ops-blas 仓 `blas/sbmv/arch22/`，从零新建（上游仅有 arch35 存根，仅作语义参考）
> **基线接口**：cuBLAS `cublasSsbmv`（语义口径对齐 Netlib BLAS `ssbmv.f`）

## 修订记录

| 版本 | 修订内容 | 修订时间 | 修订人(gitId) |
| --- | --- | --- | --- |
| v1.0 | 初始方案设计：三路径（AlphaZero / 列窗流式主路径 / 通用标量 fallback）+ UPPER/LOWER 模板双实例 + 对称双引用分解 | 2026-09-24 | potatomato111 |
| v1.1 | 实现演进收口：驻留对角形态与带首瓦片对角形态落地、16 行核切分粒度、动态行瓦片 TH 与对齐列粒度 colGran 门控，精度与性能全量达标 | 2026-09-25 | potatomato111 |

# 需求背景（required）

## 需求来源

昇腾社区任务（cann-ops-competitions tasklist）：在 Atlas A2/A3 系列产品（Ascend 910B3 / CANN 9.1.0）上使用 Ascend C 以 kernel 直调方式开发单精度对称带状矩阵-向量乘算子 `aclblasSsbmv`，基于 ops-blas 开源仓工程框架实现句柄式 BLAS 接口，完成设计、开发、测试全流程；验收通过后合入 `cann/ops-blas` 仓 `blas/sbmv/arch22/`。

## 背景介绍

### ssbmv算子实现现状分析

ops-blas 上游仓 `blas/sbmv/` 目录当前仅有 Ascend 950PR（arch35）的 SIMT 实现存根；面向 Atlas A2/A3 的 arch22（DAV_2201）路径**无任何实现**，host / kernel / tiling / 测试链均需从零新建。接口声明已存在于公共头文件 `include/cann_ops_blas.h`（与其他产品线共用，零改动），本任务不得定义产品私有平行 API。

### ssbmv算子功能现状分析

对标 cuBLAS `cublasSsbmv`，语义参考 Netlib `ssbmv.f`。功能要素：

| 参数 | 含义 | 数据类型 | 值域/约束 | 异常行为 |
| --- | --- | --- | --- | --- |
| uplo | 存储三角（UPPER/LOWER） | attr(int) | 两枚举 | 非法枚举 → INVALID_ENUM |
| n | 矩阵阶数（=向量逻辑长度） | scalar(int) | n ≥ 0 | n < 0 → INVALID_VALUE；n=0 合法 no-op |
| k | 半带宽 | scalar(int) | k ≥ 0 | k < 0 → INVALID_VALUE |
| alpha/beta | 标量乘数指针（Host 内存） | FLOAT32 | 全集 | nullptr → INVALID_VALUE |
| A | 对称带状矩阵，列主序压缩存储，仅存 uplo 三角带状部分 | tensor(FLOAT32) | 取值 FLOAT32 全集 | n>0 且 alpha≠0 时 nullptr → INVALID_VALUE |
| lda | 前导维度（存储行数） | scalar(int) | lda ≥ k+1 | lda < k+1 → INVALID_VALUE |
| x / y | 输入/输出向量（y 原地覆写） | tensor(FLOAT32) | 取值 FLOAT32 全集 | nullptr → INVALID_VALUE |
| incx / incy | 步长（支持正/负） | scalar(int) | ≠ 0 | = 0 → INVALID_VALUE |

计算公式：`y = alpha * A * x + beta * y`；handle 为空 → HANDLE_IS_NULLPTR。

# 需求分析（required）

## 需求描述

使用 Ascend C 编程语言（kernel 直调）实现 `aclblasSsbmv`：单精度实数对称带状矩阵-向量乘，支持 UPPER/LOWER 双存储布局、任意非零步长（含负步长）、BLAS 标量分支与边界语义，精度满足生态算子开源精度标准 FLOAT32 档，性能不低于 GPU 基线折算标杆。

## 需求拆解

1. **功能语义**：`y = alpha*A*x + beta*y`，A 为 n×n 对称带状矩阵（仅存 uplo 指定三角的带状部分，未存储半带按 A=A^T 隐含引用），y 原地覆写不返回视图；
2. **参数校验与错误码**：handle 判空最先；uplo 非法枚举返回 INVALID_ENUM；n<0 / k<0 / lda<k+1 / incx=0 / incy=0 / alpha、beta 指针空 / 条件判空（a、x 仅在 n>0 且 alpha≠0 时要求非空，y 在 n>0 时要求非空）返回 INVALID_VALUE；
3. **边界行为**：n=0 合法 no-op（不引用任何 Device 指针）；alpha=0 时不引用 A 与 x，beta=1 时 y 位精确不变、beta=0 时置零、其余 beta 逐元素缩放（EXACT 位精确）；beta=0 时不读取 y 旧值；k≥n 退化全带按有效带宽 min(k, n−1) 正常计算；Inf/NaN 输入不检测不拦截，按 IEEE 浮点沿带内点积传播；
4. **精度**：golden 由 cblas（Netlib 实数 ssbmv）生成，全向量逐元素验证；FLOAT32 档混合容差 + EXACT 特例（见可维可测分析）；
5. **性能**：kernel-only 计时口径，全量 200 条性能 case 平均单次耗时不高于 GPU 基线折算标杆（含任务书 §3.3 四条锚点 case）；
6. **工程化**：接口签名以 `include/cann_ops_blas.h` 已有声明为准零改动；测试链按 ops-blas 仓 test 目录 CSV 驱动 GTest 形态新建于 `test/sbmv/ssbmv/arch22/`；README 产品支持表标注 Atlas A2/A3 支持。

# 详细设计（required）

## 算子分析

### 数学公式

逐元素公式（对每个输出行 i，0 ≤ i < n，kEff = min(k, n−1)）：

```
y_i = alpha * Σ_{|i-j|≤kEff} A(i,j) * x_j + beta * y_i_old
```

带内每个存储元素经对称双引用对 y(i) 与 y(j) 各贡献一次乘加；带外（|i-j| > kEff）取 0；k ≥ n 时天然截断为对称满矩阵语义。存储元素引用公式（0-based，列主序 a[lda*n]）：

| uplo | 引用公式 | 有效范围 | 存储行布局 |
| --- | --- | --- | --- |
| UPPER | A(i,j) = a[j*lda + (k+i-j)] | 0 ≤ j-i ≤ k | 主对角线位于存储行 k，第 d 条超对角线位于存储行 k-d |
| LOWER | A(i,j) = a[j*lda + (i-j)] | 0 ≤ i-j ≤ k | 主对角线位于存储行 0，第 d 条次对角线位于存储行 d |

**对称双引用分解**（实现核心）：将存储半带视为通用带状阵 B，则 `A = B + Bᵀ − diag(B)`，对每个输出行 i 分解为两族结构：

- **N 项**（目标行点积，反对角访存）：`N(i) = Σ_{j=max(0,i-kEff)}^{i} B(i,j)·x_j`；
- **T 项**（列 run 自点积，列内连续访存）：`T(i) = Σ_{d=1}^{min(kEff, n-1-i)} B(i+d,i)·x_{i+d}`（主对角线仅计入 N 项一次，无双计）。

向量步长映射（对齐 Netlib ssbmv）：正步长逻辑 i ↔ 物理 i·|inc|；负步长逻辑 i ↔ 物理 (n−1−i)·|inc|；|inc|>1 时物理间隙元素不属于逻辑向量。

### 支持数据类型

仅 float32（a/x/y 全 float32 单组合），累加器 float32，无 dtype 分支。

### 支持形状

n 为运行时 int 入参（规格口径 n ∈ [0, 65536]，测试覆盖至 n=4096）；A 逻辑长度 lda·n（lda ≥ k+1，padding 行不引用）；x/y 逻辑长度 n（负步长物理长度 ≥ 1+(n−1)·|inc|）；A 偏移全程 uint64 核算。无广播（独立操作数）。

## 算子实现

### 实现方案

#### 3.2.1 host侧设计：

**校验序（固定）**：handle 判空 → uplo 枚举（INVALID_ENUM）→ n<0 / k<0 / lda<k+1 / incx=0 / incy=0 / alpha、beta 指针空（INVALID_VALUE）→ a/x 条件判空 → y 判空 → quick return（n=0 直接 SUCCESS 不 launch kernel；alpha=0 且 beta=1 直接返回，y 位精确不变）→ 标量读取 → kEff = min(k, n−1) 截断 → 路径判定 → tiling 计算与 launch（tiling 参数按值传入 kernel，stream 取 handle 绑定 stream）。

**tiling策略**：TilingData 为 Host→Kernel 按值直传 POD 结构，承载 n/kEff/lda/incx/incy/alpha/beta 及路径与切分字段（path / useCoreNum / rowsPerCore / th / wc）。

##### 1. 分核策略：

- 按输出维 y 切分：incy==1 且输出规模足够时按输出行分配到多 AIV 核，各核处理互不重叠的行区间；核数上限取 `GetAivCoreCount()` 运行时返回值（禁止硬编码）；跨步/负步长场景单核处理。
- **每核行数恒为 16 行（64B）的整数倍**：多核并发写回 GM 时相邻核写同一条 64B cache line 会互相覆盖，16 行粒度使核边界永远落在 cache line 边界，消除该风险。
- **每核行数下限**：驻留子形态 64 行、流式子形态 32 行（低于下限时增加核数只增加派发开销不缩短关键路径），`useCoreNum = min(ceil(n/行数下限), aivCoreNum)`，`rowsPerCore = ceil16(ceil(n/useCoreNum))`；尾块非对齐行由 DataCopyPad rightPadding 处理。
- **行瓦片高度 TH = clamp(align32(kEff), 128, 2048)**（无浪费瓦片：TH ≈ kEff 时列窗加载放大率 ≈ 1.0；小 k 取 128 下限）。

##### 2. 数据分块和内存优化策略：

- **A 列窗流式 + 双缓冲**：每行瓦片需要引用的 A 列窗按 64KB 块分块经 `DataCopyPad` GM→UB 流式搬运，两块 A 区域交替 `SET_FLAG`/`WAIT_FLAG`，实现 MTE2 搬运与向量计算重叠（大 k 域主导场景的流水收益）。
- **整窗驻留子形态**：A 窗实际搬运量（含对齐扩载）≤ 128KB 时一次性驻留 UB，免除块间流水开销；该子形态由 kernel 端以既有 tiling 字段运行时判定，不新增路径与字段。
- **GM 32B 对齐保证**：A 连续块 GM 起点（列号 × lda × 4B）须 32B 对齐，由对齐列粒度 `colGran = 8/gcd(lda, 8)` 保证——窗起点向下对齐到 colGran 倍数、块宽 wc 取 colGran 的倍数（工程上取 8 列倍数），流式步进后每块起点恒对齐；扩载列不参与贡献。
- **UB 192KB 静态偏移布局**：累加器/旧 y/x 窗/双缓冲 A 块/中间暂存/取数偏移表等区域背靠背静态布局，偏移全部 32B 对齐，`static_assert` 逐项核算总量 ≤ 192KB 防重叠；不使用 L1/Cube（AIV 向量核），无 workspace（kernel 直调）。
- **超限兜底**：kEff > 2047 或 lda 超出主路径对齐门控（lda ≤ 2048）的组合不进入主路径，走通用标量路径保证正确性（该域仅出现在精度用例，无性能要求）。

##### 3. 路径分派策略（kernel 直调，无 TilingKey）：

本算子为 ops-blas 句柄式 BLAS kernel 直调形态，tiling 参数按值传入 kernel，无编译期 TilingKey 注册体系；路径分发由 host 侧判定后写入 tiling 标量字段 `path`，kernel 内运行时分支，共三条路径：

| 路径 | 触发条件 | 覆盖 |
| --- | --- | --- |
| P0 AlphaZero | alpha == 0（beta==1 已在 host quick return） | y = beta·y 位精确缩放/置零，不引用 A 与 x |
| P1 主路径（列窗流式） | incx==1 ∧ incy==1 ∧ n ≥ 16 ∧ kEff ≤ 2047 ∧ colGran·lda·4 ≤ 64KB | 连续访存性能域全量命中 |
| P2 通用标量 fallback | 其余全部（跨步/负步长/极小 n/超门控 kEff、lda） | 全域正确性兜底，单缓冲串行 |

uplo 不进运行时分发：host 侧直接选择编译期模板实例 `SsbmvKernel<IS_UPPER>`（UPPER/LOWER 双实例翻转引用公式，运行时零开销）。

#### 3.2.2 kernel侧设计：

AIV-only 向量 kernel（`KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY)` 显式声明），`template <bool IS_UPPER>` 双实例。**核心算法为「行主输出归属 + 对称引用读取」**：按输出行切分多核后，每个存储元素恰好被其两个归属核各读取一次（N 项一次、T 项一次），元素间无跨核累加依赖——无跨核写 y、无原子操作、无 workspace、无两遍计算，核切分不改变逐元素结果。

**P0 AlphaZero**：incy==1 时向量化（y 段读回 UB → `Muls` 缩放 → 写回；beta==0 直接置零写回、不读旧值）；incy≠1 时按物理索引逐元素读改写（负步长反向遍历基址换算含于索引式）。位精确（单乘无加法）。

**P1 主路径**：本核行区间内按 TH 行瓦片循环；x 需求域窗、旧 y（beta≠0 时）单次搬运入 UB；A 列窗按上述分块策略流式（或整窗驻留）入 UB。块/窗内计算按数据形态分三档：

1. **驻留对角形态**（A 窗驻 UB 的瓦片，含带首小带宽瓦片）：按**存储对角线**逐条处理——固定对角线时 A 侧沿列步进用 `Gather`（字节偏移表）取出对角向量，x 侧为连续切片、从 x 窗的多相位对齐副本直读（副本相位按切片起点对齐设计，免去逐元素取数），每对角线一次 Gather + 一次融合乘加累入目标行段；T 项同为对角形态（UPPER/LOWER 镜像对称）。
2. **带首瓦片门控**：带首瓦片（行窗起点 < kEff）且 **kEff ≤ 16** 时同样走对角形态；被带边界裁剪的对角线行段起点对齐到 8 行边界以保持向量写回 32B 对齐，起点以下 ≤7 行/对角线的残余行用标量 FMA 补算暂存、事件同步后一次向量加法折叠入累加器（残余段仅读有效元素，规避 0×Inf 产生 NaN）；kEff > 16 的带首瓦片保持按列形态（小 n 大 k 形态下按列实测更优，门控回归保守取值、零回归面）。
3. **按列形态**（流式大 k 域）：N 段——A 列段 Gather 暂存 + 头部非对齐 lane 显式清零后，有限 x 元素走融合乘加、非有限 x 元素走乘法+清零头部+加法（全域规避 0×Inf=NaN）；T 段——两侧 Gather 暂存 + 逐元素积 + **两级低阶 `WholeReduceSum` 归约**（分片归约至固定间距槽位、压实后单次全归约），每 8 列 Gather+Add 批量并入累加器。归约原语选用低阶 WholeReduceSum：Level-2 归约接口在无 TPipe 的静态 UB 布局 kernel 内不可用（内部依赖事件 ID 获取，实测崩溃定谳）。

瓦片末输出合成：`y = alpha·acc + beta·y_orig`（beta==0 跳过旧值），`DataCopyPad` 写回 GM 本核行区间（y 物理区间仅本核写，无写冲突）。

**P2 通用标量 fallback**：128 行标量累加瓦片 + 列外层循环 GM 直读 + N/T 双贡献标量累加（主对角线单计），x/y 按步长物理索引访问（含负步长反向遍历），合成后按 incy 散布写回（物理间隙位置不属于逻辑向量，保持原值）。单缓冲串行（该域无性能要求）。

**流水与同步**：`SET_FLAG` / `WAIT_FLAG` / `PIPE_BARRIER` 显式配对；A 块双缓冲 MTE2/V 重叠；UB 区域复用依赖显式事件标注。GM↔UB 搬运一律 `DataCopyPad`（不依赖 32B 严格对齐假设）；不使用 `GlobalTensor::SetValue/GetValue`（生产代码禁止）。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas A2/A3 系列（Ascend 910B3，arch22 / DAV_2201） | √（本任务新增） |

## 算子约束限制

- 仅支持 float32；不支持超出 incx/incy/lda 语义的非连续内存访问；不涉及广播；
- y 为原地覆写，不返回视图；异步执行依赖 `aclblasSetStream` 绑定 stream，读回结果前须同步；
- kEff > 2047 或 lda 超出主路径对齐门控的组合走通用标量路径（正确性全覆盖，无性能承诺）；
- 上游 arch35（Ascend 950PR）存根与本实现同目录共存、接口共用，互不影响。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | FLOAT32 档：rtol=2⁻¹⁰（9.77e-4）、atol=2⁻¹⁶（1.53e-5）、逐元素 \|actual−golden\| ≤ atol+rtol×\|golden\|，matched_ratio ≥ 0.99 且 max_abs_error ≤ max(1e-2, 32×ULP)；alpha=0 缩放与 n=0 no-op 为 EXACT 位精确比对；golden 由 cblas（Netlib ssbmv）生成 | 生态算子开源精度标准 + 任务书 §3.2 |
| 性能标准 | kernel-only 口径（msprof op_summary Task Duration 取 Avg），warmup 后有效采样 >50 次取平均；全量 200 条性能 case 平均单次耗时不高于 GPU 基线折算标杆（gpu_ms ÷ 0.8），含任务书 §3.3 四条锚点 | 任务书 §3.3 |

**精度结果**（真机 Ascend 910B3 / CANN 9.1.0）：黑盒 ST（CSV 驱动 GTest，cblas golden，含出厂全量 + L0/L1/L2 分级 + 边界负向补充）**1226/1226 全部通过**（另经 PyTorch 适配层独立 golden 双链复跑 1226/1226，与黑盒链逐 case 零差异）；白盒用例 **810/810** 通过；UT **56/56** 通过。EXACT 特例（alpha=0 三分支、quick return、beta=0 旧值不读取）逐条位精确校验通过；Inf/NaN 传播对齐 cblas 口径。

**性能结果**（kernel-only，warmup 10 次 + 有效采样 60 次取平均、每 case 3 轮取中位）：**全量 200 条性能 case 200/200 达标**（最差 case 余量 11.9%，几何平均达标裕量约 3.7×）；四条标杆锚点全部通过且显著优于标杆：

| case | n / k / uplo | 标杆耗时（us） | 实测耗时（us） | 判定 |
| --- | --- | --- | --- | --- |
| 1 | 1024 / 16 / UPPER | 13.57 | **5.073** | ✅（余量 63%） |
| 2 | 2048 / 32 / UPPER | 24.67 | **12.386** | ✅（余量 50%） |
| 3 | 1024 / 16 / LOWER | 14.41 | **5.111** | ✅（余量 65%） |
| 4 | 4096 / 64 / UPPER | 47.74 | **21.700** | ✅（余量 55%） |

四标杆均达到并优于业界同类 GPU 实现折算标杆；全量 200 条隐含带宽峰值需求远低于 910B3 HBM 带宽（非带宽饱和域），达标关键在多核均衡与固定开销压缩。

## 兼容性分析

新算子目录从零新建（`blas/sbmv/arch22/` 与 `test/sbmv/ssbmv/arch22/`），接口声明复用 `include/cann_ops_blas.h` 已有声明（零改动，与其他产品线共用）；与上游 arch35 存根同目录共存、互不修改；无既有版本兼容性负担。

测试工程位于 ops-blas 仓 `test/sbmv/ssbmv/arch22/`：CSV 驱动 GTest 黑盒全量 + L0/L1/L2 分级 + 白盒用例 + UT + kernel-only 性能 harness，全部经真机验证；执行步骤见该目录 README，验收人可复现。
