# aclblasCtbmv 算子设计文档

## 需求背景（required）

### 需求来源

昇腾 CANN 社区任务——8月社区任务《aclblasCtbmv算子开发（A2A3）》，要求在 Atlas A2/A3 系列产品（arch22 架构）上使用 Ascend C 编程语言开发单精度复数（complex64）三角带状矩阵-向量乘算子 `aclblasCtbmv`，与 cuBLAS `cublasCtbmv` 核心功能、参数语义完全对齐，精度对标 Netlib `ctbmv` 参考实现（golden 由 cblas 生成）。

### 背景介绍

#### tbmv 算子数学语义

tbmv（Triangular Banded Matrix-Vector product）是 BLAS Level 2 标准接口，计算 `x = op(A) * x`（x 原地覆写），其中 A 为 n×n 上/下三角带状矩阵，k 为半带宽（次/超对角线条数，k ∈ [0, n-1]）：

- `trans = ACLBLAS_OP_N` 时 `op(A) = A`
- `trans = ACLBLAS_OP_T` 时 `op(A) = Aᵀ`（**纯转置，不取共轭**）
- `trans = ACLBLAS_OP_C` 时 `op(A) = Aᴴ`（**共轭转置，复数路径专有语义**）

A 按**带状格式（banded storage）列主序**存于 lda×n 数组，仅带内元素被引用：

- `uplo = ACLBLAS_UPPER`：主对角存于第 k+1 行，元素 A(i,j) 位于第 1+k+i-j 行（1-based）；数组左上角 k×k 三角区域不被引用
- `uplo = ACLBLAS_LOWER`：主对角存于第 1 行，元素 A(i,j) 位于第 1+i-j 行（1-based）；数组右下角 k×k 三角区域不被引用

`diag = ACLBLAS_UNIT` 时对角元素不被读取、视为 (1,0)；`diag = ACLBLAS_NON_UNIT` 时读取实际对角元素。`incx` 为 x 元素步长，支持负步长（从向量末端反向遍历）。`n = 0` 为合法 no-op。无 alpha/beta 标量参数。

模式组合 uplo(2) × trans(3) × diag(2) = **12 组合，全部支持**。

#### arch22 硬件特性

Atlas A2/A3（910B3/910C，arch22/DAV-220）具备以下核心特性：

- **每芯片 40 个 AIV（Vector）向量核**，无 AIC（Cube）矩阵乘引擎可用路径
- 每核 192KB Unified Buffer（UB），SIMD 向量运算
- 向量指令宽度 64 floats/cycle，支持 FMA 融合乘加
- 支持硬件跨核屏障（SyncAll），可实现单 kernel 内的全核同步

#### 仓内现状分析

ops-blas 仓当前**无 `aclblasCtbmv` 声明**（complex 路径为新增接口，头文件需新增声明，与仓内 `aclblasStbmv` 同型：float→aclblasComplex，共用同一 API，不定义产品私有平行接口）。同族参照 `aclblasStbmv`（float 三角带状矩阵-向量乘）已有实现与测试结构（`blas/tbmv/stbmv/`、`test/tbmv/stbmv/`），本算子 host 骨架与测试工程结构与其对齐。

复数类型 `aclblasComplex` 以仓内 `include/cann_ops_blas_common.h` 定义为准（实部/虚部各 float32，交错 AoS 布局，8 字节/元素）。

## 需求分析（required）

### 需求描述

在 arch22 平台上实现 `aclblasCtbmv` 接口，满足：

1. 功能与 cuBLAS `cublasCtbmv` 完全对齐（含 OP_T 纯转置与 OP_C 共轭转置的严格区分）
2. 精度满足生态算子开源精度标准：实部/虚部分别按 FLOAT32 判定（rtol=2⁻¹⁰、atol=2⁻¹⁶、matched_ratio≥0.99、max_abs_error≤1e-2 或 32×ULP），非 bit-exact，diag=UNIT 用例对角不参与比对
3. 性能对标任务书标杆（Atlas 800I A2，COMPLEX64，先 warmup 再有效采样 >50 次取平均）：

| case | n | k | uplo | trans | diag | incx | 标杆耗时（us） |
|---|---|---|---|---|---|---|---|
| 1 | 512 | 8 | UPPER | N | NON_UNIT | 1 | 6.40 |
| 2 | 1024 | 16 | UPPER | N | NON_UNIT | 1 | 10.57 |
| 3 | 2048 | 32 | LOWER | T | NON_UNIT | 1 | 12.65 |

### 需求拆解

1. **Host 侧**：参数校验（4 类返回码）、快速返回（n=0 no-op / k=0&&UNIT 短路）、动态核数获取、tiling 计算、UB 驻留预算守卫、单次 kernel 启动、异步返回
2. **Kernel 侧**：单融合 kernel（12 编译期模式体核内 switch 分发）、x 装载 incx 归一化 UB 驻留、行窗口所有权计算、A 子面板化加载、按对角线向量化复数 FMA、硬件屏障解原地竞态、按 incx 融合写回
3. **测试侧**：CSV 驱动 GTest 工程（结构对齐 test/tbmv/stbmv/），1200+ 条用例，精度 golden 由 cblas（Netlib ctbmv）生成

## 详细设计（required）

### 算子分析

#### 数学公式

UPPER（A(i,j) 非零当且仅当 max(0,j-k) ≤ i ≤ j）：

```
trans=N:  x[i] = Σ_{j=i}^{min(i+k,n-1)} A(i,j) · x[j]
trans=T:  x[j] = Σ_{i=max(0,j-k)}^{j} A(i,j) · x[i]
trans=C:  x[j] = Σ_{i=max(0,j-k)}^{j} conj(A(i,j)) · x[i]
```

LOWER（A(i,j) 非零当且仅当 j ≤ i ≤ min(j+k,n-1)）：与上对称（行列区间对偶）。

复数乘法按 fp32 分量展开（A2 无原生复数向量运算）：(a+bi)(c+di) = (ac-bd) + (ad+bc)i，即 4 实乘 + 2 实加/减；conj 仅虚部取负（Muls(-1)），OP_T 与 OP_C 严格区分。

#### 关键语义边界

| 边界 | 语义 |
|------|------|
| n=0 | 合法 no-op，直接返回 SUCCESS（A/x 可为 nullptr） |
| k=0 且 diag=UNIT | A 退化为单位对角阵，x 不变，host 短路返回 SUCCESS |
| k ≥ n-1 | 满带（覆盖整个三角），计算范围按 k_eff=min(k,n-1) 钳制；存储行寻址仍用原始 k（kStorage 双轨） |
| 未映射区 | UPPER 左上 / LOWER 右下 k×k 不读取；diag=UNIT 对角不读取 |
| incx<0 | 从向量末端反向遍历（起始偏移 kx = 1-(n-1)·incx，1-based） |
| lda | ≥ k+1（紧凑或多 padding 均支持） |

#### 支持数据类型

complex64（aclblasComplex，实部/虚部各 float32），实部、虚部分别按 FLOAT32 精度标准判定。

#### 原地覆写与确定性

x 原地输入/输出。各输出元素 x_i 由单一核独立算完（行窗口所有权，无跨核累加），累加次序核内固定，相同输入产生逐位相同输出（优于任务书 determinism 不要求的基线）。

### 算子实现

#### 整体架构

```
┌──────────────────────────────────────────────────────────────┐
│                    aclblasCtbmv (Host)                        │
│  ┌────────┐  ┌─────────┐  ┌────────┐  ┌────────┐  ┌────────┐ │
│  │参数校验 │→│ 快速返回 │→│动态核数 │→│ Tiling │→│ 单次   │ │
│  │(4类返回│  │n=0/k=0  │  │(运行时 │  │+UB驻留 │  │ launch │ │
│  │ 码)    │  │ &UNIT   │  │ 获取)  │  │ 守卫   │  │(异步)  │ │
│  └────────┘  └─────────┘  └────────┘  └────────┘  └───┬────┘ │
└────────────────────────────────────────────────────────┼──────┘
                                                         │
              ┌──────────────────────────────────────────▼──────┐
              │     ctbmv_fused_kernel (AIV，单 __global__ 入口) │
              │  ┌───────────────────────────────────────────┐ │
              │  │ 相A：x 装载（incx 归一化进 UB AoS 驻留）    │ │
              │  │   incx==1 单块 DMA / ==-1 反向 chunk /     │ │
              │  │   |incx|>1 多块 8B DMA 进 32B 槽+槽表Gather │ │
              │  └──────────────────┬────────────────────────┘ │
              │  ┌──────────────────▼────────────────────────┐ │
              │  │ 相B：行窗口所有权计算                      │ │
              │  │   A 子面板 2D DMA（双缓冲）→ 逐对角线       │ │
              │  │   对齐暂存 Gather → fp32 分量复数 FMA       │ │
              │  │  （带向 segCap 分 chunk，满带不撑爆 UB）    │ │
              │  └──────────────────┬────────────────────────┘ │
              │         SyncAll 硬件屏障（全核读完 x 后才写回）  │
              │  ┌──────────────────▼────────────────────────┐ │
              │  │ 相C：按 incx 融合写回本核行窗口            │ │
              │  └───────────────────────────────────────────┘ │
              └─────────────────────────────────────────────────┘
```

**单注册入口约束**：12 个模式体（uplo × diag × tcMode 编译期模板实例化）折叠进**单一 `__global__` 入口** `ctbmv_fused_kernel(mode, ...)`，核内 runtime switch 分发。该结构为强制性约束——当前工具链对同一模块内含跨核屏障序列的 kernel 注册数量有限制，多实例入口在真机不可正确执行，单入口使模块内含屏障 kernel 注册数 = 1，12 模式真机全通过。

#### Host 侧设计

##### 参数校验（先于任何计算与 launch）

| 条件 | 返回码 |
|------|--------|
| handle == nullptr | ACLBLAS_STATUS_HANDLE_IS_NULLPTR (9) |
| uplo / trans / diag 非法枚举 | ACLBLAS_STATUS_INVALID_ENUM (10) |
| n < 0 / k < 0 / lda < k+1 / incx == 0 | ACLBLAS_STATUS_INVALID_VALUE (3) |
| n > 0 时 A 或 x 为 nullptr | ACLBLAS_STATUS_INVALID_VALUE (3) |
| GM 步长超出硬件 stride 寄存器宽度（极端大 lda/\|incx\|） | ACLBLAS_STATUS_NOT_SUPPORTED (7) |
| UB 驻留预算超限（满带超大 n） | ACLBLAS_STATUS_NOT_SUPPORTED (7) |

##### 快速返回

- `n == 0` → SUCCESS（no-op，不触碰 A/x）
- `k == 0 && diag == UNIT` → SUCCESS（x 不变）

##### Tiling 策略

- **核间切分（行窗口所有权）**：每核拥有互斥输出行窗口 [r0, r1)，窗口均分（rowSegFormer/rowSegTail），UPPER 需读 x[r0, r1+k)、LOWER 需读 x[r0-k, r1)（仅 k 行邻核重叠）。输出单写者 → 无跨核累加、无原子操作、无零初始化
- **自适应核数**：核数运行时动态获取（`aclrtGetDeviceInfo`，禁写死）；小带宽（k≤16）按行数缩放核数（约 64 行/核），UB 驻留不满足则增核回退——削减小形状的固定调度开销
- **面板化**：A 按列面板（panelCols = max(1, min(128, ⌊25KB/ubBlkBytes⌋))）分块加载，单面板缓冲上界 25KB，双缓冲使 MTE2 与 V 重叠
- **带向分 chunk**：segCap=min(bandRows,1024)，满带大 n 时 UB 足迹不随 bandRows 线性增长
- **UB 驻留守卫**：host 侧精确复刻 kernel 分配算术，launch 前校验总分配 ≤ 168KB（192KB 物理保守余量），超限返回 NOT_SUPPORTED

##### Tiling 数据结构

```cpp
struct CtbmvTilingData {
    int64_t n, k;            // k = min(k_orig, n-1)，计算范围钳制
    int64_t kStorage;        // 原始 k，存储行寻址（UPPER rowBase = kStorage+i0-j）
    int64_t lda, incx;
    int64_t xLogStart;       // 负步长逻辑起点换算
    int64_t bandRows;        // min(k+1, n)
    int64_t segCap;          // 带向 chunk 行上限 = min(bandRows, 1024)
    int32_t useCoreNum;      // 自适应核数
    int64_t rowSegFormer;    // 非尾核行窗口长度
    int64_t rowSegTail;      // 尾核行窗口长度
    int64_t panelCols;       // A 面板列数
    int64_t panelCount, panelTailCols;  // advisory（kernel 自行折算）
};
```

#### Kernel 侧设计

##### 相A：x 装载（incx 归一化 UB 驻留）

按行窗口求本核 x 逻辑区间 [xLo, xHi)，全 DMA 化读入 UB AoS 驻留（xAoS[m] = 逻辑 x(xLo+m)，交错实虚）：

- `incx == 1`：单次连续单块 DataCopyPad（队列张量即驻留区，零 Gather 零建表）
- `incx == -1`：分 chunk 连续读逆序物理区间 + 每 chunk 一次降序复数对 Gather（倍增建表，尾 chunk 按实际长度重建表）
- `|incx| > 1`：分 chunk 多块 8B DMA 进 32B 槽 + 槽对 Gather 抽取（利用 UB 32B 块粒度而非对抗；标量 GlobalTensor 访问在多核下无 cache 一致性，全路径禁用）

##### 相B：行窗口所有权计算

- 每核对本核行窗口逐对角线（k+1 条）计算：A 子面板一次 2D DataCopyPad 载入 → 对角线元素经对齐暂存 Gather 成 32B 对齐向量 → fp32 分量复数 FMA 累加进 UB 累加器（2 累加器对断依赖链；体部 32B 对齐向量化，头部 ≤7 车道标量批处理）
- diag=UNIT：对角线 d==0 不读 A，贡献 = x 自身向量加回
- trans=C：装载后虚部 Muls(-1) 取共轭，与 trans=T 严格区分
- 未映射区：面板矩形 DMA 可带入 UB，车道钳制不消费，精度上不出带

##### SyncAll 硬件屏障（原地竞态消解）

x 原地覆写存在跨核读写竞态（核 A 写 x_j 时核 B 可能未读完）。计算相结束后、写回相之前一次 `SyncAll()`：屏障返回即保证全部核已完成相A的 GM x 读取（MTE2 在 DeQue 处已排空），此后任何核写回无竞态。相比两次 launch（compute+snapshot/copy）结构，省去第二次 launch 与 GM 快照 workspace，设备固定开销最低。空核跳过工作但参与屏障会合；真机 grid==useCoreNum 无空核。

##### 相C：按 incx 融合写回

- `|incx| == 1`：累加器逆交错 Gather 成 AoS 连续段，单块 DataCopyPad 写回（V_MTE3 事件同步）
- `|incx| > 1`：标量装 32B 槽 + 多块 DataCopyPad 写回（S_MTE3 事件同步）
- 仅逻辑向量位置被更新，步长间隙元素保持不变

#### 精度策略

- 全程 fp32 分量计算（无 FP16 中间精度路径，无 Cast）
- 累加器 fp32；复乘 4mul2add 与 cblas 参考实现同一数学式，非 bit-exact 但远优于 rtol=2⁻¹⁰ 容差
- 比对口径：实部/虚部分别判定，逐元素 `|actual-golden| ≤ atol + rtol×|golden|`，用例级 matched_ratio ≥ 0.99 且 max_abs_error ≤ 1e-2（或 32×ULP）

#### 资源规划

- **UB**：单核总分配 ≤ 168KB（192KB 物理，TPipe/标量预留余量），host 侧 launch 前精确复刻守卫
- **GM**：A 只读 / x 原地；**无 workspace**（行窗口所有权消除零初始化依赖，单 kernel 融合消除快照需求）
- **Host**：零堆分配、零调用内同步（cuBLAS 流语义异步返回）

## 测试设计（required）

### 测试工程

落位 ops-blas 仓 `test/tbmv/ctbmv/`（结构对齐 test/tbmv/stbmv/）：CMakeLists + CSV 参数结构 + cblas golden 封装（OP_T/OP_C 严格区分）+ CPU 朴素参考实现（Mock 双实现交叉验证）+ GTest 主程序（Real 设备往返 / Mock 路由双模式）。

### 用例设计（1200+ 条，CSV 驱动）

| 类别 | 覆盖 |
|------|------|
| L0 基础 | uplo×trans×diag 全枚举 12 组合 × 小尺寸 |
| L1 尺寸 | n 覆盖 0/1/小质数/2 的幂±1/非对齐/大规模（至 2048） |
| L2 带宽 | k=0/1/小值/半带/满带 n-1（含 k>n-1 钳制） |
| 前导维/步长 | lda=k+1 紧凑及多 padding；incx ∈ ±1/±2/±3 |
| 填充/特殊值 | 均匀/全零/交替/极端值/Inf/NaN；diag=UNIT 对角 NaN 不读验证 |
| 边界与负向 | n=0 quick return、空指针、非法 lda、负维度、负带宽、零步长、非法枚举（期望对应返回码） |
| 性能 | 任务书 3 典型 case + 尺寸/带宽网格（warmup + 有效采样 >50 次取平均） |

### 工程强化

- 未映射带外区与 lda padding 填 NaN 哨兵（误读即暴露）；x 步长间隙填哨兵并做原地保持逐位断言
- 异常用例返回码 + x 未修改双断言
- 溢出带判定规则：±Inf 与 |v|≥FLT_MAX/4 有限值视为同类（累加次序发散属任务书允许的非 bit-exact 范畴），NaN vs 有限值判负

## 性能设计目标

- 结构开销最小化：单次 launch + 单硬件屏障 + 无 workspace + 无 memset + 无调用内同步，设备固定开销压至结构下限
- 计算向量化：面板化批量 DMA + 合并表大 Gather + 合批 FMA + 建表向量化，逐操作开销与操作数双压
- 小形状自适应：核数随规模缩放，削减固定调度税
- 目标：3 个标杆 case 达到任务书 §3.3 性能要求（Atlas 800I A2，COMPLEX64）
