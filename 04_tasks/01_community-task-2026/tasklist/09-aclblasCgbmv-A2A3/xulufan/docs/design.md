# aclblasCgbmv 算子设计文档（Atlas A2/A3 / arch22）

| 项目 | 内容 |
| --- | --- |
| 任务 | 9月社区任务-aclblasCgbmv算子开发(A2A3) |
| 目标平台 | Atlas A2/A3 系列产品（arch22，覆盖 ascend910b\* / ascend910_93\*），性能基准设备 Atlas 800T A2（910B3） |
| 软件版本 | CANN 9.1.0 |
| 目标代码仓 | https://gitcode.com/cann/ops-blas （算子 `blas/gbmv/arch22/`，测试 `test/gbmv/cgbmv/arch22/`） |
| 对标接口 | cuBLAS `cublasCgbmv`（参数序列一致，无需映射）；精度 golden：Netlib BLAS `cgbmv` |
| 交付阶段 | 本文为设计文档（交付件①），评审合入 cann-ops-competitions 对应任务目录；文中性能/耗时均为任务书验收目标，非实测结果 |

---

# 需求背景（required）

## 需求来源

CANN 生态 9 月社区任务「aclblasCgbmv 算子开发(A2A3)」。要求在 Atlas A2/A3 系列产品（arch22
架构）上，基于 ops-blas 开源仓工程框架、以 Ascend C kernel 直调方式，实现与 cuBLAS
`cublasCgbmv` 功能与参数语义对齐的单精度复数（COMPLEX64）一般带状矩阵向量乘算子：

```
y = alpha * op(A) * x + beta * y
```

完成算子设计、开发、测试全流程：设计文档提交至 cann-ops-competitions 仓（本文件），算子代码
与测试合入 ops-blas 仓 `blas/gbmv/arch22/` 与 `test/gbmv/cgbmv/arch22/`。

## 背景介绍

### gbmv 数学语义与带状存储

gbmv（General Banded Matrix-Vector multiplication）是 BLAS Level 2 标准接口。设 A 为
m×n 一般带状矩阵：下带宽 `kl`（主对角以下最多 kl 条非零对角线）、上带宽 `ku`（主对角以上
最多 ku 条）。`trans` 决定 op(A)：

- `ACLBLAS_OP_N`：op(A) = A（m×n），x 长 n、y 长 m；
- `ACLBLAS_OP_T`：op(A) = Aᵀ（n×m），x 长 m、y 长 n；
- `ACLBLAS_OP_C`：op(A) = Aᴴ（共轭转置，n×m），x 长 m、y 长 n。

m/n 恒为矩阵 A 的行数/列数，不随 trans 翻转。A 采用 BLAS 标准带状列主序打包存储于
`lda×n` 数组（`lda ≥ kl+ku+1`）：0 基索引下元素 A(i,j) 存放在

```
A_off(i,j) = j*lda + (ku + i - j)
```

即第 j 列自上而下顺序存放该列带内元素，主对角线位于打包数组的第 ku 行，带外区域
（左上 ku×ku 三角、右下 kl×kl 三角）不引用。以 m=n=5、kl=ku=1、lda=3 为例：

```
        j=0  j=1  j=2  j=3  j=4        打包存储（lda=3，列主序）
 i=0   [ 11   12   .    .    . ]        col0: * 11 21     (* 为不引用位)
 i=1   [ 21   22  23    .    . ]        col1: 12 22 32
 i=2   [ .    32   33  34    . ]        col2: 23 33 43
 i=3   [ .    .    43  44  45 ]         col3: 34 44 54
 i=4   [ .    .    .    54  55 ]        col4: 45 55 *
```

x/y 为复数向量，按步长 incx/incy 存取，步长可为负（负步长表示逻辑向量首元素对应物理缓冲
末尾，反向取值）。`m=0` 或 `n=0` 为合法 no-op；`beta=0` 时原 y 不必为有效输入（不读取）；
`alpha=0` 时退化为 `y = beta*y`（不读 A/x），与 cuBLAS/netlib 语义一致。

### ops-blas 仓现状与工程先例

基于 ops-blas 仓 master 现状：

| 对象 | 现状 | 与本任务的关系 |
| --- | --- | --- |
| `blas/gbmv/arch35/` | 已有 `sgbmv`（float32，SIMT 编程模型：`__simt_vf__`/`asc_vf_call`/blockIdx） | host 校验/tiling 语义可参考；kernel 为 DAV-3510（A5）SIMT 特性，**不能**直接用于 arch22，且无复数支持 |
| `blas/gbmv/arch22/` | **不存在** | 本任务新增 cgbmv 实现目录 |
| `include/cann_ops_blas.h` | 已声明 `aclblasSgbmv`，**无 `aclblasCgbmv` 声明** | 本任务需新增接口声明（交付件④要求：接口声明放入 `include/cann_ops_blas.h` 供其他产品线共用） |
| `blas/gemv/arch22/` | 已有 `cgemv` 复数实现：UB 内实/虚部分离缓冲（`Ybase_r/Ybase_i` 等）、mask 反交错（`Gather`）、ping-pong 双缓冲、`GetBlockIdx` 分核 | 复数向量化的直接工程先例（gbmv 本质是带状 gemv） |
| `blas/tbsv/arch22/` | 已有 `stbsv`：列扫描/行内积双路径、多核 chunk 划分、GM 兜底、`TPipe` 事件同步（MTE2_S 等）、`DataCopyPad` 非对齐搬运 | arch22 标准 Ascend C 工程模式先例（本任务同作者风格对齐的仓库范式） |
| 根 CMake `get_soc_arch_dirs()` | `ascend910b*`、`ascend910_93*` → arch22，按架构目录 glob 源文件 | 新增 `blas/gbmv/arch22/` 目录即自动参与 arch22 构建，无需改构建脚本 |

### arch22 硬件特性（设计前提）

Atlas 800T A2（910B3，DAV-220）与 A3 系列（映射 arch22）：

- 向量核（AIV）承担 Level-2 算子全部计算，`GetAivCoreCount()` 取运行时可用 AIV 数；
- 每核 192KB Unified Buffer（UB），向量指令 64 lanes/cycle；
- 无 Level-2 可用的 Cube 矩阵乘路径，复数乘加必须走向量 SIMD 分解（参考 cgemv arch22
  的实/虚分离四实数乘加方案）；
- GM→UB 搬运 `DataCopyPad` 支持非 32B 对齐长度，覆盖任意 lda/inc 场景。

### 功能缺口

1. **arch22 gbmv 整体缺失**：无 host、无 kernel、无构建、无测试，本任务为全新实现；
2. **复数支持缺失**：arch35 sgbmv 仅 float32，复数乘法/共轭转置/交错存储布局均需新设计；
3. **接口声明缺失**：`include/cann_ops_blas.h` 需新增 `aclblasCgbmv`（`aclblasComplex`
   已由 `cann_ops_blas_common.h` 定义，复用即可）；
4. **测试工程缺失**：`test/gbmv/` 下无 cgbmv 目录，需按仓内 CSV 驱动 GTest 范式新建。

# 需求分析（required）

## 需求描述

在 arch22 平台实现 `aclblasCgbmv` 句柄式 BLAS 接口（kernel 直调，经 handle 绑定 stream
异步下发），满足：

1. **功能对齐**：与 cuBLAS `cublasCgbmv` 参数序列、语义逐项对齐——trans N/T/C、矩形
   m×n、非对称带宽 kl≠ku、退化带宽（kl=0/ku=0/kl=m-1/ku=n-1）、lda padding、正负步长
   incx/incy、alpha/beta 特值（0/1/-1/纯虚/一般复数）、m=0/n=0 no-op、全量负向校验；
2. **精度达标**：COMPLEX64 按 FLOAT32 实/虚部分别判定混合容差
   （atol=rtol=2⁻¹³、matched_ratio≥0.99、max_abs_error≤max(1e-2, 32×ULP)），
   golden 为 Netlib `cgbmv`；
3. **性能达标**：910B3 上 5 条任务书典型 case 平均单次 kernel 耗时（msprof 口径，有效
   采样>10 次取平均）不高于达标线（10.665~52.487us，详见 §可维可测分析），其余 200 条
   TC_PF 用例满足 NPU 耗时 ≤ gpu_ms/0.8；
4. **交付完整**：接口声明入公共头、README 产品支持表标注 A2/A3 支持、测试工程含 CSV
   与复现说明，A2 与 A3 均完成验收自验。

## 需求拆解

| 拆解项 | 内容 |
| --- | --- |
| 1. 接口层 | `include/cann_ops_blas.h` 新增 `aclblasCgbmv` 声明（参数序列同 `aclblasSgbmv`，标量/A/x/y 类型为 `aclblasComplex`） |
| 2. Host 层 | `blas/gbmv/arch22/cgbmv_host.cpp`：参数校验（错误码对齐任务书 §2.4）→ 快速返回（no-op / alpha=0 特例）→ tiling 计算 → kernel 下发 |
| 3. Kernel 层 | `blas/gbmv/arch22/cgbmv_kernel.cpp`：N 列扫描主路径、T/C 行内积主路径、步长 gather/scatter 支撑、beta 缩放特例路径、小规模 GM 直算兜底 |
| 4. 测试层 | `test/gbmv/cgbmv/arch22/`：CSV 驱动 GTest（随任务 1200 条 + 自补负向）、Netlib `cgbmv` golden、msprof 性能采集与 `gpu_baseline.csv` 逐条关联比对 |
| 5. 文档层 | `blas/gbmv/README.md` 增补 cgbmv 接口说明与产品支持表（Atlas A2/A3：支持） |

# 详细设计（required）

## 算子分析

### 数学公式

0 基索引。只有满足 `max(0, j-ku) ≤ i ≤ min(m-1, j+kl)` 的 A(i,j) 参与计算。

**trans=N（按输出行 i，i ∈ [0, m)）：**

```
j_begin(i) = max(0, i-kl),  j_end(i) = min(n-1, i+ku)
y[i] = alpha * Σ_{j=j_begin..j_end} A(i,j) * x[j] + beta * y[i]
```

**trans=T（按输出列 j，j ∈ [0, n)）：**

```
i_begin(j) = max(0, j-ku),  i_end(j) = min(m-1, j+kl)
y[j] = alpha * Σ_{i=i_begin..i_end} A(i,j) * x[i] + beta * y[j]
```

**trans=C（同 T，A 元素取共轭）：**

```
y[j] = alpha * Σ_{i=i_begin..i_end} conj(A(i,j)) * x[i] + beta * y[j]
```

复数乘法按实部/虚部分解：`(a_r + i·a_i)(x_r + i·x_i) = (a_r·x_r − a_i·x_i) + i·(a_r·x_i + a_i·x_r)`；
`conj(A)` 仅翻转 A 的虚部符号，x 与 alpha 均不取共轭。

### 带状存储索引换算与列段连续性（本设计的核心洞察）

代入 `A_off(i,j) = j*lda + (ku+i-j)`，固定列 j 时带内行 i 连续 ⟹ **列内元素物理连续
（stride-1 floats）**。列 j 的有效段为：

```
lo_j = ku − min(j, ku)                    （j ≥ ku 时为 0）
hi_j = min(ku+kl, ku+m−1−j)               （j ≤ m−1−kl 时为 ku+kl）
段长 L_j = hi_j − lo_j + 1 ≤ kl+ku+1 =: B
```

矩阵中部列（`ku ≤ j ≤ min(n-1, m-1-kl)`）为满带段，长度恰为 B。这一连续性直接决定两条
主路径的形态：

- **N 路径（列扫描 axpy）**：逐列把连续段搬入 UB，以 x[j] 为标量系数做复数 axpy，累加进
  行窗累加器——与 Netlib `gbmv.f` 的 trans='N' 参考实现**同构**（其内层即
  `Y(I) = Y(I) + TEMP*A(K+I,J)` 列扫描）；
- **T/C 路径（行内积）**：每个输出 j 的求和项恰为 A 的一段连续列段，做连续段复数点积
  ——与 Netlib trans='T'/'C' 分支（每输出 `TEMP = TEMP + A(K+I,J)*X(I)` 顺序点积）同构。

x 侧：N 路径按列访问 x[j]（标量）；T/C 路径第 j 个输出访问 x 的行区间
`[i_begin(j), i_end(j)]`（incx=1 时物理连续）。y 侧：N 路径输出按行窗聚集写回；T/C 路径
逐输出写 y[j]。

**地址安全**：所有 GM 偏移以 int64 计算（`j*lda` 在 m/n=4096、lda=513 时约 2.1M×8B，仍在
int32 元素计数安全域内，但按仓内惯例统一 int64，防极端扩展）；incx/incy 先转 int64 再取
绝对值，规避 `inc == INT_MIN` 取负溢出（对齐仓内 stbsv 处理）。

### 支持数据类型与形状

| 对象 | 类型 | 逻辑长度/形状 | 约束 |
| --- | --- | --- | --- |
| A | Device `aclblasComplex`（COMPLEX64，交错存储） | m×n，物理 lda×n | 只读带内元素，不触碰带外不引用区 |
| x | Device `aclblasComplex` | n（N）/ m（T/C） | incx ≠ 0，可为负 |
| y | Device `aclblasComplex` | m（N）/ n（T/C） | 原地输出，incy ≠ 0，可为负 |
| alpha/beta | Host `aclblasComplex` 指针 | 标量 | 非空；host 侧解引用后按值传 kernel（与仓内 sgbmv 一致） |
| m/n/kl/ku/lda/incx/incy | Host int | 标量 | m,n ≥ 0；0 ≤ kl ≤ m-1；0 ≤ ku ≤ n-1；lda ≥ kl+ku+1；incx,incy ≠ 0 |

## 算子实现

### 实现方案总览（路径分派）

Host 侧按参数分派到 4 条 kernel 路径（单次 launch、无 workspace、无动态 Device 分配）：

| 优先级 | 条件 | 路径 | kernel 形态 |
| --- | --- | --- | --- |
| 0 | m=0 或 n=0 | 不下发 kernel，直接 SUCCESS | — |
| 1 | alpha=0 且 beta=1 | 不下发 kernel，直接 SUCCESS（y 保持原值，对齐 netlib quick return） | — |
| 2 | alpha=0（beta≠1） | **beta 缩放路径**：y = beta·y（不读 A/x） | 单核/少核向量 scale |
| 3 | outLen < TINY（TINY=32） | **GM 直算路径**：逐输出 GM 标量复 FMA | 单核 |
| 4 | trans=N | **列扫描主路径**（复数 axpy，行窗累加器驻 UB） | 多核 |
| 5 | trans=T/C | **行内积主路径**（连续段复数点积） | 多核 |

其中 `outLen = m`（N）/ `n`（T/C）。incx/incy ≠ ±1 时由 gather/scatter 支撑层归一为 UB
内连续向量（见 §步长支撑），主路径形态不变。

### Host 侧设计

#### 参数校验与错误码（校验次序固定，保证负向用例可预期）

```
1  handle == nullptr                     → ACLBLAS_STATUS_HANDLE_IS_NULLPTR
2  alpha == nullptr || beta == nullptr   → ACLBLAS_STATUS_INVALID_VALUE
3  trans ∉ {OP_N, OP_T, OP_C}            → ACLBLAS_STATUS_INVALID_VALUE
4  m < 0 || n < 0                        → ACLBLAS_STATUS_INVALID_VALUE
5  m == 0 || n == 0                      → ACLBLAS_STATUS_SUCCESS（no-op，先于带宽上界检查）
6  kl < 0 || kl > m-1                    → ACLBLAS_STATUS_INVALID_VALUE
7  ku < 0 || ku > n-1                    → ACLBLAS_STATUS_INVALID_VALUE
8  lda < kl+ku+1                         → ACLBLAS_STATUS_INVALID_VALUE
9  incx == 0 || incy == 0                → ACLBLAS_STATUS_INVALID_VALUE
10 alpha ≠ 0 时 A == nullptr             → ACLBLAS_STATUS_INVALID_VALUE（alpha=0 时 A 不参与）
11 alpha ≠ 0 时 x == nullptr             → ACLBLAS_STATUS_INVALID_VALUE（同上）
12 y == nullptr                          → ACLBLAS_STATUS_INVALID_VALUE（y 为输出必参与）
```

说明：第 5 步 no-op 先于带宽检查，因 m=0 时 `kl ≤ m-1` 数学上不可满足（kl≥0 恒大于 -1），
任务书 §2.4 对 m=0/n=0 的定义是"合法 no-op、返回成功"，带宽上界仅对非零维度约束——该次序
与随任务 TC_ED_230/231/232（m0/n0/m0n0 均 SUCCESS）逐条一致；其余负向用例
（TC_ED_235~248：空指针×6、非法枚举、非法 lda、负维度、零步长、负 kl/ku）亦被上表全覆盖。

#### 快速返回与特例

```
m == 0 || n == 0                       → SUCCESS（不发 kernel）
alpha == 0 && beta == 1                → SUCCESS（不发 kernel，y 逐位保持原值；
                                         注意不可走 beta 缩放路径——1×y 会把 y 中的
                                         SNaN 静默化为 QNaN，破坏"不触碰"语义）
alpha == 0 && beta ≠ 1                 → beta 缩放 kernel（y = beta·y；
                                         beta == 0 时写 0 且不读 y）
beta == 0（alpha ≠ 0）                 → 主路径不读原 y，直接写 alpha·op(A)·x
                                         （y 含 Inf/NaN 时不污染，对齐 cuBLAS）
```

#### Tiling 数据结构与分核

```cpp
struct CgbmvTilingData {
    uint64_t a;                  // A 的 GM 地址
    uint64_t x;                  // x 的 GM 地址
    uint64_t y;                  // y 的 GM 地址
    uint32_t m, n, kl, ku, lda;
    uint32_t trans;              // 0=N / 1=T / 2=C
    float alphaR, alphaI;        // host 解引用后按值下发
    float betaR, betaI;
    int64_t  incx, incy;
    uint32_t coreNum;            // 实际下发核数
    uint32_t outPerCore;         // 每核输出窗长（N: 行窗；T/C: 输出列窗）
    uint32_t path;               // 2=betaScale / 3=gmTiny / 4=colSweep / 5=rowDot
    uint32_t xCached;            // N 路径 x 窗口缓存策略（0=逐列标量读 1=窗口段缓存）
};
```

分核策略（对齐仓内先例：优先满核、防碎块）：

- `coreNum = min(GetAivCoreCount(), ceil(outLen / MIN_ROWS_PER_CORE))`，
  `MIN_ROWS_PER_CORE`：N 路径 64（保证列扫描段向量效率与累加器规模）、T/C 路径 32；
- `outPerCore = ceil(outLen / coreNum)`，向上对齐到 8（向量掩码对齐友好）；
- 超出末尾的核空转直接返回（`GetBlockIdx() ≥ 实际块数` 时 return）；
- 单次 launch、`KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY)`，与仓内 arch22 算子一致。

### Kernel 侧设计

#### N 列扫描主路径（复数 axpy）

每核拥有行窗 `W = [r0, r1)`，输出 y[r0..r1)。与 W 相交的列区间为
`J = [max(0, r0-ku), min(n-1, r1-1+kl)]`。流程：

```
1. yAccR/yAccI（长度 |W|，UB 驻留）清零
2. for j in J:
     a. 计算列 j 与 W 的交叠行段 [ib, ie]：
        ib = max(r0, j-ku),  ie = min(r1-1, j+kl)          （行连续）
        对应 A 段：band_row ∈ [ku+ib-j, ku+ie-j]           （GM 连续，(ie-ib+1) 个复数）
     b. DataCopyPad 把 A 段（交错 complex）搬入 UB
     c. 反交错：GatherMask 提取 Ar[0..L)、Ai[0..L)         （L = ie-ib+1）
     d. 取标量 xr = Re(x[j]), xi = Im(x[j])                 （见 x 支撑）
     e. 复数 axpy（4 条向量指令）：
        Axpy(accR[ib-r0..], Ar,  xr);  Axpy(accR[ib-r0..], Ai, -xi)
        Axpy(accI[ib-r0..], Ar,  xi);  Axpy(accI[ib-r0..], Ai,  xr)
        其中 Axpy(dst, src, s) 语义为 dst[i] += s * src[i]
3. 合并写回：对 [r0, r1) 逐元素
     tR = alphaR·accR − alphaI·accI;  tI = alphaR·accI + alphaI·accR
     beta ≠ 0 时再读原 y 做复数乘加（beta == 0 不读）
     按 incy 散写回 y（incy=1 时整段连续写）
```

向量指令预算：每列 1 次 DataCopyPad + 2 次 GatherMask + 4 次 Axpy ≈ 6 条向量操作覆盖 L
个复数元素；对比标量逐元素复 FMA（8 标量操作/元素），指令数约降两个数量级（64 lanes）。
累加器复数分解为 accR/accI 两个实数向量，与 cgemv arch22 的实/虚分离布局同构。

#### T/C 行内积主路径（连续段复数点积）

每核拥有输出列窗 `[c0, c1)`。对每个输出 j：

```
1. 列 j 的 A 段：band_row ∈ [lo_j, hi_j]，长度 L_j ≤ B，GM 连续
2. DataCopyPad 搬入 → GatherMask 反交错为 Ar, Ai
3. x 行段 [i_begin(j), i_end(j)] 取入 UB 为 Xr, Xi
   （incx=1 时整段连续搬入后同样反交错；否则 gather，见步长支撑）
4. 复数点积（向量版，4 Mul + 4 ReduceSum + 2 Add/Sub）：
   T:  dR = Σ(Ar·Xr) − Σ(Ai·Xi);   dI = Σ(Ar·Xi) + Σ(Ai·Xr)
   C:  dR = Σ(Ar·Xr) + Σ(Ai·Xi);   dI = Σ(Ar·Xi) − Σ(Ai·Xr)
   （C 仅两处符号翻转：conj(A) = Ar − i·Ai）
5. y[j] = alpha·d + beta·y[j]（beta=0 不读原 y），按 incy 散写
```

窄带（L_j < VEC_K_MIN=64）时向量 ReduceSum 的固定开销不划算，切换**标量 4 累加器复数
点积**（与 Netlib 逐元素同序：`tR += ar·xr − ai·xi; tI += ar·xi + ai·xr`，逐位对齐
golden 的运算次序），阈值由实测标定后固化进 tiling。

#### 步长 gather/scatter 支撑与通用兜底

```
逻辑索引 k（0 ≤ k < len）的物理偏移：
  inc > 0:  k * inc
  inc < 0:  (len-1-k) * |inc|        （inc 先转 int64 再取绝对值）
```

- **x gather**（incx ≠ 1）：N 路径逐列取标量 `x[off(j)]`（2 次 GM 标量读，列序访问
  L2 友好）；T/C 路径逐元素取 x 行段（`|incx|` 步长）。incx = -1 时物理连续反向，
  可整段搬入 UB 后按倒序索引，实测标定是否启用；
- **y scatter**（incy ≠ 1）：写回阶段按物理偏移逐元素 `SetValue`；incy = 1 时整段
  `DataCopyPad` 连续写。y 读侧（beta ≠ 0）同理 gather；
- **小规模 GM 直算路径**（outLen < 32）：单核逐输出直接 GM 标量复 FMA（A 段 + x 标量），
  避免 TPipe/事件/两阶段开销——覆盖 TC_PF 小尺寸用例（kernel 仅数 us，launch/框架开销
  占比高，路径越薄越好），同时天然覆盖全部步长/转置组合；
- **A 段超 UB**（B > UB 预算上限，如 kl=ku=2047 的满带大矩阵）：T/C 路径按 K-blocking
  分段（分段搬运→分段点积→标量累加），N 路径行窗天然受限（`outPerCore` 缩小使
  累加器+A 段同落 UB），均不需要 workspace。

#### UB 内存布局（192KB 预算）

以 N 路径最坏情形估算（R = outPerCore，B = kl+ku+1，双缓冲 ping-pong）：

| Buffer | 大小 | 说明 |
| --- | --- | --- |
| accR / accI | 2 × R×4B | 行窗复数累加器（实/虚分离） |
| Aseg 交错（ping/pong） | 2 × B×8B | 列段原始交错数据 |
| Ar / Ai（ping/pong） | 4 × B×4B | 反交错后的列段实/虚部 |
| x 窗口（可选缓存） | (R+B)×8B | incx=1 且窗口 ≤ 阈值时整段缓存 |
| yOut 交错staging | R×8B | accR/accI 合并/交织后连续写回（incy=1） |
| 杂项（标量、掩码） | ≤ 4KB | — |

R=2048、B=513（case 4 量级单核分摊后）合计约 68KB；R=4096、B=321（case 5）约 96KB，
均落于 168KB 有效预算内（192KB 扣 TPipe/对齐余量，对齐仓内 stbsv 的 UB_BUDGET 口径）。
T/C 路径无 R 级累加器，预算更宽松。

#### 多核负载均衡

- **N 路径**：每核工作量 ≈ |W| × 平均段长。中部行窗各列近似满带，首/末行窗的边缘列
  段被截断，工作量略低——块分配的偏差上界为 B/R 量级（R ≥ 64 时 ≤ B/64，可接受）；
  `outPerCore` 按 8 对齐进一步抑制碎段；
- **T/C 路径**：每个输出的工作量 ∝ L_j，首末输出窗边缘段截断，同样以块分配 + 对齐处理；
- 核数不超过 `GetAivCoreCount()`，末核空转直接返回；不使用 SyncAll 跨核同步
  （输出划分互斥，无跨核数据依赖），无原子操作。

### 精度策略与误差分析

**对齐锚点是 Netlib `cgbmv` 的运算次序**（golden 生成器）：

1. **N 路径与 netlib 同构**：netlib trans='N' 即列扫描
   `Y(I) = Y(I) + TEMP*A(K+I,J)`；本实现元素访问序（列 j 升序、列内行升序）与其一致。
   差异仅在复 FMA 的实部分解次序：netlib（Fortran 复乘）
   `tR ← round(tR + round(round(xr·ar) − round(xi·ai)))`；本实现向量 4-Axpy
   `tR ← round(round(tR + round(xr·ar)) − round(xi·ai))`。每列每元素两者偏差 ≤ 1.5 ULP(tR)，
   随机数据下按 √L 累计：L=1025、|y|~O(10²) 时约 1e-3 量级，远小于
   `atol + rtol·|golden|`（|golden|~800 时 ≈ 0.098）与 max_abs_error_limit=1e-2；
2. **T/C 路径**：宽带向量 ReduceSum 为树状重排，相对误差 ~√L·2⁻²⁴ ≈ 1.4e-6（L=1025），
   远小于 rtol=1.22e-4。gbmv 无三角求解的误差放大链（区别于 tbsv 类算子——求解路径中
   逐位误差会被后续除法放大，本算子输出间无依赖），重排风险可控；确定性计算任务书明确
   不要求；
3. **Inf/NaN 语义**：不吞异常、按 IEEE 传播（Inf/NaN 填充用例 golden 同为 Inf/NaN）；
   beta=0 不读原 y（y 的 NaN 不污染结果）；alpha=0 不读 A/x；
4. **兜底**：若自验中某档实测超出混合容差（重点盯同号大数值、半带宽 k/n≈0.5 附近的大
   规模 TC_EX 档），该档 T/C 切换标量 netlib 同序点积（逐位一致）、N 路径切换逐元素
   标量复 FMA（同序），以精度优先、性能回退为代价；阈值经 1200 条全量自验后固化。

### 计算流程图

**Host 调用链：**

```
aclblasCgbmv(handle, trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy)
        │
        ▼
┌────────────────────────────────────────────────────────────┐
│ ① 校验（§Host 参数校验次序 1~12）                            │
│    handle/alpha/beta/trans/m/n → kl/ku/lda/inc → A/x/y     │
└───────────────────────┬────────────────────────────────────┘
                        ▼
┌────────────────────────────────────────────────────────────┐
│ ② 快速返回：m=0||n=0 → SUCCESS                              │
│            alpha=0 && beta=1 → SUCCESS（y 不触碰）           │
└───────────────────────┬────────────────────────────────────┘
                        ▼
┌────────────────────────────────────────────────────────────┐
│ ③ Tiling：路径选择 + coreNum/outPerCore 计算                 │
│    alpha=0→betaScale | outLen<32→gmTiny                     │
│    N→colSweep | T/C→rowDot                                  │
└───────────────────────┬────────────────────────────────────┘
                        ▼
              cgbmv_kernel_do(tiling, stream)
              （handle 绑定 stream 异步下发，host 即刻返回）
```

**N 列扫描 kernel（多核，每核行窗 [r0,r1)）：**

```
Init: SetGlobalBuffer(A/x/y) + 累加器清零
        │
        ▼
for j = max(0,r0-ku) … min(n-1, r1-1+kl):
  ┌─────────────┐   ┌──────────────┐   ┌────────────────────┐
  │ DataCopyPad  │──▶│ GatherMask   │──▶│ 4×Axpy(accR,accI)  │
  │ A 列段(ib..ie)│   │ → Ar, Ai     │   │  acc += x[j]·a     │
  │ (GM 连续)     │   │ (反交错)      │   │  (64 lanes 向量)    │
  └─────────────┘   └──────────────┘   └─────────┬──────────┘
        ▲                ping-pong 重叠搬入/计算  │
        └───────────────────────────────────────┘
        │
        ▼
┌────────────────────────────────────────────────────┐
│ 合并：t = alpha·acc (+ beta·y_old, beta≠0 才读)      │
│ 写回：incy=1 整段 DataCopy / incy≠1 逐元素散写        │
└────────────────────────────────────────────────────┘
```

**T/C 行内积 kernel（多核，每核输出列窗 [c0,c1)）：**

```
for j = c0 … c1-1:
  ┌─────────────┐   ┌──────────────┐
  │ DataCopyPad  │──▶│ GatherMask   │──▶ Ar, Ai (列段连续)
  │ A[lo_j..hi_j]│   │              │
  └─────────────┘   └──────────────┘
  ┌─────────────┐   ┌──────────────┐
  │ x 行段取入UB │──▶│ Xr, Xi       │ (incx=1 整段 / 否则 gather)
  └─────────────┘   └──────┬───────┘
                           ▼
  ┌────────────────────────────────────────────┐
  │ 复数点积：L_j ≥ 64 → 4×Mul + 4×ReduceSum ±   │
  │           L_j < 64 → 标量 4 累加器（netlib 同序）│
  └──────────────────────┬─────────────────────┘
                         ▼
  ┌────────────────────────────────────────────┐
  │ y[j] = alpha·d (+ beta·y_old, beta≠0 才读)   │
  │ 按 incy 写回（散写/连续）                     │
  └────────────────────────────────────────────┘
```

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas 800I/T A2（含 910B3，性能基准设备） | √ |
| Atlas 800I A3 | √ |
| Ascend 950PR/950DT（arch35） | 不在本任务范围（仓内已有 arch35 sgbmv，cgbmv arch35 为另行任务） |

说明：arch22 目录经根 CMake `get_soc_arch_dirs()` 同时覆盖 `ascend910b*`（A2）与
`ascend910_93*`（A3）两族 SOC，一套代码双平台验收；任务书 §7.4 要求 Atlas 800I A2 与
Atlas 800I A3 均完成验收，自验以 910B3 为主（性能口径），A3 款型补精度回归。

## 算子约束限制

| 约束项 | 内容 |
| --- | --- |
| 参数合法性 | §Host 校验次序表 1~12；非法参数返回 `ACLBLAS_STATUS_INVALID_VALUE`，handle 空返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| 数据类型 | 仅 COMPLEX64（A/x/y/alpha/beta）；实数档 sgbmv 不在本任务内 |
| 内存重叠 | y 与 A、x 不允许内存重叠（BLAS 常规约束，调用方保证）；不返回视图 |
| 非连续 Tensor | 不支持超出 lda/incx/incy 语义的非连续访问（任务书 §2.5） |
| broadcast | 不涉及（矩阵×向量，无广播） |
| dynamic shape | 不要求：m/n/kl/ku/lda/inc 均为运行时入参，host tiling 直接承载 |
| 空 Tensor | m=0 或 n=0 合法 no-op |
| 确定性 | 不要求（浮点累加顺序不保证逐位一致，任务书 §2.1.6） |
| 异步执行 | 依赖 `aclblasSetStream` 绑定 stream；读回 Device 结果前须同步 stream |

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | COMPLEX64 按 FLOAT32 实/虚部分别判定混合容差：逐元素 `\|actual−golden\| ≤ atol + rtol×\|golden\|`（atol=rtol=2⁻¹³≈1.22e-4），整体 matched_ratio ≥ 0.99 且 max_abs_error ≤ max(1e-2, 32×ULP)；含规约场景 max_abs_error_limit 可按任务书 §3.2 酌情放宽至 2ULP | 生态算子开源精度标准 + 任务书 §3.2 |
| golden | Netlib BLAS `cgbmv`（测试工程内接语义实现，随测试代码交付） | 任务书 §3.1 |
| 性能标准 | 910B3 上 5 条典型 case 平均单次 kernel 耗时（msprof Task Duration，warmup 后有效采样>10 次取平均）≤ 达标线（下表）；200 条 TC_PF 逐条 ≤ gpu_baseline.csv 的 gpu_ms/0.8 | 任务书 §3.3 + 测试指导 |
| 内存 | 任务书 §3.4 不涉及；设计为零 workspace、零动态 Device 分配（单次 launch） | 任务书 §3.4 |

任务书 5 条典型性能 case（COMPLEX64，均为 incx=incy=1、lda=kl+ku+1 纯连续访存；
达标线为验收目标非实测）：

| case | m×n | kl/ku | trans | alpha/beta | 达标耗时（us） | 有效数据量（GM 读+写） | 隐含带宽要求 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 256×256 | 16/16 | N | 1/0 | 10.665 | ~75KB | 固定开销主导（~7GB/s） |
| 2 | 512×512 | 32/32 | N | 1/1 | 17.328 | ~292KB | ~17GB/s |
| 3 | 1024×1024 | 64/64 | T | 1/0 | 25.875 | ~1.13MB | ~44GB/s |
| 4 | 2048×2048 | 128/128 | N | 1/0 | 30.640 | ~4.4MB | ~144GB/s |
| 5 | 4096×4096 | 64/256 | C | 1/0 | 52.487 | ~10.9MB | ~208GB/s |

带宽分析结论：case 4/5 进入带宽敏感区，设计以"列段连续搬运 + 反交错/点积全向量化 +
满核并行"为主攻方向（A 掷取已成 stride-1，消除一切标量 A 访问）；case 1/2 为固定开销
敏感区，以"薄路径（少核直发、无多余阶段）"为主攻方向。910B3 HBM 聚合带宽与 40 AIV
并发搬运能力对 208GB/s 量级留有达标窗口，最终以 910B3 实测闭环。

## 测试设计

测试工程落位 `test/gbmv/cgbmv/arch22/`（文件结构参考仓内 `test/tbsv/stbsv/arch22/`）：
`cgbmv_test.cpp`（CSV 驱动 GTest + 混合容差判定）、`cgbmv_test.csv`（随任务 1200 条）、
`cgbmv_golden.h`（Netlib `cgbmv` 语义 golden）、`README.md`（复现步骤）。

| 类别 | 前缀 | 条数 | 覆盖 |
| --- | --- | --- | --- |
| 基础/尺寸 | TC_L0/TC_SQ | 75 | trans 全枚举 × 尺寸 1→2048 扫描（奇数/2 的幂±1/非对齐） |
| 标量 | TC_AB | 24 | alpha/beta ∈ {0, 1, -1, 一般复数, 纯虚} × 3 枚举 |
| 矩形 | TC_RC | 36 | fat（n≫m）/ thin（n≪m）× 3 枚举 |
| 前导维/带宽 | TC_LD/TC_BW | 22 | lda padding；kl/ku ∈ {0, 上界, 全带, 单边带} |
| 步长 | TC_INC | 36 | incx×incy ∈ ±1/±2/±3 全组合（负步长） |
| 填充 | TC_FL | 12 | 均匀/全零/交替/极端值/Inf/NaN（A 与 x 各一遍） |
| 覆盖 | TC_CV | 24 | 中等尺寸 × 3 枚举 |
| 边界负向 | TC_ED | 19 | 零维/空指针×6/非法枚举/非法 lda/负维度/零步长/负 kl/ku |
| 扩展 | TC_EX | 752 | 尺寸×trans×带宽×标量×步长×padding 确定性采样 |
| 性能 | TC_PF | 200 | 5 条任务书 case 精确匹配 + 尺寸/带宽/网格扫描（纯连续访存） |

自验流程：`bash build.sh --ops=gbmv --soc=ascend910b3` → 精度
`--gtest_filter=-*TC_PF*`（逐条输出 matchedRatio/maxAbsErr）→ 性能 msprof 逐条采集
（`--gtest_filter` 收敛到单 case，取 OpBasicInfo.csv 的 Task Duration 跳过 warmup 后
平均），与 `gpu_baseline.csv` 逐条关联判 `≤ gpu_ms/0.8`。若官方用例未覆盖的场景
（如 incx=±3×incy=∓2 交叉、kl=m-1 满带 T 模式）自补用例，行为对齐 cuBLAS。

## 兼容性分析

1. **接口兼容**：`aclblasCgbmv` 为全新声明，参数序列与 `cublasCgbmv`/仓内
   `aclblasSgbmv` 一一对应（handle + trans/m/n/kl/ku/alpha/A/lda/x/incx/beta/y/incy），
   不改动任何既有接口；声明放入 `include/cann_ops_blas.h` 供其他产品线共用
   （交付件④要求）；
2. **构建兼容**：新增 `blas/gbmv/arch22/` 目录被根 CMake 架构 glob 自动纳入
   （`ascend910b*`/`ascend910_93*` → arch22），不影响 arch35 sgbmv 的既有构建；
3. **文档兼容**：`blas/gbmv/README.md` 增补 aclblasCgbmv 小节，产品支持表标注
   Atlas A2/A3 系列产品（含 Atlas 800I A2/A3）：支持；Ascend 950PR/DT 维持现状；
4. **代码仓规范**：文件按仓内 arch22 先例组织（`cgbmv_host.cpp` / `cgbmv_kernel.cpp` /
   `cgbmv_kernel.h` / `cgbmv_tiling_data.h` 四文件 + 测试目录），无冗余文件；
   合入遵循 CONTRIBUTING.md，先在个人 fork（已邀请 Ascend-CANN 为开发者）自验，
   再向 cann/ops-blas 提 PR（`blas/gbmv/arch22/` 与 `test/gbmv/cgbmv/arch22/`）。
