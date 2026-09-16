> **状态**：设计评审稿 v1.0（2026-09-15）。报名已通过；本稿申请设计评审，尚未取得 NPU 编译、精度或性能验证结果。
>
> **提交账号**：DR_Tony　　**目标任务**：广州实操工坊 aclblasCsymv（A2/A3）

---

# 需求背景

## 需求来源

昇腾 CANN 社区任务——在 Atlas A2/A3（910B3）上开发 `aclblasCsymv` 单精度复数对称矩阵-向量乘算子，合入 ops-blas 开源仓。

## 背景介绍

### 算子功能

`aclblasCsymv` 计算 `y = α·A·x + β·y`，其中 A 为 n×n 复数**对称**矩阵（A = Aᵀ，非厄米特、不取共轭），α/β 为复数标量，x/y 为复数向量。仅 uplo 指定的三角被存储，另一半由对称性推断。

### 与 cgeru/chemv 的本质区别

这个算子的核心难点不在"复数乘法"，而在**对称矩阵的访存模式**。cgeru 是秩一更新 A += α·x·yᵀ，访存规整（两个向量外积）；chemv 取共轭。csymv 的挑战是：对称性导致同一行的元素分散在两个三角区域，且列主序下"取一行"本身就是跨步访问。

# 需求分析

## 需求描述

使用 Ascend C 实现 aclblasCsymv，接口语义对标 cublasCsymv，支持 UPPER/LOWER 两种填充模式，支持正负步长，支持各种 quick return 退化路径。

## 需求拆解

1. 实现 y = α·A·x + β·y 的复数对称矩阵-向量乘核心计算
2. 正确处理对称访存——不取共轭
3. 设计高效的 Tiling 策略，解决列主序对称矩阵的不规则访存问题
4. 支持 incx/incy 正负步长
5. 处理所有 quick return / 退化路径
6. 性能达到任务书标杆要求

# 详细设计

## 算子分析

### 数学公式

```
y = α · A · x + β · y
```

复数乘法：(a+bi)(c+di) = (ac−bd) + (ad+bc)i

对称性：A(i,j) = A(j,i)（直接相等，不取共轭）

### 核心计算分析

展开到行级别，对 y 的第 i 个元素：

```
y[i] = α · Σ_{j=0}^{n-1} A(i,j) · x[j] + β · y[i]
```

对称矩阵使得这个求和并不是简单的"读一行乘一个向量"——因为 A 只存了一半。

**以 LOWER 模式为例**，A(i,j) 在 j ≤ i 时存储在 A[j*lda + i]（列主序），在 j > i 时需要读转置位置 A[i*lda + j]。把求和拆成两段：

```
y[i] = α · [ Σ_{j=0}^{i} A[j·lda+i]·x[j]          ← "自身列段"：跨 i+1 列，每列取一个元素（跨步 lda）
           + Σ_{j=i+1}^{n-1} A[i·lda+j]·x[j] ]      ← "转置列段"：第 i 列中 j=i+1..n-1（内存连续！）
       + β · y[i]
```

**关键洞察**：转置列段是第 i 列的一段连续内存，可以高效 DMA 搬运；但自身列段需要从 i+1 个不同的列各取一个元素，是跨步 lda 的离散访问。

逐行处理这种离散访问效率极低。所以我们不能按"逐行取一行→做点积"的思路来设计。

### 访存模式选型——列扫描法

参考 Netlib ssymv 的算法思路：**不按行遍历，而是按列遍历**。每处理一列 j，更新所有相关的 y[i]。这样每次访问的都是 A 的一整列（或列的一段），在列主序下是连续内存。

**LOWER 模式的列扫描伪代码**：

```
// 先做 β 缩放
y[:] = β · y[:]

// 逐列扫描
for j = 0 to n-1:
    // 取出 α·x[j]，对整列广播
    axj = α · x[j]
    
    // 对角元素
    y[j] += axj · A(j,j)
    
    // 列 j 的下三角段：A(j+1..n-1, j) → 连续内存 A[j·lda + j+1 .. j·lda + n-1]
    y[j+1..n-1] += axj · A(j+1..n-1, j)     ← 向量乘加，连续搬运
    y[j]        += α · dot(A(j+1..n-1, j), x[j+1..n-1])  ← 向量点积
```

每一列访问的 A 数据都是**列主序连续**的，DMA 友好。但问题是：每一列都写 y 的一个子范围，而且不同列写的范围不同（列 0 写 y[1..n-1]，列 n-2 只写 y[n-1]），多核并行怎么分？

## 算子实现

### 实现方案

#### host 侧设计

##### 1. 分核策略：按输出行划分

多核并行的维度选择：**按 y 的行范围划分**。每个 core 负责 y[rowStart..rowEnd) 的最终结果。

原因：按列划分会导致多个 core 写同一个 y[j]，需要 inter-core 归约，同步开销大。按行划分后，每个 core 独立写自己负责的 y 段，无需同步。

```
coreNum = min(可用核数, n)  // n 很小时不必满核
rowPerCore = ceil(n / coreNum)
// 大小核：前 (n % coreNum) 个核多处理 1 行
bigCoreCount = n % coreNum
bigCoreRows  = rowPerCore
smallCoreRows = rowPerCore - 1 (当 bigCoreCount > 0)
```

n 很小（比如 n ≤ 核数）时，每核只分 1 行甚至空跑，此时动态缩减核数避免空核。

##### 2. tiling 策略：核心在于"列段分块"

每个 core 虽然只负责 y[rowStart..rowEnd) 的输出，但计算时需要扫描矩阵的**所有 n 列**。对于每一列 j，该 core 需要：

- 读 A 的列 j 中与自己行范围相交的段
- 读 x[j]（以及部分 x 段用于点积）
- 更新 y 的局部段

**列段分块的原因**：当 n 很大时，一列的数据可能超过 UB 容量。需要将每一列的处理再分成若干 tile。

**但更重要的是**，对于每个 core 负责的行范围 [rS, rE)，处理列 j 时，实际需要的 A 数据长度取决于 j 与 [rS, rE) 的关系（以 LOWER 为例）：

```
列 j 在 LOWER 模式存储了行 j..n-1 的数据

如果 j < rS:
  A(rS..rE-1, j) 全部在存储三角内（因为 rS > j）
  → 连续读 A[j·lda + rS .. j·lda + rE-1]，长度 = rE - rS
  → 是标准的"向量乘标量再加到 y_buf"

如果 rS ≤ j < rE:（列 j 的对角线落在我的行范围内）
  分两部分：
  a) A(j..rE-1, j)：存储三角，连续读，做 axpy 到 y_buf[j-rS..rE-rS-1]
  b) A(rS..j-1, j) 不在存储三角 → 读转置位置 A[rS·lda+j .. (j-1)·lda+j]，跨步 lda
     或者换思路：这部分等价于 dot(A(j, rS..j-1), x[rS..j-1])，累加到 y_buf[j-rS]
  c) 对角 A(j,j)：单元素

如果 j ≥ rE:
  A(rS..rE-1, j) 不在存储三角（因为所有行 < j）
  → 读转置位置：对每行 i ∈ [rS,rE)，读 A[i·lda + j]
  → 这恰好是"第 rS 列的第 j 个元素、第 rS+1 列的第 j 个元素..."，跨步 lda
```

**简化策略**：将列 j 的处理分为三个阶段（phase），host 侧不需要逐列算，只需传入 n/lda/行范围，kernel 内自行判断每列属于哪个 phase。

##### 3. 数据分块和内存优化策略

910B3 的 UB 约 192~256 KB。complex64 每元素 8 字节。需要分配的 UB buffer：

```
bufCol:   一列中属于我的行范围的 A 数据      最大 rowPerCore 个元素 × 8B
bufX:     x 向量的一段（用于点积）            最大 rowPerCore 个元素 × 8B
bufY:     y 的局部累加缓冲                   rowPerCore 个元素 × 8B
bufTmpRe/bufTmpIm: 复数运算拆分缓冲          2 × rowPerCore × 4B
```

总 UB ≈ rowPerCore × (8 + 8 + 8 + 8) = 32 × rowPerCore 字节

- rowPerCore = 4096 时：32 × 4096 = 128 KB → 可以放下
- 预留 double buffer 则 ×2 = 256 KB → 需要更细的列段 tile

**列段 tile 大小计算**：

```
UB_AVAIL = GetUBSize() - 保留区域
ELEM_BYTES = 8  // complex64
BUF_COUNT = 6   // bufCol×2(double buffer) + bufX + bufY + 2×tmp
colTileSize = UB_AVAIL / (BUF_COUNT × ELEM_BYTES)
colTileSize = AlignDown(colTileSize, 4)  // 32B 对齐（4 个 complex64 = 32B）
```

当 rowPerCore > colTileSize 时，需要在行方向也做二级 tile（rowTile），让每次处理 rowTile 行 × colTileSize 列的子块。

##### 4. 路径分派策略（kernel 直调，无 TilingKey）

本算子采用 kernel 直调模式，host 侧不经过 TilingKey/框架二进制匹配流程，而由 **tiling 标量字段承载全部分派信息**：host 侧根据 `(uplo, n, incx, alpha 是否为零)` 计算 tiling 结构（uplo 方向、实际启用核数、列块大小、x 分块参数、quick return 标志等），序列化后随启动参数下发；kernel 侧读取标量字段完成分支分派。

**uplo 统一**：上下三角通过列内行区间的起止表达式统一——以对角线为界的下标比较方向翻转即可。具体地，对于列 j 在本 core 的行范围 [rS, rE) 内：

```
LOWER: 存储段 = [max(j, rS), rE)     转置段 = [rS, min(j, rE))
UPPER: 存储段 = [rS, min(j+1, rE))   转置段 = [max(j+1, rS), rE)
```

只需一个方向标志 `uplo` 控制上述两个表达式的切换，单一 kernel 模板即可覆盖 UPPER/LOWER 两种填充模式，避免维护两份代码。

**quick return 短路**：`n = 0`、`alpha = (0,0) && beta = (1,0)` 的 quick return 在 host 侧短路——完成参数校验后直接返回 `ACLBLAS_STATUS_SUCCESS`，不启动 kernel。`alpha = (0,0) && beta ≠ (1,0)` 的退化缩放路径同样在 host 侧判断，可选择启动一个轻量 scale kernel 或在主 kernel 内通过 tiling 标志字段跳过矩阵读取。

**与 TilingKey 方案的对比**：

| 维度 | TilingKey 分派 | 标量字段分派（本方案） |
|------|---------------|----------------------|
| 分支选择 | host 设 key → kernel 查表跳转 | kernel 读 tiling 字段内联判断 |
| 新增分支 | 需注册新 key + 对应 kernel 入口 | 加一个 tiling 字段即可 |
| 二进制数 | 每个 key 可能对应独立编译体 | 单一 kernel 二进制 |
| uplo 处理 | 通常 2 个 key 或 2 套循环 | 一套循环 + 方向标志 |

本方案更适合 csymv 这种"分支少但参数组合多"的场景——uplo × 步长 × 标量特殊值的排列组合用 TilingKey 会膨胀，而标量字段天然支持连续值域。

##### 5. TilingData 结构体

```cpp
struct CsymvTilingData {
    int32_t n;
    int32_t lda;
    int32_t incx;
    int32_t incy;
    int32_t uplo;             // 0=UPPER, 1=LOWER（方向标志，kernel 内统一表达式）
    float   alphaReal, alphaImag;
    float   betaReal,  betaImag;
    int32_t totalCoreNum;     // 实际启用核数（≤ 硬件核数，≤ n）
    int32_t bigCoreCount;     // 多分 1 行的核数
    int32_t bigCoreRows;      // 大核行数
    int32_t smallCoreRows;    // 小核行数
    int32_t colTileSize;      // 列段 tile 大小（32B 对齐）
    int32_t alphaIsZero;      // host 预判：α==(0,0) → 1，否则 0
    int32_t betaIsZero;       // host 预判：β==(0,0) → 1，否则 0
    int32_t betaIsOne;        // host 预判：β==(1,0) → 1，否则 0
};
```

`alphaIsZero`/`betaIsZero`/`betaIsOne` 三个标志字段替代了 TilingKey 的分支选择功能：kernel 侧据此决定是否跳过矩阵读取、是否跳过 β 缩放、是否跳过 y 旧值读取，均为 `if` 内联判断，无需查表跳转。

#### kernel 侧设计

##### 1. Init 阶段

```cpp
void Init() {
    blockIdx = GetBlockIdx();
    // 计算本 core 的行范围
    if (blockIdx < bigCoreCount) {
        rowStart = blockIdx * bigCoreRows;
        myRows = bigCoreRows;
    } else {
        rowStart = bigCoreCount * bigCoreRows + (blockIdx - bigCoreCount) * smallCoreRows;
        myRows = smallCoreRows;
    }
    rowEnd = rowStart + myRows;
    
    // 分配 UB buffer
    AllocBuffers();
    
    // 预计算步长偏移
    xOff = (incx > 0) ? 0 : -(n-1) * incx;
    yOff = (incy > 0) ? 0 : -(n-1) * incy;
}
```

##### 2. Process 阶段——列扫描核心循环

以 LOWER 模式为例：

```cpp
void ProcessLower() {
    // Phase 0: β 缩放 y
    ScaleY();
    
    // Phase 1: 列扫描
    for (int j = 0; j < n; j++) {
        complex axj = alpha * GetX(j);
        
        if (j < rowStart) {
            // 列 j 完全在我行范围的"上方"
            // A(rowStart..rowEnd-1, j) 全在存储三角内（行 > j）
            // → 连续读 A[j*lda + rowStart], 长度 myRows
            CopyColSlice(j, rowStart, myRows);
            // y_buf[0..myRows-1] += axj * bufCol[0..myRows-1]
            ComplexAxpy(bufY, bufCol, axj, myRows);
            // 同时做点积给 y[j]：这里 j 不在我的行范围，需要写回
            // → 但 y[j] 不归我管！所以这个点积的结果需要额外处理
        }
        else if (j >= rowStart && j < rowEnd) {
            // 对角列——分裂处理
            int localJ = j - rowStart;
            
            // a) 对角元素
            complex ajj = ReadA(j, j);
            bufY[localJ] += axj * ajj;
            
            // b) 下三角段 A(j+1..rowEnd-1, j)（如果有的话）
            int belowLen = rowEnd - j - 1;
            if (belowLen > 0) {
                CopyColSlice(j, j+1, belowLen);
                ComplexAxpy(&bufY[localJ+1], bufCol, axj, belowLen);
                // 点积贡献回 y[j]
                complex dot = ComplexDot(bufCol, &bufX_local[localJ+1], belowLen);
                bufY[localJ] += alpha * dot;
            }
            
            // c) 上方 A(rowStart..j-1, j) — 在存储三角内
            int aboveLen = j - rowStart;
            if (aboveLen > 0) {
                CopyColSlice(j, rowStart, aboveLen);
                ComplexAxpy(bufY, bufCol, axj, aboveLen);
            }
        }
        else {
            // j >= rowEnd：列 j 完全在我行范围的"下方"
            // A(rowStart..rowEnd-1, j) 不在存储三角（行 < j）
            // → 读转置位置：A(j, rowStart..rowEnd-1) = A[rowStart*lda+j .. (rowEnd-1)*lda+j]
            // 这是跨步 lda 的访问，每步取 1 个元素
            CopyTransposed(j, rowStart, myRows);
            ComplexAxpy(bufY, bufCol, axj, myRows);
        }
    }
    
    // Phase 2: 写回 y
    CopyOutY();
}
```

**但上面有个问题**：列 j < rowStart 时的点积 `dot(A(j+1..n-1, j), x[j+1..n-1])` 的贡献需要加到 y[j]，而 y[j] 不在本 core 的负责范围。

**解决方案**：改用"行视角"处理这部分。每个 core 只计算自己负责的 y 行，对于第 i 行（i ∈ [rS, rE)），把内积拆分处理：

```
y[i] = α · Σ_j A(i,j)·x[j] + β·y[i]

拆成：
  连续段：A(i, i+1..n-1) 从转置位置读 → A[i*lda + i+1 .. i*lda + n-1] 连续
  跨步段：A(i, 0..i) 从列位置读 → A[0*lda+i], A[1*lda+i], ..., A[i*lda+i] 跨步 lda
```

**实际策略选择：混合法**

对于本 core 的行范围 [rS, rE)，列 j 的处理分两类：

**a) 批量列处理（j < rS 或 j ≥ rE）**：
  - 列 j 对 y[rS..rE-1] 的贡献是一个"向量乘标量加"操作
  - 可以一次搬入多列的 A 数据（列 tile），批量做 axpy
  - 不涉及点积回写到范围外的 y

**b) 对角列处理（rS ≤ j < rE）**：
  - 需要精细处理对角分裂
  - 每次只处理一列，额外做列内点积

这样就完全避免了 core 间的写冲突。

##### 3. 复数运算的向量化

复数 interleaved 格式 [re0, im0, re1, im1, ...] 下，做复数向量乘标量 (sr, si)：

```
// 提取实部虚部
Ar[k] = buf[2k],   Ai[k] = buf[2k+1]

// 乘标量 (sr + si·i)
outR[k] = Ar[k]·sr - Ai[k]·si    // Mul, Mul, Sub
outI[k] = Ar[k]·si + Ai[k]·sr    // Mul, Mul, Add

// 写回 interleaved
out[2k] = outR[k], out[2k+1] = outI[k]
```

每个复数标量乘需要 4 次 Mul + 2 次 Add/Sub → 利用向量指令一次处理 colTileSize 个元素。

对于复数向量点积（归约到标量），先做逐元素复数乘，再用 ReduceSum 分别对实部虚部归约。

##### 4. 跨步列访问的处理

当需要读跨步数据（stride = lda）时，有两种方式：

- **DataCopy with stride 参数**：设置 srcStride = lda，dstStride = 1，一次 DMA 搬运
- **逐元素 gather**：循环搬运，效率低但通用

优先使用 stride DataCopy。当 lda 与 n 不等（有 padding）时 stride 模式同样适用。

##### 5. 步长处理

incx/incy ≠ 1 时，x 和 y 在 GM 中不连续。处理方式：

- **incx=1 且 incy=1（连续路径）**：kernel 内判断 incx/incy 值，直接 burst 搬运，效率最高
- **通用路径**：搬运 x/y 时使用 stride DataCopy 或循环搬运

负步长通过预计算 offset 转为正向遍历：`kx = (incx>0) ? 0 : -(n-1)*incx`

## 支持硬件

| 芯片版本 | 支持 |
|----------|------|
| Atlas 800I/T A2 (910B3) | √ |
| Atlas A3 系列 | √ |

## 算子约束限制

- 数据类型仅 COMPLEX64（实部/虚部各 float32）
- n ≥ 0，lda ≥ max(1,n)，incx ≠ 0，incy ≠ 0
- A 为对称矩阵，推断时不取共轭
- y 原地覆写
- 不支持 broadcast、不要求 dynamic shape

# 可维可测分析

## 精度标准/性能标准

| 标准 | 描述 |
|------|------|
| 精度 | COMPLEX64 实部/虚部分别按 FLOAT32 判定：rtol=2⁻¹⁰, atol=2⁻¹⁶, matched_ratio ≥ 0.99 |
| 性能 | n=512/2048/4096 各 case 不超过任务书标杆耗时（15.45~146 us） |

## 兼容性分析

新增接口 aclblasCsymv 至 `include/cann_ops_blas.h`，与已有 aclblasSsymv 同族，不涉及兼容性问题。
