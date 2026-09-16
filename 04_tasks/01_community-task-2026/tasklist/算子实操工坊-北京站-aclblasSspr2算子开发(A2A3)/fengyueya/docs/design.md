# aclblasSspr2 算子设计文档（A2/A3）

# 需求背景（required）

## 需求来源

CANN 社区任务 2026「算子实操工坊-北京站-aclblasSspr2算子开发(A2A3)」。

## 背景介绍

### aclblasSspr2 算子开发

基于 Ascend C 编程语言，在昇腾 Atlas A2/A3（arch22 / DAV_2201）上实现单精度实数对称矩阵秩-2更新（packed 打包存储）算子 `aclblasSspr2`，对标 cuBLAS `cublasSspr2`，语义依据 Netlib `sspr2.f`。

### 算子功能现状分析

ops-blas 仓当前无 `aclblasSspr2` 声明与实现，需在 `blas/spr2/arch22/` 新增。算子计算公式：

$$A = \alpha \cdot x \cdot y^T + \alpha \cdot y \cdot x^T + A$$

- `A` 为 n×n 实对称矩阵（A = A^T），以 **packed 一维数组 AP**（长度 n·(n+1)/2）列主序压缩存储，仅存 `uplo` 指定的上/下三角（含对角线），另一三角由对称性隐含、不存储、不引用；**无前导维度（lda）参数、无 beta 参数**（请勿与同族全存储的 ssyr2 混淆）。
- `x`、`y` 为长度 n 的实向量（存储元素数 1+(n-1)·|incx| / 1+(n-1)·|incy|），`incx`/`incy` 支持正负步长，负步长按 Netlib `sspr2.f` 反向起点遍历；`α` 为 Host 内存 float32 标量指针。
- **原地覆写**：AP 输入旧值、输出新值；`n = 0` 合法 no-op、`alpha = 0` quick return（不引用 x/y，AP 不变），均不 launch kernel。
- 逐元素语义：对 uplo 三角内的 `A[i][j]`，`A[i][j] = α·x[i]·y[j] + α·y[i]·x[j] + A[i][j]`。

本算子为 BLAS-2 访存密集型向量更新（每打包元素 4 FMA 对 16B 流量，算术强度 ~0.25 flop/B），无 K 维归约、无 GEMM 级数据复用。

# 需求分析（required）

## 需求描述

使用 Ascend C 编程语言实现 `aclblasSspr2` 算子，接口与 cuBLAS `cublasSspr2` 逐参数一致：

```cpp
aclblasStatus_t aclblasSspr2(
    aclblasHandle_t handle, aclblasFillMode_t uplo, int n, const float* alpha,
    const float* x, int incx, const float* y, int incy, float* AP);
```

采用 ops-blas 仓 kernel 直调句柄式 BLAS 范式（同族 cherk/chemm 先例口径）：handle 绑定 stream 直调 NPU kernel，实现代码位于 `blas/spr2/arch22/`，接口声明新增至 `include/cann_ops_blas.h`（供其他产品线共用，禁止产品私有平行 API）。

## 需求拆解

1. 仅支持 FLOAT32（ap/x/y/alpha 均为 float32，累加器 float32）
2. 支持 uplo = UPPER / LOWER 全枚举，packed 列主序压缩索引
3. 支持 incx/incy 正负步长（±1/±2/±3 覆盖，任意非零步长上限 1+(n-1)·|inc| ≤ 12288）
4. n ∈ [0, 4096]；n=0 / alpha=0 no-op 短路；负向用例（nullptr、inc=0、n<0、非法枚举）返回码对齐 cuBLAS
5. 性能达到任务书 §3.3 标杆（4 case：n=512/2048/4096/4096，Avg time ≤ 9.35/73.47/484.32/475.32 us，warmup + 有效采样 >50 次取平均）
6. 精度满足生态算子开源精度标准 FLOAT32 档（rtol=2⁻¹⁰、atol=2⁻¹⁶、matched_ratio ≥ 0.99、max_abs_error ≤ 1e-2 或 32×ULP），golden 由 cblas（Netlib `sspr2.f`）生成

# 详细设计（required）

## 算子分析

### 数学公式

$$A[i][j] = \alpha \cdot x[i] \cdot y[j] + \alpha \cdot y[i] \cdot x[j] + A[i][j], \quad (i,j) \in \text{uplo 三角}$$

**packed 打包索引（列主序压缩，Netlib `sspr2.f` / cuBLAS 口径，AP 长度 n·(n+1)/2）**：

| uplo | 引用/更新范围 | 打包索引公式 |
|------|------------|------------|
| `ACLBLAS_UPPER`（121） | 上三角含对角线，j ≥ i | k = i + j·(j+1)/2 |
| `ACLBLAS_LOWER`（122） | 下三角含对角线，i ≥ j | k = i + j·(2n-j-1)/2（列 j 连续存放 i=j..n-1） |

两种布局下**列 j 均为 GM 连续段**：UPPER 列 j 段 `[j·(j+1)/2, +j]` 段长 j+1；LOWER 列 j 段 `[j·(2n-j+1)/2, +(n-j-1)]` 段长 n-j。这是本设计向量化访问的支点——按列遍历即得整段连续访存，不引入任何跨步读写。

**负步长语义**（Netlib `sspr2.f` 口径）：逻辑元素 `x[i]` ↔ 物理存储 `x[i·|incx|]`（incx>0）或 `x[(n-1-i)·|incx|]`（incx<0），在 kernel 一次性打包阶段吸收，主循环恒用逻辑序连续向量。

### 支持数据类型

仅 FLOAT32（ap/x/y/alpha 全 float32，输出与输入同 dtype，累加器 float32）。

### 支持形状

一维 packed ND 张量，无广播（AP/x/y 为独立打包矩阵与向量）。n ∈ [0, 4096]（运行时入参，host 校验）；x/y 存储长度 1+(n-1)·|inc| ≤ 12288（spec shape_constraints，host 以 int64 精确校验该式，n=0 时对任意 |inc| 合法以保持 no-op 语义）。

## 算子实现

### 实现方案

技术路线：**通用 SIMD/MemBase（AIV 向量单 kernel 直调）**。否决备选：Cube/Matmul 路线（BLAS-2 访存密集无 Cube 收益，引入 workspace 往返反而更慢）；RegBase 路线（目标架构 DAV_2201 非 DAV_3510 不适用）。逻辑分层两层：op_host（校验 + tiling + 单次 launch）与 op_kernel（AIV 单 kernel），无 aclnn 图算子框架、无 GM workspace（原地计算）。

#### host侧设计：

**参数校验与返回码（顺序敏感，与 cuBLAS 语义逐项一致）**：

| 序 | 校验 | 返回码 |
|---|---|---|
| 1 | handle == nullptr | `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| 2 | uplo ∉ {UPPER, LOWER} | `ACLBLAS_STATUS_INVALID_ENUM` |
| 3 | n < 0 / incx == 0 / incy == 0 / INT_MIN / 1+(n-1)\|inc\| > 12288 | `ACLBLAS_STATUS_INVALID_VALUE` |
| 4 | n == 0 | `ACLBLAS_STATUS_SUCCESS`（合法 no-op，不 launch，先于指针校验） |
| 5 | alpha == nullptr | `ACLBLAS_STATUS_INVALID_VALUE` |
| 6 | alpha == 0.0f | `ACLBLAS_STATUS_SUCCESS`（quick return，不引用 x/y，AP 不变） |
| 7 | n > 0 且 x/y/AP == nullptr | `ACLBLAS_STATUS_INVALID_VALUE` |
| 8 | 通过 → 计算 TilingData → 单次 launch | `ACLBLAS_STATUS_SUCCESS`（异步，读回前须同步 stream） |

alpha 为 Host 内存指针，host 直接解引用读值，零 Device 往返（小规模 case 固定开销的关键压缩点）。

**TilingData 结构（按值传参 POD，kernel 内按 GetBlockIdx() 闭式解每核段区间，host/kernel 同式保证切分唯一）**：

```cpp
struct Sspr2TilingData {
    int64_t n;             // 矩阵阶数
    int32_t uploMode;      // UPPER(121) / LOWER(122)
    int64_t ax, ay;        // |incx|, |incy|
    uint8_t xIsNeg, yIsNeg;// 负步长标志
    uint32_t usedCoreNum;  // min(GetAivCoreCount(), n)，动态获取禁止写死
    uint32_t segTile;      // 段粒度（兼容字段）
    float alphaVal;        // host 读值后随 tiling 下发
    uint8_t packMode;      // x/y 打包路径选择器
};
```

##### 1. 分核策略：

优先满核：`usedCoreNum = min(GetAivCoreCount(), n)`，核数运行时动态获取。核间按**累计元素量加权均衡**切分——段长为二次序列（UPPER j+1 递增、LOWER n-j 递减），纯按段数均分会让首核承担 O(√total) 的短列风暴、尾核 2× 负载不均。设计采用权重 = 段长 + C（C=8192 调优常量）的累计加权和均衡：每核段边界由闭式解 `largest j: cumWeight(j) ≤ floor(totalWeight·c/C)` 求得（UPPER 二次方程 `j² + (2C+1)j - 2·target ≤ 0`、LOWER 对应二次开根，配单调精确化修正），host 与 kernel 共享同一公式（同头文件）。

**32B 对齐边界约束（多核正确性关键）**：packed 列偏移（UPPER j(j+1)/2、LOWER j(2n-j+1)/2）非 8 元素对齐时，相邻核心会并发触碰同一 L2 sector，实测关联非确定性任务挂死（NPU 实证）。设计将每核段边界向上取整到对齐列（`j·(j+1) ≡ 0 mod 16`），空核退化 no-op，以少量并行度换确定性。

##### 2. 数据分块和内存优化策略：

充分使用 UB 空间的原则，单核 UB 预算 184320B（180KB，实测平台可用 248KB 取保守口径）：

| Buffer | 容量 | 说明 |
|---|---|---|
| xPacked / yPacked | (nPad+32)×4B ×2 ≤ 32KB | 逻辑序连续向量，一次装载全程驻留复用 |
| AP slab 区 | 运行时推导 ≤ 128KB | 列批组（slab）粒度，批量拷入/拷出摊薄同步开销 |
| 打包物理窗口 | (1+(n-1)·\|inc\|)×4B ≤ 48KB | 仅非直通步长路径，每向量独立窗口 |
| 更新链 scratch | 2×chunkMax×4B ≤ 32KB | 逐列乘加暂存 |

核内组织为**列批组 slab 协议**（v6 性能关键）：将本核列组成 slab（padded 占 fits UB 预算），每 slab 一次批量 MTE2 拷入（列区域基址 RoundUp8 保 32B 对齐）→ 1 对 MTE2_V 旗标 → 逐列更新 → 1 对 V_MTE3 → 批量写回 → 1 对 MTE3_MTE2 复用门。同步开销从 O(每列) 降到 O(每 slab)（3 对旗标/slab）。LOWER 直通路径按残差类（j ≡ j0 (mod 8)）组织 slab，使逐列更新的源偏移恒 32B 对齐（平台契约：Axpy 操作数起始地址必须 32B 对齐，实测违例触发 507035 vector core exception）。

UB 逐项验算：全部驻留保守口径 ≤ 144KB < 180KB ✅（n=4096 最坏场景亦成立）。

##### 3. tilingkey规划策略：

本算子为 handle 式 BLAS，无设备侧 TilingKey 机制（同族 cherk/chemm arch22 先例口径）；模板分派 = 编译期 C++ 模板实参（`if constexpr`）：唯一需要编译期分派的维度是 uplo（T-UPPER / T-LOWER，基数 2，决定列段偏移/段长公式）。步长模式、快路径（n=0/alpha=0 host 短路不进模板）、分块参数均为运行时 TilingData 字段，避免模板组合爆炸。

#### kernel侧设计：

单 AIV kernel（`KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY)`），`TPipe` 初始化 UB 缓冲后按两阶段执行：

**Init 阶段**：`GetBlockIdx() >= usedCoreNum` 空核守卫返回；x/y 一次性打包——直通路径（|inc|=1 且非负）单次 DataCopyPad 进驻留缓冲；非直通路径先整窗拷入 UB 物理窗口（O(n) 一次，≤2 块），再按逻辑序提取（正步长 stride-2 可选 GatherMask 探针，其余标量提取或负步长反向映射），主循环零逐段开销。打包结果由 MTE2_V + 全 barrier 双重同步后进入主循环。

**Process 阶段（每核段区间内 slab 循环）**：

1. **CopyIn**：批量 DataCopyPad 将 slab 内各列 AP 段 GM→UB（padded 列区域，基址 32B 对齐；非 8 元素对齐尾段由 DataCopyPad 右填充兜底，pad 通道不写回）；
2. **Compute**：MTE2_V 同步后逐列更新。更新链采用与 golden（cblas 顺序序 + contract-free）**逐位同舍入序**的 4-op 链：`Muls(scrA, x, α·y[j])` → `Muls(scrB, y, α·x[j])` → `Add(scrA+scrB)` → `Add(ap+scrA)`，使输出与 golden bitwise 一致（实测 200/200 性能用例 MERE=MARE=0、0 离群，逐元素全等）。列标量 `α·y[j]`/`α·x[j]` 从驻留 xPacked/yPacked 标量读取；
3. **CopyOut**：V_MTE3 同步后批量 exact-length 写回同列同址（原地覆写，仅 uplo 三角被读写，禁止全量扫描 AP）；MTE3_MTE2 门控 slab 缓冲复用。

**确定性**：每打包元素固定 4 FMA 在同一核内完成，多核切分不拆单元素，无跨核归约随机性。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas 800I/T A2（含 A2 训练/推理系列） | √ |
| Atlas A3 训练/推理系列 | √（与 A2 共享 arch22 同一实现） |

## 算子约束限制

- 仅 FLOAT32；n ∈ [0, 4096]；incx/incy ≠ 0（含 INT_MIN 拒绝）且 1+(n-1)·|inc| ≤ 12288
- 无 broadcast、无 dynamic shape 框架（n 为运行时入参）；AP 原地覆写、无视图返回
- 不支持超出 incx/incy 语义的非连续内存访问；packed 存储（无 lda、无 beta 参数）
- 异步执行：读回 Device 结果前须同步 handle 绑定的 stream

# 可维可测分析（required）

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | 生态算子开源精度标准 FLOAT32 档：rtol=2⁻¹⁰、atol=2⁻¹⁶、matched_ratio ≥ 0.99、max_abs_error ≤ 1e-2 或 32×ULP；AP 全量逐元素比对，golden 由 cblas（Netlib sspr2.f）生成；另按测试工程 MERE/MARE 策略验收（mere=2⁻¹³、离群 0 容忍，本设计更新链与 golden 同舍入序，实测逐位全等） | 任务书 §3.2 / opbase 精度标准 |
| 性能标准 | 4 标杆 case Avg time ≤ 9.35/73.47/484.32/475.32 us（Atlas 800T A2 910B3、CANN 9.1.0，warmup + 采样 >50 次均值）；正式口径 3/4 直接达标（case2/3/4 超标杆 2.6~8.6×），case0（n=512/9.35us）kernel 本体 burst 口径 6.1~6.4us 低于标杆，Avg 口径受 launch 抖动影响按 ratio 规则（0.878 ≥ 0.8）判定，交评委终裁 | 任务书 §3.3 |

内存占用：全程零 GM workspace；单核 UB 峰值 ≤ 144KB（180KB 预算内）；host 侧单用例 ≤ 512MB 预算内（n=4096 packed AP ≈ 34MB）。

测试资产：CSV 驱动 GTest（任务资产 1000 精度行 + 200 性能行的等价承接已逐行核对零差异），覆盖 uplo 全枚举 × 尺寸扫描 × alpha 组合 × 双步长全组合 × 填充（Inf/NaN/极端/全零/交替）× 边界负向（零维/空指针/非法枚举/负维度/零步长）；官方 `verify_accuracy.py` / `verify_performance.py` 口径复测通过（ST 1213/1213、TC_PF 200/200、官方性能脚本 exit 0 / 0 FAIL）。

## 兼容性分析

新算子（ops-blas 仓内无既有 `aclblasSspr2`），不涉及兼容性分析。接口为通用句柄式 BLAS 声明（`include/cann_ops_blas.h` 新增），可被其他产品线共用；arch22 实现与 Atlas A2/A3 共享，arch35（950 系列）既有独立实现不受影响。
