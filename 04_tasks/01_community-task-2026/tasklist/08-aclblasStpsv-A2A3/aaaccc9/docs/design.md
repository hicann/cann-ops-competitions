# 需求背景

## 需求来源

CANN 社区任务广场 aclblasStpsv A2/A3 算子开发任务（taskId: f5d4a88ec52f4508a489ef53f8e534a1）。

## 背景介绍

在昇腾 NPU（Atlas 800I A2 / Atlas 800I A3）上使用 Ascend C 编程语言开发单精度打包三角方程组求解算子 aclblasStpsv，对标 BLAS 的 `stpsv` 接口。

# 需求分析

## 需求描述

求解 `op(A) * x = x`，其中 A 为 n×n 三角矩阵、按 packed 格式存储，解向量 x 原地覆写。

## 接口定义

```c
aclblasStatus_t aclblasStpsv(aclblasHandle_t handle, aclblasFillMode_t uplo,
                             aclblasOperation_t trans, aclblasDiagType_t diag,
                             int n, const float* AP, float* x, int incx);
```

| 参数 | 取值 | 说明 |
|---|---|---|
| handle | 非空 | aclblas 句柄，stream 取自其中 |
| uplo | ACLBLAS_UPPER / ACLBLAS_LOWER | A 的存储三角 |
| trans | ACLBLAS_OP_N / _T / _C | op(A) = A / A^T / A^H；实数域下 T 与 C 等价 |
| diag | ACLBLAS_NON_UNIT / ACLBLAS_UNIT | 对角元是否隐含为 1 |
| n | >= 0 | 阶数；n = 0 直接返回成功 |
| AP | 非空 | packed 三角，长度 n(n+1)/2 |
| x | 非空 | 步长为 incx 的解向量，原地覆写 |
| incx | != 0 | x 的步长，可为负 |

## 需求拆解

1. 支持 uplo/trans/diag 全组合共 8 种模板实例
2. 支持 incx 正负步长与任意非零步长
3. 前代 / 回代顺序依赖，单核执行
4. packed 存储，索引按 uplo 分两套公式
5. 参数非法时返回对应错误码，不下发 kernel

## 边界与异常

| 场景 | 行为 |
|---|---|
| handle == nullptr | ACLBLAS_STATUS_HANDLE_IS_NULLPTR |
| uplo / trans / diag 非法枚举 | ACLBLAS_STATUS_INVALID_VALUE |
| n < 0 | ACLBLAS_STATUS_INVALID_VALUE |
| n == 0 | ACLBLAS_STATUS_SUCCESS（不解引用 AP / x） |
| incx == 0 | ACLBLAS_STATUS_INVALID_VALUE |
| AP == nullptr 或 x == nullptr | ACLBLAS_STATUS_INVALID_VALUE |

# 详细设计

## 算子分析

### 数学公式

- 前代（op(A) 为下三角）：`x[i] = (x[i] - Σ_{j<i} op(A)[i][j] * x[j]) / A[i][i]`
- 回代（op(A) 为上三角）：`x[i] = (x[i] - Σ_{j>i} op(A)[i][j] * x[j]) / A[i][i]`

`diag = UNIT` 时跳过除法。

### packed 索引

```
UPPER: idx(i, j) = i + j * (j + 1) / 2                 (i <= j)
LOWER: idx(i, j) = i + (2 * n - j - 1) * j / 2         (i >= j)
```

两式均以列 j 为外层递增量，同一行内相邻列的存储间距随 j 变化，因此 packed 三角沿行方向**不是等步长连续**的。

### 依赖关系与并行性

第 i 个解分量依赖全部前序解分量，构成严格串行链（x[i] 的计算必须等 x[0..i-1] 全部就绪）。这使 tpsv 无法像 spr / gbmv 类算子那样按行或按列切分到多核——这是本算子与其余任务最本质的差别，也是采用单核标量路径的根本原因。

## Host/Tiling 设计

### 参数校验（host 侧）

Host 入口位于 `blas/tpsv/arch22/stpsv_host.cpp`，在构造 tiling 之前完成全部入参校验，顺序为：

`handle` → `uplo` → `trans` → `diag` → `n` → `incx` → 指针非空

校验失败直接返回对应错误码，不下发 kernel。其中 `n == 0` 的短路返回刻意放在指针检查**之前**，保证空矩阵场景不会因解引用空指针而报错，符合 BLAS 惯例。

### Tiling 结构

```c
struct StpsvTilingData {
    uint64_t ap;      // AP 的 device 地址
    uint64_t x;       // x  的 device 地址
    uint32_t n;       // 阶数
    uint32_t uplo;    // 0 = UPPER, 1 = LOWER
    uint32_t trans;   // 0 = N, 1 = T / C
    uint32_t diag;    // 0 = NON_UNIT, 1 = UNIT
    int64_t  incx;    // 步长，保留符号
};
```

地址用 `uint64_t` 承载，避免 32 位平台截断；`incx` 用有符号 `int64_t`，负步长信息不在 host 侧归一化，交由 kernel 的 `XOffset` 处理。

### 核数决策

tpsv 求解链串行，多核并行无收益。因此：

- tiling **不切分数据**，固定单核执行；
- kernel 以 `<<<1, nullptr, stream>>>` 启动，blockDim = 1；
- host 侧不做 `GetAivCoreCount()` 查询，避免在无 AIV 的形态下引入额外失败点。

对比 spr / sspr2 / gbmv 等可并行算子，其 host 侧需要 `GetAivCoreCount()` + `min(aivCoreNum, MAX_CORE_NUM)` 来决策核数；stpsv 因串行依赖而无需这一步。

### 下发路径

tiling 按值传给 `stpsv_kernel_do(const StpsvTilingData& tiling, void* stream)`，kernel 直调（不经过动态参数解析的 launch 框架），stream 取自 `handle->stream`，与上层调用者的流保持一致。

## Kernel 设计

### 模板与分派

```cpp
enum class TpsvUplo  { UPPER, LOWER };
enum class TpsvTrans { NO_TRANS, TRANS };
enum class TpsvDiag  { UNIT, NON_UNIT };

template <TpsvUplo UPLO, TpsvTrans TRANS, TpsvDiag DIAG>
class StpsvKernel { ... };
```

`DEFINE_TPSV_KERNEL(uplo, trans, diag, name)` 宏展开出 8 个 `__global__ __aicore__` 入口（uplo × {N, T/C} × {NON_UNIT, UNIT}），每个入口内声明 `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY)`，限定只跑 AIV。

`stpsv_kernel_do` 按 `tiling.uplo / tiling.trans / tiling.diag` 做 8 路 switch 分派，把运行期枚举折叠回编译期模板参数，使后续 `if constexpr` 能完全消除分支。

### 求解主体

```cpp
if constexpr (kForward) {
    for (uint32_t i = 0; i < n; ++i) {
        float sum = xGM.GetValue(XOffset(i));
        for (uint32_t j = 0; j < i; ++j) {
            sum -= GetElemOffDiag(i, j) * xGM.GetValue(XOffset(j));
        }
        if constexpr (DIAG == TpsvDiag::NON_UNIT) { sum = sum / GetDiag(i); }
        xGM.SetValue(XOffset(i), sum);
    }
} else {
    for (uint32_t i = n; i-- > 0; ) {
        float sum = xGM.GetValue(XOffset(i));
        for (uint32_t j = i + 1; j < n; ++j) {
            sum -= GetElemOffDiag(i, j) * xGM.GetValue(XOffset(j));
        }
        if constexpr (DIAG == TpsvDiag::NON_UNIT) { sum = sum / GetDiag(i); }
        xGM.SetValue(XOffset(i), sum);
    }
}
```

```cpp
static constexpr bool kForward =
    (UPLO == TpsvUplo::LOWER && TRANS == TpsvTrans::NO_TRANS) ||
    (UPLO == TpsvUplo::UPPER && TRANS == TpsvTrans::TRANS);
```

`kForward` 把"下三角 + 不转置"与"上三角 + 转置"归为同一条前代路径，其余两种归为回代路径；8 个实例因此收敛成两条循环，只需在遍历方向上区分。

### 索引与步长

```cpp
TpsvPackedUpperIdx(i, j)    = i + j * (j + 1) / 2;
TpsvPackedLowerIdx(i, j, n) = i + (2 * n - j - 1) * j / 2;

XOffset(idx) = (incx >= 0) ? idx * incx : (n - 1 - idx) * (-incx);
```

`XOffset` 把负步长折叠成"从尾部起算的正偏移"，使主循环无需在运行期判断符号。`GetElemOffDiag` 通过 `if constexpr (TRANS ...)` 决定是否交换行列下标，转置情形不需要单独的数据重排。

### 容量与溢出（G.RES.02）

`SetGlobalBuffer` 的长度必须覆盖 kernel 实际会访问的跨度，否则越界 DMA 会在 device 侧报 `MTE accesses an invalid GM address`（`retCode = 0x31`）。两处乘法都先加宽到 `uint64_t`：

```cpp
const uint64_t absIncx = static_cast<uint64_t>(incx >= 0 ? incx : -incx);
const uint64_t apCount = static_cast<uint64_t>(n) * (static_cast<uint64_t>(n) + 1U) / 2U;
const uint64_t xCount  = (n > 0) ? (absIncx * static_cast<uint64_t>(n - 1U) + 1U) : 0U;
apGM.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(tiling.ap), apCount);
xGM.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(tiling.x), xCount);
```

原因：`n * (n + 1) / 2` 在 32 位下 n > 65535 即回绕，`absIncx * (n - 1)` 在大步长下回绕得更早；任一回绕都会给 `SetGlobalBuffer` 一个远小于实际跨度的长度，kernel 随后按真实下标访问即越界。packed 三角在 n = 92682 时本身已达 4 GiB float，64 位是唯一安全域。

## 硬件与约束

| 项 | 说明 |
|---|---|
| 目标形态 | Atlas 800I A2 / A3，arch22 |
| 编译 | Ascend C；`__global__ __aicore__` kernel + host 直调 |
| 计算单元 | 单核 AIV（`KERNEL_TYPE_AIV_ONLY`，blockDim = 1） |
| 内存 | AP / x 常驻 GM，kernel 以 `GlobalTensor<float>` 标量访问，不占 UB |
| 数据类型 | 仅 float32，无混合精度 |
| 数值判据 | rtol = 2^-10，atol = 2^-16，matched_ratio >= 0.99 |
| 已知限制 | 大 n 时受 GM 标量访问带宽约束 |

### 关于向量化的约束

R1 意见提到"循环内逐元素 GM 访问"。经评估，本算子**不具备**按现有 spr / sspr2 方案向量化的条件：

1. **无跨行并行**：求解链串行，无法像 spr 那样把行分给不同核；
2. **packed 索引非连续**：`DataCopyPad` 的 `srcStride` 是常量，而 `TpsvPackedUpperIdx(i, j) = i + j*(j+1)/2` 沿内层 j 的间距随 j 线性增长，等步长的 `DataCopyPad` 无法表达该模式，需 `Gather` + 每行重算索引序列；
3. **上游同族实现为同一形态**：已合入的 `blas/tpsv/arch35/stpsv_kernel.cpp` 与 `blas/stbsv/arch35/stbsv_kernel.cpp` 同样采用逐元素 `GetValue` / `SetValue`，未做向量化。

因此本 PR 保持标量路径与上游一致；若评审认为 arch22 也必须向量化，可按"x 常驻 UB + 每行 `Gather` 取 off-diagonal 段 + `Muls` + `ReduceSum` 求内积"的方案改写，需重跑一轮精度与性能验证。

## 可维可测

### 测试组织

`test/tpsv/stpsv/arch22/` 下 CSV 驱动 GTest：

| 文件 | 职责 |
|---|---|
| `stpsv_test.csv` | 声明用例（参数 + 期望结果 + 精度阈值） |
| `stpsv_param.h` | CSV 解析为参数结构 |
| `stpsv_test.cpp` | `TpsvArch22Test` 逐行执行并 `Verifier::verifyVector` 比对 |
| `stpsv_npu_wrapper.h` | host / device 内存准备、调用、回读 |

### 用例覆盖（24 条）

| 分组 | 条数 | 覆盖点 |
|---|---|---|
| TC_L0 | 6 | UPPER / LOWER × UNIT / NON_UNIT 基本路径，n = 0 / 1 / 32 |
| TC_L1 | 10 | trans = T / C，incx = ±1 / ±2 / 3 |
| TC_GEN | 4 | n = 13 / 100 / 128，小奇数与大尺寸 |
| TC_INV | 4 | 非法 uplo / trans / diag，incx = 0 |

### 可定位性

- Host 校验失败返回明确错误码，CSV 的 `expect_result` 列直接断言，非法入参不进 kernel；
- device 侧异常以 `retCode` + `errorStr` 上报，越界类问题可定位到具体 DMA；
- 精度阈值由 CSV 的 `mere_threshold` / `mare_multiplier` 两列参数化，调整阈值无需改代码；
- 8 个模板实例各自独立入口，可单独调用定位某一组合的问题。
