# CgemmEx 算子

## 算子概述

`aclblasCgemmEx` 算子在昇腾 NPU（**Ascend 950PR**）上实现复数扩展级通用矩阵乘法，核心运算为 `C = alpha * op(A) * op(B) + beta * C`，`alpha` / `beta` 为单精度复数标量（`aclblasComplex`，Host 内存），`A` / `B` / `C` 为 Device 内存矩阵指针，采用 **BLAS 标准列主序（Column-Major）** 存储。

区别于同族的 `aclblasCgemm`（固定 `aclblasComplex` 签名），本算子的 A / B / C 矩阵数据类型由 `typeA` / `typeB` / `typeC` 三个枚举参数独立指定，支持四条 type 路径：`ACLBLAS_C_32`（COMPLEX64，主路径）、`ACLBLAS_R_32`（FLOAT32）、`ACLBLAS_H_R_32`（FLOAT16）、`ACLBLAS_H_C_32`（COMPLEX32）。执行模型为 **Ascend C kernel 直调**：通过 `aclblasHandle_t` 句柄绑定 `aclrtStream` 异步下发 NPU kernel，读回 Device 结果前须同步 stream。

语义与参数序列对标 cuBLAS `cublasCgemmEx`，单批语义参考 Netlib BLAS `cgemm` 的 quick return 语义；接口声明置于 `include/cann_ops_blas.h`，**供其他产品线共用，不定义 950PR 私有平行 API**。

数学表达式：

```
C = alpha * op(A) * op(B) + beta * C
```

其中：
- `op(A) = A`（`transa = ACLBLAS_OP_N`）、`A^T`（`transa = ACLBLAS_OP_T`，不共轭）或 `A^H`（`transa = ACLBLAS_OP_C`，共轭转置）
- `op(B) = B`（`transb = ACLBLAS_OP_N`）、`B^T`（`transb = ACLBLAS_OP_T`，不共轭）或 `B^H`（`transb = ACLBLAS_OP_C`，共轭转置）
- `op(A)` 为 `m × k`，`op(B)` 为 `k × n`，`C` 为 `m × n`
- 复数路径下 `OP_T` 与 `OP_C` 语义不同：`T` 仅交换下标（`A^T_{ij} = A_{ji}`），`C` 取共轭后交换（`A^H_{ij} = conj(A_{ji})`，实部不变、虚部取反）

包含以下接口：

| 接口名 | 功能简述 |
|--------|---------|
| aclblasCgemmEx | 复数扩展级通用矩阵乘法，支持四条 type 路径（C_32 / R_32 / H_R_32 / H_C_32）、矩阵转置 / 共轭转置与 alpha/beta 缩放 |

## 算子执行接口

### aclblasCgemmEx

#### 产品支持情况

| 产品名称 | Ascend 950PR | Ascend 950DT | Atlas A3 训练系列 | Atlas A3 推理系列 | Atlas A2 训练系列 | Atlas A2 推理系列 |
|---|---|---|---|---|---|---|
| 支持情况 | **支持** | 不支持（未适配） | 不支持（未适配） | 不支持（未适配） | 不支持（未适配） | 不支持（未适配） |

> 本批次实测环境为 Ascend 950PR（full-soc `Ascend950PR_9579`，NpuArch `DAV_3510`，variant `dav_c310`，arch35），CANN 9.1.0。其他产品线尚未适配。

#### CANN 版本要求

- **CANN 版本**：9.1.0
- **asc-devkit**：≥ 9.1（arch35 的 Cube kernel 使用 `BlockMmad` 低阶 API，须 `ASC_DEVKIT_MAJOR >= 9` 且 `ASC_DEVKIT_MINOR > 0`）
- **架构目录**：`arch35`（`DAV_3510`）

#### 函数原型

```cpp
aclblasStatus_t aclblasCgemmEx(
    aclblasHandle_t handle,
    aclblasOperation_t transa,
    aclblasOperation_t transb,
    int m, int n, int k,
    const aclblasComplex* alpha,
    const void* A, aclblasType_t typeA, int lda,
    const void* B, aclblasType_t typeB, int ldb,
    const aclblasComplex* beta,
    void* C, aclblasType_t typeC, int ldc);
```

签名要素说明：

| 要素 | 说明 |
|------|------|
| 返回值 | `aclblasStatus_t`，语义见 `include/cann_ops_blas_common.h` |
| 参数个数 | 17 个参数 + 1 个返回值 |
| `alpha` / `beta` | `const aclblasComplex*`，指向 **Host 内存**中的单精度复数标量；不可为 `nullptr`；host 端直接解引用，无 device 拷贝 |
| `A` / `B` / `C` | `const void*` / `void*`，Device 内存矩阵指针；实际元素位宽由 `typeA` / `typeB` / `typeC` 决定，实现须按 type 对应位宽读写 |
| `typeA` / `typeB` / `typeC` | `aclblasType_t`（新增 `typedef aclDataType aclblasType_t;` 于 `include/cann_ops_blas_common.h`，语义等价于兄弟 Ex 接口使用的 `aclDataType`，不引入平行 API） |
| `m` / `n` / `k` | `int`，运行时入参（与 `cublasCgemmEx` 一致，非 `int64_t`） |
| `lda` / `ldb` / `ldc` | `int`，列主序前导维 |
| 批量语义 | 无（本接口为单批语义；批量接口为独立的 `aclblasGemmBatchedEx` / `aclblasGemmStridedBatchedEx`） |
| 计算精度 / 算法选择 | 签名中**不含** `computeType` 与 `algo` 参数（与 `aclblasGemmEx` 的差异点，以任务书签名为准） |

#### 参数说明（摘要）

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | ops-blas 库上下文句柄，携带 stream，Host 内存 |
| transa | 输入 | aclblasOperation_t | 矩阵 A 的操作类型：ACLBLAS_OP_N（不转置）、ACLBLAS_OP_T（转置）、ACLBLAS_OP_C（共轭转置），Host 内存 |
| transb | 输入 | aclblasOperation_t | 矩阵 B 的操作类型（同 transa），Host 内存 |
| m | 输入 | int | op(A) 和 C 的行数，M >= 0，Host 内存 |
| n | 输入 | int | op(B) 和 C 的列数，N >= 0，Host 内存 |
| k | 输入 | int | op(A) 的列数和 op(B) 的行数，K >= 0，Host 内存 |
| alpha | 输入 | const aclblasComplex* | 复数标量 alpha 指针，实部/虚部各 FLOAT32，不可为 nullptr，Host 内存 |
| A | 输入 | const void* | 矩阵 A 的设备内存指针，列主序；元素位宽由 typeA 决定；m > 0 且 n > 0 时不可为 nullptr，Device 内存 |
| typeA | 输入 | aclblasType_t | 矩阵 A 的数据类型：ACLBLAS_C_32 / ACLBLAS_R_32 / ACLBLAS_H_R_32 / ACLBLAS_H_C_32，Host 内存 |
| lda | 输入 | int | 矩阵 A 的主维度（列主序），transa=N 时 lda >= max(1, m)，transa=T/C 时 lda >= max(1, k)，Host 内存 |
| B | 输入 | const void* | 矩阵 B 的设备内存指针，列主序；元素位宽由 typeB 决定；m > 0 且 n > 0 时不可为 nullptr，Device 内存 |
| typeB | 输入 | aclblasType_t | 矩阵 B 的数据类型（同 typeA 枚举），Host 内存 |
| ldb | 输入 | int | 矩阵 B 的主维度（列主序），transb=N 时 ldb >= max(1, k)，transb=T/C 时 ldb >= max(1, n)，Host 内存 |
| beta | 输入 | const aclblasComplex* | 复数标量 beta 指针，实部/虚部各 FLOAT32，不可为 nullptr；beta = (0,0) 时 C 不必是有效输入，Host 内存 |
| C | 输入/输出 | void* | 矩阵 C 的设备内存指针，列主序；原地覆写；元素位宽由 typeC 决定；beta 非零且 m > 0、n > 0 时不可为 nullptr，Device 内存 |
| typeC | 输入 | aclblasType_t | 矩阵 C 的数据类型（同 typeA 枚举），Host 内存 |
| ldc | 输入 | int | 矩阵 C 的主维度（列主序），ldc >= max(1, m)，Host 内存 |

> 完整参数表（含 dtype、数据排布格式、维度 shape、值域范围、异常行为）见 [docs/REQUIREMENTS.md](./docs/REQUIREMENTS.md) §「输入输出规格」；错误码映射表见同文件 §「参数合法性与错误码映射」。

#### 约束说明

**参数合法性**：
- `m >= 0`、`n >= 0`、`k >= 0`
- `transa` 必须为 `ACLBLAS_OP_N` / `ACLBLAS_OP_T` / `ACLBLAS_OP_C`
- `transb` 必须为 `ACLBLAS_OP_N` / `ACLBLAS_OP_T` / `ACLBLAS_OP_C`
- `transa = N` 时 `lda >= max(1, m)`；`transa = T/C` 时 `lda >= max(1, k)`
- `transb = N` 时 `ldb >= max(1, k)`；`transb = T/C` 时 `ldb >= max(1, n)`
- `ldc >= max(1, m)`
- `alpha` 不可为 `nullptr`
- `beta` 不可为 `nullptr`
- `m > 0` 且 `n > 0` 时 `A` 不可为 `nullptr`
- `m > 0` 且 `n > 0` 时 `B` 不可为 `nullptr`
- `beta` 非零且 `m > 0`、`n > 0` 时 `C` 不可为 `nullptr`
- `typeA` / `typeB` / `typeC` 必须取值于 `{ACLBLAS_C_32, ACLBLAS_R_32, ACLBLAS_H_R_32, ACLBLAS_H_C_32}`

**错误码触发**：`handle == nullptr` → `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`；枚举越界 → `ACLBLAS_STATUS_INVALID_ENUM`；其他非法值 → `ACLBLAS_STATUS_INVALID_VALUE`。校验顺序与上游 CPU golden 参考实现（`test/gemm/gemm_golden.h`）保持一致。

**边界情况处理**：
- `m == 0` 或 `n == 0`：合法 no-op，直接返回 `ACLBLAS_STATUS_SUCCESS`，不执行任何计算，不校验 A/B/C 指针有效性（除 handle 与负维度校验）
- `k == 0`：跳过矩阵乘，执行 `C = beta * C`（`beta == (0,0)` 时置零，`beta == (1,0)` 时不变，其他值逐元素缩放）
- `alpha == (0, 0)`（且 alpha 指针非空）：跳过矩阵乘，执行 `C = beta * C`；该路径结果**位精确（EXACT 校验）**
- `beta == (0, 0)`：C 无需在调用前初始化，实现可跳过对 C 的读回

**依赖限制**：
- arch35（Ascend 950）的 Cube kernel 使用 `BlockMmad` 低阶 API，需 `asc-devkit >= 9.1`（`ASC_DEVKIT_MAJOR >= 9` 且 `ASC_DEVKIT_MINOR > 0`）
- 硬件参数（核数、L1/L0C/UB 容量）一律通过 `PlatformAscendC(context->GetPlatformInfo())` / `GetCurNpuArch()` / `GetSocVersion()` 运行时获取，**禁止硬编码**

#### 支持数据类型

四条同型 type 路径（`typeA == typeB == typeC`，跨类型组合返回 `ACLBLAS_STATUS_INVALID_VALUE`）：

| 路径 | 枚举值 | 元素类型 | aclDataType 映射 | 元素字节数 | 实/虚部位宽 | 定位 |
|------|--------|----------|------------------|-----------|-------------|------|
| 主路径 | `ACLBLAS_C_32` | `aclblasComplex`（实部/虚部各 FLOAT32） | `ACL_COMPLEX64` | 8 | FLOAT32 + FLOAT32 | **默认主路径**（`typeA = typeB = typeC = ACLBLAS_C_32`），§3.2 精度阈值与 §3.3 性能标杆 case 1-3 均以主路径为口径 |
| 扩展路径 | `ACLBLAS_R_32` | `float` | `ACL_FLOAT` | 4 | 仅实数 | FLOAT32 扩展路径（性能标杆 case 4） |
| 扩展路径 | `ACLBLAS_H_R_32` | 半精度实数 | `ACL_FLOAT16` | 2 | 仅实数 | FLOAT16 扩展路径（上游 `aclblasGemmEx` 已有 FLOAT16 矩阵先例，本算子为新增能力） |
| 扩展路径 | `ACLBLAS_H_C_32` | 半精度复数（实部/虚部各 FLOAT16） | `ACL_COMPLEX32` | 4 | FLOAT16 + FLOAT16 | COMPLEX32 扩展路径（上游仓**无** `aclblasHalf` 类型、**无** `ACL_COMPLEX32` 类型定义，属新增能力） |

`aclblasComplex` 定义（上游 `include/cann_ops_blas_common.h`，interleaved，实部在前）：

```cpp
typedef struct aclblasComplex {
    float real;
    float imag;
} aclblasComplex;
```

`aclblasType_t` 定义（本算子新增，位于 `include/cann_ops_blas_common.h`）：

```cpp
typedef aclDataType aclblasType_t;
```

该 typedef 使 `ACLBLAS_C_32` / `ACLBLAS_R_32` / `ACLBLAS_H_R_32` / `ACLBLAS_H_C_32` 等常量可直接作为 type 参数传入，**语义等价于兄弟 Ex 接口使用的 `aclDataType`**，不引入平行 API，符合任务书「接口须可与其他产品线共用，禁止定义 950PR 私有平行 API」要求。

#### 精度验证

采用 `MIXED_TOLERANCE` 混合容差策略（对齐[生态算子开源精度标准](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md)），golden 由 cblas（Netlib BLAS `cgemm`）单标杆比对生成，输出矩阵 C（`m × n`）全矩阵验证，**实部、虚部分别比对**（各分量独立统计误差，不取模后合并）。

| 数据类型 | rtol | atol | required_matched_ratio | max_abs_error_limit |
|----------|------|------|------------------------|---------------------|
| COMPLEX64（`typeT = ACLBLAS_C_32`，实部/虚部按 FLOAT32 分量） | 2^-10 (9.77e-4) | 2^-16 (1.53e-5) | 0.99 | 1e-2 或 32 * ULP |
| FLOAT32（`typeT = ACLBLAS_R_32`） | 2^-10 (9.77e-4) | 2^-16 (1.53e-5) | 0.99 | 1e-2 或 32 * ULP |
| FLOAT16（`typeT = ACLBLAS_H_R_32`） | 2^-9 (1.95e-3) | 2^-9 (1.95e-3) | 0.99 | 1e-1 或 32 * ULP |

> `H_C_32` 无独立阈值行，按 FLOAT16 行（`typeT = ACLBLAS_H_R_32`）执行并标注咨询性。

其中 ULP（Unit in the Last Place，末位单位）表示给定浮点数与相邻可表示值之间的间距：`ULP(x) = 2^(floor(log2|x|) - mantissaBits)`（有限非零），`ULP(0) = 2^(emin - mantissaBits)`（次正规 ULP）。`max_abs_error_limit` 按输出元素逐个计算：第 `i` 个元素的误差上限为 `max(fixed_value, 32 * ULP(abs(golden[i])))`。

逐元素通过条件：`|actual - golden| <= atol + rtol * |golden|`。

用例通过条件（**双条件同时满足**）：
- 至少 `matched_ratio >= required_matched_ratio`（0.99）的实部 / 虚部分量满足逐元素条件；
- 所有实部 / 虚部分量满足 `|actual[i] - golden[i]| <= max_abs_error_limit[i]`。

`k == 0` 或 `alpha == (0, 0)` 快路径（`C = beta * C`）使用 `EXACT` 位精确校验（逐元素 `max_abs_error == 0`），不套 MIXED_TOLERANCE。

> 本算子为浮点矩阵乘累加运算，**非 bit-exact**；本算子不含随机数生成，任务书 §3.5 的正态/均匀分布是测试输入数据的生成规则，不是算子行为。

#### 性能标准

| case | m | n | k | transA | transB | typeA | typeB | typeC | 标杆耗时（Avg time，us） |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 1024 | 1024 | 1024 | N | N | C_32 | C_32 | C_32 | 415.95 |
| 2 | 2048 | 2048 | 2048 | N | N | C_32 | C_32 | C_32 | 3243.65 |
| 3 | 1024 | 1024 | 1024 | T | N | C_32 | C_32 | C_32 | 411.27 |
| 4 | 2048 | 2048 | 2048 | N | N | R_32 | R_32 | R_32 | 851.86 |

采集口径：
- 测试设备：Ascend 950PR（CANN 9.1.0）
- 采样方式：**须先 warmup 再有效采样 > 50 次取平均**（`Avg time`，单位 us）
- 判定：算子在各性能 case 下的平均单次耗时**不高于**标杆耗时
- `H_R_32` / `H_C_32` 无性能标杆，仅做功能正确性验收

#### 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](https://gitcode.com/cann/ops-blas/blob/master/docs/zh/develop/compile_and_run_example.md)。

```cpp
#include <cstdio>
#include <memory>
#include <vector>

#include "acl/acl.h"
#include "cann_ops_blas.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

class AclContext {
public:
    explicit AclContext(int32_t deviceId) : deviceId_(deviceId) {}

    ~AclContext()
    {
        if (stream_ != nullptr) {
            aclrtDestroyStream(stream_);
            stream_ = nullptr;
        }
        if (deviceSet_) {
            aclrtResetDevice(deviceId_);
            deviceSet_ = false;
        }
        if (aclInited_) {
            aclFinalize();
            aclInited_ = false;
        }
    }

    int Init()
    {
        auto ret = aclInit(nullptr);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
        aclInited_ = true;

        ret = aclrtSetDevice(deviceId_);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
        deviceSet_ = true;

        ret = aclrtCreateStream(&stream_);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
        return ACL_SUCCESS;
    }

    aclrtStream Stream() const { return stream_; }

private:
    int32_t deviceId_;
    aclrtStream stream_ = nullptr;
    bool aclInited_ = false;
    bool deviceSet_ = false;
};

struct AclMemDeleter {
    void operator()(void* p) const { aclrtFree(p); }
};

int aclblasCgemmExTest(AclContext& ctx)
{
    aclrtStream stream = ctx.Stream();

    // 1. 创建 ops-blas 句柄并绑定 stream
    aclblasHandle_t rawHandle = nullptr;
    auto blasRet = aclblasCreate(&rawHandle);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasCreate failed. ERROR: %d\n", blasRet);
              return blasRet);
    std::unique_ptr<void, aclblasStatus_t (*)(void*)> handlePtr(rawHandle, aclblasDestroy);

    blasRet = aclblasSetStream(static_cast<aclblasHandle_t>(handlePtr.get()), stream);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasSetStream failed. ERROR: %d\n", blasRet);
              return blasRet);

    // 2. 准备 Host 数据（主路径 ACLBLAS_C_32，NN，C = alpha * A * B + beta * C）
    // M=2, N=2, K=2, transA=N, transB=N, alpha=(1+0i), beta=(0+0i)
    int m = 2, n = 2, k = 2;
    int lda = m, ldb = k, ldc = m;
    aclblasType_t typeA = ACL_COMPLEX64;
    aclblasType_t typeB = ACL_COMPLEX64;
    aclblasType_t typeC = ACL_COMPLEX64;
    aclblasComplex alpha = {1.0f, 0.0f};
    aclblasComplex beta = {0.0f, 0.0f};
    aclblasOperation_t transA = ACLBLAS_OP_N;
    aclblasOperation_t transB = ACLBLAS_OP_N;

    // A (M=2, K=2, column-major): [[1+1i, 3+0i], [2+0i, 4+1i]]
    std::vector<aclblasComplex> hA = {{{1.0f, 1.0f}, {2.0f, 0.0f}, {3.0f, 0.0f}, {4.0f, 1.0f}}};
    // B (K=2, N=2, column-major): [[1+0i, 0+1i], [1+0i, 1+0i]]
    std::vector<aclblasComplex> hB = {{{1.0f, 0.0f}, {1.0f, 0.0f}, {0.0f, 1.0f}, {1.0f, 0.0f}}};
    std::vector<aclblasComplex> hC(static_cast<size_t>(ldc) * n, {0.0f, 0.0f});

    size_t aBytes = hA.size() * sizeof(aclblasComplex);
    size_t bBytes = hB.size() * sizeof(aclblasComplex);
    size_t cBytes = hC.size() * sizeof(aclblasComplex);

    // 3. 申请 Device 内存并拷贝数据
    aclblasComplex* rawA = nullptr;
    auto aclRet = aclrtMalloc(reinterpret_cast<void**>(&rawA), aBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc for A failed. ERROR: %d\n", aclRet); return aclRet);
    std::unique_ptr<aclblasComplex, AclMemDeleter> aDevicePtr(rawA);

    aclblasComplex* rawB = nullptr;
    aclRet = aclrtMalloc(reinterpret_cast<void**>(&rawB), bBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc for B failed. ERROR: %d\n", aclRet); return aclRet);
    std::unique_ptr<aclblasComplex, AclMemDeleter> bDevicePtr(rawB);

    aclblasComplex* rawC = nullptr;
    aclRet = aclrtMalloc(reinterpret_cast<void**>(&rawC), cBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc for C failed. ERROR: %d\n", aclRet); return aclRet);
    std::unique_ptr<aclblasComplex, AclMemDeleter> cDevicePtr(rawC);

    aclRet = aclrtMemcpy(aDevicePtr.get(), aBytes, hA.data(), aBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy for A failed. ERROR: %d\n", aclRet); return aclRet);

    aclRet = aclrtMemcpy(bDevicePtr.get(), bBytes, hB.data(), bBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy for B failed. ERROR: %d\n", aclRet); return aclRet);

    aclRet = aclrtMemcpy(cDevicePtr.get(), cBytes, hC.data(), cBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy for C failed. ERROR: %d\n", aclRet); return aclRet);

    // 4. 调用 aclblasCgemmEx（typeA/B/C 均传 ACL_COMPLEX64，A/B/C 显式 static_cast 到 void*）
    blasRet = aclblasCgemmEx(
        static_cast<aclblasHandle_t>(handlePtr.get()),
        transA, transB, m, n, k, &alpha,
        static_cast<const void*>(aDevicePtr.get()), typeA, lda,
        static_cast<const void*>(bDevicePtr.get()), typeB, ldb,
        &beta,
        static_cast<void*>(cDevicePtr.get()), typeC, ldc);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasCgemmEx failed. ERROR: %d\n", blasRet);
              return blasRet);

    // 5. 同步等待任务执行结束
    aclRet = aclrtSynchronizeStream(stream);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", aclRet); return aclRet);

    // 6. 将结果从 Device 拷贝回 Host 并打印
    aclRet = aclrtMemcpy(hC.data(), cBytes, cDevicePtr.get(), cBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", aclRet);
              return aclRet);

    // 打印结果（列主序存储：hC[col * ldc + row] = C[row][col]）
    LOG_PRINT("result C (column-major):\n");
    for (int col = 0; col < n; col++) {
        for (int row = 0; row < m; row++) {
            const auto& val = hC[static_cast<size_t>(col) * ldc + row];
            LOG_PRINT("  C[%d][%d] = %f + %fi\n", row, col, val.real, val.imag);
        }
    }

    return ACL_SUCCESS;
}

int main()
{
    AclContext ctx(0);
    auto ret = ctx.Init();
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = aclblasCgemmExTest(ctx);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclblasCgemmExTest failed. ERROR: %d\n", ret); return ret);
    return 0;
}
```

预期输出：

```
result C (column-major):
  C[0][0] = 2.000000 + 1.000000i
  C[1][0] = 3.000000 + 2.000000i
  C[0][1] = 4.000000 + 1.000000i
  C[1][1] = 6.000000 + 3.000000i
```

## 已知问题与本批次范围

**本批次交付范围为最小可用**（迭代一穿刺范围，见 [docs/PLAN.md](./docs/PLAN.md) `iteration_count: 2`）：

- 仅 `typeA = typeB = typeC = ACLBLAS_C_32`（主路径 COMPLEX64）与 `transa = transb = ACLBLAS_OP_N`（NN 组合）已实测通过。
- `ACLBLAS_OP_T` / `ACLBLAS_OP_C` 转置与共轭转置组合将在**迭代二**补齐。
- `ACLBLAS_R_32` / `ACLBLAS_H_R_32` / `ACLBLAS_H_C_32` 三条扩展路径将在**迭代二**补齐；`H_C_32`（`ACL_COMPLEX32`）在上游仓无先例（无 `aclblasHalf` 类型、无 `ACL_COMPLEX32` 类型定义），需新增 device 侧表示类型与读写方式。
- 未达标项与性能基准的完整实测数据将在迭代二完成后回填至自测报告（`operators/cgemm_ex/tests/reports/`）。

**明确不支持**：

- **非连续 Tensor**：不支持超出 `lda` / `ldb` / `ldc` 语义的非连续内存访问（超出 ld 的任意步长 / 子块索引不支持）。
- **broadcast**：不涉及——A / B / C 为独立矩阵，无广播规则。
- **dynamic shape**：不要求——`m` / `n` / `k` 为运行时入参，无需动态 shape 支持。
- **批量（batched）语义**：本接口为单批语义，批量接口为独立的 `aclblasGemmBatchedEx` / `aclblasGemmStridedBatchedEx`。
- **算法选择 / 计算精度参数**：签名中不含 `computeType` 与 `algo` 参数。
- **确定性计算要求**：不要求；常规路径非 bit-exact，例外仅 `k == 0` 与 `alpha == (0,0)` 两条 EXACT 位精确快路径。
- **原地与视图**：C 原地覆写，不返回视图。
- **跨类型组合**：`typeA == typeB == typeC` 强制同型，跨类型（如 `C_32 × R_32`）返回 `ACLBLAS_STATUS_INVALID_VALUE`（对齐上游 `aclblasGemmEx` 的 `Atype = Btype = Ctype` 口径）。

**内存要求**：不涉及（任务书 §3.4）。但自测报告需包含**内存占用数据**（workspace 占用 + HBM 用量），供社区验收使用。

## 代码仓库与接口位置

- **接口声明**：`include/cann_ops_blas.h`（供其他产品线共用，禁止 950PR 私有平行接口）
- **类型定义**：`aclblasType_t` typedef 与半精度复数类型（如需）位于 `include/cann_ops_blas_common.h`
- **实现代码**：`blas/gemm/arch35/`（新增文件一律 `cgemm_ex_` 前缀：`cgemm_ex_host.cpp` / `cgemm_ex_kernel.cpp` / `cgemm_ex_tiling_data.h` / `cgemm_ex_kernel.h`，避免与既有 `gemm_*` 冲突；arch35 目录由构建系统自动 glob，无需新增 CMakeLists）
- **测试代码**：`test/gemm/cgemm_ex/arch35`（含 CSV 用例文件；`test/gemm/cgemm_ex/` 下放置 CMakeLists，使用 `ops_blas_add_gtest_tests(${OPS_BLAS} cgemm_ex_test)`）
- **合入目录**：实现合入 `blas/gemm/arch35`，测试合入 `test/gemm/cgemm_ex/arch35`，文件结构参考主仓 blas 算子测试代码结构（含 csv 文件）

## 相关文档

- [需求分析](./docs/REQUIREMENTS.md)：接口签名、四条 type 路径、参数规格、错误码映射、边界与特殊行为、精度与性能口径的权威需求来源
- [详细设计](./docs/DESIGN.md)：架构分层（op_api / op_host / op_kernel）、Cube / AIV 双 kernel 分工、TilingData 与分发机制、快路径（`k == 0` / `alpha == (0,0)`）设计、3-GEMM 差分重组
- [测试计划](./docs/TEST.md)：三级 CSV（L0 / L1 / L2）用例设计与验收判据（MIXED_TOLERANCE / EXACT）、golden 来源、性能与内存数据采集
- [迭代计划](./docs/PLAN.md)：迭代一（TK0 主路径 + TK4 快路径 + TK3 半精度可行性穿刺）与迭代二（TK1 / TK2 / TK3 补齐、全 dtype 覆盖）执行计划
- [ACLNN API 接口文档](./docs/aclnnCgemmEx.md)：仓内文档门控用接口文档
- 任务书：`/workspace/aclblasCgemmEx/aclblasCgemmEx_task_doc.md`（权威需求来源）
- 配套测试用例：`/workspace/aclblasCgemmEx/test_cases/`
- 参考实现：
  - cuBLAS 参考文档（cublasCgemmEx）：<https://docs.nvidia.com/cuda/cublas/index.html#cublas-c-gemmex>
  - Netlib BLAS 参考实现（cgemm，单批语义参考）：<https://www.netlib.org/blas/cgemm.f>
  - 生态算子开源精度标准：<https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md>
  - 算子开发接口文档：<https://www.hiascend.com/document/detail/zh/canncommercial/850/API/ascendcopapi/atlasascendc_api_07_0003.html>
  - ops-blas 开源仓：<https://gitcode.com/cann/ops-blas>

---

## 修订记录

| 版本 | 日期 | 修改人 | 修改内容 |
|------|------|--------|----------|
| v1.0 | 2026-09-15 | 文档编写（developer-doc） | 首版：按 ops-blas 仓 `blas/gemm/README.md` 结构组织，覆盖算子概述 / 数学公式 / 接口签名 / 参数摘要 / 约束说明 / 数据类型支持 / 精度标准 / 性能标准 / 调用示例 / 已知问题（本批次范围 = C_32 + NN）/ 相关文档 / 代码仓库位置；产品支持表标注 Ascend 950PR：支持，其他产品线一律标「不支持（未适配）」 |
