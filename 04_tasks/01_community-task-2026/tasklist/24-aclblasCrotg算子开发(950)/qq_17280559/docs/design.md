

# AclblasCrotg_Atlas950PR任务需求背景（required）

## 需求来源

昇腾社区任务（cann-ops-competitions 社区任务 2026 · 算子实操工坊-上海站）：在昇腾 NPU（Ascend 950PR）上使用 Ascend C/CATLASS 编程语言开发单精度复数（complex64）Givens 旋转参数构造算子 `aclblasCrotg`，完成算子设计、开发、测试全流程工作。验收通过后合入昇腾算子开源仓 ops-blas。

| 来源 | 链接 |
| --- | --- |
| 任务书 | `aclblasCrotg_Atlas950PR_task_doc.md` |
| 开源仓 | https://gitcode.com/cann/ops-blas |
| 设计文档模板 | https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md |
| 对标接口 | cuBLAS `cublasCrotg`（https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-rotg ） |
| 参考实现 | Netlib BLAS `crotg`（https://www.netlib.org/blas/crotg.f ） |

## 背景介绍

### rotg 算子定义

rotg（Givens rotation generator）类算子由两个标量 (a, b) 构造 Givens 平面旋转参数 (c, s, r)，使得以 (c, s) 组成的 2×2 旋转矩阵作用于向量 (a, b)ᵀ 后，第二个分量精确消为 0：

```
[  c         s ] [ a ] = [ r ]
[ -conjg(s)  c ] [ b ]   [ 0 ]
```

其中 c 为实数（float），s 为复数（complex64），满足正交归一性 c² + |s|² = 1。Givens 旋转是数值线性代数中消零向量分量的基本工具，广泛用于 QR 分解、Lanczos/GMRES 等迭代法的三对角化与消元过程。

### aclblasCrotg 算子实现现状分析

ops-blas 仓 `blas/rotg/` 目录下已有同族**实数**算子 `aclblasSrotg` 的 arch35 实现（`srotg_host.cpp`、`srotg_kernel.cpp`），验证了"句柄式 BLAS 接口 + Host/Device 双路径 + SIMT kernel 直调"的工程模式可行性。当前仓内**缺少复数版本**：

| 现状 | 说明 |
| --- | --- |
| 接口缺失 | `include/cann_ops_blas.h` 仅有 `aclblasSrotg` 声明，无 `aclblasCrotg` |
| 类型差异 | 复数 rotg 中 c 为实数、s 为复数，与实数 rotg 的 c/s 均为实数不同 |
| 稳定性算法差异 | 复数 rotg 的数值稳定参考实现采用 Algorithm 978 安全缩放（Anderson, 2017），与实数 srotg 的经典缩放方案分支结构不同 |
| golden 差异 | 标准 CBLAS 接口集不含复数 rotg（仅 srotg/drotg），golden 须取 Netlib `crotg` 参考实现 |

### aclblasCrotg 算子功能分析

- **功能**：由输入复数标量 a、b 构造 Givens 旋转参数——实数余弦 c、复数正弦 s，并将 a 原地覆写为 r；b 只读不覆写。
- **输入**：a（复数标量，输入/输出）、b（复数标量，只读输入）、handle（上下文句柄）。
- **输出**：c（实数标量）、s（复数标量）、a 原地覆写为 r（复数标量）。
- **支持数据类型**：a/b/s 为 COMPLEX64（实部/虚部各 FLOAT32），c 为 FLOAT32。
- **支持广播**：不涉及（纯标量算子，无维度轴）。

# 需求分析（required）

## 需求描述

使用 Ascend C 编程语言，在 Ascend 950PR 上以 kernel 直调方式实现 `aclblasCrotg` 算子：输入复数标量 a、b（COMPLEX64），构造 Givens 平面旋转参数——实数余弦 c（FLOAT32）、复数正弦 s（COMPLEX64），并将 a 原地覆写为 r；b 只读不覆写。支持 Host / Device 双路径执行；采用 safmin/safmax 安全缩放算法保证极端数量级输入不溢出、不下溢；精度满足生态算子开源精度标准（FLOAT32 档）；单次调用延迟与批量连续调用吞吐不高于任务书标杆；配套 CSV 驱动 GTest 测试工程（golden 为 Netlib crotg 参考实现的 C 移植）。

## 需求拆解

| # | 需求项 | 拆解内容 | 来源 |
| --- | --- | --- | --- |
| 1 | 接口 | 在 `include/cann_ops_blas.h` 新增 `aclblasCrotg` 声明，签名与 `cublasCrotg` 逐参数一致（handle、a、b、c、s），b 为只读但保持非 const 指针；禁止定义 950PR 私有平行接口 | 任务书 §2.3 |
| 2 | 核心功能 | 由 (a, b) 计算 c、s、r；a 覆写为 r；b 只读；满足 c² + \|s\|² = 1 | 任务书 §2.1 |
| 3 | 特殊分支 | b=0、a=0（b≠0）、a=b=0 三类分支语义对齐 Netlib crotg 参考实现 | 任务书 §2.1.2 |
| 4 | 数值稳定性 | 采用 safmin/safmax 安全缩放算法（Algorithm 978），禁止直接计算 \|a\|² + \|b\|² 而不做缩放 | 任务书 §2.1.3 |
| 5 | 工程模式 | 基于 ops-blas 开源仓工程框架，代码放在 `blas/rotg/arch35/`，Host 侧参数校验与 Host/Device 双路径判别模式参照同族 `aclblasSrotg` | 任务书 §2.2 |
| 6 | 参数校验 | handle 为 nullptr 返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`；a/b/c/s 任一为 nullptr 返回 `ACLBLAS_STATUS_INVALID_VALUE`；混合 Host/Device 指针返回 `ACLBLAS_STATUS_INVALID_VALUE` | 任务书 §2.5 |
| 7 | 执行语义 | 依赖 `aclblasSetStream` 绑定 stream 异步执行；读回 Device 结果前须同步 stream | 任务书 §2.5 |
| 8 | 测试 | 参照 srotg 模式新建 `test/rotg/crotg/arch35/` CSV 驱动 GTest 测试工程；golden 为 Netlib crotg；覆盖精度、数学性质、性能、负向用例 | 任务书 §3.5 |
| 9 | 性能 | 5 个标杆 case（单次延迟 ×3 + 批量 1000/10000），warmup 后有效采样 >50 次取平均 | 任务书 §3.3 |
| 10 | 交付 | 算子设计文档 PR、测试代码 + readme、自测报告、个人仓代码 + `blas/rotg/README.md` 产品支持表更新 | 任务书 §4 |

# 详细设计（required）

## 算子分析

### 数学公式

构造 Givens 平面旋转，使 2×1 向量 (a, b)ᵀ 旋转后第二个分量消为 0：

```
[  c         s ] [ a ] = [ r ]
[ -conjg(s)  c ] [ b ]   [ 0 ]
```

其中 c 为实数（float），s 为复数（complex64），满足正交归一性 c² + |s|² = 1。数学定义为：

```
|x|      = sqrt(Re(x)² + Im(x)²)        （复数模）
conjg(x) = Re(x) − Im(x)·i              （共轭）
sgn(x)   = x/|x|（x ≠ 0）；sgn(0) = 1

c = |a| / sqrt(|a|² + |b|²)
s = sgn(a) · conjg(b) / sqrt(|a|² + |b|²)
r = sgn(a) · sqrt(|a|² + |b|²)
```

输出语义：**a 被原地覆写为 r**；c、s 为纯输出标量；**b 为只读输入，不覆写**（依据：cuBLAS 文档复数段仅说明 a 覆写为 r；Netlib 参考实现 crotg 中 B 为 [in] 参数；实数 rotg 的 z 恢复规则不适用于复数版本）。

**特殊值分支**（与 Netlib crotg 参考实现一致）：

| 分支条件 | c | s | r（覆写 a） |
| --- | --- | --- | --- |
| b = 0 | 1 | (0, 0) | a |
| a = 0 且 b ≠ 0 | 0 | conjg(b)/\|b\| | \|b\|（实数结果，虚部为 0） |
| a = b = 0 | 1 | (0, 0) | (0, 0) |

Inf/NaN 输入为规格允许场景，期望接口返回 SUCCESS 且不崩溃，数值按 IEEE 754 传播规则记录行为（不参与精度比对判据）。

### 数值稳定性算法（Algorithm 978）

直接计算 |a|² + |b|² 在大数量级输入下必然溢出（如 |a|≈1e30，平方达 1e60 > FLT_MAX≈3.4e38）；小数量级输入下平方下溢为次正规数或 0，损失精度。本实现参照 Netlib crotg（Anderson, *Algorithm 978: Safe Scaling in the Level 1 BLAS*, ACM TOMS 2017）采用安全缩放：

**缩放常量**（IEEE 754 binary32，radix=2）：

```
safmin = 2^-126  ≈ 1.1754944e-38     （= FLT_MIN）
safmax = 2^127   ≈ 1.7014118e+38
rtmin  = sqrt(safmin) = 2^-63 ≈ 1.0842022e-19
rtmax（主路径）= sqrt(safmax/4) ≈ 6.5231212e+18
```

**主路径（a≠0 且 b≠0）流程**：

```
f1 = max(|Re(a)|, |Im(a)|)；g1 = max(|Re(b)|, |Im(b)|)

1) 未缩放路径：rtmin < f1 < rtmax 且 rtmin < g1 < rtmax
   f2 = |a|²; g2 = |b|²; h2 = f2 + g2          （保证不溢出：均 < safmax）
   若 f2 ≥ h2·safmin（比值可精确表示）：
       c  = sqrt(f2/h2)
       r  = a / c
       s  = conjg(b)·(a / sqrt(f2·h2)) 或 conjg(b)·(r/h2)
   否则（f2/h2 ≤ safmin，f 远小于 g）：
       d  = sqrt(f2·h2)                        （保证在 [sqrt(safmin), sqrt(safmax)]）
       c  = f2/d
       r  = (c ≥ safmin) ? a/c : a·(h2/d)
       s  = conjg(b)·(a/d)
2) 缩放路径（任一分量越界 rtmin/rtmax）：
   u  = min(safmax, max(safmin, f1, g1))       （公共缩放因子）
   gs = b/u；若 f1/u < rtmin（f 与 g 数量级悬殊，f 需独立缩放）：
       v = min(safmax, max(safmin, f1))；w = v/u；fs = a/v；h2 = |fs|²·w² + |gs|²
   否则：w = 1；fs = a/u；h2 = |fs|² + |gs|²
   按 1) 同样公式基于 (fs, gs, h2) 求 c、r、s，最后 c ← c·w，r ← r·u 恢复量纲
```

**a=0（b≠0）分支**：c=0，r=|b|（实数结果），s = conjg(b)/|b|。其中 b 纯实（Im=0）或纯虚（Re=0）时走快路径——r 直接取对应分量绝对值（免开方），s = conjg(b)/r；b 为一般复数时以 g1 = max(|Re(b)|, |Im(b)|) 判别未缩放（rtmin < g1 < sqrt(safmax/2)）/缩放路径后计算 s = conjg(b)/|b|、r = |b|（缩放路径 r = |gs|·u 恢复量纲）。

**b=0 分支**：直接返回 c=1、s=0、r=a，无任何浮点运算，天然无溢出风险。

#### 缩放算法的正确性论证（为何禁止直接平方）

任务书 §2.1.3 明确要求"禁止直接计算 |a|² + |b|² 而不做缩放"。其数值依据如下，本设计逐分支给出溢出/下溢边界证明（变量名与实现 `crotg_core.h` 一一对应）：

| 输入形态 | 直接算法（禁止） | 本实现（安全缩放） |
| --- | --- | --- |
| \|a\|, \|b\| ≈ 1e30 | 平方达 1e60 > FLT_MAX(≈3.4e38)，`f2+g2` **溢出为 +Inf**，c=Inf/Inf=NaN | f1=g1=1e30 越界 `rtmax`，走缩放路径：u=1e30，fs=gs 归一后 f2=g2=1，h2=2，全程在 [safmin, safmax] 内，最后 c←c·w、r←r·u 恢复量纲，**无溢出** |
| \|a\|, \|b\| ≈ 1e-38（含次正规 1.4e-45） | 平方达 1e-76，**下溢为 0**，c=0/0=NaN | f1 越界 `rtmin` 下限，走缩放路径放大到 [rtmin, rtmax] 后平方，**不下溢** |
| \|a\|≫\|b\| 或反向（如 1e±40 倍差） | 小分量平方下溢为 0，旋转角被截断 | f1/u < rtmin 触发 **f 独立缩放**（w=v/u），避免 f 在 g 的尺度下丢失 |

**边界证明（三分支的区间不变式）**：

1. **未缩放路径**（`rtmin < f1, g1 < rtmax = sqrt(safmax/4)`）：由 `f1 < sqrt(safmax/4)` 得 `f2 = |a|² < safmax/4`，同理 `g2 < safmax/4`，故 `h2 = f2 + g2 < safmax/2 < safmax`，**加法不溢出**；又 `f1 > rtmin` 保证 `f2 > safmin`，**平方不下溢**。全程满足 `safmin ≤ f2 ≤ h2 ≤ safmax` 这一不变式。
2. **缩放路径（公共缩放 u）**：`u = clamp(fmax(f1, g1), safmin, safmax)` 将较大的分量归一至 ≤1，`gs = g/u` 后 `g2 ≤ 2`（因 |g|² = g1²·(实/虚占比) ≤ 2·g1²，除以 u² 后 ≤2），`f2` 同理，`h2` 有界，**无溢出**；若 `f1/u < rtmin` 说明 f 相对 g 过小，改用独立缩放因子 `v = clamp(f1, safmin, safmax)`、`w = v/u`，`fs = f/v` 使 `f2 = |fs|² ≥ safmin`，**f 不下溢**。
3. **量纲恢复**：缩放路径计算完成后 `c ← c·w`、`r ← r·u`。由于 `w ≤ 1`（v ≤ u），`c·w` 不溢出；`r·u` 中 r 为缩放后模长（≤ sqrt(2)·u 量纲），乘回 u 恢复原始量纲，结果值域落于原始输入的模长范围，**不引入新溢出**。

结论：全值域（除 Inf/NaN 规格允许场景）下结果保持 FLOAT32 精度档要求，误差仅来源于少量复数乘除与开方的舍入，远优于验收阈值（atol=2⁻¹⁶、rtol=2⁻¹⁰）。

### 支持数据类型

| 参数 | 输入/输出 | 数据类型 | 说明 |
| --- | --- | --- | --- |
| handle | 输入 | aclblasHandle_t | ops-blas 库上下文句柄，携带 stream |
| a | 输入/输出 | COMPLEX64（实部/虚部各 FLOAT32） | 输入 a，原地覆写为 r |
| b | 输入 | COMPLEX64 | 只读，不覆写 |
| c | 输出 | FLOAT32 | 实数，值域自然落于 [0,1] |
| s | 输出 | COMPLEX64 | 复数，满足 c²+\|s\|²=1 |

复数类型 `aclblasComplex` 以 ops-blas 仓 `include/cann_ops_blas_common.h` 中定义为准（实部/虚部各 float32）。

### 支持形状

纯标量算子：4 个参数均为单元素指针（shape [1]），**无向量长度、无步长、无维度轴**，不涉及 broadcast、dynamic shape、非连续 Tensor、空 Tensor 等 Tensor 类概念。

## 算子实现

### 实现方案

#### 3.2.1 host侧设计：

**总体架构（Host/Device 双路径）**：

沿用仓内 srotg 验证过的"Host/Device 双路径"模式，核心数值算法收敛为一份与运行环境无关的纯函数 `CrotgCore`（仅依赖标量浮点运算），Host 路径直接调用，Device 路径在 kernel 内调用同一份代码，保证两条路径计算结果逐位一致、行为完全对齐。

> **双路径数值一致性设计（本实现的关键优化）**：对比仓内 srotg 的"Host/Device 双份镜像实现"（`srotg_host.cpp` 内一份、`srotg_kernel.cpp` 内一份，两份代码需人工同步维护），本设计将核心算法**收敛为单一头文件 `crotg_core.h`**（`aclblas::crotg::CrotgCoreCompute`），被 Host 路径（`CrotgCpuCompute`）与 Device 路径（`crotg_simt_compute`）同时 `#include`。该设计消除双份镜像的数值漂移风险：任何算法修订仅改一处，两条路径同步生效，保证 `CrotgCore` 对同一输入在 Host 与 Device 上产生**逐位一致**的结果。核心算法仅依赖 `fabsf/fmaxf/fminf/sqrtf` 等标量浮点运算，Host 编译器与 Ascend C 设备编译器均可直接编译，无平台相关 API。

```
aclblasCrotg(handle, a, b, c, s)
        │
        ▼
┌───────────────────────────────┐
│ 1. 参数校验（Host 侧）         │
│    handle == nullptr           │──► return ACLBLAS_STATUS_HANDLE_IS_NULLPTR
│    a/b/c/s 任一 == nullptr     │──► return ACLBLAS_STATUS_INVALID_VALUE
└──────────────┬────────────────┘
               ▼
┌───────────────────────────────┐
│ 2. Host/Device 指针属性判别    │
│    （查询四个指针的内存属性）   │
│    全 Host | 全 Device 合法    │
│    混合                        │──► return ACLBLAS_STATUS_INVALID_VALUE
└──────┬────────────────┬───────┘
       ▼                ▼
┌──────────────┐  ┌──────────────────────────────┐
│ 3a. Host 路径 │  │ 3b. Device 路径               │
│  CPU 直接调用 │  │  构造 kernel 参数（4 个 GM     │
│  核心计算函数 │  │  指针），绑定 handle 所携       │
│  （同步完成， │  │  stream 直调 SIMT kernel       │
│  立即可读）   │  │  （异步提交，读回前须同步）     │
└──────────────┘  └──────────────────────────────┘
       │                        │
       └────────► return ACLBLAS_STATUS_SUCCESS ◄┘
```

**（1）参数校验**（顺序固定，先 handle 后数据指针）：

| 校验项 | 条件 | 返回 |
| --- | --- | --- |
| handle 有效性 | handle == nullptr | ACLBLAS_STATUS_HANDLE_IS_NULLPTR |
| 指针有效性 | a、b、c、s 任一为 nullptr | ACLBLAS_STATUS_INVALID_VALUE |
| 内存一致性 | a、b、c、s 须全 Host 或全 Device | ACLBLAS_STATUS_INVALID_VALUE（混合时） |

**（2）Host/Device 判别**：参照仓内 srotg 既有口径，通过运行时指针属性查询接口（如 `aclrtPointerGetAttributes`）逐一判别四个指针的内存位置；对齐 srotg 的实现方式复用其判别工具函数。

**（3）Host 路径**：CPU 侧直接调用共享核心函数 `CrotgCore`，单次调用同步完成，无 kernel 提交开销；对应用户在 Host 内存场景（如小规模迭代调试）的最低延迟。

**（4）Device 路径**：

- 通过 handle 所携带的 stream 获取执行流，使用运行时 kernel launch 接口下发 SIMT kernel，**不做任何 Host↔Device 数据搬运**（4 个指针直接以 GM 地址传入 kernel）；
- 无 tiling 数据、无 workspace：纯标量算子无切分需求，kernel 参数仅 4 个 GM 指针，launch 参数固定，Host 侧准备开销最小化（这是单次调用延迟达标的关键——标杆 9.5~11.9 us 量级主要由固定调用开销构成）；
- 异步语义：接口返回 SUCCESS 仅代表任务已提交到 stream，与仓内 BLAS 接口一致；用户读回 Device 结果前须 `aclrtSynchronizeStream`（由测试工程统一处理）。

**（5）性能优化要点**：

- 校验与判别逻辑尽量提前返回、避免冗余查询（仅 1 次指针属性查询接口批量判别）；
- 不构造 TilingData、不分配 workspace、不触发 H2D 拷贝；
- 批量连续调用场景（iters=1000/10000）下每次调用均为一次 launch，平均延迟由 launch 流水掩盖后趋近稳态，天然满足吞吐要求。

#### 3.2.2 kernel侧设计：

**（1）Kernel 形态**：SIMT 标量 kernel，1-block、`SIMT_MIN_THREAD_NUM` 线程、thread 0 单线程计算（数据量 2×8B 输入 + 20B 输出），无数据切分、无多核协同。与仓内 srotg 一致采用 `asc_vf_call` + `KERNEL_TYPE_AIV_ONLY`（AIV 核）：

```cpp
// 与实际实现 crotg_kernel.cpp 一致（对齐 srotg_kernel.cpp 的 kernel 直调写法）
constexpr uint32_t CROTG_SIMT_THREAD_NUM = SIMT_MIN_THREAD_NUM;

__simt_vf__ __aicore__ LAUNCH_BOUND(CROTG_SIMT_THREAD_NUM)
inline void crotg_simt_compute(__gm__ aclblasComplex* aGm, __gm__ aclblasComplex* bGm,
                               __gm__ float* cGm, __gm__ aclblasComplex* sGm)
{
    if (threadIdx.x != 0) { return; }           // 仅 thread 0 计算，其余线程空转
    aclblasComplex a = *aGm;                    // 1. GM -> 寄存器：a（覆写为 r）、b（只读）
    aclblasComplex b = *bGm;
    float c = 0.0f; aclblasComplex s = {0.0f, 0.0f};
    aclblas::crotg::CrotgCoreCompute(&a, b, &c, &s);   // 2. 共享核心算法（Algorithm 978）
    *aGm = a; *cGm = c; *sGm = s;               // 3. 寄存器 -> GM：写回 r、c、s（b 不写）
}

extern "C" __global__ __aicore__ void crotg_kernel(GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR s)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    asc_vf_call<crotg_simt_compute>(
        dim3{CROTG_SIMT_THREAD_NUM, 1, 1},
        reinterpret_cast<__gm__ aclblasComplex*>(a),
        reinterpret_cast<__gm__ aclblasComplex*>(b),
        reinterpret_cast<__gm__ float*>(c),
        reinterpret_cast<__gm__ aclblasComplex*>(s));
}

// Host 侧 launch 入口（声明于 crotg_kernel.h，由 crotg_host.cpp 的 Device 路径调用）
void crotg_kernel_do(uint8_t* a, uint8_t* b, uint8_t* c, uint8_t* s, void* stream)
{
    crotg_kernel<<<1, nullptr, static_cast<aclrtStream>(stream)>>>(a, b, c, s);
}
```

**（2）核心算法实现要点**：

- 复数运算全部展开为标量 float 运算（复数乘/除/共轭/模的显式公式），避免依赖设备侧复数库；
- 常量 safmin/safmax/rtmin/rtmax 编译期确定（2 的幂，可位运算或字面值）；
- 分支结构：b=0 → a=0 → 主路径（未缩放/缩放），分支简单规整，SIMT 下无 divergence 问题（单线程）；
- NaN/Inf 传播：遵循 IEEE 754 默认传播（比较运算在 NaN 下走 else 分支，结果自然携带 NaN），保证"期望 SUCCESS 不崩溃"的规格。

**（3）与 srotg 的差异对照**：

| 维度 | srotg（已有） | crotg（本任务） |
| --- | --- | --- |
| 输入类型 | float 标量对 | complex64 标量对 |
| 输出 | c、s 均 float，z 恢复规则 | c 为 float、s 为 complex，**无 z 恢复**（b 只读） |
| 稳定性算法 | 经典 rotg 缩放 | Algorithm 978 三路缩放（含 f/g 量纲悬殊的独立缩放） |
| Host/Device 双路径 | 有 | 复用同一模式 |
| kernel 计算量 | 标量运算 ~10 flop | 标量运算 ~30 flop（复数展开） |

**（4）数值正确性论证（举例）**：

| 输入 | 直接算法 \|a\|²+\|b\|² | 本实现（安全缩放） |
| --- | --- | --- |
| a=(1e30,1e30), b=(1e30,-1e30) | 2e60 + 2e60 → **溢出为 +Inf** | f1=g1=1e30 越界 → 缩放路径：u=1e30，fs/gs 归一后 h2=2，c=√2/2，r 恢复量纲 =1e30·√2，全程无溢出 |
| a=b=(1e-38,0) | 2e-76 → **下溢为 0**，c=0/0=NaN | f1=g1=1e-38 越界 → 缩放路径：u 缩放后平方量级 ~1e-38 安全；结果精确 |
| a=(3,4), b=(1,2) | 25+5=30，正常 | 未缩放路径直算，与 golden 逐位一致 |

结论：全值域（除 Inf/NaN 规格允许场景）下结果保持 FLOAT32 精度档要求，且远优于阈值（误差仅来源于少量乘除与开方的舍入）。

**（5）文件与目录规划**：

```
ops-blas/
├── include/
│   ├── cann_ops_blas.h                 # [修改] 新增 aclblasCrotg 声明（供各产品线共用）
│   └── cann_ops_blas_common.h          # [不动] aclblasComplex / 状态码定义
├── blas/rotg/
│   ├── README.md                       # [修改] 产品支持表新增 aclblasCrotg 行，标注 Ascend 950PR：支持
│   └── arch35/
│       ├── crotg_host.cpp              # [新增] Host 侧：校验/判别/双路径分发
│       ├── crotg_kernel.h              # [新增] kernel 直调入口声明（crotg_kernel_do）
│       ├── crotg_kernel.cpp            # [新增] Device 侧：SIMT kernel 直调
│       └── crotg_core.h                # [新增] 共享核心算法 CrotgCore（Host/Device 双编译）
└── test/rotg/crotg/arch35/             # [新增] CSV 驱动 GTest 测试工程（参照 srotg 模式）
    ├── crotg_test.csv                  # 用例（由 test_cases/gen_csv.py 生成，1200 条，种子可复现）
    ├── gpu_baseline.csv                # H100 性能基线（按 case_name 关联，200 条）
    ├── crotg_param.h                   # 列格式解析（ReadMap 键同 CSV 列）
    ├── crotg_main.cpp                  # GTest 入口（精度/数学性质/负向/性能四类用例）
    ├── crotg_golden_ref.c/.h           # golden：Netlib crotg 参考实现的 C 移植（编入测试进程）
    └── README.md                       # 测试步骤说明（保证验收人可复现）
```

### 性能优化方案

本算子为纯标量算子（无尺寸轴、无切分、无数据搬运），性能特征由**固定调用开销**与**批量连续调用吞吐**刻画，与数据量无关。优化目标是对齐任务书 §3.3 的五个标杆 case（单次延迟 9.5~11.9 us，批量 1000/10000 次平均 11.5~11.8 us）。

#### （1）单次调用延迟拆解

一次 `aclblasCrotg(handle, a, b, c, s)` 的耗时构成（量级为标杆 case 的近似拆解）：

| 阶段 | 操作 | 优化手段 | 目标 |
| --- | --- | --- | --- |
| 参数校验 | handle/指针判空 | 纯寄存器比较，无系统调用 | < 0.1 us |
| 指针属性判别 | `aclrtPointerGetAttributes` × 4 | **fail-fast**：以 a 为基准逐次比对，发现不同侧立即返回，避免冗余查询；Host 场景可判定后直接走 CPU 路径 | 单指针查询 ~1 us 量级 |
| 分支分发 | Host/Device 路径选择 | 无 tiling、无 workspace 分配、无 H2D 拷贝 | 常数开销 |
| kernel 提交 | Device 路径 launch | 1-block 单线程 SIMT，launch 参数仅 4 个 GM 指针，无 TilingData | 提交开销趋近 runtime 下限 |
| 数值计算 | Algorithm 978（~30 flop） | 单线程、无 divergence、无同步 | 相对调用开销可忽略 |

关键结论：**标杆 9.5~11.9 us 量级主要由 kernel launch 与指针属性查询的固定开销构成**，而非数值计算。因此优化的重点不在算法计算量，而在**最小化 Host 侧准备路径**（校验尽早返回、判别仅一次、零搬运、零分配）。

#### （2）Host 路径与 Device 路径的延迟差异

- **Host 路径**：全 Host 指针时直接 CPU 计算（`CrotgCpuCompute`），无 kernel 提交、无 stream 同步，单次调用同步完成，延迟最低——适用于小规模迭代调试（如 QR 迭代的逐步 Givens 参数构造）；
- **Device 路径**：全 Device 指针时 launch 1-block SIMT kernel，计算与结果均保留在 Device，无 H2D/D2H 搬运，便于与下游 `aclblasCrot`（应用旋转）在同一 Device 数据流上流水衔接，避免 Host 往返。

#### （3）批量连续调用吞吐优化

性能 case 4/5（iters=1000/10000）为**单 stream 内连续调用 iters 次**，每轮重置 a/b 输入，取平均单次耗时：

- 连续 launch 由 runtime 的异步下发流水掩盖，稳态后平均延迟趋近**单次 launch 的固定开销下限**；
- 本实现不构造 TilingData、不分配 workspace、每次调用仅一次 launch + 一次指针判别，无额外累积状态，天然满足吞吐要求；
- 批量场景下 `aclblasSetStream` 绑定同一 stream，多调用间无隐式同步点（除非用户主动 sync）。

#### （4）性能对标口径

验收判定为 `NPU 平均单次耗时 ≤ gpu_baseline.csv 的 gpu_ms（H100 基线）/ 0.4`。标杆耗时换算：`标杆(us) = H100 gpu_ms × 1000 / 0.4`（如 case1 基线 4.740 us → 4.740/0.4 = 11.85 us，与任务书一致）。H100 为 GPU 侧对标基线，0.4 为昇腾相对 GPU 的性能倍率阈值（≥ 0.4 即达标）。测试须 warmup 后有效采样 >50 次取平均（任务书 §7.4）。

> 补充：GTest 输出耗时含 host 准备 + kernel + golden + 比对，为保守上界；精确 kernel 耗时可用 msprof（真机）或 `npusim -g`（仿真）单独采集。

### 接口设计

在 `include/cann_ops_blas.h` 新增声明（当前头文件尚无该接口）：

```cpp
aclblasStatus_t aclblasCrotg(
    aclblasHandle_t handle,
    aclblasComplex* a,
    aclblasComplex* b,
    float* c,
    aclblasComplex* s);
```

- 签名与对标接口 `cublasCrotg` 逐参数一致（handle 及参数顺序一一对应，无需映射说明）；b 虽为只读，但为与 `cublasCrotg`（`cuComplex* b` 非 const）一致，签名保持非 const 指针；
- 返回值 `aclblasStatus_t` 语义与 `include/cann_ops_blas_common.h` 定义一致：`ACLBLAS_STATUS_SUCCESS` / `ACLBLAS_STATUS_INVALID_VALUE` / `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` 等；
- 供其他产品线共用，禁止定义 950PR 私有平行 API。

**参数校验顺序（fail-fast）**：校验严格按固定顺序执行——先 `handle == nullptr`（返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`），再 `a/b/c/s` 任一判空（返回 `ACLBLAS_STATUS_INVALID_VALUE`），最后判别四个指针的内存一致性（混合 Host/Device 返回 `ACLBLAS_STATUS_INVALID_VALUE`）。指针属性判别以 a 为基准，b/c/s 逐一比对，**一旦发现不同侧立即返回**（不再查询后续指针），既保证语义正确，又最小化失败路径的查询开销。

**线程安全与可重入性**：`aclblasCrotg` 本身无共享可变状态（不写全局变量、不分配 workspace），核心算法 `CrotgCoreCompute` 为纯函数，天然可重入、线程安全；并发安全由 handle 携带的 stream 语义保证（不同 stream 间并发提交、同一 stream 内按序执行），与仓内 BLAS 接口一致。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Ascend 950PR（arch35） | √ |

## 算子约束限制

| 约束项 | 内容 |
| --- | --- |
| 参数合法性 | a、b、c、s 均不可为 nullptr，否则返回 `ACLBLAS_STATUS_INVALID_VALUE`；handle 为 nullptr 返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| Host/Device 内存一致性 | a、b、c、s 须全部位于 Host 侧或全部位于 Device 侧；混合 Host/Device 指针返回 `ACLBLAS_STATUS_INVALID_VALUE`（对齐仓内同族 aclblasSrotg 既有口径）；Host 路径直接 CPU 计算，Device 路径走 SIMT kernel |
| 非连续 Tensor 支持 | 不涉及，本算子为纯标量运算 |
| broadcast 规则 | 不涉及 |
| dynamic shape 要求 | 不涉及，无尺寸参数 |
| 原地与视图语义 | a 原地覆写为 r；b 只读；c、s 为独立输出标量 |
| 确定性计算要求 | 不要求（单线程标量计算，天然确定） |
| 空 Tensor 与 0 维处理 | 不涉及（无维度概念；四个标量指针均必须有效） |
| 异步执行 | 依赖 `aclblasSetStream` 绑定 stream；读回 Device 结果前须同步 stream |

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | golden = Netlib crotg 参考实现（C 移植随测试工程提供）；输出 r、c 及 s 实部/虚部分别按 FLOAT32 档判定：rtol=2⁻¹⁰(9.77e-4)、atol=2⁻¹⁶(1.53e-5)、required_matched_ratio=0.99、max_abs_error_limit=1e-2 或 32×ULP；逐元素通过条件 \|actual−golden\| ≤ atol + rtol×\|golden\| | 生态算子开源精度标准（experimental_standard.md） |
| 数学性质验证（本算子特有，与逐标量比对并行） | ① 正交归一性：\|c²+\|s\|²−1\| ≤ atol+rtol；② 旋转零化：\|c·a+s·b−r\| 与 \|−conjg(s)·a+c·b\| ≤ atol+rtol·sqrt(\|a\|²+\|b\|²)；③ 模长保持：\|\|r\|−sqrt(\|a\|²+\|b\|²)\| ≤ atol+rtol·sqrt(\|a\|²+\|b\|²) | 任务书 §3.2.3（大数/小数用例走缩放路径时舍入次序可能与 golden 不同，数学性质为该类用例主判据） |
| 性能标准 | COMPLEX64 场景平均单次耗时（warmup 后采样 >50 次取平均）不高于标杆：单次一般值 11.85us / a=0 分支 9.97us / 大数缩放 9.49us / 批量 1000 次 11.84us / 批量 10000 次 11.53us | 任务书 §3.3 |
| 内存标准 | 不涉及（无 workspace/临时显存分配） | 任务书 §3.4 |

## 测试覆盖概览

测试用例由 `test_cases/gen_csv.py` 固定种子生成 **1200 条**（1000 精度 + 200 性能），固定类别保持全集覆盖：

| 类别 | 前缀 | 条数 | 设计意图 |
| --- | --- | --- | --- |
| L0 基础 | TC_L0 | 12 | 一般复数 / 纯实 3-4-5（c=0.6, s=0.8, r=5 可手算）/ 纯虚 / 等模 / 符号组合 |
| SP 特殊分支 | TC_SP | 18 | b=0（a 八相位）、a=0（b 八相位）、a=b=0（含 -0.0） |
| MAG 数量级 | TC_MAG | 16 | 大数溢出保护（1e30/3e38）、小数下溢保护（1e-30/1e-38/次正规 1.4e-45）、数量级差异（1e±20/1e±40 倍差） |
| PH 相位组合 | TC_PH | 16 | 单位圆 4×4 相位网格（\|a\|=\|b\|=1），覆盖 s 的相位正确性 |
| FL 特殊浮点 | TC_FL | 8 | ±inf / nan / 虚部 inf 传播，期望 SUCCESS 不崩溃 |
| ED 边界负向 | TC_ED | 5 | a/b/c/s 空指针 ×4（INVALID_VALUE）+ 全零合法组合 |
| EX 扩展 | TC_EX | 925 | 确定性采样：50% 均匀 [-5,5] 分量 + 50% 对数数量级（1e-30~1e30）×随机相位 |
| PF 性能 | TC_PF | 200 | 5 条任务书典型 case + 取值路径×iters 扫描 + 批量规模扫描（200~100000） |

补充覆盖口径：

- **handle 空指针负向**：GTest `TEST_F` 硬编码用例（不在 CSV 表达），期望 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`；
- **b 只读断言**：每条成功用例执行后断言 b 原值未被修改（捕获"误用实数 rotg z 恢复规则覆写 b"的常见错误）；
- **Inf/NaN 口径**：TC_FL 用例期望 SUCCESS 不崩溃，数值行为按 IEEE 传播记录备查、不参与精度比对判据（NaN≠NaN 无法比对）；
- **用例一致性自检**：gen_csv.py 生成时内置校验（random_seed 全表唯一、负向全覆盖、TC_PF 前 5 条与任务书 §3.3 典型 case 逐参数一致、特殊分支覆盖），任一失败拒绝产出 CSV；
- **正态分布说明**：任务书 §3.5 要求 50% 正态分布（μ∈[-5,5]，σ∈[0.1,2]）输入，当前测试工程仅支持均匀分布，CSV 字面值层面以对数数量级×随机相位补足覆盖广度，`gen_csv.py --dist mixed` 开关已预留。

## 实现验证状态（设计正确性的先行证据）

核心算法 `CrotgCoreCompute` 已在 CPU 侧与 Netlib crotg golden 完成交叉验证（无需 NPU 环境）：

```
总计用例        : 2000023（23 条手算/特殊/极端固定用例 + 200 万随机/极端输入，跳过非有限值 13892）
通过（比对+性质+b只读）: 1986131    失败: 0
最差逐分量误差   : 0.000e+00        ← 与 golden 逐位一致
结论            : ALL PASS
```

覆盖输入形态：均匀一般值、对数数量级（±60 个十进制量级）×随机相位、近 FLT_MAX（3e38）、次正规（1.4e-45）、特殊分支全集；同时验证三条数学性质（正交归一、旋转零化、模长保持）与 b 只读语义。该结果为设计文档 §算子分析 安全缩放算法与 §算子实现 双路径共享设计的数值正确性提供先行证据；kernel 直调、SIMT 编译与性能达标须在 950PR 环境实测确认。

## 兼容性分析

- `aclblasCrotg` 为**新增接口声明**（放入 `include/cann_ops_blas.h`），不修改任何既有接口与行为，对仓内现有算子无兼容性影响；
- 接口签名与 cuBLAS `cublasCrotg` 逐参数一致，生态用户迁移零改造；
- `blas/rotg/README.md` 产品支持表新增 aclblasCrotg 行（标注 Ascend 950PR：支持），arch35 目录与 srotg 并列共存；
- 核心算法以独立头文件（`crotg_core.h`）实现并被 Host/Device 双路径共享，后续其他产品线（arch22 等）适配时仅需替换 kernel 直调层，数值行为保持一致。

# 参考资料

1. 任务书：`aclblasCrotg_Atlas950PR_task_doc.md`（社区任务 2026）
2. cuBLAS 参考文档（cublasCrotg）：https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-rotg
3. Netlib BLAS 参考实现（crotg.f，Algorithm 978）：https://www.netlib.org/blas/crotg.f
4. Anderson E. (2017). Algorithm 978: Safe Scaling in the Level 1 BLAS. ACM TOMS 44:1–28
5. 生态算子开源精度标准：https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md
6. ops-blas 开源仓：https://gitcode.com/cann/ops-blas
7. Ascend C 算子开发文档：https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/opdevg/Ascendcopdevg/atlas_ascendc_map_10_0002.html
8. Ascend C 算子开发接口文档：https://www.hiascend.com/document/detail/zh/canncommercial/850/API/ascendcopapi/atlasascendc_api_07_0003.html
