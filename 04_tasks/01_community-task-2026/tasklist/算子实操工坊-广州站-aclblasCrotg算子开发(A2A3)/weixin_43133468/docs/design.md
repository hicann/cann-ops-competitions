# 需求背景（required）

## 需求来源

CANN 社区任务：算子实操工坊 —— 算子实操工坊-广州站 aclblasCrotg 算子开发(A2/A3)。

- 任务书：`aclblasCrotg_Atlas800IA3_task_doc.md`（算子实操工坊-广州站-aclblasCrotg算子开发(A2A3)）
- 设计文档提交方式：在 cann-ops-competitions 仓库（04_tasks/01_community-task-2026/tasklist）以 PR 形式提交，**PR 标题：【CANN社区任务】aclblasCrotg算子设计文档**
- 目标合入仓库：https://gitcode.com/cann/ops-blas
  - 算子实现目录：`blas/rotg/`（Atlas A2/A3 对应架构目录 `arch22/`）
  - 测试代码目录：`test/rotg/crotg/arch22/`
- 对标基线接口：cuBLAS `cublasCrotg`；数值语义基准：Netlib BLAS `crotg`（https://www.netlib.org/blas/crotg.f ，LAPACK 参考实现镜像 `BLAS/SRC/crotg.f90`，基于 Anderson *Algorithm 978: Safe Scaling in the Level 1 BLAS*）
- 接口声明位置：`include/cann_ops_blas.h`（当前仅有同族实数接口 `aclblasSrotg`，本任务新增复数接口，与 Atlas 950PR 同名任务共用同一声明，禁止产品私有平行 API）

## 背景介绍

### aclblasCrotg算子实现

基于 ops-blas 开源仓的 Ascend C 工程框架，使用 Ascend C kernel 直调方式，在昇腾 NPU（Atlas 800I A2 / Atlas 800I A3 系列，对应架构目录 arch22）上实现单精度复数（complex64）Givens 旋转参数构造算子 `aclblasCrotg`。

算子接收两个复数标量 a、b，构造一个 2×1 平面旋转，使向量 (a, b)ᵀ 旋转后第二分量消零：

```
┌              ┐ ┌   ┐   ┌   ┐
│  c        s  │ │ a │ = │ r │
│ -conjg(s) c  │ │ b │   │ 0 │
└              ┘ └   ┘   └   ┘
```

其中 c 为**实数**（float32）、s 为**复数**（complex64），满足正交归一性 c² + |s|² = 1；a 被原地覆写为旋转后的范数 r（复数，携带 a 的相位），**b 为只读输入、不覆写**（实数 rotg 把 b 覆写为恢复参数 z 的规则不适用于复数版本）。该算子是 QR 分解、最小二乘等迭代数值算法的基础构件，输出 c、s 可直接作为下游 `crot`/`csrot` 类算子的输入。

### 同类算子现状分析

通过对仓内同族算子的分析，当前 ops-blas 仓 rotg 家族支持的能力如下：

| 算子 | 参数含义 | 数据类型 | 仓内现状 | 输出语义 |
| --- | --- | --- | --- | --- |
| `aclblasSrotg`（实数 Givens 构造） | handle、a、b、c、s 均为 float 标量 | float32 | 已有实现：`blas/rotg/arch22/srotg_host.cpp`、`srotg_kernel.cpp`、`srotg_tiling_data.h`（arch35 同名文件）；CSV 测试工程 `test/rotg/srotg/` | a←r，b←z（恢复参数），c、s 为 float |
| `aclblasCrotg`（复数 Givens 构造） | handle；a、b、s 为 complex64 标量，c 为 float 标量 | COMPLEX64 + FLOAT32 | **仓内无实现、`include/cann_ops_blas.h` 无声明，需新增** | a←r（复数），**b 只读**，c 为 float，s 为 complex64 |

本算子与 srotg 同为**纯标量算子**（4 个单元素指针，无 n、无 incx/incy、无维度轴），工程骨架直接套用 srotg：host 侧的句柄/空指针校验、`aclrtPointerGetAttributes` Host/Device 双路径判别、1 block AIV-only kernel 启动、tiling 只承载 4 个 GM 地址的极简结构、CSV 驱动 GTest 测试工程，均可平移。差异集中在两点：①计算核由实数公式替换为 Netlib `crotg` 复数安全缩放算法（三层分支，是本算子唯一的算法难点）；②b 不再被覆写，测试需额外守护 b 的只读语义。

### aclblasCrotg算子功能分析

算子功能：由复数标量 a、b 构造 Givens 平面旋转参数（c 实数、s 复数），a 原地覆写为 r，b 只读。

- 输入：a（complex64，in/out）、b（complex64，in 只读）
- 输出：a 被覆写为 r（complex64）、c（float32）、s（complex64）
- 支持数据类型：COMPLEX64（实部/虚部各 float32）+ FLOAT32（c）
- 支持广播/形状：不涉及，纯标量运算（shape 恒为 [1]）
- 对标接口：cuBLAS `cublasCrotg(handle, a, b, c, s)`，参数序列逐参数对齐（b 虽只读，签名仍保持非 const 指针）
- 数值基线：Netlib Reference BLAS `crotg`（基于 Algorithm 978 安全缩放）；标准 CBLAS 接口集不含复数 rotg（仅 srotg/drotg），golden 不取 cblas

# 需求分析（required）

## 需求描述

在 ops-blas 开源仓工程框架下，使用 Ascend C 编程语言实现 `aclblasCrotg` 句柄式 BLAS 接口：通过 handle 绑定 stream 直调 NPU kernel，host CPU 路径与 device kernel 路径均严格按 Netlib `crotg`（Algorithm 978 安全缩放）实现，保证 1e30 量级不溢出、1e-38/次正规量级不下溢，精度、性能、内存均满足任务书验收标准，测试代码与 README 同步交付。

## 需求拆解

1. 在 `include/cann_ops_blas.h` 新增 `aclblasCrotg` 声明（紧邻 `aclblasSrotg`，与 950PR 同名任务共用，禁止产品私有平行接口），参数与 `cublasCrotg` 逐参数一致。
2. 实现 host 侧（`blas/rotg/arch22/crotg_host.cpp`）：handle/空指针校验、runtime context 校验、4 指针 Host/Device 同侧判别（4 次属性查询复用同一个 `aclrtPtrAttributes`）、全 host 走 CPU 计算、全 device 启动 1 block kernel、混合指针返回 `ACLBLAS_STATUS_INVALID_VALUE`。
3. 实现 device 侧（`blas/rotg/arch22/crotg_kernel.cpp` + `crotg_tiling_data.h`）：AIV-only 单核标量 kernel，逐行移植 Netlib `crotg`，a 原地覆写为 r、b 只读、写出 c 与 s。
4. 特殊分支对齐（判别顺序先 b 后 a）：b=0 → c=1,s=0,r=a；a=0 且 b≠0 → c=0,s=conjg(b)/|b|,r=|b|（正实数）；a=b=0 → 落在 b=0 分支 c=1,s=0,r=0。
5. 数值稳定性：采用 safmin/safmax 安全缩放（含 f、g 数量级悬殊时的二次缩放因子 w），禁止直接计算 |a|²+|b|² 裸和，禁止使用 hypot/复数 abs 内建改变舍入次序。
6. 标量热路径开销控制（code-915 已落地 8 项优化技能，统一汇总见 §3.2.3）：rtmax 阈值 constexpr 预计算（host/kernel/golden 三处同步，消除运行时 sqrtf 且位级一致）、删除每次调用的 OP_LOGD/OP_LOGI（仅保留错误路径 OP_LOGE）、4 次指针属性查询复用同一 `aclrtPtrAttributes`、tiling 免冗余零初始化、kernel 启动签名精简为 (tiling, stream) 双参数、kernel 侧无 UB/无 workspace/无 DataCopy 的极简执行框架、host/kernel/golden 三方同构位级一致、b 只读的代码级保证。
7. 新建 CSV 驱动 GTest 工程（参照 `test/rotg/srotg/`），golden 为 Netlib crotg 的 C 移植，覆盖 1000 条精度用例 + 200 条性能用例及全部负向/特殊值场景；更新 `blas/rotg/README.md` 产品支持表。

# 详细设计（required）

## 算子分析

### 数学公式

设复数标量 a、b，定义：

```
|x|      = sqrt( Re(x)² + Im(x)² )              （复数模）
conjg(x) = Re(x) − i·Im(x)                      （共轭）
sgn(x)   = x / |x|    (x ≠ 0)； sgn(x) = 1      (x = 0)
h        = |a|² + |b|²
```

则 Givens 旋转参数为：

```
c = |a| / sqrt(h)                  （非负实数）
s = sgn(a) · conjg(b) / sqrt(h)    （复数）
r = sgn(a) · sqrt(h)               （复数，携带 a 的相位；a=0 时 r = |b| 为正实数）
```

数学性质（golden 逐标量比对之外的并行验收口径）：

1. **正交归一性**：c² + |s|² = |a|²/h + |b|²/h = 1；
2. **旋转零化**：c·a + s·b = r；−conjg(s)·a + c·b = 0；
3. **模长保持**：|r| = √(|a|²+|b|²)。

**特殊分支**（与 Netlib crotg 一致，判别顺序为先 b 后 a）：

| 条件 | c | s | r（覆写 a） | b |
| --- | --- | --- | --- | --- |
| b = 0（含 a=b=0、含 ±0 分量） | 1.0 | (0,0) | a（保持原值，含 −0 符号） | 不变 |
| a = 0 且 b ≠ 0 | 0.0 | conjg(b)/\|b\| | \|b\|（正实数，虚部 0） | 不变 |
| a ≠ 0 且 b ≠ 0 | \|a\|/√h | sgn(a)·conjg(b)/√h | sgn(a)·√h | 不变 |

**安全缩放的必要性**：float32 最大有限值约 3.4e38、最小正规数约 1.18e-38。直接计算分量平方和时，1e30 量级输入平方（1e60）必溢出为 Inf，1e-38 量级输入平方（1e-76）必下溢为 0，导致 c/s/r 全错。Netlib crotg 采用 Algorithm 978 的钳位缩放：把参与平方的量统一除以 u = min(safmax, max(safmin, max(|f|,|g|)))，使缩放后分量平方落于安全区间，最终再把 r（必要时连同 c）按 u/w 放大还原。

float32 常量（radix=2、minexponent=−125、maxexponent=128，host 与 kernel 取同一组 constexpr）：

| 常量 | 表达式 | float32 值（code-915 constexpr） |
| --- | --- | --- |
| safmin | 2⁻¹²⁶ | 1.1754943508222875e-38f |
| safmax | 2¹²⁷ | 1.7014118346046923e+38f |
| rtmin | sqrt(safmin) = 2⁻⁶³ | 1.0842021724855044e-19f |
| rtmax（a=0 分支，CROTG_RTMAX_G） | sqrt(safmax/2) = 2⁶³ | 9.223372036854776e+18f（0x5f000000） |
| rtmax（一般分支初值，CROTG_RTMAX_F） | sqrt(safmax/4) = 2^62.5 | 6.521908801048674e+18f（0x5eb504f3） |

> code-915 中两个 rtmax 均为预先算好的正确舍入 float32 常量（与运行时 `sqrtf(safmax/2)`、`sqrtf(safmax/4)` 位级一致），消除 host/kernel 热路径上各 1~2 次 sqrtf。

整体分支流程（严格对应 crotg.f90 执行语句顺序）：

```
                 ┌──────────────────────────┐
                 │ f = a；g = b（拷入局部量） │
                 └─────────────┬────────────┘
                               ▼
                        g == (0,0) ?
                     ┌───────┴────────┐
                    是                否
                     ▼                ▼
          c=1, s=(0,0), r=f   f == (0,0) ?
                              ┌──────┴──────┐
                             是             否
                              ▼             ▼
                   c=0；r=|g|（正实数）    f1=max(|Re f|,|Im f|)
                   按 g 形态分三支：        g1=max(|Re g|,|Im g|)
                   · Re(g)=0（纯虚）       rtmax=CROTG_RTMAX_F
                   · Im(g)=0（纯实）       f1,g1 均 ∈(rtmin,rtmax) ?
                   · 一般复数：             ┌──────┴──────┐
                     g1∈(rtmin,rtmax)      是            否
                     → 无缩放               ▼            ▼
                     → 否则 u 钳位缩放   fs=f,gs=g     u=min(safmax,
                              │          直接求         max(safmin,max(f1,g1)))
                              │          f2,g2,h2     gs=g/u；若 f1/u<rtmin
                              │                          再取 v 缩放 f，w=v/u
                              ▼                          （否则 w=1, fs=f/u）
                   s = conjg(g缩放)/d      计算 f2、g2、h2（缩放路径 h2 含 w²）
                              └──────────────┬───────────┘
                                             ▼
                         公共尾部（f2 ≥ h2·safmin ?）：
                     ┌─────── 是 ────────┐┌─────── 否（g2≫f2 防溢出）───────┐
                     │ c=sqrt(f2/h2)     ││ d=sqrt(f2·h2); c=f2/d           │
                     │ r=fs/c            ││ c≥safmin → r=fs/c               │
                     │ f2>rtmin 且       ││ 否则 → r=fs·(h2/d)              │
                     │ h2<2·rtmax:       ││ s=conjg(gs)·(fs/d)              │
                     │  s=conjg(gs)·     │└──────────────────────────────────┘
                     │   (fs/sqrt(f2h2)) │
                     │ 否则:             │
                     │  s=conjg(gs)·r/h2 │
                     └───────────────────┘
                                             ▼
                        缩放路径：c *= w；r *= u（无缩放路径 w=u=1，乘 1 精确）
                                             ▼
                        a ← r；写出 c、s；b 全程只读
```

### 支持数据类型

| 数据 | 类型 | 说明 |
| --- | --- | --- |
| a | COMPLEX64（`aclblasComplex`，实部/虚部各 float32） | 输入/输出标量，运算后原地覆写为 r |
| b | COMPLEX64 | **只读**输入标量，不覆写 |
| c | FLOAT32（实数） | 纯输出标量，值域自然落于 [0,1] |
| s | COMPLEX64 | 纯输出标量 |

复数类型以仓内 `include/cann_ops_blas_common.h` 定义为准：

```cpp
typedef struct aclblasComplex {
    float real;
    float imag;
} aclblasComplex;
```

参数说明（对应任务书 §2.4）：

| 参数 | 输入/输出 | 内存位置 | dtype | shape | 值域 | 异常行为 |
| --- | --- | --- | --- | --- | --- | --- |
| handle | 输入 | Host | 句柄（携带 stream） | - | 已创建的有效句柄 | nullptr → `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| a | 输入/输出 | Host 或 Device | COMPLEX64 | [1] | 实部/虚部为 float32 全集（含 0、±0、纯实、纯虚、1e30、1e-38、次正规、Inf/NaN） | nullptr → `ACLBLAS_STATUS_INVALID_VALUE` |
| b | 输入（只读） | Host 或 Device | COMPLEX64 | [1] | 同 a | nullptr → `ACLBLAS_STATUS_INVALID_VALUE` |
| c | 输出 | Host 或 Device | FLOAT32 | [1] | 输出参数无输入约束 | nullptr → `ACLBLAS_STATUS_INVALID_VALUE` |
| s | 输出 | Host 或 Device | COMPLEX64 | [1] | 输出参数无输入约束 | nullptr → `ACLBLAS_STATUS_INVALID_VALUE` |

返回值 `aclblasStatus_t` 以 `include/cann_ops_blas_common.h` 为准：`ACLBLAS_STATUS_SUCCESS` / `ACLBLAS_STATUS_INVALID_VALUE` / `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` / `ACLBLAS_STATUS_NOT_INITIALIZED`。

### 支持形状

本算子为纯标量运算：4 个参数均为单元素指针，shape 恒为 [1]；无向量长度、无步长、无维度轴，不涉及 broadcast、dynamic shape、非连续 Tensor、空 Tensor/0 维。

约束：a、b、c、s 必须**全部位于 Host 侧或全部位于 Device 侧**，混合 Host/Device 指针返回 `ACLBLAS_STATUS_INVALID_VALUE`（对齐 srotg 既有口径）。

## 算子实现

### 实现方案

总体沿用 srotg arch22 的"极简标量算子"范式：**无 workspace、无 UB 张量缓冲、无 DataCopy 流水、无多核切分、无 tilingKey**；device 路径仅启动 1 个 AIV block，核内通过 `GlobalTensor::GetValue/SetValue` 直接访问 4 个标量 GM 地址，计算量仅数十条标量浮点指令（2~4 次 sqrt），耗时完全由 kernel launch 固定开销主导。host 与 kernel 各保留一份**同构算法副本**（同常量、同分支顺序、同运算次序），避免 device 编译器引入 host 头文件。在共用骨架之上，code-915 针对**标量热路径**落地 8 项优化技能（constexpr 阈值预计算、热路径零日志、指针属性查询复用、tiling 免零初始化、启动签名精简、极简执行框架、三方同构、b 只读代码级保证），统一汇总于 §3.2.3。

#### 3.2.1 host侧设计：

**（1）接口声明（加入 `include/cann_ops_blas.h`，紧邻 `aclblasSrotg`，与 950PR 共用）**

```cpp
aclblasStatus_t aclblasCrotg(aclblasHandle_t handle,
                             aclblasComplex* a,
                             aclblasComplex* b,
                             float* c,
                             aclblasComplex* s);
```

**（2）校验与双路径判别（骨架照搬 srotg arch22）**

| 顺序 | 校验项 | 判定条件 | 返回码 |
| --- | --- | --- | --- |
| 1 | handle | == nullptr | `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| 2 | a/b/c/s | 任一 == nullptr | `ACLBLAS_STATUS_INVALID_VALUE` |
| 3 | runtime context | `aclrtGetCurrentContext` 失败或无 context | `ACLBLAS_STATUS_NOT_INITIALIZED` |
| 4 | 指针位置 | 依次 `aclrtPointerGetAttributes` 取 a/b/c/s 的 location（4 次查询复用同一个 `aclrtPtrAttributes`） | 查询失败 → `ACLBLAS_STATUS_INVALID_VALUE` |
| 5 | 同侧约束 | 全 host → CPU 计算；全 device → 启 kernel；混合 → 报错 | `ACLBLAS_STATUS_INVALID_VALUE` |

全 host 路径直接在 host 调 `CrotgCpuCompute`，无 kernel、无 memcpy、无 stream 同步；全 device 路径填充 tiling（4 个 GM 地址）后启动 1 block 的 `crotg_kernel`，stream 取自 handle，host 不主动同步（由调用方管理，与 srotg 一致）。

**（3）Host CPU 核心算法（Netlib crotg.f90 逐行 C 移植）**

复数只用"实部/虚部二元组 + 实数除 + 复数乘"，**不引入通用复数除法、不使用 hypotf/复数 abs 内建**（Fortran 原注释明确 "This algorithm do not use the intrinsic complex abs"，裸内建会改变缩放路径舍入次序）。核心结构（完整实现见 `blas/rotg/arch22/crotg_host.cpp`）：

```cpp
constexpr float CROTG_SAFMIN  = 1.1754943508222875e-38f;  // 2^-126
constexpr float CROTG_SAFMAX  = 1.7014118346046923e+38f;  // 2^127
constexpr float CROTG_RTMIN   = 1.0842021724855044e-19f;  // 2^-63
constexpr float CROTG_RTMAX_G = 9.223372036854776e+18f;    // 2^63（a=0 分支）
constexpr float CROTG_RTMAX_F = 6.521908801048674e+18f;    // 2^62.5（一般分支）

struct Complex { float real; float imag; };
// CAbs/CMax/CMin 三元表达式手写；AbsSq = re*re+im*im；
// Conjg={re,-im}；CMul 为标准复数乘；CScale 为复数乘正实数。

static void CrotgCpuCompute(aclblasComplex* a, const aclblasComplex* b,
                            float* cOut, aclblasComplex* sOut)
{
    const Complex f = {a->real, a->imag};
    const Complex g = {b->real, b->imag};
    // 1) g==0（先判 b）：c=1, s=0, r=f
    // 2) f==0 且 g!=0：c=0；按 g 纯虚/纯实/一般复数三支求 r=|g|、s=conjg(g)/|g|
    //    一般复数支以 g1 与 (rtmin, RTMAX_G) 比较决定是否 u 钳位缩放
    // 3) 一般路径：f1/g1 与 (rtmin, RTMAX_F) 比较 → 无缩放(u=w=1) 或缩放
    //    缩放路径 u=min(safmax,max(safmin,max(f1,g1)))；f1/u<rtmin 时另取 v、w=v/u
    //    f2=AbsSq(fs); g2=AbsSq(gs); h2=f2*w*w+g2;
    //    公共尾部按 f2 >= h2*safmin 分两支求 c/s/r（谓词含等号）
    //    收尾：c *= w; r *= u;   // 无缩放路径乘 1 为精确恒等
    *a = {r.real, r.imag};   // a 原地覆写为 r
    *cOut = c;
    *sOut = {s.real, s.imag};
    // 不写 *b
}
```

> 与 Fortran 原结构的等价性：Fortran 把无缩放/缩放两份计算复制，本实现归并为一份——无缩放路径令 u=w=1，IEEE 754 下乘 1、乘 1² 为精确恒等，两条写法位级等价。kernel 与 golden（`crotg_golden.h`）与此块完全同构（同常量、同分支、同运算次序），三方一致性已经归一化结构 diff 与真机 2006 项测试验证。

**（4）Kernel 启动（code-915 精简后）**

```cpp
void crotg_kernel_do(const CrotgTilingData& tiling, void* stream)
{
    // Scalar op: a single block; all addresses are carried in tiling.
    crotg_kernel<<<1, nullptr, stream>>>(tiling);
}

static aclblasStatus_t LaunchCrotgKernel(aclblasHandle_t handle,
                                         aclblasComplex* a, aclblasComplex* b,
                                         float* c, aclblasComplex* s)
{
    aclrtStream useStream = handle->stream;
    // 32 字节 tiling 的 4 个字段下方全部赋值，不做冗余零初始化
    CrotgTilingData tiling;
    tiling.a = reinterpret_cast<uint64_t>(a);
    tiling.b = reinterpret_cast<uint64_t>(b);
    tiling.c = reinterpret_cast<uint64_t>(c);
    tiling.s = reinterpret_cast<uint64_t>(s);
    crotg_kernel_do(tiling, useStream);
    return ACLBLAS_STATUS_SUCCESS;
}
```

启动路径上的热开销控制手段（阈值 constexpr 预计算、零日志、`aclrtPtrAttributes` 复用、签名精简等）与 kernel 侧、精度侧的优化统一收敛为 8 项优化技能，见 §3.2.3。

##### 1. 分核策略：

不涉及多核切分。本算子为纯标量算子，全 device 路径固定启动 **1 个 block**（`crotg_kernel<<<1, nullptr, stream>>>`），无核间均分、无大小核、无 `GetBlockIdx` 尾块判定逻辑；标量计算量仅数十条指令，多核分发没有任何收益、只会增加 launch 与同步成本。

##### 2. 数据分块和内存优化策略：

不涉及数据分块/Tile 切分。kernel 不使用 TQue/TBuf/DataCopy/UB 张量缓冲，4 个操作数以 `GlobalTensor<float>` 标量方式访问：a、b、s 各 2 个连续 float（offset 0=real、1=imag），c 为 1 个 float；仅 GetValue 读入、SetValue 写出。算子无 workspace、host 侧无 device 动态分配，device 路径仅访问用户提供的 28 字节标量内存（a/b/s 各 8B + c 4B），tiling（32B）按值随 kernel 参数下发。

##### 3. tilingkey规划策略：

不需要 tilingKey。无 shape/dtype/位置等运行时多态分支；Host/Device 路径已在 host 侧分流，b=0、a=0、缩放/无缩放等分支全部在核内由标量谓词判定，device kernel 单实例覆盖全部输入。

#### 3.2.2 kernel侧设计：

kernel 分为 Init 和 Process 两个阶段，Process 内部等价于 CopyIn（标量读入）、Compute（计算）、CopyOut（标量写出）三步：

1. **Init**：用 tiling 中的 4 个 GM 地址初始化 `aGM/bGM/cGM/sGM`（`GlobalTensor<float>`，长度分别为 2/2/1/2）；
2. **Process-CopyIn**：`GetValue(0/1)` 读入 a→f、b→g 两个复数标量；bGM 此后不再访问，从代码层面保证只读；
3. **Process-Compute**：调用与 host 同构的纯函数 `Compute(f, g, c, s, r)`；`Abs/Max/Min` 用三元表达式手写（与 srotg kernel 风格一致），开方用 Ascend C 标量 `sqrt`，常量、分支谓词（严格开区间 `>`/`<`、`f2 >= h2*safmin`、`c >= safmin`、分量级 `== 0.0f` 判别）、运算次序与 host/Fortran 逐行对齐；
4. **Process-CopyOut**：`SetValue` 写 a←r（2 float）、c（1 float）、s（2 float）；**bGM 不做任何 SetValue**。

```cpp
class CrotgKernel {
public:
    __aicore__ inline void Init(const CrotgTilingData& tiling)
    {
        aGM.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(tiling.a), 2);
        bGM.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(tiling.b), 2); // 只读
        cGM.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(tiling.c), 1);
        sGM.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(tiling.s), 2);
    }
    __aicore__ inline void Process()
    {
        const Complex f = {aGM.GetValue(0), aGM.GetValue(1)};
        const Complex g = {bGM.GetValue(0), bGM.GetValue(1)};
        float cValue; Complex sValue, rValue;
        Compute(f, g, cValue, sValue, rValue);   // 与 host 同构的纯函数
        aGM.SetValue(0, rValue.real); aGM.SetValue(1, rValue.imag);
        cGM.SetValue(0, cValue);
        sGM.SetValue(0, sValue.real); sGM.SetValue(1, sValue.imag);
    }
private:
    GlobalTensor<float> aGM, bGM, cGM, sGM;
};

extern "C" __global__ __aicore__ void crotg_kernel(const CrotgTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    CrotgKernel op;
    op.Init(tiling);
    op.Process();
}
```

特殊浮点行为：

- **±0**：`-0.0f == 0.0f` 为真，a=b=0（含 −0）走 g==0 分支，r 保持 f（可能为 −0），与 golden 一致；
- **次正规**（如 1.4e-45）：比较/缩放按 IEEE 规则自然传播，Algorithm 978 本身即为次正规输入设计；
- **Inf/NaN**：不做特殊拦截，进入一般分支后按 IEEE 规则传播，验收口径仅要求返回 SUCCESS、不崩溃（TC_FL 用例不做数值比对）。

与 Netlib crotg 逐行对齐的关键检查点（本算子核心风险点）：

1. **分支顺序**：先判 `g == 0`（b=0），再判 `f == 0`（a=0），最后一般路径；a=b=0 因此落在 b=0 分支（c=1 而非 c=0）；
2. **零判别为复数分量级精确相等**：`Re==0 && Im==0`；a=0 分支内部再按 `Re(g)==0`/`Im(g)==0` 区分纯虚、纯实、一般复数；
3. **a=0 分支 r 恒为正实数**，s = conjg(g缩放)/d，虚部取 **−**g.imag/d，共轭符号不能错；
4. **谓词严格开闭区间**：`g1 > rtmin .and. g1 < rtmax`、一般路径 f1/g1 同时落区间；`f2 >= h2*safmin`（含等号）、`c >= safmin`（含等号）、`f1/u < rtmin`；
5. **双缩放因子**：f 欠缩放时另取 v、`w = v/u`、`h2 = f2*w*w + g2`，收尾 `c *= w; r *= u`，顺序不可调换；
6. **只做"复数×复数"和"复数/正实数"**，不化简出通用复数除法；s 的两种写法选择条件（f2>rtmin 且 h2<2·rtmax）照搬；
7. **不使用 hypot/complex abs 内建**，模长一律 `sqrt(Re²+Im²)`，保持与 golden 相同舍入次序；
8. **b 只读**：host、kernel、测试 wrapper 三处均不得出写 b，测试对 b 做位级不变性断言。

工程文件清单（新增/修改）：

| 文件 | 类型 | 说明 |
| --- | --- | --- |
| `include/cann_ops_blas.h` | 修改 | 在 `aclblasSrotg` 声明旁新增 `aclblasCrotg` 声明（与 950PR 共用） |
| `blas/rotg/arch22/crotg_host.cpp` | 新增 | 校验/context 检查/4 指针同侧判别（复用 PtrAttr）、host CPU 算法、1 block kernel 启动 |
| `blas/rotg/arch22/crotg_kernel.cpp` | 新增 | AIV-only 标量 kernel + 与 host 同构的 Netlib crotg 算法；`crotg_kernel_do(tiling, stream)` |
| `blas/rotg/arch22/crotg_tiling_data.h` | 新增 | `CrotgTilingData`（4 个 uint64_t GM 地址，32B） |
| `blas/rotg/README.md` | 修改 | 接口表新增 aclblasCrotg；产品支持表 Atlas A2/A3 标注"支持"；补函数原型/参数/约束/示例 |
| `test/rotg/crotg/CMakeLists.txt` | 新增 | 注册 `crotg_test`；含 ascend950 guard（arch35 无本交付源码时静默跳过） |
| `test/rotg/crotg/crotg_param.h` | 新增 | CSV 解析：复数 4 分量字面值（`null` 哨兵）、c/s 列（`out`/`null`）、iters、HasSpecialFloat |
| `test/rotg/crotg/crotg_golden.h` | 新增 | Netlib crotg 的 C 移植 golden（与 host 同构），提供 `aclblasCrotg_cpu` |
| `test/rotg/crotg/arch22/crotg_npu_wrapper.h` | 新增 | 6 个 TEST_F 使用的 device 包装；空指针透传；非空时 H2D 拷 a/b、D2H 仅拷 a/c/s（不拷 b） |
| `test/rotg/crotg/arch22/crotg_test.cpp` | 新增 | device/host 双路径 vs golden、b 位级只读、三条数学性质；PF 连发仅断言成功；6 个独立 TEST_F |
| `test/rotg/crotg/arch22/crotg_test.csv` | 新增 | 1200 条用例（配套 gen_csv.py 固定种子 20260823 生成） |

构建系统无需额外注册：`blas/CMakeLists.txt` 通过 GLOB_RECURSE 自动收集 `*/arch22/*.cpp`；测试按 `--ops=crotg` 自动发现 `test/rotg/crotg/`。

#### 3.2.3 实现优化技能（code-915 已落地优化项）

本算子为纯标量热路径算子：核内计算仅数十条标量浮点指令（2~4 次开方），单次调用耗时几乎完全由 kernel launch 固定开销与 host 侧参数组装、运行时查询、日志门检决定。code-915 的优化重心因此不在核内计算密度，而在三处：**压薄 host 侧每次调用的固定开销、砍掉 kernel 执行框架的冗余结构、以同构副本消除路径间精度分叉**。共落地 8 项优化技能，总览如下：

| # | 优化技能 | 侧别 | 实现手段（code-915 落地位置） | 收益 |
| --- | --- | --- | --- | --- |
| 1 | 安全缩放阈值 constexpr 预计算 | host + kernel + golden | CROTG_RTMAX_G/F 以正确舍入 float32 字面量写死（0x5f000000 / 0x5eb504f3） | 消除缩放分支热路径 sqrtf，位级零损失 |
| 2 | 热路径零日志 | host（`LaunchCrotgKernel`） | 删除每次调用的 OP_LOGD/OP_LOGI，仅错误路径保留 OP_LOGE | 省两次日志门检；排障级别下避免每调用 2~4 μs 格式化打印 |
| 3 | 指针属性查询单实例复用 | host（`aclblasCrotg`） | 4 次 `aclrtPointerGetAttributes` 复用同一个零初始化的 `aclrtPtrAttributes` | 免每次查询重填约 32B 栈 |
| 4 | tiling 免冗余零初始化 | host（`LaunchCrotgKernel`） | 32B 四字段全部赋值，去掉 `= {}` 零填充 | 热路径少一次 32B 清零 |
| 5 | kernel 启动签名极简 | host → kernel | `crotg_kernel_do(tiling, stream)` 双参数；操作数地址由 tiling 承载、分核数固定为 1 | 减少参数组装与传参开销 |
| 6 | 1 block AIV-only 极简执行框架 | kernel（`crotg_kernel.cpp`） | 无 TQue/TBuf/DataCopy/UB/workspace，`GlobalTensor` 标量直访 | 流水建立/UB 初始化/队列管理开销全部归零 |
| 7 | host/kernel/golden 三方同构 | 跨侧（三处算法副本） | 同常量、同分支顺序、同谓词开闭区间、同运算次序 | 双路径位级一致；golden 三方互验 |
| 8 | b 只读的代码级保证 | host + kernel + 测试 | 三处均无写 b 路径 + 测试位级不变断言 | 只读语义由结构保证，不依赖人工约定 |

**（1）host 侧热路径开销控制（技能 1~5）**

1. **安全缩放阈值 constexpr 预计算**：Netlib crotg 中 a=0 分支阈值为 `sqrt(safmax/2)`、一般分支初值为 `sqrt(safmax/4)`，早期实现每次进入分支都执行运行时 sqrtf。code-915 将两个阈值预计算为正确舍入的 float32 常量，host、kernel、golden 三处同步使用，与运行时 `sqrtf` 位级一致：

```cpp
// Precomputed, correctly-rounded float32 thresholds (== sqrtf of the runtime
// expression, bit-identical): sqrt(2^127/2) = 2^63; sqrt(2^127/4) = 2^62.5.
constexpr float CROTG_RTMAX_G = 9.223372036854776e+18f;   // 0x5f000000
constexpr float CROTG_RTMAX_F = 6.521908801048674e+18f;   // 0x5eb504f3
```

消除 host/kernel 热路径上各 1~2 次 sqrtf，且不改变任何舍入行为（性能优化与精度零损失兼得）。

2. **热路径零日志**：`LaunchCrotgKernel` 内不输出任何 OP_LOGD/OP_LOGI——即使被默认日志级别过滤，宏展开后仍需穿过运行时日志门检；INFO/DEBUG 排障级别下则是每调用 2~4 μs 的真实格式化打印。错误路径（校验失败、属性查询失败、混合指针）保留 OP_LOGE 便于定位。code-915 在启动函数留有注释锚点：

```cpp
// No per-call logging on this hot scalar path: even a level-filtered
// OP_LOGD/OP_LOGI macro costs ~1 us through the runtime log gate.
```

3. **指针属性查询单实例复用**：4 个操作数指针的 Host/Device 判别需 4 次 `aclrtPointerGetAttributes`。调用方对属性结构仅做一次零初始化，4 次查询复用同一实例，避免每次查询重填约 32 字节栈：

```cpp
// Check pointer locations (single attribute buffer reused by all four).
aclrtPtrAttributes ptrAttr{};
```

4. **tiling 免冗余零初始化**：`CrotgTilingData` 共 32B（4 × uint64_t GM 地址），4 个字段在启动路径全部赋值，声明处不做 `= {}` 冗余零填充，热路径少一次 32B 清零。

5. **kernel 启动签名极简**：启动函数收敛为 `crotg_kernel_do(tiling, stream)` 两参数——4 个操作数指针已由 tiling 承载、分核数固定为 1，均无需作为入参传递，减少参数组装与传参开销。

**（2）kernel 侧极简执行框架（技能 6）**

标量算子没有数据搬运与切分诉求，kernel 完全不引入张量编程的脚手架：

- 无 TQue/TBuf/UB 缓冲、无 DataCopy 流水、无 LocalTensor 分配：4 个操作数以 `GlobalTensor<float>` 标量直访（a/b/s 各 2 个 float、c 1 个 float），仅 GetValue 读入、SetValue 写出；
- `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY)` 单 block 执行：无核间分发、无 GetBlockIdx 尾块判定；
- 无 workspace、host 侧无 device 动态分配：仅访问用户提供的 28B 标量内存，tiling（32B）按值随 kernel 参数下发；
- 计算封装为与 host 同构的纯函数 `Compute(f, g, c, s, r)`：核内无任何内存管理语句。

该框架使 kernel 执行的固有开销（流水建立、UB 初始化、队列管理）全部归零，静态指令体积最小化，与技能 5 的启动精简共同构成单次调用 3 μs 量级稳态延迟的基础。

**（3）host/kernel/golden 三方同构位级一致（技能 7）**

host CPU 路径（`CrotgCpuCompute`）、device kernel（`Compute`）、测试 golden（`CrotgGoldenCompute`）三处保持同一份算法结构：相同常量（含预计算阈值）、相同分支顺序（先 g==0 后 f==0 再一般路径）、相同谓词开闭区间（严格 `>`/`<`、`f2 >= h2*safmin` 与 `c >= safmin` 含等号、分量级 `== 0.0f` 零判别）、相同运算次序（`h2 = f2*w*w + g2` 不可重排/结合）。开方分别用 `std::sqrt`（host/golden）与 Ascend C 标量 `sqrt`（kernel），均为 IEEE 754 正确舍入，位级一致。收益：

- device 与 host 双路径输出位级一致，调用方无论指针落侧如何结果确定；
- golden 与实现同构，实现错误（谓词抄错、共轭符号错、缩放因子遗漏）会被 2006 项真机用例的 8658 处断言即时暴露，而不是被容差掩盖。

**（4）b 只读的代码级保证（技能 8）**

b 的只读语义不依赖人工约定，由三处代码结构直接保证：host `CrotgCpuCompute` 从不写 `*b`（源码注释 "b is never written."）；kernel `bGM` 仅在 Process 起始 GetValue 一次、全无 SetValue；测试 wrapper 对 device 路径仅回拷 a/c/s 不拷 b（"b is read-only: no D2H copy"），测试主体对 b 做位级不变性断言（含 NaN payload）。

**（5）优化效果（910B4 真机实测）**

- 5 个标杆 case 稳态实测 3.15~3.56 μs，全部低于任务书标杆值（4.20~4.95 μs），余量 24%~36%（五 case 明细见"可维可测分析-精度标准/性能标准"）；
- 单 case 间差异 < 0.1 μs：耗时由 launch 固定开销主导、与输入分支（一般值/a=0/大数缩放）无关，验证热路径开销控制的有效性；
- 10 万次背靠背连发无累积错误、无 OOM；
- host 侧优化不改变任何数值行为：所有预计算阈值与运行时值位级一致，2006 项用例双路径位级一致复验通过。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas 800I/T A2（910B3，arch22） | √（2026-09-13 真机通过编译、2006 项精度/功能、400 项性能用例） |
| Atlas 800I A3（910B4，arch22） | √（2026-09-15 真机通过编译、2006 项精度/功能、400 项性能用例；§3.3 五 case 稳态 3.15~3.56 μs） |

- **CANN 版本**：CANN 9.1.0（任务书 §3.1）；
- 构建命令：`bash build.sh --ops=crotg --soc=ascend910b3`（A3 使用 `--soc=ascend910b4`）；910B3/910B4 同属 arch22，同一套源码与二进制；
- arch35（Ascend 950 系列）由同名 950PR 任务负责，本任务仅共用 `include/cann_ops_blas.h` 声明，不新增 arch35 源文件，不影响既有构建。

## 算子约束限制

| 约束项 | 内容 |
| --- | --- |
| 参数合法性 | handle 非空；a、b、c、s 均不可为 nullptr；违反分别返回 `HANDLE_IS_NULLPTR` / `INVALID_VALUE` |
| Host/Device 内存一致性 | a、b、c、s 必须全部 Host 或全部 Device；混合返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| 数值稳定性 | 必须走 Netlib Algorithm 978 安全缩放，禁止裸算分量平方和；禁止换用 hypot/复数 abs 内建 |
| b 只读语义 | host/device 任何路径不得覆写 b（与 srotg 覆写 z 相区别） |
| 非连续 Tensor | 不涉及（纯标量） |
| broadcast | 不涉及 |
| dynamic shape | 不涉及（无尺寸参数） |
| 原地与视图 | a 原地覆写为 r；c、s 独立输出；无视图 |
| 空 Tensor/0 维 | 不涉及；4 个标量指针必须有效 |
| 异步执行 | 依赖 `aclblasSetStream` 绑定 stream；读回 device 结果前调用方自行同步 |
| Inf/NaN | 不拦截，按 IEEE 传播，要求不崩溃、返回 SUCCESS |
| 多核/分块/workspace | 不涉及：固定 1 block 标量 kernel，无 UB 张量缓冲、无 device 动态分配 |

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述(不涉及说明原因) | 标准来源 |
| --- | --- | --- |
| 精度标准（逐标量） | golden = Netlib `crotg` C 移植单标杆（不取 cblas，标准 CBLAS 无复数 rotg）；r（实/虚部）、c、s（实/虚部）共 5 个分量分别按 FLOAT32 判定：逐元素 `\|actual−golden\| ≤ atol + rtol·\|golden\|`，matched_ratio ≥ 0.99 且 max_abs_error ≤ 1e-2（或 32 ULP）。参数：rtol = 2⁻¹⁰（9.77e-4），atol = 2⁻¹⁶（1.53e-5） | 任务书 §3.2；生态算子开源精度标准 experimental_standard.md；`test/frame/verify.h` |
| 精度标准（数学性质） | 每条用例以**原始 a、b** 并行验证：①正交归一 \|c²+\|s\|²−1\| ≤ atol+rtol；②旋转零化 \|c·a+s·b−r\| 与 \|−conjg(s)·a+c·b\| ≤ atol+rtol·√h；③模长保持 \|\|r\|−√h\| ≤ atol+rtol·√h | 任务书 §3.2.3 |
| 真机精度结果 | 910B3（2026-09-13）与 910B4（2026-09-15）均为 **2006/2006 PASSED、0 FAILED**（6 TEST_F + 1000 条非 PF 用例 ×device/host 双路径），8658 处断言与 golden 位级 exact match；device/host 两路径结果一致；TC_FL 8 条仅校验 SUCCESS 不崩溃 | 自测报告 `测试结果/aclblasCrotg自测报告*.md` |
| 性能标准 | 纯标量固定调用开销：5 个标杆 case（单次一般值/a=0/大数缩放 ≤ 4.65/4.95/4.66 μs；连发 1000/10000 ≤ 4.7/4.2 μs）；warmup 后有效采样 >50 次取平均。910B4 真机稳态实测 3.15~3.56 μs，全部低于标杆（余量 24%~36%）；单 case 间差异 <0.1 μs，耗时由 launch 开销主导、与输入分支无关 | 任务书 §3.3 |
| 性能用例状态 | 200 条 TC_PF（连发规模 200~100000）× device/host 双路径 = **400/400 PASSED**，10 万次背靠背无累积错误、无 OOM | 自测报告 PF 日志 |
| 内存标准 | 不涉及独立内存指标（任务书 §3.4）：算子无 workspace、无 UB 张量占用、host 侧无 device 动态分配，仅访问用户 28B 标量内存 | 任务书 §3.4 |

§3.3 五个标杆 case 与 910B4 真机实测（device 常驻标量，300 次 warmup，单次类 200 组/批量类 100 组采样取平均，默认日志级别，单位 μs）：

| case | a | b | 连续调用次数 | 标杆 Avg | 910B4 实测（最优轮 / 稳态区间） |
| --- | --- | --- | --- | --- | --- |
| 1 一般值单次 | (3,4) | (1,2) | 1 | 4.65 | 3.197 / 3.20~4.25 |
| 2 a=0 特殊分支 | (0,0) | (1,2) | 1 | 4.95 | 3.151 / 3.15~4.21 |
| 3 大数缩放路径 | (1e30,1e30) | (1e30,−1e30) | 1 | 4.66 | 3.146 / 3.15~4.22 |
| 4 批量连续调用 | (1.5,−2.5) | (0.5,0.25) | 1000 | 4.70 | 3.166 / 3.17~4.08 |
| 5 批量连续调用 | (1.5,−2.5) | (0.5,0.25) | 10000 | 4.20 | 3.192 / 3.19~4.21 |

测试工程设计：

```
test/rotg/crotg/
├── CMakeLists.txt                  // 注册 crotg_test（对齐 srotg 版）
├── crotg_param.h                   // CSV 复数 4 分量解析（"null"=空指针）、c/s 列、iters
├── crotg_golden.h                  // Netlib crotg C 移植 golden + aclblasCrotg_cpu
└── arch22/
    ├── crotg_test.cpp              // GTest 主体（device/host 双路径、数学性质、b 只读、6 个 TEST_F）
    ├── crotg_test.csv              // 1200 条用例（种子 20260823）
    └── crotg_npu_wrapper.h         // device 路径包装（服务独立 TEST_F）
```

- 每条**精度**用例：①device 路径 5 分量 vs golden；②host pinned 路径 5 分量 vs golden；③b 不变性位级比对（含 NaN payload）；④三条数学性质以原始 a、b 回代；
- **TC_PF 性能用例**：预置一次 a/b → 同流背靠背连发 iters 次 → 仅断言全部 SUCCESS（a 被原地覆写，迭代终值不对应初始输入，数值正确性由 iters=1 精度套件覆盖；不得在迭代间插入走内部流的阻塞 memcpy，否则构成跨流竞争）；
- **负向用例**：TC_ED 071~074 四条空指针经 CSV 表达；handle=nullptr、Host/Device 混合指针由独立 `TEST_F` 覆盖；
- 用例分布：TC_L0 12、TC_SP 18、TC_PH 16、TC_MAG 16、TC_FL 8、TC_ED 5、TC_EX 925、TC_PF 200，合计 1200；
- 自验命令：`bash build.sh --ops=crotg --soc=ascend910b4` 后，`./crotg_test --gtest_filter='-*TC_PF*'`（期望 2006 PASSED）、`./crotg_test --gtest_filter='*TC_PF*'`（期望 400 PASSED）；任务配套 `verify_accuracy.py`/`verify_performance.py` 亦可一键执行。

## 兼容性分析

新算子，不涉及对既有算子的兼容性改动，具体如下：

1. **接口兼容**：`aclblasCrotg` 为新增声明，加入 `include/cann_ops_blas.h`，不修改/删除任何既有接口，不破坏 ABI；参数序列与 `cublasCrotg` 一一对应，950PR 同名任务共用该声明，无产品私有平行 API；
2. **平台兼容**：仅新增 `blas/rotg/arch22/` 源文件，arch35（950）目录与其他产品线零改动；CMake 按 SOC 自动 GLOB 架构目录，构建脚本无需修改；
3. **数据布局兼容**：complex64 内存表示（`float real,imag` 交错）与仓内既有复数算子（cdotc/cgerc/caxpy 等）一致；
4. **精度兼容**：FLOAT32 标量运算，误差仅来自少量乘除与开方舍入；host 与 device 采用同构运算次序，正常实现远优于阈值，缩放路径以数学性质验证为主判据（任务书 §3.2 补充说明）。
