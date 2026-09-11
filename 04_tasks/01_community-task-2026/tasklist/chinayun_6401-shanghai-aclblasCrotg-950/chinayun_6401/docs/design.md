# aclblasCrotg 950PR 算子设计方案

## 1 需求背景（required）

### 1.1 需求来源

本任务来源于 CANN 开源社区 2026 年社区任务，目标是在昇腾 Atlas 950PR 上基于 Ascend C/CATLASS 编程方式实现单精度复数 Givens 旋转参数构造算子 `aclblasCrotg`，完成接口设计、Host/Device 实现、测试验证及开源仓合入。

算子功能与接口语义对标 cuBLAS `cublasCrotg`，数学语义参考 Netlib BLAS `crotg`。验收通过后，代码计划合入 `ops-blas` 仓库 `blas/rotg/arch35/`，测试代码合入 `test/rotg/crotg/arch35/`。

任务要求的接口声明统一放入 `include/cann_ops_blas.h`，不新增 Atlas 950PR 私有平行 API。

### 1.2 背景介绍

#### 1.2.1 aclblasCrotg 算子功能介绍

`aclblasCrotg` 用两个复数标量 `a`、`b` 构造 Givens 平面旋转参数 `c`、`s` 和旋转结果 `r`，使输入向量 `(a, b)^T` 经旋转后第二个分量消为 0：

```text
[  c          s ] [ a ] = [ r ]
[ -conjg(s)   c ] [ b ]   [ 0 ]
```

其中：

- `a`、`b` 为 `COMPLEX64` 标量；
- `c` 为 `FLOAT32` 实数；
- `s` 为 `COMPLEX64` 复数；
- `c^2 + |s|^2 = 1`；
- 输出参数 `a` 原地覆写为 `r`；
- `b` 为只读输入，不覆写；
- `c`、`s` 为独立输出。

标准数学定义为：

```text
c = |a| / sqrt(|a|^2 + |b|^2)
s = sgn(a) * conjg(b) / sqrt(|a|^2 + |b|^2)
r = sgn(a) * sqrt(|a|^2 + |b|^2)
sgn(a) = a / |a|，a = 0 时取 1
```

其中复数模为 `sqrt(Re(x)^2 + Im(x)^2)`，复数共轭为 `Re(x) - Im(x)*i`。

#### 1.2.2 特殊值语义

为保持与任务书规定的 Netlib `crotg` 语义一致，设计中对特殊输入显式分支处理：

> 说明：以下零值分支是本任务书明确规定的行为；对于 Inf/NaN 等非有限输入，不在设计文档中自行扩展新的语义，统一以 Netlib golden、任务测试口径以及最终工程实现为准。

| 输入条件 | `a` 输出 | `c` | `s` |
|---|---|---:|---|
| `b = 0` | `a` 原值 | `1` | `(0, 0)` |
| `a = 0` 且 `b != 0` | `|b|` | `0` | `conjg(b) / |b|` |
| `a = 0` 且 `b = 0` | `(0, 0)` | `1` | `(0, 0)` |

`a = 0` 的判断以复数的实部和虚部同时为 0 为准。

#### 1.2.3 当前工程基础

`ops-blas` 仓库已有同族实数算子 `aclblasSrotg` 的 arch35 实现，可作为 Host 参数校验、Host/Device 双路径判别、handle/stream 使用方式及工程目录组织的主要参考：

```text
blas/rotg/arch35/srotg_host.cpp
blas/rotg/arch35/srotg_kernel.cpp
```

本算子属于纯标量计算，不涉及向量长度、步长、维度或广播，因此不需要继承 `srotg` 的向量切分逻辑。

### 1.3 算子原型

| 名称 | 类别 | dtype | format | shape | 介绍 |
|---|---|---|---|---|---|
| handle | 输入 | - | - | - | BLAS 库上下文句柄，携带 stream |
| a | 输入/输出 | COMPLEX64 | scalar | [1] | 输入复数标量，计算后原地覆写为 `r` |
| b | 输入 | COMPLEX64 | scalar | [1] | 输入复数标量，只读 |
| c | 输出 | FLOAT32 | scalar | [1] | Givens 旋转余弦元素 |
| s | 输出 | COMPLEX64 | scalar | [1] | Givens 旋转正弦元素 |

接口声明：

```cpp
aclblasStatus_t aclblasCrotg(
    aclblasHandle_t handle,
    aclblasComplex* a,
    aclblasComplex* b,
    float* c,
    aclblasComplex* s);
```

说明：虽然 `b` 逻辑上只读，但为了与对标接口 `cublasCrotg` 的参数类型和参数序列保持一致，保留非 `const` 指针签名。

---

## 2 需求分析

### 2.1 外部组件依赖

本算子无新增外部运行时组件依赖。开发和验证依赖于任务规定的基础环境：

- Ascend 950PR；
- CANN 9.1.0；
- ops-blas 工程及其现有公共头文件、handle/stream 基础设施；
- Netlib BLAS `crotg` 参考实现，用于生成精度 golden。

### 2.2 内部适配模块

算子采用 ops-blas 既有 BLAS handle 接口模式，通过 handle 绑定 stream：

```text
用户调用
  |
  v
aclblasCrotg(handle, a, b, c, s)
  |
  +-- 参数校验
  |
  +-- Host / Device 内存路径判定
  |      |
  |      +-- Host：CPU 标量计算
  |      |
  |      +-- Device：Ascend C SIMT kernel
  |
  +-- 返回 aclblasStatus_t
```

本任务不引入 Python / ATen / PyTorch 适配层，属于 ops-blas 的 `aclblas` 直调接口。

### 2.3 相关约束

1. `handle` 为 `nullptr` 时返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`。
2. `a`、`b`、`c`、`s` 任一为 `nullptr` 时返回 `ACLBLAS_STATUS_INVALID_VALUE`。
3. 四个数据指针必须全部位于 Host 或全部位于 Device；混合 Host/Device 指针返回 `ACLBLAS_STATUS_INVALID_VALUE`。
4. `a` 必须原地更新为 `r`；`b` 不得被修改。
5. 算子为纯标量运算，不支持广播、非连续 Tensor、动态 shape 或额外维度属性。
6. Device 路径依赖 `aclblasSetStream` 绑定的 stream 异步执行；Host 读取 Device 结果前由调用方保证 stream 已同步。
7. 实现禁止直接计算 `|a|^2 + |b|^2` 后再开平方，必须采用安全缩放，覆盖大数量级和次正规数场景。

---

## 3 需求详细设计

### 3.1 使能方式

| 上层框架 | 涉及的框架勾选 |
|---|---|
| TF训练/推理 | |
| PyTorch训练/推理 | |
| ATC推理 | |
| Aclnn直调 | |
| OPAT调优 | |
| SGAT子图切分 | |
| **ops-blas aclblas 直调** | **√** |

本算子不提供新的高阶框架封装，直接通过 `aclblasCrotg` 完成调用。

### 3.2 需求总体设计

#### 3.2.1 整体流程

整体采用 Host 侧统一入口 + Host/Device 双路径：

```text
                 +----------------------+
                 | aclblasCrotg         |
                 | 参数校验             |
                 +----------+-----------+
                            |
                   判断 a/b/c/s 所在内存
                            |
               +------------+------------+
               |                         |
          全部 Host                  全部 Device
               |                         |
               v                         v
       CPU scalar crotg             Launch kernel
               |                         |
       a/c/s 直接写回            GM -> Reg -> GM
               |                         |
               +------------+------------+
                            |
                          返回
```

Device kernel 内部流程进一步划分为：

```text
Global Memory Load
      |
      v
特殊分支判断
      |
      +-- b == 0 ----------> c=1, s=0, r=a
      |
      +-- a == 0 ----------> c=0, s=conjg(b)/|b|, r=|b|
      |
      +-- 一般路径 ---------> 安全缩放求模长
                              |
                              v
                         计算 c / s / r
                              |
                              v
                       Global Memory Store
```

由于每次调用只有 4 个标量数据指针，单次 kernel 的计算量很小，主要性能瓶颈预计来自固定 kernel launch、调度和同步相关开销，而非算术吞吐。因此设计目标是保持单次调用仅启动一个轻量 kernel，并避免额外临时 Tensor、workspace 或多 kernel 拼接。

#### 3.2.2 Host 侧设计

Host 侧负责参数合法性校验、内存位置一致性判断以及 Host/Device 路径分发，具体设计如下。

**1）句柄检查**

```text
if (handle == nullptr)
    return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
```

**2）指针检查**

```text
if (a == nullptr || b == nullptr || c == nullptr || s == nullptr)
    return ACLBLAS_STATUS_INVALID_VALUE;
```

**3）Host/Device 内存路径判定**

检查 `a`、`b`、`c`、`s` 四个指针的内存属性：

- 全部为 Host：走 CPU 直接计算；
- 全部为 Device：获取当前 handle 对应 stream，启动 Ascend C kernel；
- Host/Device 混合：直接返回 `ACLBLAS_STATUS_INVALID_VALUE`。

路径判定口径与仓内同族 `aclblasSrotg` 保持一致。

**4）Host 路径**

Host 路径不启动 NPU kernel，直接在 CPU 上按与 Device 路径一致的数学流程完成计算：

```text
读取原始 a、b
  -> 特殊值分支
  -> 安全缩放求 norm
  -> 计算 c、s、r
  -> a = r
  -> 写 c、s
```

Host 与 Device 必须共享同一套语义和边界规则，尤其是 `a=0`、`b=0`、极大值、极小值以及非有限值处理。

**5）Device 路径**

Device 路径通过 handle 中绑定的 stream 启动单个 `crotg` kernel。由于输入输出均为标量，不进行 tiling，也不申请 workspace。

建议 Host 侧 kernel launch 前只准备最小参数集合：

```text
kernel(a, b, c, s)
```

`b` 在 kernel 中仅执行读取，不产生任何写操作。

#### 3.2.3 Kernel 侧设计

Kernel 使用 Ascend C SIMT 直调方式实现。由于输入规模固定为 1 个复数标量对，不设计多核分片、Tile 搬运和动态 tiling 参数。

**核心计算步骤：**

1. 从 Global Memory 读取 `a`、`b`；
2. 对 `a`、`b` 做零值与特殊条件判断；
3. 一般路径通过安全缩放计算 `|a|`、`|b|` 对应的组合模长；
4. 根据组合模长计算 `c`；
5. 根据 `sgn(a)` 和 `conjg(b)` 计算 `s`；
6. 计算 `r` 并覆盖写回 `a`；
7. 写回 `c`、`s`。

单个 kernel 的数据流如下：

```text
GM(a,b)
   |
   v
Register/Scalar
   |
   +--> classify zero/special
   |
   +--> safe norm calculation
   |
   +--> c calculation
   |
   +--> s calculation
   |
   +--> r calculation
   |
   +--> GM(a,c,s)
```

**Kernel 执行边界**

- 仅处理一个逻辑元素组 `(a,b)`；
- 不依赖 shape/stride/axis；
- 不进行跨元素通信；
- 不需要 Atomic、Reduce 或 workspace；
- 一个 kernel 完成一次 `aclblasCrotg` 调用。

#### 3.2.4 安全缩放算法设计

本算子必须遵循任务书要求的 Netlib `crotg` 安全缩放思路，核心目标是避免在单精度复数输入下直接计算

```text
|a|^2 + |b|^2
```

造成溢出或下溢。设计上不把“简单 max 缩放”作为完整算法定义，而是按 Netlib `crotg` 的安全范围分支组织计算：

```text
1. 先处理 b == 0、a == 0 的显式特殊分支；
2. 对一般输入根据 |a|、|b| 所处数量级判断是否需要缩放；
3. 在安全浮点范围内计算 f、g 及其组合范数；
4. 当原始数量级过大或过小时，通过 safmin/safmax 相关缩放避免
   中间乘法、平方和、开方过程溢出/下溢；
5. 最终恢复 r 的尺度，并计算 c、s。
```

为便于工程实现，定义复数模的稳定计算：

```text
abs(x) = hypot(Re(x), Im(x))
```

其中 `hypot` 逻辑本身必须避免直接执行 `sqrt(re*re + im*im)`，而应在内部采用安全比例计算。例如当 `x1 = max(|re|, |im|)`、`x2 = min(|re|, |im|)` 时：

```text
x1 == 0        -> abs(x) = 0
x1 > 0         -> abs(x) = x1 * sqrt(1 + (x2/x1)^2)
```

对于 `crotg` 主流程，设计原则如下：

```text
b == 0
    -> c = 1, s = 0, r = a

a == 0 且 b != 0
    -> c = 0, s = conjg(b)/|b|, r = |b|

a != 0 且 b != 0
    -> 按 Netlib 安全缩放分支计算
       scale / norm / c / s / r
```

对于一般路径，数学关系保持：

```text
c = |a| / norm
s = (a / |a|) * conjg(b) / norm
r = (a / |a|) * norm
norm = sqrt(|a|^2 + |b|^2)
```

但实现过程中不得先形成未缩放的 `|a|^2 + |b|^2`。当输入处于极大或极小数量级时，应优先将参与计算的量缩放到安全范围，在缩放域完成平方和、开方和比例运算，再恢复 `r` 的实际尺度。

特别地，`r` 的构造应保留 `a` 的相位信息，因此不能简单令 `r = norm`；当 `a != 0` 时必须按照 `sgn(a) = a/|a|` 恢复复数符号/相位。

该设计与任务书中“参照 Netlib crotg 的 safmin/safmax 缩放算法，禁止直接计算 `|a|² + |b|²` 而不做缩放”的要求一致。具体常数及分支阈值以最终工程采用的 Netlib 等价实现为准。

#### 3.2.5 数学性质保持

实现除逐输出与 Netlib golden 对比外，还需保证以下数学关系：

**正交归一性**

```text
c^2 + |s|^2 = 1
```

**旋转关系**

```text
c * a + s * b = r
-conjg(s) * a + c * b = 0
```

**模长保持**

```text
|r| = sqrt(|a|^2 + |b|^2)
```

测试时使用原始输入回代计算，避免使用已被覆写的 `a` 破坏验证条件。

#### 3.2.6 Host/Device 一致性设计

Host 与 Device 两条路径必须遵循完全一致的输入输出语义：

| 项目 | Host | Device |
|---|---|---|
| `a` | 覆写为 `r` | 覆写为 `r` |
| `b` | 只读 | 只读 |
| `c` | 输出 float | 输出 float |
| `s` | 输出 complex64 | 输出 complex64 |
| `b=0` | `c=1,s=0,r=a` | 同语义 |
| `a=0,b!=0` | `c=0,s=conjg(b)/|b|,r=|b|` | 同语义 |
| 大数/小数 | 安全缩放 | 安全缩放 |

该一致性是 Host/Device 路径验收的基本要求。

#### 3.2.7 错误码设计

错误码直接复用 `cann_ops_blas_common.h` 中已有 `aclblasStatus_t` 定义，不新增 Crotg 私有错误码。

| 条件 | 返回值 |
|---|---|
| `handle == nullptr` | `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| `a/b/c/s` 任一为空 | `ACLBLAS_STATUS_INVALID_VALUE` |
| Host/Device 指针混用 | `ACLBLAS_STATUS_INVALID_VALUE` |
| 参数合法且执行成功 | `ACLBLAS_STATUS_SUCCESS` |

### 3.3 支持硬件

| 支持的芯片版本 | 是否支持 |
|---|---|
| Atlas 950PR | √ |

CANN 版本：9.1.0。

代码实现目录：

```text
ops-blas/
├── include/
│   └── cann_ops_blas.h
└── blas/
    └── rotg/
        └── arch35/
            ├── crotg_host.cpp
            └── crotg_kernel.cpp
```

测试代码目录：

```text
test/
└── rotg/
    └── crotg/
        └── arch35/
            ├── README.md
            ├── CSV 测试用例
            └── GTest/C++ 测试代码
```

### 3.4 算子约束限制

1. 仅支持 `COMPLEX64` 输入/输出及 `FLOAT32` 的 `c`。
2. 所有数据对象均为单元素标量，不存在 shape、axis、stride 或 broadcast。
3. `a` 为原地输出；`b` 不修改。
4. `a`、`b`、`c`、`s` 必须全部位于同一内存侧。
5. Device 路径不使用额外 workspace。
6. 不需要 tiling 参数，不进行数据量动态切分。
7. 不要求确定性以外的额外并行约束；一次调用只有一个逻辑计算实例。
8. 输入可覆盖任务规定的大数、小数/次正规数、Inf/NaN 等边界场景，具体非有限值行为以对标 Netlib/cublas 及测试 golden 的既定口径为准，不能在实现中自行引入额外语义。
9. 不定义 950PR 私有 API，统一通过 `include/cann_ops_blas.h` 对外声明。

---

## 4 特性交叉分析

本算子为 BLAS 级纯标量旋转参数构造算子，不涉及 Tensor 维度变换和广播，因此功能交叉面较小，主要关注以下方面：

| 特性 | 是否涉及 | 设计说明 |
|---|---|---|
| 标量输入输出 | √ | 4 个数据指针均为单元素 |
| 原地计算 | √ | `a` 原地覆写为 `r` |
| 广播 | × | 无广播语义 |
| 非连续 Tensor | × | 不存在 Tensor layout 问题 |
| 动态 shape | × | 无 shape 参数 |
| Reduce/Scan | × | 不涉及跨元素归约 |
| Workspace | × | 不需要临时 workspace |
| 多核切分 | × | 单个逻辑实例，无数据分片 |
| 异步 stream | √ | Device 路径绑定 handle stream |
| Host/Device 双路径 | √ | Host CPU 计算、Device kernel 计算 |
| 特殊值 | √ | zero、极端数量级、Inf/NaN 需覆盖 |

核心兼容风险集中在 **复数运算顺序、特殊分支以及极端数量级下的浮点舍入**。因此精度验证除逐输出比较外，增加 Givens 旋转数学性质验证。

---

## 5 可维可测分析

### 5.1 精度标准/性能标准

#### 5.1.1 精度标准

golden 由 Netlib BLAS `crotg` 参考实现生成；由于标准 CBLAS 接口集不提供复数 `rotg`，本算子不使用 CBLAS 复数接口作为 golden。

按照任务规定，`COMPLEX64` 的实部和虚部分别按 `FLOAT32` 误差标准比较：

| 数据类型 | rtol | atol | required_matched_ratio | max_abs_error_limit |
|---|---:|---:|---:|---:|
| COMPLEX64（实部/虚部分量）/ FLOAT32 | `2^-10` ≈ `9.77e-4` | `2^-16` ≈ `1.53e-5` | `0.99` | `1e-2` 或 `32 * ULP` |

逐元素通过条件：

```text
|actual - golden| <= atol + rtol * |golden|
```

同时验证：

```text
|c^2 + |s|^2 - 1| <= atol + rtol

|c*a + s*b - r|
    <= atol + rtol * sqrt(|a|^2 + |b|^2)

|-conjg(s)*a + c*b|
    <= atol + rtol * sqrt(|a|^2 + |b|^2)

||r| - sqrt(|a|^2 + |b|^2)|
    <= atol + rtol * sqrt(|a|^2 + |b|^2)
```

其中数学性质验证统一基于**原始输入**计算，不能使用已经覆写后的 `a` 作为输入。

#### 5.1.2 性能标准

性能设备为 Atlas 950PR。测试步骤要求先 warmup，再有效采样大于 50 次并取平均值。性能指标为 `COMPLEX64` 输入下单次平均耗时，单位 `us`。

由于本算子为单标量算子，性能主要受 kernel launch、调度以及固定调用开销影响，因此测试程序必须直接使用高精度计时结果（至少能够区分微秒级差异），不得将耗时先截断为整数毫秒后再参与判断。批量 case 的统计口径为总耗时除以连续调用次数。


目标如下：

| Case | 场景 | a | b | 连续调用次数 | 标杆 Avg time/us |
|---|---|---|---|---:|---:|
| 1 | 单次调用延迟（一般值） | `(3.0, 4.0)` | `(1.0, 2.0)` | 1 | 11.85 |
| 2 | 单次调用延迟（a=0） | `(0.0, 0.0)` | `(1.0, 2.0)` | 1 | 9.97 |
| 3 | 单次调用延迟（大数缩放） | `(1e30, 1e30)` | `(1e30, -1e30)` | 1 | 9.49 |
| 4 | 批量连续调用平均延迟 | `(1.5, -2.5)` | `(0.5, 0.25)` | 1000 | 11.84 |
| 5 | 批量连续调用平均延迟 | `(1.5, -2.5)` | `(0.5, 0.25)` | 10000 | 11.53 |

性能设计原则：

1. Device 一次调用只启动一个 kernel；
2. 不引入额外 workspace 和中间 Tensor；
3. 不设计不必要的 GM 往返；
4. 将分支和计算全部控制在单次 kernel 内；
5. Host 路径直接 CPU 执行，避免小标量算子通过 NPU 产生额外 launch 开销。

### 5.2 兼容性分析

本节只描述任务书已经明确的接口和行为约束；涉及具体仓内 helper、kernel launch API、CMake 组织方式的部分，需要在集成 `ops-blas` 实际源码时与 `aclblasSrotg` 同步保持一致，不在设计文档中虚构新的工程接口。


本算子为新增 `aclblasCrotg` 接口，兼容性主要体现在以下几个层面：

**接口兼容性**

- 函数名、参数顺序、参数 dtype 与 `cublasCrotg` 对齐；
- `handle` 参数位于首位；
- `b` 虽为只读行为，但保留非 `const` 以匹配对标函数签名；
- 对外声明统一放入 `include/cann_ops_blas.h`。

**语义兼容性**

- `a` 覆写为 `r`；
- `b` 不覆写；
- `c` 为 float；
- `s` 为 complex64；
- 特殊分支与任务规定的 Netlib `crotg` 语义对齐。

**数值兼容性**

- 一般数据按 Netlib golden 进行逐分量比较；
- 大数、小数、次正规数依赖安全缩放保持稳定性；
- 数学性质作为极端数量级用例的重要判据。

**工程兼容性**

- 实现放入 `blas/rotg/arch35/`；
- 测试放入 `test/rotg/crotg/arch35/`；
- README 产品支持表新增 `aclblasCrotg`，并标注 Atlas 950PR 支持；
- 不修改或覆盖 `aclblasSrotg` 既有行为。

### 5.3 测试用例设计

测试框架参照仓内 `srotg` 的 CSV 驱动 GTest 模式，新建 `test/rotg/crotg/arch35/`。

#### 5.3.1 功能精度用例

| 类别 | 覆盖内容 |
|---|---|
| 基础正常值 | 随机复数 `a/b`、正负实部/虚部 |
| 均匀分布 | 实部/虚部独立均匀采样 `[-5,5]` |
| 正态分布 | `mu∈[-5,5]`、`sigma∈[0.1,2]` |
| 零值 | `a=0`、`b=0`、`a=b=0` |
| 纯实 | 实部非零、虚部为 0 |
| 纯虚 | 实部为 0、虚部非零 |
| 数量级差异 | `|a| >> |b|`、`|a| << |b|` |
| 大数 | `1e30` 量级 |
| 小数/次正规 | `1e-38` 量级 |
| 非有限值 | `Inf/NaN` 场景 |
| 相位组合 | 不同实部/虚部符号与相位组合 |

#### 5.3.2 接口负向用例

| 用例 | 期望结果 |
|---|---|
| `handle == nullptr` | `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| `a == nullptr` | `ACLBLAS_STATUS_INVALID_VALUE` |
| `b == nullptr` | `ACLBLAS_STATUS_INVALID_VALUE` |
| `c == nullptr` | `ACLBLAS_STATUS_INVALID_VALUE` |
| `s == nullptr` | `ACLBLAS_STATUS_INVALID_VALUE` |
| Host/Device 混合指针 | `ACLBLAS_STATUS_INVALID_VALUE` |

由于本算子没有 shape、stride、axis 等参数，任务书中通用的“零维、空 Tensor、非法步长、负维度”等场景不映射到本接口。测试中应避免人为构造不存在的参数字段，同时覆盖与本接口实际相关的指针和内存侧异常。

#### 5.3.3 Host/Device 一致性测试

至少覆盖：

1. 相同正常输入，Host/Device 输出一致；
2. `a=0` 特殊分支，Host/Device 输出一致；
3. `b=0` 特殊分支，Host/Device 输出一致；
4. 大数缩放路径，Host/Device 输出一致；
5. 小数/次正规路径，Host/Device 输出一致。

#### 5.3.4 性能测试

严格按照任务书给定的 5 个性能 case 执行：

1. warmup 后执行；
2. 有效采样次数 > 50；
3. 单次调用 case 分别测一般值、`a=0`、大数缩放；
4. 连续调用 case 分别执行 1000 次和 10000 次；
5. 统计平均单次耗时并与标杆比较；
6. 保存性能日志及截图，形成自测报告。

#### 5.3.5 测试工程结构

建议最终测试目录结构：

```text
test/rotg/crotg/arch35/
├── README.md
├── CMakeLists.txt / build 配置
├── *.csv
├── test_crotg.cpp
└── golden/            # 如测试工程需要保存离线 golden
```

README 需要说明环境准备、编译、运行命令、CSV 字段、golden 生成方式以及精度/性能结果判定方法，保证验收人员可以复现。

---

## 6 交付与合入设计

### 6.1 算子代码

算子实现放置：

```text
blas/rotg/arch35/
```

至少包括 Host 入口实现和 Device kernel 实现，并完成 `include/cann_ops_blas.h` 接口声明。

### 6.2 测试代码

测试代码放置：

```text
test/rotg/crotg/arch35/
```

测试工程需覆盖任务提供的全部自测用例，并补充任务要求的边界、负向、特殊值、Host/Device 路径以及性能验证。

### 6.3 产品支持信息

`blas/rotg/README.md` 产品支持表新增：

```text
aclblasCrotg    Atlas 950PR    支持
```

### 6.4 自测报告

自测报告需要记录：

- 测试环境：Ascend 950PR、CANN 9.1.0；
- 精度 case 输入参数；
- `r`、`c`、`s` 的实部/虚部精度结果；
- 数学性质验证结果；
- 性能 case 平均耗时及截图；
- 内存占用数据；
- 测试代码和 README 的复现步骤。

---

## 7 风险分析与规避措施

| 风险 | 影响 | 规避措施 |
|---|---|---|
| 直接计算平方和导致溢出 | 大数 case 精度失败 | 全程采用安全缩放，禁止未缩放平方和主路径 |
| 次正规数下溢 | 小数 case 精度失败 | 先归一化再计算平方和，再恢复尺度 |
| `a=0`/`b=0` 分支遗漏 | 特殊 case 失败 | Host/Device 明确独立分支并建立专门测试 |
| `b` 被错误覆写 | 接口语义不兼容 | kernel 仅读取 `b`，测试前后对比 `b` 内容 |
| Host/Device 路径语义不一致 | 集成测试不稳定 | 两路径共享同一公式和特殊值规则 |
| 标量 kernel 启动开销过高 | 性能 case 失败 | 单 kernel、无 workspace、无额外临时计算 |
| 误用 srotg 的实数规则 | 结果与 crotg 不一致 | 严格按复数 `crotg` 定义实现，不引入实数 `z` 恢复规则 |
| 指针内存侧混用 | 非法访问/行为未定义 | Host 入口统一检查四个数据指针内存属性 |

---

## 8 参考资料

1. 社区任务书：`aclblasCrotg_Atlas950PR_task_doc.md`
2. 社区任务设计文档模板：`https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md`
3. ops-blas 开源仓：`https://gitcode.com/cann/ops-blas`
4. Ascend C 算子开发文档：`https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/opdevg/Ascendcopdevg/atlas_ascendc_map_10_0002.html`
5. Ascend C API 文档：`https://www.hiascend.com/document/detail/zh/canncommercial/850/API/ascendcopapi/atlasascendc_api_07_0003.html`
6. 生态算子开源精度标准：`https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md`
7. cuBLAS `cublasCrotg`：`https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-rotg`
8. Netlib BLAS `crotg`：`https://www.netlib.org/blas/crotg.f`

