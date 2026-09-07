# 需求背景（required）

## 需求来源

本需求来自 CANN 2026 年 8 月社区任务第 43 号“aclblasCsyr2 算子开发（A2/A3）”。目标是在昇腾 NPU 的 Atlas A2/A3 系列产品上，基于 `cann/ops-blas` 工程和 Ascend C Kernel 直调模式，实现单精度复数对称秩 2 更新接口 `aclblasCsyr2`。

接口的参数序列和核心行为对齐 cuBLAS `cublasCsyr2`；三角引用、负步长起点和 quick return 语义参考 Netlib `ssyr2`，并按复数普通乘法推广。开发完成后，实现代码归档到 `blas/syr2/arch22/`，测试代码归档到 `test/syr2/csyr2/`，公共接口声明新增到 `include/cann_ops_blas.h`。

## 背景介绍

### aclblasCsyr2 算子功能

`aclblasCsyr2` 属于 BLAS Level-2，对列主序复数对称矩阵执行原地秩 2 更新：

```text
A := alpha * x * y^T + alpha * y * x^T + A
```

其中：

- `alpha` 为单精度复数标量；
- `x`、`y` 为逻辑长度为 `n` 的单精度复数向量；
- `A` 为 `n × n` 单精度复数对称矩阵，按列主序存储；
- `uplo` 决定只引用和更新 `A` 的上三角或下三角；另一侧三角不读不写；
- 对称语义是普通转置 `A = A^T`，不是共轭转置，对角元素虚部按普通复数乘加更新。

函数原型如下：

```cpp
aclblasStatus_t aclblasCsyr2(
    aclblasHandle_t handle,
    aclblasFillMode_t uplo,
    int n,
    const aclblasComplex* alpha,
    const aclblasComplex* x, int incx,
    const aclblasComplex* y, int incy,
    aclblasComplex* A, int lda);
```

### 现有基础与差异

`ops-blas` 已有实数同族接口 `aclblasSsyr2`，可复用句柄、stream、错误码、Host 直调、原子直写策略与工程组织方式；但不能直接复用实数计算 Kernel，原因如下：

1. `aclblasCsyr2` 的每个元素包含实部、虚部，需要展开普通复数乘加；
2. Netlib/cblas 没有复数 symmetric syr2，测试 golden 需要按 `ssyr2` 的三角、步长和列主序语义自实现；
3. 性能用例的矩阵规模较大（`n` 至 4096，三角写流量约 32 MB），纯逐元素更新难以达到任务书门槛，需要利用秩 2 更新的低秩结构（列系数预乘 + 旋转孪生 4-Axpy）与原子直写，把 `A` 的 GM 流量降为一次 store；
4. 仍需保留覆盖负步长、非紧凑 `lda`、非对齐尺寸和任意 `alpha` 的通用路径。

工程落点如下：

| 项目 | 设计落点 |
| --- | --- |
| 公共声明 | `include/cann_ops_blas.h` 新增 `aclblasCsyr2`，不定义产品私有平行 API |
| Host/Kernel | `blas/syr2/arch22/`，`csyr2_host.cpp` + `csyr2_kernel.cpp` 两文件布局（与 ssyr2 一致） |
| 测试 | `test/syr2/csyr2/`，CSV 驱动 GTest（param/golden + `arch22/` 测试与用例） |
| 数据类型 | `aclblasComplex`，实部和虚部均为 FLOAT32 |
| 目标产品 | Atlas A2/A3；性能验收设备为 Atlas 800I A2（Ascend 910B3） |
| CANN | 9.1.0 |

# 需求分析（required）

## 需求描述

使用 Ascend C 实现 `aclblasCsyr2`，支持 UPPER/LOWER、正负步长、`lda` padding、参数异常处理和 quick return；在指定三角内完成普通复数对称秩 2 更新，未指定三角保持不变。实现需满足任务书精度标准，并在 910B3 上按任务书 §3.3 三条性能门槛验收。

## 需求拆解

1. 新增公共 `aclblasCsyr2` 声明，参数顺序与 cuBLAS 对标接口一致。
2. 支持 `ACLBLAS_UPPER` 和 `ACLBLAS_LOWER`，只读写指定三角。
3. 支持 COMPLEX64，采用普通复数乘法，不执行共轭。
4. 支持 `incx`、`incy` 的正负步长，拒绝零步长。
5. 支持 `lda >= max(1, n)`，包括列尾 padding。
6. `n = 0` 或 `alpha = (0, 0)` 时合法 quick return，不启动 Kernel、不访问 Device 数据。
7. 参数错误返回任务书指定状态码；Host 不隐式同步 stream。
8. 提供覆盖全形状的通用回退路径和满足对齐/驻留条件时的原子直写快路径，路径选择只依赖对齐、步长、工作量与 UB 容量类谓词，不按某个精确 shape 写专用分支。
9. 使用任务书提供的官方 CSV/GTest 自验；官方工具保持原样，性能判定（`PERF_VERDICT`）内建于仓内测试交付物。
10. 精度按 matched-ratio/max-abs 规则验收；性能按任务书“warmup 后 >50 次有效采样取平均”验收，`msprof` 用于归因。

# 详细设计（required）

## 算子分析

### 数学公式

对指定三角中的元素 `(i, j)`：

```text
A(i, j) := A(i, j)
           + alpha * x(i) * y(j)
           + alpha * y(i) * x(j)
```

复数乘法展开为：

```text
(a + b*i) * (c + d*i) = (a*c - b*d) + (a*d + b*c)*i
```

列主序地址为：

```text
A(i, j) -> A[i + j * lda]
```

三角范围为：

- UPPER：`0 <= i <= j < n`；
- LOWER：`0 <= j <= i < n`。

负步长采用 Netlib 起点规则。0-based 下：

```text
xStart = 0                    , incx > 0
xStart = (n - 1) * (-incx)    , incx < 0
```

`yStart` 同理，逻辑元素 `i` 的物理地址为 `start + i * inc`。kernel 经统一的 `Csyr2VecIndex` 换算寻址。

kernel 以列为单位组织：对列 `j` 预计算 `a = alpha * y(j)`、`b = alpha * x(j)`，该列更新化为 `A(:, j) += a*x + b*y` 的复数 Axpy。

### 支持数据类型

| 操作数 | 数据类型 | 说明 |
| --- | --- | --- |
| `alpha` | `aclblasComplex` | Host 侧 complex64 标量 |
| `x`、`y` | `aclblasComplex` | Device 侧 complex64 向量 |
| `A` | `aclblasComplex` | Device 侧 complex64 矩阵，原地更新 |
| `n`、`incx`、`incy`、`lda` | `int` | Host 侧维数和步长 |
| `uplo` | `aclblasFillMode_t` | 上/下三角枚举 |

### 支持形状

| 对象 | 逻辑形状 | 物理约束 |
| --- | --- | --- |
| `x` | `[n]` | 至少 `1 + (n-1)*abs(incx)` 个 complex64 元素 |
| `y` | `[n]` | 至少 `1 + (n-1)*abs(incy)` 个 complex64 元素 |
| `A` | `[n, n]` | 列主序，至少 `lda*n` 个 complex64 元素 |

`n` 为运行时参数；不涉及广播和三维以上高维 shape。非连续访问由 `incx`、`incy` 和 `lda` 表达。

## 算子实现

### 实现方案

实现为单个 AIV Kernel：Host 负责参数校验、quick return、核数计算与启动；Kernel 按几何条件二选一执行“原子直写快路径”或“通用RMW回退路径”。路径条件只依赖步长、对齐、`n` 与UB容量，无精确shape特化。

#### Host 侧（csyr2_host.cpp）

1. 入口 `aclblasCsyr2` 按任务书顺序校验：`handle` 空返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`；非法 `uplo` 返回 `ACLBLAS_STATUS_INVALID_ENUM`；`n < 0`、`alpha` 空、`incx/incy` 为 0 或 `INT_MIN`、`lda < max(1,n)` 返回 `ACLBLAS_STATUS_INVALID_VALUE`。
2. `n == 0` 或 `alpha == (0,0)` 直接返回成功，不启动 Kernel、不触碰 Device 指针；之后才检查 `x/y/A` 非空。
3. `useCoreNum = min(n, GetAivCoreCount())`，核数动态获取；`16 <= n <= 1024` 且单位步长、`n/lda` 按 4 对齐时钳到 32。
4. gather 偏移表（10 KB 常量，旋转孪生 swap 段 + LOWER sliver 压缩段）以 `std::call_once` 进程内一次初始化，H2D 拷贝排入调用方 stream。
5. tiling（`n/lda/uplo/useCoreNum/incx/incy/alpha` 实虚部）以标量 kernel 参数直接下发，在 handle 绑定的 stream 上异步启动 Kernel；Host 不做同步，无 workspace。

#### Kernel 侧（csyr2_kernel.cpp）

1. 路径判定：`resident = n <= 4096`，`fastGeo = resident && incx == incy == 1 && lda % 4 == 0 && n % 4 == 0`；`fastGeo` 且 `n >= colsPerPanel` 走快路径，其余走回退路径。
2. UB 布局：`[gather表 | sliver/compact 缓冲 | x、Vx、y、Vy 四向量 | 2~3 个轮转面板缓冲]`；面板几何（列数/块长/缓冲数）按 192 KB UB从宽到窄搜索，最坏布局有编译期 `static_assert` 兜底。
3. 工作分配：三角列组成面板，按最长列长度降序蛇形（偶轮正序、奇轮逆序）发到各核；面板列唯一归属，无多核写冲突，没领到面板的核提前退出。
4. prologue：载入 gather 表；驻留时把 `x/y` 交错载入 UB，经一次 swap-gather 加掩码取负构建旋转孪生 `V = i*v`（`V[2k] = -imag`、`V[2k+1] = real`），并清零向量尾部 padding。
5. 列更新：预计算列系数 `a = alpha*y[j]`、`b = alpha*x[j]`，复数更新 `A(:,j) += a*x + b*y` 展开为 `x/Vx/y/Vy` 四个实向量上的 4 个向量 op（首项 Muls 覆写、后续 3 个 Axpy；norm 模式整窗 burst，组掩码一次设定），直接作用于 interleaved 的 A 数据，无拆分/重交织。
6. 快路径写回：每块增量构造完先掩码清零三角参差（LOWER 对角带折入块0、清对角以上前缀；UPPER清store窗口内每列无效后缀），再单条2D DataCopy配合 `SetAtomicAdd` 直写 GM——A 只写不读，GM 流量较读改写减半；越出三角的 lane 恒为 +0.0，原子加 0.0 不改位模式，另一三角与 padding 不被污染。
7. 回退路径：逐列 1D 拷贝把 A 读进 UB，同样4个向量op后普通覆写写回（RMW）；LOWER 对角 sliver 单独走“逐列加载→Axpy→gather 压缩→按精确长度 store”；`n > 4096` 或带步长时向量不驻留 UB，每块按 `Csyr2VecIndex`（Netlib 负步长规则）以标量读重建行段；未指定三角与 `lda` padding 不读不写。
8. 流水与同步：面板缓冲 ping-pong/三缓冲轮转，块间 load、compute、store 经屏障重叠；跨管依赖一律 `PIPE_BARRIER(ALL)`，零事件旗标（C220 上 MTE2→V 与 MTE3→V 旗标通道并存有挂死概率）。

### Tiling 通用性

Tiling 的输入是：数据布局、对齐、步长、`n`、AIV 核数与 UB 容量。以下做法明确避免：

- `if (n == 1024)`、`if (n == 2048)` 等精确规模分支；
- 为单个验收 case 写固定 core map；
- 依赖未声明的输入值分布决定正确性路径。

性能用例触发的快路径条件（单位步长、4 对齐、`n <= 4096` 驻留上限）是同一套 size-class 规则，其它满足相同条件的 shape 同样复用该路径；不满足时自动落入通用回退路径，功能不受限。

## 支持硬件

| 芯片版本 | 支持情况 | 验证计划 |
| --- | --- | --- |
| Atlas 800I A2（Ascend 910B4） | 支持 | CANN 9.1.0；功能、精度及任务书性能门槛正式验证（实测设备） |
| Atlas A2（arch22） | 支持 | 使用同一 API/Kernel 源码完成构建、功能与边界验证，并重新评估资源参数 |

内核以 `__DAV_C220_VEC__` 编译守卫，核数 `GetAivCoreCount()` 动态获取。

## 算子约束限制

1. 仅支持 COMPLEX64，不支持混合精度。
2. `A` 为列主序方阵，只更新 `uplo` 指定三角。
3. `n >= 0`，`incx != 0`，`incy != 0`，`lda >= max(1,n)`。
4. `incx`、`incy` 支持负数，但不支持 0；实现显式拒绝 `INT_MIN`，避免取绝对值溢出。
5. `A` 原地更新，调用方负责输入输出内存合法性及 stream 生命周期。
6. `n == 0` 或 `alpha == (0,0)` 为合法 no-op。
7. 无 workspace；快路径几何条件（`n <= 4096`、单位步长、`lda % 4 == 0`、`n % 4 == 0`）不满足时自动回退通用 AIV 路径，功能不受限、性能下降。

# 可维可测分析

## 精度标准/性能标准

### 精度标准

golden 按 Netlib `ssyr2` 的列主序、三角和步长语义自实现复数版本。对指定三角的实部、虚部分别按 FLOAT32 判定：

| 指标 | 标准 |
| --- | --- |
| `rtol` | `2^-10` |
| `atol` | `2^-16` |
| required matched ratio | `>= 0.99` |
| max abs error | `<= 1e-2` 或 `<= 32 ULP` |

逐元素条件为：

```text
abs(actual - golden) <= atol + rtol * abs(golden)
```

同时检查：

- 未指定三角保持原值（逐位一致，污染检查）；
- `lda` padding 不被修改；
- UPPER/LOWER、正负步长、quick return、非法参数和空指针状态码；
- Inf/NaN 按任务测试规则处理（两侧在相同元素产生相同 inf/nan 即逐位相等放行）。

### 性能标准

性能验收设备为 Atlas 800I A2（Ascend 910B4），CANN 9.1.0。任务书 §3.3 门槛与实测（10 次预热 + 100 次计时取平均，满足“>50 次有效采样”）：

| case | uplo | n | alpha | incx/incy | lda | 门槛 |
| --- | --- | ---: | --- | --- | ---: | ---: |
| 1 | UPPER | 512 | `(1,0)` | `1/1` | 512 | `<= 7.54 us` |
| 2 | LOWER | 1024 | `(1,0)` | `1/1` | 1024 | `<= 9.72 us` |
| 3 | UPPER | 2048 | `(1,0)` | `1/1` | 2048 | `<= 19.05 us` |

### 测试覆盖

仓内交付 CSV 共 1200 例（1000 精度/负向 + 200 性能，与任务配套 `syr2_test.csv` 同源），另有 `TEST_F` 空句柄用例：

| 类别 | 覆盖内容 |
| --- | --- |
| 基础功能 | UPPER/LOWER，`n=0/1`、小质数、2 的幂及 ±1、非对齐和大规模（TC_L0/TC_SQ） |
| 步长 | `incx/incy = ±1/±2/±3`（TC_INC），零步长负向用例（TC_ED） |
| 前导维 | `lda = n` 和 `lda = n + 4`（TC_LD），非法 `lda < n` 负向（TC_ED） |
| 标量 | `(0,0)`、`(1,0)`、纯实、纯虚、负值和大值（TC_AB） |
| 数据 | 均匀/正态分布、实虚独立、全零/交替/极值、Inf/NaN（TC_FL） |
| 接口 | 空 handle、非法枚举、负维度、空指针 x/y/A/alpha（TC_ED + TEST_F） |
| 内存 | 未选三角逐位保护、padding 保护、quick return 后 A 逐位不变 |

官方测试脚本、CSV 和 golden 工具保持原样；性能判定与 `PERF_VERDICT` 输出通过仓内测试代码内建提供。2026-09-05 Release 构建实测：精度 1001/1001 PASSED，性能 200 例全量采集。

## 兼容性分析

1. API 兼容：只新增 `aclblasCsyr2` 公共符号，不修改现有结构体、状态码和其它接口。
2. 产品兼容：A2/A3 共用 `arch22` 实现；不新增产品私有接口，不影响 `arch35` 代码。
3. 行为兼容：参数序列与 cuBLAS `cublasCsyr2` 对齐；负步长、三角引用和 quick return 对齐 Netlib `ssyr2` 的语义推广。
4. 构建兼容：实现与实数 `aclblasSsyr2` 同族归档（两文件布局），按 ops-blas 现有 CMake 组织自动接入。
5. 性能兼容：通用回退路径始终可用，快路径只在对齐、步长和驻留容量条件满足时启用；不满足时不影响功能正确性。
