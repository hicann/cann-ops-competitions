# 需求背景（required）

## 需求来源

本需求来自 CANN 2026 年 8 月社区任务第 43 号“aclblasCsyr2 算子开发（A2/A3）”。目标是在昇腾 NPU 的 Atlas A2/A3 系列产品上，基于 `cann/ops-blas` 工程和 Ascend C Kernel 直调模式，实现单精度复数对称秩 2 更新接口 `aclblasCsyr2`。

接口的参数序列和核心行为对齐 cuBLAS `cublasCsyr2`；三角引用、负步长起点和 quick return 语义参考 Netlib `ssyr2`，并按复数普通乘法推广。开发完成后，实现代码归档到 `blas/syr2/arch22/`，测试代码归档到 `test/syr2/csyr2/arch22/`，公共接口声明新增到 `include/cann_ops_blas.h`。

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

`ops-blas` 已有实数同族接口 `aclblasSsyr2`，可复用句柄、stream、错误码、Host 直调和工程组织方式；但不能直接复用实数计算 Kernel，原因如下：

1. `aclblasCsyr2` 的每个元素包含实部、虚部，需要展开普通复数乘加；
2. Netlib/cblas 没有复数 symmetric syr2，测试 golden 需要按 `ssyr2` 的三角、步长和列主序语义自实现；
3. 性能用例的矩阵规模较大，纯逐元素 Vector 更新难以达到任务书门槛，需要利用秩 2 更新的低秩结构构造 Cube 主路径；
4. 仍需保留覆盖负步长、非紧凑 `lda`、非对齐尺寸和任意 `alpha` 的通用路径。

工程落点如下：

| 项目 | 设计落点 |
| --- | --- |
| 公共声明 | `include/cann_ops_blas.h` 新增 `aclblasCsyr2`，不定义产品私有平行 API |
| Host/Kernel | `blas/syr2/arch22/` |
| 测试 | `test/syr2/csyr2/arch22/`，CSV 驱动 GTest |
| 数据类型 | `aclblasComplex`，实部和虚部均为 FLOAT32 |
| 目标产品 | Atlas A2/A3；性能验收设备为 Atlas 800I A2（Ascend 910B3） |
| CANN | 9.1.0 |

# 需求分析（required）

## 需求描述

使用 Ascend C 实现 `aclblasCsyr2`，支持 UPPER/LOWER、正负步长、`lda` padding、参数异常处理和 quick return；在指定三角内完成普通复数对称秩 2 更新，未指定三角保持不变。实现需满足任务书精度标准，并在 910B3 上达到三条性能门槛。

## 需求拆解

1. 新增公共 `aclblasCsyr2` 声明，参数顺序与 cuBLAS 对标接口一致。
2. 支持 `ACLBLAS_UPPER` 和 `ACLBLAS_LOWER`，只读写指定三角。
3. 支持 COMPLEX64，采用普通复数乘法，不执行共轭。
4. 支持 `incx`、`incy` 的正负步长，拒绝零步长。
5. 支持 `lda >= max(1, n)`，包括列尾 padding。
6. `n = 0` 或 `alpha = (0, 0)` 时合法 quick return，不启动 Kernel、不访问 Device 数据。
7. 参数错误返回任务书指定状态码；Host 不隐式同步 stream。
8. 提供通用 AIV 路径和满足资源/工作量条件时的 Cube–Vector 协同路径，不按某个精确 shape 写专用分支。
9. 使用任务书提供的官方 CSV/GTest 自验；官方工具保持原样，额外性能与随机测试以新增文件提供。
10. 精度按 matched-ratio/max-abs 规则验收，性能按 `msprof --task-time=on` 的 Kernel Task Duration 验收。

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

`yStart` 同理，逻辑元素 `i` 的物理地址为 `start + i * inc`。

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

整体采用“通用 AIV 路径 + 连续大工作量 Cube–Vector 协同路径”的分层设计。Host 根据接口参数、有效三角 tile 数、芯片 AIV/AIC 数量、对齐条件和 workspace 容量选择路径；选择条件来自工作量与硬件资源，不使用 `n == 某个值` 的精确 shape 特化。

#### Host 侧设计

1. 参数检查顺序：
   - `handle == nullptr` 返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`；
   - 非法 `uplo` 返回 `ACLBLAS_STATUS_INVALID_ENUM`；
   - `n < 0`、零步长或非法 `lda` 返回 `ACLBLAS_STATUS_INVALID_VALUE`；
   - `n == 0` 直接成功返回；
   - `alpha == nullptr` 返回 `ACLBLAS_STATUS_INVALID_VALUE`；
   - `alpha == (0,0)` 直接成功返回；
   - 之后检查 `x`、`y`、`A` 和 stream。
2. 填充 tiling data：保存 `n/lda/uplo/incx/incy/alpha`、输入输出地址、路径类型、活跃核数、Cube K 深度和 workspace 信息。
3. 通用路径按逻辑三角任务数限制 AIV 核数，避免启动超过有效工作量的空核。
4. Cube 路径需要单位步长、紧凑列主序、满足 Cube tile 对齐、有效三角 tile 足够且 workspace 可用；条件不满足时回退 AIV 路径。
5. 活跃 AIC 数根据有效三角输出 tile 数和“每核目标 tile 数”计算，并同时满足配套 AIV 预处理任务量；不直接绑定某个矩阵规模。
6. 通过 handle 绑定的 stream 异步启动 Kernel，Host 不做 `aclrtSynchronizeStream`。

#### 通用 AIV Kernel

通用路径覆盖任意合法 `alpha`、正负步长、非紧凑 `lda` 和非对齐 `n`：

1. 将指定三角按列或三角 tile 唯一分配给 AIV，避免多核写冲突；
2. 对单位步长使用 UB tile 和批量 DataCopy，对正负非单位步长使用标量/分段 gather 语义；
3. 在 UB 中展开复数乘法，完成 CopyIn → Compute → CopyOut；
4. 使用双缓冲或独立队列重叠 MTE2、Vector 和 MTE3；尾块使用 mask/padding 处理；
5. 未指定三角完全不读不写，`lda` padding 不修改。

#### Cube–Vector 协同 Kernel

SYR2 是两个外积之和，可把复数乘加展开为小 K 实数矩阵乘。对于满足工作量和布局条件的连续大问题，采用 MIX Kernel：

1. AIV 预处理：
   - 按固定 packet 从 GM 读取 `x/y`；
   - 把交错 complex64 拆成 Cube 友好的实数平面；
   - 将 `alpha` 融合进平面系数；
   - 写入共享 workspace 的 left/right 面板。
2. AIV 完成共享输入后，以物理 MIX 核对为单位同步并释放对应 AIC；只让实际活跃核参与同步。
3. AIC 主体：
   - 把三角主体划分为矩形 Cube tile；
   - 通过 GM→L1→L0 双槽流水预取下一 tile；
   - 小 K MMAD 计算两个外积的实部/虚部组合；
   - 采用多 CO1 槽交错 MMAD 与 Fixpipe，减少 M/FixedPipe 空泡；
   - 通过原子 Fixpipe 累加到目标矩阵，保证 AIV 边界带与 AIC 主体可安全合并。
4. AIV 收尾：
   - 处理对角、三角边界以及不适合完整 Cube tile 的带状区域；
   - 保证另一三角和 padding 不被写入。
5. 负载均衡：
   - 先按压缩三角的 tile ordinal 均分；
   - 当余数 tile 少于活跃 AIC 的一半时，将重核的尾 tile 按通用半 tile 拆给空闲核；
   - 拆分条件仅由 `totalTiles % activeCores` 决定，不检查精确 shape。

#### UB/L1/L0 与流水设计

- AIV packet 大小由 UB 容量、输入平面数、双缓冲和临时转置空间共同决定，避免过小搬运导致带宽利用率不足；
- AIC 使用两组 L1/L0 输入槽，使下一 tile 的 MTE2/MTE1 与当前 MMAD/Fixpipe 重叠；
- CO1 使用多槽轮转，按 `MMAD(q0/q1/...) → Fixpipe(q0/...)` 交错发射；
- 尾 tile 只搬运其实际使用的输入半面板，减少无效 GM→L1 和 L1→L0 数据量；
- 所有同步均限定在参与当前路径的有效核，避免空核扩大启动和 barrier 开销。

### Tiling 通用性

Tiling 的输入是：数据布局、对齐、可用 workspace、AIC/AIV 核数、完整三角 tile 数、每核目标 tile 数和余数分布。以下做法明确禁止：

- `if (n == 1024)`、`if (n == 2048)` 等精确规模分支；
- 为单个验收 case 写固定 core map；
- 依赖未声明的输入值分布决定正确性路径。

性能用例可触发同一套 size-class 规则，但其它满足相同资源和工作量条件的 shape 也会复用该路径。

## 支持硬件

| 芯片版本 | 支持情况 | 验证计划 |
| --- | --- | --- |
| Atlas 800I/T A2（Ascend 910B3） | 支持 | CANN 9.1.0；功能、精度及任务书性能门槛正式验证 |
| Atlas A3（arch22） | 支持 | 使用同一 API/Kernel 源码完成构建、功能与边界验证，并重新评估资源参数 |

## 算子约束限制

1. 仅支持 COMPLEX64，不支持混合精度。
2. `A` 为列主序方阵，只更新 `uplo` 指定三角。
3. `n >= 0`，`incx != 0`，`incy != 0`，`lda >= max(1,n)`。
4. `incx`、`incy` 支持负数，但不支持 0；实现需避免 `INT_MIN` 取绝对值溢出。
5. `A` 原地更新，调用方负责输入输出内存合法性及 stream 生命周期。
6. `n == 0` 或 `alpha == (0,0)` 为合法 no-op。
7. Cube 主路径允许使用 handle workspace；workspace 不足时回退正确的通用 AIV 路径。

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

- 未指定三角保持原值；
- `lda` padding 不被修改；
- UPPER/LOWER、正负步长、quick return、非法参数和空指针状态码；
- Inf/NaN 按任务测试规则处理。

### 性能标准

性能验收设备为 Atlas 800I A2（Ascend 910B3），CANN 9.1.0：

| case | uplo | n | alpha | incx/incy | lda | 门槛 |
| --- | --- | ---: | --- | --- | ---: | ---: |
| 1 | UPPER | 512 | `(1,0)` | `1/1` | 512 | `<= 4.82 us` |
| 2 | LOWER | 1024 | `(1,0)` | `1/1` | 1024 | `<= 6.22 us` |
| 3 | UPPER | 2048 | `(1,0)` | `1/1` | 2048 | `<= 12.19 us` |

开发和验收统一使用 `msprof --task-time=on` 导出的 Kernel `Task Duration(us)`：每轮先 warmup，再采集至少 60 个有效样本，取后 60 个样本均值；候选和基线至少做三轮相邻 A/B。ACL Event 或进程 wall time只用于快速诊断，不作为达标结论。

需要进一步归因时单独采集 PipeUtilization，观察 AIC/AIV 的 Scalar、MTE、Vector、Cube 和 Fixpipe；每轮保留原始 profile、代码 commit、动态库和汇总结果。

### 测试覆盖

| 类别 | 覆盖内容 |
| --- | --- |
| 基础功能 | UPPER/LOWER，`n=0/1`、小质数、2 的幂及 ±1、非对齐和大规模 |
| 步长 | `incx/incy = ±1/±2/±3`，零步长负向用例 |
| 前导维 | `lda=n` 和多种 `lda>n`，非法 `lda<n` |
| 标量 | `(0,0)`、`(1,0)`、纯实、纯虚、一般复数和大值 |
| 数据 | 均匀/正态分布、实虚独立、任务书指定特殊值 |
| 接口 | 空 handle、非法枚举、负维度、空指针、未初始化 stream |
| 内存 | 未选三角保护、padding 保护、workspace 边界 |

官方测试脚本、CSV 和 golden 工具保持原样；重复 launch、随机输入、profile 汇总等能力通过新增辅助文件实现。

## 兼容性分析

1. API 兼容：只新增 `aclblasCsyr2` 公共符号，不修改现有结构体、状态码和其它接口。
2. 产品兼容：A2/A3 共用 `arch22` 实现；不新增产品私有接口，不影响 `arch35` 代码。
3. 行为兼容：参数序列与 cuBLAS `cublasCsyr2` 对齐；负步长、三角引用和 quick return 对齐 Netlib `ssyr2` 的语义推广。
4. 构建兼容：实现与实数 `aclblasSsyr2` 同族归档，按 ops-blas 现有 CMake 组织接入。
5. 性能兼容：通用路径始终可用，Cube–Vector 路径只在布局、工作量和资源条件满足时启用；workspace 不足时不影响功能正确性。
