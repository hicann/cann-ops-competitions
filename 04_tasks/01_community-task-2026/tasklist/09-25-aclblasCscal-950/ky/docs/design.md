# aclblasCscal 算子设计文档

## 一、需求背景

### 1.1 需求来源

本任务来源于 CANN 社区任务 2026 `aclblasCscal` 算子开发任务，要求基于 `ops-blas` 开源工程，面向 **Ascend 950PR**、**CANN 9.1.0**，使用 Ascend C Kernel 直调方式实现单精度复数向量缩放接口 `aclblasCscal`，完成算子设计、开发、功能/精度测试、性能测试及交付。

算子实现路径：

```text
blas/scal/arch35/
```

测试实现路径：

```text
test/scal/cscal/arch35/
```

公共接口声明位于：

```text
include/cann_ops_blas.h
```

本设计不新增 Ascend 950PR 私有平行 API，复用 `ops-blas` 公共 `aclblasCscal` 接口。

### 1.2 背景介绍

#### 1.2.1 算子功能

`aclblasCscal` 属于 BLAS Level-1 算子，对 COMPLEX64 向量执行原地复数缩放。

设：

```text
alpha = ar + ai * i
x[k]  = xr + xi * i
```

则：

```text
alpha * x[k]
= (ar * xr - ai * xi)
+ (ar * xi + ai * xr) * i
```

对 `n > 0`、`incx > 0`，逻辑元素访问规则为：

```text
x[i * incx] = alpha * x[i * incx], i = 0, 1, ..., n - 1
```

其中 `aclblasComplex` 按 `ops-blas` 公共类型定义，由两个 FP32 分量组成：

```text
real: float32
imag: float32
```

算子为原地算子，不额外申请输出 Tensor。

#### 1.2.2 参考语义

功能语义对齐 cuBLAS `cublasCscal`，no-op 行为对齐 Netlib BLAS `cscal`。

关键语义如下：

1. `handle == nullptr` 时返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`。
2. `n <= 0` 或 `incx <= 0` 为合法 no-op，返回 `ACLBLAS_STATUS_SUCCESS`，且不修改 `x`。
3. no-op 判断发生在数据指针校验之前，因此上述场景不要求访问 `alpha`、`x`。
4. `n > 0 && incx > 0` 时：
   - `alpha == nullptr` 返回 `ACLBLAS_STATUS_INVALID_VALUE`；
   - `x == nullptr` 返回 `ACLBLAS_STATUS_INVALID_VALUE`。
5. `alpha == (1, 0)` 可直接返回成功。
6. `alpha == (0, 0)` **不是 no-op**，必须把全部参与计算的复数元素写为 `(+0, +0)`。
7. `incx <= 0` 不执行负步长反向遍历。

#### 1.2.3 设计难点

该算子单元素计算量较小，主要设计难点集中在以下方面：

- 大规模 `incx=1` 场景容易受 Kernel 启动、地址计算和 GM 访存效率影响，需要降低热路径固定开销。
- `incx>1` 时要求仅修改逻辑参与元素，物理 gap 必须保持不变。
- 极端 FP32 输入下，设备编译器对乘加表达式进行 FMA contraction 时，可能改变 Inf/NaN/overflow 分类结果。
- `alpha=(0,0)` 对含 Inf/NaN 的输入不能按普通乘法执行，否则可能生成 NaN，而任务语义要求写精确零。
- `incx` 为任意正整数，设备端物理地址计算需避免 32 位索引溢出。

因此最终方案采用 **Host 侧语义分流 + strict/fast 两条 SIMT 计算路径 + runtime-one FP32 product materialization + shape-aware 调度 + 64 位地址计算**。

---

## 二、需求分析

### 2.1 外部组件依赖

算子本体不引入新的第三方运行时依赖，依赖如下：

| 组件 | 用途 |
| --- | --- |
| CANN 9.1.0 Runtime | ACL Runtime、stream、Kernel 启动 |
| `ops-blas` 公共框架 | handle、状态码、Host 工具函数、构建框架 |
| Ascend C / SIMT API | AIV Kernel 与 SIMT VF 实现 |
| cblas / Netlib BLAS | 仅用于测试阶段 CPU golden，不属于算子运行时依赖 |
| GTest / ops-blas 测试框架 | CSV 驱动功能与精度测试 |

### 2.2 内部适配模块

最终实现涉及模块如下：

| 模块 | 文件 | 职责 |
| --- | --- | --- |
| 公共接口 | `include/cann_ops_blas.h` | 声明 `aclblasCscal` |
| Host 实现 | `blas/scal/arch35/cscal_host.cpp` | 参数校验、quick return、路径选择、分核与 tiling、stream 启动 |
| Kernel 实现 | `blas/scal/arch35/cscal_kernel.cpp` | strict/fast SIMT 复数缩放 |
| Tiling 数据 | `blas/scal/arch35/cscal_tiling_data.h` | Host 到 Device 的调度、标量及分块参数 |
| CSV 测试 | `test/scal/cscal/arch35/cscal_test.csv` | 任务配套功能/精度/性能场景 |
| GTest | `test/scal/cscal/arch35/cscal_test.cpp` | CSV 执行、golden 比对、接口专项与极端回归 |
| 性能脚本 | `test/scal/cscal/arch35/run_formal_perf.sh` | 950PR formal performance 采样 |
| 补充测试 | `cscal_supplemental_test.cpp` | true-normal 与 base-pointer alignment 专项自验 |

### 2.3 需求模块设计

#### 2.3.1 Ascend C 算子原型

公共接口：

```cpp
aclblasStatus_t aclblasCscal(
    aclblasHandle_t handle,
    int n,
    const aclblasComplex* alpha,
    aclblasComplex* x,
    int incx);
```

参数约束：

| 参数 | 类别 | 内存位置 | 类型 | 说明 |
| --- | --- | --- | --- | --- |
| `handle` | 输入 | Host | `aclblasHandle_t` | ops-blas 句柄，携带执行 stream |
| `n` | 输入 | Host | `int` | COMPLEX64 逻辑元素个数 |
| `alpha` | 输入 | Host | `const aclblasComplex*` | 复数缩放系数 |
| `x` | 输入/输出 | Device | `aclblasComplex*` | 原地更新的复数向量 |
| `incx` | 输入 | Host | `int` | 相邻逻辑元素物理步长 |

当 `n>0 && incx>0` 时，`x` 的最小物理 COMPLEX64 元素长度为：

```text
1 + (n - 1) * incx
```

#### 2.3.2 返回值与校验优先级

Host 侧按以下顺序处理，保证异常优先级与 no-op 语义一致：

```text
handle == nullptr
        |
        v
n <= 0 || incx <= 0
        |
        v
alpha == nullptr
        |
        v
x == nullptr
        |
        v
alpha == (1,0)
        |
        v
strict / fast dispatch
        |
        v
launch on handle->stream
```

返回行为：

| 条件 | 返回值 | 是否访问/修改 x |
| --- | --- | --- |
| `handle == nullptr` | `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` | 否 |
| `n <= 0` | `ACLBLAS_STATUS_SUCCESS` | 否 |
| `incx <= 0` | `ACLBLAS_STATUS_SUCCESS` | 否 |
| `alpha == nullptr`（执行场景） | `ACLBLAS_STATUS_INVALID_VALUE` | 否 |
| `x == nullptr`（执行场景） | `ACLBLAS_STATUS_INVALID_VALUE` | 否 |
| `alpha == (1,0)` | `ACLBLAS_STATUS_SUCCESS` | 否 |
| 合法普通场景 | `ACLBLAS_STATUS_SUCCESS` | 异步原地更新 |

---

## 三、需求详细设计

### 3.1 使能方式

本任务使用 `ops-blas` 句柄式 BLAS 直调接口。

| 上层方式 | 是否涉及 |
| --- | --- |
| TensorFlow 训练/推理 | 否 |
| PyTorch 训练/推理 | 否 |
| ATC 推理 | 否 |
| ACLNN 两段式接口 | 否 |
| ops-blas handle 直调 | **是** |
| OPAT 调优 | 否 |
| SGAT 子图切分 | 否 |

用户通过 `aclblasSetStream` 等已有 handle 能力绑定 stream，`aclblasCscal` 在该 stream 上异步启动 AIV Kernel。算子内部不执行强制 stream synchronize，Device 结果读取前由调用方完成同步。

### 3.2 需求总体设计

整体流程如下：

```text
aclblasCscal
    |
    +-- handle 校验
    |
    +-- n/incx no-op
    |
    +-- alpha/x 指针校验
    |
    +-- alpha=(1,0) quick return
    |
    +-- 获取 AIV Core 数
    |
    +-- NeedStrictMath()
    |       |
    |       +-- strict path
    |       |     小 n / strided / 大 alpha / 非有限 alpha
    |       |
    |       +-- fast path
    |             大规模连续有限输入
    |
    +-- BuildCscalTiling()
    |
    +-- cscal_simt_kernel <<<blocks, stream>>>
            |
            +-- __aicore__ shell
                    |
                    +-- asc_vf_call<CscalStrictCompute>()
                    |
                    +-- asc_vf_call<CscalFastCompute>()
```

最终实现保持 Ascend C/CANN 9.1.0 可编译的 SIMT 模式：

```text
extern "C" __global__ __aicore__ kernel shell
+
__simt_vf__ __aicore__ compute function
+
asc_vf_call<...>()
```

#### 3.2.1 Host 侧设计

##### 3.2.1.1 参数校验与 quick return

Host 侧首先完成接口语义处理，使不需要 Kernel 的场景不产生 Device 启动开销。

重点包括：

- `handle` 校验优先级最高；
- `n<=0 || incx<=0` 在 `alpha/x` 校验之前返回；
- `alpha=(1,0)` 在合法数据指针校验后返回；
- `alpha=(0,0)` 不 quick return，进入 Device 写零路径。

##### 3.2.1.2 strict / fast 路径选择

Host 通过 `NeedStrictMath` 将数值敏感场景导向 strict path。

strict path 主要覆盖：

```text
n <= 1024
incx != 1
|alpha.real| > 1e9
|alpha.imag| > 1e9
alpha 任一分量为 Inf/NaN
```

其余大规模、连续、有限普通输入进入 fast path。

设计目的：

1. strided 场景优先保证通用正确性和物理 gap 不被修改；
2. 极端数值进入严格 FP32 运算顺序；
3. 正式性能 case `incx=1` 的大 shape 进入低开销 fast path。

`alpha=(0,0)` 在两条路径中均有专门的精确写零分支。

##### 3.2.1.3 分核与对齐策略

strict path 使用均衡逻辑切分：

```text
baseCount = n / coreNum
remainder = n % coreNum
```

前 `remainder` 个 Core 多处理一个逻辑元素。

fast path 对连续 COMPLEX64 元素按块划分，并将常规 block 长度按：

```text
16 complex64 = 16 * 8 B = 128 B
```

进行对齐，使各 Core 常规起始位置保持较好的 GM 对齐特性。

##### 3.2.1.4 shape-aware 调度

连续 fast path 将中等规模向量与其它大规模向量分成两个调度区间：

```text
mid range:
1536 * 1024 <= n <= 3072 * 1024
```

中等规模使用独立的 Core 数/线程数调度参数，其余规模使用 default 参数。

调度参数通过目标机上的离线 sweep 选取并固化到 Host 实现中。这样可以针对 2M 性能点恢复 roundOne 数值屏障引入的吞吐损失，同时避免扰动已满足要求的 1M 和 4M 场景。

该策略是按规模范围进行的通用 tiling 选择，不对单个正式 case 的精确 `n` 做特判。

##### 3.2.1.5 TilingData

`CscalTilingData` 主要包含：

```text
totalN
incx
alphaReal / alphaImag
roundOne0..3
useCoreNum
startOffset[]
calCount[]
nthreads
strictMath
```

其中：

- `incx` 在 tiling 中保存为 `int64_t`；
- `roundOne0..3` 为 Host 写入的运行时 `1.0f`；
- `startOffset[] / calCount[]` 描述每个 AIV Core 的逻辑数据区间；
- `strictMath` 决定 Kernel 内调用 strict 或 fast SIMT compute。

#### 3.2.2 Kernel 侧设计

##### 3.2.2.1 Kernel Shell

Device 入口为 AIV-only `__aicore__` Kernel Shell。

Kernel 根据：

```text
blockIdx
tiling.useCoreNum
tiling.calCount[blockIdx]
tiling.strictMath
```

选择本 Core 数据区间，并通过 `asc_vf_call` 启动 SIMT VF 计算函数。

##### 3.2.2.2 Strict Compute

strict path 面向小 shape、strided、超大 alpha 及非有限 alpha 等正确性敏感场景。

普通复数乘法使用四个独立 FP32 product：

```text
realMul0 = ar * xr
realMul1 = ai * xi
imagMul0 = ar * xi
imagMul1 = ai * xr

outReal = realMul0 - realMul1
outImag = imagMul0 + imagMul1
```

中间 product 使用可强制物化的严格方式，避免编译器把原表达式跨乘法收缩为改变边界浮点分类的 FMA。

`alpha=(0,0)` 时不读取输入参与乘法，而是直接：

```text
real = +0.0f
imag = +0.0f
```

保证输入为 Inf/NaN 时仍满足任务书的精确置零语义。

##### 3.2.2.3 64 位地址计算

对任意正 `incx`，strict path 使用 64 位索引计算：

```text
logical = uint64(startOffset) + uint64(i)
complexIndex = logical * uint64(incx)
base = complexIndex * 2
```

`base` 为 reinterpret 成 FP32 数组后的下标。

最终实现中：

```text
stride       -> uint64_t
complexIndex -> uint64_t
base         -> uint64_t
```

fast path 的 Core 起始 GM offset 也使用 64 位计算。

该设计避免合法的大 `n / incx` 组合在 Device 端发生 32 位中间地址溢出。

##### 3.2.2.4 Fast Compute

fast path 仅处理大规模连续 `incx=1` 普通有限输入。

每个 SIMT thread 使用 pointer induction：

```text
ptr = xBlock + lane * 2
ptr += blockDim.x * 2
```

避免每次循环重复进行：

```text
startOffset + i
i * stride
complexIndex * 2
```

从而降低热循环中的地址计算成本。

每个 COMPLEX64 元素包含：

```text
2 x FP32 load
4 x FP32 multiply
2 x FP32 add/sub
2 x FP32 store
```

##### 3.2.2.5 runtime-one FP32 product materialization

在最初 fast path 中，直接写：

```text
ar * xr - ai * xi
ar * xi + ai * xr
```

在 CANN 9.1.0 Device 编译过程中可能产生 FMA contraction。

专项回归：

```text
n     = 1025
incx  = 1
alpha = (2, 2)
x     = (FLT_MAX, FLT_MAX)
```

`n=1025` 是当前调度中第一个越过 `n<=1024` strict-small 阈值的连续场景。

原始 fast path 实测出现：

```text
CPU cblas real classification = NaN
NPU fast real classification  = -Inf
```

说明 contraction 改变了 overflow 后的 IEEE 浮点分类。

最终 fast path 使用运行时 `roundOne0..3 = 1.0f`：

```text
realMul0 = (ar * xr) * roundOne0
realMul1 = (ai * xi) * roundOne1
imagMul0 = (ar * xi) * roundOne2
imagMul1 = (ai * xr) * roundOne3
```

每个 product 后增加 multiply-to-multiply 数据依赖，使首个乘法结果先按 FP32 物化，再进入最终加减。

选择该方案前评估过：

| 方案 | 正确性 | 性能结论 |
| --- | --- | --- |
| `#pragma clang fp contract(off)` | 当前 CANN 9.1.0 SIMT 路径未解决回归 | 放弃 |
| bit-domain barrier | 正确 | 1M/2M/4M 均明显超出性能门槛 |
| runtime-zero barrier | 正确 | 三个正式性能点均不达标 |
| runtime-one barrier | 正确 | 配合调度优化后全部达标 |

因此最终采用：

```text
runtime-one FP32 materialization
+
shape-aware scheduling
```

##### 3.2.2.6 异步与内存设计

算子直接使用：

```text
handle->stream
```

启动 Kernel。

算子内部：

```text
额外 Device workspace = 0 B
```

仅原地读写 `x`，Host 侧维护固定大小的 tiling 数据。

最大正式性能 case：

```text
n = 4,194,304
complex64 = 8 Byte
x buffer = 32 MiB
```

该 32 MiB 为用户输入/输出数据，不属于算子额外 workspace。

### 3.3 支持硬件

| 支持的芯片版本 | 是否支持 | 说明 |
| --- | --- | --- |
| Ascend 950PR | **支持** | 本任务目标硬件，已完成实机功能/精度/性能自验 |
| 其它产品 | 本任务不承诺 | 不在本任务验收范围 |

软件环境：

```text
CANN 9.1.0
```

### 3.4 算子约束限制

1. 数据类型固定为 `COMPLEX64`。
2. `alpha` 位于 Host 内存，`x` 位于 Device 内存。
3. `x` 为原地更新，不返回独立输出 Tensor。
4. `n<=0` 为合法 no-op。
5. `incx<=0` 为合法 no-op，不支持负步长反向遍历。
6. `incx>0` 支持正步长访问。
7. 非连续向量由 `incx` 表达，不引入额外 leading dimension 参数。
8. 调用方应保证 `x` 的物理存储覆盖 `1+(n-1)*incx` 个 COMPLEX64 元素。
9. `alpha=(1,0)` 可 quick return。
10. `alpha=(0,0)` 必须写精确 `(+0,+0)`。
11. 算子异步执行，读回结果前由调用方同步 stream。
12. 不申请额外 Device workspace。

---

## 四、特性交叉分析

| 交叉维度 | 风险 | 设计处理 |
| --- | --- | --- |
| no-op × null pointer | 若先校验数据指针会破坏 Netlib no-op 语义 | `handle` 后先判断 `n/incx`，no-op 不访问 `alpha/x` |
| `alpha=0` × Inf/NaN x | 普通 `0*Inf/NaN` 可能产生 NaN | Device 专用直接写 `+0` 路径 |
| fast path × FP32 overflow | FMA contraction 可能改变 NaN/Inf 分类 | runtime-one product materialization |
| strided × 物理 padding | 错误索引可能写坏 gap | strict path 按逻辑 index 寻址，测试 bitwise 检查 gap 不变 |
| 大 `incx` × 地址宽度 | 32 位中间乘法可能溢出 | stride/complexIndex/base 使用 `uint64_t` |
| 多核 × GM 对齐 | 不合理 block start 降低连续访问效率 | fast path block length 按 16 complex64 / 128B 对齐 |
| 数值安全 × 性能 | 全局 strict/volatile 会显著损失大 shape 吞吐 | Host strict/fast 分流，性能关键普通场景走 fast path |
| 2M 调优 × 1M/4M | 单一调度参数可能导致其它规模回退 | mid-size 独立 shape-aware schedule |

该算子不涉及：

```text
broadcast
随机数
跨算子状态
额外 workspace
非确定性归约
```

---

## 五、可维可测分析

### 5.1 精度、功能与性能标准

任务验收口径：

| 验收项 | 标准 |
| --- | --- |
| 功能 | 与 `cublasCscal` / Netlib `cscal` 核心语义一致 |
| Golden | cblas / Netlib `cscal` |
| COMPLEX64 精度 | real/imag 两个 FP32 分量分别比对 |
| `rtol` | `2^-10 ≈ 9.77e-4` |
| `atol` | `2^-16 ≈ 1.53e-5` |
| required matched ratio | `>= 0.99` |
| max abs error | `<= 1e-2` 或 `32 * ULP` |
| 性能采样 | warmup 后有效采样 `>50` 次，取 Avg |
| 目标硬件 | Ascend 950PR |
| CANN | 9.1.0 |

### 5.2 主测试覆盖

任务配套 CSV 共 1200 条，覆盖：

- 小 shape；
- shape 扫描；
- 2 的幂及 `±1`；
- 质数；
- `incx=1/2/3/5/7`；
- `incx=0`、负 `incx`；
- `n=0`、负 `n`；
- null `alpha/x`；
- `alpha=(0,0)`；
- `alpha=(1,0)`；
- 纯虚、负值、大 alpha；
- extreme；
- Inf / NaN；
- strided physical gap；
- 大规模性能相关 case。

仓内 GTest 另增加 4 个接口/回归场景：

```text
NullHandle
NullHandleTakesPrecedenceOverQuickReturn
QuickReturnSkipsDataPointerValidation
FastPathExtremeOverflowMatchesCpuGolden
```

最终主测试：

```text
1204 / 1204 PASS
```

其中 `FastPathExtremeOverflowMatchesCpuGolden` 专门防止未来优化重新引入 FMA overflow 分类差异。

### 5.3 任务缺口补充测试

任务书要求输入同时覆盖均匀分布与真正正态分布，并要求覆盖 base-pointer alignment offset。

任务配套 CSV 未完整覆盖这两项，因此增加 standalone supplemental acceptance。

#### 5.3.1 True Normal Distribution

覆盖：

```text
mu    ∈ [-5, 5]
sigma ∈ [0.1, 2]
```

共 50 个确定性 case，shape 从小规模覆盖至：

```text
n = 1,048,576
```

并覆盖：

```text
incx = 1 / 2 / 3 / 5 / 7
```

最终结果：

```text
NORMAL SUMMARY: 50/50 PASS
```

#### 5.3.2 Base-pointer Alignment Offset

覆盖 Device `x` 相对 128B 的多组合法 offset：

```text
0 / 8 / 16 / 24 / 40 / 72 / 120 Byte
```

组合场景包括：

```text
n=4097,    incx=1
n=1048576, incx=1
n=257,     incx=3
n=257,     incx=5
```

同时对非逻辑 stride padding 做 bitwise unchanged 检查。

最终结果：

```text
ALIGNMENT SUMMARY: 16/16 PASS
```

补充验收汇总：

```text
true normal distribution: PASS
base-pointer alignment offset: PASS
FINAL: PASS
```

### 5.4 性能验证

正式性能脚本采用：

```text
warmup = 5
effective samples = 60
statistic = arithmetic mean
```

最终代码在 Ascend 950PR / CANN 9.1.0 上的一次正式 60-sample 验收结果如下：

| n | incx | Avg (us) | Median (us) | P95 (us) | Min (us) | Max (us) | 门槛 Avg (us) | 余量 | 结果 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1,048,576 | 1 | **11.936** | 11.910 | 12.356 | 11.405 | 12.755 | 13.730 | 13.07% | **PASS** |
| 2,097,152 | 1 | **19.533** | 19.489 | 20.165 | 18.910 | 20.912 | 21.220 | 7.95% | **PASS** |
| 4,194,304 | 1 | **33.471** | 33.529 | 34.274 | 32.532 | 34.981 | 43.310 | 22.72% | **PASS** |

性能判定依据为 Avg，因此三个正式性能点均满足任务书门槛。

### 5.5 内存验证

算子为原地计算，不申请额外 Device workspace：

```text
extra Device workspace = 0 B
```

因此不存在随 `n` 增长的算子内部临时 Device Buffer。

### 5.6 可维护性与兼容性

1. 使用 `ops-blas` 公共 `aclblasCscal` API，不增加产品私有接口。
2. arch35 实现独立放在 `blas/scal/arch35/`，不修改其它架构 Kernel 的计算逻辑。
3. strict 与 fast 路径共用同一公共 Host API，对用户透明。
4. `roundOne` 的存在由专项 extreme regression 保护，后续不得仅因表面上“乘 1 可消除”而删除。
5. shape-aware 调度为通用规模区间策略，不绑定单一性能 case。
6. strided 地址计算使用 64 位整数，保证实现与“任意正 `incx`”接口要求一致。
7. 最终验收脚本验证运行前后 tracked source diff 不变，测试流程不对算子源码进行动态修改。

### 5.7 最终自验结论

最终技术自验状态：

| 项目 | 结果 |
| --- | --- |
| Static/source checks | PASS |
| Build/full test command | PASS |
| 主 GTest | **1204 / 1204 PASS** |
| FMA extreme regression | PASS |
| True Normal | **50 / 50 PASS** |
| Base-pointer Alignment | **16 / 16 PASS** |
| 1M formal performance | PASS |
| 2M formal performance | PASS |
| 4M formal performance | PASS |
| Extra Device workspace | **0 B** |

结论：

```text
aclblasCscal 在 Ascend 950PR / CANN 9.1.0 环境下，
功能、精度、边界语义、补充覆盖及任务书规定的正式性能指标均通过自验。
```

### 5.8 参考资料

1. CANN 社区任务 2026 `aclblasCscal` 任务书。
2. `ops-blas` 开源仓：`https://gitcode.com/cann/ops-blas`
3. CANN 社区任务设计文档模板：`https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md`
4. Netlib BLAS `cscal`：`https://www.netlib.org/blas/cscal.f`
5. NVIDIA cuBLAS `cublasCscal`：`https://docs.nvidia.com/cuda/cublas/index.html`
6. CANN 生态算子精度标准：`https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md`
