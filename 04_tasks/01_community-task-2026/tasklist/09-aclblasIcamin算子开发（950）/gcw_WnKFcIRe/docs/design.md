# aclblasIcamin 算子设计文档

| 项目 | 内容 |
| --- | --- |
| 算子 | `aclblasIcamin` |
| 任务 | 9 月社区任务：`aclblasIcamin` 算子开发（950） |
| 目标硬件 | Ascend 950PR |
| CANN 版本 | 9.1.0 |
| 代码基线 | `cann/ops-blas` master，`0ea22c81b4b721f723730586c48a21c6d759da6a` |
| 文档版本 | v1.1（2026-09-16 已在 950PR/CANN 9.1.0 完成全量验证：1251 条测试全过，性能 200/200 达标） |

> v1.1 实测摘要：Release 构建下 1M/2M/4M 三个典型 case 平均单次耗时 9.03/11.06/16.93 us
> （门限 24.77/24.59/29.69 us），200 条性能用例全部满足 `H100 基线 × 2.5`（最差余量比约 0.67）；
> 精度 1048 条 CSV 用例 + 手写用例全部 bit-exact 通过。关键工程结论：master 强制 Debug 构建
> 会使 kernel 性能劣化约 10 倍，本任务分支默认改为 Release（可 -D 覆盖），与 scnrm2 分支一致；
> `ReduceMin` 带索引模式的 NaN 行为随优化级别变化，NaN 处理必须显式向量化清洗，不能依赖指令语义。

> 本文只描述本任务新增的 Ascend 950PR（`arch35`）实现与公共接口声明。其他产品线的既有实现与行为不在本任务中改动。

# 需求背景（required）

## 需求来源

- 随任务下发的《aclblasIcamin 算子开发任务书》及配套测试材料（1200 条 CSV 用例、H100 GPU 基线、验收脚本）。
- 目标代码仓：[cann/ops-blas](https://gitcode.com/cann/ops-blas)。
- 社区任务流程：[2026 社区任务参与及 PR 流程](https://gitcode.com/cann/cann-competitions/blob/master/04_tasks/01_community-task-2026/README.md) 与【社区任务】流程注意事项（gitcode.com/org/cann/discussions/39）。
- 算法语义参考：cuBLAS `cublasIcamin`（[官方文档](https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-amin)）。标准参考 BLAS（Netlib）无 icamin 例程，仅有同族实数接口 [isamin.f](https://www.netlib.org/blas/isamin.f)，语义细节以 cuBLAS 官方文档为最高优先级来源。

## 背景介绍

`aclblasIcamin` 在单精度复数（COMPLEX64）向量中查找最小模元素的索引，属于 BLAS Level 1 索引归约算子。数学表达式为：

```text
result = argmin_i (|Re(x[k])| + |Im(x[k])|),  i = 1..n,  k = 1+(i-1)*incx
```

复数"模"按 BLAS icamin 惯例定义为实部绝对值加虚部绝对值（L1 模），不是欧几里得模。结果为 **1-based 索引**（兼容 Fortran 惯例）；多个元素模相同时返回**最小索引**。输入的每个 `aclblasComplex` 元素由两个交错存储的 FP32 分量组成（`{real0, imag0, real1, imag1, ...}`，n 个复数对应 2*n 个 float），输出为单个 INT32 整数。

与浮点归约算子不同，本算子输出是离散整数：`|Re|+|Im|` 的比较在 IEEE 浮点语义下结果确定，归约顺序不影响最终索引值，因此精度判定为与 golden **精确一致（bit-exact 整数索引比对，`EXPECT_EQ`）**，浮点容差标准（rtol/atol）在本算子退化为精确相等判定。

## 任务启动基线

| 项目 | 当前状态 | 本任务处理 |
| --- | --- | --- |
| 公共 API | `include/cann_ops_blas.h` 无 `aclblasIcamin` 声明 | 本任务新增声明，与 `cublasIcamin` 逐参数对齐，供其他产品线共用；禁止 950PR 私有平行接口 |
| 复数类型 | `include/cann_ops_blas_common.h` 已定义 `aclblasComplex { float real; float imag; }` | 直接复用，按两个交错 FP32 分量读取 |
| `arch35/iamin` | 同族实数接口 `aclblasIsamin` 已有完整 arch35 实现：参数校验、tiling、`ReduceMin` 带索引归约、SIMT 跨步路径、跨核二次归约、workspace 协议 | 本任务实现框架蓝本，代码放在同目录 `blas/iamin/arch35/` |
| `arch35` 其他算子 | `scnrm2` 已验证复数 AoS 的 DINTLV 拆分、`(value, index)` 归约协议与 950PR 工程约束 | 复用其复数处理与验证方法 |
| `arch22` 等既有实现 | 与本任务无关 | 保持不变 |
| 测试工程 | `test/isamin/` 已有 CSV 驱动 GTest 模式（参数/golden/wrapper/CMake），测试框架 `test/frame/fill.h` 支持任务 CSV 全部填充模式 | 参照新建 `test/iamin/icamin/`；框架目前无正态分布生成器，需按任务书值域要求补充 |
| 任务配套材料 | `icamin_test.csv` 1200 条（1000 精度 + 200 性能）、`gpu_baseline.csv`（H100 基线已回填）、`verify_accuracy.py` / `verify_performance.py` | 官方脚本原件不修改，仅作可追溯输入与粗筛；判定口径见"验收口径"节 |

## 功能价值

完成后，Ascend 950PR 可通过统一的 ops-blas 句柄式接口执行 COMPLEX64 向量最小模索引查找，语义与 cuBLAS `cublasIcamin` 对齐（仓惯例允许的差异除外，见"参数与返回值语义"），填补公共头文件中该接口的空白，并与同族实数接口 `aclblasIsamin`/`aclblasIsamax` 形成完整的最值索引族。

# 需求分析（required）

## 需求描述

### 接口

公共头文件 `include/cann_ops_blas.h` 新增声明，签名与 cuBLAS `cublasIcamin` 逐参数对齐（handle 及参数顺序一一对应，无需额外映射）：

```cpp
aclblasStatus_t aclblasIcamin(
    aclblasHandle_t handle,
    int n,
    const aclblasComplex* x,
    int incx,
    int* result);
```

`handle` 为 Host 侧句柄，携带 stream；`x` 和 `result` 均为 Device 地址；算子使用 `handle` 绑定的 stream 异步执行，调用者读回 `result` 前必须同步 stream。

### 数学定义

令逻辑输入向量为 `x_0, x_1, ..., x_(n-1)`，其中 `x_i = a_i + b_i j`，定义模：

```text
m_i = |a_i| + |b_i|        （FP32 单精度加法，非欧几里得模）
result = 1 + argmin_i m_i   （1-based；m_i 相同取最小 i）
```

正步长时逻辑元素 `i`（0-based）对应物理复数下标 `i * incx`。`incx < 1`（含 0 与负步长）为合法 quick return，**本族负步长不反向遍历向量**（与 axpy/nrm2 族不同），因此不涉及负步长地址基准问题。`n > 0` 且 `incx >= 1` 时 `x` 的物理复数长度至少为 `1+(n-1)*incx`。

### 参数与返回值语义

参数校验顺序固定，与仓内 `blas/iamin/arch35/isamin_host.cpp` 的实现逐条对齐（任务书明确以此为准）：

| 顺序 | 条件 | 行为 |
| --- | --- | --- |
| 1 | `handle == nullptr` | 返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| 2 | `n < 0` | 返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| 3 | `n == 0` 或 `incx < 1` | 合法 quick return：`result` 非空则写 `0`，返回 `ACLBLAS_STATUS_SUCCESS`；不检查 `x`，不触发 Kernel |
| 4 | `x == nullptr` | 返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| 5 | `result == nullptr` | 返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| 6 | 运行时资源失败（核数查询失败、workspace 不足等） | 映射为对应 internal/execution 状态码 |

说明：

- quick return 优先于 `x`/`result` 空指针检查：`n == 0` 或 `incx < 1` 时即使 `x == nullptr` 也返回成功（任务配套 CSV 有两条该组合用例，期望 `SUCCESS`）；`result == nullptr` 时 quick return 无法写值，仍返回成功（沿用 isamin 行为）。
- `n < 0` 返回 `ACLBLAS_STATUS_INVALID_VALUE` 是仓内同族约定，与 cuBLAS "`n <= 0` 置 0"存在差异；任务书事实表开放问题 Q3 已明确按仓内 README 约束与 isamin arch35 实现执行。
- 状态码语义以 `include/cann_ops_blas_common.h` 为准，不新增状态码。

### 特殊浮点值

复数模 `|a|+|b|` 为 NaN（任一分量为 NaN）的元素不参与比较（跳过）；任务书补充说明（开放问题 Q7 对齐结论）规定 golden 语义为：跳过 NaN 元素，首元素模为 NaN 时基准置 `FLT_MAX`。据此：

| 输入集合 | 输出 |
| --- | --- |
| 全部元素模为 NaN | `1`（基准索引初值，无任何更新） |
| 部分元素模为 NaN | 非 NaN 元素中的最小模最小索引 |
| 含 `±Inf` 分量 | `Inf` 按普通浮点值参与比较（大于任何有限值，`Inf` 模并列时取最小索引） |
| 全零 / 模全部相同 | `1`（tie 取最小索引） |
| 其余有限值 | 正常最小模索引 |

实现与 golden 的等价性论证见"正确性依据"。NaN/Inf 混合时的逐元素行为以 cblas 风格 golden 循环为唯一契约，不依赖 `ReduceMin` 等硬件指令对 NaN 的传播特性。v1.1 实测确认：`ReduceMin` 带索引模式在 Debug 下表现为跳过 NaN、在 Release（-O2）下表现为传播 NaN，因此实现改为**归约前向量化清洗**（`Compare` 按 `v != v` 生成 NaN 掩码，`Select` 将 NaN 置为 `+Inf`），归约输入中不含 NaN，行为在任意优化级别下确定；另有标量重扫兜底（tile 归约结果仍为 NaN 时对该 tile 从 GM 逐元素重扫，正常路径不触发）。

## 需求拆解

| 子任务 | 设计输出 | 验收重点 |
| --- | --- | --- |
| 公共声明 | `include/cann_ops_blas.h` 新增接口声明 | 签名与 cuBLAS 逐参数对齐，无私有平行接口 |
| Host 接口 | 校验、quick return、tiling、workspace、异步发射 | 状态码顺序、quick return 优先性、地址计算无溢出 |
| 连续 Kernel | `incx == 1` 的 AIV 单遍读取 + 带索引最小值归约 | 模计算正确、tie 取首、bit-exact、性能达标 |
| 跨步 Kernel | `incx > 1` 的 SIMT 读取和树形归约 | 与连续路径结果一致、无越界 |
| 跨核归约 | 每核 `(minVal, minIdx)` 记录 + 单核二次归约 | 合并规则保持字典序、全核记录必达 |
| 测试工程 | `test/iamin/icamin/` CSV 驱动 GTest、CPU golden、正态分布生成器补充 | 覆盖任务书全部有效维度与官方 1200 条用例 |
| 文档 | iamin README 产品支持表与约束更新 | API 与实现一致，标注 950PR 支持 |

## 验收口径

### 精度

输出为单个 INT32 整数索引，按任务书与生态算子开源精度标准的退化口径执行：

```text
actual == golden        （1-based 索引逐位相等，EXPECT_EQ）
```

任一用例不相等即失败；`matched_ratio` 退化为单值判定（该用例通过即 ratio = 1）。quick return 用例（`n == 0` 或 `incx < 1`）判定 `result == 0`。golden 由测试工程内 CPU 参考实现（cblas 风格手写循环；标准 cblas 无 icamin 接口）生成，模计算严格保持 FP32 单精度（不得以 double 累加），逐元素 `|Re|+|Im|`、严格小于取最小索引、跳过 NaN（首元素 NaN 基准置 `FLT_MAX`）。

### 性能

任务书 §3.3 标杆耗时经核对为 H100 GPU 基线（`gpu_baseline.csv`，单位 ms）按倍率 0.4 调整后的值，即判定式 `NPU 平均单次耗时 <= GPU 基线 * 2.5`，三个典型 case 预计算值：

| n | incx | H100 基线 | Ascend 950PR 平均耗时上限 |
| ---: | ---: | ---: | ---: |
| 1,048,576 | 1 | 9.907 us | 24.77 us |
| 2,097,152 | 1 | 9.836 us | 24.59 us |
| 4,194,304 | 1 | 11.876 us | 29.69 us |

200 条性能用例的其余条目按同一公式 `gpu_ms * 2.5` 判定（小尺寸条目的阈值约 15.5 us，重点盯防发射开销）。计时口径：先 warmup，再有效采样 >50 次取平均；计时区只包含 stream 上的算子执行，输入生成、Host/Device 搬运、CPU golden 与结果比对不得混入。任务配套的 `verify_performance.py` 解析 GTest 行尾毫秒（含 host 准备与 golden，属保守上界），其原件不修改、PASS/FAIL 不作为发布结论；发布数据以测试工程内计时循环打印的平均单次 us 为准，msprof 作诊断（同仓内 scnrm2 的处理方式）。

# 详细设计（required）

## 算子分析

### 实现方案比较

| 方案 | GM 读取 | 索引获取 | 结论 |
| --- | --- | --- | --- |
| AIV 单遍读 + `ReduceMin`（带索引输出），逐 tile 合并 | 1 遍 | 指令直接给出 tile 内最小值与索引 | 采用（isamin 已验证的 950PR 路径） |
| AIV 求最小值后二遍找索引 | 2 遍 | 第二遍比较定位 | 带宽翻倍，不采用 |
| 全量 SIMT 直读 | 1 遍 | 标量比较 | 跨步路径采用；连续路径作为正确性交叉验证兜底 |

本算子算术强度低于同族 nrm2（无平方/开方），性能 case 均为大规模连续普通分布，理论下限由 MTE2 供数、归约与跨核尾巴决定。连续路径单遍 GM 读取即可满足性能余量（见"性能分析"）。

### 950PR 编程机制取舍

| 机制 | 本算子结论 | 设计落点 |
| --- | --- | --- |
| 输入路径 | MTE2 → UB 单遍 | 连续 kernel 按 tile 搬入，无 AIC/L1 供数 |
| 复数模计算 | 实虚拆分后 `Abs` + `Add` | 候选 A：RegBase `DINTLV_B32` 拆分（scnrm2 已验证）；候选 B：整向量 `Abs` 后按奇偶归约；以编译报告与实测择优，均保持 FP32 单精度 |
| 带索引归约 | `ReduceMin(dst, src, work, count, true)` | isamin 同款：dst[0] 为最小值，dst[1] 为 tile 内索引（位模式）；tile 内并列取首索引的设备语义列入验证门禁 |
| 跨步路径 | SIMT 逐复数标量读取 | isamin 跨步路径同构，每元素读两个分量 |
| 跨核归约 | 每核 2 个 FP32（值 + 索引位模式）写 workspace，单核二次归约 | 完全复用 isamin 协议 |
| AIC/Cube | 不采用 | 索引归约无矩阵结构 |

### 正确性依据（任意归约顺序 bit-exact）

定义 `(value, index)` 对的字典序合并规则：

```text
better(a, b) = (b.val < a.val) || (b.val == a.val && b.idx < a.idx)
```

逐元素更新、tile 间合并、线程树归约、跨核二次归约全部使用该规则（NaN 值不参与更新）。该规则具有结合律：任何分组/顺序的归约结果都等于"跳过 NaN 后的最小模最小索引"，与 cblas 风格串行 golden 循环（严格小于才更新，故并列保留先出现者）逐一对应，因此归约顺序不影响最终索引值，bit-exact 判定成立。

边界对应关系：所有记录初值为"无有效值"（`bestVal = FLT_MAX, hasValue = false`）；若全部元素模为 NaN，各级均无更新，最终 `hasValue == false`，按约定索引取 0，出口 `+1` 得 `result = 1`，与 golden"首元素 NaN 基准置 FLT_MAX 后无更新返回 1"一致。isamin 已在 950PR 验证该协议，本算子沿用。

已知语义边界（与 golden 的唯一理论分歧）：当首元素模为 NaN 且其余非 NaN 模全部 `>= FLT_MAX` 时，串行 golden 返回 1。v1.1 实测闭环：向量化清洗后 NaN 模变为 `+Inf`，该组合下 tile 最小值为 `+Inf`，不优于 `FLT_MAX` 种子、不触发更新，最终同样返回 1，与 golden 一致；该组合不再是分歧项。

### 支持数据类型和数据排布

| 对象 | 数据类型 | 逻辑排布 | 物理排布 |
| --- | --- | --- | --- |
| `x` | COMPLEX64 | `[n]` | `{real0, imag0, real1, imag1, ...}`，物理复数长度 `1+(n-1)*incx` |
| 中间记录 | FP32 值 + UINT32 索引位模式 | 每核 2 个 FP32 | handle workspace，`2*numBlocks` 个 FP32 |
| `result` | INT32 | scalar | Device 上 4 字节 |

### 支持形状与步长

| 项目 | 支持范围 |
| --- | --- |
| 逻辑 shape | 一维 `[n]`，归约为标量索引 |
| `n` | `n >= 0`；`n == 0` quick return；`n < 0` 返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| `incx` | `incx >= 1` 正常计算；`incx < 1`（含 0 与负步长）quick return，不反向遍历 |
| 物理跨度 | `n > 0` 时至少 `1+(n-1)*incx` 个复数元素（`2*(1+(n-1)*incx)` 个 FP32） |
| broadcast / leading dimension / 非连续 Tensor | 不涉及 |

### 总体执行流程

```mermaid
flowchart TD
    A[aclblasIcamin] --> B{参数校验}
    B -->|handle 空| Z1[HANDLE_IS_NULLPTR]
    B -->|n 小于 0| Z2[INVALID_VALUE]
    B -->|n==0 或 incx 小于 1| Q[result 非空则写 0，SUCCESS]
    B -->|x/result 空| Z2
    B -->|合法| C[计算 tiling 与 workspace 检查]
    C --> D{路径选择}
    D -->|incx==1 且单核可装| S[small kernel 直通写 result]
    D -->|incx==1| E[AIV kernel：tile 归约写 workspace]
    D -->|incx 大于 1| G[SIMT kernel：跨步归约写 workspace]
    E --> F[reduce kernel：跨核合并写 result]
    G --> F
```

## 算子实现

### Host 侧设计

#### 参数校验流程

```mermaid
flowchart TD
    A[入口] --> B{handle 为空?}
    B -->|是| B1[HANDLE_IS_NULLPTR]
    B -->|否| C{n 小于 0?}
    C -->|是| C1[INVALID_VALUE]
    C -->|否| D{n==0 或 incx 小于 1?}
    D -->|是| D1[result 非空则写 0，返回 SUCCESS]
    D -->|否| E{x 为空?}
    E -->|是| E1[INVALID_VALUE]
    E -->|否| F{result 为空?}
    F -->|是| F1[INVALID_VALUE]
    F -->|否| G[计算 tiling 并发射 Kernel]
```

quick return 的写零方式对齐 isamin 实现（Host 侧直写）；950PR 上若验证 device 指针不可 Host 直写，则改为 `aclrtMemsetAsync`（INT32 `0` 与全零位模式一致），该替换不改变对外语义，列入验证门禁。

#### Tiling 策略

tiling 结构沿用 isamin 布局（字段语义不变，新增类型名 `IcaminTilingData`），所有计数单位为**复数元素**：

```cpp
struct IcaminTilingData {
    uint32_t totalN;     // 复数元素总数 n
    uint32_t perCoreN;   // 前 useCoreNum-1 核的复数元素数
    uint32_t lastCoreN;  // 末核复数元素数（perCoreN + 余数）
    uint32_t useCoreNum; // 实际使用核数
    uint32_t tileSize;   // UB 单 tile 复数元素数
    uint32_t nthreads;   // SIMT 线程数（incx > 1 时使用）
    uint32_t incx;       // 步长（1 = 连续）
};
```

```text
numBlocks  = min(n, GetAivCoreCount())
perCoreN   = n / numBlocks
lastCoreN  = perCoreN + n % numBlocks
tileSize   = 编译期常量（复数元素/tile，初版 4096：DataCopyPad 的 blockLen 为 uint16_t，
             2T floats * 4B 不得超过 65535B，4096*2*4=32768B 安全；按实测 A/B 调整）
nthreads   = incx > 1 时按 isamin 同款公式在 [SIMT_MIN, SIMT_MAX] 内取值；incx == 1 时为 0
```

路径选择：

| 条件 | 发射序列 |
| --- | --- |
| `incx == 1 && numBlocks == 1 && n <= tileSize` | `icamin_small_kernel<<<1>>>` 直通，单核归约直接写 `result` |
| `incx == 1`（其余） | `icamin_aiv_kernel<<<numBlocks>>>` + `icamin_reduce_kernel<<<1>>>` |
| `incx > 1` | `icamin_simt_kernel<<<numBlocks>>>` + `icamin_reduce_kernel<<<1>>>` |

Host 所有地址与计数中间量（`2*n`、`1+(n-1)*incx`、物理字节数）使用 UINT64 计算；`n` 为 INT32 入参，`2*n` 上界 `2^32`，UINT64 无溢出。Host 无法获知调用者实际分配长度，缓冲区容量由调用者保证。

#### UB 预算（连续 kernel，T = tileSize 复数元素 = 4096）

| 缓冲 | 大小 | 说明 |
| --- | --- | --- |
| inBuf | `8*T` = 32 KiB ×2（双缓冲 = 64 KiB） | 输入复数 tile（2T 个 FP32）；TQue 双缓冲流水 |
| reBuf | `4*T` = 16 KiB | 实部（`DeInterleave` 输出，原地 `Abs` 后兼作模值缓冲） |
| imBuf | `4*T` = 16 KiB | 虚部（`DeInterleave` 输出；`Add` 后复用为 +Inf 向量） |
| maskBuf | 4 KiB | NaN 清洗的比较掩码 |
| workBuf | 约 0.3 KiB | `ReduceMin` level1 对齐工作区 + 32B |
| slotBuf | 每 tile 32B | tile 归约结果槽位（值 + 索引位模式），末尾一次性合并 |
| outBuf | 32 B | 归约输出（值 + 索引位模式） |
| 合计 | 约 104 KiB | 远低于 950PR 单 AIV UB 预算（约 248 KiB） |

#### Workspace

每核向 handle workspace 写 2 个 FP32（`minVal`、索引位模式），共 `2*numBlocks` 个 FP32，按 64 个 FP32 对齐后向 handle 校验容量（复用 isamin 的对齐与检查公式）；容量不足返回失败状态码，默认 32 MiB handle workspace 无压力。workspace 依赖同一 stream 的顺序语义，沿用仓库约束：同一 handle 不支持多 Host 线程/多 stream 无序并发复用。

#### 发射与异步语义

- `launchBlockNum == useCoreNum`，AIV-only kernel；连续路径两次发射（主 kernel + reduce kernel）在同一 stream 上顺序执行，reduce kernel 读取所有核记录后由单核写出最终索引。
- 仓内 `*_kernel_do` 发射包装返回 `void`，Host 只检查参数与前置资源，不虚构 kernel 入队状态；异步执行错误由调用者后续同步时暴露。
- Kernel 与 quick return 均使用 `handle->stream`。
- 内部索引统一 0-based，出口处 `+1` 转 1-based。

### Kernel 侧设计

#### 连续 kernel（`icamin_aiv_kernel` / `icamin_small_kernel`）

每核处理 `[blockIdx*perCoreN, ...)` 区间复数元素，按 tile 循环（最终实现为 TQue 双缓冲流水 + 槽位化归约）：

```text
1. 双缓冲队列搬入 2T 个 FP32：GM 地址 32B 对齐走 plain DataCopy 快路径，
   未对齐走 DataCopyPad；尾 tile 以 +Inf 位模式 padding，不参与最小值
2. Abs 整向量（交错，1 pass）→ DeInterleave 拆实虚（纯排列保 Abs 语义）→ Add 得 T 个模值
3. Compare(v!=v) 生成 NaN 掩码 → Select 将 NaN 置 +Inf（向量化清洗，见"特殊浮点值"节）
4. ReduceMin(..., true) 得 tile 内 (minVal, localIdx)，直写本 tile 的 32B 对齐 UB 槽位
5. 全部 tile 完成后 PipeBarrier(V->S)，一次性标量按字典序合并各槽位；
   某 tile 归约结果仍为 NaN 时（理论上不应发生）对该 tile 从 GM 标量重扫兜底
```

核内全部 tile 完成后：AIV kernel 将 `(bestVal, bestIdx)` 以 2 个 FP32（索引按位模式）经 `DataCopyPad` 写入本核 workspace 槽位（S->MTE3 屏障后搬出）；small kernel（单核直通）直接写 `result = bestIdx + 1`。`incx == 1 && n <= tileSize` 时 Host 强制单核 small 直通，省掉 reduce kernel 发射（小尺寸用例从 ~23us 降到 ~13us 量级以下）。

#### 跨步 kernel（`icamin_simt_kernel`）

`incx > 1` 时按 isamin 跨步路径同构实现，元素粒度为复数：

```text
for (t = threadIdx.x; t < calNum; t += blockDim.x):
    logical  = blockStartOffset + t              // 复数逻辑下标
    scalar   = 2 * logical * incx                // FP32 分量下标，UINT64 中间量
    m        = |x[scalar]| + |x[scalar + 1]|     // 逐分量标量读取
    m 非 NaN 且字典序更优则更新 (bestVal, bestIdx=logical)
```

每线程得到 `(bestVal, bestIdx)` 后在 block 内做 2 的幂树形归约（逐轮 `asc_syncthreads()`，比较规则同上），thread 0 写本核 workspace 槽位。线程数按 tiling 的 `nthreads` 配置。

#### 跨核二次归约（`icamin_reduce_kernel`）

单核执行：从 workspace 读入 `2*useCoreNum` 个 FP32（按 64 对齐），逐条按字典序规则合并（跳过 NaN 值），若无有效值则索引取 0；最终 `result = bestIdx + 1`，以 4 字节 `DataCopyPad` 写出。

### 预期文件变更

| 文件 | 变更 |
| --- | --- |
| `include/cann_ops_blas.h` | 新增 `aclblasIcamin` 声明（置于同族 `aclblasIsamin` 声明邻近位置） |
| `blas/iamin/arch35/icamin_host.cpp` | 新增：Host 校验、quick return、tiling、workspace、发射 |
| `blas/iamin/arch35/icamin_kernel.cpp` | 新增：连续/跨步/二次归约 kernel 与发射包装 |
| `blas/iamin/arch35/icamin_kernel.h` | 新增：Host 发射声明 |
| `blas/iamin/arch35/icamin_tiling_data.h` | 新增：`IcaminTilingData` 与 tile 常量 |
| `test/iamin/icamin/CMakeLists.txt` | 新增：测试目标注册（arch35） |
| `test/iamin/icamin/icamin_param.h` | 新增：CSV 参数映射（n / incx / x / expect_result / random_seed） |
| `test/iamin/icamin/icamin_golden.h` | 新增：cblas 风格 CPU golden（FP32 严格、NaN 跳过、并列取首） |
| `test/iamin/icamin/icamin_npu_wrapper.h` | 新增：device 分配/H2D/D2H 封装 |
| `test/iamin/icamin/arch35/icamin_test.cpp` | 新增：CSV 驱动 GTest + 负向用例 + 特殊值/重复调用用例 + 性能计时用例 |
| `test/iamin/icamin/arch35/icamin_test.csv` | 新增：任务配套 1200 条用例入库 |
| `test/iamin/icamin/arch35/icamin_test_extra.csv` | 新增：自补用例（正态分布、对齐偏移、大规模 tie、全 NaN） |
| `test/iamin/icamin/README.md` | 新增：测试步骤与口径说明（交付件要求可复现） |
| `test/frame/fill.h` | 修改：新增 `GAUSS` 正态分布填充模式（与 scnrm2 分支同源，纯增量不影响既有模式） |
| `blas/iamin/README.md` | 更新：产品支持表标注 Ascend 950PR 支持、icamin 接口与约束说明 |

`blas/CMakeLists.txt` 按目录 glob 自动收集 `blas/iamin/arch35/*.cpp`，预计不改动；测试构建经 `--ops=icamin` 由 `blas/*/arch35/icamin_*.cpp` glob 命中与家族目录搜索命中，已在设计阶段核对规则存在。

## 支持硬件

| 硬件 | 支持情况 | 说明 |
| --- | --- | --- |
| Ascend 950PR | 支持 | 本任务目标，CANN 9.1.0 |
| 其他产品线 | 保持现状 | 不修改 `arch22` 等既有实现；公共声明可供后续产品线接入 |

## 算子约束限制

- `x` 和 `result` 必须是当前 Device 可访问地址，`result` 至少有 4 字节空间；二者按独立对象处理，不涉及原地更新与视图语义。
- `n > 0` 且 `incx >= 1` 时，`x` 的物理复数长度至少为 `1+(n-1)*incx`。
- `incx < 1`（含 0 与负步长）为 quick return，不做反向遍历；`n < 0` 返回 `ACLBLAS_STATUS_INVALID_VALUE`。
- 输出为离散整数索引，语义上确定；不要求也不承诺跨版本的浮点级一致性之外的额外确定性保证。
- 调用成功只表示参数合法且工作已入队；读取 `result` 前调用者必须同步绑定 stream。
- 本任务不新增生产依赖，不改变公共 ABI（仅新增声明）。
- 配套测试材料约定单用例 Host 侧内存不超过 4 GB（向量 `n <= 2^24`）。

# 可维可测分析

## 精度标准/性能标准

| 类别 | 标准 |
| --- | --- |
| 整数索引输出 | `actual == golden`（EXPECT_EQ 精确一致），任一用例不相等即失败 |
| quick return | `n == 0` 或 `incx < 1` 时 `result == 0` 且返回成功 |
| NaN | 模为 NaN 的元素跳过；全 NaN 返回 1 |
| 性能 | 1M/2M/4M 分别不超过 24.77/24.59/29.69 us；其余 PF 用例不超过 `gpu_ms * 2.5`；warmup 后有效采样 >50 次 |
| 确定性 | 归约顺序不影响结果；逐用例精确通过，不以比例掩盖单例失败 |

## 精度分析

### Golden

- 测试工程内 cblas 风格 CPU 循环：严格 FP32（`fabsf` 级）计算 `|Re|+|Im|`；首元素模 NaN 时基准置 `FLT_MAX`；逐元素跳过 NaN、严格小于才更新索引；`n < 1` 或 `incx < 1` 返回 0。与任务书 §3.2 及 Q7 对齐结论一致。
- `ReduceMin` 带索引模式的 tile 内并列取首、NaN 处理为设备语义，以 950PR 实测用例（全零 tie、并列最小值、NaN 混合）固化，列入验证门禁。

### 精度用例矩阵

任务配套 CSV（1200 条，固定种子）分类与覆盖：L0 基础 6 条（含全零 tie 期望索引 1）、L1 尺寸扫描 38 条（1→2^20，质数/2 的幂/2 的幂±1/非对齐）、L2 步长 17 条（正步长计算 + 非正步长 quick return）、L5 填充 12 条（随机/全零/交替/极端值/Inf/NaN）、L6 边界 12 条（零维、负 n、空指针、quick return 优先组合）、EX 扩展 915 条、PF 性能 200 条。填充模式 `RANDOM_NORM_5_5 / VALUE_NORM_0 / RANDOM_ALTER / RANDOM_EXTREME / VALUE_NORM_INF / VALUE_NORM_NAN / NULLPTR` 均被仓内 `test/frame/fill.h` 支持，复数按 `2*xLen` 个 FP32 生成（实部虚部独立采样）。

按任务书 §3.5.4 自行补充的维度：

| 维度 | 覆盖 |
| --- | --- |
| 正态分布输入 | 任务书要求 x 值域为正态（μ∈[-5,5]，σ∈[0.1,2]）占 50%，配套 CSV 仅均匀分布；在测试框架或测试工程补充确定性正态生成器（实虚独立采样），补充用例并证明均匀/正态比例 |
| 对齐偏移 | x 起始地址做 4B/32B/64B 偏移的合法性用例 |
| tie 语义 | 多位置相同模（含跨 tile/跨核并列）取最小索引 |
| 特殊值组合 | 全 NaN、全 Inf、NaN+有限值、subnormal 分量、`|FLT_MAX|+|FLT_MAX|` 模溢出为 Inf 的行为固化 |
| 重复调用 | 同一 handle/stream 连续调用（含 quick return 与正常路径交替），确认 workspace 时序无污染 |
| 测试工程层负向 | handle 空指针、result 空指针（正常路径）期望状态码 |

## 性能分析

### 关键路径与实测结果（v1.1）

性能 case 均为 `incx=1` 连续大规模，输入数据量 `8*n` 字节（1M/2M/4M 分别 8/16/32 MiB），输出与 workspace 可忽略。

**实测（Ascend 950PR、CANN 9.1.0、Release 构建、同一环境）**:

| case | 平均单次耗时 | 门限（H100 × 2.5） | 余量比 |
| --- | ---: | ---: | ---: |
| n=1,048,576 | 9.03 us | 24.77 us | 0.36 |
| n=2,097,152 | 11.06 us | 24.59 us | 0.45 |
| n=4,194,304 | 16.93 us | 29.69 us | 0.57 |
| 200 条 PF 全量 | 全部通过 | 逐条 ×2.5 | 最差约 0.67 |

**性能工作的两个根因级结论**:

1. **构建类型是首要因素**：master 的 CMakeLists 强制 `Debug`，kernel 以 -O0 编译，4M case 达 147~169 us（超标约 5 倍）；改 Release 后同一实现 9~17 us 即达标。本分支默认 Release（scnrm2 分支同款守卫，`-D` 可覆盖）。Debug 下的逐 tile 成本分析（DMA/向量/同步占比）不适用于 Release。
2. **NaN 处理不能依赖归约指令语义**：`ReduceMin` 带索引模式 Debug 跳过 NaN、Release 传播 NaN；先以标量重扫兜底（全 NaN 4M 用例退化到 1649 us），最终定为归约前 `Compare+Select` 向量化清洗（全 NaN 4M 用例 20.0 us，与常规数据相当）。

其余结构措施：TQue 双缓冲 DMA 流水、tile 归约结果写 32B 对齐 UB 槽位、末尾一次性 `PipeBarrier(V->S)` 后标量合并（消除逐 tile `GetValue` 同步）、`incx==1 && n <= tileSize` 单核直通省 reduce kernel 发射、模计算先 `Abs` 交错向量再 `DeInterleave` 省一趟向量 pass。单条大 case 测量存在主机侧噪声（曾观测 31.6 us 离群值，复测 16.6~17.2 us），验收以稳定复测为准。

### 性能验证方法

1. 固定 Ascend 950PR、CANN 9.1.0、同一频率与运行环境。
2. 测试用例内预先创建 handle/stream、分配并填充输入输出。
3. warmup 不少于 3 次，有效采样 >50 次；计时区只包含重复发射与末尾同步（chrono 或 ACL event），计算单次平均 us，打印 `[ICAMIN_PERF] average_us` 供报告采集。
4. 计时区不生成 golden、不打印日志、不申请内存、不做 H2D/D2H。
5. msprof 复核 kernel 数量、GM→UB 流量与尾段耗时，仅作诊断。

### 性能回退策略

按优先级定位与优化：确认路径选择正确（small 直通/双发射）→ 核对 tileSize 与核数 A/B（单变量）→ 检查模计算指令（DINTLV/Abs/Add）的 Hardware Loop 与寄存器 spill → 分离 MTE2 供数与归约尾巴 → 小尺寸条目确认发射开销占比。任何回退不得改变字典序归约语义与单遍读取原则。

## 兼容性分析

- 仅新增公共接口声明，API 签名、`aclblasComplex` ABI、handle/stream 使用方式均不变。
- 新代码仅进入 `arch35`，不改变 `arch22` 等既有编译产物与行为。
- `n < 0` 返回 `ACLBLAS_STATUS_INVALID_VALUE` 为仓内同族约定，与 cuBLAS 的差异已在任务书层面确认；负步长 quick return 不遍历是本族语义，不影响 axpy/nrm2 族。
- result 为 Device INT32 scalar；同步责任与 ops-blas 其他句柄式接口一致。

## 可维护性分析

- Host 校验、tiling、kernel 算法与测试 golden 分文件；复用 isamin 的命名与目录结构，降低族内维护成本。
- tiling 字段统一以复数元素为单位，注释明确 `n` 与 `2*n` 的换算点，防止单位混用。
- 新增测试进入现有 CMake/CSV 流程，不依赖任务附件脚本才能运行；官方脚本原件保留作可追溯输入。
- 性能数据记录设备、软件版本、采样方法、日期与原始日志；case 主键含完整参数。

## 交付件验收矩阵

| 交付件 | 内容 | 完成条件 |
| --- | --- | --- |
| 社区设计文档 | 按 tasklist 规则提交 `docs/design.md` PR 至 cann-competitions | 评审通过并合入 |
| 算子与测试代码 | 公共声明、arch35 Host/Kernel、CSV/GTest 测试工程、iamin README | 个人 fork 明确分支可复现构建和测试 |
| 自测报告 | 全量精度 case、三项性能 case、内存占用、截图与原始日志 | 使用任务指定模板，数据可追溯 |
| 待验收地址 | 个人仓 URL、分支 `feat/aclblas-icamin-arch35`、算子/测试目录 | 邀请 `Ascend-CANN` 为开发者 |

任务书一处写"内存要求不涉及"，交付件又要求内存占用数据；本设计按更严格口径在自测报告中记录峰值 Host/Device/workspace 占用。

## 工程实现与验收门禁（v1.1 全部关闭）

| 工程项 | 影响 | 关闭情况 |
| --- | --- | --- |
| `ReduceMin` 带索引的设备语义 | tile 内并列取首索引已实测（全零 tie、跨核 tie 用例通过）；NaN 行为随优化级别变化（Debug 跳过/Release 传播） | 已关闭：并列取首成立；NaN 改为归约前 `Compare+Select` 向量化清洗 + 标量重扫兜底，不再依赖指令 NaN 语义 |
| 模计算指令选择 | 采用整向量 `Abs` + `DeInterleave` + `Add`（FP32 单精度链路） | 已关闭：全量精度 bit-exact 通过 |
| quick return 写零方式 | isamin 同款 Host 直写 device 指针在 950PR 可用 | 已关闭：quick return 用例全过 |
| 性能计时口径 | 测试工程内计时循环（warmup 3 + 51 次）打印 `average_us`；官方 `verify_performance.py` 原件仅作粗筛 | 已关闭：200 条 PF 全部按 `gpu_ms × 2.5` 达标；主机噪声离群值需复测确认 |
| 正态分布生成器 | `test/frame/fill.h` 新增 `GAUSS`（scnrm2 分支同源），34 条自补正态用例 | 已关闭：自补 48 条全部通过 |
| baseline 元数据 | `gpu_baseline.csv` 按附件原值引用，判定公式 `×2.5` 已核对（前 3 条与任务书 §3.3 完全一致） | 已关闭 |
| 构建类型（新增） | master 强制 Debug 致 kernel -O0 劣化约 10 倍 | 已关闭：分支默认 Release（可 -D 覆盖），性能数据均为 Release 实测 |

# 参考资料

1. [社区任务设计文档模板](https://gitcode.com/cann/cann-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md)
2. [2026 社区任务参与及 PR 流程](https://gitcode.com/cann/cann-competitions/blob/master/04_tasks/01_community-task-2026/README.md)
3. [cann/ops-blas](https://gitcode.com/cann/ops-blas)
4. [cuBLAS cublasIcamin API](https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-amin)
5. [Netlib isamin.f（标准参考 BLAS 无 icamin 例程）](https://www.netlib.org/blas/isamin.f)
6. [生态算子开源精度标准](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md)
7. 《aclblasIcamin 算子开发任务书》及配套测试材料（随任务下发）
