# aclblasIcamin 算子设计文档（Ascend 950PR）

| 项目 | 内容 |
| --- | --- |
| 任务名称 | 9月社区任务-aclblasIcamin算子开发（950） |
| 参与者 | `weixin_51828836` |
| 目标硬件 | Ascend 950PR（arch35） |
| CANN 版本 | CANN 9.1.0 |
| 代码仓 | `cann/ops-blas` |
| 实现目录 | `blas/iamin/arch35/` |
| 对标接口 | cuBLAS `cublasIcamin` |

# 需求背景（required）

## 需求来源

ops-blas 当前已提供 FP32 实数向量最小绝对值索引接口 `aclblasIsamin`，但缺少对应的单精度复数接口。本任务在 Ascend 950PR 上使用 Ascend C kernel 直调方式实现 `aclblasIcamin`，接口、参数顺序和核心计算语义与 `cublasIcamin` 对齐，并复用仓内 iamin 族的 Host 分派与分层归约框架。

新增公共接口声明：

```cpp
aclblasStatus_t aclblasIcamin(
    aclblasHandle_t handle,
    int n,
    const aclblasComplex* x,
    int incx,
    int* result);
```

接口声明放入 `include/cann_ops_blas.h`，不增加仅供 Ascend 950PR 使用的私有平行接口。

## 背景介绍

`aclblasIcamin` 查找 COMPLEX64 向量中 1-范数模最小元素的索引。复数模采用 BLAS icamin 约定的 `|Re| + |Im|`，不是欧几里得模。输出采用 BLAS/Fortran 惯例，为 1-based INT32 索引；模相同时返回最小逻辑索引。

仓内 `aclblasIsamin` 已具备 arch35 多核分片、核内最小值归约、workspace 跨核归约及非连续步长处理框架。本实现保留这些成熟的结构，在连续路径中增加 COMPLEX64 解交织和 `|Re| + |Im|` 计算，并针对任务书给出的百万级连续向量性能门槛设计专用路径。

# 需求分析（required）

## 需求描述

对逻辑索引 `i = 0...n-1`，物理元素位置为 `i * incx`，计算：

```text
abs1(i) = abs(real(x[i * incx])) + abs(imag(x[i * incx]))
result = first_argmin(abs1) + 1
```

功能要求如下：

1. 输入为 COMPLEX64，实部和虚部均为 FP32，按 `real0, imag0, real1, imag1, ...` 交错存储。
2. 输出为单个 INT32 Device 标量，正常结果范围为 `[1, n]`。
3. 使用严格小于更新最优值；模相同时保持更小逻辑索引。
4. `n < 0` 返回 `ACLBLAS_STATUS_INVALID_VALUE`。
5. `n == 0` 或 `incx < 1` 时不启动自定义 kernel，将 Device 侧 `result` 异步置零并返回成功。
6. 通过 handle 绑定的调用方 stream 异步执行，不创建私有 stream，不进行 Host 侧强制同步。

## 需求拆解

| 编号 | 子项 | 验收要点 |
| --- | --- | --- |
| R1 | 公共接口 | `include/cann_ops_blas.h` 新增 `aclblasIcamin` 声明 |
| R2 | Host 实现 | 参数校验、quick return、分核、tiling、workspace 校验和 kernel 分派 |
| R3 | 连续 Kernel | COMPLEX64 解交织、`Abs+Add`、核内 argmin、跨核归约 |
| R4 | 通用 Kernel | 支持 `incx > 1` 的 SIMT 步长访问 |
| R5 | 确定性 | 相等模值取最小索引，输出与 CPU golden bit-exact |
| R6 | 边界语义 | n、incx、空指针、NaN、Inf、全零和 quick return |
| R7 | 性能 | 三个任务书硬门槛及任务包 200 条性能用例逐项记录 |
| R8 | 工程交付 | README、CSV 驱动 GTest、复现命令和自测报告 |

## 输入输出规格

| 参数 | 方向 | 类型 | 内存位置 | 说明 |
| --- | --- | --- | --- | --- |
| handle | 输入 | `aclblasHandle_t` | Host | 携带 stream 与公共 workspace |
| n | 输入 | `int` | Host | 复数逻辑元素个数，要求 `n >= 0` |
| x | 输入 | `const aclblasComplex*` | Device | 逻辑长度 n，物理长度 `1 + (n - 1) * incx` |
| incx | 输入 | `int` | Host | 正常路径要求 `incx >= 1` |
| result | 输出 | `int*` | Device | 正常返回 1-based 索引，quick return 返回 0 |

## 参数校验顺序

| 条件 | 返回与行为 |
| --- | --- |
| `handle == nullptr` | `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| `n < 0` | `ACLBLAS_STATUS_INVALID_VALUE` |
| `result == nullptr` | `ACLBLAS_STATUS_INVALID_VALUE` |
| `n == 0 || incx < 1` | 使用 handle stream 对 result 执行 4 字节异步清零，返回 SUCCESS |
| 正常路径且 `x == nullptr` | `ACLBLAS_STATUS_INVALID_VALUE` |

quick return 不读取 x，因此允许 x 为空；result 为 Device 指针，Host 侧不得直接解引用，使用 `aclrtMemsetAsync` 保持异步语义。运行时清零失败时返回执行失败状态。

# 详细设计（required）

## 算子分析

### 数学公式

设 `x_i = a_i + b_i j`，则：

```text
m_i = |a_i| + |b_i|
r = min { i | m_i = min(m_0, ..., m_(n-1)) }
result = r + 1
```

比较候选时使用二元键 `(m_i, i)`：先比较模值，再比较逻辑索引。该比较规则在不同分核及归约顺序下均得到相同结果。

### 数据类型与形状

| 项目 | 规格 |
| --- | --- |
| 输入类型 | COMPLEX64（两个 FP32 分量） |
| 输出类型 | INT32 |
| 输入形状 | 逻辑一维 `[n]` |
| 输出形状 | 单值标量 |
| 步长 | `incx >= 1`；性能路径为 `incx == 1` |
| broadcast | 不涉及 |
| 原地计算 | 不涉及 |

### 特殊浮点值

CPU golden 以 FP32 计算 `|Re| + |Im|`，严格小于时更新索引。NaN 不参与后续最小值更新；逻辑首元素为 NaN 时以 `FLT_MAX` 作为初始比较基准并保留索引 0。Inf 按 IEEE FP32 比较关系参与计算。Kernel 的初始候选、NaN 有效标记和 tie-break 规则与 golden 保持一致，另增加首元素 NaN、全 NaN、NaN/Inf/FLT_MAX 混合用例，防止使用普通数值哨兵造成错误更新。

## 算子实现

### 总体流程

```text
aclblasIcamin
  -> ValidateIcaminParams
  -> quick return 或 CalcIcaminTiling
  -> 检查 handle workspace
  -> 第一阶段：每个 AIV block 计算局部 (minValue, minIndex, valid)
  -> 第二阶段：单 block 合并各核候选并写回 1-based result
```

目录规划：

```text
include/cann_ops_blas.h
blas/iamin/README.md
blas/iamin/arch35/icamin_host.cpp
blas/iamin/arch35/icamin_kernel.cpp
blas/iamin/arch35/icamin_kernel.h
blas/iamin/arch35/icamin_tiling_data.h
test/iamin/icamin/icamin_golden.h
test/iamin/icamin/icamin_npu_wrapper.h
test/iamin/icamin/icamin_param.h
test/iamin/icamin/arch35/icamin_test.cpp
test/iamin/icamin/arch35/icamin_test.csv
```

### Host 侧设计

#### 分核策略

通过 `GetAivCoreCount()` 获取可用 AIV 数量。连续大向量优先使用全部可用 AIV：

```text
useCoreNum = min(n, aivCoreNum)
baseCount = n / useCoreNum
remainder = n % useCoreNum
```

前 `remainder` 个核各处理 `baseCount + 1` 个元素，其余核处理 `baseCount` 个元素。每个核的起点通过前缀公式计算，避免只让最后一核承担全部余数。

#### Tiling 数据

Tiling 至少包含：

```cpp
struct IcaminTilingData {
    uint32_t totalN;
    uint32_t baseCount;
    uint32_t remainder;
    uint32_t useCoreNum;
    uint32_t tileComplexCount;
    uint32_t simtThreads;
    uint32_t incx;
};
```

连续路径的 `tileComplexCount` 按 UB 容量、32 字节搬运对齐及 ReduceMin repeat 上限确定。单 tile 同时容纳交错输入、实部、虚部、模值和归约临时区；性能调优阶段在 4K/8K/16K 复数元素候选中实测选择。通用路径线程数按单核元素数动态计算，并限制在硬件允许范围内。

#### Workspace

每核只写一个局部候选，workspace 保存模值、0-based 全局逻辑索引和有效标记，按 32/256 字节边界对齐。空间复杂度为 `O(useCoreNum)`，不随 n 线性增长。Host 在 launch 前使用 `GetEffectiveWorkspaceSize()` 校验容量，不足时返回执行失败。

### Kernel 侧设计

#### 连续向量路径（incx == 1）

性能用例全部是连续访问，因此该路径为主要优化对象：

1. 通过 `DataCopy`/`DataCopyPad` 将交错 COMPLEX64 数据从 GM 搬入 UB。
2. 使用 arch35 `DeInterleave` 将交错数据拆分为 FP32 实部和虚部向量。
3. 分别执行 `Abs`，再用 `Add` 生成 `|Re| + |Im|` 模值向量。
4. 显式生成 NaN 有效掩码，保证 NaN 不作为局部候选；初始候选语义与 CPU golden 一致。
5. 使用 `ReduceMin(..., calIndex=true)` 得到 tile 内最小值及最小局部索引。
6. 使用 `(value, globalIndex)` 比较键合并本核多个 tile；严格小于更新，相等时取较小索引。
7. 将本核候选一次写入 workspace。

输入搬运与向量计算使用双缓冲流水，在当前 tile 计算时预取下一 tile，减少大向量场景的 MTE/VEC 串行等待。尾块使用 padding 和有效长度控制，padding 值不得进入候选范围。

#### 非连续通用路径（incx > 1）

复用 `aclblasIsamin` arch35 的 SIMT 分层归约结构。每个线程按逻辑索引读取一个复数元素：

```text
physicalFloatOffset = logicalIndex * incx * 2
value = abs(x[offset]) + abs(x[offset + 1])
```

线程先计算局部候选，再在 UB 中进行 block 内树归约。合并规则同时比较模值、逻辑索引和有效标记。每个 block 最终只写一个候选，随后进入统一的跨核归约。

#### 小尺寸路径

当 n 可由单核单 tile 处理时，启动单个 AIV kernel 完成模值计算和 argmin，并直接写 result，跳过 workspace 第二阶段，降低小 shape 的 kernel launch 开销。该分支与多核路径共享相同的比较函数和 NaN 语义。

#### 跨核归约

第二阶段只处理 `useCoreNum` 个候选。单 AIV 将候选搬入 UB 后顺序或向量化合并：

```text
candidate wins when
  candidate is valid, and
  (best is invalid
   or candidate.value < best.value
   or (candidate.value == best.value and candidate.index < best.index))
```

最终将 0-based 索引加一并写入 Device result。所有 kernel 均在 handle stream 上顺序提交，不增加 Host 同步。

### 性能优化方案

任务书的三项硬门槛为：

| n | incx | 平均耗时上限（us） |
| ---: | ---: | ---: |
| 1048576 | 1 | 24.77 |
| 2097152 | 1 | 24.59 |
| 4194304 | 1 | 29.69 |

针对该门槛采用以下优化：

1. `incx == 1` 与通用 stride 分离，连续路径不在循环内执行 stride 乘法。
2. 所有 AIV 连续读取各自连续区间，保证 GM burst 访问和跨核带宽利用率。
3. 模值只存在于 UB，不写回 GM；每核仅写一个固定大小候选。
4. `DeInterleave + Abs + Add + ReduceMin` 采用大 tile 和双缓冲，减少搬运次数并重叠 MTE/VEC。
5. 大 shape 固定为两次 kernel launch；跨核归约数据量仅为核数级，控制尾部开销。
6. 通过 4K/8K/16K tile、使用核数、双缓冲开关进行 A/B 实测，以三项硬门槛全部通过作为参数选择条件，不以汇总平均掩盖单项失败。
7. 性能采样先 warmup，再执行不少于 100 次有效采样，逐项保存 avg/min/max 和原始日志。

## 支持硬件

| 芯片版本 | 支持情况 |
| --- | --- |
| Ascend 950PR（arch35） | 支持 |
| 其他产品 | 本任务不要求 |

## 算子约束限制

1. 仅支持 COMPLEX64 输入和 INT32 索引输出。
2. `n < 0` 为非法参数；`n == 0` 为 quick return。
3. `incx < 1` 为 quick return，不执行负步长反向遍历。
4. 不涉及 broadcast、原地更新和视图语义。
5. 正常路径要求 x、result 为有效 Device 指针。
6. 调用方读取 result 前须同步 handle 绑定的 stream。

# 可维可测分析

## 精度标准/性能标准

| 验收项 | 判定标准 | 标准来源 |
| --- | --- | --- |
| 功能 | 返回码与任务书定义一致 | 社区任务书 |
| 精度 | NPU INT32 索引与 CPU golden 精确相等 | 社区任务书、生态算子精度标准 |
| 性能 | 三项硬门槛逐项通过，200 条性能用例逐项记录 | 社区任务书及任务包 |
| 异步 | 使用调用方 stream，无 Host 强制同步 | 社区任务书 |

## 正确性测试方案

参照 `test/isamin/` 建立 CSV 驱动 GTest。官方 `icamin_test.csv` 共 1200 条，其中 1000 条为功能/精度用例、200 条为性能用例。精度用例逐条输出 case 名、参数、返回码、golden 和 actual，不只输出汇总数字。

覆盖范围：

- n=0、1、小质数、2 的幂及其正负一、非对齐尺寸和百万级尺寸；
- incx=1/2/3，以及 0/-1/-2/-3 quick return；
- 均匀分布、正态分布、全零、正负交替、极值、Inf 和 NaN；
- 相同模值、跨 tile 相同模值和跨核相同模值，验证最小索引；
- handle、x、result 空指针，负 n，以及 quick return 不读取 x；
- 首元素 NaN、全 NaN、NaN/Inf/FLT_MAX 混合的定向回归。

## 性能测试方案

设备固定为 Ascend 950PR，CANN 固定为 9.1.0。使用 ACL Event 统计 handle stream 上的算子执行耗时，warmup 后有效采样大于 50 次，计划使用 100 次。报告必须给出三项硬门槛及任务包 200 条性能用例的逐项数据，不只给出“全部通过”或平均倍率。

对每个性能 case 保存：

```text
case_name, n, incx, warmup, iterations,
avg_us, min_us, max_us, baseline_us, result
```

性能调优后重新执行完整精度矩阵，避免连续路径优化破坏 NaN、tie 或尾块行为。

## 内存与稳定性测试

任务书未设置额外内存门槛，但验收报告仍记录输入、输出、workspace 大小和运行前后 Device 内存变化。连续执行不少于 1000 次，检查输出稳定、无内存增长、无越界和异步错误。最大输入下 workspace 仍为核数级固定空间。

## 可复现性

算子 README 和自测报告记录以下信息：

- NPU 型号、Driver、Firmware、CANN 版本；
- ops-blas commit SHA、构建命令和安装路径；
- 官方 CSV SHA256、测试命令和完整日志；
- 三项硬性能及 200 条性能明细；
- 设计文档 PR、代码 PR 和待验收分支链接。

## 兼容性分析

`aclblasIcamin` 为新增公共接口，不修改既有 `aclblasIsamin` 的签名和行为。实现仅加入 arch35 构建目标，与 iamin 族共享代码目录但使用独立的 Host、Kernel 和 tiling 符号，避免符号冲突。测试工程按仓内现有 CSV/GTest 模式接入，不引入新的三方运行时依赖。

## 风险与控制

| 风险 | 控制措施 |
| --- | --- |
| 旧任务材料中的性能阈值与本任务书不同 | 只采用本次任务书的 24.77/24.59/29.69 us，逐项判定 |
| 复数模误写为欧几里得模 | golden 与 kernel 均明确使用 `abs(real)+abs(imag)` |
| 分块归约在 tie 时返回错误索引 | 比较键包含全局逻辑索引，增加跨 tile/跨核 tie 用例 |
| NaN 或数值哨兵污染 argmin | 使用显式有效标记并增加 NaN/Inf/FLT_MAX 混合回归 |
| quick return 直接解引用 Device result | 使用 stream 上的异步 Device 清零 API |
| 仅汇总性能导致遗漏失败 case | 保存并提交全部 200 条逐项性能记录 |
