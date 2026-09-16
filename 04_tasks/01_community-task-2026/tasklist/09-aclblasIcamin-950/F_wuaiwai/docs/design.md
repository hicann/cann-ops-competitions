# aclblasIcamin 算子设计文档

# 需求背景（required）

## 需求来源

CANN 社区任务 2026：在昇腾 NPU（Ascend 950PR）上使用 Ascend C 编程语言开发单精度复数（complex64）向量最小模元素索引算子 `aclblasIcamin`，完成算子设计、开发、测试全流程，验收通过后合入昇腾算子开源仓 ops-blas（https://gitcode.com/cann/ops-blas ）。

## 背景介绍

### BLAS icamin 功能与生态现状

icamin 是 BLAS 一级（level-1）向量归约例程：在单精度复数向量中查找**模最小元素的下标**。cuBLAS 提供对应接口 `cublasIcamin`，广泛应用于科学计算与数值库（如迭代求解器中的向量归约、缩放因子选取）。标准参考 BLAS（Netlib）**没有 icamin 例程**（仅有实数版本 isamin），因此语义细节以 cuBLAS 官方文档为最高优先级来源。

当前 ops-blas 开源仓已具备同族实数接口 `aclblasIsamin`（含 arch35 实现，目录 `blas/iamin/arch35/`），但 `include/cann_ops_blas.h` 中尚无复数版本 `aclblasIcamin` 的声明，复数最小模索引能力缺失。本任务新增该接口，补齐复数 BLAS 能力拼图。

### 对标接口语义（cuBLAS cublasIcamin）

数学表达式：`result = argmin_i (|Re(x[k])| + |Im(x[k])|)`，其中 i = 1..n，k = 1+(i-1)·incx。关键语义：

1. 复数"模"按 BLAS icamin 惯例定义为 `|Re| + |Im|`（曼哈顿范数分量，**非欧几里得模**）；
2. 返回 **1-based 索引**（兼容 Fortran 惯例）；
3. 多个元素模相同时，返回**最小索引**；
4. n ≤ 0 或 incx ≤ 0 时不做计算（本任务细化为：n = 0 或 incx < 1 走 quick return 写 result = 0 返回成功；n < 0 返回参数错误，语义对齐仓内 aclblasIsamin 的 README 约束与 arch35 实现）。

### 复数类型定义

复数类型 `aclblasComplex` 以 ops-blas 仓 `include/cann_ops_blas_common.h` 中定义为准：实部/虚部各 float32，交错存储，n 个复数元素对应 2·n 个 float。

# 需求分析（required）

## 需求描述

使用 Ascend C 编程语言，基于 ops-blas 开源仓工程框架实现句柄式 BLAS 接口 `aclblasIcamin`：输入单精度复数（COMPLEX64）向量 x（长度 n，步长 incx），输出最小模元素的 1-based 整数索引（INT32 标量，Device 内存）。功能、参数语义与 cuBLAS `cublasIcamin` 完全对齐，性能不低于任务书给定标杆。

## 需求拆解

1. **接口对齐**：签名与 `cublasIcamin` 逐参数对齐，声明新增至 ops-blas 仓 `include/cann_ops_blas.h`，供各产品线共用，禁止定义 950PR 私有平行接口；
2. **语义正确性**：
   - 模定义 `|Re| + |Im|`；1-based 索引；同模取最小索引；
   - quick return：n = 0 或 incx < 1（含 0 与负步长）不触发 kernel，直接写 result = 0 并返回 `ACLBLAS_STATUS_SUCCESS`；
   - 参数校验：n < 0、x 为 nullptr（n > 0 且 incx ≥ 1 时）、result 为 nullptr 返回 `ACLBLAS_STATUS_INVALID_VALUE`；handle 为 nullptr 返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`；
   - NaN 语义：golden 为跳过 NaN 元素（首元素为 NaN 时基准置 FLT_MAX），实现须对齐（对应任务事实表开放问题 Q7，待研发确认后冻结）；
3. **精度**：输出为离散整数索引，与 CPU 参考 golden（cblas 风格循环）做 **bit-exact 精确比对**；
4. **性能**（COMPLEX64 输入，Avg time， warmup 后有效采样 >50 次取平均）：

   | case | n | incx | 标杆耗时（us） |
   |---|---|---|---|
   | 1 | 1048576 | 1 | ≤ 24.77 |
   | 2 | 2097152 | 1 | ≤ 24.59 |
   | 3 | 4194304 | 1 | ≤ 29.69 |

5. **测试**：参照仓内 `test/isamin/` 模式新建 CSV 驱动的 GTest 测试工程，覆盖任务书 §3.5 全部场景。

# 详细设计（required）

## 算子分析

### 数学公式

```
result = argmin_i ( |Re(x[k])| + |Im(x[k])| ),  i = 1..n,  k = 1 + (i-1)·incx
```

- result ∈ [1, n]，1-based；quick return 时 result = 0
- 比较规则：严格小于才更新最优值，保证同模时取最小索引

### 支持数据类型

| 参数 | 数据类型 | 说明 |
| --- | --- | --- |
| x（输入） | COMPLEX64（`aclblasComplex`，实部/虚部各 FLOAT32，交错存储） | Device 内存，只读 |
| result（输出） | INT32 | Device 内存，单值标量 |
| n、incx | int | Host 内存标量 |

### 支持形状

- 逻辑一维向量 [n]，物理存储长度 1+(n-1)·|incx|（incx ≥ 1 参与计算；incx < 1 走 quick return）
- 不涉及 broadcast、不涉及非连续 tensor 扩展语义、不要求 dynamic shape（n 为运行时入参）

## 算子实现

### 实现方案

整体采用 **Ascend C kernel 直调 + 两级归约（块内归约 → 块间归并）** 方案，实现代码放在 `blas/iamin/arch35/`，与同族实数接口 `aclblasIsamin` 同目录，最大化复用其 arch35 归约实现框架（分核策略、tiling 结构、workspace 组织方式）。

#### host 侧设计

1. **参数校验**（先于一切）：
   - handle 为 nullptr → 返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`；
   - n < 0 → 返回 `ACLBLAS_STATUS_INVALID_VALUE`；
   - n > 0 且 incx ≥ 1 时 x 为 nullptr、或 result 为 nullptr → 返回 `ACLBLAS_STATUS_INVALID_VALUE`；
2. **quick return**：n = 0 或 incx < 1（含 0、负步长）时，不触发 kernel，通过 handle 绑定的 stream 直接向 result 写 0，返回 `ACLBLAS_STATUS_SUCCESS`（负步长不做反向遍历）；
3. **分核与 tiling 策略**：
   - 通过平台接口获取 AIV 核数，优先满核；逻辑元素数 n 按核均分，不能均分时余数分配到前几个核（与仓内 isamin arch35 策略一致）；
   - 每核负责一段连续的逻辑下标区间 [start, end)，保证块间归并时下标有序、同模取最小索引的规则可正确传递；
   - 计算各核物理地址偏移：base = (start)·incx·sizeof(aclblasComplex)，物理跨度 = (len-1)·incx+1 个复数元素；
4. **workspace 组织**：每核输出一个 `(minVal: float, minIdx: int32)` 对到 workspace（核数 ≤ 数百，块间归并量极小）；空块（len = 0）写占位值（FLT_MAX, INT32_MAX），归并时自然被淘汰；
5. **tilingKey 规划**：
   - tilingKey = 0：incx = 1，物理连续，走整块搬入 + 顺序扫描主路径（性能 case 全部命中此路径）；
   - tilingKey = 1：incx > 1，走步长 gather 路径（按 stride 索引访问或 DataCopy stride 搬运）；
   - host 侧依据 incx 选择 tilingKey，kernel 注册入口与 tilingKey 严格一一对应。

#### kernel 侧设计

1. **CopyIn**：incx = 1 时按 Tile 块连续搬入复数数据（2·len 个 float），充分使用 UB 并开启 double buffer 隐藏搬运延迟；incx > 1 时按 stride 参数搬运或标量 gather；
2. **Compute（块内归约）**：
   - 对每个复数元素计算模 m = |Re| + |Im|（Abs + Add，可全向量化）；
   - 维护当前最优对 (bestVal, bestIdx)：仅当 m < bestVal（严格小于）时更新，保证同模取最小索引；
   - 向量化实现时按向量 lane 求每 lane 的 (val, idx) 序列最小，再在标量侧做 lane 间归并，归并同样遵循"严格小于更新"规则；
   - NaN 处理：对齐 golden 语义跳过 NaN 元素（首元素为 NaN 时基准置 FLT_MAX），与 |Re|+|Im| 含 NaN 时比较恒为假的行为天然兼容，需在实现中显式验证该路径；
3. **块间归并**：各核 (minVal, minIdx) 写 workspace 后，由主核（或第二趟小 kernel）串行/树形归并，同样按"严格小于更新、同模保小索引"规则得到全局 (val, idx)，将 idx 写回 result（1-based，块内下标 + 块起始偏移 + 1 计算）；
4. **结果确定性**：输出为离散整数索引，|Re|+|Im| 的比较在 IEEE 浮点语义下结果确定，归约顺序不影响最终索引值，天然满足 bit-exact 判定。

### 实现路径

- 接口声明：`include/cann_ops_blas.h`（新增，当前无 `aclblasIcamin` 声明）
- 实现代码：`blas/iamin/arch35/`（复用 aclblasIsamin 同目录框架）
- 测试工程：`test/iamin/icamin/arch35/`（参照 `test/isamin/` 模式，CSV 列格式对齐 `test/isamin/isamin_param.h`：n / incx / x 填充模式，x 列写 `NULLPTR` 表示空指针负向用例）

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Ascend 950PR | √ |

## 算子约束限制

1. n ≥ 0（n < 0 返回 `ACLBLAS_STATUS_INVALID_VALUE`）；n = 0 或 incx < 1 为合法 quick return（result 置 0，返回成功）；
2. 负步长不做反向遍历，直接 quick return；
3. 不支持非连续 tensor 扩展语义（向量步长由 incx 表达，无 leading dimension padding 场景）；
4. 不涉及 broadcast、原地更新与视图语义；result 为独立输出标量；
5. 异步执行依赖 `aclblasSetStream` 绑定 stream，读回 Device 结果前须同步 stream；
6. 特殊值：输入含 Inf 时按 |Re|+|Im| 正常比较；含 NaN 时跳过 NaN 元素（待 Q7 研发确认后冻结）。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述(不涉及说明原因) | 标准来源 |
| --- | --- | --- |
| 精度标准 | 输出 INT32 整数索引与 CPU 参考 golden（cblas 风格循环，\|Re\|+\|Im\| 逐元素比较、严格小于取最小索引）**bit-exact 精确一致**（EXPECT_EQ）；输入分量按 FLOAT32 档（生态算子开源精度标准），浮点阈值对整数索引输出退化为精确一致判定；quick return 用例判定 result == 0 | cuBLAS 官方文档 + 生态算子开源精度标准（opbase 仓）+ 任务书 §3.2 |
| 性能标准 | COMPLEX64 输入，Avg time（warmup 后有效采样 >50 次取平均）：n=1048576 ≤ 24.77us；n=2097152 ≤ 24.59us；n=4194304 ≤ 29.69us | 任务书 §3.3 标杆 |
| 内存标准 | 不涉及（任务书 §3.4） | 任务书 |

自验方式：CSV 驱动 + C++ GTest 工程（参照 `test/isamin/`），golden 由测试工程内 CPU 参考实现生成；用例覆盖 0/1/小质数/2 的幂及 ±1/≥2^20 大规模 n、正步长 1/2/3、incx=0 及负步长 quick return、负 n/空指针负向用例、全零/正负交替/极端值/Inf/NaN 特殊值，及三个性能 case。

## 兼容性分析

新增接口（ops-blas 仓头文件当前无 `aclblasIcamin` 声明），不涉及存量兼容性问题；接口声明放入 `include/cann_ops_blas.h` 供各产品线共用，不定义 950PR 私有平行 API。
