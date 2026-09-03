# aclblasIcamin 算子设计文档

> 任务：8月社区任务 - aclblasIcamin算子开发（950）
> 适配硬件：Ascend 950PR ｜ CANN：9.1.0 ｜ 代码仓：cann/ops-blas（`blas/iamin/arch35/`）
> 个人仓：https://gitcode.com/W8FXN182F/ops-blas（分支 feat-icamin）｜ 设计者：W8FXN182F

---

## 一、需求背景（required）

### 需求来源

昇腾开源算子仓（ops-blas）对齐 cuBLAS 的 BLAS 级算子族。当前仓内已有同族**实数**接口 `aclblasIsamin`（FP32 向量最小绝对值索引），缺少对应的单精度**复数**接口 `aclblasIcamin`。本任务在 `blas/iamin/arch35/` 下新增 `aclblasIcamin`，与 cuBLAS `cublasIcamin` 参数语义完全对齐。

### 背景介绍

BLAS Level-1 的 i?amin 族接口用于查找向量中模最小的元素，返回 1-based 索引，常用于稀疏矩阵预处理、数值优化中选取主元等场景。复数版本的"模"按 BLAS icamin 惯例定义为 **|Re| + |Im|**（非欧几里得模），单精度即输入 COMPLEX64、分量按 FLOAT32 处理。

## 二、需求分析（required）

### 需求描述

使用 Ascend C 语言实现 `aclblasIcamin`：

```
result = argmin_i ( |Re(x[k])| + |Im(x[k])| ),   k = 1 + (i-1)*incx,  i = 1..n
```

- 输入 `x` 为 COMPLEX64 复数向量（实部/虚部各 float32，交错存储，n 个复数对应 2n 个 float）；
- 输出 `result` 为单个 INT32 整数标量：**1-based 索引**（兼容 Fortran 惯例）；
- 多个元素模相同时返回**最小索引**（整数索引精确比对，bit-exact）；
- 参数语义与 `cublasIcamin` 逐参数对齐，句柄式接口绑定 stream 直调 NPU kernel（Ascend C kernel 直调方式）。

### 需求拆解

1. 接口声明新增至 ops-blas 仓 `include/cann_ops_blas.h`（当前头文件无该声明），禁止自定义 950PR 私有平行接口；
2. 实现代码放 `blas/iamin/arch35/`（与同族实数接口 `aclblasIsamin` 同目录，复用其 arch35 归约实现框架）；
3. 参数合法性校验与 quick return 语义与 isamin 对齐；
4. 精度：输出为整数索引，按 `EXPECT_EQ` **精确比对**（bit-exact）；
5. 性能：n = 1M/2M/4M（incx=1）COMPLEX64 场景平均单次耗时不高于任务书 §3.3 标杆耗时；
6. 提供 CSV 驱动 GTest 测试工程 `test/iamin/icamin/arch35/`，覆盖 1200 条自测用例。

## 三、详细设计（required）

### 算子分析

#### 数学公式

```
result = argmin_i (|Re(x[k])| + |Im(x[k])|),   k = 1 + (i-1)*incx,  i = 1..n
```

模定义为曼哈顿模（|Re|+|Im|，绝对向量模），逐元素比较时：

- 值更小者胜出；模相同（IEEE 相等）时取**最小 1-based 索引**；
- NaN 元素逐元素跳过：首**有效**元素无条件播种最小基准，其后 NaN 一律不参与比较（kernel 与 golden 统一语义，事实表 Q7 定项为逐元素跳过）；
- |Re|+|Im| 的运算在 IEEE FLOAT32 语义下确定（Abs 精确、Add 正确舍入），NPU 与 CPU golden 结果逐位一致，因此索引判定支持 bit-exact。

#### 支持数据类型

| 项目 | 数据类型 | 说明 |
| --- | --- | --- |
| x | COMPLEX64（aclblasComplex，实/虚部 float32 交错） | Device 内存，只读 |
| result | INT32 标量 | 0（quick return）或 [1, n]，Device 内存 |
| n / incx | int（Host 内存） | n≥0；incx≥1 正常计算 |

#### 支持形状

- x 逻辑一维 `[n]`，物理长度 `1 + (n-1)*|incx|` 个复数元素（2 倍 float）；
- dynamic shape：不要求，n 为运行时入参（tiling 按 n 动态计算）；
- 非连续访问由 incx 表达，负步长（incx<1）为合法 quick return，不做反向遍历。

### 算子实现

#### 3.2.1 host 侧设计

`blas/iamin/arch35/icamin_host.cpp` 实现 `aclblasIcamin`：

**1. 参数合法性校验（与 isamin 逐条一致）**

| 场景 | 行为 |
| --- | --- |
| handle == nullptr | 返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| n < 0 | 返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| n == 0 或 incx < 1（含 0 与负步长） | quick return：result 置 0，返回 `ACLBLAS_STATUS_SUCCESS`（先于指针检查） |
| x == nullptr（n>0 且 incx≥1） | 返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| result == nullptr | 返回 `ACLBLAS_STATUS_INVALID_VALUE` |

**2. tiling 策略（分核）**

- 通过 `GetAivCoreCount()` 获取可用 AIV 核数，`numBlocks = min(n, aivCoreNum)`（优先满核、核间数据尽量均分）；
- `perCoreN = n / numBlocks`，`lastCoreN = perCoreN + n % numBlocks`（最后一个核吸收余数，保证每核 computeNum ≥ 1）；
- 每核一个 SIMT 线程块，**全 stride（含 incx==1）统一**线程数 `nthreads = min(CeilAlign(CeilDiv(perCoreN, 128), 128), 2048)`（≤2048，128 对齐；与 isamin 的“incx!=1 才走 SIMT”不同，icamin 所有 stride 均走 SIMT）。

**3. workspace 规划**

- 每核写出 1 个 (minValue, minIndex) 二元组（2 个 float，索引按位转 float 存储），共 `numBlocks*2` 个 float；
- 按 64 float 对齐向 handle 工作空间申请，不足则返回 `ACLBLAS_STATUS_EXECUTION_FAILED`。

**4. kernel 调度**

| 条件 | kernel 序列 |
| --- | --- |
| 任意 stride（含 incx==1，含小 n） | `icamin_simt_kernel`（numBlocks 个 SIMT 块局部扫描）→ `icamin_reduce_kernel`（全局归约） |

#### 3.2.2 kernel 侧设计

`blas/iamin/arch35/icamin_kernel.cpp`：**全 stride（含 incx==1）统一 SIMT 标量扫描 + 全局归约**。

> 选型说明：初版按 isamin 范式实现了 incx==1 的 AIV 向量路径（DataCopy→Abs→GatherMask 解交织→Add→ReduceMin）。在 Ascend 950PR + CANN 9.2.0-beta.1 真机验证中，该向量路径确定性故障（357 例 ret=6 设备端错误 + 3 例索引错，全部集中在 incx=1），而 SIMT 标量路径全部正确。根因为 GatherMask 等向量解交织 API 在目标工具链上的运行语义与预期不一致（编译通过但运行异常）。遂移除向量路径，改为全 stride 统一 SIMT（该路径真机 100% 正确，性能带宽分析亦满足任务书标杆，见可维可测分析）。

**A. SIMT 分块扫描（`icamin_simt_kernel`，每核一个 SIMT 线程块）：**

- host 侧按 `numBlocks = min(n, GetAivCoreCount())` 分核，`perCoreN = n/numBlocks`、`lastCoreN` 末核吸收余数；
- 每核 `asc_vf_call` 启动 `nthreads`（≤2048，按 perCoreN 折算且 128 对齐）个 SIMT 线程；
- 线程按 `threadIdx.x` 步进扫描核内 `calNum` 个复数元素：`xGm[(start+i)*stride*2] / +1` 标量读取实、虚部，计算 `|Re|+|Im|`（Abs 精确、Add 正确舍入，与 CPU golden 逐位一致）；
- **逐元素跳过 NaN**（`!(val != val)`），严格小于更新、相等取最小全局索引（tie 语义）；
- 每线程维护 (bestVal, bestIdx)，线程间 pow2 树归约（无值线程保持 NaN@0 哨兵：合并时对称处理，NaN 与任何值都不参与比较、真实值可沿树传播），block 0 号线程将 (minVal, minIdx) 写入 workspace 槽位 `blockId*2`。

**B. 全局归约（`icamin_reduce_kernel`）：**

- 将 workspace 全部 `useCoreNum` 个 (minVal, minIdx) 槽位 DataCopy 整块搬入 UB，标量扫描；
- 合并规则同 A：NaN 槽位（哨兵）跳过、严格小于更新、相等取最小索引；全部无效（全 NaN 向量）时 bestIdx=0 → result=1；
- `result = minIdx + 1`（转 1-based），Duplicate 后 DataCopyPad 写出 INT32。

**NaN/Inf 语义（kernel 与 golden 统一，Q7 定项）：**

- 全路径逐元素跳过 NaN：kernel 的 SIMT 扫描与 CPU golden 同规则（首有效元素无条件播种，后续严格更小更新）；
- 全 NaN 向量：无有效值 → result=1（kernel 与 golden 一致）；
- Inf：非 NaN 可参与比较；含 Inf 与更小有限值时按值比较；仅当全部有效值均为 Inf（或 Inf/NaN 混合）时归于最小索引（首 Inf 或 1），kernel 与 golden 一致；
- 哨兵为 NaN 而非 FLT_MAX：FLT_MAX 是合法可达模值（如 Re=±FLT_MAX,im=0），数值哨兵会在 tie 中伪胜真实元素（H1 缺陷）；NaN 被 `v!=v` 守卫天然排除，真实值（含 Inf/FLT_MAX）永不与哨兵 tie 或互相覆盖（详见 H1 交接单）。

### 支持硬件

- Ascend 950PR / Ascend 950DT：支持（arch35）
- Atlas A3 训练/推理系列（arch22）：不支持（本任务仅 arch35，README 产品支持表按仓内规范标注）

### 算子约束限制

| 约束项 | 说明 |
| --- | --- |
| 参数合法性 | n≥0；n<0 报 INVALID_VALUE；n=0 或 incx<1 走 quick return；handle/x/result 不可为 nullptr |
| 非连续 Tensor | 不要求（步长由 incx 表达；负步长为 quick return，不做反向遍历） |
| broadcast | 不涉及（单向量归约） |
| dynamic shape | 不要求（n 为运行时入参） |
| 原地/视图语义 | 不涉及（result 为独立输出标量） |
| 确定性 | 整数索引结果语义确定（bit-exact 判定不依赖归约顺序） |
| 异步执行 | 依赖 `aclblasSetStream` 绑定 stream，读回结果前须同步 stream |
| NaN | 全路径逐元素跳过 NaN（kernel 与 golden 统一语义，Q7 已定项；含 FLT_MAX/Inf 混合边界的 H1 回归用例） |

## 四、可维可测分析

### 精度标准/性能标准

**精度标准**（生态算子开源精度标准，输出为整数索引 → 退化为精确一致判定）：

- NPU 输出索引与 CPU golden 索引**逐位相等**（EXPECT_EQ，bit-exact）；单用例即 ratio=1；
- golden 由测试工程内 cblas 风格手写循环生成（标准 cblas/Netlib 无 icamin 例程，golden 按 |Re|+|Im| 首有效元素播种、严格小于取最小索引）；
- NaN 语义：逐元素跳过 NaN（首有效元素播种，Q7 定为逐元素跳过；与 kernel 同一规则）。

**性能标准**（Ascend 950PR，COMPLEX64，warmup 后有效采样 >50 次取平均）：

| case | n | incx | 标杆耗时（us） |
| --- | --- | --- | --- |
| 1 | 1048576 | 1 | 363.55 |
| 2 | 2097152 | 1 | 718.46 |
| 3 | 4194304 | 1 | 2432.02 |

**可测性**：

- CSV 驱动 GTest：`test/iamin/icamin/arch35/icamin_test.cpp` 逐条 EXPECT_EQ 索引 + 返回码；
- 用例规模：1200 条（1000 精度 + 200 性能），由 `test_cases/gen_csv.py` 固定种子生成，含 L0 基础/tie、L1 尺寸扫描、L2 步长（含 quick return）、L5 填充（全零/交替/极端值/Inf/NaN）、L6 边界负向（n=0、n<0、x 空指针、quick return 优先）、EX 扩展、PF 性能；
- 手写边界用例（CSV 单填充模式无法表达混合输入）：`H1Boundary_MixedNanInfFltMax` / `H1Boundary_EmptyThreadSentinelSeam`（n=6143/6144 空线程分界线）/ `H1Boundary_NaNPrefixInfSuffix` / `H1Boundary_HomogeneousExtremes` 共 4 条，直接构造 Inf/FLT_MAX/NaN 混合输入断言 bit-exact 索引；
- 预跑验证：`h1reg/verify_h1_fix.py`（随代码仓提交）对 kernel 归约算法（NaN 哨兵 + 对称合并）与 golden 做逐行对应复刻验证（130 项检查 0 失败，含 H1 定向反例、6143/6144 缝合线、单模式填充零回归、随机压力），再上 950 跑真实 GTest（1206/1206 全过）。早期 `test_cases/simulate_icamin.py`（FLT_MAX 哨兵 + 旧 golden）已随 H1 修复过时，不再作为当前验证依据。

### 兼容性分析

- 接口签名与 `cublasIcamin` 逐参数对齐，无额外映射说明；
- 声明放入公共头文件 `include/cann_ops_blas.h`，供其他产品线共用；禁止定义 950PR 私有平行 API；
- 与同族 isamin 共存于 `blas/iamin/arch35/`，无符号冲突（各接口独立命名空间与 kernel 符号）；
- 测试工程 `test/iamin/icamin/` 参照 `test/isamin/` 模式新建，CSV 列格式对齐 `isamin_param.h` 的 ReadMap 键。