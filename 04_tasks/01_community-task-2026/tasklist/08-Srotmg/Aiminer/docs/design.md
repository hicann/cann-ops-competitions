# aclblasSrotmg 算子（Atlas A2/A3）设计文档

> 任务：8月社区任务-aclblasSrotmg算子开发（A2/A3）
> 提交者（GitCode）：Aiminer
> 提交路径：`04_tasks/01_community-task-2026/tasklist/08-Srotmg/Aiminer/docs/design.md`
> 任务详情页：https://www.hiascend.com/activities/task-center/details/d5af83c2942f41efaab5c9ba5f555fbf
> 开源仓：https://gitcode.com/cann/ops-blas （实现合入 `blas/rotmg/`，测试合入 `test/rotmg/srotmg/arch22/`）
>
> 注：截至提交时，8月26日批次任务尚未在 cann-competitions tasklist 中分配编号目录，本任务目录 `08-Srotmg` 由本提交按仓内既有"批次号-算子名"命名先例（如 08-aclsparseSpMM）创建；若官方后续分配编号，可同步调整目录名。

# 需求背景（required）

## 需求来源

CANN 社区任务 2026（8月批次）：在 Atlas A2/A3 系列产品（含 Atlas 800I A2 / Atlas 800I A3）上，基于 Ascend C 实现 `aclblasSrotmg` 算子，补齐 ops-blas 开源仓对 Atlas A2/A3 产品线的支持。任务书：8月社区任务-aclblasSrotmg算子开发（A2/A3）。

## 背景介绍

### aclblasSrotmg 算子功能

`aclblasSrotmg` 是 BLAS Level 1 纯标量算子：由输入标量 d1、d2、x1、y1 构造修正 Givens（modified Givens）变换矩阵 H，满足：

```
H^T * diag(d1, d2) * H = diag(d1_new, d2_new)
H * [x1, y1]^T = [x1_new, 0]^T
```

H 为 2×2 矩阵，由输出数组 param[5] 按 BLAS 标准编码：

| param[0] (flag) | H 矩阵 | 存储的元素 |
| --- | --- | --- |
| -1.0 | [[h11, h12], [h21, h22]] | param[1..4] 全部存储（h11/h21/h12/h22） |
| 0.0 | [[1, h12], [h21, 1]] | param[2]=h21，param[3]=h12 |
| 1.0 | [[h11, 1], [-1, h22]] | param[1]=h11，param[4]=h22 |
| -2.0 | [[1, 0], [0, 1]] | 无（恒等变换，仅设 flag） |

输出通常直接作为下游 rotm 算子的输入。

### 现状分析

- 对标基线：cuBLAS `cublasSrotmg`，语义参考 Netlib `srotmg`（https://www.netlib.org/blas/srotmg.f ）。
- ops-blas 仓 `blas/rotmg/arch35/` 已有 Ascend 950PR（arch35）实现，产品支持表中 Atlas A2/A3 系列当前标注"不支持"，本任务补齐 A2/A3（arch22）路径。
- 接口声明 `aclblasSrotmg` 已存在于 `include/cann_ops_blas.h`（第 374 行），与 950PR 产品线共用同一 API，禁止定义产品私有平行接口。
- 测试侧：`test/rotmg/srotmg/` 已有 CSV 驱动的 GTest 工程（含 arch35 目录与 cblas golden 包装），本任务在其下新增 arch22 测试路径与用例。

# 需求分析（required）

## 需求描述

在 Atlas A2/A3 上使用 Ascend C 实现 `aclblasSrotmg` 句柄式 BLAS 接口，语义与 cuBLAS/Netlib 完全对齐，精度满足生态算子开源精度标准（FLOAT32），性能不高于标杆耗时（Atlas 800T A2 / 910B3 实测），完成算子设计、开发、测试全流程。

## 需求拆解

1. 实现 `aclblasSrotmg` Host 侧入口：参数校验、指针位置判别（Host/Device 双路径）、arch22 kernel 启动；
2. 实现 arch22（Atlas A2/A3）Device kernel：完整覆盖 Netlib `srotmg` 全部分支语义（flag = -2/-1/0/1 与 GAM=4096 缩放保护循环）；
3. 精度达标：d1/d2/x1/param[0..4] 与 golden（Netlib `cblas_srotmg`）比对，flag 为离散值须精确相等，浮点分量按 FLOAT32 标准（rtol=2^-10，atol=2^-16，matched_ratio≥0.99，max_abs_error≤1e-2 或 32 ULP）；
4. 性能达标：单次调用延迟不高于标杆（一般值 2.67us、flag=-2 快速返回 2.35us、缩放保护路径 2.61us）；
5. 在 `test/rotmg/srotmg/` 下新增 arch22 测试路径与 CSV 用例，覆盖精度、负向、性能用例；
6. 更新 `blas/rotmg/README.md` 产品支持表：Atlas A2/A3 系列产品标注"支持"。

# 详细设计（required）

## 算子分析

### 数学公式

见"需求背景"中 H 矩阵构造式。核心分支逻辑（与 Netlib `srotmg.f` 一致）：

1. `d1 < 0`：flag = -1，H 全零，d1、d2、x1 覆写为 0；
2. `d1 ≥ 0 且 d2·y1 = 0`（含 d2=0 或 y1=0）：flag = -2（恒等变换），仅设 param[0] 即返回，d1、d2、x1 不变；
3. `|d1·x1²| > |d2·y1²|`：h21 = -y1/x1、h12 = (d2·y1)/(d1·x1)，su = 1 - h12·h21；su > 0 时 flag = 0，d1 /= su、d2 /= su、x1 *= su；su ≤ 0 时（舍入边界 safety 分支）flag = -1 全零；
4. `|d1·x1²| ≤ |d2·y1²|`：若 d2·y1² < 0 则 flag = -1 全零；否则 flag = 1，h11 = (d1·x1)/(d2·y1)、h22 = x1/y1，su = 1 + h11·h22，d1、d2 缩放后互换，x1 = y1·su；
5. 缩放保护：GAM = 4096、GAMSQ = 1.67772e7、RGAMSQ = 5.96046e-8；当 d1（或 |d2|）越出 [RGAMSQ, GAMSQ] 区间时循环缩放，缩放过程中 flag 翻转为 -1 并将 H 对应元素置为 ±1，保证极端数量级输入不溢出/不下溢。

### 支持数据类型

仅 FLOAT32（单精度实数标量）。d1/d2/x1 为输入/输出标量（原地覆写），y1 为只读输入标量，param 为 5 元素输出数组。

### 支持形状

纯标量算子：无维度、无步长、无广播、无 dynamic shape。

## 算子实现

### 实现方案

整体复用仓内已验证的 arch35 设计模式，按 arch22 架构适配。文件规划：

```
blas/rotmg/arch22/
├── srotmg_host.cpp        # Host 侧入口与 CPU 直算路径
├── srotmg_kernel.cpp      # Device kernel 与启动器
└── srotmg_tiling_data.h   # 占位 tiling 结构
```

#### Host 侧设计（srotmg_host.cpp）

1. **参数校验**：handle 为 nullptr 返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`；d1/d2/x1/y1/param 任一为 nullptr 返回 `ACLBLAS_STATUS_INVALID_VALUE`。d1/d2 取负值属合法输入（走对应数学分支），不做拦截。
2. **指针位置判别**：对 5 个指针逐一调用 `aclrtPointerGetAttributes`，判定 Host/Device 位置：
   - 全 Host：直接调用本文件内的 `SrotmgCpuCompute`（Netlib 等价 CPU 实现）完成计算，不启动 kernel、不做 memcpy；
   - 全 Device：构造空 tiling，调用 `srotmg_kernel_do` 以 1 block 启动 arch22 kernel；
   - 混合 Host/Device：返回 `ACLBLAS_STATUS_INVALID_VALUE`，并打点日志记录各指针位置（口径与仓内 `blas/rotmg/README.md` 一致）。
3. **tiling 策略**：纯标量算子无数据切分，`SrotmgTilingData` 仅含占位字段；不涉及分核、UB 切分、tilingKey 规划。
4. **异步执行**：依赖 `aclblasSetStream` 绑定 stream；Host 侧不做隐式同步，读回 Device 结果前由调用方自行同步 stream。

#### Kernel 侧设计（srotmg_kernel.cpp）

1. **执行形态**：AIV-only、单 block kernel；`GetBlockIdx() != 0` 直接返回，仅 block 0 参与计算。
2. **数据访问**：使用 `GlobalTensor<float>` + `GetValue/SetValue` 直接读写 GM 标量（d1/d2/x1/y1 各 1 个元素、param 5 个元素）。计算量仅几十条标量指令，不搬 UB、不走 DataCopy 管线，避免小数据搬运开销大于计算本身的问题。
3. **计算流程**（与 Host 侧 CPU 路径、Netlib 参考实现逐步对齐）：
   - 读入 sd1/sd2/sx1/sy1；
   - 按"数学公式"第 1~5 条执行分支判定与缩放保护循环（kernel 内不调用 `std::abs`，用条件取负实现；while 循环每次迭代后同步更新 |sd2| 局部量）；
   - 按 flag 打包 param[0..4]（flag<0 存全 h11/h21/h12/h22；flag=0 存 h21/h12；flag>0 存 h11/h22）；
   - 回写 d1/d2/x1；flag=-2 路径写 param[0]=-2 与 param[1..4]=0 后立即返回。
4. **arch22 适配点**：kernel 入口属性、头文件引用（`kernel_operator.h`、`common/helper/kernel_constant.h`）按 A2/A3 编译体系组织；启动器 `srotmg_kernel_do` 以 `<<<numBlocks=1, nullptr, stream>>>` 形式在 handle 绑定的 stream 上下发。

#### 与 arch35 的关系

算法与工程模式保持一致（评审与维护口径统一），差异仅在架构目录与编译接入（arch22 vs arch35）；不修改 `include/cann_ops_blas.h` 既有声明，两产品线共用同一 API。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas 800I/T A2（arch22） | √ |
| Atlas 800I A3（arch22） | √ |
| Ascend 950PR/950DT（arch35，既有实现，不在本任务范围） |  |

## 算子约束限制

1. d1、d2、x1、y1、param 五指针须全部位于 Host 侧或全部位于 Device 侧；混合指针返回 `ACLBLAS_STATUS_INVALID_VALUE`；
2. 本算子为纯标量运算：不支持非连续 Tensor、不涉及 broadcast、不涉及 dynamic shape、不涉及确定性计算要求（无归约，天然确定）；
3. d1、d2、x1 原地覆写；y1 只读；param 为独立输出数组；
4. 缩放保护路径下，不同 BLAS 库 GAMSQ/RGAMSQ 常量尾差可能产生 ±1 次缩放迭代的良性偏差，比对以 `cblas_srotmg` 为唯一 golden 口径。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | golden 由 cblas（Netlib `cblas_srotmg`）生成；d1/d2/x1 覆写值及 param[0..4] 逐标量比对，flag（-2/-1/0/1）精确相等；浮点分量按 FLOAT32 标准：rtol=2^-10、atol=2^-16、matched_ratio≥0.99、max_abs_error≤1e-2 或 32 ULP | 生态算子开源精度标准 + 任务书 §3.2 |
| 性能标准 | Atlas 800I A2（910B3）、FLOAT32、warmup 后采样 >50 次取平均，平均单次耗时不高于：一般值（flag=1）2.67us、flag=-2 快速返回 2.35us、缩放保护路径 2.61us | 任务书 §3.3 |

## 测试方案

1. **测试框架**：复用 `test/rotmg/srotmg/` 既有 CSV 驱动 GTest 工程与 cblas golden 包装（`srotmg_golden.h`、`srotmg_param.h`），新增 `test/rotmg/srotmg/arch22/`（npu wrapper、test cpp、CSV 用例），文件结构参考主仓 blas 算子测试代码结构。
2. **精度用例**（配合任务书 test_cases 的 `gen_csv.py` / `verify_accuracy.py`）：
   - 值域分布：d1/d2/x1/y1 均匀分布 [-5,5] 占 50%、正态分布（μ∈[-5,5]，σ∈[0.1,2]）占 50%；
   - 特殊值：d1<0（全零分支）、y1=0 或 d2=0（flag=-2 快速返回）、d1=1e10 / 1e-9 量级（GAM 缩放保护）、|d1·x1²| 与 |d2·y1²| 相等边界；
   - 负向用例：handle/d1/d2/x1/y1/param 为 nullptr，期望返回对应错误码；混合 Host/Device 指针用例，期望 `ACLBLAS_STATUS_INVALID_VALUE`；
   - 双路径覆盖：同一组数值用例分别在全 Host 指针与全 Device 指针下执行，结果一致。
3. **性能用例**（`verify_performance.py`）：任务书 §3.3 三个标杆 case + 批量连续调用吞吐场景（单流内连续调用 N 次取平均，每轮重置输入标量，模拟 QR 分解等迭代调用场景）。

## 兼容性分析

- 接口层：复用 `include/cann_ops_blas.h` 既有 `aclblasSrotmg` 声明（第 374 行），与 950PR 产品线共用，无 API 变更，不存在接口兼容性问题；
- 实现层：新增 `blas/rotmg/arch22/` 目录，不改动 arch35 既有代码；`blas/rotmg/README.md` 产品支持表将 Atlas A2/A3 由"不支持"更新为"支持"；
- 该算子在 A2/A3 上为新增能力，无历史行为需要兼容。

# 交付件与计划

| 交付件 | 说明 |
| --- | --- |
| 设计文档 PR | 本文档，提交至 cann-competitions `04_tasks/01_community-task-2026/tasklist/08-Srotmg/Aiminer/docs/design.md` |
| 算子代码 | 个人 ops-blas fork，`blas/rotmg/arch22/`（合入目标 https://gitcode.com/cann/ops-blas ） |
| 测试代码与用例 | `test/rotmg/srotmg/arch22/`（含 CSV） |
| 自测报告 | 按官方模板：用例参数、精度对比结果及截图、性能数据及截图 |
| README 更新 | `blas/rotmg/README.md` 产品支持表标注 A2/A3：支持 |

> 开发环境：Gitee AI 算力（A2/A3），开源仓免费时长额度内使用，闲时关停。
