# aclblasCtrsv 算子设计文档（Ascend 950PR / arch35）

> 状态说明（2026-09-28）：本设计文档提交时，算子实现与真机自测正在进行中。
> 本文档描述设计口径与验收对齐方式，**不宣称已完成验收**。
> 开发环境为昇腾 950PR 在线算力（CANN 9.1.0，arch35），精度 golden 由 cblas（Netlib `ctrsv`）生成。

# 需求背景（required）

## 需求来源

CANN 社区任务「9月社区任务-aclblasCtrsv算子开发(950)」。目标是在昇腾 NPU（Ascend 950PR，arch35）
上，基于 ops-blas 句柄式 BLAS 接口与 Ascend C kernel 直调框架，实现与 cuBLAS `cublasCtrsv` /
Netlib BLAS `ctrsv` 参数语义一致的单精度复数（complex64）三角矩阵求解算子。

## 背景介绍

`ctrsv` 求解单右端三角线性系统：

```text
op(A) * x = b
```

其中 A 为列主序 n×n 上三角或下三角 complex64 矩阵，b 与 x 为 n 元复数向量；入口 x 存放右端 b，
出口 x 被原地覆写为解向量。`op(A)` 支持 A（`OP_N`）、A^T（`OP_T`）、A^H（`OP_C`）三种形式；
对角支持显式非单位（`NON_UNIT`）与隐式单位（`UNIT`）两种语义。

现状分析（基于 ops-blas 仓 master 实测）：

- `include/cann_ops_blas.h` 中**仅有实数版 `aclblasStrsv` 声明**，无 `aclblasCtrsv`；
- `blas/trsv/arch35/` 下已有实数实现 `strsv_host.cpp` / `strsv_kernel.cpp` / `strsv_tiling_data.h`，
  可作为同族蓝本（host 校验顺序、句柄/stream 直调模式、SIMT kernel 结构）；
- `test/trsv/strsv/` 下已有 CSV 驱动 GTest 测试工程，其列格式（`uplo/trans/diag/n/A/lda/x/incx`）
  即本任务测试 CSV 的对齐目标；
- 仓内**无任何复数三角求解实现**可复用，复数路径需新增。

因此本设计需新增：公共 API 声明、arch35 Host/Kernel 实现、CSV 驱动测试工程与自验脚本。

# 需求分析（required）

## 需求描述

- 支持 complex64（实部/虚部各 float32）输入输出。
- 支持 `uplo`(UPPER/LOWER) × `trans`(OP_N/OP_T/OP_C) × `diag`(NON_UNIT/UNIT) 共 12 种枚举组合。
- 支持 `lda >= max(1, n)`（列主序，覆盖紧凑与 padding 场景）、`incx != 0`（含 ±1/±2/±3 及负步长）、
  运行时动态 `n`（`n >= 0`）。
- `n = 0` 为合法 no-op，返回 `ACLBLAS_STATUS_SUCCESS` 且不执行计算。
- 非法参数按任务书 §2.4 返回 `ACLBLAS_STATUS_INVALID_VALUE`；`handle` 为空返回
  `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`。
- A 与 x 为 Device 指针；x 原地更新；通过 handle 绑定 stream 异步发射，算子内部不做用户 Tensor 搬运、
  不同步 stream。
- 性能达到任务书 §3.3 门槛：5 条典型 case 平均单次耗时不高于对应标杆值。

## 需求拆解

1. **公共 API 与参数校验**：新增 `aclblasCtrsv` 声明至 `include/cann_ops_blas.h`，供其他产品线共用；
   不定义 950PR 私有平行接口。
2. **语义正确性**：正确处理列主序寻址、三角引用（仅引用 uplo 指定三角）、转置/共轭转置、单位对角
   （不读取 A 对角）、正负步长与逻辑/物理长度映射。
3. **复数算术精度**：复数乘加与复数除法需与 cblas 参考实现的舍入行为保持在生态精度标准容差内；
   三角回代误差随 n 累积，需按实部/虚部分别判定。
4. **性能方案**：三角求解存在行间串行依赖，需以分块方式降低同步次数、提高向量化程度，避免退化为
   O(n) 次全局同步。
5. **测试与自验**：建立 CSV 驱动测试工程，覆盖 12 组枚举组合、尺寸扫描、对齐与步长扫描、填充模式、
   边界与负向用例；以 cblas `ctrsv` 为 golden，实部/虚部分别比对。
6. **交付**：提供算子 README（产品支持表标注 Ascend 950PR：支持）、自验证步骤与精度/性能/内存报告。

# 详细设计（required）

## 算子分析

### 数学公式

无转置（`OP_N`）下，下三角（LOWER）前代：

```text
x[i] = ( b[i] - Σ(j=0..i-1) A[i,j] * x[j] ) / A[i,i]        i = 0..n-1
```

上三角（UPPER）回代按 i 逆序进行。`OP_T` 与 `OP_C` 会交换有效三角方向；`OP_C` 读取 A 时还需对虚部取反。

`diag = UNIT` 时省略除法，且 A 的对角元素**不得读取**。

复数运算（A、x 均为 complex64）：

```text
乘法:   (a+bi)(c+di) = (ac - bd) + (ad + bc)i
除法:   (a+bi)/(c+di) = ((ac + bd) + (bc - ad)i) / (c² + d²)
共轭:   conj(a+bi) = a - bi
```

依据任务书 §2.1.6，本算子**不进行奇异性或近奇异性检测**（与 cuBLAS/Netlib 一致），调用方须保证
`diag = NON_UNIT` 时对角非零；测试侧由 `boost = max(5, n)` 的填充策略保证。

### 支持数据类型与形状

| 对象 | 类型 | 布局 / 形状 |
| --- | --- | --- |
| A | `aclblasComplex`（complex64） | Device，列主序物理 `lda × n`；仅 uplo 指定三角被引用 |
| x | `aclblasComplex`（complex64） | Device，逻辑 `[n]`，物理长度 `1 + (n-1)·abs(incx)` |
| handle / uplo / trans / diag / n / lda / incx | Host 标量 | 运行时属性 |

`aclblasComplex` 以 `include/cann_ops_blas_common.h` 定义为准（实部/虚部各 float32）。

## 算子实现

### 接口定义

`include/cann_ops_blas.h` 新增声明（与 `aclblasStrsv` 参数次序一致，整数维参数采用 `int`）：

```cpp
aclblasStatus_t aclblasCtrsv(
    aclblasHandle_t handle,
    aclblasFillMode_t uplo,
    aclblasOperation_t trans,
    aclblasDiagType_t diag,
    int n,
    const aclblasComplex* A,
    int lda,
    aclblasComplex* x,
    int incx);
```

实现落点：`blas/trsv/arch35/ctrsv_host.cpp`、`ctrsv_kernel.cpp`、`ctrsv_tiling_data.h`。

### Host 侧设计

Host 侧分为参数校验与 kernel 发射两步：

1. **handle 检查**：`handle == nullptr` 返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`。
2. **维度与标量校验**：`n < 0`、`lda < max(1, n)`、`incx == 0` 返回 `ACLBLAS_STATUS_INVALID_VALUE`；
   `uplo` / `trans` / `diag` 取值不在任务书 §2.4 列举枚举内同样返回 `ACLBLAS_STATUS_INVALID_VALUE`
   （口径与 cuBLAS `CUBLAS_STATUS_INVALID_VALUE` 及仓内 `trsv_golden.h` 一致）。
3. **n = 0 快速返回**：标量校验通过后，`n == 0` 直接返回 `ACLBLAS_STATUS_SUCCESS`，不读取 A/x 指针、
   不发射 kernel（依据 Netlib `ctrsv` 的 `IF (N.EQ.0) RETURN`）。这使合法的空操作允许 A/x 为空。
4. **指针校验**：`n > 0` 时 A 或 x 为 `nullptr` 返回 `ACLBLAS_STATUS_INVALID_VALUE`。
5. **Tiling 组装**：将 `n`、`uplo`、`trans`、`diag`、`incx`、`lda` 与分块/线程参数写入小型 POD
   tiling 结构体，按值传入 kernel。
6. **发射**：以 A/x 的 Device 地址调用 `ctrsv_kernel_do(..., handle->stream)`。算子内部**不调用**
   `aclrtMalloc`、不做用户 Tensor 的 H2D/D2H、不调用 `aclrtSynchronizeStream`。

> 校验顺序说明：`handle` → 标量（`n`/`lda`/`incx`/枚举）→ `n == 0` 快速返回 → 指针。
> 该顺序满足任务书对 `n = 0` 合法 no-op 与 `n > 0` 时指针非空的要求，并与 cuBLAS 先校验标量、
> 再触碰指针的行为一致。

### Kernel 侧设计

三角求解的每个未知量依赖先前已解出的分量，因此 kernel 的核心矛盾是**串行依赖**与**并行度**的平衡。
本设计采用「分块 + 块内小规模直接求解 + 块外向量化更新」的结构：以 `nb`（默认 64）为一组待解列，
组内用小块求解降低同步次数，组外用连续列的向量化复乘更新右端。

三种 trans 的访存形态不同，分述如下。

#### 1. OP_N：列更新路径

列主序下 A 的**列连续**。求解第 i 个分量后，对剩余分量执行：

```text
x[target] += -x[i] * A[target, i]
```

- LOWER 自前向后、UPPER 自后向前推进；
- 对角块（`nb × nb`）按列紧凑搬入 UB，严格排除非引用三角；`diag = UNIT` 时对角位置不搬运；
- 复数向量拆分为实部/虚部分量后以向量乘加更新常驻 UB 的 x 实/虚分量，仅在最终输出时重新交织，
  避免逐列更新反复读写 x 的全局内存；
- 整段 x 在路径入口自 GM 读入一次、全部前代/回代完成后写回一次；
- `incx = 1` 使用整段 `DataCopy`；非单位步长仅在入口/出口各执行一次 O(n) 的 gather/scatter，
  逻辑求解路径与连续步长完全一致。

#### 2. OP_T / OP_C：连续列点积路径

转置后当前方程涉及 A 原存储中的一段**连续列**，可复用复数点积的拆分-复乘-规约方式：

- `OP_T`：直接使用 A 的实部/虚部；
- `OP_C`：对 A 的虚部分量取负后参与计算；
- 对实部与虚部分别做向量乘法与规约，再从当前 b 中减去点积。

同样采用 `nb × nb` 对角块求解 + 块外点积更新的结构，整段 x 入口搬入 UB、求解后写回。

#### 3. diag = UNIT 的处理

UNIT 语义下 A 的对角元素不读取、按 1 处理：

- 搬运阶段**不搬运**对角位置，或搬运后以掩码排除，确保不引用对角数据；
- 求解阶段省略复数除法（不做 `x[i] /= A[i,i]`）；
- 该分支与 `NON_UNIT` 通过模板参数在编译期分离，避免运行时分支开销。

#### 4. 步长与对齐处理

按 BLAS 规则计算 x 的逻辑偏移：

```text
incx > 0:  offset(i) = i * incx
incx < 0:  offset(i) = (n - 1 - i) * abs(incx)
```

非单位步长在入口按上式 gather、出口 scatter；非逻辑位置的 padding lane 保持原值不被改写
（测试侧以 NaN 填充该区域以发现越界写入）。

#### 5. 精度处理要点

- 三角回代误差随 n 累积，测试按任务书 §3.2 的混合容差逐元素判定，实部、虚部分别统计；
- 为与 cblas 参考实现的舍入行为保持接近，复数乘加采用融合乘加（FMA）并对齐 binary32 舍入；
- NaN/Inf 输入按任务书「规格允许的 INF/NAN 场景」处理，规约前置零无效 lane，避免 NaN/Inf 污染
  有效分量。

### 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas 950PR（arch35，SoC `ascend950`） | √ |

适配 CANN 9.1.0；实现位于 `blas/trsv/arch35/`，仅在该架构下参与构建。

### 算子约束限制

- `n >= 0`；`lda >= max(1, n)`；`incx != 0`；
- `n > 0` 时 A、x 不可为空指针；合法的 `n = 0` 空操作允许 A/x 为空；
- 仅访问 `uplo` 指定三角；`diag = UNIT` 时不读取 A 对角；
- 不做奇异性/近奇异性检测；
- 不要求支持超出 `lda`/`incx` 语义的非连续内存访问，不涉及 broadcast，不要求 dynamic shape、
  不要求确定性计算、不返回视图。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度 | 输出 x 逐元素验证，实部/虚部分别按 FLOAT32 判定：`\|actual-golden\| <= atol + rtol*\|golden\|`，`atol=2^-16`、`rtol=2^-10`；整体要求 `matched_ratio >= 0.99` 且 `max_abs_error <= max(1e-2, 32*ULP)` | 任务书 §3.2、生态算子开源精度标准 |
| 性能 1 | UPPER / N / NON_UNIT，n=512，incx=1，平均单次 ≤ 146.33 us | 任务书 §3.3 |
| 性能 2 | LOWER / N / NON_UNIT，n=1024，incx=1，平均单次 ≤ 272.53 us | 任务书 §3.3 |
| 性能 3 | UPPER / T / NON_UNIT，n=2048，incx=1，平均单次 ≤ 526.22 us | 任务书 §3.3 |
| 性能 4 | LOWER / C / NON_UNIT，n=4096，incx=1，平均单次 ≤ 1177.84 us | 任务书 §3.3 |
| 性能 5 | UPPER / N / UNIT，n=4096，incx=1，平均单次 ≤ 1030.61 us | 任务书 §3.3 |
| 内存 | 不涉及（任务书 §3.4）；仍提供分配前/峰值/释放后对照作为补充 | 任务书 §3.4 |

性能测试先 warmup 再有效采样 > 50 次取平均；性能用例均为连续访存（`incx = 1`、`lda` 紧凑）。

## 测试工程施工口径

测试工程落点 `test/trsv/ctrsv/`，结构对齐仓内同族：

```text
test/trsv/ctrsv/
├── CMakeLists.txt              # ops_blas_add_gtest_tests(${OPS_BLAS} ctrsv_test)
├── ctrsv_param.h               # CSV 列映射（uplo/trans/diag/n/A/lda/x/incx/expect_result/...）
├── ctrsv_golden.h              # cblas_ctrsv golden 封装 + 参数校验
└── arch35/
    ├── ctrsv_test.cpp          # CSV 驱动 GTest（含 NullHandle 直测）
    ├── ctrsv_npu_wrapper.h     # aclblasCtrsv_npu 封装（含 stream 同步）
    └── ctrsv_test.csv          # 用例数据
```

要点：

- **二进制路径契约**：验收脚本 `verify_accuracy.py` / `verify_performance.py` 按序查找
  `build/test/trsv/ctrsv/ctrsv_test`（首选）与 `build/test/ctrsv/ctrsv_test`。本工程采用
  `test/trsv/ctrsv/` 家族嵌套布局，构建产物落在 `build/test/trsv/ctrsv/ctrsv_test`，与脚本首选路径一致。
- **CSV 安装路径契约**：脚本将用例 CSV 复制到 `test/trsv/ctrsv/arch35/ctrsv_test.csv`；本工程在该路径
  自持一份同名 CSV，保证脚本 `--csv` 覆盖与否均可运行。
- **运行过滤契约**：精度验收以 `--gtest_filter=-*TC_PF*` 排除性能用例；因此精度用例命名不带
  `TC_PF` 前缀，性能/内存用例统一以 `TC_PF_` 前缀命名。
- **golden**：调用系统 reference BLAS 的 `cblas_ctrsv`（实部/虚部分别比对）。调用前先做参数校验，
  避免空指针进入 CBLAS 导致崩溃。`aclblasComplex` 与 CBLAS 复数内存布局二进制兼容，指针为纯
  reinterpret。
- **对角非零保障**：填充 A 后对被引用对角元加符号保持偏移量 `boost = max(5, n)`；golden 侧使用
  同一偏移后的矩阵回代，两侧输入严格一致。`diag = UNIT` 用例的 golden 不读取 A 对角。
- **三角引用校验**：未引用三角、UNIT 对角位置与 `lda` padding 行写入 NaN，可直接发现错误访问。
- **容差口径**：以仓内 `verify.h` 的混合容差为准——对复数输出的实部、虚部分别构造
  `VerifyConfig` 并调用 `applyMixedTolerance(cfg, ACL_FLOAT, ...)`，即
  `rtol = 2^-10`、`atol = 2^-16`、`matched_ratio >= 0.99`、逐元素上限 `max(1e-2, 32*ULP)`，
  与任务书 §3.2 的 complex64 规格逐项一致（已在 950PR 真机以同族 `strsv_test` 实测复现该输出口径）。
  仓内 CSV 惯用的 `mere_threshold = 2^-13` / `mare_multiplier = 10`（MERE/MARE）作为**辅助口径**
  一并统计，两套口径同时通过才判定该用例 PASS，并在交付报告中说明。

## 与任务包自带用例的口径差异及自补用例说明

对任务包 `test_cases/ctrsv_test.csv`（1200 条 = 1000 精度 + 200 性能）逐列审计后，发现以下差异，
按任务书 §3.5.4「如果任务配套提供的用例没有覆盖要求的场景需要自行补充相应的用例」处理：

1. **`lda` 最小约束的负向用例无效**：任务包 `TC_ED_095`（描述 `lda0_n1`，意图覆盖 `lda = 0` 非法）
   实际落库值为 `n = 1, lda = 1`，而 `lda >= max(1, n) = 1` 成立，属合法输入，却期望
   `ACLBLAS_STATUS_INVALID_VALUE`。根因是任务包生成脚本 `gen_csv.py` 中
   `lda = max(lda if lda is not None else n, 1)` 将 `lda = 0` 抬升为 1。
   **处理**：保留原用例并按其语义修正期望值，另**自补**一条真正非法的 `lda < max(1,n)` 且 `n > 1`
   用例，确保该负向分支被真实覆盖。
2. **`handle = nullptr` 无覆盖**：任务包 CSV 列格式无法表达 handle，1000 条精度用例的
   `expect_result` 全部为成功/`INVALID_VALUE`，`ACLBLAS_STATUS_HANDLE_IS_NULLPTR` 分支未被覆盖。
   **处理**：按仓内惯例在 `ctrsv_test.cpp` 中新增非 CSV 驱动的 `TEST_F(..., NullHandle)` 直测。
3. **正态分布输入依赖测试工程扩展**：任务书 §3.5.3 要求 A 与 x 的填充为「均匀分布 50% + 正态分布
   （μ∈[-5,5]，σ∈[0.1,2]）50%」，任务包生成器注明当前仅支持均匀分布。
   **处理**：在测试工程侧补充正态分布填充路径并按 50% 混比配置用例；若该扩展需改动
   `test/frame/` 公共框架，则随本任务一并提交并说明。
4. **仅 x 含 Inf/NaN**：任务包对 x 提供 `VALUE_NORM_INF` / `VALUE_NORM_NAN` 用例，A 未提供。
   任务书 §3.5.3 的特殊值要求仅针对 x，故 A 的 Inf/NaN **不作为缺口**，按需补充为加分项。

> 口径原则：**以验收脚本为准**（脚本决定二进制路径、CSV 安装路径与过滤规则），
> `test_cases/README.md` 次之，任务书正文用于确认语义与门禁；三者冲突时在本文档与本报告中披露。

## 兼容性分析

- 新增 API `aclblasCtrsv`；与既有 `aclblasStrsv` 参数次序保持一致，声明置于
  `include/cann_ops_blas.h`，供其他产品线共用；**不定义 950PR 私有平行接口**。
- 不修改任何既有算子的行为；新增实现仅在 arch35 下参与构建。
- 接口为纯新增，不涉及既有接口的 ABI/语义变更。

## AI 辅助披露

- 本次提交是否有 AI 参与：是（正式提交时如实勾选）。
- AI Agent 平台：DeepSeek Harness。
- AI 模型：deepseek-v4.1-flash。
- 使用范围：需求梳理、设计文档撰写、代码生成与审查、测试用例设计与问题排查。
- 所有代码、测试结果与提交内容均由参赛者检查确认；**未实际运行的 NPU 测试不写为通过**。
