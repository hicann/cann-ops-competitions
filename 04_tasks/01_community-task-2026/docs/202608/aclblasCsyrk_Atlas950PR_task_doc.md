# aclblasCsyrk 算子开发任务书

## 1. 任务概述

在昇腾 NPU（Ascend 950PR）上使用 Ascend C 编程语言开发单精度复数（complex64）对称秩-k 更新算子 `aclblasCsyrk`，完成算子设计、开发、测试全流程工作。验收通过后合入昇腾算子开源仓（请参考链接：https://gitcode.com/cann/ops-blas ）。

## 2. 核心开发要求

### 2.1 功能实现要求

1. 与 cuBLAS `cublasCsyrk` 核心功能、参数语义完全对齐，计算对称秩-k 更新 `C = alpha * op(A) * op(A)^T + beta * C`：
   - `trans = ACLBLAS_OP_N` 时：op(A) = A（A 为 n×k），`C = alpha * A * A^T + beta * C`；
   - `trans = ACLBLAS_OP_T` 时：op(A) = A^T（A 为 k×n），`C = alpha * A^T * A + beta * C`；
   - `trans = ACLBLAS_OP_C` 时：按 ACLBLAS_OP_T 等价处理（**不共轭**）——本算子为对称（非厄米特）运算，op(A) 定义不含共轭，共轭转置语义属 `aclblasCherk`；
   其中 alpha、beta、A、C 均为单精度复数。
2. C 为 n×n **对称**复数矩阵（满足 `C = C^T`，不是厄米特矩阵，对角元素虚部不假定任何值）：仅 `uplo` 指定的上三角（ACLBLAS_UPPER）或下三角（ACLBLAS_LOWER）被引用并更新，另一三角不被访问、由对称性隐含。
3. 维度约束：op(A) 为 n×k；trans=N 时 A 为 n×k（lda ≥ max(1,n)），trans=T/C 时 A 为 k×n（lda ≥ max(1,k)）；C 为 n×n（ldc ≥ max(1,n)），列主序（Column-Major）存储。
4. 复数类型 `aclblasComplex` 以 ops-blas 仓 `include/cann_ops_blas_common.h` 中定义为准（实部/虚部各 float32）。
5. no-op / quick return 行为（语义参考 Netlib `csyrk`）：
   - `n = 0` 时为合法 no-op（直接返回 `ACLBLAS_STATUS_SUCCESS`，不执行计算）；
   - `(alpha = (0,0) 或 k = 0) 且 beta = (1,0)` 时直接返回；
   - `alpha = (0,0)` 或 `k = 0` 且 beta ≠ (1,0) 时，仅对 uplo 引用三角执行 `C = beta * C`（beta = (0,0) 时引用三角置零），不是纯 no-op。
6. 对标基线接口：cuBLAS `cublasCsyrk`（语义参考 Netlib `csyrk`，请参考链接：https://www.netlib.org/blas/csyrk.f ）。
7. 接口声明放入 `include/cann_ops_blas.h`，禁止定义 950PR 私有平行接口。

### 2.2 算子工程模式

使用 Ascend C kernel 直调方式开发：基于 ops-blas 开源仓（请参考链接：https://gitcode.com/cann/ops-blas ）工程框架，实现 `aclblasCsyrk` 句柄式 BLAS 接口，通过 handle 绑定 stream 直调 NPU kernel，实现代码放在 `blas/syrk/arch35/`（与同族实数 `aclblasSsyrk` 的 arch35 实现同目录）。

### 2.3 接口定义

⚠️ ops-blas 仓 `include/cann_ops_blas.h` 中**尚无 `aclblasCsyrk` 声明，头文件需新增该声明**；新增声明须与下述定义逐参数一致（维数参数为 `int`，与同族 `aclblasSsyrk`/`aclblasSsyrkx`/`aclblasCherk` 已有声明的整型口径一致），供其他产品线共用：

```cpp
aclblasStatus_t aclblasCsyrk(
    aclblasHandle_t handle,
    aclblasFillMode_t uplo, aclblasOperation_t trans,
    int n, int k,
    const aclblasComplex* alpha,
    const aclblasComplex* A, int lda,
    const aclblasComplex* beta,
    aclblasComplex* C, int ldc);
```

### 2.4 参数说明

| 参数名 | 输入／输出/属性 | 描述 | 数据类型 | dtype类型 | 数据排布格式 | 维度(shape) | 值域范围 | 异常行为 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| handle | 输入 | ops-blas 库上下文句柄，携带 stream，Host 内存 | scalar | - | - | - | 指向已创建的有效句柄 | handle 为 nullptr 时返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| uplo | 输入 | C 矩阵存储模式：ACLBLAS_UPPER（引用上三角）或 ACLBLAS_LOWER（引用下三角），Host 内存 | attr | int（枚举） | - | - | {ACLBLAS_UPPER, ACLBLAS_LOWER} | 取值不在上述枚举时返回 `ACLBLAS_STATUS_INVALID_ENUM` |
| trans | 输入 | A 矩阵转置模式：ACLBLAS_OP_N（不转置）、ACLBLAS_OP_T（转置）或 ACLBLAS_OP_C（按 OP_T 等价，不共轭），Host 内存 | attr | int（枚举） | - | - | {ACLBLAS_OP_N, ACLBLAS_OP_T, ACLBLAS_OP_C} | 取值不在上述枚举时返回 `ACLBLAS_STATUS_INVALID_ENUM` |
| n | 输入 | 矩阵 C 的阶数、op(A) 的行数，Host 内存 | scalar | int | - | - | n ≥ 0 | n < 0 时返回 `ACLBLAS_STATUS_INVALID_VALUE`；n = 0 为合法 no-op |
| k | 输入 | op(A) 的列数（trans=N 时 A 的列数，trans=T/C 时 A 的行数），Host 内存 | scalar | int | - | - | k ≥ 0 | k < 0 时返回 `ACLBLAS_STATUS_INVALID_VALUE`；k = 0 且 beta ≠ (1,0) 时仅执行 C = beta*C |
| alpha | 输入 | 指向复数标量乘数的指针，Device 内存 | scalar | COMPLEX64 | - | - | 实部/虚部取值于 FLOAT32 全集 | alpha 为 nullptr 时返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| A | 输入 | 输入矩阵 A，Device 内存，只读 | tensor | COMPLEX64 | ND（列主序） | trans=N：n×k；trans=T/C：k×n | 实部/虚部取值于 FLOAT32 全集 | n > 0 且 k > 0 时 A 为 nullptr 返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| lda | 输入 | 矩阵 A 的前导维度（列主序），Host 内存 | scalar | int | - | - | trans=N 时 lda ≥ max(1, n)；trans=T/C 时 lda ≥ max(1, k) | 不满足前导维度约束时返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| beta | 输入 | 指向复数标量乘数的指针，Device 内存；beta = (0,0) 时 C 不必是有效输入 | scalar | COMPLEX64 | - | - | 实部/虚部取值于 FLOAT32 全集 | beta 为 nullptr 时返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| C | 输入/输出（原地输出） | n×n 对称矩阵 C，Device 内存，输入旧值、原地覆写输出新值；仅 uplo 指定三角被引用与更新 | tensor | COMPLEX64 | ND（列主序） | n×n | 实部/虚部取值于 FLOAT32 全集 | beta 非零且 n > 0 时 C 为 nullptr 返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| ldc | 输入 | 矩阵 C 的前导维度（列主序），Host 内存 | scalar | int | - | - | ldc ≥ max(1, n) | ldc < max(1, n) 时返回 `ACLBLAS_STATUS_INVALID_VALUE` |

**返回值**：`aclblasStatus_t`，状态码语义与 ops-blas 仓 `include/cann_ops_blas_common.h` 定义一致（`ACLBLAS_STATUS_SUCCESS` / `ACLBLAS_STATUS_INVALID_VALUE` / `ACLBLAS_STATUS_INVALID_ENUM` / `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` 等）。

### 2.5 算子实现约束

| 约束项 | 内容 |
| --- | --- |
| 参数合法性 | n ≥ 0、k ≥ 0；trans=N 时 lda ≥ max(1,n)、trans=T/C 时 lda ≥ max(1,k)；ldc ≥ max(1,n)；alpha、beta 不可为 nullptr；非法数值参数返回 `ACLBLAS_STATUS_INVALID_VALUE`，非法枚举返回 `ACLBLAS_STATUS_INVALID_ENUM` |
| 三角引用 | 仅引用/更新 uplo 指定三角；C 为对称（非厄米特）矩阵，另一三角不访问、由对称性隐含（不共轭） |
| 非连续 Tensor 支持 | 不要求（本批次不支持超出 lda/ldc 语义的非连续内存访问） |
| broadcast 规则 | 不涉及，A/C 为独立矩阵，无广播 |
| dynamic shape 要求 | 不要求，n/k 为运行时入参 |
| 原地与视图语义 | C 原地覆写，不返回视图 |
| 确定性计算要求 | 不要求 |
| 空 Tensor 与 0 维处理 | n = 0 为合法 no-op，返回成功且不执行计算；k = 0 且 beta ≠ (1,0) 时仅执行 C = beta*C |
| 异步执行 | 依赖 `aclblasSetStream` 绑定 stream；读回 Device 结果前须同步 stream |

## 3. 验收标准

### 3.1 软硬件环境要求

- **适配硬件**：Ascend 950PR
- **CANN 版本**：CANN 9.1.0
- **三方软件版本**：精度比对 golden 由 cblas（Netlib BLAS 复数实现）生成，随测试工程提供，无其他三方软件依赖

### 3.2 精度要求

1. golden 由 cblas（Netlib `csyrk`）单标杆比对生成，输出矩阵 C 的 uplo 引用三角（n×n 对称矩阵的有效三角）全量验证，实部、虚部分别比对。
2. 算子计算精度需满足生态算子开源精度标准（请参考链接：https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md ），本算子输入输出为单精度复数，比对时实部、虚部分别按 FLOAT32 标准判定：

    | 数据类型 | rtol | atol | required_matched_ratio | max_abs_error_limit |
    |----------|------|------|------------------------|---------------------|
    | COMPLEX64（实部/虚部按 FLOAT32 分量） | 2^-10 (9.77e-4) | 2^-16 (1.53e-5) | 0.99 | 1e-2 或 32 * ULP |

    逐元素通过条件：`|actual - golden| ≤ atol + rtol × |golden|`；当用例同时满足 matched_ratio ≥ required_matched_ratio 且 max_abs_error ≤ max_abs_error_limit 时，判定该用例精度通过。

**补充说明**：

- 本算子为浮点矩阵乘累加运算，非 bit-exact；C 为对称（非厄米特）矩阵，仅 uplo 引用三角参与生成与比对。
- golden 侧 `cblas_csyrk` 的 trans 仅支持不转置/转置两种取值；`ACLBLAS_OP_C` 用例在 golden 侧按转置计算（OP_C 与 OP_T 等价，不共轭），测试工程需做该枚举映射。
- 本算子不含随机数生成，§3.5 的正态/均匀分布是测试输入数据的生成规则，不是算子行为。

### 3.3 性能要求

1. 测试设备：Ascend 950PR。性能数据为 COMPLEX64 输入场景下的平均单次耗时（Avg time，单位 us），须先 warmup 再有效采样 >50 次取平均。
2. 算子在各性能 case 下的平均单次耗时应不高于下表标杆耗时：

| case | n | k | uplo | trans | 标杆耗时（Avg time，us） |
|---|---|---|---|---|---|
| 1 | 512 | 512 | UPPER | N | 126.27 |
| 2 | 1024 | 1024 | UPPER | N | 382.66 |
| 3 | 2048 | 2048 | UPPER | N | 1970.08 |

3. 更多参考性能用例及测试指导见测试用例目录。

### 3.4 内存要求
不涉及。

### 3.5 自验要求

1. 测试工具与方法：使用 ops-blas 仓 test 目录的测试框架完成自验——测试用例以 CSV 文件描述，提供的python 脚本调用C++ GTest 工程加载 CSV 调用 `aclblasCsyrk` 接口执行，精度 golden 由 cblas（Netlib BLAS 复数实现）生成。测试工程参照仓内 `test/syrk/ssyrk/` 扩展复数标量解析（alpha_real/alpha_imag、beta_real/beta_imag 列）。请根据随任务提供的自测用例与测试指导完成自测，并输出自测报告。
2. 本算子参数序列与对标接口 `cublasCsyrk` 的参数序列一致（handle 及参数顺序一一对应，维数参数为 int），无需额外映射说明；OP_C 在 golden 侧的映射见 §3.2 补充说明。
3. 测试用例入参生成规则（基于 §2.4 参数说明）：

| 参数名 | Tensor 值域分布 | Attr 覆盖规则 |
|---|---|---|
| handle | - | 固定为已创建的有效句柄，不随机生成 |
| uplo | - | ACLBLAS_UPPER / ACLBLAS_LOWER 全覆盖 |
| trans | - | ACLBLAS_OP_N / ACLBLAS_OP_T / ACLBLAS_OP_C 全覆盖，与 uplo 正交组合 |
| n | - | 覆盖 0、1、小质数、2 的幂及 2 的幂 ±1、非对齐值，直至大规模 |
| k | - | 同 n 的覆盖规则，且覆盖 k ≠ n 的矩形场景 |
| alpha | 均匀分布 [-5, 5] 占 50%、正态分布（μ∈[-5,5]，σ∈[0.1,2]）占 50%，实部/虚部独立采样；另含特殊值 (0,0)、(1,0)、纯虚数、大值 | - |
| A | 均匀/正态各 50%，实部/虚部独立采样；含 Inf/NaN 特殊值用例 | - |
| lda | - | 覆盖等于最小约束值的紧凑场景（区分 trans=N/T/C 两类最小约束）及多个 padding 场景 |
| beta | 同 alpha 的分布规则 | - |
| C | 均匀/正态各 50%，实部/虚部独立采样，仅 uplo 引用三角有效 | - |
| ldc | - | 同 lda 的覆盖规则 |

4. 自验用例须覆盖：小shape基础用例、shape扫描、填充模式、对齐偏移、边界与负向用例（如零维、空指针 x/y、非法步长、负维度等）、规格允许的INF/NAN场景，以及性能/内存用例。如果任务配套提供的用例没有覆盖要求的场景需要自行补充相应的用例，边界、负向用例及特殊值用例的行为对齐cublas。

## 4. 验收交付件

在社区任务IT系统中提交验收时，需要提交以下交付件：

| 序号 | 交付件名称 | 交付件要求 |
|------|-----------|------------|
| 1 | 算子设计文档 | 1. 设计文档模板：https://gitcode.com/cann/cann-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md ；<br> 2. 在 cann-competitions 仓库（https://gitcode.com/cann/cann-competitions/tree/master/04_tasks/01_community-task-2026/tasklist ）以 PR 形式提交设计文档，通过评审后合入仓库，详细说明见：https://gitcode.com/cann/cann-competitions/blob/master/04_tasks/01_community-task-2026/README.md |
| 2 | 自测用例及测试代码 | 1. 覆盖随任务提供的全部自测用例，需要清晰列出精度测试 case 和性能测试 case；<br> 2. 测试代码中的 readme 文件需要说明测试步骤，保证验收人可以复现测试结果 |
| 3 | 自测报告 | 1. 自测报告模板：https://docs.qq.com/sheet/DUmVWWndaUE12WGFB?tab=BB08J2 ；<br> 2. 需要包含用例参数、精度对比结果及截图（实部/虚部分别）、性能数据及截图、内存占用数据 |
| 4 | 待验收代码地址 | 1. 个人代码仓链接、分支、算子目录；需要在个人仓邀请账号 Ascend-CANN 作为开发者；<br> 2. 需要根据 ops-blas 仓库规范提供算子 readme 文档；README 产品支持表标注 Ascend 950PR：支持；接口声明放入 `include/cann_ops_blas.h`，供其他产品线共用 |

## 5. PR 申请合入

测试通过后，在昇腾算子开源仓提交 PR 申请，申请将开发完成的算子合入该目录：https://gitcode.com/cann/ops-blas （目录 `blas/syrk/arch35/`）；测试代码合入该目录：https://gitcode.com/cann/ops-blas （目录 `test/syrk/csyrk/arch35/`），文件结构参考主仓blas算子测试代码结构（包含csv文件）。

## 6. 参考资料

1. Ascend C算子开发文档：https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/opdevg/Ascendcopdevg/atlas_ascendc_map_10_0002.html ；
2. 算子开发接口文档：https://www.hiascend.com/document/detail/zh/canncommercial/850/API/ascendcopapi/atlasascendc_api_07_0003.html ；
3. Ascend C在线课程：https://www.hiascend.com/developer/courses/detail/1691696509765107713 ；
4. ops-blas 开源仓：https://gitcode.com/cann/ops-blas ；
5. 生态算子开源精度标准：https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md ；
6. cuBLAS 参考文档（cublasCsyrk）：https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-syrk ；
7. Netlib BLAS 参考实现（csyrk）：https://www.netlib.org/blas/csyrk.f 。

## 7. 特别注意事项

1. 所有交付件需提前完成自验证，确认符合验收标准后再提交验收申请；
2. 开发前请务必阅读【社区任务】流程及注意事项：https://gitcode.com/org/cann/discussions/39 ；
3. 接口签名以 ops-blas 仓 `include/cann_ops_blas.h` 中已有声明为准（本算子为新增接口，声明落位与 §2.3 一致），须可与其他产品线共用，禁止定义 950PR 私有平行 API；
4. 性能测试须先 warmup 再有效采样 >50 次取平均。

## 8. 环境获取（无需修改，使用模板原始内容）

1. 使用 hidevlab webIDE 算力：https://hidevlab.huawei.com/online-develop-intro?from=hiascend 。

- **【补充说明】填写示例：本人gitcode账号是 yolo，现在参与社区任务"7月社区任务-aclnnRoll算子开发"，需要申请A2/A3算力进行任务开发。**

  ![环境截图](./pics/zaixiankaifa1.png)
  ![环境截图](./pics/apply.png)

2. 开源仓提供100小时免费时长，请不使用时及时关闭，用时耗尽前请务必保存相关资料，建议及时提交备份。

   ![环境截图](./pics/yunkaifa.png)
3. 如需额外环境资源，请联系昇腾CANN小助手。
