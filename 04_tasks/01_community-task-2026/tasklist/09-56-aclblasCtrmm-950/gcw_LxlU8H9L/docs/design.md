# aclblasCtrmm 算子设计文档（Ascend 950PR）

- **团队**：gcw_LxlU8H9L
- **任务**：《算子实操工坊-杭州站-aclblasCtrmm算子开发(950)》
- **目标仓**：`https://gitcode.com/cann/ops-blas`（分支 `master`）
- **代码位置**：`blas/trmm/arch35/`
- **测试位置**：`test/trmm/ctrmm/arch35/`
- **软件环境**：CANN 9.1.0，SOC = `ascend950`（arch35）
- **实测硬件**：Ascend 950PR

> 本文引用的实测数据均来自本团队自测（数据轨口径见 §可维可测分析），
> 采集环境、命令与原始记录随自测报告提交。仓内源码行号基于写作时点的
> `cann/ops-blas` master，后续演进可能漂移，核对以文件名 + 函数/宏名为准。

# 需求背景（required）

## 需求来源

1. 任务书：`aclblasCtrmm_Atlas950PR_task_doc.md`（CANN 社区任务·算子实操工坊·杭州站）；
2. 任务测试包 `test_cases/`：`ctrmm_test.csv`（1200 条 = 精度 1000 + 性能 200）、
   `gpu_baseline.csv`（200 条 GPU 标杆）、`verify_accuracy.py` / `verify_performance.py`、`gen_csv.py`、`README.md`；
3. 设计文档模板：`cann-ops-competitions/04_tasks/01_community-task-2026/resources/design_template.md`；
4. 目标代码仓：`cann/ops-blas`，接口声明位于 `include/cann_ops_blas.h`，
   复数类型见 `include/cann_ops_blas_common.h`；
5. 对标语义：cuBLAS `cublasCtrmm`（参考实现 Netlib `ctrmm.f`）；
6. 精度标准：任务书 §3.2 + 生态算子开源精度标准（实验标准）；
7. 参考基线（仓内同族）：实数 `aclblasStrmm`（arch35 两相位）、复数 `aclblasCherk`
   （arch35 全设备三相位）、`aclblasCgemm`（arch35 复数 GEMM）。

## 背景介绍

### Ctrmm 功能

三角矩阵-矩阵乘（BLAS Level 3）：

```text
side = LEFT :  C = alpha · op(A) · B        （A 为 m×m 三角阵）
side = RIGHT:  C = alpha · B · op(A)        （A 为 n×n 三角阵）
```

- `op(A) = A`（`trans = N`）、`op(A) = Aᵀ`（`trans = T`）、`op(A) = Aᴴ`（`trans = C`，共轭转置）；
- `B` / `C` 为 m×n 复数矩阵；**离席（out-of-place）写 C**——与 cuBLAS 口径一致：
  调用方将 B 的地址传给 C 即得 BLAS 原地语义，**除 C==B 外不支持其他参数重叠**；
- 仅 `uplo` 指定的 A 三角被引用；`diag = UNIT` 时对角元固定为 1、**不读取** A 对角位置；
- 本算子为乘法（非求解）：**对角元允许为零**，无奇异性检查、无除法；
- `alpha = (0,0)` 时 A/B 不被引用，C 全部置零后返回（Netlib `ctrmm.f` 同型语义）。

计算量 `O(m·n·k)`（k = A 的阶数），属**吞吐型**问题。实现重心在三处：
把 cube 矩乘吞吐喂饱、把三角结构性的无效乘法省掉、把多相位管线的固定成本压住。

### ops-blas 现状分析与差距

**实数模板 `aclblasStrmm`（arch35，直接复用对象）**：`blas/trmm/arch35/` 三件套。
kernel 侧为两相位：Phase1 AIV SIMT mirror kernel（三角镜像打包）+ Phase2 AIC cube GEMM kernel
（Te tensor_api：GM→L1→L0A/L0B→Mmad→L0C→GM）。host 侧对 `side = RIGHT` 采用
m↔n 交换 + side 翻转技巧归一到 LEFT 形态。

**复数前例（arch35）**：

| 前例 | 架构形态 | 可借鉴点 |
| --- | --- | --- |
| `aclblasCherk`（`blas/herk/arch35/`） | **全设备三相位**：AIV 解交织 → 4×cube GEMM → AIV 合成 | 复数面处理范式（无宿主往返）；解交织**从不转置**（`logicalRows = physRowsA`） |
| `aclblasCgemm`（`blas/gemm/arch35/`） | 宿主拆解耦：D2H → H2D 四路 → 4 次实数 cube launch → AIV 合成 | 4M 实数化分解路径；宿主往返是结构性成本，本算子不沿用 |
| `gemm_kernel_do`（`blas/gemm/arch35/gemm_kernel.cpp`） | cube 配方（被 strmm/cherk/gemm 三处复用） | 成熟度高；**自带转置读**（`IsTransA` + `DNExtLayoutPtn`，见 §kernel 侧） |

**关键空白**：仓内**不存在复数 trmm**（`include/cann_ops_blas.h` 仅有实数 `aclblasStrmm`）。
需补齐：复数 alpha 两分量、`trans = C` 真共轭（实数 strmm 把 T/C 同作转置）、
24 组枚举（side2 × uplo2 × trans3 × diag2）全路径、out-of-place C 与 C==B 别名。

**集成契约（必须满足，否则官方脚本取不到产物）**：

| # | 契约 | 依据 |
| --- | --- | --- |
| 1 | 实现源码落 `blas/trmm/arch35/ctrmm_*.cpp` | `cmake/test.cmake` 源码 glob（家族名 trmm 命中，strmm 先例） |
| 2 | 测试文件名/目标名 = `ctrmm_test.cpp` / `ctrmm_test` | 官方脚本按 `f"{OP}_test"` 拼名 |
| 3 | 测试目录 `test/trmm/ctrmm/arch35/`，CSV 同目录同名 | 官方脚本硬编码；`csv_loader.h` 用 `ReplaceFileExtension2Csv(__FILE__)` |
| 4 | 二进制产出可被 `find_binary` 命中 | 官方脚本候选路径 |
| 5 | `bash build.sh --soc=<soc> --ops=ctrmm` 成立 | 官方脚本固定调用 |

### 任务测试包分析

**用例 CSV 契约（22 列）**：
`case_name, description, side, uplo, trans, diag, m, n, alpha_real, alpha_imag, nullA, nullB, nullC,
a_fill, lda, b_fill, ldb, ldc, expect_result, mere_threshold, mare_multiplier, random_seed`

- 性能 `TC_PF_*` 200 条；精度 1000 条（`TC_EX/L0/AB/CV/SQ/ED/FL/LD/TH/WS`）；
- `mere_threshold = 2⁻¹³`、`mare_multiplier = 10.0` 全表一致（实/虚分别判定）；
- 性能用例均在 512MB host 内存预算内（任务书要求 §3.4「不涉及」，README 补充为设计选择）。

**官方两个脚本的角色**：脚本不做数学，只做「构建 + 跑 GTest + 抓 stdout」。

- `verify_accuracy.py`：`--csv` 会**覆盖**仓内 `ctrmm_test.csv`（备份 .bak）⇒ 测试工程必须纯 CSV 驱动；
  以 `--gtest_filter=-*TC_PF*` 跑精度，exit 0 条件 = `failed == 0`（判定在仓内框架内部执行）；
- `verify_performance.py`：`--gtest_filter=*TC_PF*`，以 `[ OK ] <name> (N ms)` 抓**整条用例的整数毫秒
  wall-clock**，`ratio = gpu_ms / npu_ms ≥ 0.4` 判 PASS；脚本自注「含 host 准备 + kernel + golden
  计算 + 比对，为保守上界」。基线空缺或 `npu ≤ 0`（含 GTest FAILED）⇒ NO_REF。

# 需求分析（required）

## 需求描述

在 `cann/ops-blas` 的 `blas/trmm/arch35/` 目录族内实现 `aclblasCtrmm`：句柄式 BLAS 接口
（`aclblasHandle_t` 携带 stream），Host 侧完成参数校验与 tiling，kernel 直调方式下发 NPU。
公开接口签名与 cuBLAS `cublasCtrmm` 逐参数对应（整数维参数 `int`），
在 `include/cann_ops_blas.h` 新增声明：

```cpp
aclblasStatus_t aclblasCtrmm(
    aclblasHandle_t handle,
    aclblasSideMode_t side,
    aclblasFillMode_t uplo,
    aclblasOperation_t trans,
    aclblasDiagType_t diag,
    int m, int n,
    const aclblasComplex* alpha,
    const aclblasComplex* A, int lda,
    const aclblasComplex* B, int ldb,
    aclblasComplex* C, int ldc);
```

交付物：算子实现（Host + Kernel + Tiling）、算子 README（产品支持表标注 Ascend 950PR：支持）、
纯 CSV 驱动的 C++ GTest 测试工程、覆盖官方全部用例的自测报告、本设计文档。

## 需求拆解

1. **接口**：如上签名；`alpha` 支持 Host 或 Device 指针；
2. **参数校验**（顺序对齐 cuBLAS/Netlib 检查序）：`handle == nullptr` → `HANDLE_IS_NULLPTR`；
   `m < 0` / `n < 0` / 枚举非法 → `INVALID_VALUE`；
   `lda`（LEFT: ≥max(1,m) / RIGHT: ≥max(1,n)）、`ldb/ldc ≥ max(1,m)` 不足 → `INVALID_VALUE`；
   `alpha`/`C` 为 nullptr（m>0 且 n>0）→ `INVALID_VALUE`（**含 alpha=(0,0) 的 C nullptr**）；
   alpha 非零且 m>0、n>0 时 `A`/`B` 为 nullptr → `INVALID_VALUE`；
3. **24 组枚举路径**：side2 × uplo2 × trans3 × diag2 全支持；`trans = C` 真共轭；
4. **diag 语义**：UNIT 时**绝不读取**对角位置（对角为 NaN/Inf 亦不得影响输出，按 1 参与）；
   NON_UNIT 直接参与乘法（允许为零，无除法）；
5. **alpha 语义**：(0,0) → 不读 A、B 不必有效、C 置零；一般复数 alpha 在合成相位统一乘入；
6. **out-of-place 与别名**：结果写 C；C==B 同指针合法（原地），其余重叠不支持；
7. **快返**：m=0 或 n=0 合法 no-op 返回 SUCCESS（在全部校验之后）；
8. **布局**：全列主序 ND；非连续/broadcast/dynamic shape 本批不涉及；
9. **测试工程**：`test/trmm/ctrmm/`（见 §测试工程），参照实数 `test/trmm/strmm/` 新建复数变体，纯 CSV 驱动；
10. **文档**：算子 README（能力矩阵/接口说明/限制/复现步骤）与本设计文档。

# 详细设计（required）

## 算子分析

### 数学公式

`side = LEFT`（A 为 m 阶）：`C(i,j) = alpha · Σ_{k=0}^{m−1} op(A)(i,k) · B(k,j)`；
`side = RIGHT`（A 为 n 阶）：`C(i,j) = alpha · Σ_{k=0}^{n−1} B(i,k) · op(A)(k,j)`。

`op(A)` 仅在 uplo 指定三角内非零 ⇒ **三角结构性零元**在实现中可被利用以减少无效计算
（具体利用方式属实现细节，不在本文展开）。

复数乘加的标准分解（实/虚部各由实数乘加组合）：
`re = f1(reA,imA,reB,imB)`，`im = f2(reA,imA,reB,imB)`；实现可选择不同的实数乘法次数与组合公式。

**特殊值的传播边界**：当参与乘法的元素含 Inf/NaN 时，不同分解公式的非数传播位置可能不同
⇒ **受控值域用例与特殊值用例必须走各自适配的计算路径**（由路径选择策略保证，见 kernel 侧设计）。
（实测特殊填充用例均为小尺寸，与门限天然隔离，见 §风险与对策 R1）。

### 支持数据类型

| 项 | 取值 |
| --- | --- |
| 元素类型 | COMPLEX64（`aclblasComplex`，实/虚各 float32，8 字节） |
| 标量/维数参数 | `int`（`m`、`n`、`lda`、`ldb`、`ldc`） |
| 精度判定 | 实部、虚部分别按 FLOAT32 标准判定 |

### 支持形状与取值

| 参数 | 约束 |
| --- | --- |
| `m`/`n` | ≥ 0；= 0 合法 no-op；用例覆盖 1×1 → 4096×4096（含非对齐、宽窄网格、fat/thin 非方阵） |
| `A` | 列主序三角，仅 uplo 三角引用；`lda` 紧凑与 padding 两种 |
| `B`/`C` | 列主序 m×n；`ldb`/`ldc ≥ max(1,m)`；C==B 同指针合法 |
| `alpha` | 实/虚取值 FLOAT32 全集；(0,0)/(1,0)/(−1,0)/纯虚/大值均有专项用例 |
| 内存 | 单用例 host ≤ 512MB |

## 算子实现

### 实现方案

#### 3.2.1 host侧设计：

**1. 入口与校验**（顺序对齐 cuBLAS/Netlib，负向 20 条逐条对拍）：

```text
1) handle == nullptr                          → HANDLE_IS_NULLPTR
2) m < 0 / n < 0                              → INVALID_VALUE
3) side/uplo/trans/diag 枚举非法               → INVALID_VALUE
4) lda/ldb/ldc 前导维不足                     → INVALID_VALUE
5) alpha == nullptr                           → INVALID_VALUE
6) m>0 ∧ n>0 ∧ C == nullptr                  → INVALID_VALUE（含 alpha=(0,0)）
7) alpha≠(0,0) ∧ m>0 ∧ n>0 ∧ (A|B)==nullptr  → INVALID_VALUE
8) m==0 ∨ n==0                                → SUCCESS（no-op 快返）
9) alpha==(0,0)                               → 置零 kernel（C 清零，不读 A/B）
10) 一般路径                                   → 组装 tiling → 下发
```

**2. tiling 与传参**：`CtrmmTilingData` / `CtrmmSplitMirrorTilingData` / `CtrmmPhaseBTilingData` /
`CtrmmCombineTilingData` 四个结构体覆盖三相位；**传参形态 = 结构体值传递（by-value）**，
避免 host 常驻 GM 参数块 + 逐调用 H2D 的多流竞态。`alpha` 两分量由 host 读出后编入 tiling
（Device 指针经一次 `aclrtMemcpy` 同步取回，仅发生在下发前）。

**3. workspace**：三相位管线的解交织/部分积缓冲由 handle workspace 分配；
小 case 标量路径 workspace = 0。

**4. kernel 下发**：`ctrmm_kernel.h` / `ctrmm_e3.h` 声明入口；Host 侧不做流同步
（异步语义由调用方负责）。

**host 侧分核策略**：大 case 三相位 = AIV（split/mirror）→ AIC（cube ×3）→ AIV（combine），
各相位块数由 tiling 按形状算出（AIC 按 m×n tile 网格均分）。
`side = RIGHT` 沿用 strmm 的 m↔n 交换 + side 翻转技巧归一到 LEFT 形态。

**数据分块与内存优化策略**：

| 项 | 策略（抽象描述） |
| --- | --- |
| 分块 | 大形状按 **L1/L0 两级分块** 切分（tile 形状经参数扫描选定；不同长宽比按自适应规则选取，具体取值属实现细节） |
| 分核 | 按输出 tile 网格均分到 cube 核；AIV 侧按行块均分 |
| 数据搬运 | 复用 tensor_api 的分块搬运与向量原语；K 方向分块流式装载 |
| 路径选择 | 按形状规模与 alpha 特例在 host 侧选定路径（阈值取值属实现细节，编译期固化） |

**路径分派**：按形状规模（大/小）与 alpha 特例（(0,0) 置零）在 host 侧选定；
24 组枚举（side/uplo/trans/diag）**不进运行期分支**——大 case 在预处理阶段归一，
小 case 用具名特化 kernel（编译期固化方向/索引/共轭）。

##### 3. tilingkey 规划策略

| tilingKey 维度 | 取值 | 说明 |
| --- | --- | --- |
| 路径选择 | 标量路径 / 分块管线路径 / alpha 置零路径 | host 按形状规模与 alpha 特判决定；**编译期固化**，kernel 内无运行期枚举分支 |
| side | LEFT / RIGHT | RIGHT 在 host 侧归一到 LEFT（m↔n 交换 + side 翻转） |
| 方向/索引 | side × uplo × trans × diag 具名特化 | 小形状按枚举组合具名实例化；大形状在预处理阶段归一，枚举不进 kernel 分支 |

**原则**：tilingKey 只承载"会改变 kernel 控制流的 host 侧信息"；枚举组合通过**预处理归一 + 具名特化**消解，
避免 kernel 内运行期分支带来的指令膨胀与小形状固定开销。

#### 3.2.2 kernel侧设计：
**编程模型映射（对齐 Ascend C 惯例）**：

| 阶段 | 本算子内容 |
| --- | --- |
| Init | 取 tiling、按 tilingKey 选定路径与具名特化、分配 UB/L1 缓冲、初始化同步事件标志 |
| Process · CopyIn | Phase-A 的 GM→UB 复数块搬入（tensor_api `CopyGM2UB`）；Phase-B 的 GM→L1（`CopyGM2L1`）与 L1→L0A/L0B |
| Process · Compute | 预处理：复数解交织 + 三角结构归一（AIV 向量原语）；主计算：分块矩阵乘 `Mmad`（AIC cube）；收尾：结果合成与 alpha 标量乘（AIV 向量原语） |
| Process · CopyOut | Phase-A 三平面 UB→GM；Phase-B `Fixpipe` L0C→GM（临时积）；Phase-C 结果 UB→GM |



**总体架构（两条计算路径 + 一条特例）**：

```text
路径 S（小形状）  ：单 kernel 标量乘加（Netlib 同序）；
                   特殊值 / 极值 / alpha 语义天然安全。
路径 Z（alpha=0） ：置零写出 C，不读 A/B。
路径 L（大形状）  ：AIV 预处理（复数解交织为实/虚平面 + 三角结构归一）
                    → AIC 分块矩阵乘（tensor 核）
                    → AIV 收尾合成（实/虚重组 + alpha 复数乘 + 写 C）。
```

**关键设计点**：

1. **枚举归一**：24 组枚举（side/uplo/trans/diag）在预处理阶段统一消化
   （side 翻转、三角镜像、转置/共轭、UNIT 对角），使 cube 侧只面对一种规整形态；
2. **预处理与收尾**：复数解交织、三角结构填零、UNIT 对角写入、共轭取负、
   结果重组与 alpha 标量乘——均以 tensor_api 向量原语实现（分块搬运 + 向量变换）；
3. **分块矩阵乘**：cube 侧按 L1/L0 两级分块执行 `Mmad`，K 方向流式装载；
   三角结构带来的无效区域在分块层面规避（**具体裁剪策略与乘积组织方式属实现细节，不在本文展开**）；
4. **路径隔离**：受控值域与特殊值域用例走不同路径，保证 Inf/NaN 与极端值的行为与 cublas 一致；
5. **UNIT 语义**：对角位置在预处理阶段按 1 写入，全链不读取 A 的对角地址；
6. **B/C 别名**：预处理阶段对 B 走 workspace 副本，避免 C 的写入破坏尚未消费的 B 数据，
   专项回归用例覆盖（`AliasInPlaceCB`）；
7. **小形状地板**：路径 S 单 kernel、参数 by-value 传递，小形状档位控制在数十µs 量级。

**复数平面布局（三相位数据流）**：

```text
A、B 复数（列主序） ──预处理──▶ 实部/虚部平面（float32）
                      ──分块矩阵乘──▶  实数乘积（一个或多个中间结果平面）
                      ──收尾合成──▶  C（复数，含 alpha 复数乘）
```
（中间结果的个数与组合公式属实现细节。）

**关于转置读（`trans = T/C`）**：Phase-A 物化 `op(A)` 的三角掩码与共轭。
仓内 GEMM 模块自带编译期转置读（`gemm_kernel.cpp` 中
`using LayoutGM_A = std::conditional_t<IsTransA, te::DNExtLayoutPtn, te::NDExtLayoutPtn>`），
本实现的 `trans` 处理与之口径一致（`effectiveUplo = uplo ^ trans`，与 CATLASS TRMM 同式）。

#### 测试工程与验收链路接入

**工程布局**：

```text
test/trmm/ctrmm/
├── CMakeLists.txt              # ops_blas_add_gtest_tests(${OPS_BLAS} ctrmm_test)
├── ctrmm_param.h               # CSV 22 列 → 参数结构体
├── ctrmm_golden.h              # golden = Netlib cblas_ctrmm
└── arch35/
    ├── ctrmm_test.cpp          # 精度 GTest（纯 CSV 驱动）
    ├── ctrmm_npu_wrapper.h     # 直调 aclblasCtrmm（设备内存 + stream 同步）
    ├── ctrmm_test_bench.cpp    # 数据轨采样入口（warmup + 50 采样）
    ├── ctrmm_test_bench.csv    # 性能用例 CSV
    └── ctrmm_test.csv          # 官方 CSV 落地位置（同目录同名）
```

1. **CSV 驱动**：`INSTANTIATE_TEST_SUITE_P(..., ValuesIn(GetCasesFromCsv<CtrmmParam>(
   ReplaceFileExtension2Csv(__FILE__))))`——参数名即 `case_name`；**禁止硬编码用例参数**
   （官方 CSV 会被 `install_csv` 覆盖）；
2. **参数解析**：对齐 `strmm_param.h` 的 ReadMap，扩 `alpha_real/alpha_imag` 与 `nullA/nullB/nullC`；
   负向用例的非法枚举值保留原样；
3. **golden**：`cblas_ctrmm`（Netlib BLAS 复数实现）。cblas 为原地形式
   （`B := alpha*op(A)*B`），本任务的离席 C 由「先把 B 每列有效 m 行拷入 C，再调 cblas_ctrmm」服务；
   模块内注释明示该委托关系；
4. **精度判定（双策略同施，任一不过即失败）**：CSV 阈值列 MERE/MARE（实/虚分别，2⁻¹³ / ×10）
   + 任务书 §3.2 元素级判据（rtol 2⁻¹⁰ / atol 2⁻¹⁶ / matched_ratio 0.99 / max(1e-2, 32·ULP)）；
   Inf/NaN 位置 mismatch 为硬失败；
5. **负向与边界**：`expect_result` 非 SUCCESS 断言返回码（顺序对拍）；m=0/n=0 no-op 断言 C 不被触碰；
   UNIT × 对角 NaN/Inf 断言输出无污染；alpha=(0,0) 断言 C 全零且不读 A/B（A/B 传毒值缓冲验证）；
6. **性能双轨**：
   - **验收轨**：`TC_PF_*` 用例体（GTest 命名，官方脚本可抓）；
   - **数据轨**：独立采样入口（warmup 10 次 + 50 次有效采样，逐次 event 计时），
     输出逐例平均/中位，是自测报告的性能数据源；
7. **环境记录与最小复现链路**：CANN 版本/SOC/驱动记录；三步链路 =
   `bash build.sh --soc=ascend950 --ops=ctrmm` → `verify_accuracy.py` → `verify_performance.py`。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Ascend 950PR（SOC = `ascend950`，arch35） | √ |

## 算子约束限制

- 核心计算全部在 NPU kernel 执行，**无 CPU fallback**；
- `diag = UNIT` 时对角位置不被访问（NaN/Inf 均不影响结果）；NON_UNIT 对角允许为零（乘法语义）；
- 除 C==B 外不支持参数重叠；
- 非连续 Tensor、broadcast、dynamic shape：本批次不涉及；
- 单用例 host 内存 ≤ 512MB（workspace 全部设备侧）。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | golden = Netlib `cblas_ctrmm`；输出 C 全矩阵逐元素，实/虚分别判定：CSV 阈值 MERE/MARE（2⁻¹³ / ×10）与元素级判据（rtol 2⁻¹⁰、atol 2⁻¹⁶、matched_ratio ≥ 0.99、max_abs_error ≤ 1e-2 或 32·ULP）**双策略同施** | 任务书 §3.2 + 用例包 + 仓内框架 |
| 性能标准 | 数据轨：µs 级（warmup + >50 次有效采样取平均），对标 `gpu_baseline.csv` 的 `gpu_ms`（**ratio = gpu_us / npu_us ≥ 0.4**）；验收脚本口径见下 | 任务书 §3.3 + `gpu_baseline.csv` |
| 内存标准 | 不涉及（任务书 §3.4） | 任务书 §3.4 |

### 实测结果（数据轨，五轮中位）

**精度**：精度套件（1000 条非 TC_PF 用例）**1003/1003 全绿**（含 3 条测试工程内的结构性
回归用例外挂），golden 走 Netlib `cblas_ctrmm`。

**性能（全量 200 条 TC_PF，五轮中位）**：

| 指标 | 实测值 |
| --- | --- |
| **达标（ratio ≥ 0.4）** | **172 / 200（86%）** |
| 逐轮 PASS | 171 / 172 / 171 / 173 / 172 |
| 未达标 | 28 条（trans 分布：T 13 / C 12 / N 3） |

**任务书 §3.3 五条典型 case**（标杆值为任务书原文）：

| # | 形状与枚举 | 标杆耗时（µs） | 我方实测中位（µs） | ratio | 判定 |
| --- | --- | --- | --- | --- | --- |
| 1 | 256×256 LEFT/UPPER/N/NON_UNIT | 120.18 | 62.9 | **0.764** | PASS |
| 2 | 512×512 LEFT/LOWER/N/NON_UNIT | 268.37 | 267.9 | **0.401** | PASS |
| 3 | 1024×1024 RIGHT/UPPER/T/NON_UNIT | 690.54 | 548.4 | **0.504** | PASS |
| 4 | 2048×2048 LEFT/LOWER/C/NON_UNIT | 2945.91 | 1957.7 | **0.600** | PASS |
| 5 | 2048×2048 RIGHT/LOWER/N/UNIT | 2963.64 | 2477.3 | **0.479** | PASS |

**其余锚点（五轮中位）**：128² **0.822**、4096² **0.464**、512² T/C 族 0.383-0.392（未达标）。

**过程性说明**：实现由多条互相独立的优化手段叠加而成（分块与流水、结构归一、
形状自适应等），各步均以 A/B 对照验证（1003 条精度零回退 + 逐条性能中位）。
**各手段的具体内容与增益分解属实现细节，不在本文展开**；对外仅提供验收所需的结果数据。

### 验收脚本口径的说明（双轨留痕）

官方 `verify_performance.py` 抓取的是**整条 GTest 用例的整数毫秒 wall-clock**
（含 host 准备 + 采样循环 + 比对；脚本自注为「保守上界」），而任务书 §3.3 要求的是
**warmup + >50 次采样的平均单次耗时（µs）**。两者对同一批用例的口径不同：

- 我方按**双轨**准备证据：① 数据轨（上表，µs 口径，五轮中位）；② 验收轨整条 wall-clock 原始值
  （逐条 CSV，随自测报告提交）；
- 我方逐条计算了脚本口径下的可达性：**200 条中 0 条在脚本口径下可达标**
  （因为整条用例墙钟必然包含 >50 次采样调用，其下界约 51×单次耗时，而达标上限仅 2.5×`gpu_ms`；
  逐条差距 19× 至 1456×，中位 42×）。该逐条数据（含 `kernel_us`、`base_ms`、墙钟、
  下界、差距倍数）作为附件提交，供验收方核对口径；
- 同时提交 `msprof` 采集的 kernel-only 耗时，作为「单次耗时」的独立证据。
- 已做的配套优化：把整条用例中的 host 侧数据准备（复数矩阵填充）优化到 −45%~−66%
  （2048² 由 433ms 降到 65ms），使墙钟数字尽量接近 kernel 真实成本。

> 口径归属问题（脚本 wall-clock vs 任务书 µs 采样）我方无法自行裁定，已在提交材料中以中性
> 方式提出，并同时提交两轨数据与原生命令，便于验收方复核。

## 测试方案

1. **功能与精度**：纯 CSV 驱动全 1000 条（24 组枚举 × 形状 × 前导维 × alpha 特值 × 填充族），
   双策略判定，逐例输出中间量；
2. **特殊值（零容忍）**：B 全 Inf / 全 NaN（传播位置对齐 golden）；A 极值；
   UNIT × 对角 NaN/Inf（输出必须无污染）；alpha=(0,0)（A/B 毒值缓冲不被消费）；
3. **负向用例**：非法枚举 / m<0 / n<0 / 前导维不足 / nullA/nullB/nullC 组合 / handle nullptr，
   逐条断言返回码与校验顺序；
4. **结构性回归**：同流连续两次调用不同参数中间不同步；**C==B 别名**（原地语义正确性）；
   为常驻回归用例；
5. **路径选择验证**：两条计算路径的规模边界两侧各取满枚举矩阵，验证特殊填充用例落在其适配路径；
6. **性能**：数据轨（采样统计 + 五轮中位）+ 验收轨（整条墙钟原始值）+ msprof kernel-only 拆分；
7. **测量纪律（写进测试 README）**：性能结论以**多轮中位**给出（本次以五轮，
   骑线用例判据为五轮中位 ≥ 0.4）；单轮数字不作为结论（实测同版本跨重建的单轮波动可达 30-40%）。

## 兼容性分析

1. 公开接口为**新增声明**，不改既有接口签名与语义；实数 `aclblasStrmm` 零行为回归
   （ctrmm 全部为新增文件，不动 strmm/cherk/gemm 任何存量文件）；
2. 新增文件集中在 `blas/trmm/arch35/ctrmm_*` 与 `test/trmm/ctrmm/`；
3. 不引入进程级全局可变状态与跨算子锁。

# 风险与对策

| # | 风险 | 对策 |
| --- | --- | --- |
| R1 | **特殊值的 Inf/NaN 传播差异**（§数学公式 IEEE 边界例）：不同实数分解公式的非数传播位置可能不同 | 路径选择只依据形状规模；实测特殊值用例均落在小形状路径；1003 条精度回归含极值/Inf/NaN 用例常驻验证 |
| R2 | **24 组枚举方向/索引错**（trans 面 × side 翻转 × uplo 镜像） | 枚举物化集中 Phase-A 一处 + 小 case 具名特化编译期固化；24 组 × 小尺寸全量对拍先行 |
| R3 | **性能脚本墙钟口径**：整条用例整数毫秒含采样循环 | 双轨留痕（数据轨五轮中位 + 验收轨原始墙钟）+ 逐条可达性分析附件 + msprof kernel-only 证据 + 中性口径提问 |
| R4 | **T/C 枚举族性能未达标**（28 条 GAP 中 T 13 + C 12，ratio 0.383-0.398） | 已通过对照实验排除若干假设（转置读路径本身无额外开销）；后续方向为翻转三角形态下的分块效率（**待优化项，当前计入未达标**） |
| R5 | **UNIT 读取对角** | 打包期写 1 + 对角 NaN/Inf 守卫用例回归（输出必须与干净输入一致） |
| R6 | **C==B 别名写序** | Phase-A 对 B 走 workspace 副本 + 别名专项回归用例 |
| R7 | **golden 工具链差异**（cblas 版本） | 记录测试环境 cblas 实现与版本；以任务包 golden 与仓内 `test/frame` 判定为准 |
| R8 | **测量噪声导致错误结论** | 性能结论一律多轮中位（骑线 5 轮）；同版本内五轮离散度实测 ±0.001，跨版本差异可达 ±0.3，据此判定回归 |
| R9 | **小形状（≤32²）地板不可达** | 该档预算 2.8-9.7µs，单次调用 + 一次同步已接近该量级；当前 4²-32² 4 条未达标，如实计入 |

# 相关工作

- **三角矩阵乘的并行化**：tile 级三角分块（对角块串行/非对角块并行 GEMM）、结构性零元跳过
  ；
- **CATLASS TRMM**（`cann/catlass`，`include/catlass/gemm/kernel/trmm.hpp` + `examples/76_trmm/`）：
  其 `effectiveUplo = uplo ^ trans` 与本实现同式；`trans` 通过**操作数布局 dispatch** 实现
  （Row/Row、Col/Row、Row/Col）；`SelectTileVariant` 按形状选 tile；其支持的枚举为
  `trans ∈ {no, trans}`、`diag = nonunit`；
- **cuBLAS CTRMM / Netlib ctrmm.f**：离席 C 形态 + 调用方传 B 地址得原地功能（本任务接口口径来源）；
- **仓内同族参考**：实数 `aclblasStrmm`（两相位镜像 + cube）、`aclblasCherk`
  （全设备三相位复数解交织/合成范式，解交织不转置）、`aclblasCgemm`（4M 实数化分解先例）、
  `gemm_kernel_do`（cube 配方，自带 `IsTransA` 转置读）——本设计 = 上述经验的组合与三角特化。
