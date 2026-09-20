# aclblasCtpmv 算子设计文档（Ascend 950PR）

> 任务：算子实操工坊-上海站-aclblasCtpmv算子开发(950)
> 目标仓：`cann/ops-blas`，实现目录 `blas/tpmv/arch35/`，测试目录 `test/tpmv/ctpmv/arch35/`
> 代码分支：个人仓 `wib/ops-blas` `dev-ctpmv-950`

# 需求背景（required）

## 需求来源

本需求来自 CANN 社区任务「算子实操工坊-上海站-aclblasCtpmv算子开发（950）」任务书：
在昇腾 950PR（arch35 / `dav-3510`）上使用 Ascend C 实现单精度复数（complex64）三角压缩存储矩阵-向量乘算子
`aclblasCtpmv`，语义、参数序列对标 cuBLAS `cublasCtpmv` 与 Netlib `ctpmv.f`，公共声明新增到
`include/cann_ops_blas.h`（与同族实数接口 `aclblasStpmv` 逐参数对齐，不定义 950PR 私有平行接口）。

- 适配硬件：Ascend 950PR（AIV 56 核），CANN 9.1.0
- 编译目标：`--npu-arch=dav-3510`（bisheng）
- golden：cblas（Debian libblas 3.10.0，Netlib `ctpmv`）

## 背景介绍

### 算子功能

`aclblasCtpmv` 属于 BLAS Level-2，计算 `x := op(A) * x`：

- A 为 n×n 单精度复数三角矩阵，按列优先 packed（压缩）存放，仅占 `n(n+1)/2` 个 complex64，无前导维 lda；
- `trans = OP_N / OP_T / OP_C` 分别对应 `A`、`Aᵀ`（不共轭）、`Aᴴ = conj(Aᵀ)`；
- `diag = UNIT` 时主对角按 `(1,0)` 参与计算，**不得读取 AP 对角位置**；
- x 为原地输入输出，`incx` 支持正/负步长；`n = 0` 为合法 no-op。

### packed 存储布局与一处任务书勘误

0-based 列优先 packed 索引（与 Netlib/cblas 一致）：

- UPPER：`A(i,j) = AP[i + j(j+1)/2]`（`i <= j`）
- LOWER：`A(i,j) = AP[i + (2n-j-1)j/2]`（`i >= j`）

任务书 §2.1 中 LOWER 写作 `i + (2n-j+1)j/2`，该式实际是第 j 列的**列首地址**，作为元素地址多算了 j：
n=3 时 A(2,2) 会算出下标 7，而 AP 只有 6 个元素（合法下标 0~5），直接越界。
本实现以 Netlib/cblas 为准（`-1` 式），并以此为验收接口定义；错误公式仅影响 LOWER 用例，设计文档特此勘误留痕。

### 现有能力与缺口

仓内已有实数 `aclblasStpmv`（A2/A3），可复用 handle/stream/workspace/错误码/测试框架，但不能简单替换数据类型：

- 复数乘加需要实虚部拆分与保序累加，运算与搬运量高于实数路径；
- `OP_C` 只能共轭矩阵元素，不能共轭向量；
- `diag=UNIT` 必须从 AP 有效访问范围中排除对角（对角可能是 NaN/任意值）；
- 原地语义要求主计算读完旧 x 之前不得覆盖 x；
- 950 是新一代硬件，编程模型与 A2/A3 不同（见 §详细设计「编程模型选型」）。

### 本次交付范围

- `include/cann_ops_blas.h` 新增 `aclblasCtpmv` 声明；
- `blas/tpmv/arch35/`：Host、TilingData、RegBase 主线 Kernel、SIMT 回退 Kernel；
- `test/tpmv/ctpmv/arch35/`：CSV 驱动 GTest（1200 条）、性能 benchmark、UNIT 对角毒化探针、CMake 注册；
- `blas/tpmv/README.md` 产品支持表标注 Ascend 950PR；
- 设计文档、自测报告等任务书交付件。

不在本次范围：批处理/广播/稀疏/非 packed 语义、超出 `incx` 语义的非连续张量、950PR 私有平行 API。

# 需求分析（required）

## 接口与参数

```cpp
aclblasStatus_t aclblasCtpmv(
    aclblasHandle_t handle,
    aclblasFillMode_t uplo,
    aclblasOperation_t trans,
    aclblasDiagType_t diag,
    int n,
    const aclblasComplex* AP,
    aclblasComplex* x,
    int incx);
```

| 参数 | 合法范围 | 异常行为 |
| --- | --- | --- |
| handle | 有效句柄（携带 stream/workspace） | 空返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| uplo | UPPER / LOWER | 非法返回 `ACLBLAS_STATUS_INVALID_ENUM` |
| trans | OP_N / OP_T / OP_C | 非法返回 `ACLBLAS_STATUS_INVALID_ENUM` |
| diag | NON_UNIT / UNIT | 非法返回 `ACLBLAS_STATUS_INVALID_ENUM` |
| n | `n >= 0`，`n=0` 合法 no-op | `n < 0` 返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| AP | `n>0` 时非空，长度 `n(n+1)/2`，只读 | 空返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| x | `n>0` 时非空，物理跨度 `1+(n-1)|incx|`，原地 | 空返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| incx | `incx != 0` | `incx == 0` 返回 `ACLBLAS_STATUS_INVALID_VALUE` |

参数校验顺序对齐 Netlib（枚举 → `n<0` → `incx==0` → `n==0` → AP/x 指针），
与测试 golden `aclblasCtpmv_cpu` 完全一致，保证组合非法参数下状态码确定。

## 需求拆解

| 编号 | 需求 | 设计落点 |
| --- | --- | --- |
| R1 | complex64：实虚部按 FP32 参与乘加 | 复数拆分/乘加/归约 |
| R2 | UPPER/LOWER packed 语义 | packed 索引函数（uint64 中间量） |
| R3 | `OP_N` | N 路径：按列贡献 |
| R4 | `OP_T`（不共轭） | T/C 路径 |
| R5 | `OP_C`（只共轭矩阵） | T/C 路径 + 对角共轭 |
| R6 | UNIT/NON_UNIT | 对角处理，UNIT 不读 AP |
| R7 | `incx` 正/负/非单位 | SIMT 通用步长路径 + K2 scatter |
| R8 | 原地语义 | K1 写 workspace，K2 统一写回 |
| R9 | `n=0` quick return | Host 早退，不引用 AP/x |
| R10 | 地址不溢出 | 全部索引 uint64 中间量 |
| R11 | 异步执行 | Host 正常路径仅最终同步一次（K2 后） |
| R12 | 可复现验收 | CSV/脚本/CMake/benchmark/探针 |

## 精度与性能要求

精度（任务书 §3.2）：golden 由 cblas/Netlib 生成，实/虚部分别按 FLOAT32 判定：
`|a-g| <= 2^-16 + 2^-10·|g|`，`matched_ratio >= 0.99`，且逐元素硬门禁 `max(1e-2, 32·ULP(|g|))`；
测试工程实现为仓内框架 `verify.h` 的 `MIXED_TOLERANCE`（与全仓 50 个测试文件一致）。

性能（任务书 §3.3）：4 个 case，warmup 后有效采样 >50 次取平均：

| case | n | uplo | trans | diag | incx | 标杆（us） |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 512 | UPPER | N | NON_UNIT | 1 | 44.89 |
| 2 | 1024 | LOWER | N | NON_UNIT | 1 | 88.46 |
| 3 | 2048 | UPPER | T | NON_UNIT | 1 | 184.20 |
| 4 | 4096 | LOWER | C | NON_UNIT | 1 | 1842.81 |

# 详细设计（required）

## 算子分析

### 数学语义

```text
y[i] = Σ_j op(A)[i,j] · x_old[j],   0 <= i,j < n
x    = y
```

UPPER 时 `j ∈ [i, n)`，LOWER 时 `j ∈ [0, i]`。`OP_T` 只交换行列，`OP_C` 对矩阵元素取共轭，
向量不共轭。复乘语义：

```text
(a·b).re = a.re·b.re - a.im·b.im
(a·b).im = a.re·b.im + a.im·b.re
```

### 原地保护

K1 只读旧 x，把逻辑结果连续写入 handle workspace；K2（scatter）再按 `incx` 映射写回 x。
两 kernel 同一 stream 顺序执行，因此 K1 完成前不会覆盖调用方输入。

### 步长映射

```text
incx > 0: physical(i) = i·|incx|
incx < 0: physical(i) = (n-1-i)·|incx|
```

仅写有效逻辑位置，步长空洞保持不变。

## 编程模型选型（950 实测）

950 上并存三种模型，本实现的选择依据实测：

| 模型 | 结论 |
| --- | --- |
| SIMT（`__simt_vf__`） | 标量 GM load 阻塞（约 215 cycles/项），加多少预取都无 MLP，无法达到标杆，仅作回退/非 `incx=1` 路径 |
| MemBase（LocalTensor 向量） | 解交织/同步成本高，早期 288 行实现失败回退 |
| **RegBase（`AscendC::Reg`，256B=64×FP32 lane）** | **950 新引入**，以「并行轴 = 独立输出」映射，精度与性能同时达标，作为主线 |

## Kernel 设计（RegBase 主线）

### 并行轴与分块

- 64 个 FP32 lane 映射到 **32 个独立输出**（复数占用 re/im 两组 lane，W=32）；
- 每个输出内部严格保持 Netlib 累加顺序 → 精度门禁可过；
- 累加器常驻寄存器，跨面板只把 FP32 状态落 UB。

### trans=N 路径（列切片）

按列（UPPER 递增 / LOWER 递减）把 packed 三角切片经 `DataCopyPad` 搬入 UB 面板，
向量加更新 W 个输出：

- 列切片起点 `a` 对齐到 `a0 = 4·floor(a/4)`，`left = 2(a-a0)`，`right = (-(2h+left)) mod 8`，
  目的地址 32B 对齐，保证车道对齐；
- VF 内用「前缀掩码差」取 `[a, a+h)` 区间，`MERGING` 累加保尾部短链累加值。

### trans=T/C 路径（列槽 + masked Gather）

按「对角距离 d × 输出列」组织列槽面板：K=64、P=2K+8=136（float 元素）、`δb = 2·floor(b/16)`。

- UPPER 索引 `b·P + δ + 2·(cb-1-t)` 递减；LOWER 索引 `b·P + δ + 2t` 递增；
- A 与 x 均用 masked `Reg::Gather` 取数（x 的普通 DINTLV 加载有 32B 对齐约束，已探针证伪）；
- 每 tile 独立输出交织 `StoreVf`。

### 对角处理

NON_UNIT 将对角折入面板 d=0（对角仍先行参与，等价于参考顺序）；UNIT 从 DMA 源区间/索引中排除，
**不读 AP 对角**。UNIT 的复数对角 `(1,0)` 共轭后不变，`OP_C` 无需特殊处理。

### 设备相关的三个关键实现决策（与参考设计的有意偏离）

1. **对角折入面板 d=0**：参考设计用独立 diagQueue + 逐列 diag DMA；实测该路径是间歇性 fault 来源，
   改为折入面板后稳定（对角仍先行，数值等价）。
2. **每 lane 元数据改为 VF 内 `Arange` 现算**：参考设计用「标量写 UB 元数据 + 单独可见性依赖」；
   本设备上 `S_V/V_S`、`PIPE_ALL`、`PIPE_V` 全部无法可靠建立该依赖（间歇 fault），
   最终在 VF 内用 `Arange` + int32 算术现算索引与计数器，彻底删除标量元数据依赖。
3. **无效 lane 索引用 `Maxs` 钳制**：参考设计建议用 `Select` 把无效索引换成 0/1；
   实测 `Select` 在本设备后端崩溃或不生效，改用 `Maxs` 把索引钳到合法下界。

### RegBase 实测设备语义（探针结论）

- `Reg::Gather` 越界必 fault，**mask 不阻断访问**；无效 lane 的索引必须按构造在范围内；
- `Reg::Sub/Add` 仅支持 ZEROING（static_assert 拒绝 MERGING）；
- `DINTLV` 加载要求 32B 对齐；
- 标量写 UB 元数据 → VF 读的可见性无法用事件可靠保证（见上）。

### SIMT 回退路径

`incx != 1` 或 `n < 64` 时走 SIMT 内核：

- 与 cblas 逐位一致：对角先行，UPPER-N / LOWER-T·C 列递增，LOWER-N / UPPER-T·C 列递减；
- 复数乘积用 `fmaf(a,b,0)` 保证单次舍入，并用 `volatile` 阻止 FMA 融合；
- 作为通用步长与极小规模的正确性兜底。

## Host 与 Tiling

- 校验顺序见上；`n=0` 早退；
- workspace：`n·sizeof(aclblasComplex)`，经 `EnsureDefaultWorkspace` 复用 handle 工作区（稳定路径无 malloc/free），32B 对齐检查；
- `useNumBlocks = max(1, min(n, AIV核数))`，`numThreads` 按 SIMT 线程区间钳制；
- 分发：`useRegBase = (incx==1 && 64 <= n <= 8192)`，覆盖全部 uplo/trans；否则 SIMT；`useUb` 控制 SIMT 的 x 是否搬入 UB；
- K1 下发后不插入同步，K2（scatter）后统一同步一次。

## 支持硬件

| 项目 | 内容 |
| --- | --- |
| 产品 | Ascend 950PR（AIV 56 核） |
| 编译 | `--npu-arch=dav-3510`（bisheng） |
| CANN | 9.1.0 |
| 编程模型 | RegBase 主线 + SIMT 回退 |
| golden | cblas（Debian libblas 3.10.0，Netlib） |

## 算子约束限制

1. 数据类型仅 COMPLEX64（实虚部 FP32）；
2. AP 只含 uplo 指定三角，长度 `n(n+1)/2`，无 lda；
3. `diag=UNIT` 不访问 AP 对角；
4. x 为带步长一维向量，不支持超出 `incx` 语义的非连续张量；
5. AP 与 x 视为不重叠；不承诺重叠行为；
6. 地址计算用 uint64 中间量，避免 `n(n+1)/2` 溢出；
7. `n=0` 返回成功且不引用 AP/x；
8. 任务书 §3.4 不设独立内存指标；workspace 由 handle 复用。

# 性能优化方案

## 实测结果（2026-09-19，`dev-ctpmv-950`）

口径：`ctpmv_benchmark`，算子单次调用（含内部 workspace 与 scatter），warmup 20 + 100 次 `aclrtEvent` 计时取平均。

| case | 实测 us | 标杆 us | 判定 |
| --- | --- | --- | --- |
| 512/U/N/NON_UNIT/1 | 37.400 | 44.89 | 达标（0.83x） |
| 1024/L/N/NON_UNIT/1 | 74.994 | 88.46 | 达标（0.85x） |
| 2048/U/T/NON_UNIT/1 | 75.945 | 184.20 | 达标（0.41x） |
| 4096/L/C/NON_UNIT/1 | 202.521 | 1842.81 | 达标（0.11x） |

优化历程（同口径）：SIMT 全程 93.5/192.6/345.6/1486.9us → RegBase N 落地 42.95/197.91/347.37/
1565.84us → LOWER/N 落地 37.64/77.19/340.72/1583.63us → T/C 落地后为上表。

## 全量 200 条性能留档

不加过滤器运行 `ctpmv_benchmark` 覆盖全部 200 条 TC_PF（warmup 20 + 100 次），原始日志与逐条
明细随自测报告提供。结果：4 条验收 case 全达标；179/200 条不高于按 `gpu_baseline.csv` 推导的
参考标杆（`gpu_ms / 0.4`）。21 条参考 case 超标杆，分两类并有明确原因：

1. **小 n 固定开销（6 条，n <= 19）**：实测 ~17.5-19.0us，主要来自两次 kernel 下发 + workspace +
   最终同步的固定成本；H100 推导标杆仅 6.4-15.1us；
2. **大 n 的 trans=N 路径（14 条，n >= 1341，最大 1.33x）**：N 路径当前实现的性能短板；
   同规模 T/C 路径仅为标杆的 0.11~0.56x，说明差距在 N 路径本身而非设备能力。

以上均为**参考用例**，不属于任务书 §3.3 的 4 条硬性验收 case；测试指导 README 也注明
「标杆异常/测试行为无意义时可过滤并说明」。是否对 N 路径继续优化作为后续可选工作。

## 与配套 `verify_performance.py` 的口径差异

任务配套 `verify_performance.py` 解析的是 GTest 单条用例墙钟（包含输入生成、CPU cblas golden、
H2D/D2H 与比对），与 `gpu_baseline.csv` 的 GPU 算子耗时量纲不同，直接运行会对本算子给出
系统性 FAIL（脚本自身也注明该耗时是「保守上界」）。任务书 §3.3 要求 warmup 后 >50 次有效采样
的平均单次耗时，因此本设计以设备事件计时为准；该脚本结果仅作留档，不作为性能判定。

# 可维可测分析

## 精度验证

- `ctpmv_test --gtest_filter='*-*TC_PF*'`：**1001/1001 PASS**，走框架 `MIXED_TOLERANCE`
  （实/虚部分开，逐元素硬门禁），含 Inf/NaN 用例，无回退、无绕过判定；
- 测试数据由同一 seed 单流依次采样实部/虚部；golden 直接调用 `cblas_ctpmv`；
- 任务 CSV 与生成器未做任何改动（与任务包逐字节一致）。

## 测试设计

| 类别 | 内容 |
| --- | --- |
| 精度 CSV | 1200 条：L0 枚举（24）、SQ 尺寸（23）、INC 步长（72）、FL 填充（12）、CV 覆盖（96）、ED 边界（11）、EX 扩展（763） |
| 性能 CSV | TC_PF 200 条（含任务书 4 条典型 case + 小尺寸 + 对数扫描 + 枚举网格），benchmark 内断言 200 条齐全 |
| 专项探针 | `ctpmv_diag_probe_test`：AP 对角毒化为 NaN/Inf，UNIT 用例要求输出与 cblas 一致且全有限；NON_UNIT 对照要求毒化传播，排除假通过 |
| 负向 | 空 handle、非法枚举、负维度、incx=0、null AP/x、`n=0` quick return |
| 规格外补充 | 按任务书 §3.5.4，独立 GTest 文件补充，不混入任务基线 CSV |

## 复现命令

```bash
cd ops-blas
bash build.sh --soc=ascend950 --ops=ctpmv

# 精度（1001 条，排除 TC_PF）
./build/test/tpmv/ctpmv/ctpmv_test --gtest_filter='*-*TC_PF*'

# 单条验收性能（CTPMV_BENCHMARK_CASE 留空则跑全部 200 条）
CTPMV_BENCHMARK_CASE=TC_PF_1001 ./build/test/tpmv/ctpmv/ctpmv_benchmark --gtest_filter='*AcceptanceOfficialCases*'

# UNIT 对角毒化探针
./build/test/tpmv/ctpmv/ctpmv_diag_probe_test
```

## 兼容性

新增公共接口，不改动 `aclblasStpmv` 与 legacy 接口；参数校验顺序与状态码与仓内惯例/Netlib 对齐。
合入时包含公共头文件声明、`blas/tpmv/arch35/` 实现、CMake 注册与 `test/tpmv/ctpmv/arch35/` 测试与 CSV。

# 风险与待确认项

1. **任务书 LOWER packed 公式笔误**：本实现采用 Netlib/cblas 的 `i + (2n-j-1)j/2`；已实测 1001/1001
   通过（含全部 LOWER 用例），并以 cblas 为验收接口定义，特此留痕。
2. **参考性能脚本口径**：`verify_performance.py` 的 GTest 墙钟口径与 §3.3 不一致，需在验收时
   以本设计 §性能优化方案 的事件计时为准。
3. **21 条参考 case 偏差**：已分类说明（固定开销 / N 路径），非验收项；如需全量达标作为后续优化项。
4. **性能数据可复现性**：case 4 在不同运行上下文（单条 vs 全量）存在波动，验收以同口径完整重跑为准。

# 参考资料

- 《aclblasCtpmv 950 算子开发任务书》`aclblasCtpmv_Atlas950PR_task_doc.md`
- [Netlib ctpmv.f](https://www.netlib.org/blas/ctpmv.f)
- [cuBLAS cublasCtpmv](https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-tpmv)
- [生态算子开源精度标准](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md)
- [ops-blas 开源仓](https://gitcode.com/cann/ops-blas)
