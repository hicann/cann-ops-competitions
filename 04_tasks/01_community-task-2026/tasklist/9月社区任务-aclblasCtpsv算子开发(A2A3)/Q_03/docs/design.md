# 【社区任务】aclblasCtpsv 算子设计文档

- **任务编号**：9 月社区任务（A2/A3）
- **团队名称**：Q_03
- **适配硬件**：Atlas 800I/T A2（910B3，性能设备）/ Atlas A3 系列
- **CANN 版本**：9.1.0
- **开发语言**：Ascend C
- **目标仓库**：<https://gitcode.com/cann/ops-blas>
- **目标目录**：`blas/tpsv/arch22/`

## 设计文档 PR 要求

- 本 PR 只提交设计文档，不包含算子实现、二进制或尚未生成的设备测试结果。
- 文档提交位置为
  `04_tasks/01_community-task-2026/tasklist/9月社区任务-aclblasCtpsv算子开发(A2A3)/Q_03/docs/design.md`。
- 代码阶段再向 `cann/ops-blas` 提交 `blas/tpsv/arch22/` 与
  `test/tpsv/ctpsv/arch22/`；设计、实现和自测使用同一接口与验收口径。

## 需求背景（required）

## 需求来源

任务来源为 9 月社区任务 `aclblasCtpsv`（Atlas A2/A3）。目标仓库为
[`cann/ops-blas`](https://gitcode.com/cann/ops-blas)，设计文档提交到
`cann/cann-ops-competitions/04_tasks/01_community-task-2026/tasklist`，实现和测试
分别进入 `blas/tpsv/arch22/` 与 `test/tpsv/ctpsv/arch22/`。

## 背景介绍

TPSV 是 BLAS Level 2 的三角压缩（packed）存储线性方程组求解：给定 n×n 三角矩阵 A
的 packed 存储 `AP`，原地求解 `op(A) * x = b`（入口时 x 存放右端项 b，出口把解
写回 x）。现有 A2/A3 任务需要补齐 complex64 路径，并与 cuBLAS `cublasCtpsv`、
Netlib `ctpsv` 的接口、packed 存储、转置/共轭语义和原地语义保持一致。复数路径的
`trans=C` 不是普通转置：它必须对系数取共轭后再转置。

任务书随附 1200 条 CSV 用例（1000 条精度 + 200 条性能/内存），覆盖
`uplo × trans × diag` 共 12 个枚举组合、尺寸扫描、正负 stride、特殊值、
`n=0` quick return 与负向参数行。代码阶段必须由 ops-blas 自测工程完整回放该 CSV，
不能用少量代表性用例替代最终验收。

### 算子接口与数据类型

| 参数 | 含义 | 类型/布局 | 设计边界 |
| --- | --- | --- | --- |
| `handle` | 句柄与 stream | Host scalar | 为空返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| `uplo` | 上/下三角 packed | Host enum | `ACLBLAS_UPPER`、`ACLBLAS_LOWER` |
| `trans` | 操作类型 | Host enum | `ACLBLAS_OP_N`、`ACLBLAS_OP_T`、`ACLBLAS_OP_C` |
| `diag` | 对角类型 | Host enum | `ACLBLAS_NON_UNIT`、`ACLBLAS_UNIT` |
| `n` | 三角阶数/逻辑 x 长度 | Host `int` | `n >= 0`；`n=0` 为合法 no-op |
| `AP` | packed 三角矩阵 | device `const aclblasComplex*` | `n(n+1)/2` 个元素，列主序 packed，只读 |
| `x` | 原地输入/输出向量 | device `aclblasComplex*` | `n>0` 时 `1+(n-1)*abs(incx)` 个物理元素 |
| `incx` | 逻辑 stride | Host `int` | 非零，支持负值（反向取值） |

`aclblasComplex` 为两个 FP32 分量构成的 COMPLEX64 值。packed 存储没有前导维，
本算子不涉及 `lda`。

## 需求拆解

1. 在公共头文件 `include/cann_ops_blas.h` 新增与同族接口 `aclblasStpsv` 同型的
   `aclblasCtpsv` 声明，供其他产品线共用，不建立产品私有 API。
2. 实现 `N/T/C` × `UPPER/LOWER` × `UNIT/NON_UNIT` 共 12 组枚举组合，以及 packed
   下标换算、正负 stride 与原地写回语义。
3. 处理 `n=0` quick return、非法枚举、负维度、`incx=0` 以及 `n>0` 时的空指针；
   非法参数在 host 侧返回规定状态码且不发射 kernel。
4. 以 cblas（Netlib BLAS `ctpsv`）单标杆 golden 对输出向量做全长度、实部/虚部分量
   精度校验。
5. 性能按任务书 §3.3 的 5 个典型坐标验收，统计口径为 msprof 采集的平均单次 kernel
   耗时，先 warmup 再有效采样 10 次以上。
6. 自测须回放随任务提供的完整 1200 条用例，并覆盖边界、负向与特殊值场景。

## 需求分析（required）

## 语义公式

令逻辑向量索引 `q(i)` 为 x 中第 i 个逻辑元素的物理下标：

```text
incx > 0 : q(i) = i * incx
incx < 0 : q(i) = (n-1) * abs(incx) - i * abs(incx)
```

求解式为：

```text
x[q(i)] = ( b[q(i)] - sum_{j} op(A)(i,j) * x[q(j)] ) / d(i)
```

其中求和只累加依赖方向上已经解出的分量（前向代入 j<i，回代 j>i），`d(i)` 在
`diag=UNIT` 时为 `1+0j`，否则为 `op(A)(i,i)`。复数乘加按
`(ar*br-ai*bi, ar*bi+ai*br)` 展开。

**packed 存储索引（0 基，列主序）**：

| `uplo` | 元素 A(i,j) 位置 |
| --- | --- |
| UPPER（i ≤ j） | `AP[i + j*(j+1)/2]` |
| LOWER（i ≥ j） | `AP[i + (2*n-j-1)*j/2]` |

上表按 Netlib `ctpsv.f` / JIS packed 约定书写，并在实现阶段用 cblas 单标杆逐元素
验证。若任务书印刷版本的第二个下标项与之不一致，以 Netlib 与 `ops-blas` 仓内
`stpsv` 采用的同一约定为准，并在自测报告中给出对照说明（`n=3, j=1, i=2` 应落在
下标 4，而不是 5）。

**求解方向**：`op(A)` 为下三角时前向代入（i 递增），为上三角时回代（i 递减）。
`uplo=LOWER, trans=N` 与 `uplo=UPPER, trans≠N` 为前向；`uplo=UPPER, trans=N` 与
`uplo=LOWER, trans≠N` 为回代。`trans=C` 在读取 `AP` 后对系数虚部取反。

## 约束分类

- **锁定合同**：COMPLEX64、一维 packed `AP`（长度 `n(n+1)/2`）、合法枚举、`n>=0`、
  `incx != 0`、`n>0` 时 `AP/x` 非空、逻辑 x 全长度输出与物理 gap 保留。
- **可放宽合同**：任意非零 signed `incx`、运行时任意合法 `n`（不只是公开 case 的
  尺寸）、以及 `diag=UNIT` 时对角存储内容为任意值（含 Inf/NaN）的输入；这些都不能被
  当前公共 case 集合收窄。
- **自测探针**：null handle、null `AP/x`、非法枚举、负维度、`incx=0` 以及其他 CSV
  负向行。Inf/NaN 输入与 UNIT 对角不被读取属于正式数值语义，必须由完整 CSV 直接回归，
  不能被静默过滤。

## 详细设计（required）

## 算子分析

### packed 索引与原地数据所有权

packed 存储是列主序压缩：**同一列的连续元素在 `AP` 中物理连续**。这是本设计的关键。
无论 `uplo` 与 `trans` 如何组合，逐步求解所需的一整段系数都恰好是 `AP` 的某一列
（`trans=N` 用 `uplo` 指定列，`trans=T/C` 用转置后对应的同一列），因此可以用
`DataCopy` 连续搬入再向量化计算，避免按行 gather 造成的非连续访存。

`x` 是原地输入/输出：kernel 先把逻辑 `x[0..n-1]` 按 `q(i)` gather 到常驻 UB 缓冲，
求解全部完成后再按 `q(i)` scatter 写回。这样既保留了 `incx` 的 stride 与 gap 语义，
也消除了「读旧值/写新值」的别名风险。gather/scatter 阶段是正确性不变量，不得为追求
性能删除。

### Host 侧设计

1. 校验 `handle`、`uplo/trans/diag`、`n`、`incx` 以及 `n>0` 时的 `AP/x` 指针；
   错误直接返回公共状态码，不发射 kernel。
2. `n=0` 直接返回 `ACLBLAS_STATUS_SUCCESS`，不申请 workspace、不下发 kernel。
3. 计算逻辑 x 的物理起点、步长方向、每一步的 packed 列基址与有效长度，并把
   `uplo/trans/diag/n/incx` 与 UB tile 参数写入 `CtpsvTilingData`。
4. 依据 `n`、`abs(incx)` 与 UB 预算选择 `col_tile`（单次搬入的列块长度）、
   `x_ub_bytes` 与双缓冲开关：连续大 `n` 用满 UB，小 `n` 或大 stride 用较小 tile。
5. 生成 `tiling_key`：`uplo` 1 bit、`trans` 2 bits、`diag` 1 bit，并单独记录 stride
   类别与是否需要尾部 mask。key 只由合法运行时元数据生成，绝不读取 case id、文件名、
   seed 或答案数据。

### 分核与任务划分

三角求解存在严格的数据依赖：第 i 步必须使用依赖方向上已经解出的分量，单条右端向量的
依赖链不能跨核拆开。因此主路径使用**单 block**，并行度来自行内向量化以及 MTE 与
Vector 的双缓冲重叠，而不是把同一依赖链切分到多个 AI Core。

`x` 的逻辑长度最大为 4096，COMPLEX64 常驻 UB 仅约 32 KB，可全程驻留；`AP` 按列流式
搬入，不整体驻留。`n=0` 不占用 core，也不分配 workspace。该划分只使用运行时元数据，
不依赖公开 case 的尺寸分布。

### Kernel 侧设计

kernel 分为三个阶段：

1. **Gather**：按 `q(i)` 从 GM 读取逻辑 x 到常驻 UB 缓冲，实部/虚部按平面拆分以便
   向量化。尾部用 mask 覆盖非对齐长度；不参与逻辑索引的物理 gap 既不读取也不写回。
2. **Solve**：按依赖方向逐步推进。第 s 步：
   - 从 `AP` 连续搬入当前步的 packed 列块；`trans=T/C` 时对系数虚部按需取反；
   - `diag=NON_UNIT` 时读取对角元素并按 Smith 形式做复数除法（按较大分量缩放，
     避免直接平方和溢出）；`diag=UNIT` 时以 `1+0j` 代替，且不形成对角存储地址；
   - 用 broadcast 标量 `x[s]` 在剩余分量上做复数乘加，每个复数乘加展开为四个 FP32
     乘加（`real -= ar*xr - ai*xi`、`imag -= ar*xi + ai*xr`），由 Vector 指令完成。
   列块搬入与计算使用双缓冲，重叠 MTE 与 Vector。
3. **Scatter**：按 `q(i)` 把解写回 GM，逻辑长度以外与 gap 保持不变。

每一步都对列块有效长度做边界判断，并对 `n` 的尾部使用 mask；禁止按对齐后的长度访问
`AP` 的越界区域。

### 转置/共轭与求解方向

通过统一的逻辑索引函数覆盖四类路径，避免复制整套 kernel。`trans=N` 用
right-looking（先求出 `x[s]`，再把它的影响散播到剩余分量）；`trans=T/C` 用
left-looking（把已解分量对第 s 行的贡献点积出来）。两种形式都只需要 `AP` 的**同一列**
连续区间：

| `uplo` | `trans` | 求解方向 | 数据流 | 第 s 步系数（`AP` 第 s 列） |
| --- | --- | --- | --- | --- |
| LOWER | N | 前向（s 递增） | right-looking | 元素 `s+1 .. n-1` |
| UPPER | N | 回代（s 递减） | right-looking | 元素 `0 .. s-1` |
| UPPER | T/C | 前向（s 递增） | left-looking | 元素 `0 .. s-1` |
| LOWER | T/C | 回代（s 递减） | left-looking | 元素 `s+1 .. n-1` |

UPPER 的第 s 列基址为 `s*(s+1)/2`（元素 `0..s`），LOWER 的第 s 列基址为
`s*(2*n-s-1)/2`（元素 `s..n-1`）。`T` 与 `C` 共享索引路径但不共享数值路径；`C` 的
虚部符号处理必须在乘加之前完成。

### Tiling key 与 dispatch

| 运行时条件 | 路径 | 关键约束 |
| --- | --- | --- |
| `n = 0` | quick return | 不发射 kernel、不分配 workspace |
| `n > 0, diag = NON_UNIT` | 含对角除法的列推进 | 对角必须读取；非奇异性由调用方保证 |
| `n > 0, diag = UNIT` | 单位对角特化 | 不形成对角存储地址 |
| 任意 `incx != 0` | strided gather/scatter | 逻辑索引由 tiling data 计算，物理 gap 保留 |
| 列尾/非对齐长度 | masked tail | 有效长度 runtime 可见，禁止越界访存 |

Host 用枚举与运行时尺寸设置 key，kernel 用编译期组合减少分支。所有路径必须覆盖完整的
guarded contract，unsupported 输入必须显式报错，不能转到 ACLNN、Torch、CPU 或官方
whole-op 实现兜底。

## 支持硬件

| 支持的芯片版本 | arch | 状态 |
| --- | --- | --- |
| Atlas 800I/T A2（910B3） | arch22 | 任务书目标（性能测试设备） |
| Atlas A3 系列 | arch22 同族路径 | 任务书要求兼容 |

本设计不包含 950 `arch35` 路径，也不把其他架构目录的实现当作 A2/A3 结果。

## 算子约束限制

- 不涉及 broadcast、alpha/beta、前导维或非连续 view；packed `AP` 与 `incx` 语义必须保留。
- `n < 0`、`incx = 0`、非法枚举、`n>0` 时 `AP/x` 为空均为非法参数；
  `handle` 为空返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`。
- `diag=NON_UNIT` 的被引用对角必须非零；求解类对奇异矩阵无定义，算子不做奇异性或近奇异
  检测，也不会替换为其他求解器。
- `diag=UNIT` 时不得读取 `AP` 的对应对角位置（该存储可为任意值，甚至 NaN/Inf）。
- 调用方须保持 `AP/x` 与 handle stream 的生命周期；`AP` 与 `x` 不允许内存重叠；
  读回设备结果前须同步该流。

## 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度 | COMPLEX64 实部/虚部分别判定：`|actual-golden| <= atol + rtol*|golden|`，`rtol = atol = 2^-13`，matched ratio `>= 0.99`，max abs error `<= 1e-2` 或 `32*ULP`（含规约场景可酌情放宽至 2ULP） | 任务书 §3.2、生态算子精度标准 |
| 性能 | 任务书 §3.3 五个典型坐标的平均单次 kernel 耗时（us）：252.2 / 573.9 / 1228 / 1211 / 7327；先 warmup 后有效采样 >10 次 | 任务书 §3.3、`gpu_baseline.csv` |

## 自测计划

1. 先运行归档包中的 `test_cases/gen_csv.py` 并校验用例集合，确认 1200 行、覆盖类别、
   任务书阈值和性能坐标没有漂移（固定种子逐字节一致）。
2. 将完整 `tpsv_test.csv` 安装到 `test/tpsv/ctpsv/arch22/`，用 `verify_accuracy.py`
   编译并运行所有非 `TC_PF` 行，保存逐条 PASS/FAIL 与精度摘要。
3. 性能用 `msprof op` 对 `TC_PF` 用例采集 kernel 时间（`OpBasicInfo.csv` 的
   Task Duration 求平均），与 `gpu_baseline.csv` 的 `gpu_ms` 逐条对齐，要求 NPU
   平均单次 kernel 耗时 `<= gpu_ms / 0.8`；host 侧准备、gather/scatter、launch 与
   结果同步开销必须与 kernel 时间分开记录。
4. 对 `n=0`、负 stride、UNIT 对角哨兵、`trans=C` 共轭、masked tail 与非法参数单独
   保留日志，防止它们被普通随机 case 覆盖。
5. A2 与 A3 分别进行 fresh build 与设备运行（A2 为性能设备），报告中写明 SoC、CANN、
   物理卡号、采样次数、kernel/host 时间边界与内存占用；没有实测证据的项写「未生成」。
6. 设计阶段不声称任何设备测试通过；实现阶段必须保存原始 stdout、失败用例、编译版本、
   实际 NPU/SoC 和输入生成规则，才能形成验收证据。

## 兼容性与无 fallback 边界

接口声明放在公共 `include/cann_ops_blas.h`，实现只放在 `blas/tpsv/arch22/`。评测路径
不得调用官方 whole-op、Torch/Torch-NPU 等价算子、CPU/cblas 参考或其他后端实现兜底；
workspace、tiling data 与依赖的 kernel source 必须随实现或其可复现构建闭环提供。
任务书、cblas golden 与 GPU 基线只用于对照验证，不是运行时 fallback。
