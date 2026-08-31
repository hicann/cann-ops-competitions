# 【社区任务】aclblasCtbmv 算子设计文档

- **任务编号**：08-10-矩阵乘系列算子开发
- **团队名称**：Q_03
- **适配硬件**：Atlas 800I A2 / Atlas A3 系列
- **CANN 版本**：9.1.0
- **开发语言**：Ascend C
- **目标仓库**：<https://gitcode.com/cann/ops-blas>
- **目标目录**：`blas/tbmv/arch22/`

## 设计文档 PR 要求

- 本 PR 只提交设计文档，不包含算子实现、二进制或尚未生成的设备测试结果。
- 文档提交位置为
  `04_tasks/01_community-task-2026/tasklist/08-10-矩阵乘系列算子开发/Q_03/docs/design.md`。
- 代码阶段再向 `cann/ops-blas` 提交 `blas/tbmv/arch22/` 与
  `test/tbmv/ctbmv/arch22/`；设计、实现和自测使用同一接口与验收口径。

## 需求背景（required）

## 需求来源

任务来源为 Atlas A2/A3 社区任务 `aclblasCtbmv`。目标仓库为
[`cann/ops-blas`](https://gitcode.com/cann/ops-blas)，设计文档提交到
`cann/cann-ops-competitions/04_tasks/01_community-task-2026/tasklist`，实现和测试
分别进入 `blas/tbmv/arch22/` 与 `test/tbmv/ctbmv/arch22/`。

## 背景介绍

TBMV 是 BLAS Level 2 的三角带状矩阵-向量乘。现有 A2/A3 任务需要补齐 complex64
路径，并与 cuBLAS `cublasCtbmv`、Netlib `ctbmv` 的接口、带状存储、转置和原地
语义保持一致。复数路径的 `trans=C` 不是普通转置：它必须对系数取共轭后再转置。

任务书随附 1200 条 CSV 用例（包含正常语义、API 负向和性能压力行），覆盖 12 个枚举
组合、尺寸/带宽扫描、leading-dimension padding、正负 stride、特殊值和 `n=0`
quick return。代码阶段必须由 ops-blas 自测工程完整回放 CSV，不能用少量代表性用例
替代最终验收。

### 算子接口与数据类型

| 参数 | 含义 | 类型/布局 | 设计边界 |
| --- | --- | --- | --- |
| `handle` | 句柄与 stream | Host scalar | 由 ops-blas host API 校验 |
| `uplo` | 上/下三角带 | Host enum | `UPPER`、`LOWER` |
| `trans` | 操作类型 | Host enum | `N`、`T`、`C` |
| `diag` | 对角类型 | Host enum | `NON_UNIT`、`UNIT` |
| `n` | 方阵阶数/逻辑 x 长度 | Host `int` | `n>=0` |
| `k` | 半带宽 | Host `int` | `0<=k<n`；`n=0` 时 `k=0` |
| `A` | 带状矩阵 | device `aclblasComplex*` | `lda*n` 个元素，列主序 |
| `lda` | 带状 leading dimension | Host `int` | `lda>=k+1` |
| `x` | 原地输入/输出向量 | device `aclblasComplex*` | `0` 或 `1+(n-1)*abs(incx)` 个元素 |
| `incx` | 逻辑 stride | Host `int` | 非零，支持负值 |

## 需求拆解

1. 在公共头文件新增与 `aclblasStbmv` 同型的 `aclblasCtbmv` 声明，
   不建立产品私有 API。
2. 实现 `N/T/C`、`UPPER/LOWER`、`UNIT/NON_UNIT` 的完整组合以及带内索引规则。
3. 在原地输出前保存原始逻辑 `x`，确保每个输出使用同一份输入快照并保留 stride gap。
4. 处理 `k=0`、`k=n-1`、`n=0`、padding、`incx<0` 和奇数尾部等边界。
5. 以 cblas/Netlib golden 做实部/虚部分量精度校验，按任务书目标完成 A2/A3 自测。
6. 性能测试先 warmup，再进行超过 50 次有效采样；任何性能结论必须绑定真实设备和日志。

## 需求分析（required）

## 语义公式

令逻辑向量索引 `q(i) = (i * incx)`（`incx>0`）或
`(n-1)*abs(incx) - i*abs(incx)`（`incx<0`）。结果为：

```text
x_out[q(i)] = sum_j op(A)(i,j) * x_in[q(j)]
```

其中只累加带内、且属于 `uplo` 指定三角区域的元素。UPPER 的物理行是
`k+i-j`，LOWER 的物理行是 `i-j`；物理列为 `j`。`diag=UNIT` 时 `i=j`
直接使用 `1+0j`，不能读取 `A` 的对应位置。`trans=C` 在取值后把系数虚部取反。

## 约束分类

- **锁定合同**：complex64、连续一维物理 buffer、合法 enum、`n/k/lda/incx`
  关系、完整物理 x 输出和 gap 保留。
- **可放宽合同**：`lda` 的额外 padding、任意非零 signed `incx`，以及运行时的
  合法 `n`/`k` 大小；它们不能被当前公共 case 集合收窄。
- **自测探针**：null pointer、非法 enum、负维度、`k>=n`、`incx=0`、分配失败和
  其余 CSV 行。Inf/NaN/极值属于正式数值语义，必须由完整 CSV 自测直接回归。

## 详细设计（required）

## 算子分析

### 原地别名与数据所有权

TBMV 的输入和输出 `x` 是同一物理 buffer，但矩阵-向量乘的每个输出都必须读取
同一份原始输入，而不能读取已经写回的其他输出。因此 host 侧为 kernel 申请
`x_snapshot` workspace，大小为 `x_storage_len * sizeof(aclblasComplex)`。kernel
先把所有物理 x 元素复制到 snapshot，再计算和写回；未参与逻辑索引的 gap 也复制并
原样写回。这个快照阶段是正确性不变量，不得为追求性能删除。

### Host 侧设计

1. 校验 handle、enum、`n/k/lda/incx` 和 `n>0` 时的指针；错误直接返回公共状态码。
2. `n=0` 设置 `zero_work`，不申请 workspace、不设置 kernel launch。
3. 计算 `x_storage_len`、有效带宽 `k+1`、每个输出行的物理 A 起点以及 signed x 起点，
   将 `uplo/trans/diag/n/k/lda/incx` 写入 `CtbmvTilingData`。
4. 依据 `n`、`k`、`abs(incx)`、`lda` 和 UB 预算选择 `row_tile`、`core_count`、
   `snapshot_tile` 与双缓冲开关。小规模或大 stride 用较小 tile；连续大规模用满核分工。
5. 计算 `tiling_key`：`uplo` 1 bit、`trans` 2 bits、`diag` 1 bit，并单独记录 stride
   类别和是否需要尾部 mask。key 只由合法运行时元数据生成，绝不读取 case id、
   文件名或 seed。

### 分核与任务均分

快照完成后各输出行只依赖 `A` 和 `x_snapshot`，因此可按逻辑输出行区间均分到 AI Core。
`core_count = min(device_core_count, ceil_div(n, row_tile))`；余数分配给前面的 core，
保证每个 core 的行数差不超过一行。`n=0` 不占用 core。对 `incx!=1`，逻辑索引仍由
tiling data 计算，不能把物理长度误当作逻辑长度。

### Kernel 侧设计

kernel 分为三个阶段：

1. **Snapshot**：每个 core 协作从 GM 读取完整 x 物理 buffer，按 tile 搬入 UB，再写入
   `x_snapshot` GM。使用边界 mask 覆盖奇数尾部，确保 gap 不丢失。
2. **Compute**：每个 core 遍历自己的输出行。对每行计算带内列区间，使用
   `DataCopy`/向量 gather 取得 A 系数与 snapshot x，按复数乘加展开为四个 FP32
   乘加：`real += ar*xr - ai*xi`，`imag += ar*xi + ai*xr`。`trans=C` 对 `ai`
   取反；`diag=UNIT` 的对角项由寄存器常量替代。
3. **WriteBack**：将结果写回 `x[q(i)]`，对非逻辑 gap 使用 snapshot 的原值。
   输出 tile 的尾部通过 mask 写回，避免越界。

### 转置与三角方向

通过统一的逻辑行列函数处理四类路径，避免复制整套 kernel：

| `uplo` | `trans` | 有效列区间 | 读取系数 |
| --- | --- | --- | --- |
| UPPER | N | `i..min(n-1,i+k)` | `A[k+i-j,j]` |
| LOWER | N | `max(0,i-k)..i` | `A[i-j,j]` |
| UPPER | T/C | 对应转置后的 LOWER 区间 | 交换逻辑 i/j，C 额外共轭 |
| LOWER | T/C | 对应转置后的 UPPER 区间 | 交换逻辑 i/j，C 额外共轭 |

`T` 与 `C` 共享索引路径但不共享数值路径；C 的虚部符号处理必须在乘加前完成。

### Tiling key 与 dispatch

Host 用枚举和运行时尺寸设置 key，kernel 用编译期组合减少分支。尺寸、padding 和
stride 只影响 tiling data；必要时可在同一 Ascend C solution 内选择连续、strided、
小带宽和宽带实现。所有路径必须覆盖完整的 guarded contract，unsupported 输入必须
显式报错，不能转到 ACLNN、Torch、CPU 或官方 whole-op。

## 支持硬件

| 支持的芯片版本 | arch | 状态 |
| --- | --- | --- |
| Atlas 800I/T A2（910B3） | arch22 | 任务书目标 |
| Atlas A3 系列 | arch22 同族路径 | 任务书要求兼容 |

## 算子约束限制

不涉及 broadcast、alpha/beta 或非连续 view；A 和 x 的物理 buffer、`lda` 和 `incx`
语义必须保留。外部自测的非法参数和特殊值不应被静默过滤；它们必须返回规定状态或
被独立记录为明确的 probe 结果。

## 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度 | 实部/虚部按 FLOAT32：`rtol=2^-10`、`atol=2^-16`、matched ratio `>=0.99`，max abs `<=1e-2` 或 `32*ULP` | 任务书 §3.2、生态算子精度标准 |
| 性能 | Atlas 800I A2 典型坐标 6.40/10.57/12.65 us；先 warmup 后有效采样 >50 次 | 任务书 §3.3 |

## 自测计划

1. 先运行归档包中的 `test_cases/validate_cases.py --require-baseline`，确认完整
   1200 行、固定覆盖、任务书阈值和性能坐标没有漂移。
2. 将完整 `ctbmv_test.csv` 安装到 `test/tbmv/ctbmv/arch22/`，用 `verify_accuracy.py`
   编译并运行所有非 `TC_PF` 行，保存每条 PASS/FAIL 和精度摘要。
3. 用 `verify_performance.py` 运行全部 `TC_PF` 行；基线回填前只记录 NPU 时间，回填后
   依据 `(n,k,uplo,trans,diag)` 对齐任务包 GPU 基线并计算倍率。
4. 对 `n=0`、负 stride、padding、full-band、C 共轭和 UNIT 对角 NaN/Inf 单独保留日志，
   防止它们被普通随机 case 覆盖。
5. A2 与 A3 分别进行 fresh build 和设备运行，报告中写明 SoC、CANN、物理卡号、采样次数、
   kernel/host 时间边界和内存占用；没有实测证据的项写“未生成”。
6. 最终本地结论以 `validate_cases.py` 结果门为准：精度 `1000/1000`、200 条 ZIP
   性能倍率均 `>=0.8x`、三条任务书平均延迟均达标、有效采样 `>50`；不使用额外
   `>1.1x` 目标替代任务书。

## 兼容性与无 fallback 边界

接口声明放在公共 `include/cann_ops_blas.h`，实现只放在 `blas/tbmv/arch22/`。评测路径
不得调用官方 whole-op、Torch/Torch-NPU 等价算子、CPU/reference 或其他实现兜底；
workspace、snapshot、tiling data 和依赖的 kernel source 必须随实现或其可复现构建
闭环提供。任务书目标、Netlib golden 和任务包 GPU 基线只用于对照验证，不是运行时
fallback。
