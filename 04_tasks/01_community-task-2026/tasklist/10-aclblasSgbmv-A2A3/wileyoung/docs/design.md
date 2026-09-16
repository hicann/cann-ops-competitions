# aclblasSgbmv 算子设计文档

| 项目 | 内容 |
| --- | --- |
| 任务名称 | 算子实操工坊-aclblasSgbmv 算子开发（A2/A3） |
| taskId | `57fcc5dad709456caae4b2b6d6003371` |
| 任务链接 | [昇腾任务中心](https://www.hiascend.com/activities/task-center/details/57fcc5dad709456caae4b2b6d6003371?menu=trends) |
| 设计文档提交目录 | `04_tasks/01_community-task-2026/tasklist/10-aclblasSgbmv-A2A3/wileyoung/docs/design.md` |
| 适配硬件 | Atlas A2/A3，`arch22`；首要验证平台为 `ascend910b3`，A3 使用 `ascend910_93*` |
| CANN 版本 | 9.1.0 |
| 相关代码仓 | `cann/ops-blas` |
| 代码目标目录 | `blas/gbmv/arch22/`、`test/gbmv/sgbmv/arch22/` |
| 文档状态 | 设计评审阶段；本文不宣称代码、精度或性能已经完成验收 |

# 需求背景（required）

## 需求来源

本算子来自昇腾 CANN 算子实操工坊及 2026 年开源社区任务（taskId：`57fcc5dad709456caae4b2b6d6003371`），目标是在 `ops-blas` 中补充 Atlas A2/A3（arch22）上的单精度实数带状矩阵向量乘法接口，并完成设计评审、算子开发、自验和社区 PR 合入。

## 背景介绍

### aclblasSgbmv 算子实现

`SGBMV` 计算：

```text
y = alpha * op(A) * x + beta * y
```

其中 `A` 是 `m x n` 的列主序带状矩阵，`kl` 和 `ku` 分别表示主对角线下方和上方的对角线条数。算子接口必须与 `cublasSgbmv` 的参数顺序和语义一致：

```cpp
aclblasStatus_t aclblasSgbmv(
    aclblasHandle_t handle, aclblasOperation_t trans,
    int m, int n, int kl, int ku,
    const float* alpha, const float* A, int lda,
    const float* x, int incx,
    const float* beta, float* y, int incy);
```

### 现状分析

`ops-blas` 已有公共声明 `include/cann_ops_blas.h` 和 `blas/gbmv/README.md`，但 arch22 目录缺少本算子的实现和对应测试。仓库已有 arch35 的 `sgbmv` 可作为工程、接口和测试框架参考，但不能直接作为 A2/A3 实现。

### 算子功能分析

| 参数 | 含义 | 类型/位置 | 约束 |
| --- | --- | --- | --- |
| `handle` | BLAS 句柄，携带 stream | Host | 不能为空 |
| `trans` | `N`、`T` 或 `C` | Host 枚举 | `C` 在实数路径下等价于 `T` |
| `m`, `n` | A 的行数、列数 | Host 整数 | 非负；任一为 0 时 no-op |
| `kl`, `ku` | 下、上对角线数量 | Host 整数 | 非负；当对应维度大于 0 时分别满足 `kl < m`、`ku < n` |
| `alpha`, `beta` | 标量系数 | Host `const float*` | 不能为空 |
| `A` | 带状矩阵 | Device `const float*` | 仅访问带内元素 |
| `lda` | A 的物理列跨度 | Host 整数 | `lda >= kl + ku + 1` |
| `x`, `y` | 输入向量、原地输出向量 | Device `const float*` / `float*` | 逻辑长度随 `trans` 改变 |
| `incx`, `incy` | 向量步长 | Host 整数 | 非零，支持正负值 |

## 口径裁决

1. `lda` 使用 Netlib/cuBLAS 的 BLAS-2 口径 `kl + ku + 1`。任务书参数表中出现的 `2 * kl + ku + 1` 属于 LAPACK 带状分解的存储要求，不适用于 `sgbmv`；更大的 `lda` 作为合法 padding 输入兼容。
2. `trans = ACLBLAS_OP_C` 与 `ACLBLAS_OP_T` 共用转置实现。
3. 当 `m > 0` 或 `n > 0` 时，`kl < m`、`ku < n` 是带状矩阵的有效带宽约束；零维 no-op 仍允许 `kl/ku` 为非负值。
4. 非法 `trans` 返回 `ACLBLAS_STATUS_INVALID_ENUM`；其他非法参数返回 `ACLBLAS_STATUS_INVALID_VALUE`；空句柄返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`。

## 设计阶段范围

本 PR 只提交设计文档，不提交 `ops-blas` 算子代码。代码实现和测试代码在设计评审通过后，另行提交到 `cann/ops-blas`。本文中的性能数值是任务书验收目标，不是本地或 NPU 实测结果。

# 需求分析（required）

## 需求描述

使用 Ascend C kernel 直调方式，在 `blas/gbmv/arch22/` 实现 `aclblasSgbmv`，由句柄绑定的 stream 异步启动 NPU kernel。实现只支持 `float32`，不支持广播或额外的私有接口。

## 需求拆解

1. 实现 `trans=N/T/C` 三种矩阵操作，保证带状存储映射正确。
2. 支持矩形矩阵、`kl/ku` 非对称带宽、`lda` padding、正负 `incx/incy`。
3. 完成参数校验、零维 no-op、`alpha=0` 快速路径和标准状态码返回。
4. 以 Netlib `cblas_sgbmv` 为 golden，满足 FLOAT32 精度标准。
5. 在 Atlas A2 上完成四个基准 case 的预热和超过 50 次有效采样。
6. 提供 arch22 算子代码、测试代码、CSV 用例、README、自测报告和两个社区 PR 所需材料。

# 详细设计（required）

## 算子分析

### 数学公式

当 `trans=N` 时：

```text
y[i] = beta * y[i] + alpha * sum(A[i,j] * x[j])
```

其中 `j` 的范围是 `[max(0, i-kl), min(n-1, i+ku)]`。

当 `trans=T/C` 时：

```text
y[j] = beta * y[j] + alpha * sum(A[i,j] * x[i])
```

其中 `i` 的范围是 `[max(0, j-ku), min(m-1, j+kl)]`。

### 带状存储

A 的物理空间为 `lda x n`，按列主序存储。合法逻辑元素 `A[i,j]` 的物理位置为：

```text
band_row = ku + i - j
offset   = band_row + j * lda
```

仅当 `i-j <= kl` 且 `j-i <= ku` 时访问该位置；列边界产生的带外空洞不得读取。

### 向量步长

逻辑索引 `k` 映射到物理向量位置：

```text
inc > 0: offset(k) = k * inc
inc < 0: offset(k) = (length - 1 - k) * (-inc)
```

`length` 为 `trans=N` 时的逻辑输入/输出长度 `n/m`，为 `trans=T/C` 时的 `m/n`。

### 支持数据类型和形状

- A、x、y、alpha、beta 均为 `float32`。
- `m`、`n`、`kl`、`ku` 为运行时参数，不要求 dynamic-shape 编译支持。
- 支持 `m`、`n` 不相等，支持 `lda > kl + ku + 1` 的 padding。
- 不涉及 broadcast；y 为原地输出，不返回新视图。

## 算子实现

### 实现方案

#### 3.2.1 host 侧设计

Host 侧完成校验、快速返回、tiling 和异步派发：

1. 检查 `handle`、`trans`、维度、`kl/ku`、`lda`、步长和 `alpha/beta` 指针。
2. 当 `m == 0 || n == 0` 时返回成功，不启动 kernel；正维度时 `y` 必须非空，且 `alpha != 0` 时 `A`、`x` 必须非空。
3. `alpha == 0` 时不读取 A、x：
   - `beta == 1`：直接成功返回；
   - `beta == 0`：启动向量清零路径；
   - 其他 beta：启动 `y = beta * y` 缩放路径。
4. 其余场景根据输出维度选择 block 数，block 数不超过可用 AI Core 数量，并将以下字段写入 `SgbmvTilingData`：`m/n/kl/ku/lda/trans/alpha/beta/incx/incy` 及分块信息。
5. 通过 `handle` 中的 stream 调用 kernel launcher，保持异步语义；结果读取由调用方同步 stream。

参数校验顺序固定为：

1. `handle`：空句柄立即返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`。
2. `trans`：只接受 `N/T/C`，否则返回 `ACLBLAS_STATUS_INVALID_ENUM`。
3. `m/n/kl/ku`：检查非负值及正维度下的 `kl < m`、`ku < n`。
4. `incx/incy`：必须非零，正负值均合法。
5. `alpha/beta`：指针必须非空；只有在此之后才能读取标量。
6. `lda`：必须满足 `lda >= kl + ku + 1`。
7. 数据指针：正维度下 `y` 必须非空；仅当 `alpha != 0` 时要求 `A` 和 `x` 非空。

这样可以保证错误路径不会解引用空指针，也能保持 `m=0/n=0` 和 `alpha=0` 的 quick-return 语义：

- `m == 0 || n == 0`：完成上述 Host 参数检查后直接成功返回，不访问 A、x、y。
- `alpha == 0 && beta == 1`：直接成功返回，不访问 A、x，也不写 y。
- `alpha == 0 && beta != 1`：只执行 `y = beta * y`；`beta == 0` 时不读取旧 y。
- `alpha != 0`：执行完整带状矩阵向量乘加。

建议的最小 tiling 分支：

| 分支 | 条件 | 分工 |
| --- | --- | --- |
| 连续 N | `trans=N, incx=1, incy=1` | 按输出行切分 |
| 连续 T/C | `trans=T/C, incx=1, incy=1` | 按输出列切分 |
| 通用步长 | 任一向量步长非 1 或为负 | 使用步长寻址 |

Host 与 Kernel 共享的 `SgbmvTilingData` 只保存运行时参数和切分结果，不改变公共 API：

```cpp
struct SgbmvTilingData {
    uint32_t m;
    uint32_t n;
    uint32_t kl;
    uint32_t ku;
    uint32_t lda;
    int64_t incx;
    int64_t incy;
    uint32_t trans;      // Host 归一化：0 = N，1 = T/C
    float alpha;
    float beta;
    uint32_t useCoreNum;
    uint32_t tileRows;
    uint32_t tileCols;
    uint32_t numThreads;
    uint32_t ubElements;
};
```

`useCoreNum` 由 Host 根据输出维度和 `GetAivCoreCount()` 计算；小 shape 减少 block 数，连续大 shape 才使用多核切分，避免启动开销超过计算量。切分字段按以下规则确定，并在代码 PR 中固定为可复现的实现：

- Host 先将公共枚举归一化为 `transCode`：`N -> 0`，`T/C -> 1`；kernel 不直接比较公共枚举值。先读取 `coreCount = GetAivCoreCount()`，若为 0 则返回执行失败状态，禁止继续计算；否则 `outLen = (transCode == 0) ? m : n`，`useCoreNum = min(coreCount, max(1, outLen / minTileOut))`；`tileOut = ceil(outLen / useCoreNum)`。
- `transCode == 0` 时设置 `tileRows = tileOut, tileCols = 1`；`transCode == 1` 时设置 `tileRows = 1, tileCols = tileOut`。
- `ubElements = floor((ubBytes - stagingBytes) / sizeof(float))`，由编译期 UB 容量和双缓冲暂存需求计算；`numThreads` 使用 kernel 实际支持的固定线程档位，不在公共 API 中暴露。
- 尾块通过有效长度和 mask 处理；上述字段只描述本次 launch 的切分，不改变 `aclblasSgbmv` 公共签名。

#### 3.2.2 kernel 侧设计

kernel 使用 Ascend C 的 `Init -> CopyIn -> Compute -> CopyOut` 流程，并按输出维度切分，保证每个输出元素只有一个 block 写入。

**N 模式：**每个 block 负责一段输出行。对每个相关列计算 `band_row`，连续搬运该列与当前行段相交的带内元素，在 UB 中执行乘加累加，最后按 `beta` 合并并写回 y。该方案不需要原子加。

**T/C 模式：**每个 block 负责一段输出列。每列搬运合法的带内片段和对应的 x 片段，使用向量乘法及规约得到点积，再计算 `alpha * dot + beta * y[j]`。不同列之间无写冲突。

**UB 和尾块：**连续路径按 32 字节对齐的 float32 块搬运，采用双缓冲隐藏搬运延迟；首尾不对齐区域使用有效长度或 mask，确保带外空洞和尾部脏数据不参与计算。通用步长路径按逻辑索引计算地址，不假设向量连续。

## 支持硬件

| 支持芯片 | 架构标识 | 状态 |
| --- | --- | --- |
| Atlas A2（任务书性能设备名称分别写作 800T/800I，SOC 均以 910B3 为准） | `ascend910b3` / `arch22` | 首要验证 |
| Atlas A3 系列 | `ascend910_93*` / `arch22` | 设计支持，代码 PR 验证 |

`ascend910_93*` 表示按 `ops-blas` 顶层 CMake 的 `ascend910_93` 前缀匹配；提交代码 PR 时以实际 CANN 工具链接受的完整 SOC 名称为准。任务书对 A2 产品别名存在 800T/800I 两种写法，本文不据此推导不同的编译目标，统一使用可复现的 `ascend910b3`。

## 算子约束限制

| 约束 | 处理 |
| --- | --- |
| 仅 float32 | 编译期固定接口类型 |
| `lda < kl+ku+1` | 返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| `incx == 0` 或 `incy == 0` | 返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| `trans` 不在 N/T/C | 返回 `ACLBLAS_STATUS_INVALID_ENUM` |
| 空句柄 | 返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| `m/n` 为负或 `kl/ku` 为负 | 返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| 正维度下 y 为空 | 返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| `alpha != 0` 且 A 或 x 为空 | 返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| 广播、非 BLAS 步长视图 | 不支持 |

# 可维可测分析

## 精度标准/性能标准

| 验收项 | 标准 |
| --- | --- |
| Golden | Netlib/cblas `sgbmv`，全向量比对 |
| FLOAT32 精度 | `rtol=2^-10`、`atol=2^-16`、匹配率至少 0.99，最大绝对误差不超过 `1e-2` 或 `32*ULP` |
| alpha=0 | `y=beta*y` 路径位精确 |
| 性能采样 | 任务书要求先 warmup、随后有效采样超过 50 次并报告平均耗时；当前辅助脚本单次运行不能单独证明该要求，代码 PR 必须补齐重复采样或等价设备侧采样 |
| Case 1 | `1024x1024, kl=16, ku=16, N, incx=incy=1`，不高于 `12.53 us` |
| Case 2 | `2048x2048, kl=32, ku=32, N, incx=incy=1`，不高于 `15.54 us` |
| Case 3 | `1024x1024, kl=16, ku=16, T, incx=incy=1`，不高于 `10.58 us` |
| Case 4 | `4096x4096, kl=64, ku=64, N, incx=incy=1`，不高于 `21.98 us` |

## 兼容性分析

- 公共接口只使用 `include/cann_ops_blas.h` 已声明的 `aclblasSgbmv`，不增加产品私有平行 API。
- arch22 实现放在 `blas/gbmv/arch22/`；arch22 测试放在 `test/gbmv/sgbmv/arch22/`，沿用仓库 CMake 自动收集规则。
- 参数顺序、列主序带状布局、转置语义和正负步长与 cuBLAS/Netlib 保持一致，上层调用无需适配。

## 测试设计与自测流程

测试资产位于 `算子实操工坊-北京站-aclblasSgbmv算子开发(A2A3)/test_cases/`：

- `sgbmv_test.csv`：1000 条精度用例和 200 条性能用例，覆盖基础 shape、矩形 shape、三种 trans、padding、正负步长、alpha/beta 特例、Inf/NaN、零维和非法参数。
- `gpu_baseline.csv`：200 条性能/内存基线数据；代码 PR 中必须核验 `gpu_ms` 的来源、设备和测量口径后才能进行 GPU 比例判定，不能把未核验或占位值当作实测结果。
- `gen_csv.py`：固定随机种子生成或扩展 CSV。
- `verify_accuracy.py`：安装 CSV、调用 `ops-blas/build.sh` 编译并运行 GTest，golden 使用仓库 cblas 实现；代码 PR 需先补充 `ascend910_93` 到其 SOC 到架构映射。
- `verify_performance.py`：当前只运行一次 `TC_PF` GTest 并解析用例墙钟时间，且参数解析优先扫描 `arch35`，属于结果采集脚手架，不能单独满足 warmup、超过 50 次有效采样或 A2/A3 的 `arch22` 用例选择；代码 PR 需让 A2/A3 SOC 映射到 `arch22`、增加重复采样及设备侧事件或 msprof 口径，并与 `gpu_baseline.csv` 的回填状态一起报告。

在具备 CANN 9.1.0 和 Atlas A2 环境后执行 A2 验证。以下命令是代码 PR 完成 SOC 到 `arch22` 映射、性能脚本采样改造后的目标流程；当前辅助脚本不能直接作为最终验收：

```bash
python3 test_cases/gen_csv.py
python3 test_cases/verify_accuracy.py \
  --repo /path/to/ops-blas --soc ascend910b3 \
  --csv test_cases/sgbmv_test.csv --device 0 --timeout 3600
python3 test_cases/verify_performance.py \
  --repo /path/to/ops-blas --soc ascend910b3 \
  --device 0 --timeout 3600
```

A3 验证使用 `--soc ascend910_93`；在执行前，代码 PR 必须把两个验证脚本的架构映射补为 `ascend910_93 -> arch22`，修复性能脚本的 arch22 用例选择，并确认 `build.sh` 接受该完整 SOC 名称：

```bash
python3 test_cases/verify_accuracy.py \
  --repo /path/to/ops-blas --soc ascend910_93 \
  --csv test_cases/sgbmv_test.csv --device 0 --timeout 3600
python3 test_cases/verify_performance.py \
  --repo /path/to/ops-blas --soc ascend910_93 \
  --device 0 --timeout 3600
```

自测通过后提交：

1. `cann-ops-competitions` 的设计文档 PR；
2. `ops-blas` 的 `blas/gbmv/arch22/` 算子代码和 README 更新；
3. `test/gbmv/sgbmv/arch22/` 测试代码及 CSV；
4. 包含参数、精度、性能、采样次数和内存数据的自测报告；
5. 个人仓库地址、分支、算子目录，并邀请 `Ascend-CANN` 为开发者。

验收前需确认测试框架已支持任务书要求的均匀/正态混合输入；当前随附生成器在 `--dist mixed` 下仍回退为均匀分布，不能把该选项的输出直接宣称为混合分布测试。

### 测试覆盖矩阵

| 类别 | 用例前缀 | 覆盖内容 | 目标数量 |
| --- | --- | --- | ---: |
| 基础通路 | `TC_L0` | N/T/C、小 shape、主对角线和窄带 | 6 |
| 尺寸扫描 | `TC_SQ` | 1、奇数、质数、2 的幂、非对齐 shape | 69 |
| 标量组合 | `TC_AB` | alpha/beta 为 0、1、负值、小数和大值 | 24 |
| 矩形矩阵 | `TC_RC` | `m<n`、`m>n`、N/T/C | 36 |
| 带宽与 padding | `TC_LD`/`TC_BW` | `kl/ku=0`、非对称带、`lda` padding | 22 |
| 特殊填充 | `TC_FL` | 0、交替值、Inf、NaN 和极值 | 12 |
| 边界与负向 | `TC_ED` | 零维、空指针、非法枚举、非法步长和维度 | 21 |
| 扩展采样 | `TC_EX` | 固定 seed 的组合参数采样 | 750 |
| 性能 | `TC_PF` | 任务书四个 case、规模和带宽扫描 | 200 |

CSV 解析必须使用 `a_fill/x_fill/y_fill` 和 `NULLPTR` 字段，不能静默回退到旧的 `a/x/y` 默认字段。正式验收前还要确认测试框架真正生成 50% 均匀分布和 50% 正态分布；在该能力未实现前，报告中只能注明缺口。

## 设计评审与代码交付关系

设计文档 PR 和代码 PR 分开提交：

1. 当前文档提交到 `cann/cann-ops-competitions`，目标文件为 `10-aclblasSgbmv-A2A3/wileyoung/docs/design.md`。
2. 设计评审通过后，代码提交到 `cann/ops-blas` 的 `blas/gbmv/arch22/` 和 `test/gbmv/sgbmv/arch22/`。
3. 代码 PR 描述中引用本设计文档 PR，并附编译、精度、性能和内存自测结果。
4. 设计阶段不填写未经实测的 PASS 数量、平均耗时或性能提升百分比。

## 参考资料

1. [cuBLAS `cublasSgbmv`](https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-gbmv)
2. [Netlib `sgbmv`](https://www.netlib.org/blas/sgbmv.f)
3. [Ascend C 算子开发文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/opdevg/Ascendcopdevg/atlas_ascendc_map_10_0002.html)
4. [ops-blas 仓库](https://gitcode.com/cann/ops-blas)
5. [FLOAT32 精度标准](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md)
