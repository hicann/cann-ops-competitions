# aclblasCswap（Ascend 950PR）算子设计文档

# 需求背景（required）

## 需求来源

本需求来源于“CANN 训练营东南大学——aclblasCswap 算子开发（950）”社区任务。任务要求基于 CANN 9.1.0 和 ops-blas 工程框架，使用 Ascend C Kernel 直调方式，在 Ascend 950PR 上实现单精度复数向量交换接口 `aclblasCswap`，完成设计、开发、精度验证和性能验证，并按社区规范提交代码与测试交付件。

## 背景介绍

`aclblasCswap` 是 BLAS Level 1 数据搬运类接口，用于原地交换两个 `COMPLEX64` 向量中逻辑位置相同的元素。接口语义与 cuBLAS `cublasCswap`、Netlib BLAS `cswap` 保持一致。

每个 `aclblasComplex` 由两个 `float32` 分量构成：

```cpp
struct aclblasComplex {
    float real;
    float imag;
};
```

该算子不进行浮点计算，只搬运实部和虚部，因此输出应与输入逐位一致，不能引入类型转换、舍入或精度损失。

ops-blas 已在公共头文件 `include/cann_ops_blas.h` 中提供统一接口声明。本需求只增加 Ascend 950PR 对应的 `arch35` 实现，不增加芯片私有平行接口。

### aclblasCswap 现有能力分析

| 参数 | 参数含义 | 数据类型 | 存储位置 | 约束 |
| --- | --- | --- | --- | --- |
| `handle` | ops-blas 上下文句柄，携带执行 stream | `aclblasHandle_t` | Host | `n > 0` 时必须有效 |
| `n` | 参与交换的复数元素个数 | `int` | Host | 任意整数；`n <= 0` 为合法 no-op |
| `x` | 第一个输入/输出复数向量 | `aclblasComplex*` | Device | `n > 0` 时不可为空 |
| `incx` | `x` 的元素步长 | `int` | Host | `n > 0` 时不可为 0，可为负数 |
| `y` | 第二个输入/输出复数向量 | `aclblasComplex*` | Device | `n > 0` 时不可为空 |
| `incy` | `y` 的元素步长 | `int` | Host | `n > 0` 时不可为 0，可为负数 |

# 需求分析（required）

## 需求描述

在 Ascend 950PR 上实现以下句柄式 BLAS 接口：

```cpp
aclblasStatus_t aclblasCswap(
    aclblasHandle_t handle,
    int n,
    aclblasComplex* x,
    int incx,
    aclblasComplex* y,
    int incy);
```

接口在 `handle` 绑定的 stream 上异步执行，交换 `x` 和 `y` 的 `n` 个逻辑元素，并支持正步长、负步长以及两个向量使用不同步长的场景。

## 需求拆解

1. 支持 `COMPLEX64` 数据，每个元素的实部和虚部均为 `float32`。
2. 支持 `incx`、`incy` 为测试规格覆盖的非零 `int`，包括正负步长、非单位步长以及两个向量步长符号不同的场景；调用方必须提供满足物理跨度要求的 Device 内存。
3. `n <= 0` 时必须先于其他参数校验直接返回成功，且不访问或修改 `x`、`y`。
4. `n > 0` 时校验句柄、向量指针和步长，并返回任务书规定的状态码。
5. 通过 handle 中绑定的 stream 异步下发 Kernel；调用方在读取结果前同步 stream。
6. 连续向量采用大块 DMA 搬运，非连续向量采用 Compact 模式批量 gather/scatter，避免逐元素 GM 访问。
7. 根据运行时 `n`、AIV 核数和 UB 容量动态确定分核数和 Tile 大小。
8. 结果必须 bit-exact；精度用例覆盖基本形状、边界、特殊值、不同步长和负向参数。
9. Ascend 950PR 三个指定性能用例的平均耗时不得高于任务书标杆。

# 详细设计（required）

## 算子分析

### 数学公式

采用 0-based 逻辑下标 `i = 0, 1, ..., n - 1`。向量的物理下标定义为：

```text
offset(i, n, inc) = i * |inc|,                 inc > 0
                    (n - 1 - i) * |inc|,       inc < 0
```

对每个逻辑位置执行：

```text
tmp                         = x[offset(i, n, incx)]
x[offset(i, n, incx)]       = y[offset(i, n, incy)]
y[offset(i, n, incy)]       = tmp
```

该定义等价于任务书中的 Fortran 1-based 索引和 Netlib `cswap` 负步长语义。

### 支持数据类型

| 向量 | 数据类型 | 单元素大小 | 分量布局 |
| --- | --- | --- | --- |
| `x`、`y` | `aclblasComplex` / `COMPLEX64` | 8 Byte | `real: float32`，`imag: float32` |

Kernel 将复数向量映射为 `float` 全局张量，一次交换连续的两个 `float32` 分量，不进行数值计算。

### 支持形状

输入为逻辑一维向量，参与交换的逻辑长度为 `n`。当 `n > 0` 时，实际分配的最小物理元素数为：

```text
span_x = 1 + (n - 1) * |incx|
span_y = 1 + (n - 1) * |incy|
```

不涉及广播、矩阵 leading dimension 或动态 Tensor Shape 推导；`n`、`incx` 和 `incy` 均为运行时 Host 标量。

## 算子实现

### 实现方案

实现采用 Ascend C Kernel 直调模式，由 Host 参数校验和 Tiling、AIV Kernel、公共接口及测试代码组成：

```text
aclblasCswap
  -> quick return / 参数校验
  -> 获取 AIV 核数和 UB 大小
  -> 生成 CswapTilingData
  -> 在 handle->stream 上下发 cswap_kernel
       -> 连续对齐路径
       -> 连续非对齐路径
       -> 连续不对称 UB 优化路径
       -> 非单位/负步长 Compact 搬运路径
```

#### Host 侧设计

1. Quick return 和参数校验

   - `n <= 0`：立即返回 `ACLBLAS_STATUS_SUCCESS`，此判断位于所有其他参数校验之前。
   - `handle == nullptr`：返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`。
   - `x == nullptr`、`y == nullptr`、`incx == 0` 或 `incy == 0`：返回 `ACLBLAS_STATUS_INVALID_VALUE`。

2. 动态分核

   Host 通过 `GetAivCoreCount()` 获取 Ascend 950PR 可用 AIV 核数，Kernel 启动块数为：

   ```text
   numBlocks = min(n, availableAivCoreCount)
   ```

   当前实测调优结论为 Ascend 950PR 使用全部可用 AIV 核性能最佳。每核基础任务量和余数为：

   ```text
   perCoreN = n / numBlocks
   remainder = n - perCoreN * numBlocks
   ```

   前 `numBlocks - 1` 个核各处理 `perCoreN` 个元素，最后一个核处理 `perCoreN + remainder` 个元素。各核负责的逻辑区间互不重叠。

3. UB 切分

   Host 通过 `PlatformAscendCManager::GetCoreMemSize` 获取 UB 大小；平台信息不可用时使用 248 KiB 作为保守默认值。预留 256 Byte 安全空间后，按两个复数缓冲区计算单 Tile 容量：

   ```text
   bytesPerComplexInUb = 2 * sizeof(aclblasComplex) = 16 Byte
   tileComplex = floor((ubSize - 256) / 16 / 4) * 4
   ```

   `tileComplex` 向下对齐到 4 个复数，即 32 Byte DMA 对齐粒度；极端情况下至少分配 4 个复数。

4. Tiling 数据

   `CswapTilingData` 包含以下字段：

   | 字段 | 含义 |
   | --- | --- |
   | `totalN` | 总逻辑元素数 |
   | `perCoreN` | 每核基础元素数 |
   | `remainder` | 最后一个核追加处理的元素数 |
   | `tileComplex` | 单个等分 UB 缓冲区可容纳的复数元素数 |
   | `incx`、`incy` | 两个向量的步长 |

   算法分支可完全由上述字段在 Kernel 内判断，因此不额外设置 TilingKey。

5. Kernel 下发

   Host 将 `x`、`y`、Tiling 数据和 `numBlocks` 传给 `cswap_kernel_do`，并在 `handle->stream` 上异步启动 AIV Kernel。算子接口本身不执行设备同步。

#### Kernel 侧设计

Kernel 使用 `CswapAIV` 类封装 `Init` 和 `Process` 两个阶段，并根据连续性、对齐状态和单核数据规模选择路径。

1. 核间任务定位

   普通核的起始逻辑偏移为 `blockIdx * perCoreN`，处理 `perCoreN` 个元素；最后一个核额外处理 `remainder`。GM 张量长度按连续向量长度或带步长的物理跨度设置，防止地址计算越界。

2. 连续对齐路径

   当 `incx == 1 && incy == 1`，且当前核的偏移、数据量和 Tile 容量均满足 32 Byte 对齐时，使用 `DataCopy` 将一块 `x` 和 `y` 分别搬入 UB，再交叉写回 GM。循环处理完整 Tile，最后处理剩余 Tile。

3. 连续非对齐路径

   对不满足 32 Byte 对齐的头尾数据，完整对齐部分继续使用 `DataCopy`；不足 32 Byte 的尾部通过 `DataCopyPad` 精确搬入和搬出。写回长度只覆盖有效字节，不修改逻辑向量之外的数据。

4. 连续不对称 UB 优化路径

   当当前核的完整 `x` 分片可以放入两个原等分 Buffer 的总 UB、但不能放入单个 Buffer，且剩余空间至少可容纳一个 32 Byte `y` Tile 时，启用不对称 UB：

   - 一次性将本核完整 `x` 分片搬入 UB；
   - 使用剩余 UB 空间分块搬入 `y`；
   - 将原 `y` 写回 `x`，将 UB 中保留的原 `x` 写回 `y`。

   该路径减少一次 `x` 方向的重复读取，主要优化任务书中的 `n = 1048576` 连续性能用例。更大尺寸继续使用等分 Buffer 路径。

5. 非单位步长及负步长路径

   当任一步长不为 1 时，采用 `DataCopyPad<float, PaddingMode::Compact>` 批量 gather/scatter：

   - 每个逻辑复数对应一个 8 Byte block；
   - GM 相邻 block 的间隔为 `(|inc| - 1) * 8` Byte；
   - 单批最多处理 4088 个复数，保证接口参数范围和 UB 容量安全；
   - 正步长从低地址向高地址搬运；负步长根据 `n`、当前核偏移和批长度计算本批最低物理地址；
   - 当 `incx` 与 `incy` 的符号不同时，将两个 UB 批次分别反转，使交叉写回后仍按相同逻辑下标配对；符号相同时不需反转。

   每批完成 MTE2 搬入后执行流水同步，再交叉执行 MTE3 搬出，批次之间通过 MTE3 屏障保证缓冲区安全复用。

6. 数据精度

   Kernel 仅复制 `float32` 位模式。`NaN` payload、正负零、`Inf` 和普通有限值均不会经过算术运算或类型转换，因此结果应为 0 ULP、逐字节一致。

## 支持硬件

| 支持的芯片版本 | 支持情况 |
| --- | --- |
| Ascend 950PR（arch35） | 支持 |

开发和验收环境为 CANN 9.1.0。其他芯片继续使用仓库原有实现，本设计不改变其行为。

## 算子约束限制

- 仅支持 `aclblasComplex`（COMPLEX64）向量交换。
- `n <= 0` 为合法 no-op，并优先于句柄、指针和步长校验。
- `n > 0` 时 `handle`、`x`、`y` 必须有效，`incx`、`incy` 必须非零。
- 支持单位、非单位及负步长；物理内存长度由调用方按 `1 + (n - 1) * |inc|` 保证。
- 算子为异步接口，读取结果前由调用方同步 handle 绑定的 stream。
- 不涉及广播、额外 leading dimension、Workspace、随机数和浮点计算。
- `x` 与 `y` 为原地输入/输出；除完全相同的存储对象外，不定义部分重叠内存区间的额外行为。

## 内存占用分析

- 算子不申请额外 Workspace，Workspace 占用为 0 Byte。
- `x`、`y` 的 Device 内存由调用方提供，最小物理占用分别为 `span_x * 8` Byte 和 `span_y * 8` Byte，其中 `span = 1 + (n - 1) * |inc|`。
- Kernel 每个 AIV 核只使用片上 UB 临时缓冲，使用量由运行时 UB 容量和 `tileComplex` 限制，不随完整输入规模线性增长，也不产生额外全局内存中间结果。
- 测试程序为构造输入、CPU Golden 和结果回拷而申请的 Host/Device 缓冲区属于测试框架开销，不属于算子 Workspace。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | 输出实部、虚部与 golden 分别按 FLOAT32 标准校验；由于本算子为纯搬运，最终要求 bit-exact、0 ULP | 任务书与生态算子开源精度标准 |
| 性能标准 | Ascend 950PR 上每个指定用例重复 70 次，丢弃前 10 次 warmup，保留 60 次 Device Kernel 有效采样取平均，且平均耗时不高于任务书基准值 | aclblasCswap 950 算子开发任务书与配套测试脚本 |

指定性能门槛如下：

| `n` | `incx` | `incy` | 平均耗时上限（us） |
| ---: | ---: | ---: | ---: |
| 1048576 | 1 | 1 | 23.11 |
| 2097152 | 1 | 1 | 63.82 |
| 4194304 | 1 | 1 | 166.11 |

## 测试设计

测试代码按 ops-blas 标准目录独立于算子实现放置，采用 GTest 参数化和 CSV 驱动：

```text
test/swap/cswap/
├── CMakeLists.txt
├── README.md
├── cswap_param.h
├── cswap_golden.h
└── arch35/
    ├── cswap_npu_wrapper.h
    ├── cswap_test.cpp
    └── cswap_test.csv
```

正式 CSV 共包含 1200 条唯一记录，各类用例数量如下：

| 用例组 | 数量 | 覆盖内容 | 验证方式 |
| --- | ---: | --- | --- |
| `TC_L0_001~008` | 8 | 最小 Shape、单位/非单位及负步长基础场景 | NPU 与 `cblas_cswap` 逐字节比较 |
| `TC_SQ_009~046` | 38 | Shape 扫描、2 的幂及邻近值、跨 Tile/跨核边界 | NPU 与 `cblas_cswap` 逐字节比较 |
| `TC_INC_047~082` | 36 | `incx/incy in {1,2,3,-1,-2,-3}` 的组合 | NPU 与 `cblas_cswap` 逐字节比较 |
| `TC_FL_083~094` | 12 | 随机值、零值、交替值、极值、Inf、NaN | NPU 与 `cblas_cswap` 逐字节比较 |
| `TC_ED_095~106` | 12 | `n <= 0`、空指针、零步长等边界/负向行为 | 校验返回码及 quick return |
| `TC_EX_0107~1000` | 894 | 随机扩展 Shape 和步长组合 | NPU 与 `cblas_cswap` 逐字节比较 |
| `TC_PF_1001~1200` | 200 | 大规模连续向量性能规模及补充回归 | 前 3 条正式性能验收，其余条目作补充功能回归 |

其中，配套 `verify_accuracy.py` 排除全部 `TC_PF`，正式执行前 1000 条精度、功能及异常用例；`verify_performance.py` 只对任务书指定的 `TC_PF_1001`、`TC_PF_1002`、`TC_PF_1003` 进行 70 次 `msprof` 采集。`TC_PF_1004~1200` 不属于任务书三项性能门槛，可在提交前各运行一次，作为大 Shape 的补充 bit-exact 回归验证。

功能覆盖包括：

- `n = 1`、小质数、2 的幂及其邻近值、较大 Shape 和随机扩展 Shape；
- `incx`、`incy` 的 1、2、-1、-2、非常规步长及异号组合；
- 均匀/正态分布、零值、正负值、`Inf`、`NaN` 和极值；
- `n = 0`、负 `n`、空指针、零步长等 quick return 或非法参数行为；
- 连续对齐、连续尾块、非连续 Compact 搬运、负步长和跨 Tile/跨核场景；
- 任务书指定的三个连续大规模性能用例。

NPU wrapper 负责 Host/Device 内存分配、拷贝、接口调用、设备同步、结果回拷和资源释放。CPU Golden 在 `test/swap/cswap/cswap_golden.h` 中通过 `cblas_compat.h` 直接调用系统 BLAS 提供的 Netlib 语义接口 `cblas_cswap`；测试程序对 `x`、`y` 的完整物理缓冲区执行 `memcmp`，因此实部、虚部、NaN payload、正负零等位模式均按 bit-exact 标准验证。测试可执行文件链接 `libblas`，保证 Golden 与算子实现相互独立。

性能测试在 Ascend 950PR、CANN 9.1.0 和独占卡条件下执行。每个指定用例由 `msprof` 驱动 GTest 重复 70 次，丢弃前 10 条 warmup，保留后 60 条 `cswap_kernel` Device 样本，以 `Task Duration(us)` 的算术平均值作为验收数据。GTest 输出中的整用例毫秒数包含 Host 准备、内存传输和校验，不作为 Kernel 性能依据。

## 自测结果

### 精度与功能结果

修改 CPU Golden 为直接调用 `cblas_cswap` 后，在 Ascend 950PR / CANN 9.1.0 环境重新编译并完成正式精度与功能验证：

| 用例组 | 通过数/总数 | 结果 |
| --- | ---: | --- |
| `TC_L0_001~008` | 8/8 | PASS |
| `TC_SQ_009~046` | 38/38 | PASS |
| `TC_INC_047~082` | 36/36 | PASS |
| `TC_FL_083~094` | 12/12 | PASS |
| `TC_ED_095~106` | 12/12 | PASS |
| `TC_EX_0107~1000` | 894/894 | PASS |
| `CswapArch35Test.NullHandle` | 1/1 | PASS |
| 合计（CSV 正式精度/功能用例） | 1000/1000 | PASS |

### 性能结果

性能采集批次为 `20260908_061742`。每项采集 70 次，丢弃前 10 次 warmup 后使用 60 条有效 Device 样本：

| 用例 | `n` | 平均 Task Duration（us） | 上限（us） | 余量（us） | 结果 |
| --- | ---: | ---: | ---: | ---: | --- |
| `TC_PF_1001` | 1048576 | 22.568 | 23.110 | 0.542 | PASS |
| `TC_PF_1002` | 2097152 | 40.514 | 63.820 | 23.306 | PASS |
| `TC_PF_1003` | 4194304 | 66.170 | 166.110 | 99.940 | PASS |

性能截图、原始 `msprof` 数据和 `performance_summary_final.csv` 随正式自测报告提交，不以 GTest 墙钟时间替代 Device Kernel 时间。

## 兼容性分析

本变更复用已有公共 API 声明和状态码，只增加 `blas/swap/arch35/` 的 Ascend 950PR 实现及对应测试，不改变已有 A2/A3 实现，不引入 ABI 变化。`n <= 0`、空句柄、非法指针和步长的返回行为与任务书保持一致。

# 交付目录说明

算子代码与测试代码需要按目录分开，但可以在同一个 ops-blas 功能分支和同一个 PR 中提交：

```text
算子实现：blas/swap/arch35/
算子文档：blas/swap/README.md
公共接口：include/cann_ops_blas.h（本任务接口已存在时不重复修改）
测试代码：test/swap/cswap/ 与 test/swap/cswap/arch35/
```

算子设计文档不放入 ops-blas 功能 PR，应按任务书要求单独提交到 `cann/cann-ops-competitions` 仓库的 2026 社区任务 `tasklist` 目录，并单独发起设计文档 PR。自测报告、性能原始数据和截图作为社区任务 IT 系统验收材料提交，不应作为 ops-blas 源码仓中的编译交付件。

# 参考资料

1. [ops-blas 开源仓](https://gitcode.com/cann/ops-blas)
2. [Netlib BLAS `cswap` 参考实现](https://www.netlib.org/blas/cswap.f)
3. [cuBLAS `cublasCswap` 接口说明](https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-swap)
4. [生态算子开源精度标准](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md)
5. [Ascend C 算子开发文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/opdevg/Ascendcopdevg/atlas_ascendc_map_10_0002.html)
