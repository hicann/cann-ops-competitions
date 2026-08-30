# aclblasCcopy 算子设计文档

| 项目 | 内容 |
| --- | --- |
| 任务来源 | 2026 年 8 月 `aclblasCcopy` Ascend 950PR 社区任务书 |
| 目标仓库 | `cann/ops-blas` |
| 目标目录 | `blas/copy/arch35/`、`test/copy/ccopy/` |
| 公开接口 | `aclblasCcopy` |
| 适配硬件 | Ascend 950PR |
| 验证环境 | Ascend 950PR、CANN 9.1.0 |
| 实现语言 | C++ Host、Ascend C Kernel |

本文说明 `aclblasCcopy` 的接口语义、Host 分派、Kernel 数据搬运和性能优化方案。详细测试命令、
逐项数据和原始日志随自测材料单独交付，不在本文重复展开。

## 需求背景（required）

### 需求来源

任务要求在 ops-blas 现有句柄式接口下，为 Ascend 950PR 实现 complex64 向量复制：

```cpp
aclblasStatus_t aclblasCcopy(
    aclblasHandle_t handle,
    int n,
    const aclblasComplex* x,
    int incx,
    aclblasComplex* y,
    int incy);
```

接口声明复用 `include/cann_ops_blas.h`，实现位于 `blas/copy/arch35/`，通过 handle 绑定的 stream
异步直调 Ascend C Kernel，不新增 950PR 私有接口。

### 功能语义

`aclblasCcopy` 不执行浮点计算，只按 BLAS 步长语义复制 `n` 个 complex64。调用方传入物理 span
的低地址基址，第 `i` 个逻辑元素的位置为：

```text
physical(inc, i) = i * inc,                         inc > 0
                 = (n - 1 - i) * abs(inc),         inc < 0

y[physical(incy, i)] = x[physical(incx, i)],       i = 0 .. n-1
```

每个 complex64 包含两个 FP32 分量。复制必须保持 NaN payload、Inf、正负零和实虚部顺序，且不得
覆盖 y 的步长空洞，因此实现始终按原始位模式搬运。

## 需求分析（required）

### 功能与边界

| 项目 | 设计要求 |
| --- | --- |
| `n < 0` | 返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| `n == 0` | 无条件返回成功，不访问 handle、x、y 和步长 |
| 空 handle | `n > 0` 时返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| 空 x/y、零步长 | `n > 0` 时返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| 正负步长 | 支持正数、负数和异号组合 |
| 物理 span | 至少 `1 + (n - 1) * abs(inc)` 个 complex64 |
| 输出范围 | 只覆盖 y 的逻辑目标元素，其他位置保持不变 |
| 数值要求 | 实部、虚部逐 bit 复制，误差为 0 |
| 执行方式 | 在 handle stream 上异步下发，不引入 Host 侧设备同步 |

跨度、绝对步长和 GM 偏移使用 64-bit 中间值，避免 `INT_MIN` 取绝对值或跨度计算发生 32-bit
溢出。任务书没有规定部分重叠的 x/y span，当前实现不承诺 `memmove` 语义。

### 验收材料口径

任务书与随包测试材料**不能**推导出同一个性能标准：

| 来源 | 口径 |
| --- | --- |
| 任务书 §3.3 | 1M / 4M / 16M，绝对上限 2.08 / 3.79 / 10.83 us |
| 测试 README | 代表 case 为 1M / 2M / 4M |
| `gpu_baseline.csv` | 200 条 1～4194304 的 GPU 基线；没有 16M |
| 随包验证公式 | `GPU_us / NPU_us >= 0.4`，即 `NPU_us <= GPU_us / 0.4` |

任务书三个绝对值分别约等于 GPU 基线 5.203 / 9.482 / 27.075 us **乘以** 0.4，但其第二、第三个
尺寸又从 2M / 4M 错位为 4M / 16M；随包公式使用的则是 **除以** 0.4。两种倍率方向相差 6.25 倍。
其中 16M 的 10.83 us 隐含约 24.8 TB/s 读写带宽，是同机 ACL D2D 实测的 13.55 倍。

随包原始 `verify_performance.py` 的计时链路也不适合微秒级验收：它解析 GTest `[ OK ]` 行中的
整数毫秒墙钟，覆盖内存分配、数据生成、H2D/D2H、cblas golden 和结果比对，而不是单独的 Kernel
耗时；小尺寸还会因取整为 0 ms 被标记为 `NO_REF`。因此原脚本可以触发性能 shape 的功能运行，
但不能直接作为 Kernel 性能门禁。仓内重构版改用独立 benchmark 和 NPU device event，先 warmup
20 次，再采样 100 次取平均。

在任务方书面澄清前，本文分别报告两种结果：

- `package`：按 200 条 GPU 基线和随包倍率公式逐点判定；
- `taskbook`：按任务书 1M / 4M / 16M 绝对值诊断，不与 `package` 结论混写；
- 16M 因没有 GPU baseline，在 `package` 下标记为 `NO_REF`，只作为大尺寸诊断点。

## 详细设计（required）

### 总体架构

```text
aclblasCcopy
  -> Host 参数校验
  -> 查询 AIV 核数并按步长、规模分派
  -> 在 handle stream 上启动一个 AIV Kernel
       |- incx=1, incy=1, n<=3*1024*1024：SIMT 64-bit GM 直拷
       |- incx=1, incy=1, n> 3*1024*1024：TPipe + MTE 单/双 buffer
       `- 其他步长：TPipe + Compact 搬运 + 必要的复数对逆序
```

Host 不申请临时 Device 内存，不复制独立 tiling buffer，也不执行同步；TilingData 按值传入 Kernel。
Kernel GM workspace 为 0 B。

### Host 校验与分派

Host 依次执行：

1. 检查 `n < 0` 和 `n == 0`；
2. 检查 handle、x、y 和步长；
3. 动态获取 AIV 核数；
4. 连续小/中尺寸进入 SIMT 路径，其余输入计算 MTE/跨步 Tiling；
5. 在 handle stream 上异步启动 Kernel。

两类路径采用不同分核方式：

| 路径 | block 数 | 单 block 工作分配 |
| --- | --- | --- |
| SIMT 连续路径 | `min(ceil(n / 128), AIV核数)` | 线程数按 128 对齐、最大 2048，使用 grid-stride 循环 |
| MTE/跨步路径 | `min(n, AIV核数)` | 以 4 个复数（32 B）为单位均分，尾部交给最后一核 |

MTE/跨步路径的基础分配为：

```text
perCoreN        = floor(floor(n / numBlocks) / 4) * 4
leftover        = n - perCoreN * numBlocks
extraBlockCores = floor(leftover / 4)
tailElements    = leftover % 4
```

每核根据 `blockIdx` 恢复自己的逻辑 offset 和 count，不依赖固定核数或 TilingData 数组。

### 连续 SIMT 快路径

初版所有连续输入都走 TPipe/TQue。Profiler 证明小尺寸耗时主要来自队列初始化，而不是数据搬运：
n=7185 时 AIV 均值为 10.786 us。当前实现对不超过 `3 * 1024 * 1024` 个元素的连续输入使用 SIMT：

- 一个 complex64 作为一个 `uint64_t` 位模式单元 load/store；
- 线程按连续地址合并访问，并以 grid-stride 循环覆盖全部元素；
- 不构造 TPipe，不申请 UB Queue，不使用 workspace。

同一 n=7185 的 AIV 均值降至 1.641 us，无 Profiler 的 device-event 耗时由 10.658 us 降至
2.213～2.262 us。阈值以上继续使用 MTE 路径，保留大尺寸搬运带宽。

### 连续 MTE 路径

大尺寸连续输入以两个 `uint32_t` lane 表示一个复数，经 GM→UB→GM 搬运：

- 每核数据可一次放入 UB 时使用单 queue slot；
- 需要多 tile 时使用双 buffer prime-pump-drain，重叠相邻 tile 的 MTE2 与 MTE3；
- 非 32 B 尾部通过 `DataCopyPad` 搬运，写回长度严格等于有效数据。

Host 查询 UB 大小，保留 256 B 安全余量。单 buffer 使用 `tileSize * 8` B，双 buffer 使用
`2 * tileSize * 8` B，tileSize 按 4 个复数向下对齐。

### 跨步与负步长路径

跨步读写使用 `DataCopyPad<..., PaddingMode::Compact>`，每个 complex64 作为一个 8 B block：

```text
srcStride = (abs(incx) - 1) * 8
dstStride = (abs(incy) - 1) * 8
```

每批最多处理 4092 个复数，确保 Compact blockCount 可表示。跨步路径使用两个 queue slot 和一个
复数对重排 buffer，总 UB 不超过可用预算。

负步长分片的物理起点为：

```text
base = (n - elementOffset - elementCount) * abs(inc)
```

incx 与 incy 同号时，读写两侧的物理逆序相互抵消；异号时在 UB 中按 complex pair 逆序，始终把
实部和虚部作为整体移动。Compact write 只触达目标 8 B block，因此不会污染 y 的步长空洞。

### Bit-exact、异步与内存保证

- SIMT 路径使用 `uint64_t` 位模式搬运，MTE/跨步路径使用 `uint32_t` lane 搬运；
- 不执行浮点运算或类型转换，NaN payload、Inf 和有符号零保持不变；
- Host 不创建私有 stream，不执行 Device 同步；
- Kernel GM workspace 和 Host 临时 Device 分配均为 0 B。

## 支持硬件

| 芯片 | 状态 |
| --- | --- |
| Ascend 950PR | 已完成 CANN 9.1.0 实机验证；性能口径见“可维可测分析” |

未实测芯片不在本文承诺范围内。

## 算子约束限制

| 约束项 | 说明 |
| --- | --- |
| 数据类型 | complex64（两个 FP32 分量） |
| `n` | 非负 `int` |
| 步长 | 非 0 `int`，支持正负 |
| 物理 span | 至少 `1 + (n - 1) * abs(inc)` 个 complex64 |
| x/y 重叠 | 不承诺部分重叠 span 的 `memmove` 语义 |
| 执行方式 | handle stream 异步直调 |
| Kernel GM workspace | 0 B |
| 数值误差 | 不允许；完整物理输出 span 逐字节比对 |

## 可维可测分析

### 功能与精度

测试由随包 CSV 1200 项和 3 项补充回归组成，覆盖正负步长、异号步长、特殊位模式、空参数、
零长度、步长空洞和非 32 B 对齐地址。CPU golden 使用 `cblas_ccopy`，输出按完整物理 span 执行
`memcmp`。

| 检查项 | Ascend 950PR 结果 |
| --- | --- |
| 全量 GTest | 1203/1203 PASS |
| 随包官方精度脚本 | 1000/1000 PASS |
| 非零 Device offset | 212/212 PASS |
| bit-exact、前缀和空洞保护 | PASS |

### 性能

性能由同一 stream 上的 NPU event 计时，每个 shape warmup 20 次、有效采样 100 次。`package`
策略对随包 200 条 GPU baseline 逐点执行 `GPU_us / NPU_us >= 0.4`。

| n | 两轮 NPU 平均耗时 us | `package` 上限 us | 结果 |
| ---: | ---: | ---: | --- |
| 1048576 | 6.723～6.756 | 13.008 | PASS |
| 2097152 | 11.512～11.557 | 23.705 | PASS |
| 4194304 | 37.018～39.027 | 67.688 | PASS |
| 16777216 | 203.458～203.749 | 无 GPU baseline | NO_REF |

两轮全量 200 点均为 200 PASS、0 FAIL，最差 GPU/NPU 分别为 0.648 和 0.627。任务书绝对值
1M / 4M / 16M 仍分别判定为 FAIL，等待任务方确认最终口径。

### 兼容性

公开 API 声明、参数顺序和返回码未变；arch22 实现未修改。arch35 实现只使用 handle 已绑定的
stream，不引入全局状态、私有 stream 或 Host 同步。Scopy 只在 benchmark 中作为同字节诊断基线，
生产实现不受影响。

## 参考资料

1. [CANN 社区任务 2026](https://gitcode.com/cann/cann-ops-competitions/tree/master/04_tasks/01_community-task-2026)
2. [ops-blas](https://gitcode.com/cann/ops-blas)
3. [Netlib CCOPY](https://www.netlib.org/blas/ccopy.f)
4. [cuBLAS Copy API](https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-copy)
5. [生态算子开源精度标准](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md)
6. [待验收代码分支](https://gitcode.com/xukunming2025/ops-blas/tree/codex/ccopy-performance-opt)
