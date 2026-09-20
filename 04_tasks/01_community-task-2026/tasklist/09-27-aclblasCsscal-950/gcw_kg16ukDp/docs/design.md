# 需求背景（required）

## 需求来源

CANN 训练营东南大学第 27 项 `aclblasCsscal` 算子开发（Ascend 950PR）。
本目录对应当前报名账号 `gcw_kg16ukDp`，按官方设计模板提交。

本文延续已公开的 [原设计 PR #1463](https://gitcode.com/cann/cann-ops-competitions/pull/1463)
及同一开发成果，补充 2026-09-15 验收修复与自测结果；不把账号迁移描述为
新的独立实现，不删除原代码作者、提交历史或许可证。

## 背景介绍

Csscal 将单精度复数向量的实部、虚部分别乘以一个 float32 实数标量。
它不需要复数乘法交叉项，也不需要反交织。复数标量接口 `aclblasCscal`
属于另一任务，代码、原题 CSV、性能基线与报告不混用。

# 需求分析（required）

## 需求描述

```cpp
aclblasStatus_t aclblasCsscal(aclblasHandle_t handle, int n,
    const float* alpha, aclblasComplex* x, int incx);
```

复用 `include/cann_ops_blas.h` 中的公开声明与句柄，不增加 950PR 私有 ABI。
`alpha` 是 Host float 指针，`x` 是 Device complex64 指针，`incx` 单位是复数。
通过 handle 所绑定的 stream 异步下发 Ascend C kernel。

## 需求拆解

1. 先检查空 handle，再检查合法 no-op。有效 handle 下，`n<=0` 或 `incx<=0`
   返回成功，不修改 x；正常规模的空 alpha/x 返回错误。
2. alpha 为 1 时可直接返回。alpha 为正零或负零时，按任务书明确要求向
   选中元素写正零，包括原值为 Inf/NaN 的情况；保留此规则与 raw Netlib
   浮点乘法之间的差异，不将零分支误写成 no-op。
3. 任意正步长原位更新，保留步长空隙、对齐偏移与前后哨兵。
4. 原题 CSV 逐字节保留全部 1200 行，包括 1000 精度和 200 性能。
   20 条偏移补充放入独立 `csscal_supplement.csv`，不得修改官方性能数据来达标。

# 详细设计（required）

## 算子分析

### 数学公式

对 `i=0..n-1`，令 `k=i*incx`：

```text
x[k].real = alpha * x[k].real
x[k].imag = alpha * x[k].imag
```

非零 alpha 使用两次独立 float32 乘法。零分支按任务规格写正零。

### 支持数据类型与形状

输入输出均为 complex64，alpha 为 float32。逻辑 shape 为一维 n，物理跨度
为 `(n-1)*incx+1` 个复数。无广播、归约、核间通信或 Cube 计算。

## 算子实现

### Host 侧设计

实现位于 `blas/scal/arch35/`。以 64 位整数计算跨度、字节长度和偏移，
检查可计算的溢出及无效对齐；实际分配范围仍由调用方负责。
`GetAivCoreCount()` 查询可用核数，查询失败或为零时报错，不硬编码核数。

连续向量按 4 个复数组成的 32 字节组均分到活动核，仅末核处理不足一组
的尾部。每核目标工作量为 2048 个复数，活动核数受设备核数、可用组数
和任务量共同限制。跨步长路径采用 SIMT，128 个线程使用 64 位索引遍历。

TilingData 为 40 字节 POD，包含 n、incx、alpha、coreCount、tileComplex、
simtThreads、writeZero、useVector。Host 不分配额外 GM 工作区，不复制
向量，不同步 stream。

### Kernel 侧设计

连续路径使用 CopyIn、Muls、CopyOut，将 n 个复数视作 2n 个 float。
tile 上限为 2048 个复数，`DataCopyPad` 只读写有效 GM 字节，在 UB 内
处理尾部补齐，避免覆盖相邻分区或视图外数据。
输入、输出队列均采用单缓冲，以队列事件约束 MTE2、向量计算、MTE3。
TPipe 在 kernel 栈上创建并以指针传入算子类。

零路径省去输入读取，在 UB 中 Duplicate 正零后写回有效元素。
跨步长路径只访问 `2*i*incx` 和相邻虚部，保留空隙。
受测配置为连续向量、跨步长 SIMT、单缓冲、tile=2048、task_zero。
诊断用双缓冲或 auto 配置不作为本提交性能结论。

### 内存预算

单核两条单缓冲队列合计 32768 B，静态预算另预留 8192 B，低于保守
128 KiB 编译期预算。这是静态预算，不是 profiler 峰值实测。
算子额外 GM workspace 为 0。测试器 work/pristine 缓冲的最大实际申请
为 67109376 B，不把测试器内存误记成算子 workspace。

## 支持硬件

| 芯片 | 验证状态 |
| --- | --- |
| Ascend 950PR | CANN 9.1.0 板端精度、性能通过 |
| 其他芯片 | 未验证，不声明支持 |

## 算子约束限制

调用者保证 Host alpha 可读、Device x 分配足够并在读回前同步 stream。
SUCCESS 表示异步提交成功，设备执行错误在同步时检查。
哨兵验证覆盖写越界，不替代专门的读越界检测。

# 可维可测分析

## 精度标准/性能标准

实部、虚部按 float32 独立比较：rtol=2^-10、atol=2^-16、匹配比例至少
99%，最大绝对误差不超过 1e-2 或 32 ULP。非有限值单独匹配；零策略
单独验证。性能预热 10 次，有效采样 64 次，满足任务要求的超过 50 次。

| 2026-09-15 受测项目 | 结果 |
| --- | --- |
| 完整正式 GTest | 1223/1223：1200 原题、20 偏移补充、3 直接 API |
| 官方原样精度脚本 | 1020 PASS、0 FAIL：1000 原题、20 补充 |
| 独立驱动原题精度 | 1000/1000 |
| 独立驱动补充精度 | 226/226 |
| 原题全部性能 | 200/200 |
| 最差 NPU/有效性能上限 | 0.8673329039980306，低于 1 |

三个主用例 n=2^20、2^21、2^22，incx=1，任务书上限分别为 13.57、
21.05、43.02 us。以原题 GPU baseline 与任务书中更严格的有效上限为准，
不修改基线数值、放宽阈值或删除慢用例。

计时使用 ACL timeline event，单次调用前以 D2D 恢复 pristine 输入，恢复
不计入调用时间。因此结果属于 warm-reset 条件，不等同于冷缓存或 msprof
纯 kernel 时间。正式测试与独立计时分开保留逐例 CSV、XML、原始日志及
前后设备状态。2026-09-09 的 1023 条旧 ST 不充当此次 1223 条完整修复结果。

## 验收修复与证据版本

官方反馈曾指出仓内 `csscal_test.csv` 的性能数据与原题不同。修复将其恢复
为原题字节序列和 CRLF，并将偏移补充拆到独立 CSV。官方 CSV SHA256：

```text
c47efccf084e7805dcc7c5cbc812e22d02ff8b5790bda85d1b373324720cbeef
```

修复提交 `4559e7057252a2d044c3208f9cc11b62dc9cc144`；携带完整交付件的
版本 `54f7b3b277ed5a5c05fe02b9a3cd9eab72171837`。发布源文件与板端受测文件
已核对哈希。独立计时程序通过 LD_LIBRARY_PATH 加载新生产库，并保留
ldd 与库、二进制哈希证据。

交付包 `csscal950_platform_20260915.zip` 含 54 项：源码、设计、XLSX 报告、
逐例结果、XML、官方脚本输出和截图。SHA256：

```text
b0b1d6a2055eef589ac8d765f53000e2e2a1621cc737b715c51d319061f9da1f
```

2026-09-19 的账号归属修正未修改受测源码，不冒充当天新执行的 NPU 测试。
历史公开证据：[原代码及交付分支](https://gitcode.com/gcw_XdlmKQLn/ops-blas/tree/delivery/csscal950-acceptance)。
当前报名账号的[代码及交付分支](https://gitcode.com/gcw_kg16ukDp/ops-blas/tree/delivery/csscal950-acceptance-20260919)
已发布，仍是上述 `54f7b3b`，保留原提交历史；本地或自测通过不表示
官方验收通过，设计 PR 也须经维护者评审。

## 兼容性分析

沿用现有公开接口、handle 和 stream。Csscal 的实现不修改其他算子的
计算规格。README 产品表标注 Ascend 950PR，测试代码位于
`test/scal/csscal/arch35/`，可按测试 README 复现。

## 参考资料

- [ops-blas](https://gitcode.com/cann/ops-blas)
- [官方设计模板](https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md)
- [官方自测报告模板](https://docs.qq.com/sheet/DUmVWWndaUE12WGFB?tab=BB08J2)
- [Netlib csscal](https://www.netlib.org/blas/csscal.f)
