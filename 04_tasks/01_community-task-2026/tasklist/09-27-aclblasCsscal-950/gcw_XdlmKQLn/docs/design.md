# 需求背景（required）

## 需求来源

CANN 训练营东南大学 aclblasCsscal 算子开发（950）任务。依据用户提供的官方说明书与 1200 条原题 CSV，实现 `aclblasCsscal`，在 Ascend 950PR、CANN 9.1.0 上完成板端自验。

## 背景介绍

Csscal 对单精度复数向量做实数标量缩放。输入按实部、虚部交错存储，无需复数乘法的交叉项，也不需要反交织。它与复数 alpha 的 `aclblasCscal` 是两个独立接口，代码、用例、GPU 基线和报告不混用。

# 需求分析（required）

## 需求描述

```cpp
aclblasStatus_t aclblasCsscal(aclblasHandle_t handle, int n,
    const float* alpha, aclblasComplex* x, int incx);
```

`alpha` 是 Host float 指针，`x` 是 Device complex64 指针，`incx` 的单位是复数元素。接口复用上游 `include/cann_ops_blas.h` 的声明、类型和 handle 所绑定的 stream，异步直调 Ascend C kernel。

## 需求拆解

1. 按顺序检查空 handle、合法 no-op、数据指针与 identity。有效 handle 下，`n<=0` 或 `incx<=0` 成功返回；正规模时 alpha/x 空指针报错；alpha 为 1 不启动 kernel。
2. alpha 为正零或负零时，只将选中元素写为正零，包括原值为 Inf/NaN 的情况。此项按题目零策略，保留与 raw Netlib 的差异。
3. 正步长支持原位更新，保护 stride 空隙、对齐偏移和分配区前后哨兵。用 64 位计算物理跨度和偏移，拒绝跨度、地址加法溢出及无效对齐。
4. 完整覆盖原题 1000 精度、200 性能，补充 226 条精度用例，不修改原始输入表和 GPU 基线。

# 详细设计（required）

## 算子分析

### 数学公式

对 `i=0..n-1`，`k=i*incx`：

`x[k].real = alpha * x[k].real`

`x[k].imag = alpha * x[k].imag`

alpha 为零使用题目明确的写正零分支；alpha 非零使用 float32 独立乘法，不降低精度。

### 支持数据类型与形状

输入、输出均为 complex64，alpha 为 float32。逻辑形状为一维 n，物理跨度为 `(n-1)*incx+1` 个复数。无广播、归约、跨核通信或 Cube 计算。

## 算子实现

### Host 侧设计

实现位于 `blas/scal/arch35/`。先执行参数检查，再通过 `GetAivCoreCount()` 查询可用核数，查询为零时失败，不硬编码芯片核数。

连续向量以 4 个复数组成 32 字节组，按组数将工作均衡分给活动核，只有末核接收不足 4 个复数的尾部。每核目标工作量为 2048 个复数，活动核数不超过设备核数、可用组数和任务所需核数。

跨步长采用 SIMT：每核按 64 位范围分区，128 个线程以 blockDim 为步幅处理选中元素。Host 不申请 GM 工作区、不复制数据、不做 stream 同步。

TilingData 为 40 字节 POD，携带 n、incx、alpha、coreCount、tileComplex、simtThreads、writeZero、useVector；不使用核间分配数组。

### Kernel 侧设计

连续路径依次 CopyIn、Muls、CopyOut。把 n 个复数视作 2n 个 float，每个 tile 最多 2048 个复数。DataCopyPad 只搬运有效 GM 字节，尾部仅在 UB 内补齐，避免覆盖相邻核或视图外数据。输入、输出队列各一个缓冲区，队列事件建立 MTE2、向量计算、MTE3 的依赖。TPipe 在 kernel 栈上创建，通过指针交给算子类。

零路径省去输入读取，Duplicate 写入 UB 正零后，仅写回有效选中元素。strided 路径直接访问 `2*i*incx` 和相邻虚部，保持空隙不变。

提交配置为强制连续向量、跨步长 SIMT、单缓冲、tile=2048、task_zero。双缓冲和 auto 阈值配置保留为诊断候选，不作为本轮性能结论依据。

### 内存预算

单核两条单缓冲队列合计 32768 B，另按静态预算预留 8192 B，小于保守的 128 KiB 编译期预算。这是静态预算，不冒充 profiler 实测峰值。算子无额外 GM 工作区；测试器 work/pristine 缓冲逐例记录，最大实际申请 67109376 B。

## 支持硬件

| 芯片 | 验证状态 |
| --- | --- |
| Ascend 950PR | CANN 9.1.0 板端精度、性能通过 |
| 其他芯片 | 未验证，不声明支持 |

## 算子约束限制

调用者负责保证 Host alpha 可读及 Device x 实际分配足够。仅能检查可计算的跨度和地址溢出，不能据此证明任意指针的真实分配范围。返回 SUCCESS 表示异步提交，设备执行错误由调用者同步 stream 时观察。哨兵验证覆盖写越界，不能替代读越界专用工具。

# 可维可测分析

## 精度标准/性能标准

| 项目 | 标准 | 本轮结果 |
| --- | --- | --- |
| 实部、虚部 | rtol=2^-10，atol=2^-16；匹配比例≥99%；abs≤1e-2 或 ULP≤32；非有限值单独匹配 | 原题 1000/1000，补充 226/226 |
| 主用例 n=2^20 | ≤13.57 us | 最终源码 8.8357 us |
| 主用例 n=2^21 | ≤21.05 us，验收器取更严格 21.0475 us | 最终源码 15.3161 us |
| 主用例 n=2^22 | ≤43.02 us | 最终源码 28.4070 us |
| 全部性能用例 | 每例均值不超过有效上限 | 五轮各 200/200 |
| 正式社区 ST | 原题派生精度、偏移补充及直接 API 边界 | 最终 1023/1023 |

测试环境为 CPU22、NPU0、Ascend950PR、CANN9.1.0。每例 10 次 warmup、64 个有效样本，使用 ACL timeline event 的单次接口区间；每次区间前 D2D 恢复 pristine 输入，恢复时间不计入区间，因此属于 warm-reset 缓存条件，不等同于冷缓存或 msprof 纯 kernel 时间。

复测构建记录绑定上游 `7f93ab5c73916c13f2b46f80c03a90badd65f9cf`、ops-tensor、编译器、生产源码、测试源码、配置、库与二进制。原始逐例 CSV、进程退出码、前后 NPU 状态和哈希检查一并保留。SIMT 性能失败的初始尝试同样保留。

## 兼容性分析

复用现有公开接口声明，不增加私有 ABI；Csscal实现不改变其他算子的计算规格。正式社区 ST 与完整性能验收工具分别交付。最终记录为2026-09-09的community-structure-final-03，受测构建为build-community-r4。本文按官方设计模板组织；本地文档完成不表示设计 PR 已评审合入。

## 参考资料

- [任务代码仓](https://gitcode.com/cann/ops-blas)
- [Netlib csscal](https://www.netlib.org/blas/csscal.f)
- [官方设计模板](https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md)
- [官方自测报告模板](https://docs.qq.com/sheet/DUmVWWndaUE12WGFB?tab=BB08J2)
