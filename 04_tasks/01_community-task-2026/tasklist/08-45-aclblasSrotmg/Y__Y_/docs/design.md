# aclblasSrotmg A2/A3 算子设计

# 需求背景（required）

## 需求来源

本设计对应 2026 年 8 月社区任务“aclblasSrotmg 算子开发（A2/A3）”。目标是在
Atlas A2/A3 系列产品上补齐 `aclblasSrotmg` 的 Arch22 实现，并保持与
ops-blas 既有统一接口、Netlib BLAS `srotmg` 语义和 cuBLAS 对标接口一致。

代码目标仓库：<https://gitcode.com/cann/ops-blas>。

实现目录：

- Host 与 Kernel：`blas/rotmg/arch22/`
- 测试：`test/rotmg/srotmg/arch22/`
- 公共接口：`include/cann_ops_blas.h` 中既有 `aclblasSrotmg` 声明

## 背景介绍

`rotmg` 是 BLAS Level 1 标量算子。它根据四个 FP32 标量 `d1`、`d2`、
`x1`、`y1` 构造修正 Givens 旋转矩阵 H，使加权二维向量的第二个分量被消去。
算子原地更新 `d1`、`d2`、`x1`，保持 `y1` 只读，并通过 `param[5]` 返回矩阵
编码。

本算子没有长度、步长和维度参数，单次调用只处理固定数量标量。因此设计重点是：

1. 完整复现 Netlib `srotmg` 的分支和缩放保护语义；
2. 正确处理 Host/Device 指针位置和调用方 stream；
3. 降低 Device 标量 Kernel 的固定启动与访存开销；
4. 在 A2/A3 共用的 Arch22 代码中保持一致行为。

## 现状分析

ops-blas 已声明统一接口：

```cpp
aclblasStatus_t aclblasSrotmg(
    aclblasHandle_t handle,
    float* d1,
    float* d2,
    float* x1,
    const float* y1,
    float* param);
```

Arch22 需要补充 Host、Ascend C Kernel、构建接入和 CSV 驱动测试。实现遵循：

- `handle == nullptr` 返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`；
- 任一数据指针为空返回 `ACLBLAS_STATUS_INVALID_VALUE`；
- 五个数据指针必须全部位于 Host 或全部位于 Device；
- 全 Host 指针执行 Host 原生计算；
- 全 Device 指针在 handle 绑定的 stream 上异步发射 NPU Kernel；
- 混合 Host/Device 指针返回 `ACLBLAS_STATUS_INVALID_VALUE`。

# 需求分析（required）

## 需求描述

使用 Ascend C 实现 A2/A3 Arch22 的 FP32 `aclblasSrotmg`，支持 Host/Device
两种合法指针模式，输出与 Netlib `cblas_srotmg` 对齐，并满足任务书规定的精度、
性能、错误码和异步 stream 语义。

## 参数与输出

| 参数 | 方向 | 类型 | 形状 | 说明 |
| --- | --- | --- | --- | --- |
| `handle` | 输入 | `aclblasHandle_t` | - | 携带调用方 stream |
| `d1` | 输入/输出 | FP32 | `[1]` | 第一个缩放因子，原地更新 |
| `d2` | 输入/输出 | FP32 | `[1]` | 第二个缩放因子，原地更新 |
| `x1` | 输入/输出 | FP32 | `[1]` | 第一个向量分量，原地更新 |
| `y1` | 输入 | FP32 | `[1]` | 第二个向量分量，只读 |
| `param` | 输出 | FP32 | `[5]` | flag 与 H 的编码 |

`param[0]` 为 `flag`，取值为 `-2`、`-1`、`0` 或 `1`：

| flag | H 的形式 | `param` 中显式保存的元素 |
| ---: | --- | --- |
| -2 | 单位矩阵 | 仅 flag |
| -1 | 一般 2×2 矩阵 | h11、h21、h12、h22 |
| 0 | `[[1,h12],[h21,1]]` | h21、h12 |
| 1 | `[[h11,1],[-1,h22]]` | h11、h22 |

## 需求拆解

1. 参数校验和指针位置判定；
2. 全 Host 路径的 Netlib 等价标量实现；
3. 全 Device 路径的单 block AIV Kernel；
4. `flag=-2/-1/0/1` 全分支；
5. GAM、GAMSQ、RGAMSQ 缩放保护；
6. NaN/Inf 边界的可终止行为；
7. CSV 驱动的 Host/Device 双路径正确性测试；
8. A2 三个正式性能场景和 A3 功能回归测试。

# 详细设计（required）

## 算子分析

### 数学定义

算子构造 H，使：

```text
H^T * diag(d1, d2) * H = diag(d1_new, d2_new)
H * [x1, y1]^T = [x1_new, 0]^T
```

### 核心分支

1. `d1 < 0`：输出 `flag=-1`，H 元素及 `d1/d2/x1` 置零。
2. `d2*y1 == 0`：输出 `flag=-2`，`d1/d2/x1` 保持不变。
3. `abs(d1*x1^2) > abs(d2*y1^2)`：计算 h21、h12 和 `su`；当
   `su > 0` 时输出 `flag=0`，否则转入全零分支。
4. 其余情况：若 `d2*y1^2 < 0` 则全零；否则计算 h11、h22，输出
   `flag=1`。
5. 对更新后的 `d1` 和 `abs(d2)` 执行 Netlib 缩放保护。

缩放常量：

```text
GAM    = 4096
GAMSQ  = 1.67772e7
RGAMSQ = 5.96046e-8
```

### 非有限输入

Netlib 循环在 NaN/Inf 输入下可能无法终止。Host 与 Device 路径在缩放循环前统一
检测非有限值，返回单位变换编码 `flag=-2`，并保持 `d1/d2/x1` 不变。该策略保证
边界测试可终止且两条路径行为一致。

## Host 侧设计

### 参数校验与分派

1. 检查 handle；
2. 检查五个数据指针；
3. 使用 `aclrtPointerGetAttributes` 查询各指针位置；
4. 根据全部 Host、全部 Device 或混合位置进行分派。

### 全 Host 路径

全 Host 指针调用 `SrotmgCpuCompute`。该函数使用 FP32 标量实现 Netlib 算法，
不创建 stream、不发射 Kernel，也不进行 Host/Device 拷贝。此路径是接口明确支持
的 Host 计算语义，不是 Device 路径的 CPU fallback。

### 全 Device 路径

全 Device 指针使用 handle 内部绑定的 `stream` 调用 `srotmg_kernel_do`。Host 不
读取 Device 标量、不做同步，也不把 Device 核心计算迁回 CPU。

### Tiling 与分核

算子固定处理四个输入标量和五个输出标量，不存在数据规模维度。采用：

- `numBlocks = 1`；
- AIV-only Kernel；
- 空 tiling 结构仅满足统一 launcher 接口；
- 不申请 workspace。

## Kernel 侧设计

Kernel 仅允许 block 0 执行：

1. 从 GM 读取 `d1/d2/x1/y1`；
2. 检测 NaN/Inf；
3. 执行 Netlib 主分支；
4. 对 `d1` 和 `d2` 执行最多 32 次缩放保护循环；
5. 按 flag 编码 `param[0..4]`；
6. 将更新后的 `d1/d2/x1` 写回 GM。

有效数据仅 36 B，Kernel 直接访问 GM 标量地址，避免为极小数据引入额外 UB
队列、DataCopy 流水和同步开销。

## 异步与生命周期

Device Kernel 在 handle 绑定的调用方 stream 上发射。API 内部不执行 Host 强制
同步；调用者只需在读取结果或计时边界同步相同 stream。

## 错误处理

| 场景 | 返回值 |
| --- | --- |
| handle 为空 | `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| 任一数据指针为空 | `ACLBLAS_STATUS_INVALID_VALUE` |
| 指针属性查询失败 | `ACLBLAS_STATUS_INVALID_VALUE` |
| 混合 Host/Device 指针 | `ACLBLAS_STATUS_INVALID_VALUE` |
| 全 Host 或全 Device 合法输入 | `ACLBLAS_STATUS_SUCCESS` |

## 内存设计

- `d1/d2/x1/y1`：4 × 4 B；
- `param`：5 × 4 B；
- 有效数据总量：36 B；
- 额外 workspace：0 B；
- Kernel 不申请动态内存。

## 支持硬件

| 支持的芯片/产品 | 架构目录 | 支持情况 |
| --- | --- | --- |
| Atlas 800I/T A2（Ascend910B） | Arch22 | 支持 |
| Atlas 800I A3（Ascend910C） | Arch22 | 支持 |

## 算子约束限制

1. 仅支持 FP32；
2. 五个数据指针必须全部位于 Host 或全部位于 Device；
3. `y1` 只读，`d1/d2/x1` 原地更新；
4. `param` 必须提供至少 5 个 FP32 元素；
5. 本算子不涉及广播、动态 shape、非连续 Tensor 或步长。

# 可维可测分析

## 测试方案

使用 ops-blas 的 GTest 与 CSV 驱动框架。每个成功用例同时验证全 Device、全 Host
路径以及 `d1/d2/x1/param[0..4]`。`param[0]` 的离散 flag 精确比较，其余 FP32
标量按仓库 mixed tolerance 比较。

用例覆盖：

1. `flag=-2/-1/0/1` 基础路径；
2. 正负值、零值和大小关系边界；
3. GAM 缩放保护和极端数量级；
4. NaN/Inf 可终止行为；
5. 空 handle 和五类空数据指针；
6. Host/Device 双路径。

## 精度标准

golden 使用任务书指定的 Netlib BLAS `cblas_srotmg`：

| 指标 | 要求 |
| --- | ---: |
| rtol | 2^-10 |
| atol | 2^-16 |
| required matched ratio | 0.99 |
| max absolute error | 1e-2 或 32 ULP |

## 性能测试方案

A2 性能测试在指定环境中执行。每个正式场景先 warmup 10 次，再采集不少于 50 个
有效 Kernel task time；统计时排除首次编译、进程启动和 Host 环境初始化，仅比较
Kernel 稳态耗时。

| Case | 语义 | 任务书阈值 (us) |
| ---: | --- | ---: |
| 1 | 一般值，flag=1 | 2.67 |
| 2 | flag=-2 快速返回 | 2.35 |
| 3 | GAM 缩放保护 | 2.61 |

性能使用 msprof 的 AIV task time 统计，以各场景任务书阈值作为判定标准。

## 验证执行计划

在任务书指定的 CANN 9.1.0 环境，分别在 A2 与 A3 上执行构建、官方 CSV 全量
回归、空指针错误码、Host/Device 指针模式、非有限输入、定向边界用例和三个 A2
正式性能场景。验证记录作为代码 PR 和正式验收材料的依据，不在本设计文档中展示
实测数据。

## 兼容性分析

本实现复用 `include/cann_ops_blas.h` 的既有统一接口，没有新增产品私有 API。
Arch22 文件与其他架构目录隔离，通过构建系统按产品选择，不改变已有产品线实现。

## 可维护性分析

1. Host 与 Device 路径使用同一分支结构和常量；
2. CSV 同时驱动 Host/Device 测试，便于发现路径漂移；
3. tiling 为空且无 shape 依赖，后续硬件适配边界清晰；
4. 代码、测试和性能证据均以统一接口和任务书场景为索引。

## 风险与后续计划

1. 在 CANN 9.1.0 指定环境完成 A2/A3 全量复验；
2. 跟踪 ops-blas 代码 PR 的最终 CI 与评审意见；
3. 若上游统一非有限输入语义发生变化，同步调整 Host、Device 和 golden；
4. 保持 A2/A3 共用 Arch22 实现，避免产品间语义分叉。
