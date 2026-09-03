# aclblasSrotmg（Atlas A2/A3）算子设计文档

> 本文按 CANN 社区任务 2026 `design_template.md` 的必选章节编写。
> 设计模板：https://gitcode.com/cann/cann-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md

# 需求背景（required）

## 需求来源

本任务来源于 CANN 社区任务“8月社区任务-aclblasSrotmg算子开发（A2/A3）”。目标是在 ops-blas 仓库中为 Atlas A2/A3（arch22）补充单精度修正 Givens 旋转参数构造接口 `aclblasSrotmg`，并与仓内现有 arch35 产品线共用 `include/cann_ops_blas.h` 中的公开接口声明。

## 背景介绍

`srotmg` 是 BLAS Level 1 标量算子，用于构造 modified Givens 变换参数。输入为四个 float32 标量 `d1`、`d2`、`x1`、`y1`，其中 `d1`、`d2`、`x1` 原地更新，`y1` 只读，输出 `param[5]` 记录 flag 和变换矩阵的定义元素。

本实现参考 Netlib `srotmg` 算法和 ops-blas arch35 既有实现，并补充 Atlas A2/A3 的 Host/Device 双路径：全部 Host 指针时在 CPU 上执行；全部 Device 指针时通过 handle 绑定的 stream 启动一个 Ascend C kernel；混合 Host/Device 指针时返回参数错误。

Netlib 原始缩放循环在非有限中间值下可能无法终止，例如 `Inf / GAM²` 仍为 `Inf`。为保证接口终止性，arch22 使用 16 次有界缩放循环。对于中间值始终有限的输入，该上限大于实测最大迭代次数；对于 Netlib 无法终止的输入，本实现定义为可终止扩展语义。

# 需求分析（required）

## 需求描述

使用 Ascend C 在 Atlas A2/A3 上实现 `aclblasSrotmg`，满足下列要求：

1. 复用公共 `aclblasSrotmg` 接口，不新增产品私有接口。
2. 支持 float32 标量输入，覆盖 Host 和 Device 两类同侧指针。
3. 混合 Host/Device 指针、空指针和空 handle 返回约定错误码。
4. 正常输入与 cblas/Netlib golden 满足 FLOAT32 精度标准。
5. `y1` 保持只读；`param` 仅写当前 flag 定义的槽位。
6. 缩放保护路径可终止，并覆盖有限极值、Inf/NaN 和有限输入产生非有限中间结果的场景。
7. 在 Atlas 800T A2（910B3）上完成 warmup 后超过 50 次有效采样，三条任务书性能用例分别不高于 2.67 us、2.35 us、2.61 us。

## 需求拆解

1. 在 `blas/rotmg/arch22/` 新增 Host 分发、Device kernel 和空 tiling 数据结构。
2. 在 `test/rotmg/srotmg/arch22/` 新增 CSV 驱动精度、负向、终止性和性能测试。
3. 扩展公共测试参数结构，支持 CSV 空指针编码和性能 `iters`。
4. 生成 1000 条精度用例和 200 条性能用例；过滤 5 条 golden 无法终止的 CSV 用例后，实际执行 998 条精度和 197 条性能用例。
5. 提供精度、性能和缩放边界验收脚本，并对缺失、重复、异常退出和元数据错误进行门禁。
6. 输出可复现的测试 README、原始日志和源码哈希清单。

# 详细设计（required）

## 算子分析

### 数学公式

算法使用以下中间量：

```text
sp1 = d1 * x1
sp2 = d2 * y1
sq1 = sp1 * x1
sq2 = sp2 * y1
```

主要分支如下：

- `d1 < 0`：安全归零，`flag = -1`。
- `d2 * y1 == 0`：恒等变换，`flag = -2`。
- `abs(sq1) > abs(sq2)`：以 d1 路径计算，正常情况下 `flag = 0`；若 `su <= 0` 则安全归零。
- 其余情况：以 d2 路径计算，正常情况下 `flag = 1`；若 `sq2 < 0` 则安全归零。

缩放常量与 Netlib 保持一致：

```text
GAM    = 4096
GAMSQ  = 1.67772e7
RGAMSQ = 5.96046e-8
```

当更新后的 `d1` 或 `abs(d2)` 位于 `[RGAMSQ, GAMSQ]` 之外时执行缩放。arch22 将每个缩放循环限制为最多 16 次；有限中间值角落网格实测单循环最多 6 次，因此保留充足余量。

`param` 存储规则：

| flag | H 形式 | 写入槽位 |
|---|---|---|
| -2 | 单位矩阵 | 仅 `param[0]` |
| -1 | 四元素全部显式存储 | `param[0..4]` |
| 0 | `h11=1, h22=1` | `param[0]`、`param[2]`、`param[3]` |
| 1 | `h12=1, h21=-1` | `param[0]`、`param[1]`、`param[4]` |

未定义槽位保持调用方原值。测试以 `123.456f` 哨兵验证 Host、Device 和 golden 三侧行为。

### 支持数据类型

| 参数 | 类型 | 输入/输出 |
|---|---|---|
| d1 | float32 标量 | 输入/输出 |
| d2 | float32 标量 | 输入/输出 |
| x1 | float32 标量 | 输入/输出 |
| y1 | float32 标量 | 只读输入 |
| param | float32[5] | 输出及未定义槽位保持 |

### 支持形状

本算子为纯标量接口，不含 shape、维度、步长、广播、前导维或 tiling 尺寸轴。所有数据指针必须全部位于 Host 或全部位于 Device。

## 算子实现

### 实现方案

#### Host 侧设计

1. `ValidateSrotmgParams` 首先校验 handle 和五个数据指针。
2. 对五个指针分别调用 `aclrtPointerGetAttributes`，判定其内存位置。
3. 全部 Host：调用 `SrotmgCpuCompute` 直接进行标量计算。
4. 全部 Device：从 handle 取得 stream，以一个 block 启动 `srotmg_kernel`。
5. 混合指针：返回 `ACLBLAS_STATUS_INVALID_VALUE`。
6. Host 算法与 Device 算法保持逐分支对应，包括有界缩放循环和 param 条件写回。

本算子没有需要由 Host 计算的 shape/tiling 信息，因此 `SrotmgTilingData` 为空结构，仅用于保持 launcher 和仓库构建框架一致。

#### Kernel 侧设计

1. kernel 类型为 `KERNEL_TYPE_AIV_ONLY`，仅 block 0 执行。
2. 使用 `GlobalTensor<float>::GetValue/SetValue` 直接读取和写回 GM 标量，避免 DataCopy/UB 流水对纯标量任务带来的额外开销。
3. 在局部 float 变量中完成 Netlib 分支、缩放保护和 param 构造。
4. `y1` 只读取、不写回。
5. `flag=-2/0/1` 仅写相应定义槽位，`flag=-1` 写全部槽位。
6. 缩放循环采用 `SCALE_MAX_ITER=16`，保证非有限路径终止。

### 测试设计

- `CsvDrivenDevice`：998 条精度记录，覆盖 Device 路径。
- `CsvDrivenHost`：998 条精度记录，覆盖 Host pinned 路径。
- `CsvDrivenHostPlain`：993 条普通 Host 路径，跳过 5 条空指针用例。
- 4 个普通 `TEST_F`：空 handle、混合指针、Inf 终止性、有限输入溢出终止性。
- `CsvDrivenPerf`：197 条性能用例，warmup 10 次，外层采样 60 次，每次调用使用独立输入 record。
- `verify_scale_boundary.py`：18 条缩放边界断言及 707281 组有限极值角落网格。

精度比较采用 `atol=2^-16`、`rtol=2^-10`；flag 精确比较；y1 和终止性用例的有限输出使用 float32 位模式比较。NaN 按 `isnan`，Inf 同时检查类别和符号。

## 支持硬件

| 支持的芯片版本 | 支持情况 |
|---|---|
| Atlas 800I A2 / Atlas 800I A3（arch22，910B3） | 支持 |
| arch35 | 使用仓内既有实现，本任务不修改其生产代码 |

## 算子约束限制

1. 仅支持 float32 标量接口。
2. 五个数据指针必须全部为 Host 或全部为 Device，混合内存位置不支持。
3. 本算子不涉及 shape、广播、步长、前导维和多核数据切分。
4. 非有限扩展语义目前由 Netlib 等价 float32 模型和可终止的 cblas 抽样用例支撑；由于当前环境无 GPU，尚无 cuBLAS 非有限输入实测证据。
5. 性能门限以 Atlas 800T A2（910B3）上的 msprof 纯 kernel 时间判定；Host API 与 stream 区间仅作诊断。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
|---|---|---|
| 精度标准 | FLOAT32：`abs(actual-golden) <= atol + rtol*abs(golden)`，其中 `atol=2^-16`、`rtol=2^-10`；flag 精确相等 | 任务书 §3.2、生态算子开源精度标准 |
| 用例规模 | 998 条有效精度用例，Device/HostPinned 全量执行，HostPlain 执行 993 条；另有 4 个 TEST_F | 任务书 §3.5、自测用例 README |
| 边界一致性 | 18 条 SC 断言；707281 组角落网格中 A/B 类与等价参考模型逐 bit 相等，C 类保证 16 次内终止 | `verify_scale_boundary.py` |
| 性能标准 | warmup 后有效采样大于 50 次；TC_PF_1001/1002/1003 分别不高于 2.67/2.35/2.61 us | 任务书 §3.3 |
| 性能完整性 | 197 条 TC_PF 必须全部产出且无缺失、额外、重复和元数据错误 | `verify_performance.py` |

最终 v7 在 910B3 上取得 2993 PASS / 0 FAIL / 1003 SKIP。`SrotmgParam` 使用
`std::strtof` 解析输入，仅在语法错误时回退默认值；即使 C 运行库为 float32 次正规
数设置 `ERANGE`，返回的可表示数值仍会保留。因此 `TC_FL_054~057` 的
`y1=2^-149` 已按设计真实执行，Device、HostPinned 和 HostPlain 均通过。

三条任务书性能 case 分别独立运行 msprof，排除 64 次 warmup 后取最后 60 个
`srotmg_kernel` 样本平均。实测 1.733 / 1.413 / 1.729 us，均低于
2.67 / 2.35 / 2.61 us 门限；GTest 输出的 API/stream 时间不参与硬门禁。

## 兼容性分析

1. 复用公共 `aclblasSrotmg` 声明，不改变接口签名。
2. Host/Device 分发与 arch35 既有模式一致。
3. `srotmg_param.h` 新增字段均有默认值，对 arch35 CSV 解析路径保持兼容。
4. `blas/rotmg/README.md` 仅将 Atlas A2/A3 产品支持状态更新为“支持”。
5. arch22 对 param 未定义槽位采用 BLAS 语义保持原值；与旧 arch35 清零行为存在有意差异，已由哨兵用例覆盖。
6. arch22 对 Netlib 无法终止的非有限中间值采用有界扩展语义；普通有限路径保持一致。

## 风险与后续验证

1. 任务书模式只对三条文档门限硬判；其余 194 条性能 case 因 GPU 基线为空而标记 `NO_REF`，仅用于可执行性、覆盖面与元数据完整性检查。
2. Device 同步失败时测试进程会保留可能仍在使用的设备缓冲并返回错误，避免异常路径上的释放后使用；该路径会产生进程级临时泄漏，进程退出后由运行时回收。
3. workspaceSize 为 0；报告中的 91.55 MiB 峰值是最大测试批次输入缓冲估算，不是算子 workspace。
4. 若任务方要求非有限输入严格对齐真实 cuBLAS 二进制，仍需在 GPU 环境补充对账；当前依据为 cblas/Netlib 等价语义和 910B3 实测。
