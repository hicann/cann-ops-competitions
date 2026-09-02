# 需求背景（required）

## 需求来源

社区任务：**08-45｜8月社区任务-aclblasSrotmg算子开发（A2/A3）**。

任务目标是在 Atlas A2/A3 系列产品对应的 arch22 路径中，基于 Ascend C 完成 `aclblasSrotmg` 单精度实数修正 Givens 旋转参数构造算子的适配开发，并完成精度、性能及异常场景自验，最终代码提交至 `ops-blas` 开源仓。

接口沿用 `ops-blas` 现有公共接口：

```cpp
aclblasStatus_t aclblasSrotmg(
    aclblasHandle_t handle,
    float *d1,
    float *d2,
    float *x1,
    const float *y1,
    float *param);
```

## 背景介绍

### aclblasSrotmg算子功能

`aclblasSrotmg` 用于构造单精度实数 modified Givens rotation（修正 Givens 旋转）参数。算子根据输入 `d1`、`d2`、`x1`、`y1` 计算旋转参数矩阵 H，并原地更新 `d1`、`d2`、`x1`，同时将旋转参数写入 `param[0..4]`。

该算子属于 BLAS Level-1 标量类操作，计算规模很小，但包含较多条件分支以及 GAM 缩放逻辑，对特殊浮点值、边界值和接口语义的一致性要求较高。

### 现状分析

当前 `ops-blas` 已存在 `arch35` 的 `srotmg` 实现，可作为算法语义、Host/Device 路径与测试方式的参考；本任务新增 `arch22` 实现，以补齐 Atlas A2/A3 产品支持。

计划新增目录：

```text
blas/rotmg/arch22/
├── srotmg_host.cpp
├── srotmg_kernel.cpp
└── srotmg_tiling_data.h
```

测试目录：

```text
test/rotmg/srotmg/arch22/
├── srotmg_npu_wrapper.h
├── srotmg_test.cpp
└── srotmg_test.csv
```

### 接口参数分析

| 参数 | 含义 | 数据类型 | 输入/输出 |
| --- | --- | --- | --- |
| `handle` | aclBLAS 上下文句柄 | handle | 输入 |
| `d1` | 第一缩放因子 | float32 scalar | 输入输出 |
| `d2` | 第二缩放因子 | float32 scalar | 输入输出 |
| `x1` | 第一向量分量 | float32 scalar | 输入输出 |
| `y1` | 第二向量分量 | float32 scalar | 输入 |
| `param` | modified Givens 参数，长度为 5 | float32[5] | 输出 |

算子支持标量地址位于 Host 或 Device。Device 路径由 NPU Kernel 执行，Host 路径在 CPU 侧完成同语义计算；不支持一组参数中 Host/Device 地址混用。

# 需求分析（required）

## 需求描述

在 Atlas A2/A3 对应的 `arch22` 路径中实现 `aclblasSrotmg`，算法行为与 Netlib BLAS `srotmg` 及仓内既有实现保持一致，满足任务包给出的功能、精度、性能和异常场景要求。

主要能力包括：

1. 支持 float32 标量输入输出；
2. 支持 Host 内存路径与 Device 内存路径；
3. 覆盖 `flag = -2/-1/0/1` 全部分支；
4. 支持 `GAM = 4096` 的缩放保护逻辑；
5. 覆盖零值、负值、极大/极小值、Inf/NaN 与空指针等场景；
6. 精度满足任务规定的 FLOAT32 mixed tolerance 标准；
7. 性能按照任务包 `verify_performance.py` 与 `gpu_baseline.csv` 的验收方式验证；
8. 不修改公共 API 签名，不影响现有 `arch35` 实现。

## 需求拆解

1. 新增 `blas/rotmg/arch22` Host、Kernel 与 TilingData 实现；
2. Host 侧完成句柄、数据指针和内存位置合法性检查；
3. Host 数据路径直接执行 CPU modified Givens 算法；
4. Device 数据路径将标量地址写入 TilingData，并启动单个 AIV block；
5. Kernel 侧实现主分支计算与 GAM scaling；
6. 新增 `test/rotmg/srotmg/arch22` 测试路径；
7. 接入任务包 1200 条 CSV 用例，其中 1000 条为功能/精度类用例，200 条为性能类用例；
8. 对 Inf/NaN 特殊场景单独验证，确保接口无 crash、无无穷循环；
9. 最终按任务要求完成 CANN 9.1.0 与目标 A2/A3 环境回归。

# 详细设计（required）

## 算子分析

### 数学公式

`SROTMG` 根据输入 `d1`、`d2`、`x1`、`y1` 构造 modified Givens transformation：

```text
H = [[h11, h12],
     [h21, h22]]
```

并原地更新：

```text
d1
d2
x1
param[0..4]
```

`param[0]` 为 `flag`，不同 flag 对应的有效矩阵元素如下：

| flag | H 的形式 | param 有效字段 |
| --- | --- | --- |
| -1 | `[[h11,h12],[h21,h22]]` | `param[1..4]` |
| 0 | `[[1,h12],[h21,1]]` | `param[2]`,`param[3]` |
| 1 | `[[h11,1],[-1,h22]]` | `param[1]`,`param[4]` |
| -2 | 恒等矩阵 | 仅 `param[0]` |

主分支按照 Netlib `srotmg` 语义实现：

1. `d1 < 0`：进入 `flag = -1` 路径，`d1/d2/x1` 清零；
2. `d1 >= 0` 且 `d2 * y1 == 0`：进入 `flag = -2` 恒等路径；
3. 当 `|d1*x1^2| > |d2*y1^2|` 时进入 `flag = 0` 路径；
4. 其他有效情况进入 `flag = 1` 路径；
5. 部分条件不满足时退化到 `flag = -1`；
6. 主分支结束后，对 `d1` 与 `|d2|` 执行 GAM scaling，并同步修正 `x1` 与 H 中相关元素。

缩放常量：

```text
GAM    = 4096
GAMSQ  = GAM * GAM
RGAMSQ = 1 / GAMSQ
```

### 支持数据类型

仅支持：

```text
float32
```

### 支持形状

本算子为标量运算：

```text
d1    : scalar
 d2   : scalar
x1    : scalar
y1    : scalar
param : 5 个 float32 元素
```

不涉及广播、矩阵维度、batch、stride 或动态 shape 推导。

## 算子实现

### 实现方案

#### host侧设计

##### 1. 参数检查

检查：

```text
handle
d1
d2
x1
y1
param
```

`handle` 为空时返回句柄对应错误码；数据指针为空时按照现有 aclBLAS 接口约定返回参数错误码。

##### 2. 内存位置检查

判断 `d1/d2/x1/y1/param` 的内存位置：

- 全部为 Host：进入 Host 计算路径；
- 全部为 Device：进入 Device Kernel 路径；
- Host/Device 混合：返回非法参数。

##### 3. Host计算路径

Host 数据直接执行 CPU modified Givens 算法：

```text
Load scalars
  -> 主分支计算
  -> d1 scaling
  -> d2 scaling
  -> Store d1/d2/x1/param
```

Host 路径无需设备内存申请、数据搬运和 Kernel 启动。

##### 4. Device路径与Tiling策略

由于算子只处理 4 个输入标量和 5 个结果参数，不存在按数据量切块的必要，因此不设计传统 shape tiling。

TilingData 仅传递五个 Device 地址：

```cpp
struct SrotmgTilingData {
    uint64_t d1;
    uint64_t d2;
    uint64_t x1;
    uint64_t y1;
    uint64_t param;
};
```

Host 将地址写入 tiling 后启动 Kernel。

##### 5. 分核策略

固定：

```text
blockDim = 1
Kernel type = AIV
```

原因是算子只处理一组标量，多核并行无法获得有效收益，反而会增加启动与同步开销。

##### 6. TilingKey规划

不使用 TilingKey。

本算子仅支持 float32，且没有 shape、广播或多规格分支；主算法分支完全由运行时标量值决定，无需通过 TilingKey 区分 Kernel 模板。

#### kernel侧设计

Kernel 使用 `GlobalTensor<float>` 直接访问 Device 标量地址。

整体流程：

```text
初始化 GlobalTensor
      ↓
读取 d1/d2/x1/y1
      ↓
modified Givens 主分支计算
      ↓
GAM scaling
      ↓
写回 d1/d2/x1/param
```

##### 1. 数据访问

由于数据规模只有数个标量，不建立 GM→UB→GM 的批量搬运流水，直接使用标量读写接口完成访问：

```cpp
GlobalTensor<float>::GetValue(...)
GlobalTensor<float>::SetValue(...)
```

这样可以避免为极小数据量引入额外 Local Memory 搬运和同步开销。

##### 2. 主计算逻辑

Kernel 按照 Netlib `srotmg` 的控制流实现：

- `d1 < 0` 路径；
- `d2 * y1 == 0` 路径；
- `sq1 > |sq2|` 路径；
- 另一主分支路径；
- `su <= 0` 退化路径；
- `d1/d2` GAM scaling。

##### 3. GAM缩放

对正常有限值，严格保持既有算法的计算顺序：

- 过小值乘 `GAM^2`；
- 过大值除 `GAM^2`；
- 同步调整 `x1` 与 H 参数。

### 性能优化方案

`aclblasSrotmg` 的计算量极小，性能主要由接口调度、Kernel 启动和少量 GM 标量访存决定，因此优化重点不是多核吞吐，而是减少固定开销：

1. **单核执行**：固定 `blockDim = 1`，避免无意义的多核调度与同步；
2. **直接标量访存**：Device 侧直接读写 GlobalTensor 标量，不构造大块 LocalTensor，也不建立多阶段搬运流水；
3. **最小化 TilingData**：仅传递 5 个地址，不传递冗余 shape/stride 信息；
4. **Host 数据零设备开销**：输入位于 Host 时直接 CPU 计算，不启动 NPU Kernel；
5. **单次 Kernel 完成全部计算**：主分支、scaling 和结果写回全部在一次 Kernel 中完成，不拆分为多个 Kernel；
6. **避免无效分核/向量化**：标量计算没有可利用的数据级并行，保持顺序控制流可减少额外指令和资源开销。

### 特殊浮点风险与待确认方案

任务包包含 Inf/NaN 专项用例。前期可行性验证发现：

- `x1=+Inf`、`y1=+Inf`、`d1=-Inf` 及 NaN 类输入能够返回，不会触发无穷 scaling；
- 当 `d1=+Inf` 或 `d2=+Inf` 时，标准 Netlib-style scaling 可能出现“数值无进展”问题，例如：

```text
+Inf / (GAM * GAM) = +Inf
```

如果仅依赖：

```cpp
while (sd1 >= GAMSQ) {
    sd1 = sd1 / (GAM * GAM);
}
```

循环条件可能永久成立。

当前候选设计是在每次缩放前计算下一步值：

```text
nextSd1 = scaled(sd1)
nextSd2 = scaled(sd2)
```

若 `nextSd1 == sd1` 或 `nextSd2 == sd2`，则认为该非有限值缩放过程已无数值进展，终止对应循环，确保 Host/Device 接口不发生死循环。

该保护仅在“缩放后数值与缩放前完全相同”时触发，对正常 finite 输入不改变算法路径和计算顺序。

**待确认项**：任务包使用 CBLAS 作为 golden，但前期环境中的 `cblas_srotmg` 在 `d1=+Inf`、`d2=+Inf` 两类输入上同样出现不返回现象；相关 FL 用例及带 Inf 的性能用例最终 golden/验收口径已向任务方反馈，最终实现以评审/任务方确认结果为准。

## 支持硬件

| 支持硬件 | 支持情况 |
| --- | --- |
| Atlas A2 系列（arch22） | 支持 |
| Atlas A3 系列（arch22） | 支持 |

前期可行性验证设备：

```text
Ascend 910B4
```

最终按任务要求在对应 A2/A3 环境及指定软件版本完成验收。

## 算子约束限制

1. 仅支持 float32；
2. 仅支持单组标量输入；
3. `d1/d2/x1/y1/param` 需统一位于 Host 或 Device；
4. 不支持 Host/Device 混合地址；
5. 不涉及广播、stride、batch 或 shape 推导；
6. Device 路径固定单 AIV block；
7. 最终验收 CANN 版本按任务要求使用 9.1.0；前期可行性验证环境为 CANN 9.0.0。

# 可维可测分析

## 精度标准/性能标准

### 精度标准

FLOAT32 按任务包/仓库现有 mixed tolerance 方式进行精度判定，包含：

```text
rtol = 2^-10
atol = 2^-16
required matched ratio >= 0.99
per-element absolute threshold = max(1e-2, 32 * ULP(golden))
```

`param[0]` 的 flag 按离散控制值精确比对，其余数值按 FLOAT32 精度规则校验。

Golden 参考使用任务包指定的 `cblas_srotmg`。

### 性能标准

性能按照任务包提供的：

```text
verify_performance.py
gpu_baseline.csv
```

进行验证，以任务包脚本实际判定结果作为验收依据。

性能测试覆盖任务包 PF 类用例，并在任务指定 CANN 9.1.0 / 目标 A2/A3 环境完成最终测试。

### 测试用例设计

任务随附 CSV 共 1200 条：

| 类别 | 数量 | 主要覆盖内容 |
| --- | ---: | --- |
| L0 | 12 | `flag=-2/-1/0/1` 基础路径 |
| BR | 14 | 控制流分支与边界组合 |
| SC | 18 | GAM scaling 大数/小数 |
| FL | 9 | Inf/NaN 特殊浮点 |
| ED | 6 | 空指针等异常参数 |
| EX | 941 | 扩展随机与数量级组合 |
| PF | 200 | 性能测试 |
| 合计 | 1200 | 1000 功能/精度 + 200 性能 |

Host 与 Device 两条数据路径均纳入功能/精度覆盖。

### 前期可行性验证

在设计阶段进行了实现可行性验证，环境为：

```text
Device : Ascend 910B4
CANN   : 9.0.0
SOC    : ascend910b / dav-2201 / arch22
```

验证结果：

1. arch22 Host/Kernel 原型可成功编译；
2. Device Kernel 可在 910B4 上正常启动并完成标量读写；
3. L0、BR、SC、ED、EX 以及不触发 CBLAS 卡死的 FL 特殊用例已完成 Host + Device 验证；
4. 排除 PF 以及当前 CBLAS golden 会卡死的 FL045/FL046 后，安全精度回归 `1997/1997` GTest 通过；
5. 其中 EX 扩展用例 `941 Device + 941 Host = 1882/1882` 通过；
6. 空指针相关负向场景已覆盖。

上述结果仅作为设计可行性依据，不替代最终 CANN 9.1.0 / A2/A3 验收。

### 正式开发阶段验证计划

1. 根据任务方回复确定 `d1=+Inf`、`d2=+Inf` 的最终语义与 golden 口径；
2. 完成特殊浮点保护方案的代码评审与全量回归；
3. 完成全部 1000 条功能/精度用例；
4. 完成全部 200 条 PF 性能用例；
5. 在任务指定 CANN 9.1.0 与目标 A2/A3 环境完成最终回归；
6. 输出自测日志、精度结果、性能结果及验收所需材料。

## 兼容性分析

本任务仅新增 `arch22` 产品实现路径，不修改公共 API 签名和既有 `arch35` 算法实现。

兼容性策略：

1. `arch35` 现有实现保持不变；
2. API 声明和上层调用方式保持不变；
3. Host/Device 参数语义保持与现有接口一致；
4. 新增 arch22 测试目录与实现目录独立，不影响其他产品路径；
5. 构建系统根据 arch 目录选择对应实现。

因此本任务不会改变现有 `arch35` 产品行为，对上层调用接口无兼容性影响。
