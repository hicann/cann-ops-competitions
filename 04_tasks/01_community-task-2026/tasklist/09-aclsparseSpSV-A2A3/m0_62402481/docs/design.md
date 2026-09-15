# aclsparseSpSV A2/A3 算子设计

## 需求背景（required）

### 需求来源

本实现对应《aclsparseSpSV 算子开发(A2/A3)任务书》，目标是在 Atlas A2/A3
`arch22/DAV-2201` 上提供公开 Legacy API、Host 生命周期管理和 Ascend C 主计算路径。

### 背景介绍

SpSV 求解稀疏三角线性系统：

```text
op(A) * Y = alpha * X
```

`A` 为 CSR、CSC、COO 或 SLICED_ELL 格式的三角方阵，`X/Y` 为稠密向量。
接口必须支持 N/T/H、lower/upper、unit/nonunit、base 0/1、Host/Device alpha，
并保持 Analysis 到异步 Solve 完成期间的描述符和 workspace 生命周期。

## 需求分析（required）

### 需求描述

- 支持 `ACL_FLOAT` 和 `ACL_COMPLEX64`，Device 索引为 I32。
- 支持四种稀疏格式、未排序和重复坐标的确定性规范化。
- 支持 `BufferSize -> managed workspace -> Analysis -> Solve -> UpdateMatrix` 生命周期和
  X/Y 原地求解，同时保留 raw-pointer Analysis 兼容入口。
- Host 校验格式、维度、dtype、索引、pointer mode、workspace 和生命周期。
- 主求解由调用 handle 的 stream 异步下发到自有 Ascend C Kernel，不提供 CPU、
  framework、vendor whole-op 或其他后端 fallback。

### 需求拆解

1. 公开 ABI、opaque descriptor 和 managed workspace handle 在
   `include/cann_ops_sparse.h` 中闭环。
2. Host 保存 Analysis 快照，拒绝 op、属性、描述符、指针、workspace owner 或
   generation 漂移。
3. Analysis 在 Device 侧校验并规范化稀疏结构。
4. Solve 按三角依赖顺序执行递推，并覆盖 FP32/complex64 与 N/T/H。
5. GENERAL 更新替换全部 values；DIAGONAL 更新绑定逻辑行序的 `m` 个对角值。
6. C++ 合同、200 条精度、12 个性能坐标和内存门槛分别提供证据。

## 详细设计（required）

### 算子分析

#### 数学语义

对 lower 路径按行递增，对 upper 路径按行递减。每行计算：

```text
rhs = alpha * X[row] - sum(A[row, dependency] * Y[dependency])
Y[row] = rhs / diagonal
```

unit diagonal 视为 1；nonunit 对重复对角按稳定输入顺序累加。N/T/H 先形成有效
行依赖关系，H 路径同时对 complex64 稀疏值执行共轭。缺失或零 nonunit 对角通过
IEEE-754 除法传播 INF/NAN。

#### 支持边界

| 维度 | 支持范围 |
| --- | --- |
| 硬件 | Atlas A2/A3，DAV-2201 |
| 稀疏格式 | CSR、CSC、COO、SLICED_ELL |
| value/compute dtype | `ACL_FLOAT`、`ACL_COMPLEX64` |
| 索引 | Device I32，base 0/1 |
| 操作 | N、T、H |
| 三角属性 | lower/upper，unit/nonunit |
| pointer mode | Host/Device alpha |
| 输出 | 独立 Y 或 X/Y 同一 Device values 指针 |

矩阵必须为二维方阵，X/Y 长度和 dtype 必须与矩阵、computeType 一致。运行时分派
只使用公开 API 元数据，不依赖测试编号、公开 shape 或输入值分布。

### Host 侧设计

Host 层完成以下工作：

1. 校验 handle、enum、矩阵/向量 descriptor、I32 索引、dtype、方阵约束、Device
   pointer 和 alpha 位置。
2. 根据格式、op 和数值叶计算 512-byte 对齐的 workspace 布局。managed workspace
   由库分配 Device 内存，并登记调用方请求的字节数和单调 generation。
3. Analysis 保存 matrix/vector/alpha/pointer-mode/workspace owner/generation 快照并下发结构 Kernel；
   同步读取 status 以把非法 Device 索引映射为确定错误码。
4. Solve 比较当前参数与 Analysis 快照，仅在完全一致时异步下发求解 Kernel。
5. UpdateMatrix 只更新当前 values 所有者；DIAGONAL 的输入按逻辑行序绑定。

`aclsparseSpSV_analysisWithWorkspace` 使用库内 live registry 校验 requested bytes、owner
和 generation，不依赖 runtime 的 page-aligned allocation range。Solve 再次查询 registry；
handle 已销毁或同一 Host 地址被新 generation 复用时均拒绝。原 raw-pointer Analysis
入口保留 null/alignment 与兼容行为，但不承诺观察调用方任意 `aclrtFree` 事件。

### Analysis 与格式规范化

| 条件 | 结构路径 | workspace 数据 |
| --- | --- | --- |
| CSR + N | compressed-major 直通 | status |
| CSC + T/H | compressed-major 直通 | status |
| 其他 CSR/CSC | 两遍稳定散射 | rowPtr、sourceMap、dependency、cursor |
| COO | 两遍稳定散射 | rowPtr、sourceMap、dependency、cursor |
| SLICED_ELL | slice/slot 稳定遍历 | rowPtr、sourceMap、dependency、cursor |

第一遍计数，前缀和得到规范化 rowPtr，第二遍按输入遍历顺序写 sourceMap，因此未排序
和重复坐标具有确定性。SLICED_ELL padding 仅在描述符规定的编码位置被忽略。

### Kernel 侧设计

Analysis 和 Solve 均为自有 AIV Kernel。Solve 使用单个逻辑 block 串行推进三角依赖，
避免跨核依赖发布和 Host 中间结果。非直通格式由 Analysis 为每个规范化非零写入一次
dependency；Solve 直接读取该 owner，删除逐非零二分 row pointer 的 GM 读/比较闭环。
每行再从 GM 读取稀疏值及已完成的 Y/state，执行乘减、对角处理后写回 Y。

普通叶使用 FP32/complex64 递推。`complex64 + H + unit` 叶因长递推的逐行 complex64
舍入会超过任务书绝对误差上限，使用四个 FP32 分量保存 real/imag high/low，采用
TwoSum 与 Dekker TwoProduct 维护 double-double 递归状态，最后舍入到 complex64 输出。
该 guard 只依赖 dtype/op/diag 元数据。

### Workspace 设计

`spsv_arch22.h` 的 constexpr ledger 统一计算以下互斥区域：

```text
status header
row pointer
source-position map
dependency map
construction cursor
double-double complex state
```

所有区域按 512 bytes 对齐；直通 compressed-major 叶的结构区只保留 status。高精度叶
另为每行分配四个 FP32，普通叶不分配该区域。复制的公共 `resource_ledger.hpp` 与本算子 constexpr
布局共同校验 owner 的 allocation、offset、lifetime 和末元素访问；one-past 负例必须失败。
managed workspace 合同另以正例和 short/live/generation/owner 负例静态绑定，运行时 Host
复用同一 predicate。

### 生命周期与异步语义

BufferSize 和 Analysis 允许 vecX/vecY descriptor 为 NULL；Solve 必须提供有效
descriptor 和 Device values。推荐调用方以 `aclsparseSpSV_createWorkspace` 分配并通过
`aclsparseSpSV_analysisWithWorkspace` 绑定，再以 `aclsparseSpSV_destroyWorkspace` 释放。
Analysis 为完成 Device 侧结构校验会同步一次；Solve 只在调用 handle stream 上排队并
立即返回。调用方必须等 stream 完成后才销毁 managed workspace、输入、输出或 descriptor。

### 优化策略

- CSR-N 与 CSC-T/H 免去结构重排，减少 workspace 和 Analysis 流量。
- sourceMap/dependencyMap 在 Analysis 中一次构造，Solve 不重复恢复格式坐标；该闭区间
  替换使 P-02/P-03 非直通叶从逐非零约 `log2(m)` 次 row-offset GM 查询降为一次依赖读取。
- 精度增强仅作用于 task-checker 证明需要的 metadata 叶，避免扩大 P-03 H/nonunit
  性能路径的工作集。
- 当前三角依赖为串行关键路径；性能判断以完整 `UpdateMatrix + Solve` 调用范围为准，
  不把 Kernel 单段或局部样例替代 12 坐标验收。

## 可维可测分析

### 精度标准

FP32 使用 float64 Golden，complex64 使用 complex128 Golden。每个实数分量执行
`rtol=2^-10`、`atol=2^-16`，匹配率不低于 0.99，并满足
`abs_error <= max(1e-2, 32 * ULP(golden))`；同时校验 INF/NAN、无穷符号和重复执行
bitwise deterministic。

### 性能标准

每个 case/dtype/base 坐标采用 10 次预热、30 次采样，中位数指标为：

```text
native cuSPARSE GPU Event median_us / NPU update_plus_solve median_us >= 0.25x
```

坐标必须逐项判定，不允许跨 case 平均。workspace 记录在同一调用协议下。

### 兼容性分析

新增 arch22 路径与公共 Host/descriptor 能力，不以 arch35 实现作为运行时 fallback。
公开 ABI、descriptor 属性和格式映射保持在共享头文件/公共 Host 层闭环。按当前用户指定
硬件范围，实现在本机 Ascend 910C 上以 CANN 9.1、`ascend910_9382`、DAV-2201 编译和验证。

## 验收结论

当前可复现结果见 [self_validation.md](self_validation.md)。managed workspace 使本机
无需 allocation-range 能力即可验证 caller-requested extent 与 generation/liveness；当前
Release 源已通过 34/34 C++ 合同、200/200 精度、12/12 性能和 12/12 内存。
