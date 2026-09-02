# aclblasSrotmg 算子设计文档

| 项目 | 内容 |
|---|---|
| 算子 | aclblasSrotmg |
| 目标产品 | Atlas A2/A3，arch22 |
| 数据类型 | FLOAT32 |
| 软件基线 | CANN 9.1.0 |
| 参考语义 | Netlib SROTMG、cuBLAS cublasSrotmg |
| 验收提交 | b2d8c82f2cf7e808e4d6b666644850037bb252fc |
| 实测平台 | Ascend 910B3 |

## 1. 需求背景（required）

### 1.1 需求来源

本需求来自 CANN 2026 社区算子任务。在昇腾 ops-blas 开源仓中补齐 aclblasSrotmg 面向
Atlas A2/A3 产品的 arch22 实现，使公共 BLAS 接口在 A2/A3 上具备与 Netlib SROTMG 和
cuBLAS cublasSrotmg 一致的功能、参数语义和异常行为。

### 1.2 背景介绍

SROTMG 用四个单精度标量 d1、d2、x1、y1 构造 modified Givens 旋转参数。输出通常直接
传给 ROTM，在 QR 分解、最小二乘和迭代线性代数流程中用于消去二维向量的第二个分量。

ops-blas 已存在 Ascend 950PR/950DT 对应的 arch35 实现和公共 aclblasSrotmg API，但任务
基线缺少 Atlas A2/A3 的 arch22 Host 与 Kernel 代码。A2/A3 调用方因此无法在保持同一 API
的前提下使用该算子。

### 1.3 aclblasSrotmg 算子实现与优化

本任务以现有公共接口和 arch35 代码组织为基础，使用 Ascend C 完成 arch22 适配，并针对
纯标量控制流算子优化固定调用开销：

1. 全 Host 指针直接执行 CPU 标量计算。
2. 全 Device 指针在 handle 绑定的 stream 上发射单个 AIV block。
3. Device 路径不做 Host/Device 数据搬运，不申请 workspace，不生成 tiling data。
4. flag=-2 和 d1<0 等分支提前返回。
5. 仅对 arch22 的 srotmg_kernel.cpp 启用 -O2，降低 Debug 构建下的标量控制流开销。

### 1.4 历史实现路径和相关 API 路径

| 内容 | 路径 |
|---|---|
| 公共 API 声明 | include/cann_ops_blas.h |
| 公共状态码和 handle 定义 | include/cann_ops_blas_common.h |
| 历史 arch35 Host 实现 | blas/rotmg/arch35/srotmg_host.cpp |
| 历史 arch35 Kernel 实现 | blas/rotmg/arch35/srotmg_kernel.cpp |
| 本任务 arch22 Host 实现 | blas/rotmg/arch22/srotmg_host.cpp |
| 本任务 arch22 Kernel 实现 | blas/rotmg/arch22/srotmg_kernel.cpp |
| 算子 README | blas/rotmg/README.md |
| Golden 包装 | test/rotmg/srotmg/srotmg_golden.h |
| arch22 测试 | test/rotmg/srotmg/arch22/ |
| Ascend C Kernel API | CANN include/kernel_operator.h |

### 1.5 历史实现现状分析

任务基线中，arch35 已具备同名 API 的 Host/Device 双路径。公共接口能力如下：

| 参数 | 参数含义 | 参数类型 | 数据类型 | 内存位置 | 形状/元素数 |
|---|---|---|---|---|---|
| handle | ops-blas 上下文和 stream | aclblasHandle_t | - | Host | 1 |
| d1 | 第一缩放因子，原地更新 | float* | FLOAT32 | Host 或 Device | 标量 1 |
| d2 | 第二缩放因子，原地更新 | float* | FLOAT32 | Host 或 Device | 标量 1 |
| x1 | 第一向量分量，原地更新 | float* | FLOAT32 | Host 或 Device | 标量 1 |
| y1 | 第二向量分量，只读 | const float* | FLOAT32 | Host 或 Device | 标量 1 |
| param | modified Givens 参数 | float* | FLOAT32 | Host 或 Device | 5 |

历史能力和本任务差距：

| 能力 | 历史 arch35 | 任务要求 |
|---|---|---|
| 公共 aclblasSrotmg API | 支持 | 复用，不新增产品私有接口 |
| Host 指针路径 | 支持 | arch22 保持一致 |
| Device 指针路径 | 支持 | 新增 arch22 AIV Kernel |
| Host/Device 混合指针拒绝 | 支持 | arch22 保持一致 |
| Atlas A2/A3 | 不支持 | 新增支持 |
| Netlib 全分支和缩放保护 | 已有参考 | arch22 完整实现并验证 |

### 1.6 aclblasSrotmg 算子功能分析

算子功能：根据 d1、d2、x1、y1 构造 modified Givens 变换 H，原地更新 d1、d2、x1，
并输出 param[5]。

输入：handle、d1、d2、x1、y1。

输出：更新后的 d1、d2、x1 和 param；y1 保持不变。

支持数据类型：FLOAT32。

支持形状：纯标量接口，无 shape 参数。

支持广播：不涉及广播。

支持内存模式：五个数据指针全部位于 Host，或全部位于 Device。

## 2. 需求分析（required）

### 2.1 需求描述

使用 Ascend C 为 aclblasSrotmg 实现 Atlas A2/A3 对应的 arch22 Device Kernel，并实现与
ops-blas 既有模式一致的 Host 入口。功能、flag 编码、缩放保护、原地更新、错误码和异步
stream 语义必须与任务书一致；有限输入以 cblas_srotmg 为 golden。

### 2.2 需求拆解

1. 复用 include/cann_ops_blas.h 中已有的 aclblasSrotmg 公共接口。
2. 支持 FLOAT32 标量 d1、d2、x1、y1 和 param[5]。
3. 完整覆盖 flag=-2、-1、0、1 四种输出编码。
4. 完整覆盖 GAM=4096、GAMSQ 和 RGAMSQ 缩放保护。
5. 支持全 Host 指针 CPU 路径。
6. 支持全 Device 指针 arch22 AIV Kernel 路径。
7. 拒绝空 handle、空数据指针和 Host/Device 混合指针。
8. Device 路径使用 handle 绑定的 stream，保持异步接口语义。
9. 在 Atlas 910B3 / CANN 9.1.0 上满足精度与三条性能门槛。
10. 提供完整 CSV、GTest、cblas golden、原始 profiler 和可复现说明。

## 3. 详细设计（required）

### 3.1 算子分析

#### 3.1.1 数学公式

modified Givens 变换满足：

    H^T * diag(d1, d2) * H = diag(d1_new, d2_new)
    H * [x1, y1]^T = [x1_new, 0]^T

param[0] 为 flag，其余槽位按 flag 条件有效：

| flag | H | 有效输出槽位 |
|---:|---|---|
| -1 | [[h11,h12],[h21,h22]] | param[1..4] |
| 0 | [[1,h12],[h21,1]] | param[2]、param[3] |
| 1 | [[h11,1],[-1,h22]] | param[1]、param[4] |
| -2 | 单位矩阵 | 仅 param[0] |

核心中间量：

    p2 = d2 * y1
    p1 = d1 * x1
    q2 = p2 * y1
    q1 = p1 * x1

#### 3.1.2 分支规则

1. d1 < 0：
   flag=-1，d1、d2、x1 和四个 H 元素全部置零。
2. d1 >= 0 且 d2*y1 == 0：
   flag=-2，d1、d2、x1 不变并立即返回。
3. abs(q1) > abs(q2)：
   计算 h21=-y1/x1、h12=p2/p1、u=1-h12*h21。u>0 时 flag=0 并更新
   d1、d2、x1；否则进入 flag=-1 全零分支。
4. abs(q1) <= abs(q2)：
   q2<0 时进入 flag=-1 全零分支；否则 flag=1，计算 h11=p1/p2、
   h22=x1/y1 和 u=1+h11*h22，再缩放并交换 d1、d2。

计算顺序保持 Netlib SROTMG 形式，避免代数重排改变 FLOAT32 舍入边界。

#### 3.1.3 缩放保护

常量：

    GAM = 4096
    GAMSQ = 1.67772e7
    RGAMSQ = 5.96046e-8

d1 或 abs(d2) 的非零有限值越出保护区间时，以 GAM^2 循环缩放，并同步补偿 x1 和 H 元素。
首次缩放将 flag 展开为 -1。Host 与 Device 均设置 MAX_SCALE_STEPS=32；有限 FLOAT32 可在
该上限内完成归一化。非有限值跳过缩放循环，避免正 Inf 触发无界 while。

#### 3.1.4 支持数据类型

仅支持 FLOAT32。接口类型已固定为 float*，不存在 dtype 枚举和混合精度组合。

#### 3.1.5 支持形状

本算子为纯标量算子：d1、d2、x1、y1 各 1 个 float，param 为 5 个 float。不存在维度、
动态 shape、非连续 Tensor、broadcast、步长、前导维或空 Tensor。

### 3.2 算子实现

#### 3.2.1 总体流程

    aclblasSrotmg
        -> handle/空指针校验
        -> 查询五个数据指针位置
        -> 全 Host：SrotmgCpuCompute
        -> 全 Device：srotmg_kernel<<<1, stream>>>
        -> 混合位置：ACLBLAS_STATUS_INVALID_VALUE

Host 与 Device 使用相同常量、比较顺序、缩放规则和 param 编码。

#### 3.2.2 Host 侧设计

参数校验：

1. handle==nullptr 返回 ACLBLAS_STATUS_HANDLE_IS_NULLPTR。
2. 任一数据指针为空返回 ACLBLAS_STATUS_INVALID_VALUE。
3. 使用 aclrtPointerGetAttributes 查询 d1、d2、x1、y1、param 的内存位置。
4. 全 Host 进入 CPU 计算，全 Device 发射 Kernel，混合位置返回 INVALID_VALUE。

Host 计算：

全 Host 输入直接在 CPU 上完成标量运算，不创建 Device 缓冲区，不发射 Kernel，也不进行
H2D/D2H 搬运。

Tiling 策略：

本算子总数据量固定为 9 个 float，控制流串行且没有 shape。Host 侧不生成 tiling data，
不申请 workspace，也不计算 tile 大小。

分核策略：

Device 路径固定发射 1 个 AIV block。多核无法并行化单组标量依赖，且多个 block 会竞争写
同一输出，因此不采用满核、大小核或尾块分配策略。

数据分块和内存策略：

不进行数据分块。四个输入标量和 param[5] 直接映射至 GM；有效载荷为 36 字节。Kernel 不
创建 TPipe、TQue 或 UB 临时缓冲。

TilingKey 规划：

不需要 TilingKey。分支由 Kernel 读取的标量值决定，Host 不需要感知数学分支。

Stream 与错误处理：

Device 路径使用 handle->stream 发射 Kernel。发射前清除线程级历史 runtime error，发射后
检查错误；失败返回 ACLBLAS_STATUS_EXECUTION_FAILED。接口不主动同步，调用方在读回结果
前同步同一 stream。

#### 3.2.3 Kernel 侧设计

Kernel 类型为 KERNEL_TYPE_AIV_ONLY，固定 1 个 block。数据映射：

| 数据 | GlobalTensor 长度 |
|---|---:|
| d1、d2、x1、y1 | 各 1 |
| param | 5 |

Kernel 流程：

1. 使用 GlobalTensor::GetValue 读取标量。
2. d1<0 和 p2==0 分支提前写回并返回。
3. 按 q1/q2 比较执行 flag=0、flag=1 或全零分支。
4. 对有限 d1、d2 执行最多 32 次缩放保护。
5. 按 flag 写入有效 param 槽。
6. 写回 d1、d2、x1；不修改 y1。

Kernel 不划分 CopyIn、Compute、CopyOut 流水，因为 9 个标量不足以抵消 UB 搬运和队列建立
的固定开销。直接 GM 标量访问更符合本算子的性能特征。

#### 3.2.4 性能优化方案

1. 单 AIV block，避免无效多核调度。
2. 无 tiling、workspace、UB 队列和 Host/Device 搬运。
3. flag=-2 与 d1<0 早返回。
4. 缩放保护使用有界循环，避免非有限输入挂死。
5. arch22 srotmg_kernel.cpp 单文件 -O2，不影响其他算子。
6. 性能采集将输入重置放在 task-time 之外，只统计 srotmg_kernel 硬件耗时。

#### 3.2.5 输出与异常设计

| 条件 | 行为 |
|---|---|
| handle 为空 | HANDLE_IS_NULLPTR |
| 任一数据指针为空 | INVALID_VALUE |
| 指针属性查询失败 | INVALID_VALUE |
| Host/Device 混用 | INVALID_VALUE |
| Kernel 发射失败 | EXECUTION_FAILED |
| d1、d2 为负值 | 合法数学输入，按分支计算 |
| y1=0 或 d2=0 | flag=-2，正常返回 |
| NaN/Inf | 合法输入；保证不因缩放循环挂死 |

无效 param 槽遵循 BLAS 条件编码，不强制覆写；测试两侧使用相同哨兵初始化后比较。

#### 3.2.6 文件组织

    blas/rotmg/
      README.md
      arch22/
        srotmg_host.cpp
        srotmg_kernel.cpp
    test/rotmg/srotmg/
      srotmg_golden.h
      srotmg_param.h
      arch22/
        srotmg_npu_wrapper.h
        srotmg_test.cpp
        srotmg_test.csv

## 4. 支持硬件

| 支持的芯片版本 | 支持情况 | 验证说明 |
|---|---|---|
| Atlas 800I/T A2（910B3） | 支持 | CANN 9.1.0 实机构建、功能、精度、性能 PASS |
| Atlas A3（arch22 / ascend910_93） | 支持 | arch22 交叉构建 PASS |
| Ascend 950PR / 950DT | 原有支持 | 使用既有 arch35 实现，不由本任务修改 |

## 5. 算子约束限制

1. 仅支持 FLOAT32。
2. d1、d2、x1、y1、param 必须全部为 Host 指针或全部为 Device 指针。
3. param 必须提供至少 5 个 float 的可写空间。
4. d1、d2、x1 原地更新，y1 只读。
5. 参数之间的内存重叠不属于接口契约。
6. 不涉及广播、动态 shape、非连续 Tensor、步长或前导维。
7. Device 路径异步执行，读回结果前必须同步 handle 绑定的 stream。
8. 非有限输入主要保证 Host/Device 一致和不挂死；有限输入严格以 cblas 为 golden。

## 6. 可维可测分析

### 6.1 可维护性

1. 复用公共 API、状态码、handle 和仓库构建框架。
2. Host 与 Device 的常量、分支次序和缩放上限一致。
3. arch22 与 arch35 目录隔离，不修改 arch35 算法。
4. -O2 仅作用于单个 arch22 Kernel 文件。
5. 无 tiling 结构和 workspace 协议，减少跨版本维护面。

### 6.2 可测试性

1. 有限输入使用 Netlib BLAS cblas_srotmg 单一 golden。
2. 同一用例同时验证 Host 和 Device 路径。
3. param[0] 精确比较，其余标量按 FLOAT32 容差逐项比较。
4. 覆盖空 handle、五种空数据指针、五种混合内存位置和自定义 stream。
5. 覆盖均匀/正态随机分布、NaN/Inf、边界与多轮缩放。
6. 性能只采纳独占资源下 msprof 的 srotmg_kernel task-time。

### 6.3 精度、性能和内存标准

| 验收标准 | 描述 | 标准来源 |
|---|---|---|
| 精度标准 | FLOAT32：rtol=2^-10，atol=2^-16，flag 精确；max abs error 不超过 1e-2 或 32 ULP | CANN 生态算子精度标准、任务书 |
| 功能标准 | 1000 条 CSV + 4 条固定 GTest 全部通过，无失败、跳过或超时 | 任务书和 ops-blas 测试框架 |
| 性能标准 | TC_PF_1001<=2.67 us，1002<=2.35 us，1003<=2.61 us；每条 60 warmup + 60 samples | 任务书 |
| 内存标准 | 参数载荷 36 字节，workspace=0，无动态 tiling 缓冲 | 实现设计 |

### 6.4 用例覆盖

| 分类 | 数量 | 覆盖 |
|---|---:|---|
| TC_L0 | 12 | flag=-2/-1/0/1 |
| TC_BR | 14 | 零、负零、相等与符号边界 |
| TC_SC | 18 | GAMSQ/RGAMSQ 和多轮缩放 |
| TC_FL | 9 | NaN、正负 Inf |
| TC_ED | 6 | 五种空指针和全零合法输入 |
| TC_EX | 941 | 均匀 471、正态 470 |
| TC_PF | 200 | 三条硬门槛和吞吐扫描 |
| 固定 GTest | 4 | handle、混合指针、stream、随机交叉测试 |

### 6.5 验收结果

910B3 / CANN 9.1.0 非性能测试为 1004/1004 PASS。正式 msprof 结果：

| case | 平均 task-time | 门槛 | 结论 |
|---|---:|---:|---|
| TC_PF_1001 | 1.847 us | 2.67 us | PASS |
| TC_PF_1002 | 1.250 us | 2.35 us | PASS |
| TC_PF_1003 | 1.519 us | 2.61 us | PASS |

## 7. 兼容性分析

1. API 兼容：接口签名、参数顺序、状态码和 stream 语义不变。
2. 产品兼容：新增 arch22 文件，arch35 继续使用原实现。
3. ABI 兼容：不新增导出符号，不修改公共结构体。
4. 构建兼容：仅在 NPU_ARCH=dav-2201 时设置 arch22 Kernel 的编译选项。
5. 行为兼容：有限输入对齐 Netlib/cublas；Host/Device 指针模式与仓内既有约定一致。
6. 性能兼容：优化不改变数学分支和输出编码。

## 8. 参考资料

1. 本任务任务书。
2. Netlib SROTMG：https://www.netlib.org/blas/srotmg.f
3. cuBLAS ROTMG：https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-rotmg
4. CANN ops-blas：https://gitcode.com/cann/ops-blas
5. CANN 生态算子精度标准。
