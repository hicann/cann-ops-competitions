# aclblasSgemmStridedBatched A2/A3 算子设计

# 需求背景

## 需求来源

CANN 社区任务 2026：08-21-aclblasSgemmStridedBatched-A2A3。面向 Atlas A2/A3 产品，使用 Ascend C 实现 FP32 跨步批量矩阵乘，并提供可复现的精度与性能测试。

## 背景介绍

多个规格相同的矩阵通过固定 stride 定位。复用 ops-blas 的句柄式公开接口与 stream，补充 arch22 实现，已有 arch35 实现保持独立。适用于批量线性代数和多组小矩阵计算。

# 需求分析

## 需求描述

计算 C_i = alpha × op(A_i) × op(B_i) + beta × C_i。输入输出为列主序 FP32；支持 N/T/C，实数下 C 等价于 T。支持运行时 m、n、k、batchCount，leading dimension padding、非紧凑批间 stride 和 A/B 零 stride 广播。C 的各批输出不得重叠。

## 需求拆解

1. 复用 include/cann_ops_blas.h 的 aclblasSgemmStridedBatched 接口，不新增产品私有接口。
2. 在 Atlas A2、Atlas A3 上分别验证有效输出精度、异常参数处理和性能。
3. 保持原始用例与 GPU 参考值，性能不超过 gpu_ms / 0.8。
4. 提供算子 README、完整测试程序、原始实机结果与复现步骤。

# 详细设计

## 算子分析

### 数学公式

A_i = A + i × strideA，B_i = B + i × strideB，C_i = C + i × strideC，i 从 0 到 batchCount−1。stride 按元素数计，不按字节计。

### 支持数据类型与形状

矩阵及 alpha、beta 均为 FP32；矩阵位于 Device，标量位于 Host。m、n、k、batchCount 非负，stride 使用 int64_t。非转置 A 的 lda 至少为 max(1,m)，转置时至少为 max(1,k)；非转置 B 的 ldb 至少为 max(1,k)，转置时至少为 max(1,n)；ldc 至少为 max(1,m)。

## 算子实现

### Host 侧设计

检查 handle、操作枚举、尺寸、stride、leading dimension 与指针。合法零 m/n/batchCount 返回成功。k=0 或 alpha=0 时走 beta 缩放；beta=0 不读取旧 C，beta=1 的纯缩放无需计算。指针检查遵循公开接口要求。

一般计算按 128×128 输出块划分，将 batch 与二维输出块映射为任务并分配给运行时查询的 Cube 核。K 超过 4096 时分段累加。调度数据包含尺寸、转置、前导维、批间 stride、计算路径及分组参数；通过句柄绑定的 stream 异步执行。

alpha=1、beta=0 时直接写 C，包括合法非对齐 ldc。其他组合先写句柄 workspace，再由 AIV 合并 alpha 与 beta。按 workspace 容量合并多个 batch；单矩阵放不下时沿列方向分条，同一 stream 内复用空间。至少需要一列临时输出，否则返回分配失败。

### Kernel 侧设计

一般矩阵使用 Ascend C Matmul 的 FP32 Cube 路径，关闭 HF32。交换操作数，以 C 的转置视角适配列主序，显式传递 leading dimension。AIV 合并及纯缩放路径处理有效输出与尾部，只写有效元素，保留 padding 和批间间隙。

紧密 NN、alpha=1、beta=0 的 8³ 场景使用 Gather/FMA。batchCount 至少为 512 时每组处理 32 批，以减少重复初始化；小批量使用较小分组。16³/32³ 大批量使用静态 Batch Matmul，减少逐矩阵初始化与输出搬运开销。

Tiny 路径检测极大输入，在需要时使用整数实现的 FP32 FMA：保留完整 48 位乘积，完成加数对齐、sticky 位与最近偶数舍入，并处理零、非正规数及 NaN/Inf。混合核通过同步保护回退输出，避免与 Cube 写回竞争。矩阵运算不存在 CPU 数值回退。

### 分块与内存优化

默认复用 32 MiB handle workspace，直接输出和小矩阵路径不需要临时输出矩阵。临时输出 leading dimension 向上对齐至 8 个 FP32 元素。调用方负责提供足够的矩阵存储跨度。编译采用 Release，支持显式 Debug 构建。

## 支持硬件

| 产品 | 支持情况 | 已测 SoC |
|---|---|---|
| Atlas A2 系列，含 Atlas 800I A2 | 支持 | Ascend910B3 |
| Atlas A3 系列，含 Atlas 800I A3 | 支持 | Ascend910_9382 |

## 算子约束限制

输入转置、尺寸与 leading dimension 须满足 BLAS 接口约束；不支持负 stride。A/B 可跨批广播，C 各批不得重叠。非有效填充区保持不变。标量是 Host 指针，矩阵是 Device 指针。具体参数和返回值见代码仓算子 README。

# 可维可测分析

## 精度标准与性能标准

CPU golden 使用 Netlib cblas_sgemm。按 rtol=2⁻¹⁰、atol=2⁻¹⁶、匹配比例至少 0.99，并检查绝对误差不超过 0.01 或不超过 32 ULP；NaN/Inf 单独判断，padding 与间隙逐位比较。

每端共 1312 项测试：原始精度 1000 项、含精度检查的性能 200 项、补充回归 110 项、独立接口 2 项。覆盖九种转置、广播、非对齐布局、尾部、零 K、K=4097、非法参数、极大值和 NaN/Inf。

每项性能测试预热 20 次，取全部 200 次有效样本的算术平均；stream event 测量接口提交的 kernel 序列，排除数据准备、拷贝与 CPU golden。阈值为原始 GPU 时间除以 0.8，不更改原始用例及参考值。

2026 年 9 月 8 日最终测试中，A2、A3 均通过全部 1312 项测试及 200 项性能测试。两端均使用 Ubuntu 22.04 aarch64、CANN 9.1.0、GCC 11.4。前三个典型性能场景的 A/B/C 逻辑容量均为 192 MiB，设备峰值显存未采集；逻辑容量不等同于实测峰值。

## 兼容性分析

公开接口与参数顺序保持一致，arch22 与已有 arch35 分开实现。构建沿用 ops-blas 框架；现有两端验证结果不代表其他产品已经完成实测。CPU golden 的 B 布局整理仅用于参考计算，不改变 NPU 输入或计时。

## 代码与复现入口

- 代码仓：https://gitcode.com/shi-xiangyang225/aclblasSgemmStridedBatched
- 分支：sgemm-strided-batched-a2a3
- 实现：blas/gemm_strided_batched/arch22/
- 测试：test/gemm_strided_batched/arch22/
- 算子说明：blas/gemm_strided_batched/README.md

在 Ascend Linux 中安装 CANN 9.1.0、CMake、GCC、GTest 与 Netlib BLAS/LAPACK，执行 bash build.sh --soc=ascend910b3 --ops=gemm_strided_batched。A3 将 SoC 替换为 ascend910_9382。完整步骤及逐项测试结果随自测交付件提供。
