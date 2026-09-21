# aclblasChemm 950 设计文档

> **AI 协助说明**：本文档由 GPT-6 协助整理，依据 9 月上海站 aclblasChemm 950 任务书、Netlib BLAS 语义、本地实现和验证记录编写；设计取舍与验证结果以实际代码和日志为准。

## 1. 需求背景

本任务来自 9 月上海站算子实操工坊 `aclblasChemm` 950 赛题，目标是在 Ascend 950PR 上实现公共 BLAS 接口 `aclblasChemm`，覆盖 Hermitian 矩阵乘的接口语义、数值精度、性能和异常输入处理。

目标软件环境为 CANN 9.1.0，目标架构为 Ascend 950PR/arch35。接口沿用公共 `cann_ops_blas.h` 声明，不增加产品私有接口。

## 2. 需求分析与接口

输入为列主序 `complex64` 矩阵。A 为 Hermitian 矩阵，只存储 `UPPER` 或 `LOWER` 三角；B、C 为一般的 `m×n` 矩阵。根据 `side` 定义：

```text
LEFT  : C = alpha * A * B + beta * C
RIGHT : C = alpha * B * A + beta * C
```

A 的阶数由 `side` 决定：LEFT 时为 `m`，RIGHT 时为 `n`。`lda` 至少为 A 的阶数，`ldb` 和 `ldc` 至少为 `m`，维度和 leading dimension 使用 `int64_t`。

读取未存储的三角位置时，读取转置位置并对虚部取负；对角元素的虚部按 Hermitian 定义置零。输入输出均为 Device 指针，调用者保证矩阵容量和 BLAS 别名约束满足要求。

## 3. 详细设计

### 3.1 Host 参数检查与分派

Host 层检查 handle、`side`、`uplo`、维度、标量指针、矩阵指针、leading dimension 以及字节寻址溢出。`m==0` 或 `n==0` 时不访问输入并直接返回成功。`alpha==0 && beta==1` 时直接返回；`beta==0` 的路径不读取旧 C，避免无效读和 `0*Inf` 传播。

根据矩阵规模、标量类型、地址范围和 workspace 容量选择 SIMT 或 Cube 路径。所有 kernel 使用 handle 绑定的 ACL stream，不创建全局 stream、event 或数据缓存；单次接口调用不增加强制 Host 同步。

### 3.2 SIMT 与 Hermitian 顺序路径

小矩阵、alpha 为复数、数值风险较高或不满足 Cube 条件时使用 SIMT。Hermitian A 按指定三角规则读取，复数乘法显式拆分为 FP32 乘积和加减，保持与 CBLAS 接近的舍入边界。

为满足单标杆精度，LEFT 路径按参考实现的三角更新顺序计算非对角项和对角项；RIGHT 路径先计算对角贡献，再按 A 的列归约。受限尺寸内可预展开 A，LEFT 还可预计算 `alpha*B`。检测到非有限中间结果时，按原始 A/B 和参考顺序重算对应输出。

### 3.3 Cube 路径与复数分解

Cube 路径将复数矩阵乘转换为实数平面矩阵乘，并通过转置视图使连续维度对应合并搬运和输出写回。M/N 按 16 对齐，K 按 128 对齐，尾块补零；默认输出 tile 为 `128×128`，L1 K chunk 为 `256`，L0 K step 为 `32`。

纯实标量路径采用 3M：

```text
RR = Ar * Br
II = Ai * Bi
SS = (Ar + Ai) * (Br + Bi)
real = RR - II
imag = SS - RR - II
```

复数标量或数值风险较高时采用四乘积或更保守的分段路径。K 归约通常切分为 4 段，风险场景切分为 8 段；各段独立累加，在 combine 阶段成对归并以缩短 FP32 长累加链。

### 3.4 Workspace、combine 与 fallback

以 `Mp=ceil(m/16)*16`、`Np=ceil(n/16)*16`、`Kp=ceil(K/128)*128` 表示对齐维度，workspace 主要包含：

```text
3 * Np * Kp                  # B 的三平面
3 * Kp * Mp                  # A 的三平面
P * S * Mp * Np              # 实数乘积和 split-K 分段
queue capacity + per-core counters
```

其中 `P` 为 3 或 4，`S` 为 4 或 8。Host 按 float 元素数换算字节数，并将队列和边界标志计入 workspace；超出 32 位索引安全范围时保留 int64 寻址。

AIV combine 汇总 split-K 平面，完成复数 `alpha/beta` 缩放和 C 写回，并根据数据驱动的风险指标决定是否进入 ordered fallback。fallback 按原始 Hermitian 语义重算对应输出，避免固定坐标恢复影响泛化精度。

### 3.5 事件与写回

Cube 数据通路在 GM→L1、L1→L0、MMAD 和写回之间使用配对 HardEvent。ping-pong buffer 在复用前等待对应消费者事件，尾块显式清理 padding，循环结束消费剩余事件。C 地址不满足安全对齐时，使用单核写回，避免跨核共享 cache line。

## 4. 可维可测分析

精度 golden 使用 Netlib CBLAS，分别检查实部和虚部的误差、NaN/Inf 同类行为、边界、偏移、对齐和 C padding。FP64 仅用于定位舍入误差，不作为验收 golden，也不放宽任务书阈值。

性能测试先预热，再使用 ACL event 统计有效调用平均时间；性能判定固定测试程序、CSV、GPU baseline、动态库和设备，保留每条用例的原始样本。任务书四项硬性能上限按原始任务书执行。

历史 acceptance04 口径复跑曾得到 4467/4467 精度/接口项和 200/200 性能项通过。后续严格 benchmark 口径下，当前候选结果为精度 1000/1000、性能 163/200；剩余性能问题主要集中在深 K ordered fallback 和一条窄 N 短 K 形状。该差异在验证记录中单独保留，不用历史结果替代严格门禁结果。

## 5. 支持范围

支持 Ascend 950PR/arch35 和 CANN 9.1.0。其他产品线继续使用公共接口声明；不同 SOC 的核数、workspace 容量和 Cube 约束需要重新进行 tiling 评估，不能直接复用 950PR 参数。
