# MatmulLayerNormMatmul 算子设计文档

## 需求背景（required）

### 需求来源

本设计对应 2026 年社区任务 `09-45-MatmulLayerNormMatmul`，目标硬件为 Ascend 950PR。任务要求把 Matmul、行级 LayerNorm 和第二个 Matmul 融合为一次 kernel launch，并避免将中间矩阵作为对外输出写入 GM。

### 背景介绍

标杆实现依次调用 `torch.mm`、`torch.nn.functional.layer_norm`、`torch.mm`。它需要多次 kernel launch，并在算子间传递形状为 `[M0, N0]` 的中间张量。本方案在同一 MIX kernel 中衔接 Cube 和 Vector 计算，让中间数据留在片上存储。

## 需求分析（required）

### 需求描述

输入 `A0[M0,K0]`、`B0[K0,N0]`、`B1[N0,M1]`、`gamma[N0]`、`beta[N0]`，唯一输出为 `C1[M0,M1]`。对 `C0=A0@B0` 的每一行独立计算均值、方差与仿射归一化，再与 `B1` 相乘。`epsilon=1e-6`。

### 需求拆解

1. 使用单次 MIX kernel launch 完成两次矩阵乘和 LayerNorm。
2. 仅输出 `C1`；中间矩阵及统计量由算子内部管理。
3. 支持任务指定的 FP16 输入矩阵、FP32 `gamma/beta`、FP16 输出及非对齐尾块。
4. 接入 CATLASS Optest 任务测试集，并提供独立的 ATK 泛化精度交付件。
5. 在 Ascend 950PR 上采集有效设备时延，按任务测试集计算平均 `标杆时延 / 融合时延`，目标大于 `1.1`。

## 详细设计（required）

### 算子分析

#### 数学公式

对每一行 `m`：

```text
C0[m, :] = A0[m, :] @ B0
mean[m] = sum(C0[m, :]) / N0
var[m] = sum((C0[m, :] - mean[m])²) / N0
Y[m, n] = (C0[m, n] - mean[m]) / sqrt(var[m] + 1e-6) * gamma[n] + beta[n]
C1 = Y @ B1
```

#### 输入输出

| 张量 | 类别 | 数据类型 | 逻辑形状 | 布局 |
| --- | --- | --- | --- | --- |
| A0 | 输入 | FP16 | `[M0, K0]` | RowMajor |
| B0 | 输入 | FP16 | `[K0, N0]` | ColumnMajor |
| B1 | 输入 | FP16 | `[N0, M1]` | ColumnMajor |
| gamma | 输入 | FP32 | `[N0]` | ND 连续 |
| beta | 输入 | FP32 | `[N0]` | ND 连续 |
| C1 | 输出 | FP16 | `[M0, M1]` | RowMajor |

MM0 输出先落为 FP16；LayerNorm 的统计与仿射计算使用 FP32，归一化结果转换为 FP16 后参加 MM1。这个中间精度路径与任务中的 PyTorch 参考实现保持一致。

### 算子实现

#### Host 侧设计

Host 检查设备、dtype、连续性、维度及矩阵乘 shape 关系，填充 `TParams + MatmulLayerNormMatmulParams`，并分配唯一输出张量。核组数为 `min(AIC 数量, ceil(M0 / 16))`；Host 不分配对外可见的中间 workspace。

#### Kernel 侧设计

一个 AIC 与配对的两个 AIV 组成核组，每组处理最多 16 行。AIC 执行 MM0、MM1；两个 AIV 分别负责最多 8 行的 LayerNorm、MM1 分块累加及输出转换。

```text
A0/B0 (GM) → MM0 (AIC) → C0 (共享 UB)
                         → LayerNorm (AIV 私有 UB) → Y (共享 L1，zN)
B1 (GM) ─────────────────────────────────────────→ MM1 (AIC)
                                                   → FP32 分块累加 (AIV UB) → C1 (GM)
```

- MM0 的 L1 Tile 为 `16×128×128`，L0 Tile 为 `16×128×64`；使用 ping-pong 搬运。
- LayerNorm 沿 `N0` 按 128 元素分块，先计算均值，再对中心化结果求方差，以减小 `E[x²]-E[x]²` 的消减误差。
- 归一化后的 `Y` 按 16 个对齐物理行以 zN 布局驻留 L1，MM1 直接读取。MM1 的 L1/L0 Tile 为 `16×128×128`，并沿 `N0` 分块迭代。
- MM1 的分块结果由 AIV 以 FP32 累加，最后转换为 FP16 写入 `C1`。
- `M0/N0/K0/M1` 的非对齐尾块使用实际有效形状和对齐搬运参数处理；多核及第二轮调度均已覆盖。

#### 片上空间与同步

在当前最大配置 `N0=8192` 下，静态 L1 分区合计为 401408 字节，小于 Ascend 950 的 512 KiB。单 AIV 的 LayerNorm 阶段 UB 上界为 146592 字节。实现使用静态断言检查 UB/L1 分区，并按生命周期复用 UB 空间。

AIC 与 AIV 使用 MM0 就绪/复用、L1 常驻就绪、MM1 就绪/复用、行组完成等事件同步；AIC 等待两个 AIV 都完成相关阶段后，才复用共享缓冲区。`V_MTE2` 事件保护 `gamma/beta` 读取期间的 UB 缓冲区，`V_S/S_V` 事件保护均值及倒标准差的标量访问。

#### 软件接入

算子代码位于 CATLASS 私仓分支 `feature/matmul-layernorm-matmul`，提交为 `fb65cc860150657596e40cbfb9b4a03474cf21da`。交付包含 experimental C++ 样例、kernel 头文件、JIT 入口、Torch C++ adapter、Python wrapper 与 Optest 测试。ATK 四件套单独交付，不加入 CATLASS 源码 PR。

### 支持硬件

| 芯片版本 | 是否支持 |
| --- | --- |
| Ascend 950PR，`CATLASS_ARCH=3510` | 是，已上板验证 |
| Ascend 950DT，`CATLASS_ARCH=3510` | 代码路径适配，尚未单独上板验证 |
| Atlas A2 / Atlas A3，`CATLASS_ARCH=2201` | 否 |

### 算子约束限制

任务接口未给出额外业务约束。当前实现要求 `M0/N0/K0/M1` 均为正数，且片上全行常驻方案支持 `N0≤8192`；任务 CSV 的最大 `N0` 为 8192。输入必须满足表中的 dtype、布局及 shape 关系，当前不支持非连续 Tensor。`N0>8192` 需要另行设计分段 LayerNorm/MM1，不能视为已支持。

## 特性交叉分析

本任务不涉及量化、broadcast、变长序列、mask 或其他融合激活。MM0 的 FP16 中间结果、LayerNorm 的 FP32 统计和 MM1 的 FP16 输入分别对应参考计算的中间精度语义。同步仅发生在成对的 AIC/AIV 核组内，不依赖跨核组共享状态。

## 可维可测分析

### 精度标准/性能标准

| 验收项 | 标准 | 验证状态 |
| --- | --- | --- |
| 功能与边界 | 单 kernel 完成三段计算；覆盖循环、多核、尾块及上界 | 最终提交 C++ 回归 `12/12 PASS` |
| Optest 精度 | 任务 CSV 的 116 例全部通过 | 最终提交 `116/116 PASS` |
| ATK 泛化精度 | 至少 200 例，通过生态精度标准 | 最终提交 `200/200 PASS`，`acc_pass_result:Pass` |
| 性能 | 116 例平均 `标杆时延 / 融合时延 > 1.1` | 尚未取得有效设备时延，不能声明达标 |

精度测试在 Ascend 950PR 上针对最终源码提交重新构建并完成。C++、Optest、ATK 原始日志分别位于 `/workspace/mlnm_final_cpp_20260924_135325/`、`/workspace/mlnm_final_optest_20260924_140201/`、`/workspace/mlnm_final_atk_20260925_012136/`；测试日志作为验收附件保存，不放入本设计文档提交。

### 性能采集现状与后续验证

任务规定使用 `msprof op` 获取设备时延。当前环境在 `StartFFTSTask` 阶段报 `Profiling channel start failed`，未生成 `duration.bin`，因此 `0 us`、`NA` 均不是有效时延。相同环境下 CATLASS 官方 Matmul 样例也出现同类通道故障。需由平台提供可用 profiling 通道后，采集任务 CSV 的 116 例，并逐例计算 `标杆时延 / 融合时延` 的算术平均值；若环境持续不可用，应取得验收方书面处置意见。本设计不将通道故障视作性能达标结果。

### 兼容性与维护

算子使用独立的 `matmul_layer_norm_matmul` 命名，不覆盖 CATLASS 现有注册。TileShape、缓冲区偏移与同步事件使用具名常量，片上空间通过静态断言约束。后续如果扩展 dtype、`N0` 范围或硬件型号，应分别补充对应的精度、边界及性能回归。
