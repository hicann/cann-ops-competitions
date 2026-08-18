# 【社区任务】ELU 算子设计文档

# 需求背景（required）

## 需求来源

本文对应 2026 年 8 月社区任务《ELU 算子开发任务书》。任务要求在 Ascend 950 上基于 Ascend C C API 实现纯 Vector-Core ELU 样例，并提交至 `asc-devkit` 的 `examples/02_simd_c_api/03_c_api/02_reg_vector_compute` 目录。

## 背景介绍

ELU（Exponential Linear Unit）是神经网络常用的逐元素激活函数。正区间保持线性，负区间通过指数函数平滑饱和。该任务的主要挑战包括：

1. 指数运算指令链较长，需要减少中间搬运和不必要的分支。
2. 必须同时支持 FLOAT 和 FLOAT16，并满足生态算子开源精度标准。
3. 输入 shape 需要泛化，覆盖非对齐尾块、多 UB tile 和多核尾核。
4. 全程只能使用 Vector Core，Host 侧不得逐元素计算。

# 需求分析（required）

## 需求描述

在 `asc-devkit` 新增 Ascend C C API ELU 样例。`src`、`dst` 为连续 ND Tensor，按一维连续元素处理；`alpha`、`scale`、`input_scale` 为 FLOAT 标量。在 CANN ACLNN 接口语义中，Tensor 参数由 `aclTensor*` 表示，标量参数由 `aclScalar*` 表示；本直调样例在启动 Kernel 时将三个标量按值传入，因此标量参数不具有 Tensor 数据格式。

计算公式为：

```text
y = scale * x                                      (x > 0)
y = alpha * scale * (exp(x * input_scale) - 1)    (x <= 0)
```

## 需求拆解

1. 支持 FLOAT、FLOAT16 输入输出，格式为 ND。
2. 支持任意合法元素数，包括空 Tensor、非 32B 对齐尾块和超过 UB 容量的大输入。
3. 支持 `alpha`、`scale`、`input_scale` 参数泛化。
4. 所有逐元素计算在 Device Vector Core 上完成，禁止调用 Cube/Matmul API。
5. 合理使用 UB、mask、分核和流水同步，保证 GM/UB 边界安全。
6. 提供可复现的数据生成、精度校验、泛化回归和性能对比脚本。
7. 目标硬件为 Ascend 950，验证 CANN 版本为 9.0.0～9.1.0。

# 详细设计（required）

## 算子分析

### 数学公式

实现使用以下等价表达式：

```text
positive = max(x, 0) * scale
negative = alpha * scale * (exp(min(x, 0) * input_scale) - 1)
y = positive + negative
```

当 `x > 0` 时，`negative` 为 0；当 `x <= 0` 时，`positive` 为 0。该表达式与原始分段函数一致，同时避免额外的 Compare/Select 指令和中间结果回写 GM。

### 输入输出与参数

| 名称 | 类型 | 数据类型 | 格式 | 说明 |
| --- | --- | --- | --- | --- |
| src | Input | FLOAT/FLOAT16 | ND | 连续输入 Tensor；在 ACLNN 接口中对应 `aclTensor*` |
| dst | Output | FLOAT/FLOAT16 | ND | 连续输出 Tensor，与 src 同 shape、同 dtype；在 ACLNN 接口中对应 `aclTensor*` |
| alpha | Input | FLOAT | 不适用（标量） | 负区间激活系数；在 ACLNN 接口中对应 `aclScalar*`，Kernel 启动时按值传入 |
| scale | Input | FLOAT | 不适用（标量） | 输出缩放系数；在 ACLNN 接口中对应 `aclScalar*`，Kernel 启动时按值传入 |
| input_scale | Input | FLOAT | 不适用（标量） | 指数输入缩放系数；在 ACLNN 接口中对应 `aclScalar*`，Kernel 启动时按值传入 |

### 支持形状

Kernel 不依赖维度语义，仅依赖连续元素总数，因此支持任意维度的连续 ND Tensor。空 Tensor 直接返回，不启动 Kernel；非空 Tensor 支持任意正元素数。

## 算子实现

### Host 侧设计

Host 负责：

1. 解析并校验 dtype、元素数、三个标量以及输入输出路径。
2. 根据元素数动态选择启动 block 数。
3. 申请 Host/Device 内存并执行 H2D、Kernel 下发、同步、D2H 和文件写出。
4. 在统一清理路径释放 stream、Device/Host 内存并复位设备。

Host 不包含逐元素 ELU 或 golden 计算循环。golden 由独立 Python 测试脚本生成。

### 分核策略

- 目标 dav-3510 提供 64 个 AIV；实际调优表明 `1M+7` FLOAT 用 48 核耗时 `2.910 us`，64 核为 `4.283 us`，因此最终调度上限为 48。
- Host 使用 `min(48, ceil(totalLength / 8192))` 选择 block 数。小 shape 保持单核，较大 shape 自动扩展并行度。
- Kernel 内通过实际 block 数和总长度划分连续区间，前若干核在不能整除时多处理一个对齐块，最后一核截断到真实总长度。
- CMake 将目标硬件核心数和调度上限注入编译，并以静态断言防止调度上限超过目标硬件能力。

### UB Tiling 与边界处理

- 每个核按 8192 元素为一个 tile 循环处理，覆盖超过 UB 容量的大输入。
- 主体搬运使用对齐数据搬运接口；最后 tile 的 GM 搬运长度为实际有效字节数，避免越界。
- VF 计算使用 `asc_update_mask_b32` 或 `asc_update_mask_b16` 更新有效元素 mask，覆盖非 VF 整数倍尾部。
- 所有 UB 访问限定在静态分配的 tile 范围内。

### FLOAT 计算链

FLOAT 使用 `vector_float` 完成 `min/max/muls/exp/adds/add` 全链计算。单个 VF repeat 处理 64 个 FLOAT 元素，循环直到当前 tile 的有效元素全部完成。

### FLOAT16 计算链

FLOAT16 使用 `vector_half` 原生计算链，单个 VF repeat 处理 128 个 FLOAT16 元素。三个 FLOAT 标量在 Kernel/VF 入口转换为 half。未使用 unpack→FP32→pack 混合路径，原因是该组合对奇偶寄存器槽位敏感；原生 half 路径的实测误差满足任务精度标准。

### Ping-Pong 双缓冲与流水同步

每核静态分配两组输入/输出 UB：

| dtype | 输入 UB | 输出 UB | 合计 |
| --- | ---: | ---: | ---: |
| FLOAT | 2 × 32 KiB | 2 × 32 KiB | 128 KiB |
| FLOAT16 | 2 × 16 KiB | 2 × 16 KiB | 64 KiB |

偶数和奇数 tile 分别使用 EVENT_ID0/1 保护 ping/pong 缓冲。每个 tile 执行 MTE2→Vector→MTE3，只等待当前缓冲的真实依赖，使相邻 tile 的搬入、计算和搬出能够重叠；循环结束后回收两组 event。

### Kernel 入口

FLOAT 与 FLOAT16 分别使用独立 Kernel/VF 实例化，Host 按 dtype 选择入口。Kernel 参数包含 GM 输入输出地址、总元素数以及三个标量，不传递维度信息。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Ascend 950PR/Ascend 950DT（dav-3510） | √ |

验证环境为 Ascend 950PR、CANN 9.0.0、dav-3510。

## 算子约束限制

1. 输入输出必须是连续 ND 数据，输出与输入 dtype、元素数一致。
2. 仅支持 FLOAT、FLOAT16。
3. `alpha`、`scale`、`input_scale` 必须是有限 FLOAT 标量。
4. 当前工程只面向 dav-3510，不编译其他架构。
5. 性能对比的 Torch 基线使用 `alpha=scale=input_scale=1`，保证两侧语义一致。

# 可维可测分析

## 精度标准/性能标准

精度采用任务指定的生态算子开源标准：逐元素混合容差，同时满足匹配率和最大绝对误差硬上限。

| dtype | rtol | atol | matched ratio | max abs error |
| --- | ---: | ---: | ---: | ---: |
| FLOAT16 | 2^-9 | 2^-9 | >= 0.99 | <= 1e-1 |
| FLOAT32 | 2^-10 | 2^-16 | >= 0.99 | <= 1e-2 |

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | 满足上述生态算子开源精度标准 | ELU 社区任务书 |
| 性能标准 | 任务书无固定性能门槛；提供与 Torch NPU ELU 的同口径实测对比 | ELU 社区任务书 |
| 核型标准 | msprof 显示 Op Type 为 Vector，Cube 字段为 NA | ELU 社区任务书 |

### 测试设计

测试分为 task/full 两套矩阵：

1. task：FLOAT/FLOAT16 × shape 1、32、1024、2048，输入范围 [-100, 100]。
2. full：在 task 基础上增加空 Tensor、分支边界、7/8/15/17/31/33 尾块、VF 255/256/257、2047/2049、131071 多核尾核、`1048583 (1M+7)` 多 tile、非单位参数、alpha=0 和 scale=0。
3. 性能矩阵：长度 `1,31,32,257,1024,2048,2049,8193,131071,1048583`，两种 dtype，双方使用相同输入、warmup 10、repeat 50 和 NPU Event 计时，并在计时后执行精度校验。

### 实测结果

- 完整泛化矩阵 42/42 通过。
- FLOAT 最大绝对误差 `1.1921e-07`，FLOAT16 最大绝对误差 `4.8828e-04`。
- 完整性能矩阵 20/20 快于 Torch；FLOAT 提升 `33.67%～59.04%`，FLOAT16 提升 `55.25%～61.03%`。
- 最终源码另以长度 31、2049、1048583 复测 6 组，6/6 通过精度且快于 Torch。

## 可维护性分析

1. 数学链、分核、搬运和 Host 资源管理相互分离，便于独立修改。
2. 中英文 README 提供构建、单用例、完整回归和性能复现命令。
3. 参数解析拒绝非有限浮点和带尾随字符的数值，错误输入在设备初始化前失败。
4. 测试脚本保存每个 case 的输入、golden、输出、精度指标和原始日志，便于复现与回归。

## 兼容性分析

本次新增独立样例目录和上级 README 索引，不修改现有公开 C API、不改变已有样例行为，不涉及二进制兼容性。工程显式限定 dav-3510，后续如适配其他架构，需要根据对应安装头的数据搬运签名、UB 容量和 AIV 数重新验证。

## 风险与对策

| 风险 | 对策 |
| --- | --- |
| 非对齐 GM/UB 越界 | 精确有效字节搬运、VF mask，并覆盖多组非对齐长度 |
| 多 tile 覆盖尚未搬出的输出 | EVENT_ID0/1 独立保护 ping/pong 输入输出缓冲 |
| FLOAT16 指数误差 | 使用生态标准逐 case 校验并保留原始误差指标 |
| 多核尾核负载不均 | 对齐分段，最后一核按总长度截断 |
| CANN 版本接口签名差异 | 以任务目标 CANN 9.0.0～9.1.0 和 dav-3510 实机头文件、编译结果为准 |
