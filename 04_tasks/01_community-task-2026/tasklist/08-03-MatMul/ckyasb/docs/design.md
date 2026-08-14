# 【社区任务】MatMul 算子设计文档

> 参与任务昵称：ckyasb；GitCode 账号：gcw_NUTKvS80。

# 需求背景（required）

## 需求来源

本需求来源于 2026 年 8 月社区任务 MatMul 算子开发任务书。目标是在 `asc-devkit` 仓库中，基于 Ascend C C API 为 Ascend 950 开发固定规格、纯 Cube-Core、单核的 FP32 矩阵乘加样例，并提供可复现测试代码与自测报告。

数学语义为：

$$
C_{i,j}=\sum_{k=0}^{K-1}A_{i,k}B_{k,j}+Bias_j
$$

任务固定规格为 A `[30,40]`、B `[40,70]`、Bias `[1,70]`、C `[30,70]`，所有输入输出均为 float，A/B 均不转置。

## 背景介绍

### MatMul现有样例分析

`asc-devkit` 已提供 `examples/02_simd_c_api/03_c_api/03_matrix_compute/mmad` 样例，演示 int8_t 和 bfloat16 输入下的 C API 矩阵乘。该样例可作为工程结构、Host 启动和矩阵流水的参考，但本任务需要补充以下能力：

1. A、B、Bias 和 C 全部使用 float；
2. 处理 M=30、K=40、N=70 的非 16 对齐规格；
3. 使用 BiasTable 完成 `[1,70]` 偏置在 M 方向的广播；
4. 固定单个 Cube Core 执行完整矩阵乘，不在 Host 侧切块；
5. 提供满足生态算子 FLOAT32 精度标准的自动化测试。

### MatMul功能分析

| 参数 | 输入/输出 | 数据类型 | Shape | GM排布 | 是否转置 |
| --- | --- | --- | --- | --- | --- |
| `src0` / A | 输入 | float | `[30,40]` | ND | 否 |
| `src1` / B | 输入 | float | `[40,70]` | ND | 否 |
| `weight` / Bias | 输入 | float | `[1,70]` | ND | 不适用 |
| `dst` / C | 输出 | float | `[30,70]` | ND | 不适用 |

Host 只负责文件 I/O、Device 内存管理、H2D/D2H 拷贝和一次 kernel launch。矩阵搬运、布局转换、乘加和 Bias 广播全部在 Device 侧完成。

# 需求分析（required）

## 需求描述

使用 Ascend C C API 在 Ascend 950 上实现固定规格 FP32 MatMul + Bias。核函数必须声明为 `__global__ __cube__`，矩阵计算必须调用 Cube 矩阵指令对应的 C API，启动核数固定为 1，且不得在 Host 侧通过循环逐元素计算结果。

## 需求拆解

1. 支持 A `[30,40]`、B `[40,70]`、Bias `[1,70]`、C `[30,70]` 的固定规格。
2. 支持 float 输入、float 累加和 float 输出。
3. A/B 在 GM 中均为 ND 非转置布局。
4. 通过 Device 侧 ND/Nz/Zn 布局转换满足 Cube 输入要求。
5. 通过 BiasTable 在 Cube 计算阶段叠加并广播 Bias。
6. 固定单核执行，不设计多核 TilingData 或 Host 分块逻辑。
7. Local Memory 按对齐后尺寸静态申请，保证 L1、L0A、L0B 和 L0C 不越界。
8. 提供输入生成、批量执行、精度校验和自测报告。
9. 输入数据覆盖 `[-100,100]`，精度满足生态算子 FLOAT32 标准。
10. 任务无性能要求，不启用 HF32 或多核优化。

# 详细设计（required）

## 算子分析

### 数学公式

对 `0 <= i < 30`、`0 <= j < 70`：

```text
dst[i,j] = sum(src0[i,k] * src1[k,j], k=0..39) + weight[j]
```

Bias 的逻辑 shape 为 `[1,70]`，在 M 方向广播到 30 行。

### 支持数据类型

| 参数 | 数据类型 | 说明 |
| --- | --- | --- |
| A | float32 | Cube 左矩阵 |
| B | float32 | Cube 右矩阵 |
| Bias | float32 | BiasTable 初值 |
| C | float32 | FP32 累加与输出 |

### 支持形状

当前实现严格对齐任务固定规格，不提供动态 shape：

| 维度 | 固定值 |
| --- | ---: |
| M | 30 |
| K | 40 |
| N | 70 |

## 算子实现

### 实现方案

总体数据通路如下：

```text
GM(ND) --ND2NZ--> L1(Nz) --Load2Dv2--> L0A(Nz) / L0B(Zn)
Bias GM(ND) -----> L1 ----------------> BiasTable
L0A x L0B + BiasTable --asc_mmad--> L0C(Nz) --Fixpipe NZ2ND--> GM(ND)
```

#### Host侧设计

Host 侧不参与数学计算，执行流程如下：

1. 初始化 ACL 并设置 Device 0；
2. 创建 stream；
3. 为 A、B、Bias 和 C 分配 Host/Device 内存；
4. 从二进制文件读取输入并执行 H2D；
5. 以 `matmul_custom<<<1, 0, stream>>>` 启动一个 Cube 核；
6. 同步 stream，将 C 拷回 Host 并写入文件；
7. 按逆序释放资源。

Host 源码不存在用于逐元素矩阵计算的循环。Golden 计算仅存在于独立 Python 测试脚本中，不属于算子执行路径。

#### Kernel侧设计

Kernel 使用以下六个阶段：

1. `asc_copy_gm2l1_nd2nz` 将 A、B、Bias 从 GM 搬入 L1，并完成 ND 到 Nz 转换；
2. `asc_copy_l12l0a` 将 A 从 L1 搬入 L0A；
3. `asc_copy_l12l0b_transpose` 将非转置 B 从 L1 搬入 L0B，并生成 Cube 要求的 Zn 排布；
4. `asc_copy_l12bt` 将 Bias 从 L1 搬入 BiasTable；
5. `asc_mmad` 计算 A × B，以 BiasTable 作为 C 初值；
6. `asc_copy_l0c2gm` 通过 Fixpipe 将 L0C 搬回 GM，并完成 Nz 到 ND 转换。

`asc_mmad` 的关键参数为：

```cpp
asc_mmad(c_l0, a_l0, b_l0, 30, 40, 80, 0, true, true, false);
```

- `disable_gemv=true`：固定使用矩阵路径；
- `c_matrix_source=true`：C 初值来自 BiasTable；
- `c_matrix_init_val=false`：不将 C 清零；
- `right_width=80`：Cube 按完整对齐分形读取 L0B；真实 N=70 由最终 Fixpipe 裁剪，不能依赖 C API 自动处理非对齐尾部；
- 不调用 `asc_enable_hf32`，避免引入额外精度损失。

### 数据排布与Local Memory设计

FP32 每个 32B block 包含 8 个元素，Cube 的 M/N 基础分形维度为 16。B 的转置搬运由两个 `16×8` 分形拼成一个 `16×16` 方块，因此 N 对齐到 16。

| Buffer | 逻辑shape | 对齐后shape/大小 | 元素数 | 字节数 |
| --- | --- | --- | ---: | ---: |
| L1 A | `[30,40]` | `[32,40]` | 1280 | 5120 |
| L0A A | `[30,40]` | `[32,40]` | 1280 | 5120 |
| L1 B | `[40,70]` | `[48,80]` | 3840 | 15360 |
| L0B B | `[40,70]` | `[48,80]` | 3840 | 15360 |
| L1 Bias | `[1,70]` | 320B | 80 | 320 |
| BiasTable | `[1,70]` | 320B | 80 | 320 |
| L0C C | `[30,70]` | `[32,80]` | 2560 | 10240 |

L1 同时占用 `5120 + 15360 + 320 = 20800B`。各存储层按独立地址空间申请，静态数组容量覆盖搬运 API 可能写入的全部对齐区域。

### Bias对齐设计

Ascend 950 上 b32 数据从 C1 搬到 BiasTable 时，32B block 数需要为偶数。70 个 float 占 280B，即 9 个 block，继续向上对齐为 10 个 block，共 320B：

```cpp
asc_copy_l12bt(0, bias_l1, 320);
```

`asc_mmad` 从 BiasTable 读取 N 方向 Bias，并对所有有效 M 行使用同一组偏置值。

### 流水同步设计

| 数据 | 生产流水 | 消费流水 | Event |
| --- | --- | --- | ---: |
| A L1 | MTE2 | MTE1 | 0 |
| B L1 | MTE2 | MTE1 | 1 |
| Bias L1 | MTE2 | MTE1 | 2 |
| A L0A | MTE1 | M | 0 |
| B L0B | MTE1 | M | 1 |
| BiasTable | MTE1 | M | 2 |
| C L0C | M | FIX | 0 |

核函数退出前调用 `asc_sync_pipe(PIPE_ALL)`，保证所有异步搬运完成。

### 精度与测试设计

CPU 参考结果使用 float64 完成矩阵乘和 Bias 加法，最终一次舍入为 float32。校验标准为：

- `rtol = 2^-10`；
- `atol = 2^-16`；
- `matched_ratio >= 0.99`；
- `max_abs_error <= max(1e-2, 32 ULP)`。

设计六组精度用例：

| Case | 目的 | 输入范围 |
| --- | --- | --- |
| uniform | 覆盖均匀随机大范围输入 | `[-100,100]` |
| normal | 覆盖集中分布与裁剪边界 | `[-100,100]` |
| boundary | 覆盖 ±100、±1、±0.5、0 和抵消组合 | `[-100,100]` |
| bias_only | A/B 全零，验证 Bias 广播 | `[-100,100]` |
| non_aligned_tail | 直接命中 `A[29,39]`、`B[39,69]`、`Bias[69]`，验证最后有效分形 | `[-2,3.25]` |
| zeros | 验证全零边界 | 0 |

待验收代码提供统一验证入口：

```bash
cd examples/02_simd_c_api/03_c_api/03_matrix_compute/matmul
USE_REPO_HEADERS=0 ./run.sh
```

`run.sh` 自动完成 CMake 配置、ASC 编译、六组输入和 FP64 CPU Golden 生成、单核 NPU 执行以及精度校验。任一步骤失败均返回非零退出码；末尾汇总表列出各 case 的 matched ratio、最大误差、有效阈值和结果，全部通过时显示 `Cases: 6/6 passed` 与 `Overall result: PASS`。每个 case 的输入清单、NPU 输出、Golden 和 JSON 精度报告保存在 `output/cases/`，验收时检查六个报告的 `passed=true`、`matched_ratio>=0.99` 且 `max_abs_error<=effective_max_abs_error_limit`。

2026 年 8 月 3 日 Ascend950PR 真机结果为 6/6 通过，所有 case 的 matched ratio 均为 100%。最大绝对误差为 `0.01171875`，对应有效上限为 `0.25`；Bias-only 与 non-aligned-tail 误差均为 0。

## 支持硬件

| 支持的芯片版本 | 支持状态 |
| --- | --- |
| Ascend 950PR (`dav-3510`) | 支持，已真机验证 |
| Ascend 950DT (`dav-3510`) | 同架构编译支持，未真机验证 |

## 算子约束限制

1. 仅支持固定 shape 和 float32。
2. A/B 均不转置，GM 中为连续 ND 排布。
3. 固定单核，不支持多核分块或动态 Tiling。
4. 输入输出 Device buffer 必须满足声明的字节数。
5. 当前可复现接口基线为 CANN 9.1.0；CANN 9.0.0 任务范围歧义已提交 `asc-devkit #1404`。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | FLOAT32 混合容差、匹配率和 32 ULP 最大绝对误差规则；真机 6/6 case 通过 | 生态算子开源精度标准、任务书 |
| 性能标准 | 不适用；任务书明确“性能要求：无” | MatMul 社区任务书 |

## 兼容性分析

本实现新增独立 `matmul` 样例目录，仅在上级中英文 README 中增加索引，不修改既有 `mmad`、`mmad_mx` 样例或公共 API 行为，因此不涉及存量接口兼容性变化。

## 风险与规避

1. CANN 9.0.0-beta.2 安装包头文件未声明标准 `asc_mmad` 等接口。当前真机验证使用 9.0.0-beta.2 编译器/运行时与 `asc-devkit` master 头文件；正式验收建议在纯 CANN 9.1.0 环境复编。
2. M/N/K 存在非 16 对齐维度。实现按存储层要求扩大 Local Memory，Mmad 使用对齐后的 `right_width=80` 读取完整分形，并在最终 Fixpipe 搬运中只写出有效 `[30,70]` 区域；专用尾部 case 已验证最后有效 M/K/N 坐标。
3. Bias 搬运存在 b32 偶数 block 约束。实现将 280B 向上对齐至 320B，避免 C1 到 BiasTable 的非法搬运长度。
4. 当前实测环境使用 CANN 9.0.0-beta.2 编译器/运行时和 asc-devkit master 头文件。正式验收必须在纯 CANN 9.1.0 环境执行 `USE_REPO_HEADERS=0 ./run.sh`，并保存同一次运行的终端、NPU 状态和精度报告截图。

## 交付关联

- 设计评审：`cann/asc-devkit #1405`；
- 待验收代码：`cann/asc-devkit !4828`；
- 易用性反馈：`cann/asc-devkit #1404`；
- 个人代码分支：`gcw_NUTKvS80/asc-devkit:feat/asc-matmul-202608`；
- 算子目录：`examples/02_simd_c_api/03_c_api/03_matrix_compute/matmul`；
- 一键验证入口：`examples/02_simd_c_api/03_c_api/03_matrix_compute/matmul/run.sh`。
