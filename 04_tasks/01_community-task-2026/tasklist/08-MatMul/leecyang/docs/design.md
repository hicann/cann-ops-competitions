# MatMul算子设计文档

| 项目 | 内容 |
| --- | --- |
| 算子名称 | MatMul（Ascend C C API / Cube-Core 单核实现） |
| 所属任务 | 2026年8月CANN社区任务-MatMul算子开发 |
| 设计文档提交仓库 | `cann/cann-competitions` |
| 算子代码提交仓库 | `cann/asc-devkit` |
| 交付路径 | `examples/02_simd_c_api/03_c_api/03_matrix_compute/matmul` |
| 目标产品 | Ascend 950 |
| CANN版本 | 9.1.0.beta3 |

---

# 1 需求背景

## 1.1 需求来源

本需求来源于2026年8月CANN社区任务"MatMul算子开发"。任务要求基于Ascend C C API接口，使用Cube-Core单元完成带偏置的矩阵乘法，采用单核方案，不涉及Tiling切分，并以样例形式提交至`asc-devkit`仓库。

## 1.2 背景介绍

Ascend C C API提供纯C风格的接口，直接开放芯片的存储通路与矩阵计算能力，开发者需自行编排Global Memory、L1 Buffer、L0A/L0B Buffer、BiasTable Buffer与L0C Buffer之间的数据搬运及流水同步。相比高阶Matmul API，C API不做任何自动排布与Tiling推导，因此对数据分形格式和轴对齐规则的正确性要求更高。

本需求的功能语义需与Ascend C官方算子库中的`matmul`（带偏置矩阵）保持一致。

## 1.3 现状分析

`asc-devkit`现已提供如下矩阵乘相关的C API样例：

| 样例路径 | 输入类型 | 输出类型 | B矩阵来源排布 | L1→L0B接口 | Bias |
| --- | --- | --- | --- | --- | --- |
| `03_matrix_compute/mmad`（场景1） | int8_t | int32_t | ND `[K, N]`，不转置 | `asc_copy_l12l0b_trans` | 有（int32_t） |
| `03_matrix_compute/mmad`（场景2） | bfloat16_t | float | ND `[N, K]`，已转置 | `asc_copy_l12l0b` | 无 |
| `00_data_movement/data_copy_gm2l1` | half | float / int8_t | ND `[K, N]`，不转置 | `asc_copy_l12l0b_transpose` | 无 |
| `03_matrix_compute/mmad_mx` | FP4/FP8 | float | — | — | 支持三种C初值来源 |

现状可归纳为两点：

1. **缺少float输入 + 带Bias + A/B均不转置的组合。** float的C0大小、分形形状与转置搬运约束均与int8_t、half、bfloat16_t不同，现有场景的对齐常量不可直接复用。
2. **B矩阵以`[K, N]`不转置形态输入时存在两条实现路径**：`asc_copy_l12l0b_trans`（分形级转置搬运，需配置`repeat`/`frac_gap`等参数）与`asc_copy_l12l0b_transpose`（2D格式转置搬运，参数形式与`asc_copy_l12l0a`一致）。后者在`data_copy_gm2l1`样例中已有完整可运行的参数取法，且被`tensor_api`的`LoadDataL12L0BNZ2ZN`内部实现所采用，是更清晰、更易维护的路径。

## 1.4 功能分析

- 算子功能：$C_{i,j} = \sum_{k=0}^{K-1} A_{i,k} \cdot B_{k,j} + \mathrm{bias}_j$
- 输入：A、B、bias
- 输出：C
- 支持数据类型：FLOAT
- 是否支持广播：不涉及（bias沿N轴由硬件BiasTable Buffer天然按列广播）

---

# 2 需求分析

## 2.1 需求描述

使用Ascend C C API在Cube-Core上实现单核矩阵乘加算子MatMul，输入输出均为FLOAT类型，A、B矩阵均不转置，计算结果叠加偏置向量后输出。

## 2.2 需求拆解

| 编号 | 需求项 | 验收方式 |
| --- | --- | --- |
| R1 | 实现 `C[30,70] = A[30,40] × B[40,70] + bias[1,70]`，全部为FLOAT | 精度用例比对 |
| R2 | 计算全部由Cube-Core完成，禁止Host侧逐元素串行运算 | 代码走读：Host侧仅含ACL资源管理与`memcpy` |
| R3 | 单核方案，`num_blocks = 1`，不实现Tiling逻辑 | 代码走读 |
| R4 | 合理分配Local Memory，无缓冲区溢出 | 编译期常量核算 + 存储规格比对 |
| R5 | 精度满足CANN生态算子开源精度标准（FLOAT32） | `verify_result.py`判定 |
| R6 | 提供README，可复现编译、运行与验证 | 验收人复现 |

## 2.3 关键技术点识别

| 编号 | 技术点 | 说明 |
| --- | --- | --- |
| T1 | FLOAT的C0与分形规格 | `C0 = 32Byte / sizeof(float) = 8`，L0A/L0B单分形为`16 × 8 × 4Byte = 512Byte` |
| T2 | B矩阵Nz→Zn转换 | B在GM为`[K, N]`不转置，经`nd2nz`落入L1后为Nz排布，而L0B要求Zn排布，需转置搬运 |
| T3 | 32位宽转置搬运的轴对齐约束 | `asc_copy_l12l0b_transpose`在数据位宽为32时要求`k_step`为2的倍数 |
| T4 | float类型Bias通路 | GM/L1上的float bias经BiasTable Buffer作为Mmad的C矩阵初值，无需随路转换 |
| T5 | 对齐填充数据的裁剪 | 轴对齐引入的无效数据参与Mmad计算，需在Fixpipe搬出时裁剪 |

---

# 3 详细设计

## 3.1 算子分析

### 3.1.1 数学公式

$$
C_{i,j} = \sum_{k=0}^{K-1} A_{i,k} \cdot B_{k,j} + \mathrm{bias}_j,\quad i \in [0, M),\ j \in [0, N)
$$

### 3.1.2 支持数据类型

FLOAT（float32），A、B、bias、C四个输入输出Tensor数据类型一致，不支持混合精度。

### 3.1.3 支持形状

| 参数名 | 输入/输出 | 公式对应 | 数据类型 | Shape | 格式 | 是否转置 |
| --- | --- | --- | --- | --- | --- | --- |
| `src0` | 输入 | A | FLOAT | [30, 40] | ND | false |
| `src1` | 输入 | B | FLOAT | [40, 70] | ND | false |
| `weight` | 输入 | bias | FLOAT | [1, 70] | ND | - |
| `dst` | 输出 | C | FLOAT | [30, 70] | ND | - |

即 `[M, K, N] = [30, 40, 70]`，为固定Shape，不支持动态shape与广播。

## 3.2 算子实现

### 3.2.1 实现方案

#### 3.2.1.1 host侧设计

本算子为单核方案，Host侧不涉及分核与切分策略，仅完成运行时资源管理，不进行任何逐元素计算：

1. `aclInit` → `aclrtSetDevice` → `aclrtCreateStream`；
2. 为A、B、bias、C分别申请Host与Device内存（`aclrtMallocHost` / `aclrtMalloc`）；
3. 从`./input/`读入二进制输入，`aclrtMemcpy`搬运至Device；
4. 以`num_blocks = 1`启动核函数：`matmul_custom<<<num_blocks, 0, stream>>>(a, b, bias, c)`；
5. `aclrtSynchronizeStream`同步后将结果回拷至Host，写出`./output/output.bin`；
6. 逆序释放资源，`aclrtResetDevice` → `aclFinalize`。

#### 3.2.1.2 kernel侧设计

##### 1. 存储层次与数据排布

| 存储单元 | 容量（dav-3510） | 地址对齐 | 本算子中的数据排布 |
| --- | --- | --- | --- |
| Global Memory | — | — | A、B、bias、C 均为ND |
| L1 Buffer | 512KB | 32Byte | A、B 为Nz；bias 为一维连续 |
| L0A Buffer | 64KB | 512Byte | A 为Nz |
| L0B Buffer | 64KB | 512Byte | B 为Zn |
| BiasTable Buffer | 4KB | 64Byte | bias 为shape `[N]` 的一维Tensor |
| L0C Buffer | 256KB | 64Byte | C 为Nz |

FLOAT类型下的基础常量：

| 常量 | 表达式 | 取值 |
| --- | --- | --- |
| `BLOCK_CUBE` | 分形行数 | 16 |
| `C0_BYTES` | DataBlock大小 | 32 |
| `C0_ELEMENTS` | `C0_BYTES / sizeof(float)` | 8 |
| 单分形规格 | `BLOCK_CUBE × C0_ELEMENTS × 4Byte` | 16 × 8 × 4B = 512Byte |

##### 2. 流水线设计

核函数划分为四个阶段，阶段间通过`asc_sync_notify` / `asc_sync_wait`建立事件依赖，形成 `MTE2 → MTE1 → M → FIX` 的单向流水；核函数末尾调用`asc_sync_pipe(PIPE_ALL)`确保全部流水完成。

```
              PIPE_MTE2                PIPE_MTE1              PIPE_M         PIPE_FIX
  GM(ND) ──nd2nz──► L1(Nz)  ──copy_l12l0a────────► L0A(Nz) ─┐
   A                                                        │
  GM(ND) ──nd2nz──► L1(Nz)  ──copy_l12l0b_transpose► L0B(Zn)┼─► asc_mmad ──► L0C(Nz)
   B                                                        │                  │
  GM(ND) ──nd2nz──► L1      ──copy_l12bt──────────► BT ─────┘                  │
   bias                                                                        │
                                                       nz2nd + 裁剪 ◄──────────┘
                                                              ▼
                                                          GM(ND) C
```

事件分配：

| 依赖 | 事件 | 作用 |
| --- | --- | --- |
| `PIPE_MTE2 → PIPE_MTE1` | `EVENT_ID0` / `EVENT_ID1` / `EVENT_ID2` | A / B / bias 的L1数据就绪 |
| `PIPE_MTE1 → PIPE_M` | `EVENT_ID0` / `EVENT_ID1` / `EVENT_ID2` | L0A / L0B / BT 数据就绪 |
| `PIPE_M → PIPE_FIX` | `EVENT_ID0` | L0C计算结果就绪 |

##### 3. 阶段1：GM → L1（PIPE_MTE2）

使用`asc_set_gm2l1_nz_para`配置Nz目的排布参数，再由`asc_copy_gm2l1_nd2nz`完成ND到Nz的随路转换。

`config`字段编排（与`data_copy_gm2l1`样例一致）：`[15:0] matrix_num = 1`，`[31:16] dst_nz_n_stride = 1`，`[47:32] dst_nz_c0_stride`，`[63:48] loop4_dst_stride = 0`。

| 张量 | `dst_nz_c0_stride` | `loop1_src_stride` | `n_value` | `d_value` |
| --- | --- | --- | --- | --- |
| A | `ceil_align(M, 16)` = 32 | `K × 4` = 160 | `M` = 30 | `K` = 40 |
| B | `ceil_align(K, 16)` = 48 | `N × 4` = 280 | `K` = 40 | `N` = 70 |
| bias | 1 | `N × 4` = 280 | 1 | `N` = 70 |

##### 4. 阶段2：L1 → L0A / L0B / BT（PIPE_MTE1）

**A矩阵（`asc_copy_l12l0a`）**：L1中A为Nz排布的`[M, K]`矩阵，直接按2D格式搬入L0A。

| 参数 | 表达式 | 取值 |
| --- | --- | --- |
| `m_start_position` / `k_start_position` | — | 0 / 0 |
| `m_step` | `ceil_div(M, BLOCK_CUBE)` | 2 |
| `k_step` | `ceil_div(K, C0_ELEMENTS)` | 5 |
| `src_stride` | `ceil_div(M, BLOCK_CUBE)` | 2 |
| `dst_stride` | `ceil_div(M, BLOCK_CUBE)` | 2 |

**B矩阵（`asc_copy_l12l0b_transpose`）**：L1中B为Nz排布的`[K, N]`矩阵，L0B需要Zn排布，因此使用2D格式转置搬运。参数语义中的"M轴"对应源矩阵行方向（此处为K轴），"K轴"对应源矩阵列方向（此处为N轴）。

| 参数 | 表达式 | 取值 | 说明 |
| --- | --- | --- | --- |
| `m_start_position` / `k_start_position` | — | 0 / 0 | 从原点开始 |
| `m_step` | `ceil_div(K, BLOCK_CUBE)` | 3 | 单位为16个元素；位宽32时无额外倍数约束 |
| `k_step` | `ceil_align(ceil_div(N, C0_ELEMENTS), 2)` | 10 | 单位为32Byte；**位宽32时必须为2的倍数**，故9向上取整为10 |
| `src_stride` | `ceil_div(K, BLOCK_CUBE)` | 3 | L1中Nz沿列方向相邻分形的间隔，单位512Byte |
| `dst_stride` | `ceil_div(N, BLOCK_CUBE)` | 5 | L0B中Zn沿对应方向相邻分形的间隔，单位512Byte |

> **设计说明（对应技术点T3）**：`k_step = 10`意味着实际搬入L0B的列范围为`10 × 8 = 80`列，超出有效列数`N = 70`。多出的10列为对齐填充，其内容不确定，将参与Mmad计算，须在阶段4裁剪。该约束是选择`asc_copy_l12l0b_transpose`而非`asc_copy_l12l0b_trans`的直接理由之一：前者仅需保证`k_step`为偶数，无需再推导`repeat`、`dst_frac_gap`、`src_frac_gap`等分形级参数。

**bias（`asc_copy_l12bt`）**：使用高维切分搬运形式，float重载无需随路转换。

| 参数 | 表达式 | 取值 |
| --- | --- | --- |
| `dst` | BiasTable基址偏移 | 0 |
| `conv_control` | 关闭随路转换 | 0 |
| `n_burst` | — | 1 |
| `len_burst` | `BIAS_BYTES / 32` | 10 |
| `source_gap` / `dst_gap` | — | 0 / 0 |

其中 `BIAS_BYTES = ceil_align(ceil_div(N × 4, 32), 2) × 32 = 320`，即将`70 × 4 = 280Byte`按BiasTable Buffer要求的64Byte粒度向上对齐。

##### 5. 阶段3：矩阵乘加（PIPE_M）

调用Ascend 950 专有的float重载：

```cpp
asc_mmad(c_l0, a_l0, b_l0, left_height, n_dim, right_width,
         unit_flag, disable_gemv, c_matrix_source, c_matrix_init_val);
```

| 参数 | 表达式 | 取值 | 设计依据 |
| --- | --- | --- | --- |
| `left_height` | `M` | 30 | 左矩阵height |
| `n_dim` | `K` | 40 | 左矩阵width / 右矩阵height |
| `right_width` | `ceil_align(N, C0_ELEMENTS)` | 72 | 右矩阵width需按L0B实际分形对齐 |
| `unit_flag` | — | 0 | 单次Mmad，不使用与Fixpipe的细粒度并行 |
| `disable_gemv` | — | `true` | `left_height = 30 ≠ 1`，非GEMV场景，显式关闭 |
| `c_matrix_source` | — | `true` | **C矩阵初值取自BiasTable Buffer** |
| `c_matrix_init_val` | — | `false` | 不将C矩阵清零，初值由`c_matrix_source`决定 |

> **设计说明（对应技术点T4）**：通过`c_matrix_source = true` + `c_matrix_init_val = false`的组合，偏置叠加在Mmad指令内部由硬件完成，无需引入额外的向量加法，也不需要Vector Core参与，符合"纯Cube-Core实现"的约束。float bias在BiasTable Buffer上仍为float，与L0C的float输出类型匹配。


> **实测结论（HF32舍入机制验证）**：通过small用例对dav-3510上的FLOAT计算行为进行验证，结合测试结果分析当前环境下的舍入行为特征。在CANN 9.1.0.beta3环境下，dav-3510默认计算路径未启用HF32模式，无需额外调用`asc_set_fp32_mode`进行模式切换。测试中以`1.19e-07`作为判据观察FP32舍入误差特征：small用例结果符合该误差量级，表明当前实现仍按照默认FP32精度路径执行。
##### 6. 阶段4：L0C → GM（PIPE_FIX）

先由`asc_set_l0c2gm_nz2nd(1, 0, 0)`配置单矩阵的Nz→ND随路转换，再调用`asc_copy_l0c2gm`。

| 参数 | 表达式 | 取值 | 说明 |
| --- | --- | --- | --- |
| `n_size` | `N` | 70 | **按有效列数搬出，裁剪对齐填充产生的无效数据** |
| `m_size` | `M` | 30 | 按有效行数搬出 |
| `loop_dst_stride` | `N` | 70 | GM侧ND行间距 |
| `loop_src_stride` | `ceil_align(M, BLOCK_CUBE)` | 32 | L0C侧Nz分形间距 |
| `nz2nd_en` | — | `true` | 使能Nz→ND随路转换 |
| `quant_pre` / `quant_post` | `NoQuant` / `NoConv` | 0 | 不使能量化 |
| 其余控制位 | — | 0 / `false` | 不使能ReLU、Split、Eltwise、Broadcast等 |

> **设计说明（对应技术点T5）**：阶段2引入的列填充与M轴的16对齐填充均会参与Mmad计算并写入L0C，但Fixpipe按`n_size`/`m_size`裁剪后仅输出`30 × 70`的有效区域，因此无效数据不会污染最终结果。这是Cube通路上处理非对齐shape的标准做法，与`mmad`样例场景1的处理思路一致。

##### 7. 存储占用核算

| 缓冲区 | 表达式 | 元素数 | 占用 | 所在单元容量 |
| --- | --- | --- | --- | --- |
| `a_l1` | `ceil_align(M,16) × ceil_align(K,32)` | 2048 | 8KB | L1 512KB |
| `b_l1` | `ceil_align(K,32) × ceil_align(N,32)` | 6144 | 24KB | L1 512KB |
| `bias_l1` | `BIAS_BYTES / 4` | 80 | 320Byte | L1 512KB |
| `a_l0` | 同`a_l1` | 2048 | 8KB | L0A 64KB |
| `b_l0` | 同`b_l1` | 6144 | 24KB | L0B 64KB |
| `c_l0` | `ceil_align(M,16) × ceil_align(N,16)` | 2560 | 10KB | L0C 256KB |
| BiasTable | `BIAS_BYTES` | — | 320Byte | BT 4KB |

L1合计约32.3KB，各存储单元占用率均低于40%，无溢出风险。L1申请量按32元素粒度做保守对齐（与`data_copy_gm2l1`样例取法一致），代价可忽略，可避免边界分形不足导致的硬件异常。

所有缓冲区尺寸均为`constexpr`编译期常量，不存在动态分配。

## 3.3 支持硬件

| 支持的芯片版本 | 是否支持 |
| --- | --- |
| Ascend 950 | √ |

说明: 当前基于dav-3510架构实现。

## 3.4 算子约束限制

1. 仅支持FLOAT数据类型；
2. Shape固定为 A[30, 40]、B[40, 70]、bias[1, 70]、C[30, 70]；
3. A、B矩阵均不转置，输入格式均为ND；
4. 单核实现，不支持多核切分与动态shape；
5. 仅支持Ascend 950，CANN版本9.0.0 ~ 9.1.0。

---

# 4 可维可测分析

## 4.1 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | FLOAT32：`rtol = 2⁻¹⁰`，`atol = 2⁻¹⁶`，`required_matched_ratio ≥ 0.99`，`max_abs_error ≤ max(1e-2, 32 × ULP)`；逐元素判定 `\|actual − golden\| ≤ atol + rtol × \|golden\|` | [CANN生态算子开源精度标准](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md) |
| 性能标准 | 任务书未提出性能要求，提供单核kernel实测耗时作为参考数据 | 任务工作说明书 |

> **说明**：任务书要求输入数据范围为`[-100, 100]`，`K = 40`时输出量级可达`4 × 10⁵`。此量级下FLOAT32的`32 × ULP`约为`1.0`，远大于`1e-2`地板值，因此判定必须采用`max(1e-2, 32 × ULP)`形式。若沿用现网部分样例中`rtol = 1e-6`、`atol = 1e-9`的经验阈值，将显著严于生态标准，不能作为验收依据。

## 4.2 测试用例设计

| 用例编号 | 数据分布 | 取值范围 | 覆盖目的 |
| --- | --- | --- | --- |
| TC-01 | 均匀分布 | `[-100, 100]` | 任务书指定的主用例，覆盖典型量级与正负号 |
| TC-02 | 正态分布 | `μ = 0, σ = 1` | 常规数值分布 |
| TC-03 | 常量边界 | `A = 100`，`B = -100`，`bias = 100` | 量级上界与符号一致性 |
| TC-04 | 小值 | `[-1e-3, 1e-3]`，`bias = 0` | 验证`atol`分支，避免相对误差失真 |

标杆生成：以numpy在float64下完成`A × B + bias`后降精度为float32，作为更高精度的单标杆；随机用例固定seed，保证可复现。

## 4.3 自验证方案

1. **功能自验证**：先在NPU仿真模式（`-DCMAKE_ASC_RUN_MODE=sim`）下编译运行，确认无越界与流水依赖问题；再切换至NPU运行模式复测。
2. **分阶段定位手段**：将`asc_copy_l0c2gm`的`n_size`临时放大至`ceil_align(N, 16)`，把L0C全量搬出，可直接观察对齐填充区域的分布，用于区分"计算错误"与"搬运排布错误"。
3. **对照基线**：以`bias = 0`、`c_matrix_source = false`、`c_matrix_init_val = true`的配置先验证纯`A × B`通路，再打开Bias通路，逐点定位问题。
4. **性能数据采集**：使用msprof采集单次kernel执行耗时，取多次执行的中位数记入自测报告。

## 4.4 兼容性分析

本需求仅在`examples/`目录下新增样例，不新增、不修改任何对外API，不涉及`impl/`与`include/`目录，对已有功能无影响，无兼容性风险。

## 4.5 交付件清单

| 序号 | 交付件 | 路径/形式 |
| --- | --- | --- |
| 1 | 算子设计文档 | 本文档，`docs/design.md` |
| 2 | 核函数与Host侧实现 | `matmul.asc`、`matmul_impl.h` |
| 3 | 编译工程 | `CMakeLists.txt`、`data_utils.h` |
| 4 | 测试脚本 | `scripts/gen_data.py`、`scripts/verify_result.py` |
| 5 | 样例说明文档 | `README.md`、`README_en.md` |
| 6 | 自测报告 | 按官方模板提交，含用例参数、精度结果截图、性能数据截图 |
