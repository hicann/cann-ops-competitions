# MatMul（Ascend C C API，Cube-Core，单核）设计文档

## 一、需求描述

### 1.1 需求来源

基于 Ascend C C API 接口开发纯 Cube-Core 矩阵乘法算子，实现 `C = A×B + bias` 的带偏置矩阵乘计算，供开发者学习 C API 矩阵计算编程范式使用。目标交付仓为 `asc-devkit` 开源仓 `examples/02_simd_c_api/03_c_api/03_matrix_compute` 目录，面向 Ascend 950PR/950DT（DAV_3510）硬件平台，采用单核方案开发，无需处理分块（Tiling）逻辑。

### 1.2 需求分析

计算公式：`C[i,j] = Σ(k=0..K-1) A[i,k]·B[k,j] + bias[j]`

固定规格：

| 参数 | 类型 | 数据类型 | Shape | 是否转置 |
| --- | --- | --- | --- | --- |
| dst (C) | Output | FLOAT | [30, 70] | - |
| src0 (A) | Input | FLOAT | [30, 40] | false |
| src1 (B) | Input | FLOAT | [40, 70] | false |
| weight (bias) | Input | FLOAT | [1, 70] | - |

即 M=30, K=40, N=70，A、B 均不转置，带 FLOAT bias。

**与官方参考样例 `mmad` 的关系**：参考样例固定使用 M=30,N=40,K=70，与本任务的 M=30,K=40,N=70 在数值上部分重合，但 K 与 N 的角色分配不同。样例场景1（int8+bias，B 数学语义与物理存储均不转置，L1→L0B 搬运使用 `asc_copy_l12l0b_trans`）与场景2（bf16，无 bias，B 数学语义不转置但物理存储按转置字节序写入，L1→L0B 搬运使用 `asc_copy_l12l0b`）均不能直接套用于 FLOAT 场景（各自的搬运公式与 FLOAT 的 C0_SIZE 关系不完全匹配，详见 §2.1.6）。本设计采用"场景2的 B 转置存储方式 + 非-`_trans` 搬运路径 + 场景1的 Bias 叠加机制"的组合方案，数据类型全部替换为 FLOAT。

## 二、方案设计

### 2.1 接口内部实现

#### 2.1.0 与官方 matmul 实现的对齐

官方算子库中对应的 matmul 实现是 `aclnnMatmul`（`ops-nn` 仓 `matmul/mat_mul_v3/docs/aclnnMatmul.md`）。两者关系需分两个层面理解：

**编程模型层面**：`aclnnMatmul` 是标准 aclnn 两段式算子接口（`aclnnMatmulGetWorkspaceSize` + `aclnnMatmul`，基于 workspace/executor 机制），本任务实现的是底层 C API 指令级封装（直接操作 L0A/L0B/L0C 缓存的 `asc_mmad` 指令），对齐的是官方参考样例 `mmad` 的工程范式，两者不是同一层级，不存在直接的接口对齐关系。`aclnnMatmul` 本身也没有 bias 参数（计算公式仅为 `result = self @ mat2`），本设计的 Bias 叠加方案（§2.1.5）依据官方样例场景1的 BiasTable 机制设计。因此"对齐"体现在数学语义层面（矩阵乘计算结果一致）。

**精度模式层面**：`aclnnMatmulGetWorkspaceSize` 的 `cubeMathType` 参数用于控制精度降级行为：`cubeMathType=0（KEEP_DTYPE）`表示保持输入数据类型计算（不降级）；`cubeMathType=1/3` 表示 FLOAT32 转换为 HFLOAT32 计算（降级换性能）。`aclnnMatmul` 官方调用示例本身使用的是允许降级的 `cubeMathType=1`，即该接口并不存在固定的官方默认精度模式，精度模式由调用方按场景选择。本设计在 §2.1.4 的决策——不调用 `asc_enable_hf32()`，依赖硬件默认全精度行为——对应的是 `KEEP_DTYPE` 这一官方文档定义的合法精度模式，是基于任务精度要求（生态精度标准存在实际数值门槛）做出的选择，而非复刻官方默认行为（因为并不存在这样一个默认行为）。矩阵乘的数学语义结果一致，是任何正确实现都应满足的基本要求，本身不构成额外的"对齐"证据。

#### 2.1.1 FLOAT 全链路 Cube-Core 支持

Cube-Core 原生支持 FLOAT 输入的矩阵乘，是官方在 950PR 上正式测试过的路径，不需要 bf16 拆分或数值变通方案：

1. `cube_compute.h` 中存在 `asc_mmad(__cc__ float* c_matrix, __ca__ float* a_matrix, __cb__ float* b_matrix, ...)` 及其 `_sync` 版本；
2. 官方单元测试 `tests/api/c_api/npu_arch_3510/cube_compute/test_mmad.cpp` 第103行：`TEST_CUBE_COMPUTE_MMAD_INSTR(float, float, float, float);`——`npu_arch_3510` 即 950PR/DAV_3510；
3. 全链路数据搬运（`cube_datamove.h`）均有 FLOAT 重载：`asc_copy_gm2l1_nd2nz`（GM→L1）、`asc_copy_l12l0a`/`asc_copy_l12l0b`（L1→L0A/L0B）、`asc_copy_l0c2gm`（L0C→GM）、`asc_copy_l12bt`（L1→BiasTable，含 `__cbuf__ float* src` 重载）。

#### 2.1.2 FLOAT 数据的分形（Fractal）格式与对齐参数

依据官方指南《Cube矩阵计算编程.md》："L1 Buffer：分形形状为 16×(32B/sizeof(DataType))"。代入 FLOAT（4字节）：16×(32/4) = 16×8。

```
S3_C0_SIZE = 8                                      // 32B / sizeof(float) = 8，mmad_s3.h 中定义的真实常量
分形块大小 = BLOCK_CUBE * S3_C0_SIZE = 16 * 8 = 128    // 推导数值，未在代码中定义为独立命名常量，
                                                     // 体现在 §2.2.3 各 Buffer 尺寸公式的展开项里
```

`S3_C0_SIZE=8` 与官方样例场景1（int8, C0_SIZE=32）、场景2（bf16, C0_SIZE=16）的既有取值规律（32/sizeof(dtype)）完全吻合。

#### 2.1.3 `asc_mmad` 重载中位置参数语义

`asc_mmad` 存在两组签名：一组是 2201/3510 共用的旧签名（位置参数为 `k_direction_align`，影响 `right_width` 方向对齐）；另一组是仅 950PR/950DT 支持的新签名（位置参数为 `disable_gemv`，用于控制 `left_height=1` 时是否启用 GEMV 特殊路径，与 K/N 方向对齐无关）。

官方样例 `mmad_s1.h`、`mmad_s2.h` 调用 `asc_mmad` 时该位置参数命名为 `disable_gemv` 并设为 `true`，说明走的是 950PR 专属重载。本设计 M=30（不等于1），不涉及 GEMV 场景，沿用样例做法设 `disable_gemv = true`；该参数的命名与语义依据是对官方样例源码的直接阅读——M≠1 时 GEMV 分支本身不会被触发，因此真机测试结果本身无法反向验证这一语义理解，语义判断的唯一依据是源码。

#### 2.1.4 FP32 计算精度模式

`asc_set_fp32_mode()`：开启后 L0A/L0B 中的 FP32 数据在参与 Mmad 计算前不做舍入处理，即全精度路径；`asc_enable_hf32()`：开启后 FP32 数据会被舍入为 HF32 格式参与计算以换取性能提升，存在精度损失。两者互斥可选，任务性能要求为无，但精度要求引用了生态算子开源精度标准（FLOAT32：rtol=9.77e-4, atol=1.53e-5），有实际数值门槛。

真机 CANN 9.1.0-beta.3 环境下 `asc_set_fp32_mode()` 编译报错 `use of undeclared identifier`。排查确认：该函数在 `asc-devkit` 代码仓库层面对 950PR 的支持（声明、专属实现文件 `asc_set_fp32_mode_impl.h`、官方单测 `test_set_fp32_mode.cpp`）已完整落地，但已安装的 CANN 9.1.0-beta.3 软件包版本滞后于代码仓库，尚未把这部分950PR支持一并打包发布。

**最终决策**：不调用 `asc_set_fp32_mode()`，也不调用 `asc_enable_hf32()`，依赖硬件默认精度行为。官方文档未正面说明这一默认行为的具体机制（是否等同于全精度、或某种未声明的中间状态），该决策的可行性是基于真机测试结果的间接验证：§4.1 记录的连续 4 轮随机数据测试 `matched_ratio` 均为 1.000000，`max_abs_error` 远低于标准有效上限，说明硬件默认行为在实测范围内已能满足任务精度标准。

#### 2.1.5 Bias 叠加方案

任务规格 bias（weight）为 `[1,70] FLOAT`。`asc_copy_l12bt` 存在 `__cbuf__ float* src` 直接进 BiasTable 的重载，采用与官方样例场景1一致的 BiasTable 机制（而非另建 Vector Add 叠加方案），仅将 dtype 由 int32 替换为 float：GM(bias, float) → L1(float) → BT(float) → `asc_mmad` 调用时设 `c_matrix_source = true`（C矩阵初值来自 BiasTable）。`c_matrix_init_val` 设为 `false`，含义是"是否将 C 矩阵初值清零"，与 `c_matrix_source` 是两个独立开关；本设计每次调用都是全新计算且已通过 `c_matrix_source=true` 取到 bias 初值，故不需要额外清零。

#### 2.1.6 B 矩阵搬运路径的实现选择

任务规格 B "是否转置=false"，数学语义上 B 是 `[K,N]=[40,70]` 的自然（非转置）矩阵。官方两个参考场景对 B 的处理方式不同：

- 场景1（int8，B 数学语义上也不转置）在 L1→L0B 阶段使用 `asc_copy_l12l0b_trans`，其地址算术依赖 `S1_FRACTAL_NUM = BLOCK_CUBE*FRACTAL_NUM / C0_SIZE`，与 int8 的 `C0_SIZE=32 > BLOCK_CUBE=16` 关系绑定；
- 场景2（bf16，B 数学语义上转置存储）在 L1→L0B 阶段使用普通的 `asc_copy_l12l0b`（无需 `_trans`）。

FLOAT 的 `C0_SIZE=8 < BLOCK_CUBE=16`，与场景1的比例关系相反，直接照搬场景1的 `_trans`/`FRACTAL_NUM` 公式存在推导错误风险（属于硬件底层分形寻址细节，错误不会导致编译失败，而是产生静默的数值错误）。

**实现选择**：让 kernel 内部按"B 已转置存储"的方式读取 GM 数据（`gen_data.py` 写 `x2_gm.bin` 时，将逻辑上的 `B[40,70]` 按 `[70,40]`（B 的转置）字节顺序物理写入文件），从而复用官方场景2已验证正确的非转置搬运路径（`asc_copy_gm2l1_nd2nz` + `asc_copy_l12l0b`，不使用 `_trans`），只将 dtype 由 bfloat16_t 替换为 float。

该选择有明确的正面技术依据：真机 `asc_copy_l12l0b_impl.h` 源码显示，非-`_trans` 版本的 `asc_copy_l12l0b`（float 重载）函数体只有一行，直接调用硬件指令 `load_cbuf_to_cb`，把调用方传入的步长/偏移参数原样转发，函数体内部不包含任何与 C0_SIZE/BLOCK_CUBE 比例关系绑定的硬编码公式，是按 dtype 均匀参数化的通用封装，正确性完全取决于调用方（`mmad_s3.h`）传入的参数是否按 FLOAT 的 `S3_C0_SIZE=8` 正确计算。

该选择不改变算子对外的数学语义：验证阶段 Python 侧生成 golden 数据仍按逻辑形状 `A[30,40]`、`B[40,70]`（不转置）计算 `A@B+bias`；`x2_gm.bin` 的物理字节序是内部实现细节，只需保证 kernel 内部读取方式与写入方式严格对应。

#### 2.1.7 完整数据流

三路输入分别搬运至 L0A/L0B/BiasTable，经 `asc_mmad` 计算后搬出至 GM：

1. **A**：GM（float，物理ND[30,40]）→ L1（float，Nz排布，M方向外）→ L0A（float，Nz排布，`asc_copy_l12l0a` 非转置搬运），对齐官方样例场景1/2共同的 A 处理方式；
2. **B**：GM（float，物理ND[70,40]，即 B 的转置字节序）→ L1（float，Nz排布，N方向外）→ L0B（float，Zn排布，`asc_copy_l12l0b` 非转置搬运）。因 §2.1.6 所述理由，在 GM 侧以转置字节序存储，搬运时对齐官方样例场景2的非转置路径；
3. **bias**：GM（float，ND[1,70]）→ L1（float）→ BiasTable（float，`asc_copy_l12bt`），采用官方样例场景1的 BiasTable 机制；
4. L0A/L0B/BiasTable 就位后，执行 `asc_mmad(float×float→float, disable_gemv=true, c_matrix_source=true, c_matrix_init_val=false)`，结果写入 L0C（float，Nz排布）；
5. L0C → `asc_copy_l0c2gm`（NZ2ND 转换）→ GM（C，float，ND[30,70]）。

未调用 `asc_set_fp32_mode()`，最终实现依赖硬件默认精度行为，已通过 §4.1 精度验证确认可行。

### 2.2 接口设计

#### 2.2.1 Kernel 侧接口

沿用官方样例 `mmad.asc` 的核函数入口签名规范，新增场景分支（记为场景3）：

```cpp
__global__ __cube__ void matmul_custom(__gm__ uint8_t* a, __gm__ uint8_t* b, __gm__ uint8_t* bias, __gm__ uint8_t* c);
```

将占位调用 `mmad_custom<<<num_blocks, 0, stream>>>(...)` 替换为 `matmul_custom<<<num_blocks, 0, stream>>>(...)`。

#### 2.2.2 Host 侧接口

单核方案，无需处理 Tiling，Host 侧不引入 Tiling 结构体，仅保留官方样例既有的 ACL 资源申请/释放、数据读写主函数结构（`main` 函数），维度信息（M=30, K=40, N=70）以编译期常量形式直接写入 kernel，不做运行时动态推导。

#### 2.2.3 内存安全核查

本任务的强制约束共三条：①必须用 Cube-Core 单元完成计算（§2.1.1）；②禁止在 Host 侧用 for 循环逐元素运算——`mmad.asc` 的 `main()` 函数仅做 ACL 资源申请/释放、数据读写，不存在逐元素循环，计算全部在 Device 侧 kernel 完成；③内存安全（Local Memory 无越界），详见下文。

Local Memory（L1/L0A/L0B/L0C/BT）分配大小：

```
S3_A_L1_SIZE = ceil_align(M, BLOCK_CUBE) * ceil_align(K, S3_C0_SIZE)   // 32*40=1280
S3_A_L0_SIZE = ceil_align(M, BLOCK_CUBE) * ceil_align(K, S3_C0_SIZE)   // 1280
S3_B_L1_SIZE = ceil_align(N, BLOCK_CUBE) * ceil_align(K, S3_C0_SIZE)   // 80*40=3200
S3_B_L0_SIZE = ceil_align(K, S3_C0_SIZE) * ceil_align(N, BLOCK_CUBE)   // 40*80=3200
S3_C_L0_SIZE = ceil_align(M, BLOCK_CUBE) * ceil_align(N, BLOCK_CUBE)   // 32*80=2560
S3_BIAS_BYTES = ceil_align(ceil_div(N * sizeof(float), 32), 2) * 32    // 10*32=320
S3_BIAS_ELEMS = S3_BIAS_BYTES / sizeof(float)                          // 320/4=80
```

对照官方架构规格文档《NPU架构版本3510.md》给出的 950PR 精确硬件容量，逐字节核对占用比例：

| 缓冲区 | 本设计占用（float，4字节/元素） | 硬件容量 | 占用比例 |
| --- | --- | --- | --- |
| L1（A+B+bias 共享） | 5120+12800+320=18240B | 524288B (512KB) | 3.48% |
| L0A | 5120B | 65536B (64KB) | 7.81% |
| L0B | 12800B | 65536B (64KB) | 19.53% |
| L0C | 10240B | 262144B (256KB) | 3.91% |
| BiasTable | 320B | 4096B (4KB) | 7.81% |

五项占用比例均远低于100%，最高占用为 L0B 的19.53%，内存安全性已逐字节核对确认。

### 2.3 测试用例设计

| 用例编号 | 测试项 | 说明 | 真机实测结果 |
| --- | --- | --- | --- |
| T-001 | 固定规格功能正确性 | A[30,40]/B[40,70]/bias[1,70]，数据范围[-100,100]随机生成，与 CPU numpy 参考实现比对 | 连续4轮，matched_ratio=1.000000，详见 §4.1 |
| T-002 | 精度达标验证 | 生态算子开源精度标准 FLOAT32 档：rtol=9.77e-4, atol=1.53e-5, matched_ratio≥0.99, max_abs_error_limit=max(1e-2, 32×ULP) | 已通过，详见 §4.1 |
| T-003 | bias 为全零 | 退化验证：bias=0 时结果应等于纯矩阵乘 A×B | matched_ratio≥0.999，test pass，详见 §4.4 |
| T-004 | bias 含负值 | 验证 bias 叠加方向正确 | matched_ratio≥0.999，test pass，详见 §4.4 |
| T-005 | 边界数值 | 数据范围边界 ±100 附近取值，验证无数值溢出/精度异常 | matched_ratio=1.000000，test pass，详见 §4.4 |
| T-006 | Local Memory 越界检查 | 对照官方架构规格文档给出的精确硬件容量，逐一核对 L1/L0A/L0B/L0C/BT 五个缓冲区占用比例 | 全部远低于100%（最高19.53%），详见 §2.2.3 |
| T-007 | 性能测试case | 独立 `perf_test.asc`（复用 `mmad_s3.h` 流水线，不改动正式交付文件），`aclrtEvent` 打点连续20次调用 + `std::chrono` 记录整进程壁钟耗时对比 | kernel设备端均值0.0041~0.0044ms，壁钟耗时6.3~6.4秒，比值约145~155万倍，详见 `perf_test/README_perf_test.md` |

测试脚本基于官方样例 `scripts/gen_data.py` / `scripts/verify_result.py` 修改而来，T-003/T-004/T-005 专项数据由 `gen_edge_cases.py` 单独生成，T-007 性能测试独立成 `perf_test/` 目录。

## 三、可维可测

### 3.1 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | FLOAT32 档：rtol=9.77e-4, atol=1.53e-5, matched_ratio≥0.99, max_abs_error_limit=max(1e-2, 32×ULP)；数据范围[-100,100]。已在 §4.1、§4.4 真机验证通过 | 《生态算子开源精度标准》 |
| 性能标准 | 无达标门槛，已按要求补充性能测试case（T-007）产出真实耗时数据供参考 | - |

### 3.2 兼容性分析

本任务为 `asc-devkit` 仓库 `examples/` 目录下新增的教学样例，不修改场景1/2既有实现文件 `mmad_s1.h`/`mmad_s2.h`，也不改变其行为；仅在共享入口文件 `mmad.asc` 中以 `if constexpr` 新增互不影响的场景3分支，并新增独立的 `mmad_s3.h` 实现文件，不涉及存量代码兼容性问题。仅支持 Ascend 950PR/950DT（DAV_3510），与官方 `mmad` 样例现有支持范围一致。

### 3.3 测试用例的可维护性

`gen_data.py`（场景3分支）、`verify_result.py`（场景3分支）、`gen_edge_cases.py` 是随本任务新增的独立 Python 脚本，随 `mmad_s3.h`/`mmad.asc` 一并作为教学样例的配套测试代码提交，供后续维护者复现验证。与场景1/2一致，暂不接入官方 CI 回归测试体系（教学样例目录下的既有场景同样未见配套官方 CI 单测）。

## 四、自验证记录（950PR 真机，CANN 9.1.0-beta.3）

### 4.1 编译与执行结果（T-001/T-002 固定规格随机数据）

代码在真机 950PR 环境编译通过、执行通过。连续4次使用不同随机数据（数据范围[-100,100]）验证：

| 轮次 | matched_ratio（要求≥0.99） | max_abs_error | 有效上限 max(1e-2, 32×ULP) | 结论 |
| --- | --- | --- | --- | --- |
| 1 | 1.000000 | 1.5625e-2 | 2.5e-1 | test pass! |
| 2 | 1.000000 | 1.5625e-2 | 2.5e-1 | test pass! |
| 3 | 1.000000 | 1.171875e-2 | 1.25e-1 | test pass! |
| 4 | 1.000000 | 1.171875e-2 | 1.25e-1 | test pass! |

matched_ratio 连续4次均为 1.000000，远超精度要求。

### 4.2 内存安全核查

已从官方架构规格文档《NPU架构版本3510.md》获取 950PR 精确硬件容量（L1=512KB、L0A=64KB、L0B=64KB、L0C=256KB、BT=4KB），逐字节核对本设计五个缓冲区占用比例（详见 §2.2.3），全部远低于硬件上限。补充佐证：代码在真机连续多次执行均无越界保护异常、无 ACL 运行时错误、无 crash；T-003~T-005 边界场景同样连续运行无越界异常。

### 4.3 易用性问题记录

开发过程中发现以下两处已安装 CANN 9.1.0-beta.3 软件包与 `asc-devkit` 代码仓库版本不同步的问题（软件包尚未打包发布代码仓库已完整落地的950PR支持）：

1. **`asc_set_fp32_mode()`**：950PR 对应的头文件分支未声明该函数，但 `asc-devkit` 代码仓库自带头文件的声明不受架构限制，且带有 950PR 专属实现文件与官方单测，说明支持已在代码仓库层面完整落地；
2. **`ASC_L1_SIZE`/`ASC_L0A_SIZE`/`ASC_L0B_SIZE`/`ASC_L0C_SIZE`/`ASC_BT_SIZE`**：已安装软件包解析到的头文件不含这批常量定义，但 `asc-devkit` 代码仓库自带的同名文件按架构分支定义了这批常量，与官方架构规格文档数值一致。

### 4.4 边界与退化场景专项验证（T-003/T-004/T-005）

另编写 `gen_edge_cases.py` 生成三组专项数据，在 950PR 真机运行并验证（两轮独立随机数据交叉验证）：

| 用例 | 数据构造方式 | matched_ratio（要求≥0.99） | max_abs_error / 有效上限 | 结论 |
| --- | --- | --- | --- | --- |
| T-003（bias全零） | A/B 随机[-100,100]，bias=全零 | 0.999~1.000（两轮） | 均远低于有效上限 | test pass! |
| T-004（bias负值专项） | A/B 随机[-100,100]，bias 全部取[-100,-50]区间的负值 | 0.999~1.000（两轮） | 均远低于有效上限 | test pass! |
| T-005（边界±100） | A/B/bias 均从 {-100, -99.99, 100, 99.99} 等边界离散值中采样 | 1.000000（两轮） | max_abs_error=1.5625e-2，有效上限=5.0e-1 | test pass! |

T-003、T-004 两轮实测的 matched_ratio 在 99.90%~100% 之间浮动，均远超 0.99 门槛：唯一未通过混合容差判据的元素对应绝对值很小的输出值，此时相对容差本身很小，float32 累加产生的正常舍入误差相对该极小基准值显得"超差"，但绝对误差本身处于 float32 正常精度范围内；生态算子开源精度标准将 `matched_ratio` 门槛设为 0.99（而非 100%）正是为了容纳这类情形。T-004 的高匹配率同时确认了 Bias 叠加方向正确（若叠加方向有误，在 bias 全为负值的构造下会产生系统性大幅偏移，matched_ratio 会大幅跌落）。T-005 验证了在 A、B、bias 均取 ±100 附近极值、K=40 维度累加可达到 ±24万量级的场景下，未出现数值溢出或 NaN/Inf 异常。
