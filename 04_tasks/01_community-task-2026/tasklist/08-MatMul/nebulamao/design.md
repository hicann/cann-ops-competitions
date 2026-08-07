# MatMul 算子设计文档

本文档对应 [2026 年 8 月社区任务 MatMul 算子开发任务书](../../../docs/202608/asc_matmul_task_doc.md)，并按照[社区任务算子设计文档模板](../../../resources/design_template.md)编写。

# 需求背景

## 需求来源

本需求要求基于 Ascend C C API，在 Ascend 950 上开发纯 Cube-Core、单核、固定规格的矩阵乘法样例。实现参考 `asc-devkit` 仓库的 C API `mmad` 样例，最终代码计划提交到：

```text
examples/02_simd_c_api/03_c_api/03_matrix_compute/matmul/
```

相关资料如下：

- [MatMul 社区任务书](https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/docs/202608/asc_matmul_task_doc.md)
- [Ascend C C API `mmad` 样例](https://gitcode.com/cann/asc-devkit/tree/master/examples/02_simd_c_api/03_c_api/03_matrix_compute/mmad)
- [Ascend C 算子开发文档](https://www.hiascend.com/document/detail/zh/canncommercial/850/funcguide/funcguide/funcguide_03_0001.html)
- [生态算子开源精度标准](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md)

## 背景介绍

矩阵乘法是神经网络和科学计算中的基础计算。Ascend AI Core 的 Cube 单元提供矩阵乘加能力，但输入矩阵从 Global Memory 进入 Cube 前，需要完成 ND 到分形布局的转换，以及 GM、L1、L0A/L0B、BiasTable、L0C 之间的数据搬运。计算完成后还需要通过 Fixpipe 将 L0C 中的分形结果转换回 GM 上的 ND 布局。

本任务不要求实现动态 Shape、Host Tiling 或多核分块。固定 Shape 的全部输入能够一次装入 Local Memory，因此采用单 Cube 核全量计算方案，以减少调度和分块逻辑，并把主要验证重点放在 FP32 分形布局、BiasTable 累加和尾部裁剪的正确性上。

## 开发目标

实现以下固定计算：

$$
C_{i,j}=\sum_{k=0}^{39}A_{i,k}B_{k,j}+bias_j
$$

其中：

| 参数 | 类型 | 数据类型 | Shape | 数学转置 | GM 布局 |
| --- | --- | --- | --- | --- | --- |
| `src0` / A | 输入 | FLOAT | `[30, 40]` | false | ND RowMajor |
| `src1` / B | 输入 | FLOAT | `[40, 70]` | false | ND RowMajor |
| `weight` / bias | 输入 | FLOAT | `[1, 70]` | 不适用 | ND |
| `dst` / C | 输出 | FLOAT | `[30, 70]` | 不适用 | ND RowMajor |

# 需求分析

## 功能要求

1. 使用 Ascend C C API 实现 `C = A x B + bias`。
2. A、B、bias 和 C 均为 FP32。
3. A 和 B 在数学意义上均不转置。
4. 使用 `__cube__` Kernel 和 Cube 单元执行矩阵乘加。
5. 固定启动一个 Cube Core，即 `numBlocks = 1`。
6. bias 通过 BiasTable 作为 C 的初始值参与 `asc_mmad`，不增加 Host 逐元素运算或 Vector 后处理。
7. Host 仅完成初始化、文件读写、内存管理、数据拷贝、Kernel 启动和同步。
8. 输入数据范围为 `[-100, 100]`，结果满足生态算子 FP32 精度标准。
9. 所有 Local Memory 分配均按 Cube 分形要求对齐，不发生越界访问。

## 约束与非目标

- 仅支持任务书规定的固定 Shape 和 FP32 类型。
- 不支持动态 Shape、动态数据类型、A/B 数学转置或 batch 维度。
- 不实现 Host Tiling、多核切分、Double Buffer 或跨核归约。
- Host 侧禁止使用逐元素循环完成矩阵乘法或 bias 加法。
- 测试脚本允许使用 NumPy 向量化运算生成输入和 Golden 数据。
- 任务没有性能验收门槛，首版不开启 HF32，避免降低尾数精度。

## 关键技术点

1. FP32 Cube 小分形为 `[16, 8]`，各轴需要按相应分形粒度补齐。
2. B 在数学意义上不转置，但 L1 的 Nz 布局需要转换成 L0B 的 Zn 布局。该布局转换不改变算子接口的转置语义。
3. M、N 存在尾块，`asc_mmad` 在对齐区域中产生无效结果，Fixpipe 只搬出有效的 `[30, 70]` 区域。
4. bias 的逻辑长度为 70，装载到 BiasTable 前按接口要求补齐到 320 字节。

# 详细设计

## 总体方案

Kernel 采用四阶段流水线：

```text
                        单个 __cube__ Kernel

 A GM(ND) ----> A L1(Nz) ----> L0A ----+
                                         |
 B GM(ND) ----> B L1(Nz) ----> L0B(Zn) -+--> asc_mmad --> L0C(Nz)
                                         |                    |
 bias GM ------> bias L1 ------> BT -----+                    |
                                                              v
                                                    Fixpipe NZ -> ND
                                                              |
                                                              v
                                                          C GM(ND)
```

处理顺序如下：

1. 使用 MTE2 将 A、B、bias 从 GM 搬入 L1；A、B 在搬运时由 ND 转为 Nz。
2. 使用 MTE1 将 A、B 从 L1 搬入 L0A、L0B，并将 bias 从 L1 装入 BiasTable。
3. 等待三个输入准备完成后，使用 `asc_mmad` 完成矩阵乘加。
4. 等待 Cube 计算完成后，使用 Fixpipe 将 L0C 的 Nz 结果转换为 ND 并写回 GM。

## Host 侧设计

### Kernel 接口

Kernel 接口保持四个 GM 指针，顺序与数学公式一致：

```cpp
__global__ __cube__ void matmul_custom(
    __gm__ float* src0,
    __gm__ float* src1,
    __gm__ float* weight,
    __gm__ float* dst);
```

Kernel 固定使用：

```text
numBlocks = 1
l2Ctrl    = 0
```

### Host 职责

Host 主程序执行以下步骤：

1. 初始化 ACL 并设置设备。
2. 创建 ACL stream。
3. 按固定字节数分配 Host 和 Device 内存。
4. 从二进制文件读取 A、B、bias，并执行 H2D 拷贝。
5. 启动 `matmul_custom<<<1, 0, stream>>>`。
6. 同步 stream，将 C 从 Device 拷回 Host 并写入输出文件。
7. 按初始化的逆序释放 Device、Host、stream、device 和 ACL 资源。

Host 原始数据大小如下：

| 数据 | 元素数 | 字节数 |
| --- | ---: | ---: |
| A | `30 * 40 = 1200` | 4,800 B |
| B | `40 * 70 = 2800` | 11,200 B |
| bias | `70` | 280 B |
| C | `30 * 70 = 2100` | 8,400 B |

所有 ACL 调用均检查返回码。初始化中途失败时，仅释放已经成功创建的资源并返回非零错误码。Host 代码不包含 MatMul 或 bias 的逐元素计算。

## Kernel 侧设计

### 常量与对齐

固定逻辑维度为：

```text
M = 30
K = 40
N = 70
```

FP32 小分形包含 `16 * 8 = 128` 个元素，占用 512 字节。根据 A、B 和 C 在各级存储上的布局，采用如下对齐尺寸：

| 对象 | 逻辑尺寸 | 对齐尺寸 | 元素数 | 字节数 | 布局 |
| --- | ---: | ---: | ---: | ---: | --- |
| A L1 | `30 x 40` | `32 x 40` | 1,280 | 5,120 B | Nz |
| A L0A | `30 x 40` | `32 x 40` | 1,280 | 5,120 B | Nz |
| B L1 | `40 x 70` | K 轴按 8 对齐、N 轴按 16 对齐 | 3,200 | 12,800 B | Nz |
| B L0B | `40 x 70` | K 轴按 8 对齐、N 轴按 16 对齐 | 3,200 | 12,800 B | Zn |
| bias L1 | `1 x 70` | 320 字节 | 80 | 320 B | 连续 |
| BiasTable | `1 x 70` | 320 字节 | 80 | 320 B | BT |
| C L0C | `30 x 70` | `32 x 80` | 2,560 | 10,240 B | Nz |

峰值资源占用：

| 存储区域 | 计划占用 | Ascend 950 容量 | 结论 |
| --- | ---: | ---: | --- |
| L1 | 18,240 B | 512 KB | 满足 |
| L0A | 5,120 B | 64 KB | 满足 |
| L0B | 12,800 B | 64 KB | 满足 |
| L0C | 10,240 B | 256 KB | 满足 |
| BiasTable | 320 B | 4 KB | 满足 |

实现时对上述数组大小增加编译期检查，确保对齐尺寸和目标存储容量保持一致。

### 阶段一：GM 到 L1

A 使用 `asc_set_gm2l1_nz_para` 配置 ND 到 Nz 的转换，再调用 `asc_copy_gm2l1_nd2nz` 完成搬运，逻辑行跨度为 40 个 FP32 元素。B 使用配套的 `asc_set_gm2l1_nz_para` 和 `asc_copy_gm2l1_dn2nz`，将 GM 中的 `[K,N]` row-major 数据按 DN 语义转换到可直接供 L0B 使用的 Nz 排布；逻辑行跨度为 70 个 FP32 元素，N 方向按 16 个 FP32 对齐为 80，因而 B L1 只需 `40 * 80 = 3,200` 个元素。搬运指令只读取有效 GM 区域，对齐区由设备侧搬运语义处理。

bias 作为 `[1, 70]` 的连续 FP32 数据搬入 L1。bias L1 Buffer 按 320 字节分配，超出 70 个有效元素的区域不作为数学输入。

三路搬运分别使用独立事件：

| 数据 | 生产管线 | 消费管线 | 事件 |
| --- | --- | --- | --- |
| A GM -> L1 | MTE2 | MTE1 | `EVENT_ID0` |
| B GM -> L1 | MTE2 | MTE1 | `EVENT_ID1` |
| bias GM -> L1 | MTE2 | MTE1 | `EVENT_ID2` |

### 阶段二：L1 到 L0A、L0B 和 BiasTable

A 使用 `asc_copy_l12l0a` 从 L1 Nz 搬入 L0A Nz。

B 的接口语义为不转置，即输入仍是 `[K, N]`。`asc_copy_gm2l1_dn2nz` 已将 B 转换为与 L0B 访问顺序匹配的 Nz 排布，因此 L1 到 L0B 使用不转置的 `asc_copy_l12l0b`，避免 FP32 `asc_copy_l12l0b_trans` 路径在 3510 上产生错误的 K 维分形映射。

| 参数 | 计算 | 取值 |
| --- | --- | ---: |
| `dst_offset` / `src_offset` | L1/L0B 起始地址 | 0 / 0 |
| `repeat` | K 方向分形数 | 5 |
| `src_stride` / `dst_stride` | NZ 分形步长 | 5 / 5 |
| `src_gap` / `dst_gap` | 分形间隔 | 5 / 5 |

上述参数对应 `asc_copy_l12l0b(dst, src, 0, 0, 5, 5, 5, 5)`，只复制已经在 L1 中按 B 物理布局组织的五个 K 分形。该转换仅属于物理布局处理，不改变 `transposeB=false`。

bias 使用 `asc_copy_l12bt` 从 L1 搬入 BiasTable。三路 MTE1 操作完成后分别向 Cube 计算管线发送就绪事件。

### 阶段三：Cube Mmad

`asc_mmad` 的配置意图如下：

| 参数 | 取值 | 说明 |
| --- | ---: | --- |
| `left_height` | 30 | 有效 M |
| `n_dim` | 40 | 有效 K |
| `right_width` | 70 | FP32 使用逻辑 N；只有 int8 非转置 B 的特殊布局需要额外对齐 |
| `disable_gemv` | true | 强制矩阵乘路径 |
| `c_matrix_source` | true | C 初值来自 BiasTable |
| `c_matrix_init_val` | false | 不以零初始化 C |

Cube 在内部对 bias 沿 M 方向广播，计算结果写入 L0C。实现不调用 `asc_enable_hf32()`，保持默认 FP32 Cube 计算模式。

Mmad 前等待 A、B 和 bias 三个 MTE1 完成事件；Mmad 完成后通过 `PIPE_M -> PIPE_FIX` 事件通知 Fixpipe。

### 阶段四：L0C 到 GM

使用 `asc_set_l0c2gm_nz2nd` 和 `asc_copy_l0c2gm` 将 L0C Nz 转换为 GM ND。主要参数如下：

| 参数 | 取值 | 作用 |
| --- | ---: | --- |
| `n_size` | 70 | 仅搬出有效 N |
| `m_size` | 30 | 仅搬出有效 M |
| `loop_dst_stride` | 70 | GM 输出行跨度 |
| `loop_src_stride` | 32 | L0C 对齐行跨度 |
| `nz2nd_en` | true | 启用 Nz 到 ND 转换 |

Fixpipe 不搬出 M 方向的 2 行填充和 N 方向的 10 列填充，从而保证只写入 C 的 2,100 个有效 FP32 元素。

### 同步协议

| 顺序 | 同步方向 | 目的 |
| --- | --- | --- |
| 1 | `PIPE_MTE2 -> PIPE_MTE1` | 保证 A/B/bias 已进入 L1 |
| 2 | `PIPE_MTE1 -> PIPE_M` | 保证 L0A/L0B/BT 数据就绪 |
| 3 | `PIPE_M -> PIPE_FIX` | 保证 L0C 计算完成 |
| 4 | `asc_sync_pipe(PIPE_ALL)` | Kernel 返回前排空所有流水线 |

本方案没有多 Tile Buffer 复用，因此不需要计算完成到下一轮搬运的反向复用事件，也不存在跨核同步。

## 文件组织

最终 `asc-devkit` 代码计划新增：

```text
examples/02_simd_c_api/03_c_api/03_matrix_compute/matmul/
├── CMakeLists.txt
├── data_utils.h
├── matmul.asc
├── README.md
├── README_en.md
└── scripts/
    ├── gen_data.py
    └── verify_result.py
```

同时修改矩阵计算样例上级目录的 `README.md` 和 `README_en.md`，增加 MatMul 样例索引。若评审要求直接扩展现有 `mmad/`，则保持本设计的数据通路不变，仅调整文件归属。

## 异常处理与内存安全

1. Host 对 `aclInit`、设备设置、stream 创建、内存分配、拷贝、同步和释放操作检查返回码。
2. 输入文件大小必须与固定 Shape 的字节数完全一致，读取失败或长度不符时不启动 Kernel。
3. Device Buffer 按逻辑 Shape 的精确大小分配；Local Memory 按对齐后大小分配。
4. GM 到 L1 搬运只读取逻辑有效区域，Fixpipe 只写回逻辑有效区域。
5. Local Memory 元素数通过统一的 `ceil_div`、`ceil_align` 常量表达，并使用编译期断言验证。
6. Kernel 启动和 stream 同步失败时，Host 返回错误并完成已分配资源清理。

# 支持范围

## 支持硬件与软件

| 项目 | 支持范围 |
| --- | --- |
| 硬件 | Ascend 950PR / Ascend 950DT，NPU Arch 3510 |
| 编译架构 | `dav-3510` |
| CANN | `asc-devkit/master` 的 3510 C API `mmad` 样例声明 `>= 9.1.0`；实测基线为 9.1.0 |
| 输入输出类型 | FP32 |
| Shape | A `[30,40]`、B `[40,70]`、bias `[1,70]`、C `[30,70]` |
| 核数 | 1 个 Cube Core |

2026-08-04 在 Ascend 950PR 环境使用 CANN `9.0.0-beta.2`、`dav-3510` 编译时，工具包自带头文件只暴露 MX Cube API，缺少本任务数据通路需要的 `asc_copy_gm2l1_nd2nz`、`asc_copy_l12l0a`、`asc_copy_l12l0b`、`asc_copy_l12bt`、普通 `asc_mmad` 和 `asc_copy_l0c2gm`。诊断时混用 `asc-devkit/master` 头文件后能够构建和启动，但 P03 非零乘法结果错误，说明旧编译器与新头文件组合不能代替配套的 9.1 工具链。最终兼容范围以 CANN 9.1.0 的完整编译及 NPU 结果为准；任务书中的 9.0.0 下限需要维护者进一步澄清。

## 接口约束

1. 所有输入和输出均为连续 ND RowMajor。
2. 不支持空 Tensor、非连续 Tensor、其他 Shape 或其他数据类型。
3. A、B、bias 和 C 的 GM 地址不得非法重叠。
4. 输入数据必须在 FP32 可表示范围内，精度测试数据限制在 `[-100, 100]`。
5. Kernel 固定单核启动；其他 block 数不属于支持范围。

# 可维可测分析

## 验收标准

| 验收项 | 标准 | 来源 |
| --- | --- | --- |
| 功能 | 输出等价于 `A @ B + bias`，固定 Shape 和类型与任务书一致 | MatMul 社区任务书 |
| 计算单元 | `__cube__` Kernel，单 Cube 核启动，Mmad 完成矩阵乘加 | MatMul 社区任务书 |
| 精度 | FP32 混合容差、通过率和最大绝对误差硬上限 | 生态算子开源精度标准 |
| 性能 | 无强制性能指标，仅记录基础 Kernel 耗时 | MatMul 社区任务书 |
| 内存安全 | GM 只读写逻辑区域，Local Memory 不超过对应硬件容量 | MatMul 社区任务书和 Ascend 950 资源约束 |

## 精度标准

以 FP64 NumPy 向量化计算作为高精度参考：

```python
golden = (
    src0.astype(np.float64) @ src1.astype(np.float64)
    + weight.astype(np.float64)
)
```

逐元素匹配条件为：

$$
|actual-golden| \le atol + rtol\times|golden|
$$

FP32 使用：

```text
rtol = 2^-10 = 9.765625e-4
atol = 2^-16 = 1.52587890625e-5
matched_ratio >= 0.99
max_abs_error <= 1e-2 或 max_ulp_error <= 32
```

验证脚本同时报告 matched ratio、max absolute error、max ULP error、最大误差索引、实际值和参考值，避免仅用单一错误比例掩盖局部大误差。

## 精度测试用例

所有用例固定使用 A `[30,40]`、B `[40,70]`、bias `[1,70]`：

| Case | 输入构造 | 验证目标 |
| --- | --- | --- |
| P01 | A/B 全零，bias 为 `linspace(-100,100,70)` | bias 装载和 M 方向广播 |
| P02 | A/B/bias 全零 | 零值、初始化和尾部写回 |
| P03 | 周期性填充 `-100`、`100` 和 0 | 输入边界、正负乘加 |
| P04 | 正负成对构造，使部分输出接近 0 | 消减场景的绝对误差 |
| P05 | B 使用稀疏基向量模式 | B 的布局转换和转置方向 |
| P06 | seed `20260804`，均匀随机 `[-100,100]` | 常规随机精度 |
| P07 | seed `20260805`，均匀随机 `[-100,100]` | 常规随机精度 |
| P08 | seed `20260806`，均匀随机 `[-100,100]` | 常规随机精度 |

每个用例在同一二进制上重复执行 10 次，检查结果一致性，排查未同步和未初始化问题。CANN 9.1.0 真机结果为 80/80 通过。

## 构建和运行验证

验证分三层进行：

1. Python 层检查输入、Golden 文件大小和验证器边界条件。
2. NPU 仿真模式编译和运行：`CMAKE_ASC_RUN_MODE=sim`。
3. Ascend 950 实机模式编译和运行：`CMAKE_ASC_RUN_MODE=npu`。

目标构建参数为：

```bash
cmake -DCMAKE_ASC_ARCHITECTURES=dav-3510 -DCMAKE_ASC_RUN_MODE=sim ..
cmake --build . -j
```

切换到真机模式前清理 CMake Cache，重新以 `npu` 模式配置。测试报告记录 CANN、驱动、固件、硬件型号、Git commit、构建命令和每个用例的指标。

## 性能测试

任务无性能门槛，不为提高性能启用 HF32 或增加多核分块。使用 CANN 9.1.0 `msprof` 对 P08 单次运行采集基础数据：`matmul_custom` Kernel 时长为 `6.565 us`，`Block Num=1`，`aicore_time=5.51 us`。该数据用于基线记录，不作为功能通过条件；完整 100 次计时需要在外部性能表中补充。

## 可维护性

1. Shape、对齐和字节数集中定义为编译期常量，避免魔法数字分散在搬运参数中。
2. GM 到 L1、L1 到 L0/BT、Mmad 和 Fixpipe 分别封装为短函数。
3. 中文和英文 README 使用同一组构建与运行命令。
4. 数据生成使用固定随机种子，测试结果可复现。
5. 验证脚本失败时打印可定位的误差位置并返回非零退出码，便于自动化验收。

## 兼容性分析

本实现是新增独立样例，不修改已有 `mmad` Kernel 接口和行为，因此不引入已有样例的接口兼容性风险。代码仅面向 Arch 3510 和任务规定的固定 FP32 Shape，不承诺在其他架构、其他 Shape 或其他类型上运行。

开发与验收首先以 CANN 9.1.0 为基线。只有在 CANN 9.0.0 的 Arch 3510 工具链实际编译、仿真或实机验证通过后，README 和测试报告才会声明支持 9.0.0；否则按照维护者评审结论收窄版本说明。

# 风险与评审确认项

| 编号 | 事项 | 默认处理 | 完成条件 |
| --- | --- | --- | --- |
| R01 | 任务书声明 CANN 9.0.0 至 9.1.0，但官方 3510 C API `mmad` 样例声明 `>= 9.1.0`，且 9.0.0-beta.2 编译缺失普通 Cube API | 使用 9.1.0 开发和实测，不宣称已覆盖 9.0.0 | 维护者确认验收版本，或提供可编译普通 Cube API 的 9.0.0 正式工具链 |
| R02 | 任务书要求“参考修改 `mmad.asc`”，同时交付目标允许新增样例目录 | 新增独立 `matmul/`，保留原 `mmad/` | 设计评审确认最终目录 |
| R03 | 任务书中的官方 MatMul 链接跳转到通用 API 页面 | 以 `asc_mmad` C API 样例和接口文档为实现依据 | 获得准确链接或在 Issue 中记录 |
| R04 | FP32 非转置 B 的 L1 到 L0B 参数容易与数学转置混淆 | 接口保持 `transposeB=false`，物理布局使用 Nz 到 Zn 转换 | 仿真和 P05 用例共同验证 |

确认问题后，如属于文档或接口易用性缺陷，将分别以 `【AscendC CAPI社区任务】` 前缀向 `asc-devkit` 提交可复现 Issue。

# 交付与验收标准

设计完成的判定条件如下：

1. 设计文档通过社区评审并合入 `cann-competitions`。
2. CANN 9.1.0 下 `dav-3510` 仿真和 Ascend 950 实机均可编译运行。
3. 八组精度用例全部满足 FP32 生态精度标准。
4. Kernel 固定单 Cube 核运行，Host 不包含逐元素计算。
5. README 能从干净构建目录复现数据生成、编译、运行和验证过程。
6. 自测报告包含参数、环境、精度指标、基础耗时、日志和截图。
7. Local Memory 预算、Device Buffer 大小和尾部裁剪均有代码或测试证据。
