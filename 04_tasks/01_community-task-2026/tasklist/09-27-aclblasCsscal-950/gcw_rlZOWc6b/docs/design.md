# aclblasCsscal 算子设计文档

# 需求背景（required）

## 需求来源

aclblasCsscal 算子来源于昇腾算子开源社区任务，由 CANN 团队在 ops-blas 开源仓（https://gitcode.com/cann/ops-blas）发起。

当前 ops-blas 仓中：

- `aclblasSscal`：实数标量 × 实数向量 — **已有**（支持 Ascend 950PR）
- `aclblasCscal`：复数标量 × 复数向量 — **已有**（不支持 Ascend 950PR）
- `aclblasCsscal`：实数标量 × 复数向量 — **缺失**（本任务新增）

## 背景介绍

BLAS（Basic Linear Algebra Subprograms）是线性代数计算的工业标准接口，其中 Level 1 向量缩放算子用于将向量乘以标量。

在复数向量缩放场景中，存在两种变体：

- **Cscal**（复数标量 × 复数向量）：`x = α × x`，alpha 为复数
- **Csscal**（实数标量 × 复数向量）：`x = α × x`，alpha 为实数

对标 cuBLAS 和 Netlib BLAS 实现，csscal 是独立接口，需单独开发。

### aclblasCsscal 算子功能分析

aclblasCsscal 算子功能：`x[j] = α × x[j]`（复数向量原地缩放）

| 参数     | 参数含义        | 数据类型            | 支持数据类型    | 约束               |
| ------ | ----------- | --------------- | --------- | ---------------- |
| handle | ops-blas 句柄 | scalar          | -         | 非空               |
| n      | 复数元素个数      | int             | -         | n ≤ 0 为 no-op    |
| alpha  | 实数标量        | const float*    | float32   | n > 0 时非空        |
| x      | 复数向量        | aclblasComplex* | complex64 | n > 0 时非空，原地更新   |
| incx   | 步长          | int             | -         | incx ≤ 0 为 no-op |

计算公式：

```
x[idx].real = x[idx].real * alpha
x[idx].imag = x[idx].imag * alpha
idx = i * incx（跳步访问）
```

# 需求分析（required）

## 需求描述

使用 Ascend C 编程语言在 Ascend 950PR 上实现 aclblasCsscal 算子：

- 功能对标 cuBLAS `cublasCsscal` 和 Netlib BLAS `csscal`
- 支持 float32 实数标量乘以 complex64 复数向量
- 支持 incx 跳步访问
- 满足生态算子开源精度标准

## 需求拆解

1. **功能实现**
   
   - 支持 n ≤ 0、incx ≤ 0 的 no-op 语义
   - 支持 alpha = 0.0 置零操作（非 no-op）
   - 支持 alpha = 1.0 提前返回优化
   - 支持 incx != 1 跳步访问

2. **性能要求**
   
   - n=1048576, incx=1: ≤ 13.57 μs
   - n=2097152, incx=1: ≤ 21.05 μs
   - n=4194304, incx=1: ≤ 43.02 μs

3. **精度要求**
   
   - rtol = 2^-10 ≈ 9.77e-4
   - atol = 2^-16 ≈ 1.53e-5
   - matched_ratio ≥ 0.99
   - max_abs_error ≤ 1e-2 或 32×ULP

## 约束限制

- 不支持动态 shape（n 为运行时入参）
- 不支持 broadcast（向量缩放无 broadcast 语义）
- 不支持负步长（incx < 0 为 no-op）
- alpha=0.0 时为置零操作，与 axpy 的 alpha=0 no-op 语义不同

# 详细设计（required）

## 算子分析

### 数学公式

```
for i = 0 to n-1:
    idx = i * incx
    x[idx].real = x[idx].real * alpha
    x[idx].imag = x[idx].imag * alpha
```

### 支持数据类型

| 类型    | 说明                        |
| ----- | ------------------------- |
| alpha | float32 实数标量              |
| x     | complex64（实部/虚部各 float32） |

### 算子语义

| 条件                   | 行为               | 说明         |
| -------------------- | ---------------- | ---------- |
| n ≤ 0                | no-op，返回 SUCCESS | 不修改 x      |
| incx ≤ 0             | no-op，返回 SUCCESS | 不修改 x      |
| alpha = 1.0          | 可提前返回            | 数值结果一致     |
| alpha = 0.0          | **置零操作，非 no-op** | x 的实部虚部均置零 |
| n > 0, alpha=nullptr | 返回 INVALID_VALUE |            |
| n > 0, x=nullptr     | 返回 INVALID_VALUE |            |

## 算子实现

### 实现方案

#### 3.2.1 host侧设计

**Tiling 策略：**

根据 incx 分为两种路径：

**路径一：incx == 1（连续访问，AIV SIMD 模式）**

```
numBlocks = min(ceil(n / 256), aivCoreNum)     // 小向量避免启动大量空闲核
perCoreN = (n / numBlocks) 向下对齐到 CSSCAL_ALIGN_UNIT(8)
remainder = n - perCoreN * numBlocks
tileSize = min(CSSCAL_TILE_CAPACITY, 最重核实际需求向上对齐到 8)
// CSSCAL_TILE_CAPACITY = min(CSSCAL_MAX_TILE(25008), UB 容量上限 31744)
```

- 单块 UB 原地处理（DataCopy → 寄存器乘 → DataCopy），不使用 TPipe 队列
- 每个核处理连续的 `perCoreN` 个复数元素
- 最后一个核额外处理 `remainder` 个元素

**路径二：incx != 1（跳步访问，SIMT 模式）**

```
useCoreNum = min(n, CSSCAL_MAX_CORE_NUM)
baseCount = n / useCoreNum
remain = n % useCoreNum
calCount[i] = baseCount + (i < remain ? 1 : 0)
nthreads = min(ceil(n/useCoreNum) 对齐到 32, 256)
```

- 使用 SIMT 模式，每个线程处理一个复数元素
- 线程通过 grid-stride loop 遍历非连续元素
- 支持任意正步长 incx

**分核策略：**

优先使用满核原则：

- 如果核间能均分，各核数据块大小一致
- 如果核间不能均分，余出的数据块分配到前几个核

#### 3.2.2 kernel侧设计

**AIV 模式（incx == 1）：**

```
CsscalContiguous(x, perCoreN, remainder, tileSize, alpha):
    count <= tileSize: 单片处理
    否则: 完整块（编译期定长）循环 + 尾片独立处理
CsscalTile:
    CopyIn:  DataCopy/DataCopyPad 从 GM 加载到 UB（偏移 0 锚定，无 TPipe）
    Scale:   Reg::LoadAlign(DINTLV_B32 双载) → Reg::Muls → Reg::StoreAlign(INTLV_B32 双存)
             （一条指令搬两个向量并解/再交织实虚部；完整向量复用满掩码，
               仅尾向量生成部分掩码；加载用硬件 post-update 寻址）
    CopyOut: DataCopy/DataCopyPad 从 UB 写回 GM
    片间以固定 EVENT_ID0 的 MTE2_V / V_MTE3 / MTE3_MTE2 事件同步，
    末尾 MTE3_S 排空最终写回
```

寄存器乘法对复数的实部和虚部分别乘以 alpha（FP32，与参考实现一致）。

**SIMT 模式（incx != 1）：**

```
CsscalSimtCompute(calNum, startOffset, stride, alpha, xGm):
    for i = threadIdx.x; i < calNum; i += blockDim.x:
        idx = (startOffset + i) * stride   // uint64 计算，防大步长溢出
        floatIdx = idx * 2
        xGm[floatIdx]     = alpha * xGm[floatIdx]
        xGm[floatIdx + 1] = alpha * xGm[floatIdx + 1]
```

线程通过 grid-stride loop 遍历跳步元素。

#### 3.2.3 数据流图

```
Host (aclblasCsscal)
    │
    ├── 参数校验 (n≤0, incx≤0, nullptr检查)
    ├── Tiling 计算 (incx==1? AIV : SIMT)
    └── kernel launch (csscal_kernel_do)
            │
            ├── incx == 1: csscal_aiv_kernel
            │       CopyIn → Muls(α) → CopyOut
            │
            └── incx != 1: csscal_simt_kernel
                    grid-stride loop: x[i*incx] *= α
```

## 支持硬件

| 支持的芯片版本                     | 涉及勾选 |
| --------------------------- | ---- |
| Ascend 950PR / Ascend 950DT | √    |

## 关键优化

1. **多核并行**：按复数元素切分到多个核；小向量按 `min(ceil(n/256), 核数)` 避免空闲核
2. **单块 UB 原地计算**：去掉 in/out 双队列与 TPipe 管理，同 tile 下实测与多缓冲等效且启动开销最小
   （double buffer / 软件流水对本算子零收益，有控制变量对照实验，见根目录 README §四）
3. **DINTLV/INTLV_B32 双载双存**：向量加载/存储用 dav_3510 的 dual 形式，一条 vlds/vsts 搬运两个
   向量并以 32bit 粒度解/再交织（实部、虚部分落两个寄存器，对均一实数 alpha 语义精确等价）。
   相对单寄存器形式：载存指令数减半，且两个独立寄存器对打破 load→muls→store 的 WAR 串行链，
   4M 用例 vec 管耗时 10.7 → 4.8 μs；加载再叠加硬件 post-update 寻址，指针推进零标量开销
4. **Reg 满掩码复用**：完整向量共用满掩码，仅尾向量生成部分掩码；完整块长度编译期固定
5. **32字节对齐**：按 8 个 float（8 复数）对齐，尾段走 DataCopyPad
6. **SIMT 跳步**：incx != 1 时使用 grid-stride loop 支持任意步长，地址计算 64 位防溢出
7. **MAX_TILE=25008 等长切分**：4M 用例最重核负载 75024 = 3×25008，走 3 片等长编译期定长块；
   扫描 0/28672/24576/16384/25008 五点后实测 4M 再降 2.4%（详见根目录 README §6 扫描表与死路清单）

## 算子约束限制

- n ≤ 0 或 incx ≤ 0 为合法 no-op（返回 SUCCESS，不修改 x）
- alpha = 0.0 为置零操作（非 no-op），与 axpy 语义不同
- alpha = 1.0 允许提前返回优化
- 不支持负步长（incx < 0 为 no-op）
- 不支持动态 shape

# 可维可测分析

## 精度标准/性能标准

| 验收标准          | 描述              | 标准来源       |
| ------------- | --------------- | ---------- |
| 精度 rtol       | 2^-10 ≈ 9.77e-4 | 生态算子开源精度标准 |
| 精度 atol       | 2^-16 ≈ 1.53e-5 | 生态算子开源精度标准 |
| matched_ratio | ≥ 0.99          | 生态算子开源精度标准 |
| max_abs_error | ≤ 1e-2 或 32×ULP | 生态算子开源精度标准 |
| 性能 n=1048576  | ≤ 13.57 μs      | 任务书标杆      |
| 性能 n=2097152  | ≤ 21.05 μs      | 任务书标杆      |
| 性能 n=4194304  | ≤ 43.02 μs      | 任务书标杆      |

## 兼容性分析

- 本算子为新增接口，不涉及与历史版本的兼容
- 接口签名与 cuBLAS cublasCsscal 一致，参数顺序一一对应
- 遵循 ops-blas 仓规范，可与其他产品线共用

## 测试框架

测试代码位于 `test/scal/csscal/arch35/`，使用 CSV 驱动的 GTest 框架。

测试用例覆盖：

| 类别     | 说明                               |
| ------ | -------------------------------- |
| TC_L0  | 小规模基础用例（n=1,8）                   |
| TC_SQ  | n 扫描（2^n ±1 边界，n=1~1048576）      |
| TC_INC | incx 变化（1,2,3,5,7）               |
| TC_AB  | alpha 特殊值（0.0, 1.0, 负值, 大值 1e10） |
| TC_FL  | x 特殊值（INF, NaN）                  |
| TC_ED  | 边界与负向用例（n≤0, incx≤0, nullptr）    |
| TC_EX  | 扩展组合用例                           |
| TC_PF  | 性能测试用例（3条标杆）                     |

# 附录

## 接口定义

```cpp
aclblasStatus_t aclblasCsscal(
    aclblasHandle_t handle,  // 输入：ops-blas 库上下文句柄
    int n,                   // 输入：向量 x 的复数元素个数
    const float* alpha,      // 输入：实数标量乘数
    aclblasComplex* x,       // 输入/输出：复数向量（原地更新）
    int incx                 // 输入：步长
);
```

### 参数说明

| 参数名    | 类型              | 方向    | 描述                   |
| ------ | --------------- | ----- | -------------------- |
| handle | aclblasHandle_t | 输入    | 库上下文句柄，携带 stream     |
| n      | int             | 输入    | 复数元素个数，n ≤ 0 为 no-op |
| alpha  | const float*    | 输入    | 实数标量，alpha=0.0 为置零操作 |
| x      | aclblasComplex* | 输入/输出 | 复数向量，原地更新            |
| incx   | int             | 输入    | 步长，incx ≤ 0 为 no-op  |

### 数据类型

- **aclblasComplex**: 实部/虚部各 float32（complex64）
- **alpha**: float32 实数

## 错误处理

| 错误码                              | 条件                                        |
| -------------------------------- | ----------------------------------------- |
| ACLBLAS_STATUS_HANDLE_IS_NULLPTR | handle == nullptr                         |
| ACLBLAS_STATUS_INVALID_VALUE     | n > 0 且 (alpha == nullptr 或 x == nullptr) |
| ACLBLAS_STATUS_SUCCESS           | 成功执行或 no-op（n≤0, incx≤0, alpha=1.0）       |

## 性能目标

| case | n       | incx | 标杆耗时(μs) |
| ---- | ------- | ---- | -------- |
| 1    | 1048576 | 1    | 13.57    |
| 2    | 2097152 | 1    | 21.05    |
| 3    | 4194304 | 1    | 43.02    |

## 精度要求

- rtol = 2^-10 ≈ 9.77e-4
- atol = 2^-16 ≈ 1.53e-5
- matched_ratio ≥ 0.99
- max_abs_error ≤ 1e-2 或 32×ULP

## 测试用例

测试代码位于 `test/scal/csscal/arch35/`，使用 CSV 驱动的 GTest 框架。测试用例共计 1200 条。

### 精度测试 case（共 1000 条）

#### TC_L0 - 小规模基础用例（6 条）

| case_name | n   | incx | alpha | x               |
| --------- | --- | ---- | ----- | --------------- |
| TC_L0_001 | 1   | 1    | 1.0   | RANDOM_NORM_5_5 |
| TC_L0_002 | 1   | 2    | 1.0   | RANDOM_NORM_5_5 |
| TC_L0_003 | 1   | 3    | 1.0   | RANDOM_NORM_5_5 |
| TC_L0_004 | 8   | 1    | 1.0   | RANDOM_NORM_5_5 |
| TC_L0_005 | 8   | 2    | 1.0   | RANDOM_NORM_5_5 |
| TC_L0_006 | 8   | 3    | 1.0   | RANDOM_NORM_5_5 |

#### TC_SQ - n 扫描用例（38 条）

覆盖 n = 1, 2, 3, 4, 5, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257, 511, 512, 513, 1000, 1024, 1025, 2048, 4095, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 1048576

代表性用例：

| case_name | n       | incx | alpha | x               |
| --------- | ------- | ---- | ----- | --------------- |
| TC_SQ_007 | 1       | 1    | 2.5   | RANDOM_NORM_5_5 |
| TC_SQ_013 | 8       | 1    | 2.5   | RANDOM_NORM_5_5 |
| TC_SQ_030 | 512     | 1    | 2.5   | RANDOM_NORM_5_5 |
| TC_SQ_037 | 4096    | 1    | 2.5   | RANDOM_NORM_5_5 |
| TC_SQ_044 | 1048576 | 1    | 2.5   | RANDOM_NORM_5_5 |

#### TC_INC - incx 变化用例（5 条）

| case_name  | n   | incx | alpha | x               |
| ---------- | --- | ---- | ----- | --------------- |
| TC_INC_045 | 64  | 1    | 2.5   | RANDOM_NORM_5_5 |
| TC_INC_046 | 64  | 2    | 2.5   | RANDOM_NORM_5_5 |
| TC_INC_047 | 64  | 3    | 2.5   | RANDOM_NORM_5_5 |
| TC_INC_048 | 64  | 5    | 2.5   | RANDOM_NORM_5_5 |
| TC_INC_049 | 64  | 7    | 2.5   | RANDOM_NORM_5_5 |

#### TC_AB - alpha 特殊值用例（18 条）

| case_name | n   | incx | alpha         | 说明                 |
| --------- | --- | ---- | ------------- | ------------------ |
| TC_AB_050 | 8   | 1    | 1.0           | alpha=1.0 等价 no-op |
| TC_AB_051 | 64  | 1    | 1.0           | alpha=1.0 等价 no-op |
| TC_AB_052 | 8   | 1    | 0.0           | 置零操作               |
| TC_AB_053 | 64  | 1    | 0.0           | 置零操作               |
| TC_AB_054 | 8   | 1    | -1.0          | 负值                 |
| TC_AB_055 | 64  | 1    | -1.0          | 负值                 |
| TC_AB_056 | 8   | 1    | 0.5           | 小数值                |
| TC_AB_057 | 64  | 1    | 0.5           | 小数值                |
| TC_AB_058 | 8   | 1    | -0.75         | 负小数                |
| TC_AB_059 | 64  | 1    | -0.75         | 负小数                |
| TC_AB_060 | 8   | 1    | 100.0         | 大值                 |
| TC_AB_061 | 64  | 1    | 100.0         | 大值                 |
| TC_AB_062 | 8   | 1    | 10000000000.0 | 超大值（溢出测试）          |
| TC_AB_063 | 64  | 1    | 10000000000.0 | 超大值（溢出测试）          |
| TC_AB_064 | 8   | 1    | 2.5           | 常规值                |
| TC_AB_065 | 64  | 1    | 2.5           | 常规值                |
| TC_AB_066 | 8   | 1    | -3.25         | 负值                 |
| TC_AB_067 | 64  | 1    | -3.25         | 负值                 |

#### TC_FL - x 特殊值用例（6 条）

| case_name | n   | incx | alpha | x               | 说明     |
| --------- | --- | ---- | ----- | --------------- | ------ |
| TC_FL_068 | 64  | 1    | 2.5   | RANDOM_NORM_5_5 | 随机正态分布 |
| TC_FL_069 | 64  | 1    | 2.5   | VALUE_NORM_0    | 全零     |
| TC_FL_070 | 64  | 1    | 2.5   | RANDOM_ALTER    | 交错值    |
| TC_FL_071 | 64  | 1    | 2.5   | RANDOM_EXTREME  | 极值     |
| TC_FL_072 | 64  | 1    | 2.5   | VALUE_NORM_INF  | 含 INF  |
| TC_FL_073 | 64  | 1    | 2.5   | VALUE_NORM_NAN  | 含 NaN  |

#### TC_ED - 边界与负向用例（10 条）

| case_name | n   | incx | alpha | x               | 期望结果          | 说明            |
| --------- | --- | ---- | ----- | --------------- | ------------- | ------------- |
| TC_ED_074 | 0   | 1    | 1.0   | RANDOM_NORM_5_5 | SUCCESS       | n=0 no-op     |
| TC_ED_075 | 0   | 2    | 1.0   | RANDOM_NORM_5_5 | SUCCESS       | n=0 no-op     |
| TC_ED_076 | 0   | 1    | null  | RANDOM_NORM_5_5 | SUCCESS       | n=0 no-op     |
| TC_ED_077 | -5  | 1    | 1.0   | RANDOM_NORM_5_5 | SUCCESS       | n<0 no-op     |
| TC_ED_078 | -5  | 2    | 1.0   | RANDOM_NORM_5_5 | SUCCESS       | n<0 no-op     |
| TC_ED_079 | 16  | 0    | 1.0   | RANDOM_NORM_5_5 | SUCCESS       | incx=0 no-op  |
| TC_ED_080 | 16  | -1   | 1.0   | RANDOM_NORM_5_5 | SUCCESS       | incx<0 no-op  |
| TC_ED_081 | 16  | -2   | 1.0   | RANDOM_NORM_5_5 | SUCCESS       | incx<0 no-op  |
| TC_ED_082 | 16  | 1    | null  | RANDOM_NORM_5_5 | SUCCESS       | alpha=nullptr |
| TC_ED_083 | 16  | 1    | 1.0   | NULLPTR         | INVALID_VALUE | x=nullptr     |

#### TC_EX - 扩展组合用例（917 条）

覆盖多种 n、incx、alpha 组合，包括：

- n: 1~1048576 的各种尺寸
- incx: 1, 2, 3, 5, 7 等
- alpha: 0.0, 1.0, -1.0, 0.5, -0.75, 2.5, 100.0, 10000000000.0 等

代表性用例：

| case_name  | n     | incx | alpha         | x               |
| ---------- | ----- | ---- | ------------- | --------------- |
| TC_EX_0084 | 8192  | 2    | 1.0           | RANDOM_NORM_5_5 |
| TC_EX_0085 | 129   | 1    | 0.0           | RANDOM_NORM_5_5 |
| TC_EX_0086 | 255   | 7    | -1.0          | RANDOM_NORM_5_5 |
| TC_EX_0087 | 32768 | 3    | 0.5           | RANDOM_NORM_5_5 |
| TC_EX_0088 | 255   | 2    | -0.75         | RANDOM_NORM_5_5 |
| TC_EX_0089 | 513   | 1    | 100.0         | RANDOM_NORM_5_5 |
| TC_EX_0090 | 2048  | 3    | 10000000000.0 | RANDOM_NORM_5_5 |
| TC_EX_0091 | 1     | 7    | 2.5           | RANDOM_NORM_5_5 |
| TC_EX_0092 | 32    | 5    | -3.25         | RANDOM_NORM_5_5 |

### 性能测试 case（共 200 条）

性能测试 case 用于验证算子在各规模下的执行时间，须先 warmup 再有效采样 >50 次取平均。

#### 标杆性能用例（3 条，任务书规定）

| case_name  | n       | incx | alpha | 标杆耗时(μs) |
| ---------- | ------- | ---- | ----- | -------- |
| TC_PF_1001 | 1048576 | 1    | 2.5   | 13.57    |
| TC_PF_1002 | 2097152 | 1    | 2.5   | 21.05    |
| TC_PF_1003 | 4194304 | 1    | 2.5   | 43.02    |

#### 小规模性能用例（197 条）

覆盖 n = 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304 等规格，用于性能曲线绘制。

## 交付件清单

| 序号  | 交付件         | 路径                                        |
| --- | ----------- | ----------------------------------------- |
| 1   | 算子设计文档      | 本文档                                       |
| 2   | Kernel 实现   | blas/scal/arch35/csscal/csscal_kernel.cpp |
| 3   | Kernel 头文件  | blas/scal/arch35/csscal/csscal_kernel.h   |
| 4   | Host API 实现 | blas/scal/arch35/csscal/csscal_host.cpp   |
| 5   | Tiling 数据结构 | blas/scal/arch35/csscal/csscal_tiling.h   |
| 6   | 测试用例 CSV    | test/scal/csscal/arch35/csscal_test.csv   |
| 7   | 测试代码        | test/scal/csscal/arch35/csscal_test.cpp   |
| 8   | 算子 README   | blas/scal/arch35/csscal/README.md         |

## 参考资料

1. [cuBLAS cublasCsscal](https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-scal)
2. [Netlib BLAS csscal](https://www.netlib.org/blas/csscal.f)
3. [ops-blas 仓库](https://gitcode.com/cann/ops-blas)
4. [Ascend C 算子开发文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/opdevg/Ascendcopdevg/atlas_ascendc_map_10_0002.html)
5. [生态算子开源精度标准](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md)
