# 9月社区任务-aclblasCtrsmBatched 算子设计文档（A2/A3）

## 需求背景（required）

### 需求来源

本设计对应 9 月社区任务 `aclblasCtrsmBatched`（Atlas A2/A3）。目标是在
`cann/ops-blas` 的 `blas/trsmbatched/arch22/` 中新增 COMPLEX64 批量三角
矩阵求解实现，接口语义对齐 cuBLAS `cublasCtrsmBatched`。本文是设计评审
交付件，不代表实现、设备精度或性能验收已经完成。

### 背景介绍

对每个 batch `i`，算子原地求解：

```text
side = LEFT  : op(A[i]) * X[i] = alpha * B[i]
side = RIGHT : X[i] * op(A[i]) = alpha * B[i]
```

其中 `B[i]` 在设备端原地由右端项变为解，矩阵使用列主序。`A[]` 和 `B[]`
均为设备端指针数组；所有 batch 共享维度、前导维和枚举参数，但每一批的
矩阵地址独立。A2/A3 使用 `arch22` 路径，不能通过调用完整的框架、CPU、
TBE 或其他 BLAS 算子来替代本算子。

## 需求分析（required）

### 需求描述

公共接口保持句柄式 BLAS ABI：

```cpp
aclblasStatus_t aclblasCtrsmBatched(
    aclblasHandle_t handle,
    aclblasSideMode_t side,
    aclblasFillMode_t uplo,
    aclblasOperation_t trans,
    aclblasDiagType_t diag,
    int m,
    int n,
    const aclblasComplex* alpha,
    const aclblasComplex* const A[],
    int lda,
    aclblasComplex* const B[],
    int ldb,
    int batchCount);
```

`aclblasComplex` 为两个 FP32 分量的 `COMPLEX64` 值。`A[i]` 的阶数是
`side == LEFT ? m : n`，`B[i]` 是 `m x n`；`A`、`B` 的元素地址分别按
`row + col * ld` 计算。`lda`、`ldb` 可以带列尾填充，但填充区域不能参与
计算或写回。

### 需求拆解

1. 支持 `COMPLEX64`、列主序、设备端 pointer array、批次独立和 `B` 原地输出。
2. 支持 `side` 的 LEFT/RIGHT、`uplo` 的 UPPER/LOWER、`trans` 的 N/T/C、
   `diag` 的 UNIT/NON_UNIT，共 24 种合法组合。
3. `trans=T` 只做转置，`trans=C` 做共轭转置；不能把两者合并成一个未标注
   虚部符号的路径。
4. `diag=UNIT` 时逻辑对角恒为 `(1, 0)`，不得读取存储的对角元素；
   `diag=NON_UNIT` 时由调用方保证被引用的对角元素非零，算子不做奇异性检测。
5. 参数错误必须在 host 侧以仓库约定状态码返回，不得发射 kernel；合法空工作
   场景不读取无关矩阵数据。`batchCount < 1` 按当前 A2/A3 任务接口约定返回
   `ACLBLAS_STATUS_INVALID_VALUE`。
6. 所有设备计算使用 `handle` 绑定的 stream；调用者在读取 `B[]` 前负责同步。

### 输入、输出与校验边界

| 参数 | 位置/语义 | 设计约束 |
| --- | --- | --- |
| `handle`、枚举、`m/n`、`lda/ldb`、`batchCount` | Host | 先校验句柄、枚举、整数范围和前导维；不从设备数据推断形状 |
| `alpha` | Host 指针 | 非空；host 只读取一次实部和虚部，`alpha=0` 进入独立清零路径 |
| `A[]` | Device pointer array | 每项指向对应 batch 的设备三角矩阵；非零 alpha 的非空计算必须有效 |
| `B[]` | Device pointer array | 每项指向 `m x n` 设备矩阵；非空计算必须有效，结果原地写回 |

校验顺序固定为：句柄 -> 枚举/维度/前导维/批次数 -> `alpha` -> 空工作判定
-> `A[]`/`B[]` 的地址表和元素。地址表容量和索引乘法使用宽整数检查，避免
合法单项尺寸在批量相乘时溢出。无效枚举、负维度、不足前导维、空指针或
无效设备地址均返回 `ACLBLAS_STATUS_INVALID_VALUE`，不通过另一实现兜底。

## 详细设计（required）

### 算子分析

令 `T = op(A[i])`，`Y = alpha * B[i]`。复数乘法采用
`(ar*br-ai*bi, ar*bi+ai*br)`。`trans=N` 读取 `A[r+c*lda]`；`trans=T`
读取 `A[c+r*lda]`；`trans=C` 读取相同转置位置并对虚部取反。有效三角
方向与依赖顺序如下：

| `uplo` | `trans=N` 时 `T` | `trans=T/C` 时 `T` | LEFT 行顺序 | RIGHT 列顺序 |
| --- | --- | --- | --- | --- |
| UPPER | 上三角 | 下三角 | 递减 / 递增 | 递增 / 递减 |
| LOWER | 下三角 | 上三角 | 递增 / 递减 | 递减 / 递增 |

左求解的每个右端列满足：

```text
X[r,j] = (Y[r,j] - sum(T[r,p] * X[p,j])) / d[r]
```

右求解的每个右端行满足：

```text
X[r,c] = (Y[r,c] - sum(X[r,p] * T[p,c])) / d[c]
```

其中 `d` 在 UNIT 模式为复数单位元，否则为对应的 `T` 对角元素。复数除法
使用按实部/虚部较大分量缩放的 Smith 形式，避免直接计算平方和造成中间
溢出；输入 NaN/Inf 和奇异对角属于数值输入边界，不在 host 参数校验中猜测。

### Host 侧设计

Host 只负责 ABI 校验、设备地址表处理、路径选择和 tiling 数据生成，不读取
完整矩阵。先把 `A[]`/`B[]` 的设备指针值复制到受控的 host 临时表，检查
`batchCount` 范围和每个非空项，再将形状、前导维、有效三角方向、转置/共轭
标志、单位对角标志、alpha 分量、panel/tile 尺寸和 batch 范围写入 tiling。

`side=LEFT` 时三角阶数为 `m`，可并行的独立右端向量数为 `n`；`side=RIGHT`
时三角阶数为 `n`，独立右端向量数为 `m`。不同 batch 之间没有数据依赖，
同一向量的三角依赖不能跨核拆开。所有 launch 复用 handle stream，地址表和
矩阵数据保持在设备端，不把完整矩阵搬回 host。

### Kernel 侧设计

#### 小阶/高 batch 路径

将 `(batch, rhs)` 映射为一个独立任务：LEFT 的 `rhs` 是一列，RIGHT 的 `rhs`
是一行。任务沿有效三角方向串行推进，使用 UB 保存当前向量的有限 tile；
每次加载、计算和写回都以真实有效长度作边界判断。UNIT 对角在形成地址前
直接替换为 `(1,0)`，alpha 为零时只写 `B` 的逻辑区域为零，不访问 `A`。

#### 大阶/宽右端路径

采用 panel solve + trailing update。当前 panel 先在 Vector/SIMT 路径完成
顺序三角求解，再用同一 stream 上的显式依赖更新未求解区域。复数块乘法
展开为四次 FP32 实数乘法并组合实部/虚部；panel 之间保持写后读顺序，
同一 panel 的不同 batch 和非重叠输出 tile 可并行。panel、rhs tile 和尾块
长度由运行时 UB/L1 容量约束，不能写死为只对齐公开用例的尺寸。

```text
for panel p in dependency_order:
    solve diagonal panel T[p,p] against current B[p]
    synchronize on the bound stream
    parallel over non-overlapping output tiles:
        LEFT : B[q] -= T[q,p] * X[p]
        RIGHT: B[q] -= X[p] * T[p,q]
```

转置和共轭通过 tile 索引与虚部符号处理，不分配完整矩阵副本。对于 `m`、
`n`、panel 或 batch 尾部不满足搬运粒度的情况，load/store 使用有效行列数
掩码；对齐后的长度不得用于访问 `lda/ldb` 填充区。每个任务都检查
`batch < batchCount`、`row < m`、`col < n` 及三角阶数边界。

### Dispatch 设计

| 运行时条件 | 路径 | 关键约束 |
| --- | --- | --- |
| 三角阶数较小或 batch/rhs 并行度足够 | 小阶 SIMT/Vector | 一向量一任务，依赖链不跨核拆分 |
| 三角阶数较大且 rhs 宽 | panel + trailing update | panel 顺序固定，更新 tile 不重叠 |
| `alpha == (0,0)` | zero-fill | 只访问 `B` 有效区域，不访问 `A` |
| `diag == UNIT` | unit-diag specialization | 不形成对角存储地址 |
| 任意尾块/非对齐 lda/ldb | masked tail | 有效长度 runtime 可见，禁止越界访存 |

路径选择只使用 `m/n/batchCount/lda/ldb/side/uplo/trans/diag` 等合法运行时
元数据，不使用 case id、公开 shape 名称、时间值或答案数据作分派条件。

### 资源与同步约束

片上预算至少包括 A tile、B tile、实部/虚部计算 tile、四路乘积暂存和同步
保留量：

```text
A_tile + B_tile + compute_tile + temporary + reserved <= available
```

若预算不足，先缩小 rhs tile，再缩小 panel，保持至少一个有效 COMPLEX64
元素。一个 batch 的写入区间只由一个任务拥有；不同 batch 的 `B[i]` 不得
重叠。panel solve 与 trailing update 之间保留明确的 stream/event 依赖，
不得依赖未声明的隐式跨核顺序。

## 支持硬件

| 芯片 | 架构目录 | 支持范围 |
| --- | --- | --- |
| Atlas A2 训练系列 | DAV_2201 / `arch22` | COMPLEX64、全部 24 模式 |
| Atlas A3 系列 | DAV_2201 / `arch22` | COMPLEX64、全部 24 模式 |

公共声明放在 `include/cann_ops_blas.h`，实现放在
`blas/trsmbatched/arch22/`，测试入口放在
`test/trsmbatched/ctrsmbatched/arch22/`。本设计不修改已有 StrsmBatched
签名，也不把 950 `arch35` 路径冒充 A2/A3 实现。

## 算子约束限制

- `batchCount < 1`、负维度、无效枚举、前导维不足或空指针均为非法参数。
- 非空调用要求有效的 `B[]` 及其元素；非零 alpha 还要求有效的 `A[]` 及其元素。
- `diag=NON_UNIT` 的被引用对角必须非零；不检测奇异性或替换为其他求解器。
- 只有逻辑 `m x n` 元素被写回，填充区和其他 batch 的内存保持不变。
- 调用方须保持 A/B 地址和 pointer array 直到 handle stream 完成；B batch
  之间以及 A/B 有效区域不得发生未声明重叠。

## 可维可测分析

### 精度验证

测试以独立的 Netlib/Python 三角求解参考为准，逐 batch 比较 `B[i]` 的有效
区域，实部和虚部分别报告绝对/相对误差及匹配比例。用例至少覆盖：

| 类别 | 覆盖内容 |
| --- | --- |
| 模式组合 | LEFT/RIGHT、UPPER/LOWER、N/T/C、UNIT/NON_UNIT 的全组合 |
| 形状布局 | `m/n=1`、奇数和非对齐维度、非方形 B、`lda/ldb` 填充 |
| 数值边界 | alpha 为零、实数、一般复数；UNIT 对角存储哨兵；正负和近似值 |
| 批处理 | batch=1、多个 batch、非均匀地址、尾 batch |
| 错误处理 | 非法枚举、负维度、batchCount<1、空指针、前导维不足 |

设计阶段不声称任何设备测试通过；实现阶段必须保存原始 stdout、失败用例、
编译版本、实际 NPU/SoC 和输入生成规则，才能形成验收证据。

### 性能验证

性能测试分别统计 CtrsmBatched kernel 的 device event 时间，固定 warmup、
repeat、同步边界和聚合方式，并覆盖任务书的 batch、矩阵尺寸和 24 种模式
组合。参数校验、pointer-array D2H、launch 和结果同步必须与 kernel 时间
分开记录。只有在同一 shape、布局、正确性条件和采样协议下得到的 A100
与 A2/A3 数据才可计算比值；本设计文档不填入未经复跑的性能数字。

### 交付边界

设计 PR 仅新增本文件，不包含 ops-blas 源码、构建产物、私有 task/golden
文件或性能截图。后续源码 PR 必须在 `ops-blas` 中独立完成接口注册、arch22
实现、测试和 README，并在代码、测试和文档中保持同一 ABI 与无 fallback
边界。
