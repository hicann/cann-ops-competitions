# aclblasCtbmv 算子设计文档

文档版本：V1.0
目标仓库：cann/ops-blas
支持产品：Atlas A2/A3 系列（arch22）
实现语言：Ascend C / C++

# 需求背景（required）

## 需求来源

社区任务要求在 Atlas A2/A3 上新增单精度复数三角带状矩阵向量乘接口 `aclblasCtbmv`，功能和参数语义与 cuBLAS/Netlib CTBMV 对齐，并以 kernel 直调方式集成到 ops-blas。

## 背景介绍

CTBMV 计算复数三角带状矩阵与复数向量的乘积，并原地更新 x：

`x = op(A) * x`

其中 `op(A)` 可为 A、A 的转置或共轭转置。与稠密 GEMV 相比，CTBMV 只存储和访问三角矩阵对角线附近的 k 条带，具有不规则带状寻址、复数乘加和原地覆盖依赖。

### 现状分析

ops-blas 已有实数同族接口 `aclblasStbmv`，但公共头文件缺少 complex64 的 `aclblasCtbmv`。本任务新增公共 API、arch22 host/kernel 实现、CSV 驱动测试和复现说明，不创建产品私有平行接口。

# 需求分析（required）

## 需求描述

实现 COMPLEX64 CTBMV，支持：

- `uplo`：UPPER、LOWER；
- `trans`：N、T、C；
- `diag`：NON_UNIT、UNIT；
- `n >= 0`、`k >= 0`、`lda >= k+1`；
- `incx != 0`，包括负步长；
- `n=0` 合法 quick return；
- x 原地输入/输出；
- CANN 9.1.0、Atlas A2/A3 arch22。

## 需求拆解

1. 新增公共 API 声明和状态码校验。
2. 正确实现上下三角带状列主序映射。
3. 保证 UNIT 对角不读取、未映射三角槽不读取。
4. 保护原始 x，避免原地写回破坏后续计算。
5. 提供稳定通用路径和三个任务书性能 shape 的专用路径。
6. 完成官方 CSV、附加 sentinel、异常参数、性能和跨 A2/A3 复核。

# 详细设计（required）

## 算子分析

### 数学公式

对于输出元素 yᵢ：

- N：`y_i = Σ_j A(i,j) * x_j`；
- T：`y_i = Σ_j A(j,i) * x_j`；
- C：`y_i = Σ_j conj(A(j,i)) * x_j`。

复数乘法 `(a+bi)(c+di)` 分解为：

- 实部：`ac - bd`；
- 虚部：`ad + bc`。

### 带状存储

A 以 `lda × n` 列主序存储：

- UPPER：`A(i,j)` 位于 band row `k+i-j`；
- LOWER：`A(i,j)` 位于 band row `i-j`；
- 仅当元素属于指定三角且 `|i-j| <= k` 时读取；
- UNIT 模式下对角值直接取 `(1,0)`，不访问 GM 中的对角槽。

### 数据类型与形状

| 对象 | 类型 | 形状/布局 | 内存 |
|---|---|---|---|
| A | `aclblasComplex` | `lda × n`，列主序带状 | Device，只读 |
| x | `aclblasComplex` | n 个逻辑元素，物理步长 `incx` | Device，原地读写 |
| tiling | `CtbmvTilingData` | 32 B 固定 ABI | Host→Kernel 参数 |
| workspace | byte buffer | 路径相关 | Device，handle 管理 |

## 接口设计

```cpp
aclblasStatus_t aclblasCtbmv(
    aclblasHandle_t handle, aclblasFillMode_t uplo,
    aclblasOperation_t trans, aclblasDiagType_t diag,
    int n, int k, const aclblasComplex* A, int lda,
    aclblasComplex* x, int incx);
```

参数校验顺序覆盖 handle、枚举、维度/步长和指针。`n=0` 或 `k=0 && diag=UNIT` 直接成功返回，不启动 kernel。

## Host 侧设计

### 参数校验

| 条件 | 返回值 |
|---|---|
| `handle == nullptr` | `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| 非法 uplo/trans/diag | `ACLBLAS_STATUS_INVALID_ENUM` |
| `n<0`、`k<0`、`lda<k+1`、`incx==0` | `ACLBLAS_STATUS_INVALID_VALUE` |
| `n>0` 且 A/x 为空 | `ACLBLAS_STATUS_INVALID_VALUE` |

### 路径选择

Host 根据 shape、方向、连续性和可用 AIV 数选择：

1. 三个正式 shape 的专用 direct-output kernel；
2. compact band group / aligned parallel 路径；
3. 连续 N 的 column fast path；
4. 连续 T/C 的 transpose fast path；
5. 非连续或大带宽的稳定通用路径。

正式 512 shape 使用对齐专用并行布局；其他非正式 compact shape 在无法满足对齐和单 owner 条件时回退到稳定通用路径，避免非对齐原子边界竞争。

### Workspace

专用路径 workspace 由原始 x 快照、interleave offset 和 window offset 构成。三个正式 shape 的算子实际 workspace 分别约为：

| n/k | workspace |
|---|---:|
| 512/8 | 6,528 B |
| 1024/16 | 15,360 B |
| 2048/32 | 44,032 B |

这些空间来自 handle 的 32 MiB 默认 workspace，不在每次调用内动态申请。

## Kernel 侧设计

### 通用正确性策略

通用路径优先保证全参数域正确：先保存原始 x，按带状映射逐列或逐转置方向计算，最后写回。负步长由逻辑索引映射到物理起始偏移，padding 保持不变。

### 正式 shape 的 direct-output 策略

三个性能 case 使用独立 AIV 入口，整体流程为：

`GM(A,x,offset) → UB staging → complex split/Gather → complex MAC → row reduce → interleave → GM(x)`

设计要点：

- 单 owner：每个输出行只由一个 AIV 负责；
- 单写回：每个 complex 输出一次非原子写回；
- A/x 复用：compact A 和 x 每核只 staging 一次；
- 32B 对齐：UB 地址、Vector 长度和 offset 表满足 dav-2201 对齐要求；
- 边界清零：只清真正可能被 Gather/归约读取的尾部，避免未定义字节污染；
- 独立入口：去除无实际 `SyncAll` 路径的跨核同步元数据；
- Block：n512/n1024/n2048 分别规划为 36/40/40，`Mix Block Num=0`；
- MTE3：在写回完成性无法由 kernel 结束隐式保证的路径上显式等待，避免 Release 构建下出现异步写回竞态。

### Tiling 数据

`CtbmvTilingData` 压缩为 32 B，包含 n/k/lda/incx、枚举、路径标识、核数与分段参数，并由 `static_assert` 固定 ABI，降低参数传递开销和版本漂移风险。

## 性能优化策略

1. 先以通用路径建立全参数域正确性基线，再启用多 AIV 和 UB 向量化复数计算。
2. 对任务书正式 shape 使用 direct-output：输出单 owner、单写回，避免跨核归约和原子累加。
3. A 与 x 在单核内尽量只 staging 一次；offset/interleave 表驻留 UB，减少重复地址计算与 GM 访问。
4. 通过 32 B 对齐、尾部清零和有限范围 Gather 保证向量路径既满足 dav-2201 约束又不读取未定义字节。
5. 使用 Release `-O3 -DNDEBUG` 构建进行正式验收；Debug 构建只用于定位问题，不用于性能结论。
6. 使用 `msprof --task-time=on` 分析 DMA、split、算术和输出阶段；候选优化仅在独立重复采样稳定收益时保留。

## 支持硬件

| 支持的芯片版本 | 支持 |
|---|---|
| Atlas 800I/T A2（910B3/910B4，arch22） | √ |
| Atlas A3（Ascend910_93xx，arch22） | √，arch22 共用实现 |

构建必须使用机器返回的精确 SoC；910C 不能仅写泛化产品名。

## 算子约束限制

- 仅 COMPLEX64；
- A 必须为 n×n 三角带状矩阵，不存在非方阵语义；
- 不支持超出 `lda/incx` 语义的任意非连续 Tensor；
- 无 alpha/beta；
- 不要求确定性计算，但设计通过单 owner 和稳定回退路径避免跨核写竞争；
- 性能专用路径只覆盖任务书三个 formal shape，其余参数走通用稳定路径。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
|---|---|---|
| 精度 | 实/虚部分别 `rtol=2^-10`、`atol=2^-16`，matched ratio≥0.99，最大绝对误差≤1e-2 或 32 ULP | 任务书与生态算子 FLOAT32 标准 |
| 性能 | n512≤6.40 us；n1024≤10.57 us；n2048≤12.65 us | 任务书 §3.3 |
| 采样 | 5 warmup + 64 measured，取 middle-48；每轮 207 kernel rows | 项目统一 msprof 口径 |

## 测试设计

官方 CSV 1200 条：1000 条精度/边界和 200 条性能/内存类用例；另有 3 条 GTest 级检查。覆盖 12 个枚举组合、尺寸/带宽、padding、±1/±2/±3 步长、Inf/NaN、UNIT 对角不读、非法参数、n=0、n=4096 fallback 和三个正式性能 case。

## 兼容性分析

本接口为新增 complex 路径，参数序列与 `aclblasStbmv` 同型并与 cuBLAS CTBMV 对齐，不改变已有 API。代码位于 arch22，实现入口和公共声明可供 A2/A3 产品线共用。

## 可维护性

- formal kernel 分文件，通用路径与专用路径边界明确；
- tiling ABI 有静态断言；
- 测试 CSV SHA-256 固定，解析脚本校验行数和 block 元数据；
- 性能试验以 commit/revert 记录，避免保留不可归因的优化；
- README 提供构建、运行和测量口径。

## 参考资料

1. CANN 社区任务设计模板：https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md
2. Netlib CTBMV：https://www.netlib.org/blas/ctbmv.f
3. ops-blas：https://gitcode.com/cann/ops-blas
4. 生态算子精度标准：https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md
