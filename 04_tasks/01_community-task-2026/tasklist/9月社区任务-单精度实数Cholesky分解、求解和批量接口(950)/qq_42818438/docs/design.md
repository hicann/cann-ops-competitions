# 【社区任务】aclsolverSpotrf / Spotrs / Spotri / SpotrfBatched / SpotrsBatched 算子设计文档

| 项 | 内容 |
| --- | --- |
| 文档版本 | v0.3（概览版） |
| 对应任务 | 《Atlas 950 单精度实数 Cholesky 分解、求解和批量接口 任务书》 |
| 目标硬件 | Ascend 950PR（Atlas 950） |
| 说明 | 本文档为设计概览；具体实现细节以代码与接口文档为准 |

# 一、需求背景（required）

## 1.1 需求来源

通过社区任务完成开源仓（ops-solver）算子贡献：在 Ascend 950PR 上新增 5 组单精度实数 Cholesky 族接口（共 6 个函数），语义与参数序列对齐 CUDA cuSolver legacy API，满足任务书规定的功能、精度、性能与确定性验收标准。

## 1.2 背景介绍

### 1.2.1 算子实现方案概述

- **工程模式**：ops-solver Host C API + AscendC/CATLASS Kernel 直调（非 aclnn 两段式、非 PyTorch 接口）。
- **公开头文件**：`include/cann_ops_solver.h`、`include/cann_ops_solver_common.h`（补充 `aclsolverFillMode_t` 枚举）。
- **接口清单与对标**：

| 交付接口 | 对标接口 |
| --- | --- |
| `aclsolverSpotrf` / `aclsolverSpotrf_bufferSize` | `cusolverDnSpotrf` / `cusolverDnSpotrf_bufferSize` |
| `aclsolverSpotrs` | `cusolverDnSpotrs` |
| `aclsolverSpotri` / `aclsolverSpotri_bufferSize` | `cusolverDnSpotri` / `cusolverDnSpotri_bufferSize` |
| `aclsolverSpotrfBatched` | `cusolverDnSpotrfBatched` |
| `aclsolverSpotrsBatched` | `cusolverDnSpotrsBatched` |

### 1.2.2 算子现状分析

#### 1.2.2.1 支持的数据类型和数据格式

| 项目 | 说明 |
| --- | --- |
| 数据类型 | FLOAT32（单精度实数） |
| 数据排布 | 列主序（column-major）ND；`lda`/`ldb` 满足 `≥ max(1,n)`，不要求等于 n |
| 指针归属 | 矩阵、info、workspace 均为 Device 指针；Host 侧仅传标量维数、枚举与 handle |
| 维数类型 | `int`（32-bit，与 CUDA legacy API 一致） |
| 枚举 | `aclsolverFillMode_t`：LOWER=0 / UPPER=1（与 `cublasFillMode_t` 对齐） |
| 批量形态 | `Aarray`/`Barray` 为 Device 上的指针数组 |

#### 1.2.2.2 接口语义与 info 约定

1. **Spotrf**：`LOWER` 时 `A = L·Lᵀ`、`UPPER` 时 `A = Uᵀ·U`；仅处理指定三角、因子原地覆盖；`devInfo`：0 成功 / -i 参数非法 / k 第 k 阶顺序主子式不正定。
2. **Spotrs**：解 `A·X = B`（A 为已分解因子，只读），解原地覆盖 B；`devInfo` 仅报参数错误。
3. **Spotri**：输入三角因子、输出原地覆盖为对称逆矩阵的对应三角；`devInfo`：0 / -i / k（因子奇异）。
4. **SpotrfBatched**：逐矩阵执行与 Spotrf 相同的分解；`infoArray[i]` 逐矩阵独立写。
5. **SpotrsBatched**：逐矩阵求解（仅 nrhs=1）；`info` 为标量、仅报参数错误。

通用约定：空问题成功返回；相同输入、相同 stream 串行多次执行输出（含 info）bit-wise 一致；计算走调用方 stream；核心计算在 AI Core 完成，不允许 CPU fallback。

#### 1.2.2.3 对标实现算法流程

对标实现采用业界标准的**分块（blocked）算法骨架**，按列分块循环推进，每轮依次完成：① 面板分解（对当前对角块做分解）；② 三角求解（计算当前面板的非对角部分）；③ 尾部更新（对剩余矩阵做对称秩-k 更新）。重复直至全部列处理完毕，最后输出结果。

# 二、需求分析（required）

## 2.1 外部组件依赖

- CANN（9.0.0 及以上，与 ops-solver 仓 README 配套版本一致）；ACL / AscendC / CATLASS 工具链。
- 精度 golden：NumPy / SciPy `float64`（或等价 LAPACK）。
- 性能金标：cuSolver（`bench_result.json` 预采集参考数据）。
- 自验工具：ops-solver 仓测试工程、AscendOpTest。

## 2.2 内部适配模块

- handle 与 stream 管理：复用仓内既有 `aclsolverCreate / Destroy / SetStream / GetStream`。
- 公共支撑：参数校验与错误上报、workspace 计算（`_bufferSize`）、调度配置（tiling）生成与 kernel 下发封装。

## 2.3 需求模块设计

### 2.3.1 Ascend C 算子原型（接口定义）

```c
typedef enum {
    ACLSOLVER_FILL_MODE_LOWER = 0,
    ACLSOLVER_FILL_MODE_UPPER = 1
} aclsolverFillMode_t;

aclsolverStatus_t aclsolverSpotrf_bufferSize(aclsolverHandle_t handle, aclsolverFillMode_t uplo,
                                             int n, float *A, int lda, int *Lwork);
aclsolverStatus_t aclsolverSpotrf(aclsolverHandle_t handle, aclsolverFillMode_t uplo, int n,
                                  float *A, int lda, float *Workspace, int Lwork, int *devInfo);
aclsolverStatus_t aclsolverSpotrs(aclsolverHandle_t handle, aclsolverFillMode_t uplo, int n, int nrhs,
                                  const float *A, int lda, float *B, int ldb, int *devInfo);
aclsolverStatus_t aclsolverSpotri_bufferSize(aclsolverHandle_t handle, aclsolverFillMode_t uplo,
                                             int n, float *A, int lda, int *Lwork);
aclsolverStatus_t aclsolverSpotri(aclsolverHandle_t handle, aclsolverFillMode_t uplo, int n,
                                  float *A, int lda, float *Workspace, int Lwork, int *devInfo);
aclsolverStatus_t aclsolverSpotrfBatched(aclsolverHandle_t handle, aclsolverFillMode_t uplo, int n,
                                         float *Aarray[], int lda, int *infoArray, int batchSize);
aclsolverStatus_t aclsolverSpotrsBatched(aclsolverHandle_t handle, aclsolverFillMode_t uplo, int n, int nrhs,
                                         float *Aarray[], int lda, float *Barray[], int ldb,
                                         int *info, int batchSize);
```

参数名、顺序、含义、Device/Host 归属与 cuSolver legacy API 对齐。

### 2.3.2 参数与错误处理约定

- 参数错误按 LAPACK 约定以负下标上报（`-i`，handle 不计），按签名参数序校验、以首个非法参数为准；`devInfo`/`info` 自身为空时仅返回状态码。
- 空问题（`n=0` / `nrhs=0` / `batchSize=0`）成功返回。
- 列主序按 32 位索引寻址，越界按参数错误拒绝而非回绕。

### 2.3.3 算子相关约束（与标杆相比的差异/缺失功能）

- 统一返回 `aclsolverStatus_t`（状态码语义对齐 cuSolver）。
- 验收规模：`n ∈ [1,4096]`；`nrhs ∈ [1,128]`；`batchSize ∈ [1,30000]`；SpotrsBatched 仅 `nrhs=1`。
- 不支持 broadcast；不要求图融合；dynamic shape 按本次调用生成调度配置。

# 三、需求详细设计（required）

## 3.1 调用方式

**Kernel 直调**：调用方创建 handle 并设置 stream → （potrf/potri）按 `_bufferSize` 准备 workspace → 调用计算接口 → 结果与 info 写回 Device；全程走调用方 stream。

## 3.2 需求总体设计

### 3.2.1 host 侧设计

- **校验与错误上报**：按 2.3.2 的约定在 kernel 下发前完成全部校验。
- **分核与任务划分**：以多核并行为基础，按矩阵规模与调用形态进行任务划分与负载均衡；小规模调用使用轻量路径以规避调度开销；批量接口以矩阵为并行粒度映射到各核。
- **数据分块与内存规划**：依据平台存储层级与容量进行分块与缓冲复用，兼顾吞吐与访存效率；原地语义与 workspace 需求在调度配置阶段统一处理。
- **调度配置生成**：host 按本次调用的 `uplo`、规模、`nrhs`、批量形态等信息生成调度配置，供 kernel 选择执行分支。

### 3.2.2 kernel 侧设计

- **总体**：五个接口共享"分块算法 + 多核协作"的总体结构，结合平台并行与向量计算能力实现；数值全程 FLOAT32；确定性按统一约定保证（固定计算顺序、阶段化同步与写者约束）。
- **各接口要点**：
  - Spotrf：分块 Cholesky 分解；逐列检查对角元，非正定处终止并按约定写 info。
  - Spotrs：分块前代/回代求解；按右端项列并行；因子只读。
  - Spotri：基于因子的求逆实现；结果按 `uplo` 原地覆写对应三角。
  - SpotrfBatched / SpotrsBatched：逐矩阵独立计算，矩阵级并行；info 按接口约定写回。

实现流程：Host 侧完成参数校验与调度配置生成后下发 Kernel；Kernel 多核并行完成分块计算；最后写回结果与 info。

- **与标杆（cuSolver）的差异**：① 并行与分块参数按 Ascend 平台特性定制（算法骨架与对标一致）；② 确定性保证更严格（bit-wise 可复现）；③ 参数错误上报路径按工程实现约定处理，不影响成功路径的无多余同步特性。

## 3.3 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Ascend 950PR（Atlas 950） | √ |

## 3.4 算子约束限制

1. 数据类型仅 FLOAT32；列主序。
2. `uplo` 仅 LOWER/UPPER；Spotrs/Spotri 的 `uplo` 须与 Spotrf 一致。
3. 原地语义：Spotrf/Spotri 覆盖指定三角；Spotrs/SpotrsBatched 覆盖 B。
4. 规模约束：`n ∈ [1,4096]`、`nrhs ∈ [1,128]`、`batchSize ∈ [1,30000]`；SpotrsBatched 仅 `nrhs=1`。
5. 空问题成功返回；不支持 broadcast；不提供 CPU fallback。

# 四、特性交叉分析

- **handle / stream**：全部接口复用仓内 handle 管理，计算在调用方 stream 串行执行。
- **确定性 × 性能**：确定性为统一要求（含批量逐矩阵）；优化遵循"改写不改序"原则。
- **原地语义 × 工作空间**：`_bufferSize`/`Lwork` 校验与原地覆盖语义在调度阶段统一处理。
- **批量 × 指针数组**：寻址、校验与并行映射按指针数组形态统一处理。
- 不涉及广播、图融合与动态 shape 特化。

# 五、可维可测分析

## 5.1 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | 逐元素混合容差判定（FLOAT32：rtol=2⁻¹⁰、atol=2⁻¹⁶、matched_ratio≥0.99、max_abs_error≤1e-2 或 32×ULP）；不通过时按 LAPACK 残差判据复核；info 语义正确；合法用例重复执行 bit-wise 一致 | 任务书 §3.2 |
| 性能标准 | 每个验收 case 满足 `T_NPU ≤ T_GPU / 0.35`（msprof 采集口径） | 任务书 §3.3 |

## 5.2 兼容性分析

新增接口，不涉及既有行为变更；与仓内工程模式一致；NPU 与 CUDA 参数序列一致（仅命名与枚举类型差异，枚举值对齐）；handle/stream 复用既有实现。

## 5.3 可维护性与可测试性概述

代码按 host/kernel 分层组织，公共逻辑集中复用；接口契约与约束在头文件与文档中声明。自验覆盖：基础功能、uplo、padding、info、批量边界、空问题、确定性与非默认 stream，按任务书自验要求执行并输出自测报告。

# 附录 A：参考资料

- 《Atlas 950 单精度实数 Cholesky 分解、求解和批量接口 任务书》
- CUDA cuSolver legacy API 文档
- 生态算子开源精度标准（opbase）
- ops-solver 仓与 AscendOpTest
