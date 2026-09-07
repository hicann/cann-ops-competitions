# aclsparseSpSM A2/A3 (arch22) 算子设计文档

| 项目 | 内容 |
|------|------|
| 任务名称 | 9月社区任务-aclsparseSpSM算子开发(A2/A3) |
| 参与者账号 | GitCode：`gcw_93rlw6ed` |
| 目标硬件 | Atlas A2/A3（arch22 / DAV_2201）；实测 Atlas A2 训练系列（910B3/910B4）及 A3 |
| CANN 版本 | CANN 9.1.0 及后续配套版本 |
| 交付代码仓 | 目标合入 [cann/ops-sparse](https://gitcode.com/cann/ops-sparse)；实现路径 `sparse/spsm/arch22/` |
| 设计提交路径 | `04_tasks/01_community-task-2026/tasklist/09-1-aclsparseSpSM-A2A3/gcw_93rlw6ed/docs/design.md` |
| 对标接口 | cuSPARSE Generic `cusparseSpSM`（含 `cusparseSpSM_updateMatrix`） |
| 文档版本 | v1.0 |

---

## 1. 需求背景（required）

### 1.1 需求来源

社区任务「9月社区任务-aclsparseSpSM算子开发(A2/A3)」：参考 cuSPARSE SpSM 逻辑，通过 ATen Dispatcher 完成 A2/A3 平台多右端稀疏三角求解 NPU 适配，开发 `aclsparseSpSM` 算子，支持 FP32 和 complex64 数据类型。

### 1.2 背景介绍

#### 1.2.1 aclsparseSpSM 算子实现现状分析

`ops-sparse` 仓库当前（master）已存在 `sparse/spsm/arch35/` 的 SpSM 实现，目标平台为 Ascend 950PR/950DT（arch35 / DAV_3510），能力如下：

| 参数 | 当前 arch35 支持 | 本任务 arch22 目标 |
| --- | --- | --- |
| matA 格式 | CSR | **CSR / CSC / COO** |
| 数据类型 | FP32 | **FP32 / complex64** |
| opA | N / T | **N / T / H**（H 仅 complex64 有意义） |
| opB | N（签名保留，未实现 T/H） | **N / T / H** |
| alpha pointer mode | HOST | **HOST / DEVICE** |
| diag/fill | UNIT / NON_UNIT，LOWER / UPPER | 同左 |
| B/C 布局 | ROW / COL | 同左 |
| UpdateMatrix | 无 | **新增 GENERAL / DIAGONAL** |
| index base | 0 / 1 | 同左 |

结论：**本任务以 arch35 现有实现为算法基线，新增 arch22 差异层，并补齐 complex64、CSC/COO 格式归一、opB T/H、Device pointer mode 与 UpdateMatrix 能力**。

#### 1.2.2 aclsparseSpSM 功能分析

求解稀疏三角线性方程组（多右端）：

$$op(A) \cdot C = \alpha \cdot op(B)$$

其中 A 为 $m\times m$ 稀疏三角方阵，B 为 $m\times n$ 稠密右端矩阵，C 为 $m\times n$ 稠密解矩阵，$\alpha$ 为标量。

多阶段调用模型：`CreateDescr → BufferSize → Analysis → SpSM(Solve) → UpdateMatrix → DestroyDescr`。Solve 阶段全异步执行（算子内部禁止 `aclrtSynchronizeStream`），主计算在 NPU 调用流运行。

## 2. 需求分析（required）

### 2.1 需求描述

使用 Ascend C 编程语言在 arch22（DAV_2201）实现 aclsparseSpSM 算子，交付 `ops-sparse` 公开 C++ Legacy API、Host 多阶段状态管理、arch22 Ascend C Kernel、C++ UT/ST、性能脚本与文档，主计算在 NPU 调用流运行。

### 2.2 需求拆解

1. 支持 FP32、complex64 数据类型，complex64 下 H 路径执行共轭转置语义
2. 支持 CSR、CSC、COO 稀疏格式，Device 索引统一 I32、base 0/1
3. 支持 opA/opB 的 N/T/H 操作
4. 支持 LOWER/UPPER、UNIT/NON_UNIT 三角属性
5. 支持 handle 的 Host 与 Device pointer mode，alpha 依 mode 读取
6. 实现 `BufferSize → Analysis → aclsparseSpSM → UpdateMatrix` 多阶段调用；BufferSize/Analysis 允许 B/C 描述符的 values 为空
7. Analysis 缓存 A 格式、三角属性、opA/opB、dtype、尺寸、索引、B/C layout、RHS 数量与 workspace 绑定状态，跨阶段保持一致
8. 新增 `aclsparseSpSMUpdateMatrix` 与 `aclsparseSpSMUpdate_t`（GENERAL / DIAGONAL）
9. 处理未排序输入索引，确定性规范化路径实现 bitwise deterministic；B/C 可同指针原地求解
10. 主求解与格式相关计算由 NPU Kernel 执行，禁止 CPU fallback
11. 性能：P-01/P-02/P-03 各场景性能倍率 ≥ 0.25× GPU 标杆（FP32 + complex64）
12. 内存：workspace ≤ L2 Cache，或 NPU 额外内存 ≤ GPU 的 50%

## 3. 详细设计（required）

### 3.1 算子分析

#### 3.1.1 数学公式

$$op(A) \cdot C = \alpha \cdot op(B)$$

其中：

- $op(X)=X$（N），$op(X)=X^T$（T），$op(X)=X^H$（H，共轭转置，仅 complex64 有意义）
- 以 LOWER + NON_UNIT 前向代入为例，逐行：

$$C[i,:] = \frac{\alpha \cdot B[i,:] - \sum_{j<i} A[i,j]\cdot C[j,:]}{A[i,i]}$$

- UNIT 对角时 $A[i,i]\equiv 1$，省略除法；UPPER 为后向代入。

#### 3.1.2 支持数据类型

- FP32（`ACL_FLOAT`）
- complex64（`ACL_COMPLEX64`，实部/虚部相邻存储，二进制兼容 `float2`，kernel 侧 `valFloatFactor=2`）

#### 3.1.3 支持形状

- matA：$[m,m]$，动态 m/nnz
- matB：$[m,n]$，动态 RHS 列数 n 与 leading dimension
- matC：$[m,n]$，可与 matB 同 Device 指针原地求解

### 3.2 算子实现

#### 3.2.1 总体方案

复用 arch35 已闭环的算法主干（host level scheduling + NPU level-scheduling Solve kernel + vector transpose kernel），新增 arch22 差异层，并扩展以下能力：

1. **格式归一**：CSC/COO 在 Analysis 阶段 host 侧归一化为 CSR，Solve kernel 仅处理 CSR 单一格式，减少 kernel 变体。
2. **complex64**：kernel 侧模板化/特化 complex 运算（乘、减、共轭、除）；opA=H 在转置基础上对 values 共轭。
3. **opB T/H**：Solve 前按 opB 对 B 做布局/共轭处理，统一走 ROW 主计算路径。
4. **Device pointer mode**：alpha 在 Device 时经 `aclrtMemcpy` 读取到 host 后参与 tiling 与计算。
5. **UpdateMatrix**：新增接口与枚举，GENERAL 刷 values + 依赖，DIAGONAL 只刷对角。

#### 3.2.2 Host 侧设计

##### 1. 多阶段状态管理

**Create（`aclsparseSpSMCreateDescr`）**：堆上分配 `aclsparseSpSMDescr` 内部状态，初始化默认值。

**BufferSize（`aclsparseSpSMBufferSize`）**：
- 校验 handle/spsmDescr/alpha 非空；pointer mode；matA/B/C 描述符；opA/opB；format；dtype；diag/fill；alg；shape；ld。
- 计算 workspace 偏移（`ComputeSpsmWsOffsets`），按 opA=T + NON_UNIT 最大值预留。
- 缓存 `cachedBufferSize`，供 Analysis 一致性校验。
- 允许 matB/matC 的 values 为空（描述符本身保持有效），仅按 shape/layout 计算。

**Analysis（`aclsparseSpSMAnalysis`）**：
- 公共前置校验 + buffer 大小一致性校验。
- 格式归一分支：
  - CSR：D2H rowOff/colInd/values → host level scheduling。
  - CSC：D2H colOff/rowInd/values → 转为等价 CSR（host 侧 CSC→CSR 转置）。
  - COO：D2H rowInd/colInd/values → 确定性排序（row 主键 + col 次键，重复坐标合并）→ CSR。
- opA=T：host 侧 CSR→CSC 转置（=Aᵀ 的 CSR），`effectiveFillMode = swap(fillMode)`。
- opA=H：同 T 路径 + host 侧转置时对 complex64 values 共轭（虚部取反）。
- host level scheduling（`SpsmHostComputeRowLevels` + `SpsmHostBuildLevelBuckets`）+ 奇异检测（NON_UNIT 零对角）。
- H2D 写回 levelRowPtr / levelRowIdx / diagVal。
- 计算 kChunkSize，构造并缓存 TilingData（by-value），绑定 buffer，标记 analyzed。

**Solve（`aclsparseSpSM`）**：
- 公共前置校验 + 跨阶段一致性校验（opA/opB/matA/B/C 字段/buffer 绑定与 Analysis 缓存一致）。
- 从缓存取 TilingData，刷新 alpha 后 by-value 传 kernel。
- 异步 launch 序列（同 stream 顺序）：
  1. transpose-in：orderB=COL 或 opB≠N 时 B → denseBuf(ROW)。
  2. solve：主求解 kernel（level scheduling，MemBase）。
  3. transpose-out：orderC=COL 或 opB≠N 时 denseBuf(ROW) → C 目标布局。
- 禁止 `aclrtSynchronizeStream`。

**UpdateMatrix（`aclsparseSpSMUpdateMatrix`）**：
- 校验 spsmDescr 已 analyzed、newValues 非空、dtype 与 matA 一致。
- `GENERAL`：D2H 新 values（dtypeSize 感知）；opA=N 时仅刷新 diagVal（NON_UNIT，按缓存原始对角位置提取）；opA=T/H 时用 Analysis 缓存的归一化原始 CSR 结构重转置 values 写回 workspace transVal（opA=H 虚部取反），并刷新 diagVal。结构不变，无需重新 level scheduling。
- `DIAGONAL`：仅更新对角 values（长度 m）→ 刷新 diagVal（NON_UNIT，opA=H 共轭虚部）；off-diag 不变，无需刷新 transVal。

**Destroy（`aclsparseSpSMDestroyDescr`）**：释放描述符内存。

##### 2. 参数校验策略

- handle/spsmDescr/alpha nullptr 校验（空句柄返回 `ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR`）。
- opA/opB 枚举校验：N/T/H 合法，其余返回 `ACL_SPARSE_STATUS_NOT_SUPPORTED`。
- 格式校验：CSR/CSC/COO；dtype 一致性（matA/matB/matC/computeType/alpha，FP32 或 complex64）。
- diag/fill 校验：UNIT/NON_UNIT + LOWER/UPPER；shape 校验（A 方阵，B/C 行列匹配）；ld 校验（ROW→ld≥cols，COL→ld≥rows）；index base 0/1；alg 仅 DEFAULT。
- pointer mode：Host 直接解引用 alpha；Device 走 `aclrtMemcpy` 到 host。
- 跨阶段一致性：Solve/UpdateMatrix 时字段与 Analysis 缓存一致。

##### 3. Workspace 布局设计

```
[ 0             , levelRowPtrOff )  : 64B header
[ levelRowPtrOff, levelRowIdxOff )  : int32 levelRowPtr[L+1]
[ levelRowIdxOff, diagValOff     )  : int32 levelRowIdx[m]
[ diagValOff    , transRowOffOff )  : diagVal[m]  (NON_UNIT)
[ transRowOffOff, transColIndOff )  : int32 transRowOff[m+1]  (opA=T/H)
[ transColIndOff, transValOff    )  : int32 transColInd[nnz]  (opA=T/H)
[ transValOff   , denseBufOff    )  : transValues[nnz]  (opA=T/H)
[ denseBufOff   , sortBufOff     )  : denseBuf[m*n]  (COL 转置缓冲；仅 orderB==COL||orderC==COL||opB≠N 时分配)
[ sortBufOff    , endOff         )  : CSC/COO 排序归一缓冲 (条件分配)
```

- complex64 路径：values 相关区大小 ×2（`sizeof(float2)=2×sizeof(float)`）。
- 对齐：64B（`SPSM_WS_ALIGN=64`）。
- tiling by-value 传递，不落盘 GM。

##### 4. Level Scheduling 设计

- **ComputeRowLevels**：按 solveDir（LOWER→FORWARD，UPPER→BACKWARD）逐行计算 `level[i] = max(level[dep])+1`；对角元素 j==i 记录 diagVal[i]（NON_UNIT），零对角判奇异。
- **BuildLevelBuckets**：三趟（计数 → 前缀和 → 散列写入）构造 levelRowPtr[L+1] 与 levelRowIdx[m]。
- **L = max(level)+1** 为拓扑层数，level 间串行 + `SyncAll` 栅栏，level 内多核并行。

##### 5. CSC/COO 归一化

- CSC：host 侧 CSC→CSR 转置后复用 CSR level scheduling。
- COO：host 侧确定性排序（row 主键、col 次键）+ 重复坐标合并 → CSR。
- 未排序输入：确定性规范化路径，保证 bitwise deterministic。
- 归一化在 Analysis 一次性完成（Host 预处理），Solve 主计算仍在 NPU。

##### 6. complex64 设计

- kernel 模板化/特化 complex64 运算：
  - 乘：$(a+bi)(c+di)=(ac-bd)+(ad+bc)i$
  - 累减：逐元素复数减。
  - 除：$C/div$（diagVal 为实数时简化）或复数除法。
- opA=H：转置后对 values 共轭（虚部取反）。

##### 7. opB T/H 设计

- opB=N：B 原样进入 solve。
- opB=T：B 先做布局转置（host/kernel transpose）再 solve。
- opB=H：T 基础上 complex64 共轭。
- 结果按 opB 语义写回 C 目标布局（COL 时 transpose-out）。

##### 8. Tiling 策略

- tiling by-value 传递，不落盘 GM。
- `SpsmTilingData` 字段：m、n、ldb、ldc、orderB、orderC、needTranspose、effectiveFillMode、diagType、L、kChunkSize、maxRowLen、alpha（host 值）、indexBase、opA/opB 编码、dtype 编码、workspace 偏移。
- tiling key 规划：dtype × format × opA × opB × pointer mode × fill/diag。

##### 9. 分核策略

- 动态获取核数（`GetAivCoreCount()`），禁止硬编码；失败兜底 1。
- 满核优先；核间均分；余块分配到前几个核。

##### 10. 数据分块和内存优化策略

- UB 容量动态获取（`GetUbSize()`）；系统预留 8KB。
- colInd/vals 双缓冲（4 × maxRowLen × 4B 或 ×8B for complex64）。
- in/acc/tmp 缓冲 2 × kChunkAlign × dtypeSize。
- levelRowPtr 由 kernel 从 GM 直接读取（不占 UB，支持大 m）。
- `kChunkSize = min(ubAvail / (4 × dtypeSize), n)`，对齐到 8。

#### 3.2.3 Kernel 侧设计

##### 1. Solve Kernel

Init（解析 tiling + InitBuffer）+ Process（CopyIn/Compute/CopyOut）。

- 按 level 0..L-1 逐层求解，level 间 `SyncAll`。
- 每行 i：加载 colInd[rowOff[i]..] 与 values；对角 dv=diagVal[i]（NON_UNIT）或 1（UNIT）；依赖累减 `sum = Σ A[i,k]·C[k,:]`；`C[i,:] = (α·B[i,:] - sum)/dv`。
- 多 RHS 按 kChunkSize 分块，UB 内向量化。
- complex64：complex Mul/Sub/共轭/除；opA=H 共轭已在 host 侧转置时完成（kernel 不重复共轭）。

##### 2. Transpose Kernel

- SIMT/vector kernel：orderB=COL → denseBuf(ROW)（transpose-in）；denseBuf(ROW) → C(COL)（transpose-out）。
- 多核按行切分，`DataCopyPad` 跨步搬运，支持任意 ld。

##### 3. 格式转换 Kernel

- COO→CSR、CSC→CSR 归一化放在 Analysis host 预处理（一次性开销，非 Solve 热路径）。
- 主求解与格式相关核心计算在 NPU Kernel 执行，禁止 CPU fallback 代替 NPU 实现。

## 4. 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas A2 训练系列（910B3/910B4） | √ |
| Atlas A3 训练/推理系列 | √ |

## 5. 算子约束限制

- A 为二维三角方阵，op(A) 与 op(B) 后的 B/C 维度满足公式；不满足返回确定错误。
- A 的 fill mode、diag type、格式、index base 与每阶段参数保持一致。
- NON_UNIT 对角元素存在且非零。
- A/B/C 与 computeType 采用 ACL_FLOAT 或 ACL_COMPLEX64 的一致组合；Device 索引 I32。
- CSR/CSC/COO 未排序与重复坐标经确定性预处理。
- Host 尺寸元数据用公开描述符参数类型，核心计算与预处理在 NPU 路径完成。
- UpdateMatrix 后依据更新类型刷新关联状态。
- buffer 在异步 Solve 完成前保持有效且不被修改。

## 6. 可维可测分析

### 6.1 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | FP32：CPU Golden 用 float64；rtol=2^-10、atol=2^-16、A=1e-2；匹配率 ≥ 0.99；每元素 ≤ max(A, 32×ULP(golden))；complex64 实部/虚部分别适用 | 任务书 §3.2 |
| 性能标准 | P-01/P-02/P-03 各场景性能倍率 ≥ 0.25× GPU 标杆（FP32 + complex64） | 任务书 §3.3 |
| 内存标准 | workspace ≤ L2 Cache，或 NPU 额外内存 ≤ GPU 的 50% | 任务书 §3.4 |

### 6.2 兼容性分析

- 新增接口 `aclsparseSpSMUpdateMatrix` 与枚举 `aclsparseSpSMUpdate_t`，向后兼容现有 Create/BufferSize/Analysis/Solve/Destroy 流程。
- 公共 Host 层与 arch22/arch35 差异层解耦，便于 A5/950 同主干维护。
- 共享 Host 逻辑与 arch35 差异层同步回归。

## 7. 实现路径与目录结构

```
ops-sparse/
├── include/cann_ops_sparse.h          # UpdateMatrix 枚举 + 接口声明
├── sparse/spsm/arch22/                # arch22 Host + Kernel + tiling
│   ├── spsm.h
│   ├── spsm_host.cpp
│   ├── spsm_kernel.h
│   ├── spsm_kernel.cpp
│   └── spsm_tiling_data.h
├── sparse/spsm/arch35/                # 既有实现（算法基线，同步回归）
└── test/spsm/spsm/arch22/             # C++ UT/ST + 性能脚本
```

编译：`bash build.sh --ops=spsm --soc=ascend910b --run`（以实际环境为准）。
