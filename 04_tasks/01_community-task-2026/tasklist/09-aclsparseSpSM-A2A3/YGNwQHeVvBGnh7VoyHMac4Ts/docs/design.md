# aclsparseSpSM A2/A3 (arch22) 算子设计文档

| 项目 | 内容 |
|------|------|
| 任务名称 | 9月社区任务-aclsparseSpSM算子开发(A2A3) |
| 参与者账号 | `YGNwQHeVvBGnh7VoyHMac4Ts`（GitCode：`chanchp`） |
| 目标硬件 | Atlas A2/A3（arch22 / DAV_2201）；实测 Ascend 910B3 |
| CANN 版本 | CANN 9.1.0 |
| 交付代码仓 | 目标合入 [cann/ops-sparse](https://gitcode.com/cann/ops-sparse)；实现路径 `sparse/spsm/arch22/` |
| 设计提交路径 | `04_tasks/01_community-task-2026/tasklist/09-aclsparseSpSM-A2A3/YGNwQHeVvBGnh7VoyHMac4Ts/docs/design.md` |
| 对标接口 | cuSPARSE Generic `cusparseSpSM` |
| 文档版本 | v1.0 |

---

## 需求背景（required）

### 需求来源

社区任务 "9月社区任务-aclsparseSpSM算子开发(A2A3)"，在 Atlas A2/A3（arch22/DAV_2201）实现多右端稀疏三角矩阵求解。

### 背景介绍

#### aclsparseSpSM 算子实现现状分析

通过对 `ops-sparse` SpSM 路径与 910B3 实测对照，**本任务交付目标能力已实现**（arch22）：

| 参数 | 参数含义 | 支持情况 | 约束 | 形状 |
| --- | --- | --- | --- | --- |
| matA | 稀疏三角方阵 | CSR/CSC/COO；FP32/complex64；I32 索引 | LOWER/UPPER；UNIT/NON_UNIT；index base 0/1 | [m, m]，动态 m/nnz |
| matB | 多 RHS 稠密右端 | FP32/complex64；ROW/COL | op 后与 A 右维匹配 | 动态 RHS / ld |
| matC | 多 RHS 解矩阵 | 同 matB；可与 B 同指针原地 | 同左 | 同左 |
| opA / opB | 运算方式 | N / T / H（H 仅 complex64） | FP32 下 H → NOT_SUPPORTED | 单值 |
| alpha | 缩放标量 | Host + Device pointer mode | 与 dtype 一致 | 1 标量 |
| UpdateMatrix | GENERAL / DIAGONAL | 已实现 | Analysis 后刷新 values/diag | — |

计算公式：$op(A) \cdot C = \alpha \cdot op(B)$

910B3 实测摘要：GTest **129/129**；官方 ATK 精度 **200/200**；P-01..P-03 Event 倍率 **≥0.25×（6/6）**；内存 path2（workspace≤L2）**6/6**。

#### aclsparseSpSM 功能分析

aclsparseSpSM 算子功能：求解稀疏三角线性方程组 $op(A) \cdot C = \alpha \cdot op(B)$，其中 A 为稀疏三角方阵，B、C 为多列稠密矩阵。

多阶段：`CreateDescr → BufferSize → Analysis → SpSM(Solve) → UpdateMatrix → DestroyDescr`。
Solve 全异步（禁止 `aclrtSynchronizeStream`）；COL 布局经 vector transpose（`valFloatFactor` 支持 complex64）。

---

## 需求分析（required）

### 需求描述

使用 Ascend C 编程语言在 arch22 (DAV_2201) 实现 aclsparseSpSM 算子，支持 FP32 和 complex64 数据类型，支持 CSR/CSC/COO 稀疏格式，支持 opA/opB 的 N/T/H 操作，支持 Host/Device pointer mode，支持 UpdateMatrix（GENERAL/DIAGONAL），支持多阶段调用（Create/BufferSize/Analysis/Solve/Destroy）。

### 需求拆解

1. 支持 FP32、complex64 数据类型，complex64 下 H 路径执行共轭转置语义
2. 支持 CSR、CSC、COO 稀疏格式，Device 索引统一 I32、base 0/1
3. 支持 opA/opB 的 N/T/H 操作
4. 支持 LOWER/UPPER、UNIT/NON_UNIT 三角属性
5. 支持 Host/Device pointer mode，alpha 依 mode 读取
6. 支持 BufferSize → Analysis → Solve → UpdateMatrix 多阶段调用
7. 支持 UpdateMatrix GENERAL/DIAGONAL 更新模式
8. 处理未排序输入索引，bitwise deterministic
9. B/C values 可为同一 Device 指针，支持原地求解
10. 主求解和格式相关计算由 NPU Kernel 执行，禁止 CPU fallback
11. 性能：P-01/P-02/P-03 各场景性能倍率 ≥ 0.25× 标杆
12. 内存：workspace ≤ L2 Cache 或额外内存 ≤ 50% GPU

---

## 详细设计（required）

### 算子分析

#### 数学公式

$$op(A) \cdot C = \alpha \cdot op(B)$$

其中：
- $A$ 为 $m \times m$ 稀疏三角方阵（CSR/CSC/COO 格式）
- $B$ 为 $m \times n$ 稠密右端矩阵（多 RHS）
- $C$ 为 $m \times n$ 解矩阵
- $\alpha$ 为标量缩放因子
- $op(X)$ 定义：
  - $op(X) = X$（N, NON_TRANSPOSE）
  - $op(X) = X^T$（T, TRANSPOSE）
  - $op(X) = X^H$（H, CONJUGATE_TRANSPOSE，仅 complex64 有意义）

逐行前向/后向代入求解（以 LOWER + NON_UNIT 为例）：

$$c_{i,j} = \frac{1}{a_{i,i}} \left( \alpha \cdot b_{i,j} - \sum_{k < i} a_{i,k} \cdot c_{k,j} \right)$$

UNIT 对角时省略 $1/a_{i,i}$ 除法。

#### 支持数据类型

- FP32（`ACL_FLOAT`）
- complex64（`ACL_COMPLEX64`，布局 `{float x, float y}`，二进制兼容 `float2`）

#### 支持形状

- matA: $[m, m]$，动态 $m$ 和 $nnz$
- matB: $[m, n]$，动态 RHS 列数 $n$ 和 leading dimension
- matC: $[m, n]$，可与 matB 同指针（in-place）

### 算子实现

#### 3.2.1 Host 侧设计

##### 1. 多阶段状态管理

aclsparseSpSM 采用 Create → BufferSize → Analysis → Solve → UpdateMatrix → Destroy 多阶段调用模型：

**Create (`aclsparseSpSMCreateDescr`)**：在堆上分配 `aclsparseSpSMDescr` 内部状态，初始化默认值。

**BufferSize (`aclsparseSpSMBufferSize`)**：
- 校验参数（handle/spsmDescr/alpha 非空；pointer mode；matA/B/C 描述符；opA/opB；format；dtype；diag/fill；alg；shape；ld）
- 计算 workspace 偏移（`ComputeSpsmWsOffsets`），按 opA=T + NON_UNIT 最大值预留
- 缓存 `cachedBufferSize` 到描述符，供 Analysis 阶段一致性校验
- denseBuf 仅 orderB==COL || orderC==COL 时分配

**Analysis (`aclsparseSpSMAnalysis`)**：
- 公共前置校验（`ExtractSpsmInputs`）
- buffer 大小一致性校验（`ValidateBufferSizeConsistency`）
- 按 matA 格式分支：
  - **CSR**：D2H rowOff + colInd + values → host level scheduling
  - **CSC**：D2H colOff + rowInd + values → 转为等价的 CSR 表示（或独立 CSC level scheduling）
  - **COO**：D2H rowInd + colInd + values → 排序 → 转为 CSR 后 level scheduling
- opA=T 路径：host 侧 CSR→CSC 转置（或 CSC→CSR），计算转置后 maxRowLen
- opA=H 路径：同 T 路径，kernel 侧对 complex64 values 执行共轭
- host level scheduling（`SpsmHostComputeRowLevels` + `SpsmHostBuildLevelBuckets`）
- H2D 写回 levelRowPtr / levelRowIdx / diagVal
- 计算 kChunkSize（UB 容量推算）
- 构造 + 缓存 TilingData（by-value）
- 绑定 active buffer，标记 analyzed

**Solve (`aclsparseSpSM`)**：
- 公共前置校验
- 跨阶段一致性校验（analyzed 标记；opA/opB；matA/B/C 字段；buffer 绑定）
- 从 cachedTiling 取出，刷新 alpha 后 by-value 传入 kernel
- 异步 launch 序列（同 stream 顺序）：
  1. transpose-in：orderB=COL 时 B(COL) → denseBuf(ROW) [SIMT kernel]
  2. solve：by-value tiling，主求解 [MemBase kernel]
  3. transpose-out：orderC=COL 时 denseBuf(ROW) → C(COL) [SIMT kernel]
- 禁止 aclrtSynchronizeStream

**UpdateMatrix (`aclsparseSpSMUpdateMatrix`)**：
- 校验 spsmDescr 已 analyzed；newValues 非空；dtype 与 matA 一致
- `updatePart == GENERAL`：
  - D2H 原 CSR 三数组 → 替换 values → H2D 新 values 到 workspace
  - 重新执行 host level scheduling（结构不变，仅 values 变化）
  - 刷新 diagVal（NON_UNIT）和 transValues（opA=T）
  - 重建 TilingData（maxRowLen 不变，L 不变）
- `updatePart == DIAGONAL`：
  - 仅更新对角 values → 刷新 diagVal
  - 免全量 level scheduling（结构未变）
  - 刷新 TilingData 中 diagVal 相关字段

**Destroy (`aclsparseSpSMDestroyDescr`)**：释放描述符内存。

##### 2. 参数校验策略

- handle/spsmDescr/alpha nullptr 校验（空句柄返回 `ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR`）
- pointer mode 校验：Host 模式直接解引用 alpha；Device 模式走 `aclrtMemcpy` alpha 到 host
- opA/opB 枚举校验：N/T/H 三值合法，其余返回 `ACL_SPARSE_STATUS_NOT_SUPPORTED`
- matA 格式校验：CSR/CSC/COO 三格式
- dtype 一致性校验：matA/matB/matC/computeType/alpha 一致（FP32 或 complex64）
- diag/fill 校验：UNIT/NON_UNIT + LOWER/UPPER
- shape 校验：A 方阵；A.rows == B.rows == C.rows；B.cols == C.cols
- ld 校验：ROW→ld>=cols，COL→ld>=rows
- index base 校验：ZERO/ONE（信任描述符策略）
- alg 校验：仅 DEFAULT
- 跨阶段一致性校验：Solve 时 matA/B/C 字段与 Analysis 缓存一致

##### 3. Workspace 布局设计

```
[ 0             , levelRowPtrOff )  : 64B header
[ levelRowPtrOff, levelRowIdxOff )  : int32 levelRowPtr[L+1]
[ levelRowIdxOff, diagValOff     )  : int32 levelRowIdx[m]
[ diagValOff    , transRowOffOff )  : float diagVal[m]  (NON_UNIT)
[ transRowOffOff, transColIndOff )  : int32 transRowOff[m+1]  (opA=T)
[ transColIndOff, transValOff    )  : int32 transColInd[nnz]  (opA=T)
[ transValOff   , denseBufOff    )  : float transValues[nnz]  (opA=T)
[ denseBufOff   , sortBufOff     )  : float denseBuf[m*n]  (条件分配)
[ sortBufOff    , endOff         )  : CSC/COO 排序缓冲 (条件分配)
```

- denseBuf：仅 orderB==COL || orderC==COL 时分配
- sortBuf：仅 CSC/COO 路径或未排序输入时分配
- complex64 路径：values 区大小翻倍（sizeof(float2) = 2 × sizeof(float)）
- 对齐：64B 对齐（SPSM_WS_ALIGN=64）

##### 4. Level Scheduling 设计

**ComputeRowLevels**：按 solveDir（LOWER→FORWARD，UPPER→BACKWARD）逐行计算层级。

对每行 i：
- 遍历 colInd[rowOff[i]..rowOff[i+1]]
- 对角元素 j==i：记录 diagVal[i]（NON_UNIT），奇异检测（dv==0）
- 依赖元素（LOWER: j<i；UPPER: j>i）：maxLevel = max(maxLevel, levelBuf[j]+1)
- levelBuf[i] = maxLevel

**BuildLevelBuckets**：
- Pass 1：计数每 level 行数
- Pass 2：前缀和 → levelRowPtr[L+1]
- Pass 3：散列写入 levelRowIdx[m]

**L = max(levelBuf) + 1**（拓扑层数）

##### 5. CSC/COO 路径规划

**CSC 路径**：
- matA 描述为 CSC（colOffsets, rowInd, values）
- 方案 A：host 侧 CSC→CSR 转置后复用现有 CSR level scheduling
- 方案 B：独立 CSC level scheduling（按列层级，适合 UPPER）
- 推荐方案 A（复用成熟路径，减少 kernel 变体）

**COO 路径**：
- matA 描述为 COO（rowInd, colInd, values）
- host 侧排序（按 row 为主键、col 为次键）→ 转为 CSR
- 未排序输入：确定性排序（std::sort + 稳定排序保证 bitwise deterministic）
- 重复坐标：累加合并或取最后值（需明确语义，推荐累加）

**complex64 路径**：
- 新增 complex64 特化 kernel（或模板参数化）
- opA=H 时：转置 + 共轭（values.x 不变，values.y 取反）
- 乘法：complex64 乘法（(a+bi)(c+di) = (ac-bd) + (ad+bc)i）
- 除法：complex64 除以实数（diagVal 为实数时）或 complex 除法

##### 6. UpdateMatrix 设计

**GENERAL 更新**：
1. D2H 原 CSR rowOff（结构不变，无需重新 D2H）
2. H2D 新 values 到 workspace transVal 区
3. 重新执行 host ComputeRowLevels（values 变化影响 diagVal 和奇异检测）
4. 刷新 diagVal（NON_UNIT）和 transValues（opA=T）
5. 重建 TilingData（maxRowLen 和 L 不变）

**DIAGONAL 更新**：
1. 仅提取对角 values（D2H 或 Device 直接替换）
2. 刷新 diagVal
3. 免全量 level scheduling
4. 刷新 TilingData 中 diagVal 相关字段

##### 7. Tiling 策略

- tiling by-value 传递，不落盘 GM
- `SpsmTilingData` 字段：m, n, ldb, ldc, orderB, orderC, needTranspose, effectiveFillMode, diagType, L, kChunkSize, maxRowLen, alpha_host, indexBase, workspace 偏移
- tiling key 规划：
  - 按 dtype（FP32 / complex64）
  - 按 format（CSR / CSC / COO）
  - 按 opA（N / T / H）
  - 按 opB（N / T / H）
  - 按 pointer mode（Host / Device）

##### 8. 分核策略

- 动态获取核数（`GetAivCoreCount()`），禁止硬编码
- 获取失败兜底为 1（保守下限，Solve kernel 依赖 SyncAll）
- 满核优先；核间均分；余块分配到前几个核

##### 9. 数据分块和内存优化策略

- UB 容量动态获取（`GetUbSize()`）
- 系统预留 8KB（kSpsmUbReserved）
- diagBuf 固定 32B
- colIndQue_ + valsQue_：4 × maxRowLen × 4B（双缓冲）
- inQue_：2 × kChunkAlign × 4B
- accBuf_ + tmpBuf_：2 × kChunkAlign × 4B
- levelRowPtrBuf_：align32((L+1) × 4B)
- kChunkSize = min(ubAvail / (4 × sizeof(float)), n)，对齐到 8

#### 3.2.2 Kernel 侧设计

##### 1. Solve Kernel

进行 Init 和 Process 两个阶段，Process 包括 CopyIn、Compute、CopyOut 三个阶段。

**Init**：
- 解析 by-value TilingData
- InitBuffer：colIndQue_, valsQue_, inQue_, accBuf_, tmpBuf_, diagBuf_, levelRowPtrBuf_
- 预加载 levelRowPtr 到 UB（避免裸 GM 标量读取）

**Process**：
- 按 level 逐层求解（levelRowPtr[l]..levelRowPtr[l+1]）
- 每行 i：
  - CopyIn：加载 colInd[rowOff[i]..rowOff[i+1]] 和 values
  - Compute：
    - 对角元素：dv = diagVal[i]（NON_UNIT）或 1.0（UNIT）
    - 依赖求和：sum = Σ a[i,k] × c[k,j]（k 为依赖列）
    - c[i,j] = (alpha × b[i,j] - sum) / dv
  - CopyOut：写回 c[i,j]
- 多 RHS 列按 kChunkSize 分块处理

**complex64 特化**：
- 乘法：complex64 Mul
- 减法：complex64 Sub
- 除法：complex64 除以实数（diagVal 为实数时简化）
- opA=H：转置后对 values 共轭

##### 2. Transpose Kernel

- SIMT kernel，用于 orderB=COL 或 orderC=COL 时的稠密矩阵转置
- B(COL) → denseBuf(ROW)：transpose-in
- denseBuf(ROW) → C(COL)：transpose-out

##### 3. 格式转换 Kernel（可选）

- COO→CSR 排序 kernel（NPU 实现，避免 host CPU 瓶颈）
- CSC→CSR 转置 kernel（NPU 实现）
- 或走 host CPU 预处理（Analysis 一次性开销）

---

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas 800I A2 (910B3/910B4) | √ |
| Atlas A3 (DAV_2201) | √ |

---

## 算子约束限制

- A 为二维三角方阵，且 op(A) 与 op(B) 后的 B/C 维度满足公式；不满足时返回确定错误
- A 的 fill mode、diag type、格式、index base 与每阶段参数保持一致
- NON_UNIT 的对角元素存在且非零
- A/B/C 和 computeType 采用 ACL_FLOAT 或 ACL_COMPLEX64 的一致组合
- Device row/column index 为 I32
- CSR/CSC/COO 的未排序与重复坐标经确定性预处理
- Host 尺寸元数据使用公开描述符参数类型，核心计算与预处理在 NPU 路径完成
- UpdateMatrix 后依据更新类型刷新关联状态
- buffer 在异步 Solve 完成前保持有效且不被修改

---

## 可维可测分析

### 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | FP32: CPU Golden 用 float64；rtol=2^-10, atol=2^-16, A=1e-2；匹配率 ≥ 0.99；每元素 ≤ max(A, 32×ULP)；complex64 实部/虚部分别适用 | 任务书 §3.2 |
| 性能标准 | P-01/P-02/P-03 各场景性能倍率 ≥ 0.25× 标杆（FP32 + complex64） | 任务书 §3.3 |
| 内存标准 | workspace ≤ L2 Cache 或额外内存 ≤ 50% GPU | 任务书 §3.4 |

### 兼容性分析

- 新增接口 `aclsparseSpSMUpdateMatrix` 和枚举 `aclsparseSpSMUpdate_t`，向后兼容现有 Create/BufferSize/Analysis/Solve/Destroy 流程
- 公共 Host 层与 arch22 差异层解耦，便于 A5/950 同主干维护
- 共享 Host 逻辑与 arch35 差异层同步回归

---

## 附录：arch22 落地优先级与建议目录

### 落地优先级（相对初始缺口；**当前均已闭合除标注外**）

| 优先级 | 项目 | 当前状态 |
|---|---|---|
| **P0** | CSR + FP32 + opA=N/T + opB=N + Host | **已闭合**（910B3） |
| **P0** | Device pointer mode | **已闭合** |
| **P0** | UpdateMatrix GENERAL/DIAGONAL | **已闭合** |
| **P0** | complex64 + op H + COL transpose | **已闭合** |
| **P0** | opB T/H | **已闭合** |
| **P0** | CSC / COO | **已闭合**（Analysis 归一化） |
| **P1** | 未排序 + 重复坐标确定性规范化 | **已闭合**（Analysis 行内稳定排序+同 col 求和；专项 UT PASS） |
| **P1** | FP32 高精度累加 | 匹配率已达标；未单独做 Kahan |
| **P2** | CSC/COO NPU 排序 kernel | 仍为 Host 预处理 |

### 建议目录结构（已落地）

```
ops-sparse/
├── include/cann_ops_sparse.h          # UpdateMatrix 枚举+声明
├── sparse/spsm/arch22/                # Host + MemBase Solve + vector transpose
├── sparse/spsm/arch35/                # 同步实现
└── test/spsm/spsm/arch22/             # CSV GTest + Complex64 ROW/COL
```
