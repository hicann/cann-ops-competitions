# aclsparseSpSV 算子设计文档（Atlas A2/A3，arch22）

> 社区任务：9 月社区任务——aclsparseSpSV 算子开发（A2/A3）
> 开发仓库：`gitcode.com/zhangfeng1133/ops-sparse`（分支 `aclsparseSpSV-arch22-a2a3`，源码按任务书约定组织于 `sparse/spsv/arch22/`，目标以 PR 合入 [`cann/ops-sparse`](https://gitcode.com/cann/ops-sparse) `master`）
> 目标平台：Atlas A2（910B3/910B4）/ Atlas A3，DAV_2201（arch22）
> 实测环境：910B3（ascend910_9391）+ CANN 8.5.1；SyncAll 平台结论另经 CANN 9.1.0 工具链交叉验证一致

# 需求背景（required）

## 需求来源

- 来源：aclsparseSpSV 社区开源算子开发任务（A2/A3 验收口径，硬件 arch22/DAV_2201），
  任务书《aclsparseSpSV 算子开发(A2/A3)任务书》，开发仓库
  `gitcode.com/zhangfeng1133/ops-sparse`（分支 `aclsparseSpSV-arch22-a2a3`）。
- 算子在 `ops-sparse` 仓内交付：公开 C++ Legacy API、Host 参数校验与生命周期管理、
  arch22 Ascend C Kernel、C++ UT/ST 与性能脚本；主计算全程运行于 NPU 调用流，
  禁止 CPU fallback。
- 设计文档按社区任务
  [设计文档模板](https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md)
  编写，以 PR 形式提交至 `cann-ops-competitions` 任务目录；算子实现以独立代码 PR
  提交至 `cann/ops-sparse` 的 `master` 分支。

## 背景介绍

### aclsparseSpSV 算子实现优化

SpSV（Sparse Triangular Solve - Vector）为稀疏库三角求解接口。`ops-sparse` 仓内
已有 arch35（Ascend 950PR，SIMT 编程模型）母本实现；Atlas A2/A3（arch22/DAV_2201）
上无对应实现，本任务使用 Ascend C 编程语言在 AIV（AI Vector）编程模型下完成
arch22 移植与优化，功能与性能以 GPU 侧 cuSPARSE `cusparseSpSV` Generic API 作为
对照标杆（任务书 3.3 定义性能倍率的标杆）。

#### 参考实现路径

- 语义与生命周期参考：cuSPARSE `cusparseSpSV`（createDescr → bufferSize →
  analysis → solve → updateMatrix → destroyDescr，两阶段执行 + 值更新）；
- 算法母本：`ops-sparse` 仓 `sparse/spsv/arch35/`（Level Scheduling 前代/回代、
  workspace 布局、CSC 零拷贝、COO/SLICED_ELL 归一化、转置物化、numLevels 运行时
  回写等骨架整体移植）；
- 本仓 arch22 实现路径：`sparse/spsv/arch22/`（spsv.h / spsv_host.cpp /
  spsv_kernel.cpp / spsv_kernel.h / spsv_tiling_data.h）；
- 公开接口：`include/cann_ops_sparse.h`（原型逐字一致，不含本任务新增签名）。

#### aclsparseSpSV 算子实现现状分析

C API 共 6 个 SpSV 专属接口（另有 `aclsparseSpMatSetAttribute/GetAttribute`
设置/读取 fillMode、diagType 属性，为库公共接口）：

| 接口名 | 功能简述 |
| --- | --- |
| aclsparseSpSV_createDescr | 创建 SpSV 描述符（纯 Host） |
| aclsparseSpSV_destroyDescr | 销毁 SpSV 描述符（NULL 幂等） |
| aclsparseSpSV_bufferSize | 查询 workspace 字节数（纯 Host 算术；vecX/vecY 可为 NULL） |
| aclsparseSpSV_analysis | 预处理：格式归一化 + Level Scheduling，绑定 workspace（Device kernel；vecX/vecY 可为 NULL） |
| aclsparseSpSV_solve | 按 level 顺序执行前代/回代求解（Device kernel，可多次调用，支持 in-place） |
| aclsparseSpSV_updateMatrix | 更新 A 的 values（GENERAL 全量/DIAGONAL 对角），无需重新 analysis |

计算公式：`op(A) · Y = alpha · X`

| 参数 | 参数含义 | 数据类型 | 支持数据类型 | 约束 | 形状 |
| --- | --- | --- | --- | --- | --- |
| handle | 库句柄（调用流与 pointer mode） | aclsparseHandle_t | - | 须先 aclsparseSetStream | - |
| opA | A 的操作 | aclsparseOperation_t | N / T / H（CONJUGATE_TRANSPOSE） | float32 下 H 等价 T；complex64 执行共轭 | 标量 |
| alpha | 缩放系数 | const void* | FP32 / complex64 | Host 或 Device 内存（aclsparseSetPointerMode） | 标量 |
| matA | 稀疏三角方阵 | aclsparseConstSpMatDescr_t | values: FP32 / complex64；索引: I32 | CSR/CSC/COO/SLICED_ELL；方阵；base 0/1；fillMode LOWER/UPPER；diagType UNIT/NON_UNIT | m×m |
| vecX | 右端向量 | aclsparseConstDnVecDescr_t | FP32 / complex64 | Device values；BufferSize/Analysis 阶段可为 NULL | op(A) 列数（=m） |
| vecY | 解向量 | aclsparseDnVecDescr_t | FP32 / complex64 | Device values；可与 vecX 同指针（in-place）；Solve 阶段不可 NULL | op(A) 行数（=m） |
| computeType | 计算类型 | aclDataType | ACL_FLOAT / ACL_COMPLEX64 | 须与 A/X/Y 一致 | 标量 |
| alg | 算法 | aclsparseSpSVAlg_t | 仅 ACL_SPARSE_SPSV_ALG_DEFAULT | 其余返回 NOT_SUPPORTED | 标量 |
| spsvDescr | 跨阶段状态 | aclsparseSpSVDescr_t | - | Create 后有效；Solve 前须完成 Analysis | 标量 |
| externalBuffer | workspace | void* | 字节缓冲 | Device 内存，≥ bufferSize 字节，512B 对齐；Analysis 至异步 Solve 完成期间有效 | 1 维 |
| newValues / updatePart | 值更新 | void* / enum | FP32 / complex64；GENERAL / DIAGONAL | dtype 与长度为调用方契约（见约束限制） | nnz 或 m |

### aclsparseSpSV 算子功能分析

算子功能：求解稀疏三角线性方程组 `op(A) · Y = alpha · X`（前代/回代）。

- 输入：matA（四种稀疏格式之一 + 三角属性 + op + base）、vecX、alpha；
- 输出：vecY（独立或与 vecX 同址原地求解）；
- 支持值类型：FP32（ACL_FLOAT）与 complex64（ACL_COMPLEX64，对实部/虚部执行
  complex64 语义，H 路径执行共轭）；Device 索引固定 I32；
- 支持索引 base：0-based / 1-based（Fortran），analysis 阶段归一化为 0-based；
- 确定性：未排序/重复坐标经确定性预处理规范化，相同输入重复执行结果
  bit-wise 一致；NON_UNIT 缺失或零对角按 IEEE-754 传播 INF/NAN；
- A、X 输入全程只读。

# 需求分析（required）

## 需求描述

使用 Ascend C 编程语言在 arch22（DAV_2201，AIV 编程模型）实现 aclsparseSpSV
算子：支持 CSR/CSC/COO/SLICED_ELL 四种格式、FP32/complex64 两种值类型、
LOWER/UPPER、UNIT/NON_UNIT、N/T/H、base 0/1、Host/Device pointer mode、
in-place 求解与 GENERAL/DIAGONAL 值更新；实现 BufferSize → Analysis → Solve →
UpdateMatrix 完整生命周期；主求解与格式相关计算全部由 NPU Kernel 异步执行，
禁止 CPU fallback；精度满足《生态算子开源精度标准》并对齐 cuSPARSE 语义；
性能达到每个场景 ≥ 0.25× GPU 标杆（实测 12.1x ~ 18.2x）。

## 需求拆解

1. 支持 CSR / CSC / COO / SLICED_ELL 四种 A 描述符格式与
   `ACL_SPARSE_SPSV_ALG_DEFAULT` 算法枚举；
1. 支持 LOWER/UPPER、UNIT/NON_UNIT、`NON_TRANSPOSE`/`TRANSPOSE`/
   `CONJUGATE_TRANSPOSE`；complex64 的 H 路径执行共轭转置语义；
1. 支持 Host / Device 两种 pointer mode（alpha 在相应位置读取），X/Y values
   位于 Device 内存，Y 可与 X 同 Device values 指针原地求解；
1. 实现 `BufferSize → Analysis → Solve → UpdateMatrix` 生命周期：BufferSize 与
   Analysis 允许 vecX/vecY 描述符为 NULL，Solve 要求非 NULL 且 values 有效；
   Analysis 缓存格式/三角属性/op/dtype/维度/索引与 workspace 绑定状态，参数
   变化后须重新 Analysis，buffer 在异步 Solve 完成后释放；
1. `GENERAL` 更新替换全部 values 并刷新依赖，`DIAGONAL` 更新替换对角 values，
   更新后复用同一描述符直接 Solve（无需重新 Analysis）；
1. 支持未排序索引，以确定的规范化遍历与求解顺序获得 bit-wise deterministic
   结果；NON_UNIT 缺失/零对角传播 INF/NAN；
1. 三角结构、对角属性、维度、索引范围、dtype、pointer mode 与 workspace 在
   Host 侧校验并返回确定错误码；索引范围校验驻留 Device 的部分由 analysis
   kernel 在 Device 侧执行；
1. 全部 Kernel 使用调用 handle 的 stream 异步执行；精度 200 例（ATK）ALL PASS、
   UT 全绿、性能 P-01/02/03 × base0/1 六场景全部 ≥ 0.25× 标杆、workspace
   ≤ 目标硬件 L2 Cache 容量。

## 支持矩阵

| 维度 | 支持范围 |
| --- | --- |
| 值类型 | ACL_FLOAT（4B）、ACL_COMPLEX64（8B，re/im 各 float） |
| 索引类型 | ACL_SPARSE_INDEX_32I（rowPtr 与 colInd 均 I32；I64 拒绝并返回 NOT_SUPPORTED） |
| 索引 base | ZERO（0 基）、ONE（1 基，analysis 归一化为 0-based workspace 副本） |
| 稀疏格式 | CSR（主路径）、CSC（零拷贝重映射）、COO（归一化 CSR 副本）、SLICED_ELL（归一化 CSR 副本） |
| 矩阵操作 | NON_TRANSPOSE、TRANSPOSE、CONJUGATE_TRANSPOSE（float32 下 H ≡ T；complex64 共轭） |
| 三角属性 | fillMode LOWER/UPPER × diagType UNIT/NON_UNIT |
| 形状 | m×m 方阵，动态 m/nnz、空行、长尾、未排序坐标；m、nnz 可为 0 |
| pointer mode | HOST / DEVICE（alpha 相应位置读取） |
| 输出语义 | vecY 独立输出或与 vecX 同 Device values 指针 in-place；A、X 只读 |
| 确定性 | 相同输入重复执行 bit-wise 一致 |
| workspace | BufferSize 查询、Analysis 绑定、异步 Solve 完成前有效；上界 ≤ L2 192MB |
| 更新 | ACL_SPARSE_SPSV_UPDATE_GENERAL / UPDATE_DIAGONAL，更新后免重新 Analysis |

# 详细设计（required）

## 算子分析

### 数学公式

`op(A) · Y = alpha · X`，A 为 m×m 稀疏三角方阵。

- op(A) = A（N）、A^T（T）、A^H（H；complex64 取共轭，float32 等价 T）；
- 求解方向由有效 fillMode 与有效 opA 共同决定（CSC 重映射与转置归一化后）：

| 有效 fillMode | 有效 opA | 求解方向 |
| :---: | :---: | :---: |
| LOWER | NON_TRANSPOSE | forward（前代） |
| LOWER | TRANSPOSE / CONJ | backward（回代） |
| UPPER | NON_TRANSPOSE | backward（回代） |
| UPPER | TRANSPOSE / CONJ | forward（前代） |

- UNIT 对角：对角恒为 1，跳除法；NON_UNIT：`y[i] = (alpha·x[i] − Σ_{j≠i} a[i][j]·y[j]) / a[i][i]`，
  缺失/零对角按 IEEE-754 产生 INF/NAN（与 cuSPARSE 一致，不特殊防御）；
- nnz = 0：UNIT 时 Y = alpha·X（scale copy），NON_UNIT 时 Y = 0（fill zero），
  均不启动常规求解 kernel。

### 支持数据类型

| matA 值类型 | vecX / vecY 值类型 | computeType | 元素宽度 |
| --- | --- | --- | --- |
| ACL_FLOAT | ACL_FLOAT | ACL_FLOAT | 4B |
| ACL_COMPLEX64 | ACL_COMPLEX64 | ACL_COMPLEX64 | 8B（re/im 各 float） |

Device 索引一律 I32；Host 尺寸元数据使用接口声明的 int64_t。complex64 的
乘/除由 kernel 内 AIV 复数辅助函数完成（复乘复除按实部/虚部分解），H 路径按
`conjNeeded` 标志在读取侧共轭。

### 支持形状

A 为 m×m 方阵（Host 校验 rows == cols），m、nnz 动态，可为 0；vecX 元素数 =
op(A) 列数、vecY 元素数 = op(A) 行数（方阵下均为 m，Solve 阶段校验）；
一维连续向量语义，不支持广播。

## 算子实现

### 实现方案

总体结构：Host 侧完成参数校验、生命周期状态机与 workspace 布局计算，以
`SpsvTilingData` 按值传参下发；Device 侧为 AIV Ascend C kernel，analysis 阶段
完成格式归一化与 Level Scheduling，solve 阶段按 level 顺序单核标量求解。跨
kernel/跨核共享数据以 `DcciFlushRange`（D-cache 清刷）发布。算法骨架（Level
Scheduling、validCount、CSC 零拷贝、COO/SELL 归一化、转置物化、numLevels
运行时回写、updateMatrix、零 nnz 快路径）整体移植自 arch35 母本，
`SpsvTilingData` 字段序与母本对齐。

```text
aclsparseSpSV_*（公开 C++ API，include/cann_ops_sparse.h）
  ├─ Host（spsv_host.cpp）：参数校验 → 生命周期状态机（descr 缓存）
  │    → workspace 布局计算（512B/64B 对齐）→ SpsvTilingData 构建 → 起核分派
  └─ 异步下发 arch22 AIV Kernel（调用 handle 的 stream，spsv_kernel.cpp）
       ├─ analysis：格式归一化（COO/SELL/1-based → 0-based CSR 副本；CSC 零拷贝
       │    重映射；T/H 转置物化）→ level 直方图 → levelRow → diagPtr/validCount
       │    → numLevels、wsErrorFlag 回写 workspace 头部
       ├─ solve：V1 单核遍历全部 level，逐行标量前代/回代（8 模板分派）；
       │    零 nnz 快路径 scale_copy / fill_zero
       └─ update：update_values / update_diag(_csr) 值重排拷贝
            ↓
调用方 vecY 更新（独立或 in-place）
```

#### 3.2.1 host 侧设计：

**参数校验与生命周期**（`spsv_host.cpp`）：handle/alpha/bufferSize 等非空校验；
opA ∈ {N, T, H}、alg = DEFAULT、computeType ∈ {ACL_FLOAT, ACL_COMPLEX64} 且与
A/X/Y 一致；matA 方阵、rowPtr/colInd 索引均为 I32（I64 返回 NOT_SUPPORTED）；
Solve 阶段 vecX/vecY 非 NULL 且 values 为有效指针、元素数 ≥ m；spsvDescr 须已
完成 Analysis 且缓存参数与本次调用一致（不一致返回 INVALID_VALUE）；校验失败
统一 OP_LOGE 并返回确定错误码（SUCCESS / HANDLE_IS_NULLPTR / INVALID_VALUE /
NOT_SUPPORTED / ALLOC_FAILED）。

Analysis 完成后 descr 缓存运行时常量（cachedM/cachedNnz/cachedFormat/
cachedFillMode/cachedDiagType/cachedOpA/cachedIdxBase/cachedComputeType 及
workspace 指针与内部偏移），updateMatrix/零 nnz solve 在无 matA 入参的路径上
凭缓存重建正确值宽度的 tiling（arch22 增量字段 `cachedComputeType`）。

**workspace 策略**（`ComputeWorkspaceSize`）：

- 首部 512B 为 `SpsvTilingData` 运行时数据区：analysis kernel 将 `numLevels`
  与 `wsErrorFlag` 回写至此，solve kernel 从 workspace 读取（而非 kernel 参数）；
- 调度区（恒需）：levelPtr (m+1)×I32、levelRow m×I32、diagPtr m×I32、
  validCount m×I32；validCount 为每行有效依赖计数，unsorted CSR 场景按列索引
  排序前缀跳过已处理依赖项，避免冗余遍历；
- CSR 副本区（COO / SLICED_ELL / 1-based 输入时）：rowPtr (m+1)×I32 +
  colInd nnz×I32 + values nnz×elem + perm，将输入归一化为 0-based CSR；
- 转置副本区（T/H 操作时）：转置 CSR 的 rowPtr/colInd/values/perm（complex64
  的 CSC+H 组合走 CSC 派生矩阵 M + `conjNeeded` 共轭标志，不建转置副本）；
- 软同步标志区：usedCores × 32B（V2 多核预留，V1 未用）；
- 对齐：workspace 首 512B 对齐（GM 基址），内部子区 64B 对齐；nnz = 0 时
  bufferSize 返回 0，workspace 指针可为 NULL。

**tiling 数据结构**（`SpsvTilingData`，按值传 kernel 参数）：m/nnz、alpha 与
alphaIm（complex64 双分量；pointer mode 为 DEVICE 时改由 `alphaDevicePtr`
指向 Device 标量）、fillMode/diagType/opA/format/indexType/idxBase/
computeType、workspace 各子区偏移、numLevels（host 置 0，运行时由 workspace
读）、conjNeeded、numBlocks（核数）、wsErrorFlag 等；字段序对齐 arch35 母本，
arch22 增量字段已注释标注。

##### 1. 分核策略：

V1（首版）**恒单核**：`ComputeNumBlocks` 固定返回 1，单个 AIV 核在一个 kernel
内串行遍历全部 level 与全部行。该定轨是实测结论而非设计妥协——arch22 自定义
AIV kernel 中可用的核间同步原语均不可用（证据链见 3.2.2），每层多 launch 的
stream 排序备份轨又存在跨 kernel 内存一致性的非确定性；单核同核读写 D-cache
自洽，是当前唯一确定性正确的执行形态。多核并行（`kRowsPerCore = 2048` 起核、
深层级集宽度 < 256 回退单核的原始设计）保留在代码与 `SpsvTilingData.syncWsOffset`
字段中，列为 V2（依赖 CANN 工单或迁移框架 launch 路径后重测）。

##### 2. 数据分块和内存优化策略：

- 本算子为不规划（level 长度由矩阵结构决定），不采用定长 tile 双缓冲；kernel
  以 GM Scalar 标量读写直接访问 A/X/Y 与 workspace，避免 gather/scatter 的
  Vector 抽象层开销；
- workspace 即全部中间态：调度区 + 可选 CSR 副本 + 可选转置副本，均在
  BufferSize 一次算清、Analysis 一次绑定，Solve 零额外分配；
- workspace 上界实测 132.1 MB（P-03-base1：m=262,144、nnz=3,932,160、
  complex64、H、base1，含 CSR 副本 + 转置副本 + 调度区），≤ 910B3 L2
  Cache 192 MB，按任务书 3.4 第 2 条规则验收（6/6 场景 PASS）；
- 跨 kernel/跨核共享数据写后 `DcciFlushRange` 清刷 D-cache 发布，保证后续
  kernel 读到最新值（numLevels、updateMatrix 拷贝值等）。

##### 3. tilingkey 规划策略：

本实现以 `SpsvTilingData` 的 computeType（F32/C64）、有效 fillMode+opA 推导的
FORWARD/BACKWARD、diagType 三个维度在**编译期实例化 8 个求解模板**（见 3.2.2），
运行期按 tiling 字段分派，作用等价于 tiling key：dtype × 求解方向 × 对角语义
的 2×2×2 全组合各有独立特化实例，避免 kernel 内逐行分支判断；complex64 的 H
共轭以独立 `conjNeeded` 标志控制，不额外增加模板维度。

#### 3.2.2 kernel 侧设计：

`KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY)`，入口 `spsv_analysis_kernel` /
`spsv_solve_kernel` / `spsv_update_*_kernel` / 辅助 kernel，host 侧 launcher
（`spsv_*_kernel_do`）以 GM_ADDR 指针 + `SpsvTilingData` 传值 + stream 下发。

**analysis 阶段**（`spsv_analysis_kernel`，V1 单核执行；多核三阶段
serial/parallel/final kernel 保留编译）：

1. 格式归一化：CSC 零拷贝重映射（colPtr→rowPtr、rowInd→colInd，同时隐式翻转
   fillMode LOWER↔UPPER 与 opA N↔T/H，转为等效 CSR 路径）；COO / SLICED_ELL /
   1-based CSR/CSC 在 Device 侧构建 0-based CSR 副本（rowPtr/colInd/values/
   perm），重复与未排序坐标按确定性规则规范化；
2. 转置物化：T/H 操作时构建转置 CSR 副本；complex64 的 CSC+H 组合改走
   CSC 派生矩阵 M（CSC 存储 ≡ A^T 的 CSR），host 将 opA 映射为 N，kernel 依
   `conjNeeded` 对读取值共轭，省一份转置副本；
3. Level Scheduling：从 CSR 结构构建行依赖 DAG 并拓扑分层（level 直方图 →
   levelRow 散射），同 level 内行两两独立；
4. diagPtr / validCount：逐行定位对角元指针、计算有效依赖计数；
5. 运行时回写：numLevels 写入 workspace 头部 TilingData 区；wsErrorFlag 在
   kernel 开头清零，检出非法输入（base 调整后索引越界 [0, m)、rowPtr 非法、
   SLICED_ELL 有效元素数超声明 nnz）时写入对应 `SpsvWsError` 码——host 在
   analysis 返回路径 `aclrtSynchronizeStream` 一次并读回该标记，非法输入返回
   `ACL_SPARSE_STATUS_INVALID_VALUE`（这是 analysis 阶段唯一一次 host 同步，
   solve/updateMatrix 仍全异步；偏差说明见"算子约束限制"）。

**solve 阶段**（`spsv_solve_kernel`，V1 单核）：

1. 8 模板分派：`template <bool C64, bool FORWARD, bool NON_UNIT>` 的
   `SpsvSolveLevels/SolveOneRow`，2×2×2 = 8 个编译期实例；FORWARD 由有效
   fillMode/opA 查表（算子分析节）决定；
2. 单核遍历全部 level（levelIdx < 0 路径）：level 间按依赖顺序串行，level 内
   逐行标量求解，读 A/X/Y 与调度区均为 GM Scalar 访问，同核读写 D-cache 自洽；
3. 逐行计算：UNIT 跳除法；NON_UNIT 除以 diagPtr 定位的对角元（缺失/零值传播
   INF/NAN）；complex64 走 AIV 复乘复除辅助函数，H 按 conjNeeded 在读取侧
   共轭；validCount 跳过已处理依赖前缀；
4. alpha：pointer mode HOST 时用 tiling 携带的 alpha/alphaIm；DEVICE 时由
   `alphaDevicePtr` 从 Device 地址读取（kernel 内不感知 host 值）；
5. 零 nnz 快路径：UNIT → `spsv_scale_copy_kernel`（Y = alpha·X），NON_UNIT →
   `spsv_fill_zero_kernel`（Y = 0），`spsv_scale_inf_kernel` 处理奇异对角的
   INF 传播辅助路径；
6. in-place：X/Y 同 Device values 指针时逐行读改写，行内先取 x[i] 再累加，
   语义与独立输出一致。

**updateMatrix 阶段**：GENERAL 用 `spsv_update_values_kernel` 按重排 perm 将
新 values 拷入 workspace CSR/转置副本（CSR + N 路径仅更新 descr 内 values
弱引用、零 kernel 启动）；DIAGONAL 用 `spsv_update_diag_kernel` /
`spsv_update_diag_csr_kernel` 替换对角 values 并刷新对角相关状态；更新仅涉
值不涉结构，Level Schedule 不变，故无需重新 analysis；nnz = 0 时直接返回
SUCCESS。

**为何单核——多核同步原语不可用的证据链**（ascend910_9391，CANN 8.5.1 与
9.1.0 双工具链交叉验证，结论一致）：

1. 无参硬 `SyncAll()`：在 solve kernel 中以最小函数体（`SyncAll(); return;`）
   隔离测试，所有被 launch 的核在同一 PC 崩溃（错误码 507035，D-cache/UB 总线
   错误，pc+0x58 位于 FFST/MTE 指令序列内）；改变 extern "C" 链接性、按值
   200B tiling 参数、TPipe/DataCopy MTE 预热均不改变结论；2 核子集 launch 则
   表现为挂起（FFTS+ 全核栅栏语义，未 launch 的核永远不到达）；
2. 三参软 `SyncAll(gm, ub, usedCores)`：空函数体 + 单次软栅栏同样全核同 PC
   崩溃；
3. 每层多 launch 备份轨（solve 按层独立 kernel、层间靠 stream 排序，
   `levelIdx` 路径，保留在代码中）：功能可跑通，但跨 kernel 内存一致性呈
   非确定性——相同二进制与输入，一次全对、一次仅 level 0 被求解；写端
   `DcciFlushRange` 后读端（换物理核）仍可能读到过期数据，属本机自定义
   kernel launch 路径的内存模型问题，非算子代码可控；
4. 结论：V1.1 定轨单核 solve（确定性正确），多核并行列为 V2，需 CANN 工单
   或迁移框架 launch 路径后重测。

**性能定位**：深层级 SpTRSV 属延迟瓶颈型负载（P-03 的 m=262,144 对 3.9M nnz，
依赖链长、level 内并行度有限），标量 AIV 单核的依赖链访存模式恰好占优；
实测 GPU cuSPARSE 标杆 Event median / NPU 同调用范围总耗时 median =
12.1x ~ 18.2x（标杆为同步型 cuSPARSE SpSV 调用，含其内部 kernel/同步开销）。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas 800I/T A2（910B3/910B4，arch22/DAV_2201；A3 型号同源适配，性能自验证在 910B3 完成） | √ |

## 算子约束限制

- `updateMatrix` 的 `newValues` dtype 与长度为调用方契约：接口为 `void*` 无类型/
  长度元数据，运行时不可校验；dtype 须与 analysis 时 computeType 一致，长度
  GENERAL = nnz 个元素、DIAGONAL = m 个元素，顺序按用户原格式（COO 按 COO 序、
  SELL 按槽位序、CSC 按列序、CSR 按行序），违规行为未定义；
- X/Y/matA 的 values 必须为 Device 内存：接口无内存空间属性，不做运行时校验
  （与 cuSPARSE 惯例一致），误传 host 指针表现为 kernel 异步执行失败；
- analysis 返回前有一次流同步（与母本"全异步"语义的偏差）：为把 Device 侧
  索引校验结果（wsErrorFlag）以确定错误码返回，host 在 analysis 返回路径同步
  一次并读回标记；solve/updateMatrix 仍全异步；
- 索引仅支持 I32（rowPtr 与 colInd），I64 组合返回 NOT_SUPPORTED；矩阵维度
  m ≤ INT32_MAX；
- 不支持广播（方阵三角求解，一维向量语义）；
- NON_UNIT 缺失/零对角产生 INF/NAN 并向后传播（任务接口定义行为，已在接口
  文档与测试报告记录）；
- 多核 solve 与段化向量化（V2）暂未启用，见"可维可测分析-后续演进"。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | 单标杆 CPU Golden 对比：FP32 用 float64 golden、complex64 用 complex128 golden；逐元素 `｜actual−golden｜ ≤ atol + rtol×｜golden｜`（rtol=2^-10、atol=2^-16），整体匹配率 ≥ 0.99，每元素绝对误差 ≤ max(A, 32×ULP(golden))（A=1e-2）；complex64 实部/虚部分别适用全部规则；重复执行 bit-wise deterministic；检查解向量逐元素、INF/NAN、UNIT 对角、N/T/H 语义。实测 ATK 官方精度 200 例（FP32+complex64 × CSR × N/T/H × base0/1 × host/device pointer mode）**200/200 ALL PASS**，bitwise 0 失败 | 任务书 3.2、生态算子开源精度标准 |
| 性能标准 | 倍率 = 标杆接口 GPU 设备 Event 调用耗时 median / NPU 同调用范围总耗时 median，预热 10、采样 30，报告 median/P90/workspace/Analysis/Solve 时间。GPU 标杆 median_us：P-01 514,144~514,203（m=65,536/nnz=524,288，FP32/N/LOWER/NON_UNIT）、P-02 1,228,206~1,230,583（m=131,072/nnz=1,572,864，c64/T/UPPER/UNIT）、P-03 3,078,027~3,078,256（m=262,144/nnz=3,932,160，c64/H/LOWER/NON_UNIT），均 pm=device、base0/1。实测 P-01/02/03 × base0/1 六场景全部 PASS，倍率 **12.1x ~ 18.2x**（验收线 ≥ 0.25x；NPU 绝对耗时按倍率推导约 P-01 28~43ms、P-02 68~102ms、P-03 169~254ms） | 任务书 3.3、gpu_performance_result_benchmark |
| 内存标准 | workspace 由 BufferSize 查询、Analysis 绑定；固有 workspace 绝对值不超过目标硬件 L2 Cache 容量。实测 workspace 最大 **132.1 MB**（P-03-base1）≤ 910B3 L2 192MB，6/6 PASS；torch 侧峰值（含输入输出）合计最大 507 MB | 任务书 3.4 |
| UT/ST | C++ UT（`test/spsv/arch22/`）构成：48 个 CSV 参数化用例（L0/L1/L2 分层：四格式 × fill/diag × base × op × 规模，含多核路径、确定性、T/H × update 与乱序 COO 回归用例）+ 32 个固定用例（complex64 golden 专项 3 项、异常/生命周期 26 项：NULL 参数、solve-before-analysis、I64 拒绝、非法 opA/alg/base、短向量、buffer 过早释放、pointer mode device、描述符变更等、Host 修复回归 3 项）；扩充后合计 80 项，910B3 实测 **80/80 全绿**（2026-09-23）。lifecycle 回归：updateMatrix GENERAL/DIAGONAL × base1 × N/T/H sweep 24 组 0 bad | 任务书 3.5 |
| Profiler 证据 | 自测报告提供 aclsparse Kernel 耗时，Profiler 展示 Host/Kernel/workspace/调用流，证明主求解在 NPU 执行 | 任务书 3.5 |

可维测配套：

- 测试入口：仓内冒烟 `spsv_test`（GTest，依赖 `test/frame/` 与 Eigen3 golden）；
  官方精度/性能位于任务包 `test_cases/aclsparseSpSV_testCase/`
  （`accuracy_cases.json` 200 例经 `run_accuracy_atk.sh`、`performance_cases.json`
  206 例经 `benchmark_sparse_ops_npu.py`，内存经
  `collect_sparse_ops_npu_memory.py`）；
- 可维护性：host 校验失败路径统一 OP_LOGE（含参数值）并返回语义化错误码；
  kernel 常量（`kRowsPerCore`、对齐宽度、错误码枚举）集中于头文件，调参不散落；
  公共 Host 逻辑与 arch22 差异层解耦（`sparse/common/aclsparse_spsv_descr.h`
  共享描述符，arch22 增量字段注释标注），可与 arch35 路径共同维护。

## 兼容性分析

公开接口与 `include/cann_ops_sparse.h` 现有原型逐字一致，无签名变更，ABI 稳定；
arch22 实现为 arch35 母本的增量架构目录（`sparse/spsv/arch22/`），随顶层 CMake
按 SOC 版本（ascend910b* / ascend910_93*）参与构建，不影响既有 950PR 路径；
描述符内部结构为共享文件上的 arch22 增量字段（`cachedComputeType`），默认值
保证向后兼容。后续演进（不改变对外契约）：

1. 多核 solve（V2）：恢复 `kRowsPerCore` 起核 + level 宽度自适应分核与
   `syncWsOffset` 软同步标志区，依赖 CANN 侧 SyncAll/跨 kernel 一致性工单或
   迁移框架 launch 路径后重测；
2. 段化向量化（V2）：长依赖段 `DataCopy + Mul + 归约` 替代逐元素标量 gather，
   `SEG_VEC_MIN` 阈值实测后启用；
3. 多核 analysis 并行度（levelRow scatter 并行化与负载均衡）、L2 提示 API 与
   in-place CacheMode 等 L2/性能调优项，上板 msprof 实测评估。

# 参考资料

1. [CANN 社区任务 2026 算子设计文档模板](https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md)
2. [CANN 社区任务 2026 提交流程](https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/README.md)
3. [ops-sparse 代码仓](https://gitcode.com/cann/ops-sparse)（本算子实现：`sparse/spsv/arch22/`，测试：`test/spsv/arch22/`）
4. [生态算子开源精度标准](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md)
5. cuSPARSE SpSV 官方文档（语义与生命周期对照标杆）
6. Swartz et al., *A multithreaded algorithm for sparse triangular solves*（Level Scheduling 方法）
