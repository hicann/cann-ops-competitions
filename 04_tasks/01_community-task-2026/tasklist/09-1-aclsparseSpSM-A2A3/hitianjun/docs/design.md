# aclsparseSpSM A2/A3 (arch22) 算子设计文档

| 项目 | 内容 |
|------|------|
| 任务名称 | 9月社区任务-aclsparseSpSM算子开发(A2/A3) |
| 参与者账号 | GitCode：`hitianjun` |
| 目标硬件 | Atlas A2/A3（arch22 / DAV-2201）；自测环境 Atlas A3 训练系列（910C，AIV=40 实测） |
| CANN 版本 | CANN 9.2.0（aarch64），bisheng（--asc-aicore-lang） |
| 交付代码仓 | 目标合入 [cann/ops-sparse](https://gitcode.com/cann/ops-sparse)；实现路径 `sparse/spsm/arch22/` |
| 设计提交路径 | `04_tasks/01_community-task-2026/tasklist/09-1-aclsparseSpSM-A2A3/hitianjun/docs/design.md` |
| 对标接口 | cuSPARSE Generic `cusparseSpSM`（§6.6.12，含 `cusparseSpSM_updateMatrix`） |
| 文档版本 | v1.0 |

---

## 1. 需求背景（required）

### 1.1 需求来源

社区任务「9月社区任务-aclsparseSpSM算子开发(A2/A3)」任务书：在 Atlas A2/A3（arch22 /
DAV-2201）实现多右端稀疏三角求解 `op(A)·C = α·op(B)`，交付 `ops-sparse` 公开 C++ Legacy
API（`CreateDescr → BufferSize → Analysis → SpSM(Solve) → UpdateMatrix → DestroyDescr`
多阶段）、Host 状态管理、arch22 Ascend C Kernel、C++ UT/ST、性能脚本与文档。主计算在
NPU 调用流运行，禁止 CPU fallback。

### 1.2 背景介绍

#### 1.2.1 aclsparseSpSM 实现现状分析

`ops-sparse` master 已有 `sparse/spsm/arch35/`（Ascend 950PR / DAV-3510）实现，能力与
本任务差距：

| 参数 | arch35 现状 | 本任务 arch22 目标 |
| --- | --- | --- |
| matA 格式 | CSR | **CSR / CSC / COO**（Analysis 阶段归一化） |
| 数据类型 | FP32 | **FP32 / complex64** |
| opA | N / T | **N / T / H**（H 共轭转置，仅 complex64 有意义） |
| opB | N（签名保留） | **N / T / H** |
| alpha pointer mode | HOST（Solve 直接解引用） | **HOST / DEVICE** |
| UpdateMatrix | 无 | **新增 GENERAL / DIAGONAL** |
| B/C 布局 | ROW / COL（转置 kernel） | 同左 |
| index base | 0 / 1 | 同左 |
| in-place（B/C 同 values 指针） | 未显式验证 | **显式支持并 UT 覆盖** |

算法主干（host level scheduling + NPU 逐层求解 + 转置 kernel）已被 arch35 验证正确，
本设计复用主干、新增 arch22 差异层与上述能力补齐。

#### 1.2.2 aclsparseSpSM 功能分析

求解稀疏三角线性方程组（多右端）：

- A 为 m×m 稀疏三角方阵（LOWER/UPPER × UNIT/NON_UNIT）；
- B 为 m×n 稠密右端矩阵（n=RHS 列数），C 为 m×n 稠密解矩阵；
- LOWER+NON_UNIT 前向代入逐行语义：`C[i,:] = (α·B[i,:] − Σ_{j<i} A[i,j]·C[j,:]) / A[i,i]`；
- UNIT 对角省略除法；UPPER 为后向代入；opA=T/H 等价于在转置（+共轭）后的三角结构上
  求解（fill mode 交换）。

典型场景：ILU/IC 预条件回代、稀疏三角系统多 RHS 求解。

## 2. 需求分析（required）

### 2.1 需求描述

在 arch22（DAV-2201）交付：FP32/complex64、CSR/CSC/COO、opA/opB N/T/H、LOWER/UPPER、
UNIT/NON_UNIT、ROW/COL 布局、Host/Device pointer mode、base 0/1、in-place、UpdateMatrix
（GENERAL/DIAGONAL）全矩阵；多阶段调用语义对齐 cuSPARSE；solve 阶段 bit-wise 确定；
主计算 NPU 异步执行。

### 2.2 需求拆解

1. 公开 API 六件套 + `aclsparseSpSMUpdate_t` 枚举，头文件同步 `include/cann_ops_sparse.h`；
2. Host 多阶段状态机：Analysis 缓存格式/三角属性/opA/opB/dtype/尺寸/索引/布局/RHS/
   workspace 绑定；Solve 前跨阶段一致性校验；
3. BufferSize/Analysis 允许 B/C values 为 NULL（仅描述符有效）；Solve 须有效 values；
4. 格式归一：CSC/COO 在 Analysis 归一化为 CSR（确定性排序、重复坐标合并）；opA=T/H
   归一化为转置 CSR（fill swap，H 对 values 共轭）；统一 kernel 只见 CSR；
5. complex64 全链路：kernel 侧实/虚平面分离计算（Muls/Sub/Add 向量化）与交织装载/写回；
6. opB T/H：Solve 前后 transpose kernel 处理布局/共轭（方向=0/1 + conj 标志）；
7. Device pointer mode：alpha 经 D2H 读回（4/8B）参与 tiling by-value 传递；
8. UpdateMatrix：GENERAL 刷全部 values（T/H 时重转置 + 共轭，刷新 diagVal），DIAGONAL
   仅刷对角；结构不变不重跑 level scheduling；
9. in-place：B/C 同 values 指针时逐行"读 B[i]→算→写 C[i]"天然安全，显式 UT 覆盖；
10. 性能：P-01/P-02/P-03（fp32+complex64 × base 0/1）solve 阶段 ≥ 0.25× GPU 标杆
    （本设计目标 ≥ 1.0×，见 §3.1.4 算账）；
11. 内存：workspace（level 表 + 归一 CSR + 转置 values + denseBuf）远小于 L2（P-03
    ~118MB workspace 属"无等价 GPU 接口 case"口径按 L2 判定见 §6.1 说明，实测以
    collect/compare 脚本为准）；
12. UT/ST：参数/异常矩阵 + 功能矩阵 + bitwise 重复执行 + ATK 200 精度泛化。

## 3. 详细设计（required）

## 3.1 算子分析

### 3.1.1 数学公式

$$op(A)\cdot C = \alpha\cdot op(B),\quad C = \alpha\, op(A)^{-1} op(B)$$

- $op(X)=X$（N）/ $X^T$（T）/ $X^H$（H=共轭转置，仅 complex64）；
- 归一化视角：opA=T/H ⇒ 在 $A^T$（H 加共轭）的 CSR 上求解，effectiveFill =
  swap(fill)；opB=T/H ⇒ B 逻辑转置（H 加共轭）后参与求解，解再转置回 C 目标布局。

### 3.1.2 支持数据类型

| A / B / C / computeType | alpha | 状态 |
| --- | --- | --- |
| FP32（`ACL_FLOAT`） | FP32 | 本任务 |
| complex64（`ACL_COMPLEX64`） | complex64 | 本任务 |

四者 dtype 须一致（同精度，对齐 cuSPARSE）。

### 3.1.3 支持形状

matA [m,m]（动态 m/nnz）；matB/matC [m,n]（动态 RHS、ROW/COL 布局、独立 ld）；
m/nnz/RHS 覆盖 0/1 与大规模（P-03 m=131072）；B/C values 可同指针原地。

### 3.1.4 性能算账（先算账再实施）

**第一步：精确复现 P-case 结构**（本地按任务包 `operator_adapter._csr` 生成器逐行复现，
seed 20260919，`row_pattern="uniform"` 默认）：矩阵为**等宽连续带状三角阵**——每行
off-diag 列区间为紧贴对角线的连续段 `[i-k_i, i-1]`（k_avg = 7/11/14）。由此推出关键事实：
k_i ≥ 1 ⇒ 行 i 依赖行 i-1 ⇒ **依赖图为全链，level[i]=i，L=m**（32768/65536/131072，
每层恰 1 行）。cuSPARSE 的 323ms/723ms/1.83s 正是 level 串行化的镜像。

**第二步：算法裁决**——朴素 level scheduling（arch35 主干）需 m 次 SyncAll，P-03 纯
barrier ≈ 2s（15us/次 中值），40 核中 39 核空转：0.25× 边缘、1.0× 无望。**定案：分块
循环波前**（行切 B≈1024 的块，块间依赖仅前块尾 k 行，`c mod C` 循环分配核，波次
SyncAll）：P-01/02/03 只需 **1/2/4 次 barrier**；块内滑窗（依赖行 X[i-k..i-1] 连续
7KB，UB 驻留 k+1 行滑入 1 行/迭代）+ raw `__ubuf__` 访存 + c64 平面计算。

**第三步：标定预测**（向量 op issue 450ns 中值，350-600ns 实测区间全扫描稳定）：

| 场景 | m / nnz / RHS / dtype | GPU median | 预测总时 | ratio |
| --- | --- | ---: | ---: | ---: |
| P-01 | 32768 / 262144 / 16 / fp32 | 323.0ms | **~7ms** | **~46×** |
| P-02 | 65536 / 786432 / 32 / c64 | 723.0ms | **~62ms** | **~12×** |
| P-03 | 131072 / 1966080 / 64 / c64 | 1834-1888ms | **~130ms** | **~14×** |

结论：三 case 全部 ≥10×（0.25× 门槛余量 40-180 倍），且对未知常数（SyncAll 单价、
issue 精确值）不敏感。完整推导与敏感性见 `sparse/spsm/arch22/docs/PERF_EVAL.md`。
非带状泛化 pattern 走通用 level-分块波前路径兜底（正确性同源，性能无要求）。

## 3.2 算子实现

### 3.2.1 实现方案

复用 arch35 算法主干，新增 arch22 差异层与能力补齐。总体数据流：

```
Analysis（host，一次性）:
  校验 → D2H 三元组 → CSC/COO→CSR 归一（确定性）→ [opA=T/H: CSR→CSC 转置
  （H 共轭）+ fill swap] → host level scheduling（ComputeRowLevels +
  BuildLevelBuckets，奇异检测）→ H2D levelRowPtr/levelRowIdx/diagVal/
  归一 CSR → 构造并缓存 TilingData → 绑定 buffer

Solve（NPU，同 stream 三步，全异步）:
  1) transpose-in:  orderB=COL 或 opB≠N → denseBuf(ROW)（含 H 共轭）
  2) solve:         逐 level 多核求解（层内分行、层间 PipeBarrier<PIPE_MTE3>+SyncAll）
  3) transpose-out: orderC=COL 或 opB≠N → C 目标布局（含 H 共轭）

UpdateMatrix（host + 定向 H2D）:
  GENERAL: 刷归一 values（T/H 重转置+共轭）+ diagVal；DIAGONAL: 仅刷 diagVal；
  结构不变 ⇒ level 表复用，不重 Analysis
```

#### host 侧设计

##### 1. 多阶段状态机

`aclsparseSpSMDescr` 内部状态：analyzed 标志、opA/opB/dtype/format/fill/diag/base、
m/n/nnz/ldb/ldc/orderB/orderC、L、cachedTiling（by-value）、cachedBufferSize、
buffer 绑定指针、归一 CSR 主机侧副本（UpdateMatrix 重转置用）。

- **Create/Destroy**：堆分配/释放，无设备资源；
- **BufferSize**：全参数校验（空句柄/非法枚举/dtype 不一致/shape 不匹配/ld 越界/
  base 非法），按最坏归一路径计算 workspace 偏移并缓存；B/C values 允许 NULL；
- **Analysis**：一致性校验（buffer 尺寸=缓存值）→ 归一化 + level scheduling +
  H2D → 缓存 tiling；奇异（NON_UNIT 零对角/缺对角/非法 colInd）返回明确错误；
- **Solve**：跨阶段一致性逐字段校验（opA/opB/描述符字段/buffer 绑定）→ 取缓存
  tiling，刷新 alpha（Device mode 先 D2H 读回）→ 三步 launch（禁止 SyncStream）；
- **UpdateMatrix**：见上。

##### 2. workspace 布局（64B 对齐，条件分配）

```
[0, levelRowPtrOff)   64B header
levelRowPtr[L+1] · levelRowIdx[m] · diagVal[m]（NON_UNIT）
归一 CSR（CSC/COO 或 opA=T/H 时）: rowOff[m+1] · colInd[nnz] · values[nnz]
denseBuf[m×n]（orderB/orderC=COL 或 opB≠N 时）
```

fp32 路径 P-03 最坏（c64+T）：~118MB 级，与任务书 GPU workspace（97.5MB）同量级，
内存验收按任务书两口径之一执行（§6.1）。

##### 3. level scheduling（host，O(nnz)）

逐行（前向/后向）计算 `level[i]=1+max(level[j]: j∈deps(i))`；NON_UNIT 同步提取
diagVal 并检测奇异；三趟建桶（计数→前缀和→散列）得 levelRowPtr/levelRowIdx。
P-03 规模 host 单线程 ~10-20ms，一次性 Analysis 开销，单独报告不计入 solve。

##### 4. 分核策略

`GetAivCoreCount()` 运行期查询（兜底 1），层内行均分跨核；空核参与 SyncAll 握手。

##### 5. tilingkey 规划

tiling by-value 传递（不落 GM）；字段含 m/n/ldb/ldc/orderB/orderC/needTranspose/
effectiveFill/diagType/L/kChunkSize/maxRowLen/indexBase/alpha(dtype 双精度表达：
fp32 直传，c64 传实/虚两个 float)/dtype 码/workspace 偏移组。

#### kernel 侧设计

##### 1. Solve kernel（`spsm_solve_kernel`，AIV 多核，列分解标量求解）

替换 arch35 逐 level 多 pass 为主从两路径统一框架：

- **调度**：行切 B 块（host 定 B：默认 1024，自适应保证块数 ≥ 2×核数），块 c 归核
  `c mod C`，波次 SyncAll（写提交用 `PipeBarrier<PIPE_MTE3>` 先行）——P-case 仅
  1/2/4 次 barrier；
- **带状快路径**（Analysis 检测每行 colInd 为连续段 `[i-k_i, i-1]` 后启用）：
  块内滑窗——UB 驻留最近 k+1 个 X 行（平面化），每迭代滑入 1 行（单次 MTE2），
  依赖累减全部命中 UB；B 行/A 值连续装载；
- **通用路径**（非带状）：块 = 同 level 行组（level 表分块打包），块内逐行
  LoadRowCsrData + 依赖行 MTE2 预取（arch35 ReduceDeps 双缓冲沿用）；
- **raw `__ubuf__` 标量访存**：colInd/vals/rowOff/块表逐元素读全部裸指针化
  （sparse2dense 实测 16-31ns vs GetValue 85-91ns；cast 包调用和式，B7 规则）；
- **c64 平面计算**：窗口滑入时交织→平面（2 op/行摊销），依赖累减每 dep
  4 Mul + 2 Add/Sub（固定升序，bit-wise 确定），NON_UNIT 复数除，写回交织；
  fp32 为 Muls+Sub 紧邻 FMA 融合序列；
- **kChunk = n**（P-case n≤64 单 chunk；大 n 泛化回退分块）。

##### 2. Transpose kernel（`spsm_transpose_kernel`）

COL↔ROW（+c64 共轭标志）按行分段 DataCopyPad 跨步搬运（blockCount=段长，
blockLen=dtypeSize，GM stride=ld）；arch22 上 per-call issue 成本 ~591ns（L2 通道），
P-03 最坏 262k 次调用 / 40 核 ≈ 4ms，预算内。c64 共轭在转置 UB 驻留段内做向量
虚部取反（Sub from 0）后写出。

##### 3. 格式归一 kernel 化边界

CSC→CSR / COO 排序归一 / CSR→CSC 转置在 Analysis host 阶段完成（一次性开销，
非 solve 热路径）；solve 主计算 100% NPU。

## 4. 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas A2 训练系列（910B3/910B4） | √ |
| Atlas A3 训练/推理系列 | √ |

arch22 与 arch35 共享 Host 逻辑分层（公共校验/描述符在头文件层，差异在
归一化与 launch），A5/950 同主干维护。

## 5. 算子约束限制

- A 为二维三角方阵；op 后维度匹配；不满足返回确定错误码；
- NON_UNIT 对角存在且非零（Analysis 奇异检测报错）；
- dtype 四者一致（ACL_FLOAT / ACL_COMPLEX64）；Device 索引 I32；
- fill/diag/format/base 各阶段一致（Solve 逐字段校验）；
- Analysis→Solve 期间 matA 结构不变（UpdateMatrix 仅数值）；buffer 生命周期至异步
  完成；
- COO 重复坐标：归一化合并语义在文档声明（与 cuSPARSE 要求唯一坐标对齐，重复项
  合并为求和前的一致性处理以确定性顺序执行）。

## 6. 可维可测分析

### 6.1 精度标准/性能标准

| 验收标准 | 描述 | 来源 |
| --- | --- | --- |
| 精度 | CPU golden float64/complex128；rtol=2^-10、atol=2^-16、A=1e-2；匹配率≥0.99；逐元素≤max(A,32×ULP)；c64 实/虚分别适用；ATK ≥200 case | 任务书 §3.2 |
| 性能 | P-01/02/03 × {fp32,c64} × base{0,1} solve 阶段 ≥0.25× GPU 标杆（设计目标 ≥1.0×，算账见 §3.1.4） | 任务书 §3.3 |
| 内存 | 50% 规则或 workspace≤L2 二口径之一（collect/compare 脚本实测报告） | 任务书 §3.4 |

### 6.2 测试设计（C++ UT/ST + ATK）

- 阶段机：Create/BufferSize/Analysis(空 values)/Solve/UpdateMatrix(GENERAL/
  DIAGONAL)/Destroy 全序列 + 乱序调用负例（未 Analysis 先 Solve、字段变更后 Solve、
  buffer 过早释放）；
- 功能矩阵：3 格式 × 2 dtype × opA{N,T,H} × opB{N,T,H} × fill × diag × 布局 ×
  base × pointer mode × in-place × ld padding；
- 边界：m=1、n=1、nnz=0、空行、近奇异（功能验收）、未排序/重复坐标；
- 确定性：同输入重复执行 bit-wise 一致；
- ATK 泛化 200 case（任务包 accuracy_cases.json 口径）。

### 6.3 兼容性分析

- 新增 `aclsparseSpSMUpdateMatrix` 与 `aclsparseSpSMUpdate_t`，向后兼容既有五阶段；
- 公共 Host 层与 arch22/arch35 差异层解耦（先例：sparse2dense arch22 同仓合入，
  PR #211）。

## 7. 实现路径与目录结构

```
ops-sparse/
├── include/cann_ops_sparse.h          # UpdateMatrix 枚举 + 接口声明
├── sparse/spsm/arch22/                # Host + Kernel + tiling（本任务）
│   ├── spsm.h / spsm_host.cpp / spsm_kernel.h / spsm_kernel.cpp
│   └── spsm_tiling_data.h
├── sparse/spsm/arch35/                # 既有 950 实现（回归不动）
└── test/spsm/arch22/                  # C++ UT/ST + perf 脚本 + README
```

编译：`bash build.sh --ops=spsm --soc=ascend910b --run`（自测环境 910C 同源
arch22；A2 910B3/B4 验收环境复跑）。

## 8. 交付件

1. 本设计文档（PR 至 cann-ops-competitions）；
2. 代码 PR（fork 分支 + 邀请 Ascend-CANN）；
3. 自测报告（腾讯文档模板/Excel：精度 ATK 200、性能 6/12 case、内存、截图、
   Profiler 证据）；
4. 测试工程 README（环境/编译/复现步骤）。
