# aclsparseSpSM A2/A3 (arch22) 算子设计文档

| 项目 | 内容 |
|---|---|
| 任务名称 | 9月社区任务-aclsparseSpSM算子开发(A2/A3) |
| 参与者账号 | GitCode：`skytosky` |
| 交付代码仓 | [cann/ops-sparse](https://gitcode.com/cann/ops-sparse)，实现路径 `sparse/spsm/arch22/` |
| 设计提交路径 | `04_tasks/01_community-task-2026/tasklist/09-19-aclsparseSpSM-A2A3/skytosky/docs/design.md` |
| 支持硬件 | Atlas A2 训练系列（910B3/910B4）、Atlas A3（arch22/DAV_2201） |

# 需求背景（required）

## 需求来源

9月社区任务-aclsparseSpSM算子开发(A2/A3) 任务书（`aclsparseSpSM_A2A3_task_doc.md`）。

## 背景介绍

### aclsparseSpSM 算子语义

aclsparseSpSM 实现多右端稀疏三角求解：

$$op(A) \cdot C = \alpha \cdot op(B)$$

- A：稀疏三角方阵，CSR/CSC/COO，LOWER/UPPER，UNIT/NON_UNIT，index base 0/1，I32 索引，[m, m] 动态 m/nnz
- B/C：稠密多右端矩阵，ROW/COL 布局，C 可与 B 同 Device 指针（原地求解）
- opA/opB：N/T/H（complex64 的 H 为共轭转置）
- alpha：Host/Device pointer mode
- dtype：FP32（ACL_FLOAT）、complex64（ACL_COMPLEX64）
- 接口：Create → BufferSize → Analysis → SpSM(Solve) → UpdateMatrix → Destroy 多阶段模型，新增公开接口 `aclsparseSpSMUpdateMatrix`（GENERAL/DIAGONAL）与枚举 `aclsparseSpSMUpdate_t`

### ops-sparse master 现状分析

master 现有 `sparse/spsm/arch35/`（950/A5 原型）：host 侧 Analysis 完成 level scheduling，仅支持 CSR、FP32、opA=N/T、opB=N、HOST pointer mode、三阶段。本任务要求的 arch22 平台、CSC/COO、complex64、opA/opB 全 N/T/H、Device pointer mode、UpdateMatrix 公开接口、确定性规范化均为增量。仓库当前无 arch22 三角求解的合入实现。

## 需求分析（required）

## 需求描述

使用 Ascend C 在 arch22（DAV_2201/910B）实现 aclsparseSpSM 全生命周期：三格式三角求解多 RHS，覆盖任务书全部 attr 组合，主求解与格式相关计算由 NPU Kernel 执行，禁止 CPU fallback，bitwise deterministic。

## 需求拆解

1. CSR/CSC/COO 三格式输入，设备端确定性规范化为统一内部 CSR 表示
2. LOWER/UPPER、UNIT/NON_UNIT、`ACL_SPARSE_SPSM_ALG_DEFAULT`
3. opA/opB 的 N/T/H，complex64 H 路径共轭语义
4. Host/Device pointer mode 的 alpha
5. Create/BufferSize/Analysis/Solve/UpdateMatrix/Destroy 六接口，Analysis 缓存跨阶段状态与一致性校验
6. UpdateMatrix：GENERAL 全量刷新、DIAGONAL 仅刷新对角相关状态
7. 未排序/重复坐标确定性合并（bitwise deterministic）
8. B/C 同指针原地求解；调用沿 handle stream 异步，禁止 aclrtSynchronizeStream
9. 性能倍率 ≥ 0.25×（P-01/P-02/P03 × FP32/complex64 × base0/1 共 12 条有效组合）；workspace ≤ 目标硬件 L2 Cache
10. 公共 Host 层与 arch22 差异层解耦，便于 A5/950 同主干维护

# 详细设计（required）

## 算子分析

### 数学公式

对行 i（以 LOWER、opA=N 为例）：

$$c_{i,:} = \frac{\alpha \cdot b_{i,:} - \sum_{k<i} a_{i,k} \cdot c_{k,:}}{a_{i,i}}$$

UNIT 时 $a_{i,i}=1$；UPPER 时求和域为 $k>i$，求解序反向；opA=T/H 时方程以 $A^T$（/$A^H$）为系数矩阵；opB=T/H 时右端取 $B^T$（/$B^H$）对应元素。complex64 乘法 $(a+bi)(c+di)=(ac-bd)+(ad+bc)i$，H 路径对系数取 $y=-\mathrm{Im}$ 后同 T。

### 支持数据类型

- FP32：`ACL_FLOAT`，values float
- complex64：`ACL_COMPLEX64`，布局 `{float x, float y}`，二进制兼容 float2；实部/虚部独立满足精度规则

### 支持形状

- matA: [m, m]，动态 m/nnz；matB: [m, n]（op 后），动态 RHS n 与 ld；matC: [m, n]，可与 matB 同指针

## 算子实现

### 3.2.1 Host 侧设计

#### 1. 多阶段状态管理

**Create**：堆分配 `aclsparseSpSMDescr` 内部状态，初始化默认值（stage=CREATED）。

**BufferSize**：
- 参数校验（handle/descr/alpha 非空、pointer mode、format、dtype 一致、fill/diag、alg、shape、ld）
- 计算 workspace 偏移（按 opA=T/H + NON_UNIT + CSC/COO + COL 的最大组合预留，实际按本次 attr 裁剪）
- 缓存 `bufferSize` 供 Analysis 一致性校验；BufferSize/Analysis 允许 B/C values 为空但描述符有效

**Analysis**：
- 公共前置校验 + buffer 大小一致性校验
- **设备端格式规范化**（kernel，见 3.2.2-3）：CSC→CSR、COO→CSR（计数排序两遍）、未排序行内稳定排序、重复坐标按序合并——规范化结果写 workspace，全程 NPU 执行，满足任务书 2.1.8；重复坐标合并在设备端按"行主序 + 列次序"稳定规则执行，保证 bitwise deterministic
- **对角提取 kernel**：从规范化 CSR 提取 diagVal[m]（NON_UNIT，含零对角检测）
- **level scheduling**：D2H 规范化 CSR 的 rowOff/colInd → host 逐行拓扑分层（LOWER 前向/UPPER 后向）→ level buckets（计数/前缀和/散列三遍）→ H2D 写回 levelRowPtr[L+1]/levelRowIdx[m]；Analysis 是一次性预处理开销，Solve 主求解与格式计算均在 NPU，符合"Analysis 状态保存必要元数据，不使用 CPU 端同规模求解缓存"
- opA=T/H：host 侧生成转置 CSR（transRowOff/transColInd），complex64 H 在 kernel 读取时共轭，不物理共轭存储
- 计算 kChunkSize（UB 容量推算），构造 by-value TilingData 缓存，绑定 buffer，置 analyzed

**Solve**：
- 前置校验 + 跨阶段一致性校验（analyzed、opA/opB、matA/B/C 字段、externalBuffer 绑定一致）
- 异步 launch 序列（同 stream）：① orderB=COL 时 B(COL)→denseBuf(ROW)（transpose-in kernel）② 主求解 kernel ③ orderC=COL 时 denseBuf(ROW)→C(COL)（transpose-out kernel）；B/C 同指针时 matC 直接覆盖 matB 值域，满足原地语义
- 禁止 aclrtSynchronizeStream；buffer 在异步 Solve 完成前保持有效

**UpdateMatrix**：
- GENERAL：设备端用 newValues 重建规范化 values 与 diagVal → 重新 D2H + level scheduling（结构未变时层级不变，仅 diagVal 刷新）→ 重建 TilingData
- DIAGONAL：仅设备端替换对角值并刷新 diagVal，免重分层
- 校验 analyzed、newValues 非空、dtype/pattern 一致

**Destroy**：释放描述符；描述符生命周期内 stage 状态机（CREATED→SIZED→ANALYZED）防止阶段错序调用。

#### 2. 参数校验策略

- 空指针：handle/descr/alpha/bufferSize/buffer/newValues 按任务书返回明确错误码
- opA/opB/updatePart/alg 非法枚举 → NOT_SUPPORTED
- dtype 一致性：matA/matB/matC/computeType/alpha 五方一致（FP32 或 complex64）
- shape：A 方阵；op 后 A 右维 == B 行维 == C 行维；B.cols == C.cols；ld：ROW→ld≥cols、COL→ld≥rows
- NON_UNIT 对角存在且非零（Analysis 检测，奇异返回明确错误）
- index base 0/1；跨阶段描述符/参数/externalBuffer 一致性校验（任务书 2.1.5）

#### 3. Workspace 布局设计

```
[ 0              , normRowOffOff  ) : 64B header（magic/stage/attr 快照）
[ normRowOffOff  , normColIndOff ) : int32 normRowOff[m+1]        (CSC/COO/未排序时)
[ normColIndOff  , normValOff    ) : {int32,float[2]} 规范化 CSR  (同上)
[ levelRowPtrOff , levelRowIdxOff) : int32 levelRowPtr[L+1]
[ levelRowIdxOff , diagValOff    ) : int32 levelRowIdx[m]
[ diagValOff     , transRowOffOff) : dtype diagVal[m]            (NON_UNIT)
[ transRowOffOff , transColIndOff) : int32 transRowOff[m+1]      (opA=T/H)
[ transColIndOff , transValOff   ) : int32 transColInd[nnz]      (opA=T/H)
[ transValOff    , denseBufOff   ) : dtype transValues[nnz]      (opA=T/H)
[ denseBufOff    , end           ) : dtype denseBuf[m*nChunk]    (COL 布局时)
```

- 复用策略：CSR 已排序输入时规范化区零占用；denseBuf 按列分块（m×kChunkSize）而非全量 m×n，**workspace 总量 O(m + nnz + m·kChunk) 远小于稠密 m×n，满足 ≤ L2 Cache 条款**（910B3 L2 为目标口径）
- 64B 对齐；complex64 values 区按 2×float 计

#### 4. Level Scheduling 设计

- ComputeRowLevels：按 solveDir 逐行计算 levelBuf[i] = max(levelBuf[依赖列])+1，同时记录 diagVal 与奇异检测
- BuildLevelBuckets：计数 → 前缀和 → 散列三遍生成 levelRowPtr/levelRowIdx
- 层内行按行宽降序排布（同层级内负载均衡）
- UpdateMatrix-DIAGONAL 免重分层；GENERAL 结构不变时层级不变，仅刷 diagVal

#### 5. 确定性（bitwise deterministic）设计

- 规范化：设备端两遍计数排序为稳定排序；重复坐标按出现序合并，合并算子为加法（与序无关的确定结果由"先排序后顺序归约"保证，不做原子浮点加）
- 求解：行内求和按 colInd 升序固定顺序串行累加（FP32 高精度累加路径），同输入重复调用与重复 Solve 结果 bitwise 一致
- 不依赖任何原子加浮点操作

#### 6. UpdateMatrix 设计

见多阶段状态管理；关键点：GENERAL 走"设备端刷新 + 复用层级"，DIAGONAL 走"仅 diagVal"，两者均不重新分配 workspace，buffer 绑定关系保持。

#### 7. Tiling 策略

- TilingData by-value 传递：m, n, ldb, ldc, orderB, orderC, opA, effectiveFillMode, diagType, L, kChunkSize, maxRowLen, indexBase, workspace 偏移组
- alpha 在 Solve 前按 pointer mode 刷新（Device 模式由前置 kernel 或 aclrtMemcpy 同步读取，仍在异步语义内）
- tiling key：dtype × 规范化需求（是否需归一化）× opA(T/H 与否) × COL 布局（orderB/orderC）

#### 8. 分核策略

- `GetAivCoreCount()` 动态获取，失败兜底 1（Solve kernel 依赖 SyncAll）
- level 内多行并行：每 block 处理一行（行内向量并行 + 多 RHS 列分块）；行数 < 核数的浅层，block 合并连续多行提升占有率

#### 9. 数据分块和内存优化策略

- UB 动态容量（`GetUbSize()`），预留 8KB；colInd/vals 双缓冲 4×maxRowLen×4B；accBuf+tmpBuf 按 kChunkAlign
- kChunkSize = min(ubAvail / (4×sizeof(dtype)), n)，8 对齐
- RHS 列向量化：FP32 按 8×float（32B）对齐访存；complex64 按 float2 对访存，Mul 用实虚分离乘加

### 3.2.2 Kernel 侧设计

#### 1. 规范化 Kernel（Analysis 阶段）

- COO→CSR：pass1 计数行直方图（含 col 次键稳定化信息），pass2 前缀和写出
- 未排序 CSR：行内 maxRowLen 有界插入/归并排序（maxRowLen 小，UB 内完成）
- 重复坐标：有序序列顺序扫描合并（加法归约）
- CSC→CSR：两次扫描转置（列计数 → 写出）
- 以上全程 NPU 执行，满足任务书 2.1.8"格式相关计算由 NPU Kernel 执行"

#### 2. Solve Kernel

- Init：解析 by-value TilingData，InitBuffer，预载 levelRowPtr
- Process：按 level 逐层 → 层内行并行 → 行内：CopyIn(colInd+values) → Compute（依赖求和 = Σ a[i,k]·c[k,j]，k 按 colInd 升序；c[i,j] = (α·b[i,j] − sum)/dv）→ CopyOut
- opA=T/H：系数从 transCSR 读；H 对 values 虚部取反（共轭）
- opB=T/H：b[i,j] 按索引重排读取（不物理转置右端）
- complex64：专用 Mul/Div 路径；UNIT 时除法省略
- 多 RHS 按 kChunkSize 分块循环

#### 3. Transpose Kernel（COL 布局）

- 稠密分块转置（SIMT 风格 tile），denseBuf 仅 m×kChunkSize 块级复用

## 性能达标路径（P-01/P-02/P-03）

| 场景 | 规模(m/nnz/RHS) | dtype | GPU 标杆 median | NPU 预算(≥0.25×) |
|---|---|---|---|---|
| P-01 | 32768/262144/16 | FP32 | ≈0.323 s | ≈1.29 s |
| P-02 | 65536/786432/32 | complex64 | ≈0.723 s | ≈2.90 s |
| P-03 | 131072/1966080/64 | complex64 | ≈1.86 s | ≈7.50 s |

- 主要开销 = 逐层 Solve 的串行层数 L × 层内最宽行耗时；层并行由 level buckets 保证满核
- 优化手段按优先级：浅层多行合并提占有率 → RHS 列向量化访存 → diagVal 预读消除分支 → 规范化/转置 kernel 的冗余消除
- 报告口径：预热 10/采样 30，记录 median/P90/Analysis/Solve/workspace 与 msprof Profiler 证据（Host/Kernel/NPU 执行流）

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas 800I A2 (910B3/910B4) | √ |
| Atlas A3 (DAV_2201) | √ |

## 算子约束限制

- A 为二维三角方阵，op(A)/op(B) 后维度满足公式，否则返回确定错误
- A 的 fill/diag/格式/index base 各阶段保持一致；NON_UNIT 对角存在且非零
- A/B/C/computeType 为 ACL_FLOAT 或 ACL_COMPLEX64 一致组合；Device 索引 I32
- 未排序与重复坐标经确定性预处理（bitwise deterministic）
- 主求解与格式相关计算由 NPU Kernel 执行，禁止 CPU fallback
- buffer 在异步 Solve 完成前有效且不被修改；Analysis 不使用 CPU 端同规模求解缓存

## 可维可测分析

### 精度/性能/内存标准

| 验收标准 | 描述 |
| --- | --- |
| 精度 | CPU Golden float64/complex128；rtol=2^-10、atol=2^-16、A=1e-2；匹配率≥0.99 且每元素 ≤ max(A, 32×ULP)；complex64 实虚部独立适用 |
| 性能 | P-01/02/03 × FP32/complex64 × base0/1 性能倍率 ≥ 0.25× |
| 内存 | workspace ≤ 目标硬件 L2 Cache（或 in/out>500MB 时额外内存 ≤50% GPU） |

### 测试计划

- C++ UT/ST：`test/spsm/arch22/`，覆盖任务书 3.5 必测表（基础功能/操作布局/多阶段/边界异常/原地与指针），含 INF/NAN、重复调用 bitwise 一致、UpdateMatrix 后结果正确、buffer 生命周期
- 官方脚手架：`accuracy_cases.json` 200 条 + `generate_atk_cases.py` 1000 条泛化 + `performance_cases.json` 6 条 P 场景，`run_accuracy_atk.sh run` 执行
- 性能脚本：`benchmark_sparse_ops_npu.py`（torch_plugin 注册 spsm_npu hook），内存 `collect_sparse_ops_npu_memory.py` + `compare_sparse_ops_memory.py`
- README：环境/编译/复现步骤

### 兼容性分析

- 新增 `aclsparseSpSMUpdateMatrix`/`aclsparseSpSMUpdate_t` 于 `include/cann_ops_sparse.h`，向后兼容
- 公共 Host 层与 arch22 差异层解耦，与 arch35 同主干回归
- 里程碑（截止 2026-10-04）：D+3 骨架编译通 → D+6 Analysis 链路 → D+11 全功能精度过 → D+14 性能/内存达标 → D+15 文档/报告/提交
