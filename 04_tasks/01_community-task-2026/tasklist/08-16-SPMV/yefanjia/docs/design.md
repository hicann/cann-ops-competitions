# 需求背景（required）

## 需求来源

本需求来自 **8月社区任务-SPMV算子开发**（tasklist：`08-16-SPMV`）。参考 [cuSPARSE SpMV](https://docs.nvidia.com/cuda/cusparse/#cusparsespmv)，在昇腾 Ascend 950PR（`dav-3510` / arch35）上基于 Ascend C 实现 CSR 稀疏矩阵向量乘，并以 `aclsparse` 三阶段接口交付，验收通过后合入开源仓 [cann/ops-sparse](https://gitcode.com/cann/ops-sparse)。

目标公式：

```text
Y = alpha * op(A) * X + beta * Y
```

其中 `A` 为 CSR，`op(A)` 取 `A` 或 `A^T`，`X` / `Y` 为稠密向量，`Y` 支持原位累加。

## 背景介绍

### SpMV 算子功能说明

| 模式 | 语义 | X 长度 | Y 长度 |
| --- | --- | --- | --- |
| 非转置 | `y[r] = alpha * sum_c A[r,c] * x[c] + beta * y[r]` | K | M |
| 转置 | `y[c] = alpha * sum_r A[r,c] * x[r] + beta * y[c]` | M | K |

三阶段调用对齐 cuSPARSE：

```text
GetBufferSize -> Preprocess（可选） -> SpMV
```

### 交付范围（本方案）

本方案面向 Ascend 950PR / arch35 的 `aclsparse` SpMV，交付边界如下。

| 维度 | 纳入交付 | 不纳入 / 非本任务主体 |
| --- | --- | --- |
| 硬件 | Ascend 950PR（`dav-3510`） | arch22 / A2 既有产品线实现仅作对照 |
| API | `aclsparseSpMVGetBufferSize` / `Preprocess` / `SpMV` | 不新增 950 私有平行 API |
| 矩阵格式 | CSR，`INDEX_32I`，`INDEX_BASE_ZERO` | COO / CSC 作为对外输入格式 |
| 运算 | 非转置 / 转置；alpha / beta 融合；原位 Y | 复数、批量 SpMV |
| dtype | 任务书 7 组合法组合 | 任务书未列组合 |
| 稀疏泛化 | 稀疏度 50%~99.9%；空行 / 长短行 / 大尺度 | — |
| 确定性 | 同输入多次执行结果一致 | — |
| 性能目标 | 任务书 case 标杆 ×0.5（gate） | 本文不贴实测数字，以设计与门禁定义验收 |

工程落点（目标仓 `ops-sparse`）：

| 组件 | 路径 |
| --- | --- |
| Host / API | `sparse/spmv/arch35/spmv_host.cpp` |
| Tiling / 列切 | `sparse/spmv/arch35/spmv_tiling.h` |
| 转置 kernels | `sparse/spmv/arch35/spmv_kernel.cpp` |
| 非转置 kernels | `sparse/spmv/arch35/spmv_nont_kernel.cpp` |
| 多 dtype | `sparse/spmv/arch35/spmv_nont_dtype_kernels.cpp`、`spmv_dtype.h` |
| 构建 | `sparse/spmv/arch35/build_spmv.sh` |
| 测试 | `test/spmv/arch35/` |

### 仓内既有实现与设计动机

| 类型 | 路径 | 用途 |
| --- | --- | --- |
| 公共接口 | `include/cann_ops_sparse.h` | `aclsparseSpMV*` 原型 |
| arch22 参考 | `sparse/spmv/arch22/` | 910 既有 Host / Kernel（行分核 + UB；转置常依赖原子） |
| 本方案 | `sparse/spmv/arch35/` | 950PR AIV + `asc_vf_call` SIMT |

950PR 相对 arch22 的关键差异：不规则 gather 更适合 SIMT 直访 GM；大 `|x|` 时 DCache（约 128KB 量级）成为瓶颈，需在 Preprocess 做列切以换取稳态 SpMV 的访存局部性；确定性要求避免「无序原子累加」成为主路径。

# 需求分析（required）

## 需求描述

在 Ascend 950PR 上交付与 cuSPARSE SpMV 功能对齐的算子，覆盖任务书功能、精度、性能门禁与确定性要求，并以可复现的 ACL / 泛化测试支撑验收。

## 需求拆解

| 编号 | 类别 | 需求要点 |
| --- | --- | --- |
| R1 | 格式与校验 | CSR int32 零基索引；rowPtr 单调、colInd 合法、NNZ 一致 |
| R2 | 非转置 | `Y = alpha * A * X + beta * Y` |
| R3 | 转置 | `Y = alpha * A^T * X + beta * Y` |
| R4 | dtype | 任务书 7 组 A / X / compute / Y；窄类型写回规则明确 |
| R5 | alpha / beta | 含 `{0, 0.5, 1, 2}` 及一般浮点；`beta = 0` 等价纯 SpMV |
| R6 | 三阶段 API | GetBufferSize / Preprocess（可选）/ SpMV；多次 SpMV 复用同一 buffer |
| R7 | 稀疏泛化 | 50%~99.9%；常规 / 边界 / layout / 全 dtype |
| R8 | 确定性 | 同输入多次结果一致；各执行路径须有固定归约顺序或无竞态写 Y |
| R9 | 精度 | 满足生态算子开源精度标准 |
| R10 | 性能 | 任务书 4 组 case 达到标杆 ×0.5 |

## 场景轴（用于路径选择与确定性分析）

| 场景轴 | 取值 / 触发条件 | 设计意图 |
| --- | --- | --- |
| opA | nont / trans | 语义与 X / Y 长度互换 |
| 行长分布 | 短均匀 / 长行 / skew | Row-wise vs MergePath |
| `|x|` vs DCache | 单 tile 可容纳 / 需多 tile | 是否 Host 列切 |
| 转置 cols | ≤ columnLimit / 更大 | SmallColumn vs CSC + 列切 |
| dtype | 7 组 | 累加宽度与写回 |
| beta | 0 / 非 0 | 首写 vs 原位累加；列切后续 tile 的 `beta_eff` |

# 详细设计（required）

## 算子分析

### 数学公式

```text
Y = alpha * op(A) * X + beta * Y
op(A) ∈ {A, A^T}
```

### 支持数据类型

| 输入 A、X | computeType | 输出 Y | 累加 / 写回策略 |
| --- | --- | --- | --- |
| float32 | float32 | float32 | fp32（长行可启用 Kahan） |
| int8 | int32 | int32 | 整数累加 |
| int8 | float32 | float32 | 提升至 fp32 累加 |
| float16 | float32 | float32 | fp32 / Kahan |
| float16 | float32 | float16 | fp32 累加后写回 fp16 |
| bfloat16 | float32 | float32 | fp32 / Kahan |
| bfloat16 | float32 | bfloat16 | fp32 累加后 `__float2bfloat16` 写回 |

### 支持形状

CSR `M × K`；维度与 NNZ 受 `int32` 索引约束。需覆盖非对齐小 / 中 / 大规模及任务书大尺度 case（如 `160220 × 68750`、高稀疏度）。

## 算子实现

### 实现方案

```text
GetBufferSize  → 按 opA / dtype / 是否列切估计 workspace
Preprocess     → Host：统计选路；必要时 D2H →（CSR→CSC）→ 列切 → H2D 写 SpmvWsHdr
SpMV           → 读 hdr，启动对应 SIMT kernel（只计算，不再重排矩阵）
```

路径选择原则：

1. 非转置：行统计 → Row-wise 或 MergePath；若选 Row-wise 且 `|x|` 超 DCache 预算 → Host 列切。
2. 转置：cols 小 → SmallColumn（partial + stage2）；cols 大 → Host CSR→CSC，再按 `A^T` 视角做列切 Row-wise。
3. 确定性优先：主交付路径对 Y 采用「单写者 / 固定顺序归约」，避免无序 `atomicAdd` 成为大矩阵主路径。

#### 3.2.1 host侧设计：

**分核与路径选择**

非转置（`SpmvChooseNontTuning`）：

1. 统计行 nnz（min / max / mean）。
2. 短行、较均匀 → Row-wise。
3. 行长 skew / 过长 → MergePath（按 `rows + nnz` 均分工作量）。
4. `SpmvRowWiseColTiles(cols) ≥ 2` 且选 Row-wise → Host 列切（按列区间划分 nnz，不复制数值到多份完整矩阵）。

转置（`SpmvChooseTuning` + Host CSC）：

1. `cols ≤ columnLimit`（由 UB 累加预算与线程配置推导）→ SmallColumn。
2. 否则 → Host CSR→CSC，再对 CSC 视图按原 `rows`（转置后的 `|x|`）做列切。

列切动机是 `|x|` 的缓存局部性，不是「行很长」。高稀疏、短行但列维很大的矩阵仍需要列切。

**Preprocess 与 Workspace**

1. nont 多 tile：D2H → `SpmvSplitCsrByColTiles` → H2D（每 tile 一份 `rowPtr` + 共享 `col` / `val` pool）→ `SpmvWsHdr`。
2. trans 大 cols：D2H → CSR→CSC → 列切上传。

列切路径布局示意：

```text
[SpmvWsHdr]
[tile0 rowPtr] … [tileN-1 rowPtr]
[col pool]   # nnz
[val pool]   # nnz * valBytes
```

SmallColumn / MergePath 使用各自的 partial / warp 起点缓冲，规模远小于「nnz 同量级」的列切 workspace。

**启动参数**

1. AIV 核数：平台查询，缺省回退固定核数。
2. 线程块：默认 256（可环境变量覆盖）；SmallColumn 可按 cols 自适应。
3. Row-wise：`dynUBuf = 0` 以保留更多 DCache。
4. `matA->activeBuffer`：已预处理 buffer 快路径，避免重复 Preprocess。

#### 3.2.2 kernel侧设计：

**编程模型**

外层 `__aicore__`（`KERNEL_TYPE_AIV_ONLY`）+ `asc_vf_call` 派发 `__simt_vf__`。SIMT 线程直接 GM 标量访问，适配不规则稀疏 gather。

**非转置 Row-wise / col-tile**

每线程独占若干行：沿 CSR 行段顺序累加，写回 `y[r] = alpha * acc + beta * y[r]`。  
列切时对每个 tile 启动一次 Row-wise：tile0 使用真实 `beta`，后续 tile 使用 `beta_eff = 1` 累加到同一 `y`。

**非转置 MergePath**

preprocess 写 warp 对角线起点 → main 按固定 path 顺序段内归约 → fixup 按行号顺序合并跨 warp 残段。用于行长不均。

**转置 SmallColumn**

stage1 将各块 partial 写到固定槽位；stage2 按列维固定顺序归约到 `y`。不依赖 GM 原子竞争。

**转置 CSC + col-tile**

CSC 在语义上等价于 `A^T` 的 CSR；之后完全复用 nont 列切 Row-wise，因此继承其「每行单写者」确定性。

### 各场景确定性实现分析

确定性定义：相同设备、相同输入（含 alpha / beta、dtype、CSR 与向量内容）、相同算法路径下，多次调用 SpMV（含 Preprocess 后重复计算）得到逐元素一致的 `Y`。

浮点路径的「一致」指同一实现固定求值顺序下的可复现结果（与更高精度 golden 的误差仍由精度标准约束，不要求与无序归约或异构实现 bit 级相同）。

| 场景 | 路径 | 写 Y 方式 | 确定性论据 |
| --- | --- | --- | --- |
| nont × 短/均匀行 × `|x|` 可进 DCache | Row-wise | 每行唯一线程顺序扫 nnz 后单次写 | 无跨线程写冲突；行内累加顺序由 CSR 存储序固定 |
| nont × 长行 / fp16·bf16 长累加 | Row-wise + float Kahan | 同上，块内 Kahan 再外层合并 | 补偿顺序由固定分块规则决定；仍无原子 |
| nont × `|x|` 超 DCache | Host 列切 + 多趟 Row-wise | tile0：`beta`；tile k>0：`beta = 1` | 列区间划分由 Host 确定性算法完成；tile 启动顺序固定；每行每趟仍单写者 |
| nont × 行长 skew | MergePath | 完整行直接写；残段进 partial，fixup 按行合并 | path 划分与 warp 起点由 preprocess 固定；fixup 按行遍历，合并操作数顺序固定 |
| trans × 小 cols | SmallColumn | stage1 partial → stage2 定序归约 | partial 槽位与 stage2 归约顺序由分块参数固定；无 GM 竞态写 |
| trans × 大 cols | Host CSC + 列切 Row-wise | 同 nont 列切 | CSC 转换与列切为 Host 确定性重排；计算阶段无原子 |
| `beta = 0` | 各路径 | 首写或等价清零后写 | 不读入「未定义旧 Y」依赖 |
| `beta ≠ 0` | 各路径 | 读-改-写同一元素由单写者完成 | 列切多趟时后续趟 `beta_eff = 1`，等价于对已写入值的定序累加 |
| int8→int32 | Row-wise 类路径 | 整数累加 | 结合律下同序累加结果唯一；无浮点舍入顺序问题 |
| 窄类型写回 | fp16 / bf16 out | 在 fp32 累加完成后再转换 | 转换点固定在行（或归约）结束之后，避免中途反复量化 |

刻意规避的不确定源：

1. 无序 `atomicAdd` 到同一 `y[c]`：浮点原子累加顺序依赖调度，不作为大矩阵转置主路径。
2. SpMV 阶段再做矩阵重排：重排固定在 Preprocess，避免「第一次 SpMV 与后续 SpMV」行为分叉。
3. 多线程共写同一行而未定序归约：Row-wise / CSC 列切保证一行一写者；MergePath / SmallColumn 用 partial + 定序 fixup。

边界场景说明：

| 边界 | 行为 | 确定性 |
| --- | --- | --- |
| 空行 / NNZ=0 | 写 `y[r] = beta * y[r]`（或 `beta = 0` 时写 0） | 无累加循环，结果唯一 |
| 单行超长 | 倾向 MergePath 或带 Kahan 的 Row-wise | 仍走上表对应论据 |
| 多 tile 且某 tile 空列段 | 该趟对行贡献为 0 | 跳过或累加 0，不改变定序 |
| Preprocess 省略 | 若 API 允许直接 SpMV | Host 在 SpMV 入口仍按同一阈值选路；若需要列切 / CSC 则要求先 Preprocess 或在入口补齐，保证路径与 buffer 布局一致 |

关键数据流（大 `|x|` 非转置）：

```text
Device CSR
  -> Host D2H
  -> SpmvSplitCsrByColTiles
  -> H2D tiles + hdr
  -> Row-wise tile0 (beta)
  -> Row-wise tile1..n (beta_eff=1)
  -> y
```

# 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Ascend 950PR（dav-3510 / arch35） | √ |
| Atlas 800I/T A2（arch22 既有路径） | 参考，非本任务交付主体 |

# 算子约束限制

1. 仅 CSR；`INDEX_32I`；`INDEX_BASE_ZERO`。
2. 维度 / NNZ 不超过 `int32` 可表示范围。
3. DnVec 当前按 contiguous 设备缓冲处理（测试中 `strided` 标签亦走 contiguous 路径时需在验收说明中标明）。
4. 列切 / CSC 预处理在 Host 完成，workspace 与 nnz 同量级；稳态 SpMV 与重排分离。
5. 算法路径由 Host 阈值自动选择，不向用户暴露私有平行 API。
6. 确定性保证针对本实现固定路径；跨实现 / 跨精度对照以精度标准衡量，不以 bit-exact 为跨实现要求。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述(不涉及说明原因) | 标准来源 |
| --- | --- | --- |
| 精度标准 | 相对 host / 高精度参考，满足生态开源精度标准；bf16 按生态容差 | 生态算子开源精度标准；任务书 §3.2 |
| 性能标准 | 任务书 case1–4 达到标杆 ×0.5；稳态 SpMV 计时与 Preprocess 分离 | 任务书 §3.3 |
| 确定性 | 同输入重复执行结果一致；覆盖各主路径至少一例 | 任务书 §2.1 确定性要求 |

测试工程应覆盖：常规 / 边界、转置与非转置、全 dtype、alpha / beta 组合、稀疏度 50%~99.9%，以及触发 Row-wise、MergePath、列切、SmallColumn、CSC + 列切 的代表性形状。

## 兼容性分析

1. 接口落在既有 `aclsparseSpMV*`，与 arch22 共用声明，不新增 950 私有平行 API。
2. arch35 为按 `dav-3510` 路由的新增实现，不影响 arch22 产品线既有行为。

## 附录：提交信息

| 项目 | 内容 |
| --- | --- |
| 社区任务 | 8月社区任务-SPMV算子开发 |
| tasklist | `08-16-SPMV` |
| 团队目录 | `yefanjia` |
| 目标仓 | https://gitcode.com/cann/ops-sparse |
| 竞赛仓分支 | `08-16-SPMV` |
| 本文档路径 | `04_tasks/01_community-task-2026/tasklist/08-16-SPMV/yefanjia/docs/design.md` |
