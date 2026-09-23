# aclblasStpsv 算子设计文档（社区任务 PR 提交版）

> **算子**：`aclblasStpsv`（单精度实数三角方程组求解，打包存储，BLAS-2）
> **任务**：算子实操工坊-北京站——aclblasStpsv 算子开发（Atlas A2/A3）
> **目标架构**：arch22 / DAV_2201（Ascend 910B3，Atlas A2/A3），实现位于 ops-blas 仓 `blas/tpsv/arch22/`，与 950PR `arch35` 同目录共存
> **基线接口**：cuBLAS `cublasStpsv`（语义口径对齐 Netlib BLAS `stpsv.f`）

## 1. 需求背景（required）

### 1.1 需求来源

CANN 社区任务 2026 · 算子实操工坊-北京站「aclblasStpsv 算子开发（A2/A3）」。要求在 Atlas A2/A3 系列产品上以 Ascend C 实现 `aclblasStpsv`（ops-blas 句柄式 BLAS 单段接口），完成设计、开发、测试全流程，验收通过后合入 ops-blas 开源仓。

### 1.2 算子功能现状分析

ops-blas 公共头文件 `include/cann_ops_blas.h` 中已有 `aclblasStpsv` 声明（与同族 `aclblasStrsv` 相邻），950PR 侧已有 arch35 实现；**Atlas A2/A3（arch22）目录缺少对应实现**，本任务补齐 arch22 实现与测试，接口声明不新增、不修改。

`stpsv`（Triangular Packed matrix Solve）求解**打包存储**的三角线性方程组：

```text
op(A) * x = b
```

- A 为 n×n 三角实数矩阵，以一维打包数组 AP（逻辑长度 n(n+1)/2）按**列主序**逐列无间隙堆叠 `uplo` 指定三角（含对角），**无前导维 lda**；另一三角不存于 AP，由三角性隐含（op(A) 中取零）；
- x 入口存放右端项 b，出口**原地覆写**为解向量；
- `op(A)` 由 `trans` 决定：OP_N→A、OP_T→A^T、OP_C→A^H（实数矩阵共轭为其自身，**OP_C 与 OP_T 语义等价**）；
- 求解方向：LOWER/N 前代（x₁→xₙ 顺序求解）、UPPER/N 回代（xₙ→x₁ 逆序求解），OP_T/OP_C 方向相反；uplo(2) × trans(3) × diag(2) 共 **12 组枚举组合**全部支持；
- `diag=NON_UNIT` 对角元从 AP 读取并参与除法；`diag=UNIT` 对角元不被访问、按 1 处理；
- 支持任意非零 `incx`（含负步长，按 Netlib stpsv 语义反向遍历）；`n=0` 为合法 no-op（quick return）；不做奇异性/近奇异性检测（与 cuBLAS/Netlib 一致）。

## 2. 需求分析（required）

### 2.1 需求描述

使用 Ascend C 在 arch22（DAV_2201）实现 `aclblasStpsv`：FLOAT32 单 dtype、12 组枚举组合、任意非零步长（含负）、原地覆写语义、n=0 合法退化；host 校验 + 单 kernel 直调，异步执行。

### 2.2 需求拆解

1. **功能语义**：packed 列主序索引、前代/回代、UNIT 对角豁免、负步长反向遍历、n=0 quick return；
2. **参数校验与错误码**：host 固定校验序，非法参数返回任务书状态码（非法枚举口径为 `ACLBLAS_STATUS_INVALID_VALUE`，与仓内测试资产及 arch35 实现一致，非任务书字面 INVALID_ENUM）；
3. **精度**：cblas（Netlib stpsv）单标杆 golden，FLOAT32 开源精度标准（rtol=2⁻¹⁰、atol=2⁻¹⁶、matched_ratio≥0.99、max_abs_error≤max(1e-2, 32ULP)），非 bit-exact、误差随 n 累积；
4. **性能**：任务书四标杆（Atlas 800I A2 / 910B3，FLOAT32，warmup 后有效采样取平均）——n=512/1024/2048/4096 标杆 432.71 / 941.23 / 1690.94 / 5185.2 µs；
5. **工程化**：ops-blas 句柄式直调范式（`<<<numBlocks, nullptr, stream>>>`），Device 指针直达 kernel、host 零数据搬运、无内部 stream 同步；1000 条精度 + 200 条性能用例 CSV 驱动测试。

## 3. 详细设计（required）

### 3.1 算子分析

**数学公式**（0 基，LOWER 为任务书 §2.1.3 勘误后正确式，与打包布局定义、cuBLAS、Netlib、仓内 arch35 实现四方一致）：

$$
op(A) \cdot x = b,\qquad x_\text{out} = op(A)^{-1} \cdot b
$$

$$
\text{UPPER: } A(i,j) = AP\big[i + j(j+1)/2\big],\ i \le j;\qquad
\text{LOWER: } A(i,j) = AP\big[i + (2n-j-1)\,j/2\big],\ i \ge j
$$

前代/回代递推（LOWER/N 前代、NON_UNIT，1 基示意）：

$$
x_i = \frac{b_i - \sum_{j=1}^{i-1} A(i,j)\, x_j}{A(i,i)},\qquad i = 1, 2, \dots, n
$$

回代（UPPER/N）逆序 i = n, …, 1；OP_T/OP_C 方向相反。x_i 依赖所有已解出的 x_j——递推链存在 inter-iteration 顺序依赖；trailing 更新段（列 axpy / 行点积）无依赖回边，可向量化而不改变逐元素结果。

**支持数据类型**：仅 FLOAT32（AP、x 均 float32 → 输出 x float32），原 dtype 直算无 Cast。

**支持形状**：AP shape `[n(n+1)/2]`（ND，一维打包）；x 逻辑长度 n、物理长度 ≥ 1+(n-1)·|incx|；n 为运行时入参，n=0 合法退化；无广播（AP 与 x 为独立操作数）；能力上限按索引域安全约束论证（见 §3.5）。

### 3.2 实现方案

总体路线：**通用 SIMD/MemBase（AIV 向量指令 + MTE 搬运），串行递推链单 AIV 核闭合 + 每步重活宽向量化的双形态**。递推链 O(n) 步串行不可并行化（arch22 无 SIMT、无通用 AIV 全局栅栏，跨核编排收益不抵复杂度；块逆/迭代并行化在同族算子战役中已被真机否证超预算一个数量级），故默认单核闭合，把每步计算压到宽向量指令：

- **OP_N 右视 axpy**：解出 x_i 后以 AP 第 i 列做 trailing 更新，`Axpy` 一条宽指令/步；
- **OP_T/OP_C 左视 dot**：x_i 由 AP 第 i 列点积得出（逐元素乘 + 分级归约）。

packed 列主序下每列均为 AP 内**连续稠密段**（UPPER 列 j 起点 j(j+1)/2 长 j+1；LOWER 列 j 起点 j(2n-j+1)/2 长 n-j），MTE2 整段装载无空洞。

#### 3.2.1 host 侧设计

**1. 参数校验序（顺序锁定，与 arch35 先例一致）**：

① `handle == nullptr` → `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`（最优先）→ ② `n < 0` → `INVALID_VALUE` → ③ `n == 0` 先行 quick return `SUCCESS`（不引用 AP、不更新 x，**先于** AP/x 空指针校验）→ ④ uplo/trans/diag 非法枚举 → `INVALID_VALUE` → ⑤ `incx == 0` → `INVALID_VALUE` → ⑥ `n > 0` 时 AP/x 为 nullptr → `INVALID_VALUE` → tiling 组装与下发。

**2. 枚举折叠**：`trans = ACLBLAS_OP_C` 在 host 折叠为 OP_T（实数共轭为其自身），12 组枚举组合收敛到 **8 个编译期 kernel 模板**（uplo 2 × trans{N,T} 2 × diag 2）。

**3. tiling 组装与下发**：本算子为 kernel 直调模式，无 registry TilingKey 机制——路径分派由 host 侧 switch 完成，运行期参数经小型 POD tiling 结构（n / incx / AP 与 x 的 Device 指针 / 核数 / 列段分块宽度）GM 传递给 kernel。tiling H2D 一次（异步拷贝，按 stream 维度缓存复用），随后 `kernel<<<numBlocks, nullptr, stream>>>` 直调下发、立即返回；AP/x 为 Device 指针，**无任何数据 H2D/D2H、无内部 `aclrtSynchronizeStream`**（异步语义由调用方读回前同步承接）。

**4. 列段分块宽度**：按 UB 预算不等式与 repeatTimes ≤ 255 约束计算（小中 n 整列单块、启用 kernel 侧预取重叠快路径，大 n 档收缩分块回退），host 一次算定、运行期零重算。

#### 3.2.2 kernel 侧设计

单 AIV 向量内核，Init + Process 两阶段，编译期 `<UPLO, TRANS, DIAG>` 8 模板实例化、运行期零枚举分支：

- **Init**：解析 tiling 还原 AP/x GM 指针；x/b 装载 GM→UB 后**全程常驻**（fp32 n=4096 仅 16KB ≪ 192KB UB），原地覆写；incx≠1（含负）时按物理跨度装载 + 索引表映射到紧凑逻辑平面（装载与写回双向复用同一套映射）；计数模式掩码 set/reset 提升到主循环外。
- **主循环（前代或回代，每步）**：
  - 对角分母：NON_UNIT 从 AP 列段标量提取（**UPPER 模式对角 = 列 i 末元素、LOWER 模式对角 = 列 i 首元素**，列内连续可向量化抽取）；UNIT 跳过除法；
  - 右视 axpy（OP_N）：AP 列 i 去对角后的连续段 GM→UB（双缓冲预取下一列），`Axpy` 一条宽指令完成 b −= AP_col·x_i；
  - 左视 dot（OP_T/OP_C）：AP 列 i 已解段 × x 已解段逐元素乘，分级归约求和（宽段用二级级联归约原语，禁用单指令全和于超 64 长度段），`x_i = (b_i − dot) / d`；
  - 段长超 repeatTimes 上限按分块宽度分批；非 64 倍数尾块用计数模式掩码覆盖，GM→UB 装载一律显式右垫零防垃圾 lane 参与归约。
- **数据流**：AP 列段双缓冲，MTE2 搬运与 V 计算经事件同步重叠（当前列计算与下一列预取并行）；x 求解循环内读写零 GM 流量；AP 全量流式过核不常驻。
- **输出写回**：incx=1 时解平面整段一次写回 GM x 原地址；incx≠1 时经逆映射散入跨度镜像平面后整段写回（间隙位置保留装载原值，不破坏调用方数据）；极端大步长超出 UB 预算时回退逐元素 strided 搬运（仅保功能正确）。

**内存规划**（DAV_2201 总 UB 192KB/核，向量可用按 184KB 估算）：x/b 常驻平面 4n B；AP 列双缓冲 2×4×chunk B；OP_T/C 另设 dot 乘积平面 + 归约 scratch 2×4×chunk B；incx≠1 路径另增跨度镜像与索引表（仅该路径激活）。n=4096 最重模板（OP_T/C）合计约 50KB（占 27%）。**n 支持上限**由两条约束联合核定：①索引安全上限——n(n+1)/2 < 2³¹ 的 uint32 索引域安全边界，大中间积经 int64 展开；②x UB 常驻不等式解得的保守回填上限（超限走 x 分块装载的非常驻退化路径，软切换非硬墙）。

## 4. 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas A2 训练/推理系列产品（Ascend 910B3） | √（本任务交付） |
| Atlas A3 训练/推理系列产品 | √（本任务交付） |
| Ascend 950PR/DT | 支持（arch35 仓内已有实现，本任务不涉及） |

## 5. 算子约束限制

- `n ≥ 0`；`n=0` 合法 no-op（quick return，先于 AP/x 空指针校验，此时 AP/x 允许 nullptr）；
- `incx ≠ 0`（可正可负，负步长反向遍历）；`n > 0` 时 AP、x 不可为 nullptr；非法枚举返回 `ACLBLAS_STATUS_INVALID_VALUE`；
- 数据类型仅 FLOAT32；AP 为打包一维数组（无 lda）；非连续访问仅支持到 incx 步长语义；
- 仅引用 uplo 指定三角；diag=UNIT 时不读取 AP 对角、按 1 处理；
- 不做奇异性/近奇异性检测，NON_UNIT 时调用方保证对角元非零；Inf/NaN 不检测不拦截，按 IEEE 浮点沿依赖链传播；
- x 原地覆写（入口 b、出口解），不返回视图；无广播；异步执行（读回前须同步 stream）。

## 6. 可维可测分析（required）

### 6.1 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度 | cblas（Netlib stpsv）单标杆 golden，输出 x 全量逐元素：rtol=2⁻¹⁰、atol=2⁻¹⁶、matched_ratio≥0.99、max_abs_error≤max(1e-2, 32ULP)；非 bit-exact，误差随 n 累积 | 任务书 §3.2、生态算子开源精度标准 |
| 性能 | 4 条标杆（Atlas 800I A2，FLOAT32，warmup 后有效采样取平均）：n=512/1024/2048/4096 标杆 432.71 / 941.23 / 1690.94 / 5185.2 µs | 任务书 §3.3 |

测试侧设计：golden 由测试工程随包提供的 cblas 生成；NON_UNIT 对角按符号保持偏移（量级随 n 提升）保障远离零（golden 同口径生成，两侧输入严格一致）；CSV 覆盖 12 枚举组合、尺寸扫描、正负步长 ±1/±2/±3、n=0/n=1、空指针、非法枚举/步长、Inf/NaN 传播、全零右端项、正态/均匀分布输入；直测校验序门控（n=0 允许空指针等边界）；UNIT 对角位置填垃圾值验证零读取；性能用例 kernel 级计时，先 warmup 再有效采样取平均。

### 6.2 兼容性分析

实现仓内 `include/cann_ops_blas.h` 已有声明，不新增、不修改，与同族 `aclblasStrsv`（全存储版本，带 lda）接口独立、互不影响；arch22 实现仅在目标架构目录参与构建，与 arch35 同目录共存共用声明。新算子，无既有行为兼容性问题。
