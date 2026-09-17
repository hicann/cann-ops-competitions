# aclblasSspr 算子设计文档（社区任务 PR 提交版）

> **算子**：`aclblasSspr`（单精度实数对称秩-1 更新，打包存储，BLAS-2）
> **任务**：算子实操工坊-北京站——aclblasSspr 算子开发（Atlas A2/A3）
> **目标架构**：arch22 / DAV_2201（Ascend 910B3，Atlas A2/A3），实现位于 ops-blas 仓 `blas/spr/arch22/`，与 950PR `arch35` 同目录共存
> **基线接口**：cuBLAS `cublasSspr`（语义口径对齐 Netlib BLAS `sspr.f`）

## 修订记录

| 版本 | 修订内容 | 修订时间 | 修订人(gitId) |
| --- | --- | --- | --- |
| v1.0 | 初始方案设计：AIV 列级并行 + Elementwise 直算 + 双 TilingKey | 2026-09-11 | Admin05210 |
| v1.1 | 实现演进收口：Gather 向量化破相位墙、n-gated 混合精度、desc 预计算通道、UB 预算模型与标量降级（4 条性能标杆全达标） | 2026-09-13 | Admin05210 |

## 1. 算子描述

### 1.1 基本信息

| 项目 | 内容 |
| --- | --- |
| 算子名称 | `aclblasSspr`（复用 ops-blas `include/cann_ops_blas.h` 已有声明，禁止平行 API） |
| 算子类别 / 范式 | Elementwise（每个打包元素独立一次乘加，无跨元素依赖） |
| 支持数据类型 | x/AP 均为 float32，原 dtype 直算（无升精度中间 Buffer） |
| 目标芯片 | Ascend 910B3（Atlas A2/A3 系列；开发验证环境 Atlas 800T A2） |
| 编程框架 | Ascend C（AIV Vector 单范式多核并行，handle 式 BLAS 直调） |
| 运行环境 | CANN 9.1.0 |

### 1.2 功能描述

实现单精度实数**对称秩-1 更新（打包存储）**（对标 cuBLAS `cublasSspr`）：

```
A = alpha * x * x^T + A
```

- x 为 n 元素 float32 向量，A 为 n×n 对称 float32 矩阵，以打包存储（Packed Storage）一维数组 AP（长度 n(n+1)/2）表示，按列主序堆叠 `uplo` 指定三角的列（含对角）；无 lda、无 beta；
- `uplo = ACLBLAS_UPPER(121)`：第 j 列自对角 A(j,j) 向上至 A(0,j)，打包位置 `AP[j(j+1)/2 + i], 0≤i≤j`，列长 `j+1` 递增；
- `uplo = ACLBLAS_LOWER(122)`：第 j 列自对角 A(j,j) 向下至 A(n-1,j)，打包位置 `AP[j(2n-j+1)/2 + (i-j)], j≤i<n`，列长 `n-j` 递减；
- 仅 `uplo` 指定三角被引用与更新，另一三角不存于 AP、由对称性隐含；原地覆写。

边界语义（Netlib sspr 口径，host 校验序固定）：`handle=nullptr → HANDLE_IS_NULLPTR`；`n<0 / incx=0 / alpha/x/AP 为 nullptr → INVALID_VALUE`；`uplo 非法枚举 → INVALID_ENUM`（arch22 按任务书口径，与 arch35 的 INVALID_VALUE 不同，勿照抄）；`n=0` 合法 no-op（指针校验前，允许空指针）；`alpha=0` quick return（指针校验后，AP 逐位不变，不读 x）；`incx<0` 按 Netlib 语义反向遍历 `x[(n-1-i)·|incx|]`。

### 1.3 数学公式

$$
AP[\,pos(i,j)\,] \mathrel{+}= \alpha \cdot x(i) \cdot x(j),\quad
pos(i,j)=\begin{cases} j(j+1)/2+i & \text{UPPER},\ 0\le i\le j<n \\ j(2n-j+1)/2+(i-j) & \text{LOWER},\ 0\le j\le i<n \end{cases}
$$

## 2. 需求分析

| 项 | 要求 |
| --- | --- |
| 数据类型 | 仅 FLOAT32（原 dtype 直算，无 Cast） |
| 形状 | n ∈ [0, 2^31-1] 运行时入参；AP 长度 n(n+1)/2；x 物理长度 1+(n-1)\|incx\|；incx≠0 支持负步长 |
| 精度 | 生态算子开源精度标准 FLOAT32 档：atol=2⁻¹⁶、rtol=2⁻¹⁰、matched_ratio≥0.99、max_abs_error≤1e-2 或 32ULP；golden 由 cblas（Netlib sspr）生成；alpha=0 为 EXACT 位级比对 |
| 性能 | 4 条标杆（Atlas 800I A2，warmup + 有效采样>50 次取平均）：n=512/1024/2048/4096 对应 11.31/27.07/79.93/475.91 us |
| 可行性 | memory-bound（每元素 2 FLOPs），case3 所需有效带宽约 210 GB/s（≈54% 理论值），AIV 向量路线可覆盖，无需 Cube |

## 3. 详细设计

### 3.1 总体路线

- **AIV 向量路线（否决 Cube）**：算子为访存受限（memory-bound），每元素仅 2 FLOPs 且 AP 打包列内天然连续，适配 MTE 搬运 + V 管乘加；Cube 适用于稠密 GEMM 形态，对本算子带宽利用率反而更低。
- **列级→32B 行级两级切分**：核间以打包元素成本均衡分块（`C(x)=x+W·跨列数`，W 按实测标定分段取值），各核认领互不相交且 32B 对齐的 AP 区间（R2 轮真机实验证明多核 MTE3 写互不相交 32B 行无损）；核内再按窗口（≤3584 元素）分段。
- **TilingKey 双分支**：`TilingKey_0`（incx==1 连续快路径）/ `TilingKey_1`（incx≠1 通用路径，负步长反向遍历），编译期模板 `SsprAIV<IS_INCX1>` 分发；uplo/n 不进 TilingKey，由 TilingData 承载。

### 3.2 host 侧设计

**校验序（8 步固定）**：handle 判空 → n<0 → n=0 quick return → uplo 枚举（INVALID_ENUM）→ incx=0/INT_MIN → alpha/x/AP 判空 → alpha=0 quick return → tiling 计算与 launch。

**Tiling 计算**：`useCoreNum = min(GetAivCoreCount(), n)` 动态核数（禁止写死）；按成本模型均衡分块；`SsprUbBytes` UB 预算公式对全部缓冲区逐项核算（192KB 实测天花板，incx≠1 且 x 过大时按 64 元素步进收缩窗口，仍不闭合则置 `scalarMode=1` 降级单核标量路径，杜绝 UB 超限静默 no-op）。

**desc 预计算通道（小中 n 加速）**：每形状一次在 host 完成完整三角形几何（窗口表 + 每元素 row/col 字节偏移 idx 表），上传 GM 工作区（形状键缓存 ≤256 项，与 alpha/incx 无关）；kernel desc 模式每窗仅需 2×Gather + 逐列标量 αx + 全窗 Mul/Add 一条向量链，彻底消除核内列遍历（walk）与标量 fixup 遍历。desc 不可用时自动回退 legacy 路径，语义逐位不变。

### 3.3 kernel 侧设计

**相位墙与 Gather 向量化（核心难点）**：打包存储列首 `colStart mod 8 ≠ 0`（7/8 的列），AP 对齐窗口内列数据起始存在 1~7 元素相位差，对齐 `Muls+Add` 因式分解被代数证明不可行，唯一桥接原语是按元素取数。设计采用**元素级 `Gather`（srcOffset 为字节偏移）+ `ArithProgression` 索引表**精确取 `x[row]`，字节粒度偏移天然弥合任意相位（注意：`GatherMask` 是按位掩码压缩原语，不具备按索引取数语义，不可用）。

**n-gated 混合精度**：单条 `vaxpy` 为两轮舍入，与参考 BLAS 的 FMA 舍入在深抵消 + 超大 AP 时可能出现 MARE 离群（真机实测 n=3529 触发）。故采用按 n 分档：`n≤2048` 用朴素 2-op 路径（1~2 ulp，容差内），`n>2048` 保留 Dekker 精确链（`fl(fl(ap+p_r)+e)` 逐位复现 FMA）。有限性守卫（Abs→Compares→Select）拦截 |c|≥FLT_MAX 回落朴素结果，保证 NaN/Inf 传播与 golden 一致；Compares/Select 必须以 64 倍数 count 调用（910B3 实测 sub-64 损坏），缓冲区配 64 元素尾部裕量。

**流水与同步**：TPipe + `TQue<VECIN/VECOUT, 2>` 双缓冲，MTE2_V / V_MTE3 手动事件对（R4 轮实证 DeQue 自动同步在 910B3 不可靠）；desc 通道 idx 双缓冲 + 真预取（compute(w) 期间发射窗 w+1 的 MTE2）+ MTE2 等待集重定时（4→2 笔）；x 常驻 UB（≤16KB）；对齐 DataCopy / 非对齐 DataCopyPad（Ext 形态）/ 极短列标量兜底三级搬运分派；`uint64` 索引防大 n 溢出（colStart 上限 ≈2.3×10¹⁸）。

**性能实测**（Atlas 800T A2 / 910B3 / CANN 9.1.0，AMORT 口径 warmup 10 + 异步 200 次取平均，多轮中位数；标杆 = gpu_baseline ÷ 0.8）：

| case | n / uplo | 标杆 (us) | 实测 (us) | 判定 |
|---|---|---|---|---|
| 1 | 512 / UPPER | 11.31 | **10.7**（40 轮中位） | ✅ |
| 2 | 1024 / LOWER | 27.07 | **24.5** | ✅ |
| 3 | 2048 / UPPER | 79.93 | **51.0** | ✅（1.6× 裕量） |
| 4 | 4096 / LOWER | 475.91 | **188.3** | ✅（2.5× 裕量） |

调优历程：单核标量版（1627/6351/25174/125461 us）→ Gather 多核向量化（R3，↑24-193×）→ 成本均衡分核 + Axpy 融合 + 事件最小化（R5）→ 真预取 + 3 窗 gate（R11）→ MTE2 重定时 + 单窗化（R12，512 达标）。全量 200 条性能 case 回归 0 SLOWER（>10% 红线）。

### 3.4 精度设计

- golden 由 cblas（Netlib sspr）生成，AP 全数组验证（仅 uplo 三角打包位置参与比对）；
- alpha=0 quick return 位级 EXACT（x 含 NaN/Inf 亦不影响 AP）；
- NaN/Inf 输入按 cblas 口径传播，范围限 uplo 三角对应打包位置；
- 终局证据（@e9be95b，lib md5 4e4ace7c）：C++ ST **1210/1210**、UT **243/243**、torch 适配层 ST **90/90**（golden 自测 744/744）、白盒 **323/323**（306 数据 + 17 校验序 gate）、官方 verify_accuracy **1010/1010**。

## 4. 产品支持与测试

| 产品 | 支持情况 |
| --- | --- |
| Atlas A2/A3 训练与推理系列（arch22） | **支持（本任务新增）** |
| Ascend 950PR/DT（arch35） | 支持（上游已有实现，arch22 与其同目录共存、接口共用） |

测试工程位于 ops-blas 仓 `test/spr/sspr/arch22/`：CSV 驱动 GTest（任务配套 1200 条 + 补充 8 条 + 96 条 L0/L1/L2 分级用例）、CPU golden 自测、UT、torch 直调适配层、白盒用例与性能回归，全部经真机（910B3）验证；执行步骤见该目录 README，代码仓 `feat/sspr-arch22` 分支（基线 origin/master 7eae232）。
