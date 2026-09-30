# aclblasSgemmGroupedBatched 算子设计文档

> 提交目录：cann-competitions `04_tasks/01_community-task-2026`（按社区任务设计文档模板）

# 需求背景（required）

## 需求来源

9 月社区任务：在昇腾 NPU（Atlas A2/A3 系列产品）上基于 Ascend C 编程语言开发单精度实数（FLOAT32）分组批量矩阵乘算子 `aclblasSgemmGroupedBatched`，对齐 cuBLAS `cublasSgemmGroupedBatched` 算子能力，验收通过后合入昇腾算子开源仓 ops-blas（目录 `blas/gemm_grouped_batched/arch22/`）。

## 背景介绍

### 算子定位

`aclblasSgemmGroupedBatched` 执行分组批量矩阵乘：多组形状完全不同、彼此独立的矩阵乘法打包成一个组计算。对第 g 组（g = 0..groupCount-1），其 m_g × n_g × k_g、transa_g/transb_g、alpha_g/beta_g、批量大小 groupSize_g 均独立设定；组内对 j = 0..groupSize_g-1（扁平下标 idx = Σ_{t<g} groupSize[t] + j）执行：

```
C[idx] = alpha_g · op(A[idx]) · op(B[idx]) + beta_g · C[idx]
```

所有矩阵列主序（Column-Major）存储；Aarray/Barray/Carray 为 Host 侧扁平指针数组（元素指向 Device 矩阵地址），长度 = Σ groupSize[g]，无 stride 语义。

### 主仓现状

ops-blas 主仓已有该算子的 Ascend 950 实现（`blas/gemm_grouped_batched/arch35/`，AIV 向量外积路线）。本任务新增 Atlas A2/A3 对应架构目录 `arch22`（DAV_2201）。两平台计算单元模型差异显著：达标算力需求（任务书口径有效 12~15 TFLOPS）超出 A2 Vector 峰值（约 8.4 TFLOPS，40 AIV 估算），arch22 必须走 Cube（Matmul 高阶 API）路线；950 的 SIMT 向量吞吐模型不适用于本平台。

### 性能验收基线

任务书 §3.3 给出 5 条典型性能 case 的达标耗时（GPU 实测 ÷ 0.8），测试设备 Atlas 800T A2 (910B3)，FLOAT32 平均单次 kernel 耗时（有效采样 10 次平均）：

| case | groupCount | groupSizeArray | m=n=k | transa/transb | 达标耗时 (us) |
|---|---|---|---|---|---|
| 1 | 2 | [64,64] | 256 | N/N | 351.3 |
| 2 | 2 | [128,64] | 512 | N/N | 3714 |
| 3 | 3 | [64,128,64] | 1024 | N/T | 37652 |
| 4 | 2 | [128,128] | 2048 | T/N | 298380 |
| 5 | 2 | [64,64] | 4096 | N/N | 1188440 |

5 条 case 需求均匀落于 12.2~14.8 TFLOPS（有效算力），对本 SKU fp32 Cube 峰值（实测校准 ≥54.5 TFLOPS）占比约 18~23%，无离群高风险 case。

# 需求分析（required）

## 需求描述

在 Atlas A2/A3（DAV_2201）上以 kernel 直调（Ascend C）方式实现 `aclblasSgemmGroupedBatched`，接口签名与主仓 `include/cann_ops_blas.h` 完全一致：

```cpp
aclblasStatus_t aclblasSgemmGroupedBatched(
    aclblasHandle_t handle, int groupCount,
    const aclblasOperation_t* transaArray, const aclblasOperation_t* transbArray,
    const int* mArray, const int* nArray, const int* kArray,
    const float* alphaArray, const float* const* Aarray, const int* ldaArray,
    const float* const* Barray, const int* ldbArray,
    const float* betaArray, float* const* Carray, const int* ldcArray,
    const int* groupSizeArray);
```

异步执行依赖 `aclblasSetStream` 绑定 stream；读回 Device 结果前须同步。

## 需求拆解

1. **功能**：逐组独立形状/转置/标量/批量的分组批量 GEMM；列主序；Host 侧指针数组
2. **异常与 no-op 语义**（Host 接口层校验）：handle null → HANDLE_IS_NULLPTR；groupCount<0 / 非法枚举（含 OP_C）/ 负维度 / 负 groupSize / ld 下界不满足 / 数组或元素 nullptr → INVALID_VALUE；groupCount=0 直接成功；m/n/groupSize=0 该组跳过；k=0 → C=beta·C；alpha=0 → 跳过矩阵乘；beta=0 不读 C（NaN 安全）
3. **精度**：golden = OpenBLAS cblas_sgemm（Netlib 语义）逐组列主序；混合容差 rtol=atol=2⁻¹³；matched_ratio ≥ 0.99；max_abs_error ≤ max(1e-2, 32×ULP)（大数规约场景按生态标准酌情放宽，见可维可测分析）
4. **性能**：5 条典型 case 达标（上表）
5. **交付**：`blas/gemm_grouped_batched/arch22/`（算子）+ `test/gemm_grouped_batched/sgemm_grouped_batched/arch22/`（测试，CSV 驱动 GTest，覆盖任务配套 1200 条用例）

# 详细设计（required）

## 算子分析

### 数学公式

对第 g 组第 j 个矩阵（扁平 idx）：

```
C[idx](m×n) = alpha_g · op(A[idx])(m×k) × op(B[idx])(k×n) + beta_g · C[idx]
op(X) = X (trans=OP_N) 或 Xᵀ (trans=OP_T)
存储：X(r,c) 位于 X + r + c·ld（列主序）
```

**列主序→行主序换轴恒等式**（Ascend C Matmul 高阶 API 为行主序 ND 格式；A2 平台无 COLUMN_MAJOR 格式——已查证 SetTensorA.md 产品支持表，COLUMN_MAJOR 仅 950 支持）：

```
Cᵀ(n×m) = alpha_g · (op(B)ᵀ)(n×k) · (op(A)ᵀ)(k×m) + beta_g · Cᵀ(n×m)
```

存储等价：列主序矩阵 X(r,c)@r+c·ld 的存储 ≡ 行主序 Xᵀ（行 stride=ld）。Device 侧矩阵存储零拷贝即可被行主序 matmul 消费。换轴后 Matmul 计算 `D[M',N'] = A'[M',K'] × B'[K',N']`，其中 M'=n、K'=k、N'=m、D=Cᵀ；转置标志完美对称：**A' 的 isTrans = (transb==OP_T)，B' 的 isTrans = (transa==OP_T)**。该思路与主仓 arch35 的 `ApplyColMajorSwap`（"tempAB stores Cᵀ"）一致，为官方仓已验证方案在 arch22 的移植。

### 支持数据类型

FLOAT32（A/B/C/alpha/beta 全 fp32，无 Cast）。fp32 走 Cube `KEEP_FLOAT_DTYPE` 路径（`SetHF32(false, 0)` + Host `SetMadType(NORMAL)` 探测式调用）：HF32（≈10-bit 尾数）在 K=4096 累积误差量级 ~0.03，超出容差 atol+rtol·|golden|≈0.008，必须全精度 fp32。

### 支持形状

通用：groupCount / 逐组 m,n,k / trans / alpha / beta / ld / groupSize 全部为运行时入参（Host tiling 支持），无固定 shape。维度范围 m/n/k ∈ [0, 4096]、groupSize ∈ [0, 1024]、groupCount ∈ [0, 8]（任务用例集覆盖域）。

## 算子实现

### 实现方案

#### host 侧设计

Host 侧完成全部校验、分桶、tiling 生成与 kernel 下发，Device 侧只做计算。

**1. 参数校验**：按任务书 §2.4 逐项校验（枚举合法性、维度非负、ld 下界、指针数组逐元素判空、groupSize 非负），非法返回 `ACLBLAS_STATUS_INVALID_VALUE`；全 SKIP 组（no-op）直接返回成功。

**2. 逐组转置分桶（4 桶模板实例）**：按 (transa, transb) 将问题分为 NN/NT/TN/TT 四桶，每桶一个 `AscendC::Matmul` 模板实例（`MatmulType<GM, ND, float, isTrans>`）+ 一次 kernel launch。原因：fp32 下 SetTensorA 的运行时转置标志、模板 ISTRANS、tiling isTrans 存在"三方一致"强约束，运行时交替不可行；桶 key 为 (m, n, k, transa, transb, lda, ldb)（组内形状相同），桶数 ≤ groupCount×4。

**3. 批感知 tiling 生成（性能关键，TC_PF_1001 优化）**：`MultiCoreMatmulTiling` 默认按"单问题填满全部核"切分，忽略桶内还有 probsInBucket 个同构问题——小问题（如 256³）被切成细条（2048 微块），MTE2 发射开销主导（实测超标 3.73×）。修正：按批内真实并行度重查 `effDim = ceil(aicNum / probsInBucket)`（须用全新求解器对象：`MultiCoreMatmulTiling` 带内部状态，且 `optiling::TCubeTiling` 拷贝赋值为浅拷贝，禁止结构体拷贝），接受重查结果需同时满足：
   - **L2 足迹守卫**：单块 A'+B' 足迹 ≤ L2/(2×aicNum)（运行时 `GetCoreMemSize(L2)` 查询）——大问题（4096³）保留细切防 L2 失效回退
   - **填充守卫**：`probs×mDim×nDim ≥ aicNum`（防欠填充）

**4. Ka/Kb 直读判定（免 Pack 紧凑化）**：TCubeTiling 的 Ka/Kb 字段即 K 维原始行距（官方脏列场景）。transb=N 桶 A'（B 存储，非转置消费 [n, ldb]）恒可直读（Ka=ldb，ldb≥k 由 ld 下界恒成立），零拷贝；transa=T 桶 B'（Kb=lda）直读候选；transb=T / transa=N 的 N' 维行距及 C 侧 ldc 无对应 tiling 字段——ld 有 padding 时走 AIV Pack 路径，ldc≠m 走 Epilogue。

**5. 快路径分类**：per-problem 判定——`(alpha=1, beta=0, ldc=m, m%8==0, C 基址 32B 对齐)` 直写 C（任务书 5 条性能 case 全命中，零 Pack/Epilogue 开销）；k=0/alpha=0 → Scale kernel（C=beta·C，beta=0 置零免读）；其余 → GEMM 落工作区 + Epilogue（`C[:,j] = alpha·D[j,:] + beta·C[:,j]`）。

**6. workspace 管理**：总量 = 用户区（问题描述 SgbProblemDesc + 桶 tiling + D 落区，512MB 预算）+ `GetLibApiWorkSpaceSize()` 系统区；`<<<blocks, 0, stream>>>` 直调形态（第二参为 dynUBufSize 非 workspace，取 0；workspace 经 `__kfc_workspace__` 实参传递）。

#### kernel 侧设计

4 个 kernel，同 stream 串行下发：

| kernel | 核型 | 职责 |
|---|---|---|
| `sgemm_gemm_bucket_{nn,nt,tn,tt}` | `__global__ __cube__`（4 转置桶模板实例） | 主 GEMM：块表跨步分配 `blockId = coreIdx + round×blockDim`，单 Matmul 对象多 tile 循环，**每块完整闭环（SetSingleShape + SetTensorA/B + IterateAll + End，官方 SetSingleShape 示例同形态；非对齐尾块依赖每块 End 清理 L0C 残留）** |
| `sgemm_pack_kernel` | `__global__ __vector__` | padded ld 侧紧凑化到 workspace（整列 2D 块拷贝，DataCopyPad，blockCount 分批 ≤2048，A2 上限 4095） |
| `sgemm_epilogue_kernel` | `__global__ __vector__` | `C[:,j] = alpha·D[j,:] + beta·C[:,j]`（beta=0 不读 C；条带宽对齐 8 保证 GM 侧行块对齐） |
| `sgemm_scale_kernel` | `__global__ __vector__` | k=0 / alpha=0 → `C = beta·C`（beta=0 置零免读，NaN 安全） |

**块偏移公式**（官方 `matmul_multi_core_unaligned` CalcOffset 同构）：

```
offA = TRANS_AP ? mIdx·scM : mIdx·scM·Ka          （A' = B 存储行主序视图）
offB = TRANS_BP ? nIdx·scN·Kb : nIdx·scN          （B' = A 存储行主序视图）
offD = mIdx·N·scM + nIdx·scN                      （D = Cᵀ [n,m] 行主序，行距 N=m）
```

尾块以 `SetSingleShape(curM, curN, k)` 设置实际尺寸（curM/curN 为该块剩余行列）；每块 `IterateAll` 后立即 `End()` 再进入下一块（多块循环下非对齐尾块的 L0C/状态残留会污染下一块计算，每块闭环为官方文档示例用法）。

**同步策略**：AIV kernel 间事件用固定静态 ID（每个 (src,dst) 方向独立 ID、同方向 set→wait 严格成对）；无核间同步需求（块表跨步天然无依赖）；每轮末尾回向同步（MTE3_V/MTE3_MTE2）保证 depth=1 单 buffer 复用安全。冗余同步率 0%（审查逐项依赖分析确认）。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas 800I/T A2（DAV_2201，含 Ascend910_9362/910B3） | √ |
| Atlas A3 系列（同 DAV_2201 架构） | √ |

## 算子约束限制

- C 与 A/B 不允许内存重叠（离席计算）；不支持超出 ld 语义的非连续访问；无 stride/broadcast 语义
- Aarray/Barray/Carray 为 Host 侧指针数组，不可传入 Device 侧指针数组
- A3 系列产品自验未覆盖（自验环境 A2/910B3；A3 同架构，任务书 §3.1 允许自验覆盖一种款型）

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | 混合容差 rtol=atol=2⁻¹³；matched_ratio ≥ 0.99；max_abs_error ≤ max(1e-2, 32×ULP)；golden = OpenBLAS cblas_sgemm 逐组列主序；1200 条 CSV 用例全 PASS（含 30 条负向状态码断言） | 任务书 §3.2（生态算子开源精度标准 mixed_tolerance_standard.md，FLOAT32 档） |
| 精度口径补充 | golden 非有限值位置剔除出容差统计（±FLT_MAX 溢出边界的累加顺序差异，float64 复核 NPU 输出更接近真值）；k ≥ 2048 大数规约场景解除逐元素硬帽、以 matched_ratio ≥ 0.99 生态标准门控（解除打印日志可追溯）——均为主仓社区已接受范式（csymm/csyrk/cherk arch22 同款） | 任务书 §2.1.6（确定性计算不要求）+ §3.2 说明段（大数规约可酌情放宽） |
| 性能标准 | 5 条典型 case NPU 平均单次 kernel 耗时（msprof Task Duration，10 次有效采样平均）≤ 达标耗时（GPU 实测 ÷ 0.8） | 任务书 §3.3 |

## 兼容性分析

接口签名与主仓 `include/cann_ops_blas.h` 逐参数一致，与 cuBLAS `cublasSgemmGroupedBatched` 仅 2 处非语义差异（groupCount 参数位置、命名风格）；新增 arch22 目录不影响既有 arch35（950）实现，测试目录按主仓惯例置于 `test/gemm_grouped_batched/sgemm_grouped_batched/arch22/`（family/算子/arch 三层，csyrk 同款），family 层共享文件为 arch35 原有，未改动。
