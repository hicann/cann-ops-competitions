# aclblasSspr 算子设计文档

| 项目 | 内容 |
|---|---|
| 任务名称 | aclblasSspr A2/A3 算子开发 |
| taskId | 9d94810ec99446f09e9361fb47ddf758 |
| 适配硬件 | Atlas A2 / Atlas A3 系列产品（arch22） |
| CANN 版本 | CANN 9.1.0 |
| 涉及仓库 | cann/ops-blas（blas/spr/arch22/、include/cann_ops_blas.h、test/spr/sspr/） |

# 需求背景

## 需求来源

昇腾社区任务广场《aclblasSspr A2/A3 算子开发》。要求参考 cuBLAS `cublasSspr` 的功能与参数语义，基于 ops-blas 开源仓工程框架，使用 Ascend C 在 Atlas A2/A3 上开发单精度实数对称矩阵秩-1 更新（打包存储）算子 `aclblasSspr`，完成设计、开发、测试全流程。

## 背景介绍

### aclblasSspr 算子简介

BLAS 中的对称矩阵秩-1 更新（Symmetric Packed Rank-1 Update，sspr）计算 `A = alpha * x * x^T + A`，其中 A 为 n×n 对称实数矩阵，以 packed（压缩）格式存于长度 n(n+1)/2 的一维数组 AP 中，无前导维 lda。x 为 n 元素单精度实数向量，alpha 为标量。`aclblasSspr` 为其单精度实数（float32）版本，语义对齐 cuBLAS `cublasSspr` 与 Netlib `sspr`。

ops-blas 仓中同族全存储版本 `aclblasSsyr` 已有 arch22 实现，arch35 已有 `aclblasSspr` 实现（SIMT 模型），arch22 尚无 `aclblasSspr` 声明与实现。

### 接口定义

```cpp
aclblasStatus_t aclblasSspr(
    aclblasHandle_t handle, aclblasFillMode_t uplo, int n,
    const float* alpha, const float* x, int incx, float* AP);
```

与 cuBLAS `cublasSspr` 逐参数对齐（无 lda、无 beta，AP 为打包存储一维数组）。

# 需求分析

## 需求描述

在 Atlas A2/A3（910B）上通过 handle 绑定 stream 直调 Ascend C kernel，实现 `aclblasSspr`：

1. 计算 `A = alpha * x * x^T + A`，原地覆写 AP；A 为 n×n 对称实数矩阵，packed 列优先存储：
   - `uplo = ACLBLAS_UPPER`：第 j 列自对角 A(j,j) 起向上至 A(0,j)，存于 AP[j*(j+1)/2 + i]（i ≤ j）；
   - `uplo = ACLBLAS_LOWER`：第 j 列自对角 A(j,j) 起向下至 A(n-1,j)，存于 AP[j*(2*n-j+1)/2 + (i-j)]（i ≥ j）。
2. `incx` 支持正/负步长（负步长按 Netlib 语义从向量尾部反向遍历）；`incx = 0` 非法；`n = 0` 为合法 no-op；`alpha = 0` 为合法 no-op（不引用 x、AP 不变）。
3. 精度满足生态算子开源精度标准（FLOAT32：rtol 2⁻¹⁰、atol 2⁻¹⁶、matched_ratio ≥ 0.99、max_abs_error ≤ 1e-2 或 32×ULP）。
4. 性能不低于任务书 §3.3 标杆（n=512 UPPER 11.31us；n=1024 LOWER 27.07us；n=2048 UPPER 79.93us；n=4096 LOWER 475.91us，Atlas 800T A2 (910B3)，warmup 后有效采样 >50 次取平均）。

## 需求拆解

1. **接口与参数校验**：`include/cann_ops_blas.h` 已有声明；handle 为 nullptr 返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`；n < 0、incx = 0、alpha/x/AP 为 nullptr 返回 `ACLBLAS_STATUS_INVALID_VALUE`；uplo 非法返回 `ACLBLAS_STATUS_INVALID_VALUE`；n = 0 或 alpha = 0 直接返回 `ACLBLAS_STATUS_SUCCESS`。
2. **packed 布局与三角引用**：仅引用/更新 uplo 指定三角；packed 列内连续存储（stride=1）。
3. **kernel 设计**：外积更新的向量化（Muls + Add）、多核列交错负载均衡、大列分块处理、incx=1 连续 DataCopy / incx≠1 逐元素 GetValue。
4. **测试工程**：新建 `test/spr/sspr/`，CSV 驱动 GTest，golden 由 cblas（Netlib `cblas_sspr`）生成；覆盖边界/负向/精度/步长用例。

# 算子分析

计算公式：`A := alpha * x * x^T + A`。对 AP 中第 j 列、第 i 行元素：

- UPPER（i ≤ j）：`AP[j*(j+1)/2 + i] += alpha * x[j] * x[i]`
- LOWER（i ≥ j）：`AP[j*(2*n-j+1)/2 + (i-j)] += alpha * x[j] * x[i]`

关键观察（决定实现方案）：

- **packed 布局中列连续**：无论 upper/lower，A 的每一列在 AP 中连续存放（stride=1），适合 DataCopy 批量搬运。
- **外积更新无归约**：每个 AP 元素只被一个 (i,j) 对更新，无需 ReduceSum，比 tpmv 简单。
- **列间无数据依赖**：不同列的 AP 区域互不重叠，多核可并行处理不同列，无需原子操作。
- **每列内 x[j] 为标量常数**：可提取到内循环外，用 Muls（标量乘向量）一次性处理整列。

# Host/Tiling 设计

- **参数校验顺序**：handle → n<0 → n==0（no-op）→ alpha/uplo/incx/x/AP 校验 → alpha==0（no-op），与任务书 §2.4 异常行为逐项对应。
- **tiling**：`useCoreNum = min(n, numBlocks)`，numBlocks=8（启动 8 个 AIV 核）；n/uplo/alpha/incx/useCoreNum 以值传递给 kernel（`SsprTilingData` 结构体）。
- **kernel 启动**：`sspr_kernel<<<numBlocks, nullptr, stream>>>(x, ap, tiling)`，异步提交到 handle 绑定的 stream。

# Kernel 设计

整体为**单 kernel 发射**，类 `SsprAIV<float>` 封装，TPipe/TQue/TBuf 高层 Ascend C API：

1. **Init**：设置 GM buffer（xGM 物理大小 = 1+(n-1)*|incx|，apGM 大小 = n*(n+1)/2）；初始化 3 个 TQue（xQueue/apQueue VECIN，outQueue VECOUT，BUFFER_NUM=2 双缓冲）+ 1 个 TBuf（tmpBuf 用于 Muls 中间结果）；maxDataCount = 2048（8KB/buffer，3×2×2048×4 = 48KB UB，余量充足）。
2. **Process**：核心 vecIdx 在 [0, useCoreNum) 内，**交错列分配** `col = vecIdx, vecIdx + useCoreNum, vecIdx + 2*useCoreNum, ...`，实现负载均衡（UPPER 列长递增、LOWER 列长递减，交错分配使各核工作量均匀）。
3. **ProcessColumn(col)**：
   - 读取标量 `xCol = xGM[XPos(col)]`，计算 `axCol = alpha * xCol`；若 axCol==0 跳过该列。
   - 计算列基址 `colBase`（UPPER: col*(col+1)/2，LOWER: col*(2*n-col+1)/2）。
   - 确定行范围（UPPER: [0, col]，LOWER: [col, n)）。
   - **分块处理**：列长 > maxDataCount 时分多个 chunk，每 chunk 处理 maxDataCount 个元素。
   - 每 chunk 流水：CopyIn（LoadX + LoadAP）→ Compute（Muls + Add）→ CopyOut。
     - **LoadX**：incx==1 用 DataCopy/DataCopyPad 连续搬运；incx≠1 逐元素 GetValue（xGM.GetValue(XPos(start+i))）。
     - **LoadAP**：DataCopy/DataCopyPad（AP 列内始终连续，stride=1）。
     - **Compute**：`Muls(tmp, xLocal, axCol, count)` → `Add(out, apLocal, tmp, count)`，即 `AP += axCol * x`。MTE2→V 事件同步。
     - **CopyOut**：DataCopy/DataCopyPad 写回 GM。

**关键设计决策**：
- maxDataCount=2048：实测 maxDataCount=7680 时 n=4096 用例失败（DataCopy 大 count 触发硬件边界），2048 安全且 UB 余量充足。
- 无原子操作：packed 存储中每个 AP 位置唯一属于一列，多核处理不同列无竞争。
- 无跨核屏障：列间完全独立，无需同步。

# 硬件与约束

- **适配硬件**：Atlas A2/A3（910B3），arch22（dav-2201），CANN 9.1.0。
- **UB 内存**：3 TQue × 2 buffer × 2048 × 4B = 48KB + TBuf 8KB = 56KB，远低于 256KB UB 上限。
- **DataCopy 约束**：单次 DataCopy count ≤ 2048（实测 4096 在特定场景触发硬件边界），非 8 对齐 count 用 DataCopyPad。
- **GM 访问**：x 向量按 incx 步长访问（incx=1 连续，incx≠1 逐元素）；AP 列内连续（stride=1）。
- **核数**：numBlocks=8，useCoreNum=min(n, 8)，交错列分配。

# 可维可测

- **测试工程**：`test/spr/sspr/`（CMakeLists + sspr_param.h + sspr_golden.h + arch22/sspr_test.cpp + sspr_npu_wrapper.h + sspr_test.csv）。
- **golden**：由 cblas（Netlib `cblas_sspr`）生成，CPU 侧 `aclblasSspr_cpu` 实现。
- **测试用例**：54 例，覆盖 n=0/1/2/3/4/7/64/100/127/128/255/256/257/511/512/1024/2048/4096，uplo=UPPER/LOWER，incx=±1/±2/±3，alpha=1.0/-1.0/0.5/-2.0/0.0，负向用例（n<0、incx=0、incx=INT_MIN、nullptr、非法枚举）。
- **精度结果**：54/54 PASS（hidevlab A2 910B3 实测），maxAbsErr ≤ 4.77e-7，matchedRatio = 1.0。
- **复现**：`bash build.sh --soc=ascend910b3 --ops=sspr && ./build/test/spr/sspr/sspr_test`。