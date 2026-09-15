# 【CANN社区任务】aclsparseSpMV（950）算子设计文档

## 需求背景（required）

### 需求来源

CANN 社区任务"9月社区任务-aclsparseSpMV算子开发(950)"（适配 Ascend 950PR）。参考 cuSPARSE SpMV 三阶段接口语义，在昇腾 NPU 上基于 Ascend C 编程语言完善 `aclsparseSpMV` 的 arch35（Ascend950PR，A5）实现，并交付 Python/ATen 适配，统一合入 ops-sparse 仓库 master 分支。

### 背景介绍

稀疏矩阵向量乘（SpMV，`Y = alpha * op(A) * X + beta * Y`）是科学计算、图神经网络、推荐系统与大规模语言模型 MoE 层的核心计算原语。大规模稀疏场景下 SpMV 的性能主要受访存模式制约：CSR 格式的列索引随机访问导致 X 向量的非连续读取，行间非零元个数不均衡带来负载不均问题。ops-sparse 仓库现有标准 `aclsparseSpMV` 仅含 arch22（A2）路径，无法在 950PR（arch35，SIMT 编程模型）上运行；同时 arch22 版本尚未覆盖 complex64、Python/ATen 入口与部分属性组合。本任务面向新硬件补齐标准 SpMV 全规格实现，并按任务书要求提供 `torch.mv`/`aten::mv` 的 NPU 适配。

## 需求分析（required）

### 需求描述

使用 Ascend C 编程语言（SIMT 模型）实现 aclsparse 风格的 SpMV 三阶段接口（GetBufferSize / 可选 Preprocess / SpMV），与 cuSPARSE SpMV 核心功能对齐；提供 Python/ATen 适配（`torch.mv(mat, vec)` → `aten::mv`，CSR 布局走 `SparseCsrPrivateUse1` dispatch key）；在 Ascend 950PR 上满足《生态算子开源精度标准》，全部声明 dtype 达到 0.3 倍性能标杆以上，内存满足"额外内存不超过 GPU 内存总量 50%"或"方案固有 workspace ≤ L2 Cache 容量"其一，确定性计算 bit-wise 一致。

### 需求拆解

1. 三阶段接口真实实现：`aclsparseSpMVGetBufferSize` 精确查询、`aclsparseSpMVPreprocess` 可选预处理（pattern 绑定复用）、`aclsparseSpMV` 执行；复用 SpMat/DnVec/Handle 描述符与调用方 stream。
2. 9 组 dtype 组合：INT8→INT32、INT8→FP32、FP16→FP32、BF16→FP32、FP16→FP16、BF16→BF16、FP32→FP32、complex64→complex64、FP32→complex64；computeType 分别为 INT32/FP32/FP32/FP32/FP32/FP32/FP32/complex64/complex64。
3. 操作范围：`opA` = NON_TRANSPOSE / TRANSPOSE / CONJUGATE_TRANSPOSE（实数 H ≡ T，complex64 的 H 在转置物化时对 values 取共轭）。
4. I32 索引，支持 index base 0/1；CSR_ALG1/CSR_ALG2 算法；HOST pointer mode（DEVICE mode 未在本任务声明范围，不误报支持）。
5. complex64 的 alpha/beta 为复数标量；`beta=0` 按接口语义跳过原 Y 读取。
6. Python/ATen 适配：`aten::mv` 注册于 `SparseCsrPrivateUse1`（CSR 专用 key），固定按 alpha=1、beta=0 调用底层；其他 alpha/beta 组合由 aclsparse C++ 路径（测试 hook `torch.ops.ops_sparse_test.spmv_npu`）提供；越界 layout/组合返回明确错误，核心计算不回退 CPU。
7. 确定性：相同输入重复执行结果 bit-wise 一致。
8. 性能：P-01/02/03（Llama 3.1 70B MLP / Qwen3-235B MoE / DeepSeek-V3 MoE 维度锚点）全部声明 dtype 达到 ≥0.3× GPU 标杆。
9. 内存：workspace 由 GetBufferSize 精确查询，Preprocess 与执行复用；不额外持久化稠密矩阵、不泄漏描述符。

## 详细设计（required）

### 算子分析

#### 数学公式

`Y = alpha * op(A) * X + beta * Y`。A 为 M×K CSR 稀疏矩阵（csrRowPtr/csrColInd/csrVal 三数组，I32 索引，base 0/1），X/Y 为稠密向量；NON_TRANSPOSE 时 X 长 K、Y 长 M，TRANSPOSE/CONJUGATE_TRANSPOSE 时相反。complex64 组合按复数乘加：`acc += a*x` 展开为 `accRe += aRe*xRe - aIm*xIm; accIm += aRe*xIm + aIm*xRe`。

#### 支持数据类型

| A/X | computeType | Y |
|---|---|---|
| int8 | int32 | int32 |
| int8 | float32 | float32 |
| float16 | float32 | float32 |
| bfloat16 | float32 | float32 |
| float16 | float32 | float16 |
| bfloat16 | float32 | bfloat16 |
| float32 | float32 | float32 |
| complex64 | complex64 | complex64 |
| float32 | complex64 | complex64 |

#### 支持形状与边界

CSR 任意形状（M、K ≤ INT32_MAX），支持 nnz=0（退化为 `Y = beta*Y`，BetaY 内核）、nnz=1、空行、长尾行（行内并行缺失时由双链 ILP 与 block 倍增调度覆盖）；组合表外的 dtype/layout/算法组合返回 NOT_SUPPORTED。

### 算子实现

#### 实现方案

采用 Ascend C 三层结构：`__simt_vf__` 计算函数（线程级 grid-stride）+ `__global__ __aicore__` Dispatcher（KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY)）+ `kernel_do` 启动器（`<<<blocks, nullptr, stream>>>` 异步 launch），使用调用方 stream。两个内核：

- **SpmvSimtCompute（主内核）**：一行一线程，行内顺序 FMA 累加；按 CompT 三分支——`SpmvCpxF`（complex64，实/虚拆分 FMA）、`double` 标记（f16/bf16 输入组合的 fp32 无误差展开路径，aicore 禁用 fp64 运算故以标记区分）、`float`（常规浮点，DualChain 双链 ILP：行内偶/奇两条独立 FMA+gather 链并行发射，隐藏 X 随机 gather 的 L2 延迟）。`beta=0` 时短路跳过 Y 的 GM 读。INT8→INT32 走 float CompT（8bit 值的乘积与累加在 fp32 内精确，写回截断 int32 逐元素精确）。
- **SpmvBetaYSimtCompute（BetaY 内核）**：nnz=0 时 `Y = beta * Y`，按 Y 存储类型分发（float/half/bfloat16_t/SpmvCpxF/int32_t）。

##### host 侧设计

**tiling 策略**：Host 侧按矩阵规模选择调度策略并写入 SpmvTilingData（m/indexBase/algType/dtype 编码/alpha/beta（含复数分量 alphaIm/betaIm）/rowsPerBlock/dualChain）。统一调度公式 `mult = (m > 16384) ? 6 : 1`（block 倍增，长矩阵提升 X gather 的 L2 吞吐），长尾行场景启用 dualChain。

**dtype 组合校验**：`SpmvValidateCombo` 按上表 9 组合白名单校验，越界返回 NOT_SUPPORTED；`SpmvValidateCommon` 校验 handle/opA 枚举/向量长度/CSR 描述符/index base 与 H 转置合法性。

**转置（T/H）路径**：host 侧 CSC 物化——Pass1 统计每列非零数、Pass2 填充 CSC 三数组（索引按 base 归一后写入，物化结果 tiling.indexBase 强制 0）；H 时对 values 虚部取负。物化结果（workspace）与 matA pattern 绑定缓存（键 = rowPtr/colInd/values 指针 + m/k/nnz/dtype/base/conj，条目持有 pattern 张量强引用以防分配器地址复用导致键假命中，上限 8 条，超限整体换出），pattern 不变时重复调用直接复用，pattern 变化后重新预处理。

**alpha/beta**：Host pointer mode 下同步读取标量并转为 computeType（complex64 组合按 aclsparseComplex 双分量传入 tiling）。`beta=0`（双分量均为 0）时内核跳过 Y 读取。

**workspace**：`GetBufferSize` 按返回的 bufferSize 精确分配；`Preprocess` 与执行复用同一段 buffer；bufferSize 为 0 的场景仍传入非空 1 字节缓冲（与 C++ 测试套调用约定一致，避免实现层以空指针区分路径）。

##### Python/ATen 适配设计

- **ATen 注册**：`aten::mv` 以 `SparseCsrPrivateUse1` key（CSR 布局专用 dispatch key）注册 NPU 实现；`torch.mv(mat, vec)` 对 CSR mat 命中该 key，无 CPU fallback。
- **参数处理**：crow/colInd 支持 int64（torch 默认）→ I32 窄化；values/x uniform dtype；固定 alpha=1、beta=0；输出按 M 维构造稠密张量；layout 非 CSR / dtype 组合越界返回明确错误。
- **全功能入口**：`torch.ops.ops_sparse_test.spmv_npu(row_ptr, col_ind, values, x, y, alpha, beta, trans, base)`（torch.library 定义，Library 模块级持有防 GC 注销；c10::Scalar caster NYI 故 Python 侧拆分为实/虚 double）承载非 uniform 组合、N/T/H、base 0/1 与任意 alpha/beta，供测试包 hook 与 ATen 路径共同复用同一 C++ 实现。
- **描述符生命周期**：SpMat/DnVec RAII Guard；vecY 描述符 dtype 必须携带 Y 的真实存储 dtype（如 FP16→FP16 组合用 Half 而非 computeType FP32），否则内核按 4 字节写 2 字节缓冲造成越界；执行后 RecordTensors 保活至 stream 同步。

##### 精度设计

- 实数浮点组合：fp32 顺序 FMA 累加（官方容差 rtol=2^-10≈9.77e-4，n·eps 远低于容差线）；f16/bf16 输入组合采用 **fp32 无误差展开**（输入为精确半精度值 → fp32 乘积精确 → TwoSum L1/L2 级联 → alpha 积 FMA 校正拆分 (a1,a2) → 尾项合并后单次舍入），与 float64 golden 经 float 中转的半精度路径**逐位一致**；aicore 禁用 fp64 运算，故以 double 模板参数作路径标记而非真实 double 计算。
- complex64：实/虚拆分复数 FMA 顺序累加，实部虚部分别满足 float32 混合容差。
- INT8→INT32：整数精确（逐元素 exact match）。

### 测试设计

- C++ UT（`test/spmv/arch35/spmv_test.cpp`）：34 个定向用例，覆盖 9 组合 × ALG1/ALG2 × HOST/DEVICE 指针模式 × base 0/1 × N/T/H × alpha/beta 标量 × 空行/零 nnz，输出逐用例 PASS/FAIL 与 maxRel。
- ATen/Python 端到端 UT（`test/spmv/arch35/test_spmv_aten.py`）：13 个用例，覆盖 torch.mv 公开入口、hook 全组合、in-place Y、bit-wise 确定性、非法输入异常、NPU Dispatch 断言；CPU Golden 按任务书 3.2.2 口径（INT8→INT32 整数精确、f16/bf16 用 float32、float32 用 float64、complex64 用 complex128），容差按 3.2.4。
- 官方测试包：200 泛化精度用例（10 种行分布 pattern × alpha/beta 组合）、96 个 P-01/02/03 性能场景（预热 10 次、采样 30 次，报告 median 与 p90）、内存采集与对比（`compare_sparse_ops_memory.py`）。
- Profiler 证据：torch_npu profiler 的 AI Core Computing 时间线 + cpu_op 树无 BLAS 回退断言 + `dispatch_has_kernel_for_dispatch_key('aten::mv','SparseCsrPrivateUse1')=True`（`test/spmv/arch35/evidence_e2e.py`）。

## 支持硬件（required）

Ascend 950PR（A5，DAV_3510 / arch35）。

## 约束（required）

1. 仅支持 CSR 格式与 I32 索引；CSC/COO/SELL 等格式返回 NOT_SUPPORTED。行内列索引要求有序且不重复（与 cuSPARSE 契约一致），重复坐标行为未定义并在 README 声明。
2. pointer mode 仅支持 HOST（与现有 arch22 基线一致）；DEVICE mode 未实现、不误报支持。
3. TRANSPOSE 路径的 CSC 物化 workspace 与 pattern 绑定缓存（上限 8 条），调用方需保证 pattern 张量生命周期覆盖后续调用；pattern 变化后自动重新预处理。
4. 超长行（单行 nnz 数万级且高度偏斜）场景性能弱于 GPU 标杆（行内并行缺失），已在自测报告的 extra 场景中如实标注。
5. `aten::mv` 路径固定 alpha=1、beta=0（与 torch 语义一致）；任意 alpha/beta 经 aclsparse C++ 接口使用。

## 可维可测（required）

1. 全部 tiling 参数与调度公式为确定性纯函数（无环境变量开关），同一输入重复执行 bit-wise 一致。
2. C++ UT / ATen UT / 官方精度与性能包均可独立复现，复现命令见 `test/spmv/arch35/README.md`。
3. 错误码：参数校验失败返回对应 ACL_SPARSE_STATUS 错误码；不支持的组合返回 NOT_SUPPORTED。
4. Profiler/Dispatch 证据与端到端耗时可通过 `evidence_e2e.py` 一键采集。

## 兼容性（required）

1. A2/A3（arch22）与 A5（arch35）公共 Host 逻辑解耦：arch35 实现位于 `sparse/spmv/arch35/` 独立目录，标准 `aclsparseSpMV` 按 SOC 分发，不影响 arch22 既有路径；测试亦按 SOC 自动选择 `test/spmv/arch35/` 或 `arch22/`。
2. 标准 `aclsparseSpMV` 与 `aclsparseSpMVOp` 为不同接口，不混用。
3. 接口声明沿用 `include/cann_ops_sparse.h` 既有签名，不重复新增同名接口。
4. Python/ATen 适配 PyTorch 2.7+ / torch_npu 26.0.0+（自测环境：torch 2.12.0 + torch_npu 2.12.0）。
