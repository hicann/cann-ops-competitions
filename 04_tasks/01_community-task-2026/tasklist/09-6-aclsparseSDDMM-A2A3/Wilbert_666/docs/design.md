# aclsparseSDDMM 算子设计文档（Atlas A2/A3，arch22）

- 团队：Wilbert_666
- 任务：《9月社区任务-aclsparseSDDMM算子开发(A2A3)》
- 目标仓：https://gitcode.com/cann/ops-sparse ，个人仓验收分支：https://gitcode.com/Wilbert_666/ops-sparse/tree/feat/aclsparse-sddmm-a2a3-full
- 设计模板：`04_tasks/01_community-task-2026/resources/design_template.md`

# 需求背景（required）

## 需求来源

本文对应 2026 年 9 月社区任务《aclsparseSDDMM 算子开发任务书（A2/A3）》。目标是在 Atlas A2 训练系列（910B3/910B4，`ascend910b`）与 Atlas A3 系列（`ascend910_93`，DAV_2201）上完善 `aclsparseSDDMM` 全链路，保持与 cuSPARSE SDDMM 对应的三阶段接口语义，并完成 PyTorch 2.7+ / torch_npu 的 `torch.sparse.sampled_addmm` / `aten::sparse_sampled_addmm` NPU 适配，核心计算不得 CPU fallback。

设计依据：

1. 任务书 `aclsparseSDDMM_A2A3_task_doc.md`；
2. 任务测试包 `test_cases/aclsparseSDDMM_testCase/`（P-01/P-02/P-03、GPU 标杆、采集脚本）；
3. 社区设计模板与流程 README；
4. 仓库既有 `sparse/sddmm/` 三阶段 ABI 与 arch22 工程范式。

## 背景介绍

### SDDMM 功能

SDDMM（Sampled Dense-Dense Matrix Multiplication）只计算稠密矩阵乘积在稀疏矩阵既有 pattern 上的值：

$$
C_{out} = (\alpha \cdot op(X) \cdot op(Y) + \beta \cdot C_{in}) \circ spy(C)
$$

`spy(C)` 表示只保留 C 已存在的 CSR 非零位置或 BSR 非零块内位置。算子不生成新的稀疏结构，row offsets、column indices 和 BSR block pattern 在执行前后保持不变，只原地更新 values。

对 CSR 中第 `i` 行、第 `j` 列的一个既有非零位置：

$$
C_{ij} \leftarrow \alpha \sum_{p=0}^{K-1} op(X)_{ip}op(Y)_{pj} + \beta C_{ij}
$$

对 BSR 中的每个非零块，先由 block row、block column 和块内坐标恢复逻辑 `(i,j)`，再执行同一公式。`TRANSPOSE` 只交换维度和访问坐标；complex64 的 `TRANSPOSE` 不执行共轭。

### ops-sparse 现状与本任务增量

既有仓库已提供 `aclsparseSDDMMBufferSize` / `aclsparseSDDMMPreprocess` / `aclsparseSDDMM` 三阶段接口。本任务在 **arch22** 上补齐：

| 层级 | 基线缺口 | 本任务交付 |
| --- | --- | --- |
| 格式 | 以 CSR 为主 | CSR + 方块 BSR |
| dtype | FP16/FP32 同类型 | CSR：FP32、complex64、FP16→FP16/FP32；BSR 另含 BF16→BF16/FP32 |
| index | 多为 base 0 | I32，base 0/1 |
| batch | 描述符字段不完整 | DnMat/BSR strided-batch **API 齐全**；Execute 要求 X/Y/C `batchCount` 相同（见约束） |
| 性能 | 通用 AIV 路径偏慢 | P-case 走 aclnn BatchMatMul 快路径 |
| Python | 无 ATen NPU 适配 | `aten::sparse_sampled_addmm` @ SparseCsrPrivateUse1，无 CPU fallback |

# 需求分析（required）

## 需求描述

在不改变既有 SDDMM 三阶段 ABI 的前提下，于 Atlas A2/A3（arch22）交付：

1. Host 校验、Preprocess 状态绑定 matC、Execute 调度；
2. Ascend C 通用 Kernel + aclnn BatchMatMul 快路径；
3. BSR 描述符与 DnMat/BSR strided-batch 读写接口；
4. C++ UT 与 Python/ATen 端到端 UT；
5. P-01/P-02/P-03 性能与 L2 workspace 验收。

## 算子原型

```cpp
aclsparseStatus_t aclsparseSDDMMBufferSize(
    aclsparseHandle_t handle, aclsparseOperation_t opX, aclsparseOperation_t opY,
    const void *alpha, aclsparseConstDnMatDescr_t matX,
    aclsparseConstDnMatDescr_t matY, const void *beta,
    aclsparseSpMatDescr_t matC, aclDataType computeType,
    aclsparseSDDMMAlg_t alg, size_t *size);

aclsparseStatus_t aclsparseSDDMMPreprocess(
    aclsparseHandle_t handle, aclsparseOperation_t opX, aclsparseOperation_t opY,
    const void *alpha, aclsparseConstDnMatDescr_t matX,
    aclsparseConstDnMatDescr_t matY, const void *beta,
    aclsparseSpMatDescr_t matC, aclDataType computeType,
    aclsparseSDDMMAlg_t alg, void *buffer);

aclsparseStatus_t aclsparseSDDMM(
    aclsparseHandle_t handle, aclsparseOperation_t opX, aclsparseOperation_t opY,
    const void *alpha, aclsparseConstDnMatDescr_t matX,
    aclsparseConstDnMatDescr_t matY, const void *beta,
    aclsparseSpMatDescr_t matC, aclDataType computeType,
    aclsparseSDDMMAlg_t alg, void *buffer);
```

本任务同时提供 `aclsparseCreateBsr` / `aclsparseCreateConstBsr`、`aclsparseDnMatGetStridedBatch` / `aclsparseDnMatSetStridedBatch`、`aclsparseBsrSetStridedBatch`。

### 参数说明

| 参数名 | 输入/输出/属性 | 是否必选 | 描述 | 数据类型/格式 | 约束 |
| --- | --- | --- | --- | --- | --- |
| `handle` | 输入 | 必选 | 保存调用方 stream 的 aclsparse handle | Handle | 非空且 stream 有效 |
| `opX/opY` | 属性 | 必选 | 稠密矩阵变换 | NON_TRANSPOSE/TRANSPOSE | 不支持共轭转置 |
| `alpha/beta` | 输入 | 必选 | 缩放标量 | FP16/FP32 或 complex64 | 由 computeType 决定；Host/Device pointer mode |
| `matX/matY` | 输入 | 必选 | 稠密乘法输入 | FP16/BF16/FP32/complex64，ROW/COL | op 后 shape 可相乘；ld 合法 |
| `matC` | 输入输出 | 必选 | 固定稀疏 pattern，values 原地更新 | CSR/BSR，I32，base 0/1 | 结构只读 |
| `computeType` | 属性 | 必选 | 累加与标量类型 | FP16/FP32/complex64 | 与支持矩阵匹配 |
| `alg` | 属性 | 必选 | 算法 | DEFAULT | 其他值 NOT_SUPPORTED |
| `batchCount` | 属性 | batch API 必选 | DnMat/BSR batch 数 | int | **1..65535，且 matX/matY/matC 三者必须相同** |

## 需求拆解

1. 扩展 SpMat/DnMat 描述符：BSR block 元数据、batch stride、SDDMM preprocess 绑定。
2. Host 统一校验 format、dtype、shape、ld、base、block size、**相等 batchCount**、溢出。
3. BufferSize 返回精确 workspace；Preprocess 绑定 pattern 到 matC，禁止以 workspace 指针为 key 的进程全局无锁 cache。
4. Execute：P-case 均匀连续列走 BatchMatMul 快路径；其余走 Ascend C Kernel。
5. Python/ATen：CSR 走 `aten::sparse_sampled_addmm`；BSR/complex64 走扩展 Op；不支持组合显式报错。

# 详细设计（required）

## 算子分析

### 数学公式

$$
C_{out} = (\alpha \cdot op(X) \cdot op(Y) + \beta \cdot C_{in}) \circ spy(C)
$$

### 支持数据类型

| 稀疏格式 | X/Y dtype | C values dtype | computeType | 累加 |
| --- | --- | --- | --- | --- |
| CSR/BSR | FLOAT32 | FLOAT32 | FLOAT32 | FLOAT32 |
| CSR/BSR | COMPLEX64 | COMPLEX64 | COMPLEX64 | 两路 FLOAT32 复数乘加 |
| CSR/BSR | FLOAT16 | FLOAT16 | FLOAT16 或 FLOAT32 | FP32 累加后写回 |
| CSR/BSR | FLOAT16 | FLOAT32 | FLOAT32 | FLOAT32 |
| BSR | BFLOAT16 | BFLOAT16 | FLOAT32 | FLOAT32，写回转 BF16 |
| BSR | BFLOAT16 | FLOAT32 | FLOAT32 | FLOAT32 |

### 支持形状

- C 逻辑 `[M,N]` 必须等于 `op(X)×op(Y)` 的乘积形状。
- BSR：`rowBlockSize == colBlockSize ∈ {2,4,8,16,32,64,128}`，`M/N` 为 block 数与块边长之积。
- batch：`batchCount ∈ [1,65535]`，**X/Y/C 必须相等**（见约束限制）。P 锚点均为 `batchCount=1`。

## 算子实现

### 总体流程

```text
BufferSize
  → 校验 handle/描述符/op/dtype/shape/ld/base/block/batchCount 相等
  → 返回对齐后 workspace 字节数
Preprocess
  → 检测 fixed_col_run / fixed_block_run（每行或每 block-row 均匀 nnz + 连续列）
  → 打包 reorderB / A-B 面板；缓存 aclnn BMM executor
  → 将 pattern signature 绑定到 matC
Execute
  → 校验 workspace 与 matC 绑定
  → CSR P-case FP16/FP32：aclnn BatchMatMul 直写 values
  → BSR P-case：block BatchMatMul；complex64 为 4× half BMM + add/sub + cast
  → 其余 pattern：Ascend C AIV/AIC Kernel
```

### Host 侧设计

实现目录：`sparse/sddmm/arch22/`（`sddmm_host.cpp`、`sddmm_aclnn_bmm_host.cpp`）。

1. **参数层**：handle、stream、alpha/beta、alg、描述符非空。
2. **dtype 层**：仅接受上表组合；拒绝 CSR BF16 与未声明组合。
3. **shape/存储层**：op 后可乘；DnMat order/ld；CSR/BSR I32、base 0/1、方块 BSR。
4. **batch 层**：三者 `batchCount` 必须相同；stride 以元素计，0 表示按单矩阵存储跨度推导。不相等返回 `ACL_SPARSE_STATUS_INVALID_VALUE`。
5. **生命周期**：Preprocess 状态绑定 matC；pattern 指针或描述符属性变化后必须重新 Preprocess。不同 matC 可并发；同一 matC 由调用方串行。
6. **调度**：`cubeMathType` 在 910B 用 4、910_93 用 0；Launch 后不得复用已消费的 BMM executor。

### Kernel / 快路径设计

| 路径 | 触发条件 | 实现 |
| --- | --- | --- |
| CSR BMM | fixed_col_run + FP16/FP32 | aclnn BatchMatMul，形状 `(m,1,K)@(m,K,n)` |
| BSR BMM | fixed_block_run + FP16/FP32/BF16 | block BatchMatMul，形状 `(blockNnz,bs,K)@(blockNnz,K,bs)` |
| BSR complex | fixed_block_run + complex64 | 4× half BMM + 复数组合 |
| 通用 CSR/BSR | 其余 pattern | Ascend C：行重排 + 贪心分箱 + AIV 点积；泛化路径按 value 并行恢复坐标 |

通用 Kernel 对 value index 用 row offsets 二分得到行（BSR 再拆块内坐标），column index 减去 index base，按 op/order/ld/batchStride 读 X/Y，在 FP32 或 complex64 中累加后写回同一 value 位置。

### Python/ATen 适配

| 层 | 入口 | 说明 |
| --- | --- | --- |
| 公开 API | `torch.sparse.sampled_addmm` | CSR only，PyTorch 2.7+ |
| ATen | `aten::sparse_sampled_addmm` @ SparseCsrPrivateUse1 | Python 注册，内部调 `sddmm_npu` |
| 扩展 Op | `torch.ops.ops_sparse_test.sddmm_npu` | CSR/BSR 全 dtype；Execute 用 aclrtEvent 计时 |
| Session | LRU=2（`SDDMM_MAX_SESSIONS`） | Key 含 dtype/fmt/shape 与 matX/matY/C 指针，避免 allocator 复用导致误命中 |

不支持的 layout/device/dtype 通过 `TORCH_CHECK` 显式失败，不回退 CPU。complex64 的 ATen 公开入口因 torch_npu CSR complex 限制在 UT 中 skip，计算由 C++ UT 与 `sddmm_npu` 覆盖。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 | 实测说明 |
| --- | --- | --- |
| Atlas A2 训练系列产品（910B3） | √ | 已测：`--soc=ascend910b`，CANN 9.1.0-beta.3 |
| Atlas A2 训练系列产品（910B4） | √ | **代码路径与 910B3 相同（均 `ascend910b`/arch22）**；当前无独立 910B4 机器，未单独跑数 |
| Atlas A3 系列产品（ascend910_93，DAV_2201） | √ | 已测：NPU 9382 / Board 0xb1，CANN 9.1.0-beta.1 |

## 支持软件版本

| 软件 | 版本 |
| --- | --- |
| CANN | 9.1.0 及后续配套版本（实测 A3：9.1.0-beta.1；A2：9.1.0-beta.3） |
| PyTorch | 2.7 及以上（实测 2.12.0） |
| torch_npu | 26.0.0 及之后（实测 A3：2.12.0；A2：2.12.0.rc1） |

## 算子约束限制

- opX/opY 仅支持 NON_TRANSPOSE、TRANSPOSE，不支持共轭转置。
- matC 仅支持 CSR 和方块 BSR；索引仅 I32；index base 支持 0/1。
- CSR 支持 FP32、complex64、FP16→FP16/FP32；BSR 额外支持 BF16→BF16/FP32。CSR 不支持 BF16。
- BSR block size 仅 2/4/8/16/32/64/128；逻辑 M/N 必须等于 block 数 × 块边长。
- **strided-batch 限制（相对任务书 §2.4 的明确收窄）**：
  - DnMat / BSR 描述符 API 支持设置 `batchCount ∈ [1, 65535]` 与 stride。
  - **Execute 仅支持 `matX.batchCount == matY.batchCount == matC.batchCount`**，即任务书四类组合中的 **`C_i = (A_i × B_i) ∘ C_i`**（三者 batch 全相等，含全部为 1 的单 batch）。
  - **不支持** `C_i=(A×B)∘C_i`、`C_i=(A_i×B)∘C_i`、`C_i=(A×B_i)∘C_i`（即 X 或 Y 以 `batchCount=1` 向 C 广播）。
  - 不相等时 Host 返回 `ACL_SPARSE_STATUS_INVALID_VALUE`（`sddmm_host.cpp`：`batchCount mismatch across X/Y/C`）。
- alpha/beta 支持 Host/Device pointer mode；DEVICE 模式在 Host 调度时同步读取标量。
- 输入 X/Y 与 C 结构只读；C values 原地更新。
- 须按 BufferSize 分配 workspace 并先 Preprocess；pattern 指针/内容或 batch 属性变化后必须重新 Preprocess。
- ATen 公开入口仅 CSR；BSR / complex64 使用 `sddmm_npu` 扩展 Op。
- 910B4 与 910B3 共用 arch22 二进制；无 910B4 节点时不声称已在该型号实测。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| FP16 | CPU Golden 用 FP32；rtol/atol=`2^-9`，匹配率 ≥0.99，绝对误差硬上限 `max(1e-1, 32×ULP)` | 任务书 §3.2 |
| BF16 | CPU Golden 用 FP32；rtol/atol=`2^-6`，匹配率 ≥0.99，硬上限 `max(1, 32×ULP)` | 任务书 §3.2 |
| FP32 | CPU Golden 用 FP64；rtol=`2^-10`、atol=`2^-16` | 任务书 §3.2 |
| complex64 | CPU Golden 用 complex128；实部、虚部分别按 FP32 验收 | 任务书 §3.2 |
| CSR 性能 | P-01/P-02/P-03 Execute median ≥ 对应 GPU Event 标杆的 0.25 倍 | 任务书 §3.3 |
| BSR 性能 | 无 cuSPARSE BSR 等价接口，按 §3.4 workspace ≤ 目标硬件 L2（192 MB） | 任务书 §3.3/§3.4 |
| 内存 | 有 GPU 等价接口且 IO>500MB 时额外峰值 ≤ GPU 的 50%；否则 workspace ≤ L2 | 任务书 §3.4 |

### 实测摘要（验收分支 `feat/aclsparse-sddmm-a2a3-full`）

| 项 | A3（ascend910_93） | A2 910B3（ascend910b） |
| --- | --- | --- |
| C++ UT | 67/67 PASS | 67/67 PASS |
| Python UT | 13 passed / 1 skipped | 13 passed / 1 skipped |
| CSR 18 case | 18/18 ≥0.25（FP16 约 0.33–0.35） | 18/18 ≥0.25（FP16 约 0.25–0.27；P-03 base0 全量 sweep 偶发 0.245，孤立重跑与 C++ Execute ≥0.25） |
| BSR / 内存 | L2 42/42 PASS，最大 workspace ≈132 MB | 同左 |
| Profiler | P-02 FP16，`dispatch_verified`，无 CPU fallback | 同左 |

910B4：无独立硬件，书面说明见个人仓 `docs/zh/self_test_report_aclsparseSDDMM_A2.md`。

## 兼容性分析

- 公开三阶段函数签名不变；BSR/batch API 为向后兼容扩展。
- Preprocess 状态绑定 matC，不使用进程全局无锁 pattern cache。
- A2/A3 共用 `sparse/sddmm/arch22/`；与 arch35 通过平台编译隔离，互不污染 Kernel ABI。
- **batch 广播语义相对任务书 §2.4 收窄**（仅相等 `batchCount`），属本实现的明确限制，调用方不得假设 X 或 Y 可单独广播。
