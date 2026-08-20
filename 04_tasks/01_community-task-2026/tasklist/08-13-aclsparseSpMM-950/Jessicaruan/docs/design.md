# 【社区任务】aclsparseSpMM（950）算子设计文档

> 参赛账号：Jessicaruan（@gcw_DmEcQNNx）
> 任务：8月社区任务 - aclsparseSpMM 算子开发(950)
> 任务书：https://www.hiascend.com/activities/task-center/details/c62841e450064e198c34599c2b31bee5
> 活动页：https://www.hiascend.com/activities/task-center/details/c62841e450064e198c34599c2b31bee5
> 合入仓：https://gitcode.com/cann/ops-sparse （`master`，沿用现有 `aclsparseSpMM*`，不新增同名接口）
> 硬件：Ascend 950PR / 950DT（arch35）

# 一、需求背景（required）

## 1.1 需求来源

本需求来源于 CANN 社区任务「8月社区任务-aclsparseSpMM 算子开发(950)」。要求参考 PyTorch 2.7+ 的 `torch.sparse.addmm` / `aten::_sparse_addmm`，在 950 上完成 Python/ATen 适配，并**复用、扩展** `ops-sparse` 已有 aclsparse SpMM C++ 接口及 Ascend C Kernel。

- 公式：`out = β · input + α · (mat1 × mat2)`，`mat1` 为 CSR 稀疏矩阵，`mat2`/`input`/`out` 为稠密矩阵
- 必选 dtype：`float16` / `bfloat16` / `float32` / `complex64`（complex64 不是选做）
- 禁止 CPU fallback 代替 NPU 计算
- 与 A2/A3 同题并行：公共 Host 逻辑必须和硬件差异解耦，保证同一主干可共存

## 1.2 背景介绍

### 1.2.1 功能说明

Python 公开接口：

```python
torch.sparse.addmm(input, mat1, mat2, *, beta=1, alpha=1) -> Tensor
```

ATen Schema：

```text
aten::_sparse_addmm(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1) -> Tensor
```

C++ 三步流程（已有，必须保持源码兼容）：

1. `aclsparseSpMMGetBufferSize`
2. `aclsparseSpMMPreprocess`（CSR 行重排 + 分桶）
3. `aclsparseSpMM`

### 1.2.2 现有实现现状（ops-sparse `sparse/spmm/arch35`）

仓库已有 950 路径，且 README 写明 **950 支持、A2/A3 不支持**。当前能力与任务书差距如下：

| 维度 | 现有 arch35 | 本任务要求 |
| --- | --- | --- |
| 产品 | 950PR/DT | 950PR（本任务） |
| 计算 dtype | `ACL_FLOAT` / `ACL_INT32` | `ACL_FLOAT16` / `ACL_BF16` / `ACL_FLOAT` / `ACL_COMPLEX64` |
| Kernel 分发 | `SPMM_DTYPE_FP32/FP16/INT8` | 增 `BF16`、`COMPLEX64`；INT8 保留但不作为本任务验收项 |
| 组合 | fp32；fp16 compute fp32；int8→int32 | 四类实数/复数全链路；组合对齐 cuSPARSE 13.3 SpMM 官方表 |
| 索引基址 | 仅 `INDEX_BASE_ZERO` | 0 和 1 |
| opA | 仅 NON_TRANSPOSE | 按 cuSPARSE 表；CSR_ALG3 仅 NON_TRANSPOSE，且不支持 opB 共轭转置 |
| opB | NON_TRANSPOSE / TRANSPOSE，无共轭 | 声明支持的转置/共轭转置；未声明组合明确报错 |
| 算法 | DEFAULT / CSR_ALG1 / FP32_HIGH_PRECISION | CSR_ALG1 偏列主、CSR_ALG2 偏行主、CSR_ALG3 受限 CSR |
| TilingData 标量 | `float alpha_host/beta_host` | 需容纳 complex64 的 α/β |
| Python/ATen | 未见 `aten::_sparse_addmm` NPU 注册 | 必须注册，行为对齐 PyTorch 2.7+ |
| 性能 | 无 A100 对标门槛 | fp16/bf16/fp32 ≥ 1.0× A100；complex64 ≥ 0.8× A100 |

结论：**不新写一套 SpMM 接口**，在 `arch35` 上补齐 dtype/布局/算法/ATen，并保证与 `arch22`（A2 任务）通过编译期/运行期分流共存。

# 二、需求分析（required）

## 2.1 需求描述

在 Ascend 950 上打通：

`torch.sparse.addmm` → `aten::_sparse_addmm` → `aclsparseSpMM*` → `sparse/spmm/arch35` Kernel

核心计算必须在 NPU 完成。C++ 语义对标 CUDA Toolkit 13.3 Update 1 的 cuSPARSE SpMM；Python 语义对标 PyTorch 2.7+。

## 2.2 需求拆解

1. **描述符补齐**：`CreateCsr/CreateConstCsr/CreateDnMat` 支持 fp16/bf16/fp32/complex64；索引 int32；`idxBase` 0/1；非法参数返回确定错误码。
2. **Workspace / Preprocess / Compute**：950 上四类 dtype、Row/Col 布局、CSR_ALG1/2/3 的 workspace 计算、行重排分桶、Kernel 执行。
3. **ATen 适配**：NPU 注册；dtype/device/shape/广播/stride/异常对齐 PyTorch；`beta=0` 不读取 input 中 NaN/Inf；实数 dtype 拒绝虚部非零的复数标量。
4. **泛化**：动态 M/K/N/nnz；空 nnz、空行、长尾行、非连续稠密 Tensor（按任务书支持范围）。
5. **与 A2 解耦**：硬件相关代码只放 `arch35/`；公共校验/枚举/错误码放 `sparse/common` 或现有 Host 层，避免和 A2 PR 互踩。
6. **精度**：混合容差单标杆，CPU Golden（fp16/bf16→fp32，fp32→fp64，c64→c128）。
7. **性能**：P-01/P-02/P-03 及任务附带 50 条性能用例；NPU Profiler Kernel 总耗时 vs A100 NCU。

## 2.3 支持硬件与软件

| 项目 | 要求 |
| --- | --- |
| 硬件 | Ascend 950PR |
| CANN | ops-sparse 指定版本 |
| PyTorch | ≥ 2.7 |
| torch_npu | ≥ 26.0.0 |
| 开发环境 | hidevlab WebIDE / 950 算力 |

# 三、详细设计（required）

## 3.1 算子分析

### 3.1.1 数学公式

$$
\mathrm{out} = \beta \cdot \mathrm{input} + \alpha \cdot \mathrm{op}(A_{\mathrm{CSR}}) \cdot \mathrm{op}(B)
$$

`input` 按 PyTorch 广播到 `[M,N]`；`mat1` 为 `[M,K]` CSR；`mat2` 为 `[K,N]` dense。

### 3.1.2 支持数据类型

| 层 | dtype |
| --- | --- |
| Python / ATen | float16, bfloat16, float32, complex64；三者必须相同，不做提升 |
| aclsparse | `ACL_FLOAT16`, `ACL_BF16`, `ACL_FLOAT`, `ACL_COMPLEX64` |
| 索引 | 仅 `ACL_SPARSE_INDEX_32I` |

complex64：实部/虚部分别按 fp32 混合容差验收；须支持复数 α/β 及声明范围内的共轭转置。

### 3.1.3 支持形状与约束

- 动态 shape：规格内 M/K/N/nnz
- `mat1.size(1) == mat2.size(0)`
- 仅 `input` 广播；`mat1`/`mat2` 不做矩阵维广播
- 同设备 NPU；跨设备报错
- B/C：`ACL_SPARSE_ORDER_ROW`（`ld >= cols`）与 `ACL_SPARSE_ORDER_COL`（`ld >= rows`）
- 未在 cuSPARSE 13.3 官方表中的「布局 × opA × opB × dtype × alg」组合：返回明确错误，不静默降级

### 3.1.4 算法与布局策略

| 算法 | 优先路径 | 备注 |
| --- | --- | --- |
| CSR_ALG2 | 行主 | 对齐任务书 |
| CSR_ALG1 | 列主 | 现有 arch35 已有 CSR_ALG1 骨架，优先复用 |
| CSR_ALG3 | CSR 且 opA=NON_TRANSPOSE | 不支持 opB=CONJUGATE_TRANSPOSE |
| DEFAULT | 映射到任务允许的默认 ALG | 文档中写死映射，避免和 A2 不一致 |

## 3.2 算子实现

### 3.2.1 分层

```
torch.sparse.addmm
        │
        ▼
aten::_sparse_addmm  (ops-sparse Python/ATen 适配)
        │  校验 dtype/device/shape/stride/αβ
        │  CSR 元数据 → aclsparse 描述符
        ▼
aclsparseSpMMGetBufferSize / Preprocess / SpMM
        │  公共 Host：参数校验、错误码、workspace 布局
        ▼
sparse/spmm/arch35/{spmm_host,spmm_kernel,spmm_csr_mat}
        │  950 tiling / 行重排分桶 / Cube+Vector Kernel
        ▼
NPU 950
```

不新增第二套 `aclsparseSpMM*` 名字。Python 层只做注册与转换，计算走现有 C++ 入口。

### 3.2.2 Host 侧（arch35）

复用现有 workspace 布局并扩展：

```
[64B header]
[SpmmTilingData]          // 扩展：dtype、alg、idxBase、complex α/β、nnz
[int32 reorder[m]]        // 行重排
[int32 bin_edge[bin+1]]   // 分桶
[可选：complex 标量/共轭工作缓冲]
```

改动点：

1. `SpmmDataTypeFromAcl` 增加 `ACL_BF16`、`ACL_COMPLEX64`。
2. `SpmmTilingData` 的 `alpha_host/beta_host` 改为按 dtype 存储（fp32 路径保持 float；c64 存两个 float 或 `aclFloatComplex`）。
3. Preprocess：idxBase=1 时在 Host 将列索引统一到 0-base 视图（或 Kernel 内减 1，二选一，文档锁定一种并测非法索引）。
4. 未支持组合在 GetBufferSize 即失败，避免 Preprocess/SpMM 走到半截。
5. `beta=0`：ATen 层仍校验 input 的 shape/dtype/device，但不把 input 数值传入 Kernel（或 Kernel 跳过 β·C 读）。

公共逻辑（A2/A5 共用）只放：

- 描述符字段、错误码枚举、CSR 合法性检查
- 算法/布局/dtype 组合表（按 soc 查询）

950 特有 tiling、Kernel launch、workspace 公式只放 `arch35/`。

### 3.2.3 Kernel 侧（arch35）

现有 `spmm_kernel.cpp` 已按 fp32/fp16/int8 分发。本任务：

1. **fp32**：复用现有 CSR_ALG1 路径，补 ALG2/ALG3 与行主/列主。
2. **fp16 / bf16**：fp16 已有 compute=fp32 路径；bf16 按同样提升到 fp32 累加再写回，满足混合容差。
3. **complex64**：拆成实部/虚部四次实数乘加，或一条复数 Kernel；优先与现有分桶调度兼容，避免为 c64 另搞一套 tiling。共轭转置只在声明支持的组合启用。
4. 空 nnz / 空行：分桶后空桶直接写 `β·C`（β=0 则填 0）。
5. 确定性：同一 alg 重复执行 bit-wise（任务书对确定性算法的要求）；非确定性 alg 走混合容差。

禁止 Host for 循环逐元素算 SpMM。

### 3.2.4 Python / ATen 适配

1. 在 ops-sparse 现有 PyTorch 适配目录注册 `aten::_sparse_addmm` 的 NPU impl。
2. 校验：
   - 三输入同 dtype、同 NPU device
   - `mat1` 为 2D CSR，`csrRowOffsets/csrColInd` 为 int32
   - `mat2` 2D；`input` 可广播到 `[M,N]`
   - 非连续 dense：能按 stride 描述则直接用，否则 `.contiguous()` 副本
3. 标量：转成共同 compute dtype；实数拒绝 `imag != 0` 的复数。
4. 输出：`[M,N]`，dtype 与输入相同，device 相同。
5. Profiler：kernel 名必须出现在 `torch_npu.profiler` / `op_statistic.csv`，作为无 CPU fallback 证据。

## 3.3 精度与自测方案

### 3.3.1 任务附带用例（已有）

目录：`aclsparseSpMM_testCase/`

| 集合 | 数量 | dtype |
| --- | --- | --- |
| 精度 JSON | 200 | **仅 fp32 100 + c64 100** |
| 性能 JSON | 50 | **仅 fp32 25 + c64 25** |

附带用例**不覆盖 fp16/bf16**。自测报告必须额外补：

- fp16 / bf16 精度抽样（shape/nnz/空行/αβ 各若干）
- 性能表 P-02、P-03 的 fp16/bf16（及 P-03 的 c64）

否则会出现「附带脚本全绿、验收仍失败」。

### 3.3.2 精度标准

混合容差单标杆，匹配率 ≥ 0.99，且 `|err| ≤ max(A, 32×ULP(golden))`：

| dtype | rtol | atol | A | Golden |
| --- | --- | --- | --- | --- |
| float16 | 2^-9 | 2^-9 | 1e-1 | fp32 |
| bfloat16 | 2^-6 | 2^-6 | 1e0 | fp32 |
| float32 | 2^-10 | 2^-16 | 1e-2 | fp64 |
| complex64 | 实/虚各按 fp32 | 同左 | 同左 | complex128 |

### 3.3.3 性能标准（任务书硬门槛）

预热 ≥10，正式 ≥30，报中位数与 P90。正式采样复用描述符/workspace/preprocess。倍率 = GPU `kernel_total_us` / NPU `kernel_total_us`。

| ID | 规模 | 目标 |
| --- | --- | --- |
| P-01 | 2708³ 量级，nnz=10556，fp32，α=1 β=0 | ≥ 1.0× A100（87.040 μs） |
| P-02 | 169343²×128，nnz=1.16e6，fp16/bf16/fp32，α=β=1 | ≥ 1.0× |
| P-03 | 2449029²×256，nnz=6.19e7，四 dtype，α=1 β=0 | 实数 ≥1.0×；c64 ≥0.8×（c64 GPU 38806.400 μs） |

P-03 是最大风险项：内存、workspace、分桶与 Kernel 访存都可能爆。开发顺序：**fp32 小 shape 打通 → 附带 200/50 → fp16/bf16 → c64 → P-01 → P-02 → P-03**。

### 3.3.4 C++ UT 必测

GetBufferSize → Preprocess → SpMM；workspace 不足；维度/dtype 不匹配；非法索引；非法布局/alg；泄漏（反复 create/destroy）。

# 四、约束与风险

1. 不能 CPU fallback；验收看 Profiler。
2. 不能新增同名 C++ 接口；签名变更必须源码兼容。
3. A2 任务 25 人、本任务 14 人，都会改 `ops-sparse`。后合入方处理冲突；950 代码禁止写进 `arch22/`。
4. 现有 TilingData 用 `float` 存 α/β，c64 必须改结构，属于兼容性风险，需在实现时保持旧 fp32 二进制布局或加 version 字段。
5. cuSPARSE 13.3 组合表以外一律报错，避免「能跑但验收判未声明」。
6. 100 小时云端时长：不使用时关闭；大 case 先在小 shape 验证再上 P-03。

# 五、交付与合入计划

| 阶段 | 内容 | 状态 |
| --- | --- | --- |
| 报名 | 任务广场审核通过 | 已完成 |
| 设计文档 | 本文件 PR 至 `cann-ops-competitions` `tasklist/08-13-aclsparseSpMM-950/Jessicaruan/docs/design.md` | 进行中 |
| 代码仓 | Fork `ops-sparse`，邀请 `Ascend-CANN` 为开发者 | 待办 |
| 功能 | arch35 补齐四 dtype + ATen 注册 + 附带 200 条精度 | 待办 |
| 性能 | P-01/02/03 与 50 条；Profiler 证据 | 待办 |
| 验收 zip | 自测报告 + README + 仓/分支/目录 | 待办 |
| 合入 | PR 到 `ops-sparse` master，处理与 A2 的 Host 冲突 | 待办 |

# 六、参考资料

1. 任务书与自测包（本地 `0817复杂任务/aclsparseSpMM 算子开发(950)任务书/`）
2. ops-sparse SpMM README：https://gitcode.com/cann/ops-sparse/blob/master/sparse/spmm/README.md
3. arch35：`sparse/spmm/arch35/{spmm.h,spmm_host.cpp,spmm_kernel.cpp,spmm_csr_mat.*}`
4. PyTorch `torch.sparse.addmm`：https://docs.pytorch.org/docs/stable/generated/torch.sparse.addmm.html
5. cuSPARSE SpMM：https://docs.nvidia.com/cuda/cusparse/index.html#cusparseSpMM
6. 精度标准：https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md
7. 社区任务流程：https://gitcode.com/org/cann/discussions/39
