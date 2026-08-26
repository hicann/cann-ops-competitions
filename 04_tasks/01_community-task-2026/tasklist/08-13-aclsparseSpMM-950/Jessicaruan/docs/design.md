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

# 修订 v1.1：阶段实测结果（2026-08-21）

> 本节补充 v1.0 之后的真机进展。环境：Ascend 950PR（hidevlab），CANN 9.1.0，
> Release 构建 `--soc=ascend950`。代码：个人仓 `ops-sparse` 分支 `feat/spmm-950-c64`。
> 以下均为阶段数据，不构成最终验收结论。

## 1. 状态总览

| 交付项 | 状态 |
|---|---|
| 设计文档 | v1.0 已合入（MR #1073）；本 v1.1 阶段修订 |
| arch35 四 dtype kernel + ATen 桥 | 代码在分支；官方精度 200 ALL_PASS（冻结 tip 上 fp32 100 已复验） |
| P-01 fp32 | **~2.37× 已过线**（≥2.0× 门禁） |
| P-02 | fp32 ~0.20× / bf16 ~0.55× / fp16 ~0.42×；**SIMT 族 + SHFL-BC 已穷尽，性能线冻结** |
| P-03 | 未起跑（流量地板高于门禁，见 §4） |
| 官方 200 精度 | fp32+c64 **200 ALL_PASS**（午前）；冻结 tip c64 复验列入交付补采 |
| 官方 50 性能 | fp32 **12/25** @1.0×；c64 几何最高 **2/25** @0.8×（已 revert，定格 Narrow） |

## 2. 已过线锚点（P-01）

n≥512 的 RR fp32 走 UB 向量路径（TPipe 双缓冲 `bQueue_`，B 行整段 `DataCopyPad`
+ `Muls`/`Add`），每 nnz 只加载一次 A 值并连续消费 B：

```
P-01 kernel_total_us 36.6–38.3（多次采样） ；ratio vs 87.040μs = 2.27–2.38×
```

该路径实测有效带宽约 2.0 TB/s（76.1MB/36.7μs），是后续窄 N 路径的对标基线。

## 3. P-02 窄 N（n=128）调优：成本模型与已否决路径

### 3.1 带宽与成本模型（实测校准）

- 顺序流微基准（2GiB stream）：950PR 实测峰值 **≈1.60 TB/s**（官网规格 100%）。
- P-02 无 B 重用流量 ≈780MB（fp32）→ 理论地板 ≈488μs，**门禁 204.576μs 在
  「无重用」假设下不可达**，与 A100 依赖 L2 滑窗重用的口径推断一致。
- 由 fp32（1030μs）/bf16（755μs）两点拟合：`T ≈ 470μs 延迟截距 + bytes/1390GBps`。
  截距与 dtype 无关 → 瓶颈为 GM 标量 load→use 延迟，非带宽或发射率。

### 3.2 当前最优：32×4 warp 协作（GE-SpMM 式）

32 lane 共享一行、每 lane 4 列（512B 合并宽度，fp32 累加），fp32 ≈1030μs
（0.20×）。cols-per-lane 扫描（4→1030，8→1102，16→1532，128→3300）证明
**32×4 已是该维度的 LSU 甜点**。

### 3.3 已否决路径（全部真机实测后 revert，历史保留在分支）

| 实验 | 结果 | 结论 |
|---|---|---|
| 行重排赌 L2（低 CV→连续分块） | 1045μs，无变化 | 合成条带重用距离 ≈K/7 行，重排无效 |
| warp 几何 16×8 / R4(4×8×16) | 1102 / 1532μs | lane 几何单调变差，全族否决 |
| CSR (c,v) 预取 tile=8 | 三 dtype 均回退 | 寄存器溢出，SIMT 无法软件流水 |
| panel-vec 单/双缓冲（B 行 512B 逐 nnz `DataCopyPad`） | 3318 / 4159μs | DMA 粒度阈值 ≈2KB，窄 N 整行 B 逐 nnz 搬运不成立（P-01 的 5732B 行则成立） |
| col-bucket 列并行 | ~100× 慢 + 精度失败 | 无原子加下同行竞争写 |
| 换真实 ogbn-arxiv 输入 | 更慢 ~2.2×（幂律长尾放大链长） | 非捷径；合规口径另行确认中 |

### 3.4 PIPE1 / SHFL-BC（flat → SIMT 族结案）

1-deep 软件流水线（PIPE1）：合成 fp32 **1043μs**（锚点 ~1030）→ 持平，已 revert。  
SHFL-BC（少 lane 读 CSR + `asc_shfl` 广播）：合成 fp32 **~1054μs** → 仍 flat，已 revert。  
结论：dav-3510 SIMT 在飞 load≈1；几何/软流水/panel/prefetch/R4/SHFL **全族关闭**。

### 3.5 官方 50 与 c64 几何刀（性能线冻结 tip `50571b9`）

- 官方 fp32 25：pass@1.0 = **12/25**；分界主轴为 B 重读倍数 `red=m*degree/k`，非 vec 路径质量。
- 官方 c64：Narrow 基线 0/25；WARP-C64 32×4 → **2/25**（几何 ×3–4 有效，KEEP≥5 未达 → revert）；
  PANELOUTER（panel-major）A 类相对行主持平 → **L2 跨 AIV 不可用（实测）**；
  VEC-C64 在 case 000 ratio 0.187 &lt;0.5 → 立即 STOP/revert。
- **本任务周期性能线冻结**：不再开 SIMT / L2 / VEC-C64 / UB-staging / AIC 实验；未达项以「B 重用墙」论证交付。

## 4. P-03 c64 流量审计（结论先行）

现有 `KernelSpmmSimtNarrowC64` 以 32 列 tile 分解，n=256 时每行 CSR+B **重扫 8 遍**；
无重用流量 ≈132.5GB → @1.6TB/s 地板 ≈82.8ms，**门禁 48.5ms（0.8×38.8ms）不可达**。
WARP-C64 已证几何有效但未达 KEEP；P-03 本周期不作为交付依赖。

## 5. 输入口径确认（合规杠杆）

任务书要求性能输入与 A100 基线同口径，但附件未提供 P-01/02/03 的 canonical
CSR fixture。P-02 规模（169343/1166243）与 ogbn-arxiv 精确匹配，我方已实现
真实图与合成对照双路径（`p02_bench.py`）。确认请求见私仓
`docs/FIXTURE_QUERY_TO_TASK.md`，将向 `cann/ops-sparse` 以【社区任务】Issue 发出。

## 6. 交付下一步（非性能再刀）

1. 自测报告：精度 200、P-01、官方 50 分桶、否证表、失败项说明；
2. 补采：干净进程对照、workspace/异常/泄漏、冻结 tip c64 精度复验；
3. 任务方确认 fixture 后，愿按新口径复测；
4. 验收通过后再向 `ops-sparse` master 提合入（与 A2 Host 冲突按后合入方处理）。


---

## 修订记录

| 日期 | 版本 | 说明 |
|---|---|---|
| 2026-08-17 | v1.0 | 按任务书和官方模板建立设计文档（MR #1073） |
| 2026-08-21 | v1.1 | 补充 950PR 实测、成本模型、否证表；定格官方 50 与性能线冻结 |
