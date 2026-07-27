# 8月社区任务 - torch_sparse SpMM 算子开发任务书

## 任务概述

参考 pytorch_sparse（https://github.com/rusty1s/pytorch_sparse ） 中稀疏-稠密矩阵乘法（SpMM）相关接口，在昇腾 NPU 上实现 **PyTorch 适配层**及 **reduce 扩展**（mean/min/max、Autograd 等），并通过 `torch.ops` 对外提供服务。

**已有实现（本任务不重复开发）**：ops-sparse 仓 **`aclsparseSpMM`** 及 `src/spmm/arch35/spmm_kernel.cpp` 已实现 SpMM 基本功能（CSR、`reduce=sum`）；Kernel **已有** fp32 同精度、fp16→fp16（`computeType=fp32`）、int8→int32 三种组合（见 §3.2 组合 U1、M1、M5）。本任务在此基础上新增 PyTorch Host 封装，补齐 cuSPARSE 其余混精组合（§3.2 M2–M4、M6）、mean/min/max、COO 路径及 Autograd；**不得影响**现有 ACL API 行为。

pytorch_sparse 中 **稀疏×稠密（SpMM）** 须在本任务对标实现的 torch 接口如下（**不含** `SparseTensor.matmul(SparseTensor)`，该路径为 `spspmm`，不在本任务范围）：

| 层级 | 接口 | 格式 | 语义 | 本任务范围 |
|------|------|------|------|------------|
| Python | `torch_sparse.spmm(index, value, m, n, matrix)` | COO × 稠密 | 固定 `reduce=sum`；`scatter_add` 实现 | 新增适配 |
| Python | `SparseTensor.spmm(other, reduce=...)` | CSR × 稠密 | `other` 须为稠密 `Tensor`；`reduce ∈ {sum, add, mean, min, max}` | 新增适配 |
| Python | `SparseTensor.matmul(other, reduce=...)` | CSR × 稠密 | **`other` 为稠密 `Tensor` 时与 `spmm` 完全等价**（内部同一 `spmm()`） | 新增适配 |
| C++ | `torch.ops.torch_sparse.spmm_sum` | CSR × 稠密 | 求和聚合 | **Kernel 已有**，补 PyTorch/Autograd |
| C++ | `torch.ops.torch_sparse.spmm_mean` | CSR × 稠密 | 均值聚合 | 新增 |
| C++ | `torch.ops.torch_sparse.spmm_min` | CSR × 稠密 | 最小值聚合 | 新增 |
| C++ | `torch.ops.torch_sparse.spmm_max` | CSR × 稠密 | 最大值聚合 | 新增 |

SpMM 核心计算公式（CSR 格式，对标 `spmm_sum`，`α=1, β=0`）：

$$
C_{ij} = \bigoplus_{\substack{(p,q,v) \in \text{CSR}(A) \\ p=i}} v \cdot B_{qj}
$$

其中 $\bigoplus$ 为聚合算子（sum / mean / min / max）。

## 核心开发要求及验收标准

开发/测试硬件和软件要求：
- **适配硬件**：Ascend 950PR
- **CANN 版本**：算子开源仓（https://gitcode.com/cann/ops-gnn ）指定版本

### 功能实现要求

#### 1. 架构约束（强制）
本任务**仅新增 NPU 路径**，CPU/CUDA 参考由 pytorch_sparse 提供，NPU 输入不得回退 CPU。

1. **Kernel 共用**：须调用 `spmm_kernel_launch` / `spmm_kernel.cpp`；reduce/dtype 扩展通过参数化实现，禁止复制独立 Kernel。
2. **ACL 隔离**：`aclsparseSpMM*` 三接口及现有测试**不得变更**；PyTorch 路径直接Launch kernel，**不得转调** `aclsparseSpMM`。

#### 2. 接口功能对齐

##### 2.1 COO 接口：`torch_sparse.spmm`

- 签名：`spmm(index, value, m, n, matrix)`（`torch_sparse/spmm.py`）
- 输入：`index` [2,E] int64；`value` [E] 或 [E,*]（非 bool/复数）；`matrix` [..., n, K]（`n == matrix.size(-2)`）
- 输出：[..., m, K]；语义为 `scatter_add`（reduce=sum）
- NPU 路径：COO→CSR（内部转换）→ `spmm_sum`

##### 2.2 CSR 算子：`spmm_sum`

- 签名见 `csrc/sparse.h` 及下方参数表
- 要点：`rowptr`/`col` 为 int64；`value` 缺省以 `col` 占位（`has_value=false`，**非**全 1）；`row`/`colptr`/`csr2csc` 在 Autograd 时必传；`mat`/`out` 支持 batch 维
- Autograd 对标 `SPMMSum`（`csrc/spmm.cpp`）

##### 2.3 CSR 算子：`spmm_mean`

- 在 `spmm_sum` 基础上增加可选 `rowcount` [M]；输出 `sum / rowcount`；Autograd 对标 `SPMMMean`

##### 2.4 CSR 算子：`spmm_min` / `spmm_max`

- 无 `row`/`colptr`/`csr2csc`；返回 `(out, arg_out)`，`arg_out` 为 int64，无效边填 `col.numel()`；Autograd 对标 `SPMMMin`/`SPMMMax`

##### 2.5 Python 封装：`SparseTensor.spmm` / `SparseTensor.matmul`（稠密路径）

- **`SparseTensor.spmm(other: Tensor, reduce=...)`**：CSR 稀疏×稠密 SpMM 专用入口
- **`SparseTensor.matmul(other: Tensor, reduce=...)`**：当 `other` 为稠密 `Tensor` 时，与 `spmm` **完全等价**（均调用 `matmul.py` 内同一 `spmm()`）
- 支持 `reduce="sum"|"add"|"mean"|"min"|"max"`（`"add"` ≡ `"sum"`）
- **`SparseTensor.matmul(other: SparseTensor)`** 分流至 `spspmm`，**不在本任务范围**

#### 3. 数据类型支持

**对标范围**：对齐 cuSPARSE cusparseSpMM §6.6.10（https://docs.nvidia.com/cuda/cusparse/index.html#cusparsespmm ）（COO/CSR/CSC/BSR 格式）官方 dtype 表。

**列含义（与文档一致）**：

- **A/B**：稀疏 `matA` 的 values 与稠密 `matB` 使用**相同** dtype（文档单列 `A/B`）；
- **C**：稠密输出 `matC` 的 dtype；
- **computeType**：实际计算/累加精度（`α`/`β` 亦为此类型）。

**文档 NOTE**：`CUDA_R_16F`、`CUDA_R_16BF` 一律视为混精度；混精表中 fp16/bf16 各行的 **`computeType` 均为 `CUDA_R_32F`**（并非与 C 同 dtype）。

**索引**：Python 对外 int64；Kernel / `aclsparseSpMM` 当前为 int32（`ACL_SPARSE_INDEX_32I`），Host 层负责转换。

**实现分层**：

| 层级 | 说明 |
|------|------|
| **Kernel 已有** | `spmm_kernel.cpp` 已实例化对应入口，`aclsparseSpMM` Host 校验已放行 |
| **Kernel 待补充** | 须扩展 `spmm_kernel_launch` / Host `IsSupportedSpmmDtypeCombo`，禁止复制独立 Kernel 文件 |
| **PyTorch 待适配** | 本任务须在 NPU 路径打通 Python / `torch.ops`（含 Autograd，reduce=sum 优先） |

##### 3.1 同精度（Uniform-precision）

`matA` values、`matB`、`matC`、`computeType` **四者相同**：

| ID | A/B / C / computeType | Kernel 状态 |
|----|------------------------|-------------|
| U1 | `CUDA_R_32F` | **已有**（`spmm_custom_fp32`） |

##### 3.2 混精度（Mixed-precision）— COO/CSR/CSC/BSR

与线上一致（表格含 `rowspan`：M2–M4 共享 `C=CUDA_R_32F`、`computeType=CUDA_R_32F`；M5–M6 共享 `computeType=CUDA_R_32F`）：

| ID | A/B | C | computeType | Kernel 入口 | Kernel 状态 | PyTorch 适配 |
|----|-----|---|-------------|-------------|-------------|--------------|
| M1 | `CUDA_R_8I` | `CUDA_R_32I` | `CUDA_R_32I` | `spmm_custom_int8` | **已有** | **必选** |
| M2 | `CUDA_R_8I` | `CUDA_R_32F` | `CUDA_R_32F` | — | **待补充** | **必选** |
| M3 | `CUDA_R_16F` | `CUDA_R_32F` | `CUDA_R_32F` | — | **待补充** | **必选** |
| M4 | `CUDA_R_16BF` | `CUDA_R_32F` | `CUDA_R_32F` | — | **待补充** | **必选** |
| M5 | `CUDA_R_16F` | `CUDA_R_16F` | `CUDA_R_32F` | `spmm_custom_fp16` | **已有** | **必选** |
| M6 | `CUDA_R_16BF` | `CUDA_R_16BF` | `CUDA_R_32F` | — | **待补充** | **必选** |


##### 3.3 Python 层 dtype 与 Kernel 映射

| Python `value` / `matrix` / `out` | cuSPARSE 组合 | 备注 |
|-----------------------------------|---------------|------|
| float32 / float32 / float32 | U1 | COO / CSR 均须支持 |
| float16 / float16 / float16 | M5 | pytorch_sparse 常用路径 |
| float16 / float16 / float32 | M3 | fp16 输入、fp32 输出 |
| bfloat16 / bfloat16 / bfloat16 | M6 | COO 支持 bf16 |
| bfloat16 / bfloat16 / float32 | M4 | bf16 输入、fp32 输出 |
| int8 / int8 / int32 | M1 | 量化 SpMM |
| int8 / int8 / float32 | M2 | int8 输入、fp32 输出 |

COO `torch_sparse.spmm` 的 `value` 为 `[E, *]` 多特征维时，可按特征维循环调用 `spmm_sum`；`value` 为 `[E]` 时与上表一一对应。

**验收**：U1、M1、M5 须通过现有 `test/spmm/` ACL 回归 + PyTorch 对比测试；M2–M4、M6 须新增 ACL 与 PyTorch 用例后方可验收。


### 参数说明

以下 C++ 接口签名与 pytorch_sparse/csrc/sparse.h（https://github.com/rusty1s/pytorch_sparse/blob/master/csrc/sparse.h ） 保持一致。

#### spmm_sum

| 参数名 | 输入/输出 | 描述 | 数据类型 | 维度 Shape |
|--------|-----------|------|----------|------------|
| row | 输入（可选） | CSR 行索引；`value`/`mat` Autograd 时必传 | int64 | [E] |
| rowptr | 输入 | CSR 行指针 | int64 | [M+1] |
| col | 输入 | CSR 列索引 | int64 | [E] |
| value | 输入（可选） | 边权；缺省时内部以 `col` 占位（`has_value=false`） | 与 mat 同 dtype | [E] |
| colptr | 输入（可选） | CSC 列指针；`mat` Autograd 时必传 | int64 | [N+1] |
| csr2csc | 输入（可选） | CSR→CSC 边序映射；`mat` Autograd 时必传 | int64 | [E] |
| mat | 输入 | 稠密矩阵 | float32/float16/int8 等 | [..., N, K] |
| out | 输出 | SpMM 结果 | 与 mat 同 dtype | [..., M, K] |

#### spmm_mean

在 `spmm_sum` 基础上增加：

| 参数名 | 输入/输出 | 描述 | 数据类型 | 维度 Shape |
|--------|-----------|------|----------|------------|
| rowcount | 输入（可选） | 每行非零边数；Autograd 时必传 | int64 | [M] |

#### spmm_min / spmm_max

| 参数名 | 输入/输出 | 描述 | 数据类型 | 维度 Shape |
|--------|-----------|------|----------|------------|
| rowptr | 输入 | CSR 行指针 | int64 | [M+1] |
| col | 输入 | CSR 列索引 | int64 | [E] |
| value | 输入（可选） | 边权；缺省语义同 spmm_sum | 与 mat 同 dtype | [E] |
| mat | 输入 | 稠密矩阵 | float32/float16/int8 等 | [..., N, K] |
| out | 输出 | SpMM 结果 | 与 mat 同 dtype | [..., M, K] |
| arg_out | 输出 | 取最值的边索引 | int64 | [..., M, K] |

#### COO `torch_sparse.spmm`（Python 层）

| 参数名 | 输入/输出 | 描述 | 数据类型 | 维度 Shape |
|--------|-----------|------|----------|------------|
| index | 输入 | COO 索引 | int64 | [2, E] |
| value | 输入 | 边权（浮点或整数，非 bool/复数） | 与 matrix 同 dtype | [E] 或 [E, *] |
| m | 属性 | 稀疏矩阵行数 | int | 标量 |
| n | 属性 | 稀疏矩阵列数 | int | 标量 |
| matrix | 输入 | 稠密矩阵 | 与 value 同 dtype | [..., n, K] |
| out | 输出 | SpMM 结果 | 与 matrix 同 dtype | [..., m, K] |

### 接口设计建议

PyTorch 算子注册（示意，命名空间须与 pytorch_sparse CUDA 侧一致，为 `torch_sparse`）：

```cpp
torch::Tensor spmm_sum(std::optional<torch::Tensor> row,
                       torch::Tensor rowptr, torch::Tensor col,
                       std::optional<torch::Tensor> value,
                       std::optional<torch::Tensor> colptr,
                       std::optional<torch::Tensor> csr2csc,
                       torch::Tensor mat);

torch::Tensor spmm_mean(std::optional<torch::Tensor> row,
                        torch::Tensor rowptr, torch::Tensor col,
                        std::optional<torch::Tensor> value,
                        std::optional<torch::Tensor> rowcount,
                        std::optional<torch::Tensor> colptr,
                        std::optional<torch::Tensor> csr2csc,
                        torch::Tensor mat);

std::tuple<torch::Tensor, torch::Tensor>
spmm_min(torch::Tensor rowptr, torch::Tensor col,
         std::optional<torch::Tensor> value, torch::Tensor mat);

static auto registry = torch::RegisterOperators()
    .op("torch_sparse::spmm_sum",  &spmm_sum)
    .op("torch_sparse::spmm_mean", &spmm_mean)
    .op("torch_sparse::spmm_min",  &spmm_min)
    .op("torch_sparse::spmm_max",  &spmm_max);
```

内部路径：`spmm_sum` → `SPMMSum::apply` → `spmm_torch_launch` → `spmm_kernel_launch`；**禁止**经 `aclsparseSpMM` 转调。

### 测试标准

1. **功能对齐**：
   - 以 pytorch_sparse CUDA 实现为参考标杆，在相同 CSR/COO 输入下对比 NPU 输出；
   - FP32/FP16：相对误差须满足《生态算子开源精度标准》；
   - INT8：逐元素精确一致；
   - 覆盖 `reduce` 全部四种聚合类型及 COO/CSR 两条路径。

2. **Autograd 验证**：
   - 对 `value` 与 `mat` 分别开启 `requires_grad`，验证 backward 数值与 CUDA 参考一致；
   - 覆盖 sum/mean 的 SpMM backward；min/max 验证 forward 与 arg_out。

3. **ACL 回归**：现有 `test/spmm/spmm_test.cpp` 等用例须**原样通过**；同规格用例 NPU 耗时相对合入前基线**不得劣化**。

4. **PyG 冒烟**：
   - 使用 `SparseTensor.matmul(x, reduce='sum')` 在 NPU 上运行 GCN/GIN 单层前向，结果与 CUDA 参考一致。

5. **泛化场景**：
   - 矩阵规模：M/N/K 从小规模（128）到大规模（10^5 节点级图特征维度 64/128/256）；
   - 稀疏度：50%~99.9%；
   - 边界：E=0、单行超长（hub 节点）、batch 维（若支持）。

### 性能要求

本任务分 **已有 ACL 路径** 与 **新增 PyTorch 路径** 两类性能要求：

| 路径 | 对标 | 达标要求 | 说明 |
|------|------|----------|------|
| ACL SpMM（已有） | 合入前 ops-sparse 基线 | **功能回归通过，性能不劣化** | `aclsparseSpMM` / `spmm_sum` 核心 Kernel 已实现；本任务改动后须相对当前基线 **无性能回退** |
| PyTorch 适配（新增） | pytorch_sparse CUDA `spmm_*`（A100） | **≥0.5×** | PyTorch 封装、mean/min/max、COO、Autograd 等新能力 |

**性能验收说明**：本任务若有多支队伍提交实现，同规格用例下横向对比各队 NPU 实测性能，**取性能最优的提交进行验收**。

**固定参考用例**（必测）：

| 编号 | M | N | K | nnz | dtype | reduce | ops-sparse 基线 NPU | 达标要求 |
|------|---|---|---|-----|-------|--------|---------------------|----------|
| 1 | 1024 | 1024 | 64 | ~10⁴ | fp32 | sum | 待实测 | ACL 不劣化；PyTorch ≥0.5× |
| 2 | 10000 | 10000 | 128 | ~10⁵ | fp32 | sum | 待实测 | 同上 |
| 3 | 1024 | 1024 | 64 | ~10⁴ | fp16 | sum | 待实测 | 同上 |
| 4 | 1024 | 1024 | 64 | ~10⁴ | fp32 | mean | — | PyTorch ≥0.5× |

**泛化覆盖范围**（在固定参考用例之外**另抽 100 组**用例；由下列维度组合抽样或网格扫描生成，须附完整用例列表；ACL 路径测不劣化，PyTorch 路径达标 ≥0.5×）：

| 维度 | 覆盖范围 |
|------|----------|
| 矩阵规模 | `M`、`N`：128～10⁵（含参考用例 2 量级）；`K`（特征维）：16、64、128、256、512 |
| 稀疏度 | `nnz`：10³～10⁷；稀疏率约 50%～99.9%（随机 CSR / 图数据） |
| dtype | §3 已规划组合：fp32（U1）、fp16（M5）、int8→int32（M1）；M2～M4、M6 实现后纳入 |
| reduce | `sum`（必测）；`mean` / `min` / `max`（PyTorch 路径实现后必测） |
| 数据路径 | CSR `spmm_sum` 与 COO `torch_sparse.spmm` 均须抽样 |
| 结构 | 规则图、幂律图；hub 行（单行 nnz ≥10³）；`E=0` 退化 |
| batch | 若支持 batch 维：`batch` 取值须纳入 100 组抽样（如 1、4、16 等） |

COO `spmm` 路径允许格式转换额外开销，自验证报告须单独列示 COO→CSR 与 SpMM 耗时。

### 精度要求

算子计算精度须严格满足《生态算子开源精度标准》：

https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md

采用 AscendOpTest（https://gitcode.com/HIT1920/AscendOpTest ） + ATK（https://gitcode.com/AscendTest/ATK ） 双标杆L2（2 / 1.2 / 1.2）测试。


### 文档规范要求

1. 算子设计文档须说明 ACL 路径与 PyTorch 路径的模块划分、Kernel 共用方案及 backward 实现策略，并通过评审；
2. 自验证报告须包含：NPU vs CUDA 功能对比、Autograd 对比、ACL 回归测试结果、性能数据；
3. README 须说明 PyTorch 扩展编译安装方式、与 pytorch_sparse 接口映射表及 PyG 集成示例。

## 验收交付件

在社区任务IT系统中提交验收时， 需要提交以下交付件：

| 序号 | 交付件名称 | 交付件要求 |
|------|-----------|------------|
| 1 | 算子设计文档 | 1. 设计文档模板：https://gitcode.com/cann/cann-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md ，并且须说明 ACL 路径与 PyTorch 路径的模块划分、Kernel 共用方案及 backward 实现策略；<br> 2. 在cann-competitions 仓库（https://gitcode.com/cann/cann-competitions/tree/master/04_tasks/01_community-task-2026/tasklist ）以PR形式提交设计文档，通过评审后合入仓库，详细说明见：https://gitcode.com/cann/cann-competitions/blob/master/04_tasks/01_community-task-2026/README.md|
| 2 | 自测用例及测试代码 |1. 需要清晰列出精度测试case和性能测试case；<br> 2. 测试代码中的readme文件需要说明测试步骤，保证验收人可以复现测试结果|
| 3 | 自测报告 | 1. 自测报告模板：https://docs.qq.com/sheet/DUmVWWndaUE12WGFB?tab=BB08J2 ；<br> 2. NPU vs CUDA 功能对比、Autograd 对比、ACL 回归测试结果、性能数据； <br> 3. 需要包含用例参数、精度对比结果及截图、性能数据及截图 |
| 4 | 待验收代码地址 | 1. 个人代码仓链接、分支、算子目录；需要在个人仓邀请账号Ascend-CANN作为开发者，如下图所示； <br> 2. 算子目录下需要提供readme文件，readme 须说明 PyTorch 扩展编译安装方式、与 pytorch_sparse 接口映射表及 PyG 集成示例 |

![邀请示意](./pics/invite.jpeg)


### PR 申请合入

验收通过后，在 ops-gnn 开源仓提交 PR，建议合入路径：

```
ops-gnn/src/sparse
```

## 参考资料

1. pytorch_sparse 源码（https://github.com/rusty1s/pytorch_sparse ）
   - SpMM Python：`torch_sparse/spmm.py`、`torch_sparse/matmul.py`
   - SpMM C++/CUDA：`csrc/spmm.cpp`、`csrc/cuda/spmm_cuda.cu`
2. PyG SparseTensor 文档（https://pytorch-geometric.readthedocs.io/en/stable/advanced/sparse_tensor.html ）
3. ops-sparse 已有 SpMM 实现：`src/spmm/arch35/spmm_kernel.cpp`、`spmm_host.cpp`、`test/spmm/`

## 环境获取

1. 使用 hidevlab webIDE 算力：https://hidevlab.huawei.com/online-develop-intro?from=hiascend ；
   - 如果是新用户，在申请权限的时候需要备注使用的算力类型（A2、A3或者950）；
   - 如果是老用户且需要使用950算力，需要向昇腾CANN小助手反馈账号名（个人中心->基本信息，如下图所示），后台会添加账号至950使用白名单。

   ![环境截图](./pics/zaixiankaifa1.png)  
   ![账号名](./pics/account.png)  

2. 开源仓提供100小时免费时长，请不使用时及时关闭，用时耗尽前请务必保存相关资料，建议及时提交备份。

   ![环境截图](./pics/yunkaifa.png)

3. 如需额外环境资源，请联系昇腾CANN小助手。


## 特别注意事项

1. **ACL 隔离**：不得破坏 `aclsparseSpMM` 及现有测试；ACL/PyTorch 共用预处理/Launch，禁止两套 Tiling 逻辑。
2. **接口语义**：索引对外 int64；`value` 缺省以 `col` 占位（非全 1）；算子注册至 `torch_sparse` 命名空间；
3. **范围**：COO 路径允许 Host 侧转 CSR；`SparseTensor.matmul(SparseTensor)`（`spspmm`）不在本任务范围；
4. min/max 不支持时扩展 Kernel，sum 路径零回归；交付前完成自验证（含 ACL 回归）;
5. 算子注册至 `torch_sparse` 命名空间，`import torch_sparse_npu` 调用；
6. 所有交付件须提前完成自验证，确认符合验收标准后再提交；
7. 开发前务必阅读【社区任务】流程及注意事项（https://gitcode.com/org/cann/discussions/39 ）。

