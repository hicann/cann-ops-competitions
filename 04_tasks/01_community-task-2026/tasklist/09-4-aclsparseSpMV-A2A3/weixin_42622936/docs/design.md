# aclsparseSpMV 算子设计文档（A2/A3，arch22）

## 一、需求背景

### 1.1 任务来源

9月社区任务：aclsparseSpMV 算子开发（A2/A3）。面向 Atlas A2/A3（DAV_2201，arch22）NPU，使用 C++ Host 与 Ascend C Kernel 实现 CSR 稀疏矩阵-稠密向量乘 `Y = alpha * op(A) * X + beta * Y`，接口语义对标 cuSPARSE SpMV 三阶段调用（GetBufferSize / Preprocess / SpMV），并交付 Python/ATen 入口（`torch.mv` → `aten::mv`，仅 CSR 二维稀疏 × 一维稠密，NPU 执行，无 CPU fallback）。

### 1.2 现状分析

ops-sparse 仓 `sparse/spmv/arch22/` 已有基础实现，经代码走读，现状与任务要求的差距如下：

| 项目 | 现状 | 任务要求 | 差距 |
|---|---|---|---|
| dtype 组合 | FP32→FP32、FP16/BF16→FP32、FP16→FP16、BF16→BF16、INT32值→FP32、INT32→INT32（7 个 kernel 实例） | 增加 INT8 值类型（→INT32、→FP32）、complex64→complex64、FP32→complex64，共 9 组合 | **缺 INT8 值类型与 complex64 两类** |
| index base | 仅支持 base 0（base 1 显式返回 NOT_SUPPORTED） | I32 索引，base 0/1 均支持 | **缺 base 1** |
| 三阶段接口 | 仅 `aclsparseSpMV` 执行接口；GetBufferSize/Preprocess 未实现（README 注明未支持） | 三阶段完整闭环，workspace 精确查询、Preprocess 绑定 pattern 与生命周期 | **缺 GetBufferSize / Preprocess** |
| 长尾行 | Host 侧将 rowPtr 整体 D2H 拷回计算最大行长，超过 UB 容量直接报错 | 支持长尾行、空行、nnz=0，且无无谓 Host 同步 | **需重构**：行长不受 UB 限制、消除热路径 D2H 同步 |
| alpha/beta | 仅 Host 指针读取 | 按 computeType 解释（含 complex64 复数标量）；Device pointer mode 视 `aclsparseSetPointerMode` 分支实现情况验收 | 需扩展复数标量解析 |
| Python/ATen | 无（python 层仅有 spgemm 适配） | `torch.mv`/`aten::mv` NPU 注册，固定 alpha=1/beta=0，异常与 stream 语义对齐 PyTorch | **全新开发** |
| 测试 | 单个 C++ 测试文件 | C++ UT/ST + ATen UT + Python 端到端 UT + 精度/性能/内存自验 | 需体系化补全 |

## 二、方案设计

### 2.1 总体架构

```
torch.mv (Python)
    └─ ATen dispatcher: aten::mv (NPU PrivateUse1 注册, 仅 CSR×DnVec)
        └─ aclsparseSpMVGetBufferSize / aclsparseSpMVPreprocess / aclsparseSpMV (C API)
            └─ Host：参数校验 → 类型分发 → tiling → 启动 kernel（调用方 stream）
                └─ Ascend C kernel（arch22，按 dtype 组合模板实例化）
```

### 2.2 支持的 dtype 组合

| computeType | values/x 类型 | y 类型 | 说明 |
|---|---|---|---|
| INT32 | INT8 | INT32 | INT32 计算，逐元素 exact match |
| FP32 | INT8 | FP32 | FP32 计算 |
| FP32 | FP16 | FP32 | FP32 计算 |
| FP32 | BF16 | FP32 | FP32 计算 |
| FP32 | FP16 | FP16 | FP32 计算后写回 FP16 |
| FP32 | BF16 | BF16 | FP32 计算后写回 BF16 |
| FP32 | FP32 | FP32 | FP32 计算 |
| complex64 | complex64 | complex64 | 复数计算（实虚部 FP32 分量累加） |
| complex64 | FP32 | complex64 | 实数输入广播为复数（虚部 0），complex64 计算 |

索引固定 I32；index base 0/1 在 kernel 内通过偏移量统一处理（base 1 时列索引与行偏移减 1）。

### 2.3 Kernel 算法方案

- **非转置（N）**：行均衡分配。按行数将行区间均分到各 AIV 核；行内非零元按 UB 容量分块循环（CopyIn → 乘累加 → 下一块），行长不受 UB 限制（解决长尾行）；空行自然跳过；`beta=0` 时直接覆写不读 y。行内归约在 UB 完成，结果一次性写回 GM。
- **转置（T）**：`y[j] += A[i,j] * x[i]` 为散射写。采用按行遍历、对 y 的原子加（或按列段分桶后归约）；复数路径按实/虚部分量分别原子加。转置性能标杆本身约为非转置的 5 倍耗时，该方案可满足 0.25 倍率目标。
- **complex64**：values/x/y 按交错实虚存储处理，kernel 内拆为两个 FP32 分量流，乘加按 `(ar*br - ai*bi, ar*bi + ai*br)` 展开，累加全程 FP32 分量，最后按 computeType 写回。
- **确定性**：非转置每行单核归约、写回顺序固定；转置原子加引入的浮点累加顺序差异通过精度标准（rtol/atol + 匹配率 ≥0.99）覆盖，同一输入重复执行结果一致。

### 2.4 Host 三阶段设计

- `aclsparseSpMVGetBufferSize`：按 dtype/算法/shape 精确计算 workspace（含转置归约辅助区），做溢出检查；不支持组合返回 NOT_SUPPORTED。
- `aclsparseSpMVPreprocess`：绑定 matA pattern 与 externalBuffer，缓存于描述符；pattern 变更后需重新预处理；本方案 preprocess 以轻量元数据为主，不做重量计算。
- `aclsparseSpMV`：参数校验（全量对齐任务书 §2.4 异常行为）→ 描述符解包 → tiling → 在调用方 stream 上启动 kernel。**消除现有实现的 rowPtr D2H 回拷与 Host 同步**，全部依赖信息通过 tiling/描述符在 Host 侧可得或下推至 kernel 处理。
- alpha/beta 按 computeType 解释（INT32/FP32/complex64），Host 指针模式为基线；Device 指针模式在实现 `aclsparseSetPointerMode` 对应分支后覆盖。

### 2.5 Python/ATen 适配

- 新增 `python/ops_sparse_torch/csrc/spmv_torch.cpp`（仿 `spgemm_torch.cpp` 模式）：
  - 注册 `aten::mv` 的 NPU（PrivateUse1）实现，仅接受 CSR 二维稀疏矩阵 × 一维稠密向量；
  - 校验 dtype/shape/layout/stride/device，不支持组合显式报错，**不 fallback 到 CPU**；
  - 固定 alpha=1、beta=0 调底层三阶段接口，构造并返回输出 tensor；
  - alias/in-place 与 stream 异步语义对齐 PyTorch。
- 同时注册任务测试入口 `torch.ops.ops_sparse_test.spmv_npu(row_ptr, col_ind, values, x, y, alpha, beta, trans, base)`。

## 三、详细设计

### 3.1 算子分析

**数学公式**：`y = alpha * op(A) * x + beta * y`

- A：CSR 稀疏矩阵 `[M, K]`，nnz 动态，I32 索引，base 0/1
- x：非转置 `[K]`，转置 `[M]`；y：非转置 `[M]`，转置 `[K]`
- op(A)：NON_TRANSPOSE / TRANSPOSE
- 输入 A、x 只读，y 原地更新；支持空行、长尾行、nnz=0/1 边界

### 3.2 Host 侧设计

1. **参数校验**：handle/描述符非空 → 枚举合法（opA、alg、computeType、format、index type/base）→ dtype 组合表 → shape/长度匹配（x/y 长度随 opA 变化）→ 设备指针非空。非法枚举返回 `ACL_SPARSE_STATUS_INVALID_ENUM` 类错误，非法值返回 `INVALID_VALUE`，未支持组合返回 `NOT_SUPPORTED`。
2. **分核策略**：`blockDim = AIV 核数`；非转置按行均分（每核 ⌈M/blockDim⌉ 行，尾核处理余数）；转置按行均分遍历非零元。
3. **tiling 数据**：rows/cols/nnz/base/opA/dtype id/行区间表，通过 tiling struct 传 kernel；不引入运行时 Host↔Device 往返。
4. **workspace**：GetBufferSize 按场景精确返回（转置归约缓冲等），Preprocess/执行复用同一 externalBuffer。

### 3.3 Kernel 侧设计

每核 Init → Process（CopyIn / Compute / CopyOut 流水）：

- **CopyIn**：按 UB 分块搬入本行（或行段）的 colInd、values，并按 colInd gather x 的对应元素；double buffer 隐藏搬运延迟。
- **Compute**：乘累加在 FP32（或 INT32 / FP32 分量复数）域完成；行内分块间保留部分和，行结束归约得单值。
- **CopyOut**：非转置写出 `alpha*acc + beta*y[i]`（beta=0 跳过读 y）；转置对 y 做分量原子加。
- **base 1**：索引读入后统一减 1，逻辑与 base 0 收敛。
- **模板实例化**：`spmv_kernel.h` 模板按 (compute, value, out) 组合实例化 9 组，新增 complex64 与 INT8 实例。

## 四、测试设计

| 层级 | 内容 |
|---|---|
| C++ UT/ST（`test/spmv/arch22/`） | 三阶段流程、9 dtype 组合 × N/T × base 0/1 × alpha/beta 特例、空行/长尾行/nnz=0/1、非法参数与枚举、描述符 Create/Destroy 泄漏检查 |
| ATen UT | `aten::mv` NPU 注册、参数/异常、输出构造、无 CPU fallback |
| Python 端到端 UT | `torch.mv` CSR×vec 全 dtype 组合，对比 CPU golden |
| 精度自验 | 任务包 `accuracy_cases.json` 200 条；INT8→INT32 exact match；FP16/BF16 用 FP32 golden；FP32 用 FP64 golden；complex64 用 complex128 golden（实虚分别按 FP32 标准） |
| 性能自验 | P-01/P-02/P-03（Llama 3.1 70B / Qwen3-235B / DeepSeek-V3 场景），warmup ≥10、采样 ≥30，median/p90；目标 ≥0.25 倍 GPU 标杆 |
| 内存自验 | `collect/compare_sparse_ops_memory`：额外内存 ≤ GPU 50%，或固有 workspace ≤ L2 |
