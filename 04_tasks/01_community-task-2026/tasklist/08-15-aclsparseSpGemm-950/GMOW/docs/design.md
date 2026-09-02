# aclsparseSpGemm 算子设计说明

## 需求背景（required）

### 需求来源

- 通过 CANN 社区 8 月任务完成 `aclsparseSpGemm` 算子的 Ascend C NPU 开发，参考 PyTorch 稀疏矩阵乘法与 `aten::_sparse_sparse_matmul` 接口行为，在昇腾 NPU 上完成 Python/ATen 适配，并复用现有 SpGEMM 社区任务规划交付的 aclsparse C++ 多阶段接口及 Ascend C Kernel。
- 任务要求适配 Ascend 950PR，`float16`/`bfloat16`/`float32` 性能不低于 GPU（A100）的 1.0 倍，`complex64` 性能不低于 GPU（A100）的 0.8 倍。

### 背景介绍

#### 算子功能

`aclsparseSpGemm` 是稀疏矩阵-稀疏矩阵乘法（SpGEMM）算子。计算两个 CSR 格式稀疏矩阵的乘积，输出结果同样为 CSR 格式稀疏矩阵。

- 输入：稀疏矩阵 A `[M, K]`（CSR）、稀疏矩阵 B `[K, N]`（CSR）
- 输出：稀疏矩阵 C `[M, N]`（CSR）

计算公式：

$$
C = \alpha \cdot A \times B + \beta \cdot C
$$

其中 A 为 shape `[M, K]` 的稀疏矩阵，B 为 shape `[K, N]` 的稀疏矩阵，C 为 shape `[M, N]` 的稀疏输出矩阵。输出稀疏结构、`nnz(C)` 和 values 由乘法结果确定。

#### aclsparseSpGemm 算子实现路径

对标参考实现：
- Python 层：`torch.sparse.mm(mat1, mat2)`
- ATen Schema：`aten::_sparse_sparse_matmul(Tensor self, Tensor other) -> Tensor`
- cuSPARSE 参考：CUDA Toolkit 13.3 Update 1 所含 cuSPARSE 13.3 Update 1 的 SpGEMM 接口

当前 AscendC 工程路径：

```text
ops-sparse/
```

#### aclsparseSpGemm 算子现状分析

基于现有 SpGEMM 社区任务规划，已有接口定义及 `float16`、`bfloat16`、`float32` 的基础实现。本任务在此基础上：

- 复用已有 `aclsparseSpGEMM*` 多阶段接口及对应 Ascend C Kernel
- 新增 `complex64` 数据类型全链路支持
- 补齐 Python/ATen 适配层，完成 NPU 注册与稀疏输出构造
- 补齐异常处理、边界测试能力

Python/ATen 适配与 Kernel 分工：

| 阶段 | 执行位置 | 职责 |
| --- | --- | --- |
| 稀疏 layout 转换 | Python 层 | 将输入稀疏 Tensor 转为 CSR 格式 |
| NPU 注册与 dispatch | ATen 适配层 | 注册 `aten::_sparse_sparse_matmul` 的 NPU 实现 |
| 描述符创建与管理 | C++ Host 层 | 创建/销毁 SpGEMM 描述符 |
| 工作量估算 | C++ Host 层 | WorkEstimation 阶段，估算 workspace |
| 内存估算 | C++ Host 层 | EstimateMemory 阶段（ALG2/ALG3） |
| 结构及数值计算 | Ascend C Kernel（NPU） | SpGEMMCompute 核心计算 |
| 结果拷贝与组装 | C++ Host 层 | SpGEMMCopy，输出 CSR 结构组装 |
| 输出 Tensor 构造 | Python 层 | 按 PyTorch 语义构造输出稀疏 Tensor |

#### 总体流程图

![image.png](https://raw.gitcode.com/user-images/assets/10331120/53ccaddc-2382-4069-9d1e-b2412f6f2261/image.png 'image.png')

## 需求分析

### 外部组件依赖

- PyTorch 2.7 及以上版本
- torch_npu 26.0.0 及之后版本
- CANN 算子开源仓指定版本

### 内部适配模块

- 适配 `aten::_sparse_sparse_matmul` NPU 注册（torch_npu 自定义 dispatch）。
- 复用现有 SpGEMM 社区任务的 `aclsparseSpGEMM*` 接口及 Ascend C Kernel。
- 新增 `complex64` 全链路支持（描述符、多阶段接口、Kernel、输出组装、测试）。
- 覆盖不同 shape、nnz、稀疏度、空行/空列、中间乘积膨胀等泛化场景。

### 需求模块设计

#### 算子原型

**Python 公开入口：**

```python
torch.sparse.mm(mat1, mat2) -> Tensor
```

**ATen Schema：**

```text
aten::_sparse_sparse_matmul(Tensor self, Tensor other) -> Tensor
```

**C++ 多阶段接口：**

| 接口 | 功能 |
| --- | --- |
| `aclsparseSpGEMMCreateDescr` | 创建 SpGEMM 描述符 |
| `aclsparseSpGEMMWorkEstimation` | 阶段1：工作量估算 |
| `aclsparseSpGEMMGetNumProducts` | 查询中间乘积数量 |
| `aclsparseSpGEMMEstimateMemory` | 阶段2：内存估算（ALG2/ALG3） |
| `aclsparseSpGEMMCompute` | 阶段3：结构及数值计算 |
| `aclsparseSpGEMMCopy` | 阶段4：结果拷贝到 matC |
| `aclsparseSpGEMMDestroyDescr` | 销毁描述符 |

**参数说明：**

| 名称 | 类别 | dtype | format | shape | 说明 |
| --- | --- | --- | --- | --- | --- |
| mat1/self (A) | 输入 | float16/bfloat16/float32/complex64 (values)；int32 (indices) | CSR | `[M, K]` | 稀疏矩阵 A，csrRowOffsets 和 csrColInd 均为 int32 |
| mat2/other (B) | 输入 | float16/bfloat16/float32/complex64 (values)；int32 (indices) | CSR | `[K, N]` | 稀疏矩阵 B，csrRowOffsets 和 csrColInd 均为 int32 |
| output (C) | 输出 | float16/bfloat16/float32/complex64 (values)；int32 (indices) | CSR | `[M, N]` | 稀疏矩阵 C，结构和 nnz 由计算确定 |
| opA | 属性 | - | - | - | 仅支持 `ACL_SPARSE_OP_NON_TRANSPOSE` |
| opB | 属性 | - | - | - | 仅支持 `ACL_SPARSE_OP_NON_TRANSPOSE` |
| alpha | 属性 | 与 computeType 一致 | - | scalar | 乘法系数 |
| beta | 属性 | 与 computeType 一致 | - | scalar | 累加系数 |
| computeType | 属性 | aclDataType | - | - | ACL_FLOAT16/ACL_BF16/ACL_FLOAT/ACL_COMPLEX64 |
| alg | 属性 | aclsparseSpGEMMAlg_t | - | - | 算法枚举 |

相关约束：

- A、B 必须为二维稀疏矩阵，满足 `A.size(1) == B.size(0)`。
- A、B、C 均仅支持 CSR 格式，使用 `aclsparseCreateCsr` 创建描述符。
- `csrRowOffsetsType` 和 `csrColIndType` 均仅支持 `ACL_SPARSE_INDEX_32I`。
- 输入 A/B 的列索引必须有序，输出 C 的列索引必须有序。
- `opA` 和 `opB` 仅支持 `ACL_SPARSE_OP_NON_TRANSPOSE`。
- A、B、C 及 computeType 采用同一数据类型。
- 不进行矩阵维度广播。

## 需求详细设计

### 使能方式

本任务的接口分层与使能路径如下：

```
Python 层：torch.sparse.mm(mat1, mat2)              ← 本任务新增适配
    └── ATen 层：aten::_sparse_sparse_matmul          ← 本任务新增 NPU 注册
        └── aclsparse C++ 层：aclsparseSpGEMM* 多阶段接口  ← 复用现有 SpGEMM 社区任务，补齐 complex64
            └── Ascend C Kernel                        ← 复用现有 SpGEMM 社区任务，新增 complex64 Kernel
```

**与现有 SpGEMM 社区任务（7 月）的关系：**

| 层级 | 7 月 SpGEMM 社区任务 | 本任务（A5） |
| --- | --- | --- |
| Python/torch 接口 | 不涉及 | **新增**：`torch.sparse.mm` NPU 适配 |
| ATen NPU 注册 | 不涉及 | **新增**：`aten::_sparse_sparse_matmul` dispatch |
| aclsparse C++ 接口 | 交付 fp32/fp16/bf16 | **复用**并补齐 complex64 全链路 |
| Ascend C Kernel | 交付 fp32/fp16/bf16 | **复用**并新增 complex64 Kernel |
| 输出稀疏构造 | C++ 层完成 | **新增** Python 层 Sparse Tensor 构造 |

适配 PyTorch 2.7+ / torch_npu 26.0.0+，核心计算必须在 NPU 上完成，不允许 CPU fallback。

### 需求总体设计

#### Python/ATen 适配层设计

代码位置：`ops-sparse/` 相关 Python 及 ATen 适配目录

职责：
1. 注册 `aten::_sparse_sparse_matmul` 的 NPU dispatch，确保无 CPU fallback。
2. 将输入稀疏 Tensor（COO 或 CSR）统一转换为 CSR 格式。
3. 创建 `aclsparseCreateCsr` 描述符，设置 rows、cols、nnz、csrRowOffsets、csrColInd、values。
4. 按多阶段流程调用 C++ 接口完成计算。
5. 查询 `nnz(C)`，获取输出 CSR 指针，构造输出稀疏 Tensor 返回。
6. 按 PyTorch 目标版本语义处理输出 layout（COO 返回 coalesced 状态）。

#### AscendC host 侧设计

代码位置：`ops-sparse/` 中 aclsparse SpGEMM Host 实现

Host 文件职责拆分：

| 文件/模块 | 主要职责 | 关键点 |
| --- | --- | --- |
| SpGEMM 描述符管理 | 创建/销毁 SpGEMMDescr | 管理多阶段状态机、workspace 生命周期 |
| WorkEstimation | 阶段1：估算 workspace | 分析 A/B 结构，计算中间乘积上界 |
| EstimateMemory | 阶段2：内存估算 | ALG2/ALG3 分支，chunk 策略 |
| Compute 调度 | 阶段3：计算调度 | tiling 规划，Kernel launch |
| Copy 组装 | 阶段4：输出组装 | 将计算结果拷贝到 matC 描述符 |

Host 核心执行链路：

1. 解析 matA、matB 描述符获取 M、K、N、nnz(A)、nnz(B)。
2. WorkEstimation：分析行结构，估算中间乘积数量，计算 bufferSize1。
3. EstimateMemory（ALG2/ALG3）：根据 chunkFraction 估算 bufferSize2/bufferSize3。
4. Compute：根据 M、nnz 分布进行 tiling 规划，确定 blockDim 和每核任务划分，launch Ascend C Kernel。
5. Copy：从中间结果组装最终 CSR（rowOffsets、colIndices、values），更新 matC 描述符。

![image.png](https://raw.gitcode.com/user-images/assets/10331120/76983560-60b5-4e8f-860a-e1303481da97/image.png 'image.png')

Tiling 策略要点：

- 支持 dynamic shape：M、K、N、nnz(A)、nnz(B) 和中间乘积数量动态变化。
- 按行分组进行任务划分，处理长尾行分布（部分行中间乘积膨胀严重）。
- workspace 用于存储中间乘积，大小由 WorkEstimation/EstimateMemory 确定。

Tiling key 规划：

| Tiling Key | Kernel 模板 | 用途 |
| --- | --- | --- |
| 0 | `SpGEMMKernel<float16_t>` | float16 计算 |
| 1 | `SpGEMMKernel<bfloat16_t>` | bfloat16 计算 |
| 2 | `SpGEMMKernel<float>` | float32 计算 |
| 3 | `SpGEMMKernel<complex64_t>` | complex64 计算 |

#### AscendC kernel 侧设计

代码位置：`ops-sparse/` 中 SpGEMM Kernel 实现

核心计算逻辑：

对 C 的每一行 i，遍历 A 第 i 行的所有非零元素 A[i, k]，对每个 k 遍历 B 第 k 行的所有非零元素 B[k, j]，累加乘积到 C[i, j]。需完成：
1. 中间乘积生成：遍历 A 的行结构，展开与 B 对应行的乘积对。
2. 排序归并：对同一行的中间乘积按列索引排序，相同列索引的值累加合并。
3. 输出写回：生成最终 CSR 的 rowOffsets、colIndices、values。

Kernel 设计要点：

- 遵循 `Init → Process` 生命周期。
- `Init` 阶段绑定 GM 地址、读取 tiling 参数、初始化 UB buffer。
- `Process` 阶段按 tiling 划分处理分配的行范围。
- `complex64` 须正确处理复数乘加（实部/虚部交叉运算）、数值抵消及排序归并。
- 搬运使用 `DataCopy` / `DataCopyPad` 处理对齐。
- 输出 CSR 每行 colIndices 严格升序，同坐标重复项累加合并。
- 计算产生的显式零值保留并计入 `nnz(C)`。
![image.png](https://raw.gitcode.com/user-images/assets/10331120/10693327-bc07-47ab-bee1-c55c3a562f33/image.png 'image.png')
### 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Ascend 950PR | √ |

### 算子约束限制

- A、B 必须为二维稀疏矩阵，满足 `A.size(1) == B.size(0)`。
- A、B、C 均仅支持 CSR 格式（`aclsparseCreateCsr`）。
- `csrRowOffsetsType` 和 `csrColIndType` 均仅支持 `ACL_SPARSE_INDEX_32I` 且必须相同。
- 输入 A/B 的列索引必须有序，输出 C 的列索引必须有序；非法索引必须返回确定错误。
- `opA` 和 `opB` 均仅支持 `ACL_SPARSE_OP_NON_TRANSPOSE`；传入 TRANSPOSE 或 CONJUGATE_TRANSPOSE 必须返回明确错误。
- 数据类型：`float16`、`bfloat16`、`float32`、`complex64` 必须打通全链路。
- `complex64` 必须正确处理复数乘加、抵消、排序归并及声明支持的共轭语义。
- 输出 C 采用规范化 CSR 表示：`rowOffsets` 单调非降，每行 `colIndices` 严格升序，同坐标重复项累加合并，显式零值保留。
- 空输入（A/B 零 nnz、空行、空列、无交集乘积）必须正确处理。
- C++ 接口需使用调用方 stream 执行，禁止无必要的 Host 同步。
- 接口满足确定性计算要求。
- 不要求图融合。
- A 和 B 不进行矩阵维度广播。
- L2 float64 不进入 NPU kernel，仅支持上述四种类型。

## 特性交叉分析、可维可测分析

### 精度标准 / 性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | 混合容差单标杆方法验收；同时校验输出稀疏结构（rowOffsets、colIndices、nnz(C)精确一致）和 values（逐元素容差） | 《生态算子开源精度标准》及任务书 |
| 性能标准 | float16/bfloat16/float32 ≥ 1.0× GPU A100；complex64 ≥ 0.8× GPU A100 | 任务书要求 |

精度参数：

| dtype | rtol | atol | A (绝对误差硬上限系数) |
| --- | --- | --- | --- |
| float16 | 2^-9 | 2^-9 | 1e-1 |
| bfloat16 | 2^-6 | 2^-6 | 1e0 |
| float32 | 2^-10 | 2^-16 | 1e-2 |
| complex64 | 实部/虚部分别按 float32 参数 | 同 float32 | 同 float32 |

性能参考基准（GPU A100 NCU Kernel 总耗时）：

| 编号 | M×K×N | nnz(A) | nnz(B) | nnz(C) | dtype | GPU A100 耗时（μs） |
| --- | --- | --- | --- | --- | --- | --- |
| P-01 | 19717×19717×19717 | 78,868 | 78,868 | 315,472 | float32 | 289.088 |
| P-02 | 169343×169343×169343 | 1,185,401 | 1,185,401 | 8,297,807 | float16/bfloat16/float32 | 1127.296 / 1134.080 / 1121.376 |
| P-03 | 1048576×1048576×1048576 | 8,388,608 | 8,388,608 | 67,108,864 | float16/bfloat16/float32/complex64 | 6884.864 / 6876.512 / 6858.208 / 8059.968 |

### 可测性分析

- 测试目录：`ops-sparse/` 中 SpGEMM 测试目录
- 精度测试：CPU Golden 对比（float16/bfloat16 用 float32 计算 Golden，float32 用 float64，complex64 用 complex128）
- 性能测试：预热 10 次、正式采样 30 次，报告中位数及 90% 分位耗时
- 结构验证：rowOffsets、colIndices、nnz(C) 精确一致

建议覆盖场景：

| 编号 | 场景 | 覆盖点 |
| --- | --- | --- |
| TC-01 | 基础功能 - 方阵 | CSR 方阵，覆盖 float16/bfloat16/float32/complex64 |
| TC-02 | 基础功能 - 长矩阵 | M >> N 的矩形矩阵 |
| TC-03 | 基础功能 - 宽矩阵 | M << N 的矩形矩阵 |
| TC-04 | 稀疏边界 - nnz=0 | A 或 B 为零矩阵 |
| TC-05 | 稀疏边界 - nnz=1 | 最小非零输入 |
| TC-06 | 稀疏边界 - 空行/空列 | A/B 含空行或空列 |
| TC-07 | 稀疏边界 - 无交集乘积 | C 为零矩阵 |
| TC-08 | 稀疏边界 - 多项归并 | 同一输出位置多个中间乘积需累加 |
| TC-09 | 多阶段流程 | 完整 Create→Work→Estimate→Compute→Copy→Destroy 链路 |
| TC-10 | Workspace 异常 | workspace 不足场景 |
| TC-11 | 维度/dtype 不匹配 | 输入错误返回明确错误码 |
| TC-12 | ATen 端到端 | NPU dispatch 命中，无 CPU fallback |
| TC-13 | complex64 专项 | 复数乘加、抵消、共轭语义 |
| TC-14 | 泛化 - 大规模 | P-03 级别规模，性能验证 |

### 兼容性分析

- 复用现有 SpGEMM 社区任务的 `aclsparseSpGEMM*` 接口，不新增同名或同功能接口。
- A2/A3 与 A5 任务公共 Host 代码需合理解耦，确保同一主干共存。
- Python/ATen 接口语义以 PyTorch 2.7+ 为准，C++ 接口语义以 cuSPARSE 13.3 Update 1 为准。
- 合入路径 `ops-sparse`，接口声明位于 `include/cann_ops_sparse.h`。
