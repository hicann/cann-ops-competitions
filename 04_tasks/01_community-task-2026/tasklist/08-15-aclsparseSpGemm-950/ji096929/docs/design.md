# aclsparseSpGemm 算子设计文档（A5 · Ascend 950PR）

| 版本 | 日期 | 修改人 | 修改内容 |
|------|------|--------|----------|
| v1.0 | 2026.8.19 | ji096929 | 初稿（A5：complex64 补齐 + Python/ATen NPU 适配 + 测试闭环） |

---

# 一、需求背景（required）

## 1.1 需求来源

社区任务【aclsparseSpGemm 算子开发(950)】（A5）。参考 PyTorch 稀疏矩阵乘法
`aten::_sparse_sparse_matmul` 的接口行为，以及 NVIDIA cuSPARSE SpGEMM（CUDA
Toolkit 13.3 Update 1，§cusparseSpGEMM）的 C++ 多阶段接口语义，在昇腾
Ascend 950PR 上完成 Python/ATen 适配，并复用现有 SpGEMM 社区任务交付的
aclsparse C++ 多阶段接口及 Ascend C Kernel，补齐 `complex64` 全链路能力、
稀疏输出构造与测试。

## 1.2 背景介绍

### SpGEMM 计算

$$
C = \alpha \cdot op(A) \cdot op(B) + \beta \cdot C
$$

- A 为 shape `[M,K]` 稀疏矩阵，B 为 `[K,N]`，C 为 `[M,N]`；A/B/C 均 CSR；
- 输出 C 的稀疏结构、`nnz(C)` 和 values 由乘法结果确定；
- `opA`/`opB` 仅支持 `ACL_SPARSE_OP_NON_TRANSPOSE`。

### 现有 SpGEMM 社区任务交付能力

| 阶段 | 接口 | 现有能力 |
|---|---|---|
| 描述符 | `aclsparseSpGEMMCreateDescr` / `DestroyDescr` | 已交付 |
| 工作量估算 | `aclsparseSpGEMMWorkEstimation` | fp16/bf16/fp32 |
| 中间乘积查询 | `aclsparseSpGEMMGetNumProducts` | 已交付 |
| 内存估算 | `aclsparseSpGEMMEstimateMemory`（ALG2/ALG3） | fp16/bf16/fp32 |
| 计算 | `aclsparseSpGEMMCompute` | fp16/bf16/fp32 |
| 结果拷贝 | `aclsparseSpGEMMCopy` | fp16/bf16/fp32 |

### 本任务（A5）增量

1. **complex64 全链路**（必选）：描述符、多阶段接口、Ascend C Kernel、输出
   组装及测试全流程补齐；
2. **Python/ATen NPU 适配**：`torch.sparse.mm` → `aten::_sparse_sparse_matmul` /
   `_sparse_addmm` 的 SparseCsrPrivateUse1 注册，稀疏 Tensor 与描述符转换、
   `nnz(C)` 处理与输出 Sparse Tensor 构造，核心计算在 NPU 完成、无 CPU fallback；
3. **测试闭环**：ATK 精度 200 条（fp32/complex64）+ fp16/bf16 补充 100 条、
   C++ 多阶段 UT（含异常场景）、性能对标 A100 cuSPARSE NCU Kernel 总耗时。

## 1.3 交付位置

全部代码提交至 `ops-sparse` 仓 `master` 分支（https://gitcode.com/cann/ops-sparse），
沿用既有 SpGEMM 目录结构与 `include/cann_ops_sparse.h` 中的接口声明，不重复
新增同名接口。

---

# 二、需求分析（required）

## 2.1 需求描述

1. 实现 `aten::_sparse_sparse_matmul` 的 NPU 能力，参数、返回值、dtype、shape、
   device、稀疏 layout、异常行为与 PyTorch 2.7+ 一致；
2. 复用 `aclsparseSpGEMM*` 多阶段接口，新增 `complex64`（`ACL_COMPLEX64`）
   类型打通全链路；
3. 数据类型固定 `float16`/`bfloat16`/`float32`/`complex64`，A/B/C 及
   `computeType` 同类型，索引仅 `ACL_SPARSE_INDEX_32I`；
4. 算子泛化能力覆盖不同 shape、nnz、稀疏度、空行/空列、中间乘积膨胀、
   长尾行分布与合法边界输入。

## 2.2 需求拆解

| # | 拆解项 | 交付物 |
|---|---|---|
| 1 | complex64 描述符与多阶段 Host 通路 | `sparse/spgemm/arch35/spgemm_host.cpp` |
| 2 | complex64 Ascend C Kernel | `spgemm_kernel.cpp` `KernelSpgemmComplex` |
| 3 | Python NPU dispatch | `python/torch_sparse_spgemm/npu_dispatch.py` |
| 4 | ctypes 多阶段编排 | `python/torch_sparse_spgemm/spgemm.py` |
| 5 | C++ UT（正向 + 异常） | `test/spgemm/arch35/spgemm_test.cpp` |
| 6 | ATK 精度闭环 | 自写 executor/compare 插件 + 300 条用例 |
| 7 | 性能验证 | Profiler kernel 口径 vs A100 NCU 基线 |

---

# 三、详细设计（required）

## 3.1 算子分析

### 数学公式

$$
C = \alpha \cdot A \times B + \beta \cdot C_{in}, \quad A\in\mathbb{C}^{M\times K},\ B\in\mathbb{C}^{K\times N}
$$

complex64 乘加按 `(a+ai·i)(b+bi·i) = (ab-ai·bi) + (ab·bi+ai·b)i` 复数四则执行；
实型按标量乘加。累加路径保持确定性（固定归并顺序）。

### 支持数据类型

`float16`（`ACL_FLOAT16`）、`bfloat16`（`ACL_BF16`）、`float32`（`ACL_FLOAT`）、
`complex64`（`ACL_COMPLEX64`）；A/B/C/`computeType` 同类型。

### 支持形状

`[M,K] × [K,N] -> [M,N]`，`A.size(1) == B.size(0)`，不广播；dynamic shape 由
Host 侧多阶段内存及 tiling 规划承载。

## 3.2 实现方案

### 3.2.1 C++ 多阶段接口（复用 + complex64 补齐）

调用流程与 cuSPARSE 对齐，状态机为 `created(0) -> work_estimated(1) ->
memory_estimated(2) -> computed(3) -> copied(4)`：

```
CreateDescr -> WorkEstimation -> GetNumProducts -> [EstimateMemory(ALG2/3)]
    -> Compute -> Copy -> DestroyDescr
```

- `ValidateSpgemmInputs`：opA/opB 仅 NON_TRANSPOSE（否则 `NOT_SUPPORTED`）、
  CSR/int32/zero-based、四类型同精度、维度匹配（否则 `INVALID_VALUE`）；
- `SpgemmSymbolPhase`（Host 符号阶段）：D2H 取回 A/B 结构，产出 C 的
  rowPtr/colInd（每行严格升序去重）、`nnzC` 与中间乘积数 `numProducts`；
  同时校验非法索引（负值/越界/非升序返回 `INVALID_VALUE`）；
- complex64 的 alpha/beta 标量按 `{real, imag}` 两个 float 拆传。

### 3.2.2 Kernel 侧设计（Ascend C，arch35 / dav-3510）

- 四个 kernel 入口：`spgemm_custom_{fp32,fp16,bf16,complex64}`，按 dtype 分派；
- 实型走 `KernelSpgemmReal<T, AccT>` 模板（fp16/bf16 转 float 累加，fp32 原生）；
  complex64 走独立 `KernelSpgemmComplex`：每个输出元素一个独立 SIMT 工作单元，
  复数乘加后按 Host 符号阶段给定的目标槽位写入（归并顺序固定，结果确定）；
- 工作分配：行区间内所有输出元素扁平化为独立工作单元，512 个 SIMT 线程
  stride 遍历，充分利用 AIV 向量核并行度（相比"每行 1 线程"基线提升
  50×+ kernel 吞吐）；
- workspace 布局：`[TilingData | rowPtrA | colIndA | valsA | rowPtrB | colIndB |
  valsB | rowPtrC | colIndC | valsC | valsCIn]`，偏移由 `spgemm_ws_offsets`
  按 dtype 元素宽度统一计算（complex64 8 字节）。

### 3.2.3 Python/ATen NPU 适配

`torch.sparse.mm(A, B)` 的 CSR 输入在 NPU 上走 SparseCsrPrivateUse1 dispatch：

```
torch.sparse.mm -> _sparse_mm -> _sparse_addmm/_sparse_sparse_matmul
    -> SparseCsrPrivateUse1 IMPL（本实现）
    -> spgemm_csr()（ctypes 调 libops_sparse.so 多阶段接口）
    -> aclsparseSpGEMM{CreateDescr,WorkEstimation,GetNumProducts,
       EstimateMemory,Compute,Copy} + SpMatGetSize + DestroyDescr
    -> torch.sparse_csr_tensor(crow, col, values)
```

- `register_npu_sparse_matmul()`：`torch.library.Library("aten", "IMPL",
  "SparseCsrPrivateUse1")` 注册 `_sparse_sparse_matmul` / `_sparse_addmm`；
- 非 CSR 输入抛 `NotImplementedError` 并提示先 `to_sparse_csr()`（不静默转换、
  不回退 CPU）；非连续 crow/col/values 抛 `TypeError`（明确错误，无隐式拷贝）；
- ACL 运行时复用 torch_npu 已建立的 context/stream（`AclRuntime` 单例）。

### 3.2.4 输出稀疏结构

- 规范化 CSR：`rowOffsets` 单调非降、每行 `colIndices` 严格升序、无重复坐标；
- 同一坐标重复项累加合并为一个条目；计算产生的显式零值保留并计入
  `nnz(C)`；
- Python 层返回 CSR（coalesced 语义），`nnz(C)` 经 `aclsparseSpMatGetSize`
  查询后按实际长度切片返回。

## 3.3 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Ascend 950PR（dav-3510 / arch35） | √ |

## 3.4 算子约束限制

- 仅 CSR（`aclsparseCreateCsr`），不适用稠密 Row-major/Column-major 参数；
- 索引仅 `ACL_SPARSE_INDEX_32I` 且 A/B/C 一致；输入列索引须有序，输出有序；
  非法索引返回确定错误；
- `opA`/`opB` 仅 `ACL_SPARSE_OP_NON_TRANSPOSE`，TRANSPOSE/
  CONJUGATE_TRANSPOSE 返回明确错误；
- `alpha`/`beta`：实型单 float，complex64 为 `{real, imag}`；
- 规模：M/K/N/nnz 及中间乘积数量上限为 int32 表示范围，超限返回
  `NOT_SUPPORTED`；
- 异步执行：使用调用方 stream（`aclsparseSetStream`），无多余 Host 同步；
- 确定性：同输入多次执行输出结构与 values bit-wise 一致。

---

# 四、可维可测分析（required）

## 4.1 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | 《生态算子开源精度标准》混合容差单标杆：fp16 rtol=atol=2⁻⁹/A=1e-1；bf16 2⁻⁶/A=1e0；fp32 rtol=2⁻¹⁰/atol=2⁻¹⁶/A=1e-2；complex64 实虚部各按 fp32 参数；CPU Golden fp16/bf16→fp32、fp32→fp64、complex64→complex128；结构与 `nnz(C)` 精确一致 | 生态算子开源精度标准（experimental_standard） |
| 性能标准 | NCU/Profiler Kernel 总耗时口径：fp16/bf16/fp32 ≥ 1.0× A100 cuSPARSE，complex64 ≥ 0.8×；预热 10 次、采样 30 次 | 任务书性能表 P-01/02/03 |

## 4.2 测试设计

| 类别 | 覆盖 | 工具 |
| --- | --- | --- |
| ATK 精度 | fp32×100 + complex64×100（原 testCase）+ fp16×50 + bf16×50（补充），结构精确比 + values 混合容差 | 自写 executor/compare 插件，`atk task` |
| C++ 多阶段 UT | 四 dtype 基础、ALG1/2/3 全阶段、空矩阵/空行、随机矩阵、alpha/beta、beta·C_in；异常：transpose 拒绝、维度不匹配、索引非法、workspace 缺失、阶段顺序、输入只读、nnz=0 输出边界（37 用例） | `spgemm_test` |
| 性能 | P-01/02/03 专项 + 50 条随机用例，`kernel_details.csv` 口径 kernel 总耗时 vs A100 NCU 基线 | `profile_sparse_ops_npu.py` + `benchmark_sparse_ops_npu.py` |

## 4.3 兼容性分析

复用现有 SpGEMM 社区任务接口与目录，不新增同名接口；A2/A3（Ascend 310P 等）
与 A5（950PR）的公共 Host 逻辑经 `arch35` 目录隔离，硬件差异在 kernel 分派
层解耦，可共存于同一主干。
