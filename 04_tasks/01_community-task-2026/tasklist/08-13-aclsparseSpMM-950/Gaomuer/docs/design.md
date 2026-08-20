# 需求背景（required）

## 需求来源

本设计对应 2026 年 8 月社区任务《aclsparseSpMM 算子开发（950）任务书》。目标是在 Ascend 950PR 上实现 `torch.sparse.addmm` / `aten::_sparse_addmm` 的 NPU 能力，并复用、扩展 `ops-sparse` 仓库已有的 `aclsparseSpMM*` C++ 接口、Host 实现和 Ascend C Kernel。

设计依据如下：

1. 任务书：`aclsparseSpMM_A5_task_doc.md`。
2. 任务附件：`aclsparseSpMM_testCase/` 中的 200 个正确性/精度用例、50 个性能用例、CPU Golden、ATK 精度插件、NPU Event/Profiler 脚本及 A100 NCU 基线。
3. 设计文档模板：`cann-ops-competitions/04_tasks/01_community-task-2026/resources/design_template.md`。
4. 目标代码仓：`https://gitcode.com/cann/ops-sparse`，目标分支为 `master`。
5. Python/ATen 语义：PyTorch 2.7 及以上版本的 `torch.sparse.addmm` 和 `aten::_sparse_addmm`。
6. 950PR 软件基线：与目标 `ops-sparse` commit/tag 配套的 CANN 9.0.0 或后续指定版本，以及对应的 `Ascend-cann-950-ops`、driver 和 firmware；最终以开源仓配套关系为准，不混用 CANN 8.5 头文件或运行库。
7. C++ 接口语义及能力矩阵：CUDA Toolkit 13.3 Update 1 所含 cuSPARSE 13.3 SpMM 文档。
8. 精度判定：《生态算子开源精度标准》的混合容差单标杆方法。

本文只描述设计和验证方案。开发、实测数据、提交信息及评审结论将在代码开发和 Ascend 950PR 实测完成后补充，不在设计阶段预填结果。

## 背景介绍

### aclsparseSpMM 算子功能

SpMM 完成 CSR 稀疏矩阵与稠密矩阵的乘加：

```text
out = beta * input + alpha * (mat1 * mat2)
```

其中 `input`、`mat2` 和 `out` 为稠密 Tensor，`mat1` 为二维 CSR Tensor。Python 入口是：

```python
torch.sparse.addmm(input, mat1, mat2, *, beta=1, alpha=1) -> Tensor
```

ATen Schema 是：

```text
aten::_sparse_addmm(
    Tensor self,
    Tensor mat1,
    Tensor mat2,
    *,
    Scalar beta=1,
    Scalar alpha=1
) -> Tensor
```

底层复用以下公开接口，不重复新增同名或同功能接口：

```cpp
aclsparseSpMMGetBufferSize(...);
aclsparseSpMMPreprocess(...);
aclsparseSpMM(...);
```

同时复用 `aclsparseCreate`、`aclsparseSetStream`、`aclsparseSetPointerMode`、`aclsparseSetWorkspace`、`aclsparseGetWorkspace`、`aclsparseCreateCsr`、`aclsparseCreateConstCsr`、`aclsparseCreateDnMat`、`aclsparseCreateConstDnMat` 及相应资源销毁接口。

### ops-sparse 现有实现分析

设计阶段基于 `ops-sparse master` 提交 `ab9c8cf2fc4b7af6dc17c42688c53eed6939f565`（2026-08-17）完成基线分析。现有 SpMM 源码已经位于正确的 Ascend 950 架构目录：

```text
sparse/spmm/arch35/
├── spmm.h
├── spmm_host.cpp
├── spmm_kernel.cpp
├── spmm_csr_mat.h
└── spmm_csr_mat.cpp

test/spmm/
├── CMakeLists.txt
└── arch35/spmm_test.cpp
```

现有实现可复用的能力包括：

- 三段式 `GetBufferSize -> Preprocess -> SpMM` 调用流程。
- CSR 行重排、分桶以及 workspace 中 tiling/reorder/bin-edge 的组织方式。
- Ascend 950PR `arch35` / `dav-3510` 的 SIMT AIV Kernel 框架。
- FP32、FP16、INT8 的基础 Kernel 模板，FP16 使用 FP32 累加。
- B/C Row-major、Column-major 地址计算及 `opB=N/T` 的部分实现。
- `beta=0` 时不读取 C 的 Kernel 分支。
- 运行时查询 AIV Core 数量、按调用方 stream 下发 Kernel 的框架。

基线与本任务的差距如下，均属于本任务必须补齐的范围：

| 层级 | 现状 | 本任务要求 |
| --- | --- | --- |
| Python/ATen | 仓库当前无 `torch.sparse.addmm` NPU 注册 | 增加 `aten::_sparse_addmm` NPU 适配、校验、转换、输出和端到端测试 |
| dtype | FP32、FP16、INT8 | 必须打通 FP16、BF16、FP32、Complex64；Complex64 不是可选项 |
| 算法 | DEFAULT、CSR_ALG1、FP32 高精度扩展 | 在保持已有枚举 ABI 的前提下补齐 CSR_ALG2、CSR_ALG3 及合法组合 |
| 操作 | `opA=N`、`opB=N/T` | 对齐任务书声明的合法 `opA/opB` 组合；ALG3 限制为 `opA=N`、`opB!=H` |
| 索引基址 | 仅 base 0 | int32 索引下支持 base 0 和 base 1 |
| 布局和 ld | 已有部分地址路径，校验不完整 | B/C Row/Column-major、最小 ld、padding ld、转置后的逻辑维度均需完整校验 |
| 标量指针 | Host float/int 读取为主 | 明确 Host/Device pointer mode，支持四种任务 dtype，避免无必要同步 |
| 动态与边界 | M/N 受 int32 限制，零维和错误边界不完整 | 动态 M/K/N/nnz、零维、nnz=0、空行、溢出和 workspace 边界 |
| 泛化与性能 | 单一 SIMT 行分桶路径 | 针对短行、长尾行、布局、dtype 和 N 宽度进行自适应分桶/分块 |
| 多架构 | 当前 SpMM 只有 arch35 | 公共逻辑与硬件实现解耦，允许后续 arch22（A2/A3）与 arch35 共存 |
| 测试 | C++ FP32/FP16/INT8 基础样例 | 四 dtype、Python/ATen、C++ API、Kernel、错误、精度、性能和 Profiler 全覆盖 |

### 任务附件测试现状分析

任务附件不是完整验收集合，设计中必须保留附件并补齐缺口：

- `sparse_addmm_accuracy.json` 共 200 例：FP32 100 例、Complex64 100 例。
- `sparse_addmm_performance.json` 共 50 例：FP32 25 例、Complex64 25 例，shape 范围主要为 1000 到 20000，degree 为 2 到 64。
- 正确性脚本已覆盖普通值、小值、离群值、正负抵消、NaN/Inf，alpha/beta 三组取值，标量/行广播/完整 input，以及部分非连续 Tensor 和周期性空行。
- 附件的 NPU Event 脚本默认预热 10 次、正式采样 30 次，可输出 median 和 P90，作为端到端补充数据。
- 附件的 NPU Profiler 脚本当前固定为预热 5 步、active 5 步，并把 `op_statistic.csv` 总时间除以 5，只能形成初步 Kernel 证据，不能直接满足正式性能验收的“预热不少于 10 次、采样不少于 30 次、报告 median/P90”。
- 附件没有 FP16/BF16 数据集，没有任务书 P-01/P-02/P-03 全量 dtype 的正式性能 case，也没有 C++ API、workspace、布局/算法、index base、错误码和资源生命周期测试。

因此，本设计将附件 200+50 用例作为不可删减的基础集合，并增加 FP16/BF16、官方三组性能 case、C++ 接口和边界专项测试；同时改造 Profiler 采集逻辑，使正式报告真正满足任务书统计口径。

# 需求分析（required）

## 需求描述

在 Ascend 950PR 上复用并扩展 `ops-sparse` 现有 SpMM，完成从 Python/ATen 到 aclsparse Host API、Ascend C Kernel 的全链路实现。功能、shape、dtype、device、标量、广播、异常和别名行为对齐 PyTorch 2.7 及以上版本；C++ 三段式接口、描述符、workspace、算法、布局、操作、pointer mode 和生命周期对齐任务书约定及 cuSPARSE 13.3 SpMM 的对应语义。

实现必须满足以下总原则：

1. 核心计算全部在 NPU 执行，不允许以 CPU fallback 代替实现。
2. 仅支持 CSR 稀疏格式；CSR row offsets 和 column indices 均为 int32 且类型相同，支持 index base 0/1，不支持 int64。
3. 必须支持 FP16、BF16、FP32、Complex64 四种 dtype，且打通 Python、ATen、C++ API 和 Kernel；Tensor 间不做 dtype promotion。
4. `input` 可广播到 `[M, N]`；`mat1` 和 `mat2` 不做矩阵广播。
5. 支持任务书声明范围内的非连续 Tensor、B/C Row/Column-major、leading dimension padding、动态 M/K/N/nnz 和合法空输入。
6. 复用现有接口和目录，不新增重复 API；Ascend 950 代码继续位于 `arch35`。
7. 公共校验、描述符语义和硬件专用 Kernel 解耦，使 A2/A3 与 A5 实现可在同一主干共存。

## 需求拆解

1. Python/ATen：注册 `aten::_sparse_addmm` 的 NPU CSR 实现，完成输入校验、广播、布局识别/必要的 NPU contiguous、输出构造和错误转换。
2. C++ API：保持三个 `aclsparseSpMM*` 函数签名兼容，补齐四 dtype、CSR_ALG2/ALG3、base 1、布局、操作、pointer mode、workspace 和生命周期语义。
3. Host：建立统一参数规格对象，完成无溢出的逻辑维度推导、支持矩阵判定、workspace 规划、预处理缓存和 arch35 tiling 生成。
4. Kernel：在现有 dav-3510 SIMT AIV 框架上增加 BF16、Complex64、算法分支、转置/共轭、base 1、零维和长尾泛化路径。
5. 正确性：覆盖四 dtype、广播、非连续、布局/操作/算法、零维/零 nnz/空行、标量、异常、alias、确定性和无 CPU fallback。
6. 精度：严格使用任务书规定的单一 CPU Golden、混合容差、匹配率及绝对误差硬上限，Complex64 的实部和虚部分别判定。
7. 性能：完成附件 50 例及 P-01/P-02/P-03；Kernel 总耗时为主指标，端到端和 C++ 阶段为补充；满足预热、样本量、median/P90 和 A100 倍率要求。
8. 工程：补充 README、API 限制、C++ UT、端到端 UT、测试步骤、自测报告数据来源和 950PR 环境记录。

# 详细设计（required）

## 算子分析

### 数学公式

对于 CSR 矩阵 `A=mat1`、稠密矩阵 `B=mat2`、稠密输入 `D=input`：

```text
C[i, j] = beta * D_broadcast[i, j]
          + alpha * sum(A[i, p] * B[p, j])
```

C++ 接口的一般形式为：

```text
C = alpha * op(A) * op(B) + beta * C
```

Python 接口固定为 `opA=N`、`opB=N`，输出为 Row-major `[M,N]`。C++ 接口按合法能力矩阵支持 `opA/opB` 和 B/C order。

### 支持数据类型

| Python dtype | A/B/C valueType | computeType | 累加类型 | 输出 dtype |
| --- | --- | --- | --- | --- |
| `torch.float16` | `ACL_FLOAT16` | `ACL_FLOAT` | FP32 | FP16 |
| `torch.bfloat16` | `ACL_BF16` | `ACL_FLOAT` | FP32 | BF16 |
| `torch.float32` | `ACL_FLOAT` | `ACL_FLOAT` | FP32 | FP32 |
| `torch.complex64` | `ACL_COMPLEX64` | `ACL_COMPLEX64` | Complex64（实部/虚部均 FP32） | Complex64 |

`ACL_COMPLEX64` 的公开存储和标量 ABI 固定为 8 字节、先实部后虚部的两个连续 FP32。Host 侧使用显式 POD 类型并做编译期检查：

```cpp
struct alignas(8) SpmmComplex64Scalar {
    float real;
    float imag;
};
static_assert(sizeof(SpmmComplex64Scalar) == 8);
static_assert(alignof(SpmmComplex64Scalar) == 8);
```

Device 侧优先使用 CANN 9.0 为 Atlas 350/Ascend 950PR 提供的 Ascend C `complex64` 内置类型；若 SIMT 指令限制要求拆分计算，则只在 Kernel 内部转换为 `{float real, float imag}`，外部 GM 布局保持上述 ABI。C++ UT 对 Tensor、Host/Device pointer mode 标量和 GM 元素分别检查大小、对齐、实虚顺序及共轭行为，禁止直接假设 `std::complex<float>` 的实现布局。

Python 的 `input`、`mat1.values()` 和 `mat2` 必须是同一 dtype，输出沿用该 dtype，不进行 Tensor 间隐式提升。标量在进入 C++ 接口前转换为共同计算类型：

- 实数 Tensor 接受整数或实数 alpha/beta；虚部非零的复数标量报错。
- Complex64 接受实数或复数 alpha/beta，并转换为 Complex64。
- `beta=0` 仍校验 input 的 dtype、shape 和 device，但不得读取 input 数值，因此其中的 NaN/Inf 不传播。

### 支持形状和布局

Python 逻辑形状：

| 对象 | 形状 | 说明 |
| --- | --- | --- |
| `input` | 可广播到 `[M,N]` | 支持标量、`[N]`、`[1,N]`、`[M,1]`、`[M,N]` 等合法广播形状 |
| `mat1` | `[M,K]` | 二维 CSR，values 为四种任务 dtype，索引为 int32 |
| `mat2` | `[K,N]` | 二维 Dense，可直接描述为行主/列主时零拷贝，否则在 NPU 生成连续副本 |
| `output` | `[M,N]` | 二维 Dense，Python 路径输出 Row-major |

C++ 逻辑维度由操作类型推导，而不是直接使用存储 shape：

```text
opA=N: M=A.rows, K=A.cols
opA=T/H: M=A.cols, K=A.rows

opB=N: B.rows=K, B.cols=N
opB=T/H: B.rows=N, B.cols=K

C.rows=M, C.cols=N
```

稠密描述符支持：

- Row-major：`ld >= stored_cols`。
- Column-major：`ld >= stored_rows`。
- 使用 64 位整数完成 `ld * rows/cols`、workspace 和地址范围计算，校验后再转换为 Kernel 所需类型。
- Python 非连续 Tensor 若是无重叠且 stride 可等价表达为 Row/Column-major + ld，则直接建描述符；切片间隔、重叠、负 stride 或其他无法表达的布局，在 NPU 上生成连续副本，不回到 CPU。

### C++ 能力矩阵

任务只要求 CSR，不扩展 COO/CSC/BSR。算法设计如下：

| 算法 | 格式 | 推荐布局 | opA | opB | 确定性 | Preprocess |
| --- | --- | --- | --- | --- | --- | --- |
| `ACL_SPARSE_SPMM_ALG_DEFAULT` | CSR | 根据 B/C order 选择 | 合法组合 | 合法组合 | 跟随所选算法 | 跟随所选算法 |
| `ACL_SPARSE_SPMM_CSR_ALG1` | CSR | Column-major | N/T/H | N/T/H | 混合容差验收 | 可选，重复执行时建议使用 |
| `ACL_SPARSE_SPMM_CSR_ALG2` | CSR | Row-major | N/T/H | N/T/H | 混合容差验收 | 可选；本实现可复用通用元数据 |
| `ACL_SPARSE_SPMM_CSR_ALG3` | CSR | Row/Column-major | 仅 N | N/T，不支持 H | bit-wise | 可选，重复执行时建议使用 |
| `ACL_SPARSE_SPMM_CSR_FP32_HIGH_PRECISION_ALG` | CSR | 保持已有兼容行为 | 按已有兼容范围 | 按已有兼容范围 | 固定归约顺序 | 可选 |

说明：

1. DEFAULT 在 Row-major Python 常用路径选择 CSR_ALG2，在 Column-major 路径选择 CSR_ALG1；开启 PyTorch deterministic algorithms 时选择 CSR_ALG3，若组合不受 ALG3 支持则返回明确错误。
2. ALG1/ALG2 的 `opA=T/H` 通过预处理生成转置访问元数据，不修改原始 CSR。
3. Complex64 的 `H` 对访问到的复数值执行共轭；实数 dtype 的 H 与 T 等价。
4. 未在表中声明的格式、dtype、操作和算法组合返回 `ACL_SPARSE_STATUS_NOT_SUPPORTED`，不静默降级到其他不等价能力。
5. 在公开枚举末尾追加 ALG2/ALG3，保留现有 DEFAULT、ALG1 和 FP32 高精度枚举的数值，避免破坏 ABI。

## 算子实现

### 总体实现方案

```text
torch.sparse.addmm
        |
        v
aten::_sparse_addmm (SparseCsrPrivateUse1/NPU registration)
        |
        +-- 参数、dtype、device、shape、CSR 元数据校验
        +-- input 广播/非连续处理（NPU）
        +-- mat2 stride -> DnMat order/ld，必要时 NPU contiguous
        |
        v
aclsparse handle + CSR/DnMat descriptors + plan/workspace cache
        |
        +-- aclsparseSpMMGetBufferSize
        +-- aclsparseSpMMPreprocess (new pattern or invalidated plan)
        +-- aclsparseSpMM (caller current NPU stream)
        |
        v
arch35 Host tiling -> dav-3510 Ascend C/SIMT AIV Kernel
        |
        v
Dense NPU Tensor [M, N]
```

核心计算不包含任何 `.cpu()`、Host 稠密计算或 CPU fallback。Host 只处理描述符元数据和 Host pointer mode 标量；CSR 结构检查与预处理使用同一 NPU stream 上的设备 Kernel，只有为返回确定非法索引错误而读取设备状态时进行必要同步。

cuSPARSE 仅用于对齐公开接口语义、合法能力矩阵和 A100 性能基线；NPU 实现只使用 aclsparse、ACL Runtime 和 Ascend C，不引入 CUDA/cuSPARSE 代码或运行时依赖。

### 代码组织

计划在实现时修改或新增以下文件；若开始开发时上游 master 已调整 PyTorch bridge 目录，则遵循届时仓库同类模块位置，但 SpMM 核心仍复用下列现有目录：

```text
include/cann_ops_sparse.h                         # 追加 ALG2/ALG3，完善 API 注释
sparse/spmm/README.md                             # 能力、调用、限制和性能说明
sparse/spmm/arch35/spmm.h                         # arch35 tiling/workspace/类型定义
sparse/spmm/arch35/spmm_host.cpp                  # 公共校验调用、workspace、tiling、launch
sparse/spmm/arch35/spmm_csr_mat.{h,cpp}           # CSR 校验、重排、分桶、转置元数据
sparse/spmm/arch35/spmm_kernel.cpp                # FP16/BF16/FP32/Complex64 Kernel
test/spmm/CMakeLists.txt                          # 复用测试注册
test/spmm/arch35/spmm_test.cpp                    # C++ API/Kernel 参数化 UT
test/spmm/README.md                               # 950PR 编译和复现步骤

pytorch/CMakeLists.txt                            # 必选交付的 PyTorch bridge 构建目标
pytorch/sparse_addmm_npu.cpp                      # ATen NPU 注册和适配
python/pyproject.toml                             # wheel 构建、依赖和扩展安装入口
python/ops_sparse_torch/__init__.py               # 加载 _C 注册库的标准 Python 包入口
python/ops_sparse_torch/_version.py               # torch/torch_npu/CANN 兼容版本检查
test/python/test_sparse_addmm.py                  # Python/ATen E2E、异常和 dispatch 测试
test/python/aclsparseSpMM_testCase/               # 任务附件及补充用例/采集脚本
```

PyTorch bridge 是本任务必选交付件，不作为“可选功能”验收。它编译为 `ops_sparse_torch._C` 独立共享库并链接目标版本的 LibTorch、torch_npu runtime 与 `libops_sparse`；wheel/安装包同时安装 Python 包和 `_C`。任务构建、CI 和验收统一启用 `BUILD_PYTORCH_BRIDGE=ON`，缺少匹配的 PyTorch/torch_npu 依赖时配置阶段直接失败。仅供底层 C API 开发的非验收构建可显式关闭 bridge，从而不让 `libops_sparse` 本身反向依赖 Python。

标准使用方式为先加载后端和本包，再调用原生 PyTorch API：

```python
import torch
import torch_npu
import ops_sparse_torch  # import 时加载 ops_sparse_torch._C 并完成 ATen 注册

out = torch.sparse.addmm(input, mat1, mat2, beta=beta, alpha=alpha)
```

该加载方式写入安装 README、端到端 UT 和任务附件执行器，不只在 ATK 内部增加不可见的临时加载逻辑。`ops_sparse_torch` 导入时校验 PyTorch >=2.7、torch_npu >=26.0.0 及扩展编译 ABI；版本不兼容时给出明确错误。若 SIG 要求由 `import torch_npu` 自动发现外部扩展，则在编码前确认其官方插件发现机制并替换入口，禁止使用 `.pth` 自动执行或修改系统 `sitecustomize.py`。

### Python/ATen 侧设计

#### NPU 注册

使用 PyTorch 2.7 的 `SparseCsrPrivateUse1` dispatch key 注册 `aten::_sparse_addmm` 的 CSR 实现，注册代码采用标准 dispatcher 机制：

```cpp
TORCH_LIBRARY_IMPL(aten, SparseCsrPrivateUse1, m) {
    m.impl("_sparse_addmm", TORCH_FN(sparseAddmmNpu));
}
```

`_C` 必须在 `torch_npu` 初始化后加载。构建时包含目标 PyTorch/torch_npu 公开头文件，不复制 torch_npu 内部私有 ABI；链接选项沿用其扩展构建规范。注册库加载后，通过 `_dispatch_dump_table("aten::_sparse_addmm")` 或等价 C++ dispatcher 查询，确认 `SparseCsrPrivateUse1` 指向 `sparseAddmmNpu`，并增加重复导入、重复注册冲突和未加载扩展的负向测试。普通端到端 UT 必须使用安装后的包执行，不能依赖源码目录碰巧出现在 `PYTHONPATH`。

前向函数只实现任务要求的 CSR x Dense -> Dense。PyTorch 既有 `_sparse_addmm` autograd 公式继续负责梯度图组合，不在本任务中复制一份不一致的求导定义；增加小规模 backward smoke test，确认注册没有截断既有 autograd 行为。任务书验收的主要范围仍为前向正确性和性能。

#### 参数校验顺序

校验顺序固定，保证错误稳定且不在错误输入上提前访问数据：

1. `input/mat1/mat2` 均 defined，且都在 NPU。
2. 三个 Tensor 位于同一 NPU device；跨设备立即报错。
3. `mat1.layout == torch.sparse_csr` 且 `mat1.dim()==2`；不把 COO/CSC 自动转 CSR。
4. `mat2.dim()==2`；推导 M/K/N 并校验 `mat1.size(1)==mat2.size(0)`。
5. 校验 `input` 可按 PyTorch 广播规则扩展到 `[M,N]`。
6. 三个 Tensor dtype 完全一致且属于 FP16/BF16/FP32/Complex64。
7. CSR crow/col 均为 int32，dtype 相同；检查长度、首尾、单调性、列索引范围和最终 nnz。
8. alpha/beta 可转换到 compute dtype；实数 Tensor 遇到虚部非零的复数标量时报错。
9. 检查规模、leading dimension、地址计算和 workspace 计算无溢出。

不把 int64 CSR 索引静默转换为 int32，因为转换既改变任务书接口约束，也可能溢出；直接返回明确的 dtype 错误。

#### input 广播和 beta 语义

- `beta != 0`：先分配 Row-major 输出，通过 NPU expand/copy Kernel 把 `input` 广播到 `[M,N]`，再将输出作为 C 传给 `aclsparseSpMM`。
- `beta == 0`：完成 input 的 shape/dtype/device 校验后直接分配输出，不执行 input copy，SpMM Kernel 覆盖全部有效输出元素并禁止读取 C 旧值。
- M 或 N 为 0：返回正确 shape 的空输出，不下发无意义 Kernel。
- 完成 CSR 合法性校验后，K 为 0 的合法矩阵必然满足 nnz=0；任意 nnz=0 情况的结果只取决于 `beta * input`。`beta=0` 时生成全零（或直接由专用 fill Kernel 写零），不能传播 input 中的 NaN/Inf。K=0 但 nnz 非零必须先作为非法 CSR 报错，不能进入空矩阵快速路径。

Python 输出不与 input、mat1 values 或 mat2 共用 storage，符合函数式接口语义。C++ API 中 matC 是合法的 in/out；如果 C 与 A/B 的内存区间发生不安全重叠，则返回 `INVALID_VALUE`，不发生未声明覆盖。

#### 非连续 Tensor

对 input 和 mat2 分别处理：

- input 的广播和 copy 使用 TensorIterator/NPU copy 路径，支持任务书列出的非连续视图。
- mat2 为二维、无重叠且可表示为 Row-major/Column-major + ld 时，直接创建 DnMat 描述符，保留其 stride。
- 其他合法非连续 mat2 在当前 NPU stream 上生成连续副本；副本生命周期覆盖异步 SpMM，并通过 allocator `recordStream` 或等价机制防止提前释放。
- 不支持的重叠视图或非法 stride 返回确定错误，不访问 CPU。

#### 描述符和计划缓存

为满足正式性能采样期间“描述符、workspace、preprocess 结果复用”，PyTorch bridge 建立有界、按 device 和 stream 隔离的 SpMM plan cache：

```text
key = device + stream + A TensorImpl identity
      + crow/col data_ptr + crow/col version
      + stored shape + nnz + index base
      + dtype + op/order/alg
```

- cache entry 持有 aclsparse handle/descriptor、workspace、preprocess 元数据和必要的弱 Tensor 引用。
- A values 指针、B/C 指针、alpha、beta 允许在调用间变化；每次执行前更新对应值指针。
- crow/col 指针、版本、shape、nnz、算法或 order 改变时使 plan 失效并重新 preprocess。
- active buffer 不跨 stream 并发复用；每个 stream 使用独立 entry，避免竞态。
- cache 使用固定容量 LRU，退出或淘汰时按 `DnMat -> SpMat -> Handle -> workspace` 的安全顺序释放；异步资源由 stream event/allocator 记录保证完成后回收。
- 性能脚本构造一次输入并重复调用，因而稳定命中同一 cache entry；Profiler 需证明采样段内不重复执行 preprocess。

### C++ 接口与生命周期设计

#### 接口原型和参数语义

保持 `include/cann_ops_sparse.h` 的现有函数签名：

```cpp
aclsparseStatus_t aclsparseSpMMGetBufferSize(
    aclsparseHandle_t handle,
    aclsparseOperation_t opA,
    aclsparseOperation_t opB,
    const void *alpha,
    aclsparseConstSpMatDescr_t matA,
    aclsparseConstDnMatDescr_t matB,
    const void *beta,
    aclsparseDnMatDescr_t matC,
    aclDataType computeType,
    aclsparseSpMMAlg_t alg,
    size_t *bufferSize);

aclsparseStatus_t aclsparseSpMMPreprocess(
    aclsparseHandle_t handle,
    aclsparseOperation_t opA,
    aclsparseOperation_t opB,
    const void *alpha,
    aclsparseConstSpMatDescr_t matA,
    aclsparseConstDnMatDescr_t matB,
    const void *beta,
    aclsparseDnMatDescr_t matC,
    aclDataType computeType,
    aclsparseSpMMAlg_t alg,
    void *externalBuffer);

aclsparseStatus_t aclsparseSpMM(
    aclsparseHandle_t handle,
    aclsparseOperation_t opA,
    aclsparseOperation_t opB,
    const void *alpha,
    aclsparseConstSpMatDescr_t matA,
    aclsparseConstDnMatDescr_t matB,
    const void *beta,
    aclsparseDnMatDescr_t matC,
    aclDataType computeType,
    aclsparseSpMMAlg_t alg,
    void *externalBuffer);
```

| 参数 | 内存位置 | 读写属性 | 生命周期/说明 |
| --- | --- | --- | --- |
| `handle` | Host | 输入 | 在三个阶段内有效，携带 stream 和 pointer mode |
| `opA/opB` | Host | 输入 | 三个阶段的逻辑必须兼容 active plan |
| `alpha/beta` | Host 或 Device | 输入 | 位置由 pointer mode 决定；执行完成前保持有效 |
| `matA` | Host descriptor + Device arrays | 只读输入 | crow/col pattern 在 active preprocess 期间不得改变；values 可以更新 |
| `matB` | Host descriptor + Device values | 只读输入 | 执行完成前有效，values 可在不同执行间更新 |
| `matC` | Host descriptor + Device values | 输入/输出 | 执行完成前有效，保存 `beta*C + alpha*op(A)*op(B)` |
| `computeType` | Host | 输入 | 与 A/B/C dtype 组合必须位于支持矩阵 |
| `alg` | Host | 输入 | DEFAULT/CSR_ALG1/2/3 或已有 FP32 高精度兼容枚举 |
| `bufferSize` | Host | 输出 | GetBufferSize 写入最小 Device workspace 字节数 |
| `externalBuffer` | Device | 输入/工作区 | 至少为查询大小；active 期间内容和地址保持有效 |

#### 三段式调用

```cpp
size_t workspaceSize = 0;
aclsparseSpMMGetBufferSize(handle, opA, opB, alpha,
    matA, matB, beta, matC, computeType, alg, &workspaceSize);

// 分配至少 workspaceSize 字节的 Device workspace。
aclsparseSpMMPreprocess(handle, opA, opB, alpha,
    matA, matB, beta, matC, computeType, alg, workspace);

// 同一 sparsity pattern 下可复用 descriptors/workspace/preprocess。
aclsparseSpMM(handle, opA, opB, alpha,
    matA, matB, beta, matC, computeType, alg, workspace);
```

`GetBufferSize` 只读取 Host 描述符元数据，不读取设备数据或 alpha/beta 数值。`Preprocess` 可选：使用 active buffer 时复用加速元数据；直接以 inactive buffer 调用 SpMM 仍可执行，但允许缺少预处理加速。

零维和空稀疏矩阵仍先完成 handle、descriptor、dtype、op/alg、维度关系、必要指针和 CSR Host 元数据校验；只有合法问题才进入下列快速路径：

| 条件 | GetBufferSize | Preprocess | SpMM |
| --- | --- | --- | --- |
| `M==0` 或 `N==0` | 返回 0 | 接受 `externalBuffer==nullptr` 并返回 SUCCESS | 返回 SUCCESS，不下发 Kernel，不访问 A/B/C |
| `M>0 && N>0 && nnz==0`（包括合法的 `K==0`） | 返回 beta/fill 路径所需的最小对齐 tiling workspace | 可选；不构造 row reorder | 仅执行 `C=beta*C`；`beta==0` 时 fill 0，且不读取 A/B/C 旧值 |
| 其他非空问题 | 返回完整 workspace | 按正常预处理语义 | 按正常 SpMM 语义 |

现有描述符创建接口对 `ld>0` 和零维 Dense 的非空 `values` 指针要求保持不变。Python 路径在创建描述符前处理零元素 Tensor，避免把 PyTorch 空 storage 的空指针传给要求非空的 DnMat 描述符；C++ UT 使用合法占位 Device 指针验证零维 API 契约。

#### Pointer mode

| pointer mode | alpha/beta 位置 | 处理方式 |
| --- | --- | --- |
| `ACL_SPARSE_POINTER_MODE_HOST` | Host | Host 按 computeType 读取并写入 tiling scalar 区 |
| `ACL_SPARSE_POINTER_MODE_DEVICE` | Device | Kernel 从传入的 GM 标量地址读取，不执行 Device-to-Host copy |

alpha 和 beta 不能为空。FP16/BF16 的 computeType 为 FP32，因此标量指针指向 FP32；FP32 指向 FP32；Complex64 指向两个 FP32 组成的 Complex64。GetBufferSize/Preprocess 不因 scalar 数值变化而使 plan 失效，执行阶段读取当前值。

#### Active buffer 语义

每个 matA 描述符最多记录一个 active buffer。调用 Preprocess 后：

- 后续以相同 matA 和 active buffer 执行时，格式、索引、shape、nnz、op、order、computeType 和 alg 必须与 preprocess 兼容。
- alpha、beta、A values、B values 和 C values 可以变化。
- crow/col 结构、workspace 内容不得变化。
- 对同一 matA 再次以新 buffer preprocess 后，新 buffer 成为 active，旧 buffer 转为 inactive。
- Preprocess 会修改 matA 的 Host 内部状态，因此同一描述符的并发 preprocess 由调用者串行化；不同描述符可以并发。

workspace 头包含 magic、版本、required bytes、shape/alg/layout fingerprint 和 preprocess 状态，执行时先校验兼容性。公开 SpMM 函数签名不包含任意裸指针的实际分配字节数，因此采用兼容的两级校验：

1. `externalBuffer==nullptr` 且 required bytes 非零时，返回 `ACL_SPARSE_STATUS_INSUFFICIENT_RESOURCES`。
2. 若 `externalBuffer` 与 `aclsparseSetWorkspace(handle, ptr, size)` 注册的 workspace 相同，则使用 Handle 保存的 `size` 检查真实边界；`size < required` 时返回 `INSUFFICIENT_RESOURCES`，不读写 buffer。
3. 对历史调用者传入的其他非空裸指针，继续按公开契约信任其至少达到 GetBufferSize 返回值，避免新增不可实现的指针大小探测或破坏 ABI。

C++ 边界测试选择一个 required bytes 大于 1 的 case，先用 `aclsparseSetWorkspace(handle, nullptr, 0)` 清除 grow-only 的既有注册，再注册 `required-1` workspace，验证确定失败；随后重新清除并注册 exact-size workspace，在受控分配区前后设置 canary，验证成功且不越界。`nullptr` 作为独立空指针错误用例保留，但不再用它代替唯一的“小一字节”测试。

#### 错误码设计

| 错误 | 返回值 |
| --- | --- |
| handle/descriptor/必要指针为空 | `ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR` 或现有仓库对应空指针状态 |
| shape、ld、CSR 元数据、标量不合法 | `ACL_SPARSE_STATUS_INVALID_VALUE` |
| 格式不是 CSR | `ACL_SPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED` |
| dtype、index type、op、order、alg 组合不支持 | `ACL_SPARSE_STATUS_NOT_SUPPORTED` |
| workspace 为空或不可用 | `ACL_SPARSE_STATUS_INSUFFICIENT_RESOURCES` |
| ACL Runtime copy/launch/同步失败 | `ACL_SPARSE_STATUS_EXECUTION_FAILED` |
| 内部分配失败 | `ACL_SPARSE_STATUS_ALLOC_FAILED` |

三个阶段复用同一个校验入口和规格对象，避免 GetBufferSize 接受但 Preprocess/SpMM 拒绝同一合法组合。

### Host 侧设计

#### 公共规格对象

Host 首先把描述符解析为不依赖硬件的 `SpmmProblem`：

```text
stored A/B/C shape, logical M/K/N, nnz
index type/base, value/compute dtype
opA/opB, B/C order and ld
algorithm, pointer mode, stream
empty/transpose/conjugate flags
```

公共层负责参数校验、逻辑维度和能力矩阵；`arch35` 层只负责 950PR workspace、tiling 和 Kernel 选择。后续 arch22 可复用 `SpmmProblem`，在自己的目录生成硬件专用 tiling，避免在一份 Host 文件中堆叠 SOC 字符串判断。

#### Workspace 规划

所有 offset 使用 `uint64_t/size_t` 计算，并以仓库约定对齐。概念布局如下：

```text
[workspace header]
[SpmmTilingData]
[row reorder: logical M * int32]
[bin edges: (AIV core count + 1) * int32]
[CSR validation status]
[optional transpose row offsets: (logical M + 1) * int32]
[optional transpose source-position map: nnz * int32]
[algorithm-specific metadata / deterministic reduction scratch]
```

实际大小由 dtype、opA、alg、M/K/N/nnz 和平台 AIV 数决定。每次加法、乘法、对齐均检查溢出，最后再赋给 `size_t`。workspace 不保存 B/C 数据；Preprocess 一次性耗时和 workspace 峰值在自测报告单独记录。

#### CSR 预处理与校验

预处理按以下顺序在调用方 stream 上执行：

1. 校验 `crow[0]==base`、`crow[M]==base+nnz`、crow 单调非降、col 位于 `[base, base+K)`。
2. 计算每个逻辑输出行的 nnz，识别零行、短行、普通行和长尾行。
3. 根据估算工作量 `row_nnz * ceil(N/tileN)` 进行稳定分桶和行重排；空行仍保留输出写回任务。
4. `opA=T/H` 时生成转置 row offsets 和源位置映射；H 只在读取 values 时共轭，不复制 values。
5. ALG3 使用稳定顺序，禁止会改变 K 方向求和次序的无序原子归约。
6. 写入 workspace fingerprint 并将 buffer 标记为 active。

现有实现中整段 rowOffsets 的同步 D2H 以及 active tiling 的 D2H/H2D 刷新必须移除。行长度、合法性、直方图、prefix/reorder 和转置元数据由 Device preprocess Kernel 在 handle stream 上生成；Host 不读取完整 CSR。结构校验需向 Host 返回确定错误时，仅通过 `aclrtMemcpyAsync` 回传一个状态字，并只同步当前 handle stream 一次。该同步只发生在新 pattern 的 Preprocess/首次无 active plan 执行，合法 pattern 被缓存后不重复。

Host pointer mode 的小型 tiling 数据通过 `aclrtMemcpyAsync` 从 plan 持有的 pinned staging 区复制，staging 生命周期由 event/plan cache 保证；active plan 的 Host mirror 直接刷新动态字段，不先从 Device 读回旧 tiling。Device pointer mode 的 alpha/beta 地址直接写入 tiling 并由 Kernel 读取。GetBufferSize、active plan 的稳态 SpMM 和 Device pointer mode 不引入 Host 或全设备同步。

#### Tiling 和分核

Host 从 `PlatformAscendCManager` 查询实际 AIV core 数及可用片上资源，不硬编码某个 950PR SKU 的 core 数。tiling 至少包含：

```text
M/K/N/nnz, ldb/ldc, opA/opB, B/C order
index base, dtype, alg, pointer mode
tileN, row-bin offsets, transpose/conjugate flag
workspace offsets, scalar host value/device pointer
heavy-row strategy, deterministic flag
```

分核目标是按估算乘加量而不是仅按行数均衡。普通行由每个 AIV core 处理连续的重排行区间；长行按输出列 tile 分给多个 SIMT work item，避免单行独占；极端长行仅在 ALG1/2 可选择 K 分片，ALG3 保持固定的 K 遍历和固定归约树。

#### Tiling key

不为每个 shape 生成独立二进制。Tiling key 只编码会改变 Kernel 模板的离散特征：

```text
dtype x algorithm-family x order-pair x op-family x row-distribution
```

M/K/N/nnz、ld、tile 大小和 workspace offset 作为运行时 tiling 数据。这样可支持动态 shape 并控制 Kernel 实例数量。

### Ascend 950PR Kernel 侧设计

#### 950PR 专用实现

Ascend 950PR 使用 `arch35`，目标 NPU 架构为 `dav-3510`。延续现有实现的以下硬件路径：

- `__aicore__` 外层 Kernel + `asc_vf_call` + `__simt_vf__` 内层 SIMT。
- `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY)`，计算在 AIV 上执行。
- `asc_vf_call` 的架构上限为 2048 线程，但初始实现沿用现有 `__launch_bounds__(512)` 稳妥基线；在 950PR 上实测 256/512/1024 候选值，只有编译器寄存器报告、正确性和 P90 均通过时才提高，绝不直接把接口理论上限作为最优值。
- 所有 GM 地址和线性下标使用 64 位无符号计算，避免 P-03 大矩阵地址乘法溢出。
- dav-3510 的外层 core 必须一致调用 `asc_vf_call`；即使某个 bin 为空，也传递 `rowStart==rowEnd` 让内层立即返回，禁止外层提前 return 导致 AIC/AIV dispatch 握手不一致。
- 编译期检查 `__NPU_ARCH__==3510`，防止 arch35 Kernel 被误编入不兼容 SOC。

#### Kernel 数据流

```text
Init:
  读取 tiling，建立 CSR/B/C/workspace GM 指针，解析标量来源

Process:
  根据 core id 读取 bin 边界
  对 (logical row, output column tile) 分配 SIMT work
  遍历该行 CSR 非零项
  读取 A value 与 B tile
  使用 compute dtype 累加
  out = alpha * acc + (beta != 0 ? beta * C_old : 0)
  按 C order/ld 写回
```

index base 通过 `physical_offset = stored_offset - base` 和 `logical_col = stored_col - base` 归一化；不修改输入索引。

#### 四种 dtype

- FP16：A/B 读取 FP16，转换到 FP32 做乘加，写回前转换为 FP16；遵循 IEEE 转换，不额外做会改变 PyTorch 语义的人工饱和。
- BF16：A/B 读取 BF16，转换到 FP32 做乘加，写回 BF16。
- FP32：FP32 乘加；高精度兼容算法可使用 Kahan/固定补偿，默认性能路径使用任务精度标准验收。
- Complex64：以 `(real, imag)` 两个 FP32 表示，执行

```text
(ar + i*ai) * (br + i*bi)
= (ar*br - ai*bi) + i*(ar*bi + ai*br)
```

实部、虚部分别累加；`opA/opB=H` 时对对应输入虚部取反；alpha/beta 使用完整复数乘法，不能只处理实部。输出的实部和虚部均按 FP32 参数进行混合容差验收。

#### 布局地址计算

对于存储坐标 `(row,col)`：

```text
Row-major index    = row * ld + col
Column-major index = col * ld + row
```

先通过 opB 把逻辑 `(k,n)` 映射到 B 的存储坐标，再应用 B order；C 直接按 `(m,n)` 和 C order 寻址。四种 B/C order pair（RR、RC、CR、CC）均有最小 ld 和 padding ld 测试。所有路径共用经过 UT 验证的内联地址函数，避免在 dtype Kernel 中复制四份易错逻辑。

#### 算法路径

1. CSR_ALG2 / Row-major 快路径：输出列连续，SIMT work item 计算连续 `tileN`，合并 B/C 的 GM 访问；用于 Python 默认路径和 P-01/P-02/P-03。
2. CSR_ALG1 / Column-major 路径：按 Column-major 连续方向组织 work，复用预处理行分桶，避免对列主矩阵做整体转置。
3. CSR_ALG3 / 确定性路径：每个输出元素遵循唯一的 CSR 非零遍历顺序，不使用无序 atomic；极端长行若分阶段归约，使用固定分片边界和固定顺序的第二阶段归约。
4. FP32 高精度兼容路径：保留现有枚举值和行为，只对 FP32 启用补偿累加；其他 dtype 返回明确的不支持或按现有兼容策略处理，并在 README 固化，不静默产生不同语义。

#### 950PR 资源与性能预算

当前 `kChunk=8` 直接 GM 访问只能作为可运行基线，不能直接作为达到任务性能目标的依据。以每个非零项读取一个完整 B 行、C 按 beta 语义读写且暂不计算 cache 命中为估算口径，关键 case 的原始 GM 请求量和目标带宽约为：

| case | dtype/目标耗时 | 直接 GM 请求量估算 | 目标等效带宽 |
| --- | --- | --- | --- |
| P-01 | FP32 / 87.040 us | 约 76.1 MB | 约 0.87 TB/s |
| P-02 | FP32 / 204.576 us | 约 780.5 MB | 约 3.82 TB/s |
| P-03 | FP32 / 16,571.232 us | 约 66.4 GB | 约 4.00 TB/s |
| P-03 | Complex64 / 48,508.000 us | 约 132.5 GB | 约 2.73 TB/s |

该表是用于暴露压力的上界模型，不把片上 cache 命中虚构为既得收益。正式编码按以下顺序收敛：

1. 通过 `PlatformAscendCManager` 和编译报告记录实际 AIV 数、可用 UB/Reg 资源、寄存器占用及线程上限，不硬编码某个 SKU。
2. FP16/BF16/FP32 对 `tileN={8,16,32}`，Complex64 对 `{4,8,16}` 做编译期模板候选；Host 根据 N、dtype、行分布和寄存器预算选择，不在 Kernel 热循环中做动态分支。
3. Row-major ALG2 将连续 B/C 列映射到同一 warp/work item；SIMT 内层使用标量寄存器/编译器支持的短向量保存累加 tile，并对 B 发出连续合并访问。只有采用外层 SIMD+SIMT 混合搬运且存在跨工作项复用时才使用 RegTensor/UB 缓存 B tile，禁止在 SIMT VF 中误用仅适用于 SIMD 的 RegTensor，也禁止为无复用数据增加 GM->UB->Reg 往返。
4. 长尾行优先沿 N 维并行；只有 ALG1/2 且 profiler 证明单行瓶颈时才做固定 K 分片。ALG3 保持唯一归约树，不用性能路径破坏 bit-wise 确定性。
5. 候选组合用 P-01/P-02/P-03 及短行/长尾泛化集自动调参，比较 Kernel 数、GM 带宽、cache 命中、AIV 利用率、寄存器 spill、median 和 P90；最终规则固化为少量 tiling key，不在运行时做在线搜索。
6. 若直接 SIMT 路径仍无法达到目标，优先引入软件预取、双缓冲或 SIMD+SIMT 混合搬运；每项优化均需证明总 Kernel 范围变短，不能只缩短 SpMM 主 Kernel 却把全尺寸转置/格式转换移到其他 Kernel。

#### 稀疏泛化

| 行类型 | 判定依据 | 设计路径 |
| --- | --- | --- |
| 空行 | row nnz=0 | 跳过 A/B 读取，仍正确执行 beta 分支并写 C |
| 短行 | 少量 nnz | 一个 SIMT work item 处理多个输出列，降低调度开销 |
| 普通行 | 中等 nnz | 行重排 + 列 tile，均衡分配到全部 AIV core |
| 长尾行 | nnz 显著高于均值/P90 | 将不同 N tile 分配给更多 work item；必要时使用算法允许的固定 K 分片 |
| nnz=0 | 全矩阵空 | 专用 beta/fill 路径，不进入稀疏乘法循环 |

阈值由 M/N/nnz、行长度统计、dtype 字节数和平台资源共同计算，不使用只适配附件 degree=2..64 的常量，从而覆盖验收阶段的规格内泛化数据。

#### 异步与确定性

- SpMM 使用 handle 中的调用方 stream，下发后立即返回，不在正常执行路径调用全设备同步。
- Host pointer mode 的标量在 launch 前读取；Device pointer mode 由 Kernel 在同一 stream 上读取。
- 临时 Tensor/workspace 记录到当前 stream，异步执行结束前不会释放或复用。
- ALG3 重复执行必须 bit-wise 一致；ALG1/ALG2 按混合容差判定。
- 开启 PyTorch deterministic algorithms 时选择 ALG3；不受 ALG3 支持的组合明确报错，不能静默选择非确定算法。

## 支持硬件

| 支持的芯片版本 | 架构目录 | NPU 架构 | 涉及勾选 |
| --- | --- | --- | --- |
| Ascend 950PR | `arch35` | `dav-3510` | √ |

本任务只承诺 Ascend 950PR 实测结果。代码结构允许与后续 `arch22` 的 A2/A3 实现共存，但不得用 A2/A3 结果代替 950PR 验收。

950PR 软件组合必须整体配套：

| 组件 | 设计要求 |
| --- | --- |
| CANN Toolkit/ops | CANN 9.0.0 或目标 `ops-sparse` tag 指定的后续版本；安装同版本 `Ascend-cann-950-ops` |
| driver/firmware | 满足该 CANN 版本配套表，报告中记录完整版本 |
| PyTorch | 2.7 及以上，编译 bridge 时的 ABI 与运行时一致 |
| torch_npu | 26.0.0 及以后，且与 PyTorch/CANN 组合配套 |
| SOC 编译参数 | `SOC_VERSION=ascend950*`，由仓库映射到 `arch35` 和 `--npu-arch=dav-3510` |

配置阶段和自测启动阶段同时检查上述组合；任一组件不匹配时停止测试，不允许用“可以编译”替代运行时配套验证。

## 算子约束限制

1. 仅支持二维 CSR x 二维 Dense -> 二维 Dense；不新增 COO/CSC/BSR。
2. CSR crow/col 必须均为 int32 且同类型；支持 base 0/1；不支持 int64。
3. `0 <= M,K,N <= INT32_MAX`。base 0 时 `0 <= nnz <= INT32_MAX`；base 1 时因 `crow[M]=1+nnz` 必须可由 int32 表示，故 `0 <= nnz <= INT32_MAX-1`。CSR row offset 最终值、非空列空间的 `base+K-1`、索引归一化和 dense ld 均须可由 int32 表示；任意 storage 字节数、地址乘法和 workspace 还必须可由 `size_t/uint64_t` 表示并满足实际设备内存。
4. `mat1` 与 `mat2` 必须满足矩阵乘法维度；只有 input 支持广播。
5. 三个 Python Tensor dtype 必须相同，只支持 FP16/BF16/FP32/Complex64，不做 dtype promotion。
6. 三个 Python Tensor 必须位于同一 NPU device；跨设备报错。
7. B/C Row-major 要求 `ld>=cols`，Column-major 要求 `ld>=rows`；无法用 order+ld 表达的 Python 非连续输入先在 NPU contiguous。
8. ALG3 仅支持 CSR、`opA=N` 且 `opB!=H`；其他未声明组合报错。
9. Python 接口返回新 Dense Tensor，不提供 in-place/out 变体；C++ matC 按 API 作为 in/out。
10. 不要求图融合；支持当前 stream 异步执行。
11. `GetBufferSize` 返回值是所需最小 workspace；调用者负责分配并维持其生命周期。通过 `aclsparseSetWorkspace` 注册的 buffer 按 Handle 记录的真实 size 校验；其他历史裸指针调用仍遵循“调用者保证至少达到查询值”的契约。
12. 最大可运行规模还受 Ascend 950PR 实际 HBM、单次 ACL 分配和测试环境可用内存限制；测试脚本在执行前按 85% 可用内存阈值做保护，跳过必须记录而不能计为通过。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 正确性 | 四 dtype 的输出 shape/dtype/device/数值与 PyTorch 2.7+ 语义一致；合法边界成功，非法输入返回确定错误；无 CPU fallback | 任务书、PyTorch `torch.sparse.addmm` |
| 精度 | CPU 单 Golden + 混合容差；匹配率不低于 0.99，且所有元素满足绝对误差硬上限；Complex64 实虚部分开判定 | 任务书、《生态算子开源精度标准》 |
| 性能 | FP16/BF16/FP32 不低于 A100 的 1.0 倍，Complex64 不低于 A100 的 0.8 倍；主指标为相同调用范围的 Kernel 总耗时 | 任务书 P-01/P-02/P-03 及附件 A100 NCU baseline |
| 统计 | 每 case 预热不少于 10 次、正式样本不少于 30 个，报告 median 和 P90；每轮设备同步后计时 | 任务书 |
| 平台 | 所有最终功能、精度、性能和 Profiler 证据均来自 Ascend 950PR，并记录 CANN/PyTorch/torch_npu/固件/驱动版本 | 任务书 |

## 测试总体策略

测试分为四层，任何一层失败均不能宣称任务达标：

1. Host 纯校验 UT：规格推导、组合矩阵、workspace 溢出、错误码，不依赖 NPU 数据计算。
2. C++ API + Ascend C Kernel：在 950PR 上验证三段式接口、四 dtype、布局/算法/索引/边界、输入只读和资源生命周期。
3. Python/ATen E2E：直接调用 `torch.sparse.addmm`，验证公开语义、广播、非连续、dispatch、精度和异常。
4. 性能与 Profiler：附件 50 例、官方 P-01/P-02/P-03、Kernel 范围、端到端补充数据和无 fallback 证据。

### 任务附件脚本使用和补齐方案

| 文件 | 原始作用 | 本任务使用方式/必须补齐项 |
| --- | --- | --- |
| `function_sparse_ops.py` | ATK 构造 CSR/Dense 并调用 `torch.sparse.addmm` | 保留 200 例入口；加载 NPU bridge；扩展 FP16/BF16；新增零维、nnz=0/1、完整广播和错误专项执行器 |
| `accuracy_sparse_ops.py` | CSR 结构精确比较、值输出混合容差、NaN/Inf 位置检查 | 保留；确认 FP16/BF16 标准参数；Complex64 实部/虚部分别统计匹配率与硬上限 |
| `nodes_accuracy.yaml` | NPU 被测节点 + CPU Golden 节点 | 原样作为 ATK 拓扑，device id 可通过测试环境参数覆盖 |
| `sparse_addmm_accuracy.json` | 200 个 FP32/Complex64 case | 必须全量通过，不删例、不把 OOM/skip 计为 pass |
| `generate_sparse_ops_cases.py` | 固定 seed 生成 accuracy/performance case | 扩展 dtype 集；补充边界 case 单独文件，避免破坏任务附件已有 case id 和 A100 baseline 对应关系 |
| `sparse_ops_perf_common.py` | 固定输入、内存估算、构造、Event 采样、统计 | 增加 FP16/BF16 和官方 P case loader；index 按 int32；保持 CSR 列索引有序、无重复、values/alpha/beta 固定 |
| `benchmark_sparse_ops_npu.py` | NPU Event 端到端/算子调用耗时 | 默认 10 warmup + 30 samples，输出 median/P90；作为补充数据，不替代 Kernel 主指标 |
| `profile_sparse_ops_npu.py` | NPU Profiler Kernel 总耗时 | 从固定 5+5 改为可配置且正式使用 >=10+30；生成逐调用 Kernel 样本及 median/P90，`op_statistic` 聚合值只作交叉检查 |
| `benchmark_sparse_ops_gpu.py` | GPU Event 补充采集 | 用于复现/核对，不替代任务给定 A100 NCU 主基线 |
| `collect_sparse_ops_gpu_ncu.py` | NVTX 单调用范围内采集 A100 NCU Kernel 总耗时 | 保留相同调用范围定义和 kernel list；需要复采时使用同一 fixture |
| `sparse_addmm_performance.json` | 50 个 FP32/Complex64 case | 全量采集 NPU Kernel、Event、内存数据并与 `sparse_addmm_gpu_ncu_baseline.csv` 对齐 |
| `sparse_addmm_gpu_ncu_baseline.csv` | 50 例 A100 Kernel 总耗时 | 按 case id 一一 join，检查 shape/dtype/seed/degree/alpha/beta 完全一致后才计算倍率 |

任务附件生成器还包含 `sparse.mm/SpGEMM` 逻辑，但本设计只把 `sparse_addmm_*` 数据集作为 aclsparseSpMM 的验收范围，不将其他算子用例混入通过率。

### 测试命令

在实际 fork 的 `ops-sparse` 根目录，配置与源码匹配的 CANN 后执行：

```bash
source /usr/local/Ascend/cann/set_env.sh
# CANN >= 9.0.0，且源码、Toolkit、Ascend-cann-950-ops、driver/firmware版本配套
bash build.sh --ops=spmm --soc=ascend950 --run

# 必选 Python/ATen bridge；实际参数名以合入时仓库 CMake 规范为准
cmake -S . -B build -DSOC_VERSION=ascend950 \
  -DBUILD_PYTORCH_BRIDGE=ON \
  -DCMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')"
cmake --build build -j
python -m pip install ./python
```

ATK 正确性/精度基础集合：

```bash
cd test/python/aclsparseSpMM_testCase
atk task -c sparse_addmm_accuracy.json \
  -n nodes_accuracy.yaml --task accuracy -p .
```

补充 FP16/BF16 与边界集合：

```bash
atk task -c sparse_addmm_accuracy_fp16_bf16.json \
  -n nodes_accuracy.yaml --task accuracy -p .
pytest -q ../test_sparse_addmm.py
```

NPU Event 补充数据：

```bash
python benchmark_sparse_ops_npu.py \
  --case-file sparse_addmm_performance.json \
  --device 0 --warmup 10 --samples 30
```

NPU Profiler 正式 Kernel 数据（改造后的脚本）：

```bash
python profile_sparse_ops_npu.py \
  --case-file sparse_addmm_performance.json \
  --device 0 --warmup 10 --samples 30

python profile_sparse_ops_npu.py \
  --case-file sparse_addmm_performance_official.json \
  --device 0 --warmup 10 --samples 30
```

所有命令、环境变量、实际文件名及输出目录将在 `test/spmm/README.md` 和自测报告同步，保证验收人员可复现。

## 正确性测试设计

### Python/ATen 与功能矩阵

| 类别 | 用例 | 预期 |
| --- | --- | --- |
| 基础 shape | 方阵、M>K 长矩阵、M<K 宽矩阵、N=1、N 非 tile 整倍数 | shape `[M,N]`，结果满足公式 |
| dtype | FP16、BF16、FP32、Complex64 | 全链路命中 NPU，输出 dtype 不变 |
| 标量 | alpha/beta 分别覆盖 0、1、负数、普通小数 | 与 CPU Golden 一致 |
| 复数标量 | alpha/beta 具有非零实部和虚部 | 完整复数乘加正确 |
| 标量异常 | 实数 Tensor + 虚部非零复数 | 抛出与 PyTorch 语义一致的错误 |
| input 广播 | scalar、`[N]`、`[1,N]`、`[M,1]`、`[M,N]` | 广播结果正确 |
| 广播异常 | 无法广播到 `[M,N]` | 计算前报 shape 错误 |
| 非连续 | input/mat2 的转置视图、步长切片、padding storage | 直接描述或 NPU contiguous 后结果正确 |
| beta=0 | input 含 NaN/Inf | 不读取 input，NaN/Inf 不传播 |
| CSR 边界 | nnz=0、nnz=1、空行、首/尾空行、非均匀行、单个长尾行 | 结果正确，无越界或挂死 |
| 零维 | M=0、K=0、N=0 的合法组合 | shape、空输出或 beta 路径符合语义 |
| index base | base 0、base 1 | 数值相同，输入索引不被修改 |
| dtype 错误 | 三个 Tensor dtype 不同、CSR int64 index | 明确报错，不转换或 fallback |
| device 错误 | CPU/NPU 混用、不同 NPU device | 明确报错 |
| layout 错误 | 非法 ld、未知 order、重叠 stride | 返回确定错误 |
| alias | Python 输出与输入 storage 不同；C 与 A/B 非法重叠 | 不覆盖输入；非法 C++ alias 报错 |
| dynamic | 同进程连续运行多组 M/K/N/nnz | 不重新编译，plan 正确失效/复用 |

任务附件 200 例具体覆盖：每个 FP32/Complex64 各 100 例；shape 主要在 1000 到 20000；value mode 循环覆盖普通、小值、离群、正负抵消和 NaN/Inf；alpha/beta 覆盖三组组合；input mode 覆盖完整矩阵、`[1,N]`、标量及部分非连续视图；周期性空行覆盖 7/13 等间隔。补充集合不重复替代这些 case，而是覆盖附件没有的 FP16/BF16、零维、nnz=0/1、所有广播形态和接口错误。

### C++ 接口、布局和算法矩阵

对四 dtype 参数化执行下列合法组合；组合数量由测试参数生成，不手写重复代码：

| 项目 | 覆盖值 |
| --- | --- |
| B order | Row、Column |
| C order | Row、Column |
| leading dimension | 最小合法 ld、最小 ld + padding |
| opA | N；ALG1/2 补充 T/H；ALG3 只测 N |
| opB | N/T/H；ALG3 只测 N/T |
| algorithm | DEFAULT、CSR_ALG1、CSR_ALG2、CSR_ALG3；FP32 回归高精度枚举 |
| pointer mode | Host、Device |
| index base | 0、1 |
| preprocess | 不调用直接执行、调用一次后重复执行、切换 active buffer |

每个 C++ 用例均执行：

1. 创建 handle 并设置非默认 stream。
2. 创建 CSR/DnMat descriptors。
3. 调用 GetBufferSize 并检查返回码、size 非溢出。
4. exact-size workspace + canary；可选 Preprocess。
5. 调用 SpMM，stream synchronize 仅由测试等待结果。
6. 与高精度 CPU Golden 比较。
7. 比较 A crow/col/values 和 B 前后内容，确认输入只读。
8. 检查 C padding 和 workspace 前后 canary 未改变。
9. 以相反顺序或 RAII 销毁所有资源。

错误用例包括：空 handle/descriptor/size/scalar/workspace，维度不匹配，负 shape/nnz，rowOffsets 长度/首尾/单调性错误，col 越界，row/col index dtype 不同，int64 index，非法 base，非法 ld/order/op/alg，dtype/computeType 不支持，workspace 为空，规模和 offset 溢出。

### 确定性和资源测试

- ALG3 在相同输入、相同 stream 上重复 100 次，输出逐 bit 相同；Complex64 对完整 64-bit 复数元素逐 bit 比较。
- ALG1/2 重复结果按对应混合容差检查，不误用 bit-wise 标准。
- 连续 1000 次创建/设置 stream/创建描述符/查询/预处理/执行/销毁，比较执行前后 NPU 已分配内存和 Host 资源，无持续增长。
- 双 stream 交错执行不同 plan，验证 active buffer 不串用；同一描述符并发 preprocess 的未定义用法不纳入合法测试，但文档明确需调用者串行化。
- 空 bin、小 M 且 core 数较多时进行超时测试，验证所有外层 core 一致进入 `asc_vf_call`，不发生 dav-3510 dispatch 挂死。

### 无 CPU fallback 证据

必须同时提供三类证据：

1. dispatch table：记录 `aten::_sparse_addmm` 的 NPU CSR dispatch key 已注册到本实现。
2. Profiler：调用范围内出现 SpMM/广播 copy 的 NPU Kernel，记录 kernel 名称、调用次数和总耗时。
3. 负向保护：测试进程禁止/监控输入 `.cpu()` 和 CPU sparse addmm 路径；输出 device 始终为输入 NPU device。

只有 Python API 返回正确结果但无法证明 NPU dispatch 的情况不能计为通过。

## 精度测试设计

### CPU Golden

| NPU 输出 dtype | CPU Golden 计算 dtype |
| --- | --- |
| FP16 | FP32 |
| BF16 | FP32 |
| FP32 | FP64 |
| Complex64 | Complex128 |

CSR 结构和 dense 数据先按目标输入 dtype 量化，再提升到 Golden dtype 计算，防止 CPU 使用了 NPU 输入无法表示的额外精度。NPU 输出只与这一份 CPU Golden 比较，不使用多标杆择优。

### 混合容差

实数逐元素判定：

```text
abs(actual - golden) <= atol + rtol * abs(golden)
```

同时满足：

```text
匹配元素数 / 有效元素数 >= 0.99
每个元素 abs error <= max(A, 32 * ULP(golden))
```

| dtype | rtol | atol | A |
| --- | --- | --- | --- |
| FP16 | `2^-9` | `2^-9` | `1e-1` |
| BF16 | `2^-6` | `2^-6` | `1e0` |
| FP32 | `2^-10` | `2^-16` | `1e-2` |
| Complex64.real | `2^-10` | `2^-16` | `1e-2` |
| Complex64.imag | `2^-10` | `2^-16` | `1e-2` |

Complex64 实部、虚部分别计算匹配率和硬上限，两者都通过才算 case 通过。ULP 按精度标准在目标输出 dtype 的可表示间隔计算。NaN 要求位置一致；Inf 要求位置和符号一致；有限值才进入容差统计。空输出要求 shape/dtype/device 精确一致。

### 数据分布

四 dtype 均覆盖：

- 均匀/正态普通值。
- 接近零的小值。
- 正负混合及抵消序列。
- 显式零。
- 少量大离群值。
- 规格允许的 NaN、+Inf、-Inf。
- Complex64 的纯实、纯虚、实虚均非零、共轭转置和复数 alpha/beta。
- 短行、空行、长尾行及不同累加长度。

正式报告逐 case 保存 dtype、shape、nnz、seed、alpha/beta、广播模式、最大绝对/相对误差、匹配率、硬上限结果和日志/截图，不能只给总通过数。

## 性能测试设计

### 计时范围和统计方法

主指标是与 A100 NCU 相同一次 `torch.sparse.addmm` 调用范围内的所有 NPU Kernel 总耗时：

```text
performance_ratio = A100_kernel_total_us / NPU_kernel_total_us
```

- FP16/BF16/FP32 要求 ratio >= 1.0。
- Complex64 要求 ratio >= 0.8。
- Python Event 端到端/调用耗时、C++ `aclsparseSpMM` 执行阶段和 Preprocess 一次性耗时分别报告，不与 Kernel 主指标混算。

正式采集流程：

1. 固定 CSR 结构、values、dense input、mat2、alpha/beta、dtype/computeType、seed 和算法。
2. CSR 每行 column indices 排序且重复坐标已经合并，记录最终 nnz。
3. 数据生成、H2D、首次编译、描述符创建、workspace 分配和 preprocess 在计时区间外完成。
4. 先预热至少 10 次并同步，确认 plan cache 命中。
5. 正式采样至少 30 次，每次只包围一个 operator invocation；每轮结束执行设备同步。
6. 每个 invocation 按 profiler correlation/range 汇总其全部 NPU Kernel，得到 30 个 `kernel_total_us` 样本。
7. 对 30 个样本计算 median 和线性插值 P90，同时保存 min/max/mean 和 kernel list。
8. `op_statistic.csv` 总量除调用数只作为均值交叉检查，不能替代逐样本 median/P90。

`profile_sparse_ops_npu.py` 将增加 `--warmup/--samples` 参数，在 profiler 之外完成预热，用 `record_function`/step correlation 标识 30 个 active 调用，并从 Kernel 详情文件按调用聚合。若当前 CANN Profiler 版本不能可靠提供逐调用关联，则采用 30 个单调用 trace 采集后汇总；不能退回到固定 5 次均值。

### 附件 50 个性能 case

`sparse_addmm_performance.json` 的 25 个 FP32 和 25 个 Complex64 case 全量执行。结果按 `id` 与 `sparse_addmm_gpu_ncu_baseline.csv` 连接，连接前断言 operator、dtype、M/K/N、seed、degree、alpha、beta 全部相同。任何字段不一致时停止计算倍率，防止错配基线。

内存估算超过当时空闲显存 85% 的 case 可由保护逻辑停止，但必须记录为 `skipped/oom-risk`，不能计为 pass；正式验收前需通过调整空闲设备或分批运行补齐。

### 任务书 P-01/P-02/P-03

| 编号 | M x K x N | nnz | dtype | alpha/beta | A100 Kernel us | 最慢允许的 NPU median us | 目标 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| P-01 | 2708 x 2708 x 1433 | 10,556 | FP32 | 1/0 | 87.040 | 87.040 | >=1.0x |
| P-02 | 169343 x 169343 x 128 | 1,166,243 | FP16 | 1/1 | 321.536 | 321.536 | >=1.0x |
| P-02 | 同上 | 同上 | BF16 | 1/1 | 413.920 | 413.920 | >=1.0x |
| P-02 | 同上 | 同上 | FP32 | 1/1 | 204.576 | 204.576 | >=1.0x |
| P-03 | 2449029 x 2449029 x 256 | 61,859,140 | FP16 | 1/0 | 25,446.496 | 25,446.496 | >=1.0x |
| P-03 | 同上 | 同上 | BF16 | 1/0 | 33,704.096 | 33,704.096 | >=1.0x |
| P-03 | 同上 | 同上 | FP32 | 1/0 | 16,571.232 | 16,571.232 | >=1.0x |
| P-03 | 同上 | 同上 | Complex64 | 1/0 | 38,806.400 | 48,508.000 | >=0.8x |

上述“最慢允许”由 `A100_us / 目标倍率` 计算，是 median 的硬目标；P90 同时报告用于稳定性分析。P-01/P-02/P-03 使用与任务方 A100 基线配套的规范 CSR fixture，并在 README 记录来源、文件 hash、排序/去重状态、values 和 dense 数据 seed。若任务附件未提供与这三个基线配套的 CSR 文件或生成规则，验收前必须向任务方确认；不能仅生成一个 shape/nnz 相同但行分布不同的随机矩阵并声称与给定 A100 时间同口径。

### 950PR 性能优化验证

Profiler 至少检查：

- DEFAULT 在 Python Row-major 路径实际选择 CSR_ALG2。
- 正式采样区间没有 preprocess、H2D、首次编译和 CPU compute。
- `beta=0` 不出现 input broadcast/copy；`beta=1` 的 copy 和 SpMM 均计入与 A100 相同调用范围。
- 短行/长尾行的 AIV core 利用率、Kernel 数、GM 读写和负载分布。
- FP16/BF16 使用 FP32 累加但没有多余 dtype 往返 Kernel。
- Complex64 没有拆成多次全尺寸临时 Tensor 运算，复数乘加在 SpMM Kernel 内完成。
- P-03 的 64 位地址计算无溢出，workspace 和峰值 HBM 在设备容量内。

自测报告同时记录 NPU 型号明确为 Ascend 950PR、CANN 精确版本、driver/firmware、PyTorch、torch_npu、ops-sparse commit、编译参数、AIV core 查询结果、设备空闲状态和每个 case 的 profiler 文件路径。

## 测试覆盖追踪矩阵

| 任务书要求 | 附件基础覆盖 | 补充测试 | 通过条件 |
| --- | --- | --- | --- |
| Python/ATen E2E | 200 accuracy + 50 perf 调用公开 API | dispatch、异常、广播、零维、非连续专项 | 全部通过且无 fallback |
| 四 dtype | FP32/Complex64 | FP16/BF16 accuracy/perf/C++ | 四 dtype 全链路 |
| Complex64 | 100 accuracy + 25 perf | 共轭、复数标量、C++ layout/alg | 实虚精度和 >=0.8x |
| shape/nnz 泛化 | 1000..20000、degree 2..64 | 小 shape、零维、nnz 0/1、长尾、P-01..03 | 无越界/挂死，精度通过 |
| Row/Column、ld | 无完整 C++ 矩阵 | B/C 四 order pair，min/padded ld | 合法组合正确，非法报错 |
| ALG1/2/3 | 无 | C++ 参数化 + profiler 路径证明 | ALG3 bit-wise，其余精度通过 |
| 三段式 API | Python 间接 | GetBuffer/Preprocess/SpMM、active/inactive buffer | 返回码和生命周期正确 |
| workspace 边界 | 内存估算保护 | registered `required-1`、null、exact-size + canary、溢出 | 不足时确定报错，exact-size 不越界 |
| 输入只读/alias | 无 | C++ 前后 hash、Python storage 检查 | A/B/input 不被改写 |
| 资源泄漏 | 无 | 1000 次生命周期、双 stream | 无持续内存/资源增长 |
| 正式性能统计 | Event 10/30；Profiler 5/5 均值 | Profiler >=10/30 逐调用 median/P90 | 倍率达到任务目标 |
| 950PR 特性 | 脚本指定 NPU | arch35/dav-3510、AIV/SIMT、统一 dispatch | 950PR 实机证据齐全 |

## 兼容性分析

1. API/ABI：不修改三个公开函数签名；ALG2/ALG3 追加在现有 enum 末尾，不改变已有枚举数值。
2. 已有能力：本任务不删除 FP32、FP16、INT8 现有路径。虽然 INT8 不属于本任务验收 dtype，仍执行现有 INT8 回归，防止扩展四 dtype 时产生退化。
3. 多架构：硬件专用代码保持在 `arch35`；公共规格/校验可共享，后续 A2/A3 放入 `arch22`，由 `SOC_ARCH_DIRS` 编译选择，避免同名符号冲突。
4. PyTorch：bridge 是任务必选交付件，独立构建并随 `ops_sparse_torch` 包安装，适配 PyTorch >=2.7、torch_npu >=26.0.0；基础 `libops_sparse` 不反向依赖 Python。验收构建必须开启 bridge，并执行安装后 import、ABI 和 dispatch smoke test。
5. 描述符：沿用现有 Handle、SpMat、DnMat 的创建/销毁接口和 Host/Device pointer mode；旧调用者不使用新 dtype/alg 时行为不变。
6. 文档：同步更新 `sparse/spmm/README.md`、`test/spmm/README.md` 和公开 API 列表，避免源码已支持但文档仍写“base 0/FP32/FP16/INT8 only”。
7. 上游并发修改：开始编码和提交 PR 前基于最新 master rebase；若 A2/A3 已修改 Host 公共代码，合并公共校验并保留 `arch22/arch35` 各自 tiling/Kernel，完成两类已有测试回归后再提交。

## 风险与应对

| 风险 | 影响 | 应对 |
| --- | --- | --- |
| PyTorch bridge 在现有仓无既定目录 | 构建/安装方式可能与上游预期不一致 | bridge 保持必选交付；编码前以最新 master 和 SIG 意见确认目录及插件加载规范，CI 验证安装后的标准 import 和 dispatch，C API 与 bridge 保持链接边界 |
| CANN/ops-sparse/driver 版本不配套 | 950PR 编译接口或运行 ABI 不一致 | 使用 CANN 9.0.0+ 配套组合，记录精确版本并在配置阶段检查，禁止混用 8.5 头文件/库 |
| P-01..03 canonical CSR 未包含在附件 | 同 shape/nnz 不代表同工作量，倍率不可比 | 获取任务方 fixture/生成规则并记录 hash；未确认前不宣称达到给定基线 |
| Profiler 原脚本只有 5 个 active step | 不满足 10/30 和 median/P90 | 改造为逐调用 30 样本；原聚合均值仅作交叉检查 |
| P-03 数据和输出占用大 | OOM 或 allocator 抖动 | 提前估算、独占设备、分 dtype 执行、复用 workspace，禁止把 skip 计通过 |
| Complex64 吞吐不足 | 无法达到 0.8x A100 | 复数 fused Kernel、连续列 tile、避免全尺寸临时 Tensor，按 profiler 定位访存/调度瓶颈 |
| 长尾 CSR 负载不均 | 核利用率低、P90 抖动 | 基于 row work 的稳定分桶、N tile 并行、算法允许时固定 K 分片 |
| Device pointer mode 引入 Host 同步 | 破坏异步语义 | 标量地址传入 Kernel 直接 GM load，不做 D2H |
| ALG3 确定性与并行度冲突 | bit-wise 不一致 | 固定遍历/分片/归约树，独立于 ALG1/2 性能路径 |

## 参考资料

1. [PyTorch torch.sparse.addmm](https://docs.pytorch.org/docs/stable/generated/torch.sparse.addmm.html)
2. [PyTorch 2.7 native_functions.yaml](https://github.com/pytorch/pytorch/blob/v2.7.1/aten/src/ATen/native/native_functions.yaml)
3. [PyTorch 2.7 Sparse CUDA 参考实现](https://github.com/pytorch/pytorch/blob/v2.7.1/aten/src/ATen/native/sparse/cuda/SparseCUDATensorMath.cu)
4. [cuSPARSE 13.3 cusparseSpMM](https://docs.nvidia.com/cuda/archive/13.3.0/cusparse/index.html#cusparsespmm)
5. [CUDA Toolkit 13.3 Update 1 Release Notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html)
6. [ops-sparse](https://gitcode.com/cann/ops-sparse)
7. [生态算子开源精度标准](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md)
8. [CANN 9.0.0 版本说明与 Ascend 950PR 支持范围](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/releasenote/release-notes.md)
9. [Ascend C 9.0 SIMD/SIMT 编程指南](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/programug/Ascendcopdevg/atlas_ascendc_10_10053.html)
10. [Ascend C 9.0 `asc_vf_call` API](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_10303.html)
11. [Ascend C 9.0 内置数据类型](https://www.hiascend.com/document/detail/zh/canncommercial/900/API/ascendcopapi/atlas_ascendc_10_0019.html)
