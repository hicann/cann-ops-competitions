# aclsparseSpSV 算子设计文档（Ascend 950 / A5，审核优化版）

***

# 需求背景（required）

## 需求来源

1. **任务书**：`9月社区任务-aclsparseSpSV算子开发(950)/aclsparseSpSV_A5_task_doc.md`——验收合同，功能/精度/性能/交付以此为主。
2. **任务包专项测试**：`9月社区任务-aclsparseSpSV算子开发(950)/test_cases/aclsparseSpSV_testCase/`——核验 Python 适配层实际调用合同、200 个精度 case、206 个性能 case 和 GPU/NPU 性能脚本。
3. **ops-sparse 当前 master 的 SpSV arch35 源码/README**——确定现有 FP32 实现事实和本任务真实改造面。
4. **NVIDIA cuSPARSE SpSV 官方文档**——参考 SpSV 生命周期及参考语义。
5. **华为 CANN 9.1 Ascend C / SIMT 官方文档**——确定 950 SIMT 编程和内存模型。
6. **PyTorch 官方 Library/Dispatcher 文档**——确定专项测试 Hook 的 C++ 自定义算子接入方式。

## 背景介绍

### 1. 算子功能

SpSV（Sparse Triangular Solve with Vector）求解：

\[
\operatorname{op}(A)Y=\alpha X
]

其中 `A` 为稀疏三角方阵，`X/Y` 为稠密向量，`op(A)` 为 `N/T/H`。

以 `LOWER + NON_TRANSPOSE + NON_UNIT` 为例：

\[
y\_i=\frac{\alpha x\_i-\sum\_{j\<i}a\_{ij}y\_j}{a\_{ii}}
]

核心困难不是普通逐元素计算，而是：

- 行间存在真实依赖；
- 稀疏索引导致离散访存；
- 同一 level 中行长度差异可能很大；
- 四种稀疏格式、base 0/1 和 N/T/H 需要映射到统一求解视图；
- complex64 的 H 必须做共轭转置；
- Analysis / UpdateMatrix / Solve 状态跨 API 调用持续存在。

### 2. cuSPARSE 官方参考语义

NVIDIA 当前 cuSPARSE SpSV 官方文档确认：

- 生命周期包含 `createDescr → bufferSize → analysis → solve → updateMatrix → destroyDescr`；
- `bufferSize` 返回 Analysis/Solve 所需 external workspace；
- 支持 CSR / CSC / COO / SLICED\_ELL；
- 支持 LOWER/UPPER、UNIT/NON\_UNIT、N/T/H；
- `bufferSize` 与 `analysis` 接受 `NULL` vecX/vecY；
- Solve 支持 in-place、异步执行，并声明 solve 阶段 deterministic；
- 允许稀疏索引未排序；
- Analysis 到 Solve 之间参数、矩阵描述和 externalBuffer 需要保持一致；
- `updateMatrix` 用于 Analysis 后更新 values。

同时，cuSPARSE 通用存储格式文档说明：重复索引时**数值正确性不总能保证**。因此本任务若要把 duplicate 作为有效输入，必须由任务 Golden 明确语义，不能自行假定“重复项求和”。

### 3. 当前 ops-sparse master 的真实基线

审核基线为 `ops-sparse` commit `e0015bb76090487165f43dbaf2aae37648887a4b`。当前 `sparse/spsv/arch35/` **静态源码中**已存在 FP32 SpSV 路径；本机没有 PyTorch/torch\_npu/CANN Toolkit/NPU 运行环境，因此“可运行”和性能数据仍须在任务环境实测，不由本次静态审核代替。源码核验结果：

| 项目                  | 当前 master 事实                                                 | 本任务增量                                                   |
| ------------------- | ------------------------------------------------------------ | ------------------------------------------------------- |
| 平台                  | Ascend 950                                                   | 保持                                                      |
| A2/A3               | SpSV 不支持                                                     | 不新增；只做共享代码回归                                            |
| compute/value dtype | 仅 `ACL_FLOAT`                                                | 增加 `ACL_COMPLEX64`                                      |
| 算法枚举                | `ACL_SPARSE_SPSV_ALG_DEFAULT`                                | 保持                                                      |
| index               | `(I32,I32)`、`(I64,I32)`、`(I64,I64)`                          | 任务主测 I32，但不得回归 I64                                      |
| format              | CSR / CSC / COO / SLICED\_ELL 静态路径均存在                        | 先做 FP32 组合回归，再泛化 complex64                              |
| Analysis            | 源码已有格式转换/转置 + Level Scheduling                               | 修 stable scatter，再泛化 values dtype / H 共轭                |
| Solve               | 源码已有 single/multi-block Level Scheduling；当前一线程一行             | complex64 数值路径；性能不够再增加长行协作                              |
| workspace           | `diagPtr/levelPtr/levelRow/validCount` + 按需 CSR/transpose 副本 | values 区 4B→valueSize                                   |
| Host alpha          | 当前 `float`                                                   | FP32/complex64                                          |
| Device alpha        | 当前 Kernel 按 float 读                                          | FP32/complex64                                          |
| GENERAL update      | 已有实现，但 COO/SELL + transpose 的 mapping composition 有缺口        | 先修 FP32 映射，再增加 complex64 + H 共轭                         |
| DIAGONAL update     | 源码路径接收 m 个对角值                                                | FP32 回归 + complex64 + adapter 的 full-values→diag gather |
| zero nnz            | 源码为 UNIT scale-copy、NON\_UNIT scale-inf                      | FP32 实测 + complex64 等价实现                                |
| deterministic       | Solve 每行按存储顺序串行累加；但并行 COO/transpose scatter 的稳定性未证明          | 改为稳定格式转换；不得引入浮点 atomic reduction                        |

**源码已定位的 FP32 强绑定点**：

1. `ValidateSpSVCommonParams()` 只接受 `ACL_FLOAT`；
2. `ComputeWorkspaceOffsets()` 的 `csrValues/transValues` 固定 `sizeof(float)`；
3. `ReadHostScalar()` 返回 `float`；
4. `SpsvTilingData::alpha` 是 `float`；
5. Analysis/Solve/Update Kernel 的 values/X/Y/alpha 均为 `float* / float`；
6. 转置复制 `transValues[pos] = values[p]`，没有 complex H 共轭；
7. 当前 FP32 CSC 归一化会把 op 简化成“是否转置”的二值状态，对 complex H 不足。

因此本任务正确路线是**增量类型泛化和状态补强**，而不是重写现有 Level Scheduling。

***

# 需求分析（required）

## 需求描述

在 Ascend 950 平台上完善 `aclsparseSpSV`：

- 新增 `ACL_COMPLEX64`，保留 `ACL_FLOAT`；
- 保持公开 `aclsparseSpSV_*` C API 签名不变；
- 支持 CSR / CSC / COO / SLICED\_ELL；
- 支持 LOWER/UPPER、UNIT/NON\_UNIT、N/T/H；
- complex64 的 H 为真正共轭转置；
- 支持 Host/Device pointer mode alpha；
- 支持 X/Y Device values，Y 可与 X 同指针原地求解；
- 完整 BufferSize / Analysis / Solve / UpdateMatrix 生命周期；
- GENERAL / DIAGONAL update；
- NPU stream 异步提交，无 CPU fallback；
- 满足任务精度、性能、内存、确定性和异常测试。

### 证据边界

| 结论类型           | 本次已确认                                                                                    | 尚需目标环境确认                                                          |
| -------------- | ---------------------------------------------------------------------------------------- | ----------------------------------------------------------------- |
| 接口/基线          | 任务书、评测脚本、commit `e0015bb...` 的 Host/Kernel/README                                        | 待开发分支最终 commit 与公共 ABI diff                                       |
| 参考语义           | cuSPARSE 官方 SpSV 页面列出的四格式、dtype、NULL vec、unsorted、in-place、async、deterministic、update 语义 | 任务方对 NULL vec 一致性、非法 Device 内容错误码的解释                              |
| 950 路线         | 当前源码使用 `__simt_vf__` / `asc_vf_call`；华为 9.1 文档说明 SIMT 仅支持 950PR/950DT                    | CANN 9.1.0 正式配套编译器的 complex helper 可编译性与性能                        |
| PyTorch bridge | 官方 Library API 支持 `TORCH_LIBRARY`、backend-specific `TORCH_LIBRARY_IMPL` 和 custom class   | 任务镜像中的 PyTorch/torch\_npu 版本、实际 dispatch key、stream/allocator API |
| 运行结果           | 无                                                                                        | NPU 编译、UT/ST、ATK、Profiler、内存报告全部待实机完成                             |

当前实现已经是 950 SIMT 路线。RegBase 参考中的 `RegTensor/MaskReg` 数据流与本算子现有 `__simt_vf__` 稀疏离散访存主干不一致，本任务不切换 RegBase；除非另有原型和 Profiler 证明，否则切换会成为无依据重写。

## 需求拆解

### P0：必须正确

1. FP32 原能力无回归；complex64 全生命周期可用。
2. N/T/H、fill/diag 组合数值正确。
3. CSC/H 不能把 H 错当普通 T。
4. Host/Device alpha 在同一 handle pointer mode 合同下正确读取。
5. Analysis 绑定的 format/fill/diag/op/dtype/pointer mode/base/index/workspace 状态在 Solve 校验。
6. UpdateMatrix 后不重做结构 Analysis；GENERAL/DIAGONAL 只更新 values 状态。
7. X/Y alias 正确；异步资源生命周期正确。
8. C++ UT/ST 覆盖专项 Python 包没有覆盖的四格式/边界/原地/异常。

### P1：必须达到验收性能

1. 复用现有 Level Scheduling 和 single/multi-block heuristic；
2. complex64 避免无谓 GM round trip；
3. `one_long_row/highly_imbalanced/power_law` 若成为瓶颈，再引入固定顺序的 warp/线程组协作；
4. 使用 Profiler 证明瓶颈和收益，禁止凭经验增加 workspace/UB 占用。

### P2：兼容性

1. 任务主测 Device I32，但保留 master 现有 I64 index 组合；
2. 不把 arch35 实现扩散到 A2/A3；
3. 公共头文件和 descriptor 修改不破坏仓库其他算子/A2/A3 构建。

## 任务专项测试包实际覆盖

### 1. 200 个 accuracy case

实际解析 `accuracy_cases.json`：

| 维度           | 分布                                                                                         |
| ------------ | ------------------------------------------------------------------------------------------ |
| dtype        | float32 102；complex64 98                                                                   |
| base         | base0 101；base1 99                                                                         |
| op           | N 67；T 67；H 66                                                                             |
| fill         | lower 100；upper 100                                                                        |
| diag         | nonunit 100；unit 100                                                                       |
| pointer mode | host 100；device 100                                                                        |
| update       | general 101；diagonal 99                                                                    |
| row pattern  | diagonal/banded/uniform/highly\_imbalanced 各26；one\_long\_row/skewed/random/power\_law 各24 |
| m            | 2～63                                                                                       |
| nnz          | 2～2016                                                                                     |

### 2. 206 个 performance case

实际解析 `performance_cases.json`：

| 维度           | 分布                            |
| ------------ | ----------------------------- |
| dtype        | float32 102；complex64 104     |
| base         | base0/base1 各103              |
| op           | N69 / T69 / H68               |
| fill         | lower105 / upper101           |
| diag         | nonunit106 / unit100          |
| pointer mode | **全部 device**                 |
| update       | **全部 general**                |
| row pattern  | 8 类各25 + 6 个 P-01～P-03 anchor |
| m            | 1,024～262,144                 |
| nnz          | 1,024～3,932,160               |

### 3. Python 包的覆盖缺口

`operator_adapter.py::_csr()` 明确要求：

```text
m > 0
nnz >= m
format = CSR only
```

并且 Python 生命周期 hook 为：

```text
spsv_analysis_npu(row_offsets, col_indices, values,
                  op, fill, diag, base, pointer_mode) -> plan
spsv_update_npu(plan, values, update_kind) -> plan | None
spsv_npu(plan, x, alpha) -> y
```

因此以下项目不能依赖这 200/206 个 Python case，必须 C++ UT/ST 单独验证：

- CSC / COO / SLICED\_ELL；
- `m=0/1`、`nnz=0/1`；
- X/Y 同 values 指针原地求解；
- 非法 index / workspace / lifecycle；
- duplicate 明确定义后的行为；
- externalBuffer 过早释放；
- descriptor identity/属性变化；
- I64 现有能力回归。

此外，`_csr()` 为每行构造连续单调列号并强制写入一个对角项，`_rand()` 只生成有限均匀随机值。因此专项 Python case 名称中的 `random/power_law/...` 仅描述**各行 nnz 分布**，不代表列索引乱序，也不覆盖任务书要求的 70% 均匀 + 20% 正态 + 10% 特殊构造。以下因子必须由 C++ UT/ST 或补充的独立 NPU 测试实现：

- 行内未排序、duplicate；
- 正态/抵消数据；
- 缺失对角、零对角、INF/NAN；
- 同一输入重复 Solve 的 bitwise 比较。

任务包的 `generate_cases.py` 当前导入不存在的 `extra_performance_cases` 和 `extra_accuracy_cases`，静态复现会报 `ModuleNotFoundError`。开发期间把已提交的 `accuracy_cases.json`（200 条）和 `performance_cases.json`（206 条）作为不可擅改的输入；若需重新生成，先由任务方补齐这两个模块并校验生成 JSON 的哈希/条数/分布与验收版本一致。

## 任务包中的两个接口语义澄清项

### 1. DIAGONAL adapter 输入与 C API 输入不同

Python adapter 在 `update=diagonal` 时，仍把**完整 nnz values Tensor**传给 `spsv_update_npu`；而当前公开 C API/README 的 `ACL_SPARSE_SPSV_UPDATE_DIAGONAL` 语义是 `newValues` 指向**长度 m 的对角新值数组**。

因此适配层必须做：

```text
full updated values (nnz)
       │
       ├─ plan 中已有/可复用 diag position metadata
       ▼
NPU gather diagonal values -> diagValues[m]
       │
       ▼
aclsparseSpSV_updateMatrix(..., diagValues.data_ptr(), DIAGONAL)
```

`diagValues` 必须由 plan 保持生命周期，至少覆盖异步 update 到后续同 stream Solve 对该数据的消费完成。

### 2. NULL vec descriptor 一致性存在文字冲突

任务书同时要求：

- BufferSize/Analysis 允许 vecX/vecY descriptor 为 NULL；
- Solve 要求非 NULL；
- Analysis 到 Solve 相关 vecX/vecY descriptor/参数保持一致。

这三条若按“指针身份逐字相同”无法同时满足。本文采用以下**待任务方确认的可执行解释**：

- Analysis 传非 NULL：缓存 descriptor identity/size/dtype/device，Solve 必须一致；
- Analysis 传 NULL：标记 `deferred vector binding`；Solve 可提供合法非 NULL descriptor，但必须满足 m/dtype/device；
- 该规则需用 C++ UT 固化；任务方确认前不能把它写成已确定的公开接口承诺。

***

# 详细设计（required）

## 算子分析

### 数学公式

\[
B Y=\alpha X,\qquad B=\operatorname{op}(A)
]

第 i 行：

\[
s\_i=\alpha x\_i-\sum\_{j\in D(i)}b\_{ij}y\_j
]

NON\_UNIT：

\[
y\_i=s\_i/b\_{ii}
]

UNIT：

\[
y\_i=s\_i
]

### N/T/H 与求解方向

| 原始 fill | N        | T                          | H                                      |
| ------- | -------- | -------------------------- | -------------------------------------- |
| LOWER   | forward  | effective UPPER / backward | effective UPPER / backward + conjugate |
| UPPER   | backward | effective LOWER / forward  | effective LOWER / forward + conjugate  |

complex64：若 `z=a+bi`，`conj(z)=a-bi`。

### 支持数据类型

| 对象           | 本任务声明                         | 兼容要求                 |
| ------------ | ----------------------------- | -------------------- |
| A values     | `ACL_FLOAT` / `ACL_COMPLEX64` | complex64 = 8B       |
| X/Y          | 与 A/computeType 一致            | Device values        |
| alpha        | 与 computeType 一致              | Host 或 Device        |
| computeType  | `ACL_FLOAT` / `ACL_COMPLEX64` | 其他返回 NOT\_SUPPORTED  |
| Device index | 任务主测 I32                      | 不回归 master 已有 I64 组合 |

华为 ACL 官方 `aclDataType` 文档包含 `ACL_COMPLEX64`。但本文**不假设**目标 Ascend C SIMT 编译器存在某个未核验的 `Complex<T>` 运算类。

#### complex64 Host/Kernel 表示

仓库公开头 `include/cann_ops_sparse.h` 已定义：

```cpp
typedef struct aclsparseComplex {
    float x;  // real
    float y;  // imag
} aclsparseComplex;
```

Host 入口以 `aclsparseComplex` 作为 `ACL_COMPLEX64` scalar ABI。Kernel 如因编译边界需要独立类型，只允许定义布局完全相同的 trivially-copyable 类型，例如：

```cpp
struct alignas(8) SpsvComplex64 {
    float real;
    float imag;
};
static_assert(sizeof(aclsparseComplex) == 8);
static_assert(offsetof(aclsparseComplex, y) == sizeof(float));
static_assert(sizeof(SpsvComplex64) == sizeof(aclsparseComplex));
static_assert(std::is_trivially_copyable_v<SpsvComplex64>);
```

不能仅以 `aclDataType` 中存在 `ACL_COMPLEX64` 推导 Kernel 已支持复数算术；该枚举只证明类型标识存在。GM values 与 torch.complex64 的二进制互操作必须由 adapter ABI UT（实部、虚部哨兵值）验证。

提供 `ValueOps<T>`：

```text
add / sub / mul / div / conj / zero / load / store
```

complex multiply：

\[
(a+bi)(c+di)=(ac-bd)+(ad+bc)i
]

complex divide 不能直接计算 `c²+d²`。以下 Smith 分支是最低基线，对 `(a+bi)/(c+di)`：

```text
if abs(c) >= abs(d):
    r = d / c
    den = c + d * r
    real = (a + b * r) / den
    imag = (b - a * r) / den
else:
    r = c / d
    den = d + c * r
    real = (a * r + b) / den
    imag = (b * r - a) / den
```

Baudin/Smith 的公开论文指出原始 Smith 方法仍有可构造的 overflow/underflow 失败，因此任务参数表若按“全部有限值”验收，最终实现必须采用论文中的额外 scaling 改进或用等价算法证明覆盖范围，不能只抄上述基线分支。`c==0 && d==0` 单独走确定的特殊值分支，避免 `0/0` 中间值在不同编译优化下改变 NaN payload；具体 real/imag 的 INF/NAN 分类以任务 Golden 确认并写 UT。目标编译参数是否启用 fast-math/flush-to-zero 也必须记录。

#### Host alpha ABI

公开参数只有 `const void *alpha`，因此：

- FP32：读 4B float；
- complex64：调用者传 `const aclsparseComplex*`；连续 8B 布局为 `{x=real, y=imag}`；
- 使用 `memcpy` 到 scalar pack，不使用 `reinterpret_cast<std::complex<float>*>`；
- 增加 Host alpha ABI UT，验证 Python/测试扩展生成的 complex scalar 与 C API 完全一致。

Device pointer mode 下，`alpha` 必须指向一个 Device `ACL_COMPLEX64` 标量（8B）；Kernel 从该地址读取一个 `SpsvComplex64`。公开 API 缺少指针内存类型参数，本实现只能在目标 ACL Runtime 提供可靠 pointer-attribute 查询时做同步 Host/Device 类型校验；否则这是调用者合同，错误指针由 launch/runtime 错误暴露，不能承诺 Host 入口必然返回 `INVALID_VALUE`。

### 支持形状

- A：`[m,m]` 方阵；
- X/Y：长度 m；
- 动态 m/nnz；
- 任务要求边界 `m/nnz=0/1`、空行、长尾、未排序；
- 当前 master 内部 levelRow 为 int32 且 Host 限制 `m <= INT32_MAX`，本任务保持该约束；
- 任务主测 I32 时 `nnz` 需落在 I32 可表示范围；已有 I64 路径按 master 现有约束保留。

## 算子实现

### 总体原则

**复用现有 arch35 算法主干，只做必要增量：**

```text
Public C API
   │
   ├─ Host validation / descriptor state
   │
   ├─ BufferSize: existing workspace layout + valueSize
   │
   ├─ Analysis: existing format conversion / transpose / Level Scheduling
   │                                  + complex H conjugation
   │
   ├─ UpdateMatrix: corrected/composed mapping + ValueT + H conjugation
   │
   └─ Solve: existing Level Scheduling + ValueT
                         │
                         └─ optional profiled long-row cooperative path
```

不引入 CPU 求解或 CPU 稀疏格式转换 fallback。

### 3.2.1 Host 侧设计

#### 1. 公开接口

公开签名严格沿用任务书，不改变 ABI：

```c
aclsparseStatus_t aclsparseSpSV_createDescr(aclsparseSpSVDescr_t *spsvDescr);
aclsparseStatus_t aclsparseSpSV_destroyDescr(aclsparseSpSVDescr_t spsvDescr);

aclsparseStatus_t aclsparseSpSV_bufferSize(
    aclsparseHandle_t handle,
    aclsparseOperation_t opA,
    const void *alpha,
    aclsparseConstSpMatDescr_t matA,
    aclsparseConstDnVecDescr_t vecX,
    aclsparseDnVecDescr_t vecY,
    aclDataType computeType,
    aclsparseSpSVAlg_t alg,
    aclsparseSpSVDescr_t spsvDescr,
    size_t *bufferSize);

aclsparseStatus_t aclsparseSpSV_analysis(
    aclsparseHandle_t handle,
    aclsparseOperation_t opA,
    const void *alpha,
    aclsparseConstSpMatDescr_t matA,
    aclsparseConstDnVecDescr_t vecX,
    aclsparseDnVecDescr_t vecY,
    aclDataType computeType,
    aclsparseSpSVAlg_t alg,
    aclsparseSpSVDescr_t spsvDescr,
    void *externalBuffer);

aclsparseStatus_t aclsparseSpSV_solve(
    aclsparseHandle_t handle,
    aclsparseOperation_t opA,
    const void *alpha,
    aclsparseConstSpMatDescr_t matA,
    aclsparseConstDnVecDescr_t vecX,
    aclsparseDnVecDescr_t vecY,
    aclDataType computeType,
    aclsparseSpSVAlg_t alg,
    aclsparseSpSVDescr_t spsvDescr);

aclsparseStatus_t aclsparseSpSV_updateMatrix(
    aclsparseHandle_t handle,
    aclsparseSpSVDescr_t spsvDescr,
    void *newValues,
    aclsparseSpSVUpdate_t updatePart);
```

算法枚举必须是：

```cpp
ACL_SPARSE_SPSV_ALG_DEFAULT
```

#### 2. 参数校验

在现有 `ValidateSpSVCommonParams` 基础上扩展：

```text
computeType ∈ {ACL_FLOAT, ACL_COMPLEX64}
matA.valueType == computeType
vecX/vecY 非 NULL 时：dtype == computeType，size == m
format ∈ {CSR, CSC, COO, SLICED_ELL}
base ∈ {ZERO, ONE}
fill/diag/op/alg 合法
```

还必须校验：

- `rows == cols`、`rows <= INT32_MAX`、`nnz <= INT64_MAX`，并在任何 `uint64_t -> int64_t` 转换前检查；
- `nnz > 0` 时 values 和格式所需的 index 指针非 NULL；`m > 0` 时 row/column offset 指针非 NULL；
- SELL 的 `numSlices/sliceWidth/slice pointer` 元数据自洽；
- Analysis 的 `externalBuffer` 非 NULL 且满足已确认的 workspace 首地址对齐合同（当前 README/源码约定为 512B）；
- Solve 的 vec values 非 NULL、连续、与矩阵位于同一 Device；Public descriptor 没有 stride 字段，因此“连续”是创建描述符时的调用者合同；
- workspace 大小计算对乘法、加法和对齐全部做 checked arithmetic，溢出返回 `INVALID_VALUE` 或 `INSUFFICIENT_RESOURCES`，错误码在实现前统一。

索引策略：

- **任务声明能力**：I32；
- **兼容 master**：继续接受当前 `(I32,I32)`、`(I64,I32)`、`(I64,I64)`；
- 不新增 `(I32,I64)`。

不能凭 `void* newValues` 推断其长度/dtype，也不能凭 `void *externalBuffer` 推断分配容量。公开 UpdateMatrix 只校验可验证项：handle/spsvDescr/state/newValues/updatePart；类型和长度由 Analysis 时 dtype/shape 合同定义。PyTorch adapter 因为持有 Tensor，可额外检查 dtype/numel/device/workspace storage bytes。

错误校验必须按 ABI 能力分层，禁止在 README 承诺无法实现的同步错误码：

| 类别                                                            | 可执行处理                                                                                 |
| ------------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| Host metadata（NULL、enum、shape、dtype、descriptor identity、已知对齐） | API 返回前同步校验并返回 `aclsparseStatus_t`                                                    |
| Device index 内容（越界、rowPtr 非单调、末项与 nnz 不符）                     | Analysis 首个 NPU validation phase 检查并阻止后续越界访问；错误在 stream 同步点报告，除非任务方允许 Analysis 同步 D2H |
| workspace 实际容量、悬空 Device pointer                              | 裸 C API 无容量/所有权信息，属于调用者合同；ATen bridge 可通过 Tensor metadata/所有权补充校验                     |

任务书若坚持“Device index 内容错误必须由 `aclsparseSpSV_analysis` 同步返回 `aclsparseStatus_t`”，就与“Analysis 异步、无 D2H 同步”冲突，必须由任务方二选一并写入最终接口说明。

#### 3. Analysis 状态必须补齐

当前 master 已缓存 m/nnz/format/fill/diag/op/base/index widths/currentValues/workspace offsets，但任务明确要求 dtype、pointer mode 和 descriptor/parameter 一致性。

建议至少新增/确认：

```text
analysisLaunched
cachedHandle                    // handle identity
cachedStream                    // Analysis 提交 stream
cachedMatADescr                  // descriptor identity
cachedM / cachedNnz
cachedFormat
cachedFillMode / cachedDiagType
cachedOpA
cachedValueType / cachedComputeType
cachedPointerMode
cachedAlphaHostBytes             // Host mode：4B/8B 值快照
cachedAlphaDevicePtr             // Device mode：指针 identity
cachedIdxBase
cachedIndexType / cachedColIndType / cachedPermType
cachedPtrs / cachedIdxs           // 稀疏结构 pointer identity
cachedVecXDescr / cachedVecYDescr // Analysis 非NULL时才绑定
vecXDeferred / vecYDeferred
workspaceBuffer / workspaceSize
currentValues
csrValuesOffset / permOffset
transValuesOffset / transPermOffset
diagPtrOffset
updateMatrixCalled
```

Solve 校验：

- opA / computeType / alg 一致；
- handle identity 一致；UpdateMatrix 也执行同样校验；
- matA descriptor identity 与结构属性一致；
- format/fill/diag/base/index/dtype 以及结构指针一致；
- handle 当前 pointer mode 与 Analysis 缓存一致；
- Host alpha 比较缓存字节值，Device alpha 比较指针 identity；
- Analysis 非 NULL vec descriptor 时要求 identity/size/dtype/device/values pointer 一致；
- Analysis NULL vec 时按 deferred rule 校验 Solve 的 descriptor；
- workspace 仍绑定原 externalBuffer；
- values 指针变化必须经 UpdateMatrix，或属于现有 workspace-copy 路径的合法更新状态。

stream 规则采用可验证的保守合同：Analysis、UpdateMatrix、Solve 必须在缓存的同一 handle stream 上按序提交；若 handle stream 已改变则返回 `INVALID_VALUE`。若后续确需跨 stream，必须在 bridge/API 内显式记录并等待 event，不能只替换 handle 的 stream 指针。

> externalBuffer 是否“已被调用者释放”通常无法仅凭裸指针在 Host 可靠判断；接口能做的是绑定指针并声明生命周期合同。过早释放应通过运行时/UT/内存检查工具验证，不能承诺一个不存在的同步错误码检测能力。

#### 4. Workspace 规划：复用现有布局

当前 master 的真实布局为：

```text
[ aligned SpsvTilingData ]
[ optional csrRowPtr ]
[ optional csrColInd ]
[ optional csrValues ]
[ optional perm ]
[ diagPtr[m] ]
[ levelPtr[m+1] ]
[ levelRow[m] ]
[ validCount[m] ]
[ optional transRowPtr ]
[ optional transColInd ]
[ optional transValues ]
[ optional transPerm ]
```

现有策略：首部 512B 对齐，内部子区 64B 对齐。首部预留区还应保存 `validationStatus`（不额外扩展一个大数组），供后续 Analysis/Solve phase 在 Device 上发现非法索引后提前退出，避免越界访问。

complex64 的最关键修复：

```cpp
size_t valueSize = (computeType == ACL_COMPLEX64) ? 8u : 4u;
```

将：

```cpp
nnz * sizeof(float)
```

替换为 checked arithmetic 的：

```cpp
nnz * valueSize
```

仅影响 values 相关区：

- `csrValuesOffset`
- `transValuesOffset`
- CSC+H 的 conjugated-values 区（可复用 `transValuesOffset` 语义，但不得同时假装存在结构转置）

rowPtr/colInd/perm/diagPtr 的元素宽度继续按当前 index/perm 逻辑决定，**不能跟着 complex64 变成 8B**。

默认不新增 `levelOfRow/rowWorkClass` 等数组；只有性能证据显示必要时才扩展 workspace。

#### 5. `SpsvTilingData` 最小增量

当前结构包含 `float alpha`。建议最小改造为：

```text
alphaReal : float
alphaImag : float
alphaDevicePtr : uint64_t
valueType : int32_t       // 0=FP32, 1=complex64
needConjugate : int32_t
validationStatus : int32_t // 也可只存 workspace header，不要求重复传参
```

保留现有：

```text
m/nnz/fill/diag/op/format/index types/base/nthreads/numBlocks/offsets...
```

Kernel 入口根据 `valueType` 选择 ValueT 模板实例。不要为 dtype/format/fill/op 重新制造复杂 TilingKey 组合。

#### 6. CSC / N/T/H 归一化：必须分离“转置”和“共轭”

这是 complex64 最容易出错的点。

当前 FP32 代码对 CSC 通过“把 CSC 看成 A^T 的 CSR 视图”来翻转 fill/op；FP32 的 H 与 T 数值相同，因此可以把 op 压成“是否转置”。complex64 不能这样做。

Host/Analysis 必须显式得到两个状态：

```text
effectiveTranspose : bool
needConjugate      : bool
```

逻辑表：

| 原始 format    | op | CSC-as-CSR 视图后是否需要结构转置 | 是否共轭 values |
| ------------ | -- | ---------------------: | ----------: |
| CSR/COO/SELL | N  |                      否 |           否 |
| CSR/COO/SELL | T  |                      是 |           否 |
| CSR/COO/SELL | H  |                      是 |           是 |
| CSC          | N  |                      是 |           否 |
| CSC          | T  |                      否 |           否 |
| CSC          | H  |                      否 |       **是** |

fill mode 随结构转置翻转；`needConjugate` 与 fill 翻转无关。

实现建议：

- 若生成 `transValues`：写入时对 H 执行 `Conj`；
- CSC+H 没有结构转置但需要共轭：首版建议分配/复用一个 values 工作区生成 conjugated values，使 Solve 热路径不逐 nnz 分支；
- 如果为了省 workspace 选择 Solve 时 lazy conjugate，必须用 Profiler 比较，并保证 GENERAL/DIAGONAL Update 后一致；
- 无论采用哪种，**不能复用当前** **`op = (op == 0) ? 1 : 0`** **这种会丢失 H 信息的二值逻辑**。

#### 7. BufferSize

BufferSize：

- 纯 Host 计算；
- 不读取 alpha 数值，但按现有接口要求 alpha 非 NULL；
- vecX/vecY 可 NULL；
- 使用 `valueSize` 计算 complex workspace；
- CSC+H 若采用预共轭 workspace，其大小必须被 BufferSize 准确计入；
- 所有 `count * bytes`、`offset + size` 使用 overflow-safe helper。

#### 8. Analysis

在 handle stream 上提交现有 Analysis phases：

1. 格式/base 规范化；
2. Device index 内容校验并写 `validationStatus`；后续 phase 先读该状态，错误时立即退出；
3. 按需稳定地转换格式/转置结构；
4. complex H 按 `needConjugate` 生成 values；
5. 计算依赖 level；
6. 生成 `diagPtr/levelPtr/levelRow/validCount`；
7. 写运行时 `numLevels`；
8. 缓存完整 Analysis 状态和 workspace 绑定。

不把结构拷回 CPU 计算。

为满足 bitwise deterministic，首版直接复用当前单线程 `SpsvBuildCsrFromCoo` / `SpsvTransposeCsr` 的 source-order stable scatter；不能走当前使用 atomic slot allocation 的 parallel 变体。若 Analysis 性能不达标，再实现稳定并行 counting/radix scatter，并以“同一输入多次 Analysis+Solve 输出逐 bit 相等”验收，不能仅比较容差。

#### 9. UpdateMatrix

##### GENERAL

语义：newValues 为原 sparse pattern 顺序下的完整 nnz values。

- direct CSR N：沿用当前“更新 currentValues 指针”快路径；
- COO/SELL/base1 等 workspace CSR：按 perm 更新 workspace values；
- 仅转置：按 `transPermToCsr[p]` 更新 transValues；
- 先 COO/SELL 转 CSR 再转置：必须按 `source = permCsrToSrc[transPermToCsr[p]]` 从原始 `newValues[source]` 取值；
- H：在上述映射后再执行 `Conj`；
- CSC+H 若使用 conjugated workspace：更新时也必须重新共轭。

映射定义必须在代码和 UT 中固定：

```text
permCsrToSrc[q]    : 中间 CSR 位置 q -> 原始格式 values 位置
transPermToCsr[p]  : 最终转置 CSR 位置 p -> 中间 CSR 位置 q
finalToSrc[p]      : permCsrToSrc[transPermToCsr[p]]
```

不要求额外物化 `finalToSrc[nnz]`；update kernel 可做两级 gather。若为性能物化组合映射，BufferSize 必须计入且由 Profiler 证明收益。

结构不变，Level Scheduling 不重算。

##### DIAGONAL

公开 API 语义：`newValues[i]` 是第 i 行的新对角值，长度 m。

- `diagPtr[i] >= 0`：更新已有结构中的对应对角值；
- `diagPtr[i] < 0`：不能通过 values-only API 凭空增加结构项，保持 missing；NON\_UNIT Solve 继续按除 0 传播 INF/NAN；
- H 的对角满足 `conj(a_ii)`，若 Solve workspace 存的是预共轭值，DIAGONAL 更新也必须写共轭后的 `newDiag[i]`。

专项 PyTorch hook 的输入是完整 nnz values，因此 adapter 需要先在 NPU 生成 m 长度 diagonal buffer，再调用 C API。Python 专项包只生成 CSR，bridge 可在 Analysis 时用一个 NPU kernel 生成并持有 `sourceDiagPos[m]`；不得逐行 D2H 查找。对通用 C API 的 COO/SELL/CSC DIAGONAL 路径，`diagPtr` 是最终有效 CSR 位置，写入预共轭 workspace 时仍按逻辑行 `i` 读取 `newDiag[i]`。

#### 10. ATen Dispatcher / 测试 Hook

专项测试固定查找：

```text
torch.ops.ops_sparse_test.spsv_analysis_npu
torch.ops.ops_sparse_test.spsv_update_npu
torch.ops.ops_sparse_test.spsv_npu
```

推荐 C++ 扩展使用 PyTorch `TORCH_LIBRARY` 定义 operator schema，并用项目当前 torch\_npu 的 backend registration 方式实现。PyTorch 官方确认 Library API 可注册 C++ 自定义 operator 和 custom class，且“operator 使用 custom class 前必须先注册 class”。PrivateUse1 是通用第三方 backend 机制，但**实际 NPU dispatch key、torch\_npu stream/allocator API 和可接受 schema 必须以任务镜像头文件编译结果为准**。本机未安装 PyTorch/torch\_npu，本文不声称下列概念 schema 已在目标版本编译通过。

Plan 推荐使用注册的 C++ custom class/opaque holder：

```text
SpSVPlan
 ├─ aclsparseHandle / stream context
 ├─ matA descriptor
 ├─ spsvDescr
 ├─ external workspace Tensor
 ├─ rowOffsets/colIndices/values Tensor refs
 ├─ diag metadata needed by adapter
 ├─ optional diagonal-gather Tensor
 ├─ pending calls: X/Y descriptors + Tensor refs + completion event
 ├─ dtype/op/fill/diag/base/pointerMode
 └─ lifetime guard
```

schema 约束：

- `analysis_npu` 返回已注册的 `SpSVPlan` custom class；class 注册必须出现在引用它的 operator schema 之前；
- `update_npu` 返回同一个 plan（或 `None`，adapter 两者均接受），不得隐式复制底层 descriptor；
- `spsv_npu` 的 `alpha` 同时可能是 Python number 和 0-D NPU Tensor，必须在目标 PyTorch 版本选用实际可编译的 `Any` boxed schema 或等价单入口方案；不能注册两个只能通过 `.overload` 名称调用的 overload，因为评测固定调用默认名字；
- 三个 operator 都至少有一个 NPU Tensor 实参用于 backend dispatch；若 custom class/`Any` 导致 dispatch key 提取不符合预期，使用目标 torch\_npu 认可的 catch-all/boxed registration，但仍保持固定 Python 名称；
- 加载扩展后先用 dispatcher introspection/最小 Python 调用验证三个 schema，再运行 ATK。

职责：

- `analysis_npu`：根据 CSR tensors 创建 matA descriptor；由于固定 hook 没有 X/Y 参数，BufferSize 和 Analysis 必须传 `vecX=nullptr, vecY=nullptr`，走经任务方确认的 deferred binding；随后分配 NPU workspace 并提交 Analysis；
- `update_npu GENERAL`：Tensor metadata 校验后直接传 full values；
- `update_npu DIAGONAL`：从 full values 按 Analysis 结构在 NPU gather `diagValues[m]`，再调用 DIAGONAL C API；
- `spsv_npu`：为本次 x/新分配 y 创建 X/Y descriptor，按 deferred rule 校验，解析 host scalar 或 0-D Device Tensor alpha，调用 Solve，返回 NPU Tensor；
- 不使用 `torch.linalg.solve_triangular` / CPU dense solve 作为 P case NPU fallback。

bridge 的每次调用还必须执行以下运行时合同：

1. 以输入 Tensor 的 device 建立 DeviceGuard，校验所有 Tensor 同 device、contiguous、dtype/numel 正确；row offsets/columns 为 I32。
2. 获取 **torch\_npu current stream**，设置到 `aclsparseHandle`。首版要求它与 Analysis 缓存 stream 相同；不满足则报错，不隐式全局同步。
3. Host alpha 从 boxed scalar 显式转换为 `float` 或 `aclsparseComplex`；Device alpha 必须是同 device、同 dtype、contiguous 的 0-D Tensor，并把 `data_ptr()` 传给 C API。
4. plan 强引用 rowOffsets/colIndices/values/workspace/sourceDiagPos/diagValues Tensor；每次 Solve 还把 X/Y descriptor、X/Y Tensor 和 Device alpha Tensor 放入 pending-call 记录，直到对应 completion event 完成后才回收。
5. 对可能由 PyTorch caching allocator 提前复用的 Tensor，调用目标 torch\_npu 提供的 record-stream 等价 API；若该版本没有可靠接口，plan 在析构前等待最后记录的 NPU event，再销毁 C descriptors/handle。禁止让析构后的 workspace 被未完成 Kernel 使用。
6. 所有 API 返回码转成包含接口名和状态值的 `TORCH_CHECK`；Kernel 的异步错误在测试的 event/stream 同步点检查。

专项 adapter 的 `alpha` 类型在 host mode 为 Python scalar、device mode 为 0-D Tensor；operator schema/boxed IValue 处理必须能稳定区分这两种输入。精确 schema 写法以任务 PyTorch 版本编译通过为准，不能凭文档假设一个版本不支持的 union schema。

#### 11. ATen adapter 构建与加载

生产 `libops_sparse.so` 不应被迫依赖 PyTorch。新增独立、仅测试/评测启用的 shared library，例如：

```text
test/spsv/arch35/torch_adapter/
├── spsv_torch_adapter.cpp   // custom class + TORCH_LIBRARY/IMPL
└── CMakeLists.txt            // link libtorch、torch_npu、ops_sparse
```

构建目标显式接收任务环境的 PyTorch/torch\_npu include、library 和 C++ ABI 配置；构建后先执行加载冒烟：

```python
torch.ops.load_library(adapter_so)
assert hasattr(torch.ops.ops_sparse_test, "spsv_analysis_npu")
assert hasattr(torch.ops.ops_sparse_test, "spsv_update_npu")
assert hasattr(torch.ops.ops_sparse_test, "spsv_npu")
```

任务包当前没有上述加载动作。应在 `function_sparse_ops.py` 和 benchmark 共同导入的 backend loader 中，通过明确的安装模块或受控环境变量（例如 `OPS_SPARSE_TEST_EXTENSION`）加载一次；不能依赖父进程偶然预加载，因为 ATK 可能创建独立 executor 进程。loader 必须校验文件存在、加载异常和三个 schema，失败时立即报错。

### 3.2.2 Kernel 侧设计

#### 1. 类型模板化

当前 Kernel 的 `float*` 改造为：

```cpp
template <typename ValueT, typename RowPtrT, typename ColIndT, typename PermT>
...
```

其中：

```text
ValueT = float | SpsvComplex64
RowPtrT/ColIndT/PermT = 复用 master 现有 index dispatch
```

不要把 index 类型和 value 类型耦合。

#### 2. Analysis 格式转换

当前已有：

- COO/SELL → workspace CSR；
- base1 → 0-base workspace；
- 转置 CSR；
- perm/transPerm；
- Level Scheduling。

本任务只把 values copy 泛化为 ValueT，并增加 `needConjugate`：

```cpp
transValues[pos] = needConjugate
                 ? ValueOps<ValueT>::Conj(values[p])
                 : values[p];
```

对 FP32，Conj 是 identity，保持现有结果。

#### 3. Solve baseline：保持一线程一行

当前真实热路径：

```text
for level in levels:
    parallel rows in level
        one thread solves one row
    level barrier
```

complex64 只替换数值操作：

```text
sum = alpha * x[row]
for dependency p:
    sum -= value[p] * y[col[p]]
if NON_UNIT:
    sum /= diagonal
write y[row]
```

优点：

- 改动最小；
- deterministic 顺序与 FP32 逻辑一致；
- 先建立正确的 complex64 baseline；
- 便于判断性能瓶颈是否来自 complex arithmetic 还是结构调度。

#### 4. Level 同步

现有 single-block 使用 block 内同步，多 block Solve 每个 level 后使用 arch35 的跨核 `SyncAll()`。保持现有机制，避免自行引入新的同步协议。

CANN 950 SIMT 官方模型采用线程/线程块执行并支持共享内存；线程块线程数应按现有实现和编译器限制选择。当前 SpSV master 已使用 `kSimtMaxThreads=2048` 并有 64/128/256/2048 线程 heuristic，本任务优先保留，实机 Profiler 再调整。

#### 5. complex64 复除

NON\_UNIT 是 complex64 的额外高成本操作。实现要求：

- finite 常规值保持较好的数值稳定性；
- 分母 0+0i 产生规格允许的 INF/NAN；
- 不在 Host 或 CPU 做复除；
- 对 H 使用已经规范化/共轭的 diagonal value。

#### 6. 原地 X/Y

任务允许：

```text
vecX.values == vecY.values
```

现有 Level Scheduling 可支持这一语义，前提是每行先读取本行 RHS：

```text
rhs = alpha * x[row]
```

随后再覆盖 `y[row]`；依赖读取只访问已完成行的 Y。不得增加“先整向量 Copy X→临时区”这种无必要 workspace，除非实测证明现有 alias 路径存在正确性问题。

#### 7. `nnz == 0`

当前 source 实现：

- UNIT：`Y = alpha * X`；
- NON\_UNIT：按奇异系统除零语义生成 INF/NAN。

complex64 需提供等价 helper：

```text
scale_copy_complex
scale_inf_nan_complex
```

注意当前 README 若仍写成 NON\_UNIT `Y=0`，应以 source + 任务书为准并同步修 README，避免文档/代码冲突。

#### 8. 未排序与确定性

任务要求未排序输入确定性。

对 direct CSR，现有 Solve 按原始行内存储顺序串行累加，只要 Analysis 不改写该顺序即可确定。对 COO→CSR 和所有 transpose 路径，当前 parallel scatter 使用整数 `asc_atomic_add` 分配目标槽位；计数值虽确定，**同一目标行中元素的相对顺序没有源码或官方保证**，因此不能据此宣称浮点结果 bitwise deterministic。

首版确定性合同：

- 不使用浮点 atomic add 做行内求和；
- direct CSR 保留 source order；
- COO 转 CSR 按 `(row, originalPosition)` stable；transpose 按 `(newRow, intermediatePosition)` stable；
- format/transpose mapping 和 permutation 每次 Analysis 完全相同；
- 同一线程的行内累加顺序固定；
- 若后续增加长行线程组归约，使用固定 lane 切分 + 固定树形 reduction。

验收同时覆盖“同一 plan 重复 Solve”和“每次重建 plan 后 Analysis+Solve”，FP32/complex64、CSR/COO/SELL、N/T/H 各比较输出原始字节。只验证数值容差不能证明 bitwise deterministic。

### 3.2.3 性能优化设计

#### 1. 优化优先级

**P0：先复用现有 baseline**

```text
existing Level Scheduling
+ existing single/multi-block heuristic
+ ValueT generic
+ only required complex values/status workspace
```

**P1：Profiler 驱动**

重点观察：

- Solve 各 level 的有效线程比例；
- `one_long_row/highly_imbalanced/power_law` 的单线程长行耗时；
- complex mul/div 指令占比；
- GM/L2/Data Cache stall；
- 多 block `SyncAll` 占比；
- Analysis 转置/格式转换与 Solve 的时间比例。

#### 2. 长行协作仅作为候选

如果 Profiler 明确显示长行单线程成为关键路径，可增加：

```text
short row  -> current 1 thread / row
long row   -> 1 warp or fixed thread group / row
```

要求：

- threshold 通过 206 cases 实测选取，不写死为“理论最优”；
- 固定 reduction tree，保持 bitwise deterministic；
- 不使用 floating atomic reduction；
- 仅对足够长行启用，避免短行同步开销；
- 若需要 Shared Memory，计入 UB 资源和 occupancy 影响。

#### 3. 内存与 Cache

CANN 950 SIMT 官方资料确认 SIMT 线程访问 Global Memory、共享内存（UB 资源）和寄存器；全局访问经过芯片缓存体系。SpSV 的 `y[col]` 属于离散读，应依赖硬件缓存并通过并发线程隐藏延迟。

设计原则：

- 不尝试把整个 Y 或 A 缓存在 Shared Memory；
- Shared Memory 只用于可证明收益的线程组 reduction/少量局部数据；
- 不把传统 TPipe `CopyIn→Compute→CopyOut` DoubleBuffer 生搬到随机访问 Solve；
- COO/SELL/base1/transpose 等连续转换阶段若要 DoubleBuffer，需目标编译环境验证 + Profiler 证明；
- 不把未核验的 `DataCachePreload` API 作为基线依赖。

#### 4. 线程数 / 核数

保留 master 的现有 heuristic 作为首版：

```text
m <= 64   -> 64 threads
m <= 128  -> 128
m <= 256  -> 256
else       -> kSimtMaxThreads
```

多核数继续受：

- m；
- nnz/m 粗略 level 深度估计；
- AIV core 数；
- useful cores

控制。

只有 Profiler 证明 complex64 的 occupancy/register pressure 导致该 heuristic 退化时，再按 dtype 增加调优分支。

***

## 支持硬件

| 平台           | 当前 SpSV master | 本任务                  |
| ------------ | -------------- | -------------------- |
| Ascend 950PR | 支持，arch35      | √                    |
| Ascend 950DT | 支持，arch35      | √，以任务实际 SOC 为准       |
| Atlas A3     | SpSV 不支持       | 不新增 SpSV；只做共享代码/构建回归 |
| Atlas A2     | SpSV 不支持       | 不新增 SpSV；只做共享代码/构建回归 |

不在设计中硬编码 950 的 L2 容量、AIV 核数量等易随 SOC/版本变化的绝对值；验收环境记录真实 SOC/CANN/Driver/Profiler 信息。

## 算子约束限制

1. A 必须是方阵；fill 外三角项按三角语义忽略。
2. NON\_UNIT 缺失/零对角按任务要求传播 INF/NAN。
3. task 声明 Device I32；代码不得回归 master 已有 I64 index 组合。
4. `m <= INT32_MAX` 延续当前 master 内部 level row 表示约束。
5. BufferSize/Analysis 可 NULL vec descriptor；deferred binding 是本文的可执行候选，须经任务方确认后锁定。
6. `newValues` 是裸 Device pointer；公开 UpdateMatrix 无法独立验证 dtype/长度，调用者必须遵守 Analysis dtype/size 合同。
7. DIAGONAL 只更新已有结构的对角值，不插入新的非零结构。
8. duplicate 数值语义**未被任务包 Golden 明确定义**，且 cuSPARSE 对重复索引不保证总是正确；在任务方给出定义前，不宣称 duplicate 数值与 cuSPARSE 等价。
9. 所有核心格式处理和求解在 NPU Kernel 上完成，禁止 CPU fallback。

***

# 可维可测分析

## 精度标准 / 性能标准

| 验收项              | 标准                             | 来源                        |
| ---------------- | ------------------------------ | ------------------------- |
| FP32 Golden      | CPU float64                    | 任务书                       |
| complex64 Golden | CPU complex128                 | 任务书                       |
| rtol             | `2^-10`                        | 任务书                       |
| atol             | `2^-16`                        | 任务书                       |
| A                | `1e-2`                         | 任务书                       |
| 匹配率              | ≥ 0.99                         | 任务书                       |
| 绝对误差 cap         | `max(A, 32×ULP(golden))`       | 任务书                       |
| complex 比较       | real/imag 分别应用全部规则             | 任务书                       |
| deterministic    | 重复执行 bitwise 一致                | 任务书 / cuSPARSE solve 参考语义 |
| 性能               | GPU benchmark / NPU ≥ 0.3      | 任务书                       |
| 采样               | warmup 10；sample 30；median/P90 | 任务书                       |

## 必测测试矩阵

### Python 专项包直接覆盖

- CSR；
- FP32/complex64；
- base0/1；
- N/T/H；
- lower/upper；
- unit/nonunit；
- accuracy 中 host/device alpha；
- accuracy 中 GENERAL/DIAGONAL；
- performance 中 Device alpha + GENERAL；
- 8 类 row pattern。

### C++ UT/ST 必须补齐

| 类别     | 必测项                                                                                            |
| ------ | ---------------------------------------------------------------------------------------------- |
| 格式     | CSC / COO / SLICED\_ELL + CSR                                                                  |
| 边界     | m=0/1，nnz=0/1，空行                                                                               |
| dtype  | FP32/complex64，Host/Device alpha ABI                                                           |
| 复数     | N/T/H，特别是 CSC+H、GENERAL/H、DIAGONAL/H                                                           |
| 映射     | COO/SELL + T/H + GENERAL，验证 `perm[transPerm[p]]`；base0/1                                       |
| alias  | X/Y 同 Device values                                                                            |
| 生命周期   | NULL vec Analysis、deferred bind、descriptor 变化、pointer mode 变化、externalBuffer 绑定、Destroy        |
| Update | GENERAL；DIAGONAL m-values；missing diag                                                         |
| index  | I32 任务主线；master I64 组合回归                                                                       |
| 异常     | 非方阵、非法 enum、dtype mismatch、invalid base/index、空 pointer、workspace 错误；Device 内容错误在 stream 同步点检查 |
| 数值     | zero/missing diag INF/NAN、UNIT 忽略存储对角、特殊值                                                      |
| 确定性    | 未排序、多次运行 bitwise；格式转换/转置后重复执行                                                                  |

专项 `accuracy_sparse_ops.py` 当前只继承 ATK 的 `MixedToleranceBenchmarkAccuracyCompare` 并放宽 dtype 不同情况下的 shape 检查，仓内没有证据表明它实现了任务书的全部复合判据。交付前必须新增独立、可复现的 comparator 或显式配置，并逐元素实现：

```text
mixed_ok = abs(actual - golden) <= atol + rtol * abs(golden)
cap_ok   = abs(actual - golden) <= max(A, 32 * ULP(golden))
pass     = match_rate(mixed_ok && cap_ok) >= 0.99
```

complex64 对 real/imag 分别统计；NaN/Inf 按分类和 Inf 符号比较，不让普通 `allclose` 隐式决定。`ULP(golden)` 使用哪一种表示（float64/complex128 Golden 本身，还是投影到目标 FP32 后的 ULP）任务书未写清，验收前必须确认并在报告中记录实现，不能静默选择较宽标准。

## DIAGONAL adapter 专项验收

增加测试：

1. 构造 full CSR values，其中对角项不位于 `values[0:m]`；
2. Python `_updated_values()` 只修改结构中的真实 diagonal positions；
3. `spsv_update_npu(..., fullValues, DIAGONAL)`；
4. adapter gather 得到 `diagValues[i] == fullValues[diagPos[i]]`；
5. C API UpdateMatrix 使用 `diagValues[m]`；
6. Solve 与 CPU Golden 一致。

该用例能直接防止“把 full values 当 m 对角数组”这一严重错误。

## 性能测试

使用任务包实际脚本：

```text
benchmark_cusparse_gpu.cu
benchmark_sparse_ops_gpu.py
benchmark_sparse_ops_npu.py
```

内存：

```text
collect_sparse_ops_gpu_memory.py
collect_sparse_ops_npu_memory.py
compare_sparse_ops_memory.py
```

P case `m > 4096` 时专项 adapter 禁止 reference fallback，必须注册真实 NPU Hook。

### P-01～P-03 基线冲突记录

任务包存在两套不一致的原生 cuSPARSE 结果；任务书与 `baseline_results/gpu_full_results.tsv` 一致，而 `gpu_performance_result_benchmark.md` 是另一组：

| Case | 任务书 / `baseline_results` (us) | `gpu_performance_result_benchmark.md` (us) |
| ---- | ----------------------------: | -----------------------------------------: |
| P-01 |       40,413.695 ～ 40,458.531 |                  514,144.094 ～ 514,203.250 |
| P-02 |       93,850.750 ～ 94,739.203 |              1,228,205.625 ～ 1,230,583.375 |
| P-03 |     220,676.516 ～ 222,887.562 |              3,078,026.500 ～ 3,078,255.750 |

`baseline_results/manifest.json` 记录来源文件名 `cuda_result_H100` 和 206 条有效记录；另一 Markdown 报告只说明 2026-08-27 的外部 `cuda_result/cuda_result2`，且未内嵌 GPU 型号。二者不能被当作同一次采集。更严重的是，manifest 记录的 `gpu_full_results.tsv` SHA256 为 `8ed756775032af24c344b29c18bcd0890320c704e380e33062e8282c97e88fd7`，当前文件实测为 `331e0e5c5284d7a3e14a19daf92fcd36ea157f6b9e88fd98b8871f220b53769d`，完整性校验不通过。

这不是实现可以自行“修正”的问题。验收前必须：

1. 用任务指定环境重新运行原生 GPU benchmark；
2. 修复 manifest/TSV 不一致后重新校验 SHA256，并确认评测脚本实际读取/采用哪一份 baseline；
3. 自测报告记录 GPU 型号、CUDA/cuSPARSE 版本、case id、测量范围；
4. 0.3× 判定以**最终验收指定的同调用范围 GPU Event baseline**为准，不在 Kernel 中硬编码上述任一数字。

## 性能测量范围

当前脚本存在实质性的非对称测量：

- `common/benchmark_runner.py` 先调用 `reset()`（其中包含 `update`），之后才记录 start Event，所以 NPU `median_us` 只含 `spsv_npu` Solve；
- `benchmark_cusparse_gpu.cu` 分别测 refresh（update 或 reanalysis）和 Solve，输出主字段 `median_us` 时对两者逐样本求和；
- 原生 GPU 与 NPU 还使用不同的数值生成方式和 alpha 数值，虽结构相同，但不能称作完全相同输入。

因此当前主字段不能直接相除。验收前必须选择并对称实现以下一种口径：

1. **Solve-only（推荐）**：两端均只比较 Analysis 后、同 update 状态下的 Solve Event；GPU 使用独立的 solve samples/median，NPU 保持当前 Event 范围。
2. **Update+Solve**：把 NPU `reset()` 内的 update 移入 Event，并在 GPU 端使用相同更新策略；旧 cuSPARSE 若只能 reanalysis，必须单列，不能与 NPU UpdateMatrix 混比。

无论选哪种，报告仍须拆分并说明：

```text
Analysis
UpdateMatrix
Solve
Update + Solve（若评测总范围如此定义）
```

不得把 Solve-only 与 Update+Solve 混用，也不得只给一个更小的 Kernel 局部耗时冒充评测总调用耗时。最终脚本需写入 `measurement_scope` 字段，并用同 case id、dtype、结构和 exact alpha/value seed 做交叉校验。

## 内存验收

按任务书二选一：

1. I/O 总量 >500MB：NPU 相对 GPU 的 extra peak ≤ GPU total 的 50%；或
2. 固有 workspace ≤ 目标硬件 L2 Cache 容量。

当前任务包不能直接完成该验收：

- `compare_memory.py` 把比例写死为 5%，与任务书 50% 冲突；
- 按评测脚本的 index/value/X/Y/alpha storage 口径计算，现有 206 cases 的最大近似 I/O 为 52,428,812 bytes（P-03 complex64），没有 case 达到 500,000,000 bytes；
- `operator_adapter.py` 没有实现 runner 可调用的 `workspace_bytes` / `workspace_allocation_bytes`，因此报告中的 workspace 为 NULL，L2 路径也不能判定。

交付前必须由任务方确认 5%/50% 后统一脚本和 README；同时让 plan/adapter 导出 BufferSize 查询值与实际 workspace Tensor storage bytes，写入同一 case fingerprint 的结果。L2 容量必须来自**验收环境/目标 SOC 的权威信息**，不在 Legacy API 运行时强行依赖不存在的 TilingContext 查询。设计侧保证查询值、实际分配值、报告值三者一致。

## Profiler 证据

至少提供：

- FP32 与 complex64 各代表性大 case；
- uniform + one\_long\_row + highly\_imbalanced/power\_law；
- N 与 H；
- Analysis / Update / Solve timeline；
- AI Core 利用、内存 stall、同步开销、热点 Kernel；
- 若引入长行协作：优化前后同 case 对比。

***

# 兼容性分析

1. 公开 `aclsparseSpSV_*` ABI 不改变。
2. FP32 走原有 ValueT=float 路径，必须与 master 结果/性能基本无回归。
3. complex64 新增 descriptor dtype/state 字段不能改变其他算子 ABI；若修改公共 internal descriptor，执行全仓构建测试。
4. 保留 SpSV 当前 I64 index 组合；任务 I32 只是新增能力验收范围，不等于允许删除 I64。
5. A2/A3 当前没有 SpSV 支持，因此只验证共享头文件/公共 Host utility/构建和其他算子不回归，不声称 SpSV A2/A3 功能验证。


# 参考资料与证据

## A. 任务材料

1. `9月社区任务-aclsparseSpSV算子开发(950)/aclsparseSpSV_A5_task_doc.md`：本任务功能、精度、性能、内存和交付要求。
2. `9月社区任务-aclsparseSpSV算子开发(950)/test_cases/aclsparseSpSV_testCase/operator_adapter.py`：专项 hook 真实调用合同、CSR generator、DIAGONAL full-values 输入。
3. 同目录 `accuracy_cases.json`：200 个 accuracy cases。
4. 同目录 `performance_cases.json`：206 个 performance cases。
5. 同目录 `gpu_performance_result_benchmark.md`：任务包内另一组原生 cuSPARSE P-01～P-03 采集结果。
6. `cann-ops-competitions/04_tasks/01_community-task-2026/resources/design_template.md`：仅作为文档章节模板。

关键静态证据定位：

| 结论                           | 文件 / 符号                                                                                                                               |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| NPU runner 的 Event 不含 update | `9月社区任务-aclsparseSpSV算子开发(950)/test_cases/common/benchmark_runner.py::run`（`reset()` 在 start Event 前）                                 |
| 原生 GPU 主耗时为 refresh+solve    | `benchmark_cusparse_gpu.cu::run`（`all.push_back(refresh.back()+sol.back())`）                                                          |
| Python 只构造有序 CSR 和显式对角       | `operator_adapter.py::_csr`                                                                                                           |
| DIAGONAL hook 收到完整 values    | `operator_adapter.py::_updated_values` / `prepare_case::reset`                                                                        |
| 内存阈值写死 5%                    | `9月社区任务-aclsparseSpSV算子开发(950)/test_cases/common/compare_memory.py::main`                                                             |
| case 生成依赖缺失                  | `aclsparseSpSV_testCase/generate_cases.py` 第 5～6 行导入                                                                                  |
| ATen hook 没有显式加载链            | `9月社区任务-aclsparseSpSV算子开发(950)/test_cases/run_accuracy_atk.sh`、`common/backend.py::resolve_hook`，仓内无 `TORCH_LIBRARY` / `load_library` |

## B. ops-sparse master

主仓任务书地址：

- <https://gitcode.com/cann/ops-sparse>

核验的 arch35 文件：

- `sparse/spsv/arch35/spsv_host.cpp`
- `sparse/spsv/arch35/spsv_kernel.cpp`
- `sparse/spsv/arch35/spsv_tiling_data.h`
- `sparse/spsv/README.md` / `sparse/spsv/` 接口说明

| 结论                                            | 文件 / 符号                                                                                  |
| --------------------------------------------- | ---------------------------------------------------------------------------------------- |
| complex public ABI                            | `include/cann_ops_sparse.h::aclsparseComplex`                                            |
| FP32 校验/workspace/scalar 强绑定                  | `spsv_host.cpp::ValidateSpSVCommonParams` / `ComputeWorkspaceOffsets` / `ReadHostScalar` |
| CSC 二值 op 归一化                                 | `spsv_host.cpp::BuildTilingData`                                                         |
| 当前 descriptor 缺少 dtype/handle/vector identity | `sparse/common/aclsparse_spsv_descr.h::aclsparseSpSVDescr`                               |
| atomic COO/transpose scatter                  | `spsv_kernel.cpp::SpsvBuildCsrFromCooParallel` / `SpsvTransposeCsrParallel`              |
| 一线程一行串行累加                                     | `spsv_kernel.cpp::SpsvSolveRow`                                                          |
| GENERAL update 当前只选择一段 mapping                | `spsv_host.cpp::UpdateMatrixGeneral` / `spsv_kernel.cpp::SpsvUpdateValuesSimtCompute`    |

本次静态核验 commit：`e0015bb76090487165f43dbaf2aae37648887a4b`。合入开发时必须再次以 `cann/ops-sparse` 目标分支实际 commit 为准，因为 master 可能继续变化；不使用个人镜像作为事实来源。

## C. NVIDIA cuSPARSE

- cuSPARSE API Reference / `cusparseSpSV()`：\
  <https://docs.nvidia.com/cuda/cusparse/#cusparsespsv>

用于核验：生命周期、四格式、N/T/H、NULL vec、unsorted、in-place、Solve async/deterministic、UpdateMatrix。

- cuSPARSE Storage Formats：\
  <https://docs.nvidia.com/cuda/cusparse/storage-formats.html>

用于核验：CSR/CSC/COO 的 unsorted 支持，以及 duplicate index 不保证数值正确。

## D. 华为 CANN / Ascend C

- CANN 9.1.0-beta.3，SIMD 与 SIMT 混合编程 / SIMT API 编程模型：\
  <https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910beta3/API/ascendcopapi/atlasascendc_api_07_10838.html>
- CANN 9.1.0-beta.3，AI Core SIMT 编程——内存层级：\
  <https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910beta3/programug/Ascendcopdevg/docs/guide/%E7%BC%96%E7%A8%8B%E6%8C%87%E5%8D%97/%E7%BC%96%E7%A8%8B%E6%A8%A1%E5%9E%8B/AI-Core-SIMT%E7%BC%96%E7%A8%8B/%E5%86%85%E5%AD%98%E5%B1%82%E7%BA%A7.md>
- ACL `aclDataType`（包含 `ACL_COMPLEX64`；枚举存在不等于证明 Kernel 复数算术 API 可用）：\
  <https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta2/API/appdevgapi/aclcppdevg_03_1346.html>

## E. PyTorch

- Torch Library C++ API：\
  <https://docs.pytorch.org/cppdocs/api/library/index.html>
- Operator Registration：\
  <https://docs.pytorch.org/cppdocs/api/library/registration.html>
- Custom Classes：\
  <https://docs.pytorch.org/tutorials/advanced/custom_classes.html>
- PrivateUse1 backend 集成：\
  <https://docs.pytorch.org/tutorials/advanced/privateuseone.html>

## F. 数值算法

- M. Baudin, R. L. Smith, *A Robust Complex Division in Scilab*：\
  <https://arxiv.org/abs/1210.4539>

用于核验：直接 `c²+d²` 的风险、Smith 分支及其剩余失败模式；实现必须结合本任务 FP32 范围和测试选择带 scaling 的版本。
