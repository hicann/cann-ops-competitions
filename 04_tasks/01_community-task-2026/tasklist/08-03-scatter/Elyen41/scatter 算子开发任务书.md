# 8月社区任务 - scatter 算子开发任务书

## 任务概述

参考 torch_scatter.scatter（https://pytorch-scatter.readthedocs.io/en/latest/functions/scatter.html ） 及其子算子（`scatter_sum` / `scatter_add` / `scatter_mul` / `scatter_mean` / `scatter_min` / `scatter_max`），在昇腾 NPU 上基于 **Ascend C（Kernel）+ Python（PyTorch 适配层）** 实现功能与接口完全对齐的 Scatter 系列算子，完成算子设计、开发、测试全流程工作，验收通过后将算子提交至昇腾算子开源仓 **ops-gnn**。**对标参考**：torch_scatter（https://github.com/rusty1s/pytorch_scatter ）（版本建议 ≥ 2.1.0）。

**验收口径**：以 **PyTorch 层接口** 的功能、精度、性能为唯一验收基准；C++ 层（aclnn）接口为可选交付项，不作为验收必要条件。

### 功能定义

Scatter 是一种**按索引分组的归约（reduce）**操作：将 `src` 中每个元素，沿指定维度 `dim`，根据 `index` 给出的目标位置，聚合写入 `out` 对应槽位。

设 src、index 为 n 维张量，形状为 (x_0, ..., x_i, ..., x_{n-1})。若 dim = i，则 out 的形状变为：(x_0, ..., x_{i-1}, y, x_{i+1}, ..., x_{n-1})


一维 `reduce="sum"` 时：

$$
\text{out}_k = \sum_{j \,:\, \text{index}_j = k} \text{src}_j
$$

与 `segment_coo` / `segment_csr` 不同：**scatter 不要求 index 有序**，索引可任意排列、可重复。


### 典型应用场景

图神经网络（GNN）消息聚合、点云特征池化、稀疏索引归约等，是 PyTorch Geometric 生态的核心算子之一。

---

## 核心开发要求及验收标准

开发/测试硬件和软件要求：
- **适配硬件**：Ascend 950PR
- **CANN 版本**：算子开源仓（https://gitcode.com/cann/ops-gnn ）指定版本

### 功能实现要求

#### 1. 与 torch_scatter 核心功能完全对齐

须实现以下 6 种归约模式，语义与 torch_scatter 一致：

| reduce 取值 | 对应 Python API | 语义 | 空桶输出（无 src 落入） |
|-------------|-----------------|------|------------------------|
| `sum` | `scatter(..., reduce="sum")` / `scatter_sum` | 求和 | `0` |
| `add` | `scatter(..., reduce="add")` / `scatter_add` | 同 `sum`（别名） | `0` |
| `mul` | `scatter(..., reduce="mul")` / `scatter_mul` | 连乘 | `1` |
| `mean` | `scatter(..., reduce="mean")` / `scatter_mean` | 先 sum 再除以计数 | `0` |
| `min` | `scatter(..., reduce="min")` / `scatter_min` | 求最小值 | `0` |
| `max` | `scatter(..., reduce="max")` / `scatter_max` | 求最大值 | `0` |

**各 reduce 初值与更新规则（自动创建 out 时）**：

| reduce | out 初值 | 更新规则 |
|--------|----------|----------|
| sum/add | `0` | `out[idx] += src[j]` |
| mul | `1` | `out[idx] *= src[j]` |
| mean | 先按 sum 累加，再 `out /= count` | 整数用 floor 除法，浮点用 true divide |
| min | `numeric_limits::max()` | 取更小值；无更新槽最终置 `0` |
| max | `numeric_limits::lowest()` | 取更大值；无更新槽最终置 `0` |

#### 2. index 广播（Broadcast）支持

须与 torch_scatter 广播语义一致。Host 侧或算子内完成 index 扩展，规则如下：

1. `dim < 0` 时先归一化为 `dim + src.dim()`
2. 若 `index` 为 1D：在 `dim` 之前补 leading 维，在尾部补维至 `src.dim()`，再 `expand` 到 `src.shape`
3. 广播后须满足 `src.dim() == index.dim()`
4. 对 `i = 0 … index.dim()-2`：`src.size(i) >= index.size(i)`

**验收用例参考**：torch_scatter `test/test_broadcasting.py`（1D index 广播到 4D src）。

#### 3. 输出形状推断

未提供 `out` 时，输出形状与 `src` 相同，仅 `dim` 维不同：

```
out.size(dim) =
  dim_size                        # 若指定 dim_size
  0                               # 若 index 为空且未指定 dim_size
  index.max() + 1                 # 否则
```

#### 4. 提供 out 时的原地语义

| reduce | 提供 out 时的行为 |
|--------|-------------------|
| sum/add | 在 **现有 out 值上累加**（等价 `scatter_add_`） |
| mul/min/max | 以 **现有 out 值为初值** 继续归约，不重新 fill |
| mean | 在现有 out 上完成 sum 路径后做均值化（与 torch_scatter 一致） |

#### 5. min/max 额外输出 arg_out

`scatter_min` / `scatter_max` 须额外输出 `arg_out`：

| 属性 | 约束 |
|------|------|
| dtype | `INT64` |
| shape | 与 `out` 相同 |
| 含义 | `out` 每个位置极值在 `src` 沿 `dim` 维的下标 |
| 初值 | `src.size(dim)`（表示尚未更新） |
| 平局 | 允许多个 src 同值；与 torch_scatter 一致，**后写者优先**（NPU 实现可文档化说明） |

统一入口 `scatter(..., reduce="min"/"max")` 仅返回 `out`；独立 API `scatter_min` / `scatter_max` 返回 `(out, arg_out)`。

#### 6. 支持的数据类型与实现路径分级

Ascend 950PR Vector 原子操作（`SetAtomicType`）**不支持 `double`（float64）及 `int64` 原子归约**，因此 src/out 的 dtype 按实现路径分为两级：

##### 6.1 数据类型分级总表

| 级别 | src/out dtype | 实现路径 | 支持 reduce | 性能验收 |
|------|---------------|----------|-------------|----------|
| **L1 NPU 原生（必选）** | FLOAT16、BFLOAT16、FLOAT32 | Ascend C Kernel + 硬件原子归约 | sum/add/mul/mean/min/max | **纳入 NPU 性能基线** |
| **L1 NPU 原生（必选）** | INT8、INT16、INT32、UINT8 | Ascend C Kernel + 硬件原子归约 | sum/add/mul/mean/min/max | **纳入 NPU 性能基线** |
| **L2 可选路径** | **FLOAT64** | **CPU 回退** 或 **NPU 低精度拼接** | sum/add/mul/mean/min/max | **不做性能考核** |
| **L2 可选路径** | **INT64** | **CPU 回退** 或 **NPU 低精度拼接** | sum/add/mul/mean/min/max | **不做性能考核** |

**不支持**：`BOOL`、`COMPLEX64`、`COMPLEX128` 及 Float8 等新类型。

**index 数据类型**：固定 **INT64**（与 torch_scatter 一致），不支持其他类型。

**reduce 属性**：`STRING` 或枚举，取值 `sum` / `add` / `mul` / `mean` / `min` / `max`。

##### 6.2 float64 / int64 实现路径（L2，可选）

Ascend 950PR Vector 原子操作（`SetAtomicType`）**不支持 `double`（float64）及 `int64` 原子归约**。当 `src.dtype` 为 **FLOAT64** 或 **INT64** 时，PyTorch 层须保证接口与 torch_scatter 完全兼容，可采用以下**任一**路径（开发者自选）：

| 路径 | 说明 |
|------|------|
| **CPU 回退** | 透明回退 CPU 完成计算（`src`/`index`/`out` 经 D2H/H2D）；结果与 torch_scatter CPU **bit-wise 一致** |
| **NPU 低精度拼接** | 在 NPU 上以 float32/int32 等低精度分阶段计算并 cast/拼接还原；须在 README 说明精度策略与语义，功能验收通过即可 |

| 要求项 | 说明 |
|--------|------|
| 触发条件 | `src.dtype ∈ {float64, int64}`；与 device 无关 |
| 接口行为 | 函数签名、返回值类型与 torch_scatter **完全相同** |
| 性能要求 | **不做考核**；自验证报告须注明所选 L2 路径 |
| 日志策略 | 建议首次触发 L2 路径时打印 `warning`（含 dtype 与实现路径） |

**硬件依据（Ascend 950PR）**：

- `SetAtomicType` 仅支持 `int8/int16/half/bfloat16/int32/float`，**不含 double/int64**；
- 架构白皮书 Vector Core 原生浮点格式为 FP32/FP16/BF16，**无 FP64 原生计算核**。

#### 7. 算子泛化

必须实现算子泛化，满足以下场景的合法输入：

- `src` 维度：1～8 维
- `dim` 维长度：1～65535（典型 GNN 场景 10～10⁶）
- `index` 元素个数：0～10⁸
- 同一 index 被重复命中（高冲突 scatter）
- 空张量（`src.numel()==0` 或 `index.numel()==0`）
- 非连续 Tensor（strided）

#### 8. 接口分层要求

算子采用三层架构，**必选层与可选层**如下：

```
┌─────────────────────────────────────────┐
│  PyTorch 层（必选）— Python              │  ← 验收基准，接口与 torch_scatter 完全相同
├─────────────────────────────────────────┤
│  aclnn 层（可选）— C++                   │  ← 建议实现，便于 CANN 生态集成
├─────────────────────────────────────────┤
│  Kernel 层（必选）— Ascend C             │  ← NPU 算子核心实现
└─────────────────────────────────────────┘
```

##### 8.1 PyTorch 层接口（**必选**）

须在 Python 侧提供与 torch_scatter（https://github.com/rusty1s/pytorch_scatter ） **函数名、参数列表、默认值、返回值类型完全一致** 的公开 API，可直接作为 `torch_scatter` 的 NPU 后端替换使用。

**接口定义（须与原版逐字对齐，不得增删改参数）**：

```python
from typing import Optional, Tuple
import torch

def scatter_sum(src: torch.Tensor,
                index: torch.Tensor,
                dim: int = -1,
                out: Optional[torch.Tensor] = None,
                dim_size: Optional[int] = None) -> torch.Tensor: ...

def scatter_add(src: torch.Tensor,
                index: torch.Tensor,
                dim: int = -1,
                out: Optional[torch.Tensor] = None,
                dim_size: Optional[int] = None) -> torch.Tensor: ...

def scatter_mul(src: torch.Tensor,
                index: torch.Tensor,
                dim: int = -1,
                out: Optional[torch.Tensor] = None,
                dim_size: Optional[int] = None) -> torch.Tensor: ...

def scatter_mean(src: torch.Tensor,
                 index: torch.Tensor,
                 dim: int = -1,
                 out: Optional[torch.Tensor] = None,
                 dim_size: Optional[int] = None) -> torch.Tensor: ...

def scatter_min(src: torch.Tensor,
                index: torch.Tensor,
                dim: int = -1,
                out: Optional[torch.Tensor] = None,
                dim_size: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]: ...

def scatter_max(src: torch.Tensor,
                index: torch.Tensor,
                dim: int = -1,
                out: Optional[torch.Tensor] = None,
                dim_size: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]: ...

def scatter(src: torch.Tensor,
            index: torch.Tensor,
            dim: int = -1,
            out: Optional[torch.Tensor] = None,
            dim_size: Optional[int] = None,
            reduce: str = "sum") -> torch.Tensor: ...
```

**PyTorch 层实现要求**：

| 要求项 | 说明 |
|--------|------|
| 接口一致性 | 函数签名、参数名、默认值、返回值类型与 torch_scatter 源码完全一致 |
| 行为一致性 | 输出数值、shape、dtype、device 语义与 torch_scatter CPU/GPU 标杆一致 |
| 模块导出 | 须支持 `from torch_scatter import scatter, scatter_sum, ...` 等价导入方式（或提供 drop-in 替换包） |
| NPU 张量 | L1 dtype 走 NPU Kernel；**L2 dtype（float64/int64）可选 CPU 回退或 NPU 低精度拼接**；index 广播等预处理可在 Python 层完成 |
| 自动求导 | 本任务验收范围仅覆盖**前向**；backward 可作为后续扩展 |
| 测试方式 | 须能直接复用或等价移植 torch_scatter 官方 pytest 用例 |

##### 8.2 aclnn C++ 层接口（**可选，建议实现**）

建议按 CANN 标准 aclnn 风格封装，便于其他框架或 C++ 应用直接调用。各接口须提供 `GetWorkspaceSize` + 执行入口。

**（1）统一入口 `aclnnScatter`**

```C
/**
 * @brief 获取 Scatter 算子所需 workspace 大小
 */
aclnnStatus aclnnScatterGetWorkspaceSize(
    const aclTensor *src,
    const aclTensor *index,
    int64_t dim,
    const aclTensor *out,          /* 可为 NULL，由算子内部分配 */
    int64_t dimSize,               /* dimSize < 0 表示未指定 */
    const char *reduce,            /* "sum"|"add"|"mul"|"mean"|"min"|"max" */
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/**
 * @brief Scatter 归约计算入口
 */
aclnnStatus aclnnScatter(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    const aclTensor *src,
    const aclTensor *index,
    int64_t dim,
    aclTensor *out,
    int64_t dimSize,
    const char *reduce,
    aclrtStream stream);
```

**（2）子算子独立入口**

| aclnn 接口 | 对标 PyTorch API | 返回值 |
|------------|------------------|--------|
| `aclnnScatterSum` | `scatter_sum` | `out` |
| `aclnnScatterAdd` | `scatter_add` | `out`（内部转 sum） |
| `aclnnScatterMul` | `scatter_mul` | `out` |
| `aclnnScatterMean` | `scatter_mean` | `out` |
| `aclnnScatterMin` | `scatter_min` | `out` + `argOut` |
| `aclnnScatterMax` | `scatter_max` | `out` + `argOut` |

> aclnn 层未实现不影响验收；若实现，须保证与 PyTorch 层前向结果一致。

##### 8.3 Kernel 层（**必选**）

Ascend C 实现 Scatter 核心计算逻辑，由 PyTorch 层（必选）直接调用或通过 aclnn 层（可选）间接调用。

---

### 参数说明

以下罗列算子底层原始输入参数。实际开发中，Host 侧负责 index 广播、`dim` 归一化、输出 shape 推断；Kernel 侧接收已对齐的张量。

#### 表 1：通用 Scatter 参数（sum / add / mul / mean）

| 参数名 | 输入/输出/属性 | 描述 | 使用说明 | 数据类型 | 数据格式 | 维度 Shape | 非连续 Tensor |
|--------|----------------|------|----------|----------|----------|------------|---------------|
| src | 输入 | 源数据张量 | L1 dtype 走 NPU；**float64/int64 可选 CPU 回退或低精度拼接** | L1：FLOAT16、BFLOAT16、FLOAT32、INT8、INT16、INT32、UINT8；L2：FLOAT64、INT64 | ND | 1～8 维，各维 ≥ 0 | 支持 |
| index | 输入 | 散射索引张量 | 广播后与 src 同 shape；取值须在 `[0, out.size(dim)-1]` | INT64 | ND | 广播前可低于 src 维数；广播后同 src | 支持 |
| dim | 属性 | 散射维度 | 取值范围 `[-src.dim(), src.dim()-1]`；负数按 PyTorch 规则解析 | INT64 | 标量 | - | - |
| dim_size | 属性 | 输出在 dim 维的长度 | `< 0` 表示未指定，由 `index.max()+1` 推断；`index` 为空时为 `0` | INT64 | 标量 | - | - |
| reduce | 属性 | 归约方式 | `sum`/`add`/`mul`/`mean`；非法值返回参数错误 | STRING 或 ENUM | - | - | - |
| out | 输入/输出 | 输出张量 | 可为空由内部创建；提供时按原地语义处理；dtype/device 须与 src 一致 | 同 src | ND | 除 dim 外同 src；`out.size(dim)=dim_size或推断值` | 支持 |

#### 表 2：ScatterMin / ScatterMax 额外参数

| 参数名 | 输入/输出/属性 | 描述 | 使用说明 | 数据类型 | 数据格式 | 维度 Shape | 非连续 Tensor |
|--------|----------------|------|----------|----------|----------|------------|---------------|
| arg_out | 输出 | 极值索引 | 记录 out 每个位置对应极值在 src 沿 dim 的下标；无命中位置最终不参与梯度 | INT64 | ND | 与 out 相同 | 支持 |

#### 表 3：各参数取值范围与约束汇总

| 参数 | 合法范围 / 约束 | 非法输入处理 |
|------|-----------------|--------------|
| `index[i]` | `[0, out.size(dim)-1]` | 建议 Host 侧校验并返回 `ACLNN_ERR_PARAM_INVALID`；不校验时行为未定义 |
| `index[i] < 0` | 不允许 | 返回参数错误 |
| `dim` | `[-src.dim(), src.dim()-1]` | 越界返回参数错误 |
| `dim_size` | `≥ 0`；应满足 `dim_size ≥ index.max()+1` | 不足时返回参数错误 |
| `src` / `index` / `out` device | 须在同一 NPU 设备 | 跨设备返回错误 |
| `src.dtype` | L1：见上表；L2：float64/int64 可选 CPU 回退或低精度拼接 | 其他类型返回错误 |
| `index.dtype` | 仅 INT64 | 其他类型返回错误 |
| `out.dtype` | 须与 src 相同 | 不一致返回错误 |
| `src.numel()==0` | 合法；未提供 out 时 fill 0 后返回 | - |
| `index.numel()==0` | 合法；`out.size(dim)=0`（未指定 dim_size 时） | - |
| `reduce` | 6 种枚举值 | 其他值返回错误 |

---

### 算子约束限制

1. **索引无序**：scatter 不要求 index 单调递增或唯一；须正确处理重复索引的高冲突累加场景。
2. **浮点非确定性**：`sum` 在并行原子累加场景下，浮点结果允许存在微小累加顺序差异（与 torch_scatter GPU 行为一致）；验收时 float 类型采用合理误差阈值，不要求 bit-wise 一致。
3. **mean 整数除法**：整数 dtype 的 mean 必须使用 **floor 除法**（`rounding_mode='floor'`），与 torch_scatter 对齐。
4. **mul 零值**：`src` 含 0 时前向正常；反向梯度由框架层处理，本任务**仅要求前向**。
5. **广播在 Python/Host 完成**：建议 index 广播在 PyTorch 层或 aclnn Host 层完成，Kernel 接收已 expand 的 index，降低 Kernel 复杂度。
6. **float64/int64 L2 路径**：Ascend 950PR 无 float64/int64 硬件原子归约能力；上述 dtype **可 CPU 回退或 NPU 低精度拼接**，须功能验收通过，**不做性能考核**。
7. **不支持反向**：本任务验收范围仅覆盖**前向计算**；反向梯度接口可作为后续扩展项，不在本次验收范围。

---

### 功能验收用例（须全部通过）

测试用例须覆盖 torch_scatter 官方测试语义，至少包括：

| 编号 | 场景 | 参考来源 | 覆盖 reduce |
|------|------|----------|-------------|
| TC-01 | 1D src，1D index，dim=-1 | `test_scatter.py` tests[0] | 全部 6 种 |
| TC-02 | 2D src，1D index，dim=0 | tests[1] | 全部 |
| TC-03 | 2D src，2D index，dim=1 | tests[2] | 全部 |
| TC-04 | 3D src，2D index，dim=1 | tests[3] | 全部 |
| TC-05 | 广播：index 在 dim 维坍缩 | tests[4][5] | 全部 |
| TC-06 | 1D index 广播到 4D src | `test_broadcasting.py` | 全部 |
| TC-07 | 提供 out 原地累加 | `test_scatter.py::test_out` | sum/mul/min/max |
| TC-08 | 非连续 src/index | `test_scatter.py::test_non_contiguous` | 全部 |
| TC-09 | 空张量 | `test_zero_tensors.py` | 全部 |
| TC-10 | 指定 dim_size 大于 index.max()+1 | 自行构造 | 全部 |
| TC-11 | 高冲突：大量重复 index | GNN 典型 | sum/mean |
| TC-12 | min/max arg_out 正确性 | tests[0] arg_min/arg_max | min/max |
| TC-13 | 全 dtype 遍历（L1 NPU + L2 可选路径） | `testing.py dtypes` | 全部 |
| TC-14 | dim 负数索引 | dim=-1/-2 | sum |
| TC-15 | float64 L2 路径正确性 | 自行构造 | 全部 |
| TC-16 | int64 L2 路径正确性 | 自行构造 | 全部 |

**真值与精度**：详见下文「精度要求」章节；功能用例以 CPU 版 torch_scatter 为标杆，通过 PyTorch 层接口比对。

**验收调用示例**：

```python
import torch
import torch_scatter_npu  # 或等价包名，注册 NPU 后端

src = torch.randn(10, 6, 64, device="npu")
index = torch.tensor([0, 1, 0, 1, 2, 1], device="npu")
out = torch_scatter.scatter(src, index, dim=1, reduce="sum")  # 接口与原版完全相同
```

---

### 测试标准

测试用例覆盖**常规场景、边界场景、广播场景、高冲突场景、空张量、非连续 Tensor、全数据类型、六种 reduce、min/max arg_out** 等所有功能场景。

**验收测试须通过 PyTorch 层接口执行**（pytest 直接调用 `scatter` / `scatter_sum` 等 Python API），自验证报告完整、可复现，所有测试用例执行通过。aclnn C++ 层测试为可选项，不作为验收必要条件。

---

### 性能要求

- 算子所有用例的性能需大于等于0.6倍标杆性能。
- GPU（A100）的耗时如下：

#### 测试数据

按行索引分组归约，索引无序可重复。数据：随机矩阵（N行 × C特征）+ 随机组号。

#### sum

| shape (N行 × C特征) | float32 | float16 | 总元素 |
|-------|---------|---------|--------|
| 16384 × 512 | 0.062ms | 0.077ms | 8.4M |
| 32768 × 512 | 0.124ms | 0.181ms | 16.8M |
| 65536 × 512 | 0.254ms | 0.331ms | 33.6M |
| 65536 × 1024 | 0.496ms | 0.707ms | 67M |
| 65536 × 2048 | 1.053ms | 1.455ms | 134M |

#### mean

| shape (N行 × C特征) | float32 | float16 | 总元素 |
|-------|---------|---------|--------|
| 16384 × 512 | 0.096ms | 0.112ms | 8.4M |
| 32768 × 512 | 0.169ms | 0.226ms | 16.8M |
| 65536 × 512 | 0.319ms | 0.440ms | 33.6M |
| 65536 × 1024 | 0.633ms | 0.820ms | 67M |
| 65536 × 2048 | 1.252ms | 1.643ms | 134M |

#### min

| shape (N行 × C特征) | float32 | float16 | 总元素 |
|-------|---------|---------|--------|
| 16384 × 512 | 0.281ms | 0.304ms | 8.4M |
| 32768 × 512 | 0.534ms | 0.579ms | 16.8M |
| 65536 × 512 | 0.982ms | 0.955ms | 33.6M |
| 65536 × 1024 | 1.875ms | 1.940ms | 67M |
| 65536 × 2048 | 3.755ms | 3.885ms | 134M |

#### max

| shape (N行 × C特征) | float32 | float16 | 总元素 |
|-------|---------|---------|--------|
| 16384 × 512 | 0.281ms | 0.305ms | 8.4M |
| 32768 × 512 | 0.534ms | 0.483ms | 16.8M |
| 65536 × 512 | 0.927ms | 0.955ms | 33.6M |
| 65536 × 1024 | 1.873ms | 1.940ms | 67M |
| 65536 × 2048 | 3.753ms | 3.886ms | 134M |

---

#### 实现路径与优选策略

开发过程中鼓励并行探索以下路径，**验收取最优**：

| 路径 | 适用场景 | 预期相对优势 |
|------|----------|--------------|
| 全局原子归约 | 低冲突、实现简单 | 开发周期短 |
| 分桶 + 段内归约 | 高冲突 GNN | 减少原子争用，S4 场景关键 |
| 私有化直方图 + 合并 | 中低桶数 | 片上累加，降低 GM 原子 |
| SIMD / SIMT 混合 | 不规则 index | 利用 950PR SIMT 特性 |

**性能优化建议（供设计参考）**：

- `sum`：按 index 分桶后分段归约，减少原子冲突
- `mean`：融合 sum + count，避免两次全量遍历
- `min/max`：两阶段（归约 + arg 回写）或单阶段携带 arg 的 CAS 更新
- 小 shape（<10µs）允许与 GPU 存在固定开销差，须提供仿真分析说明

---

### 精度要求

算子计算精度需严格满足《生态算子开源精度标准（https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md ）》，采用 AscendOpTest（https://gitcode.com/HIT1920/AscendOpTest ） 测试。

**真值生成方式**：以 CPU 版 torch_scatter 为标杆；NPU 结果通过 **PyTorch 层接口** 调用获取并与标杆比对。

**单标杆不满足时**：采用 ATK（https://gitcode.com/AscendTest/ATK ） 双标杆比对（`cv_fused_double_benchmark`），以更高精度的 CPU 实现为真值，同时评估同精度 CPU 与 NPU 算子实现相对于该真值的误差；满足条件为 NPU/同精度 CPU 的**最大相对误差比例 ≤ 2**、**平均相对误差比例 ≤ 1.2**、**均方根误差比例 ≤ 1.2**。

| 数据类型 / 场景 | 精度策略 |
|-----------------|----------|
| FLOAT32 / FLOAT16 / BFLOAT16（NPU L1） | 与 torch_scatter CPU 标杆比对；浮点 `sum/add/mean` 允许并行累加顺序导致的非确定性微小误差 |
| INT8 / INT16 / INT32 / UINT8（NPU L1） | 要求 **bit-wise 一致** |
| **FLOAT64 / INT64（L2）** | CPU 回退：与 torch_scatter CPU **bit-wise 一致**；低精度拼接：README 文档化精度策略，功能验收通过 |
| `mul` / `min` / `max` | 整数 bit-wise；浮点 `mul` 对标 CPU；`min/max` 的 `out` 与 `arg_out` 均须一致（平局按后写者优先） |
| 全部 reduce 模式 | L1/L2 均须覆盖 sum/add/mul/mean/min/max 功能用例 |

---

### 文档规范要求

1. 算子设计文档需根据参考模板（https://gitcode.com/cann/cann-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md ）填写，内容完整、格式规范，且必须通过评审；
2. 设计文档须重点描述：**PyTorch 层与 torch_scatter 的接口对照、float64/int64 L2 路径策略（CPU 回退 / 低精度拼接）、index 广播策略、六种 reduce 的 Kernel 分流（TilingKey）、高冲突原子累加方案、mean 计数路径、min/max arg_out 两阶段实现、多路径性能对比与最优方案选取**；
3. 自验证报告需覆盖所有功能场景，参考算子自验证报告模板（https://docs.qq.com/sheet/DUmVWWndaUE12WGFB?tab=BB08J2 ），含 **PyTorch 层 pytest 执行日志**、AscendOpTest/ATK 执行日志、整体通过截图、性能数据截图；
4. README 须包含：**PyTorch 层调用示例**（与 torch_scatter 用法一致）、**L1/L2 dtype 分级与 L2 路径说明**、与 torch_scatter 的接口对照表、支持 dtype/shape 约束说明、**性能分档达标基线**；若实现 aclnn 层，另附 C++ 调用示例。

---

## 验收交付件

在社区任务IT系统中提交验收时， 需要提交以下交付件：

| 序号 | 交付件名称 | 交付件要求 |
|------|-----------|------------|
| 1 | 算子设计文档 | 1. 设计文档模板：https://gitcode.com/cann/cann-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md ；<br> 2. 在cann-competitions 仓库（https://gitcode.com/cann/cann-competitions/tree/master/04_tasks/01_community-task-2026/tasklist ）以PR形式提交设计文档，通过评审后合入仓库，详细说明见：https://gitcode.com/cann/cann-competitions/blob/master/04_tasks/01_community-task-2026/README.md|
| 2 | 自测用例及测试代码 |1. 需要清晰列出精度测试case和性能测试case；<br> 2. 测试代码中的readme文件需要说明测试步骤，保证验收人可以复现测试结果|
| 3 | 自测报告 | 1. 自测报告模板：https://docs.qq.com/sheet/DUmVWWndaUE12WGFB?tab=BB08J2 ；<br> 2. 需要包含用例参数、精度对比结果及截图、性能数据及截图 |
| 4 | 待验收代码地址 | 1. 个人代码仓链接、分支、算子目录（需包含：**Ascend C Kernel 工程**、**PyTorch 层 Python 适配代码**、算子 README、**覆盖全规格的 PyTorch 层 pytest 测试**、AscendOpTest/ATK 测试工程及结果；aclnn C++ 层代码与测试为可选项）；需要在个人仓邀请账号Ascend-CANN作为开发者，如下图所示； <br> 2. 算子目录下需要提供readme文件，参考：https://gitcode.com/cann/ops-transformer/blob/master/attention/chunk_gated_delta_rule/README.md <br> |

![邀请示意](./pics/invite.jpeg)


### PR 申请合入

验收通过后，在昇腾算子开源仓 **ops-gnn** 提交 PR，建议合入路径：

```
https://gitcode.com/cann/ops-gnn/tree/master/scatter
```

须包含：`op_host`、`op_kernel`（Ascend C）、**PyTorch 层 Python 适配**、pytest 测试；若实现 aclnn 层，一并提交 `aclnn` 接口及示例。

---

## 参考资料

1. **对标实现**
   - torch_scatter 源码：https://gitcode.com/gh_mirrors/py/pytorch_scatter
   - 核心文件：`torch_scatter/scatter.py`、`csrc/cpu/scatter_cpu.cpp`、`csrc/cuda/scatter_cuda.cu`
2. **文档类**
   - Ascend C 算子开发文档（https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/opdevg/Ascendcopdevg/atlas_ascendc_map_10_0002.html ）
   - 算子开发接口文档（https://www.hiascend.com/document/detail/zh/canncommercial/850/API/ascendcopapi/atlasascendc_api_07_0003.html ）
   - 生态算子开源精度标准（https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md ）
3. **课程类**：Ascend C 在线课程（https://www.hiascend.com/developer/courses/detail/1691696509765107713 ）
4. **相近算子参考**
   - bincount 任务书（散射累加类）：https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/docs/202605/bincount_task_doc.md
   - EmbeddingDenseGrad（scatter 累加实现思路）

---

## 环境获取

1. 使用 hidevlab webIDE 算力：https://hidevlab.huawei.com/online-develop-intro?from=hiascend ，点击 "体验 WebIDE"；
   - 如果是新用户，在申请权限的时候需要备注使用的算力类型（A2、A3或者950）。申请内容示例：本人gitcode账号是 yolo，现在参与社区任务"7月社区任务-aclnnRoll算子开发"，需要申请A2和950算力进行任务开发。
   - 如果是老用户且需要使用950算力，需要向昇腾CANN小助手反馈账号名（个人中心->基本信息，如下图所示），后台会添加账号至950使用白名单。

   ![环境截图](./pics/zaixiankaifa1.png)  
   ![账号名](./pics/account.png)  

2. 开源仓提供100小时免费时长，请不使用时及时关闭，用时耗尽前请务必保存相关资料，建议及时提交备份。

   ![环境截图](./pics/yunkaifa.png)

3. 申请算力资源后一般在1-2个工作日完成审批，如果没有审批完成，请及时在任务对应的讨论帖留言或者联系昇腾CANN小助手。

---

## 特别注意事项

1. 开发须严格遵循 Ascend C 编程规范、Python/PyTorch 扩展开发规范及 Ascend 950PR 算子开发相关要求；
2. 所有交付件须提前完成自验证，确认符合验收标准后再提交；
3. 开发前务必阅读【社区任务】流程及注意事项（https://gitcode.com/org/cann/discussions/39 ）；
4. **验收以 PyTorch 层接口为准**：函数签名须与 torch_scatter 原版完全相同，行为须与 torch_scatter 一致；aclnn C++ 层为可选，不影响验收结论；
5. **接口对齐优先级**：PyTorch 层行为对齐 torch_scatter > 性能优化；任何与 torch_scatter 语义不一致的实现不予验收；
6. `add` 与 `sum` 必须作为等价路径处理，不得出现结果差异；
7. 高冲突 index 是 GNN 典型负载，不得仅针对低冲突场景优化而导致高冲突场景功能或性能退化；
8. **float64/int64 L2 路径**可选 CPU 回退或 NPU 低精度拼接；须功能/精度验收通过，**不做性能考核**；
9. **性能验收取最优实现**：分档指标为达标基线，开发中可探索多种 Kernel 路径，合入代码须为实测性能最优方案，并在设计文档与自验证报告中给出对比数据。