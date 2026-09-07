# aclsparseGather 算子设计文档

**版本**: v1.0  
**日期**: 2026-09-05  
**适配硬件**: Atlas A2 训练系列（910B3/910B4） / Atlas A3 系列  
**开发语言**: C++ (Host), Ascend C (Kernel), Python/ATen (适配层)  

---

## 1. 需求背景

### 1.1 需求来源
基于 cuSPARSE `cusparseGather` 接口语义，面向 Atlas A2/A3（DAV_2201，`arch22`）稀疏向量访问场景，开发并完善 `aclsparseGather` 接口。

### 1.2 背景介绍
在 LLM（大语言模型）的词表 Embedding 查找、推荐系统稀疏特征聚合等场景中，存在大量的“按照索引从稠密向量中抽取元素，并写入稀疏向量”的操作。原 TBE/A5 架构下的实现无法直接适配 A2/A3 的底层指令，且现有 C++ 接口缺少 PyTorch 官方入口 (`torch.index_select`) 的无缝接入，导致在 NPU 上只能 CPU fallback，性能低下。本设计旨在通过 Ascend C 重写 Kernel，并打通 Python/ATen 的 Dispatcher 适配，实现全流程 NPU 计算。

### 1.3 现有实现现状
当前 ops-sparse 仓库的 A5 (arch35) 实现可作为逻辑参考，但需针对 A2/A3 (arch22) 的指令特性（如 DMA 路径、向量单元）进行重写。同时，**ATen 适配层完全缺失**，是本任务的核心新增交付件。

---

## 2. 需求分析

### 2.1 功能需求
实现公式：
\[
X.values[i] = Y[X.indices[i] - idxBase], \quad i \in [0, nnz)
\]
- 只更新 `vecX.values`，不修改 `vecY` 和 `vecX.indices`。
- 支持 `float16`、`bfloat16`、`float32`、`complex64` 四种 dtype。
- 支持 I32 索引，Index Base 0/1。
- 支持动态 size/nnz，乱序/重复索引，`nnz=0/1`，尾块。

### 2.2 Python/ATen 接口适配需求
必须交付 Python/torch 层、ATen Dispatcher 的 NPU 注册及调用链。
- **公开入口**：`torch.index_select(input, 0, index)`
- **内部映射**：`aten::index_select`
- **约束**：校验 `dim=0`，必须显式报错（不 CPU fallback），alias/in-place 与 stream 异步语义须与 PyTorch 对齐。

### 2.3 软硬件要求
- 硬件：Atlas A2（910B3、910B4），A3 型号。
- CANN：9.1.0 以上。
- PyTorch 2.7+，torch_npu 26.0.0+。

---

## 3. 详细设计

### 3.1 整体架构与分层
```text
[Python层] torch.index_select
    ↓
[ATen Dispatcher] at::index_select (NPU后端注册)
    ↓
[Native适配层 (C++)] npu_index_select()
    → 校验 dim/type/device
    → 转换 tensor 为 DnVec/SpVec Descriptor
    → 调用 aclsparseGather
    ↓
[Host层 (C++)] aclsparseGather()
    → 解析 Handle/Descriptor
    → 计算 Tiling
    → 下发 Kernel
    ↓
[Ascend C Kernel] arch22/equal_kernel.cpp
    → 按 nnz 切块
    → 索引映射、取数、写回
```

### 3.2 Host 侧详细设计

#### 3.2.1 数据结构定义 (Tiling Data)
在 `sparse/gather/arch22/` 下新增 `gather_tiling.h`，定义用于传参给 Kernel 的 Tiling 结构体：
```cpp
struct GatherTilingData {
    int32_t nnz;            // 稀疏元素总个数
    int32_t idxBase;        // 0 或 1
    int32_t vecYSize;       // 稠密向量长度 size
    int32_t vecXIndicesStride; // 稀疏索引的 stride（正常情况下为 1）
    int32_t vecXValuesStride;  // 稀疏值的 stride（正常情况下为 1）
    int32_t blockNum;       // 使用的 AI Core 核数
    int32_t tailSize;       // 最后一个 block 处理的元素数（处理尾块）
    int32_t dataType;       // 0: fp16, 1: bf16, 2: fp32, 3: complex64
};
```

#### 3.2.2 参数校验流程 (Host)
在 `aclsparseGather` 进入 Kernel 前，必须进行严密校验：
1. **句柄校验**：`handle` 非空，device 匹配。
2. **描述符校验**：`vecY` 和 `vecX` 非空。
3. **Dtype 校验**：`vecY` 与 `vecX.values` 的 dtype 必须一致，且必须在 `{fp16, bf16, fp32, complex64}` 范围内。
4. **索引类型校验**：`vecX.indices` 必须为 `INT32`。
5. **Base 校验**：`vecX.index_base` 必须为 `0` 或 `1`。
6. **Shape 校验**：
   - `nnz > 0` 时，`size > 0`；`nnz == 0` 直接返回成功，不启动 Kernel。
   - 对于所有索引 `indices[i]`，必须满足 `0 <= indices[i] - base < size`，否则返回参数错误或异步错误协议。
7. **内存重叠校验**：`vecY` 和 `vecX.values` 物理内存不可重叠（未声明的重叠直接拒绝）。

#### 3.2.3 Tiling 计算逻辑
- 动态读取 `size` 和 `nnz`。
- 计算 AI Core 核数（由 handle 和硬件决定，通常 20 或 40 核）。
- 将 `nnz` 均匀切分到各核（`blockNum`），最后一个核处理剩余的 `tailSize`。
- 将 Tiling 数据写入 `tiling_buf`。

### 3.3 Ascend C Kernel 详细设计 (arch22)

#### 3.3.1 Kernel 代码逻辑（核心伪代码）
在 `sparse/gather/arch22/kernel/` 下新建 `gather_kernel.cpp`：
```cpp
#include "kernel_operator.h"
#include "gather_tiling.h"

template <typename T>
__global__ __aicore__ void gather_kernel(GM_ADDR vecY, GM_ADDR indices, GM_ADDR values, GM_ADDR tiling)
{
    GatherTilingData* tiling_data = (GatherTilingData*)tiling;
    int32_t nnz = tiling_data->nnz;
    int32_t idxBase = tiling_data->idxBase;
    int32_t vecYSize = tiling_data->vecYSize;
    int32_t blockNum = tiling_data->blockNum;
    int32_t idx = GetBlockIdx(); // 获取当前核 ID

    // 计算当前核负责的 start 和 count
    int32_t per_block = (nnz + blockNum - 1) / blockNum;
    int32_t start = idx * per_block;
    int32_t count = (start + per_block > nnz) ? (nnz - start) : per_block;
    if (count <= 0) return;

    // 数据加载和计算
    LocalTensor<T> vecYLocal, idxLocal, valLocal;
    // 搬运当前块的 indices 到 UB
    DataCopy(idxLocal, indices + start, count);
    // 对 indices 减去 base，越界防御处理（虽然 Host 已校验，但 Kernel 加保护）
    for (int i = 0; i < count; i++) {
        int32_t real_idx = idxLocal[i] - idxBase;
        // 将 vecY[real_idx] 搬运过来
        DataCopy(valLocal[i], vecY + real_idx, 1);
    }
    // 将结果搬运回 GM
    DataCopy(values + start, valLocal, count);
}
```

#### 3.3.2 Complex64 特殊处理
由于 complex64 在底层由 2 个 float32（实部和虚部）组成，我们在 Kernel 中将其视为 `float2` 处理：
- 读取 vecY 时，`real = vecY[real_idx * 2]`，`imag = vecY[real_idx * 2 + 1]`。
- 写入 values 时，同时写入这两个 float。
- 保证 CPU Golden 校验时，实部和虚部都 bit-wise exact match。

### 3.4 Python/ATen 适配层详细设计

#### 3.4.1 Dispatcher 注册
在 ops-sparse 的 C++ 适配代码中（`adapter/index_select_npu.cpp`）：
```cpp
#include <ATen/ATen.h>
#include <torch/library.h>
#include "aclsparse_common.h"

at::Tensor index_select_npu(const at::Tensor &self, int64_t dim, const at::Tensor &index) {
    // 1. 强制校验，禁止 CPU fallback
    TORCH_CHECK(self.is_npu(), "Only NPU tensors are supported!");
    TORCH_CHECK(dim == 0, "aclsparseGather only supports dim=0!");
    TORCH_CHECK(self.scalar_type() != at::kDouble, "Only float16/32, bf16, complex64 supported!");

    // 2. 构造 SpVec/DnVec 描述符（调用 ops-sparse 的现有 C API）
    aclsparseHandle_t handle;
    aclsparseCreate(&handle);
    aclsparseCreateDnVecDescr(&vecY_desc, self);
    aclsparseCreateSpVecDescr(&vecX_desc, index, out, idxBase);

    // 3. 调用底层接口
    aclsparseGather(handle, vecY_desc, vecX_desc);

    // 4. 异步流同步/返回输出（根据 PyTorch 语义）
    return out;
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("index_select", index_select_npu);
}
```

#### 3.4.2 语义对齐与限制
- **异常处理**：所有不支持组合（如 dim=1，非连续内存）必须显式 `TORCH_CHECK(false)` 抛错。
- **Stream 语义**：在 NPU 上调用必须保证异步执行，并注意在当前流上同步。
- **别名处理**：如果是 Inplace 操作需注意别名关系。

### 3.5 内存管理设计
- 算子本身**不新增任何 workspace**。
- 不进行任何与输入规模相关的临时内存分配（复杂情况复用描述符内部 buffer）。
- 如果单次输入 >500MB，确保不额外申请超过 GPU 内存 50% 的内存。

---

## 4. 支持硬件
| 支持的芯片版本 | 是否支持 |
| :--- | :--- |
| Atlas A2 训练系列 (910B3/910B4) | 支持 |
| Atlas A3 系列 | 支持 |

---

## 5. 算子约束限制
1. vecX 与 vecY 的 values 必须为 `{float16, bfloat16, float32, complex64}` 且保持一致。
2. vecX.indices 必须为 INT32。
3. vecX.index_base 必须为 0 或 1。
4. 转换 base 后，所有索引必须在 [0, size) 区间内。
5. 必须为连续一维向量，不支持非连续 memory layout (除非通过 Tiling 适配 stride)。
6. 禁止 CPU fallback。

---

## 6. 可维可测分析

### 6.1 精度标准
- CPU Golden 计算采用高精度基准：
  - `float16`/`bfloat16` 输入，Golden 用 float32。
  - `float32` 输入，Golden 用 float64。
  - `complex64` 输入，Golden 用 complex128。
- 要求所有输出逐元素 bit-wise exact match，且必须校验 `vecY` 和 indices 未被修改。

### 6.2 性能标准
- 性能倍率（NPU） = GPU 标杆耗时（Event）/ NPU 同范围耗时。
- 要求：P-01 / P-02 / P-03 所有 case 倍率 ≥ 0.25 倍。
- 预热 10 次，采样 30 次，报告中位数与 p90。

### 6.3 测试用例设计矩阵
| 类别 | 必测场景 | 验证点 |
| :--- | :--- | :--- |
| 基础功能 | 不同 size/nnz，乱序/重复索引 | 输出正确性 |
| 边界 | base=0, base=1, nnz=0, nnz=1 | 边界不越界、直接返回 |
| 异常 | 非法 handle、dtype 不匹配、非法索引 | 显式报错，无 CPU fallback |
| 一致性 | 四种 dtype，重复执行 | bit-wise 一致，complex64 专项 |
| 资源 | 连续创建/销毁描述符 | 无内存泄漏，无非法同步 |
| Python/ATen | torch.index_select 端到端 | 无 CPU fallback，参数校验正确 |

---

## 7. 兼容性分析
- **前向兼容**：新增 arch22 代码，不改动 A5 (arch35) 现有代码，避免冲突。
- **后向兼容**：公开接口 `aclsparseGather` 保持不变，仅新增底层实现。
- **框架兼容**：适配 PyTorch 2.7 的 `aten::index_select` Dispatcher，确保动态图/静态图均能正确捕获。

---

## 8. 交付件清单
1. **算子设计文档**（本文件）。
2. **待验收代码地址**：`ops-sparse` 个人 fork 仓库（已邀请 Ascend-CANN 为开发者）。
3. **自测用例及测试代码**：包含 C++ UT/ST，ATen UT 和 Python 端到端 UT。
4. **自测报告**：包含性能截图、内存截图、精度对比结果及失败项说明。

---

