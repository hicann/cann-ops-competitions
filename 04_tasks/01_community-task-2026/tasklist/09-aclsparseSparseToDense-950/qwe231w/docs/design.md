# 需求背景（required）

## 需求来源

本需求来源于 CANN 2026 年社区任务“9 月社区任务-aclsparseSparseToDense 算子开发（950）”。任务要求在 `cann/ops-sparse` 仓库中完成 `aclsparseSparseToDense` 的 Ascend 950 实现，并同步交付 Python/ATen NPU 适配、测试代码和自测材料。

## 背景介绍

### aclsparseSparseToDense 算子功能

`aclsparseSparseToDense` 将二维 CSR、CSC 或 COO 稀疏矩阵转换为 ROW 或 COL 布局的二维稠密矩阵。每个稀疏坐标对应的 value 写入稠密输出，其余逻辑位置写入对应数据类型的正零。该接口用于将稀疏存储数据接入稠密计算链路。

本任务需支持 INT8、FP16、BF16、FP32、complex64 五种 values 数据类型，Device 侧 offsets 和 indices 为 I32，下标基准支持 0 和 1；输入坐标允许乱序，但必须唯一。清零与 scatter 在 Ascend 950 NPU 上完成，不得回退到 CPU。

### 标杆接口及现状分析

任务书指定 NVIDIA cuSPARSE SparseToDense 为功能语义标杆。该接口采用 `cusparseSparseToDense_bufferSize` 查询 workspace，再由 `cusparseSparseToDense` 执行转换。标杆能力来源为 cuSPARSE 公开接口文档：

- 接口说明：<https://docs.nvidia.com/cuda/cusparse/generic-api/conversion.html>
- 稀疏描述符与稠密描述符：同一文档的 Generic API 数据类型及转换章节。

| 能力 | cuSPARSE 13.3 Update 1 公开范围 | 本任务范围 |
| --- | --- | --- |
| 稀疏格式 | CSR、CSC、COO | CSR、CSC、COO |
| 稠密布局 | ROW、COL | ROW、COL |
| 索引类型 | I32、I64 | I32 |
| index base | 0、1 | 0、1 |
| values 类型 | `CUDA_R_8I`、`CUDA_R_16F`、`CUDA_R_16BF`、`CUDA_R_32F`、`CUDA_R_64F`、`CUDA_C_16F`、`CUDA_C_16BF`、`CUDA_C_32F`、`CUDA_C_64F` | INT8、FP16、BF16、FP32、complex64 |
| 算法 | DEFAULT | DEFAULT |
| 可观察属性 | BufferSize/Convert 两阶段、异步执行、确定性结果 | BufferSize/Convert 两阶段；清零和 scatter 异步；合法输入确定性 |

本任务属于稀疏库接口，不是 OPP 内置 TBE 算子，CANN 安装目录中不存在可对应的 TBE 实现文件或算子信息库注册文件。因此不引用 TBE 源码路径；CANN 侧接口与能力边界以 `ops-sparse/include/cann_ops_sparse.h`、公共描述符定义及任务书为准。

当前 `ops-sparse` 仓库已包含以下 Ascend 950 代码：

```text
include/cann_ops_sparse.h
sparse/sparse2dense/arch35/sparse2dense_host.cpp
sparse/sparse2dense/arch35/sparse2dense_kernel.cpp
sparse/sparse2dense/arch35/sparse2dense_kernel.h
sparse/sparse2dense/arch35/sparse2dense_tiling_data.h
sparse/sparse2dense/arch35/sparse2dense.h
```

现有实现已形成 Host、Tiling、清零和格式分发框架，但与任务书仍有差距：

1. values 类型未覆盖 complex64，且现有类型集合与任务书不完全一致。
2. 输出清零使用同步 Runtime 接口，未满足清零与 scatter 在 handle 绑定 stream 上执行的要求。
3. Device 侧 offsets、indices 合法性及重复坐标未形成任务书要求的错误返回路径。
4. SIMT 分核、线程配置和局部存储设置尚未按 DAV_3510 统一设计。
5. Python/ATen NPU 注册、端到端测试及无 CPU fallback 证据尚未交付。

### 标杆接口实现描述

cuSPARSE 为闭源库，公开资料未披露内部 Kernel 和 Tiling 源码，无法对其内部实现作源码级描述。本设计仅对齐其公开可观察语义：

1. 创建稀疏矩阵和稠密矩阵描述符。
2. 调用 BufferSize 接口查询临时空间。
3. 按查询结果分配 Device workspace。
4. 调用转换接口，在指定 stream 上生成稠密输出。
5. 通过 stream 或 Event 完成同步和计时。

### 标杆接口执行流程图

```text
创建 CSR/CSC/COO 描述符和 DnMat 描述符
                    │
                    ▼
     cusparseSparseToDense_bufferSize
                    │
                    ▼
          分配 Device workspace
                    │
                    ▼
        cusparseSparseToDense 转换
                    │
                    ▼
          stream/Event 同步或计时
```

# 需求分析（required）

## 需求描述

采用 aclsparse C++ 接口、Host 校验与分发、Ascend C Kernel 的工程模式，实现 CSR、CSC、COO 到 ROW、COL 稠密矩阵的转换。公开接口沿用 `ops-sparse` 头文件中的 ABI；Python 侧将 `Tensor.to_dense` 对应的 `aten::_to_dense` 注册到 NPU 稀疏 DispatchKey，并通过 aclsparse 接口完成转换。

设稀疏矩阵逻辑形状为 `M × N`、非零元素数量为 `nnz`、下标基准为 `base ∈ {0, 1}`，输出满足：

```text
B[i, j] = A.values[p]，当第 p 个稀疏坐标为 (i, j)
B[i, j] = +0，         当 (i, j) 未出现在稀疏输入中
```

三种格式的坐标解释如下：

- CSR：对 `p ∈ [rowOffsets[i]-base, rowOffsets[i+1]-base)`，坐标为 `(i, colIndices[p]-base)`。
- CSC：对 `p ∈ [colOffsets[j]-base, colOffsets[j+1]-base)`，坐标为 `(rowIndices[p]-base, j)`。
- COO：第 `p` 个坐标为 `(rowIndices[p]-base, colIndices[p]-base)`。

稠密输出物理偏移和物理元素数为：

```text
ROW: offset(i, j) = i * ld + j，physicalElements = M * ld，ld >= N
COL: offset(i, j) = j * ld + i，physicalElements = N * ld，ld >= M
```

## 外部组件依赖

| 组件 | 用途 | 版本或约束 |
| --- | --- | --- |
| CANN Toolkit | Ascend C 编译、ACL Runtime、算子运行 | 9.1.0 及后续配套版本 |
| Ascend 950PR | `arch35` Kernel 编译和运行 | DAV_3510 |
| PyTorch | `Tensor.to_dense` 与 `aten::_to_dense` schema | 2.7 及以上 |
| torch_npu | NPU Dispatch、当前 stream、内存生命周期管理 | 26.0.0 及以上 |
| GoogleTest | C++ Host、接口与 Kernel 测试 | 复用 `ops-sparse` 已有测试依赖 |

Python 扩展仅在启用对应构建选项时查找 PyTorch 和 torch_npu；关闭该选项时，aclsparse C++ 库及 Kernel 的构建不依赖 LibTorch。

## 内部适配模块

| 模块 | 适配内容 |
| --- | --- |
| 公共接口 | `include/cann_ops_sparse.h` 中的 SparseToDense 枚举和函数声明 |
| 公共描述符 | 复用 SpMat、DnMat、handle、数据类型、格式、order 和 index base 定义 |
| Host | 参数校验、workspace 计算、Tiling 构造、Runtime 调用和错误码转换 |
| Ascend C Kernel | Device 内容校验、COO/CSR/CSC scatter |
| 构建系统 | `arch35` 目标、Python/ATen 可选目标及测试目标 |
| 测试框架 | C++ UT、Kernel UT、Python/ATen UT、性能与内存脚本 |

## 需求拆解

1. 补齐 INT8、FP16、BF16、FP32、complex64 类型及 CSR、CSC、COO 格式组合。
2. 支持 I32 Device 索引、base 0、base 1、ROW/COL 和合法 `ld` padding。
3. 对 Host 元数据和 Device 索引内容进行校验，非法 offsets、越界坐标和重复坐标返回参数错误。
4. 使用当前 stream 完成 Device 校验、输出清零和 scatter；清零和 scatter 不使用 CPU 中间结果。
5. 保持合法输入的位模式和确定性，不修改稀疏结构及 values。
6. 完成 `Tensor.to_dense` 到 `aten::_to_dense` 的 NPU 注册，不允许 CPU fallback。
7. 完成精度、异常、stream、回归、性能、内存和 200 条任务用例验证。

## Ascend C 算子原型

公开接口以 `ops-sparse/include/cann_ops_sparse.h` 为准。当前仓库使用 `_bufferSize` 命名，设计保持现有 ABI，不新增同义符号。任务书中的 `aclsparseSparseToDenseGetBufferSize` 在测试和交付说明中对应此接口。

```cpp
typedef enum aclsparseSparseToDenseAlg_t {
    ACL_SPARSE_SPARSETODENSE_ALG_DEFAULT = 0
} aclsparseSparseToDenseAlg_t;

aclsparseStatus_t aclsparseSparseToDense_bufferSize(
    aclsparseHandle_t handle,
    aclsparseConstSpMatDescr_t matA,
    aclsparseDnMatDescr_t matB,
    aclsparseSparseToDenseAlg_t alg,
    size_t *bufferSize);

aclsparseStatus_t aclsparseSparseToDense(
    aclsparseHandle_t handle,
    aclsparseConstSpMatDescr_t matA,
    aclsparseDnMatDescr_t matB,
    aclsparseSparseToDenseAlg_t alg,
    void *buffer);
```

| 参数 | 位置 | 约束与异常行为 |
| --- | --- | --- |
| `handle` | Host | 必须为有效 handle；为空返回参数错误 |
| `matA` | Host 描述符、Device 数据 | 二维 CSR/CSC/COO；尺寸非负；索引 I32；base 为 0 或 1；非法格式、索引、shape、dtype、offsets、坐标或重复坐标返回参数错误 |
| `matB` | Host 描述符、Device 数据 | shape、valueType 与 `matA` 一致；ROW 或 COL；`ld` 满足布局下限；非法参数返回参数错误 |
| `alg` | Host | 仅支持 `ACL_SPARSE_SPARSETODENSE_ALG_DEFAULT` |
| `bufferSize` | Host | 必须非空；返回 Device workspace 字节数 |
| `buffer` | Device | 查询值非零时必须指向不少于该字节数的有效 Device 内存 |

## Ascend C 算子相关约束

| 类别 | 支持范围 |
| --- | --- |
| 稀疏格式 | CSR、CSC、COO |
| 稀疏索引类型 | I32 |
| 下标基准 | 0、1 |
| values 类型 | INT8、FP16、BF16、FP32、complex64 |
| 稠密布局 | ROW、COL |
| Host 尺寸类型 | `int64_t` |
| 算法枚举 | DEFAULT |
| 输入维数 | 二维矩阵 |

不支持 I64 Device 索引、重复坐标、稀疏与稠密数据地址重叠、原地转换、批处理稀疏矩阵及任务书之外的数据类型。输入坐标可乱序；合法唯一坐标下，输出与输入顺序无关。

# 详细设计（required）

## 算子分析

### 数学公式

稀疏到稠密转换由完整物理输出清零和 `nnz` 次离散写组成。对于唯一坐标，任意两个 scatter 任务写入不同地址，因此无需对 values 执行归约或原子加法，结果不受线程调度顺序影响。

### 支持数据类型

| values 类型 | Kernel 搬运类型 | 字节数 | 零值位模式 |
| --- | --- | --- | --- |
| INT8 | `uint8_t` | 1 | `0x00` |
| FP16 | `uint16_t` | 2 | 全零 |
| BF16 | `uint16_t` | 2 | 全零 |
| FP32 | `uint32_t` | 4 | 全零 |
| complex64 | `uint64_t` | 8 | 实部、虚部均为全零 |

Kernel 按同宽无符号整数搬运 values，不执行浮点计算或数值转换，从而保持正负零、Inf、NaN payload 以及 complex64 实部和虚部的原始位模式。

### 支持形状

`M`、`N`、`nnz` 和 `ld` 使用 `int64_t` 表示。Host 对下列计算执行 checked multiply 和 checked add：

```text
major = order == ROW ? M : N
physicalElements = checked_mul(major, ld)
outputBytes = checked_mul(physicalElements, sizeof(valueType))
coordinateKey = checked_add(checked_mul(row, N), col) + 1
```

I32 坐标要求 `M == 0 || M - 1 + base <= INT32_MAX`、`N == 0 || N - 1 + base <= INT32_MAX`；CSR/CSC 还要求 `nnz + base <= INT32_MAX`。不能安全转换为 Kernel 字段、`size_t` 或线性地址的尺寸返回参数错误。

## 算子实现

### 实现方案

#### 调用方式

C++ 调用采用 BufferSize/Convert 两阶段接口。Python 调用链为：

```text
Tensor.to_dense
    └─ aten::_to_dense NPU Dispatcher
       └─ aclsparseSparseToDense_bufferSize
          └─ 分配或复用 Device workspace
             └─ aclsparseSparseToDense
```

#### 总体流程

```text
Host 元数据校验和 workspace 计算
                  │
                  ▼
       清零 Device 状态区和哈希表
                  │
                  ▼
  校验 Kernel：offsets、坐标范围、坐标唯一性
                  │
                  ▼
   拷回状态字并同步当前 stream，检查参数错误
           │合法                 │非法
           ▼                     └──► 返回参数错误
  aclrtMemsetAsync 清零最终 DnMat
                  │
                  ▼
      CSR/CSC/COO scatter Kernel
                  │
                  ▼
 返回；最终输出仍由调用方按 stream 语义同步
```

Device 内容校验需要在返回前得到状态，执行接口会将校验状态异步拷贝到 handle 内复用的 4 字节页锁定 Host 状态区，并同步 handle 当前 stream。该同步仅用于满足任务书对非法 offsets、越界坐标和重复坐标立即返回参数错误的要求；状态字只承载参数错误码，不构成计算结果或数据中间张量。核心清零和转换仍完全在 NPU 执行，合法输入的清零与 scatter 在同一 stream 上按序提交，接口不等待最终稠密输出完成。

#### Host 侧设计

##### 参数校验

`ValidateSparseToDenseParams` 在提交 Device 任务前依次校验：

1. handle、描述符和必需 Host 指针非空，alg 为 DEFAULT。
2. SpMat 和 DnMat 均为二维，`M`、`N`、`nnz`、`ld` 非负，shape 和 valueType 一致。
3. 稀疏格式为 CSR、CSC 或 COO，Device 索引类型为 I32，base 为 0 或 1。
4. values 类型属于任务书规定的五种类型，DnMat order 为 ROW 或 COL。
5. ROW 满足 `ld >= N`，COL 满足 `ld >= M`。
6. 非空数组具有有效 Device 指针；输出物理跨度非空时 DnMat 数据指针有效。
7. 输出与 offsets、indices、values 的地址范围不重叠。
8. 坐标、offsets 端点、物理跨度、workspace 容量和字节数计算不溢出。

Device 校验 Kernel 检查：

- CSR/CSC offsets 首项为 `base`、末项为 `nnz + base`，全部值位于该闭区间且单调不减；
- 坐标减去 `base` 后位于 `[0, M) × [0, N)`；
- 所有格式的坐标唯一；
- 哈希探测未超过表容量。

状态字按位记录 offsets、坐标、重复项和内部探测异常。任一错误均转换为参数错误，不启动输出清零和 scatter。

##### workspace 计算

workspace 由对齐后的状态区和坐标哈希表组成：

```text
statusBytes = 32
hashCapacity = nnz == 0 ? 0 : next_power_of_two(2 * nnz)
hashBytes = checked_mul(hashCapacity, sizeof(uint64_t))
bufferSize = align_up(statusBytes, 32) + align_up(hashBytes, 32)
```

哈希表装载率不高于 0.5，槽值 0 表示空槽；合法坐标编码为 `row * N + col + 1`。BufferSize 接口只读取 Host 描述符元数据，不访问 Device 内容。`nnz == 0` 时仍保留状态区，用于校验 CSR/CSC offsets；空 COO 且无 Device 内容需要校验时允许返回 0。

任务书三组性能场景的 workspace 查询值为：

| 场景 | `nnz` | `hashCapacity` | workspace |
| --- | ---: | ---: | ---: |
| P-01 | 524288 | 1048576 | 8388640 B |
| P-02 | 262144 | 524288 | 4194336 B |
| P-03 | 458752 | 1048576 | 8388640 B |

若 workspace 计算溢出或查询值不能由目标 Runtime 表示，返回参数错误。执行接口在查询值非零而 `buffer == nullptr` 时返回参数错误。

##### 分核策略

Device 校验与 scatter 分别计算任务单元：

- COO：每个任务单元对应一个非零元素；
- CSR：每个任务单元对应一行；
- CSC：每个任务单元对应一列。

校验任务量与 scatter 任务量分开计算。CSR/CSC 校验任务量分别为 `max(M, 1)` 和 `max(N, 1)`，确保零行或零列时仍检查唯一的 offsets 端点；scatter 任务量仍分别为 `M` 和 `N`。COO 两阶段任务量均为 `nnz`。

Host 从平台信息获取可用 AIV 核数。校验 Kernel 采用多核标量/SIMD 路径，每个 AIV core 按 `blockIdx` 跨步处理任务，`validateBlockDim = min(aivCoreNum, taskUnits)`。scatter Kernel 采用 SIMT 路径，block 数按下式计算：

```text
requiredBlocks = ceil_div(taskUnits, THREADS_PER_BLOCK)
blockDim = min(aivCoreNum, requiredBlocks)
```

每个 block 获得连续且均衡的任务区间：

```text
baseCount = taskUnits / blockDim
remainder = taskUnits % blockDim
blockBegin = blockIdx * baseCount + min(blockIdx, remainder)
blockTaskCount = baseCount + (blockIdx < remainder ? 1 : 0)
blockEnd = blockBegin + blockTaskCount
```

任务量为 0 时不启动对应 Kernel。scatter 线程数采用 DAV_3510 SIMT 路径的固定编译期常量 `THREADS_PER_BLOCK = 1024`，Kernel 的 `LAUNCH_BOUND` 与 Host 启动维度使用同一常量，不将线程数放入 TilingData 动态传递。校验 Kernel 不进入纯 SIMT 路径，避免使用纯 SIMT 不支持的原子比较交换接口。

##### 数据分块和内存优化策略

scatter 直接读取 GM 中的 offsets、indices 和 values，并写入最终 DnMat，不构造 COO 副本、排序副本或完整稠密临时张量。CSR/CSC 以压缩主维度分工，避免每个非零元素对 offsets 做二分查找；COO 按 `nnz` 连续划分，使 indices 和 values 读取保持连续。

Kernel 不申请大块 Unified Buffer。编译配置通过 `SetLocalMemorySize` 为 SIMT 路径保留不少于 32 KB 的本地存储/DCache 预算；最终数值以目标 CANN 9.1.0 DAV_3510 编译器接口为准。TilingData 为固定大小，只保存地址计算和分工所需字段。

```cpp
struct SparseToDenseTilingData {
    int64_t rows;
    int64_t cols;
    int64_t nnz;
    int64_t ld;
    int64_t taskUnits;
    int32_t indexBase;
    int32_t reserved;
};
```

##### tilingKey 规划策略

TilingKey 由稀疏格式、数据宽度、输出布局和 Kernel 阶段组成：

| 维度 | 取值 |
| --- | --- |
| 阶段 | VALIDATE、SCATTER |
| 稀疏格式 | CSR、CSC、COO |
| 数据宽度 | 8 bit、16 bit、32 bit、64 bit |
| 输出布局 | ROW、COL |

VALIDATE 阶段不依赖 values 类型，数据宽度位固定为公共值；SCATTER 阶段按数据宽度实例化。FP16 与 BF16 共用 16 bit 搬运模板，FP32 使用 32 bit 模板，complex64 使用 64 bit 模板，INT8 使用 8 bit 模板。ROW/COL 为编译期参数，元素循环内不保留布局分支。

##### 清零与 Runtime 调度

Host 在校验通过后使用 `aclrtMemsetAsync`，在 handle 绑定 stream 上将 DnMat 完整物理跨度写为 `0x00`，包括 `ld` padding。全零位模式对应五种数据类型的正零。`outputBytes == 0` 时跳过清零；`nnz == 0` 时清零后直接返回，不启动 scatter Kernel。

校验、状态拷贝、清零和 scatter 均使用同一 stream。校验同步结束后，清零与 scatter 保持先后顺序。输入 Device 数组、输出和 workspace 在 stream 完成前必须保持有效。

#### Kernel 侧设计

##### Device 内容校验

校验 Kernel 首先验证格式结构，再生成全局唯一坐标键。CSR/CSC 的每个任务处理一个合法区间；若区间端点越界或逆序，则设置状态位并跳过该区间，避免非法地址访问。COO 直接处理每个坐标。

每个合法范围内的坐标使用 64 位线性键插入开放寻址哈希表。校验 Kernel 采用多核标量/SIMD 路径，通过 `AscendC::AtomicCas<uint64_t>` 向 GM 哈希表插槽执行原子比较交换；发现相同键时设置重复坐标状态，探测完整表仍未插入时设置内部异常状态。状态区通过 `AscendC::AtomicCas<uint32_t>` 保留第一个错误码，保证多核并发时错误不会被覆盖。纯 SIMT 原子接口不用于该阶段。

##### COO scatter

```text
for p in thread_assigned_nonzeros:
    row = rowIndices[p] - base
    col = colIndices[p] - base
    outOffset = ROW ? row * ld + col : col * ld + row
    dense[outOffset] = bit_copy(values[p])
```

##### CSR scatter

```text
for row in thread_assigned_rows:
    begin = rowOffsets[row] - base
    end = rowOffsets[row + 1] - base
    for p in [begin, end):
        col = colIndices[p] - base
        outOffset = ROW ? row * ld + col : col * ld + row
        dense[outOffset] = bit_copy(values[p])
```

##### CSC scatter

```text
for col in thread_assigned_columns:
    begin = colOffsets[col] - base
    end = colOffsets[col + 1] - base
    for p in [begin, end):
        row = rowIndices[p] - base
        outOffset = ROW ? row * ld + col : col * ld + row
        dense[outOffset] = bit_copy(values[p])
```

Device 校验通过后，scatter 使用同一份索引。合法输入坐标唯一，不同线程写入不同输出地址，因此无需对 values 使用原子操作，结果具有确定性。

##### Ascend C 实现流程图

```text
                    ┌─ CSR：按行读取 offsets 和 colIndices ─┐
TilingData/GM 输入 ─┼─ CSC：按列读取 offsets 和 rowIndices ─┼─► 计算 row、col
                    └─ COO：读取 rowIndices 和 colIndices ──┘
                                                              │
                                                              ▼
                                               ROW/COL 物理偏移计算
                                                              │
                                                              ▼
                                               同宽位模式写入最终 DnMat
```

##### 与标杆流程的差异

| 项目 | cuSPARSE 公开流程 | Ascend C 方案 | 原因 |
| --- | --- | --- | --- |
| Device 内容校验 | 公开文档未披露内部策略 | 使用状态区和坐标哈希表检查 offsets、范围和重复坐标 | 满足任务书的参数错误要求，并防止非法 GM 地址 |
| 输出初始化 | 内部实现不可见 | `aclrtMemsetAsync` 清零最终物理跨度 | 全零位模式可统一表示目标类型正零 |
| 格式处理 | 内部实现不可见 | CSR 按行、CSC 按列、COO 按非零元素分工 | 复用压缩格式结构，避免格式转换和 offsets 二分查找 |
| values 处理 | 公开语义为稀疏值写入稠密输出 | 按 8/16/32/64 bit 位模式搬运 | 保持特殊浮点值和 complex64 位模式 |
| workspace | 由标杆 BufferSize 查询 | 状态区加 O(nnz) 唯一性哈希表 | 支持所有格式的乱序坐标唯一性检查 |

#### Python/ATen 适配

PyTorch 2.7 的目标 schema 为：

```text
aten::_to_dense(Tensor self, ScalarType? dtype=None, bool? masked_grad=None) -> Tensor
```

NPU 适配注册到 `SparsePrivateUse1` 和 `SparseCsrPrivateUse1`：前者处理二维 COO，后者处理二维 CSR 和 CSC。适配代码置于 `python/sparse2dense/`，通过顶层可选构建开关接入；注册、bridge 和测试统一交付至 `ops-sparse` 仓库。

适配步骤如下：

1. 校验输入位于 NPU，维数为 2，layout 为 `sparse_coo`、`sparse_csr` 或 `sparse_csc`。
2. 校验 values 属于五种支持类型，所有索引位于同一 NPU 且为 I32；I64 索引不做静默窄化。
3. `dtype` 为空或等于输入类型时按位搬运；指定另一种受支持类型时，在 NPU 上生成 O(nnz) 的 values 转换张量。
4. 接受 `masked_grad` 的空值、`true` 和 `false`，该参数不改变前向结果。
5. 在当前 NPU 上创建 ROW 连续输出，构造 aclsparse 稀疏和稠密描述符，base 固定为 0。
6. 将 torch_npu 当前 stream 绑定到复用的 aclsparse handle，查询并分配 workspace，调用转换接口。
7. 记录输入、输出、workspace 和可选 values 转换张量的 stream 生命周期，销毁 Host 描述符并返回输出。

本任务交付范围为 `_to_dense` 前向。`requires_grad=false` 的前向测试覆盖 `masked_grad` 三种取值；反向若无对应 NPU Kernel，则返回明确的不支持错误，不允许转到 CPU。适配层不得调用 `.cpu()`、CPU Kernel、Host Golden 或 Host 数据转换。Profiler 中 `_to_dense` 下方只允许出现必要的 Runtime 操作和 NPU Kernel。

同类型输出要求稀疏 values 与对应稠密元素 bit-wise 一致。显式指定不同 `dtype` 时，按 PyTorch NPU cast 语义与 CPU Golden 的目标 dtype 做 exact 或该 dtype 精度标准规定的比较，不要求与转换前源 values 位模式一致。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Ascend 950PR（DAV_3510，A5，`arch35`） | √ |

## 算子约束限制

1. 仅支持二维 CSR、CSC、COO，不支持批处理稀疏矩阵。
2. Device 索引仅支持 I32，Host 尺寸元数据使用 `int64_t`。
3. values 仅支持 INT8、FP16、BF16、FP32、complex64，输入与 C++ 输出 valueType 必须一致。
4. 输入坐标允许乱序但必须唯一；重复坐标、非法 offsets 和越界坐标返回参数错误。
5. 不支持输入输出别名或原地转换。
6. C++ 接口支持 ROW/COL 及合法 `ld`；Python 标准入口返回 ROW 连续张量。
7. 为满足 Device 内容错误的同步返回要求，执行接口包含一次校验状态同步；最终清零和 scatter 保持 stream 异步语义。
8. Python 适配范围为前向；未交付的反向能力明确报错，不得 CPU fallback。

# 特性交叉分析

| 交叉特性 | 分析结论 |
| --- | --- |
| 动态 shape | Host 每次从描述符读取 `M/N/nnz/ld` 并重新计算 workspace 和 Tiling，不缓存固定 shape |
| 多 stream | 每个调用只使用 handle 绑定 stream；同一 workspace 不得被未完成的并发调用复用 |
| 确定性 | 唯一坐标保证 scatter 地址互异；重复执行 bit-wise 一致 |
| 特殊值 | values 按位搬运，不改变 NaN payload、Inf、正负零或 complex64 分量 |
| 空矩阵 | 按格式校验结构；有物理输出时只清零，不启动 scatter |
| `ld` padding | 清零覆盖完整物理跨度，scatter 仅写逻辑坐标，guard 区验证边界外不写 |
| 输入只读 | Kernel 不写 offsets、indices、values；测试前后逐字节比较 |
| A2/A3 回归 | A5 使用独立 `arch35` 构建和 TilingKey，不改变现有架构目录行为 |
| Python Dispatch | 仅注册 NPU 稀疏 DispatchKey；Profiler 检查无 CPU fallback |
| 并发与生命周期 | 输入、输出、workspace 及转换临时张量在 stream 完成前保持有效 |

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | 同类型输出逐元素 exact match，values 对应位置 bit-wise 一致；complex64 实部、虚部分别 bit-wise 一致；未覆盖位置为正零 | 任务书、CANN 算子精度标准 |
| 确定性标准 | 乱序唯一坐标和重复执行结果 bit-wise 一致 | 任务书 |
| 性能标准 | P-01、P-02、P-03 的所有声明 dtype 性能倍率均不低于 0.3 | 任务书 |
| 内存标准 | 输入输出总量超过 500 MB 时，NPU 相对 GPU 的额外内存增幅低于 50%；或固有 workspace 不超过目标硬件 L2 Cache 容量 | 任务书 |

性能倍率定义为：

```text
performanceRatio = gpuMedianUs / npuMedianUs
```

任务书给出的 GPU 设备 Event 基线为：

| 编号 | 场景 | GPU median_us 范围 | 有效场景数 | 目标 |
| --- | --- | --- | --- | --- |
| P-01 | `8192 × 28672`，每行 64 个非零元素 | 144.912–728.960 μs | 30 | 各声明 dtype 不低于 0.3 倍 |
| P-02 | `4096 × 1536`，每行 64 个非零元素 | 66.544–307.024 μs | 30 | 各声明 dtype 不低于 0.3 倍 |
| P-03 | `7168 × 2048`，每行 64 个非零元素 | 74.656–378.528 μs | 30 | 各声明 dtype 不低于 0.3 倍 |

每个 case 预热不少于 10 次，正式采样不少于 30 次。NPU 和 GPU 使用设备 Event 统计等价调用范围，复用描述符和 workspace，排除首次编译、初始化和 Host/Device 数据搬运。报告记录原始采样、median、p90、Device 校验、清零、scatter、workspace 峰值、有效带宽及 Python/ATen 端到端耗时。

## 测试设计

### 精度与功能测试

1. 覆盖 CSR、CSC、COO，ROW、COL，base 0、1 和五种 values 类型。
2. 覆盖零 `nnz`、空行/列、单元素、极稀疏、高密度、长尾、最大 shape 和 `ld` padding。
3. 普通 values 中 70% 使用 `[-1, 1]` 均匀分布，20% 使用 `μ=0、σ=1` 正态分布，10% 覆盖正负零、边界值、Inf 和 NaN；complex64 实部和虚部独立生成。
4. CPU Golden 使用 64 位地址计算，按格式和 base 写入目标 layout；同类型结果进行逐元素和逐位检查。
5. 输入坐标乱序后重复执行，验证结果确定性。
6. 对 offsets、indices 和 values 做调用前后逐字节比较，验证输入只读。
7. 在输出物理范围前后设置 guard zone，验证 `ld` padding 和边界外不写。
8. 执行任务包提供的 200 条精度用例并记录通过率。

### 参数与异常测试

覆盖空 handle、空描述符、空 `bufferSize`、非法 alg、格式、base、索引类型、values 类型、order、shape、dtype、`ld`、负尺寸、乘加溢出、空 Device 指针、输入输出重叠，以及 workspace 查询值非零时的空 buffer。Device 专项用例覆盖 offsets 首尾错误、非单调、越界坐标、重复坐标和哈希异常，验证返回参数错误且不修改输出。

### Python/ATen 测试

1. 验证 `Tensor.to_dense` 命中 `aten::_to_dense` 的 NPU 稀疏 Dispatch。
2. 覆盖 COO、CSR、CSC、五种 values 类型、`dtype` 为空/相同/受支持转换及 `masked_grad` 三种取值。
3. 覆盖 shape、layout、stride、device、异常、输出、alias/in-place 不适用行为和输入只读。
4. 覆盖默认 stream、非默认 stream、前序异步写入和后续异步消费。
5. 使用 Dispatcher 记录和 NPU Profiler 证明不存在 CPU Kernel、D2H 数据结果搬运或 CPU fallback；4 字节参数状态回传单独标识，不计为结果中间张量。

### 性能与内存测试

性能用例以 `test_cases/aclsparseSparseToDense_testCase/performance_cases.json` 为准，覆盖 P-01、P-02、P-03 及泛化场景。分别记录 aclsparse Kernel 范围和 Python/ATen 端到端范围；Device 校验同步计入 NPU 同调用范围总耗时。

内存测试使用任务书指定脚本：

```text
test_cases/aclsparseSparseToDense_testCase/collect_sparse_ops_gpu_memory.py
test_cases/aclsparseSparseToDense_testCase/collect_sparse_ops_npu_memory.py
test_cases/aclsparseSparseToDense_testCase/compare_sparse_ops_memory.py
```

按相同 case id 记录 `input_baseline_*_bytes`、`peak_*_bytes` 和 `extra_peak_*_bytes`。workspace 记录查询值和实际峰值；若采用 L2 容量准则，以目标环境平台信息记录的 L2 容量为上限。禁止创建与完整稠密输出成比例的 Host 或 Device 临时副本。

### 自测报告与回归

自测报告记录硬件、驱动、固件、CANN、编译器、PyTorch、torch_npu 和 `ops-sparse` 提交版本，包含用例参数、输出结构、精度结果、性能数据、峰值内存、截图、Profiler 证据及失败项说明。执行 A5 全量测试和 A2/A3 交叉回归，并执行仓库格式检查、静态检查、构建和相关稀疏算子回归。

## 兼容性分析

1. 保持 `include/cann_ops_sparse.h` 已有 `_bufferSize` 符号和 DEFAULT 枚举值，不改变既有 ABI。
2. complex64 作为 A5 新增分发类型接入，不将任务书未声明的 INT32 纳入 A5 能力清单。
3. `arch35` 代码、TilingKey 和构建条件独立，不改变 A2/A3 Kernel。
4. 公共描述符、状态码和 handle/stream 机制沿用仓库定义，不创建同义公共类型。
5. Python/ATen 目标为可选构建模块，不影响未启用 PyTorch 依赖时的 C++ 库构建。

## 代码目录与交付物

计划修改或新增的目录如下：

```text
include/cann_ops_sparse.h
sparse/sparse2dense/arch35/
├── sparse2dense_host.cpp
├── sparse2dense_kernel.cpp
├── sparse2dense_kernel.h
├── sparse2dense_tiling_data.h
└── sparse2dense.h

python/sparse2dense/
├── CMakeLists.txt
└── csrc/
    ├── sparse2dense_aten.cpp
    └── sparse2dense_bridge.cpp

test/sparse2dense/
├── CMakeLists.txt
├── sparse2dense_golden.h
├── sparse2dense_param.h
├── arch35/
│   ├── sparse2dense_npu_execution.h
│   ├── sparse2dense_npu_wrapper.h
│   ├── sparse2dense_test.cpp
│   ├── sparse2dense_test.csv
│   └── sparse2dense_l2_cases.csv
└── sparse2dense_aten_test.py
```

交付物包括：

1. 本设计文档。
2. `ops-sparse` 公开接口、Host、Tiling、Ascend C Kernel 和构建配置。
3. Python/torch 入口、ATen NPU 注册和 aclsparse bridge。
4. C++ UT、Kernel UT、ATen UT 和 Python 端到端 UT。
5. 200 条精度用例结果、性能报告、内存报告及无 CPU fallback 证据。
6. 自测 Excel、复现 README、待验收个人仓库分支和固定提交哈希。

## 参考资料

1. 社区任务流程及注意事项：<https://gitcode.com/org/cann/discussions/39>
2. 社区任务设计文档 Checklist：<https://docs.qq.com/sheet/DUHVGUFdmSFRjVFFU?tab=000001>
3. 社区任务设计模板：<https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md>
4. CANN 社区任务仓：<https://gitcode.com/cann/cann-ops-competitions>
5. ops-sparse：<https://gitcode.com/cann/ops-sparse>
6. CANN 算子精度标准：<https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md>
7. PyTorch `Tensor.to_dense`：<https://pytorch.org/docs/stable/generated/torch.Tensor.to_dense.html>
8. PyTorch 2.7 `aten::_to_dense` schema：<https://github.com/pytorch/pytorch/blob/v2.7.0/aten/src/ATen/native/native_functions.yaml>
9. NVIDIA cuSPARSE SparseToDense：<https://docs.nvidia.com/cuda/cusparse/generic-api/conversion.html>
10. Ascend C `AtomicCas`：<https://www.hiascend.com/document/detail/zh/canncommercial/900/API/ascendcopapi/atlasascendc_api_07_00262.html>
