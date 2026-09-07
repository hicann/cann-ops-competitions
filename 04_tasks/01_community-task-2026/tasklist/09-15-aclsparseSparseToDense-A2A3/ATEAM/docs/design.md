# 【CANN社区任务】aclsparseSparseToDense（A2/A3）算子设计文档

| 项目 | 内容 |
| --- | --- |
| 任务编号 | `09-15-aclsparseSparseToDense-A2A3` |
| 任务名称 | aclsparseSparseToDense 算子开发（A2/A3） |
| 团队目录 | `ATEAM` |
| 目标代码仓 | `https://gitcode.com/cann/ops-sparse` |
| 目标代码目录 | `sparse/sparse2dense/arch22/` |
| 目标硬件 | Atlas A2/A3，`ascend910b*` / `ascend910_93*`，arch22 / DAV-2201 |
| 对标接口 | NVIDIA cuSPARSE `cusparseSparseToDense_bufferSize` / `cusparseSparseToDense` |

# 需求背景（required）

## 需求来源

本设计对应 CANN 2026 年社区任务
`09-15-aclsparseSparseToDense-A2A3`。目标是在 `ops-sparse` 已有
SparseToDense Generic C++ API 和 arch35 实现基础上，补充 arch22（DAV-2201）
实现，使 Atlas A2/A3 支持将 CSR、CSC 或 COO 稀疏矩阵转换为 ROW 或 COL
布局的稠密矩阵。

算子执行纯数据搬运：稀疏矩阵的每个非零值写入稠密矩阵对应逻辑位置，未覆盖的
输出位置写零，不执行数值运算或数据类型转换。

### 设计依据及优先级

当不同资料存在描述差异时，按以下优先级确定本提交的接口和行为：

1. 社区任务书：确定任务范围、目标硬件和验收方向。
2. `ops-sparse/include/cann_ops_sparse.h` 公开头文件：确定实际 ABI、参数类型和符号名称。
3. `ops-sparse/sparse/sparse2dense/arch35/`：确定已有 C++ 接口的参数校验和返回行为。
4. NVIDIA cuSPARSE 公开文档：作为 SparseToDense 功能语义的对标参考。

公开头文件已经提供 `aclsparseSparseToDense_bufferSize`，因此本设计沿用该名称，
不新增 `aclsparseSparseToDenseGetBufferSize` 或其他同功能符号，避免形成第二套 ABI。

## 背景介绍

### 现有能力与任务差距

`ops-sparse` 已公开 SparseToDense Generic C++ API，并已有 arch35（DAV-3510）
实现，但缺少面向 Atlas A2/A3 的 arch22（DAV-2201）实现。本任务沿用已有接口，
补齐 arch22 Host、Ascend C Kernel 和 C++ 测试。

| 项目 | 当前基线 | 本任务 |
| --- | --- | --- |
| 公开接口 | 已有 bufferSize 和 execute 两个 Generic C++ API | 保持 ABI 不变 |
| arch35 | 已有 DAV-3510 Host/Kernel | 不改变其计算方案 |
| arch22 | 无 SparseToDense 实现 | 新增 DAV-2201 Host/Kernel |
| 测试 | 测试内容位于 arch35 目录 | 抽取公共测试主体并增加 arch22 CSV |
| value 类型 | arch35 支持 FP32/FP16/BF16/INT32/INT8 | arch22 在此基础上增加 COMPLEX64 |

除 arch22 额外支持 COMPLEX64、对三种格式统一执行 nnz 上限前置校验外，
其余接口语义和参数校验与 arch35 对齐。两个架构的清零和 Kernel 方案按各自硬件能力实现。
Python/ATen 接入和性能 benchmark 不在本提交范围内。

### 公开 C++ 接口现状

#### 算法枚举和接口

以下原型以 `ops-sparse/include/cann_ops_sparse.h` 为准：

```cpp
typedef enum aclsparseSparseToDenseAlg_t {
    ACL_SPARSE_SPARSETODENSE_ALG_DEFAULT = 0,
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

`bufferSize` 当前恒返回 0，因此执行接口的 `buffer` 可以传 `nullptr`。
接口按 handle 绑定的 stream 异步下发；调用方读取或释放相关 Device 内存前需要同步
该 stream。

#### 参数与错误行为

| 参数 | 所属接口 | 方向与位置 | 约束 | 错误行为 |
| --- | --- | --- | --- | --- |
| `handle` | 两个接口 | IN, HOST | 非空；执行接口要求已绑定非空 stream | 空 handle 返回 `ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR`；执行时 stream 未设置返回 `ACL_SPARSE_STATUS_INVALID_VALUE` |
| `matA` | 两个接口 | IN, HOST 描述符，数据在 Device | 非空；格式为 CSR/CSC/COO；I32 索引；value 类型在支持列表内 | 描述符为空或字段非法时返回 `INVALID_VALUE` 或 `NOT_SUPPORTED` |
| `matB` | 两个接口 | IN, HOST 描述符，数据在 Device | 非空；行列数和 value 类型与 `matA` 一致；values 非空；ld 合法 | 描述符、维度、ld 或 values 非法时返回 `INVALID_VALUE`；类型不一致返回 `NOT_SUPPORTED` |
| `alg` | 两个接口 | IN, HOST | 仅支持 `ACL_SPARSE_SPARSETODENSE_ALG_DEFAULT` | 其他值返回 `ACL_SPARSE_STATUS_NOT_SUPPORTED` |
| `bufferSize` | 查询接口 | OUT, HOST | 非空 | 空指针返回 `ACL_SPARSE_STATUS_INVALID_VALUE` |
| `buffer` | 执行接口 | IN, DEVICE | 当前 workspace 为 0，可传 `nullptr` | 当前实现不解引用该参数 |

可能返回的状态包括：

| 返回码 | 含义 |
| --- | --- |
| `ACL_SPARSE_STATUS_SUCCESS` | 查询或异步任务下发成功 |
| `ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR` | handle 为空 |
| `ACL_SPARSE_STATUS_INVALID_VALUE` | 描述符、维度、ld、indexBase、Device 指针或 stream 等参数非法 |
| `ACL_SPARSE_STATUS_NOT_SUPPORTED` | 格式、算法、索引类型、value 类型或规模超出支持范围 |
| `ACL_SPARSE_STATUS_INTERNAL_ERROR` | 获取 AIV 核数或 Runtime 清零等内部操作失败 |

稀疏索引数组和 offsets 的内容合法性由调用方保证，详见“算子约束限制”和
“风险与对策”章节。

# 需求分析（required）

## 需求描述

在不改变现有 SparseToDense 公开 ABI 和 arch35 计算方案的前提下，使用 Ascend C
为 arch22 实现 CSR、CSC、COO 到稠密矩阵的转换。C++ 路径支持 FP32、FP16、
BF16、INT32、INT8、COMPLEX64 六种 value 类型，支持 ROW/COL 布局、合法的
ld padding 以及 index base 0/1，并提供对应的正确性和负向测试。

## 需求拆解

1. 沿用 `aclsparseSparseToDense_bufferSize` 和
   `aclsparseSparseToDense`，不新增同功能符号。
2. 支持 CSR、CSC、COO 三种格式以及 ROW、COL 两种稠密输出布局。
3. 支持 FP32、FP16、BF16、INT32、INT8、COMPLEX64，输入输出 value
   类型必须一致。
4. Host 完成描述符、格式、dtype、indexBase、ld、规模、Device 指针和
   stream 校验，并保持约定的错误码。
5. CSR 按行、CSC 按列、COO 按 nnz 均分到 AIV 核。
6. Kernel 保留 CSR/CSC scatter、denseLine、COO scatter 三条路径，
   并使用滑动窗口、run 合并和 headSlots。
7. 非行直写路径使用 zero kernel；仅 `m/n/nnz == 0` 的退化路径使用
   `aclrtMemsetAsync`。
8. 稀疏索引、offsets 和 COO 坐标唯一性由调用方保证，Device 侧不新增检查。
9. 提供 C++ GTest + CSV 正确性、边界和负向测试；性能数据按任务书场景后续测量。
10. Python/ATen 不在本提交范围内。

## 本提交范围边界

### 本设计不覆盖 Python/ATen

当前代码没有 Python API、ATen operator registration 或 NPU dispatch 实现，本提交不新增
这些能力。

| 项目 | 本提交状态 |
| --- | --- |
| Python 入口 | 不包含 |
| ATen 入口 | 不包含 |
| Dispatch key | 不适用 |
| Python/ATen shape/dtype 约束 | 不适用 |
| CPU fallback | 不提供；C++ NPU 接口不会回退到 CPU |
| Python UT | 不包含，待后续独立任务实现 |

# 详细设计（required）

## 算子分析

### 数学公式与功能语义

设稀疏矩阵为 `A`，稠密输出为 `B`：

```text
B = dense(A)

CSR:
  for row in [0, M):
    for p in [rowOffsets[row], rowOffsets[row + 1]):
      B[row, colInd[p]] = values[p]

CSC:
  for col in [0, N):
    for p in [colOffsets[col], colOffsets[col + 1]):
      B[rowInd[p], col] = values[p]

COO:
  for p in [0, nnz):
    B[rowInd[p], colInd[p]] = values[p]

其他逻辑位置写 0。
```

ROW 布局的线性下标为 `row * ld + col`，COL 布局的线性下标为
`col * ld + row`。index base 为 1 时，kernel 在使用 offsets 和 indices 前减去 1。

COO 允许重复坐标。多个值写同一目标坐标时，最终结果为其中任意一个值，
与并行写入下的 cuSPARSE 对标语义一致；本设计不拒绝重复坐标。

### 支持数据类型

kernel 不解释 value 的数值语义，仅按位搬运，因此按搬运单元宽度实例化：

| value 类型 | ACL 枚举 | 元素字节数 | kernel 实例 | `unitsPerValue` |
| --- | --- | ---: | --- | ---: |
| FP32 | `ACL_FLOAT` | 4 | `b32` | 1 |
| FP16 | `ACL_FLOAT16` | 2 | `b16` | 1 |
| BF16 | `ACL_BF16` | 2 | `b16` | 1 |
| INT32 | `ACL_INT32` | 4 | `b32` | 1 |
| INT8 | `ACL_INT8` | 1 | `b8` | 1 |
| COMPLEX64 | `ACL_COMPLEX64` | 8 | `b32` | 2 |

不支持 FP64，不支持 value 类型转换。COMPLEX64 是 arch22 的扩展能力，
arch35 不支持该类型。

### 支持形状、格式与布局

- 输入为二维稀疏矩阵，格式为 CSR、CSC 或 COO；
- 输出为同维度的二维稠密矩阵，支持 ROW 或 COL 布局；
- ROW 要求 `ld >= n`，COL 要求 `ld >= m`；
- 支持 `m == 0`、`n == 0`、`nnz == 0`；
- `m`、`n`、`ld`、`nnz` 均不超过 INT32_MAX；
- 索引类型仅支持 `ACL_SPARSE_INDEX_32I`。

## 算子实现

### 实现方案

#### 总体调用链

```text
调用方
  ├─ aclsparseSparseToDense_bufferSize
  │    ├─ 校验 handle / matA / matB / alg / bufferSize
  │    └─ 返回 bufferSize = 0
  │
  └─ aclsparseSparseToDense
       ├─ Host 公共参数校验
       ├─ 读取 handle 中的 stream
       ├─ 计算输出物理存储大小
       ├─ 判断 m == 0、n == 0 或 nnz == 0
       │    └─ 是：aclrtMemsetAsync（有物理输出时）→ 返回
       ├─ 选择 denseLine
       ├─ 计算分核和构造 tiling
       └─ sparse2dense_kernel_do
            ├─ denseLine == 0：先下发 sparse2dense_zero_kernel
            ├─ 按 value 宽度选择 b8 / b16 / b32
            └─ 转换 kernel 按格式执行
                 ├─ CSR/CSC scatter
                 ├─ CSR 行直写 / CSC 列直写
                 └─ COO scatter
```

同一 stream 上先下发 zero kernel、再下发转换 kernel，利用 stream 顺序保证转换开始前
清零完成，不在转换 kernel 内增加全核同步。

#### 目录分层

```text
ops-sparse/
├── include/
│   └── cann_ops_sparse.h                         # 公开枚举和 C++ API
├── sparse/sparse2dense/
│   ├── README.md                                 # 用户文档
│   ├── arch22/
│   │   ├── sparse2dense_host.cpp                 # 参数校验、分核、tiling、launch
│   │   ├── sparse2dense_kernel.cpp               # zero kernel 和三条转换路径
│   │   ├── sparse2dense_kernel.h                 # kernel_do 声明
│   │   ├── sparse2dense_tiling_data.h            # tiling 与 UB 分块常量
│   │   └── sparse2dense.h                        # 描述符、dtype、format 映射
│   └── arch35/                                   # 已有实现，本提交不改变其计算方案
└── test/sparse2dense/
    ├── sparse2dense_test_body.h                  # arch22/arch35 公共 GTest 主体
    ├── sparse2dense_golden.h                     # CPU golden
    ├── sparse2dense_npu_execution.h              # NPU 执行封装
    ├── sparse2dense_npu_wrapper.h                # 测试参数和数据封装
    ├── arch22/                                   # arch22 入口和 CSV
    └── arch35/                                   # arch35 入口和 CSV
```

本架构没有 Python 层。

#### Host 侧设计

##### 参数校验

Host 侧依次校验：

- format 为 CSR、CSC 或 COO；
- ptr/index 类型为 `ACL_SPARSE_INDEX_32I`；
- indexBase 为 ZERO 或 ONE；
- value 类型在六种支持类型内；
- `m`、`n`、`ld`、`nnz` 不超过 INT32_MAX；
- `matA` 和 `matB` 维度、value 类型一致；
- ROW 时 `ld >= n`，COL 时 `ld >= m`；
- `matB.values` 非空；
- `nnz > 0` 时稀疏 ptrs、indices、values 非空；
- execute 时 handle 已绑定非空 stream。

Host 不检查 Device 数组内 offsets、indices 或坐标唯一性。

##### 1. 分核策略

按格式选择互不重叠的切分维度：

| 格式 | 切分维度 | 每核任务 |
| --- | --- | --- |
| CSR | 行数 `m` | 连续行区间 |
| CSC | 列数 `n` | 连续列区间 |
| COO | `nnz` | 连续非零元区间 |

设切分维度为 `dim`，AIV 核数为 `aivCoreNum`：

```text
perBlock  = ceil(dim / aivCoreNum)
useBlocks = ceil(dim / perBlock)
```

每个核处理
`[blockIdx * perBlock, min((blockIdx + 1) * perBlock, dim))`。
维度小于核数时不启动多余空核。

##### 2. 数据分块和内存优化策略

Host 将固定 UB 分块参数写入编译期常量，并通过 `Sparse2DenseTilingData`
向 Kernel 传递矩阵维度、布局、格式、分核任务量、value 搬运单元数、
denseLine 标志、nnz 和输出物理字节数。主要 UB 分块常量见 Kernel 侧设计。

denseLine、滑动窗口、run 合并和 headSlots 分别优化连续输出、跨行短搬入、
连续目标地址写出和非对齐段头写出。COO 当前使用深度为 1 的队列，不启用双缓冲。

##### denseLine 判定

以下场景的一行或一列在稠密输出中连续，且整片输出可由转换 kernel 覆盖：

- CSR + ROW，且 `ld == n`；
- CSC + COL，且 `ld == m`。

Host 在上述场景置 `denseLine = 1`。存在 ld padding 时不使用行直写，因为 padding
区域不属于任何逻辑行或列，必须由独立清零路径处理。

##### 清零策略

| 场景 | 清零方式 |
| --- | --- |
| `m == 0`、`n == 0` 或 `nnz == 0` | 不启动转换 kernel；存在物理输出时调用 `aclrtMemsetAsync` |
| `denseLine == 1` | 转换 kernel 在 UB 中补零并整行/整列覆盖，不下发 zero kernel |
| 其他 scatter 场景 | 先下发 `sparse2dense_zero_kernel`，再下发转换 kernel |

##### 3. TilingKey 规划策略

本实现不使用 TilingKey。格式、布局、denseLine 和 `unitsPerValue` 通过
`Sparse2DenseTilingData` 字段传递；Host 根据 value 宽度选择 b8、b16 或 b32
Kernel 入口。这样保持现有实现的三条处理路径，不额外生成格式与布局组合的
TilingKey 分支。

#### Kernel 侧设计

##### Init 与 Process

Kernel 的 Init 阶段解析 TilingData、绑定 offsets/indices/values/dense 的 GM
地址，并按当前路径初始化 TQue/TBuf。Process 阶段根据 format 和 denseLine
选择 CSR/CSC scatter、denseLine 或 COO scatter。

该算子不进行数值计算，主要过程为 CopyIn、索引读取与地址计算、run 组织以及
CopyOut。COO 路径没有跨分块双缓冲。

##### UB 分块

| 常量 | 当前值 | 用途 |
| --- | ---: | --- |
| `kSparse2DenseTileNnz` | 2048 | 一次搬入的非零元数量 |
| `kSparse2DenseTileMajor` | 1023 | CSR/CSC 一次处理的行/列数量，offsets 搬入 1024 项 |
| `kSparse2DenseBlockBytes` | 32 B | GM/UB 搬运对齐粒度 |
| `kSparse2DenseHeadSlots` | 64 | 非对齐段头暂存槽位 |
| `kSparse2DenseZeroBytes` | 32768 B | zero kernel 的 UB 零缓冲 |
| `kSparse2DenseLineBytes` | 131072 B | denseLine 的 UB 行/列缓冲 |

##### CSR/CSC scatter

CSR 按行、CSC 按列读取 offsets。indices 和 values 使用长度为
`kSparse2DenseTileNnz` 的 UB 滑动窗口；当前位置不在窗口内时再执行一次 MTE2
搬入，窗口可以跨行或跨列复用。

窗口内按目标稠密下标扫描并执行 run 合并：相邻非零元的目标地址连续时合并为一次
`DataCopyPad`；目标地址不连续时结束当前 run 并开始下一段。

##### denseLine 行/列直写

CSR + ROW 或 CSC + COL 的无 padding 场景，在 UB 中分块构造完整稠密行或列：

1. 使用 `Duplicate` 把当前 UB 分块清零；
2. 使用 `GetValue` 读取索引和值，并通过 `SetValue` 填入对应位置；
3. 使用 `DataCopyPad` 把包含零和非零值的连续分块写入 GM。

该路径不需要独立 zero kernel。

##### COO scatter

COO 按 `nnz` 均分到各核，并按 `kSparse2DenseTileNnz` 分块搬入 rowInd、
colInd 和 values。当前实现为三个独立 `TQue`，每个分块完成搬入、出队和写出后
再处理下一分块，**未启用双缓冲，也不声称搬入与计算跨分块重叠**。

COO 复用 run 合并和非对齐段头处理。重复坐标不查重、不拒绝，最终结果允许为重复值
集合中的任意一个。

##### headSlots 非对齐段头

`DataCopyPad` 的 UB 源地址需要满足 32 B 对齐。run 起点不对齐时，把不对齐段头复制到
`headBuf_` 的 32 B 槽位，累计最多 64 个槽位后统一下发 MTE3。切换滑动窗口、
结束 COO 分块或退出 kernel 前调用 `FlushHeads()`，确保暂存数据已写出。

### 64bit 溢出与物理存储

#### 输出物理存储

ld padding 属于输出物理存储的一部分。物理元素数和字节数定义为：

```text
ROW: physicalElements = int64(M) * int64(ld)
COL: physicalElements = int64(N) * int64(ld)
physicalBytes = uint64(physicalElements) * uint64(elementSize)
```

计算 `M * ld`、`N * ld` 和字节数时必须使用 64 位整型中间值，不能先以 int32
相乘再转换。该物理字节数用于退化路径的 `aclrtMemsetAsync` 范围，以及 scatter
路径的 zero kernel 范围。

#### 当前实现状态和边界

当前 Host 先检查 `m`、`n`、`ld`、`nnz <= INT32_MAX`，并在
`DenseStorageBytes` 中使用 `int64_t` 计算物理元素数，再转换为 `size_t` 计算字节数。

当前代码尚未对 `physicalElements * elementSize` 增加独立的 checked-multiply
溢出判断。因此本设计不宣称支持乘积接近 64bit 上限的极端描述符；显式乘法溢出检查
列为待实现项，完成前不扩展现有规模约束，也不把该极端场景作为已支持能力。

### Ascend C / Runtime 接口

本实现只依赖当前代码中已经使用的接口：

| 接口或机制 | 侧别 | 当前用途 |
| --- | --- | --- |
| `aclrtMemsetAsync` | Host / Runtime | 仅 `m/n/nnz == 0` 退化路径清零 |
| `<<<numBlocks, ..., stream>>>` | Host / Runtime | 下发 zero kernel 和转换 kernel |
| `GetBlockIdx` / `GetBlockNum` | Kernel | 计算本核处理区间 |
| `TPipe::InitBuffer` | Kernel | 初始化 offsets、indices、values、line、zero、head 缓冲 |
| `TQue::AllocTensor` / `EnQue` / `DeQue` / `FreeTensor` | Kernel | 管理分块搬入缓冲 |
| `TBuf::Get` | Kernel | 获取滑动窗口、denseLine 和 headSlots 缓冲 |
| `DataCopyPad` | Kernel | GM→UB 搬入以及 UB→GM 写出 |
| `Duplicate` | Kernel | denseLine 分块和 zero kernel 的 UB 清零 |
| `LocalTensor::GetValue` / `SetValue` | Kernel | 读取 offsets/indices/value，构造 denseLine 或 headSlots |
| `SetFlag` / `WaitFlag` | Kernel | 保证 Scalar、Vector、MTE2、MTE3 之间的数据依赖 |

本设计不基于未在当前实现中使用的 Device 侧查重、offsets 校验或双缓冲接口。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas A2 训练系列产品 | √ |
| Atlas A3 训练系列产品 | √ |

对应 `ascend910b*` / `ascend910_93*` 和 arch22 / DAV-2201。

## 算子约束限制

1. `matA` 和 `matB` 的行列数及 value 类型必须一致，不支持跨类型转换。
2. 仅支持 CSR、CSC、COO；索引类型仅支持 `ACL_SPARSE_INDEX_32I`。
3. 支持 FP32、FP16、BF16、INT32、INT8、COMPLEX64，不支持 FP64。
4. 仅支持 `ACL_SPARSE_SPARSETODENSE_ALG_DEFAULT`，workspace 大小为 0。
5. ROW 要求 `ld >= n`，COL 要求 `ld >= m`。
6. `m`、`n`、`ld`、`nnz` 均不得超过 INT32_MAX。
7. 调用方必须保证 indices 在矩阵边界内，CSR/CSC offsets 单调且末项与
   nnz 一致；Host 和 Device 均不检查数组内容。
8. COO 允许重复坐标，最终结果为重复值中的任意一个，不保证固定写入顺序。
9. execute 要求 handle 已绑定非空 stream；接口异步执行，调用方负责同步和
   Device 内存生命周期。
10. Python/ATen 和 CPU fallback 不在本提交范围内。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | value 仅按位搬运；正确性用例与 CPU golden 逐字节比较 | 任务书功能语义及当前实现 |
| 性能标准 | 按任务书指定场景和统一计时口径测量；本设计阶段不填写未实测耗时或倍率 | 任务书，具体门限待评审确认 |

## 测试设计

### C++ UT 组织

测试采用 GTest + CSV 参数化方式。公共测试主体、CPU golden 和 NPU 执行封装由
arch22/arch35 复用，各架构目录保留入口和独立 CSV。

arch22 当前共有 63 个用例：

- 44 个 L0/L1 正确性用例；
- 19 个 L2 负向用例。

不能只以“全部通过”作为覆盖说明，测试维度如下：

| 维度 | 已覆盖内容 |
| --- | --- |
| 稀疏格式 | CSR、CSC、COO |
| value 类型 | FP32、FP16、BF16、INT32、INT8、COMPLEX64 |
| 输出布局 | ROW、COL |
| leading dimension | 最小合法 ld、ld padding |
| indexBase | ZERO、ONE |
| 稀疏分布 | 全零、单元素、全密、不同稀疏率、空行 |
| 形状 | `m/n == 0`、1×1、单行、单列、宽矩阵、高矩阵、中等规模矩阵 |
| 分块边界 | 行数超过 major tile、单行 nnz 超过 nnz tile、COO nnz 超过 nnz tile |
| 搬运边界 | INT8 非对齐 run、COL scatter、COMPLEX64 双搬运单元 |
| 一致性 | 相同输入的 CSR/CSC/COO 输出交叉比较 |
| 负向参数 | 空 handle/描述符/输出、非法 ld/format/indexBase/alg、类型或维度不匹配、未设置 stream |
| 规模限制 | `m > INT32_MAX`、`nnz > INT32_MAX`、I64 索引不支持；Host 同样限制 `n`，当前 L2 未单独覆盖该分支 |

建议的 C++ 验证命令为：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
bash build.sh --ops=sparse2dense --soc=ascend910_93 --run
```

已有验证记录为 63 个用例完成执行。提交验收时应保留构建环境、CANN 版本和完整测试日志，
而不是只记录汇总结论。

### 尚未覆盖

- Python/ATen UT：本提交范围不含。
- 性能 benchmark：本设计阶段无测量数据，待任务书性能场景和统一计时口径明确后执行。
- 极端 64bit 乘法溢出：显式 checked-multiply 尚未实现，列为待实现测试。
- `n > INT32_MAX`：与 `m` 共用同一 Host 校验分支，当前 L2 仅覆盖 `m` 超限。
- COO 重复坐标：实现保留并行任意值语义，当前 63 个用例未提供重复坐标专项验证。

## 性能设计与验收数据

### 已落地优化

| 优化项 | 当前实现 |
| --- | --- |
| denseLine | CSR + ROW、CSC + COL 且无 padding 时在 UB 构造整行/整列，一次连续写出 |
| 滑动窗口 | CSR/CSC 的 indices 和 values 跨行/列复用 UB 窗口，减少短搬入 |
| run 合并 | 连续目标地址合并为一次 `DataCopyPad` |
| headSlots | 非对齐段头批量暂存后写出，减少逐段同步 |
| 独立 zero kernel | scatter 路径先清零、后转换，依赖 stream 顺序，不在转换 kernel 内做全核同步 |

上述条目描述的是已实现机制，不代表已测得固定加速倍率。

### 验收数据预留

本设计阶段不填写未实测耗时或倍率。后续按任务书确认的场景和计时口径补充：

| 任务书场景 | 格式 / dtype / 布局 | M / N / nnz / ld | 指标与门限 | 实测值 | 结论 |
| --- | --- | --- | --- | --- | --- |
| 待任务书确认 | 待填写 | 待填写 | 待填写 | 待测 | 待测 |
| 待任务书确认 | 待填写 | 待填写 | 待填写 | 待测 | 待测 |
| 泛化场景 | 待填写 | 待填写 | 待填写 | 待测 | 待测 |

性能测试至少应记录预热次数、测量次数、同步边界、统计量和 CANN/SOC 环境。
在 benchmark 完成前不写“加速若干倍”或“快一个量级”等结论。

## 兼容性分析

### 分层职责

| 层次 | 职责 |
| --- | --- |
| 公开头文件 | 维护稳定的枚举、接口签名和通用语义，不暴露 arch22 内部路径 |
| arch22 Host | 参数校验、物理大小计算、denseLine 判定、分核、tiling 和 kernel 下发 |
| arch22 Kernel | zero kernel、CSR/CSC scatter、denseLine、COO scatter 和搬运优化 |
| 公共测试层 | 参数解析、CPU golden、接口状态和逐字节结果比较 |
| 架构测试层 | 按架构维护 CSV，避免把 arch22 特有 dtype 或边界强加给 arch35 |

### 与 arch35 的隔离

- 构建系统根据 `SOC_ARCH_DIRS` 选择 `arch22` 或 `arch35` 源文件；
- arch22 新增实现不修改 arch35 Host/Kernel 的计算方案；
- 公共测试主体只承载接口级逻辑，arch22/arch35 各自维护用例数据；
- COMPLEX64 仅加入 arch22 支持矩阵，不改变 arch35 的 NOT_SUPPORTED 行为；
- 公共接口注释只描述“由算子内部清零”，具体清零方式留给架构实现。

## 风险与对策

| 风险 | 当前行为 | 对策或边界说明 |
| --- | --- | --- |
| 稀疏索引或 offsets 非法 | Device 侧不检查，非法数据可能导致越界访问或错误结果 | 在公开约束中明确由调用方保证：indices 在矩阵边界内，CSR/CSC offsets 单调且末项与 nnz 一致 |
| COO 重复坐标并行写 | 最终值为重复值中的任意一个 | 保留对标语义，不拒绝重复；使用包含关系而非固定写入顺序定义结果 |
| `m/n/nnz == 0` | 不启动转换 kernel，按物理输出大小调用 `aclrtMemsetAsync` 或直接返回 | 单独覆盖零维和零 nnz 用例，确保不访问空稀疏数组 |
| scatter 清零和转换的时序 | zero kernel 与转换 kernel 分开发射 | 固定在同一 stream 先下发 zero kernel，再下发转换 kernel |
| COO 未启用双缓冲 | 搬入与计算未跨分块重叠，可能限制性能 | 作为当前实现边界记录；本提交不引入双缓冲流水 |
| 随机 scatter 和非对齐小写 | 连续写合并机会减少 | 使用 run 合并和 headSlots，性能效果以后续 benchmark 为准 |
| 输出物理字节数溢出 | 当前有 64bit 中间值，但没有独立 checked-multiply | 不宣称支持乘积接近 64bit 上限的描述符；显式溢出检查列为待实现 |
| arch22 扩展影响 arch35 | 两个架构支持矩阵和清零方式不同 | 架构目录隔离、按 SOC 编译，并分别维护 CSV 用例 |

