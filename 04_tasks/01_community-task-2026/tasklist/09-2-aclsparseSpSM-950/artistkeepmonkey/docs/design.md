# 需求背景（required）

## 需求来源

本需求来源于昇腾算子开源社区 9 月任务“aclsparseSpSM 算子开发（950）”，面向 Ascend 950（A5，arch35/DAV_3510）完善多右端稀疏三角矩阵求解能力。接口语义对齐 cuSPARSE Generic SpSM，工程实现遵循 `ops-sparse` 的 Legacy C++ API、描述符和 Handle/Stream 约定。

目标交付包括公共接口、Host 多阶段状态管理、Ascend C Kernel、C++ UT/ST、精度与性能测试、内存测试、Profiler 证据及 README/设计文档。

## 背景介绍

SpSM 求解下式：

```text
op(A) · C = alpha · op(B)
```

其中 A 是稀疏三角方阵，B/C 是具有多个右端的稠密矩阵，`alpha` 是与计算类型一致的标量。当前 `ops-sparse` arch35 SpSM 已有 CSR、FP32、`opA=N/T`、`opB=N`、Host pointer mode 和 ROW/COL 数据搬运路径，内部采用 level scheduling 和 Ascend C 向量 kernel；公共头文件也已提供 SpSM 三阶段接口。

当前实现存在以下能力缺口：

1. 稀疏格式只覆盖 CSR，CSC 和 COO 不能直接使用；
2. 只支持 FP32，缺少 complex64 及 H（共轭转置）语义；
3. `opB` 只支持 N，Device pointer mode 尚未打通；
4. `UpdateMatrix` 公共接口和 GENERAL/DIAGONAL 更新状态没有完整实现；
5. 未排序坐标、重复坐标和不同输入格式缺少统一的确定性规范化路径；
6. 需要补齐大规模多 RHS 的性能、内存和 NPU profiler 证据。

本设计以现有 `sparse/spsm/arch35/` 的 level scheduling、workspace 绑定和多核同步方案为基线，扩展成格式统一、类型统一、操作统一的 NPU 求解流水线。CPU 只负责参数检查、阶段状态和 tiling 元数据组织；格式归一化、依赖元数据生成和主求解均由 NPU kernel 完成，不使用 CPU fallback 替代 NPU 计算。

# 需求分析（required）

## 需求描述

在 Ascend 950PR/950 A5 上实现 `aclsparseSpSM` 多右端稀疏三角求解，支持：

- A 的 CSR、CSC、COO 三种格式；
- FP32 和 complex64，索引统一 I32，index base 支持 ZERO/ONE；
- `opA = N/T/H`，complex64 的 H 执行共轭转置，FP32 的 H 与 T 等价；
- `opB = N/T/H`，B/C 支持 ROW/COL 稠密布局和 leading dimension；
- LOWER/UPPER、UNIT/NON_UNIT、DEFAULT algorithm、多 RHS；
- alpha 的 Host/Device pointer mode；
- B/C values 共用同一 Device 指针的 in-place 求解；
- BufferSize → Analysis → SpSM 的状态机，以及 GENERAL/DIAGONAL `UpdateMatrix`；
- 未排序和重复坐标的确定性规范化，重复执行结果 bitwise deterministic；
- Handle stream 上的异步 kernel 提交，Solve 内不做 host stream 同步。

## 需求拆解

1. 扩展 `include/cann_ops_sparse.h`：新增 `aclsparseSpSMUpdate_t` 和 `aclsparseSpSMUpdateMatrix` 声明，保留既有 SpSM 函数签名。
2. 设计统一稀疏中间格式：所有 CSR/CSC/COO 在 Analysis 阶段由 NPU kernel 转换成 0-based、按行有序的 canonical CSR。
3. 为 N/T/H 建立有效三角方向、索引和复数共轭标志；H 仅对 complex64 的 A values 做共轭。
4. 以 level scheduling 为核心，按依赖层级和 RHS tile 并行；level 之间使用跨核同步保证读写顺序。
5. 支持空 values 描述符的 BufferSize/Analysis；Solve 及 UpdateMatrix 时要求有效 Device values。
6. 建立 descriptor signature，拒绝 Analysis 到 Solve 期间的 matA/matB/matC、op、dtype、shape、layout、algorithm 或 buffer 变更。
7. 采用 CPU FP64/complex128 golden，完成精度、确定性、内存、性能和 profiler 验证。

# 详细设计（required）

## 算子分析

### 数学公式与形状

设 A 为 `m×m` 三角稀疏矩阵，RHS 数为 `nrhs`：

```text
op(A) · X = alpha · op(B)
```

输出 `X` 以 `matC` 的物理布局写回；当 `opB=N` 时 B/C 的物理逻辑形状为 `m×nrhs`，当 `opB=T/H` 时 B/C 的物理形状为 `nrhs×m`，使 `opB(B)` 和 `opB(C)` 均为 `m×nrhs`。ROW 布局的元素地址为 `row*ld + col`，COL 布局的元素地址为 `col*ld + row`。

`opA` 的定义：

| opA | FP32 | complex64 |
| --- | --- | --- |
| N | A | A |
| T | Aᵀ | Aᵀ |
| H | Aᵀ | Aᴴ，values 做共轭 |

`opB` 的定义与上表相同，但作用在稠密 RHS 上。FP32 的 H 与 T 等价；complex64 的 H 对 B values 做共轭后再转置。

### 三角方向

原始 A 的 `fillMode` 描述原始存储三角。`opA=N` 时有效三角保持不变；`opA=T/H` 时有效三角上下翻转：LOWER 变为 UPPER，UPPER 变为 LOWER。有效方向决定 level scheduling 的前向或后向替换：

- effective LOWER：第 i 行依赖 `j<i`，按 level 递增求解；
- effective UPPER：第 i 行依赖 `j>i`，按 level 递增求解（行号方向可反向遍历）；
- UNIT：对角隐式为 1，不读取或更新显式对角值；
- NON_UNIT：canonical CSR 中必须存在显式对角，且 Solve/Update 时不能为 0。

### 支持数据类型

| 项目 | 支持值 |
| --- | --- |
| A values | FP32、complex64 |
| B/C values | 与 A 和 `computeType` 一致的 FP32 或 complex64 |
| alpha | Host 或 Device 上的 FP32/complex64 标量 |
| row/column index | I32，Device 内存 |
| computeType | `ACL_FLOAT`、`ACL_COMPLEX64` |

complex64 使用交错 `{real, imag}` 存储。Kernel 中复数乘加为 FP32 分量运算；H 路的 A/B 读取执行 `conj(real+imag*i)=real-imag*i`。不把 H 当作普通转置处理。

## 公共接口

### 函数原型

已有三阶段接口保持不变，并在 `include/cann_ops_sparse.h` 新增：

```c
typedef enum aclsparseSpSMUpdate_t {
    ACL_SPARSE_SPSM_UPDATE_GENERAL = 0,
    ACL_SPARSE_SPSM_UPDATE_DIAGONAL = 1
} aclsparseSpSMUpdate_t;

aclsparseStatus_t aclsparseSpSMUpdateMatrix(
    aclsparseHandle_t handle,
    aclsparseSpSMDescr_t spsmDescr,
    void *newValues,
    aclsparseSpSMUpdate_t updatePart);
```

三阶段接口为：

```c
aclsparseStatus_t aclsparseSpSMCreateDescr(aclsparseSpSMDescr_t *spsmDescr);
aclsparseStatus_t aclsparseSpSMDestroyDescr(aclsparseSpSMDescr_t spsmDescr);
aclsparseStatus_t aclsparseSpSMBufferSize(
    aclsparseHandle_t handle, aclsparseOperation_t opA, aclsparseOperation_t opB,
    const void *alpha, aclsparseConstSpMatDescr_t matA,
    aclsparseConstDnMatDescr_t matB, aclsparseDnMatDescr_t matC,
    aclDataType computeType, aclsparseSpSMAlg_t alg,
    aclsparseSpSMDescr_t spsmDescr, size_t *bufferSize);
aclsparseStatus_t aclsparseSpSMAnalysis(
    aclsparseHandle_t handle, aclsparseOperation_t opA, aclsparseOperation_t opB,
    const void *alpha, aclsparseConstSpMatDescr_t matA,
    aclsparseConstDnMatDescr_t matB, aclsparseDnMatDescr_t matC,
    aclDataType computeType, aclsparseSpSMAlg_t alg,
    aclsparseSpSMDescr_t spsmDescr, void *buffer);
aclsparseStatus_t aclsparseSpSM(
    aclsparseHandle_t handle, aclsparseOperation_t opA, aclsparseOperation_t opB,
    const void *alpha, aclsparseConstSpMatDescr_t matA,
    aclsparseConstDnMatDescr_t matB, aclsparseDnMatDescr_t matC,
    aclDataType computeType, aclsparseSpSMAlg_t alg,
    aclsparseSpSMDescr_t spsmDescr);
```

### 参数说明与合法性

| 参数 | 类型/位置 | 设计语义与约束 |
| --- | --- | --- |
| `handle` | Handle，Host | 必须是有效句柄；空指针返回 `ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR`；stream 从句柄读取 |
| `opA` | 枚举，Host | N/T/H；非法值返回 `ACL_SPARSE_STATUS_INVALID_VALUE`；FP32 H 等价 T |
| `opB` | 枚举，Host | N/T/H；物理 B/C 形状按 opB 解释；非法值返回 `ACL_SPARSE_STATUS_INVALID_VALUE` |
| `alpha` | 标量指针 | pointer mode=HOST 时为 Host 标量，DEVICE 时为 Device 0-D 标量；类型必须与 computeType 一致 |
| `matA` | 稀疏描述符 | 二维方阵；format 为 CSR/CSC/COO；values 在 BufferSize/Analysis 可为空，Solve 必须有效；索引 I32/base 0 或 1 |
| `matB` | const dense 描述符 | B 的物理维度为 N 时 `m×nrhs`，T/H 时 `nrhs×m`；ROW 要求 `ld≥cols`，COL 要求 `ld≥rows`；Analysis 可为空 values |
| `matC` | dense 描述符 | 物理维度和 layout/ld 与 B 对应；输出 `opB(C)=X`；可与 matB 共用 values；Analysis 可为空 values |
| `computeType` | 枚举，Host | 仅 `ACL_FLOAT` 或 `ACL_COMPLEX64`，且与 A/B/C/alpha 一致 |
| `alg` | 枚举，Host | 仅 `ACL_SPARSE_SPSM_ALG_DEFAULT`；未知或不支持算法返回 `ACL_SPARSE_STATUS_NOT_SUPPORTED` |
| `spsmDescr` | opaque 描述符 | Create 后使用；跨阶段保存 signature、tiling、workspace 和 update 状态 |
| `bufferSize` | Host 输出 | 非空，返回不溢出的 workspace 字节数；BufferSize 不读取 values |
| `buffer` | Device workspace | Analysis 传入，至少为 BufferSize 大小并满足 64B 对齐；直到异步 Solve 完成前不得释放或修改 |
| `newValues` | Device 指针 | GENERAL 长度为原始 nnz；DIAGONAL 长度为 m 个显式对角位置值；类型与 A values 一致 |
| `updatePart` | 枚举，Host | GENERAL 或 DIAGONAL；其他值返回 `ACL_SPARSE_STATUS_INVALID_VALUE` |

此外，矩阵 rows/cols/nnz、RHS 数、leading dimension 和 workspace 字节计算统一使用 64 位中间量并进行 `INT32_MAX`、`size_t` 溢出检查。非法维度、shape、dtype、索引类型、索引越界、缺失 NON_UNIT 对角或不满足 leading dimension 返回明确状态码。

推荐的状态码映射如下：

| 场景 | 返回码 |
| --- | --- |
| handle 为空 | `ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR` |
| 描述符、alpha、bufferSize、newValues 为空或 shape/ld/枚举非法 | `ACL_SPARSE_STATUS_INVALID_VALUE` |
| format、dtype、op、index type、algorithm 组合不支持 | `ACL_SPARSE_STATUS_NOT_SUPPORTED` 或 `ACL_SPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED` |
| Analysis/Update/Normalize/Kernel 所需 Device 资源不足 | `ACL_SPARSE_STATUS_INSUFFICIENT_RESOURCES` |
| ACL runtime 或 kernel launch 失败 | `ACL_SPARSE_STATUS_EXECUTION_FAILED` |
| Solve 未 Analysis、descriptor 已失效或阶段顺序错误 | `ACL_SPARSE_STATUS_NOT_INITIALIZED` |

## 阶段状态机

描述符状态如下：

```text
CREATED
  └─BufferSize─> BUFFER_SIZED
                   └─Analysis + valid buffer─> ANALYZED
                                                  ├─UpdateMatrix─> ANALYZED_UPDATED
                                                  └─SpSM────────> SOLVE_PENDING
```

- `CreateDescr`：只分配 Host 描述符并置空状态；
- `BufferSize`：校验所有 Host 元数据，读取 format/shape/layout/属性，计算 workspace，不读取矩阵 values；缓存 signature；
- `Analysis`：允许 values 为空；在 Device 上完成格式规范化、索引 base 归一化、确定性排序/合并、有效三角和 level 元数据生成；绑定 external buffer；
- `SpSM`：要求已 Analysis，校验本次参数和 descriptor signature 与缓存一致，要求 A/B/C values 和 alpha 可用，异步提交 kernel；
- `UpdateMatrix`：要求已 Analysis 且没有未完成的同一 buffer 操作，更新 A values 关联；GENERAL 可复用依赖拓扑，DIAGONAL 刷新对角倒数和 singular 状态；
- `DestroyDescr`：调用方必须保证相关 stream 操作完成后再销毁；销毁后所有阶段调用返回 `ACL_SPARSE_STATUS_INVALID_VALUE`。

Analysis 到 Solve 期间，matA 的 format、rows、cols、nnz、index types/base、fill/diag，matB/matC 的 rows、cols、ld、order、dtype、computeType、opA/opB、alg 和 external buffer 必须保持一致。values 指针允许通过 UpdateMatrix 替换，但 pattern 和 descriptor 元数据不能改变。

## Host 侧设计

### 参数检查和 pointer mode

Host 侧不解引用 Device pointer mode 的 alpha。根据 `aclsparseGetPointerMode` 选择：

- HOST：读取 Host 上的 FP32/complex64 alpha，并把标量复制到本次 launch 的 tiling 参数；
- DEVICE：将 alpha 指针直接作为 kernel 参数，Solve kernel 在 stream 上读取一个标量，保证 alpha 与矩阵访问的 stream 顺序一致；
- Analysis/BufferSize 只检查 alpha 非空和类型/模式，不因 values 为空失败。

`alpha=0` 不改变 workspace/阶段语义；为保持接口一致，仍校验 descriptor 和 buffer。Solve kernel 可跳过依赖乘加，只写零或清理输出的有效范围。

### canonical CSR 方案

所有格式先转为 canonical CSR：

1. CSR：按 rowOffsets 读取；base ONE 在设备侧减一；
2. CSC：以列指针作为输入分段，生成 `(row, col, value)`，再按 row/col/原始位置稳定排序；
3. COO：读取 row/col/value 三数组，按 `(row, col, original_position)` 稳定排序并生成 rowOffsets；
4. 归一化阶段合并重复 `(row,col)` 坐标。合并顺序固定为 original_position 升序，使用固定树形/顺序累加，保证同一输入 bitwise deterministic；
5. 对 `opA=T/H`，坐标交换 `(row,col)→(col,row)`，H 路对 complex values 取共轭，再重新生成 canonical CSR；
6. effective fill 在转置时翻转，UNIT/NON_UNIT 属性保留；检查每行的有效依赖方向和对角位置。

该过程由 `spsm_format_normalize_kernel` 和必要的 prefix-scan/sort/reduce kernel 在 NPU 上完成。Host 只准备输入指针、shape、临时 offset 和 kernel 参数，不把整矩阵拷回 CPU，也不以 CPU 参考实现替代 NPU 路径。

### Analysis workspace 布局

workspace 以 64B 对齐切分，偏移和大小写入 `SpsmTilingData`/descriptor：

```text
[header/signature]
[canonical rowOffsets: m+1 × I32]
[canonical colIndices: nnzCanonical × I32]
[canonical values: nnzCanonical × valueSize]
[levelRowPtr: L+1 × I32]
[levelRowIdx: m × I32]
[diagValue/invDiag: m × valueType]
[sort/reduce temporary buffers]
[dense ROW scratch: m × nrhs × valueSize, only when layout/opB needs it]
```

BufferSize 按最坏的 format、H、NON_UNIT、ROW/COL 和排序临时空间计算，保证 Analysis 不会改变所需大小。若当前矩阵已经是 canonical CSR，可跳过格式临时区，但不能缩小已返回的 workspace。所有区间都检查乘法/加法溢出和 64B 对齐。

### Level scheduling 和 tiling

依赖图以 canonical CSR 的有效三角非对角项为边：`i→j` 表示求解第 i 行需要第 j 行的 X。Device analysis kernel 计算每行 level，并通过确定性计数/scan 生成 `levelRowPtr` 与 `levelRowIdx`；同一 level 内行之间无依赖。

Solve tiling：

- blockDim 动态取当前设备 AIV core 数，不能硬编码；
- 每个 level 的行在 core 间均分，空 core 也参与 `SyncAll`；
- RHS 按 `kChunk` 分块，kChunk 按 UB 容量、复数元素大小、maxRowLen 和双缓冲计算；
- ROW 连续路径直接 GM↔UB；COL 或 opB=T/H 使用 transpose-in 到内部 ROW scratch，求解后 transpose-out 到 C 的物理 layout；
- maxRowLen、level 数和 m/n/nnz 超过 UB/GM 可承受范围时，在 Analysis 返回 `ACL_SPARSE_STATUS_INSUFFICIENT_RESOURCES` 或 `NOT_SUPPORTED`，不让 kernel 越界。

### UpdateMatrix 设计

`aclsparseSpSMUpdateMatrix` 不改变矩阵 pattern、format、shape、fill/diag、op 或 layout：

- GENERAL：Device kernel 将新 values 按原始 format 的位置映射到 canonical values；不重新构建 level 拓扑，保留排序结果和 workspace；
- DIAGONAL：只更新 m 个对角值；刷新 `diagValue/invDiag`，NON_UNIT 中零值标记为 singular，后续 Solve 返回 `ACL_SPARSE_STATUS_NOT_SUPPORTED`；UNIT 的隐式对角不允许以 DIAGONAL 修改，返回 `INVALID_VALUE`；
- 若 format 归一化产生重复坐标，GENERAL 更新仍按保存的 stable mapping 合并到 canonical values；
- Update 与 Solve 在同一 Handle stream 上顺序提交，调用方不得在异步 Update/Solve 完成前复用 `newValues` 或释放 workspace。

## Kernel 侧设计

### 格式归一化 kernel

`spsm_format_normalize_kernel` 根据 format 分支读取 CSR/CSC/COO：

1. 生成带原始序号的 tuple `(row, col, value, source_index)`；
2. base ONE 转为 0-based，校验 row/col 范围；
3. 对 opA T/H 交换坐标并按 H 标志共轭 value；
4. 设备侧稳定 radix sort，key 为 row、col、source_index；
5. 对重复坐标做固定顺序 reduce，写出 canonical CSR 和 rowOffsets；
6. 生成 diagonal map、effective fill 和每行 maxRowLen。

格式计算均在 NPU 上完成。对于非法索引、负 row/col、rowOffsets 非单调或 COO/CSC 指针不一致，kernel 写回错误标志，Host 在下一阶段返回 `INVALID_VALUE`。

### Analysis kernel

Analysis kernel 对 canonical CSR 生成 level：

- 每行读取有效依赖列，计算 `level[i]=max(level[j]+1)`；
- 对三角矩阵使用方向保证无环，检测到 cycle 或缺失/零 NON_UNIT 对角写入错误标志；
- 通过 count/scan 形成按 level 的 row index；
- 对 H complex64，依赖值已经在 canonical values 完成共轭，求解 kernel 不再分支处理 H；
- 写入 `diagValue/invDiag` 和 `SpsmTilingData` 所需的 device 元数据。

### Solve kernel

Solve kernel 是单个多 pass AIV kernel，按 level 顺序运行：

```text
for level = 0 .. L-1:
    rows = levelRowIdx[levelRowPtr[level] : levelRowPtr[level+1]]
    每个 core 处理 rows 的一个区间
    for each row i and RHS tile:
        acc = alpha * opB(B)[i, tile]
        acc -= Σ A[i,j] * X[j, tile]
        if NON_UNIT: acc *= invDiag[i]
        写回 opB(C)[i, tile]
    PipeBarrier(MTE3)
    SyncAll()
```

FP32 走标量乘/向量减；complex64 使用复数 Muls/Sub 或等价的实虚分量 FMA。B/C alias 时，依赖行只在其 level 完成后被后续 level 读取，因此原地求解安全；同 level 不存在相互依赖。

### Dense layout transpose kernel

内部求解统一使用 ROW 连续 scratch：

- B 为 COL 或 opB=T/H 时，`transpose_in` 将物理 B 转换成逻辑 `opB(B)` 的 ROW scratch，并在 complex H 路执行共轭；
- 求解完成后，`transpose_out` 将 ROW scratch 按 C 的 order 和 opB 反变换写回 C；
- B/C 均为 ROW 且 opB=N 时跳过 scratch；B/C 共用指针时，若需要转换先把 B 复制到 scratch，避免覆盖未读 RHS。

每个转置 kernel 按行切分，使用 DataCopyPad 处理任意 leading dimension 和尾块，跨 kernel 依靠同一 stream 的提交顺序建立依赖。

### 异步和生命周期

BufferSize/Analysis 可以在 Host 上完成参数和状态处理；Solve、UpdateMatrix、格式归一化、level metadata、transpose 和求解 kernel 都在 handle stream 上提交。Solve/UpdateMatrix 不调用 `aclrtSynchronizeStream`。调用方必须在读取 C、释放 buffer、释放 values、Destroy 描述符前同步 stream。

## 工程文件规划

| 路径 | 设计内容 |
| --- | --- |
| `include/cann_ops_sparse.h` | `aclsparseSpSMUpdate_t`、`aclsparseSpSMUpdateMatrix` 及能力注释 |
| `sparse/spsm/arch35/spsm_host.cpp` | 参数校验、阶段状态、workspace、descriptor signature、kernel 调度 |
| `sparse/spsm/arch35/spsm.h` | opaque descriptor、tiling 和 workspace offset |
| `sparse/spsm/arch35/spsm_tiling_data.h` | format/op/dtype/layout/level/RHS tile 元数据 |
| `sparse/spsm/arch35/spsm_kernel.cpp/.h` | normalize、analysis、solve、transpose、update kernels |
| `sparse/spsm/README.md` | 支持规格、接口、调用流程、限制和示例 |
| `test/spsm/arch35/` | C++ UT/ST、golden、pointer/alias/update/异常用例 |
| `test_cases/aclsparseSpSM_testCase/` | ATK 精度、性能、NPU/GPU benchmark 和复现脚本 |

## 支持硬件

| 支持芯片 | 结论 |
| --- | --- |
| Ascend 950PR / A5（arch35/DAV_3510） | √，本任务主验收平台 |
| 其他产品线 | 由公共 Host 能力和对应 arch 适配情况决定，本任务不承诺性能 |

运行环境：CANN 9.1.0 及任务验收指定的后续配套版本；测试报告记录实际 CANN、驱动、SOC、设备号和 Profiler 版本。

## 算子约束限制

1. A 必须是 `m×m` 三角矩阵；B/C 的物理形状必须按 opB 与 RHS 数匹配，ROW/COL 的 ld 满足布局约束。
2. 只支持 I32 row pointer/indices、base ZERO/ONE；I64 返回 NOT_SUPPORTED。
3. 只支持 FP32 或 complex64 一致类型组合，FP16/BF16/FP64/混合 dtype 返回 NOT_SUPPORTED。
4. 只支持 DEFAULT algorithm；本任务不扩展用户自定义 SpSM algorithm。
5. NON_UNIT 必须有显式对角且求解时非零；UNIT 使用隐式 1，不能通过 DIAGONAL 更新改写。
6. values、B/C 在 BufferSize/Analysis 可为空，但 Solve 及 UpdateMatrix 的实际 Device values 必须有效；alpha 必须符合 pointer mode。
7. 不支持在 Analysis 和 Solve 之间改变 pattern、shape、layout、dtype、op、fill/diag 或 workspace；需要改变这些属性必须销毁并重新 Create/BufferSize/Analysis。
8. 输出为异步写回，调用方负责 stream 同步；buffer、values、descriptor 的生命周期必须覆盖异步操作。
9. 计算结果不保证跨不同实现的 bit-exact，但在同一输入、同一设备和同一算法路径上必须 bitwise deterministic；测试使用 FP64/complex128 golden 验证数值误差。

# 可维可测分析

## 精度标准

CPU golden 使用 FP64（FP32 输入）或 complex128（complex64 输入）的 `torch.linalg.solve_triangular`，并严格模拟 format/base、fill/diag、opA/opB、alpha、ROW/COL 和 UpdateMatrix。

逐元素判定：

```text
|actual - golden| ≤ atol + rtol × |golden|
```

FP32 以及 complex64 的 real/imag 分量分别使用：`rtol=2^-10`、`atol=2^-16`、整体 matched ratio≥0.99，且每元素 `max_abs_error≤1e-2` 或 `32×ULP(golden)`。Inf/NaN 按测试框架规则单独判定；重复运行的输出和 canonical metadata 做 bitwise 比对。

## 性能标准

性能计时覆盖任务要求的真实 Analysis/Update/Solve hook，禁止 reference fallback 作为性能结论。性能脚本默认 warmup 10 次、正式采样 30 次，报告 median、P90、mean、Analysis、Update、Solve、workspace 和 profiler kernel 时间。

性能倍率定义：

```text
性能倍率 = 标杆接口 GPU device Event 调用耗时 / NPU 同调用范围总耗时
```

所有声明 dtype 和格式场景达到 0.3 倍标杆以上。任务书 P 场景如下，精确基线以 `test_cases/aclsparseSpSM_testCase/gpu_performance_result_benchmark.md` 及 `baseline_results/` 为准：

| 场景 | m / nnz / RHS | dtype | opA / opB | fill / diag | 任务书标杆 median 范围 |
| --- | ---: | --- | --- | --- | ---: |
| P-01 | 32768 / 262144 / 16 | FP32 | N / N | lower / nonunit | 56341.984–57962.883 μs |
| P-02 | 65536 / 786432 / 32 | complex64 | T / T | upper / unit | 152160.188–152243.891 μs |
| P-03 | 131072 / 1966080 / 64 | complex64 | H / N | lower / nonunit | 304306.812–305923.281 μs |

每个 P 场景同时覆盖 base=0 和 base=1；额外性能用例覆盖 diagonal、banded、highly_imbalanced、one_long_row、skewed、ROW/COL、N/T/H、GENERAL/DIAGONAL update 和 RHS=1/多 RHS。

## 内存标准

当输入输出总量超过 500 MB 时，在等价 API 调用范围下，NPU 相对 GPU 的额外峰值内存不超过 GPU 使用内存总量的 50%；没有等价 GPU 接口的场景使用 workspace 绝对值不超过目标硬件 L2 Cache 的规则。workspace 由 BufferSize 返回，Analysis 绑定后保持有效，Solve 完成前不得释放。

内存数据使用 `collect_sparse_ops_gpu_memory.py`、`collect_sparse_ops_npu_memory.py` 和 `compare_sparse_ops_memory.py` 采集，报告 `input_baseline_*_bytes`、`peak_*_bytes`、`extra_peak_*_bytes`、workspace bytes 及 L2 容量。

## 测试设计

### C++ UT/ST

测试目录为 `test/spsm/arch35/`，至少覆盖：

| 类别 | 用例 |
| --- | --- |
| 生命周期 | Create、BufferSize、空 values Analysis、SpSM、GENERAL/DIAGONAL UpdateMatrix、Destroy |
| 格式 | CSR/CSC/COO；未排序、重复、空行、长尾和不同稀疏度 |
| 数值 | FP32/complex64；LOWER/UPPER；UNIT/NON_UNIT；N/T/H；alpha 实数/复数 |
| 布局 | B/C ROW/COL、不同 ld、B/C 同指针 in-place、opB=T/H 的物理 shape |
| 索引 | base ZERO/ONE、I32；越界、非单调 row pointer、非法 COO/CSC 指针 |
| 异常 | null handle/descriptor/alpha/values、shape/dtype/op/alg 不匹配、缺失/零对角、过早释放 buffer |
| 确定性 | 同一输入重复 Solve，比较输出和 canonical CSR/level metadata 的 bitwise 一致性 |

### 任务包精度和泛化测试

在 `test_cases/aclsparseSpSM_testCase/` 下：

```bash
cd aclsparseSpSM_testCase
bash run_accuracy_atk.sh generate
bash run_accuracy_atk.sh run
```

`accuracy_cases.json` 的基础组合覆盖 dtype、base、opA/opB、fill、diag、pointer mode、update 和 row pattern；`atk_generalization.yaml` 生成 1000 条泛化用例。NPU executor 必须注册真实 `spsm_analysis_npu`、`spsm_update_npu`、`spsm_npu` hook；`--allow-reference-fallback` 仅允许小规模框架冒烟，不得用于验收结论。

### 性能和 GPU baseline

```bash
CUDA_DEVICE_INDEX=0 bash run_all_gpu_benchmarks.sh
(cd aclsparseSpSM_testCase && python3 benchmark_sparse_ops_npu.py \
    --case-file performance_cases.json --device 0 --output results)
```

GPU 原生 cuSPARSE benchmark 优先调用 `bufferSize/analysis/updateMatrix/solve`；若运行时不具备 update symbol，则记录 `update_supported=false` 和 reanalysis 路径，不能把不同计时范围混为同一倍率。NPU 结果必须记录真实 hook、预热次数、采样次数、median/P90 和 profiler trace。

## 兼容性分析

本设计保持现有 SpSM 三阶段接口和 descriptor 类型不变，只新增 `aclsparseSpSMUpdate_t`/`aclsparseSpSMUpdateMatrix`。既有 CSR/FP32/Host/N/N 路径继续可用；新增能力通过 format、dtype、op、pointer mode 和 update 分支选择实现。公共 Host 校验与 descriptor 状态跨 arch 复用，arch35 kernel 放在 `sparse/spsm/arch35/`；不改变其他稀疏算子 ABI。

## 参考资料

1. 任务书：`9月社区任务-aclsparseSpSM算子开发(950)/aclsparseSpSM_A5_task_doc.md`；
2. 当前 `ops-sparse` SpSM 基线：`sparse/spsm/arch35/`、`include/cann_ops_sparse.h`；
3. 测试说明：`test_cases/README.md`、`test_cases/aclsparseSpSM_testCase/README.md`；
4. cuSPARSE Generic SpSM 官方文档；
5. `ops-sparse`：<https://gitcode.com/cann/ops-sparse>；
6. 昇腾生态算子精度标准：<https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md>；
7. 任务书给出的 Llama 3.1 70B、Qwen3-235B-A22B、DeepSeek-V3 配置和对应性能场景。

## 交付物和风险控制

### 交付物

1. `include/cann_ops_sparse.h` 公共接口声明；
2. `sparse/spsm/arch35/` Host、tiling、normalize/analysis/transpose/solve/update Ascend C Kernel；
3. `sparse/spsm/README.md` 支持规格和调用示例；
4. `test/spsm/arch35/` C++ UT/ST、CPU golden 和异常/alias/update 用例；
5. `test_cases/aclsparseSpSM_testCase/` ATK、性能、内存脚本及 README；
6. 自测报告：用例参数、精度、bitwise deterministic、median/P90、workspace、峰值内存、Profiler 截图和失败项；
7. `ops-sparse` PR，源码提交到 `master`，并邀请 `Ascend-CANN` 账号参与评审。

### 关键风险

- **opA/opB 形状和三角方向**：以 `opB(B)`/`opB(C)` 的 m×RHS 逻辑视图统一处理，转置后翻转 fill，使用 N/T/H 组合专项用例验证；
- **complex H 共轭**：在 canonicalization/transpose-in 阶段只对 complex64 共轭一次，避免 solve 阶段重复共轭；
- **重复坐标确定性**：排序 key 必须包含原始位置，reduce 顺序固定，禁止依赖非确定性 atomic add；
- **in-place**：B/C alias 且 layout 不同必须走 scratch，确保未读 RHS 不被提前覆盖；
- **UpdateMatrix 生命周期**：Update、Solve、buffer 释放在同一 stream 上建立顺序，描述符在异步完成前不得销毁；
- **大规模 workspace**：所有字节数用 64 位计算并检查溢出，超容量在 Host 返回，不让 kernel 进入越界路径；
- **性能计时口径**：只比较同一 API 阶段范围，记录 update 支持情况和 profiler 证据，禁止使用 CPU/reference fallback 代替 NPU 性能。


# 2026-09-13 实现状态补充



## 2026-09-13：CSC/COO 与 FP32 确定性 NPU 规范化

- 基点 `1fe0134`，分支 `feature/aclsparse-spsm-950`。
- 新增 `sparse/spsm/arch35/spsm_analysis_kernel.cpp`；修改 Host/tiling/workspace/solve dispatch、C++ 专项和 base-1 Golden 数据生成器。
- Analysis 在 NPU 执行：索引校验 → 行计数/前缀和/按输入序 scatter → 每行稳定插入排序/稳定重复值累加 → 紧凑 CSR → 对角和 level scheduling。Host 仅回收 4 个 int32 状态，无同规模 CPU 矩阵缓存，移除原 CPU 转置/level scheduling 函数。
- CSC/COO/CSR 内部统一 base 0；opA=T 在 NPU 坐标映射时交换行列。CSR 同样规范化，以满足重复对角及未排序输入语义；零额外转换 fast path 尚待性能优化。
- workspace 保存固定 tiling header、canonical CSR 和 O(m+nnz) scratch。Solve 继续使用原 AIV 向量路径，Device alpha 功能保留。
- 失败与修复：首轮 `20260913_094512` 83 个失败，SIMT by-value 大结构引发 DCache 栈访问异常；改为 workspace 元数据后，`20260913_094842` 仅余 4 个旧 base-1 用例失败。旧生成器只对 colInd 加 1，修正 rowOffsets 同步加 1 后通过。严格拒绝不一致 base 的输入，不放宽校验绕过错误。
- 完整开发回归：`test_cases/remote_results/20260913_095310/remote.log`，118/118 通过，编译/测试退出码 0；原 114 个用例全部保留。
- 新专项：3 格式 × 2 base × N/T，FP64 dense Golden 对照；每组 5 次 Analysis/Solve，canonical pattern、values、输出 bitwise 一致；18 组空矩阵/空行/越界索引组合通过。
- 防错修正：DIAGONAL UpdateMatrix 暂返回 NOT_SUPPORTED，不再将长度 m 的数组误当成 nnz 个 values；完整更新实现待下一里程碑。
- 环境：Ascend950PR / CANN 9.0.0-beta.2 / npu-smi 25.7.rc1。
- Profiler：`test_cases/remote_results/20260913_095310/profile/`，原始 archive、op_summary CSV 和 profiler_summary.json 已保存。555 次 Analysis kernel、111 次 Solve kernel、70 次 transpose kernel，共 736 个任务；采样下专项 4/4 通过。
- Profiler 环境问题：系统 python 链接到 /opt/buildtools/python-3.12.9，动态库不匹配导致 _ctypes 导出失败；在采样命令中配置配套 lib 路径后 CSV 导出成功。未改全局 Python 安装。
- 性能专项：`test_cases/remote_results/20260913_095857/remote.log`，1/1 通过；每种 pointer mode 预热 10 次，30 次采样。以下为 m=16/rhs=8 的工程验证，不替代任务书 P-01～P-03 性能验收。
- `SPSM_PERF pointer_mode=0 m=16 rhs=8 warmup=10 samples=30 workspace_bytes=1920 analysis_wall_median_us=102.309 analysis_wall_p90_us=103.856 solve_event_median_us=175.727 solve_event_p90_us=176.028`
- `SPSM_PERF pointer_mode=1 m=16 rhs=8 warmup=10 samples=30 workspace_bytes=1920 analysis_wall_median_us=101.948 analysis_wall_p90_us=103.686 solve_event_median_us=175.65 solve_event_p90_us=176.204`

- 静态检查：git diff --check 通过。峰值内存、P-01～P-03 性能和 CANN 9.1+ 验收未完成。
- 状态：CSC/COO 及 FP32 确定性规范化的功能、基线、专项与 NPU profiler 证据齐全；complex64 尚未实现，不能标记全 dtype 确定性或最终交付完成。


## 2026-09-17 CANN 9.2 实现与验收补充

以下为当前实现，历史阶段记录不代表最新能力边界：CSR/CSC/COO、FP32 N/T、complex64 N/T/H、Host/Device alpha、ROW/COL/leading dimension、alias、GENERAL/DIAGONAL 和确定性规范化均已实现。规范化/拓扑分层由 NPU 执行，Host 只回收 16 字节状态。tiling 在 Analysis 前写入 workspace+32 供 SIMT 使用，向量 Solve 使用缓存字段。

GENERAL 更新优化在 descriptor 中增加 canonicalNnz。若输入为 CSR/CSC 且 canonicalNnz==nnz，Analysis 已证明无重复坐标：第一个 NPU kernel 按原压缩维分工，用 canonical 行内二分定位唯一目标、写 values 和对角；第二个 NPU kernel 汇总 scratch 中的错误标志。pattern/levels 不变，复用 Analysis scratch，无新增 workspace。转置/共轭和 base 使用 workspace 中保存的原输入元数据。COO/重复坐标仍提交完整确定性规范化，保证原输入顺序累加。

Update 返回前同步刷新状态；Solve 保持异步。GENERAL 失败使 descriptor 未分析，DIAGONAL 验证失败保留旧 snapshot。参数中没有长度信息，newValues 逻辑长度和已释放任意指针不能由本接口可靠检测，调用者负责容量与生命周期。

Ascend950PR / CANN9.2.0-beta.2 已编译验证；详细用例、性能与版本证据见同目录 `aclsparseSpSM_CANN9.2_self_test_report.md`。首轮含大型项 127/127，优化后常规 126 通过并单独完成六个任务 P 性能用例及大规模 Profiler；六项同范围 update+solve 倍率最低 0.686×，workspace 最大 118,489,600B 小于 128MiB L2。A2/A3、未提供标杆的 dtype 和社区评审尚未完成。
