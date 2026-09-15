# 需求背景（required）

## 需求来源

CANN 社区任务 2026 —— aclsparseGather 算子开发（A2/A3）。

任务书地址：`docs/aclsparseGather_A2A3_task_doc.md`

代码目标仓：<https://gitcode.com/cann/ops-sparse>（master 分支）

## 背景介绍

### aclsparseGather 算子功能概述

aclsparseGather 是 CANN 稀疏算子库（ops-sparse）中的稀疏-稠密向量 Gather 算子，语义对标 cuSPARSE 13.3 Update 1 的 `cusparseGather`。其核心功能为：从稠密向量 Y 中，按稀疏向量 X 的索引数组进行离散取值，写入 X 的值数组（in-place），数学表达式为：

$$
X.\text{values}[i] = Y[X.\text{indices}[i] - \text{idxBase}], \quad i \in [0, \text{nnz})
$$

- **vecY**：一维稠密源向量（只读）
- **vecX**：稀疏向量，含 `indices`（只读）和 `values`（输出，in-place 写入目标）
- **idxBase**：索引基址，0（C 风格）或 1（Fortran 风格）

该算子是大模型推理场景中 vocab embedding 选取（`torch.index_select`）的核心后端，在 Llama 3.1-70B（vocab=128256）、Qwen3-235B（vocab=151936）、DeepSeek-V3（vocab=129280）等模型中被高频调用。

### 当前实现现状

ops-sparse 仓中尚无 aclsparseGather 的 A2/A3（arch22）实现。公共头文件 `include/cann_ops_sparse.h` 已声明接口签名，需要补齐 Host 侧实现、Ascend C Kernel、C++ UT 和 PyTorch ATen 适配层。

# 需求分析（required）

## 需求描述

基于 Ascend C 编程语言，在 Atlas A2（910B3/910B4）和 Atlas A3 训练硬件上实现 aclsparseGather 算子，满足以下核心要求：

1. 实现 C 接口 `aclsparseGather(handle, vecY, vecX)`，语义完全对齐 cuSPARSE
2. 支持 float16、bfloat16、float32、complex64 四种值类型，索引类型为 int32
3. 支持 index base 0 和 1
4. 精度要求：bit-wise exact（位精确匹配）
5. 性能要求：NPU/GPU ratio ≥ 0.25（GPU baseline = A100 PyTorch `torch.index_select`）
6. 不使用额外 workspace，无 CPU fallback
7. 提供 PyTorch ATen 适配层（`torch.index_select` NPU dispatch）

## 需求拆解

1. Host 侧参数校验 + Tiling 计算 + Kernel launch（`sparse/gather/arch22/gather_host.cpp`）
2. Ascend C Device Kernel 开发（`sparse/gather/arch22/gather_kernel.cpp`）
3. TilingData 结构体定义（`sparse/gather/arch22/gather_tiling_data.h`）
4. C++ GTest 单元测试（`test/gather/arch22/gather_test.cpp`）
5. PyTorch ATen NPU Dispatch 适配
6. 精度验证：全 dtype × base 组合，bit-exact 对比
7. 性能验证：24 个 P-case（3 场景 × 4 dtype × 2 base），ratio ≥ 0.25

# 详细设计（required）

## 算子分析

### 数学公式

$$
X.\text{values}[i] = Y[X.\text{indices}[i] - \text{idxBase}], \quad i \in [0, \text{nnz})
$$

其中：
- Y 为 `[size]` 一维稠密向量（只读）
- X.indices 为 `[nnz]` 一维索引数组（只读，int32）
- X.values 为 `[nnz]` 一维值数组（输出，in-place 写入）
- idxBase ∈ {0, 1}

### C 接口定义

```c
aclsparseStatus_t aclsparseGather(
    aclsparseHandle_t          handle,
    aclsparseConstDnVecDescr_t vecY,
    aclsparseSpVecDescr_t      vecX);
```

### 参数说明

| 参数 | 输入/输出 | 含义 | 数据类型 | 约束 |
|------|-----------|------|----------|------|
| handle | 输入 | 稀疏库句柄，持有 stream 和 workspace | aclsparseHandle_t | 非空，已通过 aclsparseCreate 创建，已绑定 stream |
| vecY | 输入 | 稠密源向量描述符（只读） | aclsparseConstDnVecDescr_t | 非空描述符；values 指针非空（nnz>0 时）；dtype ∈ {FP16, BF16, FP32, C64} |
| vecX | 输入+输出 | 稀疏向量描述符，values 为 in-place 输出 | aclsparseSpVecDescr_t | 非空描述符；indices/values 指针非空（nnz>0 时）；idxType=I32；idxBase ∈ {0,1}；valueType 与 vecY 一致 |

**返回值：**

| 返回码 | 含义 |
|--------|------|
| ACL_SPARSE_STATUS_SUCCESS | 执行成功 |
| ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR | handle 为空 |
| ACL_SPARSE_STATUS_INVALID_VALUE | 描述符为空、指针为空、nnz>size、nums<size、idxBase 非法、内存重叠等 |
| ACL_SPARSE_STATUS_NOT_SUPPORTED | 不支持的 dtype、dtype 不一致、索引类型非 I32 |
| ACL_SPARSE_STATUS_INTERNAL_ERROR | 内部错误（如获取核数失败） |

### 支持数据类型

| 值类型（vecY / vecX.values） | 索引类型（vecX.indices） | 索引基（idxBase） | 是否支持 |
|------------------------------|-------------------------|-------------------|----------|
| float16 | int32 | 0 / 1 | ✓ |
| bfloat16 | int32 | 0 / 1 | ✓ |
| float32 | int32 | 0 / 1 | ✓ |
| complex64 | int32 | 0 / 1 | ✓ |

**bit-copy 策略**：Kernel 内部以无符号整型视图处理数据（fp16/bf16 → uint16_t, fp32 → uint32_t, complex64 → uint64_t），避免任何浮点转换链，天然保证 bit-wise exact。

### 支持形状

- vecY：一维稠密向量，shape `[size]`，`size ≥ 0`
- vecX.indices / vecX.values：一维数组，shape `[nnz]`，`nnz ≥ 0`
- 约束：`nnz ≤ size`，`vecY.nums ≥ vecX.size`

## 算子约束与限制

### 参数校验矩阵（Host 侧顺序校验）

| 编号 | 校验项 | 条件 | 返回码 |
|------|--------|------|--------|
| #1 | handle 非空 | handle == nullptr | HANDLE_IS_NULLPTR |
| #2 | 描述符非空 | vecY == nullptr \|\| vecX == nullptr | INVALID_VALUE |
| #3 | 值类型合法 | vecX.valueType ∉ {FP16, BF16, FP32, C64} | NOT_SUPPORTED |
| #4 | 值类型一致 | vecX.valueType ≠ vecY.valueType | NOT_SUPPORTED |
| #5 | 索引类型 | vecX.idxType ≠ I32 | NOT_SUPPORTED |
| #6 | 索引基合法 | vecX.idxBase ∉ {0, 1} | INVALID_VALUE |
| #7 | nnz ≤ size | vecX.nnz > vecX.size | INVALID_VALUE |
| #8 | nums ≥ size | vecY.nums < vecX.size | INVALID_VALUE |
| #9 | 指针非空 | nnz > 0 且 indices/values/yValues 任一为空 | INVALID_VALUE |
| #10 | 无内存重叠 | vecY.values 与 vecX.values 指针区间相交 | INVALID_VALUE |
| #11 | stream 非空 | handle.stream == nullptr | INVALID_VALUE |
| #12 | nnz=0 快速路径 | nnz == 0 → 不 launch kernel，直接返回 SUCCESS | — |

### 数据约束

- 索引值范围：`indices[i] - idxBase ∈ [0, size)`（Device 侧不校验，越界行为未定义，与 cuSPARSE 一致）
- 不支持 int64 索引类型（A2/A3 arch22 限制）
- vecY 与 vecX.indices 保证只读，Kernel 不修改
- vecY 与 vecX.values 不可重叠（Host 侧指针区间检测）
- 不使用额外 workspace（零 workspace 设计）
- 不产生 Host 同步（全异步，仅写入 stream）

## 算子实现

### Host 侧设计（gather_host.cpp）

#### 整体架构

```
aclsparseGather(handle, vecY, vecX)
    ├── #1 handle 非空校验
    ├── ValidateGatherParams(vecY, vecX)    // #2 ~ #10
    └── LaunchGatherKernel(handle, vecY, vecX)
            ├── #11 stream 校验
            ├── #12 nnz=0 快速路径
            ├── Tiling 计算
            └── gather_kernel_do<<<numBlocks, stream>>>
```

#### 描述符解析

从 opaque 描述符指针中提取内部结构体字段：
- `GatherToInternalHandle(handle)` → `aclsparseContext*`（获取 stream）
- `GatherToDnVecInner(vecY)` → `aclsparseDnVecDescr*`（获取 nums, values, valueType）
- `GatherToSpVecInner(vecX)` → `aclsparseSpVecDescr*`（获取 size, nnz, indices, values, idxType, idxBase, valueType）

#### Tiling 策略

以输出 X.values 为切分对象（写侧连续、读侧离散）：

```
valSize    = sizeof(dtype)                          // 2/4/8 字节
nnzBytes   = nnz × valSize                          // 输出总字节数
totalBlocks = ⌈nnzBytes / 32⌉                       // 按 32B 块划分
useNumBlocks = min(AIV 核数, totalBlocks)            // 实际使用核数
blocksPerCore = ⌈totalBlocks / useNumBlocks⌉         // 每核块数
tileLength = min(nnz, 4096)                          // 核内 tile 元素上限
```

**设计要点：**
- 以 32B 块（GATHER_BLOCK_BYTES）为最小粒度不重叠划分给核，保证每核独占写区间，规避多核并发 GM 写共享 cacheline 冲突
- Tiling 全链 int64，无 UINT32_MAX 截断限制
- tileLength 上限 4096：UB 预算 = 4096×(4+32+8) = 176KB / 192KB UB = 91.7%
- 不使用额外 workspace，零额外内存分配

#### TilingData 结构体

```cpp
struct GatherTilingData {
    int64_t nnz;           // 非零元素数量
    int64_t idxBase;       // 索引基址偏移 0 或 1
    int64_t valType;       // 值类型编码 (FP32=0, FP16=1, BF16=2, C64=4)
    int64_t nnzBytes;      // 输出总字节数
    int64_t totalBlocks;   // 32B 块总数
    int64_t blocksPerCore; // 每核块数
    int64_t tileLength;    // 核内 tile 元素上限
};
```

### Kernel 侧设计（gather_kernel.cpp）

#### 编程模型

- 模型：SIMD（AIV 矢量核），`KERNEL_TYPE_AIV_ONLY`
- 多核：输出段 [b0, b1) 按 blockIdx 不重叠分配

#### 多核切分策略

输出 X.values 按 32B 块等分到各核：

```
blockIdx = GetBlockIdx()
b0 = blockIdx × blocksPerCore
b1 = min(b0 + blocksPerCore, totalBlocks)
i0 = b0 × 32 / valSize        // 元素起始
i1 = min(b1 × 32 / valSize, nnz)  // 元素结束
```

空核（b0 ≥ totalBlocks）直接 return。

#### UB 缓冲区预算

| Buffer | 用途 | 大小（tileLength=4096, 最坏 valSize=8） |
|--------|------|----------------------------------------|
| idxQue | 索引 tile 暂存 | 4096 × 4B = 16 KB |
| slotQue | 离散读 32B 槽位 | 4096 × 32B = 128 KB |
| outQue | 密排输出暂存 | 4096 × 8B = 32 KB |
| offBuf | Gather 偏移 ramp（仅 16/32 位） | 4096 × 4B = 16 KB |
| **合计** | | **176 KB / 192 KB UB = 91.7%** |

#### 核内数据流（4 阶段 Pipeline）

每核按 tileLength 迭代处理输出段 [i0, i1)：

**阶段 1：索引 tile 整段搬入**
```
DataCopyPad(idxLocal, idxGm[tileStart], tileLen × 4B)
EnQue / DeQue (MTE2 → S/V 同步)
```

**阶段 2：离散读 Y → 32B 槽位**
```
for e in [0, tileLen):
    j = idxLocal.GetValue(e) - idxBase
    DataCopyPad(slotLocal[e × slotStride], yGm[j], valSize)
EnQue / DeQue (MTE2 → S/V 同步)
```
- src（yGm[j]）仅 1B 对齐（DataCopyPad GM→UB 无 src 对齐要求）
- dst 槽头恒 32B 对齐（slotStride = 32 / valSize）
- 循环后单次 EnQue 覆盖全部 MTE2（FIFO 事件语义）

**阶段 3：密排提取（分两条路径）**

| 路径 | dtype | 实现 | 流水单元 |
|------|-------|------|---------|
| 向量 Gather | fp16/bf16/fp32 (valSize < 8B) | `Gather<T>(outLocal, slotLocal, offLocal, 0, tileLen)` | V 单元 |
| 标量提取 | complex64 (valSize = 8B) | `outLocal.SetValue(e, slotLocal.GetValue(e × slotStride))` 循环 | S 单元 |

**向量 Gather 路径细节：**
- 偏移 ramp：`offLocal[e] = e × 32`（字节偏移，tile 不变量，核内一次构建）
- ramp 由 S 单元写入 → Gather 由 V 单元读取：`SetFlag/WaitFlag<S_V>(0)` 跨流水同步
- Gather 指令约束：dst/src/srcOffset 均 UB 且 32B 对齐，槽头偏移 e×32B 恒满足

**阶段 4：密排 tile 写回 GM**
```
DataCopyPad(xGm[tileStart], outLocal, tileLen × valSize)
```

同步保护：
- 向量路径：outQue EnQue/DeQue (V → MTE3)，FreeTensor 置 MTE3_V 标记
- 标量路径：SetFlag/WaitFlag<S_MTE3>(0) + SetFlag/WaitFlag<MTE3_S>(1)

#### dtype 分发

Kernel 入口按 valType 编码分发 bit-copy 位视图类型：

```cpp
if (valType == FP16 || valType == BF16)  → GatherCore<uint16_t>
else if (valType == FP32)                → GatherCore<uint32_t>
else /* C64 */                           → GatherCore<uint64_t>
```

## 支持硬件

| 芯片型号 | 架构 | 编译目标 | AIV 核数 | 是否支持 |
|----------|------|----------|----------|----------|
| Atlas 800I/T A2 (910B3) | arch22 / DAV-2201 | `--npu-arch=dav-2201` | 40 | ✓ |
| Atlas 800I/T A2 (910B4) | arch22 / DAV-2201 | `--npu-arch=dav-2201` | 40 | ✓ |
| Atlas A3 (910_9382) | arch22 / DAV-C220 | `--npu-soc=Ascend910_9382` | 48 | ✓ |

# 可维可测分析

## 精度标准 / 性能标准

| 验收标准 | 描述 | 标准来源 |
|----------|------|----------|
| 精度标准 | bit-wise exact（位精确匹配）：输出与 golden 逐元素 `torch.equal`，0 字节差异。golden 参考 dtype：fp16→fp32, bf16→fp32, fp32→fp64, c64→complex128 | 任务书 §3.2；精度标准规范 |
| 性能标准 | 24 个 P-case 每个 ratio = GPU_median_us / NPU_kernel_median_us ≥ 0.25。GPU baseline = A100 PyTorch `torch.index_select` Event 计时 | 任务书 §3.3 |
| 内存标准 | 零额外 workspace；当 I/O > 500MB 时，NPU 额外峰值内存 ≤ GPU 峰值 × 50% | 任务书 §3.4 |

### 精度验证方案

- 三层 golden 交叉校验：
  1. **字节级 golden**（GatherGoldenBytes）：以 CPU 端逐字节 memcpy 模拟 gather，输出与 NPU 结果 memcmp
  2. **浮点 Exact 校验**（Verifier EXACT）：以浮点视图逐元素 `==` 比较
  3. **提升链 golden**（GatherGoldenPromoted）：dtype→高精度→gather→截断回原 dtype，验证类型提升无损
- 只读校验：D2H 回读 vecY 和 vecX.indices，验证 Kernel 未修改
- 重复执行：≥4 次连续调用，结果 bit-wise 一致

### 性能验证方案

- 测试方法：aclrtEvent 计时（Event Start → Gather → Event End → SyncStream → ElapsedTime）
- warmup=10, samples=30, 取 median
- 覆盖 3 个 LLM vocab 场景 × 4 dtype × 2 base = 24 P-case

### P-case 性能指标（GPU baseline, A100）

| 场景 | 模型 | size | nnz | GPU median_us 范围 |
|------|------|------|-----|-------------------|
| P-01 | Llama 3.1 70B | 128256 | 8192 | 60.1 ~ 112.1 μs |
| P-02 | Qwen3-235B | 151936 | 4096 | 69.6 ~ 81.9 μs |
| P-03 | DeepSeek-V3 | 129280 | 7168 | 67.8 ~ 76.0 μs |

## 测试用例设计

### 功能正确性用例

| 类别 | 场景 | 用例数 |
|------|------|--------|
| L0 基础 | 4 dtype × {base0, base1} × sorted | 12 |
| L1 扩展 | 不同 size/nnz、乱序/重复索引、不同 seed | 24 |
| L2 边界 | nnz=0、nnz=1、尾块对齐/非对齐 | 5 |
| WB 白盒 | 多核多 tile 大规模 (nnz=196609) | 6 |
| **合计** | | **47** |

### 异常处理用例

| 编号 | 场景 | 预期返回 |
|------|------|----------|
| E-01 | handle 为空 | HANDLE_IS_NULLPTR |
| E-02/03 | vecY/vecX 为空 | INVALID_VALUE |
| E-04 | size > nums | INVALID_VALUE |
| E-05 | nnz > size | INVALID_VALUE |
| E-06 | dtype 不一致 | NOT_SUPPORTED |
| E-07 | 索引类型 I64 | NOT_SUPPORTED |
| E-08 | 不支持的 dtype (FP64) | NOT_SUPPORTED |
| E-09/10 | 非法索引类型/基 | INVALID_VALUE / NOT_SUPPORTED |
| E-11~13 | 索引越界 (zero/one/负数) | 异步错误，行为未定义 |
| E-14~16 | 空指针 (indices/values/yValues) | INVALID_VALUE |
| E-17 | 内存重叠 | INVALID_VALUE |
| E-18 | 跨设备缓冲 | 进程稳定 |
| WB-01~05 | 白盒边界 (nnz>UINT32_MAX, size=0, stream 未设置, 部分重叠, 相邻不重叠) | 各校验分支 |

## 兼容性分析

- 新算子实现，不涉及已有功能兼容性
- 公共头文件 `cann_ops_sparse.h` 中接口签名已预定义，实现不改变 ABI
- 与 cuSPARSE `cusparseGather` 语义对齐，差异项：
  - cuSPARSE 额外支持 FP64/C128/INT8，本实现按任务书要求仅支持 FP16/BF16/FP32/C64
  - cuSPARSE 对重叠场景行为未定义，本实现主动拒绝（Host 侧指针区间检测）
