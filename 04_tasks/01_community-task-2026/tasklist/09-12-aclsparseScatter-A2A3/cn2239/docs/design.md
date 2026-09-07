# aclsparseScatter 设计文档

## 1. 算子接口

### 1.1 aclsparse C++ 接口（公开接口）

```C
aclsparseStatus_t aclsparseScatter(
    aclsparseHandle_t          handle,   // aclsparse 句柄，携带 stream
    aclsparseConstSpVecDescr_t vecX,     // 稀疏输入向量描述符（只读）
    aclsparseDnVecDescr_t      vecY);    // 稠密输出向量描述符（原地写入）
```

接口原型以 `include/cann_ops_sparse.h` 为准，对标 cuSPARSE `cusparseScatter` Generic API。

### 1.2 参数说明

| 参数名 | 输入/输出 | 类型 | 描述 | 约束 |
|--------|----------|------|------|------|
| handle | 输入 | aclsparseHandle_t | 库上下文句柄 | 非空，须先 `aclsparseSetStream` 绑定 stream |
| vecX | 输入 | aclsparseConstSpVecDescr_t | 稀疏向量描述符，含 size/nnz/indices/values/idxType/idxBase/valueType | 非空；nnz <= size <= vecY.nums；数据指针为 Device 内存 |
| vecY | 输入输出 | aclsparseDnVecDescr_t | 稠密向量描述符，values 被原地散布写入 | 非空；valueType 须与 vecX.valueType 一致 |

### 1.3 功能语义（任务书 §2.1）

$$Y[X.indices[i] - idxBase] = X.values[i], \quad i \in [0, nnz)$$

- `nnz == 0`：成功返回且不修改 Y（Host 端跳过 kernel launch）
- 重复索引：last-write-wins，**行为不确定**，不累加，不承诺确定顺序（与 cuSPARSE 一致）
- 无重复索引：输出 bit-wise 一致（纯数据搬运，无算术运算）
- vecX（values/indices）只读，接口不修改
- 未被写入的 Y 元素保持原值不变

### 1.4 支持的数据类型

**值类型（vecX.valueType = vecY.valueType）**：

| aclDataType | 搬运视图类型 | 字节宽度 |
|-------------|------------|---------|
| ACL_INT8 | int8_t | 1 |
| ACL_FLOAT16 | uint16_t | 2 |
| ACL_BF16 | uint16_t | 2 |
| ACL_FLOAT | uint32_t | 4 |
| ACL_COMPLEX64 | uint64_t | 8 |

**索引类型**：ACL_SPARSE_INDEX_32I (int32_t) / ACL_SPARSE_INDEX_64I (int64_t)

**索引基址**：ACL_SPARSE_INDEX_BASE_ZERO (0) / ACL_SPARSE_INDEX_BASE_ONE (1)

**bit-copy 说明**：scatter 为纯数据搬运算子，无任何数值计算。fp16/bf16 以 uint16_t、complex64 以 uint64_t 整字搬运，不做任何 float 中转或升精度，按 valBytes ∈ {1,2,4,8} 分发到 4 个模板实例，bit 级保真。

### 1.5 异常处理与返回值

| 返回值 | 触发条件 |
|--------|---------|
| ACL_SPARSE_STATUS_SUCCESS | 执行成功（含 nnz=0 快速返回） |
| ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR | handle 为 nullptr |
| ACL_SPARSE_STATUS_INVALID_VALUE | vecX/vecY 为 nullptr；idxBase 枚举非法；vecX.size > vecY.nums；vecX.nnz > vecX.size；nnz>0 时 indices/values/vecY.values 为 nullptr；stream 未绑定 |
| ACL_SPARSE_STATUS_NOT_SUPPORTED | valueType 不在支持范围；idxType 非 I32/I64；valueType 不一致；nnz > UINT32_MAX |
| ACL_SPARSE_STATUS_INTERNAL_ERROR | GetAivCoreCount() 返回 0（设备异常） |

---

## 2. 计算逻辑

### 2.1 算法描述

本算子为 1D 索引散布（scatter）。核心设计决策：

**并行策略：按稀疏输入分核（input-segmented）**。每个核负责 nnz 中的一段连续区间 `[i0, i1)`，每个稀疏元素只读一次、只写一次，单核复杂度 O(nnz/coreNum)。

与"按输出 Y 分段"方案（每核重扫全部 nnz）相比：
- 总工作量 O(nnz) vs O(nnz × coreNum)，大核数下优势显著
- 代价是不同核可能写同一 Y 位置（重复索引时），但任务书要求重复索引行为不确定，无需保序
- 无重复索引时各核写地址天然不交叉，bit-wise 一致

**写出路径：MTE3（DataCopyPad）**，非标量 `GlobalTensor::SetValue`。原因：标量 GM 写经过标量数据 cache 按 cacheline 粒度回刷，两个核写同一 cacheline 的相邻元素会互相覆盖；MTE3 绕过该 cache 且只提交请求的字节数，不同核写同一 32B block 内的不同元素是安全的。

### 2.2 三层结构

```
Layer 1: ScatterCompute<ValT, IdxT>  —— 单核计算逻辑
Layer 2: __global__ scatter_kernel   —— 按 idxType/valBytes 分发
Layer 3: scatter_kernel_do           —— <<<>>> 异步 launch
```

### 2.3 伪代码（AscendC API 级）

```
ScatterCompute<ValT, IdxT>(indices, values, yVec, nnz, size, elemsPerCore, tileLength, idxBase):
    c = GetBlockIdx()
    i0 = c * elemsPerCore
    if i0 >= nnz: return
    i1 = min(i0 + elemsPerCore, nnz)

    idxGm = SetGlobalBuffer(indices, nnz)
    valGm = SetGlobalBuffer(values, nnz)
    yGm   = SetGlobalBuffer(yVec, size)

    # UB 缓冲：idx tile + val tile + scratch slots
    pipe.InitBuffer(idxQue, tileLength * sizeof(IdxT))
    pipe.InitBuffer(valQue, tileLength * sizeof(ValT))
    pipe.InitBuffer(scratchBuf, SCRATCH_SLOTS * 32)

    for tileStart in [i0, i1) step tileLength:
        tileLen = min(tileLength, i1 - tileStart)
        # ---- CopyIn: GM -> UB（DataCopyPad 支持非 32B 对齐尾块）----
        DataCopyPad(idxLocal, idxGm[tileStart], ...)
        DataCopyPad(valLocal, valGm[tileStart], ...)
        EnQue/DeQue 同步

        # ---- Compute: 批量 scratch + MTE3 写出 ----
        for batch in [0, tileLen) step SCRATCH_SLOTS:
            batchLen = min(SCRATCH_SLOTS, tileLen - batch)
            for k in [0, batchLen):
                scratch[k * slotStride] = valLocal[batch + k]  # 标量写 scratch slot
            SetFlag<S_MTE3>; WaitFlag<S_MTE3>     # 标量写落盘后 MTE3 可读
            for k in [0, batchLen):
                j = idxLocal[batch + k] - idxBase
                if 0 <= j < size:
                    DataCopyPad(yGm[j], scratch[k * slotStride], sizeof(ValT))  # MTE3 写出
            SetFlag<MTE3_S>; WaitFlag<MTE3_S>     # MTE3 完成后才可写下一批 scratch
```

### 2.4 实现路径选择

- [x] AscendC Kernel（MTE3 写出 + DataCopyPad 搬运 + 标量 scratch 中转）
- [ ] CATLASS 模板库
- [ ] ACLNN 封装

**选择理由**：1D 随机散布，写入粒度为单元素（1~8B），无矩阵/向量计算，不适合 Cube/Vector 密集计算路径。按稀疏输入分核 + MTE3 逐元素写出是该类随机写的正确做法，避免了标量 GM 写的跨核 cacheline 冲突。

---

## 3. Tiling 策略

### 3.1 Tiling 参数结构体

```cpp
struct ScatterTilingData {
    uint64_t nnz;           // 非零元素数量
    uint64_t size;          // 稠密向量 Y 的元素数（越界索引丢弃判据）
    int64_t  elemsPerCore;  // 每核负责的稀疏元素数
    int64_t  tileLength;    // 单次搬入 UB 的元素数
    int32_t  idxBase;       // 0=ZERO_BASE, 1=ONE_BASE
    int32_t  idxType;       // SCATTER_IDX_I32(0) / SCATTER_IDX_I64(1)
    int32_t  valBytes;      // SCATTER_VAL_BYTES_{1,2,4,8}
    int32_t  reserved;      // 8 字节对齐填充
};
```

Tiling 参数经 `scatter_kernel<<<numBlocks, nullptr, stream>>>(indices, values, yVec, tiling)` 直接传入 kernel。

### 3.2 Block 级 Tiling（核间切分，按稀疏输入分段）

| 参数 | 计算公式 | 说明 |
|------|----------|------|
| aivCoreNum | `GetAivCoreCount()` 运行时获取 | 禁止硬编码 |
| useNumBlocks | `min(aivCoreNum, ceil(nnz / kScatterMinElemsPerCore))` | 每核至少 256 元素，避免小 nnz 调度开销 |
| elemsPerCore | `ceil(nnz / useNumBlocks)` | 每核负责的连续稀疏元素段 |
| 核 c 区间 | `[c*elemsPerCore, min((c+1)*elemsPerCore, nnz))` | 尾核截断 |

**kScatterMinElemsPerCore = 256**：低于此值再增核只增调度开销。

**与"按输出分核"的对比**：按输入分核使总扫描量为 O(nnz)，而非 O(nnz × coreNum)。

### 3.3 UB 级 Tiling（核内切分，按 nnz 方向分 tile）

#### UB 分配表

| Buffer 名称 | 大小（字节） | 用途 |
|------------|-----------|------|
| idxQue | tileLength × sizeof(IdxT) | 索引 tile |
| valQue | tileLength × sizeof(ValT) | 值 tile |
| scratchBuf | SCRATCH_SLOTS × 32 | MTE3 写出中转（64 个 32B 对齐 slot） |

#### tileLength 计算

| 参数 | 值 | 说明 |
|------|-----|------|
| kScatterMaxTileLength | 2048 | 单次搬入上限 |
| tileLength | `min(elemsPerCore, kScatterMaxTileLength)` | 不超过本核剩余元素数 |

#### UB 约束验证

- **最坏情形**（IdxT=int64_t, ValT=uint64_t）：idxQue = 2048×8 = 16KB，valQue = 2048×8 = 16KB，scratch = 64×32 = 2KB，总计 34KB（< 192KB 的 18%）✓
- **DataCopyPad 尾块**：非 32B 对齐时由 padParams 补齐 ✓

#### 精度处理说明

**本算子无升精度需求**：纯 bit-copy 搬运，不做任何数值计算，输出 bit-wise 一致。

---

## 4. Workspace 需求

| 项目 | 大小 | 说明 |
|------|------|------|
| tiling data | ScatterTilingData 结构体 | 经 kernel 形参直接传递，无需 tiling buffer |
| workspace | 0 | 无额外 Device/Host 内存分配 |
| 线性临时内存 | 0 | 符合任务书 §3.4：不得额外分配与输入规模线性相关的临时内存 |

---

## 5. 性能优化

### 5.1 关键优化点

1. **稀疏输入分核**：总工作量 O(nnz) 而非 O(nnz × coreNum)，大核数下显著减少冗余扫描
2. **MTE3 写出**：避免标量 GM 写的跨核 cacheline 冲突，不同核写同一 32B block 内的不同元素安全
3. **scratch slot 批量化**：每 SCRATCH_SLOTS=64 个元素一次 S_MTE3 屏障，而非每元素一次，减少同步开销
4. **早退空闲核**：i0 >= nnz 的核立即返回
5. **最少分核**：每核至少 256 元素，避免小 nnz 的调度开销
6. **DataCopyPad 尾块**：nnz 非 tileLength 整数倍时由 padParams 补齐，无需额外处理

### 5.2 算子特性

- **计算模式**：memory-bound（随机小粒度 GM 写为主）
- **访存模式**：indices/values 顺序读（tile 级）+ y 随机写（单元素，MTE3）
- **并行性**：核间按稀疏输入区间完全并行；核内标量串行
- **性能预估**（P-01, nnz=8192, ~40 核）：每核约 200 元素，tile 搬运 + scratch + MTE3 写出，目标 ≤ 4× GPU median ≈ 340μs ✓

---

## 6. Kernel 端实现要点

### 6.1 Kernel 入口

```cpp
extern "C" __global__ __aicore__ void scatter_kernel(
    GM_ADDR gmIndices, GM_ADDR gmValues, GM_ADDR gmYVec,
    const ScatterTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (tiling.nnz == 0) return;
    // 按 idxType 分发
    if (tiling.idxType == SCATTER_IDX_I32)
        DispatchScatterByValBytes<int32_t>(tiling, gmIndices, gmValues, gmYVec);
    else
        DispatchScatterByValBytes<int64_t>(tiling, gmIndices, gmValues, gmYVec);
}
```

### 6.2 valBytes 分发

```cpp
template <typename IdxT>
void DispatchScatterByValBytes(const ScatterTilingData &tiling, ...) {
    switch (tiling.valBytes) {
        case 1: ScatterCompute<int8_t,    IdxT>(...); break;
        case 2: ScatterCompute<uint16_t,  IdxT>(...); break;
        case 4: ScatterCompute<uint32_t,  IdxT>(...); break;
        case 8: ScatterCompute<uint64_t,  IdxT>(...); break;
    }
}
```

### 6.3 核心计算 ScatterCompute

1. 按 GetBlockIdx() 计算本核区间 [i0, i1)
2. 初始化 TPipe + TQue（idxQue, valQue）+ scratchBuf
3. 外层循环：按 tileLength 分 tile，DataCopyPad 搬入 indices 和 values
4. 内层循环：按 SCRATCH_SLOTS=64 分批
   - 标量写 value 到 32B 对齐 scratch slot
   - S_MTE3 同步后，DataCopyPad 逐元素写 yGm（含越界检查）
   - MTE3_S 同步保护 scratch 不被下批覆盖

### 6.4 Launch 入口

```cpp
extern "C" void scatter_kernel_do(
    GM_ADDR indices, GM_ADDR values, GM_ADDR yVec,
    const ScatterTilingData &tiling, uint32_t numBlocks, void *stream)
{
    scatter_kernel<<<numBlocks, nullptr, stream>>>(indices, values, yVec, tiling);
}
```

---

## 7. Host 端实现要点

### 7.1 调用链

```
aclsparseScatter(handle, vecX, vecY)
  ├── handle nullptr 检查 → ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR
  ├── ValidateScatterParams(vecX, vecY)
  │   ├── ValidateSpVecDescr: nullptr/valueType/idxType/idxBase/nnz<=size/指针
  │   └── ValidateDnVecAndCompatibility: nullptr/valueType一致/size<=nums/指针
  └── LaunchScatterKernel(handle, vecX, vecY)
      ├── stream nullptr 检查
      ├── nnz==0 快速返回
      ├── nnz > UINT32_MAX 检查
      ├── 分核计算: useNumBlocks, elemsPerCore, tileLength
      ├── 填充 ScatterTilingData
      └── scatter_kernel_do(indices, values, yValues, tiling, numBlocks, stream)
```

### 7.2 描述符转换

Host 端通过辅助函数（`scatter.h`）完成 opaque 描述符到内部结构体的转换：
- `ScatterToInternalHandle(handle)` → `aclsparseContext*`
- `ScatterToSpVecInner(vecX)` → `aclsparseSpVecDescr*`（解除 const）
- `ScatterToDnVecInner(vecY)` → `aclsparseDnVecDescr*`

### 7.3 索引值域校验

不做 Host 端索引值合法性校验（Host 无法读 Device 内存），与 cuSPARSE 一致。Kernel 内部对越界索引做丢弃保护（`if (j >= 0 && j < size)`），不会写坏邻近内存。

---

## 8. A2/A3 与 A5 共存设计

| 维度 | arch22 (A2/A3) | arch35 (A5) |
|------|----------------|-------------|
| 硬件能力 | 无 SIMT | 有 SIMT (`__simt_vf__`) |
| 并行策略 | 按稀疏输入分核 + UB tile + MTE3 写出 | SIMT 每线程一个元素 |
| dtype 支持 | FP32/FP16/BF16/INT8/COMPLEX64 | FP32/FP16/BF16 |
| 公共头文件 | `include/cann_ops_sparse.h` 共用 | 同左 |
| 代码目录 | `sparse/scatter/arch22/` | `sparse/scatter/arch35/` |
| 测试目录 | `test/scatter/arch22/` | `test/scatter/arch35/` |

两者共用同一份 `cann_ops_sparse.h` 公开声明与 aclDataType/描述符类型，差异仅体现在各自 Host 的 dtype 白名单和 Kernel 实现中。A2/A3 与 A5 PR 先后合入时，后续 PR 基于已合入版本处理公共 Host 冲突并完成交叉回归。

---

## 9. 文件结构（ops-sparse 仓目录）

### 9.1 公开头文件

- `include/cann_ops_sparse.h` — 新增 `aclsparseScatter` 函数声明

### 9.2 实现代码（arch22）

- `sparse/scatter/arch22/scatter.h` — 描述符转换辅助函数
- `sparse/scatter/arch22/scatter_host.cpp` — 参数校验 + tiling 计算 + kernel launch
- `sparse/scatter/arch22/scatter_kernel.h` — kernel_do 签名声明
- `sparse/scatter/arch22/scatter_kernel.cpp` — ScatterCompute 模板 + 入口 + launch
- `sparse/scatter/arch22/scatter_tiling_data.h` — ScatterTilingData 结构体（Host/Kernel 共用）

### 9.3 测试代码

- `test/scatter/scatter_param.h` — 测试参数定义（公共）
- `test/scatter/scatter_golden.h` — CPU Golden 生成（公共）
- `test/scatter/CMakeLists.txt` — 测试构建
- `test/scatter/arch22/scatter_test.cpp` — C++ UT/ST
- `test/scatter/arch22/scatter_test.csv` — 测试用例参数表
- `test/scatter/arch22/scatter_npu_wrapper.h` — NPU 调用封装

### 9.4 README

- `sparse/scatter/README.md` — 算子接口文档（面向使用者）

---

## 10. 测试验证

### 10.1 C++ UT/ST

| 类别 | 覆盖场景 |
|------|---------|
| 基础功能 | 5 种 dtype (int8/fp16/bf16/fp32/complex64) × I32/I64 × base 0/1；不同 size/nnz 组合 |
| 边界场景 | nnz=0（Y 不变）；nnz=1；nnz=size（全覆盖）；size=1 最小规模；17/129/257 尾块 |
| 索引场景 | 顺序、乱序、重复索引（验证输出来自对应 values 之一） |
| 异常测试 | 空 handle/描述符/指针、dtype/device 不一致、非法 idxType/idxBase、nnz>size |
| 性能规模 | P-01 (size=128256, nnz=8192)、P-02 (size=151936, nnz=4096)、P-03 (size=129280, nnz=7168) |

### 10.2 精度标准

- 无重复索引：逐元素 bit-wise exact match
- 重复索引：验证输出来自对应输入 values 之一
- 校验 vecX 只读（未被修改）和未写入的 vecY 元素保持原值
- 覆盖普通值、小值、正负混合、零、离群值及 INF/NAN

### 10.3 性能目标

| 编号 | 场景 | 目标 |
|------|------|------|
| P-01 | Llama 3.1 70B: size=128256, nnz=8192 | 性能倍率 ≥ 0.25 × GPU 标杆 |
| P-02 | Qwen3-235B: size=151936, nnz=4096 | 性能倍率 ≥ 0.25 × GPU 标杆 |
| P-03 | DeepSeek-V3: size=129280, nnz=7168 | 性能倍率 ≥ 0.25 × GPU 标杆 |

每个有效 case（5 dtype × I32 × base 0/1 = 10 条/场景）均须达标。

### 10.4 内存目标

无 workspace、无额外 Device/Host 内存分配。输入输出总量远小于 500MB（最大 case: size=151936 × 8B + nnz=8192 × 12B ≈ 1.3MB），满足任务书 §3.4 要求。

---

## 11. 参考资料

- **cuSPARSE**：`cusparseScatter` 接口语义
- **ops-sparse 仓库**：https://gitcode.com/cann/ops-sparse
- **Ascend C 开发文档**：https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/opdevg/Ascendcopdevg/atlas_ascendc_map_10_0002.html
- **精度标准**：https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md
- **任务书**：`aclsparseScatter_A2A3_task_doc.md`
