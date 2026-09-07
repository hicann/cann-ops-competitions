# 【CANN社区任务】aclsparseSparseToDense（A2/A3）算子设计文档

# 需求背景（required）

## 需求来源

社区任务「aclsparseSparseToDense 算子开发（A2/A3）」。参考 cuSPARSE `cusparseSparseToDense_bufferSize` /
`cusparseSparseToDense` 接口语义，使用 C++、Ascend C 和 ops-sparse 算子工程开发 `aclsparseSparseToDense`
的 Atlas A2/A3（DAV_2201，arch22）实现，代码提交至 https://gitcode.com/cann/ops-sparse 。
Python/ATen 适配为必选交付：公开入口 `Tensor.to_dense`，内部映射 `aten::_to_dense`，NPU 上完成适配与执行，
不得 CPU fallback。

## 背景介绍

### aclsparseSparseToDense 算子功能分析

将 CSR / CSC / COO 稀疏矩阵转换为 ROW / COL 布局稠密矩阵：对每个稀疏坐标 `(row,col)` 写入
`B[row,col]=A.values[p]`，未被坐标覆盖的稠密逻辑位置写对应 dtype 正零；输入稀疏结构和 values 只读。

| 参数 | 参数含义 | 类型 | 支持数据类型 | 约束 | 形状 |
| --- | --- | --- | --- | --- | --- |
| handle | 上下文及执行 stream | aclsparseHandle_t | 不涉及 | 非空 | 标量 |
| matA | 只读稀疏源描述符 | aclsparseConstSpMatDescr_t | values: INT8/FP16/BF16/FP32/complex64；索引 I32 | CSR/CSC/COO；base 0/1；offsets 单调；坐标唯一 | rows×cols，nnz 组坐标 |
| matB | 稠密输出描述符 | aclsparseDnMatDescr_t | 与 matA 一致 | ROW/COL；ROW ld≥cols，COL ld≥rows | rows×cols |
| alg | 转换算法 | aclsparseSparseToDenseAlg_t | 仅 DEFAULT | 非法枚举报错 | 标量 |
| bufferSize/buffer | Device workspace 查询/传入 | size_t* / void* | 不涉及 | 按查询值分配 | 标量 |

### 接口定义

```c
typedef enum aclsparseSparseToDenseAlg_t {
    ACL_SPARSE_SPARSETODENSE_ALG_DEFAULT = 0
} aclsparseSparseToDenseAlg_t;

aclsparseStatus_t aclsparseSparseToDense_bufferSize(
    aclsparseHandle_t handle, aclsparseConstSpMatDescr_t matA,
    aclsparseDnMatDescr_t matB, aclsparseSparseToDenseAlg_t alg, size_t *bufferSize);
aclsparseStatus_t aclsparseSparseToDense(
    aclsparseHandle_t handle, aclsparseConstSpMatDescr_t matA,
    aclsparseDnMatDescr_t matB, aclsparseSparseToDenseAlg_t alg, void *buffer);
```

公开接口以 ops-sparse 仓库 `include/cann_ops_sparse.h` 为准（仓库既有声明为 `_bufferSize` 命名）。

# 需求分析（required）

## 需求描述

在 Atlas A2/A3（arch22 / DAV-2201）上实现 `aclsparseSparseToDense`：

1. 功能：CSR/CSC/COO × ROW/COL × base 0/1，values 五种 dtype（INT8/FP16/BF16/FP32/complex64），
   索引 I32；坐标可乱序但必须唯一；转换确定性（同一输入重复执行 bit-wise 一致）；输出所有逻辑
   元素被定义（含 ld padding 边界安全）。
2. 精度：CPU Golden 逐元素 exact match；values bit-wise 保持；complex64 实/虚部分别 bit-wise；
   未覆盖元素为正零；覆盖正负零、INF/NAN、乱序唯一坐标、重复执行确定性。
3. 性能：P-01（Llama3.1-70B MLP 8192×28672，每行 64 nnz）、P-02（Qwen3-235B MoE 4096×1536）、
   P-03（DeepSeek-V3 MoE 7168×2048）三场景共 90 条 case，每 case 倍率（PyTorch GPU Event median
   / NPU 同范围总耗时）≥ 0.25。
4. 内存：IO>500MB 时 NPU 额外内存 ≤ GPU 总量 50%，或 workspace 绝对值 ≤ L2 Cache 容量。
5. Python/ATen：`Tensor.to_dense` / `aten::_to_dense` NPU 注册，不支持组合显式报错，无 CPU fallback。

## 需求拆解

1. 公开 C++ 接口 + Host 校验/分发 + Ascend C Kernel（arch22 目录、tiling、CMake）。
2. 纯数据搬运语义：kernel 只做 `B[idx]=A.val[p]` 位拷贝，无算术无类型转换，天然 bit-exact。
3. 清零与 scatter 均在 handle 绑定 stream 上异步执行（任务书 §2.2）。
4. workspace 仅按 bufferSize 查询值分配，无与稠密输出成比例的额外副本。
5. 自验：C++ UT/ST、ATen UT、Python 端到端 UT、200 条精度泛化用例、90 条性能用例、内存对比。

# 详细设计（required）

## 算子分析

### 数学公式

```
B[row, col] = A.values[p]   若 (row, col) 为第 p 个非零坐标
B[row, col] = +0            未被坐标覆盖的逻辑位置
```

无算术运算，输出与输入同 dtype 位模式搬运。

### 支持数据类型

| values dtype | 枚举 | 元素大小 | 说明 |
| --- | --- | --- | --- |
| float32 | ACL_FLOAT | 4B | |
| float16 | ACL_FLOAT16 | 2B | |
| bfloat16 | ACL_BF16 | 2B | |
| int8 | ACL_INT8 | 1B | |
| complex64 | ACL_COMPLEX64 | 8B | 按 8B 位拷贝，实/虚部逐位保持 |

索引仅 I32（ACL_SPARSE_INDEX_32I）；base 0/1；不支持 FP64。

### 支持形状

- 稀疏侧：CSR（rowOffsets m+1 + colInd nnz）、CSC（colOffsets n+1 + rowInd nnz）、COO（rowInd/colInd nnz）；
  offsets 单调；坐标经 base 换算后在 shape 内；可乱序但唯一。
- 稠密侧：ROW（ld≥cols）/ COL（ld≥rows），支持任意 ld padding；rows/cols/nnz/ld 为 Host int64 元数据，
  上限 INT32_MAX。

## 算子实现

### 实现方案

#### 总体调用链

```
用户代码
 ├─ C++：aclsparseSparseToDense_bufferSize / aclsparseSparseToDense
 └─ Python：Tensor.to_dense() → aten::_to_dense（SparsePrivateUse1 / SparseCsrPrivateUse1 注册）
        ↓
Host（sparse2dense_host.cpp）：参数校验 → 路径选择 → workspace 计算 → 按序发射 kernel
        ↓
Device（sparse2dense_kernel.cpp，AIV_ONLY）：按 format/layout/dtype/规模分流到 5 条路径
```

#### 目录分层

- `include/cann_ops_sparse.h`：公开接口与枚举；
- `sparse/sparse2dense/arch22/`：`sparse2dense_host.cpp`（Host）、`sparse2dense_kernel.cpp/.h`
  （Ascend C Kernel）、`sparse2dense_tiling_data.h`（tiling 结构）；
- `test/sparse2dense/arch22/`：C++ UT（CSV 参数化 + golden 共用 arch35 框架）；
- `python/`：PyTorch/ATen 适配（`csrc/sparse_to_dense_torch.cpp`、`ops_sparse_torch`、`preload`、
  `tests`）。

#### Host 侧设计

**参数校验**（execute 前全量）：空 handle → HANDLE_IS_NULLPTR；空描述符/指针、indexBase 越界、
shape 不一致、ld 非法、nnz>0 但数据指针空、workspace 路径 buffer 空 → INVALID_VALUE；format 非
CSR/CSC/COO、索引非 I32、dtype 不在支持集、matA/matB dtype 不一致、m/n/ld 超 INT32_MAX、
alg 非 DEFAULT → NOT_SUPPORTED。

**1. 分核策略**：`useBlocks = min(AIV 核数, splitDim)`，`perBlock = ceil(splitDim/useBlocks)`；
AIV 核数运行期经 `PlatformAscendCManager::GetCoreNumAiv()` 获取（910B3 40 核 / 910C 48 核自适应）。
splitDim 按路径取：桶式/ContigSweep = 输出外维，TileTranspose = 内维块数，COO = nnz。

**2. 数据分块与 workspace 策略**（`aclsparseSparseToDense_bufferSize` 返回值）：

| 路径 | workspace | 公式 |
| --- | --- | --- |
| 桶式（CSC+行主序大输出） | counts/countsA + groupBase + rowHist + records | `2·nblk·ceil8(nGrp)·4B + (nGrp+1)·4B + nblk·nGrp·32·4B + nnz·8B + nblk·nGrp·32B`，nGrp=⌈m/32⌉，256B 对齐 |
| COO / CSC+row 通用 | 有序性校验 flag | 64B（Host 预清零） |
| 其余 | 无 | 0 |

例：P-01（m=8192、nnz=524288、nGrp=256、nblk=48）≈ 17.3MB，远小于 L2（内存验收准则 2）。

**清零策略**：发射 kernel 时一律由 kernel 自清零（同步版 memset 不保序、异步版实测带宽低）；
仅零维/零 nnz 输出与 64B flag 预清零使用 `aclrtMemsetAsync`，均在 handle 绑定 stream 上有序执行。

**3. TilingKey 规划**：tiling 按值传入（m/n/ld/nnz、valueType、路径/模式位、perBlockCols 等），
kernel 运行期按 format/layout/dtype/规模分流，无编译期 tilingkey 分叉。

#### Kernel 侧设计（arch22 / DAV-2201）

**路径分流**：

| 路径 | 触发条件 | 形态 |
| --- | --- | --- |
| 桶式三 kernel 流水 | CSC+行主序+4B/2B 元素+0<m≤8192+0<nnz≤32M+ld%8==0+dense≥128MB+perBlockCols+1≤2048 | A1 直方图 → A2 记录散布 → B 行组装写回 |
| Path A ContigSweep | CSR+row / CSC+col | 每核独占整行/列，96KB chunk「清零→载入→散写 UB→整块写回」，每字节恰写一次 |
| Path T TileTranspose | CSR+col | (IB×OB) tile 转置，IB=max(1,32B/elem) |
| Path F CscFast | CSC+row 未命中桶式门控 | 块内列 span 驻留 UB；核级 CoreSpanSorted + 块级 SpanSorted 向量化有序性校验，乱序/超容量回退流式 |
| Path C COO | COO | coo_check kernel 字典序校验（GM flag）；有序+行主序走 RowSweep（二分定位+向量预计算），否则 CooSlow |

门控不满足静默回退通用路径，不报错；CSC+行主序且 elemSize≥8（complex64）时 Host 先发射
zero_kernel 清零、主 kernel skipMode 只写 32B 脏块。

**桶式三 kernel 流水**（P-01 CSC 场景核心）：CSC 转行主序稠密本质是大规模转置散布，设计为
「按 32 行一组分桶、两趟直方图计数排序、整组 UB 聚合后连续写回」：

- A1 histogram：每核统计本核列条的行组（row>>5）直方图，产出 counts（精确）/ countsA（对齐）/
  rowHist[nblk][nGrp][32] 三表写 GM（UB：idx 16KB + 计数表 2KB + rowHist 32KB）。
- A2 scatter：向量化累加 countsA 得 (组,核) 子桶基址（core0 顺带写 groupBase 前缀）；重扫列条把
  8B 记录 `w0=(col<<5)|(row&31), w1=val` append 进子桶；UB staging 128KB（64 条/组×256 组），
  满 64 条成对 Set+Wait 同步后 512B DataCopy flush 到 GM records 区。
- B assemble：核 round-robin 领行组；读 rowHist 组切片，**二叉树向量 Add 折叠**（48→24→12→6→3→1，
  dst/src 不重叠）出行直方图；整组 records 一次 DataCopyPad 进 UB，单遍计数排序（8 路展开）进
  sortBuf；逐行「Duplicate 清零 → 8 路散布 → 含 ld padding 整行连续 DataCopy 写回」。组记录超容量
  退化逐行 GM 重扫兜底（正确性路径）。

设计要点：输出每字节只写一次 GM（满带宽连续写），随机散布全部约束在 UB 内。

**有序性校验**：COO 由独立 coo_check kernel 向量化校验（int32 Sub → Cast f32 → Maxs/Mins 截断 →
两级 BlockReduceMin → 标量扫极小值 + 分片边界衔接），违例写 GM flag；CSC 在主 kernel 内做
核级/块级向量化校验，乱序块自动回退流式路径。

**DAV-2201 平台约束与对策**（全部实测，详见 optimization_worklog.md）：

| 约束 | 对策 |
| --- | --- |
| 无向量 Scatter 指令（c220 报 NOT_SUPPORT） | 散布走标量写 UB + DataCopyPad 整块写回 |
| COO 跨核 GM 标量直写丢写 | 输出一律 UB 聚合 + 整块写回，不直写 |
| MTE3 事件 split Set/Wait 必 hang（3 种形态实证） | 事件 Set+Wait 成对紧挨，单缓冲串行 |
| TQue 双缓冲队列开销过大（实测 6229μs） | 裸 TBuf + 手动 flag 同步 |
| 向量指令内原地累加（dstRepStride=0）同地址冒险 | 直方图折叠改二叉树，dst/src 永不重叠 |
| Compare int32 EQ 微码 5.7μs/窗、GatherMask 1.65μs 平底 | 放弃核内压实提取（拉取模型），保留桶式 |
| 驱动 aclrtMemset 带宽低 | kernel 自清零 / zero_kernel |

#### Python/ATen 适配设计

- 公开路径：`Tensor.to_dense()` → `aten::_to_dense`；`TORCH_LIBRARY_IMPL(aten, SparseCsrPrivateUse1)`
  覆盖 CSR/CSC，`TORCH_LIBRARY_IMPL(aten, SparsePrivateUse1)` 覆盖 COO（未 coalesced 先 `coalesce()`
  保求和语义）；同 key 重复注册替换 torch_npu 旧实现（旧实现内部 CPU fallback）。
- 调试入口：`TORCH_LIBRARY(ops_sparse_test)` 的 `sparse_to_dense_npu(...)`，impl 注册 PrivateUse1，
  CPU key 显式注册为报错（无 CPU fallback 的机械保证）。
- 校验：format/layout/base、rows/cols∈[0,INT32_MAX]、三 tensor 同 NPU device、索引 int32/int64
  （int64 转 int32）、numel 一致；dtype 五选一+int32；`_to_dense` 拒绝 masked_grad/BSR/BSC/sparse_dim≠2。
- 输出构造：`rows*effLd` 扁平 buffer + `as_strided` 视图；ld hook 返回 (view, buffer) 供 padding 校验。
- 加载：`ops_sparse_torch/__init__.py` import 即 `torch.ops.load_library`；`preload/sitecustomize.py`
  经 PYTHONPATH 预加载。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas A2 训练系列（910B3/910B4，arch22 / DAV_2201） | √ |
| Atlas A3（910C / Ascend910_938x，arch22 / DAV_2201） | √（实测平台） |

构建：`bash build.sh --soc=<ascend910b3|ascend910_9382> --ops=sparse2dense`（两 target 同为 arch22、
二进制兼容，核数运行期自适应）。**性能测量必须用 `CMAKE_BUILD_TYPE=Release`**：默认 Debug 的 kernel
标量循环几乎不优化，实测慢 ~3 倍（P-01 CSC f32 4338μs vs 1500μs）。

## 算子约束限制

1. 索引仅 I32；m/n/ld ≤ INT32_MAX；不支持 FP64。
2. COO 重复坐标的并行写顺序不确定（与 cuSPARSE last-write-wins 等价）；任务书要求坐标唯一。
3. 桶式路径有门控（m≤8192、nnz≤32M、ld%8==0、dense≥128MB、perBlockCols+1≤2048），不满足静默回退
   通用路径（功能正确，性能不同）。
4. `Tensor.to_dense` 的 masked_grad、BSR/BSC、sparse_dim≠2 显式报错，不 CPU fallback。
5. 重复坐标不做设备端查重（任务书约定唯一坐标）。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | CPU Golden 逐元素 exact match；五 dtype bit-wise，complex64 实/虚部分别 bit-wise；覆盖 CSR/CSC/COO × ROW/COL × base 0/1、ld padding、±0、INF/NAN、乱序唯一坐标、确定性 | 任务书 §3.2 + opbase 精度标准 |
| 性能标准 | P-01/P-02/P-03 每 case 倍率 ≥ 0.25（PyTorch GPU Event median / NPU 同范围总耗时）；预热 ≥10、采样 ≥30 | 任务书 §3.3 |
| 内存标准 | IO>500MB 时 NPU 额外内存 ≤ GPU 50%，或 workspace ≤ L2 容量 | 任务书 §3.4 |

## 测试设计

- C++ UT：`test/sparse2dense/arch22/` 41 条 L0/L1（三格式×5 dtype×base×layout×边界，含 7 条
  complex64，BitwiseGolden 逐位比对）+ 16 条 L2 负例（空指针/维度/类型/ld/format/alg/索引越界）。
- ATen UT + Python E2E：`python/tests/test_sparse_to_dense.py` 9 个 pytest（契约矩阵、空 nnz/空行、
  乱序 COO、ld padding、异常路径、to_dense 公开路径、不支持组合报错、profiler 无 CPU fallback）。
- 精度：200 条 accuracy_cases.json，CPU Golden 逐元素比对（Tensor 生成：70% U[-1,1] / 20% N(0,1) /
  10% 零、边界、INF/NAN）。
- 性能/内存：90 条 P-case + 泛化 case，`benchmark_sparse_ops_npu.py` / `collect/compare 内存三脚本`。

## 验收实测数据（910C / Ascend910_9382，Release 构建，warmup 10 + samples 30，NPU Event）

- 性能：P-01/02/03 共 90 条 **全部达标**（最差倍率 0.49 ≥ 0.25）。最难点 P-01-csc-f32 base0/base1
  = 1506/1508μs（预算 3848/3858，倍率 0.64）；f16 = 1857/1853μs（倍率 ~0.49）。
- 精度：200/200 全过；pytest 9/9。
- 内存：P-01 f32 IO 总量 943.9MB、extra_peak 945.9MB（≈输出+workspace）；最大 workspace 17.3MB（桶式）。
- 优化历程与证伪记录见任务包 `optimization_worklog.md`（续1-续11）。

## 兼容性分析

- 新增算子，与既有 arch35（Ascend 950）实现共用公开头文件与测试框架，kernel/host 按 SOC_ARCH_DIRS
  自动分流，两架构互不影响。
- arch35 不支持 complex64 且零 workspace；arch22 支持 complex64 且按路径返回 workspace，
  差异已在 `sparse/sparse2dense/README.md` 分架构说明。
- 接口与 cuSPARSE 语义对齐，不涉及存量接口变更。
