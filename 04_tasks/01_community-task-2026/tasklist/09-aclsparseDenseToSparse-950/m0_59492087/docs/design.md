# aclsparseDenseToSparse 算子设计文档（Ascend 950PR / A5）

> 交付件①。按 `resources/design_template.md` 组织，四个必需一级章节齐全。
> 代码交付至 `gitcode.com/m0_59492087/ops-sparse` 分支 `feature/densetosparse-complex64-aten`（PR 待提至 `cann/ops-sparse:master`）。

---

# 需求背景（required）

## 需求来源

CANN 社区任务「aclsparseDenseToSparse 算子开发(950)」A5 档。目标是在 `ops-sparse` 仓的 Ascend 950（DAV_3510，`arch35`）实现上，把稠密矩阵到稀疏矩阵的三阶段转换能力补齐到任务书规格：补 complex64、补 Blocked-ELL 非整除尾块、新增 Python/ATen 适配、优化性能并完善测试与文档。

## 背景介绍

### 这不是数值计算算子

`DenseToSparse` 做两件事：**发现结构**（哪些位置非零、有多少）与**搬运位**（把非零元素的原始 bit 模式写到目标格式的 values 里）。它不做任何算术，因此任务书 §3.2 要求 **exact match / bit-wise 一致**，不是混合容差。整套设计的重心是结构语义的边界与确定性：±0 是否入结构、INF/NAN 是否保留、base 0/1 的 offsets 端点、尾块 padding、重复执行逐位可复现。

### ops-sparse 现状分析

`sparse/densetosparse/arch35/` 已有 host/kernel/tiling/UT 骨架。已具备 INT8/FP16/BF16/FP32 × CSR/CSC/COO/Blocked-ELL 能力。三个缺口：complex64 值类型（`GetElementBytes` 返回 0 → NOT_SUPPORTED）、Blocked-ELL 非整除尾块（`ValidateBell` 要求整除）、Python/ATen NPU 适配（仓内完全缺失，torch_npu 2.12 原 `_to_sparse_csr` 在 NPU 上 CPU fallback）。

### 功能分析：三阶段契约决定两遍扫描

`GetBufferSize → Analysis → 分配 payload → SetPointers → Convert → 同步 stream`。Analysis 统计非零、写 CSR/CSC offsets、更新 `matB.nnz`；Convert 按 nnz 与结构写 indices/values。BELL 不扫描、不改容量（workspace=0）。kernel 因此有 count、prefix-scan、convert 三类工作。

# 需求分析（required）

## 需求描述

在 Ascend 950 arch35 上实现 `aclsparseDenseToSparse{GetBufferSize,Analysis,Convert}`，values 支持 INT8/FP16/BF16/FP32/complex64，目标格式 CSR/CSC/COO/Blocked-ELL，索引 I32（CSR/CSC offset 与 index 类型须一致），index base 0/1，ROW/COL 布局。并提供 Python/ATen NPU 适配，禁 CPU fallback，适配 PyTorch 2.7+/torch_npu 26.0.0+。

## 需求拆解

1. 补齐 complex64：nonzero 谓词为「实部或虚部任一非零、或任一分量为 NaN」，bit 搬运。
2. Blocked-ELL 非整除尾块：rows/cols/ellCols 非 blockSize 整除时，越界逻辑元素写正零。
3. Python/ATen NPU 适配：`Tensor.to_sparse*` → `aten::_to_sparse*` NPU 注册，无 fallback。
4. 性能达 0.3× 标杆（CSR/CSC 对标 PyTorch `to_sparse_*`；Blocked-ELL 对标原生 cuSPARSE）。
5. 精度 exact match、内存 workspace≤L2 或额外内存≤50%GPU、C++/端到端 UT 覆盖边界。

# 详细设计（required）

## 算子分析

### 数学公式

无算术。CSR：`crow[i+1]-crow[i]` 为第 i 行非零数；`col_ind`、`values` 按 (row,col) 升序。CSC 同理按列。COO 输出 (row,col) 升序。BELL 按 `task = blockRow*slots + slot`、`valuePos = task*b*b + innerCol*b + innerRow` 提取固定块。

### 支持数据类型

values：INT8 / FP16 / BF16 / FP32 / **complex64（本次补齐）**。不支持 FP64/complex128。索引 I32/I64，CSR/CSC 二者须一致。base 0/1。

### 支持形状

rows×cols ∈ [0, INT32_MAX]（上界由算子强制，描述符创建接受任意 int64、使用时拒）。零维矩阵有效。BELL 几何：blockSize>0；arch35 要求 block 对齐几何，非整除尾块由 hook 零填充处理。

## 算子实现

### Host 侧设计（`sparse/densetosparse/arch35/densetosparse_host.cpp`）

- 参数校验：handle/matA/matB 非空、alg 仅 DEFAULT、format 合法、index type/base 合法、CSR/CSC offset 与 index 类型一致、value 类型支持且 dense 与 sparse 一致、shape 一致且 ≤ INT32_MAX、dense span/ld/overflow、BELL 几何。
- workspace：`ComputeUnitCount` 按 unit 粒度（CSR/CSC 为 major×minor-chunk，COO 采用 CSR 式 row-major chunking）算 unit count；`ComputeWorkspace` 多级前缀和对齐（header 64B + 各级 uint64）。
- tiling：`DenseToSparseTilingData`（rows/cols/ld/nnz/ellBlockSize/ellCols/unitCount/各偏移/format/order/base/index types/elementBytes/numBlocks）。
- launch：`GetLaunchBlocks(work)`，grid = min(AIV 核数, ceil(work/kWorkPerCore))。

### Kernel 侧设计（`sparse/densetosparse/arch35/densetosparse_kernel.cpp`）

SIMT（`kDenseToSparseThreads=256`，`asc_vf_call`），按 elementBytes×indexType 模板分发。

1. `CountUnits`：每 unit 统计非零数。CSR/COO 扫 row-major（major=row, minor=col），CSC 转置（major=col, minor=row）。
2. 多级 `ScanInclusive` + `AddChunkBases`：前缀和（workspace 内，`kScanChunk=1024` 分块）。
3. `WriteOffsets`/`WriteTotal`/`ValidateOffsets`：CSR/CSC offsets 与一致性校验（Device 端校验，stream 异步）。
4. `ConvertUnits`：按 prefix 写 indices/values；COO 写 row+col 两个索引。
5. `ConvertBell`：按固定 pattern 提取块值，padding 写正零。
6. `IsNonzero<T>`：fp16/bf16 用 `bits & 0x7FFF`，fp32 用 `& 0x7FFFFFFF`（清符号位判 ±0，保留 INF/NaN）。

### complex64：非零谓词统一成「掩符号位后看剩余位」

complex64 视为两个连续 fp32 通道。`IsNonzero<uint64_t>`：`(bits & 0x7FFFFFFF) != 0 || ((bits>>32) & 0x7FFFFFFF) != 0`。即任一通道非 ±0（覆盖有限非零/Inf/NaN payload）即保留，匹配「实部或虚部任一非零或为 NaN」。dispatch `elementBytes==8`（count/convert/bell × I32/I64）。Golden `DenseToSparseIsZero` width==8 对齐。

### Blocked-ELL 非整除尾块（R3，风险最高项）

arch35 `ValidateBell` 要求 `rows%block==0 && cols%block==0 && ellCols%block==0`，kernel `ConvertBell` 不处理尾块。本设计**不改 arch35**：在 Python/ATen hook 侧把 dense 零填充到 `ceil(rows/block)*block × ceil(cols/block)*block`，aclsparse 在填充 dense 上跑，padding 区=0 ⇒ 越界元素自然得 +0，与 `_reference` 的 ceil + 尾块截断完全一致。arch35 几何仍满足整除。

### Python/ATen 适配（`python/ops_sparse_test/`）

torch C++ 扩展，链 `libops_sparse.so + libascendcl.so + libtorch_npu.so`，`torch.ops.load_library` 加载。

- 测试 hook `torch.ops.ops_sparse_test.dense_to_sparse_npu(format, dense, base, layout, block)`：四格式，返回与 `operator_adapter._reference` 一致的规范元组。
- ATen dispatch：`TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)` 注册 `_to_sparse_csr/_to_sparse_csc/_to_sparse`，使 `Tensor.to_sparse*()` 在 NPU 运行、不再 fallback。CSC 用 `at::empty(layout=kSparseCsc)` + `SparseCsrTensorImpl::set_member_tensors` 构造（`_sparse_csr_tensor_unsafe` 只接 SparseCsr）；COO 索引转 int64 后 `sparse_coo_tensor`。
- handle 绑定 `c10_npu::getCurrentNPUStream().stream(false)`。
- BELL pattern 用 torch ops 在 NPU 发现（mask=`(dense!=0).any`，sort+where 建 columns，无 CPU fallback）。
- **stream 同步坑**：BELL pattern-discovery 的 torch_npu ops 与 aclsparse 跨 stream，`aclrtSynchronizeDevice` 不覆盖 torch_npu ops，须 `getCurrentNPUStream().synchronize()` 在 Convert 前后调用。

### 性能设计：两个标杆对应两种瓶颈

- CSR/CSC/COO 对标 PyTorch `to_sparse_*`（较慢，宽松）。
- Blocked-ELL 对标原生 cuSPARSE ~35μs（紧，NPU 需 ≤115μs）。

调优（本次）：`kWorkPerCore` 4096→1（填满 56 向量核，原 P-02 仅 1 核）、`kMajorChunk` 4096→256（提高 unitCount 充分用核）、COO 由线性 tile + 每元素 `pos/cols` 除法改为 CSR 式 major/minor 嵌套（去除每元素除法）。结果（中位 μs）：P-02/P-03 CSR 0.70/0.70、CSC 1.16/1.06（达标），COO 0.24/0.20（未达）。COO 受限因 PyTorch COO 基准比 CSR 快约 3×、而本算子 COO 与 CSR 同量 scan；达标需将标量 SIMT 逐元素读改为 AscendC 向量化（DataCopy + 向量 mask-replace），属后续工作。

### workspace 与生命周期

CSR/CSC/COO workspace 为 per-unit uint64 前缀（P-02 ~192KB、P-01 ~7.3MB），≪ L2(128MB)。BELL workspace=0。Analysis 与 Convert 可复用同一 workspace。核心路径不开与输入/输出成比例的额外 Host 缓冲。

## 支持硬件

Ascend 950PR / 950DT（arch35 / dav-3510）。CANN 9.1.0+。

## 算子约束限制

- values 仅 INT8/FP16/BF16/FP32/complex64；dense 与 sparse value 类型须一致。
- CSR/CSC offset 与 element index 类型须一致（I32/I32 或 I64/I64，混合返回 NOT_SUPPORTED）。
- rows/cols/nnz ∈ [0, INT32_MAX]。
- BELL 几何须 block 对齐（非整除由 hook 填充处理）。
- 仅 `ACL_SPARSE_DENSETOSPARSE_ALG_DEFAULT`。

# 可维可测分析

## 精度标准/性能标准

- 精度：exact match，complex64 实/虚 bit-wise。CPU Golden = `operator_adapter._reference` 与 `densetosparse_golden.h`。
- 性能：倍率 = GPU_median/NPU_median ≥ 0.3。
- 内存：workspace ≤ L2 或额外内存 ≤ 50%GPU。

## 测试策略

- C++ UT（`test/densetosparse/`，CSV 驱动 122 例）：覆盖公开入口/参数/dtype/shape/layout/stride/device/异常/三阶段状态/pointer rebinding/零维/int32max/overflow。
- Python 端到端（`test_cases` 脚手架）：200 ATK accuracy + 320 performance 用例，对 CPU Golden bit-exact。
- NPU Dispatch/Profiler：`to_sparse_csr/csc` 在 NPU 运行、0 fallback（日志 0 条）。

精度结果：C++ UT 122/122、perf 320/320、ATK 200/200，合计 642/642 全过。

## 兼容性分析

- 描述符创建：`aclsparseCreateDnMat` 允许零维（rows==0/cols==0），[0,INT32_MAX] 上界由算子强制；不影响其他算子（零维本就应合法）。
- ATen 注册仅影响 NPU(PrivateUse1) backend 的 `_to_sparse*`，不改变 CPU/CUDA 行为。
- arch35 BELL 几何约束未放宽（hook 侧处理尾块），不破坏既有 BELL 契约。

# 风险点

1. **COO/Blocked-ELL 性能未达 0.3×**：已实测 SIMT warp 协作 coalesced 读（变慢 2.4×，AIV SIMT per-thread 读不 coalesce，证伪）；唯一路径是 AscendC `DataCopy` 向量级框架重写（TPipe/Que/LocalTensor + tiling host + op 类），等同重写算子，多天高风险，convert 的 stream-compaction 尤难。CSR/CSC 已达标，COO 差 ~1.2×；不影响精度与功能。建议评审后单独立项。
2. **计时口径矛盾**：任务书 §3.3 标杆区间与 `gpu_performance_result_benchmark.md` 实测对不上（C++ 差约 2×、PyTorch 区间不重合），验收前需确认。
3. **torch_npu 版本**：任务书要 26.0.0+，公共镜像最高 2.12.0（cp312）；26.0.0 可能是配套 CANN9.1 的较新商业版未上镜像，当前用 2.12.0 smoke-test 通过。
4. **BELL pattern 发现开销**：hook 的 torch ops + stream sync 有 host 开销，影响 Blocked-ELL 紧门槛；待向量化后用 event-based sync 优化。

# 希望检视人员了解

- complex64 的 nonzero 谓词采用「两 fp32 通道分别掩符号位判 ±0」，与 cuSPARSE 一致，需确认 payload-NaN 保留语义符合预期。
- BELL 非整除尾块用 hook 零填充而非改 arch35 kernel，需确认该方案符合「arch35 BELL 整除约束保持不变、尾块在适配层处理」的验收口径。
- 性能：CSR/CSC 达标，COO/BELL 待向量化；需确认是否接受分阶段达标。
