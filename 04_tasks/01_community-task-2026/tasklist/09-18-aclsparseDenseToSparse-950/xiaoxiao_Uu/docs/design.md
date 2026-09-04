# 需求背景（required）

## 需求来源

本设计对应 2026 年 9 月社区任务《aclsparseDenseToSparse 算子开发（950）》（任务编号 09-18）。任务要求在 Ascend 950（arch35 / A5）上完善 `aclsparseDenseToSparse*` 三阶段接口：将 ROW/COL 布局的稠密矩阵转换为 CSR / CSC / COO / Blocked-ELL（BELL）稀疏格式；在已有 INT8/FP16/BF16/FP32 能力基础上补齐 **complex64** 支持；优化性能并完善测试与文档；同时交付 **Python/ATen** 适配（禁止 CPU fallback）。

设计依据优先级：

1. 任务书《aclsparseDenseToSparse 算子开发任务书（A5）》§2–§4（含 PTA 适配与 A2 复测要求）
2. `ops-sparse` 公开头文件 `include/cann_ops_sparse.h`
3. 仓内 `sparse/densetosparse/arch35/` 既有实现与公共描述符语义
4. cuSPARSE Generic DenseToSparse（GetBufferSize / Analysis / Convert）三阶段约定
5. PyTorch `Tensor.to_sparse*` / `aten::_to_sparse*` schema（PTA 接口定义见 `python/npu_sparse_densetosparse/` 与 ATK `function_sparse_ops.py`）

公开接口命名以头文件为准：`GetBufferSize` / `Analysis` / `Convert`（任务书示例若写 `aclsparseDenseToSparseGetBufferSize` 等则同名）。

## 背景介绍

### 算子要做什么

对齐 cuSPARSE 三阶段语义：

1. **GetBufferSize**：查询 Analysis/Convert 共用的 workspace 字节数
2. **Analysis**：扫描 Dense 矩阵的非零结构，更新 `matB` 稀疏描述符元数据（nnz / offsets 端点）
3. 调用方按 nnz 分配 indices/values 存储并调用 `*SetPointers` 绑定
4. **Convert**：写入 indices 与 values（Blocked-ELL 按预置 block-column pattern 抽取）

非零判定规则：

- 浮点：`+0/-0` 不进入稀疏结构；INF/NAN 保留
- complex64：实部或虚部任一为非零/INF/NAN 即保留；仅两侧均为 signed zero 时丢弃
- 输出坐标顺序确定；同一输入重复执行结果 bit-wise 一致；Dense 与 pattern 输入只读

数学上，CSR/CSC/COO 属于“按数值发现结构”的 Dense→Sparse 转换；BELL 属于“按固定 block-column pattern 抽取数值”，不扩展稀疏结构。

### 现状与增量

仓内 `sparse/densetosparse/arch35/` 已有 SIMT 实现（count → inclusive scan → offsets/convert / bell）。本任务增量如下：

| 缺口 | 方案 |
|------|------|
| complex64 | Host 侧 `GetElementBytes=8`；Kernel 侧 `IsNonzero` 按 uint64 处理；Dispatch 增加 8B 分支 |
| Python/ATen | `python/npu_sparse_densetosparse/` ctypes 三阶段；注册 `dense_to_sparse_npu` 与 `_to_sparse*` |
| 任务包验收 | `test/densetosparse/task_cases/`（200 精度用例 + P 场景性能/内存）+ 950 一键脚本 |
| 文档 | README 声明 complex64；本设计文档 |

### PyTorch 映射

| 公开入口 | ATen | aclsparse |
|----------|------|-----------|
| `Tensor.to_sparse_csr/csc` / `to_sparse` / `to_sparse_bsr` | `_to_sparse_csr/_csc/_to_sparse/_to_sparse_bsr` | DenseToSparse 三阶段 |
| 任务包 hook | `ops_sparse_test.dense_to_sparse_npu` | 同上 |

约束：二维 Dense；索引 I32、base 0/1；Analysis/Convert/payload 构造禁止 CPU fallback。

### 与 SparseToDense 关系

| 算子 | 方向 | 阶段 |
|------|------|------|
| DenseToSparse | 稠密 → 稀疏 | GetBufferSize / Analysis / Convert |
| SparseToDense | 稀疏 → 稠密 | bufferSize / execute（对偶 scatter） |

---

# 需求分析（required）

## 需求描述

在 Ascend 950 上交付完整 DenseToSparse：五种 dtype（INT8/FP16/BF16/FP32/COMPLEX64） × 四种格式（CSR/CSC/COO/BELL） × base 0/1 × ROW/COL；精度与参考实现 exact match；性能达到任务书 P 场景 ≥0.3× GPU 标杆；内存满足 IO>500MB 场景相对 GPU 增量 ≤50% 或固有 workspace ≤ L2；交付 C++ ST、ATK、Python UT（PTA）测试。

## 需求拆解

| 编号 | 子项 | 验收要点 |
|------|------|----------|
| R1 | 三阶段 API | GetBufferSize / Analysis / Convert 对齐 cuSPARSE |
| R2 | dtype | INT8/FP16/BF16/FP32/COMPLEX64 |
| R3 | 格式 | CSR/CSC/COO/Blocked-ELL；I32 索引；base 0/1 |
| R4 | 确定性 | nnz/坐标/values bit-wise 可复现 |
| R5 | Python/ATen | dense_to_sparse_npu + aten::_to_sparse*；无 CPU fallback |
| R6 | 精度 | 任务包 200 用例 + C++ CSV（含 c64 SPECIAL 用例） |
| R7 | 性能/内存 | P-01/P-02/P-03 及泛化；≥0.3×；workspace/IO 规则 |

## 输入输出规格

| 参数 | 方向 | 类型 | 说明 |
|------|------|------|------|
| handle | 输入 | aclsparseHandle_t | 携带 stream |
| matA | 输入 | aclsparseConstDnMatDescr_t | 只读 Dense，ROW/COL，ld 合法 |
| matB | 输入/输出 | aclsparseSpMatDescr_t | 目标稀疏；Analysis 更新元数据；Convert 写 payload |
| alg | 输入 | aclsparseDenseToSparseAlg_t | 仅 DEFAULT |
| bufferSize | 输出 | size_t* | Analysis/Convert 共用 workspace |
| buffer | 输入 | void* | 查询为 0 时可空；两阶段可复用同块 |

| value dtype | index | base | format | order |
|-------------|-------|------|--------|-------|
| INT8/FP16/BF16/FP32/COMPLEX64 | I32 | 0/1 | CSR/CSC/COO/BELL | ROW/COL |

---

# 详细设计（required）

## 算子分析

### 数学公式

Dense 矩阵元素判定（保留/丢弃）与坐标映射：

```text
keep(row, col) = (row >= 0 && row < M && col >= 0 && col < N)
                 && IsNonzero(dense[row * ld + col])

CSR:   row_ptr[i] = 前 i 行的非零数；col_ind[ k ] = 列号；values[ k ] = 值
CSC:   col_ptr[j] = 前 j 列的非零数；row_ind[ k ] = 行号；values[ k ] = 值
COO:   row_ind / col_ind / values 按行主序输出全部非零
BELL:  按 (blockRow, blockCol) 预置 pattern 抽取，越界逻辑槽写 +0
```

### 支持数据类型

INT8、FP16、BF16、FP32、COMPLEX64（本任务新增）。

### 支持形状

二维稠密矩阵（M×N），ROW 主序或 COL 主序（ld 合法）；支持 CSR/CSC/COO/BELL 四种稀疏格式；索引类型仅 I32；base 0/1。

### 计算流程

```text
Validate(dtype/format/ld/base/alg)
  → Tiling(unitCount, scan levels, workspace offsets)
  → Analysis: count_kernel → multi-level inclusive scan → WriteOffsets/WriteTotal
  → Host: SpMatGetSize(nnz) → 分配 payload → SetPointers
  → Convert: 复用 count/scan → validate offsets → emit indices/values
  → BELL: 跳过 count/scan，按 ellColInd 抽块写 values（越界槽写 +0）
```

### complex64 非零语义

将每个元素视为 `uint64_t = real_f32 | (imag_f32 << 32)`：

```text
keep = ((real & 0x7fffffff) != 0) || ((imag & 0x7fffffff) != 0)
```

与 FP32 路径一致地丢弃 signed zero，保留 NaN/Inf。

### 编程模型

arch35 SIMT（`asc_vf_call` + grid-stride）。DenseToSparse 为不规则稀疏发现，SIMD/RegBase 难以表达动态前缀和写回，保留 SIMT。线程块规模常量 `kDenseToSparseThreads = 256`。

### Workspace

- Header（status/nnz）+ level0 counts + 多层 scan 缓冲，32B 对齐
- BELL 路径 workspace 可为 0
- 大 IO 场景：任务书 §3.4（IO>500MB 相对 GPU 增量 ≤50%，或固有 workspace ≤ L2）

## Host/Tiling 设计

### Host 侧设计（`densetosparse_host.cpp`）

1. **GetElementBytes**：INT8=1，FP16/BF16=2，FP32=4，COMPLEX64=8
2. **校验**：layout/ld、format、index 类型一致性（CSR/CSC offset 与 index 同宽）、alg=DEFAULT
3. **Tiling**：计算 `unitCount`（COO tile / CSR·CSC major×chunk）与 `numBlocks`；填充 `DenseToSparseTilingData`
4. **Launch**：Analysis/Convert 后按需同步，读 Device status

### Tiling 策略

- 分核：按 `numBlocks`（AI Core 数 × 每核任务数）切分行/列块，grid-stride 循环均分负载
- 分块：`unitCount` 决定单次 scan 块大小；COO 按元素 tile，CSR/CSC 按 major 维 chunk
- tilingkey：以 `format + elementBytes` 组合区分 count/convert/bell 分派路径
- workspace：各段（counts / scan / 输出 offsets）按 32B 对齐连续排布，避免二次分配

关键 tiling 字段：`rows/cols/ld/nnz/ellBlockSize/ellCols/unitCount`、workspace 各段 offset、`format/order/base/elementBytes/numBlocks`。

## Kernel 设计

### 核函数实现

- `IsNonzero` 特化：u8 / u16（fp16·bf16）/ u32（fp32）/ u64（c64）
- `densetosparse_count_kernel` / `convert_kernel` / `bell_kernel` 按 `elementBytes` 分派
- Scan 与 offsets 与 dtype 无关；CSR/CSC 写 offsets 端点；COO 只发布 nnz

### Ascend C 流程

```text
CopyIn:  按 tiling 搬运 Dense 块到 UB / SIMT 直接访存
Compute: count → 多级 inclusive scan → 计算 offsets / 输出坐标
CopyOut: 写 indices / values / offsets 到 Global Memory，32B 对齐批量写回
```

## Python/ATen（`python/npu_sparse_densetosparse/`）

```text
dense_to_sparse_npu(format, dense, base, layout, block_size)
  → ctypes 三阶段 + CreateCsr/Csc/Coo/BlockedEll + SetPointers
  → 返回与任务包一致的 canonical tuple

aten::_to_sparse / _to_sparse_csr / _to_sparse_csc / _to_sparse_bsr
  → PrivateUse1 注册；无 CPU fallback
```

Blocked-ELL：前端先发现非零块列 pattern，再走 ACL 路径（Torch 无完全等价公开 API）。

## 支持硬件

| 芯片 | 勾选 |
|------|------|
| Ascend 950 / 950PR / 950DT | ✅ 本任务验收 |
| Atlas A2/A3 | —（交叉回归时验证公共 Host，不在本设计范围扩 arch22） |

## 算子约束限制

- 仅 DEFAULT 算法；非法 alg → 参数错误
- 索引仅 I32；CSR/CSC offsets 与 indices 同宽
- BELL：pattern 只读；尾块越界逻辑元素写正零
- workspace 仅按 GetBufferSize 分配；核心路径不得另建与 Dense/payload 成比例的 Host 缓冲
- 输入 Dense 与 BELL pattern 只读

---

# 可维可测分析（required）

## 精度标准 / 性能标准

| 验收标准 | 描述 | 标准来源 |
|----------|------|----------|
| 精度 ST | CSV bit-exact + L2 负向用例 | 仓内 GTest |
| 精度 Fallback | 无 ATK schema 同 case | 任务包 |
| 精度 ATK | 200 泛化用例 | 任务包 ATK |
| 性能 | GPU/NPU ≥ **0.3** | 任务书 §3.3 P-01~P-03 |
| 内存 | IO>500MB 增量≤50% 或 workspace≤L2 | 任务书 §3.4 |
| PTA | pytest | 仓内 |

## P-01 / P-02 / P-03 场景定义

| 场景 | Dense shape | 每行非零 | BELL blockSize | 说明 |
|------|-------------|---------|----------------|------|
| P-01 | 8192×28672 | 64 | 16 | Llama 3.1 70B 类 |
| P-02 | 4096×1536 | 64 | 32 | Qwen3-235B 类 |
| P-03 | 7168×2048 | 64 | 64 | DeepSeek-V3 类 |

每场景覆盖五 dtype × 四格式 × base0/1（明细见任务包 `gpu_performance_result_benchmark.md`）。

## 测试与验收方案

| 类别 | 内容 | 位置 |
|------|------|------|
| 精度 ST | CSV L0/L1/WB + COMPLEX64 SPECIAL | `test/densetosparse/arch35/` |
| 负向 | L2 CSV / TEST_F | `densetosparse_l2_cases.csv` |
| 任务包 | ATK 200 + perf + memory | `test/densetosparse/task_cases/` |
| Python | pytest / ATK `function_sparse_ops.py` | `python/npu_sparse_densetosparse/tests/` |
| 950 上传 | `scripts/ci/run_densetosparse_950_upload_logs.sh` → `logs/densetosparse-950` | 私仓 |

## 兼容性分析

- 扩展 arch35 complex64 与既有 FP/INT 分派兼容；与 A2/A3 公共 Host 交叉回归
- `build.sh --soc=ascend950 --ops=densetosparse` 仅编 arch35
- 与 SparseToDense / Scatter / Gather 共用 handle 与描述符体系

## 风险与对策

| 风险 | 对策 |
|------|------|
| c64 非零语义与 Torch nonzero 不一致 | 对齐 cuSPARSE 分量级 signed-zero；SPECIAL ST 锁死 |
| BELL pattern 与 Torch 无等价 API | PTA 自发现 pattern，与任务包 reference 同算法 |
| ATK 插件注册/环境抖动 | Fallback exact 路径兜底；修复 ATK bootstrap 后重跑 |
| 大 shape 性能 | 调 `unitCount`/`numBlocks`；对标 GPU median ≥0.3 |
| 多参与者路径冲突 | 本账号独立目录 `xiaoxiao_Uu/` |

---

# 代码目录（ops-sparse）

```text
sparse/densetosparse/
├── README.md
└── arch35/
    ├── densetosparse_host.cpp
    ├── densetosparse_kernel.cpp
    ├── densetosparse_kernel.h
    └── densetosparse_tiling_data.h

test/densetosparse/
├── densetosparse_param.h / densetosparse_golden.h
├── arch35/          # CSV ST + wrapper
└── task_cases/      # 任务包 ATK / perf / memory

python/npu_sparse_densetosparse/
├── densetosparse_npu.py / acl_binding.py
└── tests/
```

---

# 交付物

1. 本设计文档（本 PR）
2. 代码：`xiaoxiao_Uu/ops-sparse` @ `feature/aclsparse-densetosparse-950`
3. 自测报告（腾讯文档模板）+ 950 日志分支证据
4. 邀请 Ascend-CANN 为开发者
5. PTA/A2 复测结果（A2 复测跑通后将 JSON 贴入 PR 描述第四节）