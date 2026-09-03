# aclsparseDenseToSparse 算子设计文档（Ascend 950 / A5）

| 项目 | 内容 |
|------|------|
| 任务名称 | 9月社区任务-aclsparseDenseToSparse算子开发（950） |
| 参与者账号 | `-7m2FRYfKJXCaXD8P2gMKd1d`（GitCode：`xiaoxiao_Uu`） |
| 目标硬件 | Ascend 950（arch35 / DAV_3510，A5） |
| CANN 版本 | CANN 9.1.0 |
| 交付代码仓 | [xiaoxiao_Uu/ops-sparse](https://gitcode.com/xiaoxiao_Uu/ops-sparse) 分支 `feature/aclsparse-densetosparse-950` → 合入 [cann/ops-sparse](https://gitcode.com/cann/ops-sparse) `master` |
| 设计提交路径 | `04_tasks/01_community-task-2026/tasklist/09-aclsparseDenseToSparse-950/-7m2FRYfKJXCaXD8P2gMKd1d/docs/design.md` |
| 对标接口 | cuSPARSE DenseToSparse（bufferSize / analysis / convert）；PyTorch `Tensor.to_sparse*` / `aten::_to_sparse*` |
| 文档版本 | v1.1 |
| 950 验证 commit | `0a04f103`（日志 stamp `20260902_074307`；CSV/任务包精度与内存已过，ATK 仍在修） |

---

# 需求背景（required）

## 需求来源

本设计对应 2026 年 9 月社区任务《aclsparseDenseToSparse 算子开发（950）》。任务要求在 Ascend 950 上完善 `aclsparseDenseToSparse*` 三阶段接口：将 ROW/COL 稠密矩阵转换为 CSR / CSC / COO / Blocked-ELL；在已有 INT8/FP16/BF16/FP32 能力上补齐 **complex64**，优化性能并完善测试与文档；并交付 **Python/ATen** 适配（禁止 CPU fallback）。

设计依据优先级：

1. 任务书《aclsparseDenseToSparse 算子开发任务书（A5）》§2–§4
2. `ops-sparse` 公开头文件 `include/cann_ops_sparse.h`
3. 仓内 `sparse/densetosparse/arch35/` 实现与公共描述符语义
4. cuSPARSE Generic DenseToSparse 三阶段约定
5. PyTorch 2.7+ `to_sparse*` / `_to_sparse*` schema

公开接口命名以头文件为准：`GetBufferSize` / `Analysis` / `Convert`（任务书示例若写 `aclsparseDenseToSparseGetBufferSize` 等同）。

## 背景介绍

### 算子要做什么

对齐 cuSPARSE：

1. **GetBufferSize**：查询 Analysis/Convert 共用 workspace 字节数
2. **Analysis**：扫描 Dense 非零结构，更新 `matB` 元数据（nnz / offsets 端点）
3. 调用方按 nnz 分配 indices/values 并 `*SetPointers`
4. **Convert**：写入 indices 与 values（Blocked-ELL 按预置 pattern 抽块）

非零判定：

- 浮点：`+0/-0` 不入结构；INF/NAN 保留
- complex64：实部或虚部任一非零/INF/NAN 即保留；仅两侧均为 signed zero 时丢弃
- 输出坐标顺序确定性；同一输入重复执行 bit-wise 一致；Dense/pattern 只读

数学上，CSR/CSC/COO 为“按数值发现结构”的 Dense→Sparse；BELL 为“按固定 block-column pattern 抽值”，不扩结构。

### 现状与增量

仓内 `sparse/densetosparse/arch35/` 已有 SIMT 实现（count → inclusive scan → offsets/convert / bell）。本任务增量：

| 缺口 | 方案 |
|------|------|
| complex64 | Host `GetElementBytes=8`；Kernel `IsNonzero` uint64；Dispatch 增加 8B 分支 |
| Python/ATen | `python/npu_sparse_densetosparse/` ctypes 三阶段；注册 `dense_to_sparse_npu` 与 `_to_sparse*` |
| 任务包验收 | `test/densetosparse/task_cases/`（200 精度 + P 场景性能/内存）+ 950 一键脚本 |
| 文档 | README 声明 complex64；本设计文档 |

### PyTorch 映射

| 公开入口 | ATen | aclsparse |
|----------|------|-----------|
| `Tensor.to_sparse_csr/csc` / `to_sparse` / `to_sparse_bsr` | `_to_sparse_csr/_csc/_to_sparse/_to_sparse_bsr` | DenseToSparse 三阶段 |
| 任务包 hook | `ops_sparse_test.dense_to_sparse_npu` | 同上 |

约束：二维 Dense；索引 I32、base 0/1；Analysis/Convert/payload 构造 **禁止 CPU fallback**。

### 与 SparseToDense 关系

| 算子 | 方向 | 阶段 |
|------|------|------|
| DenseToSparse | 稠密 → 稀疏 | GetBufferSize / Analysis / Convert |
| SparseToDense | 稀疏 → 稠密 | bufferSize / execute（对偶 scatter） |

---

# 需求分析（required）

## 需求描述

在 Ascend 950 上交付完整 DenseToSparse：五 dtype × 四格式 × base0/1 × ROW/COL；精度 exact match；性能达到任务书 P 场景 ≥0.3× GPU 标杆；内存满足 IO>500MB 相对 GPU 增量 ≤50% 或固有 workspace≤L2；含 C++ ST、ATK、Python UT。

## 需求拆解

| 编号 | 子项 | 验收要点 | 950 状态（`0a04f103` / `20260902_074307`） |
|------|------|----------|---------------------------------------------|
| R1 | 三阶段 API | GetBufferSize / Analysis / Convert 对齐 cuSPARSE | ✅ |
| R2 | dtype | INT8/FP16/BF16/FP32/COMPLEX64 | ✅ |
| R3 | 格式 | CSR/CSC/COO/Blocked-ELL；I32；base 0/1 | ✅ |
| R4 | 确定性 | nnz/坐标/values bit-wise 可复现 | ✅ CSV |
| R5 | PTA | dense_to_sparse_npu + aten::_to_sparse*；无 CPU fallback | ✅ pytest |
| R6 | 精度 | 任务包 200 + C++ CSV（含 c64 SPECIAL） | ✅ CSV/ACC_FB；ATK 🟡（`ATK_RC=1`，排障中） |
| R7 | 性能/内存 | P-01/P-02/P-03 及泛化；≥0.3×；workspace/IO 规则 | ✅ PERF/MEM |

## 输入输出规格

| 参数 | 方向 | 类型 | 说明 |
|------|------|------|------|
| handle | 输入 | aclsparseHandle_t | 携带 stream |
| matA | 输入 | aclsparseConstDnMatDescr_t | 只读 Dense，ROW/COL，ld 合法 |
| matB | 输入/输出 | aclsparseSpMatDescr_t | 目标稀疏；Analysis 更新元数据；Convert 写 payload |
| alg | 输入 | aclsparseDenseToSparseAlg_t | 仅 DEFAULT |
| bufferSize | 输出 | size_t* | Analysis/Convert 共用 workspace |
| buffer | 输入 | void* | 查询为 0 时可空；两阶段可复用同块 |

| value dtype | index | base | format | order | 950 |
|-------------|-------|------|--------|-------|-----|
| INT8/FP16/BF16/FP32/COMPLEX64 | I32 | 0/1 | CSR/CSC/COO/BELL | ROW/COL | ✅ |

---

# 详细设计（required）

## 算子分析

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

arch35 **SIMT**（`asc_vf_call` + grid-stride）。DenseToSparse 为不规则稀疏发现，SIMD/RegBase 难以表达动态前缀和写回，保留 SIMT。线程块规模常量 `kDenseToSparseThreads = 256`。

### Workspace

- Header（status/nnz）+ level0 counts + 多层 scan 缓冲，32B 对齐
- BELL 路径 workspace 可为 0
- 大 IO 场景：任务书 §3.4（IO>500MB 相对 GPU 增量 ≤50%，或固有 workspace ≤ L2）

## Host 设计（`densetosparse_host.cpp`）

1. **GetElementBytes**：INT8=1，FP16/BF16=2，FP32=4，COMPLEX64=8
2. **校验**：layout/ld、format、index 类型一致性（CSR/CSC offset 与 index 同宽）、alg=DEFAULT
3. **Tiling**：计算 `unitCount`（COO tile / CSR·CSC major×chunk）与 `numBlocks`；填充 `DenseToSparseTilingData`
4. **Launch**：Analysis/Convert 后按需同步，读 Device status

关键 tiling 字段：`rows/cols/ld/nnz/ellBlockSize/ellCols/unitCount`、workspace 各段 offset、`format/order/base/elementBytes/numBlocks`。

## Kernel 设计（`densetosparse_kernel.cpp`）

- `IsNonzero` 特化：u8 / u16(fp16·bf16) / u32(fp32) / u64(c64)
- `densetosparse_count_kernel` / `convert_kernel` / `bell_kernel` 按 `elementBytes` 分派
- Scan 与 offsets 与 dtype 无关；CSR/CSC 写 offsets 端点；COO 只发布 nnz

## Python/ATen（`python/npu_sparse_densetosparse/`）

```text
dense_to_sparse_npu(format, dense, base, layout, block_size)
  → ctypes 三阶段 + CreateCsr/Csc/Coo/BlockedEll + SetPointers
  → 返回与任务包一致的 canonical tuple

aten::_to_sparse / _to_sparse_csr / _to_sparse_csc / _to_sparse_bsr
  → PrivateUse1 注册；无 CPU fallback
```

Blocked-ELL：PTA 侧先发现非零块列 pattern，再走 ACL 路径（Torch 无完全等价公开 API）。

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

## 精度 / 性能 / 内存标准

| 验收标准 | 描述 | 标准来源 | 950 实测（`0a04f103` / `20260902_074307`） |
|----------|------|----------|---------------------------------------------|
| 精度 ST | CSV bit-exact + L2 负向 | 仓内 GTest | ✅ `densetosparse_test` **115 PASSED**；ACC **83 PASSED** |
| 精度 Fallback | 无 ATK schema 同 case | 任务包 | ✅ `ACC_FB_RC=0` |
| 精度 ATK | 200 泛化 | 任务包 ATK | 🟡 `ATK_RC=1`（插件/环境排障中；Fallback 已绿） |
| 性能 | GPU/NPU ≥ **0.3** | 任务书 §3.3 P-01~P-03 | ✅ `PERF_RC=0` |
| 内存 | IO>500MB 增量≤50% 或 workspace≤L2 | 任务书 §3.4 | ✅ `MEM_RC=0` |
| PTA | pytest | 仓内 | ✅ `PYTEST_RC=0` |

## P-01 / P-02 / P-03 场景定义

| 场景 | Dense shape | 每行非零 | BELL blockSize | 说明 |
|------|-------------|---------|----------------|------|
| P-01 | 8192×28672 | 64 | 16 | Llama 3.1 70B 类 |
| P-02 | 4096×1536 | 64 | 32 | Qwen3-235B 类 |
| P-03 | 7168×2048 | 64 | 64 | DeepSeek-V3 类 |

每场景覆盖五 dtype × 四格式 × base0/1（明细见任务包 `gpu_performance_result_benchmark.md`）。

## 测试设计

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
| 多参与者路径冲突 | 本账号独立目录 `-7m2FRYfKJXCaXD8P2gMKd1d/` |

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
3. 自测报告（腾讯文档模板）+ 950 日志分支证据（`20260902_074307`）
4. 邀请 Ascend-CANN 为开发者
