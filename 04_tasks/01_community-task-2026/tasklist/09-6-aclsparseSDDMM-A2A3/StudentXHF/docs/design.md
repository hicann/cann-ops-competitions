# aclsparseSDDMM 算子设计文档（Atlas A2/A3）

> 任务：9月社区任务-aclsparseSDDMM算子开发（A2/A3）
> 参与者：StudentXHF
> 目标代码分支：`StudentXHF/ops-sparse:feature/aclsparse-sddmm-a2a3-complete-studentxhf`
> 目标平台：Atlas A2 训练系列（910B3/910B4）与 Atlas A3（DAV_2201）

# 需求背景（required）

## 需求来源

本文对应 2026 年 9 月 CANN 社区任务《aclsparseSDDMM 算子开发任务书（A2/A3）》。任务要求在
`ops-sparse` 的 `arch22` 路径补齐 SDDMM 的 C++ 三阶段接口、Host 校验与调度、Ascend C Kernel、
CSR/BSR 描述符、strided batch、Python/ATen NPU 适配、测试及文档。

主要依据如下：

1. 官方 `aclsparseSDDMM_A2A3_task_doc.md`；
2. 官方精度、性能、内存测试包及 H100/cuSPARSE 标杆结果；
3. `ops-sparse` 的公开头文件、描述符和已有 arch22 实现；
4. PyTorch 2.7+ 的 `torch.sparse.sampled_addmm` / `aten::sparse_sampled_addmm` 语义。

## 背景介绍

SDDMM（Sampled Dense-Dense Matrix Multiplication）只计算稠密乘积在稀疏矩阵 pattern 指定位置的值：

```text
C.values <- (alpha * op(X) * op(Y) + beta * C) masked by spy(C)
```

其中 `C` 的 CSR/BSR 结构只读，values 原地更新。与先计算完整稠密矩阵再取样相比，直接按非零位置
计算可避免 `M*N` 级中间结果，适用于图学习、稀疏注意力和稀疏优化场景。

### 现状分析

本分支以公开 MR `cann/ops-sparse!188` 的代码内容为工程基线，并与当时最新 `master` 合并；
MR !188 已实现大部分 arch22、BSR、混合精度、complex64、Python/ATen 和测试框架能力，但其说明明确
列出两个验收缺口：前三种 strided-batch 广播未支持、ATen complex64 用例跳过。本设计在提交信息中以
`Co-authored-by` 保留原作者署名，并在此基础上补齐这些缺口，同时加强 pattern 内容失效和描述符
stride/index 边界校验。为满足仓库 `stat/needs-squash` 门禁，线上分支保持单提交。

# 需求分析（required）

## 需求描述

实现目标如下：

1. 公开 API 保持 `aclsparseSDDMMBufferSize`、`aclsparseSDDMMPreprocess`、`aclsparseSDDMM` 三阶段；
2. 支持 CSR/BSR、I32 索引、index base 0/1、ROW/COL、NON_TRANSPOSE/TRANSPOSE；
3. CSR 支持 FP16、FP32、complex64，BSR 另支持 BF16 混合精度组合；
4. BSR 方块大小支持 2/4/8/16/32/64/128，块内支持 ROW/COL；
5. 以 matC 的 batchCount 为输出批数，支持四种组合；
6. complex64 支持复数 alpha/beta，Python 与公开 ATen 入口不得跳过或 CPU fallback；
7. Preprocess 状态绑定 matC 完整 pattern，指针、元数据或内容变化时必须失效；
8. 在 910B3、910B4 和 A3 上完成精度、性能、内存、Profiler 与回归验证。

## 需求拆解

| 子系统 | 交付内容 | 关键验收点 |
| --- | --- | --- |
| 描述符 | BSR create、DnMat/CSR/BSR strided batch | batch 1..65535；stride 非负、无重叠、无溢出 |
| Host API | 参数、shape、dtype、layout、index、workspace 校验 | 错误码稳定；不访问非法结构 |
| Preprocess | pattern 校验、摘要、优化路径 sidecar | 状态只绑定 matC；无全局非线程安全缓存 |
| Execute | 四类 batch 广播、base、CSR/BSR、混合精度 | 每批使用正确 X/Y/C 指针和稀疏结构 |
| Kernel | AIV/Cube/BMM 路径 | NPU 核心计算，无 Host CPU 代算 |
| Python/ATen | sampled_addmm 公开入口与 Dispatcher | NPU device、alias、stride、异常、异步 |
| 测试 | C++ UT、Python UT、官方 case、Profiler | 真实 A2/A3 日志和可复现命令 |

# 详细设计（required）

## 算子分析

令 `op(X)` 形状为 `[M,K]`，`op(Y)` 形状为 `[K,N]`。对于 CSR 中第 `i` 行的第 `p` 个非零项，
列号为 `j=colInd[p]-base`：

```text
out[p] = alpha * sum(t=0..K-1, Xop[i,t] * Yop[t,j]) + beta * oldC[p]
```

BSR 将一个非零块展开为 `blockDim*blockDim` 个标量位置；块内地址按描述符的 ROW/COL 顺序解释。
complex64 按复数乘加计算，实部和虚部分别按 FP32 混合容差验收。

### 支持矩阵

| format | X/Y | C values | computeType |
| --- | --- | --- | --- |
| CSR/BSR | FP32 | FP32 | FP32 |
| CSR/BSR | complex64 | complex64 | complex64 |
| CSR/BSR | FP16 | FP32 | FP32 |
| CSR/BSR | FP16 | FP16 | FP32 |
| BSR | BF16 | FP32 | FP32 |
| BSR | BF16 | BF16 | FP32 |

## Host 侧设计

### 三阶段流程

```text
BufferSize
  -> 校验 handle/enum/descriptor/dtype/shape/batch/ld
  -> 根据通用 AIV 或可选重排/BMM 路径计算精确 workspace

Preprocess
  -> D2H 校验完整 CSR/BSR pattern
  -> 检查 base、offset 单调性与端点、column index 范围
  -> 计算 pattern hash 并记录指针、shape、format、base、block、batch stride
  -> 在用户 workspace 中构造重排数据和执行计划

Execute
  -> O(1) 校验描述符指针与 pattern 元数据，热路径不做 D2H 同步
  -> 指针或元数据变化则清理旧 sidecar；同地址原地改内容后调用方须重新 Preprocess
  -> 按 batch 选择 X/Y/C/rowOffsets/columns 的设备地址
  -> 在 handle.stream 异步下发 NPU Kernel
```

### strided-batch 广播

输出批数由 `matC.batchCount` 决定。X、Y 的 batchCount 必须等于 1 或等于 C 的 batchCount：

| X batches | Y batches | C batches | 语义 |
| --- | --- | --- | --- |
| 1 | 1 | B | `C_i=(A*B)∘C_i` |
| B | 1 | B | `C_i=(A_i*B)∘C_i` |
| 1 | B | B | `C_i=(A*B_i)∘C_i` |
| B | B | B | `C_i=(A_i*B_i)∘C_i` |

执行第 `i` 批时，singleton 输入批偏移固定为 0，否则使用 `i*batchStride`。CSR pattern 可由全部批共享；
BSR 的 offsets/columns stride 为 0 时共享 pattern，非零时逐批移动，values 始终按有效 stride 移动。

### stride 与溢出校验

- ROW DnMat 的最小 batch stride 为 `rows*ld`，COL 为 `cols*ld`；
- CSR values stride 至少为 `nnz`；
- BSR values stride 至少为 `blockNnz*rowBlockSize*colBlockSize`；
- BSR offsets/columns stride 为 0 表示共享，非零时分别至少覆盖 `blockRows+1` 与 `blockNnz`；
- 所有乘法在执行前进行上界/溢出检查。

### pattern cache

缓存字段位于 matC 描述符内部，包括结构指针、rows/nnz、format/base/block、batch 元数据和完整
offsets/indices 内容 hash。Preprocess 每次重新校验内容并更新 hash；Execute 可直接发现指针和元数据
变化，同地址原地修改 offsets/indices 时调用方按接口约定重新调用 Preprocess。这样热 Execute 不引入
D2H 拷贝或 stream 同步。不同 matC 和不同 workspace 无共享的可变全局表，可安全独立使用。

## Kernel 侧设计

通用 AIV 路径按稀疏项/块分核，直接从 X/Y 读取 dot-product 所需元素，并将结果写回 C.values。
tiling 中包含 M/N/K、nnz、dtype、layout、op、base、block、batchIndex 及 alpha/beta 的实虚部。

规则 pattern 在满足 dtype、alpha/beta、布局、batch 等前提时使用重排加 BMM/Cube 路径；否则回退到
通用 NPU Kernel，而不是 Host CPU。批处理目前优先走通用 Kernel，以保证四类广播和逐批 pattern 的
正确性；后续仅在真实 Profiler 证明收益且不改变语义时扩展 batch 快路径。

## Python/ATen 设计

Python 层校验 tensor 全部位于 NPU、layout/dtype/shape/stride 合法，然后创建 RAII 描述符并在当前
NPU stream 上依次调用三阶段接口。公开测试必须直接调用 `torch.sparse.sampled_addmm`，并验证输出
稀疏结构、values、device 与无 CPU fallback。

complex64 的 alpha/beta 不能压缩成实数。packed metadata 同时携带 alpha/beta 的实部和虚部；
C++ 扩展保留对旧 7 元素 metadata 的兼容，并支持新的 9 元素编码。

## 异常与资源生命周期

- 无效 handle、空指针、非法 enum、shape、dtype、layout、index、stride 返回明确错误码；
- `nnz=0` 不读取空 indices/values；非零 nnz 要求结构和值指针有效；
- workspace 由 BufferSize 精确查询，Preprocess 与 Execute 复用；
- sidecar 随 matC、workspace 或 pattern 失效释放；
- API 使用调用方 stream，不在正常路径做全局同步，不保存完整稠密输出。

## 支持硬件

| 平台 | SoC/架构 | 计划验证 |
| --- | --- | --- |
| Atlas A2 训练系列 | 910B3 / DAV_2201 | CANNLab 全量功能、精度、性能、内存、Profiler |
| Atlas A2 训练系列 | 910B4 / DAV_2201 | 官方可用环境回归 |
| Atlas A3 | 任务环境提供型号 / DAV_2201 | 官方可用环境回归 |

# 可维可测分析

## 测试设计

| 类别 | 覆盖 |
| --- | --- |
| 基础 | 方/长/宽、空行、nnz=0/1、动态 M/N/K/nnz、alpha/beta |
| 属性 | CSR/BSR、base 0/1、ROW/COL、opX/opY、全部 dtype 组合 |
| BSR | block 2/4/8/16/32/64/128、块内 ROW/COL、共享/独立 pattern |
| Batch | 四类广播；不匹配 batch；重叠和溢出 stride |
| Cache | 指针、offsets 内容、columns 内容、batch stride 变化后失效 |
| Python/ATen | 公开入口、Dispatcher、complex64 复数标量、异常、alias、device |
| 性能 | P-01/P-02/P-03，warmup 后 Execute，倍率不低于任务书阈值 |
| 内存 | workspace 精确值、额外峰值、无完整 M*N 输出、重复调用无泄漏 |

## 精度与性能判定

CPU Golden 采用任务书规定的高精度：FP16/BF16 用 FP32、FP32 用 FP64、complex64 用 complex128。
逐元素采用官方 rtol/atol、匹配率和 32 ULP 硬上限；complex64 实虚部分别判定。性能报告分别列出
aclsparse Execute 与 Python/ATen 端到端时间，使用官方固定 case、GPU 标杆和任务规定的计时口径。

## 当前验证状态

截至实现提交 `37aacde82e1e457159f468a3647cf89ffd6c4eb7`：Python 文件 `compileall`、
`git diff --check` 与官方第五轮完整 CI 已通过；CI 覆盖 SCA、antipoison、Check_Pr、pre_comment、
A2/A5 的 x86/ARM/Ubuntu 24 编译、CodeCheck、CodeCheck Style、pre-commit 与 PreSmoke_A900。
CANNLab A2 实例启动接口仍返回“当前资源不足，请稍后重试”，因此本文不声明当前提交的 NPU UT、
精度、性能、内存、Profiler 或 A3 测试已通过。
获得算力后将回填真实 CANN/驱动/固件/硬件/commit、逐 case 结果、失败项、Profiler 与日志链接。

# 兼容性分析

1. 公开 C ABI 和既有函数名不变；metadata 解析同时接受旧版与新版长度；
2. singleton batch 是原单批语义的自然扩展；不匹配批数显式报错；
3. fast path 条件不满足时进入同一 NPU 通用 Kernel，不进行 CPU fallback；
4. 修改限定于通用描述符校验、arch22 SDDMM、测试和 Python 扩展，需回归 arch35/A5 构建；
5. 线上单提交通过 `Co-authored-by` 保留公开 MR !188 原作者署名，并在 PR 正文明确说明工程基线；
   StudentXHF 只声明实际补充的差异，不将公开实现冒充为本人原创。

# AI 辅助披露

当前 PR 有 AI 参与：是。

- AI Agent 平台：OpenAI Codex；
- AI 模型：GPT-5；
- 使用范围：需求梳理、代码生成、代码审查、测试设计、文档撰写和问题排查；
- 所有代码、测试结果和提交内容由参赛者检查；未运行的测试不会标记为通过。
