# 需求背景（required）

## 需求来源

本设计对应《CANN训练营北京邮电大学-torch_sparse工具类接口开发(950)》任务书：在 Ascend 950PR 上使用 Ascend C、C++ 和 PyTorch 扩展实现 `non_diag_mask`、`random_walk` 的 NPU 路径。

- 目标：[cann/ops-gnn](https://gitcode.com/cann/ops-gnn)，目录 `experimental/torch_sparse_utils/`。
- 对标：[rusty1s/pytorch_sparse](https://github.com/rusty1s/pytorch_sparse)，记录实际 CPU/GPU 标杆版本、硬件及运行环境。
- 边界：仅新增 ops-gnn 的 NPU 路径，与 ops-sparse 的 `aclsparse*` 解耦，不修改 ops-sparse，不回退 CPU 计算。

本文描述 2026-09-08 最终修复副本 `fixed_run_142100` 的实际实现。该版本已完成 clean build、官方 400 条 EXACT、原自测及新增前端回归、输入契约和补充功能检查；官方附件原 400 条性能集合的 Event、Kernel 与内存证据完整，固定集及两接口各自 200 项集的两种平均性能比均达到 0.3×。此通过结论限定于已验证的官方附件集合，不能把该集合范围等同于任务书全部书面范围；单卡环境仍有 3 个跨设备 SKIP。本设计及拟议 PR 修改保留本地，未上传。

## 背景介绍

### 现状分析

两接口分别用于稀疏图对角线设置与随机采样。对标库的 CPU/CUDA 路径需要原生 NPU 补充。任务书的 `non_diag_mask` 采用逐边判断再尾部补 False 的简化语义，与 upstream 为有序插边生成的散布掩码存在排列差异。NPU 前端需要保持任务书公共接口结果，同时保证 `SparseTensor.set_diag/fill_diag` 的 COO 排序及 CSR 存储一致。

随机游走的多邻居路径受 RNG 算法、seed、布局及消费顺序共同影响。相同 seed 不足以保证三端路径一致；任务书仅在 RNG 对齐时要求该场景逐元素比较。

### 功能分析

| 接口 | 功能 | 输入 | 输出 |
| --- | --- | --- | --- |
| `torch_sparse::non_diag_mask` | 非目标对角边掩码与补对角占位 | int64 `row[E]`、`col[E]`，M、N、k | bool `[E+num_diag]` |
| `torch_sparse::random_walk` | CSR 均匀邻边游走 | int64 `rowptr[V+1]`、`col[E]`、`start[S]`，L | int64 `[S,L+1]` |

# 需求分析（required）

## 需求描述

采用 torch 接口工程化开发模式，交付 Ascend C Kernel、Host Launch、PyTorch 注册、Python 加载入口和测试。执行 `import torch_sparse_npu` 后，下列接口应在 NPU 输入下可用：

```text
torch_sparse::non_diag_mask(row, col, M, N, k) -> mask
torch_sparse::random_walk(rowptr, col, start, walk_length) -> out
SparseTensor.set_diag / fill_diag -> non_diag_mask
SparseTensor.random_walk(start, walk_length) -> random_walk
```

任务书的正常输入语义、异常报错、EXACT 精度、性能、内存及前端集成要求均保持不变。官方脚本必须执行，未被脚本覆盖的书面契约也须验证，不能以正常用例通过替代异常与前端组合性质的验证。

## 需求拆解

| 项目 | 本次实现 | 验证要求 |
| --- | --- | --- |
| mask | 融合 COO 值域校验、以 `col-row` 判断对角线、完整输出 | 空/矩形/正负及极端 k，错误索引与属性，逐元素 EXACT |
| walk | 融合完整 CSR/col/start/rand 校验、每次依赖读取的范围保护、死端 -1 后缀 | 单邻居 EXACT；孤立/hub/大图；异常不得静默转为 -1 |
| Host | 元信息、设备、属性与输出溢出检查；64 位 L/nnz 参数 | 空 start、L=0、大 L 空输出仍校验合法性 |
| Python 前端 | NPU 对角适配器排序 COO、重建缓存；NPU ind2ptr/ptr2ind | `set_diag/fill_diag` 后 COO/CSR/后续使用一致，双导入顺序 |
| 性能与内存 | 新校验始终处于同一公共调用；无内容有效性缓存 | 修复版本重新测量，不能复用旧性能结论 |
| 交付 | 源码、构建、用例、日志、CSV/JSON、文档和 PR | 用同一版本的源码/加载二进制哈希关联证据 |

# 详细设计（required）

## 算子分析

### 数学公式

**non_diag_mask**：

```text
num_diag = max(0, k >= 0 ? min(M, N-k) : min(M+k, N))
mask[i] = (row[i] + k != col[i])       0 <= i < E，数学整数表达
mask[i] = False                       E <= i < E+num_diag
```

k=0 为主对角，k>0 满足 row+k=col，k<0 满足 row=col+|k|。M/N 非负、k 为任意 int64；越出矩阵的对角线长度按 0 处理。

代码在验证 `0 <= row < M`、`0 <= col < N` 后使用 `col-row != k`，两个合法非负 int64 索引之差可表示，避免 `row+k` 有符号溢出及 `abs(INT64_MIN)`。Host 先检查 `E <= INT64_MAX-num_diag` 再分配输出。

**random_walk**：V 为节点数，E 为 nnz，S 为起点数，L 为步长。

```text
out[n,0] = start[n]
begin = rowptr[cur]; end = rowptr[cur+1]
degree = end-begin
degree > 0: 从 [0,degree) 选择 idx，next=col[begin+idx]
degree == 0: 不读取 col，后续输出为 -1
```

输出是 int64 `[S,L+1]`。L=0 时只写首列；S=0 时返回 `[0,L+1]`，仍验证图内容。非法图或非法起点触发错误；-1 仅用于合法死端后的未写入位置，不代替非法输入报错。

### 支持数据类型

| 参数 | 类型 | 约束 |
| --- | --- | --- |
| row、col、rowptr、start | NPU int64 Tensor，ND | 一维，同一调用全部位于同一 NPU 设备 |
| mask | NPU bool Tensor | kernel 以 uint8 写入 0/1 |
| out | NPU int64 Tensor | 合法节点编号或 -1 |
| M、N、k、L | int64 scalar | M/N/L≥0，k 可为任意 int64 |
| rand（辅助接口） | NPU float32 Tensor `[S,L]` | 同设备、有限值且 `0 <= rand < 1` |

L 和 nnz 的 launch 参数使用 64 位，不再采用旧代码的 uint32 上限。L+1 必须可由 int64 表示；Host 检查非空输出的 `8*S*(L+1)` 不超过 int64 字节范围。可分配资源不足仍由分配器报错。S=0 时支持 L 超过 2^32 的合法空输出。

### 支持形状

- COO row/col 都为 `[E]`，长度相等、E≥0。M/N 可为 0，非空索引仍须满足值域。
- CSR rowptr 为 `[V+1]`、V≥0；首项 0、末项 E、非降，全部项在 `[0,E]`。col 为 `[E]`，全部项在 `[0,V)`。
- start 为 `[S]`，S≥0，全部起点在 `[0,V)`。V=0 时必须同时有空 col 和空 start。
- 输出 `[S,L+1]`；无 batch 维或广播。非连续输入先在 NPU 连续化，其开销属于公共调用。

## 算子实现

### 实现方案

两计算路径使用 950PR arch35 SIMT：`__simt_vf__`、`AscendC::Simt::VF_CALL`、每 block 256 个线程、GM int64 读写。mask 逐元素独立，walk 每线程处理一条完整路径，路径内部依赖 gather 串行。

#### 3.2.1 host侧设计

Host 使用 `TORCH_CHECK` 检查元信息、属性及尺寸溢出，以输入设备的 DeviceGuard 和当前 NPU stream 连续化、分配并提交计算。

| 校验 | 实现位置 | 错误方式 |
| --- | --- | --- |
| 已定义、NPU、int64、一维、同设备 | Host | 调用时立即抛出 RuntimeError |
| row/col 等长、M/N/L 非负、rowptr 至少一项 | Host | 调用时立即抛出 RuntimeError |
| 输出长度/字节数溢出 | Host | 在分配前抛出 RuntimeError |
| COO row/col 值域 | mask kernel，和逐边计算融合 | 设备异常在同步时暴露 |
| CSR 首尾、非降、范围及全部 col 值域 | walk kernel 全量并行扫描 | 设备异常在同步时暴露 |
| 每个 start 值域 | walk kernel，写首列前 | 设备异常在同步时暴露 |
| rand float32、`[S,L]`、同设备 | Host | 调用时立即抛出 RuntimeError |
| rand 有限且在 `[0,1)` | 注入 kernel 全量并行扫描 | 设备异常在同步时暴露 |

内容检查失败时通过公共 Ascend C SIMT `__trap` 中断 kernel，附 `ascendc_assert` 诊断；始终执行的 `__trap` 保证检查不会因 NDEBUG 或 ASCENDC_DUMP=0 而消失。950PR 支持该接口，见[官方说明](https://asc.gitcode.com/api/Utils-API/tuning_interface/__trap.html)。本次独立进程验证已确认 52 项非法输入均被捕获为 Python RuntimeError，其中 24 项记录为显式同步阶段抛出；全部子进程正常退出并释放设备。没有把进程崩溃或超时计为通过。

正常路径没有新增 Host 状态回传或显式同步。内容错误可能在 `torch.npu.synchronize()` 或后续同步读取时才暴露；调用返回 Tensor 不等于设备执行完成，空输出也不自动触发错误观察。元信息错误与内容错误的异常时机分别记录。

公共随机路径在 L>0 且 S>0 时，持锁从默认 NPU generator 获取 `philox_engine_inputs(4)`。由于设备内容检查是异步的，内容非法的调用可能已经消费 seed/offset；不承诺非法调用不推进 RNG 状态。

##### 1. 分核策略

```text
threadsPerBlock = 256
blockDim = max(1, min(ceil(total/256), availableAivCores))
```

mask 的 total 为 E+num_diag；walk 的 total 为 `max(V+1,E,S)`，注入模式还包含 S*L。按平台 API 获取 AIV 核数，以 grid-stride 遍历。mask 空输出可跳过 launch；walk 即使 S=0 或 L=0 仍执行 CSR 内容扫描，L=0 的首列由同一 kernel 写入。

##### 2. 数据分块和内存优化策略

mask 把索引校验与输出融合。walk 在同一 launch 中先按线程分工扫描 rowptr/col/rand，再处理路径；每个依赖 gather 另有范围保护，所以不依赖跨 block 的校验完成屏障。没有“缓存此前校验结果”、benchmark 特判或可关闭的必需校验。

当前两 kernel 没有动态 workspace、显式 UB 中转、全量随机张量或全图 Host 副本，launch 使用 nullptr workspace。输出的每个位置由 kernel 写入，不需额外预清零。校验增加的 O(V+E+S) 读取及注入模式 O(S*L) 扫描属于真实计算成本，须计入修复后性能。

有效输出载荷为 mask 的 `E+num_diag` 字节和 walk 的 `8*S*(L+1)` 字节，但额外峰值还可能包含连续化副本、前端排序/索引转换临时张量、allocator 对齐或内存池变化；不能写“额外内存等于输出”。任务书允许满足：大于 500 MB 输入输出时的相对额外内存限制，或固有 workspace≤目标 L2。kernel 的零 workspace 是第二项设计依据，实际峰值及前端所需空间仍需记录。

##### 3. tilingkey规划策略

索引 dtype 单一、没有广播，无需按 dtype/shape 设置 tilingKey。rand 来源由编译期 INJECTED 模板区分。S/E/V/L/k 等标量直接传入，所有内容检查均在当前对应计算 launch 内执行。

#### 3.2.2 kernel侧设计

mask 读取一条 COO 边，检查值域后以 `col-row!=k` 写前 E 位；尾部 num_diag 位写 False。

walk 首先在 GM 全量检查：rowptr 首项、末项、各项范围和相邻非降；col 各项范围；注入 rand 的每个元素范围。每条路径开始前检查 start，然后写首列。每一步在读取 rowptr 前检查 current，在读取 col 前检查 `0<=begin<=end<=E`；degree=0 返回内部死端标记并填 -1 后缀。非法 current、行段、偏移或 col 触发设备错误，绝不静默伪装成合法死端。

**RNG 实现**：

- Philox4x32-10 每次 refill 生成 4 个 uint32；key 为 seed/offset 高低 32 位对应异或，`counter[2:3]` 编码 walk n，`counter[0:1]` 每次 refill 自增。
- 每条 walk 按步消费一个 uint32，包括首次到 degree=0 的那一步；随后 current=-1 时只填后缀，不再消费。L=0 或 S=0 不取默认 generator 状态。
- 选边为 `floor(u32*degree/2^32)`，用 degree 高低 32 位拆分乘法实现，不截断 64 位 degree，也不依赖 uint128。
- 辅助 `random_walk_with_rand` 使用共同 float32 `[S,L]` 输入，以 float32 乘法和截断选边；若乘法舍入到 degree，则限制为 degree-1，以免读到下一行。合法死端同样终止。
- 注入模式与 `tests/common.py` 的独立任务语义参考进行 EXACT 校验；它不证明公共 Philox 与 upstream CPU/GPU RNG 对齐，也不无条件承诺 upstream 的死端或浮点舍入行为相同。

**SparseTensor 前端实现**：

NPU `set_diag/fill_diag` 调用公共 task-book mask，保留非目标边、拼接目标对角线后，按 col 与 row 做稳定排序，避免 `row*cols+col` 的 int64 乘法溢出。重新生成 rowptr/rowcount/colcount，清空依赖排列的 colptr/csr2csc/csc2csr 缓存。`ind2ptr` 用 NPU 计数与前缀和，`ptr2ind` 用已知 output_size 的 NPU repeat_interleave；不把索引转到 CPU。num_diag=0 直接构造空对角线，避免对 INT64_MIN 求反后建立 arange。

适配只处理 NPU Tensor；CPU/CUDA 前端继续使用 upstream 路径。本次最终版本的原三项自测、新增前端回归及双导入顺序检查已通过；旧版本的孤立节点与 COO/CSR 缺陷复现用例已确认修复。证据按最终加载二进制及源码哈希关联。

## 支持硬件

| 项目 | 要求 |
| --- | --- |
| NPU | Ascend 950PR，dav-3510 / arch35 SIMT |
| GPU | 任务指定 GPU 标杆环境、pytorch_sparse GPU 实现 |
| CANN/PyTorch/torch_npu | 按 ops-gnn 指定版本及配套关系 |
| 已用环境记录 | CANN 9.1.0、torch 2.9.1+cpu、torch_npu 2.9.1、torch_sparse 0.6.18、Python 3.12.13；最终版本以新复验 environment 与 artifact manifest 为准 |

## 算子约束限制

1. 公共 mask 保持任务书简化语义，排序适配位于 NPU 前端；不改变公共 mask 以迎合 upstream 散布布局。
2. 空/矩形 COO、正负或越界 k、空 start、零步长与孤立节点均为正常场景；不能从验收集合剔除。
3. 元信息错误立即报告，内容错误异步报告。内容非法后不得继续把已返回输出当作有效结果；独立子进程用于验证该错误契约。
4. 无 batch 维、广播；L/nnz 不再人为限制为 uint32，仍受输出维度和分配资源限制。
5. NPU 计算不得回退 CPU。测试用 CPU 参考与诊断性 CPU 拷贝不属于 NPU 实现路径。
6. 未提供三端 RNG 对齐证据时，不声称多邻居路径跨设备逐元素一致。

# 可维可测分析

## 精度标准/性能标准

| 项目 | 任务书判定 |
| --- | --- |
| 功能与异常 | 正常语义、shape/dtype、COO/CSR/start 非法报错；覆盖空输入、正负 k、孤立/hub/≥1e5 节点等要求 |
| 精度 | bool/int64 EXACT；确定性路径对标 CPU/GPU，多邻居按 RNG 对齐状态执行任务规定校验 |
| 性能 | 固定集、mask 原 200 项、walk 原 200 项分别计算逐 case GPU/NPU 比的算术平均，两项均值均 ≥0.3× |
| 内存 | 满足任务书 §3.4 两项条件之一，报告峰值/额外内存 |
| 交付 | 源码、构建、用例、日志、CSV/JSON、README、设计、验证指南、PR；版本到证据可追溯 |

精度来源为任务书 §3.2/§7及[实验算子精度标准](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md)。附带 ATK 配置不构成任务书正文未指定的额外硬门槛；没有执行报告时不得声称“ATK 80 例通过”。

### 功能与异常验证

修复前的独立复验已有官方 400/400 EXACT 和三项旧自测通过，但额外合法输入诊断同时复现孤立节点跳到下一行，以及对角设置后的 COO/CSR 不一致。上述记录只说明修复动机和旧测试覆盖不足。

本次 `final_functional_summary.json` 记录 clean build 退出码 0，官方 400/400 EXACT，原三项自测及新增前端回归通过。契约检查 69 项中 66 通过、0 失败、3 项因单卡环境跳过：14 项合法输入 EXACT、52 项非法输入捕获 RuntimeError；69 个子进程均正常退出并释放设备，其中 24 项错误明确发生于显式同步阶段。另有 7/7 补充功能 EXACT，两个原缺陷复现均已修复。3 个跨设备 SKIP 不写成通过。

最终源码快照为 `/root/tsu_review_20260908/fixed_run_142100/torch_sparse_utils`；33 个源码哈希已核对，实际加载 `_C.so` 的 SHA256 为 `34bc517b00b36c3e7757f02a838fa70d1a4c7ad29ba15db1a3cd47f12acba946`。新功能材料已归档于本地复核包的 `evidence/fixed_20260908/`。尚未放入实现仓的材料不作为仓内相对链接。官方脚本、输入与参考未修改；私有 `tests/common.py` 仅按任务书修正合法死端参考。

此外，本次对原 400 条性能参数另做逐例正确性检查：200 条 mask 对 CPU 任务参考 EXACT；200 条 walk 检查全部转移合法性、首列/shape/dtype/值域、死端后缀和同 seed 重复，全部通过。后者不声称未对齐 CPU/GPU RNG 的随机路径 EXACT。7 条补充功能暂无相同场景 GPU 性能基线，不能用于宣称书面全性能覆盖。

### 性能集合与计算

| 固定项 | 参数 | GPU Event wall（ms） | GPU Kernel 总和（ms） |
| --- | --- | ---: | ---: |
| fixed_mask_01 | M=N=10,000，E=100,000，k=0 | 0.009904 | 0.004800 |
| fixed_walk_01 | V=10,000，E=100,000，S=1024，L=32，regular | 0.135451 | 0.058784 |

两固定项单独组成固定集；每接口原 validation_200 文件各完整保留 200 条，其中包含固定项，不删成 199 条。对原始集合 C、N=原始例数：

```text
wall_ratio_i = GPU_Event_wall_ms_i / NPU_Event_wall_ms_i
kernel_ratio_i = GPU_kernel_sum_ms_i / NPU_kernel_sum_ms_i
mean_wall_ratio(C) = sum(wall_ratio_i) / N
mean_kernel_ratio(C) = sum(kernel_ratio_i) / N
```

不能采用两端均值之比，也不能把两个接口混合后代替各自均值。失败、不支持、未执行项保留在集合与分母中并明确未完成，不只筛成功项宣称通过。单项按 0.3× 折算的时间上限仅作定位参考。

官方 benchmark 测量完整公共 API，新增的内容扫描、读前检查、连续化及分配始终处于被测调用内；没有测量用校验开关、输入有效性缓存或特殊 case 路径。本次已重新完成原 400 条 Event wall 和内存采集；GPU 数值取当前官方附件 CSV，未重新测量 GPU。Event 和 Kernel 来自不同测量窗口，差值不能全部归因于 Host 派发。

| 修复后集合 | 完整分母 | Event GPU/NPU 比均值 | Kernel GPU/NPU 比均值 | 两项≥0.3× |
| --- | ---: | ---: | ---: | --- |
| non_diag_mask 原 validation 集 | 200 | 2.257141233232× | 1.396470407473× | 通过 |
| random_walk 原 validation 集 | 200 | 0.787161342070× | 3.085682684560× | 通过 |
| 固定集合 | 2 | 1.200896656058× | 1.007575883960× | 通过 |

最终数据来源为复核包 `evidence/performance_fixed_20260908/fresh_comparison.json` 与 `finalization_summary.json`。400 条逐 case 独立 PID 均正常退出，400 份身份检查均匹配已测二进制，每例严格核对五次目标 CPU 调用/五个 kernel 事件。采集 warmup=5、active=5，每步保留输出并同步，前后同步划清窗口；NPU 锁持有到进程正常退出，并行只用于 CPU 预加载，NPU 负载串行。当前冻结代码对这些官方调用每次恰好一个计算 launch，校验融合其中，全部设备事件计入总量，没有过滤校验开销。

原始 flow 仍缺少起点，CPU-device 逐事件可视化关联标记保留 `provisional_missing_or_unmapped_flow`；这不等于 case 的 Kernel 总量不完整。边界同步、独立进程、代码/二进制身份与当前每调用一个 launch 的事实，加上五次 CPU/五个 kernel 严格计数，共同支持本次每 case 总时长。未声称缺失的 flow 链接已经恢复。

首次新采集有 156 条 case 出现 4/6 个设备事件，已归档 `rejected_capture/`，其 Kernel 数值及自动通过结论不使用。最终只替换经过完整审计的 Kernel 结果；既有新 Event、内存与正确性文件的 SHA256 全部保持不变。

### 书面覆盖与当前官方附件差异

任务书要求 mask 覆盖 M/N=1e2～1e5、E=0～1e6、k=0/±1/±3/±10；walk 覆盖 V=1e3～1e5、E=1e3～1e6、S=0/1/64/1024/4096、L=0/1/8/32/64，以及 regular/hub/isolated。

当前官方 200 项附件实际存在覆盖缺口：mask 的 M 仅 1/10,000，E 最大 100,000，k 仅 0/1/3/10；walk 的 V 最大 10,000，S 为 1/16/64/256/1024，L 为 1/8/32/64。记录无漏项不等于书面范围全覆盖。

本轮从登录后的官方页面重新下载原 zip，核验任务书及 18 个测试文件与本地附件一致，记录于 `live_taskbook_comparison.json`；因此不能把差异归因于本地附件被改动。保留官方 case 和 GPU baseline 原样，缺少场景单独登记补充证据；没有对应 GPU 基线时标明比较未完成，不能凭 NPU-only 数据声称全量性能验收通过。

### 历史性能证据边界

修复前数据按正确公式重算：mask 200 项接口/Kernel 均值 2.228142×/1.972190×；walk 200 项 0.883574×/17.224177×；固定集 1.118625×/1.072737×。这些仅为历史记录，不用于本次修复版本判定。

修复前原 trace 曾核对运行目录、active step、CPU 调用、kernel 事件和时长算术，但其历史数值不参与最终判定。本次最终 Kernel 已按前述独立 PID、同步边界、身份与事件计数方法重新采集和核验；缺失 flow 链接的限制单列说明，不用旧数据替代新数据，也不只凭 `status: ok` 或固定除以 5 判定可信。

### 本次交付状态

| 事项 | 当前状态 |
| --- | --- |
| 源码修复 | 最终版本 clean build 完成；33 个源码哈希及实际加载二进制已关联 |
| 官方功能/自测/前端 | 官方 400 EXACT、原自测及新增前端通过，7 条补充 EXACT，两个原缺陷均已修复 |
| 异常契约 | 69 项中 66 通过、0 失败、3 个单卡跨设备 SKIP；异步 RuntimeError 已实证 |
| 原性能集合正确性 | mask 200 EXACT；walk 200 转移合法与 seed 重复通过，不声称跨端随机 EXACT |
| 性能/内存 | 原400条Event、最终Kernel和内存齐全；两个200项集及固定集两项性能均达标，首次无效Kernel已隔离归档 |
| 书面额外覆盖 | 官方附件缺口单列，GPU/NPU 对照未齐时不宣称全覆盖 |
| PR | 本 PR 更新比赛仓设计说明；设计需重新评审并实际合入，产品代码和后台验收按下述流程分别推进 |

### 任务流程与交付顺序

依据 2026-09-08 核对的[北邮专场任务引导](https://gitcode.com/org/cann/discussions/285#tid-77501f53280c4cd1b747834a1ee099d0)、[开发流程及注意事项](https://gitcode.com/org/cann/discussions/39)和本任务书，报名、设计 PR、进展与验收使用同一 GitCode 账号 `bluerain117`。本任务为 [torch_sparse 工具类接口开发(950)](https://www.hiascend.com/activities/task-center/details/18a81f3be4d4437f9d4b6a45ff898305)，[官方任务书与测试附件](https://www.hiascend.com/p/resource/202609/9cdc329cf10d4f389699e18b5dd4f994.zip)保持原件。

1. 本 PR 为设计评审，标题使用 `【CANN社区任务】torch_sparse工具类接口算子设计文档`，路径固定为 `04_tasks/01_community-task-2026/tasklist/09-41-torch_sparse_utils-950/bluerain117/docs/design.md`。按官方四章模板描述接口、实现、资源、验证和兼容性；本次修订继续使用 PR #1405，不重复新建设计 PR。
2. 设计阶段在任务页面登记设计 PR，代码链接按流程填写“暂无”；后续设计修改在原 PR 完成，无需重复更新进展。更新后按流程通知评审人并处理意见。CI、CLA 和评审留言分别核对，不能把旧提交的通过状态当作新提交已通过。
3. 正式申请验收前，设计必须评审通过并实际合入，保存合入截图；同时固定待验收实现的源码、构建产物和报告版本，准备个人代码仓链接、分支及 `experimental/torch_sparse_utils/` 目录，并按流程确保验收人员可访问。设计 PR、个人待验收代码仓、产品代码 PR 是不同交付对象。
4. 在任务截止前，通过任务页面提交设计合入截图、功能/精度脚本与用例、逐例日志及报告、性能与内存 CSV/JSON、构建/复现指南等完整交付件。当前平台曾退回的逐例证据缺项已补充，但本文自测结果不表示后台验收已通过；书面性能覆盖与附件差异及跨设备 SKIP 仍需如实说明并完成必要确认。
5. 后台测试通过后，按专场流程在通知后的规定时限内向 `cann/ops-gnn` 提交需求 issue 和产品代码 PR，目录按任务书 §5；依据意见修改直至合入。只有完成对应后续流程，才能认定任务流程结束。

本次自测证据仅支持文中明确列出的集合与版本。对当前附件没有覆盖且缺少 GPU 基线的性能场景，应请评审确认补充集合和基线，不能自行降低书面要求或把功能补测等同于性能补测。最后的两份随机游走源文件只做等行数注释澄清，同路径重编译后 `_C.so` 与 kernel so 的字节 SHA256 均与上述已测版本完全相同；源码哈希通过 `comment_amendment.json` 和 `expected_sha256_final.json` 关联，不混用不同实现的性能证据。

## 兼容性分析

注册在 `torch_sparse` 相同 schema 的 PrivateUse1 key，辅助命名空间 `torch_sparse_npu` 提供直接调用与 rand 注入。Python 加载时先尝试 upstream 导入，避免重复 schema 定义；NPU-only 前端适配和转换注册保留在已加载 upstream 模块上，CPU/CUDA 前端保持原实现。

兼容性验收须使用真实依赖环境验证两种导入顺序及 `set_diag/fill_diag/random_walk` 后续 NPU 使用，不能仅凭注册代码或设计 PR 的 CI/评审状态认定整个任务验收完成。
