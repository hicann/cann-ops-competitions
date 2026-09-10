# torch_sparse SpMM（Ascend 950）算子设计文档

提交人：Eternal_ZCL。对应任务：CANN训练营北京邮电大学—torch_sparse SpMM 算子开发（950）。

# 需求背景（required）

## 需求来源

- [SpMM 任务页面](https://www.hiascend.com/activities/task-center/details/43e2b3753e664baaa013bb2dd0b31ba0)
- [北邮专场任务指引](https://gitcode.com/org/cann/discussions/285)
- [官方设计模板](https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md)
- 实现目标仓：`cann/ops-gnn`，目录 `experimental/torch_sparse_spmm`。

## 背景介绍

图神经网络的邻居聚合需要将稀疏邻接矩阵与稠密节点特征相乘。算子在 Ascend 950PR 上实现 torch_sparse SpMM，支持 COO/CSR、sum/mean/min/max、混合输出精度和 sum/mean 一阶梯度，并接入 SparseTensor 与 PyG 聚合调用。

实现复用 `cann/ops-sparse` 的 arch35 SIMT CSR 遍历和直接 launch 思路，参考提交为 `9a514470c2cc99e469c34a615048eb8b30a66694`，参考文件为 `sparse/spmm/arch35/spmm_kernel.cpp`。目标 ops-gnn base 为 `2a6922c46accd89a94aa464a36558c0efddeffc6`。实现源文件保留来源及许可证说明。

# 需求分析（required）

## 需求描述

使用 Ascend C 实现稀疏矩阵乘稠密矩阵，通过 PyTorch 扩展直接启动 NPU kernel。公开索引使用 int64，稠密输入支持前导 batch 维度，CSR 支持四种归约，COO 前端支持 sum。计算结果在 NPU 上产生，不将输入 Tensor 转到主机 CPU 执行 DUT 计算。

## 需求拆解

| 项目 | 设计内容 |
| --- | --- |
| 格式 | CSR 直接计算；COO 在 NPU 上排序并转换为 CSR |
| 归约 | sum/add、mean、min、max |
| 边权 | 无边权、每边标量、每边 K 个逐特征权重 |
| batch | mat 为 `[...,N,K]`，前导维度展平为 batch |
| 精度 | 7 种输入输出组合；浮点累加 fp32，整数输出累加 int32 |
| 前端 | torch_sparse COO 接口、CSR 注册接口、SparseTensor、扩展 helper |
| 反向 | 浮点 sum/mean 对 mat 和 value 的一阶梯度 |
| 校验 | shape、dtype、设备、CSR 结构、COO 原始边权长度及可选元数据一致性 |
| 性能 | 避免 E×K 中间消息 Tensor；按场景分派通用路径和四特征路径 |

# 详细设计（required）

## 算子分析

### 数学公式

对 batch b、目标行 r、特征 k，定义边消息：

```text
message[b,e,k] = weight[e,k] * mat[b,col[e],k]
e ∈ [rowptr[r], rowptr[r+1])
```

无边权时 `weight=1`；标量边权在特征维复用；逐特征边权读取 `value[e,k]`。sum 对该行边消息求和，mean 除以该行边数（重复边计入边数），min/max 对消息逐特征取最值。

### 接口与形状

导入 `torch_sparse_npu` 后注册以下入口：

```text
torch_sparse.spmm(index, value, m, n, matrix) -> out
torch_sparse::spmm_sum(row, rowptr, col, value, colptr, csr2csc, mat) -> out
torch_sparse::spmm_mean(row, rowptr, col, value, rowcount, colptr, csr2csc, mat) -> out
torch_sparse::spmm_min/max(rowptr, col, value, mat) -> (out, arg)
SparseTensor.spmm/matmul(dense, reduce) -> out
csr_spmm(rowptr, col, value, mat, reduce="sum", *, output_dtype=None)
```

| 参数 | 形状 | 类型与约束 |
| --- | --- | --- |
| index | `[2,E]` | int64，COO 行/列范围有效 |
| rowptr | `[M+1]` | int64，首项 0、末项 E、单调不降 |
| col / row | `[E]` | int64，col 范围 `[0,N)`；已提供 row 需与 rowptr 一致 |
| rowcount | `[M]` | int64，等于 rowptr 相邻差 |
| colptr / csr2csc | `[N+1]` / `[E]` | int64，描述同一稀疏结构的转置排列 |
| value | 缺省，或首维 E | 与 mat 同 dtype；其余维度展平后每边 1 或 K 个权重 |
| mat | `[...,N,K]` | 稠密输入，允许非连续布局 |
| out / arg | `[...,M,K]` | out 按精度表；min/max 的 arg 为 int64 边位置 |

输入位于同一 NPU。前端先校验原始 value，再重排或进入 SparseTensor 路径，避免多余边权被丢弃或非法 dtype 被隐式转换。helper 在需要反向时构造元数据；直接注册接口要求提供当前梯度路径所需的元数据。已提供的元数据即使在 no_grad 下也进行检查。

### 支持数据类型

| 输入 | 输出 | 累加 |
| --- | --- | --- |
| fp32 | fp32 | fp32 |
| fp16 | fp16 / fp32 | fp32 |
| bf16 | bf16 / fp32 | fp32 |
| int8 | int32 / fp32 | int32 / fp32 |

公开接口默认保持浮点 dtype，int8 默认输出 int32。扩展 helper 的 `output_dtype` 显式选择混合输出，同时保持原有公开注册接口签名。

## 算子实现

### Host 侧设计

Python 负责前端和 Autograd 适配，C++ Host 负责结构检查、连续化、输出分配和类型分派。Host 向 Kernel 传递 `SpmmShape`：M/N/K/E、展平 batch 数、边权宽度及归约编号。输出仅在 min/max 路径附带 arg。

Host 从实际平台信息取得 AIV 核数。通用路径每 block 使用 512 个 SIMT 线程，四特征路径使用 256 个线程，以 grid-stride 循环覆盖输出元素或四元素组。偏移及最终步长均可安全表示时采用 int32 内部索引，否则采用 int64；公开索引存储仍为 int64。

在连续化和分配入队之后获取 PyTorch 当前 NPU stream；该接口会排空 torch_npu 主机分发队列，从而让直接 kernel launch 与前置数据操作保持顺序。

### Kernel 侧设计

通用路径由一个线程负责一个输出元素，读取所属 CSR 行的边并融合边权乘法与归约。相邻线程对应相邻特征，以利于连续特征读取。线程完成该行后一次写回结果，无需对输出做原子累加。

四特征路径让一个线程计算四个相邻特征，在同一边遍历中复用 rowptr、col 和标量边权。适用条件集中为：

```text
reduce 为 sum/mean
K % 4 == 0
batch * M * K >= 1,000,000
展平偏移和 grid stride 均可安全使用 int32
```

其余形状、min/max 和非整齐宽度使用通用路径。K 或特征组数为 2 的幂时，用移位与掩码实现商/余数；batch=1 时省去 batch 与行号拆分。sum/mean 在指定累加类型中计算，完成后转换输出类型。

此方案不分配 E×K 消息 Tensor，融合前向 CSR kernel 的 workspace 为 0。API 层按需分配输出/arg、连续副本、校验状态、COO 排序/转换和反向元数据；这些分配在内存测量中保留。

### COO 转换与结构校验

COO 前端校验原始 index/value 后，检查 NPU 行索引范围，按行稳定排序并同步重排 col/value，通过 Ascend C 二分查找转换 kernel 生成 rowptr。之后进入 CSR 校验和计算。int64 排序由 torch_npu 分派，可能使用设备侧 AiCPU。

设备校验检查 CSR 首尾项、单调性、边界和列索引，仅向 Host 返回小规模状态标量。验证缓存通过弱引用、参数位置、Tensor 身份、版本计数及 N 判断复用条件；原地修改使缓存失效，无版本计数的 inference Tensor 每次验证。

### 一阶反向

sum 的边权梯度为 `dvalue[e,k] = sum_b grad[b,row[e],k] * mat[b,col[e],k]`；标量边权还需沿 k 求和。mean 额外除以原目标行的边数。专用 Ascend C kernel 用 fp32 累加后转换到输入 dtype。

稠密梯度转换为转置稀疏图上的 SpMM，通过 colptr 和 csr2csc 描述反向邻接。mean 先按原行度数缩放上游梯度。共享边权的梯度跨所有前导 batch 维累加。

## 支持硬件

Ascend 950PR，编译目标 `dav-3510`，验证栈为 CANN 9.1.0、Python 3.12.13、torch 2.7.1、torch_npu 2.7.1.post8。扩展仅面向该架构构建。

## 算子约束限制

- 输入为稀疏矩阵与稠密矩阵；COO 公开入口支持 sum。
- value 每边支持 1 或 K 个权重，其他广播形状报错。
- 空行当前输出 0，min/max 的 arg 为 E；相等最值保留第一条边。
- min/max 当前忽略 NaN，接受首个非 NaN（含 Inf）；全 NaN 行返回 0/E。
- 整数 mean 向零截断，整数输出的累加范围为 int32。
- min/max backward 和二阶梯度显式报错；sum/mean 一阶梯度用于浮点输入。
- 绕过 PyTorch 版本计数的外部原始指针写入后，需调用 `clear_validation_cache()`。

# 可维可测分析

## 精度标准/性能标准

| 标准 | 验证方法 | 来源 |
| --- | --- | --- |
| 浮点精度 | CPU FP64 Golden；至少 99% 有限值满足混合容差，同时检查逐元素最大误差、shape、dtype 及特殊值 | 任务书与测试比较器 |
| 整数精度 | int8→int32 逐元素 EXACT | 任务书 |
| 性能 | 固定/泛化分别计算 Event、Kernel 的逐 case GPU/NPU 比值算术平均，四项分别 ≥0.3 | 任务性能要求 |
| 内存 | 按实际 API 输入统计 batch、输出及 arg，测量冷/热 peak 与额外分配；单列 forward workspace | 任务内存要求 |

浮点比较参数按输出 dtype：fp16 的 rtol/atol 均为 2^-9，bf16 均为 2^-6，fp32 分别为 2^-10/2^-16。基础最大绝对误差分别为 0.1/1.0/0.01，逐元素最大界取 `max(基础界, 32×ULP)`；NaN/Inf 位置及 Inf 符号单独比较。

精度测试使用确定性 CPU 输入生成器，保存种子和输入哈希。性能通过公开 registered API 执行，每 case 先验精度，10 次预热、30 次 Event 计时，预热后采集 1 次调用并累计 `kernel_details.csv` 的 `Duration(us)`。首次同步调用主机时间单列，COO 转 CSR 时间另记。

### 已有开发验证

以下结果来自实现提交 `472a07e52567828f0a610f208e27e5c3d78b9775` 的 950PR 实测。性能比使用附件 GPU CSV 中的耗时除以 NPU 实测耗时；附件对应 4 条固定用例和 196 条泛化用例，另外测试 4 条空图。

| 项目 | 结果 |
| --- | --- |
| 单元、PyG 和回归 | 394/394，无失败、无跳过 |
| CPU FP64 精度 | 200+112 条通过 |
| ATK 精度 | 200+28 条通过 |
| AscendOpTest 文件比较与任务容差 | 28/28 |
| 附件 CSV 对应性能 | 4 固定 +196 泛化，全部记录 |
| 空图补充性能 / 冷热内存 | 4 条 / 200+4 条 |
| 固定 Event / Kernel 平均比 | 0.406556 / 0.310550 |
| 泛化 Event / Kernel 平均比 | 0.408132 / 0.577960 |
| 源码与 trace 对应 | 33 个源文件哈希、204 条 kernel 明细已复核 |

### 复现步骤

在上述依赖已安装、CANN 环境已加载的 950PR Linux 环境中，进入实现源码目录执行以下命令。

```bash
set -euo pipefail
MAX_JOBS=2 python -m pip install --no-build-isolation --no-deps -e .
export SPMM_RESULTS="$(mktemp -d "$PWD/results-review.XXXXXX")"

python -m pytest -q tests/test_spmm.py tests/test_pyg.py tests/test_hardening.py \
  --junitxml="$SPMM_RESULTS/pytest.xml"
python tests/run_cases.py --cases tests/cases/spmm_precision_200.json \
  --output "$SPMM_RESULTS/precision_200.json"
python tests/run_cases.py --cases tests/cases/supplemental_112.json \
  --output "$SPMM_RESULTS/supplemental_112.json"
python tests/run_cases.py --cases tests/cases/gpu_baseline.csv \
  --api registered --performance --kernel --output "$SPMM_RESULTS/performance_200.json"
python tests/run_cases.py --cases tests/cases/empty_performance_4.json \
  --api registered --performance --kernel --output "$SPMM_RESULTS/empty_4.json"
python tests/run_memory.py --cases tests/cases/gpu_baseline.csv \
  --api registered --output "$SPMM_RESULTS/memory_200.json"
python tests/run_memory.py --cases tests/cases/empty_performance_4.json \
  --api registered --output "$SPMM_RESULTS/memory_empty_4.json"
```

精度/性能 JSON 核对 `requested = completed = passed`，内存 JSON 核对 `requested = completed = successful`，数量分别对应用例文件。性能结果内读取 `case_type`、`event_ratio` 和 `kernel_ratio`，在固定和泛化集合内分别求算术平均；保留同名 `_trace` 目录中的原始 kernel CSV。

ATK 使用原始与扩展用例文件，NPU 与 CPU Golden 配置位于 `nodes_accuracy.yaml`：

```bash
bash tests/atk/run_accuracy_atk.sh fixed
(
  cd tests/atk
  atk task -c accuracy_extended_28.json -n nodes_accuracy.yaml \
    --task accuracy -p execute_torch_sparse_spmm.py
)
```

日志中 `task_output_path` 指向 ATK 输出目录，报告的执行成功和精度通过数应分别为 200/200、28/28。AscendOpTest 使用可用源码目录作为参数：

```bash
python tests/run_ascendoptest_compare.py \
  --ascendoptest /path/to/AscendOpTest \
  --output "$SPMM_RESULTS/ascendoptest_28"
```

核对 `verify_result.json` 的 `total_cases = passed_cases = 28`，保留 actual/golden 二进制文件。此步骤复用文件比较模块，不使用 ACLNN 项目生成器。

## 兼容性分析

导入扩展前后，上游 CPU 调用保持原行为；NPU 注册沿用 torch_sparse 原签名。混合输出通过 helper 参数选择。扩展需在导入 torch_sparse/PyG 前端函数别名之前加载。

PyG 2.8 对非 CUDA SparseTensor 的默认优化会进入原生 sparse CSR mm。适配层将 NPU SparseTensor 的 SpMM 工具函数及 GCN/GIN 已捕获别名接到扩展。集成测试覆盖预处理邻接的 GCN/GIN 单层前向和梯度；邻接归一化和自环插入在调用 SpMM 前完成。
