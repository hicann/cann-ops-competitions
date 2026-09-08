# torch_sparse SpMM NPU 适配设计文档

## 1. 需求背景

### 1.1 需求来源

本需求来自 CANN 社区 Ascend 950PR `torch_sparse SpMM NPU 适配开发任务`。
目标仓库为 `cann/ops-gnn`，功能对标 `rusty1s/pytorch_sparse`，Kernel 设计
参考 `cann/ops-sparse` commit
`9a514470c2cc99e469c34a615048eb8b30a66694` 的 `sparse/spmm/arch35`。

### 1.2 背景介绍

PyG 中的消息聚合依赖 `torch_sparse` 的 COO/CSR SpMM。本文设计在
Ascend 950PR 上提供原生 NPU 实现，覆盖 sum/add、mean、min、max、批次、
任务书规定的 U1/M1-M6 精度模式，以及 sum/mean 对边权和稠密矩阵的反向传播。

实现位于 `experimental/torch_sparse_spmm/`，由 Python 前端、PyTorch
Dispatcher、C++ Host Launch/Tiling 和 `dav-3510` Ascend C Kernel 组成。
输入数据不得回退 CPU，也不修改 `ops-sparse`。

## 2. 需求分析

### 2.1 需求描述

支持接口：

```text
torch_sparse.spmm(index, value, m, n, matrix) -> out
torch_sparse::spmm_sum(row, rowptr, col, value, colptr, csr2csc, mat) -> out
torch_sparse::spmm_mean(row, rowptr, col, value, rowcount, colptr, csr2csc, mat) -> out
torch_sparse::spmm_min/max(rowptr, col, value, mat) -> (out, arg_out)
SparseTensor.spmm(dense, reduce) -> out
SparseTensor.matmul(dense, reduce) -> out
```

COO 接口固定为 sum；`add` 归一化为 `sum`。不支持 SparseTensor 与
SparseTensor 相乘，不要求 min/max backward。

### 2.2 需求拆解

1. 使用 Ascend C 实现 CSR sum/mean/min/max，并由 COO 前端在 NPU 内转换
   CSR 后复用；
2. 通过 PyTorch PrivateUse1 注册到 `torch_sparse`，导入
   `torch_sparse_npu` 后生效；
3. 支持 U1、M1-M6 dtype/累加组合、batch、空图、重复边和可选元数据；
4. 为 sum/mean 实现 `value`、`mat` Autograd，且所有 NPU 输入和计算不得
   回退 CPU；
5. 完成功能、精度、ATK、AscendOpTest、GPU/NPU 性能、内存、PyG 冒烟和
   仓库回归验证；
6. 交付源码、构建与使用说明、设计文档、自测报告和机器可读结果。

## 3. 详细设计

### 3.1 总体架构

```text
torch_sparse / SparseTensor API
            |
            v
torch_sparse_npu.frontend
  参数校验、COO 排序/CSR 转换、Autograd
            |
            v
torch_sparse Dispatcher (PrivateUse1)
            |
            v
C++ Host: dtype/reduction 选择、tiling、current stream
            |
            v
Ascend C SIMT Kernel (dav-3510)
```

`torch_sparse_npu.__init__` 先加载 `_C` 完成 schema 和 PrivateUse1 注册，
再安装仅针对 NPU tensor 的上游 Python hook。CPU/CUDA 调用保留上游实现。

### 3.2 数据与接口设计

CSR 元数据 `row/rowptr/col` 对外为 int64 NPU Tensor。Kernel 使用 int32
索引，Host 通过 NPU cast 得到连续 int32 临时 Tensor，不读取其 Host 值。
`rowptr` 为 `[M+1]`，`row/col/value` 首维为 `E`。`mat` 为
`[...,N,K]`，输出为 `[...,M,K]`，前导维展平为 batch count。

`value` 按 `torch_sparse 0.6.18` 的标量边权契约实现为 `[E]`。任务书参数表
同时写有 `[E,*]`，但公式、任务 case 和上游实现均未定义其广播规则；上游
`spmm` 明确要求 `value.dim() == 1`。因此不扩展未定义语义，二维及以上 value
按非法 shape 报错。

`value=None` 时仍以 `col` 作为 ABI 占位，但 tiling 的 `hasValue=0`，Kernel
直接使用标量 1，绝不把 `col` 内容当边权。

公开 ABI 没有 output dtype 参数。精度模式通过 `value/mat` dtype 组合推导：

| 模式 | value | mat | out | 累加 |
|---|---|---|---|---|
| U1 | fp32 | fp32 | fp32 | fp32 |
| M1 | int8 | int8 | int32 | int32 |
| M2 | int8 | fp32 | fp32 | fp32 |
| M3 | fp16 | fp32 | fp32 | fp32 |
| M4 | bf16 | fp32 | fp32 | fp32 |
| M5 | fp16 | fp16 | fp16 | fp32 |
| M6 | bf16 | bf16 | bf16 | fp32 |

### 3.3 COO 设备内转换

前端检查 index 为 `[2,E]` int64 NPU Tensor，按 row 稳定排序并同步重排
col/value，因此同一行内保留原 edge 顺序。随后在 NPU 上统计 row count，
再用 `cumsum` 构造 rowptr。空 COO 直接构造长度 `M+1` 的零 rowptr。索引
范围检查在 NPU 上归约，只把布尔标量传回 Host 以抛出异常；稀疏输入和
SpMM 计算不回退 CPU。

### 3.4 Host 与 Tiling

Host 只读取 Tensor 元数据，完成：

1. rank、dtype、shape、device 和可选元数据一致性校验；
2. 连续化及 int64 到 int32 的设备 cast；
3. 输出与 min/max `arg_out` 分配；
4. 填充 `M/N/K/E/batch/featureTiles/reduction/hasValue/dtype/coreCount`；
5. 获取 `c10_npu::getCurrentNPUStream(device).stream()` 并异步 Launch。

算子内部不创建、同步或销毁私有 ACL stream。测试和性能测量只在边界同步。

### 3.5 Forward Kernel

基础路径将工作划分为 `(batch,row,featureTile)`。每个 SIMT work item 负责
连续 8 个特征，遍历该行 `[rowptr[r],rowptr[r+1])`，完成一次写回。空行无需
特殊 kernel：空行对所有 reduce 输出 0；min/max 的 `arg_out` 保持 E。

min/max 同时维护 `(value, globalEdgeId)`。遍历顺序按 CSR edge position
递增，只有严格更小/更大时更新，因此相等值自然保留最低 edge id；空行
`arg_out=E`。

FP16/BF16 load 后转 FP32 计算与累加，M5/M6 写回时转换。M1 使用 int32
乘加；mean 使用 C++ 有符号整数除法，向零截断。

性能实测采用连续行/特征块分配，并按总工作量在 256/512 SIMT threads
之间自适应选择。四种 reduction 使用编译期特化，sum/mean 路径不携带
min/max 的运行时分支和 arg 状态。Host 对 CSR 内容校验、int64 到 int32
转换及可选转置元数据校验使用 TensorImpl、version、stream 和 N 作为
缓存键，输入原地修改会使缓存失效。

### 3.6 Autograd

sum/mean 使用自定义 `torch.autograd.Function`。对 edge `e=(r,c)`：

```text
scale[e] = 1                         (sum)
scale[e] = 1 / max(degree[r], 1)    (mean)
grad_value[e] = scale[e] * dot(grad_out[...,r,:], mat[...,c,:])
grad_mat[...,c,:] += scale[e] * value[e] * grad_out[...,r,:]
```

当 `mat` 需要梯度时，`grad_mat` 使用 `colptr/csr2csc` 做转置 CSR 遍历，
避免 Host 转换；此时缺失或不一致的转置元数据在 Launch 前报错。仅
`value` 需要梯度时不消费也不要求这两项元数据。`value=None` 不返回
value 梯度。

### 3.7 校验与异常

Python/Host 分层拒绝以下输入：

- COO 非 `[2,E]`、非 int64、负索引、越界或负 M/N；
- CSR rank/dtype 错误、rowptr 首项非零、非单调、末项非 E 或 col 越界；
- value 长度/dtype 不合法，或 value/mat dtype 组合不支持；
- mat rank 小于 2、N 不一致或跨设备；
- rowcount、colptr、csr2csc 形状或映射不一致；
- 非 sum/add/mean/min/max reduction。

涉及输入内容的检查由 NPU 算子完成，仅将最终布尔标量物化到 CPU 用于
报错；CSR/COO 数据与实际计算均不回退 CPU。

## 4. 支持硬件与约束

### 4.1 支持硬件

| 支持芯片 | Kernel 架构 | 状态 |
|---|---|---|
| Ascend 950PR | `dav-3510` | 支持 |

### 4.2 算子约束限制

- 仅支持 NPU Tensor；CPU/CUDA 行为继续由上游 `torch_sparse` 提供；
- 对外 CSR/COO 索引必须为 int64，Kernel 内部转换为 int32；
- `value` 遵循 `torch_sparse 0.6.18`，仅支持标量边权 `[E]`；
- min/max 不提供 backward；SparseTensor 与 SparseTensor 相乘不在范围内；
- 不依赖 `aclsparseSpMM*`，不修改 `ops-sparse`。

## 5. 可维可测分析

### 5.1 测试与验收

测试采用 RED-GREEN 循环，覆盖 schema 注册、CSR/COO、重复列、空行、
`K=1/513`、batch、U1/M1-M6、min/max edge id、非法输入、value=None、
sum/mean weighted-loss backward 和 PyG GCN/GIN smoke。

AscendOpTest 主入口面向 IR JSON、部署后的 ACLNN API 和 FrameworkLaunch
可执行文件，与本任务 PyTorch 扩展直接 Launch Kernel 的工程模式不同。验收
适配器只复用其可独立调用的输入生成、CPU Golden 与结果比较模块，DUT 仍是
`import torch_sparse_npu` 后注册到 `torch_sparse` 的 PrivateUse1 算子，不增加
第二条 ACLNN 生产路径。由于通用随机生成器不能表达 rowptr 单调及端点等关联
约束，适配器先落盘显式 CSR 结构，再由 AscendOpTest 生成独立 value/mat。
只有 requested/generated/executed/compared/passed 全部相等且 failures 为空时
才返回成功，工具控制台文本不作为验收证据。

验收运行任务包固定功能测试、200 条精度 case、ATK fixed/generalized、固定
性能 case、泛化性能 case 和仓库回归。浮点对 CPU FP64 Golden 使用混合
容差，M1 逐元素精确比较。

任务包提供的 `spmm_performance.json` 实际为 80 条；另保留由其生成器
确定性产生的 200 行文件。该文件及配套 GPU CSV 实际分组为 4 fixed + 196
generalized，且不含任务书列出的 `E=0` 性能 case；任务书正文则写为 4 fixed
之外另有 200 generalized。因此当前 200 行比较用于证明随附 GPU/NPU 包的
同 workload 性能，不能替代缺失的四条 generalized GPU 测量或空图基线。
GPU baseline 脚本虽标注 power_law/hub/regular，实际对三者都生成 uniform
row；比较时用显式的 baseline-compatible 模式复现该输入，同时另行保留按
声明结构生成的结果。

接口耗时使用 NPU Event；Kernel duration sum 来自一次调用的 profiler trace；
COO 转换单列耗时。显存基线在输入构造后、首次算子调用前记录，因此算子创建
并持有的 CSR/int32/转置元数据缓存计入额外峰值；计时 warmup 独立执行。
失败/不支持 case 保留在请求集合和性能分母中。

### 5.2 精度与性能标准

| 验收项 | 标准 | 证据 |
|---|---|---|
| 浮点精度 | 任务书混合容差，matched ratio 不低于 99% | CPU FP64 Golden、ATK、AscendOpTest |
| M1 精度 | int8 到 int32 逐元素完全一致 | 固定/泛化精度 case |
| 性能 | fixed 与 generalized 的接口、Kernel 平均 GPU/NPU 比均不低于 0.3 | Event、profiler kernel CSV、严格比较 JSON |
| 内存 | 固有 workspace 不超过 950PR L2 | Kernel launch workspace 为 0 |

Profiler 报告按 case 保留接口 Event wall time、一次接口调用内全部 Kernel
duration 总和、COO 到 CSR 独立耗时、峰值已分配内存和 reserved 增长。所有
结果记录用例 ID、shape、dtype、reduce、结构与 seed，便于复现和定位回归。

## 6. 兼容性分析

该实现通过 PrivateUse1 增量注册，不替换上游 CPU/CUDA Kernel。Python hook
仅在输入为 NPU Tensor 时接管 `torch_sparse.spmm` 和 SparseTensor 稠密乘法；
其他设备路径保持原行为。公开接口与 `torch_sparse 0.6.18` 对齐，导入扩展前
不改变进程内任何 `torch_sparse` 行为。

## 7. 复用与许可证

行/特征 SIMT 组织参考 `cann/ops-sparse` commit
`9a514470c2cc99e469c34a615048eb8b30a66694` 的
`sparse/spmm/arch35`。本工程不调用 `aclsparseSpMM*`。新增代码遵循仓库的
CANN Open Software License Agreement Version 2.0。
