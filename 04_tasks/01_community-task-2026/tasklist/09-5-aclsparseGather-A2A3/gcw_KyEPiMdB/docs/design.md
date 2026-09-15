# aclsparseGather 算子设计文档（A2/A3）

> 任务来源：昇腾 CANN 社区任务「9月社区任务-aclsparseGather算子开发(A2/A3)」
> 代码提交仓：https://gitcode.com/cann/ops-sparse （`master`）
> 个人代码分支：https://gitcode.com/gcw_KyEPiMdB/ops-sparse （分支 `gather-arch22`）
> 目标硬件：Atlas A2 训练系列（910B3 / 910B4）、Atlas A3（DAV_2201，`arch22`）
> 参考语义：cuSPARSE `cusparseGather`（13.3 Update 1）、PyTorch `torch.index_select` / `aten::index_select`

---

# 1. 需求背景（required）

## 1.1 需求来源

昇腾 CANN 社区 2026 年 9 月算子开发任务，算子名 `aclsparseGather`，面向 A2/A3 平台。任务要求参考 cuSPARSE `cusparseGather` 语义，使用 C++ Host 与 Ascend C Kernel 完成算子开发并合入 `ops-sparse`，同时交付 Python/ATen 适配，适配 PyTorch 2.7 及以上、torch_npu 26.0.0 及之后版本，公开入口为 `torch.index_select(input, 0, index)`，不得 CPU fallback。

## 1.2 背景介绍

大模型推理中的词表查找 / embedding 取行等场景，本质是按一组索引从稠密向量中聚集元素。cuSPARSE 在 Generic API 体系下提供 `cusparseGather`：以稀疏向量的索引数组为输入，从稠密向量取值并原地写入稀疏向量的 values。NPU 侧此前没有与之对齐的稀疏向量聚集能力，本任务补齐该能力的 A2/A3 实现，并打通 PyTorch 侧入口。

计算公式：

$$X.values[i] = Y\big[X.indices[i] - idxBase\big],\quad i \in [0, nnz)$$

## 1.3 现有实现现状分析

`ops-sparse` 现状：

- `sparse/gather/arch35/` 已有 A5（950）实现：kernel 采用 SIMT 编程模型（`__simt_vf__` + `asc_vf_call`，每线程处理一个元素），支持 float/float16/bfloat16/double 与 32I/64I 索引，不支持 complex64；`sparse/gather/README.md` 中 A2/A3 标注为不支持。
- 公共层已提供 Handle（`aclsparseContext`，携带 stream）、DnVec/SpVec 描述符与创建/销毁接口、`aclsparse_host_utils.h` 中的核数与 UB 查询、`CHECK_RET` / `CHECK_ACL` 等工具。
- `sparse/scatter/arch22/` 提供了 arch22 的矢量编程范式参考（`KernelXxx` 类 + TPipe/TQue + `DataCopyPad`，Host 分核 + tiling 随启动参数下发）。

关键结论：**arch35 的 SIMT 实现无法在 arch22 复用**（A2/A3 不支持 SIMT），必须在 arch22 重新用 Ascend C 矢量编程实现；同时任务要求的 dtype 集合（fp16/bf16/fp32/complex64）与 arch35（fp16/bf16/fp32/double）不同，complex64 为新增必选能力。

---

# 2. 需求分析（required）

## 2.1 算子原型

```c
aclsparseStatus_t aclsparseGather(
    aclsparseHandle_t handle,
    aclsparseConstDnVecDescr_t vecY,
    aclsparseSpVecDescr_t vecX);
```

| 参数 | 方向 | 说明 |
| --- | --- | --- |
| `handle` | 输入 | aclsparse 句柄，携带调用方 stream；空句柄返回参数错误 |
| `vecY` | 输入 | 稠密源向量 Y，只读；一维连续，dtype 须与 `vecX.values` 一致 |
| `vecX` | 输入/输出 | SpVec 描述符：读 `indices`、写 `values`（原地输出），`indices` 不被修改 |

## 2.2 需求描述

1. 支持 dtype：`float16`、`bfloat16`、`float32`、`complex64`（`vecX.values` 与 `vecY` 必须一致）。
2. 索引：I32（`ACL_SPARSE_INDEX_32I`），index base 0/1；支持乱序与重复索引。
3. 只更新 `vecX.values`，不修改 `vecY` 与 `vecX.indices`；未声明的重叠返回参数错误。
4. 沿用调用方 stream 异步执行；无 workspace、无 preprocess；`nnz=0` 成功返回且不启动 Kernel。
5. 精度：逐元素 bit-wise exact（重复执行结果一致）。
6. 禁止 CPU fallback。
7. Python/ATen：`torch.index_select(input, 0, index)` → `aten::index_select` 的 NPU 注册，任何不支持组合显式报错。

## 2.3 需求拆解

1. 公开 C 接口在 `include/cann_ops_sparse.h` 中已存在，保持源代码兼容，仅补充 arch22 能力说明（不在本任务修改原型）。
2. arch22 Host：参数校验、分核、tiling 计算、stream 异步下发。
3. arch22 Kernel：Ascend C 矢量实现，覆盖四种 dtype、乱序/重复索引、tile 尾块与动态规模。
4. Python/ATen：ATen Dispatcher 注册、dense Tensor 与 SpVec 描述符转换、输出构造、异常与 stream 语义对齐。
5. 测试：C++ UT/ST、ATen/Python 端到端 UT、官方测试包的精度/性能/内存用例。

---

# 3. 详细设计（required）

## 3.1 Host 侧设计

文件：`sparse/gather/arch22/gather_host.cpp`、`gather.h`、`gather_kernel.h`

1. **参数校验**：handle / vecY / vecX 空指针；`vecX.valueType != vecY.valueType`；dtype 是否属于 {fp16, bf16, fp32, complex64}；`idxType` 是否为 I32；`idxBase` 是否为 0/1；`vecY.nums >= vecX.size`；`size`、`nums` ≤ INT32_MAX，`nnz` ≤ UINT32_MAX。校验顺序与返回码与 arch35 保持一致（dtype 不匹配/不支持 → `NOT_SUPPORTED`，长度约束与空指针 → `INVALID_VALUE`）。
2. **早退路径**：`nnz=0` 直接成功返回、不启动 Kernel；indices/values 的空指针检查放在该早退之后，保证零长稀疏向量可用空指针。
3. **分核**：`GetAivCoreCount()` 取 AIV 核数并按 `GATHER_MAX_CORE_NUM`（64）钳制，`blockNum = min(coreNum, nnz)`，余数分配到前若干核，使各核元素数相差不超过 1。
4. **tiling**：`tileNn` 取本核最大元素数并按 32 元素对齐（保证 indices 的 4B buffer 与 values 的 2/4/8B buffer 长度均为 32B 整数倍），上限 4096；`GatherTilingData` 携带 `nnz / blockNum / tileNn / yElemNum / idxBaseShift / valType` 与 per-core `offset / count` 数组。
5. **下发**：tiling 随 Kernel 启动参数传递（Host 不额外分配 Device 内存），使用 `handle->stream` 异步下发，不新增 Host 同步。

## 3.2 Kernel 侧设计

文件：`sparse/gather/arch22/gather_kernel.cpp`

1. **结构**：`KernelGather<ValT>` 类（`Init` / `Process`）+ `TPipe` / `TQue`；入口为 `__global__ __vector__ gather_custom<ValT>`，启动器 `gather_kernel_do` 按 `valType` 分发容器类型。
2. **数据类型处理**：按位宽等价容器类型搬运——fp16/bf16 → `uint16_t`，fp32 → `uint32_t`，complex64 → `uint64_t`。本算子只做数据搬移、不做数值运算，因此输出与输入的位模式完全一致，bit-wise exact 是结构上保证的，而不是靠精度对齐；同时规避了 Host 侧无法构造 `__bf16` 类型的编译器限制。
3. **搬运流程**：每核把 indices 按 tile 经 MTE2 搬入 UB（深度 2 的 DoubleBuffer，处理当前 tile 时预取下一 tile）；对 tile 内每个元素按 `pos = indices[j] - idxBaseShift` 做单元素 MTE2 取 Y 值并写入输出 tile 的 UB 缓冲；最后整 tile 经 MTE3 一次性写回 `X.values`。
4. **索引乱序 / 重复**：逐元素随机访问天然正确；重复索引重复写入相同值，不影响 bit-wise 一致性与确定性。
5. **尾块**：输出按 tile 实际元素个数（而非对齐长度）拷贝，`DataCopyPad` 的字节数参数天然处理非对齐尾块。
6. **性能取舍**：逐元素取 Y 是基础版本。实测（§5.2）显示该方案已满足验收要求；若后续需要进一步提升，可按顺序优化：① 索引连续段聚合为整块搬运；② 使用 Ascend C 的批量 `Gather` 类接口；③ 调整 tile 与核数配比。

## 3.3 Python/ATen 适配设计

目录：`sparse/gather/torch_extension/`（`gather.py`、`csrc/gather.cpp`）+ `torch_extension/cann_ops_sparse/docs/zh/gather.md`

1. **注册**：
   - `aten::index_select` 的 `PrivateUse1` 实现：`torch.index_select(input, dim, index)`，仅支持 `dim=0`；`dim != 0` 在 C++ 层以 `TORCH_CHECK` 显式报错，不回退 CPU 或其他实现。
   - 测试 hook `ops_sparse_test::gather_npu(Tensor values, Tensor indices, int base) -> Tensor`：社区任务测试包与 ATK 执行器要求的入口，schema 由本扩展定义（无对应标准 ATen schema，且需显式索引基址参数）。
2. **C++ 桥接**（`csrc/gather.cpp`）：校验 NPU 设备、同设备、一维、连续、I32 索引（`index_select` 的 int64 索引在设备侧转 int32）、index base 与 dtype；`DeviceGuard` 先于输出申请生效；使用 `AclSparseContext` 缓存的 Handle 与当前 NPU stream；描述符 RAII 释放；执行路径不做任何同步。
3. **输出语义**：输出为新分配的 `[nnz]` 连续张量，dtype/device 与输入一致，不共享 storage；`values` 与 `indices` 只读。
4. **异常语义**：不支持的 dim、非连续输入、非 I32/I64 索引、非法 base、不支持的 dtype、非 NPU 输入均稳定抛 `RuntimeError`，错误信息包含参数名与实际值。

## 3.4 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas A2 训练系列产品（910B3 / 910B4） | √ |
| Atlas A3 训练系列产品 / Atlas A3 推理系列产品 | √ |
| Ascend 950PR / Ascend 950DT | 由 arch35 路径支持（本任务未修改） |

## 3.5 算子约束限制

- arch22 索引仅支持 I32；传入 64I 返回 `NOT_SUPPORTED`。
- arch22 dtype 仅支持 fp16 / bf16 / fp32 / complex64，不支持 `ACL_DOUBLE`（arch35 支持）。
- `size`、`vecY.nums` ≤ INT32_MAX，`nnz` ≤ UINT32_MAX，`nnz <= size`。
- 索引越界（`indices[i] - idxBase` 不在 `[0, size)`）属调用方前置条件，设备端不做同步 D2H 检查；越界访问为未定义行为。
- `torch.index_select` 适配仅支持 `dim=0`、一维连续输入。

---

# 4. 可维可测分析

## 4.1 精度标准 / 性能标准 / 内存标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | 逐元素 bit-wise exact match；CPU Golden 为基准（fp16/bf16 用 fp32，fp32 用 fp64，complex64 用 complex128），重复执行结果一致 | 生态算子开源精度标准 |
| 性能标准 | 性能倍率 = GPU 标杆 Event 耗时 / NPU 同调用范围耗时；P-01/P-02/P-03 每个有效 case ≥ 0.25 | 任务书 §3.3 |
| 内存标准 | 输入输出 >500 MB 时 NPU 额外内存 ≤ GPU 的 50%；方案固有 workspace 绝对值 ≤ 目标硬件 L2 Cache | 任务书 §3.4 |

## 4.2 测试设计

| 层级 | 位置 | 覆盖内容 |
| --- | --- | --- |
| C++ UT/ST | `test/gather/arch22/` | 23 条 CSV 用例（四种 dtype × base 0/1 × 随机/顺序/逆序/重复/首尾/等距索引 × 尾块 × nnz=0/1）+ 异常返回码；输入只读性、重复执行一致性、P-01/02/03 性能用例 |
| Python/ATen UT | `test/gather/test_torch_extension.py` | 48 条：两条入口的功能与精度、int32/int64 索引、非默认 stream、dim≠0 / 非连续 / 非法 base / 不支持 dtype / CPU 输入等异常路径 |
| 官方测试包 | `test_cases/` | 200 条精度用例、224 条性能与内存用例 |

## 4.3 兼容性分析

- 公开 C 接口与 arch35 保持一致，`GatherTilingData` 等内部结构按 arch 目录隔离，A2/A3 与 A5 的源码集合由构建系统按 SOC 自动选择，不共存于同一构建。
- A2/A3 与 A5 的 PR 先后合入时，后合入方基于已合入版本处理公共 Host 冲突并完成交叉回归。
- `torch.index_select` 注册仅在导入本扩展后生效，且只覆盖 `PrivateUse1` 的 `dim=0` 路径。

---

# 5. 实测结果（Atlas A2 910B4 + CANN 9.1.0）

环境：910B4（HBM 32GB）、CANN 9.1.0、torch 2.9.0、torch_npu 2.9.0.post8。

## 5.1 精度

- 官方 `accuracy_cases.json` 200 条用例按 ATK 等价语义执行（环境无 ATK，按 `function_sparse_ops.py` 入参规则与 `sparse_gather_exact` 精确比对标准）：**200/200 通过**。
- C++ 测试 24/24、Python 端到端 48/48 通过。

## 5.2 性能

官方 `benchmark_sparse_ops_npu.py`，预热 10 次、采样 30 次，与官方 GPU 基线逐 case 对比（224/224 匹配）：

| 范围 | 用例数 | 性能倍率 |
| --- | --- | --- |
| 全部 | 224 | 0.365 ~ 0.792（中位 0.563） |
| P-01（size=128256, nnz=8192） | 8 | 0.415 ~ 0.782 |
| P-02（size=151936, nnz=4096） | 8 | 0.487 ~ 0.573 |
| P-03（size=129280, nnz=7168） | 8 | 0.487 ~ 0.542 |

低于 0.25 的用例数：0。纯 Kernel 调用范围的 Device Event 实测为 P-01 52.9–54.1μs、P-02 33.5μs、P-03 47.8μs。

## 5.3 内存

无 workspace、无 preprocess；官方 benchmark 同次执行记录 `extra_peak_allocated_bytes` 最大值为 NPU 66048 bytes、GPU 131072 bytes，NPU 额外分配不超过 GPU。

## 5.4 Profiler 证据

`torch_npu.profiler`（CPU + NPU）采集 3 次调用，`kernel_details.csv` 中恰好 3 个 device kernel、无其他算子：

```
0,_ZN12_GLOBAL__N_113gather_customIjEEvPhS1_S1_16GatherTilingData,...,42.722,...,40
```

即 `(anonymous namespace)::gather_custom<unsigned int>(...)`，`Device_id=0`、`Block Num=40`；`operator_details.csv` 中无 `index_select` 记录，确认无 CPU fallback。

## 5.5 未执行项与原因

| 项 | 原因 | 替代验证 |
| --- | --- | --- |
| 官方 1000 条泛化精度用例（ATK） | 环境未预装 ATK，CANN 镜像与镜像源均无该工具 | 用官方 200 条精度用例（同入参规则、同精确标准）替代 |
| 910B3 与 A3 型号验证 | 环境仅提供 910B4 | 同属 DAV_2201/arch22，编译 SOC 与运行环境一致；待具备对应型号时补充 |
| GPU 侧基线重采 | 环境无 GPU | 采用任务包给定的官方 `baseline_results` |

---

# 6. 交付件

| 序号 | 交付件 | 位置 |
| --- | --- | --- |
| 1 | 算子设计文档 | 本文档（`cann-ops-competitions`：`04_tasks/01_community-task-2026/tasklist/09-5-aclsparseGather-A2A3/gcw_KyEPiMdB/docs/design.md`） |
| 2 | 自测用例及测试代码 | `ops-sparse`：`test/gather/arch22/`、`test/gather/test_torch_extension.py` |
| 3 | 自测报告 | 精度/性能/内存/Profiler 数据见 §5，明细报告随验收提交 |
| 4 | 待验收代码地址 | https://gitcode.com/gcw_KyEPiMdB/ops-sparse 分支 `gather-arch22`（已邀请 `Ascend-CANN` 为开发者） |
