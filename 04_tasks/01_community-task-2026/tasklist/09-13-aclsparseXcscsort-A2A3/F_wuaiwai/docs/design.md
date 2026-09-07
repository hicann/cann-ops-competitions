# 需求背景（required）

## 需求来源

本设计对应社区任务“aclsparseXcscsort 算子开发（A2/A3）”，依据任务书
`aclsparseXcscsort_A2A3_task_doc.md` 和社区设计模板编写。交付目标是在
`cann/ops-sparse` 的 `master` 分支提供 Legacy C++ 接口、Host、Ascend C Kernel、C++
UT/ST 及 README；设计文档提交至 `cann-ops-competitions` 对应任务目录。

## 背景介绍

`cusparseXcscsort` 用于对 CSC 矩阵每一列的 row index 做稳定升序排序，并返回与排序结果
一致的置换向量，供调用方同步重排 values。`aclsparseXcscsort` 沿用这一接口语义，面向
纯索引场景：输入是 I32 的 `cscColPtr`、`cscRowInd` 和调用方维护的 `P`，不携带 values、
compute dtype、转置或算法枚举。

目标设备 Atlas A2/910B3 和 Atlas A3/910_93 均通过 DAV_2201（`arch22`）执行同一套排序
实现。该实现针对短列吞吐、超宽列可扩展性和稳定性做了统一设计，并由 Host 根据 SOC 路由
到对应的算子注册入口。

# 需求分析（required）

## 需求描述

实现以下公开接口，并保持 Legacy C++ API 的描述符、index base、workspace 和 stream 语义：

```cpp
aclsparseStatus_t aclsparseXcscsort_bufferSizeExt(
    aclsparseHandle_t handle, int m, int n, int nnz,
    const int *cscColPtr, const int *cscRowInd,
    size_t *pBufferSizeInBytes);

aclsparseStatus_t aclsparseXcscsort(
    aclsparseHandle_t handle, int m, int n, int nnz,
    const aclsparseMatDescr_t descrA,
    const int *cscColPtr, int *cscRowInd, int *P, void *pBuffer);
```

对第 `j` 列定义：

```text
begin = cscColPtr[j] - indexBase
end   = cscColPtr[j + 1] - indexBase
```

算子输出满足：

```text
cscRowInd_out[begin:end] = stable_sort(cscRowInd_in[begin:end])
P_out[begin:end] = P_in[同一稳定排列]
```

`cscColPtr` 保持只读，`cscRowInd` 和 `P` 原地更新。`P` 由调用方初始化，常见初始化方式
为 `P[i] = i`；算子只负责将该向量按 row index 的稳定排列同步重排。

## 需求拆解

1. 在 A2/910B3 和 A3/910_93 上完成 DAV_2201 NPU dispatch，核心排序在 Ascend C
   Kernel 中执行。
2. 支持 I32 索引、`ACL_SPARSE_INDEX_BASE_ZERO` 和 `ACL_SPARSE_INDEX_BASE_ONE`，覆盖
   空矩阵、空列、`nnz=0/1`、已排序、逆序、重复 row、单长列及多核边界。
3. 对超过 UB 单次处理能力的列支持多 run 排序和多轮稳定归并，保证任意合法列长均可处理。
4. `bufferSizeExt` 精确返回 `2 * nnz * sizeof(int32_t)`，并在 Host 侧检查尺寸计算溢出。
   执行阶段使用足量、128 字节对齐且与输入输出不重叠的 workspace。
5. 保持 handle 绑定 stream 的异步提交语义，提供 native ACL、CPU Golden exact、ATK、
   性能、内存和 Profiler 验证材料。
6. 公共接口和排序语义与现有 Legacy 能力保持一致；A2/A3 采用同一 arch22 算法和数据
   结构，SOC 差异由构建和注册路由处理；公共 API 语义通过 arch22/arch35 交叉回归。

# 详细设计（required）

## 算子分析

### 数学公式

对每列的局部位置 `k = 0 ... (end - begin - 1)` 构造比较记录：

```text
record(k) = (row = cscRowInd[begin + k], position = k, payload = P[begin + k])
```

按 `(row, position)` 升序排列记录，并将 `row` 和 `payload` 作为绑定字段写回。硬件 Sort
热路径将其映射为
`key = (segMaxRow - row) * 2^posBits + (len - 1 - position)`，再执行降序排序；第二关键字
明确表达稳定性，不依赖硬件 Sort 对相等 key 的实现细节。`P` 的数值只作为 payload 搬运，
不参与排序比较；当 key 位宽超过 float32 精确范围时切换稳定归并。

### 支持数据类型

- `cscColPtr`、`cscRowInd` 和 `P` 均为 I32（`int32_t`）。
- 本接口是纯索引操作，不涉及 values、compute dtype、标量参数或浮点容差。
- `indexBase` 支持 `ACL_SPARSE_INDEX_BASE_ZERO` 和 `ACL_SPARSE_INDEX_BASE_ONE`。

### 支持形状

- `m`、`n`、`nnz` 均为非负整数，覆盖 0/1 边界；`m==0` 或 `n==0` 时 `nnz` 为 0。
- `cscColPtr` 长度为 `n+1`，`cscRowInd` 和 `P` 长度为 `nnz`。
- base=0 时首尾列指针为 `0` 和 `nnz`；base=1 时为 `1` 和 `nnz+1`。
- 合法输入的列指针单调递增，换算后的 row index 位于 `[0, m)`；任意列长度均支持。

## 算子实现

### 实现方案

#### 3.2.1 Host 侧设计

Host 实现位于 `sparse/cscsort/arch22/cscsort_host.cpp`，职责是完成接口契约校验、运行时
资源规划和 Kernel 异步下发：

1. 校验 handle、维度、描述符、index base 及必要的 Device 指针。`nnz==0` 直接返回成功，
   不产生无效 Kernel 任务。
2. 在 `bufferSizeExt` 中以安全算术计算 `2 * nnz * sizeof(int32_t)`；执行接口校验 P、
   workspace 非空、128 字节对齐，以及 workspace 与 `cscColPtr`、`cscRowInd`、`P` 已知存储
   区间不重叠。
3. 不同步读取 Device 上的列指针和 row index。列指针单调性、端点和 row 范围由调用方按
   CSC 接口前置条件保证，从而维持异步 stream 的可组合性。
4. 根据运行时 AIV 核数、UB 容量、`n` 和 `nnz` 生成 tiling。启动核数受 AIV 核数、列数和
   nnz 共同约束；kernel 再以累计 nnz 偏移定位列边界，给每个核分配连续的完整列区间，
   空区间在 kernel 侧快速返回。
5. 将 `m`、`n`、`nnz`、index base、`runSize`、core 数及 workspace 布局编码到 tiling，
   保证 kernel 不需要额外的 Host 往返。

#### 3.2.2 Kernel 侧设计

Kernel 位于 `sparse/cscsort/arch22/cscsort_kernel.cpp`，编译目标为 `dav-2201`，采用
Ascend C vector/AIV 完成以下数据路径：

1. **短列快速路径**：将一列搬入 UB，生成包含 row 和局部原始位置的唯一排序 key，使用
   arch22 可用的硬件 Sort 完成排序，再依据排序下标同步 Gather row 和 P。key 采用单调
   映射以实现 row 升序；当 `m` 和位置范围可在 float32 中精确表示时走该热路径。
2. **精确性回退路径**：当合成 key 超出 float32 的精确整数范围，或运行时无法获得可靠的
   key 宽度时，使用 UB 内稳定两路归并。归并比较 `(row, position)`，因此与 CPU stable-sort
   的结果完全一致。
3. **短列批处理**：按相同 padded 槽宽聚合短列；32 元素槽位采用独立组排序，其余槽宽在
   key 中编码列隔离信息。批处理只在各列自己的槽位内生效，不跨列交换记录。
4. **长列路径**：将列按 `runSize` 切分为多个有序 run，run 结果写入 workspace，再在原
   数组与 workspace 之间执行 bottom-up 双路稳定归并。归并按 UB chunk 流式处理，列长不受
   单次 UB 容量限制；每轮合并后交换输入/输出缓冲区，必要时将最终结果拷回原数组。
5. **数据一致性**：row 和 P 始终以二元组同步搬运、排序和写回；尾块通过 `DataCopyPad`
   处理，避免非 32B 对齐列边界造成越界或相邻列写冲突。
6. **并行模型**：每个核只访问自己负责的完整列区间，核间不共享中间排序状态，不需要全局
   barrier；MTE2、Vector 和 MTE3 通过事件依赖形成搬入—计算—写回流水。

算法空间复杂度为 `O(nnz)`（公开 workspace），单列时间复杂度为 `O(L log L)`；长列的
归并阶段将排序工作分摊到多个 UB-sized run，避免为极端列长引入额外的动态内存。

#### 3.2.3 测试设计

C++ UT/ST 覆盖接口返回码、描述符生命周期、I32/base0/1、空列和 0/1 边界、原地输出、
workspace 对齐与重叠、异步 stream 及错误参数；CSC 接口不含 values、layout 或 stride
属性，对应维度按接口契约验证。CPU Golden 逐列执行 stable-sort，并同时校验
`cscColPtr`、`cscRowInd` 和 `P`。

### workspace 与 tiling

公开 workspace 口径固定为：

```text
workspace_bytes = 2 * nnz * sizeof(int32_t)
```

workspace 前半区存放 row index，后半区存放 P；调用方依据查询值分配并保证 128 字节对齐。
当 `nnz==0` 时查询结果为 0，执行接口不启动 kernel。workspace 为长列 run 和归并阶段提供
双缓冲，短列遵循同一 ABI，不引入额外动态临时区。tiling 中包含 `m/n/nnz/indexBase`、
`runSize`、`coreNum` 和 Sort 临时空间大小；`runSize` 同时受 UB 容量、DataCopyPad
对齐、单次 Sort 上限和双缓冲预算约束。

## 支持硬件

| 支持的芯片版本 | 实现路径 | 执行架构 |
| --- | --- | --- |
| Atlas A2 / Ascend 910B3 | `sparse/cscsort/arch22/` | DAV_2201 / arch22 |
| Atlas A3 / Ascend 910_93 | `sparse/cscsort/arch22/` | DAV_2201 / arch22 |
| Ascend 950 | 仓库既有 `arch35` 路由 | arch35 |

A2/A3 的 SOC 名称由 CMake 和算子注册表映射到同一 DAV_2201 kernel，公共 Legacy API
不因设备型号变化。

## 算子约束限制

- 仅支持 I32 CSC 纯索引接口；不支持 values、其他 dtype、CSR/COO、转置或算法枚举。
- `P` 必须由调用方初始化并作为输入输出使用；接口不创建 identity permutation。
- `cscColPtr` 只读；`cscRowInd` 和 `P` 原地更新。
- workspace 由 `bufferSizeExt` 查询，大小为查询值，地址 128 字节对齐，且不与输入输出区间
  重叠。
- API 采用异步 stream 语义；调用方在读取输出前同步 handle 绑定的 stream。
- 列指针单调性、端点与 base/nnz 一致性以及 row index 合法范围是调用方的 CSC 输入契约。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述（不涉及说明原因） | 标准来源 |
| --- | --- | --- |
| 精度标准 | `cscColPtr`、`cscRowInd`、`P` 与 CPU stable-sort Golden 逐项 exact match；重复 row 保持原相对顺序；相同输入重复执行 bit-wise 一致 | 任务书、生态算子开源精度标准 |
| 功能标准 | A2/A3 native ACL、I32、base 0/1、边界、workspace 和异步调用流程通过；Profiler 显示核心计算在 NPU dispatch | 任务书 |
| 性能标准 | 性能倍率 = GPU 设备 Event 调用耗时 / NPU 同调用范围总耗时；P-01/P-02/P-03 均不低于 0.25 倍 | 任务书 3.3 |
| 内存标准 | 固有 workspace 不超过目标硬件 L2；等价内存场景的 NPU 额外峰值不超过 GPU 的 50% | 任务书 3.4 |

### 性能采集口径

性能表使用固定的 P-01/P-02/P-03 CSC 维度和稀疏结构生成规则，两侧使用相同的
`cscColPtr`、`cscRowInd`、`P`、index base 和 descriptor。输入、P、descriptor 与 128B
对齐 workspace 在正式采样前预分配；warmup=10、samples=30，每轮在设备 Event 上计时并
同步 stream。计时范围不包含首次编译、数据生成、Host 到 Device 搬运和无关初始化。

倍率严格按任务书定义计算：GPU 标杆接口的设备 Event median 除以 NPU 同调用范围内所有
Kernel 的总耗时 median。NPU 表同时报告 p90，便于观察尾延迟；GPU 数据取任务书提供的同
一 workload 标杆结果。

### A2/910B3 实测结果

| 平台 | Case | CSC 维度 | nnz | NPU median (us) | NPU p90 (us) | GPU median (us) | GPU/NPU |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| A2/910B3 | P-01 | 8192×28672 | 524288 | 3568.03 | 3575.94 | 2961.70 | 0.83x |
| A2/910B3 | P-02 | 4096×1536 | 262144 | 729.99 | 749.30 | 2610.35 | 3.58x |
| A2/910B3 | P-03 | 7168×2048 | 458752 | 928.06 | 943.12 | 2780.82 | 3.00x |

原始 NPU 结果位于 `test_cases/aclsparseXcscsort_testCase/results_perf_npu_final_verified/`，
GPU median 取任务书提供的同 workload 标杆导出值。

### A3/910_93 实测结果

| 平台 | Case | CSC 维度 | nnz | NPU median (us) | NPU p90 (us) | GPU median (us) | GPU/NPU |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| A3/910_93 | P-01 | 8192×28672 | 524288 | 3572.34 | 3588.56 | 2961.70 | 0.83x |
| A3/910_93 | P-02 | 4096×1536 | 262144 | 733.11 | 748.62 | 2610.35 | 3.56x |
| A3/910_93 | P-03 | 7168×2048 | 458752 | 942.41 | 954.56 | 2780.82 | 2.95x |

原始 NPU 结果位于 `test_cases/aclsparseXcscsort_testCase/results_perf_npu_a3_final_verified/`。

六组实测倍率均满足任务书 `>=0.25x` 门槛。P-01 为超宽列场景，主要成本来自多 run
归并；P-02/P-03 的列长分布更适合短列批处理，因此 NPU 端到端耗时显著低于 GPU 标杆。
该差异与实现路径一致：性能优化重点是减少 UB/GM 往返、提高短列批处理占比，并在长列
归并中维持连续访存和双缓冲流水。

### 功能与资源验证摘要

- A2/910B3 native ACL：20/20 PASS；任务包 direct hook：600/600 exact；ATK：200/200 PASS。
- A3/910_93 native ACL：20/20 PASS；任务包 direct hook：600/600 exact；ATK：200/200 PASS。
- 三个性能 case 的 workspace 分别为 4 MiB、2 MiB、3.5 MiB；均低于本地 32 MiB L2
  验收口径，内存采样 `extra_peak_allocated_bytes=0`。
- Profiler 记录 `kernel_name=cscsort_kernel`、`kernel_type=AI_VECTOR_CORE`；A3 P-01
  同规模 profile 的 Block Num=40，核心计算由 NPU vector core 执行。

## 兼容性分析

兼容性由三层契约保证：

1. **设备路由**：`ascend910b*` 和 `ascend910_93` 统一映射到 DAV_2201/arch22，Ascend
   950 继续通过仓库既有 arch35 注册路径选择对应 kernel。
2. **API 语义**：公开函数、描述符生命周期、index base 换算、P 的输入输出属性以及
   `bufferSizeExt` 查询值与 Legacy API 保持一致；`cscColPtr` 只读且输出为原地更新。
3. **运行时语义**：workspace 采用固定双平面布局，执行前完成对齐和重叠检查；kernel 通过
   handle 绑定 stream 异步提交，调用方以 stream 同步作为结果可见性边界。

A2/A3 共用算法的原因是两款设备均采用 DAV_2201 指令集和 Ascend C 编程模型，差异由
SOC 路由、运行时核数及 UB 参数在 tiling 阶段吸收。代码、测试和 README 中的接口示例
与本设计保持同一 I32/base0/1/稳定排序契约；交叉回归以 arch35 的既有 Legacy API
行为作为对照，校验描述符、index base、P 同步重排和错误码语义不受 arch22 注册影响。

## 参考资料

- cuSPARSE `cusparseXcscsort` 接口文档：用于接口语义和稳定置换定义对照。
- Ascend C 算子开发文档：<https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/opdevg/Ascendcopdevg/atlas_ascendc_map_10_0002.html>
- Ascend C API 文档：<https://www.hiascend.com/document/detail/zh/canncommercial/850/API/ascendcopapi/atlasascendc_api_07_0003.html>
- 生态算子开源精度标准：<https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md>
- 社区任务设计模板：<https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md>
- 社区任务流程及提交注意事项：<https://gitcode.com/org/cann/discussions/39>
