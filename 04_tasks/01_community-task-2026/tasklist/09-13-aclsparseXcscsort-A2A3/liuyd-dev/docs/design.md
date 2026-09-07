# aclsparseXcscsort 算子开发（A2/A3）设计文档

# 需求背景（required）

## 需求来源

CANN 2026 年 9 月社区任务：面向 Atlas A2 训练系列产品与 Atlas A3 系列产品（DAV_2201，`arch22`）
完善 `aclsparseXcscsort` CSC 纯索引排序接口，并合入 ops-sparse 仓 master 分支。

## 背景介绍

### aclsparseXcscsort 算子现状分析

`aclsparseXcscsort` 对齐 cuSPARSE Legacy API `cusparseXcscsort`，对 CSC 格式稀疏矩阵的每一列
执行行索引原地稳定升序排序，并用相同的排列同步重排置换向量 `P`；列偏移数组 `cscColPtr` 不变。

任务开始时，ops-sparse 仓中该算子的状态：

| 项 | 状态 |
| --- | --- |
| 公开头文件 `include/cann_ops_sparse.h` | 两个接口已声明 |
| `docs/zh/api_list.md` | 已有接口文档 |
| `sparse/cscsort/arch35/` | 已有 Ascend 950 实现（host + kernel + tiling，约 670 行） |
| `test/cscsort/arch35/` | 已有 48 个 GTest 用例 |
| `sparse/cscsort/README.md` | 明确标注「Atlas A2 / A3：不支持」 |
| `sparse/cscsort/arch22/` | **不存在** |

因此本任务不是新增接口，而是为 arch22 补齐一份实现，使同一组公开接口在 A2/A3 上可用。

### arch35 现有实现依赖的能力

arch35 的 kernel 建立在两项 Ascend 950 专属能力上：

1. `Sort<int32_t, false, SortConfig{RADIX_SORT}>`：直接以 int32 为排序键、输出稳定升序。
2. SIMT 编程模型（`simt_api/asc_simt.h`、`__simt_vf__`、`asc_vf_call`、`asc_syncthreads`）：
   长列在 GM 上做 bottom-up merge-path 多线程归并。

这两项在 arch22 上均不可用，详见「详细设计 → 算子分析 → 硬件能力实测」。

# 需求分析（required）

## 需求描述

在 Atlas A2 训练系列产品与 Atlas A3 系列产品上，用 Ascend C 实现 CSC 每列内 row index 的
原地稳定升序排序，并同步重排 `P`，满足 `sortedVal[i] = origVal[P[i]]`。
接口签名、两阶段调用流程、workspace 语义、index base 处理与错误码均与既有实现保持一致。

## 需求拆解

1. 复用现有 Legacy API 与矩阵描述符，不新增同名或同功能接口。
2. 对每列区间 `[cscColPtr[j] - base, cscColPtr[j+1] - base)` 稳定排序 `cscRowInd`，`P` 同步重排。
3. `bufferSizeExt` 按 `nnz` 精确计算 workspace 并检查溢出。
4. 覆盖空矩阵、空列、`nnz = 0/1`、重复索引稳定性、单长列、多核与多 run 边界。
5. Host、Kernel、UT/ST 与文档统一交付，核心计算不得回退到 CPU。
6. A2/A3 与 A5（arch35）的公共排序逻辑解耦，互不影响，并完成交叉回归。
7. 性能：P-01/P-02/P-03 三个大模型维度锚点均不低于标杆接口 GPU 设备 Event 耗时的 0.25 倍。

# 详细设计（required）

## 算子分析

### 数学定义

对第 `col` 列，记 `base` 为 index base：

```
begin = cscColPtr[col] - base
end   = cscColPtr[col + 1] - base
cscRowInd_out[begin:end] = stable_sort(cscRowInd_in[begin:end])
P_out[begin:end]         = P_in[stable_permutation(begin:end)]
```

「稳定」指行索引相等时保持原有相对顺序。调用方通常把 `P` 初始化为 `0..nnz-1`，
排序后 `P[i]` 即输出位置 `i` 对应的原始非零元下标。

### 支持数据类型

纯 I32 索引接口。`cscColPtr`、`cscRowInd`、`P` 均为 `int32_t`。
本算子不含 values 数组，不涉及 float16 / bfloat16 / float32 / complex64 等计算 dtype。

### 支持形状

任意合法 CSC 结构。`m`、`n`、`nnz` 非负；`cscColPtr` 单调不减且
`cscColPtr[n] - base == nnz`；换算后的行索引位于 `[0, m)`。

### 硬件能力实测

设计前在 Ascend910_9382 上用 `bisheng --npu-arch=dav-2201` 逐项编译验证，
结论直接决定了 kernel 的技术路线：

| 探针 | 结果 | 依据 |
| --- | --- | --- |
| `Sort<int32_t, uint32_t, false, SortConfig>` | 编译失败 | `SortConfig` / `SortType` 在 2201 上未声明 |
| `Sort<float, isFullSort>`（proposal 链路） | 可用 | `kernel_operator_proposal_intf.h` |
| `Concat` / `Extract` / `MrgSort` | 可用 | 同上 |
| `asc_bitsort`（指令级 Sort32） | 可用，key 仅 half/float | `c_api/vector_compute/vector_compute.h` |
| `TopK<float, true>` | 可用 | `lib/sort/topk.h` 守卫含 2201 |
| SIMT (`asc_vf_call` 等) | 不可用 | 无 2201 实现 |

根因：`asc/include/adv_api/sort/sort.h` 与 `tikcfw/lib/sort/sort.h` 中整个 Sort 家族被
`#if __NPU_ARCH__ == 3510 || 5102 || 3003 || 3113` 守卫包住，**2201 不在列表内**。
需要注意的是，host 侧的 `GetSortMaxMinTmpSize` 与 `SortConfig` 结构体在
`sort_tiling_intf.h` 中确实存在（host 代码不区分架构），但设备侧没有对应实现，
只看 host 头文件会得出错误结论。

进一步在真机上实测了 arch22 排序原语的行为，这是稳定性设计的依据：

- `asc_bitsort` 为**降序**，且键相等时保持输入顺序（稳定）。
- `asc_mrgsort4` 跨队列归并时，键相等取靠前队列（稳定）。
- 排序的 tie-break **只看元素位置，不比较 index 字段**。
  用打乱的 `P` 作为 payload 验证：结果与 CPU `stable_sort` 逐元素一致。

这些实测结论中，前两条（降序、跨队列取靠前）被用于确定排序方向，
第三条（等键只看位置）**没有被用于保证稳定性**：该行为不是文档承诺的契约，
把正确性建立在它上面是不稳妥的。本方案改为让块内每个键唯一，从编码上消除等键，
稳定性因此与硬件的 tie 行为无关。

## 算子实现

### 实现方案

#### 1. Host 侧设计

**接口与校验。** 两个 `extern "C"` 接口的参数校验、错误码与 arch35 保持一致：
handle 空指针返回 `ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR`；维度为负、空矩阵带非零 nnz、
必需指针为空、index base 非法、workspace 未 128 字节对齐，均返回
`ACL_SPARSE_STATUS_INVALID_VALUE`。

**workspace 查询。** `nnz == 0` 时置 0 并返回成功；否则为 `2 * nnz * sizeof(int32_t)`，
并在乘法前用 `numeric_limits<size_t>::max() / kFactor` 判溢出。
该大小与 arch35 一致，保证同一份调用代码在两种硬件上行为相同。

**分核策略。** 以累计 `nnz` 为权重把**完整列区间**分配给各核：
第 `id` 个核负责元素区间 `[nnz*id/coreNum, nnz*(id+1)/coreNum)`，
列边界由 kernel 内对 `cscColPtr` 二分定位，不拆分单列。
启动核数取 `min(AIV 核数, min(n, nnz))`，避免大量空列矩阵启动无实际工作的 AIV。
各核只访问自己列区间对应的 `cscRowInd` / `P` / workspace，互不重叠，无需核间同步。

**UB 切分。** 单块排序在 UB 上同时存活的缓冲区为每元素 40 字节
（rowInd 4 + P 4 + keyBase 4 + key 4 + index 4 + outP 4 + sorted 8 + Sort 共享 tmp 8；
`Extract` 的 value 输出复用 keyBase，`Gather` 的 rowInd 输出复用 key，二者此时均已消费完毕），
另需为长列归并阶段常驻的 6 个暂存区预留固定空间。由 UB 容量反推 32 对齐的 `runSize`。
实测两台机器 UB 均为 196352 字节，`runSize` 为 4064。

**tilingkey 规划。** 本算子无 dtype 与算法分支，不需要 tilingkey。
host 侧额外向 kernel 传两个由 `m` 推导的量：

- `rowSpan = m + indexBase`：行号字段的取值个数。
- `maxSegLen = min(runSize, 2^24 / rowSpan)`：单次 UB 排序的最大元素数。
  第二项来自单段路径的键宽预算（该路径无列号字段，预算退化为 `rowSpan * segLen <= 2^24`），
  它同时决定了「多长的列必须走长列归并路径」。

`m` 越大，`maxSegLen` 越小，长列被切得越碎、GM 归并轮数越多。这是唯一键编码的固有代价，
换来的是稳定性不依赖硬件行为。

#### 2. Kernel 侧设计

kernel 只有 `Init` 与 `Process` 两个阶段。`Process` 内按列长分派到两条路径。

**排序键：三段唯一编码**

arch22 的排序指令是降序的，且键只能是 float。本方案把三个字段编码进一个 float：

```
key = jTerm * rowSpan * posSpan + (maxRow - row) * posSpan + (len - 1 - pos)
      └ 列号字段，随列号递减 ┘    └ 行号字段，随行号递减 ┘   └ 位置字段，随位置递减 ┘
```

其中 `rowSpan = m + indexBase`（行号字段的取值个数），`posSpan` 取块内最长列，
`jTerm = colCount - 1 - j`。三段的取值区间互不重叠，且都随
（列号，行号，列内位置）的字典序递减，因此**降序排序的结果就是
列升序、列内行升序、同行按原始位置升序**——正是稳定升序的定义。

关键性质是：`(列号, 行号, 列内位置)` 三元组在块内唯一，故**键唯一，不存在等键**。
排序结果完全确定，**与硬件对等键的处理方式无关**。稳定性由编码给出数学保证，
而不是建立在「实测发现硬件等键保序」这类未被文档承诺的行为上。

排序 payload 走块内原始下标，排完用两次 `Gather` 依据下标同时重排 `cscRowInd` 与 `P`。

**路径一：连续分块排序（`len <= maxSegLen`）**

这是短列与中等长度列的主路径，也是性能的关键。

一次 `DataCopyPad` 载入覆盖若干**整列**的连续区间，按上式构键后整块一次全排序。
由于块内元素本就按列分组，排序后各列自然成段、段内有序，用 `Gather` 拆回即可。

收益是把 GM 访存从「每列 4 次 `DataCopyPad`」降到「每块 4 次」。

**键宽预算**：键最大值小于 `colCount * rowSpan * posSpan`，必须落在 float32 可精确表示的
`2^24` 以内。攒块时逐列检查该预算，且 `posSpan` 随加入的最长列增大而收紧，
因此块内列数是动态决定的，不是固定值。

**键表的铺设**：列在块内的起始偏移任意，而向量指令要求 32 字节对齐的起址，
所以键表不能整块用向量指令生成。做法是把前两段字段等价改写成 `keyBase[j] = B_k - j`，其中

```
B_k = (colCount - 1 - k) * rowSpan * posSpan + (len_k - 1) + offset_k
```

对第 `k` 列是一个常数。键表因此等于「分段常数数组减去全局等差数列」，两部分都能向量化：

- 常数段够长时，用一条 `Duplicate` 铺掉该列内部 32 字节对齐的整块，
  只有跨列边界、凑不满一块的零头逐元素标量写。
- 列太短时一条向量指令的固定开销不划算，整列退回标量写；但此时写的是同一个常数，
  两个 `int32` 可以合成一次 64 位写，写次数减半。
- 等差数列复用排序 payload 已有的 `ArithProgression`，整块只多一条 `Sub`。

有一处精度约束必须守住：`B_k` 中途最多超出 `2^24` 一个 `offset`，
落到 float 上就不再是精确整数，因此整段留在 int32 域，`Cast` 成 float 放在减法之后。
减完的结果与逐元素铺表逐位相同。

单段路径（长列切段）只有一列、不含列号字段，起址天然对齐，
直接用 `ArithProgression` 生成递减序列。

**标量访问一律走裸指针**：键表铺设、`cscColPtr` 读取与长列归并的内层循环仍有逐元素的标量访问，
`GetValue` / `SetValue` 的封装开销在这种循环里占比很高。改用 `__ubuf__` / `__gm__` 裸指针后
语义与同步要求都不变。攒块与铺表沿列扫描时，相邻列共用一个列指针端点，
把端点留在标量寄存器里滚动，每前进一列只读新出现的那一个，这两段的 GM 标量读减半。

**路径二：长列 GM 归并（`len > maxSegLen`）**

先按 `maxSegLen` 把该列切成若干段，每段在 UB 内单独排序；
再在 GM 上做自底向上二路归并，`width = maxSegLen, 2*maxSegLen, ...`，
原数组与 workspace 交替充当源和目的。
归并阶段直接比较 int32 行索引，不经浮点；等键取靠前 run，保持稳定。
归并通过 UB 暂存分块进出，GM 访问全部走 `DataCopyPad`，不做标量 GM 读写。

两处使这条路径不必付多余代价：

- **趟数奇偶决定分段排序的落点。** 归并趟数 `ceil(log2(len / maxSegLen))` 开工前即可算出。
  趟数为奇数时让分段排序直接把结果写进 workspace，乒乓就正好停在原数组上，
  省掉最后整列的一次 GM 往返回拷。
- **队头连同 `P` 一起缓存在标量寄存器里。** 归并内层每出一个元素只刷新推进的那一侧，
  不重复读没有动过的一侧，UB 标量访问从每元素五次降到四次。

**补位元素处理。** UB 缓冲区按 32 元素对齐，尾部补位元素铺成 `INT32_MAX`，
取负后成为最小键，排序后落在块尾被忽略。
这里有一个必须显式处理的硬件行为：`DataCopyPad` 往 UB 写入以 32 字节块为粒度，
块内多出来的元素由 `paddingValue` 决定，**不显式指定就会被清零**，
把预先铺好的哨兵冲掉，使补位元素被当作「行 0」参与排序。
因此载入时须设置 `DataCopyPadExtParams{true, 0, rightPadding, kCscsortPadRow}`。

#### 3. 与 arch35 的解耦

两套实现分别位于 `sparse/cscsort/arch35/` 与 `sparse/cscsort/arch22/`，
由根 `CMakeLists.txt` 的 `get_soc_arch_dirs()` 按 `SOC_VERSION` 二选一编译，
互不引用、互不包含。公共部分是公开头文件、矩阵描述符、平台查询工具
（`sparse/common/aclsparse_host_utils.h`）与 workspace 大小定义。
测试同理放在 `test/cscsort/arch22/`，由 `cmake/test.cmake` 按架构选择。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas A2 训练系列产品 | √ |
| Atlas A3 训练系列产品 / Atlas A3 推理系列产品 | √ |
| Ascend 950PR / Ascend 950DT | √（沿用既有 arch35 实现） |

## 算子约束限制

- `m >= 0`、`n >= 0`、`nnz >= 0`；`m == 0` 或 `n == 0` 时 `nnz` 必须为 0。
- `cscColPtr` 单调不减，`cscColPtr[n] - base == nnz`；行索引位于 `[base, base + m)`。
- `P` 由调用方分配并初始化，不是 workspace。
- `pBuffer` 必须来自 `bufferSizeExt` 且 128 字节对齐，不得与输入输出重叠。
- **arch22 特有限制**：`m` 不得超过 `16777216`（2^24），超过返回
  `ACL_SPARSE_STATUS_NOT_SUPPORTED`。原因是该架构没有 int32 键的排序指令，
  排序键须以 float32 参与比较，而 float32 尾数只有 24 位。arch35 无此限制。
- 键宽预算还会随 `m` 增大压缩单次 UB 排序的段长（`maxSegLen = min(runSize, 2^24 / (m + base))`），
  `m` 很大时长列会被切得更碎、GM 归并轮数增加，属性能特性而非功能限制。
- 本接口异步执行，读取结果前须同步 handle 绑定的 stream。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | 单标杆，与 CPU Golden（逐列 stable_sort）exact match；行索引、置换向量与列内有序性三项全部逐元素相等 | 《生态算子开源精度标准》 |
| 性能标准 | P-01/P-02/P-03 均不低于标杆接口 GPU 设备 Event 耗时的 0.25 倍 | 任务书 3.3 节 |
| 内存标准 | 无等价 GPU 调用范围，按方案固有 workspace 不超过目标硬件 L2 Cache 容量验收 | 任务书 3.4 节 |

实测结果（详见自测报告）：

| 项目 | Atlas A2 (910B3) | Atlas A3 (910_9382) |
| --- | --- | --- |
| C++ UT/ST | 48/48 通过 | 48/48 通过 |
| 泛化精度用例 | 200/200 exact | 200/200 exact |
| 随机差分用例 | 60000/60000 exact | 60000/60000 exact |
| P-01 base0 median / 倍率 | 991.8 us / 2.99x | 745.7 us / 3.98x |
| P-02 base0 median / 倍率 | 491.8 us / 5.31x | 338.2 us / 7.72x |
| P-03 base0 median / 倍率 | 781.9 us / 3.56x | 572.7 us / 4.86x |
| 峰值 workspace | 4194304 B，占 L2 的 2.08% | 同左 |
| Profiler Task Type | AI_VECTOR_CORE 328/328，AI_CPU 0 | AI_VECTOR_CORE 328/328，AI_CPU 0 |

门槛为标杆的 0.25 倍，十二个 P case（含 base 0/1）全部达标。
性能数据由任务包官方脚本 `benchmark_sparse_ops_npu.py` 采集，warmup 10 次、采样 30 次，
分母是任务书给出的标杆接口 GPU 设备 Event `median_us`。
任务书只要求提交 A3 的性能结果，A2 一并采集作为补充。

随机差分用例是本地自研的 C++ 差分测试，与 CPU `stable_sort` 逐元素对拍，
覆盖 base 0/1、重复行、空列、单长列、打乱的 `P` 与多核切分边界，两台机器各 60000 例。
全部证据均按代码版本 `a1b2bfeda345258267f340ace0df1733` 采集。

## 流水同步的自验

Ascend C 的 `SetFlag<A_B>` 在 A 流水置位、`WaitFlag<A_B>` 在 **B 流水**等待，
所以一个方向的标志只挡住一条流水。本实现有两处必须显式补标志，
`-O0` 构建下标量代码慢到能把它们掩盖，`-O2` 下会直接算错：

| 位置 | 冒险 | 只有 `MTE3_MTE2` / `PipeBarrier` 为何不够 | 补的标志 |
| --- | --- | --- | --- |
| 长列归并搬出结果后 | 调用方立刻用标量写覆盖同一块输出缓冲，而搬出可能还在读它（WAR） | `MTE3_MTE2` 只让 MTE2 流水等，标量单元照跑 | `MTE3_S` |
| 分块路径清零后铺键表 | 清零是向量写，铺表是标量写，落在同一块 UB（WAW） | `PipeBarrier<PIPE_V>` 只在向量流水内部排序 | `V_S` |

两处都已在 `CMAKE_BUILD_TYPE=Release` 下复验：两台机器各 48/48 GTest 通过、
各 60000 例随机差分 exact。**只在默认的 Debug 构建下验证等于没有验证这一类问题**，
因此本实现的验收自验固定跑两种构建类型。

## 兼容性分析

- 接口签名、参数顺序、返回码与 workspace 大小均未改变，对既有调用方无影响。
- **arch35 源码零改动**，两套实现通过构建系统按 SOC 二选一，不会同时参与编译。
  **交叉编译验证**（在 Ascend910_9382 机器上以 `--soc=ascend950` 交叉编译，
  编译参数为 `--npu-arch=dav-3510`，非本机的 `dav-2201`）：构建系统选中
  `SOC_ARCH_DIRS=arch35`，实际送编译的是 `cscsort/arch35/` 的 host 与 kernel，
  测试目标选中 `arch35/cscsort_test.cpp`，链接产出的 `libops_sparse.so` 导出
  Xcscsort 符号 2 个，arch22 目录未被卷入。
  **该验证只覆盖编译期，不覆盖运行期**：二进制从未在 950 硬件上执行，
  arch35 的运行时精度与性能本次没有任何实测数据，因为手上只有 A2 与 A3。
  其能证明的是「新增 arch22 目录不会污染 arch35 的编译路径」，
  不能证明「arch35 在 950 上仍然算得对」。后者需在有 950 算力时补做，
  或由后合入方按任务书「PR 申请合入」第 3 条处理。
- **两个跨架构共享的文档有改动，需 A5 侧一并确认**：
  `sparse/cscsort/README.md` 的产品支持情况由「A2/A3 不支持」改为「支持」，
  算法说明由原先的单节拆成「公共部分 / Ascend 950（arch35）/ Atlas A2、A3（arch22）」三节。
  拆分时 arch35 的原有表述逐条保留，包括 `Sort<int32_t>` 的 RADIX_SORT 稳定升序、
  以单元素有序段为起点的 bottom-up SIMT merge-path 归并、
  以及「归并阶段 `left.key <= right.key` 时选左侧」的稳定性保证。
  `docs/zh/api_list.md` 为纯新增，未删改任何原有内容。
- `OAT.xml` 补了 `*.tsv` / `*.png` / `*.json` 三项审计过滤，是收录 `test_cases/`
  数据文件所必需，属仓库级配置，与 A5 实现无关。
- 新增的 `m <= 2^24` 限制只在 arch22 生效，是该架构硬件能力的如实反映，已写入接口文档。
