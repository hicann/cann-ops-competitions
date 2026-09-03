# aclsparseXcscsort 算子（Atlas A2/A3）设计文档

| 项目 | 内容 |
| --- | --- |
| 算子名称 | aclsparseXcscsort / aclsparseXcscsort_bufferSizeExt（Legacy C++ API） |
| 目标硬件 | Atlas A2 训练/推理系列（arch22，DAV-2201）、Atlas A3 训练/推理系列 |
| 代码仓 | cann/ops-sparse：`sparse/cscsort/arch22/`、`test/cscsort/arch22/` |
| 对齐接口 | cuSPARSE Legacy API `cusparseXcscsort` / `cusparseXcscsort_bufferSizeExt` |
| 状态 | 已上板验证：C++ UT 44/44；官方测试包精度 213/213 exact；性能三锚点全部 ≥ 0.25× GPU 标杆 |

# 需求背景（required）

## 需求来源

社区任务 2026 稀疏算子任务包「aclsparseXcscsort（A2/A3）」。ops-sparse 仓库已有 `sparse/cscsort/arch35/`（Ascend 950PR/DT）实现，Atlas A2/A3 缺失对应能力，需按同一公开 C++ 接口补齐 arch22 实现。

## 背景介绍

行索引有序的 CSC 矩阵是 SpMV/SpMM、稀疏三角求解等上层算子走快速路径的前提，排序属于稀疏格式规范化的基础算子。cuSPARSE 以 `cusparseXcscsort` 提供该能力：对每列行索引原地稳定升序排序，并以同一稳定置换同步重排置换数组 `P`（调用方预填 identity），据此可同步重排值数组 `sortedVal[i] = origVal[P[i]]`。

### 接口语义

```text
begin = cscColPtr[col] - indexBase        # indexBase ∈ {0,1}，来自 descrA
end   = cscColPtr[col + 1] - indexBase
cscRowInd_out[begin:end] = stable_sort_asc(cscRowInd_in[begin:end])
P_out[begin:end]         = P_in[同一稳定置换]
cscColPtr                = 只读，不变
```

纯 I32 索引算子：无 values 输入、无 compute dtype；`cscColPtr` / `cscRowInd` / `P` 均为 `int32`。NPU 输出须与 CPU Golden（逐列 I32 稳定升序）**exact match**，不得仅检查有序性，不得以 CPU fallback 冒充 NPU Kernel。

### arch22 实现约束（探针实测结论）

| 约束 | 内容 | 来源 |
| --- | --- | --- |
| 排序原语键类型 | 旧式 `Sort<T, isFullSort>`（vbitsort 32 元素/迭代 + vmrgsort4 四路归并树）仅支持 half/float 键；int32 键版本（RADIX_SORT）仅在 950PR/DT | Sort.md + 上板编译期证伪（探针 P1） |
| 排序方向 | 纯降序 | 探针 P3 |
| 单次全排上限 | repeatTime ∈ [0,255] × 32 = 8160 元素 | Sort.md |
| index 通道 | 输出每秩源下标（调用方预填 iota） | 探针 P2 |
| 输出布局 | dst/tmp 各 2N float（(score,index) 8B 交织对），`Extract` 解交织在 2201 可用 | 探针 P4 |
| 稳定性 | 旧式 Sort 无书面稳定承诺，稳定性不能依赖硬件 tie 行为 | Sort.md |
| float 精度 | float32 整数精确表示上限 2^24 | IEEE754 |
| SIMT | arch22 不支持，不可照搬 arch35 的 SIMT merge-path 归并 | 官方编程模型文档 |
| 跨流水一致性 | MTE3 写回后标量单元读 GM 须 `SetFlag/WaitFlag(MTE3_S)` | 探针 P5（长列归并缺该同步即出错） |

# 需求分析（required）

## 需求描述

使用 Ascend C 在 `sparse/cscsort/arch22/` 实现 aclsparseXcscsort 的 Atlas A2/A3 支持：公开接口、Host 校验与生命周期、Ascend C Kernel、C++ UT/ST 与性能脚本，合入 ops-sparse master。

## 需求拆解

1. **接口**：`aclsparseXcscsort_bufferSizeExt`（按 nnz 精确查询 workspace）+ `aclsparseXcscsort`（执行排序），签名与 cuSPARSE 一致；
2. **Host**：参数校验（空指针 / m,n,nnz 非负 / 空矩阵一致性 / indexBase 合法性 / pBuffer 128B 对齐）、workspace 计算、tiling 生成；
3. **Kernel**：每列稳定升序 + `P` 同步重排；覆盖空矩阵/空列、nnz=0/1、重复索引、单长列、多 run、index base 0/1、随机/有序/逆序分布；
4. **测试**：C++ UT（功能/边界/异常/白盒/性能）+ 官方测试包（`test_cases/`：ATK 精度、性能、内存三线）；
5. **性能**：三模型锚点对 GPU cuSPARSE 标杆 ≥ 0.25×——P-01 Llama3.1-70B `8192×28672, nnz=524288`（标杆 ≈2.96ms）；P-02 Qwen3-235B `4096×1536, nnz=262144`（≈2.60ms）；P-03 DeepSeek-V3 `7168×2048, nnz=458752`（≈2.78ms）；
6. **内存**：workspace ≤ L2 容量（2201 官方 Cache 表 192MB）。

# 详细设计（required）

## 算子分析

### 数学公式

核心是把「逐列稳定升序」转化为「单列内唯一 float 合成键的一次降序全排」。对列长 `len`、列内位置 `pos ∈ [0,len)`、行索引 `row`（原始 base 值）：

```text
rowNorm = m - 1 + indexBase - row          # 行号归一化：非负且随 row 单调减
posTerm = len - 1 - pos                    # 位置反转：同值时先出现者键大
key     = rowNorm × 2^posBits + posTerm    # posBits = BitWidth(len - 1)
```

性质与推论：

1. **单调性**：`row` 越小 ⇒ `rowNorm` 越大 ⇒ `key` 越大；`row` 相同时 `pos` 越小 ⇒ `key` 越大。降序排 `key` ⇔ 行号升序、同行按原始位置升序 = 稳定升序；
2. **精确性**：`key` 为非负整数，需 `BitWidth(m-1+indexBase) + posBits ≤ 24`（float32 尾数精确上限），Host 以 `KeyCap = 2^(24-rowBits)` 封顶 runSize 保证；kernel 侧同口径二次防御（超预算列跳过、数据不动）；
3. **唯一性**：列内 `(row,pos)` 二元组唯一 ⇒ `key` 唯一 ⇒ 排序结果**完全确定，稳定性不依赖硬件 tie 行为**（旧式 Sort 无稳定承诺亦不受影响）；
4. **pad 沉底**：对齐填充位键置 `-1.0f`，小于一切真实键（≥0），降序沉底且与真实键零同分。

### 支持数据类型

`cscColPtr` / `cscRowInd` / `P`：`int32`（纯 I32 索引算子）。kernel 内部计算键为 `float`（合成键），索引通道为 `uint32`（iota 位型复用 int32 iota，值非负等价）。

### 支持形状

任意 `m × n`、`nnz ≥ 0` 的 CSC 矩阵；单列长度无上限（`len > runSize` 的长列走分 run + GM 稳定归并路径）；列数、列长任意偏斜（多核按元素量均衡，空列/单元素列跳过）。

## 算子实现

### 实现方案

#### host侧设计：

**1. 分核策略：**

- 核数 `coreNum = min(GetAivCoreCount(), n, nnz)`，单 kernel `KERNEL_TYPE_AIV_ONLY` 一次下发 `blockDim = coreNum`；
- 各核分得元素区间 `[nnz·id/coreNum, nnz·(id+1)/coreNum)`（按累计 nnz 均衡，抗列长偏斜），经 `cscColPtr` 二分（`FindColBoundary`）折算为列边界；列不跨核拆分，核间 rowInd/P/workspace 访问区间互不重叠，**无核间同步**；
- 空核（区间落空列带）直接返回。

**2. 数据分块和内存优化策略：**

- runSize（单次 UB 全排的列长上限）三重封顶取 min：
  1. **UB 预算**：二分求最大 `runSize` 使 `align32(runSize) × 48B ≤ usable`（48B/元素 = 8 个 4B 数据段 + dstKey 2N 交织 8B + sort tmp 8B），所有段统一 32 元素粒度（vbitsort repeat 对齐，host/kernel 必须同口径）；
  2. **键宽预算**：`runSize ≤ KeyCap(m-1+indexBase) = 2^(24-rowBits)`；
  3. **硬件上限**：`runSize ≤ min(kMaxFullSortElems=8160, DataCopyPad 单块上限)`；
- workspace：`bufferSizeExt = align128(4·nnz·4B)`（统一申报口径；其中前 `2·nnz·4B` 为长列归并区 wsRow+wsP，其余为预留余量）；`nnz==0` 返回 0，`aclsparseXcscsort` 不下发 kernel。P-01 约 8.4MB ≪ L2 192MB。

**3. tiling 规划策略：**

`TilingData = {n, nnz, m, indexBase, runSize, sortTmpBytes, coreNum}` 单结构下发。kernel 按同口径由 `m-1+indexBase` 推 `rowBits` 做键宽二次防御；无 tilingkey 分支（单一路径，长短列在 kernel 内按 `len ≤ runSize` 动态分流）。

#### kernel侧设计：

每核处理自己列区间；`len ≤ 1` 跳过，`2 ≤ len ≤ runSize` 走 UB 全排，`len > runSize` 走长列路径。

**短列 UB 全排（三段流水，逐列）：**

```text
MTE2: DataCopyPad 载入 rowInd/P 段（blockLen=len×4B，32 元素对齐 pad）
V-1 : 构键：Duplicate(key,-1.0f,padded)；CreateVecIndex(iota)；
      Muls/Adds 生成 posTerm；Cast/Muls/Adds/Add 合成 key
V-2 : Sort<float,true>(dstKey, key, iota, tmp, repeat)   # 降序全排
      Extract(key, dstIdx, dstKey, repeat)               # 2N 交织 → (score, 源下标) 解交织
V-3 : ShiftLeft(dstIdx, 2)                               # 元素下标 → 字节偏移
      Gather(outRow, rowIndIn, dstIdx)；Gather(outP, pIn, dstIdx)
MTE3: DataCopyPad 写回原位（rowInd 与 P 同偏移）
      SetFlag/WaitFlag(MTE3_S)                           # 长列后续标量读 GM 的同步屏障
```

每列独立 `pipe_->Reset()` + 按需 InitBuffer，短列常驻 UB 占用 O(align32(len))；事件链 MTE2→V→MTE3→S 显式同步，避免跨流水读写竞态。

**长列路径（len > runSize）：**

1. 按 `runSize` 切 run，逐 run 走上述 UB 全排写回原位（每 run 键宽预算内）；
2. GM bottom-up 稳定归并：相邻 run 两两归并（`a.key ≤ b.key` 取 a，即相等取左 run ⇒ 稳定），log2(runCount) 轮在 GM 原数组与 workspace（wsRow/wsP）间 ping-pong；
3. 末轮若结果落在 workspace，一次性拷回 rowInd/P 原位（保持原地语义）。

标量 GetValue/SetValue 实现正确性优先；性能锚点（列长 ≤ 224）恒不触发该路径。

**A2 工程约束落地**（编译期/运行期实测）：

- `CreateVecIndex` 用 int32 版本（uint32 版内部 Adds 不被 A2 支持，非负值位型等价）；
- aicore 禁 float↔unsigned 变量直转，标量常数经有符号 int32 中转；
- `rowBits` 按 `BitWidth(m-1+indexBase)` 计算（`m=2^k ∧ base=1` 时 rowNorm 最大值需 k+1 位，防止键溢入符号域）。

## 支持硬件

| 硬件 | 支持 | 实现 |
| --- | --- | --- |
| Atlas A2 训练/推理系列（arch22 / DAV-2201） | ✅ | 本任务交付：float 合成键 + 旧式 Sort 全排 + 长列 GM 归并 |
| Atlas A3 训练/推理系列 | ✅ | 同 arch22 路径 |
| Ascend 950PR / Ascend 950DT（arch35） | ✅（既有） | `sparse/cscsort/arch35/`：`Sort<int32_t>` RADIX + SIMT merge-path 归并 |

注意：两 arch 的 `bufferSizeExt` 返回值不同——A2/A3 为 `align128(4·nnz·4B)`，950 为 `2·nnz·4B`；调用方必须以查询值为准分配 `pBuffer`（README 已注明）。

## 算子约束限制

**aclsparseXcscsort_bufferSizeExt：**

- `handle` 为空 → `ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR`；`pBufferSizeInBytes` 为空 → `ACL_SPARSE_STATUS_INVALID_VALUE`；
- `m,n,nnz < 0` → `ACL_SPARSE_STATUS_INVALID_VALUE`；`m==0 ∨ n==0` 时须 `nnz==0`；
- `n>0` 时 `cscColPtr` 非空；`nnz>0` 时 `cscRowInd` 非空；
- `nnz==0`：返回成功且 `*pBufferSizeInBytes = 0`；本接口纯 Host 计算，不引入 Device 同步。

**aclsparseXcscsort：**

- 上述校验全部适用，另有：`descrA` 为空 → `ACL_SPARSE_STATUS_INVALID_VALUE`；indexBase 仅支持 `ACL_SPARSE_INDEX_BASE_ZERO / ONE`；
- `pBuffer` 须 128 字节对齐（非对齐 → `ACL_SPARSE_STATUS_INVALID_VALUE`），且不得与 `cscColPtr/cscRowInd/P` 存储区重叠；接口不要求欠配检测（ABI 不传实际 size）；
- 调用方保证 CSC 结构合法：`cscColPtr` 单调不减、`cscColPtr[n] - indexBase == nnz`、行索引 ∈ `[indexBase, indexBase + m)`；
- `P` 须预先初始化为 identity permutation（`0,1,...,nnz-1`），算子仅同步重排不负责创建；
- `nnz==0` 直接成功返回；接口异步下发到 handle 绑定 stream，读取结果前须同步。

# 可维可测分析

## 精度标准/性能标准

**精度标准**：CPU Golden（逐列 `stable_sort` + `P` 同步重排）与 NPU 输出对 `cscRowInd`、`P` 双输出 **exact match**（逐位一致），重复索引处次序必须与输入一致（稳定性断言）。

| 验证线 | 用例 | 结果（Ascend910 上板，2026-09-03） |
| --- | --- | --- |
| C++ UT（`test/cscsort/arch22/`） | 44 例：L0 基础 / L1 功能（重复索引稳定性、base 0/1、空列、多核偏斜、长列多 run）/ L2 边界（runSize 三值边界、128B 对齐拒绝、百万 nnz）/ 异常路径（空指针、非法参数）/ 白盒归并树 / PERF 锚点 | **44/44 通过**，全部与 CPU golden exact |
| 官方测试包精度（ATK 链路等价复现） | 200 条泛化用例 + 13 条确定性复测，rowInd/P 双输出逐位比对 | **213/213 exact**，真实 kernel，无 fallback |
| 官方测试包性能 | 206 用例（3 锚点 × base0/1 + 200 泛化），warmup=10 / samples=30，ACL Event 计时取 median/P90 | 全部通过，`skipped_cases` 为空 |

**性能标准（对 GPU cuSPARSE 标杆 ≥ 0.25×，实测倍率 = 标杆/实测 median）：**

| 锚点 | 形状 | NPU median（base0/base1） | GPU 标杆 | 达标倍率 |
| --- | --- | --- | --- | --- |
| P-01 Llama3.1-70B | 8192×28672, nnz=524288 | 1629.9 / 1627.7 µs | ≈2965 µs | **≈1.8×** |
| P-02 Qwen3-235B | 4096×1536, nnz=262144 | 728.0 / 779.6 µs | ≈2603 µs | **≈3.3~3.6×** |
| P-03 DeepSeek-V3 | 7168×2048, nnz=458752 | 1058.0 / 1070.5 µs | ≈2782 µs | **≈2.6×** |

**内存**：workspace = `align128(4·nnz·4B)`（P-01 ≈8.4MB ≪ L2 192MB）；UB 峰值 ≤ 192KB/AIV（tiling 二分保证）。

**复现步骤**（环境：aarch64 Linux + CANN ≥8.5，Atlas A2/A3）：

```bash
export EAGER_LIBRARY_PATH=$ASCEND_HOME_PATH/lib64
export LINUX_INCLUDE_PATH=$ASCEND_HOME_PATH/aarch64-linux/include
cmake -B build -DSOC_VERSION=ascend910_9362 \
    -DASCEND_CANN_PACKAGE_PATH="$ASCEND_HOME_PATH" \
    -DASCEND_CANN_PACKAGE_LINUX_PATH="$ASCEND_HOME_PATH/aarch64-linux" \
    -DCMAKE_BUILD_TYPE=Release -DASC_DIR="$ASCEND_HOME_PATH/aarch64-linux/lib64/cmake" \
    -DBUILD_TEST=ON -DOP_LIST=cscsort
cmake --build build --target cscsort_test -j8
ASCEND_DEVICE_ID=0 ./build/test/cscsort/cscsort_test            # 44 用例
ASCEND_DEVICE_ID=0 ./build/test/cscsort/cscsort_test --gtest_filter=CscsortTest.PERF_*   # 性能

# 官方测试包（test_cases/，从该目录执行；NPU 精度/性能/内存三线）
cd test_cases
bash run_accuracy_atk.sh                                 # ATK 精度（需 ATK 环境）
cd aclsparseXcscsort_testCase
python3 benchmark_sparse_ops_npu.py --case-file performance_cases.json --device 0 --output results
python3 collect_sparse_ops_npu_memory.py --case-file performance_cases.json --device 0 --output results
```

## 兼容性分析

1. **接口兼容**：签名/参数顺序/`P` identity 语义/`descrA` indexBase 用法与 cuSPARSE 一致；差异两点——`pBuffer` 要求 128 字节对齐（cuSPARSE 无此要求，`aclrtMalloc` 默认满足），`bufferSizeExt` 返回值按硬件区分（见上）；
2. **arch 并存**：arch22 与既有 arch35 同目录并存、按 SoC 编译期择一，公开头文件 `include/cann_ops_sparse.h` 无接口变更，不重复新增同名接口；A5（950）后合入时基于已合入版本处理公共 Host 冲突并完成 A2/A3 与 A5 双硬件回归；
3. **可观测性**：Host 侧 OP_LOGI 输出 tiling 关键量（m/n/nnz/runSize/coreNum/blockDim），便于问题定位；kernel 行为确定（唯一合成键 ⇒ 结果与硬件 tie 行为无关）；参数/描述符类异常在 Host 校验层拦截，CSC 结构合法性按约束由调用方保证。
