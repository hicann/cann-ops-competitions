# aclsparseXcscsort 算子设计文档

| 项目 | 内容 |
| --- | --- |
| CANN 版本 | 9.1.0 |
| 适配硬件 | Atlas A2 训练系列产品（Ascend 910B 系列，DAV-2201/arch22）、Atlas A3 训练系列产品（Ascend 910_93，DAV-2201/arch22） |
| 开发语言 | Ascend C（SIMD 向量编程模型） |
| 目标仓库 | ops-sparse（个人 fork `Xcscsort` 分支，合入目标 master） |
| 参考接口 | cuSPARSE Legacy API `cusparseXcscsort` / `cusparseXcscsort_bufferSizeExt` |
| 交付范围 | aclsparse 公开接口复用 + Host 实现 + Ascend C Kernel（arch22 全新排序网络）+ C++ UT/ST + 任务包等价测试 |

---

# 需求背景（required）

## 需求来源

本任务为昇腾 9 月社区任务「aclsparseXcscsort 算子开发（A2/A3）」：面向 Atlas A2 训练系列产品和 Atlas A3 系列产品（DAV_2201，arch22）完善 `aclsparseXcscsort` CSC 纯索引排序接口并合入 ops-sparse 仓库。功能与 C++ 接口语义参考 cuSPARSE `cusparseXcscsort`，统一使用 `aclsparseXcscsort*` 命名；bufferSizeExt 查询、执行阶段、CSC 描述符、index base、workspace、stream 和错误处理保持接口语义一致。本算子仅处理 I32 索引，不存在 values 或 compute dtype。

核心计算必须在 NPU 上完成（AIV 向量核），不得 CPU fallback；性能要求 P-01/P-02/P-03 三组开源大模型维度锚点 case 达到 0.25 倍性能标杆以上。

## 背景介绍

### CSC 稀疏存储格式

CSC（Compressed Sparse Column，压缩稀疏列）格式使用三个数组描述稀疏矩阵：

- `cscColPtr`：长度 n+1，第 j 列非零元素在数据数组中的起止位置（`[cscColPtr[j]-base, cscColPtr[j+1]-base)`）
- `cscRowInd`：长度 nnz，每个非零元素的行索引
- （可选）`cscValues`：长度 nnz，每个非零元素的值（本算子不涉及）

### 排序语义

对每一列 j，在区间 `[begin, end)` 内对 `cscRowInd` 执行**稳定升序排序**，并使用相同排列同步重排置换向量 `P`：

```
begin = cscColPtr[j]     - indexBase
end   = cscColPtr[j + 1] - indexBase

rowInd_out[begin:end] = stable_sort(rowInd_in[begin:end])        # 升序
P_out[begin:end]      = P_in[stable_perm(begin:end)]             # 同一排列
```

重复行索引保持原始相对顺序（稳定性）。当调用方将 `P` 初始化为 `0..nnz-1` 时，输出满足 `sortedVal[i] = origVal[P[i]]`，可据此同步重排稀疏矩阵的值数组。`cscColPtr` 只读不修改；`cscRowInd` 与 `P` 原地更新。

cscsort 是 csrsort 的列向对偶（csrsort 按 CSR 行排序 `csrColInd`，cscsort 按 CSC 列排序 `cscRowInd`）。

### ops-sparse 仓库现状

仓库 master（及本 fork 的 `Xcscsort` 分支基线）中已有 **arch35（Ascend 950PR/DT，DAV-3510）** 的完整实现：

- `sparse/cscsort/arch35/cscsort_host.cpp`：参数校验、tiling 计算、kernel launch；`bufferSizeExt` 返回 `2 * nnz * sizeof(int32_t)`
- `sparse/cscsort/arch35/cscsort_kernel.cpp`：短列（≤runSize）走 AscendC 高层 `Sort<int32_t, false, RADIX_SORT>` 单键排序 + `Gather` 重排 P；长列走 SIMT（`asc_vf_call` merge-path 多线程归并）
- `test/cscsort/arch35/`：GTest 全场景 UT/ST

**A2/A3（arch22）当前不支持**（`sparse/cscsort/README.md` 支持矩阵明确标注），本任务补齐 arch22 实现。

### arch22 与 arch35 的能力差异（关键技术约束）

对 CANN 9.1.0（aarch64）头文件逐项核实，DAV-2201（dav_c220）与 DAV-3510 的向量算子能力差异如下：

| 能力 | arch35 (950/DAV-3510) | arch22 (910B/DAV-2201) | 对本算子的影响 |
| --- | --- | --- | --- |
| 高层 `Sort` API（RADIX_SORT，int32 单键+索引） | 支持 | **不支持**（`asc/include/adv_api/sort/sort.h` 各重载 `#if __NPU_ARCH__ == 3510 \|\| 3003 \|\| 3113` 门控，2201 编译为空实现） | arch35 短列方案不可复用 |
| SIMT 编程模型（`asc_vf_call`/`asc_simt.h`/`threadIdx`） | 支持 | **不支持**（simt_api 仅 dav_3510 实现） | arch35 长列方案不可复用 |
| Proposal 指令（`MrgSort4`/`MrgSort`/`Sort32`/`Sort`） | 支持 | 仅 half/float dtype | int32 行索引不可用 |
| `Select`（bint8 掩码三目） | 支持 | **仅 half/float**（`kernel_operator_vec_cmpsel_impl.h` dtype 门控） | 交换网络需纯算术实现 |
| `And`/`Or`/`Xor` 位运算（int32 张量） | 支持 | **不可用**（`And`/`Or` 仅 int16/uint16，无张量 `Xor`） | 索引/掩码生成需移位算术替代 |
| `Gather<int32_t>`（Level 2 count 版） | 支持 | 支持（64 元素/repeat，尾部自动拆分；偏移张量须 uint32，可用 `ReinterpretCast` 零开销转换） | 排序网络核心原语可用 |
| `Max/Min/Add/Sub/Mul<int32_t>`（张量×张量） | 支持 | 支持 | 交换网络核心原语可用 |
| `Adds/Muls/Maxs/Mins`（标量版）、`ShiftLeft/ShiftRight`（标量版，int32） | 支持 | 支持 | 符号位抽取、常量运算可用 |
| `Duplicate<int32_t>`、`CreateVecIndex(dst, firstValue, count)`（Level 2） | 支持 | 支持 | 位置序列 iota 一次生成 |
| `DataCopyPad`、`LocalTensor::ReinterpretCast` | 支持 | 支持 | 非对齐列段拷贝可用 |

**结论**：arch22 需要在「无高层 Sort、无 SIMT、无 Select/位运算」的约束下，用纯向量算术自研稳定排序网络。本设计采用 **(K, X) 双张量字典序 bitonic 排序网络**，全部原语均为 2201 已核实的向量 API。

---

# 需求分析（required）

## 需求描述

在 ops-sparse 仓库 `sparse/cscsort/arch22/` 新增 Atlas A2/A3 的 `aclsparseXcscsort` 实现，复用 `include/cann_ops_sparse.h` 已声明的 Legacy API 与矩阵描述符，实现 CSC 每列内 row index 的**原地稳定升序排序**并同步重排 `P`；同步交付 `test/cscsort/arch22/` C++ UT/ST、与任务包等价结构的测试（含 `torch.ops.ops_sparse_test.xcscsort_npu` 注册扩展）、README 更新，并完成 A2 实测精度（exact match）与性能（P-01/02/03 ≥ 0.25×标杆）自验证。

## 需求拆解

1. **接口语义对齐**：`aclsparseXcscsort_bufferSizeExt` 按.nnz 精确计算并检查溢出；`aclsparseXcscsort` 校验 m/n/nnz、colPtr 端点、index base（0/1）、空指针、pBuffer 128 字节对齐；nnz=0 早退且允许数据指针为空；使用调用方 handle 绑定的 stream，接口内不引入同步。
2. **arch22 Kernel 全新排序网络**：短列（列长 ≤ runSize）单 run 向量排序快速路径；长列（列长 > runSize）多 run 切分 + 逐级归并；稳定排序由 (K, X) 双张量字典序全序保证。
3. **多核切分**：按累计 nnz 权重把完整列区间分配到 AIV 核，kernel 内二分定位列边界，不拆分单列、无需核间同步。
4. **C++ UT/ST**：覆盖空矩阵/空列、nnz=0/1、重复索引稳定性、单长列、随机 CSC、index base 0/1、多核切分、run 边界、workspace 精确值/空指针/未对齐、白盒归并与异常路径。
5. **任务包等价测试**：`test_cases/`（aclsparseXcscsort_testCase + common + baseline_results）+ `xcscsort_npu` torch 扩展 + README 复现说明；200 条 accuracy 用例 exact match，206 条 performance 用例按任务包方法采集。
6. **性能门槛**：P-01（8192×28672，nnz=524288）、P-02（4096×1536，nnz=262144）、P-03（7168×2048，nnz=458752）NPU 同调用范围全部 kernel 总耗时满足 性能倍率 = GPU 标杆 median / NPU 耗时 ≥ 0.25（标杆 median 约 2962/2596/2781 μs，即 NPU 需 ≤ 约 11.8/10.4/11.1 ms，1× 对应约 740/649/695 μs）。

---

# 详细设计（required）

## 算子分析

### 数学公式

对每列 `j ∈ [0, n)`，设 `begin = cscColPtr[j] - base`、`end = cscColPtr[j+1] - base`、`len = end - begin`：

```
π = stable_argsort(rowInd[begin:end])                # len 内部稳定升序排列
rowInd_out[begin + i] = rowInd_in[begin + π(i)]      # i ∈ [0, len)
P_out[begin + i]      = P_in[begin + π(i)]
cscColPtr             = 不变
```

纯整数索引运算，无浮点、无 values dtype、无 INF/NAN 语义；精度验收为与 CPU golden 的 **exact match（bit-wise 一致）**，稳定性（重复键保持原相对顺序）为硬性要求。

### 支持数据类型

仅 INT32 索引：`cscColPtr`/`cscRowInd`/`P` 均为 `int*`。不适用 FP16/BF16/FP32/complex64 values dtype（任务书明确本算子为纯索引算子）。

### 支持形状

- `m ≥ 0`、`n ≥ 0`、`nnz ≥ 0`；`m == 0 || n == 0` 时必须 `nnz == 0`
- `cscColPtr` 长度 n+1，单调非降；base=0 时端点为 `0/nnz`，base=1 时端点为 `1/nnz+1`
- 换算后行索引位于 `[0, m)`；重复键允许（稳定性保证）
- 列长可退化：空列（len=0）、单元素列（len=1）到超长单列（len 接近 nnz）均需正确处理

## 接口设计

### 公开接口（复用公共头声明，无需修改）

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

### 调用流程

```
aclsparseCreate(&handle); aclsparseSetStream(handle, stream);
aclsparseCreateMatDescr(&descrA);                     // index base 0 或 1
aclsparseXcscsort_bufferSizeExt(handle, m, n, nnz, colPtr, rowInd, &wsSize);
aclrtMalloc(&pBuffer, wsSize, 128 字节对齐);            // 等大内存接口天然对齐，另做校验
// P 由调用方初始化（通常 0..nnz-1，可用 aclsparseCreateIdentityPermutation 语义）
aclsparseXcscsort(handle, m, n, nnz, descrA, colPtr, rowInd, P, pBuffer);  // 异步下发
aclrtSynchronizeStream(stream);                        // 由调用方同步
```

### 参数与校验规则

| 参数 | 方向 | 校验（失败返回码） |
| --- | --- | --- |
| handle | IN | 非空（`HANDLE_IS_NULLPTR`） |
| m/n/nnz | IN | 非负（`INVALID_VALUE`）；空矩阵时 nnz==0（`INVALID_VALUE`） |
| cscColPtr | IN | n>0 时非空（`INVALID_VALUE`）；只读不修改 |
| cscRowInd | IN/OUT | nnz>0 时非空（`INVALID_VALUE`） |
| descrA | IN | 非空（`INVALID_VALUE`）；indexBase ∈ {0, 1}（`INVALID_VALUE`） |
| P | IN/OUT | nnz>0 时非空（`INVALID_VALUE`） |
| pBuffer | workspace | nnz>0 时非空且 128 字节对齐（`INVALID_VALUE`）；不得与输入输出重叠 |
| pBufferSizeInBytes | OUT | 非空（`INVALID_VALUE`）；nnz==0 时输出 0 |

`aclsparseXcscsort` 不校验 colPtr 内容与 rowInd 取值范围（属接口前置条件，与 cuSPARSE 一致：调用方保证输入合法）；执行 ABI 不传 workspace 实际大小，因此只验收精确查询值、空指针与对齐，不要求可靠检测欠配。

### workspace 布局与公式

arch22 的排序网络以 **(K, X) 双张量**贯穿长列归并（见 kernel 设计），workspace 为四个 int32 数组的 ping-pong 缓冲：

```
pBuffer（128B 对齐）:
┌──────────────┬──────────────┬──────────────┬──────────────┐
│  KA[0:nnz)   │  KB[0:nnz)   │  XA[0:nnz)   │  XB[0:nnz)   │
└──────────────┴──────────────┴──────────────┴──────────────┘
  键 ping-pong      位置 ping-pong
```

- **`bufferSizeExt = 4 * nnz * sizeof(int32_t)`**（含 `size_t` 乘法溢出检查，溢出返回 `INSUFFICIENT_RESOURCES`；nnz==0 返回 0）
- 与 arch35 的 `2 * nnz * sizeof(int32_t)` 不同：arch22 长列归并需同时携带键与原始位置两张量并各自 ping-pong；公共头未承诺具体公式（仅要求 "from bufferSizeExt"），跨架构调用方按各自查询值分配即可。差异在 README 与本文档明示
- 规模评估：nnz=524288（P-01）时 8 MB，远小于 A2 L2 Cache，满足任务书「方案固有 workspace 不超过 L2」的内存验收路径；且性能 case 列长 ≤ 224，全部走快速路径，实际不触碰该 workspace

## 算子实现

### 总体方案

arch22 无高层 Sort/SIMT，本设计以 **(K, X) 双张量字典序 bitonic 排序网络** 为核心：

- **K** = `cscRowInd` 键（int32，值域 `[0, m)` ⊂ `[0, 2^31-2]`）
- **X** = 元素在列内的原始位置（int32，值域 `[0, nnz)`，列内唯一）

(K, X) 构成**严格全序**（X 无重复），任意正确的比较排序网络作用在该全序上，结果即按键 K 的**稳定排序**——稳定性由 X 的唯一性从数学上保证，无需额外打包位约束（单 int32 打包 `(key<<s)|pos` 需要 `m·2^s < 2^31`，存在 m 上限；int64 又无 Min/Max/Gather 向量支持，均被否决）。

**比较交换（compare-exchange）的纯算术实现**是本设计的关键：2201 无 `Select`（仅 half/float）、无 int32 `And/Or/Xor`、无 `Subs`，因此用移位符号位抽取替代布尔算子。利用值域非负且 `< 2^31-1` 的性质，`d = a - b ∈ [-(2^31-2), 2^31-2]` 无溢出：

```
d        = Sub(a, b)                       # 张量减
gt(a,b)  = Adds(ShiftRight(Adds(d, -1), 31), 1)     # a>b ? 1 : 0
           #  d ≥ 1: (d-1)>>31 = 0 → 1；d ≤ 0: (d-1)>>31 = -1（算术右移）→ 0
eq(a,b)  = 1 + (d>>31) + ((0-d)>>31)       # d==0 ? 1 : 0（三个常量/张量加）

swap     = Add(gtK, Mul(eqK, gtX))         # 字典序 (Ka,Xa)>(Kb,Xb) ? 1 : 0 ∈ {0,1}
Ka'      = Sub(Ka, Mul(swap, dK))          # swap=1 时 Ka'=Kb
Kb'      = Add(Kb, Mul(swap, dK))          # swap=1 时 Kb'=Ka（K、X 两张量同法交换）
```

每次交换约 26~36 条向量指令（含每 stage 重算 partner 索引与方向掩码），全部落在 `Gather/Min/Max/Add/Sub/Mul/ShiftLeft/ShiftRight/Adds/Duplicate` 这组 2201 已核实 API 上。

**网络结构**（迭代式 bitonic，标准 (k, j) 双重循环）：

```
for (k = 2; k ≤ paddedLen; k *= 2)          # 构建的有序块大小
  for (j = k/2; j ≥ 1; j /= 2)              # 块内比较跨度
    对全部 i：l = i ^ j
      partnerIdx[i] = i ^ j                 # iota ⊕ j（移位算术合成，见下）
      B = Gather(A, partnerIdx)             # B[i] = A[i^j]
      方向掩码 M[i] = bit_j(i) XOR bit_k(i) # 上/下半块方向
      按 (K,X) 字典序与方向执行比较交换      # 上式纯算术
```

- **iota**：`CreateVecIndex(dst, firstValue, count)` 一次生成 `[begin, begin+len)` 位置序列
- **partnerIdx = i ⊕ j**（j 为 2 的幂，设第 b 位）：`bit = (i>>b) - ((i>>(b+1))<<1)`（ShiftRight/ShiftLeft/Sub），`partnerIdx = i + j - 2·j·bit`（Sub/Mul/Adds）；`Gather` 偏移张量用 `ReinterpretCast<uint32_t>()` 零开销转换
- **方向掩码 M01**（0/1 int32）：`M01 = bit_j(i) XOR bit_k(i)`，同法由 iota 移位合成；交换结果 `out = L + M01·(H−L)`，其中 `L/H` 为两张量的 Min/Max——该式仅在非比较方向翻转，与字典序交换等价且无 `Select`
- **哨兵填充**：列长补齐到 2 的幂（`paddedLen = next_pow2(len)`），填充位 K = X = `INT32_MAX`（严格大于一切真实 (K,X)，排序后沉底，输出时截断前 len 个）

### Kernel 侧设计：三条执行路径

#### 路径 1：快速路径（len ≤ runSize，性能 case 全覆盖）

任务包性能用例列长完全均匀（P-01 最大 19、P-02 最大 171、P-03 全 224），全部命中本路径：

```
1. DataCopyPad 拷入列段 rowInd[begin:end) → UB 的 K 缓冲（非 32B 对齐由 DataCopyPad 处理）
2. CreateVecIndex 生成 X = [begin, begin+len)
3. 哨兵补齐 K/X 至 paddedLen
4. 完整 bitonic 网络（k=2..paddedLen）原地排序
5. 输出：rowInd ← K[0:len)（DataCopyPad 写回）
6. DataCopyPad 拷入 P[begin:end) → UB；Gather(P, X[0:len)) → P'；DataCopyPad 写回 P
```

无 GM workspace 流量，单列在单核 UB 内闭环。

#### 路径 2：长列路径（len > runSize，功能用例触发）

单列数据量超过 UB 承载（runSize 由 tiling 反推，典型 2048~4096）时切 run 归并：

```
阶段 1（run 排序）：按 runSize 切 r = ceil(len/runSize) 个 run；逐 run 在 UB 内
   执行快速路径的 1~4 步，排序结果 (K, X) 写入 GM workspace 的 KA（或 KB）与
   XA（或 XB）对应列区间。
阶段 2（逐级归并）：共 ceil(log2(r)) 趟；每趟把相邻两个有序 run 归并为一个：
   - 归并采用 bitonic-merge 网络：seq = A_run ++ reverse(B_run)（B 的反转用
     Gather + 反向索引 (bLen-1)-iota 合成），补哨兵至 2R；对 bitonic 序列只需
     单一 k=2R 的 merge 网络（log2(2R) 个 stage，比全排序便宜一半以上）
   - (K, X) 两张量使用同一 swap 决策同步交换
   - 读写经 KA/KB 与 XA/XB ping-pong（workspace 4 数组即为此设计）
阶段 3（finalize）：最终有序 (K, X) 中，K → 写回 rowInd 列段；P 列段拷入 UB 后
   Gather(P_orig, X) 写回 P。
```

归并趟数 `log2(nrRun)`，全列 GM 流量 `O(len · log(len/runSize) / runSize)` 次 UB 往返；单列 10 万元素、runSize=2048 时约 6 趟，估算单核毫秒级，满足功能用例（性能 case 不触发本路径）。

#### 路径 3：平凡路径（len ≤ 1）

len=0/1 直接跳过（无工作）。

### 多核切分（与 arch35 同构）

- Host 以**累计 nnz 为权重**把 `[0, n)` 列空间划分为 `coreNum` 个连续区间：核 `id` 处理元素区间 `[nnz·id/coreNum, nnz·(id+1)/coreNum)` 覆盖的完整列
- Kernel 内对 `cscColPtr` 二分查找定位列边界（`colPtrGM_.GetValue` 标量读，`O(log n)` 次/核，与 arch35 `FindColBoundary` 相同）
- **不拆分单列**：长列整体归属一个核（列内串行 run 归并，天然无核间同步）；空列矩阵不启动空转核（`coreNum = min(AIV 数, min(n, nnz))`）
- 各核只访问自己列区间对应的 rowInd/P/workspace 段，无原子、无 barrier、无交叉写

### Host 侧设计

`cscsort_host.cpp`（extern "C" 两 API）：

1. **参数校验**：与 arch35 `ValidateCscsortCommonParams` 同语义（见接口设计表格），dlog `OP_LOGE` 记录
2. **tiling 计算**（`ComputeCscsortTiling`）：
   - `aivCoreNum = GetAivCoreCount()`（R2 动态获取，禁止硬编码）、`ubSize = GetUbSize()`（platform_ascendc 运行时查询）
   - `runSize`：按 UB 预算反推——排序相约 8 个 runSize 级 int32 缓冲（K/X/A/B/d/swap/M01/iota 复用后 ≤8），归并相约 8 个 mergeTile 级；`runSize = clamp(2^floor(log2((ubSize - 8KB) / (8·4))), 1024, 4096)`，`mergeTile = 2048`
   - `coreNum = min(aivCoreNum, min(n, nnz))`
   - TilingData（R4 无数组）：`{ uint32_t n, nnz, indexBase, runSize, mergeTile, coreNum }`
3. **launch**：`cscsort_kernel_do(colPtr, rowInd, P, pBuffer, tiling, coreNum, stream)`，tiling 以 const 引用传入、kernel 侧 by value 接收（禁止 H2D 拷贝 tiling）；接口异步返回，不调用 `aclrtSynchronizeStream`
4. **日志**：OP_LOGE（校验失败）、OP_LOGI（launch 参数）、OP_LOGD（tiling 明细）

### 实现演进：真机调试后的最终方案（A2 实测驱动）

上述总体方案在 DAV-2201 真机上完成五轮系统性单元验证后，有六处关键演进（均已以独立单元 kernel 在真机复现并验证绕过方案）：

**① iota 表改 Host 预填 + MTE2 装载**：`CreateVecIndex` 在 2201 是标量 SetValue 实现（慢且 S 管线写对向量管线存在 DCache 可见性问题），改为 Host 每次 launch 前同步 `aclrtMemcpy` 预填 iota 到 workspace 尾区，kernel 内一次 MTE2 装载。

**② 长列路径升级为 (K, X, Pv) 三张量**：原设计 finalize 阶段按 X 值域窗口 Gather 重排 P，但 (K,X) 排序后 X 非单调，二分定位无从谈起（实测切点错乱）。改为 Pv（原始 P 值）在 run 起点与 K/X 同源连续载入，随网络与归并同步交换，finalize 退化为顺序回写。workspace 由 4 数组扩为 `KA|KB|XA|XB|PA|PB` 六数组，`bufferSizeExt = 6*(nnz+8)*4 + iotaBytes + 2048`（2048B 为跨核写序旗标区）。

**③ 归并切点读改 MTE2 探针**：S 管线直读 MTE3 写过的 GM 不可靠（实测三种读法——裸读/DataSyncBarrier/dcci——均读到滞后或中间态数据，二分切点偏 1 至数十、甚至 unsigned 下溢触发 507035 崩溃）。`CoRankA` 的每次比较改为"MTE2 装载 8 元素对齐块 → SyncMte2S → UB GetValue"，槽位全缓冲轮转避开 S 读缓存对同一线重写的短窗口陈旧。

**④ 管线同步补全**：实测 `SyncMte3Mte2` 只约束 MTE2 管线——MTE3 读某缓冲后循环回到 V 写（Duplicate）必须补 `SyncMte3V`，否则 V 写可插入 MTE3 读完成之前（观察到读回下一轮哨兵）。另外 Level-0 `Gather` 标量 mask 的语义是"有效元素个数"而非位掩码（`~0ULL` 会被截为 -1 仅激活 63 lane），快路径大 padded 网络的数据错误由此而来。

**⑤ `isPad=true` 的 DataCopyPad 弃用**：2201 实测会破坏源地址读取，归并瓦片的 +INF 虚填充改为 Duplicate 预铺 + staging 精确拷贝（`Copy` mask-count 模式）。

**⑥ 多核写序旗标链**：各核列区间的保尾写回（RMW 32B 块读改写）E-尾越界块可跨列延伸至多 7 个元素、跨核落入相邻核区间。凡列终点落在 `[boundary-7, boundary)` 的列（本核尾部窗口）先处理完并在 workspace 旗标区置位魔数，下一核在首次 GM 写前经 MTE2 探针自旋等待前核旗标，其余列全并行；空核/全跳过列的核置位前补等前核保证链条不断裂。该协议使各核写阶段全序、读不加序，实测 40 核高压（每核仅约 8 元素）稳定通过。

**其他定稿参数**：`kRunSizeMax = 1024`（`runSize = mergeTile`，按 23 个网络缓冲反推，UB 占用 92KB）；kernel 经 ascendc.cmake AIV 专属管线编译为独立静态库（混合编译管线对本 kernel 的向量算子序列存在数据相关异常，同源代码在 AIV 管线下行为正确）。

### 设计正确性与溢出分析

- 键 K ∈ [0, m-1] ⊂ [0, 2^31-2]，X ∈ [0, nnz-1] ⊂ [0, 2^31-2]：`d = a-b`、`Adds(d,-1)`、交换式 `a - swap·d`、`b + swap·d` 的所有中间量均落在 int32 值域（swap∈{0,1} 时交换结果即另一操作数，非负）
- 哨兵 `INT32_MAX` 严格大于任何真实 (K, X)：真实值 ≤ 2^31-2；哨兵彼此相等亦无害（沉底后被截断）
- bitonic 网络对严格全序的正确性是经典结论；(K, X) 无重复 → 排序结果唯一 → 与 CPU 稳定排序 golden **bit-wise 一致**（相同输入重复执行结果确定，满足任务书重复执行一致性要求）

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas A2 训练系列产品（910B 系列，含 910B1/B2/B3/B4） | √ |
| Atlas A3 训练系列产品（910_93） | √ |

（arch22/dav-2201 目录实现，与 arch35/DAV-3510 目录互斥编译；Ascend 950PR/DT 由既有 arch35 实现覆盖，两者解耦互不影响。）

## 算子约束限制

- `m/n/nnz ≥ 0`；`m==0 || n==0` 时 `nnz == 0`；`cscColPtr` 单调非降且端点与 base/nnz 一致（接口前置条件，kernel 不检测）
- 换算后行索引 ∈ `[0, m)`；重复键允许（稳定排序保证相对顺序）
- `nnz == 0` 时相关数据指针允许为空，接口直接返回成功
- `cscColPtr` 只读；`cscRowInd`/`P` 原地更新；`pBuffer` 不得与输入输出重叠，须按 `bufferSizeExt` 查询值（`4·nnz·4B`）足量分配且 128 字节对齐；执行 ABI 不传实际大小，不做欠配可靠检测
- 最大规模边界：nnz 受 int32 索引与 `size_t` workspace 乘法溢出检查约束（`4·nnz` 溢出 `size_t` 时返回 `INSUFFICIENT_RESOURCES`）；m/n 无额外上限（双张量设计无打包位约束）
- 仅 INT32 索引；不支持 64 位索引、values dtype、复数与算法枚举参数

---

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | CPU golden 采用逐列稳定 I32 升序排序并同步重排 P；NPU 输出的 `cscRowInd`、`P` 与 golden **exact match（bit-wise）**；`cscColPtr` 不变、`n`/`nnz`/列边界逐项校验；相同输入重复执行结果一致；不得以仅检查有序性替代完整置换与稳定性校验 | 《生态算子开源精度标准》实验标准 + 任务书 |
| 性能标准 | 性能倍率 = 标杆接口 GPU 设备 Event median / NPU 同调用范围全部 kernel 总耗时；P-01/P-02/P-03 各 case ≥ 0.25（预热 10 次、采样 30 次，报告中位数与 P90；矩阵描述符与 workspace 复用，每轮恢复输入；不计首次编译、数据生成与 H2D 搬运） | 任务书 3.3 节 |
| 内存标准 | 方案固有 workspace（`4·nnz·4B`，最大 case 8 MB）不超过目标硬件 L2 Cache 容量 | 任务书 3.4 节（纯索引算子走 L2 路径） |

## 兼容性分析

- **与 arch35 解耦**：arch22 实现位于 `sparse/cscsort/arch22/`，与 `arch35/` 目录互斥编译（顶层 CMake 按 `SOC_VERSION` 反选），同名 `aclsparseXcscsort*` 符号不冲突；不修改 `include/cann_ops_sparse.h` 与 arch35 任何文件，满足任务书「A2/A3 与 A5 公共排序逻辑解耦」要求
- **交叉回归**：arch22 合入后以 `--soc=ascend950` 编译确认 arch35 构建不受影响（本地做编译级回归；950 实机回归由评审环境补充）
- **接口行为不变**：两 API 签名、返回码语义、stream 异步语义与 arch35 完全一致；仅 workspace 查询值因实现差异不同（调用方按查询值分配，符合接口契约）

## 测试方案

### C++ UT/ST（`test/cscsort/arch22/cscsort_test.cpp`，GTest + 仓库 test/frame 框架）

golden：主机侧逐列 `stable_sort` 按键生成排列并重排 (rowInd, P)（与 arch35 golden 同语义），NPU 结果逐元素 exact 比对。

| 类别 | 用例 |
| --- | --- |
| 基础功能 | Basic、EmptyMatrix（m/n=0、nnz=0）、AlreadySorted、Reverse、DuplicateRows（重复键稳定）、NonIdentityP（非单位 P）、Base1、EmptyCols、SingleElemCols、ManySmallCols、MultiCoreSkewedCols（偏斜列多核）、LongCol（单长列 > runSize）、LongColMultiRun（多 run 归并）、RandomMedium、LastColEmpty、SingleRow、nnz=0/1 |
| 边界泛化 | RunSizeBoundaries（len=runSize±1、2 的幂边界）、NonAlignedBuffer（128B 对齐拒绝）、LargeNnzMultiCore（>50 万 nnz）、全重复键列（稳定性极限）、多次执行 bit-wise 一致、cscColPtr 只读校验、P 恒为输入重排 |
| 白盒（直调 `cscsort_kernel_do` + 定制 tiling） | bitonic 单 run 排序正确性、多 run 归并奇/偶趟（结果落 KA/KB 两侧）、哨兵填充截断、空列区间空闲核、尾核非对齐、mergeTile 边界、方向掩码/伙伴索引合成 |
| 异常路径 | NullHandle、NullBufferSize、InvalidM/N/Nnz、EmptyMatrixWithNnz、NullColPtr/NullRowInd/NullDescr/NullP/NullBuffer、未对齐 pBuffer、非法 indexBase；bufferSizeExt 精确值（`4·nnz·4B`）与溢出拒绝 |
| 接口流程 | bufferSizeExt 查询 → workspace 分配 → Xcscsort 原地排序 → P 同步重排全链路；stream 异步语义（多 stream 乱序下发 + 显式同步后校验）；描述符连续创建/执行/销毁无泄漏 |
| 性能冒烟 | aclrtEvent 计时（Small/Medium/Large/MillionNnz/Skewed/SingleLongCol/MultiCoreSmallCols），仅输出耗时不设门槛 |

### 任务包等价测试（仓库 `test_cases/`）

- 结构与任务包 1:1：`aclsparseXcscsort_testCase/`（accuracy_cases.json 200 条 + performance_cases.json 206 条 + ATK function/accuracy 脚本）、`common/`（benchmark/memory 框架）、`baseline_results/`（GPU 标杆数据）、`README.md`（环境、编译、运行、复现步骤）
- **`xcscsort_npu` torch 扩展**：C++ 源码 + 构建脚本，`TORCH_LIBRARY` 注册 `torch.ops.ops_sparse_test.xcscsort_npu(col_ptr, row_ind, base) -> (sorted_rows, permutation)`；内部调用 aclsparse 公开接口（CreateMatDescr → bufferSizeExt → workspace/P 分配 → Xcscsort），供 ATK 精度执行器与 NPU benchmark/内存脚本挂接
- 精度执行：`atk task -c accuracy_cases.json -n nodes_accuracy.yaml --task accuracy`（CPU golden 节点 + NPU 节点 exact 比对）；性能执行：`benchmark_sparse_ops_npu.py --case-file performance_cases.json`（Event 计时，warmup 10 / samples 30）

## 性能预估

以双张量网络 ~30 条向量指令/交换、910B 单核 1.6 GHz 估算（列 padded 至 2 的幂）：

| case | 维度 | 列长/列数 | 单核列数 | 估算单核耗时 | 门槛（0.25×） |
| --- | --- | --- | --- | --- | --- |
| P-01 | 8192×28672, nnz=524288 | 19（padded 32）/28672 | ~1434 | ≈0.35 ms | ≤11.8 ms |
| P-02 | 4096×1536, nnz=262144 | 171（padded 256）/1536 | ~77 | ≈0.7 ms | ≤10.4 ms |
| P-03 | 7168×2048, nnz=458752 | 224（padded 256）/2048 | ~103 | ≈0.25 ms | ≤11.1 ms |

预估裕量 >14×，同时接近 1× 水平（约 0.65~0.74 ms）；若实测不达预期，备选优化：`m·2^s < 2^31` 时切换 packed 单张量快路径（`key<<s | pos`，交换指令数约减半）、短列（padded ≤ 64）特化网络、按列长分桶调度。

---

# 交付物清单

| 交付件 | 位置（ops-sparse `Xcscsort` 分支） |
| --- | --- |
| Host/Kernel/Tiling | `sparse/cscsort/arch22/{cscsort_host.cpp, cscsort_kernel.cpp, cscsort_kernel.h, cscsort_tiling_data.h}` |
| C++ UT/ST | `test/cscsort/arch22/{cscsort_test.cpp, cscsort_golden.h}` |
| 任务包等价测试 | `test_cases/{aclsparseXcscsort_testCase/, common/, baseline_results/, README.md}`（含 xcscsort_npu torch 扩展源码与构建脚本） |
| 算子文档 | `sparse/cscsort/README.md`（支持矩阵补 A2/A3、workspace 公式、runSize/UB、稳定性与边界说明） |
| 本设计文档 | cann-ops-competitions `tasklist/09-1-Xcscsort/qq_64858158/design.md` |

提交策略：A2/A3 实现合入 ops-sparse master（`sparse/cscsort/arch22/` + `test/cscsort/arch22/`），公共声明不变；A5（arch35）既有实现不受影响，后合入 PR 基于已合入版本处理公共 Host 冲突并完成 A2/A3 与 A5 回归。
