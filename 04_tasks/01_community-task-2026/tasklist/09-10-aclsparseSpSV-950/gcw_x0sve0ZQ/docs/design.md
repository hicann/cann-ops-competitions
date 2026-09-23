# aclsparseSpSV(A5/950) 算子设计文档

## 一、需求背景

### 1.1 需求来源

社区任务《9月社区任务-aclsparseSpSV算子开发(950)》。在 Ascend 950PR（DAV_3510，`arch35`，A5）上，
基于 `ops-sparse` 仓已有的 SpSV 实现，补齐 `ACL_COMPLEX64` 数据类型、完善测试与性能，并交付文档。
代码合入 [ops-sparse](https://gitcode.com/cann/ops-sparse)，源码目录 `sparse/spsv/arch35/`，
测试目录 `test/spsv/arch35/`。

### 1.2 背景介绍

#### 1.2.1 算子功能

`aclsparseSpSV` 求解稀疏三角线性方程组，语义与 cuSPARSE `cusparseSpSV` 逐阶段对齐：

```
op(A) · Y = alpha · X
```

`A` 为 `[m, m]` 稀疏三角方阵，`X`/`Y` 为长度 `m` 的稠密向量，`alpha` 为标量。

```
op(A) = A        opA == ACL_SPARSE_OP_NON_TRANSPOSE
op(A) = Aᵀ       opA == ACL_SPARSE_OP_TRANSPOSE
op(A) = Aᴴ       opA == ACL_SPARSE_OP_CONJUGATE_TRANSPOSE
```

FP32 下 `Aᴴ ≡ Aᵀ`；complex64 下 `Aᴴ = conj(Aᵀ)`，两条分支数值不同，必须分别实现。

生命周期为六阶段：`createDescr → bufferSize → analysis → solve → updateMatrix → destroyDescr`，
跨阶段状态保存在不透明描述符 `aclsparseSpSVDescr_t` 中。

#### 1.2.2 基线现状审计

工程基线为 `cann/ops-sparse` master 的 `sparse/spsv/arch35/`，共 3029 行 Host+Kernel、
1488 行测试。逐文件审计结论如下。

**已具备，本次不改动**

| 能力 | 证据 |
|---|---|
| 六个公开接口签名与任务书 §2.3 逐字一致 | `include/cann_ops_sparse.h:1974-1997` |
| 四种格式 CSR/CSC/COO/SLICED_ELL | `spsv_host.cpp:37-40` 格式白名单 |
| N/T/H 三枚举均被接受 | `spsv_host.cpp:110-112` |
| index base 0/1、I32 索引（并兼容 I64） | `spsv_host.cpp` ptrType/idxType 校验 |
| Host/Device pointer mode | `PrepareSolveKernelArgs` 内解析 alpha |
| BufferSize/Analysis 允许 vecX/vecY 为 NULL | `ValidateSpSVDenseVecValues` 只在 Solve 路径调用 |
| 无 CPU fallback | Host TU 内无 `aclrtMemcpy`/`aclrtSynchronize`，全部为 stream 上 kernel 下发 |
| level scheduling 分析/求解框架 | `SpsvAnalysisCommon` + `SpsvSolveSingleCore`/`SpsvSolveMultiCore` |

**核心缺口，本次工作主体**

| 缺口 | 现状证据 |
|---|---|
| complex64 完全缺失 | `spsv_host.cpp:78` 与 `:87` 对 `computeType != ACL_FLOAT`、`valueType != ACL_FLOAT` 直接返回 `NOT_SUPPORTED`；`grep -c complex sparse/spsv/arch35/spsv_kernel.cpp` = **0** |
| H 未做共轭 | `NeedsTranspose`（`spsv_host.cpp:121`）把 H 与 T 同等处理，转置搬运不取共轭 |
| alpha 为单 float | `SpsvTilingData::alpha` 是 `float`，device 模式只读 4 字节 |
| workspace 按 4 字节定尺 | 值段按 `sizeof(float)` 计算，complex64 需翻倍 |
| 描述符未缓存 value type | Analysis→Solve 的一致性校验覆盖 format/op/fill/diag/base，独缺 dtype |
| 测试无 dtype 维度 | `spsv_test.csv` 145 条用例无 dtype 列，全部 FP32；H 分支在 FP32 下与 T 等价，等于未被真正覆盖 |
| 交付测试包未接通 | 任务包需要 `torch.ops.ops_sparse_test.spsv_{analysis,update}_npu` 与 `spsv_npu` 三个 hook，仓内无注册点，精度/性能/内存三条链路同时阻塞 |

**执行模型**：SpSV 是纯 SIMT 实现（`KERNEL_TYPE_AIV_ONLY`），不含 TPipe/TQue/LocalTensor/DataCopy，
取值全部是标量 `__gm__` 访存。这一点决定了复数化路线——不存在 UB 切分与向量位宽假设需要迁移，
复数化收敛为「标量类型参数化 + 复数算术」，不必重写 tiling。

---

## 二、需求分析

### 2.1 需求描述

在保持现有接口与四格式能力不变的前提下：补齐 complex64 全链路、实现 `Aᴴ` 共轭语义、
接通交付测试包、使 6 条性能用例达到标杆的 0.3 倍以上、精度满足生态算子开源精度标准。

### 2.2 需求拆解

1. complex64 端到端：Host 校验、workspace 定尺、alpha 双分量、Kernel 复数算术、golden 与测试。
2. `Aᴴ` 共轭转置，含 CSC+H 这一「只共轭不转置」的组合。
3. 求解阶段按**真实依赖层结构**自适应选择执行路径（见 §3.2.4，性能达标的决定项）。
4. 注册三个 NPU hook，打通任务包的精度 200 例 / 性能 206 例 / 内存对比。
5. C++ UT 增加 dtype 维度，覆盖四格式 × fill/diag × N/T/H × pointer mode × in-place × UpdateMatrix。
6. A2/A3 公共 Host 层联合回归。

### 2.3 外部组件依赖

不引入新的第三方依赖。复用 handle 机制、`sparse/common/` 描述符层、`simt_api/asc_simt.h`、
`test/frame` 测试公共件。测试侧新增环境依赖：`eigen3-devel`（golden 用）。

### 2.4 内部适配模块

| 模块 | 文件 | 职责 |
|---|---|---|
| 复数标量层 | `sparse/spsv/arch35/spsv_complex.h`（新增） | `SpsvComplex64` POD、内联复数算术、SIMT 限定符封装 |
| Tiling | `spsv_tiling_data.h` | 增加 `alphaImag`、`valueType`；`opA` 扩为位域（bit0 转置 / bit1 共轭） |
| Host | `spsv_host.cpp` | dtype 校验与一致性、按值宽定尺 workspace、复数 alpha、共轭标志 |
| Kernel | `spsv_kernel.cpp` | 求解链/分析链按 `ValueT` 模板化；转置时施加共轭；入口按 valueType 二分派；**求解路径自适应** |
| 跨阶段状态 | `sparse/common/aclsparse_spsv_descr.h` | 增加 `cachedValueType`/`cachedValueSize` |
| 复数 golden | `test/spsv/spsv_complex_golden.h`（新增） | complex128 稠密回代 golden、逐分量混合容差判定 |
| 功能测试 | `test/spsv/arch35/spsv_test.{cpp,csv}` | 增加 dtype 列与 complex64 用例 |
| NPU hook | `test_cases/aclsparseSpSV_testCase/spsv_npu_registration.py`（新增） | 注册三个 `torch.ops.ops_sparse_test.*` |

`sparse/CMakeLists.txt` 按 `SOC_ARCH_DIRS` 自动收集 `sparse/*/arch35/*.cpp`，源码侧无需改构建脚本。

### 2.5 接口原型

与仓内 `include/cann_ops_sparse.h` 现有声明逐字一致，**本次不修改公开头文件**：

```cpp
aclsparseStatus_t aclsparseSpSV_createDescr(aclsparseSpSVDescr_t *spsvDescr);
aclsparseStatus_t aclsparseSpSV_destroyDescr(aclsparseSpSVDescr_t spsvDescr);
aclsparseStatus_t aclsparseSpSV_bufferSize(aclsparseHandle_t handle, aclsparseOperation_t opA,
    const void *alpha, aclsparseConstSpMatDescr_t matA, aclsparseConstDnVecDescr_t vecX,
    aclsparseDnVecDescr_t vecY, aclDataType computeType, aclsparseSpSVAlg_t alg,
    aclsparseSpSVDescr_t spsvDescr, size_t *bufferSize);
aclsparseStatus_t aclsparseSpSV_analysis(/* 同上 */, void *externalBuffer);
aclsparseStatus_t aclsparseSpSV_solve(/* 同上，无 externalBuffer */);
aclsparseStatus_t aclsparseSpSV_updateMatrix(aclsparseHandle_t handle,
    aclsparseSpSVDescr_t spsvDescr, void *newValues, aclsparseSpSVUpdate_t updatePart);
```

---

## 三、详细设计

### 3.1 算子分析

**数学定义**。下三角、NON_UNIT、`op = N` 时为前代：

```
y_i = ( alpha·x_i − Σ_{j<i, A_ij≠0} A_ij · y_j ) / A_ii
```

上三角为回代，方向相反。UNIT 对角时分母恒为 1 且对角元不参与累加。
`op = T/H` 时求解的是 `Aᵀ`/`Aᴴ`，对 CSR 存储等价于按列遍历，实现上统一归一化为
「转置后的 CSR 视图」再走同一套回代。

**支持数据类型**：`ACL_FLOAT`（FP32）、`ACL_COMPLEX64`。索引 I32（兼容 I64），base 0/1。

**支持形状**：`A` 为 `[m, m]`，`X`/`Y` 为 `[m]`，动态 `m` 与 `nnz`。

### 3.2 算子实现

#### 3.2.1 Host 侧设计

**dtype 校验与绑定**。`computeType` 放开到 `{ACL_FLOAT, ACL_COMPLEX64}`，并强制
`matA->valueType == computeType`。value type 写入描述符（`cachedValueType`/`cachedValueSize`），
与既有的 format/op/fill/diag/base 同级参与 Analysis→Solve 一致性校验；不一致返回
`ACL_SPARSE_STATUS_INVALID_VALUE`。

**workspace 定尺**。现有 `ComputeWorkspaceOffsets` 中值段硬编码 `sizeof(float)`，
改为 `cachedValueSize`（FP32 为 4、complex64 为 8）。段布局与对齐规则不变。
仅按 `bufferSize` 查询值分配，Analysis 与 Solve 复用同一块，异步 Solve 完成前不释放。

**alpha 双分量**。`SpsvTilingData` 增加 `alphaImag`。Host pointer mode 下在主机侧拆实虚部写入 tiling；
Device pointer mode 下 tiling 只携带指针，由 kernel 按 `valueType` 读 4 或 8 字节。

**opA 位域**。`opA` 编码为 bit0=需要转置、bit1=需要共轭：

| opA | dtype | 转置 | 共轭 |
|---|---|---|---|
| N | 任意 | 否 | 否 |
| T | 任意 | 是 | 否 |
| H | FP32 | 是 | 否（等价 T） |
| H | complex64 | 是 | 是 |

CSC 存储本身是 `Aᵀ` 的 CSR 视图，因此 **CSC + H 落到「只共轭、不转置」**，
这是唯一一个共轭与转置标志不同号的组合，需单独用例覆盖。

#### 3.2.2 Kernel 侧设计

求解行原语 `SpsvSolveRow`、转置构建、UpdateMatrix 值搬运三条链路按 `ValueT` 模板化。
入口按 `tiling.valueType` 做一次二分派，`ValueT ∈ {float, SpsvComplex64}`。

`SpsvComplex64` 为 8 字节 POD（交织 re/im），提供 `operator+ - *`、复数除法与 `conj`。
复数除法采用 **Smith（按较大分量缩放）** 形式，避免 `re²+im²` 在大幅值下溢出为 `Inf`、
在小幅值下下溢为 0：

```
|c| ≥ |d|:  r = d/c;  (a+bi)/(c+di) = ((a + b·r) + (b − a·r)i) / (c + d·r)
|c| <  |d|: r = c/d;  (a+bi)/(c+di) = ((a·r + b) + (b·r − a)i) / (c·r + d)
```

零对角 NON_UNIT 时按 IEEE-754 传播，complex64 统一产出 `(NaN, NaN)`。

共轭只在**转置构建**这一处施加（`transValues[pos] = conj(values[p])`），
求解主循环不感知 op，保持单一数值路径，减少组合爆炸。

#### 3.2.3 四格式归一化

CSC/COO/SLICED_ELL 在 Analysis 阶段统一归一化为 CSR 视图 + 置换映射（`perm`），
Solve 只面对 CSR。未排序坐标在归一化时做确定性排序，保证重复执行 bit-wise 一致。
这套机制已在基线中实现且被 145 条 FP32 用例覆盖，复数化只需让值搬运走 `ValueT`。

#### 3.2.4 性能设计：按真实层结构自适应选择求解路径

这是 6 条性能用例能否达标的决定项，单独展开。

**标杆口径**。任务包 `benchmark_cusparse_gpu.cu` 的计时区间为
`updateMatrix` 与 `solve` 两段之和（`all.push_back(refresh + sol)`，输出到 `median_us`），
**`analysis` 不在计时区间内**。这一点可由随包基线数据独立核对：

| 用例 | median_us | solve_mean_us | update_mean_us | analysis_mean_us（不计时） |
|---|---|---|---|---|
| spsv-P-01-base0 | 40458.5 | 40245.9 | 17.8 | 268038.4 |
| spsv-P-02-base0 | 93850.8 | 94163.5 | 38.5 | 565429.2 |
| spsv-P-03-base0 | 222887.6 | 221718.1 | 84.2 | 1437240.9 |

`median ≈ solve + update`，与源码一致。

NPU 侧再看任务包 `common/benchmark_runner.py` 的采样循环：`reset()`（内含 `updateMatrix`）
在 `start.record()` **之前**，Event 窗口内只有 `invoke()`，即只有 `solve`。
两侧因此略有不对称：GPU 的 `median_us` 含 `updateMatrix`，NPU 不含。
该项占比极小（P-01 为 17.8 / 40458.5 ≈ 0.04%）且方向利于 NPU，
本设计据实说明、不加利用，性能报告中会同时给出 NPU 的 `updateMatrix` 单独耗时。

**0.3 倍判据的预算换算**。倍率 = 标杆 ÷ NPU ≥ 0.3，即 **NPU 耗时不超过标杆的 3.333 倍**：

| 用例 | 标杆 median_us | NPU 预算（× 3.333） |
|---|---|---|
| P-01 | 40,458.5 | 134.9 ms |
| P-02 | 93,850.8 | 313.5 ms |
| P-03 | 222,887.6 | 742.9 ms |

**P 场景的真实结构**。三条 P 用例都由任务包 `operator_adapter._csr` 以
`row_pattern="uniform"` 生成。按其 `_uniform_counts` 逻辑，`nnz = (b+1)·m` 会让第 `i` 行
取到 `min(i, b)` 个非对角元，列号为 `i−b … i−1`，即**带状三角阵**：

| 用例 | m | nnz | 带宽 b | 真实依赖层数 | 真实平均层宽 |
|---|---|---|---|---|---|
| P-01 | 65,536 | 524,288 | 7（28 行为 8） | 65,536 | 1 |
| P-02 | 131,072 | 1,572,864 | 11（66 行为 12） | 131,072 | 1 |
| P-03 | 262,144 | 3,932,160 | 14（105 行为 15） | 262,144 | 1 |

（`b = nnz/m − 1`；首 `b` 行受三角容量限制递增，余数按 `_uniform_counts` 的
`remaining` 分支补到少数行，故最大值为 `b+1`。）

第 `i` 行依赖第 `i−1` 行，依赖图退化成长度 `m` 的**纯串行链**，不存在层内并行。
标杆在 H100 上 solve 要 40–222 ms（P-01 折合每个非零元 77 ns），慢的原因正是逐层同步
×`m` 次，而不是算力不足。

**基线实现的问题**。`ComputeNumBlocks(m, nnz, nthreads)`（`spsv_host.cpp:354`）
在主机侧用 `estimatedNumLevels ≈ nnz/m` 估层数，据此算平均层宽并与 256 比较：

| 用例 | 启发式估计层宽 | 真实层宽 | 高估倍数 | 主机据此选择 |
|---|---|---|---|---|
| P-01 | 8,192 | 1 | 8,192× | 多核 |
| P-02 | 10,922 | 1 | 10,922× | 多核 |
| P-03 | 17,476 | 1 | 17,476× | 多核 |

源码注释本身也标注了这个风险：「numLevels is only known after analysis, so this estimate
may miss some deep cases」。落到多核路径后，`SpsvSolveMultiCore` 每层一次 `SyncAll()`
（`spsv_kernel.cpp:1493`），P-01 要做 65,536 次跨核同步，而每层真正的计算只有一行 7 个非零元。

**实测标定**（Ascend 950PR 空闲卡，FP32，warmup 3 / samples 10，只计 `solve`）。
为分离这一项的代价，把 `ComputeNumBlocks` 临时改为恒返回 1 再重测：

| 尺寸 | 现状（多核） | 强制单核 | 标杆 | 单核倍率 |
|---|---|---|---|---|
| P-01 | 95.39 ms | **83.45 ms** | 40.46 ms | 0.485 |
| P-02 | — | 204.53 ms | 93.85 ms | 0.459 |
| P-03 | — | 467.53 ms | 222.89 ms | 0.477 |

结论需要据实修正：逐层 `SyncAll()` 确实存在且可省，但只占 12.5%（95.39 → 83.45 ms），
**并不是量级差距**；FP32 在现状实现下倍率已是 0.42，本就过线。

真正的成本在别处。单核路径下 P-01 每层耗时 83.45 ms / 65,536 = **1.27 µs**，
而每层只解一行、7 次乘加。`SpsvSolveSingleCore` 对每个 level 都要读两次 `levelPtr`、
一次 `levelRow`，并在 2048 线程上做一次 `asc_syncthreads()`——层宽为 1 时，
2047 个线程空转且每层都要付一次全块栅栏。**开销与层数成正比、与实际计算量无关。**

因此本设计的性能目标不是「改掉分核启发式」这一件事，而是**识别串行链并换用与之匹配的执行形态**。

**设计方案**。关键观察是：**真实 `numLevels` 已经在设备侧可得**。Analysis kernel 把它写入
workspace（`spsv_kernel.cpp:800/838`），Solve kernel 在 `PrepareSolveKernelArgs` 里
从 workspace 读回（`:1537`，注释明确「NOT from tiling.numLevels」）。
因此不需要任何主机-设备同步，就能在 kernel 内部拿到真值。

据此把单核/多核的选择从主机下沉到 kernel：

1. 主机仍按现有规则给出 `numBlocks` 上限并下发（不改变 launch 配置）。
2. Solve kernel 读回真实 `numLevels` 后计算 `avgWidth = m / numLevels`，
   以此替代主机估计值做分支：`avgWidth` 低于阈值走单块串行路径，其余块立即返回；
   否则走现有多块逐层路径。
3. 所有块读到的是同一个 workspace 值，分支在块间一致，不产生分叉风险。

这条改动把 `SPSV_DISPATCH_SOLVE_BLOCKS` 的判据从「主机猜的 `numBlocks`」换成
「设备已知的真实层宽」，不新增同步、不改接口、不改 workspace 布局。

**串行链路径的设计**。判定为串行链后，瓶颈是 `m` 步依赖链的**延迟**，
而不是带宽（P-03 也只有 3.9 M 非零元）。按上面的实测拆解，要消掉的是
「每层一次全块栅栏 + 三次 GM 元数据读」这项与计算量无关的固定开销：

- **按层宽选择线程数**。层宽为 1 时用单线程执行，栅栏退化为空操作，
  不再为 2047 个空转线程付 `asc_syncthreads()`。
- **串行链专用走法**。层宽恒为 1 时 `levelRow` 是恒等映射，可直接按行号推进，
  省掉逐层的 `levelPtr`/`levelRow` 三次 GM 读。
- **依赖窗口驻留**。带状结构下依赖窗口固定为 `b ≤ 16`，把最近 `b` 个已解出的 `y`
  留在寄存器中循环复用，避免每步回读刚写出的 `y`。
- **行内展开**。行内 `b` 个乘加彼此独立，展开后并行发射，每步真正串行的只有一次除法。

这些手段都只作用在「真实层宽为 1」的分支上，不改变宽层场景的现有行为。

**为什么仍要做**。FP32 现状已过线（0.42–0.49），但性能表里 P-02、P-03 共四条用例是
**complex64**（见下），而 complex64 是 8 字节元素、且除法需按 Smith 形式做多次乘除。
若其耗时约为 FP32 的两倍，倍率将落到 0.23 附近而**不达标**。
串行链优化正是为 complex64 留出余量。

| 用例 | dtype | fill / diag / op | 标杆 median_us | 预算（×3.333） |
|---|---|---|---|---|
| P-01 base0/1 | float32 | lower / nonunit / N | 40,458.5 | 134.9 ms |
| P-02 base0/1 | **complex64** | upper / unit / T | 93,850.8 | 313.5 ms |
| P-03 base0/1 | **complex64** | lower / nonunit / H | 222,887.6 | 742.9 ms |

**具体倍率以实测为准，在自测报告中记录。** 本节的 NPU 数字均为本机实测，
采集条件（空闲卡、warmup/samples、只计 `solve`）已在表头注明。

**泛化用例**。206 条性能用例中还包含非带状的 `row_pattern`（`banded`/`highly_imbalanced`/
`one_long_row`/`skewed`/`power_law`/`random`/`diagonal`）。`diagonal` 层数为 1、层宽 `m`，
是多核路径的最佳场景；`highly_imbalanced` 等介于两者之间。用真实层宽做判据，
对这些分布同样成立，不需要按 pattern 特判——这正是把判据换成真值而不是换个常数的原因。

### 3.3 支持硬件

| 芯片版本 | 勾选 |
|---|---|
| Ascend 950PR（DAV_3510 / arch35） | √ |

公共 Host 层改动同步在 Atlas A2（ascend910b3）与 A3（ascend910_93）上做联合回归。

### 3.4 算子约束限制

- `A` 为二维三角方阵；fill mode 之外的非目标三角项按三角语义忽略。
- NON_UNIT 下缺失或零对角按 IEEE-754 传播 INF/NAN，complex64 统一为 `(NaN, NaN)`，并在文档与报告中记录。
- A/X/Y/`computeType` 的 dtype 必须一致。
- Device 索引为 I32（实现兼容 I64）；四格式均接受未排序坐标并做确定性归一化。
- `externalBuffer` 在 Analysis 至异步 Solve 完成期间有效且不被外部修改。
- `Y` 可与 `X` 共用同一 Device values 指针做原地求解；`A`、`X` 只读。

---

## 四、可维可测分析

### 4.1 精度标准

以 CPU Golden 为单标杆：FP32 用 float64，complex64 用 complex128。
判定参数 `rtol = 2⁻¹⁰`、`atol = 2⁻¹⁶`、`A = 1e-2`；逐元素
`|actual − golden| ≤ atol + rtol·|golden|`，整体匹配率 ≥ 0.99，
且每元素绝对误差 ≤ `max(A, 32·ULP(golden))`。complex64 的实部、虚部分别适用全部规则。
覆盖普通值、抵消、INF/NAN、UNIT 对角、N/T/H 与 UpdateMatrix 后的解向量，
重复执行校验 bit-wise 确定性。

### 4.2 性能标准

性能倍率 = 标杆 GPU 设备 Event 耗时 / NPU 同调用范围耗时 ≥ 0.3，
即 **NPU 耗时不超过标杆的 3.333 倍**。计时范围按 §3.2.4：标杆侧为
`updateMatrix + solve`，NPU 侧按任务包 runner 为 `solve`，`analysis` 两侧均不计入；
两侧的 `updateMatrix` 差额（约 0.04%）在报告中单列说明。
预热 10 次、正式采样 30 次，报告中位数、P90、Analysis、Solve、workspace 与 profiler 证据。
测量在无其他负载的空闲卡上进行，并在报告中附 `npu-smi` 佐证。

### 4.3 内存标准

按任务书 §3.4 第 2 条验收：方案固有 workspace 绝对值不超过目标硬件 L2 Cache 容量。
SpSV 无等价公开 Torch 稀疏接口（任务包 `operator_adapter.py` 的
`GPU_BASELINE_UNAVAILABLE_REASON` 已说明），故不走 50% 对比路径。
workspace 仅按 `bufferSize` 查询值分配，Analysis 状态只保存必要元数据与 Device workspace，
不引入 CPU 求解副本。

### 4.4 测试设计

| 类别 | 覆盖 |
|---|---|
| 功能 | 四格式 × LOWER/UPPER × UNIT/NON_UNIT × N/T/H × FP32/complex64 |
| 生命周期 | Create、vecX/vecY 为 NULL 的 BufferSize/Analysis、Solve、GENERAL/DIAGONAL UpdateMatrix、Destroy；Analysis→Solve 的描述符/参数/buffer 一致性；异步完成后释放 buffer |
| 边界 | m/nnz 为 0/1、空行、未排序坐标、NON_UNIT 零对角的 INF/NAN 传播、非法索引、workspace 不足 |
| 运行语义 | Host/Device pointer mode、X/Y 原地、stream 异步、重复执行确定性、描述符变更与过早释放 buffer |
| 专项 | CSC+H 的「只共轭不转置」；complex64 大/小幅值复数除法（`1e20+1e20i` 量级）不溢出 |

上游 `densetosparse` 等算子存在与本次改动无关的既有失败用例，回归时以
「失败集合与基线逐字一致」为判据，集合变化才判 FAIL。

### 4.5 兼容性分析

不修改公开头文件，不改变既有 FP32 路径的数值行为与接口语义，
FP32 用例的输出保持 bit-wise 不变。`opA` 位域化、workspace 按值宽定尺、
描述符增加 dtype 字段均为内部结构调整，对外不可见。
公共 Host 层改动完成 A2/A3 联合回归。
