# 需求背景（required）

## 需求来源

- 任务：`9月社区任务-aclblasIcamin算子开发（950）`（社区任务 2026 列表 9 月 #58；同一任务书在 8 月为 #47，两月均发放）
- 任务书：`cann/cann-competitions` → `04_tasks/01_community-task-2026/docs/`（8 月实例见 `docs/202608/`，9 月见 `docs/README.md` 9 月表 #58）
- 目标合入仓：`cann/ops-blas`，实现目录 `blas/iamin/arch35/`，测试目录 `test/iamin/icamin/arch35/`
- 算力环境：hidevlab WebIDE 云端 DevEnv，Ascend 950PR（`npu_arch_3510` / dav_3510，56 AIV 核，HBM 128G），CANN 9.1.0

## 背景介绍

### BLAS Level-1 索引归约族现状

`amin/amax` 族是 BLAS 一级例程中"归约到标量索引"的一类。ops-blas master 现状（本次核对 `git ls-tree -r HEAD` 共 1117 个文件）：

| 接口 | 类型 | 仓内状态 |
| --- | --- | --- |
| `aclblasIsamin` / `aclblasIsamax` | FP32 | 已合入（`blas/iamin/arch35/isamin_*`、`blas/iamax/arch35+arch22`），声明在 `include/cann_ops_blas.h:195` |
| `aclblasIcamin` | COMPLEX64 | **缺失**，本任务新增 |
| `aclblasIcamax` | COMPLEX64 | 缺失（8 月 #46 同批任务，非本任务范围） |

即：本任务不是从零设计一个归约骨架，而是在**已合入的同族实数实现基础上做复数扩展**。任务书 §2.2 亦明确要求："实现代码放在 `blas/iamin/arch35/`（与同族实数接口 aclblasIsamin 同目录，**可复用其 arch35 归约实现框架**）"。本设计即遵循该约束。

### 复数语义带来的三处实质差异

1. **模定义**：BLAS 惯例 `|Re| + |Im|`（绝对值和，非欧氏模），因此无需开方，纯加减与绝对值即可，适合向量流水。
2. **数据宽度翻倍**：`aclblasComplex` 为 `{float real; float imag;}`（`include/cann_ops_blas_common.h:68`），n 个复数元素在 GM 中是 `2n` 个连续 float；一次归约前必须先把 2n 个 float 折叠为 n 个模值。
3. **步长单位**：`incx` 以**复数元素**为单位（cuBLAS 语义），地址为 `(i * incx) * 2` 个 float，非连续场景下无法用 `DataCopy` 一次搬入，需 SIMT 逐元素路径。

# 需求分析（required）

## 需求描述

实现句柄式 BLAS 接口，签名与 `cublasIcamin` 逐参数对齐（任务书 §2.3）：

```cpp
aclblasStatus_t aclblasIcamin(
    aclblasHandle_t handle, int n, const aclblasComplex* x, int incx, int* result);
```

语义要求（任务书 §2.1）：

| 项 | 要求 |
| --- | --- |
| 计算 | `result = argmin_i (|Re(x[k])| + |Im(x[k])|)`，`i = 1..n`，`k = 1 + (i-1)*incx` |
| 索引基 | **1-based**（Fortran 惯例） |
| 并列最小 | 严格返回**最小索引** |
| NaN | 跳过 NaN 元素；首元素为 NaN 时基准置 `FLT_MAX`（§3.2 补充说明，对齐 golden 语义） |
| quick return | `n == 0` 或 `incx < 1`（含 0 与负步长）：不触发 kernel，写 `result = 0`，返回 SUCCESS |
| 非法入参 | `n < 0` → `INVALID_VALUE`；`handle`/`result` 为 nullptr → `INVALID_VALUE`（handle 空按仓内约定返回 `HANDLE_IS_NULLPTR`）；`n > 0 且 incx ≥ 1` 时 `x == nullptr` → `INVALID_VALUE` |
| 精度判定 | 输出 INT32 索引，与 CPU 参考 golden **bit-exact 相等**（浮点容差表在整数输出下退化为精确一致判定） |
| 性能 | n=1048576 / 2097152 / 4194304（incx=1）平均单次耗时 ≤ **24.77 / 24.59 / 29.69 us**（warmup 后有效采样 >50 次取平均） |
| 声明位置 | `include/cann_ops_blas.h`（紧邻 `aclblasIsamin`），禁止 950PR 私有平行 API |

## 需求拆解

1. 复数模折叠 + 最小值/最小索引归约的 kernel 实现（`incx == 1` 向量路径、`incx > 1` SIMT 路径）。
2. 多核分块、workspace 部分结果与跨核归并的正确性（含跨核存储可见性）。
3. 边界与特殊值语义对齐 golden：n=0/1、2 的幂 ±1、全零、正负交替、±0、Inf、NaN、极端值。
4. CSV 驱动的测试工程（精度用例 + 性能用例 + 负向用例）与 us 级性能校验工具。
5. 设计文档、算子 README、自测报告、个人仓与 `cann/ops-blas` PR。

# 详细设计（required）

## 算子分析

### 数学公式

```
mag[i]  = |Re(x[k])| + |Im(x[k])| ,  k = 1 + (i-1)*incx ,  i = 1..n
result  = min{ i | mag[i] == min_j mag[j] }        （1-based，跳过 NaN）
```

GM 布局：`x` 为 `2n` 个 float 交错序列，`mag[i]` 由第 `2*i`、`2*i+1` 个 float 得到（`incx == 1`）。

### 支持数据类型

| 参数 | 类型 | 说明 |
| --- | --- | --- |
| x | COMPLEX64（分量 FLOAT32） | 只读，Device 内存 |
| incx / n | INT32（Host） | `incx` 以复数元素为单位 |
| result | INT32（Device 单值） | 1-based 索引或 0 |

### 支持形状

逻辑一维 `[n]`，物理长度 `1 + (n-1)*|incx|`；不涉及广播、不涉及高维 leading dimension。

## 算子实现

### 总体结构

沿用已合入 `isamin` 的**两 kernel + 单核快路**骨架（`blas/iamin/arch35/isamin_kernel.cpp:370-386`），按复数语义替换计算体：

```
incx == 1 :
   numBlocks == 1  →  icamin_small_kernel <<<1>>>            （直写 result，省一次 launch）
   否则              →  icamin_aiv_kernel   <<<numBlocks>>>   （分块归约，写 workspace slot）
                        icamin_reduce_kernel<<<1>>>           （跨核归并，写 result）
incx >  1 :
                      →  icamin_simt_kernel  <<<numBlocks>>>   （SIMT 逐元素，slot 同布局）
                        icamin_reduce_kernel<<<1>>>
n == 0 / incx < 1 :  host 侧 H2D 写 0，不 launch 任何 kernel
```

slot 布局与 `isamin` 一致：每核 2 个 float —— `[2*blockIdx] = 块内最小模`，`[2*blockIdx+1] = 该模的下标（以 float 位型存放 uint32_t）`。

### host 侧设计

**参数校验**（先于任何设备操作，返回码见上表）。

**分核策略**：遵循"满核 + 余数归尾核"，与 `isamin_host.cpp:49-71` 同式：

```
aivCoreNum = GetAivCoreCount()                    // 动态获取，禁止硬编码 56
numBlocks  = min(n, aivCoreNum)
perCoreN   = totalN / numBlocks                   // totalN = n * |incx| 展平后的复数元素跨度上限
lastCoreN  = perCoreN + totalN % numBlocks
useCoreNum = numBlocks                            // 每核都有活，slot 必然被写满
tileSize   = ICAMIN_TILE_COMPLEX                // 16384 复数元素/单槽，见 UB 预算
```

`useCoreNum == numBlocks` 是刻意选择：避免出现"某核无数据不写 slot，而 reduce 仍读该 slot"读到脏值的整类问题。

**tile 与 UB 预算（单核，`incx == 1`）**：折叠为 RegBase 寄存器累加，无 mag 中间数组、无逐 tile ReduceMin，UB 只存输入槽 + 64 通道累加器 + 收尾临时：

| buffer | 字节 |
| --- | --- |
| inLocal（(2×tileSize + 128) float，双载整对含尾部 slack） | (2×16384+128)×4 = 131584 |
| accVal / accIdx（各 64 元素，分槽便于 ReduceMin 取整 LocalTensor） | 2 × 256 |
| out（32B 块）/ work（ReduceMin 中间层） | 256 + 512 |
| 合计 | ≈ 132864 B ≈ 130 KB |

dav_3510 单核 UB 为 248 KB（253952 B，`kernel_utils_constants.h` 的 `__NPU_ARCH__==3510` 分支 `TOTAL_UB_SIZE = 248*1024`）。`tileSize = 16384` 时单槽 + 累加器合计 ≈ 130 KB，余量充足，host 侧按常量固定下发，不写死（原先按 `ReduceMin repeatTimes≤255` 推出的 16320 上限随逐 tile ReduceMin 一起废弃）。

**workspace 需求**：`totalFloats = numBlocks * 2`，按 64 float 对齐：`requiredBytes = CeilAlign(totalFloats, 64) * 4`（≤ 512 B）；不足则 `OP_LOGE` + `ACLBLAS_STATUS_EXECUTION_FAILED`。不额外放大分配。

### kernel 侧设计

**阶段 1（`icamin_aiv_kernel`，每核）** —— RegBase 寄存器折叠，无 mag 中间数组：

```
seed: 每 lane accVal = +inf, accIdx = 0（IcaminSeedVf）
for tile in 本核各 tile:
    DataCopy(inLocal, inGM[2*currOffset], 2*tileCount)          # 整块 2n float（对齐）或 DataCopyPad（尾块）
    SetFlag<MTE2_V>；WaitFlag<MTE2_V>                            # 拷贝→折叠 一次事件往返
    IcaminAccVf: 双载 DIST_DINTLV_B32 拆 128 float → 64 Re + 64 Im
                 Abs+Abs+Add → 64 模值；Compare<LT> + Select 与 lane 累加器 (val, idx) 合并
                 lane j 拥有位置 j, j+64, j+128,...（跨步），严格小于才替换 → lane 内并列取小下标
    SetFlag<V_MTE2>（仅当该槽后续还会复用，避免内核退出残留未消费标志——残留会死锁收尾的标量 UB 读）
尾块：DataCopyPad 不补 +FLT_MAX，改用 lane 尾掩码（UpdateMask 按实际计数）令其保持 +inf 种子、天然不参与 min
CombineSlice: ReduceMin<float> 得 bestVal → 若 bestVal ≥ FLT_MAX（全 NaN/Inf/饱和）走 SIMT 逐元素回扫
              否则 IcaminTieFixVf 把非最小 lane 的 idx 改 INT32_MAX → ReduceMin<int32_t> 得最小全局下标
PublishSlice: SetValue(outLocal) + DataCopyPad 走 MTE3 写 (val, idx) 到 ws slot，不用标量 GM 赋值
```

折叠体为 `__simd_vf__` 内联调用（`asc_vf_call` 不带 dim3 即普通内联，天然与后续同 pipe 有序，无需额外事件对）。**显式 `uint16_t loopNum` 计数 `for`** 驱动（`__simd_vf__` 内数据相关的 `while` 实测只会执行一次）；`UpdateMask(uint32_t&)` 取非常量左值引用，须先落具名变量。

**同步与事件**：`WaitFlag/SetFlag<HardEvent::MTE2_V>` 管拷贝→折叠，`WaitFlag/SetFlag<HardEvent::V_MTE2>` 管槽复用，收支严格平衡（见上）；`DataCopy` 回读后 `GetValue` 前由 `ReduceMin` 自身的 V→S 同步保证。

**阶段 2（`icamin_reduce_kernel`，单核）** —— `DataCopy` 读回 `useCoreNum * 2` float 到 UB → `asc_vf_call` 起 64 线程 SIMT 树归约（`IcaminReduceSimtCompute`，线程内先合并跨步槽位，`asc_syncthreads` 树形再合并；NaN 跳过、并列取小下标，槽位 value=NaN 且 idx=0 表示该 slice 全 NaN）→ `threadIdx.x == 0` 写 `int32(result) = bestIdx + 1`。

**`incx > 1` 路径**：SIMT 内核 `icamin_simt_kernel` 以 `dim3{nthreads,1,1}` 启动，线程按 `stride = incx * 2`（float 单位）跨步读取，先线程内求 `(val, idx)`，UB 内树形归约后由 `threadIdx.x == 0` 写 slot。`nthreads = min(CeilAlign(perCoreN / SIMT_MIN_THREAD_NUM, SIMT_MIN_THREAD_NUM), SIMT_MAX_THREAD_NUM)`。

### 跨核同步方案选型（本任务实测依据）

| 方案 | 结论 | 依据 |
| --- | --- | --- |
| A. 两 kernel（阶段 1 全部块写完 → kernel 边界 → 阶段 2 单核读） | **采用** | kernel 边界由调度器保证前序全部块完成且 GM 写入对后续 kernel 可见；与已合入 `isamin` 一致，评审基线明确 |
| B. 单 kernel 内 `AscendC::SyncAll<true>()` + block 0 归并 | 不采用 | 两条独立理由。① dav_3510 上 `SyncAll` → `SoftSyncAllImpl` 仅做 `pipe_barrier(PIPE_ALL)`（`impl/basic_api/dav_3510/kernel_operator_sync_impl.h`），只保证本核流水线有序，不提供跨核标量 GM 存储的可见性：实测以标量 `__gm__` 赋值写 slot 时，同一 n=8 输入在多次运行中返回 7 / 4 / 2（未写/陈旧 slot 被当作候选），而同输入的 SIMT 两 kernel 路径稳定正确。② 融合形态要求"所有启动的 core（含非活跃 core）都必须到达 SyncAll"（仓内唯一 landed 融合先例 `blas/nrm2/arch35/snrm2_ex_kernel.cpp:505-553`，注释见 :543），任何 `computeNum == 0 → return` 的提前退出都会破坏该前提 |
| C. 单 kernel + MTE3 `DataCopyPad` 写 slot + `SyncAll` | 不采用 | 可见性仍未获规范保证；收益仅省一次 launch（≈4.8 us），而预算为 24.77 us，风险收益比不划算 |

方案 A 的两次 launch 成本用**单核快路**对冲：`numBlocks == 1` 的小 n 用例走 `icamin_small_kernel` 一次 launch 直写 result（对齐 `isamin_kernel.cpp:376-377`）。

### 性能预算核对

| 项 | 量级 |
| --- | --- |
| 单次 kernel launch 下界（msprof 实测，950PR） | ≈ 4.8 us |
| 阶段 1 数据量（n=1048576，incx=1） | 8 MB，56 核均分 |
| §3.3 预算（n=1048576） | 24.77 us |
| 结论 | 8 MB / 56 核在 HBM 带宽下为个位数 us 级，两次 launch（≈9.6 us）+ 计算可容纳于预算内；瓶颈在阶段 1 是否走向量流水（`Abs`+`PairReduceSum`+`ReduceMin`）而非 SIMT 逐元素扫描——实测 SIMT 扫描在 1M 上约 20 us，单独即逼近预算，故 `incx == 1` 必须使用向量路径 |

## 支持硬件

| 支持的芯片版本 | 是否支持 |
| --- | --- |
| Ascend 950PR / 950DT（arch35 / dav_3510） | 支持 |
| Atlas A3 训练/推理系列 | 不支持（不提供 arch22 实现） |
| Atlas A2 训练/推理系列 | 不支持 |

## 算子约束限制

1. `n ≥ 0`；`n = 0` 与 `incx < 1`（含 0、负步长）为 quick return，不 launch kernel，`result = 0`。
2. `incx` 以复数元素为单位；不支持负步长反向遍历（按 quick return 处理）。
3. 并列最小值返回最小索引，实现不得依赖归约顺序产生歧义（每层合并均带 `==` 时取小下标分支）。
4. NaN 元素跳过；全为 NaN 时输出 1（对齐 golden）。
5. workspace 由 handle 统一管理，需满足 `CeilAlign(numBlocks*2, 64) * 4` 字节。
6. UB 单核用量受 `tileSize` 约束，超限时须在 host 侧下调 `tileSize` 而不是增加 kernel 复杂度。
7. 接口声明只进 `include/cann_ops_blas.h`，不得另立 950PR 私有头。

# 可维可测分析

## 精度标准 / 性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度 | 输出 INT32 索引与 CPU 参考 golden `EXPECT_EQ` 精确一致（浮点容差表退化为 bit-exact：rtol 2^-10 / atol 2^-16 / required_matched_ratio 0.99 → 单值判定） | `cann/opbase` ops_precision_standard/experimental_standard.md + 任务书 §3.2 |
| 性能 | 三个性能 case 平均单次耗时 ≤ 24.77 / 24.59 / 29.69 us；warmup 后有效采样 >50 次取平均，单位 us | 任务书 §3.3（标杆 = `gpu_baseline.csv` 的 H100 耗时 / 0.4 倍率） |
| 内存 | 任务书 §3.4 标注"不涉及"；自测报告仍给出 workspace 与 UB 占用 | 任务书 §3.4 / §4-3 |
| 用例 | CSV 驱动 GTest：基础小 shape、shape 扫描、填充模式、对齐偏移、边界与负向、Inf/NaN、性能/内存用例 | 任务书 §3.5-4 |

**自测结果（随验收进度回填，未回填项即为 ⚠ 未验证，不作为通过依据）**

| 项 | 数值 | 状态 |
| --- | --- | --- |
| §3.3 单次 kernel 耗时（Release 构建，tile=4096 TQue 双缓冲 + 纯 SIMD 折叠，commit `e3405b9`） | 10.02 / 14.72 / 23.26 us（限值 24.77 / 24.59 / 29.69） | ✅ 三档全部达标（msprof `task_time` 口径，81 实例，aiv+reduce 求和均值；回显见 `/workspace/final_run.log`） |
| 同口径 aiv / reduce 拆分 | 7.75+2.27 / 12.39+2.33 / 20.84+2.42 us | ✅ 实测；4M 档 33.5 MB / 20.84 us ≈ 1.61 TB/s，已在访存带宽墙附近 |
| 精度（旧 Debug 构建，VF 折叠版） | 1206 / 1206 | ✅ 实测通过（`[  PASSED  ] 1206 tests.`） |
| 精度（当前 Release 构建，清理诊断代码后复测） | 1206 / 1206 | ✅ 实测通过（`index mismatch` 计数 0；此前 1184/1204 的 20 例失败已定位修复，见下条） |
| workspace 占用 | ≤ 512 B（`CeilAlign(numBlocks*2,64)*4`） | 设计值 |
| 单核 UB 占用 | ≈ 101 KB：双槽 2×(2·4096)×4=65536 B + re/im 2×16384 + mask 4096 + work 288 + slot/out 96；dav_3510 UB 上限 248 KB | 实测值 |

> **§3.3 达标的关键（2026-09-19 实测，推翻此前"物理不可达"结论）**：本仓库 `CMakeLists.txt:66` 把 `CMAKE_BUILD_TYPE` 以 `FORCE` 写死为 `Debug`，内核因此以 `-O0` 编译，实测比 Release 慢约 10 倍（4M aiv 73 → 20.9 us）。此前所有"拷贝与折叠无法重叠""固定开销 18.7 us 无法消除"的测量都建立在 Debug 构建上，其绝对值与结论均不成立：改为 Release 后，`TQue<VECIN,2>` 双缓冲 + 纯 SIMD（`Abs`→`DeInterleave`→`Add`→`ReduceMin(calIndex)`）折叠确实把 MTE2 拷贝与 V 折叠重叠了起来，三档标杆全部达标。注意 `FORCE` 意味着命令行 `-DCMAKE_BUILD_TYPE=Release` 无效，只能改文件。

> 折叠必须是纯 SIMD 的 LocalTensor 算子：`asc_vf_call`/`__simd_vf__` 寄存器折叠与 MTE2 拷贝确实不能重叠（这是早期 Debug 测量的真实成因）。NaN 一律改写为 +Inf（`Compare(EQ)` 自比较 + `Select`），因为 `ReduceMin` 对 NaN 的处理在两种构建模式下不一致；参考实现的 (FLT_MAX, 首元素) 种子规则则在 `CombineSlice` 里用一次条件标量 GM 读回补，避免走 SIMT 兜底扫描（该兜底路径在 Release 下会 `ACLBLAS_STATUS_INTERNAL_ERROR`）。

> **Release 下才暴露的两个缺陷（2026-09-20 定位，均已在 `e3405b9` 修掉）**
>
> 1. 单 block 发布缺 `PipeBarrier<PIPE_ALL>`。`PublishSlice` 的 `writeMode_==1` 分支用 `Duplicate`（V 流水）写 UB，再用 `DataCopyPad`（MTE3）读同一块 UB，两者之间没有硬件事件依赖：`-O0` 下被时序侥幸串行化，Release 下 MTE3 抢跑，发出的是 VECCALC 槽位的**残留字节**（因此 n=1 这种唯一解为 1 的用例返回 0x7F800000=+Inf，其余为上一轮内核留下的幅值/索引位）。定位手法：同一个二进制里用环境变量切换"强制 V 折叠 / 强制标量扫描"两条完全不同的计算路径，20 例失败输出逐字节相同 ⇒ 计算无关，问题在发布。同族参考实现 `isamin_kernel.cpp` 的 `WriteResult` 是同形写法（同样缺屏障），不能照抄了事。
> 2. 短切片不能用带索引的 `ReduceMin`。其收尾的跨 lane tie 通道只对被 gather 的 lane 生效，其余 lane 保留 scratch 残值；当切片短到只有少数 lane 承载真数据时，返回的索引由残值决定（值槽仍精确）。处置：`calNum_ < 512`（8 个 64-lane 折叠宽度）改走标量扫描，语义完全对齐参考实现（首元素无条件登记、后续须严格 `<`，故 tie 归最小下标）；该阈值之上的切片每 lane 必有真数据，且 §3.3 最小切片为 18728 complex，不受影响。

## 验收工具说明（已知缺陷与处置）

任务随附的 `test_cases/verify_performance.py` 以 GTest 输出的整数毫秒为采样口径：
`re.match(r"\[\s+OK\s+\]\s+\S+/(\S+)\s+\((\d+)\s*ms\)", line)`。本算子单次耗时为 us 级，整数毫秒恒为 0 或被判 `NO_REF`，而脚本在 `NO_REF` 时仍以 `exit 0` 结束——即该脚本**无法**按 §3.3 判定。处置：新增 `verify_performance_us.py`，以 msprof `task_time_*.csv` 的 `task_time(us)` 为采样口径（PerfBench 循环 81 实例，跳 11 warmup、取 60 有效样本，按 kernel 分组求均值再求和），基线取 `gpu_baseline.csv` 中 `icamin-base-*` 三行（n=1048576/2097152/4194304 → 24.77/24.59/29.69 us），`avg <= limit` 判 PASS，有未达标 `exit 1`、无样本/无基线 `exit 2`。实测产出 `results_icamin_us.csv`（三 case 均 FAIL），并在自测报告中给出两种口径差异说明。

## 兼容性分析

新增接口，无历史行为需要兼容；与同族 `aclblasIsamin` 共用 `blas/iamin/` 目录与 README 文档结构，不改写既有 `isamin` 代码路径。`incx == 1` 与 `incx > 1` 两条路径必须给出一致结果（测试用例覆盖交叉比对）。

## 测试用例设计与两套 CSV 的分工

仓库对 CSV 驱动用例有硬性命名口径：`case_name` 只用 `TC_L0_ / TC_L1_ / TC_L2_`（`agent/skills/repo-test-develop/references/test/{op}/arch35/op_test.csv`；已合入的 `test/isamin/arch35/isamin_test.csv` 即 22 L0 + 28 L1 + 11 L2，且**不含性能用例**）。任务随附的 `test_cases/icamin_test.csv`（1200 条）使用私有前缀 `TC_L0 / TC_SQ / TC_INC / TC_FL / TC_ED / TC_EX / TC_PF`，其 README 已说明它们分别映射到 L0~L6 能力层级。二者按用途分工，不做混用：

| 用途 | 用例集 | 命名 | 说明 |
| --- | --- | --- | --- |
| 社区任务 IT 系统验收（交付件 2/3） | 随任务 1200 条全集 | 原样保留（TC_SQ/TC_INC/TC_FL/TC_ED/TC_EX/TC_PF…） | 1000 条精度 + 200 条性能；`gpu_baseline.csv` 提供 H100 基线；性能判定用本任务自研 us 级脚本（见下） |
| `cann/ops-blas` PR（§5） | 由 1200 条裁剪重编号 | 仅 TC_L0/L1/L2 | 剔除 200 条 TC_PF：仓内规则要求提交前删除性能/白盒临时用例（`repo-build-guide`、`repo-test-develop`）；`TC_SQ→L1`、`TC_INC/TC_FL/TC_ED→L0/L2`、`TC_EX` 按能力抽样并入 L1/L2，保持全集语义覆盖不减 |

配套 6 文件结构（对齐 `repo-test-develop` 与 `test/isamin/`）：`test/iamin/icamin/{icamin_param.h, icamin_golden.h, CMakeLists.txt}` + `arch35/{icamin_npu_wrapper.h, icamin_test.cpp, icamin_test.csv}`。要点：golden 放芯片无关的 `icamin_golden.h`（手写 cblas 风格循环，NaN 跳过 + tie 取小索引）；结果用整数 `EXPECT_EQ`，不走浮点 Verifier；空 `handle` 用 `TEST_F` 表达、不入 CSV；`RANDOM` 必须带显式值域（如 `RANDOM_NORM_5_5`）；`test/` 侧 CMakeLists 调 `ops_blas_add_gtest_tests()`（`blas/` 侧由 GLOB 自动收集 arch35，无需 CMakeLists）。

## 合入流程与提交前自查

| 环节 | 要求 |
| --- | --- |
| 前置 | 需有 `Requirement/需求建议` Issue 并经 blas SIG 评审分配路径（`CONTRIBUTING.md:32-59`） |
| 必交 | Host + Kernel + 测试 + `blas/iamin/README.md` 增补；`include/cann_ops_blas.h` 新增 `aclblasIcamin` 声明（紧邻 `aclblasIsamin`，:195） |
| README | 编辑既有 `blas/iamin/README.md`（不可新建文件）：接口表加行 + `产品支持情况`（Ascend 950PR：支持）/`函数原型`（单行、无分号）/`参数说明`（标注 Host/Device 内存）/`约束说明`（NaN 跳过、tie 取小索引、`incx < 1` 写 0）/`调用示例`（RAII，本地实际跑通）；删除模板说明注释块 |
| 编码门禁 | clang-format v16（`.pre-commit-config.yaml`）、`check_pragma_once.sh`、许可证头（R9，CSV 豁免）、禁 `printf/cout` 改 `OP_LOG*`、禁 `aclrtMalloc` 申请 workspace（用 `GetEffectiveWorkspace/GetEffectiveWorkspaceSize`）、禁 host 重复定义 `GetAivCoreCount`、除零前判 `coreNum == 0`、`{op}_kernel_do` 经 `{op}_kernel.h` 引入而非 `extern` 前向声明、launch 后禁 `aclrtSynchronizeStream` |
| 规模 | 单函数 CCN ≤ 20 / 嵌套 ≤ 5 / NBNC ≤ 50（R5/R6/R7）；按 `Validate/Launch/CopyIn/Compute/WriteWorkspace` 拆分而非堆进 `Process()` |
| 提交信息 | `type(scope): 中文描述`，正文含 `## issue 关联 / ## 问题 / ## 根因 / ## 修复 / ## 验证`（参照 master 现例 commit 7a80f5c） |
| 门禁交互 | PR 评论区 `compile` 触发 CI；`/lgtm` → `/approve`；codecheck 误报交 SIG 屏蔽 |
| 自查四项 | C++ 编程规范符合性 / `build.sh --ops --soc` 编译通过 / `--run` 用例全过 / Markdown 语法 |

