# aclblasIcamin 算子设计文档（Ascend 950PR / ops-blas）

> 算子 `aclblasIcamin`：COMPLEX64 向量最小模元素索引，`result = argmin_i (|Re(x[k])| + |Im(x[k])|)`，k = 1+(i-1)·incx，**1-based** 索引，并列取最小索引
> 硬件 Ascend 950PR（arch35 / DAV_3510）；CANN 9.1.0；Ascend C kernel 直调
> 代码落 [cann/ops-blas](https://gitcode.com/cann/ops-blas) `blas/iamin/arch35/`（与同族实数接口 `aclblasIsamin` 同目录，复用其 arch35 归约框架）；测试落 `test/iamin/icamin/arch35/`
> 接口声明入 `include/cann_ops_blas.h` 供各产品线共用，禁止 950PR 私有平行接口
> 团队 `zhangfeng1133`。本文为**开发前设计文档**，只含设计方案、验收标准与验证方法，不含实测结果

# 需求背景（required）

## 需求来源

| 项 | 内容 |
| --- | --- |
| 任务 | 9月社区任务-aclblasIcamin算子开发（950） |
| 对标接口 | cuBLAS `cublasIcamin`（语义最高优先级来源）；Netlib 标准 BLAS 无 icamin 例程（参考同族 `isamin.f`） |
| 适配硬件 / CANN / 三方软件 | Ascend 950PR / CANN 9.1.0 / 无额外三方依赖（golden 由测试工程内 cblas 风格 CPU 参考实现生成，随测试工程提供） |
| 交付仓库与目录 | https://gitcode.com/cann/ops-blas；算子 `blas/iamin/arch35/`，测试 `test/iamin/icamin/arch35/`（含 csv） |
| 交付件 | ① 设计文档（PR 至 cann-competitions 并通过评审）；② 自测用例及测试代码（README 保证可复现）；③ 自测报告（精度/性能/内存数据与截图）；④ 待验收代码地址（个人仓邀请 Ascend-CANN） |

## 背景介绍

### aclblasIcamin算子功能分析

```
result = argmin_{i=1..n} ( |Re(x[k])| + |Im(x[k])| ),   k = 1 + (i-1)*incx
```

| 项 | 约定 | 说明 |
| --- | --- | --- |
| 索引基准 | **1-based** | 兼容 Fortran / BLAS 惯例，与 cuBLAS 对齐 |
| 复数"模" | `\|Re\| + \|Im\|` | BLAS icamin 惯例（1-范数），**非**欧氏模 `sqrt(Re²+Im²)` |
| 并列(tie) | 取**最小索引** | 严格小于才更新（`abs1 < best`），后续相等元素不更新 |
| 复数存储 | 实虚各 FLOAT32、交错 | 类型 `aclblasComplex` 见 `include/cann_ops_blas_common.h`；n 个复数 = 2n 个 float |
| 输出 | 单个 INT32 索引 | Device 标量；`[1,n]` 或 quick return 时的 0 |
| 精度判定 | **整数索引 bit-exact** | 最小值与最小索引是离散量，归约顺序不影响结果，非浮点容差比对 |
| 负步长 | 不反向遍历 | `incx < 1`（含 0 与负步长）为 quick return |

与同族算子的关系：在 `aclblasIsamin` 的「Abs → ReduceMin」流水上，把 FP32 的 `Abs` 换成 COMPLEX64 的 `|Re|+|Im|`；Host 校验、分核、workspace 二段归约、SIMT 非连续路径对齐 isamin，复数模与特殊值语义对齐 icamax。

| 算子 | 输入 | 归约 | 模定义 | 仓内参照 |
| --- | --- | --- | --- | --- |
| `aclblasIsamin` | FP32 | ReduceMin | `\|x\|` | `blas/iamin/arch35/`，**主复用框架** |
| `aclblasIcamax` | COMPLEX64 | ReduceMax | `\|Re\|+\|Im\|` | `blas/iamax/arch35/`，复数模与 NaN/Inf 参照 |
| `aclblasIcamin` | COMPLEX64 | **ReduceMin** | `\|Re\|+\|Im\|` | 本任务新增 |

### 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | dtype | 排布 | 维度 | 值域 | 异常行为 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| handle | 输入 | 库上下文句柄，携带 stream，Host 内存 | scalar | - | - | - | 有效的已创建句柄 | nullptr → `HANDLE_IS_NULLPTR` |
| n | 输入 | x 的复数元素个数，Host 内存 | scalar | int | - | - | n ≥ 0 | n < 0 → `INVALID_VALUE`；n = 0 为合法 quick return（不触发 kernel，写 result = 0 并返回成功） |
| x | 输入 | 复数向量，Device 只读 | tensor | COMPLEX64 | ND | 逻辑一维 [n]，物理 1+(n-1)·\|incx\| | 实虚取 FLOAT32 全集 | n > 0 且 incx ≥ 1 时 nullptr → `INVALID_VALUE` |
| incx | 输入 | 相邻元素步长，Host 内存 | scalar | int | - | - | incx ≥ 1 正常计算 | incx < 1（含 0 与负步长）为合法 quick return（写 result = 0） |
| result | 输出 | 最小模元素的 1-based 索引，Device 内存 | scalar | INT32 | - | 单值标量 | 0 或 [1,n] | nullptr → `INVALID_VALUE` |

返回值 `aclblasStatus_t`，语义与 `include/cann_ops_blas_common.h` 一致（`SUCCESS` / `INVALID_VALUE` / `HANDLE_IS_NULLPTR`）。

# 需求分析（required）

## 需求描述

在 Ascend 950PR 上用 Ascend C 实现句柄式 BLAS 接口 `aclblasIcamin`：Host 参数校验与 tiling → 经 handle 绑定 stream 下发 arch35 kernel → Device 写回 1-based INT32 索引。精度为整数索引 bit-exact；性能为 COMPLEX64 输入下的平均单次耗时（us），三档标杆 24.77 / 24.59 / 29.69 us（n = 1048576 / 2097152 / 4194304，incx = 1），**不得高于标杆**。

## 需求拆解

| 编号 | 子项 | 验收要点 |
| --- | --- | --- |
| R1 | 公开 API | `include/cann_ops_blas.h` 新增声明，签名与 `cublasIcamin` 逐参数对齐 |
| R2 | 功能正确 | 1200 条 CSV 精度用例 PASS；tie / quick return / 负 n / 空指针语义对齐 |
| R3 | 框架复用 | 落 `blas/iamin/arch35/`，复用 isamin 的 host 分核与二段归约，不重复造状态机 |
| R4 | 复数模 | `\|Re\|+\|Im\|`，禁止欧氏模；模写独立 buffer，不 in-place 覆盖源 |
| R5 | 测试工程 | `test/iamin/icamin/arch35/` CSV 驱动 GTest，列格式对齐 `isamin_param.h` |
| R6 | 性能 | TC_PF 性能用例平均耗时 ≤ 三档标杆；采集前 warmup 且有效采样 > 50 次 |
| R7 | 文档与交付 | ops-blas 规范 README + 产品支持表标注 950PR 支持；自测报告含精度/性能/内存数据与截图 |

# 详细设计（required）

## 算子分析

### 数学公式

$$
\text{result} = 1 + \underset{i \in [1,n]}{\arg\min}\ \bigl|Re(x[k])\bigr| + \bigl|Im(x[k])\bigr|,\qquad k = 1 + (i-1)\cdot incx
$$

golden 由测试工程内 cblas 风格 CPU 参考循环生成（标准 BLAS 无 icamin 例程）：

```cpp
// x 以 (re, im) 交错存储；严格小于才更新 → 并列取最小索引
float best = FLT_MAX;      // 首元素为 NaN 时基准保持 FLT_MAX
int   bestIdx = 1;         // 1-based
for (int i = 1; i <= n; ++i) {
    const float* p = (const float*)x + 2 * (i - 1) * incx;
    float abs1 = fabsf(p[0]) + fabsf(p[1]);
    if (abs1 < best) { best = abs1; bestIdx = i; }   // 严格 <
}
result = (n <= 0 ? 0 : bestIdx);
```

与 `aclblasIcamax` 的 golden 严格对称：icamax 用 `>` 更新，icamin 用 `<` 更新。

### 支持数据类型

输入 COMPLEX64（`aclblasComplex` = 2 × FLOAT32，交错），输出 INT32。中间量：模与最小值为 FLOAT32，索引为 INT32/UINT32——索引**不得以 FLOAT32 承载**，`n > 2^24` 时 float 无法精确表示整数索引。

### 支持形状

一维向量：逻辑 `[n]`（n ∈ [0, 2^24]，上限为任务包内存预算的设计选择），物理长度 `1 + (n-1)·|incx|`。

| 访存形态 | 条件 | 路径 |
| --- | --- | --- |
| 连续 | `incx == 1` | 向量（AIV）主路径；性能用例只在该形态考核 |
| 非连续 | `incx > 1` | SIMT 步长访存路径 |
| 退化 | `n == 0` 或 `incx < 1` | quick return，不触发 kernel |

## 算子实现

### 实现方案

沿用 ops-blas iamin 族 arch35 二段归约结构：

```text
Host (icamin_host.cpp): 校验 → quick return → CalcTiling → workspace 视图 → icamin_Kernel<<<numBlocks, stream>>>
Kernel 三分支（对齐 isamin_kernel_do 的 dispatch 风格）:
  ├─ n ≤ smallThreshold  → icamin_small_kernel   单 block 核内归约，直写 result
  ├─ incx == 1 && n 较大 → icamin_aiv_kernel     模 → ReduceMin → per-core 部分结果
  └─ incx > 1            → icamin_simt_kernel    每线程一元素 + 块内树归约
        （后两者均接）      icamin_reduce_kernel   跨核 min+index 二段归约 → result
```

```text
ops-blas/
├── include/cann_ops_blas.h                      # 新增 aclblasIcamin 声明
├── blas/iamin/arch35/{icamin_host.cpp, icamin_kernel.cpp, icamin_kernel.h, icamin_tiling_data.h}
├── blas/iamin/README.md                         # 补充 icamin 章节 + 产品支持表
└── test/iamin/icamin/{icamin_golden.h, icamin_param.h, icamin_npu_wrapper.h}
    └── arch35/{icamin_test.cpp, icamin_test.csv}
```

#### 3.2.1 Host 侧设计

接口声明（`include/cann_ops_blas.h`，本任务新增，仅此一处公开声明）：

```cpp
aclblasStatus_t aclblasIcamin(aclblasHandle_t handle, int n,
                             const aclblasComplex* x, int incx, int* result);
```

校验顺序固定，保证 quick return 优先于 x 指针检查（与测试集期望一致）：

| 序 | 判定 | 条件 | 返回 / 动作 |
| --- | --- | --- | --- |
| 1 | 句柄 | `handle == nullptr` | `HANDLE_IS_NULLPTR` |
| 2 | 维度 | `n < 0` | `INVALID_VALUE` |
| 3 | 输出指针 | `result == nullptr` | `INVALID_VALUE`（quick return 亦需写回，故先判） |
| 4 | quick return | `n == 0 \|\| incx < 1` | 绑定 stream 上 `aclrtMemsetAsync(result, sizeof(int), 0, stream)`，返回 `SUCCESS`；**不检查 x、不下发 kernel** |
| 5 | 输入指针 | `x == nullptr`（此时 n > 0 且 incx ≥ 1） | `INVALID_VALUE` |
| 6 | 正常计算 | 其余 | 走 tiling 与 kernel |

quick return 用 stream 内写零而非 CPU 拷贝，避免引入 Host 同步；`incx < 1` 一律置 0 且**不反向遍历**（与 axpy/nrm2 族不同，属本族约定）。校验失败不触碰 Device。

Tiling 字段对齐 `IsaminTilingData`（`totalN / perCoreN / lastCoreN / useCoreNum / tileSize / incx / nthreads`），避免为新算子另造结构：

| 项 | 策略 |
| --- | --- |
| 核数 | `useCoreNum = min(CeilDiv(n, minNPerCore), GetAivCoreCount())`；`minNPerCore` 为单核最小处理量（口径与 isamin 一致，避免为极小 n 起过多核），优先满核、核间近似均分 |
| 尾块 | `lastCoreN` 承接余量；核内按 `tileSize` 分块，最后一块按 actual count 归约 |
| tileSize | 沿用 isamin 的 UB tile 口径（`FP32_MAX_DATA_COUNT` 同源），保证 `ReduceMin` repeat 对齐 |
| 分支 | `incx == 1` 走向量主路径；`incx > 1` 把 `nthreads` 交给 SIMT；`n ≤ smallThreshold` 走 small kernel |

Workspace：每核写一对 `(minVal: float32, idx: int32)` 的 per-core 部分结果，按 64 B 对齐排布，尺寸与 isamin 一致，由 Host 内部申请/释放，**不作为公开契约参数暴露**。下发：`icamin_kernel_do<<<numBlocks, nullptr, stream>>>` 沿句柄绑定的 stream 异步执行，二层归约在原 stream 串行下发，不引入额外同步点；读回 Device 结果前由调用方同步 stream。

#### 3.2.2 Kernel 侧设计

数据流（连续路径，`incx == 1`）：

```text
GM x[2n] (re,im 交错) → DataCopy/Pad 入 UB → 拆实虚(DINTLV 或 stride-2 取数)
  → mag[i] = |re[i]| + |im[i]|        # FLOAT32，写入独立 magBuf
  → ReduceMin(mag, calIndex=true) → (tileMin, tileLocalIdx)
  → tile 间合并：严格 < 或（相等 && 更小全局索引）→ 每核写 workspace
  → icamin_reduce_kernel 跨核合并 → result = bestIdx（1-based）
```

| 步骤 | Ascend C API | 说明 |
| --- | --- | --- |
| 搬运 | `DataCopy` / `DataCopyPad` | GM→UB；非 32B 对齐尾块用 Pad，padding 不参与比较 |
| 解交错 | `DINTLV`（RegBase）或 stride-2 逐元素取值 | 复数拆成 re/im 两路 |
| 模 | `Abs` + `Add` | 输出 FLOAT32 `mag` |
| 块内归约 | `ReduceMin<T>(dst, mag, work, count, calIndex=true)` | 同时得到最小值与块内索引 |
| 跨核归约 | `icamin_reduce_kernel` | 读 workspace，严格 `<` + 最小索引 tie-break |

两条实现红线：① **模必须写入独立 `magBuf`**，禁止 in-place 覆写源数据（icamax 曾因 Reg 路径 in-place 覆盖出现精度故障，沿用其修复后写法）；② **索引换算在整数域完成**：`ReduceMin` 返回的 `localIdx` 为块内相对索引，须按 `globalIdx = tileStart + localIdx` 换算，最终 `result = globalIdx + 1`；任何把索引放进 float 比较键的实现都必须给出整数不丢失的证明。

索引语义待验证项：`ReduceMin(calIndex=true)` 返回的块内索引是"首个最小值"还是"末个最小值"未在设计阶段假定成立。若 API 语义不确定或取末个，tie-break 必须在合并层显式纠正——对相等的最小值一律取更小的**全局**索引，必要时对包含最小值的 tile 做二次逐元素复扫定位首个最小索引。该行为由 TC_L0_005/006（全零填充、多元素模相同、期望索引 1）等 tie 用例作为准入门禁。

| NaN / Inf 场景 | golden 与实现 |
| --- | --- |
| 一般 NaN 元素 | 跳过（`NaN < x` 与 `NaN > x` 均为假，不更新基准） |
| 首元素为 NaN | 基准置 `FLT_MAX`，后续有限元素可正常成为最小值 |
| 全 NaN | 无元素满足严格 `<`，索引保持初始值 1（实现与 golden 同口径） |
| +Inf / 极大值 | 正常参与比较；多个相等时取最小索引 |

任务书把 NaN 口径列在随任务事实表的**开放问题 Q7（待研发确认）**；实现阶段以 golden 同实现 + 逐 case 比对作为准入门禁，若 Q7 结论变更则同步修订 golden 与 kernel 两处。

非连续路径（`incx > 1`）：复用 isamin 的 SIMT 框架，每线程处理一个复数 `x[(offset+i)*incx]`（读 `p[0]`、`p[1]` 两个 float），线程内算 `abs1 = |re|+|im|`，块内树归约取 (min, idx)（`IsaminSimtTreeReduce` 的 min 语义，逻辑一致），再写 workspace 走二段归约；选模时优先对齐 isamin 的 `nthreads` tiling，不移植 icamax 的 max 专属调优策略。

| dispatch 维度 | 处理 |
| --- | --- |
| 规模特化 | `n ≤ smallThreshold` → small kernel（单核直写 result）；否则二段归约 |
| 访存特化 | `incx == 1` → 向量主路径；`incx > 1` → SIMT 步长路径 |
| 数据类型 | 仅 COMPLEX64/INT32 一种组合，无需额外 dtype 分支 |
| 禁止项 | 不新增 950PR 私有接口或私有 tiling 结构；不依赖隐式类型转换 |

同步约束：二层归约须等第一层所有核写完 workspace（由 kernel 边界提供，不得合并为单 kernel 内无 flag 配对的跨核轮询）；`magBuf` 与 `re/im` buffer 的 UB 复用需 `PipeBarrier` 保护；调试构建记录 block id 与部分结果，生产构建关闭 device print。

#### 3.2.3 测试工程与用例设计

按任务书 §3.5 使用 ops-blas 仓 `test/` 的 CSV 驱动框架，参照同族 `test/isamin/` 新建：`icamin_param.h`（CSV 列定义，对齐 `isamin_param.h`：`case_name, description, n, incx, x` + 期望状态码与随机种子；`x` 列写 `NULLPTR` 表示空指针负向用例）、`icamin_golden.h`（cblas 风格 CPU 循环，**不用** Netlib icamax 改符号）、`icamin_npu_wrapper.h`（调用 `aclblasIcamin` 并按 `(n, incx)` 关联 golden）、`arch35/icamin_test.cpp`（GTest 加载 CSV，整数索引 `EXPECT_EQ` 精确比对，性能用例单独统计 us）、`arch35/icamin_test.csv`。

用例规模与分类（任务包 `test_cases/icamin_test.csv`，1200 条，`gen_csv.py` 固定种子生成；按该 CSV 复算：`incx` 分布 1:394 / 2:165 / 3:109 / 0:131 / -1:142 / -2:147 / -3:112，`n` 覆盖 -8 ~ 4194304，期望 `SUCCESS` 1195 条、`INVALID_VALUE` 5 条，x 填充模式含随机/全零/交替/极端/Inf/NaN/NULLPTR 七类）：

| 类别 | 前缀 | 条数 | 要点 |
| --- | --- | --- | --- |
| L0 基础 | TC_L0 | 6 | n=1/8 × incx=1/2；全零填充的 tie 语义（期望最小索引 1） |
| L1 尺寸 | TC_SQ | 38 | 1 → 1048576，含质数、2 的幂 ±1、非对齐值 |
| L2 步长 | TC_INC | 17 | 正步长 1/2/3 正常计算；0 与负步长（-1/-2/-3）quick return |
| L5 填充 | TC_FL | 12 | 均匀随机 / 全零 / 正负交替 / 极端值 / Inf / NaN × 2 尺寸 |
| L6 边界 | TC_ED | 12 | n=0 quick return（×5 步长）、负 n（×3）、x 空指针（×2，期望 `INVALID_VALUE`）、quick return 优先于指针检查（×2，期望 `SUCCESS`） |
| EX 扩展 | TC_EX | 915 | 尺寸 × 步长 × 填充的确定性采样（精度主力） |
| PF 性能 | TC_PF | 200 | 含与三档标杆精确匹配的 n=1048576/2097152/4194304 且 incx=1；一律连续访存 |

覆盖要求（任务书 §3.5）：小 shape 基础、shape 扫描、填充模式、对齐偏移、边界与负向（零维/空指针/非法步长/负维度）、规格允许的 INF/NAN，以及性能与内存用例；任务包未覆盖者由本设计补齐。自验入口：

```bash
python verify_accuracy.py  --repo /path/to/ops-blas --soc ascend950 --csv ./icamin_test.csv
python verify_performance.py --repo /path/to/ops-blas --soc ascend950   # NPU 平均耗时 ≤ 标杆/倍率
```

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Ascend 950PR（arch35 / DAV_3510） | √ |

接口声明在 `include/cann_ops_blas.h` 由各产品线共用；950PR 专属实现细节留在 `blas/iamin/arch35/`，不外泄为平行 API。

## 算子约束限制

| 约束项 | 内容 |
| --- | --- |
| 参数合法性 | n ≥ 0（n < 0 → `INVALID_VALUE`）；n = 0 或 incx < 1 走 quick return（result 置 0，返回成功）；handle/result 不可为空；n > 0 且 incx ≥ 1 时 x 不可为空 |
| 非连续 Tensor | 不额外要求：向量步长由 incx 表达；负步长为 quick return、不做反向遍历；无 leading dimension padding 场景 |
| 广播 / dynamic shape | 不涉及广播（单向量归约到标量索引）；不要求 dynamic shape，n 为运行时入参 |
| 原地与视图 / 确定性 | 不涉及原地更新，result 为独立输出标量；确定性不要求（bit-exact 判定不依赖归约顺序） |
| 空 Tensor / 0 维 | n = 0 为合法 quick return（result 写 0）；n < 0 → `INVALID_VALUE` |
| 异步执行 | 依赖 `aclblasSetStream` 绑定 stream；读回 Device 结果前须同步 stream |
| 越界责任 | 物理长度 `1+(n-1)·\|incx\|` 的缓冲区大小由调用方保证；Host 不读 Device 内存做越界校验（与 cuBLAS / 仓内 isamin 一致，越界按异步错误协议） |
| 索引范围 | n ≤ 2^24（任务包内存预算的设计选择，非硬件上限）；索引用 INT32 承载 |

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度 | 输出 INT32 索引与 golden **逐位精确相等**（`EXPECT_EQ`），任一用例不等即失败；quick return 用例判定 `result == 0` | 任务书 §3.2；[生态算子开源精度标准](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md) |
| 分量档阈值（参考） | COMPLEX64 分量按 FLOAT32 档：rtol 2^-10 (9.77e-4)、atol 2^-16 (1.53e-5)、required_matched_ratio 0.99、max_abs_error_limit 1e-2 或 32×ULP；索引输出时退化为精确一致判定（单值用例通过即 ratio = 1） | 任务书 §3.2 表格 |
| 性能 | COMPLEX64 输入的平均单次耗时（Avg time，us）不高于标杆；采集前 warmup 且有效采样 > 50 次 | 任务书 §3.3 |
| 内存 | 不涉及 | 任务书 §3.4 |
| 交付 | 自测报告含精度对比结果与截图、性能数据与截图、内存占用数据；测试 README 保证可复现 | 任务书 §4 |

性能标杆（来源：任务书 §3.3，测试设备 Ascend 950PR）：

| case | n | incx | 标杆耗时（Avg time，us） |
| --- | ---: | ---: | ---: |
| 1 | 1048576 | 1 | 24.77 |
| 2 | 2097152 | 1 | 24.59 |
| 3 | 4194304 | 1 | 29.69 |

判定式：`NPU_avg_us(case) ≤ 标杆_us(case)`，三档全部满足方为达标。三档标杆随 n 增长仅从 24.77 us 增至 29.69 us，说明该归约由访存与启动开销主导而非算力主导，故实现重点是连续访存的向量化主路径、tile 粒度与尾部（drain/回写）开销，而非复杂分块算法。

测试目录另提供 `gpu_baseline.csv`（200 条性能/内存基线，前 3 条与上表三档典型 case 对应）；按该目录 README，`gpu_ms` 列为基线占位、待基线测试后回填，判定口径为「NPU 平均耗时 ≤ 标杆耗时 / 倍率」，回填前相关内容标 NO_REF 仅采集 NPU 耗时。**正式验收标杆以上表任务书三档数值为准。**

## 兼容性分析

| 维度 | 分析 |
| --- | --- |
| 公开接口 | 仅在 `include/cann_ops_blas.h` 新增 `aclblasIcamin` 声明，不修改既有声明；签名与 `cublasIcamin` 逐参数对齐（handle 及参数顺序一一对应，无需映射说明） |
| 多产品线共用 | 接口为通用 BLAS 形态，禁止定义 950PR 私有平行接口；950PR 实现细节收敛在 `blas/iamin/arch35/` |
| 同族隔离 | 与 `aclblasIsamin` 同目录但 symbol 与 tiling 结构独立；复用归约框架不改变 isamin 行为；新增 `test/iamin/icamin/` 不改既有 isamin 用例 |
| 行为差异 | `n < 0` 返回 `INVALID_VALUE` 与 cuBLAS「n ≤ 0 置 0」存在差异，按仓内 isamin README 约束与 arch35 实现执行（测试用例目录标注为事实表开放问题 Q3）；NaN 口径见 Q7，均以任务事实表最终结论为准 |
| 版本依赖 | CANN 9.1.0；实现基于合入时的 ops-blas 主干，避免与并行 PR 冲突 |
