# aclblasIcamin 算子设计文档（Ascend 950PR）
| 项目 | 内容 |
|------|------|
| 任务名称 | 9月社区任务-aclblasIcamin算子开发（950） |
| 参与者账号 | `vanerin` |
| 目标硬件 | Ascend 950PR（arch35 / DAV_3510） |
| CANN 版本 | CANN 9.1.0 |
| 交付代码仓 | [cann/ops-blas](https://gitcode.com/cann/ops-blas) |
| 设计提交路径 | `04_tasks/01_community-task-2026/tasklist/08-47-aclblasIcamin/vanerin/docs/design.md` |
| 对标接口 | cuBLAS `cublasIcamin` |
| 文档版本 | v1.0 |
| 验收环境 | Ascend 950PR（PCIe 档，`GetCoreNumAiv() = 56`）/ CANN 9.1.0 / Ubuntu 22.04 / gcc 11.4 |

---
## AI 申明

### 当前文档是否有 AI 参与:

- [ ] 否
- [x] 是

__1. AI Agent 平台: DeepSeek Harness
__2. AI 模型: DeepSeek V4.1
__3. Prompt上下文: 依据《9月社区任务-aclblasIcamin算子开发（950）》任务书、CANN 社区任务设计文档模板（design_template.md）与 Ascend 950PR 真机实测记录，整理算子设计、实现路径与自测数据；文中全部设计与实测数据均对应真实实现与真机运行结果，未虚构数据或截图。

---

# 需求背景（required）
## 需求来源
本设计对应 2026 年 9 月社区任务《9月社区任务-aclblasIcamin算子开发（950）》，任务书文件为 `aclblasIcamin_Atlas950PR_task_doc.md`。任务要求在 Ascend 950PR 上使用 Ascend C **kernel 直调**（direct launch）方式，基于 ops-blas 开源仓工程框架，实现句柄式 BLAS 接口 `aclblasIcamin`：在 COMPLEX64 向量中查找**1-范数模**最小元素的 **1-based 索引**，模相同时取最小索引。验收通过后合入 [cann/ops-blas](https://gitcode.com/cann/ops-blas) 的 `blas/iamin/arch35/`，测试代码合入 `test/iamin/icamin/arch35/`。

设计文档按 cann-competitions 官方模板 [design_template.md](https://gitcode.com/cann/cann-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md) 撰写，并在 cann-competitions 仓以 PR 形式提交评审。公开接口声明放入 ops-blas 的 `include/cann_ops_blas.h`，**禁止定义 950PR 私有平行接口**。
## 背景介绍
### 算子要做什么
| 项目 | 内容 |
|------|------|
| 输入 `x` | COMPLEX64 向量，逻辑一维 [n]；由 `aclblasComplex`（`struct { float real; float imag; }`，实部/虚部各 float32，**交错存储**）承载，即 n 个复数对应 2n 个 float32 |
| 输入 `n` | 复数元素个数，Host 标量，int，n ≥ 0 |
| 输入 `incx` | 相邻复数元素之间的步长（以**复数元素**为单位），Host 标量，int |
| 输出 `result` | Device 侧单个 INT32 标量，取值 0（quick return）或 [1, n] |

数学表达式：`result = argmin_{i=1..n} ( |Re(x_k)| + |Im(x_k)| )`，其中 `k = 1 + (i-1) * incx`。

- 结果为 **1-based** 索引（Fortran / BLAS 惯例）。
- 「模」为 **1-范数模**（又称曼哈顿模、BLAS 的 SCABS1 语义），即 `|Re| + |Im|`，**不是**欧几里得模 `sqrt(Re² + Im²)`。两者数值不等价，不可互相替换。
- 比较语义为**严格小于才更新**（`abs1 < best`），因此模相等的多个元素中返回**最小索引**。
- 标准参考 BLAS（Netlib）**无 icamin 例程**；语义细节以 cuBLAS `cublasIcamin` 官方文档为最高优先级来源，精度真值（golden）由测试工程内的 CPU 参考循环生成。
### 与同族算子的关系
| 算子 | 输入 | 归约 | 模定义 | 仓内参照 |
|------|------|------|--------|----------|
| `aclblasIsamin` | FP32 实数 | ReduceMin | `\|x\|` | `blas/iamin/arch35/`（Host 校验链、分核、二段归约、SIMT 通路的**框架复用对象**） |
| `aclblasIcamax` | COMPLEX64 | ReduceMax | `\|Re\|+\|Im\|` | `blas/iamax/arch35/`（复数分量组合与索引路径的参照） |
| **`aclblasIcamin`**（本任务） | COMPLEX64 | **ReduceMin** | `\|Re\|+\|Im\|` | `blas/iamin/arch35/icamin_*`（本任务新增） |

**核心映射**：在 `isamin` 的「搬入 → 求模 → ReduceMin」流水线上，把实数 `Abs` 替换为 COMPLEX64 的 `Abs ×2 + Add`，并新增**解交织**（把交错的实部/虚部分离到两个缓冲）与**索引恢复**（本平台归约接口的索引输出不可靠，见「详细设计 → 算子实现」）；Host 校验链、分核方式、workspace 二段归约结构、`incx≠1` 的 SIMT 通路均与 `isamin` 同构。
### 术语定义
| 术语 | 定义 |
|------|------|
| UB / GM / AIV | UB（Unified Buffer）为片上向量缓存（本平台 248 KiB）；GM（Global Memory）为设备全局内存（HBM）；AIV（AI Vector core）为向量计算核，核数由运行时 `GetAivCoreCount()` 获取（验收机为 56） |
| tile / tiling | tile 为核内一次循环处理的**连续元素段**，长度由 `tileSize`（单 tile 复数元素数）界定；tiling 为 Host 侧依据 n、核数、UB 预算计算核间与核内切分参数的步骤 |
| 跨流水同步事件 | 在**不同硬件流水**之间建立显式先后依赖的 `SetFlag`/`WaitFlag<HardEvent::XXX>` 事件；本算子涉及 MTE2（GM→UB 搬运）、V（向量计算）、S（标量）、MTE3（UB→GM 搬运）四条流水 |
| WS 槽位（workspace slot） | 跨核归约用的固定长度记录，每个核写一条到 GM workspace；本算子每条 **32 B（8 个 float）**：`[0]` 为该核最小模值、`[1]` 为该值的全局索引位模式、`[2..8)` 为 0 补齐 |
| 加性保护带（guard band） | 在归约源缓冲尾部额外分配、并恒定填充 `+Inf` 的 **64 车道**区域，用于吸收 `ReduceMin` 的整块/整 repeat 过读 |
| SIMT / DCache 预留 | SIMT 为单指令多线程编程模型，仅用于 `incx ≠ 1` 的等步长访存通路；使用 `__simt_vf__`/`asc_vf_call` 的 kernel 需为 DCache 预留 ≥ 32 KiB，可用 UB = 256 KiB − 8 KiB（编译器预留）− 32 KiB（DCache）= 216 KiB |
### Quick return 与异常语义
| 条件 | 行为 |
|------|------|
| `handle == nullptr` | 返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| `n < 0` | 返回 `ACLBLAS_STATUS_INVALID_VALUE`（该分支返回前不写 `result`） |
| `n = 0` 或 `incx < 1`（含 0 与负步长） | **不触发 kernel**，写 `result = 0`，返回 `ACLBLAS_STATUS_SUCCESS`；该 quick return **优先于指针检查**（此时 `x` 可为 nullptr）；负步长**不反向遍历** |
| `n > 0` 且 `incx ≥ 1` 且 `x == nullptr` | 返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| `n > 0` 且 `incx ≥ 1` 且 `result == nullptr` | 返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| 输入含 NaN（部分或全部） | **跳过 NaN 元素**（不参与比较）；首元素为 NaN 时基准置 `FLT_MAX`；全 NaN 向量返回 1 |
| 全部元素的模均为 `+Inf`（例如全部为 `(FLT_MAX, FLT_MAX)`） | 严格小于恒假 ⇒ tie 取最小索引，返回 1 |
| 输入同时含 NaN 与 `+Inf` 且**没有任何有限值** | 返回 1（登记为已知限制，见文末） |

---
# 需求分析（required）
## 需求描述
在 Ascend 950PR 上完成调用链 `aclblasIcamin(handle, n, x, incx, result)` → Host 参数校验与 tiling → Ascend C kernel（目录 `arch35` / `--soc=ascend950`）→ Device 侧写回 INT32 标量（最小模元素的 1-based 索引，或 quick return 的 0）。其中 **quick return** 指无需启动 kernel 即可确定结果的提前返回路径。

- **精度**：整数索引 **bit-exact**（逐位精确一致，即 `actual == golden` 的零容差判定）；输入分量按任务书 §3.2 的 FLOAT32 档阈值表作背景说明，不作为输出判定口径。
- **性能**：NPU 平均单次耗时 ≤ 标杆耗时，标杆耗时 = `gpu_baseline.csv::gpu_ms × 1000 / 0.4`；三个必达标 case 取任务书 §3.3 的值。
- **异步语义**：依赖 `aclblasSetStream` 绑定的 stream；调用方读回 Device 结果前须自行同步 stream。
## 需求拆解
| 编号 | 子项 | 验收要点 |
|------|------|----------|
| R1 | 公开 API | `include/cann_ops_blas.h` 新增 `aclblasIcamin` 声明，签名与 `cublasIcamin` 逐参数对齐；不新增 950PR 私有平行接口 |
| R2 | 功能正确 | 1254 条 CSV 用例（精度 1000 + 性能 200 + 补测 54）逐条与 CPU golden 比对一致；NaN / Inf / 溢出 / 并列 / quick return / 负 n 语义全部对齐 |
| R3 | 复用 isamin 框架 | 实现落 `blas/iamin/arch35/`，Host 校验链、分核、workspace 二段归约、SIMT 通路与 `isamin` 同构 |
| R4 | 复数模 | 仅允许 `Abs + Add` 得到 `\|Re\|+\|Im\|`；**禁止**使用 `blasComplexAbs`（欧几里得模） |
| R5 | 测试工程 | `test/iamin/icamin/arch35/` 下 CSV 驱动 GTest + 白盒 + 偏移 + 特殊值 + 性能采集；golden 为工程内 CPU 循环 |
| R6 | 性能 | `TC_PF_*` 200 条采集；三个必达标 case（n=1048576 / 2097152 / 4194304，incx=1）门槛 24.77 / 24.59 / 29.69 µs |
| R7 | 文档 | `blas/iamin/README.md` 补充 `aclblasIcamin` 小节；产品支持表标注 Ascend 950PR：支持 |
| R8 | 边界与特殊值 | 小 shape、shape 扫描、填充模式、对齐偏移、零维、空指针、非法步长、负维度、INF/NAN、性能/内存用例全覆盖 |
## 输入输出规格
参数规格照任务书 §2.4：

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | dtype | 数据排布 | 维度(shape) | 值域范围 | 异常行为 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| handle | 输入 | ops-blas 库上下文句柄，携带 stream，Host 内存 | scalar | - | - | - | 指向已创建的有效句柄 | `nullptr` → `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| n | 输入 | 向量 x 的复数元素个数，Host 内存 | scalar | int | - | - | n ≥ 0 | `n < 0` → `ACLBLAS_STATUS_INVALID_VALUE`；`n = 0` 为合法 quick return（不触发 kernel，写 `result = 0`，返回 `ACLBLAS_STATUS_SUCCESS`） |
| x | 输入 | 复数向量，Device 内存，只读 | tensor | COMPLEX64 | ND | 逻辑一维 [n]，物理长度 `1+(n-1)*\|incx\|` | 实部/虚部取值于 FLOAT32 全集 | `n > 0` 且 `incx ≥ 1` 时 `nullptr` → `ACLBLAS_STATUS_INVALID_VALUE` |
| incx | 输入 | x 中相邻元素之间的步长，Host 内存 | scalar | int | - | - | `incx ≥ 1` 正常计算 | `incx < 1`（含 0 与负步长）为合法 quick return（不触发 kernel，写 `result = 0`，返回 `ACLBLAS_STATUS_SUCCESS`） |
| result | 输出 | 最小模元素的 1-based 索引（整数标量），Device 内存 | scalar | INT32 | - | 单值标量 | 0（quick return）或 [1, n] | `nullptr` → `ACLBLAS_STATUS_INVALID_VALUE` |

**返回值**：`aclblasStatus_t`，状态码语义与 ops-blas 仓 `include/cann_ops_blas_common.h` 定义一致（`ACLBLAS_STATUS_SUCCESS` / `ACLBLAS_STATUS_INVALID_VALUE` / `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` 等）。

接口原型：

```cpp
aclblasStatus_t aclblasIcamin(
    aclblasHandle_t handle,
    int n,
    const aclblasComplex* x,
    int incx,
    int* result);
```
## 算子实现约束
约束照任务书 §2.5：

| 约束项 | 内容 |
| --- | --- |
| 参数合法性 | n ≥ 0（`n < 0` 报 `ACLBLAS_STATUS_INVALID_VALUE`）；`n = 0` 或 `incx < 1` 走 quick return（`result` 置 0，返回成功）；`x`/`result`/`handle` 不可为 `nullptr` |
| 非连续 Tensor 支持 | 不要求（向量步长由 `incx` 表达；负步长为 quick return，不做反向遍历；无额外 leading dimension padding 场景） |
| broadcast 规则 | 不涉及（本算子为单向量归约到标量索引） |
| dynamic shape 要求 | 不要求（n 为运行时入参） |
| 原地与视图语义 | 不涉及原地更新；`result` 为独立输出标量 |
| 确定性计算要求 | 不要求（整数索引结果在语义上确定，bit-exact 判定不依赖归约顺序） |
| 空 Tensor 与 0 维处理 | `n = 0` 为合法 quick return，返回成功且 `result` 写 0；`n < 0` 返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| 异步执行 | 依赖 `aclblasSetStream` 绑定 stream；读回 Device 结果前须同步 stream |

---
# 详细设计（required）
## 算子分析
### 数学公式与 golden 语义
**计算范式**：纯 Vector 归约（无矩阵乘，不含 Cube），SIMD 载体为 Ascend 950PR 的 **RegBase 向量路径**。计算流程为「逐元素取 `|Re|`、`|Im|`（绝对值无舍入）→ 相加得 1-范数模 → 取最小值并定位其车道号 → 换算为全局 1-based 索引」。

**golden 语义**（测试工程内 CPU 参考实现，与 kernel 使用**同一条** FP32 表达式）：

```cpp
// 首元素即基准；首元素为 NaN 时基准置 FLT_MAX
float absVal = std::fabs(x[0].real) + std::fabs(x[0].imag);   // 同一条 FP32 表达式
float minAbs = (absVal != absVal) ? FLT_MAX : absVal;
int   minIdx = 1;                                             // 1-based
for (int i = 1; i < n; ++i) {                                 // i 为 0-based 偏移
    const aclblasComplex* e = &x[static_cast<size_t>(i) * incx];
    absVal = std::fabs(e->real) + std::fabs(e->imag);
    if (!(absVal != absVal) && absVal < minAbs) {              // 非 NaN 且严格小于
        minAbs = absVal;
        minIdx = i + 1;
    }
}
return minIdx;                                                // 严格小于保证 tie 取最小索引
```

| 场景 | golden / kernel 行为 |
|------|----------------------|
| 模的定义 | `\|Re\| + \|Im\|`（1-范数模），非欧几里得模 |
| 更新条件 | 非 NaN 且**严格小于**当前最优值才更新；因此并列时保留最早（最小）索引 |
| NaN 元素 | 跳过（不参与比较）；首元素为 NaN 时基准置 `FLT_MAX` |
| 全 NaN 向量 | 所有比较为假 ⇒ 返回 1 |
| 模溢出为 `+Inf`（如 `(FLT_MAX, FLT_MAX)`） | 溢出即 `+Inf` 参与比较（golden 与 kernel 同表达式，不引入第二套真值）；全为 `+Inf` 时返回 1 |
| `n = 0` 或 `incx < 1` | golden 与实现一致：写 `result = 0` 返回 `SUCCESS`（优先于指针检查） |
| 可复算算例 | `x=[(1,2),(0,0),(3,4)]`、n=3、incx=1 → 模 3.0 / 0.0 / 7.0，严格小于只接受第 2 个 ⇒ **result = 2** |
### 支持数据类型与芯片
| 项目 | 内容 |
|------|------|
| 输入 | COMPLEX64（`aclblasComplex`，实部/虚部各 float32，交错存储，8 B/元素），分量按 FLOAT32 档 |
| 输出 | INT32 单值标量 |
| 目标芯片 | Ascend 950PR |
| NpuArch / 目录 / 编译参数 | `DAV_3510` / `arch35` / `--soc=ascend950` |
| CANN 版本 | CANN 9.1.0 |
| 代码架构 | SIMD（RegBase 向量路径，不含 Cube）；`incx ≠ 1` 分支使用 SIMT |
| dtype 分派 | 单一 dtype（输入 COMPLEX64、输出 INT32），**无** float16/bfloat16 变体、无 dtype 分派分支 |
| 资源关键量 | UB 248 KiB（以 `UB_SIZE` 常量作**编译期**预算推导，见「UB 预算与内存占用」）；**核数不硬编码**，运行期由 `GetAivCoreCount()` 获取；运行期实际校验的是 **workspace 尺寸**（`GetEffectiveWorkspaceSize()` 与推导出的 `requiredBytes` 比较） |
### 支持形状
- 逻辑一维向量 [n]；物理长度为 `1+(n-1)*|incx|` 个复数元素（`2*(1+(n-1)*|incx|)` 个 float32），排布格式 ND。
- `n ∈ [0, 2^24]`（`2^24` 为任务包给出的测试内存预算上限，属设计选择而非硬件限制）；性能用例覆盖至 `4194304 = 2^22`。
- `incx = 1`：连续访存，走向量主路径（三个性能 case 均为 `incx = 1`）。
- `incx ≥ 2`：等步长访存，走 SIMT 通路；对齐要求为 float32 标量访存的 4 B 对齐。
- `incx < 1`：quick return。
- 不涉及 broadcast、dynamic shape、原地与视图语义；不要求确定性归约顺序（整数索引结果与归约顺序无关）。

---
## 算子实现
### 实现方案总览
**三条执行路径**与 kernel 分发：

```text
Host（icamin_host.cpp）
  ValidateIcaminParams → CalcIcaminTiling → workspace 尺寸推导与校验 → icamin_kernel_do

Kernel 分发（icamin_kernel_do）
  ├─ incx == 1 且 numBlocks == 1 且 lastCoreN <= tileSize
  │       → icamin_small_kernel            （单核，核内直接写 result，无 workspace 往返）
  ├─ incx == 1（其余规模）
  │       → icamin_aiv_kernel（向量主通路）+ icamin_reduce_kernel（跨核收尾与 1-based 写回）
  └─ incx != 1
          → icamin_simt_kernel（SIMT 等步长通路）+ icamin_reduce_kernel（同一收尾 kernel）
```

共 **4 个 kernel 入口**：`icamin_aiv_kernel`、`icamin_small_kernel`、`icamin_simt_kernel`、`icamin_reduce_kernel`；分发点为 `icamin_kernel_do`（`icamin_kernel.cpp:787` 起）。`icamin_small_kernel` 与主通路**复用同一个基类**（同 8 个 UB buffer），仅输出去向不同。

**目录规划**：

```text
ops-blas/
├── include/cann_ops_blas.h        # + aclblasIcamin 声明（197 行）
├── blas/iamin/
│   ├── README.md                  # 补充 aclblasIcamin 小节
│   └── arch35/                    # icamin_tiling_data.h（tiling 结构体与共享常量）
│                                  # icamin_kernel.h（icamin_kernel_do 声明）
│                                  # icamin_kernel.cpp（4 个 kernel 入口）
│                                  # icamin_host.cpp（校验链、tiling、workspace、下发）
└── test/iamin/icamin/
    ├── CMakeLists.txt             # 测试目标定义
    ├── icamin_param.h             # CSV 参数类型与 ReadMap 键
    ├── icamin_golden.h            # CPU golden + 暴力第二实现（交叉核对）
    ├── icamin_nan_fill.h          # NaN 位置可控填充
    ├── icamin_whitebox_tiling.h   # tiling 镜像 + static_assert
    └── arch35/                    # icamin_npu_wrapper.h（设备内存/句柄 RAII 与用例派发）
                                   # icamin_test.cpp（CSV 驱动 + 白盒 + 偏移 + 性能采集）
                                   # icamin_test.csv（1255 行 = 表头 1 + 数据 1254）
```
### Host 侧设计
#### 参数校验链
校验顺序（与仓内同族实数接口 `aclblasIsamin` 的语义一致，仅入参类型由 `const float*` 改为 `const aclblasComplex*`）：

| 顺序 | 判定 | 行为 |
|---|---|---|
| 1 | `handle == nullptr` | 返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| 2 | `n < 0` | 返回 `ACLBLAS_STATUS_INVALID_VALUE`（**不写** `result`） |
| 3 | `n == 0 \|\| incx < 1` | quick return：`result != nullptr` 时以 `aclrtMemcpyAsync(..., ACL_MEMCPY_HOST_TO_DEVICE, handle->stream)` 写 4 B 零值；返回 `ACLBLAS_STATUS_SUCCESS`。**排在指针检查之前**；该 memcpy 失败时 `OP_LOGE` 并返回 `EXECUTION_FAILED` |
| 4 | `x == nullptr` | 返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| 5 | `result == nullptr` | 返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| 6 | kernel 下发前复查 | `n <= 0 \|\| incx < 1` 直接返回，不调用 `icamin_kernel_do` |

> **`result` 是设备输出指针，Host 侧禁止解引用**；quick return 的零值必须经 `aclrtMemcpyAsync`（Host→Device）写入。`n < 0` 分支返回前不写 `result`，因此负向用例不应断言 `result` 的内容。
#### 分核与 tiling 公式
```text
CalcIcaminTiling(n, numBlocks, incx) -> IcaminTilingData:
  totalN = n；useCoreNum = numBlocks；incx = incx
  perCoreN  = totalN / numBlocks                     // 前 useCoreNum-1 个核的元素数
  lastCoreN = perCoreN + (totalN % numBlocks)        // 末核补余数
  tileSize  = 8192（COMPLEX64_MAX_TILE_SIZE）；若 perCoreN < 8192 则改为
              max(CeilAlign(perCoreN, 64), 64)       // 下限 64（COMPLEX64_TILE_ALIGN），避免得 0
  nthreads  = (incx != 1) ? min(CeilAlign(CeilDiv(perCoreN, 128), 128), 2048) : 0
```

| 项目 | 取值 / 规则 |
|------|-------------|
| 切分维度 | 逻辑一维 [n] 按**元素**连续均分为 `numBlocks` 段（非跨步切分）；核内再按 `tileSize` 分 tile |
| 核数 | `numBlocks = min(n, GetAivCoreCount())`；为 0 时返回 `INTERNAL_ERROR`。验收机 `GetCoreNumAiv() = 56` |
| 核内迭代 | `repeatTimes = computeNum / tileSize` 个满 tile + 1 个余量 tile（`remainNum = computeNum % tileSize`） |
| 每核元素数（三性能 case；`perCoreN` 为整数除法**向下取整**，末核 `lastCoreN` 补余数） | 56 核：`18724 / 37449 / 74898`（末核 `18756 / 37457 / 74914`）；64 核：`16384 / 32768 / 65536`（整除，无余数） |
| 每核 tile 迭代数（tileSize=8192） | 56 核：`3 / 5 / 10`；64 核：`2 / 4 / 8` |
| 每核搬入总量（复数 8 B/元素） | 56 核：`146.3 / 292.6 / 585.1 KiB`；64 核：`128 / 256 / 512 KiB` |
| tileSize 收缩必须保持的性质 | `tileSize % 64 == 0`（`ReduceMin` 的 repeat 粒度）、`2*tileSize` 为偶数（`DeInterleave` 约束） |
| 小规模快速路径阈值 | `numBlocks == 1 && lastCoreN <= tileSize` |
| `nthreads`（仅 `incx ≠ 1`） | 从 128 起以 128 为步长向上取整，上限 2048；`perCoreN < 128` 时由实现夹到 128 |
#### workspace 尺寸推导与校验
```text
wsSlots       = max(ICAMIN_WS_SLOT_FLOATS = 8, next_pow2(useCoreNum))
totalFloats   = wsSlots * ICAMIN_WS_SLOT_FLOATS
alignedFloats = CeilAlign(totalFloats, ALIGN_FLOATS = 64)
requiredBytes = alignedFloats * 4
若 requiredBytes > GetEffectiveWorkspaceSize(handle) → ACLBLAS_STATUS_EXECUTION_FAILED
```

| 项目 | 口径 |
|------|------|
| 每核槽位 | **32 B（`ICAMIN_WS_SLOT_FLOATS = 8` 个 float）**：`[0]` = 该核最小模值；`[1]` = 其全局 0 基索引的 float32 位模式；`[2..8)` = 置 0 补齐 |
| 槽位数 | `max(8, next_pow2(useCoreNum))`；Host 与 kernel **各自用同一幂次规则推导**，tiling 结构体不新增槽位字段 |
| 三性能 case（56 核） | `wsSlots = 64` → `totalFloats = 512` → `alignedFloats = 512` → **2048 B** |
| 为何必须是 32 B 槽位 | `DataCopyPad` 的 `blockLen + leftPadding + rightPadding` 不满足 32 B 对齐时，框架会填充 dummy 数据凑齐 32 B，**落点由实现决定**；旧的「每核 2 float（8 B）、偏移 `blockIdx*8`」写法必然落入该路径（仅 `blockIdx % 4 == 0` 时对齐），真机表现为奇数核记录落点位移。改为 32 B 槽位 + 幂次槽位数后，UB 端与 GM 端（`blockIdx * 32 B`）双双满足 32 B 对齐，普通 `DataCopy` 即可 |
### Kernel 侧设计
#### 连续主路径（`incx == 1`）的数据流与 API 映射
单核视角、单 tile 的数据流：

```text
x(GM, 交错 re/im float32 对)
  --DataCopy（按 float32 元素计数）--> inLocal[0 .. 2*tileSize)
  --DeInterleave(inLocal, srcCount = 2*tileSize)-->  reLocal（偶数位 = 实部）/ imLocal（奇数位 = 虚部）
  --Abs(reLocal); Abs(imLocal)-->                    // 原地：|Re|、|Im|
  --Add(magLocal, reLocal, imLocal)-->               // |Re|+|Im|，FP32 加法语义（含溢出→+Inf）
  --条件化 NaN 过滤（仅当归约值为 NaN 时执行）--> Compare(mag, mag, EQ) 得位压缩掩码（1 = 非 NaN）
                                                  Select(mag, mask, mag, +Inf) 把 NaN 车道改为 +Inf
  --ReduceMin(outLocal, magLocal, workLocal, count, calIndex=false)-->   // 只取值、不取索引
  --Adds(selBuf_, magLocal, 0.0f)-->                 // 被接受 tile 的模值数组 + 保护带暂存
  ...（tile 循环结束）--> FinalizeIndex()            // 每核唯一一次索引恢复
  --SetValue + SetFlag/WaitFlag<S_MTE3> + DataCopy(UB→GM)--> wsGM[blockIdx*8 .. +8)  // 32 B 槽位
```

| 步骤 | Ascend C API | 说明 |
|------|--------------|------|
| 搬入 | `DataCopy(dst, src, count)` | GM→UB 连续搬入交错复数；按 float32 元素计数，满 tile 为 `2*tileSize` 个 float |
| 尾块搬入 | `DataCopyPad(dst, src, DataCopyExtParams, DataCopyPadExtParams<float>)` | 右侧填充 `+Inf`（**值语义**，`__builtin_inff()`）；`leftPadding ≡ 0`；`padding` 单位为**元素个数** |
| 解交织 | `DeInterleave(dst0, dst1, src, srcCount)` | `srcCount` = **输入元素个数**（两个输出各得一半），满 tile 传 `2*tileSize`；必须为偶数 |
| 求模 | `Abs(re, re, count)` ×2 + `Add(mag, re, im, count)` | `Abs` 为精确取绝对值（无舍入），`Add` 为 IEEE 近邻舍入加法；**禁止**使用 `blasComplexAbs`（欧几里得模） |
| NaN 过滤 | `Compare(..., CMPMODE::EQ)` + `Select(..., VSEL_TENSOR_SCALAR_MODE)` | 位压缩掩码（1 bit/元素）；掩码张量须声明为 `LocalTensor<uint8_t>` |
| 归约 | `ReduceMin(dst, src, sharedTmpBuffer, count, calIndex=false)` | 只取值，不使用其索引输出；`dst` 的存储顺序为「最小值, 最小值索引」 |
| 暂存 | `Adds(selBuf_, magLocal, 0.0f, tileSize + 64)` | 整段复制被接受 tile 的模值数组，顺带带过 `+Inf` 保护带 |
| 索引恢复 | `CreateVecIndex` + `CompareScalar` + `Select` + `ReduceMin` | 以 0 基车道号栅格反查首个等值车道 |
| 结果写回 | `Duplicate<int32_t>` + `DataCopyPad` | 写入 1 个 int32；`V` → `MTE3` 之间插 `V_MTE3` |
| 跨核发布 | `SetValue` + `SetFlag/WaitFlag<S_MTE3>` + `DataCopy` | 见「跨流水同步与平台约束」 |
| SIMT 通路 | `__simt_vf__` + `asc_vf_call` + `asc_syncthreads()` | 仅 `incx ≠ 1` |
| 核编号 / 核数 | `GetBlockIdx()` / `GetAivCoreCount()` | 运行期获取，不硬编码 |

**单条搬运指令的容量约束**：`DataCopy` 的 `blockLen` 为 `uint16_t`、单位字节，类型上限 65535 B；`tileSize = 8192` 的单 tile 搬入量为 `8192 × 8 = 65536 B`，**超出 1 字节**。实现取「保留 `tileSize = 8192`，把满 tile 的 16384 个 float 拆成 2 次 `DataCopy`」的方案，常量 `DATA_COPY_MAX_FLOAT32 = 8192`（每次 8192 float = 32768 B）；该拆分不改变 UB 预算、tile 迭代数与 `ReduceMin` 的 `count`。

**满 tile 与尾块**：满 tile 走 `DataCopy`；尾块（`remainNum = computeNum % tileSize`）走 `DataCopyPad` 右侧填充 `+Inf`。`ReduceMin` 的 `count` 传**向上 32 B 对齐后的实际车道数** `alignedRemain`（`srcLanes = 2*remainNum` → `alignedSrcLanes = CeilAlign(srcLanes, 8)` → `alignedRemain = alignedSrcLanes/2`），并用 `laneFound < bestValidElements_`（即 `remainNum`）判定索引是否落在真实元素上；不成立则该 tile 不参与比较。填充值**必须为 `+Inf`**：`|Re|+|Im|` 的最大值不可能超过 `+Inf`，故填充车道在 `ReduceMin` 中永不胜出；**禁止用 `FLT_MAX`**（最大有限值会被真实 `+Inf` 车道战胜）。
#### 索引恢复
**为何不使用归约接口的索引输出**：本平台 `ReduceMin(..., calIndex=true)` 的索引在**全并列** tile 上「有时返回 0、有时返回 64」，而文档要求返回第一个（最小）索引；其架构实现按整 repeat 推进，并列时的选择不可预测。因此 `calIndex` 恒为 `false`，索引改为「值归约 + 由值反查首车道」。

**实现（每核唯一一次）**：

```text
CreateVecIndex(rampLocal, 0.0f, count)                     // ramp[k] = k（0 基车道号栅格）
CompareScalar(maskLocal, bestLocal, bestVal_, EQ, count)    // 1 = 等于该 tile 的最小值
Select(selOut, maskLocal, rampLocal, LANE_NO_MATCH, ..., count)  // 命中取车道号，未命中取 FLT_MAX
PipeBarrier<PIPE_V>()
ReduceMin(outLocal, selOut, workLocal, bestLanes_, false)   // 取最小车道号 = 首个等值车道
V_S 同步 → laneFound
bestIdx_ = (laneFound < bestValidElements_) ? (bestTileOffset_ + (laneFound - bestLeftPadLanes_)) : 0
```

- tile 循环内**不做恢复**：只把「被接受 tile 的模值数组 + 几何（`bestTileOffset_` / `bestLanes_` / `bestValidElements_` / `bestLeftPadLanes_`）」用 `Adds` 暂存进 `selBuf_`；循环结束后由 `FinalizeIndex()` 执行唯一一次恢复。
- 恢复段中 `CreateVecIndex` / `CompareScalar` / `Select` 的 `count` 统一取 `tileSize + MAG_BUF_GUARD_FLOATS`（= 8256），以覆盖与 `ReduceMin` 相同的过读窗口；最终那次取值归约 `ReduceMin` 的 `count` 取 `bestLanes_`（被接受 tile 的实际车道数）。
- **正确性依据**：接受判据是严格更小，暂存发生在被接受的那一刻，因此暂存的那一份恒为「最早达到 `bestVal_` 的 tile」，其内首个等值车道即全局最小索引（并列取最小索引）；跨核再按「更小，或相等且索引更小」合并，全局同样取最小索引。
- **零新增 UB 缓冲**：暂存复用 `selBuf_`（原本只在恢复时瞬时使用），`Select` 输出落到 `magBuf_`（tile 循环结束后内容已无用）。
#### 跨核归约（`icamin_reduce_kernel`）
- 单核收尾，两个通路共用。`DataCopy` 把 `alignedFloats`（`WsSlotCount(useCoreNum)*8` 按 64 元素对齐）的 workspace 一次搬入 UB（`TBuf<TPosition::VECCALC>`，长度 `alignedFloats*4` B；56 核时为 512 float = 2048 B），随后插入 `SetFlag/WaitFlag<HardEvent::MTE2_S>`。
- 槽位是 8 float（32 B）间隔的稀疏记录，模值位于每槽 `[0]`，而 `ReduceMin` 只能归约**连续**区间，故此处保持**标量循环**，按 `i * ICAMIN_WS_SLOT_FLOATS` 步进，谓词为「非 NaN 且（更小，或相等且索引更小）」。
- `hasValue == false`（全 NaN 或无核有值）时 `bestIdx = 0`；最终 `result = static_cast<int32_t>(bestIdx) + 1`，经 `Duplicate<int32_t>` + `DataCopyPad` 写 1 个 int32 到 `result`（`+1` 即 1-based 索引的落地点）。
- 对齐要求：搬入 `alignedFloats*4` 字节需 32 B 对齐，`alignedFloats` 按 64 元素（256 B）对齐已满足；槽内起始地址 `blockIdx * 32 B` 亦为 32 B 倍数。
#### SIMT 非连续路径（`incx ≠ 1`）
- **不使用「等步长搬入 UB」**：连续搬运以 **32 B dataBlock 为搬运单元**，`DataCopy`/`DataCopyPad` 的 `srcStride` 只能在**块间**跳间隔，无法在 dataBlock 内部做 4 B 级分量选择；而 `incx` 是**复数元素步长**（相邻实部之间只隔 `8 B × incx`），需在 8 B 内取 4 B 的实部或虚部，小于一个搬运单元。故搬运指令族无法覆盖本算子的任意 `incx` 步长搬入。
- **采用 SIMT 通路**（与 `isamin` 的 `incx != 1` 做法同构）：`__simt_vf__` 内每个线程按 `i = threadIdx.x; i < calNum; i += blockDim.x` 遍历，直接读 `xGm[(start + i) * 2 * incx]`（实部）与 `+1`（虚部），各自取绝对值后相加，按「非 NaN 且（更小，或相等且索引更小）」维护 `(bestVal, bestIdx)`，再 `asc_syncthreads()` 后以 `ubPartialVals` / `ubPartialIdxs` 双数组做树形归约。
- **未采用 `asc_reduce_min`**：该接口在 950PR 上支持，但只归约**值**、不携带索引；沿用 isamin 的手写双数组树形归约可避免值/索引不同步。
- **对齐约束**：线程按 float32 标量访存，只需 4 B 对齐（`incx` 为偶数时两者均 8 B 对齐；`incx` 为奇数时虚部偏移为 4 B 对齐）；设备侧基址由 `aclrtMalloc` 类接口分配，天然 ≥ 256 B 对齐，故 4 B 对齐成立。真机以 `incx ∈ {1,2,3,5,8}` × 28 尺寸 × 6 填充 = 870 例独立 golden 对拍 **0 错**（含 `incx = 3` 的 4 B 对齐虚部），确认无需更强对齐。
- **该分支不参与性能判定**（任务书 §3.3 的三个必达标 case 均为 `incx = 1`）。
#### 小 n 路径（`icamin_small_kernel`）
- 触发条件：`numBlocks == 1 && lastCoreN <= tileSize`。
- 与主通路**复用同一个基类与同 8 个 UB buffer**，UB 预算与主通路完全相同（`24*tileSize + 1920`）；差别仅在输出去向：核内直接写 `result`（`bestIdx_ + 1` 经 `Duplicate<int32_t>` + `DataCopyPad` 写 1 个 int32），省去一次 kernel launch 与 workspace 往返。
### 跨流水同步与平台约束
#### 6 项跨流水同步事件
裸 `TBuf`（不经 `TQue`/`DeQue`）时，框架**不提供任何隐含依赖**：每一行的生产者与消费者属于不同硬件流水（或不同 kernel），必须显式插入同步；缺一项即表现为真机上的随机错值。本算子逐处核对后共落地 6 项：

| # | 生产者 → 消费者 | 事件 | 缺失时的真机表现 |
|---|---|---|---|
| ① | `DataCopy`/`DataCopyPad`(MTE2) → `DeInterleave`/`Abs`/`Add`(V) | `MTE2_V` | 逐核「半对半垃圾」（向量流水早于 DMA 完成即读 `inBuf_`，读到未落盘的旧内容） |
| ② | `Abs`/`Add`(V) → `ReduceMin`(V)：**同管线但整块语义**（`ReduceMin` 自带 mask register，按 8 lane 块 + 64 lane repeat 推进），不能依赖普通程序序 | `PipeBarrier<PIPE_V>` | 整块结果错误 |
| ③ | `ReduceMin`(V) → `GetValue`(S)（标量读归约结果） | `V_S` | 标量读到陈旧的归约结果 |
| ④ | `DataCopy`(MTE2) → `GetValue`(S)（收尾 kernel 搬入 workspace 后按标量读槽位） | `MTE2_S` | 标量读到陈旧 UB，输出位模式乱（如 `0x7F7FFFFF`、`0xFFFFFFFF`） |
| ⑤ | `Duplicate`(V) → `DataCopyPad`(MTE3)（结果搬出） | `V_MTE3` | 结果值可能未落盘 |
| ⑥ | `SetValue`(S，UB) → `DataCopy`(MTE3，UB→GM 发布槽位) | `S_MTE3` | 跨核槽位随机丢失（见下文平台约束） |

**同为「不需要」的两项结论**（避免复核者误判为缺口）：

- `MTE3_MTE2` **不需要**：主通路 kernel 的 MTE3 写与收尾 kernel 的 MTE2 读属于**两次独立 kernel launch**，由同一 `aclStream` 顺序下发保证 GM 可见性；仅同一 kernel 内才需要该事件，本算子无此形态。
- `S_V` **不需要**：本算子无「`SetValue` 写 UB → 向量读」的模式（`+Inf` 播种用 `Duplicate` 走 V 管线，`outBuf_` 的 `SetValue` 之后只接 UB→GM 搬出）；`Duplicate`/`DeInterleave`/`Abs`/`Add` 之间的顺序依赖全部落在同一条 V 管线，硬件保证发射序。
#### `ReduceMin` 的三条平台语义
| # | 语义 | 真机现象与根因 | 本算子的处置 |
|---|---|---|---|
| 1 | 按**整块（8 lane）+ 整 repeat（64 lane）**归约，`count` 之外的车道同样参与比较 | `magBuf_` 恰好按 `tileSize` 分配、只填到 `lanes` 时，归约读到 `tileSize` 之后的 UB（落在 `outBuf_`/`workBuf_` 区域），出现 `0.37` / `0.69` 这类**不可能来自输入**的「最小值」并胜出 | 三项缺一不可：① `MAG_BUF_GUARD_FLOATS = 64`，`magBuf_`/`reBuf_`/`selBuf_` 各多配一个完整 repeat 的 `+Inf` 保护带；② 保护带从**偏移 0** 整段播种（`InitBase` 播种一次，`ComputeMagnitudes` **每个 tile 都整段重播**，防止尾块的过读窗口留着上一个满 tile 的真实模值）；③ 所有覆盖过读窗口的向量调用 `count` 统一用 `tileSize + 64` 而非 `lanes` |
| 2 | `calIndex=true` 的索引输出在并列时**不可靠** | 全并列 tile 上「有时返回 0、有时返回 64」，而语义要求返回第一个（最小）索引 | `calIndex` 恒为 `false`，索引由 `FinalizeIndex()` 的向量化恢复得出（见「索引恢复」） |
| 3 | 单车道 NaN **毒化整块**归约值 | tile 内只要有一个 NaN 车道，整个 tile 的归约值即为 NaN ⇒ 接受判定 `!(v != v)` 判假 ⇒ 整块被丢弃；`incx = 1` + 部分 NaN 填充实测 14 例错、大 n 恒返回 1（而 `incx ≠ 1` 的逐元素判定全程 0 错） | 归约前把 NaN 车道改写为 `+Inf`（语义 = 该元素不参与比较），且**条件化**：先做值归约，仅当归约值为 NaN 时才 `Compare` + `Select` 过滤并重归约。过滤范围用 `lanes` 而非 `tileSize+64`（过读窗口恒为 `+Inf`，不可能含 NaN） |

> **保护带的不变式**：两个有限 float 之和最多溢出为 `+Inf`，不可能超过 `+Inf`，因此 `+Inf` 保护带对结果完全惰性。这也解释了尾块填充值**禁止使用 `FLT_MAX`** 的原因——它是最大**有限**值，会被真实 `+Inf` 车道战胜。
>
> **条件化的必要性**：无条件过滤（在每个 tile 都覆写 `tileSize+64` 车道）在 case3 上增加约 3.8 µs，使 case3 达 32.8~33.3 µs、超门槛约 10%；条件化后干净 tile 零额外开销。
>
> **保护带播种的地址对齐约束**：不得使用 `Duplicate(magLocal[lanes], +Inf, 64)` 这类「从 `magLocal[lanes]` 起写」的形式——`lanes*4` 在 `alignedRemain ≡ 4 (mod 8)` 时只有 16 B 对齐，会触发向量单元对齐异常（真机表现为 trap）。正确做法是从偏移 0 整段播种。
#### pure-AIV class kernel 的裸 `__gm__` 标量写失效
- **现象**：多核用例大面积失败——各核应把 `(最小值, 索引)` 写入 workspace 槽位，跨核归约却读到陈旧值；同一输入、同一 kernel、同一设备下，**部分核写对、部分核完全没有写**，且每轮丢失的核集合都不同（哨兵实验：workspace 预填 `0x12345678`，6 轮残留 23~25 / 56 槽，`zeroed = 0`，即没有任何东西把它们清零；丢失率实测 **12%~45%**）。
- **根因**：本平台按内核形态分类的实现陷阱——**pure-AIV class kernel**（`__global__ __aicore__` + `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY)` + class 封装，即本算子主通路形态）中，`GlobalTensor<T>::SetValue(idx, val)` 与裸 `__gm__ p[i] = v` **两者都是静默 no-op**，写不会到达 GM；机理是 scalar-pipe 写 GM 对后续跨核读不相干，普通 pipe barrier 栅栏不住该路径。
- **处置（本算子采用的配方）**：`UB.SetValue(...)` → `SetFlag/WaitFlag<HardEvent::S_MTE3>` → `DataCopy(UB→GM)`。普通 `DataCopy` 即可，长度取 32 B 的整数倍。修复后哨兵探针 **6/6 轮残留 0**。
- **例外与边界**：被 `asc_vf_call` 包裹的 **SIMT VF 语境**内裸 `__gm__` 写**可用**（本算子 SIMT 通路中途发布标量即属此情形）。判定「能不能用裸 `__gm__` 写」的依据是**当前代码位置是否在 `asc_vf_call` 之内**，而不是「这个 kernel 是不是 SIMT kernel」。
- **登记项**：`icamin_simt_kernel` 内有一个 kernel 体层的空核 `else` 分支，以裸 `__gm__` 标量 store 发布中性槽位，**不在 VF 语境内**，形态上违反上述配方；该分支经论证**不可达**（`numBlocks = min(n, aivCoreNum)` 且调用点保证 `n > 0` ⇒ `numBlocks ≤ n` ⇒ `calNum = perCoreN ≥ 1`）。本轮显式延后修改并登记整改配方：替换为与 AIV 通路 `WriteWorkspace()` 同构的「UB 暂存 + `S_MTE3` + `DataCopy`」发布路径；**任何放宽 `numBlocks` 钳制的改动都必须同时改造此分支**。
### UB 预算与内存占用
**单 tile 的 UB 占用**（按交付件 `InitBase` 的 8 次 `InitBuffer` 逐项核算）：

| # | buffer | 位置 | 用途 | 代数式（字节） | `tileSize = 8192` 时 |
|---|--------|------|------|----------------|----------------------|
| 1 | `inBuf_` | VECIN | 交错复数搬入区 | `8t + 32`（`2t+8` 个 float，`IN_BUF_GUARD_ELEMENTS = 8`） | 65568 |
| 2 | `reBuf_` | VECCALC | 解交织实部；索引恢复期兼作车道号栅格 | `4t + 256`（`t+64` 个 float） | 33024 |
| 3 | `imBuf_` | VECCALC | 解交织虚部 | `4t` | 32768 |
| 4 | `magBuf_` | VECCALC | `\|Re\|+\|Im\|` 结果、`ReduceMin` 的 src；索引恢复期兼作 `Select` 输出 | `4t + 256` | 33024 |
| 5 | `selBuf_` | VECCALC | 被接受 tile 的模值暂存（索引恢复的输入） | `4t + 256` | 33024 |
| 6 | `maskBuf_` | VECCALC | 位压缩掩码（`Compare`/`CompareScalar` 的 dst） | `ceil((t+64)/8)` B 再向上 32 B 取整 | 1056 |
| 7 | `outBuf_` | VECCALC | 归约输出（值槽）与标量写回 staging | `UB_BYTENUM_PER_BLOCK` = 32 | 32 |
| 8 | `workBuf_` | VECCALC | `ReduceMin` 的 `sharedTmpBuffer` 占位块 | 32 | 32 |
| — | **合计** | — | — | **`24t + 1920`** | **198528** |

- **闭式与取值**：`24 × 8192 + 1920 = 198528 B`；对 `UB_SIZE`（248 KiB = 253952 B）利用率 **78.18%**；对含 SIMT 通路的 216 KiB（= 221184 B）预算利用率 **89.76%**。若再叠加 SIMT kernel 内建 `ubPartialVals` / `ubPartialIdxs`（各 2048 × 4 B，共 16384 B，位于**另一个 kernel**，不与主 kernel 的 `TPipe` 分配共存，此处为最保守叠加核算）→ 合计 **214912 B = 216 KiB 的 97.16%**，仍在预算内。
- **无「起始地址对齐填充」项**：`TPipe::InitBuffer` 内部对长度做 `ceil(len/32)*32` 取整，而本实现 8 个长度**已全部是 32 B 的整数倍**（65568 / 33024 / 32768 / 33024 / 33024 / 1056 / 32 / 32），故取整附加量 = 0 B；`TBuf` 句柄在元数据区，不占 UB 数据区。
- **`sharedTmpBuffer` 说明**：950PR/950DT 上 `ReduceMin` 无需使用 `sharedTmpBuffer`，可直接传 src 或任意大小的缓冲；实现仍保留 32 B `workBuf_` 占位（兜底「实现把 scratch 写回 src」的可能），相对满尺寸 workBuffer（`tileSize` 个 float = 32768 B）省 32736 B。
- **`ReduceMin` 的 `count` 校验**：`count = 8192`，repeat 次数 `8192/64 = 128`，满足本平台「不超过 UB 大小限制」的约束（与 255 无关）。

**`tileSize` 上限推导**（双预算取公共保守值）：

| 预算口径 | 约束 | 上限 |
|---|---|---|
| 主体 SIMD kernel | `24t + 1920 ≤ 253952`（248 KiB） | **10501** |
| SIMT 通路（kernel 内含 `__simt_vf__`，需为 DCache 预留 ≥32 KiB） | `24t + 1920 ≤ 221184`（= 256 KiB − 8 KiB 编译器预留 − 32 KiB DCache） | **9136** |

取两预算的**公共保守值 `tileSize = 8192`**：距 9136 上限余 944 个元素（相对余量 10.3%），距 10501 上限余 2309（相对余量 22.0%）。该取值不依赖「DCache 预留是 kernel 级还是模块级」这一未证事实，故结论成立。

**workspace 与结果内存**：

| 项目 | 大小 | 说明 |
|------|------|------|
| 跨核 workspace（三性能 case） | **2048 B** | 56 核 → 64 槽 × 32 B；Host 侧同式推导 `requiredBytes` 并校验 |
| 收尾 kernel 的 UB 搬入 | **2048 B** | `alignedFloats = CeilAlign(512, 64) = 512` 个 float |
| `result` 输出 | 4 B | Device 侧单个 INT32 |
| 上板实测 UB 占用 | **未直接读取** | 无 UB 读取接口；198528 B 为**源码逐 `InitBuffer` 求和**的核算值 |

---
## 精度与测试设计
### golden 实现
| 项目 | 内容 |
|------|------|
| 落点 | `test/iamin/icamin/icamin_golden.h`（107 行） |
| 参考实现 | `icamin_golden`：cblas 风格 CPU 手写循环，`\|Re\|+\|Im\|` 逐元素比较、严格小于才更新、首元素即基准、首元素 NaN 时基准置 `FLT_MAX`；与 kernel 使用**同一条** FP32 表达式 |
| 校验链 | `aclblasIcamin_cpu`：handle → `n < 0` → （`n == 0 \|\| incx < 1`）quick return → `x`/`result` 空指针；顺序与实现一致 |
| 交叉核对 | `icamin_golden_bruteforce`：**独立第二实现**（先算出全部模数组再取 argmin，含 NaN 预过滤），用于定位 golden 自身缺陷，**不作为通过依据** |
| 判定 | `EXPECT_EQ(actual, golden)`，整数索引 bit-exact、**零容差**；quick return 用例额外判定 `ret == SUCCESS && result == 0` |
| 负向用例 | 只判返回码（`INVALID_VALUE` / `HANDLE_IS_NULLPTR`），不检查 `result` 内容 |
### CSV 驱动测试工程
| 文件 | 行数 | 说明 |
|------|------|------|
| `test/iamin/icamin/CMakeLists.txt` | 25 | 测试目标定义（含 `unified_dlog` 链接） |
| `test/iamin/icamin/icamin_param.h` | 55 | 参数类型与用例结构；`ReadMap` 键与 `isamin_param.h` 一致（`n` / `incx` / `x` / `expect_result` / `random_seed`） |
| `test/iamin/icamin/icamin_golden.h` | 107 | CPU golden |
| `test/iamin/icamin/icamin_nan_fill.h` | 61 | NaN 混合填充构造 |
| `test/iamin/icamin/icamin_whitebox_tiling.h` | 258 | tiling 镜像，以 `static_assert` 绑定算子常量（`kDataCopyMaxFloat` 与实现同步为 8192） |
| `test/iamin/icamin/arch35/icamin_test.cpp` | 2512 | CSV 驱动 + 白盒 + 偏移 + 特殊值 + 性能采集 + 运行级报告 |
| `test/iamin/icamin/arch35/icamin_npu_wrapper.h` | 372 | 设备内存/句柄 RAII 与用例派发；**透传真实状态码**（不改写为 `NOT_INITIALIZED`） |
| `test/iamin/icamin/arch35/icamin_test.csv` | 1255（表头 1 + 数据 **1254**） | 精度 1000 + 性能 200 + 补测 54；**前 1201 行（表头 + 任务方原始 1200 行）与任务方原件逐行完全相同** |
| `test/frame/fill.h` | 975 | 公共填充框架，本次新增 Gauss 生成器（`P_GAUSS` / `GaussGenerator` / `RANDOM_GAUSS_<μ>_<σ>`） |

CSV 列定义：`case_name,description,n,incx,x,expect_result,random_seed`；`x` 列为填充模式标识（`RANDOM_NORM_5_5` / `VALUE_NORM_0` / `RANDOM_ALTER` / `RANDOM_EXTREME` / `VALUE_NORM_INF` / `VALUE_NORM_NAN` / `NULLPTR` / `RANDOM_GAUSS_<μ>_<σ>`）。

复现命令（须在不带 `ICAMIN_DEBUG_MARK` 的 Release 构建上采集性能）：

```bash
bash build.sh --soc=ascend950 --ops=icamin
cmake -S . -B build -DSOC_VERSION=ascend950 -DASCEND_CANN_PACKAGE_PATH=${ASCEND_HOME_PATH}
cmake --build build -j8
./build/test/iamin/icamin/icamin_test --gtest_filter=-*TC_PF*                       # 全量精度
./build/test/iamin/icamin/icamin_test --gtest_filter='*TC_PF_1001*:*TC_PF_1002*:*TC_PF_1003*'
ICAMIN_PERF_ALL=1 ./build/test/iamin/icamin/icamin_test                              # 全部性能行
python3 test_cases/verify_performance.py                                             # 任务方脚本口径
```
### 用例构成
| 类别 | 条数 | 说明 |
|------|------|------|
| 精度用例 | 1000 | 任务方随附 CSV |
| 性能用例 `TC_PF_*` | 200 | 一律 `incx = 1`；含形状扫描、小尺寸、特殊填充 |
| 补测用例 `TC_EXT_*` | 54 | 正态分布 48 + `VALUE_NORM_INF` 补强 2 + 规模下探 4 |
| **CSV 数据合计** | **1254** | 任务方原始 1200 行作为前缀逐字节保留，补测续写其后 |
| C++ 层补测（不入 CSV） | 45 | 对齐偏移 32 + 特殊值判别力直接构造项 13。此为**补测数据项**口径：偏移为 32 个参数化实例；13 条判别力数据分布在 `IcaminSpecialValueTest` 的 4 个 `TEST_F` 内（其中 `L1_08_DiscriminatingCases` 为 5 条表驱动数据、`L1_08d_NanPrefixDiscriminating` 含 3 个 seed）。与「执行计数口径」不同——后者按 **gtest 实例**计数，C++ 独有实例为 32 + 25 = 57 |
### 用例分级表
| 分级 | 评测集用例 | 补测用例 | 合计 | 构成 |
|------|-----------|----------|------|------|
| L0 门槛 | 6 | 0 | **6** | `TC_L0_001~006`：n=1/8 × incx=1/2、n=64 全零 tie |
| L1 功能 | 1174 | 99 | **1273** | 尺寸扫描 38 + 正步长跨步 9 + 填充族 12 + 组合采样 915 + 性能 200；补测 99 |
| L2 异常 | 20 | 0 | **20** | n=0 quick return 5 + 非正步长 quick return 8 + n<0 3 + x 空指针 2 + quick-return-优先 2 |
| **合计** | **1200** | **99** | **1299** | 另 C++ 层 3 条不入 CSV 的异常用例（handle 空 / result 空 / n<0 且 x 空） |

补测明细（可逐项相加）：

| 补测项 | 条数 | CSV | C++ | 说明 |
|--------|------|-----|-----|------|
| 正态分布填充 | 48 | 48 | 0 | n ∈ {64,1024,1048576} × (μ,σ) 4 组 × seed 4 组 |
| 对齐偏移 | 32 | 0 | 32 | n ∈ {16,1024} × incx ∈ {1,2} × offset ∈ {0,1,2,3,4,8,16,32}；CSV 无 offset 列 |
| 特殊值判别力 | 15 | 2 | 13 | 首元素 NaN / 模溢出 / 全溢出 / NaN 前置等；带判别力断言的 4 组 |
| 精度规模下探 | 4 | 4 | 0 | n ∈ {2^21, 2^22, 2^23, 2^24}，`TC_EXT_SCALE_*` |
| **小计** | **99** | **54** | **45** | 48 + 32 + 15 + 4 = 99 |

特殊值判别力用例的断言（与构造形态严格匹配）：首元素 NaN → `actual == 2`；`(FLT_MAX, FLT_MAX)` 混有限小值 → `actual == 2`；全 `(FLT_MAX, FLT_MAX)` → `actual == 1`；**NaN 前置**（前一半为 NaN、后一半为有限值）→ `actual ∈ [513, 1024]`。该四条可区分「跳过 NaN」与「IEEE 比较」两种谓词；NaN 后置与 NaN 居中的用例仅作对照，**不加**判别力断言。
### 白盒覆盖
静态枚举口径（逐判定点列出可达结果），覆盖阈值来源为「**全部可达判定点必须有用例命中；不可达必须逐条论证**」：

| 文件 | `if` | `for` | `while` | 三目 + 短路或 | 判定点 | 结果数 | 可达结果 | 已覆盖 | 可达覆盖率 |
|---|---|---|---|---|---|---|---|---|---|
| `icamin_kernel.cpp` | 18 | 9 | 2 | 12 | **41** | 82 | 76 | 76 | **100%** |
| `icamin_host.cpp` | 15 | 0 | 1 | 3 | **19** | 38 | 36 | 36 | **100%** |
| `icamin_tiling_data.h` | 0 | 0 | 0 | 0 | 0 | 0 | — | — | n/a |
| **合计** | **33** | **9** | **3** | **15** | **60** | **120** | **112** | **112** | **100%** |

不可达的 8 个结果落在 4 个判定点上，逐条论证：环境依赖 2 个（`aivCoreNum == 0`、quick return 的 `aclrtMemcpyAsync` 失败分支）、被上游钳制 2 个（`tileSize == 0` 守卫、`nthreads == 0` 守卫，前提 `perCoreN == 0` 而 `perCoreN ≥ 1`）、SIMT 空核 1 个（`calNum == 0`）、条件编译 3 个（`ICAMIN_DEBUG_MARK` 关闭时的调试分支）。本算子**无 TILING_KEY、无模板派发**，运行期分支完全由 `IcaminTilingData` 字段与 `numBlocks` 决定，故 tilingkey 覆盖契约不适用。
### 执行计数口径
| 口径 | 数值 | 构成 |
|------|------|------|
| 源码级执行实例数 | **2165** | `IcaminPrecision` 参数集 1054 ×（`CsvDriven` + `Performance` 两个 `TEST_P` 体）= 2108 + 偏移用例 32 + 其余 `TEST_F` 25 |
| 有精度判定效力的实例 | **1111** | 1054（逐例 `EXPECT_EQ(result, golden)`）+ 32（偏移）+ 25（特殊值/白盒）；另 1054 条为 `Performance` 体的 `NO_REF` 重复执行，只断言返回码 |
| `ICAMIN_PERF_ALL=1` 全量补跑 | **2565** | 全量性能体实例化后执行实例数；`PASSED 2565 / FAILED 0 / exit=0` |
| 计时行 / 无基线行 / 无门槛行 | **1249 / 176 / 1242** | `1254 − 5`（负例 `TC_ED_079~083` 提前返回，无内核可计时）；176 = `178 − 2`；有数值门槛的仅 7 行 |

**真机验收结果**：精度 `PASSED 2165 / FAILED 0 / exit=0`（无 `Segmentation fault`、无 `aclrtMalloc failed`、无 `107000`）；`ICAMIN_PERF_ALL=1` → `PASSED 2565 / FAILED 0`。独立于测试工程的 golden 对拍探针：`icamin_rand` 870/0、`icamin_big` 28/0、`icamin_seq` 5 轮 0 bad、`icamin_simtstab` SIMT 0/8 + AIV 0/8、`icamin_sentinel` 6/6 轮哨兵残留 0。全部为整数索引 bit-exact 断言，非容差比对。

---
## 性能设计
### 标杆与门槛
标杆为 `gpu_baseline.csv` 的 GPU 数据；门槛公式为 `threshold_us = gpu_baseline.csv::gpu_ms × 1000 / 0.4`（即要求相对 GPU 基线的倍率 ≥ 0.4）。三个必达标 case 取任务书 §3.3 的值：

| case | 用例 | n | incx | 标杆耗时（µs） | 实测 `accept_us` | 余量 | 判定 |
|------|------|---|------|----------------|------------------|------|------|
| 1 | `TC_PF_1001` | 1048576 | 1 | **24.77** | **15.677** | 36.7% | PASS |
| 2 | `TC_PF_1002` | 2097152 | 1 | **24.59** | **18.974** | 22.8% | PASS |
| 3 | `TC_PF_1003` | 4194304 | 1 | **29.69** | **26.933** | 9.3% | PASS |

采样口径：`kPerfWarmup = 20`（不计入采样）+ 每轮 `kPerfSamplesPerRound = 100` 次有效采样（单轮即 > 50，满足任务书要求）；`aclrtEvent` 记时且事件记在**算子实际使用的同一 stream**；`kPerfMaxRounds = 3`，`CV > 5%` 取中位数、否则取均值（本轮 `cv_pct` = 1.74 / 0.16 / 0.00，均取均值）。性能数据必须在**不带** `ICAMIN_DEBUG_MARK` 的构建上采集（该调试宏给 n=4M 增加约 15 µs）。

**门槛口径登记**：`gpu_baseline.csv` 以 `(n, incx)` 为键，(4194304, 1) 键**重复 5 次**，任务方脚本按 dict 赋值取后者，实际生效门槛为 30.595 µs（比任务书宽松约 0.9 µs）。本设计**按更严的任务书口径 29.69 µs** 做余量分析与判定；case3 稳态最坏值 29.521 µs，两套口径均满足。

**标杆取值的版本说明**：本设计只采用**本次任务书 §3.3** 给出的 24.77 / 24.59 / 29.69 µs（n = 1048576 / 2097152 / 4194304，incx = 1），与任务书表格逐项一致。部分历史任务材料中出现的 363.55 / 718.46 / 2432.02 µs 属不同版本的 GPU 基线，与本任务书不同源，**不适用于本任务**，不应作为本算子的达标依据。
### 实测数据
| 项目 | case1 | case2 | case3 |
|------|-------|-------|-------|
| 实测耗时（判定值） | 15.677 µs | 18.974 µs | 26.933 µs |
| 任务书门槛 | 24.7675 µs | 24.590 µs | 29.690 µs |
| **实测有效带宽** | **535 GB/s** | **884 GB/s** | **1246 GB/s** |
| 达标所需有效带宽 | 339 GB/s | 682 GB/s | 1130 GB/s |
| 数据量（十进制 MB） | 8.389 MB | 16.777 MB | 33.554 MB |
| 同机实测纯读下界（按数据量线性折算；标定值 18.4~18.9 µs @ 33.554 MB） | 4.60~4.73 µs | 9.20~9.45 µs | 18.40~18.90 µs |

case3 的实测有效带宽 1246 GB/s 为同机实测读带宽 **1.74 TB/s（十进制 MB）/ 1.82 TB/s（MiB）** 的 **68%~72%**。
### 瓶颈归因
- **实测第一瓶颈 = 搬入（MTE2 / GM→UB）**，与设计预判一致。依据为三重独立证据：
  1. 同体量**纯读下界 18.4~18.9 µs**（仓内只读算子 `aclblasIsamin` 读 32 MiB = 18.4 µs、`aclblasSnrm2` = 18.9 µs，事件记在算子同一 stream），折算 case3 占比 **68%~72%**；
  2. 33.554 MB D2D 拷贝代理的独立复算给出同一量级占比（注意：D2D 拷贝原始实测 102.2 µs 含读 + 写两份流量与拷贝路径开销，**不能**当作搬入下界使用，其有效带宽约 0.33 TB/s，远低于纯读）；
  3. 定点消融实验的「无标量扫描对照」实测 24.504 µs（见下条）。
- **修复前的瓶颈不属方案预判错误，而是实现级缺陷**：索引恢复的第一版把「由值反查首车道」写成**逐元素 UB 标量扫描**（`magLocal.GetValue(k)`，每元素一次、约 9 ns/元素），实测占 92%~97% 总耗时（定点实验：n=1M/2M/4M **有扫描 229.8 / 399.0 / 720.4 µs** vs **无扫描 13.741 / 17.071 / 24.504 µs**）。该扫描把「每 tile 一次比较」写成了「每元素一次」，与设计对「每 tile 一次接受判定」的量化（标量占比约 0.02%）不矛盾。
- **优化后的瓶颈转为搬入**：条件化 NaN 过滤 + 向量化索引恢复 + 每核只恢复一次之后，三 case 降至 15.677 / 18.974 / 26.933 µs，第一瓶颈变为 MTE2。
- **计算未被判定为瓶颈，但该结论附条件**：按「4 条向量指令/复数元素、1.65 GHz、发射并行度 8」的模型，case3 单核约 262144 条向量指令、约 19.9 µs，与搬入下界同量级；反解所需并行度约 5.4 条/cycle，处于与 256 B/指令数据宽度可比的量级，因此「计算不是瓶颈」以「向量流水与 MTE2 充分重叠、发射宽度足够」为前提。权威流水拆分（msprof 的 MTE2 / Vector / MTE3 占比）**本轮未采集**，登记为证据补录项；但不改变搬入为第一瓶颈的结论。
### 多轮离散度与测量上下文
- **同一构建的多轮读数**：`TC_PF_1001` 14.922 / 15.091 / 15.677 / 16.684 µs；`TC_PF_1002` 18.535 / 18.974 / 19.239 µs；`TC_PF_1003` 26.573 / 26.653 / 26.933 / 27.385 / **29.521** µs。**最坏值 29.521 µs < 29.690 µs**。
- **处置后的两轮全量复验独立读数**：15.184 / 18.689 / 26.846 µs 与 15.635 / 19.000 / 27.018 µs，均 PASS。
- **测量上下文效应（单独登记）**：同一个 `TC_PF_1003`（同形状、同填充、同构建）在**短序列稳态**为 26.6~27.4 µs，在 `ICAMIN_PERF_ALL=1` 的 **1249 条计时行连续压测尾部**（进程满载约 70 s）约为 **30.78 µs（+14%）**。性能结论一律注明上下文；本设计的判定值采用任务书 §3.3 三个 case 在**默认路径下的稳态值**。
- **基线适用口径（单独登记）**：`gpu_baseline.csv` 以 `(n, incx)` 为键，会把 `(4194304,1)` 的随机数据基线套用到该形状下的 NaN/Inf 等特殊填充行，而这些行会走条件化 NaN 过滤路径，实测 31.2~37.4 µs。该现象属**测试侧基线适用口径**问题（任务书补充说明允许对特殊场景 case 过滤或修改并给出说明），**不是**算子相对其约定形状的性能缺陷；已由「仅对随机均匀填充 `RANDOM_NORM_5_5` 且 `incx = 1` 且 n ∈ {2^20, 2^21, 2^22} 的行赋予数值门槛，其余行记 `NO_REF` 但仍完整测量并打印耗时」的口径修正消除。
### 优化策略
| 编号 | 策略 | 面向的瓶颈 | 落地状态 |
|------|------|------------|----------|
| O1 | **UB 预算拉满：`tileSize = 8192`**（248 KiB 利用率 78.18%、216 KiB 利用率 89.76%），把每核 tile 迭代次数压到 56 核 3/5/10 次、64 核 2/4/8 次，减少搬运指令数与同步次数 | 搬入（MTE2） | **已落地**；实测三 case 15.677 / 18.974 / 26.933 µs 全达标 |
| O2 | 搬入拆条：`DATA_COPY_MAX_FLOAT32 = 8192`，满 tile 的 16384 个 float 拆成 **2 次** `DataCopy`（每次 32768 B），规避 `DataCopyParams.blockLen` 的 `uint16_t` 上限 65535 B | 搬入 | **已落地**；8192 float/条相比 4096 float/条（4 次调用）实测略优 |
| O3 | **向量化索引恢复 + 每核只恢复一次**：`CreateVecIndex` + `CompareScalar` + `Select` + 值归约取最小车道号；tile 循环内只做 `Adds` 暂存，循环结束后 `FinalizeIndex()` 执行唯一一次恢复 | 标量/延迟 | **已落地**；替代逐元素标量扫描（92%~97% 占比）与「每 improving tile 恢复一次」（+5.0 µs） |
| O4 | **条件化 NaN 过滤**：仅当归约值为 NaN 时才过滤并重归约，干净 tile 零额外交付 | 计算 | **已落地**；无条件过滤实测给 case3 增加约 3.8 µs |
| O5 | 尾块填充值取 **`+Inf` 值语义**（`DataCopyPadExtParams<float>` + `__builtin_inff()`），并配 64 lane `+Inf` 加性保护带从偏移 0 整段播种 | 搬入 / tail 正确性 | **已落地**（正确性相关） |
| O6 | **V1 不做双缓冲**：`TQue` 双缓冲至少要把 `inBuf_` 备两份（额外 65536 B），在 `tileSize = 8192` 下 216 KiB 剩余 22656 B、248 KiB 剩余 55424 B，**均不足**；若要「仅 `inBuf_` 双缓冲」成立，需满足 `32t + 1952 ≤ 221184`（216 KiB 口径）⇒ `tileSize ≤ 6851`（248 KiB 口径为 `≤ 7875`） | 延迟（潜在） | **未落地**（V1 锁定大 tile 路线）；实测耗时随 n 近线性、瓶颈为带宽而非延迟，双缓冲未升级为必落地项 |
### 接口声明
`include/cann_ops_blas.h:197`（紧随 `aclblasIsamin`，第 195 行之后）：

```cpp
/**
 * @brief 查找 COMPLEX64 向量中 |Re|+|Im| 最小元素的 1-based 索引（对标 cuBLAS cublasIcamin）。
 * @param handle ops-blas 库上下文句柄，携带 stream；为 nullptr 时返回 ACLBLAS_STATUS_HANDLE_IS_NULLPTR
 * @param n      向量 x 的复数元素个数；n < 0 返回 ACLBLAS_STATUS_INVALID_VALUE；
 *               n == 0 为合法 quick return（不触发 kernel，写 result = 0，返回 SUCCESS）
 * @param x      Device 侧 COMPLEX64 输入向量（实部/虚部各 float32，交错存储），逻辑一维 [n]，
 *               物理长度 1+(n-1)*|incx|；n > 0 且 incx >= 1 时为 nullptr 返回 INVALID_VALUE
 * @param incx   相邻复数元素之间的步长；incx < 1（含 0 与负步长）为合法 quick return
 *               （不触发 kernel，写 result = 0，返回 SUCCESS；不反向遍历）
 * @param result Device 侧 INT32 单值标量输出：最小模元素的 1-based 索引，取值 0（quick return）
 *               或 [1, n]；为 nullptr 返回 ACLBLAS_STATUS_INVALID_VALUE
 * @note 异步执行：读回 result 前须对 handle 绑定的 stream 执行同步（如 aclrtSynchronizeStream）。
 */
aclblasStatus_t aclblasIcamin(aclblasHandle_t handle, int n,
                              const aclblasComplex* x, int incx, int* result);
```

---
# 可维可测分析
## 精度标准
| 验收标准 | 描述 | 标准来源 |
|----------|------|----------|
| 精度标准 | 输出 `result`（INT32）为**整数索引精确相等**：`actual == golden`，零容差、无 rtol/atol；matched_ratio 退化为单值判定（该用例通过即 ratio = 1）。任一用例不相等即失败 | 任务书 §3.2 第 1 条；`test_cases/README.md`「精度阈值」 |
| 输入分量档（背景参照，**不是**输出判定口径） | COMPLEX64 的分量按 FLOAT32 档：rtol = 2^-10（9.77e-4）、atol = 2^-16（1.53e-5）、required_matched_ratio = 0.99、max_abs_error_limit = 1e-2 或 32 × ULP。任务书明确「输出为整数索引时上述浮点阈值退化为精确一致判定」 | 任务书 §3.2 第 2 条附表 |
| 本地断言实现 | `EXPECT_EQ(actualResult, expectedResult)`（期望值来自 `icamin_golden.h::aclblasIcamin_cpu`）；quick return 断言 `EXPECT_EQ(actualResult, 0)` + `EXPECT_EQ(ret, ACLBLAS_STATUS_SUCCESS)`；负向用例断言返回码枚举等值 | `test/iamin/icamin/arch35/icamin_test.cpp` |
| 边界口径 | 溢出（`FLT_MAX + FLT_MAX → +Inf`）在 golden 与实现两侧同为 `+Inf`，使用**同一条** FP32 表达式，不引入第二套真值；NaN 语义为跳过（首元素 NaN 置 `FLT_MAX`） | 任务书 §3.2 补充说明；需求确认结论 |

**可测性**：判定对象是单标量整数，无容差参数、无统计量，用例可通过 `--gtest_filter` 按前缀（`TC_L0` / `TC_SQ` / `TC_INC` / `TC_FL` / `TC_ED` / `TC_EX` / `TC_PF` / `TC_EXT_*`）选取；白盒用例以 `[ICAMIN_WB_*]` / `[ICAMIN_PRECISION]` 日志行输出分支命中证据与逐例 actual/threshold/verdict。
## 性能标准
| 验收标准 | 描述 | 标准来源 |
|----------|------|----------|
| 判定式 | `NPU 平均单次耗时 ≤ gpu_baseline.csv::gpu_ms × 1000 / 0.4`（要求相对 GPU 基线倍率 ≥ 0.4）；须先 warmup 再有效采样 > 50 次取平均 | 任务书 §3.3；`test_cases/verify_performance.py` |
| 判定值 | 取**任务书 §3.3 的严格口径**：case1 1048576 → 24.77 µs、case2 2097152 → 24.59 µs、case3 4194304 → 29.69 µs（脚本因基线重复键实际生效 30.595 µs，仅作双口径登记，不作为达标依据） | 任务书 §3.3；需求确认结论 |
| 环境要求 | 测试设备 Ascend 950PR；性能数据为 COMPLEX64 输入场景下的平均单次耗时（Avg time，µs）；采集构建不得开启 `ICAMIN_DEBUG_MARK` | 任务书 §3.3 第 1 条 |
| 覆盖范围 | 三个必达标 case 均为 `incx = 1`；`incx ≠ 1` 的 SIMT 通路**不参与性能判定**（该通路仅承载功能用例） | 任务书 §3.3；需求确认结论 |

**可测性**：门槛换算常量集中定义在测试工程源码（`kPerfWarmup = 20`、`kPerfSamplesPerRound = 100`、`kPerfThresholdRatio = 0.4`、`kMsToUs = 1000.0`、`kPerfMaxRounds = 3`），便于复核；每行输出 `[ICAMIN_PERF] case=… accept_us=… threshold_us=… verdict=…`，无基线行输出 `[ICAMIN_PERF_NOREF]`，运行末输出运行级报告与瓶颈分解段。
## 兼容性分析
- **新算子，不涉及兼容性分析**：`aclblasIcamin` 为本任务新增的公开接口，ops-blas 仓此前**无**该声明，不存在既有行为需要保持；新增声明紧随同族 `aclblasIsamin`，不修改既有接口。
- **不引入第三方依赖**：精度比对 golden 由测试工程内 CPU 参考实现生成（cblas 风格手写循环，标准 cblas/Netlib 无 icamin 例程），随测试工程提供；无其他三方软件依赖。
- **不新增 950PR 私有平行接口**：声明放入公共头文件 `include/cann_ops_blas.h`，可与其他产品线共用。
- **对既有产品线无影响**：实现位于 `arch35/` 目录，仅随 `--soc=ascend950` 编入；构建脚本对其他 SOC 的编译不受影响。
- **`incx ≠ 1` 的兼容性边界**：该通路走 SIMT、效率低于连续通路，且不计入性能判定；若上游要求 `incx ≠ 1` 也达标，需另立需求（本设计不承诺）。
- **公共文件改动**：`test/frame/fill.h` 为追加式改动（新增 Gauss 生成器与解析分支，不改动既有分支），须回归 `isamin` 等既有测试。
## 风险与开放问题
| 编号 | 项 | 现状与处置 |
|------|-----|-----------|
| Q1 | `n < 0` 语义与 cuBLAS「n ≤ 0 置 0」的差异 | 按任务书 §2.1-4 / §2.4 / §2.5 与仓内同族 `blas/iamin/README.md` 的一致口径落地为 `ACLBLAS_STATUS_INVALID_VALUE`；3 条相关负向用例（n = -1 / -8 / -5）在真机全量用例中 PASS。若上游改判，只需修改 Host 的一个校验分支与 3 条 CSV 期望值 |
| Q2 | 判定门槛双口径（任务书 29.69 µs vs 脚本实际生效 30.595 µs） | 差异来源为 `gpu_baseline.csv` 中 `(4194304, 1)` 键重复 5 行、脚本按 dict 赋值取后者。本设计按更严的 29.69 µs 判定；case3 稳态最坏值 29.521 µs，两套口径均满足，不影响通过判定 |
| Q3 | 超长序列尾部的测量上下文（同一 case +14%） | 同一 `TC_PF_1003` 在短序列稳态 26.6~27.4 µs，在 1249 条计时行连续压测尾部约 30.78 µs。处置：所有性能结论注明测量上下文，判定采用默认路径下的稳态值；该现象已单独登记 |
| Q4 | 权威流水拆分（msprof）未采集 | 设计中的证伪条件「msprof 向量指令占比 > 50%」本轮未取数，属遗留证据缺口；不影响「搬入（MTE2）为第一瓶颈」的结论（由纯读下界、D2D 代理复算、定点消融三重独立证据支撑） |
| Q5 | `icamin_simt_kernel` 空核 `else` 分支的裸 `__gm__` 标量写 | 该分支**不可达**（`numBlocks = min(n, aivCoreNum)` 且 `n > 0` ⇒ `calNum = perCoreN ≥ 1`），本轮显式延后修改并登记整改配方（替换为 UB 暂存 + `S_MTE3` + `DataCopy`）；任何放宽 `numBlocks` 钳制的改动都必须同时改造该分支 |
| Q6 | 测试用例分布缺口（正态分布占比、对齐偏移、部分 NaN、2^20~2^24 精度规模） | 任务方随附用例在这四个维度覆盖为 0 或不可判别，已按任务书 §3.5-4「自行补充」的要求补测 99 条（正态 48 / 偏移 32 / 特殊值判别力 15 / 规模 4）；补测集内部正态与均匀为 1:1，全量口径正态占比约 3.7%，该偏离已确认为最终口径，随附 1200 条不重写 |
## 已知限制与登记项
| # | 项 | 说明 |
|---|----|------|
| 1 | NaN 与 `+Inf` 并存且无有限值 → 返回 1 | 输入同时含 NaN 与 `+Inf`（无任何有限值）时返回 1，而非「第一个 `+Inf` 元素的索引」。原因：NaN 车道被条件化过滤改写为 `+Inf`，此后整块归约值为 `+Inf`，与 BLAS「最小值等于 `+Inf` 时取首个 1-based 索引」的约定一致；该行为在 NaN 修复前后一致。**登记为已知限制、不加分支**。全 `+Inf` 输入（如全 `(FLT_MAX, FLT_MAX)`）与全 NaN 输入为本语义的等价情形，同样返回 1，符合 BLAS 语义 |
| 2 | 全量 run 中 176 行无基线记 `NO_REF` | `NO_REF` 表示「没有可比基线」，**不等于 FAIL**：该 176 行仍完整测量并打印 `accept_us`，只是不出具达标裁定。口径为 `178 − 2`（178 = CSV 中填充 ≠ `RANDOM_NORM_5_5` 的行数；被减的 2 条为 `NULLPTR` 负例，提前返回故不打印）；按前缀分布 `TC_EX_*` 108 + `TC_EXT_GAUSS_*` 48 + `TC_FL_*` 10 + `TC_PF_*` 4 + `TC_L0_*` 2 + `TC_EXT_SP_*` 2 + `TC_ED_*` 2 = 176。有数值门槛的仅 7 行（`TC_PF_1001/1002/1003` + `TC_SQ_044` + `TC_EX_0461` + `TC_EXT_SCALE_01/02`） |
| 3 | `msprof` 权威流水拆分未采集 | `msprof op` 的 `aic_mte2_time` / `aic_vec_time` / `aic_mte3_time` 未采集。最终版四维分解打印行**已取到并归档**（测试工程汇总 suite 的执行顺序缺陷已修复为挂在 `IcaminCsvFixture::TearDownTestSuite`，幂等守卫保证恰好输出一次；其中 `mem_in` 已改名为 `mem_in_d2d_copy_proxy` 并标注「不同维度、不出占比」，另给出同体量纯读下界行，故输出中不出现 >100% 的占比）。登记为证据补录项 |
| 4 | UB 占用为源码逐 `InitBuffer` 求和 | 无 UB 读取接口，198528 B 为**按源码逐 buffer 求和的核算值**（8 项已逐条列出），不是硬件计数器读数；上板实测 UB 占用亦未直接读取，如需硬证据可用算子编译产物的 UB 分配报告 |
| 5 | 调试探针与行数口径 | 调试代码（`ICAMIN_DEBUG_MARK`）全部由 `#ifdef` 包围且默认关闭，不参与交付构建；交付件行数按换行符（LF）计数（末尾换行计入），与部分工具显示的「非空行数」可能不同 |
## 交付清单
**算子实现（4 文件 + 1 接口声明）**：

| # | 路径 | 行数 | MD5 / 说明 |
|---|------|------|------------|
| 1 | `blas/iamin/arch35/icamin_tiling_data.h` | 69 | `#pragma once`；常量 `COMPLEX64_MAX_TILE_SIZE = 8192`、`COMPLEX64_TILE_ALIGN = 64`、`ICAMIN_WS_SLOT_FLOATS = 8` 与 `IcaminTilingData` 结构体 |
| 2 | `blas/iamin/arch35/icamin_kernel.h` | 21 | `icamin_kernel_do` 声明 |
| 3 | `blas/iamin/arch35/icamin_kernel.cpp` | 803 | MD5 `C69EA82428B8DCC16597E688F58C58F8`；4 个 kernel 入口 |
| 4 | `blas/iamin/arch35/icamin_host.cpp` | 177 | MD5 `9C0CE3141653E390E2E46E5B8B327058` |
| 5 | `include/cann_ops_blas.h` | 556（+2 行） | `aclblasIcamin` 声明落点：**197 行** |

**测试工程与文档**（逐文件行数与说明见「详细设计 → 精度与测试设计」）：

| # | 交付件 | 落点与说明 |
|---|--------|------------|
| 1 | 测试工程代码（8 文件，含 CSV 表头 + 1254 例） | `test/iamin/icamin/` 与 `test/iamin/icamin/arch35/`；另在公共 `test/frame/fill.h` 追加 Gauss 生成器（追加式改动） |
| 2 | 算子 README | `blas/iamin/README.md` 新增 `aclblasIcamin` 小节；产品支持表标注 Ascend 950PR：支持 |
| 3 | 测试步骤说明 | 随测试工程提供，覆盖构建、精度、性能三条可复现命令 |
| 4 | 自测报告 | 含用例参数、精度对比结果、性能数据、内存占用数据 |

**其他交付件**：本文档（设计提交路径见元信息表）。
## 参考资料
1. 任务书：`aclblasIcamin_Atlas950PR_task_doc.md`（《9月社区任务-aclblasIcamin算子开发（950）》）
2. cuBLAS `cublasIcamin` 参考文档：https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-amin
3. ops-blas 开源仓：https://gitcode.com/cann/ops-blas
4. 生态算子开源精度标准：https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md
5. 仓内同族算子：`blas/iamin/arch35/`（isamin，Host/分核/二段归约/SIMT 框架复用对象）与 `blas/iamax/arch35/`（icamax，复数模与索引路径参照）
6. CANN 社区任务设计文档模板：https://gitcode.com/cann/cann-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md
