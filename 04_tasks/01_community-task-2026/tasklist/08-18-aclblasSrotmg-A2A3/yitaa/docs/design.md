# 【社区任务】aclblasSrotmg算子设计文档

# 需求背景（required）

## 需求来源

通过社区任务完成昇腾算子开源仓（ops-blas）算子贡献：在 Atlas A2/A3 系列产品（arch22，含 Atlas 800I A2 / Atlas 800I A3）上新增 `aclblasSrotmg` 算子实现，与仓内已有 950PR（arch35）实现共用同一 API，补齐该算子在 A2/A3 产品线的支持。

## 背景介绍

### aclblasSrotmg 算子实现（A2/A3 新增）

ops-blas 开源仓（https://gitcode.com/cann/ops-blas ）已实现 `aclblasSrotmg` 的 arch35（Ascend 950PR）版本：

- 算子实现路径为：`blas/rotmg/arch35/`（srotmg_host.cpp / srotmg_kernel.cpp / srotmg_tiling_data.h）
- 接口声明路径为：`include/cann_ops_blas.h`（第 374 行，`aclblasSrotmg`，与各产品线共用）
- 测试工程路径为：`test/rotmg/srotmg/`（含 arch35 测试目录与 cblas golden 包装）

本任务在 `blas/rotmg/arch22/` 下新增 A2/A3 实现，并在 `test/rotmg/srotmg/arch22/` 下新增对应测试路径，接口签名、参数语义、Host/Device 双路径行为与 arch35 版本完全一致。

（说明：本任务为 BLAS 社区任务，无 TBE 算子源码与算子信息库；对应"源码获取路径"角色的是上表仓内 arch35 参考实现路径与 Netlib `srotmg.f` 参考源码。）

### 标杆算子（cuBLAS cublasSrotmg / Netlib srotmg）现状分析

#### 标杆算子支持的数据类型和格式

| 接口 | 数据类型 | 参数形态 |
| --- | --- | --- |
| cublasSrotmg | FLOAT32 | 5 个标量指针（d1、d2、x1、y1、param[5]），无数组长度/步长/维度参数 |

语义参考 Netlib BLAS `srotmg`（https://www.netlib.org/blas/srotmg.f ），cuBLAS 与之行为一致。

#### 标杆算子实现描述

rotmg 为纯标量计算：读取 4 个标量（d1、d2、x1、y1），构造修正 Givens（modified Givens）变换矩阵 H，使 2×1 向量 (sqrt(d1)·x1, sqrt(d2)·y1)ᵀ 经 H 变换后第二个分量消为 0：

```
H^T * diag(d1, d2) * H = diag(d1_new, d2_new)
H * [x1, y1]^T = [x1_new, 0]^T
```

H 按 BLAS 标准编码进 param[5]，flag 隐含的 1.0 / -1.0 / 0.0 不存入 param：

| param[0] (flag) | H 矩阵 | 存储的元素 |
| --- | --- | --- |
| -1.0 | [[h11, h12], [h21, h22]] | param[1..4] 全部存储 |
| 0.0 | [[1, h12], [h21, 1]] | param[2]=h21，param[3]=h12 |
| 1.0 | [[h11, 1], [-1, h22]] | param[1]=h11，param[4]=h22 |
| -2.0 | [[1, 0], [0, 1]] | 无（恒等变换） |

Netlib 参考实现分支语义（本算子须完整覆盖）：

1. `d1 < 0`：flag = -1，H 元素全置 0，d1、d2、x1 覆写为 0；
2. `d2·y1 = 0`（含 d2 = 0 或 y1 = 0）：flag = -2（恒等），仅设 param[0]，d1、d2、x1 不变；
3. `|d1·x1²| > |d2·y1²|`：h21 = -y1/x1、h12 = (d2·y1)/(d1·x1)，su = 1 - h12·h21；su > 0 时 flag = 0，d1 /= su、d2 /= su、x1 *= su；su ≤ 0 时 flag = -1 全零（舍入边界 safety 分支）；
4. `|d1·x1²| ≤ |d2·y1²|`：d2·y1² < 0（即 d2 < 0）时 flag = -1 全零；否则 flag = 1，h11 = (d1·x1)/(d2·y1)、h22 = x1/y1，su = 1 + h11·h22，d1、d2 缩放后互换，x1 = y1·su；
5. 缩放保护：GAM = 4096、GAMSQ = 1.67772e7、RGAMSQ = 5.96046e-8，当 d1 或 |d2| 越出 [RGAMSQ, GAMSQ] 时循环缩放，缩放过程中 flag 翻转为 -1 并将 H 对应元素置为 ±1。

输出语义：d1、d2、x1 为输入/输出标量，原地覆写；y1 只读；param 为纯输出 5 元素数组。

#### 标杆算子实现流程图

```
            ┌─────────────┐
            │  读入 d1,d2, │
            │  x1,y1       │
            └──────┬──────┘
                   ▼
            ┌─────────────┐   是   ┌──────────────────────┐
            │  d1 < 0 ?   ├──────▶│ flag=-1, H/d1/d2/x1=0 │──▶ 存储
            └──────┬──────┘        └──────────────────────┘
                   │ 否
                   ▼
            ┌─────────────┐   是   ┌──────────────────────┐
            │ d2·y1 == 0 ?├──────▶│ flag=-2, 仅设param[0] │──▶ 返回
            └──────┬──────┘        └──────────────────────┘
                   │ 否
                   ▼
        ┌────────────────────┐
        │ sp1=d1·x1 sq1=sp1·x1│
        │ sp2=d2·y1 sq2=sp2·y1│
        └────────┬───────────┘
                 ▼
      ┌───────────────────────┐
      │ |sq1| > |sq2| ?       │
      └──┬────────────────┬───┘
      是 │                │ 否
         ▼                ▼
 ┌───────────────┐  ┌───────────────┐
 │ h21=-y1/x1    │  │ sq2 < 0 ?     │──是──▶ flag=-1 全零 ──▶ 缩放检查
 │ h12=sp2/sp1   │  └───────┬───────┘
 │ su=1-h12·h21  │          │ 否
 └──────┬────────┘          ▼
        ▼            ┌────────────────┐
 ┌────────────┐      │ flag=1         │
 │ su > 0 ?   │      │ h11=sp1/sp2    │
 └──┬─────┬───┘      │ h22=x1/y1      │
 是 │     │ 否       │ su=1+h11·h22   │
    ▼     ▼          │ d1↔d2 缩放互换  │
 flag=0   flag=-1    │ x1=y1·su       │
 d1,d2,x1  全零      └───────┬────────┘
 更新                │
    └───────┬────────┘
            ▼
 ┌───────────────────────────────┐
 │ 缩放检查：d1(或|d2|) 越出      │◀──┐
 │ [RGAMSQ, GAMSQ] 时循环 ×/÷GAM² │   │ 循环
 │ flag 翻转 -1, H 元素置 ±1      │───┘
 └──────────────┬────────────────┘
                ▼
         ┌──────────────┐
         │ 按 flag 形式  │
         │ 存储 param[0..4]│
         │ 覆写 d1,d2,x1 │
         └──────────────┘
```

# 需求分析（required）

## 需求描述

使用 Ascend C 编程语言在 Atlas A2/A3 系列产品（arch22）上实现 `aclblasSrotmg` 算子，float32 单精度，Kernel 直调（handle 式 BLAS 接口）方式开发，功能与 cuBLAS `cublasSrotmg` / Netlib `srotmg` 完全对齐，精度满足生态算子开源精度标准（FLOAT32：rtol 2⁻¹⁰、atol 2⁻¹⁶、matched_ratio ≥ 0.99、max_abs_error ≤ 1e-2 或 32 ULP；flag 离散值精确相等），单次调用延迟不高于标杆（2.35~2.67 us）。

## 外部组件依赖

| 依赖组件 | 版本/来源 | 用途 |
| --- | --- | --- |
| CANN Toolkit | 9.1.0（昇腾算子开发要求版本） | Ascend C 编译链（ccec）、Ascend C API（kernel_operator.h）、ACL 运行时（aclrt* 接口） |
| ops-blas 工程框架 | https://gitcode.com/cann/ops-blas （master） | 算子工程骨架、handle/stream 管理、构建体系（build.sh / CMake 按 SOC 自动收集 arch22 目录） |
| Netlib BLAS（cblas） | libblas 3.10（仅测试 golden 依赖，非算子运行时依赖） | 精度比对 golden `cblas_srotmg` |

## 内部适配模块

| 模块 | 路径 | 用途 |
| --- | --- | --- |
| 句柄与 stream 管理 | `common/helper/aclblas_handle_internal.h` | `aclblasHandle_t` 携带 stream，`aclblasSetStream` 绑定后 kernel 直调下发 |
| 日志模块 | `log/log.h` | OP_LOGE/OP_LOGD 错误与调试日志 |
| ACL 指针属性接口 | `aclrtPointerGetAttributes` | Host/Device 指针位置判别（双路径分发前提） |
| CSV 驱动测试框架 | `test/frame/`（blas_test.h / csv_loader.h / verify.h / device.h） | GTest 参数化、CSV 解析、mixed tolerance 比对、DeviceBuffer 封装 |

## 需求模块设计

### Ascend C 算子原型

接口声明复用 `include/cann_ops_blas.h` 已有声明（第 374 行，与 950PR 等产品线共用同一 API，禁止定义产品私有平行接口）：

```cpp
aclblasStatus_t aclblasSrotmg(
    aclblasHandle_t handle, float* d1, float* d2, float* x1, const float* y1, float* param);
```

| 参数名 | 输入/输出 | 数据类型 | 说明 |
| --- | --- | --- | --- |
| handle | 输入 | aclblasHandle_t | 库上下文句柄，携带 stream；nullptr 返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| d1 | 输入/输出 | float*（FP32 标量） | x 的缩放因子，原地覆写 |
| d2 | 输入/输出 | float*（FP32 标量） | y 的缩放因子，原地覆写 |
| x1 | 输入/输出 | float*（FP32 标量） | 向量第一分量，原地覆写为 x1_new |
| y1 | 输入 | const float*（FP32 标量） | 向量第二分量，只读 |
| param | 输出 | float*（FP32 [5]） | param[0]=flag，param[1..4] 按 flag 形式存储 H 元素 |

除任务书明确不涉及的维度/形状/枚举参数外，算子行为与标杆 cuBLAS `cublasSrotmg`（语义参考 Netlib `srotmg`）完全对齐。

### Ascend C 算子相关约束（与标杆算子的功能差异）

| 差异项 | 说明 |
| --- | --- |
| 数据类型 | 仅 FLOAT32（与接口声明及标杆一致，非功能缺失） |
| Host/Device 指针约束 | 五指针须全 Host 或全 Device，混合返回 `ACLBLAS_STATUS_INVALID_VALUE`；cuBLAS 无此显式约束（通过 pointer mode 统一管理），本约束与仓内 arch35 版本及 ops-blas 纯标量算子口径一致 |
| 非常规输入行为 | ±Inf / NaN 输入与 Netlib 参考实现逐行一致（如 d1/d2=+Inf 时缩放循环不收敛），不做额外保护，保证与 golden 行为对齐 |
| 其余 | 无缺失功能：flag = -2/-1/0/1 全分支、GAM 缩放保护、原地覆写语义均完整实现 |

## 需求拆解

1. 新增 `blas/rotmg/arch22/` 实现（host + kernel + tiling data），Host 侧参数校验与 Host/Device 双路径判别模式与仓内 arch35 版本保持一致；
2. 新增 `test/rotmg/srotmg/arch22/` 测试路径（CSV 驱动 GTest + NPU wrapper + 测试用例 CSV），并扩展 `srotmg_param.h` 解析随任务提供的 CSV 列格式（空指针负向编码列）；
3. 更新 `blas/rotmg/README.md` 产品支持表，标注 Atlas A2/A3 系列产品支持；
4. 接口声明复用 `include/cann_ops_blas.h` 已有声明，禁止定义产品私有平行 API；
5. 精度自验（golden：cblas_srotmg）与性能自验（warmup + 采样 >50 次取平均）通过并输出自测报告。

# 详细设计（required）

## 算子分析

### 数学公式

```
H^T * diag(d1, d2) * H = diag(d1_new, d2_new)
H * [x1, y1]^T = [x1_new, 0]^T
```

H 由 param[0]（flag）按上表编码；各分支的 h11/h12/h21/h22 计算式与 Netlib `srotmg` 参考实现逐行对齐（见背景介绍分支语义）。

### 支持数据类型

FLOAT32（d1、d2、x1、y1、param 全部为 float 标量/标量数组）。

### 支持形状

不涉及维度概念：5 个参数均为标量指针（param 为 [5] 标量数组），无数组长度、无步长、无维度轴、无枚举参数。

## 算子实现

### 3.1 调用方式

**Kernel 直调（handle 式 BLAS 接口）**。用户经 `include/cann_ops_blas.h` 暴露的 `aclblasSrotmg(aclblasHandle_t handle, ...)` 直接调用，通过 `aclblasCreate`/`aclblasSetStream` 管理上下文与 stream；全 Host 指针时 Host 侧 CPU 直算，全 Device 指针时经绑定的 stream 下发 Ascend C kernel 直调（非 ACLNN、非 Pytorch 框架接入）。

### 3.2 需求总体设计

#### 3.2.1 host 侧设计

##### 3.2.1.1 分核策略

不涉及。纯标量算子（5 个标量指针，总数据量 9 × 4B = 36B），Device 路径固定启动 1 个 block（AIV）完成全部计算，无数据切分，不存在多核划分问题。

##### 3.2.1.2 数据分块和内存优化策略

不涉及。kernel 仅通过 `GlobalTensor<float>` 的 `GetValue/SetValue` 直接读写 GM 标量（d1/d2/x1/y1 各 1 元素、param 5 元素），不搬运至 LocalMemory/UB，无数据分块；UB/Workspace 占用为 0，无内存优化空间与计算公式。

##### 3.2.1.3 tilingKey规划策略

不涉及。SrotmgTilingData 仅承载 5 个 GM 指针地址（uint64_t），无形状/分块参数，不存在多 tiling 分支，不设置 tilingKey。

Host 侧流程（与 arch35 版本保持一致）：

1. **参数校验**：handle 为 nullptr 返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`；d1/d2/x1/y1/param 任一为 nullptr 返回 `ACLBLAS_STATUS_INVALID_VALUE`；
2. **指针位置判别**：对 5 个指针分别调用 `aclrtPointerGetAttributes` 判别 Host/Device 位置；
3. **双路径分发**：
   - 全部 Host 指针 → CPU 直算（`SrotmgCpuCompute`，与 kernel 侧算法逐行一致，保证双路径结果一致），无 kernel 开销、无数据搬运；
   - 全部 Device 指针 → 填充 SrotmgTilingData（5 个指针地址）后启动 1 block kernel，计算与结果留在 Device，供 rotm 等下游算子直接消费；
   - 混合 Host/Device 指针 → 返回 `ACLBLAS_STATUS_INVALID_VALUE`（对齐仓内 blas/rotmg/README.md 既有口径）。

#### 3.2.2 kernel 侧设计

##### 3.2.2.1 kernel 侧实现描述

Device 路径启动单 block（AIV）kernel，采用 arch22 仓内同族纯标量算子（srotg/arch22）的 class 式结构：

- `SrotmgKernel::Init(tiling)`：以 `GlobalTensor<float>::SetGlobalBuffer` 绑定 5 个 GM 标量地址（d1/d2/x1/y1 各 1 元素、param 5 元素）；
- `SrotmgKernel::Process()`：`GetValue` 读入 4 标量 → 按 Netlib 分支语义计算（含 GAM 缩放保护 while 循环）→ 按 flag 形式 `SetValue` 写 param[0..4] 并覆写 d1/d2/x1；
- kernel 入口 `extern "C" __global__ __aicore__ void srotmg_kernel(const SrotmgTilingData tiling)`，`KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY)`，由 Host 侧 `srotmg_kernel_do(...<<<1, nullptr, stream>>>...)` 经 handle 绑定的 stream 下发；
- kernel 内不使用 std::abs/std::sqrt 等 CRT 函数，以私有内联 Abs 辅助函数实现取绝对值（arch22 kernel 侧惯例），算法主体与 Host 侧 CPU 直算逐行一致。

##### 3.2.2.2 Ascend C 实现流程图

（与背景介绍中标杆流程图一致，此处给出双路径分发视角）

```
aclblasSrotmg(handle, d1, d2, x1, y1, param)
        │
        ▼
 ┌──────────────┐  否  ┌───────────────────────────┐
 │ handle 有效? ├─────▶│ HANDLE_IS_NULLPTR          │
 └──────┬───────┘      └───────────────────────────┘
        │ 是
        ▼
 ┌──────────────┐  否  ┌───────────────────────────┐
 │ 5 指针非空?  ├─────▶│ INVALID_VALUE              │
 └──────┬───────┘      └───────────────────────────┘
        │ 是
        ▼
 ┌──────────────────────────────┐  失败  ┌────────────┐
 │ aclrtPointerGetAttributes ×5 │───────▶│INVALID_VALUE│
 └──────┬───────────────────────┘        └────────────┘
        │ 成功
        ▼
 ┌────────────────────────────────────┐
 │ 全 Host? ──是──▶ SrotmgCpuCompute   │
 │ 全 Dev ? ──是──▶ srotmg_kernel<<<1>>>│
 │ 混合     ──是──▶ INVALID_VALUE      │
 └────────────────────────────────────┘
```

##### 3.2.2.3 Ascend C 实现流程图与标杆算子流程图存在的差异点和原因

| 差异点 | 原因 |
| --- | --- |
| cuBLAS 在 CUDA stream 上执行；本实现 Device 路径经 handle 绑定 stream 下发单 block kernel | 昇腾 NPU 执行模型，经 aclblasSetStream 绑定 stream，行为等价 |
| cuBLAS 无显式 Host/Device 双路径；本实现全 Host 指针时 CPU 直算 | 与仓内 arch35 版本及 ops-blas 纯标量算子（srotg 等）既有模式保持一致，避免纯标量场景不必要的 kernel 启动与搬运开销 |
| 缩放保护路径 GAMSQ/RGAMSQ 常量取值与 Netlib 一致（1.67772e7 / 5.96046e-8） | Netlib 文档自述该二常量 may be inexact，统一采用 Netlib 数值，保证与 cblas golden 缩放迭代次数一致 |

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas 800I A2（910B3）等 Atlas A2 训练系列 / Atlas A2 推理系列产品 | √ |
| Atlas A3 训练系列 / Atlas A3 推理系列产品 | √ |

（arch22 目录随 `--soc=ascend910b*` 构建自动纳入；arch35 版本不受影响）

## 算子约束限制

- 仅支持 FLOAT32，不支持其他数据类型（对齐接口声明与标杆）；
- d1/d2/x1/y1/param 五指针须全部位于 Host 侧或全部位于 Device 侧，混合返回 `ACLBLAS_STATUS_INVALID_VALUE`；
- d1/d2 取负值为合法输入（走对应数学分支），不是错误；
- Device 路径结果读回前须同步 stream（`aclblasSetStream` 绑定的 stream）；
- 无维度/形状/broadcast/动态 shape 概念，不涉及。

# 特性交叉分析（required）

## 与其他特性的依赖和影响

| 交叉特性 | 交互关系 | 结论 |
| --- | --- | --- |
| arch35 既有 srotmg 实现 | 本次仅新增 `blas/rotmg/arch22/` 目录，不触碰 arch35 代码；构建体系按 SOC 维度（ascend910b3 → arch22）自动选择实现目录 | 无影响 |
| 仓内其他纯标量算子（srotg/srot/sdsdot 等） | 复用同一 handle/stream 框架、`common/helper` 公共头与测试框架，不修改公共代码（仅本算子私有的 `srotmg_param.h` 列解析扩展，向后兼容旧 CSV 列格式） | 无影响 |
| `include/cann_ops_blas.h` 接口声明 | 未改动，所有产品线共用同一 API 声明 | 无影响 |
| 多进程/多线程 | 纯标量计算无全局状态、无 UB/Workspace、无跨调用缓存，天然线程安全 | 无影响 |
| 动态 shape/断点续训/昇腾混合基准等周边特性 | 单标量算子无 shape 概念、无迭代状态 | 不涉及 |

## 功能失效模式分析

无失败场景：算子不依赖其它特性的使能开关，任意时序下调用行为一致；异常输入（空指针、混合指针位置）在 host 侧校验阶段即返回错误码，不下发 kernel。

# 可维可测分析（required）

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | golden 为 cblas（Netlib `cblas_srotmg`），输出 d1、d2、x1 覆写值及 param[0..4] 逐标量比对；FLOAT32：rtol 2⁻¹⁰、atol 2⁻¹⁶、required_matched_ratio 0.99、max_abs_error_limit 1e-2 或 32 ULP；flag（param[0]）为离散取值 -2/-1/0/1，须精确相等 | 生态算子开源精度标准 + 任务书 §3.2 |
| 性能标准 | Atlas 800I A2（910B3）上 FLOAT32 平均单次耗时不高于标杆：case1（一般值 flag=1）2.67 us、case2（flag=-2 快速返回）2.35 us、case3（缩放保护路径）2.61 us；批量连续调用场景同口径；先 warmup 再采样 >50 次取平均 | 任务书 §3.3 |

## 兼容性分析

非新算子（A2/A3 新增架构实现）：接口签名复用 `include/cann_ops_blas.h` 已有声明，与其他产品线共用；arch35 既有实现、测试路径与行为不受本次改动影响；`blas/rotmg/README.md` 产品支持表新增 A2/A3 支持标注。

