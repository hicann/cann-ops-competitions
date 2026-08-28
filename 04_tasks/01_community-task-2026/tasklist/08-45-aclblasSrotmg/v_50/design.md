# aclblasSrotmg（Atlas A2/A3 / arch22）算子设计文档

> 状态：**定稿**（M0~M7 全程迭代；2026-08-27 云端 CANN 9.1 + 910B3 终验全绿：精度 4000/4000 ALL PASS、全量 4802 PASSED）。依据 cann-competitions `04_tasks/01_community-task-2026/resources/design_template.md` 结构撰写
> 目标仓：gitcode.com/cann/ops-blas，新增 `blas/rotmg/arch22/` 与 `test/rotmg/srotmg/arch22/`
> 母本沿革：初始参照 `blas/rotmg/arch35/`（ascend950）；M2 起计算体按 golden（cblas_srotmg）提供方 **OpenBLAS `interface/rotmg.c` 结构对齐重写**（详见"计算体说明"节），host 与 kernel 两路径始终同源位级一致

---

# 需求背景（required）

## 需求来源

8 月社区任务「aclblasSrotmg 算子开发（A2A3）」任务书（`aclblasSrotmg_Atlas800IA3_task_doc.md`）：在 ops-blas 开源仓以 Ascend C 直调方式为 Atlas A2/A3 产品开发单精度实数标量修正 Givens 旋转参数构造算子 `aclblasSrotmg`，并接入仓内 CSV 驱动 GTest 测试工程完成自测。

## 背景介绍

### aclblasSrotmg 算子现状分析

`srotmg` 属 BLAS Level 1 修正 Givens 旋转型算子族。它不作用于向量，而是**根据 4 个标量输入构造出描述一个 2×2 修正 Givens 矩阵 H 的参数数组**，供同族的向量算子 `srotm` 反复使用（比显式计算 sin/cos 的普通 Givens 旋转 `srot` 数值稳定性更好，可避免中间量溢出）。

接口已在开源仓头文件中声明（本次不改）：`include/cann_ops_blas.h:374`

```cpp
aclblasStatus_t aclblasSrotmg(
    aclblasHandle_t handle, float* d1, float* d2, float* x1, const float* y1, float* param);
```

| 参数 | 方向 | 含义 | 类型 | 形状 | 说明 |
| --- | --- | --- | --- | --- | --- |
| handle | 入 | 句柄（携带 stream） | aclblasHandle_t | — | nullptr 返回 HANDLE_IS_NULLPTR |
| d1 | 入/出 | 对角系数 1（in-place 更新） | float* | 标量 | | 
| d2 | 入/出 | 对角系数 2（in-place 更新） | float* | 标量 | |
| x1 | 入/出 | 点坐标 x 分量（in-place 更新为 x1'） | float* | 标量 | |
| y1 | 入 | 点坐标 y 分量（只读，const） | const float* | 标量 | 不做更新 |
| param | 出 | 编码 H 的 5 元素数组 | float* | 5 元素 | param[0]=flag 离散枚举 |

运行时返回码集合（`include/cann_ops_blas_common.h`）：SUCCESS / NOT_INITIALIZED / ALLOC_FAILED / INVALID_VALUE / EXECUTION_FAILED / INTERNAL_ERROR / HANDLE_IS_NULLPTR 等；本算子主要产生 SUCCESS、INVALID_VALUE、HANDLE_IS_NULLPTR、NOT_INITIALIZED。

上游仓库现状：`blas/rotmg/arch35/`（Ascend950 系列）已有一份可用的 Ascend C 实现；**Atlas A2/A3（arch22/dav-2201）目录缺失**，A2/A3 产品上无法链接到该能力。本任务即补齐 arch22 路径。

---

# 需求分析（required）

## 需求描述

使用 Ascend C 编程语言在 Atlas A2/A3（dav-2201）上实现 `aclblasSrotmg` 算子：

1. 数学语义与 Netlib 参考 BLAS `cblas_srotmg` 等价，golden 取 cblas_srotmg；
2. 支持五指针「全 Host」「全 Device」两种内存布局的全自动分发（句柄式接口特征），混合布局按标准拒绝；
3. 单次 Device 路径调用延迟对标任务书 §3.3 基杆（约 2.35~2.67us 量级）；
4. 通过仓内 CSV 驱动 GTest 测试框架覆盖功能、边界、特殊值、错误注入与性能用例。

## 需求拆解

1. 支持 float32 单一数据类型；
2. 实现参数校验链（handle 空 / 数据指针空 / context 未初始化）并返回规范错误码；
3. 基于 `aclrtPointerGetAttributes` 实现五指针内存位置判定与三态分发；
4. Host 路径：CPU 直接计算（无 kernel、无拷贝，延迟最优）；
5. Device 路径：1-block AIV-only kernel，计算与结果保留在 device 侧供下游 `srotm` 连续使用；
6. 缩放保护（GAM/GAMSQ/RGAMSQ）循环的有界化加固，杜绝异常输入下的死循环挂死；
7. 测试接入：`test/rotmg/srotmg/arch22/` 新增 CSV 用例工程（1200 条），扩展共享 param 解析支持空指针注入用例；
8. 性能采集改造：新增计时 TEST_P 输出微秒级均耗时，与任务书 §3.3 标杆对标。

---

# 详细设计（required）

## 算子分析

### 数学公式

给定标量 `(d1, d2, x1, y1)`，寻找 2×2 矩阵 `H` 及更新后的 `(d1', d2', x1')`，满足：

```
Hᵀ · diag(d1, d2) · H  =  diag(d1', d2')        （保二次型不变换）
H  · [x1, y1]ᵀ  =  [x1', 0]ᵀ                     （将第二分量旋转变换后归零）
```

核心分支决策依赖比较量：`p1=d1·x1`，`p2=d2·y1`，`q1=p1·x1`，`q2=p2·y1`，`u=1∓h12·h21`。

为了防止 `d1/d2` 经 `u` 归约或经乘法放大后跨过 float32 表示域，引入常量
`GAM = 4096`，`GAMSQ = GAM⁴ = 2²⁴ = 16777216`，`RGAMSQ = 5.9604645e-8 = 2⁻²⁴`（取与 golden 参考一致的两精确幂常量）：
检测按 golden 参考结构分为四个独立循环（d1/d2 × 放大/缩小）：缩小方向 `≤ RGAMSQ 且 ≠0` 时 `sd* ×= GAM²`；放大方向 `|sd*| > GAMSQ`（严格大于）时 `sd* /= GAM²`，同循环内的两个 H 元素同向缩放、flag 收敛为 −1。每轮量级变化 `GAM²≈1.7e7` 倍，理论最多约 7 轮必回到界内。

#### param[5] 编码（flag 四态，与 srotm.f 消费端约定一致）

| param[0]（flag） | param[1] | param[2] | param[3] | param[4] | 还原的矩阵 H |
| --- | --- | --- | --- | --- | --- |
| −2 | 0 | 0 | 0 | 0 | 退化：无可表达旋转（`d2·y1==0` 时快速返回，d1/d2/x1 原样保持） |
| −1 | h11 | h21 | h12 | h22 | `[[h11,h12],[h21,h22]]` 一般形 |
| 0 | 0 | h21 | h12 | 0 | `[[1,h12],[h21,1]]` 对角省略形 |
| +1 | h11 | 0 | 0 | h22 | `[[h11,0],[0,h22]]` 秩一对角形 |

主流程伪码（与 Netlib srotmg.f 等价）：

```
if d1 < 0            → flag=-1，输出清零并置 d1=d2=x1=0
else if d2·y1 == 0   → flag=-2 快速返回（仅写 param[0..4]，d1/d2/x1 不变）
else if |q1| > |q2|  → h21=-y1/x1, h12=p2/p1, u=1-h12·h21
                         u>0 → flag=0，(d1,d2,x1)/u、sx1·u
                         否则 → flag=-1 失败态清零
else if q2 < 0       → flag=-1 失败态清零
else                 → flag=1，h11=p1/p2, h22=x1/y1, u=1+h11·h22，
                       swap(d1,d2)、sx1=y1·u
随后分别对 sd1、|sd2| 执行 GAM 缩放保护循环
最后按 flag 三态把 (h11,h21,h12,h22) 编码进 param[1..4]
```

IEEE-754 特殊值天然性质：NaN 参与的全部大小比较均为 false，使缩放循环/分支自然跳过，无需显式 `isnan` 判定，CPU/golden/device 三方行为一致。

### 支持数据类型

| 项 | 支持类型 |
| --- | --- |
| d1 / d2 / x1 / y1 / param | float32（FP32）唯一类型 |

### 支持形状

纯标量操作数：`d1/d2/x1/y1` 各 1 个元素、`param` 5 个元素；不存在 shape、广播或维度概念。唯一的结构性约束是**内存位置一致性**——五个指针必须全部位于 host 或全部位于 device（通过运行时指针属性查询判定），这是句柄式 aclBLAS 双路径设计的核心前提。

## 算子实现

### 实现方案

整体结构：Host 侧薄校验 + 三态分发 + 双执行路径；Kernel 侧单 block 标量直算。计算总量固定为「4 个标量读入 + 少量浮点运算 + 9 个标量写回」，无 workspace、无 tiling 切分需求。

#### host 侧设计

```
aclblasSrotmg(handle, d1, d2, x1, y1, param)
 ├─ ValidateSrotmgParams
 │    ├─ handle==nullptr                 → OP_LOGE + HANDLE_IS_NULLPTR
 │    └─ 任一数据指针==nullptr            → OP_LOGE + INVALID_VALUE
 ├─ context 就绪检查（arch22 家族惯例）    → ACL_SUCCESS&&ctx!=nullptr 否则 NOT_INITIALIZED
 ├─ SrotmgCheckPtrLocation ×5            → aclrtPointerGetAttributes 失败 → OP_LOGE + INVALID_VALUE
 ├─ allHost  ?  → SrotmgCpuCompute(...) 写回 5 元素 param，立即返回 SUCCESS（最优延迟路径）
 ├─ allDevice?  → SrotmgTilingData tiling{}（占位空壳）
 │               srotmg_kernel_do(d1,d2,x1,y1,const_cast<float*>(y1),param, tiling, /*numBlocks=*/1, h->stream)
 │               异步下发后返回 SUCCESS（同步由调用方/stream 语义负责）
 └─ 混合布局   → OP_LOGE（打印 5 指针各自 dev/host 属性）+ INVALID_VALUE
```

设计取舍说明：

1. **混合布局拒绝而非隐式搬运**：隐式搬运会引入两次 D2H/H2D 同步拷贝，破坏 BLAS 层「位置契约」，且下游 `srotm` 预期结果留在 device 侧；与 arch35 及生态实现保持一致。
2. **tiling 为占位空壳**（`struct SrotmgTilingData { uint8_t _placeholder; }`）：kernel 启动框架要求透传 tiling 结构，本算子无任何切分参数，故维持占位体以保证与仓内启动宏/其余算子的结构统一，也不引入 tilingkey（无双分支）。
3. **CPU 计算函数与 kernel 计算体由同一份 golden 对齐算法逐分支移植**，除绝对值取法（device 手写三元）与缩放循环上限护栏外逐字符一致（含 `GAM=4096.0f`、`GAMSQ=16777216.0f`、`RGAMSQ=5.9604645e-8f` 精确字面量），确保两路径 bit 级一致。
4. **y1 只读保护**：接口为 `const float*`，kernel 入参统一 `uint8_t*`，传参处 `const_cast` 并保证只读访问。

#### kernel 侧设计

```cpp
__global__ __aicore__ void srotmg_kernel(GM_ADDR d1, …, GM_ADDR param, SrotmgTilingData t)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);   // 仅矢量核参与
    if (GetBlockIdx() != 0) return;                    // 单核执行，其余核早退
    // GlobalTensor<float>::SetGlobalBuffer + GetValue/SetValue 直访 GM 标量
    // ……与 Host 相同的四分支 + GAM 缩放保护循环 + 三态 STORE……
}

void srotmg_kernel_do(uint8_t* d1, …, uint8_t* param,
                      const SrotmgTilingData& tiling, uint32_t numBlocks, void* stream)
{
    srotmg_kernel<<<numBlocks, nullptr, stream>>>(d1, d2, x1, y1, param, tiling);
}
```

关键选型理由：

1. **AIV-only + 1 block**：总数据量 9 个 float，多核并行只会带来 launch 与核间开销；`numBlocks=1` 固定下发。`KERNEL_TYPE_AIV_ONLY` 显式声明只用矢量核，避免 Cube 资源被占位。`GetBlockIdx()!=0` 早退保留将来按 load 动态扩块的余地。
2. **GetValue/SetValue 直访 GM**（而非 DataCopy→UB→CopyOut 流水）：数据不足一个 vector 长度，走 MTE 搬运流水反而引入对齐要求（MTE 要求 32B 对齐块）、事件等待与 pipe 配置成本；标量直访是最短路径。启动开销才是本算子延迟的大头。
3. **device 侧禁用 `std::abs`/`isnan`**：一律用手写三元 `(v < 0) ? -v : v`；NaN 的处理完全依赖比较语义。
4. **计算体与 cblas_srotmg golden 位级对齐**：验收精度判定逐标量对照 `cblas_srotmg`，故算法体不是 netlib srotmg.f 的字面移植，而是采用主流 BLAS 发行版（OpenBLAS `interface/rotmg.c`，经 lapack-3.5.0 测试验证）的算法结构——快速返回前置、`(d1==0‖x1==0)&&d2>0` 特例分支、small/big×d1/d2 四个独立缩放循环且循环内元素同向缩放、big 界严格 `> GAMSQ`。已在 Host 路径以 824 组用例（边界值/denormal/near-max/NaN/-0/随机 logmag）验证与 golden 位级全等；Device kernel 为同一份逻辑的 GetValue/SetValue 机械改写，仅绝对值改手写三元，保证两路径 bit 一致。
5. **缩放保护循环有界化**（本实现在 golden 参考之上的差异化加固）：每个缩放循环附加 `kMaxScaleIters = 64` 上限（理论必要轮次 ≈7）。有限输入下提前收敛不受影响（上限远未触达）；输入为 ±inf 等病态值时终止循环转稳定出口，杜绝死循环挂死整个 stream。Host 与 Device 同一上限策略。
6. **可见性与同步**：所有写回经由 stream 语义完成（测试 wrapper 以 `aclrtSynchronizeDevice` 后再做 D2H 校验）；GM 标量直访不经过 UB，无需额外 pipe barrier。
7. **无 workspace**：内存分配量为 0（验收报告随附 `aclrtGetMemInfo` 差值数据佐证）。

---

# 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas 800I A2 / Atlas 800T A2（ascend910b*，dav-2201） | √ |
| Atlas 800I A3 | √ |

环境备注：本设计文档的实现于 Chaoqiang A800I A2 整机（910B4-1 ×8，同属 dav-2201 二进制兼容族）完成编码与编译验证；任务书性能标杆设备为 Atlas 800T A2（910B3），二者同架构指令集，正式性能终验数据在标杆机型环境出具。

---

# 算子约束限制

1. 仅支持 float32；
2. 五个指针必须同侧（全 host / 全 device），混合布局返回 `INVALID_VALUE`；
3. `y1` 是只读输入（`const float*`），不支持将其作为输出复用；
4. `d1`/`d2`/`x1` 为 in-place 输入兼输出，调用方需保证可写且生命周期覆盖异步执行期；
5. `d1`/`d2` 为 ±inf 的病态输入下，GAM 缩放保护循环受 `kMaxScaleIters=64` 上限约束提前终止，此情形输出与参考实现可能不同（避免挂死的取舍）；
6. NaN 输入按 IEEE-754 比较语义自然流经分支，不引入显式 nan 分支。

---

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | 输入输出均为 FP32 标量，按生态 FLOAT32 标准判定：`atol=2⁻¹⁶`、`rtol=2⁻¹⁰`、`matched_ratio≥0.99`、`max_abs_error≤1e-2`（或 32ULP）；`flag` 为离散整数编码需与 golden 精确相等（允许 inf==inf，双方同为 NaN 判过）。golden 由 Netlib `cblas_srotmg` 生成。不同 BLAS 库在缩放边界的尾差可能出现 ±1 轮缩放良性偏差，按任务书 §3.2 口径解释 | 任务书 §3.2；cann/opbase `experimental_standard.md` |
| 性能标准 | Device 路径单次调用延迟：case1（flag=1 一般路径）≤2.67us、case2（flag=-2 快速返回路径）≤2.35us、case3（缩放保护路径）≤2.61us，按任务书规定倍率放宽；主张口径为剥离 H2D/D2H 搬运后的稳定单次调用延迟，同时报告 msprof 纯 kernel 耗时交叉佐证 | 任务书 §3.3 |

## 可测性设计

- 测试载体：`test/rotmg/srotmg/arch22/` CSV 驱动 GTest 工程（CSV 与 .cpp 同名同目录自动装配），两种 TEST_P——Device 包装器路径与 Host pinned memory 路径分别驱动；
- 用例分组：TC_L0 功能正确性 / TB_* 边界 / TC_SC 缩放边界（RGAMSQ~GAMSQ 全区间段）/ TC_FL 特殊值（±inf、±NaN、denormal）/ TC_ED 错误注入（null 指针、混合布局，断言错误码）/ TC_PF 性能用例（微秒级 `[PERF_RESULT] avg_us=` 格式输出）；
- 结果可视性：OP_LOGE 覆盖每个错误出口并带 `aclblasSrotmg` tag 便于日志检索；glog 级别可通过环境变量调节。

## 兼容性分析

- 新增文件全部位于独立新目录 `blas/rotmg/arch22/`、`test/rotmg/srotmg/arch22/`；`blas/rotmg/arch35/` 一行不动；除共享 `srotmg_param.h` 做向后兼容的字段扩展外不触碰存量文件；
- 构建侧零改动：CMake 按 soc 自动 glob `<op>/arch22/*.cpp`，不影响其他算子目标；
- API/ABI：`aclblasSrotmg` 声明为既存接口（`include/cann_ops_blas.h` 不修改），无签名、错误码枚举变更；
- 行为兼容：校验顺序、错误码、日志格式与 arch35 现网实现一致，便于生态侧无缝切换芯片型号。
