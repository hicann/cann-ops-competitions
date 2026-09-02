# aclblasCdotc 算子设计文档

## 一、需求背景

### 1.1 需求来源

通过**社区任务**完成开源仓算子贡献的需求。在昇腾 NPU（Ascend 950PR / arch35）上使用 Ascend C 开发 `aclblasCdotc`（complex64 向量共轭点积）算子，对齐 cuBLAS `cublasCdotc` / Netlib BLAS `cdotc` 语义，纳入 `ops-blas` 开源仓。

- 任务书算子：`aclblasCdotc`
- 目标硬件：Ascend 950PR（arch35 / DAV_3510）
- 参考标杆：cuBLAS `cublasCdotc`、Netlib BLAS `cdotc`
- 代码仓：`ops-blas`（Ascend BLAS 算子库），源码路径 `blas/dot/arch35/`

### 1.2 背景介绍

#### 1.2.1 aclblasCdotc 算子实现优化

本任务在 `ops-blas` 开源仓中新增 `aclblasCdotc` 算子实现，属 Level-1 级 BLAS 向量规约算子（vector reduction），无需 TBE 算子源码迁移（该算子不来自传统 TBE 算子仓），而是对齐 BLAS 标准语义进行原生实现。

**参考资源路径：**
- 接口定义（算子信息库 / 头文件）：`ops-blas/include/cann_ops_blas.h`（第 188-190 行已声明 `aclblasCdotc`）
- 算子实现目录：`ops-blas/blas/dot/arch35/`（`cdotc_kernel.cpp` / `cdotc_host.cpp` / `cdotc_tiling_data.h`）
- 测试目录：`ops-blas/test/dot/cdotc/`
- 语义参考：Netlib BLAS `cdotc`（`zc.dotc` 复数共轭点积）、cuBLAS `cublasCdotc`
- 同族已有实现（复用参考）：`ops-blas/blas/dot/arch35/cdotu_kernel.cpp`（非共轭点积，语义差异仅为共轭）、`ops-blas/blas/asum/arch35/scasum_kernel.cpp`（成功验证的 SIMT 归约范式）

#### 1.2.2 aclblasCdotc 算子现状分析

`ops-blas` 仓内已存在 `aclblasCdotu`（非共轭复数点积）实现，但缺少共轭版本 `aclblasCdotc`。本任务补齐该接口。

##### 1.2.2.1 标杆算子支持的数据类型和数据格式

与算子信息库/接口声明保持一致：

| 项 | 取值 |
|----|------|
| 输入类型 | `aclblasComplex`（complex64，float real + float imag，8 字节对齐） |
| 输出类型 | `aclblasComplex`（单个复数标量） |
| 数据布局 | 向量 x、y 连续存储；支持任意步长 incx/incy（含负步长） |
| 索引语义 | `k = 1+(i-1)*incx`，`j = 1+(i-1)*incy`，`i = 1..n`（1-based，Netlib 语义） |

##### 1.2.2.2 标杆算子实现描述

Netlib/cuBLAS `cdotc` 计算定义（与标准 BLAS 完全一致）：

```
result = Σ_{i=1..n} conj(x[k]) * y[j]
  k = 1 + (i-1)*incx
  j = 1 + (i-1)*incy
```

展开为实部/虚部分量（第一操作数共轭，虚部取反）：

```
real = Σ( x[k].re * y[j].re + x[k].im * y[j].im )
imag = Σ( x[k].re * y[j].im - x[k].im * y[j].re )
```

与 `cdotu`（非共轭）的核心差异：`cdotu` 的实部为 `x.re*y.re - x.im*y.im`、虚部为 `x.re*y.im + x.im*y.re`（共轭号位置相反）。

**边界语义（对齐 cuBLAS）：**
- `n <= 0`：不执行计算，`result = (0, 0)`，返回 SUCCESS
- `incx == 0` 或 `incy == 0`：返回 `ACLBLAS_STATUS_INVALID_VALUE`
- `n < 0`：返回 `ACLBLAS_STATUS_INVALID_VALUE`
- `x`/`y`/`result` 为空指针（n>0）：返回 `ACLBLAS_STATUS_INVALID_VALUE`
- 负步长（`incx < 0`/`incy < 0`）：合法，反向索引 `(n-1-i)*|inc|`

##### 1.2.2.3 标杆算子实现流程图

```
┌─────────────────────────────────────────┐
│           aclblasCdotc (入口)            │
└──────────────────┬──────────────────────┘
                   ▼
        ┌─────────────────────┐
        │  参数校验            │
        │  handle/null/inc==0  │───→ INVALID_VALUE / NOT_INITIALIZED
        └─────────┬───────────┘
                  ▼
        ┌─────────────────────┐
        │  n <= 0 ?           │───yes──→ result=(0,0), SUCCESS
        └─────────┬───────────┘
                  ▼ no
        ┌─────────────────────┐
        │  计算 tiling         │
        │  分核 / 分块         │
        └─────────┬───────────┘
                  ▼
        ┌─────────────────────┐
        │  启动 SIMT kernel    │
        │  (多核并行归约)      │
        └─────────┬───────────┘
                  ▼
        ┌─────────────────────┐
        │  单核最终归约        │
        │  → result           │
        └─────────┬───────────┘
                  ▼
           返回 SUCCESS
```

## 二、需求分析

### 2.1 外部组件依赖

| 依赖组件 | 版本/说明 | 用途 |
|----------|-----------|------|
| ACL (AscendCL) | CANN 9.1.0 | 设备内存管理、stream、kernel launch |
| AscendC kernel 运行时 | CANN 9.1.0 | `__gm__`/`__ubuf__` 地址空间、`asc_ldcg`、`asc_syncthreads`、`CrossCoreSetFlag/WaitFlag` |
| SIMT API (`simt_api/asc_simt.h`) | CANN 9.1.0 | SIMT 编程模型（`asc_vf_call`、`threadIdx.x`） |
| GTest | 系统已装 | 参数化测试框架 |
| CBLAS/LAPACK | 系统已装 | 参考 golden 对照（间接） |

### 2.2 内部适配模块

| 内部模块 | 说明 |
|----------|------|
| `common/helper/aclblas_handle_internal.h` | handle 结构、workspace 管理（`GetEffectiveWorkspace`） |
| `common/helper/kernel_constant.h` | SIMT 线程常量（`SIMT_MIN/MAX_THREAD_NUM`） |
| `common/helper/kernel_utils.h` | `RoundUpPow2` 等工具 |
| `log/log.h` | 日志输出（`OP_LOGE`/`OP_LOGD`） |
| `test/frame`、`test/utils`（fill.h） | 测试框架、数据填充 |

### 2.3 需求模块设计

#### 2.3.1 Ascend C 算子原型

```c
aclblasStatus_t aclblasCdotc(
    aclblasHandle_t handle,        // aclblas 句柄
    int n,                         // 向量长度
    const aclblasComplex* x,       // 输入向量 x（设备指针）
    int incx,                      // x 步长
    const aclblasComplex* y,       // 输入向量 y（设备指针）
    int incy,                      // y 步长
    aclblasComplex* result         // 输出复数标量（设备指针）
);
```

#### 2.3.2 Ascend C 算子相关约束

与标杆算子（cuBLAS `cublasCdotc`）相比，当前实现存在以下功能差距/约束：

| 功能点 | cuBLAS | 本设计 | 差异说明 |
|--------|--------|--------|----------|
| 数据类型 | complex64 | complex64 | 一致 |
| 正步长 | 支持 | 支持 | 一致 |
| 负步长 | 支持 | 支持（索引对齐 Netlib 语义） | 一致 |
| 大规模 n≥16384 | 支持 | 支持（Kahan 补偿 + 多核归约保证精度） | 一致 |
| n<=0 quick-return | 支持 | 支持 | 一致 |

## 三、需求详细设计

### 3.1 调用方式

本任务采用 **Kernel 直调（ops-blas 内部 API）** 方式：
- 上层通过 `aclblasCdotc(handle, n, x, incx, y, incy, result)` 调用
- host 侧（`cdotc_host.cpp`）完成参数校验、tiling 计算、workspace 分配，调用 `cdotc_kernel_do` 启动 kernel
- kernel 侧（`cdotc_kernel.cpp`）执行 SIMT 归约
- 输出 `result` 为设备指针，由调用方负责 D2H 拷贝

### 3.2 需求总体设计

整体采用**单 kernel 多核 SIMT 归约**架构：

```
host: 参数校验 → tiling 计算 → workspace 准备 → launch kernel
kernel: [各核 SIMT 计算部分和] → [cross-core 同步] → [block0 最终归约 → result]
```

#### 3.2.1 host 侧设计

##### 3.2.1.1 分核策略

- 使用核数 `useCoreNum = min(16, n)`（n>0 时最多 16 核，n<16 时按 n 取）
- 每核处理元素数：`base = n / useCoreNum`，前 `n % useCoreNum` 核各多 1 个
- 每核数据范围 `[startOffset[i], startOffset[i]+calNum[i])`，其中
  `startOffset[i] = Σ_{k<i} calNum[k]`
- 核内由 1024 线程按 `threadIdx.x + k*blockDim.x` 步长遍历

##### 3.2.1.2 数据分块和内存优化策略

**workspace 规划：**
- 每核输出 `(real_partial, imag_partial)` 两个 float → 每核 8 字节
- workspace 总大小 = `useCoreNum * 2 * sizeof(float)`（≤ 16*8=128B），从 handle 默认 workspace（32MB）中借用，无需额外分配
- 内存布局：`workspace[2*i] = 核 i 的实部部分和`，`workspace[2*i+1] = 核 i 的虚部部分和`

**LocalMemory（UB）使用：**
- 每线程 4 个 float 寄存器级累加器（sum_real/c_real/sum_imag/c_imag），零 UB 开销于计算
- UB 仅用于块内归约缓冲：`ubPartialReal[1024] + ubPartialImag[1024]`，共 `2*1024*4 = 8KB`
- 950PR UB 容量 248KB，8KB 占比 ~3.2%，余量充足

**数据访问：**
- 复数元素按 `float2`（8B）向量化读取：`asc_ldcg(&xf2[idx])`，每元素单次 8B 加载
- 支持任意步长：逻辑索引 → 物理索引 `xi = elemIdx*incx`（正步长）或 `(n-1-elemIdx)*|incx|`（负步长）

##### 3.2.1.3 tilingKey 规划策略

本算子为 Level-1 级固定模式规约算子，计算路径单一（SIMT 归约），**不引入 tilingKey 分支**。tiling 数据（`CdotcTilingData`）整体由 host 计算后随 kernel 参数传入：

```cpp
struct CdotcTilingData {
    int64_t n;               // 元素总数
    int64_t incx, incy;      // 步长
    int32_t useCoreNum;      // 使用核数
    uint32_t startOffset[16];// 每核起始偏移
    uint32_t calNum[16];     // 每核计算元素数
    uint32_t nthreads;       // 每核线程数 1024
};
```

#### 3.2.2 kernel 侧设计

##### 3.2.2.1 kernel 侧实现描述

kernel 分三步执行：

1. **SIMT 部分和计算**（每核独立，`CdotcSimtCompute`）：
   - 1024 线程按步长遍历本核 `calNum` 个元素
   - 每元素读取 `x[xi]`、`y[yi]`（float2 向量化），计算
     `prod_real = x.re*y.re + x.im*y.im`
     `prod_imag = x.re*y.im - x.im*y.re`
   - **Kahan 补偿**：每个线程维护 4 个累加器（sum/c 各 real/imag），抑制浮点累加 round-off

2. **块内归约**（每核，UB tree-reduce）：
   - 每线程部分和写入 `ubPartialReal[threadIdx.x]` / `ubPartialImag[threadIdx.x]`
   - `asc_syncthreads()` + 对数级 tree-reduce（1024→1）
   - thread 0 将 `(sum_real, sum_imag)` 写入 `workspace[blockIdx*2]` / `workspace[blockIdx*2+1]`

3. **跨核同步 + 最终归约**（单核）：
   - `CrossCoreSetFlag<0, PIPE_MTE3>(0)` / `CrossCoreWaitFlag<0, PIPE_MTE3>(0)` 保证各核部分和已落 workspace
   - block 0 顺序累加 `Σ_{i} workspace[2i]` 与 `Σ_{i} workspace[2i+1]`，写入 `result`

##### 3.2.2.2 Ascend C 实现流程图

```
┌────────────────────────────────────────────────────────┐
│                cdotc_simt_kernel                        │
│  KERNEL_TASK_TYPE_DEFAULT(AIV_ONLY)                     │
└─────────────────────────┬──────────────────────────────┘
                          ▼
        ┌────────────────────────────────────┐
        │  blockIdx = GetBlockIdx()          │
        │  calNum = tdata.calNum[blockIdx]   │
        └─────────────────┬──────────────────┘
                          ▼
        ┌────────────────────────────────────┐
        │  calNum > 0 ?                      │
        └─────────┬──────────────────────────┘
        yes       ▼
   ┌───────────────────────────────┐
   │ asc_vf_call<CdotcSimtCompute> │
   │   dim3{1024,1,1}              │
   │   线程步长遍历 calNum 元素     │
   │   asc_ldcg 读 x/y (float2)    │
   │   prod_real/prod_imag 计算    │
   │   Kahan 补偿 (4 累加器)        │
   │   ─────────────────────────   │
   │   UB tree-reduce (1024→1)     │
   │   asc_syncthreads 同步        │
   │   thread0 → workspace[2*bi]   │
   │            workspace[2*bi+1]  │
   └───────────────┬───────────────┘
                   ▼
        ┌────────────────────────────────────┐
        │  CrossCoreSetFlag<0,MTE3>(0)       │
        │  CrossCoreWaitFlag<0,MTE3>(0)      │
        └─────────────────┬──────────────────┘
                          ▼
        ┌────────────────────────────────────┐
        │  blockIdx == 0 ?                   │
        └─────────┬──────────────────────────┘
        yes       ▼
   ┌──────────────────────────────────┐
   │  for i in 0..useCoreNum:         │
   │    real += ws[2i]; imag+=ws[2i+1]│
   │  resultGm[0] = {real, imag}      │
   └──────────────────────────────────┘
```

`CdotcSimtCompute` 内部详细流程：

```
threadIdx.x 步长遍历:
  i = threadIdx.x; i < calNum; i += blockDim.x
    elemIdx = startOffset + i
    xi = incx>0 ? elemIdx*incx : (n-1-elemIdx)*(-incx)
    yi = incy>0 ? elemIdx*incy : (n-1-elemIdx)*(-incy)
    (xr,xi_) = asc_ldcg(&xf2[xi]); (yr,yi_) = asc_ldcg(&yf2[yi])
    prod_real = xr*yr + xi_*yi_
    prod_imag = xr*yi_ - xi_*yr
    KahanAdd(sum_real,c_real,prod_real)
    KahanAdd(sum_imag,c_imag,prod_imag)
→ ubPartial[threadIdx] = sum
→ tree-reduce (s=512,256,...,1)
→ thread0: workspace[0]=real, workspace[1]=imag
```

##### 3.2.2.3 Ascend C 实现流程图与标杆算子流程图存在的差异点和原因

| 差异点 | 标杆（Netlib/cuBLAS） | Ascend C 本实现 | 原因 |
|--------|------------------------|-----------------|------|
| 求和顺序 | 标量顺序累加（串行） | 多核并行 tree-reduce + Kahan | 多核并行提速；Kahan 抑制浮点误差 |
| 单核/多核 | 单核串行 | 最多 16 核并行 | 950PR 多核并行降低延迟 |
| 复数乘法 | 通用复数乘（含共轭） | 手工拆解为 2 个实数乘法累加 | SIMT 标量指令更高效，避免复数指令开销 |
| 归约策略 | 单累加器 | 两段归约（块内 tree + 跨核串行） | 块内并行归约降低深度，跨核由 block0 兜底 |
| 快速返回 | n<=0 直接返回 | host 侧 n<=0 写 (0,0) 后返回 | 语义一致，host 侧实现减少 launch 开销 |

### 3.3 支持硬件

与《算子任务书》要求一致：**Ascend 950PR（arch35 / DAV_3510）**，CANN 9.1.0。

### 3.4 算子约束限制

| 约束 | 说明 |
|------|------|
| 数据类型 | 仅支持 complex64（`aclblasComplex`）；不支持 fp16/fp64/bf16 输入 |
| 输出 | 单复数标量（实部/虚部各 float） |
| 步长 | 支持任意非零整数步长（正/负）；`incx==0`/`incy==0` 返回 INVALID_VALUE |
| 向量长度 | `int n`，`n<0` 返回 INVALID_VALUE，`n==0` 返回 (0,0) |
| 核数 | 最多使用 16 个 AIV 核 |

## 四、特性交叉分析

| 特性 | 说明 |
|------|------|
| 多核并行 | 与 `cdotu`/`scasum` 同一 SIMT 多核归约范式，无冲突 |
| workspace 复用 | 复用 handle 默认 workspace（32MB），与同仓其他算子共享机制一致，无互斥 |
| 精度 | 与 §3.2 精度标准对齐；golden 用 double 精度累加，避免顺序累加误差误判 |
| 并发 | kernel 为纯读输入 + 写单标量，无跨算子共享状态，可安全并发 |

## 五、可维可测分析

### 5.1 精度标准 / 性能标准

**精度标准（对齐任务书 §3.2 标量容差）：**
- 输出为 COMPLEX64 标量，实部/虚部分别按 FLOAT32 标准验证
- `|actual - golden| ≤ atol + rtol × |golden|`，其中 `rtol = 2^-10`，`atol = 2^-16`
- `max_abs_error ≤ 1e-2` 或 `32 × ULP`
- Inf/NaN 特殊值按一致性判定（同 NaN 或同号 Inf）
- golden 采用 double 精度累加（近精确，与求和顺序无关）

**性能标准：**
- 目标：对齐任务书 §3.3 性能验收标准，接近 `aclblasScasum`/`aclblasCdotu` 同规模性能（SIMT 路径一致）
- 950PR 硬件限制：kernel launch 固定开销 ~17μs（硬件特性）
- 大规模（n≥1M）受 HBM 带宽限制

### 5.2 兼容性分析

参考 https://gitcode.com/cann/cann-ops-competitions/pull/1182 的评审口径：

| 兼容性项 | 说明 |
|----------|------|
| 接口兼容 | `aclblasCdotc` 接口声明与仓内 `aclblasCdotu` 同构（仅语义为共轭），对调用方透明 |
| 数据格式 | complex64 内存布局与仓内其他复数算子（cdotu/scasum/caxpy）一致，无格式差异 |
| 多代际 | arch35 专用实现位于 `blas/dot/arch35/`，不影响其他架构（arch22 等）路径 |
| 语义对齐 | n<=0/负步长/空指针等边界语义与 cuBLAS 对齐 |
| 同仓复用 | 复用 handle workspace、tiling 结构、SIMT 归约范式，无重复造轮子 |