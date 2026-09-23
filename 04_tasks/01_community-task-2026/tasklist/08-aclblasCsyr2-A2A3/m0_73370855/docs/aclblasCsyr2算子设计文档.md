# 需求背景（required）

## 需求来源

昇腾 8 月社区任务：`aclblasCsyr2` 算子开发（A2/A3），任务书编号 aclblasCsyr2_AtlasA2A3_task_doc.md。

## 背景介绍

### aclblasCsyr2 算子实现

基于 ops-blas 开源仓（https://gitcode.com/cann/ops-blas ）使用 Ascend C 编程语言为 Atlas A2/A3（arch22）实现 BLAS Level-2 单精度复数对称秩-2 更新算子 `aclblasCsyr2`：

```
A = alpha * x * y^T + alpha * y * x^T + A
```

其中 alpha 为单精度复数标量，x、y 为 n 元素复数向量，A 为 n×n **对称（symmetric，普通转置不共轭）** 复数矩阵，列主序存储，仅 `uplo` 指定的上/下三角被引用与更新。与 Hermitian 族（her2 等）不同：本算子不含共轭运算，对角元素虚部无特殊假定。

对标接口：cuBLAS `cublasCsyr2`（语义参考 Netlib `ssyr2`，https://www.netlib.org/blas/ssyr2.f ；cblas/netlib 无复数 syr2 实现）。

实现现状分析：

- ops-blas 仓 `blas/syr2/arch22/` 已存在 **ssyr2（float32 实数版）** 实现（`ssyr2_host.cpp` / `ssyr2_kernel.cpp`），AIV 向量内核、C220 intrinsic 实现，但不处理 incx/incy/lda、无参数校验、无 alpha=0 quick return；
- `blas/axpy/arch22/` 存在 **caxpy（complex64 版）** 实现（`caxpy_host.cpp` / `caxpy_kernel.cpp`），提供了复数交织存储的实部/虚部平面拆解处理范例；
- `include/cann_ops_blas.h` 中当前**无** `aclblasCsyr2` 声明，本任务需按任务书 §2.3 签名新增声明（与 950PR 版任务共用同一声明，禁止平行 API）；
- `aclblasComplex`（`{float real; float imag;}`）定义于 `include/cann_ops_blas_common.h`（68-71 行）。

### aclblasCsyr2 算子功能分析

| 参数 | 参数含义 | 输入/输出 | 数据类型 | 支持数据类型 | 约束 | 形状 |
| --- | --- | --- | --- | --- | --- | --- |
| handle | 上下文句柄，携带 stream | 输入 | scalar | - | 有效句柄，否则返回 HANDLE_IS_NULLPTR | - |
| uplo | 指定 A 的存储三角：ACLBLAS_UPPER(121)/ACLBLAS_LOWER(122) | 输入 | attr | int（枚举） | 非法枚举返回 INVALID_ENUM | - |
| n | 矩阵阶数，向量 x/y 元素个数 | 输入 | scalar | int | n ≥ 0；n=0 为合法 no-op | - |
| alpha | 复数标量乘数指针（Host 内存），alpha=(0,0) 时 quick return | 输入 | scalar | COMPLEX64 | 非空 | - |
| x | 复数向量（Device 内存，只读） | 输入 | tensor | COMPLEX64 | n>0 且 alpha≠(0,0) 时非空 | 逻辑 [n]，物理 1+(n-1)·\|incx\| |
| incx | x 相邻元素步长，支持负步长 | 输入 | scalar | int | ≠ 0 | - |
| y | 复数向量（Device 内存，只读） | 输入 | tensor | COMPLEX64 | n>0 且 alpha≠(0,0) 时非空 | 逻辑 [n]，物理 1+(n-1)·\|incy\| |
| incy | y 相邻元素步长，支持负步长 | 输入 | scalar | int | ≠ 0 | - |
| A | 输入/输出对称矩阵（Device 内存，原地更新，列主序） | 输出 | tensor | COMPLEX64 | n>0 且 alpha≠(0,0) 时非空 | lda×n（逻辑 n×n） |
| lda | A 的前导维度 | 输入 | scalar | int | lda ≥ max(1, n) | - |

计算公式（列主序，仅更新 uplo 三角内的 (i, j) 元素，0-based）：

```
A[i + j*lda] = alpha * x[i] * y[j] + alpha * y[i] * x[j] + A[i + j*lda]
```

LOWER：更新 i ≥ j 区域；UPPER：更新 i ≤ j 区域；另一三角不读取、不写入。

复数乘法按 `(a+bi)(c+di) = (ac−bd) + (ad+bc)i` 展开为实部/虚部分别计算，运算次序与 Netlib `ssyr2` 参考实现一致。

支持数据类型：COMPLEX64（实部/虚部各 float32）。

支持正/负步长：负步长按 Netlib 约定从反向起点遍历（列主序 0-based 下 kx = -(n-1)·incx 为逻辑元素 0 的物理偏移；调用方传入的 x/y 基址即首访问元素地址，`ptr[i·inc]` 即逻辑元素 i）。

# 需求分析（required）

## 需求描述

使用 Ascend C 编程语言为 Atlas A2/A3（arch22）实现 `aclblasCsyr2`，支持 complex64 数据类型、UPPER/LOWER 双三角、正/负步长、lda padding、边界与负向输入语义，满足生态算子开源精度标准与任务书性能标杆，并完成 CSV 驱动 GTest 自测工程。

## 需求拆解

1. 功能语义对齐 cuBLAS `cublasCsyr2`：`A = alpha·x·y^T + alpha·y·x^T + A`，对称不共轭，仅更新 uplo 三角；
2. 返回码语义对齐任务书 §2.4：handle 为空 → `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`；uplo 非法 → `ACLBLAS_STATUS_INVALID_ENUM`；n<0 / incx=0 / incy=0 / lda<max(1,n) / alpha 为空 → `ACLBLAS_STATUS_INVALID_VALUE`；n>0 且 alpha≠(0,0) 时 x/y/A 为空 → `ACLBLAS_STATUS_INVALID_VALUE`；
3. quick return：n=0 或 alpha=(0,0) 时直接返回 `ACLBLAS_STATUS_SUCCESS`，不更新 A；
4. 支持正/负步长（incx/incy 可正可负）与 lda ≥ n 的 padding 场景；
5. 精度满足生态算子开源精度标准：实部/虚部分别按 FLOAT32 判定（rtol=2^-10、atol=2^-16、matched_ratio≥0.99、max_abs_error≤1e-2 或 32×ULP），golden 由测试工程按 Netlib ssyr2 语义自实现复数版生成；
6. 性能满足任务书 §3.3 标杆（Atlas 800I A2 / 910B3，warmup 后有效采样 >50 次取平均）：n=512（UPPER）≤ 7.54us、n=1024（LOWER）≤ 9.72us、n=2048（UPPER）≤ 19.05us；
7. 交付自测用例及测试代码：覆盖任务方提供的 1200 条用例（1000 精度 + 200 性能），CSV 驱动 GTest，接入 `test/syr2/csyr2/arch22/` 目录，可复现。

# 详细设计（required）

## 算子分析

### 数学公式

对每个 uplo 三角内的元素 (i, j)（复数 alpha = ar+ai·i，x[i] = xr+xi·i，y[j] = yr+yi·i，列主序地址 i + j·lda）：

```
// 行标量（每个 i 计算一次）
cx_i = alpha · x[i] ：  cxr = ar·xr − ai·xi ，  cxi = ar·xi + ai·xr
cy_i = alpha · y[i] ：  cyr = ar·yr − ai·yi ，  cyi = ar·yi + ai·yr

// 行内向量更新（对三角内每个 j）
A_r[i,j] += cxr·yr_j − cxi·yi_j + cyr·xr_j − cyi·xi_j
A_i[i,j] += cxr·yi_j + cxi·yr_j + cyr·xi_j + cyi·xr_j
```

### 支持数据类型

COMPLEX64（`aclblasComplex`，实部/虚部各 float32，交织存储；32B 对齐单元 = 4 个复数）。

### 支持形状

- x/y：逻辑一维 [n]（n ≥ 0），物理长度 1+(n-1)·|inc|，inc ∈ ±1/±2/±3 等任意非零整数；
- A：列主序 lda×n（lda ≥ max(1, n)），仅 uplo 三角被读写；
- 不涉及广播；不支持维度 ≥ 3 的高维运算。

## 算子实现

### 实现方案

文件布局（对齐仓内 syr2 同族归档规范）：

```
blas/syr2/arch22/csyr2_host.cpp      // 接口实现：参数校验、quick return、tiling、kernel 直调
blas/syr2/arch22/csyr2_kernel.cpp    // Ascend C 向量内核
include/cann_ops_blas.h              // 新增 aclblasCsyr2 声明（任务书 §2.3 签名）
test/syr2/csyr2/arch22/              // CSV 驱动 GTest 测试工程 + 用例
```

#### 3.2.1 host 侧设计：

**参数校验与 quick return**（严格对齐任务书 §2.4 异常行为列，先于一切设备操作）：

1. handle 为空 → `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`；
2. uplo ∉ {ACLBLAS_UPPER(121), ACLBLAS_LOWER(122)} → `ACLBLAS_STATUS_INVALID_ENUM`；
3. n < 0 → `ACLBLAS_STATUS_INVALID_VALUE`；n == 0 → 返回 SUCCESS（no-op）；
4. alpha 为空 → `ACLBLAS_STATUS_INVALID_VALUE`；`*alpha == (0,0)` → 返回 SUCCESS（quick return，不更新 A）；
5. x/y/A 为空、incx == 0、incy == 0、lda < max(1, n) → `ACLBLAS_STATUS_INVALID_VALUE`。

**tiling 策略**：所有标量（uplo、n、lda、alphaReal、alphaImag、coreNum）直接以**内核形参**传递，不分配 tiling 设备内存、无 aclrtMemcpy 拷贝——小 shape 下消除 host 侧额外开销。alpha 由 host 解引用 `*alpha` 拆成实/虚两个 float 传入。按步长形态分两条执行路径：

- **连续路径（incx == 1 && incy == 1）**：x/y 设备指针直传主内核，零额外开销；
- **步长路径（其余情况，含负步长）**：先启动一个轻量 gather 内核，将 x、y 按带符号步长压实为连续复数向量，写入 handle 的**持久 workspace**（2·n·8B，复用句柄内存，接口内不分配/释放，无 host 同步），再启动主内核。步长语义在 gather 内核一处收口（负步长按 Netlib 约定锚定最低地址元素升序搬运），主内核逻辑统一。

接口全程**异步**：不做流同步，workspace 复用句柄内存不在接口内释放，满足任务书 §2.5 异步语义。

##### 1. 分核策略：

优先使用满核的原则。arch22 平台向量核数默认 40（对齐仓内 syr2/axpy 现有实现）：

- A 按**列轮转（round-robin）分核**：核 k 处理列 j = k, k+coreNum, k+2·coreNum…；
- 每列被且仅被一个核处理 → 列内更新天然无竞争，**无需 atomicAdd**（对比仓内 ssyr2 的"两遍 + 原子加"方案，A 只读写一次）；
- 三角形的列长度各异（LOWER 列 j 长 n−j，UPPER 列 j 长 j+1），轮转分核使各核累计工作量天然均衡，无需额外负载均衡逻辑。

##### 2. 数据分块和内存优化策略：

充分使用 UB 空间的原则（arch22 每核约 192KB 可用）：

- 复数保持**交织存储**处理，不拆实/虚平面：复数乘法用"成对交换 vgather + (−1,+1) 符号向量"实现跨通道配对，每项更新（A += x[i]·temp1）仅需 4 条矢量指令（2×Muls + 1×Mul + 2×Add）；
- 列数据按 2048 复数（16KB float）为一块流水处理；n ≤ 2048 时 x/y 全量驻留 UB，零重复搬运；
- 每列的标量 x[j]、y[j] 按组（512 列/组）批量搬运：DataCopyPad 逐元素写入 32B 对齐槽位，再经 vgather 压实为连续向量，随后 temp1 = alpha·y[j]、temp2 = alpha·x[j] 在 UB 内用矢量指令整组完成；
- 搬入搬出全部走 DataCopyPad（GM 侧支持任意 4B 对齐偏移），三角边界的非对齐段由 DataCopyPad 天然承载，保证未引用三角不被写入。

##### 3. 路径选择策略：

host 侧依 incx/incy 选择是否先跑 gather 预内核；两个内核按 stream 顺序串行，无 host 介入。主内核不感知步长，输入总是连续复数向量。

#### 3.2.2 kernel 侧设计：

AIV 向量内核（`__DAV_C220_VEC__`），Init + Process（CopyIn → Compute → CopyOut）三段式。主内核按 n 分三档处理（兼顾小 shape 开销与大 shape 带宽）：

1. **驻留路径（n ≤ 2048）**：x/y 的旋转副本（xr/yr）与预交换副本（xss/yss）及预乘 alpha 后的 p/q 共 6 个向量全量驻留 UB，A 列段经 4 缓冲流水进出；每列标量直接从驻留的 p/q 提取（零 GM 重复流量）；LOWER 利用核数 40 是 4 的倍数（列偏移天然 32B 对齐）免反转处理；
2. **bigRes 路径（2048 < n ≤ 4096）**：x/xss/y/yss 驻留 + A 三缓冲 + 1024 复数分块，列标量经 UB 内 gather 批量预取（`csyr2_prepare_temps`），复数乘法同样走 4-AXPY 形式；
3. **流式回退（n > 4096）**：UB 放不下 x/y 时按列段流式搬运，保证任意大规模可用（官方性能用例最大 n=4096，不触发）。

公共流程：初始化构建 (−1,+1) 符号向量与 vgather 偏移表（标量播种 + DataCopy/Adds 倍增扩展，避免逐元素初始化拖慢小 shape）；驻留加载时预乘 alpha：p = alpha·x、q = alpha·y（含旋转副本）在 UB 内向量化生成，热循环每列直接 GetValue 取标量，无逐列复数乘链；列内三角段（UPPER：行 [0, j]；LOWER：行 [j, n−1]）按 4 条 AXPY 就地累加（`A += x·yre; A += xr·yim; A += y·xre; A += yr·xim`，xr/yr 为预旋转副本），每项更新 4 条矢量指令；单次遍历同时完成两项外积，A 每元素只读一次、写一次，无原子操作。AXPY 分组累加的舍入次序与 golden 的分组不同，属于生态精度标准（rtol=2⁻¹⁰/atol=2⁻¹⁶/matched_ratio≥0.99）允许的合法浮点求值顺序，实测全量用例 matchedRatio ≥ 0.99997。

**gather 内核（步长路径）**：AIV 多核均分逻辑下标，逐元素 `ws[i] = src[i·inc]`（带符号索引，负步长锚定最低地址元素后升序搬运），全部经 MTE2/MTE3 DataCopyPad（AIV 标量 GM 访问不可靠，已规避）。

Host 侧不做流同步语义对齐任务书 §2.5（异步执行，读回前由调用方同步）。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas 800I/T A2 | √ |
| Atlas A3 系列 | √ |

## 算子约束限制

- 不支持广播（BLAS Level-2 语义，x/y/A 为独立操作数）；
- dilation 类参数不涉及；不支持维度 ≥ 3 的高维运算；
- 仅 uplo 指定三角被引用/更新，对称性（A = Aᵀ，不共轭）由调用方保证，对角元素虚部不做特殊处理；
- n = 0 或 alpha = (0,0) 为合法 quick return；
- Inf/NaN 输入按浮点语义传递（不报错），精度比对按特殊值规则匹配；
- A 原地覆写，不返回视图；host 侧不做流同步。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述(不涉及说明原因) | 标准来源 |
| --- | --- | --- |
| 精度标准 | 实部/虚部分别按 FLOAT32 生态精度标准：rtol=2^-10、atol=2^-16、required_matched_ratio≥0.99、max_abs_error≤1e-2 或 32×ULP；golden 由测试工程按 Netlib ssyr2 语义自实现复数参考（对称不共轭、负步长起点、quick return），输出 A 按 uplo 三角逐元素比对 | 任务书 §3.2 / 生态算子开源精度标准 |
| 性能标准 | 任务书 §3.3 标杆（910B3，Avg，warmup 后有效采样 >50 次）：n=512 UPPER ≤ 7.54us；n=1024 LOWER ≤ 9.72us；n=2048 UPPER ≤ 19.05us；其余 TC_PF 用例对照 gpu_baseline.csv 逐条判定 | 任务书 §3.3 / 测试指导 |
| 内存标准 | 任务书 §3.4 明确"不涉及"；设计上单用例 host 侧 ≤ 512MB，设备侧仅 2·n·8B workspace（步长路径） | 任务书 §3.4 / 测试指导 |

## 兼容性分析

- `include/cann_ops_blas.h` 新增 `aclblasCsyr2` 声明（任务书 §2.3 规定签名，与 950PR 版任务共用同一声明），不修改任何既有接口，无 ABI 变更；
- 实现放置于 `blas/syr2/arch22/`，与既有 ssyr2 并存（ssyr2 文件不改动），不影响其他算子与平台行为；
- 测试工程接入 `test/syr2/csyr2/arch22/`，复用仓内 test 框架（CSV 驱动 GTest），构建方式与既有算子测试一致；
- 参数序列与 cuBLAS `cublasCsyr2` 一一对应（handle 及 int 维数参数顺序一致），无需额外映射。
