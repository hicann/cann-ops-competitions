# 需求背景（required）

## 需求来源

昇腾算子开源仓社区任务「算子实操工坊-杭州站-aclblasCgbmv算子开发(950)」。

- 验收通过后在昇腾算子开源仓提交 PR 合入：`gitcode.com/cann/ops-blas`，目录 `blas/gbmv/arch35/`（与实数 `aclblasSgbmv` 同族）；
- 测试代码合入 `test/gbmv/cgbmv/arch35/`；
- 接口声明放入 `include/cann_ops_blas.h`，禁止定义 950PR 私有平行接口；
- 适配硬件 Ascend 950PR，CANN 9.1.0，使用 hidevlab webIDE 算力（免费 100h）。

## 背景介绍

### Cgbmv算子实现优化

基于 cuBLAS `cublasCgbmv` 语义与 Netlib `cgbmv` 参考实现，使用 Ascend C 编程语言在昇腾 NPU（Ascend 950PR）上实现复数带状矩阵-向量乘算子；工程实现基于 ops-blas 仓内同族实数算子 `aclblasSgbmv`（Ascend C 版）扩展复数支持。

Cgbmv算子实现路径和相关API路径

Cgbmv算子实现路径为：ops-blas 仓 blas/gbmv/arch35/（与实数 `aclblasSgbmv` 同族目录，PR 合入目标）

Cgbmv算子接口声明路径为：ops-blas 仓 include/cann_ops_blas.h（本算子为新增声明，随实现一并提交）

Cgbmv算子复数类型定义路径为：ops-blas 仓 include/cann_ops_blas_common.h（`aclblasComplex`，实部/虚部各 float32）

Cgbmv算子对标参考实现：cuBLAS `cublasCgbmv`（https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-gbmv ）、Netlib `cgbmv`（https://www.netlib.org/blas/cgbmv.f ）

### Cgbmv算子实现现状分析

通过对 ops-blas 仓 `blas/gbmv/` 的功能分析，当前族内已有实数版 `aclblasSgbmv`（float32，trans=N/T），复数版 `aclblasCgbmv` 缺失。本算子在 Sgbmv 实现骨架基础上扩展 complex64 数据类型与 trans=C（共轭转置）模式。当前支持的能力如下：

| 参数 | 参数含义 | 数据类型 | 支持数据类型 | 约束 | 形状 |
| --- | --- | --- | --- | --- | --- |
| handle | 库上下文句柄，携带 stream | scalar | - | 指向已创建的有效句柄 | - |
| trans | 矩阵操作类型 | attr | int（枚举） | {ACLBLAS_OP_N, ACLBLAS_OP_T, ACLBLAS_OP_C} | - |
| m | 矩阵 A 的行数 | scalar | int | m ≥ 0；m=0 为合法 no-op | - |
| n | 矩阵 A 的列数 | scalar | int | n ≥ 0；n=0 为合法 no-op | - |
| kl | A 的下带宽 | scalar | int | 0 ≤ kl ≤ m-1 | - |
| ku | A 的上带宽 | scalar | int | 0 ≤ ku ≤ n-1 | - |
| alpha | 复数标量乘数 | scalar | complex64 | 指针非空 | - |
| A | 输入带状矩阵 | tensor | complex64 | lda ≥ kl+ku+1；带外三角区不引用 | lda×n（列主序带状存储） |
| x | 输入向量 | tensor | complex64 | incx ≠ 0（支持负步长） | trans=N 时 n 个元素，否则 m 个元素 |
| beta | 复数标量乘数 | scalar | complex64 | 指针非空；beta=(0,0) 时 y 不必是有效输入 | - |
| y | 输出向量（原地覆写） | tensor | complex64 | incy ≠ 0（支持负步长） | trans=N 时 m 个元素，否则 n 个元素 |

计算公式：y = alpha * op(A) * x + beta * y

### Cgbmv算子功能分析

Cgbmv算子功能：y = alpha * op(A) * x + beta * y，op(A) 按 trans 取值：ACLBLAS_OP_N → A、ACLBLAS_OP_T → A^T、ACLBLAS_OP_C → A^H（共轭转置）

输入：A（m×n 带状复数矩阵，kl 条下次对角线 + ku 条上次对角线，列主序带状存储）、x、y（读旧值）、alpha、beta、trans

输出：y（原地覆写）

支持数据类型：complex64（`aclblasComplex`，实部/虚部各 float32）

支持广播：不支持（A/x/y 为独立张量，仅支持 lda/incx/incy 描述的行跨步与向量步长）

### 接口定义

```cpp
/**
 * @brief  复数带状矩阵-向量乘：y = alpha * op(A) * x + beta * y
 * @param  handle [in]      ops-blas 库上下文句柄，携带 stream，Host 内存
 * @param  trans  [in]      矩阵操作类型：ACLBLAS_OP_N / ACLBLAS_OP_T / ACLBLAS_OP_C，Host 内存
 * @param  m      [in]      矩阵 A 的行数，m ≥ 0
 * @param  n      [in]      矩阵 A 的列数，n ≥ 0
 * @param  kl     [in]      A 的下带宽（主对角线以下的非零对角线数），0 ≤ kl ≤ m-1
 * @param  ku     [in]      A 的上带宽（主对角线以上的非零对角线数），0 ≤ ku ≤ n-1
 * @param  alpha  [in]      复数标量乘数指针，Host/Device 内存
 * @param  A      [in]      带状矩阵 A，Device 内存只读；列主序带状存储 A(ku+1+i-j, j)，带外三角区（左上 ku×ku、右下 kl×kl）不引用
 * @param  lda    [in]      A 的前导维度，lda ≥ kl+ku+1
 * @param  x      [in]      向量 x，Device 内存只读；trans=N 时 n 个元素、否则 m 个元素，按 incx 步长
 * @param  incx   [in]      x 相邻元素步长，incx ≠ 0（支持负步长反向遍历）
 * @param  beta   [in]      复数标量乘数指针，Host/Device 内存；beta=(0,0) 时 y 不必是有效输入
 * @param  y      [in,out]  向量 y，Device 内存，输入旧值、原地覆写；trans=N 时 m 个元素、否则 n 个元素，按 incy 步长
 * @param  incy   [in]      y 相邻元素步长，incy ≠ 0（支持负步长反向遍历）
 * @return aclblasStatus_t，语义与 cann_ops_blas_common.h 一致
 */
aclblasStatus_t aclblasCgbmv(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n, int kl, int ku,
    const aclblasComplex* alpha, const aclblasComplex* A, int lda,
    const aclblasComplex* x, int incx, const aclblasComplex* beta,
    aclblasComplex* y, int incy);
```

> D5 阶段把上面这段代码落到 `include/cann_ops_blas.h` 即可（include 路径以仓内实际结构为准，见 C4）。注释取任务书 §2.4 参数表口径：`kl/ku` 用**下带宽/上带宽**，不用任务书 §2.5 那套"下次对角线/上次对角线"说法（虽同源但易误读）。

参数序列与 cuBLAS `cublasCgbmv` 一一对应（handle 及参数顺序一致，维数参数为 int）。类型名以 ops-blas 仓 `include/cann_ops_blas.h` / `cann_ops_blas_common.h` 实际 typedef 为准（`aclblasStatus_t` / `aclblasHandle_t` / `aclblasOperation_t` / `aclblasComplex`）；本算子为新增声明，随实现一并提交，禁止定义 950PR 私有平行接口。

**返回值**：`aclblasStatus_t`——ACLBLAS_STATUS_SUCCESS / ACLBLAS_STATUS_HANDLE_IS_NULLPTR / ACLBLAS_STATUS_INVALID_ENUM / ACLBLAS_STATUS_INVALID_VALUE 等，语义与 `cann_ops_blas_common.h` 定义一致。

# 需求分析（required）

## 需求描述

使用 Ascend C 编程语言在 Ascend 950PR 上实现 `aclblasCgbmv`：

1. 计算 `y = alpha * op(A) * x + beta * y`，A/x/y/alpha/beta 均为单精度复数（complex64）；
2. 支持 `trans = N / T / C` 三模式，C 为共轭转置；
3. A 为 m×n 带状矩阵（kl 次对角线 + ku 超对角线），列主序带状格式存储；
4. 支持 lda / incx / incy 描述的行跨步与向量步长（含负步长），不支持其他非连续内存访问；
5. 参数合法性校验与 cuBLAS 对齐的 no-op / quick-return 语义；
6. 精度满足 COMPLEX64 开源标准（rtol=2^-10, atol=2^-16, matched≥0.99）；
7. 性能不高于标杆耗时（N/1024² ≈ 25.58us 等 5 档）。

## 需求拆解

1. 支持 complex64 数据类型：`aclblasComplex`（实部/虚部各 float32，以 `include/cann_ops_blas_common.h` 定义为准）
2. 支持 trans = N / T / C 三模式，C 模式取共轭
3. 支持带状存储（列主序带状 A(ku+1+i-j, j)，带外三角区不引用）
4. 支持 lda / incx / incy 步长（含负步长反向遍历）
5. 参数校验与 no-op / quick-return 语义（对齐 cublas）
6. y 原地覆写（先读旧值再计算，beta=(0,0) 时不读旧值）
7. 精度达标（rtol=2^-10, atol=2^-16, matched≥0.99）
8. 性能达标（5 档标杆，warmup + >50 次采样平均）

# 详细设计（required）

## 算子分析

### 数学公式

`y = alpha * op(A) * x + beta * y`

其中（0-based 索引，i∈[0,m), j∈[0,n)）：

- **trans = N**：y 含 m 个元素，x 含 n 个元素
  `y[i] = alpha * Σ_{j} A[i,j]·x[j] + beta·y[i]`，j 遍历带内窗口 `[max(0, i-ku), min(n-1, i+kl)]`
- **trans = T**：y 含 n 个元素，x 含 m 个元素
  `y[j] = alpha * Σ_{i} A[i,j]·x[i] + beta·y[j]`，i 遍历带内窗口 `[max(0, j-ku), min(m-1, j+kl)]`
- **trans = C**：同 T 但 A[i,j] 取**共轭**
  `y[j] = alpha * Σ_{i} conj(A[i,j])·x[i] + beta·y[j]`

**带状存储索引**：A(i,j)（1-based）存于 `A(ku+1+i-j, j)`；0-based 线性下标为 `ku + i - j + j*lda`（j 列内第 `ku+i-j` 行）。带外三角区（左上 ku×ku、右下 kl×kl）不被引用，实现不得读取。

### 支持数据类型

`aclblasComplex`（COMPLEX64，实部/虚部各 float32）。

### 支持形状

m×n 带状矩阵，kl 次对角线 + ku 超对角线。无广播；非连续 Tensor 不支持（仅 lda/incx/incy 描述的行跨步与向量步长）；dynamic shape 不要求（m/n/kl/ku 运行时入参）。

## 算子实现

### 实现方案

整体复用 ops-blas `blas/gbmv/arch35/` 下实数版 `aclblasSgbmv` 的工程框架（句柄式 BLAS 接口 + handle 绑定 stream 直调 NPU kernel），在其上做复数扩展。以下为设计要点，**具体 tiling 参数待 clone ops-blas 后按 Sgbmv 对齐微调**。

#### 3.2.1 host侧设计：

参数合法性校验（入口第一关，早于任何计算）：

| 检查 | 失败返回 |
| --- | --- |
| handle == nullptr | ACLBLAS_STATUS_HANDLE_IS_NULLPTR |
| trans ∉ {N, T, C} | ACLBLAS_STATUS_INVALID_ENUM |
| m < 0 或 n < 0 | ACLBLAS_STATUS_INVALID_VALUE |
| kl < 0 或 kl > m-1 | ACLBLAS_STATUS_INVALID_VALUE |
| ku < 0 或 ku > n-1 | ACLBLAS_STATUS_INVALID_VALUE |
| lda < kl + ku + 1 | ACLBLAS_STATUS_INVALID_VALUE |
| incx == 0 或 incy == 0 | ACLBLAS_STATUS_INVALID_VALUE |
| alpha / beta 指针 == nullptr | ACLBLAS_STATUS_INVALID_VALUE |
| m>0 且 n>0 时 A==nullptr / x==nullptr | ACLBLAS_STATUS_INVALID_VALUE |
| beta 非零且 m>0、n>0 时 y==nullptr | ACLBLAS_STATUS_INVALID_VALUE |

no-op / quick-return 语义（对齐 cuBLAS）：

- m = 0 或 n = 0 时为合法 no-op（返回 ACLBLAS_STATUS_SUCCESS，不执行计算，不引用 A/x/y）；
- alpha = (0,0) 且 beta = (1,0) 时 quick return；
- alpha = (0,0) 时仅执行 y = beta * y（不读 A/x）；
- beta = (0,0) 时不读 y 的旧值（y 不必是有效输入）。

负步长归一化：Host 侧将负步长转换为正步长 + 起始地址偏移，Kernel 侧仅处理正步长。当 incx < 0 时，x 的有效起始地址调整为 `x_base + (len_x - 1) * incx`，步长取 |incx|；incy 同理。Kernel 内无需负步长分支判断，简化实现。

tiling策略：

本算子为带状矩阵-向量乘，输出为向量 y（trans=N 时长度 m，否则长度 n），每个输出元素为带内窗口的复数内积。host 侧将输出向量按核数均分，单核内再按 UB 容量切分为 tile 流水处理；trans、m、n、kl、ku、lda、incx、incy 等参数通过 TilingData 传入 kernel 侧。

任务均分：coreNum 根据输出向量长度和块大小动态调整，确保每个核心处理的数据块数均匀。

批量搬运：单核内按 tileLength 循环 CopyIn→Compute→CopyOut，x 与 A 的带内窗口数据按 tile 批量搬入 UB，减少 GM 访问次数；尾块（不完整 tile）合并进计算流程，避免数据碎片。

##### 1. 分核策略：

优先使用满核的原则。

如果核间能均分，可视作无大小核区分，各核数据块一致；

如果核间不能均分，需要将余出的输出元素分配到前几个核上。

输出数据大小计算：通过 trans 确定输出向量长度 L（N 模式 L=m，T/C 模式 L=n），complex64 每元素 8 字节，计算输出数据总字节数。

UB内存大小和核心数量获取：通过平台信息获取 UB 内存大小和核心数量，并根据这些信息调整核心数量（参照仓内 Sgbmv 的 GetBlockNum 逻辑，8/4/2/1 阶梯，小规模输入减少核数以降低启动开销）。

##### 2. 数据分块和内存优化策略：

充分使用UB空间的原则。

需要考虑不同硬件的UB大小不同、是否开启double buffer、kernel侧API实现过程中是否需要临时数据的储存，综合考虑单核内切分的大小。

UB内存大小获取：通过 GetCoreMemSize 函数获取 UB 内存的大小，用于后续的数据切分计算。

Tile块计算：根据 UB 内存大小和预定义的 BLOCK_SIZE 及 BUFFER_NUM，计算出每个 Tile 块的数据数量；complex64 按实部/虚部两个 float32 处理。

数据切分：将输出向量按照计算出的 Tile 块大小进行切分，计算出每个 core 需要处理的数据块数量和最后一个 block 的剩余数据量；每个输出元素所需的 x 与 A 带内窗口数据随之分块搬入。

内存优化：开启 double buffer，TQue 队列深度与 BUFFER_NUM 保持一致（=2），使 CopyIn/Compute/CopyOut 流水线重叠；带状元素索引在 kernel 侧按 ku+i-j+j*lda 现场计算，无需额外索引 buffer；alpha/beta 为标量，直接从 GM 读取，不占用 UB。

设置切分参数：将计算出的切分参数（每个 core 的输出元素数、Tile 块大小、尾块大小等）设置到 CgbmvTilingData 对象中。

##### 3. tilingkey规划策略：

需要tilingkey的情况：需要感知host侧信息对kernel侧走不同分支。在host侧获取 trans 取值，如果 trans 为 ACLBLAS_OP_N 则 tilingkey 为 0（对输出元素 i 遍历带内 j 窗口做内积），trans 为 ACLBLAS_OP_T 则 tilingkey 为 1（对输出元素 j 遍历带内 i 窗口做内积），trans 为 ACLBLAS_OP_C 则 tilingkey 为 2（同 T 但取 A 的共轭）。

#### 3.2.2 kernel侧设计：

进行Init和Process两个阶段，其中Process包括数据搬入（CopyIn）、计算（Compute）、搬出（CopyOut）三个阶段。

1. Init 阶段通过 GlobalTensor<T>::SetGlobalBuffer 绑定 A/x/y/alpha/beta（alpha/beta 为标量指针，直接 __gm__ 取数）；y 为原地输出，Compute 阶段需先读 y 旧值再覆写（beta=(0,0) 时不读旧值）；x/y 按下标×步长（incx/incy，支持负步长反向遍历）寻址。
2. 复数乘加采用实部/虚部拆解实现：Ascend C 无原生复数向量指令，complex64 按 (real, imag) 两个 float32 处理，(a+bi)(c+di)=(ac-bd)+(ad+bc)i，使用 float32 向量指令（AscendC::Mul、AscendC::Add 等）完成计算；C 模式在取 A 元素时虚部取反实现共轭。BLAS 复数为 interleaved 存储（real/imag 交替）：M2 标量实现可直接访问实/虚部字段；M5 向量化实现需在 CopyIn 后显式去交错（deinterleave）为实部/虚部两个连续 LocalTensor，计算完成写回 y 时重新交错（interleave）为 BLAS 标准布局。
3. 根据不同的 tilingkey 执行不同的核函数分支（N/T/C），避免 kernel 内运行时反复判断 trans。
4. Compute 核心逻辑（以 trans=N、单核为例）：

```
for each output index i in [blockStart, blockEnd):
    acc_real = 0; acc_imag = 0
    for j in [max(0, i-ku), min(n-1, i+kl)]:         // 带内窗口
        a  = A[ku + i - j + j*lda]                   // 带状取值
        xv = x[j * incx]                             // 按 incx 步长
        acc_real += a.real*xv.real - a.imag*xv.imag  // 复数乘加
        acc_imag += a.real*xv.imag + a.imag*xv.real
    // y_new = alpha * acc + beta * y_old（全复数）
    tmp_real = alpha.real*acc_real - alpha.imag*acc_imag
    tmp_imag = alpha.real*acc_imag + alpha.imag*acc_real
    if beta != (0,0):
        yold = y[i * incy]
        tmp_real += beta.real*yold.real - beta.imag*yold.imag
        tmp_imag += beta.real*yold.imag + beta.imag*yold.real
    y[i * incy] = (tmp_real, tmp_imag)
```

5. T/C 模式将循环变量对调（对输出 j 遍历带内 i 窗口），C 模式在取 A[i,j] 时取共轭；x/y 长度按模式切换（N→x:n/y:m，T/C→x:m/y:n）。
6. 边界与负向检查全部在 host 侧完成，kernel 侧仅处理合法区间；带外三角区（左上 ku×ku、右下 kl×kl）索引天然不进入循环窗口，不会被引用。

### 性能优化设计（M5 阶段实施，正确性优先于性能）

**瓶颈定位**：GBMV 为典型访存受限算子——计算强度 ≈ (kl+ku+1)/(kl+ku+3) × 8FLOPs/8B ≈ 0.9 FLOPs/Byte，远低于 950PR 计算带宽平衡点，优化核心是提升访存带宽利用率。

**优化项（按优先级）**：

| 优先级 | 优化项 | 说明 |
| --- | --- | --- |
| P0 | 计算顺序重排为列块遍历 | 逐输出实现中 A 相邻元素步长为 (lda-1)×8B 斜跳；列块顺序下 A 列内元素步长 1 连续。按列块 j_block 批量加载 A 与对应 x，向受影响输出行区间累加贡献 |
| P0 | 复数向量化 | interleaved complex64 在 UB 内去交错拆为实部/虚部两个 float32 向量（SoA），复数乘 (a+bi)(c+di)=(ac-bd)+(ad+bc)i 用 4 次 Mul + 2 次 Add/Sub 向量指令完成 |
| P0 | y 的 UB 累加 | 本核输出切片的 y 累加驻留 UB，全部列块处理完后一次性写回 GM，y 的 GM 访存从每列块读写降为最终一次写回 |
| P1 | CopyIn 阶段完成去交错 | 在数据搬入 UB 时同步完成 interleaved→SoA 拆解，避免额外的内存往返 |
| P1 | 按计算量分核 | total_work = L×(kl+ku+1) 低于阈值时降核数，避免小规模多核启动开销（阈值对齐仓内 Sgbmv 框架，不私造） |
| P2 | kl=ku=0 对角特化 | 可选，小收益 |

**实施原则**：M2 阶段（NoTrans 正确版）使用朴素逐输出实现先过全部精度用例；M5 阶段实测性能不达标再逐项上表优化，每上一项回归一次精度。

### 关键设计决策说明

| 决策点 | 选择 | 理由 |
| --- | --- | --- |
| 复用 Sgbmv 框架 | 是 | 同族目录、接口序列一致、float→complex 最小改动 |
| 复数乘加方式 | 实部虚部拆解 | Ascend C 无原生复数向量指令；M2 标量循环保正确，M5 向量化提性能 |
| 带状索引 | kernel 内现场计算 | 开销小，免传索引 buffer |
| 双缓冲 | QUEUE_DEPTH=BUFFER_NUM=2 | 流水线重叠，规避 ReLU 模板曾有的 QUEUE_DEPTH=1 退化 |
| y 原地输出 | Compute 先读旧值 | 对齐 cublas 语义，beta=0 时不读 |
| alpha/beta 取值 | kernel 内从 GM 标量读取 | alpha/beta 为 Host/Device 内存（§2.4），Device 指针时 host 无法解引用，走 TilingData 传值会破坏异步语义 |
| T/C 模式实现 | 直接按列窗口计算，不做转置预处理 | T 模式固定 j 的 i 窗口在带状存储中本就列内连续，天然适配向量化；host 侧转置 Device 只读数据需 D2H/H2D 往返，破坏异步契约且引入双倍 A 访存 |
| 负步长处理 | Host 侧归一化为正步长 | Kernel 免负步长分支；起始地址 +(len-1)×incx 一次性换算 |

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Ascend 950PR | √ |

## 算子约束限制

- 参数合法性：m≥0、n≥0；0≤kl≤m-1、0≤ku≤n-1；lda≥kl+ku+1；incx≠0、incy≠0；alpha、beta 指针不可为 nullptr；非法参数返回 `ACLBLAS_STATUS_INVALID_VALUE`，非法枚举（trans）返回 `ACLBLAS_STATUS_INVALID_ENUM`
- 带状存储布局：列主序带状格式，元素 A(i,j)（1-based）存于 A(ku+1+i-j, j)；带外三角区（左上 ku×ku、右下 kl×kl）不引用，实现不得读取这些位置
- 非连续 Tensor 支持：不要求（仅支持 lda/incx/incy 描述的行跨步与向量步长，不支持其他非连续内存访问）
- broadcast 规则：不涉及，A/x/y 为独立张量，无广播
- dynamic shape 要求：不要求，m/n/kl/ku 为运行时入参
- 原地与视图语义：y 原地覆写，不返回视图；x/y 不支持与 A 重叠的原地混用
- 确定性计算要求：不要求
- 空 Tensor 与 0 维处理：m=0 或 n=0 为合法 no-op，返回成功且不执行计算；alpha=(0,0) 且 beta=(1,0) 时 quick return；alpha=(0,0) 时仅执行 y=beta*y；beta=(0,0) 时不读 y 旧值
- 异步执行：依赖 `aclblasSetStream` 绑定 stream；读回 Device 结果前须同步 stream

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | COMPLEX64（实部/虚部按 FLOAT32 分量）：rtol=2^-10 (9.77e-4)，atol=2^-16 (1.53e-5)，required_matched_ratio≥0.99，max_abs_error≤1e-2 或 32*ULP。逐元素通过：\|actual-golden\| ≤ atol + rtol×\|golden\| | 生态算子开源精度标准（experimental_standard.md） |
| 性能标准 | 标杆耗时（Avg time, us，需 warmup + >50 次采样平均）：case1 N/1024²/32,32→25.58；case2 N/2048²/64,64→34.25；case3 T/2048²→35.85；case4 C/2048²→35.88；case5 N/4096²/128,128→52.79 | 任务书 §3.3 |

**golden 生成**：由 cblas（Netlib `cgbmv` 复数实现）单标杆比对生成，随测试工程提供；输出向量 y 全量验证，实部、虚部分别比对。

**本地离线对拍策略**：开发期（无 NPU 环境）使用 Python 黄金参考实现 `Cgbmv_python_reference.py`（已验证 N/T/C 三模式误差全为 0）逐条对齐 CSV 用例布局，离线校验算法与带状索引正确性，再上云做精度/性能验收。

**测试用例覆盖**（与任务包 `cgbmv_test.csv` 1200 条用例分类对应）：

- 常规尺寸：m=n=128/256/512/1024/2048/4096（尺寸扫描 TC_SQ）
- 边界尺寸：m=1、n=1、m=0、n=0（no-op，TC_ED）
- 带宽边界：kl=0、ku=0（对角）、kl=m-1、ku=n-1（全带，TC_BW）
- 步长边界：incx/incy ∈ ±1/±2/±3 全组合（TC_INC，含负步长）
- 标量边界：alpha=0、beta=0、alpha=0 且 beta=1（quick-return）、纯虚数、大值（TC_AB）
- lda padding：lda > kl+ku+1 的合法场景（TC_LD）
- 填充与特殊值：全零、交替、极端值、Inf/NaN（TC_FL）
- 非方阵：fat（n≫m）/ thin（n≪m）（TC_RC）
- 性能用例：TC_PF 200 条（含任务书 §3.3 五档标杆对应 case）

## 兼容性分析

本算子为新增声明（`include/cann_ops_blas.h` 中 `aclblasCgbmv` 当前不存在），合入时随实现一并提交，供其他产品线共用，禁止定义 950PR 私有平行 API。新算子，不涉及与既有接口的兼容性冲突。

---

## 附录：开发里程碑（实现期路线，非验收交付件）

| 阶段 | 目标 | 验收关联 |
| --- | --- | --- |
| M0 | hidevlab 开 950PR 环境 → clone ops-blas → 阅读 Sgbmv 框架 → 部署 cgbmv_test.csv | 环境 §8 |
| M1 | 跑通 Sgbmv，理解 host/kernel 接线与 tiling 传参 | 框架对齐 |
| M2 | 复数 NoTrans 单核正确版（先过精度 L0/L1） | 精度 §3.2 |
| M3 | 加 T / C 模式 + incx/incy 步长 | 参数覆盖 §3.5 |
| M4 | 边界与负向用例（零维/空指针/非法步长/越界带宽） | 负向 §3.5 |
| M5 | 性能优化（列块遍历重排、复数 SoA 向量化、y 的 UB 累加、负步长归一化，按"性能优化设计"表逐项上、逐项回归精度） | 性能 §3.3 |
| M6 | 自测报告 + PR 合入 ops-blas | 交付件 §4/§5 |

> 实现状态：本设计文档为**设计阶段产物**，kernel 代码待在 hidevlab 云 IDE 中基于 Sgbmv 骨架开发，本地（4090 笔记本）仅用于算法验证与文档编写。
