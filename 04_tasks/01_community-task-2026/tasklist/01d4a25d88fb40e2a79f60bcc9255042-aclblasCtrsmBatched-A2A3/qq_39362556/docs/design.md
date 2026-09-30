# 需求背景

## 需求来源

[9月社区任务-aclblasCtrsmBatched算子开发(A2/A3)](https://www.hiascend.com/activities/task-center/details/01d4a25d88fb40e2a79f60bcc9255042?menu=trends)要求在 Atlas A2/A3 系列产品上实现单精度复数批量三角求解，接口和验收范围以该任务书为依据。目标代码仓库为 [ops-blas](https://gitcode.com/cann/ops-blas)。

## 背景介绍

`aclblasCtrsmBatched` 对 `batchCount` 个大小一致、相互独立的三角线性系统求解。矩阵使用列主序，`A` 和 `B` 是 Device 内存中的指针数组，结果写回各批次的 `B[i]`。`side`、`uplo`、`trans` 和 `diag` 的全部合法组合均需支持。

# 需求分析

## 算子功能

对于每个批次 `i`，令 `T_i = op(A[i])`，`X_i` 为写回 `B[i]` 的结果：

- `side=ACLBLAS_SIDE_LEFT`：`T_i X_i = alpha B_i`，`A[i]` 的阶数为 `m`。
- `side=ACLBLAS_SIDE_RIGHT`：`X_i T_i = alpha B_i`，`A[i]` 的阶数为 `n`。

`ACLBLAS_OP_N`、`ACLBLAS_OP_T`、`ACLBLAS_OP_C` 分别对应原矩阵、转置和共轭转置。`ACLBLAS_UNIT` 将对角元视为复数 `1+0i`，不读取对角存储。`m=0` 或 `n=0` 返回成功；`alpha=0+0i` 时将有效的 `B[i]` 元素置零，不读取 `A[i]` 的矩阵内容。只处理 `B[i]` 中的 `m×n` 有效元素，保留 `ldb` 引入的填充区域。

## 公开接口与范围

在 `include/cann_ops_blas.h` 声明任务书规定的 13 个参数接口：

```cpp
aclblasStatus_t aclblasCtrsmBatched(
    aclblasHandle_t handle, aclblasSideMode_t side, aclblasFillMode_t uplo,
    aclblasOperation_t trans, aclblasDiagType_t diag, int m, int n,
    const aclblasComplex* alpha, const aclblasComplex* const A[], int lda,
    aclblasComplex* const B[], int ldb, int batchCount);
```

支持 `COMPLEX64`、统一批次维度、列主序和 `lda/ldb` 填充。`m`、`n`、`batchCount` 在运行时确定。求解只使用 `A[i]` 指定三角部分的系数；`B[i]` 原地覆写，批次之间的 `B[i]` 不重叠。算子不检测非单位对角的奇异性。

# 详细设计

## 算子分析

设 `k=m`（LEFT）或 `k=n`（RIGHT）。求解顺序由 `T_i` 的有效三角方向决定：上三角从末端向起始端处理，下三角从起始端向末端处理。`trans=T/C` 时交换原矩阵的有效三角方向；`trans=C` 在读取元素时同时将虚部变号。RIGHT 路径沿 `B` 的列方向处理，求解顺序与相同 `T_i` 的 LEFT 路径相反。每个批次共享相同的控制参数，但从各自的指针数组元素读取矩阵地址。

UNIT 矩阵按独立右端逐步求解，保留 Netlib 对应 side/trans 的复数乘加顺序。每个独立右端由同一个 AIV 完成，已解变量按原参考顺序参与后续计算。RIGHT T/C 将连续的多个 B 行同时放入向量，LEFT N 及符合条件的逐行路径对一次更新中的多个位置使用向量指令。向量化只改变独立元素的并行安排，每个结果的依赖顺序保持不变。单位对角直接使用 `1+0i`，读取 A 时排除对角存储。

NON_UNIT 的 `k<128` 使用 AIV 标量求解，`k≥128` 将阶数划为连续的对角面板。每轮由 AIV 求解当前面板，再由 AIC 使用已解面板更新尚未求解的区域：

- LEFT：`B_tail ← B_tail − T_tail,panel × X_panel`。
- RIGHT：`B_tail ← B_tail − X_panel × T_panel,tail`。

分块路径在 `handle` 绑定的 stream 上启动一个 MIX Kernel，内部使用 AIC/AIV 同步组织面板与更新。不同批次以及同一批次的独立右端区域并行。分块路径在 B 规范化时乘 alpha；标量路径根据 side/trans 保留参考实现的 alpha 位置：LEFT N 在对应右端求解前处理，LEFT T/C 和 RIGHT N 在各位置累减已解项前处理，RIGHT T/C 保留未乘 alpha 的中间解并在其求解依赖完成后乘 alpha。`alpha=0+0i` 直接将 B 的有效元素置零。

## 算子实现

### Host 侧设计

Host 校验句柄、枚举、维度、前导维和指针数组。`handle=nullptr` 返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`；非法枚举、负维度、`batchCount<1` 或空 alpha 返回 `ACLBLAS_STATUS_INVALID_VALUE`。零维参数完成上述检查后返回成功。非零维要求 `lda ≥ max(1,k)`、`ldb ≥ max(1,m)` 和有效 B；alpha 非零时还要求有效 A。Host 通过 Device 到 Host 复制读取指针数组地址，逐批检查必要矩阵的空指针；复制失败返回 `ACLBLAS_STATUS_INTERNAL_ERROR`。矩阵计算下发到 `handle` 绑定的 stream，调用者读取结果前同步该 stream。性能采样记录 Kernel 任务时间，Host 地址检查、工作区准备和数据传输另行记录。

Host 根据路径获取 AIV 或 AIC 核心数量。AIV tiling 包含维度、前导维、批次、枚举和 alpha。基本调度在 LEFT 以一列 B 为任务，在 RIGHT 以连续八行 B 为任务，核心在任务组内顺序处理各行。`ldb` 为八的整数倍时按任务轮转分配；其他 `ldb` 由一个 AIV 完成同一 batch 的全部写出，不同 batch 并行。RIGHT T/C 的多行向量路径在 Kernel 内按所选行数重新组织任务，各 AIV 轮转处理互不重叠的 B 行组。

分块路径把列主序 B 视为其转置矩阵的行主序视图，交换内部 side、m/n 和 uplo，保留原始 lda/ldb 用于地址计算。阶数与独立右端维度补为八的整数倍，面板长度在 `k≤1024` 时为 16，`k>1024` 时为 32。满足阶数至少 128、补齐后右端数至少 128、`ldb%8=0` 时，两个 AIV 共同处理一个 AIC 对应的工作区域。其他场景由每个 AIV 独立处理一个 batch，所属 AIC 完成两个 batch 的尾部更新。批次数量不超过 AIC 数量的一半时，可继续划分独立右端；每片至少包含 64 个原始右端，普通分片边界按 16 个右端计算，最后一片保存剩余长度。

Host 使用 `MultiCoreMatmulTiling` 为 FP32、ND 格式的 A/B/C 生成 Cube tiling，由平台 API 计算 L1/L0 资源安排。工作区由 `EnsureDefaultWorkspace` 申请或复用，包含算子 tiling、Cube tiling、规范化 A、各右端分片的 B 与 GEMM 中间值，以及至少 16 MB 的系统工作区。大小计算使用宽整数并检查溢出；准备 tiling、初始化中间工作区后，在同一 stream 启动 MIX Kernel。核心循环处理未完成的 batch 或分片，单个输出区域只由其所属任务写入。

### Kernel 侧设计

Kernel 从指针数组取得当前批次 `A[i]`、`B[i]`。标量路径把当前独立右端放入 UB，LEFT 每个复数连续占两个 float，RIGHT 的分段 DMA 在 UB 中为每个复数保留八个 float 的间距。A 列使用连续 DMA，A 行按列主序的 lda 间隔分段读取，每段最多 256 个复数。UNIT 的读取范围只包含当前解依赖的三角元素，随后逐项累减复数乘积。

标量复数乘法使用 Berkeley SoftFloat 3e 的 Float32 FMA 完成对应单次舍入，来源版权及许可证保留在 `ctrsmbatched_softfloat.h`。有限输入且除数非零时，复数除法按参考 BLAS 的 Smith 分支比较除数两个分量的绝对值，计算分量比值及分母，再用 Float32 FMA 计算分子；零除数、NaN 和 Inf 分别处理浮点类别与符号。运算按参考实现的分支保留零乘数处理：LEFT N 跳过为复数零的已解变量，RIGHT 跳过为复数零的 A 系数。双分量 NaN 出现后，LEFT 可将后续解写为相同双 NaN；RIGHT 记录对应已解位置，只有其系数非零时才直接传播双 NaN，系数为零时继续原顺序求解。

RIGHT UNIT T/C 在 `m≥64`、`n≥64`、`ldb%8=0` 时使用 `CtrsmRightUnitRows`。每个任务处理同一 batch 的连续 B 行，并按三角方向逐列求解。当前列的已解项依次参与复数乘法和减法；A 系数按最多 256 项读取，UNIT 对角元不进入读取范围。B 列通过连续 DMA 读入，使用 GatherMask 分离实部和虚部，结果通过 Gather 恢复交错布局后写回。前若干已解列保存在 UB，其他已解列从 B 读取。所有依赖完成后按参考顺序乘 alpha。

向量复数乘法使用 `CtrsmVectorFma::BasicBlock` 实现 Graillat–Muller 的 Algorithm 6/9。ExactMult 和 TwoSum 保存乘积、加法的舍入误差，最终按 nearest-even 修正结果。RIGHT 路径先将前列按二的整数次幂归一化，两个 FMA 共享实部分量的高低位预处理。有效元素掩码同时限定原始值、归一化值、系数、乘积与求解状态的范围，保证所选元素满足中间运算不发生上溢或下溢的前提；未选中的元素按原始值执行完整标量复数运算。掩码与缓存列一起保存在 UB，特殊浮点输入和尾部元素均按有效长度处理。

阶数至少为 64 的 LEFT UNIT N，以及未进入多行路径的 RIGHT UNIT T/C，使用 64 个元素的向量更新。其余参数组合使用逐项标量求解。各路径使用相同的复数舍入和特殊值语义。

分块路径根据 side/trans 选择 A 的直接视图、复制、转置或共轭转换。转换后的 A/B 在全局工作区中按内核需要的布局保存；阶数补齐部分的 A 设置单位对角，其他补齐元素置零，B 的补齐区域置零。共轭复制同时受源数据缓冲区和符号掩码容量限制，逐段计算输入长度、行数和源目标偏移。完整对角面板一次搬入 UB 后，使用两次 GatherMask 分别选取实部和虚部；不足整块的面板按实际行数读取并填充其余位置。面板求解使用复数加法、乘法和对角除法，按有效三角方向完成当前面板。

尾部更新使用复数矩阵乘法。对于 `P=P_r+iP_i`、`Q=Q_r+iQ_i`，计算 `P_rQ_r−P_iQ_i` 和 `P_rQ_i+P_iQ_r`；四个 FP32 矩阵乘积通过 AIC 执行。关闭 HF32 模式，使用 FP32 累加。AIV 完成面板结果布局转换，并通过同步事件通知 AIC；AIC 更新尾部区域后通知 AIV 进入下一面板。规范化 A、B 及分片中间矩阵保存在独立工作区，原始 B 的有效区域在对应阶段写回。

向原始 B 写回时按 m/n 与 ldb 限定有效元素，保留列间填充。写出 tile 的数据量与实际 UB 容量比较，容纳完整输出时进行分段 DMA；其他情况由工作区中已计算的有效元素写回。零维由 Host 返回，`alpha=0+0i` 使用 AIV 置零路径。

### 资源与调度

逐行路径的 A 缓冲区至少容纳 `8k` 字节；UNIT 行读取还要求容纳 `32×min(k,256)` 字节。B 缓冲区在 LEFT 为 `8k` 字节，在 RIGHT 为 `32k` 字节，分配量补为 32 字节的整数倍。64 元素向量路径还要求 A 缓冲区容纳 `8×(ceil(k/64)×64+64)` 字节，并另分配 7296 字节的更新工作区。4096 阶 RIGHT 的逐行向量路径共分配 171648 字节。

RIGHT 多行路径根据总右端行数 `m×batchCount` 和启动 AIV 数选择行数。`m≥1024` 且总行数至少为 AIV 数的 512 倍时，每组处理 1024 行、缓存前 2 列；其他满足 `m≥512` 且总行数至少为 AIV 数的 256 倍的场景，每组处理 512 行、缓存前 12 列；其余场景每组处理 256 行、缓存前 34 列。三种布局的显式 UB 分配分别为 181024、177216、183648 字节，包含复数工作区、FMA、缓存、系数、Gather 偏移和掩码元数据，并为 Select 内部临时区保留容量，均在 192 KB 预算内。最后一组仅处理实际行数，写回保持列间填充。

分块 AIV 使用 192 KB 的 UB 预算。固定部分包括两份 `nb×nb` 的 FP32 A 面板和行暂存；随 `colTile` 增长的部分包括 B 面板、复数转换与负值暂存、累加区域、共轭掩码和 gather 偏移。`colTile` 按八个元素递减，直到基于实际 buffer 容量的总量满足预算。源转换 buffer 至少为 `max(64×64, 2×nb×colTile)` 个 float，共轭掩码至少为 `max(64×16, nb×colTile)` 个 float；每次复制必须分别遵守这两个上限。AIC 的 L1/L0 安排由平台 Matmul tiling API 产生，系统工作区与矩阵工作区分区保存。

全局工作区大小计算使用宽整数，超出可表示范围返回参数错误；申请失败返回 `EnsureDefaultWorkspace` 的状态，设备查询、tiling 生成或数据准备失败返回内部错误。工作区依附 handle，矩阵计算和后续消费者使用其绑定的 stream。

`batchCount=1`、尺寸为 1、非对齐长度、`lda/ldb` 含填充、最后一个面板不足整块、`m=0`、`n=0` 和 `alpha=0+0i` 均进入对应的明确分支。`B[i]` 地址按批次隔离，所有启动均使用 `handle` 中的 stream。

## 支持硬件

目标为 Atlas A2/A3 系列产品，使用 `arch22` 目录。性能验收设备按任务书为 Atlas 800T A2 (910B3)；产品支持说明同时列出 Atlas 800I A2/A3。

## 算子约束限制

只支持 `COMPLEX64`、列主序、统一批次尺寸及任务书规定的枚举组合。`A` 和 `B` 为 Device 指针数组；`alpha` 为 Host 指针。每批 `B[i]` 原地覆写，批次输出不得重叠。非单位对角必须非零，算子不执行奇异性检测。结果的浮点累加顺序不保证逐位一致。

# 可维可测分析

## 精度标准与验证

以 Netlib BLAS `cblas_ctrsm` 逐批生成 Golden，比较写回后各 `B[i]` 的完整有效区域。COMPLEX64 的实部和虚部分别按 FLOAT32 混合容差统计：`rtol=atol=2^-13`、`matched_ratio≥0.99`。最大绝对误差只统计 actual 与 Golden 均有限的元素，并与最大误差位置 Golden 对应的 `max(0.01,32×ULP(g_low))` 比较。双 NaN、同号 Inf 按官方标准匹配，其余非有限差异计入未匹配元素。

精度日志与 XML 保存 case 名称、错误分量、匹配比例、最大有限误差、对应 Golden 和失败位置；参数及随机种子保留在原始 CSV。输入覆盖 LEFT/RIGHT、UPPER/LOWER、N/T/C、UNIT/NON_UNIT、不同 alpha、填充维度、批次边界、零维、空指针和非法参数。官方脚本、CSV 与补充诊断分别保存，完整原始输出和逐 case 状态同时保留。参数接口及边界语义存在官方来源冲突时，保留相应原始失败状态，等待官方澄清后重新执行确认的流程。

向量 FMA 的补充诊断与 `std::fma` 逐位比较，并覆盖同一被乘数的连续复用、抵消和 halfway 舍入。公开接口补充诊断使用真实 NPU 输出与 Netlib 逐位比较，覆盖三种行数布局、八行尾部、二的不同整数次幂缩放、特殊浮点输入和上下三角方向。性能场景可通过 `CTRSM_SUPPLEMENTARY_PERF_ACCURACY=1` 单独启用 Golden 比较，日志和 XML 明确标记为补充精度验证，性能采样分别执行。

## 性能标准与验证

按照任务书在 Atlas 800T A2 (910B3) 上测量以下 COMPLEX64 场景的平均单次耗时，单位为微秒：

1. `batchCount=64, m=n=256, LEFT, LOWER, N, NON_UNIT`：不高于 `895.1`。
2. `batchCount=128, m=384, n=512, LEFT, UPPER, C, NON_UNIT`：不高于 `5551`。
3. `batchCount=32, m=n=1024, RIGHT, LOWER, T, UNIT`：不高于 `13884`。
4. `batchCount=8, m=n=2048, LEFT, UPPER, N, NON_UNIT`：不高于 `21816`。
5. `batchCount=2, m=n=4096, RIGHT, UPPER, C, UNIT`：不高于 `40969`。

按官方流程执行 warmup、有效采样、msprof 采集和平均值比较，保留逐次计时、Profiler 输出、完整命令与环境版本。CPU 域调试和仿真用于核函数功能检查及性能问题分析；正式性能结论由规定设备上的实测数据给出。

## 兼容性分析

公开声明位于 `include/cann_ops_blas.h`。Host、AIV Kernel、向量 FMA、RIGHT 多行求解、SoftFloat 运算和分块调用入口位于 `blas/trsmbatched/arch22/`；MIX Kernel 的 AIC/AIV 计算、面板求解、矩阵转换和 tiling 数据使用 `experimental/aclblasCtrsmBatched2/` 中的实现。两处实现目录共同参与构建。`blas/trsmbatched/README.md` 说明产品、参数、计算路径和调用方式；CSV 测试及精度比较位于 `test/trsmbatched/ctrsmbatched/arch22/`，公开 API 数值诊断位于其上层 `diagnostics/`。
