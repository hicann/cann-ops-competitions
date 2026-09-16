# 需求背景（required）

为 [8 月社区任务 aclblasCsyr2（A2/A3）](https://www.hiascend.com/activities/task-center/details/afe2af29c47947cb93ccc227b6827522) 实现单精度复数对称秩 2 更新。算子使用 Ascend C，接入 `cann/ops-blas` 公共接口；开发者为 yuhui（GitCode：yucheery）。

计算公式为 `A += alpha*x*y^T + alpha*y*x^T`。它使用普通转置，不取共轭；只更新指定三角，对角线虚部正常参与计算。语义参考 [cuBLAS syr2](https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-syr2) 和 [Netlib ssyr2](https://www.netlib.org/blas/ssyr2.f) 的三角遍历规则。该任务新增 BLAS 接口，不涉及 TBE 或 ACLNN 算子迁移。

代码见 [ops-blas PR #455](https://gitcode.com/cann/ops-blas/merge_requests/455)，设计见 [PR #1578](https://gitcode.com/cann/cann-ops-competitions/merge_requests/1578)。以下代码路径均相对于 ops-blas 仓库。

# 需求分析（required）

## 接口与数据布局

公共声明位于 `include/cann_ops_blas.h`：

```cpp
aclblasStatus_t aclblasCsyr2(
    aclblasHandle_t handle, aclblasFillMode_t uplo, int n,
    const aclblasComplex* alpha, const aclblasComplex* x, int incx,
    const aclblasComplex* y, int incy, aclblasComplex* A, int lda);
```

| 参数 | 约束 |
| --- | --- |
| `alpha` | Host complex64 标量指针，不能为空 |
| `x / y` | Device complex64 向量；活动更新时至少分配 `1+(n-1)*abs(inc)` 个元素 |
| `A` | Device complex64 列主序矩阵，至少分配 `lda*n` 个元素；元素地址为 `row+col*lda`，原地更新 |
| `n / lda` | `n>=0`，`lda>=max(1,n)` |
| `incx / incy` | 任意非零 int，支持正负步长 |
| `uplo` | UPPER 更新 `row<=col`，LOWER 更新 `row>=col` |

负步长起点为 `-(n-1)*inc`，调用方传入分配区间的首地址。索引乘法先提升到 int64，避免 `INT_MIN` 取反或跨度计算溢出。不提供广播、批处理、重叠别名或 Device alpha 模式。

## 调用行为

1. 依次校验句柄、枚举、维数、步长、lda 和 alpha 指针。空句柄返回 `HANDLE_IS_NULLPTR`，非法枚举返回 `INVALID_ENUM`，其余非法参数返回 `INVALID_VALUE`，均使用 `ACLBLAS_STATUS_` 前缀。
2. 合法 `n=0` 或精确零 alpha 直接返回成功，不访问 x/y/A。活动更新再检查三个 Device 指针非空。
3. 将参数按值传入一个 kernel，在 handle 绑定的 stream 上异步执行；Host 不分配临时 Device 内存，也不在内部同步。调用方读取结果前同步该 stream。

未选三角、lda 填充、向量空隙和只读 x/y 保持不变。输入值域遵循 FP32；下文的幅值条件仅用于选择计算路径。

# 详细设计（required）

## Host 分派与核数

Host 保存 alpha 的原始位，以区分零、单位值和次正规数。基础核数为 `min(AIV核数, ceil(n/4))`；A 首地址非 32B 对齐时使用单核，避免核间部分存储块冲突。此处的 32/40 核均指 AIV 计算核，不是 CPU 核或 NPU 卡数。

| 路径 | Host 选择条件 | 工作划分 |
| --- | --- | --- |
| 小规模 UPPER | 单位 alpha、单位步长，n 为 256/320/384/448/512 | 通常每组四列；紧凑对齐 UPPER512 且基础预算为 40 核时使用 32 核 |
| LOWER1024 | 单位 alpha、单位步长，`lda%4=0`、基础核数为 4 的倍数、A 对齐 | 紧凑布局且基础预算为 40 核时使用 32 核，其他布局按实际核数处理 |
| UPPER2048 | 单位 alpha、单位步长、`lda=n`、A 对齐 | 使用基础核数；本文测试环境为 40 核，每核每轮至多三列 |
| 通用入口 | 其他合法输入 | 每组四列，按实际启动核数分配 |

专用入口在 Device 检查完整逻辑 x/y。向量计算要求单位 alpha、所有实虚分量有限且绝对值不超过 5；从 A 开始执行四次 Axpy 的路径还要求 x 的每个分量为非零正规数。检查不满足时依次尝试通用向量路径和软件高精度路径。分派不读取用例名称、随机种子或性能基线，回退始终使用实际启动核数。

## Kernel 计算与搬运

三类计算路径共用三角范围、64 位寻址和原地更新约定：

| 路径 | 计算与数据复用 |
| --- | --- |
| 专用向量 | 将 x/y 缓存为行向量及其 `[-imag,real]` 旋转向量，重复使用列系数；批量搬入有效 A 列段，向量计算后精确写回 |
| 通用向量 | 支持单位 alpha、n≤8192 及任意合法步长和 lda；每次最多处理 1024 个复数行，先形成四项乘积更新量，再加 A |
| 软件高精度 | 覆盖其余输入；使用整数实现必要的 binary64 运算，只在输出时舍入到 FP32。n≤8192 时缓存完整 x/y，更大 n 按最多 2048 个复数行分块 |

**UPPER512：** 32 核中，核 b 负责 `col=b+32*k`，每核 16 列。一次读入各列有效 A 段，按四个实数乘积项依次执行非融合 Axpy，最后写回；其他小规模 UPPER 将四项分成两对，各自求和后加 A。

**LOWER1024：** 32 核中，核 b 负责 `col=b+32*(8*g+k)`，每核四组、每组八列。两组 UB 缓冲交替使用，使下一组搬入与当前组计算重叠。最后一次 Axpy 分前四列、后四列执行，前半写出与后半计算重叠；两半写出均提交后发出回收事件；后继输入等待全部写回完成，再复用整组缓冲。

**UPPER2048：** 每核每轮最多处理三列，各列独立管理输入、计算与输出依赖。先计算更新量，再加 A，三组缓冲提高搬运与计算的重叠度。

通用路径按四列一组分配所有权：核 b 从 `4*b+4*blocks*r` 开始处理至多四列。对齐 A 下，组边界为 `4*lda*8` 字节，是 32B 的倍数。每个有效元素只由一个核更新，不使用原子操作或全矩阵中间结果。

所有 GM 搬运均以真实有效三角段为界。LOWER 的向量起点向下对齐时，前缀仅在 UB 中补零，写回前移除；不会因此访问未选三角。全零 x(j)/y(j) 列跳过更新，保留原字节；极小系数在部分专用标量路径中的处理计入数值误差界。

## 数值计算

独立 CPU 参考把每个 FP32 输入精确提升为 double，以以下分组计算并保留 double 结果到误差比较：

```text
tr = ar*yjr - ai*yji;   ti = ar*yji + ai*yjr;
ur = ar*xjr - ai*xji;   ui = ar*xji + ai*xjr;
real = A.real + ((xir*tr - xii*ti) + (yir*ur - yii*ui));
imag = A.imag + ((xir*ti + xii*tr) + (yir*ui + yii*ur));
```

参考实现禁用乘加收缩，不复用 kernel 的索引、打包或整数算术。软件高精度路径按相同分组执行 binary64 round-to-nearest、ties-to-even，最后转为 FP32。输入和输出均通过整数位搬运，支持次正规数；NaN、Inf、溢出和符号零显式处理。

软件运算只实现 FP32 解码、binary64 加减、binary64×FP32 乘法及 FP32 编码。中间幅值小于 `2^388`、非零量子不小于 `2^-447`，不触及 binary64 溢出或次正规范围。53×24 位乘积通过 64 位整数分段保留全部有效位，不依赖设备原生 FP64。

向量路径的误差推导以当前编译目标的非融合 Axpy 语义为前提。在单位 alpha、有限 `|x/y|≤5` 的条件下，每个分量只有四个幅值≤25 的乘积，与矩阵阶数无关。令 `e=2^-20`、`eta=2^-126`，当 FP64 参考分量 `|G|<1` 时：

| 计算顺序 | 绝对误差上界 |
| --- | --- |
| 从 A 开始四次 Axpy | `11.0625e + 2^-45 + 32eta < 2^-16` |
| 两对乘积分组后加 A | `12.0625e + 2^-45 + 64eta < 2^-16` |
| 先形成更新量再加 A | `14.0625e + 2^-45 + 64eta < 2^-16` |

较大 `|G|` 的相对误差、最大误差和次正规数处理见 `blas/syr2/arch22/csyr2_numerics.md`。这些界仅适用于通过检查的向量输入，其他输入由软件高精度路径处理。更新区的 NaN 比较分类，不要求 payload 一致；未更新区按字节保留。

## 内存与流水依赖

| 路径 | 显式 UB 地址上界（字节/核） |
| --- | ---: |
| 软件高精度 | 147456 |
| 通用向量 | 180224 |
| UPPER512 | 139520 |
| LOWER1024 | 192000 |
| UPPER2048 | 183808 |

上述布局均低于 192 KiB，生产接口的增量 GM workspace 为 0。UB 地址上界不等于进程设备内存占用。

流水通过 MTE2→V/S、V/S→MTE3 和输出→复用事件保证读写顺序；每个 bank/slot 单独管理依赖。输入检查的临时区与在途 A 搬运区不重叠，回退前完成旧操作，kernel 退出前执行完整流水同步。

## 硬件、代码组织与兼容性

目标架构为 arch22（Atlas A2/A3），开发验证环境为 A3 / Ascend910_9362 / CANN 9.1.0。以下实测数据来自该 A3 环境，不作为其他硬件的实测结果。

| 位置 | 职责 |
| --- | --- |
| `blas/syr2/arch22/csyr2_host.cpp`、`csyr2_tiling_data.h` | 参数检查、负步长起点、启动参数 |
| `csyr2_kernel.cpp` 与 `csyr2_*_kernel.h` | 分派、三角搬运及各计算路径 |
| `csyr2_small_upper_bounds.h`、`csyr2_vector_helpers.h` | 完整数据检查、索引及系数读取 |
| `csyr2_fp64.h` | 软件高精度算术 |
| `test/syr2/csyr2/` | 独立参考、测试用例、测试代码和运行脚本 |

新增 `aclblasCsyr2` 公共接口，不改变现有 Ssyr2 或其他 BLAS 接口。

# 可维可测分析

## 精度标准与测试覆盖

令 G 为未经 FP32 窄化的参考分量，误差为 `abs(double(actual)-G)`。按 [CANN 精度标准](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md) 分别判定实部、虚部：

- 混合误差：`error<=2^-16+2^-10*abs(G)`，两部分各自匹配率≥0.99。
- 最大误差：每个有限分量同时满足 `error<=max(0.01,32*ULP)`；ULP 采用 FP32 向正无穷的相邻间距，正 FLT_MAX 使用有限向下间距。
- 特殊值：NaN 比分类，Inf 比符号；有限参考的绝对值达到或超过 FP32 RN 溢出中点 `2^128-2^103` 时，才允许输出同号 Inf。
- 内存保护：逐字节检查未选三角、lda 填充、向量空隙、外部哨兵、只读 x/y 和跳过列。

测试包含任务附件 1000 个精度用例、200 个性能用例及 8 个补充 GTest，共 1208 项。补充用例覆盖双三角、正负步长、padding、地址对齐、零维和非法参数，以及巨大相消、NaN/Inf、次正规数、零列和计算路径回退。非法枚举用例 `TC_ED_151` 按任务书要求判定为 `ACLBLAS_STATUS_INVALID_ENUM`。

该 A3 环境完整执行 1208 项，均通过。测试入口同时检查执行数量、进程返回值和 GTest XML；构建依赖、SoC 设置及运行命令见 `test/syr2/csyr2/README.md`。

## 性能标准与实测结果

测量前检查宿主机、目标 NPU 及 A3 配对 DIE 的占用，测量期间持续采样。每个场景执行 1 次正确性检查、10 次预热和 100 次计时；从 msprof 记录中按开始时间排除前 11 次，使用其余全部 100 次内核耗时计算均值。ACL stream-event 时间单独统计，不与内核时长混用。

三个场景均采用 `alpha=(1,0)`、单位步长和 `lda=n`。每个场景独立测量三轮，下表取三轮均值的最大值：

| 场景 | AIV 核数 | 实测最大均值（μs） | 任务门槛（μs） |
| --- | ---: | ---: | ---: |
| UPPER / n=512 | 32 | 6.95244 | 7.54 |
| LOWER / n=1024 | 32 | 9.68539 | 9.72 |
| UPPER / n=2048 | 40 | 17.68424 | 19.05 |

三轮共 900 个计时样本，九个均值均满足对应门槛。算子接口说明见 `blas/syr2/arch22/README.csyr2.md`；完整误差依据见 `blas/syr2/arch22/csyr2_numerics.md`；测试代码、CSV 用例和复现入口统一位于 `test/syr2/csyr2/`。
