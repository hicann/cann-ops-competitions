# aclblasCgeru（A2/A3）算子设计

状态：设计评审稿 v1.0（2026-09-14）。报名已通过；本稿申请设计评审，尚未取得 NPU 编译、精度或性能验证结果。

提交账号：gcw_rBObG5J6。目标任务：广州实操工坊 aclblasCgeru（A2/A3）。

## 需求背景（required）

### 需求来源

- 任务：算子实操工坊-广州-aclblasCgeru算子开发(A2/A3)。
- [任务页面](https://www.hiascend.com/activities/task-center/details/ea91394a5da445609913b84469060101?menu=guide)，截止日期 2026-10-10。
- 任务书：任务页面“查看任务书”中的 `aclblasCgeru_Atlas800IA3_task_doc.md`。
- [官方设计模板](https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md)。
- [任务引导](https://gitcode.com/org/cann/discussions/283#tid-5b6e305a9b8440b2a3a78550dab541ca)：目前目录编号为 04，任务页引用的编号仍为 03。

### 背景介绍与实现现状

实现 BLAS Level 2 单精度复数无共轭秩 1 更新，为列主序矩阵提供原地更新接口。

调研基于 ops-blas master 提交 `7eae2328a65753bf55cffc489253eb434ea3317e`。
该版本 `include/cann_ops_blas.h` 已有 `aclblasCgerc`，没有 `aclblasCgeru`。
`aclblasComplex` 定义为 `float real; float imag;`，沿用仓库类型，不另行定义公开复数类型。

可参考 `blas/ger/arch22/sger_*` 的结构体传参和 kernel 直调方式；
`blas/gerc/arch22/cgerc_*` 提供复数拆分参考，但其 host 路径没有将 incx、incy、lda 传入 kernel，
且存在每次调用分配、拷贝、同步的开销，不能直接改名作为本任务实现。

## 需求分析（required）

### 需求描述

```cpp
aclblasStatus_t aclblasCgeru(
    aclblasHandle_t handle, int m, int n, const aclblasComplex* alpha,
    const aclblasComplex* x, int incx, const aclblasComplex* y, int incy,
    aclblasComplex* A, int lda);
```

接口声明加入 `include/cann_ops_blas.h`，与其他产品线共用。实现落在
`blas/ger/arch22/`，测试落在 `test/ger/cgeru/arch22/`。
不采用 ACLNN 自定义算子工程或新增产品私有 API。

| 对象 | 需求 |
| --- | --- |
| 数据类型 | complex64，实部和虚部均为 float32 |
| 计算 | `A[i+j*lda] += alpha * x[kx+i*incx] * y[ky+j*incy]` |
| 存储 | A 列主序，物理长度 lda×n，仅更新每列前 m 个元素 |
| 步长 | incx、incy 为非零 int，支持正数和负数 |
| 标量 | alpha 指向 Host 内存；x、y、A 为 Device 内存 |
| 执行 | 使用 handle 绑定的 stream 异步下发；调用方同步后读回 |
| 空计算 | 合法零维或 alpha=(0,0) 时返回成功，不启动 kernel、不写 A |
| 产品 | 目标支持 Atlas A2/A3（arch22）；性能验收为 910B3 |
| 软件 | 任务验收要求 CANN 9.1.0 |

### 需求拆解

1. 完整实现 host 参数校验及 no-op 分支。
2. 实现列主序复数更新，明确 y 不取共轭。
3. 支持负步长、任意合法 lda、尾块与非对齐地址。
4. 覆盖原包 1000 条精度和 200 条性能用例，补充接口与内存边界测试。
5. 使用 Netlib CBLAS golden，分别核验实部和虚部；另行校验 padding 和输入未被修改。
6. 通过设备计时采集有效采样大于 50 次的均值，达到四个指定性能阈值。

## 详细设计（required）

### 算子分析

对每个列 j，先计算 `t = alpha * y[j]`，再执行 `A[:,j] += x * t`。

```text
tr = alpha.real * yr - alpha.imag * yi
ti = alpha.real * yi + alpha.imag * yr
dr = xr * tr - xi * ti
di = xr * ti + xi * tr
out.real = A.real + dr
out.imag = A.imag + di
```

先计算复数乘积的实部和虚部，再分别加上 A 的旧值。实现保持上述阶段次序，
避免将实部重排成 `(A.real + xr * tr) - xi * ti` 而引入额外的消减误差。
采用这一结合次序以接近 [Netlib CGERU](https://www.netlib.org/blas/cgeru.f) 的列循环。
该参考实现先检查整数参数，再执行 quick return；当某列 y 元素为复数零时跳过该列。
设计保留零 y 列跳过，避免含 Inf/NaN 的 x 在零列上产生无意义的新 NaN。
特殊值最终还需在目标环境用 cblas 和 cuBLAS 复核，记录两者存在的差异。

### Host 侧设计

按以下次序处理，避免在非法或无计算场景访问 Device 数据：

1. handle 为 nullptr：返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`。
2. m<0、n<0、incx=0、incy=0、lda<max(1,m)：返回 `ACLBLAS_STATUS_INVALID_VALUE`。
3. alpha 为 nullptr：返回 `ACLBLAS_STATUS_INVALID_VALUE`（任务书明确要求）。
4. m=0 或 n=0：返回成功，不检查 x/y/A，不查询设备或下发计算。
5. 将 alpha 的两个分量复制为 kernel 参数；两分量为零时直接返回成功。
6. 需要计算时 x/y/A 任一个为 nullptr：返回 `ACLBLAS_STATUS_INVALID_VALUE`。
7. 按仓库有效句柄与 stream 约定准备 launch；在目标环境确认默认 stream 的合法行为，
   不直接照搬参考算子对空 stream 的额外限制。
8. 计算 tiling，直调 kernel，按实际 launch 返回能力映射错误；不在算子内同步 stream。

偏移均先提升到 int64_t，再进行减法、乘法和取负：

```text
kx = incx < 0 ? -(int64(m)-1) * int64(incx) : 0
ky = incy < 0 ? -(int64(n)-1) * int64(incy) : 0
xOffset(i) = kx + int64(i) * int64(incx)
yOffset(j) = ky + int64(j) * int64(incy)
aOffset(i,j) = int64(j) * int64(lda) + int64(i)
```

不调用 `abs(int)`，不因方便而拒绝 INT_MIN 步长。元素偏移与字节尺寸分别使用足够宽的整数，
乘以 sizeof(aclblasComplex) 前检查溢出，不能静默截断。超出可寻址空间的调用拒绝下发，
相应边界与状态码在实现和评审中明确。

Tiling 数据只包含地址、m/n/lda、步长、alpha、活跃核数及行分段长度等普通值。
参考当前 Sger 方式按值传入 launch，避免每次申请 GM tiling、偏移表或临时矩阵。
算子新增 GM workspace 为 0；handle 自身已有 workspace 单独统计，不计作该算子的新增分配。

### 分核及分块

采用行片段 R × 8 列的 tile；每个 tile 内复用同一组 x/ix，按列搬运 A。
R 为 32 个复数元素的整数倍，按平台 UB 容量计算，上限 2048：

```text
R = floor(min(2048, (UB_bytes - 8192) / 56) / 32) * 32
rowTiles = ceil(m / R)
colTiles = ceil(n / 8)
```

A 基址为 32 字节对齐且 lda 为 4 的倍数时，任务总数为 rowTiles×colTiles，
任务按 blockIdx、blockIdx+coreCount 的方式分配，各行段及列起点均保持 32 字节边界隔离。
A 基址对齐但 lda 不是 4 的倍数时，一个任务拥有一组 8 列的全部行段：
8 列组之间的地址差是 64×lda 字节，避免相邻核共享搬运边界块。
A 基址非 32 字节对齐时，使用单核处理全部任务。
活跃核数不超过平台可用 AIV 数量和任务数；上述分配不要求列数大于核数。

每个输出元素只属于一个核，无归约或原子更新。对于非对齐基址，单核退化可能明显影响性能；
四个硬性能场景均为紧凑、对齐、单位步长，走行列分块路径。

### 尾段与非对齐访问

A 使用 DataCopyPad，blockLen 精确为当前 count×8 字节，不把 UB 填充写回 padding。
CPU 模型检查有效范围、padding、保护区、每元素恰好一次写入及不同核 32 字节块的隔离。
该模型不模拟 DMA 原子性、缓存和流水线；CANN 9.1.0 的搬运限制及真机边界仍必须在 目标环境验证阶段 验证。

### Kernel 侧设计

1. 每组最多读取 8 个 y 复数并计算 t=alpha×y。复数零 y 列直接跳过，保留原 A 的位模式。
2. incx=1 时将 x 搬入交错缓冲区，并通过移位填充和偶数通道取负构造 ix=[-xi,xr]。
   其他步长按 64 位地址逐元素收集 x/ix，不分配与物理跨度成正比的 UB。
3. 输入 A 使用两个队列槽，在计算当前列之前预取下一列。每个输出槽保存一列的一个行片段。
4. 交错计算 `out=x*tr; out+=ix*ti; out=oldA+out`，分别对应 Muls、Axpy、Add，
   乘积在加 A 之前形成。Axpy 的融合舍入和非有限值行为需与目标 CBLAS/cuBLAS 复核。
5. TQue 管理 A 的搬入、计算、搬出依赖；复用 x 缓冲区前显式等待 Vector 完成。
   标量加载路径使用 S_V，连续搬运路径使用 MTE2_V，向量阶段之间设置 PIPE_V 屏障。
6. 只写回 count 个有效复数；x/y 始终只读。核函数通过值参数接收 tiling。

### UB 预算与性能优化

| 缓冲区 | 字节数 | 生命周期 |
| --- | ---: | --- |
| x 交错输入 | 8R | 当前行片段跨最多 8 列复用 |
| ix=[-xi,xr] | 8R | 与 x 相同 |
| 布局转换 scratch | 8R | 构造 ix，下一片段复用 |
| A 输入队列，2 槽 | 16R | 当前列与预取列 |
| A 输出队列，2 槽 | 16R | 计算、搬出及槽复用 |
| 显式缓冲区合计 | 56R | R=2048 时为 114688 字节 |

Host 额外预留 8192 字节作为容量余量；它不是额外分配的 TBuf，也不是设备实测峰值。
平台没有可用核或 UB 时返回 NOT_INITIALIZED；容量无法放下最小 R 时返回 NOT_SUPPORTED。
GM 无算子专用临时区；handle、运行时和测试框架的分配需另外记录。

目标环境验证阶段 先验证目标编译和同步，再测量四个指定 case。结合 profiling 调整 R、列分组和核数；
非单位步长的标量收集路径以正确性为先，收益必须由设备测量确认。
正常非零列更新的主矩阵读写流量约为 16mn 字节，该估算不代表性能达标。

### 工程文件

```text
include/cann_ops_blas.h                       # 新增公共声明
blas/ger/README.md                            # 补充 Cgeru API 和产品支持表
blas/ger/arch22/cgeru_host.cpp
blas/ger/arch22/cgeru_kernel.cpp
blas/ger/arch22/cgeru_tiling_data.h
test/ger/cgeru/CMakeLists.txt                  # ops_blas_add_gtest_tests
test/ger/cgeru/cgeru_param.h
test/ger/cgeru/cgeru_golden.h
test/ger/cgeru/arch22/cgeru_npu_wrapper.h
test/ger/cgeru/arch22/cgeru_test.cpp
test/ger/cgeru/arch22/cgeru_test.csv           # 原始 1200 条，逐字节保留
test/ger/cgeru/arch22/cgeru_supplemental.csv   # 新增 144 条
test/ger/cgeru/cgeru_data.h
test/ger/cgeru/cgeru_precision.h
test/ger/cgeru/scripts/                       # 生成、运行、解析及回归检查
test/ger/cgeru/portable/                      # CPU 顺序模型，不进入 NPU 构建
test/ger/cgeru/README.md
```

`blas/CMakeLists.txt` 已按架构收集源码，不为该算子另建平行构建系统。
测试复用仓库 `test/frame`，补全架构路径、参数解析和 CSV 注册。
保留 Sger、Cgerc 现有接口和其他架构实现；合入前再次检查共享头文件的最新主干声明，避免与 950PR 重复。

### 支持硬件与约束

目标支持 Atlas A2/A3，验收 CANN 9.1.0；910B3 完成四个硬性能用例。
不支持超出 incx/incy/lda 语义的任意 Tensor 视图。无额外广播、无运行时类型分派。
A 与 x/y 的内存重叠未由任务书规定，不主动承诺；如验收要求重叠输入，需单独确认语义和实现。

## 可维可测分析

### 精度及功能

保留配套 1200 条 CSV：1000 精度、200 性能。其中 1190 条期望 SUCCESS、10 条期望 INVALID_VALUE；全部 case 名称保持可追踪。
有效参数调用 `cblas_cgeru(CblasColMajor, ...)` 生成 golden；非法参数用状态码断言，避免调用
可能触发 XERBLA 的非法 CBLAS。禁止使用带共轭的 cgerc golden。

有限值分量分别统计，任务书要求为：

- `abs(actual-golden) <= 2^-16 + 2^-10 * abs(golden)`。
- 实部和虚部各自 matched_ratio >= 0.99。
- 最大绝对误差按任务书 `1e-2 或 32 ULP` 判断，保留绝对误差和 ULP 原始统计。
  首版以同时达到 1e-2 上限为保守目标；ULP 放行规则需在设计评审中明确。

NaN 采用同位置分类检查，Inf 检查位置及符号；不能用忽略非有限值后的均值掩盖错误。
alpha=0、零 y 列的未修改区域及 lda padding 做按位检查，保留 NaN payload/有符号零。
原包的 MERE/MARE 指标作为补充报告，不替代任务书的实虚分量混合容差判定。

补充空 handle、no-op 配合空 x/y/A、非法参数与零维组合、INT_MIN 步长单元素、
实虚部符号敏感算例、所有 36 种有效步长组合、指针偏移、padding 哨兵及 stream 顺序测试。
增加乘积实部相消而 A 非零的用例，验证先计算复数乘积、再加旧 A 的次序。
原包生成器未实现正态输入，alpha 使用固定值集合。新增 144 条补充 CSV，均匀和正态各 72 条，
两类各覆盖 36 种步长组合；记录均值、标准差、种子和指针偏移，alpha 与输入的实虚部分别采样。
这 144 条补充集满足两种分布各 50%，不声称未改动的原始 1200 条也符合该比例。
测试不允许空运行、崩溃或只通过部分用例时输出“全部通过”。

### 性能及内存

| m×n（incx=incy=1，lda=m） | 平均单次耗时上限（us） |
| --- | ---: |
| 512×512 | 5.70 |
| 1024×1024 | 10.10 |
| 2048×2048 | 55.71 |
| 4096×4096 | 219.35 |

每个性能用例先 warmup 10 次，再采样 100 次，保留有效次数、均值、最小值和最大值。
当前测试使用 aclrtEventElapsedTime 的 stream_event 时间区间，不使用 GTest 外层整数毫秒。
事件区间可能包含 Host 下发造成的流空闲时间；目标环境验证阶段 必须用 msprof 核对 kernel 时间与事件测量的差异，
在获得证据前不能把本脚本的 PASS 视为官方纯 kernel 性能结论。
golden 计算、内存分配、H2D/D2H、A 重置置于 kernel 计时区间外；重复更新时避免溢出，
同时提供单次功能校验及重复运行后的可解释性说明。
API 端到端耗时可另行记录，不能与 kernel 耗时混用。

四个用例以任务书上限为准；其余 196 条关联配套 GPU baseline，按倍率 0.8 单独报告。
未获取计时、缺失基线和测试失败分别记录，任何一种都不能作为性能达标证据。
记录输入/输出分配、框架分配、算子新增 workspace、可获得的设备内存统计与峰值采样限制。
真机结果、截图、编译日志、源码提交和 CANN/驱动/硬件版本共同进入自测报告。

### 兼容性与待验证事项

新增公共符号，不修改已有函数签名。编译时依 arch22 收集实现；只验证过某一架构时不宣称全产品测试通过。
本稿尚未附目标编译、设备精度、性能和内存证据。后续在 CANN 9.1.0 的 A2/A3 环境完成验证，
按真实结果调整实现参数；本地实现不降低任务书的最终验收要求。

## 提交范围与评审关注点

本次提交仅含设计文档。报名已通过，现申请设计评审；后续根据评审意见修订，
在设计评审通过并合入、完成目标设备自验证后申请验收。

当前任务目录中未发现广州 A2/A3 Cgeru 的独立编号目录；本稿以任务全名建立目录，
个人目录使用报名账号 gcw_rBObG5J6。如维护者已有统一编号，请告知以便调整。
请重点确认 32 ULP 放行口径、零 y 列特殊值语义及性能计时边界。

目标编译、真实硬件精度、流水线同步、性能和内存证据仍待补充。
