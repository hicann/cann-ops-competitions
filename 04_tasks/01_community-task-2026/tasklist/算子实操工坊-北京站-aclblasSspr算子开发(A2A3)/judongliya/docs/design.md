# aclblasSspr A2/A3 算子设计

提交人：GitCode `judongliya`。日期：2026-09-12。

活动任务：https://www.hiascend.com/activities/task-center/details/9d94810ec99446f09e9361fb47ddf758

任务讨论：https://gitcode.com/cann/ops-blas/discussions/29

实现仓库：https://gitcode.com/judongliya/ops-blas/tree/codex/sspr-arch22

源码提交：`a5c006e50bb336600bbec44c48f4759a2d4439b7`。

代码草稿MR：https://gitcode.com/cann/ops-blas/merge_requests/428

本文供方案评审。自测通过不表示官方方案评审、任务验收或代码合并已完成。

## 一、需求背景（必填）

本任务为ops-blas现有公共接口`aclblasSspr`补充Atlas A2/A3对应arch22实现，使调用方通过BLAS句柄和绑定stream执行FP32对称矩阵的packed秩-1更新。

目标计算为：

```text
A[i,j] ← A[i,j] + x[i] × (alpha × x[j])
```

对称矩阵仅存储上三角或下三角，AP长度为`n*(n+1)/2`，不构造完整矩阵、不引入lda参数。保留原有公共接口，避免产品私有平行API。适用场景为使用BLAS packed布局、希望减少矩阵存储量的对称秩-1更新。

实现支持范围为Atlas A2/A3的arch22；本次实际验证设备为910B3，CANN 9.1.0。A3尚未单独真机测试，不能将910B3结果外推为所有A2/A3产品已完成验收。

## 二、需求分析（必填）

### 2.1 接口与输入输出

声明复用`include/cann_ops_blas.h`中的现有接口：

```cpp
aclblasStatus_t aclblasSspr(
    aclblasHandle_t handle, aclblasFillMode_t uplo, int n,
    const float* alpha, const float* x, int incx, float* AP);
```

| 参数 | 位置/方向 | 类型与形状 | 约束与行为 |
|---|---|---|---|
| handle | Host输入 | aclblasHandle_t | 有效句柄，携带stream；空指针返回HANDLE_IS_NULLPTR |
| uplo | Host输入 | ACLBLAS_UPPER或ACLBLAS_LOWER | 非法枚举返回INVALID_ENUM |
| n | Host输入 | int标量 | n≥0；负值返回INVALID_VALUE |
| alpha | Host输入 | 指向FP32标量 | 指针不可为空；值可为FP32特殊值；本实现读取Host标量 |
| x | Device只读输入 | FP32，物理长度至少`1+(n-1)*abs(incx)`，n>0时适用 | 正、负非零步长；n>0且alpha非零时不可为空 |
| incx | Host输入 | int标量 | 不可为0；支持INT_MIN的64位转换，不在int32内直接取负 |
| AP | Device原地输入输出 | FP32，一维长度`n*(n+1)/2` | 旧值参与计算，原地覆写；n>0且alpha非零时不可为空 |

n=0时合法quick return，不解引用alpha，但仍检查其指针非空。alpha为+0或−0时不读取x/AP，允许二者为空，AP保持原位串。其余非法参数返回公共状态码。

接口依赖调用方持有有效Device内存和正确stream上下文；实现不校验实际分配容量，不增加超出BLAS约定的输入别名保证。不支持超出incx语义的任意非连续tensor，无broadcast，不新增随机行为。

### 2.2 存储与数值语义

采用0-based标准Netlib列主序packed布局：

- UPPER第j列行号为0..j，起点`j*(j+1)/2`。
- LOWER第j列行号为j..n−1，起点`j*(2*n−j+1)/2`。
- 正步长逻辑行i映射`i*incx`；负步长映射`(n−1−i)*(-int64_t(incx))`。

乘法顺序为`x[row]*(alpha*x[column])`，再加原AP；x[column]为零时按Netlib跳过该列，避免改变0/Inf/NaN混合场景的语义。

任务书“上三角从对角向上”文字与其引用的Netlib标准列序存在差异，本文按Netlib及主仓接口语义实现。alpha=0的空指针行为遵循任务书参数表，附件README关于先校验x/AP的说法不采用。

### 2.3 验收指标

精度golden由CBLAS Netlib `cblas_sspr`生成，比较全部packed AP：

- FP32逐元素条件：`abs(actual-golden) ≤ 2^-16 + 2^-10*abs(golden)`。
- 满足条件的比例≥99%，并要求每个有限结果绝对误差≤`max(0.01,32*ULP)`。
- NaN须对应NaN，Inf符号一致；alpha=0额外逐字节检查AP不变。

性能在910B3/CANN9.1.0测试，预热后有效采样超过50次并取平均。前四条门限分别为11.31、27.07、79.93、475.91us，对应n=512 UPPER、1024 LOWER、2048 UPPER、4096 LOWER，incx=1、alpha=1。附件其他196条按`gpu_ms*1000/0.8`换算门限；附件GPU数据不是本次独立实测。

## 三、详细设计（必填）

### 3.1 工程与模块

| 模块 | 路径 | 职责 |
|---|---|---|
| 公共接口 | include/cann_ops_blas.h | 复用现有声明，不新增API |
| Host | blas/spr/arch22/sspr_host.cpp | 参数检查、quick return、分核、绑定stream下发 |
| Tiling | blas/spr/arch22/sspr_tiling_data.h | 按值传递n、incx、uplo、alpha、packed长度、AP对齐余量、每核块数 |
| Kernel | blas/spr/arch22/sspr_kernel.cpp | tiny、packed批量和逐列回退路径 |
| 自测 | test/spr/arch22/ | CSV适配、CBLAS精度、边界/分布、性能计时与基线比对 |

当前主仓测试框架选择`test/spr/arch22`，与任务材料中的旧路径`test/spr/sspr/arch22`不同；本文按当前框架集成，目录差异请维护者评审确认。arch35实现保留，不改公共框架或其他产品线行为。

### 3.2 Host与tiling

依次检查handle、uplo、n/stride、alpha非空，再处理n=0。读取Host alpha后处理alpha=0，再检查x/AP。取得可用AIV核数，按实际AP地址的32字节块划分工作，核数不超过实际块数与可用核数。

packed长度、地址偏移和步长乘积使用64位整数。tiling按值传入kernel，不申请额外GM workspace。通过handle绑定stream异步launch；调用方读取结果前同步stream。

数据流为：

```text
Host参数校验 → 按值tiling与分核 → stream异步launch
Device x/AP → UB缓存与计算 → 原AP地址写回 → 调用方同步后读取
```

### 3.3 n=1独立路径

n=1使用独立kernel，两块32字节UB缓冲，共64字节。先读取x[0]，若为零直接返回；否则按规定乘法次序更新一个AP元素。incx的正负与大小不改变该唯一逻辑元素。

### 3.4 连续packed批量路径

条件为incx=1且32≤n≤4096，适用于对齐及偏移AP视图。

各核按实际32字节GM块边界获得连续packed区间，每片最多4096个元素。x缓存于UB，packed索引向量通过8个scalar种子及向量Adds倍增生成。根据三角反函数估计列号，求出行号，再Gather对应x行/列值。

n≤256时使用`floor(sqrt(2*q+0.25)-0.499f)`。该偏置只在910B3上对q=0..32895穷举32896项零错误后用于这个有限域，不声称一般Sqrt误差保证。n>256保留三角边界的向下/向上比较修正。LOWER通过镜像packed索引还原行列。

Compare需要完整256字节repeat，FP32计算数向上补齐到64个元素。补齐索引截断到合法末项，使Gather地址有效；只有实际validCount元素参加AP的Add/Select与GM拷贝，因此不读取未初始化AP padding，也不写出合法AP范围。

零列mask在乘alpha之前生成，最后Select原AP或更新值，保留Netlib跳过零列语义。计算保持分开的乘法与加法，不以不同结合顺序替换。

### 3.5 回退与分核边界

除n=1外，n<32、n>4096或incx不为1时走逐列分片，每片最多4096个元素。非连续x通过scalar GetValue收集，负步长遵循Netlib起点。

一般路径以AP实际地址32字节块分核，避免部分列尾跨核写同一块。连续、AP对齐且n>4096时按16列组轮转分核：两种packed格式的16列边界均为8个float对齐，保证组间不共享GM块。起始列可通过二分定位。

incx=1且n≤16384时可缓存一份x；不满足缓存条件则按片DMA或按stride收集。大n仍用64位packed计算，不使用有限域浮点反解索引。

### 3.6 数据依赖与内存

scalar索引种子通过S_V保护向量读取，片结束V_S保护下一片scalar复用。向量操作间使用PIPE_V屏障，CompareScalar后也有屏障。MTE2_V保护DMA输入被向量消费；V_MTE3保护结果搬出；MTE3_MTE2保护缓冲及部分GM块复用。

bulk的xCache只由Gather向量读取，已有MTE2_V保护，不额外等待MTE2_S；回退的scalar缓存读取保留MTE2_S。非连续收集使用对应S_V/V_S同步。

显式UB缓存上限：tiny为64字节；bulk为112KiB工作缓存、512字节mask及最多16KiB的x缓存，共128.5KiB；逐列回退为32KiB工作缓存，可选x缓存最多64KiB。以上均不含运行时附加开销。

算子无额外GM workspace。测试x/AP的aclrtMalloc请求含guard，n=4096最大33579264字节（约32.024MiB）；这是申请量，不是物理HBM峰值，也不含ACL运行时开销。

### 3.7 兼容性与边界

复用句柄、状态码和公共接口，不要求调用方改API；不影响arch35。支持正负非零incx及INT_MIN的安全转换、两种uplo、n=0、alpha=±0、空指针负向检查、非对齐AP以及规定的Inf/NaN场景。

A2/A3为实现目标范围，实测证据仅覆盖910B3/CANN9.1.0；未单独验证A3及其他工具链版本。任意有效分配的大n采用回退路径，但有限测试规模不等于已穷举所有可表示n与stride。

## 四、可维可测分析

### 4.1 测试与可复现性

最终源码提交的真机结果：

| 项目 | 结果 | 范围 |
|---|---|---|
| 完整精度 | 1204个启用GTest全部通过 | 原包1200CSV及4个专项，零失败/skip |
| 配对分布 | 2384次更新 | 1192个有效配置各均匀/正态一次，各占50% |
| 偏移/尾部 | 528组组合 | 上下三角、±1/±2/±3、形状与AP偏移 |
| 性能 | 200/200达标 | 每条统一200预热、1000有效调用，ACL事件算术平均 |
| 独立边界memcheck | 3/3测试通过 | 553次kernel检测未报错 |
| 独立典型形状memcheck | 4/4测试通过 | n=512/1024/2048/4096 |
| 索引probe | 32896/32896正确 | q=0..32895，910B3有限域 |
| 浮点极端候选probe | 60/60组AP[1]检查通过 | 5种n×3种incx×2种uplo×2场景 |

精度XML包含1个默认禁用性能测试，因此总数1205、启用1204；性能另行显式执行。专项中的内部组合不是额外GTest数量，避免重复统计。

四典型实际性能分别为7.66968、17.1521、47.7337、191.767us，均低于对应11.31、27.07、79.93、475.91us门限。n=1实际4.28476us、门限4.30125us，余量仅0.01649us，正式验收环境需要复测，不保证跨环境结果恒定。

60组浮点极端probe覆盖n=31/32/33/256/257、incx=−1/1/2、上下三角：次正规中间值场景AP[1]=0.5，次正规输入场景AP[1]=256，均与实际CBLAS和volatile分步FP32计算一致。该probe仅检查AP[1]，不是全数组极端值验证，不推广到所有次正规数或设备寄存器模式。

### 4.2 诊断与维护

测试保留原包CSV，独立适配字段和枚举名。精度失败只打印最多8个违反元素规则的位置、actual/golden及参数；性能日志记录每case计数、平均us、门限和设备申请字节。参数错误返回公共状态码，设备调用错误由运行环境及同步结果诊断。

测试入口与命令见实现仓库`test/spr/arch22/README.md`。交付包保留`validation/npu-final/`中的XML/log、两份逐case结果CSV、环境与源码SHA256，以及独立probe证据；`validation/summarize_npu.py`可从原始记录重建摘要，拒绝缺失/重复case或日志/XML不一致，不择取多轮最佳结果。

mssanitizer按suite分独立进程运行。曾有组合运行在首suite结束ResetDevice/Finalize后，第二suite报invalid_handle，该次不计通过；跨context工具注册失效属于原因推断。最终边界与典型形状分别在新进程检查通过。

尚未将racecheck/synccheck/initcheck列为最终完成的工具验证。CPU模型仅辅助同步算术/边界审查，不能替代设备流水、硬件数值或性能证据。

### 4.3 交付与评审状态

设计文档提交供评审，代码MR仍为草稿；尚未官方验收或合并，不宣称首位获奖。正式活动材料仍需按指定自测报告模板整理截图等交付项，并完成维护者评审及活动流程。

任务书测试目录与当前主仓结构差异、设计支持范围与仅910B3实测的区分，均请在方案评审中确认。个人仓权限及活动报名等流程事项应以平台实际状态为准，本设计文档不代替流程完成凭证。
