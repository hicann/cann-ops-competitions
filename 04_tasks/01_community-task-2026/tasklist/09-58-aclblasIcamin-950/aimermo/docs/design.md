# aclblasIcamin 算子设计

## 一、需求背景（required）

### 1.1 功能定位

`aclblasIcamin` 为单精度复数向量提供最小绝对值和元素的索引查询。
该接口服务于需要按复数分量幅度选择元素的 BLAS 调用场景。
这里的绝对值和定义为 `abs(real) + abs(imag)`，不是欧几里得模。
输入类型为 COMPLEX64，实部、虚部各为 FLOAT32，输出为单个 INT32。
接口遵循 BLAS 的 1-based 逻辑索引约定，重复最小值选择最小索引。

### 1.2 目标与边界

- 目标产品为 Ascend 950PR，目标软件环境为 CANN 9.1.0。
- 在 ops-blas 公共头文件中提供统一句柄式接口，不建立产品私有平行接口。
- 使用 Ascend C kernel 直调方式，在 handle 绑定的 stream 上执行。
- 实现目录为 `blas/iamin/arch35/`，沿用同族算子的工程组织方式。
- 支持正整数步长，零或负步长采用写零的 quick return 语义。
- 不涉及广播、矩阵维度、批处理、输入原地更新或负步长反向遍历。
- 本文描述接口契约与算法设计，不记录开发过程，不代替自测报告。

### 1.3 设计依据

需求依据为本任务 `aclblasIcamin_Atlas950PR_task_doc.md`。
实现依据为当前仓库的 `icamin_host.cpp`、`icamin_kernel.cpp`、`icamin_tiling_data.h`。
公共类型及声明以 `include/cann_ops_blas_common.h` 和 `include/cann_ops_blas.h` 为准。
cuBLAS `cublasIcamin` 是目标对标接口；标准 Netlib/CBLAS 不提供 icamin 例程。
任务方未提供任务书提及的 Icamin 事实表，Q7 特殊值规则保持 **pending（待确认）**。

## 二、需求分析（required）

### 2.1 数学语义

设逻辑索引 `i` 满足 `1 <= i <= n`，其物理复数位置为 `1 + (i - 1) * incx`。
在 C/C++ 的零起始数组中，访问地址为 `x[(i - 1) * incx]`。
每个元素的比较值记为 `score(i) = abs(real(i)) + abs(imag(i))`。
正常有限值路径返回使二元组 `(score(i), i)` 按字典序最小的 `i`。
同分数时比较的是逻辑索引，而不是物理数组位置，也不是线程或核编号。
例如 `n=3, incx=2` 时访问物理元素 1、3、5；第二个逻辑元素胜出应返回 2。
计算 score 使用 FLOAT32 加法，不提升为双精度，也不进行平方、开方。
两个有限分量相加溢出时，score 可为正无穷，按该 FLOAT32 结果比较。

### 2.2 公共接口与参数

接口参数顺序固定为 `aclblasIcamin(handle, n, x, incx, result)`。
返回类型为 `aclblasStatus_t`，结果值写入 Device 内存，而非通过返回值传出。

| 参数 | 类型 | 位置与方向 | 契约 |
| --- | --- | --- | --- |
| handle | aclblasHandle_t | Host，输入 | 已创建的有效句柄，携带 stream 与工作区 |
| n | int | Host，输入 | 复数逻辑元素数量，正常路径大于零 |
| x | const aclblasComplex* | Device，输入 | 实虚交错、只读 COMPLEX64 向量 |
| incx | int | Host，输入 | 以复数元素为单位的步长 |
| result | int* | Device，输出 | 一个 4 字节 INT32，正常结果为 1..n |

正常路径要求物理存储至少容纳 `1 + (n - 1) * incx` 个复数元素。
调用方负责保证输入范围可访问、输出可写，以及缓冲区的正确生命周期。
Host 不检查 Device 指针归属和实际分配长度，也不检测缓冲区重叠。
result 应与输入和工作区独立，输入间隔中的填充元素不参与计算。

### 2.3 参数校验与异常优先级

校验顺序是公共行为的一部分，组合异常不能任意交换判断次序。

| 顺序 | 条件 | 处理 |
| --- | --- | --- |
| 1 | handle 为空 | 返回 ACLBLAS_STATUS_HANDLE_IS_NULLPTR |
| 2 | n 小于零或 result 为空 | 返回 ACLBLAS_STATUS_INVALID_VALUE |
| 3 | n 等于零或 incx 小于 1 | 在 stream 上异步将 result 写零 |
| 4 | 正常路径 x 为空 | 返回 ACLBLAS_STATUS_INVALID_VALUE |
| 5 | 查询到的 AIV 核数为零 | 返回 ACLBLAS_STATUS_EXECUTION_FAILED |
| 6 | 多核工作区为空或容量不足 | 返回 ACLBLAS_STATUS_EXECUTION_FAILED |

因此 `n<0, incx<1` 仍是非法 n，不属于 quick return。
quick return 允许 x 为空，但要求 handle 与 result 有效。
写零使用 runtime 异步内存清零操作；当前实际 API 名为 `aclrtMemsetAsync`。
调用中的目标容量和清零长度均为 `sizeof(int)`，即 4 字节。
该路径不提交归约 kernel；不能将“不提交 kernel”理解为 Host 已完成 Device 写入。
异步清零提交失败映射为 ACLBLAS_STATUS_EXECUTION_FAILED。
正常 kernel 提交后接口返回成功，异步执行错误仍需通过 stream 同步检查。

### 2.4 特殊值与 Q7 待确认边界

任务书给出的暂定 CPU 参考规则为：首 score 为 NaN 时以 FLT_MAX 初始化基准。
对应索引仍为逻辑索引 1，后续 NaN 跳过，仅严格更小的 score 才更新。
当前实现据此将首 NaN 归一化为候选 `(FLT_MAX, 0)`，内部索引从零开始。
其余 NaN 不形成候选，不将 NaN 本身送入比较归约。

| 输入 score 情形 | 暂定结果与依据 |
| --- | --- |
| 首项 NaN，后续有小于 FLT_MAX 的有限值 | 选择后续最小有限值的最早索引 |
| 首项 NaN，后续只有 NaN 或正无穷 | 返回 1，初始 FLT_MAX 小于正无穷 |
| 首项 NaN，后续最小值等于 FLT_MAX | 返回 1，同分数保留较小索引 |
| 全 NaN | 返回 1，保留首项的暂定候选 |
| 全正无穷且首项非 NaN | 返回 1，正无穷是有效候选 |
| 正零、负零及重复最小值 | score 相同，选择最小逻辑索引 |

Q7 状态为 pending，不声称上述 NaN 行为已经与 cuBLAS 实机对齐。
任务方需确认规则、cuBLAS 版本及特殊值对照依据；确认前保留此语义边界。

### 2.5 验收目标

输出只按 `actual == golden` 进行整数精确比对，不使用浮点容差替代索引检查。
CPU golden 按 FLOAT32 的 `abs(real)+abs(imag)` 顺序扫描，严格小于才更新。
任务性能目标为 incx=1 时，n=1048576、2097152、4194304 的平均耗时分别不超过 24.77、24.59、29.69 微秒。
性能测量需先预热，有效采样次数大于 50；该目标不是本文声明的实测结论。

## 三、详细设计（required）

### 3.1 分层职责

Host 层承担参数校验、quick return、分块参数计算和工作区容量检查。
第一阶段 kernel 完成输入扫描、线程局部候选更新、核内归约。
单核配置直接写结果，多核配置先写每个核的候选。
第二阶段 kernel 只在 blocks 大于 1 时启动，归约工作区候选并写结果。
两个 kernel 使用同一 stream 的先后顺序建立依赖，不引入 Host 中间读回。
整个计算为纯索引归约，不使用浮点原子最小值或跨核自旋锁。

### 3.2 Tiling 参数

`IcaminTilingData` 含五个 uint32_t 字段，由 Host 计算后随启动参数传递。

| 字段 | 含义 | 计算方式 |
| --- | --- | --- |
| n | 逻辑元素数量 | 已校验的正 int 转 uint32_t |
| incx | 正步长 | 已校验的正 int 转 uint32_t |
| blocks | 第一阶段 AIV block 数 | 小规模取 1，否则 min(可用核数, ceil(n/4096)) |
| elementsPerBlock | 每块逻辑区间长度 | ceil(n/blocks) |
| threads | 每块 SIMT 线程数 | n<=4096 为 256，否则为 1024 |

`n<=4096` 固定单核单 kernel，避免第二次启动与工作区读写。
`n>4096` 按可用核数和目标分块粒度分配，一般使用多核两 kernel。
若可用核数只有 1，即使 n 大于 4096，仍走单 block 直接输出路径。
第二阶段固定一个 block、256 个线程，不沿用第一阶段的 1024 线程配置。
4096 是启动策略阈值，不是输入长度上限。

### 3.3 分块与线程扫描

block b 的起点为 `start=b*elementsPerBlock`。
终点为 `end=min(start+elementsPerBlock,n)`，采用左闭右开区间。
线程 t 从 `j=start+t` 开始，每次递增 blockDim.x，直到 j 不小于 end。
该方案覆盖各块逻辑区间，每个有效逻辑元素恰好访问一次。
尾块用 end 限制访问范围，未分配到数据的线程保留无效候选。
即使整个 block 无有效元素，也会完成归约并写入无效候选，不读取越界输入。

### 3.4 输入读取与 score 计算

kernel 把 COMPLEX64 存储视为实虚交错的 float 数组。
对零起始逻辑位置 j，实部 float 偏移为 `uint64_t(j)*incx*2`，虚部为下一位置。
地址乘法先提升到 uint64_t，避免普通 uint32_t 乘积截断。
`incx==1` 与通用正步长分别实例化模板，连续路径省去运行时步长乘法。
读取采用 SIMT 线程直接访问全局内存（GM），每元素读取两个 float。
绝对值通过清除 FLOAT32 符号位实现，随后执行 FLOAT32 加法。
使用 `score==score` 判断非 NaN，首 NaN 按 Q7 暂定规则单独转换。
当前设计没有显式 SIMD 批量搬运、DataCopy 输入流水线或双缓冲。
不把连续逻辑访问等同于已经验证的硬件带宽或指令级优化效果。

### 3.5 候选表示与比较规则

`IcaminCandidate` 包含 `float score` 与 `uint32_t index`，当前布局为 8 字节。
index 保存原始零起始逻辑位置，始终保持整数表示，不数值转换为 float。
无效索引为 `0xffffffffU`，即 UINT32_MAX，与合法 int 范围的逻辑索引分离。
线程初始状态为 score=0、index=INVALID_INDEX；初始 score 不参与有效性判断。
候选合并先判断传入 index 是否有效，再按以下规则决定是否覆盖当前候选：

1. 当前候选无效，则任何有效候选都可覆盖。
2. 传入 score 更小，则覆盖。
3. score 相等且传入 index 更小，则覆盖。
4. 其余情况保留当前候选。

无效性由 index 表达，因此有效的正无穷不会被零初值或有限哨兵错误排除。
首 NaN 的 FLT_MAX 是任务语义候选，不是所有线程通用的初始化哨兵。
经过 NaN 预处理后，所有层级使用同一比较规则，保证跨线程、跨核 tie 处理一致。
大于 2^24 的索引仍通过 uint32_t 精确保留，不依赖 float 的整数精度。
最后只在写输出时执行 `bestIndex+1` 并转换为 int32_t。

### 3.6 32-lane 核内归约

SIMT 线程按 32 个 lane 组成一个 warp。
`WarpMinimum` 依次以偏移 16、8、4、2、1 执行向下 shuffle。
每次同时交换 score 和 uint32_t index，并按同一候选合并规则更新。
warp 的 lane 0 把该 warp 的最优候选写入核内统一缓冲区（UB）数组。
所有线程调用 `asc_syncthreads`，确保各 warp 写入对后续归约可见。
第一个 warp 从 UB 读取各 warp 的候选，再执行一次 32-lane shuffle 归约。
超出实际 warp 数量的 lane 装入 INVALID_INDEX，不参与最小值选择。
256 线程对应 8 个 warp，1024 线程对应 32 个 warp。
第一阶段为两个 32 项数组预留 UB，共 32*4+32*4=256 字节的候选数据。
第二阶段使用两个 8 项数组，共 64 字节；这些是源码显式数组大小，不代表全部编译后资源占用。
归约完成后仅线程 0 写入 Device result 或本 block 的工作区槽位。

### 3.7 跨核工作区与二阶段归约

单 block 路径无需读取工作区，启动描述中以 result 地址替代空工作区指针。
这一描述参数不会触发工作区写入，输出仍仅由直接输出分支写一个 INT32。
多 block 路径通过 `GetEffectiveWorkspace` 取得 handle 当前工作区。
所需容量为 `ceil_align(sizeof(IcaminCandidate)*blocks,64)` 字节。
64 字节对齐指容量向上取整；当前 Host 代码不另外检查指针的 64 字节地址对齐。
实际工作区应使用合规的 Device 分配，不据容量取整推断任意偏移地址都受支持。
每块只写 `workspace[blockIdx.x]`，score 与 index 同时构成一个完整候选。
第二阶段线程按自身线程号起步、以 blockDim.x 为步长扫描候选槽位。
之后复用相同 BlockMinimum 逻辑，由线程 0 写最终索引。
同一 stream 的 kernel 顺序保证候选全部写完后再由第二阶段读取。
因此无需预先清空工作区，也无需通过全局计数器推断第一阶段是否完成。

### 3.8 资源、生命周期与并发

Icamin 调用自身不申请新的 GM 工作区，也不调用工作区扩容接口。
工作区为空或过小时立即报执行失败，不能默默回退为越界写入。
句柄默认工作区的分配由公共句柄管理逻辑负责，不属于本算子的逐次临时分配。
同一 stream 串行复用同一工作区可依靠流顺序保证安全。
不同 stream 并发使用相同工作区时，调用方必须建立同步或采用独立工作区。
输入、输出、工作区必须保持有效，直至相应 stream 完成。
Host 不同步等待正常计算结束；调用方在读回结果前检查同步状态。

### 3.9 性能设计与复杂度

有效输入扫描复杂度为 O(n)，块内归约代价为固定规模的分层 shuffle。
多核汇总处理 O(blocks) 个候选，临时容量同样为 O(blocks)。
连续输入理论有效读取量为 8*n 字节，实际总线传输量需实测。
正大步长仍只访问逻辑元素，但可能降低访存局部性，不能套用连续场景性能结论。
单 kernel 小规模路径优先降低启动开销，多核路径优先提高大输入扫描并行度。
分块阈值、线程数及两次启动成本是性能复核重点，调参不得改变比较或 Q7 语义。

## 四、可维可测

### 4.1 可维护性

- 公共接口、Host 校验、tiling 定义、kernel 归约分别承担单一职责。
- 候选比较集中于 MergeCandidate，各归约层级保持相同的 tie 规则。
- 连续与通用步长由模板选择，避免重复实现不同的索引比较语义。
- 单核直接输出与多核中间输出共用扫描逻辑，降低边界修复不一致风险。
- 修改阈值时同时审查 Host 线程数、kernel 启动上限、UB 数组和 warp 数量。
- 修改 Q7 时同步维护首元素处理、CPU golden、特殊值用例及公开约束。

### 4.2 功能验证设计

| 类别 | 重点覆盖 | 判定方法 |
| --- | --- | --- |
| 基本功能 | n=1、小质数、最小值在首中尾 | 与 CPU golden 的 INT32 完全相同 |
| 索引与步长 | incx=1/2/3，间隔填充更小值 | 返回逻辑索引，忽略填充元素 |
| 分块边界 | n=4095/4096/4097、幂次及相邻值 | 检查直接输出与二阶段边界 |
| 层级 tie | 同线程、跨 warp、跨 block 等分数 | 总是选择较小逻辑索引 |
| 无效线程 | 不满 warp、不满 block、尾块 | 无效候选不覆盖有效候选 |
| 大索引 | 索引超过 2^24 的稀疏最小值 | 检查整数候选精度 |
| 特殊值 | 零、极值、Inf、首 NaN、后续 NaN、全 NaN | Q7 用例明确标记暂定语义 |
| 参数异常 | 空 handle、负 n、空 result、正常路径空 x | 精确检查状态码及判断优先级 |
| quick return | n=0、incx=0/-1/-2/-3，x 可空 | 成功、result=0、不启动归约 kernel |
| 内存与工作区 | 输入偏移、尾部保护、工作区容量不足 | 不越界，失败状态符合契约 |

随机输入按任务要求覆盖均匀与正态分布，实部和虚部独立生成。
测试应固定并记录随机种子，输出失败用例的 n、incx、填充模式与期望索引。
quick return 先把 result 写成非零哨兵，再验证其被异步改为 0。
输出前后保护区与输入只读校验用于发现非目标区域写入。
次正规数、极小差值与 FLOAT32 舍入边界需专项复核，不以设计推导替代硬件验证。

### 4.3 性能与资源验证设计

在 Ascend 950PR、CANN 9.1.0 上记录环境、预热次数、采样次数与计时范围。
对三个任务性能 case 逐项比较平均微秒耗时，不用最小耗时替代平均值。
计时需要包括完整算子路径，多核场景应覆盖第一阶段和汇总阶段。
区分输入/输出内存、句柄已有工作区与算子候选区的占用，避免重复计入。
采用工具检查异步任务、实际 kernel 数量和内存访问，不从源码臆测实测结果。

### 4.4 使用示例与交付边界

独立示例为 `test/iamin/icamin/examples/icamin_example.cpp`，不依赖 GTest。
示例使用 AclContext 和 unique_ptr 管理资源，逐项检查调用与清理状态。
`n=3, incx=2` 的 score 为 4、1、1，验证并列时返回逻辑索引 2 而非物理位置 3。
附带零维、零步长与负步长用例，Device 输出在同步后拷回 Host。
README 提供完整代码及仓库根目录编译命令；编译运行记录由独立验证环节产生。
