# 需求背景（required）

## 需求来源

2026 年 9 月社区任务第 70 号《aclblasCgbmv_A2A3 任务书》，目标为 CANN 9.1.0、Atlas A2/A3。
工程基于 ops-blas master，基线提交 `0e8a27b7aa94566d431a40e980bbc1f0820916b4`。
本文沿用社区设计模板的需求背景、需求分析、详细设计和可维可测分析结构。

## 背景介绍

主仓已有 arch35 的实数 Sgbmv；本任务补充 arch22 的 complex64 句柄接口。
接口遵循列主序带状压缩存储，直接在压缩数据上计算，避免展开成稠密矩阵。

# 需求分析（required）

## 需求描述

计算 `y = alpha * op(A) * x + beta * y`。m/n 始终是原矩阵 A 的形状。
支持 N/T/C、上下非对称带宽、lda padding、正负非零步长、空维度和特殊复数值。
返回值采用公共 aclblasStatus_t；由 handle 绑定 stream，接口保持异步。

## 需求拆解

1. 公共头文件新增签名；arch22 Host 负责校验和按值 tiling 下发。
2. AIV SIMD 实现复数乘法、带内访问和输出规约；A3 复用 arch22。
3. GTest 注册原始 CSV 与补充 CSV，CBLAS 生成 golden。
4. 原始 GPU 基线保留，性能采集和报告工具严格区分待测与通过。

# 详细设计（required）

## 算子分析

### 数学公式

零基下 `A(i,j)` 的带状地址是 `j*lda + ku+i-j`。
N 模式输出 i 的输入范围为 `[max(0,i-kl), min(n,i+ku+1))`，
A 相邻系数的复数元素步距为 lda-1。
T/C 模式输出 j 的输入范围为 `[max(0,j-ku), min(m,j+kl+1))`，
A 在该列内连续；C 模式对虚部取负。
对于步长 s<0，向量起始偏移为 `(length-1)*(-s)`，逻辑元素 k 位于 start+k*s。

### 支持数据类型及形状

输入/输出为 COMPLEX64，使用 FP32 算术，无降精度转换。
尺寸和步长为 int API；地址计算、带宽和跨度校验使用 int64_t。
运行时 m/n/带宽决定遍历范围，不依赖测试 shape 白名单。

## 算子实现

### Host 侧设计

校验顺序：空 handle；非法枚举/负维度/负带宽/零步长/不足 lda/空标量；
带宽上界；空维 no-op；alpha=0,beta=1 no-op；参与计算的指针；地址跨度。
零维与带宽上界的矛盾按 `kl <= max(m-1,0)`、`ku <= max(n-1,0)` 处理。
非法枚举返回 INVALID_VALUE。beta=0 只免除原 y 输入读取，不免除输出指针要求。
alpha/beta 是 Host 标量，只复制数值；A/x/y 生命周期由调用者保持到 stream 完成。
主机不申请/释放临时设备内存，不复制 tiling 到独立 GM，不同步 stream。

#### 分核策略

输出按四个复数一组，block 数为 `min(AIV核心数,ceil(outCount/4))`，
每核领取连续的四元素整数倍区间。最后一核裁剪到真实输出长度。
每个输出只由一个核负责，不使用原子加法、跨核归约或 GM 临时结果。

#### 数据分块和内存策略

沿规约维按 256 个系数分块。N 模式使用二维 DataCopyPad 收集步距 lda-1 的系数；
T/C 连续搬运。x 对绝对步长进行搬运，负步长在 UB 中反向 Gather。
每个非连续复数搬运到 32 字节槽，再用预生成 byte-offset 索引分离实部/虚部。
负步长按当前块长度在 32 字节对齐的表起点生成反向索引，尾块也保持地址对齐。
所有 DMA 只搬运逻辑有效元素，不触碰带外角区。
当字节 stride 超出 DataCopyExtParams 的 32 位范围时，退化为 64 位 GM 标量读取。
这保证 int API 的大步长泛化；该极大跨度路径仍待真机验证。

每核显式 UB：A 8,192 字节、x 8,192 字节、8 个 FP32 工作向量 8,192 字节、
4 个索引向量 4,096 字节、标量区 96 字节，总计 28,768 字节。
新增 GM workspace 为零，与问题规模无关；输入输出自身内存按 lda 和步长决定。

#### Tiling 规划

`CgbmvTilingData` 为共享的 POD，包含原始形状、带宽、步长、负步长起始偏移、
模式编码、每核输出数以及四个 FP32 标量分量。kernel 直调按值传递，不使用 tiling key。

### Kernel 侧设计

复数乘法拆成四次 Mul、一次 Sub、一次 Add。
N 模式先对 x 乘 alpha，与 Netlib 的列更新语义一致；T/C 模式在点积之后乘 alpha。
通常使用 ReduceSum 对每块实部/虚部规约，再对块结果累加。
短 N 输出（长度不超过 128）在 SIMD 乘法后按 BLAS 顺序进行标量累加，并先应用 beta，
减轻长规约抵消成小输出时与参考实现的漂移。带宽不超过八项时也避免规约设置开销。
这是数值精度/延迟的策略选择，需要真机数据评估其开销；不声明当前性能已达标。

MTE2_V 保证输入搬运完成；S_V 保证索引/标量填充对向量计算可见；
V_S 等待规约或向量产品后才读取标量；输出由 S_MTE3、MTE3_S 保证写回与缓冲复用顺序。
beta=0 路径不读取旧 y；其他情况使用精确八字节搬运，保留 stride 空洞。
Inf/NaN 传播保留普通 FP32 复数运算，beta=1 保留旧 y 分量而不做额外复数乘法。

## 支持硬件

目标支持 Atlas A2/A3 训练与推理系列（含 Atlas 800I A2/A3）。
本地开发环境为 macOS arm64，没有 CANN 或 NPU。当前仅完成本机语义验证，
Ascend 编译、流水与多核行为、A2/A3 产品支持结论均需真机确认。

## 算子约束限制

y 不得与 A/x 内存重叠，调用者需保证设备存储容量正确。
不接受超出有符号 64 位字节偏移表示能力的跨度。
不支持 Device alpha/beta、额外广播、任意 view 或非 lda/inc 意义的布局。
接口为异步；同步和异步执行错误检查由调用者在 stream 完成时进行。

# 可维可测分析

## 精度标准/性能标准

精度要求：rtol=atol=2^-13，混合容差匹配率至少 99%，每个分量最大误差不超过 max(0.01, 32 ULP)。
原始 1,200 条数据不修改，
补充 1,494 条覆盖，共 2,694 条 CSV；CPU 另有 Host 错误和 stream 传递的合并契约测试。
本机直接编译生产 Host 和 kernel 的算法代码，使用测试专用 ACL/AscendC 语义替身，
对照从 Netlib 下载的原始 Fortran CGBMV，经窄 CBLAS adapter 调用。
ASan/UBSan 检查地址和未定义行为；为故意传入的非法枚举关闭 enum 子检查。
替身不模拟硬件时序、Cache、DMA 竞争、Ascend 编译器或真实规约顺序。
参考库在 arm64 禁用 FCMLA 扩展，保留可移植的非融合复杂乘法顺序；其他编译器
可能采用不同的合法浮点表达式，仍以 NPU 环境所要求的 Netlib 比对结果为准。

官方性能标准：前五条采用任务书 §3.3 的 10.665、17.328、25.875、30.640、52.487 μs；
其余 PF 用例按附件测试指导采用 `gpu_ms*1000/0.8` 微秒上限。
内部竞争目标：争取第一名，以相对 GPU 基线 8× 为当前优化里程碑。
定义 `speedup = GPU 耗时 / NPU 平均 kernel 耗时`，内部目标耗时为 `gpu_ms*1000/8`。
报告将官方性能判定和内部目标达成情况独立列出；8× 不改变主办方验收规则，
也不构成第一名的证明或性能优化上限。官方活动页按提交时间顺序验收，率先通过后暂停其他开发者测试；
因此优先推进设计评审与真实达标交付，8× 不是推迟验收的附加条件。
全部 200 个 GPU 基线按 CSV 保留；每条单独预热，有效样本至少 11，默认 20。
当前缺少 NPU，未生成性能测量或峰值内存数据，报告明确标为待测。
现有实现尚不能证明达到 8×；取得真机数据后，重点评估每输出 DMA/同步开销、
N 路径的跨列访存和小形状启动开销，再决定多输出分块、数据复用及专用路径优化。

## 兼容性分析

公共声明只增加 aclblasCgbmv；现有 Sgbmv 代码不改动。
主仓 CMake 自动收集 arch22 下源码，新增测试目录注册同一 ops_blas_add_gtest_tests 宏。
接口复用公共句柄和状态码，A2/A3 共用 arch22，arch35 不提供该新接口实现。
尚未执行跨算子回归和 CANN 集成构建；这是取得环境后的必测项。

## 参考资料

- [ops-blas](https://gitcode.com/cann/ops-blas)，基线提交见本文开头。
- [Netlib CGBMV](https://www.netlib.org/blas/cgbmv.f)，用于布局、负步长和 golden。
- [cuBLAS GBMV](https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-gbmv)，接口语义。
- [CANN 9.1 DataCopyPad GM→UB](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/ascendcopapi/docs/en/api/SIMD-API/basic_api/memory_vector_compute/data_move/DataCopyPad_GMToUB.md)。
- [社区设计模板](https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md)。

## 当前实现与证据状态

本设计已有本地实现与 CPU 语义验证，2695/2695 项通过；A2/A3 真机编译和验证待环境到位。
本 PR 提交设计文档，不声明 NPU 性能达标。官方验收将提交固定代码 commit 与真实测试证据。
