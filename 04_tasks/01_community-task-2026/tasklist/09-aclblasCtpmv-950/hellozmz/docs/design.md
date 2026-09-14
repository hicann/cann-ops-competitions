# 需求背景（required）

## 需求来源

算子实操工坊上海站 `aclblasCtpmv` 950 赛题。任务链接：https://www.hiascend.com/activities/task-center/details/87b85654a9444c119727ee5339f149df?menu=tasks 。
本设计按 `04_tasks/01_community-task-2026/resources/design_template.md`
的章节组织。基座 ops-blas commit：`2760346765b92e40579d16d8fffdb9f0988c35df`。

## 背景介绍

新增句柄式 complex64 三角 packed 矩阵向量乘法，公开声明放入 `include/cann_ops_blas.h`，
实现路径为 `blas/tpmv/arch35/`。目标芯片是 Ascend 950PR，CANN 9.1.0，编译目标 dav-3510。
使用 Ascend C SIMT kernel 直调。

# 需求分析（required）

实现 `x = op(A)*x` 的 UPPER/LOWER × N/T/C × UNIT/NON_UNIT 共 12 组语义，
支持任意非零 int 步长及负向 Netlib 遍历。UNIT 不读 AP 对角，x 原地更新。
非法参数在 Host 返回规定状态码。合法零维不引用 AP/x，也不发射 kernel。

任务书的 LOWER 0-based 显式公式与其 packed 布局描述不一致：`i+(2*n-j+1)*j/2` 会在末列越过 packed 数组。
依据同一任务书要求的列优先压缩布局、数组长度和 [Netlib ctpmv](https://www.netlib.org/blas/ctpmv.f)，
采用 `i+j*(2*n-j-1)/2`，等价于列起点 `j*(2*n-j+1)/2` 加列内偏移 `i-j`。
例如 n=3，LOWER 的列起点为 0、3、5，最大元素下标 5。

# 详细设计（required）

## 数学、类型与形状

`aclblasComplex` 为两个 float32；AP 长度 `n*(n+1)/2`，x 长度 `1+(n-1)*abs(incx)`。
UPPER 下标 `i+j*(j+1)/2`；LOWER 下标如上。T 交换行列，C 交换行列并取矩阵虚部相反数。
对角 UNIT 直接贡献旧 x 分量，避免 `(1+0i)*x` 对 Inf/NaN 引入额外污染。

## Host 和 tiling

检查次序为 handle、枚举、n/incx、零维 quick return、AP/x 指针。维数和步长类型与 Stpmv 公共 API 一致。
步长在 tiling 中扩展为 int64，避免 `abs(INT_MIN)` 的 32 位溢出。
packed 和向量元素偏移用 64 位算术。

`blocks=GetAivCoreCount()`；每核固定 1024 线程（32 个 warp）。
工作区以 work-item k 顺序存放，每 block 每轮负责连续 32 个复数（256 字节），
固定 launch 几何使复用 handle、改变 n 时同一位置仍归属同一 block。
每 32 线程负责一个输出行，行号从首尾交替映射，缓解三角长度差异。

工作区需求为 `8*n` 字节，通过 `EnsureDefaultWorkspace` 管理。
库工作区正常情况下预先存在；必要扩容遵循框架同步和容量规则。
用户工作区不足返回 ALLOC_FAILED，不替换借用内存。

## Kernel

1. 计算 kernel 只读 AP/旧 x，向工作区写连续结果。每个 warp 的 lanes 分担三角行中的项，
   以 shuffle 向下归约实部、虚部。
2. UNIT 跳过对角 AP。OP_N 遵循 Netlib 的零 x 跳过行为。
3. 检测非有限或绝对值超过 16 的操作数时，该行由 lane 0 按 Netlib 顺序重算。
   这覆盖大幅值交替数据的消去误差和极端值溢出；普通 [-5,5] 随机输入走并行路径。
   此路径的四个 float 乘积使用 volatile 临时值，保留逐步舍入/溢出行为。
   普通有限值仍在 NPU 并行执行。两条路径都没有 CPU 计算回退。
4. scatter 使用单 block，按 incx 写回逻辑元素，保留 x 空隙。
   逻辑行 i 通过 `i<(n+1)/2 ? 2*i : 2*(n-1-i)+1` 映射回工作区下标。
   单 block 避免带偏移/负步长的相邻写入由不同核负责。

两个 kernel 在同一个 handle stream 排队，第二个 kernel 开始前，第一个 kernel 已完成所有旧 x 读取。
不同 block 不直接读写同一输出元素，无跨核原子加或自旋屏障。
工作区在 handle 生命周期内保持有效，同 stream 连续调用可复用；并发 stream 使用不同 handle。

仅 Ctpmv kernel 编译为 O2；保留仓库其他文件的构建选项。特殊值路径使用显式中间值约束。
时间复杂度 O(n²)，临时空间 O(n)。不支持与 AP 重叠的 x，也不额外提供 broadcast、lda 或私有接口。

## 支持硬件

| 芯片 | 支持 |
| --- | --- |
| Ascend 950PR | 是 |

# 可维可测分析

原始 1200 条 CSV 使用 cblas_ctpmv 生成 Golden，检查状态码、全部逻辑分量、AP 完整性、x 空隙和边界哨兵。
所有 UNIT 用例毒化对角。额外 144 个分布子场景保证均匀/正态各一半且实部/虚部独立。
补充 24 个 n=255/256/257/511/512/513、incx=±1/±3 的边界场景。
补充极端 int 步长、非有限单位对角、零 x 跳过、用户工作区及排队调用。

精度：分别统计实部/虚部，atol=2^-16、rtol=2^-10，比例至少 0.99；
有限分量误差上界 max(1e-2,32 ULP)，NaN/Inf 分类需逐项一致。
性能：每条 TC_PF 恢复输入，预热 10 次、有效采样 60 次；统计 API 到 stream 同步的平均 us。
四项门槛为 44.89、88.46、184.2、1842.81 us。其余基线为空的 case 只记 NO_REF。

复现与测试步骤见 [测试README](https://gitcode.com/hellozmz/ops-blas/blob/feat/ctpmv-950/test/tpmv/ctpmv/README.md)。
代码提交为 `21087112c3308a4f64c90bfa916feca561dfb706`。
完整设计、自测报告及原始证据见 [交付包](https://gitcode.com/hellozmz/ops-blas/tree/delivery/ctpmv-950/deliverables/ctpmv_950)。
950实测1206项GTest、2724分量精度记录通过，4项性能门槛通过；5项契约测试384次kernel内存检查无告警。
补充随机测试144场景均匀/正态各72，原始CSV和特殊/边界场景另列，不声称原始与补充全集50/50。

## 兼容性分析

新增公共 API，现有 Stpmv 接口和实现保持原样。扩容、stream 切换、borrowed workspace 和销毁沿用 ops-blas 机制。
macOS用于资料管理与源代码审查，所有NPU验证在Ascend950PR真实环境执行。

## 请评审确认

NVIDIA cuBLAS 公开文档也出现同一 LOWER 公式；当前采用列起点加 i-j 推导出的 Netlib 索引，尚无官方勘误确认或 cuBLAS GPU 对照。请确认该实现口径。随机输入的50/50比例由独立补充集144场景（各72）验证，原始随题CSV保持不变；请确认统计范围。当前目录为上海950任务的描述性路径，请维护者确认目录编号；不套用A2/A3任务编号。
