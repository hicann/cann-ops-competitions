# 需求背景（required）

## 需求来源

2026 年 9 月社区任务：aclblasCtpsv 算子开发（A2/A3）。任务链接：https://www.hiascend.com/activities/task-center/details/45f49d0ea0fd4f6e9771346c5ae0eb46?menu=guide 。GitCode 账号：Aveline77。

## 背景介绍

在 ops-blas 中提供单精度复数三角打包线性方程求解，供三角回代等数值计算使用。对齐 cublasCtpsv 的列主序 packed、原地更新、转置/共轭转置和正负步长语义。输入 AP 为 Device 指针，x 初始保存 b，返回后 x 保存解。采用 Ascend C kernel 直调，使用 handle 绑定的 stream。

# 需求分析（required）

## 需求描述

接口：

```cpp
aclblasStatus_t aclblasCtpsv(aclblasHandle_t handle, aclblasFillMode_t uplo,
    aclblasOperation_t trans, aclblasDiagType_t diag, int n,
    const aclblasComplex* AP, aclblasComplex* x, int incx);
```

覆盖 UPPER/LOWER × N/T/C × NON_UNIT/UNIT 共 12 种组合；n≥0；incx≠0，支持负步长；UNIT 对角完全不读取；AP 不写入，x 的步长间隙保持不变。AP 和 x 不允许内存重叠，不检测奇异/病态系统。

## 需求拆解

1. 完成公共声明、arch22 host 校验与 kernel 调度。
2. 使用 64 位索引计算 packed 地址与负步长地址，避免 int 中间乘法溢出。
3. 实现原地复数三角代入，保持流异步语义。
4. 在 910B3、CANN 9.1.0 上使用仓库 GTest/CSV 框架与 Netlib cblas_ctpsv 验证。
5. 使用 msprof 采集核时间；不能用包含数据生成、CPU golden 和传输的 GTest 总时间代替。

# 详细设计（required）

## 算子分析

求解 `op(A)x=b`，op 为恒等、转置或共轭转置。AP 共 n(n+1)/2 个 complex64 元素，0 基索引为：

- UPPER：`i+j(j+1)/2`，0≤i≤j<n。
- LOWER：`i+(2n-j-1)j/2`，0≤j≤i<n。
- x：incx>0 时 `i*incx`；incx<0 时 `(n-1-i)*(-incx)`。

UPPER/N 与 LOWER/T,C 从 n-1 向 0 求解；LOWER/N 与 UPPER/T,C 从 0 向 n-1 求解。每一步严格依赖已解出的元素，使用一个 AIV，避免全局同步和核间自旋。

## 算子实现

### host 侧设计

先检查空 handle，返回 HANDLE_IS_NULLPTR；再校验 n、incx 和枚举，返回 INVALID_VALUE；合法 n=0 直接 SUCCESS，不检查 AP/x，也不发射 kernel；n>0 时检查两个指针。tiling 以值传递，包含 AP/x 地址、n、incx 和三个枚举。入口不分配临时 Device 内存，不进行 stream 同步，不访问 Device 指针指向的内容。

任务书 §2.4 与 §2.5 对 n=0 的 x 空指针存在文字差异。本实现遵循 §2.4 的 n>0 指针检查及 BLAS quick-return 约定；非法枚举/零步长始终在 quick return 之前校验，需评审确认。

### kernel 侧设计

1. 65≤n≤4096 使用单 AIV 的 UB 向量路径。N 路径的 pitch 向上对齐 32 个复数，匹配 GatherMask 完整 repeat 的读写边界；n≤2048 一次预取 8 列，其余一次 2 列。N 路径 UB 总量为 `(2*batch+6)*pitch*4` 字节，峰值 176 KiB；T/C 路径沿用 160 KiB。
2. N 路径批量发射多个不含对角的 AP 列片段后统一等待 MTE2，依次求出 x[j] 并用 4 条 Axpy 更新未解部分。八列批次的各槽位对齐前缀恒定，因此只在首次清零不会被 DMA 覆盖的前缀；两列批次保留逐批清零。零右端项沿用 BLAS 的跳过行为。UPPER/LOWER 和 UNIT/NON_UNIT 使用模板分支，减少内层枚举判断。
3. N 路径将已求解结果保存在独立的两个 UB 平面，工作 x 的对齐前缀可被后续更新覆盖。即使无限大乘数与补零产生 NaN，也只会影响已经求解且不再参与计算的前缀；最后仅从保存结果写回。这样避免每列额外的首块掩码和标量写后同步。
4. T/C 路径使用 AP 的连续列片段与已求解的 x 做复数点积，C 路径翻转矩阵虚部符号，再处理对角复数除法。规约前清零对齐产生的无效前缀。
5. AP 通过 DataCopyPad 按实际字节数搬入；GatherMask 将交错实虚分离。AP 仅搬运严格位于对角之外的有效系数，通过左补零处理偏移。补零超过 7 个 float 时先清空一个 32 字节块，再从后一块搬运，避免 DataCopyPad 的左填充限制。
6. 向量流水之间使用 PipeBarrier，Scalar/Vector、MTE2/Vector 之间使用配对事件同步。输出仅写逻辑 x 元素，核退出前调用 ResetMask 恢复向量掩码，最后清理 Device 数据缓存。
7. n≤64 或 n>4096 保留常量空间的标量泛化路径，不限制接口合法阶数；此路径性能须另行验证，不属于任务给出的 5 组性能规模。
8. 常规向量 NON_UNIT 使用按较大分量构造比值的复数除法，避免直接计算对角模平方，并处理零/无限对角。UNIT 分支不求取对角元素，测试会在对角填 NaN 检查此约束。
9. NPU 标量路径的复数乘法保持 Netlib 的操作数及舍入顺序。`ctpsv_scalar_math.h` 使用 64 位整数尾数、独立指数和 guard/round/sticky 位实现 binary32 融合乘加；有限复数除法保留 48 位精确乘积，以扩展整数相加和商位迭代完成一次舍入，不依赖 AICore 不支持的 double。
10. 复数乘法及除法保留 Inf/NaN 分类与恢复条件。向量路径若求解过程中出现非有限数，返回公共分派处，由 NPU 标量路径从 GM 中尚未修改的原始 RHS 重算；没有 CPU 求解或 Host 拷贝回退。最终写回前才修改 GM，保证重算输入完整。

### 工程结构

- `include/cann_ops_blas.h`：公共接口声明。
- `blas/tpsv/arch22/ctpsv_host.cpp`：校验和异步发射。
- `blas/tpsv/arch22/ctpsv_tiling_data.h`：值传递参数。
- `blas/tpsv/arch22/ctpsv_kernel.cpp`：标量与向量路径。
- `blas/tpsv/arch22/ctpsv_scalar_math.h`：可单独验证的标量浮点计算。
- `test/tpsv/ctpsv/arch22/ctpsv_test.cpp`、`ctpsv_test.csv`：复用仓库 CSV/GTest 框架。
- `ctpsv_test_supplement.csv`：336 条独立补充用例，不修改原始 CSV。
- `test/tpsv/ctpsv/host_math_check.cc`：直接包含实际算术头文件的主机交叉检查。

## 支持硬件

| 硬件 | 实现适配 | 自验证状态 |
|---|---|---|
| Atlas A2（含 Atlas 800I A2） | arch22 | 本次使用 910B3 实测，详见报告 |
| Atlas A3（含 Atlas 800I A3） | arch22 | 尚未实测，不声称 A3 验收通过 |

## 算子约束限制

单向量 complex64；不广播；AP/x 不重叠；内存由调用者分配且足够容纳 packed 和 strided 布局。允许负 incx。对奇异、近奇异系统不保证有限解。

# 可维可测分析

## 精度标准/性能标准

| 项目 | 标准 |
|---|---|
| 精度 | Netlib cblas_ctpsv；rtol=atol=2^-13；通过比例≥0.99；逐分量绝对误差≤max(0.01,32ULP)；NaN/Inf 分类一致 |
| 参数 | 12 种枚举组合、n=0/1/非对齐/大规模、正负步长、空指针、非法枚举/阶数/步长 |
| 内存 | AP 逐字节不变；x 步长空隙不变；不分配额外 Device workspace |
| 性能 | n=256/512/1024/2048/4096 五组门槛分别 252.2/573.9/1228/1211/7327 us |
| 采样 | 同一输入每次恢复 b；5 次 warmup 后至少 20 次有效采样；保留 msprof 原始数据 |

测试 AP/x 实虚使用独立随机种子；非单位对角实部加 max(5,n)。UNIT 的非对角按 1/n 缩放以避免大阶数病态三角系统溢出，所有输入处理对 NPU 与 golden 完全相同；这一条件化方式需在验收时明确记录，不据此声称覆盖所有病态矩阵。

## 兼容性分析

补充用例包含 192 条均匀/正态各半的输入（实虚独立随机流）、12 种枚举组合、n=1/7/65/129、正负步长、AP/x 起点偏移 8/16/24 字节，以及 144 条 AP/x Inf/NaN 用例（n=1/7/65）。AP 全区、x 前后保护区和步长间隙均验证不变。原始 1200 条测试的输入及容差保持不变。

主机算术检查直接使用实际头文件：200 万组融合乘加与 `std::fma` 比较；100 万组生成的除法输入中，984520 组有限且除数非零，与 double 中间值对照。两项本次均未发现数值差异。该检查不能替代 NPU 测试。

新增函数，不改变原有 Stpsv 或其他算子接口。公共声明独立于 arch22 实现，其他架构需要另外实现。所有性能与精度结论以对应源码版本的真实日志为准。

## 当前自验证结果（2026-09-28）

在 910B3/CANN 9.1.0 上，原始 1200 条和补充 336 条用例合计 1536 条已全部通过。原先 TC_FL_142、TC_FL_143、TC_FL_146 的大数舍入与 Inf/NaN 问题已修复，三条用例与 Netlib 输出逐元素一致。ULP 判定已处理 FLT_MAX 相邻值溢出边界，最终回归日志保留在交付包中。

全量性能采样得到 5200 条核记录；5 组性能的 20 次有效采样均值为 119.5733、432.5283、989.0649、1173.8474、6818.3650 us，5/5 均达到任务书门槛。数据来自相同 kernel 源码；最终测试程序另收紧了 FLT_MAX 的 ULP 边界判定，输入与性能调用路径保持不变。

本次只完成 910B3 与 Netlib 的自验证；尚未在 NVIDIA 硬件交叉核对 cuBLAS 特殊值，也未完成 A3 实测。任务书允许只选一种硬件自验，不据此声明正式平台验收或跨硬件验证完成。

内存检测覆盖 200 条性能输入、336 条补充输入和 8 条极值输入，共 544 条，分两个独立进程执行，全部未发现 memcheck 错误；见内存报告与完整日志。该结果不等同于 initcheck/racecheck 或泄漏检查。
