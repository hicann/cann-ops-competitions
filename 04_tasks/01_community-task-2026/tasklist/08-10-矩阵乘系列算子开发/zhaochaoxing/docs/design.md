# aclblasChemm 950 设计文档

状态：acceptance04 复跑验收完成。正式构建的 4467 项精度/接口测试和 200 项性能测试均已通过；社区设计 PR 正在提交。

## 需求背景

需求来源为上海站算子实操工坊aclblasChemm 950赛题。按社区设计模板的需求背景、需求分析、详细设计、可维可测分析组织本文；模板原文保存在evidence/design_template.md。

## 需求分析与接口

以 `required/aclblasChemm_Atlas950PR_task_doc.md` 为需求依据，公共签名保持不变，代码位于 `blas/symm/arch35`，测试位于 `test/symm/chemm/arch35`。
目标实测为Ascend950PR_9579/arch3510、CANN9.1.0。910B工作流仅用于开发方法，未把arch22代码宣称为950实现。

输入complex64列主序。A的维度K由side决定（LEFT为m、RIGHT为n），B/C为m×n。Hermitian三角读取规则为：已存储位置直接读取，另一三角读取转置位置并对虚部取负，对角虚部为零。

## 详细设计

计算公式：LEFT为C=alpha*A*B+beta*C，RIGHT为C=alpha*B*A+beta*C。支持complex64，不广播，索引和前导维使用int64。用户负责保证Device矩阵容量与参数一致，A/B与输出C不作别名保证。

### Host与生命周期

验证handle、枚举、非负维度；零维直接返回。正尺寸验证alpha/beta/A/B/C、lda≥K、ldb/ldc≥m，并检查字节寻址乘法溢出。
alpha0/beta1可直接返回。其他alpha0使用SIMT缩放，beta0分支不读取旧C。指针为Host标量与Device矩阵，不采用参考PR中的Host矩阵拷贝接口。

直接SIMT不需要工作区；ordered SIMT 在受限尺寸内预展开 Hermitian A，LEFT 另外预计算 alpha*B。Cube使用 `EnsureDefaultWorkspace` 与 `GetEffectiveWorkspace`，库所有权和stream切换遵循既有handle实现。
未引入全局数据缓存、全局stream或全局event。所有三阶段kernel使用绑定stream，公共接口不为逐次计算添加Host同步。

### Kernel与tiling

1. alpha0、小输出及不能进入 Cube 的情况使用 SIMT。复数 alpha 或标量任一相关分量幅度大于1时，采用按 Netlib 顺序计算的路径。LEFT 重建每个输出元素在三角更新中的累加顺序；RIGHT 先计算对角项，再按列归约顺序计算。复数乘法的两个乘积经独立 FP32 临时量保留舍入边界，避免仅依赖 contraction 开关。适用范围内预展开 A，LEFT 另预乘 alpha*B；RIGHT 在 ldc 是64倍数时每线程处理16列。输出按物理 C 分区并跳过 padding，窄索引仅在全部地址计算安全时启用。
2. Cube计算C转置视图，交换左右矩阵和输出维，使B读取与C合并连续。M/N补齐16，K补齐128，越界填零。
3. 设备拆分左/右复数矩阵的实部、虚部和实虚部和。纯实alpha采用三次实数矩阵乘产生RR、II、SS，再由real=RR-II，imag=SS-RR-II恢复复数。保留四次实数矩阵乘实现，但当前 Host 将复数 alpha 分派到顺序路径。
4. K通常分四段；复数alpha或标量幅度较大时分八段，各段分别累加；合并阶段成对相加缩短FP32长链累积。128×128输出tile，L1 Kchunk256，L0 Kstep32。
5. FP32 C0=8。L1/LoadData/MMAD/Fixpipe沿用本仓ssymm的基础数据通路，使用成对HardEvent，循环末消费剩余事件。
6. Cube拆分平面的尺寸由Host限制在32位范围内，使用uint32索引；合并在ldc*n不超过UINT32_MAX时使用32位除法，否则保留64位路径。矩阵寻址仍使用int64前导维。
7. SIMT合并执行complex alpha/beta缩放，Cpadding不写。beta0只写输出不读旧C；纯实标量不引入0*Inf交叉项。3M预加产生非有限中间结果时，在NPU按原输入重算对应元素。输出C地址不满足512字节对齐时，单AIV写回避免跨核共享缓存行。

工作区以float元素数计为 `3*Np*Kp + 3*Kp*Mp + P*S*Mp*Np`（P=3或4，S=4或8），乘4得到字节数；段起始自然对齐。
默认实标量1024方阵需求72MiB，2048方阵需求288MiB；K>2048使用SIMT；满足 ordered pack 条件时，RIGHT 工作区为8*K*K字节，LEFT为8*(K*K+M*N)字节；实际句柄分配按库容量增长策略取整。

## 可维可测分析

### 精度与性能设计依据

最初四次GEMM全K累加通过配套基础精度，但补充正态输入发现最大误差问题。分四段后补充精度通过；三次GEMM降低矩阵乘工作量，使四项任务书性能通过首轮验证。
中间结果不降为FP16/BF16，使用FP32 Cube；独立FP64计算仅用于误差诊断，验收始终用Netlib CBLAS。
完整结果以最终自测报告与原始日志为准，开发中失败轮次不能计入最终通过数量。

### 验证与交付

保留1200条任务CSV和此前补充的46条数据与数值场景，再补充3416条分布测试（854种布局，每种均匀/正态分布各有随机标量与原标量对照）；总计4663条CSV，另有4项独立GTest（句柄、生命周期、偏移指针、最大有限值判据）。实现实/虚分量阈值、NaN/Inf分类和padding验证。
性能以预热5次、60次有效调用的ACL event均值判断。四项有明确任务书上限；实际gpu_baseline.csv的200条数值均纳入参考对标（gpu_ms/0.4），不得因原README写占位而忽略。
交付包含独立源码diff、设计文档、测试代码/CSV、原始日志与Profiler。未提交PR或验收系统时明确记录为本地待提交。

## 数值诊断证据

TC_EX_0394 的FP64诊断显示：分段Cube结果距独立FP64最大误差约0.00133，而Netlib约0.010007，二者差值在一个分量达到0.010498。为满足任务书单标杆判据，复数右乘改用BLAS累加顺序后，实虚部最大误差均降至0.001709。未放宽判据或删去用例。参考算法：[Netlib chemm](https://www.netlib.org/blas/chemm.f)。

### 支持硬件与兼容性

支持Ascend 950PR（arch3510）和CANN 9.1.0。公共接口沿用cann_ops_blas.h，无950私有平行接口；arch22的测试选择保持原路径。工作区由已有handle机制管理。长归约顺序路径优先满足单标杆精度，额外参考性能未达标项必须在报告中保留。

## 历史验证结果（初始版本，不能代表当前源码）

正式仓库构建后4467/4467项通过，200条性能均有60次有效采样。四项任务书正式采样为460.509/2999.100/460.448/2997.040us，均低于564.14/3824.68/547.6/3683.6us。
200条参考性能全部执行并在原始日志中保留；报告同时记录其耗时和内存，不以硬指标替代参考集。
测试进程树Host峰值RSS为800184KiB；每条性能用例的Device空闲内存快照和所需工作区见工作簿，快照差值不等同设备峰值。原始结果与最终Profiler均保存在evidence。

## acceptance04 复跑验证状态

本轮在 DevEnv_319499、Ascend950PR、CANN 9.1.0、NPU 0 上重新执行完整验收。精度脚本报告 4467/4467 通过，性能脚本报告 200/200 通过；每项性能用例均为 60 次有效采样。四项任务书硬性能均值为 460.650/2998.090/458.423/2993.310us，对应上限为 564.14/3824.68/547.60/3683.60us，全部满足。完整动作记录和原始日志见 `evidence/acceptance04_rerun_report.md` 与 `evidence/acceptance04/`。社区设计 PR、代码 PR、评审合入和验收系统提交仍按任务书流程推进。
