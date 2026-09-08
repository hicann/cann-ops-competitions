# nearest 算子设计文档

## 1. 需求背景（required）

### 1.1 需求来源

[CANN训练营北京邮电大学-nearest算子开发(950)](https://www.hiascend.com/activities/task-center/details/718ae9ad67ad47b4b82e5d366f6aa2f5)。流程参考[03任务引导](https://gitcode.com/org/cann/discussions/285)，格式参考[官方设计模板](https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md)。

### 1.2 背景介绍与现状分析

nearest为每个查询点寻找同一batch内最近的候选点，是图和点云1-NN分配的基础算子。上游torch_cluster包含CPU参考路径和CUDA实现，本设计在Ascend950提供完整NPU kernel，不以SciPy回退代替计算。逐查询SIMT会重复读取候选；本设计以AIV分块共享候选，并为F3/FP32融合寄存器距离和winner计算。

### 1.3 功能分析

输入x[N,F]与y[M,F]，输出每个x最近的y全局索引cluster[N]。batch数组须有序且非空ID集合相同，计算限定在同batch中。一维输入视为F=1。

## 2. 需求分析（required）

### 2.1 需求描述

实现nearest(x,y,batch_x=None,batch_y=None)，支持FP16/FP32 NPU输入和同设备int64输出。任务书18形状x2dtype逐项要求标杆耗时/NPU接口耗时>=0.45。性能表是任务给定预算，不改称CPU性能基线。

### 2.2 需求拆解

| 模块 | 职责 |
|---|---|
| Python | shape/dtype/device及batch校验，复用ID/count构造CSR |
| Host | 四Tensor注册、连续化、设备保护、平台查询、统一tiling和流提交 |
| AIV | 候选共享、双输入缓冲、确定性最小值及索引归约 |
| F3 FP32 | 寄存器距离与winner融合，减少UB中间读写 |
| SIMT | 多batch、大F及其他fallback形状的完整设备计算 |
| 验证 | 原始测试、补充契约、36项接口预算、正式安装和诊断 |

### 2.3 数值口径评审确认项

任务书CPU torch_cluster/scipy的逐位一致要求与随附test_nearest.py的输入dtype逐操作舍入和1024-lane平局规则存在差异。等距候选位于1与1024时，随附参考选1024，真实SciPy选1；测试环境SciPy vq不直接接受FP16。当前实现按未修改的随附参考自测。**请评审确认最终数值口径；本地通过不代表该差异已获官方批准。** 不通过修改原样例来掩盖差异。

## 3. 详细设计（required）

### 3.1 算子分析

#### 数学公式

cluster[i]=argmin(j in same batch) sum(d=0..F-1,(y[j,d]-x[i,d])^2)。

距离按维度递增执行subtract、multiply、add；FP16每步RNE到half。AIV用float操作后转half再扩宽，SIMT用对应intrinsic。不能默认用重新结合的ReduceSum、FMA或矩阵恒等式替代。等距按batch局部编号l的(l%1024,l//1024)字典序，最后转全局y索引。

#### 数据类型和形状

| 参数 | 类型 | 约束 |
|---|---|---|
| x | FP16/FP32 | NPU，[N,F]或[N] |
| y | 同x | 同设备同dtype同正F，[M,F]或[M] |
| batch_x/batch_y | 可选int64 | [N]/[M]，同设备、非负、非递减 |
| cluster | int64 | 同设备[N]，全局y索引 |

不支持广播；支持非连续输入规整。单侧batch省略时补batch零再检查集合。通过形状、设备及batch等入口校验后，N=0返回空；N>0且M0、F0、非法元数据拒绝。

### 3.2 算子实现

#### 3.2.1 Host侧设计

调用链为Python、torch.ops.torch_cluster.nearest、PrivateUse1、launcher。底层四个必填Tensor为x/y/ptr_x/ptr_y。普通CSR做长度、首尾、单调性、设备检查；两个空NPU int64指针为无batch快速扩展，普通[0,N]/[0,M]仍支持。兼容已存在的上游schema。

分核：平台API读取AIV数与UB。AIV按query商余分核，余数给前部核心，每输出一个owner；SIMT按query grid-stride，传入真实batch数做有界查找，单batch不读额外边界。

分块：Host生成NearestVecTiling，包含n/m/mPad/f/R/B/C/blockDim/S。C=ceil(M/R)，mPad=C*R，kernel不另算R。优先R1024，B按每核query和资源尝试8/4/2/1；不足时选择64对齐的较小R，无法容纳则转SIMT。每buffer按32B对齐，另留4096B保守余量。

令A(v)=32*ceil(v/32)，s为元素字节，Q在F3/FP32/R1024时1、否则2：

| UB用途 | 字节 |
|---|---:|
| 双输入 | `2*A(F*R*s)` |
| 距离和索引状态 | `A(4*Q*B*R)+A(4*B*R)` |
| 两float临时向量 | `2*A(4*R)` |
| half专用扩宽/舍入 | `A(4*R)+A(2*R)，float不分配` |
| 五int32向量及mask | `5*A(4*R)+A(R)` |
| 两归约scratch | `2*A(4*(floor(R/64)+8))` |
| 最小值/key槽 | `2*A(64*B)` |
| int64 stage和query标量 | `A(8*S)+A(4*B*F)` |

F3/B8/R1024/S512请求FP32 125216B、FP16 151840B，另加余量。实测56 AIV/28 AIC/253952B UB；实现动态查询，不固定这些数值。

分派/tilingkey规划：本扩展用Host条件和模板实例，不虚构单一GE tilingkey。无batch或完整单batch且资源满足时，F3/FP32/R1024走寄存器AIV；F<=96其他合法方案走通用AIV；多batch、大F或资源不足走SIMT。

Host转置y为[F,mPad]、尾部填Inf，half query扩宽供广播。half对齐与奇数尾读拥有完整32位存储范围。全部ATen前置操作后获取已提交的受管理stream，在同流追加kernel，无no-op、首项哨兵和重复重试。

#### 3.2.2 Kernel侧设计

1. Init读取统一plan，设置每核范围、UB和事件。
2. CopyIn双队列搬入F列候选，B查询共享；half列扩宽在组内复用。
3. Compute的F3/FP32用64-lane寄存器片段完成三维距离及Compare/Select。通用路径逐维累计，当前query立即消费mask，不让共享scratch/mask被其他query覆盖。R1024利用key单调性strict-LT更新；其他R显式比较distance/key。
4. 全部候选块完成后归约最小距离，再归约等距最小key。改R必须同步key公式。
5. CopyOut用精确int64暂存与有效字节DataCopyPad；S到MTE3保证写入可见，MTE3到S保证读完再复用。

SIMT以首个有效候选初始化，避免1e30有限哨兵；局部tie与全局输出分别维护。

### 3.3 支持硬件

| 硬件 | 状态 |
|---|---|
| Ascend950/arch35 | 支持，实测Ascend950PR_9579/dav-3510 |
| 其他架构 | 未验证，不声明兼容 |

依赖CANN>=9.1.0、PyTorch>=2.7及配套torch_npu。已验证CANN9.1.0、PyTorch2.9.1+cpu、torch_npu2.9.1、Python3.12.13；其他组合重新构建验证。

### 3.4 算子约束限制

有限坐标为验证域；NaN不声明统一winner行为，入口不做全量isfinite扫描。half溢出按既定舍入。AIV int32 key有范围保护，超出转SIMT；输出始终int64，已覆盖>2^24索引。batch有序且非空集合一致，不用CPU回退，不定义离散索引梯度。

## 4. 可维可测分析

### 4.1 精度标准/性能标准

| 标准 | 描述 | 来源 |
|---|---|---|
| 正式精度要求 | 与CPU torch_cluster/scipy输出索引逐位一致，平局处理与CPU一致；尚无口径变更批准 | 任务书，见2.3 |
| 当前开发自测 | 与未修改随附参考逐索引一致；不能替代正式CPU一致性要求 | test_nearest.py，差异见2.3 |
| 性能 | 每项接口耗时<=标杆/0.45 | 任务书18x2表 |
| 完整性 | 完整输出及shape/dtype/device/范围、异常与并发 | 原始和补充测试 |

独立安装包自测：原始42项加补充测试共111项通过，实际56核每核512条输出边界另1项通过，合计112个不同用例。36/36性能项通过，每项两轮20预热/100计时取较慢均值。最紧32K平方/F3/FP16为7.188306ms，预算7.333333ms，余量1.98%。这不代表官方评审或后台验收通过。

| F3/FP32形状 | 实测ms | 预算ms |
|---|---:|---:|
| 16K x 16K | 0.790630 | 1.295556 |
| 32K x 32K | 2.916589 | 4.651111 |
| 32K x 16K | 1.572525 | 2.422222 |

覆盖单batch、局部tie、F96跨块、非连续/偏移、非法输入、多流、首调、真实torch_cluster共存和大int64索引。wheel在独立环境安装并核对真实动态库映射。Profiler采用独立进程0/5/5，8组双NPU诊断与大F3时间线完成，17份Host发射/设备任务/CSV行数/统计Count一致；设备诊断不替代接口预算。显式batch仍有优化空间。

### 4.2 兼容性分析

保持Python可选batch和四Tensor dispatcher，上游存在时复用schema、追加NPU实现，已验证torch_cluster1.6.3+pt29cpu。普通CSR和非连续输入兼容，empty ptr为NPU内部扩展。技术共存不代表CPU数值差异已获批准。

实现目录为ops-gnn的csrc/npu/nearest、python/ops_gnn/nearest.py、test/nearest，设计归档docs/experiment/design/nearest。本PR只提交设计。设计评审通过并合入后申请验收；后台测试通过通知后再提交代码PR。
