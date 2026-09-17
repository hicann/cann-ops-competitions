# aclblasIcamin（Ascend 950PR）设计文档

任务编号：09-58；团队：gcw_F4nbZni2；实现版本：candidate-007；2026-09-16。
本文按官方设计模板组织，提交内容为设计说明。开发自测结果用于支持设计评审，
不代表社区验收通过；文末列出需任务研发或验收负责人确认的语义和性能范围。

# 需求背景（required）

## 需求来源

官方任务：9月社区任务-aclblasIcamin算子开发（950），任务ID
1434c269e41b494b8b48d00063e58d45。
任务原件ZIP SHA256：6f658e1116f383ac94f6bb2264c7169285837199bfcb9bda2a0a11b559ff313d。
设计模板与任务目录已核对至 cann/cann-ops-competitions commit bad7e130f01673fbcccc55e4384261b337a0b357。
任务列表的“9月份发放的任务”第58项对应本任务。
设计模板路径：04_tasks/01_community-task-2026/resources/design_template.md。
模板Git blob原始字节SHA256：a6ed8b29fbe1850499e1f5bb08db951cd6d75cac00dddda692c86b54b97114d5。

## 背景介绍

ops-blas基线7eae2328a65753bf55cffc489253eb434ea3317e已有实数isamin，缺少本任务要求的复数Icamin公开接口。
新增接口计算复数向量的最小L1模位置，供BLAS调用方通过统一公共头与句柄stream使用。
实现位于blas/iamin/arch35，测试位于test/iamin/icamin/arch35。

### 功能分析与参数

| 参数 | 位置/类型 | 含义及约束 |
| --- | --- | --- |
| handle | Host，aclblasHandle_t | 有效库句柄，提供stream与已有workspace；空句柄返回HANDLE_IS_NULLPTR |
| n | Host，int | 逻辑复数数目；负数INVALID_VALUE，零为quick return |
| x | Device，const aclblasComplex* | 实虚部均FP32、交错存储；正常计算时不可为空 |
| incx | Host，int | 逻辑元素间隔；正数计算，零或负数quick return |
| result | Device，int* | INT32一基索引；quick return为0；空指针INVALID_VALUE |

# 需求分析（required）

## 需求描述

新增五参数公共接口：

```cpp
aclblasStatus_t aclblasIcamin(aclblasHandle_t handle, int n,
    const aclblasComplex* x, int incx, int* result);
```

正常计算返回FP32 abs(real)+abs(imag)最小元素的位置，相同模取最早位置。
quick return必须写Device结果0，且不触发计算kernel。
目标硬件Ascend950PR，工具链CANN9.1.0；接口声明放入include/cann_ops_blas.h。

## 需求拆解

1. Host校验错误码、物理跨度和workspace，复用句柄stream。
2. SIMD求复数L1模并归约；处理stride、未对齐GM地址、尾部及跨核tie。
3. 保持全局索引为整数，避免超过2^24后的float索引精度丢失。
4. 使用真实Device结果、独立CPU golden与原始1000精度/200性能CSV验证。
5. 三个任务书明确性能均值上限24.77/24.59/29.69μs；先warmup，有效样本数大于50。
6. 保留任务书自身Q7待确认状态，不以暂定测试代替官方语义确认。

# 详细设计（required）

## 算子分析

### 数学公式

令逻辑元素i对应x[i×incx]，i从0开始，m_i=FP32(|Re(x_i)|+|Im(x_i)|)。
有限值正常路径返回1+min{i | m_i=min_j m_j}。
NaN暂按任务书文字处理：首逻辑元素模为NaN时采用FLT_MAX基准，其他NaN跳过；
首NaN与Inf、全NaN等规则仍等待任务书Q7最终确认。溢出得到的Inf保持可参与比较。

### 支持数据类型与形状

输入COMPLEX64交错实虚FP32，输出单个INT32；逻辑一维，不涉及广播。
正常物理跨度为1+(n−1)×incx个复数。Host以64位验证其字节跨度不超PTRDIFF_MAX。
调用方负责提供足够的Device内存；Host不访问输入内容。

## 算子实现

### Host侧设计

检查顺序为handle、负n/result空、quick return、正常x空、跨度。
quick return通过aclrtMemsetAsync在句柄stream异步写4字节0；失败映射为EXECUTION_FAILED。
正常路径读取平台AIV核数。可用核为0返回INTERNAL_ERROR；所需workspace不足返回EXECUTION_FAILED。
不在调用过程中分配设备内存、替换调用方workspace或同步stream。

#### 分核与tiling

tile固定8192复数；cores=min(availableAivCores,ceil(n/8192))。
每核基础长度floor(n/cores)，最后一核处理余数；传递totalN、incx、perCoreN、useCoreNum。
本次950PR实测56AIV核。采用直接C++启动参数结构，不使用aclnn注册或通用tilingkey机制。
单核直接写结果，多核先写局部结果，再在同stream启动一个merge核。

#### 缓冲与内存规划

| 缓冲 | 字节 |
| --- | --- |
| 双输入队列 | 2×8192×8=131072 |
| real、imag、局部索引 | 3×8192×4=98304 |
| 比较mask | 1024 |
| reduction临时区 | 2048 |
| 输出临时区 | 32 |
| 总计 | 232480 |

平台API报告可用UB253952B，剩余21472B。此实现仅对本次目标950PR配置验证。
多核workspace每核32B，只使用8B值/索引字段；56核需要1792B，使用句柄已有workspace。
不同stream应使用不同句柄，并确保workspace互不重叠；不保证同一workspace上的无序并发安全。

### Kernel侧设计

Init建立GM映射与缓冲，并生成0至8191的float局部索引，均可精确表示。
Process先搬第一块，每轮预取下一块，再计算当前块，最多保留两个输入tensor。

1. CopyIn：incx=1用DataCopyPad；incx>1用arch35 Compact聚集复数块。
   每批最多4092块；满tile分4092+4092+8，下一UB起址满足32B对齐。
2. MakeMagnitude：DeInterleave拆实虚，Abs后Add。比较长度向上取整至64个float；非整齐尾部初始化扩展区，归约仍只处理真实count。
3. ReduceTile：Compare/Select屏蔽NaN后ReduceMin取值；再次比较原始模并归约局部索引，避免把无效NaN选为合法Inf位置。
4. 若已有有效best且新tile最小值不更小，可跳过第二次索引归约。tile按全局顺序遍历，因此相等值不会替换更早索引。
5. imag完成最后一次读取及vector barrier后，复用为Select输出；队列和显式事件保证搬运、vector、scalar与输出的依赖。
6. 全局索引始终uint32，0xffffffff为无效哨兵。多核各写32B独立slot，merge按有效性、值、索引选择最终结果。

merge中的标量循环只访问有界的每核元数据，不以标量遍历输入向量。
同核8192+1及16384+1尾部、stride1/3、4B偏移和跨tile tie均有定向测试。

## 支持硬件

| 芯片版本 | 本次验证 |
| --- | --- |
| Ascend 950PR / CANN9.1.0 | 支持，单设备0实测 |
| 其他产品线 | 公共API声明共用；本任务未验证实现支持 |

## 算子约束限制

不涉及广播、负步长反向遍历或Host结果指针模式。零/负incx均按任务书quick return。
同stream顺序执行；异步调用结果须由调用方按stream/event同步后读取。
NaN/Inf最终官方语义尚未闭合；未宣称与cuBLAS所有特殊值/错误参数组合完全等价。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 方法与结果 | 来源 |
| --- | --- | --- |
| 精度 | INT32精确一致；1000原CSV+12补充测试通过 | 任务书§3.2、verify-007 |
| 输入分布 | 405官方随机shape各配uniform/normal，另39对补充；实虚独立 | 任务书§3.5、分布日志与源码 |
| quick return | Device哨兵通过；profiler为11内存任务、3占位、0计算任务 | 任务书§2.4、profile-007 |
| 性能 | 三轮每轮200case，20warmup+100样本/项；三个明确门槛每轮通过 | 任务书§3.3、原始60000设备样本 |
| 内存数据 | UB/workspace公式及四规模各100次设备空闲量观测 | 任务书§4；§3.4无数值门槛 |

性能事件覆盖一次API的全部kernel；输入生成、分配、H2D、golden和D2H不在区间内。
三轮明确case均值范围分别为12.01391–12.45880、15.61782–16.16015、21.99076–22.45608μs。
其余197项正式口径UNKNOWN；首轮TC_PF_1157超过附件参考值，后两轮未超参考，原始慢样本保留。
内存仅为整设备空闲快照，不是进程峰值或无泄漏证明。

## 兼容性分析

公共API扩展与iamin下新增实现，不修改共享依赖或原isamin逻辑。
相同62项isamin对照，原库48失败、候选47失败，未观察新增失败；不能称原算子整体正确或回归全通过。
isamin在两个独立测试套件进程中仍有相同失败，根因未定。

候选源码SHA256：1dc39ae555d710d7aafd01a0bc02ebc358b1027d1bb7f9bd8fe868f78b1fa31e。
被测libops_blas.so SHA256：8ddf5d9adc72a0518aeff78f3028531139769e4e1ec3ead485139846f152c159。
完整源码、构建命令、原始CSV/XML/样本日志、环境与二进制哈希已保留于自测证据包，
作为后续代码评审和验收交付材料。本设计PR仅提交此文档，未随附源码、二进制或完整原始日志。

## 明确性能门槛的实测结果

Ascend 950PR、CANN 9.1.0、单设备0。每轮完整200项，每项20次预热、100次设备事件样本，
取全部样本的算术平均。计时包含一次API调用在同stream上的全部kernel。

| n，incx=1 | 任务书上限 μs | 第1轮 μs | 第2轮 μs | 第3轮 μs |
| --- | --- | --- | --- | --- |
| 1048576 | 24.77 | 12.41158 | 12.45880 | 12.01391 |
| 2097152 | 24.59 | 16.16015 | 15.61782 | 15.83088 |
| 4194304 | 29.69 | 22.45608 | 21.99076 | 22.07327 |

三个明确门槛每轮均通过。其余197项保留参考比较，不声明已获正式性能验收。
TC_PF_1157（n=124122，incx=1）三轮均值为20.33085、8.70494、8.87125 μs，
附件参考值按ms×1000/0.4换算为17.575 μs。首轮超出参考值，未删除慢样本或用后两轮覆盖。
该波动原因尚未确定，不能直接归因于设备环境。

## 请求评审确认的事项

### Q7：NaN/Inf最终语义

任务书§3.2注明待研发确认。本实现暂按“首元素模为NaN时基准设FLT_MAX，后续NaN跳过”，
保留初始一基索引1。请确认该规则是否为最终验收标准。
以下x为复数向量，写成(a,b)表示实部a、虚部b；下表是当前实现和golden的暂定结果。

| x，incx=1 | 当前一基索引 | 待确认点 |
| --- | --- | --- |
| [(NaN,0),(Inf,0)] | 1 | 首NaN的FLT_MAX基准优先于Inf |
| [(NaN,0),(FLT_MAX,0)] | 1 | 与FLT_MAX相等时保留首索引 |
| [(NaN,0),(NaN,0),(NaN,NaN)] | 1 | 全NaN的约定结果 |
| [(Inf,0),(NaN,0),(FLT_MAX,0)] | 3 | 跳过NaN并选择有限最小模 |
| [(FLT_MAX,FLT_MAX),(FLT_MAX,0)] | 2 | FP32模溢出为Inf仍参与比较 |

若最终规则变化，将同步修改特殊值处理、CPU golden与定向测试，并重跑受影响精度及性能验证。

### 其余197项性能的验收范围

请确认任务书§3.3中三项明确门槛之外的“更多参考性能用例”，是否也要求逐项满足
gpu_baseline_ms×1000/0.4；若属于硬门槛，请同时确认计时范围和多轮结果波动的判定规则。
当前采用同stream全部kernel的设备事件平均值，预热20次、有效采样100次，全部原始样本保留。

## 参考来源

- [官方任务页面](https://www.hiascend.com/activities/task-center/details/1434c269e41b494b8b48d00063e58d45)
- [官方任务附件](https://www.hiascend.com/p/resource/202609/ffa2213abf8d4ea697af5fa46c9047f9.zip)
- [本次核对的任务列表](https://gitcode.com/cann/cann-ops-competitions/blob/bad7e130f01673fbcccc55e4384261b337a0b357/04_tasks/01_community-task-2026/docs/README.md)
- [设计模板](https://gitcode.com/cann/cann-ops-competitions/blob/bad7e130f01673fbcccc55e4384261b337a0b357/04_tasks/01_community-task-2026/resources/design_template.md)
- [ops-blas实现基线](https://gitcode.com/cann/ops-blas/tree/7eae2328a65753bf55cffc489253eb434ea3317e)
