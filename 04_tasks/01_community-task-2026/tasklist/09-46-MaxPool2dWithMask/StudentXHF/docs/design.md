# MaxPool2dWithMask 算子设计

贡献者：StudentXHF；任务编号09-46；版本2026-09-19。本文为前向算子设计，不涉及梯度计算。A2/910B3、CANN9.0.0候选已构建并完成165/165独立公共API精度用例，性能对照仍在进行，310P未验证；本文不表示已通过官方验收。

当前实现方向：在`experimental/pooling/max_pool2d_with_mask`复用既有手写ACLNN的参数校验，保留正式公开函数原型。设备端按256个连续输出点划分所有权，按行搬运、Gather、FP32严格比较并更新int32索引，特殊跨度保留通用设备路径。索引按附件golden连续编码，尾部在设备清零，核心计算不在Host回退。1×1容量不足当前返回错误并明确标为合同缺口，不宣称该行为已经获准。性能测量先固定为5次预热、20次ACL事件采样，计量公开第二段中的设备工作；该诊断口径不等于官方TBE验收口径。

# 需求背景（required）

## 需求来源

[9月社区任务-MaxPool2dWithMask算子开发](https://www.hiascend.com/activities/task-center/details/6e72d6cde6a84ddb867f6b7d599914a5)要求以Ascend C实现与既有算子一致的功能，适配Atlas 800T A2、Atlas 300V Pro，使用CANN9.0.0或9.1.0，最终代码进入`cann/ops-nn/experimental/pooling`。

[原始任务包](https://www.hiascend.com/p/resource/202609/33d5365752d34991b3d90ef52b3fb35a.zip)的SHA-256为`c1fd6e6f07810c820b477e71c203dfe15f1c4c13d71e165c6c45a67ef9f5e0d3`。包内包含任务书、153条用例JSON、golden/shape函数、op配置，未包含TBE性能数据和完整计时脚本。

## 背景与已有实现分析

正式入口是`aclnnMaxPool2dWithMaskGetWorkspaceSize`与`aclnnMaxPool2dWithMask`。以[ops-nn d9ba860版本源码](https://gitcode.com/cann/ops-nn/blob/d9ba8606dd4e9ebaf02b14b7fbf78906d439a6b8/pooling/max_pool3d_with_argmax_v2/op_api/aclnn_max_pool2d_with_indices.cpp)为审计基线：A2非1×1窗口可经reshape走3D int32 argmax路径，310P及1×1窗口走原mask路径。因此公开接口的当前输出布局不能直接视为附件统一argmax语义已在所有分支成立。

附件golden把每个通道的线性位置`ih*W+iw`作为小端int32，将全部N/C输出索引连续写入INT8容器前缀，剩余字节为零。它没有为各通道分别留下完整mask跨度。现有仓内相关ST只返回池化值，不能替代任务要求的完整indices验证。

# 需求分析（required）

## 需求描述

对CHW/NCHW输入逐通道执行二维最大池化，返回池化值及indices；保留公开API、参数检查、非连续Tensor处理、workspace与调用者stream语义。支持合法泛化输入，不能按官方case名称或输入数据进行特殊分派。

| 项目 | 需求与依据 |
| --- | --- |
| shape/layout | 输入3或4维；ND/NCHW；3维视为N=1。任务参数表缺self行且只列4维，但JSON包含30条3维用例，公共API也允许3维 |
| dtype | A2为FLOAT/FLOAT16/BFLOAT16；310P的FLOAT16要求与当前API仅FLOAT检查存在差异，待确认 |
| kernelSize | 长度1或2，元素正数，单值扩为两维 |
| stride | 长度0/1/2；空时等于kernelSize，其他元素正数 |
| padding | 长度1或2，同时满足非负且不超过对应kernel的一半；任务书“或者”按源码联合检查解释 |
| dilation | 长度1或2，值均为1 |
| ceilMode | 支持floor/ceil；ceil时去掉起点完全落入右/下padding的最后一个窗口 |
| 最大值相等 | 按窗口行优先扫描，严格大于才更新，保留首个最大位置 |
| 特殊值 | NaN/-Inf按任务约束不支持；+Inf、有限极值和正负零加入补充测试 |
| 空Tensor | N=0继承原接口空执行；C/H/W不允许0；不静默改变错误码 |
| 非连续 | self与out沿用现有设备端连续化/ViewCopy能力；实际stride及storageOffset需通过公开入口实测 |

原始153条：FLOAT49条、FLOAT16 54条、BFLOAT16 50条；3维30条、4维123条；全部ND、ceilMode=False；窗口仅2×2/3×3/7×7，输入均匀分布[-5,5]。泛化能力不能仅由这153条推定。

## 待评审确认的问题

问题及可复现输入已在[官方讨论评论](https://gitcode.com/cann/ops-nn/discussions/8#discussion-comment-2eaa0ee79a9943709d0f8b442078d223)提出。以下不确定项不改变已明确的数学逻辑，但须在承诺完整平台支持前关闭。

| 问题 | 证据 | 请求确认 |
| --- | --- | --- |
| 310P编码/dtype | 当前ACLNN走mask路径且只允许FLOAT，附件为统一int32字节格式 | 是否统一golden，以及是否允许调整现有派发和dtype校验 |
| 1×1容量 | x=[1,1,8,8]、k=1、s=1、p=0：容器160字节，64个int32需要256字节，原golden实际抛出ValueError | 保留原mask、修订容量公式或另有支持范围；不可自行截断、越界或宣布该输入非法 |
| 性能基线 | 包内无TBE计时数据；当前公共入口可能已调到其他Ascend C实现 | TBE基准入口、工具版本、warmup/次数、计时范围、统计量及95%整体聚合方式 |
| 310P测试资源 | 任务为Atlas300V Pro，而资源平台出现310P3等型号 | 可接受具体型号及其软件栈 |

# 详细设计（required）

## 算子分析

### 数学公式

对输出(n,c,oh,ow)，窗口候选为`ih=oh*sh-ph+kh`、`iw=ow*sw-pw+kw`；仅在`0<=ih<H && 0<=iw<W`时参与比较。padding视作-inf。选出值最大的首个位置，值写入out；golden编码分支的位置为`ih*W+iw`，不包含N/C偏移。

`Ho=floor((H+2*ph-kH)/sh)+1`，W同理。ceilMode使用ceil除法，然后当`(Ho-1)*sh>=H+ph`时减一。计算使用有符号floor/ceil，不能用C++负数截零除法替代。中间乘加、元素数和字节数均做int64溢出检查。

out shape为`[N,C,Ho,Wo]`；indices shape为`[N,C,kH*kW,32*(ceil(Ho*Wo/16)+1)]`；3维输入去掉N维。拟按附件编码时有效前缀长度为`4*N*C*Ho*Wo`，其余字节须显式置零。

### 支持数据类型与精度

A2的FP16/BF16可精确扩为FP32进行比较；FLOAT按FP32比较，再按输入dtype写回选中的输入值。当前源码对FLOAT降为FP16的判断限定原始ASCEND910，不能据此推断A2也降精。310P集合待前述问题确认。

最大值初始化为-inf，不能初始化为有限最小值，否则输入恰好等于有限最小值时严格大于更新可能漏掉真实索引。NaN/-Inf输入不扩展支持承诺。

## 算子实现

### 工程与API整合

拟议代码目录为`experimental/pooling/max_pool2d_with_mask`。依据[既有ACLNN复用说明](https://gitcode.com/org/cann/discussions/37)，复用正式手写ACLNN并使用`ACLNNTYPE aclnn_exclude`，避免自动生成同名接口。保留已有Contiguous、ViewCopy、dtype/shape检查和错误返回语义，改动仅限本算子派发；不改变相邻pooling API。

实际L0名称及挂接位置由评审确认的编码方案确定。附件JSON的op_name不能证明公开ACLNN必然会调用同名自定义kernel。需要Profiler和动态库身份共同确认实际执行路径。

如果要求兼容原TBE mask和int32两种编码，共享最大值与位置计算，分别实现编码写回模块；若全部统一int32，须先确认容量及允许调整的公共派发范围。两种互斥编码不能在同一输入条件下同时宣称满足。

### Host侧设计

1. 将CHW规范为N=1的逻辑视图；校验属性、推导Ho/Wo、out/indices shape及元素/字节数，保留空N语义。Host仅执行元数据和tiling计算，不执行池化。
2. 从平台接口获取可用核数、UB容量。按照实际dtype、窗口跨度和编码临时空间计算tile；不硬编码A2/310P核数或UB大小。
3. 全局输出点按16点粒度划分段；该粒度使16位out与32位indices的分段起点均按32字节对齐。余数分配给前若干核，最后一个核拥有部分尾段。跨NC平面时重新映射窗口，不能按一行连续输入处理。
4. UB预算包括输入搬运缓冲（可双缓冲）、FP32当前最大值、int32索引、比较谓词、dtype转换输出及API临时空间。满足`B_input + 4*T + 4*T + B_output + B_predicate + B_api <= UB_available`后选择T；具体API临时大小在目标CANN版本确认。
5. 先实现对任意合法k/s/p/ceil成立的通用设备路径，再根据真实窗口/步长/尾块特征选择优化路径。候选tiling key划分通用窗口、2×2/s2、3×3、7×7，具体数值与启用条件由构建和泛化测试确定，不能仅为153条用例定义路径。
6. 检查int32位置表示范围及容器容量，超出范围的合同需明确；禁止静默窄化。未确认的1×1规则不作为已实现合法性结论。

### Kernel侧设计

处理流程为CopyIn→Compute→CopyOut。每个输出点只有一个拥有者，无需原子操作。

- CopyIn按窗口覆盖范围搬运连续行，对重叠窗口尽量复用输入行。输出tile跨平面时拆开处理。A2/310P分别确认搬运、尾块和类型转换API支持，不能默认目标平台支持相同DataCopyPad用法。
- Compute使用FP32当前最大值和int32位置向量。按窗口行优先顺序，用严格大于谓词同步选择值和位置；padding候选由有效性谓词屏蔽。相等值保持第一次选择，不做不稳定的并行argmax归约。
- CopyOut按dtype写出值。golden分支的索引全局位置为`4*outputPoint`；完整有效前缀连续，不能按通道mask跨度填充。
- 处理索引尾部时，以硬件写块为单位分配唯一拥有者。有效前缀的最后一个写块由其拥有者补零；纯零尾部从下一个对齐边界分段清零，避免清零与有效索引共享写块。若目标API无法证明该方案安全，使用同stream的独立初始化kernel，再执行索引写回，全部开销纳入计时。
- 最后一个out尾块需使用目标平台合法的尾搬运方案，禁止把上取整写出当成不会越界。异常小shape仍必须正确，后续根据真实profiling决定是否减少核数和启动开销。

### 性能优化方案

先验证通用路径，再逐项测量：减少重叠窗口重复搬运、规则窗口向量化、选择合理核数/UB tile、在收益可证实时启用双缓冲。比较/编码与输入拷贝共同优化，不能只测内部最大值kernel。

7×7时mask零尾部可能显著大于int32有效前缀，清零带宽及框架连续化/ViewCopy都属于真实成本。性能报告同时保留端到端和kernel数据，正式判定按官方确认口径；禁止在计时范围外偷偷清零输出。

### 内存、确定性与异步

无原子写；固定比较顺序，输出写块唯一拥有者。所有设备工作进入调用者传入的stream，Host实现不增加强制同步。测试驱动为了读取结果的同步不进入算子实现。

workspace报告第一段真实返回值。即使核心kernel无需额外GM，框架连续化或转换也可能申请workspace，不能提前宣称零workspace。Host RSS、NPU采样与分配器高水位分别记录，不能相互替代。

## 支持硬件

| 硬件 | 要求 |
| --- | --- |
| Atlas800T A2 | 支持FP16/FP32/BF16；任务性能参考具体写为910B3；按控制台+npu-smi+构建系统确认实际SoC |
| Atlas300V Pro（310P） | dtype与编码待确认；独立采集软件/硬件、精度、耗时和Profiler证据 |

CANN版本为9.0.0或9.1.0。其他任务的A3/9362不能代替两类平台验证，也不能直接复制其构建SoC。构建只针对目标算子，保留实际命令、返回码及编译日志。

## 算子约束限制

遵循任务书与正式API约束，dilation仅1，输入NaN/-Inf不在支持范围。310P的ceilMode限制需按官方接口验证。1×1、索引表示范围等未明确项单列，不自行缩减任务支持范围以规避测试。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 执行方式 | 来源 |
| --- | --- | --- |
| 原始精度 | 原JSON完整153条，不改shape/dtype/输入范围；out按官方精度工具，indices全字节精确比较 | 原包与[生态精度标准](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md) |
| A2性能 | 不低于TBE的95%；基线入口及聚合统计须确认；100us以下场景超过TBE耗时30%时按任务补仿真图和分析 | 任务书性能要求1、2 |
| 310P性能 | 参考同shape的A2/910B3耗时×5；超过参考30%时补仿真证据 | 任务书性能要求3 |
| 泛化/内存 | ceil、空/单值stride、矩形窗口、不规则边缘、tie、极值、非连续、空N、错误参数、stream、尾块与大shape | 任务书泛化要求、公共接口 |

官方JSON的expect_func与shape_depend引用两个不同绝对路径。优先把同内容golden布置到所需位置并原样运行；若必须改路径，保存原始失败、仅路径差异的副本和diff，不改变数据或阈值。

每阶段保存命令、开始/结束、stdout/stderr、返回码、Git HEAD/tree、程序/库SHA-256。顺序为preflight→build→smoke→diagnostic→full→evidence，前一阶段失败时停止，不将未执行用例计为通过。

TBE基线使用未加载自定义库的独立干净进程；候选使用相同入口和输入，统一warmup、采样次数、同步范围与统计量。性能临界项串行跑三轮，以最差轮判定，同一NPU不并发竞争。原始PROF目录保留，证明不存在CPU fallback、错误旧库或额外同步。

## 当前验证状态

已完成本地准备：4275组C++ shape/参数检查、55组分段检查、153条原始参数+72条补充CPU计算与原golden对照；18组序列化/比较器自测；20组Linux CPU ASan/UBSan检查，均通过。1×1容量异常已实际复现。

上述CPU原型不属于最终算子，不会链接为CPU fallback。最终A2/910B3普通构建015完成165/165独立公共API精度检查（153个附件组合+12个补充），完整out/mask按原golden精确匹配，含三种dtype、非连续、ceil/tie、空N、近邻数值。27/27参数拒绝检查符合预期；1×1容量冲突仍是任务缺口。

同源插桩s016在5个用例上运行memcheck/racecheck/initcheck/synccheck，共20次rc=0且候选kernel无ERROR，输出均匹配golden。synccheck存在冗余set_flag警告；SDK StridedSlice另有24字节GM写边界警告，部分内置kernel未插桩，不能声称整图全部检查通过。历史s014发现Gather偏移向量非活动槽位未初始化，015已初始化完整向量并重跑验证。

SDK独立进程165例执行成功，输出值匹配，但162个非空完整mask与附件尾部语义不同。162例各5次预热+20次设备事件采样，取排序第11个；SDK/候选耗时比中位数0.8352258242，85/162低于参考0.95。SDK尚未认证为官方TBE，当前不能宣称性能达标。Profiler已确认候选AI_VECTOR_CORE派发；完整PROF、输入/输出和逐文件哈希证据已归档。

被测实现提交464e4b199aceae4803fd1cae8a14e30b3e5999cf；[首版报告与逐例材料](https://gitcode.com/StudentXHF/ops-nn/blob/feat%2Fmaxpool2dwithmask-studentxhf/experimental/pooling/max_pool2d_with_mask/validation/20260919/REPORT.md)。310P、官方AscendOpTest/TBE、性能优化和适用仿真仍待完成。报名仍审核中，首版个人分支发布及设计PR不等于正式验收提交。

## 兼容性分析

复用既有公开API，维持参数、返回码、dtype/layout、workspace/executor及stream契约；保留非连续输入/输出能力。索引编码或310P dtype需要变更时必须显式评审，不能以“新算子”为由跳过兼容性分析。邻近pooling接口需要回归测试。

## 交付与AI辅助说明

设计评审通过并合入后推进正式实现；验收材料包含最终源码身份、原始测试、精度/性能/内存/Profiler、复现步骤和已验证ZIP。验收通过后按[社区流程](https://gitcode.com/org/cann/discussions/39)提交上游代码PR并检查CLA/CI。

OpenAI Codex（GPT-6）辅助源码审计、设计编写和本地测试。未使用其他参与者代码；引用官方代码遵循其CANN Open Software License 2.0。所有结果按实际执行范围披露。
