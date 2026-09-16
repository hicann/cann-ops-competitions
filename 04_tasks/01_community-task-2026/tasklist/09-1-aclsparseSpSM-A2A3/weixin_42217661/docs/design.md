# aclsparseSpSM A2/A3 算子设计（v0.1，评审稿）

作者：weixin_42217661。任务编号：09-1。

任务：[9月社区任务-aclsparseSpSM算子开发(A2/A3)](https://www.hiascend.com/activities/task-center/details/afcb74da613e40638eb07ca6dcc82286)。
依据：[官方任务书及测试包](https://www.hiascend.com/p/resource/202609/d904f6b5e8674eb0ab5e92ec98f5af59.zip)，2026-09-14下载版本。
代码基线：[cann/ops-sparse](https://gitcode.com/cann/ops-sparse)，bc674eff97365d9f8ab80c8d94b4dba56beb291c。

状态：设计评审稿，尚无新算子的NPU编译、精度、性能和内存结果。下文优化策略均为待验证方案。本次只提交设计，正式实现另交ops-sparse；请优先评审末尾的接口及验收口径问题。

## 1. 需求背景

实现多右端稀疏三角求解 op(A)C = alpha op(B)。A为稀疏三角方阵，B/C为多列稠密矩阵。以任务书的C++ Legacy API要求为工程入口；Python hook用于连接官方测试，不能替代C++公开接口。

基线检查：sparse/spsm/arch35/spsm_host.cpp目前校验CSR、FP32、Host pointer mode、opB=N、opA=N/T；公开头文件的Analysis描述包含CPU格式转换及level scheduling。基线没有SpSM arch22实现及公开UpdateMatrix声明。需要补齐设备路径与接口，不能直接复制arch35目录或只解除校验。

## 2. 需求分析

| 项目 | 范围 |
|---|---|
| 类型 | A/B/C/computeType/alpha一致，FP32或complex64 |
| 稀疏格式 | CSR、CSC、COO；I32设备索引；base0/1 |
| 三角属性 | LOWER/UPPER、UNIT/NON_UNIT |
| 运算 | opA/opB=N/T/H；complex64的H执行共轭 |
| 稠密布局 | B/C支持ROW/COL、leading dimension、多RHS |
| 执行 | handle stream异步；Host/Device alpha；B/C同指针 |
| 多阶段 | Create、BufferSize、Analysis、Solve、GENERAL/DIAGONAL Update、Destroy |
| 确定性 | 未排序/重复坐标确定性处理，同一输入重复执行逐位一致 |
| 禁止项 | CPU fallback主求解、CPU同规模求解缓存、未统计的临时内存 |

BufferSize/Analysis允许B/C values为空，描述符有效；Solve要求有效Device values。NON_UNIT对角必须存在且非零。m、nnz、RHS、布局/ld及空输入的合法组合均需覆盖；非法枚举、维度、类型和索引元数据明确报错。

## 3. 详细设计

### 3.1 工程和接口

新增sparse/spsm/arch22/下Host、Kernel、tiling文件；公开原型更新include/cann_ops_sparse.h。专项C++ UT/ST按任务书放test/spsm/arch22/，实现时核对测试发现规则。顶层CMake已按SOC选择arch22/arch35，接入时核对源文件发现与符号唯一性。

沿用现有CreateDescr、DestroyDescr、BufferSize、Analysis、SpSM函数签名。按任务书补充：

~~~c
typedef enum aclsparseSpSMUpdate_t {
    ACL_SPARSE_SPSM_UPDATE_GENERAL = 0,
    ACL_SPARSE_SPSM_UPDATE_DIAGONAL
} aclsparseSpSMUpdate_t;
aclsparseStatus_t aclsparseSpSMUpdateMatrix(
    aclsparseHandle_t handle, aclsparseSpSMDescr_t spsmDescr,
    void *newValues, aclsparseSpSMUpdate_t updatePart);
~~~

复用项目已有描述符、stream及合法参数工具，遵守公开源码许可。共享接口更改需要arch35编译及接口回归，不能用A2结果代替950回归。

### 3.2 多阶段状态与资源

Created → Sized → Analysed → Solve/Updated → Destroyed。失败调用不提升状态。BufferSize使用受检宽整数计算区间、字节数和对齐，不读取B/C values。Analysis在handle stream上构造设备元数据；Solve在相同stream消费。

Host状态保存A格式/尺寸/索引类型/base、fill/diag、opA/opB、dtype、B/C形状/layout/ld、RHS、设备、描述符身份、workspace绑定和版本。改变结构或相关参数后分析失效，需重新Analysis。并发使用同一描述符的规则明确记录，首版不支持跨stream并发修改同一状态。

descriptor、externalBuffer和相关输入需保持至异步完成。Destroy不能提前释放仍被设备使用的资源。裸指针接口不能可靠识别任意悬空指针或实际分配长度；采用已有运行时机制，无法检测项明确为调用方前置条件，不承诺无实现依据的异常检测。

### 3.3 Device格式规范化

将op(A)构造为逻辑CSR；T/H交换坐标，H共轭values，三角方向随转置反转。UNIT把对角解释为1，不用存储对角做除法。

未排序输入在Device按(row,col,original_position)稳定排序，同坐标按固定原序合并，禁止无固定顺序的浮点原子累加。保留原始条目到合并条目的映射供更新复用。排序/scan临时空间统一计入BufferSize，不隐式借用未统计缓存。

固定输入重复执行要求逐位一致；是否还要求不同输入排列之间逐位一致需确认。重复坐标求和、三角范围外条目、零/缺失对角及NaN/Inf语义须与任务测试口径一致。设备内容错误采用确认后的异步错误协议，禁止每次Solve把全量索引搬到CPU做校验。

### 3.4 求解、tiling与同步

Device Analysis建立三角依赖层级；每层行之间独立，RHS按列tile分配，每个输出元素有唯一写入者。行内按固定非零元顺序累加。FP32用FP32运算，complex64显式实现实虚乘加及除法；高精度CPU计算仅用作Golden。

首版正确性路径逐层下发kernel，stream顺序保证层间依赖。长链矩阵会导致大量launch，这是待解决的性能风险。持久化调度仅在目标设备跨核同步/可见性得到验证后引入，不能直接假设arch35的SyncAll适用于arch22，也不能采用可能因核超额调度而死锁的自旋方案。两路径按层数、层宽、nnz/RHS及实测选择。

运行时查询核数和UB容量；tile大小按输入values/indices、RHS片段、输出累加、临时空间和对齐预算计算。先单缓冲覆盖正确性；双缓冲以实测收益为依据。RHS尾块和ld padding使用有效长度写入，不能越界覆盖。

### 3.5 布局、原地与alpha

按opB及ROW/COL/ld计算逻辑读取偏移。B/C同指针且有转置或布局变化时，不能直接写C覆盖未来仍需的B。首个保守方案在覆盖前快照必要RHS，快照计入workspace；若超过内存门槛，需经证明安全的分块/置换方案，不能声称原地无需额外空间。

Host alpha通过合法异步传参送入kernel，Device alpha由kernel在调用流内读取，Host不得直接解引用Device指针。源A/B只读，C写入行为与允许的alias一致。

### 3.6 UpdateMatrix

GENERAL保持pattern，替换全部原始values，沿稳定映射重新合并并刷新数值、对角及tiling相关状态。结构变化必须重新Analysis。DIAGONAL仅更新约定顺序的对角和派生状态；UNIT和重复对角的解释需明确。

newValues绑定或派生缓冲由描述符管理，保持调用方原matA只读；清楚约定输入生命周期。连续GENERAL/DIAGONAL与Solve交错测试，禁止复用过期数值缓存。接口不含长度参数，对角输入长度/顺序及长度错误检测方法需要评审。

### 3.7 Workspace与峰值

区间包括规范化索引/置换、合并映射、规范化values、依赖层级、对角状态、排序scratch、必要RHS快照、错误状态。按生存期复用，不省略实际同存活区间；记录BufferSize、实测峰值和各型号L2容量。

P-03的全量RHS快照约64MiB（131072×64×8）。因此全量快照不意味着能满足L2门槛。设计阶段把大规模原地布局变换列为风险，需在实现前确认可行内存路径和验收条件。

## 4. 特性交叉分析

| 交叉项 | 处理与验证 |
|---|---|
| T/H × fill | 转置后反转三角方向，H同时共轭 |
| 重复坐标 × Update | 复用原始位置映射，以固定顺序重新合并 |
| UNIT × DIAGONAL | 隐式对角与显式更新冲突，需明确口径 |
| opB/布局 × 原地 | 防止覆盖未读RHS；快照或安全分块计入内存 |
| Device alpha × stream | 不强制Host读回，同stream保持依赖 |
| 公共头文件 × arch35 | 编译及接口回归分别执行 |
| 空values × 多阶段 | BufferSize/Analysis不访问B/C，Solve校验真实指针 |

## 5. 可维可测分析

硬件：910B3、910B4、A3具体型号分别执行功能/精度/性能。CANN9.1.0及后续配套版本，报告SOC与软件版本。单型号结果不复制到其他型号。

官方文件包含200条精度用例和206条性能用例（6条P、200条extra）。补C++ UT/ST覆盖三格式、双布局、N/T/H、ld padding sentinel、两种pointer mode、重复/未排序、B/C同指针、空values Analysis、参数变更、更新序列、资源生命周期和异常。

CPU Golden为float64/complex128。rtol=2^-10、atol=2^-16，匹配率至少0.99，每元素绝对误差不超过max(1e-2,32×ULP(golden))；复数实虚分别判定。NaN/Inf按确认后的规则分类检查，不直接套普通有限值误差公式。

性能：每case预热10、采样30，记录median/P90及Analysis、Update、Solve、workspace和Profiler。目标GPU同调用范围耗时/NPU耗时至少0.25。官方GPU脚本无update接口时计reanalysis+solve，支持时计update+solve；逐条核对update_supported/update_mode，不能拿NPU Solve单段对比另一端全流程。

Python测试hook必须连真实analysis/update/solve三阶段实现，缺少真实hook即失败；禁止用allow-reference-fallback生成性能结论。Profiler需显示本人kernel、公共接口和stream。

当前仅完成小规模CPU语义检查（不包含在本次文档PR中）：3个测试方法通过，其中覆盖LOWER/UPPER×UNIT/NON_UNIT×opA N/T/H×opB N/T/H共36组组合，另含已知多RHS解、重复坐标、base1和非法输入。它们不证明NPU实现、设备精度、确定性、异步或性能；后续另随实现交付正式可复现测试。

## 6. 请求评审确认

1. 内存：任务书为50%，测试README写5%；输入输出小于500MB且workspace超过L2的case如何验收？
2. DIAGONAL：newValues长度、顺序、重复对角及UNIT语义；无长度参数时如何判定长度错误？
3. 索引与数值：重复坐标合并顺序、特殊值/奇异系统及Device异步错误返回协议。
4. 性能：各GPU基线对应update+solve还是reanalysis+solve；补全GPU型号/软件版本和比较口径。
5. 原地与兼容性：B/C同指针时不同布局/opB转置的完整范围，以及arch35回归的资源安排。

在这些问题得到明确前，相应条目保持待验证。本稿按官方任务书与公开主线接口独立整理，不复用其他参与者设计。
