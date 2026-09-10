# segment_csr 算子设计文档（Ascend 950 / arch35）

## 1. 需求背景（required）

### 1.1 需求来源

任务为 [CANN训练营北京邮电大学-segment_csr算子开发(950)](https://www.hiascend.com/activities/task-center/details/41ae8154c3e64bfd8802ff89f6c43dc2)。
依据随附《segment_csr 算子开发任务书》、[专场03任务引导](https://gitcode.com/org/cann/discussions/285)和[设计模板](https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md)提交设计评审。
参考接口为 [torch_scatter.segment_csr](https://pytorch-scatter.readthedocs.io/en/latest/functions/segment_csr.html)，任务建议版本不低于2.1.0，本次CPU直接对照版本为2.1.2。

### 1.2 背景介绍与现状分析

Segment CSR利用预先给定的CSR行指针执行分段归约，用于GNN邻接聚合和稀疏特征池化。
与任意索引scatter相比，CSR段边界已知，输出可以分配给独立核心或线程，避免全局原子争用。
通用SIMT逐元素归约存在候选输入重复读取和循环开销；本设计对满足布局条件的输入采用AIV分块搬运及寄存器归约，保留SIMT处理全部必选dtype和通用形状。

### 1.3 功能分析

提供sum/add、mean、min、max及通用segment_csr接口；段由indptr相邻元素指定。
公共验收入口是PyTorch接口，aclnn层为可选交付，不以CPU回退替代NPU计算。

## 2. 需求分析（required）

### 2.1 需求描述与接口

六个Python函数按任务书保持参数名称、顺序、默认值和返回类型，不增加dim_size，不支持mul。

```python
def segment_sum_csr(src: torch.Tensor, indptr: torch.Tensor,
                    out: Optional[torch.Tensor] = None) -> torch.Tensor: ...
def segment_add_csr(src: torch.Tensor, indptr: torch.Tensor,
                    out: Optional[torch.Tensor] = None) -> torch.Tensor: ...
def segment_mean_csr(src: torch.Tensor, indptr: torch.Tensor,
                     out: Optional[torch.Tensor] = None) -> torch.Tensor: ...
def segment_min_csr(src: torch.Tensor, indptr: torch.Tensor,
                    out: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]: ...
def segment_max_csr(src: torch.Tensor, indptr: torch.Tensor,
                    out: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]: ...
def segment_csr(src: torch.Tensor, indptr: torch.Tensor,
                out: Optional[torch.Tensor] = None,
                reduce: str = "sum") -> torch.Tensor: ...
```

| 参数 | 方向和类型 | 形状与约束 |
| --- | --- | --- |
| src | 输入Tensor；FP16/FP32/BF16/INT8/UINT8/INT32/INT64 | NPU strided张量，支持非连续 |
| indptr | 输入Tensor，INT64 | 1 <= indptr.dim() <= src.dim()；末维长度至少1；前导维为1或匹配src |
| out | 可选输入/输出Tensor，与src同dtype、同设备 | src形状的归约维替换为indptr末维长度减1；支持非连续视图写回 |
| reduce | 字符串 | sum/add/mean/min/max；mul或其他值由Python抛ValueError |
| arg_out | min/max子接口输出，INT64 | 与out同形状、同设备，表示src归约维内的行号；通用入口不返回它 |

### 2.2 需求拆解

| 要求 | 设计落实 |
| --- | --- |
| FP16/FP32/INT32/INT64功能与性能 | AIV寄存器路径及SIMT通用路径；完整464项性能矩阵 |
| BF16/INT8/UINT8功能 | 必选NPU SIMT路径，不要求这三种类型的任务书性能达标 |
| indptr广播、空段、非连续、out | Host统一维度展开和工作缓冲；设备处理真实段边界 |
| 确定性 | 每个输出单独归属，固定归约及相等值首次索引规则；无全局原子 |
| 精度 | 任务书指定CPU torch_scatter单标杆；已发现的参考差异见3.2.4及4.1，需明确评审 |
| 交付 | 设计文档、用例及代码、含参数和截图的报告、复现步骤、可访问的个人代码仓 |

## 3. 详细设计（required）

### 3.1 算子分析与数学公式

令dim=indptr.dim()-1，M=src.size(dim)，第s段J_s=[indptr[s],indptr[s+1])，L_s为段长。批维及通道维固定后：

- sum/add：out_s = sum(src_j, j in J_s)。
- mean：非空段对段内和及段长执行源dtype规定的转换后相除；空段按零值处理。
- min/max：out_s为段内极值，arg_s为该极值首次出现的归约维行号。
- 非空src中的空段输出零，arg_s=M；src整体为空且用户提供out时保留其值，详见Host处理。

浮点算术顺序、half舍入和整数回绕/向零截断影响实际结果，不能仅用实数公式替代实现约束。
无dim_size，无mul；indptr前导维广播，src/out非连续可规整后计算。

### 3.2 算子实现

#### 3.2.1 Host侧设计：维度、输出及indptr校验

```text
dim  = indptr.dim() - 1
E1   = prod(src.shape[:dim])
M    = src.shape[dim]
K    = prod(src.shape[dim + 1:])
nSeg = indptr.shape[-1] - 1
src[e, j, k] 地址 = ((e * M) + j) * K + k
```

indptr 前导维全为 1 时共享一份指针；否则通过 expand + contiguous
物化为每批一份。非连续 src、indptr 在当前设备上转为连续张量。
out 形状与 src 相同，归约维替换为 nSeg；提供的 out 校验 dtype、
device、shape。非连续 out 通过连续工作缓冲计算后复制回原视图。
src 为空时，未提供 out 则初始化零值，已提供 out 则保留原值；
min/max 始终返回值与索引两个张量，包括零段、零批次和零通道输出。

校验维度关系后才读取 shape。算子使用 src 所在设备的当前 NPU 流，
不创建临时流、不进行 host 流同步。导入模块不启动设备 kernel；
生产入口不需要首次重复发射。连续输入可直接使用原存储，
非连续输入、广播物化和非连续 out 写回会产生实际设备拷贝。

**indptr校验边界：**任务书合法输入要求末维非降序且0<=ptr<=M；首端为0、末端为M是常见形态，不擅自收窄为唯一合法形态。
Host当前检查dtype、device、维数、广播关系、末维长度及out规格。一般路径在设备裁剪访问边界；均匀段优化对其使用的整个缓存序列验证等差关系及范围。
这不是对任意非法indptr的全量校验或统一报错机制，不将优化路径验证描述为所有输入均已完成单调性扫描。
合法CSR为当前计算契约；若评审要求非法数值指针必须同步抛异常，应在产品实现中补充并重新验证完整接口开销。


#### 3.2.2 计算路径与分派

Host 将满足以下条件的输入交给 AIV 寄存器 kernel：

- dtype 为 float32、float16、int32 或 int64；
- E1 == 1 且共享 indptr，M 非零且不超过 INT32_MAX；
- 每行字节数为 32 的倍数，K <= 1024；
- int64 min/max 额外要求每行字节数为 256 的倍数。

其余输入走 SIMT，包括 bf16/int8/uint8、批维广播、逐批指针、
奇形 K、超出寄存器路由边界；空 src 在 host 按输出契约处理。
路由条件是实现约束，不要求用户将通用输入预先填充成特定 K。

##### AIV分块、UB与指针规划

每核获得一段连续的输出段编号。核数来自 `GetCoreNumAiv()`，
UB 容量来自 `GetCoreMemSize(UB)`。输入块预算为：

```text
tileBytes = min(64 KiB, floor((UB - 80 KiB) / 2 / 256) * 256)
maxRows = tileBytes / (K * sizeof(T))
maxSegments = 2048 / K
```

TPipe 在 kernel 入口创建并传给计算类。UB 分配如下：

| 缓冲 | 用途 |
| --- | --- |
| 两个 input queue slot | 各 tileBytes + 512B，供搬运与计算重叠 |
| saved | 长段跨块累加值；half 使用 float32 |
| savedIndex | 长段跨块的 int32 绝对行号 |
| result / argResult | 最多 2048 个输出元素及 int64 索引 |
| ptrCache | 最多 2048 个 int64 CSR 边界 |

读取至寄存器的尾部区域具有额外可读空间；存储始终受有效 lane mask
控制。GM 与 UB 之间统一使用 `DataCopyPad`。
TQue 管理 input slot 的生产、消费和复用；计算结果完成后通过
V→MTE3 事件回写，MTE3→V 事件保护输出缓冲复用。

##### 均匀段识别

仅当一核的所有边界均可缓存、边界范围合法时，才尝试验证整个指针序列
是否为 `base + index * length`。设备端寄存器校验逐块检查每个 int64
边界的高低 32 位，并汇总错误标志，包含最后不足 64 个边界的部分块。

验证通过后，规划器使用已证明的等差序列直接计算段边界与组大小。
不能缓存全部边界或校验失败时，逐段读取实际指针、裁剪边界，只有
连续且长度相同的段才合并搬运。不能仅凭首尾边界或 shape 推断均匀。

长度超过 maxRows 的段拆成多个块，通过 `first / last` 管理 saved
状态。空段不读取输入，仍写零输出及 M 索引。段组与输入块均受 UB
预算和输出容量约束。

上表对应实际InitBuffer字节账本（s=sizeof(T)，a=sizeof(RegisterType<T>::Value)）：

| UB分配 | 字节数 |
| --- | ---: |
| 两个input slot | 2*(tileBytes+512) |
| saved | 2048*a+256 |
| savedIndex | 2048*4+256 |
| result | 2048*s+256 |
| argResult | 2048*8+512 |
| ptrCache | 2048*8 |

FP16的a=4；FP32/INT32的a=4；INT64的a=8。tileBytes=65536时，各总分配分别为186624、190720、190720、207104字节。
这些请求已按至少32B对齐；额外可读padding已计入，80KiB是当前规划的保守预留，不是额外重复申请。
平台参数从API查询，当前实现及UB预算针对Ascend950。
该PyTorch扩展通过Host条件及Kernel内部模板/分支分派，不使用GE tilingkey。


#### 3.2.3 Kernel侧设计：寄存器归约与同步

通用寄存器函数以通道块处理数据：值常驻 RegTensor，不在每一行之间
反复往返 UB。较长块使用四条固定行序链隐藏依赖，最后按固定顺序合并；
短块使用较小循环。K=32 的适用完整段可以将相邻两行装入 64 个
float32/int32 lane，再在寄存器内合并两半。

`segment_csr_short_sum.h` 为完整的 2/4/8 行短段提供 sum/mean 特化，
支持 K=64/128/256 与四种性能 dtype。它按连续输出每次处理 64 个
元素，用编译期行数展开装载与求和，减少循环与地址计算。
浮点 mean 在输出 dtype 舍入后以 0.5/0.25/0.125 进行二进制精确缩放；
整数 mean 使用整数域移位与负数偏置，实现精确的向零截断。
其他段长使用通用归约与除法。

`segment_csr_narrow_half_sum.h` 处理 fp16、K=32、完整段长 16/32/64
的 sum/mean。每个 float32 寄存器容纳两行，四条累加链直接以输入值
初始化，其余装载与加法按编译期行数展开，最后通过寄存器重排合并
两半通道。求和保持 float32 累加；mean 先舍入 half 和，再执行对应
二次幂段长的精确缩放。其他形状和分块续算继续使用通用路径。

sum/mean 的值按固定顺序累加。min/max 的每次合并同时处理值与行号：
值严格更优时更新，相等时选择较小的原始行号。该规则适用于四路合并、
通道打包合并和跨块续算。

##### fp16 极值

完整、非空且满足打包条件的段使用 `segment_csr_half_extrema.h`。
K=32/64 可在 128 个 half lane 中分别并行处理 4/2 行，K=128/256
按完整向量宽度处理。每条行链进行严格比较；打包链合并时显式处理
相等值的首次行号。段内索引使用 uint16，路由要求段长 <= 65535，
输出时加绝对行基址并扩为 int64。该路径只比较和选择已有 half 值，
不进行 half 求和或复杂算术。

不满足完整段、长度或打包条件时，通用寄存器路径将值提升为 float32
后比较。fp16 求和始终在 float32 中累加。

##### int64 极值

`segment_csr_int64_extrema.h` 采用单条顺序归约链。一次
`DIST_DINTLV_B32` 装载将 int64 拆为两组 32 位寄存器：
高位按 int32 比较，高位相等时低位按 uint32 比较。
这一字典序保留完整 int64 范围，不转换为浮点数，不依赖输入值较小。

值相等时不更新，所以保留先前的首次索引。跨块加载 saved 值和索引，
从本块第 0 行继续；中间块保存状态，末块以交错存储写回 int64 值与索引。
int64 极值因此不需要通用四路链的 64 位比较与索引合并开销。

#### 3.2.4 数值语义、SIMT与待确认差异

浮点 mean 按真实 CPU torch_scatter 的顺序执行：

1. 先完成归约，将和舍入或回绕到 src dtype。
2. 段长也转换到 src dtype，再将分子与分母转到 float32。
3. 执行 float32 除法，输出转换采用对应浮点舍入。

因此 fp16 的段长 2049 会先舍入为 2048，65520 会舍入为 infinity；
这属于 CPU 参考的实际行为。整数 mean 则在源 dtype 中回绕和与段长，
再使用精确整数除法向零截断，不经 float32 中转。二次幂正段长可通过
负数偏置与算术右移等价实现，其他情况使用整数除法。int8/uint8 的段长
转换若回绕为零，CPU 的整数除零行为未定义，本实现明确返回零避免设备异常。

任务包附带 Python golden 的大整数 mean 和空段 arg 与编译后的
CPU torch_scatter 存在差异。原始样例文件保留不改，另加直接调用真实
CPU 扩展的对照测试覆盖上述语义，相关差异在自测报告中单独记录。

ATK 补充对照发现 CPU torch_scatter 2.1.2 的 uint8 max 索引边界：
src=[1,7,2,0]、indptr=[0,0,3,4] 返回 arg=[4,1,1]，最后一个索引
不在其对应段内。本实现按首次索引定义返回 [4,1,3]，该差异提请
设计评审确认。ATK 对照使用源 dtype 调用 CPU 接口，避免工具默认
升精度改变 fp16/bf16 mean 的舍入语义；最终补充检查为 34/35，
剩余一项即上述 uint8 max 参考索引差异，不计入通过项。

SIMT 以输出元格或 16B 通道包为工作项，每个线程独占自己的输出。
适用 dtype 的对齐包使用 16B 装载；其他 dtype/形状逐元素处理。
fp16/bf16 提升到 float32，整数使用 int64 累加；fp16 位编解码覆盖
subnormal、infinity、overflow 和 round-to-nearest-even。
SIMT 浮点 mean 同样先将和与段长转换到 src dtype，再转回 float32
执行除法，避免窄通道回退与寄存器路径采用不同的舍入顺序。
浮点和整数边界、mean 的 CPU 语义均由独立回归用例核验。

浮点 NaN 的极值传播未纳入当前兼容性契约。合法 CSR 指针非降序，
kernel 的范围裁剪并不等于完整的无效 CSR 参数检查。

### 3.3 支持硬件与软件

| 项目 | 要求与验证边界 |
| --- | --- |
| 硬件 | Ascend950系列；当前实现arch35，其他芯片未声明验证 |
| CANN | 任务书9.1.0及以上 |
| PyTorch | 任务书2.7及以上；自测2.9.1+cpu |
| torch_npu | 与PyTorch/CANN配套；自测2.9.1 |
| CPU标杆 | 实际编译的torch_scatter 2.1.2，报告记录动态库身份 |

### 3.4 约束与限制

任务书七种src dtype和INT64指针为正式接口范围。SIMT工作项小于2^32，通道数及段数受Host的uint32边界约束；AIV内部行索引上限和对齐条件见3.2.2，不满足时转SIMT。
NaN极值传播未纳入当前声明；int8/uint8 mean的源dtype段长回绕到零时采用返回零的防护，CPU除零不定义可移植结果。
uint8 max的CPU索引差异需要评审明确确认处理口径。

## 4. 可维可测分析

### 4.1 精度标准、性能标准及来源

| 验收标准 | 口径 | 来源 |
| --- | --- | --- |
| 浮点及窄整数功能 | 通过PyTorch接口与真实CPU torch_scatter比对；浮点按单标杆精度标准 | 任务书“精度要求”；[生态算子开源精度标准](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md) |
| INT32/INT64 | 与CPU bit-wise一致；mean在整数域处理 | 任务书精度表 |
| min/max | 同时检查值和arg_out，空段及相等值规则明确 | 任务书接口及精度表；3.2.4记录例外 |
| 性能 | 每项标杆耗时/NPU完整接口耗时>=0.45 | 任务书29形状×4dtype×4归约=464项 |
| 可复现性 | 版本、环境、源码/二进制哈希、用例参数、步骤和结果截图 | 任务书四类交付件 |

已有完整仓库自测记录如下，均为开发自测，不代表官方后台验收通过：

| 自测范围 | 结果与证据边界 |
| --- | --- |
| 原始样例 | 651项通过；原始样例与benchmark文件未修改 |
| 完整功能 | 1377项及16项子测试通过；包含原始样例，不将子集重复相加 |
| 真实CPU库直接对照 | 282项，属于完整功能子集；不能外推为464项性能均直接对照原库 |
| 全部性能矩阵 | 464/464项达到0.45，最低比值0.472740；NPU计时实测，基线取任务书表 |
| ATK补充 | 34/35；uint8 max的索引差异未隐藏，见3.2.4 |

现有pytest通过只证明其声明的断言和容差通过，不将其自动解释为生态精度标准所有度量均已核验。
报告应分别列用例参数、精度指标/结论及截图、性能数据及截图，并提供复现步骤。
对应产品源码固定版本为5cc55204ba2693ccc195dcc89fc4e1bc6db4759f；与实测源码之间的注释/版权清理差异已在provenance中逐文件记录。本次设计文档更新复核已有测试记录，未重新运行NPU测试。

### 4.2 兼容性、构建与验证方案

交付采用完整 ops-gnn 仓库，加载 CANN 环境后执行
`python3 setup.py build_ext --inplace`。CSR 与样例所需的 COO 适配器
由主仓统一编译和注册；使用 `_pybind.so` 与
`libopsgnn_npu_kernel.so`。开发阶段独立工程的动态库不能替代
完整仓库的集成检查。主仓命令见
[目标代码仓](https://gitcode.com/cann/ops-gnn)。

完整仓库通过 `_pybind.segment_csr / segment_coo` 连接 Python 接口。
附带的 COO 接口仅作为 CSR 一致性辅助适配，非空输入时 out 按 CSR
目标缓冲覆写，不参与初值归约；这与原生 CPU COO 的 out 初值语义不同，
不属于完整 COO 契约交付。52 项适配测试验证该辅助接口自身行为。
公开 `segment_max_csr` 对 int32 指针保留旧 kernel、仅返回值、初始输出
参与最大值比较的行为；对 int64 指针调用新算子族并返回值与索引。
六个新 CSR 接口保持任务书的精确签名，不使用额外关键字参数或特殊
默认值。历史 `optional_out=` 仅由显式导出的 `segment_max_csr_legacy`
接受，该别名直接引用原有 int32 接口函数；新 `segment_max_csr` 保留
int32 指针的第三个位置参数与 `out=` 转发。开发工程的 `_C` 包装接口
保持任务书原型。

验证分为四组：

- 功能回归：官方、扩展、向量回归与 out 契约，覆盖全部 dtype、广播、
  不均匀/空/长段、首现索引、确定性、内部指针扰动、数值边界和非连续 out。
- 真实 CPU 标杆：加载 torch_scatter 2.1.2 的 `_segment_csr_cpu.so`，
  直接核验值与索引；记录实际动态库哈希，不以近似 golden 替代。
- 任务书验收：`run_segment_csr_acceptance.py --all`，464 例逐例使用自写CPU PyTorch组合参考验证
  完整值与索引，并用完整接口计时判定0.45门槛；这不是464项逐例调用真实CPU torch_scatter。
- Profiler：`profile_segment_csr.py`，固定 warmup=5、active=5，
  自定义实现与同设备 NPU 张量组合分别采集，汇总
  `op_statistic.csv` 的 `Total Time(us)`。它解释设备耗时，不替代任务书验收。

当前运行的 Markdown 报告记录源码与实际加载二进制哈希、设备、
软件版本、计时参数和逐例状态。阶段性精度通过或 quick 子集结果
只能证明相应构建与用例；最终状态必须来自完成的全矩阵报告。

### 4.3 代码归档与交付流程

产品目标仓为[ops-gnn](https://gitcode.com/cann/ops-gnn)：Host在csrc/npu/segment_csr/op_host/，Kernel在op_kernel/arch35/，Python在python/ops_gnn/segment_csr.py，测试在test/segment_csr/，设计归档在docs/experiment/design/segment_csr/。
个人待验收仓为[ops-gnn-segment-csr](https://gitcode.com/Starlink_/ops-gnn-segment-csr)，分支submit-segment-csr-950。

本PR只提交设计文档。设计评审通过并合入后，补齐合入截图、测试用例、报告和测试步骤，在hiascend正式提交验收；提供代码仓、分支及算子目录并确认Ascend-CANN开发者访问。
后台测试通过通知后，再按活动流程提交需求issue及产品代码PR。代码自测、设计批准、设计合入与官方验收是不同状态。
