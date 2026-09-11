# aclblasCgetrfBatched 算子设计文档

本文定义 <code>aclblasCgetrfBatched</code> 在 Ascend 950PR 上的接口
语义、Host 校验、Ascend C Kernel 方案、性能目标和测试判定规则。测试
章节只描述待执行的测试内容、测试方法和判定标准。

| 项目 | 内容 |
| --- | --- |
| 任务名称 | 8 月社区任务：aclblasCgetrfBatched 算子开发（950） |
| 算子接口 | <code>aclblasCgetrfBatched</code> |
| 目标硬件 | Ascend 950PR，<code>arch35</code> |
| 目标软件 | CANN 9.1.0 |
| 数据类型 | <code>complex64</code>，实部和虚部均为 <code>float32</code> |
| 参考接口 | cuBLAS <code>cublasCgetrfBatched</code> |
| 数学参考 | Netlib LAPACK <code>cgetrf</code> 语义 |
| 算子目录 | <code>blas/getrf_batched/arch35/</code> |
| 测试目录 | <code>test/getrf_batched/cgetrf_batched/arch35/</code> |
| 文档版本 | V2.0 |
| 提交作者 | 待提交人补充 |
| 设计日期 | 2026 年 9 月 10 日 |

## 一、测试主线与判定总览

本节先给出验收主线，再展开接口、实现和专项测试细节。设计书将所有
可观察行为落到接口状态、Device 输出、数值误差、设备计时和资源约束
上；测试执行只依据后续章节的用例、方法和标准。

| 验收门类 | 将测试什么 | 怎么测试 | 用什么标准判断 |
| --- | --- | --- | --- |
| 接口行为 | 函数签名、参数校验、quick return 和 stream | C++ GTest 调用公共 API，检查状态和 Kernel 事件 | 公共状态码、空工作和异步规则一致 |
| 分解功能 | complex64 列主序批量 LU | CPU Golden 与 Device Aarray/Pivot/info 逐批次比较 | L/U、1-based pivot 和 info 满足接口契约 |
| 模式覆盖 | PIVOT、NO_PIVOT、奇异和混合 batch | 分别构造矩阵、模式和奇异位置 | 不发生错误分支串用，info>0 仍是合法分解状态 |
| 边界存储 | n、batchSize、lda、尾块和 padding | 尺寸扫描、保护值、空指针和非法参数用例 | 返回指定状态，逻辑区正确，padding 不改写 |
| 数值精度 | 非奇异 L/U，奇异 info/pivot，特殊值 | 复数实部/虚部分开比对 CPU Golden | FLOAT32 rtol/atol、匹配比例和误差上限满足要求 |
| 性能资源 | 三组固定 case、扩展规模、内存和异步 | warmup、设备事件、基线关联和分配统计 | 固定 us 上限、倍率、采样数和资源门槛满足要求 |

## 二、测试设计与判定标准

本节只定义将测试什么、如何执行以及什么条件下判定符合要求。测试工程
使用 CSV 描述用例，由 ops-blas C++ GTest 调用公共接口。

### 2.1 测试环境和工具

测试环境固定为：

- Ascend 950PR，<code>arch35</code>；
- CANN 9.1.0 及配套驱动、固件和编译器；
- ops-blas 对应源码和 <code>build.sh</code> 构建流程；
- C++ GTest 测试工程；
- 测试包中的 <code>cgetrf_batched_test.csv</code>；
- 精度检查脚本 <code>verify_accuracy.py</code>；
- 性能检查脚本 <code>verify_performance.py</code>；
- 性能基线表 <code>gpu_baseline.csv</code>，只用于同 shape 基线关联。

精度工程应放在
<code>test/getrf_batched/cgetrf_batched/arch35/</code>，CSV 字段与同族
S 版测试工程的参数读取方式保持一致。测试命令示例如下：

~~~bash
cd 8月社区任务-aclblasCgetrfBatched算子开发（950）/test_cases
python verify_accuracy.py --repo /path/to/ops-blas \
    --soc ascend950 --csv ./cgetrf_batched_test.csv
python verify_performance.py --repo /path/to/ops-blas \
    --soc ascend950
~~~


### 2.2 CPU Golden 和比对对象

精度测试为每个 batch 矩阵生成独立 CPU Golden，Golden 按 LAPACK
<code>cgetrf</code> 语义执行 complex64 部分主元 LU：

1. 以列主序读取输入矩阵；
2. 使用 <code>|real| + |imag|</code> 选择当前列主元；
3. 使用 1-based pivot；
4. 按 PIVOT 或 NO_PIVOT 模式执行相同分解顺序；
5. 记录每个矩阵的 info；
6. 生成原地 L/U 因子。

被测侧和 Golden 逐 batch 比较以下对象：

- 接口返回状态码；
- infoArray；
- PivotArray；
- Aarray 中的 L/U 因子；
- lda padding 的保护区；
- quick return 场景的 Kernel 提交行为。

Golden 只用于比对，不得被被测算子调用，也不得以 CPU 计算替代
Kernel 执行。

### 2.3 精度用例覆盖

精度用例按以下维度组合，覆盖基础、扫描、矩阵类型、模式、边界和特殊
值场景：

| 测试维度 | 覆盖内容 |
| --- | --- |
| n | 0、1、小质数、2 的幂、2 的幂 ±1、非对齐值以及中大规模 |
| batchSize | 0、1、小批量、中批量和大批量 |
| lda | <code>max(1,n)</code> 紧凑布局，以及带 padding 的多个值 |
| pivot_mode | PIVOT、NO_PIVOT |
| matrix_type | 随机非奇异、对角占优、单位阵、病态 Hilbert、全零列奇异、相关行奇异、混合 batch 奇异 |
| Aarray | Device 指针数组、列主序矩阵、矩阵之间非重叠 |
| 数值分布 | 均匀分布占 50%，取值范围为 [-5, 5]；正态分布占 50%，均值 μ 在 [-5, 5] 内取值，标准差 σ 在 [0.1, 2] 内取值；复数实部、虚部分别独立采样 |
| 特殊值 | 规格允许的正负零、Inf、NaN 组合 |
| 参数异常 | 空 handle、空 Aarray、Pivot 非空而 info 为空、负 n、负 batch、非法 lda |

尺寸扫描应包含小规模和较大规模，batch 扫描应覆盖单矩阵与批量并行
场景。padding 用例在 padding 区预置保护值，检查 Kernel 仅访问逻辑
<code>n × n</code> 区域。

测试数据生成器须按上述比例实现均匀分布和正态分布；混合分布模式须
分别生成两类样本，不能将正态分布用例映射为均匀分布。

### 2.4 参数和边界测试方法

每类参数用例使用独立的输入构造和状态检查：

| 场景 | 测试方法 | 判定标准 |
| --- | --- | --- |
| 正常 PIVOT | 分配 Device 指针数组、矩阵、PivotArray 和 infoArray，执行后同步并与 Golden 比较 | 状态码为 SUCCESS；info/pivot 按整数规则一致；非奇异 L/U 满足精度阈值 |
| 正常 NO_PIVOT | 将 PivotArray 置 NULL，准备有效矩阵和 infoArray，执行后与无主元 Golden 比较 | 不返回参数错误；不发生行交换；L/U 和 info 满足对应 Golden 及精度阈值 |
| NO_PIVOT 且 info 为空 | 将 PivotArray 和 infoArray 均置为 NULL，准备有效矩阵并执行无主元分解 | 不返回参数错误；不写 Pivot 或 info；L/U 满足无主元 Golden 及精度阈值 |
| n 为 0 | 使用合法 handle，传入空工作尺寸，记录 Kernel 提交事件 | 返回 SUCCESS；不启动 Kernel；不访问或改写输出数组 |
| batchSize 为 0 | 使用合法 handle，传入空 batch，记录 Kernel 提交事件 | 返回 SUCCESS；不启动 Kernel；不访问或改写输出数组 |
| handle 为空 | 传入其他参数组合，调用接口 | 返回 HANDLE_IS_NULLPTR，不启动 Kernel |
| Aarray 为空 | 使用非空 n 和 batchSize | 返回 INVALID_VALUE，不启动 Kernel |
| n 或 batchSize 为负 | 分别构造负维度和负 batch | 返回 INVALID_VALUE，不启动 Kernel |
| lda 不足 | 令 <code>lda &lt; max(1,n)</code> | 返回 INVALID_VALUE，不启动 Kernel |
| Pivot 非空、info 为空 | 使用 PIVOT 模式并将 infoArray 置 NULL | 返回 INVALID_VALUE，不启动 Kernel |
| 奇异 batch | 使用全零列、相关行和混合奇异矩阵 | 接口返回 SUCCESS；info 为对应奇异步号；pivot 按 1-based 规则一致；不以 info&gt;0 判为接口错误 |

非法参数用例在调用前设置 Device 保护区，调用后检查保护区未被写入。
参数检查的判定以公共状态码为准，不以异常文本或 Host 日志为准。

### 2.5 特殊值测试方法

对输入矩阵构造有限值、正负零、Inf 和 NaN 的组合，并用同样的特殊值
规则驱动 CPU Golden。比较时执行以下规则：

- 有限实部和虚部分量按 FLOAT32 误差阈值比较；
- Golden 为 NaN 的位置要求被测侧同样分类为 NaN，不用普通数值误差
  公式比较 NaN；
- Golden 为 Inf 的位置要求被测侧具有相同的无穷分类和符号；
- 正负零按任务书允许的复数分量语义检查；
- 特殊值不应导致 Host 状态码被错误转换为参数异常。

如果某个特殊值组合不属于对标接口允许范围，则将其标记为非适用输入，
不以非适用组合推导算子功能结论。

### 2.6 精度执行步骤

精度测试按以下顺序执行：

1. 将 CSV 用例安装到
   <code>test/getrf_batched/cgetrf_batched/arch35/</code>。
2. 依据 <code>--soc=ascend950</code> 选择 <code>arch35</code> 测试工程。
3. 通过 <code>build.sh</code> 编译被测算子和 C++ GTest 测试工程。
4. 为每条非性能用例生成输入、CPU Golden 和 Device 输出缓冲。
5. 调用 <code>aclblasCgetrfBatched</code>，同步同一 stream 后读回
   Aarray、PivotArray 和 infoArray。
6. 按矩阵类型、pivot_mode、边界和特殊值规则进行分组判定。
7. 对奇异 batch 只执行 info/pivot 的整数和语义检查，按照任务书要求
   跳过 L/U 数值误差判定。

测试程序可以按 <code>TC_L0</code>、<code>TC_SQ</code>、<code>TC_BC</code>、
<code>TC_LD</code>、<code>TC_MT</code>、<code>TC_PV</code>、<code>TC_ED</code>
和扩展用例前缀筛选执行，但设计文档只保留用例分类和判定规则。

### 2.7 精度判定标准

对于非奇异 batch，复数 L/U 的实部和虚部分别按 FLOAT32 标准比较：

| 数据类型 | rtol | atol | 最低匹配比例 | 最大绝对误差上限 |
| --- | ---: | ---: | ---: | ---: |
| COMPLEX64 分量 | 2^-10（约 9.77e-4） | 2^-16（约 1.53e-5） | 0.99 | 1e-2 或 32 × ULP |

逐元素比较条件为：

~~~text
|actual - golden| <= atol + rtol * |golden|
~~~

一个非奇异用例同时满足以下条件时，判定为符合精度要求：

- infoArray 逐元素精确一致；
- PivotArray 在 PIVOT 模式下逐元素 bit-exact 一致，且为 1-based；
- 实部和虚部的 matched ratio 均不低于 0.99；
- 实部和虚部的 max absolute error 均不超过
  <code>1e-2</code> 或 <code>32 × ULP</code> 上限；
- lda padding 保护区未被改写。

测试框架若同时统计 MERE/MARE，则辅助标准为 L/U 实部和虚部分别满足
MERE 小于 <code>2^-13</code>，MARE 离群倍率不超过 <code>10.0</code>。
奇异 batch 的接口状态、info 和 pivot 必须满足整数及语义规则，L/U 不
执行数值阈值判定。

### 2.8 性能测试方法

性能测试只选择 <code>TC_PF</code> 性能用例，并按以下步骤执行：

1. 准备 COMPLEX64、列主序、紧凑 <code>lda</code> 的批量矩阵。
2. 为每个 shape 和 batchSize 准备同一数据定义下的 GPU 基线和 NPU
   被测调用。
3. 复用输入和 Device 分配，避免把分配与初始化计入接口计时。
4. 对每个 case 先 warmup，再采集超过 50 次有效 Device 时间。
5. 取有效采样平均值，记录调用范围、shape、batchSize、pivot_mode 和
   matrix_type。
6. 对任务书列出的固定 case 使用 us 上限判定；对扩展 case 使用同
   shape GPU 基线计算性能倍率。

设备事件的起止位置必须包围同一段 API 调用和其必要同步，GPU 与 NPU
不能采用不同的计时范围。性能脚本按 case 名称和
<code>(n, batch_size)</code> 关联基线，不将缺少基线的 case 当作固定
阈值的替代来源。

<code>verify_performance.py</code> 用于组织性能用例并关联基线。固定任务
阈值应以 Device Event 的平均耗时或 msprof 采集的设备侧耗时进行判断；
GTest 用例整体墙钟时间可能包含 Host 数据准备、Kernel 执行、Golden 计算
和结果比较，不得替代验收计时数据。

### 2.9 性能判定标准

性能判定分为固定上限和倍率两类：

| 判定类型 | 适用场景 | 符合条件 |
| --- | --- | --- |
| 固定耗时上限 | n=64、batchSize=512；n=256、batchSize=64；n=512、batchSize=32，均为 PIVOT + DIAGONALLY_DOMINANT | NPU 平均单次耗时分别不高于 294.43 us、3398.93 us、25619 us |
| GPU 相对倍率 | 任务书固定表以外的扩展性能用例 | GPU 平均设备耗时 / NPU 平均设备耗时不低于 0.4 |
| 采样有效性 | 全部性能用例 | 先执行 warmup；有效采样次数大于 50；计时范围一致 |

任一性能 case 未满足其对应的固定上限或倍率，判定该 case 不符合性能
标准；不以减少采样次数、改变计时范围或更换输入规模规避判定。

### 2.10 内存测试方法和标准

内存测试使用与精度/性能测试相同的 shape、batchSize、lda 和
complex64 数据规模：

1. 按 <code>batchSize × lda × n × 8</code> 计算输入矩阵预算。
2. 采集输入矩阵、指针数组、PivotArray、infoArray 和 Kernel 临时缓冲
   的分配量。
3. 检查调用前后是否出现与完整批次矩阵同阶的额外 Device 副本。
4. 检查每个用例的 Host 侧准备量不超过 512 MiB 设计预算。
5. 检查 UB 使用量符合选定 tile 的容量约束。

由于任务书的内存验收项标注为“不涉及”，内存测试用于资源安全和设计
约束检查，不另行虚构独立的性能数值阈值。出现越界访问、超出预算的
临时分配或 padding 被改写时，该资源约束不符合。

### 2.11 异步和无回退测试

异步语义使用独立 stream 验证：

- 在同一 stream 上连续提交两次不同 batch 的分解；
- 只在读回前同步该 stream；
- 检查第二次调用没有读取第一次调用尚未完成的输出；
- 检查 quick return 不产生 Kernel 事件；
- 通过 profiler 或设备事件确认非空调用在 NPU stream 上执行；
- 检查算子入口没有调用 CPU Golden、CPU LU 或其他设备回退路径。

判定以输出依赖关系、stream 顺序和设备执行轨迹为准，不以 Host 调用
返回的先后顺序推断异步行为。

## 三、接口与功能契约

本节说明算子的功能边界、公开接口和用户可观察的行为。接口语义以
任务书、ops-blas 公共头文件和 cuBLAS 对标接口为准。

### 3.1 需求来源

<code>aclblasCgetrfBatched</code> 面向 Ascend 950PR，使用 Ascend C
开发单精度复数批量 LU 部分主元分解。算子对 Device 指针数组中的每个
列主序复数方阵独立执行分解，并将 L/U 因子原地写回输入矩阵。

任务要求使用 ops-blas 公共接口和 Ascend C Kernel 直调模式。Host 接口
从 handle 获取 stream，在该 stream 上异步发起 NPU Kernel。实现目录和
测试目录分别沿用 ops-blas 的 getrf_batched 族目录结构。

### 3.2 功能目标

对批次中第 <code>i</code> 个 n×n 复数矩阵执行带部分主元的 LU 分解：

~~~text
P * Aarray[i] = L * U
~~~

其中 P 是由主元序列构造的行交换置换矩阵，L 是单位下三角矩阵，U 是
上三角矩阵。分解结果原地写回 <code>Aarray[i]</code>：

- 严格下三角区域保存 L 的非单位对角元素；
- 对角线及上三角区域保存 U；
- L 的单位对角线不写回；
- <code>PivotArray</code> 保存每一步的行交换位置，采用 1-based
  LAPACK 下标；
- <code>infoArray[i] == 0</code> 表示该批次矩阵未发现零主元；
- <code>infoArray[i] == k &gt; 0</code>，其中 <code>k ∈ [1,n]</code>，表示
  <code>U(k,k) == 0</code>，表示合法的奇异分解状态，接口状态码仍为
  <code>ACLBLAS_STATUS_SUCCESS</code>。

当 <code>PivotArray == NULL</code> 时，接口进入 NO_PIVOT 模式，不做
行交换，执行非主元 LU 分解。该模式是合法功能路径，不属于异常输入。

### 3.3 非目标范围

本次设计只覆盖任务书声明的公共接口和 Ascend 950PR 场景，以下能力不
属于本次范围：

- <code>complex128</code>、实数类型和其他 batch 数据布局；
- 广播、非连续矩阵视图或超出 <code>lda</code> 语义的地址访问；
- 通过 CPU、GPU 或其他设备实现替代 NPU Kernel；
- 需要额外公共参数的 950PR 私有平行接口；
- 对相互重叠的不同矩阵指针提供定义明确的行为；
- 以全矩阵临时副本替代原地分解。

### 3.4 公开接口

接口声明新增至 ops-blas 的
<code>include/cann_ops_blas.h</code>，参数顺序和 cuBLAS 同族接口
保持一致，不定义 950PR 私有接口。

~~~cpp
aclblasStatus_t aclblasCgetrfBatched(
    aclblasHandle_t handle,
    int n,
    aclblasComplex* const Aarray[],
    int lda,
    int* PivotArray,
    int* infoArray,
    int batchSize);
~~~

### 3.5 参数契约

下表定义 Host 参数、Device 数据和异常行为。矩阵采用列主序，矩阵元素
<code>Arow,col</code> 的地址为
<code>Arow + col * lda</code>。

| 参数 | 方向 | 类型和位置 | 语义与布局 | 合法性及异常 |
| --- | --- | --- | --- | --- |
| <code>handle</code> | 输入 | <code>aclblasHandle_t</code>，Host | BLAS 上下文，携带 stream | 空句柄返回 <code>ACLBLAS_STATUS_HANDLE_IS_NULLPTR</code> |
| <code>n</code> | 输入 | <code>int</code>，Host | 每个方阵的行数和列数 | <code>n &lt; 0</code> 返回 <code>ACLBLAS_STATUS_INVALID_VALUE</code>；<code>n == 0</code> 为合法 quick return |
| <code>Aarray</code> | 输入/输出 | Device 指针数组；元素指向 Device 矩阵 | 每个元素指向一个 <code>lda × n</code> 存储区，逻辑矩阵为 n×n，分解原地写回 | 非空工作场景下为空返回 <code>ACLBLAS_STATUS_INVALID_VALUE</code>；各矩阵不得重叠 |
| <code>lda</code> | 输入 | <code>int</code>，Host | 每个矩阵的 leading dimension | 必须满足 <code>lda &gt;= max(1,n)</code>，否则返回 <code>ACLBLAS_STATUS_INVALID_VALUE</code> |
| <code>PivotArray</code> | 输出 | Device <code>int32</code> 数组 | 大小为 n×batchSize，按批次连续排布，第 i 个矩阵第 j 步位于 <code>PivotArray[i*n+j]</code>，值为 1-based 行下标 | NULL 表示 NO_PIVOT 模式；非 NULL 时要求 <code>infoArray</code> 非 NULL |
| <code>infoArray</code> | 输出 | Device <code>int32</code> 数组 | 大小为 batchSize；每个元素记录对应矩阵的分解状态 | Pivot 模式下为空返回 <code>ACLBLAS_STATUS_INVALID_VALUE</code>；奇异信息不作为接口错误码 |
| <code>batchSize</code> | 输入 | <code>int</code>，Host | 待分解矩阵数量 | <code>batchSize &lt; 0</code> 返回 <code>ACLBLAS_STATUS_INVALID_VALUE</code>；<code>batchSize == 0</code> 为合法 quick return |

NO_PIVOT 模式下 <code>infoArray</code> 可以为空；若非空，则按正常规则写入
每个矩阵的分解状态。

<code>Aarray</code> 中的每个矩阵使用 <code>aclblasComplex</code> 存储，
实部和虚部均为 FLOAT32。输入矩阵的 padding 区域不属于逻辑矩阵，Kernel
不得读取或改写该区域。

### 3.6 主元、奇异和返回状态语义

置换矩阵 P 按照以下序列构造：

~~~text
P = P1 * P2 * ... * Pn
~~~

在第 j 步，从当前列的候选行中按照
<code>|real(x)| + |imag(x)|</code> 选择最大元素，选择规则与 LAPACK
<code>cgetrf</code> 的 ICAMAX 口径一致。发生行交换时，交换两行的全部
<code>n</code> 列数据；主元下标写入 1-based 值。

当第 k 步主元为零时，写入对应 <code>infoArray</code> 值，并继续执行任务
书规定的合法奇异分解路径。接口返回值只表示 Host 参数和 Kernel 提交
状态，不把 <code>infoArray[i] &gt; 0</code> 转换为接口失败状态。

### 3.7 Host 校验顺序

Host 使用固定校验顺序，避免在参数非法时访问 Device 地址，并保证 quick
return 和状态码行为稳定：

1. 检查 <code>handle</code>，空句柄返回
   <code>ACLBLAS_STATUS_HANDLE_IS_NULLPTR</code>。
2. 检查 <code>n</code> 和 <code>batchSize</code>，负值返回
   <code>ACLBLAS_STATUS_INVALID_VALUE</code>。
3. 检查 <code>lda</code> 是否满足
   <code>max(1,n)</code> 的最小约束。
4. 当 <code>n == 0</code> 或 <code>batchSize == 0</code> 时直接返回
   <code>ACLBLAS_STATUS_SUCCESS</code>，不启动 Kernel，不解引用
   <code>Aarray</code>、<code>PivotArray</code> 或 <code>infoArray</code>。
5. 对非空工作场景检查 <code>Aarray</code>，为空返回
   <code>ACLBLAS_STATUS_INVALID_VALUE</code>。
6. 当 <code>PivotArray</code> 非空时检查 <code>infoArray</code>，为空
   返回 <code>ACLBLAS_STATUS_INVALID_VALUE</code>。
7. 生成 tiling 参数，在 handle 绑定的 stream 上发起 Kernel。

quick return 的优先级仅作用于 <code>n == 0</code> 或
<code>batchSize == 0</code> 的合法空工作场景；非空工作场景仍须执行
全部指针和数组约束检查。

## 四、方案与工程边界

本节明确公共接口、对标语义和实现路线，保证算子可以按 ops-blas 工程
规范组织。

### 4.1 ops-blas 集成方式

实现放入 <code>blas/getrf_batched/arch35/</code>，测试放入
<code>test/getrf_batched/cgetrf_batched/arch35/</code>。公共声明位于
<code>include/cann_ops_blas.h</code>，公共状态码和复数类型沿用
<code>include/cann_ops_blas_common.h</code>。

Host 层承担参数检查、tiling 生成、Kernel 发起和 stream 绑定；Device
侧承担批量矩阵读取、主元搜索、行交换、复数除法、尾部更新以及结果写回。
实现不增加额外的公共参数，也不通过 CPU 计算替代 Device 计算。

### 4.2 对标接口和数学参考

cuBLAS <code>cublasCgetrfBatched</code> 提供句柄式批量 LU 接口，矩阵指针
以指针数组传入，结果原地写回，并使用 pivot/info 数组表达主元和奇异
状态。本设计沿用其参数顺序、列主序和奇异语义。

Netlib LAPACK <code>cgetrf</code> 用于定义单矩阵 CPU Golden 的分解顺序、
1-based 主元约定和部分主元选择口径。由于 CBLAS 不提供该 LU 接口，
精度参考由测试工程按上述语义实现，不把其他函数误作为 Golden。

### 4.3 方案对比

候选方案从语义一致性、数据搬运、并行度和内存峰值四方面比较，采用
Host 校验加 Ascend C 直调 Kernel 的方案。

| 方案 | 优点 | 风险或代价 | 取舍 |
| --- | --- | --- | --- |
| CPU 分解后搬回 Device | 参考逻辑简单 | 不满足 NPU 计算要求，搬运成本高 | 不采用 |
| 全局内存逐元素串行更新 | 代码路径直接 | 访存不连续，批量并行度和性能不足 | 不作为主实现 |
| 一个矩阵一个任务，UB 分块 | 批次之间天然独立，适合小矩阵批量 | 大矩阵需要循环处理 panel 和尾部 | 采用 |
| 全矩阵搬入 UB | 局部访问简单 | n 增大时 UB 不足，内存占用不可控 | 不采用 |
| 全矩阵 Device 临时副本 | 行交换逻辑较简单 | 增加显存和搬运，破坏原地路径 | 不采用 |

### 4.4 设计原则

实现遵循以下原则：

1. 先保持 cuBLAS/LAPACK 可观察语义，再进行硬件特化。
2. 所有地址计算以列主序和 <code>lda</code> 为准，不访问 padding。
3. Pivot、info、奇异矩阵和 NO_PIVOT 使用独立可验证的 Kernel 路径。
4. 仅使用固定大小或随 tile 大小变化的 UB 缓冲，不申请全矩阵副本。
5. 性能测试采用与 GPU 基线一致的 shape、batch、数据类型和计时范围。
6. 计算结果在调用方同步同一 stream 后读取，Host 不插入全局同步。

## 五、Kernel 实现与数据路径

本节描述一次调用从公共 API 到批量矩阵完成分解的模块边界和数据流。

### 5.1 总体架构

一次非空调用从 Host 参数校验进入 Tiling 和 Kernel，随后按 batch 独立
完成 Device 分解：

~~~mermaid
flowchart LR
    U[调用者] --> H[aclblasCgetrfBatched]
    H --> V[Handle/维度/lda/指针校验]
    V --> Q{空工作场景}
    Q -->|n=0 或 batchSize=0| R1[返回 SUCCESS]
    Q -->|非空| T[Tiling 生成]
    T --> S[绑定 Handle Stream]
    S --> K[Ascend C LU Kernel]
    K --> A[读取 Aarray 指针数组]
    A --> P[主元搜索与行交换]
    P --> F[复数缩放与尾部更新]
    F --> W[原地写回 L/U、Pivot、info]
    W --> R2[异步提交返回]
~~~

对于非空输入，Kernel 按 batch 矩阵独立处理。调用接口只返回提交状态；
调用方在读回矩阵、PivotArray 或 infoArray 前，须同步当前 handle 绑定的
stream。

### 5.2 Host 层职责

Host 层按接口、参数、tiling 和启动四类职责组织：

| 模块 | 主要职责 | 设计要求 |
| --- | --- | --- |
| API 入口 | 接收公共接口参数 | 与公共头文件声明逐参数一致 |
| 参数检查 | 检查 handle、维度、lda、数组关系和 quick return | 不解引用非法 Device 地址 |
| 模式选择 | 选择 PIVOT、NO_PIVOT、空工作和规模路径 | 不改变公开状态码语义 |
| tiling 生成 | 计算矩阵任务数、tile 大小和工作区偏移 | 不申请全矩阵临时存储 |
| Kernel 启动 | 获取 stream、设置 Kernel 参数、异步发起 | 不调用 CPU fallback 或全局同步 |

建议的内部 tiling 数据如下。该结构只作为 Host 与 Kernel 的内部契约，
不对用户公开：

~~~cpp
struct CgetrfBatchedTilingData {
    int32_t n;
    int32_t lda;
    int32_t batch_size;
    int32_t tile_size;
    int32_t panel_tiles;
    int32_t block_count;
    int32_t pivot_mode;
    int32_t info_mode;
    uint64_t pivot_stride;
};
~~~

每个矩阵的基地址必须从 <code>Aarray[i]</code> 独立读取，不能假定 batch
中的矩阵具有固定的矩阵间步长。<code>pivot_stride</code> 仅描述主元结果中
相邻矩阵对应区域的跨度。

<code>pivot_mode</code> 区分 PIVOT 和 NO_PIVOT；
<code>info_mode</code> 用于控制 info 写回和合法奇异路径。实际字段数量
以 ops-blas arch35 Kernel 框架为准，但不得把内部字段扩展为公共 API。

### 5.3 批次和 Block 划分

每个批次矩阵相互独立，任务编号按矩阵指针数组下标划分。对于小矩阵和
较大 batch，优先采用一个矩阵一个 block，增加并行矩阵数量并降低调度
开销。对于大矩阵和较小 batch，在单个矩阵内部以 panel 和 trailing tile
循环处理，使用向量化完成同一 panel 下的行更新。

初始任务划分规则如下：

| 场景 | 任务粒度 | 目的 |
| --- | --- | --- |
| 小 n、大 batch | 一个矩阵一个 block | 提高批次并行度，减少单矩阵同步 |
| 中等 n、中等 batch | 一个矩阵一个 block，矩阵内分 tile | 平衡调度、UB 占用和计算量 |
| 大 n、小 batch | 一个矩阵分 panel 循环 | 控制 UB 使用量，覆盖大矩阵 |
| batchSize 为 1 | 单矩阵路径 | 避免生成无效 batch 任务 |

Block 数不超过可用计算单元数和有效矩阵任务数。一个矩阵内的 panel
更新保持严格的 <code>k</code> 步依赖；下一步主元搜索必须在当前步行交换
和 trailing update 完成后开始。

### 5.4 LU Kernel 算法

对于每个矩阵，Kernel 按 <code>k = 0 ... n-1</code> 处理当前列：

1. 读取当前列候选行的复数元素，计算
   <code>|real(x)| + |imag(x)|</code>。
2. 对候选值执行最大值归约；相同值的选择顺序遵循 ICAMAX 口径。
3. 在 PIVOT 模式写入 1-based 主元下标，并交换第 k 行和主元行的全部
   <code>n</code> 列数据。
4. 若主元为零，将 <code>k + 1</code> 记录到 info，并按奇异路径继续处理。
5. 用主元对当前列下方元素执行复数除法，形成 L 的严格下三角部分。
6. 以 rank-1 更新方式更新 trailing matrix，形成 U 和后续 L/U 数据。

复数除法和更新均按 FLOAT32 分量执行。复数乘法使用：

~~~text
(ar + i*ai) * (xr + i*xi)
  = (ar*xr - ai*xi) + i*(ar*xi + ai*xr)
~~~

计算中间值不提升为 FLOAT64。Kernel 在完成所有输入读取后再写回对应
tile，确保原地写入不会覆盖同一 tile 尚未使用的数据。

### 5.5 Panel、UB 和数据搬运

Kernel 使用有限大小 UB 缓冲承载 panel、行交换片段和 trailing tile，
不把完整 n×n 矩阵搬入 UB。建议使用以下流水：

~~~mermaid
flowchart LR
    G[GM Aarray[i]] --> M2[MTE2 搬入 panel/tile]
    M2 --> U1[UB 双缓冲]
    U1 --> R[主元归约/行交换准备]
    R --> V[复数除法与向量更新]
    V --> U2[输出 UB tile]
    U2 --> M3[MTE3 写回]
    M3 --> G
~~~

输入矩阵地址统一按
<code>base + row + col * lda</code> 计算。每个 tile 的有效行列范围
由 n 和 tile size 截断；尾块使用 mask，不读取或写入逻辑区域之外的
元素。PivotArray 按 <code>i*n+j</code> 写入，infoArray 按批次下标写入。

### 5.6 PIVOT 与 NO_PIVOT 路径

两种模式共享矩阵更新骨架，仅在主元处理和输出上区分：

| 模式 | 主元搜索 | 行交换 | PivotArray | infoArray |
| --- | --- | --- | --- | --- |
| PIVOT | 执行 ICAMAX 归约 | 按主元交换两行的全部 n 列 | 写入 1-based 序列 | 写入 0 或奇异步号 |
| NO_PIVOT | 不做行交换选择 | 不交换 | 指针为 NULL，不写回 | 非空时写入 0 或 k+1；为空时不写 |

NO_PIVOT 不应被实现为非法参数或降级到 PIVOT。两种模式都必须使用
相同的 <code>complex64</code> 计算类型和列主序地址规则。

### 5.7 奇异矩阵处理

奇异矩阵属于功能输入的一部分。Kernel 发现
<code>U(k,k) == 0</code> 时，按以下方式处理：

- 将 <code>k + 1</code> 写入对应 <code>infoArray[i]</code>；
- 保留本步此前形成的 L/U 数据；
- 执行任务书允许的后续分解步骤，不因该状态提前把接口转换为错误；
- 接口状态码保持 <code>ACLBLAS_STATUS_SUCCESS</code>；
- 测试侧对奇异 batch 精确检查 info 和 pivot，对 L/U 数值按照任务书
  规定跳过数值误差判定。

奇异矩阵包括全零列、相关行和混合 batch。混合 batch 中不同矩阵的
<code>infoArray</code> 必须分别对应各自分解状态。

### 5.8 边界与异步生命周期

<code>n == 0</code> 或 <code>batchSize == 0</code> 时，Host 直接返回成功，
不启动 Kernel，不访问任何 Device 数组。非空矩阵使用的
<code>lda</code> 必须满足最小约束，尾部 padding 保持不变。

Kernel 使用 handle 绑定的 stream。以下对象在 stream 完成前必须保持有效：

- Aarray 指针数组和其引用的矩阵 storage；
- PivotArray 和 infoArray storage；
- Kernel 参数与 tiling 数据；
- handle 及其绑定的 stream。

Host 不在正常调用路径中执行全局同步；测试程序在读回结果前同步同一
stream。

## 六、性能和资源策略

本节定义性能计时范围、任务书规定的性能上限以及资源检查方法。

### 6.1 性能指标和计时范围

性能测试只计入被测接口对应的 Device 执行耗时，不计入随机输入生成、
Host 端分配、Host 到 Device 的准备搬运和首次编译。采用同一 shape、
同一 batchSize、同一数据类型、同一 pivot 模式和同一矩阵类型分别准备
GPU 参考侧与 NPU 被测侧。

性能倍率定义为：

~~~text
性能倍率 = GPU 参考侧平均设备耗时
         / NPU 被测侧平均设备耗时
~~~

NPU 设备计时须先执行 warmup，再进行超过 50 次有效采样并取平均。GPU
基线和 NPU 测量须使用一致的调用范围、同步方式和数据准备约束。
固定任务阈值以 Device Event 平均耗时或 msprof 设备侧耗时为判定依据，
GTest 用例整体墙钟时间仅用于流程观测，不能代替性能验收计时。

### 6.2 任务书性能上限

以下为任务书明确给出的 COMPLEX64 典型性能 case 和平均单次耗时上限。
平均耗时单位为 us：

| case | n | batchSize | pivot_mode | matrix_type | 平均耗时上限（us） |
| --- | ---: | ---: | --- | --- | ---: |
| 1 | 64 | 512 | PIVOT | DIAGONALLY_DOMINANT | 294.43 |
| 2 | 256 | 64 | PIVOT | DIAGONALLY_DOMINANT | 3398.93 |
| 3 | 512 | 32 | PIVOT | DIAGONALLY_DOMINANT | 25619 |

上述三组 case 的判定条件是：NPU 平均单次耗时不高于对应上限。测试
用例同时覆盖 <code>n=1024, batchSize=16</code> 规模扩展场景；对于
任务书未单独给出固定 us 上限的扩展场景，使用同 shape 的 GPU 基线
计算性能倍率，倍率须不低于 0.4。

### 6.3 性能优化方向

实现按以下方向调优：

- 小矩阵大 batch 使用矩阵级并行，降低单矩阵启动和空转开销；
- 主元归约使用向量化绝对值和复数分量组合计算；
- panel、行交换片段和 trailing tile 使用 UB 双缓冲；
- 对连续列段使用合并搬运，对尾块使用 mask；
- 减少不必要的全局内存往返，避免重复读取同一 panel；
- 对 PIVOT 和 NO_PIVOT 共享更新代码，减少运行时分支；
- 按 n、batchSize 和 UB 容量选择 tile，避免单一 tile 覆盖所有规模。

初始 tile 只作为调优起点，不作为公共语义的一部分。任何 tile 调整都
必须保持 pivot、info、L/U、padding 和异步行为不变。

### 6.4 内存约束

任务书未设置独立的内存性能指标，但自验必须包含内存场景。资源检查
采用以下规则：

- complex64 每元素按 8 字节估算，单用例输入矩阵累计占用按
  <code>batchSize × lda × n × 8</code> 计算；
- 测试用例的 Host 侧矩阵准备量控制在 512 MiB 设计预算内；
- 实现只使用固定规模或 tile 规模的 UB 临时缓冲，不创建与完整批次
  矩阵同阶的额外 Device 副本；
- 统计接口调用前后 Device 分配变化和临时缓冲大小，检查内存申请
  不超过用例预算；
- 若资源检查发现超出预算，按资源约束不符合处理，不以修改输入规模
  代替判定。

## 七、风险和回归关注点

本节记录设计阶段需要重点控制的技术风险和对应的测试关注点。

### 7.1 主元下标和布局

主元是 1-based，且 PivotArray 按批次连续排布。测试必须同时检查下标
范围、批次间隔、每个矩阵的 pivot 序列和行交换后的 L/U 布局，避免把
0-based 或按步交错布局误认为正确。

### 7.2 奇异矩阵状态

<code>infoArray[i] &gt; 0</code> 是合法分解状态。测试将接口返回状态和
info 值分开判定，避免把合法奇异 batch 误判为接口错误；奇异 batch 的
L/U 数值检查遵循任务书指定的跳过规则。

### 7.3 PIVOT 与 NO_PIVOT 分支

NO_PIVOT 的 PivotArray 必须为 NULL，不能以空数组或非法状态替代。
测试分别生成两种 Golden，并检查 NO_PIVOT 不发生额外行交换。

### 7.4 panel 依赖与并行

LU 的第 k 步依赖前一步的主元和 trailing update。Kernel 调度必须保持
panel 处理顺序，测试覆盖 batchSize 为 1、多个 batch、大 n 和非对齐 n，
以检查并行任务边界不会跨矩阵或跨步骤污染。

### 7.5 lda padding 和原地写回

矩阵物理列跨度由 lda 决定，padding 不属于输出。测试预置保护值并检查
每个矩阵的逻辑区域和 padding，确保 MTE 搬运和尾块 mask 没有越界。

### 7.6 性能计时一致性

批量 LU 的性能对 n、batchSize、pivot 模式和矩阵类型敏感。GPU 与 NPU
必须使用同一输入定义、同一计时边界和同一同步方式；固定上限和倍率
判定分开执行，避免把不同口径的数值混在一起。

## 八、名词解释

下表统一本文使用的接口和数学术语。

| 名词 | 解释 |
| --- | --- |
| complex64 | 实部和虚部均为 FLOAT32 的单精度复数 |
| LU 分解 | 将矩阵分解为置换矩阵、单位下三角矩阵和上三角矩阵的过程 |
| PIVOT | 启用部分主元行交换的分解模式 |
| NO_PIVOT | PivotArray 为 NULL 时的不选主元分解模式 |
| PivotArray | 记录每一步主元行的 1-based 整数数组 |
| infoArray | 记录每个 batch 矩阵分解状态的整数数组 |
| lda | 列主序矩阵的 leading dimension |
| panel | LU 当前处理的列块或列向量区域 |
| trailing matrix | 当前 panel 右下方尚未更新的矩阵区域 |
| ICAMAX | 按复数分量绝对值组合选择最大元素的主元搜索口径 |
| quick return | 对合法空工作输入直接返回且不启动 Kernel |

## 九、参考资料

参考资料和工作区测试入口如下：

1. [aclblasCgetrfBatched 任务书](../aclblasCgetrfBatched_Atlas950PR_task_doc.md)。
2. [测试用例说明](../test_cases/README.md)。
3. [测试用例生成脚本](../test_cases/gen_csv.py)。
4. [精度测试脚本](../test_cases/verify_accuracy.py)。
5. [性能测试脚本](../test_cases/verify_performance.py)。
6. [ops-blas 开源仓](https://gitcode.com/cann/ops-blas)。
7. [ops-blas 公共头文件](https://gitcode.com/cann/ops-blas/blob/master/include/cann_ops_blas.h)。
8. [Ascend C 算子开发文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/opdevg/Ascendcopdevg/atlas_ascendc_map_10_0002.html)。
9. [生态算子精度标准](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md)。
10. [cuBLAS cublasCgetrfBatched 文档](https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-getrfbatched)。
11. [Netlib LAPACK cgetrf 文档](https://www.netlib.org/lapack/explore-html/dd/dd1/group__getrf.html)。
12. [社区任务设计文档模板](https://gitcode.com/cann/cann-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md)。
