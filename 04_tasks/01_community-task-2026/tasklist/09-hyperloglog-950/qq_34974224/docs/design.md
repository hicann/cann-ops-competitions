# HyperLogLog 容器开发设计（Ascend 950）

## 1. 需求背景

### 1.1 需求来源

本设计对应[9月社区任务-hyperloglog容器开发(950)](https://www.hiascend.com/activities/task-center/details/f975dbdd02cb43eaacd0d7e3d3fa8821)，功能、精度和性能要求以[官方任务书及测试附件](https://www.hiascend.com/p/resource/202609/66448a152b6f4bc1aaff5aad233cd467.zip)为准。任务讨论入口为 [ops-collections discussions/4](https://gitcode.com/cann/ops-collections/discussions/4)。

### 1.2 背景介绍

HyperLogLog 使用固定大小的 Sketch 近似估算输入集合的去重基数。添加元素更新寄存器，合并两个兼容 Sketch 得到集合并集的 Sketch，估算结果保留近似计数语义。它不保存全部输入，也不提供精确去重集合或成员查询。

参考接口为 cuCollections 的 `hyperloglog`，目标实现位于 `cann/ops-collections`。采用该仓已有容器的纯头文件、C++ 资源管理和 Ascend C Kernel 工程模式。

## 2. 需求分析

### 2.1 功能与输入范围

- Key 支持 `int32_t`、`int64_t`，输入为 Device 上连续存放的 ND 一维数组，形状为 `[NumInputs]`，覆盖相应类型的完整整数范围。
- 支持按 SketchSizeKB、标准差或精度构造，以及析构、Clear、Add、Merge 和 Estimate。
- 按 KB 构造支持 8、16、32、64、128、256 六档。附件以一个 8 位寄存器保存一个 rank，因此 `m = SketchSizeKB × 1024 = 2^p`，六档对应 p=13–18。本文容量数值采用 1024 字节为 1 KB，与附件一致。
- 支持空输入、重复输入、分批输入、清空后重用，以及空集、重叠集合、不相交集合的合并。
- 相同输入、配置和哈希策略产生相同 Sketch 和估算结果；重复添加幂等，分批添加等价于一次添加，Estimate 不修改 Sketch。

### 2.2 工程和资源要求

目标硬件为 Atlas 950 系列及后续支持 SIMT 的昇腾产品。要求 CANN 9.0.0-beta.2 及以上、CMake 3.16 及以上、Catch2 3.5.4 及以上，使用仓库支持的 ccec 或毕昇 ASC 编译器，架构参数与实际 SDK 和设备匹配，C++ 标准沿用仓库的 C++17。

容器仅保存 Sketch、配置和必要工作空间，不复制或保存与输入规模线性增长的另一份 Device 输入。

## 3. 详细设计

### 3.1 容器原型

```cpp
namespace aclco {

template<class Key>
class HyperLogLog {
public:
    using KeyType = Key;
    using ExtentType = aclco::Extent<std::size_t>;

    static HyperLogLog CreateWithSketchSizeKB(
        std::uint32_t sketchSizeKb, aclrtStream stream = nullptr);
    static HyperLogLog CreateWithStandardDeviation(
        double standardDeviation, aclrtStream stream = nullptr);
    static HyperLogLog CreateWithPrecision(
        std::uint32_t precision, aclrtStream stream = nullptr);

    HyperLogLog(HyperLogLog const&) = delete;
    HyperLogLog& operator=(HyperLogLog const&) = delete;
    HyperLogLog(HyperLogLog&& other) noexcept;
    HyperLogLog& operator=(HyperLogLog&& other) noexcept;
    ~HyperLogLog();

    std::uint32_t SketchSizeKB() const noexcept;
    void Clear(aclrtStream stream = nullptr);
    void Add(void* keys, ExtentType keyNum, aclrtStream stream = nullptr);
    void Merge(HyperLogLog const& other, aclrtStream stream = nullptr);
    std::uint64_t Estimate(aclrtStream stream = nullptr) const;
};

} // namespace aclco
```

三个工厂对应参考实现的三种构造配置，并返回拥有资源的对象。任务书的 Create/Destroy 通过工厂与析构表达，与官方测试调用一致；hll 对应 C++ 对象，不另外引入 C 句柄层。Add 以指针和元素数替代起止迭代器。Merge 和 Estimate 不接收输入数量，分别处理另一份 Sketch 和自身状态。

| 参数或返回值 | 含义及约束 |
| --- | --- |
| `Key` | 编译期只允许 int32_t 或 int64_t，其他类型通过 static_assert 拒绝。 |
| `sketchSizeKb` | 表示 Sketch 的字节容量，必须为上述六档之一，非法值报错。 |
| `precision` | 本设计限定为 13–18；范围外报错。该边界与标准差工厂的关系见 3.3 节。 |
| `standardDeviation` | 表示请求的相对标准差上界，必须是正的有限数；容量换算方案见 3.3 节。 |
| `keys`、`keyNum` | 表示 Device 输入首地址及元素数量。数量为 0 时允许空指针；非零数量要求非空、按 Key 对齐并满足可访问范围。 |
| `other` | 表示另一份有效且类型、精度、哈希策略兼容的容器；合并不修改它。 |
| `stream` | 使用调用方的 ACL 流，末尾默认 nullptr，与仓内接口保持一致；ACL 调用失败须报错。 |
| `Estimate` 返回值 | 返回 Host 侧 uint64_t 基数估计，空 Sketch 精确返回 0。 |

### 3.2 生命周期、同步和异常

工厂校验配置、分配设备存储、清零 Sketch，并在初始化流完成后返回。部分构造失败时，先确保已提交操作不再访问待释放内存，再释放已分配资源并报告异常。移动转移配置和资源所有权，源对象置为无存储状态；禁止复制，析构释放 Sketch 和工作空间且不抛异常。

Clear、非空 Add、Merge、Estimate 在返回前完成指定流上的必要工作。零长度 Add 不启动 kernel，可以直接返回，不将它定义为流同步屏障。对无存储对象调用依赖 Sketch 的操作报错。调用方须保证对象和输入在操作完成前有效，并在 ACL runtime 销毁前销毁容器。不同流或 Host 线程对同一 Sketch 的并发修改不属于本设计的接口保证，调用方需建立依赖。

非法配置、非零长度空指针、对齐错误或不兼容 Merge 使用 `std::invalid_argument`；数量到字节数的算术溢出使用 `std::length_error`；无存储状态使用 `std::logic_error`；分配失败使用 `std::bad_alloc`；ACL 执行错误使用包含操作名和错误码的 `std::runtime_error`。设备操作报错后不保证回滚到调用前状态，不使用可能部分更新的 Sketch 继续推断正常结果。

### 3.3 标准差构造及参考实现差异

三种工厂使用统一的六档配置。标准差工厂在 p=13–18 中选择满足 `1.04 / sqrt(2^p) <= standardDeviation` 的最小 p；没有可用配置时报错，不静默截断到最大容量。请求较宽松时使用最小容量。实现时枚举六档阈值，避免对数取整在边界附近引入不一致。

| 附件标准差 | 本设计选定 p | SketchSizeKB | cuCollections 换算得到的 p |
| --- | --- | --- | --- |
| 0.0115 | 13 | 8 | 14 |
| 0.0082 | 14 | 16 | 15 |
| 0.0058 | 15 | 32 | 16 |
| 0.0041 | 16 | 64 | 17 |
| 0.0029 | 17 | 128 | 18 |
| 0.0021 | 18 | 256 | 19 |

这里存在明确的容量选择差异。固定版本 cuCollections 以 `ceil(2*log2(1.106/sigma))` 选择精度，使用 int32_t 寄存器；其引用的 Spark 同时用 1.106 选择精度、用 `1.04/sqrt(m)` 表示实际相对标准差。验收误差公式中的 1.04 并不能单独证明构造也应使用这一系数。官方附件的标准差测试只检查构造成功和空估计为 0，未断言容量。

本设计采用 1.04 换算并统一限定 p=13–18，以保持任务的六档容量和误差模型一致。附件要求 sigma=0.0021 构造成功，并拒绝按 KB 创建 512 KB；六个标准差在本规则下恰好对应六档容量。若采用 1.106，sigma=0.0021 将需要 p=19 和 512 KB，从而必须为标准差工厂另行扩展容量范围。本设计不作该扩展，也不通过静默截断改变换算结果。标准差参数仍表示按本任务误差模型请求的相对标准差上界，不保证与 cuCollections 对同一参数选择相同精度或容量；该统计模型也不是任意输入的确定误差上界。配置选择位于 Host 工厂，不改变固定 p 下 Add 和 Merge 的寄存器语义。

### 3.4 哈希与寄存器更新

拟复用仓内 `aclco::xxhash_64<Key>`，默认 seed=0。I32 按 4 字节、I64 按 8 字节计算哈希，不先将所有输入提升为 64 位。负数按原类型位表示处理；同类型、同种子才属于相同哈希策略。

设 64 位无符号哈希为 h：

```text
j = h >> (64 - p)
w = (h << p) | (uint64_t(1) << (p - 1))
rank = count_leading_zeros(w) + 1
M[j] = max(M[j], rank)
```

保护位使剩余哈希全零时 rank 仍有界，最大值为 `64-p+1`，在 p=13–18 下不超过 52。rank=0 表示该寄存器尚未被更新，因此一个字节足以表达所有状态。逐寄存器最大值使重复添加、输入排列变化和批次划分不改变最终 Sketch。

Device 存储按 32 位字对齐，每个字打包四个 8 位寄存器。Add 的 SIMT 线程按网格步长遍历输入，计算 j 和 rank。更新时原子读取目标字，若目标字节已有不小于 rank 的值则结束；否则仅替换目标字节并执行 32 位 CAS，失败则基于 CAS 返回的新字重新计算。不能用整字 AtomicMax 代替逐字节最大值，也不能在 CAS 失败后写回过期的邻接字节。

`hyperloglog_ref.h` 提供非拥有型 Device 引用，持有设备存储地址、p 和必要哈希状态，复用单元素 rank 更新逻辑，不分配或释放资源。Host 容器通过普通 kernel 参数传入这些数据；Ref 的有效期不得超过所属存储。

### 3.5 Clear 与 Merge

Clear 将完整 Sketch 置零，并同步指定流；容量和配置保持不变，后续 Add 可重新使用存储。

Merge 先检查两个对象的有效性和配置兼容性，再按字分配线程，对四个字节分别取最大值并写入接收方，来源保持不变。每个输出字只由一个线程写入；接口不允许与 Add 无依赖地并发执行，因此该阶段不需要逐字节 CAS。自合并保持原状态，返回前遵循同步语义。

### 3.6 Estimate

定义 `C[r]` 为值等于 r 的寄存器数量，`V=C[0]`，`Z=sum(C[r]*2^(-r))`。原始估计为 `E=alpha_m*m*m/Z`，其中本任务 m 范围使用 `alpha_m=0.7213/(1+1.079/m)`。

拟由 Device 统计寄存器 rank 的整数直方图，再将小规模直方图传回 Host。每个核生成私有直方图，随后按整数加法合并；所有计数均不超过 m。可以固定预留 65 个 bin 覆盖 64 位哈希的 rank 范围。Host 按递增 r 的固定顺序用 double 计算 Z，执行修正并舍入为 uint64_t。这样无需设备 FP64 原子归约，最终计数和求和顺序不随 SIMT 调度变化。

估算修正对照固定版本 cuCollections 的 HLL++ finalizer：空集返回 0；存在零寄存器且原始估计较小时使用线性计数 `m*log(m/V)`；其余区域按对应 p 的阈值和偏差表选择线性计数、偏差修正或原始估计，最后进行整数舍入。不把未经修正的原始公式直接用于全部基数范围。移植偏差数据或实现时保留来源和许可证要求，并以独立 Host 参考和逐例误差同时验证。

Estimate 每次依据当前 Sketch 计算，不修改寄存器。直方图为必要工作空间，大小只与核数和 rank 范围有关。

### 3.7 Host 校验与内存边界

Key 类型由模板在编译期限制为 int32_t/int64_t。非空 Add 在下发 kernel 前检查对象状态、空指针、对齐和 `keyNum*sizeof(Key)` 溢出；使用 `aclrtPointerGetAttributes` 校验 Device 内存位置及设备归属，使用 `aclrtMemGetAddressRange` 获取所属分配块的基址和字节数。输入范围须位于该查询返回的单个分配块内，以偏移和剩余大小比较，避免计算末地址时溢出。查询失败报错；不承诺支持跨多个独立映射段的输入范围。

CANN 9.0.0-beta.2 文档已提供上述 API，本实现已在 Ascend950PR、CANN 9.1.0 上验证相关检查。查询返回的分配块不是逻辑数组：池内子分配和填充区域不能据此还原，属性查询也不提供元素 dtype。调用方必须保证实际元素类型与 Key 一致、从 keys 起至少包含 keyNum 个有效元素，并保证缓冲区在同步 Add 返回前保持有效。

上述责任划分沿用目标仓已合入容器的约定：StaticSet 的 void* 接口要求实际数据类型和数量与数组匹配；RoaringBitmap 明确说明仅含指针的构造无法获知真实缓冲区长度。保留 `void* + keyNum` 原型，不增加强制 dtype 或逻辑长度参数。任务书中的非法 dtype 检查落实为模板类型约束，可观察的越界落实为字节数溢出和分配块范围检查；实际 dtype 错配和池内逻辑子数组越界属于调用方违反前提，不能声称库能检测全部此类错误。运行时检查保留在正式计时路径中。

### 3.8 性能方案与内存占用

直接 CAS 更新作为通用实现。小 Sketch、大输入场景拟比较每核私有 Sketch 后合并的方案，以减少全局原子竞争；分片数量依据输入量、Sketch 大小、可用核心及片上空间选择。仅在满足目标 SDK 原子操作、存储与同步约束时启用相应路径，最终 Sketch 与通用路径逐字节比较。

核心存储为 m 字节，直方图工作空间为 `O(B*65)` 个整数，B 为核数；若启用分片，额外 Sketch 空间为 `O(S*m)`，S 为有限的分片数。所有分配均检查乘法溢出，报告 Sketch 与工作空间各自占用，不将长期四倍寄存器存储隐藏为工作空间。Add 不建立 O(NumInputs) 的 Device 副本。1e8 个输入本身分别需要约 400 MB（I32）和 800 MB（I64），不含调用方其他内存。

性能分析使用官方 msopprof，结合原子竞争、访问行为、核数和各阶段时延分析直接更新与分片方案，不以局部 kernel 加速代替接口总时延。

### 3.9 目录组织

```text
ops-collections/
├── include/
│   ├── hyperloglog.h
│   ├── hyperloglog_ref.h
│   └── detail/hyperloglog/
├── tests/
│   ├── common/hll_test_common.h
│   ├── hyperloglog/
│   └── performance/hyperloglog/
└── docs/
    └── hyperloglog_API文档和使用示例.md
```

在 `tests/CMakeLists.txt` 中接入功能与性能目录，沿用现有构建脚本、Catch2 及性能测试框架。README 说明构建、运行和复现步骤。

## 4. 可维可测分析

### 4.1 精度标准

对精确去重基数 N>0，逐例要求 `abs(Estimate-N)/N <= 3*1.04/sqrt(m)`；N=0 时 Estimate 必须为 0。该阈值用于逐例验收，不将其解释为任意输入都必然满足的数学保证。

| SketchSizeKB | 寄存器数 m | p | 相对误差上限 |
| --- | --- | --- | --- |
| 8 | 8192 | 13 | 约 3.447145558% |
| 16 | 16384 | 14 | 2.4375% |
| 32 | 32768 | 15 | 约 1.723572779% |
| 64 | 65536 | 16 | 1.21875% |
| 128 | 131072 | 17 | 约 0.861786390% |
| 256 | 262144 | 18 | 0.609375% |

使用 Host 精确去重计数验证误差，使用独立 Host 哈希、rank 和寄存器更新参考验证中间状态。与 cuCollections 对照时固定 Key 字节宽度、种子和 p；由于寄存器存储宽度不同，相同 p 的 cuCollections Sketch 字节数是本设计的四倍，不能把相同 KB 误当作相同配置。性能验收仍按任务书的原始 KB 用例和标杆执行，不自行替换。

### 4.2 功能测试方法

任务书给出的展开用例数如下，按原始 dtype、GENERATE 和 SECTION 组合执行，不删减断言：

| 接口 | 展开用例数 | 主要检查 |
| --- | --- | --- |
| Create | 38 | 验证三种构造、六档配置、空估计和非法参数。 |
| Destroy | 24 | 验证重复构造销毁和移动后的资源所有权。 |
| Clear | 180 | 验证空容器清空、重复清空及清空后重用。 |
| Add | 726 | 验证空输入、重复度、尾部长度、分批一致性、空指针和大规模输入。 |
| Merge | 192 | 验证空集、相同集合、重叠和不相交集合，以及配置不兼容。 |
| Estimate | 120 | 验证空集至大基数的误差、独立构建一致性和重复估算不修改状态。 |
| 合计 | 1280 | 保留全部执行结果和错误信息。 |

### 4.3 性能标准与测量方法

共 72 项，每项要求 `T_950 <= T_reference / 0.4`。未达标时按任务书要求给出解释，不以平均加速比抵消某一项不达标。原始标杆取自任务书 3.3 节，以下按接口给出上限范围，实际仍按各行对应的 dtype 和 KB 判断：

| 接口 | 用例数 | 配置 | 单例时延上限范围（ms） |
| --- | --- | --- | --- |
| Create | 12 | I32/I64 × 六档 KB。 | 1.008795–1.285595 |
| Destroy | 12 | I32/I64 × 六档 KB。 | 1.105345–1.2198275 |
| Clear | 12 | I32/I64 × 六档 KB。 | 0.072770–0.0830825 |
| Add | 12 | I32/I64 × 六档 KB，N=1e8，UNIFORM，Multiplicity=1。 | 0.7917275–5.3465625 |
| Merge | 12 | I32/I64 × 六档 KB，预先添加 N=1e8。 | 0.077055–0.1447525 |
| Estimate | 12 | I32/I64 × 六档 KB，预先添加 N=1e8。 | 0.0927075–0.2893125 |

沿用附件性能框架的初始化、预热和重复测量规则。附件 Measure 使用 CPU 墙钟微秒，而标杆为毫秒，比较前统一单位。它将同一墙钟结果写入两个时延字段，不能将其中一个当作独立测得的设备核耗时。

Create 计时包含构造及初始化完成，不包含随后析构；Destroy 在构造完成后单独计时；Add 只计同步 Add，随后 Clear 位于计时之外；Merge 与 Estimate 的输入添加发生在准备阶段。Merge 性能附件的两份 Sketch 来自相同输入，功能验证另覆盖重叠与不相交集合。UNIFORM 附件实际使用 1 至 1e8 的顺序整数，测试数据保持原样。

每项报告记录 dtype、KB、输入数量、标杆、实测时延、性能比、SDK/编译器/设备及测量方法。输入合法性检查、必要同步和结果回传均计入相应接口时间，不将异步提交耗时用于验收。

### 4.4 兼容性与接口边界

公共命名、RAII、Device 指针加数量和流参数顺序沿用 ops-collections 惯例；同步操作与对应 cuCollections 接口语义一致。无需修改其他容器公共接口。

标准差构造采用 3.3 节的 1.04 换算，三种工厂统一支持 p=13–18；与 cuCollections 的精度选择差异已明确列出。裸指针接口采用 3.7 节的模板类型约束、运行时检查和调用方前提。8 位寄存器遵循官方附件，不要求与参考实现的物理存储字节数相同。任务的误差标准和 72 项性能标准不因这些实现选择而放宽。

## 5. 参考资料

1. [任务书与原始测试附件](https://www.hiascend.com/p/resource/202609/66448a152b6f4bc1aaff5aad233cd467.zip)。
2. [cuCollections HyperLogLog 接口](https://github.com/NVIDIA/cuCollections/blob/532795b81e72e3fe4ce2b26eb0c5abc8abb1e2b4/include/cuco/hyperloglog.cuh)。
3. [cuCollections 容量换算及寄存器实现](https://github.com/NVIDIA/cuCollections/blob/532795b81e72e3fe4ce2b26eb0c5abc8abb1e2b4/include/cuco/detail/hyperloglog/hyperloglog_impl.cuh)及 [HLL++ finalizer](https://github.com/NVIDIA/cuCollections/blob/532795b81e72e3fe4ce2b26eb0c5abc8abb1e2b4/include/cuco/detail/hyperloglog/finalizer.cuh)。
4. [Spark HyperLogLogPlusPlusHelper](https://github.com/apache/spark/blob/6a27789ad7d59cd133653a49be0bb49729542abe/sql/catalyst/src/main/scala/org/apache/spark/sql/catalyst/util/HyperLogLogPlusPlusHelper.scala)。
5. [ops-collections BloomFilter 接口惯例](https://gitcode.com/cann/ops-collections/blob/9d12996d4317e28420d74bcb1ec4d3b3507599ce/docs/BloomFilter_API文档和使用示例.md)。
6. [CANN 9.0.0-beta.2 aclrtMemGetAddressRange](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta2/API/appdevgapi/aclcppdevg_03_2135.html)及 [aclrtPointerGetAttributes](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta2/API/appdevgapi/aclcppdevg_03_1803.html)。
7. [ops-collections StaticSet 输入约定](https://gitcode.com/cann/ops-collections/blob/9d12996d4317e28420d74bcb1ec4d3b3507599ce/include/static_set.h)及 [RoaringBitmap 有界解析与兼容构造](https://gitcode.com/cann/ops-collections/blob/9d12996d4317e28420d74bcb1ec4d3b3507599ce/docs/RoaringBitmap_design.md)。
