# 需求背景（required）

## 需求来源

本设计对应 **9月社区任务-hyperloglog容器开发(950)**，任务 ID 为 `f975dbdd02cb43eaacd0d7e3d3fa8821`，提交账号为 `AIAFKK`。

- [任务页面](https://www.hiascend.com/activities/task-center/details/f975dbdd02cb43eaacd0d7e3d3fa8821)
- [本题讨论帖](https://gitcode.com/cann/ops-collections/discussions/4)
- [设计模板](https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md)
- [社区任务流程](https://www.hiascend.com/developer/activities/cann-community-task)

设计依据为本题 ZIP 内 `hyperloglog_task_doc.md` 和配套原始测试；上游工程基线为 `cann/ops-collections` 提交 `9d12996d4317e28420d74bcb1ec4d3b3507599ce`；参考实现为 NVIDIA cuCollections 提交 `532795b81e72e3fe4ce2b26eb0c5abc8abb1e2b4` 的 `include/cuco/hyperloglog.cuh` 及其 detail 实现。任务书明确的接口、容量和验收标准优先于参考库的实现细节。

## 背景介绍

HyperLogLog（HLL）以固定容量 Sketch 估计整数集合的去重基数，避免存储全部输入。一个寄存器记录落入其桶内的最大 rank；Add 为单调更新，Merge 为逐寄存器最大值，Estimate 为只读近似估算。

ops-collections 已有 BloomFilter、StaticMap 等纯头文件容器，本题复用其命名空间、Extent、ACL 生命周期、SIMT 编译与测试框架。新实现不增加 ACLNN 注册、不引入 PyTorch 绑定，不新增仓库根目录下的独立算子工程。

# 需求分析（required）

## 需求描述

支持 I32、I64，提供按 SketchSizeKB、标准差及精度构造、析构、Clear、Add、Merge、Estimate。输入是一维连续 Device 数组；数量类型为 `aclco::Extent<std::size_t>`，计算和越界检查使用 64 位无符号整数。必须支持空输入、重复、分批、清空重用和兼容容器合并，同样配置和数据产生可重复状态及结果。

## 需求拆解

| 编号 | 原文位置 | 实现约束 | 验证与交付 |
| --- | --- | --- | --- |
| HLL-01 | 2.1、2.3、2.4 | 六类接口与三种构造；大驼峰；参数语义一致 | 原始六组功能测试、API 文档 |
| HLL-02 | 2.1、3.2 | I32/I64；整数哈希、rank、合并及估计 | 完整 Sketch 与独立 CPU 参考逐寄存器对照；原始精度判据 |
| HLL-03 | 2.1、3.2 | 顺序/重复/分批/清空重用、Estimate 只读且确定 | 原始用例及额外变形测试 |
| HLL-04 | 2.2、5 | 纯头文件，Host 管理与 Ascend C Kernel 分层 | 头文件多翻译单元构建、CTest 发现、README |
| HLL-05 | 2.4 | 容量、句柄、指针范围、流及合并兼容性检查 | 非法配置、移动后调用、越界、错误设备和流负例 |
| HLL-06 | 3.1 | Atlas 950 SIMT；CANN ≥9.0.0-beta.2 | 真机环境和编译器记录；当前验证以 CANN 9.1.0 为准 |
| HLL-07 | 3.2 | 每例相对误差 ≤ `3*1.04/sqrt(m)` | 精确去重基数、误差、阈值逐例输出；空集合必须为 0 |
| HLL-08 | 3.3 | 72 个配置均达到 ≥0.4 倍标杆性能 | 原始同步调用墙钟及补充 Kernel profiling；逐行时延上限 |
| HLL-09 | 3.4 | 不保存/复制 O(N) 输入副本；必要工作空间有上界 | 内存预算、分配记录、峰值和析构回收检查 |
| HLL-10 | 3.5、4 | 1280 个功能展开组合、72 个性能配置及全部日志 | 原始测试不删减；设计、源码、报告、复现步骤和代码地址 |

# 详细设计（required）

## 算子分析

### 容量和数学定义

本题明确 **1 字节存储 1 个 rank**，因此 `m = SketchSizeKB * 1024 = 2^p`。支持容量 `{8,16,32,64,128,256}` KB，对应 `p={13,14,15,16,17,18}`。cuCollections 当前使用 int32 寄存器；只借鉴其算法和 API 语义，不沿用其字节容量到寄存器数的换算。

默认哈希策略采用固定种子 0 的 64 位 xxHash，与参考实现的默认策略一致；复用/适配本仓 `xxhash_64<Key>`。类型宽度参与哈希，I32 处理 4 字节、I64 处理 8 字节；负整数按原始位模式处理，不通过有符号算术乘法产生溢出。Host 与 SIMT 哈希必须用独立已知值对照，不能仅让二者复制同一个错误。

对哈希值 `h`：

```text
j = h >> (64-p)
w = (h << p) | (1ULL << (p-1))
rank = count_leading_zeros(w) + 1
M[j] = max(M[j], rank)
```

哨兵位使 rank 落入 `[1,65-p]`，不会对 0 进行未定义的前导零运算。初始所有寄存器为 0。采用与 cuCollections 一致的高位桶索引，避免低位索引与参考实现产生不可比的状态。

Merge：`dst[j] = max(dst[j], src[j])`。合并满足交换律、结合律、幂等性；源 Sketch 不变，自合并允许且不改变结果。

Estimate：先统计 rank 直方图 `H[r]`，再以固定 r 顺序计算 `Z=sum(H[r]*2^-r)`、零寄存器数 `V=H[0]`、原始估计 `alpha_m*m*m/Z`。最终使用冻结 cuCollections 的 finalizer 所定义的小基数线性计数、偏差修正和精度相关阈值，适配本题的 `m=2^p`。保留参考代码许可说明。所有零寄存器返回 0；最终结果按参考 finalizer 的取整方式生成 `uint64_t`，不将原始 HLL 公式在修正区间的系统偏差误当成可接受随机误差。移植时对 p=13..18 的阈值和邻近切换点补充参考测试。

### 三种构造

按 KB 和 precision 构造仅接受上述集合。标准差 `s` 要求为有限正数：计算满足 `1.04/sqrt(2^p) <= s` 的最小 p，向上取精度，并限制在 `[13,18]`；若所需 p 小于 13，使用最小支持容量；若 p 大于 18，报错，不向下截断精度。使用比较校验修正 log2 在幂次边界处的舍入偏差。拒绝 0、负数、NaN、Inf 和超过支持精度的要求。

这是本题的有意兼容性差异：冻结 cuCollections 的标准差映射使用 1.106，本题按精度判据 1.04 与测试给定容量上界选择。原始测试 `s=0.0021` 对应本方案 `p=18、256 KiB`，而上游公式得到 p=19、超出本题容量。`s >= 1.04/sqrt(8192)` 使用 8 KiB；`s=1.04/sqrt(262144)` 使用 256 KiB，严格小于后者则拒绝。补测实际 SketchSizeKB 及这些阈值相邻浮点值。若社区要求完全复用 1.106，则须对这项任务/测试冲突取得裁定，不在实现时默改。

## 算子实现

### Host 公共接口

核心兼容接口如下；三种静态工厂直接对应任务测试的调用方式，Create/Destroy 的资源语义由工厂和 RAII 析构实现，不伪造 C 风格句柄：

```cpp
namespace aclco {
template<class Key>
class HyperLogLog {
public:
    static HyperLogLog CreateWithSketchSizeKB(uint32_t kb, aclrtStream stream = nullptr);
    static HyperLogLog CreateWithStandardDeviation(double sd, aclrtStream stream = nullptr);
    static HyperLogLog CreateWithPrecision(uint32_t precision, aclrtStream stream = nullptr);
    ~HyperLogLog() noexcept;
    HyperLogLog(HyperLogLog const&) = delete;
    HyperLogLog& operator=(HyperLogLog const&) = delete;
    HyperLogLog(HyperLogLog&& other) noexcept;
    HyperLogLog& operator=(HyperLogLog&& other) noexcept;
    void Clear(aclrtStream stream = nullptr);
    void Add(void* keys, aclco::Extent<std::size_t> keyNum, aclrtStream stream = nullptr);
    void Merge(HyperLogLog const& other, aclrtStream stream = nullptr);
    uint64_t Estimate(aclrtStream stream = nullptr) const;
    uint32_t SketchSizeKB() const noexcept;
    HyperLogLogRef<Key> Ref() &;
};
}
```

Key 使用编译期约束，只支持 `int32_t`、`int64_t`；不同 Key 的 Merge 编译期拒绝。配置中固定哈希算法/种子，不提供不能在 Merge 中核验一致性的任意状态哈希器。后续若增加可配置哈希，需要同时更新兼容性契约、测试和设计。

`include/hyperloglog_ref.h` 提供可传入 SIMT 的非持有引用：canonical Sketch 指针、p，以及单 Key 更新和只读配置访问。通过有效非 const 左值容器的 `Ref()` 取得；其生命周期短于容器；不分配、不释放内存。移动、失效标记及销毁均使此前取得的所有 Ref 失效，移动目标需要重新取得 Ref。公共容器的批量接口为本题主要验收面，Ref 的并发更新必须使用相同的原子协议，不允许普通字节 store 覆盖相邻桶。

### 生命周期、流和异常契约

| 状态/操作 | 行为 |
| --- | --- |
| 创建 | 校验参数和当前设备；分配 Sketch、固定上界的工作空间、必要的小结果区；清零并同步完成后返回；失败按下述清退协议回收，不在设备仍访问时 free |
| Clear/Add/Merge/Estimate | 均为同步 Host 接口，等待所用流完成再返回；错误通过异常报告；本轮不提供额外异步公共接口 |
| `stream=nullptr` | 直接向 Kernel/ACL 传入 nullptr，使用当前上下文默认流；与上游同步入口一致调用 aclrtSynchronizeStream(nullptr)，不替换成私有流；若后端不支持则明确报错并作为兼容性问题处理 |
| 空 Add | `(nullptr,0)` 合法；仍校验容器及流并遵循同步契约；非零数量空指针报错 |
| 移动 | 资源及设备归属唯一转移；源对象进入无资源状态；移动后源对象可析构/重新赋值，其他计算接口报错；自移动赋值不破坏对象 |
| 析构 | 不抛异常、不重复释放；正常公开接口均已同步；按下述清退协议释放；释放失败记录明确的 ACL 错误，不静默忽略 |
| 运行时失败 | 不伪装返回成功；若 Kernel 部分执行后失败，容器标为失效，先按下述协议清退，再允许安全释放/重新创建，不对外保证原子回滚 |

失败清退协议按提交边界区分：

1. 在首次设备提交前失败，可立即回收本次分配；输入未被设备借用。
2. 任何操作开始提交前先记录所用 stream、创建 context、已暴露给设备的内部资源和借用 keys；后续 launch、复制或同步失败时，先对已提交流尝试同步清退，必要时对所属设备同步。只有明确确认相关工作完成/停止后，才回收内部资源、解除 keys 借用并抛出普通运行错误。清退开销计入失败调用，不掩盖原始错误。
3. 若同步持续失败且不能确认设备访问已停止，禁止 free/复用涉及的 Sketch、workspace 和借用输入。内部资源移交进程内隔离记录，错误类型 `HyperLogLogRuntimeError` 提供 `RequiresQuiescence()` 状态、ACL 错误码和涉及的流/资源说明；调用者收到此类错误后必须继续保留 keys，直到外部确认 context 工作停止。Owner 不擅自 reset、重启或销毁调用方 context。只有确认静止后才能释放隔离资源；没有这个确认时明确报告未回收，不宣称无泄漏或恢复成功。
4. 构造失败、析构和 noexcept 移动赋值复用同一不抛出的清退/隔离底层策略。调用者必须保证创建 context 在清退及对象释放前有效；释放在资源所属 context 下执行并恢复调用线程原 context，不全局重置设备。外部 Ref 的工作由调用者按下一段先完成，Owner 不猜测其流。

增补故障注入：首次分配成功后失败、清零提交后失败、producer 提交成功而归并失败、结果复制失败、流与设备同步均失败。Host stub 只用于验证清退调用顺序和“不提前释放”的资源状态，不代替真机运行；正常析构无泄漏与致命设备错误的隔离记录分别报告。

同一容器的 Host 调用要求调用者串行化；不同容器可各自使用独立流。当前方案不宣称同一容器的跨流并发 Add/Clear/Merge 安全。顺序跨流调用受同步接口保护；不得假设所有外部输入都由本容器所在流生产，跨流输入由调用者通过 ACL event 或同步建立依赖。

记录创建时设备编号，操作前校验当前设备与容器一致，Merge 两侧设备必须一致。输入先检查 `N*sizeof(Key)` 乘法和指针加法溢出、对齐、Device 地址及剩余分配范围，再启动 Kernel。已安装 CANN 9.1.0 提供 `aclrtPointerGetAttributes` 与 `aclrtMemGetAddressRange`；按其真实支持范围实现，不以 `void*` 为由承诺能恢复已释放对象或任意无效 C++ 引用。非空流先通过已安装 SDK 的 aclrtStreamGetId 校验，ACL 拒绝的流和地址错误必须传递；支持范围内的普通 Device 分配必须通过负例实测，不能仅写文档。对 SDK 不支持查询的内存类型，明确报错，不静默跳过范围检查。

### Kernel 结构与同步

```text
Host 校验 → 按 N、m、UB 和核数分派
    小 N：直接更新 canonical Sketch
    大 N、小 Sketch：每核 UB 私有 rank 表 → 写出核局部表
    大 Sketch：有界 GM 私有表或直接 canonical 更新
→ 在同一流后续 Kernel 归并 → 同步 → 返回
```

canonical Sketch 为按 4 字节对齐的 m 字节区，4 个 rank 打包成一个 uint32。并发字节最大值更新必须通过整个 uint32 的 CAS 循环：提取目标字节，构造只替换该字节的新值，CAS 失败后以返回旧值重新计算；不能把整字比较大小当逐字节最大值。原子读取也使用已确认的原子语义，避免普通读和原子写混用的数据竞争。

大 N 的主要优化方向是每核先在 UB 内归约，避免大量全局随机原子。每核 UB 使用 uint32 rank 暂存并执行原子 max；初始化后及读取前所有活跃线程执行 `asc_syncthreads()`，不可把屏障放入尾部线程可能跳过的分支。线程通过 64 位 grid-stride 遍历输入，最后不足一 warp 的尾部有边界保护。核局部结果再由同流后续 Kernel 逐字归并至原有 Sketch，保留此前 Add 的状态。第二次 Kernel 启动提供跨核完成与可见性边界，不使用未验证的自旋全核屏障。

所有路径对相同输入产生相同 canonical Sketch。不得缓存输入指针或基数来跳过重复 Add，不得使用任务数据的连续整数特征替代读取与哈希；优化决策只依据公开配置、长度、设备资源和已完成测量。

### 分核、UB 与工作空间

核数由平台接口取得，当前 950PR 实测有 56 个 AIV 核；不在库中写死 56。小 N 根据 `ceil(N/每核最小工作量)` 减少核数，大 N 使用可用核数。线程数为编译期常量，`LAUNCH_BOUND` 与 `Simt::Dim3` 相同；初始候选 512/1024，按实测寄存器压力和吞吐选择。

| Sketch KB | m | uint32 私有表/核 | 初始策略 |
| --- | --- | --- | --- |
| 8 | 8192 | 32 KiB | UB 私有表 |
| 16 | 16384 | 64 KiB | UB 私有表 |
| 32 | 32768 | 128 KiB | UB 私有表，实测检查 DCache 与编译限制 |
| 64 | 65536 | 256 KiB | 不放入 UB；用有界 GM 私有表/直接更新 |
| 128 | 131072 | 512 KiB | 有界 GM 私有表/直接更新 |
| 256 | 262144 | 1 MiB | 有界 GM 私有表/直接更新 |

SIMT 必须为 DCache 和运行时保留空间；KG 指南给出的 256 KiB UB、至少 32 KiB DCache、8 KiB 保留是规划参考，实际分派服从当前 SDK/芯片查询与编译验证。共享表、规约暂存及栈合计必须在可用容量内，不能把 256 KiB 全部分配给共享表。大 Sketch 不采用多次扫描全部输入的桶分区方案作为默认实现。

GM 工作空间上界采用 `4*C*m` 字节（C 为实际 producer 核数），与输入数量 N 无关；56 核时最大为 56 MiB。可按容量分派缩减 C 或使用直接 canonical 更新降低内存。Sketch 本体最多 256 KiB；Estimate 暂存上界为 `65*C*sizeof(uint32_t)`，Host 接收数据仅为固定 rank 直方图，不复制输入。生命周期分配由 Create/Destroy 承担，其开销须在对应性能配置中体现；每次 Add 所需的清零/归并必须在 Add 计时范围内，不得隐藏到上次 Clear。

### Estimate 与 Merge 实现

Estimate Kernel 按固定分片生成整数直方图，后续固定顺序规约或 Host 按核、rank 顺序合并，使用足够位宽保证计数精确。只将 O(C*65) 小结果复制到 Host，再按冻结 finalizer 算法以固定顺序计算 double 标量和偏差修正。此过程不读取 Host 输入、不精确去重，不属于用 CPU 代替 Device 容器计算。Estimate 计时包含 Kernel、结果传输和同步以及最终标量计算。

Merge 先校验 Key、m/p、哈希配置和设备；按 uint32 字块并行处理、逐字节 max、每个输出字仅一个线程写入。因公共操作串行且源容器不变，Merge 无需全局 CAS。源 Sketch、目标原值和自合并均有测试覆盖。

### 代码布局与构建集成

```text
include/hyperloglog.h
include/hyperloglog_ref.h
include/detail/hyperloglog/hyperloglog.inl
include/detail/hyperloglog/hyperloglog_kernels.h
include/detail/hyperloglog/hyperloglog_estimator.h
tests/common/hll_test_common.h
tests/hyperloglog/{create,destroy,clear,add,merge,estimate}_test.cpp
tests/hyperloglog/contract_test.cpp
tests/performance/hyperloglog/perf_{create,destroy,clear,add,merge,estimate}.cpp
docs/hyperloglog_API文档和使用示例.md
```

更新 `tests/CMakeLists.txt` 的功能/性能源发现列表，以及 README 容器说明和运行入口；保留原始用例与基线。使用仓库实际 CMake 配置，ccec 为 `dav-c310`、毕昇为 `dav-3510`，目标选择以安装版本为准。头文件中非模板定义采用适当 inline，新增专用严格多翻译单元 consumer/link target：两个 TU 分别实例化并调用 HLL，禁用或不继承 `-Wl,--allow-multiple-definition`，适用编译器采用 `-fno-common`，保存真实链接参数。原仓通用测试目标含放宽重复定义选项，不能据其链接成功宣称 ODR 合格。若 SDK 固有符号必须放宽，先隔离确定这些符号，再单独检查 HLL 定义和导出符号，不能整体豁免。

## 支持硬件

Atlas 950 系列及后续具备所需 SIMT 能力的设备为适配目标。当前实际环境是 Ascend950PR、CANN 9.1.0、毕昇 ASC、Linux x86_64；后续设备及 CANN 最低版本的实际运行结果未验证时保持单独状态，不以一个目标架构编译成功宣称全系列已验收。

## 算子约束限制

输入须是当前设备上可查询分配范围的一维连续 I32/I64 数组，调用期间保持有效；对齐满足 Key；非连续布局由调用方整理。容器使用固定哈希策略，容量不可在线调整；不同容量/类型/设备不允许 Merge。无删除元素、精确计数、序列化或跨设备合并承诺。调用者保证 Ref 使用期间所有者有效；外部 Ref 工作须在所有 Owner API 前建立完成依赖，包括 const Estimate、作为 Merge 源/目标、再次 Ref 获取、移动和销毁。Owner 不跟踪外部流，不支持无依赖的 Ref 写入与任何读写并发。增补“外部流 Ref 更新→event/同步→Estimate/Merge”和“移动后重新取 Ref”的正例；旧 Ref 在移动后不得再使用。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 判据及统计范围 | 来源 |
| --- | --- | --- |
| 精度 | 非空集合 `abs(estimate-N)/N <= 3*1.04/sqrt(m)`；N=0 必须 estimate=0 | 任务书 3.2，原始 helper |
| 可重复性 | 相同配置/数据多次运行的 Sketch 和 Estimate 相同；分批/排列/重复/合并的等价输入保持状态一致 | 任务书 2.1、3.2 |
| 性能 | 每个 case 的 `baseline_ms/measured_ms >= 0.4`；使用完整同步 API 调用 | 任务书 3.3，原始 Measure |
| 内存 | 无 O(N) Device 输入副本；有界 workspace；销毁无资源泄漏 | 任务书 3.4 |

配套 Measure 用 Host chrono 包围 API 且外部不再同步，因此同步语义必须由公开 API 实现。cuCollections 的 `add`/`merge` 和本仓 BloomFilter 同步入口均明确等待流。原始框架把同一墙钟值填入 CPU/Device 两列，这不是真实 Kernel 时间；报告必须标明口径。补充 msprof 结果单列，不替换主性能判据。

原始测试包含 1280 个展开功能组合：Create 38、Destroy 24、Clear 180、Add 726、Merge 192、Estimate 120；另有 72 个性能配置。保存 Catch2 原始输出、生成参数、SECTION 轨迹和性能逐行结果；顶层 testcase 数、断言数与组合数分别统计，不能互相替代。跳过、失败和异常不得从最终汇总删除。

增补测试覆盖：标准差全部边界及非有限输入；有符号极值、随机负数、全部重复、不同排列、非整 warp 尾部；同一字内四个寄存器并发；一次/分批与多级 Merge 状态相同；连续两流顺序调用；move 赋值/自赋值/源对象失效；空指针与计数溢出、分配尾部越界、Host 指针；非法流和设备；分配失败回收；大容量路径；反复创建/析构及 ODR 链接。输入 golden 使用独立 CPU 实现与精确集合基数，哈希对照固定参考的已知向量。

### 当前可行性证据与待完成风险

开发前已在 950PR 运行独立 **8 KiB、fmix64 哈希、低位索引**原型，不能直接代表本设计默认 xxHash64/高位索引的性能或精度。百万输入 I32/I64 的完整 Sketch 与 CPU 对照一致；一亿输入测得相对误差约 1.22%。每核私有表方案的完整 Add+归并+同步均值为 I32 1.410347 ms、I64 1.591331 ms（1 次预热、5 次测量）。

这两个结果仍未达到正式上限 0.7917275 ms 和 0.941525 ms，约需 1.78/1.69 倍提速；且原型未实现完整容器、所有容量、默认哈希、负例及全部用例。本设计不声明验收已通过，也不将“给出解释”视为性能豁免。开发先验证默认哈希成本与 UB 原子 max、线程数/循环展开和分核策略，再按各容量选择已证明状态等价的路径。优化失败保留原始结果，不修改标杆、计时范围或数据。

### 开发、验证及交付顺序

1. 按社区流程提交本设计评审，处理意见；本地隔离实验与测试准备不作为官方评审通过证据。
2. 先接入原始测试，记录缺失实现的失败；实现 Host 生命周期/公共契约和完整正确性路径。
3. 优先攻克一亿 Add 的小 Sketch 性能，再验证 64/128/256 KiB 路径、Estimate/Merge 与生命周期性能；每次运行绑定候选源码、编译参数、二进制哈希、设备和日志。
4. 跑原始 1280 组合和 72 配置，补充契约/内存/干净克隆复现；增量修改只重跑受影响用例，候选收口再全量验证。
5. 准备设计合入证明、代码仓/分支/SHA/目录、完整自测报告和原始日志、测试步骤及 README。平台验收前核验设计已评审并合入及验收账号可访问代码；实现 PR 按本题测试通过后的官方流程提交。

## 兼容性分析

新增头文件和测试目录，不改变已有容器 API、分配器或哈希公开契约。库保持 C++/Ascend C 纯头文件，不强制运行时新依赖。与 cuCollections 保持输入/状态/同步及近似计数语义；差异明确为本题规定的 byte rank 存储、标准差容量映射系数 1.04（上游为 1.106，见构造章节）、静态工厂、ACL stream 与 `void*+Extent` 参数。最低版本和后续设备未验证的能力不冒称兼容通过。

## 设计依据与知识图谱使用

2026-09-16 已实际使用 Ascend KG 检索与加载 `ascendc-simt-best-practices`、`ascendc-simt-tiling-design` 及配套 guide，核对线程常量、grid-stride、UB/DCache 和原子操作路线。API 签名进一步由目标机 CANN 9.1.0 头文件核实；KG 的通用 ascend-kernel/PyTorch 工程模板不适用于本题，工程和验收仍以任务书与 ops-collections 为准。不能把 KG 说明代替编译、实跑或官方裁定。

# 附录：72 项性能验收上限

单位 ms；Estimate/Merge/Add 的 NumInputs 均为 100000000；Add 分布 UNIFORM、Multiplicity=1。

| 接口 | dtype | Sketch KB | 标杆 ms | 最大允许 ms |
| --- | --- | --- | --- | --- |
| create | I32 | 8 | 0.509452 | 1.2736300 |
| create | I32 | 16 | 0.437249 | 1.0931225 |
| create | I32 | 32 | 0.497847 | 1.2446175 |
| create | I32 | 64 | 0.431705 | 1.0792625 |
| create | I32 | 128 | 0.468345 | 1.1708625 |
| create | I32 | 256 | 0.447700 | 1.1192500 |
| create | I64 | 8 | 0.510325 | 1.2758125 |
| create | I64 | 16 | 0.445346 | 1.1133650 |
| create | I64 | 32 | 0.514238 | 1.2855950 |
| create | I64 | 64 | 0.442902 | 1.1072550 |
| create | I64 | 128 | 0.499370 | 1.2484250 |
| create | I64 | 256 | 0.403518 | 1.0087950 |
| destroy | I32 | 8 | 0.450497 | 1.1262425 |
| destroy | I32 | 16 | 0.442138 | 1.1053450 |
| destroy | I32 | 32 | 0.487931 | 1.2198275 |
| destroy | I32 | 64 | 0.480021 | 1.2000525 |
| destroy | I32 | 128 | 0.460845 | 1.1521125 |
| destroy | I32 | 256 | 0.470879 | 1.1771975 |
| destroy | I64 | 8 | 0.474998 | 1.1874950 |
| destroy | I64 | 16 | 0.471362 | 1.1784050 |
| destroy | I64 | 32 | 0.457632 | 1.1440800 |
| destroy | I64 | 64 | 0.464786 | 1.1619650 |
| destroy | I64 | 128 | 0.477563 | 1.1939075 |
| destroy | I64 | 256 | 0.450429 | 1.1260725 |
| clear | I32 | 8 | 0.031079 | 0.0776975 |
| clear | I32 | 16 | 0.029187 | 0.0729675 |
| clear | I32 | 32 | 0.029462 | 0.0736550 |
| clear | I32 | 64 | 0.029917 | 0.0747925 |
| clear | I32 | 128 | 0.030977 | 0.0774425 |
| clear | I32 | 256 | 0.033233 | 0.0830825 |
| clear | I64 | 8 | 0.029108 | 0.0727700 |
| clear | I64 | 16 | 0.029241 | 0.0731025 |
| clear | I64 | 32 | 0.029420 | 0.0735500 |
| clear | I64 | 64 | 0.029899 | 0.0747475 |
| clear | I64 | 128 | 0.031733 | 0.0793325 |
| clear | I64 | 256 | 0.033185 | 0.0829625 |
| estimate | I32 | 8 | 0.037083 | 0.0927075 |
| estimate | I32 | 16 | 0.040014 | 0.1000350 |
| estimate | I32 | 32 | 0.046175 | 0.1154375 |
| estimate | I32 | 64 | 0.054031 | 0.1350775 |
| estimate | I32 | 128 | 0.068271 | 0.1706775 |
| estimate | I32 | 256 | 0.099492 | 0.2487300 |
| estimate | I64 | 8 | 0.037359 | 0.0933975 |
| estimate | I64 | 16 | 0.040540 | 0.1013500 |
| estimate | I64 | 32 | 0.047773 | 0.1194325 |
| estimate | I64 | 64 | 0.060547 | 0.1513675 |
| estimate | I64 | 128 | 0.075881 | 0.1897025 |
| estimate | I64 | 256 | 0.115725 | 0.2893125 |
| merge | I32 | 8 | 0.032128 | 0.0803200 |
| merge | I32 | 16 | 0.031713 | 0.0792825 |
| merge | I32 | 32 | 0.033607 | 0.0840175 |
| merge | I32 | 64 | 0.036923 | 0.0923075 |
| merge | I32 | 128 | 0.043673 | 0.1091825 |
| merge | I32 | 256 | 0.057226 | 0.1430650 |
| merge | I64 | 8 | 0.030822 | 0.0770550 |
| merge | I64 | 16 | 0.031887 | 0.0797175 |
| merge | I64 | 32 | 0.033782 | 0.0844550 |
| merge | I64 | 64 | 0.036920 | 0.0923000 |
| merge | I64 | 128 | 0.043677 | 0.1091925 |
| merge | I64 | 256 | 0.057901 | 0.1447525 |
| add | I32 | 8 | 0.316691 | 0.7917275 |
| add | I32 | 16 | 0.317554 | 0.7938850 |
| add | I32 | 32 | 0.317865 | 0.7946625 |
| add | I32 | 64 | 0.321785 | 0.8044625 |
| add | I32 | 128 | 0.458247 | 1.1456175 |
| add | I32 | 256 | 2.031375 | 5.0784375 |
| add | I64 | 8 | 0.376610 | 0.9415250 |
| add | I64 | 16 | 0.377037 | 0.9425925 |
| add | I64 | 32 | 0.377799 | 0.9444975 |
| add | I64 | 64 | 0.381595 | 0.9539875 |
| add | I64 | 128 | 0.509386 | 1.2734650 |
| add | I64 | 256 | 2.138625 | 5.3465625 |
