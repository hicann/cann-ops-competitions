# HyperLogLog 容器设计文档

# 需求背景（required）

## 需求来源

- 来源：CANN 社区任务「9月社区任务-hyperloglog容器开发(950)」。
- 对标接口：cuCollections `cuco::hyperloglog`（HyperLogLog++ 算法），https://github.com/NVIDIA/cuCollections/blob/dev/include/cuco/hyperloglog.cuh 。
- 目标仓库：https://gitcode.com/cann/ops-collections （容器库 `ops-collections`）。
- 交付形态：与 `bloom_filter`、`roaring_bitmap`、`static_map` 等既有容器同构的头文件容器 + Catch2 功能用例 + 性能用例 + API 文档。

## 背景介绍

基数估计（cardinality estimation）用于统计海量数据流中的不重复元素个数。HyperLogLog 用固定大小的 Sketch 存储每个寄存器（register）观测到的哈希前缀「最长零串长度」，通过调和平均与偏差修正得到基数估计，空间复杂度为 O(m)，相对标准误差约为 `1.04 / sqrt(m)`。

cuCollections 的实现基于 HyperLogLog++（HLL++）论文：

- 小基数范围使用线性计数（linear counting），避免原始估计器在稀疏分布下的高估；
- 中段使用论文给出的经验 bias 表 + 插值做偏差修正；
- 大基数范围回退到经典 HLL 估计器（`p = 19` 以上无 bias 表）。

ops-collections 仓库已有 `bloom_filter` 容器，其 SIMT kernel 组织方式（`kernel_operator.h` + `simt_api` + `COLLECTION_AIV_GLOBAL` / `VF_CALL`）、内存分配器（`aclco::DefaultAllocator`）、`Extent` 计数类型、测试脚手架（`tests/common/acl_env.h`、`tests/common/device_buffer.h`、`tests/performance/performance_test_framework.h`）均为本容器复用，未新增仓库根目录下的独立算子目录。

### 现状分析

| 维度 | cuCollections（GPU/CUDA） | 本容器（Ascend 950 / SIMT） |
| --- | --- | --- |
| 线程模型 | CUDA block/warp，warp 内先归约再原子更新 | SIMT 核 × 1024 线程，逐 key 原子 CAS |
| 寄存器更新 | warp 级 max 归约 + 单次原子 max | 32-bit word 打包 4 个 8-bit 寄存器 + CAS 循环 |
| 估计器归约 | device 侧 block 归约后写回 | kernel 写每线程独占 partial，host 侧固定顺序求和 |
| 内存 | `cuda::std::span<std::byte>` 动态 span | 单次 `aclrtMalloc`（Sketch + partial 合并为一块） |

# 需求分析（required）

## 需求描述

实现 `aclco::HyperLogLog<Key>` 容器，使其在 Atlas 950 系列产品上支持如下能力：

1. 支持 `CreateWithSketchSizeKB` / `CreateWithStandardDeviation` / `CreateWithPrecision` 三种构造入口，以及析构、`Clear`、`Add`、`Merge`、`Estimate`。
2. `Key` 支持 `int32_t`、`int64_t`（均须满足精度要求）。
3. 接口入参顺序与 cuCollections 保持一致；Ascend C 当前缺少 device iterator，用「`void*` 起始地址 + `Extent<size_t>` 元素个数 + `aclrtStream`」替代。
4. Sketch 规模与寄存器数一一对应：`sketchSizeKB ∈ {8,16,32,64,128,256}`，`m = sketchSizeKB × 1024`，精度 `p = log2(m) ∈ [13,18]`。
5. 精度：对精确基数为 N 的输入，`|Estimate - N| / N ≤ 3 × 1.04 / sqrt(m)`；空 Sketch 的 `Estimate` 必须为 0。
6. 确定性：相同输入的重复执行结果必须一致；分批 `Add` 与一次性 `Add` 的估计值必须完全相等。
7. 非法输入必须抛出异常：`sketchSizeKB ∈ {0,7,512}`、`precision = 0`、`Add(nullptr, n>0)`、配置不兼容的 `Merge`。
8. 性能：所有性能用例时延 ≤ 标杆时延 / 0.4。

## 需求拆解

| 编号 | 需求项 | 设计拆解 |
| --- | --- | --- |
| 1 | 构造 / 析构 | 校验 sketch 配置 → 单次 device 分配（Sketch + 估计 partial 缓冲）→ `Clear`；析构释放整块内存 |
| 2 | Clear | `m/4` 个 32-bit word 置零的 SIMT 核，按核数网格步长写 |
| 3 | Add | `xxhash_64` 求 64 位哈希 → 高 `p` 位取寄存器下标 → 余位取 rank（哨兵位保证 rank 有界）→ word 级 CAS 更新 |
| 4 | Merge | 同配置 Sketch 逐 word 做「四寄存器逐字节取大」，幂等、可交换 |
| 5 | Estimate | device 侧把每个寄存器折算为 `2^-M` 定点值（40 位小数）并统计零寄存器个数 → 每线程写独占 partial → host 侧固定顺序归约 → HLL++ finalizer（线性计数 / bias 表插值 / 经典 HLL） |
| 6 | dtype 适配 | 统一 64 位哈希路径，`int32_t` 走 4 字节尾哈希、`int64_t` 走 8 字节尾哈希 |
| 7 | 精度达标 | 移植 HLL++ bias 表（p13–p18）与插值逻辑，小基数用线性计数，中段做偏差修正 |
| 8 | 性能达标 | Add 先读后 CAS（已饱和的寄存器不再产生原子流量）；Clear/Merge 为纯带宽路径；Estimate 固定核数（2 核）避免核启动开销 |
| 9 | 异常语义 | 参数非法抛 `std::invalid_argument`，分配失败抛 `std::bad_alloc`，ACL 调用失败抛 `std::runtime_error`，move-from 对象使用抛 `std::logic_error` |

## 接口设计

```cpp
template <class Key, class Extent = aclco::Extent<std::size_t>,
          class Allocator = aclco::DefaultAllocator<std::uint8_t>>
class HyperLogLog {
 public:
  using KeyType = Key;
  using ExtentType = Extent;
  using AllocatorType = Allocator;

  HyperLogLog(HyperLogLog const&) = delete;
  HyperLogLog& operator=(HyperLogLog const&) = delete;
  HyperLogLog(HyperLogLog&&) noexcept;
  HyperLogLog& operator=(HyperLogLog&&) noexcept;
  ~HyperLogLog();

  static HyperLogLog CreateWithSketchSizeKB(std::uint32_t sketchSizeKB, aclrtStream stream = nullptr);
  static HyperLogLog CreateWithStandardDeviation(double standardDeviation, aclrtStream stream = nullptr);
  static HyperLogLog CreateWithPrecision(std::uint32_t precision, aclrtStream stream = nullptr);

  std::uint32_t SketchSizeKB() const noexcept;
  std::uint32_t Precision() const noexcept;
  std::uint64_t RegisterCount() const noexcept;
  double StandardError() const noexcept;

  void Add(void* keys, ExtentType keyNum, aclrtStream stream = nullptr);
  void AddAsync(void* keys, ExtentType keyNum, aclrtStream stream = nullptr);
  void Merge(HyperLogLog const& other, aclrtStream stream = nullptr);
  void MergeAsync(HyperLogLog const& other, aclrtStream stream = nullptr);
  void Clear(aclrtStream stream = nullptr);
  void ClearAsync(aclrtStream stream = nullptr);
  std::uint64_t Estimate(aclrtStream stream = nullptr) const;
};
```

| 接口 | 语义 |
| --- | --- |
| `CreateWithSketchSizeKB(kb, stream)` | 分配 `kb × 1024` 个寄存器（1 字节/寄存器）的 Sketch，并在 `stream` 上清零；`kb ∉ {8,16,32,64,128,256}` 抛异常 |
| `CreateWithStandardDeviation(sd, stream)` | 取满足 `1.04 / sqrt(2^p) ≤ sd` 的最小 `p`（超出上限时取 256KB） |
| `CreateWithPrecision(p, stream)` | `p ∈ [13,18]`，对应 `2^(p-10)` KB；`p = 0` 抛异常 |
| `Add(keys, n, stream)` | 把 `n` 个连续 device key 加入 Sketch；`n = 0` 为空操作；`keys = nullptr && n > 0` 抛异常 |
| `Merge(other, stream)` | 逐寄存器取大，原地合并；Sketch 配置不同抛异常；幂等、可交换 |
| `Clear(stream)` | 全部寄存器清零，可重复调用 |
| `Estimate(stream)` | 返回去重基数估计；空 Sketch 返回 0；重复调用结果一致且不改变状态 |

# 详细设计（required）

## 算子分析

### 数学公式

设寄存器数为 `m = 2^p`，第 `j` 个寄存器的值为 `M[j] ∈ [0, 64 - p + 1]`，`V` 为零寄存器个数。

对 key 的 64 位哈希 `H(key)`：

$$
idx = H(key) \gg (64 - p)
$$

$$
M[idx] \leftarrow \max\left(M[idx],\ \text{clz}\left((H(key) \ll p) \mid (1 \ll (p-1))\right) + 1\right)
$$

其中 `1 << (p-1)` 为哨兵位，保证哈希低位全零时 rank 仍有定义，且 `rank ∈ [1, 64-p+1] ≤ 52`，即 8-bit 寄存器足够。

原始估计器与调和平均：

$$
Z = \sum_{j=0}^{m-1} 2^{-M[j]}, \qquad E = \alpha_m \frac{m^2}{Z}
$$

$$
\alpha_m = \frac{0.7213}{1 + 1.079 / m}
$$

HLL++ finalizer：

$$
LC = m \ln\frac{m}{V}
$$

- 若 `V > 0` 且 `E ≤ 2.5m`：返回 `round(LC)`；
- 若 `V > 0` 且 `E > 2.5m`：`LC ≤ threshold(p)` 时返回 `round(LC)`，否则返回 `round(E - bias(E))`；
- 若 `V = 0`：`p < 19` 时返回 `round(E - bias(E))`，否则返回 `round(E)`。

`bias(E)` 为论文经验表中的最近 6 个 anchor 做均值（anchor 由二分插值定位），`threshold(p)` 为论文给出的线性计数切换阈值（p13–p18：6500 / 15500 / 20000 / 50000 / 120000 / 350000）。

### 支持数据类型

| dtype | 哈希路径 | 说明 |
| --- | --- | --- |
| `int32_t` | XXH64 4 字节尾（`key * prime1` → rotl 23 → `* prime2` → finalize） | 与仓库 `xxhash_64<int32_t>` 完全一致 |
| `int64_t` | XXH64 8 字节尾（lane 混合 → rotl 27 → finalize） | 与仓库 `xxhash_64<int64_t>` 完全一致 |

### 支持形状

输入为 device 上连续存放的一维 key 数组，元素个数由 `Extent<std::size_t>` 指定，不涉及多维 shape / stride。Sketch 为固定大小的一维寄存器数组。

## 算子实现

### 实现方案

#### host 侧设计

1. **配置映射**：`sketchSizeKB ↔ precision` 严格一一对应（8KB↔13 … 256KB↔18）；`CreateWithStandardDeviation` 按 `ceil(2·log2(1.04 / sd))` 求精度并夹到 `[13,18]`。所有校验在分配前完成，非法参数不产生任何 device 资源。
2. **单次分配**：Sketch 字节数（`2^p`，必为 8 的倍数）与估计 partial 缓冲（2048 线程 × 2 × 8B = 32KB）合并为一次 `Allocator::Allocate`，Sketch 在前、partial 紧随其后，天然满足 8 字节对齐；析构时一次 `Deallocate`。
3. **流语义**：`Add` / `Merge` / `Clear` / `Estimate` 在返回前同步调用方 stream；`*Async` 版本只入队。`Estimate` 内部在 `stream` 上做 `aclrtMemcpyAsync`(D2H, 32KB) 后再同步，保证读到的是当前 Sketch 对应的 partial。
4. **异常语义**：构造/参数校验抛 `std::invalid_argument`；`aclrtMalloc` 失败抛 `std::bad_alloc`；ACL 调用返回非 `ACL_SUCCESS` 抛 `std::runtime_error`；move-from 对象参与运算抛 `std::logic_error`。
5. **分核策略**：`LaunchCoreNum(workItems) = clamp(ceil(workItems / 1024), 1, AIV核数)`。Add 以 key 数为工作量（大输入自动打满 256 核），Clear/Merge 以 word 数为工作量，Estimate 固定 2 核。

#### kernel 侧设计

1. **寄存器打包**：4 个 8-bit 寄存器打包进 1 个 32-bit word（`m/4` words）。选 32-bit 为 CAS 粒度是因为 CANN 当前无字节级原子操作；rank ≤ 52 只占低 7 位，更新时只改目标字节、其余字节原样保留。
2. **Add kernel**：每线程按全局线程号做网格步长遍历 key；先读取目标 word，若目标寄存器已 ≥ 本次 rank 直接返回（热路径不入原子），否则 `asc_atomic_cas` 循环重试。
3. **Merge kernel**：逐 word 做 `PackedRegisterMax`（4 组 `max` 后回写），无原子、无竞争。
4. **Clear kernel**：逐 word 写 0。
5. **Estimate kernel**：`2` 核 × `1024` 线程 = 2048 线程，每线程网格步长遍历寄存器，累加 40 位定点 `sum(2^-M)` 与零寄存器个数，写入 `partials[2·t]` / `partials[2·t+1]`（每线程独占槽位，无原子、无 block 内通信）；host 侧按固定槽位顺序求和后交给 finalizer。
6. **确定性**：寄存器取大与顺序无关；估计值由 host 侧固定顺序求和 + 纯函数 finalizer 得到，因此与核数、线程调度、分批方式无关，满足「分批 == 一次性」「重复调用 == 首次调用」。

### 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas 950 系列（SIMT 能力） | √ |
| Atlas A2 / A3 | 不涉及（依赖 SIMT 原子与 VF 线程模型） |

### 算子约束限制

1. `sketchSizeKB` 必须为 `{8,16,32,64,128,256}` 之一，`precision` 必须为 `[13,18]`。
2. `Key` 仅支持 `int32_t` / `int64_t`（`static_assert` 约束）。
3. `Merge` 要求两个 Sketch 的 `sketchSizeKB` 相同，否则抛异常。
4. 容器对象本身不做线程安全承诺：同一对象上的 host 侧调用需串行；不同对象可并发在不同 stream 上使用。
5. `Add` 的输入必须是 device 可访问的连续 key 数组，元素类型与 `Key` 一致。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | 对精确基数 N：相对误差 ≤ `3 × 1.04 / sqrt(m)`；空 Sketch `Estimate == 0`；分批/一次性结果一致 | 任务书 3.2 |
| 性能标准 | 所有性能用例时延 ≤ 标杆时延 / 0.4（覆盖 create/destroy/clear/add/estimate/merge 共 72 例） | 任务书 3.3 |

## 测试方案

功能用例（提供的 Catch2 用例，按 dtype 实例、`GENERATE` 参数组合与 `SECTION` 分支展开）：

| 接口 | dtype | 用例数 | 关键覆盖点 |
| --- | --- | --- | --- |
| Create | I32、I64 | 38 | 6 种 sketchSizeKB、6 种标准差、6 种 precision、非法参数抛异常 |
| Destroy | I32、I64 | 24 | 反复构造/析构、move 构造转移所有权 |
| Clear | I32、I64 | 180 | 空 Sketch 幂等清空、清空后 Estimate 为 0、清空后复用 |
| Add | I32、I64 | 726 | 空/小/中/大输入、重复键分布、分批 == 一次性、nullptr 抛异常、1e8 大输入 |
| Merge | I32、I64 | 192 | 空与空、空与非空、幂等、重叠/相离、配置不兼容抛异常 |
| Estimate | I32、I64 | 120 | 基数 0…1048576 的精度判据、两实例结果一致、重复调用不变 |
| 合计 | - | 1280 | - |

性能用例：create/destroy/clear（36 例）、estimate/merge（24 例）、add（12 例），合计 72 例。

复现步骤：

```bash
cd ops-collections
bash scripts/build.sh -b                                        # 构建（含 Catch2 v3.5.4）
bash scripts/build.sh -r --test-name hyperloglog                # 运行 HyperLogLog 功能用例
bash scripts/build.sh -p                                        # 构建性能用例
bash scripts/build.sh -rp                                       # 运行性能用例
```

## 性能优化策略

| 路径 | 策略 |
| --- | --- |
| Add | 先读后 CAS：寄存器已达目标 rank 时零原子流量；大输入打满 AIV 核，避免核内串行 |
| Clear / Merge | 纯带宽路径（写零 / 逐 word 取大），核数按 word 数自适应 |
| Estimate | 固定 2 核 + 每线程独占 partial，避免全局原子与 block 内同步；D2H 仅 32KB |
| 构造 / 析构 | Sketch 与 partial 合并为一次分配/释放，减少一半 ACL 内存调用 |

## 风险与规避措施

| 风险 | 影响 | 规避措施 |
| --- | --- | --- |
| word 级 CAS 竞争（大基数、小 Sketch） | Add 时延上升 | 先读后 CAS + 网格步长保证同 warp 相邻线程落在不同 word；如不达标可再加 warp 内 rank 归约 |
| 定点量化误差 | 估计值轻微偏移 | 40 位小数定点（`2^-40` 量化步长），误差 < 1e-12 量级，与 finalizer 的 double 路径等价 |
| bias 表与哈希分布不匹配 | 中段精度变差 | 采用论文 p13–p18 全量 anchor/bias 表 + 6 点插值，哈希使用仓库既有 `xxhash_64`（雪崩效应充分） |
| `Estimate` 的 D2H 同步开销 | 时延靠近门槛 | partial 缓冲固定 32KB，memcpy 与 finalizer 均在 host 侧一次完成 |

## 兼容性分析

新容器，不涉及既有接口兼容性；头文件与既有容器并列（`include/hyperloglog.h`、`include/hyperloglog_ref.h`、`include/detail/hyperloglog/`），不改动既有容器行为，仅新增 CMake 注册与 README 说明。
