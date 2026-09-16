# HyperLogLog 容器算子设计文档

本文面向 `ops-collections` 中的 HyperLogLog 容器实现，定义公开接口、
数据结构、Ascend C 设备侧执行流程、资源管理、异常处理和验收方案。
设计以本任务书要求和 cuCollections `hyperloglog` 的接口语义为依据，
只描述本容器需要实现的行为，不修改官方任务书。

## 需求背景

在流量统计、日志分析和数据去重等场景中，系统经常需要估算大规模输入
集合的不同元素数量。精确维护集合会使内存占用随输入规模增长，而
HyperLogLog 使用固定大小的 Sketch 保存统计状态，可以在受控误差范围内
提供基数估算。

本需求将 cuCollections 的 HyperLogLog 容器语义迁移到昇腾 NPU 和
`ops-collections` 的 Ascend C 容器工程中。容器需要适配 Device 侧批量
输入、ACL 执行流和 NPU Kernel，并为 `I32`、`I64` 输入提供统一的生命周期
和计算接口。

## 需求分析

需求分析将功能、接口、正确性、资源和测试边界拆分为可实现、可验证的
设计约束。

- **功能需求**：支持按 Sketch 容量、标准差或精度创建容器，并提供析构、
  `Clear`、`Add`、`Merge` 和 `Estimate` 操作；设备侧需要完成 Key 哈希、
  寄存器更新、Sketch 合并和基数估算。
- **接口需求**：公开接口采用大驼峰命名，输入 Key 来自 Device 地址，
  操作通过 `aclrtStream` 提交；接口参数顺序和语义与 cuCollections 对应
  接口保持一致。
- **状态需求**：支持空输入、重复输入、分批追加、清空后重用和兼容 Sketch
  合并；同一输入、配置和哈希策略必须得到可重复的状态变化与估算语义。
- **正确性需求**：同时支持 `I32` 和 `I64`，空 Sketch 的估算值为零，
  非法容量、非法精度、空句柄、非法输入地址和不兼容合并必须被校验；
  估算误差按 HyperLogLog 的规定标准判定。
- **性能与资源需求**：`Add` 不产生与输入规模线性重复的额外 Device
  拷贝，其他操作使用与 Sketch 容量相关的固定资源；性能测试需要覆盖
  全部支持的 Key 类型和 Sketch 配置，并按任务规定的相对标杆条件判定。
- **工程需求**：实现遵循纯头文件容器模式，将公开接口、设备侧引用、
  实现细节、功能测试和性能测试分层组织，便于后续提交到目标开源仓库。

## 1. 设计概述

HyperLogLog 是一种使用固定内存近似统计基数的数据结构。本设计将输入的
`I32` 或 `I64` 元素映射到固定数量的寄存器，通过寄存器最大值估算去重后
的元素数量。容器状态保存在 Device 侧，`Add`、`Clear`、`Merge` 和
`Estimate` 通过 Ascend C Kernel 完成。

本设计需要满足以下目标：

- 提供按 Sketch 容量、标准差和精度创建容器的方式。
- 支持 `Create`、`Destroy`、`Clear`、`Add`、`Merge` 和 `Estimate` 全部
  生命周期与计算操作。
- 支持 `int32_t` 和 `int64_t` 两种 Key 类型，并保证同一输入、配置和
  哈希策略下结果可重复。
- 不因输入数量产生线性额外 Device 内存拷贝。
- 在 Atlas 950 和支持 SIMT 能力的昇腾设备上使用 Ascend C 完成设备侧
  批量计算。

本次实现不提供序列化、跨设备迁移和动态改变 Sketch 配置的接口。容器的
Sketch 容量在创建时确定，后续只能清空、追加、合并或估算。

## 2. 约束与术语

本节统一实现中使用的参数、缩写和环境约束，避免 API 文档与测试代码对
同一概念产生不同解释。

### 2.1 参数与术语

下表给出核心术语的含义。

| 术语 | 含义 |
| --- | --- |
| Key | 输入元素的编译期类型，仅支持 `int32_t` 和 `int64_t`。 |
| Sketch | 由 `m` 个 8 位寄存器组成的 Device 侧状态。 |
| `SketchSizeKB` | Sketch 寄存器区的容量，单位为 KiB；任务接口名称保留为 `KB`。 |
| `m` | 寄存器数量，等于 `SketchSizeKB * 1024`。 |
| `p` | 精度参数，满足 `m = 2^p`，取值为 13 至 18。 |
| `keyNum` | 本次 `Add` 需要处理的输入元素数量。 |
| `Estimate` | 根据当前寄存器状态返回的近似去重基数。 |

Sketch 使用一个字节保存一个寄存器，因此支持的寄存器数量为 8192、
16384、32768、65536、131072 和 262144。

### 2.2 软硬件环境

实现和验收使用任务书指定的环境约束。

| 项目 | 要求 |
| --- | --- |
| 设备 | Atlas 950 系列及后续支持 SIMT 的昇腾产品。 |
| CANN | CANN 9.0.0-beta.2 或更高版本。 |
| 开发语言 | Ascend C 和 C++。 |
| 编译器 | 仓库支持的 `ccec` 或毕昇 ASC 编译器。 |
| CMake | 3.16 或更高版本。 |
| 测试框架 | Catch2 v3.5.4 或更高版本。 |

目标架构和编译选项必须与实际安装的 CANN 版本及设备型号匹配。

## 3. 对外接口设计

公开接口位于 `include/hyperloglog.h`。容器通过模板参数确定 Key 类型，
通过 `aclrtStream` 控制设备侧操作的执行顺序。接口命名采用大驼峰，
参数顺序和参数语义与 cuCollections 对应接口保持一致。

### 3.1 类接口

以下接口构成最小可用的 HyperLogLog 容器 API。`Extent` 使用仓库现有
类型，接口中的 `std::size_t` 仅用于表达元素数量的底层类型。

```cpp
template <typename Key>
class HyperLogLog {
 public:
  static HyperLogLog CreateWithSketchSizeKB(
      std::uint32_t sketch_size_kb, aclrtStream stream = nullptr);

  static HyperLogLog CreateWithStandardDeviation(
      double standard_deviation, aclrtStream stream = nullptr);

  static HyperLogLog CreateWithPrecision(
      std::uint32_t precision, aclrtStream stream = nullptr);

  HyperLogLog(HyperLogLog &&other) noexcept;
  HyperLogLog &operator=(HyperLogLog &&other) noexcept;
  HyperLogLog(HyperLogLog const &) = delete;
  HyperLogLog &operator=(HyperLogLog const &) = delete;
  ~HyperLogLog();

  void Clear(aclrtStream stream = nullptr);

  void Add(void *keys, Extent<std::size_t> key_num,
           aclrtStream stream = nullptr);

  void Merge(HyperLogLog const &other, aclrtStream stream = nullptr);

  std::uint64_t Estimate(aclrtStream stream = nullptr) const;

  std::uint32_t SketchSizeKB() const noexcept;
};
```

其中，模板参数 `Key` 在编译期限定为 `int32_t` 或 `int64_t`。`keys` 是
Device 侧 Key 数组的首地址，不能传入 Host 侧数组地址。`key_num` 为零
时允许 `keys` 为空；`key_num` 非零时 `keys` 必须为非空有效 Device
地址。

### 3.2 接口行为

各接口的状态变化和同步语义如下。

| 接口 | 行为 | 状态变化 | 返回值 |
| --- | --- | --- | --- |
| `CreateWithSketchSizeKB` | 校验容量、分配寄存器区并初始化为零。 | 产生空 Sketch。 | 新容器对象。 |
| `CreateWithStandardDeviation` | 将目标标准差映射为 `p` 后创建容器。 | 产生空 Sketch。 | 新容器对象。 |
| `CreateWithPrecision` | 校验 `p` 并按 `2^p` 个寄存器创建。 | 产生空 Sketch。 | 新容器对象。 |
| `Clear` | 将所有寄存器重置为零。 | 当前 Sketch 变为空。 | 无。 |
| `Add` | 读取 `key_num` 个 Device 侧 Key，并更新寄存器最大值。 | 累加输入信息。 | 无。 |
| `Merge` | 对两个相同配置的 Sketch 逐寄存器取最大值。 | 修改当前对象，不修改 `other`。 | 无。 |
| `Estimate` | 读取寄存器并计算近似基数。 | 不修改 Sketch。 | `uint64_t`。 |
| 析构 | 等待本对象已提交的设备操作完成后释放 Device 资源。 | 释放对象持有的资源。 | 无。 |

`stream == nullptr` 使用 ACL 默认流。对于非空流，调用方必须保证同一
容器上的操作按需要使用同一流，或者在不同流之间显式建立同步关系。
`Add`、`Clear` 和 `Merge` 只提交设备操作；`Estimate` 需要返回 Host
侧标量，因此在返回前完成结果拷贝和必要的流同步。

任务书中的 `Create` 和 `Destroy` 是容器生命周期操作。本设计通过三个
静态工厂表达 `Create`，通过析构函数和 RAII 表达 `Destroy`，不再提供可
被重复调用的独立 `Destroy` 成员函数。这样与现有测试中使用对象析构、
`std::optional::reset` 和智能指针释放的方式一致。

### 3.3 创建参数映射

三种创建方式最终都转换为同一个精度参数 `p` 和寄存器数量 `m`。

| 创建方式 | 合法输入 | `p` | `m` |
| --- | --- | --- | --- |
| `SketchSizeKB` | 8、16、32、64、128、256 | 13、14、15、16、17、18 | `SketchSizeKB * 1024` |
| `Precision` | 13、14、15、16、17、18 | 输入值 | `2^Precision` |
| `StandardDeviation` | 约为 0.0115、0.0082、0.0058、0.0041、0.0029、0.0021 | 13 至 18 | `2^p` |

标准差按照 HyperLogLog 的理论误差 `1.04 / sqrt(m)` 计算并与支持的
精度档位匹配。实现必须对超出支持范围或无法匹配的参数抛出异常，不能
静默选择相邻配置。

### 3.4 参数校验

参数校验在 Host 侧完成，设备 Kernel 只接收已完成基本校验的参数。

| 参数 | 校验规则 | 失败行为 |
| --- | --- | --- |
| `Key` | 只能是 `int32_t` 或 `int64_t`。 | 编译期拒绝不支持的类型。 |
| `hll` | 对象必须已成功创建且资源有效。 | 抛出异常。 |
| `keys` | `key_num > 0` 时必须是有效 Device 地址。 | 抛出异常。 |
| `key_num` | 不能使输入范围发生整数溢出。 | 抛出异常。 |
| `sketch_size_kb` | 只能取六个支持的容量值。 | 抛出异常。 |
| `precision` | 只能取 13 至 18。 | 抛出异常。 |
| `other` | 必须与当前对象使用相同 Key 类型和 `p`。 | 抛出异常。 |
| `stream` | 必须是有效 ACL 流或 `nullptr`。 | 抛出异常。 |

`Merge` 不接受不同 Sketch 配置。拒绝不兼容合并可以防止不同寄存器索引
空间被错误解释，也与参考实现的配置约束一致。

## 4. 算法设计

HyperLogLog 的核心状态只保存每个桶观察到的最大排名。输入 Key 先经过
稳定的 64 位哈希，再由哈希值的索引部分选择寄存器，由剩余部分计算排名。

### 4.1 哈希和寄存器更新

实现使用固定的设备可执行 64 位混合函数，建议采用 SplitMix64 风格的
整数混合步骤。哈希策略必须同时满足以下条件：

- 对同一个 Key 和同一个 Key 类型始终返回相同的 64 位结果。
- 不依赖 Host 地址、线程编号、流编号或执行批次。
- Host 参考路径和 Device Kernel 使用完全相同的常量与移位顺序。
- 对 `I32` 先保留 32 位补码位模式，再零扩展到 64 位；对 `I64` 直接
  保留 64 位补码位模式。

对哈希结果 `h`，取最高 `p` 位作为寄存器索引，剩余位用于计算排名：

```text
index = h >> (64 - p)
suffix = h << p
rank = leading_zero_count(suffix) + 1
```

当 `suffix` 为零时，排名取 `64 - p + 1`，避免对零值调用未定义的
前导零计数操作。寄存器更新为：

```text
register[index] = max(register[index], rank)
```

排名范围小于一个字节，因此 `uint8_t` 足以保存全部合法排名。

### 4.2 Add Kernel

`Add` 采用 Grid-stride 循环处理输入数组，使线程数不依赖单次输入规模。
每个线程执行以下步骤：

1. 计算当前线程负责的输入下标。
2. 从 Device 侧读取一个 `Key`。
3. 将 Key 转换为无符号位模式并计算稳定的 64 位哈希。
4. 根据 `p` 计算寄存器索引和排名。
5. 对目标寄存器执行原子最大值更新。

Ascend C 目标环境不假设支持任意地址的 8 位原子最大值。为保留一个
寄存器一个字节的内存布局，设计采用 32 位对齐的 Compare-and-swap
循环：一个 32 位字包含四个相邻寄存器，线程读取对应字节，计算该字节
与新排名的最大值，再通过 `atomicCAS` 进行读-改-写；CAS 失败时重新读取
并重试。不同寄存器的并发更新不会覆盖较大的排名，且不需要把寄存器区
扩大为四倍。

当 `key_num == 0` 时，`Add` 不启动输入处理循环，Sketch 保持不变。

### 4.3 Clear Kernel

`Clear` 启动一个按寄存器区间划分的 Kernel，将所有 `uint8_t` 寄存器写为
零。由于寄存器区连续且容量是 2 的幂，Kernel 可以使用向量化或 32 位
批量写入；尾部处理必须保持寄存器区完整覆盖。该操作不读取输入，不产生
与输入规模相关的临时内存。

### 4.4 Merge Kernel

`Merge` 为每个寄存器分配一个逻辑处理位置，执行：

```text
register[i] = max(register[i], other_register[i])
```

当前对象和 `other` 的 `p`、寄存器数量及 Key 类型必须一致。每个寄存器
只有一个逻辑写入者，因此 Merge Kernel 不需要对结果区使用原子操作。
`other` 始终作为只读输入，合并过程不会清空或改变 `other`。

### 4.5 Estimate Kernel

`Estimate` 分两个阶段完成。第一阶段对寄存器求和，第二阶段在 Host 侧
完成标量修正和返回值转换。设备侧为每个寄存器计算 `2^-register[i]`，
使用分块归约得到：

```text
Z = sum(2^-register[i])
E_raw = alpha_m * m * m / Z
```

其中 `alpha_m` 对 `m >= 128` 使用标准 HyperLogLog 系数：

```text
alpha_m = 0.7213 / (1 + 1.079 / m)
```

估算结果按以下规则进行修正：

- 空 Sketch 直接返回 0。
- 当 `E_raw <= 2.5 * m` 且零寄存器数量 `V > 0` 时，使用线性计数
  `m * log(m / V)`，提升小基数区间的准确性。
- 其他情况返回 `E_raw` 的非负整数结果。
- 对超出 `uint64_t` 表示范围的中间结果进行饱和处理，避免转换溢出。

设备归约使用 FP32 累加并保留固定大小的归约临时区。由于 `m` 最大为
262144，临时区大小与 Sketch 配置相关但与输入 Key 数量无关。若目标
设备的 FP32 累加误差影响验收精度，允许改为分块 FP64 或定点累加；算法
接口和寄存器语义不变。

## 5. 数据布局与资源管理

本节定义容器对象的 Host 元数据和 Device 资源，重点保证状态生命周期
清晰，并限制额外内存占用。

### 5.1 Host 元数据

每个容器在 Host 侧保存以下固定大小的元数据：

- `Key` 类型对应的编译期信息。
- `sketch_size_kb`、`precision` 和寄存器数量 `m`。
- Device 寄存器区指针。
- Estimate 归约结果的固定大小 Device 临时区和 Host 接收变量。
- 当前对象资源是否有效的生命周期标记。

容器不保存输入 Key 的 Host 副本，也不保存与 `key_num` 成正比的 Device
缓存。调用方负责保证 `keys` 在对应设备操作完成前保持有效。

### 5.2 Device 布局

Device 资源按以下逻辑布局组织：

```text
HyperLogLog<Key>
├── registers: uint8_t[m]
├── reduction_workspace: fixed-size block partial sums
└── estimate_scalar: one device-side uint64 result
```

`registers` 是唯一与 Sketch 容量线性相关的持久化状态，大小为
`SketchSizeKB * 1024` 字节。归约临时区按设备实现的最大活动块数分配，
不按输入数量分配。输入数组由调用方提供，容器不复制输入。

### 5.3 生命周期和移动语义

构造函数完成资源申请和空 Sketch 初始化。拷贝构造和拷贝赋值被删除，
防止两个对象错误地释放同一份 Device 资源。移动构造和移动赋值转移
全部资源所有权，并将源对象置为空句柄状态。

析构前必须保证已经提交到相关流的 Kernel 不再访问寄存器区。实现可以
通过记录最近使用的流并在释放前同步，或者使用设备事件建立释放依赖。
析构不得重复释放移动后的源对象资源。

### 5.4 流语义

每个接口都把 Kernel、Device-to-Host 拷贝和必要的内存操作提交到传入
流。容器不隐式切换设备，也不创建输入规模相关的辅助流。

`Estimate` 是同步返回接口：它在传入流上完成归约、结果拷贝和同步后才
返回估算值。其他修改状态的接口在提交操作后返回，调用方若要在其他流
中读取同一对象，必须先建立 ACL 流间同步。

## 6. 代码组织

实现遵循 `ops-collections` 的纯头文件容器模式，公开接口、设备引用和
实现细节分层组织。

```text
ops-collections/
├── include/
│   ├── hyperloglog.h
│   ├── hyperloglog_ref.h
│   └── detail/
│       └── hyperloglog/
│           ├── hash.h
│           ├── storage.h
│           ├── add.h
│           ├── clear.h
│           ├── merge.h
│           └── estimate.h
├── tests/
│   ├── hyperloglog/
│   └── performance/
│       └── hyperloglog/
└── docs/
    └── hyperloglog_API 文档和使用示例.md
```

文件职责如下：

- `hyperloglog.h`：公开类、参数校验、资源生命周期和 ACL 流管理。
- `hyperloglog_ref.h`：设备侧可引用的类型、常量和 Kernel 入口声明。
- `detail/hyperloglog/`：哈希、寄存器访问、各操作 Kernel 及归约细节。
- `tests/hyperloglog/`：功能和异常行为测试。
- `tests/performance/hyperloglog/`：创建、析构、清空、添加、合并和估算
  的性能测试。

## 7. 正确性设计

正确性验证同时覆盖算法误差、状态转换、参数校验和资源生命周期。所有
测试都使用 `I32` 与 `I64` 两个模板实例。

### 7.1 精度判定

对精确基数为 `N` 的输入，验收允许的相对误差为：

```text
abs(Estimate - N) / N <= 3 * 1.04 / sqrt(m)
```

当 `N == 0` 时，要求结果严格为 0。测试辅助代码中的寄存器数量按
`m = SketchSizeKB * 1024` 计算，与本设计的数据布局一致。

### 7.2 状态语义

以下状态属性必须成立：

- 新创建的容器估算值为 0。
- 空输入不会改变容器状态。
- 重复输入只影响对应寄存器最大值，不按重复次数增加估算值。
- 一次 `Add` 与拆分成多个批次的 `Add` 得到相同估算结果。
- `Estimate` 可重复调用，调用本身不改变寄存器状态。
- `Clear` 对空容器幂等，对非空容器恢复为空状态。
- `Clear` 后重新 `Add` 只反映清空后的输入。
- 合并相同输入具有幂等性；空 Sketch 与非空 Sketch 合并结果等于非空
  Sketch。
- `Merge` 只更新左侧对象，右侧对象保持不变。
- 移动构造后目标对象保留原容器状态，源对象不再拥有资源。

### 7.3 异常场景

功能测试必须覆盖以下失败场景：

- 不支持的 Sketch 容量，包括 0、7 和 512 KiB。
- 不支持的精度，包括 0 和超出 13 至 18 的值。
- 非零 `key_num` 配合空指针。
- 未初始化或已移动的容器句柄。
- 不同 Sketch 配置之间执行 `Merge`。
- 无效 ACL 流或不满足设备访问要求的地址。

## 8. 性能与内存设计

性能目标以任务书提供的 cuCollections 标杆时延为基准。算子每个性能用
例的时延必须满足：

```text
算子时延 <= 标杆时延 / 0.4
```

### 8.1 复杂度

| 操作 | 时间复杂度 | 额外 Device 空间 |
| --- | --- | --- |
| Create | `O(m)` 初始化 | `O(m)` 持久化寄存器区 |
| Destroy | `O(1)` 主机资源释放 | 无新增空间 |
| Clear | `O(m)` | 固定工作空间 |
| Add | `O(keyNum)` | 固定工作空间 |
| Merge | `O(m)` | 固定工作空间 |
| Estimate | `O(m)` | 固定归约工作空间 |

Add 的吞吐主要由输入读取、哈希和寄存器原子更新决定。Merge 和 Estimate
的开销主要由 Sketch 容量决定。实现不得把输入数组复制到新的线性 Device
缓存中。

### 8.2 性能测试对象

性能测试覆盖两种 Key 类型和全部六个 Sketch 容量，定义输入规模、计时
边界和判定公式。

| 接口 | Key 类型 | 测试参数 |
| --- | --- | --- |
| `Create`、`Destroy`、`Clear` | `I32`、`I64` | `SketchSizeKB` 取 8、16、32、64、128、256。 |
| `Estimate`、`Merge` | `I32`、`I64` | `NumInputs` 取 100000000，`SketchSizeKB` 取全部支持值。 |
| `Add` | `I32`、`I64` | `Distribution=UNIFORM`、`NumInputs` 取 100000000、`Multiplicity=1`，`SketchSizeKB` 取全部支持值。 |

### 8.3 性能计时方法与判定标准

性能测试先完成容器创建、输入生成、Host-to-Device 拷贝和前置同步，
再只测量待评估接口。测试计时使用单个 ACL 流，并在需要返回设备结果
时完成结果同步；准备阶段和清理阶段不计入接口时延。

每个参数组合都与相同设备、相同编译配置和相同输入规模下的参考实现
标杆进行比较。设参考实现标杆时延为 `T_ref`，被测接口时延为 `T_op`，
判定标准为：

```text
T_op <= T_ref / 0.4
```

### 8.3 性能优化顺序

性能调优按以下顺序进行，避免在正确性未稳定时引入难以定位的并发问题：

1. 确认寄存器区连续对齐，并使用合适的访存粒度。
2. 调整 Add 的线程布局和 Grid-stride 步长，减少输入读取空洞。
3. 降低 packed CAS 对同一寄存器字的竞争，必要时使用分块局部聚合。
4. 对 Clear、Merge 和 Estimate 使用连续区间划分及分层归约。
5. 在不增加输入规模线性内存的前提下调整固定工作空间大小。

## 9. 测试方案

本节定义功能测试范围、执行方式和判定条件。

### 9.1 功能测试范围

功能测试文件与覆盖重点如下。

| 文件 | 覆盖重点 |
| --- | --- |
| `create_test.cpp` | 六档容量、六档标准差、六档精度和非法配置。 |
| `destroy_test.cpp` | 重复创建析构和移动构造后的所有权转移。 |
| `clear_test.cpp` | 空容器清空、非空容器清空和清空后重用。 |
| `add_test.cpp` | 空输入、重复输入、不同基数、分批追加、空指针校验和大输入。 |
| `merge_test.cpp` | 空与非空、相同输入、重叠输入、互斥输入及不兼容配置。 |
| `estimate_test.cpp` | 空及非空基数、重复估算、重复执行一致性和误差边界。 |

### 9.2 性能测试

性能测试位于 `test-cases/benchmark/hyperloglog/`，覆盖所有接口的参数
组合。测试程序必须按第 8.2 节准备输入，并按第 8.3 节划定计时范围：

- `Create`、`Destroy` 和 `Clear` 遍历所有支持的 Sketch 容量。
- `Estimate` 和 `Merge` 使用指定输入规模，遍历所有支持的 Sketch 容量。
- `Add` 使用均匀分布和指定重复度，遍历所有支持的 Sketch 容量。

性能判定使用第 8.3 节的时延关系。

## 10. 风险与应对

以下风险直接影响精度、并发正确性或验收性能，需要在实现和自测中重点
关注。

| 风险 | 影响 | 应对措施 |
| --- | --- | --- |
| 哈希实现跨 Host 和 Device 不一致 | 相同输入得到不同寄存器状态。 | 将哈希常量和移位顺序集中在设备可引用头文件中，并增加重复性测试。 |
| 8 位寄存器并发更新竞争 | 排名丢失或相邻寄存器被覆盖。 | 使用四寄存器打包的 32 位 CAS 循环，并测试高冲突输入。 |
| Estimate 归约精度不足 | 相对误差超出验收阈值。 | 使用分层归约、非零 Sketch 小范围修正，并在不同容量上验证误差。 |
| 不同流访问同一对象 | 读写顺序不确定。 | 明确流语义，调用方负责跨流同步；Estimate 返回前完成必要同步。 |
| 析构早于设备操作完成 | Device 访问已释放资源。 | 析构同步相关流或使用事件建立释放依赖。 |
| 输入指针不是 Device 地址 | Kernel 访问非法地址。 | 在 Host 侧校验指针属性，并拒绝非零数量的空指针。 |
| 大输入数量导致 Host 类型截断 | Add 漏处理输入或发生越界。 | 统一使用 `Extent<std::size_t>`，校验转换和地址范围。 |

## 11. 交付内容

完成实现和自验后，按以下结构提交代码和文档：

- `include/hyperloglog.h`：公开 API 和 Host 侧生命周期管理。
- `include/hyperloglog_ref.h`：设备侧引用接口。
- `include/detail/hyperloglog/`：哈希、寄存器、Kernel 和归约实现。
- `tests/hyperloglog/`：功能测试及公共测试辅助代码。
- `tests/performance/hyperloglog/`：性能测试代码。
- `docs/`：API 文档、使用示例和本设计文档。
- README 更新：构建方法、测试入口、运行环境和结果复现步骤。

## 12. 参考资料

本设计使用以下资料作为接口和开发环境依据：

1. cuCollections HyperLogLog 参考实现：
   <https://github.com/NVIDIA/cuCollections/blob/dev/include/cuco/hyperloglog.cuh>
2. Ascend C 算子开发文档：
   <https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/opdevg/Ascendcopdevg/atlas_ascendc_map_10_0002.html>
3. Ascend C 算子开发接口文档：
   <https://www.hiascend.com/document/detail/zh/canncommercial/850/API/ascendcopapi/atlasascendc_api_07_0003.html>
4. Ascend C 在线课程：
   <https://www.hiascend.com/developer/courses/detail/1691696509765107713>
5. 昇腾社区任务流程及注意事项：
   <https://gitcode.com/org/cann/discussions/39>
