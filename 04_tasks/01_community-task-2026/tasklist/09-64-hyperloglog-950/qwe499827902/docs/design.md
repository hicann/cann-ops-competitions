# HyperLogLog 容器（950）设计

作者：qwe499827902
任务：[9月社区任务-hyperloglog容器开发(950)](https://www.hiascend.com/activities/task-center/details/f975dbdd02cb43eaacd0d7e3d3fa8821?menu=guide)
讨论：[ops-collections #4](https://gitcode.com/cann/ops-collections/discussions/4)

# 需求背景（required）

## 需求来源

参考 NVIDIA cuCollections 的 [hyperloglog](https://github.com/NVIDIA/cuCollections/blob/dev/include/cuco/hyperloglog.cuh)，在昇腾 950PR 上使用 Ascend C（SIMT）实现功能一致的 HyperLogLog 基数估算容器，按 ops-collections 纯头文件容器工程模式提交。

## 背景介绍

HyperLogLog（HLL）以 m 个寄存器为代价，用 O(1) 内存估算海量去重基数，广泛用于去重统计场景。cuCollections 将 HLL 移植到 GPU：寄存器更新依赖 CUDA 原生 `atomicMax`。昇腾 SIMT 体系没有等价的"按寄存器原生 max"硬件原子，且本任务规定每寄存器 8-bit 存储（cuCO 为 int32），因此寄存器更新与归约路径需要针对 950 重新设计，这正是本设计的核心问题。

# 需求分析（required）

## 需求描述

使用 Ascend C 实现 HyperLogLog 容器，提供 Create（三种构造）/ Destroy（RAII 析构）/ Clear / Add / Merge / Estimate 接口，支持 I32、I64 键类型；Estimate 相对误差不大于 3×1.04/√m；相同输入下状态变更与 Estimate 结果确定性可重复。

## 需求拆解

1. m = SketchSizeKB × 1024 个 8-bit 寄存器，precision = log2(m) ∈ [13, 18]；三种构造等价并校验非法配置（抛异常）。
2. Add：哈希（仓库 `aclco::xxhash_64<Key>`，seed 0）→ 寄存器索引 = 哈希高 precision 位 → rank = clz(h << precision) + 1（饱和 65-p）→ 寄存器取 max。
3. Merge：两 sketch 逐寄存器取 max；配置不兼容报错。
4. Estimate：HLL++ 口径（原始估计 → 线性计数 LC（e ≤ 2.5m 且有空寄存器）→ threshold 比较与经验偏差校正插值），空 sketch 恰返回 0。
5. 确定性：max 运算满足交换/结合/幂等，配合固定顺序归约，实现"同输入位级相等、分批 = 一次性"。
6. 性能：全部用例 ≥ 0.4 倍标杆；Add 档受主机读带宽与原子路径物理约束（见"性能设计与实测"）。

# 详细设计（required）

## 算子分析

### 数学公式

- 寄存器索引：`index = h >> (64 - p)`，h 为 64 位哈希，p = precision，m = 2^p。
- rank：`rank = clz(h << p) + 1`，若 `h << p == 0` 则 rank = 65 - p（与 cuCO `countl_zero((h<<p)|padding)+1` 逐分支等价）；rank ≤ 52，8-bit 可存。
- 寄存器更新：`M[index] = max(M[index], rank)`。
- 估计：`Z = Σ 2^(-M[j])`，`e = α(m)·m²/Z`，α(m) = 0.7213/(1+1.079/m)（m ≥ 128 恒用）；
  若存在空寄存器且 e ≤ 2.5m：`estimate = m·ln(m/zeros)`（线性计数）；否则取 e（>5m 不做偏差校正）。
- 验收判据：相对误差 ≤ 3×1.04/√m（m=8192 时约 3.45%）。

### 支持数据类型与形状

| 项 | 取值 |
| --- | --- |
| Key 类型 | int32_t、int64_t |
| 寄存器 | 8-bit × m，m = SketchSizeKB × 1024（8192~262144） |
| SketchSizeKB | 8 / 16 / 32 / 64 / 128 / 256 |
| standard_deviation | 0.0115 / 0.0082 / 0.0058 / 0.0041 / 0.0029 / 0.0021 |

## 算子实现

### 实现方案

#### host侧设计

- `HyperLogLog<Key>` 模板类（aclco 命名空间），三个静态工厂（`CreateWithSketchSizeKB / CreateWithStandardDeviation / CreateWithPrecision`）完成配置校验（非法配置抛 `std::invalid_argument`）、sketch 分配（m 字节，四寄存器/字打包）与 8KB Estimate 工作区分配，并同步清零。
- `σ → precision`：`precision = round(log2((1.04/σ)²))`，带舍入前越界防护（避免极端 σ 触发 lround 未指定行为）。
- `Finalize`（宿主 double 定稿）：与 cuCO `finalizer` 逐分支一致——2.5m 线性计数门槛、`threshold(precision)` 比较、k=6 滑窗经验偏差插值（HLL++ 调参表改自 cuCollections tuning.cuh，裁剪至 p13-18，逐数值一致）。
- 接口为同步语义（内部 `aclrtSynchronizeStream` 后返回），与官方测试用法一致；Estimate 通过容器私有 8KB scratch 传递结果，不改 sketch 状态。
- 未初始化（moved-from/析构后）对象调用 Clear/Add/Merge/Estimate 抛 `std::logic_error`。

#### kernel侧设计

- **AddSimt**（网格跨步 + 每线程 8 深度加载）：每键 `xxhash_64` → `(index, rank)` → `FoldHashByteMax`：定位 index 所在 32 位字（`word = idx >> 2`，lane = `idx & 3`），读-比较-`asc_atomic_cas` 循环完成字节级 max。CAS 重试保证原子性；max 的交换/结合/幂等性保证寄存器终态与更新顺序无关。
- **MergeSimt**：逐字 `MergeWordBytes`（四 lane 逐字节 max）后经 `AtomicCas` 提交，合并语义 = 逐寄存器 max。
- **ClearSimt**：字粒度清零。
- **EstimateSimt**：单块 1024 线程。每线程按固定步长累加所辖字的 4 个 rank 的 2^-rank（IEEE-754 指数域位构造精确幂，float 累加，固定顺序）与零寄存器计数；`asc_syncthreads` 后线程 0 按下标串行归约 1024 个部分和，将 (Z 的位模式, zeros) 打包写入 scratch。宿主解包后以 double 完成 Finalize。
- 确定性论证：寄存器终态唯一（max 归约）⇒ Z/zeros 归约顺序固定 ⇒ Estimate 位级可重复；分批 Add 与一次性 Add 终态相同 ⇒ Estimate 位级相等。

### 内存预算

| 项 | 大小 | 说明 |
| --- | --- | --- |
| sketch | m 字节（8KB~256KB） | 寄存器本体 |
| Estimate scratch | 8KB（1024×8B） | 容器私有工作区，常驻 |
| 合计 | 最大约 264KB | 无与输入规模线性相关的额外 Device 拷贝 |

## 与参考实现（cuCollections）的差异

| 项目 | cuCollections | 本实现 | 说明 |
| --- | --- | --- | --- |
| 寄存器存储 | int32 × m（4B/寄存器，配合 CUDA 原生 atomicMax） | 8-bit × m 打包（任务书规定） | m 定义一致（本任务 m = KB×1024），精度口径一致 |
| 归约精度 | double | 单块固定顺序 float | 相对舍入 ~1e-5 量级，远小于 3×1.04/√m 容差；固定顺序使位级可重复 |
| Merge 原子性 | thread_scope 原子 update_max | 字节 CAS（单流顺序使用前提） | 官方测试为单流顺序使用；多流并发需外部分步同步 |
| 接口形态 | async/sync 变体 | 同步接口 | 与官方测试一致 |
| SketchSizeKB→m | m = KB×256 | m = KB×1024 | 按任务书 8-bit/寄存器定义 |
| σ→precision | 常数 1.106 + ceil(2·log2(1.106/σ)) | 常数 1.04 + round(log2((1.04/σ)²)) | 两式对官方 6 个 σ 测试值相差 1 档；官方测试对 σ 构造仅验证 Estimate==0，不钉死 precision；本实现取理论常数 1.04 以对齐 §3.2 误差口径 |
| 错误处理 | CUCO_EXPECTS | std::invalid_argument / std::logic_error | 官方测试 REQUIRE_THROWS 仅要求抛出 |

## 性能设计与实测

- 五项操作大余量达标：create 7.3~17.2µs、destroy 0.98~1.12µs、clear 6.68~6.78µs、merge 6.5~6.68µs、estimate 69.1~71.4µs（上限分别为 1.08ms~1.29ms / 1.13~1.19ms / 72.8~83.1µs / 77~144.8µs / 92.7~248.7µs）。
- **Add（1 亿键）实测 2984~5843µs，超出 0.4× 标杆上限（792~5347µs），按 §3.3"如不达标，须给出合理解释"如实说明，数据链如下**：
  1. 探针二分实测：纯读+哈希（不含任何 sketch 更新）= 1005µs（有效读带宽约 400GB/s），已高于 8KB 档上限 792µs——任何必须逐键读取全部输入的正确实现在该硬件上无法低于此地板。
  2. 三代实现实测：GM 字节 CAS 直连 3.4~5.8ms；UB 块私有化字节 CAS 3.2~5.8ms；int32 寄存器 + GM `asc_atomic_max` 280~945ms（该原子在 950 SIMT 为慢路径，实测后弃用）。
  3. 架构差异：标杆来自 GPU（HBM 读约 1.26TB/s + 原生 atomicMax + 约 22 万硬件线程）；950PR 为 56 AIV × 1024 线程 ≈ 5.7 万线程，且字节级更新无原生原子指令，只能走 CAS 循环。

## 支持硬件

Atlas 950 系列（dav-3510，Ascend 950PR 实测），后续支持 SIMT 能力的昇腾系列产品。

## 算子约束限制

1. 单流顺序使用（与官方测试一致）；跨流并发操作同一容器未定义。
2. `Add` 的 keys 须为 Device 侧可访问且满足 Key 对齐的指针；dtype 由模板参数编译期确定。
3. Merge 要求两 sketch 的 SketchSizeKB 与 precision 一致，否则抛异常。

# 可维可测分析

## 精度标准/性能标准

- 精度：官方功能用例（按 dtype/GENERATE/SECTION 展开共 1280 例）+ 大用例（1 亿键 × 3 sketch × 2 dtype）全部通过；确定性专项（两实例位级相等、分批=一次性、重复 Estimate 不变）通过；实测精度余量见自测报告。
- 性能：官方 72 个性能用例全量采集，除 Add 档外全部达标且有大余量；Add 档不达标，解释与数据链见"性能设计与实测"。

# 测试与验证

- 构建：干净检出（git clone --depth 1）后 `bash scripts/build.sh -b` 一次性通过；功能测试 `bash scripts/build.sh -r --test-name hyperloglog` 全绿（6/6 二进制，0 失败，官方汇总"通过 6 个，失败 0 个"）。
- 功能：Add 726 / Clear 180 / Create 56(展开) / Destroy 108 / Estimate 480 / Merge 336 断言全过；含 1 亿键大用例（I32 400MB / I64 800MB）与空输入、重复输入、清空重用、非法配置、空指针负向、配置不兼容负向等场景。
- 确定性：estimate_test 的双实例位级相等与 add_test 的分批=一次性断言覆盖。
