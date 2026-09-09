# 【社区任务】segment_coo算子设计文档

# 一、需求背景

## 1.1 需求来源

本需求来源于“8 月社区任务 - segment_coo 算子开发（950）”社区任务平台任务书。目标是在 Ascend 950 系列（arch35）上，
参考 [torch_scatter.segment_coo](https://pytorch-scatter.readthedocs.io/en/latest/functions/segment_coo.html)
（版本建议 ≥ 2.1.0）及其子算子（`segment_sum_coo` / `segment_add_coo` / `segment_mean_coo` / `segment_min_coo` / `segment_max_coo`），
基于 **Ascend C（Kernel）+ Python（PyTorch 适配层）** 实现功能与接口完全对齐的 Segment COO 系列算子，并完成算子设计、开发、测试全流程，
验收通过后提交至昇腾算子开源仓 ops-gnn。

本算子为**新增生态算子**：ops-gnn 仓库与 CANN 内置算子中均不存在 segment_coo 的历史 TBE 实现，不涉及 TBE 算子实现路径迁移；
标杆算子信息以 torch_scatter 开源实现为准，涉及文件名如下：

| 标杆文件 | 内容 |
| --- | --- |
| `segment_coo.py` | Python 层接口（`segment_coo` 与五个具名入口）与 CPU 回退路径 |
| `csrc/cpu/segment_coo_cpu.cpp` | CPU 标杆实现（有序段扫描） |
| `csrc/cuda/segment_coo_cuda.cu` | CUDA 标杆实现（原子散射） |
| `test_cases/test_segment_coo.py` | 任务书官方功能用例（TC-01～TC-18） |
| `test_cases/benchmark_segment_coo.py` | 任务书官方性能基准（29 形状 × 4 dtype × 4 reduce） |

结合 CANN 接口判断：torch_scatter 的 PyTorch 层接口与 CANN 生态接口在参数语义上一致（src/index/out/dim_size/reduce），
差异在于 CANN 侧需通过 pybind + Host tiling + Ascend C Kernel 的分层实现承接，接口语义映射关系见 2.3.1。

验收口径：以 **PyTorch 层接口**的功能、精度、性能为唯一验收基准；C++ 层（aclnn）接口为可选交付项，不作为验收必要条件。

其他参考来源：

- 精度标准：《生态算子开源精度标准（experimental）》
- Ascend C 算子开发文档与 ops-gnn 仓库既有算子目录规范（gather_coo / gather_csr / segment_max_csr 等）

## 1.2 背景介绍

### 1.2.1 segment_coo算子现状分析

#### 1.2.1.1 标杆算子支持的数据类型和数据格式

标杆 torch_scatter.segment_coo 的 PyTorch 层支持的参数、数据类型与约束如下：

| 参数 | 参数含义 | 数据类型 | 支持数据类型 | 约束 | 形状 |
| --- | --- | --- | --- | --- | --- |
| src | 输入tensor | tensor | float16, float32, int32, int64, bfloat16, int8, uint8 | 维度 1～8 | (…, n, C) |
| index | 分组索引tensor | tensor | int64 | 沿 `index.dim()-1` 非降序；取值 ∈ [0, 分组数-1] | (…, n)，m ≤ src.dim() |
| out | 输出tensor | tensor | 同 src | 可选；与 src 同形，仅归约维大小为分组数 y | (…, y, C) |
| dim_size | 分组数 | int | int | 可选；未提供 out 时决定 out 归约维大小 | - |
| reduce | 归约模式 | str | sum / add / mean / min / max | 不含 mul，`reduce="mul"` 抛 ValueError | - |
| arg_out | 极值行下标（min/max 具名入口） | tensor | int64 | 与 out 同形；平局 last-writer-wins、空桶为 0 | (…, y, C) |

数据格式说明：segment_coo 作用于稠密张量，不涉及 NCHW/NHWC 等图像格式；`index` 前 `index.dim()-1` 维与 `src` 广播对齐，
归约维固定为 `dim = index.dim()-1`。标杆支持非连续内存布局的输入（内部自动做 contiguous 化）。

#### 1.2.1.2 标杆算子全面描述

Segment COO 是基于 COO 格式排序索引的分组归约操作：沿 `index` 的**最后一维**，将 `src` 中属于同一分组索引的元素
按 reduce 模式归约写入 `out`。设 `src` 为 n 维、`index` 为 m 维（m ≤ n），归约维为 `dim = index.dim()-1`，
则 `out` 形状与 `src` 相同，仅 `out.size(dim)` 为分组数 y。

一维 `reduce="sum"` 时：

```text
out[i] = sum(src[j] for j where index[j] == i)
```

归约语义与空桶规则（自动创建 out 时）：

| reduce | 语义 | 空桶输出 |
| --- | --- | --- |
| sum / add | 段内求和（add 为 sum 别名） | 0 |
| mean | 段内均值（空桶不参与计数） | 0 |
| min / max | 段内极值（空桶不参与） | 0 |

min/max 自动创建 out 时先取与 dtype 对应的“反向极值”初值，归约完成后将空桶置 0；提供 out 时空桶一律保留用户原值
（sum/add 覆盖桶写段内归约结果，mean/min/max 覆盖桶折叠用户 out，均为 torch_scatter 官方语义）。

提供 out 时 `dim_size` 推断顺序：显式 `dim_size` > `out.size(dim)`（`dim = index.dim()-1`）> `index.max()+1`
（index 为空时为 0）。

Segment COO 是图邻接 COO 格式边聚合、有序分组统计、PyG 稀疏消息传递（index 已排序时优先选用）中的基础算子。
与 `scatter` 的关键差异：

| 特性 | segment_coo | scatter |
| --- | --- | --- |
| index 顺序 | 须沿 `index.dim()-1` 非降序排列 | 任意顺序 |
| reduce 模式 | sum/add/mean/min/max | 另含 mul |
| 典型实现 | 有序段扫描，段与核连续绑定，每段一次归约加单次写回 | 按元素散射，跨核任意写，通常需要原子操作 |
| 典型性能 | 通常快于 scatter（可利用有序性） | 通用但慢 |

有序性把“每元素一次原子”变成“每段一次定位 + 段内顺序归约”，这是滑动行窗口、段内树归并、批量写回等优化成立的前提，
也是本设计与 scatter 在选型上的根本区别。

#### 1.2.1.3 标杆算子实现流程图

标杆以 torch_scatter CPU/CUDA 参考实现为准（本算子无 TBE 历史版本）。标杆实现流程如下：

```text
┌─────────────────────────────────────────────────────────────────┐
│                       torch_scatter segment_coo                  │
├─────────────────────────────────────────────────────────────────┤
│ 1. 参数检查：reduce 合法性（mul → ValueError）、维度关系检查       │
│ 2. dim_size/out 归约维推断：dim_size > out.size(dim) > index+1   │
│ 3. 输入规范化：src/index contiguous 化，广播对齐 [B, n] / [B,n,C] │
│ 4. 分组归约：                                                    │
│    CPU：有序段扫描——沿 index 非降序定位段边界，段内顺序归约         │
│    CUDA：按元素散射——atomicAdd/atomicMin/atomicMax 写全局 out，    │
│          min/max 以原子 CAS/交换跟踪 arg（结果非确定）              │
│ 5. mean：段内求和 + 桶计数，count 归一化                           │
│ 6. 空桶处理：自动创建 out 时反向极值初值归约后置 0；               │
│    提供 out 时保留用户值                                          │
│ 7. arg_out：极值行下标写回（int64）                               │
└─────────────────────────────────────────────────────────────────┘
```

要点：CUDA 标杆的原子散射路径结果非确定（浮点顺序、平局归属均非确定），其 CPU 路径为顺序归约；
本设计以任务书契约为准（见 3.2.2.3 差异点说明）。

# 二、需求分析

## 2.1 外部依赖分析

| 依赖 | 用途 | 是否运行时依赖 |
| --- | --- | --- |
| PyTorch + torch_npu | PyTorch 层接口为验收基准；device 张量分配与同步 | 是 |
| CANN Toolkit（Ascend C） | Kernel 编译与运行（arch35） | 是 |
| torch_scatter | 标杆参考：仅功能对拍与 CPU golden，不参与线上运行 | 否（仅测试期） |
| pytest | 功能用例执行框架 | 否（仅测试期） |

无其他第三方运行时依赖。

## 2.2 内部耦合模块

ops-gnn 仓库内部需要联动适配的模块：

| 模块 | 耦合内容 | 影响 |
| --- | --- | --- |
| `csrc/pybind.cpp` | 注册 `segment_coo` / `segment_coo_arg` 两个入口 | 新增注册项，不改既有接口 |
| `python/ops_gnn/__init__.py` | 导出 COO/CSR 系列函数（官方 TC-12 需要 `segment_{reduce}_csr` 交叉验证） | 新增导出项 |
| `segment_max_csr.py` | 增加 int64 indptr 便捷转换 | int32 原路径行为不变 |
| `test/segment_coo/` | golden 参考实现、功能用例、benchmark、核算脚本 | 新增目录 |
| 仓库既有算子（gather_coo / gather_csr / scatter 等） | 仅目录规范一致，无代码耦合 | 无 |

## 2.3 需求模块设计

### 2.3.1 Ascend C算子原型

PyTorch 层接口（须与标杆逐字对齐）：

```python
def segment_sum_coo(src, index, out=None, dim_size=None) -> Tensor: ...
def segment_add_coo(src, index, out=None, dim_size=None) -> Tensor: ...
def segment_mean_coo(src, index, out=None, dim_size=None) -> Tensor: ...
def segment_min_coo(src, index, out=None, dim_size=None) -> Tuple[Tensor, Tensor]: ...
def segment_max_coo(src, index, out=None, dim_size=None) -> Tuple[Tensor, Tensor]: ...
def segment_coo(src, index, out=None, dim_size=None, reduce="sum") -> Tensor: ...
```

内部 C++/Kernel 原型（pybind 两入口）：

```text
segment_coo(src, index, out, dim_size, reduce) -> out
segment_coo_arg(src, index, out, arg_out, dim_size, reduce) -> (out, arg_out)
    # min/max 带 arg 跟踪时走 arg 入口，arg_out 固定 INT64
```

张量参数与属性约束同 1.2.1.1。归约维固定 `dim = index.dim()-1`，输出形状与 src 相同、归约维大小为分组数 y。

### 2.3.2 Ascend C算子相关约束

1. `index` 必须沿 `index.dim()-1` 非降序，取值 ∈ [0, y-1]；无序输入为未定义行为（与标杆一致，不做 device-side 校验）。
2. 无 `mul` 模式：`reduce="mul"` 抛 `ValueError`。
3. `index` / `arg_out` 固定 INT64；`src` 支持 1～8 维、`index.dim() <= src.dim()`。
4. 数据类型分级（实现路径详见 3.2.1）：FLOAT16/FLOAT32 走 NPU 原生 Kernel（功能 + 性能验收）；
   INT32/INT64 走 NPU 原生精确整数路径（功能 + 性能验收，bit-wise）；BFLOAT16/INT8/UINT8 走 host 预转换路径（仅功能验收）。
5. 不支持 BOOL、COMPLEX、Float8 等 dtype。
6. aclnn 层为可选交付项；PyTorch 层为验收基准。

# 三、需求详细设计

## 3.1 调用方式

本算子采用 **PyTorch 框架调用**方式，调用链自上而下：

```text
用户 Python 调用 segment_coo / segment_{sum,add,mean,min,max}_coo
  ▼
Python 适配层（python/ops_gnn/segment_coo.py）
  │  shape 推断 / dtype 分发 / 精度策略 / i64 输出端校验 / CPU 兜底
  ▼
pybind（csrc/pybind.cpp：segment_coo / segment_coo_arg）
  ▼
Host 层（csrc/npu/segment_coo/op_host/segment_coo.cpp）
  │  Canonicalise、PlanDtype、tiling/分核、counts 缓冲、native int64 门控
  ▼
Ascend C Kernel（csrc/npu/segment_coo/op_kernel/arch35/）
    segment_coo_kernel.cpp（dtype dispatch）
    segment_coo_tiling.h / segment_coo_kernel_impl.h（有序段扫描主循环）
```

aclnn kernel 直调为可选交付项；验收以 PyTorch 层为唯一基准。

## 3.2 需求总体设计

### 3.2.1 host侧设计

tiling 总策略：Segment COO 的计算以“有序段”（连续相同 index 的行区间）为基本单元。host 侧将输入规范化为逻辑视图
`src [B, n, C]`、`index [B, n]`、`out [B, y, C]` 后，以段为粒度做分核与切分，段内行数不改变计算逻辑。
host 侧主要步骤：

1. **Canonicalise**：将 `src/index` 规范化为 `[B, n, C] / [B, n]`（归约维 move 到 0、特征维展平、按 32B 对齐补 padding）。
   多 batch 精确形式保留 `[B, n, C] / [B, n]`；其余形式 `movedim(dim, 0)` 折叠为 `[1, n, C] / [1, n]`。
2. **PlanDtype**：dtype → (STORAGE, COMPUTE, CAST_MODE) 分发键，并决定是否启用 counts 缓冲。
3. **分核/Tiling**：核数与段区间分配见 3.2.1.1，UB 分块见 3.2.1.2。
4. **counts 缓冲**：整型 mean/原子 mean 所需的桶计数由 Kernel 以 `DataCopyPad` 直写连续 GM 缓冲（规避非 32B 对齐丢数据）。
5. **native int64 门控**：i64 min/max 值口径的门控以 `tiling.computeBytes` 为单一真值源，host 与 kernel、wrapper 三端公式
   逐字对齐并交叉注释（已经 dav_3510 CANN 9.1.0 头文件白名单静态核实：Min/Max 白名单含 int64_t，Compare 仅 half/float）。
6. **校验**：Host 不做 device-side assert；提供 out 且未给 dim_size 时由 Python 层做 `index.max() < out.size(dim)` 越界防御。

#### 3.2.1.1 分核策略

优先使用满核的原则。段与核连续绑定：

```text
coreNum      = min(可用 AIV 核数, totalSegs)          # totalSegs = B × y
segsPerCore  = totalSegs / coreNum
extraCoreCount = totalSegs % coreNum                   # 不能均分时，余出的段分给前 extraCoreCount 核
segBegin     = blockIdx < extraCoreCount
                 ? blockIdx × (segsPerCore + 1)
                 : extraCoreCount × (segsPerCore + 1) + (blockIdx − extraCoreCount) × segsPerCore
segEnd       = segBegin + (blockIdx < extraCoreCount ? segsPerCore + 1 : segsPerCore)
```

- 段间能均分时无大小核区分；不能均分时前 `extraCoreCount` 个核每核多承担 1 段。
- 输入数据大小经 Canonicalise 后的形状与 `GetDataTypeLength` 计算；UB 内存大小与核心数量通过平台信息 API 获取
  （不硬编码），并据此调整核数。
- 单核内部按段循环顺序处理，段定位游标跨段单调推进（继承上一段终点，摊销 O(1)）。

#### 3.2.1.2 数据分块和内存优化策略

充分使用 UB 空间的原则：

- 考虑不同硬件 UB 大小差异、是否开启 double buffer（BUFFER_NUM=2）、以及 kernel 侧所需的临时缓冲
  （原始 STORAGE + 转换后 COMPUTE 双缓冲、16 组累加器、arg/idx 窗口缓存、输出 staging 乒乓），按 UB 预算推导单核内
  滑动行窗口行数：

```text
rowsPerChunk = (UB可用预算 − 固定缓冲占用) / (C_pad × (sizeof(STORAGE) + sizeof(COMPUTE)))
```

- **滑动行窗口**：相邻段的行在 GM 中物理连续，一次 MTE2 载入 `blockRows` 行窗口服务所有覆盖段，把每段一次小 DMA 的
  固定延迟摊销为每窗口一次；MTE2 单次搬运量不超过硬件上限（arch35 ≤128KB）。
- **批量输出 staging**：结果行在 UB 攒批（≤16 行，乒乓双缓冲），每批一次连续大 MTE3 写回；提供 out 时退回逐段写回以
  保留空桶语义。
- **非对齐行**：行宽非 32B 对齐（极小 C）时走逐行 `DataCopyPad` 修正路径。
- 设置切分参数：每核段区间、滑窗行数、窗口行数、counts 缓冲地址等写入 TilingData 下发 kernel。

#### 3.2.1.3 tiling key 规划策略

需要 tilingkey 的情况：kernel 需要感知 host 侧信息走不同分支。规划按以下维度编码：

1. dtype 分发键（STORAGE / COMPUTE / CAST_MODE）：决定 kernel 内 dtype dispatch 与是否需要预转 f32
   （bf16/int8/uint8 及 f16 min/max 走 host 预转 f32 路径）；
2. reduce 模式（sum/add/mean/min/max）：决定归约指令选择与是否需要桶计数；
3. batch 维数（单 batch / 多 batch）：决定行号是否需要 batch 内局部换算；
4. 是否启用原子流式路径（sum/mean 的短段窄行场景，触发条件：`avgSegRows ≤ 4` 且 dtype/对齐满足）；
5. native int64 门控（由 `tiling.computeBytes` 推导）：决定 i64 min/max 值口径走原生 int64 域还是 float 域。

数据检测：host 侧不做 device-side assert；越界防御由 Python 层承担（见 3.2.1 第 6 条）。

### 3.2.2 kernel侧设计

#### 3.2.2.1 kernel伪代码描述

kernel 分 Init 和 Process 两个阶段，Process 含 CopyIn / Compute / CopyOut：

```text
Init(blnkIdx):
    设置 GM 地址（src/index/out/argOut/counts）
    计算本核段区间 [segBegin, segEnd)          # 见 3.2.1.1 公式
    初始化 UB 缓冲：滑窗(STORAGE+COMPUTE 双缓冲)、累加器×16、arg、idx 窗口、staging 乒乓
    counts 缓冲清零（整型 mean / 原子 mean 时）

Process():
    cursor = segBegin 起始行
    for seg in [segBegin, segEnd):              # 段间游标单调推进
        lo = cursor                             # 继承上一段 hi，摊销 O(1)
        hi = lower_bound(index, seg+1)          # gallop 跳步探测 + 窗口内二分回退
        CopyIn:  MTE2 载入 [lo, hi) 覆盖的行窗口（STORAGE）
                 需要时 Cast → COMPUTE（bf16/int8/uint8/f16-minmax 预转 f32）
        Compute: 段内归约，按形状特征选路径：
                 - 树归并快路径: rows ≤ 64 且行 32B 对齐且无 arg
                     log2(rows) 次不相交原位宽 Add/Min/Max
                 - 分组累加路径（通用）: 16 累加器 lane（4 组轮转）摊平 RAW 链
                 - 原子流式路径（sum/mean 短段）: 行均分跨核，DMA read-add-write 原子向量加
                 - 逐行修正路径（行宽非 32B 对齐）: 逐行 DataCopyPad + 累加
                 min/max 带 arg 时: Compare(GE/LE) 生成掩码 → Select 更新行号（int32，last-writer-wins）
        CopyOut: 结果行写入 UB staging（≤16 行乒乓）→ 每批一次连续 MTE3 写回
                 提供 out 时逐段写回以保留空桶用户值
        cursor = hi
    写回 arg_out（int64）/ counts（int32，整型 mean 时）
```

#### 3.2.2.2 Ascend C主流程图

```text
                    ┌──────────────┐
                    │ Init         │
                    │ GM/UB/段区间  │
                    └──────┬───────┘
                           ▼
              ┌──── totalSegs>0? ─────┐
             否│                      │是
              ▼                      ▼
        直接退出           ┌────────────────────┐
                          │ 段循环 seg∈[b,e)    │◄──────────────┐
                          └─────────┬──────────┘               │
                                    ▼                          │
                    ┌───────────────────────────┐              │
                    │ CopyIn: MTE2 滑动行窗口     │              │
                    │ STORAGE 双缓冲 + Cast 预转  │              │
                    └─────────────┬─────────────┘              │
                                  ▼                            │
                    ┌───────────────────────────┐              │
                    │ Compute: 有序段归约         │              │
                    │  段定位: 游标+gallop+二分    │              │
                    │  路径选择:                  │              │
                    │   树归并/分组累加/           │              │
                    │   原子流式(sum,mean)/        │              │
                    │   逐行修正                  │              │
                    │  arg 跟踪: Compare+Select   │              │
                    └─────────────┬─────────────┘              │
                                  ▼                            │
                    ┌───────────────────────────┐              │
                    │ CopyOut: UB staging 攒批    │              │
                    │ ≤16 行乒乓 → MTE3 批量写回   │──────────────┘
                    └───────────────────────────┘  下一段
                           │  段循环结束
                           ▼
                    ┌───────────────────────────┐
                    │ 收尾: arg_out/counts 写回   │
                    └───────────────────────────┘
```

#### 3.2.2.3 Ascend C实现流程图与标杆算子流程图存在的差异点说明附图

与标杆（torch_scatter CUDA 原子散射 / CPU 顺序段扫描）的实现差异点及原因：

| # | 差异点 | 标杆实现 | 本设计实现 | 原因 |
| --- | --- | --- | --- | --- |
| 1 | 归约范式 | CUDA 按元素原子散射，跨核任意写、结果非确定 | 有序段扫描：段与核连续绑定，段内向量归约 + 单次写回 | 利用 index 有序性，消除跨核原子与浮点顺序非确定性（短段 sum 原子流式路径为任务书豁免的唯一非确定路径，int32 原子加保持精确） |
| 2 | 分核粒度 | CUDA 按 block 分元素块 | 按段区间分核（3.2.1.1 公式），段为原子单位 | 段内行物理连续，MTE2 滑动窗口可跨段复用，摊销 DMA 固定延迟 |
| 3 | mean 实现路径 | 段内求和 + 计数除法 | 整型 mean 内核只写精确 SUM + counts 经 arg 槽位吐出，除法放 Python 层 `rounding_mode="trunc"` | arch35 float→int CAST_TRUNC 实为 RINT、float 仿真整数除法不精确，除法不能下核 |
| 4 | i64 min/max | GPU 原子 + 非确定 arg | 值口径：原生 int64 域（C_pad≥128 且 C_pad·n≤64M）；其余走 float 域 + 输出端校验（\|out\|<2²⁴ ⇒ bit-wise），越界回落 Python 两段稳定排序 CPU 兜底 | arch35 无 int64 Compare（已静态核实），arg 掩码无法原生生成；输出端校验利用 int64↔float64 转换单调性证明 bit-wise |
| 5 | dtype 预转换 | 无（模板直接实例化） | bf16/int8/uint8（及 f16 min/max 默认）host 预转 f32，Kernel 归约后转回 | 降低 kernel dtype 分支数；f16 原生 half 路径指令前提已静态核实（half Min/Max + half Compare + int32 Select 白名单齐备），作为性能优化项经专项验证后启用 |
| 6 | 空桶/用户 out 语义 | 分散在各分支 | 统一约定：自动创建 out 空桶置 0；提供 out 空桶保留用户值，staging 批量写回退回逐段写回 | 保证与 torch_scatter 官方语义逐条对齐，且不与批量写回优化冲突 |

差异总览附图：

```text
   标杆 CUDA                     本设计 Ascend C
┌──────────────┐            ┌──────────────────────────┐
│ 元素块分核    │            │ 段区间分核(3.2.1.1)        │
│ 每元素原子写  │   ──────►  │ 每段一次归约+单次写回       │
│ 非确定       │   范式替换  │ 确定（除豁免路径）          │
└──────────────┘            └──────────────────────────┘
┌──────────────┐            ┌──────────────────────────┐
│ 单一 dtype 模板│  ──────►  │ dtype 分级 + 预转/门控      │
└──────────────┘   分层增强  │ (i64 三层策略, 5.1)        │
                            └──────────────────────────┘
```

## 3.3 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Ascend 950 系列（arch35） | √ |

## 3.4 算子限制

1. `index` 必须沿 `index.dim()-1` 非降序，取值 ∈ [0, y-1]；违反为未定义行为（与标杆一致）。
2. 无 `mul` 模式：`reduce="mul"` 抛 `ValueError`。
3. `index` 为 INT64；支持 `index.dim() <= src.dim()`、`src` 维度 1～8。
4. 不支持 BOOL、COMPLEX、Float8 等 dtype。
5. aclnn 层为可选交付项；PyTorch 层为验收基准。
6. 已知实现级限制：单段行数达数百万量级（pathological 长段）时分核以段为原子单位、并行度受限，
   不在验收形状表内，后续可做段内并行优化。

# 四、特性交叉分析

| 交叉项 | 分析 |
| --- | --- |
| segment_csr 系列 | 共享 `python/ops_gnn/__init__.py` 导出；官方 TC-12 要求 COO 与 CSR 结果一致，作为交叉验证用例。CSR 侧 `segment_max_csr.py` 增加 int64 indptr 便捷转换，int32 原路径行为不变 |
| pybind 注册 | `csrc/pybind.cpp` 新增 `segment_coo` / `segment_coo_arg` 两个注册项，为追加式改动，不影响既有算子入口 |
| gather_coo / gather_csr / scatter | 仅遵循同一目录与代码规范，无代码级耦合；segment_coo 与 scatter 的选型差异见 1.2.1.2 |
| 确定性 | 主路径无跨核原子、float 结果确定；仅短段 sum 原子流式路径存在任务书豁免的浮点顺序非确定性，对其他算子无影响 |
| 多 batch / 非连续输入 | 输入规范化在 Python/Host 层完成，不改变其他算子的内存布局约定 |

# 五、可维可测分析

## 5.1 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 功能标准 | PyTorch 层执行，覆盖任务书官方用例 TC-01～TC-18（标准 shape/dtype/reduce、提供 out、非连续、空张量、全类型、与 segment_csr 一致、泛化形状、sum≡add、mul 负向、单分组、空段、非连续 index），预期全部通过 | 任务书 |
| 精度标准（float16/float32/bfloat16/int8/uint8） | 与 CPU torch_scatter 标杆按《生态算子开源精度标准》容差比对 | 《生态算子开源精度标准（experimental）》 |
| 精度标准（int32/int64） | 与 CPU 标杆 bit-wise 一致 | 任务书 |
| 精度标准（min/max arg） | `out` 与 `arg_out` 与标杆一致；平局 last-writer、空桶 arg=0 | 任务书契约 |
| 性能标准 | 464 格（29 形状 × 4 dtype × 4 reduce）全部 ≥ 0.45× 标杆（耗时 ≤ 2.22×），基准复用任务书 `benchmark_segment_coo.py`，逐格核算用 `check_bar.py` | 任务书 |

i64 min/max 全值域 bit-wise 的三层保证：

1. 值口径在 `C_pad ≥ 128` 且 `C_pad·n ≤ 64M` 形状走原生 int64 域（arch35 无 int64 Compare，但有原生 int64 Min/Max；
   已经 dav_3510 CANN 9.1.0 头文件白名单静态核实）。
2. 其余值口径与全部 arg 口径走 float 域 + 输出端校验（|out| < 2²⁴ ⇒ bit-wise 证明）；越界自动回落 Python 两段稳定排序
   CPU 兜底。
3. native/float/fallback 三层由极端场景专项用例覆盖（inf/NaN、整数回绕、2⁵³ 边界、平局、空桶、用户 out 折叠、多 batch、
   2³² false-safe 窗口、2²⁴ 定理边界、±2⁶³ 全值域等），native 门控 host/kernel/wrapper 三端逐字对齐；
   校验/兜底结果按 tensor 身份缓存，命中后零开销。

可测性配套：功能用例 `pytest test/segment_coo/test_segment_coo.py`；极端等价性专项 `extreme_equiv_test.py`；
golden 参考实现与标杆 CPU 实现做大规模 fuzz 对拍；复现步骤与命令见 `test/segment_coo/README.md`。

## 5.2 兼容性分析

- ops-gnn 中新增 `segment_coo` 模块，属新算子，不涉及对历史 TBE 算子或 aclnn 接口的兼容；PyTorch 层接口与 torch_scatter 对齐。
- 共享文件改动均为追加式：`csrc/pybind.cpp` 注册新入口；`python/ops_gnn/__init__.py` 新增导出；
  `segment_max_csr.py` 增加 int64 indptr 便捷转换且 int32 原路径行为不变。对既有 API 无破坏。
- 硬件适配范围限定 Ascend 950（arch35），不宣称支持其他芯片架构。
