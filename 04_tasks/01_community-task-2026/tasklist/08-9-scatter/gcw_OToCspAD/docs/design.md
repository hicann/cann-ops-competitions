# 需求背景（required）

## 需求来源

本算子来自“8 月社区任务 - scatter 算子开发任务书”。验收以 PyTorch 层公开接口的
功能、精度和性能为唯一基准，目标硬件为 Ascend 950PR，验收通过后按流程提交到
`ops-gnn`。

## 背景介绍

Scatter 把 `src` 沿指定维度按任意、可重复的 int64 `index` 写入输出并归约。GNN
聚合中的 index 通常无序且存在热点 bucket，既需要广播/shape/out 语义与
torch_scatter 对齐，也要避免全局原子在高冲突负载下退化。

### Baseline 实现现状分析

zip baseline 提供了 Ascend C 工程骨架和基础 scatter 计算思路，但缺少任务书要求的
完整 PyTorch drop-in API、六归约全规格、广播、provided out、min/max arg_out、L2
回退、正式精度工具链、40 点性能门禁及高冲突优化。本实现保留 Ascend C + Host +
Python 三层结构，补齐上述要求并建立 fail-closed 验收链。

### 对标行为

接口对标官方 `torch-scatter 2.1.2`。CPU reference 从隔离安装目录加载，禁止引用本仓
`python/torch_scatter`。参考源码 tag commit 为
`140d3ad677aae615767412873b90982cbf97d35d`，源 zip SHA256 为
`571CEA2BE7EE2D1E36B33E3998D850E23693E31BA3EBCC115687220E2AFBABAF`。

min/max 平局场景以任务书规定的“后写者优先”为验收规则。value 使用隔离的官方 CPU
实现生成真值；arg_out 的平局位置由独立 verifier 按任务书规则复核，并在 case
manifest 中显式记录。

# 需求分析（required）

## 需求描述

实现 `scatter`、`scatter_sum/add/mul/mean/min/max` 七个公开前向 API；覆盖
sum/add/mul/mean/min/max 六种归约、L1 七种原生 dtype、L2 float64/int64、rank 1～8、
广播、负 dim、shape 推断、provided out、非连续、空张量、错误输入和高冲突 index。

### PyTorch 接口对照

公开函数名、参数顺序、默认值和返回类型与 `torch_scatter 2.1.2` 对齐：

```python
from typing import Optional, Tuple
import torch

def scatter_sum(src: torch.Tensor, index: torch.Tensor, dim: int = -1,
                out: Optional[torch.Tensor] = None,
                dim_size: Optional[int] = None) -> torch.Tensor: ...
def scatter_add(src: torch.Tensor, index: torch.Tensor, dim: int = -1,
                out: Optional[torch.Tensor] = None,
                dim_size: Optional[int] = None) -> torch.Tensor: ...
def scatter_mul(src: torch.Tensor, index: torch.Tensor, dim: int = -1,
                out: Optional[torch.Tensor] = None,
                dim_size: Optional[int] = None) -> torch.Tensor: ...
def scatter_mean(src: torch.Tensor, index: torch.Tensor, dim: int = -1,
                 out: Optional[torch.Tensor] = None,
                 dim_size: Optional[int] = None) -> torch.Tensor: ...
def scatter_min(src: torch.Tensor, index: torch.Tensor, dim: int = -1,
                out: Optional[torch.Tensor] = None,
                dim_size: Optional[int] = None) \
        -> Tuple[torch.Tensor, torch.Tensor]: ...
def scatter_max(src: torch.Tensor, index: torch.Tensor, dim: int = -1,
                out: Optional[torch.Tensor] = None,
                dim_size: Optional[int] = None) \
        -> Tuple[torch.Tensor, torch.Tensor]: ...
def scatter(src: torch.Tensor, index: torch.Tensor, dim: int = -1,
            out: Optional[torch.Tensor] = None,
            dim_size: Optional[int] = None,
            reduce: str = "sum") -> torch.Tensor: ...
```

| API | reduce | 返回值 |
| --- | --- | --- |
| `scatter_sum` / `scatter_add` | sum / add（等价路径） | `out` |
| `scatter_mul` | mul | `out` |
| `scatter_mean` | mean | `out` |
| `scatter_min` / `scatter_max` | min / max | `(out, arg_out)` |
| `scatter` | 由 `reduce` 分发六种模式 | 仅 `out` |

## 需求拆解

| 层级 | 责任 |
| --- | --- |
| Python | 原版签名、广播、dim/shape/out 校验、L2 回退、热点检测与缓存、返回值语义 |
| C++ Host/pybind | 当前 stream、ND workspace、shape 折叠、TilingKey/路径选择、launcher |
| Ascend C Kernel | L1 七 dtype、六 reduce、通用路径、vector 快路、mean count、arg_out |
| 测试与证据 | 337 pytest、180 case、官方 AscendOpTest、条件 ATK、40 点性能、高冲突、profiler |

# 详细设计（required）

## 算子分析

### 数学公式

把 shape 沿 dim 折叠为 `before × dim_length × after`。对逻辑位置 `(b,e,a)`：

```text
target = index[b,e,a] 或 compact_index[e]
out[b,target,a] = REDUCE(out[b,target,a], src[b,e,a])
```

无 provided out 时，sum/add/mean 初始值为 0，mul 为 1，min/max 内部先用极值哨兵，
finalize 后空桶 value 为 0。mean 对每个目标 bucket 计数：浮点做普通除法，整数使用
floor 除法。min/max 还输出沿 dim 的源位置；空桶 arg 为 `dim_length`，平局取最大源
位置（后写者）。

| reduce | 自动创建 out 的初值/空桶 | provided out 行为 |
| --- | --- | --- |
| sum/add | 0 | 在现有值上累加；add 与 sum 共用实现 |
| mul | 1 | 以现有值为初值继续连乘 |
| mean | 0 | 现有值进入 sum 路径，再按本次命中计数做均值化 |
| min/max | 极值哨兵，空桶 finalize 为 0 | 以现有值为初值继续比较，不重新 fill |

`scatter_min/max` 的 `arg_out` 为 int64、shape 与 out 相同；空桶为
`src.size(dim)`，平局返回沿 dim 的最大源位置。

### 支持数据类型

| 级别 | dtype | 路径 | 性能考核 |
| --- | --- | --- | --- |
| L1 | float16、bfloat16、float32、int8、int16、int32、uint8 | Ascend C native | 是 |
| L2 | float64、int64 | 同步 CPU 回退、结果 H2D | 否 |

index 固定为 int64；arg_out 对外固定为 int64。L2 选择 CPU 回退是因为 950PR 不提供
float64/int64 原子归约，且该路径可保证 CPU 参考语义和 provided out 身份。

### 支持形状与参数约束

| 项目 | 约束与处理 |
| --- | --- |
| `src` | rank 1～8；L1/L2 dtype 如上；空 Tensor 合法 |
| `index` | dtype 固定 int64；元素数 0～10^8；值域为 `[0, out.size(dim)-1]` |
| `dim` | `[-src.dim(), src.dim()-1]`，负值先加 `src.dim()` 归一化 |
| `dim_size` | `None` 或非负 int；不得小于 `index.max()+1` |
| `out` | dtype/device 与 src 相同；rank 及非 scatter 维 shape 一致 |
| `reduce` | `sum/add/mul/mean/min/max`，其他值直接报参数错误 |
| device | src/index/out 必须位于同一设备；L1 执行于 NPU |

未提供 out 时，输出 dim 长度依次由显式 `dim_size`、空 index 的 0、或
`index.max()+1` 决定。非连续 src/index 在 Python 层形成连续工作副本；非连续 out
计算后 copy-back 并返回原对象。

index 广播按以下固定步骤执行：

1. 先归一化负 dim；
2. 对 1D index，在 dim 之前补 leading 维，并在尾部补维到 `src.dim()`；
3. 将 index expand 到 src 的逻辑 shape；compact 1D 场景可向 Kernel 传递原始向量并
   由地址映射隐式广播，结果必须与显式 expand 等价；
4. 广播后保证 `src.dim() == index.dim()`，且各非末维满足任务书的 shape 约束；
5. 不可广播、越界 index 或非法 dim/dim_size 在 Python/Host 层 fail-fast。

## 算子实现

### PyTorch/Host 侧设计

1. `_normalize_dim`、`_validate_dtype` 和广播先 fail-fast；index min/max 绑定 Tensor
   identity 与 `_version` 缓存，原地修改自动失效。
2. output shape 由 out、dim_size 或 index max 决定；provided out 校验 dtype/device/rank
   和非 scatter 维。
3. L2 真正发生 NPU↔CPU 往返时发出一次 warning；CPU 输入不会错误消耗该 warning。
4. L1 调用 native extension，并在 torch-npu 当前 stream 上发射，workspace 使用
   `empty_with_format(..., ACL_FORMAT_ND)` 保证物理布局。
5. 对 compact 1D index 的热点检测仅发生在 warm-up 首次调用：index D2H 后 bincount，
   只有严格多数 bucket 才返回 `hot_target`。结果按 identity/_version 缓存，因此正式
   Event 计时循环不包含同步。

### Kernel 侧设计

#### 分核策略

| 路径 | 场景 | 并行方式 |
| --- | --- | --- |
| lane-owner | 通用确定性/非原子规格 | 每个输出 lane 独占遍历 dim |
| SIMT atomic | 通用 sum/mean 等 | src 元素并行原子归约 |
| vector atomic | compact 1D、连续对齐、f32 sum/mean（支持 out） | 特征 tile 向量读写与原子 |
| f16 float workspace | compact 1D、f16 sum/mean | FP32 累加，finalize 一次收窄 |
| vector min/max | compact 1D、f16/f32、无 out | value/arg/finalize 多阶段 |

非对齐、小张量、mul 和其他 dtype 回退通用路径；provided out 只有满足 compact/连续对齐
条件的 f32 sum/mean 可走 vector atomic，其余仍回退通用路径。任务书五档的 sum/mean
最多用 56 AIV，min/max 最多用 48 AIV；block 数同时受工作量限制。

#### TilingKey 规划

Host 将 reduce、path、indexMode、hasOut、blockNum、hotTarget、before/dim/after/output
等写入 `ScatterTilingData`。实际 TilingKey 与分流如下：

| TilingKey | 路径 | 主要触发条件 |
| ---: | --- | --- |
| 0 / 1 / 2 / 3 / 4 | lane sum / mul / mean / min / max | 通用确定性路径及快路不满足时的回退 |
| 10 / 12 | atomic sum / mean | L1 dtype 的通用并行原子路径 |
| 20 / 22 | vector atomic sum / mean | compact 1D、连续对齐、float32 |
| 30 / 32 | FP16→FP32 workspace sum / mean | compact 1D、float16、无不兼容 out |
| 43 / 44 | vector min / max | compact 1D、float16/float32、无 out |

add 在 Host/Python 层复用 sum 的 TilingKey。dtype 由 launcher 模板分派，不复制
Python 语义逻辑；任一快路约束不满足时均回退到功能完整的 lane/atomic 路径。

#### 高冲突策略

启用条件为 float16/float32、compact 1D index、`before=1`、`after<=4096`、无 provided
out、reduce 属于 sum/mean/min/max，且某 bucket 命中严格过半。每个 AIV 在 UB 中对
热点 bucket 做局部归约，再对每个 feature 仅执行一次全局 atomic；普通 vector kernel
跳过该 bucket。

- sum：局部求和后单次全局累加；
- mean：value 和 count 都先局部归约，热点 count 使用独立 compact hot-count kernel；
- min/max：局部保留极值，并以最大源位置实现后写者；value/arg 各只提交一次；
- random/非多数负载：`hotTarget=-1`，完全保留原 vector 路径。

热点判定不放入每次 timed call，避免以 D2H 同步换取虚假的 Kernel 加速。最终高冲突
门禁覆盖 random、90% hot 和 all-zero 三种 pattern，24/24 通过，最大慢化仅
`1.089914x`。

#### mean 计数路径

通用 ordered lane-owner 与 generic SIMT atomic mean 按官方 torch-scatter 语义使用
`src.dtype`（即输出 dtype）的完整 shape count，因此会保留整数溢出及低精度舍入行为。
compact/连续对齐的 f32 vector atomic 路径使用 int32 count，并同时支持有、无 `out`；
f16 float-workspace 路径同样使用 int32 count，但只在无 `out` 时启用。这两类路径只分配
`before × output_dim` count，而不是完整 out shape，最大任务档约 64 KiB，避免额外清零
约 134 MiB。compact count 从 index 生成并广播到 feature 维；f32 provided out 先把现值
带入 vector atomic 累加，再按 count clamp(min=1) 除；未命中该优化的 provided out 走
通用 dtype-count 路径并保持相同公开语义。

#### min/max 与 arg_out 多阶段路径

vector 路径先初始化 FP32 value workspace 和 int32 arg workspace，再原子归约 value，
随后只对等于最终极值的位置更新最大源位置，最后 cast/copy 到用户 dtype 和 int64 arg。
热点路径并行缓存局部 value/arg，普通 arg kernel 跳过热点 target，避免竞争。空桶在
finalize 写 value=0、arg=dim_length。

#### 初值与空张量

vector zero kernel 负责无 out 的 destination 和 compact count 初始化。通用 lane 路径
按 reduce 构造初值；空 src/index 不发射非法工作量，直接返回正确 shape/initial value。

### L2 CPU 回退

float64/int64 的 src、index 和可选 out 同步复制到 CPU，使用同一前向参考语义执行，
然后把 value/arg 复制回原 NPU。非连续 out 仍 copy-back 到原对象；结果 device、dtype、
shape 和返回类型与 torch_scatter 对齐。该路径不参加性能门禁，但已覆盖六 reduce、
provided out 和 arg_out bit-wise 测试。

### 可选 aclnn 层

工程内部提供 ACL launcher 供 Kernel smoke 和 pybind 使用，但任务书明确 aclnn 公共 API
不是必选验收项。本交付不以 direct-ACL smoke 替代 PyTorch 层正式结果。

## 支持硬件

| 硬件 | 状态 |
| --- | --- |
| Ascend 950PR（dav-3510） | 已构建、测试、性能与 profiler 验证 |
| 其他架构 | 未声明支持，需单独移植验证 |

## 算子约束限制

1. 仅前向；backward 不在本任务验收范围。
2. 浮点 atomic 的求和次序不确定，按生态精度标准比较，不要求 bit-wise。
3. L2 回退会同步且性能不作承诺。
4. 热点缓存依赖 Tensor identity 和 `_version`；原地更新会触发重算。
5. 当前高冲突专用快路只针对任务书核心 compact f16/f32 sum/mean/min/max 场景，其他
   场景仍有功能正确的通用路径。

# 特性交叉分析

| 特性交叉 | 处理 |
| --- | --- |
| 广播 × 非连续 | 广播逻辑保持 stride 语义，native 前形成连续工作副本 |
| provided out × mean | 保留 out 现值参与 sum，再用 count clamp(min=1) 除 |
| min/max × 平局 × 热点 | value 后再求最大源位置；热点/普通路径互斥提交 |
| empty × dim_size/out | 不读 index max，直接按显式 shape 构造输出 |
| L2 × provided out | CPU 工作副本计算后 copy-back，返回原 out 对象 |
| 高冲突 × profiler | 既做独立 slowdown 门禁，也核验实际 Kernel 名称 |

# 可维可测分析

## 精度标准/性能标准

精度链路以隔离官方 CPU torch-scatter 为单标杆，NPU 结果必须通过公开 Python API
获取。整数与 arg 精确比较；浮点记录 max absolute、max/mean/RMS relative error。
官方 AscendOpTest compare 所有输出通过时，按任务书 ATK 双标杆不触发；若任一输出
失败，则必须运行 ATK `cv_fused_double_benchmark` 并满足 2/1.2/1.2 比例门限。

性能采用 NPU Event，warmup 50/repeat 200，逐点检查
`A100_baseline_ms / NPU_P50_ms >= 0.6`。高冲突另以相同 reduce/dtype/shape 的 random
P50 为基准，hot90/all-zero 不得超过 2 倍。

## 测试矩阵

| 层级 | 覆盖 | 最终结果 |
| --- | --- | ---: |
| pytest | API、异常、广播、out、L1/L2、非连续、空张量、缓存失效 | 337/337 |
| 公开 API case | TC-01～TC-16、rank 1～8、9 dtype、6 reduce | 180/180 |
| 官方 AscendOpTest | value 与 arg 二进制输出 | 240/240 |
| 任务书性能 | 4 reduce × 2 dtype × 5 shape | 40/40 |
| 高冲突 | 4 reduce × 2 dtype × 3 pattern | 24/24 |
| profiler | f32 sum/mean，四种预期 Kernel | PASS，312 events |

## 多路径对比与最优方案选取

代表档 `16384×512`、48 blocks、warmup 5/repeat 30 的 ACL Event kernel-only 结果
如下（ms；sum/mean 为 lane / atomic / vector，min/max 为 lane / vector）：

| reduce | float32 | float16 | 选择 |
| --- | --- | --- | --- |
| sum | 5.876497 / 0.353220 / **0.053641** | 5.711532 / 0.347732 / **0.081703** | vector |
| mean | 8.528815 / 0.416770 / **0.088849** | 8.461747 / 0.409772 / **0.089881** | vector |
| min | 6.677403 / **0.170036** | 6.604655 / **0.179396** | vector |
| max | 6.716998 / **0.170004** | 6.673584 / **0.179442** | vector |

sum/mean 的 24/32/40/48/56/64 block 扫描在 56 blocks 最优，f32
`65536×512` 分别为 `0.368687/0.442709 ms`；64 blocks 回退。min/max 的
`16384×512` 四组合均在 48 blocks 最优。因此最终 Host 选择 sum/mean 上限 56、
min/max 上限 48。历史原始对比保存在 `docs/evidence/path_compare_r32.txt`，正式结论仍以
公开 PyTorch API 的 40 点结果为准。

## 兼容性分析

API 名称、参数顺序、默认值、返回类型、shape/dtype/device/out 行为均与目标接口对齐。
min/max 平局按任务书规定的后写者优先执行，并通过 manifest 记录和独立 verifier
验收。CANN 9.1.0 beta3 与 torch-npu 2.10 的当前 stream、ND format 和 profiler 均已
实机验证。

## 实施与交付状态

截至 2026-08-04，Ascend 950PR 实机完成完整构建、337 pytest、180 NPU case、官方
AscendOpTest 240 outputs、条件 ATK 记录、40 点性能、24 点高冲突和 profiler。最终
fail-closed 审计为 PASS。设计、自测、测试代码、原始日志、JSON/CSV、trace、整体结果
图和确定性打包工具均在交付目录中。

上述结果来自本项目截至 2026-08-04 的自验证链。2026-08-07 已在同一 Ascend 950PR
环境复核官方仓后来新增的 `self_test_case/ops-gnn/test/test_scatter.py` 与
`benchmark/run_benchmark.py`。复核发现官方测试脚本存在固定 device 4、跨 dtype
`allclose`、min/max 空桶 reference、BF16 固定 1e-3 容差、无关算子预加载和 benchmark
参数作用域问题；不以这些脚本缺陷反向修改任务书语义。任务书一致版 pytest 结果为
496 passed、204 skipped、50 failed，剩余 50 项仅为 FP16/BF16 并行累加次序，当前
明确记为 `PENDING_OFFICIAL_CONFIRMATION`。修正后的官方 benchmark 副本已完成四归约、两 dtype、
五 shape 的正式复跑，40/40 达到门槛，最低 ratio `0.680965`。完整结果见
`docs/evidence/official_self_test_20260807.md`。


官方新增测试框架的判定与修正问题已在 [CANN 项目讨论 #22](https://gitcode.com/org/cann/discussions/22) 公开咨询。
