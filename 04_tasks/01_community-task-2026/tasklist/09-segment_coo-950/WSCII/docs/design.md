# 【社区任务】segment_coo 算子设计文档

- **任务**：CANN训练营北京邮电大学-segment_coo算子开发(950)
- **提交团队/目录**：`WSCII`
- **目标硬件**：Ascend950PR（arch35 / dav-3510），CANN 9.1.0
- **实现候选**：`c31-all-packed-coordinates`
- **个人实现仓**：**尚未创建**（GitCode `WSCII` 名下当前无 `ops-gnn` 系列仓库）。
  按任务书要求，验收阶段将创建个人仓、推送与验证一致的固定 SHA，并邀请 `Ascend-CANN` 为 Developer。
  本文档（设计阶段）不依赖该仓库已存在。
- **参考标杆**：torch_scatter 2.1.2 `segment_coo` 系列接口

> 说明：本设计描述的是已验证的 c31 实现。文中保留该实现在自测中发现的一条性能未达标项与一处已知对齐风险，
> 不隐藏、不美化；设计评审与性能验收是两件事，本文件不主张性能已通过。

# 需求背景（required）

## 需求来源

用户附件《CANN训练营北京邮电大学-segment_coo算子开发(950)》。
参考 torch_scatter 2.1.2 的 Segment COO 系列接口，在 Ascend C Kernel + PyTorch 适配层实现前向归约。
以任务书明确条款为依据；存在的上游语义冲突单列说明。
设计结构采用官方 `design_template.md`，不沿用其 Addcdiv 示例内容。

## 背景介绍

图神经网络消息按目标节点编号排序后，同一段在输入中连续。
通用 scatter 支持无序索引，通常需要原子或通用归约；Segment COO 可先定位有序段边界，
再连续读取段内数据，由唯一输出拥有者写回，减少输出竞争。
本实现每次使用真实 index，不假设测试中的等长分组，不缓存边界或输出。

### segment_coo 与 scatter 选型差异

| 维度 | 通用 scatter | 本 segment_coo 设计 |
| --- | --- | --- |
| index 前提 | 任意顺序、可重复；需原子或冲突归约 | **要求 index 非降序**，据此可定位连续段 |
| 输出竞争 | 多写者可能竞争同一输出位置 | **每个段由唯一拥有者写回**，段内串行、段间并行 |
| 边界定位 | 无段概念 | `DirectLowerBound` 求 `L_s=lower_bound(index,s)`、`R_s=lower_bound(index,s+1)` |
| 空段语义 | 不适用 | 空段输出 0，arg 哨兵为 N |
| 性能取向 | 通用性优先 | 有序前提换取连续访存与免原子 |

选型结论：在「index 已有序」这一任务前提下，段扫描路径比通用 scatter 更省同步与原子开销；
但该优势需要同图对照实测才能量化，当前为 **NOT_RUN**（见「路径及性能对比的证据边界」）。

### index 有序性校验的边界

任务书把「index 非降序、非负且不超过输出范围」作为**调用前提**。本设计据此约定：

- Host 侧校验形状、类型、rank、广播关系与 out 一致性；
- **不对设备侧 index 内容做额外全量遍历校验**（代价高且与性能目标冲突）；
- 无序/越界属**未定义行为**，不以非法输入触发设备 assert 作为测试手段；
- `DirectLowerBound` 对真实首尾 index 做插值提示并前驱验证，失败回退精确二分，
  因此对**极不均匀、有空洞、非零起点**的分布不会按等长段猜测输出。

# 需求分析（required）

## 需求描述

提供 segment_coo 及 sum/add/mean/min/max 子接口，PyTorch 层验收；
支持七种类型、广播、1–8 维、非连续 tensor、空段、out 和 min/max 索引输出。
464 个公布性能单元均须达到 `reference_ms / measured_ms >= 0.45`。

## 需求拆解

| 内容 | 实现 / 验证边界 |
| --- | --- |
| 有序段定位 | 检查 Host 形状/类型/设备；有序及范围是调用前提，不对设备内容额外遍历校验 |
| 精确整数 | INT32/INT64 保持原生整数累加/除法；不经 FP32 转换 |
| 低精度浮点 | FP16/BF16 以 FP32 累加后转回输出类型 |
| 广播及 layout | index 前导维支持 1 或对应 src 维；归约轴为 `index.dim()-1` |
| out 和 arg | 按任务书明确规则，语义差异见后文 |
| 性能 | 连续段扫描、packed 读取、协作行归约、INT64 mean 融合边界、精确坐标分解 |
| 验收 | 原题 646 测试保留；1458 功能与 252 独立精度通过；性能 463/464（1 项未达标） |

# 详细设计（required）

## 算子分析

### 数学公式

将 src 逻辑展平为 `[B,N,C]`，index 广播为 `[I,N]`（I=1 或 B），输出为 `[B,S,C]`。
每个批次中，段 s 的真实区间为

```
L_s = lower_bound(index, s)
R_s = lower_bound(index, s+1)
```

无 out 时，sum 为区间逐项求和，mean 为 `sum/(R_s-L_s)`，min/max 为区间极值。
空段输出 0，arg 哨兵为 N。min/max 相等时选择较大的段内源坐标（后写者）。
整数 mean 向零截断；浮点归约允许精度标准内的舍入差异。
INT8/UINT8 mean 遵循 CPU 实现的窄化 sum 和 count 后除法；当非空 count 窄化为 0，
CPU 原生除零未定义，本实现走安全数学均值，作为显式边界而不是 CPU 一致性宣称。

### 支持数据类型

src/out：float16、float32、bfloat16、int8、uint8、int32、int64。
index 及专用 min/max 的 arg：int64。通用 segment_coo 只返回值，不构造不需要的 arg。
sum/add 等价；mul 抛 `ValueError`。七类型参与功能；任务书四类型参与性能。

### 支持形状

src rank 1–8；`1 <= index.rank <= src.rank`。最后 index 维长度等于归约轴 N；
index 各前导维可以是 1 或对应 src 维度；尾部特征共享 index。
out 维数、非归约维、dtype、device 需与 src 匹配。
S 优先取 out 的归约维长度，其次 `dim_size`，否则取每批有序 index 末元素的最大值+1；
空 index 自动推导 S=0。动态推导中的 `.item()` 可能同步，未移到计时外。

支持的非连续 layout 由连续化暂存处理；但 c31 对「已连续但地址未对齐」的偏移视图
没有额外防护，所以不能笼统声称所有非连续/偏移视图均安全。

## 算子实现

### Host 侧设计

Host 验证 NPU 设备、index int64、rank、广播、reduce 及 out 形状；用 DeviceGuard 选择设备。
将非共享 index 展开并连续化，src 连续化。按 `[B,N,C,S,I]` 填充 `SegmentCooPlan`，
并传入 reduce/dtype/hasOut/blocks；blocks 采用平台 AIV 核心数，至少 1。

一般路径每次分配 `I*(S+1)` 个 int64 指针元素（`8*I*(S+1)` 字节）；
packed INT64 mean 在受限 32 位地址域内直接传 index，免指针 workspace 和前置 kernel。
out 按需分配；arg 仅专用 min/max 分配。非连续 out 使用暂存并 `copy_` 回原视图。

使用当前 NPU stream 与 `OpCommand::RunOpApi`，lambda 按值持有 src/index/out/arg/ptr。
这样连续化、内核与 `copy_` 保持框架队列顺序，临时张量活到 launch 执行。
不以全局同步代替异步生命周期；返回 ACL 线程级 last error。

### Kernel 侧设计（有序段扫描算法）

本实现使用 **arch35 SIMT 模板化归约**，并非模板示例中的通用 CopyIn/Compute/CopyOut 双缓冲流水。
没有独立自动调优器、数据相关跨调用缓存或 CPU 归约兜底。

1. **PointerKernel32** 用于 N、S 和 `I*(S+1)` 在 INT32 正范围内的坐标域；
   宽域保留 int64 坐标 PointerKernel。
2. **DirectLowerBound** 依据真实首尾 index 计算整数插值提示，读取提示和前驱验证
   lower_bound 条件，失败回退精确二分。极不均匀、空洞、非零起点不会按等长段猜输出。
3. **ReductionKernel** 按输出坐标逐段扫描，是 256 线程的通用标量回退。
4. **PackedKernel** 在 C 为 8 的倍数且输入/输出总元素均 `< INT32_MAX` 时使用 packed 列：
   普通类型 width8，INT64 一般 width4。所有访存仍依赖实际区间。
5. 当 C 及 S 的 packed 坐标是 2 的幂时，用无符号 mask/shift 精确分解坐标，
   其他形状保留除余。c31 将这条精确坐标路径扩展到全部 packed 归约；
   不对有符号归约算术做位移替代。
6. 长非 INT64 sum/mean（`C%32=0`、`N>=16*S`）采用**协作归约**：
   512 线程，4 个列 lane 与 8 个行分片，XOR 4/8/16 合并。
7. INT64 mean 融合边界定位；`N<=16*S` 且 `C<=32` 采用 width2/1024 线程，
   同段长且 `C>32` 采用 width4/512，较长段 width4/1024。
8. INT64 其他 packed 值输出一般 1024 线程；需要 arg 的极值路径 512 线程。
   通用 min/max 值接口避免 arg 张量；专用接口保留后写者 arg 语义。

### 多路径性能对比的证据边界

| 路径 / 对照 | 选择理由 | c31 现有证据 |
| --- | --- | --- |
| 标量 / packed / 2 幂坐标 | 通用性与地址计算开销折中 | 统一 dispatch 在 1458 项中测试；**无逐路径禁用 ablation** |
| 协作长段 | 在 lane 间分担扫描长度 | quick 48/48、critical 16/16 仅筛选，**非完整验收** |
| INT64 mean 融合 | 少一次 pointer 生成与 launch | 完整 INT64 性能记录在 464 表内；**无同环境单因素测速** |
| 四种 reduce 总体 | 同一完整原题运行 | sum 116、mean 115、min 116、max 116，各总数 116 |
| native scatter 同图对照 | 衡量有序路径优势 | **NOT_RUN**；不能用 CPU fallback 替代 |

推测减少地址除法/启动次数有助于相应形状，但现有记录**不足以量化各项单独收益**。
不把历史不同候选的最好单项拼成对照结论；仍保留唯一失败项。

## 支持硬件

验证：Ascend950PR，arch35 / dav-3510，CANN9.1.0；
**未验证 Atlas A2/A3 或其他架构**。本包构建依据 ops-gnn 基线 CMake，不宣称跨架构可运行。
历史环境：x86_64、runtime 可见 `total_memory=126112MB`、driver 25.7.rc1.6、Python 3.12.13、
torch 2.7.1+cpu、torch_npu 2.7.1.post8；CPU quota 32，`OMP_NUM_THREADS=8`。

## 算子约束限制

index 非降序、非负且不超过输出范围是前提；无序/越界行为未定义，不以非法输入
触发设备 assert 作为测试。backward/autograd、NaN/Inf、多卡互调未作已验证承诺。

任务书 sum/add out 明确要求覆盖，与 torch_scatter CPU 的非零 out 累加不同；
本实现覆盖且空桶置 0。mean 以现有 out 为初值参与 sum/count，min/max 与现有值比较；
mean/min/max 保留有 out 时的空桶。任务书后写者 tie 又与 CPU first-writer 不同。
测试针对这些明确规则独立断言，**不能宣称上游所有语义完全一致**。

**已知风险（未修复）**：c31 的 `src.contiguous()` / `out.contiguous()` 可能保留非零 storage offset；
packed 访存没有 `data_ptr` 对齐检查。后续 c32 同类访问已出现 507035，
**c31 本次未补测该风险也未修复**。若修复，必须另立新版本重新验收，不能沿用 463/464。

# 可维可测分析

## 精度标准/性能标准

| 标准 | 方法与结论 | 来源 |
| --- | --- | --- |
| 原题功能 | 原 646 用例保留；扩展后 1458/1458 通过 | build-screen/correctness.log |
| 单标杆精度 | 真正 CPU torch_scatter 2.1.2+pt27cpu；252/252 | precision/precision-edges.json |
| 浮点审计 | ≥99% 元素满足 `abs_err<=atol+rtol*abs(ref)`，且最大绝对误差 ≤ limit | audit_precision_edges.py |
| 整数审计 | bit-wise 精确；不使用浮点近似 | 同上 |
| **全量性能** | **464 中 463 通过**；要求每项 ≥0.45，不是中位数 | full/full/round-1/reports/ |
| 同图 scatter 优势 | **NOT_RUN** | 待补 |

浮点审计阈值 `(rtol, atol, 最大误差)`：FP32=(2^-10, 2^-16, 0.01)，
FP16=(2^-9, 2^-9, 0.1)，BF16=(2^-6, 2^-6, 1.0)。
完整参数、实际误差及整数 mismatch count 在 252 项 CSV/XLSX，**不为未记录的 1458 项编造误差数值**。

计时原脚本不改：20 warmup，100 次完整 Python API 调用，前后同步的 wall 平均；
计入 out 分配、dispatch、pointer 生成及 kernel，不用 device event 或五轮中位数代替。
任务书表的 ms 统一换算为 us；对原始三位小数打印采用 `+0.0005ms` 保守上界。
**唯一失败** mean FP16 `(4194304,64,65536)`：4.629ms，参考 0.919ms，上限 2.042222ms。
两轮是项目预登记的可重复性检查，非另造官方计分法；第一轮失败后第二轮 NOT_RUN。

## 兼容性分析

原题测试、benchmark 和上游绑定保持；CSR 一致性适配并非独立真值，
独立 CPU 审计用于避免循环验证。详细命令见 `test/segment_coo/README.md`。
新 `golden.py` 是实际 CPU 参考入口，并明确列出任务书差异，不替换原题用例。

整理版 C++ tokens / Python AST 与冻结源码等价，重编译正确性另存 package-validation；
**463/464 仅绑定原始二进制 SHA，不能自动归属重编译二进制**。
大输入 464 项采样正确性对 c31 为 NOT_RUN。最终外部验收仍需全量性能与其他待项闭环。

## 官方参考

- 任务书（用户附件）
- <https://gitcode.com/cann/ops-gnn> 及其 `development_guide.md`
- <https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md>
- <https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md>
