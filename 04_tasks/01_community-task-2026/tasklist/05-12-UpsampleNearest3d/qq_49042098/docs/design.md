# 需求背景（required）

## 需求来源

社区任务要求在现有 `aclnnUpsampleNearest3d` 基础上补齐 `UINT8` 数据类型支持，完成算子设计、Ascend C 开发与测试，并将结果提交到 `ops-cv`。任务目标覆盖 Atlas A2/A3、rank-5 Tensor、`ND/NCDHW/NDHWC` 格式、非连续 Tensor 和完整 `UINT8` 值域。

本文依据当前任务书、当前 8 月测试集和 `ops-cv` 中现有 Ascend C 实现编写。测试 case 是合同覆盖样本，不参与生产路径分派。

## 背景介绍

UpsampleNearest3d 对五维 Tensor 的 D、H、W 三个空间维执行最近邻采样。每个输出位置只复制一个输入元素，不进行加权插值。现有接口已经支持浮点类型，本任务新增 `UINT8` 的 OpDef、Host tiling、A2/A3 binary 和 AI Core 数据流闭环。

当前实现位于 `experimental/image/upsample_nearest3d`，沿用正式算子的 OpType、kernel 名称和 ACLNN 接口。experimental 与正式 `image/` 构建树由 `ENABLE_EXPERIMENTAL` 隔离，避免同一次构建重复注册同名算子。

# 需求分析（required）

## 需求描述

实现任务书合同内的 UpsampleNearest3d，支持 `FLOAT32`、`FLOAT16`、`BFLOAT16`、`DOUBLE`、`UINT8`。输入和输出 dtype、format 必须一致，输入 rank 为 5；`outputSize` 指定输出 D/H/W，`scalesD/H/W` 可显式指定缩放系数。所有支持 dtype 的输出都必须与 CPU nearest-neighbor 参考逐元素完全一致；其中 `UINT8` 对完整 `[0,255]` 值域执行原始字节复制，浮点与 DOUBLE 同样只复制被选中的输入元素，不进行插值运算。

公开 ACLNN 原型如下：

```cpp
aclnnStatus aclnnUpsampleNearest3dGetWorkspaceSize(
    const aclTensor *self,
    const aclIntArray *outputSize,
    double scalesD,
    double scalesH,
    double scalesW,
    aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

aclnnStatus aclnnUpsampleNearest3d(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);
```

参数合同如下：

| 参数 | 输入/输出/属性 | dtype | shape/取值 | 说明 |
| --- | --- | --- | --- | --- |
| `self` | 输入 | `FLOAT32`、`FLOAT16`、`BFLOAT16`、`DOUBLE`、`UINT8` | rank 5 | `ND` 按 NCDHW 解释；支持 NCDHW、NDHWC |
| `outputSize` | 输入 | `INT64` | 长度 3，元素均大于 0 | 输出 `[outD,outH,outW]` |
| `scalesD/H/W` | 属性 | `double` | 三者均为正时显式生效 | 否则由输入/输出 shape 推导 |
| `out` | 输出 | 与 `self` 相同 | rank 5 | N、C 与输入一致，空间维等于 `outputSize` |

输入与输出空间维必须为正，总元素数均不得超过 `INT32_MAX`。非连续输入由 ACLNN 在 device 上执行 `Contiguous`，非连续输出通过 executor 的 `ViewCopy` 写回。

## 需求拆解

1. 在 A2/A3 OpDef、binary config、Host tiling 和 kernel 模板中闭合 `UINT8 + ND/NCDHW/NDHWC` 签名。
2. 精确实现显式 scale 与 shape 推导两种最近邻源坐标公式，使所有支持 dtype 与 CPU nearest-neighbor 参考逐元素一致。
3. 根据 runtime dtype、format、shape、scale、UB 容量和 AIV core 数选择通用 packed、row gather、depth run 等数据流。
4. 为每条路径定义不重叠的 GM 输出所有权、UB 资源预算、尾块和同步协议。
5. 保留浮点原有能力和 Ascend 950 arch35 路径，不用它们替代 A2/A3 `UINT8` 实现。
6. 对不满足 guard 或资源合同的专用路径回到本算子自有通用实现；非法合同明确失败，不调用 CPU、Python、框架或 vendor whole-op。

# 详细设计（required）

## 算子分析

### 数学公式

当三个显式 scale 均为正时：

```text
id = min(floor(od / scalesD), inD - 1)
ih = min(floor(oh / scalesH), inH - 1)
iw = min(floor(ow / scalesW), inW - 1)
```

未显式提供有效 scale 时：

```text
id = min(floor(od * inD / outD), inD - 1)
ih = min(floor(oh * inH / outH), inH - 1)
iw = min(floor(ow * inW / outW), inW - 1)
```

输出关系为：

```text
NCDHW/ND: out[n,c,od,oh,ow] = self[n,c,id,ih,iw]
NDHWC:    out[n,od,oh,ow,c] = self[n,id,ih,iw,c]
```

### 支持数据类型

任务书支持 `FLOAT32`、`FLOAT16`、`BFLOAT16`、`DOUBLE`、`UINT8`，输入输出 dtype 必须一致。该算子只做源坐标选择和元素复制，因此所有 dtype 共用同一个 exact 语义，不存在“UINT8 exact、浮点近似”的双重精度合同。本任务的代码开发重点是补齐 A2/A3 `UINT8` 路径；这不缩窄浮点与 DOUBLE 的任务书语义。

### 支持形状

- 输入和输出 rank 均为 5，N、C 相同，D/H/W 均为正。
- `ND` 按 NCDHW 语义解释；任务书支持 dtype 均遵循 `ND`、`NCDHW`、`NDHWC` 的相同最近邻 shape 语义。
- `outputSize` 长度固定为 3；输入/输出元素数位于 `(0, INT32_MAX]`。
- 当前 `public_v2` 测试集含 381 个合同样本：307 个 `UINT8`、24 个 FP16、23 个 FP32、27 个 BF16；格式分布为 336 个 ND、27 个 NCDHW、18 个 NDHWC；包含 369 个连续输入、12 个非连续输入和 11 个显式 scale case。
- 当前测试集覆盖零值、255、随机值域、上采样、下采样、等尺寸、非整数 scale、31/32/33 等对齐边界、大输出与多核压力。case id 只用于验证和日志定位。

## 算子实现

### 实现方案

#### 3.2.1 host侧设计

调用链如下：

```text
aclnnUpsampleNearest3dGetWorkspaceSize
  -> 参数校验、Contiguous、ViewCopy 准备
  -> L0 UpsampleNearest3d
  -> OpDef/config 选择 A2/A3 binary
  -> Host tiling 解析 dtype/format/shape/scale/UB/core
  -> SelectDispatchLeaf
  -> Ascend C template key + runtime tiling
  -> solution-owned AI Core kernel
```

Host 从 input storage format 解析 NCDHW 或 NDHWC 轴，校验输入输出 dtype、rank、正维度和 `INT32_MAX` 上限。三项 scale 均为正时标记 `explicitScale=1`；否则保存 `in/out` 比例用于 source-index 计算。

平台侧读取 AIV core 数与 UB 容量。A2/A3 新路径不申请 GM user workspace，`workspace[0]=0`；所需临时区均由 kernel 内 UB TBuf/TQue 承载。Host 将 UB 的 7/8 作为调度预算，剩余空间留给 runtime 和对齐开销。

调度分两层：

1. 先选择实现机制对应的 `ScheduleVariant`。当前 registry 使用 5-bit key，注册 0..25 共 26 个编译变体；多个 metadata leaf 可以共享一个编译变体。
2. 再由 `SelectDispatchLeaf` 根据最终 tiling 字段解析唯一逻辑 leaf，并要求 `LeafVariant(selectedLeaf) == scheduleVariant`。不一致时 Host 直接失败；kernel 入口也复核 leaf、schedule 和容量 guard，不一致时 `Trap()`。

主要分派族如下：

| runtime 条件 | 实现族 | block/GM 所有权 | 专用 guard 不满足时 |
| --- | --- | --- | --- |
| 输入输出 shape 完全相同且无显式 scale | identity copy | 每 block 独占对齐连续区间 | 进入其他合法自有路径 |
| `UINT8` 小输出，或 row-gather UB 预算不足 | packed generic | 每 block 独占 64-byte 对齐输出区间，末 block 独占尾部 | 保持 packed 通用路径 |
| `UINT8` row-gather 预算满足 | row gather | 每 block 独占完整输出 row/plane 组 | 回到 packed generic |
| `UINT8`, `outD>inD` 且 source-plane owner 充足 | depth run | 每个 source-D plane/H split 独占连续 output-D run | 回到 row/packed 自有路径 |
| `UINT8`, `H=outH=1`, `outW=2*inW` 且完整 plane 可装入 UB | width-2x full plane | 一个 block 独占一个 N*C plane | 回到普通 depth run |
| 浮点 dtype | typed/generic | 每 block 独占 row、plane 或连续输出区间 | 回到浮点 generic |

row/depth 家族内部还按 H 重复、完整 plane、窄列、2x2 plane、整倍 depth boundary、full/paired depth slab 等 metadata 选择专用 leaf。这些条件只来自 runtime shape、format、scale、UB/core 和已计算 tiling 字段，不读取输入值、case id、文件名或 timing。

Host 计算并写入：

- source/output row stride、tile rows、rows/core、block factor；
- input/output byte buffer、half buffer、H/W offset buffer 容量；
- depth H split、depth batch rows、width2x flag；
- `selectedLeafId` 和 `selectedScheduleVariant`；
- 某些 compact plane 路径的 H/W offset payload。

所有容量与 offset 使用 64 位中间值；超出 tiling buffer、UB、Gather offset 或 `INT32_MAX` 时拒绝该专用路径或明确失败。

#### 3.2.2 kernel侧设计

##### Packed generic

每个 block 处理一个 64-byte 对齐的连续输出元素区间。kernel 以 32-bit word 作为 GM 写所有权单位，每个 word 包含四个 `UINT8` lane；每个有效 lane 由线性输出索引恢复 N/C/D/H/W，再计算唯一输入字节地址。最后一个 block 只写有效 lane，避免跨核 subword 覆盖。

##### Row gather

目标 dav-2201 不支持直接 `Gather<uint8_t>`，因此使用本算子拥有的完整转换链：

```text
GM uint8 row
  -> DataCopyPad 到 UB
  -> Cast(uint8, half)
  -> Gather<half>
  -> Cast(half, uint8)
  -> DataCopyPad 到 GM
```

W offset 只生成一次，后续 H row 通过 row base 复用。input/output row stride 按 32 元素对齐；Gather byte offset、tile rows 和 UB 总量均由 Host guard 限制。`UINT8 -> half -> UINT8` 对 `[0,255]` 为精确可逆，不改变 payload。

##### Depth run

最近邻 D 映射单调。对每个 source D，kernel 通过边界计算得到对应的连续 `[outputDStart, outputDEnd)`，只生成一次 H/W tile，然后复制到该 output-D run。Host 可按 H split 增加 source-plane task 并行度，但每个 task 的输出范围互斥。

full-depth/paired-depth slab、unique-H plane、compact narrow plane 等 leaf 复用同一不变量：一个 owner 完成所负责的完整 row/plane/run，offset 与输入 tile 在 UB 内复用，不经 GM 中间张量回写再读。

##### Width-2x full plane

满足 NCDHW、shape 推导 scale、`H=outH=1`、`outW=2*inW` 和 UB 容量条件时，一个 block 加载完整 source D rows。字节先精确转为 16-bit 表示，通过 `x | (x << 8)` 生成相邻两个相同输出字节，再按 output-D run 在 UB 内广播并整 plane 写回。

##### 尾块与同步

- `DataCopyPad` 的 block length 使用真实 row byte 数，padding 只存在于 UB stride，不写入逻辑输出。
- packed 路径按有效 lane 处理最后一个 32-bit word；block 边界保持 64-byte 对齐。
- row/plane 路径只由其 owner 写完整输出范围，其他 core 不触碰该范围。
- TQue 路径通过 EnQue/DeQue/FreeTensor 管理 buffer 生命周期；TBuf 路径在 MTE2、Vector、MTE3 复用点使用对应 event flag。
- 需要在同一 buffer 上连续执行 copy/compute/copy-out 时，上一阶段完成事件必须在下一阶段读取或复用前闭合。

所有新增 TBuf/TQue 与 GM footprint 都由 `upsample_nearest3d_resource_contracts.hpp` 的编译期 resource ledger 绑定到真实分配和访问点。正向/负向编译测试用于阻止容量字段、队列深度和 GM owner 合同漂移。

### 优化策略

1. 小输出使用 packed word 计算，避免 Gather 初始化和多 buffer 开销。
2. 常规 row 路径复用 W offset，并用 half Gather 替代不支持的 B8 Gather。
3. depth run 利用单调 source-D 映射，一次生成 H/W tile 后在连续 output-D 区间复用。
4. width-2x full-plane 路径以 16-bit packed byte broadcast 同时完成 W 复制和 D run 广播。
5. identity、完整 plane、H 重复和 slab 路径扩大单个 owner 的连续工作，减少坐标重复计算和小任务调度成本。
6. logical leaf 与 compile variant 分离：同一实现机制服务多个合法 metadata 区域，避免按公开 shape 或 case identity 建分支。

## 支持硬件

| 支持的芯片版本 | 编译 SoC | 本任务 UINT8 | 涉及勾选 |
| --- | --- | --- | --- |
| Atlas A2 | `ascend910b` | ND/NCDHW/NDHWC | √ |
| Atlas A3 | `ascend910_93` | ND/NCDHW/NDHWC | √ |
| Ascend 950 | `ascend950` | 保留原 arch35 能力，不作为本次新增证据 | - |

OpDef registration、`op_host/config/<soc>`、CMake compute unit 和 package binary 必须声明一致的 SoC/dtype/format 集合。

## 算子约束限制

1. 任务书支持 `FLOAT32`、`FLOAT16`、`BFLOAT16`、`DOUBLE`、`UINT8`；输入输出均为 rank 5 且 dtype/format 一致，空间维与 `outputSize` 均为正。
2. 输入输出元素数不超过 `INT32_MAX`；专用 leaf 还必须满足其 UB、offset、row/plane owner 和并行度 guard。
3. `ND` 使用 NCDHW 轴语义；NDHWC 由 storage format 明确选择，不能靠 shape 猜测。
4. 专用 leaf 不满足时只进入本目录的通用 packed/row/depth 实现；非法输入明确失败。
5. 生产路径不读取 OpForge task/case/golden/result，不调用 Python、CPU、PyTorch/torch_npu 等价算子、vendor `UpsampleNearest3d` whole-op、peer 或其他 backend fallback。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 全 dtype 精度 | `FLOAT32`、`FLOAT16`、`BFLOAT16`、`DOUBLE`、`UINT8` 的输出 dtype、shape 和每个元素都必须与 CPU nearest-neighbor 参考完全一致 | 社区任务书 3.2 |
| 性能背景 | 任务书给出同 shape UINT8 相对 FP16 劣化不超过 5% 的条款；该条款仅作为任务背景，本轮文档更新不声明性能达标，也不把性能作为语义/shape/精度对齐的完成条件 | 社区任务书 |
| 泛化标准 | 覆盖连续/非连续、三种格式、显式/推导 scale、上下采样、对齐尾块和大输出；不得按 case identity 分派 | 当前测试集 |
| fallback 标准 | 每个合法输入由当前 Ascend C 目录内实现执行，不支持输入明确失败 | 当前实现边界 |

### 任务书 3.2 精度判定

1. CPU reference 按本文数学公式计算每个 `(od,oh,ow)` 唯一对应的 `(id,ih,iw)`，并以输入 dtype 原样复制该元素；reference 的输出 dtype、format 和 `[N,C,outD,outH,outW]`/NDHWC 对应 shape 必须与接口合同一致。
2. candidate 与 CPU reference 的 dtype、rank、各维 shape 和元素数必须完全一致，任一结构不一致直接失败。
3. 对 `FLOAT32`、`FLOAT16`、`BFLOAT16`、`DOUBLE`、`UINT8` 全部执行逐元素 exact compare；任一输出元素与 CPU reference 不一致即失败，不设置 rtol、atol 或 matched-ratio 豁免。
4. AscendOpTest 默认阈值只能作为工具的通用比较行为，不能放宽本算子的 exact 语义。即使通用容差比较返回通过，只要存在非 exact 元素，本任务仍判定失败。
5. CPU reference 仅用于离线验收，不进入 ACLNN/AI Core 生产执行路径，也不是 runtime fallback。

当前验证边界如下：

- 当前源码仓已保存的 reviewer 记录覆盖 365 个 case，正式 ACLNN 路径从源码重建后记录为 365/365 exact；该集合覆盖 UINT8、FP16、FP32、BF16，不含 DOUBLE，因而不是任务书全部 dtype 的完整验收。
- 当前 `public_v2` 已扩展到 381 cases，新增 16 个大输出 pressure case；现有 365-case 源码仓记录和 partial-public evaluator 不能写成当前 381-case 完整通过。
- 本轮仅更新设计文档，未重新执行 381-case、DOUBLE 或其他设备测试，因此不新增精度通过结论。
- 性能条款仅保留为任务背景，本轮不声明性能达标。

本轮设计文档更新只确认语义、shape 与上述 exact 精度合同文本对齐，不包含新的设备测试或性能完成结论。

## 兼容性分析

本设计保持 `aclnnUpsampleNearest3d` 原接口、OpType 和 kernel 名称，只在 A2/A3 能力注册和自有实现中新增 `UINT8`。浮点路径和 Ascend 950 arch35 路径继续由原目录代码维护；它们不作为 `UINT8` fallback。非连续 Tensor 通过 ACLNN executor 的设备侧连续化/ViewCopy 处理，生产 kernel 始终由当前源码构建，因而不存在框架、CPU、vendor whole-op 或其他 backend 的静默替代路径。
