# MaxPool2dWithMaskBackward 算子设计文档

> 社区任务：9月社区任务-MaxPool2dWithMaskBackward 算子开发
> 适配硬件：Atlas A2 训练系列产品（Atlas 800T A2）/ Atlas 推理系列产品（Atlas 300V Pro）
> CANN 版本：CANN 9.0.0 / CANN 9.1.0
> 对标接口：`aclnnMaxPool2dWithMaskBackward` / `aclnnMaxPool2dWithMaskBackwardGetWorkspaceSize`
> 目标合入仓：cann/ops-nn → `experimental/pooling/max_pool2d_with_mask_backward/`
> 开发语言：Ascend C
> 配套正向算子：MaxPool2dWithMask（09-45 / 09-46 任务）

---

## 一、需求背景

### 1.1 需求来源

通过社区任务完成昇腾算子开源仓（ops-nn）的算子贡献需求。本任务为在 Atlas A2 / Atlas 300V Pro
上使用 Ascend C 编程语言实现 `aclnnMaxPool2dWithMask` 正向算子的反向传播算子
`MaxPool2dWithMaskBackward`，与正向严格配套，完成算子设计、开发、测试全流程工作，
验收通过后合入昇腾算子开源仓 ops-nn 的 `experimental/pooling/` 目录。

任务书来源：`docs/README.md` 9月发放任务 `#46 9月社区任务-MaxPool2dWithMaskBackward算子开发`。

### 1.2 背景介绍

#### 1.2.1 MaxPool2dWithMaskBackward 算子实现优化

基于 MaxPool2dWithMaskBackward 算子的历史 TBE 版本，使用 Ascend C 编程语言进行优化实现。

**TBE 算子实现路径**（以 CANN 9.1.0 为例，`/usr/local/Ascend/ascend-toolkit/latest` 为 CANN 安装根目录）：

| 作用 | 路径（含文件名） |
|------|------------------|
| TBE 算子注册入口 | `opp/built-in/op_impl/ai_core/tbe/impl/ops_legacy/dynamic/max_pool_grad_with_argmaxv1.py` |
| TBE 实现（FLOAT32 / NCHW 路径） | `opp/built-in/op_impl/ai_core/tbe/impl/ops_legacy/dynamic/max_pool_grad_with_argmax_v1_dsl.py` |
| TBE 实现（非 FLOAT32 路径） | `opp/built-in/op_impl/ai_core/tbe/impl/ops_legacy/dynamic/max_pool_grad_with_argmaxv2.py` |
| TBE 实现（H / W 切分辅助） | `opp/built-in/op_impl/ai_core/tbe/impl/ops_legacy/dynamic/max_pool_grad_with_argmax_cut_h_v1.py`、`max_pool_grad_with_argmax_cut_w_v1.py` |
| 算子信息库 | `opp/built-in/op_impl/ai_core/tbe/config/ascend910b/aic-ascend910b-ops-info-legacy.json`（条目 `MaxPoolGradWithArgmaxV1`） |

**TBE 实现中调用的算子 API 路径**：

| 作用 | 路径（含文件名） |
|------|------------------|
| 计算原语（NCHW + int32 argmax 路径） | `python/site-packages/tbe/dsl/compute/max_pool_grad_with_argmax.py` |
| 计算原语（NCDHW / NC1HWC0 mask 路径） | `python/site-packages/tbe/dsl/compute/max_pool2d.py` |

**参考文件选取依据**（结合 aclnn 接口判断）：

1. aclnn 接口 `aclnnMaxPool2dWithMaskBackward` 的官方说明位于
   `pooling/max_pool3d_grad_with_argmax/docs/aclnnMaxPool2dWithMaskBackward.md`（ops-nn 仓），
   其中声明输入为 `gradOutput / self / indices / kernelSize / stride / padding / dilation / ceilMode`，
   `indices` 为 INT8、`gradOutput`/`self` 为 FLOAT32/FLOAT16/BFLOAT16，均 NCHW。
2. 对应的 aclnn 实现为 `pooling/max_pool3d_grad_with_argmax/op_api/aclnn_max_pool2d_with_indices_backward.cpp`，
   其核心是调用内部算子 **`MaxPoolGradWithArgmaxV1`**（FLOAT32 时）——与上面注册入口
   `max_pool_grad_with_argmaxv1.py` 中 `@register_operator("MaxPoolGradWithArgmaxV1")` 一致，
   因此可确认参考文件选取正确。
3. `max_pool_grad_with_argmaxv1.py` 内部按 dtype 分派：
   `grad.get("dtype") == "float32"`（或未知 rank）时走
   `max_pool_grad_with_argmax_v1_dsl.py` → `tbe.dsl.compute.max_pool_grad_with_argmax`；
   其余 dtype 走 `max_pool_grad_with_argmaxv2.py` 的 `MaxpoolGrad` 类（NC1HWC0 + uint16 mask 路径）。

#### 1.2.2 标杆算子现状分析

##### 1.2.2.1 标杆算子支持的数据类型和数据格式

与算子信息库 `aic-ascend910b-ops-info-legacy.json` 中 `MaxPoolGradWithArgmaxV1` 条目保持一致：

| 参数 | 参数含义 | 数据类型 | 支持数据类型 | 数据格式 | 约束 | 形状 |
|------|----------|----------|--------------|----------|------|------|
| x | 正向输入 | tensor | float16 / float32 / float32 | NC1HWC0 / NC1HWC0 / NCHW | 无 | all（动态 shape 支持） |
| grad | 上游梯度 | tensor | float16 / float32 / float32 | NC1HWC0 / NC1HWC0 / NCHW | 与 x 一致 | all |
| argmax | 最大值索引 | tensor | uint16 / uint16 / int32 | NC1HWC0 / NC1HWC0 / NCHW | 与 x 一一对应 | all |
| y | 输出梯度 | tensor | float16 / float32 / float32 | NC1HWC0 / NC1HWC0 / NCHW | 与 x 一致 | all |
| ksize | 窗口大小 | listInt | all | — | 必选，长度 1 或 3 | — |
| strides | 步长 | listInt | all | — | 必选，长度 0/1/3 | — |
| pads | 填充 | listInt | all | — | 必选，长度 1 或 3 | — |
| dtype | argmax 类型选择 | int | all | — | 可选，默认 3（int32） | — |
| dilation | 窗口内步幅 | listInt | all | — | 可选，默认 [1,1,1,1] | — |
| ceil_mode | 输出取整方式 | bool | all | — | 可选，默认 false | — |

约束与特性（来自算子信息库）：`dynamicCompileStatic=true`、`dynamicRankSupport=true`、
`dynamicShapeSupport=true`、`coreType=AiCore`、`aclnnSupport=support_aclnn`。

> 说明：算子信息库共登记 3 组 dtype/format 组合，前两组为 uint16 + NC1HWC0 的**位平面 mask 语义**，
> 第三组为 int32 + NCHW 的**argmax 下标语义**。本任务书要求 `indices` 为 INT8 容器、NCHW，
> 对应第三组语义；910B 上 FLOAT16/BFLOAT16 由 aclnn 层 `Cast` 归一到 FLOAT32 后调用该算子。

##### 1.2.2.2 标杆算子实现描述

标杆 TBE 实现按 dtype 分两条路径，本任务书口径（NCHW + int32 argmax）走
`max_pool_grad_with_argmax_v1_dsl.py` → `tbe/dsl/compute/max_pool_grad_with_argmax.py`
的 `max_pool_grad_with_argmax_nchw`，该函数以 TVM 计算图描述、由 `tbe.auto_schedule`
自动调度生成 Ascend C kernel。其数据流与源码逻辑逐条对应如下：

| 步骤 | 源码节点 | 语义 |
|------|----------|------|
| ① 输入下标 | `x_index` | 生成输入平面线性下标 `x_index(i_hi,i_wi) = i_hi * wi + i_wi`，dtype 与 argmax 一致（int32） |
| ② padding | `x_p` | 对 `x_index` 做 `pads` 填充，pad 区域填 `MIN_VALUES[argmax.dtype]`（保证不会与合法 argmax 相等） |
| ③ img2col | `x_img2col` | 展开为 `[kh*kw, nc, ho, wo]`：`x_p[i_nc, i_ho*sh + i_khw//kw, i_wo*sw + i_khw%kw]` |
| ④ 掩码 | `mask0` | 向量比较 `EQ(argmax[nc,ho,wo], x_img2col[khw,nc,ho,wo])` → `uint1` 掩码 |
| ⑤ 选梯度 | `dy_sel` | 向量 select：`mask0 ? grad[nc,ho,wo] : 0` |
| ⑥ col2img 归约 | `dp` | 对 padding 后的图做条件 reduce_sum：`dp(nc,hp,wp) = Σ dy_sel[khw,nc,ho,wo]`，条件 `hp == ho*sh + kh && wp == wo*sw + kw`，即把输出位置的梯度**散射回**输入位置并累加 |
| ⑦ 去 padding | `dx` | `dx(nc,hi,wi) = dp(nc, hi+pt, wi+pl)` |

算法要点：

1. **全程向量化**：掩码生成用 `elewise_binary_vcmpv_eq`（向量比较），选梯度用
   `elewise_multiple_sel`（向量 select），回填用 `reduce_sum`（向量条件归约），
   不存在逐元素标量访存。
2. **重叠窗口的累加**由 ⑥ 的条件归约天然完成：`stride < kernel` 时同一个 `(hp,wp)`
   会被多个 `(ho,wo)` 命中，归约即累加，因此无需原子操作、结果顺序固定。
3. **padding 位置被丢弃**由 ② + ④ 保证：pad 位置填入 `MIN_VALUES`，永远不等于任何合法
   argmax，因此 ④ 处掩码恒为 0，等价于"命中 padding 区的梯度被丢弃"。
4. **非 FLOAT32 路径**（`max_pool_grad_with_argmaxv2.py` 的 `MaxpoolGrad` 类）走
   NC1HWC0 + uint16 位平面 mask 语义：先 `vector_dup` 清零 col2img（float32）缓冲，
   `_clean_mask` 清理 mask 缓冲，`_vsel_grad_col` 用 `mov_tensor_to_cmpmask` + `vsel`
   选出命中的梯度，`_mov_func` / `_move_func_block` 做 col2img 累加，并按
   `kh - stride_h` 保留尾部重叠行的临时缓冲以处理窗口重叠；最后 `_vconv` 转回目标 dtype。
   该路径按 `_not_tilling` / `_tilling_ho` / `_tilling_ho_wo` 等分支做 H/W 切分。

##### 1.2.2.3 标杆算子实现流程图

```
┌──────────────────────────────────────────────────────────────────────────┐
│ 输入：grad (nc,ho,wo)、argmax (nc,ho,wo) int32、ksize、strides、pads      │
└───────────────────────────────┬──────────────────────────────────────────┘
                                ▼
                  x_index = i_hi*wi + i_wi（输入线性下标，int32）
                                ▼
                  x_p = pad(x_index, pads, MIN_VALUE)      ← pad 位置填最小值
                                ▼
                  x_img2col[khw, nc, ho, wo]
                    = x_p[nc, ho*sh + khw//kw, wo*sw + khw%kw]
                                ▼
                  mask0 = EQ(argmax[nc,ho,wo], x_img2col[khw,nc,ho,wo])  ← 向量比较
                                ▼
                  dy_sel = mask0 ? grad[nc,ho,wo] : 0                    ← 向量 select
                                ▼
                  dp(nc,hp,wp) = Σ dy_sel[khw,nc,ho,wo]
                    条件：hp == ho*sh + kh  且  wp == wo*sw + kw          ← 条件归约(累加)
                                ▼
                  dx(nc,hi,wi) = dp(nc, hi+pt, wi+pl)                    ← 去 padding
                                ▼
                          输出 dx (nc,hi,wi)
```

---

## 二、需求分析

### 2.1 外部组件依赖

- Ascend C 算子开发工具链：CANN 9.0.0 / CANN 9.1.0（含 `ccec_compiler`、`tikicpulib`）。
- ops-nn 开源仓 `experimental/pooling/` 工程框架（`op_host` + `op_kernel` + `op_graph`）。
- 精度 golden 由任务随附的 `max_pool2d_with_mask_backward_golden.py` 提供（仅依赖 numpy /
  ml_dtypes），无其他三方库依赖。
- 正向算子 `aclnnMaxPool2dWithMask` 用于产出 `indices`（自测数据链路与正反向配套验证）。

### 2.2 内部适配模块

| 模块 | 适配内容 |
|------|----------|
| `op_graph` 算子原型 | 新增 `MaxPool2dWithMaskBackward` 原型与 dtype/format 注册，`gradInput` 跟随 `self` |
| `op_host` 推导 | `InferShape`（`gradInput` = `self` shape）、`InferDataType`（`gradInput` dtype = `self` dtype） |
| `op_host` tiling | 正反向一致性校验、UB 预算与 tile 切分、输入行归属分核、TilingData 下发 |
| `op_kernel` | argmax 前缀解码、命中回填与重叠累加、未命中位置置 0 |
| 正向算子 `MaxPool2dWithMask` | 复用其 `indices` 编码口径，保证正反向配套（不修改正向实现） |

### 2.3 需求模块设计

#### 2.3.1 Ascend C 算子原型

与 `aclnnMaxPool2dWithMaskBackwardGetWorkspaceSize` 的官方签名严格对齐（参数顺序、dtype、format）：

```cpp
aclnnStatus aclnnMaxPool2dWithMaskBackwardGetWorkspaceSize(
  const aclTensor   *gradOutput,   // FLOAT32 / FLOAT16 / BFLOAT16, NCHW, rank 4
  const aclTensor   *self,         // FLOAT32 / FLOAT16 / BFLOAT16, NCHW, rank 4
  const aclTensor   *indices,      // INT8, NCHW, rank 4
  const aclIntArray *kernelSize,   // 长度 1 或 2
  const aclIntArray *stride,       // 长度 0、1 或 2（0 表示等于 kernelSize）
  const aclIntArray *padding,      // 长度 1 或 2，0 <= padding <= kernelSize
  const aclIntArray *dilation,     // 长度 1 或 2，值仅支持 1
  bool               ceilMode,     // 默认 false
  const aclTensor   *gradInput,    // FLOAT32 / FLOAT16 / BFLOAT16, NCHW, rank 4
  uint64_t          *workspaceSize,
  aclOpExecutor    **executor);
```

`indices` 的 shape 契约（与正向完全一致）：

$$
[N,\ C,\ k_h \cdot k_w,\ (\lceil H_{out} \cdot W_{out} / 16 \rceil + 1) \times 2 \times 16]
$$

其前 $N \cdot C \cdot H_{out} \cdot W_{out} \times 4$ 字节为一维连续、小端序的 INT32 argmax 数组，
第 $l = h \cdot W_{out} + w$ 个 INT32 给出正向最大值在该输入平面内的线性下标
$idx = h_s \cdot W_{in} + w_s$。该口径与算子信息库中 `MaxPoolGradWithArgmaxV1` 的
`int32 + NCHW` 组合一致，也与内置正向算子 `aclnnMaxPool2dWithMask` 的实际输出逐字节一致。

#### 2.3.2 Ascend C 算子相关约束

与标杆算子相比的能力对齐情况：

| 标杆能力（算子信息库） | Ascend C 版本 | 说明 |
|------------------------|---------------|------|
| float32 + NCHW + int32 argmax | ✅ 支持 | 本任务书口径，逐一对齐 |
| float16 + NCHW + int32 argmax | ✅ 支持 | 搬到 FP32 域累加后 `CAST_NONE` 回落 |
| bfloat16 + NCHW + int32 argmax | ✅ 支持 | 仅 Atlas A2；FP32 域累加后 `CAST_RINT` 回落 |
| float16/float32 + NC1HWC0 + uint16 mask | ❌ 不支持 | 任务书要求 NCHW，且 `indices` 仅 INT8；不实现 NC1HWC0 位平面 mask 语义 |
| ksize 长度 1 或 3（3D） | ⚠️ 仅 2D | 本算子只对齐 2D（ksize 长度 1 或 2），不实现 3D 路径 |
| 动态 shape / 动态 rank | ⚠️ 部分 | rank 固定为 4；shape 在 tiling 阶段读取，支持动态 shape |
| `data_format` 属性（NCDHW/NDHWC） | ❌ 不支持 | 本算子以 NCHW/ND 验收，不提供 `data_format` 属性 |
| 确定性开关 | ✅ 支持 | 默认非确定性语义下本实现仍为确定性结果（见 3.2.2.1） |

功能缺失项均为**任务书未要求适配**的部分（3D、NC1HWC0 位平面 mask、`data_format` 属性），
其余能力与标杆对齐。

---

## 三、需求详细设计

### 3.1 调用方式

采用 **ACLNN 两段式接口**调用：先调用
`aclnnMaxPool2dWithMaskBackwardGetWorkspaceSize` 完成入参校验、shape 推导与 workspace 计算，
再调用 `aclnnMaxPool2dWithMaskBackward` 在指定 stream 上执行。调用示例见
`experimental/pooling/max_pool2d_with_mask_backward/examples/`。

算子按 ops-nn 规范实现 `op_host`（`_def.cpp` / `_infershape.cpp` / `_tiling.cpp`）与
`op_kernel`，由 `op_graph` 原型注册；除 ACLNN 外不提供 Kernel 直调或 PyTorch 原生接口。

### 3.2 需求总体设计

#### 3.2.1 host 侧设计

##### 3.2.1.1 分核策略

以 **输入行（`(n,c)` 平面内的 H 行）为所有权单位**分核：

```
totalTiles = N * C * ceil(H_in / tileRows)
usedCores  = min(aivCoreNum, totalTiles)
coreTiles  = totalTiles / usedCores
extraCores = totalTiles % usedCores      // 前 extraCores 个核各多处理 1 个 tile
```

每个 tile 独占某个 `(n,c)` 上连续 `[ihStart, ihStart + rows)` 行输入，核内按 tile 顺序串行处理。
选择该策略的原因：

1. 每个输出位置经 argmax 只映射到**唯一**一个输入位置，而 tile 拥有的输入行区间互不相交，
   因此每个输入位置只会被唯一一个 tile 写入 → **无需 `SetAtomicAdd`、无跨核竞争**；
2. 每个 tile 写回的是自己完整的、已清零的行区间，未命中位置天然为 0 →
   `gradInput` 全量覆盖写，不需要额外的清零 kernel 或 workspace；
3. 核内累加顺序固定（tile 升序、chunk 升序、chunk 内元素升序），**结果确定性**。

并行度不足（`N*C` 远小于 AI Core 数）时，在 `(n,c)` 内部继续按输入行切分：

```
desiredPerNc = max(1, ceil(TILES_PER_CORE_TARGET * aivCoreNum / (N*C)))
tileRows     = min(tileRows_ub, ceil(H_in / desiredPerNc))
```

##### 3.2.1.2 数据分块和内存优化策略

**① tileRows（输入行数）**：先由「累加缓冲 + 输出缓冲不超过可用 UB 的一半」定出上限，再由
并行度约束收敛：

$$
tileRows_{ub} = \left\lfloor \frac{safeUb / 2}{W_{in} \times (\text{sizeof}(float) + [\text{T} \ne float]\cdot \text{sizeof}(T))} \right\rfloor
$$

$$
tileRows = \min\left(tileRows_{ub},\ \lceil H_{in} / desiredPerNc \rceil,\ H_{in}\right)
$$

其中：

$$
safeUb = UB_{size} - 16KB \quad (\text{预留对齐与 tiling 元数据余量})
$$

**② chunkElems（单次搬入的输出元素数上限）**：由剩余 UB 与单元素搬运开销反推，
并 clamp 到 `[1, min(H_out*W_out, accLen)]`：

$$
chunkElems = \left\lfloor \frac{safeUb - accBytes - outBytes - n_{chunkBuf} \times 512}{elemBytes} \right\rfloor
$$

$$
elemBytes = \text{sizeof}(T) + 4 + [\text{T} \ne float]\times 4,\qquad
n_{chunkBuf} = \begin{cases} 2 & T = float \\ 3 & T \ne float\end{cases}
$$

**③ Local Memory（UB）Buffer 规划**（host 与 kernel 两侧常量口径必须一致）：

| Buffer | 类型 | 容量口径 | 用途 |
|--------|------|----------|------|
| `accBuf_` | `TBuf<VECCALC>` | `align32((accLen + 256) * 4)` | FP32 累加缓冲；尾部 256 元素冗余吸收向量指令按 repeat 对齐的越界写，并额外提供 trash 槽 |
| `outBuf_` | `TBuf<VECCALC>` | `align32((accLen + 256) * sizeof(T))`，仅非 FP32 | `Cast` 回原 dtype 的输出缓冲 |
| `gradBuf_` | `TBuf<VECCALC>` | `align32(chunkElems * sizeof(T) + 512)` | `gradOutput` 片段搬入 |
| `idxBuf_` | `TBuf<VECCALC>` | `align32(chunkElems * 4 + 512)` | int32 argmax 前缀片段搬入 |
| `f32Buf_` | `TBuf<VECCALC>` | `align32(chunkElems * 4 + 512)`，仅非 FP32 | `gradOutput` 的 FP32 副本 |

其中 `accLen = tileRows * W_in`。UB 大小与核数由 `GetCoreMemSize(UB)` /
`GetCoreNumAiv()` 运行期获取，不硬编码，同一源码适配 Atlas A2 与 Atlas 300V Pro。

##### 3.2.1.3 tilingKey 规划策略

本算子的分支差异**全部落在 TilingData 字段**上，不需要按 tilingKey 编译多份 kernel：

| 分支依据 | TilingData 字段 | 取值 |
|----------|------------------|------|
| dtype | 由编译期 `DTYPE_GRADOUTPUT` 宏区分 | float / half / bfloat16_t，每 dtype 一份 binary |
| 窗口是否重叠（决定回填是否需要读-改-写） | `overlap` | 1：`sH < kH` 或 `sW < kW`；0：非重叠，可直接覆盖写 |
| 切分粒度 | `tileRows` / `tilesPerNc` / `totalTiles` / `chunkElems` / `accLen` | 由 3.2.1.2 公式计算 |
| 分核 | `usedCores` / `coreTiles` / `extraCores` | 由 3.2.1.1 公式计算 |

`ceilMode` 只影响 Host 侧 `H_out/W_out` 推导，不下发到 kernel（kernel 只按 `H_out/W_out`
遍历），因此也不需要单独的 tilingKey。

#### 3.2.2 kernel 侧设计

##### 3.2.2.1 kernel 侧实现描述

kernel 入口读取 `MaxPool2dWithMaskBackwardTilingData`，分为 `Init` 与 `Process` 两阶段，
`Process` 由 `CopyIn → Compute → CopyOut` 三段构成（CopyIn 与 Compute 在每个 chunk 内交错）。
与标杆算子保持一致的流程为「解码 argmax → 选定梯度 → 回填累加 → 写出」。

单个 tile 的处理流程：

```
① 清零：Duplicate(acc, 0, tileRows * W_in)
② 确定候选输出行范围（避免全量扫描）：
     ohMin = floor((ihStart + pH - kH) / sH) + 1
     ohMax = ceil((ihStart + rows + pH) / sH) - 1        // 再按 [0, H_out) 截断
   推导：窗口行区间 [oh*sH - pH, oh*sH - pH + kH) 需与 tile 行区间 [ihStart, ihStart+rows) 相交
   →  oh > (ihStart + pH - kH)/sH  且  oh < (ihStart + rows + pH)/sH
③ 对候选范围内的输出行分 chunk：
     CopyIn  int32 argmax 前缀片段  -> idxUb
     CopyIn  gradOutput 片段        -> gradUb
     Cast    gradUb -> FP32（非 FP32 输入）
     回填：off = idx[e] - ihStart*W_in
           if 0 <= off < rows*W_in:  acc[off] += grad[e]   （重叠）
           else:                     acc[off]  = grad[e]   （非重叠）
④ Cast：acc(F32) -> T（非 FP32 输入）
⑤ CopyOut：acc -> gradInput[nc, ihStart : ihStart+rows, :]（整体覆盖写）
```

关键设计点：

1. **解码口径**：直接按小端 INT32 argmax 前缀读取 `indices`，`idx` 即输入平面线性下标；
   用 `off = idx - ihStart * W_in` 与 `[0, rows * W_in)` 比较即可完成"落在本 tile 内"的判定，
   **无需除法/取模**（`ih = idx / W_in`、`iw = idx % W_in`）。
2. **丢失 padding 区梯度**：argmax 恒指向窗口内的合法输入位置（正向 pad 位置填 `-inf`），
   因此解码结果必然落在 `[0, H_in*W_in)` 内；落在本 tile 之外的直接丢弃，与标杆的
   `MIN_VALUES` + 掩码语义等价。
3. **重叠累加**：`stride < kernelSize` 时同一输入位置可能被多个输出窗口命中，
   此处做读-改-写累加；非重叠场景退化为直接覆盖写。
4. **确定性**：输入行归属分核 + 核内固定顺序累加 ⇒ 每个 `gradInput` 位置只被写一次，
   结果 bit-wise 确定。`aclrtCtxSetSysParamOpt` 开启确定性时行为不变（本就确定）。
5. **精度口径**：非 FP32 输入统一搬到 FP32 域累加，最后一次性 Cast 回原 dtype；
   FP32→FP16 用 `CAST_NONE`，FP32→BF16 用 `CAST_RINT`，与标杆的
   "FP32 累加 + 单次转回" 口径一致。
6. **标量回填的性能优化**（对标杆的向量化路径做针对性优化）：
   - 内层全部使用 int32 运算——AI Core 标量单元对 int64 需要软件模拟；
   - **无分支化**：越界目标被重定向到累加缓冲尾部的 trash 槽并写入/累加 0，
     避免数据相关分支造成的流水线冲刷；
   - **8 路展开 + 批量读取 idx**，为 UB 标量访问提供指令级并行。

##### 3.2.2.2 Ascend C 实现流程图

```
┌────────────────────────────────────────────────────────────────────────────┐
│ Kernel 入口：读取 TilingData（ncDim/hiDim/wiDim/hoDim/woDim/kH/sH/pH/       │
│              tileRows/tilesPerNc/totalTiles/accLen/chunkElems/overlap …）    │
└───────────────────────────────────┬────────────────────────────────────────┘
                                    ▼
                    bid = GetBlockIdx(); 按分核取本核 tile 区间
                                    ▼
                    ┌─────────────── 对每个 tile ───────────────┐
                    ▼                                          │
      nc = tile / tilesPerNc;  ihStart = (tile % tilesPerNc)*tileRows
                    ▼
      Duplicate(acc, 0, tileRows * W_in)                     ← ① 清零
                    ▼
      ohMin/ohMax = 与本 tile 输入行区间相交的输出行范围        ← ② 去冗余扫描
                    ▼
      ┌──────────── 对候选输出行分 chunk ────────────┐
      │  CopyIn idx 片段 (int32 argmax) -> idxUb     │
      │  CopyIn gradOutput 片段 -> gradUb            │
      │  Cast gradUb -> FP32（非 FP32）              │
      │  for e in chunk:                             │
      │      off = idx[e] - ihStart*W_in             │
      │      0 <= off < rows*W_in ?                  │
      │          acc[off] += grad[e]   (重叠)        │
      │          acc[off]  = grad[e]   (非重叠)      │
      └──────────────────────────────────────────────┘   ← ③ 回填累加
                    ▼
      Cast acc(FP32) -> T（非 FP32）                          ← ④ 单次转回
                    ▼
      DataCopyPad(acc -> gradInput[nc, ihStart:ihStart+rows, :])  ← ⑤ 覆盖写
                    ▼
                    └──────────────── 下一个 tile ────────────────┘
                    ▼
                  结束
```

##### 3.2.2.3 Ascend C 实现流程图与标杆算子流程图存在的差异点和原因

| 差异点 | 标杆算子（TBE / TVM auto_schedule） | Ascend C 实现 | 原因 |
|--------|-------------------------------------|---------------|------|
| argmax 解码方式 | 先生成输入线性下标 `x_index`，再 img2col 展开后与 `argmax` 做向量相等比较（`mask0`） | 直接读取 int32 argmax 得到输入线性下标，用 `off = idx - ihStart*W_in` 做区间判定 | 任务书口径为 int32 下标语义（非位平面 mask），可直接解码，无需 img2col 展开 |
| 选定梯度 | `elewise_multiple_sel` 向量 select（`mask0 ? grad : 0`） | 逐元素 `off` 区间判定后取 `grad[e]`；越界重定向到 trash 槽 | 输入行归属分核下"是否命中本 tile"是一次区间比较，标量判定即可；配合无分支化避免流水线冲刷 |
| 回填方式 | `col2img` 条件归约（`reduce_sum`，条件 `hp == ho*sh+kh && wp == wo*sw+kw`）向量化散射 | 在 UB 累加缓冲上做标量读-改-写，最后整体写回 GM | 输入行归属分核要求 tile 的写集与其他 tile 严格互斥，UB 内累加后可整体覆盖写，无需跨核原子累加 |
| 零初始化 | TBE 非 F32 路径 `vector_dup` 清 col2img；NCHW 路径依赖 `col2img` 归约的稀疏性 | 每个 tile 先在 UB 内 `Duplicate(0)`，再整行区间写回 | 输入行归属要求 gradInput 全量覆盖写（未命中位置必须为 0），UB 内清零后可一次覆盖写出 |
| 窗口重叠处理 | 条件归约天然累加；非 F32 路径另存 `kh - stride_h` 行临时缓冲 | 同一输入位置被多次命中时做读-改-写累加（`overlap` 字段区分） | 两者语义一致；本实现由分核保证不跨核，累加顺序固定 ⇒ 结果确定 |
| 数据格式 | 除 NCHW 外还支持 NC1HWC0（uint16 位平面 mask） | 仅 NCHW/ND | 任务书只要求 NCHW 与 INT8 `indices` |
| 切分维度 | 按 N/C1 或 H/W 切分（`_not_tilling` / `_tilling_ho` / `_tilling_ho_wo`） | 按 `(n,c)` 平面内 H 行切分，`N*C` 不足时继续在平面内切行 | 保证 tile 写集互斥（同一 `(n,c)` 内按行切分写集天然不重叠），获得无原子、确定性、全量覆盖写三个特性 |
| kernel 生成方式 | TVM 计算图 + `tbe.auto_schedule` 自动调度 | 手写 Ascend C（`PipeBarrier` 显式同步 + 手工 tile/分核） | Ascend C 手工实现需自行承担 UB 预算、同步与分核规划 |

### 3.3 支持硬件

| 芯片版本 | 支持 |
|----------|:---:|
| Atlas A2 训练系列产品（Atlas 800T A2，910B） | ✅ |
| Atlas 推理系列产品（Atlas 300V Pro） | ✅ |
| Atlas 训练系列产品 | ✅（FLOAT32 输入内部转 FLOAT16 计算，存在精度损失） |
| Ascend 950PR / 950DT | ❌（内部映射请使用 `aclnnMaxPool2dWithIndicesBackward`） |

与《算子任务书》要求支持的硬件（Atlas 800T A2、Atlas 300V Pro）保持一致；
BFLOAT16 仅 Atlas A2 支持。

### 3.4 算子约束限制

1. **确定性计算**：`aclnnMaxPool2dWithMaskBackward` 默认非确定性实现，支持通过
   `aclrtCtxSetSysParamOpt` 开启确定性；本实现两种设置下结果均 bit-wise 确定。
2. **NaN/-Inf 限制**：输入数据暂不支持 NaN、-Inf。
3. **dtype**：`gradOutput` / `self` / `gradInput` 支持 FLOAT32、FLOAT16、BFLOAT16
   （BFLOAT16 仅 Atlas A2），三者 dtype 必须一致；`indices` 仅支持 INT8。
4. **dilation 限制**：dilation 值仅支持 1。
5. **padding 约束**：padding 值必须 ≥ 0 且 ≤ kernelSize。
6. **ceilMode 限制**：`ceilMode=True` 时，若 `(H_out-1)*stride[0] >= H_in + padding[0]`
   （W 维同理），结果未定义。
7. **indices 依赖**：`indices` 必须由正向 MaxPool2dWithMask 产出
   （每个输出位置恰好一个 argmax）；多 argmax 等于同一输出位置等非法输入行为未定义。
8. **rank/format**：rank 固定为 4，format NCHW/ND；不支持 3D（ksize 长度 3）与
   NC1HWC0 位平面 mask 语义。
9. **kernelSize/stride/padding 长度**：kernelSize 长度仅支持 1、2；stride 长度仅支持 0、1、2；
   padding 长度仅支持 1、2。

---

## 四、特性交叉分析

| 特性 | 分析 |
|------|------|
| 精度与性能 | 非 FP32 统一在 FP32 域累加、单次 Cast 回落，精度与标杆同口径；性能上小 shape 由更细的分核粒度获益，大 shape 的随机回填为主要优化点 |
| 正反向配套 | `indices` 的 shape、语义、越界丢弃规则与正向严格一致；`gradOutput` 对齐正向 `out`、`gradInput` 对齐正向 `self`，正反向可互换配套 |
| 确定性 × 分核 | 输入行归属分核下 tile 写集互斥 ⇒ 天然确定性，且不需要 `SetAtomicAdd`；若后续为提升大 shape 性能改为按输出位置分核，则需引入 `SetAtomicAdd` 并由确定性开关切换路径 |
| 重叠 × 非重叠 | `overlap` 字段区分两种回填语义（累加 / 覆盖写），分核策略不变，两条路径的 0/非 0 位置集合一致 |
| dtype × 硬件 | BFLOAT16 仅 A2；Atlas 训练系列产品 FLOAT32 内部转 FLOAT16；tiling 全部由运行期平台信息计算，同一源码适配多硬件 |
| 与正向算子复用 | 共用 `indices` 编码契约与 shape 推导公式，正向文档与反向文档逐条对齐，避免正反向口径漂移 |
| UB 预算 × tile 粒度 | `tileRows` 与 `chunkElems` 由 UB 预算反推，300V Pro 的 UB 更小 ⇒ tile 更小、分核更细，属预期适配行为 |

---

## 五、可维可测分析

### 5.1 精度标准 / 性能标准

| 验收标准 | 描述 | 标准来源 |
|----------|------|----------|
| 精度标准 | 满足生态算子开源精度标准的混合容差（atol/rtol）；以标杆算子的 `gradInput` 为黄金参考逐元素比对，0/非 0 位置集合完全一致 | 生态算子开源精度标准（`opbase/docs/zh/ops_precision_standard/experimental_standard.md`） |
| 性能标准（A2） | 算子整体性能与现有实现持平，不低于原 TBE 算子的 95% | 任务书性能要求 1 |
| 小 shape 口径（A2） | 100 µs 以下场景若超过 TBE 耗时 30% 以上，须提供性能仿真图与分析结论，证明与 TBE 一致或更优 | 任务书性能要求 2 |
| 性能标准（300V Pro） | 验收算子功能与精度；耗时基准参考 Atlas 800T A2（910B3）的 5 倍，部分 shape 超过基准 30% 须提供性能仿真图 | 任务书性能要求 3 |
| 确定性 | 开关开启时多次运行 bit-wise 一致；关闭时差异仅允许出现在浮点累加末位 | 任务书约束说明 1 |
| 模型级精度 | Atlas 300V Pro 上 yolov11 + DOTAv1 的 mAP50 与 Atlas 800T A2 误差 ≤ 0.01 | 任务书模型验证章节 |

> 本节为**验收口径**。本文件为开发前设计文档，不含实测数值；实测结论以开发完成后的自测报告为准。

### 5.2 兼容性分析

**接口兼容**：ACLNN 签名与参数顺序（`gradOutput, self, indices, kernelSize, stride, padding,
dilation, ceilMode, gradInput`）与官方一致，可与现网 TBE 的正向/反向互换配套，
这也是 `indices` 编码必须逐字节与正向对齐的原因。

**算子仓兼容**：新增算子目录落在 `experimental/pooling/`，不改动既有算子文件；
复用 `pooling/` 下既有的公共工具与工程骨架。

**硬件/版本兼容**：tiling 的容量参数（UB 大小、AI Core 数）全部由
`GetCoreMemSize` / `GetCoreNumAiv` 运行期计算，不硬编码；CANN 9.0.0 与 9.1.0 的
接口与 tiling 结构一致，不依赖版本私有接口。

**兼容性判定流程**：

```
┌─────────────────────────────────────────────────────────────┐
│ 输入：gradOutput / self / indices + kernelSize/stride/       │
│       padding/dilation/ceilMode                              │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
              rank == 4 且 dtype 合法？ ── 否 ──► ACLNN_ERR_PARAM_INVALID
                           │ 是
                           ▼
       self 与 gradInput shape 一致？ ── 否 ──► ACLNN_ERR_PARAM_INVALID
                           │ 是
                           ▼
  由 self + 属性推导 (H_out, W_out)；gradOutput shape 一致？
                           ├── 否 ──► ACLNN_ERR_PARAM_INVALID
                           ▼ 是
   indices dtype == INT8 且 shape == [N,C,kH*kW,(⌈Ho*Wo/16⌉+1)*32]？
                           ├── 否 ──► ACLNN_ERR_PARAM_INVALID
                           ▼ 是
   kernelSize 长度 ∈ {1,2} 且元素 > 0？
   stride 长度 ∈ {0,1,2} 且元素 > 0？（长度 0 → 归一化为 kernelSize）
   padding 长度 ∈ {1,2} 且 0 ≤ padding ≤ kernelSize？
   dilation 值 == 1？
                           ├── 否 ──► ACLNN_ERR_PARAM_INVALID
                           ▼ 是
   与内置 aclnnMaxPool2dWithMaskBackward 签名/参数顺序一致 → 可直接替换
                           ▼
                进入 tiling 与 kernel 执行
```

**兼容性边界**（与标杆相比）：

| 场景 | 标杆（TBE） | 本实现 | 兼容性 |
|------|-------------|--------|:---:|
| FLOAT32 / NCHW / int32 argmax | 支持 | 支持 | ✅ |
| FLOAT16、BFLOAT16 / NCHW / int32 argmax | 支持（aclnn 层 Cast 归一） | 支持（FP32 域累加后回落） | ✅ |
| 窗口重叠（stride < kernel） | 支持 | 支持 | ✅ |
| padding 命中被丢弃 | 支持 | 支持 | ✅ |
| ceilMode=True 的边界组合 | aclnn 层可能返回参数错误 | 同口径校验 | ✅ |
| NC1HWC0 + uint16 位平面 mask | 支持 | 不支持 | ➖ 任务书未要求 |
| 3D（ksize 长度 3） | 支持 | 不支持 | ➖ 任务书未要求 |
| `data_format` 属性 | 支持 | 不提供 | ➖ 任务书未要求 |
| 确定性开关 | 支持 | 支持（两种设置下均确定） | ✅ |
