# 需求背景（required）

## 需求来源

本需求来源于社区任务《aclnnUpsampleNearest3d 算子开发任务书》。任务要求在
`ops-cv` 仓库现有 UpsampleNearest3d 实现基础上，为 `aclnnUpsampleNearest3d`
新增 `UINT8` 数据类型支持，适配 Atlas A2 训练系列产品，
并完成算子设计、开发、测试、文档和性能验证。

## 背景介绍

### UpsampleNearest3d算子实现优化

UpsampleNearest3d 对 5D 输入张量执行 D、H、W 三个空间维度的最近邻上采样。输出位置
直接映射到输入位置并复制输入值，不进行线性或三线性插值。本次任务的优化目标是在
保持原有 FLOAT32、FLOAT16、BFLOAT16 等数据类型行为不变的基础上，补齐 UINT8 的
接口、注册、Tiling 和 Kernel 执行链路。

### UpsampleNearest3d算子原实现现状分析

原有 A2 普通 Kernel 已支持 FLOAT16、FLOAT32 和 BFLOAT16 模板，主要使用 Gather 完成
W 方向离散采样；本任务在 Host、算子注册、TilingKey 和 Kernel 模板分发中增加 UINT8 支持。

原有 Gather 路径按至少 2 Byte 的元素粒度生成偏移并执行采样，而 UINT8 元素宽度为
1 Byte，不能直接机械实例化原有 `Gather<T>`。如果直接使用 1 Byte 数据，会造成偏移
含义、UB 布局和尾块处理不匹配。

### UpsampleNearest3d算子功能分析

`outputSize` 用于指定输出 D/H/W；`outputSize` 和 `scalesD/H/W` 均为 aclnn 接口的
必传参数，其中三个 `scales` 值同时大于 0 时作为坐标映射倍率生效，否则根据输入空间
尺寸和 `outputSize` 推导映射比例。输入和输出数据类型、数据格式保持一致。支持
NCDHW、NDHWC 和 ND 格式；
ND 按 NCDHW 语义处理，NDHWC 在 Host 侧转换为 NCDHW 后执行计算，再转换回原格式。

| 参数 | 参数含义 | 数据类型 | 支持数据类型 | 约束 | 形状/格式 |
| --- | --- | --- | --- | --- | --- |
| self | 输入张量 | tensor | FLOAT32、FLOAT16、BFLOAT16、UINT8 | C/D/H/W 大于 0 | 5D，NCDHW/NDHWC/ND |
| outputSize | 输出 D/H/W | array | INT64 | 长度为 3，各元素大于 0 | `[D_out,H_out,W_out]` |
| scalesD/H/W | D/H/W 缩放倍率 | scalar | double | 必须传入；三个值同时大于 0 时生效，否则使用 `outputSize` 推导 | 浮点数 |
| out | 输出张量 | tensor | 与 self 一致 | N、C 轴与输入一致 | 5D，格式与输入一致 |

# 需求分析（required）

## 需求描述

使用 Ascend C 工程方式实现 `aclnnUpsampleNearest3d` 的 UINT8 支持，保持原有两段式
aclnn 接口、属性名称、shape 推导规则和已有数据类型的计算语义不变。最终代码放入
`experimental/image/upsample_nearest3d/`。

## 需求拆解

1. 在 aclnn 参数校验、Graph 原型、OpDef 和平台配置中加入 UINT8。
2. 为 UINT8 增加独立 TilingKey 和 Kernel 模板分发。
3. 解决 UINT8 不能直接使用原 Gather 元素粒度的问题。
4. 覆盖 W 非对齐尾块、小 W、多 batch、混合缩放和超大 stride 等合法输入。
5. 保证 UINT8 输出与 CPU 最近邻参考逐元素一致。
6. 保证同一 shape 下 UINT8 相对 FLOAT16 的性能劣化不超过 5%。
7. 保持原 FLOAT16、FLOAT32、BFLOAT16 路径兼容，并使用 `--experimental` 构建。

# 详细设计（required）

## 算子分析

### 数学公式

以 NCDHW 为例，输入和输出 shape 分别为 `[N,C,D_in,H_in,W_in]` 和
`[N,C,D_out,H_out,W_out]`。指定 `outputSize` 时：

```text
scaleD = D_out / D_in
scaleH = H_out / H_in
scaleW = W_out / W_in
```

指定正 `scales` 时直接使用三个缩放系数。对任意输出坐标：

```text
id = min(floor(d_out / scaleD), D_in - 1)
ih = min(floor(h_out / scaleH), H_in - 1)
iw = min(floor(w_out / scaleW), W_in - 1)
out[n,c,d_out,h_out,w_out] = self[n,c,id,ih,iw]
```

### 支持数据类型

本次 A2 验证范围内，输入和输出支持 FLOAT32、FLOAT16、BFLOAT16 和 UINT8，且两者
数据类型必须一致。A2 AICore 路径分别使用对应模板，UINT8 使用独立模板分支。

### 支持形状

支持合法的 5D ND、NCDHW 和 NDHWC 输入输出，支持 N/C/D/H/W 为 1、非整数倍缩放、
上采样、下采样和混合缩放。输入输出 N、C 轴必须一致，输入输出空间维度必须大于 0，
各维不超过 `INT32_MAX`，Tensor 元素数量满足底层接口约束。非连续 Tensor 由 Host
侧 `Contiguous` 和 `ViewCopy` 处理。

## 算子实现

### 实现方案

执行流程为：参数校验、输入连续化、必要的 NDHWC 转换、L0 算子选择、Host Tiling、
Kernel 执行以及输出格式恢复。A2 使用普通 Tiling 和 AICore Kernel。

#### 3.2.1 host侧设计：

Host 侧校验输入输出 Tensor、dtype、rank、outputSize 和 shape，根据输入格式完成必要的
连续化与格式转换，并根据 scales 确定坐标映射比例。普通路径为 UINT8 选择独立模板
key，并沿用原有 D/H/W 切片和分核策略。

##### 1. 分核策略：

根据 D/H/W 切片数量和平台 AIV 核数平均分配完整切片。不能整除时，将余数分给前面的
核；尾切片继续按 N*C 行拆分。只有实际有任务的核参与执行。D/H 映射到相同输入位置
时复用对应输出范围，减少重复搬运。

##### 2. 数据分块和内存优化策略：

根据 UB 大小、数据类型和 W 方向映射范围确定 `slideSizeW`、每次搬运的输入行数和
缓冲区大小。所有物理搬运空间按 32 Byte 对齐，实际有效长度由 DataCopy 参数控制。
当输入 stride 超过 `uint32_t` 可表示范围时，将 batch 搬运数降为 1，逐块处理，避免
stride 截断。

##### 3. tilingkey规划策略：

原有数据类型继续使用原 TilingKey，UINT8 使用独立的
`UPSAMPLE_NEAREST3D_TPL_UINT8 = 40`，并通过 `GET_TPL_TILING_KEY` 生成 Kernel
模板 key。TilingData 传递输入输出 D/H/W、scale、W 切片、分核、尾块和小 W 路径等
参数。

#### 3.2.2 kernel侧设计：

A2 普通 Kernel 按 TilingKey 分发 FLOAT16、FLOAT32、BFLOAT16 和 UINT8 模板。UINT8
路径在 UB 中先将输入临时转换为 FLOAT16，再使用稳定的 FLOAT16 Gather 完成离散
采样，最后将结果转回 UINT8 写回 GM：

```text
UINT8 -> FLOAT16 -> Gather -> UINT8
```

UINT8 的 `[0,255]` 全部整数都能由 FLOAT16 精确表示，因此桥接不引入数值误差。输入
搬入区域和 FLOAT16 视图使用 true-2N 物理布局，下一次 MTE2 搬入前通过 MTE2/V 和
V/MTE2 硬件事件等待当前 Gather 完整消费，避免覆盖仍在使用的 UB 数据。W 非对齐尾块
使用 DataCopyPad；小 W 和一维视图使用独立批量路径；超大 stride 使用单块降级路径。

UINT8 模板通过编译期分支与其他数据类型隔离，避免为原数据类型生成无关的转换代码。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas 800I/T A2 | √ |

## 算子约束限制

1. 输入和输出 Tensor 必须为 5 维，且 dtype、format 一致。
2. 输入输出 N、C 轴必须一致，C/D/H/W 及输出空间维度必须大于 0。
3. `outputSize` 必须非空、长度为 3 且各元素大于 0；三个 `scales` 同时大于 0 时用于
   坐标映射，否则根据输入空间尺寸和 `outputSize` 推导映射比例。
4. ND 按 NCDHW 语义处理，NDHWC 由 Host 侧转换处理。
5. 输入 C/D/H/W 和输出空间维度必须大于 0，各维不超过 `INT32_MAX`。
6. UINT8 值域为 `[0,255]`，输出必须保持整数值和边界值。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述（不涉及说明原因） | 标准来源 |
| --- | --- | --- |
| 精度标准 | UINT8 输出与 CPU 最近邻参考逐元素完全一致 | 社区任务书 |
| 工具精度 | 满足 AscendOpTest 默认阈值 | 社区任务书 |
| 边界值 | 输入 0、254、255 经计算后保持原值 | UINT8 数据语义 |
| 性能标准 | 同设备、同 shape 下 UINT8 相对 FLOAT16 劣化不超过 5% | 社区任务书 |
| 回归标准 | FLOAT16/FLOAT32/BFLOAT16 原路径功能不退化 | 兼容性要求 |

A2 精度测试包括 100 个 5D ND（按 NCDHW 语义）AscendOpTest case，覆盖 FLOAT16、
BFLOAT16、FLOAT32、UINT8、常规 shape、边界 shape、D/H/W 为 1、W 非对齐、小 W、
多 batch、上采样、下采样和混合缩放；另有一组 NCDHW UINT8 边界值 aclnn 专项测试。
性能测试使用任务书规定的三组 UINT8 shape 与同 shape FLOAT16 对比，并在精度通过后
统计平均耗时。

## 兼容性分析

本任务属于既有算子的 dtype 能力扩展，不改变公开接口签名、属性名称、shape 推导规则
和原有数据类型的数学语义。UINT8 使用独立 TilingKey 和编译期模板分支，原有
FLOAT16、FLOAT32、BFLOAT16 路径不进入 UINT8 bridge。普通
`image/upsample_nearest3d/` 目录不变，待提交代码位于
`experimental/image/upsample_nearest3d/`。
