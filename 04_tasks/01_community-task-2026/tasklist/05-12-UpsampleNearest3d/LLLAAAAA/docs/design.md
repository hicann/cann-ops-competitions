# aclnnUpsampleNearest3d 算子设计文档

## 需求背景（required）

### 需求来源

CANN 社区任务《8月社区任务-UpsampleNearest3d算子开发》。

当前 `aclnnUpsampleNearest3d` 算子在 Atlas A2/A3 硬件上不支持 uint8 数据类型，需要在原有代码基础上进行再开发，使其在 A2/A3 上支持 uint8 数据类型，并完成算子设计、开发、测试全流程工作。

### 背景介绍

#### UpsampleNearest3d 算子实现现状分析

UpsampleNearest3d 算子基于 Ascend C 编程语言开发，位于 `ops-cv` 开源仓 `image/upsample_nearest3d` 目录下。通过对现有实现的功能分析，当前各硬件配置支持的数据类型如下：

| 硬件配置 | 支持数据类型 | 说明 |
| --- | --- | --- |
| ascend910b (A2) | FLOAT32, FLOAT16, BF16 | **不支持 UINT8** |
| ascend910_93 (A3) | FLOAT32, FLOAT16, BF16 | **不支持 UINT8** |
| ascend310p | FLOAT32, FLOAT16 | 不支持 UINT8（非本次目标） |
| ascend950 (Regbase) | FLOAT32, FLOAT16, BF16, UINT8, DOUBLE | **已支持 UINT8** |

当前算子在 950（Regbase）配置下已经支持 UINT8 数据类型，使用的是 `upsample_nearest3d_apt.cpp` 中的 Regbase 代码路径。但在 A2（ascend910b）和 A3（ascend910_93）配置下，算子仅支持 FLOAT32、FLOAT16 和 BF16，不支持 UINT8。

本次开发的核心目标是在 A2/A3 硬件上扩展 UINT8 数据类型支持。

#### 现有代码结构

```
image/upsample_nearest3d/
├── op_host/
│   ├── upsample_nearest3d_def.cpp          # 算子定义（dtype/format注册）
│   ├── upsample_nearest3d_tiling.cpp       # Tiling实现
│   ├── upsample_nearest3d_tiling.h         # Tiling数据结构
│   ├── upsample_nearest3d_tiling_common.h  # Tiling通用逻辑
│   ├── upsample_nearest3d_tiling_arch3.h   # Arch3 Tiling
│   ├── upsample_nearest3d_infershape.cpp   # Infershape实现
│   └── op_api/
│       ├── aclnn_upsample_nearest_3d.h      # ACLNN API头文件
│       ├── upsample_nearest_3d.cpp          # ACLNN API实现
│       └── ...
├── op_kernel/
│   ├── upsample_nearest3d.h                 # Kernel核心实现（模板类）
│   ├── upsample_nearest3d.cpp              # Kernel入口
│   ├── upsample_nearest3d_struct.h         # Kernel数据结构和模板常量
│   ├── upsample_nearest3d_apt.cpp          # Regbase Kernel入口
│   ├── upsample_nearest3d_310p.h           # 310P Kernel
│   └── arch35/                              # Arch35专用头文件
├── docs/
│   └── aclnnUpsampleNearest3d.md           # API文档
├── examples/
│   └── test_aclnn_upsample_nearest3d.cpp   # 调用示例
├── tests/                                   # 测试用例
├── CMakeLists.txt
└── README.md
```

#### 算子功能分析

UpsampleNearest3d 算子对 5D 输入张量 `[N, C, D, H, W]` 执行最近邻 3D 上采样。

**计算公式**：

```
out[n, c, d_out, h_out, w_out] = self[n, c, floor(d_out * scaleD), floor(h_out * scaleH), floor(w_out * scaleW)]
```

其中：
- 当指定 outputSize 时：scaleD = outputSize[0] / D，scaleH = outputSize[1] / H，scaleW = outputSize[2] / W
- 当指定 scales 时：直接使用 scalesD、scalesH、scalesW 作为缩放系数
- D_out = floor(D * scaleD)，H_out = floor(H * scaleH)，W_out = floor(W * scaleW)

**关键特性**：
1. 最近邻插值：输出像素值取输入张量中对应坐标 floor 后的最近邻像素值，不涉及插值权重计算
2. 两种缩放模式：支持 outputSize（指定输出尺寸）或 scales（指定缩放系数），outputSize 优先
3. 数据格式：支持 NCDHW、NDHWC、ND 三种排布格式
4. UINT8 支持：uint8 为无符号 8 位整数，取值范围 [0, 255]

## 需求分析（required）

### 需求描述

在现有 UpsampleNearest3d 算子基础上，于 Atlas A2/A3 硬件上新增 UINT8 数据类型支持，使其支持 FLOAT32、FLOAT16、BF16、DOUBLE、UINT8 共 5 种数据类型。同时修改相应文档，补充 uint8 的支持情况。

### 需求拆解

1. 在 `op_host/upsample_nearest3d_def.cpp` 中为 A2（ascend910b）和 A3（ascend910_93）配置新增 `ge::DT_UINT8` 数据类型注册
2. 在 `op_host/op_api/upsample_nearest_3d.cpp` 的 `AICORE_DTYPE_SUPPORT_LIST` 中新增 `DT_UINT8`
3. 在 `op_host/op_api/aclnn_upsample_nearest_3d.cpp` 的 `DTYPE_SUPPORT_LIST` 中新增 `DT_UINT8`
4. 在 `op_kernel/upsample_nearest3d_struct.h` 中新增 `UPSAMPLE_NEAREST3D_TPL_UINT8` 模板常量
5. 在 `op_kernel/upsample_nearest3d.h` 的 `UpsampleNearest3dKernelImpl` 函数中新增 `uint8_t` 模板实例化分支
6. 在 `op_host/upsample_nearest3d_tiling.cpp` 的 `GetTilingKey()` 函数中新增 UINT8 的 tiling key 映射
7. 确保算子泛化功能满足各类合法输入场景
8. UINT8 性能与 FP16 相比劣化不超过 5%
9. 更新 README.md 和 API 文档，补充 uint8 支持说明

## 详细设计（required）

### 算子分析

#### 数学公式

```
out[n, c, d_out, h_out, w_out] = self[n, c, floor(d_out * scaleD), floor(h_out * scaleH), floor(w_out * scaleW)]
```

#### 支持数据类型

| 参数 | 现有支持（A2/A3） | 新增支持 | 备注 |
| --- | --- | --- | --- |
| self | FLOAT32, FLOAT16, BF16 | **UINT8** | 输入张量 |
| out | FLOAT32, FLOAT16, BF16 | **UINT8** | 输出张量，与 self 同 dtype |

#### 支持形状

5D 张量 `[N, C, D, H, W]`，所有维度取值 ≤ 2^31-1，C/D/H/W 维 size > 0，不支持空 Tensor。

### 算子实现

#### 实现方案

##### 3.2.0 整体执行流程

```
Host侧:
  1. aclnnUpsampleNearest3dGetWorkspaceSize → 入参校验 + Tiling推导
  2. aclnnUpsampleNearest3d → 下发Kernel任务

Device侧 (Kernel):
  1. ParseTilingData → 解析Tiling参数
  2. 根据isView1DAndSmallW选择计算路径:
     a. ComputeView1DSmallW: 1D视图+小W优化路径
     b. 标准路径: GatherData → GetRangeD/H/W → CopyIn → Gather → CopyOut
  3. 通过Gather指令实现最近邻索引取值
```

##### 3.2.1 Host侧设计

**算子定义修改**（`upsample_nearest3d_def.cpp`）：

在 A2（ascend910b）和 A3（ascend910_93）配置的 Input/Output `DataType` 列表中新增 `ge::DT_UINT8`：

```cpp
// 修改前（A2/A3配置）:
this->Input("x")
    .ParamType(REQUIRED)
    .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
    .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
    .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

this->Output("y")
    .ParamType(REQUIRED)
    .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
    .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
    .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

// 修改后（A2/A3配置）:
this->Input("x")
    .ParamType(REQUIRED)
    .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_UINT8})
    .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
    .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

this->Output("y")
    .ParamType(REQUIRED)
    .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_UINT8})
    .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
    .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
```

**Tiling Key 修改**（`upsample_nearest3d_tiling.cpp`）：

在 `GetTilingKey()` 函数中新增 UINT8 的 tiling key 映射：

```cpp
// 新增 UINT8 分支:
} else if (dtype_x == ge::DataType::DT_UINT8) {
    D_T_X = static_cast<uint32_t>(UPSAMPLE_NEAREST3D_TPL_UINT8);
    D_T_Y = static_cast<uint32_t>(UPSAMPLE_NEAREST3D_TPL_UINT8);
}
```

**Tiling策略说明**：

Tiling 逻辑（`upsample_nearest3d_tiling_common.h` 中的 `UpsampleNearest3dCommonTiling`）是数据类型无关的，仅依赖 shape 和 scale 参数计算分块策略。因此 UINT8 的 Tiling 逻辑与现有 FP16/FP32/BF16 完全一致，无需额外修改 Tiling 通用逻辑。

UINT8 的 `sizeof(T)` 为 1 字节，相比 FP16 的 2 字节更小，在数据搬运（DataCopyPad）和 Gather 操作中，单次可处理的数据量更大，不会引入额外的内存压力。

##### 3.2.2 Kernel侧设计

**模板常量定义**（`upsample_nearest3d_struct.h`）：

新增 UINT8 模板常量：

```cpp
// 新增:
#define UPSAMPLE_NEAREST3D_TPL_UINT8 40
```

并在 `ASCENDC_TPL_ARGS_DECL` 和 `ASCENDC_TPL_SEL` 中注册 UINT8 模板参数。

**Kernel 实例化**（`upsample_nearest3d.h`）：

在 `UpsampleNearest3dKernelImpl` 函数中新增 `uint8_t` 模板实例化分支：

```cpp
// 新增 UINT8 分支:
} else if constexpr (D_T_X == UPSAMPLE_NEAREST3D_TPL_UINT8 && D_T_Y == UPSAMPLE_NEAREST3D_TPL_UINT8) {
    UpsampleNearest3d::UpsampleNearest3dND<uint8_t> op;
    op.Init(x, y, isNearestExact, userWS, tilingData);
    op.Process();
}
```

**UINT8 数据类型适配分析**：

Kernel 核心类 `UpsampleNearest3dND<T>` 是模板类，通过模板参数 `T` 泛化数据类型。UINT8 的适配主要关注以下几点：

1. **Gather 操作兼容性**：算子核心计算使用 `Gather` 指令实现最近邻索引取值。`Gather` 指令支持 `uint8_t` 类型（`T = uint8_t`），可直接用于 UINT8 数据的索引取值，无需额外类型转换。

2. **numPerRep 计算**：在 `ComputeView1DSmallW` 路径中，`numPerRep` 根据 `sizeof(T)` 计算：
   - 当前逻辑：`if constexpr (std::is_same<T, float>::value) numPerRep = 64; else numPerRep = 128;`
   - UINT8 走 else 分支，`numPerRep = 128`，与 FP16 一致。由于 UINT8 字节更小，实际每 repeat 可处理更多元素，但保守使用 128 不会导致功能错误，仅可能存在轻微性能差异。

3. **DataCopyPad 对齐**：`BYTE_BLOCK = 32` 字节是硬件对齐要求。UINT8 每元素 1 字节，32 字节对齐对应 32 个元素，相比 FP16（16 个元素）对齐要求更低，不会引入对齐问题。

4. **sizeof(T) 使用**：代码中多处使用 `sizeof(T)` 计算字节偏移和 buffer 大小，`sizeof(uint8_t) = 1`，逻辑自洽。

5. **确定性计算**：最近邻插值仅涉及索引取值（Gather），不涉及浮点运算，UINT8 天然满足确定性计算要求——相同输入多次执行结果完全一致。

##### 3.2.3 算子泛化功能设计

为满足全场景兼容的泛化要求，UINT8 支持需覆盖以下场景：

1. **多种 dtype 覆盖**：UINT8 与现有 FP32/FP16/BF16/DOUBLE 共存，验证各 dtype 独立正确性
2. **两种缩放模式**：outputSize 指定输出尺寸 + scales 指定缩放系数
3. **三种数据排布**：NCDHW、NDHWC、ND
4. **非连续 Tensor**：支持 stride 不等于 shape 乘积的非连续输入
5. **边界场景**：
   - 最小输入 shape（如 [1,1,1,1,1]）
   - 最大上采样倍率
   - UINT8 边界值（0 和 255）
6. **性能对比**：UINT8 vs FP16 性能对比，劣化不超过 5%

##### 3.2.4 UINT8 专属适配说明

**UINT8 技术细节**：

- **数据范围**：uint8 为无符号 8 位整数，取值范围 [0, 255]。最近邻插值是纯粹的索引取值操作，不涉及数值运算，因此 UINT8 数据在 Gather 过程中不会产生精度损失，输出与输入值完全一致（bit-wise exact）。
- **内存效率**：UINT8 每元素仅占 1 字节，是所有支持类型中内存占用最小的。在相同 buffer 大小下，可容纳更多元素，理论上数据搬运效率更高。
- **Gather 偏移计算**：`CalculateGatherOffsetW` 函数中 `Muls(srcOffsetTensor, srcOffsetTensor, static_cast<int32_t>(sizeof(T)), dataCount)` 对 UINT8 计算偏移时 `sizeof(uint8_t) = 1`，偏移量即为元素索引值，逻辑正确。

### 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas A2 训练系列产品（ascend910b） | √ |
| Atlas A3 系列产品（ascend910_93） | √ |

### 算子约束限制

1. out 的数据类型和数据排布格式需与 self 保持一致
2. 输入和输出 shape 的 N、C 轴必须相同
3. 输入、输出 tensor 元素个数不超过 int32_t 最大值
4. outputSize 和 scales 两种缩放方式使用其一，outputSize 优先
5. 所有 tensor 支持非连续 Tensor
6. 不支持空 Tensor

### 可维可测分析

#### 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | 算子输出结果需与 CPU 参考实现完全一致，满足 AscendOpTest 工具默认阈值 | 任务书要求 |
| 性能标准 | UINT8 数据类型性能与 FP16 数据类型相比，劣化不超过 5% | 任务书要求 |

#### 测试用例规划

| 测试场景分类 | 用例描述 | 输入维度(Shape)及属性 | 数据类型 | 预期结果 |
| --- | --- | --- | --- | --- |
| **基础正向** | 常规 5D 上采样 | `[1, 3, 64, 64, 64]`, outputSize=[128,128,128] | UINT8 | 与 golden 完全一致 |
| **基础正向** | 多 batch 上采样 | `[2, 16, 32, 32, 32]`, outputSize=[64,64,64] | UINT8 | 与 golden 完全一致 |
| **基础正向** | 大 shape 上采样 | `[1, 1, 128, 128, 128]`, outputSize=[256,256,256] | UINT8 | 与 golden 完全一致 |
| **scale模式** | scale_factor 模式 | `[1, 3, 10, 10, 10]`, scale=(2.0, 2.0, 2.0) | UINT8 | 与 golden 完全一致 |
| **边界场景** | 最小 shape | `[1, 1, 1, 1, 1]`, outputSize=[2, 2, 2] | UINT8 | 与 golden 完全一致 |
| **边界场景** | UINT8 边界值 | `[1, 1, 4, 4, 4]`, 含0和255 | UINT8 | 与 golden 完全一致 |
| **非连续** | 非连续 Tensor | 非连续 stride 输入 | UINT8 | 与 golden 完全一致 |
| **性能对比** | UINT8 vs FP16 | `[1, 3, 64, 64, 64]`, outputSize=[128,128,128] | UINT8/FP16 | UINT8劣化≤5% |
| **dtype覆盖** | FP32 验证 | 常规 shape | FP32 | 与 golden 完全一致 |
| **dtype覆盖** | FP16 验证 | 常规 shape | FP16 | 与 golden 完全一致 |
| **dtype覆盖** | BF16 验证 | 常规 shape | BF16 | 与 golden 完全一致 |

### 兼容性分析

本次开发为现有算子的数据类型扩展，不改变已有 FP32/FP16/BF16 的功能行为。新增 UINT8 支持是在现有模板框架内增加新的模板实例化，对已有代码路径无影响，向后兼容。

## 附录：修订记录

| 日期 | 修订版本 | 修改描述 | 作者 |
| --- | --- | --- | --- |
| 2026-08-26 | v1.0.0 | 初稿：UpsampleNearest3d 算子 UINT8 数据类型扩展设计文档 | LLLAAAAA |
