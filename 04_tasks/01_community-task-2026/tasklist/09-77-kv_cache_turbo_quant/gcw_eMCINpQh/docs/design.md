# KvCacheTurboQuant 算子设计文档

# 一、需求背景（required）

## 1.1 需求来源

通过社区任务完成面向标准MHA/GQA推理的KV cache在线压缩算子贡献。算子采用Ascend C实现，通过ACLNN工程化接口调用。

## 1.2 背景介绍

### 1.2.1 KvCacheTurboQuant算子实现优化

自回归推理需要持续保存历史Key/Value，BF16 KV cache的存储量随上下文长度线性增长。KvCacheTurboQuant将每个KV向量编码为bit-packed主量化索引、QJL残差符号以及两个尺度标量，以降低持久化cache占用。

该算子为新增量化算子，CANN内置OPP中没有同名TBE实现。功能标杆采用包内PyTorch参考实现，性能标杆采用功能等价的PyTorch eager实现。


### 1.2.2 KvCacheTurboQuant标杆实现现状分析

#### 1.2.2.1 标杆算子支持的数据类型和数据格式

令`T`为token数、`H`为KV head数、`D`为head维度、`J`为QJL维度、`b`为主量化位宽。

| 参数 | 方向 | 数据类型 | 数据格式 | 形状 |
|---|---|---|---|---|
| `kv_vectors` | 输入 | BF16 | ND | `[T,H,D]` |
| `rotation_matrix` | 输入 | FP32/BF16 | ND | `[D,D]` |
| `qjl_matrix` | 输入 | FP32/BF16 | ND | `[J,D]` |
| `quant_idx` | 输出 | UINT8 | ND | `[T,H,ceil(D*b/8)]` |
| `quant_qjl` | 输出 | UINT8 | ND | `[T,H,ceil(J/8)]` |
| `quant_norm` | 输出 | BF16/FP16 | ND | `[T,H]` |
| `quant_gamma` | 输出 | BF16/FP16 | ND | `[T,H]` |

属性`mse_bits`支持2、3、4，默认值为3；`output_dtype`取0时输出BF16 metadata，取1时输出FP16 metadata。

#### 1.2.2.2 标杆算子实现描述

标杆实现先计算输入向量范数并单位化，然后使用旋转矩阵将单位向量映射到量化空间。旋转结果依据固定centroid边界逐维量化并打包为主编码；主重构与旋转结果之差形成残差，残差经QJL矩阵投影后取符号并打包。两个标量分别保存输入范数和原尺度残差范数。

```text
norm = ||x||₂
u = x / norm                         # norm=0时u=0
y = H @ u
idx = scalar_quantize(y, mse_bits)
r = y - centroid[idx]
rho = ||r||₂
gamma = norm * rho
v = r / rho                           # rho=0时v=0
qjl = sign(S @ v)
```

QJL只保留投影符号，因此`rho>0`时使用`r`或`v`投影得到的符号相同；接口语义使用单位残差`v`，`gamma`单独保存残差幅度。零残差时投影值为0，按照`>=0`编码为1。

主编码和QJL均采用低位优先打包。每8个主索引先组合为无符号整数，再依次写出低地址字节；QJL的第0个符号写入byte的bit 0。每个head重新从bit 0开始，不跨head拼接：

```text
idxWord = OR(idx[8*g+lane] << (lane*mse_bits)), lane=0..7
quant_idx[g*mse_bits+byte] = (idxWord >> (8*byte)) & 0xff
quant_qjl[g] = OR(qbit[8*g+lane] << lane), lane=0..7
```

默认128维、3-bit配置中，主编码48 bytes、QJL编码16 bytes、norm和gamma各2 bytes，共68 bytes/head；BF16原向量为256 bytes/head。

#### 1.2.2.3 标杆算子实现流程

标杆实现依次完成输入范数计算和单位化、H矩阵旋转、centroid量化与idx打包、量化残差及gamma计算、单位残差的S矩阵投影，最后执行符号量化和QJL打包。四个输出分别为主编码、QJL编码、输入范数和原尺度残差范数。

# 二、需求分析（required）

## 2.1 外部组件依赖

| 组件 | 用途 |
|---|---|
| CANN 9.1.0及以上 | ACLNN工程构建、算子注册与运行时调用 |
| Ascend C | AI Core Kernel实现 |
| Ascend Matmul API | AIC矩阵投影计算 |

PyTorch、NumPy和ml_dtypes仅用于测试数据及参考结果生成，不属于算子运行依赖。

## 2.2 内部适配模块

| 模块 | 作用 |
|---|---|
| OpDef与Shape/Dtype推导 | 定义输入、输出、属性及输出形状 |
| Tiling模块 | 选择执行mode，计算分核、stride、Cube tiling和workspace |
| Vector编码模块 | norm、量化、残差、metadata转换和bit-pack |
| Cube投影模块 | 完成`H@u`和`S@r`矩阵计算 |

## 2.3 需求模块设计

### 2.3.1 Ascend C算子原型

```text
KvCacheTurboQuant(
    kv_vectors, rotation_matrix, qjl_matrix,
    mse_bits=3, output_dtype=0)
  -> quant_idx, quant_qjl, quant_norm, quant_gamma
```

### 2.3.2 Ascend C算子相关约束

- 面向标准MHA/GQA KV向量，不支持MLA latent cache。
- `num_kv_heads`支持4～32，`mse_bits`支持2/3/4。
- 实现支持`D/J`取64、128、256；BF16矩阵回退路径限定`D=J=128`。
- 输入为连续ND Tensor，不支持广播。
- FP16只用于metadata输出，KV输入固定为BF16。

# 三、需求详细设计（required）

## 3.1 调用方式

算子采用ACLNN两阶段调用：先调用GetWorkspaceSize接口获取workspace大小及executor，再分配workspace并在指定stream执行算子。OpDef、shape推导和tiling由Host工程完成，调用方无需直接启动Kernel。

## 3.2 需求总体设计

整体采用Vector与Cube协同结构。AIV负责向量归约、centroid量化、残差处理及bit-pack；AIC负责两个矩阵投影。Host根据矩阵dtype、维度和向量数量选择批量Cube路径或Vector路径。

### 3.2.1 Host侧设计

#### 3.2.1.1 分核策略

设总向量数`V=T*H`。执行mode选择如下：

| 条件 | mode | 执行路径 |
|---|---:|---|
| `V=0` | 0 | 空输入直接返回 |
| FP32矩阵、`D=J=128`、`V>=128`且`V%128=0` | 2 | 128行批量Cube流水 |
| FP32矩阵、`D=J=128`的其他非空输入 | 1 | 8行小批Cube流水 |
| FP32矩阵、支持的非128维 | 3 | 通用Vector路径 |
| 任一矩阵为BF16且`D=J=128` | 4 | dtype-aware Vector路径 |

mode 1在`V<=8`时使用1个AIV任务组，其余使用2个AIV任务组。mode 2的AIC组数为：

```text
aicGroups = max(1, min(V/128, platformAicNum, 8))
```

mixed Kernel按1个AIC对应2个AIV启动，因此Host写入的Vector任务数为`2*aicGroups`。其他Vector路径按`min(V, platformAivNum)`确定有效AIV任务数。

#### 3.2.1.2 数据分块和内存优化策略

mode 1以8个向量为一个Cube行块，mode 2以128个向量为一个Cube行块；尾部向量数向8对齐：

```text
Vpad = ceil(V/8) * 8
```

Cube路径用户workspace依次保存输入FP32、norm、旋转结果、残差和QJL投影：

```text
userWorkspace = Vpad * (4 * 128 * sizeof(float) + sizeof(float))
              = Vpad * 2052 bytes
totalWorkspace = 16 MiB系统workspace + userWorkspace
```

批量Vector阶段以`BATCH`个向量为tile：

```text
ELEMENTS = BATCH * 128
tileCount = ceil(V / BATCH)
```

Norm阶段LocalMemory主要包括`ELEMENTS`个BF16和`2*ELEMENTS+2*BATCH`个FP32；量化阶段按主编码、mask、offset、codebook和metadata分别划分TBuf。最后一个tile先清零无效行，写回时仅输出`valid=min(BATCH,V-base)`个向量。

#### 3.2.1.3 tilingKey规划策略

当前二进制统一使用`tilingKey=0`。矩阵dtype、维度、位宽、metadata类型和mode均通过tiling data传递，在Kernel入口内部dispatch，避免为同一算法生成大量tilingKey组合。

#### 3.2.1.4 数据检测

Host检查三个输入的rank、矩阵维度相容性、head范围、支持维度、`mse_bits`和`output_dtype`，并检查元素数量可由32位tiling字段表示。矩阵的正交性及统计属性属于调用方数据契约，不在每次在线编码时从Device回读检测。

### 3.2.2 Kernel侧设计

#### 3.2.2.1 Kernel侧实现描述

Kernel入口为mixed AIC/AIV类型。Cube路径的处理顺序为：

1. AIV读取BF16向量，转FP32，计算L2范数和单位向量。
2. AIC执行`H@u`矩阵投影。
3. AIV根据centroid阈值计算索引，生成主编码、残差和gamma。
4. AIC执行`S@r`矩阵投影。
5. AIV按投影结果生成符号位，打包QJL，并将norm/gamma转换为目标metadata类型。

Vector路径在AIV上逐向量或逐批完成相同语义。主编码和QJL均按低位优先打包。各阶段通过硬件事件同步，保证AIC读取已完成的中间数据，并避免AIV提前覆盖workspace。

#### 3.2.2.2 Ascend C实现流程

Ascend C实现按“AIV预处理 → AIC旋转投影 → AIV主量化和残差处理 → AIC QJL投影 → AIV符号打包”的顺序执行。AIV预处理结果和两个投影阶段的中间结果通过用户workspace传递；主编码、norm和gamma在量化阶段写出，QJL编码在最后阶段写出。

#### 3.2.2.3 Ascend C实现与标杆实现的差异点和原因

| 差异点 | Ascend C实现 | 原因 |
|---|---|---|
| 矩阵投影 | 批量路径使用Cube Matmul | 提高大批量矩阵计算吞吐 |
| 阶段间数据 | 使用GM workspace连接AIV/AIC | mixed Kernel中AIV与AIC需要共享中间结果 |
| 量化和打包 | Vector并行生成索引、mask和packed bytes | 减少标量循环和临时Tensor |
| 小批处理 | 使用8行Cube或Vector路径 | 避免小输入启动128行tile造成浪费 |
| 计算精度 | BF16输入转FP32完成归约和量化 | 控制norm、残差和边界判断误差 |

## 3.3 支持硬件

| 芯片版本 | 是否支持 |
|---|---|
| Atlas 800T A2（Ascend 910B系列） | 是 |

## 3.4 算子约束限制

- 不支持MLA latent cache和广播。
- 输入KV固定为BF16；矩阵支持FP32/BF16，metadata输出支持BF16/FP16。
- 主量化位宽仅支持2、3、4。
- 调用方须保证旋转矩阵、QJL矩阵与编码消费端使用同一约定。

# 四、特性交叉分析

| 特性交叉项 | 处理方式 |
|---|---|
| `mse_bits × 输出shape` | `quant_idx`末维按`ceil(D*mse_bits/8)`推导 |
| 矩阵dtype × 维度 | BF16矩阵限定128维；FP32矩阵支持64/128/256 |
| metadata类型 × Kernel路径 | 计算保持FP32，仅在写回阶段选择BF16或FP16转换 |
| 批量大小 × 执行单元 | 大批量走Cube，小批量或扩展dtype/维度走Vector |
| 尾部 × bit-pack | 无效行清零且不写回，避免越界和脏数据 |

# 五、可维可测分析

## 5.1 精度标准/性能标准

| 验收标准 | 要求 |
|---|---|
| 重构精度 | 量化重构向量相对均方误差小于0.05 |
| 内积精度 | 量化向量内积相对误差95分位小于0.1 |
| 性能 | 相对功能等价PyTorch参考实现达到10倍及以上加速 |
| 存储 | 默认配置持久化KV cache压缩比约3.76倍 |

## 5.2 兼容性分析

算子输出四个cache字段的shape、打包顺序和metadata类型由接口固定。消费端需使用匹配的`mse_bits`、centroid表及矩阵约定。算子为新增接口，不改变调用方原始BF16 KV输入格式。
