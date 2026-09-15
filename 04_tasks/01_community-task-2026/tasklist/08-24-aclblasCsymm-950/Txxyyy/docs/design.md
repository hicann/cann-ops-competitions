# aclblasCsymm 算子设计文档

版本：v3.0，2026-09-13。

## 一、需求背景

### 1.1 需求来源

在Ascend 950PR上使用Ascend C实现单精度复数对称矩阵乘法
`aclblasCsymm`，补齐`ops-blas`的复数Symm接口。任务为2026年8月社区任务
`aclblasCsymm算子开发（950）`。

目标软件环境为CANN 9.1.0，算子使用句柄式BLAS接口和kernel直调方式。
语义参考cuBLAS `cublasCsymm`与Netlib `csymm`，精度参考为任务指定的
Netlib `cblas_csymm`。

### 1.2 功能与现有能力

计算定义如下：

```text
side = LEFT:   C = alpha * A * B + beta * C
side = RIGHT:  C = alpha * B * A + beta * C
```

A为复数对称矩阵，满足`A = A^T`，不取共轭；对角元素的虚部参与计算。
仅引用`uplo`指定的三角，另一三角通过对称关系访问。B和C为普通复矩阵，
C原地覆写。

`ops-blas`已有实数Symm及复数矩阵运算接口，可复用接口风格、句柄、
stream、workspace和测试框架。复数乘法通过实数乘法分解，在AIV上进行
数据变换与合并，在Cube上执行FP16输入、FP32累加的矩阵乘。
对不进入专用路径的参数和部分数值情形，采用通用FP32计算。

### 1.3 参考资料

- [任务代码仓](https://gitcode.com/cann/ops-blas)
- [Netlib csymm参考实现](https://www.netlib.org/blas/csymm.f)
- [cuBLAS Symm接口](https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-symm)
- [生态算子精度标准](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md)

## 二、需求分析

### 2.1 接口、数据类型与布局

```cpp
aclblasStatus_t aclblasCsymm(
    aclblasHandle_t handle,
    aclblasSideMode_t side,
    aclblasFillMode_t uplo,
    int m, int n,
    const aclblasComplex* alpha,
    const aclblasComplex* A, int lda,
    const aclblasComplex* B, int ldb,
    const aclblasComplex* beta,
    aclblasComplex* C, int ldc);
```

| 项目 | 定义 |
| --- | --- |
| 对外数据类型 | COMPLEX64；每个元素由FP32实部和FP32虚部组成 |
| A形状 | LEFT时为m×m，RIGHT时为n×n |
| B、C形状 | m×n |
| 数据布局 | 列主序，元素位置为`row + column * ld`，实虚部交错存储 |
| 标量 | alpha、beta为COMPLEX64，支持同为Host或同为Device指针 |
| 前导维 | lda≥max(1,dimA)，ldb、ldc≥max(1,m) |
| 输出 | C原地更新，仅写逻辑m×n区域，不改padding |

FP16仅作为内部乘法操作数格式，不增加对外dtype或私有接口。

### 2.2 接口行为

- handle为空返回`ACLBLAS_STATUS_HANDLE_IS_NULLPTR`。
- 非法side、uplo返回`ACLBLAS_STATUS_INVALID_ENUM`；负维数、非法前导维、
  不满足条件的空指针等返回`ACLBLAS_STATUS_INVALID_VALUE`。
- 先检查handle、枚举和维数；m=0或n=0随后直接成功，不访问矩阵或标量指针。
- 非空尺寸下解析alpha、beta。Host与Device标量混用返回INVALID_VALUE；
  Device标量按绑定stream的顺序读回，再以值传入kernel。
- alpha=0时不要求A、B为有效输入；beta=0时不读取C旧值。
  当前接口约定beta=0且C为空时成功返回。
- alpha=0、beta=1时C不变；alpha=0、beta=0时只清零C逻辑区域。
  alpha=0的其他情况采用通用复数乘法计算`beta * C`，保留特殊值行为。
- 所有设备工作提交到handle绑定的stream。同一stream上的输入更新、算子调用和
  结果读回遵循流顺序，读回前由调用方同步stream。
- 不支持超出lda、ldb、ldc语义的任意非连续视图；不涉及广播。
  m、n为运行时参数，不要求逐位确定性。

### 2.3 依赖与模块边界

| 模块 | 用途 |
| --- | --- |
| CANN 9.1.0、Ascend C | 运行时、SIMD/SIMT编程、Cube计算与数据搬运 |
| 项目已有Tensor API | GEMM视图、L1/L0布局和MMAD/Fixpipe封装 |
| 现有handle辅助接口 | 绑定stream、查询核数、管理库自有及用户workspace |
| Netlib BLAS 3.12 | 测试期生成FP32复数golden，不参与设备主计算 |
| GTest与CSV测试框架 | 参数生成、接口检查、精度比对、完整API计时 |

当前Csymm主路径不调用`aclblasSgemmStridedBatched`，不使用辅助stream，
不进行矩阵的Host拆分或CPU结果合并。

## 三、详细设计

### 3.1 总体分派

```text
aclblasCsymm
  -> 参数检查与零尺寸返回
  -> alpha/beta解析、矩阵指针检查
  -> alpha=0：返回、逻辑区域置零或复数缩放
  -> 满足专用参数、核数及workspace条件：
       单个MIX kernel
         AIV打包并收集统计
         同步后AIC计算；AIV汇总分派统计
         AIC结果就绪后，AIV合并或执行通用FP32回退
  -> 其余情况：AIV-only通用FP32 kernel
```

两种专用路径均要求：

```text
m = n = lda = ldb = ldc
alpha = (1, 0)
beta  = (0, 0)
```

| 路径 | 附加条件 | 活跃计算核 |
| --- | --- | --- |
| 256阶路径 | m=n=256，LEFT/UPPER | 24个AIC、48个AIV |
| 1024阶路径 | m=n=1024，side/uplo四种组合 | 28个AIC、56个AIV |
| 通用路径 | 不满足上述条件，或核数/workspace条件不满足 | 查询得到的AIV核数 |

Host查询实际AIC核数后选择固定任务布局；不满足所需核数时使用通用路径。
用户workspace容量不足或地址未按32字节对齐时同样使用通用路径，
不擅自重新分配用户内存。
库自有workspace按需扩容，分配失败时返回对应错误码。

### 3.2 高低位拆分与实数乘法

对每个经过变换的FP32实数操作数x，计算：

```text
xH = round_to_nearest_even_FP16(x)
xL = round_to_nearest_even_FP16(x - FP32(xH))
```

对实矩阵X、Y，使用三个FP16乘积项并在FP32中累加：

```text
X * Y ≈ XH * YH + XH * YL + XL * YH
```

低低项`XL * YL`未计算。该方法降低单次FP16量化的影响，但仍有残差截断、
变换加减和累加舍入误差；不能把上述近似写成对所有输入精确等价。
数值分派、通用回退和按任务参考的实际精度检查共同用于处理精度问题。

#### 3.2.1 256阶差分三乘

A、B的三组实操作数为：

```text
A侧：(Ar - Ai), Ai, (Ar + Ai)
B侧：Br, (Br - Bi), Bi

P0 = (Ar - Ai) * Br
P1 = Ai * (Br - Bi)
P2 = (Ar + Ai) * Bi

Cr = P0 + P1
Ci = P2 + P1
```

每个P使用上述三个高低位乘积项。输入变换在FP32中执行，随后拆为FP16高低位；
输出按实虚交错方式写回。该路径限于LEFT/UPPER和alpha=1、beta=0。

#### 3.2.2 1024阶三乘

令LEFT时`X=A, Y=B`，RIGHT时`X=B, Y=A`：

```text
P0 = Xr * Yr
P1 = Xi * Yi
P2 = (Xr + Xi) * (Yr + Yi)

Cr = P0 - P1
Ci = (P2 - P0) - P1
```

每个P同样使用三个高低位乘积项。RIGHT通过交换A、B的打包目标角色处理，
保持数学上的乘法顺序。Cube视图与GM地址映射负责适配列主序存储。

### 3.3 分核、分块与数据搬运

#### 3.3.1 256阶路径

- 20个AIV任务打包A：对角块拆成半块任务，跨块区域按上三角读取。
  Cube搬运时通过对称块访问和转置复用，避免完整镜像打包的重复工作。
- 28个AIV任务连续打包B，每个常规任务处理2368个元素，末任务处理剩余元素。
- 24个AIC对应3个实乘结果平面、每平面8个输出块，每核处理一个块。
  Cube视图的块为128×64，K1=K0=128，完整K=256。
- AIV合并按1408个输出元素分块，末块按实际数量处理。
  不承担输出的核仍参与所需同步。

#### 3.3.2 1024阶路径

A、B按64×64块搬入UB。A通过设备坐标选择指定三角或对称位置，
B按普通矩阵搬运；索引在设备侧计算，不依赖调用方预先上传索引表。
打包输出每个操作数的六个half平面，并融合范围统计。

Cube计算使用宽块和均衡尾波：

| 波次 | 分工 | Cube视图输出块 | K1 | K0 |
| --- | --- | --- | ---: | ---: |
| 前三波 | 每波28核各处理一个完整块，共84块 | 128×256 | 128 | 64 |
| 最后一波 | 剩余12个完整块拆成24个半块，由24核处理 | 128×128 | 128 | 128 |

三个实乘平面共有96个完整宽块。末波拆分后共108个计算任务，
覆盖三个1024×1024结果平面的全部元素且无重叠。
宽块减少重复操作数搬运，半块尾波降低末尾任务的不均衡。

AIV合并以4096个输出元素为块，使用双缓冲搬入三个FP32结果平面，
完成复数合成和交错写回。

### 3.4 数值分派与通用FP32路径

#### 3.4.1 256阶分派

打包时汇总A所引用元素的虚部最小值l与最大值h，令：

```text
w = h - l
s = max(abs(l), abs(h))
```

仅当`w > s/32`且`s <= 256`时采用混合精度合并结果，否则回退。
比较条件不成立时也回退。该条件识别虚部近常量等数值情形，
不等同于对A实部、B或全部输出误差的完整范围检查。

#### 3.4.2 1024阶分派

分别汇总A、B实部和虚部的最小值、最大值。每分量使用上述w、s定义：

- 任一分量不满足`h >= l`时回退。
- 对每个矩阵，取实虚两分量s的最大值。该幅值不在
  `[1e-4, 256]`内时回退。
- A实部和虚部均满足`w <= s/32`时回退。
- 对一个矩阵，其实部和虚部均满足`abs(l+h) > 0.5*w`时标记为偏置显著。
  A、B同时满足时回退。

这些统计规则是实现的数值分派策略，不是对任意FP32输入的误差证明。
精度结论采用第5章的任务参考和实测覆盖，不要求每个输入逐位复刻参考。

#### 3.4.3 通用计算与回退执行

通用路径使用AIV SIMT，以FP32复数乘法和加法计算每个输出元素，
按side/uplo对应的Netlib逐输出运算次序组织求和。
分别舍入复数乘法的实乘与加减，并处理复数Inf/NaN恢复逻辑。

Host直接选择通用路径时，只发射AIV-only kernel，无矩阵临时workspace。
专用MIX kernel内部选择回退时，在Cube计算结束后由AIV读取原始A、B计算C；
不是先在Host读回矩阵检查，也不是跳过已经发射的Cube计算。
原始矩阵保持只读，合并与回退均只写C。

### 3.5 Workspace与片上资源

令`q=m*n`，两种紧凑方阵专用路径的workspace需求相同：

```text
PA：6个half平面       12*q字节
PB：6个half平面       12*q字节
P ：3个float平面      12*q字节
统计及预留空间        4096字节
合计                  36*q + 4096字节
```

256阶需求2363392字节，1024阶需求37752832字节。
库自有workspace按现有增长策略扩容，实际分配容量可能大于需求。
用户workspace不重新分配；容量或对齐条件不满足时使用通用路径。

| 资源 | 256阶路径 | 1024阶路径 |
| --- | --- | --- |
| L1 | 单槽操作数至多96KiB，双槽基址间隔256KiB | 完整宽块单槽至多192KiB，双槽基址间隔256KiB |
| L0A/L0B | 每个ping-pong槽不超过32KiB | 每个ping-pong槽不超过32KiB |
| L0C | 单个128×64 FP32块32KiB | 完整块128KiB、尾块64KiB，两个槽间隔128KiB |
| UB | 打包、合并及统计各阶段按固定窗口复用 | 打包/合并双缓冲与统计窗口分阶段使用 |

各阶段使用的UB最大地址均小于256KiB，L1最大访问地址低于512KiB。
统计区按核分隔，避免多个核在同一写入块上竞争。

### 3.6 同步与调用生命周期

- 两个专用kernel使用MIX AIC:AIV=1:2模式。打包发布后Cube读取操作数，
  Cube结果就绪后AIV合并；回退分支不跳过其他核等待的发布事件。
- 256阶统计读取额外建立MTE3到MTE2依赖，保证所有打包核发布的统计在读取前就绪。
  向量统计转为标量判断时建立V到S依赖。
- L1、L0缓冲复用分别由对应MTE/MMAD事件约束；L0C写回与复用由M_FIX/FIX_M约束。
  MMAD最后一次累加发布相应L0C区域，Fixpipe消费对应完成标志。
- 所有kernel与置零操作都在handle stream上提交，无跨调用辅助流交叠，
  不限制调用方在同一stream上异步更新下一次输入。
- 每次调用重新读取输入、打包并计算，不复用跨调用输入快照、输出或数值分派结论。
- stream切换、workspace切换和Destroy沿用现有handle的同步及所有权管理。

### 3.7 实现文件与硬件范围

以下路径相对于`ops-blas`仓根：

| 文件或目录 | 当前职责 |
| --- | --- |
| `include/cann_ops_blas.h` | 公共接口声明 |
| `blas/symm/arch35/csymm_host.cpp` | 参数解析、快速返回、路径选择与workspace管理 |
| `csymm_fast_small_kernel.cpp`、`csymm_fast_small_pack.h` | 256阶变换、打包、Cube计算和合并 |
| `csymm_fast_small_range.h` | 256阶虚部范围分派 |
| `csymm_fast_large_kernel.cpp` | 1024阶打包、宽块与尾波计算、合并 |
| `csymm_fast_range.h` | 范围汇总、1024阶分派及MIX内回退调用 |
| `csymm_reference_common.h`、`csymm_reference_kernel.cpp` | 共用FP32计算及AIV-only入口 |
| `blas/symm/README.md` | 接口、布局及使用约束 |
| `test/symm/csymm/` | CSV、golden、wrapper、GTest与构建配置 |

表中未带目录的实现文件均位于`blas/symm/arch35/`。
验证设备为Ascend 950PR，代码目标为arch35；其他产品不作为本任务实测结论。

## 四、特性交叉分析

| 特性 | 处理方式 |
| --- | --- |
| 精度与性能 | 专用路径降低规定性能场景的耗时，通用路径覆盖其他参数；数值分派选择适用路径 |
| side/uplo | LEFT/RIGHT保持不同乘法顺序，UPPER/LOWER只读取指定三角，对角虚部不修正 |
| 标量与特殊值 | alpha=0、beta=0/1有专门行为；其他缩放和回退使用共用复数乘法 |
| 输入修改与异步调用 | 同一stream按序执行，每次重新处理输入，支持调用间异步更新 |
| workspace | 复用现有所有权管理；用户内存不被重新分配，通用路径不使用矩阵workspace |
| 并发 | 不引入跨handle的输入或结果缓存；同一handle沿用库的使用约定 |
| 其他算子 | 新计算代码限定在Csymm路径，不改变Ssymm或通用GEMM计算逻辑 |

## 五、可维可测分析

### 5.1 精度标准与自测结果

golden由Netlib BLAS 3.12的`cblas_csymm`生成，按任务要求使用FP32复数参考。
实部和虚部分别检查：

```text
abs(actual - golden) <= 2^-16 + 2^-10 * abs(golden)
matched_ratio >= 0.99
每元素误差硬上限 = max(1e-2, 32 * ULP(abs(golden)))
```

匹配比例和逐元素误差硬上限均须满足。双方均为NaN或为同号Inf时按既有验证器规则处理。
验收不要求bit-exact，也不以FP64参考替换任务golden。

2026-09-13，同一生产库完成以下验证：

| 验证项 | 结果 |
| --- | --- |
| 任务CSV 1200项、补充CSV 2项、fixture 9项 | 1211/1211通过，无遗漏、重复或跳过 |
| 通用参数接口依赖 | 4组通过 |
| 专用形状接口依赖 | 4组通过 |
| 抵消与特殊值等补充API验证 | 11组，首次与计时后共22次检查全部通过 |

补充验证涉及低低项抵消、可精确表示输入的累加抵消、分量抵消、偏置随机输入、
次正规值、有限值溢出、半精度边界附近值和NaN/Inf。
接口依赖覆盖同stream更新B及重置C、切换stream、Device标量、Destroy排空，
以及足量用户workspace前后保护区。补充验证记录随自测报告提供。

### 5.2 性能标准与结果

性能仅按任务指定的三个case设置硬线，不把同形状的所有其他输入都视为这三个case。
采用标准测试工程，在Ascend 950PR/CANN 9.1.0上调用完整`aclblasCsymm` API。
每进程预热50次，采样500次；计时包含API发射及末尾stream同步。
输入初始分配、上传和golden计算不计入该调用区间，打包、统计、Cube计算与合并均计入。

| case | 形状与方向 | 要求（us） | 五进程均值范围（us） | 中位数（us） |
| --- | --- | ---: | ---: | ---: |
| TC_PF_1001 | 256×256，LEFT/UPPER | ≤7.53 | 6.512970–6.563250 | 6.532700 |
| TC_PF_1002 | 1024×1024，LEFT/LOWER | ≤89.27 | 79.581794–81.741596 | 80.118058 |
| TC_PF_1003 | 1024×1024，RIGHT/UPPER | ≤86.34 | 80.813384–82.432974 | 81.522946 |

五个独立进程均通过三条性能断言及结果精度检查，使用同一生产库。
上述结论对应规定输入和测量环境，不外推为所有形状或回退输入的性能。
其他CSV中的性能类输入也参与完整精度回归，不另外赋予未规定的耗时硬线。

### 5.3 测试组织与复现

使用`ops-blas`的CSV驱动GTest六件套：param、golden、CMakeLists，以及arch35下
wrapper、test.cpp、test.csv。输入沿用任务的固定种子和均匀/正态分布规则。

完整精度集合选择：

```text
--gtest_filter=CsymmArch35Test.*:Csymm/CsymmArch35Test.CsvDriven/*
```

三个性能case选择：

```text
--gtest_filter=*.CsvDrivenPerf/TC_PF_1001:*.CsvDrivenPerf/TC_PF_1002:*.CsvDrivenPerf/TC_PF_1003
```

测试按用例ID核对实际执行结果。首次输出与连续调用后的输出分别检查，
不以预热后的正确性替代首次正确性。实现或分派条件变化后重新执行相关用例，
最终对同一库完成精度、接口依赖与三项性能验证。

### 5.4 兼容性与维护

公共接口沿用`aclblasCsymm`及仓内状态码，不新增专用API。
混合精度是内部实现选择，用户仍传入COMPLEX64矩阵。
片上资源及固定任务数与专用形状绑定，Host核数检查和通用分派保留。
维护时应同步更新分派条件、布局、同步依赖、workspace计算和对应自测，
以保持设计描述与实际实现一致。
