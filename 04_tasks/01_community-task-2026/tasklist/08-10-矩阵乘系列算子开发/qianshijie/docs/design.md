# 【社区任务】aclblasSgemmStridedBatched 算子设计文档

> **任务编号**：08-10-矩阵乘系列算子开发<br>
> **团队名称**：qianshijie<br>
> **适配硬件**：Atlas 800I A2 / Atlas 800I A3（A2/A3 系列，910B3）<br>
> **CANN 版本**：9.1.0<br>
> **开发语言**：Ascend C<br>
> **目标仓库**：<https://gitcode.com/cann/ops-blas><br>
> **目标目录**：`blas/gemm_strided_batched/arch22/`

## 设计文档 PR 要求

- 本文只提交设计，不包含算子实现、二进制、测试结果或性能通过结论。
- 文档提交位置为
  `04_tasks/01_community-task-2026/tasklist/08-10-矩阵乘系列算子开发/qianshijie/docs/design.md`。
- 代码阶段再向 `cann/ops-blas` 提交 `blas/gemm_strided_batched/arch22/`
  和对应 `test/gemm_strided_batched/arch22/`；两阶段的硬件、接口和精度
  口径必须保持一致。

# 一、需求背景（required）

## 1.1 需求来源

本设计依据 2026 年 8 月社区任务资料包
`aclblasSgemmStridedBatched_Atlas800IA3_task_doc.md` 及其测试目录。资料包
SHA256 为
`9835e04807a593e63acf0afcf33e410e24b8a07aa93b4a99d91b3a8d7855f3b3`；其中
`gemm_strided_batched_test.csv` SHA256 为
`61699c9d88369c583dc2be74605188781a3f51bc2bb92448a9c5746ada6541b0`。

资料包成员冻结如下（大小为字节）：

| 成员 | 大小 | SHA256 |
| --- | ---: | --- |
| `aclblasSgemmStridedBatched_Atlas800IA3_task_doc.md` | 19217 | `2284e200ea6deff4b58d882128d87ca0370545cb2fde9bc7958ce70a0f33a04e` |
| `test_cases/gemm_strided_batched_test.csv` | 193664 | `61699c9d88369c583dc2be74605188781a3f51bc2bb92448a9c5746ada6541b0` |
| `test_cases/gen_csv.py` | 15228 | `71cc6f2d64759712a9303573295a4037270e378d8cf75b0064ba800e92010a79` |
| `test_cases/gpu_baseline.csv` | 9230 | `e3017190664a336a9179539447deb6a98f6520d78a58ef24189499c68a549f54` |
| `test_cases/README.md` | 5495 | `6745bd2e192d96cc3e29bb99fc33b8439eed427e421c374289e64788085bc389` |
| `test_cases/verify_accuracy.py` | 4663 | `41ec207fd986cd1f91ee7d73127b561f6c225357835da873d869d9f7c7a01eeb` |
| `test_cases/verify_performance.py` | 6622 | `8c4b10328101b883e250466aa082f59e2373344504c67bad12be307810f4a002` |

算子对标两类公开语义来源：

1. cuBLAS `cublasSgemmStridedBatched` 的接口顺序、列主序矩阵和跨步批量
   组织方式；
2. Netlib BLAS `sgemm` 的实数矩阵乘累加和空维/标量边界语义。

目标是为 A2/A3 `arch22` 增加共用的 `aclblasSgemmStridedBatched` 实现，保留
`include/cann_ops_blas.h` 中已有的公共声明，不创建产品私有平行接口。

## 1.2 背景介绍

### 1.2.1 功能和数学语义

对 `i = 0, ..., batchCount - 1`：

```text
C_i = alpha * op(A_i) * op(B_i) + beta * C_i
A_i = A + i * strideA
B_i = B + i * strideB
C_i = C + i * strideC
```

`op(X)` 的取值为：

| trans | 运算 |
| --- | --- |
| `ACLBLAS_OP_N` | `X` |
| `ACLBLAS_OP_T` | `X^T` |
| `ACLBLAS_OP_C` | `X^H`，实数 FP32 下等价于 `X^T` |

三个矩阵均为 FP32、Column-Major。`strideA` 和 `strideB` 使用元素个数而非
字节数；为 0 时表示跨 batch 复用同一只读矩阵。`strideC=0` 且
`batchCount>1` 会造成写覆盖，属于未定义行为，由调用方规避。

### 1.2.2 API 原型和参数边界

```cpp
aclblasStatus_t aclblasSgemmStridedBatched(
    aclblasHandle_t handle, aclblasOperation_t transA, aclblasOperation_t transB,
    int m, int n, int k, const float* alpha, const float* A, int lda,
    int64_t strideA, const float* B, int ldb, int64_t strideB,
    const float* beta, float* C, int ldc, int64_t strideC, int batchCount);
```

| 参数 | 语义/布局 | 合法范围和错误行为 |
| --- | --- | --- |
| `handle` | 已创建的 ops-blas Host 句柄，携带 stream | 为空返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| `transA/transB` | Host 枚举，N/T/C | 其他值返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| `m/n/k` | 输出行数、列数和归约长度 | 非负；负值返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| `alpha/beta` | Host FP32 标量指针，全批共用 | 空指针返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| `A` | batch 0 的 Device 首地址，物理尺寸为 `lda`×存储列数 | 非空乘法路径必须有效；空维/跳过乘法时不读取 |
| `B` | batch 0 的 Device 首地址，物理尺寸为 `ldb`×存储列数 | 非空乘法路径必须有效；空维/跳过乘法时不读取 |
| `C` | batch 0 的 Device 输入/输出首地址 | 非空输出必须可写；`beta=0` 只表示不读取旧值 |
| `lda/ldb/ldc` | Column-Major 前导维度 | 分别不小于存储行数、`max(1,m)` |
| `strideA/strideB/strideC` | 相邻 batch 的元素偏移 | A/B 可为 0；C 的 0 步长多 batch 未定义 |
| `batchCount` | GEMM 三元组数量 | 非负；0 为合法 no-op |

边界行为与任务书一致：`m=0`、`n=0` 或 `batchCount=0` 返回成功且不执行
计算；`k=0` 或 `alpha=0` 跳过乘法，只执行 `C_i=beta*C_i`；`beta=0` 时
不需要读取旧 C。C 的不同 batch 子矩阵不得重叠，A/B 的 stride 和缓冲区长度
由调用方保证不越界。

### 1.2.3 现有实现分析与本次范围

当前 `ops-blas` 主线已有同名公共头文件声明和其他架构目录，但 A2/A3
`arch22` 路径仍需独立实现。本次设计不把其他 SoC 的实现、官方整算子或
Host 计算当作候选实现，也不改变公共 API 的名字、参数顺序或类型。

## 1.3 测试资料包角色和覆盖

资料包中的 `gemm_strided_batched_test.csv` 共 1200 行：1000 行精度、200 行
性能/内存。生成器固定默认种子 20260823，类别如下：

| 类别 | 前缀 | 行数 | 覆盖重点 |
| --- | --- | ---: | --- |
| L0 基础 | `TC_L0` | 9 | N/T/C 全组合、小尺寸、多 batch |
| L1 尺寸 | `TC_SQ` | 22 | 1 到 2048 的方阵、质数和非对齐尺寸 |
| L2 标量 | `TC_AB` | 8 | alpha/beta 特殊值和组合 |
| L3 批量 | `TC_BC` | 13 | batchCount 1 到 1024 |
| L4 布局 | `TC_LD`/`TC_SD`/`TC_BR` | 7 | leading dimension、空隙和 A/B 广播 |
| L5 填充值 | `TC_FL` | 6 | 随机、全零、交替、极值、Inf、NaN |
| L6 边界 | `TC_ED` | 24 | no-op、缩放、非法参数和空指针 |
| 扩展精度 | `TC_EX` | 911 | 确定性尺寸/转置/标量/批量组合 |
| 性能/内存 | `TC_PF` | 200 | 任务书典型值、大方阵、矩形和大 batch |

其中 17 行是句柄、枚举、负维度、非法 leading dimension 或空指针返回码
探针，不产生张量输出；它们仍必须由 ops-blas C++ GTest 保留并逐项验证，
不能被描述成 OpForge 精度 case。实现提交时不得删除这些探针，也不得从
`TC_PF` 中挑选子集来代替 200 行性能验收。

# 二、需求分析（required）

## 2.1 需求描述

使用 Ascend C 在 A2/A3 `arch22` 上实现 FP32 Column-Major
`aclblasSgemmStridedBatched`，对齐 cuBLAS/Netlib 的核心数学语义，并完成：

1. N/T/C 转置组合、矩形和非对齐矩阵、leading dimension padding；
2. A/B 零 stride 广播和非紧凑正 stride；
3. `k=0`、`alpha=0`、`beta=0`、零维和零 batch 快速路径；
4. 固定 64 位 stride 地址计算、C batch 不重叠和尾块内存安全；
5. 全量精度、性能、内存和 API 负向自验。

## 2.2 需求拆解

- **接口兼容**：复用公共 `aclblasHandle_t`、`aclblasOperation_t` 和状态码。
- **语义正确**：列主序寻址、FP32 累加、alpha/beta 规则和完整 C 物理缓冲
  输出必须一致，未覆盖的 padding/gap 保持不变。
- **运行时泛化**：m/n/k、batchCount、转置、stride 和 leading dimension
  从真实调用参数读取；不能使用 case 名、CSV 行号、固定公共 shape 或答案
  进行分发。
- **硬件路径**：仅在 `arch22` 编译路径中选择 A2/A3 支持的 Cube/Vector
  原语，不能把其他架构二进制或另一后端作为替代。
- **性能与内存**：在任务书定义的 A2/A3 设备和 CANN 9.1.0 上，先 warmup
  后完成有效采样；单 case Host 侧内存不超过 512 MiB，生成器的 FP32
  A/B/C 元素预算按 256 MiB 口径检查。

## 2.3 依赖和内部模块

| 模块 | 计划位置 | 职责 |
| --- | --- | --- |
| 公共 API | `include/cann_ops_blas.h` | 复用已有 `aclblasSgemmStridedBatched` 声明 |
| Host 校验/tiling | `blas/gemm_strided_batched/arch22/` | 参数检查、资源查询、分核和 launch |
| Kernel | `blas/gemm_strided_batched/arch22/` | A/B 搬运、Cube GEMM、alpha/beta 和 C 写回 |
| 测试工程 | `test/gemm_strided_batched/arch22/` | CSV loader、cblas golden、GTest、NPU wrapper |
| 公共 runtime | CANN ACL/ops-blas handle | stream、Device 内存和异步执行 |

测试侧可使用 cblas/Netlib、GTest 和 msprof；这些仅是验证依赖，不进入算子
运行时，也不替代 Device 计算。

## 2.4 Ascend C 算子原型

调用链为：

```text
aclInit -> aclblasCreate -> aclblasSetStream
       -> aclblasSgemmStridedBatched -> caller synchronizes the bound stream
```

接口正常路径异步提交；不在函数内隐式把完整结果同步回 Host。Host 只负责
元数据和 tiling，不逐元素计算。

# 三、需求详细设计（required）

## 3.1 Host 侧设计

### 3.1.1 参数校验顺序

校验顺序固定为：Handle 和枚举、维度和 leading dimension、stride 非负性、
零输出维快速返回、alpha/beta 指针、按实际路径读取的 A/B/C 指针，最后做
64 位字节数和地址上界检查。这样既保持错误码稳定，也避免在 `beta=0` 或
空维路径读取不需要的输入。

所有 `ld*`、stride、batch 偏移、元素字节数和乘积中间值先在 `int64_t`/
`uint64_t` 中计算，再检查溢出后写入有界 TilingData。非法参数只返回明确
状态码，不继续 launch。

### 3.1.2 物理布局和寻址

存储矩阵的物理行数/列数由转置属性决定：

```text
trans=N: A physical = lda x k, B physical = ldb x n
trans=T/C: A physical = lda x m, B physical = ldb x k
C physical = ldc x n
```

列主序元素地址为 `base + batch*stride + column*ld + row`。A/B 的零 stride
只复用只读 panel；C 每个 batch 仍拥有不重叠的写区。尾块以有效
`validM/validN/validK` 控制，padding 不参加数学运算。

### 3.1.3 分核和 tiling

Host 从平台查询可用 Cube/Vector 核数和 Local Memory，按输出 tile 数选择
有效核数，避免小矩阵盲目满核。首版候选 tile 为 `M,N ∈ {128,64,32,16}`、
`K ∈ {128,64,32,16,8}`，按以下顺序筛选：

1. 满足 A2/A3 Cube 指令的 FP32 对齐和单指令尺寸限制；
2. A/B 双缓冲、累加 tile 和临时向量不超过 L1/L0/UB；
3. 尾块浪费率最小、有效 Cube 面积最大；
4. workspace 最小且各核的 C 写区不重叠。

均分时各核处理相同数量的 `(batch, tileM, tileN)` 工作；不能均分时把余数
分配给前几个核。A/B panel 的 K 循环在同一输出 tile 内完成，避免把中间
结果写回 GM 再加载。

### 3.1.4 TilingKey

TilingKey 只编码实现所需的真实运行时元数据，不编码 CSV case 名或输入值：

```text
bits 1:0   transA/transB family (N/T/C)
bit  2     beta == 0 fast path
bit  3     alpha == 0 or k == 0 scale-only path
bit  4     zero-output no-op (Host returns before launch)
bits 6:5   tiny / regular / long shape family
```

同一 key 下仍使用实际 m/n/k/ld/stride/batch 参数。Key 的枚举顺序不是
所有权证明；Host 单测和 Kernel 入口必须共同验证每条路径的完整边界。

## 3.2 Kernel 侧设计

### 3.2.1 数据流

正常路径采用 `CopyIn -> Cube Mmad -> alpha/beta epilogue -> CopyOut` 的
双缓冲流水：

```text
GM A/B (Column-Major)
        | 读取有效 panel，处理 N/T/C 地址映射
        v
L1/L0 A/B tiles -> FP32 Cube Mmad -> FP32 accumulator
                                      |
                                      v
                         alpha * product + beta * C
                                      |
                                      v
                             GM C logical cells
```

A/B 为 0 stride 时，panel 在各 batch 间复用；C 不允许跨 batch 重叠。Vector
阶段只做标量融合、尾块 mask 和必要的布局搬运，不在 Host 侧执行矩阵乘。

### 3.2.2 快速路径

- `m==0 || n==0 || batchCount==0`：Host 直接返回 SUCCESS，不启动 Kernel。
- `k==0 || alpha==0`：Kernel 不读取 A/B，只按 beta 对每个 C 逻辑元素
  缩放；`beta==0` 用零写回，`beta==1` 可直接结束该 tile。
- `beta==0`：C 旧值不加载，累加器以零初始化。

快速路径必须保留 C padding/gap，且不能因为省略输入读取而改变输出缓冲区
的长度或异步 stream 语义。

### 3.2.3 对齐和尾块安全

GM/Local 地址按 A2/A3 公开搬运要求对齐；不满一个向量块的 M/N/K 使用
mask 或 `DataCopyPad`。每个 load/store 都以实际剩余长度裁剪，不能只在
caller 侧用整除条件保护。写回范围严格为 `validM * validN`，避免覆盖
leading-dimension padding 和相邻 batch。

### 3.2.4 资源和 workspace

Host 以 64 位计算：

```text
storage(A) = strideA == 0 ? lda*stored_cols
                         : strideA*(batchCount-1) + lda*stored_cols
storage(B) = strideB == 0 ? ldb*stored_cols
                         : strideB*(batchCount-1) + ldb*stored_cols
storage(C) = strideC*(batchCount-1) + ldc*n
```

实际分配还要覆盖空 batch/空维规则。workspace 只保存当前 tile 的必要
双缓冲，不保存完整尺寸的 A/B/C 副本；当可用 UB/L1 不足时缩小 tile 或
返回明确资源错误，不能越界运行或切换到 Host/CPU 计算。

## 3.3 支持硬件和限制

| 硬件 | 状态 |
| --- | --- |
| Atlas 800I A2 / 910B3 | 本设计目标，`arch22` |
| Atlas 800I A3 | 本设计目标，`arch22` 兼容路径 |

只承诺本任务书声明的 FP32、Column-Major、N/T/C、非负维度和 A/B 广播
语义。C 重叠写、越界 stride、负 stride 和未声明 dtype 不在合同内；API
负向返回码仍由测试工程覆盖。

## 3.4 与对标实现的差异

| 项目 | cuBLAS/Netlib 对标 | A2/A3 Ascend C 设计 |
| --- | --- | --- |
| 批量寻址 | Host API + long long stride | Handle 参数转 TilingData，Device 计算元素偏移 |
| 核心计算 | GPU GEMM | A2/A3 Cube FP32 Mmad |
| 转置 | N/T/C | 同语义；实数 C 走 T 地址映射 |
| 空维/标量 | BLAS no-op、beta-only | Host/Kernel 快速路径保持同一规则 |
| 内存布局 | Column-Major | 保留 ld/stride padding，尾块 mask 写回 |

差异只来自执行硬件和数据搬运，不改变公开数学、错误码或指针读取条件。

# 四、特性交叉分析

| 特性 | 涉及 | 设计处理 |
| --- | :---: | --- |
| 动态 shape | 是 | m/n/k/batch/ld/stride 运行时读取，不能按 case 分支 |
| 转置 | 是 | N/T/C 三类地址映射，C 在 FP32 下等价 T |
| 广播 | 是 | 仅 A/B 零 stride；复用只读 panel |
| 非紧凑布局 | 是 | 64 位元素偏移和 leading dimension padding |
| 尾块 | 是 | 每次搬运和写回都使用 valid 尺寸 mask |
| 空 Tensor | 是 | m/n/batch 为 0 时无 Kernel 快返 |
| alpha/beta 快速路径 | 是 | beta=0 不读 C，alpha=0/k=0 不读 A/B |
| 异步执行 | 是 | 绑定 handle stream，调用者负责同步 |
| 确定性 | 目标 | 每个输出 tile 单一所有者，避免 C 重叠和未定义竞争 |

# 五、可维可测分析（required）

## 5.1 精度标准

Golden 由 cblas/Netlib `sgemm` 按 batch、列主序逐矩阵生成，覆盖完整
`m×n` C 逻辑区域；padding/gap 也要确认保持不变。FP32 使用生态算子标准：

```text
rtol = 2^-10 = 9.765625e-4
atol = 2^-16 = 1.52587890625e-5
matched_ratio >= 0.99
max_abs_error <= 1e-2 或 32 * ULP
```

逐元素通过条件为 `abs(actual-golden) <= atol + rtol*abs(golden)`。Inf/NaN
只接受与 golden 的同类精确匹配；不能用放宽阈值或删除特殊用例来掩盖错误。
负向 API 行按预期状态码判断，不混入数值精度比率。

## 5.2 性能和内存标准

性能测试设备为 Atlas 800I A2/A3，先 warmup，再对每条 `TC_PF` 完成超过
50 次有效采样，报告 Avg time（us）及必要的 Device profiler 数据。任务书
列出的三个典型门槛为：

| transA | transB | m | n | k | batchCount | 任务书表值（us） |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| N | N | 1024 | 1024 | 1024 | 16 | 1858.46 |
| N | N | 2048 | 2048 | 2048 | 8 | 3697.51 |
| N | N | 4096 | 4096 | 4096 | 4 | 5781.46 |

资料包 `gpu_baseline.csv` 的对应原始值以 `gpu_ms` 记录（例如前三行
2.323070、4.621890、7.226820 ms），验证脚本另使用 `gpu_ms/npu_ms >= 0.8`
的比率表述。两种来源在单位/公式上不能静默合并：代码验收前必须由测试
工程维护者明确“表值是阈值还是转换后的参考值”，并在自测报告中同时列出
原始值、单位、采样次数和最终判定。本文不把资料包中的 GPU 数字冒充 A2/A3
实测，也不在设计阶段宣称性能通过。

任务书 §3.3 的第三个典型 case 是 `4096x4096x4096/batchCount=4`；资料包
`gen_csv.py` 的 `DOC_CASES` 和生成 CSV 当前写成 `batchCount=1`。本设计以
任务书表格作为验收目标，同时保留 CSV 的原始行用于语料追踪；代码验收前
必须由测试工程维护者修正或明确这一个 descriptor，不能把两个 case 静默
当成同一性能门槛。

内存验收记录 Host 侧峰值，单 case 不超过 512 MiB；A/B/C 的 batch、ld 和
stride 计算必须计入总量。超预算用例应保留并由测试工具报告明确的资源
结果，不能通过删行或缩小 shape 改写任务合同。

## 5.3 自验命令和交付证据

在 `ops-blas` 仓中安装资料包 CSV 后，使用任务提供的脚本：

```bash
python gen_csv.py
python verify_accuracy.py --repo <ops-blas> --soc ascend910b3 \
  --csv gemm_strided_batched_test.csv --timeout 3600
python verify_performance.py --repo <ops-blas> --soc ascend910b3 --timeout 3600
```

需要分别在 A2 和 A3 完成功能、负向 API、200 条性能/内存、采样次数和
报告截图/日志验证。最终代码 PR 还需包含 ops-blas 规定的算子 README、
测试工程和 `include/cann_ops_blas.h` 公共接口变更（如主线尚未包含）。

# 六、兼容性与无回退边界

1. 只新增/补齐 A2/A3 `arch22` 实现，不定义第二个产品私有 API。
2. 公共 handle、stream、异步提交和状态码保持不变。
3. Candidate 计算必须全部在当前 ops-blas Ascend C 源码和生成的 Device
   binary 中完成；禁止调用官方整算子、Torch/Torch-NPU、CPU/cblas golden、
   GPU baseline、其他 backend、外部预编译 binary 或 peer solution。
4. Host 侧不逐元素计算；不支持的参数必须返回明确错误，不得静默切换
   到 fallback。
5. 公开 CSV 的 case 名、随机种子、顺序、性能值和答案模式只能用于测试
   追踪，不能成为 kernel dispatch 条件。

# 七、CheckList 覆盖映射

| 审核项 | 本文位置 |
| --- | --- |
| 任务来源、硬件、版本、目标目录 | 1.1、文档头 |
| API 原型、参数和错误码 | 1.2.2 |
| CSV 角色和完整覆盖 | 1.3 |
| 需求拆解和模块边界 | 2.1-2.4 |
| Host 校验、分核、tiling、Key | 3.1 |
| Kernel 数据流、快路径、尾块、workspace | 3.2 |
| 硬件和限制 | 3.3 |
| 对标差异和特性交叉 | 3.4、第四章 |
| 精度、性能、内存和自验 | 第五章 |
| 兼容性和 no-fallback | 第六章 |

# 参考资料

1. 资料包任务书：`aclblasSgemmStridedBatched_Atlas800IA3_task_doc.md`。
2. cuBLAS `cublasSgemmStridedBatched`：
   <https://docs.nvidia.com/cuda/cublas/index.html#cublassgemmstridedbatched>
3. Netlib BLAS `sgemm`：<https://www.netlib.org/blas/sgemm.f>
4. ops-blas：<https://gitcode.com/cann/ops-blas>
5. 公共 API 头文件：
   <https://gitcode.com/cann/ops-blas/blob/master/include/cann_ops_blas.h>
6. CANN FP32 精度标准：
   <https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md>
