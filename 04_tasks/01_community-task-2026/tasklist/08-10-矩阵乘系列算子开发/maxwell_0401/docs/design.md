# 【社区任务】aclblasSgemmGroupedBatched 算子设计文档

| 项目 | 内容 |
| --- | --- |
| 任务编号 | 08-10-矩阵乘系列算子开发 |
| 团队名称 | maxwell_0401 |
| 适配硬件 | Atlas 800I A2 / Atlas 800I A3（910B3/A3，`arch22`） |
| CANN 版本 | 9.1.0 |
| 开发语言 | Ascend C / CATLASS |
| 目标仓库 | <https://gitcode.com/cann/ops-blas> |
| 目标目录 | `blas/gemm_grouped_batched/arch22/` |

## 设计文档阶段边界

本文只提交算子设计，不包含算子实现、二进制、设备自测结果或性能通过
结论。代码阶段再向 `cann/ops-blas` 提交
`blas/gemm_grouped_batched/arch22/` 和对应测试目录
`test/gemm_grouped_batched/sgemm_grouped_batched/arch22/`；两阶段必须保持
同一接口、数据布局、错误码和精度口径。

## 资料包冻结

设计依据为仓库资料包
`docs/qsj/aclblasSgemmGroupedBatched.zip`，ZIP SHA256 为
`e71a70ac7f32d774257911bf656baf234dfbf6b924713f4b0c97424c8bff6d33`。
资料包包含任务书、CSV 生成器、GPU 参考数据、测试说明和精度验证脚本；
它们是设计和后续验收的输入，不是运行时依赖。

| 成员 | SHA256 |
| --- | --- |
| `aclblasSgemmGroupedBatched_A2A3_task_doc.md` | `eaebad32e5fbb850edd875a2f91db0d22261e343bde8fd141df0d8f906071629` |
| `test_cases/gen_csv.py` | `6a6bb1809746aba3be55a7d823cad7bcece3623aeca34fe5945fe5fcc0d76c3c` |
| `test_cases/gpu_baseline.csv` | `2aee9a94072570885111e3453e423912ea90db8cb2107593f37dd627bf51abe7` |
| `test_cases/README.md` | `c6e666a883993391caa87760e96869926c9bdd3b366b7e53ba81b2c8a6d9b3e9` |
| `test_cases/sgemm_grouped_batched_test.csv` | `5405d643f1eb3784e8adc9fcefbe546af4db10746991d905fce4b94f1625ce9e` |
| `test_cases/verify_accuracy.py` | `ff1324edd42104e33b48f1df9258babf8020ccc7170ac7d7d539e6b1b1348060` |

## 一、需求背景（required）

## 1.1 需求来源

本设计依据 2026 年 9 月社区任务
`aclblasSgemmGroupedBatched_A2A3` 任务书及其测试目录。目标是在
Atlas A2/A3 上新增 `arch22` 实现，完成设计、开发和自验证后合入
`cann/ops-blas`。数学和接口语义对标 cuBLAS
`cublasSgemmGroupedBatched`，逐组正确性参考使用 Netlib CBLAS `sgemm`。

## 1.2 功能与数学语义

算子把 `groupCount` 个形状彼此独立的 GEMM 组打包到一次异步调用中。第
`g` 组有独立的 `m_g/n_g/k_g`、转置、`alpha_g/beta_g` 和
`groupSize_g`。组内第 `j` 个矩阵乘的扁平下标为：

```text
idx = sum(groupSize[t] for t in [0, g)) + j
```

其计算为：

```text
C[idx] = alpha_g * op(A[idx]) * op(B[idx]) + beta_g * C[idx]
```

其中 `op(X)` 由逐组 `transaArray[g]`/`transbArray[g]` 选择：

| 枚举 | 实数 FP32 语义 |
| --- | --- |
| `ACLBLAS_OP_N` | `X` |
| `ACLBLAS_OP_T` | `X^T` |
| 其他值（包括 `ACLBLAS_OP_C`） | 非法，返回 `ACLBLAS_STATUS_INVALID_VALUE` |

所有矩阵均为 FP32、Column-Major。A/B/C 是 Host 侧指针数组，数组元素是
Device 矩阵地址；不存在 stride 或 Device 侧指针数组语义。一个组的矩阵
布局为：

```text
transa=N: A physical = lda x k       transa=T: A physical = lda x m
transb=N: B physical = ldb x n       transb=T: B physical = ldb x k
C physical = ldc x n
```

逻辑元素的列主序地址分别为 `base + column * ld + row`。组间的形状、
前导维度、标量和 batch 数量都不可互相借用。

## 1.3 API 原型

接口必须与 `include/cann_ops_blas.h` 中的公共声明完全一致：

```cpp
aclblasStatus_t aclblasSgemmGroupedBatched(
    aclblasHandle_t handle, int groupCount,
    const aclblasOperation_t* transaArray,
    const aclblasOperation_t* transbArray,
    const int* mArray, const int* nArray, const int* kArray,
    const float* alphaArray,
    const float* const* Aarray, const int* ldaArray,
    const float* const* Barray, const int* ldbArray,
    const float* betaArray,
    float* const* Carray, const int* ldcArray,
    const int* groupSizeArray);
```

调用链为：

```text
aclInit -> aclblasCreate -> aclblasSetStream
       -> aclblasSgemmGroupedBatched -> caller synchronizes the bound stream
```

调用在 handle 绑定的 stream 上异步提交。接口不把 Device 结果同步回 Host，
调用方在读回 C 前负责同步该 stream。

## 二、需求分析（required）

## 2.1 参数合同与错误行为

| 参数 | 合同与边界 |
| --- | --- |
| `handle` | 有效 Host 句柄；`nullptr` 返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| `groupCount` | `>=0`；负值返回 `ACLBLAS_STATUS_INVALID_VALUE`；0 为合法 no-op |
| `transaArray/transbArray` | `groupCount>0` 时长度为 `groupCount`，每项仅 N/T；空指针或非法枚举返回 INVALID_VALUE |
| `mArray/nArray/kArray` | `groupCount>0` 时长度为 `groupCount`，每项非负；负值返回 INVALID_VALUE |
| `alphaArray/betaArray` | `groupCount>0` 时为长度 `groupCount` 的 Host FP32 标量数组；空指针返回 INVALID_VALUE |
| `Aarray/Barray/Carray` | 扁平 Host 指针数组，长度 `sum(groupSizeArray)`；需要读取/写入的地址不可为空 |
| `ldaArray/ldbArray/ldcArray` | Column-Major 前导维度；不满足下述最小值返回 INVALID_VALUE |
| `groupSizeArray` | `groupCount>0` 时长度为 `groupCount`，每项非负；负值返回 INVALID_VALUE |

前导维度约束为：

```text
lda[g] >= max(1, transa[g] == N ? m[g] : k[g])
ldb[g] >= max(1, transb[g] == N ? k[g] : n[g])
ldc[g] >= max(1, m[g])
```

对需要执行乘法的 batch，A/B/C 的元素指针必须有效；对合法 no-op 或
`alpha=0`/`k=0` 的不读路径，设计不额外读取不需要的 A/B 数据。C 与 A/B
内存重叠不在合同内，越界指针和 Device 侧指针数组不在合同内。

## 2.2 空维、标量和异步规则

- `m[g] == 0`、`n[g] == 0` 或 `groupSize[g] == 0`：该组合法 no-op，不
  启动该组计算；所有组均为空时返回成功。
- `groupCount == 0`：合法 no-op，返回成功。
- `k[g] == 0`：该组等价于 `C = beta[g] * C`，不读取 A/B。
- `alpha[g] == 0`：跳过矩阵乘，仅执行 `C = beta[g] * C`，不读取 A/B。
- `beta[g] == 0`：输出 epilogue 不读取旧 C，直接以零初始化累加器。
- `beta[g] == 1`：仍须遵守正常 FP32 累加和写回语义；只有经过同一边界
  证明的快速路径才可省略多余操作。
- 任何非空计算仍依赖 `aclblasSetStream`，不以 Host 同步替代异步合同。

## 2.3 资料包用例角色

`gen_csv.py` 默认固定种子生成 1200 条用例：1000 条精度、200 条性能。
精度集合覆盖组间异构形状、N/T 正交组合、批量大小、前导维 padding、随机
及特殊值、零维和负向参数；性能集合覆盖任务书典型 case、小尺寸多组、组间
异构、转置组合、矩形和大尺寸混合。用例编号和 CSV 行号只用于测试追踪，
不能成为 kernel 的分支条件或支持边界。

## 三、详细设计（required）

## 3.1 模块边界与调用流程

代码阶段保持 ops-blas 的 kernel 直调工程模式：

| 模块 | 计划目录 | 职责 |
| --- | --- | --- |
| 公共接口 | `include/cann_ops_blas.h` | 复用/补齐唯一公共声明和状态码 |
| Host API 与校验 | `blas/gemm_grouped_batched/arch22/` | 参数校验、组描述整理、workspace 与 launch |
| Tiling | `blas/gemm_grouped_batched/arch22/` | 资源查询、组/矩阵 tile 规划、kernel 参数编码 |
| Device kernel | `blas/gemm_grouped_batched/arch22/` | A/B 搬运、Cube FP32 Mmad、alpha/beta 和 C 写回 |
| 测试工程 | `test/gemm_grouped_batched/sgemm_grouped_batched/arch22/` | CSV loader、CBLAS golden、GTest、负向和 profiler 接口 |

Host 侧不逐元素计算，不调用官方整算子。测试侧的 CBLAS 只生成 golden，
不进入 candidate 运行时。

## 3.2 Host 侧参数校验与组描述

校验顺序固定为：

1. 校验 handle 和 `groupCount`；`groupCount==0` 立即返回 SUCCESS，不读取
   其他数组；
2. 对 `groupCount>0` 的调用校验所有 group-level 数组指针；
3. 校验每组转置枚举、非负维度、非负 `groupSize` 和最小 `ld`；
4. 用 64 位前缀和计算每组扁平 batch 起点，并检查总 batch 数可表示；
5. 识别空组、全空组和只缩放组，决定是否需要读取 A/B/C 指针数组；
6. 对实际需要的每个扁平地址执行非空检查，计算 workspace/描述表容量；
7. 校验元素计数、字节数和地址偏移的 64 位溢出，再生成 launch 描述。

组描述表至少包含 `m,n,k,lda,ldb,ldc,alpha,beta,transa,transb,groupSize`
及扁平 batch 起点。描述表由 Host 在 workspace 中按调用生成，Device 只读取
当前调用的内容；不从 task 文件、CSV 或答案缓存加载元数据。

所有前缀和、矩阵偏移、字节数和 workspace 乘积先提升到 `int64_t/uint64_t`
计算，溢出时返回明确资源/参数错误。组间独立 descriptor 让不同尺寸和
转置可以共存，不把第一组参数广播到其他组。

## 3.3 分组调度与 tiling

调度单位是 `(group, batch_in_group, tile_m, tile_n)`。Host 根据实际组
元数据形成连续工作描述，并按可用 AI Core 数量和 tile 数量选择有效核数：

1. 优先让一个输出 tile 由一个核独占，避免 C 写冲突；
2. 在同一输出 tile 内沿 K 循环复用 A/B panel，不采用 split-K 原子累加；
3. 不足整除时把余数分给前面的工作单元，保持不同组之间负载可控；
4. 对 tiny、regular、large 仅按运行时的尺寸、对齐、dtype 和资源预算
   选择 tile，不按公开 case 或精确 shape 列表选择；
5. 当某组形状不适合 Cube tile 时，用同一 Ascend C solution 内的合法
   tail/小 tile 路径承接，不能调用 CPU、Torch 或官方 whole-op fallback。

候选 tile 的最终值由 A2/A3 的公共编译能力和平台 API 查询结果确定。初始
设计保留 `M/N/K` 的多级候选（例如 128/64/32/16 的输出边界和 128/64/32
的 K 分块），每次只改变一个调度变量；实际工程不得把这些候选误写成封闭
的支持 shape。每个 tile 都必须同时计入 L0A/L0B/L0C、L1/UB、对齐 padding、
双缓冲和尾块 scratch 的总量。

## 3.4 TilingData 与 dispatch identity

TilingData 保存调用级组表位置、组数、总 batch 数、工作单元范围、tile 参数、
workspace 大小和实际 `trans`/标量路径信息。若需要 TilingKey，只编码合法的
运行时元数据族：

```text
family: transa/transb (N/T)  |  tiny/regular/large runtime size class
path:   normal GEMM          |  beta-only / alpha-zero-or-k-zero
tail:   aligned              |  M/N/K tail requiring mask
```

这些 key 只缩短同一语义实现的 dispatch；完整尺寸、leading dimension、组
起点和标量仍由 TilingData 传入。selector 顺序不是所有权证明，所有分支都
必须对其 guard 下的完整合法值域正确。

## 3.5 Device kernel 数据流

正常路径为：

```text
Host group descriptors -> workspace descriptor table
GM A/B -> L1/L0A/L0B -> Cube FP32 Mmad -> FP32 accumulator
                                            |
                              alpha * product + beta * old C
                                            v
                                        GM C tile
```

- A/B panel 读取按组自己的 `trans`、`ld` 和扁平 batch 地址计算；列主序
  padding 不参加数学运算。
- K 循环只在当前输出 tile 内累加；不把部分和写回 GM 后再跨核合并。
- 结果写回严格限制在 `valid_m x valid_n`，不覆盖 `ldc` padding 或相邻
  batch；同一 C tile 只有一个工作单元写入。
- `alpha=0` 或 `k=0` 路径不加载 A/B；`beta=0` 路径不加载旧 C。
- `m/n`、组 batch 或 tile 尾部不整除时，每次 GM load/store 都以运行时
  valid extent 做 mask/DataCopyPad，不能由 caller 的整除条件代替。

## 3.6 流水、内存与同步

A2/A3 路径只采用 910-class 已证明的 GM/MTE、L1/L0、UB、Cube、Vector 和
Scalar/control 机制，不引入 950 专属 SIMT、NDDMA、STARS2.0 或跨 Die
通信假设。正常 tile 使用有限深度的 CopyIn/Cube/epilogue/CopyOut 重叠；
每个队列、event、buffer 都有明确的 producer、consumer、释放和尾部 drain。

| 资源 | 设计用途 | 预算要求 |
| --- | --- | --- |
| GM/HBM | A/B/C 与调用级 descriptor table | 64 位偏移；不物化完整矩阵副本 |
| L1/L0A/L0B/L0C | Cube 输入 panel 与累加 tile | 以实际 SoC/CANN 查询容量，留编译器和 ABI 余量 |
| UB | 标量融合、尾块 mask、必要布局搬运 | 计入输入/输出队列、临时、tail scratch 和双缓冲 |
| workspace | group descriptor、调度工作表和有限中间缓存 | Host 计算并检查上界，不能静默扩容或越界 |

不同组之间不共享可变累加器。若未来引入跨核 producer/consumer，必须先闭合
work identity、payload 所有权、event 编码、生命周期和数值顺序，再纳入
实现；当前设计用单核 tile owner 避免 split-K 同步复杂度。

## 四、支持范围、边界与对标差异

## 4.1 支持硬件

| 硬件 | 目标 |
| --- | --- |
| Atlas 800I A2 / 910B3 | `arch22`，主性能设备 |
| Atlas 800I A3 | `arch22`，功能和验收设备 |

精确 core 数、UB/L0 容量、编译器 lowering 和可用模板以选定设备及 CANN
9.1.0 的平台查询/编译证据为准，不在设计文档中硬编码 SKU 规格。

## 4.2 约束

- 仅支持 FP32、Column-Major、逐组 N/T 转置和任务书声明的非负维度。
- A/B/C 指针数组必须在 Host 侧，元素指向 Device 矩阵；不支持 Device
  侧指针数组、额外 stride/broadcast 约定和 C 与输入重叠。
- 组间形状、标量、转置和 batch 完全独立；不能按第一组参数推断后续组。
- 浮点累加顺序不要求逐位确定，但每个合法输出必须满足规定混合容差。
- 未支持的参数、非法枚举、负维度、负 batch 或非法 leading dimension
  必须返回明确状态码，不得降级到其他实现。

## 4.3 与 cuBLAS 的非语义差异

本接口把 `groupCount` 放在 handle 后面，而 cuBLAS 将其放在参数末尾；
本接口采用 `mArray/nArray/kArray/groupSizeArray` 命名。两者的逐组数组、
列主序、转置、alpha/beta 和扁平指针数组语义相同，不增加 stride 解释。

## 五、可维可测分析（required）

## 5.1 精度标准

Golden 由 Netlib CBLAS `sgemm` 按组、按 batch、按 Column-Major 生成，对每个
有效 C 矩阵的完整 `m x n` 区域比较。FP32 采用任务书的混合容差：

```text
rtol = 2^-13 ≈ 1.2207e-4
atol = 2^-13 ≈ 1.2207e-4
matched_ratio >= 0.99
max_abs_error <= 1e-2 或 32 * ULP
```

逐元素判定为 `abs(actual - golden) <= atol + rtol * abs(golden)`；Inf/NaN
场景必须与 golden 的类别和位置一致，不能靠放宽阈值或删除用例掩盖错误。
负向用例按预期状态码验证，不混入数值匹配率。

## 5.2 任务书性能目标

性能目标是 Atlas 800T A2（910B3）上的 FLOAT32 平均单次耗时，先完成
msprof 自带 warmup，再做有效采样；任务书给出的五个典型门槛如下：

| case | groupCount | groupSize | m/n/k | transa/transb | 达标耗时（us） |
| --- | ---: | --- | --- | --- | ---: |
| 1 | 2 | [64,64] | [256,256,256] | [N,N]/[N,N] | 351.3 |
| 2 | 2 | [128,64] | [512,512,512] | [N,N]/[N,N] | 3714 |
| 3 | 3 | [64,128,64] | [1024,1024,1024] | [N,N,N]/[T,T,T] | 37652 |
| 4 | 2 | [128,128] | [2048,2048,2048] | [T,T]/[N,N] | 298380 |
| 5 | 2 | [64,64] | [4096,4096,4096] | [N,N]/[N,N] | 1188440 |

设计阶段不宣称上述门槛已达到。代码阶段必须保留全部 `TC_PF` 性能行，
按测试说明获取 NPU kernel 平均耗时，并同时记录设备、CANN、warmup、有效
采样数、单位和 profiler 原始路径。`gpu_baseline.csv` 只能作为对标输入，
不能冒充 A2/A3 设备执行证据。

## 5.3 自验证计划

代码阶段在 `ops-blas` 测试工程内复用资料包脚本和 CSV：

```bash
python gen_csv.py
python verify_accuracy.py --repo <ops-blas> --soc ascend910b3 \
  --csv sgemm_grouped_batched_test.csv --timeout 3600
```

精度计划至少逐项保留以下覆盖：

| 类别 | 必须覆盖 |
| --- | --- |
| 基础/shape | 1 组及多组、异构 m/n/k、质数、2 的幂及非对齐尾块 |
| 转置 | 每组独立 N/T，含组间异构组合 |
| batch/布局 | 0/1/中大 batch、逐组 lda/ldb/ldc padding |
| 标量 | alpha/beta 的 0、1、-1、随机和组间差异 |
| 边界 | groupCount=0、零维、零 batch、k=0、非法枚举/维度/ld/空指针 |
| 数据值 | 均匀/正态、全零、正负交替、Inf/NaN（按任务书规则） |

性能阶段使用全部 200 条 `TC_PF`，先 warmup 后完成有效采样，使用 msprof
`OpBasicInfo.csv` 或仓库最新等价工具获取 kernel 明细；不以 Host 端到端
时间替代 kernel 口径。内存阶段记录 A/B/C、指针数组、descriptor 和
workspace 的峰值，不通过删减大 shape 逃避预算。

交付报告须分别列出精度 case 数/失败数、每条性能 case 的参数和平均值、
采样协议、内存数据、设备身份和原始日志/截图路径。缺少设备实测时只能
标记为未完成，不能写成通过。

## 六、兼容性与 no-fallback 边界

1. 计算路径只允许当前 `cann/ops-blas` `arch22` Ascend C/CATLASS 源码和
   其由仓库构建生成的 Device binary。
2. 禁止调用 cuBLAS、官方 whole-op、ACLNN/Torch/Torch-NPU 等等价整算子，
   禁止 CPU/CBLAS golden、GPU baseline、其他 backend、peer workspace、
   任务 golden 或预编译外部 binary 作为 candidate 输出。
3. CANN/ACL/runtime 和公开 CATLASS 模板只能作为普通编译/运行依赖；算子
   专属 kernel、tiling、descriptor、dispatch 和打包代码必须由目标仓自有
   源码生成。
4. runtime guard 只能在同一 Ascend C 实现内选择对完整 guard 值域正确的
   路径；不支持的输入直接返回错误，不能静默 fallback。
5. 公开 case id、CSV 行号、文件名、随机种子、答案模式、性能值和观察到的
   输入内容不得参与 dispatch，也不得固化成 shape 白名单。
6. 设计中的组表、tile 和 workspace 仅为计划结构；实现阶段须用实际
   CANN/SoC 能力和源码/编译/设备证据闭合每个资源与同步假设。

## 七、审查清单映射

| 审核项 | 本文位置 |
| --- | --- |
| 任务来源、硬件、版本、资料包哈希 | 文档头、资料包冻结、1.1 |
| API 原型、数组语义、错误码 | 1.2、1.3、2.1 |
| 空维、alpha/beta、异步合同 | 2.2 |
| 模块边界和 Host 校验 | 3.1、3.2 |
| 分组调度、tiling 和 dispatch | 3.3、3.4 |
| Cube/Vector 数据流、尾块、内存和同步 | 3.5、3.6 |
| A2/A3 范围及对标差异 | 第四章 |
| 精度、性能、内存与自验计划 | 第五章 |
| no-fallback 和 benchmark hacking 边界 | 第六章 |

## 参考资料

1. `aclblasSgemmGroupedBatched_A2A3_task_doc.md`（资料包内，SHA256 见上）。
2. `test_cases/README.md`、`gen_csv.py`、`sgemm_grouped_batched_test.csv` 和
   `verify_accuracy.py`（资料包内，SHA256 见上）。
3. cuBLAS `cublasSgemmGroupedBatched`：
   <https://docs.nvidia.com/cuda/cublas/index.html>。
4. Netlib BLAS `sgemm`：<https://www.netlib.org/blas/sgemm.f>。
5. ops-blas：<https://gitcode.com/cann/ops-blas>。
6. CANN 混合容差标准：
   <https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/mixed_tolerance_standard.md>。
