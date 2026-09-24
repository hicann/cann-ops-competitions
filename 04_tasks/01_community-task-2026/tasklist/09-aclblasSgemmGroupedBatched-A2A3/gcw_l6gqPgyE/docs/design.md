# aclblasSgemmGroupedBatched 算子开发设计文档（Atlas A2/A3）

| 项目 | 内容 |
| --- | --- |
| 任务名称 | 9月社区任务-aclblasSgemmGroupedBatched算子开发（A2/A3） |
| 提交团队 | gcw_l6gqPgyE |
| 文档版本 | V1.0，2026-09-23，开发前设计评审稿 |
| 目标环境 | Atlas A2/A3 系列产品，CANN 9.1.0，Ascend C kernel 直调 |
| 实现仓库 | `cann/ops-blas`，`blas/gemm_grouped_batched/arch22/` |
| 源码分析基线 | ops-blas `2c9b9b77bda979c25a6b89726614b082e73226fa` |
| 文档性质 | 定义接口契约、实现方案、测试方法与交付门槛；功能和性能指标为验收目标，尚无本任务 NPU 实测结论 |

## 1 需求背景（required）

### 1.1 需求来源

依据《aclblasSgemmGroupedBatched_A2A3 任务书》及随附 `test_cases/` 中的 CSV、自测指导、生成器和精度脚本，在既有 ops-blas 句柄式 BLAS 工程中补齐 Atlas A2/A3 的单精度分组批量矩阵乘能力。设计文档按社区模板提交到 competitions 仓库，经设计评审后开展实现、自测及代码合入。

归档任务书 SHA256：`eaebad32e5fbb850edd875a2f91db0d22261e343bde8fd141df0d8f906071629`。压缩包内任务书与独立提供的 Markdown 内容一致。

### 1.2 背景介绍与标杆分析

分组批量 GEMM 用一次接口调用表达若干形状、转置、标量和 batch 数均独立的矩阵乘运算，减少逐矩阵提交开销，并提高异构小矩阵的设备利用率。标杆为 cuBLAS `cublasSgemmGroupedBatched`；本任务的 FP32、列主序、逐组参数和扁平指针数组语义以任务书为直接契约，不扩大为 strided-batched、广播或复数接口。

对 `g∈[0,groupCount)`、`j∈[0,groupSizeArray[g])`：

```text
prefix[0] = 0
prefix[g+1] = prefix[g] + groupSizeArray[g]
idx = prefix[g] + j
C[idx] = alphaArray[g] × op(A[idx]) × op(B[idx]) + betaArray[g] × C[idx]
```

标杆的逻辑计算流程为：接收逐组元数据 → 根据 batch 前缀定位 A/B/C → 按组执行相应转置与 GEMM → 按各组 alpha/beta 更新 C。该流程描述接口语义，不推断 cuBLAS 内部调度实现。

### 1.3 ops-blas 实现现状

分析基线已经在 `include/cann_ops_blas.h` 声明相同接口，在 `blas/gemm_grouped_batched/arch35/` 提供 Host、Kernel、Tiling 三个实现文件。README 当前仅标注 950 支持，A2/A3 不支持；当前任务新增 `arch22` 后更新产品支持信息。

现有 Host 完成元数据校验、逐组 tiling、将 Host 指针数组转换为 Device 可读地址表，在 handle 工作区内布置参数并启动 Kernel。现有 Kernel 使用 AIV 与 `AscendC::MicroAPI`，按 batch 分配计算，包含列方向累加和点积路径；UB 常量为 248 KiB。现状流程如下：

```text
aclblasSgemmGroupedBatched
  → ValidateCommonParams / ValidateGroupParams
  → BuildGroupParams / FlattenHostPtrArrays
  → 按 batch 数分核、计算工作区布局
  → Host 参数与地址表拷贝至工作区
  → arch35 AIV Kernel → 按组查参 → GEMM 或 beta-only → C
```

`arch35` 的指令、UB 常量及分核策略不能直接视作 A2/A3 可用实现。本任务复用公共 ABI、handle、构建和测试基础设施，采用面向 `arch22` 的资源查询与计算实现；不在本次设计提交中修改 950 算子行为。

## 2 需求分析（required）

### 2.1 外部组件与内部适配

| 组件 | 使用方式与适配边界 |
| --- | --- |
| CANN 9.1.0 / Ascend C | 使用 ACL stream、内存与事件接口，以及 A2/A3 支持的 Matmul/Mmad、Vector 和搬运能力 |
| ops-blas 公共 ABI | 保持 `include/cann_ops_blas.h` 中既有声明；不新增同名重载或更改参数位置 |
| handle / workspace | 复用 `blas/common/helper/aclblas_handle_internal.h`，遵守库所有与用户所有工作区的生命周期 |
| 构建系统 | 在既有 `arch22` 选择机制内接入；A2 为 `ascend910b*`，A3 为 `ascend910_93*`，使用设备实际 SoC 型号 |
| GTest / CSV | 接入仓内测试框架，扩展本算子参数解析、负向注入与 Mixed Tolerance 判定 |
| Netlib BLAS / CBLAS | 验收 Golden 使用任务指定的 Netlib `cblas_sgemm`；记录实际动态链接库、版本和编译选项 |
| msprof | 采集每次完整接口调用对应的全部计算 Kernel，形成逐 case 的设备耗时报告 |

已有测试 CMake 查找 OpenBLAS，因此不能仅依据调用了 `cblas_sgemm` 就认定 Golden 是 Netlib。验收测试目标需显式关联任务指定实现并核验加载路径。

### 2.2 接口与数据契约

```cpp
aclblasStatus_t aclblasSgemmGroupedBatched(
    aclblasHandle_t handle, int groupCount,
    const aclblasOperation_t* transaArray,
    const aclblasOperation_t* transbArray,
    const int* mArray, const int* nArray, const int* kArray,
    const float* alphaArray, const float* const* Aarray,
    const int* ldaArray, const float* const* Barray,
    const int* ldbArray, const float* betaArray,
    float* const* Carray, const int* ldcArray,
    const int* groupSizeArray);
```

| 对象 | 位置、类型和语义 |
| --- | --- |
| handle | 已创建的 Host 句柄，执行 stream 由 `aclblasSetStream` 绑定 |
| groupCount、groupSizeArray | 非负 int；groupSize 数组长度为 groupCount；batch 总数使用受检 64 位累加 |
| transaArray、transbArray | Host 数组，每组分别取 OP_N 或 OP_T；OP_C 及其他非法枚举按任务要求返回 INVALID_VALUE |
| mArray、nArray、kArray | Host 非负 int 数组；分别定义逻辑 M/N/K，各组独立 |
| alphaArray、betaArray | Host FP32 值数组，每组一个标量；不依据 handle 标量指针模式把它们解释成 Device 数组 |
| Aarray、Barray、Carray | Host 扁平指针数组，长度为 batch 总数，元素为独立 Device 矩阵地址；不要求矩阵彼此连续 |
| A/B/C 数据 | FP32、ND、列主序；输入 A/B 只读，C 为输入输出；C 不与 A/B 重叠，各输出矩阵写区互不重叠 |
| leading dimension | A：N 时 lda≥max(1,M)，T 时 lda≥max(1,K)；B：N 时 ldb≥max(1,K)，T 时 ldb≥max(1,N)；C：ldc≥max(1,M) |

矩阵物理存储范围分别为 `lda×(transa=N ? K : M)`、`ldb×(transb=N ? N : K)` 和 `ldc×N` 个 float。列内有效行之外的 padding 不参与数值判定，并通过哨兵检查验证未被修改。

不提供 stride 参数、自动广播或通用非连续 Tensor 视图。多个 A/B 指针可指向调用者安排的只读数据，但接口不会自动推导广播。调用者保证矩阵分配范围、Host 数组可读性和地址有效性；普通 C 指针签名不能验证实际数组长度或识别所有悬空地址。

### 2.3 边界、错误与校验顺序

| 场景 | 设计行为 |
| --- | --- |
| handle=nullptr | 优先返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| groupCount<0 | 返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| groupCount=0 | 有效 handle 下直接 SUCCESS，不读取任何数组、不启动 Kernel |
| groupCount>0 且任一必需数组为空 | INVALID_VALUE；包括转置、尺寸、标量、leading dimension、groupSize 以及三张指针表 |
| 任一组负维度、负 batch、非法枚举或 leading dimension 不足 | INVALID_VALUE；所有组完成校验后再下发，避免后组非法时前组已经改写 C |
| M=0、N=0 或 groupSize=0 | 参数校验通过后该组无计算、无 C 写入；保留原始 batch 前缀，不把其后各组错误前移 |
| K=0 或 alpha=0 | 不读取 A/B 矩阵数据；执行 beta-only 分支 |
| beta-only 且 beta=0 / 1 / 其他值 | 分别直接写零、不写 C、按 FP32 缩放 C；beta=0 不读取 C 旧值 |
| 普通 GEMM 且 beta=0 | epilogue 只写 alpha×乘积，避免读取未初始化或 NaN 的 C 旧值 |
| 活跃组 A/B/C 某元素地址为空 | INVALID_VALUE；alpha=0/K=0 时仍按任务参数表校验地址非空，但不读取 A/B 内容 |
| 前缀、tile 数、字节数或地址区间计算溢出 | 在下发前返回 INVALID_VALUE，不进行整数截断 |
| 工作区不足、运行时分配或下发失败 | 按公共状态定义报告 ALLOC_FAILED / EXECUTION_FAILED / INTERNAL_ERROR，并记录失败阶段 |

对 no-op 组，本稿采用严格元数据校验：`groupCount>0` 时仍要求逐组枚举和 leading dimension 合法；对其存在的指针元素采用任务参数表的非空规则。`groupCount=0` 是明确的数组校验豁免。此优先级纳入评审，避免沿用 arch35 对 `groupSize=0` 提前跳过枚举校验的隐式行为。

### 2.4 任务资料与源码差异处置

| 编号 | 已核实差异 | 本任务处置 |
| --- | --- | --- |
| D1 | arch35 对非法转置返回 INVALID_ENUM；任务 CSV 期望 INVALID_VALUE | arch22 遵循任务书；负向用例直接检查真实接口返回值；跨架构统一由维护者评审 |
| D2 | 测试 README 分类数量与 CSV 不同，并残留 stride/broadcast 描述 | 以归档 CSV 的实际 1,200 行为基础，分类见 4.2；不引入未要求的 stride/broadcast 功能 |
| D3 | CSV 增加三列 Aarray_null/Barray_null/Carray_null；现有参数类未消费这些列 | 显式解析并注入真实接口；禁止忽略列后将负向 case 计为已覆盖 |
| D4 | 任务采用 Mixed Tolerance；arch35 使用 MERE/MARE 且存在阈值≤0跳过比较路径 | arch22 明确配置 Mixed Tolerance，所有非空正向 case 必须比较，不能因缺少旧阈值列而跳过 |
| D5 | 任务代码目录为 `test/gemm_grouped_batched/sgemm_grouped_batched/arch22/`；脚本与当前 arch35 使用较浅目录 | 新增任务指定目录并适配 CMake、CSV 定位和二进制发现，保留 arch35 测试入口 |
| D6 | 生成器实际只有均匀输入，正态混合尚为预留 | 补充固定种子的均匀/正态各 50% 输入与标量采样，单列补充 case，不改写原 case 身份 |
| D7 | 正文性能要求写 10 次，交付与注意事项写有效采样>10 次 | 采用 5 次预热后 20 次有效采样；保存全部原始记录并计算平均值 |
| D8 | 自验证允许一种机型，最终要求 A2/A3 均验收 | 自验优先 910B3，分别列出 A2/A3 证据与平台验收状态，不用单平台结果代表双平台 |
| D9 | 精度章节同时出现 1e-2、32×ULP 和酌情放宽至 2ULP | 默认采用固定 max_abs_error≤1e-2，与 matched_ratio 条件同时满足；其他 ULP 口径须在验收前明确公式和适用范围，不自动放宽 |

## 3 详细设计（required）

### 3.1 总体方案与调用方式

采用 handle 绑定 stream 的 Ascend C Kernel 直调方案。主路径为关闭 HF32 降精度模式的 FP32 分块矩阵乘，配合逐组任务调度；极小规模、退化形状及 beta-only 采用 FP32 Vector 路径。初始版本不采用跨核 Split-K 或对 C 的原子累加，确保每个输出 tile 有唯一写入者。

```text
公共 BLAS 接口
  → 完整参数校验、受检前缀与存储范围计算
  → 逐组生成 GroupDesc，选择 Cube / Vector / beta-only
  → 合并同路径且 tiling 兼容的组，生成 tile 前缀表
  → 获取工作区，按同一 stream 上传不可变参数快照
  → Cube GEMM 或 Vector Kernel
  → alpha/beta epilogue（需要时）
  → 记录工作区使用完成事件 → 异步返回
调用者同步 stream → 读取 C
```

A2/A3 的普通 FP32 Mmad 与 HF32 模式需显式区分，主路径关闭 HF32；不以把 FP32 转成 FP16/BF16 替代精度实现。Matmul/底层 Mmad 的具体类型组合与布局约束，以 CANN 9.1.0 头文件、编译和目标板结果为准。若高级 Matmul 无法直接表达某种 leading dimension，则在 tile 内做受控搬运与布局转换；不通过整批矩阵重排增加不可控工作区。

### 3.2 Host 侧设计

#### 3.2.1 元数据与分核

`GroupDesc` 保存原始 groupId、batchOffset、groupSize、M/N/K、lda/ldb/ldc、transa/transb、alpha/beta、路径标识、tileM/N/K、M/N tile 数与工作量前缀。全局描述保存组数、batch 总数、地址表位置、参与核数、描述区大小和分批处理范围。跨组累计量、地址及字节偏移采用 uint64_t/size_t，设备需要窄类型时先验证上限。

对组 g，定义 `TMg=ceil(Mg/BMg)`、`TNg=ceil(Ng/BNg)`，工作单元数为 `Wg=groupSize[g]×TMg×TNg`。每个工作单元对应一个 `(g,batch,tileM,tileN)`，串行完成该输出 tile 的 K 维累加。no-op 组的 Wg 为零，但原始指针表前缀保留。

同计算路径内按 K 深度和 tiling 档位分桶，保留描述符到原始组的映射，不重排用户 A/B/C 数组。各核采用 `workId=blockIdx+t×blockNum` 遍历任务前缀，通过小组表查找或二分定位组。大 batch 与大矩阵均可沿输出 tile 展开，避免仅按 batch 均分导致单矩阵无法利用多核。短 K 和长 K 分桶减少组间工作量差异；桶数量受限，防止逐组启动放大开销。

#### 3.2.2 tiling 与存储预算

从目标平台查询 AIC/AIV 数量及 UB、L1、L0A/L0B/L0C 容量。候选 tile 以 32/64/128 等档位构成，经数据类型、矩阵指令与缓存约束过滤；初始候选为 BM×BN×BK=`64×64×32`，仅作为调优起点，不作为固定最优参数或全部输入的强制限制。

双缓冲 A/B tile 的逻辑载荷为 `2×4×(BM×BK+BK×BN)` 字节，FP32 累加块为 `4×BM×BN` 字节；另计格式对齐、API 内部缓冲、epilogue 输入输出及同步开销。分别约束实际所在的 L1、L0、UB，不能把不同层容量合并为一个总预算。Host 仅采用通过全部资源约束的配置；尾块维度独立记录，禁止利用 padding 掩盖越界访问。

元数据工作区估算为 `align(header)+align(G×sizeof(GroupDesc))+align(3×B×8)+桶前缀与对齐余量`，其中 G 为组数、B 为 batch 总数。若输出需要临时乘积，工作区增加 `4×Σchunk(groupSize×M×N)`，限制为当前 chunk 的容量，不分配全批规模的乘积缓冲。

#### 3.2.3 路径选择与 tiling key

| key/路径 | 条件 | 处理 |
| --- | --- | --- |
| NO_OP | 无活跃组，或 beta-only 的 beta=1 | Host 完成校验后返回；混合组中只移除相应工作项 |
| SCALE | K=0 或 alpha=0，且需更新 C | 按有效 C 元素分块，beta=0 直接写零 |
| VECTOR | 极小 M/N/K、长窄矩阵等候选区域 | FP32 向量分块累加；以实测交叉点决定适用边界 |
| CUBE_NN/NT/TN/TT | 常规 GEMM | FP32 矩阵乘，转置和 alpha/beta 均取当前 GroupDesc |

路径判定只依赖公开参数、资源约束和已验证的调优规则，不根据 case 名、随机种子或输入数据内容选路。调优参数需经过四种转置、非对齐、padding 和异构组回归。

#### 3.2.4 异步执行与工作区生命周期

Host 数组在接口调用期间读取并复制为内部快照；A/B/C Device 数据、用户工作区和 handle 应由调用者保留至相关 stream 工作完成。参数上传、GEMM、epilogue 和完成事件必须遵循同一 stream 的次序，不在正常热路径调用全设备同步。

不能直接复用 arch35 的“同步拷贝参数至固定工作区后立即启动”方式来证明多次异步调用安全。arch22 采用每次调用独立参数槽位：固定 Host staging 保存元数据，异步上传至对应 Device 槽位；完成事件之前不释放 staging、不覆盖槽位。内部管理器以 handle/stream 为归属维护空闲槽位，后续调用仅回收已完成事件对应的槽位；handle 销毁、切换 stream/workspace 时等待相关任务完成并清理资源。若需扩展公共生命周期钩子，单独集中在公共 helper，回归现有算子，不引入公共 ABI 变化。

用户工作区不足时明确报错，不擅自释放或替换用户内存。库工作区增长与槽位耗尽的慢路径可等待该 handle 的相关事件，记录其 Host 延迟；热路径维持异步。并发 Host 线程对同一 handle 的提交遵循库线程安全约定并序列化元数据更新；不同 handle 的事件、槽位和 stream 完全隔离。

### 3.3 Kernel 侧设计

#### 3.3.1 列主序地址与转置

对逻辑坐标 `i∈[0,M)`、`j∈[0,N)`、`p∈[0,K)`，单位为 float 元素：

```text
op(A)(i,p): transa=N → A[i+p×lda]；transa=T → A[p+i×lda]
op(B)(p,j): transb=N → B[p+j×ldb]；transb=T → B[j+p×ldb]
C(i,j): C[i+j×ldc]
```

四种转置组合均由上述映射推导。若底层 Matmul 采用行主序逻辑，可用 `Cᵀ=op(B)ᵀ×op(A)ᵀ` 交换操作数及 M/N 表达，或在片上 tile 中转换布局；不进行整矩阵 Host 转置。padding 和非 32 字节对齐尾部采用 A2/A3 支持的带尾块搬运或受控标量补偿，只访问有效分配区。

#### 3.3.2 Cube 主路径

各核读取工作项及对应地址，沿 K 维循环搬运 A/B tile 到片上缓冲，完成所需 ND/矩阵指令布局转换，使用普通 FP32 矩阵乘并累加 FP32 输出。第一个 K tile 初始化累加器，后续 tile 累加；K 尾块补零只发生在片上无效区域。双缓冲使用明确的搬运/计算依赖，复用缓冲前等待使用者完成。

`alpha=1,beta=0` 优先直接写回 C，有效避免全部五个典型性能 case 的额外 epilogue 流量。通用 alpha/beta 在支持的路径中完成 FP32 epilogue；初版若使用独立 Vector epilogue，则 Cube 写 chunk 临时乘积，同一 stream 的后续 Kernel 读取乘积和必要的 C 旧值后写回。必须将这两个 Kernel 的耗时共同纳入接口性能统计。

#### 3.3.3 Vector 与退化路径

Vector 小矩阵路径在 UB 内组织 A/B/C tile，按列或点积执行 FP32 乘加与规约，使用 A2/A3 支持的 Ascend C API，不直接搬用 arch35 MicroAPI 实现。beta-only 只遍历有效 M×N；beta=0 直接生成零，beta=1 无写入，其他 beta 进行缩放。带 Inf/NaN 的输入按实际参与运算的数据传播；已被 alpha=0 或 beta=0 短路的操作数不得因提前读取而污染输出。

不保证浮点结果逐位确定；不要求 NaN payload 一致。常规用例依验收容差判断，NaN/Inf 类别依 4.1 的专门规则判断。

### 3.4 资源、性能与稳定性设计

以 `FLOPs=Σg 2×groupSize[g]×M[g]×N[g]×K[g]` 计算有效工作量，分别分析小矩阵启动开销、片上搬运、K 维计算和 C 写回。优先进行跨 batch 调度、同路径合并、A/B 双缓冲、beta=0 免读 C、尾块专用搬运和 tiling 档位优化；性能调优前后使用同一输入、Golden、编译配置和采样口径回归。

典型第 5 案例仅 A/B/C Device 数据约 `128×3×4096²×4=24 GiB`，第 4 案例约 12 GiB，第 3 案例约 3 GiB，不含工作区及 Host Golden。测试在分配前检查设备与 Host 可用容量；Golden 分矩阵生成并比较，限制 Host 峰值。不能将 batch 数缩小或分多次公共接口调用后冒充原典型 case。算子内部 chunk 化须仍对应一次完整逻辑调用，并统计其全部 Kernel。

通用规模按实际地址空间和受检整数范围处理，不把 CSV 的最大维度 4096 或组数 8 写成 API 上限。对极大输入遇到资源不足时返回既有错误码，不截断 batch、跳过组或返回部分成功。

### 3.5 工程文件规划与硬件支持

```text
ops-blas/
├── include/cann_ops_blas.h                         # 复用既有声明
├── blas/gemm_grouped_batched/
│   ├── README.md                                  # 更新 A2/A3 支持及约束
│   ├── arch35/                                    # 原实现保持兼容
│   └── arch22/
│       ├── gemm_grouped_batched_host.cpp
│       ├── gemm_grouped_batched_kernel.cpp
│       └── gemm_grouped_batched_tiling_data.h
└── test/gemm_grouped_batched/sgemm_grouped_batched/
    ├── CMakeLists.txt / README.md
    └── arch22/
        ├── sgemm_grouped_batched_test.cpp
        ├── sgemm_grouped_batched_npu_wrapper.h
        ├── sgemm_grouped_batched_test.csv
        └── gpu_baseline.csv
```

公共参数、Golden 和验收脚本放置于最小必要共享层，不复制整套测试框架。Kernel 若因 Cube/Vector 编译要求必须拆分，按真实职责拆分并在代码评审说明；不新增无用途的占位文件。

| 产品 | 本任务目标 | 验证安排 |
| --- | --- | --- |
| Atlas A2 训练/推理系列（含 800I A2） | 支持 | 910B3 优先完成精度及性能自验 |
| Atlas A3 训练/推理系列（含 800I A3） | 支持 | 独立构建与运行，留存 A3 验收证据 |
| Ascend 950PR/950DT | 保持已有支持 | 公共接口、构建及测试基础设施变更需做 arch35 回归 |

## 4 可维可测分析

### 4.1 精度标准

Golden 由 Netlib `cblas_sgemm` 按组逐矩阵生成，使用相同列主序、转置、leading dimension、初始 C 与 alpha/beta。对 alpha=0、K=0 等快捷路径先按接口契约处理；负向校验测试直接检查真实 NPU 接口，不能用 wrapper 返回模拟错误代替接口调用。

对每个有效输出元素，有限值匹配条件为 `abs(actual-golden)≤2^-13+2^-13×abs(golden)`。每个用例要求匹配比例≥0.99，且最大绝对误差≤1e-2，两项同时满足。统计每个矩阵、每组和整个用例的有效元素数、匹配数、最大误差及最差位置；padding 不进入比例分母，单独检查保持不变。

Golden 为 NaN 时 actual 必须为 NaN；Golden 为 Inf 时 actual 必须为同符号 Inf；有限/非有限不一致直接失败。特殊值匹配计入元素匹配，有限配对计算绝对误差，任一特殊值类别不匹配均不得被 99% 比例豁免。零有效元素的 case 不计算空分母，检查状态码、无写入及无越界。

### 4.2 配套用例与补充覆盖

归档 CSV 实际为 1,200 条，其中精度 1,000 条、性能 200 条。全包期望状态为 SUCCESS 1,170 条、INVALID_VALUE 29 条、HANDLE_IS_NULLPTR 1 条。实际分类如下：

| 分类 | 前缀 | 条数 | 主要验证内容 |
| --- | --- | ---: | --- |
| 基础 | TC_L0 | 6 | NN/NT/TN/TT 与逐组混合转置 |
| 尺寸扫描 | TC_SQ | 88 | 1～2048 尺寸池及转置组合 |
| 标量 | TC_AB | 12 | 0、1、负值、不同组 alpha/beta |
| 组数 | TC_GC | 6 | 多组独立形状与参数 |
| batch | TC_BC | 13 | 组内 batch 变化 |
| 前导维 | TC_LD | 5 | lda/ldb/ldc padding |
| 填充 | TC_FL | 6 | 随机、零、正负交替、极值、Inf、NaN |
| 边界负向 | TC_ED | 40 | no-op、缩放、空指针、枚举及维度错误 |
| 扩展 | TC_EX | 824 | 形状、组、转置、标量、batch 的组合 |
| 性能 | TC_PF | 200 | 五条任务书典型 case 加扩展性能组合 |

新增覆盖包括：逐组正态/均匀各 50% 的输入与标量分布；K=0 且 beta=0/非 1；alpha=0 且 A/B 含 NaN/Inf；beta=0 且 C 初值为 NaN；缺失的 nArray/kArray/ldbArray/ldcArray/groupSizeArray 空指针；任意非法枚举数值；第二组及后续 batch 的单元素空指针；不同矩阵内容与非连续地址分配；32 字节对齐偏移及边界哨兵；64 位前缀/字节数溢出；长短 K 混合负载；连续异步调用、多 handle、多 stream、workspace 更换与销毁。

附加 no-op 与非法参数交叉用例，验证 2.3 中的校验优先级。对别名限制增加 Host 区间计算校验测试；不以实际设备越界访问或并发写竞争验证被禁止的输入。

### 4.3 测试框架修正与复现

1. 保留原 CSV、GPU 基线及哈希，建立 case_name 到源行、参数和 Golden 的映射。生成器输出到单独目录，不覆盖已填入实测值的 `gpu_baseline.csv`；原生成器重跑会写出待回填基线。
2. 解析三张指针表的 null 标记：`NULLPTR` 置空整表；分号组标记通过 `prefix[g]` 定位；单值 `2` 按配套定义置空首矩阵。测试构造层记录实际注入的扁平下标，避免把 groupId 当作 batchId。
3. `null_handle` 必须真正传 nullptr；负维度、空数组等测试用安全占位分配后传入非法元数据，不按负维度申请内存。不得由测试 wrapper 提前合成任务预期错误码。
4. 新测试目标使用显式 Mixed Tolerance，报告必须给出实际比较元素数。旧 MERE/MARE 默认值或缺列行为不作为通过依据。
5. 修正精度脚本的嵌套 CSV 路径、SoC 映射、二进制发现及过滤参数。使用明确的 GTest 正向模式和负向排除模式；通过 `--gtest_list_tests` 核验选中的 case 集合。
6. 同时校验进程退出码、GTest XML、预期执行数、PASS/FAIL/SKIP 数；超时、崩溃、零测试、漏 case 或无法解析一律失败。修正原脚本仅按解析失败数决定退出码而可能在零 case 时返回 0 的行为。

接入后的目标命令示例如下，实际二进制路径以构建产物和 CMake 注册结果为准，并写入交付 README：

```bash
bash build.sh --soc=ascend910b3 --ops=sgemm_grouped_batched
./build/test/gemm_grouped_batched/sgemm_grouped_batched/sgemm_grouped_batched_test --gtest_list_tests
./build/test/gemm_grouped_batched/sgemm_grouped_batched/sgemm_grouped_batched_test \
  --gtest_filter='*-*TC_PF*' --gtest_output=xml:accuracy.xml
```

A3 使用设备报告的 `ascend910_93*` SoC，不沿用附包脚本将 `ascend910b4` 注释为 A3 的写法。命令为工程接入目标，设计阶段未执行 CANN 构建或 NPU 测试。

### 4.4 性能标准与计时口径

性能设备为 Atlas 800T A2（910B3）。每 case 预热 5 次、有效采样 20 次，报告算术平均值（µs）、样本数、原始样本与环境。`gpu_baseline.csv` 的 `gpu_ms` 先乘 1000 转成 µs；扩展 case 要求 `NPU_mean_us≤gpu_ms×1000/0.8`。典型 case 同时核对任务书显式阈值；表值存在四舍五入差异时执行更严格的数值。

| case_name | G | groupSize | M=N=K（各组） | transa/transb | 任务书上限 µs | CSV 换算上限 µs |
| --- | ---: | --- | --- | --- | ---: | ---: |
| TC_PF_1001 | 2 | [64,64] | [256,256] | NN/NN | 351.3 | 351.28625 |
| TC_PF_1002 | 2 | [128,64] | [512,512] | NN/NN | 3714 | 3714.125 |
| TC_PF_1003 | 3 | [64,128,64] | [1024,1024,1024] | NNN/TTT | 37652 | 37651.875 |
| TC_PF_1004 | 2 | [128,128] | [2048,2048] | TT/NN | 298380 | 298380 |
| TC_PF_1005 | 2 | [64,64] | [4096,4096] | NN/NN | 1188440 | 1188440 |

上表五例 alpha 均为 1、beta 均为 0、leading dimension 为对应紧凑最小值。全部 200 条性能 case 逐行按 case_name 关联基线，并校验组数、转置、M/N/K 和 groupSize 一致，禁止错配或漏报。

通过单 case GTest filter 与 `msprof op --application=...` 采集 `OpBasicInfo.csv`。正式测试驱动在测量区间只进行接口调用，Golden、数据生成、分配及 D2H 校验位于区间之外；beta≠0 时每次采样恢复相同初始 C，并把恢复操作从计算 Kernel 统计中单列排除。记录工具版本及其实际 warmup/replay 行为，确认恰有 20 次有效逻辑调用，不能默认工具会满足采样数。

一次接口可能包含分桶、chunk、转换、乘法、缩放和 epilogue 多个 Kernel：按调用序号关联后求全部相关 Kernel 耗时之和，再对 20 次调用求平均；不跨异构 Kernel 直接平均，不除以 group 数或 batch 数。另列 stream 上完整设备区间与 Host 端到端时间，避免用单一最快 Kernel 替代接口完成耗时。参数 H2D 等非计算开销作为单独列保留。

### 4.5 内存、兼容性与可维护性

任务无独立内存性能门限，仍提交内存报告：A/B/C 大小、handle 工作区、参数槽位、临时乘积、Host staging 与 Golden 峰值、重复调用后的残留和释放结果。预分配后反复调用，验证内存不持续增长；工作区不足须可诊断，错误路径不得泄漏资源。

保持现有函数符号、参数顺序、头文件与 950 架构实现。构建需验证 A2/A3/950 各架构仅链接一个公共入口；公共 helper 修改回归 handle 创建、stream/workspace 切换与销毁。日志记录 groupId、参数类别、所需/可用资源和状态码，不记录矩阵数据或完整设备地址。

### 4.6 交叉特性分析

| 交叉特性 | 约束与验证 |
| --- | --- |
| 转置 × padding × 尾块 | 对四种转置分别验证非对齐 M/N/K 与非最小 lda/ldb/ldc，逐矩阵核对地址映射 |
| 异构组 × no-op × beta-only | 同次调用混合正常组、空组、缩放组，保留原始 batch 前缀，检查每组 C 与哨兵 |
| 异步 × 工作区复用 × 多次调用 | 改变组参数后连续提交，最后统一同步，验证每次调用使用独立参数快照 |
| 多 stream × handle 生命周期 | 各 handle 独立资源；stream/workspace 切换与销毁等待所属在途任务；不扩展未声明的并发保证 |
| FP32 × 特殊值 × 快捷路径 | 对 alpha=0/beta=0 的免读规则和普通 GEMM 非有限值传播分别验收 |
| 动态入参 × 架构选择 | 运行时 shape 由 Host tiling 处理；编译期架构选择保持 arch22/arch35 符号隔离 |
| Graph/模型框架/自动求导 | 本期为 BLAS Kernel 直调，未增加 ACLNN 图算子、图捕获或反向算子能力 |

## 5 开发阶段、风险与交付

### 5.1 阶段门槛

| 阶段 | 开发输出 | 进入下一阶段的条件 |
| --- | --- | --- |
| G0 设计评审 | 本设计、差异处置、任务目录与接口契约 | 社区评审通过，确认 no-op 校验顺序和精度/计时口径 |
| G1 工程与正确性 | arch22 接入、FP32 基础路径、beta-only、真实负向测试 | ABI/构建正确，基础与边界用例实际比较并通过 |
| G2 泛化与异步 | 转置/padding、异构组调度、工作区槽位与生命周期 | 1,000 条配套精度及补充场景通过，无越界和异步串扰 |
| G3 性能与双平台 | tiling 调优、200 条性能、A2/A3 记录、内存测试 | 各 case 达标，原始样本可复算，双平台证据完备 |
| G4 验收与合入 | 测试代码、报告、README、实现 PR | 设计批准、自验完成、平台验收及仓库审核按流程完成 |

### 5.2 主要风险及应对

| 风险 | 验证及应对 |
| --- | --- |
| Cube 布局/精度或资源不满足预期 | 优先验证四种转置及大 K；关闭 HF32；资源查询加静态预算；保留 FP32 Vector 正确性路径，不以降精度换取未审核性能 |
| 异构分组负载失衡 | 按 K 深度与 tiling 分桶，按输出 tile 展开，比较整批吞吐与尾核时间 |
| Host 快照或工作区过早复用 | 同 stream 有序上传、独立参数槽位、事件回收；连续无同步提交后整体同步验证每次输出 |
| 大规模内存不足 | 预核算 24 GiB 典型输入，限制临时乘积 chunk；保留完整任务规模并记录资源失败 |
| 测试假通过 | 强制实际比较元素数、真实负向入参、XML 与退出码联合校验，零 case/漏 case 失败 |
| 资料与代码目录/验收口径差异 | 按 D1～D9 逐项闭环，在评审记录与交付 README 标明最终确认值 |

### 5.3 交付件与评审记录

设计文档以单独 PR 提交至 competitions 对应任务目录的 `gcw_l6gqPgyE/docs/design.md`，标题使用 `【社区任务】aclblasSgemmGroupedBatched算子设计文档`。本任务在已查官方任务列表中未列出序号，目录暂采用 `09-aclblasSgemmGroupedBatched-A2A3`；后续以平台正式编号对齐。

设计评审通过并完成实现、自验后，向任务指定个人代码仓提交可复现代码、分支与目录，按要求配置验收访问。实现 PR 进入 `cann/ops-blas`，含算子、测试、CSV 和产品支持 README；验收资料集中于 `task_submission/`：

```text
task_submission/
├── 1 自验证步骤说明.md
├── 2.1 精度自验证报告.xlsx
├── 2.2 精度自验证日志.log
├── 3.1 性能自验证报告.xlsx
├── 3.2 性能自验证日志.log
├── 4.1 内存自验证报告.xlsx
└── 4.2 内存自验证日志.log
```

报告包含源码提交、构建选项、CANN/驱动/固件、SoC 型号、Golden 实现、随机种子、用例参数、精度结果、性能原始数据、内存记录及截图。未执行测试的栏目不得填写为通过；设计 PR 合入、代码合入和平台验收分别记录。

## 6 参考资料

1. 《aclblasSgemmGroupedBatched_A2A3 任务书》及 A3zip 随附 `test_cases/README.md`、`gen_csv.py`、`verify_accuracy.py`、两份 CSV。
2. [官方设计模板](https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md)与[社区任务提交规范](https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/README.md)。
3. [社区流程及注意事项](https://gitcode.com/org/cann/discussions/39)与[设计文档检查清单](https://docs.qq.com/sheet/DUHVGUFdmSFRjVFFU?tab=000001)。
4. [ops-blas 分组批量 GEMM 基线](https://gitcode.com/cann/ops-blas/tree/2c9b9b77bda979c25a6b89726614b082e73226fa/blas/gemm_grouped_batched)及同提交的 `include/`、`blas/common/helper/`、`test/gemm_grouped_batched/`、构建文件。
5. [生态 Mixed Tolerance 精度标准](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/mixed_tolerance_standard.md)。本任务的具体 FP32 门限遵循任务书及评审确认记录。
6. [Ascend C SetHF32Mode 接口说明](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0258.html)；最终 API 使用与资源约束以 CANN 9.1.0 目标环境验证为准。
