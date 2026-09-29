# aclblasStbsv 算子设计文档（Atlas A2/A3）

版本：v1.2，2026-09-28，设计评审稿。目标环境：CANN 9.1.0；Atlas A2/A3。  
实现仓：`cann/ops-blas`；新增 `blas/tbsv/arch22/`，沿用已有公共接口及 arch35 实现。  
本稿对应候选5；Kernel SHA256：`dbee3b80fcbd721ac49605aff3cf524abcda799f6cd00016d6f55ff73fa79fe9`。v1.0、v1.1及各自验证范围保存在 `artifacts/design_versions/`。

## 1. 需求背景（required）

本设计对应[9月社区任务 aclblasStbsv 算子开发（A2/A3）](https://www.hiascend.com/activities/task-center/details/29c59d7ef9fa4aa8ad26b82969c5e2b6?menu=tasks)，依据任务书与随附 `test_cases/`，参考[NVIDIA cuBLAS tbsv](https://docs.nvidia.com/cuda/cublas/#cublas-t-tbsv)及[Netlib STBSV](https://www.netlib.org/blas/stbsv.f)。文档按[官方模板](https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md)编写，模板核对版本为 `c2b149e43613cdcae482a1eb83bcd1afb9643a14`。

三角带状求解使用主对角及其一侧有限条对角线，矩阵压缩为列主序带状数组，减少存储与访问。右端向量与解共用 Device 缓冲区。本任务新增 A2/A3 Ascend C 内核，完成接口校验、泛化、精度与性能验证。

本地 ops-blas 分析基线为 `25f17d3a7c5a0c97120f1c28637e2d78b4954f6a`，已有 `include/cann_ops_blas.h` 声明及 `blas/tbsv/arch35/` 实现，README 将 A2/A3 列为不支持。arch35 的 SIMT 指令与调度不能直接用于 arch22；本方案复用公共 ABI、句柄、stream、构建及测试基础设施，新增目标架构实现。

## 2. 需求分析（required）

### 2.1 数学语义与接口

求解 `op(A) × x = b`，输入时 x 存 b，完成后解原地覆盖 x。A 为 FP32 三角带状矩阵，只读；x 为 FP32，读写；其余参数位于 Host。N 表示 A，T 表示 Aᵀ，实数 C 与 T 等价。

```cpp
aclblasStatus_t aclblasStbsv(
    aclblasHandle_t handle, aclblasFillMode_t uplo,
    aclblasOperation_t trans, aclblasDiagType_t diag,
    int n, int k, const float* A, int lda, float* x, int incx);
```

| 参数 | 约束与异常 |
| --- | --- |
| handle | nullptr 返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| uplo/trans/diag | 支持 UPPER/LOWER、N/T/C、NON_UNIT/UNIT；非法枚举返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| n、k、lda | n≥0、k≥0、lda≥max(1,k+1)；校验用 lda>k 避免 k+1 的 int 溢出，非法值返回 INVALID_VALUE |
| incx | 正负非零步长；按附件拒绝 INT_MIN，非法值返回 INVALID_VALUE |
| A、x | n>0 时为非空 Device 地址且不重叠；任一空地址返回 INVALID_VALUE |

当前公共头文件枚举实际为 `ACLBLAS_UPPER/LOWER`、`ACLBLAS_NON_UNIT/UNIT`，沿用其名称与数值，不新增任务书语义名称的重复枚举。参数顺序与 cublasStbsv 一致，无标量、广播或 batch 参数。

### 2.2 带状地址与边界

以下为从 0 开始的逻辑索引 i,j∈[0,n)，数组偏移单位为 float 元素。

| 布局 | 有效范围 | 压缩偏移 | 主对角 |
| --- | --- | --- | --- |
| UPPER | max(0,j−k)≤i≤j | `j*lda+k+i−j` | `j*lda+k` |
| LOWER | j≤i≤min(n−1,j+k) | `j*lda+i−j` | `j*lda` |

循环有效带宽 `k_eff=min(k,n−1)`；k≥n 仍合法，UPPER 地址保留原始 k，不以 k_eff 代替压缩行。列 padding、带外位置和 UNIT 主对角不参与计算。UNIT 不加载对角，NON_UNIT 读取并除法。

向量访问为 `x[start+i*incx]`：incx>0 时 start=0，incx<0 时 start=(1−n)*incx；跨度、起点与跨列偏移使用 int64 计算。物理长度为 `1+(n−1)*|incx|`，只写 n 个有效元素，不修改间隙或保护区。

Host 先校验 handle、数值及枚举。n=0 时允许 A/x 为空，成功返回且不发射；其余非法元数据仍报错。n>0 时先检查 A/x，随后 k=0且UNIT 成功返回、x不变；k=0且NON_UNIT 为对角求解。与 BLAS 一致不检查奇异性或近奇异性，实际分配长度与不重叠由调用者保证。

### 2.3 资料差异处置

| 问题 | 本方案处理 |
| --- | --- |
| 附件 README 写 k≤n−1，任务书/BLAS 仅 k≥0 | 支持 k≥n，补充原始压缩行测试 |
| 附件期望 INT_MIN 步长非法，任务简表仅 incx≠0 | 对齐附件及既有 README，明示拒绝 INT_MIN |
| 任务要求 Netlib CBLAS，附件文字称自实现 golden | 正式测试使用官方Netlib STBSV私有库，并核验符号来源；独立模型仅辅助验证 |
| 自验款型与采样次数前后不一致 | 最终分别覆盖 A2/A3；至少5次预热、20次有效采样 |
| 1e−2 或 32 ULP 的最大误差口径 | 每个有限元素使用 max(0.01,32×对应 golden ULP)，同时报告最大绝对误差 |

## 3. 详细设计（required）

### 3.1 Host 流程与工程结构

新增 `stbsv_host.cpp`、`stbsv_kernel.cpp`、`stbsv_tiling_data.h` 于 `blas/tbsv/arch22/`，测试放 `test/tbsv/stbsv/arch22/`。接口声明继续使用 `include/cann_ops_blas.h`，README 按任务更新 A2/A3 产品支持。

Host 校验并处理 no-op → 构造带有 Device 地址及 n/k/lda/incx/uplo/trans/diag 的 tiling → 在 `handle->stream` 发射一次内核 → 调用者同步后读回 x。C 归一到 T，UPPER/LOWER×N/T×UNIT/NON_UNIT 共8个模板分支，避免 pivot 内重复判断。

blockDim=1，单个 AIV 完成依赖链，向量化每个 pivot 内的独立更新。无需 Device 临时分配或外部 workspace；tiling 按值传入启动参数，不上传可被后续调用覆盖的共享参数表。正常接口不做 Host 结果同步；调用方保持句柄、stream 与 A/x 生命周期至执行完成，同流连续调用依次执行。

### 3.2 Kernel 计算与内存

| uplo / trans | 方向 | 每步计算 |
| --- | --- | --- |
| LOWER / N | j=0→n−1 | 求 x[j]，更新后面带内未求解的 RHS |
| UPPER / N | j=n−1→0 | 求 x[j]，更新前面带内未求解的 RHS |
| UPPER / T或C | j=0→n−1 | 求 x[j]，更新后面带内未求解的 RHS |
| LOWER / T或C | j=n−1→0 | 求 x[j]，更新前面带内未求解的 RHS |

有效计算量 O(n×min(k,n−1))。N 路径保持 Netlib 列更新顺序，RHS pivot 为零时跳过该步。T/C 将 `op(A)` 视为反向三角矩阵，改为数学等价的列更新：全局 pivot 升序/降序恰使每个未来 RHS 接收贡献的次序与 Netlib 原始逐项消元一致，保留 FP32 每次乘、减、除及中间舍入。T/C 即使 pivot 为零也必须执行乘减，保留 `0×Inf/NaN` 的传播；UNIT 绝不加载主对角。

Init 建立 GlobalTensor、有效带宽与负步长地址。n≤16384 时完整 x 放入 UB，最多64 KiB；incx=1 使用 DataCopyPad，其余步长逐有效位置收集/写回。N 每次连续搬运当前列的有效非对角系数，复用系数 UB 经 Muls、Sub 完成独立更新，两块 UB 总计最多128 KiB。头部非对齐元素及尾块按有效长度处理，不写 x 间隙。

T/C 使用16个连续 pivot 共享的非对角 panel，将未来各列所需的连续系数分块搬到列主序 UB。一块内依次求解16个 pivot，每步由固定 Gather 索引和当前 pivot 偏移提取系数、向量更新未来 RHS，贡献顺序保持不变。内部列批量 DMA，左右边界裁剪至合法三角带；不足16个 pivot 的末块保留16个float列槽位。显式搬运排除 UNIT 对角、padding和带外位置。

panel 仅在 `k_eff≥64、n≥128`、DMA步幅合法且四块 UB 总和≤128 KiB时启用。令 `align8(t)` 为向上对齐8个元素，预算为 `4×[align8(n)+2×align8(k_eff+7)+16×(k_eff+22)]` 字节，分别对应完整x、系数、Gather索引与panel。其他T/C形状仍走最多512个系数的 strided DMA+Gather，预算最多86304字节（约84.3 KiB）；两种路径都不展开完整A。Scalar/MTE2/Vector/MTE3 使用对应事件同步，禁止缓冲区提前覆盖。

n>16384 时按 min(n,k_eff+1) 组织环形依赖窗口；窗口超过16384个float则走 GM 标量 fallback，不限制合法 n/k。两条后备路径及标量头部均通过显式 FP32 中间乘积抑制乘减收缩；T/C 维持 Netlib 减法顺序。保留原始 lda/k 与 int64 偏移，单核无跨核自旋或竞争写入。

### 3.3 性能优化与风险

| 方向 | 作用与验证条件 |
| --- | --- |
| 一次发射、模板分支、完整 xUB | 已实现；减少 Host 发射及 x 的重复 GM 访问，保留大 n 后备路径 |
| N 连续列搬运及向量乘减 | 已实现；不同目标 RHS 更新独立，乘积先舍入再减，保持零 pivot 语义 |
| T/C等价列更新及Gather | 已实现；消除逐行串行 dot，保持每个目标分量原始贡献顺序 |
| T/C 16-pivot非对角panel | 已实现；复用一批系数并减少DMA/事件次数，以128 KiB资源门限选路；两机新增panel特殊值/末块边界8次设备调用通过 |

优化依据实际 profiling；每轮保留源码哈希，先精度与边界回归，再比较同一输入与采样口径。保留512系数路径及大n后备路径；panel未满足资源条件即回退。不采用低精度计算、近似倒数或改变减法顺序的树形规约。单位三角递推可能产生幅值增长或溢出，完整记录输出分类，不过滤困难用例。

## 4. 支持硬件与算子约束（required）

| 产品 | 实现与验证安排 |
| --- | --- |
| Atlas A2 系列（含 Atlas 800I/T A2） | arch22；910B3 优先完成任务精度及规定性能 |
| Atlas A3 系列（含 Atlas 800I/T A3） | 同份 arch22 源码独立构建和验收 |
| Ascend 950 | 保留主仓 arch35 能力与公共 ABI |

支持 FP32、列主序、单矩阵/单 RHS、lda padding 及 x 正负步长。n/k 是运行时入参；不要求额外非连续 Tensor 描述、广播、视图或逐位确定性。A与x不可重叠，A及其padding、x间隙不改写。支持表示实现目标，已完成的验证范围见下节。

## 5. 可维可测分析（required）

### 5.1 精度与功能验证

使用仓库GTest/CSV，正式golden为官方Netlib `stbsv.f`，通过官方CBLAS适配调用。`scripts/netlib_reference/` 保存原始源码/哈希、私有库构建及符号身份核验，关闭FP收缩与fast-math，只对本测试命令启用，不改变系统BLAS或旧项目。附件共有1200条，含1000条精度/功能、200条TC_PF性能，以及10条期望错误状态的负向case；原附件与补充用例独立归档。

全部 n 个有效输出参与比较：有限值混合容差为 `abs(actual−golden)≤2^-13+2^-13*abs(golden)`，匹配率≥0.99；同时每个有限元素误差≤`max(1e−2,32*ULP(golden))`，ULP按对应位置FP32标杆计算。单独记录最大绝对误差、ULP信息及严格0.01条件是否满足，不用张量内最大ULP替代逐元素条件。NaN须同位置为NaN，Inf须同符号；分类不一致直接失败，不能由99%比例豁免。标准来源为任务书§3.2及[生态精度标准](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/mixed_tolerance_standard.md)。

覆盖12枚举组合、尺寸/带宽扫描、正负步长、padding、非对齐、n=0、k=0、非法参数和Inf/NaN。补充空handle、no-op非法参数组合、k≥n、UNIT对角毒值、A/gaps/guards不变、连续异步调用及UB窗口边界。真实负向输入调用公共接口，不由wrapper合成预期状态；核对退出码、GTest执行数、日志与XML，零case或漏case不能计为通过。

### 5.2 A2/A3 性能与内存

| case | n | k | uplo | trans | diag | 达标Avg（μs） | A2 Avg（μs） | A3 Avg（μs） |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 256 | 8 | UPPER | N | NON_UNIT | 198.0 | 93.6207 | 94.4279 |
| 2 | 512 | 32 | LOWER | N | NON_UNIT | 379.8 | 214.1353 | 212.1752 |
| 3 | 1024 | 16 | UPPER | T | UNIT | 659.5 | 591.3620 | 588.8478 |
| 4 | 2048 | 64 | LOWER | T | NON_UNIT | 1725 | 705.1901 | 699.3280 |
| 5 | 4096 | 128 | UPPER | N | UNIT | 2365 | 1415.0884 | 1417.8704 |

每case至少5次预热、20次有效采样，使用msprof `op_summary` 的 `Task Duration(us)` 求平均；每次先恢复原始RHS，数据恢复、CPU golden、Host准备不计入kernel耗时。保留工具版本、样本数与原始记录，接近门槛时做独立复测。

附件200项按原始TC_PF参数、随机种子、A/RHS生成方式运行，关联 `gpu_baseline.csv`，目标μs=`gpu_ms*1000/0.8`；前5行按任务显式μs值验收。两机正式各200/200达到耗时门槛，全输出、分类与guard检查均通过。性能harness使用 `-O2 -ffp-contract=off -fno-fast-math`；同机独立exporter直接使用原始fill.h，O0、O2关闭收缩与实际性能运行的200项参数、物理字节数及A/RHS指纹完全一致。原始日志、构建参数、源码/二进制哈希及5000条kernel采样分别保存；正式指标只取 `vector5_exact`。早期诊断和旧flags结果独立归档。Host/GTest端到端不计为kernel时间；静态UB预算与实际内存分别记录。

A2实际内存测试 n=4096、k=128，连续32次求解全输出与guard检查通过。A/x请求分配分别2113664/16512字节，Device workspace为0，该形状静态UB预算16928字节；HBM分配差额4194304字节，32次求解的新增增长为0。释放后设备整体空闲HBM仍比运行前少4194304字节，未恢复；此指标可能包含驱动分配池或其他进程，不能单独判为算子泄漏。进程生命周期peak RSS由79924升至84004 KiB，原始阶段值保留于 `artifacts/npu_vector5/a2/evidence/a2_memory_vector5.log`。

A3同形状32次全输出与guard通过；分配差额4194304字节，求解期间设备整体HBM空闲量另减少196608字节，释放后保留4390912字节，peak RSS183036 KiB。独立128次补充观察全输出及guard通过，期间整体HBM空闲量减少1089536字节，释放后保留5283840字节；80/96/112/128四个checkpoint空闲量均为65318948864字节、RSS均183036 KiB。仅在该有限采样窗口观察到稳定，不证明checkpoint间或其他负载恒定，也不单凭设备整体计数认定算子泄漏或完全物理回收。

### 5.3 当前验证范围与交付

| 范围 | 实际结果 | 证据 |
| --- | --- | --- |
| 前期 CPU与Host辅助验证 | 语义简测240/240、扩展432/432；Host78/78断言 | 对应 `artifacts/local_cpu_*.json`、`host_contract_result.json`；不代替NPU验证 |
| 私有官方Netlib标杆 | A2/A3各构建成功，并各通过288项独立已知解验证 | `scripts/netlib_reference/`；原始源码哈希与实际符号来源单独记录 |
| 候选5两机完整GTest | A2/A3各1210/1210通过，含原始1200 CSV与10项补充GTest，使用私有Netlib；补充含1296/96组内部调用、8次大n及新增8次panel边界调用 | 绑定首部候选5；GTest、原始CSV与1449次补充内部调用分别计数 |
| 候选5两机性能 | 正式原任务输入各200/200通过，全输出与分类/guard通过 | 逐case5次预热+20次有效kernel样本，同机独立输入QA各200/200通过 |
| 候选5 A2内存 | 32次全输出及guard通过，HBM重复增长0，释放后仍保留4 MiB，RSS末阶段84004 KiB | 见5.2及实际内存日志；不按虚构内存门限判定 |
| 候选5 A3内存 | 32次整体HBM变化196608字节；128次补充变化1089536字节，末四checkpoint相同；全输出及guard均通过 | 原始阶段/RSS独立报告，保留波动及有限观察窗口限制 |

前期OpenBLAS0.3.20结果归档于 `artifacts/npu_initial/`，其T/C计算顺序与指定Netlib不同；初始标量候选另发现乘减收缩舍入差异。历史误差与失败保留，正式结果以指定Netlib、同一源码版本重测为准，不混合不同候选的通过记录。

本稿记录候选5方案与双产品已完成的自验证。交付含arch22源码、公共README、CSV/GTest、复现步骤、三份自验证报告及原始日志，记录源码、CANN、驱动、SoC、Netlib身份与随机种子。报告集中于task_submission，精度、性能、内存各自保持证据和计算口径；设计评审、代码合入及官方验收由对应流程确认。

按[官方提交规范](https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/README.md)，设计文件提交至 `04_tasks/01_community-task-2026/tasklist/{官方任务编号}-{算子名称}/{实际团队名}/docs/design.md`，代码及测试分别进入ops-blas的 `blas/tbsv/arch22/`、`test/tbsv/stbsv/arch22/`。固定官方版本尚无本任务目录，提交时按实际任务编号创建，不用活动UUID代替编号；最终验收须完成全部要求的自验证。

## 6. 兼容性分析

保持已有aclblasStbsv函数符号、参数顺序、公共枚举和handle stream，按架构选择arch22/arch35。A2/A3新增实现不替换950内核；C与T共用实数转置路径。浮点不要求跨产品逐位相同，按明确混合容差和特殊值规则验收；INT_MIN拒绝策略与附件/既有README一致，并在评审中明示。
