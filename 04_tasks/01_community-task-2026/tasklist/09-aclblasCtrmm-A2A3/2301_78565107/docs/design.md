# aclblasCtrmm 算子开发任务书与设计计划

本计划面向 9 月社区任务 aclblasCtrmm A2A3，拟由 KousakaReina（GitCode 2301_78565107）在 ops-blas 框架内开发 complex64 三角矩阵乘。本文仅说明需求、拟采用方案及验证计划，不包含开发经过或实测结论。自验证按本次任务约定使用 Atlas 800T A2 910B3 和 CANN 9.1.0。

# 需求背景（required）

## 需求来源

以附件 aclblasCtrmm_A2A3_task_doc.md、原始 trmm_test.csv、gpu_baseline.csv 和测试指导为验收依据。目标是提供与 cuBLAS cublasCtrmm 对齐的句柄式异步接口，补充 Atlas A2/A3 的复数三角矩阵乘能力。实现拟放入 blas/trmm/arch22，测试放入 test/trmm/ctrmm/arch22，接口声明放入 include/cann_ops_blas.h。

## 功能语义

LEFT 模式计算 C = alpha × op(A) × B；RIGHT 模式计算 C = alpha × B × op(A)。A、B、C 均为列主序 complex64；每个复数由两个 float32 表示。LEFT 时 A 为 m×m，RIGHT 时 A 为 n×n，B 和 C 均为 m×n。

op(A) 支持不转置、转置及复共轭转置。只引用 uplo 指定的上三角或下三角；UNIT 时对角按 1+0i 处理，不读取存储的对角值。NON_UNIT 的对角允许为零，本算子不是三角求解。

# 需求分析（required）

## 接口和边界

```cpp
aclblasStatus_t aclblasCtrmm(
    aclblasHandle_t handle, aclblasSideMode_t side,
    aclblasFillMode_t uplo, aclblasOperation_t trans,
    aclblasDiagType_t diag, int m, int n,
    const aclblasComplex *alpha, const aclblasComplex *A,
    int lda, const aclblasComplex *B, int ldb,
    aclblasComplex *C, int ldc);
```

handle 与 alpha 位于 Host；A、B、C 位于 Device。alpha 由 Host 读取后随 tiling 参数传给 kernel。句柄绑定的 stream 决定下发顺序，调用方读回结果前同步。

| 约束 | 拟实现行为 |
|---|---|
| 枚举与维度 | 检查 side、uplo、trans、diag；m、n 非负；非法值返回 INVALID_VALUE |
| 前导维 | lda ≥ max(1, LEFT?m:n)，ldb、ldc ≥ max(1,m)；支持各自 padding |
| 空句柄及标量 | 空 handle 返回 HANDLE_IS_NULLPTR；空 alpha 返回 INVALID_VALUE |
| 零维 | 完成标量参数校验后 m=0 或 n=0 直接成功，不下发计算 |
| alpha 为零 | 不引用 A/B，允许 A/B 空指针，只将有效输出 C 置零 |
| UNIT | 不读取存储的主对角；未选择的三角区不参与数值计算 |
| 原地 | 按任务书第 2.5 节支持 B 与 C 为同一矩阵；拟先保存输入再计算，避免并行覆写 |
| 其他重叠 | C 与 A 或 B 的其他重叠不属于支持范围；不返回视图 |
| 内存与布局 | 支持 lda/ldb/ldc 定义的列主序访问，不扩展为任意 stride 张量 |

附件测试指导称输出与输入不允许重叠，但任务书第 2.5 节明确允许 B≡C，拟按较完整的接口要求实现并补测。环境部分第 3.1 节允许自验一种款型，而末尾存在两款均验收的表述；本次自验范围为用户指定的 910B3，报告将据实记录 A2 实测，不宣称已测试 A3。

# 详细设计（required）

## Host 与 tiling

拟先校验参数、处理空尺寸和零标量，再根据 m、n、三角阶数、布局及可用工作区确定计算路径。tiling 记录矩阵地址、前导维、枚举、复标量、输出分块和工作区偏移。地址和容量计算使用 64 位整数并检查溢出。

核数和块数依据设备资源与有效输出工作量决定。路径选择只依赖数学尺寸、布局和硬件容量，不读取 case 名、随机种子或性能表，不匹配官方 shape 列表。

## Kernel 方案

小尺寸拟使用单次 AIV 启动直接完成复数计算和写回，减少准备开销。将输出分配给不同核，每个输出或输出块只由一个核写入；按三角与转置关系裁剪归约区间。必要时使用向量乘加与分块搬运，提高矩形矩阵或中小尺寸效率。

大尺寸拟利用 Ascend C Matmul 原语完成分块实矩阵乘，以实部与虚部组合得到复数乘积。对输入做三角裁剪、UNIT 对角补值及实虚分离，转置与共轭由地址映射和虚部符号处理。准备阶段、矩阵乘阶段与输出组合阶段均为本任务 kernel；计划不调用已有 aclblasCtrmm、系统 TRMM 或 CPU 运算来实现生产计算。

拟比较四次实乘与归约维拼接方案的精度和资源开销。大尺寸按三角范围缩减 K 区间，避免对必为零的整个三角做无效乘法。最终采用能在完整输入覆盖下满足精度与性能约束的通用方案。

## 内存和同步

拟将输入及输出按 UB/L1/L0 容量分块，完整处理非对齐尾块、padding 和矩形尺寸。GM 工作区由句柄管理，使用前核验容量；不足时提供可用的分块路径或按框架返回明确资源错误，不越界写入。

各阶段在同一 stream 顺序下发，kernel 内按数据流同步搬入、计算和写回。B≡C 时先保存输入或生成独立打包输入，在所有需要的原 B 数据可安全访问后才写 C。输出 padding 和边界哨兵应保持不变。

## 数值与特殊值

复数实部为 ar×br−ai×bi，虚部为 ar×bi+ai×br。拟保持 float32 计算语义，重点验证长归约、相消及大幅值输入；不以放宽任务阈值代替实现修正。Inf/NaN 将单独验证位置与符号，并核对与 Netlib 参考的分支语义。零 alpha 和 UNIT 对角路径不应读取被规范排除的数据。

# 验证计划

## 官方用例与补充覆盖

原始 CSV 共 1200 条，其中 1000 条精度类和 200 条 TC_PF 性能类。拟对全部 1200 条执行精度或预期状态校验，同时对全部 200 条性能用例采集耗时。保持原始 CSV 和 GPU baseline 字节不变，按完整参数与原顺序关联，保留重复参数对应的独立用例。

随机填充以官方种子为准，并补齐均匀与正态分布覆盖；拟分别执行完整的原分布和正态分布测试，特殊填充保持原意。正态分布参数和种子在测试代码与报告中明确。补充用例独立编号，覆盖空句柄、UNIT 对角污染、未使用三角污染、不同前导维、原地 B≡C、零 alpha 空输入、非对齐和未在官方列表中的尺寸。

精度 golden 使用外部构建的 Netlib BLAS cblas_ctrmm；生产代码不链接参考计算。参考库来源、版本、构建参数及动态库 SHA 应可追溯。测试先将 B 复制至 CPU 参考输出，再由 Netlib 原地计算，并逐元素检查 m×n 全部有效输出。

## 精度判定

实部、虚部分别使用 atol=rtol=2^-13；逐元素满足 |actual−golden| ≤ atol+rtol×|golden| 计为匹配。每个分量 matched_ratio 必须 ≥0.99，同时执行 1e-2 或 32 ULP 的绝对误差限制；不把两个分量混合后掩盖单个分量失败。NaN/Inf 匹配单列，状态及空输出用例的数值指标标为 N/A。

## 性能判定

采用 msprof 的 Device kernel Task Duration。拟每个性能点预热 5 次、有效采样 64 次，保留完整 64 次算术均值，满足附件“有效采样 >10 次”的要求。一次 API 内所有准备、矩阵乘和写回 kernel 均计时；排除 Host 分配、CPU golden、测试侧输入恢复 DMA。原地测试每次调用前恢复原 B，不将前次输出作为下次输入。

全部 200 个性能点要求 NPU_us ≤ GPU_ms×1000/0.8，即 ratio=GPU_us/NPU_us ≥0.8。五个任务书典型点还应满足以下上限，不能只报告这五点而遗漏其余点。

| 官方点 | 参数摘要 | 耗时上限 us |
|---|---|---|
| TC_PF_1001 | 256×256 LEFT UPPER N NON_UNIT | 104.6 |
| TC_PF_1002 | 512×512 LEFT LOWER C NON_UNIT | 235.2 |
| TC_PF_1003 | 1024×1024 LEFT UPPER T UNIT | 803.8 |
| TC_PF_1004 | 2048×2048 RIGHT LOWER N NON_UNIT | 3241 |
| TC_PF_1005 | 4096×4096 LEFT UPPER C NON_UNIT | 19931 |

任务书内存硬指标为“不涉及”。拟记录 Device 输入输出与工作区的实际申请量，明确不是进程峰值 HBM，不虚构内存性能门槛。TBE 栏将标为 N/A，性能参照为官方 cuBLAS baseline。

# 工作计划与交付

第一阶段核对接口、环境和原始附件，提交仅含计划的设计文档；第二阶段实现 Host、tiling、kernel 及 CSV 驱动测试，完成边界与全量精度；第三阶段按完整性能点优化和回归；第四阶段从固定源码建立新的正式批次，生成报告、真实截图、复现说明和包哈希。

计划使用 KousakaReina 账号提交活动仓设计 PR、建立私人代码仓并邀请 Ascend-CANN。设计 PR、私人验收仓及最终上游代码合入属于不同入口，报告分别记录真实状态。

最终包拟沿用 Ctbmv 的简明结构，包含计划书、设计与实现说明、完整 XLSX 自验收报告、原始任务文档、官方及带结果 CSV、测试复现 README、验收提交说明、证据目录和 MANIFEST.sha256。私仓 task_submission 同时提供独立的精度、性能和内存报告及日志。

真实截图拟通过本地 SSH 转发打开服务器终端，现场运行全量结果核验或汇总命令后从浏览器截取，保留终端原始输出、命令和图片哈希。只报告一个完整正式批次；调试轮次与失败记录在本地归档，不混拼多个版本的最优结果。
