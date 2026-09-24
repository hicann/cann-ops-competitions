# 需求背景（required）

## 需求来源

- 任务：9 月社区任务 `aclblasCgemmStridedBatched` 算子开发（A2/A3）。
- 开发者及提交账号：Gaomuer。
- 输入依据：任务配套 `aclblasCgemmStridedBatched_A2A3_task_doc.md`、`test_cases/README.md`、1200 条官方 CSV、200 条 GPU 基线及 Python 工具。
- 设计文档按社区 `resources/design_template.md` 的章节组织；参考同任务 [PR #1816](https://gitcode.com/cann/cann-ops-competitions/pull/1816) 的目录结构，增加 9 月前缀 `09-`，提交至 `04_tasks/01_community-task-2026/tasklist/09-aclblasCgemmStridedBatched-A2A3/Gaomuer/docs/design.md`。
- 实现目标仓：`cann/ops-blas`。开发语言为 Ascend C，采用 handle 绑定 stream 的 kernel 直调工程模式。
- 本文是实现与验证方案，不包含尚未执行的实测结论。实验采用 Atlas 800T A2（910B3）、CANN 9.1.0；代码按 arch22 的 A2/A3 共同能力设计。

## 背景介绍

### 算子实现目标

为一组尺寸相同、存储起点按固定步长变化的单精度复数矩阵提供批量矩阵乘。与逐 batch 从 Host 重复调用 GEMM 相比，批量调度可减少下发开销，并让小矩阵通过 batch 维度填满计算核。应用包括复数信号处理、科学计算及频域线性代数。

### 当前工程分析

已检查本地 ops-blas 参考快照：现有 `blas/gemm_strided_batched/arch35/` 为实数 FP32 路径，其 handle、校验及三阶段组织可参考，但 arch35 的硬件指令、Blaze 实现和常量不能直接移植到 arch22。COMPLEX64 的共轭、交错存储及四路实数积需要新增实现。正式编码前以目标仓最新兼容分支核对公开声明与构建入口，避免重复导出符号。

本任务不是 TBE/ACLNN 自定义算子工程；不新增无关 `op_host/op_kernel` 外壳，不要求 PyTorch/ATen 适配。

### 算子功能分析

对每个 batch `b`：

`C_b = alpha * op(A_b) * op(B_b) + beta * C_b`。

`A_b=A+b*strideA`，`B_b=B+b*strideB`，`C_b=C+b*strideC`。步长单位是复数元素，1 个 COMPLEX64 元素包含两个 FP32 分量，共 8 字节。`N/T/C` 分别表示不转置、转置、共轭转置；所有 batch 共用尺寸、转置模式和复数标量。

# 需求分析（required）

## 需求描述

实现下列公开接口，保持参数顺序、类型和返回码与任务书及 ops-blas 定义一致：

```cpp
aclblasStatus_t aclblasCgemmStridedBatched(
    aclblasHandle_t handle, aclblasOperation_t transa,
    aclblasOperation_t transb, int m, int n, int k,
    const aclblasComplex *alpha, const aclblasComplex *A,
    int lda, long long strideA, const aclblasComplex *B,
    int ldb, long long strideB, const aclblasComplex *beta,
    aclblasComplex *C, int ldc, long long strideC, int batchCount);
```

| 参数组 | 约束与处理 |
|---|---|
| handle | 空指针返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`；使用库已有上下文、设备和 stream 管理 |
| transa/transb | 仅 N/T/C 合法，其他返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| m/n/k/batchCount | 非负；任一为 0 时按本任务定义 no-op |
| alpha/beta | Host COMPLEX64 标量，指针不可为空；提交时复制标量值，不在异步 kernel 内解引用 Host 地址 |
| A/B | Device 只读；alpha 非零且确需运算时不能为空；alpha=0 时不访问 |
| C | Device 输出；非空输出时不能为空；beta=0 不读取初值 |
| lda | N 时至少 max(1,m)，T/C 时至少 max(1,k) |
| ldb | N 时至少 max(1,k)，T/C 时至少 max(1,n) |
| ldc | 至少 max(1,m) |
| stride | 64 位有符号元素偏移；不以 32 位变量保存 batch 地址乘积；A/B 的零步长复用合法 |
| 布局 | 列主序；支持合法 ld padding、batch 间隙与偏移；C 不与 A/B 重叠 |

## 需求拆解

| 编号 | 要求 | 实现/验证安排 |
|---|---|---|
| R01 | 复数运算、N/T/C 九种组合 | 四路实数 GEMM、共轭符号处理、Netlib cgemm Golden |
| R02 | batch、ld、stride 泛化 | 运行时 tiling、64 位寻址、广播及非紧凑测试 |
| R03 | no-op、标量特殊值、异常 | Host 校验顺序、独立 fill/scale 路径、官方 ED 用例及补充 UT |
| R04 | COMPLEX64 精度 | 实/虚分量各自检查混合容差与最大绝对误差，不放宽官方阈值 |
| R05 | 官方全部测试 | 1000 条精度逐条运行，200 条 PF 逐条精度预检与性能采样 |
| R06 | 输入分布 | 保留官方均匀分布数据，补充实/虚独立的均匀/正态各 50% 测试 |
| R07 | 性能 | 全部 200 条对齐 GPU 基线，NPU Avg us ≤ gpu_ms×1000/0.8 |
| R08 | 异步、复用、内存边界 | handle stream、跨流独立 workspace、padding 哨兵与重复执行 |
| R09 | 工程与产品适配 | arch22、公开头声明、README 标注 A2/A3 支持设计；A2 真机实验 |
| R10 | 交付 | 模板设计、自测步骤、精度/性能/内存报告和日志统一放入 task_submission |

### 任务资料的口径处理

1. **目录**：§2.2 的 `blas/gemm/` 与 §5 不同，采用 §5 的具体交付路径 `blas/gemm_strided_batched/arch22/`；测试采用 `test/gemm_strided_batched/cgemm_strided_batched/arch22/`，构建接入时保持仓库既有目标习惯。
2. **广播**：以 §2.4 和配套用例为准支持 A/B 零步长，不采用 §2.5 “无广播”的概括性描述。
3. **输出零步长**：任务书称 strideC=0 合法，测试 README 又将多 batch 输出重叠视为未定义。设计不以零步长本身报错：batchCount=1 正常处理；多 batch 的 strideC=0 采用按 b 递增的同 stream 串行回写，避免数据竞争，明确这是实现选定的重叠处理顺序，不将其称为 cuBLAS 保证。官方 PF 不含该组合。其他部分重叠仍按调用方约束处理，不新增“步长过小必报错”的规则。
4. **零 K**：遵守本任务明确的 no-op，不调用一般 BLAS 的 `beta*C` 退化计算替代它。Golden 包装层先执行任务 no-op 语义，其他正常运算才调用 cblas。
5. **采样次数**：正式每项 5 次预热、20 次有效采样，满足 §4/§7 的 “>10 次”；同时保留有效序列前 10 次均值，供 §3.3 的 10 次口径核对，两种结果均不得择优删除样本。
6. **内存**：§3.4 无单独达标阈值，但 §4 明确需要内存交付，仍记录输入输出字节、workspace、设备峰值和释放情况，不虚构阈值。
7. **实测硬件**：按本次实验安排在 A2/910B3 验证。A3 采用共同 arch22 能力适配；设计兼容性不等于已有 A3 真机结果，不以封装方式推导指令/性能完全一致。

# 详细设计（required）

## 算子分析

### 数学公式

设 `op(A)=Ar+iAi`、`op(B)=Br+iBi`，则：

`Pr = Ar*Br - Ai*Bi`，`Pi = Ar*Bi + Ai*Br`。

令 `alpha=ar+i ai`，`beta=br+i bi`，则：

`Cr' = ar*Pr - ai*Pi + br*Cr - bi*Ci`

`Ci' = ar*Pi + ai*Pr + br*Ci + bi*Cr`。

读取 C 时先保存一对原实/虚分量，再计算两个输出，不让原地覆盖污染虚部。beta=0 的 epilogue 不发出 C 读指令。初始方案使用四次实数乘积，不使用增加相消误差的 3M 公式。

### 支持数据类型

A/B/C、alpha/beta 均为 COMPLEX64，实数乘积与累加为 FP32。不得将 COMPLEX64 偷换成单路 FP16/BF16 乘法；快速路径的乘法精度模式必须显式配置并通过全量精度校验。启用 HF32 或拆分近似算法不是默认方案。

### 支持形状与地址映射

不将官方尺寸列表作为支持白名单。合法运行时参数在可分配地址空间与实现的安全整数范围内均按统一逻辑分块处理。

列主序元素地址：`base[b*stride + row + col*ld]`。实际输入形状如下：

| 矩阵 | N 的物理行×列 | T/C 的物理行×列 | 逻辑读取 (r,s) 或 (s,c) |
|---|---|---|---|
| A | m×k | k×m | N: r+s*lda；T/C: s+r*lda |
| B | k×n | n×k | N: s+c*ldb；T/C: c+s*ldb |
| C | m×n | 不涉及 | r+c*ldc |

共轭只在 `C` 模式下翻转输入虚部符号；`T` 不做共轭。A/B padding 和 batch 间隙不参与计算，C padding 不写回。输入首址偏移不保证 32 字节对齐，搬运按受支持的非对齐接口或安全尾处理实现。

对负 stride 使用有符号 64 位偏移，不转成无符号数；调用方须提供覆盖全部实际地址的存储和适当的初始指针。batchCount=1 时 stride 不参与寻址。仅校验可检测的整数乘加/字节转换溢出，不推测 Device 分配对象边界。

## 算子实现

### 实现方案

#### 3.2.1 host侧设计

**校验及分流顺序**：

1. 检查 handle，然后检查枚举、负维度、前导维度和 alpha/beta 指针。
2. 任一 m/n/k/batchCount 为 0：返回 SUCCESS，不引用 A/B/C，不下发 kernel。
3. 非空结果检查 C；读取 Host 标量。alpha=0 时绕过 A/B 指针和数据访问：beta=0 填零，beta=1 原样返回，其余执行复数 scale。
4. alpha 非零时校验 A/B，构造安全地址范围和 tiling；任何尺寸乘积、workspace 长度或偏移计算不得发生整数回绕。
5. 根据公开的尺寸、布局、标量和设备能力选择通用 Vector 或分块 Cube 路径。分支不读取 case ID、种子或预期结果。
6. 在 handle 当前 stream 上按序下发。不得在正常 API 中同步整个 Device；测试读取结果时由调用者同步。

如果官方边界用例要求特定错误优先级，逐条锁定到 UT；不以 quick return 绕过明确要求的非法 handle/枚举检查。

**tiling 参数**：m/n/k、batchCount、lda/ldb/ldc、strideA/B/C、两组标量实虚部、转置/共轭标记、M/N/K tile、每核任务数、尾块大小和 workspace 偏移。维度与地址元数据用合适的 64 位类型，仅在验证范围后转换到指令的有限宽参数。

**任务分核**：逻辑任务 `q=b*ceil(m/Mt)*ceil(n/Nt)+tileId`，按核号轮转分配；每个输出 tile 由一个逻辑 owner 写回。小矩阵优先在 batch 维并行，大矩阵在 M/N 维并行。默认不做跨核 split-K，从而避免全局原子加、非确定性额外误差和过多中间结果。

**分流候选**：

| 路径 | 适用场景 | 设计 |
|---|---|---|
| no-op | 零维/零批 | Host 返回 |
| fill/scale | alpha=0 | 单独 Vector kernel，禁止读 A/B |
| 小尺寸 Vector | 小矩阵、高 batch、Cube 启动成本占优 | 多输出向量批处理；K 外层循环，避免每个元素单独标量 GM load/store |
| 分块 Cube | 中大规模 | 复数解交错/转置打包，四路 FP32 实数 GEMM，融合复数 epilogue |
| 通用 FP32 路径 | 特殊布局、数值或资源不满足快速路径 | 安全搬运及 FP32 Vector 归约，保证支持范围不依赖已知测试 |

阈值根据 A2 实测选择并以注释解释，不能凭估计写入达标结果。初始 tile 搜索考虑 32/64/128 的 M/N 候选与 32/64/128 的 K 分块，最终由 API 支持、寄存器和各级容量约束筛选，不承诺某一固定 tile 适用于全部矩阵。

**资源与异步生命周期**：复用仓内 handle/workspace 管理机制；按需使用有上限的 batch/tile 分组，避免为最大 batch 分配四份完整输出矩阵。workspace 只在所属 stream 的使用完成后释放或复用；两个 handle/stream 不共享可写临时区域。重复调用可以缓存设备无关 tiling，但不能缓存 A/B/C 的计算值。内存申请失败返回仓内对应状态码，不吞掉下发错误。

#### 3.2.2 kernel侧设计

**主流水**：GM 输入 → MTE2 搬运 → UB 解交错/共轭/转置 → Cube 可用的实数 tile → L1/L0A/L0B → Cube FP32 累加/L0C → 复数 epilogue → GM C。

采用 CANN 9.1.0 的 arch22 Ascend C Matmul/基础搬运能力。涉及 `Matmul`、`TCubeTiling`、`TPipe/TQue`、`DataCopy/DataCopyPad`、向量乘加与归约时，以目标 Toolkit 安装头文件、SoC 支持表和编译探针确认接口；不使用仅 arch35 支持的 SIMT 或 Tensor API。

1. **CopyIn**：每次只搬入合法 M/N/K 范围；尾块补零仅发生在本地 buffer，不能越界读用户 GM。A/B 零 stride 可在一次调用内复用打包结果，但不得跨调用复用输入值。
2. **复数分量处理**：交错 `[real,imag]` 在 Vector 中转为平面分量；通过编译期 N/T/C 模板避免热循环逐元素分支。转置由布局转换/Matmul 能力承担，共轭合并到虚部分量变号步骤。
3. **四路乘积**：依次形成 Pr/Pi，FP32 保存部分和，K 分块累加不降精度。FP32 Matmul 是否隐式启用 HF32 必须检查实际模式，不能仅根据输入类型认定精度。
4. **数值保护**：对长 K、严重相消与极值建立针对性测试；必要时采用 FP32 分段/补偿累加。非有限输入遵循数值语义，不得将 NaN/Inf 当成零。任何优化必须同时通过普通、特殊值及大 K 测试后才能启用。
5. **Epilogue**：融合 alpha/beta 复数乘加、交错写回及尾块 mask；beta=0 使用不读 C 的专用分支。写回精确覆盖每列前 m 行，不改 padding/间隙哨兵。
6. **双缓冲**：输入 tile 使用 ping/pong，DMA 完成才能消费，计算完成才能覆盖；输出 buffer 在搬出完成前不能复用。只在本地存储预算允许时开两级缓冲，非对齐路径保留同样的依赖关系。
7. **流水证据**：通过 msprof op 检查搬运、Vector、Cube、同步等待和时间线；仅观察到 MTE 非零不足以证明双缓冲重叠，必须比较实际重叠及总耗时。

各级存储预算按实际平台查询及 tiling API 决定。以 FP32 估算，单路实数 A/B tile 占用 `4*(Mt*Kt+Kt*Nt)` 字节，双缓冲翻倍；复数分量同时保留再乘分量数，Pr/Pi 至少 `8*Mt*Nt` 字节。该估算分别对应不同存储层，不把 L0C、UB 和 L1 容量混为一谈；同时留出 API 临时区和对齐开销。全部设备核数和存储上限在 Host 查询，不照搬 950PR 常量。

### 工程目录与最小改动范围

```text
ops-blas/
├── include/cann_ops_blas.h
├── blas/gemm_strided_batched/
│   ├── README.md
│   └── arch22/
│       ├── cgemm_strided_batched_host.cpp
│       ├── cgemm_strided_batched_kernel.cpp
│       ├── cgemm_strided_batched_kernel.h
│       └── cgemm_strided_batched_tiling_data.h
├── test/gemm_strided_batched/cgemm_strided_batched/
│   ├── cgemm_strided_batched_param.h
│   ├── cgemm_strided_batched_golden.h
│   └── arch22/
│       ├── cgemm_strided_batched_test.cpp
│       ├── cgemm_strided_batched_npu_wrapper.h
│       ├── gemm_strided_batched_test.csv
│       └── gpu_baseline.csv
└── task_submission/
    ├── 1 自验证步骤说明.md
    ├── 2.1 精度自验证报告.xlsx
    ├── 2.2 精度自验证日志.log
    ├── 3.1 性能自验证报告.xlsx
    ├── 3.2 性能自验证日志.log
    ├── 4.1 内存自验证报告.xlsx
    └── 4.2 内存自验证日志.log
```

文件拆分以实际复用为准，不为凑目录复制整套框架。性能采集/报告汇总脚本置于既有测试工具目录或该算子测试目录，构建配置只修改必要入口。参考仓同族实数 CSV 不覆盖、不替换；官方脚本默认安装路径与本任务嵌套目录的差异通过明确的路径适配解决并记录 diff。

## 支持硬件

| 产品系列 | 适配设计 | 本次自验安排 |
|---|---|---|
| Atlas 800I/T A2 | arch22，支持 | Atlas 800T A2 / 910B3 实测 |
| Atlas A3（含 Atlas 800I A3） | arch22 共同 API，支持 | 复用共同实现，本次实验在 A2 进行 |

按 2026-09-23 只读预检，`ssh 910b` 可连接，8 张 910B3 健康状态均为 OK；3 号卡有其他任务。空闲快照不构成长期占用权，测试前重新检查设备进程与利用率，发现占用即让步。采用独立容器仅映射选中卡；物理卡号和容器逻辑号均记录到报告。不停止他人容器，不重启服务器，不改变他人环境。

已列出的镜像标签主要为 CANN 8.5/9.0，尚未据此确认存在合规 9.1.0 工具链。准备阶段优先核验现有自有环境；不足则新建适配宿主架构和驱动的 CANN 9.1.0 开发容器，固定镜像摘要/安装包校验值。不能用 9.0 测量结果冒充 9.1.0。设计阶段不预占 NPU。

## 算子约束限制

仅提供本任务 COMPLEX64、列主序、Host 标量和固定步长批量语义，不扩展到其他 dtype、行主序公开接口或不规则每 batch shape。A/B/C 所指存储由调用方保证有效，C 不允许与 A/B 重叠；不要求对全部 Device 内存别名进行昂贵运行时探测。支持 ld/stride 表达的非紧凑存储，不支持任意张量维度步长。

# 可维可测分析

## 精度标准/性能标准

| 验收项 | 标准 | 来源及落实 |
|---|---|---|
| Golden | Netlib BLAS cgemm，逐 batch | 任务书 §3.1/3.2，固定库版本与链接来源 |
| 混合容差 | rtol=atol=2^-13，matched_ratio≥0.99 | 实、虚分量分别计算，通过率不跨 batch 掩盖单 batch 异常 |
| 最大误差 | 1e-2 或 32×ULP | 同时报告绝对值与 ULP；有限分量采用逐点 max(1e-2,32×ULP(golden)) 硬上限，并额外报告固定 1e-2 的检查结果，不用大值的全局 ULP 掩盖小值错误 |
| 性能 | 全 200 条，NPU Avg us ≤1000×gpu_ms/0.8 | 全调用 kernel 耗时，5 次预热、20 次有效采样 |
| 内存 | 无任务独立阈值 | 仍交付峰值/临时区/释放与越界检查数据 |

任务书关于“可酌情放宽至 2ULP”的说明与前表存在表述差异，本设计不据此自动放宽阈值。NaN 按对应分量均为 NaN 判定，Inf 检查位置与符号，有限/非有限错配直接失败；非有限分量不进入相对误差除法。复数模长误差可作为附加诊断，不能代替实虚分量判定。

### 官方数据完整性

官方 CSV 已静态核对：总 1200 行，其中精度 1000、PF 200；GPU 基线 200 行且所有 gpu_ms 为正。两份文件的 200 条 PF 按顺序的 transA/transB/m/n/k/batchCount 均匹配。基线 ID 与 CSV case_name 不同，生成并固定显式映射表，以参数和次序双重校验；重复 shape 不能仅按 shape 关联。ld、stride、alpha/beta 和 seed 取自原始 CSV，一并保存为指纹。

| 文件 | SHA-256 |
|---|---|
| gemm_strided_batched_test.csv | `2a6e955930c8b7e650036e261ece0f94f56dee14e2ef9c37c5687e277630b7e3` |
| gpu_baseline.csv | `b759a976777ba6d161c060de9f808fc1e0a4057dc2268f42f57cbb5ad1966ab8` |
| gen_csv.py | `41238e365d5e7d26b3e4d4f1a58a17907225e958f5cc45d5743e81300a3f05cf` |
| verify_accuracy.py | `297e5dee34e0d3455bad5fb47d39b05c5bf28704e9645f584fac9df47e4ea1ae` |

不在原目录运行生成器覆盖已填写 GPU 数据的基线；补充数据输出到单独文件。原始附件只读保留。

### 精度与功能测试计划

| 官方类别 | 条数 | 覆盖 |
|---|---:|---|
| L0 | 9 | N/T/C 正交组合 |
| SQ | 22 | 尺寸扫描 |
| AB | 24 | 复数标量、零值与单位值 |
| BC | 13 | batch 规模 |
| RC | 16 | 矩形 |
| LD/SD/BR | 4/2/3 | 前导维、非紧凑步长、输入广播 |
| FL | 6 | 填充、极值、Inf/NaN |
| CV | 36 | 中等尺寸与转置组合 |
| ED | 33 | no-op、非法值、空指针等 |
| EX | 832 | 确定性扩展采样 |
| PF | 200 | 每条精度预检后采集性能，不只测试前五条 |

官方生成器注释确认 `RANDOM_NORM_5_5` 实际为 [-5,5] 均匀分布，不能因名字含 NORM 认定已有正态分布。新增独立补充集：固定 1000 个合法参数样本，500 个均匀、500 个正态；A/B/C 与 alpha/beta 的实虚独立生成并记录分布参数和 seed。特殊零标量、Inf/NaN、负向样本单列，不计入 50% 配额。补充包括大 K 相消、长尾 tile、转置边界、未对齐首址、负 stride 有效地址、batchCount=1 极大 stride、C padding 哨兵、多流并发及生命周期。

官方 `verify_accuracy.py` 仅解析 GTest 文本，存在超时/未解析用例可能返回成功、未检查子进程退出码及未核验总数的问题。保持官方用例不变，在适配脚本/外围校验中要求：GTest 退出码为 0、JUnit 无失败/错误/意外跳过、官方 1000 个 ID 恰好各一次、PF 200 个 ID 恰好各一次。缺项、重复、超时、零条运行或过滤错误均失败。修订脚本记录差异，不以自建更小集合替代原件。

Golden 包装处理 no-op、alpha=0 不引用、beta=0 不读取语义，再逐 batch 调用 Netlib cblas_cgemm；保留原始 C 供 Golden，不能将 NPU 已更新 C 作为输入。检查 A/B 只读、C 外围及 padding 哨兵。跨 kernel 切换和重复调用覆盖相同参数不同数据，排除缓存结果特化。

### 性能测试与优化闭环

| PF | batch | m/n/k | 转置 | 实际官方 padding | 任务书上限 us |
|---|---:|---|---|---|---:|
| 1001 | 8 | 256/256/256 | N/N | 紧凑 | 153.7 |
| 1002 | 32 | 512/512/512 | N/N | 紧凑 | 2757 |
| 1003 | 16 | 1024/1024/1024 | N/T | 紧凑 | 10061 |
| 1004 | 8 | 2048/2048/2048 | T/N | ld=2056；stride=4210688 | 36073 |
| 1005 | 4 | 4096/4096/4096 | N/N | ld=4352；stride=17825792 | 143326 |

GPU 毫秒转换后除以 0.8 的精确上限与上述文本四舍五入值略有差异，前五项采用二者较小值判定，其余使用 CSV 精确上限。保留两列来源。至少给出每条 Avg、采样数、min/max、标准差和 GPU/NPU 比率；不以最快一次或中位数代替平均数。

性能附件没有跑测脚本，需补充基于官方 GTest 的逐 case msprof op 驱动。先枚举真实 GTest 名称，单 case profile；确认工具有效回放次数，最终每项恰好 20 个有效算子调用、预热排除。一次 API 若包含 pack/GEMM/epilogue 多个 kernel，按调用分组后求和再平均，不只统计某个 GEMM kernel。原始 OpBasicInfo.csv、调用关联、工具命令、设备/频率/温度及代码指纹一起归档。

beta 非零时每次调用恢复同一初始 C，恢复操作在计时范围外；不能在回放中不受控地反复累计而仍声称输入相同。若 msprof op 回放无法保持这种语义，使用显式可分组的重复调用采集并注明，不擅自把 beta 改成 0。Golden、Host 初始化、H2D/D2H 不进入 Kernel 时间，但实现内部必须的预处理和转换必须计入。

优化顺序：先确保所有语义/精度通过，再定位小矩阵下发、MTE 搬运、解交错、Cube 占用、epilogue 与同步瓶颈。逐项比较单/双缓冲、tile、batch 分组和搬运复用；只保留能降低同口径总耗时且全量精度通过的改动。不预设“Vector 占比高必定不是访存受限”，需结合访存路径、带宽、停顿和时间线判断。

### 构建与环境复现

使用 CANN 9.1.0 的正式编译器与驱动兼容组合，保存宿主架构、驱动/固件、Toolkit、编译器、Netlib/GTest 版本、镜像摘要和源码 commit。以目标仓 `build.sh --help` 的真实选项确定 Release 构建，算子选择为 gemm_strided_batched、SoC 为 ascend910b3；保存展开后的命令和 flags，不在设计中虚构尚未核验的参数。

主验收使用常规 Release 优化，不通过全局 fast-math、关闭 NaN/Inf 或放宽比较阈值取得性能。设计完成后依次完成头文件/Matmul精度模式编译探针、基础 UT、官方精度、补充分布、200 PF 全量、内存记录及最终干净构建回归。

### 交付与可维护性

`task_submission` 使用任务书规定的七个文件名称，报告按官方模板填写，日志可追溯至每个 case、seed、Golden 和二进制 SHA。工作区中间 build/profiler 产物与提交树分开，最终包不嵌套压缩包、不带凭据、不含无关算子工程。

设计 PR 以 Gaomuer 提交；正式代码先推送其 ops-blas 个人 fork，邀请 Ascend-CANN 为开发者后按社区流程提交验收，再进行上游需求 issue/代码 PR。竞赛仓只提交设计文档所需文件，不提前堆放实现或伪造测试报告。

## 兼容性分析

新增公开 CgemmStridedBatched 声明不改变已有 Sgemm/Dgemm 等 ABI。实现沿用 ops-blas handle、状态码及 stream 机制，保持 arch35 和现有实数测试不受影响。A2/A3 共用源码时按平台能力确定核数与 tiling，不以 A2 单卡实测推导另一产品的绝对性能。

## 参考依据

1. 本任务官方任务书、`test_cases/README.md` 和对应原始 CSV/脚本。
2. 社区设计模板：`04_tasks/01_community-task-2026/resources/design_template.md`；本地官方仓模板作为章节来源。
3. [ops-blas](https://gitcode.com/cann/ops-blas)：公开头、handle helper、同族算子及贡献规范。
4. [cuBLAS GEMM Strided Batched](https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-gemmstridedbatched)：列主序及批量接口参考；任务显式差异单列。
5. [Ascend C Matmul 使用说明](https://asc.gitcode.com/api/SIMD-API/adv_api/cube_compute/Matmul_Kernel/Matmul_usage.html)及目标 CANN 9.1.0 安装头文件：布局、类型和硬件能力核验。
6. [生态混合容差标准](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/mixed_tolerance_standard.md)：以任务声明的 COMPLEX64 阈值和正式标准实现逐分量判定。
