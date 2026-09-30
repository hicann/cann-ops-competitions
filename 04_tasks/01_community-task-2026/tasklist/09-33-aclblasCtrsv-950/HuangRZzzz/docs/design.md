# 需求背景（required）

## 需求来源

- 任务：2026 年 9 月 aclblasCtrsv 算子开发（Ascend 950PR），竞赛仓任务编号 09-33。
- 团队及提交账号：HuangRZzzz。
- 依据：任务方《aclblasCtrsv 950 算子开发任务书》、`test_cases/`（CSV、GPU 基线、测试指导）。
- 目标代码仓：[cann/ops-blas](https://gitcode.com/cann/ops-blas)，硬件 Ascend 950PR，CANN 9.1.0。
- 文档版本：2026-09-29。本文档为**已完成真机实现与自测**的设计说明：架构、关键决策与全部性能/精度数据均来自实测，自测结论不代表官方评审或验收状态。

## 背景介绍

### aclblasCtrsv 功能与工程现状

Ctrsv 求解复数单精度三角线性系统 `op(A) * x = b`：单右端 BLAS Level 2 接口，无 side/alpha/beta/batch 参数；输入右端 b 与输出解 x 共用 Device 存储（原地覆写）。

工程参考提交为 ops-blas 的 trsv 家族：公共头文件已有 `aclblasStrsv`（实数），`blas/trsv/arch35/` 已有实数 Strsv 的 Host/Tiling/SIMT 内核。本次新增复数版本 `aclblasCtrsv`，复用句柄/stream/构建注册/测试组织，独立实现复数运算、共轭、转置与边界语义；实数实现中 T/C 共用的做法不适用于复数。

实现方式为 Ascend C kernel 直调：`<<<>>>` 启动 + SIMT VF（vector core 线程），不涉及 TBE/ACLNN/算子注册 JSON。

# 需求分析（required）

## 需求描述

在 `include/cann_ops_blas.h` 新增公共声明（整数维参数保持 `int`，类型复用 `cann_ops_blas_common.h`）：

```cpp
aclblasStatus_t aclblasCtrsv(
    aclblasHandle_t handle, aclblasFillMode_t uplo, aclblasOperation_t trans, aclblasDiagType_t diag,
    int n, const aclblasComplex* A, int lda, aclblasComplex* x, int incx);
```

| 参数 | 位置与方向 | 类型/布局 | 约束与含义 |
| --- | --- | --- | --- |
| handle | Host 输入 | aclblasHandle_t | 有效句柄，使用绑定 stream；异步语义 |
| uplo | Host 属性 | aclblasFillMode_t | UPPER/LOWER，指定原始 A 的可引用三角 |
| trans | Host 属性 | aclblasOperation_t | N/T/C；C 额外取共轭 |
| diag | Host 属性 | aclblasDiagType_t | NON_UNIT/UNIT；UNIT 不读取 A 对角 |
| n | Host 输入 | int | n ≥ 0，n=0 为合法空操作 |
| A | Device 只读 | COMPLEX64 列主序 | 逻辑 n×n，物理 lda×n；实/虚部各 float32 |
| lda | Host 输入 | int | lda ≥ max(1,n)，支持行尾 padding |
| x | Device 输入/输出 | COMPLEX64 步长向量 | 输入 b、输出解；原地更新 |
| incx | Host 输入 | int | 任意非零步长，支持正负 |

## 需求拆解

1. 覆盖 uplo×trans×diag 全部 12 组组合，正确处理实/虚部与共轭。
2. 仅访问指定三角；UNIT 不加载对角；只更新 x 逻辑元素，不改写步长间隙、不写 A。
3. 支持运行时 n、lda padding、负 incx、零维、非对齐尾部与非法参数返回。
4. 以 Netlib `cblas_ctrsv` 为参考，实部/虚部分别核对精度标准。
5. 覆盖附带 1000 条功能、200 条性能用例，并补齐分布/毒化/步长保护/特殊值测试。
6. 五条典型用例满足任务书耗时上限，其余用例与 GPU 基线逐项对比。
7. 按目标仓规范交付：公共头文件、README、测试注册、精度/性能/复现证据。

# 详细设计（required）

## 算子分析

### 数学公式

令 `M = op(A)`，解 `M x = b`；下三角正序前代、上三角逆序回代：

```text
lower: x[i] = (b[i] - sum_{j<i} M[i,j] * x[j]) / M[i,i]
upper: x[i] = (b[i] - sum_{j>i} M[i,j] * x[j]) / M[i,i]
```

有效三角 `effectiveUpper = (uplo == UPPER) XOR (trans != N)`；UNIT 省略除法且不读对角。复数乘法 `(a+ib)(c+id)=(ac-bd)+i(ad+bc)`；共轭仅翻转虚部符号；复数除法按分子/分母复共轭相乘实现，除数为对角元（测试矩阵对角占优），不做钳位或以特殊值替换掩盖误差。

### 支持数据类型与形状

仅 COMPLEX64（内部按 2×FP32）。附带用例覆盖 n≤2048（精度）与 n≤4096（性能）；更大合法 n 由有界 UB 分片 + GM 波前处理，受设备内存与地址可表示范围约束。无广播；n 为运行时标量。

### 地址计算

以复数元素为单位、64 位中间量：

```text
A(row,col) offset = int64(col) * lda + row
x 起始 = incx < 0 ? int64(n-1) * (-int64(incx)) : 0        // 负步长由实现内部处理
N: M[i,j]=A(i,j)   T: M[i,j]=A(j,i)   C: M[i,j]=conj(A(j,i))
```

## 算子实现

### 实现方案

#### 3.2.1 Host 侧设计

- **分核与并行策略**：`Sec = (n<=1024 ? 64 : 128)` 行一段，`numSections=ceil(n/Sec)`，一个 AIV block 负责一个 section（求解顺序：LOWER+N、UPPER+T/C 正向，其余反向）；workspace 可用时 `numBlocks=numSections`（≤40 AIV），否则单 block 回退。
- **分块与内存策略**：UB 单块文件级数组按 `CtrsvLayout<Sec>` 切片（rhs/暂存 x/section 对角块/A-tile），`Sec` 模板化，host 与 kernel 用同一规则分派两实例。
- **路径规划**：workspace 存在时走多 block 段波前；否则单 block 顺序求解所有段（batch 化 catch-up）。host 每次下发前清零 flags 区（`numSections×128B`），并使用绑定 stream 保证清零/内核/后续操作有序。

#### 3.2.2 Kernel 侧设计

- **段波前**：段间依赖通过 GM 中每段独立 flag 槽（128B 填充，避免同 cache line 抖动）传递。发布：`stcg` 写 x（写穿）→ `asc_threadfence` → `syncthreads` → `asc_threadfence` → `atomic_add(flag)`；消费：`volatile` 轮询本段 flag → `ldcg` 回读 x（绕缓存）。跨核数据必须 stcg/ldcg，且 fence 必须在 syncthreads 之前，否则大 n 会出现读未发布数据。
- **段内求解**：32 行 panel 推进。panel 内 warp0 用 `shfl` 广播逐列 rank-1 消元（串行核心），随后 256 线程做本 panel 之下的尾部更新；两类访问均只读预置 UB 块。OP_N 走直三角访存（`secBlk[j*Sec+i]`）；OP_T/OP_C 走转置三角（`B[r][c]=A[c][r]`，C 带共轭）。
- **对角块 staging（性能关键）**：OP_N 由标量上下文用 MTE2 逐列 1-D burst 预搬（列主序镜像）；OP_T/OP_C 因 MTE 无转置能力，用 SIMT loader 以**行主序**（转置镜像）置入 UB，使转置侧 sweep 与尾部更新都满足 lane 连续寻址，消除 1KB 跨 lane 的 UB bank 冲突。
- **catch-up**：block s 逐前面段增量吸收（等 flag → `ldcg` 暂存 x_t → 行并行 4 路展开 rank-1 更新），与其它 block 求解并行推进；单 block 回退用宽 batch 降 barrier 次数。

### 性能优化方案

| 项 | 做法 | 实测 |
| --- | --- | --- |
| 转置 staging 布局 | OP_T/OP_C 改行主序，消除 UB bank 冲突 | 1003 −23%（571→438us）、1004 −23%（1259→966us） |
| 段长模板化 | n≤1024 用 64 段、n>1024 用 128 段（A/B 实测阈值） | 1001/1002 小幅收益；n=2048 保持 128 更快 |
| MTE 预搬（OP_N） | 纯拷贝交 MTE2，释放 SIMT LSU | 中性（消融：成本为数据延迟本身），保留 |
| 增量 catch-up | 与其它 block 求解重叠，不等全前缀 | catch-up 仍占 12~32%，为下一优化目标 |
| 已否决方案 | 全局 barrier 协作轮次、深层尾部行重叠、64bit 打包 shfl、R=4 分块 | 均实测更慢（见自测报告归档） |

四段消融（同窗口）：sweep（warp0 串行链）占 47~52%，catch-up 占 12~32%，尾部更新占 8~14%，staging 中性。

### 工程组织与构建安装

新增/修改（目标仓）：`blas/trsv/arch35/ctrsv_{kernel.h,kernel.cpp,host.cpp,tiling_data.h}`、`blas/trsv/arch35/ctrsv_design.md`、`blas/trsv/README.md`（接口章节）、`include/cann_ops_blas.h`（声明）、`test/trsv/ctrsv/**`（CSV 驱动测试 + golden + NPU wrapper）。

```bash
cmake --build build -j
./build/test/trsv/ctrsv/ctrsv_test --gtest_filter="-*TC_PF*"      # 全量精度
./build/test/trsv/ctrsv/ctrsv_test --gtest_filter="*TC_PF_100*"   # 性能
```

## 支持硬件

- Ascend 950PR / Ascend 950DT（arch35 / DAV_3510），AIV-only（`KERNEL_TYPE_AIV_ONLY`），CANN 9.1.0。

## 算子约束限制

- n ≥ 0；incx ≠ 0；lda ≥ max(1,n)；diag=UNIT 时 A 对角不被引用；不写 A，x 原地更新且不触碰步长间隙。

# 可维可测分析

## 精度标准/性能标准

- 精度：按任务书标准（complex64 实/虚部按 FP32：atol=2⁻¹⁶、rtol=2⁻¹⁰、matched≥0.99、maxabs≤1e-2/32ULP），以 Netlib cblas_ctrsv 为参考。
- 性能：五条任务书上限与实测（同窗口多轮 avg，单位 us）：

| 用例 | 形状 | 上限 | 实测 | 余量 |
| --- | --- | --- | --- | --- |
| TC_PF_1001 | n=512 | 146.33 | 132~144 | 2~10% |
| TC_PF_1002 | n=1024 | 272.53 | 242~256 | 6~11% |
| TC_PF_1003 | n=2048 UPPER/T | 526.22 | 432~445 | 15~18% |
| TC_PF_1004 | n=4096 LOWER/C | 1177.84 | 922~963 | 18~22% |
| TC_PF_1005 | n=4096 UPPER/N UNIT | 1030.61 | 873~937 | 9~15% |

## 测试输入与覆盖

- CSV 驱动功能用例（uplo×trans×diag×尺寸×incx×padding）**1002/1002 通过**（最终二进制复测，exit=0）；另以 host 独立前代/回代参考解交叉验证小尺寸多组合（n=129 覆盖 12 组合）。
- 性能用例 1001~1009，按同窗口多轮 avg/min 口径采集。

## 配套用例核对及修正计划

- 发现性能/精度判定脚本口径与用例描述存在个别不一致（如按墙钟计时、随机分布覆盖），已记录并保留原始资料；不影响本算子实现与门禁结论。

## 计时与复现

- 环境：容器内 CANN 9.1.0，Ascend 950PR；`cmake --build build -j` 后直接运行 gtest 过滤集。
- 性能口径：每用例 50 次采样取 avg（并记录 min/max），全部在**同一窗口**内完成对比（避免频率漂移干扰）。

## 交付与可维护性

- 与 trsv 家族同构：Host/Tiling/SIMT 内核分层，段长模板化，UB 布局集中定义；设计文档（`ctrsv_design.md`）记录架构、消融证据与被否方案，便于后续接续优化。

## 兼容性分析

- 仅新增接口与文件，不改动既有算子行为；`include/cann_ops_blas.h` 为增量声明；测试目录独立注册，不影响其它用例。

## 参考资料

- 《aclblasCtrsv 950 算子开发任务书》及 `test_cases/`（CSV、GPU 基线、测试指导）
- Netlib BLAS `cblas_ctrsv`
- ops-blas 仓库 trsv 家族实现与贡献指南
