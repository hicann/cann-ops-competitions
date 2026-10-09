# 需求背景（required）

## 需求来源

9 月社区任务 aclblasCtrsmBatched 算子开发（Atlas A2/A3，arch22）。接口对齐 cuBLAS 的 complex64 批量三角求解：解原地写回 B，不增加 Carray。

## 背景介绍

每个 batch 的矩阵地址独立，维度和枚举属性一致。小矩阵依靠 batch 和右端向量并行，大矩阵需要将三角依赖限制在面板内，并把余下更新转为 Cube 可执行的矩阵乘法。

# 需求分析（required）

## 需求描述

LEFT 求解 op(A[i])X[i]=alpha·B[i]；RIGHT 求解 X[i]op(A[i])=alpha·B[i]。支持 UPPER/LOWER、N/T/C 和 NON_UNIT/UNIT，A、B 均为列主序 Device 指针数组，alpha 为 Host 复数标量。

## 需求拆解

1. 完成全部 24 组枚举组合，支持合法 lda/ldb padding、非方阵和 batch 独立寻址。
2. 按配套 CSV 约定处理零维、零 batch、alpha=0、空指针和非法参数。
3. 使用 Netlib CBLAS golden 验证有效输出，并检查 padding 不被修改。
4. 采集一次接口调用的全部 kernel，生成精度、性能和内存记录。

# 详细设计（required）

## 算子分析

### 数学公式

op(A) 为 A、Aᵀ 或 Aᴴ。UNIT 对角的逻辑值为 1。对一个已求解面板 Xp，剩余块更新为 Btrail−Atrail,pXp（LEFT）或 Btrail−XpAp,trail（RIGHT）。复数乘法展开为实部 ArXr−AiXi、虚部 ArXi+AiXr。

### 支持数据类型和形状

输入输出 complex64，内部计算 FP32。m、n、batchCount 为运行时非负整数；非空任务 lda≥max(1,k)、ldb≥max(1,m)，k=m（LEFT）或 n（RIGHT）。批次间不广播，不允许 B 矩阵重叠。

## 算子实现

### 实现方案

#### 3.2.1 host 侧设计

先验证 handle、枚举和维度。零维或零 batch 在指针及前导维检查前成功返回。非空任务检查 alpha、B、lda/ldb，alpha 非零时检查 A；通过复制 Device 指针数组检查各元素。alpha=0 直接调用 B 置零 kernel。

通用路径为 splitA → splitB → 多个 panel/GEMM 阶段 → merge。同一 stream 保证阶段顺序，工作区通过 handle 复用。核心数量动态查询，启动核数不超过本阶段的任务单元数。

小尺寸大 batch LEFT（128≤m,n≤512、m%32==0、n%16==0、batch≥16、lda/ldb 防溢出）分流到 **MIX 单 kernel 融合路径**（`ctrsmbatched_mixed_kernel`）：列主序 LEFT 经 m/n 对调、LEFT→RIGHT、UPPER/LOWER 对调映射为原行主序 RIGHT 问题（trans 语义保持，含共轭转置），直接消费 Device 指针数组，单次 launch（MIX_AIC_1_2，每 batch 一个 block = 1 AIC + 2 AIV）内完成全部求解。工作区仍复用 handle 默认工作区；tiling 与 cube tiling 数组随 kernel 参数按值传入。alpha==0 依旧走 B 置零快速路径，不满足形状条件的输入回退通用路径。

两个 side 均使用列主序 Mat=op(A)。转置使原三角方向翻转：有效下三角在 LEFT 正向消元，有效上三角在 RIGHT 正向消元；其他情况反向消元。

##### 分核与数据分块

面板宽度 32。LEFT 每个 lane chunk 包含最多 128 个 B 列，RIGHT 包含最多 256 个 B 行；任务编号同时编码 batch 和 chunk。GEMM 任务按 batch 与输出 tile 划分，矩阵以 plain ND 视图传递给 Matmul API。

##### 工作区布局

定义 kAligned=max(ceil(k/8)·8,16)、kPad=ceil(k/32)·32、ldAligned=ceil(m/32)·32。

| 区域 | 单 batch 的 FP32 元素数 | 排布 |
|------|------------------------|------|
| Mat | 2·kAligned·kPad | 每列先 real[kAligned]，后 imag[kAligned] |
| Br / Bi | 每平面 ldAligned·n | 列主序，alpha 已折入 |
| LEFT strips | 每条 2·32·n，共两条 | 每行依次交错一个面板的实虚分量 |
| RIGHT strips | 每条 32·ldAligned，共三条 | Sr=−Xr、SiP=+Xi、SiN=−Xi |

分区起点按 512 字节对齐。大小计算检查 64 位乘加溢出；额外尾部空间覆盖分区对齐开销。最终内存报告同时记录实际 HBM 占用和重复调用后的变化。

#### 3.2.2 kernel 侧设计

- splitA：N 直接按列解交错；T/C 对 16 列组分块转置，C 同时对虚部取负。尾块只写有效行，防止跨越实虚平面。
- splitB：按列分段加载复数输入，形成实虚两个平面，按 alpha 的实数、纯虚和一般复数情况缩放。
- panel：按三角依赖遍历面板行或列，将同一个 A 系数应用于整组 B 向量；计算和回写使用向量指令及批量 DMA。
- LEFT 更新：两条 paired-K GEMM 分别产生复数更新的实部和虚部。
- RIGHT 更新：四条实 GEMM 使用三条 strip 组合成复数更新。
- merge：Gather 交错实虚分量，只回写每列有效的 m 个复数。

MIX 路径 kernel 侧（`mixed/` 目录，AIC/AIV 协作）：

- 双 AIV 分工：前端 AoS→SoA 转置与末端 SoA→AoS 回写按半分摊（mode1 跨核屏障同步）；面板求解按输出列对半分列，各自持有列半区。
- panel 三角求解（AIV）：nb=16 面板，交错 [实|虚] 行布局，对角用 Smith 商除法（防 1e±20 对角溢出），尾行更新 4 路展开 Axpy；解出的面板行解交织后经 Xneg 双缓冲平面写回 GM。
- trailing 更新（AIC）：direct GEMM 只更新下一面板相邻行后立刻放行（AIV 求解与 GEMM 流水重叠），远端行 pre-update 与 AIV 并行，LIM_GROUP=16 分组按需合并大 K GEMM；复数乘法用两条实 GEMM（交错 A 列 × [-Xr;Xi] / [-Xi;-Xr] 平面）实现。
- 跨核同步：FLAG_TRSV（AIV→AIC 面板就绪）/ FLAG_GEMM（AIC→AIV 更新完成）配对；AIV 等待窗口预载下一面板 A 对角块。
- UB 预算 192KB 内静态分配全部缓冲；面板系数 Gather 偏移表构建一次复用。

DMA 的 GM stride 以字节计，UB stride 以 32 字节块计。RIGHT 尾 chunk 仍保持固定的 UB 行距；panel 的尾块加载和存储只覆盖实际 panelSize。GEMM 的 B 视图沿 N 方向偏移列索引，不将列偏移乘以行跨度。

TPipe 为 kernel 函数局部变量；tiling 结构仅包含标量；无逐元素 SetValue。

## 支持硬件

| 产品 | 实现目录 |
|------|----------|
| Atlas A2 系列（含 Atlas 800I A2） | blas/trsmbatched/arch22 |
| Atlas A3 系列（含 Atlas 800I A3） | blas/trsmbatched/arch22 |

## 算子约束限制

不做奇异性检测。alpha 仅支持 Host 指针。允许浮点累加顺序差异，不承诺逐位确定性。工作区分配失败返回错误，调用方读回输出前须同步 handle 绑定的 stream。

# 可维可测分析

## 精度标准 / 性能标准

| 标准 | 要求 |
|------|------|
| 精度 | 实虚部分别使用 atol=rtol=2⁻¹³、matched ratio≥0.99、任务规定的绝对误差 / ULP 上限 |
| 性能 | 五条指定 case 的 NPU kernel 总耗时≤GPU 基线耗时/0.8；其余 200 条 PF 逐条记录 |
| 内存 | 记录输入、工作区及重复调用时的实际 HBM 占用；不设置任务书未要求的占用阈值 |

实际环境、最终通过数量和性能结果以随附报告及原始日志为准，设计文档不以估算代替实测。

## 兼容性分析

新增 complex64 接口，不改变已有 StrsmBatched 接口。公共声明位于 include/cann_ops_blas.h，测试遵循仓库 GTest + CSV 结构。

配套 CSV 来自带 C 输出的旧模板：null_carray/null_c_elem/invalid_ldc 分别映射为本接口的空 B 数组、空 B 元素和非法 ldb；其余输入、随机种子和精度阈值保持原用例含义。batchCount=0 按交接明确的 CSV 语义成功返回。
