# aclblasCgemmEx 算子设计文档（社区任务 · 评审用简版）

> 任务：算子实操工坊-上海站 · aclblasCgemmEx 算子开发（950）
> 团队：dzzw123　芯片：Ascend 950PR（arch35）　CANN：9.1.0
> 说明：本文为设计评审用文档，聚焦需求理解与方案方向；实现级参数与实测数据以代码 PR 与自测报告为准。

# 需求背景

## 需求来源
社区任务《aclblasCgemmEx 算子开发(950)》：在 ops-blas 仓为 Ascend 950PR（arch35）新增
BLAS 复数广义矩阵乘扩展接口 `aclblasCgemmEx`，语义对标 cuBLAS `cublasCgemmEx`。

## 背景介绍

### 对标接口现状分析
cuBLAS 提供 `cublasCgemm`（固定复数类型）与 `cublasCgemmEx`（混合精度扩展：A/B/C 类型
可独立指定，覆盖 32F/16F 实数与复数组合）。ops-blas 已有 `aclblasCgemm`（纯 C_32），
缺少允许调用方按吞吐/带宽需求选择操作数精度的 Ex 变体。

### ops-blas 仓 arch35 现状分析
arch35（950 系列）gemm 族已有实数算子框架：host 侧 tiling/workspace 管理 +
device 侧 cube 流水的目录结构与构建注册方式均可复用；复数 Ex 类型路径为空白。

### 功能分析
- 计算：C = alpha · op(A) · op(B) + beta · C，列主序，前导维独立；
- op() ∈ {N, T, C}（不转置/转置/共轭转置），alpha/beta 为复标量；
- typeA/typeB/typeC ∈ {R_32, C_32, H_R_32, H_C_32}（新增 `aclblasType_t` 枚举，
  取值对齐 cudaDataType，便于生态迁移）。

### 硬件层面的实现难点
1. 复数矩阵乘无法在单个实数 MMA 指令上直接表达，需分解为实数 GEMM 组合；
2. 950PR 上 FP32 数据通路的可用吞吐不足以直接支撑复数大 shape 标杆，方案需要
   借助更高吞吐的实数数据通路并解决其精度表达问题；
3. 混合类型（实/复 × 32F/16F）组合的语义与升/降精度截断规则需逐类界定；
4. 大 shape 下中间累加结果的 workspace 占用受库级上限约束。

### 交付范围界定
交付 `aclblasCgemmEx` 接口 + arch35 实现 + CSV 驱动的 ST 测试工程 + 自测报告；
不改动既有算子行为（纯增量）。

# 需求分析

## 需求描述
按任务书：4 种数据类型 × 3 种转置组合、列主序前导维、复标量 α/β；边界与负向参数
返回正确状态码；精度按任务书 MIXED_TOLERANCE 双口径判定；性能满足任务书 4 条标杆。

## 需求拆解
1. 接口与枚举：`cann_ops_blas.h` 声明 + `aclblasType_t`；
2. 数值核心：复数 GEMM → 实数 GEMM 的分解与精度重构；
3. 类型适配层：typeA/B/C 组合的载入/转换/截断语义；
4. host 侧：参数校验、tiling、workspace 规划（含大 shape 降级预案）；
5. 测试工程：1200 条 CSV 用例（骨架/尺寸/标量/矩形/ld/填充/组合/边界负向/扩展类型/性能族）。

# 详细设计

## 算子分析

### 数学公式
C = alpha · op(A)·op(B) + beta·C；复数乘通过 Gauss 型恒等式分解为若干个实数 GEMM
（乘法次数少于朴素 4-GEMM 方案），实部/虚部由分解结果线性组合还原。

### 支持数据类型
typeA/typeB/typeC ∈ {R_32, C_32, H_R_32, H_C_32}；typeC 为实类型时按测试工程口径
取实部（截断语义，文档与用例固化）。

### 支持形状
m/n/k 正整数 + 各 lda/ldb/ldc ≥ 对应维；k=0、m/n=0 走 C=beta·C 快速路径。

## 算子实现

### 实现方案（总览）
全程设备侧执行（无 host 数据往返），由 host 完成参数解析与 tiling，device 侧以
"分解 → 多内核流水"实现：

1. **数值扫描与平面拆分**：对输入做幅值扫描，将实数平面按运行时缩放拆分为
   高/低两部分窄类型表示，使后续可走高吞吐低精度数据通路、并在合成端还原
   fp32 级精度；共轭与解交错在同一遍完成；
2. **cube 实数 GEMM**：Gauss 分解出的实数 GEMM 以高/低部分的多趟乘加组合，
   fp32 累加；K 方向按分块（split-K）控制累加误差与并行度；
3. **合成与回写**：反缩放、实/虚组装、α/β 复合、typeC 降精度输出；
4. **极端数值兜底**：对幅值扫描命中的极端量级分块做条件精确重算，保证
   近抵消/溢出类元素的判定口径。

### host 侧设计
- 参数校验顺序与状态码对齐既有 aclblas 家族（非法枚举/负维/空指针/ld 不足）；
- workspace 采用解析式规划：平面缓冲 + 累加临时区 + 扫描辅助区，总量受库级
  上限约束；超出时按 M 方向分带（banding）循环下发，复用单带临时区，小 shape
  等价于不分带路径；
- tiling：按核数对 M/N 做负载均衡切分，尾块补齐到指令基块。

### kernel 侧设计
设备侧为多个协作内核（扫描/拆分、cube 主计算、合成、条件重算），cube 主计算采用
双缓冲流水以隐藏搬运延迟；具体块尺寸、趟数与调度参数属实现细节，见代码 PR。

### 扩展类型适配
H_* 入口在拆分阶段升 fp32 处理后按 fp16 网络表示，出口按 typeC 口径降精度；
R/C 混合输出按上述截断语义。

### 边界行为实现
alpha=(0,0) 或 k=0：不经过 cube 的 C=beta·C 快速路径，位精确语义；
Inf/NaN 传播与 golden（Netlib）行为对齐。

### 支持硬件
Ascend 950PR / arch35（DAV_3510），CANN ≥ 9.1.0。

### 算子约束限制
多核分块累加顺序不固定，不承诺跨运行位级确定性（EXACT 快速路径除外）。

# 可维可测分析

## 精度标准/性能标准
- 精度：任务书 MIXED_TOLERANCE（rtol=2^-10 / atol=2^-16 / matched_ratio≥0.99，
  复数实虚部分别统计，逐元素硬上限）与 CSV MERE/MARE 双口径；alpha=(0,0) EXACT；
- 性能：任务书 4 条标杆 case（1024³/2048³ 组合），以算子端到端 kernel 时间判定；
- 内存：workspace 解析式 + 真机进程峰值口径双轨记录。
（各项实测结论见交付自测报告，本文不引用具体数值。）

## 测试用例规划
GTest + CSV（1200 条，固定种子生成）：TC_L0 骨架、TC_SQ 尺寸扫描、TC_AB 标量、
TC_RC 矩形、TC_LD 前导维、TC_FL 极值填充、TC_CV 组合全覆盖、TC_ED 边界负向、
TC_EX 扩展类型采样、TC_PF 性能族（4 标杆 + 形状扫描）。

## 兼容性分析
纯增量：新增接口/枚举/文件，不改动既有算子与共享头文件；`ASC_DEVKIT_GE_9_1` 宏隔离。

## 风险与降级预案
1. 高吞吐通路的精度表达 → 窄类型高/低部分拆分 + 运行时缩放 + 分块累加（已采纳）；
2. 大 shape workspace 超上限 → 按 M 分带循环（已落地验证）；
3. 极端量级元素 → 扫描 + 条件精确重算兜底（已落地）；
4. H_* 类型组合的生态语义存在实现选择空间 → 以用例与文档固化口径，评审确认。

## 文件与接口落位清单
- `include/cann_ops_blas.h`、`include/cann_ops_blas_common.h`（接口与枚举）；
- `blas/gemm/arch35/cgemm_ex_{host,kernel,tiling_data}.*`（实现）；
- `test/gemm/cgemm_ex/`（ST 工程与 CSV）；
- `blas/gemm/README.md`、`cmake/asc_devkit_version.cmake`（文档与注册）。

# 附录

## 参考资料
- cuBLAS `cublasCgemmEx` 文档；Netlib BLAS cgemm 参考实现；
- 任务书与 design_template（cann-ops-competitions）。

## 修订记录
- v0.2（2026-09-20）：公开简版，方案方向与实现细节收敛至代码 PR。
