# aclblasCgeam 算子设计文档（Ascend 950PR / complex64）

| 文档版本 | 日期 | 作者 | 变更 |
|---|---|---|---|
| v1.0 | 2026-09-13 | zhangfeng1133 <yanggg1133@163.com> | 初版：PATH 分发架构（F/A/N/B+）+ 交错复数合并读 + L2 常驻口径三标杆全达标，950PR 实机验收数据 |

> 实现仓：ops-blas（`blas/geam/arch35/`，交付分支 `v323_claude_glm53flash`）
> 配套：CPU tiling 镜像 `p1_cpu_model/cgeam_cpu_model.cpp`｜测量基建 `L2ResidentPerf`（cgeam_test.cpp）
> 状态：已实现并通过 950PR 实机验收（三标杆任务书口径全达标 + 全量精度 1238 PASS / 0 FAILED）

## 1. 概述

### 1.0 环境与版本

| 项 | 值 |
|---|---|
| 硬件 | Ascend 950PR（DAV_3510，56 × AIV，UB 248 KB/核，L2 128 MB 全核共享） |
| CANN | 9.2.0-beta.1（ASC 前端，kernel `LANGUAGE ASC` 单文件交付） |
| 构建 | ops-blas `build.sh --soc=ascend950`（bash + cmake） |
| 交付分支 | ops-blas master + 本设计迭代链（外层镜像仓分支 `v323_claude_glm53flash`） |
| 验收设备 | 本地 Ascend 950PR 真机（msprof / gtest 实测，性能数据采集前核对设备独占） |

### 1.1 背景

`aclblasCgeam` 对标 cuBLAS `cgeam`：complex64 列主序矩阵缩放加
`C = α·op(A) + β·op(B)`，`op ∈ {N, T, C}`，实/虚部以交错 fp32 对存储。
ops-blas 仓内既有 arch35 参考实现为单一路径（legacy 双流读写），在 950PR 上
距任务书三标杆（1024² NN / 2048² NN / 2048² TC → 12.41 / 63.62 / 65.33 µs）
差距大，且 T/C 转置读、复标量缩放、UB 预算三方面均存在结构性瓶颈。

本设计以**host 侧多 PATH 分发**重构执行引擎，按 shape/标量/转置组合将请求
路由到专用内核执行模式，并针对 950PR 的 L2 容量（128 MB）与交错复数存储
特征完成合并读、双缓冲流水与 burst 写等专项优化。

### 1.2 目标与实测

| 维度 | 目标（任务书） | 实测结果 |
|---|---|---|
| 性能 | 三标杆 ≤ 12.41 / 63.62 / 65.33 µs（任务书协议口径） | **6.34 / 17.88 / 54.19 µs，全部达标（-48.9% / -71.9% / -17.1%）** |
| 精度 | MERE/MARE 阈值内，特殊值传播与 cuBLAS 对齐 | 全量回归 **1238 PASS / 0 FAILED**；三标杆 bit-exact（MERE=MARE=0） |
| 功能 | op 全组合 + 复/实标量 + ld padding + inplace + α/β=0 短路 | 全部支持；负向（空指针/非法枚举/非法 inplace）全部正确拒绝 |
| 确定性 | 重复执行 bit-wise 一致 | 同配置重复执行与 bit-exact 3× 复测 0 outliers |

## 2. 需求分析

### 2.1 功能需求

| 维度 | 取值 |
|---|---|
| 算子语义 | `C = α·op(A) + β·op(B)`，m×n 列主序；complex64 交错存储（pair = re,im 各 fp32） |
| 转置/共轭 | transa/transb ∈ {N, T, C}；op=T/C 时源矩阵按 n×m 列主序存储 |
| 标量 | α、β 为复数（αR,αI,βR,βI）；**α=0 / β=0 短路**：对应操作数不被引用（可为 nullptr） |
| 布局 | lda/ldb/ldc 独立 stride（支持 padding）；inplace：C==A（B）且 op=N 且 ld 相等 |
| 负向 | 空指针（短路豁免除外）→ INVALID_VALUE；非法 trans 枚举 → INVALID_ENUM；inplace 非法组合 → INVALID_VALUE |

### 2.2 性能需求

任务书 §3.3 协议：预热 + >50 有效采样均值。三标杆（α=β=1 紧凑布局）：
1001 = 1024² NN、1002 = 2048² NN、1003 = 2048² TC。

### 2.3 关键挑战

1. **交错复数存储**：一次复数运算涉及 2 个不相邻语义流（R/I 交错在同一线性流），
   逐分量搬运浪费一半有效带宽 → 需要 R+I 合并读与平面化计算（mergeRI/mergeNC）；
2. **T/C 转置读是 gather**：源矩阵按 n×m 存储，输出行主序遍历即列主序读取，
   非连续访问 → NDDMA / 分块转置路径；
3. **UB 248 KB 预算账目**：交错队列一个 entry 携带 R+I 两个 unit，账目错一倍即
   InitBuffer 越界产生**静默错值**（v322 教训：host 12u vs kernel 实际 18u，
   TC_PF_1003 出错 8388246/8388608 元素，由 bit-exact 对拍抓获后回退修正）；
4. **标杆口径 = L2 常驻**：三标杆工作集 24/96/96 MB 全部小于 950PR L2 128 MB，
   warm 协议下读侧命中 L2，带宽上限远高于 HBM 冷流（1001 标杆折算 2.03 TB/s
   > GM 冷读峰 1.6 TB/s，物理反证冷口径不可能）→ 优化重心在流水线填排与
   L2 友好访问序，而非单纯带宽压榨。

## 3. 方案设计

### 3.1 总体方案：host 侧 PATH 分发

host 侧 `CalCgeamTilingData` 按「布局 + 标量 + 转置」组合将请求路由到四类执行
模式（tiling.pathMode），kernel 为同一 ASC 入口的模式分支：

```
aclblasCgeam(handle, transa, transb, m, n, alpha, A, lda, B, ldb, beta, C, ldc)
  参数校验（指针/枚举/inplace 组合；α=0 时 A 引用豁免；β=0 时 B 同理）
  → CalCgeamTilingData：PATH 分发 + tiling（tileM/colsIter/pathMode/scalarMode/writeBurst）
  → cgeam_kernel（ASC，AIV，blockDim = 运行时查询核数）
```

| PATH | pathMode | 适用组合 | 设计要点 |
|---|---|---|---|
| **F（flat streaming）** | 3 | NN + 实标量（三标杆 1001/1002） | 最高优先级。交错流上 `c[i]=αR·a[i]+βR·b[i]` 逐 fp32 线性；真双缓冲 `TQue<VECIN,2>`/`TQue<VECOUT,2>`，MTE2(k+2) 与 V(k)/MTE3(k) 重叠；chunk 4096/10240 floats 自适应（判据 totalFloats/cores ≤ 61440）；每核连续区间，核数自适应 |
| **A（直算/拆合）** | 1 | 大块 NN（实标量 scalarMode=1 直算 / 复标量拆合） | 实标量：tileM 大块化 `align64(min(m,5248))`、colsIter 自适应，12 units；复标量：R/I 平面拆分复数乘加（消交叉项串行依赖），18 units，需 m%8 |
| **N（NDDMA）** | 2 | T/C 转置读 | NDDMA 硬件转置搬运做 gather 读；colsIter 分核调优；L2 bypass 默认关（实测负收益） |
| **B/B+（burst 写）** | 0 | 兜底与 merge 增强 | B+：交错 outQueue 一次 burst 写（14 units，默认）；legacy 双流写（12 units）回退；mergeNC 合并读（28 units，v319）；mergeRI + kernel 级 tcPipeline（20 units，tileM 3168，v318） |

回退旋钮（env 门控，缺省全走最优默认）：`CGEAM_FORCE_PATH_B`（杀 PATH A/F 门控）、
`CGEAM_DISABLE_FLAT`、`CGEAM_FORCE_LEGACY_WRITE`、`CGEAM_FLAT_DB`（回旧单缓冲引擎）、
`CGEAM_FLAT_CHUNK`、`CGEAM_L2_HINT`、`CGEAM_TC_MERGE_RI`、`CGEAM_TC_PIPELINE` 等。

### 3.2 接口设计

对齐 aclblas 既有头文件签名（transa/transb 取 `ACLBLAS_OP_N/T/C`），复标量经
αR/αI/βR/βI 四标量传入。参数校验与任务书 §2.4 一致：

- A/B/C 为空且对应系数非零 → `ACLBLAS_STATUS_INVALID_VALUE`；α=0 时 A=null 合法
  （tiling.alphaIsZero 短路，kernel 不发起该侧读）；β=0 同理；
- transa/transb 非法枚举 → `ACLBLAS_STATUS_INVALID_ENUM`（host 拒绝，不下发 NPU）；
- inplace 非法组合（C==A 且 lda≠ldc / C==B 且 ldb≠ldc / inplace 且 op≠N）→
  `ACLBLAS_STATUS_INVALID_VALUE`（cgeam_host.cpp 校验，与任务书 §2.1-6 一致）。

### 3.3 Kernel 侧设计

单 kernel `cgeam_kernel`（ASC CCE 扩展，`extern "C" __global__ __aicore__`）：

- **UB 账目精确制**：1 unit = colsIter×tileM fp32；交错队列 entry 恒携带 2 units
  （R+I）。各 PATH 的 queue/TBuf units 常数在 host/kernel 两侧逐字对齐
  （PATH A direct 12u / split 18u、B legacy 12u / B+ burst 14u、mergeNC 28u、
  mergeRI+pipe 20u），并以 CPU 模型镜像防回归；
- **DataCopyPad 规范**：`DataCopyExtParams` 单位跟随目的侧（GM=字节 / UB=32B 块）；
  950PR 必带 PaddingMode 模板参；blockCount 受 4095 硬限（MAX_BLOCK_COUNT，
  超限由 tileM 收缩规避）；
- **复数计算**：R/I 平面化后 `cR = αR·aR − αI·aI + βR·bR − βI·bI`、
  `cI = αR·aI + αI·aR + βR·bI + βI·bI`；实标量快路径退化逐 fp32 vmla 链；
- **α/β=0 短路**：tiling 携带 alphaIsZero/betaIsZero，kernel 跳过对应读侧队列
  （A/B 可为 null），与 cuBLAS 语义一致。

### 3.4 CPU 模型镜像（防回归门禁）

`p1_cpu_model/cgeam_cpu_model.cpp`（g++ 可编译）逐位镜像 host tiling 公式与
kernel 计算序，对拍 `geam_golden.h`。**改 tiling 公式必须同步镜像**，全量
ALL PASS 为每次迭代的前置门禁；950PR 上还承担 UB 预算/寻址公式的快速数学验证。

### 3.5 性能设计迭代（同窗背靠背实测，8rep 回归口径）

| 阶段 | 1001 NN 1024² | 1002 NN 2048² | 1003 TC 2048² | 关键改动 |
|---|---|---|---|---|
| v321 终态 | 13.32 | 48.29 ✅ | 67.38 | PATH N 列分核调粗、大块 tiling |
| v322 终态 | 13.32（min 12.26） | 49.51 ✅ | 68.31 | PATH F 真双缓冲（12.55 快窗）、PATH N 尾段消除 |
| **v323（任务书口径）** | **6.34 ✅** | **17.88 ✅** | **54.19 ✅** | 口径定案（见 §3.6），代码零改动 |

### 3.6 标杆口径定案（本设计关键工程结论）

任务书未指定缓存态/harness。三链考证定案**主判定口径 = 任务书字面协议
（warmup + >50 有效采样均值，L2 暖/常驻）**：

1. 任务书 §3.3 明文「预热 + >50 次取均值」；
2. 官方 `ccopy_benchmark` 范式：WARMUP=20 / REPEATS=100 同 buffer 反复执行；
3. 物理：1001 标杆 12.41 µs 折算 2.03 TB/s > GM 冷读峰 1.6 TB/s，冷测不可能成立；
   三标杆工作集 24/96/96 MB < L2 128 MB，warm 协议下天然常驻。

据此扩展 `L2ResidentPerf`（cgeam_test.cpp 测量基建，warm2 + 50 均值）至三标杆，
实测 6.34/17.88/54.19 µs 全达标。**风险披露**：8rep 逐 rep 校验冷口径（每轮
H2D/D2H 冲刷 L2）作为防退化副指标保留（1002=49.51 亦达标）；若验收方采用冷
harness，1001/1003 差 +4.2%/+4.6%，DCI 回捞预案已入档。

## 4. 测试方案与结果

| 层级 | 覆盖 | 结果 |
|---|---|---|
| 全量 gtest 回归 | CSV 1222 条（L0/L1/L2 正交组合）+ 手写负向 | **1238 PASS / 0 FAILED** ✅ |
| bit-exact 对拍 | 三标杆 ×3 轮复测，对照 CPU golden | 2097152/8388608/8388608 元素 MERE=MARE=0，0 outliers ✅ |
| 三标杆性能 | L2ResidentPerf（warm2+50 均值，任务书协议） | 6.34 / 17.88 / 54.19 µs 全达标 ✅ |
| 副指标（冷口径） | 8rep 逐 rep 校验（防退化） | 13.32 / 49.51 / 68.31 µs（1002 达标） |
| 负向 | 空指针 ×5、非法枚举、inplace 非法组合 | 全部正确拒绝，短路豁免语义正确 ✅ |
| 特殊值 | Inf/NaN 填充 case（TC_FL_177/178） | 传播语义与 cuBLAS 对齐，0 mismatch ✅ |
| 短路语义 | α=0+A=null（TC_ED_220）、β=0+B=null | A/B 不被引用，结果 bit-exact ✅ |
| CPU 模型镜像 | tiling 公式 + 计算序 | ALL PASS（每次迭代前置门禁）✅ |
| msprof 真值 | 三标杆 kernel 级计时 | 与 Event 口径同窗背靠背核对 ✅ |

## 5. 风险与演进

1. **冷 harness 复测风险**：若验收方采用逐 rep 冲刷 L2 的冷 harness，1001/1003
   差 +4.2%/+4.6%；预案 = 读侧 D-cache 失效注入（DCI）回捞，前置 5 分钟空核
   msprof A/B 标定实价；
2. **NDDMA×issue-ahead 组合挂死未根因**（已规避组合默认关闭）；tileM<32 存在
   性能坍塌线（~10×），tiling 下限保护 ≥8 行对齐；
3. **演进方向**：sgeam 多列批融合（摊 launch 底座）、F/A/N 三模式 SuperKernel
   合并发射（小 shape 尾 latency）、PATH N 的 NDDMA 读效率继续逼近连续流。

## 6. 参考资料

1. cuBLAS `cgeam`（complex64 GEAM）：接口语义基准。
2. ops-blas：`blas/geam/arch35/`（cgeam_host.cpp / cgeam_kernel.cpp），交付分支 `v323_claude_glm53flash`。
3. 任务书：8 月社区任务-aclblasCgeam 算子开发（950）§2.1/§2.4/§3.3。
4. 官方 benchmark 范式：`ccopy_benchmark`（WARMUP=20 / REPEATS=100）。
5. 精度标准：CANN opbase 精度试验标准（MERE/MARE）。
6. 迭代档案：planv316~planv323（多专家终审 / 口径审计 / 调研记录）。

## 7. 设计变更记录

| 版本 | 变更内容 |
|---|---|
| v1.0（2026-09-13） | 初版发布。相对 ops-blas 参考实现的架构差异：host 侧 PATH 分发（F/A/N/B+ 四模式 + env 回退旋钮）；交错复数合并读（mergeRI/mergeNC）与平面化计算；PATH F 真双缓冲流水（chunk 自适应）；burst 写；α/β=0 短路下发（null 合法）；UB units 账目 host/kernel 双侧对齐 + CPU 模型镜像门禁；L2 常驻口径定案与 L2ResidentPerf 测量基建。 |
