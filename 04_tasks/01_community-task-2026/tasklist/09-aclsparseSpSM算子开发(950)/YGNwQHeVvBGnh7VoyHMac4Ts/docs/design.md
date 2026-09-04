# aclsparseSpSM A5/950 (arch35) 算子设计文档

| 项目 | 内容 |
|------|------|
| 任务名称 | 9月社区任务-aclsparseSpSM算子开发(950) / A5 |
| 参与者账号 | `YGNwQHeVvBGnh7VoyHMac4Ts`（GitCode：`chanchp`） |
| 目标硬件 | Ascend 950PR/DT（arch35 / DAV_3510） |
| CANN 版本 | CANN 9.1.0+ |
| 交付代码仓 | 目标合入 [cann/ops-sparse](https://gitcode.com/cann/ops-sparse)；路径 `sparse/spsm/arch35/` |
| 设计提交路径 | `04_tasks/01_community-task-2026/tasklist/09-aclsparseSpSM算子开发(950)/YGNwQHeVvBGnh7VoyHMac4Ts/docs/design.md` |
| 对标接口 | cuSPARSE Generic `cusparseSpSM` |
| 与 A2/A3 关系 | 算法与 Host 状态机同源；arch22 为 A2/A3 差异层，本任务交付 arch35 |
| 文档版本 | v0.1（代码已自 A2/A3 同步；950 实测待填） |

---

## 需求背景（required）

### 需求来源

社区任务「9月社区任务-aclsparseSpSM算子开发(950)」，在 Ascend 950（A5）完善多右端稀疏三角求解。

### 背景介绍

#### 现状

`ops-sparse` 已有 arch35 SpSM 路径；本任务在 A2/A3（arch22）完整能力闭合基础上，将同一套 Host 状态机 + MemBase level-scheduling Kernel 对齐到 arch35，并按 A5 任务书验收（性能门槛 **≥0.3×**）。

| 参数 | 支持情况 |
| --- | --- |
| matA | CSR/CSC/COO；FP32/complex64；I32；LOWER/UPPER；UNIT/NON_UNIT |
| matB/matC | ROW/COL；多 RHS；可同指针原地 |
| opA/opB | N/T/H（H 仅 complex64） |
| alpha | Host + Device pointer mode |
| UpdateMatrix | GENERAL / DIAGONAL |

公式：$op(A)\cdot C=\alpha\cdot op(B)$

多阶段：`CreateDescr → BufferSize → Analysis → SpSM → UpdateMatrix → Destroy`。Solve 异步，禁止 `aclrtSynchronizeStream`。

---

## 需求分析（required）

1. 复用公开 `aclsparseSpSM*` + UpdateMatrix，补齐 CSC/COO、c64、op N/T/H、Device pointer、规范化。  
2. arch35 Ascend C Kernel + C++ UT/ST + 性能脚本。  
3. P-01/P-02/P-03 **≥0.3×** GPU Event 标杆；内存 workspace≤L2 或额外峰值规则。  
4. 与 arch22 联合回归。

---

## 详细设计（required）

### 总体方案

与 A2/A3 设计同构：

1. **Analysis（Host）**：格式归一（CSC/COO→CSR）、确定性排序/合并、level scheduling、diag 提取、workspace 绑定。  
2. **Solve（NPU）**：按 level 串行、`SyncAll` 栅栏；level 内多核；`levelRowPtr` **按 level 从 GM 取 2×int32**（固定 UB 槽，避免 L≈m 撑爆）。  
3. **COL / complex64**：vector transpose，`valFloatFactor=2`。  
4. **UpdateMatrix**：GENERAL 刷 values+依赖；DIAGONAL 刷对角。

### arch35 差异

- NPU_ARCH / tiling 走 DAV_3510；UB/核数经 `GetUbSize` / `GetAivCoreCount` 动态取。  
- 源码目录：`sparse/spsm/arch35/`、`test/spsm/spsm/arch35/`。  
- 编译：`bash build.sh --ops=spsm --soc=ascend950 --run`。

### 关键与验收口径

- 精度：同任务书 §3.2（rtol/atol/A/匹配率）。  
- 性能：倍率 = GPU Event / NPU，门槛 **0.3**。  
- 内存：path2 workspace≤950 L2，或 GPU 额外峰值规则。

---

## 支持硬件

| 芯片 | 勾选 |
| --- | --- |
| Ascend 950PR / 950DT | √ |

---

## 可维可测

- C++ GTest（CSV L0–L3 + Exception：UpdateMatrix / Device / CSC/COO / c64 / Unsorted）  
- 任务包 ATK / P-bench / memory compare  
- Profiler：Host/Kernel 时间线  

**950 实测数据**：待 NPU 测试机回填（见 `delivery/SELFTEST_REPORT_A5.md`）。
