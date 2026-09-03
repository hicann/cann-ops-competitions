# aclsparseSpSM A5/950 算子开发计划书

> 文档版本：V0.1  
> 作者：Herbert DaShuAi  
> 编制日期：2026-09-02  
> 文档状态：可执行草案；进入功能开发前必须先通过 G0“需求与验收合同冻结”  
> 计划对象：Ascend 950（A5，arch35 / DAV_3510）上的 aclsparseSpSM 能力补齐  
> 推荐排期：单人全职 8 周、约 40 个有效工作日；不含外部设计评审、算力排队和 PR 审核等待时间

---

## 0. 执行摘要

本任务不应按“从零开发一个 SpSM 算子”规划。2026-09-02 实时核查的官方 ops-sparse master 已经存在一版 arch35 SpSM，实现了 CSR、FP32、opA=N/T、opB=N、Host alpha、ROW/COL 和原地求解等基础能力；但它与本任务书要求之间仍有较大差距，尤其是：

1. 现有 Analysis 包含 Device→Host 拷贝、CPU CSR→CSC、CPU level scheduling、Host→Device 回写，无法满足本任务“格式相关计算由 NPU 完成”和“不引入 CPU 同规模缓存”的要求。
2. 缺少 CSC/COO、complex64、opA/opB 的完整 N/T/H、Device pointer mode。
3. 缺少公开 aclsparseSpSMUpdateMatrix 接口及 GENERAL/DIAGONAL 更新能力。
4. 现有跨阶段状态校验不覆盖任务书要求的全部描述符、参数、pointer mode、workspace 和异步生命周期合同。
5. 附带测试包本身存在缺失文件、覆盖缺口和互相矛盾的性能/内存口径，不能直接作为最终验收器。

因此，推荐开发主线是：

~~~text
冻结需求和基线
    ↓
设计评审与测试合同先行
    ↓
NPU Analysis + CSR/FP32 垂直切片
    ↓
三格式、complex64、N/T/H、布局与原地能力
    ↓
UpdateMatrix 与完整状态机
    ↓
精度/异常/确定性/异步测试闭环
    ↓
性能和内存调优
    ↓
A2/A3/950 回归、文档和 PR 准备
~~~

本计划设置九个阶段、九个退出门禁。任何阶段只有在“产物、测试、证据”三者齐备后才算完成。尤其不能把编译成功、某个脚本显示 PASS、PR 已创建或 Kernel 能启动等同于任务验收完成。

---

## 1. 需求基线

### 1.1 数学、平台与工程合同

- 数学语义：op(A) · C = alpha · op(B)。
- 目标平台：Ascend 950（A5），arch35 / DAV_3510。
- 主验收版本：CANN 9.1.0 及任务方认可的配套版本；实际 Toolkit、驱动、固件、编译器和 SoC 型号必须在 G0 冻结。
- 公开形态：aclsparse C/C++ Legacy API、Host 多阶段状态管理、Ascend C Kernel、C++ UT/ST、性能和内存脚本、README、设计文档与自测报告。
- 执行要求：Solve 沿 handle stream 异步提交；不得以 CPU fallback 代替 NPU 主求解或格式相关计算。

### 1.2 能力矩阵

| 维度 | 必须支持 |
|---|---|
| A 稀疏格式 | CSR、CSC、COO |
| A values / computeType | ACL_FLOAT、ACL_COMPLEX64 |
| 索引 | Device I32，base 0/1 |
| A 属性 | LOWER/UPPER、UNIT/NON_UNIT、二维三角方阵 |
| opA | N、T、H；complex64 的 H 必须执行共轭转置 |
| opB | N、T、H |
| B/C 布局 | ROW、COL，合法 leading dimension/padding |
| RHS | 动态多 RHS，覆盖 0/1/多列边界 |
| alpha | Host pointer mode、Device pointer mode |
| 原地 | B.values 与 C.values 为同一 Device 指针 |
| 算法枚举 | DEFAULT |
| 非排序输入 | 未排序 CSR/CSC/COO 进入确定性规范化 |
| 更新 | GENERAL、DIAGONAL UpdateMatrix |
| 确定性 | 同一合同下重复执行结果 bitwise deterministic |

### 1.3 公开 API

现有接口：

- aclsparseSpSMCreateDescr
- aclsparseSpSMDestroyDescr
- aclsparseSpSMBufferSize
- aclsparseSpSMAnalysis
- aclsparseSpSM

需要新增：

- aclsparseSpSMUpdate_t
  - ACL_SPARSE_SPSM_UPDATE_GENERAL
  - ACL_SPARSE_SPSM_UPDATE_DIAGONAL
- aclsparseSpSMUpdateMatrix

需要同步核查的不仅是 include/cann_ops_sparse.h，还包括：

- 符号导出或版本脚本；
- 安装头；
- 文档 API 索引；
- mock、wrapper、测试声明；
- CMake 自动收集与架构派发；
- ABI 与枚举值兼容性。

### 1.4 多阶段合同

最小生命周期：

~~~text
Create
  → BufferSize
  → 分配 externalBuffer
  → Analysis
  → 可选 UpdateMatrix
  → SpSM
  → 可重复 UpdateMatrix / SpSM
  → 等待 stream 中异步工作完成
  → Destroy / 释放 externalBuffer
~~~

正式要求包括：

- BufferSize/Analysis 允许 B/C values 为空，但描述符必须有效；
- Solve 必须使用有效 Device values；
- Analysis 到 Solve 之间，描述符身份、结构参数和 externalBuffer 必须满足冻结后的“一致性合同”；
- externalBuffer 在异步 Solve 完成前不得释放或修改；
- UpdateMatrix 后刷新与 values、对角、规范化映射和 tiling 相关的状态；
- 错误阶段调用返回明确状态；
- BufferSize 计算必须检查 size_t 溢出；
- workspace 的设备属性、大小和对齐规则必须明确。

### 1.5 精度门槛

CPU Golden：

- FP32 输入使用 float64 计算；
- complex64 输入使用 complex128 计算。

有限数值的主判据：

~~~text
abs(actual - golden) ≤ 2^-16 + 2^-10 × abs(golden)
~~~

并且：

- 整体匹配率不低于 0.99；
- 每个元素的绝对误差不超过 max(1e-2, 32 × ULP(golden))；
- complex64 的实部、虚部分别应用全部规则；
- 需要单独定义并测试 INF、NaN、signed zero 的通过语义；
- 必须重复执行并做字节级确定性比较。

### 1.6 性能门槛

正式公式：

~~~text
性能倍率 = GPU 设备 Event 对应调用范围耗时 / NPU 同调用范围总耗时
要求：性能倍率 ≥ 0.3
~~~

采样合同：

- 预热 10 次；
- 正式采样 30 次；
- 报告原始样本、median、P90；
- 分开报告 Analysis、Update、Solve、总调用范围；
- Profiler 采样和无 Profiler 性能采样分开进行；
- 记录 workspace 和环境指纹。

任务书表中的三个规模：

| 场景 | m / nnz / RHS | 任务书 GPU median 范围 | 按 0.3 倍推导的 NPU 上限范围 |
|---|---:|---:|---:|
| P-01 | 32,768 / 262,144 / 16 | 56,341.984–57,962.883 μs | 187,806.613–193,209.610 μs |
| P-02 | 65,536 / 786,432 / 32 | 152,160.188–152,243.891 μs | 507,200.627–507,479.637 μs |
| P-03 | 131,072 / 1,966,080 / 64 | 304,306.812–305,923.281 μs | 1,014,356.040–1,019,744.270 μs |

右列只是按公式推导的规划值。最终必须逐 case、逐 dtype 使用冻结后的对应 GPU baseline 计算，不能使用范围中对自己更有利的一端。

### 1.7 内存门槛

任务书给出二选一条件：

1. 输入输出总量超过 500 MB 时，NPU 相对 GPU 的额外内存不超过 GPU 使用内存总量的 50%；
2. 方案固有 workspace 绝对值不超过目标硬件 L2 Cache 容量。

由于当前 P 场景很可能低于 500 MB，L2 路径可能成为实际硬门槛。G0 必须确认：

- 500 MB 使用十进制还是二进制；
- alias 时输入输出存储如何计数；
- 50% 的分母和 extra peak 公式；
- 低于 500 MB 且存在 GPU 等价接口时究竟走哪条规则；
- 目标 Ascend 950 SKU 的 L2 容量和权威读取方式。

---

## 2. 基线现状与能力差距

### 2.1 上游基线

计划编制时核查的上游基线如下：

- 官方仓库：cann/ops-sparse；
- master 提交：e0015bb76090487165f43dbaf2aae37648887a4b；
- 当前 SpSM 源码：sparse/spsm/arch35；
- 当前测试实际路径含 test/spsm/spsm/arch35，和任务书写的 test/spsm/arch35 不完全一致；
- 开发启动时重新 fetch，以冻结后的 master 提交和维护者评审意见为准。

### 2.2 当前 master 能力差距矩阵

| 维度 | 当前 master 观察值 | 任务目标 | 工作量判断 |
|---|---|---|---|
| 稀疏格式 | CSR | CSR/CSC/COO | 大：需 Device canonicalization |
| dtype | FP32 | FP32/complex64 | 大：Host、workspace、Kernel、Golden 全链路模板化 |
| opA | N/T | N/T/H | 中到大：complex H 共轭 |
| opB | N | N/T/H | 大：shape、地址映射、原地安全 |
| alpha | Host | Host/Device | 中：Device 标量沿 stream 读取 |
| UpdateMatrix | 无公开接口 | GENERAL/DIAGONAL | 大：API、状态和 Device 更新链路 |
| B/C ROW/COL | 已有基础 | 完整组合 | 中：需与 opB、alias、ld 交叉验证 |
| in-place | 已有基础 | 完整合法组合 | 中到大：opB T/H 和异布局需 staging |
| 未排序 | 声称支持 CSR | 三格式确定性规范化 | 大：需固定排序/重复项语义 |
| Analysis | D2H→CPU 转换和分层→H2D | NPU 格式相关计算、无 CPU 同规模缓存 | 很大：核心重构 |
| 状态校验 | 缓存部分 shape/order/op/buffer | 精确关联全部合同 | 中到大 |
| 性能 | 尚无 950 实测基线 | 三场景两 dtype ≥0.3 | 高风险，必须早测 |

### 2.3 需优先替换的实现路径

当前 spsm_host.cpp 可观察到：

- opA=T 时进行 Device→Host 数据拷贝；
- CPU 侧 CSR→CSC；
- CPU 侧 level scheduling；
- 再把规范化结果和层级数据拷回 Device；
- Host 侧持有与 m/nnz 同规模的 vector。

本任务建议将职责边界改为：

- Host：只做标量参数校验、描述符快照、workspace 划分、Kernel 编排和小型 tiling 决策；
- NPU：做所有随 m/nnz 扩展的格式转换、base 归一化、排序/重复项处理、对角提取、依赖层级构建、values 更新和主求解；
- CPU：只在测试进程中生成高精度 Golden，不进入正式算子计算路径。

### 2.4 随附测试包的阻塞问题

| 编号 | 当前问题 | 影响 | 计划处理 |
|---|---|---|---|
| TP-01 | generate_cases.py 导入的 extra_performance_cases.py 和 extra_accuracy_cases.py 缺失 | JSON 无法由随包源码重建，provenance 断裂 | G0/G2 补齐或重写生成器，并检查确定性重建 |
| TP-02 | NPU runner 只定义 hook 协议，包内没有注册实现和构建文件 | NPU 脚本无法独立运行 | G2 建立 ops-sparse C++ 测试桥或直接接公共 API |
| TP-03 | operator_adapter.py 实际只生成 CSR | 不能验 CSC/COO | G2 扩展格式感知数据模型 |
| TP-04 | 用例没有 B/C layout、ld、alias、空 values、workspace 生命周期等合同字段 | case 数多但关键能力缺失 | G2 新建合同清单和 C++ 专项用例 |
| TP-05 | accuracy_sparse_ops.py 未显式实现任务书全部容差、ULP、复数分量和特殊值规则 | ATK PASS 不足以证明正式精度通过 | G2 写独立比较器单测 |
| TP-06 | performance_cases 有 206 条，但 P-01 仅 FP32，P-02/P-03 仅 complex64 | 每个 P 场景未覆盖两 dtype | 冻结后至少补为 3×2 dtype×2 base=12 条 |
| TP-07 | NPU runner 在 Event 开始前执行 Update，GPU native 报告的是 Update+Solve 总时间 | 性能倍率不是同调用范围 | G0 冻结计时范围，G2 修 runner |
| TP-08 | 任务书/central TSV 与任务书链接的 benchmark Markdown 相差约 4.8–6.2 倍 | 0.3 门槛无法唯一计算 | G0 要求任务方指定权威 baseline |
| TP-09 | 任务书内存门槛为 50%，README 和 compare_memory.py 实现为 5% | 相同数据会得到不同 PASS/FAIL | G0 冻结正式阈值；内部可保留更严告警 |
| TP-10 | manifest 记录的 TSV SHA256 与当前 LF 文件不一致，转成 CRLF 后才匹配 | Linux 重放可能误报基线损坏 | 改用 canonical LF hash 或重建 manifest |

### 2.5 两套性能数据的现状

任务书表格与 test_cases/baseline_results/gpu_full_results.tsv 一致；manifest 标记来源为 H100。

但任务书链接的 gpu_performance_result_benchmark.md 使用另一组更慢结果：

| 场景 | 任务书 / central TSV | 链接 Markdown |
|---|---:|---:|
| P-01 | 56.342–57.963 ms | 322.955–323.942 ms |
| P-02 | 152.160–152.244 ms | 722.870–724.188 ms |
| P-03 | 304.307–305.923 ms | 1,834.474–1,887.569 ms |

在任务方确认前：

- 设计和功能开发可以继续；
- 任何“性能已经达标”的结论必须停止；
- 不得自行选择更宽松的那套基线；
- 需要保留 GPU 型号、CUDA/cuSPARSE 版本、源 case hash、计时范围和原始 30 个样本。

---

## 3. 项目范围

### 3.1 实施范围

1. 公开 API 和枚举补齐。
2. Host 参数校验、状态机、workspace 计算与 Kernel 编排。
3. arch35 NPU Analysis/normalization Kernel。
4. arch35 FP32/complex64 Solve Kernel。
5. CSR/CSC/COO 与 opA/opB N/T/H。
6. ROW/COL、leading dimension、Host/Device alpha、in-place。
7. GENERAL/DIAGONAL UpdateMatrix。
8. C++ UT/ST、CPU Golden、ATK/端到端适配。
9. 精度、确定性、性能、内存、Profiler 和异步证据。
10. A2/A3 与 950 的约定回归。
11. README、API 说明、设计文档和自测报告。
12. 两个仓库的 PR 准备与评审闭环。

### 3.2 非目标

- 新增任务书未声明的数据类型或 64 位索引；
- 新增 DEFAULT 之外的私有算法枚举；
- 用 CPU 求解或 CPU 稀疏格式转换作为正式 fallback；
- 修改 cuSPARSE 或重新定义其公开 API；
- arch20、arch22 的全新 SpSM 功能实现；
- 为满足脚本而降低或改写任务书的验收门槛。

### 3.3 变更控制

出现以下情况必须更新本计划版本并重新评估排期：

- 任务方改变 UpdateMatrix 数据合同；
- 性能计时从 Solve 改为 Analysis+Update+Solve；
- 内存阈值从 50% 改为 5%，或强制所有 case 受 GPU 比较规则约束；
- 要求实现 arch22 SpSM 新能力，而不只是联合回归；
- 新增 dtype、算法枚举、64 位索引或部分重叠 alias；
- 上游 master 对 SpSM 目录、接口或核心实现发生大改；
- 设计评审否决 Device canonicalization 或自适应 Solve 方案。

---

## 4. 推荐技术方案

### 4.1 总体分层

~~~text
Public Legacy API
    │
    ├─ 参数/枚举/shape/dtype/descriptor 校验
    ├─ SpSM descriptor 状态机与合同快照
    ├─ workspace size/offset 计算与溢出检查
    └─ handle stream 上的 Kernel 编排
             │
             ├─ Analysis Pipeline
             │    ├─ 格式/base/opA 规范化
             │    ├─ 确定性排序与重复项处理
             │    ├─ 对角定位/奇异状态
             │    ├─ 依赖 level 或调度元数据
             │    └─ original→canonical values 映射
             │
             ├─ Update Pipeline
             │    ├─ GENERAL：按原格式 values 更新 canonical values
             │    └─ DIAGONAL：按逻辑行序更新 m 个对角值
             │
             └─ Solve Pipeline
                  ├─ opB/layout/alpha 输入适配
                  ├─ 自适应 chain/level 求解
                  ├─ FP32/complex64 固定顺序归约
                  └─ C layout 写回与 alias 保护
~~~

### 4.2 Host 职责

Host 仅保留 O(1) 或小型元数据工作：

- 检查 handle、descriptor、enum、shape、dtype、index type、base、fill/diag、ld；
- 判断 BufferSize/Analysis/Solve/Update/Destroy 的合法阶段；
- 快照 Analysis 合同；
- 计算对齐后的 workspace 分区和上溢；
- 选择 Kernel variant；
- 将 Kernel 依次提交到 handle stream；
- 记录 debug/profiler 所需的稳定 metadata。

Host 不应：

- 拷贝完整 row offsets、indices、values 到 CPU；
- 在 CPU 上做 CSR↔CSC/COO 转换；
- 在 CPU 上为 m 行构建完整 level 数组；
- 在 CPU 上缓存与 m/nnz/RHS 同规模的正式计算数据；
- 在 Solve 内隐式同步 stream。

### 4.3 Device canonical representation

建议 Analysis 将三种输入统一为 effective op(A) 的 canonical CSR：

- index base 统一为 0；
- 逻辑方向统一为实际求解方向；
- complex H 在规范化或读取时执行共轭；
- 每行按列索引稳定排序；
- 对重复坐标按冻结后的规则处理；
- 生成对角位置/对角值；
- 生成原始 values 顺序到 canonical values 的稳定映射；
- 保留足够元数据，使 UpdateMatrix 无需重做 pattern analysis。

快速路径：

- 已排序 CSR、base 0、opA=N、无重复且布局满足条件时，尽量复用原 index 结构；
- 仍要保证状态、对角和确定性证据；
- 是否直接引用输入 values 或复制到 workspace，需结合 UpdateMatrix 和生命周期合同决定。

通用路径：

- CSC/COO、T/H、base 1、未排序或重复坐标进入 Device conversion/sort/scan；
- 所有 sort key 和 tie-breaker 必须固定；
- 同一输入在相同环境下产生相同 canonical 顺序。

### 4.4 依赖分析与调度

三角系统的并行度取决于 pattern。P 场景的带状构造可能包含相邻行依赖，level 数可接近 m；此时“每 level 一次全核 barrier”会有很高开销。

建议至少提供两种 Solve 调度：

1. Chain / narrow-frontier 路径
   - 适合 level 数高、每层行数少；
   - 用少量核或单核顺序处理行；
   - 重点并行 RHS 和行内 nnz；
   - 避免每行跨全核 SyncAll。

2. Wide-level 路径
   - 适合同层行数多；
   - 行在核间分配；
   - 固定行内归约顺序；
   - 层间使用严格、可证明的同步。

Analysis 生成以下调度指标：

- level 数；
- 平均/最大 level width；
- max/avg row nnz；
- RHS；
- dtype；
- 是否需要 B staging；
- canonical values 是否连续。

Host/tiling 根据这些指标选 variant。第一版先保证正确和稳定，再用 P-01/P-02/P-03 Profiler 确认瓶颈。

### 4.5 Solve Kernel

Solve 需要覆盖：

- effective LOWER/UPPER 方向；
- UNIT 对角忽略显式对角值；
- NON_UNIT 对角加载、零/缺失语义；
- FP32 与 complex64；
- opB=N/T/H；
- B/C ROW/COL 和 ld padding；
- exact in-place；
- Host/Device alpha；
- RHS 1、非 2 次幂、16/32/64 等典型值；
- 长尾行、空行和边界行。

推荐实现原则：

- RHS 作为主要向量化维度；
- 行内 nnz 采用固定 chunk 顺序；
- complex64 的实部/虚部运算顺序固定；
- 避免以完整 maxRowLen 为 UB 一次性容量前提，采用流式/分块加载；
- 不要求把全部 levelRowPtr 放入 UB，支持分块或 GM 读取；
- 为 ROW/COL/opB 组合做编译期或少量运行时特化；
- 只在 alias 不安全组合使用 B staging，减少 workspace。

### 4.6 opB、布局和原地

建议把输入地址映射与 alias 策略显式化：

| 情况 | 建议路径 |
|---|---|
| B/C 不重叠，opB/layout 可直接寻址 | 直接读 B、写 C |
| exact alias 且读写顺序可证明安全 | 直接原地 |
| exact alias 且 opB=T/H 或布局改变导致覆盖风险 | 先把逻辑 op(B) 规范化到 workspace，再求解 |
| 部分重叠但起始指针不同 | 默认返回 INVALID_VALUE，除非任务方明确要求支持 |

必须单测：

- B/C 相同起始指针、相同 order；
- B/C 相同起始指针、不同 order；
- opB=T/H；
- ld 大于逻辑维度；
- padded 区域不被越界修改；
- alias staging 的 workspace 计入 BufferSize。

### 4.7 alpha pointer mode

- Host mode：在公开调用进入时读取 computeType 对应标量，按值传给 Kernel。
- Device mode：不得在 Host 解引用；Solve Kernel 在 handle stream 上读取 Device 标量。
- BufferSize/Analysis 不应为了计算 workspace 解引用 alpha。
- complex64 Device alpha 必须覆盖非零实部和虚部。
- alpha 指针身份/值是否允许在 Analysis 后变化，必须在 G0 冻结并写入合同快照规则。

### 4.8 UpdateMatrix

按 cuSPARSE 官方参考语义，推荐默认方案是：

- GENERAL：newValues 含与原 A 存储顺序一致的 nnz 个新 values，pattern 不变；
- DIAGONAL：newValues 只含逻辑对角顺序的 m 个新 values。

但这只是推荐默认，必须经任务方确认后才成为正式合同。

GENERAL 路径：

- 校验阶段和 pointer mode；
- 使用 Analysis 保存的 original→canonical 映射；
- Device gather/reorder/conjugate；
- 更新 canonical values 和对角缓存；
- pattern 与依赖拓扑不变，不重复做完整排序和 level analysis；
- 更新 generation、values pointer 和 tiling 状态。

DIAGONAL 路径：

- 读取 m 个逻辑对角值；
- 更新 canonical diagonal 或独立 diag buffer；
- UNIT 模式的行为由 G0 冻结；
- NON_UNIT 更新为 0、NaN、Inf 时按冻结语义处理；
- 不得错误地把 m 个对角值当成 nnz values。

### 4.9 状态机

建议用显式枚举替换单一 analyzed 布尔值：

| 状态 | 允许操作 | 结果 |
|---|---|---|
| CREATED | BufferSize、Destroy | 进入 SIZED 或释放 |
| SIZED | BufferSize、Analysis、Destroy | 可重复查询；Analysis 成功进入 ANALYZED |
| ANALYZED | Solve、Update、重新 Analysis、Destroy | 保持可复用，generation 递增 |
| ERROR/INVALIDATED | 重新 Analysis 或 Destroy | 禁止 Solve，返回明确状态 |
| DESTROYED | 不应再解引用 | C API use-after-free 的可检测性需明确限制 |

描述符至少保存：

- magic/version/state；
- analysis generation、values generation；
- handle/stream 关联策略；
- opA/opB、computeType、alg、pointer mode；
- A descriptor 身份、format、m/n/nnz、index types/base、fill/diag；
- B/C descriptor 身份、rows/cols/order/ld；
- 允许变化的 values pointer 状态；
- externalBuffer 指针和查询尺寸；
- canonical offsets；
- mapping、diag、level/scheduling metadata；
- Kernel variant 和 tiling；
- 上次异步提交的序列信息。

注意：销毁后的裸指针再次调用在普通 C API 中无法安全、完全检测。实现不得通过保留泄漏对象来模拟销毁后检测，应遵循仓库既有对象生命周期规范及评审结论。

### 4.10 workspace 规划

候选分区：

| 分区 | 规模 | 是否总需要 |
|---|---:|---|
| device status / counters | O(1) | 是 |
| canonical row offsets | O(m) | 通用路径 |
| canonical col indices | O(nnz) | 通用路径 |
| canonical values | O(nnz × sizeof(dtype)) | 通用或 Update 路径 |
| original→canonical map | O(nnz) | 非直接引用路径 |
| sort/scan scratch | 依算法 | Analysis 临时 |
| diagonal positions/values | O(m) | NON_UNIT / Update |
| levels / level rows | O(m) | level 路径 |
| level offsets | O(L) | level 路径 |
| B staging | O(m×RHS×sizeof(dtype)) | alias/opB/layout 条件性 |

BufferSize 必须：

- 对每次乘加做 size_t 上溢检查；
- 使用仓库统一对齐；
- 区分 permanent analysis state 与可复用 scratch；
- 对所有格式/op/dtype/layout/alias 返回足够尺寸；
- 不依赖 B/C values 非空；
- 尽量避免同时保留多个完整 nnz 副本；
- 在 G1 之前完成 P-03 complex64 的字节预算并与 L2 门槛比较。

### 4.11 确定性

bitwise deterministic 需要在设计上定义，而不是只重复跑几次：

- sort key 固定；
- 相同 key 的 tie-breaker 固定；
- 重复坐标的归并顺序固定；
- 行内归约树固定；
- level 中行到核的映射固定；
- atomic 只用于不影响最终浮点求和顺序的计数/地址计算；
- complex 实部、虚部计算顺序固定；
- 不使用未初始化 workspace；
- 测试固定 seed、环境和 case hash。

需要区分两种目标：

1. 同一输入字节序列重复执行 bitwise 相同；
2. 同一个数学矩阵采用不同未排序排列后也 bitwise 相同。

任务书“确定性规范化”倾向于同时要求第二种，但必须在 G0 明确。

### 4.12 异步与错误边界

Solve 必须：

- 只在 handle stream 上提交；
- 不在 Host 隐式 synchronize；
- 返回时允许 Kernel 尚未完成；
- 通过同 stream 顺序保证 Analysis/Update/Solve 依赖；
- 文档明确 externalBuffer、A/B/C values、Device alpha 和 newValues 的有效期。

Device indices 的越界、重复、缺失对角等内容错误如果由 NPU 才能发现，就与“公开 API 立即返回明确错误码”和“完全异步”存在张力。G0/G1 必须决定：

- Analysis 是否允许为了结构校验同步一次；
- 或者结构错误通过 device status 在后续同步点报告；
- 或者部分输入结构由调用者合同保证。

在该问题定案前，不得声称所有非法 Device 内容都能同步返回精确状态码。

---

## 5. G0：开发前必须冻结的十项决策

| ID | 必须确认的问题 | 推荐默认 | 未确认的影响 |
|---|---|---|---|
| D-01 | 固定哪个 ops-sparse master commit | 开工日 fetch 后记录 SHA；每周评估 rebase | 文件清单、ABI、测试入口可能漂移 |
| D-02 | DIAGONAL newValues 的长度与顺序 | m 个 values，按逻辑对角 0…m-1 | Update Kernel 和测试无法定型 |
| D-03 | GENERAL newValues 的合同 | nnz 个 values，按原 descriptor 存储顺序，pattern 不变 | mapping 与生命周期不明确 |
| D-04 | 零规模、零 workspace、空 values | m/nnz/RHS 合法组合做 quick return；size=0 时 buffer 可空 | 参数校验顺序会反复返工 |
| D-05 | 重复坐标 | 稳定排序后按固定原始次序归并求和 | 数学语义和确定性均不明确 |
| D-06 | INF/NaN/零对角 | 特殊值单独分类；零/缺失 NON_UNIT 返回冻结状态 | Golden 和错误码无法验收 |
| D-07 | 原地覆盖范围 | exact alias 支持；部分重叠拒绝；不安全组合 staging | workspace 和 opB 实现变化大 |
| D-08 | 性能 0.3 的计时范围和 baseline | 与 central TSV 对齐的 Update+Solve total；另报 Analysis/Solve | 当前 GPU/NPU runner 不能公平比较 |
| D-09 | 内存 50%/5%、500 MB 与 L2 规则 | 正式按任务方确认；内部同时显示 5% warning 和 50% gate | PASS/FAIL 会相反 |
| D-10 | A2/A3/arch22 回归范围 | 回归共享代码和既有能力，不新增 arch22 SpSM 功能 | 工作量可能从 8 周大幅扩大 |

额外必须记录：

- 主验收 CANN、驱动、固件、编译器、SoC；
- workspace 对齐字节数；
- Analysis 是否允许同步结构校验；
- alpha/newValues 指针和值在各阶段的生命周期；
- Update 后是否可直接 Solve；
- Unit diagonal 下 DIAGONAL update 的行为；
- exact error-code 映射；
- 设计 PR、个人仓验收地址、代码 PR 的实际顺序。

G0 退出标准：

- requirements_freeze.md 或等价评审记录已形成；
- 十项决策均有“确认值、确认人/来源、日期”；
- 上游 SHA 和环境指纹已记录；
- 未确认项不会影响下一阶段的接口或内存布局。

---

## 6. 八周开发进程

### 6.1 总排期

| 周次 | 阶段 | 有效工日 | 主要目标 | 退出门禁 |
|---|---|---:|---|---|
| 第 1 周前半 | P0 需求/验收冻结 | 3 | 上游、环境、接口、性能、内存、Update 合同冻结 | G0 |
| 第 1 周后半 | P1 干净仓与回归基线 | 2 | 当前 CSR FP32 编译、UT/ST、行为与 Profiler 基线 | G1a |
| 第 2 周前半 | P1 设计与字节预算 | 2 | 设计文档、状态机、workspace、算法 ADR | G1 |
| 第 2 周后半 | P2 测试合同与桥接 | 3 | 修复生成器、比较器、NPU bridge、最小 case | G2 |
| 第 3 周 | P3 NPU Analysis + 垂直切片 | 5 | CSR FP32 N/N Host out-of-place 全 NPU 闭环 | G3 |
| 第 4 周 | P4 三格式与 opA | 5 | CSR/CSC/COO、base、未排序、N/T/H canonicalization | G4a |
| 第 5 周 | P4 dtype/opB/layout | 5 | complex64、opB N/T/H、ROW/COL、ld、in-place | G4 |
| 第 6 周 | P5 Update 与状态机闭环 | 4 | GENERAL/DIAGONAL、Device alpha、生命周期和异常 | G5 |
| 第 7 周 | P6 完整自测闭环 | 5 | C++ UT/ST、Golden、特殊值、确定性、异步 | G6 |
| 第 8 周前半 | P7 性能与内存 | 4 | P case 调优、Profiler、workspace/L2、峰值内存 | G7 |
| 第 8 周后半 | P8 回归与交付准备 | 2 | A2/A3/950 回归、README、自测报告、PR 检查 | G8 |

合计约 40 个有效开发日。外部等待另计：

- 设计文档评审：预计 3–10 个自然日，可能多轮；
- 950 算力排队：未知；
- 代码 PR review/CI：预计 3–15 个自然日，可能多轮；
- A2/A3 回归资源：未知。

如果两名开发者并行，预计可压缩到 6–7 周，但以下仍在关键路径上，不能简单并行：

- G0 合同冻结；
- workspace/descriptor ABI；
- NPU Analysis 数据格式；
- 共享 NPU 环境上的构建和性能采样；
- 最终集成、回归和 PR review。

### 6.2 P0：需求、环境与验收冻结（3 日）

任务：

1. 在干净位置 fetch 官方 ops-sparse。
2. 记录 master SHA、remote、分支和工作树状态。
3. 读取当前 CONTRIBUTING、CMake、SpSM README、Host/Kernel/UT。
4. 记录 CANN Toolkit、compiler、driver、firmware、SoC、核数、L2。
5. 向任务方提交第 14 节的待确认问题。
6. 冻结 error-code 表。
7. 冻结性能 baseline、GPU 环境、计时范围和样本文件。
8. 冻结内存规则。
9. 确认测试包缺失模块的权威版本是否可补发。
10. 确认设计文档实际任务目录与团队目录。

产物：

- requirements_freeze.md；
- environment_fingerprint.txt/json；
- upstream_baseline.md；
- acceptance_contract.md；
- issue/评审链接清单。

门禁 G0：

- D-01 至 D-10 已闭环；
- 任何未闭环项都被证明不会改变接口、workspace 或硬门槛；
- 不满足时只允许做无争议的 baseline 和原型，不进入正式性能承诺。

### 6.3 P1：现有实现基线与设计评审（4 日）

任务：

1. 按当前仓库帮助核对构建 token，再运行目标构建。
2. 运行现有 SpSM C++ UT/ST，记录 parsed case 数和退出码。
3. 在 950 上跑最小 CSR FP32 正向 case。
4. 用 Profiler 确认现有 Kernel、Host D2H/H2D 和同步边界。
5. 建立当前行为表：支持、NOT_SUPPORTED、INVALID_VALUE 等。
6. 画出当前 descriptor、workspace、Host/Kernel 数据流。
7. 完成新方案 ADR：
   - ADR-001 canonical CSR；
   - ADR-002 Device normalization；
   - ADR-003 chain/wide-level 自适应；
   - ADR-004 Update mapping；
   - ADR-005 alias policy；
   - ADR-006 determinism；
   - ADR-007 async error model。
8. 按社区模板完成正式 design.md。

建议基线命令形态：

~~~bash
bash build.sh --ops=spsm --soc=ascend950
bash build.sh --ops=spsm --soc=ascend950 --run
~~~

以上只是当前 CONTRIBUTING 中的命令形态。实际 soc token、产物路径和测试可执行文件必须先查看当前 build.sh --help，不得盲目照抄。

门禁 G1：

- 当前 master 在目标环境可复现；
- 现有 case 数、通过数、失败数明确；
- 新方案的 Host/Device 边界、workspace 上界和 P-03 complex64 字节预算可评审；
- 设计文档评审至少没有阻塞性架构异议。

### 6.4 P2：测试合同、生成器与 NPU bridge（3 日）

任务：

1. 补齐或重写缺失的 extra case 生成逻辑。
2. 生成前后检查：
   - case 数；
   - ID 唯一；
   - seed；
   - schema；
   - canonical LF；
   - SHA256；
   - 重复生成字节相同。
3. 把任务书容差写成显式比较器。
4. 为比较器写自测：
   - 边界等于容差；
   - 99% 比例；
   - 32 ULP；
   - complex 实/虚分别失败；
   - Inf/NaN；
   - 0 case 必须失败。
5. 建立直接调用公共 aclsparse API 的 C++ ST。
6. 如继续保留 Python runner，建立真实 C++ NPU bridge，不允许默认 reference fallback。
7. 修复 GPU/NPU 计时范围不一致。
8. 建立 requirement ID 到 case ID 的 manifest。

最小冒烟 case：

- CSR；
- FP32；
- base 0；
- LOWER + NON_UNIT；
- opA=N、opB=N；
- B/C ROW；
- Host alpha；
- out-of-place；
- m=4、nnz 合法、RHS=2。

门禁 G2：

- 用例可以从源码重建；
- runner 缺 hook 时明确失败；
- 预期 case 数不符时明确失败；
- comparator 单测通过；
- 最小 ST 能调用真实公共 API。

### 6.5 P3：NPU Analysis 与 FP32 垂直切片（5 日）

优先做一条端到端正确链路，而不是同时铺开所有枚举。

实现顺序：

1. descriptor 新状态骨架；
2. workspace offset/overflow/alignment；
3. CSR base 0 canonical 快速路径；
4. Device 对角提取；
5. Device level/scheduling metadata；
6. chain Solve；
7. wide-level Solve；
8. FP32 N/N ROW out-of-place；
9. 异步 stream；
10. CPU Golden 比对。

必须同时删除或旁路正式路径中的：

- D2H 完整 index/value；
- CPU Csr2Csc；
- CPU O(m+nnz) level scheduling；
- H2D 完整 analysis 结果。

门禁 G3：

- 最小 case 从 BufferSize 到 Analysis 到 Solve 全通；
- Profiler 证明格式分析和求解由 NPU Kernel 完成；
- Host 无 O(m/nnz) 缓存；
- Solve 无隐式 stream synchronize；
- 现有 CSR FP32 N/T 基础能力没有回退。

### 6.6 P4：完整功能矩阵（10 日）

子阶段 P4a：格式/opA（5 日）

1. CSR base 1；
2. CSC；
3. COO；
4. 未排序；
5. 重复坐标；
6. opA=T；
7. opA=H；
8. LOWER/UPPER 翻转；
9. UNIT/NON_UNIT；
10. canonical 三格式一致性。

子阶段 P4b：dtype/opB/layout/alias（5 日）

1. complex64 数据结构和 Kernel；
2. complex H 共轭；
3. opB=T/H；
4. B/C ROW/COL；
5. leading dimension/padding；
6. Device alpha；
7. exact in-place；
8. unsafe alias staging；
9. RHS=1、非 2 次幂和 64；
10. 特化派发与 fallback 边界。

门禁 G4：

- 能力矩阵每个声明值至少有一个正向 case；
- 高风险交互有专项 case；
- CSR/CSC/COO 对同一数学矩阵输出一致；
- complex H 有能区分 T 与 H 的非实数测试；
- padding 区域守卫值未被破坏；
- exact alias 与 out-of-place 结果一致。

### 6.7 P5：UpdateMatrix 与状态机闭环（4 日）

任务：

1. 公共枚举和 API。
2. 导出符号、文档索引、mock/wrapper。
3. GENERAL Device mapping/update。
4. DIAGONAL Device update。
5. 对角/奇异状态刷新。
6. values generation 和 analysis generation。
7. 同 stream Update→Solve 顺序。
8. 重新 Analysis 替换旧状态。
9. descriptor/参数/workspace 不一致校验。
10. Destroy 与 pending work 的文档合同。

必测序列：

~~~text
Create → Update                         预期阶段错误
Create → BufferSize → Solve            预期未初始化
Create → Analysis → Solve              成功
Create → Analysis → GENERAL → Solve    成功
Create → Analysis → DIAGONAL → Solve   成功
Create → Analysis → Solve → Update → Solve
Create → Analysis → 修改 shape/layout → Solve
Create → Analysis → 更换 buffer/stream → Solve
Create → Analysis → 重 Analysis → Solve
Destroy 前同步 / 未同步的合同
~~~

门禁 G5：

- Update 后无需重做 pattern analysis，除非冻结合同另有要求；
- GENERAL/DIAGONAL 数值和状态均正确；
- 所有阶段错误有确定状态码；
- Device alpha 和 newValues 沿同 stream 有正确依赖。

### 6.8 P6：完整自测闭环（5 日）

测试分层：

1. Host UT
   - 参数；
   - overflow；
   - workspace offsets；
   - state transition；
   - contract fingerprint。

2. Kernel 白盒/小型 ST
   - normalization；
   - sort/scan；
   - diagonal；
   - level metadata；
   - chain/wide-level；
   - layout/addressing；
   - complex arithmetic。

3. 公共 API 端到端 ST
   - 全能力矩阵；
   - alias；
   - update；
   - async；
   - error code。

4. CPU Golden 精度
   - float64/complex128；
   - 明确 comparator；
   - 特殊值专项。

5. 确定性
   - 同输入重复至少 20 次；
   - 不同合法输入排列；
   - Update 后重复；
   - 不同 layout 路径。

6. 回归
   - 原 master SpSM case；
   - 共享 descriptor/common code；
   - 相邻 sparse operators。

门禁 G6：

- C++ UT/ST 全部通过；
- 实际解析 case 数等于预期；
- 失败、超时、崩溃、0 case 都不能汇总为 PASS；
- 有限值、特殊值、determinism 分别报告；
- 未验证场景明确列出。

### 6.9 P7：性能与内存（4 日）

第一轮在 P3 垂直切片后就应做早期采样；第 8 周是收口，不是第一次测性能。

性能步骤：

1. 冻结代码 commit 和 case hash。
2. 固定功耗、频率、设备、后台负载和 stream。
3. 每个 case 预热 10 次。
4. 正式采样 30 次并保留原数组。
5. GPU/NPU 使用完全相同调用范围。
6. 分开记录：
   - Analysis；
   - Update/reanalysis；
   - Solve；
   - Update+Solve total；
   - Host API wall time；
   - Kernel time。
7. 用 median 算正式比值，P90 用于稳定性。
8. Profiler 单独运行，避免工具开销污染正式样本。

至少正式覆盖：

~~~text
P-01 × FP32/complex64 × base 0/1
P-02 × FP32/complex64 × base 0/1
P-03 × FP32/complex64 × base 0/1
~~~

即至少 12 个 P case；如任务方确认只要求 6 个，再在冻结合同中记录。

优化顺序：

1. 消除 Host 同步和多余 memcpy；
2. chain/wide-level 选择；
3. RHS 向量化；
4. row length 分桶；
5. layout/opB 特化；
6. complex64 加载和算术；
7. Update mapping；
8. workspace 复用；
9. Kernel launch 数；
10. 尾块与非对齐。

内存步骤：

- 同一 performance_cases.json；
- 同一 case hash；
- 分别采 GPU/NPU input baseline、peak、extra peak；
- 单独记录 BufferSize 查询值和实际分配；
- 记录 exact alias 和 out-of-place；
- 按冻结后的 50%/5%/L2 规则输出；
- 报告 reserved 与 allocated，但只用正式指定字段判定。

门禁 G7：

- 每个正式 case 有原始样本和环境；
- 所有声明 dtype 满足冻结后的 0.3；
- workspace 满足冻结后的内存门槛；
- Profiler 无 CPU fallback，Kernel 名和调用范围清楚；
- 未达标 case 有 root cause、实验对比和后续措施。

### 6.10 P8：联合回归与交付准备（2 日）

任务：

1. ascend950 完整构建与全量测试。
2. A2/A3/arch22 冻结范围回归。
3. README：
   - 环境；
   - 编译；
   - 测试；
   - 调用；
   - 限制；
   - 生命周期；
   - 错误码。
4. API 文档和设计文档同步。
5. 自测报告填充：
   - case 参数；
   - 输出；
   - 精度；
   - 性能；
   - 内存；
   - 截图；
   - Profiler；
   - 失败项。
6. 检查源码目录、测试目录、公开头文件和符号。
7. 运行格式、静态检查、git diff --check、CI。
8. 按提交时最新模板如实填写 AI 参与、CLA 和 PR 信息。

门禁 G8：

- Definition of Done 全部满足；
- 设计 PR 和代码 PR 的状态分别报告；
- 设计文档、源码、测试与验收材料按社区流程完成提交和评审闭环。

---

## 7. 测试与验收设计

### 7.1 覆盖策略

不做不可控的全笛卡尔积，采用三层覆盖：

1. 每个属性取值至少出现；
2. pairwise 覆盖普通交互；
3. 对高风险交互做强制专项组合。

高风险强制组合：

- complex64 + opA=H；
- complex64 + opB=H；
- CSC/COO + base 1 + unsorted；
- duplicate + deterministic canonicalization；
- Device alpha + Update + same stream；
- DIAGONAL + NON_UNIT；
- B/C exact alias + opB=T/H；
- B/C 不同 layout + ld padding；
- Analysis 空 B/C values → Solve 有效 values；
- descriptor 变更/stream 变更/buffer 过早释放；
- m/nnz/RHS=0/1；
- long tail + wide RHS；
- P-03 complex64。

### 7.2 精度比较器伪流程

~~~text
assert parsed_case_count == expected_case_count
assert process_exit_code == 0
assert output_shape == golden_shape

for each finite real component:
    primary_ok = abs_error <= atol + rtol * abs(golden)
    hard_cap_ok = abs_error <= max(1e-2, 32 * ulp(golden))

matched_ratio = primary_ok_count / finite_component_count
assert matched_ratio >= 0.99
assert every component satisfies hard_cap_ok

for Inf/NaN/signed-zero:
    apply the separately frozen classification rule

for complex64:
    run the complete rule independently on real and imaginary components
~~~

报告必须同时给出：

- 总元素/分量数；
- primary match 数和比例；
- hard-cap 最大误差；
- 最差元素索引；
- FP32/complex64 分开结果；
- 特殊值计数；
- 是否 bitwise deterministic。

### 7.3 异步测试

至少验证：

- API 返回后 stream 中仍可观察到工作；
- 在同 stream 排队的事件能界定完成；
- 不同 stream 不会错误复用未完成的 descriptor/workspace；
- Update→Solve 同 stream 顺序正确；
- Solve 不偷偷调用全局同步；
- externalBuffer 在完成事件前保持有效；
- 错误生命周期 case 放在隔离进程，避免一个非法释放破坏整个测试进程。

“提前释放 buffer”不能安全地靠主测试进程随意 free 后继续读写；应使用子进程/death test、运行时诊断或明确的合同测试。

### 7.4 确定性测试

每个确定性 case 保存：

- 输入结构和值的 hash；
- canonical index/value 的 debug hash；
- 输出字节 hash；
- 运行次数；
- CANN/驱动/SoC；
- Kernel variant；
- seed。

判定：

- 同一输入重复 20 次，所有输出 hash 相同；
- 如任务方确认 permutation-invariant，则对同一数学矩阵的多种未排序排列也要求 hash 相同；
- FP32 和 complex64 分别测试；
- GENERAL/DIAGONAL Update 后分别测试。

### 7.5 性能证据格式

每个结果行至少包含：

| 字段 | 说明 |
|---|---|
| commit_sha | 被测源码 |
| case_file_sha256 | 用例合同 |
| case_id | 唯一 case |
| dtype/format/base/op/layout/update | 能力标签 |
| warmup/samples | 10/30 |
| samples_us | 原始样本 |
| median_us/p90_us | 汇总 |
| analysis_us/update_us/solve_us/total_us | 同范围拆分 |
| gpu_baseline_us | 对应 GPU 值 |
| performance_ratio | GPU/NPU |
| workspace_bytes | BufferSize |
| device/toolkit/driver/firmware | 环境 |
| profiler_artifact | 证据路径 |
| pass | 依据冻结合同 |

### 7.6 追踪矩阵

| Requirement ID | 要求 | 主要实现 | 主要测试 | 证据 |
|---|---|---|---|---|
| R-API | 新枚举和 Update API | header/host/export | ABI/compile UT | symbol/API report |
| R-FMT | CSR/CSC/COO | Analysis Kernel | 三格式同矩阵 ST | canonical/output hash |
| R-DTYPE | FP32/complex64 | templated Host/Kernel | Golden | accuracy report |
| R-OP | opA/opB N/T/H | canonicalization/addressing | 非实数 H 专项 | output diff |
| R-LAYOUT | ROW/COL/ld | Solve load/store | guard/padding ST | guard report |
| R-PTR | Host/Device alpha | Host dispatch/Kernel load | pointer mode ST | stream trace |
| R-ALIAS | B/C exact in-place | direct/staging | alias matrix | memory/output |
| R-STATE | 多阶段一致性 | descriptor state | transition UT/ST | status log |
| R-UPD | GENERAL/DIAGONAL | Update Kernel | repeated update ST | generation/output |
| R-DET | bitwise deterministic | stable sort/reduction | 20-run hash | determinism report |
| R-NPU | 无 CPU fallback | Device Analysis/Solve | profiler/static audit | profiler |
| R-PERF | ≥0.3 | adaptive tiling | 12 P case | raw samples |
| R-MEM | 50%/L2 或冻结值 | compact workspace | memory scripts | memory report |
| R-REG | A2/A3/950 | shared/arch split | CI/device | regression matrix |
| R-DOC | 可复现交付 | README/design/report | clean-room replay | replay log |

---

## 8. 分支、提交与集成策略

### 8.1 工作树

- 为设计文档和算子代码分别创建独立、干净的分支或 worktree；
- 任务资料与测试基线作为输入资料管理，不混入无关源码提交；
- 正式开发仓内不提交私密账号、凭证或超出仓库限制的大型原始数据；
- 每次 rebase 前先记录基线 SHA 和本地状态；
- 已进入评审的公共分支不改写历史、不 force push。

### 8.2 两个 PR 分开管理

PR-A：cann-ops-competitions 设计文档

- 只包含本任务计划内的 design.md 等设计文档；
- 使用当时最新任务目录和团队目录；
- 通过设计评审并合入后，才记为设计交付完成。

PR-B：ops-sparse 代码

- 包含公开 API、Host、Kernel、UT/ST、README 和复现说明；
- 目标 master；
- 以当前 CONTRIBUTING、PR 模板和 CI 为准；
- 设计 PR 已合入不等于代码 PR 已完成。

### 8.3 建议提交切分

1. test: establish SpSM acceptance scaffolding
2. refactor: add SpSM descriptor contract and workspace layout
3. feat: move SpSM analysis normalization to arch35 NPU
4. feat: add CSC/COO and opA N/T/H
5. feat: add complex64 and opB/layout support
6. feat: add SpSM updateMatrix API
7. test: complete accuracy/lifecycle/determinism coverage
8. perf: optimize SpSM chain and wide-level paths
9. docs: update SpSM API, design and self-test report

实际提交信息必须符合提交时仓库的 AI 合规和 commit 规范，不提前写死模型名或机器人格式。

---

## 9. 风险登记册

| ID | 风险 | 概率 | 影响 | 预防/应对 | 触发信号 |
|---|---|---|---|---|---|
| RK-01 | Update 合同未定义 | 高 | 高 | G0 书面确认，先不固化 ABI 外行为 | reviewer 对 m/nnz 语义有异议 |
| RK-02 | 两套 GPU baseline 冲突 | 高 | 高 | 指定权威原始数据、环境和计时范围 | 同 case 相差数倍 |
| RK-03 | 50% 与 5% 内存冲突 | 高 | 高 | 双阈值报告，正式 gate 只用确认值 | 脚本和任务书 PASS 相反 |
| RK-04 | P-03 workspace 超过 L2 | 中高 | 高 | G1 先做字节预算，减少完整副本，条件 staging | 预算接近/超过 L2 |
| RK-05 | Device level scheduling 复杂 | 高 | 高 | 先做确定的 NPU 路径，再优化；保留 chain variant | Analysis 不正确或 launch 过多 |
| RK-06 | 相邻行依赖导致多核 barrier 很慢 | 高 | 高 | chain/wide-level 自适应，RHS 向量化 | L≈m、level width≈1 |
| RK-07 | complex64 H 精度/性能不足 | 中高 | 高 | 非实数专项、固定顺序、早测 P-02/P-03 | 实虚误差或吞吐明显异常 |
| RK-08 | opB + alias + layout 覆盖写 | 中 | 高 | exact alias 分类和 staging，guard memory | padding/未读 B 被改 |
| RK-09 | 异步错误与立即状态码冲突 | 高 | 中高 | ADR 明确 Analysis 同步边界和 device status | 需读 Device status 才知错误 |
| RK-10 | 测试包 case 多但合同缺失 | 高 | 高 | requirement→case manifest，覆盖优先于数量 | 200 case 仍无 CSC/alias |
| RK-11 | 上游 master 漂移 | 中高 | 中高 | 周期 fetch、早集成、小提交 | 目录/API/公共描述符变化 |
| RK-12 | 950/A2/A3 环境不可用 | 中高 | 高 | 第 1 周申请并保存资料；离线完成 Host/Golden | 队列、时长耗尽、设备中断 |
| RK-13 | 跨架构共享改动引发回归 | 中 | 高 | Host common 与 arch35 分层，早跑 CI | A2/A3 编译或现有算子失败 |
| RK-14 | PR 合规/CLA/AI 声明失败 | 中 | 中 | 提交前重读模板和 bot 原文，如实声明 | CI 通过但合规标签失败 |
| RK-15 | 性能首次测试太晚 | 中 | 高 | P3 后立即采早期 P case | 第 8 周才发现数量级差距 |

---

## 10. 质量门禁与完成定义

### 10.1 每个阶段的通用完成条件

- 代码/文档范围与该阶段一致；
- 有可复现命令；
- 记录命令退出码；
- 记录实际解析 case 数；
- 失败 case 没有被跳过或吞掉；
- 提交范围不包含无关文件或无关历史；
- 新增行为有测试；
- 原有行为有回归；
- 结论只基于当前 commit、当前制品和当前设备；
- 未执行的层级明确标为未验证。

### 10.2 最终 Definition of Done

功能：

- [ ] CSR/CSC/COO
- [ ] FP32/complex64
- [ ] opA N/T/H
- [ ] opB N/T/H
- [ ] LOWER/UPPER
- [ ] UNIT/NON_UNIT
- [ ] ROW/COL + ld
- [ ] base 0/1
- [ ] Host/Device alpha
- [ ] exact in-place
- [ ] GENERAL/DIAGONAL Update
- [ ] 未排序/重复项冻结语义
- [ ] bitwise deterministic

API/生命周期：

- [ ] 公开头和导出符号
- [ ] Create/BufferSize/Analysis/Solve/Update/Destroy
- [ ] B/C 空 values 的 BufferSize/Analysis
- [ ] Solve 有效 Device values
- [ ] contract fingerprint
- [ ] workspace size/alignment/lifetime
- [ ] 同 stream 异步
- [ ] 错误码表

验证：

- [ ] C++ Host UT
- [ ] C++ Kernel/ST
- [ ] CPU Golden
- [ ] comparator 自测
- [ ] 特殊值
- [ ] determinism 20-run
- [ ] P case 性能
- [ ] 内存/L2
- [ ] Profiler
- [ ] A2/A3/950 回归
- [ ] clean-room README 重放

交付：

- [ ] design.md 已评审合入
- [ ] ops-sparse 代码地址和分支
- [ ] README/API/限制说明
- [ ] 自测代码和用例
- [ ] 自测报告
- [ ] 环境与原始数据
- [ ] PR CI/CLA/合规/review 闭环

只有上述相关项满足且没有未豁免的 P0/P1 问题，才可宣布任务完成。

---

## 11. 证据与报告目录建议

在正式 ops-sparse 开发分支内，按仓库规范放源码和测试；验收证据可在独立、可忽略目录中组织：

~~~text
artifacts/spsm/
├─ environment/
│  ├─ npu_environment.json
│  ├─ gpu_environment.json
│  └─ upstream_sha.txt
├─ contracts/
│  ├─ requirements_freeze.md
│  ├─ error_codes.md
│  └─ case_manifest.json
├─ accuracy/
│  ├─ raw/
│  ├─ summary.json
│  └─ determinism.json
├─ performance/
│  ├─ gpu_raw.json
│  ├─ npu_raw.json
│  ├─ comparison.json
│  └─ samples.csv
├─ memory/
│  ├─ gpu_memory.json
│  ├─ npu_memory.json
│  └─ comparison.json
├─ profiler/
│  ├─ P-01/
│  ├─ P-02/
│  └─ P-03/
├─ regression/
│  ├─ ascend950.txt
│  ├─ a2.txt
│  └─ a3.txt
└─ self_test_report/
   ├─ report.md
   └─ screenshots/
~~~

是否将 artifacts 纳入 Git，按仓库大小和维护者要求决定。大型 Profiler 原始文件通常只保存到验收附件，不直接提交源码仓。

---

## 12. 日常开发节奏

每日开始：

1. 确认当前分支、SHA、工作树；
2. 确认设备和环境；
3. 选择一个 requirement ID；
4. 先写/更新最小失败测试；
5. 只修改该合同所需范围。

每日结束：

1. 运行 focused UT/ST；
2. 运行一次无变化重复；
3. 检查 parsed case 数；
4. 记录 build/test/profiler 结果；
5. 更新风险和未验证项；
6. 提交小而可审的 commit。

每周结束：

1. fetch 上游并评估 drift；
2. 跑一次较完整回归；
3. 更新 requirement traceability；
4. 更新性能趋势，不只保存最好值；
5. 清理无效实验，但保留关键对照数据；
6. 将阻塞问题提交给任务方/maintainer。

共享 950 工作区上的构建和性能采样必须串行，避免多个进程同时写 build/cache 或抢占设备导致不可信结果。

---

## 13. 前三个工作日操作清单

### 13.1 Day 1：冻结源码和环境

- [ ] 建立干净 ops-sparse 工作树
- [ ] fetch master
- [ ] 记录 SHA
- [ ] 读取 build.sh --help
- [ ] 记录 CANN/driver/firmware/SoC/L2
- [ ] 构建当前 spsm
- [ ] 保存完整构建输出和退出码

### 13.2 Day 2：复现当前行为

- [ ] 跑现有 SpSM UT/ST
- [ ] 核对实际 case 数
- [ ] 跑最小 CSR FP32
- [ ] 采一次当前 Profiler
- [ ] 记录 Host D2H/H2D/CPU 分析证据
- [ ] 建立现状能力和状态码表

### 13.3 Day 3：冻结合同

- [ ] 向任务方确认第 14 节问题
- [ ] 对齐 central TSV 与 Markdown baseline
- [ ] 对齐 50% 与 5%
- [ ] 对齐 UpdateMatrix m/nnz 语义
- [ ] 对齐零规模、重复项、特殊值
- [ ] 对齐 A2/A3 回归范围
- [ ] 输出 requirements_freeze.md
- [ ] 通过 G0 后再锁定详细设计

---

## 14. 待任务方确认的合同项

以下问题直接影响公开接口、workspace、验收门槛或总体排期，须在 G0 阶段形成书面结论，并记录确认来源与日期：

1. aclsparseSpSMUpdateMatrix 的 DIAGONAL 模式中，newValues 是否固定为 m 个、按逻辑对角行序排列；GENERAL 是否为原 matA 存储顺序的 nnz 个 values，且 pattern 不变。
2. UNIT 模式调用 DIAGONAL update 的预期行为：成功忽略、更新隐式对角或返回不支持。
3. m/nnz/RHS 为 0 时的返回语义，以及 bufferSize=0 时 buffer=nullptr 是否合法。
4. 重复坐标按求和、覆盖或报错处理；bitwise deterministic 是否要求同一数学矩阵的不同未排序排列得到相同 bit pattern。
5. NON_UNIT 的零/缺失对角所对应的 aclsparseStatus_t，以及 Inf/NaN 的传播、报错或分类比对规则。
6. B/C 原地是否仅要求相同起始 Device 指针；opB=T/H、B/C 不同 ROW/COL 和不同 ld 的 exact-alias 组合是否全部纳入支持；部分重叠是否直接判为非法。
7. Analysis 是否允许同步一次以返回 Device 结构错误，或所有 Analysis Kernel 均须完全异步。
8. 性能倍率 0.3 的正式计时范围：Solve、Update+Solve 或 Analysis+Update+Solve；Host 调度是否计入。
9. 任务书表/central TSV 与 gpu_performance_result_benchmark.md 的 P-01/P-02/P-03 数据相差约 4.8–6.2 倍，需指定正式 baseline、GPU/CUDA/cuSPARSE 环境和对应原始结果。
10. 每个 P 场景是否均要求 FP32 和 complex64；若是，正式矩阵为 3 场景×2 dtype×2 base=12 case。
11. 内存正式阈值采用任务书的 50% 或脚本的 5%；输入输出不足 500 MB 时，有 GPU 等价接口的 case 采用 GPU 比较或 L2 workspace 规则。
12. 目标 950 SKU 的 L2 Cache 容量所采用的权威接口或文档值。
13. A2/A3 与 arch22 的范围是既有共享逻辑回归，或包含本次新增 arch22 SpSM 功能。
14. 设计文档任务目录、团队目录，以及设计 PR、验收分支、ops-sparse 代码 PR 的先后顺序。

---

## 15. 参考资料

- aclsparseSpSM A5/950 社区任务书及随附测试资料
- 官方 ops-sparse：https://gitcode.com/cann/ops-sparse
- CANN 社区任务 2026：https://gitcode.com/cann/cann-ops-competitions/tree/master/04_tasks/01_community-task-2026
- 设计文档模板：https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md
- cuSPARSE SpSM 官方文档：https://docs.nvidia.com/cuda/cusparse/index.html
- 生态算子开源精度标准：https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md

---

## 16. 实施启动条件

开发从 P0 阶段启动，首先冻结上游源码提交、950 环境、UpdateMatrix 语义、性能基线、计时范围、内存门槛和回归范围，并形成 requirements_freeze.md。G0 通过后进入正式设计评审与源码开发；尚未闭环且会影响公开接口、workspace 或硬性验收门槛的问题，不得带入实现阶段。
