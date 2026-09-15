# 设计资料核对记录

核对日期：2026-09-14。对应文档：[aclblasStpsv A2/A3 算子设计](design.md)。

## 官方资料

| 资料 | 本次读取内容 | 用途 |
| --- | --- | --- |
| 官方任务 ZIP，资源 ID `88e97c82a4274bed8dc0b6ab6091f322` | 任务书、测试 README、gen_csv.py、两份验证脚本、两份 CSV | 需求与验收依据 |
| `cann-ops-competitions/master/04_tasks/01_community-task-2026/resources/design_template.md` | 官方 raw 文本全文 | 保留需求背景、需求分析、详细设计、可维可测四部分结构 |
| 同目录 `README.md` | 官方 raw 文本全文 | 设计文档提交、评审及开发流程 |
| `ops-blas/master/include/cann_ops_blas.h` | 公共头文件中的 Stpsv 声明 | 接口参数顺序及类型 |
| `ops-blas/master/include/cann_ops_blas_common.h` | 官方 raw 文本全文 | diag 实际名称、枚举及状态码 |
| `ops-blas/master/blas/tpsv/README.md`、`blas/trsv/README.md` | 官方 raw 文本全文 | 支持范围、Device 指针及 stream 语义 |
| `ops-blas/master/blas/tpsv/arch35/stpsv_host.cpp` | 官方 raw 文本全文 | 当前 n=0 顺序、错误码和分派参考 |
| `ops-blas/master/test/tpsv/stpsv/stpsv_param.h` | 官方 raw 文本全文 | 现有测试参数解析 |
| Netlib `stpsv.f` | 参考实现 | packed 排布、前后向依赖和零分量处理 |

以上仓库资料读取自当日 master。开发分支确定后，测试报告补记使用的提交版本。

## KG 检索留痕

score 为搜索得分；N/A 表示通过索引或关联直接读取。

| id | source_file | score | 采用内容 |
| --- | --- | ---: | --- |
| `ascendc-operator-design` | `agent-skills/community/Op/ascendc-operator-design/SKILL.md` | N/A | 接口、算法、Tiling、UB、workspace、测试的设计流程 |
| `ascendc-operator-testcase-gen` | `agent-skills/community/Op/ascendc-operator-testcase-gen/SKILL.md` | N/A | 参数约束、边界与覆盖统计 |
| `agentskills_community_op_ascendcoperatordesign_templates_designtemplate_designtemplatemd` | `Official-skills/agent-skills/community/Op/ascendc-operator-design/templates/design-template.md` | N/A | 设计要素核对 |
| `agentskills_community_op_ascendcoperatordesign_references_generaltilingprinciples_generaltilingprinciplesmd` | `Official-skills/agent-skills/community/Op/ascendc-operator-design/references/general-tiling-principles.md` | 0.962795 | 核间依赖与UB规划 |
| `agentskills_community_op_ascendcoperatordesign_references_reductiontiling_reductiontilingmd` | `Official-skills/agent-skills/community/Op/ascendc-operator-design/references/reduction-tiling.md` | N/A | 分段归约设计 |
| `agentskills_community_op_ascendcoperatordesign_references_indextiling_indextilingmd` | `Official-skills/agent-skills/community/Op/ascendc-operator-design/references/index-tiling.md` | N/A | 索引和步长访问 |
| `ascdevkit_docs_zh_api_utilsapi_platforminfo_platformascendc_getcorememsize_ascendcplatform_getcorememsize_platform_ascendc_corememtype_ub_ub_size` | `ascend-c-kernel/asc-devkit/docs/zh/api/Utils-API/platform_info/PlatformAscendC/GetCoreMemSize.md` | 0.951643 | 平台UB容量查询 |
| `opsblas_blas_tpsv_readme_aclblasstpsv` | `cann-ops/ops-blas/blas/tpsv/README.md` | 0.934930 | float32、原地输出与步长语义 |
| `arch35_stpsv_kernel_stpsvkernel` | `cann-ops/ops-blas/blas/tpsv/arch35/stpsv_kernel.cpp` | 0.903182 | 同族求解参考 |
| `arch22_strsv_host` | `cann-ops/ops-blas/blas/trsv/arch22/strsv_host.cpp` | 0.906782 | arch22 工程参考 |
| `cannbotskills_pluginscommunity_ascendcportorchestrator_kb_target_ascendc_migration_memorybasevector_reducesum_sharedtmpbuffer` | `Official-skills/cannbot-skills/plugins-community/ascendc-port-orchestrator/kb/target/ascendc/migration/memory-base-vector/ReduceSum.md` | 0.920546 | 四参数 ReduceSum、A2/A3 float 支持、归约顺序和临时空间 |
| `cannbotskills_ops_ascendcperformancebestpractices_references_broadcast_datacopypaddesign_ascendc_datacopypad_ub_gm_ext_pad` | `Official-skills/cannbot-skills/ops/ascendc-performance-best-practices/references/broadcast/datacopypad_design.md` | 0.944701 | DataCopyPad 按有效字节搬运的调用方式 |

设计技能的配套模板、general-tiling-principles、reduction-tiling、index-tiling 已经读取；正文结构采用本任务指定的官方模板，接口和交付路径采用 BLAS 任务约定。

## 数学方案检查

用 Python 浮点数对本文 packed 地址和四类求解顺序做了离线检查：n 取 1、2、3、5、8、17、33，上下三角、N/T/C、两类 diag 与六种步长组合共 504 例。输入由已知解反算右端项，最大绝对误差为 `4.441e-16`；同时检查了列打包连续性、单位对角填 NaN、负步长和间隙保留。

该检查用于核对数学方案。Ascend C 编译、float32 精度和目标 NPU 性能结果在后续自测报告中记录。
