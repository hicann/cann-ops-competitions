# 【CANN社区任务】SpMV（950）算子设计文档

## 需求背景（required）

### 需求来源

CANN 社区任务"8月社区任务-SpMV算子开发"（适配 Ascend 950PR）。参考 cuSPARSE SpMV 实现，在昇腾 NPU 上基于 Ascend C 编程语言实现功能一致的稀疏矩阵向量乘算子。

### 背景介绍

稀疏矩阵向量乘（SpMV）是科学计算、图神经网络、推荐系统等领域的核心计算原语。大规模稀疏场景下（稀疏度 50%~99.9%），SpMV 的性能主要受访存模式制约：CSR 格式的列索引随机访问导致 x 向量的非连续读取，且行间非零元个数不均衡带来负载不均问题。现有的 ops-sparse 仓库 spmv 目录（A2 版本）基于 Vector 编程模型实现，无法在 950PR（arch35，SIMT 编程模型）上运行，需要面向新硬件重新实现。

## 需求分析（required）

### 需求描述

使用 Ascend C 编程语言（SIMT 模型）实现 aclsparse 风格的 SpMV 三阶段接口，与 cuSPARSE SpMV 核心功能对齐，在 Ascend 950PR 上满足生态算子开源精度标准，整体性能达到 0.5 倍标杆水平，且支持确定性计算。

### 需求拆解

1. 支持三阶段调用：GetBufferSize / Preprocess / SpMV
2. 支持 5 组数据类型组合：fp32→fp32、i8→i32、i8/fp16/bf16→fp32、fp16→fp16、bf16→bf16
3. 支持转置（opA = A^T）与非转置
4. 支持 CSR_ALG1 / CSR_ALG2 两种算法
5. 支持原位累加（y_vec 输入输出）、alpha/beta 标量（含 0）、HOST/DEVICE 指针模式
6. 支持稀疏度 50%~99.9% 泛化、行偏移 i32/i64
7. 确定性计算：相同输入多次执行结果一致
8. 性能：任务书 4 case 全部达到 ≥0.5 倍标杆（耗时 ≤2×标杆）

## 详细设计（required）

### 算子分析

#### 数学公式

`Y = alpha * op(A) * X + beta * Y`，其中 A 为 M×K CSR 稀疏矩阵（csrRowPtr/csrColInd/csrVal 三数组表示），op(A)=A^T 时输出维度由 M 变为 K。

#### 支持数据类型

| A/X | computeType | Y |
|---|---|---|
| float32 | float32 | float32 |
| int8 | int32 | int32 |
| int8 / float16 / bfloat16 | float32 | float32 |
| float16 | float32 | float16 |
| bfloat16 | float32 | bfloat16 |

#### 支持形状

CSR 任意形状（M、K ≤ INT32_MAX），稀疏度 50%~99.9%，支持 nnz=0（退化为 Y=beta*Y）。

### 算子实现

#### 实现方案

采用 Ascend C 三层结构：`__simt_vf__` 计算函数（线程级 grid-stride）+ `__global__ __aicore__` Dispatcher（KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY)）+ `kernel_do` 启动器（`<<<blocks, nullptr, stream>>>` 异步 launch）。

##### host 侧设计

**tiling 策略**：Host 侧按矩阵规模选择调度策略并写入 SpmvTilingData（含 m/indexBase/algType/dtype 编码/alpha/beta/rowsPerBlock/dualChain）：

1. 分核策略：`useBlocks = min(aivCoreNum * mult, CeilDiv(m, 256))`。小矩阵（行数 ≤ 16384，即不超过全部 AIV 核的线程总容量）mult=1，一线程一行模型；长行大矩阵 mult=6，block 倍增提升 x 随机 gather 的访存级并行度。
2. 双链 ILP：大矩阵启用 dualChain，kernel 内每行拆偶/奇两条独立 FMA+gather 链并行发射，隐藏 L2 访存延迟；小矩阵保持单链顺序累加。
3. ALG2 预处理：Preprocess 阶段（device kernel，单线程）按行 nnz 降序归并排序生成 reorder 表，并按负载均衡生成 bin_edge 行切分边界，execute 按 bin_edge 分配各 core 行区间。
4. 转置：Preprocess 阶段将 CSR 物化为 CSC（host 侧 D2H→CPU 列计数+顺序散填→H2D，列内按行序保证确定性），execute 复用主 kernel 消费 CSC；无 workspace 时退化为单线程确定性 scatter kernel。
5. workspace 布局：ALG2 区域（reorder/tmpReorder/scratch/binEdge）+ 可选段并行区，均 64B 对齐。

tilingkey 规划：不使用 tilingkey，dtype/alg/dualChain 分支经 tiling 字段在 dispatcher 内按模板参数分发（6 参模板 ValT/XT/YT/CompT/RowT/UseReorder + DualChain）。

##### kernel 侧设计

1. 主计算（SpMV）：每线程处理一行（或双链下半行），行内顺序 FMA 累加（float 路径）/int32 顺序累加（i8→i32 路径），最后 `z = alpha*acc + beta*y[r]` 原位写回。beta=0 短路跳过 y 的 GM 读。
2. 半精度处理：fp16/bf16 输入经 static_cast 提升 float 计算（与 toolkit half/bfloat16_t 类型衔接），输出按 Y dtype 降写。
3. 确定性：行内顺序累加（无原子、无跨线程归约），ALG2 重排后行内元素顺序固定，相同输入多次执行结果位级一致。
4. 精度：顺序 FMA 累加误差 ~n*eps（n 为行长），远低于生态算子开源精度标准（FP32 rtol=2^-10）；int8→int32 路径整数累加精确。
5. nnz=0：走 BetaY 快捷 kernel（Y = beta*Y）。

##### 性能优化要点

1. 去 Priest 三重补偿：生态算子精度标准实测查证后，顺序 FMA 累加即可满足容差，消除 ~15 指令/nnz 的计算链（case4 -14%）。
2. 线程规模 256/block + block 倍增：提高 x 随机 gather 的访存级并行度（case4 807→418us）。
3. 双链 ILP：偶/奇双 FMA 链并行发射（case4 418→353us，数学等价）。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Ascend 950PR | √ |

## 算子约束限制

1. m、k 不超过 INT32_MAX；
2. csrRowPtr 须单调递增，csrColInd 须在有效列范围内（违例触发参数校验报错）；
3. 确定性模式下 ALG1/ALG2 各自结果一致，但两算法间因行内累加顺序差异允许精度容差内的差异。

## 可维可测分析

算子日志通过 OP_LOGD/OP_LOGI/OP_LOGW/OP_LOGE 输出（log 模块统一管理）；launch 关键参数（m/nnz/numBlocks/alg/dtype）均有 INFO 级日志。测试工程覆盖定向精度（19 case）、性能对标（4 case）、官方泛化用例（case_200.json 全 200 case，ALG1/ALG2 双跑），全部可复现（见算子仓 sparse/spmv/arch35/test/README.md）。

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | 生态算子开源精度标准（experimental_standard）：FP32 rtol=2^-10/atol=2^-16、FP16 rtol=2^-9、BF16 rtol=2^-6，matched_ratio ≥ 0.99；确定性输出 | https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md |
| 性能标准 | 任务书 4 case 整体性能 ≥ 0.5 倍标杆 | 社区任务书 3.3 |

实测（Ascend 950PR，aclrtEvent 计时）：case1=2.45us（标杆 43.9）、case2=3.57us（46.3）、case3=43.48us（45.4）、case4=353.24us（193.2），全部达标；定向精度 19/19 PASS、泛化 case_200 200/200 PASS。

## 兼容性分析

新增 arch35 算子目录（sparse/spmv/arch35/），与既有 A2 版本实现并存，通过 build.sh 的 SOC 过滤（ascend950 匹配 arch35 目录）按硬件自动选择编译，不影响既有版本兼容性。接口与 cann_ops_sparse.h 已声明的 aclsparse 三阶段 API 保持一致。
