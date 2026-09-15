# 需求背景（required）

## 需求来源

社区任务《aclblasCsyr2 算子开发任务书》（csyr2.zip）。设计结构采用 cann-competitions 的 design_template.md，模板来源提交 `1fcd3c85dddbbaff46988704de7a1bfbbafbfded`。

## 背景介绍

在 Atlas A2/A3 上实现 complex64 对称秩 2 更新，接入 ops-blas 公共句柄接口。矩阵按列主序存储，复数由两个 float32 分量组成。对称性采用普通转置，不使用共轭。

# 需求分析（required）

## 需求描述

实现 `A += alpha*x*y^T + alpha*y*x^T`。仅更新 uplo 指定三角，保留另一三角及 lda padding。支持正负非零向量步长、运行时 n、复数 alpha 和异步 stream 执行。

## 需求拆解

1. 共享 `include/cann_ops_blas.h` 中的 `aclblasCsyr2` 声明。
2. Host 验证 handle、uplo、维度、步长、lda 和 alpha 指针；合法 n=0 或 alpha=0 不启动 kernel。
3. alpha=(1,0)、incx=incy=1 全部进入专用向量路径，支持合法 lda padding；其他步长和复数 alpha 使用通用路径。
4. 按附件执行 996 精度和 200 性能用例，实虚部分别验证，并补足附件生成器未实现的正态分布测试。
5. 按任务书归档源码、测试、复现说明及报告。评审/合入状态必须由实际 PR 证据证明。

# 详细设计（required）

## 算子分析

### 数学公式

令 p=alpha*x[c]、q=alpha*y[c]，则 `A[r,c] += p*y[r] + q*x[r]`。每次复数乘法分别计算实部差与虚部和，然后合并两项，避免展开连续累加引入额外抵消误差。

### 支持数据类型和形状

输入输出为 complex64。n>=0，lda>=max(1,n)，incx/incy 非零。逻辑矩阵为 n×n，物理存储为 lda×n；矩形覆盖通过 padding 表达，不引入与 API 不符的独立 m 参数。

## 算子实现

### Host 侧设计

接口通过 handle 获取 stream，不主动同步。非法枚举返回 INVALID_ENUM，其余参数错误按任务书返回 INVALID_VALUE；空 handle 返回 HANDLE_IS_NULLPTR。只有 n>0 且 alpha 非零时才要求 x/y/A 非空。

### Kernel 侧设计

Host 通过 `GetAivCoreCount()` 获取 AIV 数量，当前测试设备为 48。alpha=(1,0)、incx=incy=1 的非空更新全部使用专用向量实现，不要求 n 对齐或 lda=n。

- n<=2048：四列分组、矩阵双缓冲；上三角按加权前缀划分列区间，下三角按后缀工作量划分。每核至多一组时直接分配四列，避免空核。
- UPPER、2048<n<=6144：单列双缓冲与 2048/3072 行块；UB 容量决定行块大小，只加载及旋转本核需要的向量前缀。分核代价包含三角元素量、每列固定代价和行块搬运代价。
- LOWER、2048<n<=4096：仅驻留本核需要的向量后缀。四个向量占用为 `4*align256(8*(n-first))`；剩余 192 KiB UB 扣除 256 字节对角线暂存区后，分给两个四列矩阵缓冲，动态计算行块长度。两个矩阵缓冲分别使用独立对角线暂存区。
- 更大的 unit-stride 更新：使用固定 512 行、四列分组的分块路径，不分配随 n 增长的 UB 向量。

通用标量路径处理其他 alpha/步长，负步长逻辑下标为 `(n-1-i)*abs(inc)`，n<64 使用一个核。复数乘法采用分组运算控制抵消误差。共用旋转索引、复数向量更新和下三角负载函数，不保留 Dense/Cube/Mix 算法或环境变量开关，不按测试数据分派。

### 存储与三角访问

上三角按每列有效长度搬运。下三角把对角线短头与列体分开，短头通过 Gather 打包后按实际长度写回，列体使用批量搬运；尾组使用实际列数。生产 helper 的布局检查验证地址覆盖及内容，设备测试还分别检查未引用三角污染、未更新三角和 lda padding 逐位不变。缓存行填充与 API 的逻辑搬运范围是不同层次，不将内容不变单独作为未读取的证据。

## 支持硬件

目标产品：Atlas A2/A3。实际设备、CANN 精确版本以及正式版本验证结果见验收报告；尚未完成的硬件验证不宣称通过。

## 算子约束限制

无广播；alpha 位于 Host，x/y/A 位于 Device。调用方负责读回前同步 stream。

# 可维可测分析

## 精度标准/性能标准

实部和虚部分别要求 rtol=2^-10、atol=2^-16、matched_ratio>=0.99，且分量误差满足 1e-2 或 32 ULP 上限。NaN 按分类匹配，Inf 要求同号，有限值采用数值阈值。

任务书要求预热后采样超过 50 次取平均；正式复现使用 30 次预热、60 次采样。快速调优使用 3 次预热、6 次采样，单独标识。GPU 基线 `gpu_ms` 乘 1000 转为微秒，再除以 0.8 作为门槛；NPU 采用暖缓存 profiler Task Duration。

覆盖当前 996 个精度用例、200 个性能用例及额外边界测试，实部、虚部分别判定。正负非单位步长和 lda padding 没有附件单独性能门槛。逐项结果、最新源码测试范围及仍未达标的性能项以 [自测报告](https://gitcode.com/guodong54_/ops-blas/blob/test/csyr2-complete-acceptance/docs/csyr2/delivery/report.md) 为准，不将不同版本结果合并声明为当前全量通过。

## 兼容性分析

不新增私有平行 API，公共声明位于 `include/cann_ops_blas.h`。复用 CSV/GTest 与 harness 远端构建运行链路。Atlas A2/A3 标注支持；实际测试使用 Ascend910_9382 / CANN 9.1.0-beta.3，其他环境结果不作推断。设计评审合入、仓库开发者邀请和任务系统提交状态见 [交付清单](https://gitcode.com/guodong54_/ops-blas/blob/test/csyr2-complete-acceptance/docs/csyr2/delivery/README.md)。
