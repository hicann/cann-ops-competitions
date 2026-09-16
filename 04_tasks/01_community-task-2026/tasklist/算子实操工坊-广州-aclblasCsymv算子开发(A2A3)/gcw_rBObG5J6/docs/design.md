# aclblasCsymv（Atlas A2/A3）算子设计

> 状态：设计评审稿 v1.0（2026-09-15）。报名已通过，已在 CANN 9.1.0 / Ascend910B3 验证内存类型查询；尚未进行算子编译、精度或性能验证。
> 对应任务：算子实操工坊-广州-aclblasCsymv算子开发(A2/A3)。
> 提交账号：gcw_rBObG5J6。
> 源码基线：ops-blas `7eae2328a65753bf55cffc489253eb434ea3317e`。

## 需求背景（required）

### 需求来源

- [昇腾任务页面](https://www.hiascend.com/activities/task-center/details/b00c534a68654c9eaae9248a5e22f883?menu=guide)
- 任务页面“查看任务书”下载包中的 `aclblasCsymv_Atlas800IA3_task_doc.md`，尤其 §2、§3、§4、§5。
- [活动任务引导](https://gitcode.com/org/cann/discussions/283#tid-5b6e305a9b8440b2a3a78550dab541ca)
- [官方设计模板](https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md)

### 背景介绍及现状

实现单精度复数对称矩阵与向量乘加：`y = alpha * A * x + beta * y`。
目标矩阵满足 `A = A^T`，不要求也不执行共轭；对角元素的虚部参与计算。

基线仓有 `blas/symv/arch22/ssymv_*` 和 `blas/gemv/arch22/cgemv_*`，但没有
`aclblasCsymv` 公共声明或 `test/symv/csymv` 测试工程。
现有实数 host 实现的负维数、枚举错误码及空指针检查不完全符合本任务，不能直接沿用。
现有 `_aclblas_handle` 包含 stream 和 workspace，但没有 BLAS pointer-mode 字段或设置接口。

## 需求分析（required）

### 公共接口

在 `include/cann_ops_blas.h` 新增以下声明，不增设产品私有接口，不改变现有接口签名：

```cpp
aclblasStatus_t aclblasCsymv(
    aclblasHandle_t handle, aclblasFillMode_t uplo, int n,
    const aclblasComplex* alpha, const aclblasComplex* A, int lda,
    const aclblasComplex* x, int incx, const aclblasComplex* beta,
    aclblasComplex* y, int incy);
```

`aclblasComplex` 使用公共头文件现有定义：`float real; float imag;`，每元素 8 字节。

### 参数与功能拆解

| 参数 | 语义及约束 |
| --- | --- |
| handle | 已创建的句柄，kernel 使用其绑定的 stream |
| uplo | 仅接受 `ACLBLAS_UPPER=121`、`ACLBLAS_LOWER=122` |
| n | 逻辑矩阵阶数与向量长度，`n >= 0` |
| A、lda | Device 内存，列主序，地址 `row + col * lda`，`lda >= max(1,n)` |
| x、incx | Device 内存，逻辑长度 n，步长非零，至少 `1+(n-1)*abs(incx)` 个元素 |
| y、incy | Device 内存，原地输出，步长非零，存储长度计算同 x |
| alpha、beta | 指向复数标量；不可为 nullptr；分别识别 Host/Device，允许位置不同 |

仅访问指定三角中的数学有效元素。未指定三角和 lda padding 不作为有效输入。
除 lda/incx/incy 外不支持任意 Tensor stride、broadcast 或额外视图语义。
4096 是给定性能测试的最大规模，不作为接口的合法维数上限。
输入输出任意重叠未在任务书中定义；本设计不额外承诺 x/y 或 A/y 别名支持。

### 错误检查与退化路径

拟定检查顺序如下，多参数同时非法时以该顺序为实现约定，并单列测试：

1. handle 为空返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`。
2. 非法 uplo 返回 `ACLBLAS_STATUS_INVALID_ENUM`。
3. 负 n、不合法 lda、零 incx/incy、空 alpha/beta 返回 `ACLBLAS_STATUS_INVALID_VALUE`。
4. 合法 `n=0` 立即成功返回，不启动 kernel，不读取 A/x/y 或标量指向的内容。
5. 取得标量值后，`alpha=0 && beta=1` 立即成功返回，不要求 A/x/y 有有效地址。
6. alpha 非零时检查 A/x；需要写回时检查 y。
7. alpha 为零只执行 `y=beta*y`，不读 A/x；beta 为零直接置零，不读旧 y。
8. 一般计算的 beta 为零分支不加载旧 y，避免 `0 * NaN` 污染结果。

所有长度、负步长起点与地址乘法使用 64 位计算；先扩宽 inc 再取负，避免 `INT_MIN`
取绝对值溢出；对 workspace 字节数及容量进行溢出和边界检查。

## 详细设计（required）

### 数学与存储映射

对于逻辑元素 `(i,j)`：

```text
UPPER: row=min(i,j), col=max(i,j)
LOWER: row=max(i,j), col=min(i,j)
Aeff(i,j)=A[row+col*lda]

offset(k,inc) = k*inc                       (inc>0)
                (n-1-k)*(-int64(inc))       (inc<0)

(ar+i*ai)*(xr+i*xi) = (ar*xr-ai*xi) + i*(ar*xi+ai*xr)
```

归约与复数乘加以 FP32 执行，不转换为 FP16/BF16，也不使用 Hermitian 对角实化规则。

### Host 侧设计

- 校验参数并选择 quick return、scale、小规模通用或三角分块路径。
- kernel 使用 Ascend C 直调，tiling 结构按现有工程惯例按值传递。
- 从目标平台获取 AIV 核数和 UB 容量；分核数受任务数量、UB 预算约束，避免照搬固定 8 核。
- 一般 Host 标量路径仅提交 stream 工作，不在每次调用末尾同步或动态申请释放临时内存。
- 分块路径使用 handle 已有 workspace；容量不足时沿用仓内管理接口及错误返回，避免悬空内存。
- 同一 handle/workspace 不承诺多 stream 并发复用；不同 handle 以独立 workspace 测试隔离性。

### Host/Device 标量处理与评审点

任务书要求 alpha/beta 可在 Host 或 Device，但基线 handle 没有 pointer mode。
不能在 Host 上直接解引用 Device 指针，也不能将所有查询失败的指针默认为 Host 指针。

2026-09-15 在 CANN 9.1.0、Ascend910B3 上编译运行独立 AscendCL 探针，
`aclrtPointerGetAttributes` 对以下有效指针均返回 `ACL_SUCCESS`：

| 内存来源 | location.type 实测值 | 处理方式 |
| --- | --- | --- |
| 普通栈、malloc | `ACL_MEM_LOCATION_TYPE_UNREGISTERED (2)` | 由调用方保证为有效 Host 标量，复制其值 |
| aclrtMallocHost | `ACL_MEM_LOCATION_TYPE_HOST (0)` | 复制 Host 标量值 |
| aclrtMalloc 及其有效偏移地址 | `ACL_MEM_LOCATION_TYPE_DEVICE (1)` | 按 Device 标量处理 |

类型查询失败直接映射运行时错误，不尝试解引用；未知类型不作为普通 Host 内存处理。
该查询识别内存位置，不证明指针指向至少 8 字节有效对象；对象有效性及生命周期由调用方保证。
初版范围为普通 Host、AscendCL Host 和当前设备 Device 内存，托管内存及跨设备指针另行明确。

Host 标量路径将值复制到 tiling 参数，kernel 异步提交到 handle 绑定的 stream。
存在 Device 标量时，拟先同步该 stream，保证此前标量写入完成，再将两个分量拷回 Host，
随后执行依赖标量值的 quick return 和条件空指针检查；一般计算的 kernel 仍提交到同一 stream。
两个标量只需一次前置 stream 同步，混合 Host/Device 场景分别读取。读取失败不启动计算。

**提请评审确认：** 当前公共句柄无 pointer mode，上述自动识别及 Device 标量前置同步是否符合统一接口约定？
若要求 Device 标量路径也完全异步，需要明确依赖标量内容的条件空指针错误码如何返回。
本稿不新增公共 pointer-mode API。已验证的是位置查询，Device 标量读回、stream 顺序和异常映射将在实现阶段测试。
性能测试将明确标量位置，另列 Device 标量路径的 Host 总耗时，不能以 Host 路径结果替代 Device 路径验证。

### Kernel 路径一：通用正确性路径

按输出行分配工作，每个输出行遍历逻辑列，通过上述三角映射访问 A；分块加载、复数乘加并归约。
每个输出元素仅由一个逻辑任务写回。支持 padding、正负步长、尾块与小规模输入。
非连续 y 的写回先采用单核，避免多个核心在同一物理写回块内更新不同元素造成覆盖。
通用路径也用于大于当前优化容量的合法 n，不能依赖完整矩阵在 UB 中驻留。

### Kernel 路径二：三角分块性能路径

目标：给定 `n=512/2048/4096`、`incx=incy=1` 场景。

1. 将有效三角划分为 B×B tile，初始候选 B=32 或 64；按 tile 工作量均衡分配给 P 个 AIV 核。
2. 非对角 tile `A[I,J]` 同时产生 `s[I]+=A[I,J]*x[J]` 与
   `s[J]+=A[I,J]^T*x[I]`，矩阵元素搬入一次，完成两侧贡献。转置不取共轭。
3. 对角 tile 仅搬入指定三角，逐列控制有效拷贝长度，再在 UB 内按对称性补齐，
   不从 GM 读取无效三角；对角贡献只计算一次。
4. 每核维护独立的实部、虚部部分和；写入 `partial[core][component][index]`。
   各核写不同 workspace 区间，不直接向 y 原子累加。
5. 同 stream 的归约 kernel 读取全部核心部分和，计算 alpha/beta，按输出块独占写回 y。
   workspace 通过 stream 顺序建立依赖，不跨 kernel 使用未定义的核间屏障。

三角 tile 内部以实部、虚部分离的 FP32 数据运算，连续列片段使用 DataCopy/DataCopyPad，
通过 TQue/TPipe 管理搬入、向量运算及搬出的依赖。所有尾部填充值为零，且归约只覆盖逻辑有效项。
缓冲区的消费结束后才能复用；不在不同流水之间省略必要事件。

### UB、workspace 与调优边界

以 B=64、n=4096 的候选方案估算：

| 项目 | 候选预算 |
| --- | --- |
| 两个交错复数矩阵 tile | `2*64*64*8 = 64 KiB` |
| 一个 tile 的实/虚分离工作区 | `64*64*8 = 32 KiB` |
| 每核完整部分和 | `4096*8 = 32 KiB` |
| 每核 x 缓存 | `4096*8 = 32 KiB` |
| 合计，不含临时向量/队列对齐 | `160 KiB` |

最终预算必须加上 reduction scratch、队列和对齐空间，并与设备查询的 UB 大小核对。
如果预算超限则缩小 B、减少缓冲或切换通用路径，禁止超过 UB 容量。
每核部分和行做至少 32 字节对齐；两分量总 workspace 为约 `P*align_up(n,8)*8` 字节，
另按实际实现添加对齐或临时区。所有部分和每次调用先清零，不读取历史调用残留。

大尺寸优先降低矩阵重复搬运及三角任务负载不均衡；小尺寸比较单次 kernel 与两阶段 kernel 的启动开销。
切换阈值由实测确定；以上为设计候选，不代表已达到性能指标。

### 代码集成范围

```text
include/cann_ops_blas.h                 新增公共声明
blas/symv/README.md                    新增参数说明、示例与 A2/A3 产品支持表
blas/symv/arch22/csymv_host.cpp         参数校验、tiling、调度
blas/symv/arch22/csymv_kernel.cpp       Ascend C kernel 与直调封装
blas/symv/arch22/csymv_kernel.h         kernel 启动声明
blas/symv/arch22/csymv_tiling_data.h    tiling 定义
test/symv/csymv/CMakeLists.txt          注册 GTest 工程
test/symv/csymv/csymv_param.h           CSV 解析与输入生成参数
test/symv/csymv/csymv_golden.h          独立 CPU 复数参考实现
test/symv/csymv/arch22/                 测试代码、wrapper 与 CSV
```

`blas/CMakeLists.txt` 自动收集 arch22 源码；测试注册使用 `ops_blas_add_gtest_tests`。
不得复制 competitions 通用 ACLNN 示例目录作为本算子的最终实现工程。

## 可维可测分析

### 环境与构建

验收环境要求 CANN 9.1.0、Atlas A2/A3；性能设备为 910B3。
当前开发环境已确认 CANN 安装路径为 `/usr/local/Ascend/cann-9.1.0`，
`acl.get_soc_name()` 返回 `Ascend910B3`，运行时识别一个逻辑设备，内存属性探针执行成功。
该结果仅证明环境和所测 API 可用，不等同于本算子通过编译或验收。
任务书中机型名出现 800T/800I 两种写法，记录实际机型和完整 `npu-smi` 信息，按 910B3 对齐测试。
候选构建命令：`bash build.sh --soc=ascend910b3 --ops=csymv`，须以实际构建产物确认成功。

### 精度与覆盖

- 实部和虚部分别计算 `abs_err <= 2^-16 + 2^-10*abs(golden)`，匹配比例至少 0.99。
- 为避免任务书“1e-2 或 32 ULP”未明确选择方式，初始验收采用明确的 `max_abs_error <= 1e-2`；
  若需 ULP 分支，先明确规则再实现并披露。同步输出框架要求的 MERE/MARE，不以一种指标替代另一种。
- NaN、Inf 先按类别比较（Inf 检查符号），有限值单独计算误差；分类不匹配直接失败，禁止 NaN 比较绕过失败。
- CPU golden 独立实现三角遍历及复数算术，用手算小矩阵和完整展开矩阵参考交叉验证；
  数值累加精度和特殊值传播规则固定并记录，避免使用不含 csymv 的 cblas 函数充当 golden。
- 保留原始 1200 条用例映射；修正 `TC_ED_190` 为 `INVALID_ENUM`，另附差异记录。
- 补充独立实/虚的 50% 均匀与 50% 正态输入及随机复数标量；保留给定固定标量和特殊值用例。
- 补测 nullptr handle、alpha=0 时空 A/x、quick return 时空 y、n=0 与非法参数组合、
  未引用三角填 NaN、非零虚部对角、lda padding 毒值、步长间隙哨兵、偏移地址和重复调用。
- 添加独立 stream/handle、Host/Device 混合标量、同 stream 先写后读标量及 Device quick return 测试。

### 性能标准与测量

| 用例 | uplo | n | 平均单次算子耗时上限（us） |
| --- | --- | --- | --- |
| 1 | UPPER | 512 | 15.45 |
| 2 | UPPER | 2048 | 46.14 |
| 3 | LOWER | 2048 | 38.57 |
| 4 | UPPER | 4096 | 146.00 |
| 5 | LOWER | 4096 | 99.55 |

先完成编译、数据生成、拷贝与预热，预热建议 10 次，有效测量至少 100 次。
以 AscendCL 设备事件对同 stream 完整算子 kernel 序列计时，或用 msprof 交叉验证；
多 kernel 的准备、部分和、归约均计入算子时间，不能只报告最快的某一个 kernel。
报告各次样本、均值、测量单位、计时方式、设备和软件版本。GTest 的整数毫秒总时长不作为 NPU 算子时间。
Host 端总调用耗时可另列，不能与设备计时混淆。固定输入，重复调用 beta 非零用例时在计时区外恢复 y，
防止多次原地计算造成输入溢出；已给性能用例 beta 为零，可直接重复。

首五项直接与任务书上限比较，不再将任务书上限除以 0.8。
其他 195 项报告实测值；缺少明确基准的项目标记 `NO_REF`，不能作为性能通过证据。
测量输入、输出、workspace 的分配字节和设备峰值占用；任务书没有内存通过阈值，不虚构门限。

### 验证结果完整性

测试程序非零退出、超时、零用例、缺失用例、重复结果、解析失败、NaN 耗时均使自验失败。
测试脚本读取专用结构化结果或 GTest XML，保存原始日志；不只依赖控制台 PASS/FAIL 正则。
报告区分“已执行通过”“失败”“未执行”“无基线”，并附可复现命令与源码提交号。

## 兼容性分析

新增公共符号，A2/A3 同用 arch22 实现，不改变既有 Ssymv 行为。
README 仅在完成对应验证后标注 A2/A3 支持；不扩展到任务外产品。
标量位置识别已作环境验证；Device 标量同步约定在本稿中明确列为评审点，依评审结论实现并验证。

## 交付和评审顺序

1. 提交本设计文档 PR，更新任务进展及讨论区链接，按意见完善设计并合入。
2. 按确认后的设计完成算子和测试工程，在 CANN 9.1.0 / 910B3 上执行全量自验与性能优化。
3. 整理验收代码目录、自验报告、原始日志及截图，提交社区验收。
4. 按官方验收结果和任务引导提交需求 Issue、代码 PR，跟进评审合入。
