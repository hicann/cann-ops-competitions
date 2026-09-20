# Ascend C Basic API 指针化（VECTOR 分册）设计文档

本设计的范围采用 VECTOR 在线清单，精度采用最新官方文档，开发与验证环境采用 CANN 9.2。第 3.4 节说明验收细节的确认方式。

## 一、需求描述

### 1.1 需求来源

本设计对应[9 月社区任务：AscendC Basic_API 优化实现（VECTOR 矢量接口扩展）](https://www.hiascend.com/activities/task-center/details/7dda608eaee54623b53a0eb72f366c53)。在 Ascend C Basic API 的现有模板接口中增加裸指针入参能力，支持开发者使用 LocalTensor 或等价 UB 指针调用同名 API，并保持原 Tensor 调用兼容。

本任务属于基础 API 扩展，不新增 ACLNN 算子。目标硬件为 Ascend 950 系列，软件环境采用 CANN 9.2，编译器及目标架构参数与实际环境匹配。

### 1.2 交付范围

交付对象为 VECTOR 在线清单中各个重载，包括其中明确列出的排序、转置等接口。矩阵专用接口、Proposal 拆分接口以及其他分册接口不因所在头文件被修改而自动纳入交付。

范围以[在线表格](https://docs.qq.com/sheet/DYVNBU3BUUVdVRFRu?tab=000001)为准，2026-09-19 核对为 85 个接口名、256 条记录。逐条映射到源码和测试，区分重复声明与独立重载，不能仅依据总数验收。清单列出的 `Copy`、`Fill`、`MrgSort` 等接口也纳入范围，不按头文件所属类别排除。

已核对[任务讨论帖发布的 XLSX](https://gitcode.com/user-attachments/files/8547062/2ba444b55f58403b9b0436f92a261108.xlsx)，其 256 条记录的接口名、重载序号、重载总数、头文件及完整签名均与在线清单一致。

依照[讨论帖 26](https://gitcode.com/cann/asc-devkit/discussions/26)中 2026-09-17 的回复，仅修改 Ascend 950 支持的接口；`ProposalConcat`、`RpSort16`、`MrgSort4`、`ProposalExtract` 不作为验收对象。清单中名称不同的 `MrgSort`、`Sort` 不因此排除。

## 二、接口及方案设计

### 2.1 算子原型与接口语义

本任务保持每个现有 API 的计算原型。以 `Add` 为例，其有效输出元素满足 `dst[i] = src0[i] + src1[i]`；参与运算的元素、访问位置及重复次数继续由原有 count、mask 和 repeat/stride 参数决定，不新增广播或类型转换规则。

现有三种 `Add` 调用原型如下，原 Tensor 调用及其显式模板参数方式需要继续有效。

```cpp
template <typename T, bool isSetMask = true>
__aicore__ inline void Add(
    const LocalTensor<T>& dst,
    const LocalTensor<T>& src0,
    const LocalTensor<T>& src1,
    uint64_t mask[], const uint8_t repeatTime,
    const BinaryRepeatParams& repeatParams);

template <typename T, bool isSetMask = true>
__aicore__ inline void Add(
    const LocalTensor<T>& dst,
    const LocalTensor<T>& src0,
    const LocalTensor<T>& src1,
    uint64_t mask, const uint8_t repeatTime,
    const BinaryRepeatParams& repeatParams);

template <typename T>
__aicore__ inline void Add(
    const LocalTensor<T>& dst,
    const LocalTensor<T>& src0,
    const LocalTensor<T>& src1,
    const int32_t& count);
```

指针化后的接口行为要求是：上述每个 LocalTensor 操作数均可独立选择 Tensor 或等价 UB 指针，函数名及其余实参的顺序和含义不变。例如，在缓冲区和同步条件均合法时，以下三种调用应计算相同结果。

```cpp
// dstPtr, src0Ptr and src1Ptr are equivalent __ubuf__ float* buffers.
AscendC::Add(dstTensor, src0Tensor, src1Tensor, count);
AscendC::Add(dstPtr, src0Ptr, src1Ptr, count);
AscendC::Add(dstPtr, src0Tensor, src1Ptr, count);
```

其他重载按相同原则扩展，但保留其各自的输入输出类型关系。例如 `Cast` 可以有不同的源与目标元素类型，比较结果可以是独立的掩码类型，不能为了统一模板而强制它们相同。类静态成员、临时缓冲区及结构体包含的 Tensor 操作数按在线清单逐项处理。

### 2.2 参数、返回值与约束

| 参数或行为 | 接口约定 |
| --- | --- |
| 输入及输出操作数。 | 接受原 Tensor 表示或等价的有效 UB 指针；每个操作数独立选择表示方式。输出位置必须可写，缓冲区大小满足原 API 要求。 |
| 元素类型。 | 沿用该重载在目标硬件支持的类型及类型配对，不因支持指针而新增数据类型。 |
| 标量与枚举。 | 保留 scalar、round mode、比较模式等原类型、取值约束及含义。 |
| mask、repeat 和 stride。 | 保留原计算范围、步长单位、重复次数及设置 mask 的行为，不改变默认选项。 |
| count。 | 继续表示对应原 API 规定的元素数量或计数，不改成字节数，也不统一更改其整数类型。 |
| 返回值及附加输出。 | 保留原返回类型、状态输出、索引输出和参数副作用。`Add` 返回 `void`，结果写入 dst。 |
| 布局、对齐和别名。 | 沿用原 API 对地址空间、对齐、临时空间、地址重叠及同步的约束，不承诺新增对非法缓冲区的容错行为。 |
| Tensor 兼容性。 | 原有自动推导和显式模板调用均保持有效，原 Tensor 检查及功能不得因指针支持而退化。 |
| 非法类型。 | 错误地址空间或不满足原类型关系的调用按模板约束拒绝，不通过无条件强制转换使其通过。 |

### 2.3 实现原则

对外声明位于 `include/basic_api`，模板实现位于 `impl/basic_api`。在编译期区分操作数表示并统一提取硬件指针与元素类型，调用既有 `*Impl`。沿用原数值实现、内存使用及同步语义，不引入与输入规模线性相关的额外 Device 拷贝。

各操作数需要独立推导表示类型，以支持混合调用；同时保持旧的显式模板实参和重载选择行为。模板适配的具体组织方式不改变本文的接口行为约定。原 Tensor 调试检查继续保留；裸指针不包含 Tensor 的全部长度和位置元数据，其诊断能力按实际可用信息处理。

### 2.4 测试设计

基于 asc-devkit 官方 `examples/01_simd_cpp_api/03_basic_api/` 和任务附件改造及回归，复用既有样例工程。对每个最终交付重载记录类型、输入、布局、调用表示、编译结果和数值结果。

讨论帖已说明附件样例主要用于验证示范，并建议多 Shape 回归。测试按照接口合法约束覆盖多种 Shape，不以附件默认规格作为全部覆盖范围。

| 测试项 | 检验内容 |
| --- | --- |
| 原 Tensor 回归。 | 改造前后使用同一组输入和参数，保留原结果和接口调用行为。 |
| 纯指针调用。 | 使用等价底层缓冲区和布局，验证指针路径与对应 Tensor 路径的结果及副作用。 |
| 混合调用。 | 各操作数独立切换表示；二输入一输出接口覆盖全部 8 种 Pointer/Tensor 组合，其他接口按实际操作数结构设计。 |
| 模板兼容性。 | 覆盖自动推导、既有显式类型参数和布尔模板参数，以及合法和非法的类型关系。 |
| 任务指定数据。 | 对适用接口覆盖 Shape `1/32/1024/2048` 和 `[-100,100]` 内的合法输入，固定布局接口的适用方式按第 3.4 节确认。 |
| 边界与内存行为。 | 覆盖 mask 未写区域、尾部、非默认 stride、合法地址重叠、临时缓冲区及输出有效范围。 |
| 特殊输出。 | 对整数、索引、位掩码和搬运结果按业务语义精确验证，对 NaN/Inf 单独判断。 |

Header Checker 和 CPU UT 验证接口编译及相关检查；最终计算正确性由 Ascend 950 真机验证。完整日志需包括实际设备、CANN/编译器版本、源码版本、构建运行命令、输入规格和结果判定。退出码、编译通过和设备精度通过分别报告。

## 三、可维可测与验收指标

### 3.1 精度指标

原 Tensor 用例须全部回归通过，指针路径和混合路径须在相同输入和布局下保持对应数值语义。浮点计算按[最新生态算子混合容差精度标准](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/mixed_tolerance_standard.md)执行：逐元素采用 `abs(actual-golden) <= atol + rtol*abs(golden)`，每个用例同时满足匹配率至少 0.99 和该类型的最大绝对误差上限。

2026-09-19 核实的最新文档修改提交为 [`233ee24cc3500d36bd4c2a51f8872d67857cec08`](https://gitcode.com/cann/opbase/blob/233ee24cc3500d36bd4c2a51f8872d67857cec08/docs/zh/ops_precision_standard/mixed_tolerance_standard.md)。对应通过阈值如下；只对各接口实际支持的类型应用，不据此新增接口类型支持。

| dtype | rtol | atol | 最大绝对误差上限 |
| --- | --- | --- | --- |
| FLOAT16 | `2^-10` | `2^-10` | `1e-1 or 32 * ULP`。 |
| BFLOAT16 | `2^-7` | `2^-7` | `1e-1 or 32 * ULP`。 |
| FLOAT32 | `2^-13` | `2^-13` | `1e-2 or 32 * ULP`。 |
| HiFLOAT32 | `2^-10` | `2^-10` | `1e-1 or 32 * ULP`。 |
| FLOAT8 E4M3 | `2^-3` | `2^-3` | `1 or 32 * ULP`。 |
| FLOAT8 E5M2 | `2^-2` | `2^-2` | `1 or 32 * ULP`。 |

上限中的 `or` 沿用标准原文，具体选择规则需核实，不能直接解释为取较宽松值。标准对大数归约的放宽说明不自动适用于所有归约用例。附件校验作为原样例回归证据，正式浮点判据以最新标准为准；自测报告记录实际采用的文档版本。

整数、索引、布尔和搬运不套用浮点容差标准，按原 API 的业务语义验证。golden 的生成方式、有效输出元素数、特殊值处理及容差统计范围写入自测说明。

### 3.2 性能与内存指标

本任务没有额外性能门槛和标杆时延要求，性能 case 可注明“无”。相对改造前的性能观察可作为自测报告补充，不设置额外加速比或流水线占比条件。

接口扩展应在编译期完成，不改变原 `*Impl` 数值语义，不引入随输入规模线性增长的额外 Device 数据拷贝，并保持原 Tensor 内存使用语义。若执行性能对比，使用同环境同输入的原 Tensor 版本作为基线，以官方 `msopprof` 数据说明测量范围和差异。

### 3.3 交付与复现

提供设计评审材料、最终接口覆盖说明、可复现的样例和 README、精度用例、自测报告与截图、全部测试日志，以及发现问题时的易用性反馈链接。待验收代码提供个人私有仓库和分支地址，并邀请 `Ascend-CANN` 为开发者。设计评审、自测通过、任务验收及后续代码合入分别确认。

验收材料标明确切 `commitID`，测试报告对应同一版本。按照讨论帖回复，代码提交时间必须早于发起验收的时间；完成实现、测试和代码提交后再申请验收。

### 3.4 验收细节的确认方式

1. 浮点自测先采用上表的固定绝对上限。若具体用例需要使用 `32*ULP` 分支或归约放宽，应提交该用例并明确 ULP 的计算位置、输出类型及适用条件，再对齐验收判据。
2. 按多 Shape 回归要求覆盖任务书四种 Shape，固定布局接口和数学定义域的具体合法例外需以接口约束明确列出。
3. 使用 CANN 9.2 验证时，需明确 `reduce` 输入长度不一致和错误退出码问题的正式样例修正方式，并记录匹配的源码基线。
4. 在线清单中的 Tensor 源列表、数值极限静态成员及弃用地址接口均逐条处理，其具体扩展语法应保持原调用兼容。
