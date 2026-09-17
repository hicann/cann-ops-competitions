# MaxPool2dWithMask 算子设计文档

本文用于对齐算子原型、接口定义、精度与性能验收指标。第 6 节列出需要评审确认的契约边界。

## 1. 需求背景（required）

### 1.1 需求来源

本任务来自 2026 年 9 月 CANN 社区任务，要求在 Atlas 800T A2 和 Atlas 300V Pro 上使用 Ascend C 实现与 `aclnnMaxPool2dWithMask` 功能一致的算子，并在 Atlas 800T A2 上达到不低于原 TBE 实现 95% 的整体性能。

设计依据包括任务书、任务附件中的算子原型、153 条自测用例和 Golden，以及 `ops-nn` 基线提交 `33d73923c46a8149c57f62fc5cbbe9ae23a79bfa` 的 `aclnnMaxPool2dWithMask` 接口说明与源码。适配环境为 CANN 9.0.0 或 9.1.0。任务附件是本次验收输入输出的直接依据；历史 TBE 行为与附件不一致的边界在本文中单独列出。

### 1.2 算子功能

`MaxPool2dWithMask` 对输入的每个通道独立执行二维最大池化，输出最大值 `out` 和最大值位置 `indices`。输入支持三维 `[C,H,W]` 与四维 `[N,C,H,W]`，三维输入在计算时等价为 `N=1`。

对输出位置 `(n,c,oh,ow)`，候选输入坐标为：

$$
ih=oh\times s_H-p_H+kh\times d_H,
\qquad
iw=ow\times s_W-p_W+kw\times d_W
$$

其中 `0<=kh<kH`、`0<=kw<kW`。越界候选按负无穷处理。按 `(kh,kw)` 行优先顺序扫描，并且只在候选值严格大于当前最大值时更新，因此重复最大值选择第一个位置。

### 1.3 现有实现与本任务的关系

历史 TBE `MaxPoolWithArgmaxV1` 输出按窗口位置组织的位掩码。当前 A2 的 `aclnnMaxPool2dWithMask` 对非 `1x1` 窗口使用另一条路径：把二维输入扩为深度为 1 的五维输入，调用 `MaxPool3DWithArgmaxV2Ncdhw` 生成 `INT32` 绝对索引，再把结果直接写入用户传入的 `INT8 indices` 存储。任务 Golden 与后一路径一致。

因此，本任务提供的 153 条用例采用以下索引契约：

- 索引值是当前通道输入平面内的绝对偏移 `ih*W+iw`；
- 每个索引按 little-endian `int32` 表示；
- 所有索引按 `N,C,Hout,Wout` 顺序连续写入整个 `INT8 indices` 容器的起始区域；
- 容器中未承载有效索引的剩余字节写 0。

任务附件未覆盖 `1x1` 窗口。该窗口在部分大输出下无法用任务规定的容器形状容纳每个输出一个 `int32`，而当前 ACLNN 也会退回历史位掩码路径。本设计在任务方明确契约前不把这类场景错误地归入连续 `INT32` 字节流路径。

## 2. 需求分析（required）

### 2.1 外部 ACLNN 接口

```cpp
aclnnStatus aclnnMaxPool2dWithMaskGetWorkspaceSize(
    const aclTensor *self,
    const aclIntArray *kernelSize,
    const aclIntArray *stride,
    const aclIntArray *padding,
    const aclIntArray *dilation,
    bool ceilMode,
    aclTensor *out,
    aclTensor *indices,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

aclnnStatus aclnnMaxPool2dWithMask(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);
```

实验算子通过构建系统的 `ACLNNTYPE aclnn` 配置自动生成上述两个 ACLNN 符号，生成代码直接调度内部 `MaxPool2dWithMask`，并对非连续输入调用 `Contiguous`。本次不修改稳定目录中现有的同名 ACLNN 源码；正式集成时再按维护者意见处理同名接口的路由与归档位置，外部函数原型保持不变。

### 2.2 内部 Ascend C 算子原型

内部算子名、输入输出和属性名与任务附件 `max_pool2d_with_mask_op.json` 一致：

```text
MaxPool2dWithMask(
    x,
    kernelSize,
    stride,
    padding,
    dilation,
    ceilMode=false
) -> (out, indices)
```

| 名称 | 类别 | 数据类型 | 形状或取值 | 说明 |
| --- | --- | --- | --- | --- |
| `x` | 输入 | FLOAT16、FLOAT；A2 额外支持 BFLOAT16 | `[C,H,W]` 或 `[N,C,H,W]` | ND/NCHW 逻辑布局 |
| `kernelSize` | 属性 | LIST_INT | 长度 1 或 2，值大于 0 | 长度 1 时 H/W 相同 |
| `stride` | 属性 | LIST_INT | 长度 0、1 或 2，非空值大于 0 | 空数组时等于 `kernelSize` |
| `padding` | 属性 | LIST_INT | 长度 1 或 2，`0<=p<=floor(k/2)` | H/W 两侧对称填充 |
| `dilation` | 属性 | LIST_INT | 长度 1 或 2，当前只支持 1 | 窗口内采样间隔 |
| `ceilMode` | 属性 | BOOL | `false` 或 `true` | 输出尺寸取整方式 |
| `out` | 输出 | 与 `x` 相同 | `[C,Ho,Wo]` 或 `[N,C,Ho,Wo]` | 最大值输出 |
| `indices` | 输出 | INT8 | 见 2.3 节 | 索引字节容器 |

当前 ACLNN 源码对 310P 仅登记 FLOAT，而任务书仅排除 310P 的 BFLOAT16。本文按任务书准备 310P 的 FLOAT16/FLOAT 实现，但在正式声明兼容性前以任务方答复和实机编译结果为准。

### 2.3 输出形状

令有效窗口大小为：

$$
kH_{eff}=d_H(k_H-1)+1,
\qquad
kW_{eff}=d_W(k_W-1)+1
$$

`ceilMode=false` 时：

$$
H_{out}=\left\lfloor\frac{H+2p_H-kH_{eff}}{s_H}\right\rfloor+1,
\qquad
W_{out}=\left\lfloor\frac{W+2p_W-kW_{eff}}{s_W}\right\rfloor+1
$$

`ceilMode=true` 时把上述除法改为向上取整，并执行末窗修正：

```text
if ((Hout - 1) * sH >= H + pH) Hout--;
if ((Wout - 1) * sW >= W + pW) Wout--;
```

`indices` 的形状为：

```text
maskW = (ceil(Hout * Wout / 16) + 1) * 2 * 16
4D: [N, C, kH*kW, maskW]
3D: [C, kH*kW, maskW]
```

Host 侧使用 64 位无符号整数计算所有形状、元素数和字节数，并在乘法前检查溢出。连续 `INT32` 索引路径还需校验：

$$
4\times N\times C\times H_{out}\times W_{out}
\le
N\times C\times k_H\times k_W\times maskW
$$

### 2.4 功能与验收范围

任务附件共有 153 条用例，覆盖三维/四维输入、FLOAT16/FLOAT/BFLOAT16，以及 `2x2/s2/p0`、`3x3/s2/p1`、`3x3/s1/p1` 和 `7x7/s2/p3`。全部用例的 `ceilMode=false`、`dilation=[1,1]`，并且 `indices` 容器均能容纳连续 `INT32` 索引。

泛化实现还必须补充覆盖：`ceilMode=true`、长度为 1 的属性、空 `stride`、非方形属性、非 32 字节对齐宽度、重复最大值、三维输入、非连续输入与输出，以及非法参数。

## 3. 详细设计（required）

### 3.1 实现范围

Ascend C 实现位于 `experimental/pooling/max_pool2d_with_mask/`，包括 Host 注册、形状推导、
Tiling、Kernel、接口示例和测试。外部函数签名、数据类型、形状与第 2 节保持一致。

### 3.2 Host 侧设计

Host 校验 rank、类型、格式、属性及输出形状，将空间属性归一化为 H/W 二元组，计算输出与索引容器
大小，并检查整数溢出和容器容量。三维输入按内部 N=1 处理。根据实际 SoC、核数及 UB 容量进行
分核和分块，每个输出由一个核负责。核间写入区间还须满足平台缓存一致性要求，不能仅以逻辑字节
区间不重叠判定安全。索引容量不足的 1x1 场景属于第 6 节待对齐的契约问题。

### 3.3 Kernel 侧设计

Kernel 分块加载合法输入，对窗口内候选求稳定最大值，并同时保留通道内原始平面索引。相等时选择
行优先首个位置，FLOAT32 不因中间表示而损失比较精度。最大值写回应保持对应输入值的原始位模式。
A2 可复用现有 `MaxPool3DWithArgmaxV2` 的向量化归约能力，将深度固定为 1；310P 根据该平台支持的
搬运与比较指令实现同一语义。具体分解、局部布局和流水不改变公共接口或验收指标。

有效索引字节数为 `4*N*C*Ho*Wo`。有效区写入小端 INT32 argmax，其余容器字节显式清零，不能依赖
显存初值；写回、清零和未对齐边界处理必须保证无越界及无核间缓存行覆盖。

### 3.4 非连续 Tensor

内部 Kernel 处理连续张量。ACLNN 层应对非连续输入执行 `Contiguous`，并在需要时将连续计算结果通过 `ViewCopy` 写回非连续 `out`。非连续输入和输出分别通过真机用例验证，不能仅凭接口生成就认定全部支持；`indices` 按接口约束不支持非连续张量。

### 3.5 支持硬件

| 平台 | FLOAT16 | FLOAT | BFLOAT16 | 状态 |
| --- | :---: | :---: | :---: | --- |
| Atlas 800T A2 | √ | √ | √ | 任务明确要求 |
| Atlas 300V Pro | √ | √ | × | 按任务书准备 FLOAT16/FLOAT，现行接口注册差异见第 6 节 |

同一数学实现按平台实际核数、UB 容量和 API 支持范围重新切分。A2 与 310P 分别编译、安装和测试，不以 A2 构建成功替代 310P 兼容性证据。

## 4. 测试设计

### 4.1 功能与精度

- A2 全量执行任务附件的 153 条用例；310P 按 FLOAT16/FLOAT 范围执行 103 条，BFLOAT16 的 50 条仅在 A2 验证；
- `out` 与 CPU Golden 比较，设计目标是逐元素精确一致；正式判定同时满足生态算子开源精度标准；
- 把 `indices` 有效头部按 little-endian `int32` 解码，与 CPU argmax 逐元素精确比较，并检查全部尾部字节为 0；
- 构造重复最大值，验证行优先首个索引；
- 补充 `ceilMode=true`、空 `stride`、长度 1 属性、非方形窗口、非对齐宽度、三维输入和非连续 Tensor；
- 补充非法 rank、dtype、属性长度和值、输出形状不匹配及整数溢出用例；
- 同一输入重复执行，验证 `out` 和 `indices` bit-wise 一致。

### 4.2 性能

在同一设备、CANN 版本、输入、stream、预热次数和采样方法下比较 Ascend C 与现有 TBE：

- 使用 CANN 官方 `msopprof` 采集性能数据；
- 同时报告 ACLNN 端到端耗时和 Kernel 耗时，避免把格式转换收益错误归因于 Kernel；
- 大 shape 重点分析核利用率、GM 搬运、UB 利用率和 Vector/MTE 流水；
- 小于 100 微秒且相对 TBE 慢 30% 以上的场景提供仿真图和分析；
- A2 的正式目标为整体性能不低于 TBE 的 95%；统计聚合口径在任务方明确后固化；
- 310P 以 A2(910B3) 耗时的 5 倍为参考，超过该参考 30% 的场景提供性能仿真图。

## 5. 可维可测分析

### 5.1 精度与性能标准

任务附件的 `out` 误差阈值统一为 `[0.001,0.001]`，`indices` 为 `[0,0]`。正式数值判定引用
[生态算子开源精度标准](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md)，
本次核对的构建依赖版本为 `opbase@0c5e25790f67e2c6f60c1df36d74d28f0e08ecfc`。
该版本逐元素判定为 `abs(actual-golden) <= atol + rtol*abs(golden)`，通过率不低于 0.99，且最大绝对误差
满足硬上限。FLOAT16 的 rtol/atol 均为 2^-9，BFLOAT16 均为 2^-6，FLOAT32 分别为 2^-10/2^-16；
硬上限依次为 `0.1 or 32 ULP`、`1 or 32 ULP`、`0.01 or 32 ULP`。正式验收应确认采用的版本和
ULP 检查脚本，避免混用不同容差口径。开发自测另外比较 `out` 与整个 `indices`
的原始字节，保持 MaxPool 复制输入值的语义；逐字节要求是实现自测目标，不替代任务引用的正式标准。

A2 的 95% 若按吞吐比解释，等价于 `T_TBE/T_AscendC >= 0.95`，不是简单规定耗时至多增加 5%。
310P 的补充材料触发条件为 `T_310P > 1.3*5*T_A2`。不同平台必须使用相同 shape、类型和语义，
分别提供实际时间，不能将参考线直接宣称为功能精度的失败判据。


| 验收项 | 标准 |
| --- | --- |
| `out` 精度 | 满足生态算子开源精度标准；实现目标为最大值与输入逐元素精确复制 |
| `indices` 精度 | 有效 `INT32` argmax 与容器尾部逐字节一致 |
| 确定性 | 同一输入重复执行和不同合法切分下均 bit-wise 一致 |
| A2 性能 | 不低于原 TBE 整体性能的 95% |
| 310P 性能 | 功能与精度必须通过；性能按任务书参考线提供数据和必要分析 |

### 5.2 兼容性分析

- 外部 ACLNN 两段式函数原型不变；
- 内部算子名、属性名、输出形状和任务附件一致；
- 三维输入在内部统一为 `N=1`，不会改变用户可见维数；
- 新实现不依赖固定核数或固定 UB 容量；
- 任务附件范围采用连续 `INT32` 索引字节流；310P 与 `1x1` 的历史位掩码兼容边界需要任务方评审确认；
- 实验构建自动生成公共 ACLNN 符号；验收通过后再按 `ops-nn` 维护者意见处理稳定接口与实验实现的最终路由，避免形成两套长期维护的同名实现。

## 6. 待确认项

1. 310P 是否与 A2 一样按任务 Golden 的连续 `INT32` 字节流验收；
2. `1x1` 窗口在容器容量不足时采用何种索引契约；
3. 310P 是否需要支持 FLOAT16；
4. A2 的 95% 性能指标按逐 case 还是汇总值判定，以及附件中哪些 case 属于性能集合；
5. TBE 对照应使用的 CANN 版本、实际算子入口、计时范围，以及 100 us 小 shape 的划分依据；
6. FLOAT32 是否以保持原始类型的编译配置验收，以及通用精度标准采用的版本和 ULP 检查规则；
7. 泛化 ceilMode 是否遵守 Golden 的末窗修正及任务书所列推理平台 stride 限制。

上述问题不阻塞附件 153 条用例的 A2 实现，但会影响泛化边界、310P 注册和最终兼容性声明。
