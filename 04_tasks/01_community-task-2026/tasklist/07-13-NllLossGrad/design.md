# NLLLossGrad 算子设计说明

## 需求背景（required）

### 需求来源

- 通过西工大CANN暑期训练营社区任务完成 `NLLLossGrad` 算子的 AscendC 改造与性能优化，保持与现有 TBE 算子语义一致,已通过cann judge测试。
- 任务要求适配 Atlas A2 训练系列产品/Atlas A3 系列产品，性能在所有核参与计算场景下不低于原 TBE 算子的 95%。

### 背景介绍

#### 算子功能

`NLLLossGrad` 是负对数似然损失 `NLLLoss` 的反向算子。输出 `x_grad` 与输入 `x` shape 一致，除目标类别位置外，其余位置为 0。

- 输入：`x`、`y_grad`、`target`、`weight`、`total_weight`
- 输出：`x_grad`
- 属性：`reduction`、`ignore_index`
- 支持模式：`reduction = none/sum/mean`

计算规则可表示为：

$$
\mathrm{valid}(i)=
\begin{cases}
1, & 0 \le target[i] < C \land target[i] \ne ignore\_index \\
0, & \text{otherwise}
\end{cases}
$$

$$
s(i)=
\begin{cases}
y\_grad[i], & reduction=\text{none} \\
y\_grad[0], & reduction=\text{sum} \\
\operatorname{div\_no\_nan}(y\_grad[0], total\_weight[0]), & reduction=\text{mean}
\end{cases}
$$

$$
x\_grad[i,j]=
\begin{cases}
-weight[j] \cdot s(i), & \mathrm{valid}(i)=1 \land j=target[i] \\
0, & \text{otherwise}
\end{cases}
$$

其中 `x` 为 `[C]` 时按 `N=1, C=x[0]` 处理；`x` 为 `[N,C]` 时按实际二维 shape 处理。

#### NLLLossGrad 算子实现优化

- TBE kernel  
  `/usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_impl/ai_core/tbe/impl/dynamic/nll_loss_grad.py`
- 算子原型  
  `/usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_proto/inc/`
- 算子信息库  
  `/usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_impl/ai_core/tbe/config/ascend910b`

当前 AscendC 工程路径：

```text
NllLossGrad/
```

#### NLLLossGrad 算子现状分析

基于 `docs/NllLossGrad/nll_loss_grad.py` 可归纳：

- 参数与类型校验：
  - `x` 支持 1D/2D；
  - `y_grad`、`target`、`weight`、`total_weight` 均为 1D；
  - `x/y_grad/weight/total_weight/x_grad` 支持 `float32/bfloat16`；
  - `target` 当前支持 `int32`；
  - `reduction` 支持 `"none"`、`"sum"`、`"mean"`。
- 入口实现：
  - TBE 入口为 `@register_operator("NLLLossGrad")` 注册的 `nll_loss_grad` 函数；
  - 入口只做 shape/dtype/reduction 校验，然后构造 `NllLossGradCompute`；
  - `NllLossGradCompute.nll_loss_compute_start()` 通过 `BuildCCE(..., flowtable=[tiling_gm])` 生成 TIK 动态 kernel；
  - 该实现没有 DSL API 绕路，核心逻辑全部在 TIK 类中完成。
- TBE 使用 `flowtable` 传递 tiling，控制参数定义为 `TILING_CTRL_PARAM=("int64", 128, 43)`：
  - `get_tiling_params()` 将 tiling GM 搬到 UB；
  - `init_tiling_params()` 读取前 43 个 int64 参数；
  - 运行时按 `key` 选择 `key=2000` 或 `key=2001` 两类路径。
- 当前 AscendC 工程中：
  - tiling key `0/1` 用于区分 kernel 模板 dtype；
  - `bigWeight=0/1` 用于区分 NormalWeight/BigWeight 策略。

TBE tiling 关键字段：

| 字段 | 含义 | 对 AscendC 的影响 |
| --- | --- | --- |
| `c_dim` / `n_dim` | 类别数 C、样本行数 N | 决定输出矩阵逻辑 shape 与分核粒度 |
| `ignore_index` | 忽略类别下标 | target 等于该值时输出保持 0 |
| `big_weight` / `key` | 普通 weight 或大 weight 路径标记 | 对应 AscendC 的 `bigWeight` 策略字段 |
| `core_num` / `loop_time` / `max_line` / `lower_line` | 多核行切分参数 | 控制每个 core 处理的样本行数和尾行 |
| `dup_ub_size` / `weight_ub_size` / `target_ub_size` / `refactor_weight_ub_size` | UB buffer 大小 | 决定是否整段搬入 `weight`，以及每次可处理的输出 tile |
| `weight_burst` / `target_burst` / `max_out_burst` / `last_out_burst` | GM/UB 搬运 burst 参数 | 影响 data_move 对齐和尾块处理 |
| `max_vmul_repeat` / `last_vmul_repeat` / `core_dup_repeat` / `last_dup_repeat` | 向量计算 repeat 参数 | 影响清零、乘法和 bf16 转换的循环次数 |
| `move_out_time` / `align_repeat_size` / `single_max_repeat` / `tail_repeat` | 大 weight 输出分段参数 | 用于 C 维过大时按列分段清零写回 |

TBE 运行时分发门控（与代码一致）：

| 顺序 | 触发门控 | 执行路径 |
| --- | --- | --- |
| 1 | `cycle < core_num && key == 2000` | `normal_two_dim_compute` |
| 2 | `cycle < core_num && key == 2001` | `two_dim_with_big_weight_compute` |

TBE 两类路径的应用场景与处理策略：

| TBE 路径 | 应用场景 | 处理策略 |
| --- | --- | --- |
| `key=2000` NormalWeight | `weight` 和单次输出 tile 能放入 UB，适合常规 C 维 | 先搬入完整 `weight`。`sum/mean` 模式在 UB 中预处理 `weight=-weight*y_grad[0]` 或 `weight=-weight*div_no_nan(y_grad[0],total_weight[0])`；`none` 模式按行读取 `y_grad[i]`。每个 core 先将输出 tile 清零，再按 `target[i]` 选出对应权重写到目标列，最后连续写回 GM。 |
| `key=2001` BigWeight | C 维过大，完整 `weight` 或完整输出行不适合放入 UB | 每个 core 按样本行处理。先按 `move_out_time` 将该行输出分段清零写回；再读取 `target[i]`，若合法且非 `ignore_index`，只从 GM 读取 `weight[target[i]]`，完成 `mean/sum/none` 缩放和取负，最后用尾块修正逻辑把单个有效梯度写回目标列。 |

TBE 需要继承的关键实现点：

- `target < 0`、`target >= C`、`target == ignore_index` 均不写有效梯度，输出保持 0。
- `mean` 模式使用 `div_no_nan` 语义；TBE 中优先使用 `vdiv`，否则退化为 `vrec + vmul`。
- `bfloat16` 输入在 UB 中转为 `float32` 计算，写回前再 `vconv round` 到 `bfloat16`。
- 优先使用 `data_move_pad` 处理非 32B 对齐搬运；不支持时通过尾块临时 UB 修正最后一个 block。
- NormalWeight 适合整 `weight` 复用，BigWeight 适合按 target 稀疏读取，AscendC tiling 需要按 UB 容量选择对应策略。

TBE/AscendC 总体流程图：

![nll_loss_grad_overall_flow.png](https://raw.gitcode.com/user-images/assets/10331120/b48be80a-40b4-48b0-8796-e87caae61a1c/nll_loss_grad_overall_flow.png 'nll_loss_grad_overall_flow.png')

TBE 两类路径与 AscendC 策略映射：

| TBE 路径 | AscendC 策略 | 触发条件 | 处理策略 |
| --- | --- | --- | --- |
| `key=2000` | `bigWeight=0`，NormalWeight | `weight` 和输出 tile 可放入 UB | 先搬入完整 `weight`，按多行 tile 清零输出并写 target 位置 |
| `key=2001` | `bigWeight=1`，BigWeight | `C` 很大，完整 `weight` 或输出行无法放入 UB | 逐行读取 `target`，按列 tile 清零输出，按需读取 `weight[target]` |

当前 AscendC tiling key 规划：

| Tiling Key | kernel 模板 | 用途 |
| --- | --- | --- |
| 0 | `NllLossGrad<bfloat16_t>` | bfloat16 输入输出 |
| 1 | `NllLossGrad<float>` | float32 输入输出 |

## 需求分析

### 外部组件依赖

- 无新增外部组件依赖。

### 内部适配模块

- 适配 AclNN 接口调用。
- 覆盖动态 shape 下 1D/2D 输入、不同 reduction、非法 target、ignore_index、bf16/float32 场景。
- 覆盖 NormalWeight 与 BigWeight 两类 tiling 策略。

### 需求模块设计

#### 算子原型

| 名称 | 类别 | dtype | format | shape | 说明 |
| --- | --- | --- | --- | --- | --- |
| x | 输入 | float32/bfloat16/float16 | ND | `[C]` 或 `[N,C]` | 前向输入，决定输出 shape |
| y_grad | 输入 | float32/bfloat16/float16 | ND | `[N]` 或 `[1]` | 上游梯度 |
| target | 输入 | int32/int64/uint8 | ND | `[N]` | 目标类别下标 |
| weight | 输入 | float32/bfloat16/float16 | ND | `[C]` | 类别权重 |
| total_weight | 输入 | float32/bfloat16/float16 | ND | `[1]` | `mean` 模式归一化权重 |
| x_grad | 输出 | float32/bfloat16/float16 | ND | 同 `x` | 输出梯度 |
| reduction | 属性 | string | - | 标量 | `"none"`、`"sum"`、`"mean"` |
| ignore_index | 属性 | int | - | 标量 | 默认 `-100` |

相关约束：

- `x` 仅支持 1D/2D。
- `x_grad` shape 与 `x` 一致。
- `weight` 长度必须等于 `C`。
- `target` 长度必须等于 `N`。
- `reduction="none"` 时 `y_grad` 长度为 `N`；`sum/mean` 时 `y_grad` 长度为 1。
- `target < 0`、`target >= C` 或 `target == ignore_index` 时，该行输出保持 0。

## 需求详细设计

### 使能方式

| 上层框架 | 涉及的框架勾选 |
| --- | --- |
| TF 训练/推理 |  |
| Pytorch 训练/推理 |  |
| ATC 推理 |  |
| Aclnn 直调 | √ |
| OPAT 调优 |  |
| SGAT 子图切分 |  |

### 需求总体设计

#### AscendC host 侧设计

代码位置：`NllLossGrad/op_host/`

Host 文件职责拆分：

| 文件 | 主要职责 | 关键点 |
| --- | --- | --- |
| `nll_loss_grad_def.cpp` | 算子注册与 AICore 配置 | 注册 5 输入 1 输出、`reduction/ignore_index` 属性、`ascend910b` 配置 |
| `nll_loss_grad_infershape.cpp` | 形状推导 | `x_grad.shape = x.shape`，输出 dtype 等于 `x` dtype |
| `nll_loss_grad_tiling.cpp` | tiling 生成 | 解析 `(N,C)`、dtype、reduction、ignore_index，计算策略和分核 |
| `nll_loss_grad_tiling_data.h` | tiling 数据结构 | host/kernel 共享参数 |
| `nll_loss_grad_tiling_key.h` | 模板 tiling key | key 0 为 bf16，key 1 为 float32 |

Host 核心执行链路：

1. 读取 `x` shape，归一化为 `(N,C)`。
2. 读取 dtype，判断是否为 bfloat16。
3. 解析 `reduction`：
   - `none -> 0`
   - `sum -> 1`
   - `mean -> 2`
4. 解析 `ignore_index`，默认 `-100`。
5. 获取平台信息：
   - AIV core 数；
   - UB 大小。
6. 按 `N` 维进行分核：
   - `validCore = min(AIVCoreNum, N)`；
   - 前 `tailLine` 个 core 处理 `baseLine + 1` 行；
   - 其余 core 处理 `baseLine` 行。
7. 根据 UB 判断策略：
   - UB 可容纳完整 `weight` 和输出 tile 时，设置 `bigWeight=0`；
   - 否则设置 `bigWeight=1`，按列方向切分输出。
8. 填充 `NllLossGradTilingData`，设置 `blockDim` 和模板 tiling key。

`NllLossGradTilingData` 关键字段：

| 字段 | 用途 |
| --- | --- |
| `nDim`、`cDim` | 样本数与类别数 |
| `coreNum` | 实际参与计算的核数 |
| `reduction` | kernel 侧选择 none/sum/mean 路径 |
| `ignoreIndex` | ignore_index 判断 |
| `bigWeight` | 0 表示 NormalWeight，1 表示 BigWeight |
| `maxLine`、`lowerLine`、`redundantLine` | 多核行切分参数 |
| `lineTile` | NormalWeight 单次处理行数 |
| `cAlign` | `C` 按 8 对齐后的长度 |
| `outUbSize` | NormalWeight 输出 UB 大小 |
| `colTile`、`moveOutTime` | BigWeight 列方向切分参数 |

AscendC host 流程图：
![nll_loss_grad_host_tiling.png](https://raw.gitcode.com/user-images/assets/10331120/b8c3036e-d0d4-4b6c-a7d5-8728268bb8c5/nll_loss_grad_host_tiling.png 'nll_loss_grad_host_tiling.png')

#### AscendC kernel 侧设计

代码位置：`NllLossGrad/op_kernel/`

入口分发文件：`op_kernel/nll_loss_grad.cpp`。  
根据模板 tiling key 实例化不同 dtype kernel：

| Tiling Key | 实例化类型 | 说明 |
| --- | --- | --- |
| 0 | `NllLossGrad<bfloat16_t/float16>` | 16bit浮点数输入输出，内部转 float 计算 |
| 1 | `NllLossGrad<float>` | float32 输入输出 |

Kernel 共性策略：

- 遵循 `Init -> Process` 生命周期。
- `Init` 阶段绑定 GM 地址、读取 tiling 参数、初始化 UB buffer。
- `Process` 阶段根据 `bigWeight` 进入 `ProcessNormal` 或 `ProcessBigWeight`。
- 输出先清零，再写 target 位置梯度。
- 搬运统一使用 `DataCopyPad` 处理非 32B 对齐尾块。

各策略的 kernel 处理方式：

| 策略 | 触发字段 | 应用场景 | 处理策略 |
| --- | --- | --- | --- |
| NormalWeight | `bigWeight=0` | `weight` 可放入 UB，中小 `C` 场景 | 搬入完整 `weight`，按 `lineTile` 处理多行；UB 中清零 dense 输出，再写 target 位置 |
| BigWeight | `bigWeight=1` | `C` 很大，完整 `weight` 放不进 UB | 逐行读取 target，按列 tile 清零输出；有效 target 时只读取 `weight[target]` |

NormalWeight 关键流程：

1. 搬入 `weight[0:C]`，bf16 场景 cast 到 float。
2. `sum/mean` 场景预计算 scale，并将 `weight` 预乘 `-scale`。
3. 循环处理 target tile：
   - 搬入 target；
   - `none` 场景搬入 y_grad tile；
   - 清零输出 tile；
   - 逐行校验 target 并写入 `out[row*C + target]`；
   - bf16 场景 cast 回 bfloat16；
   - `DataCopyPad` 写回 `x_grad`。

BigWeight 关键流程：

1. `sum/mean` 场景每 core 计算一次 scale。
2. 逐行读取 target，判断是否合法。
3. target 有效时读取 `weight[target]`，并按 reduction 计算梯度。
4. 按 `colTile` 处理输出行：
   - 清零当前列 tile；
   - 如果 target 落在当前列 tile，写入梯度；
   - bf16 / half 场景 cast 回 bfloat16/ half；
   - `DataCopyPad` 写回输出。

AscendC kernel 分发流程图：

![nll_loss_grad_kernel_flow.png](https://raw.gitcode.com/user-images/assets/10331120/aa910808-66bf-46cd-b3c2-a59040835007/nll_loss_grad_kernel_flow.png 'nll_loss_grad_kernel_flow.png')

### 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas A2 训练系列产品 | √ |
| Atlas A3 系列产品 | √ |

### 算子约束限制

- 当前工程支持 `float32/bfloat16/half`。
- 当前工程支持 `target=int32/int64/uint8`。
- 当前工程支持 `ND` format。
- `reduction` 仅支持 `"none"`、`"sum"`、`"mean"`。
- BigWeight 策略用于大 `C` 场景，性能主要受 dense 输出写回和 `weight[target]` 随机读影响。

## 特性交叉分析、可维可测分析

### 精度标准 / 性能标准

| 验收标准 | 描述（不涉及说明原因） | 标准来源 |
| --- | --- | --- |
| 精度标准 | 满足 CANN Judge 默认阈值，采用逻辑值比较 | 任务开发书 |
| 性能标准 | 所有核参与计算场景不低于原 TBE 算子 95% | 任务开发书 |

### 可测性分析

- 测试目录：`NllLossGrad/tests/ut/`
- host tiling 测试：`tests/ut/op_host/test_nll_loss_grad_tiling.cpp`
- kernel ICPU 测试：`tests/ut/op_kernel/test_nll_loss_grad.cpp`
- 数据生成：`tests/ut/op_kernel/nll_loss_grad_data/gen_data.py`
- 数据比较：`tests/ut/op_kernel/nll_loss_grad_data/compare_data.py`
- aclnn 样例：`NllLossGrad/examples/test_aclnn_nll_loss_grad.cpp`

建议覆盖场景：

| 场景 | 覆盖点 |
| --- | --- |
| `[C]` | 1D 输入 |
| `[N,C]` | 2D 输入 |
| `reduction=none` | 每行读取 `y_grad[i]` |
| `reduction=sum/mean` | 标量 y_grad 与 total_weight |
| `target` 含 ignore/非法值 | 输出保持 0 |
| `C` 非 8/16 对齐 | 尾块与 `DataCopyPad` |
| 大 `C` | BigWeight 路径 |
| bfloat16 | cast 与精度 |

### 兼容性分析

- 新算子 AscendC 实现，保持与现有 TBE `NLLLossGrad` 输入输出语义一致；
- 不改变上层接口定义；

