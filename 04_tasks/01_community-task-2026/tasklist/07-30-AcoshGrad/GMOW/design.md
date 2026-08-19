# AcoshGrad 算子设计说明

## 需求背景（required）

### 需求来源

- 通过西工大CANN暑期训练营社区任务完成 `AcoshGrad` 算子的 AscendC 改造与性能优化，保持与现有 TBE 算子语义一致。


- 当前参考实现为 `docs/AcoshGrad/acosh_grad.py` 中的 TBE DSL 动态算子，实现目标是在 AscendC 中复现 DSL 计算路径、动态 shape 行为和 dtype 处理策略，并通过手写 kernel 显式控制 UB 复用、分核和流水并行。

### 背景介绍

#### 算子功能

`AcoshGrad` 是 `Acosh` 的反向算子。输入 `y` 为前向 `acosh` 输出，`dy` 为上游梯度，输出 `z` 为前向输入的梯度。

- 输入：`y`、`dy`
- 输出：`z`
- 属性：无
- 计算规则：

$$
z = \frac{dy}{\sinh(y)}
$$

由于前向满足 `y = acosh(x)`，有：

$$
\frac{d\,acosh(x)}{dx} = \frac{1}{\sinh(acosh(x))}
$$

因此反向梯度可表示为：

$$
z_i = \frac{dy_i}{\sinh(y_i)}
$$

TBE 代码没有直接调用 `sinh` 或 `exp`，而是使用 Taylor 多项式和倍角公式近似计算 `sinh(y)`：

$$
s = \frac{y}{8}
$$

$$
p(s)=s\left(1+s^2\left(\frac{1}{3!}+s^2\left(\frac{1}{5!}+s^2\frac{1}{7!}\right)\right)\right)
$$

$$
r(v)=2v\sqrt{v^2+1}
$$

$$
\sinh(y) \approx r(r(r(p(y/8))))
$$

最终：

$$
z_i = \frac{dy_i}{r(r(r(p(y_i/8))))}
$$

#### AcoshGrad 算子实现优化

- TBE kernel  
  `/usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_impl/ai_core/tbe/impl/dynamic/acosh_grad.py`
- 算子原型  
  `/usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_proto/inc/`
- 算子信息库  
  `/usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_impl/ai_core/tbe/config/ascend910b`

当前 AscendC 工程建议路径：

```text
AcoshGrad/
```

#### AcoshGrad 算子现状分析

基于 `docs/AcoshGrad/acosh_grad.py` 可归纳：

- 参数与类型校验：
  - `y` 支持 `float16/float32/bfloat16`；
  - `dy` 支持 `float16/float32/bfloat16`；
  - `y` 与 `dy` dtype 必须一致；
  - TBE 文件注释约束 `y` 与 `dy` shape 必须一致，shape size 上限为 `2147483648`；
  - 输出 `z` dtype 和 shape 与输入保持一致。
- 入口实现：
  - TBE 入口为 `@register_operator("AcoshGrad")` 注册的 `acosh_grad` 函数；
  - 入口通过 `classify([y, dy], OpPatternMode.ELEWISE)` 做动态 shape elementwise 分类；
  - `shape_util.variable_shape([_y, _dy])` 生成动态 shape placeholder；
  - `acosh_grad_compute()` 构造 DSL 计算图；
  - `tbe.auto_schedule(res)` 自动调度；
  - `tbe.build(schedules, config)` 完成编译。
- 当前实现不是手写 TIK kernel，没有 `flowtable` tiling，也没有多模式 TIK 分支。它本身就是 DSL elementwise 主路径，不存在类似 `SplitV` 那种“长 TIK 文件但最终绕到 DSL API”的情况。
- dtype 处理：
  - `float16` 输入在平台支持 `te.lang.cce.vadd` 的 `float32` 实现时，会先 `cast_to(float32)`，最后再 cast 回 `float16`；
  - `float32` 输入直接按 `float32` 计算；
  - `bfloat16` 由 DSL 的 `support_bfp16=True` 路径支持，代码未显式转 `float32`，AscendC 中建议使用 `float32` 计算并按容差与 TBE 对齐。

TBE DSL 计算图可归纳为：

| 顺序 | DSL 操作 | 说明 |
| --- | --- | --- |
| 1 | `cast_to(float32)` | 仅 `float16` 且平台支持高精度路径时执行 |
| 2 | `vmuls(y, 0.125)` | 将输入缩小到 `y/8` |
| 3 | `vmul/vadds` | 计算 `s * (1 + s^2 * (1/3! + s^2 * (1/5! + s^2/7!)))` |
| 4 | `vsqrt(..., TBE_HIGH_PRECISION)` | 每次倍角中计算 `sqrt(v^2 + 1)` |
| 5 | 3 次倍角 | 连续执行 `v = 2 * v * sqrt(v^2 + 1)` |
| 6 | `vdiv(dy, sinh)` | 得到 `z = dy / sinh(y)` |
| 7 | `cast_to(float16)` | 仅原始输入为 `float16` 时执行 |

TBE/AscendC 总体流程图：

![acosh_grad_overall_flow.png](https://raw.gitcode.com/user-images/assets/10331120/dba917b7-9b48-4fd0-a0a4-d7c1255400d2/acosh_grad_overall_flow.png 'acosh_grad_overall_flow.png')

TBE 路径与 AscendC 策略映射：

| TBE 路径 | AscendC 策略 | 触发条件 | 处理策略 |
| --- | --- | --- | --- |
| DSL elementwise + `auto_schedule` | `PIPE_DOUBLE_BUFFER` | 默认路径，大中型 tensor | 展平成一维连续数组，多核切分，按 tile 执行 `CopyIn -> Compute -> CopyOut` |
| DSL elementwise + `auto_schedule` | `STATIC_SMALL_OR_LATENCY` | 小 shape，可作为第二阶段优化 | 静态 UB 分配，减少 `TPipe/TQue` 管理开销 |

当前 AscendC tiling key 规划：

| Tiling Key | 策略名 | 用途 |
| --- | --- | --- |
| 1000 | `PIPE_DOUBLE_BUFFER` | 默认实现，使用 `TPipe/TQue` 和双缓冲 |
| 1001 | `STATIC_SMALL_OR_LATENCY` | 小 shape 低延迟优化，可选实现 |

## 需求分析

### 外部组件依赖

- 无新增外部组件依赖。

### 内部适配模块

- 适配 AclNN 接口调用。
- 覆盖动态 shape/rank、`float16/float32/bfloat16`、非 32B 对齐尾块和特殊值场景。
- 复刻 TBE DSL 中 `y/8 + Taylor + 3 次 sqrt 倍角 + Div` 的计算路径。

### 需求模块设计

#### 算子原型

| 名称 | 类别 | dtype | format | shape | 说明 |
| --- | --- | --- | --- | --- | --- |
| y | 输入 | float16/float32/bfloat16 | ND | 任意 shape | 前向 `acosh` 的输出 |
| dy | 输入 | float16/float32/bfloat16 | ND | 与 `y` 相同 | 上游梯度 |
| z | 输出 | 与 `y` 相同 | ND | 与 `y` 相同 | 输出梯度 |

相关约束：

- `y` 与 `dy` dtype 必须一致。
- `y` 与 `dy` shape 必须一致。
- `z` shape 与 `y` 一致。
- `z` dtype 与 `y` 一致。
- 无属性输入。
- MVP 支持 `ND` format；其他同 shape 同 format 的连续 elementwise format 可作为扩展。
- 输入元素总数建议不超过 TBE 约束中的 `2147483648`。

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

代码位置：`AcoshGrad/op_host/`

Host 文件职责拆分：

| 文件 | 主要职责 | 关键点 |
| --- | --- | --- |
| `acosh_grad_def.cpp` | 算子注册与 AICore 配置 | 注册 2 输入 1 输出，无属性 |
| `acosh_grad_infershape.cpp` | 形状推导 | `z.shape = y.shape`，`z.dtype = y.dtype` |
| `acosh_grad_tiling.cpp` | tiling 生成 | 计算 totalLength、coreNum、tileLength、尾块和策略 key |
| `acosh_grad_tiling_data.h` | tiling 数据结构 | host/kernel 共享参数 |
| `acosh_grad_tiling_key.h` | tiling key 定义 | key 1000 为双缓冲，key 1001 为小 shape 静态分支 |

Host 核心执行链路：

1. 读取 `y/dy/z` shape、dtype、format。
2. 校验 `y` 与 `dy` shape、dtype、format 一致。
3. 展平 shape，计算 `totalLength`。
4. 根据 dtype 设置 `dtypeMode`：
   - `float32 -> 0`
   - `float16 -> 1`
   - `bfloat16 -> 2`
5. 默认设置 `computeMode=0`，表示内部使用 `float32` 计算。
6. 获取平台信息：
   - AIV core 数；
   - UB 大小。
7. 估算 UB 占用：
   - 输入队列：`y`、`dy`；
   - 输出队列：`z`；
   - 计算 buffer：`sinhWork`、`dyWork`、`tmpWork` 三个 `float32` tile。
8. 选择 `tileLength`：
   - 按 UB 可用容量计算；
   - 按 64 个 `float32` 元素向下对齐；
   - 不小于 32B 搬运粒度。
9. 多核切分：
   - 小 shape 减少 core 数；
   - 大 shape 使用 `min(AIVCoreNum, ceil(totalLength / 2048))`；
   - 每个 core 处理连续线性区间。
10. 设置 tiling key：
   - 默认 `1000`；
   - 若实现小 shape 静态分支且命中阈值，设置 `1001`。
11. 填充 `AcoshGradTilingData`，设置 `blockDim=coreNum`。

`AcoshGradTilingData` 关键字段：

| 字段 | 用途 |
| --- | --- |
| `totalLength` | 展平后的元素总数 |
| `coreNum` | 实际参与计算的 AIV 核数 |
| `blockLength` | 普通 core 处理的元素数 |
| `tailBlockLength` | 尾 core 处理的元素数 |
| `tileLength` | 单次 tile 处理元素数 |
| `tileNum` | 普通 core 的 tile 数 |
| `lastTileLength` | 普通 core 最后一个 tile 的真实长度 |
| `tailTileNum` | 尾 core 的 tile 数 |
| `tailLastTileLength` | 尾 core 最后一个 tile 的真实长度 |
| `dtypeMode` | 输入输出 dtype 类型 |
| `computeMode` | 计算精度模式，默认 float32 |
| `tilingKey` | 策略 key，`1000/1001` |

AscendC host 流程图：

![acosh_grad_host_tiling.png](https://raw.gitcode.com/user-images/assets/10331120/6fa13ab0-abd3-4807-bb0f-4136233308ee/acosh_grad_host_tiling.png 'acosh_grad_host_tiling.png')

#### AscendC kernel 侧设计

代码位置：`AcoshGrad/op_kernel/`

入口分发文件：`op_kernel/acosh_grad.cpp`。  
根据 tiling key 进入不同策略：

| Tiling Key | 策略 | 说明 |
| --- | --- | --- |
| 1000 | `KernelAcoshGradPipe<T>` | 默认双缓冲实现 |
| 1001 | `KernelAcoshGradStatic<T>` | 小 shape 低延迟优化，可选 |

Kernel 共性策略：

- 遵循 `Init -> Process` 生命周期。
- `Init` 阶段绑定 GM 地址、读取 tiling 参数、初始化 queue/buffer。
- `Process` 阶段按当前 core 的连续线性区间循环处理 tile。
- 每个 tile 执行 `CopyIn -> Compute -> CopyOut`。
- 计算阶段内部使用 `float32` buffer，输出前 cast 回原 dtype。
- 搬运使用 `DataCopyPad` 或真实 count/mask 处理非 32B 对齐尾块。

PIPE_DOUBLE_BUFFER 关键流程：

1. 初始化 `TPipe`。
2. 初始化输入队列：
   - `inQueueY`
   - `inQueueDy`
3. 初始化输出队列：
   - `outQueueZ`
4. 初始化计算 buffer：
   - `sinhWork`
   - `dyWork`
   - `tmpWork`
5. 循环处理每个 tile：
   - 搬入 `y` 和 `dy`；
   - 非 `float32` 输入 cast 到 `float32`；
   - 计算 `s = y * 0.125`；
   - 使用 Taylor 多项式得到 `sinh(s)` 近似；
   - 连续 3 次执行 `sinh = 2 * sinh * sqrt(sinh * sinh + 1)`；
   - 执行 `z = dy / sinh`；
   - cast 回输出 dtype；
   - 写回 `z`。

Compute 关键路径：

| 阶段 | AscendC 指令 | 说明 |
| --- | --- | --- |
| 输入转换 | `Cast` | `float16/bfloat16` 转 `float32` |
| 缩放 | `Muls` | `s = y * 0.125` |
| Taylor 多项式 | `Mul/Adds/Muls` | 复刻 TBE `_taylor_sinh_compute` |
| 倍角放大 | `Mul/Adds/Sqrt/Mul/Muls` | 连续执行 3 次，`Sqrt` 使用高精度路径 |
| 梯度计算 | `Div` | `z = dy / sinh(y)` |
| 输出转换 | `Cast` | `float32` 转回原 dtype |

AscendC kernel 分发流程图：

![acosh_grad_kernel_flow.png](https://raw.gitcode.com/user-images/assets/10331120/dce96d6d-ee2c-45ed-bb57-5c58bbf73b18/acosh_grad_kernel_flow.png 'acosh_grad_kernel_flow.png')

### 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas A2 训练系列产品 | √ |
| Atlas A3 系列产品 | √ |

### 算子约束限制

- 当前设计支持 `float16/float32/bfloat16`。
- MVP 支持 `ND` format。
- `y/dy/z` shape 必须一致。
- `y/dy/z` dtype 必须一致。
- 无 reduction、broadcast、索引和属性逻辑。
- `sinh(y)==0` 时输出按硬件 `Div` 行为产生 `inf/nan`，不额外 clamp。
- 默认内部使用 `float32` 计算；低精度 native compute 只能作为验证后的性能分支。

## 特性交叉分析、可维可测分析

### 精度标准 / 性能标准

| 验收标准 | 描述（不涉及说明原因） | 标准来源 |
| --- | --- | --- |
| 精度标准 | 与 TBE `acosh_grad.py` 的 DSL 计算路径保持一致 | 现有 TBE 实现 |
| 性能标准 | 关键大 shape 场景不低于原 TBE DSL 基线 | 社区任务要求 |

测试 golden 建议优先使用 TBE 等价公式：

```python
s = y * 0.125
x2 = s * s
sinh = s * (1 + x2 * (1/6 + x2 * (1/120 + x2 * (1/5040))))
for _ in range(3):
    sinh = 2 * sinh * np.sqrt(sinh * sinh + 1)
z = dy / sinh
```

建议覆盖：

| 场景 | 覆盖点 |
| --- | --- |
| 小 shape | `[1]`、`[7]`，验证尾块和低延迟 |
| 常规 shape | `[1024]`、`[16,1024]`，验证基本流水 |
| 大 shape | 大连续 tensor，验证多核和 double buffer |
| dtype | `float16/float32/bfloat16` |
| 数值范围 | `y` 接近 0、常规值、较大值、`inf/nan` |
| tail | 元素数不是 8/16/64 倍数 |

### 可测性分析

- 测试目录建议：`AcoshGrad/tests/ut/`
- host tiling 测试：`tests/ut/op_host/test_acosh_grad_tiling.cpp`
- kernel ICPU 测试：`tests/ut/op_kernel/test_acosh_grad.cpp`
- 数据生成：`tests/ut/op_kernel/acosh_grad_data/gen_data.py`
- 数据比较：`tests/ut/op_kernel/acosh_grad_data/compare_data.py`
- aclnn 样例：`AcoshGrad/examples/test_aclnn_acosh_grad.cpp`

重点验证项：

| 验证项 | 说明 |
| --- | --- |
| TBE 等价公式 | 优先对齐 Taylor + 3 次倍角路径 |
| dtype cast | `float16/bfloat16` 内部 `float32` 计算后回写 |
| tail 搬运 | 非 32B 对齐时不越界、不污染相邻数据 |
| small tensor | 判断是否需要启用 `STATIC_SMALL_OR_LATENCY` |
| profiling | 重点观察 Vector `Sqrt/Div`、MTE2/MTE3 和 Scalar 占比 |

### 兼容性分析

- 新算子 AscendC 实现，保持与现有 TBE `AcoshGrad` 输入输出语义一致；
- 不改变上层接口定义；
- 当前 TBE 为 DSL elementwise 主路径，AscendC 不需要复现复杂 TIK tiling 字段；
- AscendC 默认复刻 TBE 中的 Taylor + sqrt 倍角近似，不切换为 `exp` 公式；
- `bfloat16` 内部 `float32` 计算可能与 DSL bf16 低位舍入略有差异，需按 TBE 输出和任务容差验收。
