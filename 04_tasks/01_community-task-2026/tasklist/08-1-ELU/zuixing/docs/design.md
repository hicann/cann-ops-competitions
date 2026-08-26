# ELU 接口设计方案

# 一、需求描述

## 1.1 需求来源

CANN 社区任务「8月社区任务 - ELU 算子开发」。要求基于 Ascend C **C API**，在纯 **Vector-Core** 上实现 ELU 样例算子，提交至 `asc-devkit`：

`examples/02_simd_c_api/03_c_api/02_reg_vector_compute/elu`

适配硬件：Ascend 950PR / Ascend 950DT（`dav-3510`）。  
CANN 版本：9.0.0 ~ 9.1.0。  
任务书：https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/docs/202608/asc_elu_task_doc.md

## 1.2 需求分析

ELU（Exponential Linear Unit）对正半轴保持线性，对负半轴引入指数项，用于缓解 Dead ReLU，并促使输出均值更接近 0。对输入特征 \(x\)，ELU 计算过程可表示为：

\[
ELU(x)=
\begin{cases}
scale\cdot x, & x>0 \\
\alpha\cdot scale\cdot\big(e^{x\cdot inputScale}-1\big), & x\le 0
\end{cases}
\]

本任务需求拆解如下：

1. 使用 Ascend C C API（Reg Vector）实现 Device 侧 `elu_custom` 核函数；
2. 全程禁止调用 Cube-Core；禁止 Host 侧 `for` 逐元素计算；
3. 支持数据类型：`float`、`float16`；
4. 功能与官方算子库 `elu` 语义对齐（参数含 `alpha`、`scale`、`inputScale`）；
5. 精度满足生态算子开源精度标准（混合容差）；
6. 验收覆盖 shape：`1 / 32 / 1024 / 2048`，值域：`[-100, 100]`。

# 二、方案设计

## 2.1 接口内部实现

ELU 为逐元素一元激活。核函数接收 GM 上的输入 `x`、输出 `y` 以及标量参数 `alpha/scale/input_scale`。实现流程：

1. **CopyIn**：将当前核负责的数据块从 GM 搬运至 UB（`asc_copy_gm2ub_align`）；
2. **Compute**：在寄存器向量域完成分段计算：
   - `pos = scale * x`（`asc_mul_scalar`）
   - `t = x * inputScale`（`asc_mul_scalar`）
   - `e = exp(t)`（`asc_exp`）
   - `neg = alpha * scale * (e - 1)`（`asc_add_scalar` + `asc_mul_scalar`）
   - `mask = (x > 0)`（`asc_gt_scalar`）
   - `y = select(pos, neg, mask)`（`asc_select`）
3. **CopyOut**：将结果从 UB 写回 GM（`asc_copy_ub2gm_align`）；
4. 使用 `asc_sync_notify/wait` 保证 MTE2 → V → MTE3 流水依赖正确。

图1 接口计算流程图

```text
Host读入bin/申请Device内存
        ↓
     H2D(src)
        ↓
  elu_custom<<<numBlocks>>>
        ↓
  GM→UB → elu_vf(寄存器向量) → UB→GM
        ↓
     D2H + 与golden精度对比
```

所有计算均在 Vector / 搬运 / 同步 API 上完成，不调用任何 Cube 接口。  
`float16` 采用 half 原生路径（`vector_half` + `asc_update_mask_b16`）；尾块通过 mask 屏蔽无效 lane。  
当 `n_burst=1` 时，使用 dav-3510 高维切分搬运接口完成连续数据搬移。

图2 函数实现流程图（正/负半轴分支选择）

```text
loadalign(x)
   ├─ mul_scalar → pos = scale * x
   └─ mul_scalar → t = x * inputScale
                 → exp(t)
                 → add_scalar(-1)
                 → mul_scalar(alpha*scale) → neg
gt_scalar(x, 0) → mask
select(pos, neg, mask) → y
storealign(y)
```

边界说明：`x=0` 落入负分支，公式结果为 0，与正分支在有限精度下连续。

## 2.2 接口设计

### Kernel侧接口

本任务交付形态为 **C API `<<<>>>` 直调样例**（与 `abs` 一致），核函数签名如下。

float 路径：

```cpp
__vector__ __global__ void elu_custom(
    __gm__ float* x, __gm__ float* y,
    float alpha, float scale, float input_scale);
```

float16 路径（编译宏 `ELU_DTYPE_FP16=1`）：

```cpp
__vector__ __global__ void elu_custom(
    __gm__ half* x, __gm__ half* y,
    float alpha, float scale, float input_scale);
```

表1 接口参数说明

| 参数名 | 输入/输出 | 描述 |
|--------|-----------|------|
| x | 输入 | 源操作数，位于 Global Memory。数据类型为 `float` 或 `half`（由编译路径决定）。 |
| y | 输出 | 目的操作数，位于 Global Memory，shape/dtype 与 `x` 一致。 |
| alpha | 输入 | 激活系数 α，标量，Host 以 `float` 下发。 |
| scale | 输入 | 缩放系数，标量，Host 以 `float` 下发。 |
| input_scale | 输入 | 输入缩放系数 inputScale，标量，Host 以 `float` 下发。 |

Kernel接口约束说明

1. 禁止调用 Cube-Core；仅允许 Vector / 数据搬运 / 同步 API。
2. 禁止在 Host 侧对 ELU 做逐元素串行计算；Host 仅负责内存申请、数据搬运与 Kernel launch。
3. `y` 与 `x` 的元素个数、dtype 必须一致。
4. UB 分配大小需覆盖单核 `block_length` 对应字节数；目的 UB 地址需满足 32 字节对齐要求。
5. 样例默认 `NUM_BLOCKS=1`；后续可按 `totalLength` 多核均分扩展。
6. 非 32B 对齐尾块依赖向量 mask，不得越界写回有效输出区。

### Host侧接口

Host 侧不提供高阶 API 的 Tiling 查询接口（本交付不是 aclnn 三件套，也不是 LocalTensor 高阶 API），调用流程如下：

1. `aclInit` / `aclrtSetDevice`
2. 申请 Device 内存，`aclrtMemcpy` 完成 H2D
3. 启动核函数：

```cpp
elu_custom<<<num_blocks, 0>>>(x_device, y_device, alpha, scale, input_scale);
```

4. 同步后 D2H，与 Python golden 按混合容差比对

表2 Host侧关键参数说明

| 参数名 | 输入/输出 | 描述 |
|--------|-----------|------|
| num_blocks | 输入 | 启动核数；样例默认 1。 |
| x_device / y_device | 输入/输出 | Device 侧 GM 指针。 |
| alpha / scale / input_scale | 输入 | 与 Kernel 一致的标量超参。 |
| TOTAL_LENGTH | 编译期输入 | 本样例处理的一维元素个数（任务书覆盖 1/32/1024/2048）。 |
| ELU_DTYPE_FP16 | 编译期输入 | `0`=float 路径，`1`=float16 路径。 |

## 2.3 测试用例设计

精度判定采用生态算子开源混合容差标准：  
\(|actual-golden| \le atol + rtol\times|golden|\)，且 `matched_ratio ≥ 0.99`，`max_abs_error` 不超过 dtype 上限。

| 用例编号 | 测试项 | 测试前端表达 |
|----------|--------|--------------|
| P-F32-1 | float32，最小 shape | dtype=float32，length=1，value=uniform(-100,100)，alpha/scale/inputScale=1 |
| P-F32-32 | float32，小 shape | dtype=float32，length=32，value=uniform(-100,100)，含 0/+eps/-eps/-1/1 边界点 |
| P-F32-1024 | float32，中等 shape | dtype=float32，length=1024，value=uniform(-100,100) |
| P-F32-2048 | float32，任务书最大 shape | dtype=float32，length=2048，value=uniform(-100,100) |
| P-F16-1 | float16，最小 shape | dtype=float16，length=1，value=uniform(-100,100) |
| P-F16-32 | float16，小 shape | dtype=float16，length=32，value=uniform(-100,100) |
| P-F16-1024 | float16，中等 shape | dtype=float16，length=1024，value=uniform(-100,100) |
| P-F16-2048 | float16，任务书最大 shape | dtype=float16，length=2048，value=uniform(-100,100) |
| F-BRANCH | 分支正确性 | 强制写入 x∈{0,+0,-0,-1,1,low}，校验正负分支选择 |
| F-ATTR | 非默认超参 | alpha=0.5，scale=2.0，inputScale=0.1，与公式一致 |
| F-MASK | 非对齐尾块 | length=33/100，校验 mask 尾块正确、无越界写 |

自测脚本：

- `scripts/gen_data.py`：生成 input / golden（将 abs 样例中的 `np.abs` 替换为 ELU 参考实现）
- `./demo`：Device 侧执行 `elu_custom`
- `scripts/verify_precision.py`：按混合容差离线复核
- `scripts/run_all_tests.sh`：串联上述流程，覆盖 8 组主验收用例

真机自测环境与结论（2026-08-07）：

| 项 | 结果 |
|----|------|
| 硬件 | Ascend950PR_9579 |
| CANN | 9.1.0-beta.3 |
| 架构 | dav-3510 |
| 命令 | `bash scripts/run_all_tests.sh` |
| 汇总 | TOTAL pass=8 fail=0 |

# 三、可维可测

## 3.1 精度标准/性能标准

| 验收标准 | 描述(不涉及说明原因) | 标准来源 |
|----------|----------------------|----------|
| 精度标准 | 采用生态算子开源混合容差：float32 使用 rtol=9.77e-4、atol=1.53e-5、matched_ratio≥0.99、max_abs_error_limit=1e-2；float16 使用 rtol=1.95e-3、atol=1.95e-3、matched_ratio≥0.99、max_abs_error_limit=1e-1。真机 8 组主验收用例全部通过（matched_ratio=1）。 | [生态算子开源精度标准](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md#2-误差指标与通过标准)；任务书要求 |
| 性能标准 | 任务书明确性能要求为“无”，本样例不设立强制性能验收门槛。 | 任务书 |

## 3.2 兼容性分析

`elu` 为 `asc-devkit/examples/02_simd_c_api/03_c_api/02_reg_vector_compute/` 目录下新增 C API 样例条目，不修改任何既有接口行为，不涉及存量算子兼容性/回归问题。产品支持范围：Ascend 950PR / Ascend 950DT（`dav-3510`），CANN 9.0.0 ~ 9.1.0。
