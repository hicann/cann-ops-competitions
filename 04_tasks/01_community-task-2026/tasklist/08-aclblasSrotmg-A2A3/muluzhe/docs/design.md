# aclblasSrotmg A2/A3 算子设计文档

## 需求背景

### 需求来源

本设计对应昇腾 CANN 社区任务 `aclblasSrotmg 算子开发（A2A3）`，目标是为 Atlas A2/A3（A2 使用 Ascend 910B，A3 使用 Ascend 910C）补齐 `ops-blas` 中 `aclblasSrotmg` 的 `arch22` 实现。实现基于 CANN 9.1.0、Ascend C 和 `ops-blas` 现有 BLAS handle/stream 框架。

参考入口：活动页（[11014a50a8794171a4a08688fd398774](https://www.hiascend.com/developer/activities/details/11014a50a8794171a4a08688fd398774#tab0)）、官方算子仓（[cann/ops-blas](https://gitcode.com/cann/ops-blas)）、精度标准（[opbase experimental_standard.md](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md)）。本设计文档 PR 只提交设计和交付边界，真机精度、性能结果在后续 `ops-blas` 代码 PR 和自测报告中提供。

### 背景介绍

`srotmg` 是 BLAS Level 1 的纯标量算子，用四个单精度标量构造修正 Givens 旋转参数，输出三个原地更新标量和一个五元素参数数组。当前 `ops-blas/blas/rotmg/arch35` 已有 Ascend 950 实现，但 A2/A3 产品支持表和 `arch22` 代码尚未补齐。本任务复用该实现的 Netlib 算法、host 参数校验和测试工程，仅完成 A2/A3 架构适配。

## 需求分析

### 功能与接口

接口声明沿用 `include/cann_ops_blas.h` 中已有公共 API，不新增产品私有接口：

```cpp
aclblasStatus_t aclblasSrotmg(
    aclblasHandle_t handle,
    float* d1,
    float* d2,
    float* x1,
    const float* y1,
    float* param);
```

参数语义如下：

| 参数 | 方向 | 说明 |
| --- | --- | --- |
| `handle` | 输入 | 有效 BLAS handle，携带 stream；为空返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| `d1` | 输入/输出 | x 分量缩放因子，原地更新 |
| `d2` | 输入/输出 | y 分量缩放因子，原地更新 |
| `x1` | 输入/输出 | 第一个向量分量，原地更新 |
| `y1` | 输入 | 第二个向量分量，只读 |
| `param` | 输出 | 五元素旋转参数，原地写入 |

五个数据指针必须全部为 host 指针或全部为 device 指针。混合 host/device 指针返回 `ACLBLAS_STATUS_INVALID_VALUE`。任一参数为空也返回 `ACLBLAS_STATUS_INVALID_VALUE`。

### 数学定义

算子构造矩阵 H，使得：

```text
H^T * diag(d1, d2) * H = diag(d1_new, d2_new)
H * [x1, y1]^T = [x1_new, 0]^T
```

参数编码遵循 Netlib `srotmg`/cuBLAS `cublasSrotmg`：

| `param[0]` | H 的隐式形式 | 输出字段 |
| --- | --- | --- |
| `-1` | `[[h11,h12],[h21,h22]]` | `param[1..4]=h11,h21,h12,h22` |
| `0` | `[[1,h12],[h21,1]]` | `param[2]=h21,param[3]=h12` |
| `1` | `[[h11,1],[-1,h22]]` | `param[1]=h11,param[4]=h22` |
| `-2` | 单位矩阵 | `param[0]=-2`，其余输出清零 |

### 产品范围

本任务只支持 `FLOAT32` 标量输入输出。无 shape、维度、stride、broadcast、矩阵尺寸和枚举属性；模板中与这些概念相关的通用条目不适用于本算子。

## 详细设计

### 工程目录

```text
blas/rotmg/arch22/
├── srotmg_host.cpp
├── srotmg_kernel.cpp
└── srotmg_tiling_data.h

test/rotmg/srotmg/arch22/
├── srotmg_npu_wrapper.h
├── srotmg_test.cpp
└── srotmg_test.csv
```

`arch22` 文件由现有 `arch35` 实现迁移而来，接口名、公共头文件和 CMake 入口保持不变。`blas/rotmg/README.md` 的产品支持表同步标记 A2/A3 为支持。

### Host 侧设计

1. 检查 handle 和五个数据指针。
2. 对每个数据指针调用 `aclrtPointerGetAttributes`，判断其位于 host 还是 device。
3. 全部为 host 时，读取 `*y1`，调用 `SrotmgCpuCompute` 直接计算，不启动 kernel。
4. 全部为 device 时，在 handle 绑定的 stream 上启动一个 block 的 `srotmg_kernel`，不在 API 内同步 stream。
5. 出现混合内存位置时返回 `ACLBLAS_STATUS_INVALID_VALUE`。

Host 路径使用局部 `paramLocal[5]` 计算完成后再复制到用户输出，避免异常路径留下未初始化字段。测试中的 host 标量使用 `aclrtMallocHost` 分配，以保证运行时能够正确识别指针属性。

### CPU/Kernel 算法

算法严格复用 Netlib `srotmg`，使用以下单精度常量：

```text
GAM=4096.0f
GAMSQ=1.67772e7f
RGAMSQ=5.96046e-8f
```

核心分支：

1. `d1 < 0`：设置 flag 为 `-1`，四个 h 参数和 `d1/d2/x1` 全部置零。
2. `d1 >= 0` 且 `d2*y1 == 0`：设置 flag 为 `-2`，保留 `d1/d2/x1`，清零 `param[1..4]`。
3. 当 `abs(d1*x1*x1) > abs(d2*y1*y1)` 时，计算 `h21=-y1/x1`、`h12=(d2*y1)/(d1*x1)` 和 `su=1-h12*h21`。`su>0` 走 flag `0`，否则走 flag `-1` 全零分支。
4. 其他情况：若 `d2*y1*y1 < 0`，走 flag `-1`；否则计算 `h11`、`h22` 并走 flag `1`。
5. 对 `d1` 和 `abs(d2)` 分别执行 Netlib 缩放保护循环。缩放过程中按标准规则切换 flag 和 h 参数，使用 `<=`、`>=` 边界判断。

Kernel 侧通过 `GlobalTensor<float>::GetValue/SetValue` 直接访问五个 GM 标量，仅 block 0 执行。计算使用显式比较实现绝对值，避免 A2 device 编译器对 host-only 数学函数的依赖。`SrotmgTilingData` 无实际字段，仅作为 kernel ABI 占位；若具体 CANN 版本不接受按值传递，则改用 arch22 常见的 `GM_ADDR tilingGm` 形式，kernel 不读取该参数。

### 异步与资源设计

算子只使用用户传入的五个标量地址，不申请额外 workspace。device 路径将 kernel 排入 handle 的 stream 后立即返回，结果读取由调用方在需要时同步。禁止在 `aclblasSrotmg` 内部调用 `aclrtSynchronizeStream`、`aclrtMalloc` 或 `aclrtMemcpy` 作为每次调用的固定开销。

## 可维可测分析

### 精度标准

golden 使用 CBLAS `cblas_srotmg` 生成。FLOAT32 逐元素判定：

```text
rtol = 2^-10
atol = 2^-16
matched_ratio >= 0.99
max_abs_error <= 1e-2 或 32*ULP
```

`param[0]` 是离散 flag，必须与 golden 精确相等；`d1/d2/x1` 和 `param[1..4]` 使用上述 FLOAT32 容差比较。Inf/NaN 用例不额外定义产品私有行为，保持与 CBLAS golden 的分支和传播结果一致。

### 测试设计

测试工程复用 `test/rotmg/srotmg` 的 GTest、CSV 参数和 golden 包装，并新增 `arch22` 测试目录。测试覆盖：

- flag `-2/-1/0/1` 四条主路径；
- d1/d2/x1/y1 的零值、负值、相等边界；
- GAM 缩放的大数、小数和多轮缩放；
- Inf/NaN；
- handle 为空、五个数据指针为空、混合 host/device 指针；
- 全 host 和全 device 两种执行位置；
- 任务书给出的三条单次性能 case 和连续调用 case。

任务 CSV 的 `param`、`iters`、`expect_result` 为可选扩展列，旧 CSV 缺失时分别默认 `out`、`1`、`ACLBLAS_STATUS_SUCCESS`，保证与仓内既有测试兼容。`iters>1` 时每轮重置输入，在同一 stream 中连续调用并计算平均耗时。

### 性能标准

测试设备为 Atlas A2/910B3，先 warmup，再使用不少于 50 次有效采样取平均。三条核心目标为：

| 场景 | 目标平均耗时 |
| --- | ---: |
| 一般 flag=1 路径 | `<= 2.67 us` |
| flag=-2 快速路径 | `<= 2.35 us` |
| 缩放保护路径 | `<= 2.61 us` |

使用 `msprof` 采集 kernel 时间，GTest 结果同时记录 host 准备、kernel、同步和 golden 比对的保守上界。

## 兼容性分析

公共 API 与 950PR 保持一致，不新增产品私有符号。A2/A3 使用 `arch22`，Ascend 950 继续使用 `arch35`，通过 CMake 的 SOC 到架构目录映射自动选择实现。host/device 语义、返回码、参数编码和异步 stream 语义在两种架构上保持一致。

## 编译、验收与交付

环境准备：CANN 9.1.0、Ascend 910B3、已加载 `set_env.sh`。调试编译：

```bash
bash build.sh --soc=ascend910b3 --ops=srotmg
```

精度验证使用任务目录的 `verify_accuracy.py`，性能验证使用 `verify_performance.py`，通过后再执行：

```bash
bash build.sh --pkg --soc=ascend910b3 --ops=srotmg
```

当前设计文档 PR 不宣称上述命令已经在 A2 真机通过；合入前由 `muluzhe/ops-blas` 的代码分支补充实际编译日志、GTest 输出和 `msprof` 结果。

交付包括：

1. `ops-blas` 中的 `arch22` 算子、测试和 README 支持表 PR；
2. `cann-ops-competitions` 对应任务目录下的 `docs/design.md` PR；
3. 自测报告，包含环境、commit、用例参数、逐标量精度结果、性能采样和截图；
4. 在个人 `ops-blas` fork 中邀请 `Ascend-CANN` 为 Developer。