# add算子 测试报告

## 1. 测试目标

本版本围绕 Add 算子的 6 个 API 进行测试设计与验证：

- `aclnnAdd`
- `aclnnAdds`
- `aclnnInplaceAdd`
- `aclnnInplaceAdds`
- `aclnnAddV3`
- `aclnnInplaceAddV3`

测试目标是尽可能覆盖以下评分文件中的主要执行路径：

- `op_api/aclnn_add.cpp`
- `op_api/aclnn_add_v3.cpp`
- `op_api/add.cpp`
- `op_host/arch35/add_tiling_arch35.cpp`

同时满足题目要求：结果验证、`[PASS]/[FAIL]` 输出、最终汇总、失败返回非 0。

## 2. 设计思路

本版本基于官方 Add example 进行扩展，主要思路如下：

1. 为正常执行用例在 CPU 侧计算期望值，并与设备侧输出进行比较。
2. 覆盖不同的数据类型、`alpha` 取值、shape 组合和 API 变体。
3. 增加异常输入与参数校验场景，用于覆盖 API 前置检查逻辑。
4. 保持输出格式统一，便于直接在日志中观察每个 case 的执行结果。

## 3. 测试覆盖范围

### 3.1 `aclnnAdd`（tensor + alpha * tensor）

覆盖了以下类型场景：

- `float32` 基础同 shape 加法
- `float32` 广播加法
- `float32` 的 `alpha=0`
- 非连续张量输入
- `float16`、`bf16` 同类型路径
- `float16/float32`、`bf16/float32` 混合路径
- `int32`、`int64`、`int16`、`int8`、`uint8`、`bool`
- `complex64`
- 空 tensor 的 zero-workspace 场景

### 3.2 `aclnnAdds`（tensor + alpha * scalar）

覆盖了以下路径：

- `float32` 标量加法
- `float32` 不同 `alpha` 缩放值
- 非连续张量场景
- `float16`、`bf16`、`int32`、`bool` 等类型路径
- 空 tensor 场景

### 3.3 原地 API：`aclnnInplaceAdd` / `aclnnInplaceAdds`

覆盖了以下原地更新路径：

- `float32` 原地加法
- 广播原地加法
- `bool` 原地路径
- `float`、`int64` 等原地 scalar 路径

### 3.4 `aclnnAddV3` / `aclnnInplaceAddV3`

覆盖了 V3 版本 API 的典型场景：

- scalar + tensor
- `float`、`float16`、`bf16`、`int32`、`int8` 等代表性路径
- 空 tensor 与输出类型变化场景

### 3.5 异常输入与边界值

补充了以下校验类场景：

- `nullptr`
- 输出 shape 不匹配
- 不支持的数据类型
- 广播不合法
- 维度过高
- `NaN / Inf`
- 整数溢出/回绕

## 4. 结果验证方法

结果验证在 CPU 侧独立完成：

- `float / bf16 / float16`：按浮点逻辑构造期望值，并使用容差比较
- 整数类型：精确比较
- `complex`：分别比较实部和虚部
- 广播、非连续张量、标量输入等场景都按对应 shape/stride 规则计算期望值

## 5. 输出与返回值

本版本输出格式如下：

- 每个用例输出一条 `[PASS]` 或 `[FAIL]`
- 程序结尾输出汇总：`=== Summary: X failed, Y total ===`
- 若存在失败用例，程序返回非 0；全部通过时返回 0

## 6. 使用方式

可按题目文档中的流程执行：

```bash
bash build.sh --pkg --soc=ascend950 --ops=add --vendor_name=custom --cov
./build_out/cann-ops-math-custom_linux-x86_64.run
bash build.sh --run_example add eager cust --vendor_name=custom --simulator --soc=ascend950 --cov
```

## 7. 说明

本版本重点关注 Add 算子的主调度逻辑、`alpha` 缩放路径、V3 API 逻辑和典型参数校验路径，目标是在保证可验证性的前提下尽可能提升覆盖率。
