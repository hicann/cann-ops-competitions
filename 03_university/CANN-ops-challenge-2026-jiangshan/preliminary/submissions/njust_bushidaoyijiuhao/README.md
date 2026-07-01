# Huawei Operator Competition - Erf Custom Operator

本仓库实现了一个面向昇腾 Ascend 平台的自定义 `Erf` 算子工程，包含 host 侧算子注册、shape/type 推导、tiling 选择，以及 AICore kernel 侧的向量化误差函数近似计算。

## 团队信息

团队名称：不是倒一就好
- 所属单位：南京理工大学
- 团队成员：  
- 王子龙，负责人
- 联系人：王子龙
-  联系邮箱：2515585075@qq.com

## 项目概览

- 算子名称：`Erf`
- 输入：`x`，`float32`，`ND` 格式
- 输出：`y`，`float32`，`ND` 格式
- 目标平台：`ascend910b`、`ascend910_93`
- 构建方式：Ascend CMake / 自定义算子 package

kernel 使用多项式近似计算 `erf(x)`，并根据输入长度自动选择执行策略：

- 小规模或单 block 场景：单核直接计算
- 多核场景：按 block 划分输入区间并行计算
- 长输入场景：使用双缓冲队列流水化搬运、计算和写回

## 目录结构

```text
.
├── CMakeLists.txt
├── op_host
│   ├── CMakeLists.txt
│   └── erf.cpp
└── op_kernel
    ├── CMakeLists.txt
    ├── erf.cpp
    ├── erf_tiling.h
    └── tiling_key_erf.h
```

## 关键文件说明

`op_host/erf.cpp`

- 注册 `Erf` 算子定义
- 声明输入输出数据类型与格式
- 完成输出 shape/type 推导
- 根据输入长度和 AIV core 数量设置 `blockDim`
- 选择模板参数 `DT_X`、`USE_PIPE`、`ONE_CORE`

`op_kernel/erf.cpp`

- 实现 AICore kernel 入口 `erf`
- 实现单核、多核、流水线三类执行路径
- 处理非 32B 对齐输入输出的 padding copy
- 使用向量指令完成截断、多项式计算和结果写回
- 对部分范围进行快速路径优化：
  - 大正数区间近似输出 `1`
  - 大负数区间近似输出 `-1`
  - 小绝对值区间使用线性近似 `2 / sqrt(pi) * x`

`op_kernel/erf_tiling.h`

- 定义 tiling 数据结构，当前主要传递输入总长度

`op_kernel/tiling_key_erf.h`

- 定义 AscendC 模板参数选择组合

## 环境要求

请在已安装并配置 Ascend CANN / Ascend Toolkit 的环境中构建，且 CMake 可以找到 `ASC` package。

常见环境初始化示例：

```bash
source /home/developer/Ascend/ascend-toolkit/set_env.sh
```

实际路径请以本机 Ascend Toolkit 安装位置为准。

## 构建

```bash
mkdir -p build
cd build
cmake ..
make -j$(nproc)
```

构建完成后，自定义算子 package 会生成在 `build` 目录下。

如果目标任务要求生成二进制 package，可根据竞赛环境使用：

```bash
make binary -j$(nproc)
```

## 实现思路

host 侧 tiling 逻辑以 `1024` 个元素作为 core 划分参考，并在单 block、普通多 block、长输入流水线之间选择不同模板实例。kernel 侧以 `4096` 个元素作为 tile 长度，长输入会使用双缓冲队列减少搬运和计算等待。

误差函数近似采用如下形式：

```text
erf(x) ~= x * P(x^2)
```

其中输入会先截断到 `[-2.24, 2.24]` 区间，随后通过多项式系数进行向量化计算。

## 注意事项

- 当前算子仅注册了 `float32` 输入输出。
- workspace 大小设置为 `0`。
- `erf_tiling.h` 中保留了 `blockLength` 字段，但当前 kernel 主要使用 `length` 字段自行计算分块。
- 构建、运行和精度验证需要在 Ascend 设备环境中完成。

