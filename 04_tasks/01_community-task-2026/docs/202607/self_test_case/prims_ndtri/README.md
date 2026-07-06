# Ndtri接口测试用例

## 概述

本用例用于验证Ndtri高阶API的功能与性能。Ndtri按元素计算标准正态分布的CDF逆函数（分位函数），即给定概率值p，返回满足Φ(x)=p的分位数x。

$$\mathrm{Ndtri}(p) = \sqrt{2}\,\sum_{k=0}^{\infty} c_k\left(\frac{p-1/2}{1/2}\right)^{2k+1}$$

其中 $\Phi(x)=p$，级数在 $|p-1/2|<1/2$ 内收敛，$c_k$ 为 Acklam 算法系数。

> **说明：** 本用例以Digamma高阶API进行功能与性能验证（kernel 内调用 `AscendC::Digamma`，golden 对齐 `scipy.special.digamma`）。待 Ndtri API 开发完成后，将 `.asc` 中的 `AscendC::Digamma` 替换为 `AscendC::Ndtri`、`gen_data.py` 的 golden 替换为 `scipy.special.ndtri`、输入范围改回 `(0,1)` 即可。

## 支持的产品及CANN软件版本

| 产品 | CANN软件版本 |
|------|-------------|
| Ascend 950PR | >= CANN 9.1.0 |

## 目录结构介绍

```text
├── ndtri
│   ├── scripts
│   │   └── gen_data.py         // 输入数据和真值数据生成脚本
│   ├── CMakeLists.txt          // 编译工程文件
│   ├── data_utils.h            // 数据读入写出函数
│   ├── ndtri.asc               // Ascend C高阶API用例实现
│   ├── run.sh                  // 编译执行脚本（功能/性能验证）
│   └── README.md               // 用例说明文档
```

## 用例规格

- 接口功能：按元素计算标准正态分布CDF逆函数Φ⁻¹(p)，输入p取值范围(0,1)。
  $$dstTensor_i = \Phi^{-1}(srcTensor_i)$$
- 接口规格：

  <table>
  <tr><td rowspan="1" align="center">用例类型</td><td colspan="4" align="center">NdtriCustom</td></tr>

  <tr><td rowspan="3" align="center">用例输入</td></tr>
  <tr><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">x</td><td align="center">[2048]</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td rowspan="2" align="center">用例输出</td></tr>
  <tr><td align="center">y</td><td align="center">[2048]</td><td align="center">float</td><td align="center">ND</td></tr>

  <tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">ndtri_custom</td></tr>
  </table>

## 用例说明

固定shape为输入x[2048]，输出y[2048]，单核处理，数据一次性搬入UB计算。

### 1. 功能验证

CopyIn → Compute(Ndtri) → CopyOut，输出与真值对比验证精度。

### 2. 性能验证

采用AIV_VEC占比计算法，评估计算时间占比，标准为占比 ≥ 90%。

$$\text{AIV\_VEC占比} = \frac{\text{computeTime}}{\text{total2} - \text{total1}}$$

- `total1`：纯搬运耗时基线；`total2`：搬运+计算1000次耗时；`computeTime`：AIV_VEC计算时间。三者均通过msProf采集。
- 分母 `total2 - total1` 扣除搬运开销，得到纯计算增量时间。

**性能标准：AIV_VEC占比 ≥ 90%。**

### 3. 性能数据

性能验证通过msProf采集AI Core性能数据，存放在 `op_summary_*.csv` 中。主要字段说明如下：

| 字段名 | 字段含义 |
|:---:|:---|
| Task Duration(μs) | Task整体耗时，包含调度到加速器的时间、加速器上的执行时间以及响应结束时间。 |
| aiv_time(μs) | Task在AI Vector Core上的执行时间。 |
| aiv_vec_time(μs) | vec类型指令（向量类运算指令）耗时。 |
| aiv_vec_ratio | vec类型指令的cycle数在total cycle数中的占用比。 |
| aiv_scalar_time(μs) | scalar类型指令（标量类运算指令）耗时。 |
| aiv_scalar_ratio | scalar类型指令的cycle数在total cycle数中的占用比。 |
| aiv_mte2_time(μs) | mte2类型指令（GM->UB搬运类指令）耗时。 |
| aiv_mte2_ratio | mte2类型指令的cycle数在total cycle数中的占用比。 |
| aiv_mte3_time(μs) | mte3类型指令（UB->GM搬运类指令）耗时。 |
| aiv_mte3_ratio | mte3类型指令的cycle数在total cycle数中的占用比。 |

本用例性能验证实际采集以下字段用于占比计算：
- `aiv_time`：分别取 TEST_MODE=2（空跑基线）的值作为 `total1`、TEST_MODE=3（计算场景）的值作为 `total2`。
- `aiv_vec_time`：取 TEST_MODE=3 的值作为 `computeTime`（AIV_VEC计算时间）。

### 4. 验证范围

- 功能要求：
  - shape 覆盖对齐与非对齐场景，`1`/`32`/`1023`/`2048`
  - 覆盖 `float` 类型，正常输入范围(0,1)、及`±Inf`特殊值处理
- 性能要求：数据量为 `1k`/`4k`/`8k`/`16k`/`32k`/`64k` 时，AIV_VEC 占比均需满足 `≥ 90%`

## 编译运行

- 配置环境变量
  请根据当前环境上CANN开发套件包的[安装方式](../../../../docs/quick_start.md#prepare&install)，配置环境变量。
  ```bash
  source ${install_path}/cann/set_env.sh
  ```

  > **说明：** `${install_path}` 为CANN包安装目录，未指定安装目录时默认安装至 `/usr/local/Ascend` 下。

- 用例执行

  通过 `run.sh` 一键编译并执行用例，脚本会自动完成 cmake 编译、生成测试数据、运行可执行程序。

  ```bash
  bash run.sh -r <cpu|sim|npu> --size <N> --is_perf <0|1>
  ```

  参数说明（均为必传）：

  | 参数 | 可选值 | 说明 |
  |------|--------|------|
  | `-r` | `cpu`、`sim`、`npu` | 运行模式：CPU调试、NPU仿真、NPU上板（性能验证仅支持`npu`模式） |
  | `--size` | 正整数 | 计算量，即 TOTAL_LENGTH |
  | `--is_perf` | `0`、`1` | `0` 功能验证；`1` 性能验证 |

  示例：
  ```bash
  bash run.sh -r cpu  --size 2048 --is_perf 0   # CPU功能验证
  bash run.sh -r sim  --size 4096 --is_perf 0   # NPU仿真功能验证
  bash run.sh -r npu  --size 2048 --is_perf 1   # NPU上板性能验证
  ```

  > **说明：** `--is_perf 1` 性能验证时，脚本自动完成 msProf 采集并计算 AIV_VEC 占比是否达标。

- 执行结果

  功能验证（`--is_perf 0`）通过时输出：
  ```bash
  test pass!
  ```
  性能验证（`--is_perf 1`）达标时输出：
  ```bash
  性能验证达标，当前占比: xx%
  ```
  > `--is_perf 1` 用于性能验证，输出非计算结果，不做精度对比。
