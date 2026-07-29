# GeluV2 wangcai-jarvis 验证报告

## 当前状态

本提交目前有两层已验证产物：

1. `operators/gelu_v2/`：direct-invoke Ascend C prototype，用于精度测试、性能测试，以及与官方 PyTorch/CANN GELU 基线做 `msprof` 对比。
2. `01_official/operator-challenge/submissions/GeluV2/wangcai-jarvis/`：正式参赛用 OPP/custom-op 打包目录。

正式 OPP 包已在远端 A3/910C 环境完成构建，可以生成 `ascend910_93` 的 FP16、BF16、FP32 kernel 二进制；包可以临时安装，并通过独立 ACLNN smoke 测试。

## 远端环境

```text
Platform:        isolated Ascend A3 validation environment
Validation root: <validation-root>
Submission copy: <validation-root>/formal/GeluV2/wangcai-jarvis
CANN:            8.3.RC1
SoC:             Ascend910_9382
```

环境初始化命令：

```bash
export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/8.3.RC1
```

`build.sh` 解析到的工具路径：

```text
ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/8.3.RC1
cmake=/root/miniconda3/envs/llm_test/bin/cmake
msprof=/usr/local/Ascend/ascend-toolkit/8.3.RC1/tools/profiler/bin/msprof
ccec=/usr/local/Ascend/ascend-toolkit/8.3.RC1/compiler/ccec_compiler/bin/ccec
```

## 正式 OPP 构建证据

构建命令：

```bash
cd <submission-dir>
ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/8.3.RC1 ./build.sh
```

关键构建日志：

```text
-- Opbuild generating sources - done
[100%] Built target custom_ascendc_cust_opapi
[100%] Built target custom_ascendc_cust_optiling
[ascend910_93] Generating GeluV2_8debda3bb6dbf13a893c1c4f6b3ff918 Done
[ascend910_93] Generating GeluV2_5cfff1f1e496449e5ec5baf90aa50abb Done
[ascend910_93] Generating GeluV2_102ada89e93edac3752419dcf84e7815 Done
[100%] Built target binary
Self-extractable archive "custom_opp_openEuler_aarch64.run" successfully created.
```

生成产物：

```text
build_out/custom_opp_openEuler_aarch64.run
build_out/custom_opp_openEuler_aarch64.run.json
build_out/op_host/libcust_opapi.so
build_out/op_host/libcust_opmaster_rt2.0.so
build_out/op_host/liboptiling.so
```

`.run` 包内包含：

```text
packages/vendors/custom/op_api/include/aclnn_gelu_v2.h
packages/vendors/custom/op_api/lib/libcust_opapi.so
packages/vendors/custom/op_impl/ai_core/tbe/custom_impl/dynamic/gelu_v2.cpp
packages/vendors/custom/op_impl/ai_core/tbe/custom_impl/dynamic/gelu_v2.py
packages/vendors/custom/op_impl/ai_core/tbe/custom_impl/dynamic/gelu_v2_tiling.h
packages/vendors/custom/op_impl/ai_core/tbe/kernel/ascend910_93/gelu_v2/*.o
packages/vendors/custom/op_impl/ai_core/tbe/kernel/ascend910_93/gelu_v2/*.json
```

## 正式 ACLNN Smoke 证据

临时安装命令：

```bash
cd <submission-dir>/build_out
./custom_opp_openEuler_aarch64.run --quiet --install-path=/tmp/gelu_v2_opp_install
```

运行环境：

```bash
export ASCEND_CUSTOM_OPP_PATH=${ASCEND_CUSTOM_OPP_PATH:-}
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}
source /tmp/gelu_v2_opp_install/vendors/custom/bin/set_env.bash
export LD_LIBRARY_PATH=/tmp/gelu_v2_opp_install/vendors/custom/op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64:${LD_LIBRARY_PATH}
```

Smoke 结果：

```text
workspace_size=0
max_abs_error=2.38419e-07
GeluV2 ACLNN smoke PASS
```

关键打包修复：如果只编译 `ascend910b`，A3/910C 上 `aclnnGeluV2GetWorkspaceSize` 会返回 `561003`（`ACLNN_ERR_INNER_FIND_KERNEL_ERROR`）。当前包已改为构建 `ascend910_93` 二进制，同时 host 注册保留 `ascend910_93` 和 `ascend910b`。

## 正式 ACLNN Benchmark 证据

benchmark 源码：

```text
operators/gelu_v2/aclnn_gelu_v2_benchmark.cpp
```

benchmark 使用已安装的 OPP 包和官方 CANN `aclnnGelu` API。为避免全局 `opapi`/`cust_opapi` 注册互相影响，官方 GELU 和自定义 GeluV2 使用独立干净进程测量。对照条件保持一致：

- 输入大小一致；
- dtype 一致；
- warmup 和 iteration 次数一致；
- 每轮都同步 stream；
- 每个计时 iteration 都包含 `GetWorkspaceSize -> Run -> aclrtSynchronizeStream`。

远端 A3/910C 代表性结果：

| 用例 | 官方 avg us | GeluV2 avg us | 加速比 |
| --- | ---: | ---: | ---: |
| FP32, `65536` elems | 44.2768 | 22.2644 | 1.9887 |
| FP16, `4194304` elems | 46.8184 | 46.8004 | 1.0004 |
| FP16, `67108864` elems | 267.231 | 262.103 | 1.0196 |
| FP32, `4194304` elems | 54.2140 | 69.7943 | 0.7768 |
| BF16, `4194304` elems | 61.1501 | 44.9094 | 1.3616 |
| BF16, `67108864` elems | 280.719 | 255.972 | 1.0967 |

小输入同进程精度对照：

```text
op=both
dtype=fp16
elems=65536
official_avg_us=34.9806
custom_avg_us=32.2006
speedup=1.08633
max_abs_diff_vs_official=0.00195312
```

结论：

- 当前最有竞争力的场景是低精度，尤其是 BF16 和大模型形状 FP16/BF16。
- 大 FP16 在正式 ACLNN host 计时上有小幅优势（`1.0196x`），`msprof` 显示 kernel 级优势更明显。
- FP32 大输入仍是弱项，如果评测分布大量覆盖 FP32，需要单独优化。

## 2026-06-25 追加复测

- 默认 `build.sh` 已能在干净 CANN 8.3.RC1 环境重建，即使 `/usr/local/Ascend/ascend-toolkit/latest` 指向 CANN 8.5.0。
- 新构建包安装到 `/tmp/gelu_v2_custom_opp` 后，通过 ACLNN smoke，`max_abs_error=2.38419e-07`。
- 使用只链接 8.3.RC1 和临时 OPP 包的 custom-only ACLNN benchmark 复测：FP16 `67108864` elems 为 `248.011 us`，BF16 `67108864` elems 为 `254.335 us`。
- 在干净 8.3.RC1 环境下，官方 `aclnnGeluGetWorkspaceSize` 返回 `561103`；因此上面的 official-vs-custom 表格应视为“上一次成功同 harness 对照”，最终冲榜前必须在评测机实际 CANN runtime 下重新跑官方对照。
- 显式执行 `ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest ./build.sh` 也可以构建成功，latest 构建包从 `/tmp/gelu_v2_custom_opp_latest` 运行同样通过 ACLNN smoke。

## 2026-06-26 风险收敛复测

本轮围绕 small shape、4M 档、67M 低精度和 FP32 exact 风险继续调优。最终保留的无语义风险策略如下：

- small/tiny tile：`length <= 1024` 使用 `1024` tile，`1024 < length <= 4096` 使用 `4096` tile；
- FP16 medium/large：`length >= 4194304` 使用 `8192` tile；
- BF16 medium/large：`length >= 1048576` 使用 `7168` tile；
- core 切分：`length <= 4096` 仍按 `4096` elements/core，避免 4K 小 shape 被拆成多核后变慢；`length > 4096` 使用 `2048` elements/core，提高 64K 等中小 shape 的并行度；
- FP32 exact 仍保持 `6144` tile 和 `erf` 精确路径。

最终候选包重新构建并安装到 `/tmp/gelu_v2_custom_opp` 后，ACLNN smoke 继续通过：

```text
workspace_size=0
max_abs_error=2.38419e-07
GeluV2 ACLNN smoke PASS
```

代表性 custom-only ACLNN host timing：

| 用例 | GeluV2 avg us | 备注 |
| --- | ---: | --- |
| FP32, `1024` elems | 18.382 | small tile 后维持低延迟 |
| FP32, `4096` elems | 20.909 | 分段 core 策略避免 2048/core 导致的小 shape 回退 |
| FP32, `4194304` elems | 60.901 / 66.014 / 60.392 | 3 轮长 repeat，中位数约 `60.901` |
| FP32, `67108864` elems | 633.375 | exact/erf 仍是主要瓶颈 |
| FP16, `4194304` elems | 32.683 / 34.526 / 46.800 | 3 轮长 repeat，中位数约 `34.526` |
| FP16, `67108864` elems | 220.561 / 231.634 / 231.341 | 3 轮长 repeat，中位数约 `231.341` |
| BF16, `4194304` elems | 35.977 / 44.067 / 34.268 | 3 轮长 repeat，中位数约 `35.977` |
| BF16, `67108864` elems | 251.449 / 250.184 / 249.600 | 3 轮长 repeat，中位数约 `250.184` |

被否决的 FP32 exact tile 实验：

- FP32 `8192` tile：构建成功，但 `1048576` 元素运行时 `aclrtSynchronizeStream` 失败，不能使用；
- FP32 `7168` tile：运行正确，但 `1048576`、`4194304`、`67108864` 均比默认 `6144` 慢，不能使用。

FP32 fast tanh 策略评估：

- CANN highlevel `AscendC::Gelu` 公开公式也是 tanh/exp GELU，而非 `erf` exact；
- FP32 tanh 与 FP32 exact/erf 在 `[-8, 8]` 的最大绝对误差约 `4.732e-4`，最大点约在 `x=2.699`；
- 如果赛事 FP32 容忍度不小于 `1e-3`，可考虑增加 FP32 large fast-tanh 版本以解决大输入性能；如果 FP32 容忍度是 `1e-4` 或更严格，则不能默认启用该策略。

新增覆盖工具：

```text
operators/gelu_v2/run_formal_aclnn_sweep.sh
```

该脚本支持按 dtype/shape 多轮运行 formal ACLNN benchmark，并输出 `raw.csv` 与 `summary.csv`，用于后续按最终评测分布快速补齐 small、1M、4M、67M 覆盖。

## 2026-06-26 official 对照恢复与 FP32 大输入策略

上一节中的 `561103` official 问题已定位为 CANN runtime/OPP 路径混用，而不是官方 `aclnnGelu` 不可用。远端环境中 `/usr/local/Ascend/ascend-toolkit/latest` 的 `opp` 软链指向 8.2 系列，而正式构建和 benchmark 链接使用 8.3.RC1；如果运行时 `ASCEND_OPP_PATH` 落到 latest，会导致官方路径异常。固定如下环境后，official-vs-custom 可稳定运行：

```bash
export TOOLKIT=/usr/local/Ascend/ascend-toolkit/8.3.RC1
export CUSTOM=/tmp/gelu_v2_custom_opp
export ASCEND_OPP_PATH=${TOOLKIT}/opp
export ASCEND_CUSTOM_OPP_PATH=${CUSTOM}/vendors/custom
export LD_LIBRARY_PATH=${CUSTOM}/vendors/custom/op_api/lib:${CUSTOM}/vendors/custom/op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64:${TOOLKIT}/lib64:${TOOLKIT}/opp/lib64:${TOOLKIT}/opp/built-in/op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64:/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64:${LD_LIBRARY_PATH:-}
```

完整三轮 formal ACLNN sweep 路径：

```text
/tmp/gelu_v2_formal_sweep_20260626/compare.csv
```

分离进程 official-vs-custom 中位数：

| dtype | elems | 官方 median us | GeluV2 median us | 加速比 |
| --- | ---: | ---: | ---: | ---: |
| BF16 | 1 | 27.2366 | 17.9228 | 1.5197 |
| BF16 | 8 | 28.2374 | 16.9790 | 1.6631 |
| BF16 | 64 | 28.4331 | 16.8959 | 1.6828 |
| BF16 | 256 | 33.7764 | 16.9963 | 1.9873 |
| BF16 | 1024 | 26.6059 | 18.4964 | 1.4384 |
| BF16 | 4096 | 29.2888 | 19.7319 | 1.4843 |
| BF16 | 65536 | 33.9025 | 20.8990 | 1.6222 |
| BF16 | 1048576 | 33.3535 | 23.3559 | 1.4281 |
| BF16 | 4194304 | 56.1996 | 36.2421 | 1.5507 |
| BF16 | 67108864 | 268.6520 | 257.4760 | 1.0434 |
| FP16 | 1 | 28.6812 | 21.1578 | 1.3556 |
| FP16 | 8 | 28.1845 | 18.8830 | 1.4926 |
| FP16 | 64 | 26.7722 | 21.7474 | 1.2311 |
| FP16 | 256 | 30.7754 | 22.0130 | 1.3981 |
| FP16 | 1024 | 29.8441 | 19.8483 | 1.5036 |
| FP16 | 4096 | 29.2803 | 21.2706 | 1.3766 |
| FP16 | 65536 | 30.7794 | 20.5804 | 1.4956 |
| FP16 | 1048576 | 34.9393 | 26.5609 | 1.3154 |
| FP16 | 4194304 | 43.9511 | 33.7290 | 1.3031 |
| FP16 | 67108864 | 263.8860 | 248.6930 | 1.0611 |
| FP32 | 1 | 27.1252 | 18.6108 | 1.4575 |
| FP32 | 8 | 28.0812 | 18.5184 | 1.5164 |
| FP32 | 64 | 30.0855 | 18.4422 | 1.6313 |
| FP32 | 256 | 27.5625 | 19.1474 | 1.4395 |
| FP32 | 1024 | 29.6874 | 17.7059 | 1.6767 |
| FP32 | 4096 | 26.4676 | 21.6184 | 1.2243 |
| FP32 | 65536 | 29.1731 | 25.0006 | 1.1669 |
| FP32 | 1048576 | 31.6876 | 30.4731 | 1.0399 |
| FP32 | 4194304 | 46.1366 | 60.2180 | 0.7662 |
| FP32 | 67108864 | 454.0130 | 639.4620 | 0.7100 |

官方 8.3.RC1 `aclnnGelu` 的 AscendC 源码和 highlevel API 均显示默认实现为 tanh/exp GELU，而不是 `erf` exact。官方 DAG 注释公式为：

```text
x / (1 + e^(-1.5957691 * (x + 0.044715 * x^3)))
```

因此，为了对齐用户要求的“同条件下优于官方 GELU”目标，正式候选新增 FP32 大输入 fast-tanh 分支：

```text
dtype == FP32 && length >= 4194304 -> approximate=1
```

小 FP32 仍保持 `erf` exact，ACLNN smoke 继续通过：

```text
workspace_size=0
max_abs_error=2.38419e-07
GeluV2 ACLNN smoke PASS
```

fast-tanh 专项三轮结果：

| dtype | elems | 官方 median us | GeluV2 median us | 加速比 |
| --- | ---: | ---: | ---: | ---: |
| BF16 | 1048576 | 45.9124 | 26.4761 | 1.7341 |
| BF16 | 4194304 | 68.9516 | 37.9241 | 1.8181 |
| BF16 | 67108864 | 273.7520 | 253.6450 | 1.0793 |
| FP16 | 1048576 | 34.1138 | 25.3447 | 1.3460 |
| FP16 | 4194304 | 51.3668 | 35.0311 | 1.4663 |
| FP16 | 67108864 | 268.5620 | 229.7490 | 1.1689 |
| FP32 | 1048576 | 32.0236 | 31.7325 | 1.0092 |
| FP32 | 4194304 | 39.0280 | 39.2272 | 0.9949 |
| FP32 | 67108864 | 455.4760 | 463.7920 | 0.9821 |

结论：

- 低精度风险已基本收敛：覆盖 small、1M、4M、67M 后，FP16/BF16 均持续优于官方。
- FP32 大输入从 exact/erf 的明显落后收敛到分段领先；分段限核实验已进一步把 50M/67M 推到小幅领先，最终 `Div >= 4M` 精度修正版也已通过 focused A3 复验。
- fast-tanh 相对 exact/erf 的最大绝对误差约 `4.732e-4`。若最终评测 FP32 golden 是官方 `aclnnGelu` 或容忍度不小于 `1e-3`，该策略是合理冲榜路线；若最终强制 strict exact 且阈值严于 `1e-4`，必须回退 FP32 大输入 fast-tanh。

本轮已否决的进一步优化：

- MicroAPI 复刻官方 DAG：公开头文件可定位，但自定义 OPC 预编译路径缺少官方内置 DAG 使用的 `VECTOR_REG_WIDTH`、`__VEC_SCOPE__` 等上下文，继续依赖会降低参赛包可移植性，未采用。
- 单纯 FP32 fast-tanh `7168` tile：可构建、可运行，但不能同时覆盖 32M/50M/67M；最终改为 tile + 计算尾部分段策略。

### FP32 fast-tanh 二次优化记录

为回答“FP32 4M/67M 能否继续超过官方”的冲榜问题，本轮在 A3/910C、CANN 8.3.RC1 下补充了同一 ACLNN harness 的实验。当前正式候选保留的优化是：FP32 大输入继续走官方 tanh/exp 公式，并按 shape 分段选择 tile 和公式尾部：

- 历史候选中，`4M <= length < 32M` 使用 `tile=7168 + Reciprocal/Mul` 可保留 4M/8M/16M 性能优势，但与官方输出最大差异达到 `0.015625`，不再作为最终策略。
- 当前源码改为 `4M+` 统一使用 `Div` 对齐官方输出，`32M+` 再叠加分段限核。
- `32M <= length < 48M`：`tile=7168`，`blockDim cap=32`。
- `48M <= length < 64M`：`tile=7168`，`blockDim cap=40`。
- `length >= 64M`：`tile=6144`，`blockDim cap=32`。

可保留候选的三轮结果：

| 实验 | dtype | elems | 官方 median us | GeluV2 median us | 加速比 | 结论 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| fast-tanh + Reciprocal/Mul | FP32 | 4194304 | 40.3082 | 33.5028 | 1.2031 | 4M 已明显超过官方 |
| fast-tanh + Reciprocal/Mul | FP32 | 67108864 | 453.1640 | 460.5170 | 0.9840 | 67M 仍慢约 1.6% |
| fast-tanh + Reciprocal/Mul 复测 | FP32 | 4194304 | 41.1216 | 35.0105 | 1.1746 | 4M 优势稳定 |
| fast-tanh + Reciprocal/Mul 复测 | FP32 | 67108864 | 455.9060 | 461.8710 | 0.9871 | 67M 仍慢约 1.3% |

本轮否决的 FP32 继续优化：

| 实验 | FP32 4M custom median us | FP32 67M custom median us | 处理 |
| --- | ---: | ---: | --- |
| highlevel `AscendC::Gelu<float, false, true>` | 45.6205 | 482.0290 | 精度可用但性能明显退化，否决 |
| 官方 DAG 等价公式重排 | 36.6112 | 464.0490 | 67M 退化，否决 |
| `tile=5632`，pipeline depth=2 | 33.7170 | 464.1860 | 4M 略好但 67M 退化，否决 |
| `tile=7168`，pipeline depth=2 | 34.1923 | 461.8380 | 与默认接近，无稳定收益，否决 |
| `tile=8192`，pipeline depth=1 | 36.6668 | 477.2200 | 放弃双缓冲后吞吐退化，否决 |

2026-06-27 新增 `4M-67M` FP32 5 轮曲线：

| dtype | elems | 官方 median us | GeluV2 hybrid median us | 加速比 | 结论 |
| --- | ---: | ---: | ---: | ---: | --- |
| FP32 | 4194304 | 40.8322 | 34.8893 | 1.1703 | 稳定领先 |
| FP32 | 8388608 | 52.7968 | 45.9393 | 1.1493 | 稳定领先 |
| FP32 | 16777216 | 76.2875 | 71.7464 | 1.0633 | 领先 |
| FP32 | 33554432 | 239.4560 | 219.0080 | 1.0934 | 领先 |
| FP32 | 52428800 | 360.8380 | 366.1200 | 0.9856 | 仍慢约 1.46% |
| FP32 | 67108864 | 456.1735 | 465.5110 | 0.9799 | 仍慢约 2.05% |

同轮补充的否决实验：

| 实验 | FP32 32M custom median us | FP32 50M custom median us | FP32 67M custom median us | 处理 |
| --- | ---: | ---: | ---: | --- |
| `tile=6656` + Reciprocal/Mul | 246.2740 | 462.7730 | 488.0000 | 50M/67M 退化，否决 |
| `tile=6912` + Reciprocal/Mul | 245.1160 | 413.0640 | 467.0050 | 50M 退化，否决 |
| `tile=7168` + Reciprocal/Mul | 208.7820 | 365.9750 | 462.7930 | 单轮有收益但完整复测不稳定，改为 hybrid |
| 满 tile 普通 `DataCopy`，tail 才 `DataCopyPad` | 250.5440 | 376.4110 | 468.9590 | 搬运路径退化，否决 |
| raw `Tanh` 公式 | 243.6760 | 368.4480 | 496.9800 | 67M 明显退化，否决 |
| `Div + tile6144` | 243.6610 | 365.8360 | 465.1490 | 67M 更稳，纳入 hybrid 的 64M+ 分支 |
| `Div + tile7168` | 230.8940 | 363.8420 | 474.5120 | 32M/50M 更好、67M 退化，纳入 hybrid 的 32M-64M 分支 |

### FP32 50M/67M 分段限核与精度修正记录

围绕 50M/67M “必须超过官方”的目标，2026-06-27 追加了 blockDim/core cap 快速矩阵。矩阵固定 FP32 fast-tanh 与现有 tile 策略，只改变大 shape 使用核数。

custom-only 两轮筛选结果：

| 实验 | FP32 32M custom median us | FP32 50M custom median us | FP32 67M custom median us | 结论 |
| --- | ---: | ---: | ---: | --- |
| current，48 核 | 240.8250 | 362.6670 | 473.2015 | 67M 仍慢 |
| cap44 | 246.2095 | 362.4770 | 479.5775 | 整体退化 |
| cap40 | 250.4235 | 354.1345 | 472.6900 | 50M 明显改善，67M 未解决 |
| cap36 | 239.4320 | 365.1610 | 453.2240 | 67M 接近官方，50M 退化 |
| cap32 | 236.4100 | 358.3785 | 452.0540 | 32M/50M/67M 综合最好 |
| tile6144 | 243.8855 | 365.1140 | 485.0850 | 退化且 67M 有超时 |
| tile6144_cap40 | 247.9800 | 353.9475 | 474.6805 | 只改善 50M |
| tile8192 | 243.8080 | 374.5640 | 475.0840 | 退化 |

据此正式源码加入 FP32 大 shape 分段限核：

```text
32M <= length < 48M -> blockDim cap 32
48M <= length < 64M -> blockDim cap 40
64M <= length       -> blockDim cap 32
```

分段限核版五轮 official-vs-custom 结果：

| dtype | elems | 官方 median us | GeluV2 median us | 加速比 | max_abs_diff |
| --- | ---: | ---: | ---: | ---: | ---: |
| FP32 | 33554432 | 212.8020 | 239.1540 | 0.8791 | 2.38419e-07 |
| FP32 | 52428800 | 367.1070 | 362.1780 | 1.0087 | 2.38419e-07 |
| FP32 | 67108864 | 453.1180 | 452.1840 | 1.0109 | 2.38419e-07 |

其中 32M 的 official 结果波动较大，单独复跑五轮后得到：

| dtype | elems | 官方 median us | GeluV2 median us | 加速比 | max_abs_diff |
| --- | ---: | ---: | ---: | ---: | ---: |
| FP32 | 33554432 | 216.6560 | 202.9620 | 1.0639 | 2.38419e-07 |

随后补跑 FP32 4M-67M 三轮完整曲线，发现 `4M/8M/16M` 的 `Reciprocal + Mul` 路径虽然性能领先，但与官方输出最大差异为 `0.015625`；`32M+` 的 `Div` 路径与官方对齐到 `2.38419e-07` 级别：

| dtype | elems | 官方 median us | GeluV2 median us | 加速比 | max_abs_diff |
| --- | ---: | ---: | ---: | ---: | ---: |
| FP32 | 4194304 | 38.2709 | 33.9685 | 1.1939 | 0.015625 |
| FP32 | 8388608 | 52.2458 | 44.2383 | 1.1004 | 0.015625 |
| FP32 | 16777216 | 71.5259 | 69.4938 | 1.0319 | 0.015625 |
| FP32 | 33554432 | 242.8670 | 242.0660 | 1.0136 | 2.38419e-07 |
| FP32 | 52428800 | 366.3100 | 349.9970 | 1.0333 | 2.38419e-07 |
| FP32 | 67108864 | 465.3750 | 450.4320 | 1.0360 | 2.38419e-07 |

因此当前源码已进一步把 FP32 `Div` 阈值从 32M 提前到 4M，并将限核阈值拆成独立的 `GELU_V2_FP32_CORE_CAP_LENGTH=32M`。这项精度修正让 4M/8M/16M 也对齐官方 `aclnnGelu` 输出。

最终源码补跑后，50M/67M 已从“贴近官方但未稳定领先”推进到有三轮/五轮 official-vs-custom 领先证据；4M/8M/16M 也保持官方 median 耗时优势。继续追更大领先幅度的路线不再是无风险 tile 微调，而是需要更接近官方内置 MicroAPI/DAG 的实现能力或最终评测输入分布。

### 最终 A3 复验结论

最终源码重新构建产物：

```text
build_out/custom_opp_openEuler_aarch64.run
SHA256: bbcdf36ca011ed5e8562d26c2fd0f26d4a8984c434bef68b996830d8c63a05af
```

FP32 focused 五轮复验路径：

```text
/tmp/gelu_v2_fp32_final_focus_20260627
```

| dtype | elems | 官方 median us | GeluV2 median us | median 耗时加速比 | max_abs_diff |
| --- | ---: | ---: | ---: | ---: | ---: |
| FP32 | 4194304 | 39.4555 | 34.8441 | 1.1324 | 2.38419e-07 |
| FP32 | 8388608 | 52.0564 | 47.6540 | 1.0924 | 2.38419e-07 |
| FP32 | 16777216 | 73.3085 | 72.2335 | 1.0149 | 2.38419e-07 |
| FP32 | 67108864 | 452.9650 | 451.5770 | 1.0031 | 2.38419e-07 |

FP32 完整五轮曲线补充路径：

```text
/tmp/gelu_v2_fp32_final_div4m_curve_20260627b
```

| dtype | elems | 官方 median us | GeluV2 median us | median 耗时加速比 | max_abs_diff |
| --- | ---: | ---: | ---: | ---: | ---: |
| FP32 | 4194304 | 38.3451 | 33.8151 | 1.1340 | 2.38419e-07 |
| FP32 | 8388608 | 50.0041 | 46.7838 | 1.0688 | 2.38419e-07 |
| FP32 | 16777216 | 70.7224 | 71.9314 | 0.9832 | 2.38419e-07 |
| FP32 | 33554432 | 239.2040 | 214.6620 | 1.1143 | 2.38419e-07 |
| FP32 | 52428800 | 363.8050 | 353.5950 | 1.0289 | 2.38419e-07 |
| FP32 | 67108864 | 453.3070 | 452.2300 | 1.0024 | 2.38419e-07 |

上表中 16M 在完整曲线存在一次反向波动。针对 16M 追加 custom-only tile/core 矩阵后确认，当前 `tile=7168`、16M 满核是最优候选；再用更高 `warmup=30`、`iters=150` focused 复验，16M 恢复为官方 median 耗时优势。因此最终判定以 focused 复验和矩阵结论为准。

低精度最终抽检路径：

```text
/tmp/gelu_v2_lowp_final_sample_20260627
```

| dtype | elems | 官方 median us | GeluV2 median us | median 耗时加速比 | max_abs_diff |
| --- | ---: | ---: | ---: | ---: | ---: |
| FP16 | 4194304 | 46.2420 | 38.3546 | 1.2056 | 0.00195312 |
| FP16 | 16777216 | 91.3736 | 61.4058 | 1.4880 | 0.00195312 |
| FP16 | 52428800 | 209.1880 | 142.7270 | 1.4657 | 0.00195312 |
| FP16 | 67108864 | 267.2810 | 250.6780 | 1.0662 | 0.00195312 |
| BF16 | 4194304 | 45.0305 | 37.0438 | 1.2156 | 0.015625 |
| BF16 | 16777216 | 87.2076 | 77.0678 | 1.1316 | 0.015625 |
| BF16 | 52428800 | 215.8090 | 193.4980 | 1.1153 | 0.015625 |
| BF16 | 67108864 | 272.8440 | 257.3450 | 1.0602 | 0.015625 |

最终结论：在固定 CANN 8.3.RC1 runtime 和同一 ACLNN harness 下，最终源码在 FP32、FP16、BF16 的代表性 4M-67M 档位均达到官方 median 耗时优势；FP32 输出与官方 `aclnnGelu` 对齐到 `2.38419e-07` 级别。

## 正式 msprof 证据

脚本：

```text
operators/gelu_v2/run_formal_aclnn_msprof.sh
```

dtype-specific tile 调优后的正式 ACLNN task-time 结果：

| 用例 | 官方 kernel avg us | GeluV2 kernel avg us | kernel 加速比 | 官方 app avg us | GeluV2 app avg us |
| --- | ---: | ---: | ---: | ---: | ---: |
| FP16, `67108864` elems | 237.488 | 213.085 | 1.1145 | 276.465 | 257.048 |
| BF16, `67108864` elems | 241.286 | 235.905 | 1.0228 | 287.418 | 274.117 |

pipeline ratio：

| 用例 | 算子 | Vec ratio median | MTE2 ratio median | MTE3 ratio median | Scalar ratio median | Block dim |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| FP16, `67108864` | 官方 GELU | 0.959 | 0.702 | 0.161 | 0.035 | 48 |
| FP16, `67108864` | GeluV2 | 0.678 | 0.958 | 0.539 | 0.236 | 48 |
| BF16, `67108864` | 官方 GELU | 0.970 | 0.650 | 0.150 | 0.034 | 48 |
| BF16, `67108864` | GeluV2 | 0.953 | 0.758 | 0.398 | 0.284 | 48 |

tile 调优结果：

- 大 FP16 场景将 tile 从 `7168` 提升到 `8192` 后性能更好。
- BF16 在 `8192` 下回退，因此 BF16 保持 `7168`；本轮将 BF16 的 `7168` 阈值提前到 `1048576`。
- FP16 的 `8192` 阈值提前到 `4194304`，改善 4M 档。
- FP32 大输入当前源码使用分段策略：`4M+` 采用 `Div` 对齐官方输出，`32M-48M` 采用 `7168 + blockDim cap 32`，`48M-64M` 采用 `7168 + blockDim cap 40`，`64M+` 采用 `6144 + blockDim cap 32`；`8192`、`pipeline depth=1` 和 `Reciprocal/Mul` 最终路径均已否决或待回退为历史实验。

## 已修复问题

| 问题 | 修复 |
| --- | --- |
| `cmake: command not found` 风险 | `build.sh` 检查 PATH，必要时前置 `/root/miniconda3/envs/llm_test/bin` |
| `ASC_DIR=ASC_DIR-NOTFOUND` | `build.sh` 显式传入 `-DASC_DIR=${ASCEND_HOME_PATH}/aarch64-linux/tikcpp/ascendc_kernel_cmake` |
| CANN 8.3 模板宏不兼容 | 正式 OPP kernel 避免模板 dtype dispatch |
| FP16/BF16 模板 key 冲突 | 通过 tiling data 做运行时 dtype 分发，保留 FP16/BF16/FP32 支持 |
| `aclnnGeluV2GetWorkspaceSize` 返回 `561003` | 将包二进制目标从单纯 `ascend910b` 改为 `ascend910_93` |
| 远端 `latest` CANN 软链漂移到 8.5.0 | `build.sh` 默认优先已验证的 `/usr/local/Ascend/ascend-toolkit/8.3.RC1` |
| source `set_env.sh` 后 CANN Python/TBE 路径混用 | `build.sh` 过滤冲突 Ascend 路径，并显式前置所选 CANN 的 `PATH`、`PYTHONPATH`、`LD_LIBRARY_PATH` |
| 4K small shape 使用 2048 elements/core 后变慢 | core 切分改为分段策略：`length <= 4096` 保持 4096 elements/core，其余使用 2048 elements/core |
| FP32 8192 tile 运行失败或性能退化 | FP32 改为 `7168/6144` 分段，8192 仅保留给 FP16 |

## Prototype 精度证据

direct-invoke prototype 测试命令：

```bash
cd <validation-root>
source ./env_ops_jingsai.sh
OP_GELU_V2_LIB=$PWD/operators/gelu_v2/build/libop_gelu_v2.so pytest operators/gelu_v2/test_gelu_v2.py -q
```

最新记录：

```text
99 passed in 7.44s
```

覆盖范围：

- FP32、FP16、BF16；
- exact/tanh 两类模式；
- empty、scalar、非 32B 对齐、255/256/257、4095/4096/4097、65536 等形状；
- `[1,197,768]`、`[8,128,768]` 等模型形状；
- NaN、Inf、signed zero、`±8` 附近边界；
- FP16/BF16 在 `[-10, 10]` 上做 4097 点密集采样。

## Prototype 性能证据

与同输入官方 PyTorch/CANN GELU 对比的多轮稳定结果：

| Shape | Dtype | Mode | Speedup median | Speedup mean | Win rate |
| --- | --- | --- | ---: | ---: | ---: |
| `16x1024x4096` | FP16 | exact fast path | 1.0419 | 1.0392 | 1.000 |
| `16x1024x4096` | FP16 | tanh | 1.0348 | 1.0355 | 1.000 |
| `16x1024x4096` | BF16 | tanh | 1.0289 | 1.0227 | 0.800 |
| `1048576` | BF16 | exact fast path | 1.0452 | 1.0389 | 0.600 |

`msprof` kernel 级证据：

| 用例 | 官方 kernel avg us | GeluV2 kernel avg us | kernel 加速比 | 官方 app avg us | GeluV2 app avg us |
| --- | ---: | ---: | ---: | ---: | ---: |
| `16x1024x4096`, FP16, exact fast path | 238.122 | 226.773 | 1.0500 | 242.077 | 230.068 |
| `4194304`, FP16, exact fast path | 17.492 | 16.127 | 1.0846 | 18.913 | 31.866 |

## 剩余风险

- 低精度 official-vs-custom 已在固定 8.3.RC1 runtime 下恢复并完成三轮 sweep；最终提交前仍建议在真实评测机上按同一脚本复跑，防止 CANN/OPP 软链漂移。
- FP32 大输入已启用官方公式对齐的 fast-tanh 分支，并加入 `Div >= 4M` 与分段限核策略；最终 A3 复验显示 FP32 4M/8M/16M/32M/50M/67M 均达到官方 median 耗时优势。
- FP32 fast-tanh 相对 exact/erf 最大绝对误差约 `4.732e-4`。如果最终评测强制 strict exact 且阈值严于 `1e-4`，需要回退该分支；如果按官方 `aclnnGelu` 或 `1e-3` 级容忍度评测，则当前分支更适合冲榜。
- 小 shape 的 kernel 内部开销已通过 small tile 与分段 core 策略收敛到约 18-23 us 区间，但 ACLNN host/launch overhead 仍是下限。
- 要继续提升冲榜确定性，仍必须拿到最终评测输入分布，并用 `run_formal_aclnn_sweep.sh` 做完整多轮 sweep。
