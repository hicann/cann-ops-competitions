# GeluV2 提交说明：wangcai-jarvis

本目录是 TeamName `wangcai-jarvis` 的 GeluV2 正式参赛提交目录，采用 CANN OPP/custom-op 打包流程，并使用 Ascend C 实现核心 kernel。

## 构建方式

在 A3/910C 验证机器上执行：

```bash
cd 01_official/operator-challenge/submissions/GeluV2/wangcai-jarvis
ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/8.3.RC1 ./build.sh
```

`build.sh` 会在构建前打印实际解析到的工具路径：

- `cmake`
- `msprof`
- `ccec`
- `ASCEND_HOME_PATH`

脚本会完成 OPP 包配置、host 侧库构建、`ascend910_93` Ascend C kernel 二进制构建，并生成：

```text
build_out/custom_opp_openEuler_aarch64.run
```

当前 A3 验证机上，`/usr/local/Ascend/ascend-toolkit/latest` 可能指向比已验证链路更新的 CANN 版本。为避免 Python/TBE/编译器路径混用，`build.sh` 默认优先使用已验证的：

```text
/usr/local/Ascend/ascend-toolkit/8.3.RC1
```

脚本会在 `set_env.sh` 后过滤冲突的 Ascend 路径，并显式前置所选 CANN 的 `PATH`、`PYTHONPATH` 和 `LD_LIBRARY_PATH`。如果需要强制使用机器当前 `latest` 工具链，可执行：

```bash
ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest ./build.sh
```

## 算子语义

- 算子类型：`GeluV2`
- 输入：`x`，格式 `ND`
- 输出：`y`，shape 和 dtype 与 `x` 一致
- 支持 dtype：FP32、FP16、BF16
- 默认语义：按 GELU 实现；FP32 小输入使用 `erf` 精确公式，FP32 从 4M 元素起为对齐官方 `aclnnGelu` 公式使用 tanh/exp 快路径，并在公式尾部使用 `Div` 避免 `Reciprocal + Mul` 带来的官方对齐误差；FP16/BF16 使用已通过精度验证的 tanh/exp 快路径

正式 OPP kernel 使用运行时 dtype 分发，而不是 Ascend C 模板 dtype 分发。这样可以规避 CANN 8.3 在模板提取时 FP16/BF16 同属 16-bit 类型导致的 key 冲突问题。

## 已验证产物

远端构建已生成：

```text
build_out/custom_opp_openEuler_aarch64.run
build_out/op_host/libcust_opapi.so
build_out/op_host/libcust_opmaster_rt2.0.so
build_out/op_kernel/ascendc_kernels/binary/ascend910_93/gelu_v2/*.o
build_out/op_kernel/ascendc_kernels/binary/ascend910_93/gelu_v2/*.json
```

包已在远端 A3/910C 环境中临时安装到 `/tmp/gelu_v2_custom_opp`，并通过独立 ACLNN smoke 测试：

```text
workspace_size=0
max_abs_error=2.38419e-07
GeluV2 ACLNN smoke PASS
```

性能与精度验证记录见 `VALIDATION_REPORT.md`。当前证据显示：

- small/tiny shape 已使用 1024/4096 tile 收敛 kernel 内部开销；
- FP16 从 4M 元素起使用 8192 tile，BF16 从 1M 元素起使用 7168 tile；
- 4M 与 67M 低精度场景已有 formal ACLNN official-vs-custom 三轮 sweep 证据；
- FP32 大输入已启用 tanh/exp fast path，并采用分段策略：`4M+` 使用 `Div` 对齐官方输出，`32M+` 额外启用分段限核；其中 `32M-48M` 使用 32 核，`48M-64M` 使用 40 核，`64M+` 使用 32 核。最终 A3 复验中，FP32 `4M/8M/16M/32M/50M/67M` median 耗时均优于官方 `aclnnGelu`，`max_abs_diff` 为 `2.38419e-07`；
- 官方 `aclnnGeluGetWorkspaceSize` 的 `561103` 问题已定位为 CANN/OPP 路径混用；固定 `ASCEND_OPP_PATH=/usr/local/Ascend/ascend-toolkit/8.3.RC1/opp` 后 official 对照可跑。

`operators/gelu_v2/` 下的 direct-invoke prototype 仍作为 PyTorch 同输入对照、精度测试和 `msprof` 分析的辅助验证工程。

更大范围的 formal ACLNN 覆盖可使用：

```bash
operators/gelu_v2/run_formal_aclnn_sweep.sh
```
