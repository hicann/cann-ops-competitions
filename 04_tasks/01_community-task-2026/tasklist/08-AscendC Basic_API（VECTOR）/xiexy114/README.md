# AscendC Basic_API 指针化扩展（VECTOR 分册）— 交付说明

- 提交人（GitCode）：`xiexy114`
- 任务：9 月社区任务 - AscendC Basic_API 优化实现（VECTOR 矢量接口扩展）

## 目录结构

```
xiexy114/
├── README.md                 # 本文档（使用方法/编译步骤/测试命令）
├── docs/
│   ├── design.md             # 设计文档（评审申请正文）
│   └── selftest_report.md    # 自测报告（用例参数 + 精度 + 性能）
└── tests/
    ├── run_selftests.sh      # 一键正确性自测（17 项，期望 ALL TESTS PASSED）
    ├── run_perf.sh           # 一键性能对比（Tensor vs 指针）
    ├── run_msprof.sh         # msprof 指标采集
    └── perf/                 # 性能微基准源码
        ├── bench_addrelu.asc
        ├── bench_sweep.asc
        └── CMakeLists.txt
```

## 交付内容

| 交付件 | 位置 |
| --- | --- |
| 算子（接口）设计文档 | `docs/design.md` |
| 自测报告 | `docs/selftest_report.md` |
| 自测用例及测试代码 | `tests/` + 代码仓分支 |
| 待验收代码地址 | `https://gitcode.com/xiexy114/asc-devkit` 分支 `feat/vector-basic-api-pointer` |
| 覆盖接口范围 | 见 `docs/selftest_report.md` 附录 |

代码仓变更目录：`include/basic_api/`、`impl/basic_api/`（新增
`impl/basic_api/utils/kernel_utils_pointer.h`），共 16 文件、`+1065`。

## 覆盖范围

当前已实现并实机验证 **13 个接口族**：`Add`、`AddRelu`、`Axpy`、`LeakyRelu`、`Duplicate`、
`CreateVecIndex`、`Cast`、`ReduceDataBlock`、`ReducePairElem`、`ReduceRepeat`、`ReduceMax`、
`ReduceMin`、`ReduceSum`。其余接口见 `docs/selftest_report.md` 附录，可按同一范式继续扩展。

- 精度测试 case：17 项，全部 `test pass!`（误差比 0）
- 性能测试 case：任务书性能项为「无」；本报告给出无回退佐证（指针/张量 ≈ 1.0x）

## 环境准备

```bash
# 1) CANN 环境
source /usr/local/Ascend/cann-9.1.0/set_env.sh

# 2) 让官方样例编译到本仓源码（而非 CANN 内置 devkit 快照）
mkdir -p /root/ascrepo
for e in /usr/local/Ascend/cann-9.1.0/*; do ln -sfn "$e" "/root/ascrepo/$(basename "$e")"; done
rm -f /root/ascrepo/asc && ln -sfn <asc-devkit仓库路径> /root/ascrepo/asc
export ASCEND_HOME_PATH=/root/ascrepo

# 3) PyTorch/torch_npu（可选，样例脚本仅需 numpy）
#   pip3 install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
#   pip3 install torch_npu==2.7.1.post10
```

> 说明：本机 CANN 9.1.0 编译器早于仓库 master，`impl/utils/debug/asc_printf_simt_impl.h` 中的
> `__nop()` 需临时替换为旧版 `asm volatile("NOP wait:0b0000000 stall:15"::);` 才能编译；
> 该改动不属交付内容。若使用配套的新编译器则无需此步。

## 测试命令

```bash
# 正确性（需先完成"环境准备"）
bash tests/run_selftests.sh
# 期望输出： PASS=17 FAIL=0 / ALL TESTS PASSED

# 性能
bash tests/run_perf.sh
bash tests/run_msprof.sh
```

官方样例逐项编译运行方式（以 element_wise_compound_compute 为例）：

```bash
cd <asc-devkit>/examples/01_simd_cpp_api/03_basic_api/01_memory_vector_compute/element_wise_compound_compute
mkdir -p build && cd build
cmake -DSCENARIO_NUM=1 -DCMAKE_ASC_ARCHITECTURES=dav-3510 .. && make -j
python3 ../scripts/gen_data.py -scenarioNum=1
./demo
python3 ../scripts/verify_result.py output/output.bin output/golden.bin
```

## 备注

- 本机 `git@gitcode.com:22` 被防火墙拦截，代码通过 `ssh://git@ssh.gitcode.com:443/...` 推送。
- 自测使用的 test-cases 来自任务附件；其中 `cast.asc` 场景 1 补了显式模板实参（见自测报告第 4 节）。
