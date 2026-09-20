# 自测报告 - AscendC Basic_API 指针化扩展（VECTOR 分册）

- 提交人：`xiexy114`
- 代码分支：`xiexy114/asc-devkit` → `feat/vector-basic-api-pointer`（commit `c6b05a58c`）
- 验证环境：Ascend950PR_9579 / `__NPU_ARCH__=3510` / `dav-3510` / CANN 9.1.0 / bisheng clang 15.0.5
- 验证方式：官方样例改造 + 实机（NPU）编译运行 + `scripts/gen_data.py` / `verify_result.py` 精度比对
- 测试日期：2026-09-17

## 1. 精度测试用例与结果

数据范围与 Shape 按各官方样例自带规格生成（含任务书要求的 [-100,100] 范围与 1/32/1024/2048 级别
规模），全部用例误差比 **0**，满足实验标准 `matched_ratio >= 0.99`。

| # | 用例 | 接口（指针路径） | 元素类型 | 结果(误差比) |
| - | --- | --- | --- | --- |
| 1 | element_wise_arithmetic | `LeakyRelu` | half | PASS (0.0000) |
| 2 | element_wise_compound_compute (SCENARIO=1) | `AddRelu<half>` | half | PASS (0.0000) |
| 3 | element_wise_compound_compute (SCENARIO=2) | `Axpy<half,half>` | half | PASS (0.0000) |
| 4 | create_vec_index | `Duplicate` + `CreateVecIndex` | int16_t | PASS (0.0000) |
| 5 | duplicate | `Duplicate<half>` | half | PASS (0.0000) |
| 6 | cast (SCENARIO=0) | `Cast<int4b_t, half>` | half→int4b_t | PASS (0.0000) |
| 7 | cast (SCENARIO=1) | `Cast<int32_t, half>` | half→int32_t | PASS (0.0000) |
| 8 | reduce | `ReduceRepeat`+`ReduceDataBlock`+`Add<float,false>` | float | PASS |
| 9 | reduce_repeat | `ReduceRepeat<SUM/MIN,T>` | half | PASS (0.0000) |
| 10 | reduce_data_block | `ReduceDataBlock<MAX/MIN/SUM,T>` | half/float | PASS (0.0000) |
| 11 | reduce_pair_elem | `ReducePairElem<SUM,half>` | half | PASS (0.0000) |
| 12 | reduce_computation (SCENARIO=1..6) | `ReduceMax/ReduceMin/ReduceSum<half>` | half | PASS×6 (0.0000) |

**合计 17 项实机运行，全部 `test pass!`，误差比 0。**

## 2. 性能测试用例与结果

任务书性能项为「无」，本节为"无回退"佐证。用例：`AddRelu`，512 half，循环 10000 次，
device 侧 `GetSystemCycle()` 计时；另用 `msprof` 采集 AIV 指标。

### 2.1 Tensor 路径（标杆）vs 指针路径（本次交付）

| 指标 | Tensor 路径 | 指针路径 | 当前/标杆 |
| --- | --- | --- | --- |
| 内核 Task Duration | 265.42 us | 261.69 us | 0.986x |
| aiv_total_cycles | 436504 | 431091 | 0.988x |
| aiv_vec_ratio | 0.985 | 0.997 | — |
| aiv_mte2_time | 0.862 us | 0.110 us | — |
| aiv_mte3_time | 0.281 us | 0.081 us | — |
| per-iter cycles (512 half) | 26.058 | 25.978 | 0.997x |

### 2.2 per-element cycle 规模扫描（1000 iters）

| count | Tensor cyc/el | 指针 cyc/el | 相对标杆(指针/张量) | 指针 cyc/iter | roofline | 当前/roofline |
| --- | --- | --- | --- | --- | --- | --- |
| 512 | 0.051088 | 0.049518 | 0.969 | 25.35 | 4 | 6.3x |
| 1024 | 0.027509 | 0.027112 | 0.986 | 27.76 | 8 | 3.5x |
| 2048 | 0.017068 | 0.016945 | 0.993 | 34.70 | 16 | 2.2x |
| 4096 | 0.012155 | 0.012058 | 0.992 | 49.39 | 32 | 1.5x |
| 8192 | 0.009601 | 0.009547 | 0.994 | 78.21 | 64 | 1.2x |
| 16384 | 0.008257 | 0.008221 | 0.996 | 134.69 | 128 | **1.05x** |

结论：瓶颈为 AIV 向量计算管道（~99%，compute-bound），搬运占比 <0.4%；指针路径在所有规模下
均不慢于 Tensor 路径；大块时已达向量吞吐上限（roofline 1.05x）。**无性能回退。**

## 3. 复现方法

见 `../README.md` 与 `../tests/`。核心命令：

```bash
source /usr/local/Ascend/cann-9.1.0/set_env.sh
bash tests/run_selftests.sh     # 期望 ALL TESTS PASSED (PASS=17 FAIL=0)
bash tests/run_perf.sh          # 输出标杆/当前 cycle 对比
bash tests/run_msprof.sh        # 输出 aiv_time / vec_ratio / mte
```

## 4. 已知边界与说明

1. 本机 CANN 9.1.0（编译器 2026-07-30）早于 asc-devkit master（9.2.0-dev，2026-09-17）。
   master 的 `impl/utils/debug/asc_printf_simt_impl.h` 使用新编译器内建 `__nop()`，本机不可编译，
   自测时临时替换为等价的旧版 `asm volatile("NOP...")` 语句（**不属交付内容**，提交未包含）。
2. 任务包 test-cases 中 `cast.asc` 场景 1 的指针调用全为裸地址且无类型锚点，
   自测时补显式模板实参 `<int32_t, half>`（语义与任务书 `__ubuf__ T*` 规范一致）。
3. 本机无 ATK / AscendOpTest，采用任务书 3.1/3.5 所述的官方样例 CMake 流程完成实机验证。

## 附录：接口覆盖缺口清单（13 / 90 已实现）

- **已实现并实机验证（13）**：Add, AddRelu, Axpy, LeakyRelu, Duplicate, CreateVecIndex, Cast,
  ReduceDataBlock, ReduceRepeat, ReducePairElem, ReduceMax, ReduceMin, ReduceSum
- **未实现（77）**：DataCopy, DataCopyPad, SetPadValue, SetLoopModePara, ResetLoopModePara, Copy,
  Exp, Ln, Abs, Reciprocal, Sqrt, Rsqrt, Relu, Neg, Sub, Mul, Div, Max, Min, BilinearInterpolation,
  Prelu, Mull, Adds, Muls, Maxs, Mins, Subs, Divs, Not, And, Or, Ands, Ors, ShiftLeft, ShiftRight,
  CastDequant, AddReluCast, AddDeqRelu, SubRelu, SubReluCast, MulAddDst, MulCast, FusedMulAdd,
  MulAddRelu, AbsSub, ExpSub, MulsCast, Compare, Compares, GetCmpMask, SetCmpMask, Select, GatherMask,
  SetDeqScale, Truncate, GetReduceRepeatSumSpr, GetReduceRepeatMaxMinSpr, Transpose, TransDataTo5HD,
  Brcb, VectorPadding, ProposalConcat, ProposalExtract, RpSort16, MrgSort4, Sort32, MrgSort,
  GetMrgSortResult, Gather, Gatherb, Scatter, SetMaskCount, SetMaskNorm, SetVectorMask, ResetMask,
  Interleave, DeInterleave
