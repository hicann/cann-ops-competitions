# gpu_test-ops-sparse

**SDDMM / SpGEMM / SpSM** 的 GPU 内部验证：cuSPARSE 精度（vs scipy）+ 固定 3 组性能 benchmark（填任务书 GPU 参考耗时）。

不涉及 NPU，不走 ATK 框架。

## 目录结构

```
gpu_test-ops-sparse/
├── run_all_gpu.sh          # 精度：三算子各 200 case
├── run_perf_gpu.sh         # 性能：每算子 3 组 fixed case
├── run_perf_gpu.py
├── run_sddmm_gpu.py
├── run_spgemm_gpu.py
├── run_spsm_gpu.py
├── generate_cases.py       # 重新生成 cases/*.json
├── cases/
│   ├── sddmm_cases.json    # 200 条（前 3 条 = 任务书 fixed case）
│   ├── spgemm_cases.json
│   └── spsm_cases.json
└── lib/
    ├── sparse_gpu_cusparse.py
    ├── sparse_common.py
    ├── case_generator.py
    ├── case_builder.py
    ├── runner_common.py
    ├── perf_cases.py
    └── gpu_timing.py
```

## 依赖

- Python 3 + PyTorch（CUDA）
- `libcusparse.so`（可设 `CUSPARSE_LIB_PATH`）
- `numpy`、`scipy`（SpGEMM β≠0 必需）

## 用法

```bash
cd /home/z00889627/gpu_test-ops-sparse

# 精度：三算子 smoke（前 3 条 fixed case）
./run_all_gpu.sh --start 0 --end 3

# 精度：单算子全量 200 case
python3 run_spgemm_gpu.py

# 性能：三算子 fixed case，输出填任务书
./run_perf_gpu.sh --markdown

# 性能：单算子
./run_perf_gpu.sh --op sddmm --markdown

# 重新生成 200 case JSON
python3 generate_cases.py
```

## 与 ATK 的关系

| 项目 | gpu_test-ops-sparse | ATK_ops-sparse |
|------|---------------------|----------------|
| 目的 | GPU 参考实现验证 + 填任务书耗时 | NPU 正式验收 |
| 用例 | `cases/*.json`（200/算子） | `generator_*.py`（~500/算子） |
| NPU | 不涉及 | 被测对象 |

修改 cuSPARSE 封装时，需同步 `lib/sparse_gpu_cusparse.py` 与 `ATK_ops-sparse/sparse_gpu_cusparse.py`。
