# ATK_ops-sparse

aclSparse **SpGEMM** 的 ATK 精度验收工程（NPU 被测，CPU golden 单标杆）。

说明：本 ATK 测试工程未完成完整 NPU 全量验证，仅作为参考示例提供。开发者实际开展测试时，若遇到执行异常、精度偏差等问题，可根据业务场景对脚本进行合理调整与适配修改。

## 目录结构

```
atk_test/
├── run_sparse_atk.sh              # 统一入口
├── sparse_executor_common.py      # CSR 生成、seed、精度工具（CPU golden）
├── sparse_npu_acl.py              # NPU aclsparse 绑定
└── aclSparseSpgemm/
    ├── aclSparseSpgemm.yaml
    ├── generator_aclSparseSpgemm.py
    ├── executor_aclSparseSpgemm.py
    └── result/aclSparseSpgemm/json/all_aclSparseSpgemm.json
```

## 依赖

- Python 3 + PyTorch
- **NPU**：`torch_npu`、已安装 `libops_sparse.so`（可设 `OPS_SPARSE_LIB_PATH`）
- **CPU golden**：`scipy`（SpGEMM β≠0 必需）
- ATK 框架：`atk` 命令可用（https://gitcode.com/Ascend/ATK）

## 用法

```bash
cd self_test_case/sp_gemm/atk_test

# NPU 精度验收（推荐）
./run_sparse_atk.sh spgemm npu

# 切片用例
./run_sparse_atk.sh spgemm npu -s 0 -e 10
```

参数说明：
- 第 1 个参数：`sddmm` | `spgemm` | `spsm`
- 第 2 个参数：`auto` | `npu` | `cpu`（默认 `auto`；精度验收用 `npu`）
- `--gen`：重新生成用例 JSON（首次或改 generator 后）
- 其余参数透传给 `atk task`（如 `-s 0 -e 10` 切片用例）

## 用例

- 预置 200 条：`aclSparseSpgemm/result/aclSparseSpgemm/json/all_aclSparseSpgemm.json`
- 用例定义：`generator_aclSparseSpgemm.py` + `aclSparseSpgemm.yaml`
- 生成：`atk case -f aclSparseSpgemm.yaml -p generator_aclSparseSpgemm.py`
- 运行时会自动 `--gen`（若 JSON 不存在）
