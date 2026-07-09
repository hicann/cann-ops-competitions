# ATK_ops-sparse

aclSparse **SDDMM ** 的 ATK 精度验收工程（NPU 被测，CPU golden；可选 GPU 对照）。

说明：本 ATK 测试工程未完成完整 NPU 全量验证，仅作为参考示例提供。开发者实际开展测试时，若遇到执行异常、精度偏差等问题，可根据业务场景对脚本进行合理调整与适配修改。

## 目录结构

```
ATK_ops-sparse/
├── run_sparse_atk.sh              # 统一入口
├── sparse_executor_common.py      # CSR 生成、seed、精度工具
├── sparse_npu_acl.py              # NPU aclsparse 绑定
├── sparse_gpu_cusparse.py         # GPU cuSPARSE 参考路径
├── aclSparseSddmm/
│   ├── aclSparseSddmm.yaml
│   ├── generator_aclSparseSddmm.py
│   ├── executor_aclSparseSddmm.py
│   └── result/aclSparseSddmm/json/all_aclSparseSddmm.json
├── aclSparseSpgemm/               # 同上结构
└── aclSparseSpsm/                 # 同上结构
```

## 依赖

- Python 3 + PyTorch
- **NPU**：`torch_npu`、已安装 `libops_sparse.so`（可设 `OPS_SPARSE_LIB_PATH`）
- **GPU**（可选）：CUDA + `libcusparse.so`
- **CPU golden**：`scipy`（SpGEMM β≠0 必需）
- ATK 框架：`atk` 命令可用

## 用法

```bash
cd /home/ATK_ops-sparse

# NPU 精度验收（推荐）
./run_sparse_atk.sh sddmm npu --gen

# GPU 对照（cuSPARSE vs CPU）
./run_sparse_atk.sh spgemm gpu -s 0 -e 3

# 自动选 backend：npu → gpu → cpu
./run_sparse_atk.sh sddmm auto --gen
```

参数说明：
- 第 1 个参数：`sddmm` | `spgemm` | `spsm`
- 第 2 个参数：`auto` | `npu` | `gpu` | `cpu`（默认 `auto`）
- `--gen`：重新生成用例 JSON（首次或改 generator 后）
- 其余参数透传给 `atk task`（如 `-s 0 -e 10` 切片用例）

## 用例

- 用例定义：`aclSparse*/generator_*.py` + `*.yaml`
- 生成：`atk case -f aclSparseSddmm.yaml -p generator_aclSparseSddmm.py`
- 运行时会自动 `--gen`（若 JSON 不存在）
