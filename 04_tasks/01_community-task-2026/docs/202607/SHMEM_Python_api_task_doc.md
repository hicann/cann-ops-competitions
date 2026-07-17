# SHMEM Python 接口开发任务书

## 基础信息

- **技术标签**：Python 接口开发 / pybind11 封装
- **适配硬件**：Atlas A2/A3 训练系列产品或推理系列产品
- **开源仓地址**：[https://gitcode.com/cann/shmem](https://gitcode.com/cann/shmem)
- **CANN 版本**：CANN 9.0.0 或社区版较新版本（推荐 ≥ 9.0.0）
- **开发语言**：C++（pybind11）+ Python
  
## 任务概述
  
SHMEM（ACLSHMEM）是面向昇腾集群的对称共享内存与单边 RMA 通信库。当前主线仓已提供部分 Host C++ 接口的 Python 绑定（`src/host/python_wrapper/pyshmem.cpp`、`src/python/shmem/`），但仍有一批 Host API 与 `shmem.core` 高层封装尚未对齐。
请基于主线已有 Python 工程，使用 **pybind11** 将下表所列 **尚未实现** 的 C++ Host 接口封装为 Python 接口，并补齐对应 `shmem.core` 高层封装、测试与文档，保证与 C++ 功能、语义一致。验收通过后提交 PR 合入 [cann/shmem](https://gitcode.com/cann/shmem) 主线。
现有实现与文档可参考：
- 绑定实现：[pyshmem.cpp](https://gitcode.com/cann/shmem/blob/master/src/host/python_wrapper/pyshmem.cpp)
- Python 包：[src/python/shmem](https://gitcode.com/cann/shmem/tree/master/src/python/shmem)
- API 文档：[docs/api/pythonAPI.md](https://gitcode.com/cann/shmem/blob/master/docs/api/pythonAPI.md)
- 测试样例：[examples/python_extension](https://gitcode.com/cann/shmem/tree/master/examples/python_extension)
  
## 核心开发要求
  
### 功能实现要求

1. 使用 pybind11 将下列 **待实现** Host C++ 接口完整封装到 `shmem._pyshmem`，参数、返回值、集合语义、错误行为与 C++ 保持一致。
2. 在 `shmem.core` 中补齐对应高层 Pythonic 封装（含异常、Buffer/Stream 约定），风格与现有 `init` / `buffer` / `put` / `get` 一致。
3. 同步更新 `docs/api/pythonAPI.md`、QuickStart 示例与 `examples/python_extension` 测试，保证文档与实现一致。
4. Device 侧 Kernel API（`aclshmemx_roce_*` 等）不在本任务范围。
   
### 待实现接口列表
   
#### 1. 初始化与多实例
   
| 序号  | C++ 接口名称                         | 说明                         |
| --- | -------------------------------- | -------------------------- |
| 1   | aclshmemx_set_attr_uniqueid_args | UID 初始化属性构造（需独立导出，当前仅内部调用） |
| 2   | aclshmemx_instance_ctx_get       | 获取当前实例上下文                  |
| 3   | aclshmemx_instance_ctx_set       | 按 instance_id 切换实例上下文      |
   
#### 2. 对称堆内存（含 mem_type）
   
| 序号  | C++ 接口名称         | 说明                |
| --- | ---------------- | ----------------- |
| 4   | aclshmemx_malloc | 按 mem_type 分配对称内存 |
| 5   | aclshmemx_calloc | 按 mem_type 分配并清零  |
| 6   | aclshmemx_align  | 按 mem_type 对齐分配   |
| 7   | aclshmemx_free   | 按 mem_type 释放     |

#### 3. 引擎配置

| 序号  | C++ 接口名称                  | 说明                                 |
| --- | ------------------------- | ---------------------------------- |
| 8   | aclshmemx_set_mte_config  | 配置 MTE UB offset / size / sync_id  |
| 9   | aclshmemx_set_sdma_config | 配置 SDMA UB offset / size / sync_id |
| 10  | aclshmemx_set_rdma_config | 配置 RDMA UB offset / size / sync_id |
| 11  | aclshmemx_set_udma_config | 配置 UDMA 相关参数                       |

#### 4. 集合通信与同步

| 序号  | C++ 接口名称                        | 说明                             |
| --- | ------------------------------- | ------------------------------ |
| 12  | aclshmem_barrier                | Team 内 barrier                 |
| 13  | aclshmem_barrier_all            | 全局 barrier                     |
| 14  | aclshmem_sync                   | Team 内 sync                    |
| 15  | aclshmem_sync_all               | 全局 sync                        |
| 16  | aclshmemx_barrier_on_stream     | 指定 ACL Stream 的 team barrier   |
| 17  | aclshmemx_barrier_all_on_stream | 指定 ACL Stream 的全局 barrier      |
| 18  | aclshmemx_handle_wait           | Host 侧等待 RDMA/Stream handle 完成 |

#### 5. 日志与诊断

| 序号  | C++ 接口名称            | 说明                   |
| --- | ------------------- | -------------------- |
| 19  | aclshmemx_get_prof  | 获取 profiling 数据      |
| 20  | aclshmemx_show_prof | 打印 / 展示 profiling 信息 |

#### 6. shmem.core 高层封装（需在 `_pyshmem` 之上补齐）

| 序号  | Python 接口                                                 | 说明                        |
| --- | --------------------------------------------------------- | ------------------------- |
| 21  | barrier / barrier_all / sync / sync_all（含 on_stream）      | 集合通信与同步的 Pythonic 封装      |
| 22  | multi_instance（ctx_get / ctx_set + 带 mem_type 的 malloc 族） | 多实例场景封装                   |
| 23  | handle_wait                                               | RDMA / Stream handle 等待封装 |

### 接口约束限制

- Python 绑定需正确处理指针（建议 `intptr_t`）、枚举、`aclrtStream`、GIL 释放（`py::gil_scoped_release`），与现有 `pyshmem.cpp` 风格一致。
- `barrier` / `sync` 等为集合操作，多 PE 必须同步调用；测试需使用 `torchrun` 多卡验证。
- 多实例切换（`instance_ctx_set/get`）需保证单进程内实例隔离可验证。
- `handle_wait` 在跨机 RDMA 场景下需与现有 C++ 示例（如 `examples/rdma_handlewait_test`）语义一致。
- 已知 C++ 侧限制（如部分 Stream/Signal 跨机能力）需在 Python 文档中明确标注，不得隐瞒。
  
## 测试标准

请根据给出的自测用例和测试指导完成自测，并输出自测报告。

### 功能要求

1. 上述全部待实现接口均有对应 Python 用例，功能与 C++ 对齐。
2. `examples/python_extension/run.sh` 原有用例全部通过；新增 CC（barrier/sync）、多实例、handle_wait、引擎 config、mem_type heap 用例通过。
3. 至少在 **MTE** 引擎下完成主路径验证；若环境具备 RDMA/SDMA/UDMA，需补充对应 smoke 测试。
   
### 性能要求

1. Python Host RMA（以 `putmem_on_stream` 为代表）相对同路径 C++ 实现 overhead **≤ 5%**（同 shape、同 stream，2/4/8 卡场景）。
2. `barrier` / `barrier_on_stream` 功能正确，无明显异常超时或死锁。
   
### 精度 / 正确性要求

1. RMA put/get 结果与 C++ 参考路径 bit-exact 或在约定误差内一致。
2. Signal / handle_wait 完成后，对端数据可见性符合 C++ 语义。
3. 多实例切换后，各实例 heap / team 状态互不污染。
   
## 验收交付件

1. **设计 / 说明文档**：说明 Python 绑定分层（`_pyshmem` / `shmem.core`）、待实现接口映射表、Stream/指针/异常约定；内容完整、格式规范。
2. **可指导使用与测试的交付件**，包括：
    - 自测用例、测试代码脚本及脚本运行 README；
    - 测试结果报告：需包含全部自测用例结果、执行日志/截图、整体测试通过截图、性能对比数据截图。
3. **代码交付**：PR 链接或私仓邀请链接、代码仓路径、分支、相关目录。README 文档内容完整、规范，能指导环境准备、构建、安装与测试。
    
## PR 申请合入

测试通过后，在 SHMEM 开源仓提交 PR，申请将开发完成的 Python 接口合入主线相关目录：
- 绑定：`src/host/python_wrapper/`
- Python 包：`src/python/shmem/`
- 文档：`docs/api/pythonAPI.md`
- 测试与示例：`examples/python_extension/`
  仓库地址：https://gitcode.com/cann/shmem
  
## 参考资料

1. 文档类：[快速开始](https://gitcode.com/cann/shmem/blob/master/docs/quickstart.md)、[Python API](https://gitcode.com/cann/shmem/blob/master/docs/api/pythonAPI.md)、[Host API](https://gitcode.com/cann/shmem/blob/master/docs/api/host_api.rst)、[Stream API 说明](https://gitcode.com/cann/shmem/blob/master/docs/api/stream_api_usage.md)、[多实例设计](https://gitcode.com/cann/shmem/blob/master/docs/multi_instance.md)
2. 代码样例：[examples/python_extension](https://gitcode.com/cann/shmem/tree/master/examples/python_extension)、[examples/rdma_handlewait_test](https://gitcode.com/cann/shmem/tree/master/examples/rdma_handlewait_test)
3. 依赖说明：pybind11、Python ≥ 3.9、torch + torch_npu（多卡测试）
   
## 环境获取

1. 开源仓提供100小时免费时长，请不使用时及时关闭，用时耗尽前请务必保存相关资料，建议及时提交备份。

   ![环境截图](pics/yunkaifa.png)

2. 使用 hidevlab 算力（[https://hidevlab.huawei.com/online-develop-intro?from=hiascend](https://hidevlab.huawei.com/online-develop-intro?from=hiascend)）

   ![环境截图](pics/zaixiankaifa1.png)  

3. 如需额外环境资源，请联系昇腾小助手。
   
## 特别注意事项

1. 开发需严格遵循现有 SHMEM Python 封装风格（命名、异常、GIL、指针约定），避免破坏已有 `_pyshmem` / `shmem.core` API。
2. 所有交付件需提前完成自验证，确认符合验收标准后再提交验收申请。
3. 验收以提交验收申请时的代码为准；代码更新请重新提交验收申请。
4. 开发前请务必阅读[【社区任务】流程及注意事项](https://gitcode.com/org/cann/discussions/39)，会例行更新。
