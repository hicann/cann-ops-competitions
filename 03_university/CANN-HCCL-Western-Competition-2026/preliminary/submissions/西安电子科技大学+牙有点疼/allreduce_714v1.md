# AllReduce 714v1 实现说明

## 1. 赛题内容存放位置

赛题官方页面：

https://competition.gitcode.com/competition/2066850206835970049/publish

本地工程位置：

`/data/mingwei/hccl_allreduce_problem_222_template`

本地说明文件：

`/data/mingwei/hccl_allreduce_problem_222_template/README.md`

本题要求实现集合通信算子 `AllReduce`：通信域内所有 rank 的输入数据按 `op` 指定的归约操作进行归约，然后把归约结果写回所有 rank 的 `recvBuf`。

评测拓扑为 `2 * 8` 卡 Ascend 950 仿真环境：

- 2 个 Server，每个 Server 8 个 NPU。
- Server 内 8 个 NPU 为 Full-Mesh 互联。
- Server 间通过 Clos 网络互通。
- 每个 NPU 接入 Clos 网络的带宽约为 Server 内单条直连链路带宽的 8 倍。

工程要求选手只修改以下 3 个文件：

- `include/custom.h`
- `op_host/allreduce.cc`
- `op_kernel_aicpu/exec_op.cc`

本次实现实际修改了：

- `op_host/allreduce.cc`
- `op_kernel_aicpu/exec_op.cc`

`include/custom.h` 未修改。

## 2. 要点解析

AllReduce 可以拆成两个逻辑阶段：

1. 数据交换：让每个 rank 拿到所有 rank 的同一段输入数据。
2. 本地归约：每个 rank 对这些输入数据执行 `SUM`、`PROD`、`MAX`、`MIN` 等归约，写入自己的输出 buffer。

模板中 Host 侧负责资源申请和 AICPU Kernel 下发，Device/AICPU 侧负责通信任务编排：

- Host 侧入口：`HcclAllReduce`。
- AICPU 侧入口：`ExecOp`。
- 通信资源通过 `AlgResourceCtx` 序列化后传给 AICPU。
- HCCL buffer 用作通信中转区。
- Channel 用作 rank 间单边写和 notify 同步。

本次实现优先保证正确性，并做了有限并行化。核心思想是：每个 rank 把自己的输入分片写到所有 peer 的 HCCL buffer 对应槽位，所有 rank 完成写入后，各 rank 在本地 HCCL buffer 上完成归约。

## 3. 本次实现设计

### 3.1 Host 侧资源申请

文件：`op_host/allreduce.cc`

主要实现内容：

- 增加 `sendBuf`、`recvBuf`、`comm`、`stream` 空指针检查。
- 增加 `count * dataTypeSize` 溢出检查。
- 支持识别 HCCL 常见数据类型大小。
- 支持 `HCCL_REDUCE_SUM`、`HCCL_REDUCE_PROD`、`HCCL_REDUCE_MAX`、`HCCL_REDUCE_MIN`。
- 获取 `rankId` 和 `rankSize`，检查 rank 信息合法性。
- 申请 CPU_TS thread，用于 Host/Device 同步。
- 申请 AICPU_TS thread：
  - `rankSize > 1` 时申请 `rankSize - 1` 个 thread。
  - 每个 peer channel 可使用一个独立通信 thread。
- 通过 `HcclRankGraphGetLayers` 和 `HcclRankGraphGetLinks` 查询链路。
- 优先选择 `COMM_PROTOCOL_UBC_CTP` 链路；如果不存在，则回退到第一个非 reserved 链路。
- 对每个 peer 申请一个 channel。
- 通过 `HcclChannelGetHcclBuffer` 获取 peer 侧 HCCL buffer 地址。
- 将 `threads`、`channels`、本端 HCCL buffer 等资源放入 `AlgResourceCtx`，序列化到 AICPU engine context。
- 修正 DFX 注册信息的初始化和注册时序，使其发生在 CPU_TS thread 申请之后。

### 3.2 AICPU 侧通信编排

文件：`op_kernel_aicpu/exec_op.cc`

主要实现内容：

- 校验输入指针、输出指针、HCCL buffer 和 thread 资源。
- 支持 `rankSize == 1` 的退化场景，直接本地拷贝。
- 根据 HCCL buffer 大小进行分块：
  - 每个 rank 在本地 HCCL buffer 中占一个 slot。
  - 单次分片大小不超过 `localBuffer.size / rankSize`。
  - 分片按 128 字节向下对齐。
  - 单次分片上限为 256 MB。
- 每个分片的执行流程：
  1. 主 thread 把当前 rank 的输入分片拷贝到本地 HCCL buffer 的 `myRank` slot。
  2. 本地 thread 同步，确保通信 thread 开始前数据已准备好。
  3. 对每个 peer：
     - 通过 channel notify 做 ACK 同步。
     - 使用 `HcommWriteOnThread` 把本地 slot 写到 peer HCCL buffer 的 `myRank` slot。
     - 使用 DATA_SIGNAL notify 确认写入完成。
  4. 本地 thread 同步，确保所有 peer 通信完成。
  5. 将本地 HCCL buffer 中自己的 slot 拷贝到 `recvBuf` 当前分片。
  6. 遍历其他 rank 的 slot，调用 `HcommLocalReduceOnThread` 归约到 `recvBuf`。

本质上这是一个 allgather + local reduce 的 AllReduce 实现。

### 3.3 Buffer 布局

单个分片内，本地 HCCL buffer 被划分为 `rankSize` 个连续 slot：

```text
slot 0      : rank 0 的当前输入分片
slot 1      : rank 1 的当前输入分片
...
slot N - 1  : rank N - 1 的当前输入分片
```

每个 rank 都把自己的输入分片写入所有 peer 的对应 slot。写入完成后，所有 rank 的本地 HCCL buffer 都包含完整的 `rankSize` 份分片数据，因此可以在本地完成归约。

## 4. 本次实现的优点

- 正确性路径清晰，易于审查。
- 不依赖额外自定义结构，`custom.h` 保持不变。
- 支持大 `count` 分块，避免 HCCL buffer 不足。
- 支持 in-place 风格的 `sendBuf == recvBuf` 场景。
- 每个 peer 一个 channel，并按 peer 数申请 AICPU thread，通信阶段具备一定并行度。
- 链路选择优先使用低层 rank graph，并优先使用 `UBC_CTP` 协议。

## 5. 本次实现的不足之处

- 未在真实 CANN 9.1 评测环境完成完整编译验证。
  - 本地 `ASCEND_HOME_PATH` 未配置时，`bash build.sh` 直接失败。
  - 设置为 `/data/mingwei/Ascend/ascend-toolkit/latest` 后，发现该路径实际指向 8.2.RC1。
  - 本地环境缺少比赛模板需要的头文件，例如 `hccl/hccl_res_expt.h`、`hccl/hccl_res.h`、`hcomm/hcomm_primitives.h`。
- 目前只完成了语法级验证：
  - `op_host/allreduce.cc` 通过 `g++ -std=c++17 -Wall -Werror -fsyntax-only`。
  - `op_kernel_aicpu/exec_op.cc` 通过 `g++ -std=c++17 -Wall -Werror -fsyntax-only`。
- 算法不是拓扑最优。
  - 当前是全 peer 写入，通信复杂度接近 `O(N^2)`。
  - 没有做 Server 内 reduce、跨 Server reduce、再 broadcast 的分层算法。
  - 没有专门利用 Clos 纵向带宽约为单条 Mesh 链路 8 倍这一特性。
- 当前本地规约阶段在主 thread 上串行执行，未并行化 local reduce。
- 对不同数据类型和 `PROD` 等 reduce op 的最终支持依赖底层 HCOMM primitive 的实际能力，需要在 CANN 9.1 A5 环境确认。
- 没有运行多卡仿真正确性测试，也没有性能 benchmark 数据。

## 6. 后续优化方向

- 在 CANN 9.1 + Ascend 950 仿真环境中完成完整编译和多 rank 正确性验证。
- 基于 2 Server * 8 NPU 拓扑实现分层 AllReduce：
  1. Server 内先 reduce-scatter 或 local reduce。
  2. Server 间通过 Clos 做跨 server 交换/归约。
  3. Server 内再 broadcast/allgather。
- 对大数据量实现 pipeline，重叠写入、等待和本地归约。
- 尝试使用 `HcommWriteReduceOnThread` 或 `HcommWriteReduceWithNotifyOnThread` 减少中转 buffer 和本地 reduce 开销。
- 根据数据大小选择不同算法：
  - 小数据量：直接 allgather + local reduce。
  - 大数据量：ring 或 reduce-scatter + allgather。
  - 双 server 场景：分层算法。
