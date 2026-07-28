# AllReduce 714v2 实现说明

## 1. 背景

本文件记录当前 AllReduce 最新实现，以及在 Docker 容器 `hccl-vm` 内使用北极星/HCCL-VM 做编译和仿真的方法。

工程位置：

```bash
/data/mingwei/hccl_allreduce_problem_222_template
```

容器内映射位置：

```bash
/home/workspace/hccl_allreduce_problem_222_template
```

README 中注明选手允许修改的实现文件只有：

- `include/custom.h`
- `op_host/allreduce.cc`
- `op_kernel_aicpu/exec_op.cc`

v2 相比 v1 的最新算法改动只在：

- `op_kernel_aicpu/exec_op.cc`

`op_host/allreduce.cc` 和 `include/custom.h` 本轮没有继续改动。

## 2. v1 到 v2 的变化

v1 的核心路径是 flat allgather + local reduce：

1. 每个 rank 把自己的输入分片写到所有 peer 的 HCCL buffer。
2. 每个 rank 的本地 HCCL buffer 得到所有 rank 的分片。
3. 每个 rank 在本地串行归约所有 slot，写入 `recvBuf`。

v2 保留 v1 路径作为通用 fallback，同时新增了一个针对比赛拓扑的 16 rank、2 server、大消息优化路径。

触发条件：

```cpp
rankSize == 16 && totalBytes >= 64 * 1024
```

不满足该条件时，仍走 v1 的 flat allgather + local reduce。

## 3. 当前 AICPU 侧实现

入口函数仍是：

```cpp
HcclResult ExecOp(const OpParam &param, const AlgResourceCtx &resCtx)
```

主要流程：

1. 校验 `inputPtr`、`outputPtr`、本地 HCCL buffer、AICPU thread。
2. 校验 `rankSize`、`myRank`。
3. 解析并校验 HCCL data type 和 reduce op。
4. `count == 0` 直接返回成功。
5. `rankSize == 1` 时，如果不是 inplace，就本地拷贝 `sendBuf -> recvBuf`。
6. 校验 channel 数量至少覆盖 `rankSize - 1` 个 peer。
7. 根据算法选择 slot 数并计算单次分片大小：
   - v1 fallback：slot 数为 `rankSize`。
   - v2 two-server：slot 数为 `9`。
8. 分片大小按 128 字节向下对齐，并限制单片最大 256 MB。
9. 根据条件进入：
   - `RunTwoServerAllReduce`
   - `RunFlatAllGatherReduce`

当前支持的数据类型大小映射覆盖 int、uint、fp、bf16、fp8、hif8 等常见 HCCL 类型；当前支持的归约操作为：

- `HCCL_REDUCE_SUM`
- `HCCL_REDUCE_PROD`
- `HCCL_REDUCE_MAX`
- `HCCL_REDUCE_MIN`

## 4. v2 two-server 算法

### 4.1 拓扑假设

该优化路径假设 16 个 rank 按如下方式分布：

```text
Server 0: rank 0  - rank 7
Server 1: rank 8  - rank 15
```

每个 rank 的本地编号：

```cpp
localRank = myRank % 8
```

跨 server 配对 rank：

```cpp
crossPeer = (myRank + 8) % 16
```

也就是同 local id 的两张卡互相交换 server 内局部归约结果。

### 4.2 HCCL buffer slot 布局

v1 fallback 使用 `rankSize` 个 slot：

```text
slot 0      : rank 0 当前分片
slot 1      : rank 1 当前分片
...
slot N - 1  : rank N - 1 当前分片
```

v2 two-server 使用 9 个 slot：

```text
slot 0      : 本 server local rank 0 当前分片
slot 1      : 本 server local rank 1 当前分片
...
slot 7      : 本 server local rank 7 当前分片
slot 8      : 对端 server 同 local id rank 的局部归约结果
```

slot 数从 16 降到 9 后，同样大小的 HCCL buffer 可以给每个 slot 分到更大的分片空间，减少大消息时的分片轮数。

### 4.3 单个分片的执行步骤

对每个分片，`RunTwoServerAllReduce` 的流程是：

1. 将当前 rank 的输入分片拷贝到本地 HCCL buffer 的 `localRank` slot。
2. 与同 server 的 7 个 peer 交换分片：
   - 每个 peer 把自己的输入写入对方对应的 local slot。
   - 交换完成后，本 server 8 张卡在各自本地 buffer 中都有 server 内 8 份数据。
3. 在本地把 slot 0 到 slot 7 归约到自己的 `localRank` slot。
4. 只与跨 server 的同 local id peer 交换一次局部归约结果：
   - 对端写入本地 slot 8。
5. 将 slot 8 归约到自己的 `localRank` slot。
6. 将最终结果从 `localRank` slot 拷贝到 `recvBuf` 当前分片。

该路径没有使用 write-reduce 原语，仍使用：

- `HcommWriteOnThread`
- `HcommLocalReduceOnThread`
- channel notify ACK / DATA_SIGNAL 同步

### 4.4 通信量变化

以 16 rank 为例：

v1 flat allgather：

- 每个 rank 写 15 个 peer。
- 其中同 server 7 个 peer，跨 server 8 个 peer。
- 每个 rank 跨 server fanout 为 8。
- 本地 HCCL buffer 需要 16 个 slot。

v2 two-server 大消息路径：

- 每个 rank 写 8 个 peer。
- 其中同 server 7 个 peer，跨 server 1 个 peer。
- 每个 rank 跨 server fanout 从 8 降到 1。
- 本地 HCCL buffer 从 16 个 slot 降到 9 个 slot。

因此 v2 主要减少跨 server all-to-all，同时减少 HCCL buffer 分片次数。对 2 server * 8 NPU 的比赛拓扑，大消息通信时间应比 v1 更有优势。

## 5. 代码结构变化

`op_kernel_aicpu/exec_op.cc` 中新增或调整的关键逻辑：

- `TWO_SERVER_RANK_SIZE = 16`
- `RANKS_PER_SERVER = 8`
- `TWO_SERVER_SLOT_NUM = 9`
- `TWO_SERVER_ALGO_MIN_BYTES = 64KB`
- `ShouldUseTwoServerAlgo`
- `GetLocalServerPeers`
- `GetCrossServerPeer`
- `BuildExchangeEntries`
- `ExchangeWithEntries`
- `ReduceSlotsToSlot`
- `RunFlatAllGatherReduce`
- `RunTwoServerAllReduce`

其中 `BuildExchangeEntries` 会在分片循环外预先把 peer rank 映射到 channel 和 AICPU thread，避免每个分片重复查找 channel。

`ExchangeWithEntries` 复用预先构造好的通信表执行写入和 notify 同步。

## 6. 当前不足

- two-server 路径只覆盖 `rankSize == 16 && totalBytes >= 64KB`。
- rank 排布假设为 `0..7` 在 server 0，`8..15` 在 server 1。
- 本地 reduce 仍在主 AICPU thread 上串行执行。
- 暂未使用 `HcommWriteReduceOnThread` 或 write-reduce-with-notify 类原语。
- 小消息仍使用 flat fallback，未专门做低延迟优化。
- 当前文档记录的是编译通过后的实现状态，尚未记录完整性能 benchmark 数据。

## 7. Docker 内北极星环境

当前使用的容器：

```bash
docker exec -it hccl-vm bash -l
```

容器内已配置的关键路径：

```bash
export HCCL_VM_WORKSPACE=/opt/hccl-vm-workspace
export HCCL_VM_INSTALL_DIR=/opt/hccl-vm-workspace/hcomm/test/hccl_vm/hccl_vm_install
export ASCEND_HOME_PATH=/opt/hccl-vm-workspace/Ascend/cann-9.1.0
```

建议进入容器后先加载统一环境：

```bash
source /etc/profile.d/hccl-vm.sh
```

该脚本会设置：

- CANN 9.1 环境变量
- `PATH`
- `LD_LIBRARY_PATH`
- `RANK_TABLE_FILE`
- `HCCL_OP_EXPANSION_MODE=AI_CPU`

本项目是 AICPU 展开模式，运行北极星前需要确认：

```bash
echo $HCCL_OP_EXPANSION_MODE
```

期望输出：

```bash
AI_CPU
```

## 8. 在容器内重新编译本项目

进入容器：

```bash
docker exec -it hccl-vm bash -l
```

加载环境并编译：

```bash
source /etc/profile.d/hccl-vm.sh
cd /home/workspace/hccl_allreduce_problem_222_template
bash build.sh
```

当前已验证该命令可以编译通过，并生成：

```bash
build/lib64/libhccl.so
build/lib64/libhccl_device.so
build/include/hccl.h
```

如果要让 `hccl_test` 使用最新实现，需要把构建产物替换到 CANN/HCCL-VM 使用的位置：

```bash
cp -f build/include/hccl.h ${ASCEND_HOME_PATH}/x86_64-linux/include/hccl/
cp -f build/lib64/libhccl.so ${ASCEND_HOME_PATH}/x86_64-linux/lib64/
cp -f build/lib64/libhccl_device.so ${HCCL_VM_INSTALL_DIR}/lib/aarch64/
```

## 9. 在容器内启动北极星

进入北极星 bin 目录：

```bash
source /etc/profile.d/hccl-vm.sh
cd ${HCCL_VM_INSTALL_DIR}/bin
```

启动比赛拓扑：

```bash
./hccl-vm start ascend950_cluster_4_server_competition.yaml
```

如果只做快速流程验证、希望减少内存占用，可以加 `--check-only`：

```bash
./hccl-vm start ascend950_cluster_4_server_competition.yaml --check-only
```

注意：`--check-only` 会复用大块内存，适合快速跑通流程；要确认真实数据正确性时，应优先不用该参数。

启动后会进入北极星交互环境，提示符类似：

```bash
(hvm)$>
```

v2 two-server 路径需要 16 rank 通信域，使用 `128` 这个 topo meta：

```bash
(hvm)$> hccl-vm mock-comm 128
```

`128.yaml` 对应：

```text
podNum  = 1
serNum  = 2
rankNum = 16
server0 = 8 ranks
server1 = 8 ranks
```

运行 AllReduce 小规模正确性用例：

```bash
(hvm)$> mpirun --allow-run-as-root --oversubscribe -np 16 ${ASCEND_HOME_PATH}/tools/hccl_test/bin/all_reduce_test -b 65536 -e 65536 -d int32 -o sum -w 0 -n 1 -c 1 -p 8
```

参数含义：

- `-np 16`：启动 16 个进程，对应 16 rank。
- `-b 65536 -e 65536`：只测试 64KB，刚好触发 v2 大消息路径。
- `-d int32`：数据类型。
- `-o sum`：归约操作。
- `-w 0`：不做 warmup。
- `-n 1`：跑 1 轮。
- `-c 1`：开启 quiet result check。
- `-p 8`：单 server 使用 8 个 NPU。

运行更大的消息可以提高 `-b` 和 `-e`，例如：

```bash
(hvm)$> mpirun --allow-run-as-root --oversubscribe -np 16 ${ASCEND_HOME_PATH}/tools/hccl_test/bin/all_reduce_test -b 1048576 -e 1048576 -d int32 -o sum -w 1 -n 10 -c 1 -p 8
```

运行 checker：

```bash
(hvm)$> hccl-vm plugin run @checker
```

退出北极星：

```bash
(hvm)$> exit
```

## 10. 常用排查命令

确认容器状态：

```bash
docker ps --filter name=^/hccl-vm$
```

确认北极星命令可用：

```bash
source /etc/profile.d/hccl-vm.sh
which hccl-vm
hccl-vm --help
```

确认 AICPU 模式：

```bash
echo $HCCL_OP_EXPANSION_MODE
```

确认通信域配置存在：

```bash
ls ${HCCL_VM_INSTALL_DIR}/config/topo_meta/128.yaml
```

确认 `all_reduce_test` 存在：

```bash
ls ${ASCEND_HOME_PATH}/tools/hccl_test/bin/all_reduce_test
```

如果进入北极星后需要重置状态：

```bash
hccl-vm reset
```

## 11. 后续优化方向

- 对小消息增加更低延迟的路径，避免 full allgather 的同步成本。
- 评估 `HcommWriteReduceOnThread`，减少本地中转和串行 reduce。
- 将 server 内 8 slot reduce 做并行化，降低 AICPU 主 thread 归约时间。
- 继续根据消息大小做算法切换阈值调参。
- 在北极星中记录 v1/v2 的通信时间对比，并按数据类型、消息大小分组。
