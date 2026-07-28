# 北极星 Docker 使用说明

本文记录在 `hccl-vm` Docker 容器中编译本 AllReduce 工程，并使用 HCCL-VM/北极星验证数值正确性、
任务图和重复迭代稳定性的方法。HCCL-VM 不提供可信的设备性能计时，真实性能必须在正式评测环境或
真实设备上验证。

## 1. 基本路径

宿主机工程路径：

```bash
/data/mingwei/hccl_allreduce_problem_222_template
```

容器内工程路径：

```bash
/home/workspace/hccl_allreduce_problem_222_template
```

容器名：

```bash
hccl-vm
```

容器内关键环境变量：

```bash
HCCL_VM_WORKSPACE=/opt/hccl-vm-workspace
HCCL_VM_INSTALL_DIR=/opt/hccl-vm-workspace/hcomm/test/hccl_vm/hccl_vm_install
ASCEND_HOME_PATH=/opt/hccl-vm-workspace/Ascend/cann-9.1.0
HCCL_OP_EXPANSION_MODE=AI_CPU
```

进入容器：

```bash
docker exec -it hccl-vm bash -l
```

进入后建议先确认环境：

```bash
source /etc/profile.d/hccl-vm.sh
which hccl-vm
echo ${HCCL_OP_EXPANSION_MODE}
```

期望 `HCCL_OP_EXPANSION_MODE` 为：

```text
AI_CPU
```

## 2. 编译并替换库

在容器中编译：

```bash
source /etc/profile.d/hccl-vm.sh
cd /home/workspace/hccl_allreduce_problem_222_template
bash build.sh
```

生成物：

```bash
build/include/hccl.h
build/lib64/libhccl.so
build/lib64/libhccl_device.so
```

让 `hccl_test` 和 HCCL-VM 使用最新实现：

```bash
cp -f build/include/hccl.h ${ASCEND_HOME_PATH}/x86_64-linux/include/hccl/
cp -f build/lib64/libhccl.so ${ASCEND_HOME_PATH}/x86_64-linux/lib64/
cp -f build/lib64/libhccl_device.so ${HCCL_VM_INSTALL_DIR}/lib/aarch64/
```

替换后建议确认时间戳：

```bash
ls -l ${ASCEND_HOME_PATH}/x86_64-linux/lib64/libhccl.so
ls -l ${HCCL_VM_INSTALL_DIR}/lib/aarch64/libhccl_device.so
```

## 3. 启动北极星

比赛拓扑使用 4 server cluster 配置，通信域使用 2 server * 8 rank 的 `128` topo meta。

启动 HCCL-VM：

```bash
source /etc/profile.d/hccl-vm.sh
cd ${HCCL_VM_INSTALL_DIR}/bin
./hccl-vm start ascend950_cluster_4_server_competition.yaml
```

启动后进入交互 shell，提示符类似：

```text
(hvm)$>
```

Runner 插件默认关闭。需要检查数值正确性时，必须先在 `(hvm)$>` 内安装 runner，并确认 runner 和
checker 都处于 `RUNNING`：

```bash
hccl-vm plugin install @runner
hccl-vm plugin list
```

然后初始化 16 rank 通信域：

```bash
hccl-vm mock-comm 128
```

注意：

- 未安装 runner 时，HCCL-VM 只收集任务而不执行任务。此时 checker 仍可能成功，但 `all_reduce_test -c 1`
  会读取到未更新的输出并报错。
- `mock-comm` 必须在 `hccl-vm start ...` 进入的 `(hvm)$>` shell 中执行。
- 直接在普通容器 shell 执行 `hccl-vm mock-comm 128` 会缺少 cluster 上下文，典型报错是尝试读取 `/superpod0/server0/topo.json`。
- runner 只在当前 HCCL-VM 会话内有效。退出并重新启动北极星后，需要重新执行
  `hccl-vm plugin install @runner`。
- 重启北极星时先在 `(hvm)$>` 输入 `exit`，等待 checker 和 runner 正常退出，再重新执行
  `hccl-vm start ...`。

## 4. 正确性仿真

正确性必须使用 normal 模式，也就是 `start` 时不要加 `--check-only`，并且必须已经安装 runner。

建议先确认插件状态，然后在每个独立用例前重新初始化通信域：

```bash
hccl-vm plugin list
hccl-vm mock-comm 128
```

`plugin list` 至少应包含：

```text
checker  ...  RUNNING
runner   ...  RUNNING
```

4B 功能用例：

```bash
mpirun --allow-run-as-root --oversubscribe -np 16 \
  ${ASCEND_HOME_PATH}/tools/hccl_test/bin/all_reduce_test \
  -b 4 -e 4 -d fp32 -o sum -w 0 -n 1 -c 1 -p 8
```

512KB 功能用例：

```bash
mpirun --allow-run-as-root --oversubscribe -np 16 \
  ${ASCEND_HOME_PATH}/tools/hccl_test/bin/all_reduce_test \
  -b 524288 -e 524288 -d fp32 -o sum -w 0 -n 1 -c 1 -p 8
```

512MB 功能用例：

```bash
mpirun --allow-run-as-root --oversubscribe -np 16 \
  ${ASCEND_HOME_PATH}/tools/hccl_test/bin/all_reduce_test \
  -b 536870912 -e 536870912 -d fp32 -o sum -w 0 -n 1 -c 1 -p 8
```

400MB + 4B 功能用例：

```bash
mpirun --allow-run-as-root --oversubscribe -np 16 \
  ${ASCEND_HOME_PATH}/tools/hccl_test/bin/all_reduce_test \
  -b 419430404 -e 419430404 -d fp32 -o sum -w 0 -n 1 -c 1 -p 8
```

关注输出中的：

```text
check_result: success
```

如果出现 `check result failed` 或 `total err is ...`：

1. 先执行 `hccl-vm plugin list`，确认 runner 为 `RUNNING`。
2. 如果错误数恰好等于全部 FP32 元素数，例如 4B 报 1 个错误、512KB 报 131072 个错误，优先检查
   runner 是否未安装或已异常退出。这通常表示整块输出没有执行更新。
3. 只有在 normal 模式、runner 正常运行且 VM 会话干净的前提下，才按算法数值错误继续排查。

当前 v5 已按正确 runner 流程复验：4B、512KB 和 2MB hierarchical 用例均为
`check_result: success`。

## 5. Checker 仿真

`all_reduce_test` 运行后，可以执行 checker 检查任务图：

```bash
hccl-vm plugin run @checker
```

成功时会看到类似：

```text
[RunChecker] op[0] Checker Success.
```

checker 主要看任务图结构、内存冲突、死锁等问题。当前 checker 对同一 notify 资源多组 record/wait 的“多打一”问题不一定能完全拦截，因此算法里仍要自查 notify 时序。

## 6. 重复迭代与性能结论

### 6.1 HCCL-VM 不提供真实性能数据

当前 HCCL-VM 的事件计时 stub 位于：

```text
/data/mingwei/hcomm/test/hccl_vm/src/proxy/aclrt_event_stub.cc
```

其中 `aclrtEventElapsedTime()` 固定返回 1ms：

```cpp
*ms = 1;
```

因此 `all_reduce_test` 输出中的时间和带宽不能用于比较算法性能：

```text
data_size(Bytes): | aveg_time(us): | alg_bandwidth(GB/s): | check_result:
```

例如同一个 512KB 实现，`-n 1` 会显示 `1000 us`，`-n 2` 会显示 `500 us`。`alg_bandwidth`
只是根据这个固定时间计算出的派生值，不代表链路、DMA、规约或任务下发性能。

HCCL-VM 中不同消息大小、不同算法版本或不同迭代次数的 `aveg_time`、`alg_bandwidth` 不可横向比较。
真实性能及性能排名必须以正式评测环境或真实设备数据为准。

### 6.2 重复迭代稳定性测试

以下命令仍有价值，但用途是检查重复调用、engine context 复用、notify 计数和数值稳定性，不是性能测量。
建议保留 `-c 1`。

512KB 重复迭代：

```bash
mpirun --allow-run-as-root --oversubscribe -np 16 \
  ${ASCEND_HOME_PATH}/tools/hccl_test/bin/all_reduce_test \
  -b 524288 -e 524288 -d fp32 -o sum -w 3 -n 20 -c 1 -p 8
```

512MB 重复迭代：

```bash
mpirun --allow-run-as-root --oversubscribe -np 16 \
  ${ASCEND_HOME_PATH}/tools/hccl_test/bin/all_reduce_test \
  -b 536870912 -e 536870912 -d fp32 -o sum -w 3 -n 10 -c 1 -p 8
```

400MB + 4B 重复迭代：

```bash
mpirun --allow-run-as-root --oversubscribe -np 16 \
  ${ASCEND_HOME_PATH}/tools/hccl_test/bin/all_reduce_test \
  -b 419430404 -e 419430404 -d fp32 -o sum -w 3 -n 10 -c 1 -p 8
```

记录稳定性结果时建议保存日志，并只把 `check_result` 和错误日志作为有效结论：

```bash
mpirun --allow-run-as-root --oversubscribe -np 16 \
  ${ASCEND_HOME_PATH}/tools/hccl_test/bin/all_reduce_test \
  -b 524288 -e 524288 -d fp32 -o sum -w 3 -n 20 -c 1 -p 8 | tee allreduce_512kb.log
```

## 7. check-only 模式

仅用于快速 checker 或大规模任务图检查：

```bash
./hccl-vm start ascend950_cluster_4_server_competition.yaml --check-only
```

限制：

- 200MB 到 4GB 的大块内存会复用同一块共享区。
- buffer 内容不保证正确。
- 不适合 `all_reduce_test -c 1` 的数值正确性验证。
- 不适合作为最终性能结论。

如果目标是验证真实 `check_result`，不要加 `--check-only`，并确保 runner 已安装。无论 normal 还是
check-only 模式，HCCL-VM 输出的 `alg_bandwidth` 都不是可信的设备性能数据。

## 8. 常见问题

### 8.1 `mock-comm` 读 `/superpod0/server0/topo.json`

原因：在普通 shell 里直接执行了 `hccl-vm mock-comm 128`，没有先通过 `hccl-vm start ...` 初始化 cluster。

修复：

```bash
cd ${HCCL_VM_INSTALL_DIR}/bin
./hccl-vm start ascend950_cluster_4_server_competition.yaml
(hvm)$> hccl-vm mock-comm 128
```

### 8.2 topo/ranktable 初始化失败

先确认通信域不超过 cluster：

```bash
ls ${HCCL_VM_INSTALL_DIR}/config/cluster/ascend950_cluster_4_server_competition.yaml
ls ${HCCL_VM_INSTALL_DIR}/config/topo_meta/128.yaml
```

`128.yaml` 应为：

```text
1 pod, 2 servers, 16 ranks, each server has 8 ranks
```

初始化成功后可检查 `ranktable.json`：

```bash
python3 - <<'PY'
import json, os
p = os.path.join(os.environ["HCCL_VM_INSTALL_DIR"], "data/ranktable.json")
with open(p) as f:
    data = json.load(f)
for rank in data["rank_list"]:
    print("rank", rank["rank_id"])
    for level in rank.get("level_list", []):
        print(" ", level.get("net_instance_id"), len(level.get("rank_addr_list", [])))
PY
```

当前修正后的 16 rank 期望形态：

```text
rank 0-7:  superPod0_rack0 = 7, az0 = 1
rank 8-15: superPod0_rack1 = 7, az0 = 1
```

### 8.3 `/dev/shm` 空间不足

normal 模式跑 512MB 或 400MB+4B 会使用较多共享内存。检查：

```bash
df -h /dev/shm
```

如果只是 checker 任务图检查，可以使用 `--check-only`。如果要验证数值正确性，必须释放共享内存或
扩大 `/dev/shm` 后使用 normal 模式并安装 runner。真实性能不能通过 HCCL-VM 判断。

### 8.4 退出与重启

退出当前 HCCL-VM：

```bash
(hvm)$> exit
```

如果状态异常，可重新进入后执行：

```bash
hccl-vm reset
```

`hccl-vm reset` 主要清理数据库表，不等价于结束遗留的 QEMU device 进程。退出后可在普通容器 shell 检查：

```bash
ps -eo pid,ppid,stat,etime,args | grep 'qemu-aarch64-static.*/device' | grep -v grep
```

如果仍有长时间运行的孤立 device 进程，先确认没有其他人在使用该容器，再重启容器获得干净环境。

### 8.5 全部元素都报错

典型表现：

```text
4B:     total err is 1
512KB:  total err is 131072
```

首先检查 runner：

```bash
hccl-vm plugin list
hccl-vm plugin install @runner
```

安装 runner 后需要重新执行 `hccl-vm mock-comm 128` 和测试用例。checker Success 不能替代 runner 的
真实数值执行结果。

### 8.6 时间固定为 1000 us 或随迭代次数反比变化

这是 HCCL-VM 事件计时 stub 的预期行为，不是算法真实性能。不要使用该时间或派生的
`alg_bandwidth` 判断优化是否生效；只检查 `check_result`、runner 错误和 checker 结果。
