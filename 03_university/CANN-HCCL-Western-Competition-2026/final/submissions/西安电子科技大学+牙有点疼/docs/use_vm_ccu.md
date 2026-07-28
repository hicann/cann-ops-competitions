# HCCL-VM CCU 仿真环境使用说明

## 1. 当前环境

当前使用的 Docker 容器为：

```text
hccl-vm-ccu246
```

容器环境：

| 项目 | 配置 |
|---|---|
| 基础系统 | Ubuntu 22.04 |
| 环境镜像 | `hccl-vm-ccu246-full9:20260720` |
| CANN | 9.1.0 |
| CCU 模式 | `HCCL_OP_EXPANSION_MODE=CCU_MS` |
| 网络 | host |
| 权限 | privileged |
| `/dev/shm` | 64GB |

`hccl-vm-ccu246-full9:20260720` 是基于原始 `ubuntu:22.04` 保存的仿真环境快照。原始 Ubuntu 镜像只有基础系统，不包含 CANN、OpenMPI 和 HCCL-VM。

当前 bind mounts：

```text
/data/mingwei/hccl_allreduce_ccu_problem_246_template
  -> /data/mingwei/hccl_allreduce_ccu_problem_246_template
  -> /home/workspace/hccl_allreduce_ccu_problem_246_template

/data/mingwei/hcomm
  -> /data/mingwei/hcomm
  -> /home/workspace/hcomm
```

工程和 HCOMM 源码都是 bind mount，宿主机修改会立即反映到容器中。

## 2. 创建容器

当容器不存在时，可使用以下命令创建：

```bash
docker run -dit \
  --name hccl-vm-ccu246 \
  --network host \
  --privileged \
  --security-opt label=disable \
  --shm-size=64g \
  -e HCCL_OP_EXPANSION_MODE=CCU_MS \
  --mount type=bind,src=/data/mingwei/hccl_allreduce_ccu_problem_246_template,dst=/data/mingwei/hccl_allreduce_ccu_problem_246_template \
  --mount type=bind,src=/data/mingwei/hccl_allreduce_ccu_problem_246_template,dst=/home/workspace/hccl_allreduce_ccu_problem_246_template \
  --mount type=bind,src=/data/mingwei/hcomm,dst=/data/mingwei/hcomm \
  --mount type=bind,src=/data/mingwei/hcomm,dst=/home/workspace/hcomm \
  hccl-vm-ccu246-full9:20260720 \
  bash
```

环境镜像中的旧 profile 可能强制设置 `AI_CPU`。新建容器后应验证登录 shell：

```bash
docker exec hccl-vm-ccu246 bash -lc \
  'echo ${HCCL_OP_EXPANSION_MODE}'
```

如果输出不是 `CCU_MS`，可以在每次登录后显式执行：

```bash
source /etc/profile.d/hccl-vm.sh
export HCCL_OP_EXPANSION_MODE=CCU_MS
```

当前 `hccl-vm-ccu246` 已调整 profile，登录 shell 默认输出 `CCU_MS`。

## 3. `/dev/shm` 配置

### 3.1 为什么 8GB 不够

旧容器的 `ShmSize` 是 8589934592，即创建时设置了 8GB。该值是 Docker 容器的独立 tmpfs 上限，不代表宿主机总内存。

HCCL-VM 会为每个模拟 rank 建立设备内存后端。以 `2x8 / 16 rank / 512MB` 为例，共享内存需求会叠加：

- 每个 rank 的输入和输出 buffer
- HCCL 通信 buffer 和 scratch
- runner 的模拟内存后端
- 其他控制和元数据区域

因此总需求远大于单个 512MB。8GB 可以运行小消息，但不足以覆盖 12/16 rank 的正式大消息数值测试。

### 3.2 如何增大

推荐在创建容器时使用：

```bash
--shm-size=64g
```

Docker 一般不能通过 `docker update` 修改已有容器的 `/dev/shm` 大小，需要重新创建容器。也可以使用 `--ipc=host` 直接共享宿主机 `/dev/shm`，但隔离性更差，推荐保留 private IPC 并显式设置大小。

检查当前值：

```bash
docker exec hccl-vm-ccu246 df -h /dev/shm
docker inspect hccl-vm-ccu246 \
  --format '{{.HostConfig.ShmSize}}'
```

64GB 对当前九个正式功能用例已经足够。

## 4. 启动和进入容器

查看状态：

```bash
docker ps --filter name=^/hccl-vm-ccu246$
```

如果容器已停止：

```bash
docker start hccl-vm-ccu246
```

进入登录 shell：

```bash
docker exec -it hccl-vm-ccu246 bash -l
```

进入后检查环境：

```bash
echo ${HCCL_OP_EXPANSION_MODE}
echo ${ASCEND_HOME_PATH}
echo ${HCCL_VM_INSTALL_DIR}
command -v hccl-vm
df -h /dev/shm
```

期望输出包括：

```text
CCU_MS
/opt/hccl-vm-workspace/Ascend/cann-9.1.0
/opt/hccl-vm-workspace/hcomm/test/hccl_vm/hccl_vm_install
```

## 5. 编译并安装算子

在容器登录 shell 中执行：

```bash
cd /home/workspace/hccl_allreduce_ccu_problem_246_template
bash build.sh
```

工程默认使用 Release 模式，并启用 `-Wall -Werror`。构建产物为：

```text
build/lib64/libhccl.so
build/include/hccl.h
```

将最新库安装到当前容器的 CANN 环境：

```bash
cp -f build/lib64/libhccl.so \
  ${ASCEND_HOME_PATH}/x86_64-linux/lib64/libhccl.so
```

核对源文件和安装文件一致：

```bash
sha256sum \
  build/lib64/libhccl.so \
  ${ASCEND_HOME_PATH}/x86_64-linux/lib64/libhccl.so
```

修改源码后，应先退出正在运行的 HCCL-VM 会话，再重新编译、复制库和启动 HCCL-VM，避免进程继续使用旧动态库。

## 6. 启动 HCCL-VM

在容器登录 shell 中执行：

```bash
cd ${HCCL_VM_INSTALL_DIR}/bin
hccl-vm start ascend950_cluster_4_server_competition.yaml
```

启动成功后提示符变为：

```text
(hvm)$>
```

安装 runner 并确认插件状态：

```bash
hccl-vm plugin install @runner
hccl-vm plugin list
```

期望 checker 和 runner 都是 `RUNNING`：

```text
checker  ... RUNNING
runner   ... RUNNING
```

> 重要：`mock-comm` 和 `mpirun` 测试命令必须在同一个 `(hvm)$>` 会话内执行。不要从另一个 `docker exec` shell 直接启动测试，否则测试进程没有 HCCL-VM 注入的模拟设备环境，通常会报 `The number of device is 0`。

## 7. 拓扑参数

| 比赛拓扑 | topo meta | rank 数 | `mpirun -np` | `all_reduce_test -p` |
|---|---|---:|---:|---:|
| `2x8` | `128` | 16 | 16 | 8 |
| `4x1` | `141` | 4 | 4 | 1 |
| `8+4` | `12_8_4` | 12 | 12 | 8 |

每个独立测试前都应重新执行对应的 `hccl-vm mock-comm`，清理上一用例的通信域和共享内存。

## 8. 正式功能用例

数据量：

| 名称 | 字节数 |
|---|---:|
| 512KB | 524288 |
| 512MB | 536870912 |
| 400MB+4B | 419430404 |

以下命令都在 `(hvm)$>` 中执行。

### 8.1 `2x8`

```bash
for BYTES in 524288 536870912 419430404; do
  hccl-vm mock-comm 128
  /usr/bin/timeout 600s mpirun \
    --allow-run-as-root --oversubscribe -np 16 \
    "${ASCEND_HOME_PATH}/tools/hccl_test/bin/all_reduce_test" \
    -b "${BYTES}" -e "${BYTES}" \
    -d fp32 -o sum -w 0 -n 1 -c 1 -p 8
done
```

### 8.2 `4x1`

```bash
for BYTES in 524288 536870912 419430404; do
  hccl-vm mock-comm 141
  /usr/bin/timeout 600s mpirun \
    --allow-run-as-root --oversubscribe -np 4 \
    "${ASCEND_HOME_PATH}/tools/hccl_test/bin/all_reduce_test" \
    -b "${BYTES}" -e "${BYTES}" \
    -d fp32 -o sum -w 0 -n 1 -c 1 -p 1
done
```

### 8.3 `8+4`

```bash
for BYTES in 524288 536870912 419430404; do
  hccl-vm mock-comm 12_8_4
  /usr/bin/timeout 600s mpirun \
    --allow-run-as-root --oversubscribe -np 12 \
    "${ASCEND_HOME_PATH}/tools/hccl_test/bin/all_reduce_test" \
    -b "${BYTES}" -e "${BYTES}" \
    -d fp32 -o sum -w 0 -n 1 -c 1 -p 8
done
```

每个用例的结果表格最后一列应为：

```text
success
```

`-c 1` 会执行数值校验。未安装 runner 时输出 buffer 不会被模拟执行更新，不能用 checker 成功代替数值校验。

## 9. CheckerV3

运行一个功能用例后，不要重新执行 `mock-comm`，直接在 `(hvm)$>` 中运行：

```bash
hccl-vm plugin run @checker
```

成功结果包括：

```text
CheckerV3 stage finished, stage=SingleTaskCheck, status=success
CheckerV3 stage finished, stage=MemConflict, status=success
CheckerV3 stage finished, stage=SemanticCheck, status=success
op[1] Checker Success
```

推荐至少检查：

- `2x8 / 512KB`：覆盖完整双 die 基本任务图
- `8+4 / 400MB+4B`：覆盖非均匀拓扑、多分段和 4B 尾段

Checker 日志可能出现 CCU local-post 未消费 warning。当前实现中对应的是每个 CCU mission 自动生成的启动序言，不影响三个 CheckerV3 阶段和最终成功结论。

## 10. 当前验证结果

当前挂载源码在 `hccl-vm-ccu246` 中完整重编译后，九个正式组合均通过：

| 拓扑 | 512KB | 512MB | 400MB+4B |
|---|---|---|---|
| `2x8` | success | success | success |
| `4x1` | success | success | success |
| `8+4` | success | success | success |

CheckerV3 的基本任务图和大消息多分段任务图均通过。

## 11. 退出和清理

退出 `(hvm)$>`：

```bash
exit
```

该命令会正常停止 checker、runner 和 HCCL-VM host，但 Docker 容器仍保持运行。

退出容器登录 shell：

```bash
exit
```

停止容器：

```bash
docker stop hccl-vm-ccu246
```

## 12. 常见问题

### 12.1 `The number of device is 0`

原因通常是从普通 Docker shell 直接运行了 `mpirun`，没有使用 `hccl-vm start` 进入的 `(hvm)$>` 会话。重新启动 HCCL-VM，并在该会话内运行 `mock-comm` 和测试。

### 12.2 所有输出元素都校验失败

先运行：

```bash
hccl-vm plugin list
```

确认 runner 为 `RUNNING`。只有 checker 而没有 runner 时，任务会被收集但不会执行数据搬运。

### 12.3 `No space left on device` 或共享内存创建失败

检查：

```bash
df -h /dev/shm
```

如果上限仍是 8GB，需要使用 `--shm-size=64g` 重新创建容器。仅增加 Docker memory limit 不会自动增加 `/dev/shm`。

### 12.4 修改代码后结果没有变化

确认重新执行了：

```bash
bash build.sh
cp -f build/lib64/libhccl.so \
  ${ASCEND_HOME_PATH}/x86_64-linux/lib64/libhccl.so
```

然后退出并重新启动 HCCL-VM。可以用 `sha256sum` 排除加载旧库。

### 12.5 带宽数值看起来异常

HCCL-VM 的事件耗时是模拟值，`aveg_time` 和 `alg_bandwidth` 不能代表真实硬件性能。仿真环境只用于数值正确性、任务图、同步和内存冲突检查。
