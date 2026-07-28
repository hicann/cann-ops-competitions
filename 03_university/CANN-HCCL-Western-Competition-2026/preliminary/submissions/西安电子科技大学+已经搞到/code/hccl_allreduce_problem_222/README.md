# AllReduce 分层 V8.1 等价精简版

## 版本定位

V8.1从V8派生，不改变4B、512KB、512MB和400MB+4B四个固定尺寸的通信算法和任务图。本版本修复错误恢复路径，删除历史RHD/ring代码，并将Channel查找改为确定下标。

V8保留不动作为重构前基线。

## 等价重构

### Channel固定布局

每rank仍只申请8条唯一对端Channel，序列固定为：

1. 本server内除自身外的7个rank，按local rank升序排列；
2. paired rank固定放在下标7。

Device通过local rank直接计算Channel下标，不再对8条Channel反复执行线性搜索。`CheckResources`同时校验每个下标的`remoteRank`是否符合固定布局。

### 删除历史代码

- 删除未调用的`ExecuteRecursiveHalvingDoubling`。
- 删除`BuildRingOrder`。
- 删除`ringIndex`、`prevRank`和`nextRank`及其序列化字段。
- 删除Recursive Doubling优先排列Channel的逻辑和过期8-ring错误信息。

Device `exec_op.cc`由508行降至429行，Host `allreduce.cc`由249行降至219行。

## 错误恢复修复

- `HcommBatchModeStart`成功后，即使Host notify等待、`ExecOp`或Device notify失败，也统一执行`HcommBatchModeEnd`。
- AICPU入口检查`resCtx`非空、`ctxSize`非零，并验证反序列化结果恰好包含8个thread和8条Channel。
- 复用Host context时检查地址非空且大小至少为`sizeof(ThreadHandle)`。

## 保持不变的算法

- 4B和512KB继续使用V8 smallopt的3+1+3路径。
- 512MB继续使用16个32MiB tile。
- 400MB+4B继续使用16个约25MiB tile。
- 大包仍为8条持久tile-owner lane、每Channel 40个notify。
- context tag：`hccl_custom_allreduce_v8_1_direct_index`。
- 构建目录：`build-v8-1`。
- 本地运行脚本：`../scripts/run-allreduce-hierarchical-v8-1.sh`。

## 本地验证

Release Host/Device双产物构建成功。四个固定尺寸的任务数与V8完全一致：

| 尺寸 | 日志 | tasks | 结果 |
| --- | --- | ---: | --- |
| 4B | `run_logs/allreduce_hierarchical_v8_1_16rank_20260717-120012` | 492 | 三个进程退出码均为0 |
| 512KB | `...-120137` | 656 | 三个进程退出码均为0 |
| 400MB+4B | `...-120348` | 16896 | 三个进程退出码均为0 |
| 512MB | `...-120731` | 16896 | 三个进程退出码均为0 |

CheckerV3仍在环境已知的Host/AICPU local-notify建图处报告`ErrorCode 102`。以上验证证明目录、资源申请和任务生成等价；数值正确性与性能仍以评分平台为准。

## 下一步

V8.1可替代V8进入平台A/B。下一阶段应继续压缩512KB控制面：优先改造定长资源context和AICPU零堆分配反序列化，再评估kernel function/args handle缓存。
