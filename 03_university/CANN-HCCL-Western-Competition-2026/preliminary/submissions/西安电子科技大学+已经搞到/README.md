# HCCL AllReduce AICPU 分层优化

## 团队信息

- 团队名称：已经搞到
- 所属单位：西安电子科技大学
- 参赛阶段：2026 HCCL 通信库创新大赛西部赛区初赛

## 作品信息

- 赛题：AllReduce 集合通信算子
- 通信引擎：AICPU
- 归档版本：`hccl_allreduce_problem_222_hierarchical_v8_1`
- 代码目录：`code/hccl_allreduce_problem_222/`

本实现面向双 Server、每 Server 8 Rank 的拓扑设计分层 AllReduce。小消息采用
3 + 1 + 3 的 Server 内归约、跨 Server 交换和 Server 内分发路径；大消息按
tile 划分，由持久化 owner lane 完成 Reduce-Scatter 与 AllGather，以降低重复
资源申请和控制面开销。

V8.1 在保持 V8 通信任务图不变的基础上，将 Channel 布局固定化，删除未使用的
历史算法代码，并修复批量模式错误路径，确保 `HcommBatchModeStart` 成功后统一
执行 `HcommBatchModeEnd`。

## 提交内容

代码基于官方初赛模板整理，包含完整的 Host 与 AICPU 可构建工程。算法及可靠性
改动主要集中在：

- `include/custom.h`
- `op_host/allreduce.cc`
- `op_kernel_aicpu/aicpu_kernel.cc`
- `op_kernel_aicpu/exec_op.cc`

本归档不包含 `build-v8-1/`、动态库、对象文件或本地运行日志。

## 构建方法

配置 CANN Toolkit 9.1.0 环境后执行：

```bash
cd code/hccl_allreduce_problem_222
bash build.sh
```

构建产物输出至 `build-v8-1/`。

## 验证情况

Release 模式的 Host 与 Device 双产物构建成功。4 B、512 KiB、400 MiB + 4 B
和 512 MiB 四个固定尺寸的任务生成与 V8 基线一致，相关本地进程均正常退出。
数值正确性与性能结果以赛事平台实测为准。详细算法与验证说明见代码目录中的
`README.md`。
