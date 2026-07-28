# HCCL AllReduce CCU 优化

## 团队信息

- 团队名称：已经搞到
- 所属单位：西安电子科技大学
- 参赛阶段：2026 HCCL 通信库创新大赛西部赛区决赛

## 作品信息

- 赛题：AllReduce 集合通信算子
- 通信引擎：CCU
- 代码目录：`code/hccl_allreduce_ccu_problem_246/`

本实现针对赛题给定的 4 Rank、12 Rank 和 16 Rank 三种拓扑进行 AllReduce
优化。大消息采用按 Rank 分片的 Reduce-Scatter、Combine 与 AllGather 路径；
512 KiB 数据点为不同拓扑注册独立 CCU 图，以减少 Host launch、串行同步和无效
数据搬运。其他小消息与大消息继续使用已经过回归验证的稳定路径。

## 提交内容

代码基于官方决赛模板整理，保留完整可构建工程。按照赛题约束，算法改动集中在：

- `include/custom.h`
- `op_host/allreduce.cc`
- `op_host/exec_op.h`
- `op_host/exec_op.cc`
- `op_kernel_ccu/ccu_kernel.h`
- `op_kernel_ccu/ccu_kernel.cc`

本归档不包含 `build/`、动态库、对象文件或本地运行日志。

## 构建方法

配置 CANN Toolkit 9.1.0 环境后执行：

```bash
cd code/hccl_allreduce_ccu_problem_246
bash build.sh
```

格式化代码：

```bash
bash build.sh --format
```

## 验证情况

三个拓扑的 512 KiB 数据点已通过本地 HCCL VM 的资源、同步、数值语义与
checker 检查；4 B 和 2 MiB + 4 B 回归点也已通过。性能结果以赛事平台实测为准。
更详细的算法设计与验证说明见代码目录中的 `README.md`。
