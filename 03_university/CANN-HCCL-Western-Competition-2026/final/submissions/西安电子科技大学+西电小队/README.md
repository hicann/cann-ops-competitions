# 2026 HCCL通信库创新大赛（西部赛区）决赛代码归档

## 团队信息

- 学校：西安电子科技大学
- 队伍：西电小队
- 提交账号：YUIOP_xdjce

## 作品信息

- 阶段：决赛
- 题目：`hccl_allreduce_ccu`
- 算子：AllReduce
- 通信引擎：CCU

## 代码目录

`code/hccl_allreduce_ccu/` 保留了 CANNJudge 决赛空工程的完整目录结构；其中包含构建脚本、CMake 文件、公共头文件、Host 侧资源与任务编排代码，以及 CCU Kernel 代码。

## 实现概述

实现按通信域规模、拓扑和数据量选择通信路径。控制面申请并序列化 CCU 通信资源，数据面通过 CCU 的 Copy、Reduce、Record 和 Wait 任务完成确定性的 AllReduce。实现保留每个 Die 的独立通信资源和缓冲区分片，避免不必要的跨 Die 串行化，并针对小数据量和大数据量分别选择低启动开销与高带宽利用率路径。

## 构建与验证

在与赛题一致的 CANN/HCCL 环境中进入 `code/hccl_allreduce_ccu/` 后执行：

```bash
bash build.sh
```

该归档版本对应决赛期间功能验证通过的稳定版本，线上功能测试为 18/18 通过。
