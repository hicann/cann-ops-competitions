# 2026 HCCL通信库创新大赛粤港澳赛区决赛代码归档

## 团队信息

- 队伍名称：OK
- GitCode账号：`halo1314`
- 参赛赛区：粤港澳赛区
- 参赛阶段：决赛

本目录使用正式队伍名称 `OK`，GitCode账号为 `halo1314`。

## 作品信息

- 题目：`hccl_reducescatter_ccu_problem_265`
- 算子：ReduceScatter
- 通信引擎：CCU
- 代码目录：[`code/hccl_reducescatter_ccu_problem_265`](code/hccl_reducescatter_ccu_problem_265)

## 实现概述

实现面向 Ascend 950 仿真环境中的 2×8、4×1 和 8+4 通信域，使用 CCU 完成
ReduceScatter。Host 侧按通信域和数据规模申请、序列化通信资源并下发任务，CCU
侧按固定顺序执行跨 rank 读取、本地规约、结果写回和同步，保证浮点输入下的结果
确定性。

归档版本为决赛期间功能稳定的 v9 基线版本：保留经过线上验证的双 Die 资源分配、
大消息分块、固定规约顺序和跨调用同步清理逻辑；没有包含后续出现功能或性能回退
的实验性版本。

## 构建

在与赛题一致的 CANN 9.1.0 环境中执行：

```bash
cd code/hccl_reducescatter_ccu_problem_265
source /usr/local/Ascend/cann-9.1.0/set_env.sh
bash build.sh
```

若 CANN 安装在其他目录，请将 `source` 命令中的路径替换为实际安装路径。

## 验证记录

- 线上功能测试：18/18 通过，输出错误占比 0%。
- v9 稳定基线性能记录（测试点 10–18）：22µs、1.74ms、1.34ms、17µs、
  2.51ms、1.94ms、33µs、2.24ms、1.76ms。
- 归档中不包含编译产物、临时日志、评测凭据或其他敏感信息。
