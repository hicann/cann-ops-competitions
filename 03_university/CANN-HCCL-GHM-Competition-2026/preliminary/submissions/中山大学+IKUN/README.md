# 中山大学 + IKUN

## 提交信息

- 赛区：2026 CANN HCCL 通信库创新大赛粤港澳赛区
- 学校：中山大学
- 队伍：IKUN
- 团队成员（GitCode）：`2301_78565107`、`m0_75180272`、`2301_76761127`
- 联系方式：请通过本次 Pull Request 讨论区联系
- 阶段：初赛
- 算子：ReduceScatter（AICPU 模式）
- 归档版本：V148G2A
- CANNJudge 代表提交 ID：110294
- CANNJudge 结果：7 / 7 测试点通过
- 性能结果：P5 39 us，P6 1.15 ms，P7 0.903 ms

## 实现摘要

该版本是队伍初赛阶段最后一份具有完整 CANNJudge 实测结果的联合强版本，三个性能点
按消息规模分别选择经过验证的数据流：

- P5 使用固定展开的 XOR pull，小消息路径裁剪循环和多余控制开销。
- P6 使用 compact-packed 大消息流水，三组条带比例为 `9 / 13 / 10`，在保持
  16-thread 资源族和固定浮点归约顺序的同时平衡各阶段关键路径。
- P7 使用 `4 / 8 / 10 / 10` 条带划分及 7-worker pair-reduce 完成树，保持已验证
  的覆盖保护、唯一写入和确定性归约顺序。

实现遵守每个输出元素只归约一次、同一输出地址单写入者、固定浮点累加顺序和 Host /
AICPU 参数一致等约束。大消息使用分条带流水，将通信、规约和完成同步映射到独立 worker，
降低串行尾部开销。

`code/` 是官方初赛模板的完整工程。比赛算法修改集中在：

- `code/include/custom.h`
- `code/op_host/reduce_scatter.cc`
- `code/op_kernel_aicpu/exec_op.cc`

其余模板文件一并保留，以便独立编译和复核。

## 比赛经验

- **先正确、再优化**：先用最朴素、易验证的数据流跑通全部功能点作为可信基线，
  再按性能点逐个替换为实测过的数据流；任何优化都以不破坏确定性为前提。
- **按消息规模分流**：小消息瓶颈在延迟与控制开销，用固定展开、裁剪循环降低
  启动成本；大消息瓶颈在带宽与串行尾部，用条带流水把通信、规约、完成同步
  映射到独立 worker，各阶段条带比例按实测瓶颈链路容量分配。
- **守住核心不变量**：每个输出元素只归约一次、同一输出地址单写入者、固定
  浮点累加顺序、Host / AICPU 参数一致，这四条贯穿所有版本，是避免返工的关键。
- **提交前多轮验证**：六档规模各两次正确性验证加双侧 `-Werror` 编译检查，
  归档版本取自 CANNJudge 代表提交的网页回读，确保与实测源码逐文件一致。

决赛阶段（CCU 模式）的完整经验总结见决赛目录 README。

## 验证与来源

- CANNJudge 代表提交 110294：7 / 7 Pass，本提交性能为 `39 / 1150 / 903 us`。
- 同一冻结源码的 110258、110282、110294 三份网页回读归档逐文件一致。
- 冻结源码树 SHA-256：
  `ed81d52ba1c80e01fc44520f39cf7a6411fa3cf25856c13361986cadfbcaf66b`。
- 提交前通过六档各两次正确性验证及 Host / AICPU 双侧 `-Werror` 编译检查。
- 归档源码取自代表提交 110294 的提交后网页回读，不包含后续未取得更优平台真值的
  实验版本。
- 归档前已检查源码完整性，并确认不包含构建产物、缓存、访问凭据、服务器身份或私人
  实验工作流信息。

## CANNJudge 文件 SHA-256

```text
3b25a0aaac46afab911656c70b85ad22d6c04d4ad5c9ad3a8cb34122299755e9  code/include/custom.h
a66d389d3331febcc399b2c1130298ccf05e5d42fe184d47541c9b68403f0e91  code/op_host/launch_aicpu_kernel.cc
d2f05f44f884c535703f4121281dde8cac2e193a031430fbb4e0620e71ec3ed0  code/op_host/launch_aicpu_kernel.h
8e1bd39beaf7021e2d6a318d3489eef9f8d6d33f544adcbc8e8008860d2a44b1  code/op_host/reduce_scatter.cc
de220bb11d9797936ccdd74dfc65d1c8fdc38061654fcde0b31b76d7b95f37c2  code/op_kernel_aicpu/aicpu_kernel.cc
62b95a605a5198662a64799429ef5b72889be8f4c41a12e6d0edc06a6e81a276  code/op_kernel_aicpu/exec_op.cc
fd8cbe30d479707fc9ea24f32361362ce8dc8682647ff96b50f32542f971208a  code/op_kernel_aicpu/exec_op.h
```
