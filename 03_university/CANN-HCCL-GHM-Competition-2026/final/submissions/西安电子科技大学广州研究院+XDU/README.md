# 西安电子科技大学广州研究院 + XDU

## 提交信息

- 赛区：2026 CANN HCCL 通信库创新大赛粤港澳赛区
- 阶段：决赛
- 算子：ReduceScatter（CCU 模式）
- CANNJudge 提交 ID：121389
- 提交时间：2026-07-31 19:00:14
- 功能结果：18 / 18 测试点通过
- 性能结果：P10 19 us，P11 1.37 ms，P12 1.11 ms；P13 15 us，P14 2.27 ms，P15 1.81 ms；P16 20 us，P17 2.05 ms，P18 1.66 ms

## 实现摘要

该代码面向 2 x 8 双层拓扑，将大消息规约拆为 Mesh 与 Clos 协同的数据流。不同层通信放入独立 CCU Kernel，以满足双 IO Die/双 CCU 的设备约束；通过固定输出所有权、确定性分组规约和分阶段完成边避免多 writer、notify 多打一及跨层依赖问题。小消息与单层用例保留已经过线上验证的低任务数路径。

`code/` 为官方决赛模板的完整可编译工程，实际比赛修改文件为：

- `code/include/custom.h`
- `code/op_host/exec_op.cc`
- `code/op_host/exec_op.h`
- `code/op_host/reduce_scatter.cc`
- `code/op_kernel_ccu/ccu_kernel.cc`
- `code/op_kernel_ccu/ccu_kernel.h`

## 验证

- CANNJudge 最终提交状态为 Pass，18 个功能测试点全部通过。
- Host notify 审计保持 Record/Wait 一一平衡，无额外 Host notify 或重复 Record。
- 每个 CCU Kernel 只使用同一 IO Die 上对应网络设备；layer-0 与 layer-1 分属不同 Kernel。
- 源码静态审计通过：声明/定义、Kernel 注册、LaunchArg/LoadArg、CCL 地址范围和 16 源数据覆盖均一致。

## 修改源码 SHA-256

```text
5cd50ea878d9f160d45f83e8a875d4f681061def58479236fb08dec60eb0a27e  code/include/custom.h
97d22ba0b18d0a477f755e7064f94e92d78fef60669df132e749d9e794c8d402  code/op_host/exec_op.cc
da1698f5ca99cad87371ffd36dc03553e7117987dd9c28d1246b84808e6c8315  code/op_host/exec_op.h
653a203c5e7215a63759916006240e52100dfb5f32b3a4e2d0d01b79ccb29575  code/op_host/reduce_scatter.cc
d924bef8d6e782c9228d80b0980894154ef2d7e443b0a0115b5fd0dbe9d80ddf  code/op_kernel_ccu/ccu_kernel.cc
ed9a8d41aee46529f502fbcf94f215c30f835b5ba8f73064ce35c83f3cd1fc91  code/op_kernel_ccu/ccu_kernel.h
```
