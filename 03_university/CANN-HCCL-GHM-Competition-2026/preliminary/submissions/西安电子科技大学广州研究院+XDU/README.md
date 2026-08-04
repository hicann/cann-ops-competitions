# 西安电子科技大学广州研究院 + XDU

## 提交信息

- 赛区：2026 CANN HCCL 通信库创新大赛粤港澳赛区
- 阶段：初赛
- 算子：ReduceScatter（AICPU 模式）
- 线上成绩：83.29 分
- 性能结果：40 us / 1.34 ms / 1.07 ms

## 实现摘要

该版本基于双层拓扑的直连输入条带流水：小消息采用递归折半，大消息同时利用 layer-0 与 layer-1 链路，并通过确定性的本地规约顺序保证 FP32 SUM 结果一致。该版本采用K4/K3 主数据流，并将部分传输与完成通知合并为两描述符批量提交。

`code/` 为官方初赛模板的完整可编译工程，实际比赛修改文件为：

- `code/include/custom.h`
- `code/op_host/reduce_scatter.cc`
- `code/op_kernel_aicpu/exec_op.cc`

## 验证

- Release 主机与 AICPU 设备库编译通过，包含 `-Werror` 检查。
- HCCL-VM CheckerV3 的 64 B、512 KB、512 MB、400 MB + 4 B 用例通过。
- 512 MB 与 400 MB + 4 B 在 `HCCL_BUFFSIZE=400` 条件下通过。
- 未使用比赛已知不可用的 `HcommWriteWithNotifyOnThread` 和 `HcommWriteReduceWithNotifyOnThread`。

## 修改源码 SHA-256

```text
57e3aa319cd3481afb09328f5ada2f4928a17d0be08d746c4e36444b22c1623c  code/include/custom.h
6d713bed8ef094f0b01afda8309783bdf1fe759168dd24f4632ecec875598d41  code/op_host/reduce_scatter.cc
ee0bbdf36bd5e689e05e15ac1655380e45ee50bf32722849999c7013aaa65146  code/op_kernel_aicpu/exec_op.cc
```
