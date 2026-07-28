# F097：点10 AllGather 循环边着色

## 版本定位

F097 以 F085 为直接基线，只修改精确
`rankSize=16, dataBytes=524288`（测试点10）的最终 AllGather 写入顺序。它不改变
通信图、数据量、归约树、Host 路由、资源注册或另外八条 exact route。

本版本已通过本地合同和远端功能门槛，因此可以提交官方平台做性能 A/B；远端
HCCL-VM 的固定 `50.00us` 只证明功能和时序，不能说明点10已经提速。

## 单变量改动

F085 原路径按绝对 peer 顺序枚举每个 owner 的七条 AllGather `Write`。F097 在
点10私有 helper 中改为：

```text
low offsets : 1,2,3,4,5,6,7
high offsets: 1,3,2,7,6,5,4
pair = (owner + offset[slot]) % 8
```

每个方向在每个 issue slot 都形成一个 perfect matching；两个方向的 matching
不复用同一条物理边。相对原顺序：

- 最大同时接收拥塞：`7 -> 1`；
- 同 slot 双向物理边：`14 -> 0`；
- 反向 Write 的位置间隔：`3,3,3,3,3,3,6`；
- Event bit 仍为 `0..6`，最终 wait mask 仍为 `0x7f`。

精确接线为：

```text
CcuGroupedPairKernel      -> RunRank16ClosRsag
CcuRank16SmallProbeKernel -> RunRank16ClosRsagF097CyclicAg
```

## 冻结账本

- 网络调用：`240`；
- 网络字节：`30D`；
- 数据阶段：`4`；
- 动态 API：`1328`；
- 每 rank：`20 data primitive / 30 WVV / 26 Notify / 7 EventWait`；
- 固定 FP32 树、matching combine、alias snapshot 与 final fence 均不变。

五个允许提交源码中只有 `op_kernel_ccu/ccu_kernel.cc` 相对 F085 改变：

```text
include/custom.h                  2FB442D208D5D791EC85D8F7F966684B6421A5D7F67781A23AB43BA749A8AD9D
op_host/allreduce.cc              1E974C43E4AE8C744E11BE56FA14CB393352CD03D31CCE189A6B040760800EF8
op_host/exec_op.cc                C4BD93D8D704DB2ED45B77CE219168F9BFDC4EBE53F5044D633A739684DBE658
op_kernel_ccu/ccu_kernel.cc       5BAC20B190B6D5C9788A899A6C24A8BBA47644AB5AD9224324E13A536C5A0248
op_kernel_ccu/ccu_kernel.h        B1907B4379D985E857CFB96432571F931C6BC0A379816407F3860CF3FFCE5067
```

## 验证结果

本地：

- `.tmp/analyze_f066_rank16_edge_stagger.js`：PASS；
- `.tmp/verify_f097_rank16_cyclic_ag_independent_redteam.js`：`7/7 PASS`，
  定向突变 `10/10 caught`；
- `g++ -std=c++17 -fsyntax-only`：PASS；
- `cppcheck` 检查三个生产 `.cc`：退出码 `0`，无诊断。

远端：

- clean Release build 与 CCU translator：PASS；
- 候选库 SHA-256：
  `30268856B7DF6515E881DA95FEA15CFEA581CA42637487BAC2F673C0EBCF4757`；
- rank16 / 512KiB Runner `n=20`：PASS；
- 三拓扑乘三个正式尺寸 fresh Checker：`9/9 PASS`；
- gate：`F097_GATE_RC=0`；
- 结束后部署库恢复 F029，SHA-256：
  `7F4B181548ADF4FEDD76C26104B53B51FAA52A9593151F113E1B1A58C70C3F4D`。

## 提交包与裁决

严格五源码包为 `决赛项目目录/F097_submit_5files.zip`，独立解包确认只有五个
允许路径且逐文件 SHA 与上表一致。ZIP SHA-256：

```text
B96530FB654F986B6BF3D89904D908183924409C1581720078BFAF8BBAA29C67
```

线上点10裁决：`<=32us` 采用，`33us` 持平归档，`>=34us` 停止该路线。主要剩余
不确定性是硬件可能在同一个 EventWait 前重排七条 Write，使源码枚举顺序不影响
真实执行，因此必须等待官方性能结果。
