# 中山大学 + IKUN

## 提交信息

- 赛区：2026 CANN HCCL 通信库创新大赛粤港澳赛区
- 学校：中山大学
- 队伍：IKUN
- 团队成员（GitCode）：`2301_78565107`、`m0_75180272`、`2301_76761127`
- 联系方式：请通过本次 Pull Request 讨论区联系
- 阶段：决赛
- 算子：ReduceScatter（CCU 模式）
- CANNJudge 提交 ID：121466
- 功能结果：18 / 18 测试点通过
- 性能结果：P10 23 us，P11 1.61 ms，P12 1.22 ms；P13 24 us，
  P14 2.40 ms，P15 1.84 ms；P16 21 us，P17 2.08 ms，P18 1.63 ms

## 优化里程碑

| 版本 | 说明 | 2×8 (μs/ms/ms) | 4×1 | 8+4 | 关键改进 |
|------|------|-----------------|-----|-----|----------|
| v1 | 首个 CCU 算子（单 kernel 全 channel） | 功能未过 | - | - | dies of channels are not same |
| v3 | 并行读 | 51/2.52/1.96 | 19/2.56/1.95 | 53/3.31/2.54 | 组内所有 channel Read 一次性下发 |
| v5 | chunk 流水 + 树形归约 | 23/1.64/1.31 | 19/2.45/1.94 | 24/2.08/1.66 | in-kernel chunk 交错+奇偶双缓冲+event bit 按对等待 |
| v6 | 按对增量等待 | 23/1.64/1.30 | 19/2.46/1.88 | 23/2.06/1.64 | slot 首次使用精准等待一次（ClearCKE 兼容） |
| v14 | 重复调用免前后同步 + 小包单线程 | 24/1.60/1.23 | 20/2.43/1.86 | 21/2.07/1.63 | syncMode 跳过已缓存地址的 PreSync/PostSync，小包省线程握手 |

## CCU 开发核心约束（比赛实战验证）

1. **channel 变量槽 ≤ 2**：`GetResByChannel` 的 varIdx 只能取 0/1，新增交换变量需复用旧槽 + 错开 notify 掩码位
2. **kernel 注册期全量翻译**：CCU_IF 所有分支在 `HcommCcuKernelRegister` 时执行，翻译期路径不可返回错误码（返回 CCU_SUCCESS 空过）
3. **EventWait = ClearCKE**：等待后清位（`src/base_comm/resources/ccu/ccu_representation/reps/translator/ccu_ins_generater_v1.cc`），每位只等一次，流水用多 event 或位隔离
4. **channel 变量先 NotifyWait 后读**：checker 静态校验 XN 初始化时序（ErrorCode 108）
5. **跨 rank 交换的 buffer 写入方必须唯一**：多 writer 并发写同一区判冲突（ErrorCode 302）
6. **slave 线程任务图 ABI**：首任务本地 WAIT、尾任务 RECORD（ErrorCode 203）

## 实现摘要

该版本针对 2 x 8、4 x 1 和 8 + 4 三类评分拓扑分别选择已验证的数据流，
并保持每个 CCU Kernel 只使用对应 IO Die 的本地 endpoint。跨层通信采用独立
Kernel 和显式完成依赖，贡献集合、私有 partial、scratch 生命周期与输出所有权均
按拓扑固定，避免跨 Die 资源混用、多 writer 和重复完成通知。

在最终版本中，rank-16 小消息路径裁剪为已实测的数据流，8 + 4 单 chunk 路径移除
静态不可达的后续 chunk 分支；大消息与尾块继续使用已经通过动态验证的固定树，避免
把完成态证明尚不完整的候选混入最终提交。

`code/` 为官方决赛模板的完整可编译工程。实际比赛修改文件为：

- `code/include/custom.h`
- `code/op_host/exec_op.cc`
- `code/op_host/exec_op.h`
- `code/op_host/reduce_scatter.cc`
- `code/op_kernel_ccu/ccu_kernel.cc`
- `code/op_kernel_ccu/ccu_kernel.h`

## 比赛经验

### 方案演进

1. **先正确、再优化**：第一版采用最朴素的串行 ReadReduce——先把自己 slice 拷入
   输出作为归约基数，再逐 peer 串行 ReadReduce 累加。逻辑简单，便于验证正确性
   与确定性，是后续所有优化的可信基线。
2. **并行化**：改为所有 peer 并行 Read 到 scratch，再做固定顺序的树形归约，
   将归约阶段从 O(N) 串行降到 O(log N) 层。
3. **分块与分批**：大消息在 Host 侧按 256MB 单次传输上限切块；Kernel 内对
   peer 分批处理并复用 scratch，缓解 HCCL buffer（400MB）压力。
4. **按拓扑×尺寸特化**：最终版本不再追求单一"通用最优"算法，而是为九个评分点
   （三类拓扑 × 三种数据量）分别选择实测验证过的数据流，小消息路径裁剪冗余
   分支，大消息路径做 chunk 级流水。

### 实测踩到的硬件约束

- N 路 `LocalReduce` 的 buffer 数量存在上限（报错形如
  "count and mem size must less than 8"），超过 7 个输入必须分组、分多趟归约。
- `ccu::Loop` / `LoopGroup` 的 body 内有语法限制：不能使用条件分支
  （`CCU_IF`），也不能使用 `LocalReduce(LocalAddr, LocalAddr)`；要用 LoopGroup
  流水，需先把数据搬入 CcuBuffer 再做 N 路归约，这显著提高了该特性的使用门槛。
- 两个 Rank 同时对同一输出地址做 WriteReduce 会触发 memory conflict 错误：
  非 primary kernel 必须写各自的私有 partial，最后由 primary kernel 按固定
  顺序合并。
- 多 IO Die 拓扑（2×8、8+4）下，单个 Kernel 内的通道必须归属同一 die；
  endpoint 属性查询只使用当前 rank 已选 channel 的 localEndpoint，不做跨 die
  的地址 join。

### 确定性

赛题要求 FP32 累加顺序在所有 rank 上完全一致。浮点加法不满足结合律，
"理论上等价"的归约顺序会直接导致精度校验失败。固定树形、固定 peer 顺序、
固定 partial 合并顺序、固定 chunk 内累加顺序，是功能验收的前提，也是设计
任何并行化/流水变换时首先要守住的不变量。

### 验证与提交策略

- 建立本地 HCCL-VM 全量验证后再提交 CANNJudge 的流程：本地 Checker /
  Semantic Checker 全通过再消耗提交次数，避免盲目试错。
- CANNJudge 的报错非常简略（往往只有 `hcclRet=4` 这类码），没有本地复现
  手段几乎无法定位问题；HCCL-VM 是本赛题最重要的调试基础设施。
- 只提交本地实测过的配置：每个评分点只选在 HCCL-VM 上实测过性能的数据流，
  最终版本不押注"理论更快但未验证"的方案。
- 每个迭代版本独立备份（v1~v5 逐级留档），优化失败时可以直接回退到上一个
  已通过版本，避免破坏已验证成果。

### 分场景优化方向

- **小消息（512KB）**：延迟主导。关键是减少同步轮次与 kernel 发射次数，
  并按评分尺寸静态裁剪不可达分支。
- **大消息（512MB / 400MB+4B）**：带宽主导。关键是 chunk 流水化、远端 Read
  与本地归约重叠（双 scratch bank 交替使用），并按瓶颈链路容量分配 chunk
  比例——Clos 跨机带宽约为机内链路的 4 倍，跨机与机内通信的切分要匹配
  各自的容量上限。

## 验证

- CANNJudge 最终提交状态为 Pass，18 个功能测试点全部通过。
- 完整 HCCL-VM 验证的 11 个会话全部通过，Checker 与 Semantic Checker 均无错误。
- 每个 CCU Kernel 仅注册并使用对应 IO Die，通信地址均来自当前 Rank 已选择
  channel 的 local endpoint。
- 静态检查覆盖贡献集合、依赖边、scratch 与 partial 地址范围、通知配对、Kernel
  注册和 Host/Kernel 参数一致性。
- 归档前已检查源码完整性，并确认不包含构建产物、缓存、访问凭据或私人实验材料。

## 修改源码 SHA-256

```text
53a4ec9f7540e3ba35a233d5c3e2a2396ea48cec604cddf08e5358f21d7948e1  code/include/custom.h
5b16c47c2ecc6d6dbdeb91b65f3af2efcf079d102fb2b9caf12a4f00c3f27921  code/op_host/exec_op.cc
12271549234677902128906c759cf4623f69c03ebc2e7d1467437d157858b863  code/op_host/exec_op.h
9d56c4b590442236d3aa11c4f3b033b6e04c8de485016af08f8c8eb79b8e4d00  code/op_host/reduce_scatter.cc
4f3d7d5d035aa64481ccf6b0f1b97fa7143b92bb4fa4d05727c87d7be759f037  code/op_kernel_ccu/ccu_kernel.cc
e372a4b361601bf4fccb7ad98abad88b23613e69cd9605ce59f6e219c438a4f4  code/op_kernel_ccu/ccu_kernel.h
```
