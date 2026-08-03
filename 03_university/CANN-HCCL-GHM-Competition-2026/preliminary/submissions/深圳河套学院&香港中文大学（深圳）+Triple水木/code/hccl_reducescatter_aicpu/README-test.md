# ReduceScatter 算法设计与演进

本文记录 2 Server × 8 NPU 环境下 ReduceScatter 的算法演进、当前实现和关键工程约束。重点是说明为什么最终从分层 Mesh/Clos 算法转向直接 All-to-All，以及当前代码如何通过数据分治、动态多Bank流水和减少任务数量提升性能。

## 1. 赛题语义

通信域固定包含 16 个 Rank，数据类型为 FP32，规约操作为 SUM。

`recvCount` 是每个 Rank 的输出元素数，因此每个 Rank 的输入包含 16 个连续目标切片：

```text
sendBuf of rank i
+---------+---------+-----+----------+
| target0 | target1 | ... | target15 |
+---------+---------+-----+----------+
   S         S                S
```

Rank `r` 的结果为：

```text
recvBuf[r] = sum(sendBuf[i][target=r]), i = 0..15
```

比赛页面最初把性能点描述为“输出数据大小”，后续官方澄清实际是每个 Rank 的输入 Buffer 总大小。因此对于 16 Rank：

```text
512 MiB 输入      -> 每个 Rank 输出 32 MiB
400 MiB + 4B 输入 -> 每个 Rank 输出约 25 MiB + 尾部
```

这个澄清非常重要：大数据路径实际处理的单 Rank 输出远小于早期理解，过深流水线往往无法充分进入稳态，任务数量和 tile 粒度反而更加重要。

## 2. 实际拓扑与带宽模型

```text
Server 0: Rank 0  1  2  3  4  5  6  7
             \   |   |   |   |   |   /
                  Full-Mesh

                 Clos Network

Server 1: Rank 8  9 10 11 12 13 14 15
             \   |   |   |   |   |   /
                  Full-Mesh
```

- Server 内 8 张 NPU 为 Full-Mesh，每条卡间链路带宽记为 `V`。
- 每张 NPU 到 Clos 的链路带宽约为 `8V`。
- 早期设计误以为整个 Clos 的聚合带宽只有 `8V`。
- 后续确认 Clos 聚合带宽为 `64V`，8张卡可以同时使用各自的 `8V` 接口能力。
- 每两个 Rank 之间只能申请一条 Channel。

因此，直接 16 Rank All-to-All 并不会像早期估计那样被单个 `8V` Clos 瓶颈限制。它可以同时利用 Server 内 Full-Mesh 和跨 Server Clos，成为当前拓扑下带宽利用率最高、编排最直接的大数据方案。

## 3. 方法演进概览

```text
方法1：分层 Mesh -> Local Reduce -> Clos -> Write
   |
   | 增加不同 tile 的阶段重叠
   v
方法2：64-slot 四阶段流水线
   |
   | Reduce 后只剩两个有效部分和，提前释放 Mesh 区
   v
方法3：38-slot 双环流水线
   |
   | 增加环深、细化 tile，希望吸收下游波动
   v
方法4：72-slot 4+4 深流水线
   |
   | 修正拓扑认识：Clos 实际为 64V，分层算法不再占优
   v
方法5：直接 All-to-All + 本地 Reduce（当前方案）
```

方法1～4建立在“Clos 总带宽只有 `8V`”的旧认识上，目标都是把流量尽量留在 Server 内，只把两个 Server 的部分和通过 Clos 交换。拓扑信息修正后，这些复杂分层方案的意义明显下降：它们减少了跨机流量，却增加了阶段、Notify、Thread 调度和流水线气泡。

## 4. 方法1：基础分层算法

每个 tile 顺序执行：

```text
Mesh -> Local Reduce -> Clos -> Write
```

### 4.1 Scratch 布局

每个 local rank 同时负责两个目标：本 Server 中与自己 local rank 相同的 Rank，以及另一个 Server 中对应的 Rank。

```text
8 source local ranks × 2 target servers = 16 slots

                 target server 0    target server 1
source local 0         slot 0              slot 1
source local 1         slot 2              slot 3
...                     ...                 ...
source local 7         slot 14             slot 15
```

以 Rank 0 为例，Mesh 后它拥有：

```text
Rank 0..7 对 Rank 0 的贡献
Rank 0..7 对 Rank 8 的贡献
```

### 4.2 四个阶段

1. Mesh：Server 内8张卡通过 Full-Mesh把数据写到负责该 local rank 的卡。
2. Local Reduce：对两组8份贡献分别执行固定树形规约。
3. Clos：`rank r` 与 `rank r^8` 交换部分和，并完成最后一级规约。
4. Write：把本 Rank 的完整结果写入 `recvBuf`。

固定树形顺序为：

```text
第一层：0+=1, 2+=3, 4+=5, 6+=7
第二层：0+=2, 4+=6
第三层：0+=4
```

其优点是布局清晰、通信量容易推导；缺点是四个阶段完全串行，任务数量较多。

## 5. 方法2：64-slot 四阶段流水线

方法2不改变方法1的数据流，只使用4组16-slot bank让不同 tile 重叠：

```text
时间       Bank0       Bank1       Bank2       Bank3
T1         Mesh A
T2         Reduce A    Mesh B
T3         Clos A      Reduce B    Mesh C
T4         Write A     Clos B      Reduce C    Mesh D
T5         Mesh E      Write B     Clos C      Reduce D
```

理想稳态吞吐由四阶段耗时之和变为最慢阶段耗时：

```text
串行：Tmesh + Treduce + Tclos + Twrite
流水：max(Tmesh, Treduce, Tclos, Twrite)
```

问题是64个 slot 会缩小单 tile，增加 tile 数量和同步次数；而 Reduce 后每个16-slot bank只有两个结果继续有效，其余14个 slot 在 Clos/Write 阶段被闲置。

## 6. 方法3：38-slot 双环流水线

方法3把 Mesh 区和 Result 区拆开：

```text
2 × 16-slot Mesh bank   = 32 slots
3 ×  2-slot Result bank =  6 slots
总计                      38 slots
```

```text
Mesh ring
+----------------------+----------------------+
| Mesh0: 16 slots      | Mesh1: 16 slots      |
+----------------------+----------------------+

Result ring
+------------+------------+------------+
| Result0: 2 | Result1: 2 | Result2: 2 |
+------------+------------+------------+
```

Reduce 把两个 Server 目标的部分和写入独立 Result bank 后，原16-slot Mesh bank即可提前复用；Clos 和 Write只占用2-slot Result bank。第三个 Result bank用于吸收 Clos/Write 的短期波动。

这个方案减少了闲置 slot，但同步关系变复杂，需要分别维护 MeshFree 和 ResultFree。实测没有优于方法2，说明此时性能主要由 tile 粒度和任务开销主导，而不是 scratch 利用率。

## 7. 方法4：72-slot 4+4 深流水线

方法4继续增加环深：

```text
4 × 16-slot Mesh bank   = 64 slots
4 ×  2-slot Result bank =  8 slots
总计                      72 slots
```

同时将 tile 切得更细，希望用更多 tile填满流水线并吸收阶段抖动。实际结果反而变慢：

```text
81.00 us / 1.96 ms / 1.42 ms
```

主要原因是：

- 官方数据量是总输入大小，每 Rank 输出要除以16，能够进入流水线的 tile 本来就不多。
- slot越多、tile越细，每个 tile 的固定 Notify和任务下发开销占比越高。
- 分层算法包含 Mesh、Reduce、Clos、Write多段依赖。
- 拓扑修正后，刻意压缩 Clos 流量并无必要，反而浪费其 `64V` 聚合带宽。

方法1～4的核心经验是：流水线更深、slot更多并不等于更快；当数据量有限且通信原语调用成本明显时，应优先减少阶段和任务数。

## 8. 方法5：按数据规模分治（当前方案）

### 8.1 方案概述

当前按比赛定义的总输入大小选择路径：

```text
总输入 <= 512 KiB（每 Rank输出 <= 32 KiB）：ReadReduce Recursive Halving
总输入 >  512 KiB：直接 All-to-All + 本地 Reduce
```

小数据的瓶颈主要是任务和 Notify延迟，因此选择通信轮数少的 Recursive Halving；大数据的瓶颈主要是链路带宽，因此选择能够同时利用 Full-Mesh和 `64V` Clos的直接 All-to-All。

### 8.2 大数据路径：直接 All-to-All

当前大数据路径直接执行16 Rank All-to-All。每个源 Rank把目标 Rank `r` 的切片直接写到 Rank `r` 的 scratch，目标 Rank收齐16份后在本地完成确定性 Reduce。

```text
source rank 0..15（跳过 target r）
    -- target r slice --> rank r remote slot 0..14
self contribution
    -- LocalCopy --> output tile
```

它不再区分 Mesh 和 Clos 阶段：15条对端 Channel一起提交，Server 内流量使用 Full-Mesh，跨 Server流量使用 Clos。

每 Rank的输出切片沿元素方向切成多个Pass。每个Bank包含15个远端slot，Bank数量和各Pass大小根据输入规模动态选择：400 MiB + 64 B使用3个常驻Bank，512 MiB使用2个Bank并复用Bank0。

每个 slot的含义始终是“某个远端 source Rank 对当前目标 Rank 的贡献”。本 Rank自己的贡献直接复制到输出 tile，最后再与15个远端贡献的规约结果相加。

不同Bank通过独立offset隔离；Bank内部按 `AlignUp(tileBytes, 4 KiB)` 使用动态物理stride。网络传输和最终输出仍只处理真实的 `tileBytes`，slot尾部padding只用于维持连续Fold布局。

### 8.3 大数据关键优化

#### 8.3.1 15-slot布局

每个目标 Rank只在 scratch中保存15个远端 Rank的贡献，本 Rank贡献直接复制到输出：

```text
scratch bank: slot0 ... slot14   // 15个远端贡献
recvBuf tile: self contribution  // 本Rank贡献
```

这样，同样大小的 HCCL Buffer可以容纳更大的 tile。远端 source Rank按照“跳过目标 Rank后的编号”映射到连续的 `slot0..14`，所有 Rank使用相同的确定性映射。

每个物理 slot的 stride按4 KiB对齐：

```text
slotStride = AlignUp(tileBytes, 4 KiB)
```

网络只传输真实的 `tileBytes`，对齐产生的 padding仅用于维持连续 Fold布局。

#### 8.3.2 根据数据规模选择 Pass布局

大数据性能同时受到两类开销影响：

- Pass过少时，最后一块 Reduce较长，无法再被后续通信隐藏。
- Pass过多时，会增加15路 Write、DATA Record/Wait和本地 Fold的固定任务开销。

当前实测最优选择不是统一切块，而是针对两个性能用例分别布局。

400 MiB + 64 B总输入使用三个常驻 Bank，按 `40% / 40% / 20%`切成三个 Pass：

```text
Pass0：160 MiB总输入，10 MiB / Rank
Pass1：160 MiB总输入，10 MiB / Rank
Pass2： 80 MiB + 64 B总输入，约5 MiB / Rank
```

三个 Bank同时放入约400 MiB的 HCCL Buffer，整个过程不复用 Bank，因此不需要 Bank-free同步。执行关系为：

```text
Communication Threads: A2A0 -------- A2A1 -------- A2A2
Reduce Thread:                 Fold0 -------- Fold1 -------- Fold2
```

前两个 Pass较大，可以保持大块传输效率；最后一个 Pass较小，可以缩短流水线末尾无法被后续通信隐藏的 Reduce。仅采用该布局时，测试点7从两Pass的约918 us降到约899 us；叠加后述的分组提前Reduce后进一步降到约887 us。

512 MiB总输入无法同时放入三个15-slot Bank，因此使用两个 Bank、三个 Pass：

```text
Pass0：约13.33 MiB / Rank
Pass1：约13.33 MiB / Rank
Pass2：约 5.34 MiB / Rank
```

前两个 Pass使用满Bank，第三个 Pass复用 Bank0。相比旧16-slot布局，前两块由约12.5 MiB增大到约13.33 MiB，最终尾块由约7 MiB缩小到约5.34 MiB；叠加分组提前Reduce后，测试点6约为1.14 ms。

这两条路径体现了同一个原则：优先让前面的 Pass承担更多数据，把最后一个不可隐藏的 Reduce尾巴压小；但不能无限增加Pass，否则新增任务的固定开销会抵消收益。

#### 8.3.3 Bank复用与 Notify时序

400 MiB路径的三个 Bank互不复用。每条 Channel现有3个 Notify，在该路径中分别作为三个 Pass的 DATA通知：

```text
Pass0 DATA：Notify 1
Pass1 DATA：Notify 2
Pass2 DATA：Notify 0
```

每个 Notify在整条路径中只 Record和Wait一次，不存在“多打一”。Notify 0通常承担 READY，但400 MiB路径不执行 Bank复用，因此可以安全地用于第三个 DATA。

512 MiB路径只使用 `Notify 1/2`作为两个 Bank的 DATA通知。第三个 Pass复用 Bank0前必须执行：

1. Reduce Thread完成 Pass0并向 Coordinator发送本地 `BANK_FREE`。
2. Coordinator等待 `BANK_FREE`。
3. 所有 Rank通过 Channel `READY`握手，确认目标 Bank均已释放。
4. Coordinator唤醒15条通信 Thread，允许第三轮覆盖 Bank0。

第三轮再次使用 Bank0的 DATA Notify前，第一轮对应Wait已经完成并清零寄存器，因此不会出现两个Record命中同一个Wait的问题。

#### 8.3.4 独立通信 Thread与生命周期门控

15个对端各使用一条 Channel和一条通信 Thread，确保Full-Mesh和Clos链路能够并发工作。每个 Pass在对应通信 Thread上按顺序下发：

```text
Write -> DATA Record
```

Reduce相关 Thread执行：

```text
Reduce Main:  Self Copy -> 等待slot7和两组完成 -> 后三层Fold -> output += remote sum
Group Worker 0: 等待第一组8路DATA -> 连续Reduce 4对 -> GROUP_DONE
Group Worker 1: 等待第二组6路DATA -> 连续Reduce 3对 -> GROUP_DONE
```

把全部Channel压到同一Thread会让15条Write在同一任务队列中串行，实测大数据耗时会扩大约7～8倍，因此不能用减少Thread数量换取更少的调用。

所有额外申请的从Thread仍遵守官方时序约束：第一项任务必须等待Thread0的LocalWait，最后一项任务必须LocalRecord通知Thread0。START和DONE只在整场大数据操作的首尾各执行一次，不随Pass重复。

#### 8.3.5 第一层分组提前 Reduce

旧实现先等待15路 DATA全部到达，再通过4次连续Fold规约远端贡献，最后合入self：

```text
slots[0..6] += slots[8..14]  // slot7暂时保留
slots[0..3] += slots[4..7]
slots[0..1] += slots[2..3]
slot[0]     += slot[1]
recvBuf     += slot[0]
```

这会形成 Tile级全局屏障，第一层Reduce必须等最慢Channel。曾尝试把self作为第16个逻辑slot，构造8个独立叶子并继续执行 `8 -> 4 -> 2 -> 1` 平衡树。该方案虽然可以做到“任意一对先到先算”，但每个Pass会从5次Reduce增加到15次Reduce，并引入大量Thread Record/Wait。实测测试点7由899 us劣化到约903 us，说明细粒度重叠收益不足以覆盖任务下发和同步开销。

当前实现采用两组折中方案：

```text
Group 0: 等待slot[0..3]和slot[8..11]
         slots[0..3] += slots[8..11]   // 4对，一次连续Reduce

Group 1: 等待slot[4..6]和slot[12..14]
         slots[4..6] += slots[12..14]  // 3对，一次连续Reduce

Reduce Main:
         等待slot7、Group 0和Group 1
         slots[0..3] += slots[4..7]
         slots[0..1] += slots[2..3]
         slot[0]     += slot[1]
         recvBuf     += slot[0]
```

两组分别只依赖自己的8路或6路Channel。某组数据先到齐，就能在其他通信尚未完成时提前执行第一层Reduce；同时第一层只从1次连续调用增加到2次，保留了大块连续Reduce的低调用开销。

每个 Tile的本地操作为：

```text
1次 Self Copy + 2次分组第一层Reduce
+ 3次后续Fold + 1次最终Reduce
```

相比旧实现，每Pass只多1次Reduce和少量组完成同步，却获得部分通信/计算重叠。该优化将测试点6从约1.15 ms降至1.14 ms，测试点7从约899 us降至887 us。

物理stride大于真实tile时，分组Reduce和后续Fold也会访问padding，但padding始终只与对应padding相加，不会进入有效输出范围。最终写入和最后一次Reduce只处理真实的 `tileBytes`。

远端15份贡献的Fold顺序固定，最后再合入self贡献，因此浮点加法顺序确定，满足比赛的确定性要求。

### 8.4 小数据路径：Recursive Halving

当总输入不超过512 KiB，即每 Rank输出切片不超过32 KiB时，使用4轮异或 Recursive Halving：

```text
peer = rank XOR 8
peer = rank XOR 4
peer = rank XOR 2
peer = rank XOR 1
```

每一轮后，当前有效区间减半：

```text
16份 -> 8份 -> 4份 -> 2份 -> 1份
```

设每个 Rank最终输出切片大小为 `S`。四轮分别交换 `8S`、`4S`、`2S` 和 `S`，每 Rank总通信量为：

```text
8S + 4S + 2S + S = 15S
```

这已经达到16 Rank精确 ReduceScatter 的通信量下限。若不使用压缩、低精度或交换机内规约，就无法继续减少数据量。因此，小数据优化的重点不是继续减少字节数，而是减少通信原语、Notify和本地任务的调用数量。

每轮中，两个伙伴保留相反的一半数据。本 Rank只读取并累加自己下一轮需要保留的区间；四轮结束后，每个 Rank只剩下属于自己的结果切片。伙伴顺序和每轮规约顺序固定，因此浮点加法树也固定，满足确定性要求。

### 8.5 小数据关键优化

#### 8.5.1 从多阶段算法缩短为4轮通信

小数据路径经历了以下演进：

```text
初始分层方案                 约 81 us
MR + CW 两阶段方案           约 68 us
Recursive Halving           约 44 us
ReadReduce Recursive Halving   39 us
```

初始方案仍包含 Server 内通信、局部规约、跨 Server通信和写回等多个阶段。小数据本身只需要传输很少的字节，链路尚未成为主要耗时，阶段切换和同步任务反而占据关键路径。

Recursive Halving把16 Rank规约固定为 `log2(16)=4` 轮，每轮只与一个异或伙伴通信。它不需要15轮 Ring，也不需要先做 Server 内汇聚再跨 Server交换，从而显著减少任务数量。81 us到44 us的变化说明：对小数据而言，缩短编排路径比追求峰值带宽更重要。

#### 8.5.2 ReadReduce合并通信与规约

早期 Recursive Halving 每轮的任务近似为：

```text
Write
DATA Record
DATA Wait
LocalReduce
```

加上首尾两次 LocalCopy，四轮约为：

```text
2 + 4 × 4 = 18 个任务
```

每轮执行 READY握手后，通过 `HcommReadReduceOnThread` 直接读取并规约对端保留区间，避免先 Read、再单独调用 LocalReduce。

当前每轮变为：

```text
READY Record
READY Wait
ReadReduce
```

完整关键路径约为：

```text
初始 LocalCopy                           1
4 × (READY Record + READY Wait)          8
4 × ReadReduce                           4
最终 LocalCopy                           1
合计                                    14 个任务
```

从18个任务压缩到14个任务后，小数据耗时由约44 us降到39 us。收益主要来自少下发4个 LocalReduce/通信相关任务，而不是卡内 Reduce计算本身；官方也确认卡内规约接近访存速度，其计算时间远小于通信和任务调度开销。

READY不能继续直接删除。第 `k+1` 轮读取的是对端在第 `k` 轮刚形成的部分和，不同 Rank执行进度可能不同，因此读取前必须确认伙伴已经完成上一轮。否则可能读到尚未规约完成的数据。

#### 8.5.3 复用统一资源

控制面不再预先按整个输入大小选择两套资源。大小数据共用15条对端 Channel；小数据执行时只访问4个异或伙伴：

```text
rank XOR 8 / 4 / 2 / 1
```

小数据任务只下发到 Thread0，另外15条从 Thread不下发任务。Recursive Halving只使用4条 Channel的 READY Notify和16个连续 scratch slot；ReadReduce直接在原有有效区间上更新。

统一资源仍允许其他大数据路径在出现很小尾块时切换到Recursive Halving；当前两个性能用例均由专用的大数据Pass布局完整覆盖。

小数据实际使用为：

```text
活跃 Thread:        1
活跃 Channel:       4（XOR 8/4/2/1）
使用 Notify:        READY
Scratch:            16 slots
```

#### 8.5.4 当前结论

小数据路径已经同时满足：

```text
通信量：15S，达到算法下限
通信轮数：4，达到二进制 Recursive Halving 的最小轮数
任务数量：约14个
活跃 Channel数量：4
活跃 Thread数量：1
```

后续若继续优化，应优先寻找能够合并 READY 与 ReadReduce、或减少任务下发固定开销的受支持原语，而不是改成更多轮的 Ring或继续切细数据。当前普通 `ReadReduce` 路径也避开了 `WithNotify` 原语在 HCCL-VM中的兼容性问题。

## 9. 当前资源配置

### 小数据

```text
控制面统一申请19条 Thread和15条 Channel；执行时仅使用：
  Thread0: 1
  Channel: 4（XOR 8/4/2/1）
  Channel Notify: READY
  Scratch: 16 slots
```

### 大数据

```text
Thread: 19
  Thread0: Coordinator
  Thread1..15: 每个对端一条独立通信 Thread
  Thread16: Reduce Main，执行Self Copy、等待分组完成、后三层Fold和最终Reduce
  Thread17..18: 第一层分组Reduce Worker

Channel: 15（每个对端一条）
Channel Notify: 3（READY/DATA2、DATA0、DATA1）

Scratch:
  400 MiB + 64 B：3 × 15 slots，三个动态Bank，不复用
  512 MiB：2 × 15 slots，第三个Pass复用Bank0
```

## 10. 正确性与性能注意事项

- 仅使用 AICPU+TS。
- 每个对端只能申请一条 Channel。
- FP32 SUM必须采用固定规约顺序，不能按数据到达顺序累加。
- 同一 Notify在前一次 Wait消费前不能再次 Record，避免“多打一”。
- 并发驻留的Pass必须使用不同DATA Notify。
- 从 Thread首任务必须是等待 Thread0的 LocalWait，末任务必须是通知 Thread0的 LocalRecord。
- Bank复用必须等待Reduce释放Bank，再完成全对端READY和本地通信Thread门控。
- `HcommWriteWithNotifyOnThread` 和 `HcommWriteReduceWithNotifyOnThread` 曾在 HCCL-VM中触发 `not support opcode[0]`，当前继续使用独立 Write和Notify。
- 官方的 `400MB+4B` 用例实际补齐为 `400MB+64B` 总输入；当前按40%/40%/20%切成三个大数据Pass，不产生4B尾部Recursive Halving。
- 数据量有限时，过细 tile会增加固定任务开销；不要只根据流水线深度判断性能。

## 11. 当前代码对应关系

```text
include/custom.h
  拓扑、Thread/Channel/Notify数量和大小数据阈值

op_host/reduce_scatter.cc
  参数检查、资源申请、Channel建立和AICPU Kernel下发

op_kernel_aicpu/exec_op.cc
  小数据Recursive Halving
  大数据All-to-All数据分治
  400 MiB三Bank三Pass布局
  512 MiB双Bank三Pass及Bank复用同步
  Communication Thread生命周期门控
  15-slot第一层分组提前Reduce、后三层连续Fold和确定性最终Reduce

op_host/launch_aicpu_kernel.cc
op_kernel_aicpu/aicpu_kernel.cc
  用户Stream与AICPU Thread0之间的前后序同步
```

当前主线结论是：在 Clos聚合带宽为 `64V` 的真实拓扑下，大数据应优先使用直接 All-to-All充分利用所有链路，再通过数据分治解决 HCCL Buffer容量问题；性能优化重点是减少阶段、Thread和Notify任务，而不是继续加深分层流水线。
