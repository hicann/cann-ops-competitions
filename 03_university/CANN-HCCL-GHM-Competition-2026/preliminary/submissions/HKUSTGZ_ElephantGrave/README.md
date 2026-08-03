# Preliminary Round: AICPU ReduceScatter

## Overview

This directory contains the two contestant-modified source files for the preliminary-round ReduceScatter operator. In the archived competition tree they are placed at `op_host/reduce_scatter.cc` and `op_kernel_aicpu/exec_op.cc`; the remaining files come from the official competition template. The implementation targets a fixed communicator of 16 ranks arranged as two servers with eight NPUs per server. It supports FP32 input and the SUM reduction.

The main design goal is to select a communication schedule according to the real message size. Small messages are dominated by task launch and synchronization latency, while large messages benefit from using all peer links concurrently. The host code therefore creates one channel for every remote rank, allocates one main AICPU thread and fifteen peer worker threads, and stores the serialized resource description in an HCCL engine context. Later invocations reuse these channels, threads, remote CCL-buffer addresses, and Notify resources.

## Small-message path: Recursive Halving

When the total input size is at most 512 KiB, the operator uses four-stage Recursive Halving. The complete input is first copied to the local CCL buffer. At each stage, rank `r` communicates with `r xor mask`, where the masks are processed in the order 8, 4, 2, and 1. Processing the high bit first performs the only inter-server exchange in the first stage; the remaining stages use intra-server links.

Each stage keeps only the destination-block range that can contain the calling rank's final output. `HcommWriteReduceOnThread` transfers and reduces the complementary range directly at the peer, so communication and arithmetic are combined in one data-plane operation. The transferred volume is 8, 4, 2, and 1 output blocks, respectively, for a total of 15 blocks per rank. This is the standard communication-volume bound for a 16-rank ReduceScatter. A matched channel Notify handshake surrounds every exchange, and a local copy writes the surviving block to `recvBuf` after the fourth stage.

This path also covers the minimum functional case. Address calculations use `recvCount` and checked 64-bit byte arithmetic rather than benchmark-specific absolute addresses, so uneven and one-element outputs follow the same logic.

## Large-message path: striped direct reduction

For larger inputs, using one peer per stage leaves most links idle. The large-message path therefore assigns one worker thread and one channel to each of the fifteen remote ranks. Every destination output block is divided into fifteen non-overlapping stripes. The first fourteen stripes are aligned to 4 KiB whenever possible, and the last stripe receives the complete remainder.

Each output owner initializes a single CCL-buffer accumulator with its local contribution. The fifteen source ranks then call `HcommWriteReduceOnThread` to reduce their contributions directly into disjoint accumulator stripes. A cyclic round-robin mapping changes the source-to-stripe assignment in every round. Consequently, each source contributes to every stripe exactly once, every output interval has only one active remote writer in a round, and all fifteen links can make progress concurrently. After fifteen rounds the accumulator contains all sixteen contributions and is copied once to `recvBuf`.

The schedule deliberately avoids fifteen full-size receive buffers and a separate local reduction pass. It needs only one output-sized accumulator, while the network operation performs the reduction during transfer. Deterministic source ordering is preserved for every stripe, which prevents run-to-run changes caused by concurrent FP32 updates to the same address.

## Synchronization and safety

The main thread starts a round by recording one thread Notify for every worker. A worker waits for that signal, performs READY record/wait on its channel, issues the WriteReduce operation, completes the DATA record/wait handshake, and reports DONE to the main thread. The main thread waits for all fifteen DONE notifications before reusing a stripe or Notify epoch. Each record therefore has one matching wait, and no Notify is used as a one-to-many broadcast.

Only one channel is created for each peer pair, as required by the topology. The final handshake also proves that a remote read or reduction has finished before the source buffer can be reused. Tail lengths, integer multiplication, CCL-buffer capacity, rank count, data type, and reduction operation are validated before tasks are submitted.

## Optimization history and result

The initial direct implementation minimized control complexity but did not distinguish fixed overhead from bandwidth cost. Recursive Halving was introduced for the 512 KiB case to reduce the data-plane schedule to four operations. For large messages, an early full-peer design used separate receive areas followed by local reduction; replacing those areas with direct WriteReduce accumulation removed the extra HBM pass. The final refinement added aligned striping, cyclic source assignment, exact tail handling, and a strict per-round DONE barrier to eliminate concurrent destination updates and unsafe Notify reuse.

The final preliminary submission passed all recorded functional checks and placed fourth among 55 teams. Its measured latency was:

| Per-rank input size | Latency |
|---|---:|
| 512 KiB | 42 us |
| 512 MiB | 1.17 ms |
| 400 MiB + 4 B | 971 us |

## Build

Use the official CANN 9.1 environment for the target architecture, then run:

```bash
source /usr/local/Ascend/cann/set_env.sh
bash build.sh
```

`bash build.sh --debug` builds with debug information, and `bash build.sh --format` applies the supplied `.clang-format` rules.
