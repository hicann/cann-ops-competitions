# Final Round: CCU ReduceScatter

## Overview

This directory contains the complete final-round ReduceScatter submission. The operator implements FP32 SUM ReduceScatter for the three official communicator layouts: 16 ranks on two eight-NPU servers, four ranks distributed across four servers, and an asymmetric 12-rank layout with eight ranks on one server and four on the other. The implementation uses topology- and message-size-specific CCU schedules instead of forcing all layouts through one communication graph.

The host discovers topology layers, creates one channel for every peer, registers CCU kernels, and serializes the resources into an HCCL engine context. Later invocations refresh the stream-bound thread handle and obtain tokens for the current memory addresses. Kernel arguments contain checked byte ranges, channel handles, rank metadata, and stripe descriptions.

## Small-message processing

Small messages use compact Memory Slice reduction. Input segments are loaded into MS buffers, reduced locally, and written either to the final output or to a compact scratch partial. For a topology with two communication layers, the partial kernels run on different CCU threads and are followed by one local combine operation.

The four-rank 512 KiB case has a dedicated packed-MS layout. It packs the complete reduction into checker-safe MS slices, uses the three peer channels in parallel, and retains a full peer PostSync handshake. This removes large HBM workspaces and minimizes kernel-launch overhead without relaxing the input-lifetime proof.

## Four-rank large-message processing

For large messages in the four-server topology, every rank first copies its own contribution to `recvBuf`. The output is split into three aligned stripes, one for each remote peer. Sources then use `WriteReduce` to update the destination output directly. A cyclic assignment gives every peer a different stripe in each round, so active writes are disjoint.

The DATA epoch alternates between rounds, and an aggregate acknowledgement closes each round before destination addresses advance. A local `EventWait` does not by itself order independent senders targeting the same output. The acknowledgement prevents write-after-write hazards and proves remote consumption is complete before input reuse.

## Twelve-rank and sixteen-rank processing

The two-server layouts separate local-node and cross-node work and divide the output into two halves. Four CCU kernels execute a two-stage schedule: local reduction for the first half runs concurrently with cross-node reduction for the second half, a thread rendezvous joins the stage, and the topology roles are reversed for the remaining halves. This arrangement uses both IO Dies while ensuring that one CCU kernel never mixes network devices from different topology layers.

For the asymmetric eight-plus-four layout, the host builds different local channel groups on the two sides but keeps the cross-node DATA and acknowledgement epochs aligned. `WriteReduce` updates disjoint destination stripes, and the stage barrier prevents either node from entering the reversed phase early.

The two official large-message counts on the 16-rank layout use an output-owner schedule. Each owner copies its own contribution to the final output and pulls the other contributions with `ReadReduce`. The seven local peers are partitioned across seven disjoint output regions, while the eight cross-node peers use a one-partition perfect-matching rotation. Every round maps active channels to non-overlapping owner regions and calls `EventWait` before moving any destination address. Input addresses and tokens are published once, and a final matched PostSync on every edge closes the source-buffer lifetime.

Moving the reduction to the output owner eliminates distributed sender collisions and reduces data-plane operations. Other counts remain on the general topology-safe path.

## Optimization history

Development started with layer-by-layer partial reduction, which required large scratch areas and a final combine. Wide-lane reads and direct `WriteReduce` output accumulation then reduced HBM traffic. Early sender-driven cross-node schedules allowed different senders to reach the same output stripe concurrently; the final protocol added DATA epochs, aggregate acknowledgements, and stage barriers.

Small-message work then focused on fixed overhead. Compact MS arguments removed unnecessary workspace construction, and the packed four-rank path reduced launch and synchronization cost while preserving full PostSync. For the largest 16-rank cases, changing from sender-driven WriteReduce to owner-driven ReadReduce removed the remaining distributed-target conflict and improved measured latency. More aggressive unmeasured combinations were intentionally excluded from the archived submission; this directory is the complete package that passed all nine functional and all nine performance tests together.

## Final measured performance

| Topology | 512 KiB | 512 MiB | 400 MiB + 4 B |
|---|---:|---:|---:|
| 2 x 8 ranks | 17 us | 1507 us | 1222 us |
| 4 x 1 ranks | 13 us | 2256 us | 1780 us |
| 8 + 4 ranks | 17 us | 2101 us | 1676 us |

The four-rank small-message implementation is byte-identical to the version that also measured 12 us in an earlier run; the table reports the latency from this complete final package.

## Build

After loading the official CANN 9.1 environment, build the operator with:

```bash
source /usr/local/Ascend/cann/set_env.sh
bash build.sh
```

Use `bash build.sh --debug` for a debug build or `bash build.sh --format` to apply the supplied formatting rules.
