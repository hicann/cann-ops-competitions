#include <algorithm>
#include <array>
#include <limits>

#include "custom.h"
#include "log.h"
#include "exec_op.h"

namespace
 {
constexpr uint32_t SERVER_RANK_NUM = 8
;
constexpr uint32_t SERVER_NUM = 2
;
constexpr uint32_t TOTAL_RANK_NUM = SERVER_RANK_NUM * SERVER_NUM
;

// 1条控制Thread + 7条通信Worker + 1条本地拷贝Thread。
constexpr uint32_t CONTROL_THREAD_IDX = 0
;
constexpr uint32_t FIRST_WORKER_THREAD_IDX = 1
;
constexpr uint32_t WORKER_NUM = 7
;
constexpr uint32_t COPY_THREAD_IDX = 8
;
constexpr uint32_t THREAD_NUM = 9
;

constexpr uint64_t TINY_DATA_THRESHOLD = 4ULL * 1024ULL
;
constexpr uint64_t SMALL_DATA_THRESHOLD = 1024ULL * 1024ULL
;

// 小消息两阶段广播使用独立的Channel Notify。
constexpr uint32_t SMALL_INTRA_NOTIFY = 1
;
constexpr uint32_t SMALL_INTER_NOTIFY = 2
;
constexpr uint64_t PIPELINE_TILE_BYTES = 64ULL * 1024ULL * 1024ULL
;
constexpr uint32_t PIPELINE_SLOT_NUM = 4
;

// 四个流水slot分别使用独立Channel Notify，避免同一Notify在重叠tile间丢信号。
constexpr uint32_t CHANNEL_NOTIFY_PER_SLOT = 3
;
constexpr uint32_t AckNotify(uint32_t slot
)
{
    return slot * CHANNEL_NOTIFY_PER_SLOT
;
}
constexpr uint32_t DataNotify(uint32_t slot
)
{
    return slot * CHANNEL_NOTIFY_PER_SLOT + 1
;
}
constexpr uint32_t AllGatherNotify(uint32_t slot
)
{
    return slot * CHANNEL_NOTIFY_PER_SLOT + 2
;
}

// Thread notify也按slot区分。
constexpr uint32_t WorkerReadyNotify(uint32_t slot
)
{
    return slot
;
}
constexpr uint32_t CopyStartNotify(uint32_t slot
)
{
    return slot
;
}
constexpr uint32_t CopyReadyNotify(uint32_t slot
)
{
    return PIPELINE_SLOT_NUM + slot
;
}

using ChannelTable = std::array<const ChannelInfo *, TOTAL_RANK_NUM>
;

HcclResult BuildChannelTable(const OpParam &param, const AlgResourceCtx &resCtx, ChannelTable &table
)
{
    table.fill(nullptr
);
    for (const auto &channel : resCtx
.channels) {
        CHK_PRT_RET(channel.remoteRank >= TOTAL_RANK_NUM
,
            HCCL_ERROR("Invalid remote rank[%u]", channel.remoteRank), HCCL_E_INTERNAL
);
        table[channel.remoteRank] = &channel
;
    }

    for (uint32_t rank = 0; rank < param.rankSize; ++rank
) {
        if (rank == param
.myRank) {
            continue
;
        }
        CHK_PRT_RET(table[rank] == nullptr
,
            HCCL_ERROR("Channel to rank[%u] not found", rank), HCCL_E_INTERNAL
);
    }
    return HCCL_SUCCESS
;
}

uint32_t ToAbsoluteRank(uint32_t relativeRank, uint32_t root, uint32_t rankSize
)
{
    return (relativeRank + root) % rankSize
;
}

HcclResult RunBinomialTreeChunk(const OpParam &param, const AlgResourceCtx &resCtx, const ChannelTable &channels
,
    ThreadHandle thread, uint64_t dataOffset, uint64_t chunkBytes
)
{
    char *input = static_cast<char *>(param
.inputPtr);
    char *output = static_cast<char *>(param
.outputPtr);
    char *localBuffer = static_cast<char *>(resCtx
.localBuffer.addr);

    if (param.myRank == param
.root) {
        CHK_RET(HcommLocalCopyOnThread(thread, localBuffer, input + dataOffset, chunkBytes
));
    }

    const uint32_t relativeRank = (param.myRank + param.rankSize - param.root) % param
.rankSize;
    for (uint32_t mask = 1; mask < param.rankSize; mask <<= 1
) {
        if (relativeRank < mask
) {
            const uint32_t childRelativeRank = relativeRank + mask
;
            if (childRelativeRank < param
.rankSize) {
                const uint32_t childRank = ToAbsoluteRank(childRelativeRank, param.root, param
.rankSize);
                const ChannelInfo *childChannel = channels[childRank
];
                CHK_PTR_NULL(childChannel
->remoteCclMem.addr);
                CHK_RET(HcommWriteOnThread
(
                    thread, childChannel->handle, childChannel->remoteCclMem.addr, localBuffer, chunkBytes
));
                CHK_RET(HcommChannelNotifyRecordOnThread
(
                    thread, childChannel->handle, NOTIFY_IDX_DATA_SIGNAL
));
            }
        } 
else if (relativeRank < (mask << 1
)) {
            const uint32_t parentRank = ToAbsoluteRank(relativeRank - mask, param.root, param
.rankSize);
            const ChannelInfo *parentChannel = channels[parentRank
];
            CHK_RET(HcommChannelNotifyWaitOnThread
(
                thread, parentChannel->handle, NOTIFY_IDX_DATA_SIGNAL, 0
));
        }
    }

    if (param.myRank != param
.root) {
        CHK_RET(HcommLocalCopyOnThread(thread, output + dataOffset, localBuffer, chunkBytes
));
    }
    return HCCL_SUCCESS
;
}

// 512 KiB等小消息采用两阶段拓扑感知广播：
// 1) root使用7条worker并行发送给Server 0其余7张卡；
// 2) Server 0的8张卡分别通过8条Clos链路发送给Server 1对应卡。
// 相比16卡二项树，将关键通信轮次从4轮降为2轮。
HcclResult RunTwoStageSmallBroadcast(const OpParam &param, const AlgResourceCtx &resCtx
,
    const ChannelTable &channels, ThreadHandle controlThread, uint64_t dataOffset, uint64_t chunkBytes
)
{
    char *input = static_cast<char *>(param
.inputPtr);
    char *output = static_cast<char *>(param
.outputPtr);
    char *localBuffer = static_cast<char *>(resCtx
.localBuffer.addr);

    const uint32_t serverId = param.myRank / SERVER_RANK_NUM
;
    const uint32_t localRank = param.myRank % SERVER_RANK_NUM
;

    // 性能用例root位于Server 0（root=0或7）。其他root保留二项树兜底。
    CHK_PRT_RET(param.root >= SERVER_RANK_NUM
,
        HCCL_ERROR("Two-stage small broadcast requires root in Server 0, root[%u]", param.root), HCCL_E_PARA
);

    if (param.myRank == param
.root) {
        // 用户输入先进入本地HCCL Buffer。
        CHK_RET(HcommLocalCopyOnThread(controlThread, localBuffer, input + dataOffset, chunkBytes
));

        // 第一阶段：7条worker并行发送给本Server其余7张卡。
        uint32_t workerIdx = 0
;
        for (uint32_t peerLocalRank = 0; peerLocalRank < SERVER_RANK_NUM; ++peerLocalRank
) {
            if (peerLocalRank == localRank
) {
                continue
;
            }

            const uint32_t peerRank = peerLocalRank
;
            const ChannelInfo *peerChannel = channels[peerRank
];
            ThreadHandle workerThread = resCtx.threads[FIRST_WORKER_THREAD_IDX + workerIdx
];

            CHK_RET(HcommThreadNotifyRecordOnThread
(
                controlThread, workerThread, WorkerReadyNotify(0
)));
            CHK_RET(HcommThreadNotifyWaitOnThread
(
                workerThread, WorkerReadyNotify(0), 0
));

            CHK_PTR_NULL(peerChannel
->remoteCclMem.addr);
            CHK_RET(HcommWriteOnThread(workerThread, peerChannel
->handle,
                peerChannel->remoteCclMem.addr, localBuffer, chunkBytes
));
            CHK_RET(HcommChannelNotifyRecordOnThread
(
                workerThread, peerChannel->handle, SMALL_INTRA_NOTIFY
));
            ++workerIdx
;
        }
    } 
else if (serverId == 0
) {
        // Server 0非root等待第一阶段数据到达。
        CHK_RET(HcommChannelNotifyWaitOnThread
(
            controlThread, channels[param.root]->handle, SMALL_INTRA_NOTIFY, 0
));
    }

    // 第二阶段：Server 0每张卡向Server 1对应卡发送完整小消息。
    if (serverId == 0
) {
        const uint32_t partnerRank = param.myRank + SERVER_RANK_NUM
;
        const ChannelInfo *partnerChannel = channels[partnerRank
];
        CHK_PTR_NULL(partnerChannel
->remoteCclMem.addr);

        CHK_RET(HcommWriteOnThread(controlThread, partnerChannel
->handle,
            partnerChannel->remoteCclMem.addr, localBuffer, chunkBytes
));
        CHK_RET(HcommChannelNotifyRecordOnThread
(
            controlThread, partnerChannel->handle, SMALL_INTER_NOTIFY
));

        // Server 0非root需要把HCCL Buffer复制到用户输出。
        if (param.myRank != param
.root) {
            CHK_RET(HcommLocalCopyOnThread
(
                controlThread, output + dataOffset, localBuffer, chunkBytes
));
        }
    } 
else
 {
        const uint32_t partnerRank = param.myRank - SERVER_RANK_NUM
;
        CHK_RET(HcommChannelNotifyWaitOnThread
(
            controlThread, channels[partnerRank]->handle, SMALL_INTER_NOTIFY, 0
));
        CHK_RET(HcommLocalCopyOnThread
(
            controlThread, output + dataOffset, localBuffer, chunkBytes
));
    }

    return HCCL_SUCCESS
;
}


HcclResult WaitOldTileAck(const OpParam &param, const ChannelTable &channels
,
    ThreadHandle controlThread, uint32_t slot
);

HcclResult LaunchRootCopy(const OpParam &param, const AlgResourceCtx &resCtx
,
    ThreadHandle controlThread, ThreadHandle copyThread, uint64_t dataOffset
,
    uint32_t slot, uint64_t slotOffset, uint64_t tileBytes
);

HcclResult LaunchReceiverCopyAndAck(const OpParam &param, const AlgResourceCtx &resCtx
,
    const ChannelTable &channels, ThreadHandle controlThread, ThreadHandle copyThread
,
    uint64_t dataOffset, uint32_t slot, uint64_t slotOffset, uint64_t tileBytes
);


// 大消息采用两阶段完整分块广播：
// 1) root通过7条Server内直连并行发送完整tile给Server 0其余rank；
// 2) Server 0的8个rank分别通过8条Clos链路发送完整tile给Server 1对应rank。
// 该路径移除Scatter/AllGather，减少通信阶段和Notify数量。
HcclResult LaunchRootFullTileFanout(const OpParam &param, const AlgResourceCtx &resCtx
,
    const ChannelTable &channels, ThreadHandle controlThread, uint32_t slot
,
    uint64_t slotOffset, uint64_t tileBytes
)
{
    char *localBuffer = static_cast<char *>(resCtx
.localBuffer.addr);
    const uint32_t rootLocalRank = param.root % SERVER_RANK_NUM
;

    uint32_t workerIdx = 0
;
    for (uint32_t peerLocalRank = 0; peerLocalRank < SERVER_RANK_NUM; ++peerLocalRank
) {
        if (peerLocalRank == rootLocalRank
) {
            continue
;
        }

        const uint32_t peerRank = peerLocalRank
;
        const ChannelInfo *peerChannel = channels[peerRank
];
        ThreadHandle workerThread = resCtx.threads[FIRST_WORKER_THREAD_IDX + workerIdx
];

        CHK_RET(HcommThreadNotifyRecordOnThread
(
            controlThread, workerThread, WorkerReadyNotify(slot
)));
        CHK_RET(HcommThreadNotifyWaitOnThread
(
            workerThread, WorkerReadyNotify(slot), 0
));

        char *remoteBuffer = static_cast<char *>(peerChannel
->remoteCclMem.addr);
        CHK_PTR_NULL(remoteBuffer
);
        CHK_RET(HcommWriteOnThread(workerThread, peerChannel
->handle,
            remoteBuffer + slotOffset, localBuffer + slotOffset, tileBytes
));
        CHK_RET(HcommChannelNotifyRecordOnThread
(
            workerThread, peerChannel->handle, DataNotify(slot
)));
        ++workerIdx
;
    }

    return HCCL_SUCCESS
;
}

__attribute__((unused
))
HcclResult RunPipelinedTwoStageFanout(const OpParam &param, const AlgResourceCtx &resCtx
,
    const ChannelTable &channels, uint64_t totalBytes, uint64_t tileCapacity
)
{
    ThreadHandle controlThread = resCtx.threads[CONTROL_THREAD_IDX
];
    ThreadHandle copyThread = resCtx.threads[COPY_THREAD_IDX
];
    char *localBuffer = static_cast<char *>(resCtx
.localBuffer.addr);

    const uint32_t serverId = param.myRank / SERVER_RANK_NUM
;

    uint64_t dataOffset = 0
;
    uint64_t tileIndex = 0
;
    while (dataOffset < totalBytes
) {
        const uint64_t tileBytes = std::min(tileCapacity, totalBytes - dataOffset
);
        const uint32_t slot = static_cast<uint32_t>(tileIndex % PIPELINE_SLOT_NUM
);
        const uint64_t slotOffset = static_cast<uint64_t>(slot) * tileCapacity
;

        // 双缓冲slot复用前，等待两轮前所有非root rank完成输出。
        if (param.myRank == param.root && tileIndex >= PIPELINE_SLOT_NUM
) {
            CHK_RET(WaitOldTileAck(param, channels, controlThread, slot
));
        }

        if (param.myRank == param
.root) {
            CHK_RET(LaunchRootCopy(param, resCtx, controlThread, copyThread
,
                dataOffset, slot, slotOffset, tileBytes
));
            CHK_RET(LaunchRootFullTileFanout(param, resCtx, channels
,
                controlThread, slot, slotOffset, tileBytes
));
        } 
else if (serverId == 0
) {
            // Server 0非root等待root完整tile到达。
            CHK_RET(HcommChannelNotifyWaitOnThread
(
                controlThread, channels[param.root]->handle, DataNotify(slot), 0
));

            // 输出拷贝与跨Server发送并行排布。
            CHK_RET(LaunchReceiverCopyAndAck(param, resCtx, channels
,
                controlThread, copyThread, dataOffset, slot, slotOffset, tileBytes
));
        }

        if (serverId == 0
) {
            // 8条Clos链路并行发送完整tile到对应rank。
            const uint32_t partnerRank = param.myRank + SERVER_RANK_NUM
;
            const ChannelInfo *partnerChannel = channels[partnerRank
];
            char *remoteBuffer = static_cast<char *>(partnerChannel
->remoteCclMem.addr);
            CHK_PTR_NULL(remoteBuffer
);

            CHK_RET(HcommWriteOnThread(controlThread, partnerChannel
->handle,
                remoteBuffer + slotOffset, localBuffer + slotOffset, tileBytes
));
            CHK_RET(HcommChannelNotifyRecordOnThread
(
                controlThread, partnerChannel->handle, AllGatherNotify(slot
)));
        } 
else
 {
            const uint32_t partnerRank = param.myRank - SERVER_RANK_NUM
;
            CHK_RET(HcommChannelNotifyWaitOnThread
(
                controlThread, channels[partnerRank]->handle, AllGatherNotify(slot), 0
));
            CHK_RET(LaunchReceiverCopyAndAck(param, resCtx, channels
,
                controlThread, copyThread, dataOffset, slot, slotOffset, tileBytes
));
        }

        dataOffset += tileBytes
;
        ++tileIndex
;
    }

    // 回收最后两个slot尚未消费的ACK。
    if (param.myRank == param
.root) {
        const uint32_t pendingSlotNum = static_cast<uint32_t>
(
            std::
min<uint64_t>(tileIndex, PIPELINE_SLOT_NUM
));
        for (uint32_t slot = 0; slot < pendingSlotNum; ++slot
) {
            CHK_RET(WaitOldTileAck(param, channels, controlThread, slot
));
        }
    }

    return HCCL_SUCCESS
;
}

void BuildSlices(uint64_t tileBytes, uint64_t typeSize
,
    std::
array<uint64_t, SERVER_RANK_NUM> &sliceOffsets
,
    std::
array<uint64_t, SERVER_RANK_NUM> &sliceBytes
)
{
    const uint64_t tileCount = tileBytes / typeSize
;
    const uint64_t baseCount = tileCount / SERVER_RANK_NUM
;
    const uint64_t remainder = tileCount % SERVER_RANK_NUM
;

    uint64_t currentCount = 0
;
    for (uint32_t idx = 0; idx < SERVER_RANK_NUM; ++idx
) {
        const uint64_t count = baseCount + (idx < remainder ? 1 : 0
);
        sliceOffsets[idx] = currentCount * typeSize
;
        sliceBytes[idx] = count * typeSize
;
        currentCount += count
;
    }
}

HcclResult WaitOldTileAck(const OpParam &param, const ChannelTable &channels, ThreadHandle controlThread, uint32_t slot
)
{
    for (uint32_t rank = 0; rank < param.rankSize; ++rank
) {
        if (rank == param
.root) {
            continue
;
        }
        CHK_RET(HcommChannelNotifyWaitOnThread
(
            controlThread, channels[rank]->handle, AckNotify(slot), 0
));
    }
    return HCCL_SUCCESS
;
}

HcclResult LaunchRootCopy(const OpParam &param, const AlgResourceCtx &resCtx
,
    ThreadHandle controlThread, ThreadHandle copyThread, uint64_t dataOffset
,
    uint32_t slot, uint64_t slotOffset, uint64_t tileBytes
)
{
    char *input = static_cast<char *>(param
.inputPtr);
    char *localBuffer = static_cast<char *>(resCtx
.localBuffer.addr);

    CHK_RET(HcommThreadNotifyRecordOnThread(controlThread, copyThread, CopyStartNotify(slot
)));
    CHK_RET(HcommThreadNotifyWaitOnThread(copyThread, CopyStartNotify(slot), 0
));
    CHK_RET(HcommLocalCopyOnThread(copyThread, localBuffer + slotOffset, input + dataOffset, tileBytes
));
    CHK_RET(HcommThreadNotifyRecordOnThread(copyThread, controlThread, CopyReadyNotify(slot
)));
    CHK_RET(HcommThreadNotifyWaitOnThread(controlThread, CopyReadyNotify(slot), 0
));
    return HCCL_SUCCESS
;
}

HcclResult LaunchReceiverCopyAndAck(const OpParam &param, const AlgResourceCtx &resCtx
,
    const ChannelTable &channels, ThreadHandle controlThread, ThreadHandle copyThread
,
    uint64_t dataOffset, uint32_t slot, uint64_t slotOffset, uint64_t tileBytes
)
{
    char *output = static_cast<char *>(param
.outputPtr);
    char *localBuffer = static_cast<char *>(resCtx
.localBuffer.addr);

    CHK_RET(HcommThreadNotifyRecordOnThread(controlThread, copyThread, CopyStartNotify(slot
)));
    CHK_RET(HcommThreadNotifyWaitOnThread(copyThread, CopyStartNotify(slot), 0
));
    CHK_RET(HcommLocalCopyOnThread(copyThread, output + dataOffset, localBuffer + slotOffset, tileBytes
));
    CHK_RET(HcommChannelNotifyRecordOnThread
(
        copyThread, channels[param.root]->handle, AckNotify(slot
)));
    return HCCL_SUCCESS
;
}

HcclResult LaunchRootScatterAndAllGather(const OpParam &param, const AlgResourceCtx &resCtx
,
    const ChannelTable &channels, ThreadHandle controlThread, uint32_t slot, uint64_t slotOffset
,
    const std::array<uint64_t, SERVER_RANK_NUM> &sliceOffsets
,
    const std::array<uint64_t, SERVER_RANK_NUM> &sliceBytes
)
{
    char *localBuffer = static_cast<char *>(resCtx
.localBuffer.addr);
    const uint32_t rootLocalRank = param.root % SERVER_RANK_NUM
;

    uint32_t workerIdx = 0
;
    for (uint32_t peerLocalRank = 0; peerLocalRank < SERVER_RANK_NUM; ++peerLocalRank
) {
        if (peerLocalRank == rootLocalRank
) {
            continue
;
        }

        const uint32_t peerRank = peerLocalRank
;
        const ChannelInfo *peerChannel = channels[peerRank
];
        ThreadHandle workerThread = resCtx.threads[FIRST_WORKER_THREAD_IDX + workerIdx
];

        CHK_RET(HcommThreadNotifyRecordOnThread(controlThread, workerThread, WorkerReadyNotify(slot
)));
        CHK_RET(HcommThreadNotifyWaitOnThread(workerThread, WorkerReadyNotify(slot), 0
));

        char *remoteBuffer = static_cast<char *>(peerChannel
->remoteCclMem.addr);
        CHK_PTR_NULL(remoteBuffer
);

        // Scatter：把该peer负责的分片送给它。
        if (sliceBytes[peerLocalRank] > 0
) {
            CHK_RET(HcommWriteOnThread(workerThread, peerChannel
->handle,
                remoteBuffer + slotOffset + sliceOffsets[peerLocalRank
],
                localBuffer + slotOffset + sliceOffsets[peerLocalRank], sliceBytes[peerLocalRank
]));
            CHK_RET(HcommChannelNotifyRecordOnThread
(
                workerThread, peerChannel->handle, DataNotify(slot
)));
        }

        // root本来就拥有完整tile；再把root负责的分片发给该peer，用于机内AllGather。
        if (sliceBytes[rootLocalRank] > 0
) {
            CHK_RET(HcommWriteOnThread(workerThread, peerChannel
->handle,
                remoteBuffer + slotOffset + sliceOffsets[rootLocalRank
],
                localBuffer + slotOffset + sliceOffsets[rootLocalRank], sliceBytes[rootLocalRank
]));
            CHK_RET(HcommChannelNotifyRecordOnThread
(
                workerThread, peerChannel->handle, AllGatherNotify(slot
)));
        }
        ++workerIdx
;
    }
    return HCCL_SUCCESS
;
}

HcclResult LaunchLocalAllGather(const OpParam &param, const AlgResourceCtx &resCtx
,
    const ChannelTable &channels, ThreadHandle controlThread, uint32_t slot, uint64_t slotOffset
,
    const std::array<uint64_t, SERVER_RANK_NUM> &sliceOffsets
,
    const std::array<uint64_t, SERVER_RANK_NUM> &sliceBytes
)
{
    char *localBuffer = static_cast<char *>(resCtx
.localBuffer.addr);
    const uint32_t serverId = param.myRank / SERVER_RANK_NUM
;
    const uint32_t localRank = param.myRank % SERVER_RANK_NUM
;
    const uint32_t serverBaseRank = serverId * SERVER_RANK_NUM
;

    uint32_t workerIdx = 0
;
    for (uint32_t peerLocalRank = 0; peerLocalRank < SERVER_RANK_NUM; ++peerLocalRank
) {
        if (peerLocalRank == localRank
) {
            continue
;
        }

        const uint32_t peerRank = serverBaseRank + peerLocalRank
;
        // Server 0上的root已经保留完整输入，不需要其他rank把分片写回root。
        if (serverId == 0 && peerRank == param
.root) {
            continue
;
        }

        const ChannelInfo *peerChannel = channels[peerRank
];
        ThreadHandle workerThread = resCtx.threads[FIRST_WORKER_THREAD_IDX + workerIdx
];
        CHK_RET(HcommThreadNotifyRecordOnThread(controlThread, workerThread, WorkerReadyNotify(slot
)));
        CHK_RET(HcommThreadNotifyWaitOnThread(workerThread, WorkerReadyNotify(slot), 0
));

        if (sliceBytes[localRank] > 0
) {
            char *remoteBuffer = static_cast<char *>(peerChannel
->remoteCclMem.addr);
            CHK_PTR_NULL(remoteBuffer
);
            CHK_RET(HcommWriteOnThread(workerThread, peerChannel
->handle,
                remoteBuffer + slotOffset + sliceOffsets[localRank
],
                localBuffer + slotOffset + sliceOffsets[localRank], sliceBytes[localRank
]));
            CHK_RET(HcommChannelNotifyRecordOnThread
(
                workerThread, peerChannel->handle, AllGatherNotify(slot
)));
        }
        ++workerIdx
;
    }
    return HCCL_SUCCESS
;
}

HcclResult WaitLocalAllGather(const OpParam &param, const ChannelTable &channels
,
    ThreadHandle controlThread, uint32_t slot
)
{
    const uint32_t serverId = param.myRank / SERVER_RANK_NUM
;
    const uint32_t localRank = param.myRank % SERVER_RANK_NUM
;
    const uint32_t serverBaseRank = serverId * SERVER_RANK_NUM
;

    for (uint32_t peerLocalRank = 0; peerLocalRank < SERVER_RANK_NUM; ++peerLocalRank
) {
        if (peerLocalRank == localRank
) {
            continue
;
        }
        const uint32_t peerRank = serverBaseRank + peerLocalRank
;
        CHK_RET(HcommChannelNotifyWaitOnThread
(
            controlThread, channels[peerRank]->handle, AllGatherNotify(slot), 0
));
    }
    return HCCL_SUCCESS
;
}

__attribute__((unused
))
HcclResult RunPipelinedTwoServerMesh(const OpParam &param, const AlgResourceCtx &resCtx
,
    const ChannelTable &channels, uint64_t totalBytes, uint64_t typeSize, uint64_t tileCapacity
)
{
    ThreadHandle controlThread = resCtx.threads[CONTROL_THREAD_IDX
];
    ThreadHandle copyThread = resCtx.threads[COPY_THREAD_IDX
];
    char *localBuffer = static_cast<char *>(resCtx
.localBuffer.addr);

    const uint32_t serverId = param.myRank / SERVER_RANK_NUM
;
    const uint32_t localRank = param.myRank % SERVER_RANK_NUM
;

    uint64_t dataOffset = 0
;
    uint64_t tileIndex = 0
;
    while (dataOffset < totalBytes
) {
        const uint64_t tileBytes = std::min(tileCapacity, totalBytes - dataOffset
);
        const uint32_t slot = static_cast<uint32_t>(tileIndex % PIPELINE_SLOT_NUM
);
        const uint64_t slotOffset = static_cast<uint64_t>(slot) * tileCapacity
;

        std::
array<uint64_t, SERVER_RANK_NUM> sliceOffsets
{};
        std::
array<uint64_t, SERVER_RANK_NUM> sliceBytes
{};
        BuildSlices(tileBytes, typeSize, sliceOffsets, sliceBytes
);

        // 四槽流水：slot复用前，root等待四轮之前tile的接收完成ACK。
        // 允许最多4个tile同时处于复制、通信或输出阶段。
        if (param.myRank == param.root && tileIndex >= PIPELINE_SLOT_NUM
) {
            CHK_RET(WaitOldTileAck(param, channels, controlThread, slot
));
        }

        if (param.myRank == param
.root) {
            CHK_RET(LaunchRootCopy(param, resCtx, controlThread, copyThread
,
                dataOffset, slot, slotOffset, tileBytes
));
            CHK_RET(LaunchRootScatterAndAllGather(param, resCtx, channels
,
                controlThread, slot, slotOffset, sliceOffsets, sliceBytes
));
        } 
else if (serverId == 0
) {
            // Server 0非root：先接收root的Scatter分片。
            CHK_RET(HcommChannelNotifyWaitOnThread
(
                controlThread, channels[param.root]->handle, DataNotify(slot), 0
));
        }

        // 8条Clos链路并行：Server 0每个rank发送自己的1/8分片到Server 1对应rank。
        const uint32_t partnerRank = serverId == 0
            ? 
param.myRank + SERVER_RANK_NUM
            : 
param.myRank - SERVER_RANK_NUM
;
        const ChannelInfo *partnerChannel = channels[partnerRank
];
        if (sliceBytes[localRank] > 0
) {
            if (serverId == 0
) {
                char *remoteBuffer = static_cast<char *>(partnerChannel
->remoteCclMem.addr);
                CHK_PTR_NULL(remoteBuffer
);
                CHK_RET(HcommWriteOnThread(controlThread, partnerChannel
->handle,
                    remoteBuffer + slotOffset + sliceOffsets[localRank
],
                    localBuffer + slotOffset + sliceOffsets[localRank], sliceBytes[localRank
]));
                CHK_RET(HcommChannelNotifyRecordOnThread
(
                    controlThread, partnerChannel->handle, DataNotify(slot
)));
            } 
else
 {
                CHK_RET(HcommChannelNotifyWaitOnThread
(
                    controlThread, partnerChannel->handle, DataNotify(slot), 0
));
            }
        }

        if (param.myRank != param
.root) {
            CHK_RET(LaunchLocalAllGather(param, resCtx, channels, controlThread
,
                slot, slotOffset, sliceOffsets, sliceBytes
));
            CHK_RET(WaitLocalAllGather(param, channels, controlThread, slot
));
            CHK_RET(LaunchReceiverCopyAndAck(param, resCtx, channels
,
                controlThread, copyThread, dataOffset, slot, slotOffset, tileBytes
));
        }

        dataOffset += tileBytes
;
        ++tileIndex
;
    }

    // 循环中只在复用slot之前等待旧tile的ACK。最后仍有最多四个slot的
    // ACK尚未消费，必须在算子结束前统一回收，否则checker会报告
    // cross-rank Record没有匹配的Wait。
    if (param.myRank == param
.root) {
        const uint32_t pendingSlotNum = static_cast<uint32_t>
(
            std::
min<uint64_t>(tileIndex, PIPELINE_SLOT_NUM
));
        for (uint32_t slot = 0; slot < pendingSlotNum; ++slot
) {
            CHK_RET(WaitOldTileAck(param, channels, controlThread, slot
));
        }
    }

    return HCCL_SUCCESS
;
}
} 
// namespace

namespace ops_hccl
 {
HcclResult ExecOp(const OpParam &param, const AlgResourceCtx &resCtx
)
{
    HCCL_INFO("Executing four-slot pipelined Broadcast on Ascend NPU"
);

    CHK_PRT_RET(param.root >= param
.rankSize,
        HCCL_ERROR("Invalid root[%u], rankSize[%u]", param.root, param.rankSize), HCCL_E_PARA
);

    if (param.rankSize <= 1 || param.count == 0
) {
        return HCCL_SUCCESS
;
    }

    CHK_PRT_RET(param.rankSize != TOTAL_RANK_NUM
,
        HCCL_ERROR("Unsupported rankSize[%u]", param.rankSize), HCCL_E_PARA
);
    CHK_PRT_RET(resCtx.threads.size() < THREAD_NUM
,
        HCCL_ERROR("Need %u threads, got[%zu]", THREAD_NUM, resCtx.threads.size()), HCCL_E_INTERNAL
);
    CHK_PTR_NULL(resCtx
.localBuffer.addr);

    const auto sizeIter = SIZE_TABLE.find(param
.dataType);
    CHK_PRT_RET(sizeIter == SIZE_TABLE
.end(),
        HCCL_ERROR("Unsupported dataType[%d]", static_cast<int32_t>(param.dataType)), HCCL_E_PARA
);
    const uint64_t typeSize = sizeIter
->second;
    CHK_PRT_RET(param.count > std::numeric_limits<uint64_t>::max() / typeSize
,
        HCCL_ERROR("Data size overflow, count[%llu]", static_cast<unsigned long long>(param.count)), HCCL_E_PARA
);

    const uint64_t totalBytes = param.count * typeSize
;
    uint64_t bufferCapacity = resCtx
.localBuffer.size;
    for (const auto &channel : resCtx
.channels) {
        bufferCapacity = std::min(bufferCapacity, channel
.remoteCclMem.size);
    }
    bufferCapacity -= bufferCapacity % typeSize
;
    CHK_PRT_RET(bufferCapacity == 0, HCCL_ERROR("HCCL buffer size is zero"), HCCL_E_INTERNAL
);
    CHK_PRT_RET(resCtx.channels.size() + 1 != param
.rankSize,
        HCCL_ERROR("Channel count[%zu] does not match rankSize[%u]", resCtx.channels.size(), param
.rankSize),
        HCCL_E_INTERNAL
);

    ChannelTable channels
{};
    CHK_RET(BuildChannelTable(param, resCtx, channels
));

    ThreadHandle controlThread = resCtx.threads[CONTROL_THREAD_IDX
];

    // 极小消息继续使用二项树，避免7条worker启动开销。
    if (totalBytes <= TINY_DATA_THRESHOLD || param.root >= SERVER_RANK_NUM
) {
        CHK_RET(RunBinomialTreeChunk(param, resCtx, channels, controlThread, 0, totalBytes
));
        return HCCL_SUCCESS
;
    }

    // 512 KiB性能用例使用两阶段8路并行广播，大数据路径完全保持v3.1不变。
    if (totalBytes <= SMALL_DATA_THRESHOLD
) {
        CHK_RET(RunTwoStageSmallBroadcast
(
            param, resCtx, channels, controlThread, 0, totalBytes
));
        return HCCL_SUCCESS
;
    }

    uint64_t tileCapacity = std::min(PIPELINE_TILE_BYTES, bufferCapacity / PIPELINE_SLOT_NUM
);
    tileCapacity -= tileCapacity % typeSize
;
    CHK_PRT_RET(tileCapacity == 0
,
        HCCL_ERROR("Pipeline tile capacity is zero, bufferCapacity[%llu]"
,
            static_cast<unsigned long long>(bufferCapacity)), HCCL_E_INTERNAL
);

    // 大数据恢复v3.1稳定路径：Scatter -> 8路Clos并行 -> AllGather。
    // 仅保留v5在512 KiB小消息上的两阶段优化。
    CHK_RET(RunPipelinedTwoServerMesh
(
        param, resCtx, channels, totalBytes, typeSize, tileCapacity
));
    return HCCL_SUCCESS
;
}
} 
// namespace ops_hccl