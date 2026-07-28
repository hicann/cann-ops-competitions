#include <vector>
#include <hcomm/hcomm_primitives.h>
#include "common.h"
#include "custom.h"
#include "ccu_kernel.h"
#ifndef CCU_CHK_RET
#define CCU_CHK_RET(call) \
    do { \
        const CcuResult ccuRet = (call); \
        if (ccuRet != CCU_SUCCESS) { \
            return ccuRet; \
        } \
    } while (0)
#endif
namespace {

constexpr uint32_t OUTPUT_XN_ID = 1;
constexpr uint32_t TOKEN_XN_ID = 2;
constexpr uint32_t PHASE_SYNC_ID = 3;
constexpr uint32_t FINAL_SYNC_ID = 4;
constexpr uint32_t CKE_IDX_0 = 0;

constexpr uint16_t OUTPUT_MASK = static_cast<uint16_t>(1U << OUTPUT_XN_ID);
constexpr uint16_t TOKEN_MASK = static_cast<uint16_t>(1U << TOKEN_XN_ID);
constexpr uint16_t PRE_SYNC_MASK = static_cast<uint16_t>(OUTPUT_MASK | TOKEN_MASK);
constexpr uint16_t PHASE_SYNC_MASK = static_cast<uint16_t>(1U << PHASE_SYNC_ID);
constexpr uint16_t FINAL_SYNC_MASK = static_cast<uint16_t>(1U << FINAL_SYNC_ID);

constexpr uint32_t DIRECT_ARG_BASE_ADDR = 0;
constexpr uint32_t DIRECT_ARG_MEM_TOKEN = 1;
constexpr uint32_t DIRECT_ARG_DATA_SIZE = 2;

constexpr uint32_t SAG_ARG_BASE_ADDR = 0;
constexpr uint32_t SAG_ARG_MEM_TOKEN = 1;
constexpr uint32_t SAG_ARG_SLICE_OFFSET_BASE = 2;

constexpr uint32_t P84_ARG_ROOT_OFFSET = 2;
constexpr uint32_t P84_ARG_ROOT_SIZE = 3;
constexpr uint32_t P84_ARG_PEER_OFFSET_BASE = 4;

constexpr uint32_t CHAIN_ARG_BASE_ADDR = 0;
constexpr uint32_t CHAIN_ARG_MEM_TOKEN = 1;
constexpr uint32_t CHAIN_ARG_OFFSET_BASE = 2;
constexpr uint32_t CHAIN_SLOT_COUNT = 4;
constexpr uint32_t CHAIN_DATA_BIT_BASE = 3;
constexpr uint32_t CHAIN_ACK_BIT_BASE = 7;
constexpr uint32_t CHAIN_FINAL_BIT = 11;
constexpr uint16_t CHAIN_FINAL_MASK = static_cast<uint16_t>(1U << CHAIN_FINAL_BIT);

int32_t FindRemoteRankIndex(const BroadcastCcuKernelArg *kernelArg, uint32_t remoteRank)
{
    for (uint32_t idx = 0; idx < kernelArg->channelCount; ++idx) {
        if (kernelArg->remoteRanks[idx] == remoteRank) return static_cast<int32_t>(idx);
    }
    return -1;
}

CcuResult ExchangeScatterBufferInfo(BroadcastCcuKernelArg *kernelArg, AscendC::ccu::Variable &localBase, AscendC::ccu::Variable &localToken)
{
    using namespace AscendC::ccu;
    if (kernelArg->myRank == kernelArg->root) {
        for (uint32_t idx = 0; idx < kernelArg->channelCount; ++idx) {
            CCU_CHK_RET(NotifyWait(kernelArg->channels[idx], CKE_IDX_0, PRE_SYNC_MASK));
        }
        return CCU_SUCCESS;
    }
    const int32_t rootIndex = FindRemoteRankIndex(kernelArg, kernelArg->root);
    if (rootIndex < 0) return CCU_SUCCESS;
    
    const ChannelHandle rootChannel = kernelArg->channels[static_cast<uint32_t>(rootIndex)];
    CCU_CHK_RET(WriteVariableWithNotify(rootChannel, localBase, OUTPUT_XN_ID, CKE_IDX_0, OUTPUT_MASK));
    CCU_CHK_RET(WriteVariableWithNotify(rootChannel, localToken, TOKEN_XN_ID, CKE_IDX_0, TOKEN_MASK));
    return CCU_SUCCESS;
}

CcuResult ExchangeChainBufferInfo(BroadcastCcuKernelArg *kernelArg, AscendC::ccu::Variable &localBase, AscendC::ccu::Variable &localToken)
{
    using namespace AscendC::ccu;
    const bool hasPrev = kernelArg->prevRank != INVALID_VALUE_RANKID;
    const bool hasNext = kernelArg->nextRank != INVALID_VALUE_RANKID;
    if (hasPrev) {
        const ChannelHandle prevChannel = kernelArg->channels[kernelArg->prevChannelIndex];
        CCU_CHK_RET(WriteVariableWithNotify(prevChannel, localBase, OUTPUT_XN_ID, CKE_IDX_0, OUTPUT_MASK));
        CCU_CHK_RET(WriteVariableWithNotify(prevChannel, localToken, TOKEN_XN_ID, CKE_IDX_0, TOKEN_MASK));
    }
    if (hasNext) {
        const ChannelHandle nextChannel = kernelArg->channels[kernelArg->nextChannelIndex];
        CCU_CHK_RET(NotifyWait(nextChannel, CKE_IDX_0, PRE_SYNC_MASK));
    }
    return CCU_SUCCESS;
}

CcuResult InitLargeBaseContext(BroadcastCcuKernelArg *kernelArg, AscendC::ccu::Variable &localBase, AscendC::ccu::Variable &localToken, std::vector<AscendC::ccu::Variable> &remoteBase, std::vector<AscendC::ccu::Variable> &remoteToken)
{
    using namespace AscendC::ccu;
    remoteBase.resize(kernelArg->channelCount);
    remoteToken.resize(kernelArg->channelCount);
    for (uint32_t idx = 0; idx < kernelArg->channelCount; ++idx) {
        remoteBase[idx] = GetResByChannel<Variable>(kernelArg->channels[idx], OUTPUT_XN_ID);
        remoteToken[idx] = GetResByChannel<Variable>(kernelArg->channels[idx], TOKEN_XN_ID);
    }
    CCU_CHK_RET(LoadArg(localBase, SAG_ARG_BASE_ADDR));
    CCU_CHK_RET(LoadArg(localToken, SAG_ARG_MEM_TOKEN));
    return CCU_SUCCESS;
}

CcuResult RunScatterAndSignalLean(BroadcastCcuKernelArg *kernelArg, AscendC::ccu::Variable &localBase, AscendC::ccu::Variable &localToken, std::vector<AscendC::ccu::Variable> &remoteBase, std::vector<AscendC::ccu::Variable> &remoteToken)
{
    using namespace AscendC::ccu;
    if (kernelArg->myRank == kernelArg->root) {
        std::vector<Variable> peerOffsets(kernelArg->channelCount);
        std::vector<Variable> peerSizes(kernelArg->channelCount);
        const uint32_t sizeBase = SAG_ARG_SLICE_OFFSET_BASE + kernelArg->channelCount;
        for (uint32_t idx = 0; idx < kernelArg->channelCount; ++idx) {
            CCU_CHK_RET(LoadArg(peerOffsets[idx], SAG_ARG_SLICE_OFFSET_BASE + idx));
            CCU_CHK_RET(LoadArg(peerSizes[idx], sizeBase + idx));
        }

        Event scatterEvent;
        uint16_t totalMask = 0;
        for (uint32_t idx = 0; idx < kernelArg->channelCount; ++idx) {
            LocalAddr src; src.addr = localBase; src.addr += peerOffsets[idx]; src.token = localToken;
            RemoteAddr dst; dst.addr = remoteBase[idx]; dst.addr += peerOffsets[idx]; dst.token = remoteToken[idx];
            const uint16_t eventMask = static_cast<uint16_t>(1U << idx);
            CCU_CHK_RET(Write(kernelArg->channels[idx], dst, src, peerSizes[idx], scatterEvent, eventMask));
            totalMask = static_cast<uint16_t>(totalMask | eventMask);
        }
        CCU_CHK_RET(EventWait(scatterEvent, totalMask));
        for (uint32_t idx = 0; idx < kernelArg->channelCount; ++idx) {
            CCU_CHK_RET(NotifyRecord(kernelArg->channels[idx], CKE_IDX_0, PHASE_SYNC_MASK));
        }
        return CCU_SUCCESS;
    }

    const int32_t rootIndex = FindRemoteRankIndex(kernelArg, kernelArg->root);
    if (rootIndex >= 0) {
        CCU_CHK_RET(NotifyWait(kernelArg->channels[static_cast<uint32_t>(rootIndex)], CKE_IDX_0, PHASE_SYNC_MASK));
    }
    return CCU_SUCCESS;
}

CcuResult RunAllGatherLean(BroadcastCcuKernelArg *kernelArg,
    AscendC::ccu::Variable &localBase,
    AscendC::ccu::Variable &localToken,
    std::vector<AscendC::ccu::Variable> &remoteBase,
    std::vector<AscendC::ccu::Variable> &remoteToken)
{
    using namespace AscendC::ccu;

    Variable ownOffset;
    Variable ownSize;
    // AllGather每个rank只发送自己的slice，使用连续的2、3号TaskArg。
    CCU_CHK_RET(LoadArg(ownOffset, SAG_ARG_SLICE_OFFSET_BASE));
    CCU_CHK_RET(LoadArg(ownSize, SAG_ARG_SLICE_OFFSET_BASE + 1U));

    LocalAddr src;
    src.addr = localBase;
    src.addr += ownOffset;
    src.token = localToken;

    Event gatherEvent;
    uint16_t totalMask = 0;
    for (uint32_t idx = 0; idx < kernelArg->channelCount; ++idx) {
        // 16卡SAG中root原本就持有完整输入，peer无需把自己的slice写回root。
        if (kernelArg->rankSize == 16 &&
            kernelArg->myRank != kernelArg->root &&
            kernelArg->remoteRanks[idx] == kernelArg->root) {
            continue;
        }

        RemoteAddr dst;
        dst.addr = remoteBase[idx];
        dst.addr += ownOffset;
        dst.token = remoteToken[idx];

        const uint16_t mask = static_cast<uint16_t>(1U << idx);
        CCU_CHK_RET(Write(kernelArg->channels[idx],
            dst,
            src,
            ownSize,
            gatherEvent,
            mask));
        totalMask = static_cast<uint16_t>(totalMask | mask);
    }
    if (totalMask != 0) {
        CCU_CHK_RET(EventWait(gatherEvent, totalMask));
    }
    return CCU_SUCCESS;
}

CcuResult ExchangeBufferInfo(BroadcastCcuKernelArg *kernelArg, AscendC::ccu::Variable &localBase, AscendC::ccu::Variable &localToken)
{
    using namespace AscendC::ccu;
    for (uint32_t idx = 0; idx < kernelArg->channelCount; ++idx) {
        CCU_CHK_RET(WriteVariableWithNotify(kernelArg->channels[idx], localBase, OUTPUT_XN_ID, CKE_IDX_0, OUTPUT_MASK));
        CCU_CHK_RET(WriteVariableWithNotify(kernelArg->channels[idx], localToken, TOKEN_XN_ID, CKE_IDX_0, TOKEN_MASK));
    }
    for (uint32_t idx = 0; idx < kernelArg->channelCount; ++idx) {
        CCU_CHK_RET(NotifyWait(kernelArg->channels[idx], CKE_IDX_0, PRE_SYNC_MASK));
    }
    return CCU_SUCCESS;
}

CcuResult ExchangeAllGatherBufferInfoLean(BroadcastCcuKernelArg *kernelArg,
    AscendC::ccu::Variable &localBase,
    AscendC::ccu::Variable &localToken)
{
    using namespace AscendC::ccu;

    if (kernelArg->rankSize != 16) {
        return ExchangeBufferInfo(kernelArg, localBase, localToken);
    }

    if (kernelArg->myRank == kernelArg->root) {
        for (uint32_t idx = 0; idx < kernelArg->channelCount; ++idx) {
            CCU_CHK_RET(NotifyWait(
                kernelArg->channels[idx], CKE_IDX_0, PRE_SYNC_MASK));
        }
        return CCU_SUCCESS;
    }

    for (uint32_t idx = 0; idx < kernelArg->channelCount; ++idx) {
        CCU_CHK_RET(WriteVariableWithNotify(kernelArg->channels[idx],
            localBase, OUTPUT_XN_ID, CKE_IDX_0, OUTPUT_MASK));
        CCU_CHK_RET(WriteVariableWithNotify(kernelArg->channels[idx],
            localToken, TOKEN_XN_ID, CKE_IDX_0, TOKEN_MASK));
    }
    for (uint32_t idx = 0; idx < kernelArg->channelCount; ++idx) {
        if (kernelArg->remoteRanks[idx] == kernelArg->root) {
            continue;
        }
        CCU_CHK_RET(NotifyWait(
            kernelArg->channels[idx], CKE_IDX_0, PRE_SYNC_MASK));
    }
    return CCU_SUCCESS;
}

CcuResult PairwiseBarrier(BroadcastCcuKernelArg *kernelArg, uint16_t mask)
{
    using namespace AscendC::ccu;
    for (uint32_t idx = 0; idx < kernelArg->channelCount; ++idx) {
        CCU_CHK_RET(NotifyRecord(kernelArg->channels[idx], CKE_IDX_0, mask));
    }
    for (uint32_t idx = 0; idx < kernelArg->channelCount; ++idx) {
        CCU_CHK_RET(NotifyWait(kernelArg->channels[idx], CKE_IDX_0, mask));
    }
    return CCU_SUCCESS;
}

CcuResult OneWayAllGatherCompletion(BroadcastCcuKernelArg *kernelArg, uint16_t mask)
{
    using namespace AscendC::ccu;

    if (kernelArg->myRank == kernelArg->root) {
        for (uint32_t idx = 0; idx < kernelArg->channelCount; ++idx) {
            CCU_CHK_RET(NotifyRecord(kernelArg->channels[idx], CKE_IDX_0, mask));
        }
        return CCU_SUCCESS;
    }

    for (uint32_t idx = 0; idx < kernelArg->channelCount; ++idx) {
        if (kernelArg->remoteRanks[idx] == kernelArg->root) {
            continue;
        }
        CCU_CHK_RET(NotifyRecord(kernelArg->channels[idx], CKE_IDX_0, mask));
    }
    for (uint32_t idx = 0; idx < kernelArg->channelCount; ++idx) {
        CCU_CHK_RET(NotifyWait(kernelArg->channels[idx], CKE_IDX_0, mask));
    }
    return CCU_SUCCESS;
}

CcuResult Run8p4ScatterPrefetch(BroadcastCcuKernelArg *kernelArg,
    AscendC::ccu::Variable &localBase,
    AscendC::ccu::Variable &localToken,
    std::vector<AscendC::ccu::Variable> &remoteBase,
    std::vector<AscendC::ccu::Variable> &remoteToken)
{
    using namespace AscendC::ccu;

    if (kernelArg->myRank != kernelArg->root) {
        const int32_t rootIndex = FindRemoteRankIndex(kernelArg, kernelArg->root);
        if (rootIndex >= 0) {
            // 只等待“本rank自己的slice已就绪”。root slice随后在后台预取，
            // 本rank可以立即开始向其他非root传播自己的slice。
            CCU_CHK_RET(NotifyWait(kernelArg->channels[static_cast<uint32_t>(rootIndex)],
                CKE_IDX_0,
                PHASE_SYNC_MASK));
        }
        return CCU_SUCCESS;
    }

    Variable rootOffset;
    Variable rootSize;
    CCU_CHK_RET(LoadArg(rootOffset, P84_ARG_ROOT_OFFSET));
    CCU_CHK_RET(LoadArg(rootSize, P84_ARG_ROOT_SIZE));

    std::vector<Variable> peerOffsets(kernelArg->channelCount);
    std::vector<Variable> peerSizes(kernelArg->channelCount);
    const uint32_t peerSizeBase = P84_ARG_PEER_OFFSET_BASE + kernelArg->channelCount;
    for (uint32_t idx = 0; idx < kernelArg->channelCount; ++idx) {
        CCU_CHK_RET(LoadArg(peerOffsets[idx], P84_ARG_PEER_OFFSET_BASE + idx));
        CCU_CHK_RET(LoadArg(peerSizes[idx], peerSizeBase + idx));
    }

    // 第一步只发送每个peer自己的slice。完成后立即放行非root AllGather，
    // 不等待root slice预取，缩短Scatter->AllGather关键路径。
    Event peerScatterEvent;
    uint16_t peerMaskAll = 0;
    for (uint32_t idx = 0; idx < kernelArg->channelCount; ++idx) {
        LocalAddr src;
        src.addr = localBase;
        src.addr += peerOffsets[idx];
        src.token = localToken;

        RemoteAddr dst;
        dst.addr = remoteBase[idx];
        dst.addr += peerOffsets[idx];
        dst.token = remoteToken[idx];

        const uint16_t mask = static_cast<uint16_t>(1U << idx);
        CCU_CHK_RET(Write(kernelArg->channels[idx],
            dst,
            src,
            peerSizes[idx],
            peerScatterEvent,
            mask));
        peerMaskAll = static_cast<uint16_t>(peerMaskAll | mask);
    }
    if (peerMaskAll != 0) {
        CCU_CHK_RET(EventWait(peerScatterEvent, peerMaskAll));
    }
    for (uint32_t idx = 0; idx < kernelArg->channelCount; ++idx) {
        CCU_CHK_RET(NotifyRecord(
            kernelArg->channels[idx], CKE_IDX_0, PHASE_SYNC_MASK));
    }

    // 第二步在非root已经开始AllGather后，root并行预发自己的slice。
    // root slice与各非root的own slice位于不同地址区间，不存在写冲突。
    Event rootPrefetchEvent;
    uint16_t rootMaskAll = 0;
    for (uint32_t idx = 0; idx < kernelArg->channelCount; ++idx) {
        LocalAddr src;
        src.addr = localBase;
        src.addr += rootOffset;
        src.token = localToken;

        RemoteAddr dst;
        dst.addr = remoteBase[idx];
        dst.addr += rootOffset;
        dst.token = remoteToken[idx];

        const uint16_t mask = static_cast<uint16_t>(1U << idx);
        CCU_CHK_RET(Write(kernelArg->channels[idx],
            dst,
            src,
            rootSize,
            rootPrefetchEvent,
            mask));
        rootMaskAll = static_cast<uint16_t>(rootMaskAll | mask);
    }
    if (rootMaskAll != 0) {
        CCU_CHK_RET(EventWait(rootPrefetchEvent, rootMaskAll));
    }
    return CCU_SUCCESS;
}

CcuResult Exchange8p4AllGatherInfo(BroadcastCcuKernelArg *kernelArg,
    AscendC::ccu::Variable &localBase,
    AscendC::ccu::Variable &localToken)
{
    using namespace AscendC::ccu;

    if (kernelArg->myRank == kernelArg->root) {
        return CCU_SUCCESS;
    }

    for (uint32_t idx = 0; idx < kernelArg->channelCount; ++idx) {
        if (kernelArg->remoteRanks[idx] == kernelArg->root) {
            continue;
        }
        CCU_CHK_RET(WriteVariableWithNotify(kernelArg->channels[idx],
            localBase,
            OUTPUT_XN_ID,
            CKE_IDX_0,
            OUTPUT_MASK));
        CCU_CHK_RET(WriteVariableWithNotify(kernelArg->channels[idx],
            localToken,
            TOKEN_XN_ID,
            CKE_IDX_0,
            TOKEN_MASK));
    }
    for (uint32_t idx = 0; idx < kernelArg->channelCount; ++idx) {
        if (kernelArg->remoteRanks[idx] == kernelArg->root) {
            continue;
        }
        CCU_CHK_RET(NotifyWait(
            kernelArg->channels[idx], CKE_IDX_0, PRE_SYNC_MASK));
    }
    return CCU_SUCCESS;
}

CcuResult Run8p4AllGatherSkipRoot(BroadcastCcuKernelArg *kernelArg,
    AscendC::ccu::Variable &localBase,
    AscendC::ccu::Variable &localToken,
    std::vector<AscendC::ccu::Variable> &remoteBase,
    std::vector<AscendC::ccu::Variable> &remoteToken)
{
    using namespace AscendC::ccu;

    if (kernelArg->myRank == kernelArg->root) {
        return CCU_SUCCESS;
    }

    Variable ownOffset;
    Variable ownSize;
    CCU_CHK_RET(LoadArg(ownOffset, SAG_ARG_SLICE_OFFSET_BASE));
    CCU_CHK_RET(LoadArg(ownSize, SAG_ARG_SLICE_OFFSET_BASE + 1U));

    LocalAddr src;
    src.addr = localBase;
    src.addr += ownOffset;
    src.token = localToken;

    Event gatherEvent;
    uint16_t totalMask = 0;
    for (uint32_t idx = 0; idx < kernelArg->channelCount; ++idx) {
        if (kernelArg->remoteRanks[idx] == kernelArg->root) {
            continue;
        }

        RemoteAddr dst;
        dst.addr = remoteBase[idx];
        dst.addr += ownOffset;
        dst.token = remoteToken[idx];

        const uint16_t mask = static_cast<uint16_t>(1U << idx);
        CCU_CHK_RET(Write(kernelArg->channels[idx],
            dst,
            src,
            ownSize,
            gatherEvent,
            mask));
        totalMask = static_cast<uint16_t>(totalMask | mask);
    }
    if (totalMask != 0) {
        CCU_CHK_RET(EventWait(gatherEvent, totalMask));
    }
    return CCU_SUCCESS;
}

} // namespace

namespace ops_hccl {

CcuResult CcuBroadcastDirectRootKernel(CcuKernelArg arg)
{
    using namespace AscendC::ccu;
    auto *kernelArg = static_cast<BroadcastCcuKernelArg *>(arg);
    if (kernelArg == nullptr) return CCU_E_PTR;
    if (kernelArg->myRank != kernelArg->root || kernelArg->channelCount == 0 || kernelArg->channelCount >= MAX_RANK_SIZE) return CCU_E_INTERNAL;

    Variable baseAddr, memToken, dataSize;
    CCU_CHK_RET(LoadArg(baseAddr, DIRECT_ARG_BASE_ADDR));
    CCU_CHK_RET(LoadArg(memToken, DIRECT_ARG_MEM_TOKEN));
    CCU_CHK_RET(LoadArg(dataSize, DIRECT_ARG_DATA_SIZE));

    LocalAddr src; src.addr = baseAddr; src.token = memToken;
    Event writeEvent;
    uint16_t totalWriteMask = 0;
    
    // 微秒级极限解耦：Wait 与 Write 合并流水优化指令量
    for (uint32_t idx = 0; idx < kernelArg->channelCount; ++idx) {
        const ChannelHandle channel = kernelArg->channels[idx];
        CCU_CHK_RET(NotifyWait(channel, CKE_IDX_0, PRE_SYNC_MASK));
        RemoteAddr dst; dst.addr = GetResByChannel<Variable>(channel, OUTPUT_XN_ID); dst.token = GetResByChannel<Variable>(channel, TOKEN_XN_ID);
        const uint16_t mask = static_cast<uint16_t>(1U << idx);
        CCU_CHK_RET(Write(channel, dst, src, dataSize, writeEvent, mask));
        totalWriteMask = static_cast<uint16_t>(totalWriteMask | mask);
    }
    CCU_CHK_RET(EventWait(writeEvent, totalWriteMask));
    for (uint32_t idx = 0; idx < kernelArg->channelCount; ++idx) {
        CCU_CHK_RET(NotifyRecord(kernelArg->channels[idx], CKE_IDX_0, PHASE_SYNC_MASK));
    }
    return CCU_SUCCESS;
}

CcuResult CcuBroadcastDirectPeerKernel(CcuKernelArg arg)
{
    using namespace AscendC::ccu;
    auto *kernelArg = static_cast<BroadcastCcuKernelArg *>(arg);
    if (kernelArg == nullptr) return CCU_E_PTR;
    if (kernelArg->myRank == kernelArg->root || kernelArg->channelCount != 1) return CCU_E_INTERNAL;

    Variable baseAddr, memToken;
    CCU_CHK_RET(LoadArg(baseAddr, DIRECT_ARG_BASE_ADDR));
    CCU_CHK_RET(LoadArg(memToken, DIRECT_ARG_MEM_TOKEN));

    const ChannelHandle rootChannel = kernelArg->channels[0];
    CCU_CHK_RET(WriteVariableWithNotify(rootChannel, baseAddr, OUTPUT_XN_ID, CKE_IDX_0, OUTPUT_MASK));
    CCU_CHK_RET(WriteVariableWithNotify(rootChannel, memToken, TOKEN_XN_ID, CKE_IDX_0, TOKEN_MASK));
    CCU_CHK_RET(NotifyWait(rootChannel, CKE_IDX_0, PHASE_SYNC_MASK));
    return CCU_SUCCESS;
}

CcuResult CcuBroadcastScatterKernel(CcuKernelArg arg)
{
    using namespace AscendC::ccu;
    auto *kernelArg = static_cast<BroadcastCcuKernelArg *>(arg);
    if (kernelArg == nullptr) return CCU_E_PTR;
    if (kernelArg->channelCount == 0 || kernelArg->channelCount >= MAX_RANK_SIZE || kernelArg->rankSize == 0 || kernelArg->rankSize > MAX_RANK_SIZE) return CCU_E_INTERNAL;

    Variable localBase, localToken;
    std::vector<Variable> remoteBase, remoteToken;
    CCU_CHK_RET(InitLargeBaseContext(kernelArg, localBase, localToken, remoteBase, remoteToken));
    CCU_CHK_RET(ExchangeScatterBufferInfo(kernelArg, localBase, localToken));
    CCU_CHK_RET(RunScatterAndSignalLean(kernelArg, localBase, localToken, remoteBase, remoteToken));
    return CCU_SUCCESS;
}

CcuResult CcuBroadcastAllGatherKernel(CcuKernelArg arg)
{
    using namespace AscendC::ccu;

    auto *kernelArg = static_cast<BroadcastCcuKernelArg *>(arg);
    if (kernelArg == nullptr) {
        return CCU_E_PTR;
    }
    if (kernelArg->channelCount == 0 || kernelArg->channelCount >= MAX_RANK_SIZE ||
        kernelArg->rankSize == 0 || kernelArg->rankSize > MAX_RANK_SIZE) {
        return CCU_E_INTERNAL;
    }

    Variable localBase;
    Variable localToken;
    CCU_CHK_RET(LoadArg(localBase, SAG_ARG_BASE_ADDR));
    CCU_CHK_RET(LoadArg(localToken, SAG_ARG_MEM_TOKEN));

    std::vector<Variable> remoteBase(kernelArg->channelCount);
    std::vector<Variable> remoteToken(kernelArg->channelCount);
    for (uint32_t idx = 0; idx < kernelArg->channelCount; ++idx) {
        if (kernelArg->rankSize == 16 &&
            kernelArg->myRank != kernelArg->root &&
            kernelArg->remoteRanks[idx] == kernelArg->root) {
            continue;
        }
        remoteBase[idx] =
            GetResByChannel<Variable>(kernelArg->channels[idx], OUTPUT_XN_ID);
        remoteToken[idx] =
            GetResByChannel<Variable>(kernelArg->channels[idx], TOKEN_XN_ID);
    }

    CCU_CHK_RET(ExchangeAllGatherBufferInfoLean(
        kernelArg, localBase, localToken));
    CCU_CHK_RET(RunAllGatherLean(
        kernelArg, localBase, localToken, remoteBase, remoteToken));
    if (kernelArg->rankSize == 16) {
        CCU_CHK_RET(OneWayAllGatherCompletion(kernelArg, PHASE_SYNC_MASK));
    } else {
        CCU_CHK_RET(PairwiseBarrier(kernelArg, PHASE_SYNC_MASK));
    }
    return CCU_SUCCESS;
}

// 【4*1 优化核心】DMA Write 与 EventWait 并行重叠起跑，隐匿流水线气泡！
CcuResult CcuBroadcastChainPipelineKernel(CcuKernelArg arg)
{
    using namespace AscendC::ccu;
    auto *kernelArg = static_cast<BroadcastCcuKernelArg *>(arg);
    if (kernelArg == nullptr) return CCU_E_PTR;
    if (kernelArg->rankSize != 4 || kernelArg->channelCount == 0 || kernelArg->channelCount > 2 || kernelArg->pipelineChunkCount == 0 || kernelArg->pipelineChunkBytes == 0) return CCU_E_INTERNAL;

    const bool hasPrev = kernelArg->prevRank != INVALID_VALUE_RANKID;
    const bool hasNext = kernelArg->nextRank != INVALID_VALUE_RANKID;
    if ((hasPrev && kernelArg->prevChannelIndex >= kernelArg->channelCount) ||
        (hasNext && kernelArg->nextChannelIndex >= kernelArg->channelCount)) return CCU_E_INTERNAL;

    Variable localBase, localToken;
    CCU_CHK_RET(LoadArg(localBase, CHAIN_ARG_BASE_ADDR));
    CCU_CHK_RET(LoadArg(localToken, CHAIN_ARG_MEM_TOKEN));

    std::vector<Variable> remoteBase(kernelArg->channelCount), remoteToken(kernelArg->channelCount);
    for (uint32_t idx = 0; idx < kernelArg->channelCount; ++idx) {
        remoteBase[idx] = GetResByChannel<Variable>(kernelArg->channels[idx], OUTPUT_XN_ID);
        remoteToken[idx] = GetResByChannel<Variable>(kernelArg->channels[idx], TOKEN_XN_ID);
    }

    std::vector<Variable> chunkOffsets(kernelArg->pipelineChunkCount), chunkSizes(kernelArg->pipelineChunkCount);
    for (uint32_t chunk = 0; chunk < kernelArg->pipelineChunkCount; ++chunk) {
        CCU_CHK_RET(LoadArg(chunkOffsets[chunk], CHAIN_ARG_OFFSET_BASE + chunk));
    }
    const uint32_t sizeBase = CHAIN_ARG_OFFSET_BASE + kernelArg->pipelineChunkCount;
    for (uint32_t chunk = 0; chunk < kernelArg->pipelineChunkCount; ++chunk) {
        CCU_CHK_RET(LoadArg(chunkSizes[chunk], sizeBase + chunk));
    }

    CCU_CHK_RET(ExchangeChainBufferInfo(kernelArg, localBase, localToken));

    ChannelHandle prevChannel = hasPrev ? kernelArg->channels[kernelArg->prevChannelIndex] : 0;
    ChannelHandle nextChannel = hasNext ? kernelArg->channels[kernelArg->nextChannelIndex] : 0;
    Event forwardEvent;

    for (uint32_t chunk = 0; chunk < kernelArg->pipelineChunkCount; ++chunk) {
        const uint32_t slot = chunk % CHAIN_SLOT_COUNT;
        const uint16_t dataMask = static_cast<uint16_t>(1U << (CHAIN_DATA_BIT_BASE + slot));
        const uint16_t ackMask = static_cast<uint16_t>(1U << (CHAIN_ACK_BIT_BASE + slot));
        const uint16_t writeMask = static_cast<uint16_t>(1U << slot);

        if (hasPrev) {
            CCU_CHK_RET(NotifyWait(prevChannel, CKE_IDX_0, dataMask));
            CCU_CHK_RET(NotifyRecord(prevChannel, CKE_IDX_0, ackMask));
        }
        
        if (hasNext) {
            if (chunk >= CHAIN_SLOT_COUNT) {
                CCU_CHK_RET(NotifyWait(nextChannel, CKE_IDX_0, ackMask));
            }
            LocalAddr src; src.addr = localBase; src.addr += chunkOffsets[chunk]; src.token = localToken;
            RemoteAddr dst; dst.addr = remoteBase[kernelArg->nextChannelIndex]; dst.addr += chunkOffsets[chunk]; dst.token = remoteToken[kernelArg->nextChannelIndex];
            
            // 下发 Write 后不等完成，异步走入下一个循环去 Wait 前段！
            CCU_CHK_RET(Write(nextChannel, dst, src, chunkSizes[chunk], forwardEvent, writeMask));
        }

        if (hasNext && chunk > 0) {
            const uint32_t prevSlot = (chunk - 1) % CHAIN_SLOT_COUNT;
            const uint16_t prevWriteMask = static_cast<uint16_t>(1U << prevSlot);
            const uint16_t prevDataMask = static_cast<uint16_t>(1U << (CHAIN_DATA_BIT_BASE + prevSlot));
            CCU_CHK_RET(EventWait(forwardEvent, prevWriteMask));
            CCU_CHK_RET(NotifyRecord(nextChannel, CKE_IDX_0, prevDataMask));
        }
    }

    if (hasNext && kernelArg->pipelineChunkCount > 0) {
        const uint32_t lastChunk = kernelArg->pipelineChunkCount - 1;
        const uint32_t lastSlot = lastChunk % CHAIN_SLOT_COUNT;
        const uint16_t lastWriteMask = static_cast<uint16_t>(1U << lastSlot);
        const uint16_t lastDataMask = static_cast<uint16_t>(1U << (CHAIN_DATA_BIT_BASE + lastSlot));
        CCU_CHK_RET(EventWait(forwardEvent, lastWriteMask));
        CCU_CHK_RET(NotifyRecord(nextChannel, CKE_IDX_0, lastDataMask));
    }

    if (hasNext) {
        const uint32_t firstOutstanding = kernelArg->pipelineChunkCount > CHAIN_SLOT_COUNT ? kernelArg->pipelineChunkCount - CHAIN_SLOT_COUNT : 0;
        for (uint32_t chunk = firstOutstanding; chunk < kernelArg->pipelineChunkCount; ++chunk) {
            const uint32_t slot = chunk % CHAIN_SLOT_COUNT;
            CCU_CHK_RET(NotifyWait(nextChannel, CKE_IDX_0, static_cast<uint16_t>(1U << (CHAIN_ACK_BIT_BASE + slot))));
        }
        CCU_CHK_RET(NotifyWait(nextChannel, CKE_IDX_0, CHAIN_FINAL_MASK));
    }
    if (hasPrev) {
        CCU_CHK_RET(NotifyRecord(prevChannel, CKE_IDX_0, CHAIN_FINAL_MASK));
    }
    return CCU_SUCCESS;
}

CcuResult CcuBroadcast8p4ScatterPrefetchKernel(CcuKernelArg arg)
{
    using namespace AscendC::ccu;

    auto *kernelArg = static_cast<BroadcastCcuKernelArg *>(arg);
    if (kernelArg == nullptr) {
        return CCU_E_PTR;
    }
    if (kernelArg->rankSize != 12 || kernelArg->channelCount == 0 ||
        kernelArg->channelCount > 8) {
        return CCU_E_INTERNAL;
    }

    Variable localBase;
    Variable localToken;
    std::vector<Variable> remoteBase;
    std::vector<Variable> remoteToken;
    CCU_CHK_RET(InitLargeBaseContext(
        kernelArg, localBase, localToken, remoteBase, remoteToken));
    CCU_CHK_RET(ExchangeScatterBufferInfo(kernelArg, localBase, localToken));
    CCU_CHK_RET(Run8p4ScatterPrefetch(
        kernelArg, localBase, localToken, remoteBase, remoteToken));
    return CCU_SUCCESS;
}

CcuResult CcuBroadcast8p4AllGatherSkipRootKernel(CcuKernelArg arg)
{
    using namespace AscendC::ccu;

    auto *kernelArg = static_cast<BroadcastCcuKernelArg *>(arg);
    if (kernelArg == nullptr) {
        return CCU_E_PTR;
    }
    if (kernelArg->rankSize != 12 || kernelArg->channelCount == 0 ||
        kernelArg->channelCount > 8) {
        return CCU_E_INTERNAL;
    }

    Variable localBase;
    Variable localToken;
    std::vector<Variable> remoteBase;
    std::vector<Variable> remoteToken;
    CCU_CHK_RET(InitLargeBaseContext(
        kernelArg, localBase, localToken, remoteBase, remoteToken));
    CCU_CHK_RET(Exchange8p4AllGatherInfo(kernelArg, localBase, localToken));
    CCU_CHK_RET(Run8p4AllGatherSkipRoot(
        kernelArg, localBase, localToken, remoteBase, remoteToken));
    CCU_CHK_RET(PairwiseBarrier(kernelArg, FINAL_SYNC_MASK));
    return CCU_SUCCESS;
}

} // namespace ops_hccl