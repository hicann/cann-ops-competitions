#include <array>
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

// 小数据Direct Kernel参数。Broadcast为原地接口，input/output使用同一个buf，
// 因此只加载一次地址，减少一条LoadArg指令。
constexpr uint32_t DIRECT_ARG_BASE_ADDR = 0;
constexpr uint32_t DIRECT_ARG_MEM_TOKEN = 1;

// 大数据SAG Kernel参数：baseAddr, token, offsets[rankSize], sizes[rankSize]。
constexpr uint32_t SAG_ARG_BASE_ADDR = 0;
constexpr uint32_t SAG_ARG_MEM_TOKEN = 1;
constexpr uint32_t SAG_ARG_SLICE_OFFSET_BASE = 2;

// 8+4专用Scatter参数：base/token/root offset/root size/peer offsets/peer sizes。
constexpr uint32_t P84_ARG_ROOT_OFFSET = 2;
constexpr uint32_t P84_ARG_ROOT_SIZE = 3;
constexpr uint32_t P84_ARG_PEER_OFFSET_BASE = 4;
constexpr uint32_t P84_MAX_CHANNELS = 8;

// 4x1链式流水参数：baseAddr, token, offsets[chunkCount], sizes[chunkCount]。
constexpr uint32_t CHAIN_ARG_BASE_ADDR = 0;
constexpr uint32_t CHAIN_ARG_MEM_TOKEN = 1;
constexpr uint32_t CHAIN_MAX_CHANNELS = 2;
constexpr uint32_t CHAIN_SLOT_COUNT = 4;
constexpr uint32_t CHAIN_DATA_BIT_BASE = 3;
constexpr uint32_t CHAIN_ACK_BIT_BASE = 7;
constexpr uint32_t CHAIN_FINAL_BIT = 11;
constexpr uint16_t CHAIN_FINAL_MASK = static_cast<uint16_t>(1U << CHAIN_FINAL_BIT);

int32_t FindRemoteRankIndex(const BroadcastCcuKernelArg *kernelArg, uint32_t remoteRank)
{
    for (uint32_t idx = 0; idx < kernelArg->channelCount; ++idx) {
        if (kernelArg->remoteRanks[idx] == remoteRank) {
            return static_cast<int32_t>(idx);
        }
    }
    return -1;
}

CcuResult ExchangeScatterBufferInfo(BroadcastCcuKernelArg *kernelArg,
    AscendC::ccu::Variable &localBase,
    AscendC::ccu::Variable &localToken)
{
    using namespace AscendC::ccu;

    if (kernelArg->myRank == kernelArg->root) {
        for (uint32_t idx = 0; idx < kernelArg->channelCount; ++idx) {
            CCU_CHK_RET(NotifyWait(kernelArg->channels[idx], CKE_IDX_0, PRE_SYNC_MASK));
        }
        return CCU_SUCCESS;
    }

    const int32_t rootIndex = FindRemoteRankIndex(kernelArg, kernelArg->root);
    if (rootIndex < 0) {
        // 本Die没有root链路，无需参与Scatter阶段。
        return CCU_SUCCESS;
    }
    const ChannelHandle rootChannel = kernelArg->channels[static_cast<uint32_t>(rootIndex)];
    CCU_CHK_RET(WriteVariableWithNotify(
        rootChannel, localBase, OUTPUT_XN_ID, CKE_IDX_0, OUTPUT_MASK));
    CCU_CHK_RET(WriteVariableWithNotify(
        rootChannel, localToken, TOKEN_XN_ID, CKE_IDX_0, TOKEN_MASK));
    return CCU_SUCCESS;
}

CcuResult ExchangeChainBufferInfo(BroadcastCcuKernelArg *kernelArg,
    AscendC::ccu::Variable &localBase,
    AscendC::ccu::Variable &localToken)
{
    using namespace AscendC::ccu;

    const bool hasPrev = kernelArg->prevRank != INVALID_VALUE_RANKID;
    const bool hasNext = kernelArg->nextRank != INVALID_VALUE_RANKID;
    if (hasPrev) {
        const ChannelHandle prevChannel = kernelArg->channels[kernelArg->prevChannelIndex];
        CCU_CHK_RET(WriteVariableWithNotify(
            prevChannel, localBase, OUTPUT_XN_ID, CKE_IDX_0, OUTPUT_MASK));
        CCU_CHK_RET(WriteVariableWithNotify(
            prevChannel, localToken, TOKEN_XN_ID, CKE_IDX_0, TOKEN_MASK));
    }
    if (hasNext) {
        const ChannelHandle nextChannel = kernelArg->channels[kernelArg->nextChannelIndex];
        CCU_CHK_RET(NotifyWait(nextChannel, CKE_IDX_0, PRE_SYNC_MASK));
    }
    return CCU_SUCCESS;
}

CcuResult InitLargeKernelContext(BroadcastCcuKernelArg *kernelArg,
    AscendC::ccu::Variable &localBase,
    AscendC::ccu::Variable &localToken,
    std::vector<AscendC::ccu::Variable> &remoteBase,
    std::vector<AscendC::ccu::Variable> &remoteToken,
    std::vector<AscendC::ccu::Variable> &sliceOffsets,
    std::vector<AscendC::ccu::Variable> &sliceSizes)
{
    using namespace AscendC::ccu;

    remoteBase.resize(kernelArg->channelCount);
    remoteToken.resize(kernelArg->channelCount);
    for (uint32_t idx = 0; idx < kernelArg->channelCount; ++idx) {
        remoteBase[idx] = GetResByChannel<Variable>(kernelArg->channels[idx], OUTPUT_XN_ID);
        remoteToken[idx] = GetResByChannel<Variable>(kernelArg->channels[idx], TOKEN_XN_ID);
    }

    sliceOffsets.resize(kernelArg->rankSize);
    sliceSizes.resize(kernelArg->rankSize);

    CCU_CHK_RET(LoadArg(localBase, SAG_ARG_BASE_ADDR));
    CCU_CHK_RET(LoadArg(localToken, SAG_ARG_MEM_TOKEN));
    for (uint32_t rank = 0; rank < kernelArg->rankSize; ++rank) {
        CCU_CHK_RET(LoadArg(sliceOffsets[rank], SAG_ARG_SLICE_OFFSET_BASE + rank));
    }
    const uint32_t sizeBase = SAG_ARG_SLICE_OFFSET_BASE + kernelArg->rankSize;
    for (uint32_t rank = 0; rank < kernelArg->rankSize; ++rank) {
        CCU_CHK_RET(LoadArg(sliceSizes[rank], sizeBase + rank));
    }

    return CCU_SUCCESS;
}

// Split SAG专用轻量初始化。Scatter/AllGather不再无条件加载全部rank的
// offsets/sizes；只绑定当前Die的远端资源并加载base/token。
CcuResult InitLargeBaseContext(BroadcastCcuKernelArg *kernelArg,
    AscendC::ccu::Variable &localBase,
    AscendC::ccu::Variable &localToken,
    std::vector<AscendC::ccu::Variable> &remoteBase,
    std::vector<AscendC::ccu::Variable> &remoteToken)
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

// root只加载当前Die组内各peer对应的offset/size。非root不加载任何切片参数。
CcuResult RunScatterAndSignalLean(BroadcastCcuKernelArg *kernelArg,
    AscendC::ccu::Variable &localBase,
    AscendC::ccu::Variable &localToken,
    std::vector<AscendC::ccu::Variable> &remoteBase,
    std::vector<AscendC::ccu::Variable> &remoteToken)
{
    using namespace AscendC::ccu;

    if (kernelArg->myRank == kernelArg->root) {
        std::vector<Variable> peerOffsets(kernelArg->channelCount);
        std::vector<Variable> peerSizes(kernelArg->channelCount);
        // 每个Die Scatter Kernel只加载本组实际对端的offset/size。
        // 索引必须连续，否则Launch侧无法用紧凑TaskArg与LoadArg数严格匹配。
        const uint32_t sizeBase = SAG_ARG_SLICE_OFFSET_BASE + kernelArg->channelCount;
        for (uint32_t idx = 0; idx < kernelArg->channelCount; ++idx) {
            CCU_CHK_RET(LoadArg(peerOffsets[idx], SAG_ARG_SLICE_OFFSET_BASE + idx));
            CCU_CHK_RET(LoadArg(peerSizes[idx], sizeBase + idx));
        }

        Event scatterEvent;
        uint16_t totalMask = 0;
        for (uint32_t idx = 0; idx < kernelArg->channelCount; ++idx) {
            LocalAddr src;
            src.addr = localBase;
            src.addr += peerOffsets[idx];
            src.token = localToken;

            RemoteAddr dst;
            dst.addr = remoteBase[idx];
            dst.addr += peerOffsets[idx];
            dst.token = remoteToken[idx];

            const uint16_t eventMask = static_cast<uint16_t>(1U << idx);
            CCU_CHK_RET(Write(kernelArg->channels[idx],
                dst,
                src,
                peerSizes[idx],
                scatterEvent,
                eventMask));
            totalMask = static_cast<uint16_t>(totalMask | eventMask);
        }
        CCU_CHK_RET(EventWait(scatterEvent, totalMask));
        for (uint32_t idx = 0; idx < kernelArg->channelCount; ++idx) {
            CCU_CHK_RET(NotifyRecord(
                kernelArg->channels[idx], CKE_IDX_0, PHASE_SYNC_MASK));
        }
        return CCU_SUCCESS;
    }

    const int32_t rootIndex = FindRemoteRankIndex(kernelArg, kernelArg->root);
    if (rootIndex >= 0) {
        CCU_CHK_RET(NotifyWait(kernelArg->channels[static_cast<uint32_t>(rootIndex)],
            CKE_IDX_0,
            PHASE_SYNC_MASK));
    }
    return CCU_SUCCESS;
}

// AllGather中每个rank只发送自己的一个slice，因此仅加载myRank对应的
// offset/size，而不是在每个Die Kernel中重复加载rankSize组参数。
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
    CCU_CHK_RET(EventWait(gatherEvent, totalMask));
    return CCU_SUCCESS;
}

CcuResult ExchangeBufferInfo(BroadcastCcuKernelArg *kernelArg,
    AscendC::ccu::Variable &localBase,
    AscendC::ccu::Variable &localToken)
{
    using namespace AscendC::ccu;

    for (uint32_t idx = 0; idx < kernelArg->channelCount; ++idx) {
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
        CCU_CHK_RET(NotifyWait(kernelArg->channels[idx], CKE_IDX_0, PRE_SYNC_MASK));
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

CcuResult RunScatter(BroadcastCcuKernelArg *kernelArg,
    AscendC::ccu::Variable &localBase,
    AscendC::ccu::Variable &localToken,
    std::vector<AscendC::ccu::Variable> &remoteBase,
    std::vector<AscendC::ccu::Variable> &remoteToken,
    std::vector<AscendC::ccu::Variable> &sliceOffsets,
    std::vector<AscendC::ccu::Variable> &sliceSizes)
{
    using namespace AscendC::ccu;

    if (kernelArg->myRank == kernelArg->root) {
        Event scatterEvent;
        uint16_t totalMask = 0;

        for (uint32_t idx = 0; idx < kernelArg->channelCount; ++idx) {
            const uint32_t peerRank = kernelArg->remoteRanks[idx];

            LocalAddr src;
            src.addr = localBase;
            src.addr += sliceOffsets[peerRank];
            src.token = localToken;

            RemoteAddr dst;
            dst.addr = remoteBase[idx];
            dst.addr += sliceOffsets[peerRank];
            dst.token = remoteToken[idx];

            const uint16_t mask = static_cast<uint16_t>(1U << idx);
            CCU_CHK_RET(Write(kernelArg->channels[idx],
                dst,
                src,
                sliceSizes[peerRank],
                scatterEvent,
                mask));
            totalMask = static_cast<uint16_t>(totalMask | mask);
        }
        CCU_CHK_RET(EventWait(scatterEvent, totalMask));
    }

    return CCU_SUCCESS;
}

CcuResult RunAllGather(BroadcastCcuKernelArg *kernelArg,
    AscendC::ccu::Variable &localBase,
    AscendC::ccu::Variable &localToken,
    std::vector<AscendC::ccu::Variable> &remoteBase,
    std::vector<AscendC::ccu::Variable> &remoteToken,
    std::vector<AscendC::ccu::Variable> &sliceOffsets,
    std::vector<AscendC::ccu::Variable> &sliceSizes)
{
    using namespace AscendC::ccu;

    LocalAddr src;
    src.addr = localBase;
    src.addr += sliceOffsets[kernelArg->myRank];
    src.token = localToken;

    Event gatherEvent;
    uint16_t totalMask = 0;
    for (uint32_t idx = 0; idx < kernelArg->channelCount; ++idx) {
        RemoteAddr dst;
        dst.addr = remoteBase[idx];
        dst.addr += sliceOffsets[kernelArg->myRank];
        dst.token = remoteToken[idx];

        const uint16_t mask = static_cast<uint16_t>(1U << idx);
        CCU_CHK_RET(Write(kernelArg->channels[idx],
            dst,
            src,
            sliceSizes[kernelArg->myRank],
            gatherEvent,
            mask));
        totalMask = static_cast<uint16_t>(totalMask | mask);
    }
    CCU_CHK_RET(EventWait(gatherEvent, totalMask));
    return CCU_SUCCESS;
}


// 8+4专用Scatter：在分发每个peer自身slice的同时，将root slice预发给所有peer。
// 这样后续AllGather可彻底跳过root方向，删除11条非root->root冗余写，并把
// root->peer的必要写与Scatter阶段重叠。
CcuResult Publish8p4ScatterBufferToRoot(BroadcastCcuKernelArg *kernelArg,
    AscendC::ccu::Variable &localBase,
    AscendC::ccu::Variable &localToken)
{
    using namespace AscendC::ccu;

    if (kernelArg->myRank == kernelArg->root) {
        return CCU_SUCCESS;
    }

    const int32_t rootIndex = FindRemoteRankIndex(kernelArg, kernelArg->root);
    if (rootIndex < 0) {
        return CCU_SUCCESS;
    }

    const ChannelHandle rootChannel =
        kernelArg->channels[static_cast<uint32_t>(rootIndex)];
    CCU_CHK_RET(WriteVariableWithNotify(
        rootChannel, localBase, OUTPUT_XN_ID, CKE_IDX_0, OUTPUT_MASK));
    CCU_CHK_RET(WriteVariableWithNotify(
        rootChannel, localToken, TOKEN_XN_ID, CKE_IDX_0, TOKEN_MASK));
    return CCU_SUCCESS;
}

// 8+4专用Scatter TaskGraph：root按Channel“地址就绪即发送”，并把
// peer-own slice与root slice连续投递到同一Channel。peer-own slice完成后
// 立即放行AllGather；root slice全部写完后单向通知peer最终数据已就绪。
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
            CCU_CHK_RET(NotifyWait(
                kernelArg->channels[static_cast<uint32_t>(rootIndex)],
                CKE_IDX_0,
                PHASE_SYNC_MASK));
        }
        return CCU_SUCCESS;
    }

    Variable rootOffset;
    Variable rootSize;
    CCU_CHK_RET(LoadArg(rootOffset, P84_ARG_ROOT_OFFSET));
    CCU_CHK_RET(LoadArg(rootSize, P84_ARG_ROOT_SIZE));

    std::array<Variable, P84_MAX_CHANNELS> peerOffsets;
    std::array<Variable, P84_MAX_CHANNELS> peerSizes;
    const uint32_t peerSizeBase =
        P84_ARG_PEER_OFFSET_BASE + kernelArg->channelCount;
    for (uint32_t idx = 0; idx < kernelArg->channelCount; ++idx) {
        CCU_CHK_RET(LoadArg(
            peerOffsets[idx], P84_ARG_PEER_OFFSET_BASE + idx));
        CCU_CHK_RET(LoadArg(peerSizes[idx], peerSizeBase + idx));
    }

    LocalAddr rootSrc;
    rootSrc.addr = localBase;
    rootSrc.addr += rootOffset;
    rootSrc.token = localToken;

    Event peerScatterEvent;
    Event rootPrefetchEvent;
    uint16_t peerMaskAll = 0;
    uint16_t rootMaskAll = 0;

    for (uint32_t idx = 0; idx < kernelArg->channelCount; ++idx) {
        const ChannelHandle channel = kernelArg->channels[idx];

        // XN只在PRE_SYNC完成后读取；同时把等待与此前Channel的数据搬运重叠。
        CCU_CHK_RET(NotifyWait(channel, CKE_IDX_0, PRE_SYNC_MASK));

        LocalAddr peerSrc;
        peerSrc.addr = localBase;
        peerSrc.addr += peerOffsets[idx];
        peerSrc.token = localToken;

        RemoteAddr peerDst;
        peerDst.addr = remoteBase[idx];
        peerDst.addr += peerOffsets[idx];
        peerDst.token = remoteToken[idx];

        const uint16_t mask = static_cast<uint16_t>(1U << idx);
        CCU_CHK_RET(Write(channel,
            peerDst,
            peerSrc,
            peerSizes[idx],
            peerScatterEvent,
            mask));
        peerMaskAll = static_cast<uint16_t>(peerMaskAll | mask);

        // 同一Channel严格有序：root slice紧随peer-own slice排队。
        RemoteAddr rootDst;
        rootDst.addr = remoteBase[idx];
        rootDst.addr += rootOffset;
        rootDst.token = remoteToken[idx];
        CCU_CHK_RET(Write(channel,
            rootDst,
            rootSrc,
            rootSize,
            rootPrefetchEvent,
            mask));
        rootMaskAll = static_cast<uint16_t>(rootMaskAll | mask);
    }

    if (peerMaskAll != 0) {
        CCU_CHK_RET(EventWait(peerScatterEvent, peerMaskAll));
    }
    for (uint32_t idx = 0; idx < kernelArg->channelCount; ++idx) {
        CCU_CHK_RET(NotifyRecord(
            kernelArg->channels[idx], CKE_IDX_0, PHASE_SYNC_MASK));
    }

    if (rootMaskAll != 0) {
        CCU_CHK_RET(EventWait(rootPrefetchEvent, rootMaskAll));
    }
    // root不再启动空AllGather Kernel；Scatter结束时直接单向通知peer。
    for (uint32_t idx = 0; idx < kernelArg->channelCount; ++idx) {
        CCU_CHK_RET(NotifyRecord(
            kernelArg->channels[idx], CKE_IDX_0, FINAL_SYNC_MASK));
    }
    return CCU_SUCCESS;
}

// 非root AllGather TaskGraph：先向全部非root peer发布地址，再按Channel
// “远端地址就绪即Write”。最终屏障只在非root之间进行，同时等待root
// Scatter发来的单向root-slice完成通知。
CcuResult Run8p4AllGatherTaskGraph(BroadcastCcuKernelArg *kernelArg,
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

    // 所有rank先发布，避免对称peer逐Channel等待形成环。
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

        const ChannelHandle channel = kernelArg->channels[idx];
        CCU_CHK_RET(NotifyWait(channel, CKE_IDX_0, PRE_SYNC_MASK));

        RemoteAddr dst;
        dst.addr = remoteBase[idx];
        dst.addr += ownOffset;
        dst.token = remoteToken[idx];

        const uint16_t mask = static_cast<uint16_t>(1U << idx);
        CCU_CHK_RET(Write(channel,
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

    // 先向非root peer报告本rank发送完成。
    for (uint32_t idx = 0; idx < kernelArg->channelCount; ++idx) {
        if (kernelArg->remoteRanks[idx] != kernelArg->root) {
            CCU_CHK_RET(NotifyRecord(
                kernelArg->channels[idx], CKE_IDX_0, FINAL_SYNC_MASK));
        }
    }

    // root只单向报告root slice已完成，不等待非root。
    const int32_t rootIndex = FindRemoteRankIndex(kernelArg, kernelArg->root);
    if (rootIndex >= 0) {
        CCU_CHK_RET(NotifyWait(
            kernelArg->channels[static_cast<uint32_t>(rootIndex)],
            CKE_IDX_0,
            FINAL_SYNC_MASK));
    }

    for (uint32_t idx = 0; idx < kernelArg->channelCount; ++idx) {
        if (kernelArg->remoteRanks[idx] != kernelArg->root) {
            CCU_CHK_RET(NotifyWait(
                kernelArg->channels[idx], CKE_IDX_0, FINAL_SYNC_MASK));
        }
    }
    return CCU_SUCCESS;
}

} // namespace

namespace ops_hccl {

CcuResult CcuBroadcastDirectRootKernel(CcuKernelArg arg)
{
    using namespace AscendC::ccu;

    auto *kernelArg = static_cast<BroadcastDirectCcuKernelArg *>(arg);
    if (kernelArg == nullptr) {
        return CCU_E_PTR;
    }
    if (kernelArg->myRank != kernelArg->root ||
        kernelArg->channelCount == 0 ||
        kernelArg->channelCount >= MAX_RANK_SIZE ||
        kernelArg->directDataBytes == 0 ||
        kernelArg->directDataBytes > MAX_DATA_SIZE) {
        return CCU_E_INTERNAL;
    }

    Variable baseAddr;
    Variable memToken;
    CCU_CHK_RET(LoadArg(baseAddr, DIRECT_ARG_BASE_ADDR));
    CCU_CHK_RET(LoadArg(memToken, DIRECT_ARG_MEM_TOKEN));
    Variable dataSize;
    dataSize = kernelArg->directDataBytes;

    LocalAddr src;
    src.addr = baseAddr;
    src.token = memToken;

    Event writeEvent;
    uint16_t totalWriteMask = 0;
    for (uint32_t idx = 0; idx < kernelArg->channelCount; ++idx) {
        const ChannelHandle channel = kernelArg->channels[idx];
        // 每条Channel地址就绪后立即发起Write，不再等待全部peer统一就绪。
        CCU_CHK_RET(NotifyWait(channel, CKE_IDX_0, PRE_SYNC_MASK));

        RemoteAddr dst;
        dst.addr = GetResByChannel<Variable>(channel, OUTPUT_XN_ID);
        dst.token = GetResByChannel<Variable>(channel, TOKEN_XN_ID);

        const uint16_t mask = static_cast<uint16_t>(1U << idx);
        CCU_CHK_RET(Write(channel, dst, src, dataSize, writeEvent, mask));
        totalWriteMask = static_cast<uint16_t>(totalWriteMask | mask);
    }

    CCU_CHK_RET(EventWait(writeEvent, totalWriteMask));
    // 所有拓扑统一采用root->peer单向完成通知；下一轮PRE_SYNC承担回压。
    for (uint32_t idx = 0; idx < kernelArg->channelCount; ++idx) {
        CCU_CHK_RET(NotifyRecord(
            kernelArg->channels[idx], CKE_IDX_0, PHASE_SYNC_MASK));
    }
    return CCU_SUCCESS;
}

CcuResult CcuBroadcastDirectPeerKernel(CcuKernelArg arg)
{
    using namespace AscendC::ccu;

    auto *kernelArg = static_cast<BroadcastCcuKernelArg *>(arg);
    if (kernelArg == nullptr) {
        return CCU_E_PTR;
    }
    if (kernelArg->myRank == kernelArg->root || kernelArg->channelCount != 1) {
        return CCU_E_INTERNAL;
    }

    // peer不使用dataSize，只加载base/token两项TaskArg。
    Variable baseAddr;
    Variable memToken;
    CCU_CHK_RET(LoadArg(baseAddr, DIRECT_ARG_BASE_ADDR));
    CCU_CHK_RET(LoadArg(memToken, DIRECT_ARG_MEM_TOKEN));

    const ChannelHandle rootChannel = kernelArg->channels[0];
    CCU_CHK_RET(WriteVariableWithNotify(
        rootChannel, baseAddr, OUTPUT_XN_ID, CKE_IDX_0, OUTPUT_MASK));
    CCU_CHK_RET(WriteVariableWithNotify(
        rootChannel, memToken, TOKEN_XN_ID, CKE_IDX_0, TOKEN_MASK));
    CCU_CHK_RET(NotifyWait(rootChannel, CKE_IDX_0, PHASE_SYNC_MASK));
    return CCU_SUCCESS;
}

CcuResult CcuBroadcastScatterKernel(CcuKernelArg arg)
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
    std::vector<Variable> remoteBase;
    std::vector<Variable> remoteToken;

    CCU_CHK_RET(InitLargeBaseContext(
        kernelArg, localBase, localToken, remoteBase, remoteToken));
    // Scatter阶段只在root与目标peer之间交换地址并同步完成；
    // root仅加载当前Die组真正使用的切片参数。
    CCU_CHK_RET(ExchangeScatterBufferInfo(kernelArg, localBase, localToken));
    CCU_CHK_RET(RunScatterAndSignalLean(
        kernelArg, localBase, localToken, remoteBase, remoteToken));
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
    std::vector<Variable> remoteBase;
    std::vector<Variable> remoteToken;

    CCU_CHK_RET(InitLargeBaseContext(
        kernelArg, localBase, localToken, remoteBase, remoteToken));
    CCU_CHK_RET(ExchangeBufferInfo(kernelArg, localBase, localToken));
    CCU_CHK_RET(RunAllGatherLean(
        kernelArg, localBase, localToken, remoteBase, remoteToken));
    CCU_CHK_RET(PairwiseBarrier(kernelArg, PHASE_SYNC_MASK));
    return CCU_SUCCESS;
}

CcuResult CcuBroadcastScatterAllGatherKernel(CcuKernelArg arg)
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
    std::vector<Variable> remoteBase;
    std::vector<Variable> remoteToken;
    std::vector<Variable> sliceOffsets;
    std::vector<Variable> sliceSizes;

    CCU_CHK_RET(InitLargeKernelContext(kernelArg,
        localBase,
        localToken,
        remoteBase,
        remoteToken,
        sliceOffsets,
        sliceSizes));
    CCU_CHK_RET(ExchangeBufferInfo(kernelArg, localBase, localToken));

    CCU_CHK_RET(RunScatter(kernelArg,
        localBase,
        localToken,
        remoteBase,
        remoteToken,
        sliceOffsets,
        sliceSizes));
    CCU_CHK_RET(PairwiseBarrier(kernelArg, PHASE_SYNC_MASK));

    CCU_CHK_RET(RunAllGather(kernelArg,
        localBase,
        localToken,
        remoteBase,
        remoteToken,
        sliceOffsets,
        sliceSizes));
    CCU_CHK_RET(PairwiseBarrier(kernelArg, FINAL_SYNC_MASK));
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
    CCU_CHK_RET(Publish8p4ScatterBufferToRoot(
        kernelArg, localBase, localToken));
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
    CCU_CHK_RET(Run8p4AllGatherTaskGraph(
        kernelArg, localBase, localToken, remoteBase, remoteToken));
    return CCU_SUCCESS;
}

CcuResult CcuBroadcastChainPipelineKernel(CcuKernelArg arg)
{
    using namespace AscendC::ccu;

    auto *kernelArg = static_cast<BroadcastChainCcuKernelArg *>(arg);
    if (kernelArg == nullptr) {
        return CCU_E_PTR;
    }
    if (kernelArg->rankSize != 4 || kernelArg->channelCount == 0 ||
        kernelArg->channelCount > 2 || kernelArg->pipelineChunkCount == 0 ||
        kernelArg->pipelineChunkCount > BCAST_PIPELINE_CHUNK_NUM) {
        return CCU_E_INTERNAL;
    }

    const bool hasPrev = kernelArg->prevRank != INVALID_VALUE_RANKID;
    const bool hasNext = kernelArg->nextRank != INVALID_VALUE_RANKID;
    if ((hasPrev && kernelArg->prevChannelIndex >= kernelArg->channelCount) ||
        (hasNext && kernelArg->nextChannelIndex >= kernelArg->channelCount)) {
        return CCU_E_INTERNAL;
    }

    Variable localBase;
    Variable localToken;
    CCU_CHK_RET(LoadArg(localBase, CHAIN_ARG_BASE_ADDR));
    CCU_CHK_RET(LoadArg(localToken, CHAIN_ARG_MEM_TOKEN));

    std::array<Variable, CHAIN_MAX_CHANNELS> remoteBase;
    std::array<Variable, CHAIN_MAX_CHANNELS> remoteToken;
    for (uint32_t idx = 0; idx < kernelArg->channelCount; ++idx) {
        remoteBase[idx] = GetResByChannel<Variable>(kernelArg->channels[idx], OUTPUT_XN_ID);
        remoteToken[idx] = GetResByChannel<Variable>(kernelArg->channels[idx], TOKEN_XN_ID);
    }

    // 只沿数据反方向交换地址：每个非root把本地buffer信息交给上游，
    // 每个发送者只等待下游地址，避免无用的双向交换。
    CCU_CHK_RET(ExchangeChainBufferInfo(kernelArg, localBase, localToken));

    ChannelHandle prevChannel = 0;
    ChannelHandle nextChannel = 0;
    if (hasPrev) {
        prevChannel = kernelArg->channels[kernelArg->prevChannelIndex];
    }
    if (hasNext) {
        nextChannel = kernelArg->channels[kernelArg->nextChannelIndex];
    }

    Event forwardEvent;
    for (uint32_t chunk = 0; chunk < kernelArg->pipelineChunkCount; ++chunk) {
        const uint32_t slot = chunk % CHAIN_SLOT_COUNT;
        const uint16_t dataMask =
            static_cast<uint16_t>(1U << (CHAIN_DATA_BIT_BASE + slot));
        const uint16_t ackMask =
            static_cast<uint16_t>(1U << (CHAIN_ACK_BIT_BASE + slot));

        // 先消费上游数据通知并立即ACK，保证上游复用该slot前不会多打一。
        if (hasPrev) {
            CCU_CHK_RET(NotifyWait(prevChannel, CKE_IDX_0, dataMask));
            CCU_CHK_RET(NotifyRecord(prevChannel, CKE_IDX_0, ackMask));
        }

        if (hasNext) {
            // 四槽循环使用；发送同一slot的新块前，必须确认下游已消费旧通知。
            if (chunk >= CHAIN_SLOT_COUNT) {
                CCU_CHK_RET(NotifyWait(nextChannel, CKE_IDX_0, ackMask));
            }

            Variable chunkOffset;
            Variable chunkSize;
            chunkOffset = kernelArg->pipelineChunkOffsets[chunk];
            chunkSize = kernelArg->pipelineChunkSizes[chunk];

            LocalAddr src;
            src.addr = localBase;
            src.addr += chunkOffset;
            src.token = localToken;

            RemoteAddr dst;
            dst.addr = remoteBase[kernelArg->nextChannelIndex];
            dst.addr += chunkOffset;
            dst.token = remoteToken[kernelArg->nextChannelIndex];

            CCU_CHK_RET(Write(nextChannel,
                dst,
                src,
                chunkSize,
                forwardEvent,
                static_cast<uint16_t>(1U)));
            CCU_CHK_RET(EventWait(forwardEvent, static_cast<uint16_t>(1U)));
            CCU_CHK_RET(NotifyRecord(nextChannel, CKE_IDX_0, dataMask));
        }
    }

    if (hasNext) {
        // 消费最后仍在飞行的至多四个ACK，随后等待下游端到端完成回传。
        const uint32_t firstOutstanding =
            kernelArg->pipelineChunkCount > CHAIN_SLOT_COUNT
                ? kernelArg->pipelineChunkCount - CHAIN_SLOT_COUNT
                : 0;
        for (uint32_t chunk = firstOutstanding;
             chunk < kernelArg->pipelineChunkCount;
             ++chunk) {
            const uint32_t slot = chunk % CHAIN_SLOT_COUNT;
            const uint16_t ackMask =
                static_cast<uint16_t>(1U << (CHAIN_ACK_BIT_BASE + slot));
            CCU_CHK_RET(NotifyWait(nextChannel, CKE_IDX_0, ackMask));
        }
        CCU_CHK_RET(NotifyWait(nextChannel, CKE_IDX_0, CHAIN_FINAL_MASK));
    }

    // 最末rank完成后逐级向root回传最终完成信号，确保root不会提前结束。
    if (hasPrev) {
        CCU_CHK_RET(NotifyRecord(prevChannel, CKE_IDX_0, CHAIN_FINAL_MASK));
    }

    return CCU_SUCCESS;
}

} // namespace ops_hccl
