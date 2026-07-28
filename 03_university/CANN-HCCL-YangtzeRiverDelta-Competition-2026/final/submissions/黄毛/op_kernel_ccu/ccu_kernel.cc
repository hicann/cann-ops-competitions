#include <vector>

#include <hcomm/hcomm_primitives.h>

#include "ccu_kernel.h"
#include "log.h"

#define CCU_CHK_RET(call) \
    do { \
        CcuResult ccuResult = (call); \
        if (ccuResult != CCU_SUCCESS) { \
            return ccuResult; \
        } \
    } while (0)

namespace ops_hccl {
namespace ccu = ::AscendC::ccu;

constexpr uint32_t OUTPUT_XN_ID = 1;
constexpr uint32_t TOKEN_XN_ID = 2;
constexpr uint32_t POST_SYNC_ID = 3;
constexpr uint32_t CKE_INDEX = 0;

CcuResult CcuKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<BroadcastKernelArg *>(arg);
    if (kernelArg == nullptr || kernelArg->channelCount == 0 || kernelArg->channelCount >= MAX_RANK_SIZE) {
        return CCU_E_PARA;
    }

    std::vector<ccu::Variable> output(kernelArg->channelCount + 1);
    std::vector<ccu::Variable> token(kernelArg->channelCount + 1);
    const uint32_t localIndex = kernelArg->channelCount;
    for (uint32_t index = 0; index < kernelArg->channelCount; ++index) {
        output[index] = ccu::GetResByChannel<ccu::Variable>(kernelArg->channels[index], OUTPUT_XN_ID);
        token[index] = ccu::GetResByChannel<ccu::Variable>(kernelArg->channels[index], TOKEN_XN_ID);
    }

    ccu::Variable dataSize;
    ccu::Variable root;
    uint32_t argIndex = 0;
    CCU_CHK_RET(ccu::LoadArg(output[localIndex], argIndex++));
    CCU_CHK_RET(ccu::LoadArg(token[localIndex], argIndex++));
    CCU_CHK_RET(ccu::LoadArg(dataSize, argIndex++));
    CCU_CHK_RET(ccu::LoadArg(root, argIndex++));

    // Phase 2 immediately follows phase 1 for the same slice and uses the same
    // channel-shared address/token slots. Reuse the values exchanged by phase 1
    // instead of paying for a second control-plane round trip.
    if (kernelArg->phaseMask != 2) {
        for (uint32_t index = 0; index < kernelArg->channelCount; ++index) {
            CCU_CHK_RET(ccu::WriteVariableWithNotify(kernelArg->channels[index], output[localIndex],
                OUTPUT_XN_ID, CKE_INDEX, 1U << OUTPUT_XN_ID));
            CCU_CHK_RET(ccu::WriteVariableWithNotify(kernelArg->channels[index], token[localIndex],
                TOKEN_XN_ID, CKE_INDEX, 1U << TOKEN_XN_ID));
        }
        constexpr uint32_t addressReadyMask = (1U << OUTPUT_XN_ID) | (1U << TOKEN_XN_ID);
        for (uint32_t index = 0; index < kernelArg->channelCount; ++index) {
            CCU_CHK_RET(ccu::NotifyWait(kernelArg->channels[index], CKE_INDEX, addressReadyMask));
        }
    }

    ccu::LocalAddr source;
    source.addr = output[localIndex];
    source.token = token[localIndex];
    ccu::Event writeDone;

    if (kernelArg->phaseMask == 0) {
        CCU_IF(root == kernelArg->rankId)
        {
            uint16_t writeMask = 0;
            for (uint32_t index = 0; index < kernelArg->channelCount; ++index) {
                const uint16_t rankMask = static_cast<uint16_t>(1U << index);
                ccu::RemoteAddr destination;
                destination.addr = output[index];
                destination.token = token[index];
                CCU_CHK_RET(ccu::Write(kernelArg->channels[index], destination, source, dataSize, writeDone,
                    rankMask));
                writeMask = static_cast<uint16_t>(writeMask | rankMask);
            }
            CCU_CHK_RET(ccu::EventWait(writeDone, writeMask));
            for (uint32_t index = 0; index < kernelArg->channelCount; ++index) {
                CCU_CHK_RET(ccu::NotifyRecord(kernelArg->channels[index], CKE_INDEX, 1U << POST_SYNC_ID));
            }
        }
        for (uint32_t index = 0; index < kernelArg->channelCount; ++index) {
            CCU_IF(root == kernelArg->remoteRanks[index])
            {
                CCU_CHK_RET(ccu::NotifyWait(kernelArg->channels[index], CKE_INDEX, 1U << POST_SYNC_ID));
            }
        }
        return CCU_SUCCESS;
    }

    // Large-message path. Split the payload into rankSize contiguous pieces. The
    // root scatters one piece to each rank, then every rank fans its own piece
    // out to all peers. This moves about one payload per rank instead of one
    // complete payload per tree level.
    ccu::Variable chunkSize;
    CCU_CHK_RET(ccu::LoadArg(chunkSize, argIndex++));
    ccu::Variable lastChunkSize;
    CCU_CHK_RET(ccu::LoadArg(lastChunkSize, argIndex++));

    if (kernelArg->phaseMask == 1) {
        CCU_IF(root == kernelArg->rankId)
        {
            uint16_t writeMask = 0;
            for (uint32_t index = 0; index < kernelArg->channelCount; ++index) {
                ccu::Variable chunkOffset;
                chunkOffset = 0;
                for (uint32_t step = 0; step < kernelArg->remoteRanks[index]; ++step) {
                    chunkOffset += chunkSize;
                }
                ccu::Variable copySize;
                copySize = chunkSize;
                if (kernelArg->remoteRanks[index] + 1 == kernelArg->rankSize) {
                    copySize = lastChunkSize;
                }
                ccu::LocalAddr chunkSource;
                chunkSource.addr = output[localIndex];
                chunkSource.addr += chunkOffset;
                chunkSource.token = token[localIndex];
                ccu::RemoteAddr destination;
                destination.addr = output[index];
                destination.addr += chunkOffset;
                destination.token = token[index];
                const uint16_t rankMask = static_cast<uint16_t>(1U << index);
                CCU_CHK_RET(ccu::Write(kernelArg->channels[index], destination, chunkSource, copySize,
                    writeDone, rankMask));
                writeMask = static_cast<uint16_t>(writeMask | rankMask);
            }
            CCU_CHK_RET(ccu::EventWait(writeDone, writeMask));
            for (uint32_t index = 0; index < kernelArg->channelCount; ++index) {
                CCU_CHK_RET(ccu::NotifyRecord(kernelArg->channels[index], CKE_INDEX, 1U << POST_SYNC_ID));
            }
        }
        for (uint32_t index = 0; index < kernelArg->channelCount; ++index) {
            CCU_IF(root == kernelArg->remoteRanks[index])
            {
                CCU_CHK_RET(ccu::NotifyWait(kernelArg->channels[index], CKE_INDEX, 1U << POST_SYNC_ID));
            }
        }
        return CCU_SUCCESS;
    }

    ccu::Variable localOffset;
    localOffset = 0;
    for (uint32_t step = 0; step < kernelArg->rankId; ++step) {
        localOffset += chunkSize;
    }
    ccu::Variable localSize;
    localSize = chunkSize;
    if (kernelArg->rankId + 1 == kernelArg->rankSize) {
        localSize = lastChunkSize;
    }
    source.addr += localOffset;
    for (uint32_t index = 0; index < kernelArg->channelCount; ++index) {
        CCU_IF(root != kernelArg->remoteRanks[index])
        {
            ccu::RemoteAddr destination;
            destination.addr = output[index];
            destination.addr += localOffset;
            destination.token = token[index];
            const uint16_t rankMask = static_cast<uint16_t>(1U << index);
            CCU_CHK_RET(ccu::Write(kernelArg->channels[index], destination, source, localSize,
                writeDone, rankMask));
        }
    }
    for (uint32_t index = 0; index < kernelArg->channelCount; ++index) {
        CCU_IF(root != kernelArg->remoteRanks[index])
        {
            const uint16_t rankMask = static_cast<uint16_t>(1U << index);
            CCU_CHK_RET(ccu::EventWait(writeDone, rankMask));
            CCU_CHK_RET(ccu::NotifyRecord(kernelArg->channels[index], CKE_INDEX, 1U << POST_SYNC_ID));
        }
    }
    CCU_IF(root != kernelArg->rankId)
    {
        for (uint32_t index = 0; index < kernelArg->channelCount; ++index) {
            CCU_CHK_RET(ccu::NotifyWait(kernelArg->channels[index], CKE_INDEX, 1U << POST_SYNC_ID));
        }
    }
    return CCU_SUCCESS;
}
} // namespace ops_hccl
