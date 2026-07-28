#include "ccu_kernel.h"

namespace ops_hccl {
#ifndef CCU_CHK_RET
#define OPS_HCCL_LOCAL_CCU_CHK_RET
#define CCU_CHK_RET(call) \
    do { \
        CcuResult result = (call); \
        if (result != CCU_SUCCESS) { \
            return result; \
        } \
    } while (0)
#endif

namespace {
constexpr uint32_t REMOTE_BUFFER_ADDR_SLOT = 0;
constexpr uint32_t REMOTE_BUFFER_TOKEN_SLOT = 1;
constexpr uint32_t CHANNEL_NOTIFY_INDEX = 0;

constexpr uint16_t BUFFER_ADDR_READY_MASK = 1U << 0;
constexpr uint16_t BUFFER_TOKEN_READY_MASK = 1U << 1;
constexpr uint16_t TRANSFER_DONE_MASK = 1U << 2;
} // namespace

CcuResult CcuPairKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<AllReduceCcuKernelArg *>(arg);
    if (kernelArg == nullptr || kernelArg->rankSize <= 1 || kernelArg->rankSize > MAX_RANK_SIZE
        || kernelArg->rankId >= kernelArg->rankSize || kernelArg->peerRank >= kernelArg->rankSize
        || kernelArg->rankId == kernelArg->peerRank || kernelArg->channelCount != 1) {
        return CCU_E_PARA;
    }

    ccu::Variable outputAddr;
    ccu::Variable outputToken;
    ccu::Variable cclBufferAddr;
    ccu::Variable cclBufferToken;
    ccu::Variable tileBytes;

    const ChannelHandle channel = kernelArg->channels[0];

    ccu::Variable remoteBufferAddr;
    ccu::Variable remoteBufferToken;
    remoteBufferAddr = ccu::GetResByChannel<ccu::Variable>(channel, REMOTE_BUFFER_ADDR_SLOT);
    remoteBufferToken = ccu::GetResByChannel<ccu::Variable>(channel, REMOTE_BUFFER_TOKEN_SLOT);

    uint32_t argId = 0;
    CCU_CHK_RET(ccu::LoadArg(outputAddr, argId++));
    CCU_CHK_RET(ccu::LoadArg(outputToken, argId++));
    CCU_CHK_RET(ccu::LoadArg(cclBufferAddr, argId++));
    CCU_CHK_RET(ccu::LoadArg(cclBufferToken, argId++));
    CCU_CHK_RET(ccu::LoadArg(tileBytes, argId++));

    ccu::LocalAddr output;
    output.addr = outputAddr;
    output.token = outputToken;

    ccu::LocalAddr scratch;
    scratch.addr = cclBufferAddr;
    scratch.addr += tileBytes;
    scratch.token = cclBufferToken;

    ccu::Event event;
    constexpr uint16_t bufferReadyMask = BUFFER_ADDR_READY_MASK | BUFFER_TOKEN_READY_MASK;

    if (kernelArg->phase == PairKernelPhase::REDUCE_TO_ROOT) {
        if (kernelArg->isRoot) {
            CCU_CHK_RET(ccu::NotifyWait(channel, CHANNEL_NOTIFY_INDEX, bufferReadyMask));

            ccu::RemoteAddr remoteSource;
            remoteSource.addr = remoteBufferAddr;
            remoteSource.token = remoteBufferToken;

            CCU_CHK_RET(ccu::Read(channel, scratch, remoteSource, tileBytes, event));
            CCU_CHK_RET(ccu::EventWait(event));

            CCU_CHK_RET(
                ccu::LocalReduce(output, scratch, tileBytes, kernelArg->dataType, kernelArg->reduceType, event));
            CCU_CHK_RET(ccu::EventWait(event));

            CCU_CHK_RET(ccu::NotifyRecord(channel, CHANNEL_NOTIFY_INDEX, TRANSFER_DONE_MASK));
        } else {
            CCU_CHK_RET(ccu::WriteVariableWithNotify(
                channel, cclBufferAddr, REMOTE_BUFFER_ADDR_SLOT, CHANNEL_NOTIFY_INDEX, BUFFER_ADDR_READY_MASK));

            CCU_CHK_RET(ccu::WriteVariableWithNotify(
                channel, cclBufferToken, REMOTE_BUFFER_TOKEN_SLOT, CHANNEL_NOTIFY_INDEX, BUFFER_TOKEN_READY_MASK));

            CCU_CHK_RET(ccu::NotifyWait(channel, CHANNEL_NOTIFY_INDEX, TRANSFER_DONE_MASK));
        }
    } else if (kernelArg->phase == PairKernelPhase::BROADCAST_FROM_ROOT) {
        if (kernelArg->isRoot) {
            CCU_CHK_RET(ccu::WriteVariableWithNotify(
                channel, cclBufferAddr, REMOTE_BUFFER_ADDR_SLOT, CHANNEL_NOTIFY_INDEX, BUFFER_ADDR_READY_MASK));

            CCU_CHK_RET(ccu::WriteVariableWithNotify(
                channel, cclBufferToken, REMOTE_BUFFER_TOKEN_SLOT, CHANNEL_NOTIFY_INDEX, BUFFER_TOKEN_READY_MASK));

            CCU_CHK_RET(ccu::NotifyWait(channel, CHANNEL_NOTIFY_INDEX, TRANSFER_DONE_MASK));
        } else {
            CCU_CHK_RET(ccu::NotifyWait(channel, CHANNEL_NOTIFY_INDEX, bufferReadyMask));

            ccu::RemoteAddr remoteSource;
            remoteSource.addr = remoteBufferAddr;
            remoteSource.token = remoteBufferToken;

            CCU_CHK_RET(ccu::Read(channel, output, remoteSource, tileBytes, event));
            CCU_CHK_RET(ccu::EventWait(event));

            CCU_CHK_RET(ccu::NotifyRecord(channel, CHANNEL_NOTIFY_INDEX, TRANSFER_DONE_MASK));
        }
    } else {
        return CCU_E_PARA;
    }

    return CCU_SUCCESS;
}

} // namespace ops_hccl

#ifdef OPS_HCCL_LOCAL_CCU_CHK_RET
#undef CCU_CHK_RET
#undef OPS_HCCL_LOCAL_CCU_CHK_RET
#endif