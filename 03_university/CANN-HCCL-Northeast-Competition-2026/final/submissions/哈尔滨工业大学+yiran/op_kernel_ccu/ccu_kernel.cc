/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software: you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 */

#include <hcomm/hcomm_primitives.h>

#include "ccu_kernel.h"
#include "log.h"

namespace ops_hccl {
namespace ccu = ::AscendC::ccu;

// Large direct-only transfers: receivers advertise their output in parallel.
// Root writes each slice and acknowledges all writes before returning.
CcuResult CcuScatterDirectPush(CcuKernelArg arg)
{
    auto *k = static_cast<ScatterCcuKernelArg *>(arg);
    if (k == nullptr || k->channelCount == 0 || k->channelCount > MAX_RANK_SIZE) {
        return CcuResult::CCU_E_PARA;
    }
    if (k->isRoot == 0U) {
        ccu::Variable output;
        ccu::Variable token;
        CCU_CHK_RET(ccu::LoadArg(output, 0));
        CCU_CHK_RET(ccu::LoadArg(token, 1));
        CCU_CHK_RET(ccu::WriteVariableWithNotify(k->channels[0], output, XN_ID_OUTPUT, CKE_IDX_0, 1U));
        CCU_CHK_RET(ccu::WriteVariableWithNotify(k->channels[0], token, XN_ID_TOKEN, CKE_IDX_0, 2U));
        CCU_CHK_RET(ccu::NotifyWait(k->channels[0], CKE_IDX_0, 1U << DATA_SIGNAL_ID));
        return CcuResult::CCU_SUCCESS;
    }
    ccu::Variable token;
    ccu::Variable bytes;
    CCU_CHK_RET(ccu::LoadArg(token, 0));
    CCU_CHK_RET(ccu::LoadArg(bytes, 1));
    ccu::Event done;
    uint32_t argIndex = k->copyLocalSlice != 0U ? 4U : 2U;
    for (uint32_t i = 0; i < k->channelCount; ++i) {
        ccu::Variable input;
        CCU_CHK_RET(ccu::LoadArg(input, argIndex++));
        ccu::LocalAddr src;
        src.addr = input;
        src.token = token;
        CCU_CHK_RET(ccu::NotifyWait(k->channels[i], CKE_IDX_0, 3U));
        ccu::RemoteAddr dst;
        dst.addr = ccu::GetResByChannel<ccu::Variable>(k->channels[i], XN_ID_OUTPUT);
        dst.token = ccu::GetResByChannel<ccu::Variable>(k->channels[i], XN_ID_TOKEN);
        const uint32_t bit = 1U << i;
        CCU_CHK_RET(ccu::Write(k->channels[i], dst, src, bytes, done, bit));
    }
    // Start network traffic before submitting the independent local slice.
    if (k->copyLocalSlice != 0U) {
        ccu::Variable output;
        ccu::Variable input;
        CCU_CHK_RET(ccu::LoadArg(output, 2));
        CCU_CHK_RET(ccu::LoadArg(input, 3));
        ccu::LocalAddr src;
        src.addr = input;
        src.token = token;
        ccu::LocalAddr dst;
        dst.addr = output;
        dst.token = token;
        const uint32_t localBit = 1U << k->channelCount;
        CCU_CHK_RET(ccu::LocalCopy(dst, src, bytes, done, localBit));
    }
    if (k->legacySchedule != 0U) {
        const uint32_t mask = (1U << (k->channelCount + k->copyLocalSlice)) - 1U;
        CCU_CHK_RET(ccu::EventWait(done, mask));
    }
    // v18: record DATA at submission time. The writes drain past kernel
    // return; the op-end stream sync flushes the channel DMA before any
    // consumer reads the buffer (platform-validated by the small-packet
    // in-flight reads and the peer's in-flight Write schedule).
    for (uint32_t i = 0; i < k->channelCount; ++i) {
        CCU_CHK_RET(ccu::NotifyRecord(k->channels[i], CKE_IDX_0, 1U << DATA_SIGNAL_ID));
    }
    if (k->legacySchedule == 0U && k->copyLocalSlice != 0U) {
        CCU_CHK_RET(ccu::EventWait(done, 1U << k->channelCount));
    }
    return CcuResult::CCU_SUCCESS;
}

// Coarse Scatter uses a mixed protocol on the same one-channel-per-peer graph:
// root-local relays pull their gather payload; remote ranks publish an output
// capability and receive the direct prefix via root Write.
CcuResult CcuScatterHybridRoot(CcuKernelArg arg)
{
    auto *k = static_cast<ScatterCcuKernelArg *>(arg);
    if (k == nullptr || k->isRoot == 0U || k->channelCount == 0U || k->channelCount >= MAX_RANK_SIZE) {
        return CcuResult::CCU_E_PARA;
    }
    ccu::Variable input;
    ccu::Variable token;
    ccu::Variable sliceBytes;
    CCU_CHK_RET(ccu::LoadArg(input, 0));
    CCU_CHK_RET(ccu::LoadArg(token, 1));
    if (k->legacySchedule != 0U) {
        CCU_CHK_RET(ccu::LoadArg(sliceBytes, 2));
    }
    // Publish pull sources first, including when a topology places both kinds
    // of channels on this die. Never wait for a remote output before doing so.
    for (uint32_t i = 0; i < k->channelCount; ++i) {
        if ((k->pushMask & (1U << i)) == 0U) {
            CCU_CHK_RET(ccu::WriteVariableWithNotify(k->channels[i], input, XN_ID_OUTPUT, CKE_IDX_0, 1U));
            CCU_CHK_RET(ccu::WriteVariableWithNotify(k->channels[i], token, XN_ID_TOKEN, CKE_IDX_0, 2U));
        }
    }
    ccu::Event done;
    uint32_t nextArg = k->legacySchedule != 0U ? 3U : 2U;
    if (k->legacySchedule == 0U && (k->pushMask != 0U || k->copyLocalSlice != 0U)) {
        CCU_CHK_RET(ccu::LoadArg(sliceBytes, nextArg++));
    }
    if (k->pushMask != 0U && k->legacySchedule == 0U) {
        ccu::Variable directBytes;
        ccu::Variable firstSource;
        ccu::LocalAddr source;
        CCU_CHK_RET(ccu::LoadArg(directBytes, nextArg++));
        CCU_CHK_RET(ccu::LoadArg(firstSource, nextArg++));
        source.addr = firstSource;
        source.token = token;
        uint32_t previousRank = 0;
        bool first = true;
        for (uint32_t i = 0; i < k->channelCount; ++i) {
            if ((k->pushMask & (1U << i)) == 0U) { continue; }
            if (!first) {
                for (uint32_t rank = previousRank; rank < k->peerRanks[i]; ++rank) {
                    source.addr += sliceBytes;
                }
            }
            previousRank = k->peerRanks[i];
            first = false;
            CCU_CHK_RET(ccu::NotifyWait(k->channels[i], CKE_IDX_0, 3U));
            ccu::RemoteAddr target;
            target.addr = ccu::GetResByChannel<ccu::Variable>(k->channels[i], XN_ID_OUTPUT);
            target.token = ccu::GetResByChannel<ccu::Variable>(k->channels[i], XN_ID_TOKEN);
            CCU_CHK_RET(ccu::Write(k->channels[i], target, source, directBytes, done, 1U << i));
        }
    } else if (k->pushMask != 0U) {
        ccu::Variable directBytes;
        ccu::Variable cursor;
        CCU_CHK_RET(ccu::LoadArg(directBytes, nextArg++));
        CCU_CHK_RET(ccu::LoadArg(cursor, nextArg++));
        uint32_t previousRank = 0;
        bool first = true;
        for (uint32_t i = 0; i < k->channelCount; ++i) {
            if ((k->pushMask & (1U << i)) == 0U) {
                continue;
            }
            if (!first) {
                for (uint32_t rank = previousRank; rank < k->peerRanks[i]; ++rank) {
                    cursor += sliceBytes;
                }
            }
            previousRank = k->peerRanks[i];
            first = false;
            ccu::LocalAddr source;
            source.addr = cursor;
            source.token = token;
            CCU_CHK_RET(ccu::NotifyWait(k->channels[i], CKE_IDX_0, 3U));
            ccu::RemoteAddr target;
            target.addr = ccu::GetResByChannel<ccu::Variable>(k->channels[i], XN_ID_OUTPUT);
            target.token = ccu::GetResByChannel<ccu::Variable>(k->channels[i], XN_ID_TOKEN);
            CCU_CHK_RET(ccu::Write(k->channels[i], target, source, directBytes, done, 1U << i));
        }
    }
    if (k->copyLocalSlice != 0U) {
        ccu::Variable output;
        ccu::Variable ownSource;
        CCU_CHK_RET(ccu::LoadArg(output, nextArg++));
        CCU_CHK_RET(ccu::LoadArg(ownSource, nextArg++));
        ccu::LocalAddr source;
        source.addr = ownSource;
        source.token = token;
        ccu::LocalAddr target;
        target.addr = output;
        target.token = token;
        const uint32_t localBit = 1U << k->channelCount;
        CCU_CHK_RET(ccu::LocalCopy(target, source, sliceBytes, done, localBit));
    }
    if (k->legacySchedule != 0U) {
        const uint32_t mask = k->pushMask | (k->copyLocalSlice != 0U ? (1U << k->channelCount) : 0U);
        if (mask != 0U) {
            CCU_CHK_RET(ccu::EventWait(done, mask));
        }
    }
    // Write completion protects the input lifetime; receiver completion is
    // ordered by this signal. Pull relays still owe their original read ACK.
    for (uint32_t i = 0; i < k->channelCount; ++i) {
        if ((k->pushMask & (1U << i)) != 0U) {
            // v18: asymmetric-direct roots (relay disabled, persistent == 0)
            // record DATA at Write submission; the writes drain past kernel
            // return and the op-end stream sync flushes the channel DMA
            // before any consumer reads the buffer. Coarse roots
            // (persistent == 1) keep the per-channel completion wait.
            if (k->legacySchedule == 0U && k->persistent != 0U) {
                CCU_CHK_RET(ccu::EventWait(done, 1U << i));
            }
            CCU_CHK_RET(ccu::NotifyRecord(k->channels[i], CKE_IDX_0, 1U << DATA_SIGNAL_ID));
        }
    }
    if (k->legacySchedule == 0U && k->copyLocalSlice != 0U) {
        CCU_CHK_RET(ccu::EventWait(done, 1U << k->channelCount));
    }
    for (uint32_t i = 0; i < k->channelCount; ++i) {
        if ((k->pushMask & (1U << i)) == 0U) {
            CCU_CHK_RET(ccu::NotifyWait(k->channels[i], CKE_IDX_0, 1U << DATA_SIGNAL_ID));
        }
    }
    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuScatterHybridReceiver(CcuKernelArg arg)
{
    auto *k = static_cast<ScatterCcuKernelArg *>(arg);
    if (k == nullptr || k->isRoot != 0U || k->channelCount != 1U) {
        return CcuResult::CCU_E_PARA;
    }
    ccu::Variable output;
    ccu::Variable token;
    CCU_CHK_RET(ccu::LoadArg(output, 0));
    CCU_CHK_RET(ccu::LoadArg(token, 1));
    const auto root = k->channels[0];
    CCU_CHK_RET(ccu::WriteVariableWithNotify(root, output, XN_ID_OUTPUT, CKE_IDX_0, 1U));
    CCU_CHK_RET(ccu::WriteVariableWithNotify(root, token, XN_ID_TOKEN, CKE_IDX_0, 2U));
    if (k->receiveRelayCount != 0U) {
        ccu::Variable tailBytes;
        ccu::Variable sourceOffset;
        ccu::Variable cursor;
        CCU_CHK_RET(ccu::LoadArg(tailBytes, 2));
        CCU_CHK_RET(ccu::LoadArg(sourceOffset, 3));
        CCU_CHK_RET(ccu::LoadArg(cursor, 4));
        for (uint32_t index = 0; index < k->receiveRelayIndices[0]; ++index) {
            cursor += tailBytes;
        }
        ccu::Event done;
        for (uint32_t i = 0; i < k->receiveRelayCount; ++i) {
            const auto channel = k->receiveRelayChannels[i];
            CCU_CHK_RET(ccu::NotifyWait(channel, CKE_IDX_0, 3U));
            ccu::RemoteAddr source;
            source.addr = ccu::GetResByChannel<ccu::Variable>(channel, XN_ID_OUTPUT);
            source.addr += sourceOffset;
            source.token = ccu::GetResByChannel<ccu::Variable>(channel, XN_ID_TOKEN);
            ccu::LocalAddr target;
            target.addr = cursor;
            target.token = token;
            CCU_CHK_RET(ccu::Read(channel, target, source, tailBytes, done, 1U << i));
            if (i + 1U < k->receiveRelayCount) {
                for (uint32_t index = k->receiveRelayIndices[i]; index < k->receiveRelayIndices[i + 1U]; ++index) {
                    cursor += tailBytes;
                }
            }
        }
        CCU_CHK_RET(ccu::EventWait(done, (1U << k->receiveRelayCount) - 1U));
        for (uint32_t i = 0; i < k->receiveRelayCount; ++i) {
            CCU_CHK_RET(ccu::NotifyRecord(k->receiveRelayChannels[i], CKE_IDX_0, 1U << DATA_SIGNAL_ID));
        }
    }
    CCU_CHK_RET(ccu::NotifyWait(root, CKE_IDX_0, 1U << DATA_SIGNAL_ID));
    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuScatterCoarseRelay(CcuKernelArg arg)
{
    auto *k = static_cast<ScatterCcuKernelArg *>(arg);
    if (k == nullptr || k->channelCount != 1U || k->gatherCount == 0U || k->gatherCount >= MAX_RANK_SIZE) {
        return CcuResult::CCU_E_PARA;
    }
    ccu::Variable stage;
    ccu::Variable output;
    ccu::Variable token;
    ccu::Variable scratch;
    ccu::Variable sliceBytes;
    ccu::Variable tailBytes;
    ccu::Variable ownOffset;
    CCU_CHK_RET(ccu::LoadArg(stage, 0));
    ccu::Variable offloadOffset;
    const auto root = k->channels[0];
    auto input = ccu::GetResByChannel<ccu::Variable>(root, XN_ID_OUTPUT);
    auto inputToken = ccu::GetResByChannel<ccu::Variable>(root, XN_ID_TOKEN);
    ccu::Event done;
    std::vector<ccu::RemoteAddr> sources(k->gatherCount);
    std::vector<ccu::LocalAddr> targets(k->gatherCount);
    ccu::RemoteAddr ownSource;
    ccu::LocalAddr ownTarget;
    ccu::Variable gatherCursor;
    const uint32_t ownMask = 1U << k->gatherCount;
    CCU_IF(stage == uint64_t{0}) {
        if (k->legacySchedule != 0U) {
            CCU_CHK_RET(ccu::LoadArg(output, 1));
        }
        CCU_CHK_RET(ccu::LoadArg(token, 2));
        CCU_CHK_RET(ccu::LoadArg(scratch, 3));
        CCU_CHK_RET(ccu::LoadArg(sliceBytes, 4));
        CCU_CHK_RET(ccu::LoadArg(tailBytes, 5));
        if (k->legacySchedule != 0U) {
            CCU_CHK_RET(ccu::LoadArg(ownOffset, 6));
        }
        CCU_CHK_RET(ccu::LoadArg(offloadOffset, 7));
        CCU_CHK_RET(ccu::NotifyWait(root, CKE_IDX_0, 3U));
        gatherCursor = input;
        gatherCursor += offloadOffset;
        for (uint32_t i = 0; i < k->gatherCount; ++i) {
            sources[i].addr = gatherCursor;
            sources[i].token = inputToken;
            targets[i].addr = scratch;
            targets[i].token = token;
            CCU_CHK_RET(ccu::Read(root, targets[i], sources[i], tailBytes, done, 1U << i));
            scratch += tailBytes;
            if (i + 1U < k->gatherCount) {
                for (uint32_t rank = k->gatherRanks[i]; rank < k->gatherRanks[i + 1U]; ++rank) {
                    gatherCursor += sliceBytes;
                }
            }
        }
        // Keep the Mesh queue busy immediately after the outbound payload.
        // Stage 1 consumes this same kernel's own-copy event before root may
        // release its input. Only the outbound events gate forwarding.
        if (k->legacySchedule == 0U) {
            CCU_CHK_RET(ccu::LoadArg(output, 1));
            CCU_CHK_RET(ccu::LoadArg(ownOffset, 6));
        }
        ownSource.addr = input;
        ownSource.addr += ownOffset;
        ownSource.token = inputToken;
        ownTarget.addr = output;
        ownTarget.token = token;
        CCU_CHK_RET(ccu::Read(root, ownTarget, ownSource, sliceBytes, done, ownMask));
        CCU_CHK_RET(ccu::EventWait(done, (1U << k->gatherCount) - 1U));
    }
    CCU_IF(stage == uint64_t{1}) {
        // Root's published input remains live until this read completes.
        CCU_CHK_RET(ccu::EventWait(done, ownMask));
        CCU_CHK_RET(ccu::NotifyRecord(root, CKE_IDX_0, 1U << DATA_SIGNAL_ID));
    }
    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuScatterResidentReceiver(CcuKernelArg arg)
{
    auto *k = static_cast<ScatterCcuKernelArg *>(arg);
    if (k == nullptr || k->channelCount != 1U || k->receiveRelayCount == 0U
        || k->receiveRelayCount >= MAX_RANK_SIZE) {
        return CcuResult::CCU_E_PARA;
    }
    ccu::Variable output;
    ccu::Variable token;
    ccu::Variable rankOffset;
    ccu::Variable stride;
    ccu::Variable directBytes;
    ccu::Variable tailBytes;
    ccu::Variable repeat;
    ccu::Variable lastDirect;
    ccu::Variable lastTail;
    CCU_CHK_RET(ccu::LoadArg(output, 0));
    CCU_CHK_RET(ccu::LoadArg(token, 1));
    CCU_CHK_RET(ccu::LoadArg(rankOffset, 2));
    CCU_CHK_RET(ccu::LoadArg(stride, 3));
    CCU_CHK_RET(ccu::LoadArg(directBytes, 4));
    CCU_CHK_RET(ccu::LoadArg(tailBytes, 5));
    CCU_CHK_RET(ccu::LoadArg(repeat, 6));
    CCU_CHK_RET(ccu::LoadArg(lastDirect, 7));
    CCU_CHK_RET(ccu::LoadArg(lastTail, 8));
    const auto root = k->channels[0];
    auto input = ccu::GetResByChannel<ccu::Variable>(root, XN_ID_OUTPUT);
    auto inputToken = ccu::GetResByChannel<ccu::Variable>(root, XN_ID_TOKEN);
    std::vector<ccu::Variable> relayInput(k->receiveRelayCount);
    std::vector<ccu::Variable> relayToken(k->receiveRelayCount);
    for (uint32_t i = 0; i < k->receiveRelayCount; ++i) {
        relayInput[i] = ccu::GetResByChannel<ccu::Variable>(k->receiveRelayChannels[i], XN_ID_OUTPUT);
        relayToken[i] = ccu::GetResByChannel<ccu::Variable>(k->receiveRelayChannels[i], XN_ID_TOKEN);
    }
    // Reuse these static resources across every loop iteration and the tail.
    ccu::RemoteAddr rootSource;
    ccu::LocalAddr rootTarget;
    std::vector<ccu::RemoteAddr> relaySources(k->receiveRelayCount);
    std::vector<ccu::LocalAddr> relayTargets(k->receiveRelayCount);
    ccu::Variable sourceOffset;
    ccu::Variable targetCursor;
    ccu::Variable one;
    one = uint64_t{1};
    ccu::Event done;
    CCU_CHK_RET(ccu::NotifyWait(root, CKE_IDX_0, 3U));
    rootSource.addr = input;
    rootSource.addr += rankOffset;
    rootSource.token = inputToken;
    rootTarget.token = token;
    const auto receiveTile = [&]() -> CcuResult {
        rootTarget.addr = output;
        CCU_CHK_RET(ccu::Read(root, rootTarget, rootSource, directBytes, done, 1U));
        sourceOffset = uint64_t{0};
        for (uint32_t i = 0; i < k->remoteIndex; ++i) {
            sourceOffset += tailBytes;
        }
        targetCursor = output;
        targetCursor += directBytes;
        for (uint32_t i = 0; i < k->receiveRelayCount; ++i) {
            const auto channel = k->receiveRelayChannels[i];
            CCU_CHK_RET(ccu::NotifyWait(channel, CKE_IDX_0, 3U));
            relaySources[i].addr = relayInput[i];
            relaySources[i].addr += sourceOffset;
            relaySources[i].token = relayToken[i];
            relayTargets[i].addr = targetCursor;
            relayTargets[i].token = token;
            CCU_CHK_RET(ccu::Read(channel, relayTargets[i], relaySources[i], tailBytes, done, 1U << (i + 1U)));
            targetCursor += tailBytes;
        }
        CCU_CHK_RET(ccu::EventWait(done, (1U << (k->receiveRelayCount + 1U)) - 1U));
        for (uint32_t i = 0; i < k->receiveRelayCount; ++i) {
            CCU_CHK_RET(ccu::NotifyRecord(k->receiveRelayChannels[i], CKE_IDX_0, 1U << DATA_SIGNAL_ID));
        }
        return CcuResult::CCU_SUCCESS;
    };
    // Host supplies -tileCount: run tileCount-1 uniform tiles, then the exact
    // final tile. No count/address/token is specialized into a cached kernel.
    CCU_WHILE(repeat != UINT64_MAX) {
        CCU_CHK_RET(receiveTile());
        output += stride;
        rootSource.addr += stride;
        repeat += one;
    }
    directBytes = lastDirect;
    tailBytes = lastTail;
    CCU_CHK_RET(receiveTile());
    CCU_CHK_RET(ccu::NotifyRecord(root, CKE_IDX_0, 1U << DATA_SIGNAL_ID));
    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuScatterKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<ScatterCcuKernelArg *>(arg);
    if (kernelArg == nullptr || kernelArg->channelCount == 0 || kernelArg->channelCount > MAX_RANK_SIZE) {
        return CcuResult::CCU_E_PARA;
    }

    if (kernelArg->isRoot != 0U) {
        ccu::Variable inputChunk;
        ccu::Variable inputToken;
        ccu::Variable chunkBytes;
        CCU_CHK_RET(ccu::LoadArg(inputChunk, 0));
        CCU_CHK_RET(ccu::LoadArg(inputToken, kernelArg->copyLocalSlice != 0U ? 2 : 1));
        if (kernelArg->legacySchedule != 0U && kernelArg->copyLocalSlice != 0U) {
            CCU_CHK_RET(ccu::LoadArg(chunkBytes, 3));
        }
        for (uint32_t channelIndex = 0; channelIndex < kernelArg->channelCount; ++channelIndex) {
            CCU_CHK_RET(ccu::WriteVariableWithNotify(
                kernelArg->channels[channelIndex], inputChunk, XN_ID_OUTPUT, CKE_IDX_0, 1U << XN_ID_OUTPUT));
            CCU_CHK_RET(ccu::WriteVariableWithNotify(
                kernelArg->channels[channelIndex], inputToken, XN_ID_TOKEN, CKE_IDX_0, 1U << XN_ID_TOKEN));
        }

        if (kernelArg->copyLocalSlice != 0U) {
            if (kernelArg->legacySchedule == 0U) {
                CCU_CHK_RET(ccu::LoadArg(chunkBytes, 3));
            }
            ccu::Variable outputChunk;
            CCU_CHK_RET(ccu::LoadArg(outputChunk, 1));
            ccu::Variable sourceCursor;
            CCU_CHK_RET(ccu::LoadArg(sourceCursor, 4));
            ccu::LocalAddr source;
            source.addr = sourceCursor;
            source.token = inputToken;
            ccu::LocalAddr destination;
            destination.addr = outputChunk;
            destination.token = inputToken;
            ccu::Event localEvent;
            CCU_CHK_RET(ccu::LocalCopy(destination, source, chunkBytes, localEvent, 1U));
            CCU_CHK_RET(ccu::EventWait(localEvent, 1U));
        }

        // v16: small pulls skip the completion receipt round entirely. The
        // framework's per-round rendezvous protects the root input buffer, so
        // waiting here would only serialise the root behind its slowest reader.
        if (kernelArg->noRootAck == 0U) {
            for (uint32_t channelIndex = 0; channelIndex < kernelArg->channelCount; ++channelIndex) {
                CCU_CHK_RET(ccu::NotifyWait(kernelArg->channels[channelIndex], CKE_IDX_0, 1U << DATA_SIGNAL_ID));
            }
        }
        return CcuResult::CCU_SUCCESS;
    }

    ccu::Variable outputChunk;
    ccu::Variable outputToken;
    ccu::Variable chunkBytes;
    ccu::Variable rankOffset;
    CCU_CHK_RET(ccu::LoadArg(outputChunk, 0));
    CCU_CHK_RET(ccu::LoadArg(outputToken, 1));
    CCU_CHK_RET(ccu::LoadArg(chunkBytes, 2));
    CCU_CHK_RET(ccu::LoadArg(rankOffset, 3));
    const ChannelHandle rootChannel = kernelArg->channels[0];
    ccu::Variable remoteInput = ccu::GetResByChannel<ccu::Variable>(rootChannel, XN_ID_OUTPUT);
    ccu::Variable remoteToken = ccu::GetResByChannel<ccu::Variable>(rootChannel, XN_ID_TOKEN);
    constexpr uint16_t inputReadyMask = static_cast<uint16_t>((1U << XN_ID_OUTPUT) | (1U << XN_ID_TOKEN));
    ccu::Variable firstTile;
    ccu::Variable lastTile;
    if (kernelArg->persistent != 0U) {
        const uint32_t flagIndex = kernelArg->gatherCount != 0U ? 6U + kernelArg->gatherCount : 4U;
        CCU_CHK_RET(ccu::LoadArg(firstTile, flagIndex));
        CCU_CHK_RET(ccu::LoadArg(lastTile, flagIndex + 1U));
        // Root publishes a stable capability once for the entire operation.
        // Subsequent tiles use that channel resource without another exchange.
        CCU_IF(firstTile == uint64_t{1}) {
            CCU_CHK_RET(ccu::NotifyWait(rootChannel, CKE_IDX_0, inputReadyMask));
        }
    } else {
        CCU_CHK_RET(ccu::NotifyWait(rootChannel, CKE_IDX_0, inputReadyMask));
    }

    ccu::Variable sourceCursor;
    sourceCursor = remoteInput;
    sourceCursor += rankOffset;
    ccu::RemoteAddr source;
    source.addr = sourceCursor;
    source.token = remoteToken;
    ccu::LocalAddr destination;
    destination.addr = outputChunk;
    destination.token = outputToken;
    ccu::Event transferEvent;
    CCU_CHK_RET(ccu::Read(rootChannel, destination, source, chunkBytes, transferEvent, 1U));
    uint32_t completionMask = 1U;
    if (kernelArg->gatherCount != 0U) {
        ccu::Variable scratch;
        ccu::Variable tailBytes;
        CCU_CHK_RET(ccu::LoadArg(scratch, 4));
        CCU_CHK_RET(ccu::LoadArg(tailBytes, 5));
        for (uint32_t i = 0; i < kernelArg->gatherCount; ++i) {
            ccu::Variable offset;
            CCU_CHK_RET(ccu::LoadArg(offset, 6 + i));
            ccu::Variable cursor;
            cursor = remoteInput;
            cursor += offset;
            ccu::RemoteAddr gatherSource;
            gatherSource.addr = cursor;
            gatherSource.token = remoteToken;
            ccu::LocalAddr gatherDestination;
            gatherDestination.addr = scratch;
            gatherDestination.token = outputToken;
            const uint32_t bit = 1U << (i + 1);
            CCU_CHK_RET(ccu::Read(rootChannel, gatherDestination, gatherSource, tailBytes, transferEvent, bit));
            scratch += tailBytes;
            completionMask |= bit;
        }
    }
    // Issue the root read before waiting for relay readiness. The disjoint
    // regions can transfer concurrently even when their channels share a die.
    if (kernelArg->receiveRelayCount != 0U) {
        const uint32_t firstArg = kernelArg->persistent != 0U ? 6U : 4U;
        ccu::Variable tailBytes;
        ccu::Variable sourceOffset;
        CCU_CHK_RET(ccu::LoadArg(tailBytes, firstArg));
        CCU_CHK_RET(ccu::LoadArg(sourceOffset, firstArg + 1U));
        ccu::Variable targetCursor;
        CCU_CHK_RET(ccu::LoadArg(targetCursor, firstArg + 2U));
        for (uint32_t index = 0; index < kernelArg->receiveRelayIndices[0]; ++index) {
            targetCursor += tailBytes;
        }
        for (uint32_t i = 0; i < kernelArg->receiveRelayCount; ++i) {
            const auto channel = kernelArg->receiveRelayChannels[i];
            auto address = ccu::GetResByChannel<ccu::Variable>(channel, XN_ID_OUTPUT);
            auto token = ccu::GetResByChannel<ccu::Variable>(channel, XN_ID_TOKEN);
            CCU_CHK_RET(ccu::NotifyWait(channel, CKE_IDX_0, inputReadyMask));
            ccu::Variable cursor;
            cursor = address;
            cursor += sourceOffset;
            ccu::RemoteAddr relaySource;
            relaySource.addr = cursor;
            relaySource.token = token;
            ccu::LocalAddr target;
            target.addr = targetCursor;
            target.token = outputToken;
            const uint32_t bit = 1U << (i + 1U);
            CCU_CHK_RET(ccu::Read(channel, target, relaySource, tailBytes, transferEvent, bit));
            completionMask |= bit;
            if (i + 1U < kernelArg->receiveRelayCount) {
                for (uint32_t index = kernelArg->receiveRelayIndices[i];
                    index < kernelArg->receiveRelayIndices[i + 1U]; ++index) {
                    targetCursor += tailBytes;
                }
            }
        }
    }
    if (kernelArg->receiveRelayCount != 0U) {
        // Relay buffers can be released as soon as their disjoint tails arrive.
        // Hide these acknowledgements behind the longer direct root transfer.
        CCU_CHK_RET(ccu::EventWait(transferEvent, completionMask & ~1U));
        for (uint32_t i = 0; i < kernelArg->receiveRelayCount; ++i) {
            CCU_CHK_RET(ccu::NotifyRecord(kernelArg->receiveRelayChannels[i], CKE_IDX_0, 1U << DATA_SIGNAL_ID));
        }
        CCU_CHK_RET(ccu::EventWait(transferEvent, 1U));
    } else if (kernelArg->noRootAck == 0U) {
        // v17: small pulls leave the read in flight past kernel return; the
        // op-end stream sync flushes the channel DMA before the buffer is read.
        CCU_CHK_RET(ccu::EventWait(transferEvent, completionMask));
    }
    if (kernelArg->persistent != 0U) {
        CCU_IF(lastTile == uint64_t{1}) {
            CCU_CHK_RET(ccu::NotifyRecord(rootChannel, CKE_IDX_0, 1U << DATA_SIGNAL_ID));
        }
    } else if (kernelArg->noRootAck == 0U) {
        // v16: paired with the root-side skip; deleting one side alone would
        // leave an unmatched Record in the channel notify FIFO.
        CCU_CHK_RET(ccu::NotifyRecord(rootChannel, CKE_IDX_0, 1U << DATA_SIGNAL_ID));
    }
    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuScatterRelayRead(CcuKernelArg arg)
{
    auto *k = static_cast<ScatterCcuKernelArg *>(arg);
    if (k == nullptr || k->channelCount == 0 || k->channelCount > MAX_RANK_SIZE) {
        return CcuResult::CCU_E_PARA;
    }
    ccu::Variable token;
    ccu::Variable bytes;
    ccu::Variable sourceOffset;
    CCU_CHK_RET(ccu::LoadArg(token, 0));
    CCU_CHK_RET(ccu::LoadArg(bytes, 1));
    CCU_CHK_RET(ccu::LoadArg(sourceOffset, 2));
    ccu::Event done;
    uint32_t mask = 0;
    for (uint32_t i = 0; i < k->channelCount; ++i) {
        const auto channel = k->channels[i];
        ccu::Variable destination;
        CCU_CHK_RET(ccu::LoadArg(destination, 3 + i));
        auto address = ccu::GetResByChannel<ccu::Variable>(channel, XN_ID_OUTPUT);
        auto remoteToken = ccu::GetResByChannel<ccu::Variable>(channel, XN_ID_TOKEN);
        CCU_CHK_RET(ccu::NotifyWait(channel, CKE_IDX_0, 3U));
        ccu::Variable cursor;
        cursor = address;
        cursor += sourceOffset;
        ccu::RemoteAddr source;
        source.addr = cursor;
        source.token = remoteToken;
        ccu::LocalAddr target;
        target.addr = destination;
        target.token = token;
        const uint32_t bit = 1U << i;
        CCU_CHK_RET(ccu::Read(channel, target, source, bytes, done, bit));
        mask |= bit;
    }
    CCU_CHK_RET(ccu::EventWait(done, mask));
    for (uint32_t i = 0; i < k->channelCount; ++i) {
        CCU_CHK_RET(ccu::NotifyRecord(k->channels[i], CKE_IDX_0, 1U << DATA_SIGNAL_ID));
    }
    return CcuResult::CCU_SUCCESS;
}
} // namespace ops_hccl
