/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <hcomm/hcomm_primitives.h>

#include "ccu_kernel.h"

namespace ops_hccl {
namespace ccu = ::AscendC::ccu;

namespace {
constexpr uint32_t OUTPUT_XN_ID = 1;
constexpr uint32_t TOKEN_XN_ID = 2;
constexpr uint32_t CKE_IDX = 0;
constexpr uint16_t OUTPUT_READY_BIT = 1U << OUTPUT_XN_ID;
constexpr uint16_t TOKEN_READY_BIT = 1U << TOKEN_XN_ID;
constexpr uint16_t DATA_DONE_BIT = 1U << 3;
constexpr uint16_t FUSED_ROOT_DONE_BIT = 1U << 5;
constexpr uint16_t READY_MASK = OUTPUT_READY_BIT | TOKEN_READY_BIT;
constexpr uint32_t PIPELINE_NOTIFY_SLOT_COUNT = 13;
constexpr uint32_t PIPELINE_SPLIT_CHUNK_COUNT = 26;
constexpr uint32_t PIPELINE_DATA_BIT_BASE = 3;
constexpr char PIPELINE_SPLIT_LOCAL_NOTIFY_TAG_0[] = "hccl_bcast_pipeline_split_die_0";
constexpr char PIPELINE_SPLIT_LOCAL_NOTIFY_TAG_1[] = "hccl_bcast_pipeline_split_die_1";

uint16_t PipelineDataBit(uint32_t chunkIdx)
{
    return static_cast<uint16_t>(
        1U << (PIPELINE_DATA_BIT_BASE + chunkIdx % PIPELINE_NOTIFY_SLOT_COUNT));
}

uint32_t PipelineSplitNotifyIndex(uint32_t chunkIdx)
{
    return chunkIdx / PIPELINE_NOTIFY_SLOT_COUNT;
}

const char *PipelineSplitLocalNotifyTag(uint32_t chunkIdx)
{
    return PipelineSplitNotifyIndex(chunkIdx) == 0 ?
        PIPELINE_SPLIT_LOCAL_NOTIFY_TAG_0 : PIPELINE_SPLIT_LOCAL_NOTIFY_TAG_1;
}

void LoadRemoteResources(const CcuKernelArgBroadcast *arg, std::vector<ccu::Variable> &remoteBuffers,
    std::vector<ccu::Variable> &remoteTokens)
{
    remoteBuffers.resize(arg->channelCount);
    remoteTokens.resize(arg->channelCount);
    for (uint32_t channelIdx = 0; channelIdx < arg->channelCount; ++channelIdx) {
        remoteBuffers[channelIdx] =
            ccu::GetResByChannel<ccu::Variable>(arg->channels[channelIdx], OUTPUT_XN_ID);
        remoteTokens[channelIdx] =
            ccu::GetResByChannel<ccu::Variable>(arg->channels[channelIdx], TOKEN_XN_ID);
    }
}

uint32_t GetWorkerIndex(const CcuKernelArgBroadcast *arg, uint32_t rank)
{
    const bool rootParticipates =
        (arg->rankSize == 4 || arg->rankSize == 12 || arg->rankSize == 16);
    return rootParticipates ? rank : (rank < arg->root ? rank : rank - 1);
}

bool IsLastWorker(const CcuKernelArgBroadcast *arg, uint32_t rank)
{
    const bool rootParticipates =
        (arg->rankSize == 4 || arg->rankSize == 12 || arg->rankSize == 16);
    const uint32_t workerCount = rootParticipates ? arg->rankSize : arg->rankSize - 1;
    return GetWorkerIndex(arg, rank) + 1 == workerCount;
}

CcuResult DirectReceiver(const CcuKernelArgBroadcast *arg)
{
    ccu::Variable localBuffer;
    ccu::Variable localToken;
    CCU_CHK_RET(ccu::LoadArg(localBuffer, 0));
    CCU_CHK_RET(ccu::LoadArg(localToken, 1));
    CCU_CHK_RET(ccu::WriteVariableWithNotify(
        arg->channels[0], localBuffer, OUTPUT_XN_ID, CKE_IDX, OUTPUT_READY_BIT));
    CCU_CHK_RET(ccu::WriteVariableWithNotify(
        arg->channels[0], localToken, TOKEN_XN_ID, CKE_IDX, TOKEN_READY_BIT));
    CCU_CHK_RET(ccu::NotifyWait(arg->channels[0], CKE_IDX, DATA_DONE_BIT));
    return CCU_SUCCESS;
}

CcuResult DirectSender(const CcuKernelArgBroadcast *arg)
{
    ccu::Variable localBuffer;
    ccu::Variable localToken;
    ccu::Variable dataSize;
    CCU_CHK_RET(ccu::LoadArg(localBuffer, 0));
    CCU_CHK_RET(ccu::LoadArg(localToken, 1));
    CCU_CHK_RET(ccu::LoadArg(dataSize, 2));

    std::vector<ccu::Variable> remoteBuffers;
    std::vector<ccu::Variable> remoteTokens;
    LoadRemoteResources(arg, remoteBuffers, remoteTokens);

    ccu::LocalAddr src;
    src.addr = localBuffer;
    src.token = localToken;
    ccu::Event event;
    for (uint32_t channelIdx = 0; channelIdx < arg->channelCount; ++channelIdx) {
        CCU_CHK_RET(ccu::NotifyWait(arg->channels[channelIdx], CKE_IDX, READY_MASK));
        ccu::RemoteAddr dst;
        dst.addr = remoteBuffers[channelIdx];
        dst.token = remoteTokens[channelIdx];
        const uint16_t eventBit = static_cast<uint16_t>(1U << channelIdx);
        CCU_CHK_RET(ccu::Write(arg->channels[channelIdx], dst, src, dataSize, event, eventBit));
    }
    for (uint32_t channelIdx = 0; channelIdx < arg->channelCount; ++channelIdx) {
        const uint16_t eventBit = static_cast<uint16_t>(1U << channelIdx);
        CCU_CHK_RET(ccu::EventWait(event, eventBit));
        CCU_CHK_RET(ccu::NotifyRecord(arg->channels[channelIdx], CKE_IDX, DATA_DONE_BIT));
    }
    return CCU_SUCCESS;
}

CcuResult DirectPullRoot(const CcuKernelArgBroadcast *arg)
{
    ccu::Variable localBuffer;
    ccu::Variable localToken;
    CCU_CHK_RET(ccu::LoadArg(localBuffer, 0));
    CCU_CHK_RET(ccu::LoadArg(localToken, 1));

    for (uint32_t channelIdx = 0; channelIdx < arg->channelCount; ++channelIdx) {
        CCU_CHK_RET(ccu::WriteVariableWithNotify(arg->channels[channelIdx], localBuffer,
            OUTPUT_XN_ID, CKE_IDX, OUTPUT_READY_BIT));
        CCU_CHK_RET(ccu::WriteVariableWithNotify(arg->channels[channelIdx], localToken,
            TOKEN_XN_ID, CKE_IDX, TOKEN_READY_BIT));
    }
    for (uint32_t channelIdx = 0; channelIdx < arg->channelCount; ++channelIdx) {
        CCU_CHK_RET(ccu::NotifyWait(arg->channels[channelIdx], CKE_IDX, DATA_DONE_BIT));
    }
    return CCU_SUCCESS;
}

CcuResult DirectPullReceiver(const CcuKernelArgBroadcast *arg)
{
    ccu::Variable localBuffer;
    ccu::Variable localToken;
    ccu::Variable dataSize;
    CCU_CHK_RET(ccu::LoadArg(localBuffer, 0));
    CCU_CHK_RET(ccu::LoadArg(localToken, 1));
    CCU_CHK_RET(ccu::LoadArg(dataSize, 2));

    std::vector<ccu::Variable> remoteBuffers;
    std::vector<ccu::Variable> remoteTokens;
    LoadRemoteResources(arg, remoteBuffers, remoteTokens);
    CCU_CHK_RET(ccu::NotifyWait(arg->channels[0], CKE_IDX, READY_MASK));

    ccu::RemoteAddr src;
    src.addr = remoteBuffers[0];
    src.token = remoteTokens[0];
    ccu::LocalAddr dst;
    dst.addr = localBuffer;
    dst.token = localToken;
    ccu::Event event;
    CCU_CHK_RET(ccu::Read(arg->channels[0], dst, src, dataSize, event, 1));
    CCU_CHK_RET(ccu::EventWait(event, 1));
    CCU_CHK_RET(ccu::NotifyRecord(arg->channels[0], CKE_IDX, DATA_DONE_BIT));
    return CCU_SUCCESS;
}

CcuResult ScatterSender(const CcuKernelArgBroadcast *arg)
{
    ccu::Variable localToken;
    ccu::Variable baseSliceSize;
    ccu::Variable tailSize;
    CCU_CHK_RET(ccu::LoadArg(localToken, 0));
    CCU_CHK_RET(ccu::LoadArg(baseSliceSize, 1));
    CCU_CHK_RET(ccu::LoadArg(tailSize, 2));

    std::vector<ccu::Variable> localBuffers(arg->channelCount);
    std::vector<ccu::Variable> dataSizes(arg->channelCount);
    for (uint32_t channelIdx = 0; channelIdx < arg->channelCount; ++channelIdx) {
        CCU_CHK_RET(ccu::LoadArg(localBuffers[channelIdx], 3 + channelIdx));
        dataSizes[channelIdx] = baseSliceSize;
        if (IsLastWorker(arg, arg->peerRanks[channelIdx])) {
            dataSizes[channelIdx] += tailSize;
        }
    }

    std::vector<ccu::Variable> remoteBuffers;
    std::vector<ccu::Variable> remoteTokens;
    LoadRemoteResources(arg, remoteBuffers, remoteTokens);

    ccu::Event event;
    for (uint32_t channelIdx = 0; channelIdx < arg->channelCount; ++channelIdx) {
        CCU_CHK_RET(ccu::NotifyWait(arg->channels[channelIdx], CKE_IDX, READY_MASK));
        ccu::LocalAddr src;
        src.addr = localBuffers[channelIdx];
        src.token = localToken;
        ccu::RemoteAddr dst;
        dst.addr = remoteBuffers[channelIdx];
        dst.token = remoteTokens[channelIdx];
        const uint16_t eventBit = static_cast<uint16_t>(1U << channelIdx);
        CCU_CHK_RET(ccu::Write(
            arg->channels[channelIdx], dst, src, dataSizes[channelIdx], event, eventBit));
    }
    for (uint32_t channelIdx = 0; channelIdx < arg->channelCount; ++channelIdx) {
        const uint16_t eventBit = static_cast<uint16_t>(1U << channelIdx);
        CCU_CHK_RET(ccu::EventWait(event, eventBit));
        CCU_CHK_RET(ccu::NotifyRecord(arg->channels[channelIdx], CKE_IDX, DATA_DONE_BIT));
    }
    return CCU_SUCCESS;
}

CcuResult FusedScatterRoot(const CcuKernelArgBroadcast *arg)
{
    ccu::Variable localToken;
    ccu::Variable localBase;
    ccu::Variable baseSliceSize;
    ccu::Variable tailSize;
    ccu::Variable rootOffset;
    ccu::Variable rootSize;
    CCU_CHK_RET(ccu::LoadArg(localToken, 0));
    CCU_CHK_RET(ccu::LoadArg(localBase, 1));
    CCU_CHK_RET(ccu::LoadArg(baseSliceSize, 2));
    CCU_CHK_RET(ccu::LoadArg(tailSize, 3));
    CCU_CHK_RET(ccu::LoadArg(rootOffset, 4));
    CCU_CHK_RET(ccu::LoadArg(rootSize, 5));

    std::vector<ccu::Variable> localOffsets(arg->channelCount);
    std::vector<ccu::Variable> dataSizes(arg->channelCount);
    std::vector<ccu::Variable> remoteBuffers;
    std::vector<ccu::Variable> remoteTokens;
    LoadRemoteResources(arg, remoteBuffers, remoteTokens);
    for (uint32_t channelIdx = 0; channelIdx < arg->channelCount; ++channelIdx) {
        CCU_CHK_RET(ccu::LoadArg(localOffsets[channelIdx], 6 + channelIdx));
        dataSizes[channelIdx] = baseSliceSize;
        if (IsLastWorker(arg, arg->peerRanks[channelIdx])) {
            dataSizes[channelIdx] += tailSize;
        }
    }

    ccu::Event event;
    uint16_t dataEventMask = 0;
    for (uint32_t channelIdx = 0; channelIdx < arg->channelCount; ++channelIdx) {
        CCU_CHK_RET(ccu::NotifyWait(arg->channels[channelIdx], CKE_IDX, READY_MASK));
        ccu::LocalAddr src;
        src.addr = localBase;
        src.addr += localOffsets[channelIdx];
        src.token = localToken;
        ccu::RemoteAddr dst;
        dst.addr = remoteBuffers[channelIdx];
        dst.addr += localOffsets[channelIdx];
        dst.token = remoteTokens[channelIdx];
        const uint16_t eventBit = static_cast<uint16_t>(1U << channelIdx);
        CCU_CHK_RET(ccu::Write(
            arg->channels[channelIdx], dst, src, dataSizes[channelIdx], event, eventBit));
        dataEventMask = static_cast<uint16_t>(dataEventMask | eventBit);
    }
    CCU_CHK_RET(ccu::EventWait(event, dataEventMask));
    for (uint32_t channelIdx = 0; channelIdx < arg->channelCount; ++channelIdx) {
        CCU_CHK_RET(ccu::NotifyRecord(arg->channels[channelIdx], CKE_IDX, DATA_DONE_BIT));
    }

    ccu::LocalAddr rootSrc;
    rootSrc.addr = localBase;
    rootSrc.addr += rootOffset;
    rootSrc.token = localToken;
    uint16_t rootEventMask = 0;
    for (uint32_t channelIdx = 0; channelIdx < arg->channelCount; ++channelIdx) {
        ccu::RemoteAddr rootDst;
        rootDst.addr = remoteBuffers[channelIdx];
        rootDst.addr += rootOffset;
        rootDst.token = remoteTokens[channelIdx];
        const uint16_t rootEventBit = static_cast<uint16_t>(1U << channelIdx);
        CCU_CHK_RET(ccu::Write(
            arg->channels[channelIdx], rootDst, rootSrc, rootSize, event, rootEventBit));
        rootEventMask = static_cast<uint16_t>(rootEventMask | rootEventBit);
    }
    CCU_CHK_RET(ccu::EventWait(event, rootEventMask));
    for (uint32_t channelIdx = 0; channelIdx < arg->channelCount; ++channelIdx) {
        CCU_CHK_RET(ccu::NotifyRecord(
            arg->channels[channelIdx], CKE_IDX, FUSED_ROOT_DONE_BIT));
    }
    return CCU_SUCCESS;
}

CcuResult FusedScatterReceiver(const CcuKernelArgBroadcast *arg)
{
    ccu::Variable localBase;
    ccu::Variable localToken;
    CCU_CHK_RET(ccu::LoadArg(localBase, 0));
    CCU_CHK_RET(ccu::LoadArg(localToken, 1));
    for (uint32_t channelIdx = 0; channelIdx < arg->channelCount; ++channelIdx) {
        CCU_CHK_RET(ccu::WriteVariableWithNotify(
            arg->channels[channelIdx], localBase, OUTPUT_XN_ID, CKE_IDX, OUTPUT_READY_BIT));
        CCU_CHK_RET(ccu::WriteVariableWithNotify(
            arg->channels[channelIdx], localToken, TOKEN_XN_ID, CKE_IDX, TOKEN_READY_BIT));
    }
    for (uint32_t channelIdx = 0; channelIdx < arg->channelCount; ++channelIdx) {
        if (arg->peerRanks[channelIdx] == arg->root) {
            CCU_CHK_RET(ccu::NotifyWait(
                arg->channels[channelIdx], CKE_IDX, DATA_DONE_BIT));
        }
    }
    return CCU_SUCCESS;
}

CcuResult PullAllgather(const CcuKernelArgBroadcast *arg)
{
    ccu::Variable localSource;
    ccu::Variable localToken;
    ccu::Variable baseSliceSize;
    ccu::Variable tailSize;
    CCU_CHK_RET(ccu::LoadArg(localSource, 0));
    CCU_CHK_RET(ccu::LoadArg(localToken, 1));
    CCU_CHK_RET(ccu::LoadArg(baseSliceSize, 2));
    CCU_CHK_RET(ccu::LoadArg(tailSize, 3));

    std::vector<ccu::Variable> localDestinations(arg->channelCount);
    std::vector<ccu::Variable> remoteSizes(arg->channelCount);
    for (uint32_t channelIdx = 0; channelIdx < arg->channelCount; ++channelIdx) {
        CCU_CHK_RET(ccu::LoadArg(localDestinations[channelIdx], 4 + channelIdx));
        remoteSizes[channelIdx] = baseSliceSize;
        if (IsLastWorker(arg, arg->peerRanks[channelIdx])) {
            remoteSizes[channelIdx] += tailSize;
        }
        CCU_CHK_RET(ccu::WriteVariableWithNotify(arg->channels[channelIdx], localSource,
            OUTPUT_XN_ID, CKE_IDX, OUTPUT_READY_BIT));
        CCU_CHK_RET(ccu::WriteVariableWithNotify(
            arg->channels[channelIdx], localToken, TOKEN_XN_ID, CKE_IDX, TOKEN_READY_BIT));
    }
    std::vector<ccu::Variable> remoteBuffers;
    std::vector<ccu::Variable> remoteTokens;
    LoadRemoteResources(arg, remoteBuffers, remoteTokens);

    ccu::Event event;
    for (uint32_t channelIdx = 0; channelIdx < arg->channelCount; ++channelIdx) {
        CCU_CHK_RET(ccu::NotifyWait(arg->channels[channelIdx], CKE_IDX, READY_MASK));
        ccu::LocalAddr dst;
        dst.addr = localDestinations[channelIdx];
        dst.token = localToken;
        ccu::RemoteAddr src;
        src.addr = remoteBuffers[channelIdx];
        src.token = remoteTokens[channelIdx];
        const uint16_t eventBit = static_cast<uint16_t>(1U << channelIdx);
        CCU_CHK_RET(ccu::Read(
            arg->channels[channelIdx], dst, src, remoteSizes[channelIdx], event, eventBit));
    }
    for (uint32_t channelIdx = 0; channelIdx < arg->channelCount; ++channelIdx) {
        const uint16_t eventBit = static_cast<uint16_t>(1U << channelIdx);
        CCU_CHK_RET(ccu::EventWait(event, eventBit));
        CCU_CHK_RET(ccu::NotifyRecord(arg->channels[channelIdx], CKE_IDX, DATA_DONE_BIT));
    }
    for (uint32_t channelIdx = 0; channelIdx < arg->channelCount; ++channelIdx) {
        CCU_CHK_RET(ccu::NotifyWait(arg->channels[channelIdx], CKE_IDX, DATA_DONE_BIT));
    }
    return CCU_SUCCESS;
}

CcuResult PushAllgather(const CcuKernelArgBroadcast *arg)
{
    ccu::Variable localBase;
    ccu::Variable localSource;
    ccu::Variable localToken;
    ccu::Variable localOffset;
    ccu::Variable localSize;
    CCU_CHK_RET(ccu::LoadArg(localBase, 0));
    CCU_CHK_RET(ccu::LoadArg(localSource, 1));
    CCU_CHK_RET(ccu::LoadArg(localToken, 2));
    CCU_CHK_RET(ccu::LoadArg(localOffset, 3));
    CCU_CHK_RET(ccu::LoadArg(localSize, 4));

    for (uint32_t channelIdx = 0; channelIdx < arg->channelCount; ++channelIdx) {
        CCU_CHK_RET(ccu::WriteVariableWithNotify(arg->channels[channelIdx], localBase,
            OUTPUT_XN_ID, CKE_IDX, OUTPUT_READY_BIT));
        CCU_CHK_RET(ccu::WriteVariableWithNotify(
            arg->channels[channelIdx], localToken, TOKEN_XN_ID, CKE_IDX, TOKEN_READY_BIT));
    }
    std::vector<ccu::Variable> remoteBuffers;
    std::vector<ccu::Variable> remoteTokens;
    LoadRemoteResources(arg, remoteBuffers, remoteTokens);

    ccu::LocalAddr src;
    src.addr = localSource;
    src.token = localToken;
    ccu::Event event;
    for (uint32_t channelIdx = 0; channelIdx < arg->channelCount; ++channelIdx) {
        CCU_CHK_RET(ccu::NotifyWait(arg->channels[channelIdx], CKE_IDX, READY_MASK));
        ccu::RemoteAddr dst;
        dst.addr = remoteBuffers[channelIdx];
        dst.addr += localOffset;
        dst.token = remoteTokens[channelIdx];
        const uint16_t eventBit = static_cast<uint16_t>(1U << channelIdx);
        CCU_CHK_RET(ccu::Write(arg->channels[channelIdx], dst, src, localSize, event, eventBit));
    }
    for (uint32_t channelIdx = 0; channelIdx < arg->channelCount; ++channelIdx) {
        const uint16_t eventBit = static_cast<uint16_t>(1U << channelIdx);
        CCU_CHK_RET(ccu::EventWait(event, eventBit));
        CCU_CHK_RET(ccu::NotifyRecord(arg->channels[channelIdx], CKE_IDX, DATA_DONE_BIT));
    }
    for (uint32_t channelIdx = 0; channelIdx < arg->channelCount; ++channelIdx) {
        CCU_CHK_RET(ccu::NotifyWait(arg->channels[channelIdx], CKE_IDX, DATA_DONE_BIT));
    }
    return CCU_SUCCESS;
}

CcuResult FusedPushAllgather(const CcuKernelArgBroadcast *arg)
{
    ccu::Variable localBase;
    ccu::Variable localSource;
    ccu::Variable localToken;
    ccu::Variable localOffset;
    ccu::Variable localSize;
    CCU_CHK_RET(ccu::LoadArg(localBase, 0));
    CCU_CHK_RET(ccu::LoadArg(localSource, 1));
    CCU_CHK_RET(ccu::LoadArg(localToken, 2));
    CCU_CHK_RET(ccu::LoadArg(localOffset, 3));
    CCU_CHK_RET(ccu::LoadArg(localSize, 4));

    bool resourcesPrepublished = false;
    for (uint32_t channelIdx = 0; channelIdx < arg->channelCount; ++channelIdx) {
        if (arg->peerRanks[channelIdx] == arg->root) {
            resourcesPrepublished = true;
        }
    }
    if (!resourcesPrepublished) {
        for (uint32_t channelIdx = 0; channelIdx < arg->channelCount; ++channelIdx) {
            CCU_CHK_RET(ccu::WriteVariableWithNotify(arg->channels[channelIdx], localBase,
                OUTPUT_XN_ID, CKE_IDX, OUTPUT_READY_BIT));
            CCU_CHK_RET(ccu::WriteVariableWithNotify(
                arg->channels[channelIdx], localToken, TOKEN_XN_ID, CKE_IDX, TOKEN_READY_BIT));
        }
    }
    std::vector<ccu::Variable> remoteBuffers;
    std::vector<ccu::Variable> remoteTokens;
    LoadRemoteResources(arg, remoteBuffers, remoteTokens);

    ccu::LocalAddr src;
    src.addr = localSource;
    src.token = localToken;
    ccu::Event event;
    uint16_t eventMask = 0;
    for (uint32_t channelIdx = 0; channelIdx < arg->channelCount; ++channelIdx) {
        if (arg->peerRanks[channelIdx] == arg->root) {
            continue;
        }
        CCU_CHK_RET(ccu::NotifyWait(arg->channels[channelIdx], CKE_IDX, READY_MASK));
        ccu::RemoteAddr dst;
        dst.addr = remoteBuffers[channelIdx];
        dst.addr += localOffset;
        dst.token = remoteTokens[channelIdx];
        const uint16_t eventBit = static_cast<uint16_t>(1U << channelIdx);
        CCU_CHK_RET(ccu::Write(arg->channels[channelIdx], dst, src, localSize, event, eventBit));
        eventMask = static_cast<uint16_t>(eventMask | eventBit);
    }
    if (eventMask != 0) {
        CCU_CHK_RET(ccu::EventWait(event, eventMask));
    }
    for (uint32_t channelIdx = 0; channelIdx < arg->channelCount; ++channelIdx) {
        if (arg->peerRanks[channelIdx] != arg->root) {
            CCU_CHK_RET(ccu::NotifyRecord(arg->channels[channelIdx], CKE_IDX, DATA_DONE_BIT));
        }
    }
    for (uint32_t channelIdx = 0; channelIdx < arg->channelCount; ++channelIdx) {
        if (arg->peerRanks[channelIdx] == arg->root) {
            CCU_CHK_RET(ccu::NotifyWait(arg->channels[channelIdx], CKE_IDX, FUSED_ROOT_DONE_BIT));
        } else {
            CCU_CHK_RET(ccu::NotifyWait(arg->channels[channelIdx], CKE_IDX, DATA_DONE_BIT));
        }
    }
    return CCU_SUCCESS;
}

CcuResult RecursiveDynamic(const CcuKernelArgBroadcast *arg)
{
    ccu::Variable localToken;
    ccu::Variable localSource;
    ccu::Variable localDestination;
    ccu::Variable dataSize;
    ccu::Variable operation;
    CCU_CHK_RET(ccu::LoadArg(localToken, 0));
    CCU_CHK_RET(ccu::LoadArg(localSource, 1));
    CCU_CHK_RET(ccu::LoadArg(localDestination, 2));
    CCU_CHK_RET(ccu::LoadArg(dataSize, 3));
    CCU_CHK_RET(ccu::LoadArg(operation, 4));

    std::vector<ccu::Variable> remoteBuffers;
    std::vector<ccu::Variable> remoteTokens;
    LoadRemoteResources(arg, remoteBuffers, remoteTokens);

    if (arg->partnerRank == static_cast<uint32_t>(BroadcastKernelRole::RECURSIVE_SCATTER_SENDER)) {
        CCU_IF(operation == 0) {
            CCU_CHK_RET(ccu::NotifyWait(arg->channels[0], CKE_IDX, READY_MASK));
            ccu::LocalAddr src;
            src.addr = localSource;
            src.token = localToken;
            ccu::RemoteAddr dst;
            dst.addr = remoteBuffers[0];
            dst.token = remoteTokens[0];
            ccu::Event event;
            CCU_CHK_RET(ccu::Write(arg->channels[0], dst, src, dataSize, event, 1));
            CCU_CHK_RET(ccu::EventWait(event, 1));
            CCU_CHK_RET(ccu::NotifyRecord(arg->channels[0], CKE_IDX, DATA_DONE_BIT));
        }
    }

    if (arg->partnerRank == static_cast<uint32_t>(BroadcastKernelRole::RECURSIVE_SCATTER_RECEIVER)) {
        CCU_IF(operation == 1) {
            CCU_CHK_RET(ccu::WriteVariableWithNotify(
                arg->channels[0], localDestination, OUTPUT_XN_ID, CKE_IDX, OUTPUT_READY_BIT));
            CCU_CHK_RET(ccu::WriteVariableWithNotify(
                arg->channels[0], localToken, TOKEN_XN_ID, CKE_IDX, TOKEN_READY_BIT));
            CCU_CHK_RET(ccu::NotifyWait(arg->channels[0], CKE_IDX, DATA_DONE_BIT));
        }
    }

    CCU_IF(operation == 2) {
        CCU_CHK_RET(ccu::WriteVariableWithNotify(
            arg->channels[0], localDestination, OUTPUT_XN_ID, CKE_IDX, OUTPUT_READY_BIT));
        CCU_CHK_RET(ccu::WriteVariableWithNotify(
            arg->channels[0], localToken, TOKEN_XN_ID, CKE_IDX, TOKEN_READY_BIT));
        CCU_CHK_RET(ccu::NotifyWait(arg->channels[0], CKE_IDX, READY_MASK));
        ccu::LocalAddr src;
        src.addr = localSource;
        src.token = localToken;
        ccu::RemoteAddr dst;
        dst.addr = remoteBuffers[0];
        dst.token = remoteTokens[0];
        ccu::Event event;
        CCU_CHK_RET(ccu::Write(arg->channels[0], dst, src, dataSize, event, 1));
        CCU_CHK_RET(ccu::EventWait(event, 1));
        CCU_CHK_RET(ccu::NotifyRecord(arg->channels[0], CKE_IDX, DATA_DONE_BIT));
        CCU_CHK_RET(ccu::NotifyWait(arg->channels[0], CKE_IDX, DATA_DONE_BIT));
    }
    return CCU_SUCCESS;
}

CcuResult PipelineSplitRoot(const CcuKernelArgBroadcast *arg)
{
    ccu::Variable localToken;
    ccu::Variable firstAddress;
    ccu::Variable secondAddress;
    ccu::Variable firstChunkSize;
    ccu::Variable firstLastSize;
    ccu::Variable secondChunkSize;
    ccu::Variable secondLastSize;
    CCU_CHK_RET(ccu::LoadArg(localToken, 0));
    CCU_CHK_RET(ccu::LoadArg(firstAddress, 1));
    CCU_CHK_RET(ccu::LoadArg(secondAddress, 2));
    CCU_CHK_RET(ccu::LoadArg(firstChunkSize, 3));
    CCU_CHK_RET(ccu::LoadArg(firstLastSize, 4));
    CCU_CHK_RET(ccu::LoadArg(secondChunkSize, 5));
    CCU_CHK_RET(ccu::LoadArg(secondLastSize, 6));

    std::vector<ccu::Variable> remoteBuffers;
    std::vector<ccu::Variable> remoteTokens;
    LoadRemoteResources(arg, remoteBuffers, remoteTokens);
    std::vector<ccu::LocalAddr> sources(arg->channelCount);
    std::vector<ccu::RemoteAddr> destinations(arg->channelCount);
    for (uint32_t channelIdx = 0; channelIdx < arg->channelCount; ++channelIdx) {
        CCU_CHK_RET(ccu::NotifyWait(arg->channels[channelIdx], CKE_IDX, READY_MASK));
        const bool firstHalf = arg->peerRanks[channelIdx] == arg->relayRankA;
        sources[channelIdx].addr = firstHalf ? firstAddress : secondAddress;
        sources[channelIdx].token = localToken;
        destinations[channelIdx].addr = remoteBuffers[channelIdx];
        destinations[channelIdx].token = remoteTokens[channelIdx];
    }

    ccu::Event event;
    for (uint32_t chunkIdx = 0; chunkIdx < PIPELINE_SPLIT_CHUNK_COUNT; ++chunkIdx) {
        const uint32_t notifyIdx = PipelineSplitNotifyIndex(chunkIdx);
        uint16_t eventMask = 0;
        for (uint32_t channelIdx = 0; channelIdx < arg->channelCount; ++channelIdx) {
            const bool firstHalf = arg->peerRanks[channelIdx] == arg->relayRankA;
            const ccu::Variable &transferSize = firstHalf ?
                (chunkIdx + 1 == PIPELINE_SPLIT_CHUNK_COUNT ?
                    firstLastSize : firstChunkSize) :
                (chunkIdx + 1 == PIPELINE_SPLIT_CHUNK_COUNT ?
                    secondLastSize : secondChunkSize);
            const uint16_t eventBit = static_cast<uint16_t>(1U << channelIdx);
            CCU_CHK_RET(ccu::Write(arg->channels[channelIdx], destinations[channelIdx],
                sources[channelIdx], transferSize, event, eventBit));
            eventMask = static_cast<uint16_t>(eventMask | eventBit);
        }
        CCU_CHK_RET(ccu::EventWait(event, eventMask));
        const uint16_t dataBit = PipelineDataBit(chunkIdx);
        for (uint32_t channelIdx = 0; channelIdx < arg->channelCount; ++channelIdx) {
            CCU_CHK_RET(ccu::NotifyRecord(arg->channels[channelIdx], notifyIdx, dataBit));
            if (chunkIdx + 1 != PIPELINE_SPLIT_CHUNK_COUNT) {
                const bool firstHalf = arg->peerRanks[channelIdx] == arg->relayRankA;
                const ccu::Variable &transferSize = firstHalf ? firstChunkSize : secondChunkSize;
                sources[channelIdx].addr += transferSize;
                destinations[channelIdx].addr += transferSize;
            }
        }
    }
    return CCU_SUCCESS;
}

CcuResult PipelineSplitRelay(const CcuKernelArgBroadcast *arg)
{
    ccu::Variable localToken;
    ccu::Variable firstAddress;
    ccu::Variable secondAddress;
    ccu::Variable firstChunkSize;
    ccu::Variable firstLastSize;
    ccu::Variable secondChunkSize;
    ccu::Variable secondLastSize;
    CCU_CHK_RET(ccu::LoadArg(localToken, 0));
    CCU_CHK_RET(ccu::LoadArg(firstAddress, 1));
    CCU_CHK_RET(ccu::LoadArg(secondAddress, 2));
    CCU_CHK_RET(ccu::LoadArg(firstChunkSize, 3));
    CCU_CHK_RET(ccu::LoadArg(firstLastSize, 4));
    CCU_CHK_RET(ccu::LoadArg(secondChunkSize, 5));
    CCU_CHK_RET(ccu::LoadArg(secondLastSize, 6));

    const bool ownsFirstHalf = arg->rankId == arg->relayRankA;
    int32_t rootChannelIdx = -1;
    int32_t partnerChannelIdx = -1;
    for (uint32_t channelIdx = 0; channelIdx < arg->channelCount; ++channelIdx) {
        if (arg->peerRanks[channelIdx] == arg->root) {
            rootChannelIdx = static_cast<int32_t>(channelIdx);
            CCU_CHK_RET(ccu::WriteVariableWithNotify(arg->channels[channelIdx],
                ownsFirstHalf ? firstAddress : secondAddress,
                OUTPUT_XN_ID, CKE_IDX, OUTPUT_READY_BIT));
            CCU_CHK_RET(ccu::WriteVariableWithNotify(arg->channels[channelIdx], localToken,
                TOKEN_XN_ID, CKE_IDX, TOKEN_READY_BIT));
        } else if (arg->peerRanks[channelIdx] == arg->partnerRank) {
            partnerChannelIdx = static_cast<int32_t>(channelIdx);
            CCU_CHK_RET(ccu::WriteVariableWithNotify(arg->channels[channelIdx],
                ownsFirstHalf ? secondAddress : firstAddress,
                OUTPUT_XN_ID, CKE_IDX, OUTPUT_READY_BIT));
            CCU_CHK_RET(ccu::WriteVariableWithNotify(arg->channels[channelIdx], localToken,
                TOKEN_XN_ID, CKE_IDX, TOKEN_READY_BIT));
        }
    }

    std::vector<ccu::Variable> remoteBuffers;
    std::vector<ccu::Variable> remoteTokens;
    LoadRemoteResources(arg, remoteBuffers, remoteTokens);
    std::vector<ccu::RemoteAddr> destinations(arg->channelCount);
    uint32_t outgoingCount = 0;
    for (uint32_t channelIdx = 0; channelIdx < arg->channelCount; ++channelIdx) {
        if (arg->peerRanks[channelIdx] == arg->root) {
            continue;
        }
        CCU_CHK_RET(ccu::NotifyWait(arg->channels[channelIdx], CKE_IDX, READY_MASK));
        destinations[channelIdx].addr = remoteBuffers[channelIdx];
        destinations[channelIdx].token = remoteTokens[channelIdx];
        ++outgoingCount;
    }

    ccu::LocalAddr source;
    source.addr = ownsFirstHalf ? firstAddress : secondAddress;
    source.token = localToken;
    ccu::Event event;
    const uint32_t windowSize =
        outgoingCount == 0 ? MAX_RANK_SIZE : MAX_RANK_SIZE / outgoingCount;
    std::vector<uint16_t> pendingMasks(windowSize, 0);
    uint16_t finalEventMask = 0;
    for (uint32_t chunkIdx = 0; chunkIdx < PIPELINE_SPLIT_CHUNK_COUNT; ++chunkIdx) {
        const uint32_t notifyIdx = PipelineSplitNotifyIndex(chunkIdx);
        const uint16_t dataBit = PipelineDataBit(chunkIdx);
        if (rootChannelIdx >= 0) {
            CCU_CHK_RET(ccu::NotifyWait(
                arg->channels[static_cast<uint32_t>(rootChannelIdx)], notifyIdx, dataBit));
            if (arg->pipelineCrossDie) {
                CCU_CHK_RET(ccu::EventRecord(PipelineSplitLocalNotifyTag(chunkIdx), dataBit));
            }
        } else {
            CCU_CHK_RET(ccu::EventWait(PipelineSplitLocalNotifyTag(chunkIdx), dataBit));
        }

        const ccu::Variable &transferSize = ownsFirstHalf ?
            (chunkIdx + 1 == PIPELINE_SPLIT_CHUNK_COUNT ? firstLastSize : firstChunkSize) :
            (chunkIdx + 1 == PIPELINE_SPLIT_CHUNK_COUNT ? secondLastSize : secondChunkSize);
        const uint32_t slotIdx = chunkIdx % windowSize;
        if (outgoingCount != 0 && chunkIdx >= windowSize) {
            CCU_CHK_RET(ccu::EventWait(event, pendingMasks[slotIdx]));
        }
        uint16_t eventMask = 0;
        uint32_t outgoingIdx = 0;
        for (uint32_t channelIdx = 0; channelIdx < arg->channelCount; ++channelIdx) {
            if (arg->peerRanks[channelIdx] == arg->root) {
                continue;
            }
            const uint16_t eventBit = static_cast<uint16_t>(
                1U << (slotIdx * outgoingCount + outgoingIdx));
            CCU_CHK_RET(ccu::Write(arg->channels[channelIdx], destinations[channelIdx],
                source, transferSize, event, eventBit));
            eventMask = static_cast<uint16_t>(eventMask | eventBit);
            ++outgoingIdx;
        }
        pendingMasks[slotIdx] = eventMask;
        finalEventMask = eventMask;
        for (uint32_t channelIdx = 0; channelIdx < arg->channelCount; ++channelIdx) {
            if (arg->peerRanks[channelIdx] != arg->root) {
                if (chunkIdx + 1 != PIPELINE_SPLIT_CHUNK_COUNT) {
                    destinations[channelIdx].addr += transferSize;
                }
            }
        }
        if (chunkIdx + 1 != PIPELINE_SPLIT_CHUNK_COUNT) {
            source.addr += transferSize;
        }
    }

    if (finalEventMask != 0) {
        CCU_CHK_RET(ccu::EventWait(event, finalEventMask));
    }
    constexpr uint32_t lastChunkIdx = PIPELINE_SPLIT_CHUNK_COUNT - 1;
    for (uint32_t channelIdx = 0; channelIdx < arg->channelCount; ++channelIdx) {
        if (arg->peerRanks[channelIdx] != arg->root) {
            CCU_CHK_RET(ccu::NotifyRecord(arg->channels[channelIdx],
                PipelineSplitNotifyIndex(lastChunkIdx), PipelineDataBit(lastChunkIdx)));
        }
    }
    if (partnerChannelIdx >= 0) {
        CCU_CHK_RET(ccu::NotifyWait(
            arg->channels[static_cast<uint32_t>(partnerChannelIdx)],
            PipelineSplitNotifyIndex(lastChunkIdx), PipelineDataBit(lastChunkIdx)));
    }
    return CCU_SUCCESS;
}

CcuResult PipelineSplitTail(const CcuKernelArgBroadcast *arg)
{
    ccu::Variable localToken;
    ccu::Variable firstAddress;
    ccu::Variable secondAddress;
    ccu::Variable firstChunkSize;
    ccu::Variable firstLastSize;
    ccu::Variable secondChunkSize;
    ccu::Variable secondLastSize;
    CCU_CHK_RET(ccu::LoadArg(localToken, 0));
    CCU_CHK_RET(ccu::LoadArg(firstAddress, 1));
    CCU_CHK_RET(ccu::LoadArg(secondAddress, 2));
    CCU_CHK_RET(ccu::LoadArg(firstChunkSize, 3));
    CCU_CHK_RET(ccu::LoadArg(firstLastSize, 4));
    CCU_CHK_RET(ccu::LoadArg(secondChunkSize, 5));
    CCU_CHK_RET(ccu::LoadArg(secondLastSize, 6));

    for (uint32_t channelIdx = 0; channelIdx < arg->channelCount; ++channelIdx) {
        const bool firstHalf = arg->peerRanks[channelIdx] == arg->relayRankA;
        CCU_CHK_RET(ccu::WriteVariableWithNotify(arg->channels[channelIdx],
            firstHalf ? firstAddress : secondAddress,
            OUTPUT_XN_ID, CKE_IDX, OUTPUT_READY_BIT));
        CCU_CHK_RET(ccu::WriteVariableWithNotify(arg->channels[channelIdx], localToken,
            TOKEN_XN_ID, CKE_IDX, TOKEN_READY_BIT));
    }
    constexpr uint32_t lastChunkIdx = PIPELINE_SPLIT_CHUNK_COUNT - 1;
    for (uint32_t channelIdx = 0; channelIdx < arg->channelCount; ++channelIdx) {
        CCU_CHK_RET(ccu::NotifyWait(
            arg->channels[channelIdx], PipelineSplitNotifyIndex(lastChunkIdx),
            PipelineDataBit(lastChunkIdx)));
    }
    return CCU_SUCCESS;
}

} // namespace

CcuResult CcuKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgBroadcast *>(arg);
    if (kernelArg == nullptr || kernelArg->rankSize <= 1 || kernelArg->rankSize > MAX_RANK_SIZE ||
        kernelArg->channelCount == 0 || kernelArg->channelCount >= MAX_RANK_SIZE) {
        return CCU_E_PARA;
    }
    if (kernelArg->role != BroadcastKernelRole::PUSH_RECEIVER &&
        kernelArg->role != BroadcastKernelRole::PUSH_SENDER &&
        kernelArg->role != BroadcastKernelRole::SCATTER_RECEIVER &&
        kernelArg->role != BroadcastKernelRole::SCATTER_SENDER &&
        kernelArg->role != BroadcastKernelRole::PULL_ALLGATHER &&
        kernelArg->role != BroadcastKernelRole::PUSH_ALLGATHER &&
        kernelArg->role != BroadcastKernelRole::PULL_ROOT &&
        kernelArg->role != BroadcastKernelRole::PULL_RECEIVER &&
        kernelArg->role != BroadcastKernelRole::RECURSIVE_DYNAMIC &&
        kernelArg->role != BroadcastKernelRole::PIPELINE_SPLIT_ROOT &&
        kernelArg->role != BroadcastKernelRole::PIPELINE_SPLIT_RELAY &&
        kernelArg->role != BroadcastKernelRole::PIPELINE_SPLIT_TAIL &&
        kernelArg->role != BroadcastKernelRole::FUSED_SCATTER_ROOT &&
        kernelArg->role != BroadcastKernelRole::FUSED_SCATTER_RECEIVER &&
        kernelArg->role != BroadcastKernelRole::FUSED_PUSH_ALLGATHER) {
        return CCU_E_PARA;
    }
    if ((kernelArg->role == BroadcastKernelRole::PUSH_RECEIVER ||
        kernelArg->role == BroadcastKernelRole::SCATTER_RECEIVER ||
        kernelArg->role == BroadcastKernelRole::PULL_RECEIVER) &&
        kernelArg->channelCount != 1) {
        return CCU_E_PARA;
    }

    switch (kernelArg->role) {
        case BroadcastKernelRole::PUSH_RECEIVER:
            return DirectReceiver(kernelArg);
        case BroadcastKernelRole::PUSH_SENDER:
            return DirectSender(kernelArg);
        case BroadcastKernelRole::SCATTER_RECEIVER:
            return DirectReceiver(kernelArg);
        case BroadcastKernelRole::SCATTER_SENDER:
            return ScatterSender(kernelArg);
        case BroadcastKernelRole::PULL_ALLGATHER:
            return PullAllgather(kernelArg);
        case BroadcastKernelRole::PUSH_ALLGATHER:
            return PushAllgather(kernelArg);
        case BroadcastKernelRole::PULL_ROOT:
            return DirectPullRoot(kernelArg);
        case BroadcastKernelRole::PULL_RECEIVER:
            return DirectPullReceiver(kernelArg);
        case BroadcastKernelRole::RECURSIVE_DYNAMIC:
            return RecursiveDynamic(kernelArg);
        case BroadcastKernelRole::PIPELINE_SPLIT_ROOT:
            return PipelineSplitRoot(kernelArg);
        case BroadcastKernelRole::PIPELINE_SPLIT_RELAY:
            return PipelineSplitRelay(kernelArg);
        case BroadcastKernelRole::PIPELINE_SPLIT_TAIL:
            return PipelineSplitTail(kernelArg);
        case BroadcastKernelRole::FUSED_SCATTER_ROOT:
            return FusedScatterRoot(kernelArg);
        case BroadcastKernelRole::FUSED_SCATTER_RECEIVER:
            return FusedScatterReceiver(kernelArg);
        case BroadcastKernelRole::FUSED_PUSH_ALLGATHER:
            return FusedPushAllgather(kernelArg);
        default:
            return CCU_E_PARA;
    }
}
} // namespace ops_hccl
