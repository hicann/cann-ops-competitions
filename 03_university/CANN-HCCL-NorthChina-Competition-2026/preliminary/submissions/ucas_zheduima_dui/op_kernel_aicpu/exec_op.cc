/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "custom.h"
#include "log.h"
#include "exec_op.h"

#include <algorithm>
#include <cstdint>
#include <limits>

namespace {
HcclResult GetAllGatherDataTypeSize(HcclDataType dataType, uint64_t &dataTypeSize)
{
    switch (dataType) {
        case HCCL_DATA_TYPE_INT8:
        case HCCL_DATA_TYPE_UINT8:
        case HCCL_DATA_TYPE_HIF8:
        case HCCL_DATA_TYPE_FP8E4M3:
        case HCCL_DATA_TYPE_FP8E5M2:
        case HCCL_DATA_TYPE_FP8E8M0:
            dataTypeSize = 1;
            break;
        case HCCL_DATA_TYPE_INT16:
        case HCCL_DATA_TYPE_UINT16:
        case HCCL_DATA_TYPE_FP16:
        case HCCL_DATA_TYPE_BFP16:
            dataTypeSize = 2;
            break;
        case HCCL_DATA_TYPE_INT32:
        case HCCL_DATA_TYPE_UINT32:
        case HCCL_DATA_TYPE_FP32:
            dataTypeSize = 4;
            break;
        case HCCL_DATA_TYPE_INT64:
        case HCCL_DATA_TYPE_UINT64:
        case HCCL_DATA_TYPE_FP64:
            dataTypeSize = 8;
            break;
        case HCCL_DATA_TYPE_INT128:
            dataTypeSize = 16;
            break;
        default:
            HCCL_ERROR("Unsupported data type[%d]", static_cast<int32_t>(dataType));
            return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}

HcclResult ValidateTopologyPlan(const OpParam &param, const AlgResourceCtx &resCtx)
{
    CHK_PRT_RET(resCtx.topologyPlanValid > 1, HCCL_ERROR("Invalid topology-plan flag[%u]", resCtx.topologyPlanValid),
        HCCL_E_INTERNAL);
    if (resCtx.topologyPlanValid == 0) {
        return HCCL_SUCCESS;
    }

    CHK_PRT_RET(param.rankSize != OPTIMIZED_RANK_NUM || resCtx.localGroupIndex >= OPTIMIZED_GROUP_SIZE,
        HCCL_ERROR("Invalid optimized topology dimensions"), HCCL_E_INTERNAL);
    bool groupBasesValid = (resCtx.localGroupBaseRank == 0 && resCtx.remoteGroupBaseRank == OPTIMIZED_GROUP_SIZE)
                           || (resCtx.localGroupBaseRank == OPTIMIZED_GROUP_SIZE && resCtx.remoteGroupBaseRank == 0);
    CHK_PRT_RET(!groupBasesValid,
        HCCL_ERROR("Invalid optimized group bases: local[%u], remote[%u]", resCtx.localGroupBaseRank,
            resCtx.remoteGroupBaseRank),
        HCCL_E_INTERNAL);
    CHK_PRT_RET(param.myRank != resCtx.localGroupBaseRank + resCtx.localGroupIndex
                    || resCtx.proxyRank != resCtx.remoteGroupBaseRank + resCtx.localGroupIndex,
        HCCL_ERROR("Optimized topology rank mapping is inconsistent"), HCCL_E_INTERNAL);
    CHK_PRT_RET(resCtx.recursiveDoublingChannelIndices.size() != RECURSIVE_DOUBLING_ROUND_NUM,
        HCCL_ERROR("Invalid recursive-doubling channel count[%zu]", resCtx.recursiveDoublingChannelIndices.size()),
        HCCL_E_INTERNAL);

    for (size_t channelIndex = 0; channelIndex < resCtx.channels.size(); ++channelIndex) {
        const ChannelInfo &channel = resCtx.channels[channelIndex];
        uint32_t representedRank = INVALID_VALUE_RANKID;
        if (channel.remoteRank >= resCtx.localGroupBaseRank
            && channel.remoteRank < resCtx.localGroupBaseRank + OPTIMIZED_GROUP_SIZE) {
            representedRank = resCtx.remoteGroupBaseRank + channel.remoteRank - resCtx.localGroupBaseRank;
        } else if (channel.remoteRank >= resCtx.remoteGroupBaseRank
                   && channel.remoteRank < resCtx.remoteGroupBaseRank + OPTIMIZED_GROUP_SIZE) {
            representedRank = resCtx.localGroupBaseRank + channel.remoteRank - resCtx.remoteGroupBaseRank;
        }
        CHK_PRT_RET(channel.representedRank != representedRank,
            HCCL_ERROR("Invalid proxy mapping: CCL owner[%u], represented rank[%u], expected[%u]", channel.remoteRank,
                channel.representedRank, representedRank),
            HCCL_E_INTERNAL);
    }

    for (uint32_t round = 0; round < RECURSIVE_DOUBLING_ROUND_NUM; ++round) {
        uint32_t channelIndex = resCtx.recursiveDoublingChannelIndices[round];
        CHK_PRT_RET(channelIndex >= resCtx.channels.size(),
            HCCL_ERROR("Recursive-doubling channel index[%u] is out of range", channelIndex), HCCL_E_INTERNAL);
        for (uint32_t previousRound = 0; previousRound < round; ++previousRound) {
            CHK_PRT_RET(channelIndex == resCtx.recursiveDoublingChannelIndices[previousRound],
                HCCL_ERROR("Recursive-doubling rounds reuse channel index[%u]", channelIndex), HCCL_E_INTERNAL);
        }
        uint32_t expectedPartner = resCtx.proxyRank;
        if (round + 1 < RECURSIVE_DOUBLING_ROUND_NUM) {
            uint32_t mask = 1U << round;
            expectedPartner = resCtx.localGroupBaseRank + (resCtx.localGroupIndex ^ mask);
        }
        CHK_PRT_RET(resCtx.channels[channelIndex].remoteRank != expectedPartner,
            HCCL_ERROR("Recursive-doubling round[%u] uses rank[%u], expected[%u]", round,
                resCtx.channels[channelIndex].remoteRank, expectedPartner),
            HCCL_E_INTERNAL);
    }
    return HCCL_SUCCESS;
}

HcclResult ValidateAndGetBufferSize(
    const OpParam &param, const AlgResourceCtx &resCtx, uint64_t dataTypeSize, uint64_t &bufferSize)
{
    CHK_PRT_RET(param.rankSize == 0 || param.myRank >= param.rankSize,
        HCCL_ERROR("Invalid rank information: rank[%u], rankSize[%u]", param.myRank, param.rankSize), HCCL_E_PARA);
    CHK_PRT_RET(resCtx.threads.size() != resCtx.channels.size() + 1 + COPIER_THREAD_NUM,
        HCCL_ERROR(
            "Thread count[%zu] does not match channel count[%zu]", resCtx.threads.size(), resCtx.channels.size()),
        HCCL_E_INTERNAL);
    CHK_PRT_RET(resCtx.channels.size() != static_cast<size_t>(param.rankSize - 1),
        HCCL_ERROR("Channel count[%zu] does not match rankSize[%u]", resCtx.channels.size(), param.rankSize),
        HCCL_E_INTERNAL);
    CHK_PRT_RET(resCtx.threads.empty() || resCtx.threads[0] == 0 || resCtx.aicpuThread != resCtx.threads[0],
        HCCL_ERROR("Invalid main thread resource"), HCCL_E_INTERNAL);
    CHK_RET(ValidateTopologyPlan(param, resCtx));

    if (param.rankSize == 1) {
        bufferSize = 0;
        return HCCL_SUCCESS;
    }

    CHK_PRT_RET(resCtx.localBuffer.addr == nullptr || resCtx.localBuffer.size < dataTypeSize,
        HCCL_ERROR("Invalid local CCL buffer, size[%llu]", static_cast<unsigned long long>(resCtx.localBuffer.size)),
        HCCL_E_INTERNAL);

    bufferSize = resCtx.localBuffer.size;
    for (size_t channelIndex = 0; channelIndex < resCtx.channels.size(); ++channelIndex) {
        const ChannelInfo &channel = resCtx.channels[channelIndex];
        CHK_PRT_RET(channel.remoteRank >= param.rankSize || channel.remoteRank == param.myRank,
            HCCL_ERROR("Invalid remote rank[%u] for channel[%zu]", channel.remoteRank, channelIndex), HCCL_E_INTERNAL);
        for (size_t previousIndex = 0; previousIndex < channelIndex; ++previousIndex) {
            CHK_PRT_RET(resCtx.channels[previousIndex].remoteRank == channel.remoteRank,
                HCCL_ERROR("More than one channel was created for remote rank[%u]", channel.remoteRank),
                HCCL_E_INTERNAL);
        }
        CHK_PRT_RET(channel.handle == 0 || channel.notifyNum < CHANNEL_NOTIFY_NUM
                        || channel.remoteCclMem.addr == nullptr || channel.remoteCclMem.size < dataTypeSize,
            HCCL_ERROR("Invalid resource for remote rank[%u]", channel.remoteRank), HCCL_E_INTERNAL);
        bufferSize = std::min(bufferSize, channel.remoteCclMem.size);
    }
    for (size_t threadIndex = 1; threadIndex < resCtx.threads.size(); ++threadIndex) {
        CHK_PRT_RET(
            resCtx.threads[threadIndex] == 0, HCCL_ERROR("Invalid worker thread[%zu]", threadIndex), HCCL_E_INTERNAL);
        for (size_t previousIndex = 0; previousIndex < threadIndex; ++previousIndex) {
            CHK_PRT_RET(resCtx.threads[threadIndex] == resCtx.threads[previousIndex],
                HCCL_ERROR("Thread[%zu] unexpectedly reuses thread[%zu]", threadIndex, previousIndex), HCCL_E_INTERNAL);
        }
    }

    // 取所有rank通信Buffer的全局最小值，因此各rank会独立算出相同的分块参数。
    bufferSize = bufferSize / dataTypeSize * dataTypeSize;
    CHK_PRT_RET(bufferSize == 0, HCCL_ERROR("No usable CCL buffer space"), HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}

// 本rank自身分片的本地拷贝由主Thread与全部拷贝Thread均摊，与网络传输重叠。
HcclResult ScheduleSelfCopy(
    ThreadHandle thread, uint32_t partIndex, uint32_t partNum, uint8_t *selfOutput, uint8_t *input, uint64_t inputSize)
{
    uint64_t begin = inputSize * partIndex / partNum / PIPELINE_SLICE_ALIGNMENT * PIPELINE_SLICE_ALIGNMENT;
    uint64_t end = (partIndex + 1 == partNum)
                       ? inputSize
                       : inputSize * (partIndex + 1) / partNum / PIPELINE_SLICE_ALIGNMENT * PIPELINE_SLICE_ALIGNMENT;
    if (end > begin) {
        CHK_RET(HcommLocalCopyOnThread(thread, selfOutput + begin, input + begin, end - begin));
    }
    return HCCL_SUCCESS;
}

// 2x8拓扑的小消息路径：前三轮在本server内递归倍增，最后一轮与另一server的同位置rank交换。
// 每轮使用不同Channel，READY/READ_DONE各一打一收；全部任务位于主Thread，不创建小消息从流图。
HcclResult ExecRecursiveDoubling(const OpParam &param, const AlgResourceCtx &resCtx, uint64_t inputSize)
{
    ThreadHandle mainThread = resCtx.threads[0];
    uint8_t *input = static_cast<uint8_t *>(param.inputPtr);
    uint8_t *output = static_cast<uint8_t *>(param.outputPtr);
    uint8_t *localCclBuffer = static_cast<uint8_t *>(resCtx.localBuffer.addr);

    CHK_RET(HcommLocalCopyOnThread(
        mainThread, localCclBuffer + static_cast<uint64_t>(resCtx.localGroupIndex) * inputSize, input, inputSize));

    for (uint32_t round = 0; round + 1 < RECURSIVE_DOUBLING_ROUND_NUM; ++round) {
        uint32_t mask = 1U << round;
        uint32_t partnerBlockStart = (resCtx.localGroupIndex ^ mask) & ~(mask - 1U);
        uint64_t blockOffset = static_cast<uint64_t>(partnerBlockStart) * inputSize;
        uint64_t blockSize = static_cast<uint64_t>(mask) * inputSize;
        const ChannelInfo &channel = resCtx.channels[resCtx.recursiveDoublingChannelIndices[round]];

        CHK_RET(HcommChannelNotifyRecordOnThread(mainThread, channel.handle, CHANNEL_NOTIFY_CHUNK0_READY));
        CHK_RET(
            HcommChannelNotifyWaitOnThread(mainThread, channel.handle, CHANNEL_NOTIFY_CHUNK0_READY, CUSTOM_TIMEOUT));
        CHK_RET(HcommReadOnThread(mainThread, channel.handle, localCclBuffer + blockOffset,
            static_cast<uint8_t *>(channel.remoteCclMem.addr) + blockOffset, blockSize));
        CHK_RET(HcommChannelNotifyRecordOnThread(mainThread, channel.handle, CHANNEL_NOTIFY_CHUNK0_READ_DONE));
    }

    const ChannelInfo &crossServerChannel
        = resCtx.channels[resCtx.recursiveDoublingChannelIndices[RECURSIVE_DOUBLING_ROUND_NUM - 1]];
    uint64_t groupSize = static_cast<uint64_t>(OPTIMIZED_GROUP_SIZE) * inputSize;
    uint8_t *remoteGroupOutput = output + static_cast<uint64_t>(resCtx.remoteGroupBaseRank) * inputSize;
    CHK_RET(HcommChannelNotifyRecordOnThread(mainThread, crossServerChannel.handle, CHANNEL_NOTIFY_CHUNK0_READY));
    CHK_RET(HcommChannelNotifyWaitOnThread(
        mainThread, crossServerChannel.handle, CHANNEL_NOTIFY_CHUNK0_READY, CUSTOM_TIMEOUT));
    CHK_RET(HcommReadOnThread(
        mainThread, crossServerChannel.handle, remoteGroupOutput, crossServerChannel.remoteCclMem.addr, groupSize));
    CHK_RET(HcommChannelNotifyRecordOnThread(mainThread, crossServerChannel.handle, CHANNEL_NOTIFY_CHUNK0_READ_DONE));

    uint8_t *localGroupOutput = output + static_cast<uint64_t>(resCtx.localGroupBaseRank) * inputSize;
    CHK_RET(HcommLocalCopyOnThread(mainThread, localGroupOutput, localCclBuffer, groupSize));

    // 所有Channel只在本路径使用一次，但仍须在算子结束前排空READ_DONE，保护下一次调用复用CCL。
    for (uint32_t channelIndex : resCtx.recursiveDoublingChannelIndices) {
        const ChannelInfo &channel = resCtx.channels[channelIndex];
        CHK_RET(HcommChannelNotifyWaitOnThread(
            mainThread, channel.handle, CHANNEL_NOTIFY_CHUNK0_READ_DONE, CUSTOM_TIMEOUT));
    }
    return HCCL_SUCCESS;
}

// 2x8拓扑的大消息连续流水。rank(g, i)把自身输入按2MiB连续块Write到mate rank(1-g, i)
// 的CCL；该owner把每个chunk发布给其余14个rank。每条非mate链路只需一次连续Read，mate
// 链路只承载Write，因此15条物理链路各搬运完整S且staging可与读取重叠。
HcclResult ExecMateOwnerPipelined(const OpParam &param, const AlgResourceCtx &resCtx, uint64_t inputSize)
{
    ThreadHandle mainThread = resCtx.threads[0];
    uint8_t *input = static_cast<uint8_t *>(param.inputPtr);
    uint8_t *output = static_cast<uint8_t *>(param.outputPtr);
    uint8_t *localCclBuffer = static_cast<uint8_t *>(resCtx.localBuffer.addr);
    size_t workerNum = resCtx.channels.size();
    uint64_t chunkCount64 = (inputSize - 1) / MATE_PIPELINE_CHUNK_SIZE + 1;
    CHK_PRT_RET(chunkCount64 == 0 || chunkCount64 > MATE_PIPELINE_MAX_CHUNK_NUM,
        HCCL_ERROR("Invalid mate-pipeline chunk count[%llu]", static_cast<unsigned long long>(chunkCount64)),
        HCCL_E_INTERNAL);
    uint32_t chunkCount = static_cast<uint32_t>(chunkCount64);

    size_t proxyChannelIndex = workerNum;
    for (size_t channelIndex = 0; channelIndex < workerNum; ++channelIndex) {
        if (resCtx.channels[channelIndex].remoteRank == resCtx.proxyRank) {
            proxyChannelIndex = channelIndex;
            break;
        }
    }
    CHK_PRT_RET(proxyChannelIndex == workerNum, HCCL_ERROR("No channel was found for mate rank[%u]", resCtx.proxyRank),
        HCCL_E_INTERNAL);
    const ChannelInfo &proxyChannel = resCtx.channels[proxyChannelIndex];
    CHK_PRT_RET(inputSize > resCtx.localBuffer.size || inputSize > proxyChannel.remoteCclMem.size,
        HCCL_ERROR("Mate pipeline exceeds CCL buffer, inputSize[%llu]", static_cast<unsigned long long>(inputSize)),
        HCCL_E_INTERNAL);

    size_t readerCount = 0;
    for (size_t channelIndex = 0; channelIndex < workerNum; ++channelIndex) {
        if (channelIndex == proxyChannelIndex) {
            continue;
        }
        const ChannelInfo &channel = resCtx.channels[channelIndex];
        CHK_PRT_RET(channel.representedRank == param.myRank || channel.representedRank == resCtx.proxyRank,
            HCCL_ERROR("Invalid mate-pipeline represented rank[%u] for owner[%u]", channel.representedRank,
                channel.remoteRank),
            HCCL_E_INTERNAL);
        ++readerCount;
    }
    CHK_PRT_RET(readerCount != OPTIMIZED_RANK_NUM - 2,
        HCCL_ERROR("Invalid mate-pipeline reader count[%zu]", readerCount), HCCL_E_INTERNAL);

    // 全部15个Channel Worker和3个拷贝Thread都有显式main->slave启动边。
    for (size_t workerIndex = 0; workerIndex < workerNum; ++workerIndex) {
        CHK_RET(HcommThreadNotifyRecordOnThread(mainThread, resCtx.threads[workerIndex + 1], THREAD_NOTIFY_START));
    }
    for (uint32_t copierIndex = 0; copierIndex < COPIER_THREAD_NUM; ++copierIndex) {
        CHK_RET(HcommThreadNotifyRecordOnThread(
            mainThread, resCtx.threads[workerNum + 1 + copierIndex], THREAD_NOTIFY_START));
    }

    // main和copier2各拷self的一半；避免让15个网络Worker额外争用本地搬运资源。
    uint8_t *selfOutput = output + static_cast<uint64_t>(param.myRank) * inputSize;
    bool selfCopyRequired = selfOutput != input;
    if (selfCopyRequired) {
        CHK_RET(ScheduleSelfCopy(mainThread, 0, 2, selfOutput, input, inputSize));
    }

    // copier0的完成信号表示14个远端reader、本地owner拷贝以及mate RELEASE均已完成。
    // Channel任务不回到main；每条READY Channel固定由同一个copier发布，避免两个producer stream交替推进SQ。
    CHK_RET(HcommThreadNotifyWaitOnThread(mainThread, MAIN_NOTIFY_STAGED_BASE + 2 * workerNum, CUSTOM_TIMEOUT));
    for (size_t workerIndex = 0; workerIndex < workerNum; ++workerIndex) {
        CHK_RET(HcommThreadNotifyWaitOnThread(
            mainThread, MAIN_NOTIFY_STAGED_BASE + workerNum + static_cast<uint32_t>(workerIndex), CUSTOM_TIMEOUT));
    }
    CHK_RET(HcommThreadNotifyWaitOnThread(
        mainThread, MAIN_NOTIFY_STAGED_BASE + 2 * workerNum + COPIER_THREAD_NUM - 1, CUSTOM_TIMEOUT));

    // mate Worker只负责连续Write；其余14个Worker从对应owner的CCL连续Read到representedRank输出。
    for (size_t workerIndex = 0; workerIndex < workerNum; ++workerIndex) {
        const ChannelInfo &channel = resCtx.channels[workerIndex];
        ThreadHandle workerThread = resCtx.threads[workerIndex + 1];
        uint8_t *remoteCclBuffer = static_cast<uint8_t *>(channel.remoteCclMem.addr);
        CHK_RET(HcommThreadNotifyWaitOnThread(workerThread, THREAD_NOTIFY_START, CUSTOM_TIMEOUT));

        if (workerIndex == proxyChannelIndex) {
            for (uint32_t chunkIndex = 0; chunkIndex < chunkCount; ++chunkIndex) {
                uint64_t offset = static_cast<uint64_t>(chunkIndex) * MATE_PIPELINE_CHUNK_SIZE;
                uint64_t currentChunkSize = std::min(MATE_PIPELINE_CHUNK_SIZE, inputSize - offset);
                CHK_RET(HcommWriteOnThread(
                    workerThread, channel.handle, remoteCclBuffer + offset, input + offset, currentChunkSize));
                CHK_RET(HcommChannelNotifyRecordOnThread(workerThread, channel.handle, chunkIndex));
            }
            CHK_RET(HcommChannelNotifyWaitOnThread(
                workerThread, channel.handle, CHANNEL_NOTIFY_MATE_FINAL, CUSTOM_TIMEOUT));
        } else {
            uint8_t *representedOutput = output + static_cast<uint64_t>(channel.representedRank) * inputSize;
            for (uint32_t chunkIndex = 0; chunkIndex < chunkCount; ++chunkIndex) {
                uint64_t offset = static_cast<uint64_t>(chunkIndex) * MATE_PIPELINE_CHUNK_SIZE;
                uint64_t currentChunkSize = std::min(MATE_PIPELINE_CHUNK_SIZE, inputSize - offset);
                CHK_RET(HcommChannelNotifyWaitOnThread(workerThread, channel.handle, chunkIndex, CUSTOM_TIMEOUT));
                CHK_RET(HcommReadOnThread(workerThread, channel.handle, representedOutput + offset,
                    remoteCclBuffer + offset, currentChunkSize));
            }
            CHK_RET(HcommChannelNotifyRecordOnThread(workerThread, channel.handle, CHANNEL_NOTIFY_MATE_FINAL));
        }
        CHK_RET(HcommThreadNotifyRecordOnThread(
            workerThread, mainThread, MAIN_NOTIFY_STAGED_BASE + workerNum + static_cast<uint32_t>(workerIndex)));
    }

    // copier0逐chunk消费mate STAGE，并通过独占的本地Notify把同一chunk转发给copier1。
    // 14条非mate Channel固定7/7归两个copier，避免同一Channel的READY跨copier stream；
    // owner本地拷贝仍按偶/奇chunk均摊。每个chunk使用独立Notify，可安全重叠下发。
    uint8_t *ownerOutput = output + static_cast<uint64_t>(resCtx.proxyRank) * inputSize;
    for (uint32_t copierIndex = 0; copierIndex < 2; ++copierIndex) {
        ThreadHandle copierThread = resCtx.threads[workerNum + 1 + copierIndex];
        CHK_RET(HcommThreadNotifyWaitOnThread(copierThread, THREAD_NOTIFY_START, CUSTOM_TIMEOUT));
        for (uint32_t chunkIndex = 0; chunkIndex < chunkCount; ++chunkIndex) {
            uint64_t offset = static_cast<uint64_t>(chunkIndex) * MATE_PIPELINE_CHUNK_SIZE;
            uint64_t currentChunkSize = std::min(MATE_PIPELINE_CHUNK_SIZE, inputSize - offset);

            if (copierIndex == 0) {
                CHK_RET(HcommChannelNotifyWaitOnThread(copierThread, proxyChannel.handle, chunkIndex, CUSTOM_TIMEOUT));
                CHK_RET(HcommThreadNotifyRecordOnThread(
                    copierThread, resCtx.threads[workerNum + 2], THREAD_NOTIFY_MATE_CHUNK_BASE + chunkIndex));
            } else {
                CHK_RET(HcommThreadNotifyWaitOnThread(
                    copierThread, THREAD_NOTIFY_MATE_CHUNK_BASE + chunkIndex, CUSTOM_TIMEOUT));
            }

            size_t readerOrdinal = 0;
            for (size_t channelIndex = 0; channelIndex < workerNum; ++channelIndex) {
                if (channelIndex == proxyChannelIndex) {
                    continue;
                }
                if (readerOrdinal % 2 == copierIndex) {
                    CHK_RET(HcommChannelNotifyRecordOnThread(
                        copierThread, resCtx.channels[channelIndex].handle, chunkIndex));
                }
                ++readerOrdinal;
            }
            if (chunkIndex % 2 == copierIndex) {
                CHK_RET(HcommLocalCopyOnThread(
                    copierThread, ownerOutput + offset, localCclBuffer + offset, currentChunkSize));
            }
        }

        size_t readerOrdinal = 0;
        for (size_t channelIndex = 0; channelIndex < workerNum; ++channelIndex) {
            if (channelIndex == proxyChannelIndex) {
                continue;
            }
            if (readerOrdinal % 2 == copierIndex) {
                CHK_RET(HcommChannelNotifyWaitOnThread(
                    copierThread, resCtx.channels[channelIndex].handle, CHANNEL_NOTIFY_MATE_FINAL, CUSTOM_TIMEOUT));
            }
            ++readerOrdinal;
        }

        if (copierIndex == 0) {
            CHK_RET(HcommThreadNotifyWaitOnThread(copierThread, THREAD_NOTIFY_MATE_COPIER_DONE, CUSTOM_TIMEOUT));
            CHK_RET(HcommChannelNotifyRecordOnThread(copierThread, proxyChannel.handle, CHANNEL_NOTIFY_MATE_FINAL));
            CHK_RET(HcommThreadNotifyRecordOnThread(copierThread, mainThread, MAIN_NOTIFY_STAGED_BASE + 2 * workerNum));
        } else {
            CHK_RET(HcommThreadNotifyRecordOnThread(
                copierThread, resCtx.threads[workerNum + 1], THREAD_NOTIFY_MATE_COPIER_DONE));
        }
    }

    ThreadHandle selfCopyThread = resCtx.threads[workerNum + COPIER_THREAD_NUM];
    CHK_RET(HcommThreadNotifyWaitOnThread(selfCopyThread, THREAD_NOTIFY_START, CUSTOM_TIMEOUT));
    if (selfCopyRequired) {
        CHK_RET(ScheduleSelfCopy(selfCopyThread, 1, 2, selfOutput, input, inputSize));
    }
    CHK_RET(HcommThreadNotifyRecordOnThread(
        selfCopyThread, mainThread, MAIN_NOTIFY_STAGED_BASE + 2 * workerNum + COPIER_THREAD_NUM - 1));
    return HCCL_SUCCESS;
}

// 2x8拓扑的大消息空间条带路径。每个rank把输入切成8条，第j条经对应的layer-1 Channel
// 写到远端组owner-j的CCL；owner-j的CCL因而保存另一server全部8个source的第j条。
// 收齐后，7个layer-0 owner和8个layer-1 owner同时被拉取，15条物理链路各承载约S。
//
// 离散条带使用普通Read而不使用BatchTransfer：当前AICPU checker会把多描述符Batch
// 折叠为一段连续内存，无法正确表达本路径的strided最终输出。
HcclResult ExecSpatialStriped(const OpParam &param, const AlgResourceCtx &resCtx, uint64_t inputSize)
{
    ThreadHandle mainThread = resCtx.threads[0];
    uint8_t *input = static_cast<uint8_t *>(param.inputPtr);
    uint8_t *output = static_cast<uint8_t *>(param.outputPtr);
    uint8_t *localCclBuffer = static_cast<uint8_t *>(resCtx.localBuffer.addr);
    size_t workerNum = resCtx.channels.size();
    uint64_t stripeSize = inputSize / OPTIMIZED_GROUP_SIZE;

    CHK_PRT_RET(inputSize % OPTIMIZED_GROUP_SIZE != 0 || stripeSize == 0 || stripeSize % PIPELINE_SLICE_ALIGNMENT != 0,
        HCCL_ERROR("Input cannot be split into aligned spatial stripes, inputSize[%llu]",
            static_cast<unsigned long long>(inputSize)),
        HCCL_E_INTERNAL);

    size_t localChannelCount = 0;
    size_t crossChannelCount = 0;
    for (const ChannelInfo &channel : resCtx.channels) {
        bool ownerIsLocal = channel.remoteRank >= resCtx.localGroupBaseRank
                            && channel.remoteRank < resCtx.localGroupBaseRank + OPTIMIZED_GROUP_SIZE;
        bool ownerIsRemote = channel.remoteRank >= resCtx.remoteGroupBaseRank
                             && channel.remoteRank < resCtx.remoteGroupBaseRank + OPTIMIZED_GROUP_SIZE;
        CHK_PRT_RET(!ownerIsLocal && !ownerIsRemote,
            HCCL_ERROR("Spatial owner rank[%u] is outside both topology groups", channel.remoteRank), HCCL_E_INTERNAL);
        localChannelCount += ownerIsLocal ? 1 : 0;
        crossChannelCount += ownerIsRemote ? 1 : 0;
    }
    CHK_PRT_RET(localChannelCount != OPTIMIZED_GROUP_SIZE - 1 || crossChannelCount != OPTIMIZED_GROUP_SIZE,
        HCCL_ERROR("Invalid spatial channel split, local[%zu], cross[%zu]", localChannelCount, crossChannelCount),
        HCCL_E_INTERNAL);

    // 所有Worker都有显式main->worker启动边。copier1/2立即并行完成self输出；copier0需等
    // owner CCL收齐后再拷其中一半条带到远端组输出。
    for (size_t workerIndex = 0; workerIndex < workerNum; ++workerIndex) {
        CHK_RET(HcommThreadNotifyRecordOnThread(mainThread, resCtx.threads[workerIndex + 1], THREAD_NOTIFY_START));
    }
    uint8_t *selfOutput = output + static_cast<uint64_t>(param.myRank) * inputSize;
    bool selfCopyRequired = selfOutput != input;
    if (selfCopyRequired) {
        for (uint32_t copierIndex = 1; copierIndex < COPIER_THREAD_NUM; ++copierIndex) {
            CHK_RET(HcommThreadNotifyRecordOnThread(
                mainThread, resCtx.threads[workerNum + 1 + copierIndex], THREAD_NOTIFY_START));
        }
    }

    // 每个跨服Worker的STAGE Record严格排在Write之后；本端main直接等待8条跨服Channel，
    // 收齐时local CCL的[0, S)已由远端8个source以不相交区间填满。
    for (size_t workerIndex = 0; workerIndex < workerNum; ++workerIndex) {
        const ChannelInfo &channel = resCtx.channels[workerIndex];
        bool ownerIsRemote = channel.remoteRank >= resCtx.remoteGroupBaseRank
                             && channel.remoteRank < resCtx.remoteGroupBaseRank + OPTIMIZED_GROUP_SIZE;
        if (ownerIsRemote) {
            CHK_RET(HcommChannelNotifyWaitOnThread(
                mainThread, channel.handle, CHANNEL_NOTIFY_STRIPE_STAGE_READY, CUSTOM_TIMEOUT));
        }
    }
    for (const ChannelInfo &channel : resCtx.channels) {
        CHK_RET(HcommChannelNotifyRecordOnThread(mainThread, channel.handle, CHANNEL_NOTIFY_STRIPE_BUNDLE_READY));
    }

    ThreadHandle ownerCopyThread = resCtx.threads[workerNum + 1];
    CHK_RET(HcommThreadNotifyRecordOnThread(mainThread, ownerCopyThread, THREAD_NOTIFY_START));
    for (uint32_t sourceIndex = 0; sourceIndex < OPTIMIZED_GROUP_SIZE; sourceIndex += 2) {
        uint8_t *ownerOutput = output + static_cast<uint64_t>(resCtx.remoteGroupBaseRank + sourceIndex) * inputSize
                               + static_cast<uint64_t>(resCtx.localGroupIndex) * stripeSize;
        CHK_RET(HcommLocalCopyOnThread(
            mainThread, ownerOutput, localCclBuffer + static_cast<uint64_t>(sourceIndex) * stripeSize, stripeSize));
    }

    // DATA_DONE证明全部15个远端消费者已读完本端owner CCL；copier0完成则证明本地消费者
    // 也已读完。此后向8个跨服writer发RELEASE，允许它们结束并在下一次调用中复用本端CCL。
    for (const ChannelInfo &channel : resCtx.channels) {
        CHK_RET(HcommChannelNotifyWaitOnThread(
            mainThread, channel.handle, CHANNEL_NOTIFY_STRIPE_DATA_DONE, CUSTOM_TIMEOUT));
    }
    CHK_RET(HcommThreadNotifyWaitOnThread(mainThread, MAIN_NOTIFY_STAGED_BASE + 2 * workerNum, CUSTOM_TIMEOUT));
    if (selfCopyRequired) {
        for (uint32_t copierIndex = 1; copierIndex < COPIER_THREAD_NUM; ++copierIndex) {
            CHK_RET(HcommThreadNotifyWaitOnThread(
                mainThread, MAIN_NOTIFY_STAGED_BASE + 2 * workerNum + copierIndex, CUSTOM_TIMEOUT));
        }
    }
    for (const ChannelInfo &channel : resCtx.channels) {
        bool ownerIsRemote = channel.remoteRank >= resCtx.remoteGroupBaseRank
                             && channel.remoteRank < resCtx.remoteGroupBaseRank + OPTIMIZED_GROUP_SIZE;
        if (ownerIsRemote) {
            CHK_RET(HcommChannelNotifyRecordOnThread(mainThread, channel.handle, CHANNEL_NOTIFY_STRIPE_OWNER_RELEASE));
        }
    }
    for (size_t workerIndex = 0; workerIndex < workerNum; ++workerIndex) {
        CHK_RET(HcommThreadNotifyWaitOnThread(
            mainThread, MAIN_NOTIFY_STAGED_BASE + workerNum + static_cast<uint32_t>(workerIndex), CUSTOM_TIMEOUT));
    }

    for (size_t workerIndex = 0; workerIndex < workerNum; ++workerIndex) {
        const ChannelInfo &channel = resCtx.channels[workerIndex];
        ThreadHandle workerThread = resCtx.threads[workerIndex + 1];
        uint8_t *remoteCclBuffer = static_cast<uint8_t *>(channel.remoteCclMem.addr);
        bool ownerIsLocal = channel.remoteRank >= resCtx.localGroupBaseRank
                            && channel.remoteRank < resCtx.localGroupBaseRank + OPTIMIZED_GROUP_SIZE;

        CHK_RET(HcommThreadNotifyWaitOnThread(workerThread, THREAD_NOTIFY_START, CUSTOM_TIMEOUT));
        uint32_t ownerIndex = 0;
        if (ownerIsLocal) {
            ownerIndex = channel.remoteRank - resCtx.localGroupBaseRank;
        } else {
            ownerIndex = channel.remoteRank - resCtx.remoteGroupBaseRank;
            CHK_RET(HcommWriteOnThread(workerThread, channel.handle,
                remoteCclBuffer + static_cast<uint64_t>(resCtx.localGroupIndex) * stripeSize,
                input + static_cast<uint64_t>(ownerIndex) * stripeSize, stripeSize));
            CHK_RET(HcommChannelNotifyRecordOnThread(workerThread, channel.handle, CHANNEL_NOTIFY_STRIPE_STAGE_READY));
        }

        CHK_RET(HcommChannelNotifyWaitOnThread(
            workerThread, channel.handle, CHANNEL_NOTIFY_STRIPE_BUNDLE_READY, CUSTOM_TIMEOUT));
        for (uint32_t sourceIndex = 0; sourceIndex < OPTIMIZED_GROUP_SIZE; ++sourceIndex) {
            if (!ownerIsLocal && sourceIndex == resCtx.localGroupIndex) {
                continue;
            }
            uint32_t destinationRank
                = ownerIsLocal ? resCtx.remoteGroupBaseRank + sourceIndex : resCtx.localGroupBaseRank + sourceIndex;
            uint8_t *destination = output + static_cast<uint64_t>(destinationRank) * inputSize
                                   + static_cast<uint64_t>(ownerIndex) * stripeSize;
            CHK_RET(HcommReadOnThread(workerThread, channel.handle, destination,
                remoteCclBuffer + static_cast<uint64_t>(sourceIndex) * stripeSize, stripeSize));
        }
        CHK_RET(HcommChannelNotifyRecordOnThread(workerThread, channel.handle, CHANNEL_NOTIFY_STRIPE_DATA_DONE));
        if (!ownerIsLocal) {
            CHK_RET(HcommChannelNotifyWaitOnThread(
                workerThread, channel.handle, CHANNEL_NOTIFY_STRIPE_OWNER_RELEASE, CUSTOM_TIMEOUT));
        }
        CHK_RET(HcommThreadNotifyRecordOnThread(
            workerThread, mainThread, MAIN_NOTIFY_STAGED_BASE + workerNum + static_cast<uint32_t>(workerIndex)));
    }

    // copier0负责owner stripe的奇数source；copier1/2各负责self输出的一半。每个从Thread
    // 的第一个任务均为START Wait，最后一个任务均为独占main Notify的完成Record。
    CHK_RET(HcommThreadNotifyWaitOnThread(ownerCopyThread, THREAD_NOTIFY_START, CUSTOM_TIMEOUT));
    for (uint32_t sourceIndex = 1; sourceIndex < OPTIMIZED_GROUP_SIZE; sourceIndex += 2) {
        uint8_t *ownerOutput = output + static_cast<uint64_t>(resCtx.remoteGroupBaseRank + sourceIndex) * inputSize
                               + static_cast<uint64_t>(resCtx.localGroupIndex) * stripeSize;
        CHK_RET(HcommLocalCopyOnThread(ownerCopyThread, ownerOutput,
            localCclBuffer + static_cast<uint64_t>(sourceIndex) * stripeSize, stripeSize));
    }
    CHK_RET(HcommThreadNotifyRecordOnThread(ownerCopyThread, mainThread, MAIN_NOTIFY_STAGED_BASE + 2 * workerNum));

    if (selfCopyRequired) {
        for (uint32_t copierIndex = 1; copierIndex < COPIER_THREAD_NUM; ++copierIndex) {
            ThreadHandle copierThread = resCtx.threads[workerNum + 1 + copierIndex];
            CHK_RET(HcommThreadNotifyWaitOnThread(copierThread, THREAD_NOTIFY_START, CUSTOM_TIMEOUT));
            CHK_RET(
                ScheduleSelfCopy(copierThread, copierIndex - 1, COPIER_THREAD_NUM - 1, selfOutput, input, inputSize));
            CHK_RET(HcommThreadNotifyRecordOnThread(
                copierThread, mainThread, MAIN_NOTIFY_STAGED_BASE + 2 * workerNum + copierIndex));
        }
    }
    return HCCL_SUCCESS;
}

// 双来源大消息路径：把输入等分为A/B，本端CCL保存own-A和mate-B。A本地staging与B跨服
// staging并行；随后每个CCL owner q同时提供A(q)和B(mate(q))，保持layer-0 7S、layer-1 8S
// 的均衡流量，同时消除完整S跨服staging的硬前置屏障。
HcclResult ExecDualSourceSingleShot(const OpParam &param, const AlgResourceCtx &resCtx, uint64_t inputSize)
{
    ThreadHandle mainThread = resCtx.threads[0];
    uint8_t *input = static_cast<uint8_t *>(param.inputPtr);
    uint8_t *output = static_cast<uint8_t *>(param.outputPtr);
    uint8_t *localCclBuffer = static_cast<uint8_t *>(resCtx.localBuffer.addr);
    size_t workerNum = resCtx.channels.size();

    uint64_t firstPartSize = inputSize / 2 / PIPELINE_SLICE_ALIGNMENT * PIPELINE_SLICE_ALIGNMENT;
    CHK_PRT_RET(firstPartSize == 0 || firstPartSize >= inputSize,
        HCCL_ERROR("Failed to split dual-source input, inputSize[%llu]", static_cast<unsigned long long>(inputSize)),
        HCCL_E_INTERNAL);
    uint64_t secondPartSize = inputSize - firstPartSize;

    size_t proxyChannelIndex = workerNum;
    for (size_t channelIndex = 0; channelIndex < workerNum; ++channelIndex) {
        if (resCtx.channels[channelIndex].remoteRank == resCtx.proxyRank) {
            proxyChannelIndex = channelIndex;
            break;
        }
    }
    CHK_PRT_RET(proxyChannelIndex == workerNum, HCCL_ERROR("No channel was found for proxy rank[%u]", resCtx.proxyRank),
        HCCL_E_INTERNAL);
    const ChannelInfo &proxyChannel = resCtx.channels[proxyChannelIndex];

    // 只启动15个Worker和copier0；其余copier仅供通用流水路径使用。先启动A staging、mate
    // Worker及7个本server Worker，让layer-0的A传输尽早与layer-1的B Write重叠。
    ThreadHandle aStagingThread = resCtx.threads[workerNum + 1];
    CHK_RET(HcommThreadNotifyRecordOnThread(mainThread, aStagingThread, THREAD_NOTIFY_START));
    CHK_RET(HcommThreadNotifyRecordOnThread(mainThread, resCtx.threads[proxyChannelIndex + 1], THREAD_NOTIFY_START));
    for (size_t workerIndex = 0; workerIndex < workerNum; ++workerIndex) {
        const ChannelInfo &channel = resCtx.channels[workerIndex];
        bool ownerIsLocal = channel.remoteRank >= resCtx.localGroupBaseRank
                            && channel.remoteRank < resCtx.localGroupBaseRank + OPTIMIZED_GROUP_SIZE;
        if (workerIndex != proxyChannelIndex && ownerIsLocal) {
            CHK_RET(HcommThreadNotifyRecordOnThread(mainThread, resCtx.threads[workerIndex + 1], THREAD_NOTIFY_START));
        }
    }

    // 主Thread只负责B：Write和STAGE Record位于同一Thread，规避不可用的WithNotify接口。
    // 对端proxy Worker消费STAGE后向其本端main上报，main随后才能发布B_READY。
    CHK_RET(HcommWriteOnThread(mainThread, proxyChannel.handle,
        static_cast<uint8_t *>(proxyChannel.remoteCclMem.addr) + firstPartSize, input + firstPartSize, secondPartSize));
    CHK_RET(HcommChannelNotifyRecordOnThread(mainThread, proxyChannel.handle, CHANNEL_NOTIFY_DUAL_B_READY_OR_STAGE));
    // B Write之后再启动7个跨server非mate Worker；它们的第一段数据依赖B_READY，不损失并行度。
    for (size_t workerIndex = 0; workerIndex < workerNum; ++workerIndex) {
        const ChannelInfo &channel = resCtx.channels[workerIndex];
        bool ownerIsLocal = channel.remoteRank >= resCtx.localGroupBaseRank
                            && channel.remoteRank < resCtx.localGroupBaseRank + OPTIMIZED_GROUP_SIZE;
        if (workerIndex != proxyChannelIndex && !ownerIsLocal) {
            CHK_RET(HcommThreadNotifyRecordOnThread(mainThread, resCtx.threads[workerIndex + 1], THREAD_NOTIFY_START));
        }
    }
    CHK_RET(HcommThreadNotifyWaitOnThread(
        mainThread, MAIN_NOTIFY_STAGED_BASE + static_cast<uint32_t>(proxyChannelIndex), CUSTOM_TIMEOUT));
    for (size_t channelIndex = 0; channelIndex < workerNum; ++channelIndex) {
        if (channelIndex == proxyChannelIndex) {
            continue;
        }
        CHK_RET(HcommChannelNotifyRecordOnThread(
            mainThread, resCtx.channels[channelIndex].handle, CHANNEL_NOTIFY_DUAL_B_READY_OR_STAGE));
    }

    uint8_t *selfOutput = output + static_cast<uint64_t>(param.myRank) * inputSize;
    if (selfOutput != input) {
        CHK_RET(HcommLocalCopyOnThread(mainThread, selfOutput, input, inputSize));
    }

    // 每个peer只回一个合并DONE：普通peer已读完owner A和represented B；mate peer已读完A。
    // 本端mate-B拷贝由proxy Worker completion覆盖。
    for (const ChannelInfo &channel : resCtx.channels) {
        CHK_RET(
            HcommChannelNotifyWaitOnThread(mainThread, channel.handle, CHANNEL_NOTIFY_DUAL_PEER_DONE, CUSTOM_TIMEOUT));
    }
    for (size_t workerIndex = 0; workerIndex < workerNum; ++workerIndex) {
        CHK_RET(HcommThreadNotifyWaitOnThread(
            mainThread, MAIN_NOTIFY_STAGED_BASE + workerNum + static_cast<uint32_t>(workerIndex), CUSTOM_TIMEOUT));
    }
    CHK_RET(HcommThreadNotifyWaitOnThread(mainThread, MAIN_NOTIFY_STAGED_BASE + 2 * workerNum, CUSTOM_TIMEOUT));

    // 所有远端读取和本地mate-B拷贝完成后互换RELEASE，防止下一次调用提前覆盖B区。
    CHK_RET(HcommChannelNotifyRecordOnThread(mainThread, proxyChannel.handle, CHANNEL_NOTIFY_DUAL_RELEASE));
    CHK_RET(
        HcommChannelNotifyWaitOnThread(mainThread, proxyChannel.handle, CHANNEL_NOTIFY_DUAL_RELEASE, CUSTOM_TIMEOUT));

    for (size_t workerIndex = 0; workerIndex < workerNum; ++workerIndex) {
        const ChannelInfo &channel = resCtx.channels[workerIndex];
        ThreadHandle workerThread = resCtx.threads[workerIndex + 1];
        CHK_RET(HcommThreadNotifyWaitOnThread(workerThread, THREAD_NOTIFY_START, CUSTOM_TIMEOUT));

        if (workerIndex == proxyChannelIndex) {
            // mate Worker消费对端B_STAGE，确认本端CCL的B区可读后再上报main。
            CHK_RET(HcommChannelNotifyWaitOnThread(
                workerThread, channel.handle, CHANNEL_NOTIFY_DUAL_B_READY_OR_STAGE, CUSTOM_TIMEOUT));
            CHK_RET(HcommThreadNotifyRecordOnThread(
                workerThread, mainThread, MAIN_NOTIFY_STAGED_BASE + static_cast<uint32_t>(proxyChannelIndex)));

            uint8_t *proxyOutput = output + static_cast<uint64_t>(resCtx.proxyRank) * inputSize;
            CHK_RET(HcommChannelNotifyWaitOnThread(
                workerThread, channel.handle, CHANNEL_NOTIFY_DUAL_A_READY, CUSTOM_TIMEOUT));
            CHK_RET(
                HcommReadOnThread(workerThread, channel.handle, proxyOutput, channel.remoteCclMem.addr, firstPartSize));
            CHK_RET(HcommLocalCopyOnThread(
                workerThread, proxyOutput + firstPartSize, localCclBuffer + firstPartSize, secondPartSize));
        } else {
            uint8_t *ownerOutput = output + static_cast<uint64_t>(channel.remoteRank) * inputSize;
            uint8_t *representedOutput = output + static_cast<uint64_t>(channel.representedRank) * inputSize;
            uint8_t *remoteCclBuffer = static_cast<uint8_t *>(channel.remoteCclMem.addr);
            bool ownerIsLocal = channel.remoteRank >= resCtx.localGroupBaseRank
                                && channel.remoteRank < resCtx.localGroupBaseRank + OPTIMIZED_GROUP_SIZE;
            if (ownerIsLocal) {
                // 阶段1走layer-0传A，阶段2仍走layer-0传represented B。
                CHK_RET(HcommChannelNotifyWaitOnThread(
                    workerThread, channel.handle, CHANNEL_NOTIFY_DUAL_A_READY, CUSTOM_TIMEOUT));
                CHK_RET(HcommReadOnThread(workerThread, channel.handle, ownerOutput, remoteCclBuffer, firstPartSize));
                CHK_RET(HcommChannelNotifyWaitOnThread(
                    workerThread, channel.handle, CHANNEL_NOTIFY_DUAL_B_READY_OR_STAGE, CUSTOM_TIMEOUT));
                CHK_RET(HcommReadOnThread(workerThread, channel.handle, representedOutput + firstPartSize,
                    remoteCclBuffer + firstPartSize, secondPartSize));
            } else {
                // 与上面交错：阶段1走layer-1传represented B，阶段2走layer-1传A。
                CHK_RET(HcommChannelNotifyWaitOnThread(
                    workerThread, channel.handle, CHANNEL_NOTIFY_DUAL_B_READY_OR_STAGE, CUSTOM_TIMEOUT));
                CHK_RET(HcommReadOnThread(workerThread, channel.handle, representedOutput + firstPartSize,
                    remoteCclBuffer + firstPartSize, secondPartSize));
                CHK_RET(HcommChannelNotifyWaitOnThread(
                    workerThread, channel.handle, CHANNEL_NOTIFY_DUAL_A_READY, CUSTOM_TIMEOUT));
                CHK_RET(HcommReadOnThread(workerThread, channel.handle, ownerOutput, remoteCclBuffer, firstPartSize));
            }
        }
        CHK_RET(HcommChannelNotifyRecordOnThread(workerThread, channel.handle, CHANNEL_NOTIFY_DUAL_PEER_DONE));
        CHK_RET(HcommThreadNotifyRecordOnThread(
            workerThread, mainThread, MAIN_NOTIFY_STAGED_BASE + workerNum + static_cast<uint32_t>(workerIndex)));
    }

    // copier0与B跨服Write并行完成A staging，并直接发布A_READY，避免经过main形成串行屏障。
    CHK_RET(HcommThreadNotifyWaitOnThread(aStagingThread, THREAD_NOTIFY_START, CUSTOM_TIMEOUT));
    CHK_RET(HcommLocalCopyOnThread(aStagingThread, localCclBuffer, input, firstPartSize));
    for (const ChannelInfo &channel : resCtx.channels) {
        CHK_RET(HcommChannelNotifyRecordOnThread(aStagingThread, channel.handle, CHANNEL_NOTIFY_DUAL_A_READY));
    }
    CHK_RET(HcommThreadNotifyRecordOnThread(aStagingThread, mainThread, MAIN_NOTIFY_STAGED_BASE + 2 * workerNum));
    return HCCL_SUCCESS;
}

// 输入可一次性放入CCL Buffer时的极简路径：拷贝 -> READY -> 对端拉取 -> READ_DONE -> 完成，
// 不引入拷贝Thread；READ_DONE用于保护跨调用复用的CCL Buffer。
HcclResult ExecSingleShot(const OpParam &param, const AlgResourceCtx &resCtx, uint64_t inputSize)
{
    ThreadHandle mainThread = resCtx.threads[0];
    uint8_t *input = static_cast<uint8_t *>(param.inputPtr);
    uint8_t *output = static_cast<uint8_t *>(param.outputPtr);
    uint8_t *localCclBuffer = static_cast<uint8_t *>(resCtx.localBuffer.addr);
    size_t workerNum = resCtx.channels.size();

    // 主Thread：放行Worker -> 数据入CCL Buffer -> 通知全部对端 -> 拷出自身分片 -> 回收Worker。
    for (size_t workerIndex = 0; workerIndex < workerNum; ++workerIndex) {
        CHK_RET(HcommThreadNotifyRecordOnThread(mainThread, resCtx.threads[workerIndex + 1], THREAD_NOTIFY_START));
    }
    CHK_RET(HcommLocalCopyOnThread(mainThread, localCclBuffer, input, inputSize));
    for (const ChannelInfo &channel : resCtx.channels) {
        CHK_RET(HcommChannelNotifyRecordOnThread(mainThread, channel.handle, CHANNEL_NOTIFY_CHUNK0_READY));
    }
    uint8_t *selfOutput = output + static_cast<uint64_t>(param.myRank) * inputSize;
    if (selfOutput != input) {
        CHK_RET(HcommLocalCopyOnThread(mainThread, selfOutput, input, inputSize));
    }
    for (const ChannelInfo &channel : resCtx.channels) {
        CHK_RET(HcommChannelNotifyWaitOnThread(
            mainThread, channel.handle, CHANNEL_NOTIFY_CHUNK0_READ_DONE, CUSTOM_TIMEOUT));
    }
    for (size_t workerIndex = 0; workerIndex < workerNum; ++workerIndex) {
        CHK_RET(HcommThreadNotifyWaitOnThread(
            mainThread, MAIN_NOTIFY_STAGED_BASE + workerNum + static_cast<uint32_t>(workerIndex), CUSTOM_TIMEOUT));
    }

    // 每个Worker从对应对端拉取完整输入。一个对端始终只使用一条Channel。
    for (size_t workerIndex = 0; workerIndex < workerNum; ++workerIndex) {
        const ChannelInfo &channel = resCtx.channels[workerIndex];
        ThreadHandle workerThread = resCtx.threads[workerIndex + 1];
        uint8_t *remoteOutput = output + static_cast<uint64_t>(channel.remoteRank) * inputSize;
        CHK_RET(HcommThreadNotifyWaitOnThread(workerThread, THREAD_NOTIFY_START, CUSTOM_TIMEOUT));
        CHK_RET(
            HcommChannelNotifyWaitOnThread(workerThread, channel.handle, CHANNEL_NOTIFY_CHUNK0_READY, CUSTOM_TIMEOUT));
        CHK_RET(HcommReadOnThread(workerThread, channel.handle, remoteOutput, channel.remoteCclMem.addr, inputSize));
        CHK_RET(HcommChannelNotifyRecordOnThread(workerThread, channel.handle, CHANNEL_NOTIFY_CHUNK0_READ_DONE));
        CHK_RET(HcommThreadNotifyRecordOnThread(
            workerThread, mainThread, MAIN_NOTIFY_STAGED_BASE + workerNum + static_cast<uint32_t>(workerIndex)));
    }
    return HCCL_SUCCESS;
}

// 完整输入可放入CCL Buffer时的大消息路径：15个Worker分片并行staging，随后一轮并行直连拉取；
// 3个拷贝Thread与网络传输并行完成本rank输出，避免主Thread串行拷贝成为启动瓶颈。
HcclResult ExecParallelSingleShot(const OpParam &param, const AlgResourceCtx &resCtx, uint64_t inputSize)
{
    ThreadHandle mainThread = resCtx.threads[0];
    uint8_t *input = static_cast<uint8_t *>(param.inputPtr);
    uint8_t *output = static_cast<uint8_t *>(param.outputPtr);
    uint8_t *localCclBuffer = static_cast<uint8_t *>(resCtx.localBuffer.addr);
    uint8_t *selfOutput = output + static_cast<uint64_t>(param.myRank) * inputSize;
    size_t workerNum = resCtx.channels.size();
    uint32_t selfCopyPartNum = 1 + COPIER_THREAD_NUM;

    // 先只启动Worker，优先完成供所有对端读取的CCL Buffer staging。
    for (size_t workerIndex = 0; workerIndex < workerNum; ++workerIndex) {
        CHK_RET(HcommThreadNotifyRecordOnThread(mainThread, resCtx.threads[workerIndex + 1], THREAD_NOTIFY_START));
    }
    for (size_t workerIndex = 0; workerIndex < workerNum; ++workerIndex) {
        CHK_RET(HcommThreadNotifyWaitOnThread(
            mainThread, MAIN_NOTIFY_STAGED_BASE + static_cast<uint32_t>(workerIndex), CUSTOM_TIMEOUT));
    }

    // 本端数据完整后通知全部对端；随后启动self-copy，使其与15路网络读取重叠。
    for (const ChannelInfo &channel : resCtx.channels) {
        CHK_RET(HcommChannelNotifyRecordOnThread(mainThread, channel.handle, CHANNEL_NOTIFY_CHUNK0_READY));
    }
    for (uint32_t copierIndex = 0; copierIndex < COPIER_THREAD_NUM; ++copierIndex) {
        CHK_RET(HcommThreadNotifyRecordOnThread(
            mainThread, resCtx.threads[workerNum + 1 + copierIndex], THREAD_NOTIFY_START));
    }
    if (selfOutput != input) {
        CHK_RET(ScheduleSelfCopy(mainThread, 0, selfCopyPartNum, selfOutput, input, inputSize));
    }

    // READ_DONE保护整块CCL Buffer的跨调用复用，所有从Thread完成后才结束本轮。
    for (const ChannelInfo &channel : resCtx.channels) {
        CHK_RET(HcommChannelNotifyWaitOnThread(
            mainThread, channel.handle, CHANNEL_NOTIFY_CHUNK0_READ_DONE, CUSTOM_TIMEOUT));
    }
    for (size_t workerIndex = 0; workerIndex < workerNum; ++workerIndex) {
        CHK_RET(HcommThreadNotifyWaitOnThread(
            mainThread, MAIN_NOTIFY_STAGED_BASE + workerNum + static_cast<uint32_t>(workerIndex), CUSTOM_TIMEOUT));
    }
    for (uint32_t copierIndex = 0; copierIndex < COPIER_THREAD_NUM; ++copierIndex) {
        CHK_RET(HcommThreadNotifyWaitOnThread(
            mainThread, MAIN_NOTIFY_STAGED_BASE + 2 * workerNum + copierIndex, CUSTOM_TIMEOUT));
    }

    // 每个Worker先填充本端CCL Buffer的独占切片，再从对应对端直接拉取到最终输出。
    for (size_t workerIndex = 0; workerIndex < workerNum; ++workerIndex) {
        const ChannelInfo &channel = resCtx.channels[workerIndex];
        ThreadHandle workerThread = resCtx.threads[workerIndex + 1];
        uint8_t *remoteOutput = output + static_cast<uint64_t>(channel.remoteRank) * inputSize;
        CHK_RET(HcommThreadNotifyWaitOnThread(workerThread, THREAD_NOTIFY_START, CUSTOM_TIMEOUT));

        uint64_t stripBegin = inputSize * workerIndex / workerNum / PIPELINE_SLICE_ALIGNMENT * PIPELINE_SLICE_ALIGNMENT;
        uint64_t stripEnd = (workerIndex + 1 == workerNum) ? inputSize
                                                           : inputSize * (workerIndex + 1) / workerNum
                                                                 / PIPELINE_SLICE_ALIGNMENT * PIPELINE_SLICE_ALIGNMENT;
        if (stripEnd > stripBegin) {
            CHK_RET(HcommLocalCopyOnThread(
                workerThread, localCclBuffer + stripBegin, input + stripBegin, stripEnd - stripBegin));
        }
        CHK_RET(HcommThreadNotifyRecordOnThread(
            workerThread, mainThread, MAIN_NOTIFY_STAGED_BASE + static_cast<uint32_t>(workerIndex)));

        CHK_RET(
            HcommChannelNotifyWaitOnThread(workerThread, channel.handle, CHANNEL_NOTIFY_CHUNK0_READY, CUSTOM_TIMEOUT));
        CHK_RET(HcommReadOnThread(workerThread, channel.handle, remoteOutput, channel.remoteCclMem.addr, inputSize));
        CHK_RET(HcommChannelNotifyRecordOnThread(workerThread, channel.handle, CHANNEL_NOTIFY_CHUNK0_READ_DONE));
        CHK_RET(HcommThreadNotifyRecordOnThread(
            workerThread, mainThread, MAIN_NOTIFY_STAGED_BASE + workerNum + static_cast<uint32_t>(workerIndex)));
    }

    for (uint32_t copierIndex = 0; copierIndex < COPIER_THREAD_NUM; ++copierIndex) {
        ThreadHandle copierThread = resCtx.threads[workerNum + 1 + copierIndex];
        CHK_RET(HcommThreadNotifyWaitOnThread(copierThread, THREAD_NOTIFY_START, CUSTOM_TIMEOUT));
        if (selfOutput != input) {
            CHK_RET(ScheduleSelfCopy(copierThread, 1 + copierIndex, selfCopyPartNum, selfOutput, input, inputSize));
        }
        CHK_RET(HcommThreadNotifyRecordOnThread(
            copierThread, mainThread, MAIN_NOTIFY_STAGED_BASE + 2 * workerNum + copierIndex));
    }
    return HCCL_SUCCESS;
}

// 大输入流水路径：首个chunk由全体Worker并行切片填充（启动延迟约为单线程拷贝的1/workerNum），
// 后续chunk由3个拷贝Thread各独占一个slot流水供给，Worker持续拉取，拷贝与传输全重叠。
HcclResult ExecPipelined(const OpParam &param, const AlgResourceCtx &resCtx, uint64_t inputSize, uint64_t sliceSize)
{
    ThreadHandle mainThread = resCtx.threads[0];
    uint8_t *input = static_cast<uint8_t *>(param.inputPtr);
    uint8_t *output = static_cast<uint8_t *>(param.outputPtr);
    uint8_t *localCclBuffer = static_cast<uint8_t *>(resCtx.localBuffer.addr);
    uint8_t *selfOutput = output + static_cast<uint64_t>(param.myRank) * inputSize;
    size_t workerNum = resCtx.channels.size();
    uint32_t selfCopyPartNum = 1 + COPIER_THREAD_NUM;

    uint64_t chunkCount = (inputSize - 1) / sliceSize + 1; // >= 2
    uint64_t chunk0Size = sliceSize;
    uint8_t *chunk0Local = localCclBuffer + static_cast<uint64_t>(PIPELINE_SLOT_NUM) * sliceSize;

    // ==============================================
    // 主Thread：放行全部从Thread，聚合首chunk后通知对端，再均摊自身分片拷贝并回收从Thread。
    // ==============================================
    for (size_t threadIndex = 1; threadIndex < resCtx.threads.size(); ++threadIndex) {
        CHK_RET(HcommThreadNotifyRecordOnThread(mainThread, resCtx.threads[threadIndex], THREAD_NOTIFY_START));
    }
    for (size_t workerIndex = 0; workerIndex < workerNum; ++workerIndex) {
        CHK_RET(HcommThreadNotifyWaitOnThread(
            mainThread, MAIN_NOTIFY_STAGED_BASE + static_cast<uint32_t>(workerIndex), CUSTOM_TIMEOUT));
    }
    for (const ChannelInfo &channel : resCtx.channels) {
        CHK_RET(HcommChannelNotifyRecordOnThread(mainThread, channel.handle, CHANNEL_NOTIFY_CHUNK0_READY));
    }
    if (selfOutput != input) {
        CHK_RET(ScheduleSelfCopy(mainThread, 0, selfCopyPartNum, selfOutput, input, inputSize));
    }
    for (const ChannelInfo &channel : resCtx.channels) {
        CHK_RET(HcommChannelNotifyWaitOnThread(
            mainThread, channel.handle, CHANNEL_NOTIFY_CHUNK0_READ_DONE, CUSTOM_TIMEOUT));
    }
    for (size_t workerIndex = 0; workerIndex < workerNum; ++workerIndex) {
        CHK_RET(HcommThreadNotifyWaitOnThread(
            mainThread, MAIN_NOTIFY_STAGED_BASE + workerNum + static_cast<uint32_t>(workerIndex), CUSTOM_TIMEOUT));
    }
    for (uint32_t copierIndex = 0; copierIndex < COPIER_THREAD_NUM; ++copierIndex) {
        CHK_RET(HcommThreadNotifyWaitOnThread(
            mainThread, MAIN_NOTIFY_STAGED_BASE + 2 * workerNum + copierIndex, CUSTOM_TIMEOUT));
    }

    // ==============================================
    // Worker：并行填充首chunk切片，随后逐chunk拉取对端数据，读完即回READ_DONE。
    // ==============================================
    for (size_t workerIndex = 0; workerIndex < workerNum; ++workerIndex) {
        const ChannelInfo &channel = resCtx.channels[workerIndex];
        ThreadHandle workerThread = resCtx.threads[workerIndex + 1];
        uint8_t *remoteCclBuffer = static_cast<uint8_t *>(channel.remoteCclMem.addr);
        uint8_t *remoteOutput = output + static_cast<uint64_t>(channel.remoteRank) * inputSize;
        CHK_RET(HcommThreadNotifyWaitOnThread(workerThread, THREAD_NOTIFY_START, CUSTOM_TIMEOUT));

        // 首chunk切片（4KB对齐）：[w * size / workerNum, (w + 1) * size / workerNum)
        uint64_t stripBegin
            = chunk0Size * workerIndex / workerNum / PIPELINE_SLICE_ALIGNMENT * PIPELINE_SLICE_ALIGNMENT;
        uint64_t stripEnd = (workerIndex + 1 == workerNum) ? chunk0Size
                                                           : chunk0Size * (workerIndex + 1) / workerNum
                                                                 / PIPELINE_SLICE_ALIGNMENT * PIPELINE_SLICE_ALIGNMENT;
        if (stripEnd > stripBegin) {
            CHK_RET(HcommLocalCopyOnThread(
                workerThread, chunk0Local + stripBegin, input + stripBegin, stripEnd - stripBegin));
        }
        CHK_RET(HcommThreadNotifyRecordOnThread(
            workerThread, mainThread, MAIN_NOTIFY_STAGED_BASE + static_cast<uint32_t>(workerIndex)));

        CHK_RET(
            HcommChannelNotifyWaitOnThread(workerThread, channel.handle, CHANNEL_NOTIFY_CHUNK0_READY, CUSTOM_TIMEOUT));
        CHK_RET(HcommReadOnThread(workerThread, channel.handle, remoteOutput,
            remoteCclBuffer + static_cast<uint64_t>(PIPELINE_SLOT_NUM) * sliceSize, chunk0Size));
        CHK_RET(HcommChannelNotifyRecordOnThread(workerThread, channel.handle, CHANNEL_NOTIFY_CHUNK0_READ_DONE));

        for (uint64_t chunkIndex = 1; chunkIndex < chunkCount; ++chunkIndex) {
            uint32_t slot = static_cast<uint32_t>((chunkIndex - 1) % PIPELINE_SLOT_NUM);
            uint64_t offset = chunkIndex * sliceSize;
            uint64_t currentChunkSize = std::min(sliceSize, inputSize - offset);
            CHK_RET(HcommChannelNotifyWaitOnThread(
                workerThread, channel.handle, CHANNEL_NOTIFY_READY_BASE + slot, CUSTOM_TIMEOUT));
            CHK_RET(HcommReadOnThread(workerThread, channel.handle, remoteOutput + offset,
                remoteCclBuffer + static_cast<uint64_t>(slot) * sliceSize, currentChunkSize));
            CHK_RET(
                HcommChannelNotifyRecordOnThread(workerThread, channel.handle, CHANNEL_NOTIFY_READ_DONE_BASE + slot));
        }
        CHK_RET(HcommThreadNotifyRecordOnThread(
            workerThread, mainThread, MAIN_NOTIFY_STAGED_BASE + workerNum + static_cast<uint32_t>(workerIndex)));
    }

    // ==============================================
    // 拷贝Thread k：独占slot k，流水供给chunk 1+k, 4+k, ...；复用slot前先收齐上轮READ_DONE。
    // ==============================================
    for (uint32_t copierIndex = 0; copierIndex < COPIER_THREAD_NUM; ++copierIndex) {
        ThreadHandle copierThread = resCtx.threads[workerNum + 1 + copierIndex];
        CHK_RET(HcommThreadNotifyWaitOnThread(copierThread, THREAD_NOTIFY_START, CUSTOM_TIMEOUT));
        bool stagedAny = false;
        for (uint64_t chunkIndex = 1 + copierIndex; chunkIndex < chunkCount; chunkIndex += PIPELINE_SLOT_NUM) {
            if (stagedAny) {
                for (const ChannelInfo &channel : resCtx.channels) {
                    CHK_RET(HcommChannelNotifyWaitOnThread(
                        copierThread, channel.handle, CHANNEL_NOTIFY_READ_DONE_BASE + copierIndex, CUSTOM_TIMEOUT));
                }
            }
            uint64_t offset = chunkIndex * sliceSize;
            uint64_t currentChunkSize = std::min(sliceSize, inputSize - offset);
            CHK_RET(HcommLocalCopyOnThread(copierThread,
                localCclBuffer + static_cast<uint64_t>(copierIndex) * sliceSize, input + offset, currentChunkSize));
            for (const ChannelInfo &channel : resCtx.channels) {
                CHK_RET(HcommChannelNotifyRecordOnThread(
                    copierThread, channel.handle, CHANNEL_NOTIFY_READY_BASE + copierIndex));
            }
            stagedAny = true;
        }
        if (selfOutput != input) {
            CHK_RET(ScheduleSelfCopy(copierThread, 1 + copierIndex, selfCopyPartNum, selfOutput, input, inputSize));
        }
        // 排空最后一个chunk的READ_DONE信用，保证所有READY/READ_DONE Notify成对消耗。
        if (stagedAny) {
            for (const ChannelInfo &channel : resCtx.channels) {
                CHK_RET(HcommChannelNotifyWaitOnThread(
                    copierThread, channel.handle, CHANNEL_NOTIFY_READ_DONE_BASE + copierIndex, CUSTOM_TIMEOUT));
            }
        }
        CHK_RET(HcommThreadNotifyRecordOnThread(
            copierThread, mainThread, MAIN_NOTIFY_STAGED_BASE + 2 * workerNum + copierIndex));
    }
    return HCCL_SUCCESS;
}
// 注册输出直推可用性守卫：本端recvBuf注册地址与本次调用一致、全部对端recvBuf地址已交换
// 且容量足够。任一不满足则回退CCL staging路径，保证16个rank进入同一分支。
bool CanUseRegisteredOutputPush(const OpParam &param, const AlgResourceCtx &resCtx, uint64_t inputSize)
{
    if (resCtx.registeredOutput.addr == nullptr || resCtx.registeredOutput.addr != param.outputPtr) {
        return false;
    }
    uint64_t outputSize = inputSize * static_cast<uint64_t>(param.rankSize);
    if (resCtx.registeredOutput.size < outputSize || resCtx.remoteOutputs.size() != resCtx.channels.size()) {
        return false;
    }
    uint64_t selfEnd = static_cast<uint64_t>(param.myRank + 1) * inputSize;
    for (const CommBuffer &remoteOutput : resCtx.remoteOutputs) {
        if (remoteOutput.addr == nullptr || remoteOutput.size < selfEnd) {
            return false;
        }
    }
    return true;
}

// 注册输出直推大消息路径：每个Worker把完整输入S经独立Channel直接Write到对端已注册的
// recvBuf的[myRank*S, +S)区间，15条物理链路全双工各搬运恰好S（AllGather流量下限）。
// 零CCL staging、零中转、零流水门控；Write为即发即忘语义，单链路吞吐显著高于同条件Read。
// 每rank的自分片由main与3个copier四等分并行拷贝，与网络写重叠。
HcclResult ExecRegisteredOutputPush(const OpParam &param, const AlgResourceCtx &resCtx, uint64_t inputSize)
{
    ThreadHandle mainThread = resCtx.threads[0];
    uint8_t *input = static_cast<uint8_t *>(param.inputPtr);
    uint8_t *output = static_cast<uint8_t *>(param.outputPtr);
    size_t workerNum = resCtx.channels.size();
    uint64_t selfOffset = static_cast<uint64_t>(param.myRank) * inputSize;

    // 全部Worker与拷贝Thread都有显式main->slave启动边。
    for (size_t workerIndex = 0; workerIndex < workerNum; ++workerIndex) {
        CHK_RET(HcommThreadNotifyRecordOnThread(mainThread, resCtx.threads[workerIndex + 1], THREAD_NOTIFY_START));
    }
    for (uint32_t copierIndex = 0; copierIndex < COPIER_THREAD_NUM; ++copierIndex) {
        CHK_RET(HcommThreadNotifyRecordOnThread(
            mainThread, resCtx.threads[workerNum + 1 + copierIndex], THREAD_NOTIFY_START));
    }

    // self分片由main与copier四等分；对端只写[myRank*S)之外的区间，与本地拷贝无地址冲突。
    uint8_t *selfOutput = output + selfOffset;
    bool selfCopyRequired = selfOutput != input;
    if (selfCopyRequired) {
        CHK_RET(ScheduleSelfCopy(mainThread, 0, 1 + COPIER_THREAD_NUM, selfOutput, input, inputSize));
    }

    // 收齐15路READY表示本端recvBuf已收齐全部对端数据；随即发布RELEASE允许对端Worker结束，
    // 并保证下一次调用不会先于本端完成而覆盖本端recvBuf（READY/RELEASE均一打一收）。
    for (const ChannelInfo &channel : resCtx.channels) {
        CHK_RET(
            HcommChannelNotifyWaitOnThread(mainThread, channel.handle, CHANNEL_NOTIFY_PUSH_READY, CUSTOM_TIMEOUT));
    }
    for (const ChannelInfo &channel : resCtx.channels) {
        CHK_RET(HcommChannelNotifyRecordOnThread(mainThread, channel.handle, CHANNEL_NOTIFY_PUSH_RELEASE));
    }
    for (size_t workerIndex = 0; workerIndex < workerNum; ++workerIndex) {
        CHK_RET(HcommThreadNotifyWaitOnThread(
            mainThread, MAIN_NOTIFY_STAGED_BASE + workerNum + static_cast<uint32_t>(workerIndex), CUSTOM_TIMEOUT));
    }
    for (uint32_t copierIndex = 0; copierIndex < COPIER_THREAD_NUM; ++copierIndex) {
        CHK_RET(HcommThreadNotifyWaitOnThread(
            mainThread, MAIN_NOTIFY_STAGED_BASE + 2 * workerNum + copierIndex, CUSTOM_TIMEOUT));
    }

    // 每个Worker独占一条Channel：Write后沿同Channel发READY（流内保序，READY触发即数据
    // 已落对端recvBuf）；Wait RELEASE排空本轮信用后再上报完成。
    for (size_t workerIndex = 0; workerIndex < workerNum; ++workerIndex) {
        const ChannelInfo &channel = resCtx.channels[workerIndex];
        ThreadHandle workerThread = resCtx.threads[workerIndex + 1];
        CHK_RET(HcommThreadNotifyWaitOnThread(workerThread, THREAD_NOTIFY_START, CUSTOM_TIMEOUT));
        CHK_RET(HcommWriteOnThread(workerThread, channel.handle,
            static_cast<uint8_t *>(resCtx.remoteOutputs[workerIndex].addr) + selfOffset, input, inputSize));
        CHK_RET(HcommChannelNotifyRecordOnThread(workerThread, channel.handle, CHANNEL_NOTIFY_PUSH_READY));
        CHK_RET(
            HcommChannelNotifyWaitOnThread(workerThread, channel.handle, CHANNEL_NOTIFY_PUSH_RELEASE, CUSTOM_TIMEOUT));
        CHK_RET(HcommThreadNotifyRecordOnThread(
            workerThread, mainThread, MAIN_NOTIFY_STAGED_BASE + workerNum + static_cast<uint32_t>(workerIndex)));
    }

    for (uint32_t copierIndex = 0; copierIndex < COPIER_THREAD_NUM; ++copierIndex) {
        ThreadHandle copierThread = resCtx.threads[workerNum + 1 + copierIndex];
        CHK_RET(HcommThreadNotifyWaitOnThread(copierThread, THREAD_NOTIFY_START, CUSTOM_TIMEOUT));
        if (selfCopyRequired) {
            CHK_RET(ScheduleSelfCopy(copierThread, 1 + copierIndex, 1 + COPIER_THREAD_NUM, selfOutput, input,
                inputSize));
        }
        CHK_RET(HcommThreadNotifyRecordOnThread(
            copierThread, mainThread, MAIN_NOTIFY_STAGED_BASE + 2 * workerNum + copierIndex));
    }
    return HCCL_SUCCESS;
}
} // namespace

namespace ops_hccl {
HcclResult ExecOp(const OpParam &param, const AlgResourceCtx &resCtx)
{
    CHK_PTR_NULL(param.inputPtr);
    CHK_PTR_NULL(param.outputPtr);

    uint64_t dataTypeSize = 0;
    CHK_RET(GetAllGatherDataTypeSize(param.dataType, dataTypeSize));
    CHK_PRT_RET(param.count > std::numeric_limits<uint64_t>::max() / dataTypeSize, HCCL_ERROR("Input size overflows"),
        HCCL_E_PARA);
    uint64_t inputSize = param.count * dataTypeSize;
    CHK_PRT_RET(inputSize != 0 && param.rankSize > std::numeric_limits<uint64_t>::max() / inputSize,
        HCCL_ERROR("Output size overflows"), HCCL_E_PARA);

    HCCL_INFO("Executing AllGather on rank[%u]", param.myRank);

    uint64_t bufferSize = 0;
    CHK_RET(ValidateAndGetBufferSize(param, resCtx, dataTypeSize, bufferSize));
    ThreadHandle mainThread = resCtx.threads[0];
    uint8_t *input = static_cast<uint8_t *>(param.inputPtr);
    uint8_t *output = static_cast<uint8_t *>(param.outputPtr);

    if (inputSize == 0) {
        return HCCL_SUCCESS;
    }

    if (param.rankSize == 1) {
        if (output != input) {
            CHK_RET(HcommLocalCopyOnThread(mainThread, output, input, inputSize));
        }
        return HCCL_SUCCESS;
    }

    if (inputSize <= SERIAL_SINGLE_SHOT_MAX_SIZE) {
        if (resCtx.topologyPlanValid != 0 && inputSize <= bufferSize / OPTIMIZED_GROUP_SIZE) {
            return ExecRecursiveDoubling(param, resCtx, inputSize);
        }
        return ExecSingleShot(param, resCtx, inputSize);
    }
    // 大消息优先走注册输出直推；注册不可用或recvBuf地址变化时自动回退CCL staging路径。
    if (CanUseRegisteredOutputPush(param, resCtx, inputSize)) {
        return ExecRegisteredOutputPush(param, resCtx, inputSize);
    }
    if (resCtx.topologyPlanValid != 0 && inputSize <= bufferSize) {
        uint64_t mateChunkCount = (inputSize - 1) / MATE_PIPELINE_CHUNK_SIZE + 1;
        if (mateChunkCount <= MATE_PIPELINE_MAX_CHUNK_NUM) {
            return ExecMateOwnerPipelined(param, resCtx, inputSize);
        }
        bool canUseSpatialStripes = inputSize % OPTIMIZED_GROUP_SIZE == 0
                                    && inputSize / OPTIMIZED_GROUP_SIZE >= PIPELINE_SLICE_ALIGNMENT
                                    && inputSize / OPTIMIZED_GROUP_SIZE % PIPELINE_SLICE_ALIGNMENT == 0;
        if (canUseSpatialStripes) {
            return ExecSpatialStriped(param, resCtx, inputSize);
        }
        return ExecDualSourceSingleShot(param, resCtx, inputSize);
    }
    if (inputSize <= bufferSize) {
        return ExecParallelSingleShot(param, resCtx, inputSize);
    }

    uint64_t sliceSize = bufferSize / PIPELINE_REGION_NUM;
    sliceSize = sliceSize / PIPELINE_SLICE_ALIGNMENT * PIPELINE_SLICE_ALIGNMENT;
    CHK_PRT_RET(sliceSize == 0, HCCL_ERROR("Failed to calculate pipeline slice size"), HCCL_E_INTERNAL);
    return ExecPipelined(param, resCtx, inputSize, sliceSize);
}
} // namespace ops_hccl
