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

constexpr uint32_t OUTPUT_XN_ID = 1;
constexpr uint32_t TOKEN_XN_ID = 2;
constexpr uint32_t CKE_INDEX = 0;
constexpr uint32_t POST_SYNC_ID = 3;
constexpr uint32_t PIPELINE_STEPS = 4;
constexpr uint32_t FINE_PIPELINE_STEPS = 8;
constexpr uint32_t DEEP_PIPELINE_STEPS = 12;

#define CCU_CHECK(call)           \
    do {                          \
        CcuResult ret = (call);   \
        if (ret != CCU_SUCCESS) { \
            return ret;           \
        }                         \
    } while (0)

CcuResult CcuBroadcastKernel(CcuKernelArg arg)
{
    auto *kernelArg =
        static_cast<CcuKernelArgBroadcast *>(arg);

    if (kernelArg == nullptr ||
        kernelArg->rankSize <= 1 ||
        kernelArg->root >= kernelArg->rankSize ||
        kernelArg->rankId >= kernelArg->rankSize ||
        kernelArg->channelCount == 0 ||
        kernelArg->channelCount >= MAX_RANK_SIZE) {
        return CCU_E_PARA;
    }

    ccu::Variable localOutput;
    ccu::Variable localToken;
    ccu::Variable sliceOffset;
    ccu::Variable sliceSize;

    std::vector<ccu::Variable>
        remoteOutput(kernelArg->channelCount);
    std::vector<ccu::Variable>
        remoteToken(kernelArg->channelCount);

    if (kernelArg->treeEnabled) {
        if (kernelArg->channelCount + 1 != kernelArg->rankSize ||
            kernelArg->rankSize < 12) {
            return CCU_E_PARA;
        }

        uint32_t argIndex = 0;
        CCU_CHECK(ccu::LoadArg(localOutput, argIndex++));
        CCU_CHECK(ccu::LoadArg(localToken, argIndex++));
        CCU_CHECK(ccu::LoadArg(sliceOffset, argIndex++));
        CCU_CHECK(ccu::LoadArg(sliceSize, argIndex++));

        const uint32_t seedCount =
            kernelArg->rankSize == MAX_RANK_SIZE ? 4 : 3;
        std::vector<uint32_t> otherRanks;
        for (uint32_t rank = 0; rank < kernelArg->rankSize; ++rank) {
            if (rank != kernelArg->root) {
                otherRanks.push_back(rank);
            }
        }

        uint32_t parentRank = kernelArg->rankSize;
        std::vector<uint32_t> childRanks;
        if (kernelArg->rankId == kernelArg->root) {
            for (uint32_t i = 0; i < seedCount; ++i) {
                childRanks.push_back(otherRanks[i]);
            }
        } else {
            uint32_t rankIndex = 0;
            for (uint32_t i = 0; i < otherRanks.size(); ++i) {
                if (otherRanks[i] == kernelArg->rankId) {
                    rankIndex = i;
                    break;
                }
            }

            if (rankIndex < seedCount) {
                parentRank = kernelArg->root;
                for (uint32_t i = seedCount + rankIndex;
                     i < otherRanks.size();
                     i += seedCount) {
                    childRanks.push_back(otherRanks[i]);
                }
            } else {
                parentRank = otherRanks[
                    (rankIndex - seedCount) % seedCount];
            }
        }

        uint32_t parentChannel = kernelArg->channelCount;
        if (parentRank < kernelArg->rankSize) {
            for (uint32_t i = 0; i < kernelArg->channelCount; ++i) {
                if (kernelArg->remoteRanks[i] == parentRank) {
                    parentChannel = i;
                    break;
                }
            }
            if (parentChannel == kernelArg->channelCount) {
                return CCU_E_PARA;
            }
        }

        std::vector<uint32_t> childChannels;
        for (const uint32_t childRank : childRanks) {
            uint32_t childChannel = kernelArg->channelCount;
            for (uint32_t i = 0; i < kernelArg->channelCount; ++i) {
                if (kernelArg->remoteRanks[i] == childRank) {
                    childChannel = i;
                    break;
                }
            }
            if (childChannel == kernelArg->channelCount) {
                return CCU_E_PARA;
            }

            childChannels.push_back(childChannel);
            remoteOutput[childChannel] =
                ccu::GetResByChannel<ccu::Variable>(
                    kernelArg->channels[childChannel],
                    OUTPUT_XN_ID);
            remoteToken[childChannel] =
                ccu::GetResByChannel<ccu::Variable>(
                    kernelArg->channels[childChannel],
                    TOKEN_XN_ID);
        }

        if (parentChannel != kernelArg->channelCount) {
            CCU_CHECK(ccu::WriteVariableWithNotify(
                kernelArg->channels[parentChannel],
                localOutput,
                OUTPUT_XN_ID,
                CKE_INDEX,
                1U << OUTPUT_XN_ID));
            CCU_CHECK(ccu::WriteVariableWithNotify(
                kernelArg->channels[parentChannel],
                localToken,
                TOKEN_XN_ID,
                CKE_INDEX,
                1U << TOKEN_XN_ID));
        }

        constexpr uint32_t addressReadyMask =
            (1U << OUTPUT_XN_ID) |
            (1U << TOKEN_XN_ID);
        for (const uint32_t childChannel : childChannels) {
            CCU_CHECK(ccu::NotifyWait(
                kernelArg->channels[childChannel],
                CKE_INDEX,
                addressReadyMask));
        }

        if (parentChannel != kernelArg->channelCount) {
            CCU_CHECK(ccu::NotifyWait(
                kernelArg->channels[parentChannel],
                CKE_INDEX,
                1U << POST_SYNC_ID));
        }

        if (!childChannels.empty()) {
            ccu::LocalAddr source;
            source.addr = localOutput;
            source.addr += sliceOffset;
            source.token = localToken;

            ccu::Event writeDone;
            uint16_t writeMask = 0;
            for (const uint32_t childChannel : childChannels) {
                ccu::RemoteAddr destination;
                destination.addr = remoteOutput[childChannel];
                destination.addr += sliceOffset;
                destination.token = remoteToken[childChannel];

                const uint16_t eventMask =
                    static_cast<uint16_t>(1U << childChannel);
                writeMask =
                    static_cast<uint16_t>(writeMask | eventMask);
                CCU_CHECK(ccu::Write(
                    kernelArg->channels[childChannel],
                    destination,
                    source,
                    sliceSize,
                    writeDone,
                    eventMask));
            }

            CCU_CHECK(ccu::EventWait(writeDone, writeMask));
            for (const uint32_t childChannel : childChannels) {
                CCU_CHECK(ccu::NotifyRecord(
                    kernelArg->channels[childChannel],
                    CKE_INDEX,
                    1U << POST_SYNC_ID));
            }

        }

        /*
         * 每个rank只需保证自己的接收完成，并在作为父节点时保证对子节点的
         * Write已经EventWait完成。无需再从叶子向root反向确认；该反向链既
         * 不影响本地完成语义，又会增加小数据关键路径和循环等待风险。
         */
        return CCU_SUCCESS;
    }

    if (kernelArg->rankId == kernelArg->root) {
        for (uint32_t i = 0; i < kernelArg->channelCount; ++i) {
            remoteOutput[i] =
                ccu::GetResByChannel<ccu::Variable>(
                    kernelArg->channels[i],
                    OUTPUT_XN_ID);

            remoteToken[i] =
                ccu::GetResByChannel<ccu::Variable>(
                    kernelArg->channels[i],
                    TOKEN_XN_ID);
        }
    }

    uint32_t argIndex = 0;

    CCU_CHECK(ccu::LoadArg(
        localOutput,
        argIndex++));

    CCU_CHECK(ccu::LoadArg(
        localToken,
        argIndex++));

    CCU_CHECK(ccu::LoadArg(
        sliceOffset,
        argIndex++));

    CCU_CHECK(ccu::LoadArg(
        sliceSize,
        argIndex++));

    /* 只有root会执行远端写，因此仅非root上报接收地址和token。 */
    constexpr uint32_t addressReadyMask =
        (1U << OUTPUT_XN_ID) |
        (1U << TOKEN_XN_ID);

    if (kernelArg->rankId == kernelArg->root) {
        for (uint32_t i = 0; i < kernelArg->channelCount; ++i) {
            CCU_CHECK(ccu::NotifyWait(
                kernelArg->channels[i],
                CKE_INDEX,
                addressReadyMask));
        }
    } else {
        for (uint32_t i = 0; i < kernelArg->channelCount; ++i) {
            CCU_CHECK(ccu::WriteVariableWithNotify(
                kernelArg->channels[i],
                localOutput,
                OUTPUT_XN_ID,
                CKE_INDEX,
                1U << OUTPUT_XN_ID));

            CCU_CHECK(ccu::WriteVariableWithNotify(
                kernelArg->channels[i],
                localToken,
                TOKEN_XN_ID,
                CKE_INDEX,
                1U << TOKEN_XN_ID));
        }
    }

    /*
     * 只有root执行远端写。
     * root同一层中的多个Write会先全部下发，再统一等待Event。
     */
    if (kernelArg->rankId == kernelArg->root) {
        ccu::LocalAddr source;

        source.addr = localOutput;
        source.addr += sliceOffset;
        source.token = localToken;

        ccu::Event writeDone;

        uint16_t allWritesMask = 0;
        for (uint32_t i = 0; i < kernelArg->channelCount; ++i) {
            ccu::RemoteAddr destination;

            destination.addr = remoteOutput[i];
            destination.addr += sliceOffset;
            destination.token = remoteToken[i];

            const uint16_t eventMask =
                static_cast<uint16_t>(1U << i);
            allWritesMask = static_cast<uint16_t>(allWritesMask | eventMask);

            CCU_CHECK(ccu::Write(
                kernelArg->channels[i],
                destination,
                source,
                sliceSize,
                writeDone,
                eventMask));
        }

        CCU_CHECK(ccu::EventWait(
            writeDone,
            allWritesMask));
    }

    /* root写完成后单向唤醒对端；地址上报已保证所有对端都已进入Kernel。 */
    if (kernelArg->rankId == kernelArg->root) {
        for (uint32_t i = 0; i < kernelArg->channelCount; ++i) {
            CCU_CHECK(ccu::NotifyRecord(
                kernelArg->channels[i],
                CKE_INDEX,
                1U << POST_SYNC_ID));
        }
    } else {
        for (uint32_t i = 0; i < kernelArg->channelCount; ++i) {
            CCU_CHECK(ccu::NotifyWait(
                kernelArg->channels[i],
                CKE_INDEX,
                1U << POST_SYNC_ID));
        }
    }

    return CCU_SUCCESS;
}

CcuResult CcuBroadcastMeshKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgBroadcast *>(arg);

    if (kernelArg == nullptr || kernelArg->rankSize <= 2 || kernelArg->root >= kernelArg->rankSize ||
        kernelArg->rankId >= kernelArg->rankSize || kernelArg->channelCount + 1 != kernelArg->rankSize ||
        kernelArg->channelCount >= MAX_RANK_SIZE ||
        (kernelArg->pipelineSteps != PIPELINE_STEPS &&
            kernelArg->pipelineSteps != FINE_PIPELINE_STEPS &&
            kernelArg->pipelineSteps != DEEP_PIPELINE_STEPS)) {
        return CCU_E_PARA;
    }

    ccu::Variable localOutput;
    ccu::Variable localToken;
    ccu::Variable sliceOffset;
    ccu::Variable blockSize;
    ccu::Variable lastBlockSize;
    ccu::Variable blockSegmentSize;
    ccu::Variable blockLastSegmentSize;
    ccu::Variable tailSegmentSize;
    ccu::Variable tailLastSegmentSize;

    std::vector<ccu::Variable> remoteOutput(kernelArg->channelCount);
    std::vector<ccu::Variable> remoteToken(kernelArg->channelCount);

    for (uint32_t i = 0; i < kernelArg->channelCount; ++i) {
        if (kernelArg->remoteRanks[i] != kernelArg->root) {
            remoteOutput[i] = ccu::GetResByChannel<ccu::Variable>(kernelArg->channels[i], OUTPUT_XN_ID);
            remoteToken[i] = ccu::GetResByChannel<ccu::Variable>(kernelArg->channels[i], TOKEN_XN_ID);
        }
    }

    uint32_t argIndex = 0;
    CCU_CHECK(ccu::LoadArg(localOutput, argIndex++));
    CCU_CHECK(ccu::LoadArg(localToken, argIndex++));
    CCU_CHECK(ccu::LoadArg(sliceOffset, argIndex++));
    CCU_CHECK(ccu::LoadArg(blockSize, argIndex++));
    CCU_CHECK(ccu::LoadArg(lastBlockSize, argIndex++));
    CCU_CHECK(ccu::LoadArg(blockSegmentSize, argIndex++));
    CCU_CHECK(ccu::LoadArg(blockLastSegmentSize, argIndex++));
    CCU_CHECK(ccu::LoadArg(tailSegmentSize, argIndex++));
    CCU_CHECK(ccu::LoadArg(tailLastSegmentSize, argIndex++));

    if (kernelArg->rankId != kernelArg->root) {
        for (uint32_t i = 0; i < kernelArg->channelCount; ++i) {
            CCU_CHECK(ccu::WriteVariableWithNotify(
                kernelArg->channels[i], localOutput, OUTPUT_XN_ID, CKE_INDEX, 1U << OUTPUT_XN_ID));
            CCU_CHECK(ccu::WriteVariableWithNotify(
                kernelArg->channels[i], localToken, TOKEN_XN_ID, CKE_INDEX, 1U << TOKEN_XN_ID));
        }
    }

    constexpr uint32_t addressReadyMask = (1U << OUTPUT_XN_ID) | (1U << TOKEN_XN_ID);
    for (uint32_t i = 0; i < kernelArg->channelCount; ++i) {
        if (kernelArg->remoteRanks[i] != kernelArg->root) {
            CCU_CHECK(ccu::NotifyWait(kernelArg->channels[i], CKE_INDEX, addressReadyMask));
        }
    }

    if (kernelArg->rankId == kernelArg->root) {
        std::vector<ccu::Event> scatterDone(kernelArg->pipelineSteps);
        ccu::LocalAddr scatterSource[MAX_RANK_SIZE]{};
        ccu::RemoteAddr scatterDestination[MAX_RANK_SIZE]{};

        for (uint32_t i = 0; i < kernelArg->channelCount; ++i) {
            const uint32_t ownerIndex =
                kernelArg->remoteRanks[i] < kernelArg->root
                    ? kernelArg->remoteRanks[i]
                    : kernelArg->remoteRanks[i] - 1;
            scatterSource[i].addr = localOutput;
            scatterSource[i].addr += sliceOffset;
            scatterDestination[i].addr = remoteOutput[i];
            scatterDestination[i].addr += sliceOffset;
            for (uint32_t block = 0; block < ownerIndex; ++block) {
                scatterSource[i].addr += blockSize;
                scatterDestination[i].addr += blockSize;
            }
            scatterSource[i].token = localToken;
            scatterDestination[i].token = remoteToken[i];
        }

        for (uint32_t step = 0; step < kernelArg->pipelineSteps; ++step) {
            uint16_t scatterMask = 0;
            for (uint32_t i = 0; i < kernelArg->channelCount; ++i) {
                const uint32_t ownerIndex = kernelArg->remoteRanks[i] < kernelArg->root
                    ? kernelArg->remoteRanks[i]
                    : kernelArg->remoteRanks[i] - 1;
                const bool tailOwner = ownerIndex + 1 == kernelArg->rankSize - 1;
                const ccu::Variable *segmentSize = tailOwner ? &tailSegmentSize : &blockSegmentSize;
                const ccu::Variable *transferSize = step + 1 == kernelArg->pipelineSteps
                    ? (tailOwner ? &tailLastSegmentSize : &blockLastSegmentSize)
                    : segmentSize;

                const uint16_t eventMask = static_cast<uint16_t>(1U << i);
                scatterMask = static_cast<uint16_t>(
                    scatterMask | eventMask);
                CCU_CHECK(ccu::Write(
                    kernelArg->channels[i],
                    scatterDestination[i],
                    scatterSource[i],
                    *transferSize,
                    scatterDone[step],
                    eventMask));
            }

            CCU_CHECK(ccu::EventWait(
                scatterDone[step],
                scatterMask));
            for (uint32_t i = 0; i < kernelArg->channelCount; ++i) {
                CCU_CHECK(ccu::NotifyRecord(
                    kernelArg->channels[i], CKE_INDEX, 1U << (POST_SYNC_ID + step)));
            }

            if (step + 1 < kernelArg->pipelineSteps) {
                for (uint32_t i = 0;
                     i < kernelArg->channelCount;
                     ++i) {
                    const uint32_t ownerIndex =
                        kernelArg->remoteRanks[i] < kernelArg->root
                            ? kernelArg->remoteRanks[i]
                            : kernelArg->remoteRanks[i] - 1;
                    const bool tailOwner =
                        ownerIndex + 1 == kernelArg->rankSize - 1;
                    const ccu::Variable *segmentSize =
                        tailOwner
                            ? &tailSegmentSize
                            : &blockSegmentSize;
                    scatterSource[i].addr += *segmentSize;
                    scatterDestination[i].addr += *segmentSize;
                }
            }
        }
    } else {
        const uint32_t ownerIndex =
            kernelArg->rankId < kernelArg->root ? kernelArg->rankId : kernelArg->rankId - 1;
        const bool tailOwner = ownerIndex + 1 == kernelArg->rankSize - 1;
        const ccu::Variable *segmentSize = tailOwner ? &tailSegmentSize : &blockSegmentSize;
        std::vector<ccu::Event> gatherDone(kernelArg->pipelineSteps);
        ccu::LocalAddr gatherSource;
        gatherSource.addr = localOutput;
        gatherSource.addr += sliceOffset;
        for (uint32_t block = 0; block < ownerIndex; ++block) {
            gatherSource.addr += blockSize;
        }
        gatherSource.token = localToken;

        ccu::RemoteAddr gatherDestination[MAX_RANK_SIZE]{};
        for (uint32_t i = 0; i < kernelArg->channelCount; ++i) {
            if (kernelArg->remoteRanks[i] == kernelArg->root) {
                continue;
            }
            gatherDestination[i].addr = remoteOutput[i];
            gatherDestination[i].addr += sliceOffset;
            for (uint32_t block = 0; block < ownerIndex; ++block) {
                gatherDestination[i].addr += blockSize;
            }
            gatherDestination[i].token = remoteToken[i];
        }

        for (uint32_t step = 0; step < kernelArg->pipelineSteps; ++step) {
            for (uint32_t i = 0; i < kernelArg->channelCount; ++i) {
                if (kernelArg->remoteRanks[i] == kernelArg->root) {
                    CCU_CHECK(ccu::NotifyWait(
                        kernelArg->channels[i], CKE_INDEX, 1U << (POST_SYNC_ID + step)));
                    break;
                }
            }

            const ccu::Variable *transferSize = step + 1 == kernelArg->pipelineSteps
                ? (tailOwner ? &tailLastSegmentSize : &blockLastSegmentSize)
                : segmentSize;
            uint16_t gatherMask = 0;
            for (uint32_t i = 0; i < kernelArg->channelCount; ++i) {
                if (kernelArg->remoteRanks[i] == kernelArg->root) {
                    continue;
                }

                const uint16_t eventMask = static_cast<uint16_t>(1U << i);
                gatherMask = static_cast<uint16_t>(gatherMask | eventMask);
                CCU_CHECK(ccu::Write(
                    kernelArg->channels[i],
                    gatherDestination[i],
                    gatherSource,
                    *transferSize,
                    gatherDone[step],
                    eventMask));
            }

            CCU_CHECK(ccu::EventWait(gatherDone[step], gatherMask));
            for (uint32_t i = 0; i < kernelArg->channelCount; ++i) {
                CCU_CHECK(ccu::NotifyRecord(
                    kernelArg->channels[i], CKE_INDEX, 1U << (POST_SYNC_ID + step)));
            }

            if (step + 1 < kernelArg->pipelineSteps) {
                gatherSource.addr += *segmentSize;
                for (uint32_t i = 0;
                     i < kernelArg->channelCount;
                     ++i) {
                    if (kernelArg->remoteRanks[i] != kernelArg->root) {
                        gatherDestination[i].addr += *segmentSize;
                    }
                }
            }
        }
    }

    /*
     * 所有Gather发送端在进入本阶段前，已经依次记录了每一个step的完成位。
     * 因此同一channel无需逐bit重复等待，可一次等待完整step mask。
     * 仅合并同层、同channel的既有通知，不新增通知或跨Kernel依赖。
     */
    const uint32_t allStepsMask =
        ((1U << kernelArg->pipelineSteps) - 1U) << POST_SYNC_ID;
    for (uint32_t i = 0; i < kernelArg->channelCount; ++i) {
        if (kernelArg->remoteRanks[i] == kernelArg->root) {
            if (kernelArg->rankId == kernelArg->root) {
                CCU_CHECK(ccu::NotifyWait(
                    kernelArg->channels[i], CKE_INDEX, allStepsMask));
            }
            continue;
        }
        CCU_CHECK(ccu::NotifyWait(
            kernelArg->channels[i], CKE_INDEX, allStepsMask));
    }

    return CCU_SUCCESS;
}

CcuResult CcuBroadcastScatterKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgBroadcast *>(arg);

    if (kernelArg == nullptr || kernelArg->rankSize <= 2 || kernelArg->root >= kernelArg->rankSize ||
        kernelArg->rankId >= kernelArg->rankSize || kernelArg->channelCount == 0 ||
        kernelArg->channelCount >= MAX_RANK_SIZE) {
        return CCU_E_PARA;
    }

    ccu::Variable localOutput;
    ccu::Variable localToken;
    ccu::Variable sliceOffset;
    ccu::Variable blockSize;
    ccu::Variable lastBlockSize;
    ccu::Variable readyAddress;
    ccu::Variable readyValue;
    std::vector<ccu::Variable> remoteOutput(kernelArg->channelCount);
    std::vector<ccu::Variable> remoteToken(kernelArg->channelCount);

    if (kernelArg->rankId == kernelArg->root) {
        for (uint32_t i = 0; i < kernelArg->channelCount; ++i) {
            remoteOutput[i] = ccu::GetResByChannel<ccu::Variable>(kernelArg->channels[i], OUTPUT_XN_ID);
            remoteToken[i] = ccu::GetResByChannel<ccu::Variable>(kernelArg->channels[i], TOKEN_XN_ID);
        }
    }

    uint32_t argIndex = 0;
    CCU_CHECK(ccu::LoadArg(localOutput, argIndex++));
    CCU_CHECK(ccu::LoadArg(localToken, argIndex++));
    CCU_CHECK(ccu::LoadArg(sliceOffset, argIndex++));
    CCU_CHECK(ccu::LoadArg(blockSize, argIndex++));
    CCU_CHECK(ccu::LoadArg(lastBlockSize, argIndex++));
    CCU_CHECK(ccu::LoadArg(readyAddress, argIndex++));
    CCU_CHECK(ccu::LoadArg(readyValue, argIndex++));

    constexpr uint32_t addressReadyMask = (1U << OUTPUT_XN_ID) | (1U << TOKEN_XN_ID);
    if (kernelArg->rankId == kernelArg->root) {
        for (uint32_t i = 0; i < kernelArg->channelCount; ++i) {
            CCU_CHECK(ccu::NotifyWait(kernelArg->channels[i], CKE_INDEX, addressReadyMask));
        }
    } else {
        for (uint32_t i = 0; i < kernelArg->channelCount; ++i) {
            CCU_CHECK(ccu::WriteVariableWithNotify(
                kernelArg->channels[i], localOutput, OUTPUT_XN_ID, CKE_INDEX, 1U << OUTPUT_XN_ID));
            CCU_CHECK(ccu::WriteVariableWithNotify(
                kernelArg->channels[i], localToken, TOKEN_XN_ID, CKE_INDEX, 1U << TOKEN_XN_ID));
        }
    }

    if (kernelArg->rankId == kernelArg->root) {
        ccu::Event scatterDone;
        for (uint32_t i = 0; i < kernelArg->channelCount; ++i) {
            const uint32_t ownerIndex = kernelArg->remoteRanks[i] < kernelArg->root
                ? kernelArg->remoteRanks[i]
                : kernelArg->remoteRanks[i] - 1;

            ccu::LocalAddr source;
            source.addr = localOutput;
            source.addr += sliceOffset;
            ccu::RemoteAddr destination;
            destination.addr = remoteOutput[i];
            destination.addr += sliceOffset;
            for (uint32_t block = 0; block < ownerIndex; ++block) {
                source.addr += blockSize;
                destination.addr += blockSize;
            }
            source.token = localToken;
            destination.token = remoteToken[i];

            const ccu::Variable *transferSize = ownerIndex + 1 == kernelArg->rankSize - 1
                ? &lastBlockSize
                : &blockSize;
            const uint16_t eventMask = static_cast<uint16_t>(1U << i);
            CCU_CHECK(ccu::Write(
                kernelArg->channels[i], destination, source, *transferSize, scatterDone, eventMask));
        }

        for (uint32_t i = 0; i < kernelArg->channelCount; ++i) {
            const uint16_t eventMask = static_cast<uint16_t>(1U << i);
            CCU_CHECK(ccu::EventWait(scatterDone, eventMask));
            CCU_CHECK(ccu::NotifyRecord(
                kernelArg->channels[i], CKE_INDEX, 1U << POST_SYNC_ID));
        }
    } else {
        CCU_CHECK(ccu::NotifyWait(kernelArg->channels[0], CKE_INDEX, 1U << POST_SYNC_ID));
        CCU_CHECK(ccu::Store(readyAddress, readyValue));
    }

    return CCU_SUCCESS;
}

CcuResult CcuBroadcastGatherKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgBroadcast *>(arg);

    if (kernelArg == nullptr || kernelArg->rankSize <= 2 || kernelArg->root >= kernelArg->rankSize ||
        kernelArg->rankId >= kernelArg->rankSize || kernelArg->channelCount == 0 ||
        kernelArg->channelCount >= MAX_RANK_SIZE) {
        return CCU_E_PARA;
    }

    ccu::Variable localOutput;
    ccu::Variable localToken;
    ccu::Variable sliceOffset;
    ccu::Variable blockSize;
    ccu::Variable lastBlockSize;
    ccu::Variable readyAddress;
    ccu::Variable readyValue;
    ccu::Variable observedEpoch;
    std::vector<ccu::Variable> remoteOutput(kernelArg->channelCount);
    std::vector<ccu::Variable> remoteToken(kernelArg->channelCount);

    if (kernelArg->rankId != kernelArg->root) {
        for (uint32_t i = 0; i < kernelArg->channelCount; ++i) {
            if (kernelArg->remoteRanks[i] != kernelArg->root) {
                remoteOutput[i] = ccu::GetResByChannel<ccu::Variable>(kernelArg->channels[i], OUTPUT_XN_ID);
                remoteToken[i] = ccu::GetResByChannel<ccu::Variable>(kernelArg->channels[i], TOKEN_XN_ID);
            }
        }
    }

    uint32_t argIndex = 0;
    CCU_CHECK(ccu::LoadArg(localOutput, argIndex++));
    CCU_CHECK(ccu::LoadArg(localToken, argIndex++));
    CCU_CHECK(ccu::LoadArg(sliceOffset, argIndex++));
    CCU_CHECK(ccu::LoadArg(blockSize, argIndex++));
    CCU_CHECK(ccu::LoadArg(lastBlockSize, argIndex++));
    CCU_CHECK(ccu::LoadArg(readyAddress, argIndex++));
    CCU_CHECK(ccu::LoadArg(readyValue, argIndex++));

    if (kernelArg->rankId != kernelArg->root) {
        for (uint32_t i = 0; i < kernelArg->channelCount; ++i) {
            if (kernelArg->remoteRanks[i] == kernelArg->root) {
                continue;
            }
            CCU_CHECK(ccu::WriteVariableWithNotify(
                kernelArg->channels[i], localOutput, OUTPUT_XN_ID, CKE_INDEX, 1U << OUTPUT_XN_ID));
            CCU_CHECK(ccu::WriteVariableWithNotify(
                kernelArg->channels[i], localToken, TOKEN_XN_ID, CKE_INDEX, 1U << TOKEN_XN_ID));
        }
    }

    constexpr uint32_t addressReadyMask = (1U << OUTPUT_XN_ID) | (1U << TOKEN_XN_ID);
    if (kernelArg->rankId != kernelArg->root) {
        for (uint32_t i = 0; i < kernelArg->channelCount; ++i) {
            if (kernelArg->remoteRanks[i] != kernelArg->root) {
                CCU_CHECK(ccu::NotifyWait(kernelArg->channels[i], CKE_INDEX, addressReadyMask));
            }
        }
    }

    if (kernelArg->rankId != kernelArg->root) {
        CCU_CHECK(ccu::Load(readyAddress, observedEpoch));
        CCU_WHILE(observedEpoch != 1ULL) {
            CCU_CHECK(ccu::Load(readyAddress, observedEpoch));
        }

        const uint32_t ownerIndex =
            kernelArg->rankId < kernelArg->root ? kernelArg->rankId : kernelArg->rankId - 1;
        ccu::LocalAddr source;
        source.addr = localOutput;
        source.addr += sliceOffset;
        for (uint32_t block = 0; block < ownerIndex; ++block) {
            source.addr += blockSize;
        }
        source.token = localToken;

        const ccu::Variable *transferSize = ownerIndex + 1 == kernelArg->rankSize - 1
            ? &lastBlockSize
            : &blockSize;
        ccu::Event gatherDone;
        uint16_t gatherMask = 0;
        for (uint32_t i = 0; i < kernelArg->channelCount; ++i) {
            if (kernelArg->remoteRanks[i] == kernelArg->root) {
                continue;
            }

            ccu::RemoteAddr destination;
            destination.addr = remoteOutput[i];
            destination.addr += sliceOffset;
            for (uint32_t block = 0; block < ownerIndex; ++block) {
                destination.addr += blockSize;
            }
            destination.token = remoteToken[i];

            const uint16_t eventMask = static_cast<uint16_t>(1U << i);
            gatherMask = static_cast<uint16_t>(gatherMask | eventMask);
            CCU_CHECK(ccu::Write(
                kernelArg->channels[i], destination, source, *transferSize, gatherDone, eventMask));
        }
        if (gatherMask != 0) {
            CCU_CHECK(ccu::EventWait(gatherDone, gatherMask));
        }
    }

    if (kernelArg->rankId != kernelArg->root) {
        for (uint32_t i = 0; i < kernelArg->channelCount; ++i) {
            CCU_CHECK(ccu::NotifyRecord(kernelArg->channels[i], CKE_INDEX, 1U << POST_SYNC_ID));
        }
    }
    for (uint32_t i = 0; i < kernelArg->channelCount; ++i) {
        if (kernelArg->remoteRanks[i] == kernelArg->root) {
            if (kernelArg->rankId == kernelArg->root) {
                CCU_CHECK(ccu::NotifyWait(
                    kernelArg->channels[i], CKE_INDEX, 1U << POST_SYNC_ID));
            }
            continue;
        }
        CCU_CHECK(ccu::NotifyWait(kernelArg->channels[i], CKE_INDEX, 1U << POST_SYNC_ID));
    }

    return CCU_SUCCESS;
}

#undef CCU_CHECK

} // namespace ops_hccl