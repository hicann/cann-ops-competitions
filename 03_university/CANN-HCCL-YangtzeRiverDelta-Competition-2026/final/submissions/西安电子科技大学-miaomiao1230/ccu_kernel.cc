/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <cstdint>
#include <vector>

#include <ccu/ccu_primitives.hpp>

#include "ccu_kernel.h"
#include "custom.h"

namespace ops_hccl {
namespace {

#define CCU_CHECK(call)          \
    do {                         \
        CcuResult ret = (call);  \
        if (ret != CCU_SUCCESS) { \
            return ret;          \
        }                        \
    } while (0)

constexpr uint32_t ARG_LOCAL_ADDRESS = 0;
constexpr uint32_t ARG_LOCAL_TOKEN = 1;
constexpr uint32_t ARG_TRANSFER_LENGTH = 2;
constexpr uint32_t ARG_NORMAL_SLICE_LENGTH = 2;
constexpr uint32_t ARG_LAST_SLICE_LENGTH = 3;
constexpr uint32_t ARG_CROSS_ADDRESS = 4;
constexpr uint32_t ARG_CROSS_NORMAL_SLICE_LENGTH = 5;
constexpr uint32_t ARG_CROSS_LAST_SLICE_LENGTH = 6;

constexpr uint32_t CHANNEL_VAR_ADDRESS = 0;
constexpr uint32_t CHANNEL_VAR_TOKEN = 1;
constexpr uint32_t CHANNEL_NOTIFY_ADDRESS = 0;
constexpr uint32_t CHANNEL_NOTIFY_TOKEN = 1;
constexpr uint32_t CHANNEL_NOTIFY_DATA_READY = 2;
constexpr uint32_t CHANNEL_NOTIFY_PHASE2_DONE = 3;
constexpr uint32_t CHANNEL_NOTIFY_PHASE3_DONE = 4;
constexpr uint32_t CHANNEL_NOTIFY_BALANCED_STAGE1_ADDRESS = 4;
constexpr uint32_t CHANNEL_NOTIFY_BALANCED_STAGE1_TOKEN = 5;
constexpr uint32_t CHANNEL_NOTIFY_BALANCED_STAGE1_DATA = 6;

// v33 merge: v27 4x1 chain pipeline notify slots. RunPipelinedChain4 owns its
// own kernel instance (mode PIPELINED_CHAIN4, rankSize 4), so these indices
// never collide with the rankSize-12 balanced/hierarchical notifies above.
constexpr uint32_t CHANNEL_NOTIFY_CHAIN_DATA_BASE = 2;
constexpr uint32_t CHANNEL_NOTIFY_CHAIN_ACK_BASE = 4;
constexpr uint32_t CHANNEL_NOTIFY_CHAIN_POST_SYNC = 6;
constexpr uint32_t CHAIN_PIPELINE_SLOT_COUNT = 2;

// Tree-only packed bootstrap and static-pipeline DATA signals.  The generic
// Scatter/AllGather kernels above retain their original separate notify
// indices; these constants are consumed only by CcuKernel below.
constexpr uint32_t TREE_NOTIFY_CONTROL = 0;
constexpr uint16_t TREE_NOTIFY_ADDRESS_MASK = 1U << 0;
constexpr uint16_t TREE_NOTIFY_TOKEN_MASK = 1U << 1;
constexpr uint16_t TREE_NOTIFY_CONTROL_MASK =
    TREE_NOTIFY_ADDRESS_MASK | TREE_NOTIFY_TOKEN_MASK;
constexpr uint32_t TREE_NOTIFY_SMALL_DATA = 1;
constexpr uint32_t TREE_NOTIFY_PIPELINE_BIT_OFFSET = 2;
constexpr uint32_t TREE_NOTIFY_MASK_BITS = 16;

constexpr uint32_t FIRST_SERVER_SIZE = 8;

using AscendC::ccu::Address;
using AscendC::ccu::Event;
using AscendC::ccu::LocalAddr;
using AscendC::ccu::RemoteAddr;
using AscendC::ccu::Variable;

uint32_t FindPeerIndex(const CcuKernelArgBase *kernelArg, uint32_t peerRank)
{
    for (uint32_t peerIdx = 0; peerIdx < kernelArg->peerCount; ++peerIdx) {
        if (kernelArg->peerRanks[peerIdx] == peerRank) {
            return peerIdx;
        }
    }
    return kernelArg->peerCount;
}

void AddSliceOffset(Address &address, const Variable &normalSliceLength, uint32_t sliceIndex)
{
    for (uint32_t step = 0; step < sliceIndex; ++step) {
        address += normalSliceLength;
    }
}

void SetLocalSlice(LocalAddr &slice, const Variable &baseAddress, const Variable &token,
    const Variable &normalSliceLength, uint32_t sliceIndex)
{
    slice.addr = baseAddress;
    slice.token = token;
    AddSliceOffset(slice.addr, normalSliceLength, sliceIndex);
}

void SetRemoteSlice(RemoteAddr &slice, const RemoteAddr &baseAddress,
    const Variable &normalSliceLength, uint32_t sliceIndex)
{
    slice.addr = baseAddress.addr;
    slice.token = baseAddress.token;
    AddSliceOffset(slice.addr, normalSliceLength, sliceIndex);
}

CcuResult ExchangePeerAddressesAt(const CcuKernelArgBase *kernelArg,
    const Variable &localAddressValue, const Variable &localTokenValue,
    uint32_t addressNotifyIndex, uint32_t tokenNotifyIndex,
    std::vector<RemoteAddr> &peerAddresses)
{
    using namespace AscendC::ccu;

    for (uint32_t peerIdx = 0; peerIdx < kernelArg->peerCount; ++peerIdx) {
        const ChannelHandle channel = kernelArg->peerChannels[peerIdx];
        CCU_CHECK(WriteVariableWithNotify(
            channel, localAddressValue, CHANNEL_VAR_ADDRESS,
            addressNotifyIndex));
        CCU_CHECK(WriteVariableWithNotify(
            channel, localTokenValue, CHANNEL_VAR_TOKEN,
            tokenNotifyIndex));
    }

    peerAddresses.clear();
    peerAddresses.reserve(kernelArg->peerCount);
    for (uint32_t peerIdx = 0; peerIdx < kernelArg->peerCount; ++peerIdx) {
        const ChannelHandle channel = kernelArg->peerChannels[peerIdx];
        CCU_CHECK(NotifyWait(channel, addressNotifyIndex));
        CCU_CHECK(NotifyWait(channel, tokenNotifyIndex));

        Variable peerAddressValue = GetResByChannel<Variable>(channel, CHANNEL_VAR_ADDRESS);
        Variable peerTokenValue = GetResByChannel<Variable>(channel, CHANNEL_VAR_TOKEN);
        peerAddresses.emplace_back();
        peerAddresses.back().addr = peerAddressValue;
        peerAddresses.back().token = peerTokenValue;
    }
    return CCU_SUCCESS;
}

CcuResult ExchangePeerAddresses(const CcuKernelArgBase *kernelArg,
    const Variable &localAddressValue, const Variable &localTokenValue,
    std::vector<RemoteAddr> &peerAddresses)
{
    return ExchangePeerAddressesAt(kernelArg, localAddressValue,
        localTokenValue, CHANNEL_NOTIFY_ADDRESS, CHANNEL_NOTIFY_TOKEN,
        peerAddresses);
}

uint32_t GetOwnerRank(uint32_t rootGroupBegin, uint32_t rootGroupSize,
    uint32_t root, uint32_t sliceIndex)
{
    const uint32_t rootLocalIndex = root - rootGroupBegin;
    return rootGroupBegin + (rootLocalIndex + sliceIndex) % rootGroupSize;
}

bool OwnsSlice(uint32_t rank, uint32_t rootGroupBegin, uint32_t rootGroupSize,
    uint32_t root, uint32_t sliceIndex)
{
    return GetOwnerRank(rootGroupBegin, rootGroupSize, root, sliceIndex) == rank;
}

CcuResult RunMeshScatterAllgather(const CcuKernelArgBase *kernelArg,
    const Variable &localAddressValue, const Variable &localTokenValue,
    Variable &normalSliceLength, Variable &lastSliceLength,
    const std::vector<RemoteAddr> &peerAddresses)
{
    using namespace AscendC::ccu;

    if (kernelArg->rankSize != 4 || kernelArg->peerCount != kernelArg->rankSize - 1) {
        return CCU_E_PARA;
    }

    std::vector<LocalAddr> localSlices(kernelArg->peerCount);
    std::vector<RemoteAddr> remoteSlices(kernelArg->peerCount);
    Event transferDone;

    if (kernelArg->myRank == kernelArg->root) {
        uint16_t scatterMask = 0;
        for (uint32_t peerIdx = 0; peerIdx < kernelArg->peerCount; ++peerIdx) {
            const uint32_t sliceIndex = kernelArg->peerRanks[peerIdx];
            SetLocalSlice(localSlices[peerIdx], localAddressValue, localTokenValue,
                normalSliceLength, sliceIndex);
            SetRemoteSlice(remoteSlices[peerIdx], peerAddresses[peerIdx],
                normalSliceLength, sliceIndex);
            Variable &sliceLength = sliceIndex + 1 == kernelArg->rankSize ?
                lastSliceLength : normalSliceLength;
            const uint16_t writeMask = static_cast<uint16_t>(1U << peerIdx);
            CCU_CHECK(Write(kernelArg->peerChannels[peerIdx], remoteSlices[peerIdx],
                localSlices[peerIdx], sliceLength, transferDone, writeMask));
            scatterMask = static_cast<uint16_t>(scatterMask | writeMask);
        }
        CCU_CHECK(EventWait(transferDone, scatterMask));
        for (uint32_t peerIdx = 0; peerIdx < kernelArg->peerCount; ++peerIdx) {
            CCU_CHECK(NotifyRecord(
                kernelArg->peerChannels[peerIdx], CHANNEL_NOTIFY_DATA_READY));
        }
    } else {
        const uint32_t rootPeerIdx = FindPeerIndex(kernelArg, kernelArg->root);
        if (rootPeerIdx >= kernelArg->peerCount) {
            return CCU_E_PARA;
        }
        CCU_CHECK(NotifyWait(
            kernelArg->peerChannels[rootPeerIdx], CHANNEL_NOTIFY_DATA_READY));
    }

    uint16_t allgatherMask = 0;
    for (uint32_t peerIdx = 0; peerIdx < kernelArg->peerCount; ++peerIdx) {
        SetLocalSlice(localSlices[peerIdx], localAddressValue, localTokenValue,
            normalSliceLength, kernelArg->myRank);
        SetRemoteSlice(remoteSlices[peerIdx], peerAddresses[peerIdx],
            normalSliceLength, kernelArg->myRank);
        Variable &sliceLength = kernelArg->myRank + 1 == kernelArg->rankSize ?
            lastSliceLength : normalSliceLength;
        const uint16_t writeMask = static_cast<uint16_t>(1U << peerIdx);
        CCU_CHECK(Write(kernelArg->peerChannels[peerIdx], remoteSlices[peerIdx],
            localSlices[peerIdx], sliceLength, transferDone, writeMask));
        allgatherMask = static_cast<uint16_t>(allgatherMask | writeMask);
    }
    CCU_CHECK(EventWait(transferDone, allgatherMask));

    for (uint32_t peerIdx = 0; peerIdx < kernelArg->peerCount; ++peerIdx) {
        CCU_CHECK(NotifyRecord(
            kernelArg->peerChannels[peerIdx], CHANNEL_NOTIFY_PHASE2_DONE));
    }
    for (uint32_t peerIdx = 0; peerIdx < kernelArg->peerCount; ++peerIdx) {
        CCU_CHECK(NotifyWait(
            kernelArg->peerChannels[peerIdx], CHANNEL_NOTIFY_PHASE2_DONE));
    }
    return CCU_SUCCESS;
}

// v33 merge: ported verbatim from v27 (chain4-pipeline16). Root-anchored
// forwarding chain for the 4x1 topology, split into 16 segments per 256 MiB
// host chunk with a two-slot rotating DATA/ACK credit pipeline.
CcuResult RunPipelinedChain4(const CcuKernelArgBase *kernelArg,
    const Variable &localAddressValue, const Variable &localTokenValue,
    Variable &normalSegmentLength, Variable &lastSegmentLength,
    const std::vector<RemoteAddr> &peerAddresses)
{
    using namespace AscendC::ccu;

    constexpr uint32_t CHAIN_RANK_SIZE = 4;
    if (kernelArg->rankSize != CHAIN_RANK_SIZE ||
        kernelArg->peerCount != CHAIN_RANK_SIZE - 1 ||
        kernelArg->root >= CHAIN_RANK_SIZE) {
        return CCU_E_PARA;
    }

    LocalAddr localSegment;
    RemoteAddr remoteSegment;
    Event transferDone;
    const uint32_t myRank = kernelArg->myRank;
    const uint32_t chainPosition =
        (myRank + CHAIN_RANK_SIZE - kernelArg->root) % CHAIN_RANK_SIZE;
    const bool hasParent = chainPosition != 0;
    const bool hasChild = chainPosition + 1 != CHAIN_RANK_SIZE;
    uint32_t parentPeerIdx = kernelArg->peerCount;
    uint32_t childPeerIdx = kernelArg->peerCount;
    if (hasParent) {
        const uint32_t parentRank =
            (myRank + CHAIN_RANK_SIZE - 1) % CHAIN_RANK_SIZE;
        parentPeerIdx = FindPeerIndex(kernelArg, parentRank);
        if (parentPeerIdx >= kernelArg->peerCount) {
            return CCU_E_PARA;
        }
    }
    if (hasChild) {
        const uint32_t childRank = (myRank + 1) % CHAIN_RANK_SIZE;
        childPeerIdx = FindPeerIndex(kernelArg, childRank);
        if (childPeerIdx >= kernelArg->peerCount) {
            return CCU_E_PARA;
        }
    }

    // The root injects every byte exactly once. The two middle ranks forward
    // each segment as soon as it arrives, so the three links form one CCU-side
    // pipeline instead of making the root participate in an allgather.
    for (uint32_t segmentIndex = 0;
        segmentIndex < CHAIN4_PIPELINE_SEGMENT_COUNT; ++segmentIndex) {
        const uint32_t slot = segmentIndex % CHAIN_PIPELINE_SLOT_COUNT;
        const uint32_t dataNotify = CHANNEL_NOTIFY_CHAIN_DATA_BASE + slot;
        const uint32_t ackNotify = CHANNEL_NOTIFY_CHAIN_ACK_BASE + slot;

        if (hasParent) {
            CCU_CHECK(NotifyWait(
                kernelArg->peerChannels[parentPeerIdx], dataNotify));
            // Return the credit immediately after consuming the DATA notify.
            // This guarantees that the sender never records twice into one
            // notify slot before the matching wait has reset it.
            CCU_CHECK(NotifyRecord(
                kernelArg->peerChannels[parentPeerIdx], ackNotify));
        }

        if (hasChild) {
            if (segmentIndex >= CHAIN_PIPELINE_SLOT_COUNT) {
                CCU_CHECK(NotifyWait(
                    kernelArg->peerChannels[childPeerIdx], ackNotify));
            }

            SetLocalSlice(localSegment, localAddressValue, localTokenValue,
                normalSegmentLength, segmentIndex);
            SetRemoteSlice(remoteSegment, peerAddresses[childPeerIdx],
                normalSegmentLength, segmentIndex);
            Variable &segmentLength =
                segmentIndex + 1 == CHAIN4_PIPELINE_SEGMENT_COUNT ?
                lastSegmentLength : normalSegmentLength;
            CCU_CHECK(Write(kernelArg->peerChannels[childPeerIdx],
                remoteSegment, localSegment, segmentLength, transferDone, 1U));
            CCU_CHECK(EventWait(transferDone, 1U));
            CCU_CHECK(NotifyRecord(
                kernelArg->peerChannels[childPeerIdx], dataNotify));
        }
    }

    // Drain the final two credits so the ACK notify slots are clear before the
    // next 256 MiB chunk reuses this registered kernel.
    if (hasChild) {
        for (uint32_t segmentIndex =
            CHAIN4_PIPELINE_SEGMENT_COUNT - CHAIN_PIPELINE_SLOT_COUNT;
            segmentIndex < CHAIN4_PIPELINE_SEGMENT_COUNT; ++segmentIndex) {
            const uint32_t ackNotify = CHANNEL_NOTIFY_CHAIN_ACK_BASE +
                segmentIndex % CHAIN_PIPELINE_SLOT_COUNT;
            CCU_CHECK(NotifyWait(
                kernelArg->peerChannels[childPeerIdx], ackNotify));
        }
    }

    // Keep an all-rank completion barrier after the tail has consumed the last
    // segment. It also gives every rank identical launch-to-launch ordering.
    for (uint32_t peerIdx = 0; peerIdx < kernelArg->peerCount; ++peerIdx) {
        CCU_CHECK(NotifyRecord(
            kernelArg->peerChannels[peerIdx], CHANNEL_NOTIFY_CHAIN_POST_SYNC));
    }
    for (uint32_t peerIdx = 0; peerIdx < kernelArg->peerCount; ++peerIdx) {
        CCU_CHECK(NotifyWait(
            kernelArg->peerChannels[peerIdx], CHANNEL_NOTIFY_CHAIN_POST_SYNC));
    }
    return CCU_SUCCESS;
}

CcuResult RunBipartiteScatterAllgather(const CcuKernelArgBase *kernelArg,
    const Variable &localAddressValue, const Variable &localTokenValue,
    Variable &normalSliceLength, Variable &lastSliceLength,
    const std::vector<RemoteAddr> &peerAddresses)
{
    using namespace AscendC::ccu;

    if ((kernelArg->rankSize != 12 && kernelArg->rankSize != 16) ||
        kernelArg->root >= kernelArg->rankSize) {
        return CCU_E_PARA;
    }

    const bool rootInFirstServer = kernelArg->root < FIRST_SERVER_SIZE;
    const uint32_t rootGroupBegin = rootInFirstServer ? 0 : FIRST_SERVER_SIZE;
    const uint32_t rootGroupSize = rootInFirstServer ? FIRST_SERVER_SIZE :
        kernelArg->rankSize - FIRST_SERVER_SIZE;
    const uint32_t scatterGroupBegin = rootInFirstServer ? FIRST_SERVER_SIZE : 0;
    const uint32_t sliceCount = rootInFirstServer ?
        kernelArg->rankSize - FIRST_SERVER_SIZE : FIRST_SERVER_SIZE;
    const bool rankInRootGroup = (kernelArg->myRank < FIRST_SERVER_SIZE) == rootInFirstServer;
    const uint32_t expectedPeerCount = rankInRootGroup ? sliceCount : rootGroupSize;
    if (kernelArg->peerCount != expectedPeerCount || rootGroupSize == 0 || sliceCount == 0) {
        return CCU_E_PARA;
    }

    if (kernelArg->myRank == kernelArg->root) {
        std::vector<LocalAddr> localSlices(kernelArg->peerCount);
        std::vector<RemoteAddr> remoteSlices(kernelArg->peerCount);
        Event transferDone;
        uint16_t scatterMask = 0;
        for (uint32_t sliceIndex = 0; sliceIndex < sliceCount; ++sliceIndex) {
            const uint32_t destinationRank = scatterGroupBegin + sliceIndex;
            const uint32_t peerIdx = FindPeerIndex(kernelArg, destinationRank);
            if (peerIdx >= kernelArg->peerCount) {
                return CCU_E_PARA;
            }
            SetLocalSlice(localSlices[peerIdx], localAddressValue, localTokenValue,
                normalSliceLength, sliceIndex);
            SetRemoteSlice(remoteSlices[peerIdx], peerAddresses[peerIdx],
                normalSliceLength, sliceIndex);
            Variable &sliceLength = sliceIndex + 1 == sliceCount ?
                lastSliceLength : normalSliceLength;
            const uint16_t writeMask = static_cast<uint16_t>(1U << peerIdx);
            CCU_CHECK(Write(kernelArg->peerChannels[peerIdx], remoteSlices[peerIdx],
                localSlices[peerIdx], sliceLength, transferDone, writeMask));
            scatterMask = static_cast<uint16_t>(scatterMask | writeMask);
        }
        CCU_CHECK(EventWait(transferDone, scatterMask));
        for (uint32_t peerIdx = 0; peerIdx < kernelArg->peerCount; ++peerIdx) {
            CCU_CHECK(NotifyRecord(
                kernelArg->peerChannels[peerIdx], CHANNEL_NOTIFY_DATA_READY));
        }
    } else if (!rankInRootGroup) {
        const uint32_t rootPeerIdx = FindPeerIndex(kernelArg, kernelArg->root);
        if (rootPeerIdx >= kernelArg->peerCount) {
            return CCU_E_PARA;
        }
        CCU_CHECK(NotifyWait(
            kernelArg->peerChannels[rootPeerIdx], CHANNEL_NOTIFY_DATA_READY));
    }

    if (!rankInRootGroup) {
        const uint32_t sliceIndex = kernelArg->myRank - scatterGroupBegin;
        std::vector<LocalAddr> localSlices(kernelArg->peerCount);
        std::vector<RemoteAddr> remoteSlices(kernelArg->peerCount);
        Event transferDone;
        uint16_t phase2Mask = 0;
        for (uint32_t peerIdx = 0; peerIdx < kernelArg->peerCount; ++peerIdx) {
            if (kernelArg->peerRanks[peerIdx] == kernelArg->root) {
                continue;
            }
            SetLocalSlice(localSlices[peerIdx], localAddressValue, localTokenValue,
                normalSliceLength, sliceIndex);
            SetRemoteSlice(remoteSlices[peerIdx], peerAddresses[peerIdx],
                normalSliceLength, sliceIndex);
            Variable &sliceLength = sliceIndex + 1 == sliceCount ?
                lastSliceLength : normalSliceLength;
            const uint16_t writeMask = static_cast<uint16_t>(1U << peerIdx);
            CCU_CHECK(Write(kernelArg->peerChannels[peerIdx], remoteSlices[peerIdx],
                localSlices[peerIdx], sliceLength, transferDone, writeMask));
            phase2Mask = static_cast<uint16_t>(phase2Mask | writeMask);
        }
        if (phase2Mask != 0) {
            CCU_CHECK(EventWait(transferDone, phase2Mask));
        }
        for (uint32_t peerIdx = 0; peerIdx < kernelArg->peerCount; ++peerIdx) {
            CCU_CHECK(NotifyRecord(
                kernelArg->peerChannels[peerIdx], CHANNEL_NOTIFY_PHASE2_DONE));
        }

        std::vector<uint32_t> activeOwners;
        activeOwners.reserve(rootGroupSize);
        for (uint32_t currentSlice = 0; currentSlice < sliceCount; ++currentSlice) {
            const uint32_t ownerRank = GetOwnerRank(
                rootGroupBegin, rootGroupSize, kernelArg->root, currentSlice);
            if (std::find(activeOwners.begin(), activeOwners.end(), ownerRank) == activeOwners.end()) {
                activeOwners.push_back(ownerRank);
            }
        }
        for (uint32_t ownerRank : activeOwners) {
            const uint32_t ownerPeerIdx = FindPeerIndex(kernelArg, ownerRank);
            if (ownerPeerIdx >= kernelArg->peerCount) {
                return CCU_E_PARA;
            }
            CCU_CHECK(NotifyWait(
                kernelArg->peerChannels[ownerPeerIdx], CHANNEL_NOTIFY_PHASE3_DONE));
        }
        return CCU_SUCCESS;
    }

    bool ownsAnySlice = false;
    for (uint32_t sliceIndex = 0; sliceIndex < sliceCount; ++sliceIndex) {
        ownsAnySlice = ownsAnySlice || OwnsSlice(kernelArg->myRank, rootGroupBegin,
            rootGroupSize, kernelArg->root, sliceIndex);
    }

    if (ownsAnySlice) {
        std::vector<LocalAddr> localSlices(kernelArg->peerCount);
        std::vector<RemoteAddr> remoteSlices(kernelArg->peerCount);
        Event transferDone;
        for (uint32_t sliceIndex = 0; sliceIndex < sliceCount; ++sliceIndex) {
            if (!OwnsSlice(kernelArg->myRank, rootGroupBegin,
                rootGroupSize, kernelArg->root, sliceIndex)) {
                continue;
            }
            if (kernelArg->myRank != kernelArg->root) {
                const uint32_t sourceRank = scatterGroupBegin + sliceIndex;
                const uint32_t sourcePeerIdx = FindPeerIndex(kernelArg, sourceRank);
                if (sourcePeerIdx >= kernelArg->peerCount) {
                    return CCU_E_PARA;
                }
                CCU_CHECK(NotifyWait(
                    kernelArg->peerChannels[sourcePeerIdx], CHANNEL_NOTIFY_PHASE2_DONE));
            }

            uint16_t phase3Mask = 0;
            const uint32_t sourceRank = scatterGroupBegin + sliceIndex;
            for (uint32_t peerIdx = 0; peerIdx < kernelArg->peerCount; ++peerIdx) {
                if (kernelArg->peerRanks[peerIdx] == sourceRank) {
                    continue;
                }
                SetLocalSlice(localSlices[peerIdx], localAddressValue, localTokenValue,
                    normalSliceLength, sliceIndex);
                SetRemoteSlice(remoteSlices[peerIdx], peerAddresses[peerIdx],
                    normalSliceLength, sliceIndex);
                Variable &sliceLength = sliceIndex + 1 == sliceCount ?
                    lastSliceLength : normalSliceLength;
                const uint16_t writeMask = static_cast<uint16_t>(1U << peerIdx);
                CCU_CHECK(Write(kernelArg->peerChannels[peerIdx], remoteSlices[peerIdx],
                    localSlices[peerIdx], sliceLength, transferDone, writeMask));
                phase3Mask = static_cast<uint16_t>(phase3Mask | writeMask);
            }
            if (phase3Mask != 0) {
                CCU_CHECK(EventWait(transferDone, phase3Mask));
            }
        }
        for (uint32_t peerIdx = 0; peerIdx < kernelArg->peerCount; ++peerIdx) {
            CCU_CHECK(NotifyRecord(
                kernelArg->peerChannels[peerIdx], CHANNEL_NOTIFY_PHASE3_DONE));
        }
    }

    for (uint32_t sliceIndex = 0; sliceIndex < sliceCount; ++sliceIndex) {
        const bool phase2AlreadyConsumed = kernelArg->myRank != kernelArg->root &&
            OwnsSlice(kernelArg->myRank, rootGroupBegin, rootGroupSize,
                kernelArg->root, sliceIndex);
        if (phase2AlreadyConsumed) {
            continue;
        }
        const uint32_t sourceRank = scatterGroupBegin + sliceIndex;
        const uint32_t sourcePeerIdx = FindPeerIndex(kernelArg, sourceRank);
        if (sourcePeerIdx >= kernelArg->peerCount) {
            return CCU_E_PARA;
        }
        CCU_CHECK(NotifyWait(
            kernelArg->peerChannels[sourcePeerIdx], CHANNEL_NOTIFY_PHASE2_DONE));
    }
    return CCU_SUCCESS;
}

uint32_t GetRootServerSliceIndex(uint32_t rank, uint32_t root)
{
    return (rank + FIRST_SERVER_SIZE - root) % FIRST_SERVER_SIZE;
}

CcuResult RunHierarchicalIntraScatter(const CcuKernelArgBase *kernelArg,
    const Variable &localAddressValue, const Variable &localTokenValue,
    Variable &normalSliceLength, Variable &lastSliceLength,
    const std::vector<RemoteAddr> &peerAddresses)
{
    using namespace AscendC::ccu;

    if ((kernelArg->rankSize != 12 && kernelArg->rankSize != 16) ||
        kernelArg->root >= FIRST_SERVER_SIZE ||
        kernelArg->myRank >= FIRST_SERVER_SIZE ||
        kernelArg->peerCount != FIRST_SERVER_SIZE - 1) {
        return CCU_E_PARA;
    }

    if (kernelArg->myRank == kernelArg->root) {
        std::vector<LocalAddr> localSlices(kernelArg->peerCount);
        std::vector<RemoteAddr> remoteSlices(kernelArg->peerCount);
        Event transferDone;
        uint16_t scatterMask = 0;
        for (uint32_t sliceIndex = 1; sliceIndex < FIRST_SERVER_SIZE; ++sliceIndex) {
            const uint32_t destinationRank =
                (kernelArg->root + sliceIndex) % FIRST_SERVER_SIZE;
            const uint32_t peerIdx = FindPeerIndex(kernelArg, destinationRank);
            if (peerIdx >= kernelArg->peerCount) {
                return CCU_E_PARA;
            }
            SetLocalSlice(localSlices[peerIdx], localAddressValue, localTokenValue,
                normalSliceLength, sliceIndex);
            SetRemoteSlice(remoteSlices[peerIdx], peerAddresses[peerIdx],
                normalSliceLength, sliceIndex);
            Variable &sliceLength = sliceIndex + 1 == FIRST_SERVER_SIZE ?
                lastSliceLength : normalSliceLength;
            const uint16_t writeMask = static_cast<uint16_t>(1U << peerIdx);
            CCU_CHECK(Write(kernelArg->peerChannels[peerIdx], remoteSlices[peerIdx],
                localSlices[peerIdx], sliceLength, transferDone, writeMask));
            scatterMask = static_cast<uint16_t>(scatterMask | writeMask);
        }
        CCU_CHECK(EventWait(transferDone, scatterMask));
        for (uint32_t peerIdx = 0; peerIdx < kernelArg->peerCount; ++peerIdx) {
            CCU_CHECK(NotifyRecord(
                kernelArg->peerChannels[peerIdx], CHANNEL_NOTIFY_DATA_READY));
        }
        return CCU_SUCCESS;
    }

    const uint32_t rootPeerIdx = FindPeerIndex(kernelArg, kernelArg->root);
    if (rootPeerIdx >= kernelArg->peerCount) {
        return CCU_E_PARA;
    }
    CCU_CHECK(NotifyWait(
        kernelArg->peerChannels[rootPeerIdx], CHANNEL_NOTIFY_DATA_READY));
    return CCU_SUCCESS;
}

CcuResult RunHierarchicalInterTransfer(const CcuKernelArgBase *kernelArg,
    const Variable &localAddressValue, const Variable &localTokenValue,
    Variable &normalSliceLength, Variable &lastSliceLength,
    const std::vector<RemoteAddr> &peerAddresses)
{
    using namespace AscendC::ccu;

    if ((kernelArg->rankSize != 12 && kernelArg->rankSize != 16) ||
        kernelArg->root >= FIRST_SERVER_SIZE || kernelArg->peerCount == 0) {
        return CCU_E_PARA;
    }

    const bool rankOnRootServer = kernelArg->myRank < FIRST_SERVER_SIZE;
    const uint32_t expectedPeerCount = rankOnRootServer ?
        kernelArg->rankSize - FIRST_SERVER_SIZE : FIRST_SERVER_SIZE;
    if (kernelArg->peerCount != expectedPeerCount) {
        return CCU_E_PARA;
    }

    if (rankOnRootServer) {
        const uint32_t sliceIndex =
            GetRootServerSliceIndex(kernelArg->myRank, kernelArg->root);
        std::vector<LocalAddr> localSlices(kernelArg->peerCount);
        std::vector<RemoteAddr> remoteSlices(kernelArg->peerCount);
        Event transferDone;
        uint16_t transferMask = 0;
        for (uint32_t peerIdx = 0; peerIdx < kernelArg->peerCount; ++peerIdx) {
            if (kernelArg->peerRanks[peerIdx] < FIRST_SERVER_SIZE) {
                return CCU_E_PARA;
            }
            SetLocalSlice(localSlices[peerIdx], localAddressValue, localTokenValue,
                normalSliceLength, sliceIndex);
            SetRemoteSlice(remoteSlices[peerIdx], peerAddresses[peerIdx],
                normalSliceLength, sliceIndex);
            Variable &sliceLength = sliceIndex + 1 == FIRST_SERVER_SIZE ?
                lastSliceLength : normalSliceLength;
            const uint16_t writeMask = static_cast<uint16_t>(1U << peerIdx);
            CCU_CHECK(Write(kernelArg->peerChannels[peerIdx], remoteSlices[peerIdx],
                localSlices[peerIdx], sliceLength, transferDone, writeMask));
            transferMask = static_cast<uint16_t>(transferMask | writeMask);
        }
        CCU_CHECK(EventWait(transferDone, transferMask));
        for (uint32_t peerIdx = 0; peerIdx < kernelArg->peerCount; ++peerIdx) {
            CCU_CHECK(NotifyRecord(
                kernelArg->peerChannels[peerIdx], CHANNEL_NOTIFY_DATA_READY));
        }
        return CCU_SUCCESS;
    }

    for (uint32_t peerIdx = 0; peerIdx < kernelArg->peerCount; ++peerIdx) {
        if (kernelArg->peerRanks[peerIdx] >= FIRST_SERVER_SIZE) {
            return CCU_E_PARA;
        }
        CCU_CHECK(NotifyWait(
            kernelArg->peerChannels[peerIdx], CHANNEL_NOTIFY_DATA_READY));
    }
    return CCU_SUCCESS;
}

CcuResult RunRelayInterTransfer(const CcuKernelArgBase *kernelArg,
    const Variable &localAddressValue, const Variable &localTokenValue,
    Variable &normalSliceLength, Variable &lastSliceLength,
    const std::vector<RemoteAddr> &peerAddresses)
{
    using namespace AscendC::ccu;

    if ((kernelArg->rankSize != 12 && kernelArg->rankSize != 16) ||
        kernelArg->root != 0) {
        return CCU_E_PARA;
    }

    const uint32_t remoteRankCount = kernelArg->rankSize - FIRST_SERVER_SIZE;
    const uint32_t slicesPerRemoteRank = FIRST_SERVER_SIZE / remoteRankCount;
    const bool rankOnSourceServer = kernelArg->myRank < FIRST_SERVER_SIZE;
    if (rankOnSourceServer) {
        if (kernelArg->peerCount != 1 || peerAddresses.size() != 1) {
            return CCU_E_PARA;
        }
        const uint32_t sliceIndex =
            GetRootServerSliceIndex(kernelArg->myRank, kernelArg->root);
        const uint32_t destinationRank =
            FIRST_SERVER_SIZE + sliceIndex / slicesPerRemoteRank;
        if (kernelArg->peerRanks[0] != destinationRank) {
            return CCU_E_PARA;
        }

        LocalAddr localSlice;
        RemoteAddr remoteSlice;
        SetLocalSlice(localSlice, localAddressValue, localTokenValue,
            normalSliceLength, sliceIndex);
        SetRemoteSlice(remoteSlice, peerAddresses[0],
            normalSliceLength, sliceIndex);
        Variable &sliceLength = sliceIndex + 1 == FIRST_SERVER_SIZE ?
            lastSliceLength : normalSliceLength;
        Event transferDone;
        CCU_CHECK(Write(kernelArg->peerChannels[0], remoteSlice, localSlice,
            sliceLength, transferDone, 1));
        CCU_CHECK(EventWait(transferDone, 1));
        CCU_CHECK(NotifyRecord(
            kernelArg->peerChannels[0], CHANNEL_NOTIFY_DATA_READY));
        return CCU_SUCCESS;
    }

    if (kernelArg->peerCount != slicesPerRemoteRank) {
        return CCU_E_PARA;
    }
    const uint32_t firstSlice =
        (kernelArg->myRank - FIRST_SERVER_SIZE) * slicesPerRemoteRank;
    for (uint32_t sliceOffset = 0; sliceOffset < slicesPerRemoteRank; ++sliceOffset) {
        const uint32_t sourceRank =
            (kernelArg->root + firstSlice + sliceOffset) % FIRST_SERVER_SIZE;
        const uint32_t peerIdx = FindPeerIndex(kernelArg, sourceRank);
        if (peerIdx >= kernelArg->peerCount) {
            return CCU_E_PARA;
        }
        CCU_CHECK(NotifyWait(
            kernelArg->peerChannels[peerIdx], CHANNEL_NOTIFY_DATA_READY));
    }
    return CCU_SUCCESS;
}

CcuResult RunHierarchicalIntraAllgather(const CcuKernelArgBase *kernelArg,
    const Variable &localAddressValue, const Variable &localTokenValue,
    Variable &normalSliceLength, Variable &lastSliceLength,
    const std::vector<RemoteAddr> &peerAddresses)
{
    using namespace AscendC::ccu;

    if ((kernelArg->rankSize != 12 && kernelArg->rankSize != 16) ||
        kernelArg->root >= FIRST_SERVER_SIZE ||
        kernelArg->myRank >= FIRST_SERVER_SIZE ||
        kernelArg->peerCount != FIRST_SERVER_SIZE - 1) {
        return CCU_E_PARA;
    }

    const uint32_t ownedSlice =
        GetRootServerSliceIndex(kernelArg->myRank, kernelArg->root);
    Variable ownedLength;
    if (ownedSlice + 1 == FIRST_SERVER_SIZE) {
        ownedLength = lastSliceLength;
    } else {
        ownedLength = normalSliceLength;
    }

    std::vector<LocalAddr> localBlocks(kernelArg->peerCount);
    std::vector<RemoteAddr> remoteBlocks(kernelArg->peerCount);
    Event transferDone;
    uint16_t allgatherMask = 0;
    for (uint32_t peerIdx = 0; peerIdx < kernelArg->peerCount; ++peerIdx) {
        SetLocalSlice(localBlocks[peerIdx], localAddressValue, localTokenValue,
            normalSliceLength, ownedSlice);
        SetRemoteSlice(remoteBlocks[peerIdx], peerAddresses[peerIdx],
            normalSliceLength, ownedSlice);
        const uint16_t writeMask = static_cast<uint16_t>(1U << peerIdx);
        CCU_CHECK(Write(kernelArg->peerChannels[peerIdx], remoteBlocks[peerIdx],
            localBlocks[peerIdx], ownedLength, transferDone, writeMask));
        allgatherMask = static_cast<uint16_t>(allgatherMask | writeMask);
    }
    CCU_CHECK(EventWait(transferDone, allgatherMask));

    for (uint32_t peerIdx = 0; peerIdx < kernelArg->peerCount; ++peerIdx) {
        CCU_CHECK(NotifyRecord(
            kernelArg->peerChannels[peerIdx], CHANNEL_NOTIFY_DATA_READY));
    }
    for (uint32_t peerIdx = 0; peerIdx < kernelArg->peerCount; ++peerIdx) {
        CCU_CHECK(NotifyWait(
            kernelArg->peerChannels[peerIdx], CHANNEL_NOTIFY_DATA_READY));
    }
    return CCU_SUCCESS;
}

CcuResult RunParallelIntraSag(const CcuKernelArgBase *kernelArg,
    const Variable &localAddressValue, const Variable &localTokenValue,
    Variable &normalSliceLength, Variable &lastSliceLength,
    const std::vector<RemoteAddr> &peerAddresses)
{
    using namespace AscendC::ccu;

    if (kernelArg->rankSize != 2 * FIRST_SERVER_SIZE || kernelArg->root != 0 ||
        kernelArg->myRank >= kernelArg->rankSize ||
        kernelArg->peerCount != FIRST_SERVER_SIZE - 1 ||
        peerAddresses.size() != kernelArg->peerCount) {
        return CCU_E_PARA;
    }

    const uint32_t serverBegin = kernelArg->myRank < FIRST_SERVER_SIZE ?
        0 : FIRST_SERVER_SIZE;
    const uint32_t serverEnd = serverBegin + FIRST_SERVER_SIZE;
    for (uint32_t peerIdx = 0; peerIdx < kernelArg->peerCount; ++peerIdx) {
        if (kernelArg->peerRanks[peerIdx] < serverBegin ||
            kernelArg->peerRanks[peerIdx] >= serverEnd ||
            kernelArg->peerRanks[peerIdx] == kernelArg->myRank) {
            return CCU_E_PARA;
        }
    }

    const uint32_t serverRoot = serverBegin;
    if (kernelArg->myRank == serverRoot) {
        std::vector<LocalAddr> localSlices(kernelArg->peerCount);
        std::vector<RemoteAddr> remoteSlices(kernelArg->peerCount);
        Event transferDone;
        uint16_t scatterMask = 0;
        for (uint32_t sliceIndex = 1; sliceIndex < FIRST_SERVER_SIZE; ++sliceIndex) {
            const uint32_t destinationRank = serverBegin + sliceIndex;
            const uint32_t peerIdx = FindPeerIndex(kernelArg, destinationRank);
            if (peerIdx >= kernelArg->peerCount) {
                return CCU_E_PARA;
            }
            SetLocalSlice(localSlices[peerIdx], localAddressValue, localTokenValue,
                normalSliceLength, sliceIndex);
            SetRemoteSlice(remoteSlices[peerIdx], peerAddresses[peerIdx],
                normalSliceLength, sliceIndex);
            Variable &sliceLength = sliceIndex + 1 == FIRST_SERVER_SIZE ?
                lastSliceLength : normalSliceLength;
            const uint16_t writeMask = static_cast<uint16_t>(1U << peerIdx);
            CCU_CHECK(Write(kernelArg->peerChannels[peerIdx], remoteSlices[peerIdx],
                localSlices[peerIdx], sliceLength, transferDone, writeMask));
            scatterMask = static_cast<uint16_t>(scatterMask | writeMask);
        }
        CCU_CHECK(EventWait(transferDone, scatterMask));
        for (uint32_t peerIdx = 0; peerIdx < kernelArg->peerCount; ++peerIdx) {
            CCU_CHECK(NotifyRecord(
                kernelArg->peerChannels[peerIdx], CHANNEL_NOTIFY_DATA_READY));
        }
    } else {
        const uint32_t rootPeerIdx = FindPeerIndex(kernelArg, serverRoot);
        if (rootPeerIdx >= kernelArg->peerCount) {
            return CCU_E_PARA;
        }
        CCU_CHECK(NotifyWait(
            kernelArg->peerChannels[rootPeerIdx], CHANNEL_NOTIFY_DATA_READY));
    }

    const uint32_t ownedSlice = kernelArg->myRank - serverBegin;
    Variable ownedLength;
    if (ownedSlice + 1 == FIRST_SERVER_SIZE) {
        ownedLength = lastSliceLength;
    } else {
        ownedLength = normalSliceLength;
    }

    std::vector<LocalAddr> localBlocks(kernelArg->peerCount);
    std::vector<RemoteAddr> remoteBlocks(kernelArg->peerCount);
    Event transferDone;
    uint16_t allgatherMask = 0;
    for (uint32_t peerIdx = 0; peerIdx < kernelArg->peerCount; ++peerIdx) {
        SetLocalSlice(localBlocks[peerIdx], localAddressValue, localTokenValue,
            normalSliceLength, ownedSlice);
        SetRemoteSlice(remoteBlocks[peerIdx], peerAddresses[peerIdx],
            normalSliceLength, ownedSlice);
        const uint16_t writeMask = static_cast<uint16_t>(1U << peerIdx);
        CCU_CHECK(Write(kernelArg->peerChannels[peerIdx], remoteBlocks[peerIdx],
            localBlocks[peerIdx], ownedLength, transferDone, writeMask));
        allgatherMask = static_cast<uint16_t>(allgatherMask | writeMask);
    }
    CCU_CHECK(EventWait(transferDone, allgatherMask));
    for (uint32_t peerIdx = 0; peerIdx < kernelArg->peerCount; ++peerIdx) {
        CCU_CHECK(NotifyRecord(
            kernelArg->peerChannels[peerIdx], CHANNEL_NOTIFY_PHASE2_DONE));
    }
    for (uint32_t peerIdx = 0; peerIdx < kernelArg->peerCount; ++peerIdx) {
        CCU_CHECK(NotifyWait(
            kernelArg->peerChannels[peerIdx], CHANNEL_NOTIFY_PHASE2_DONE));
    }
    return CCU_SUCCESS;
}

bool IsCrossFirstConfiguration(const CcuKernelArgBase *kernelArg)
{
    return (kernelArg->rankSize == 12 || kernelArg->rankSize == 16) &&
        kernelArg->root == 0;
}

void GetCrossFirstOwnedBlock(const CcuKernelArgBase *kernelArg,
    Variable &normalSliceLength, Variable &lastSliceLength,
    uint32_t &firstSlice, Variable &blockLength)
{
    const uint32_t ownerCount = kernelArg->rankSize - FIRST_SERVER_SIZE;
    const uint32_t sliceCount = FIRST_SERVER_SIZE / ownerCount;
    firstSlice = (kernelArg->myRank - FIRST_SERVER_SIZE) * sliceCount;
    if (firstSlice + 1 == FIRST_SERVER_SIZE) {
        blockLength = lastSliceLength;
    } else {
        blockLength = normalSliceLength;
    }
    for (uint32_t sliceOffset = 1; sliceOffset < sliceCount; ++sliceOffset) {
        if (firstSlice + sliceOffset + 1 == FIRST_SERVER_SIZE) {
            blockLength += lastSliceLength;
        } else {
            blockLength += normalSliceLength;
        }
    }
}

CcuResult RunCrossFirstInterScatter(const CcuKernelArgBase *kernelArg,
    const Variable &localAddressValue, const Variable &localTokenValue,
    Variable &normalSliceLength, Variable &lastSliceLength,
    const std::vector<RemoteAddr> &peerAddresses)
{
    using namespace AscendC::ccu;

    if (!IsCrossFirstConfiguration(kernelArg)) {
        return CCU_E_PARA;
    }
    const uint32_t ownerCount = kernelArg->rankSize - FIRST_SERVER_SIZE;
    const uint32_t slicesPerOwner = FIRST_SERVER_SIZE / ownerCount;

    if (kernelArg->myRank == kernelArg->root) {
        if (kernelArg->peerCount != ownerCount) {
            return CCU_E_PARA;
        }

        std::vector<LocalAddr> localBlocks(kernelArg->peerCount);
        std::vector<RemoteAddr> remoteBlocks(kernelArg->peerCount);
        Event transferDone;
        uint16_t transferMask = 0;
        for (uint32_t peerIdx = 0; peerIdx < kernelArg->peerCount; ++peerIdx) {
            const uint32_t peerRank = kernelArg->peerRanks[peerIdx];
            if (peerRank < FIRST_SERVER_SIZE || peerRank >= kernelArg->rankSize) {
                return CCU_E_PARA;
            }
            const uint32_t firstSlice =
                (peerRank - FIRST_SERVER_SIZE) * slicesPerOwner;
            Variable blockLength;
            if (firstSlice + 1 == FIRST_SERVER_SIZE) {
                blockLength = lastSliceLength;
            } else {
                blockLength = normalSliceLength;
            }
            for (uint32_t sliceOffset = 1; sliceOffset < slicesPerOwner; ++sliceOffset) {
                if (firstSlice + sliceOffset + 1 == FIRST_SERVER_SIZE) {
                    blockLength += lastSliceLength;
                } else {
                    blockLength += normalSliceLength;
                }
            }
            SetLocalSlice(localBlocks[peerIdx], localAddressValue, localTokenValue,
                normalSliceLength, firstSlice);
            SetRemoteSlice(remoteBlocks[peerIdx], peerAddresses[peerIdx],
                normalSliceLength, firstSlice);
            const uint16_t writeMask = static_cast<uint16_t>(1U << peerIdx);
            CCU_CHECK(Write(kernelArg->peerChannels[peerIdx], remoteBlocks[peerIdx],
                localBlocks[peerIdx], blockLength, transferDone, writeMask));
            transferMask = static_cast<uint16_t>(transferMask | writeMask);
        }
        CCU_CHECK(EventWait(transferDone, transferMask));
        for (uint32_t peerIdx = 0; peerIdx < kernelArg->peerCount; ++peerIdx) {
            CCU_CHECK(NotifyRecord(
                kernelArg->peerChannels[peerIdx], CHANNEL_NOTIFY_DATA_READY));
        }
        return CCU_SUCCESS;
    }

    if (kernelArg->myRank < FIRST_SERVER_SIZE ||
        kernelArg->peerCount != 1 || kernelArg->peerRanks[0] != kernelArg->root) {
        return CCU_E_PARA;
    }
    CCU_CHECK(NotifyWait(
        kernelArg->peerChannels[0], CHANNEL_NOTIFY_DATA_READY));
    return CCU_SUCCESS;
}

CcuResult RunCrossFirstInterReturn(const CcuKernelArgBase *kernelArg,
    const Variable &localAddressValue, const Variable &localTokenValue,
    Variable &normalSliceLength, Variable &lastSliceLength,
    const std::vector<RemoteAddr> &peerAddresses)
{
    using namespace AscendC::ccu;

    if (!IsCrossFirstConfiguration(kernelArg) || kernelArg->myRank == kernelArg->root) {
        return CCU_E_PARA;
    }
    const uint32_t ownerCount = kernelArg->rankSize - FIRST_SERVER_SIZE;
    if (kernelArg->myRank < FIRST_SERVER_SIZE) {
        if (kernelArg->peerCount != ownerCount) {
            return CCU_E_PARA;
        }
        for (uint32_t peerIdx = 0; peerIdx < kernelArg->peerCount; ++peerIdx) {
            if (kernelArg->peerRanks[peerIdx] < FIRST_SERVER_SIZE ||
                kernelArg->peerRanks[peerIdx] >= kernelArg->rankSize) {
                return CCU_E_PARA;
            }
            CCU_CHECK(NotifyWait(
                kernelArg->peerChannels[peerIdx], CHANNEL_NOTIFY_DATA_READY));
        }
        return CCU_SUCCESS;
    }

    if (kernelArg->peerCount != FIRST_SERVER_SIZE - 1) {
        return CCU_E_PARA;
    }
    uint32_t firstSlice = 0;
    Variable blockLength;
    GetCrossFirstOwnedBlock(kernelArg, normalSliceLength, lastSliceLength,
        firstSlice, blockLength);

    std::vector<LocalAddr> localBlocks(kernelArg->peerCount);
    std::vector<RemoteAddr> remoteBlocks(kernelArg->peerCount);
    Event transferDone;
    uint16_t transferMask = 0;
    for (uint32_t peerIdx = 0; peerIdx < kernelArg->peerCount; ++peerIdx) {
        const uint32_t peerRank = kernelArg->peerRanks[peerIdx];
        if (peerRank == kernelArg->root || peerRank >= FIRST_SERVER_SIZE) {
            return CCU_E_PARA;
        }
        SetLocalSlice(localBlocks[peerIdx], localAddressValue, localTokenValue,
            normalSliceLength, firstSlice);
        SetRemoteSlice(remoteBlocks[peerIdx], peerAddresses[peerIdx],
            normalSliceLength, firstSlice);
        const uint16_t writeMask = static_cast<uint16_t>(1U << peerIdx);
        CCU_CHECK(Write(kernelArg->peerChannels[peerIdx], remoteBlocks[peerIdx],
            localBlocks[peerIdx], blockLength, transferDone, writeMask));
        transferMask = static_cast<uint16_t>(transferMask | writeMask);
    }
    CCU_CHECK(EventWait(transferDone, transferMask));
    for (uint32_t peerIdx = 0; peerIdx < kernelArg->peerCount; ++peerIdx) {
        CCU_CHECK(NotifyRecord(
            kernelArg->peerChannels[peerIdx], CHANNEL_NOTIFY_DATA_READY));
    }
    return CCU_SUCCESS;
}

CcuResult RunCrossFirstIntraAllgather(const CcuKernelArgBase *kernelArg,
    const Variable &localAddressValue, const Variable &localTokenValue,
    Variable &normalSliceLength, Variable &lastSliceLength,
    const std::vector<RemoteAddr> &peerAddresses)
{
    using namespace AscendC::ccu;

    if (!IsCrossFirstConfiguration(kernelArg) ||
        kernelArg->myRank < FIRST_SERVER_SIZE) {
        return CCU_E_PARA;
    }
    const uint32_t ownerCount = kernelArg->rankSize - FIRST_SERVER_SIZE;
    if (kernelArg->peerCount != ownerCount - 1) {
        return CCU_E_PARA;
    }

    uint32_t firstSlice = 0;
    Variable blockLength;
    GetCrossFirstOwnedBlock(kernelArg, normalSliceLength, lastSliceLength,
        firstSlice, blockLength);

    std::vector<LocalAddr> localBlocks(kernelArg->peerCount);
    std::vector<RemoteAddr> remoteBlocks(kernelArg->peerCount);
    Event transferDone;
    uint16_t transferMask = 0;
    for (uint32_t peerIdx = 0; peerIdx < kernelArg->peerCount; ++peerIdx) {
        const uint32_t peerRank = kernelArg->peerRanks[peerIdx];
        if (peerRank < FIRST_SERVER_SIZE || peerRank >= kernelArg->rankSize ||
            peerRank == kernelArg->myRank) {
            return CCU_E_PARA;
        }
        SetLocalSlice(localBlocks[peerIdx], localAddressValue, localTokenValue,
            normalSliceLength, firstSlice);
        SetRemoteSlice(remoteBlocks[peerIdx], peerAddresses[peerIdx],
            normalSliceLength, firstSlice);
        const uint16_t writeMask = static_cast<uint16_t>(1U << peerIdx);
        CCU_CHECK(Write(kernelArg->peerChannels[peerIdx], remoteBlocks[peerIdx],
            localBlocks[peerIdx], blockLength, transferDone, writeMask));
        transferMask = static_cast<uint16_t>(transferMask | writeMask);
    }
    CCU_CHECK(EventWait(transferDone, transferMask));
    for (uint32_t peerIdx = 0; peerIdx < kernelArg->peerCount; ++peerIdx) {
        CCU_CHECK(NotifyRecord(
            kernelArg->peerChannels[peerIdx], CHANNEL_NOTIFY_PHASE2_DONE));
    }
    for (uint32_t peerIdx = 0; peerIdx < kernelArg->peerCount; ++peerIdx) {
        CCU_CHECK(NotifyWait(
            kernelArg->peerChannels[peerIdx], CHANNEL_NOTIFY_PHASE2_DONE));
    }
    return CCU_SUCCESS;
}

void GetBalancedOwnerBlock(uint32_t ownerRank, Variable &normalSliceLength,
    Variable &lastSliceLength, uint32_t &firstSlice, Variable &blockLength)
{
    constexpr uint32_t OWNER_COUNT = 4;
    constexpr uint32_t SLICES_PER_OWNER = FIRST_SERVER_SIZE / OWNER_COUNT;
    firstSlice = (ownerRank - FIRST_SERVER_SIZE) * SLICES_PER_OWNER;
    if (firstSlice + 1 == FIRST_SERVER_SIZE) {
        blockLength = lastSliceLength;
    } else {
        blockLength = normalSliceLength;
    }
    for (uint32_t sliceOffset = 1; sliceOffset < SLICES_PER_OWNER; ++sliceOffset) {
        if (firstSlice + sliceOffset + 1 == FIRST_SERVER_SIZE) {
            blockLength += lastSliceLength;
        } else {
            blockLength += normalSliceLength;
        }
    }
}

CcuResult RunBalancedCrossScatterRoot(const CcuKernelArgBase *kernelArg,
    const Variable &crossAddressValue, const Variable &localTokenValue,
    Variable &crossNormalSliceLength, Variable &crossLastSliceLength,
    const std::vector<RemoteAddr> &peerAddresses)
{
    using namespace AscendC::ccu;

    if (kernelArg->myRank != kernelArg->root || kernelArg->root != 0 ||
        kernelArg->rankSize != 12 || kernelArg->peerCount != 4 ||
        peerAddresses.size() != kernelArg->peerCount) {
        return CCU_E_PARA;
    }

    std::vector<LocalAddr> localBlocks(kernelArg->peerCount);
    std::vector<RemoteAddr> remoteBlocks(kernelArg->peerCount);
    Event transferDone;
    uint16_t transferMask = 0;
    for (uint32_t peerIdx = 0; peerIdx < kernelArg->peerCount; ++peerIdx) {
        const uint32_t peerRank = kernelArg->peerRanks[peerIdx];
        if (peerRank < FIRST_SERVER_SIZE || peerRank >= kernelArg->rankSize) {
            return CCU_E_PARA;
        }
        uint32_t firstSlice = 0;
        Variable blockLength;
        GetBalancedOwnerBlock(peerRank, crossNormalSliceLength,
            crossLastSliceLength, firstSlice, blockLength);
        SetLocalSlice(localBlocks[peerIdx], crossAddressValue, localTokenValue,
            crossNormalSliceLength, firstSlice);
        SetRemoteSlice(remoteBlocks[peerIdx], peerAddresses[peerIdx],
            crossNormalSliceLength, firstSlice);
        const uint16_t writeMask = static_cast<uint16_t>(1U << peerIdx);
        CCU_CHECK(Write(kernelArg->peerChannels[peerIdx],
            remoteBlocks[peerIdx], localBlocks[peerIdx], blockLength,
            transferDone, writeMask));
        transferMask = static_cast<uint16_t>(transferMask | writeMask);
    }
    CCU_CHECK(EventWait(transferDone, transferMask));
    for (uint32_t peerIdx = 0; peerIdx < kernelArg->peerCount; ++peerIdx) {
        CCU_CHECK(NotifyRecord(kernelArg->peerChannels[peerIdx],
            CHANNEL_NOTIFY_BALANCED_STAGE1_DATA));
    }
    return CCU_SUCCESS;
}

CcuResult RunBalancedCrossScatterReceive(const CcuKernelArgBase *kernelArg,
    const std::vector<RemoteAddr> &peerAddresses)
{
    using namespace AscendC::ccu;

    if (kernelArg->mode != BroadcastKernelMode::BALANCED_INTER_SCATTER_RECV ||
        kernelArg->rankSize != 12 || kernelArg->root != 0 ||
        kernelArg->myRank < FIRST_SERVER_SIZE ||
        kernelArg->myRank >= kernelArg->rankSize ||
        kernelArg->peerCount != 1 ||
        peerAddresses.size() != 1 ||
        kernelArg->peerRanks[0] != kernelArg->root) {
        return CCU_E_PARA;
    }
    CCU_CHECK(NotifyWait(kernelArg->peerChannels[0],
        CHANNEL_NOTIFY_BALANCED_STAGE1_DATA));
    return CCU_SUCCESS;
}

CcuResult RunBalancedInterStage2(const CcuKernelArgBase *kernelArg,
    const Variable &meshAddressValue, const Variable &localTokenValue,
    Variable &meshNormalSliceLength, Variable &meshLastSliceLength,
    const Variable &crossAddressValue, Variable &crossNormalSliceLength,
    Variable &crossLastSliceLength,
    const std::vector<RemoteAddr> &peerAddresses)
{
    using namespace AscendC::ccu;

    if (kernelArg->mode != BroadcastKernelMode::BALANCED_INTER_FUSED ||
        kernelArg->rankSize != 12 || kernelArg->root != 0 ||
        kernelArg->myRank >= kernelArg->rankSize) {
        return CCU_E_PARA;
    }

    const bool rankOnSourceServer = kernelArg->myRank < FIRST_SERVER_SIZE;
    const uint32_t expectedPeerCount =
        rankOnSourceServer ? kernelArg->rankSize - FIRST_SERVER_SIZE :
                             FIRST_SERVER_SIZE;
    if (kernelArg->peerCount != expectedPeerCount ||
        peerAddresses.size() != expectedPeerCount) {
        return CCU_E_PARA;
    }

    if (rankOnSourceServer) {
        const uint32_t ownedSlice =
            GetRootServerSliceIndex(kernelArg->myRank, kernelArg->root);
        Variable ownedLength;
        if (ownedSlice + 1 == FIRST_SERVER_SIZE) {
            ownedLength = meshLastSliceLength;
        } else {
            ownedLength = meshNormalSliceLength;
        }

        std::vector<LocalAddr> localSlices(kernelArg->peerCount);
        std::vector<RemoteAddr> remoteSlices(kernelArg->peerCount);
        Event transferDone;
        uint16_t transferMask = 0;
        for (uint32_t peerIdx = 0; peerIdx < kernelArg->peerCount; ++peerIdx) {
            if (kernelArg->peerRanks[peerIdx] < FIRST_SERVER_SIZE) {
                return CCU_E_PARA;
            }
            SetLocalSlice(localSlices[peerIdx], meshAddressValue, localTokenValue,
                meshNormalSliceLength, ownedSlice);
            SetRemoteSlice(remoteSlices[peerIdx], peerAddresses[peerIdx],
                meshNormalSliceLength, ownedSlice);
            const uint16_t writeMask = static_cast<uint16_t>(1U << peerIdx);
            CCU_CHECK(Write(kernelArg->peerChannels[peerIdx],
                remoteSlices[peerIdx], localSlices[peerIdx], ownedLength,
                transferDone, writeMask));
            transferMask = static_cast<uint16_t>(transferMask | writeMask);
        }
        CCU_CHECK(EventWait(transferDone, transferMask));
        for (uint32_t peerIdx = 0; peerIdx < kernelArg->peerCount; ++peerIdx) {
            CCU_CHECK(NotifyRecord(kernelArg->peerChannels[peerIdx],
                CHANNEL_NOTIFY_DATA_READY));
        }
        if (kernelArg->myRank != kernelArg->root) {
            for (uint32_t peerIdx = 0; peerIdx < kernelArg->peerCount; ++peerIdx) {
                CCU_CHECK(NotifyWait(kernelArg->peerChannels[peerIdx],
                    CHANNEL_NOTIFY_PHASE2_DONE));
            }
        }
        return CCU_SUCCESS;
    }

    std::vector<RemoteAddr> peerCrossAddresses(kernelArg->peerCount);
    for (uint32_t peerIdx = 0; peerIdx < kernelArg->peerCount; ++peerIdx) {
        if (kernelArg->peerRanks[peerIdx] >= FIRST_SERVER_SIZE) {
            return CCU_E_PARA;
        }
        peerCrossAddresses[peerIdx] = peerAddresses[peerIdx];
        for (uint32_t sliceIdx = 0; sliceIdx + 1 < FIRST_SERVER_SIZE; ++sliceIdx) {
            peerCrossAddresses[peerIdx].addr += meshNormalSliceLength;
        }
        peerCrossAddresses[peerIdx].addr += meshLastSliceLength;
        CCU_CHECK(NotifyWait(kernelArg->peerChannels[peerIdx],
            CHANNEL_NOTIFY_DATA_READY));
    }

    uint32_t firstSlice = 0;
    Variable blockLength;
    GetBalancedOwnerBlock(kernelArg->myRank, crossNormalSliceLength,
        crossLastSliceLength, firstSlice, blockLength);
    LocalAddr localBlock;
    SetLocalSlice(localBlock, crossAddressValue, localTokenValue,
        crossNormalSliceLength, firstSlice);

    Event returnDone;
    uint16_t returnMask = 0;
    uint32_t returnCount = 0;
    for (uint32_t peerIdx = 0; peerIdx < kernelArg->peerCount; ++peerIdx) {
        if (kernelArg->peerRanks[peerIdx] == kernelArg->root) {
            continue;
        }
        AddSliceOffset(peerCrossAddresses[peerIdx].addr,
            crossNormalSliceLength, firstSlice);
        const uint16_t writeMask =
            static_cast<uint16_t>(1U << returnCount);
        CCU_CHECK(Write(kernelArg->peerChannels[peerIdx],
            peerCrossAddresses[peerIdx], localBlock, blockLength,
            returnDone, writeMask));
        returnMask = static_cast<uint16_t>(returnMask | writeMask);
        ++returnCount;
    }
    if (returnCount != FIRST_SERVER_SIZE - 1) {
        return CCU_E_PARA;
    }
    CCU_CHECK(EventWait(returnDone, returnMask));
    for (uint32_t peerIdx = 0; peerIdx < kernelArg->peerCount; ++peerIdx) {
        if (kernelArg->peerRanks[peerIdx] != kernelArg->root) {
            CCU_CHECK(NotifyRecord(kernelArg->peerChannels[peerIdx],
                CHANNEL_NOTIFY_PHASE2_DONE));
        }
    }
    return CCU_SUCCESS;
}

CcuResult WriteFullToPeers(const CcuKernelArgBase *kernelArg,
    const Variable &localAddressValue, const Variable &localTokenValue,
    Variable &transferLength, const std::vector<RemoteAddr> &peerAddresses)
{
    using namespace AscendC::ccu;

    LocalAddr localAddress;
    localAddress.addr = localAddressValue;
    localAddress.token = localTokenValue;
    Event transferDone;
    uint16_t transferMask = 0;
    for (uint32_t peerIdx = 0; peerIdx < kernelArg->peerCount; ++peerIdx) {
        const uint16_t writeMask = static_cast<uint16_t>(1U << peerIdx);
        CCU_CHECK(Write(kernelArg->peerChannels[peerIdx], peerAddresses[peerIdx],
            localAddress, transferLength, transferDone, writeMask));
        transferMask = static_cast<uint16_t>(transferMask | writeMask);
    }
    if (transferMask != 0) {
        CCU_CHECK(EventWait(transferDone, transferMask));
    }
    for (uint32_t peerIdx = 0; peerIdx < kernelArg->peerCount; ++peerIdx) {
        CCU_CHECK(NotifyRecord(
            kernelArg->peerChannels[peerIdx], CHANNEL_NOTIFY_DATA_READY));
    }
    return CCU_SUCCESS;
}

CcuResult WriteFullToPeer(const CcuKernelArgBase *kernelArg, uint32_t peerIdx,
    const Variable &localAddressValue, const Variable &localTokenValue,
    Variable &transferLength, const std::vector<RemoteAddr> &peerAddresses)
{
    using namespace AscendC::ccu;

    if (peerIdx >= kernelArg->peerCount) {
        return CCU_E_PARA;
    }
    LocalAddr localAddress;
    localAddress.addr = localAddressValue;
    localAddress.token = localTokenValue;
    Event transferDone;
    constexpr uint16_t WRITE_MASK = 1;
    CCU_CHECK(Write(kernelArg->peerChannels[peerIdx], peerAddresses[peerIdx],
        localAddress, transferLength, transferDone, WRITE_MASK));
    CCU_CHECK(EventWait(transferDone, WRITE_MASK));
    CCU_CHECK(NotifyRecord(
        kernelArg->peerChannels[peerIdx], CHANNEL_NOTIFY_DATA_READY));
    return CCU_SUCCESS;
}

CcuResult WaitFullFromPeer(const CcuKernelArgBase *kernelArg, uint32_t peerRank)
{
    using namespace AscendC::ccu;

    const uint32_t peerIdx = FindPeerIndex(kernelArg, peerRank);
    if (peerIdx >= kernelArg->peerCount) {
        return CCU_E_PARA;
    }
    CCU_CHECK(NotifyWait(
        kernelArg->peerChannels[peerIdx], CHANNEL_NOTIFY_DATA_READY));
    return CCU_SUCCESS;
}

CcuResult RunTwoShotPhase(const CcuKernelArgBase *kernelArg,
    const Variable &localAddressValue, const Variable &localTokenValue,
    Variable &transferLength, const std::vector<RemoteAddr> &peerAddresses)
{
    if (kernelArg->rankSize != 2 * FIRST_SERVER_SIZE || kernelArg->root != 0 ||
        kernelArg->myRank >= kernelArg->rankSize) {
        return CCU_E_PARA;
    }

    const bool intraPhase =
        kernelArg->mode == BroadcastKernelMode::TWO_SHOT_INTRA_BROADCAST;
    const bool interPhase =
        kernelArg->mode == BroadcastKernelMode::TWO_SHOT_INTER_PAIRED;
    if (!intraPhase && !interPhase) {
        return CCU_E_PARA;
    }
    const uint32_t expectedPeerCount = intraPhase ? FIRST_SERVER_SIZE - 1 : 1;
    if (kernelArg->peerCount != expectedPeerCount ||
        peerAddresses.size() != expectedPeerCount) {
        return CCU_E_PARA;
    }

    if (intraPhase) {
        const uint32_t serverRoot = kernelArg->myRank < FIRST_SERVER_SIZE ?
            kernelArg->root : FIRST_SERVER_SIZE;
        if (kernelArg->myRank == serverRoot) {
            return WriteFullToPeers(kernelArg, localAddressValue, localTokenValue,
                transferLength, peerAddresses);
        }
        return WaitFullFromPeer(kernelArg, serverRoot);
    }

    const uint32_t pairedRank = kernelArg->myRank < FIRST_SERVER_SIZE ?
        kernelArg->myRank + FIRST_SERVER_SIZE :
        kernelArg->myRank - FIRST_SERVER_SIZE;
    if (kernelArg->peerRanks[0] != pairedRank) {
        return CCU_E_PARA;
    }
    if (kernelArg->myRank < FIRST_SERVER_SIZE) {
        return WriteFullToPeer(kernelArg, 0, localAddressValue, localTokenValue,
            transferLength, peerAddresses);
    }
    return WaitFullFromPeer(kernelArg, pairedRank);
}

void GetTreePipelineDataNotify(uint32_t chunkIdx, uint32_t &notifyIndex,
    uint16_t &notifyMask)
{
    const uint32_t signalBit = TREE_NOTIFY_PIPELINE_BIT_OFFSET + chunkIdx;
    notifyIndex = signalBit / TREE_NOTIFY_MASK_BITS;
    notifyMask = static_cast<uint16_t>(1U << (signalBit % TREE_NOTIFY_MASK_BITS));
}

CcuResult WriteTreeChunkToChildren(const CcuKernelArgBase *kernelArg,
    const std::vector<RemoteAddr> &childAddresses, const LocalAddr &localAddress,
    const Variable &transferLength, Event &writeDone, uint32_t dataNotifyIndex,
    uint16_t dataNotifyMask, bool forwardingChildrenOnly)
{
    using namespace AscendC::ccu;

    uint16_t allWritesMask = 0;
    for (uint32_t childIdx = 0; childIdx < kernelArg->childCount; ++childIdx) {
        const uint16_t childMask = static_cast<uint16_t>(1U << childIdx);
        CCU_CHECK(Write(kernelArg->childChannels[childIdx], childAddresses[childIdx],
            localAddress, transferLength, writeDone, childMask));
        allWritesMask = static_cast<uint16_t>(allWritesMask | childMask);
    }
    CCU_CHECK(EventWait(writeDone, allWritesMask));

    for (uint32_t childIdx = 0; childIdx < kernelArg->childCount; ++childIdx) {
        if (forwardingChildrenOnly && kernelArg->childNeedsChunkNotify[childIdx] == 0) {
            continue;
        }
        CCU_CHECK(NotifyRecord(
            kernelArg->childChannels[childIdx], dataNotifyIndex, dataNotifyMask));
    }
    return CCU_SUCCESS;
}

} // namespace

CcuResult CcuKernel(CcuKernelArg arg)
{
    using namespace AscendC::ccu;

    auto *kernelArg = static_cast<CcuKernelArgBase *>(arg);
    if (kernelArg == nullptr) {
        return CCU_E_PTR;
    }

    if (kernelArg->pipelineChunkBytes == 0) {
        Variable localAddressValue;
        Variable localTokenValue;
        Variable transferLength;
        CCU_CHECK(LoadArg(localAddressValue, ARG_LOCAL_ADDRESS));
        CCU_CHECK(LoadArg(localTokenValue, ARG_LOCAL_TOKEN));
        CCU_CHECK(LoadArg(transferLength, ARG_TRANSFER_LENGTH));

        LocalAddr localAddress;
        localAddress.addr = localAddressValue;
        localAddress.token = localTokenValue;
        if (kernelArg->hasParent != 0) {
            CCU_CHECK(WriteVariableWithNotify(kernelArg->parentChannel, localAddressValue,
                CHANNEL_VAR_ADDRESS, TREE_NOTIFY_CONTROL, TREE_NOTIFY_ADDRESS_MASK));
            CCU_CHECK(WriteVariableWithNotify(kernelArg->parentChannel, localTokenValue,
                CHANNEL_VAR_TOKEN, TREE_NOTIFY_CONTROL, TREE_NOTIFY_TOKEN_MASK));
        }

        std::vector<RemoteAddr> childAddresses;
        childAddresses.reserve(kernelArg->childCount);
        for (uint32_t childIdx = 0; childIdx < kernelArg->childCount; ++childIdx) {
            const ChannelHandle childChannel = kernelArg->childChannels[childIdx];
            CCU_CHECK(NotifyWait(
                childChannel, TREE_NOTIFY_CONTROL, TREE_NOTIFY_CONTROL_MASK));
            Variable childAddressValue =
                GetResByChannel<Variable>(childChannel, CHANNEL_VAR_ADDRESS);
            Variable childTokenValue =
                GetResByChannel<Variable>(childChannel, CHANNEL_VAR_TOKEN);
            childAddresses.emplace_back();
            childAddresses.back().addr = childAddressValue;
            childAddresses.back().token = childTokenValue;
        }

        if (kernelArg->hasParent != 0) {
            CCU_CHECK(NotifyWait(kernelArg->parentChannel, TREE_NOTIFY_SMALL_DATA));
        }
        if (kernelArg->childCount != 0) {
            Event writeDone;
            CCU_CHECK(WriteTreeChunkToChildren(kernelArg, childAddresses, localAddress,
                transferLength, writeDone, TREE_NOTIFY_SMALL_DATA, 1, false));
        }
        return CCU_SUCCESS;
    }

    // V25 platform-proven large-message mode.  The C++ loops execute while
    // registering the kernel and emit one straight-line instruction stream.
    Variable localAddressValue;
    Variable localTokenValue;
    CCU_CHECK(LoadArg(localAddressValue, ARG_LOCAL_ADDRESS));
    CCU_CHECK(LoadArg(localTokenValue, ARG_LOCAL_TOKEN));

    LocalAddr localAddress;
    localAddress.addr = localAddressValue;
    localAddress.token = localTokenValue;
    if (kernelArg->hasParent != 0) {
        CCU_CHECK(WriteVariableWithNotify(kernelArg->parentChannel, localAddressValue,
            CHANNEL_VAR_ADDRESS, TREE_NOTIFY_CONTROL, TREE_NOTIFY_ADDRESS_MASK));
        CCU_CHECK(WriteVariableWithNotify(kernelArg->parentChannel, localTokenValue,
            CHANNEL_VAR_TOKEN, TREE_NOTIFY_CONTROL, TREE_NOTIFY_TOKEN_MASK));
    }

    std::vector<RemoteAddr> childAddresses;
    childAddresses.reserve(kernelArg->childCount);
    for (uint32_t childIdx = 0; childIdx < kernelArg->childCount; ++childIdx) {
        const ChannelHandle childChannel = kernelArg->childChannels[childIdx];
        CCU_CHECK(NotifyWait(childChannel, TREE_NOTIFY_CONTROL, TREE_NOTIFY_CONTROL_MASK));
        Variable childAddressValue =
            GetResByChannel<Variable>(childChannel, CHANNEL_VAR_ADDRESS);
        Variable childTokenValue =
            GetResByChannel<Variable>(childChannel, CHANNEL_VAR_TOKEN);
        childAddresses.emplace_back();
        childAddresses.back().addr = childAddressValue;
        childAddresses.back().token = childTokenValue;
    }

    if (kernelArg->childCount == 0) {
        uint32_t notifyIndex = 0;
        uint16_t notifyMask = 0;
        GetTreePipelineDataNotify(0, notifyIndex, notifyMask);
        CCU_CHECK(NotifyWait(kernelArg->parentChannel, notifyIndex, notifyMask));
        return CCU_SUCCESS;
    }

    Variable chunkLength;
    Variable addressStep;
    chunkLength = kernelArg->pipelineChunkBytes;
    addressStep = kernelArg->pipelineChunkBytes;
    Event writeDone;
    for (uint32_t chunkIdx = 0; chunkIdx < kernelArg->pipelineFullChunks; ++chunkIdx) {
        uint32_t notifyIndex = 0;
        uint16_t notifyMask = 0;
        GetTreePipelineDataNotify(chunkIdx, notifyIndex, notifyMask);
        if (kernelArg->hasParent != 0) {
            CCU_CHECK(NotifyWait(kernelArg->parentChannel, notifyIndex, notifyMask));
        }
        CCU_CHECK(WriteTreeChunkToChildren(kernelArg, childAddresses, localAddress,
            chunkLength, writeDone, notifyIndex, notifyMask, true));
        localAddress.addr += addressStep;
        for (uint32_t childIdx = 0; childIdx < kernelArg->childCount; ++childIdx) {
            childAddresses[childIdx].addr += addressStep;
        }
    }

    if (kernelArg->pipelineTailBytes != 0) {
        Variable tailLength;
        tailLength = kernelArg->pipelineTailBytes;
        const uint32_t chunkIdx = kernelArg->pipelineFullChunks;
        uint32_t notifyIndex = 0;
        uint16_t notifyMask = 0;
        GetTreePipelineDataNotify(chunkIdx, notifyIndex, notifyMask);
        if (kernelArg->hasParent != 0) {
            CCU_CHECK(NotifyWait(kernelArg->parentChannel, notifyIndex, notifyMask));
        }
        CCU_CHECK(WriteTreeChunkToChildren(kernelArg, childAddresses, localAddress,
            tailLength, writeDone, notifyIndex, notifyMask, true));
    }

    uint32_t finalNotifyIndex = 0;
    uint16_t finalNotifyMask = 0;
    GetTreePipelineDataNotify(0, finalNotifyIndex, finalNotifyMask);
    for (uint32_t childIdx = 0; childIdx < kernelArg->childCount; ++childIdx) {
        if (kernelArg->childNeedsChunkNotify[childIdx] == 0) {
            CCU_CHECK(NotifyRecord(kernelArg->childChannels[childIdx],
                finalNotifyIndex, finalNotifyMask));
        }
    }
    return CCU_SUCCESS;
}

CcuResult CcuScatterAllgatherKernel(CcuKernelArg arg)
{
    using namespace AscendC::ccu;

    auto *kernelArg = static_cast<CcuKernelArgBase *>(arg);
    if (kernelArg == nullptr || kernelArg->peerCount == 0 ||
        kernelArg->peerCount > MAX_RANK_SIZE) {
        return CCU_E_PTR;
    }

    Variable localAddressValue;
    Variable localTokenValue;
    Variable normalSliceLength;
    Variable lastSliceLength;
    CCU_CHECK(LoadArg(localAddressValue, ARG_LOCAL_ADDRESS));
    CCU_CHECK(LoadArg(localTokenValue, ARG_LOCAL_TOKEN));
    CCU_CHECK(LoadArg(normalSliceLength, ARG_NORMAL_SLICE_LENGTH));
    CCU_CHECK(LoadArg(lastSliceLength, ARG_LAST_SLICE_LENGTH));

    std::vector<RemoteAddr> peerAddresses;
    CCU_CHECK(ExchangePeerAddresses(
        kernelArg, localAddressValue, localTokenValue, peerAddresses));

    if (kernelArg->mode == BroadcastKernelMode::MESH_SCATTER_ALLGATHER) {
        return RunMeshScatterAllgather(kernelArg, localAddressValue, localTokenValue,
            normalSliceLength, lastSliceLength, peerAddresses);
    }
    if (kernelArg->mode == BroadcastKernelMode::PIPELINED_CHAIN4) {
        return RunPipelinedChain4(kernelArg, localAddressValue, localTokenValue,
            normalSliceLength, lastSliceLength, peerAddresses);
    }
    if (kernelArg->mode == BroadcastKernelMode::BIPARTITE_SCATTER_ALLGATHER) {
        return RunBipartiteScatterAllgather(kernelArg, localAddressValue, localTokenValue,
            normalSliceLength, lastSliceLength, peerAddresses);
    }
    return CCU_E_PARA;
}

CcuResult CcuHierarchicalKernel(CcuKernelArg arg)
{
    using namespace AscendC::ccu;

    auto *kernelArg = static_cast<CcuKernelArgBase *>(arg);
    if (kernelArg == nullptr || kernelArg->peerCount == 0 ||
        kernelArg->peerCount > MAX_RANK_SIZE) {
        return CCU_E_PTR;
    }

    Variable localAddressValue;
    Variable localTokenValue;
    Variable normalSliceLength;
    Variable lastSliceLength;
    CCU_CHECK(LoadArg(localAddressValue, ARG_LOCAL_ADDRESS));
    CCU_CHECK(LoadArg(localTokenValue, ARG_LOCAL_TOKEN));
    CCU_CHECK(LoadArg(normalSliceLength, ARG_NORMAL_SLICE_LENGTH));
    CCU_CHECK(LoadArg(lastSliceLength, ARG_LAST_SLICE_LENGTH));

    std::vector<RemoteAddr> peerAddresses;
    CCU_CHECK(ExchangePeerAddresses(
        kernelArg, localAddressValue, localTokenValue, peerAddresses));

    if (kernelArg->mode == BroadcastKernelMode::HIERARCHICAL_INTRA_SCATTER) {
        return RunHierarchicalIntraScatter(kernelArg, localAddressValue, localTokenValue,
            normalSliceLength, lastSliceLength, peerAddresses);
    }
    if (kernelArg->mode == BroadcastKernelMode::HIERARCHICAL_INTER_TRANSFER) {
        return RunHierarchicalInterTransfer(kernelArg, localAddressValue, localTokenValue,
            normalSliceLength, lastSliceLength, peerAddresses);
    }
    if (kernelArg->mode == BroadcastKernelMode::HIERARCHICAL_INTRA_ALLGATHER) {
        return RunHierarchicalIntraAllgather(kernelArg, localAddressValue, localTokenValue,
            normalSliceLength, lastSliceLength, peerAddresses);
    }
    if (kernelArg->mode == BroadcastKernelMode::PARALLEL_INTRA_SAG) {
        return RunParallelIntraSag(kernelArg, localAddressValue, localTokenValue,
            normalSliceLength, lastSliceLength, peerAddresses);
    }
    if (kernelArg->mode == BroadcastKernelMode::RELAY_INTER_TRANSFER) {
        return RunRelayInterTransfer(kernelArg, localAddressValue, localTokenValue,
            normalSliceLength, lastSliceLength, peerAddresses);
    }
    if (kernelArg->mode == BroadcastKernelMode::CROSS_FIRST_INTER_SCATTER) {
        return RunCrossFirstInterScatter(kernelArg, localAddressValue, localTokenValue,
            normalSliceLength, lastSliceLength, peerAddresses);
    }
    if (kernelArg->mode == BroadcastKernelMode::CROSS_FIRST_INTER_RETURN) {
        return RunCrossFirstInterReturn(kernelArg, localAddressValue, localTokenValue,
            normalSliceLength, lastSliceLength, peerAddresses);
    }
    if (kernelArg->mode == BroadcastKernelMode::CROSS_FIRST_INTRA_ALLGATHER) {
        return RunCrossFirstIntraAllgather(kernelArg, localAddressValue, localTokenValue,
            normalSliceLength, lastSliceLength, peerAddresses);
    }
    return CCU_E_PARA;
}

CcuResult CcuTwoShotKernel(CcuKernelArg arg)
{
    using namespace AscendC::ccu;

    auto *kernelArg = static_cast<CcuKernelArgBase *>(arg);
    if (kernelArg == nullptr || kernelArg->peerCount == 0 ||
        kernelArg->peerCount > MAX_RANK_SIZE) {
        return CCU_E_PTR;
    }

    Variable localAddressValue;
    Variable localTokenValue;
    Variable transferLength;
    CCU_CHECK(LoadArg(localAddressValue, ARG_LOCAL_ADDRESS));
    CCU_CHECK(LoadArg(localTokenValue, ARG_LOCAL_TOKEN));
    CCU_CHECK(LoadArg(transferLength, ARG_TRANSFER_LENGTH));

    std::vector<RemoteAddr> peerAddresses;
    CCU_CHECK(ExchangePeerAddresses(
        kernelArg, localAddressValue, localTokenValue, peerAddresses));
    return RunTwoShotPhase(kernelArg, localAddressValue, localTokenValue,
        transferLength, peerAddresses);
}

CcuResult CcuBalancedPlaneKernel(CcuKernelArg arg)
{
    using namespace AscendC::ccu;

    auto *kernelArg = static_cast<CcuKernelArgBase *>(arg);
    if (kernelArg == nullptr || kernelArg->peerCount == 0 ||
        kernelArg->peerCount > MAX_RANK_SIZE) {
        return CCU_E_PTR;
    }

    Variable meshAddressValue;
    Variable localTokenValue;
    Variable meshNormalSliceLength;
    Variable meshLastSliceLength;
    Variable crossAddressValue;
    Variable crossNormalSliceLength;
    Variable crossLastSliceLength;
    CCU_CHECK(LoadArg(meshAddressValue, ARG_LOCAL_ADDRESS));
    CCU_CHECK(LoadArg(localTokenValue, ARG_LOCAL_TOKEN));
    CCU_CHECK(LoadArg(meshNormalSliceLength, ARG_NORMAL_SLICE_LENGTH));
    CCU_CHECK(LoadArg(meshLastSliceLength, ARG_LAST_SLICE_LENGTH));
    CCU_CHECK(LoadArg(crossAddressValue, ARG_CROSS_ADDRESS));
    CCU_CHECK(LoadArg(crossNormalSliceLength,
        ARG_CROSS_NORMAL_SLICE_LENGTH));
    CCU_CHECK(LoadArg(crossLastSliceLength,
        ARG_CROSS_LAST_SLICE_LENGTH));

    if (kernelArg->mode ==
        BroadcastKernelMode::BALANCED_INTER_SCATTER_RECV) {
        std::vector<RemoteAddr> crossPeerAddresses;
        CCU_CHECK(ExchangePeerAddressesAt(kernelArg, crossAddressValue,
            localTokenValue, CHANNEL_NOTIFY_BALANCED_STAGE1_ADDRESS,
            CHANNEL_NOTIFY_BALANCED_STAGE1_TOKEN,
            crossPeerAddresses));
        return RunBalancedCrossScatterReceive(
            kernelArg, crossPeerAddresses);
    }
    if (kernelArg->mode != BroadcastKernelMode::BALANCED_INTER_FUSED) {
        return CCU_E_PARA;
    }

    if (kernelArg->myRank == kernelArg->root) {
        std::vector<RemoteAddr> crossPeerAddresses;
        CCU_CHECK(ExchangePeerAddressesAt(kernelArg, crossAddressValue,
            localTokenValue, CHANNEL_NOTIFY_BALANCED_STAGE1_ADDRESS,
            CHANNEL_NOTIFY_BALANCED_STAGE1_TOKEN,
            crossPeerAddresses));
        CCU_CHECK(RunBalancedCrossScatterRoot(kernelArg,
            crossAddressValue, localTokenValue, crossNormalSliceLength,
            crossLastSliceLength, crossPeerAddresses));
    }

    std::vector<RemoteAddr> meshPeerAddresses;
    CCU_CHECK(ExchangePeerAddresses(kernelArg, meshAddressValue,
        localTokenValue, meshPeerAddresses));
    return RunBalancedInterStage2(kernelArg, meshAddressValue,
        localTokenValue, meshNormalSliceLength, meshLastSliceLength,
        crossAddressValue, crossNormalSliceLength,
        crossLastSliceLength, meshPeerAddresses);
}

#undef CCU_CHECK

} // namespace ops_hccl
