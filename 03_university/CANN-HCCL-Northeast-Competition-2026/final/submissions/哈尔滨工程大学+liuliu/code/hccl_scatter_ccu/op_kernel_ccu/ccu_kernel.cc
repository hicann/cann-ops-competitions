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
#include "ccu_kernel.h"

// Kernel functions return CcuResult; the template's CHK_RET_CCU is Host-only.
#define SCATTER_CCU_TRY(call) \
    do { \
        const CcuResult result = (call); \
        if (result != CCU_SUCCESS) { \
            return result; \
        } \
    } while (0)

namespace ops_hccl {
namespace ccu = AscendC::ccu;
CcuResult CcuKernel(CcuKernelArg arg)
{
    auto *cfg = static_cast<CcuKernelArgBase *>(arg);
    if (cfg == nullptr || cfg->rankSize == 0 || cfg->rankSize > MAX_RANK_SIZE || cfg->myRank >= cfg->rankSize
        || cfg->channelCount >= cfg->rankSize || cfg->rootRank >= cfg->rankSize) {
        return CCU_E_PARA;
    }
    uint16_t transferMask = 0;
    for (uint32_t ch = 0; ch < cfg->channelCount; ++ch) {
        const uint32_t peer = cfg->peerRanks[ch];
        if (peer >= cfg->rankSize || peer == cfg->myRank || (transferMask & (1U << peer)) != 0) {
            return CCU_E_PARA;
        }
        transferMask |= static_cast<uint16_t>(1U << peer);
    }
    ccu::Variable output, outputToken;
    std::vector<ccu::Variable> peerOutput(cfg->channelCount), peerToken(cfg->channelCount);
    for (uint32_t i = 0; i < cfg->channelCount; ++i) {
        peerOutput[i] = ccu::GetResByChannel<ccu::Variable>(cfg->channels[i], 0);
        peerToken[i] = ccu::GetResByChannel<ccu::Variable>(cfg->channels[i], 1);
    }
    // Resource contexts are cached separately for each root rank.
    if (cfg->rootRank == cfg->myRank) {
        ccu::Variable input, inputToken, bytes, stride, copyOwn;
        SCATTER_CCU_TRY(ccu::LoadArg(input, 0));
        SCATTER_CCU_TRY(ccu::LoadArg(output, 1));
        SCATTER_CCU_TRY(ccu::LoadArg(inputToken, 2));
        SCATTER_CCU_TRY(ccu::LoadArg(outputToken, 3));
        SCATTER_CCU_TRY(ccu::LoadArg(bytes, 4));
        SCATTER_CCU_TRY(ccu::LoadArg(stride, 5));
        SCATTER_CCU_TRY(ccu::LoadArg(copyOwn, 6));
        std::vector<ccu::LocalAddr> sources(cfg->channelCount);
        ccu::LocalAddr ownSource;
        std::vector<ccu::RemoteAddr> targets(cfg->channelCount);
        ccu::Variable cursor;
        cursor = input;
        ccu::Event transfers, ownDone;
        const bool narrowSources = cfg->rankSize == 12 && cfg->singleChunk;
        const bool preparesOwn = !narrowSources || cfg->groupIndex == 0;
        uint32_t firstRank = 0, endRank = cfg->rankSize;
        if (narrowSources) {
            firstRank = cfg->rankSize;
            endRank = 0;
            for (uint32_t ch = 0; ch < cfg->channelCount; ++ch) {
                firstRank = std::min(firstRank, cfg->peerRanks[ch]);
                endRank = std::max(endRank, cfg->peerRanks[ch] + 1);
            }
            if (preparesOwn) {
                firstRank = std::min(firstRank, cfg->myRank);
                endRank = std::max(endRank, cfg->myRank + 1);
            }
            if (firstRank >= endRank) {
                return CCU_E_PARA;
            }
        }
        // The Host passed the first served slice as input for this guarded path.
        for (uint32_t peer = firstRank; peer < endRank; ++peer) {
            for (uint32_t ch = 0; ch < cfg->channelCount; ++ch) {
                if (cfg->peerRanks[ch] == peer) {
                    sources[ch].addr = cursor;
                    sources[ch].token = inputToken;
                }
            }
            if (preparesOwn && peer == cfg->myRank) {
                ownSource.addr = cursor;
                ownSource.token = inputToken;
            }
            if (peer + 1 < endRank) {
                cursor += stride;
            }
        }
        if ((cfg->rankSize == 4 || (cfg->rankSize == 12 && preparesOwn)) && cfg->singleChunk) {
            // copyOwn is set only for output disjoint from the entire input.
            CCU_IF(copyOwn != 0)
            {
                ccu::LocalAddr own;
                own.addr = output;
                own.token = outputToken;
                SCATTER_CCU_TRY(ccu::LocalCopy(own, ownSource, bytes, ownDone));
            }
        }
        // A ready destination can start receiving while later peers publish.
        // Every remote address read remains behind its own READY barrier.
        for (uint32_t ch = 0; ch < cfg->channelCount; ++ch) {
            SCATTER_CCU_TRY(ccu::NotifyWait(cfg->channels[ch], 0, 3));
            targets[ch].addr = peerOutput[ch];
            targets[ch].token = peerToken[ch];
            SCATTER_CCU_TRY(ccu::Write(cfg->channels[ch], targets[ch], sources[ch], bytes, transfers,
                static_cast<uint16_t>(1U << cfg->peerRanks[ch])));
        }
        if (((cfg->rankSize != 4 && cfg->rankSize != 12) || !cfg->singleChunk) && preparesOwn) {
            // Retain the large and other-topology copy schedule.
            CCU_IF(copyOwn != 0)
            {
                ccu::LocalAddr own;
                own.addr = output;
                own.token = outputToken;
                SCATTER_CCU_TRY(ccu::LocalCopy(own, ownSource, bytes, ownDone));
                SCATTER_CCU_TRY(ccu::EventWait(ownDone));
            }
        }
        if (cfg->channelCount != 0) {
            SCATTER_CCU_TRY(ccu::EventWait(transfers, transferMask));
        }
        for (uint32_t i = 0; i < cfg->channelCount; ++i) {
            SCATTER_CCU_TRY(ccu::NotifyRecord(cfg->channels[i], 1, 1));
        }
        // A short receiver consumes DONE before its next READY.
        // That READY protects notification reuse without a separate final ACK.
        if ((cfg->rankSize != 4 && cfg->rankSize != 12) || !cfg->singleChunk) {
            for (uint32_t i = 0; i < cfg->channelCount; ++i) {
                SCATTER_CCU_TRY(ccu::NotifyWait(cfg->channels[i], 2, 1));
            }
        }
        if ((cfg->rankSize == 4 || (cfg->rankSize == 12 && preparesOwn)) && cfg->singleChunk) {
            CCU_IF(copyOwn != 0)
            {
                SCATTER_CCU_TRY(ccu::EventWait(ownDone));
            }
        }
    } else {
        SCATTER_CCU_TRY(ccu::LoadArg(output, 0));
        SCATTER_CCU_TRY(ccu::LoadArg(outputToken, 1));
        for (uint32_t ch = 0; ch < cfg->channelCount; ++ch) {
            const uint32_t peer = cfg->peerRanks[ch];
            if (cfg->rootRank == peer) {
                SCATTER_CCU_TRY(ccu::WriteVariableWithNotify(cfg->channels[ch], output, 0, 0, 1));
                SCATTER_CCU_TRY(ccu::WriteVariableWithNotify(cfg->channels[ch], outputToken, 1, 0, 2));
                SCATTER_CCU_TRY(ccu::NotifyWait(cfg->channels[ch], 1, 1));
                if ((cfg->rankSize != 4 && cfg->rankSize != 12) || !cfg->singleChunk) {
                    SCATTER_CCU_TRY(ccu::NotifyRecord(cfg->channels[ch], 2, 1));
                }
            }
        }
    }
    return CCU_SUCCESS;
}

namespace {

    CcuResult PublishDestination(ChannelHandle channel, ccu::Variable address, ccu::Variable token)
    {
        SCATTER_CCU_TRY(ccu::WriteVariableWithNotify(channel, address, 0, 0, 1));
        SCATTER_CCU_TRY(ccu::WriteVariableWithNotify(channel, token, 1, 0, 2));
        return CCU_SUCCESS;
    }

    // Scratch and forwarded-tail receives retain ACK through the default argument.
    // Only guarded final output receives use the next READY as confirmation.
    CcuResult FinishReceive(ChannelHandle channel, bool finalAck = true)
    {
        SCATTER_CCU_TRY(ccu::NotifyWait(channel, 1, 1));
        if (finalAck) {
            SCATTER_CCU_TRY(ccu::NotifyRecord(channel, 2, 1));
        }
        return CCU_SUCCESS;
    }

    CcuResult ReceiveBlocking(ChannelHandle channel, ccu::Variable address, ccu::Variable token, bool finalAck = true)
    {
        SCATTER_CCU_TRY(PublishDestination(channel, address, token));
        SCATTER_CCU_TRY(FinishReceive(channel, finalAck));
        return CCU_SUCCESS;
    }

    CcuResult SendBlocking(ChannelHandle channel, ccu::LocalAddr source, ccu::Variable length)
    {
        SCATTER_CCU_TRY(ccu::NotifyWait(channel, 0, 3));
        ccu::RemoteAddr target;
        target.addr = ccu::GetResByChannel<ccu::Variable>(channel, 0);
        target.token = ccu::GetResByChannel<ccu::Variable>(channel, 1);
        ccu::Event done;
        SCATTER_CCU_TRY(ccu::Write(channel, target, source, length, done));
        SCATTER_CCU_TRY(ccu::EventWait(done));
        SCATTER_CCU_TRY(ccu::NotifyRecord(channel, 1, 1));
        SCATTER_CCU_TRY(ccu::NotifyWait(channel, 2, 1));
        return CCU_SUCCESS;
    }
} // namespace

// Large-message relay kernel.  Every registered mission owns channels from one die only.
// Distribution and forwarding are ordered by Host thread dependencies.
static CcuResult CcuRelayTransferKernel(CcuKernelArg arg, std::vector<ccu::Variable> &taskArgs)
{
    auto *cfg = static_cast<CcuKernelArgBase *>(arg);
    if (cfg == nullptr || cfg->rankSize == 0 || cfg->rankSize > MAX_RANK_SIZE || cfg->myRank >= cfg->rankSize
        || cfg->rootRank >= cfg->rankSize || cfg->channelCount >= cfg->rankSize || cfg->role > SCATTER_ROLE_DIRECT) {
        return CCU_E_PARA;
    }
    uint16_t peerMask = 0;
    for (uint32_t i = 0; i < cfg->channelCount; ++i) {
        const uint32_t peer = cfg->peerRanks[i];
        if (peer >= cfg->rankSize || peer == cfg->myRank || (peerMask & (1U << peer)) != 0) {
            return CCU_E_PARA;
        }
        peerMask |= static_cast<uint16_t>(1U << peer);
    }
    if ((cfg->role == SCATTER_ROLE_HELPER || cfg->role == SCATTER_ROLE_TARGET)
        && (cfg->partnerRank >= cfg->rankSize || cfg->partnerRank == cfg->myRank)) {
        return CCU_E_PARA;
    }
    ccu::Variable &input = taskArgs[0];
    ccu::Variable &output = taskArgs[1];
    ccu::Variable &scratch = taskArgs[2];
    ccu::Variable &inputToken = taskArgs[3];
    ccu::Variable &outputToken = taskArgs[4];
    ccu::Variable &scratchToken = taskArgs[5];
    ccu::Variable &offset = taskArgs[6];
    ccu::Variable &normalLen = taskArgs[7];
    ccu::Variable &relayLen = taskArgs[8];
    ccu::Variable &prefixLen = taskArgs[9];
    ccu::Variable &relayStart = taskArgs[10];
    ccu::Variable &stride = taskArgs[11];
    ccu::Variable &copyOwn = taskArgs[13];

    const bool compact16 = cfg->rankSize == 16 && cfg->singleChunk;
    // Compact single-block arguments guarantee offset zero. Alias the
    // loaded value instead of emitting a redundant ADD instruction.
    ccu::Variable normalOutput = [&]() -> ccu::Variable {
        if (compact16) {
            return output;
        }
        ccu::Variable value;
        value = output + offset;
        return value;
    }();

    if (cfg->role == SCATTER_ROLE_ROOT) {
        // Allocate source addresses only for the peers served by this die mission.
        uint32_t relayIndices[MAX_RANK_SIZE]{}, helperCount = 0;
        for (uint32_t i = 0; i < cfg->channelCount; ++i) {
            if (cfg->peerRoles[cfg->peerRanks[i]] == SCATTER_ROLE_HELPER) {
                relayIndices[i] = helperCount++;
            }
        }
        std::vector<ccu::LocalAddr> normalSources(cfg->channelCount);
        std::vector<ccu::LocalAddr> relaySources(helperCount);
        std::vector<ccu::LocalAddr> ownSources(cfg->groupIndex == 0 ? 1 : 0);
        ccu::Variable rankBase, relayAddress;
        if (compact16) {
            rankBase = input;
        } else {
            rankBase = input + offset;
        }
        for (uint32_t rank = 0; rank < cfg->rankSize; ++rank) {
            // The peer mapping is fixed at registration. Emit address arithmetic
            // only for slices this mission actually sends; no runtime branches.
            bool needsRelay = false;
            for (uint32_t i = 0; i < cfg->channelCount; ++i) {
                const uint32_t peer = cfg->peerRanks[i];
                needsRelay = needsRelay || (cfg->peerRoles[peer] == SCATTER_ROLE_HELPER && cfg->partners[peer] == rank);
            }
            if (needsRelay) {
                relayAddress = rankBase + relayStart;
            }
            for (uint32_t i = 0; i < cfg->channelCount; ++i) {
                const uint32_t peer = cfg->peerRanks[i];
                if (peer == rank) {
                    normalSources[i].addr = rankBase;
                    normalSources[i].token = inputToken;
                }
                if (cfg->peerRoles[peer] == SCATTER_ROLE_HELPER && cfg->partners[peer] == rank) {
                    relaySources[relayIndices[i]].addr = relayAddress;
                    relaySources[relayIndices[i]].token = inputToken;
                }
            }
            if (cfg->groupIndex == 0 && rank == cfg->myRank) {
                ownSources[0].addr = rankBase;
                ownSources[0].token = inputToken;
            }
            if (rank + 1 < cfg->rankSize) {
                rankBase += stride;
            }
        }
        ccu::Event relayDone, directDone, selectedDone, ownDone, copyDone;
        uint16_t relayMask = 0, directMask = 0, selectedMask = 0, ownMask = 0;
        std::vector<ccu::RemoteAddr> targets(cfg->channelCount);
        if (cfg->groupIndex == 0) {
            CCU_IF(copyOwn != 0)
            {
                ccu::LocalAddr destination;
                destination.addr = normalOutput;
                destination.token = outputToken;
                SCATTER_CCU_TRY(ccu::LocalCopy(destination, ownSources[0], normalLen, copyDone));
            }
        }
        CCU_IF(relayLen != 0)
        {
            for (uint32_t i = 0; i < cfg->channelCount; ++i) {
                const uint32_t peer = cfg->peerRanks[i];
                if (cfg->peerRoles[peer] != SCATTER_ROLE_HELPER) {
                    continue;
                }
                SCATTER_CCU_TRY(ccu::NotifyWait(cfg->channels[i], 0, 3));
                targets[i].addr = ccu::GetResByChannel<ccu::Variable>(cfg->channels[i], 0);
                targets[i].token = ccu::GetResByChannel<ccu::Variable>(cfg->channels[i], 1);
                SCATTER_CCU_TRY(ccu::Write(cfg->channels[i], targets[i], relaySources[relayIndices[i]], relayLen,
                    relayDone, static_cast<uint16_t>(1U << i)));
                relayMask |= static_cast<uint16_t>(1U << i);
            }
        }
        for (uint32_t i = 0; i < cfg->channelCount; ++i) {
            const uint32_t peer = cfg->peerRanks[i];
            if (cfg->peerRoles[peer] == SCATTER_ROLE_DIRECT) {
                SCATTER_CCU_TRY(ccu::NotifyWait(cfg->channels[i], 0, 3));
                targets[i].addr = ccu::GetResByChannel<ccu::Variable>(cfg->channels[i], 0);
                targets[i].token = ccu::GetResByChannel<ccu::Variable>(cfg->channels[i], 1);
                SCATTER_CCU_TRY(ccu::Write(cfg->channels[i], targets[i], normalSources[i], normalLen, directDone,
                    static_cast<uint16_t>(1U << i)));
                directMask |= static_cast<uint16_t>(1U << i);
            }
        }
        CCU_IF(prefixLen != 0)
        {
            for (uint32_t i = 0; i < cfg->channelCount; ++i) {
                const uint32_t peer = cfg->peerRanks[i];
                if (cfg->peerRoles[peer] != SCATTER_ROLE_TARGET) {
                    continue;
                }
                SCATTER_CCU_TRY(ccu::NotifyWait(cfg->channels[i], 0, 3));
                targets[i].addr = ccu::GetResByChannel<ccu::Variable>(cfg->channels[i], 0);
                targets[i].token = ccu::GetResByChannel<ccu::Variable>(cfg->channels[i], 1);
                SCATTER_CCU_TRY(ccu::Write(cfg->channels[i], targets[i], normalSources[i], prefixLen, selectedDone,
                    static_cast<uint16_t>(1U << i)));
                selectedMask |= static_cast<uint16_t>(1U << i);
            }
        }
        CCU_IF(relayLen != 0)
        {
            if (relayMask != 0) {
                SCATTER_CCU_TRY(ccu::EventWait(relayDone, relayMask));
            }
        }
        CCU_IF(relayLen != 0)
        {
            for (uint32_t i = 0; i < cfg->channelCount; ++i) {
                if ((relayMask & (1U << i)) == 0) {
                    continue;
                }
                SCATTER_CCU_TRY(ccu::NotifyRecord(cfg->channels[i], 1, 1));
            }
            if (cfg->rankSize != 16 || !cfg->singleChunk) {
                for (uint32_t i = 0; i < cfg->channelCount; ++i) {
                    if ((relayMask & (1U << i)) != 0) {
                        SCATTER_CCU_TRY(ccu::NotifyWait(cfg->channels[i], 2, 1));
                    }
                }
            }
        }
        // Helper-owned data starts as soon as the relay tail is available.  It overlaps both forwarding and
        // the already-issued direct cross-server writes.
        for (uint32_t i = 0; i < cfg->channelCount; ++i) {
            const uint32_t peer = cfg->peerRanks[i];
            if (cfg->peerRoles[peer] != SCATTER_ROLE_HELPER) {
                continue;
            }
            SCATTER_CCU_TRY(ccu::NotifyWait(cfg->channels[i], 0, 3));
            targets[i].addr = ccu::GetResByChannel<ccu::Variable>(cfg->channels[i], 0);
            targets[i].token = ccu::GetResByChannel<ccu::Variable>(cfg->channels[i], 1);
            SCATTER_CCU_TRY(ccu::Write(
                cfg->channels[i], targets[i], normalSources[i], normalLen, ownDone, static_cast<uint16_t>(1U << i)));
            ownMask |= static_cast<uint16_t>(1U << i);
        }
        if (cfg->rankSize == 16 && cfg->singleChunk && relayMask != 0) {
            CCU_IF(relayLen != 0)
            {
                for (uint32_t i = 0; i < cfg->channelCount; ++i) {
                    if ((relayMask & (1U << i)) != 0) {
                        SCATTER_CCU_TRY(ccu::NotifyWait(cfg->channels[i], 2, 1));
                    }
                }
            }
        }
        CCU_IF(prefixLen != 0)
        {
            if (selectedMask != 0) {
                SCATTER_CCU_TRY(ccu::EventWait(selectedDone, selectedMask));
            }
        }
        if (directMask != 0) {
            SCATTER_CCU_TRY(ccu::EventWait(directDone, directMask));
        }
        if (ownMask != 0) {
            SCATTER_CCU_TRY(ccu::EventWait(ownDone, ownMask));
        }
        // Announce all completed outputs before draining final acknowledgements.
        for (uint32_t i = 0; i < cfg->channelCount; ++i) {
            if ((directMask & (1U << i)) != 0) {
                SCATTER_CCU_TRY(ccu::NotifyRecord(cfg->channels[i], 1, 1));
            }
        }
        CCU_IF(prefixLen != 0)
        {
            for (uint32_t i = 0; i < cfg->channelCount; ++i) {
                if ((selectedMask & (1U << i)) != 0) {
                    SCATTER_CCU_TRY(ccu::NotifyRecord(cfg->channels[i], 1, 1));
                }
            }
        }
        for (uint32_t i = 0; i < cfg->channelCount; ++i) {
            if ((ownMask & (1U << i)) != 0) {
                SCATTER_CCU_TRY(ccu::NotifyRecord(cfg->channels[i], 1, 1));
            }
        }
        // Tail ACKs above still protect the scratch/own phase transition.
        // Short final outputs use the receiver's next READY to protect DONE reuse.
        if (cfg->rankSize != 16 || !cfg->singleChunk) {
            for (uint32_t i = 0; i < cfg->channelCount; ++i) {
                if ((directMask & (1U << i)) != 0) {
                    SCATTER_CCU_TRY(ccu::NotifyWait(cfg->channels[i], 2, 1));
                }
            }
            CCU_IF(prefixLen != 0)
            {
                for (uint32_t i = 0; i < cfg->channelCount; ++i) {
                    if ((selectedMask & (1U << i)) != 0) {
                        SCATTER_CCU_TRY(ccu::NotifyWait(cfg->channels[i], 2, 1));
                    }
                }
            }
            for (uint32_t i = 0; i < cfg->channelCount; ++i) {
                if ((ownMask & (1U << i)) != 0) {
                    SCATTER_CCU_TRY(ccu::NotifyWait(cfg->channels[i], 2, 1));
                }
            }
        }
        if (cfg->groupIndex == 0) {
            CCU_IF(copyOwn != 0)
            {
                SCATTER_CCU_TRY(ccu::EventWait(copyDone));
            }
        }
        return CCU_SUCCESS;
    }

    int32_t rootChannel = -1;
    for (uint32_t i = 0; i < cfg->channelCount; ++i) {
        if (cfg->peerRanks[i] == cfg->rootRank) {
            rootChannel = static_cast<int32_t>(i);
        }
    }
    if (cfg->role == SCATTER_ROLE_HELPER && rootChannel >= 0) {
        const ChannelHandle rootCh = cfg->channels[rootChannel];
        CCU_IF(relayLen != 0)
        {
            SCATTER_CCU_TRY(ReceiveBlocking(rootCh, scratch, scratchToken));
        }
        SCATTER_CCU_TRY(ReceiveBlocking(rootCh, normalOutput, outputToken, cfg->rankSize != 16 || !cfg->singleChunk));
    }
    if (cfg->role == SCATTER_ROLE_TARGET && rootChannel >= 0) {
        CCU_IF(prefixLen != 0)
        {
            SCATTER_CCU_TRY(ReceiveBlocking(
                cfg->channels[rootChannel], normalOutput, outputToken, cfg->rankSize != 16 || !cfg->singleChunk));
        }
    }
    if (cfg->role == SCATTER_ROLE_DIRECT && rootChannel >= 0) {
        SCATTER_CCU_TRY(ReceiveBlocking(
            cfg->channels[rootChannel], normalOutput, outputToken, cfg->rankSize != 16 || !cfg->singleChunk));
    }
    return CCU_SUCCESS;
}

// Host thread completion separates distribution from forwarding. No cross-die
// shared CCU event or receiver waiting on a concurrently blocked mission is needed.
static CcuResult CcuRelayForwardKernel(CcuKernelArg arg, std::vector<ccu::Variable> &taskArgs)
{
    auto *cfg = static_cast<CcuKernelArgBase *>(arg);
    if (cfg->role != SCATTER_ROLE_HELPER && cfg->role != SCATTER_ROLE_TARGET) {
        return CCU_SUCCESS;
    }
    for (uint32_t i = 0; i < cfg->channelCount; ++i) {
        if (cfg->peerRanks[i] != cfg->partnerRank) {
            continue;
        }
        CCU_IF(taskArgs[8] != 0)
        {
            if (cfg->role == SCATTER_ROLE_HELPER) {
                ccu::LocalAddr source;
                source.addr = taskArgs[2];
                source.token = taskArgs[5];
                SCATTER_CCU_TRY(SendBlocking(cfg->channels[i], source, taskArgs[8]));
            } else {
                ccu::Variable destination;
                destination = taskArgs[1] + taskArgs[10];
                destination += taskArgs[6];
                SCATTER_CCU_TRY(ReceiveBlocking(cfg->channels[i], destination, taskArgs[4]));
            }
        }
    }
    return CCU_SUCCESS;
}

// On validated relay topologies, the helper receives the tail in a completed mission before
// the Host releases forwarding. Own-data receive and forwarding can then overlap
// across die threads without a shared CCU event. Same-die forwarding publishes
// the own-data destination first so root never waits behind the forward send.
static CcuResult CcuRelayHelperOverlap(CcuKernelArg arg, std::vector<ccu::Variable> &taskArgs)
{
    auto *cfg = static_cast<CcuKernelArgBase *>(arg);
    int32_t rootChannel = -1, partnerChannel = -1;
    for (uint32_t i = 0; i < cfg->channelCount; ++i) {
        if (cfg->peerRanks[i] == cfg->rootRank) {
            rootChannel = static_cast<int32_t>(i);
        }
        if (cfg->peerRanks[i] == cfg->partnerRank) {
            partnerChannel = static_cast<int32_t>(i);
        }
    }
    if (cfg->singleChunk && rootChannel >= 0 && partnerChannel >= 0) {
        // Host selects this mission only when both channels belong to one die.
        // Preserve the two old missions' protocol order within one launch:
        // receive tail -> ACK tail -> publish own output -> forward -> finish own.
        const ChannelHandle rootCh = cfg->channels[rootChannel];
        CCU_IF(taskArgs[8] != 0)
        {
            SCATTER_CCU_TRY(ReceiveBlocking(rootCh, taskArgs[2], taskArgs[5]));
        }
        SCATTER_CCU_TRY(PublishDestination(rootCh, taskArgs[1], taskArgs[4]));
        CCU_IF(taskArgs[8] != 0)
        {
            ccu::LocalAddr source;
            source.addr = taskArgs[2];
            source.token = taskArgs[5];
            SCATTER_CCU_TRY(SendBlocking(cfg->channels[partnerChannel], source, taskArgs[8]));
        }
        SCATTER_CCU_TRY(FinishReceive(rootCh, cfg->rankSize != 16));
        return CCU_SUCCESS;
    }
    if (rootChannel >= 0) {
        const ChannelHandle rootCh = cfg->channels[rootChannel];
        CCU_IF(taskArgs[12] == 2)
        {
            SCATTER_CCU_TRY(ReceiveBlocking(rootCh, taskArgs[2], taskArgs[5]));
            if (cfg->rankSize == 16 && cfg->singleChunk) {
                // The tail is complete and the next destination is already valid.
                // Single-block output has no offset; reuse the loaded arguments.
                SCATTER_CCU_TRY(PublishDestination(rootCh, taskArgs[1], taskArgs[4]));
            }
        }
        CCU_IF(taskArgs[12] == 3)
        {
            ccu::Variable destination = [&]() -> ccu::Variable {
                if (cfg->rankSize == 16 && cfg->singleChunk) {
                    return taskArgs[1];
                }
                ccu::Variable value;
                value = taskArgs[1] + taskArgs[6];
                return value;
            }();
            if (cfg->rankSize == 16 && cfg->singleChunk) {
                // A nonzero tail published this destination in mode 2.
                CCU_IF(taskArgs[8] == 0)
                {
                    SCATTER_CCU_TRY(PublishDestination(rootCh, destination, taskArgs[4]));
                }
            } else {
                SCATTER_CCU_TRY(PublishDestination(rootCh, destination, taskArgs[4]));
            }
            if (partnerChannel >= 0) {
                CCU_IF(taskArgs[8] != 0)
                {
                    ccu::LocalAddr source;
                    source.addr = taskArgs[2];
                    source.token = taskArgs[5];
                    SCATTER_CCU_TRY(SendBlocking(cfg->channels[partnerChannel], source, taskArgs[8]));
                }
            }
            SCATTER_CCU_TRY(FinishReceive(rootCh, cfg->rankSize != 16 || !cfg->singleChunk));
        }
    } else if (partnerChannel >= 0) {
        CCU_IF(taskArgs[12] == 1)
        {
            CCU_IF(taskArgs[8] != 0)
            {
                ccu::LocalAddr source;
                source.addr = taskArgs[2];
                source.token = taskArgs[5];
                SCATTER_CCU_TRY(SendBlocking(cfg->channels[partnerChannel], source, taskArgs[8]));
            }
        }
    }
    return CCU_SUCCESS;
}

// Publish both incoming destinations before waiting on either transfer. Otherwise
// a target with both channels on one thread cannot accept the helper tail while
// its prefix receive is waiting for root's distribution mission to finish.
static CcuResult CcuRelayTargetEager(
    CcuKernelArg arg, std::vector<ccu::Variable> &taskArgs, uint32_t rootChannel, uint32_t partnerChannel)
{
    auto *cfg = static_cast<CcuKernelArgBase *>(arg);
    CCU_IF(taskArgs[12] == 4)
    {
        ccu::Variable prefixDestination = [&]() -> ccu::Variable {
            if (cfg->rankSize == 16 && cfg->singleChunk) {
                return taskArgs[1];
            }
            ccu::Variable value;
            value = taskArgs[1] + taskArgs[6];
            return value;
        }();
        ccu::Variable tailDestination;
        tailDestination = taskArgs[1] + taskArgs[10];
        if (cfg->rankSize != 16 || !cfg->singleChunk) {
            tailDestination += taskArgs[6];
        }
        CCU_IF(taskArgs[9] != 0)
        {
            SCATTER_CCU_TRY(PublishDestination(cfg->channels[rootChannel], prefixDestination, taskArgs[4]));
        }
        CCU_IF(taskArgs[8] != 0)
        {
            SCATTER_CCU_TRY(PublishDestination(cfg->channels[partnerChannel], tailDestination, taskArgs[4]));
        }
        // Root completion is consumed before waiting for the helper; root never
        // depends on helper ACK to release this target's prefix completion.
        CCU_IF(taskArgs[9] != 0)
        {
            SCATTER_CCU_TRY(FinishReceive(cfg->channels[rootChannel], cfg->rankSize != 16 || !cfg->singleChunk));
        }
        CCU_IF(taskArgs[8] != 0)
        {
            SCATTER_CCU_TRY(FinishReceive(cfg->channels[partnerChannel]));
        }
    }
    return CCU_SUCCESS;
}

static CcuResult CcuRelayPhases(CcuKernelArg arg, std::vector<ccu::Variable> &taskArgs)
{
    auto *cfg = static_cast<CcuKernelArgBase *>(arg);
    // A root compact transfer has only mode 0. The deferred alias copy
    // sentinel is already handled by CcuRelayCompactKernel before here.
    if (cfg->rankSize == 16 && cfg->singleChunk && cfg->role == SCATTER_ROLE_ROOT) {
        return CcuRelayTransferKernel(arg, taskArgs);
    }
    if (cfg->role == SCATTER_ROLE_HELPER) {
        return CcuRelayHelperOverlap(arg, taskArgs);
    }
    if (cfg->role == SCATTER_ROLE_TARGET) {
        uint32_t rootChannel = INVALID_VALUE_RANKID, partnerChannel = INVALID_VALUE_RANKID;
        for (uint32_t i = 0; i < cfg->channelCount; ++i) {
            if (cfg->peerRanks[i] == cfg->rootRank) {
                rootChannel = i;
            }
            if (cfg->peerRanks[i] == cfg->partnerRank) {
                partnerChannel = i;
            }
        }
        if (rootChannel != INVALID_VALUE_RANKID && partnerChannel != INVALID_VALUE_RANKID) {
            return CcuRelayTargetEager(arg, taskArgs, rootChannel, partnerChannel);
        }
    }
    CCU_IF(taskArgs[12] == 0)
    {
        SCATTER_CCU_TRY(CcuRelayTransferKernel(arg, taskArgs));
    }
    CCU_IF(taskArgs[12] == 1)
    {
        SCATTER_CCU_TRY(CcuRelayForwardKernel(arg, taskArgs));
    }
    return CCU_SUCCESS;
}

// Copy construction aliases Variable handles; it emits neither new resources
// nor runtime assignments. Keep the validated transfer's internal field layout.
static CcuResult CcuRelayCompactPhases(CcuKernelArg arg, std::vector<ccu::Variable> &raw)
{
    auto *cfg = static_cast<CcuKernelArgBase *>(arg);
    ccu::Variable zero;
    zero = 0;
    const bool isRoot = cfg->role == SCATTER_ROLE_ROOT;
    std::vector<ccu::Variable> taskArgs = {raw[0], raw[1], raw[0], raw[2], raw[3], raw[2], zero, raw[4], raw[5], raw[6],
        raw[6], raw[4], isRoot ? zero : raw[7], isRoot ? raw[7] : zero};
    return CcuRelayPhases(arg, taskArgs);
}

static CcuResult CcuRelayCompactKernel(CcuKernelArg arg)
{
    auto *cfg = static_cast<CcuKernelArgBase *>(arg);
    std::vector<ccu::Variable> raw(SCATTER_COMPACT_TASK_ARGS);
    for (uint32_t i = 0; i < SCATTER_COMPACT_TASK_ARGS; ++i) {
        SCATTER_CCU_TRY(ccu::LoadArg(raw[i], i));
    }
    if (cfg->role != SCATTER_ROLE_ROOT || cfg->groupIndex != 0) {
        return CcuRelayCompactPhases(arg, raw);
    }
    CCU_IF(raw[7] == ~uint64_t{0})
    {
        ccu::LocalAddr source, destination, temporary;
        source.addr = raw[0];
        destination.addr = raw[1];
        temporary.addr = raw[2];
        source.token = raw[3];
        destination.token = raw[4];
        temporary.token = raw[5];
        ccu::Event done;
        SCATTER_CCU_TRY(ccu::LocalCopy(temporary, source, raw[6], done));
        SCATTER_CCU_TRY(ccu::EventWait(done));
        SCATTER_CCU_TRY(ccu::LocalCopy(destination, temporary, raw[6], done));
        SCATTER_CCU_TRY(ccu::EventWait(done));
    }
    CCU_ELSE
    {
        SCATTER_CCU_TRY(CcuRelayCompactPhases(arg, raw));
    }
    return CCU_SUCCESS;
}

CcuResult CcuRelayKernel(CcuKernelArg arg)
{
    auto *cfg = static_cast<CcuKernelArgBase *>(arg);
    if (cfg == nullptr) {
        return CCU_E_PTR;
    }
    if (cfg->singleChunk) {
        return CcuRelayCompactKernel(arg);
    }
    // Each task argument must be loaded exactly once at mission entry. The SDK
    // splits the leading LoadArg instructions across SQEs in groups of eight.
    // Loading arguments inside CCU_IF breaks that layout for multi-SQE tasks.
    std::vector<ccu::Variable> taskArgs(SCATTER_RELAY_TASK_ARGS);
    for (uint32_t i = 0; i < SCATTER_RELAY_TASK_ARGS; ++i) {
        SCATTER_CCU_TRY(ccu::LoadArg(taskArgs[i], i));
    }
    if (cfg->role != SCATTER_ROLE_ROOT || cfg->groupIndex != 0) {
        return CcuRelayPhases(arg, taskArgs);
    }
    CCU_IF(taskArgs[12] == ~uint64_t{0})
    {
        ccu::LocalAddr source, destination, temporary;
        source.addr = taskArgs[0];
        destination.addr = taskArgs[1];
        temporary.addr = taskArgs[2];
        source.token = taskArgs[3];
        destination.token = taskArgs[4];
        temporary.token = taskArgs[5];
        ccu::Event done;
        SCATTER_CCU_TRY(ccu::LocalCopy(temporary, source, taskArgs[6], done));
        SCATTER_CCU_TRY(ccu::EventWait(done));
        SCATTER_CCU_TRY(ccu::LocalCopy(destination, temporary, taskArgs[6], done));
        SCATTER_CCU_TRY(ccu::EventWait(done));
    }
    CCU_ELSE
    {
        SCATTER_CCU_TRY(CcuRelayPhases(arg, taskArgs));
    }
    return CCU_SUCCESS;
}

// Used only for overlapping root buffers, after every remote source has been read.
CcuResult CcuCopyKernel(CcuKernelArg)
{
    ccu::Variable src, dst, scratch, srcToken, dstToken, scratchToken, bytes;
    SCATTER_CCU_TRY(ccu::LoadArg(src, 0));
    SCATTER_CCU_TRY(ccu::LoadArg(dst, 1));
    SCATTER_CCU_TRY(ccu::LoadArg(scratch, 2));
    SCATTER_CCU_TRY(ccu::LoadArg(srcToken, 3));
    SCATTER_CCU_TRY(ccu::LoadArg(dstToken, 4));
    SCATTER_CCU_TRY(ccu::LoadArg(scratchToken, 5));
    SCATTER_CCU_TRY(ccu::LoadArg(bytes, 6));
    ccu::LocalAddr a, b, tmp;
    a.addr = src;
    a.token = srcToken;
    b.addr = dst;
    b.token = dstToken;
    tmp.addr = scratch;
    tmp.token = scratchToken;
    ccu::Event done;
    SCATTER_CCU_TRY(ccu::LocalCopy(tmp, a, bytes, done));
    SCATTER_CCU_TRY(ccu::EventWait(done));
    SCATTER_CCU_TRY(ccu::LocalCopy(b, tmp, bytes, done));
    SCATTER_CCU_TRY(ccu::EventWait(done));
    return CCU_SUCCESS;
}
} // namespace ops_hccl
