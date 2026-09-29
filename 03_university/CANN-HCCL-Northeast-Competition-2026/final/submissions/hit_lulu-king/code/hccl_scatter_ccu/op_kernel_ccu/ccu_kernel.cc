#include <hcomm/hcomm_primitives.h>
#include <algorithm>
#include <vector>

#include "common.h"
#include "custom.h"
#include "ccu_kernel.h"

#ifndef CCU_CHK_RET
#define CCU_CHK_RET(call) \
    do { \
        const CcuResult ret = (call); \
        if (ret != CCU_SUCCESS) { \
            return ret; \
        } \
    } while (0)
#endif

namespace {
constexpr uint32_t OUTPUT_ADDR_ID = 1;
constexpr uint32_t OUTPUT_TOKEN_ID = 2;
constexpr uint32_t COMPLETE_ID = 3;
constexpr uint32_t CKE_INDEX = 0;
constexpr uint16_t OUTPUT_ADDR_MASK = static_cast<uint16_t>(1U << OUTPUT_ADDR_ID);
constexpr uint16_t OUTPUT_TOKEN_MASK = static_cast<uint16_t>(1U << OUTPUT_TOKEN_ID);
constexpr uint16_t OUTPUT_READY_MASK = static_cast<uint16_t>(OUTPUT_ADDR_MASK | OUTPUT_TOKEN_MASK);
constexpr uint64_t PULL_CHUNK_BYTES = 128ULL * 1024 * 1024;
constexpr uint32_t PULL_WINDOW = 2;
constexpr uint16_t COMPLETE_MASK = static_cast<uint16_t>(1U << COMPLETE_ID);
} // namespace

namespace ops_hccl {
CcuResult CcuScatterSmallRootKernel(CcuKernelArg opaque)
{
    using namespace AscendC::ccu;
    auto *arg = static_cast<ScatterCcuKernelArg *>(opaque);
    if (arg == nullptr) { return CCU_E_PTR; }
    if (arg->mode != SCATTER_PEER_PULL || arg->myRank != arg->root ||
        arg->channelCount == 0 || arg->channelCount >= MAX_RANK_SIZE ||
        arg->recvBytes == 0 || arg->recvBytes >= 1024ULL * 1024) { return CCU_E_PARA; }
    Variable sendBase, sendToken, recvBase, recvToken;
    // All root Dies consume the same four launch arguments, including non-copy Dies.
    CCU_CHK_RET(LoadArg(sendBase, 0));
    CCU_CHK_RET(LoadArg(sendToken, 1));
    CCU_CHK_RET(LoadArg(recvBase, 2));
    CCU_CHK_RET(LoadArg(recvToken, 3));
    for (uint32_t i = 0; i < arg->channelCount; ++i) {
        CCU_CHK_RET(WriteVariableWithNotify(arg->channels[i], sendBase,
            OUTPUT_ADDR_ID, CKE_INDEX, OUTPUT_ADDR_MASK));
        CCU_CHK_RET(WriteVariableWithNotify(arg->channels[i], sendToken,
            OUTPUT_TOKEN_ID, CKE_INDEX, OUTPUT_TOKEN_MASK));
    }
    if (arg->copyLocal) {
        LocalAddr source, destination;
        source.addr = sendBase;
        if (arg->root != 0) {
            Variable offset;
            offset = static_cast<uint64_t>(arg->root) * arg->recvBytes;
            source.addr += offset;
        }
        source.token = sendToken;
        destination.addr = recvBase;
        destination.token = recvToken;
        Variable size;
        size = arg->recvBytes;
        Event copied;
        CCU_CHK_RET(LocalCopy(destination, source, size, copied, 1));
        CCU_CHK_RET(EventWait(copied, 1));
    }
    for (uint32_t i = 0; i < arg->channelCount; ++i) {
        CCU_CHK_RET(NotifyWait(arg->channels[i], CKE_INDEX, COMPLETE_MASK));
    }
    return CCU_SUCCESS;
}

CcuResult CcuScatterSmallPeerKernel(CcuKernelArg opaque)
{
    using namespace AscendC::ccu;
    auto *arg = static_cast<ScatterCcuKernelArg *>(opaque);
    if (arg == nullptr) { return CCU_E_PTR; }
    if (arg->mode != SCATTER_PEER_PULL || arg->myRank == arg->root ||
        arg->channelCount != 1 || arg->recvBytes == 0 || arg->recvBytes >= 1024ULL * 1024) {
        return CCU_E_PARA;
    }
    Variable recvBase, recvToken;
    CCU_CHK_RET(LoadArg(recvBase, 0));
    CCU_CHK_RET(LoadArg(recvToken, 1));
    // Prepare local operands before waiting; remote values are read only after READY.
    LocalAddr destination;
    destination.addr = recvBase;
    destination.token = recvToken;
    Variable bytes, offset;
    bytes = arg->recvBytes;
    if (arg->myRank != 0) { offset = static_cast<uint64_t>(arg->myRank) * arg->recvBytes; }
    const ChannelHandle channel = arg->channels[0];
    CCU_CHK_RET(NotifyWait(channel, CKE_INDEX, OUTPUT_READY_MASK));
    RemoteAddr source;
    source.addr = GetResByChannel<Variable>(channel, OUTPUT_ADDR_ID);
    if (arg->myRank != 0) { source.addr += offset; }
    source.token = GetResByChannel<Variable>(channel, OUTPUT_TOKEN_ID);
    Event read;
    CCU_CHK_RET(Read(channel, destination, source, bytes, read, 1));
    CCU_CHK_RET(EventWait(read, 1));
    return NotifyRecord(channel, CKE_INDEX, COMPLETE_MASK);
}

CcuResult CcuScatterRootKernel(CcuKernelArg arg)
{
    using namespace AscendC::ccu;

    auto *kernelArg = static_cast<ScatterCcuKernelArg *>(arg);
    if (kernelArg == nullptr) {
        return CCU_E_PTR;
    }
    if (kernelArg->myRank != kernelArg->root || kernelArg->channelCount == 0 ||
        kernelArg->channelCount >= MAX_RANK_SIZE || kernelArg->recvBytes == 0) {
        return CCU_E_INTERNAL;
    }

    Variable sendBase;
    Variable sendToken;
    CCU_CHK_RET(LoadArg(sendBase, 0));
    CCU_CHK_RET(LoadArg(sendToken, 1));

    Variable recvBase, recvToken;
    // Every root Die receives four launch arguments, including non-copy Dies.
    // CCU validates the number of LoadArg instructions against launch arity.
    CCU_CHK_RET(LoadArg(recvBase, 2));
    CCU_CHK_RET(LoadArg(recvToken, 3));

    Event writeEvent;
    uint64_t chunkOffset = 0;
    while (chunkOffset < kernelArg->recvBytes) {
        const uint64_t remaining = kernelArg->recvBytes - chunkOffset;
        const uint64_t chunkBytes = remaining > MAX_DATA_SIZE ? MAX_DATA_SIZE : remaining;
        uint16_t writeMask = 0;
        if (kernelArg->copyLocal) {
            LocalAddr source, destination;
            source.addr = sendBase;
            Variable sourceOffset;
            sourceOffset = static_cast<uint64_t>(kernelArg->root) * kernelArg->recvBytes + chunkOffset;
            source.addr += sourceOffset;
            source.token = sendToken;
            destination.addr = recvBase;
            if (chunkOffset != 0) {
                Variable destinationOffset;
                destinationOffset = chunkOffset;
                destination.addr += destinationOffset;
            }
            destination.token = recvToken;
            Variable size;
            size = chunkBytes;
            // Bit 15 is disjoint from the at-most-15 remote channel bits.
            CCU_CHK_RET(LocalCopy(destination, source, size, writeEvent, 0x8000));
            writeMask |= 0x8000;
        }
        for (uint32_t idx = 0; idx < kernelArg->channelCount; ++idx) {
            const ChannelHandle channel = kernelArg->channels[idx];
            if (chunkOffset == 0) {
                CCU_CHK_RET(NotifyWait(channel, CKE_INDEX, OUTPUT_READY_MASK));
            }

            LocalAddr source;
            source.addr = sendBase;
            Variable sourceOffset;
            sourceOffset = static_cast<uint64_t>(kernelArg->remoteRanks[idx]) * kernelArg->recvBytes + chunkOffset;
            source.addr += sourceOffset;
            source.token = sendToken;

            RemoteAddr destination;
            destination.addr = GetResByChannel<Variable>(channel, OUTPUT_ADDR_ID);
            if (chunkOffset != 0) {
                Variable destinationOffset;
                destinationOffset = chunkOffset;
                destination.addr += destinationOffset;
            }
            destination.token = GetResByChannel<Variable>(channel, OUTPUT_TOKEN_ID);

            const uint16_t mask = static_cast<uint16_t>(1U << idx);
            Variable size;
            size = chunkBytes;
            CCU_CHK_RET(Write(channel, destination, source, size, writeEvent, mask));
            writeMask = static_cast<uint16_t>(writeMask | mask);
        }
        CCU_CHK_RET(EventWait(writeEvent, writeMask));
        chunkOffset += chunkBytes;
    }

    for (uint32_t idx = 0; idx < kernelArg->channelCount; ++idx) {
        CCU_CHK_RET(NotifyRecord(kernelArg->channels[idx], CKE_INDEX, COMPLETE_MASK));
    }
    return CCU_SUCCESS;
}

CcuResult CcuScatterPeerKernel(CcuKernelArg arg)
{
    using namespace AscendC::ccu;

    auto *kernelArg = static_cast<ScatterCcuKernelArg *>(arg);
    if (kernelArg == nullptr) {
        return CCU_E_PTR;
    }
    if (kernelArg->myRank == kernelArg->root || kernelArg->channelCount != 1 || kernelArg->recvBytes == 0) {
        return CCU_E_INTERNAL;
    }

    Variable recvBase;
    Variable recvToken;
    CCU_CHK_RET(LoadArg(recvBase, 0));
    CCU_CHK_RET(LoadArg(recvToken, 1));

    const ChannelHandle rootChannel = kernelArg->channels[0];
    CCU_CHK_RET(
        WriteVariableWithNotify(rootChannel, recvBase, OUTPUT_ADDR_ID, CKE_INDEX, OUTPUT_ADDR_MASK));
    CCU_CHK_RET(
        WriteVariableWithNotify(rootChannel, recvToken, OUTPUT_TOKEN_ID, CKE_INDEX, OUTPUT_TOKEN_MASK));
    CCU_CHK_RET(NotifyWait(rootChannel, CKE_INDEX, COMPLETE_MASK));
    return CCU_SUCCESS;
}
} // namespace ops_hccl

namespace ops_hccl {
namespace {
using namespace AscendC::ccu;
constexpr uint32_t SCRATCH_ADDR_ID = 4;
constexpr uint32_t SCRATCH_TOKEN_ID = 5;
constexpr uint16_t SCRATCH_READY = (1U << 4) | (1U << 5);
constexpr uint16_t FILLED[2] = {1U << 6, 1U << 7};
constexpr uint16_t RELEASED[2] = {1U << 8, 1U << 9};
constexpr uint32_t WRITER_ID = 0; // Channel provides only XN slots 0..7.
constexpr uint16_t WRITER_READY = 1U << 10;

std::vector<uint32_t> Targets(const ScatterCcuKernelArg &arg, uint32_t helper)
{
    std::vector<uint32_t> result;
    for (uint32_t rank = 0; rank < arg.rankSize; ++rank) {
        if (arg.helperForRank[rank] == helper) {
            result.push_back(rank);
        }
    }
    return result;
}

CcuResult Publish(ChannelHandle channel, Variable addr, Variable token, uint32_t addrId, uint32_t tokenId)
{
    CCU_CHK_RET(WriteVariableWithNotify(channel, addr, addrId, 0, static_cast<uint16_t>(1U << addrId)));
    CCU_CHK_RET(WriteVariableWithNotify(channel, token, tokenId, 0, static_cast<uint16_t>(1U << tokenId)));
    return CCU_SUCCESS;
}

// Every rank acquires the same peer set, but only root/helper payload edges
// advertise a writer. This avoids guessing remote server membership on Host.
CcuResult Announce(const ScatterCcuKernelArg &arg, const std::vector<Variable> &params)
{
    Variable output, token;
    output = params[2];
    token = params[3];
    for (uint32_t i = 0; i < arg.channelCount; ++i) {
        CCU_CHK_RET(Publish(arg.channels[i], output, token, OUTPUT_ADDR_ID, OUTPUT_TOKEN_ID));
        // Root knows the payload graph and never reads a peer writer flag.
        // Omit both sides of that handshake; no notification is left pending.
        if (arg.remoteRanks[i] != arg.root) {
            Variable writer;
            writer = static_cast<uint64_t>(arg.myRank == arg.root ||
                arg.helperForRank[arg.remoteRanks[i]] == arg.myRank);
            CCU_CHK_RET(WriteVariableWithNotify(arg.channels[i], writer, WRITER_ID, 0, WRITER_READY));
        }
    }
    if (arg.myRank != arg.root) {
        for (uint32_t i = 0; i < arg.channelCount; ++i) {
            CCU_CHK_RET(NotifyWait(arg.channels[i], 0, WRITER_READY));
        }
    }
    return CCU_SUCCESS;
}

struct PutOperands {
    LocalAddr source;
    RemoteAddr destination;
    Variable srcOffset, dstOffset, size;
};

CcuResult PutPrepared(PutOperands &operands, ChannelHandle channel, Variable base, Variable token, uint64_t sourceOffset,
    uint32_t dstAddrId, uint32_t dstTokenId, uint64_t destinationOffset,
    uint64_t bytes, Event event, uint16_t bit)
{
    LocalAddr &source = operands.source;
    source.addr = base;
    Variable &srcOffset = operands.srcOffset;
    srcOffset = sourceOffset;
    source.addr += srcOffset;
    source.token = token;
    RemoteAddr &destination = operands.destination;
    destination.addr = GetResByChannel<Variable>(channel, dstAddrId);
    Variable &dstOffset = operands.dstOffset;
    dstOffset = destinationOffset;
    destination.addr += dstOffset;
    destination.token = GetResByChannel<Variable>(channel, dstTokenId);
    Variable &size = operands.size;
    size = bytes;
    return Write(channel, destination, source, size, event, bit);
}

CcuResult Put(ChannelHandle channel, Variable base, Variable token, uint64_t sourceOffset,
    uint32_t dstAddrId, uint32_t dstTokenId, uint64_t destinationOffset,
    uint64_t bytes, Event event, uint16_t bit)
{
    PutOperands operands;
    return PutPrepared(operands, channel, base, token, sourceOffset, dstAddrId, dstTokenId,
        destinationOffset, bytes, event, bit);
}

CcuResult RelayRoot(const ScatterCcuKernelArg &arg, const std::vector<Variable> &params)
{
    CCU_CHK_RET(Announce(arg, params));
    Variable sendBase, sendToken, recvBase, recvToken;
    sendBase = params[0];
    sendToken = params[1];
    recvBase = params[2];
    recvToken = params[3];
    std::vector<std::vector<uint32_t>> targets;
    for (uint32_t i = 0; i < arg.channelCount; ++i) {
        targets.push_back(Targets(arg, arg.remoteRanks[i]));
        if (targets.back().size() > 1) { return CCU_E_PARA; }
    }

    // Each helper gets its relay prefix BEFORE its own final result.
    // Non-helper destinations on this Die can progress immediately.
    constexpr uint64_t directChunk = 128ULL * 1024 * 1024;
    const uint32_t directRounds = (arg.recvBytes + directChunk - 1) / directChunk;
    std::vector<Event> directEvents(directRounds);
    std::vector<uint16_t> directMasks(directRounds, 0);
    // -1 starts the direct paths; a nonnegative index starts just that helper's result.
    auto enqueueDirect = [&](int selectedHelper) -> CcuResult {
        for (uint32_t r = 0; r < directRounds; ++r) {
            const uint64_t offset = static_cast<uint64_t>(r) * directChunk;
            for (uint32_t i = 0; i < arg.channelCount; ++i) {
                if (selectedHelper < 0 ? !targets[i].empty() : i != static_cast<uint32_t>(selectedHelper)) {
                    continue;
                }
                const uint32_t peer = arg.remoteRanks[i];
                const uint64_t start = arg.helperForRank[peer] == INVALID_VALUE_RANKID ? 0 : arg.relayBytes;
                const uint64_t length = arg.recvBytes - start;
                if (offset >= length) { continue; }
                if (offset == 0) { CCU_CHK_RET(NotifyWait(arg.channels[i], 0, OUTPUT_READY_MASK)); }
                const uint16_t bit = static_cast<uint16_t>(1U << i);
                CCU_CHK_RET(Put(arg.channels[i], sendBase, sendToken,
                    static_cast<uint64_t>(peer) * arg.recvBytes + start + offset,
                    OUTPUT_ADDR_ID, OUTPUT_TOKEN_ID, start + offset,
                    std::min(directChunk, length - offset), directEvents[r], bit));
                directMasks[r] |= bit;
            }
            if (selectedHelper < 0 && arg.copyLocal) {
                LocalAddr source, destination;
                source.addr = sendBase;
                Variable sourceOffset;
                sourceOffset = static_cast<uint64_t>(arg.root) * arg.recvBytes + offset;
                source.addr += sourceOffset;
                source.token = sendToken;
                destination.addr = recvBase;
                Variable destinationOffset;
                destinationOffset = offset;
                destination.addr += destinationOffset;
                destination.token = recvToken;
                Variable size;
                size = std::min(directChunk, arg.recvBytes - offset);
                CCU_CHK_RET(LocalCopy(destination, source, size, directEvents[r], 0x8000));
                directMasks[r] |= 0x8000;
            }
        }
        return CCU_SUCCESS;
    };
    const uint64_t rounds = (arg.relayBytes + arg.tileBytes - 1) / arg.tileBytes;
    bool hasHelpers = false;
    for (const auto &list : targets) { hasHelpers = hasHelpers || !list.empty(); }
    if (hasHelpers) {
        Event feedEvent;
        std::vector<std::unique_ptr<PutOperands>> feedOperands(arg.channelCount);
        for (uint32_t i = 0; i < arg.channelCount; ++i) {
            if (!targets[i].empty()) { feedOperands[i] = std::make_unique<PutOperands>(); }
        }
        for (uint64_t r = 0; r < rounds; ++r) {
            const uint32_t bank = r % 2;
            const uint64_t offset = r * arg.tileBytes;
            const uint64_t bytes = std::min(arg.tileBytes, arg.relayBytes - offset);
            for (uint32_t i = 0; i < arg.channelCount; ++i) {
                if (targets[i].empty()) { continue; }
                if (r == 0) { CCU_CHK_RET(NotifyWait(arg.channels[i], 0, SCRATCH_READY)); }
                if (r >= 2) { CCU_CHK_RET(NotifyWait(arg.channels[i], 0, RELEASED[bank])); }
                const uint16_t bit = static_cast<uint16_t>(1U << i);
                CCU_CHK_RET(PutPrepared(*feedOperands[i], arg.channels[i], sendBase, sendToken,
                    static_cast<uint64_t>(targets[i][0]) * arg.recvBytes + offset,
                    SCRATCH_ADDR_ID, SCRATCH_TOKEN_ID, bank * arg.tileBytes, bytes, feedEvent, bit));
            }
            // Queue direct traffic while the first feeds are in flight, not after
            // their completion. Separate physical paths can start concurrently.
            if (r == 0) { CCU_CHK_RET(enqueueDirect(-1)); }
            for (uint32_t i = 0; i < arg.channelCount; ++i) {
                if (!targets[i].empty()) {
                    CCU_CHK_RET(EventWait(feedEvent, static_cast<uint16_t>(1U << i)));
                    CCU_CHK_RET(NotifyRecord(arg.channels[i], 0, FILLED[bank]));
                    // Do not make this helper wait for all other feed completions.
                    if (r + 1 == rounds) { CCU_CHK_RET(enqueueDirect(static_cast<int>(i))); }
                }
            }
        }
    } else {
        CCU_CHK_RET(enqueueDirect(-1));
    }
    for (uint32_t r = 0; r < directRounds; ++r) {
        if (directMasks[r]) { CCU_CHK_RET(EventWait(directEvents[r], directMasks[r])); }
    }
    for (uint32_t i = 0; i < arg.channelCount; ++i) {
        CCU_CHK_RET(NotifyRecord(arg.channels[i], 0, COMPLETE_MASK));
    }
    for (uint32_t i = 0; i < arg.channelCount; ++i) {
        if (!targets[i].empty()) {
            for (uint32_t bank = 0; bank < std::min<uint64_t>(rounds, 2); ++bank) {
                CCU_CHK_RET(NotifyWait(arg.channels[i], 0, RELEASED[bank]));
            }
        }
    }
    return CCU_SUCCESS;
}
CcuResult ResidentHelper(const ScatterCcuKernelArg &arg, const std::vector<Variable> &params)
{
    CCU_CHK_RET(Announce(arg, params));
    // Registration places the root channel first; every channel is on one Die.
    const auto targets = Targets(arg, arg.myRank);
    if (arg.channelCount != targets.size() + 1 || targets.empty() ||
        arg.remoteRanks[0] != arg.root) {
        return CCU_E_PARA;
    }
    Variable output, outputToken, scratch, scratchToken;
    output = params[2];
    outputToken = params[3];
    scratch = params[4];
    scratchToken = params[5];
    const ChannelHandle rootChannel = arg.channels[0];
    CCU_CHK_RET(NotifyWait(rootChannel, 0, OUTPUT_READY_MASK));
    CCU_CHK_RET(Publish(rootChannel, scratch, scratchToken, SCRATCH_ADDR_ID, SCRATCH_TOKEN_ID));
    for (uint32_t i = 1; i < arg.channelCount; ++i) {
        CCU_CHK_RET(NotifyWait(arg.channels[i], 0, OUTPUT_READY_MASK));
    }
    Event forwarded[2];
    uint16_t pending[2] = {0, 0};
    std::vector<PutOperands> forwardOperands(2 * targets.size());
    const uint64_t rounds = (arg.relayBytes + arg.tileBytes - 1) / arg.tileBytes;
    for (uint64_t r = 0; r < rounds; ++r) {
        const uint32_t bank = r % 2;
        const uint64_t offset = r * arg.tileBytes;
        const uint64_t bytes = std::min(arg.tileBytes, arg.relayBytes - offset);
        if (pending[bank] != 0) {
            CCU_CHK_RET(EventWait(forwarded[bank], pending[bank]));
            CCU_CHK_RET(NotifyRecord(rootChannel, 0, RELEASED[bank]));
            pending[bank] = 0;
        }
        CCU_CHK_RET(NotifyWait(rootChannel, 0, FILLED[bank]));
        uint16_t mask = 0;
        for (uint32_t i = 1; i < arg.channelCount; ++i) {
            const auto it = std::find(targets.begin(), targets.end(), arg.remoteRanks[i]);
            if (it == targets.end()) {
                return CCU_E_PARA;
            }
            const uint64_t slot = static_cast<uint64_t>(it - targets.begin());
            const uint16_t bit = static_cast<uint16_t>(1U << (i - 1));
            CCU_CHK_RET(PutPrepared(forwardOperands[bank * targets.size() + slot], arg.channels[i], scratch, scratchToken,
                (bank * targets.size() + slot) * arg.tileBytes,
                OUTPUT_ADDR_ID, OUTPUT_TOKEN_ID, offset,
                bytes, forwarded[bank], bit));
            mask |= bit;
        }
        pending[bank] = mask;
    }
    for (uint32_t bank = 0; bank < 2; ++bank) {
        if (pending[bank] != 0) {
            CCU_CHK_RET(EventWait(forwarded[bank], pending[bank]));
            CCU_CHK_RET(NotifyRecord(rootChannel, 0, RELEASED[bank]));
        }
    }
    for (uint32_t i = 1; i < arg.channelCount; ++i) {
        CCU_CHK_RET(NotifyRecord(arg.channels[i], 0, COMPLETE_MASK));
    }
    return NotifyWait(rootChannel, 0, COMPLETE_MASK);
}
} // namespace

CcuResult RelayDispatch(const ScatterCcuKernelArg *arg, const std::vector<AscendC::ccu::Variable> &params)
{
    using namespace AscendC::ccu;
    if (arg == nullptr) {
        return CCU_E_PTR;
    }
    if (arg->channelCount == 0 || arg->channelCount >= MAX_RANK_SIZE || arg->tileBytes == 0 ||
        arg->relayBytes == 0 || arg->relayBytes >= arg->recvBytes) {
        return CCU_E_PARA;
    }
    if (arg->kind == RELAY_INPUT_ONESHOT || arg->kind == RELAY_OUTPUT_ONESHOT) {
        if (arg->relayBytes != arg->tileBytes) { return CCU_E_PARA; }
        ScatterCcuKernelArg stage = *arg;
        stage.kind = arg->kind == RELAY_INPUT_ONESHOT ? RELAY_INIT_INPUT : RELAY_FORWARD;
        CCU_CHK_RET(RelayDispatch(&stage, params));
        stage.kind = arg->kind == RELAY_INPUT_ONESHOT ? RELAY_WAIT_0 : RELAY_FINISH_OUTPUT;
        return RelayDispatch(&stage, params);
    }
    if (arg->kind == RELAY_ROOT) {
        return RelayRoot(*arg, params);
    }
    if (arg->kind == RELAY_RESIDENT_HELPER) {
        return ResidentHelper(*arg, params);
    }
    if (arg->kind == RELAY_RECEIVER || arg->kind == RELAY_INIT_INPUT) {
        CCU_CHK_RET(Announce(*arg, params));
        for (uint32_t i = 0; i < arg->channelCount; ++i) {
            CCU_CHK_RET(NotifyWait(arg->channels[i], 0, OUTPUT_READY_MASK));
        }
        if (arg->kind == RELAY_INIT_INPUT) {
            Variable scratch, scratchToken;
            scratch = params[4];
            scratchToken = params[5];
            CCU_CHK_RET(Publish(arg->channels[0], scratch, scratchToken, SCRATCH_ADDR_ID, SCRATCH_TOKEN_ID));
        } else {
            for (uint32_t i = 0; i < arg->channelCount; ++i) {
                Variable writer = GetResByChannel<Variable>(arg->channels[i], WRITER_ID);
                CCU_IF(writer == 1) {
                    CCU_CHK_RET(NotifyWait(arg->channels[i], 0, COMPLETE_MASK));
                }
            }
        }
        return CCU_SUCCESS;
    }
    if (arg->kind >= RELAY_WAIT_0 && arg->kind <= RELAY_REUSE_1) {
        const uint32_t bank = (arg->kind == RELAY_WAIT_1 || arg->kind == RELAY_REUSE_1) ? 1 : 0;
        if (arg->kind == RELAY_REUSE_0 || arg->kind == RELAY_REUSE_1) {
            CCU_CHK_RET(NotifyRecord(arg->channels[0], 0, RELEASED[bank]));
        }
        return NotifyWait(arg->channels[0], 0, FILLED[bank]);
    }
    if (arg->kind == RELAY_FINISH_INPUT) {
        const uint64_t rounds = (arg->relayBytes + arg->tileBytes - 1) / arg->tileBytes;
        for (uint32_t bank = 0; bank < std::min<uint64_t>(rounds, 2); ++bank) {
            CCU_CHK_RET(NotifyRecord(arg->channels[0], 0, RELEASED[bank]));
        }
        return NotifyWait(arg->channels[0], 0, COMPLETE_MASK);
    }
    if (arg->kind == RELAY_INIT_OUTPUT || arg->kind == RELAY_FINISH_OUTPUT) {
        if (arg->kind == RELAY_INIT_OUTPUT) { CCU_CHK_RET(Announce(*arg, params)); }
        for (uint32_t i = 0; i < arg->channelCount; ++i) {
            if (arg->kind == RELAY_INIT_OUTPUT) {
                CCU_CHK_RET(NotifyWait(arg->channels[i], 0, OUTPUT_READY_MASK));
            } else {
                CCU_CHK_RET(NotifyRecord(arg->channels[i], 0, COMPLETE_MASK));
            }
        }
        return CCU_SUCCESS;
    }
    if (arg->kind != RELAY_FORWARD) {
        return CCU_E_PARA;
    }
    Variable scratch, scratchToken, offset, bytes, bankOffset;
    scratch = params[4];
    scratchToken = params[5];
    offset = params[6];
    bytes = params[7];
    bankOffset = params[8];
    const auto targets = Targets(*arg, arg->myRank);
    Event event;
    uint16_t mask = 0;
    for (uint32_t i = 0; i < arg->channelCount; ++i) {
        const auto it = std::find(targets.begin(), targets.end(), arg->remoteRanks[i]);
        if (it == targets.end()) {
            return CCU_E_PARA;
        }
        LocalAddr source;
        source.addr = scratch;
        if (arg->relayBytes != arg->tileBytes) {
            source.addr += bankOffset;
        }
        Variable slotOffset;
        if (it != targets.begin()) {
            slotOffset = static_cast<uint64_t>(it - targets.begin()) * arg->tileBytes;
            source.addr += slotOffset;
        }
        source.token = scratchToken;
        RemoteAddr destination;
        destination.addr = GetResByChannel<Variable>(arg->channels[i], OUTPUT_ADDR_ID);
        // Single-block relay owns the prefix and uses bank zero.
        if (arg->relayBytes != arg->tileBytes) {
            destination.addr += offset;
        }
        destination.token = GetResByChannel<Variable>(arg->channels[i], OUTPUT_TOKEN_ID);
        const uint16_t bit = static_cast<uint16_t>(1U << i);
        CCU_CHK_RET(Write(arg->channels[i], destination, source, bytes, event, bit));
        mask |= bit;
    }
    return EventWait(event, mask);
}

CcuResult CcuScatterRelayKernel(CcuKernelArg opaque)
{
    using namespace AscendC::ccu;
    auto *arg = static_cast<ScatterCcuKernelArg *>(opaque);
    if (arg == nullptr) { return CCU_E_PTR; }
    std::vector<Variable> params(10);
    for (uint32_t i = 0; i < params.size(); ++i) { CCU_CHK_RET(LoadArg(params[i], i)); }
    // One mission per ingress/egress thread, not six separate missions.
    if (arg->kind == RELAY_INIT_INPUT || arg->kind == RELAY_INIT_OUTPUT) {
        std::vector<uint32_t> stages;
        if (arg->relayBytes == arg->tileBytes) {
            stages = arg->kind == RELAY_INIT_INPUT
                ? std::vector<uint32_t>{RELAY_INPUT_ONESHOT, RELAY_FINISH_INPUT}
                : std::vector<uint32_t>{RELAY_INIT_OUTPUT, RELAY_OUTPUT_ONESHOT};
        } else {
            const uint32_t last = arg->kind == RELAY_INIT_INPUT ? RELAY_FINISH_INPUT : RELAY_FINISH_OUTPUT;
            for (uint32_t kind = arg->kind; kind <= last; ++kind) { stages.push_back(kind); }
        }
        for (uint32_t kind : stages) {
            ScatterCcuKernelArg stage = *arg;
            stage.kind = kind;
            CCU_IF(params[9] == static_cast<uint64_t>(kind)) {
                CCU_CHK_RET(RelayDispatch(&stage, params));
            }
        }
        return CCU_SUCCESS;
    }
    return RelayDispatch(arg, params);
}
} // namespace ops_hccl

namespace ops_hccl {
CcuResult CcuScatterPullRootKernel(CcuKernelArg arg)
{
    using namespace AscendC::ccu;

    auto *kernelArg = static_cast<ScatterCcuKernelArg *>(arg);
    if (kernelArg == nullptr) {
        return CCU_E_PTR;
    }
    if (kernelArg->mode != SCATTER_PEER_PULL || kernelArg->myRank != kernelArg->root ||
        kernelArg->channelCount == 0 || kernelArg->channelCount >= MAX_RANK_SIZE || kernelArg->recvBytes == 0) {
        return CCU_E_INTERNAL;
    }

    Variable sendBase;
    Variable sendToken;
    Variable recvBase;
    Variable recvToken;
    CCU_CHK_RET(LoadArg(sendBase, 0));
    CCU_CHK_RET(LoadArg(sendToken, 1));
    CCU_CHK_RET(LoadArg(recvBase, 2));
    CCU_CHK_RET(LoadArg(recvToken, 3));

    // Publish the root input once; every peer independently pulls its own slice.
    for (uint32_t idx = 0; idx < kernelArg->channelCount; ++idx) {
        CCU_CHK_RET(WriteVariableWithNotify(
            kernelArg->channels[idx], sendBase, OUTPUT_ADDR_ID, CKE_INDEX, OUTPUT_ADDR_MASK));
        CCU_CHK_RET(WriteVariableWithNotify(
            kernelArg->channels[idx], sendToken, OUTPUT_TOKEN_ID, CKE_INDEX, OUTPUT_TOKEN_MASK));
    }

    // Only one Die kernel copies the root slice.  It overlaps the peer reads.
    if (kernelArg->copyLocal != 0) {
        Event copyEvent;
        uint64_t chunkOffset = 0;
        while (chunkOffset < kernelArg->recvBytes) {
            const uint64_t remaining = kernelArg->recvBytes - chunkOffset;
            const uint64_t chunkBytes = remaining > MAX_DATA_SIZE ? MAX_DATA_SIZE : remaining;

            LocalAddr source;
            source.addr = sendBase;
            Variable sourceOffset;
            if (kernelArg->root != 0 || chunkOffset != 0) {
                sourceOffset = static_cast<uint64_t>(kernelArg->root) * kernelArg->recvBytes + chunkOffset;
                source.addr += sourceOffset;
            }
            source.token = sendToken;

            LocalAddr destination;
            destination.addr = recvBase;
            Variable destinationOffset;
            if (chunkOffset != 0) {
                destinationOffset = chunkOffset;
                destination.addr += destinationOffset;
            }
            destination.token = recvToken;

            Variable size;
            size = chunkBytes;
            CCU_CHK_RET(LocalCopy(destination, source, size, copyEvent, 1));
            CCU_CHK_RET(EventWait(copyEvent, 1));
            chunkOffset += chunkBytes;
        }
    }

    for (uint32_t idx = 0; idx < kernelArg->channelCount; ++idx) {
        CCU_CHK_RET(NotifyWait(kernelArg->channels[idx], CKE_INDEX, COMPLETE_MASK));
    }
    return CCU_SUCCESS;
}

CcuResult CcuScatterPullPeerKernel(CcuKernelArg arg)
{
    using namespace AscendC::ccu;

    auto *kernelArg = static_cast<ScatterCcuKernelArg *>(arg);
    if (kernelArg == nullptr) {
        return CCU_E_PTR;
    }
    if (kernelArg->mode != SCATTER_PEER_PULL || kernelArg->myRank == kernelArg->root ||
        kernelArg->channelCount != 1 || kernelArg->recvBytes == 0) {
        return CCU_E_INTERNAL;
    }

    Variable recvBase;
    Variable recvToken;
    CCU_CHK_RET(LoadArg(recvBase, 0));
    CCU_CHK_RET(LoadArg(recvToken, 1));

    const ChannelHandle rootChannel = kernelArg->channels[0];
    CCU_CHK_RET(NotifyWait(rootChannel, CKE_INDEX, OUTPUT_READY_MASK));

    Event readEvent;
    struct PullOperands {
        RemoteAddr source;
        LocalAddr destination;
        Variable sourceOffset, destinationOffset, size;
    };
    std::vector<PullOperands> operands(PULL_WINDOW);
    uint64_t chunkOffset = 0;
    uint64_t issued = 0;
    uint16_t pendingMask = 0;
    while (chunkOffset < kernelArg->recvBytes) {
        const uint64_t remaining = kernelArg->recvBytes - chunkOffset;
        const uint64_t chunkBytes = remaining > PULL_CHUNK_BYTES ? PULL_CHUNK_BYTES : remaining;
        const uint16_t slotMask = static_cast<uint16_t>(1U << (issued % PULL_WINDOW));
        if (issued >= PULL_WINDOW) {
            // Reclaim just the oldest slot; other reads remain in flight.
            // Never issue two outstanding reads against the same completion bit.
            CCU_CHK_RET(EventWait(readEvent, slotMask));
        }

        PullOperands &slot = operands[issued % PULL_WINDOW];
        RemoteAddr &source = slot.source;
        source.addr = GetResByChannel<Variable>(rootChannel, OUTPUT_ADDR_ID);
        Variable &sourceOffset = slot.sourceOffset;
        if (kernelArg->myRank != 0 || chunkOffset != 0) {
            sourceOffset = static_cast<uint64_t>(kernelArg->myRank) * kernelArg->recvBytes + chunkOffset;
            source.addr += sourceOffset;
        }
        source.token = GetResByChannel<Variable>(rootChannel, OUTPUT_TOKEN_ID);

        LocalAddr &destination = slot.destination;
        destination.addr = recvBase;
        Variable &destinationOffset = slot.destinationOffset;
        if (chunkOffset != 0) {
            destinationOffset = chunkOffset;
            destination.addr += destinationOffset;
        }
        destination.token = recvToken;

        Variable &size = slot.size;
        size = chunkBytes;
        CCU_CHK_RET(Read(rootChannel, destination, source, size, readEvent, slotMask));
        pendingMask = static_cast<uint16_t>(pendingMask | slotMask);
        ++issued;
        chunkOffset += chunkBytes;
    }
    // DONE protects root's input lifetime and must follow every outstanding read.
    CCU_CHK_RET(EventWait(readEvent, pendingMask));

    CCU_CHK_RET(NotifyRecord(rootChannel, CKE_INDEX, COMPLETE_MASK));
    return CCU_SUCCESS;
}
} // namespace ops_hccl
