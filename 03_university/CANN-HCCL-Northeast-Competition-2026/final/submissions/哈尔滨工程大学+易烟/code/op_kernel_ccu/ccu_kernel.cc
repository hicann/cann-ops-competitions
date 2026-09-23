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
#include "ccu_kernel.h"

#include <string>
#include <vector>

namespace ops_hccl {
namespace {
namespace ccu = AscendC::ccu;

#define PULL_CCU_TRY(call) \
    do { \
        const CcuResult result = (call); \
        if (result != CCU_SUCCESS) { \
            return result; \
        } \
    } while (0)

constexpr uint32_t ROOT_FIELD_COUNT = 4;
constexpr uint32_t OFFER_FIELD_COUNT = 7;
static_assert(OFFER_META_MASK == (1U << OFFER_FIELD_COUNT) - 1U, "Full offer fields match the receiver mask");
static_assert(OF_ARG_COUNT == OFFER_FIELD_COUNT, "One launch publishes one complete descriptor");
enum RootField : uint32_t { ROOT_BASE, ROOT_TOKEN, ROOT_LAYOUT, ROOT_MASK };
enum OfferField : uint32_t {
    OFFER_OFFSET = OF_OFFSET, OFFER_BYTES = OF_BYTES, OFFER_NEGATIVE_END = OF_NEGATIVE_END,
    OFFER_ADDRESS = OF_ADDRESS, OFFER_TOKEN = OF_TOKEN,
    OFFER_NEGATIVE_LAYOUT = OF_EXPECTED_LAYOUT, OFFER_NEGATIVE_MASK = OF_EXPECTED_MASK
};
static_assert(OFFER_BYTES == 1, "V20 zero offers publish the existing byte-count slot");
static_assert((OFFER_META_MASK & OFFER_READY_MASK) == 0, "Zero metadata must not promise READY");

struct ReadTransfer {
    ccu::LocalAddr local;
    ccu::RemoteAddr remote;
    ccu::Variable bytes;
};

struct OfferDescriptor {
    ccu::Variable words[OFFER_FIELD_COUNT];
};

std::string NextConditionLabel()
{
    // Registration is synchronous. Keep labels distinct even when the same
    // helper is expanded repeatedly or nested within another condition.
    static thread_local uint64_t sequence = 0;
    return "scatter_pull_if_" + std::to_string(++sequence);
}

// Propagate construction failures instead of silently omitting a conditional
// body. These are the same public entry points used by the 9.1 condition macros.
template <typename ThenBody>
CcuResult EmitIf(const ccu::CondExpr &condition, ThenBody thenBody)
{
    const std::string label = NextConditionLabel();
    PULL_CCU_TRY(CcuIfBegin(condition.var->handle, condition.imm, condition.cond, label.c_str()));
    const CcuResult bodyResult = thenBody();
    const CcuResult endResult = CcuIfEnd(label.c_str());
    return bodyResult == CCU_SUCCESS ? endResult : bodyResult;
}

template <typename ThenBody, typename ElseBody>
CcuResult EmitIfElse(const ccu::CondExpr &condition, ThenBody thenBody, ElseBody elseBody)
{
    const std::string label = NextConditionLabel();
    PULL_CCU_TRY(CcuIfBegin(condition.var->handle, condition.imm, condition.cond, label.c_str()));
    CcuResult bodyResult = thenBody();
    if (bodyResult == CCU_SUCCESS) {
        bodyResult = CcuIfElse(label.c_str());
        if (bodyResult == CCU_SUCCESS) {
            bodyResult = elseBody();
        }
    }
    const CcuResult endResult = CcuIfEnd(label.c_str());
    return bodyResult == CCU_SUCCESS ? endResult : bodyResult;
}

bool ValidGroup(const PullKernelArg *arg)
{
    if (arg == nullptr || arg->rankSize == 0 || arg->rankSize > MAX_RANK_SIZE ||
        arg->myRank >= arg->rankSize || arg->peerCount == 0 || arg->peerCount >= arg->rankSize) {
        return false;
    }
    uint32_t seen = 0;
    for (uint32_t i = 0; i < arg->peerCount; ++i) {
        const uint32_t peer = arg->peers[i];
        if (peer >= arg->rankSize || peer == arg->myRank || arg->channels[i] == 0 ||
            (seen & RankBit(peer)) != 0) {
            return false;
        }
        seen |= RankBit(peer);
    }
    return true;
}

bool ValidPeer(const PullPeerArg *arg)
{
    return arg != nullptr && arg->channel != 0 && arg->rankSize != 0 &&
        arg->rankSize <= MAX_RANK_SIZE && arg->myRank < arg->rankSize;
}

bool ValidNonrootGroup(const PullKernelArg *arg)
{
    return ValidGroup(arg) && arg->staticRoot < arg->rankSize && arg->staticRoot != arg->myRank;
}

CcuResult LoadArguments(ccu::Variable *args, uint32_t count)
{
    for (uint32_t i = 0; i < count; ++i) {
        PULL_CCU_TRY(ccu::LoadArg(args[i], i));
    }
    return CCU_SUCCESS;
}

CcuResult WaitChannel(ChannelHandle channel, uint32_t notify, uint16_t mask)
{
    // Early 9.1 NotifyWait does not register its channel for SelectDie (fixed
    // upstream by 7b293b12). Bind an existing XN explicitly so wait-only kernels
    // use the channel's die. This emits no instruction and never modifies XN0.
    const auto channelBinding = ccu::GetResByChannel<ccu::Variable>(channel, 0);
    (void)channelBinding;
    return ccu::NotifyWait(channel, notify, mask);
}

CcuResult CaptureWords(ChannelHandle channel, ccu::Variable *words, uint32_t count)
{
    for (uint32_t i = 0; i < count; ++i) {
        // Assign from an lvalue to copy the value. Assigning the temporary returned
        // by GetResByChannel would rebind the handle instead of taking a snapshot.
        auto channelWord = ccu::GetResByChannel<ccu::Variable>(channel, i);
        words[i] = channelWord;
    }
    return CCU_SUCCESS;
}

CcuResult CaptureOfferDescriptor(ChannelHandle channel, ccu::Variable *words, uint32_t rankSize)
{
    if (rankSize != 12) {
        // Preserve the original N16 program without a runtime format branch.
        return CaptureWords(channel, words, OFFER_FIELD_COUNT);
    }
    // The caller has already consumed the complete 0x7f META. In N12, only
    // BYTES is valid for a zero offer; the other channel XNs may never have
    // been initialized. Snapshot BYTES first, then define every local field.
    auto channelBytes = ccu::GetResByChannel<ccu::Variable>(channel, OFFER_BYTES);
    words[OFFER_BYTES] = channelBytes;
    return EmitIfElse(words[OFFER_BYTES] != 0,
        [&]() -> CcuResult {
            for (uint32_t field = 0; field < OFFER_FIELD_COUNT; ++field) {
                if (field != OFFER_BYTES) {
                    auto channelWord = ccu::GetResByChannel<ccu::Variable>(channel, field);
                    words[field] = channelWord;
                }
            }
            return CCU_SUCCESS;
        },
        [&]() -> CcuResult {
            for (uint32_t field = 0; field < OFFER_FIELD_COUNT; ++field) {
                if (field != OFFER_BYTES) {
                    words[field] = 0;
                }
            }
            return CCU_SUCCESS;
        });
}

CcuResult PublishWords(ChannelHandle channel, ccu::Variable *words, uint32_t count, uint32_t notify)
{
    for (uint32_t i = 0; i < count; ++i) {
        PULL_CCU_TRY(ccu::WriteVariableWithNotify(channel, words[i], i, notify,
            static_cast<uint16_t>(1U << i)));
    }
    return CCU_SUCCESS;
}

CcuResult ConsumeOfferReady(ChannelHandle channel, ccu::Variable *descriptor, ccu::Variable &difference)
{
    // READY is promised by the raw descriptor, independently of root approval
    // or this die's ability to read the payload. Rejected offers must drain it
    // too, so no notification can survive the collective frame release.
    return EmitIf(descriptor[OFFER_BYTES] != 0, [&]() -> CcuResult {
        // Reuse one condition register across sequential peer waits; it is
        // never an operand of an in-flight payload Read.
        difference = static_cast<uint64_t>(PullLayout::TWO_BY_EIGHT_PREFIXES);
        difference += descriptor[OFFER_NEGATIVE_LAYOUT];
        PULL_CCU_TRY(EmitIf(difference == 0, [&]() -> CcuResult {
            return WaitChannel(channel, OFFER_NOTIFY, OFFER_READY_MASK);
        }));
        difference = static_cast<uint64_t>(PullLayout::FOUR_BY_THREE_PREFIXES);
        difference += descriptor[OFFER_NEGATIVE_LAYOUT];
        PULL_CCU_TRY(EmitIf(difference == 0, [&]() -> CcuResult {
            return WaitChannel(channel, OFFER_NOTIFY, OFFER_READY_MASK);
        }));
        difference = static_cast<uint64_t>(PullLayout::EIGHT_PLUS_FOUR_PREFIXES);
        difference += descriptor[OFFER_NEGATIVE_LAYOUT];
        // Distinct layout values ensure at most one branch waits. Legacy
        // EIGHT_PLUS_FOUR_BANDS still publishes after prefetch without READY.
        return EmitIf(difference == 0, [&]() -> CcuResult {
            return WaitChannel(channel, OFFER_NOTIFY, OFFER_READY_MASK);
        });
    });
}

CcuResult SetAuthorization(ccu::Variable *header, const ccu::Variable &negativeLayout,
    const ccu::Variable &negativeMask, ccu::Variable &authorized)
{
    ccu::Variable layoutDifference;
    ccu::Variable maskDifference;
    layoutDifference = header[ROOT_LAYOUT] + negativeLayout;
    maskDifference = header[ROOT_MASK] + negativeMask;
    authorized = 0;
    return EmitIf(header[ROOT_LAYOUT] != static_cast<uint64_t>(PullLayout::DIRECT), [&]() -> CcuResult {
        return EmitIf(layoutDifference == 0, [&]() -> CcuResult {
            // Compare independently: adding the two differences could cancel.
            return EmitIf(maskDifference == 0, [&]() -> CcuResult {
                authorized = 1;
                return CCU_SUCCESS;
            });
        });
    });
}

CcuResult IssueRead(ChannelHandle channel, ReadTransfer &transfer, ccu::Event completed, uint16_t mask)
{
    return EmitIf(transfer.bytes != 0, [&]() -> CcuResult {
        return ccu::Read(channel, transfer.local, transfer.remote, transfer.bytes, completed, mask);
    });
}

CcuResult WaitRead(ReadTransfer &transfer, ccu::Event completed, uint16_t mask)
{
    // The length register stays unchanged between IssueRead and WaitRead.
    // Zero-length entries neither produce nor consume a completion bit.
    return EmitIf(transfer.bytes != 0, [&]() -> CcuResult {
        return ccu::EventWait(completed, mask);
    });
}

CcuResult ReadAndWait(ChannelHandle channel, ReadTransfer &transfer)
{
    ccu::Event completed;
    return EmitIf(transfer.bytes != 0, [&]() -> CcuResult {
        PULL_CCU_TRY(ccu::Read(channel, transfer.local, transfer.remote, transfer.bytes, completed, 1));
        return ccu::EventWait(completed, 1);
    });
}

CcuResult CopyChunk(ccu::Variable &sourceAddress, ccu::Variable &sourceToken,
    ccu::Variable &outputAddress, ccu::Variable &outputToken, ccu::Variable &bufferAddress,
    ccu::Variable &bufferToken, ccu::Variable &bytes, ccu::Variable &staging)
{
    ccu::LocalAddr source;
    ccu::LocalAddr output;
    ccu::LocalAddr buffer;
    source.addr = sourceAddress;
    source.token = sourceToken;
    output.addr = outputAddress;
    output.token = outputToken;
    buffer.addr = bufferAddress;
    buffer.token = bufferToken;
    ccu::Event completed;
    return EmitIf(bytes != 0, [&]() -> CcuResult {
        PULL_CCU_TRY(EmitIfElse(staging != 0,
            [&]() -> CcuResult {
                PULL_CCU_TRY(ccu::LocalCopy(buffer, source, bytes, completed));
                PULL_CCU_TRY(ccu::EventWait(completed));
                return ccu::LocalCopy(output, buffer, bytes, completed);
            },
            [&]() -> CcuResult { return ccu::LocalCopy(output, source, bytes, completed); }));
        return ccu::EventWait(completed);
    });
}

CcuResult PrepareRootGaps(const PullKernelArg &kernel, ccu::Variable *args, ccu::Variable *header,
    std::vector<ReadTransfer> &transfers)
{
    ccu::Variable base;
    ccu::Variable cursor;
    ccu::Variable negativeCursor;
    ccu::Variable descriptor[OFFER_FIELD_COUNT];
    ccu::Variable authorized;
    base = header[ROOT_BASE] + args[GT_SOURCE_OFFSET];
    cursor = 0;
    negativeCursor = 0;
    for (uint32_t source = 0; source < kernel.rankSize; ++source) {
        if (source == kernel.myRank) {
            continue;
        }
        auto &transfer = transfers[source];
        transfer.bytes = 0;
        if (source == kernel.staticRoot) {
            continue;
        }
        uint32_t sourceIndex = kernel.peerCount;
        for (uint32_t i = 0; i < kernel.peerCount; ++i) {
            if (kernel.peers[i] == source) {
                sourceIndex = i;
            }
        }
        if (sourceIndex == kernel.peerCount) {
            continue;
        }
        // Only descriptors on the root channel's actual die can replace a root
        // interval. Authorized layouts order nonoverlapping intervals by supplier rank.
        PULL_CCU_TRY(CaptureOfferDescriptor(kernel.channels[sourceIndex], descriptor, kernel.rankSize));
        PULL_CCU_TRY(SetAuthorization(header, descriptor[OFFER_NEGATIVE_LAYOUT],
            descriptor[OFFER_NEGATIVE_MASK], authorized));
        PULL_CCU_TRY(EmitIf(authorized != 0, [&]() -> CcuResult {
            return EmitIf(descriptor[OFFER_BYTES] != 0, [&]() -> CcuResult {
                transfer.bytes = descriptor[OFFER_OFFSET] + negativeCursor;
                transfer.remote.addr = base;
                transfer.remote.addr += cursor;
                transfer.remote.token = header[ROOT_TOKEN];
                transfer.local.addr = args[GT_OUTPUT];
                transfer.local.addr += cursor;
                transfer.local.token = args[GT_OUTPUT_TOKEN];
                cursor = descriptor[OFFER_OFFSET] + descriptor[OFFER_BYTES];
                negativeCursor = descriptor[OFFER_NEGATIVE_END];
                return CCU_SUCCESS;
            });
        }));
    }
    // The skipped self entry supplies a distinct event bit and address/length
    // registers for the suffix. Every in-flight gap retains its own registers.
    auto &suffix = transfers[kernel.myRank];
    suffix.bytes = args[GT_BYTES] + negativeCursor;
    suffix.remote.addr = base;
    suffix.remote.addr += cursor;
    suffix.remote.token = header[ROOT_TOKEN];
    suffix.local.addr = args[GT_OUTPUT];
    suffix.local.addr += cursor;
    suffix.local.token = args[GT_OUTPUT_TOKEN];
    return CCU_SUCCESS;
}

CcuResult ReadRelayedFrame(const PullKernelArg &kernel, ccu::Variable *args,
    ccu::Variable *header, uint32_t rootIndex)
{
    std::vector<ReadTransfer> rootTransfers(kernel.rankSize);
    ccu::Event rootCompleted;
    PULL_CCU_TRY(PrepareRootGaps(kernel, args, header, rootTransfers));
    // Every META was consumed before constructing the complete root-gap list.
    // Issue all real reads before waiting; each retains its own address/length.
    for (uint32_t source = 0; source < kernel.rankSize; ++source) {
        PULL_CCU_TRY(IssueRead(kernel.channels[rootIndex], rootTransfers[source], rootCompleted,
            static_cast<uint16_t>(1U << source)));
    }
    std::vector<ReadTransfer> helperTransfers(kernel.peerCount);
    ccu::Event helperCompleted;
    ccu::Variable descriptor[OFFER_FIELD_COUNT];
    ccu::Variable authorized;
    ccu::Variable readyLayoutDifference;
    for (uint32_t i = 0; i < kernel.peerCount; ++i) {
        const uint16_t mask = static_cast<uint16_t>(1U << i);
        if (i == rootIndex) {
            continue;
        }
        PULL_CCU_TRY(CaptureOfferDescriptor(kernel.channels[i], descriptor, kernel.rankSize));
        // Root-gap reads are already in flight. Early descriptors reserve the
        // helper interval before prefetch, so consume READY before reading it.
        // This also drains a nonzero early offer rejected by SetAuthorization.
        PULL_CCU_TRY(ConsumeOfferReady(kernel.channels[i], descriptor, readyLayoutDifference));
        auto &transfer = helperTransfers[i];
        transfer.remote.addr = descriptor[OFFER_ADDRESS];
        transfer.remote.token = descriptor[OFFER_TOKEN];
        transfer.local.addr = args[GT_OUTPUT];
        transfer.local.addr += descriptor[OFFER_OFFSET];
        transfer.local.token = args[GT_OUTPUT_TOKEN];
        transfer.bytes = 0;
        // Legacy 8+4 META follows prefetch; early layouts have now consumed READY.
        PULL_CCU_TRY(SetAuthorization(header, descriptor[OFFER_NEGATIVE_LAYOUT],
            descriptor[OFFER_NEGATIVE_MASK], authorized));
        PULL_CCU_TRY(EmitIf(authorized != 0, [&]() -> CcuResult {
            transfer.bytes = descriptor[OFFER_BYTES];
            return CCU_SUCCESS;
        }));
        PULL_CCU_TRY(IssueRead(kernel.channels[i], transfer, helperCompleted, mask));
    }
    for (uint32_t i = 0; i < kernel.peerCount; ++i) {
        if (i != rootIndex) {
            PULL_CCU_TRY(WaitRead(helperTransfers[i], helperCompleted, static_cast<uint16_t>(1U << i)));
        }
    }
    for (uint32_t source = 0; source < kernel.rankSize; ++source) {
        PULL_CCU_TRY(WaitRead(rootTransfers[source], rootCompleted, static_cast<uint16_t>(1U << source)));
    }
    // Host joins both dies before ROOT_DONE. Root releases every rank only
    // after all reads have completed, including the final frame of a call.
    return CCU_SUCCESS;
}

CcuResult ReadDirectFrameAndDrainOffers(const PullKernelArg &kernel, ccu::Variable *args,
    ccu::Variable *header, uint32_t rootIndex)
{
    // This authorized role needs no supplier on this die. Keep a complete root
    // Read in flight while draining the unchanged offer protocol on this die.
    // Its registers are independent of the descriptor snapshots below.
    ReadTransfer transfer;
    ccu::Event completed;
    transfer.local.addr = args[GT_OUTPUT];
    transfer.local.token = args[GT_OUTPUT_TOKEN];
    transfer.remote.addr = header[ROOT_BASE];
    transfer.remote.addr += args[GT_SOURCE_OFFSET];
    transfer.remote.token = header[ROOT_TOKEN];
    transfer.bytes = args[GT_BYTES];
    PULL_CCU_TRY(IssueRead(kernel.channels[rootIndex], transfer, completed, 1));
    for (uint32_t i = 0; i < kernel.peerCount; ++i) {
        if (i != rootIndex) {
            PULL_CCU_TRY(WaitChannel(kernel.channels[i], OFFER_NOTIFY, OFFER_META_MASK));
        }
    }
    ccu::Variable descriptor[OFFER_FIELD_COUNT];
    ccu::Variable readyLayoutDifference;
    for (uint32_t i = 0; i < kernel.peerCount; ++i) {
        if (i != rootIndex) {
            PULL_CCU_TRY(CaptureOfferDescriptor(kernel.channels[i], descriptor, kernel.rankSize));
            // An unused nonzero offer still promises READY. This role reads
            // its entire frame from root, but must drain that promise once.
            PULL_CCU_TRY(ConsumeOfferReady(kernel.channels[i], descriptor, readyLayoutDifference));
        }
    }
    return WaitRead(transfer, completed, 1);
}

uint32_t FindPeerIndex(const PullKernelArg &kernel, uint32_t rank)
{
    for (uint32_t i = 0; i < kernel.peerCount; ++i) {
        if (kernel.peers[i] == rank) {
            return i;
        }
    }
    return kernel.peerCount;
}

CcuResult WaitOfferMetadata(const PullKernelArg &kernel, uint32_t rootIndex, uint32_t consumedIndex)
{
    for (uint32_t i = 0; i < kernel.peerCount; ++i) {
        if (i != rootIndex && i != consumedIndex) {
            PULL_CCU_TRY(WaitChannel(kernel.channels[i], OFFER_NOTIFY, OFFER_META_MASK));
        }
    }
    return CCU_SUCCESS;
}

CcuResult DrainOfferReadiness(const PullKernelArg &kernel, uint32_t rootIndex, uint32_t consumedIndex)
{
    ccu::Variable descriptor[OFFER_FIELD_COUNT];
    ccu::Variable difference;
    for (uint32_t i = 0; i < kernel.peerCount; ++i) {
        if (i != rootIndex && i != consumedIndex) {
            PULL_CCU_TRY(CaptureOfferDescriptor(kernel.channels[i], descriptor, kernel.rankSize));
            PULL_CCU_TRY(ConsumeOfferReady(kernel.channels[i], descriptor, difference));
        }
    }
    return CCU_SUCCESS;
}

uint32_t PrefixSupplier(const PullKernelArg &kernel, uint32_t rootMask)
{
    if ((rootMask & RankBit(kernel.myRank)) != 0) {
        return kernel.rankSize;
    }
    uint32_t remoteOrdinal = 0;
    for (uint32_t rank = 0; rank < kernel.myRank; ++rank) {
        if ((rootMask & RankBit(rank)) == 0) {
            ++remoteOrdinal;
        }
    }
    uint32_t helperOrdinal = 0;
    for (uint32_t rank = 0; rank < kernel.rankSize; ++rank) {
        if ((rootMask & RankBit(rank)) != 0 && rank != kernel.staticRoot) {
            if (helperOrdinal == remoteOrdinal) {
                return rank;
            }
            ++helperOrdinal;
        }
    }
    // Seven helpers cover the first seven remote ranks in numeric order.
    // This sentinel also covers the eighth remote, regardless of rank layout.
    return kernel.rankSize;
}

CcuResult ReadPairedPrefixFrame(const PullKernelArg &kernel, ccu::Variable *args,
    ccu::Variable *header, uint32_t rootIndex, uint32_t supplierIndex)
{
    const ChannelHandle supplier = kernel.channels[supplierIndex];
    PULL_CCU_TRY(WaitChannel(supplier, OFFER_NOTIFY, OFFER_META_MASK));
    ccu::Variable descriptor[OFFER_FIELD_COUNT];
    PULL_CCU_TRY(CaptureWords(supplier, descriptor, OFFER_FIELD_COUNT));
    ccu::Variable authorized;
    PULL_CCU_TRY(SetAuthorization(header, descriptor[OFFER_NEGATIVE_LAYOUT],
        descriptor[OFFER_NEGATIVE_MASK], authorized));
    ccu::Variable endDifference;
    endDifference = descriptor[OFFER_BYTES] + descriptor[OFFER_NEGATIVE_END];
    ccu::Variable prefixBytes;
    ccu::Variable negativePrefix;
    prefixBytes = 0;
    negativePrefix = 0;
    PULL_CCU_TRY(EmitIf(authorized != 0, [&]() -> CcuResult {
        return EmitIf(descriptor[OFFER_OFFSET] == 0, [&]() -> CcuResult {
            return EmitIf(endDifference == 0, [&]() -> CcuResult {
                // Host bounds this actual offer by both frame size and scratch.
                // Do not substitute the nominal 4/11 share for a clipped offer.
                prefixBytes = descriptor[OFFER_BYTES];
                negativePrefix = descriptor[OFFER_NEGATIVE_END];
                return CCU_SUCCESS;
            });
        });
    }));

    ReadTransfer suffix;
    ccu::Event rootCompleted;
    suffix.local.addr = args[GT_OUTPUT];
    suffix.local.addr += prefixBytes;
    suffix.local.token = args[GT_OUTPUT_TOKEN];
    suffix.remote.addr = header[ROOT_BASE];
    suffix.remote.addr += args[GT_SOURCE_OFFSET];
    suffix.remote.addr += prefixBytes;
    suffix.remote.token = header[ROOT_TOKEN];
    suffix.bytes = args[GT_BYTES] + negativePrefix;
    PULL_CCU_TRY(IssueRead(kernel.channels[rootIndex], suffix, rootCompleted, 1));

    // Drain the raw promise even if its layout/mask/interval was rejected.
    // No path after this point can repeat the supplier's META or READY wait.
    ccu::Variable readyDifference;
    PULL_CCU_TRY(ConsumeOfferReady(supplier, descriptor, readyDifference));
    ReadTransfer prefix;
    ccu::Event helperCompleted;
    prefix.local.addr = args[GT_OUTPUT];
    prefix.local.token = args[GT_OUTPUT_TOKEN];
    prefix.remote.addr = descriptor[OFFER_ADDRESS];
    prefix.remote.token = descriptor[OFFER_TOKEN];
    prefix.bytes = prefixBytes;
    PULL_CCU_TRY(IssueRead(supplier, prefix, helperCompleted, 1));
    PULL_CCU_TRY(WaitOfferMetadata(kernel, rootIndex, supplierIndex));
    PULL_CCU_TRY(DrainOfferReadiness(kernel, rootIndex, supplierIndex));
    PULL_CCU_TRY(WaitRead(prefix, helperCompleted, 1));
    return WaitRead(suffix, rootCompleted, 1);
}

CcuResult PublishZeroOffers(const PullKernelArg &kernel, const ccu::Variable &zero)
{
    for (uint32_t i = 0; i < kernel.peerCount; ++i) {
        if (kernel.peers[i] == kernel.staticRoot) {
            continue;
        }
        // Root8's fused sender has no payload. One atomic write makes BYTES
        // visible before the unchanged META mask; N12 receivers normalize the
        // remaining fields locally. No READY is sent for this zero descriptor.
        PULL_CCU_TRY(ccu::WriteVariableWithNotify(kernel.channels[i], zero, OFFER_BYTES,
            OFFER_NOTIFY, OFFER_META_MASK));
    }
    return CCU_SUCCESS;
}

CcuResult ReadDirectFrameAndPublishOffers(const PullKernelArg &kernel, ccu::Variable *args,
    ccu::Variable *header, uint32_t rootIndex, const ccu::Variable &zero)
{
    ReadTransfer transfer;
    ccu::Event completed;
    transfer.local.addr = args[GT_OUTPUT];
    transfer.local.token = args[GT_OUTPUT_TOKEN];
    transfer.remote.addr = header[ROOT_BASE];
    transfer.remote.addr += args[GT_SOURCE_OFFSET];
    transfer.remote.token = header[ROOT_TOKEN];
    transfer.bytes = args[GT_BYTES];
    PULL_CCU_TRY(IssueRead(kernel.channels[rootIndex], transfer, completed, 1));
    // All writes share an immutable zero register, separate from Read operands.
    // Both dies publish before peer waits, including when root is on this die.
    PULL_CCU_TRY(PublishZeroOffers(kernel, zero));
    PULL_CCU_TRY(WaitOfferMetadata(kernel, rootIndex, kernel.peerCount));
    PULL_CCU_TRY(DrainOfferReadiness(kernel, rootIndex, kernel.peerCount));
    return WaitRead(transfer, completed, 1);
}

} // namespace

CcuResult CcuSmallPullRootKernel(CcuKernelArg arg)
{
    const auto *kernel = static_cast<const PullKernelArg *>(arg);
    if (!ValidGroup(kernel)) {
        return CCU_E_PARA;
    }
    ccu::Variable args[SR_ARG_COUNT];
    PULL_CCU_TRY(LoadArguments(args, SR_ARG_COUNT));
    for (uint32_t i = 0; i < kernel->peerCount; ++i) {
        PULL_CCU_TRY(ccu::WriteVariableWithNotify(kernel->channels[i], args[SR_INPUT],
            ROOT_BASE, PULL_NOTIFY, 1));
        PULL_CCU_TRY(ccu::WriteVariableWithNotify(kernel->channels[i], args[SR_INPUT_TOKEN],
            ROOT_TOKEN, PULL_NOTIFY, 2));
    }
    // Mode 2 is the copy-timing comparison: retain this ten-argument kernel,
    // but move its copy after META and before DONE. Host enables it only when
    // the output cannot overlap any rank's slice of the complete root input.
    auto copy = [&]() -> CcuResult {
        return CopyChunk(args[SR_COPY_SOURCE], args[SR_INPUT_TOKEN], args[SR_OUTPUT],
            args[SR_OUTPUT_TOKEN], args[SR_BUFFER], args[SR_BUFFER_TOKEN], args[SR_COPY_BYTES], args[SR_STAGING]);
    };
    PULL_CCU_TRY(EmitIf(args[SR_DO_COPY] == 2, copy));
    for (uint32_t i = 0; i < kernel->peerCount; ++i) {
        PULL_CCU_TRY(WaitChannel(kernel->channels[i], PULL_NOTIFY, PULL_DONE_MASK));
    }
    // Host enables fusion only after proving that this group's completion is
    // sufficient for the copy, including any source/output overlap.
    return EmitIf(args[SR_DO_COPY] == 1, copy);
}

CcuResult CcuDirectRootKernel(CcuKernelArg arg)
{
    const auto *kernel = static_cast<const PullKernelArg *>(arg);
    if (!ValidGroup(kernel)) {
        return CCU_E_PARA;
    }
    ccu::Variable args[DR_ARG_COUNT];
    PULL_CCU_TRY(LoadArguments(args, DR_ARG_COUNT));
    for (uint32_t i = 0; i < kernel->peerCount; ++i) {
        PULL_CCU_TRY(ccu::WriteVariableWithNotify(kernel->channels[i], args[DR_INPUT],
            ROOT_BASE, PULL_NOTIFY, 1));
        PULL_CCU_TRY(ccu::WriteVariableWithNotify(kernel->channels[i], args[DR_INPUT_TOKEN],
            ROOT_TOKEN, PULL_NOTIFY, 2));
    }
    for (uint32_t i = 0; i < kernel->peerCount; ++i) {
        PULL_CCU_TRY(WaitChannel(kernel->channels[i], PULL_NOTIFY, PULL_DONE_MASK));
    }
    return CCU_SUCCESS;
}

CcuResult CcuDirectRootCopyKernel(CcuKernelArg arg)
{
    const auto *kernel = static_cast<const PullKernelArg *>(arg);
    if (!ValidGroup(kernel)) {
        return CCU_E_PARA;
    }
    ccu::Variable args[DRC_ARG_COUNT];
    PULL_CCU_TRY(LoadArguments(args, DRC_ARG_COUNT));
    for (uint32_t i = 0; i < kernel->peerCount; ++i) {
        PULL_CCU_TRY(ccu::WriteVariableWithNotify(kernel->channels[i], args[DRC_INPUT],
            ROOT_BASE, PULL_NOTIFY, 1));
        PULL_CCU_TRY(ccu::WriteVariableWithNotify(kernel->channels[i], args[DRC_INPUT_TOKEN],
            ROOT_TOKEN, PULL_NOTIFY, 2));
    }
    // These registers are independent of channel XNs and remain unchanged
    // until the local completion event is consumed. Host assigns this role to
    // one die only, and proves output disjoint from the complete root input.
    ccu::LocalAddr source;
    ccu::LocalAddr output;
    ccu::Variable copyBytes;
    ccu::Event copyCompleted;
    source.addr = args[DRC_SOURCE];
    source.token = args[DRC_INPUT_TOKEN];
    output.addr = args[DRC_OUTPUT];
    output.token = args[DRC_OUTPUT_TOKEN];
    copyBytes = args[DRC_BYTES];
    PULL_CCU_TRY(EmitIf(copyBytes != 0, [&]() -> CcuResult {
        return ccu::LocalCopy(output, source, copyBytes, copyCompleted);
    }));
    for (uint32_t i = 0; i < kernel->peerCount; ++i) {
        PULL_CCU_TRY(WaitChannel(kernel->channels[i], PULL_NOTIFY, PULL_DONE_MASK));
    }
    return EmitIf(copyBytes != 0, [&]() -> CcuResult {
        return ccu::EventWait(copyCompleted);
    });
}

namespace {
CcuResult BuildCompactDirectRootCopy(const PullKernelArg *kernel)
{
    ccu::Variable args[DRC_ARG_COUNT];
    PULL_CCU_TRY(LoadArguments(args, DRC_ARG_COUNT));
    for (uint32_t i = 0; i < kernel->peerCount; ++i) {
        PULL_CCU_TRY(ccu::WriteVariableWithNotify(kernel->channels[i], args[DRC_INPUT],
            ROOT_BASE, PULL_NOTIFY, 1));
        PULL_CCU_TRY(ccu::WriteVariableWithNotify(kernel->channels[i], args[DRC_INPUT_TOKEN],
            ROOT_TOKEN, PULL_NOTIFY, 2));
    }
    // Host only selects this copy variant for positive Small frames whose
    // output is disjoint from the complete root input. Retain independent
    // address/token resources until completion, while peer Reads run in parallel.
    ccu::LocalAddr source;
    ccu::LocalAddr output;
    ccu::Event copyCompleted;
    source.addr = args[DRC_SOURCE];
    source.token = args[DRC_INPUT_TOKEN];
    output.addr = args[DRC_OUTPUT];
    output.token = args[DRC_OUTPUT_TOKEN];
    PULL_CCU_TRY(ccu::LocalCopy(output, source, args[DRC_BYTES], copyCompleted));
    for (uint32_t i = 0; i < kernel->peerCount; ++i) {
        PULL_CCU_TRY(WaitChannel(kernel->channels[i], PULL_NOTIFY, PULL_DONE_MASK));
    }
    return ccu::EventWait(copyCompleted);
}
} // namespace

CcuResult CcuFourByOneDirectRootCopyKernel(CcuKernelArg arg)
{
    const auto *kernel = static_cast<const PullKernelArg *>(arg);
    if (!ValidGroup(kernel) || kernel->rankSize != 4) {
        return CCU_E_PARA;
    }
    return BuildCompactDirectRootCopy(kernel);
}

CcuResult CcuFourByThreeDirectRootCopyKernel(CcuKernelArg arg)
{
    const auto *kernel = static_cast<const PullKernelArg *>(arg);
    if (!ValidGroup(kernel) || kernel->rankSize != 12) {
        return CCU_E_PARA;
    }
    return BuildCompactDirectRootCopy(kernel);
}

CcuResult CcuSmallPullPeerKernel(CcuKernelArg arg)
{
    const auto *kernel = static_cast<const PullPeerArg *>(arg);
    if (!ValidPeer(kernel)) {
        return CCU_E_PARA;
    }
    ccu::Variable args[SP_ARG_COUNT];
    PULL_CCU_TRY(LoadArguments(args, SP_ARG_COUNT));
    PULL_CCU_TRY(WaitChannel(kernel->channel, PULL_NOTIFY, PULL_META_MASK));
    ccu::Variable header[2];
    PULL_CCU_TRY(CaptureWords(kernel->channel, header, 2));
    ReadTransfer transfer;
    transfer.local.addr = args[SP_OUTPUT];
    transfer.local.token = args[SP_OUTPUT_TOKEN];
    transfer.remote.addr = header[ROOT_BASE];
    transfer.remote.addr += args[SP_SOURCE_OFFSET];
    transfer.remote.token = header[ROOT_TOKEN];
    transfer.bytes = args[SP_BYTES];
    PULL_CCU_TRY(ReadAndWait(kernel->channel, transfer));
    PULL_CCU_TRY(ccu::NotifyRecord(kernel->channel, PULL_NOTIFY, PULL_DONE_MASK));
    return CCU_SUCCESS;
}

namespace {
CcuResult BuildCompactPullPeer(const PullPeerArg *kernel)
{
    ccu::LocalAddr local;
    ccu::RemoteAddr remote;
    ccu::Variable outputAddress;
    ccu::Variable sourceOffset;
    ccu::Variable bytes;
    ccu::Event completed;
    PULL_CCU_TRY(ccu::LoadArg(outputAddress, SP_OUTPUT));
    PULL_CCU_TRY(ccu::LoadArg(local.token, SP_OUTPUT_TOKEN));
    PULL_CCU_TRY(ccu::LoadArg(sourceOffset, SP_SOURCE_OFFSET));
    PULL_CCU_TRY(ccu::LoadArg(bytes, SP_BYTES));
    local.addr = outputAddress;
    PULL_CCU_TRY(WaitChannel(kernel->channel, PULL_NOTIFY, PULL_META_MASK));
    // Copy channel values directly into independent transfer resources.
    // Named lvalues avoid rebinding remote.token away from RemoteAddr's token.
    auto rootBase = ccu::GetResByChannel<ccu::Variable>(kernel->channel, ROOT_BASE);
    auto rootToken = ccu::GetResByChannel<ccu::Variable>(kernel->channel, ROOT_TOKEN);
    remote.addr = rootBase;
    remote.addr += sourceOffset;
    remote.token = rootToken;
    // ExecDirectPull only launches positive frames, including N4 Large frames.
    // Keep the original completion wait before acknowledging root.
    PULL_CCU_TRY(ccu::Read(kernel->channel, local, remote, bytes, completed));
    PULL_CCU_TRY(ccu::EventWait(completed));
    return ccu::NotifyRecord(kernel->channel, PULL_NOTIFY, PULL_DONE_MASK);
}
} // namespace

CcuResult CcuFourByOnePullPeerKernel(CcuKernelArg arg)
{
    const auto *kernel = static_cast<const PullPeerArg *>(arg);
    if (!ValidPeer(kernel) || kernel->rankSize != 4) {
        return CCU_E_PARA;
    }
    return BuildCompactPullPeer(kernel);
}

CcuResult CcuFourByThreePullPeerKernel(CcuKernelArg arg)
{
    const auto *kernel = static_cast<const PullPeerArg *>(arg);
    if (!ValidPeer(kernel) || kernel->rankSize != 12) {
        return CCU_E_PARA;
    }
    return BuildCompactPullPeer(kernel);
}

CcuResult CcuRootPublishKernel(CcuKernelArg arg)
{
    const auto *kernel = static_cast<const PullKernelArg *>(arg);
    if (!ValidGroup(kernel)) {
        return CCU_E_PARA;
    }
    ccu::Variable args[RP_ARG_COUNT];
    PULL_CCU_TRY(LoadArguments(args, RP_ARG_COUNT));
    for (uint32_t i = 0; i < kernel->peerCount; ++i) {
        PULL_CCU_TRY(PublishWords(kernel->channels[i], args, ROOT_FIELD_COUNT, PULL_NOTIFY));
    }
    return CCU_SUCCESS;
}

CcuResult CcuEightPlusFourRootPhaseKernel(CcuKernelArg arg)
{
    const auto *kernel = static_cast<const PullKernelArg *>(arg);
    if (!ValidGroup(kernel) || kernel->rankSize != 12) {
        return CCU_E_PARA;
    }
    ccu::Variable args[RPH_ARG_COUNT];
    PULL_CCU_TRY(LoadArguments(args, RPH_ARG_COUNT));
    for (uint32_t i = 0; i < kernel->peerCount; ++i) {
        PULL_CCU_TRY(PublishWords(kernel->channels[i], args, ROOT_FIELD_COUNT, PULL_NOTIFY));
    }
    // The first four task fields retain ROOT4. Host supplies a distinct flag
    // per group, assigning the safe frame copy to exactly one actual die.
    // Keep both address/token resources stable through the final EventWait.
    ccu::LocalAddr source;
    ccu::LocalAddr output;
    ccu::Event copyCompleted;
    source.addr = args[RPH_COPY_SOURCE];
    source.token = args[RPH_TOKEN];
    output.addr = args[RPH_OUTPUT];
    output.token = args[RPH_OUTPUT_TOKEN];
    PULL_CCU_TRY(EmitIf(args[RPH_DO_COPY] == 1, [&]() -> CcuResult {
        return ccu::LocalCopy(output, source, args[RPH_COPY_BYTES], copyCompleted);
    }));
    for (uint32_t i = 0; i < kernel->peerCount; ++i) {
        PULL_CCU_TRY(WaitChannel(kernel->channels[i], PULL_NOTIFY, PULL_DONE_MASK));
    }
    // Host joins the other die before releasing any peer. In-place and alias
    // cases pass flag zero and retain their original final-copy handling.
    return EmitIf(args[RPH_DO_COPY] == 1, [&]() -> CcuResult {
        return ccu::EventWait(copyCompleted);
    });
}

CcuResult CcuRootWaitKernel(CcuKernelArg arg)
{
    const auto *kernel = static_cast<const PullKernelArg *>(arg);
    if (!ValidGroup(kernel)) {
        return CCU_E_PARA;
    }
    for (uint32_t i = 0; i < kernel->peerCount; ++i) {
        PULL_CCU_TRY(WaitChannel(kernel->channels[i], PULL_NOTIFY, PULL_DONE_MASK));
    }
    return CCU_SUCCESS;
}

CcuResult CcuRootReleaseKernel(CcuKernelArg arg)
{
    const auto *kernel = static_cast<const PullKernelArg *>(arg);
    if (!ValidGroup(kernel)) {
        return CCU_E_PARA;
    }
    // Host launches this only after RootWait has finished on BOTH dies.
    for (uint32_t i = 0; i < kernel->peerCount; ++i) {
        PULL_CCU_TRY(ccu::NotifyRecord(kernel->channels[i], PULL_NOTIFY, ROOT_RELEASE_MASK));
    }
    return CCU_SUCCESS;
}

CcuResult CcuRootCaptureKernel(CcuKernelArg arg)
{
    const auto *kernel = static_cast<const PullPeerArg *>(arg);
    if (!ValidPeer(kernel)) {
        return CCU_E_PARA;
    }
    // The root keeps these channel XNs stable until this rank sends ROOT_DONE.
    // Later kernels on this same die can take their own register snapshots.
    PULL_CCU_TRY(WaitChannel(kernel->channel, PULL_NOTIFY, PULL_ROOT_META_MASK));
    return CCU_SUCCESS;
}

CcuResult CcuPrefetchKernel(CcuKernelArg arg)
{
    const auto *kernel = static_cast<const PullPeerArg *>(arg);
    if (!ValidPeer(kernel)) {
        return CCU_E_PARA;
    }
    ccu::Variable args[PF_ARG_COUNT];
    PULL_CCU_TRY(LoadArguments(args, PF_ARG_COUNT));
    ccu::Variable header[ROOT_FIELD_COUNT];
    PULL_CCU_TRY(CaptureWords(kernel->channel, header, ROOT_FIELD_COUNT));
    ccu::Variable authorized;
    PULL_CCU_TRY(SetAuthorization(header, args[PF_EXPECTED_LAYOUT], args[PF_EXPECTED_MASK], authorized));
    return EmitIf(authorized != 0, [&]() -> CcuResult {
        ReadTransfer transfer;
        transfer.remote.addr = header[ROOT_BASE];
        transfer.remote.addr += args[PF_INPUT_OFFSET];
        transfer.remote.token = header[ROOT_TOKEN];
        transfer.local.addr = args[PF_BUFFER];
        transfer.local.addr += args[PF_BUFFER_OFFSET];
        transfer.local.token = args[PF_BUFFER_TOKEN];
        transfer.bytes = args[PF_BYTES];
        return ReadAndWait(kernel->channel, transfer);
    });
}

CcuResult CcuOfferSendKernel(CcuKernelArg arg)
{
    const auto *kernel = static_cast<const PullPeerArg *>(arg);
    if (!ValidPeer(kernel)) {
        return CCU_E_PARA;
    }
    ccu::Variable args[OF_ARG_COUNT];
    PULL_CCU_TRY(LoadArguments(args, OF_ARG_COUNT));
    // One peer and one descriptor per launch. Early layouts describe an
    // interval before prefetch; a separate READY later permits payload use.
    // The legacy 8+4 bands retain their prefetch-before-META ordering.
    return PublishWords(kernel->channel, args, OFFER_FIELD_COUNT, OFFER_NOTIFY);
}

CcuResult CcuOfferGroupKernel(CcuKernelArg arg)
{
    const auto *kernel = static_cast<const PullKernelArg *>(arg);
    if (!ValidNonrootGroup(kernel) ||
        (kernel->peerCount == 1 && kernel->peers[0] == kernel->staticRoot)) {
        return CCU_E_PARA;
    }
    ccu::Variable args[BG_ARG_COUNT];
    PULL_CCU_TRY(LoadArguments(args, BG_ARG_COUNT));
    // A descriptor retains distinct source registers for every peer until
    // the kernel finishes. Never rewrite a previously published bytes/end
    // register while WriteVariableWithNotify can still refer to it.
    std::vector<OfferDescriptor> descriptors(kernel->peerCount);
    for (uint32_t i = 0; i < kernel->peerCount; ++i) {
        if (kernel->peers[i] == kernel->staticRoot) {
            continue;
        }
        auto *words = descriptors[i].words;
        words[OFFER_OFFSET] = 0;
        words[OFFER_BYTES] = 0;
        words[OFFER_NEGATIVE_END] = 0;
        words[OFFER_ADDRESS] = args[BG_ADDRESS];
        words[OFFER_TOKEN] = args[BG_TOKEN];
        words[OFFER_NEGATIVE_LAYOUT] = args[BG_EXPECTED_LAYOUT];
        words[OFFER_NEGATIVE_MASK] = args[BG_EXPECTED_MASK];
        PULL_CCU_TRY(EmitIf(args[BG_TARGET] == kernel->peers[i], [&]() -> CcuResult {
            words[OFFER_BYTES] = args[BG_BYTES];
            words[OFFER_NEGATIVE_END] = args[BG_NEGATIVE_END];
            return CCU_SUCCESS;
        }));
        // Include all seven fields even for zero offers. The receiver still
        // consumes 0x7f from every nonroot channel on its own die.
        PULL_CCU_TRY(PublishWords(kernel->channels[i], words, OFFER_FIELD_COUNT, OFFER_NOTIFY));
    }
    return CCU_SUCCESS;
}

namespace {
CcuResult PublishPreparedOffers(const PullKernelArg &kernel, ccu::Variable *args)
{
    // Each asynchronous publication keeps its own source registers. In the
    // root-channel group these are also independent of the in-flight Read.
    std::vector<OfferDescriptor> descriptors(kernel.peerCount);
    for (uint32_t i = 0; i < kernel.peerCount; ++i) {
        if (kernel.peers[i] == kernel.staticRoot) {
            continue;
        }
        auto *words = descriptors[i].words;
        words[OFFER_OFFSET] = 0;
        words[OFFER_BYTES] = 0;
        words[OFFER_NEGATIVE_END] = 0;
        words[OFFER_ADDRESS] = args[HP_ADDRESS];
        words[OFFER_TOKEN] = args[HP_TOKEN];
        words[OFFER_NEGATIVE_LAYOUT] = args[HP_NEGATIVE_LAYOUT];
        words[OFFER_NEGATIVE_MASK] = args[HP_NEGATIVE_MASK];
        PULL_CCU_TRY(EmitIf(args[HP_TARGET] == kernel.peers[i], [&]() -> CcuResult {
            words[OFFER_BYTES] = args[HP_BYTES];
            words[OFFER_NEGATIVE_END] = args[HP_NEGATIVE_END];
            return CCU_SUCCESS;
        }));
        PULL_CCU_TRY(PublishWords(kernel.channels[i], words, OFFER_FIELD_COUNT, OFFER_NOTIFY));
    }
    return CCU_SUCCESS;
}
} // namespace

CcuResult CcuTwoByEightPrepareKernel(CcuKernelArg arg)
{
    const auto *kernel = static_cast<const PullKernelArg *>(arg);
    if (!ValidNonrootGroup(kernel) || kernel->rankSize != 16) {
        return CCU_E_PARA;
    }
    ccu::Variable args[HP_ARG_COUNT];
    PULL_CCU_TRY(LoadArguments(args, HP_ARG_COUNT));
    uint32_t rootIndex = kernel->peerCount;
    for (uint32_t i = 0; i < kernel->peerCount; ++i) {
        if (kernel->peers[i] == kernel->staticRoot) {
            rootIndex = i;
        }
    }
    if (rootIndex == kernel->peerCount) {
        // This die cannot access the root header. It still loads all HP8
        // arguments and publishes the raw complete descriptors unconditionally.
        return PublishPreparedOffers(*kernel, args);
    }
    const ChannelHandle rootChannel = kernel->channels[rootIndex];
    PULL_CCU_TRY(WaitChannel(rootChannel, PULL_NOTIFY, PULL_ROOT_META_MASK));
    ccu::Variable header[ROOT_FIELD_COUNT];
    PULL_CCU_TRY(CaptureWords(rootChannel, header, ROOT_FIELD_COUNT));
    ccu::Variable authorized;
    PULL_CCU_TRY(SetAuthorization(header, args[HP_NEGATIVE_LAYOUT], args[HP_NEGATIVE_MASK], authorized));
    ReadTransfer transfer;
    ccu::Event completed;
    ccu::Variable readIssued;
    readIssued = 0;
    PULL_CCU_TRY(EmitIf(authorized != 0, [&]() -> CcuResult {
        return EmitIf(args[HP_BYTES] != 0, [&]() -> CcuResult {
            transfer.remote.addr = header[ROOT_BASE];
            transfer.remote.addr += args[HP_SOURCE_OFFSET];
            transfer.remote.token = header[ROOT_TOKEN];
            transfer.local.addr = args[HP_ADDRESS];
            transfer.local.token = args[HP_TOKEN];
            transfer.bytes = args[HP_BYTES];
            PULL_CCU_TRY(ccu::Read(rootChannel, transfer.local, transfer.remote, transfer.bytes, completed));
            readIssued = 1;
            return CCU_SUCCESS;
        });
    }));
    // Even rejected authorization must publish the original nonzero promise.
    // Host sends its READY once after joining both preparation groups.
    PULL_CCU_TRY(PublishPreparedOffers(*kernel, args));
    return EmitIf(readIssued != 0, [&]() -> CcuResult { return ccu::EventWait(completed); });
}

CcuResult CcuOfferReadyKernel(CcuKernelArg arg)
{
    const auto *kernel = static_cast<const PullPeerArg *>(arg);
    if (!ValidPeer(kernel)) {
        return CCU_E_PARA;
    }
    // Host enqueues one READY after the prefetch stage for each nonzero early
    // descriptor, even when the root header rejected that prefetch's layout.
    return ccu::NotifyRecord(kernel->channel, OFFER_NOTIFY, OFFER_READY_MASK);
}

namespace {
CcuResult BuildPullGroupKernel(CcuKernelArg arg, bool allowEarlyDirect)
{
    const auto *kernel = static_cast<const PullKernelArg *>(arg);
    if (!ValidNonrootGroup(kernel)) {
        return CCU_E_PARA;
    }
    ccu::Variable args[GT_ARG_COUNT];
    PULL_CCU_TRY(LoadArguments(args, GT_ARG_COUNT));
    uint32_t rootIndex = kernel->peerCount;
    for (uint32_t i = 0; i < kernel->peerCount; ++i) {
        if (kernel->peers[i] == kernel->staticRoot) {
            rootIndex = i;
        }
    }
    if (allowEarlyDirect && kernel->rankSize == 12 && rootIndex != kernel->peerCount) {
        // RootCapture has already consumed root META on this die. Its XNs
        // stay stable through ROOT_DONE, so capture before waiting for offers.
        ccu::Variable header[ROOT_FIELD_COUNT];
        PULL_CCU_TRY(CaptureWords(kernel->channels[rootIndex], header, ROOT_FIELD_COUNT));
        ccu::Variable earlyDirect;
        earlyDirect = 0;
        PULL_CCU_TRY(EmitIf(header[ROOT_LAYOUT] == static_cast<uint64_t>(PullLayout::DIRECT), [&]() -> CcuResult {
            earlyDirect = header[ROOT_MASK];
            return CCU_SUCCESS;
        }));
        // In a valid 12-rank context, only 8+4/root8 has DIRECT and a nonzero
        // root mask. UNKNOWN has mask zero; 4x3 and root4 use prefix layouts.
        return EmitIfElse(earlyDirect != 0,
            [&]() -> CcuResult { return ReadDirectFrameAndDrainOffers(*kernel, args, header, rootIndex); },
            [&]() -> CcuResult {
                for (uint32_t i = 0; i < kernel->peerCount; ++i) {
                    if (i != rootIndex) {
                        PULL_CCU_TRY(WaitChannel(kernel->channels[i], OFFER_NOTIFY, OFFER_META_MASK));
                    }
                }
                return ReadRelayedFrame(*kernel, args, header, rootIndex);
            });
    }
    for (uint32_t i = 0; i < kernel->peerCount; ++i) {
        if (i != rootIndex) {
            PULL_CCU_TRY(WaitChannel(kernel->channels[i], OFFER_NOTIFY, OFFER_META_MASK));
        }
    }
    if (rootIndex == kernel->peerCount) {
        // This die cannot authorize helper data against the root header.
        // All META is consumed above. Drain every promised READY as well;
        // the root die supplies these inaccessible intervals directly.
        ccu::Variable descriptor[OFFER_FIELD_COUNT];
        ccu::Variable readyLayoutDifference;
        for (uint32_t i = 0; i < kernel->peerCount; ++i) {
            PULL_CCU_TRY(CaptureOfferDescriptor(kernel->channels[i], descriptor, kernel->rankSize));
            PULL_CCU_TRY(ConsumeOfferReady(kernel->channels[i], descriptor, readyLayoutDifference));
        }
        return CCU_SUCCESS;
    }
    ccu::Variable header[ROOT_FIELD_COUNT];
    PULL_CCU_TRY(CaptureWords(kernel->channels[rootIndex], header, ROOT_FIELD_COUNT));
    // No kernel accesses another die's XNs. The root die supplies every
    // rejected or inaccessible interval directly.
    PULL_CCU_TRY(ReadRelayedFrame(*kernel, args, header, rootIndex));
    return CCU_SUCCESS;
}
} // namespace

CcuResult CcuPullGroupKernel(CcuKernelArg arg)
{
    return BuildPullGroupKernel(arg, false);
}

CcuResult CcuEightPlusFourPullGroupKernel(CcuKernelArg arg)
{
    return BuildPullGroupKernel(arg, true);
}

CcuResult CcuTwoByEightPrefixPullGroupKernel(CcuKernelArg arg)
{
    const auto *registration = static_cast<const PrefixRegArg *>(arg);
    if (registration == nullptr || !ValidNonrootGroup(&registration->kernel)) {
        return CCU_E_PARA;
    }
    const auto &kernel = registration->kernel;
    const uint32_t rootMask = registration->expectedRootMask;
    const uint32_t rootIndex = FindPeerIndex(kernel, kernel.staticRoot);
    if (kernel.rankSize != 16 || rootIndex == kernel.peerCount ||
        (rootMask & ~AllRankMask(kernel.rankSize)) != 0 || RankMaskSize(rootMask) != 8 ||
        (rootMask & RankBit(kernel.staticRoot)) == 0) {
        return CCU_E_PARA;
    }
    ccu::Variable args[GT_ARG_COUNT];
    PULL_CCU_TRY(LoadArguments(args, GT_ARG_COUNT));
    // Host already consumed root META. Take a stable snapshot on its actual die
    // before any offer is consumed, so a header mismatch can use the old body.
    ccu::Variable header[ROOT_FIELD_COUNT];
    PULL_CCU_TRY(CaptureWords(kernel.channels[rootIndex], header, ROOT_FIELD_COUNT));
    ccu::Variable usePrefix;
    usePrefix = 0;
    PULL_CCU_TRY(EmitIf(header[ROOT_LAYOUT] == static_cast<uint64_t>(PullLayout::TWO_BY_EIGHT_PREFIXES),
        [&]() -> CcuResult {
            return EmitIf(header[ROOT_MASK] == rootMask, [&]() -> CcuResult {
                usePrefix = 1;
                return CCU_SUCCESS;
            });
        }));
    const uint32_t supplierIndex = FindPeerIndex(kernel, PrefixSupplier(kernel, rootMask));
    return EmitIfElse(usePrefix != 0,
        [&]() -> CcuResult {
            if (supplierIndex == kernel.peerCount) {
                // Local helper, eighth remote, or supplier on the other die:
                // this group reads the entire output frame from root.
                return ReadDirectFrameAndDrainOffers(kernel, args, header, rootIndex);
            }
            return ReadPairedPrefixFrame(kernel, args, header, rootIndex, supplierIndex);
        },
        [&]() -> CcuResult {
            PULL_CCU_TRY(WaitOfferMetadata(kernel, rootIndex, kernel.peerCount));
            return ReadRelayedFrame(kernel, args, header, rootIndex);
        });
}

CcuResult CcuEightPlusFourFusedPullGroupKernel(CcuKernelArg arg)
{
    const auto *kernel = static_cast<const PullKernelArg *>(arg);
    // A root-only group is valid: it still consumes META and reads the output.
    if (!ValidNonrootGroup(kernel) || kernel->rankSize != 12) {
        return CCU_E_PARA;
    }
    ccu::Variable args[GT_ARG_COUNT];
    PULL_CCU_TRY(LoadArguments(args, GT_ARG_COUNT));
    ccu::Variable zero;
    zero = 0;
    const uint32_t rootIndex = FindPeerIndex(*kernel, kernel->staticRoot);
    if (rootIndex == kernel->peerCount) {
        // No cross-die XN access. Publish everything before waiting for peers;
        // LaunchGroups executes this group concurrently with the root group.
        PULL_CCU_TRY(PublishZeroOffers(*kernel, zero));
        PULL_CCU_TRY(WaitOfferMetadata(*kernel, rootIndex, kernel->peerCount));
        return DrainOfferReadiness(*kernel, rootIndex, kernel->peerCount);
    }
    // The fused Host path skipped Capture: this is the sole root META wait.
    PULL_CCU_TRY(WaitChannel(kernel->channels[rootIndex], PULL_NOTIFY, PULL_ROOT_META_MASK));
    ccu::Variable header[ROOT_FIELD_COUNT];
    PULL_CCU_TRY(CaptureWords(kernel->channels[rootIndex], header, ROOT_FIELD_COUNT));
    ccu::Variable earlyDirect;
    earlyDirect = 0;
    PULL_CCU_TRY(EmitIf(header[ROOT_LAYOUT] == static_cast<uint64_t>(PullLayout::DIRECT), [&]() -> CcuResult {
        earlyDirect = header[ROOT_MASK];
        return CCU_SUCCESS;
    }));
    return EmitIfElse(earlyDirect != 0,
        [&]() -> CcuResult {
            return ReadDirectFrameAndPublishOffers(*kernel, args, header, rootIndex, zero);
        },
        [&]() -> CcuResult {
            // A locally inferred root8 plan is always zero, even if root uses
            // another header. All N12 receivers accept zero bytes with the
            // full META mask, including UNKNOWN and the generic fallback.
            PULL_CCU_TRY(PublishZeroOffers(*kernel, zero));
            PULL_CCU_TRY(WaitOfferMetadata(*kernel, rootIndex, kernel->peerCount));
            return ReadRelayedFrame(*kernel, args, header, rootIndex);
        });
}

CcuResult CcuRootAckKernel(CcuKernelArg arg)
{
    const auto *kernel = static_cast<const PullPeerArg *>(arg);
    if (!ValidPeer(kernel)) {
        return CCU_E_PARA;
    }
    PULL_CCU_TRY(ccu::NotifyRecord(kernel->channel, PULL_NOTIFY, PULL_DONE_MASK));
    PULL_CCU_TRY(WaitChannel(kernel->channel, PULL_NOTIFY, ROOT_RELEASE_MASK));
    return CCU_SUCCESS;
}

CcuResult CcuCopyKernel(CcuKernelArg /*arg*/)
{
    ccu::Variable args[COPY_ARG_COUNT];
    PULL_CCU_TRY(LoadArguments(args, COPY_ARG_COUNT));
    return CopyChunk(args[COPY_SOURCE], args[COPY_SOURCE_TOKEN], args[COPY_OUTPUT], args[COPY_OUTPUT_TOKEN],
        args[COPY_BUFFER], args[COPY_BUFFER_TOKEN], args[COPY_BYTES], args[COPY_STAGING]);
}

#undef PULL_CCU_TRY

} // namespace ops_hccl
