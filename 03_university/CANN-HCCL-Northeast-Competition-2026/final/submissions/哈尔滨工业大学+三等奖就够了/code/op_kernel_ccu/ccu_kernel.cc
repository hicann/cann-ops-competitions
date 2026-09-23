/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <ccu/ccu_primitives.hpp>
#include <algorithm>
#include <vector>
#include "custom.h"
#include "ccu_kernel.h"

#define CCU_CHK_RET(call)                         \
    do {                                        \
        const CcuResult result = (call);         \
        if (result != CCU_SUCCESS) {             \
            return result;                      \
        }                                       \
    } while (0)

namespace ops_hccl {
namespace ccu = AscendC::ccu;
namespace {
constexpr uint32_t INPUT_SLOT = 0;
constexpr uint32_t TOKEN_SLOT = 1;
constexpr uint16_t INPUT_READY = 1;
constexpr uint16_t TOKEN_READY = 2;
constexpr uint16_t READ_DONE = 4;
constexpr uint16_t RELEASE = 8;
constexpr uint16_t LARGE_READY = 1 | 2 | 16 | 32;

CcuResult CopyRoot(ccu::Variable input, ccu::Variable output, ccu::Variable inputToken,
    ccu::Variable outputToken, ccu::Variable offset, ccu::Variable fullChunks, ccu::Variable tail)
{
    ccu::LocalAddr src, dst;
    ccu::Variable remaining, chunk, decrement;
    ccu::Event copied;
    src.addr = input;
    src.addr += offset;
    src.token = inputToken;
    dst.addr = output;
    dst.token = outputToken;
    remaining = fullChunks;
    decrement = UINT64_MAX; // CCU 9.1 supports addition; unsigned addition wraps modulo 2^64.
    chunk = SCATTER_TRANSFER_BYTES;
    CCU_WHILE(remaining != 0) {
        CCU_CHK_RET(ccu::LocalCopy(dst, src, chunk, copied, 1));
        CCU_CHK_RET(ccu::EventWait(copied, 1));
        src.addr += chunk;
        dst.addr += chunk;
        remaining += decrement;
    }
    CCU_IF(tail != 0) {
        CCU_CHK_RET(ccu::LocalCopy(dst, src, tail, copied, 1));
        CCU_CHK_RET(ccu::EventWait(copied, 1));
    }
    return CCU_SUCCESS;
}

CcuResult ReadSlice(ChannelHandle channel, ccu::Variable output, ccu::Variable outputToken,
    ccu::Variable offset, ccu::Variable fullChunks, ccu::Variable tail)
{
    ccu::Variable input = ccu::GetResByChannel<ccu::Variable>(channel, INPUT_SLOT);
    ccu::Variable token = ccu::GetResByChannel<ccu::Variable>(channel, TOKEN_SLOT);
    ccu::RemoteAddr src;
    ccu::LocalAddr dst;
    ccu::Variable remaining, chunk, decrement;
    ccu::Event read;
    src.addr = input;
    src.addr += offset;
    src.token = token;
    dst.addr = output;
    dst.token = outputToken;
    remaining = fullChunks;
    decrement = UINT64_MAX; // CCU 9.1 supports addition; unsigned addition wraps modulo 2^64.
    chunk = SCATTER_TRANSFER_BYTES;
    CCU_WHILE(remaining != 0) {
        CCU_CHK_RET(ccu::Read(channel, dst, src, chunk, read, 1));
        CCU_CHK_RET(ccu::EventWait(read, 1));
        src.addr += chunk;
        dst.addr += chunk;
        remaining += decrement;
    }
    CCU_IF(tail != 0) {
        CCU_CHK_RET(ccu::Read(channel, dst, src, tail, read, 1));
        CCU_CHK_RET(ccu::EventWait(read, 1));
    }
    return CCU_SUCCESS;
}
CcuResult Publish(ChannelHandle channel, ccu::Variable address, ccu::Variable token,
    ccu::Variable length, ccu::Variable offset)
{
    CCU_CHK_RET(ccu::WriteVariableWithNotify(channel, address, INPUT_SLOT, 0, INPUT_READY));
    CCU_CHK_RET(ccu::WriteVariableWithNotify(channel, token, TOKEN_SLOT, 0, TOKEN_READY));
    CCU_CHK_RET(ccu::WriteVariableWithNotify(channel, length, 2, 0, 16));
    CCU_CHK_RET(ccu::WriteVariableWithNotify(channel, offset, 3, 0, 32));
    return CCU_SUCCESS;
}

CcuResult PublishGroup(const ScatterKernelArg &params, ccu::Variable input, ccu::Variable inputToken,
    ccu::Variable scratch, ccu::Variable scratchToken, ccu::Variable q, ccu::Variable direct)
{
    ccu::Variable zero;
    zero = 0;
    if (params.rankId == params.root) {
        for (uint32_t peer = 0; peer < params.rankSize; ++peer) {
            if (peer != params.rankId && params.channels[peer] != 0) {
                CCU_CHK_RET(Publish(params.channels[peer], input, inputToken, direct, zero));
            }
        }
        return CCU_SUCCESS;
    }
    if (!ScatterIsLocal(params.localMask, params.root)) {
        for (uint32_t peer = 0; peer < params.rankSize; ++peer) {
            if (peer != params.rankId && params.channels[peer] != 0) {
                CCU_CHK_RET(Publish(params.channels[peer], zero, zero, zero, zero));
            }
        }
        return CCU_SUCCESS;
    }
    ccu::Variable cursor, destinationOffset;
    cursor = scratch;
    destinationOffset = 0;
    if (ScatterIsLocal(params.localMask, params.root)) {
        CCU_IF(q != 0) {
            destinationOffset = direct;
            for (uint32_t peer = 0; peer < params.rankId; ++peer) {
                if (peer != params.root && ScatterIsLocal(params.localMask, peer)) {
                    destinationOffset += q;
                }
            }
        }
    }
    for (uint32_t peer = 0; peer < params.rankSize; ++peer) {
        if (peer != params.rankId && params.channels[peer] != 0) {
            const ChannelHandle channel = params.channels[peer];
            if (ScatterIsLocal(params.localMask, params.root)
                && !ScatterIsLocal(params.localMask, peer)) {
                CCU_CHK_RET(Publish(channel, cursor, scratchToken, q, destinationOffset));
            } else {
                CCU_CHK_RET(Publish(channel, zero, zero, zero, zero));
            }
        }
        if (!ScatterIsLocal(params.localMask, peer)) {
            cursor += q;
        }
    }
    return CCU_SUCCESS;
}

CcuResult StageTails(const ScatterKernelArg &params, ccu::Variable scratch, ccu::Variable scratchToken,
    ccu::Variable q, ccu::Variable direct)
{
    struct Transfer {
        ccu::RemoteAddr src;
        ccu::LocalAddr dst;
    };
    const ChannelHandle channel = params.channels[params.root];
    ccu::Variable input = ccu::GetResByChannel<ccu::Variable>(channel, INPUT_SLOT);
    ccu::Variable token = ccu::GetResByChannel<ccu::Variable>(channel, TOKEN_SLOT);
    constexpr uint32_t BATCH = 8;
    const uint32_t remoteCount = params.rankSize - ScatterLocalCount(params.localMask, params.rankSize);
    std::vector<Transfer> transfers(std::min<uint32_t>(remoteCount, BATCH));
    ccu::Event event;
    ccu::Variable cursor, bytes, destination;
    CCU_IF(q != 0) {
        bytes = direct;
        for (uint32_t peer = 0; peer < params.rankSize; ++peer) {
            if (peer != params.root && ScatterIsLocal(params.localMask, peer)) {
                bytes += q;
            }
        }
        cursor = input;
        cursor += direct;
        for (uint32_t peer = 0; peer < params.rankId; ++peer) {
            if (peer != params.root && ScatterIsLocal(params.localMask, peer)) {
                cursor += q;
            }
        }
        destination = scratch;
        for (auto &transfer : transfers) {
            transfer.src.token = token;
            transfer.dst.token = scratchToken;
        }
        uint32_t peer = 0;
        while (peer < params.rankSize) {
            uint32_t count = 0;
            for (; peer < params.rankSize && count < BATCH; ++peer) {
                if (peer != 0) {
                    cursor += bytes;
                }
                if (!ScatterIsLocal(params.localMask, peer)) {
                    auto &transfer = transfers[count];
                    transfer.src.addr = cursor;
                    transfer.dst.addr = destination;
                    const uint16_t mask = static_cast<uint16_t>(uint32_t(1) << count);
                    CCU_CHK_RET(ccu::Read(channel, transfer.dst, transfer.src, q, event, mask));
                    destination += q;
                    ++count;
                }
            }
            // Keep each address stable until the entire batch finishes, then reuse
            // the same addresses and event bits within this STAGE launch.
            for (uint32_t index = 0; index < count; ++index) {
                const uint16_t mask = static_cast<uint16_t>(uint32_t(1) << index);
                CCU_CHK_RET(ccu::EventWait(event, mask));
            }
        }
    }
    return CCU_SUCCESS;
}

CcuResult ReadProviders(const ScatterKernelArg &params, ccu::Variable output, ccu::Variable outputToken,
    ccu::Variable ownOffset)
{
    struct Transfer {
        ccu::RemoteAddr src;
        ccu::LocalAddr dst;
    };
    std::vector<uint32_t> peers;
    if (params.channels[params.root] != 0) {
        peers.push_back(params.root);
    }
    for (uint32_t peer = 0; peer < params.rankSize; ++peer) {
        if (peer != params.rankId && peer != params.root && params.channels[peer] != 0) {
            peers.push_back(peer);
        }
    }
    // Allocate once: later C++ batches reuse the same static CCU addresses and
    // event bits only after all earlier positive-length reads have completed.
    constexpr uint32_t BATCH = 16;
    std::vector<Transfer> transfers(std::min<size_t>(peers.size(), BATCH));
    ccu::Event completed;
    for (uint32_t begin = 0; begin < peers.size(); begin += BATCH) {
        const uint32_t count = std::min<uint32_t>(BATCH, peers.size() - begin);
        for (uint32_t index = 0; index < count; ++index) {
            const uint32_t peer = peers[begin + index];
            const ChannelHandle channel = params.channels[peer];
            CCU_CHK_RET(ccu::NotifyWait(channel, 0, LARGE_READY));
            ccu::Variable address = ccu::GetResByChannel<ccu::Variable>(channel, INPUT_SLOT);
            ccu::Variable token = ccu::GetResByChannel<ccu::Variable>(channel, TOKEN_SLOT);
            ccu::Variable length = ccu::GetResByChannel<ccu::Variable>(channel, 2);
            ccu::Variable offset = ccu::GetResByChannel<ccu::Variable>(channel, 3);
            CCU_IF(length != 0) {
                auto &transfer = transfers[index];
                transfer.src.addr = address;
                if (peer == params.root) {
                    transfer.src.addr += ownOffset;
                }
                transfer.src.token = token;
                transfer.dst.addr = output;
                transfer.dst.addr += offset;
                transfer.dst.token = outputToken;
                const uint16_t mask = static_cast<uint16_t>(uint32_t(1) << index);
                CCU_CHK_RET(ccu::Read(channel, transfer.dst, transfer.src, length, completed, mask));
            }
        }
        for (uint32_t index = 0; index < count; ++index) {
            const ChannelHandle channel = params.channels[peers[begin + index]];
            ccu::Variable length = ccu::GetResByChannel<ccu::Variable>(channel, 2);
            CCU_IF(length != 0) {
                const uint16_t mask = static_cast<uint16_t>(uint32_t(1) << index);
                CCU_CHK_RET(ccu::EventWait(completed, mask));
            }
        }
    }
    return CCU_SUCCESS;
}
} // namespace

CcuResult CcuKernel(CcuKernelArg arg)
{
    const auto *params = static_cast<const ScatterKernelArg *>(arg);
    if (params == nullptr || params->rankSize == 0 || params->rankSize > SCATTER_MAX_RANKS
        || params->rankId >= params->rankSize || params->root >= params->rankSize
        || !ScatterIsLocal(params->localMask, params->rankId)) {
        return CCU_E_PARA;
    }
    // rankSize is fixed at registration; this graph never needs a relay phase.
    if (params->rankSize <= 5 && params->rankId != params->root) {
        ccu::Variable output, outputToken, offset, fullChunks, tail;
        CCU_CHK_RET(ccu::LoadArg(output, 0));
        CCU_CHK_RET(ccu::LoadArg(outputToken, 1));
        CCU_CHK_RET(ccu::LoadArg(offset, 2));
        CCU_CHK_RET(ccu::LoadArg(fullChunks, 3));
        CCU_CHK_RET(ccu::LoadArg(tail, 4));
        if (params->channels[params->root] != 0) {
            const ChannelHandle channel = params->channels[params->root];
            CCU_CHK_RET(ccu::NotifyWait(channel, 0, INPUT_READY | TOKEN_READY));
            CCU_CHK_RET(ReadSlice(channel, output, outputToken, offset, fullChunks, tail));
            CCU_CHK_RET(ccu::NotifyRecord(channel, 0, READ_DONE));
        }
        return CCU_SUCCESS;
    }
    ccu::Variable input, output, outputToken, phase, offset, fullChunks, tail;
    if (params->rankId == params->root) {
        ccu::Variable inputToken, copyRoot, publishSmall, waitPhase, direct;
        CCU_CHK_RET(ccu::LoadArg(input, 0));
        CCU_CHK_RET(ccu::LoadArg(output, 1));
        CCU_CHK_RET(ccu::LoadArg(inputToken, 2));
        CCU_CHK_RET(ccu::LoadArg(outputToken, 3));
        if (params->rankSize <= 5) {
            CCU_CHK_RET(ccu::LoadArg(offset, 4));
            CCU_CHK_RET(ccu::LoadArg(copyRoot, 5));
            CCU_CHK_RET(ccu::LoadArg(fullChunks, 6));
            CCU_CHK_RET(ccu::LoadArg(tail, 7));
            CCU_CHK_RET(ccu::LoadArg(publishSmall, 8));
            CCU_CHK_RET(ccu::LoadArg(waitPhase, 9));
        } else {
            CCU_CHK_RET(ccu::LoadArg(phase, 4));
            CCU_CHK_RET(ccu::LoadArg(offset, 5));
            CCU_CHK_RET(ccu::LoadArg(copyRoot, 6));
            CCU_CHK_RET(ccu::LoadArg(fullChunks, 7));
            CCU_CHK_RET(ccu::LoadArg(tail, 8));
            CCU_CHK_RET(ccu::LoadArg(publishSmall, 9));
            CCU_CHK_RET(ccu::LoadArg(waitPhase, 10));
            CCU_CHK_RET(ccu::LoadArg(direct, 11));
        }
        // Host supplies 0/1 flags; equality avoids the longer V1 inequality lowering.
        CCU_IF(publishSmall == 1) {
            for (uint32_t peer = 0; peer < params->rankSize; ++peer) {
                if (peer != params->rankId && params->channels[peer] != 0) {
                    const ChannelHandle channel = params->channels[peer];
                    CCU_CHK_RET(ccu::WriteVariableWithNotify(channel, input, INPUT_SLOT, 0, INPUT_READY));
                }
            }
            // Interleave independent channels before sending the second metadata word.
            // Each receiver still waits for both atomic value/notification writes.
            for (uint32_t peer = 0; peer < params->rankSize; ++peer) {
                if (peer != params->rankId && params->channels[peer] != 0) {
                    const ChannelHandle channel = params->channels[peer];
                    CCU_CHK_RET(ccu::WriteVariableWithNotify(channel, inputToken, TOKEN_SLOT, 0, TOKEN_READY));
                }
            }
        }
        if (params->rankSize > 5) {
            CCU_IF(phase != 0) {
                CCU_IF(phase == static_cast<uint64_t>(ScatterPhase::PUBLISH)) {
                    CCU_CHK_RET(PublishGroup(*params, input, inputToken, phase, waitPhase, phase, direct));
                }
                CCU_IF(phase == static_cast<uint64_t>(ScatterPhase::ROOT_WAIT)) {
                    for (uint32_t peer = 0; peer < params->rankSize; ++peer) {
                        if (peer != params->rankId && params->channels[peer] != 0) {
                            CCU_CHK_RET(ccu::NotifyWait(params->channels[peer], 0, LARGE_READY));
                        }
                    }
                }
                CCU_IF(phase == static_cast<uint64_t>(ScatterPhase::ROOT_RELEASE)) {
                    for (uint32_t peer = 0; peer < params->rankSize; ++peer) {
                        if (peer != params->rankId && params->channels[peer] != 0) {
                            CCU_CHK_RET(ccu::NotifyRecord(params->channels[peer], 0, RELEASE));
                        }
                    }
                }
            }
        }
        CCU_IF(waitPhase == 1) {
            CCU_IF(copyRoot != static_cast<uint64_t>(ScatterCopyMode::WITH_READS)) {
                for (uint32_t peer = 0; peer < params->rankSize; ++peer) {
                    if (peer != params->rankId && params->channels[peer] != 0) {
                        CCU_CHK_RET(ccu::NotifyWait(params->channels[peer], 0, READ_DONE));
                    }
                }
            }
            // Shared with the direct path: one static copy allocation site.
            CCU_IF(copyRoot != static_cast<uint64_t>(ScatterCopyMode::NONE)) {
                CCU_CHK_RET(CopyRoot(input, output, inputToken, outputToken, offset, fullChunks, tail));
                CCU_IF(copyRoot != static_cast<uint64_t>(ScatterCopyMode::AFTER_READS)) {
                    for (uint32_t peer = 0; peer < params->rankSize; ++peer) {
                        if (peer != params->rankId && params->channels[peer] != 0) {
                            CCU_CHK_RET(ccu::NotifyWait(params->channels[peer], 0, READ_DONE));
                        }
                    }
                }
            }
        }
        return CCU_SUCCESS;
    }

    const bool helper = ScatterIsLocal(params->localMask, params->root);
    CCU_CHK_RET(ccu::LoadArg(output, 0));
    CCU_CHK_RET(ccu::LoadArg(outputToken, 1));
    CCU_CHK_RET(ccu::LoadArg(phase, 2));
    CCU_CHK_RET(ccu::LoadArg(offset, 3));
    CCU_CHK_RET(ccu::LoadArg(fullChunks, 4));
    CCU_CHK_RET(ccu::LoadArg(tail, 5));
    auto runNonRoot = [&](ccu::Variable scratch, ccu::Variable scratchToken,
                          ccu::Variable q, ccu::Variable direct) -> CcuResult {
        ccu::Variable &readOwn = input;
        if (params->channels[params->root] != 0) {
            const ChannelHandle channel = params->channels[params->root];
            CCU_IF(readOwn == 1) {
                if (helper) {
                    CCU_IF(phase == static_cast<uint64_t>(ScatterPhase::SMALL_FULL)) {
                        CCU_CHK_RET(ccu::NotifyWait(channel, 0, INPUT_READY | TOKEN_READY));
                    }
                } else {
                    CCU_CHK_RET(ccu::NotifyWait(channel, 0, INPUT_READY | TOKEN_READY));
                }
                CCU_CHK_RET(ReadSlice(channel, output, outputToken, offset, fullChunks, tail));
                if (helper) {
                    CCU_IF(phase == static_cast<uint64_t>(ScatterPhase::SMALL_FULL)) {
                        CCU_CHK_RET(ccu::NotifyRecord(channel, 0, READ_DONE));
                    }
                } else {
                    CCU_CHK_RET(ccu::NotifyRecord(channel, 0, READ_DONE));
                }
            }
        }
        CCU_IF(readOwn == 0) {
            if (params->channels[params->root] != 0) {
                const ChannelHandle channel = params->channels[params->root];
                if (helper) {
                    CCU_IF(phase == static_cast<uint64_t>(ScatterPhase::STAGE)) {
                        CCU_CHK_RET(ccu::NotifyWait(channel, 0, LARGE_READY));
                        CCU_CHK_RET(StageTails(*params, scratch, scratchToken, q, direct));
                    }
                }
                CCU_IF(phase == static_cast<uint64_t>(ScatterPhase::FINISH)) {
                    CCU_CHK_RET(ccu::NotifyRecord(channel, 0, READ_DONE));
                    CCU_CHK_RET(ccu::NotifyWait(channel, 0, RELEASE));
                }
            }
            CCU_IF(phase == static_cast<uint64_t>(ScatterPhase::PUBLISH)) {
                CCU_CHK_RET(PublishGroup(*params, output, outputToken, scratch, scratchToken, q, direct));
            }
            if (helper) {
                CCU_IF(phase == static_cast<uint64_t>(ScatterPhase::DRAIN)) {
                    for (uint32_t peer = 0; peer < params->rankSize; ++peer) {
                        if (peer != params->rankId && peer != params->root && params->channels[peer] != 0) {
                            CCU_CHK_RET(ccu::NotifyWait(params->channels[peer], 0, LARGE_READY));
                        }
                    }
                }
            } else {
                CCU_IF(phase == static_cast<uint64_t>(ScatterPhase::READ_PROVIDERS)) {
                    CCU_CHK_RET(ReadProviders(*params, output, outputToken, offset));
                }
            }
        }
        return CCU_SUCCESS;
    };
    if (helper) {
        ccu::Variable scratch, scratchToken, q, direct;
        CCU_CHK_RET(ccu::LoadArg(scratch, 6));
        CCU_CHK_RET(ccu::LoadArg(scratchToken, 7));
        CCU_CHK_RET(ccu::LoadArg(q, 8));
        CCU_CHK_RET(ccu::LoadArg(direct, 9));
        CCU_CHK_RET(ccu::LoadArg(input, 10));
        return runNonRoot(scratch, scratchToken, q, direct);
    }
    CCU_CHK_RET(ccu::LoadArg(input, 6));
    ccu::Variable scratch = input;
    ccu::Variable scratchToken = input;
    ccu::Variable q = phase;
    ccu::Variable direct = input;
    return runNonRoot(scratch, scratchToken, q, direct);
}
} // namespace ops_hccl
