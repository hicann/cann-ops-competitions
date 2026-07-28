/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <hcomm/ccu/ccu_primitives.hpp>

#include "ccu_kernel.h"
#include "custom.h"

#define RETURN_IF_CCU_ERROR(command) \
    do { \
        CcuResult result = (command); \
        if (result != CCU_SUCCESS) { \
            return result; \
        } \
    } while (0)

namespace ops_hccl {
namespace {
    namespace ccu = AscendC::ccu;
    constexpr uint32_t ADDRESS_NOTIFY = 0;
    constexpr uint32_t COMPLETE_NOTIFY = 1;
    constexpr uint16_t ADDRESS_MASK = 0x3;
    constexpr uint16_t DATA_TRANSFER_DONE = 0x1;
    constexpr uint16_t SCATTER_RECEIVER_ACK = 0x2;

    CcuResult PublishReceiverAddress(const BroadcastKernelArg &plan, ccu::Variable base, ccu::Variable token)
    {
        if (plan.myRank == plan.root) {
            for (uint32_t index = 0; index < plan.channelCount; ++index) {
                RETURN_IF_CCU_ERROR(ccu::NotifyWait(plan.channels[index], ADDRESS_NOTIFY, ADDRESS_MASK));
            }
        } else {
            RETURN_IF_CCU_ERROR(ccu::WriteVariableWithNotify(plan.channels[0], base, 0, ADDRESS_NOTIFY, 0x1));
            RETURN_IF_CCU_ERROR(ccu::WriteVariableWithNotify(plan.channels[0], token, 1, ADDRESS_NOTIFY, 0x2));
        }
        return CCU_SUCCESS;
    }

    CcuResult ExchangeAddresses(const HierarchicalBroadcastKernelArg &plan, ccu::Variable base, ccu::Variable token)
    {
        for (uint32_t index = 0; index < plan.channelCount; ++index) {
            RETURN_IF_CCU_ERROR(ccu::WriteVariableWithNotify(plan.channels[index], base, 0, ADDRESS_NOTIFY, 0x1));
            RETURN_IF_CCU_ERROR(ccu::WriteVariableWithNotify(plan.channels[index], token, 1, ADDRESS_NOTIFY, 0x2));
        }
        for (uint32_t index = 0; index < plan.channelCount; ++index) {
            RETURN_IF_CCU_ERROR(ccu::NotifyWait(plan.channels[index], ADDRESS_NOTIFY, ADDRESS_MASK));
        }
        return CCU_SUCCESS;
    }

    uint64_t ChunkBytes(const HierarchicalBroadcastKernelArg &plan)
    {
        return (plan.dataBytes / sizeof(float) / plan.groupSize) * sizeof(float);
    }

    uint64_t ChunkOffset(const HierarchicalBroadcastKernelArg &plan, uint32_t index)
    {
        return static_cast<uint64_t>(index) * ChunkBytes(plan);
    }

    uint64_t ChunkSize(const HierarchicalBroadcastKernelArg &plan, uint32_t index)
    {
        if (index + 1U == plan.groupSize) {
            return plan.dataBytes - ChunkOffset(plan, index);
        }
        return ChunkBytes(plan);
    }

    CcuResult RunScatterSend(const HierarchicalBroadcastKernelArg &plan, ccu::Variable base, ccu::Variable token)
    {
        RETURN_IF_CCU_ERROR(ExchangeAddresses(plan, base, token));
        ccu::Event event;
        uint16_t eventMask = 0;
        for (uint32_t channel = 0; channel < plan.channelCount; ++channel) {
            ccu::Variable offset;
            ccu::Variable length;
            offset = plan.transferOffsets[channel];
            length = plan.transferLengths[channel];
            ccu::LocalAddr source;
            source.addr = base;
            source.addr += offset;
            source.token = token;
            ccu::RemoteAddr destination;
            destination.addr = ccu::GetResByChannel<ccu::Variable>(plan.channels[channel], 0);
            destination.addr += offset;
            destination.token = ccu::GetResByChannel<ccu::Variable>(plan.channels[channel], 1);
            const uint16_t eventBit = static_cast<uint16_t>(1U << channel);
            RETURN_IF_CCU_ERROR(ccu::Write(plan.channels[channel], destination, source, length, event, eventBit));
            eventMask = static_cast<uint16_t>(eventMask | eventBit);
        }
        RETURN_IF_CCU_ERROR(ccu::EventWait(event, eventMask));
        for (uint32_t channel = 0; channel < plan.channelCount; ++channel) {
            RETURN_IF_CCU_ERROR(ccu::NotifyRecord(plan.channels[channel], COMPLETE_NOTIFY, DATA_TRANSFER_DONE));
        }
        for (uint32_t channel = 0; channel < plan.channelCount; ++channel) {
            RETURN_IF_CCU_ERROR(ccu::NotifyWait(plan.channels[channel], COMPLETE_NOTIFY, SCATTER_RECEIVER_ACK));
        }
        return CCU_SUCCESS;
    }

    CcuResult RunScatterRecv(const HierarchicalBroadcastKernelArg &plan, ccu::Variable base, ccu::Variable token)
    {
        RETURN_IF_CCU_ERROR(ExchangeAddresses(plan, base, token));
        for (uint32_t channel = 0; channel < plan.channelCount; ++channel) {
            RETURN_IF_CCU_ERROR(ccu::NotifyWait(plan.channels[channel], COMPLETE_NOTIFY, DATA_TRANSFER_DONE));
        }
        for (uint32_t channel = 0; channel < plan.channelCount; ++channel) {
            RETURN_IF_CCU_ERROR(ccu::NotifyRecord(plan.channels[channel], COMPLETE_NOTIFY, SCATTER_RECEIVER_ACK));
        }
        return CCU_SUCCESS;
    }

    CcuResult RunAllGather(const HierarchicalBroadcastKernelArg &plan, ccu::Variable base, ccu::Variable token)
    {
        RETURN_IF_CCU_ERROR(ExchangeAddresses(plan, base, token));
        ccu::Variable offset;
        ccu::Variable length;
        offset = ChunkOffset(plan, plan.myIndex);
        length = ChunkSize(plan, plan.myIndex);
        ccu::LocalAddr source;
        source.addr = base;
        source.addr += offset;
        source.token = token;
        ccu::Event event;
        uint16_t eventMask = 0;
        for (uint32_t channel = 0; channel < plan.channelCount; ++channel) {
            ccu::RemoteAddr destination;
            destination.addr = ccu::GetResByChannel<ccu::Variable>(plan.channels[channel], 0);
            destination.addr += offset;
            destination.token = ccu::GetResByChannel<ccu::Variable>(plan.channels[channel], 1);
            const uint16_t eventBit = static_cast<uint16_t>(1U << channel);
            RETURN_IF_CCU_ERROR(ccu::Write(plan.channels[channel], destination, source, length, event, eventBit));
            eventMask = static_cast<uint16_t>(eventMask | eventBit);
        }
        RETURN_IF_CCU_ERROR(ccu::EventWait(event, eventMask));
        for (uint32_t channel = 0; channel < plan.channelCount; ++channel) {
            RETURN_IF_CCU_ERROR(ccu::NotifyRecord(plan.channels[channel], COMPLETE_NOTIFY, DATA_TRANSFER_DONE));
        }
        for (uint32_t channel = 0; channel < plan.channelCount; ++channel) {
            RETURN_IF_CCU_ERROR(ccu::NotifyWait(plan.channels[channel], COMPLETE_NOTIFY, DATA_TRANSFER_DONE));
        }
        return CCU_SUCCESS;
    }
} // namespace

CcuResult BroadcastKernel(CcuKernelArg arg)
{
    auto *plan = static_cast<BroadcastKernelArg *>(arg);
    if (plan == nullptr || plan->channelCount == 0 || plan->channelCount > BROADCAST_MAX_PEERS) {
        return CCU_E_PARA;
    }

    ccu::Variable localBase;
    ccu::Variable localToken;
    ccu::Variable offset;
    ccu::Variable length;
    RETURN_IF_CCU_ERROR(ccu::LoadArg(localBase, 0));
    RETURN_IF_CCU_ERROR(ccu::LoadArg(localToken, 1));
    RETURN_IF_CCU_ERROR(ccu::LoadArg(offset, 2));
    RETURN_IF_CCU_ERROR(ccu::LoadArg(length, 3));
    RETURN_IF_CCU_ERROR(PublishReceiverAddress(*plan, localBase, localToken));

    if (plan->myRank == plan->root) {
        ccu::LocalAddr source;
        source.addr = localBase;
        source.addr += offset;
        source.token = localToken;
        ccu::Event writeEvent;
        uint16_t writeMask = 0;
        for (uint32_t index = 0; index < plan->channelCount; ++index) {
            ccu::RemoteAddr destination;
            destination.addr = ccu::GetResByChannel<ccu::Variable>(plan->channels[index], 0);
            destination.addr += offset;
            destination.token = ccu::GetResByChannel<ccu::Variable>(plan->channels[index], 1);
            const uint16_t eventBit = static_cast<uint16_t>(1U << index);
            RETURN_IF_CCU_ERROR(ccu::Write(plan->channels[index], destination, source, length, writeEvent, eventBit));
            writeMask = static_cast<uint16_t>(writeMask | eventBit);
        }
        RETURN_IF_CCU_ERROR(ccu::EventWait(writeEvent, writeMask));
        for (uint32_t index = 0; index < plan->channelCount; ++index) {
            RETURN_IF_CCU_ERROR(ccu::NotifyRecord(plan->channels[index], COMPLETE_NOTIFY, DATA_TRANSFER_DONE));
        }
    } else {
        // The root records completion only after every remote write becomes visible.
        RETURN_IF_CCU_ERROR(ccu::NotifyWait(plan->channels[0], COMPLETE_NOTIFY, DATA_TRANSFER_DONE));
    }
    return CCU_SUCCESS;
}

CcuResult HierarchicalBroadcastKernel(CcuKernelArg arg)
{
    auto *plan = static_cast<HierarchicalBroadcastKernelArg *>(arg);
    if (plan == nullptr) {
        return CCU_E_PARA;
    }
    // All registered phases use the same launch ABI.  Load before the
    // no-op branch so a rank outside the active phase still consumes two args.
    ccu::Variable base;
    ccu::Variable token;
    RETURN_IF_CCU_ERROR(ccu::LoadArg(base, 0));
    RETURN_IF_CCU_ERROR(ccu::LoadArg(token, 1));
    if (plan->mode == BroadcastKernelMode::NOOP) {
        return CCU_SUCCESS;
    }
    if (plan->groupSize == 0 || plan->channelCount == 0 || plan->channelCount > BROADCAST_MAX_PEERS) {
        return CCU_E_PARA;
    }
    switch (plan->mode) {
        case BroadcastKernelMode::SCATTER_SEND:
            return RunScatterSend(*plan, base, token);
        case BroadcastKernelMode::SCATTER_RECV:
            return RunScatterRecv(*plan, base, token);
        case BroadcastKernelMode::ALLGATHER:
            return RunAllGather(*plan, base, token);
        default:
            return CCU_E_PARA;
    }
}
} // namespace ops_hccl