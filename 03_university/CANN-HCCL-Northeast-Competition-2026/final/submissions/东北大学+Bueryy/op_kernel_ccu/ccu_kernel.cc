/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_kernel.h"

namespace ops_hccl {
namespace {
constexpr uint16_t ADDRESS_SLOT = 1;
constexpr uint16_t TOKEN_SLOT = 2;
constexpr uint16_t NOTIFY_INDEX = 0;
constexpr uint16_t READY_MASK = (1U << ADDRESS_SLOT) | (1U << TOKEN_SLOT);
constexpr uint16_t FINISHED_MASK = 1U << 3;
constexpr uint16_t READ_MODE_SLOT = 0;
constexpr uint16_t READ_READY_MASK = 1U << READ_MODE_SLOT;
// 写路径的缓存使用槽0和槽3；Read 不与它并发，使用模式通知时每轮重写自己的槽0。
constexpr uint16_t CACHED_OUTPUT_SLOT = 0;
constexpr uint16_t CACHED_TOKEN_SLOT = 3;
constexpr uint16_t CACHED_OUTPUT_MASK = 1U << 4;
constexpr uint16_t CACHED_TOKEN_MASK = 1U << 5;
constexpr uint16_t REUSED_FINISHED_MASK = 1U << 6;

// 小消息及 8+4 大机 root 的 Clos 共用定向写，仍注册在原 Read mission 中。
CcuResult BuildDirectedWrite(const ScatterKernelArg &arg, ccu::Variable root,
    ccu::Variable inputAddress, ccu::Variable inputToken, ccu::Variable stride,
    ccu::Variable bytes, ccu::LocalAddr output, ccu::LocalAddr copySource,
    ccu::Variable outputAddress, ccu::Variable outputToken, ccu::Variable copyAddress,
    ccu::Variable copyBytes, ccu::Event copied,
    const ccu::Variable *linkedOffset = nullptr)
{
    const bool linked = linkedOffset != nullptr;
    // GetChannelForCcu 按远端 rank 递增建组；显式校验后仅使用目标 CANN 支持的加法。
    for (uint32_t i = 1; i < arg.channelCount; ++i) {
        if (arg.ranks[i] <= arg.ranks[i - 1]) return CCU_E_PARA;
    }
    ccu::Event written;
    CCU_IF(root == arg.rank) {
        if (linked && arg.selfCopy) {
            CCU_IF(copyBytes != 0) {
                // 免 READY 的纯接收角色不准备本地复制地址。
                output.addr = outputAddress;
                output.token = outputToken;
                copySource.addr = copyAddress;
                copySource.token = inputToken;
                CCU_CHK_RET(ccu::LocalCopy(output, copySource, copyBytes, copied, 1));
            }
        }
        ccu::Variable position;
        position = inputAddress;
        for (uint32_t i = 0; i < arg.channelCount; ++i) {
            if (i != 0) {
                for (uint32_t rank = arg.ranks[i - 1]; rank < arg.ranks[i]; ++rank) {
                    position += stride;
                }
            }
            ccu::LocalAddr source;
            source.addr = position;
            source.token = inputToken;
            ccu::RemoteAddr target;
            if (linked) {
                target.addr = arg.roleLinkTargets[i].address;
                target.addr += *linkedOffset;
                target.token = arg.roleLinkTargets[i].token;
            } else {
                CCU_CHK_RET(ccu::NotifyWait(arg.channels[i], NOTIFY_INDEX, READY_MASK));
                target.addr = ccu::GetResByChannel<ccu::Variable>(arg.channels[i], ADDRESS_SLOT);
                target.token = ccu::GetResByChannel<ccu::Variable>(arg.channels[i], TOKEN_SLOT);
            }
            CCU_CHK_RET(ccu::Write(arg.channels[i], target, source, bytes,
                written, static_cast<uint16_t>(1U << i)));
        }
        if (arg.selfCopy && !linked) {
            CCU_IF(copyBytes != 0) {
                // 先发起网络搬运，再发本地拷贝；完成等待仍覆盖两者。
                CCU_CHK_RET(ccu::LocalCopy(output, copySource, copyBytes, copied, 1));
            }
        }
        if (linked || (arg.optimization == WriteOptimization::ChannelReady && arg.rank < 8 && arg.channelCount == 4)) {
            // 8+4 大机 root 的定向写：全部发起后逐笔通知，前几笔收尾与后续搬运重叠。
            for (uint32_t i = 0; i < arg.channelCount; ++i) {
                CCU_CHK_RET(ccu::EventWait(written, static_cast<uint16_t>(1U << i)));
                CCU_CHK_RET(ccu::NotifyRecord(arg.channels[i], NOTIFY_INDEX, FINISHED_MASK));
            }
        } else {
            CCU_CHK_RET(ccu::EventWait(written, static_cast<uint16_t>((1U << arg.channelCount) - 1)));
            for (uint32_t i = 0; i < arg.channelCount; ++i) {
                CCU_CHK_RET(ccu::NotifyRecord(arg.channels[i], NOTIFY_INDEX, FINISHED_MASK));
            }
        }
        if (arg.selfCopy) {
            CCU_IF(copyBytes != 0) {
                CCU_CHK_RET(ccu::EventWait(copied, 1));
            }
        }
    }
    CCU_ELSE {
        for (uint32_t i = 0; i < arg.channelCount; ++i) {
            CCU_IF(root == arg.ranks[i]) {
                if (!linked) {
                    CCU_CHK_RET(ccu::WriteVariableWithNotify(arg.channels[i], outputAddress,
                        ADDRESS_SLOT, NOTIFY_INDEX, 1U << ADDRESS_SLOT));
                    CCU_CHK_RET(ccu::WriteVariableWithNotify(arg.channels[i], outputToken,
                        TOKEN_SLOT, NOTIFY_INDEX, 1U << TOKEN_SLOT));
                }
                // 本轮输出就绪通知在 root 搬运完成后到达，并在返回前消费。
                CCU_CHK_RET(ccu::NotifyWait(arg.channels[i], NOTIFY_INDEX, FINISHED_MASK));
            }
        }
    }
    return CCU_SUCCESS;
}

// 正式 4×1 两个 mission 按发送、接收分工；大小消息都没有通用模式判断。
CcuResult BuildRoleSendKernel(const ScatterKernelArg &arg)
{
    if (arg.channelCount != 3 || !arg.selfCopy) return CCU_E_PARA;
    ccu::Variable inputToken, bytes, copyAddress, outputAddress, outputToken, copyBytes;
    std::array<ccu::Variable, 3> sources;
    uint32_t index = 0;
    CCU_CHK_RET(ccu::LoadArg(inputToken, index++));
    for (auto &source : sources) CCU_CHK_RET(ccu::LoadArg(source, index++));
    CCU_CHK_RET(ccu::LoadArg(bytes, index++));
    CCU_CHK_RET(ccu::LoadArg(copyAddress, index++));
    CCU_CHK_RET(ccu::LoadArg(outputAddress, index++));
    CCU_CHK_RET(ccu::LoadArg(outputToken, index++));
    CCU_CHK_RET(ccu::LoadArg(copyBytes, index++));
    if (index != scatter_plan::ROLE_SEND_ARGS) return CCU_E_PARA;
    ccu::Event written, copied;
    // 延续 V14 小消息的提前自拷贝，大小消息均等待其完成后返回。
    CCU_IF(copyBytes != 0) {
        ccu::LocalAddr target, source;
        target.addr = outputAddress;
        target.token = outputToken;
        source.addr = copyAddress;
        source.token = inputToken;
        CCU_CHK_RET(ccu::LocalCopy(target, source, copyBytes, copied, 1));
    }
    for (uint32_t i = 0; i < 3; ++i) {
        ccu::LocalAddr source;
        source.addr = sources[i];
        source.token = inputToken;
        ccu::RemoteAddr target;
        CCU_CHK_RET(ccu::NotifyWait(arg.channels[i], NOTIFY_INDEX, READY_MASK));
        target.addr = ccu::GetResByChannel<ccu::Variable>(arg.channels[i], ADDRESS_SLOT);
        target.token = ccu::GetResByChannel<ccu::Variable>(arg.channels[i], TOKEN_SLOT);
        CCU_CHK_RET(ccu::Write(arg.channels[i], target, source, bytes,
            written, static_cast<uint16_t>(1U << i)));
    }
    for (uint32_t i = 0; i < 3; ++i) {
        CCU_CHK_RET(ccu::EventWait(written, static_cast<uint16_t>(1U << i)));
        CCU_CHK_RET(ccu::NotifyRecord(arg.channels[i], NOTIFY_INDEX, FINISHED_MASK));
    }
    CCU_IF(copyBytes != 0) {
        CCU_CHK_RET(ccu::EventWait(copied, 1));
    }
    return CCU_SUCCESS;
}

CcuResult BuildRoleReceiveKernel(const ScatterKernelArg &arg)
{
    if (arg.channelCount != 3 || !arg.selfCopy) return CCU_E_PARA;
    ccu::Variable role, outputAddress, outputToken, publishAddress;
    uint32_t index = 0;
    CCU_CHK_RET(ccu::LoadArg(role, index++));
    CCU_CHK_RET(ccu::LoadArg(outputAddress, index++));
    CCU_CHK_RET(ccu::LoadArg(outputToken, index++));
    CCU_CHK_RET(ccu::LoadArg(publishAddress, index++));
    if (index != scatter_plan::ROLE_RECEIVE_ARGS) return CCU_E_PARA;
    for (uint32_t i = 0; i < 3; ++i) {
        CCU_IF(role == i) {
            CCU_IF(publishAddress != 0) {
                CCU_CHK_RET(ccu::WriteVariableWithNotify(arg.channels[i], outputAddress,
                    ADDRESS_SLOT, NOTIFY_INDEX, 1U << ADDRESS_SLOT));
                CCU_CHK_RET(ccu::WriteVariableWithNotify(arg.channels[i], outputToken,
                    TOKEN_SLOT, NOTIFY_INDEX, 1U << TOKEN_SLOT));
            }
            CCU_ELSE {
                // 地址保留在同一发送者的槽内；当前 token 写同时置两个 READY 位。
                CCU_CHK_RET(ccu::WriteVariableWithNotify(arg.channels[i], outputToken,
                    TOKEN_SLOT, NOTIFY_INDEX, READY_MASK));
            }
            CCU_CHK_RET(ccu::NotifyWait(arg.channels[i], NOTIFY_INDEX, FINISHED_MASK));
        }
    }
    return CCU_SUCCESS;
}

// 4×1 两个 mission 分别承担统一 Write 回退和固定初始角色的五参数 Read。
CcuResult BuildRoleLinkSendKernel(const ScatterKernelArg &arg)
{
    if (arg.channelCount != 3 || !arg.selfCopy) return CCU_E_PARA;
    std::array<ccu::Variable, scatter_plan::ROLE_UNIFIED_WRITE_ARGS> values;
    for (uint32_t i = 0; i < values.size(); ++i) CCU_CHK_RET(ccu::LoadArg(values[i], i));
    CCU_IF(values[0] == 3) {
        ccu::Event written, copied;
        CCU_IF(values[9] != 0) {
            ccu::LocalAddr source, target;
            source.addr = values[6];
            source.token = values[1];
            target.addr = values[7];
            target.token = values[8];
            CCU_CHK_RET(ccu::LocalCopy(target, source, values[9], copied, 1));
        }
        for (uint32_t i = 0; i < 3; ++i) {
            ccu::LocalAddr source;
            source.addr = values[2 + i];
            source.token = values[1];
            ccu::RemoteAddr target;
            CCU_IF(values[10] != 0) {
                target.addr = arg.roleLinkTargets[i].address;
                target.token = arg.roleLinkTargets[i].token;
            }
            CCU_ELSE {
                CCU_CHK_RET(ccu::NotifyWait(arg.channels[i], NOTIFY_INDEX, READY_MASK));
                target.addr = ccu::GetResByChannel<ccu::Variable>(arg.channels[i], ADDRESS_SLOT);
                target.token = ccu::GetResByChannel<ccu::Variable>(arg.channels[i], TOKEN_SLOT);
            }
            CCU_CHK_RET(ccu::Write(arg.channels[i], target, source, values[5], written,
                static_cast<uint16_t>(1U << i)));
        }
        for (uint32_t i = 0; i < 3; ++i) {
            CCU_CHK_RET(ccu::EventWait(written, static_cast<uint16_t>(1U << i)));
            CCU_CHK_RET(ccu::NotifyRecord(arg.channels[i], NOTIFY_INDEX, FINISHED_MASK));
        }
        CCU_IF(values[9] != 0) { CCU_CHK_RET(ccu::EventWait(copied, 1)); }
    }
    CCU_ELSE {
        for (uint32_t i = 0; i < 3; ++i) {
            CCU_IF(values[0] == i) {
                CCU_IF(values[10] == 0) {
                    CCU_IF(values[3] != 0) {
                        CCU_CHK_RET(ccu::WriteVariableWithNotify(arg.channels[i], values[1],
                            ADDRESS_SLOT, NOTIFY_INDEX, 1U << ADDRESS_SLOT));
                        CCU_CHK_RET(ccu::WriteVariableWithNotify(arg.channels[i], values[2],
                            TOKEN_SLOT, NOTIFY_INDEX, 1U << TOKEN_SLOT));
                    }
                    CCU_ELSE {
                        CCU_CHK_RET(ccu::WriteVariableWithNotify(arg.channels[i], values[2],
                            TOKEN_SLOT, NOTIFY_INDEX, READY_MASK));
                    }
                }
                CCU_CHK_RET(ccu::NotifyWait(arg.channels[i], NOTIFY_INDEX, FINISHED_MASK));
            }
        }
    }
    return CCU_SUCCESS;
}

CcuResult BuildRoleLinkReceiveKernel(const ScatterKernelArg &arg)
{
    if (arg.channelCount != 3 || !arg.selfCopy) return CCU_E_PARA;
    // 首次是大消息时无输入描述，仍注册一个确定角色的备用序列；host 不会选择它。
    const uint32_t fixedRoot = arg.registeredRoot < 4 ? arg.registeredRoot : 0;
    // 仅绑定部署 die，不生成任何运行期 channel 操作。
    auto dieAnchor = ccu::GetResByChannel<ccu::Variable>(arg.channels[0], ADDRESS_SLOT);
    (void)dieAnchor;
    ccu::Variable sourceAddress, outputAddress, bytes;
    ccu::LocalAddr output;
    ccu::Event completed;
    CCU_CHK_RET(ccu::LoadArg(sourceAddress, 0));
    CCU_CHK_RET(ccu::LoadArg(outputAddress, 1));
    if (arg.rank == fixedRoot) {
        ccu::LocalAddr source;
        CCU_CHK_RET(ccu::LoadArg(source.token, 2));
        CCU_CHK_RET(ccu::LoadArg(output.token, 3));
        CCU_CHK_RET(ccu::LoadArg(bytes, 4));
        source.addr = sourceAddress;
        output.addr = outputAddress;
        CCU_IF(bytes != 0) {
            CCU_CHK_RET(ccu::LocalCopy(output, source, bytes, completed, 1));
            CCU_CHK_RET(ccu::EventWait(completed, 1));
        }
    } else {
        ccu::RemoteAddr source;
        CCU_CHK_RET(ccu::LoadArg(source.token, 2));
        CCU_CHK_RET(ccu::LoadArg(output.token, 3));
        CCU_CHK_RET(ccu::LoadArg(bytes, 4));
        source.addr = sourceAddress;
        output.addr = outputAddress;
        uint32_t rootIndex = 0;
        while (rootIndex < 3 && arg.ranks[rootIndex] != fixedRoot) ++rootIndex;
        if (rootIndex == 3) return CCU_E_PARA;
        CCU_CHK_RET(ccu::Read(arg.channels[rootIndex], output, source, bytes, completed, 1));
        CCU_CHK_RET(ccu::EventWait(completed, 1));
    }
    return CCU_SUCCESS;
}

// 大机 root 的前五个机内对端主动 Read，最后两个由 root 定向 Write。
CcuResult BuildMixedMeshKernel(const ScatterKernelArg &arg, ccu::Variable root,
    ccu::Variable inputAddress, ccu::Variable inputToken, ccu::Variable sourceOffset,
    ccu::Variable secondWriteSource, ccu::Variable bytes,
    ccu::LocalAddr output, ccu::LocalAddr copySource, ccu::Variable outputAddress,
    ccu::Variable outputToken, ccu::Variable copyBytes, ccu::Event copied, bool linked = false)
{
    using namespace scatter_plan;
    if (arg.channelCount != MIXED_MESH_READS + MIXED_MESH_WRITES) return CCU_E_PARA;
    const auto writePeers = MixedMeshWritePeers(arg.rank);
    for (uint32_t i = 0; i < arg.channelCount; ++i) {
        if (arg.ranks[i] >= 8 || arg.ranks[i] == arg.rank ||
            (i != 0 && arg.ranks[i] <= arg.ranks[i - 1])) return CCU_E_PARA;
    }
    for (uint32_t i = 0; i < MIXED_MESH_WRITES; ++i) {
        if (arg.ranks[MIXED_MESH_READS + i] != writePeers[i]) return CCU_E_PARA;
    }
    ccu::Event written, readDone;
    auto writePeersNow = [&]() -> CcuResult {
        for (uint32_t j = 0; j < MIXED_MESH_WRITES; ++j) {
            const uint32_t i = MIXED_MESH_READS + j;
            ccu::LocalAddr source;
            source.addr = j == 0 ? sourceOffset : secondWriteSource;
            source.token = inputToken;
            ccu::RemoteAddr target;
            if (linked) {
                // host 仅在整条接收量不超过单次上限时启用，当前切片偏移为零。
                target.addr = arg.roleLinkTargets[i].address;
                target.token = arg.roleLinkTargets[i].token;
            } else {
                CCU_CHK_RET(ccu::NotifyWait(arg.channels[i], NOTIFY_INDEX, READY_MASK));
                target.addr = ccu::GetResByChannel<ccu::Variable>(arg.channels[i], ADDRESS_SLOT);
                target.token = ccu::GetResByChannel<ccu::Variable>(arg.channels[i], TOKEN_SLOT);
            }
            CCU_CHK_RET(ccu::Write(arg.channels[i], target, source, bytes, written,
                static_cast<uint16_t>(1U << j)));
        }
        return CCU_SUCCESS;
    };
    CCU_IF(root == arg.rank) {
        CCU_IF(copyBytes != 0) {
            // 保持原 Mesh 自拷贝位置，与所有网络搬运重叠。
            CCU_CHK_RET(ccu::LocalCopy(output, copySource, copyBytes, copied, 1));
        }
        // 预交换两笔 Write 先进入传输，再发布五路 Read 的动态输入元数据。
        if (linked) CCU_CHK_RET(writePeersNow());
        for (uint32_t i = 0; i < MIXED_MESH_READS; ++i) {
            CCU_CHK_RET(ccu::WriteVariableWithNotify(arg.channels[i], inputAddress,
                ADDRESS_SLOT, NOTIFY_INDEX, 1U << ADDRESS_SLOT));
            CCU_CHK_RET(ccu::WriteVariableWithNotify(arg.channels[i], inputToken,
                TOKEN_SLOT, NOTIFY_INDEX, 1U << TOKEN_SLOT));
        }
        // 回退分支保留 V19 的发布、READY 等待及发起顺序。
        if (!linked) CCU_CHK_RET(writePeersNow());
        for (uint32_t j = 0; j < MIXED_MESH_WRITES; ++j) {
            CCU_CHK_RET(ccu::EventWait(written, static_cast<uint16_t>(1U << j)));
            CCU_CHK_RET(ccu::NotifyRecord(arg.channels[MIXED_MESH_READS + j], NOTIFY_INDEX, FINISHED_MASK));
        }
        for (uint32_t i = 0; i < MIXED_MESH_READS; ++i) {
            CCU_CHK_RET(ccu::NotifyWait(arg.channels[i], NOTIFY_INDEX, FINISHED_MASK));
        }
        CCU_IF(copyBytes != 0) {
            CCU_CHK_RET(ccu::EventWait(copied, 1));
        }
    }
    CCU_ELSE {
        for (uint32_t i = 0; i < arg.channelCount; ++i) {
            CCU_IF(root == arg.ranks[i]) {
                // rank 和静态 channel 对端都在生成期已知，不使用变量位运算。
                if (IsMixedMeshWritePeer(arg.rank, arg.ranks[i])) {
                    if (!linked) {
                        CCU_CHK_RET(ccu::WriteVariableWithNotify(arg.channels[i], outputAddress,
                            ADDRESS_SLOT, NOTIFY_INDEX, 1U << ADDRESS_SLOT));
                        CCU_CHK_RET(ccu::WriteVariableWithNotify(arg.channels[i], outputToken,
                            TOKEN_SLOT, NOTIFY_INDEX, 1U << TOKEN_SLOT));
                    }
                    CCU_CHK_RET(ccu::NotifyWait(arg.channels[i], NOTIFY_INDEX, FINISHED_MASK));
                } else {
                    CCU_CHK_RET(ccu::NotifyWait(arg.channels[i], NOTIFY_INDEX, READY_MASK));
                    ccu::RemoteAddr source;
                    source.addr = ccu::GetResByChannel<ccu::Variable>(arg.channels[i], ADDRESS_SLOT);
                    source.addr += sourceOffset;
                    source.token = ccu::GetResByChannel<ccu::Variable>(arg.channels[i], TOKEN_SLOT);
                    CCU_CHK_RET(ccu::Read(arg.channels[i], output, source, bytes, readDone, 1));
                    CCU_CHK_RET(ccu::EventWait(readDone, 1));
                    CCU_CHK_RET(ccu::NotifyRecord(arg.channels[i], NOTIFY_INDEX, FINISHED_MASK));
                }
            }
        }
    }
    return CCU_SUCCESS;
}

// 小消息 Clos 使用预交换输入；4×3 取消完成回报，2×8 和 8+4 保留回报。
CcuResult BuildLinkedClosRead(const ScatterKernelArg &arg, ccu::Variable root,
    ccu::Variable sourceAddress, ccu::Variable sourceToken, ccu::Variable outputAddress,
    ccu::Variable outputToken, ccu::Variable bytes)
{
    if (arg.optimization == WriteOptimization::Compact && arg.registeredRoot != INVALID_VALUE_RANKID) {
        if (arg.rank == arg.registeredRoot) return CCU_SUCCESS;
        for (uint32_t i = 0; i < arg.channelCount; ++i) {
            if (arg.ranks[i] != arg.registeredRoot) continue;
            ccu::LocalAddr output;
            ccu::RemoteAddr source;
            output.addr = outputAddress;
            output.token = outputToken;
            source.addr = sourceAddress;
            source.token = sourceToken;
            ccu::Event completed;
            CCU_CHK_RET(ccu::Read(arg.channels[i], output, source, bytes, completed, 1));
            CCU_CHK_RET(ccu::EventWait(completed, 1));
        }
        return CCU_SUCCESS;
    }
    CCU_IF(root == arg.rank) {
        for (uint32_t i = 0; i < arg.channelCount; ++i) {
            CCU_CHK_RET(ccu::NotifyWait(arg.channels[i], NOTIFY_INDEX, FINISHED_MASK));
        }
    }
    CCU_ELSE {
        ccu::LocalAddr output;
        output.addr = outputAddress;
        output.token = outputToken;
        ccu::RemoteAddr source;
        source.addr = sourceAddress;
        source.token = sourceToken;
        ccu::Event completed;
        for (uint32_t i = 0; i < arg.channelCount; ++i) {
            CCU_IF(root == arg.ranks[i]) {
                CCU_CHK_RET(ccu::Read(arg.channels[i], output, source, bytes, completed, 1));
                CCU_CHK_RET(ccu::EventWait(completed, 1));
                // 4×3 小消息由用户授权取消回报，其他拓扑保留原确认。
                if (arg.optimization != WriteOptimization::Compact) {
                    CCU_CHK_RET(ccu::NotifyRecord(arg.channels[i], NOTIFY_INDEX, FINISHED_MASK));
                }
            }
        }
    }
    return CCU_SUCCESS;
}

// 预交换 Read 只等本地输出完成；4×3 中继实验另外等待两笔既有 Write 尾段。
CcuResult BuildUnconfirmedRead(const ScatterKernelArg &arg, ccu::Variable sourceAddress,
    ccu::Variable sourceToken, ccu::Variable outputAddress, ccu::Variable outputToken,
    ccu::Variable bytes, ccu::Variable copyBytes, bool triTail)
{
    if (arg.registeredRoot == INVALID_VALUE_RANKID) return CCU_SUCCESS;
    ccu::Event completed;
    ccu::LocalAddr output;
    output.addr = outputAddress;
    output.token = outputToken;
    if (arg.rank == arg.registeredRoot) {
        ccu::LocalAddr source;
        source.addr = sourceAddress;
        source.token = sourceToken;
        CCU_IF(copyBytes != 0) {
            CCU_CHK_RET(ccu::LocalCopy(output, source, copyBytes, completed, 1));
            CCU_CHK_RET(ccu::EventWait(completed, 1));
        }
        return CCU_SUCCESS;
    }
    uint32_t rootIndex = 0;
    while (rootIndex < arg.channelCount && arg.ranks[rootIndex] != arg.registeredRoot) ++rootIndex;
    // 同机之外的未选用 die 不含 root，不生成数据序列。
    if (rootIndex == arg.channelCount) return CCU_SUCCESS;
    ccu::RemoteAddr source;
    source.addr = sourceAddress;
    source.token = sourceToken;
    CCU_CHK_RET(ccu::Read(arg.channels[rootIndex], output, source, bytes, completed, 1));
    if (triTail) {
        for (uint32_t i = 0; i < arg.channelCount; ++i) {
            if (arg.ranks[i] / 3 == arg.registeredRoot / 3 && i != rootIndex) {
                // 延续旧 Clos Write 的接收方通知；不是 Read 完成回报。
                CCU_CHK_RET(ccu::NotifyRecord(arg.channels[i], NOTIFY_INDEX, FINISHED_MASK));
            }
        }
        for (uint32_t i = 0; i < arg.channelCount; ++i) {
            if (arg.ranks[i] / 3 == arg.registeredRoot / 3 && i != rootIndex) {
                CCU_CHK_RET(ccu::NotifyWait(arg.channels[i], NOTIFY_INDEX, FINISHED_MASK));
            }
        }
    }
    CCU_CHK_RET(ccu::EventWait(completed, 1));
    return CCU_SUCCESS;
}

CcuResult BuildReadKernel(CcuKernelArg opaque)
{
    auto *arg = static_cast<ScatterKernelArg *>(opaque);
    if (arg == nullptr || arg->channelCount == 0 || arg->channelCount >= 16) {
        return CCU_E_PARA;
    }
    if (arg->roleWriteOnly) {
        return arg->roleLinkBytes != 0 ? BuildRoleLinkReceiveKernel(*arg) : BuildRoleReceiveKernel(*arg);
    }
    // 大机 Mesh 承担 root 自拷贝；大消息五读两写，小消息保留原刷新 Read。
    if (arg->meshReadWriteOnly && (arg->directedWrite || arg->rank >= 8 ||
        arg->channelCount != 7 || !arg->selfCopy || arg->optimization != WriteOptimization::ChannelReady)) {
        return CCU_E_PARA;
    }
    ccu::Variable root, mode, inputAddress, inputToken, outputAddress, outputToken;
    ccu::Variable sourceOffset, bytes, copyAddress, copyBytes, registeredAddress;
    // 全部 LoadArg 放在条件外；保持 11 项连续且各占独立 Variable。
    CCU_CHK_RET(ccu::LoadArg(root, 0));
    CCU_CHK_RET(ccu::LoadArg(mode, 1));
    CCU_CHK_RET(ccu::LoadArg(inputAddress, 2));
    CCU_CHK_RET(ccu::LoadArg(inputToken, 3));
    CCU_CHK_RET(ccu::LoadArg(outputAddress, 4));
    CCU_CHK_RET(ccu::LoadArg(outputToken, 5));
    CCU_CHK_RET(ccu::LoadArg(sourceOffset, 6));
    CCU_CHK_RET(ccu::LoadArg(bytes, 7));
    CCU_CHK_RET(ccu::LoadArg(copyAddress, 8));
    CCU_CHK_RET(ccu::LoadArg(copyBytes, 9));
    CCU_CHK_RET(ccu::LoadArg(registeredAddress, 10));
    ccu::LocalAddr output, copySource;
    ccu::Event completed;
    constexpr uint16_t COMPLETE_MASK = 1;
    const bool refreshReadSupported = arg->optimization == WriteOptimization::ChannelReady ||
        arg->optimization == WriteOptimization::FusedMetadata;

    auto publishLegacyMetadata = [&]() -> CcuResult {
        // 小消息沿用建链地址恢复和动态刷新协议，每轮仍发布当前 token。
        CCU_IF(mode == static_cast<uint64_t>(ReadMetadataMode::Refresh)) {
            for (uint32_t i = 0; i < arg->channelCount; ++i) {
                CCU_CHK_RET(ccu::WriteVariableWithNotify(arg->channels[i], inputAddress,
                    ADDRESS_SLOT, NOTIFY_INDEX, 1U << ADDRESS_SLOT));
            }
        }
        for (uint32_t i = 0; i < arg->channelCount; ++i) {
            CCU_CHK_RET(ccu::WriteVariableWithNotify(arg->channels[i], inputToken,
                TOKEN_SLOT, NOTIFY_INDEX, 1U << TOKEN_SLOT));
        }
        for (uint32_t i = 0; i < arg->channelCount; ++i) {
            CCU_CHK_RET(ccu::WriteVariableWithNotify(arg->channels[i], mode,
                READ_MODE_SLOT, NOTIFY_INDEX, READ_READY_MASK));
        }
        return CCU_SUCCESS;
    };

    auto publishRefreshMetadata = [&]() -> CcuResult {
        for (uint32_t i = 0; i < arg->channelCount; ++i) {
            CCU_CHK_RET(ccu::WriteVariableWithNotify(arg->channels[i], inputAddress,
                ADDRESS_SLOT, NOTIFY_INDEX, 1U << ADDRESS_SLOT));
            CCU_CHK_RET(ccu::WriteVariableWithNotify(arg->channels[i], inputToken,
                TOKEN_SLOT, NOTIFY_INDEX, 1U << TOKEN_SLOT));
        }
        return CCU_SUCCESS;
    };

    auto buildRead = [&](bool refreshOnly = false) -> CcuResult {
        CCU_IF(root == arg->rank) {
            if (arg->selfCopy) {
                CCU_IF(copyBytes != 0) {
                    // root 自身分片直接落到 recvBuf，与对端 Read 重叠；别名时 host 已置零。
                    CCU_CHK_RET(ccu::LocalCopy(output, copySource, copyBytes, completed, COMPLETE_MASK));
                }
            }
            if (refreshOnly) {
                CCU_CHK_RET(publishRefreshMetadata());
            } else if (refreshReadSupported) {
                CCU_IF(mode == static_cast<uint64_t>(ReadMetadataMode::LargeRefresh)) {
                    // 公共参数选择刷新协议；大小消息均按通道连续发布地址和当前 token。
                    CCU_CHK_RET(publishRefreshMetadata());
                }
                CCU_ELSE {
                    CCU_CHK_RET(publishLegacyMetadata());
                }
            } else {
                CCU_CHK_RET(publishLegacyMetadata());
            }
            for (uint32_t i = 0; i < arg->channelCount; ++i) {
                CCU_CHK_RET(ccu::NotifyWait(arg->channels[i], NOTIFY_INDEX, FINISHED_MASK));
            }
            if (arg->selfCopy) {
                CCU_IF(copyBytes != 0) {
                    CCU_CHK_RET(ccu::EventWait(completed, COMPLETE_MASK));
                }
            }
        }
        CCU_ELSE {
            for (uint32_t i = 0; i < arg->channelCount; ++i) {
                auto remoteMode = ccu::GetResByChannel<ccu::Variable>(arg->channels[i], READ_MODE_SLOT);
                auto remoteAddress = ccu::GetResByChannel<ccu::Variable>(arg->channels[i], ADDRESS_SLOT);
                auto remoteToken = ccu::GetResByChannel<ccu::Variable>(arg->channels[i], TOKEN_SLOT);
                auto waitLegacyMetadata = [&]() -> CcuResult {
                    CCU_CHK_RET(ccu::NotifyWait(arg->channels[i], NOTIFY_INDEX, READ_READY_MASK));
                    CCU_IF(remoteMode == static_cast<uint64_t>(ReadMetadataMode::RegisteredAddress)) {
                        CCU_CHK_RET(ccu::NotifyWait(arg->channels[i], NOTIFY_INDEX, 1U << TOKEN_SLOT));
                        remoteAddress = registeredAddress;
                    }
                    CCU_ELSE {
                        CCU_CHK_RET(ccu::NotifyWait(arg->channels[i], NOTIFY_INDEX, READY_MASK));
                    }
                    return CCU_SUCCESS;
                };
                CCU_IF(root == arg->ranks[i]) {
                    if (refreshOnly) {
                        CCU_CHK_RET(ccu::NotifyWait(arg->channels[i], NOTIFY_INDEX, READY_MASK));
                    } else if (refreshReadSupported) {
                        CCU_IF(mode == static_cast<uint64_t>(ReadMetadataMode::LargeRefresh)) {
                            CCU_CHK_RET(ccu::NotifyWait(arg->channels[i], NOTIFY_INDEX, READY_MASK));
                        }
                        CCU_ELSE {
                            CCU_CHK_RET(waitLegacyMetadata());
                        }
                    } else {
                        CCU_CHK_RET(waitLegacyMetadata());
                    }
                    ccu::RemoteAddr source;
                    source.addr = remoteAddress;
                    source.addr += sourceOffset;
                    source.token = remoteToken;
                    CCU_CHK_RET(ccu::Read(arg->channels[i], output, source, bytes, completed, COMPLETE_MASK));
                    CCU_CHK_RET(ccu::EventWait(completed, COMPLETE_MASK));
                    // Read 完成后才释放 root 的输入；本轮 READY 已消费，下一轮可安全复用通知位。
                    CCU_CHK_RET(ccu::NotifyRecord(arg->channels[i], NOTIFY_INDEX, FINISHED_MASK));
                }
            }
        }
        return CCU_SUCCESS;
    };
    auto dispatchOriginal = [&]() -> CcuResult {
        // 完整 Read/混合路径仍准备其局部地址；免 READY 定向接收不经过这里。
        output.addr = outputAddress;
        output.token = outputToken;
        copySource.addr = copyAddress;
        copySource.token = inputToken;
        if (arg->meshReadWriteOnly) {
            CCU_IF(mode == scatter_plan::LINKED_MIXED_MESH_MODE) {
                CCU_CHK_RET(BuildMixedMeshKernel(*arg, root, inputAddress, inputToken,
                    sourceOffset, registeredAddress, bytes, output, copySource,
                    outputAddress, outputToken, copyBytes, completed, true));
            }
            CCU_ELSE {
                CCU_IF(mode == static_cast<uint64_t>(ReadMetadataMode::MixedMesh)) {
                    CCU_CHK_RET(BuildMixedMeshKernel(*arg, root, inputAddress, inputToken,
                        sourceOffset, registeredAddress, bytes, output, copySource,
                        outputAddress, outputToken, copyBytes, completed));
                }
                CCU_ELSE {
                    CCU_CHK_RET(buildRead(true));
                }
            }
        } else if (arg->directedWrite) {
            CCU_IF(mode == static_cast<uint64_t>(ReadMetadataMode::DirectedWrite)) {
                CCU_CHK_RET(BuildDirectedWrite(*arg, root, inputAddress, inputToken, sourceOffset,
                    bytes, output, copySource, outputAddress, outputToken, copyAddress, copyBytes, completed));
            }
            CCU_ELSE {
                CCU_CHK_RET(buildRead());
            }
        } else {
            CCU_CHK_RET(buildRead(arg->meshReadWriteOnly != 0));
        }
        return CCU_SUCCESS;
    };
    auto dispatchRead = [&]() -> CcuResult {
        if (arg->roleLinkBytes != 0) {
            CCU_IF(mode == scatter_plan::LINK_WRITE_MODE) {
                CCU_CHK_RET(BuildDirectedWrite(*arg, root, inputAddress, inputToken, sourceOffset,
                    bytes, output, copySource, outputAddress, outputToken, copyAddress, copyBytes, completed,
                    &registeredAddress));
            }
            CCU_ELSE {
                CCU_CHK_RET(dispatchOriginal());
            }
        } else {
            CCU_CHK_RET(dispatchOriginal());
        }
        return CCU_SUCCESS;
    };
    const bool linkedClos = SCATTER_CANDIDATE == 0 &&
        ((!arg->selfCopy &&
            ((arg->optimization == WriteOptimization::FusedMetadata && arg->channelCount == 8) ||
             (arg->optimization == WriteOptimization::Compact && arg->channelCount == 9))) ||
         (arg->optimization == WriteOptimization::ChannelReady && arg->directedWrite &&
            (arg->channelCount == 4 || arg->channelCount == 8)));
    auto dispatchExisting = [&]() -> CcuResult {
        if (linkedClos) {
            CCU_IF(mode == scatter_plan::LINKED_CLOS_READ_MODE) {
                CCU_CHK_RET(BuildLinkedClosRead(*arg, root, inputAddress, inputToken,
                    outputAddress, outputToken, bytes));
            }
            CCU_ELSE { CCU_CHK_RET(dispatchRead()); }
        } else {
            CCU_CHK_RET(dispatchRead());
        }
        return CCU_SUCCESS;
    };
    const bool pullSupported = SCATTER_CANDIDATE == 0 &&
        ((arg->optimization == WriteOptimization::Compact && !arg->selfCopy) ||
         arg->optimization == WriteOptimization::ChannelReady);
    if (pullSupported) {
        CCU_IF(mode == scatter_plan::UNCONFIRMED_READ_MODE) {
            CCU_CHK_RET(BuildUnconfirmedRead(*arg, inputAddress, inputToken,
                outputAddress, outputToken, bytes, copyBytes, false));
        }
        CCU_ELSE {
            if (arg->optimization == WriteOptimization::Compact && !arg->selfCopy) {
                CCU_IF(mode == scatter_plan::TRI_PULL_READ_MODE) {
                    CCU_CHK_RET(BuildUnconfirmedRead(*arg, inputAddress, inputToken,
                        outputAddress, outputToken, bytes, copyBytes, true));
                }
                CCU_ELSE { CCU_CHK_RET(dispatchExisting()); }
            } else {
                CCU_CHK_RET(dispatchExisting());
            }
        }
    } else {
        CCU_CHK_RET(dispatchExisting());
    }
    return CCU_SUCCESS;
}

// 仅替代地址与 Token 的入口交换；阶段完成通知仍由各路径完整执行。
CcuResult BindLinkedTargets(const ScatterKernelArg &arg, ccu::Variable linkMode,
    std::vector<ccu::Variable> &addresses, std::vector<ccu::Variable> &tokens)
{
    if (arg.roleLinkBytes == 0) return CCU_SUCCESS;
    CCU_IF(linkMode == 1) {
        for (uint32_t i = 0; i < arg.channelCount; ++i) {
            addresses[i] = arg.roleLinkTargets[i].address;
            tokens[i] = arg.roleLinkTargets[i].token;
        }
    }
    CCU_IF(linkMode == 2) {
        for (uint32_t i = 0; i < arg.channelCount; ++i) {
            addresses[i] = arg.roleLinkTargets[i].scratchAddress;
            tokens[i] = arg.roleLinkTargets[i].scratchToken;
        }
    }
    return CCU_SUCCESS;
}

// 4×3 共用一个写 mission；空波、单波和九波使用各自完整的执行序列。
CcuResult BuildCompactWriteKernel(const ScatterKernelArg &arg)
{
    const bool mesh = scatter_plan::SameServer(scatter_plan::Topology::FourByThree,
                                                arg.rank, arg.ranks[0]);
    if ((mesh && (arg.channelCount != 2 || arg.slots != 9 || !arg.selfCopy)) ||
        (!mesh && (arg.channelCount != 9 || arg.slots != 1 || arg.selfCopy))) {
        return CCU_E_PARA;
    }
    ccu::Variable publishedAddress, publishedToken, copySourceAddress, copyTargetAddress, copyBytes;
    ccu::LocalAddr copySource, copyTarget;
    ccu::Variable activeCount, sourceStride, sourceServerGap, sharedToken, sharedOffset, linkMode;
    std::vector<ccu::Variable> rootServer(mesh ? 0 : 1);
    // Mesh 用阶段模式区分「阶段0 发布两个缓冲区」「阶段1 复用缓存的 Output」「单阶段刷新」。
    std::vector<ccu::Variable> mode(mesh ? 1 : 0);
    std::vector<ccu::Variable> finalFields(mesh ? 4 : 0);
    std::vector<ccu::Variable> cachedAddress(mesh ? 1 : 0), cachedToken(mesh ? 1 : 0);
    struct FirstTransfer {
        ccu::Variable address, token, offset, bytes;
    };
    std::vector<FirstTransfer> first(mesh ? 2 : 1);
    uint32_t argIndex = 0;
    // Mesh 原二十项、Clos 原十项末尾均追加建链模式，LoadArg 全在运行时条件之外。
    CCU_CHK_RET(ccu::LoadArg(publishedAddress, argIndex++));
    CCU_CHK_RET(ccu::LoadArg(publishedToken, argIndex++));
    if (mesh) {
        CCU_CHK_RET(ccu::LoadArg(mode[0], argIndex++));
        CCU_CHK_RET(ccu::LoadArg(cachedAddress[0], argIndex++));
        CCU_CHK_RET(ccu::LoadArg(cachedToken[0], argIndex++));
        CCU_CHK_RET(ccu::LoadArg(copySourceAddress, argIndex++));
        CCU_CHK_RET(ccu::LoadArg(copyTargetAddress, argIndex++));
        CCU_CHK_RET(ccu::LoadArg(copyBytes, argIndex++));
    }
    CCU_CHK_RET(ccu::LoadArg(activeCount, argIndex++));
    if (mesh) {
        CCU_CHK_RET(ccu::LoadArg(sharedToken, argIndex++));
        CCU_CHK_RET(ccu::LoadArg(sharedOffset, argIndex++));
    }
    for (auto &lane : first) {
        CCU_CHK_RET(ccu::LoadArg(lane.address, argIndex++));
        if (!mesh) {
            CCU_CHK_RET(ccu::LoadArg(lane.token, argIndex++));
            CCU_CHK_RET(ccu::LoadArg(lane.offset, argIndex++));
        }
        CCU_CHK_RET(ccu::LoadArg(lane.bytes, argIndex++));
    }
    CCU_CHK_RET(ccu::LoadArg(sourceStride, argIndex++));
    for (auto &field : finalFields) CCU_CHK_RET(ccu::LoadArg(field, argIndex++));
    if (!mesh) {
        CCU_CHK_RET(ccu::LoadArg(sourceServerGap, argIndex++));
        CCU_CHK_RET(ccu::LoadArg(rootServer[0], argIndex++));
    }
    if (argIndex != (mesh ? scatter_plan::TRI_MESH_WRITE_ARGS : scatter_plan::TRI_CLOS_WRITE_ARGS)) {
        return CCU_E_PARA;
    }
    CCU_CHK_RET(ccu::LoadArg(linkMode, argIndex++));
    std::vector<uint32_t> allPeers;
    std::vector<ccu::Variable> remoteAddress, remoteToken, remoteCachedAddress, remoteCachedToken;
    for (uint32_t i = 0; i < arg.channelCount; ++i) {
        allPeers.push_back(i);
        remoteAddress.push_back(ccu::GetResByChannel<ccu::Variable>(arg.channels[i], ADDRESS_SLOT));
        remoteToken.push_back(ccu::GetResByChannel<ccu::Variable>(arg.channels[i], TOKEN_SLOT));
        if (mesh) {
            remoteCachedAddress.push_back(
                ccu::GetResByChannel<ccu::Variable>(arg.channels[i], CACHED_OUTPUT_SLOT));
            remoteCachedToken.push_back(
                ccu::GetResByChannel<ccu::Variable>(arg.channels[i], CACHED_TOKEN_SLOT));
        }
    }
    if (mesh) {
        CCU_CHK_RET(BindLinkedTargets(arg, linkMode, remoteAddress, remoteToken));
    } else {
        // Clos 纯接收角色只消费完成通知，不准备任何远端 Write 目标。
        CCU_IF(activeCount != 0) {
            CCU_CHK_RET(BindLinkedTargets(arg, linkMode, remoteAddress, remoteToken));
        }
    }
    if (mesh && arg.roleLinkBytes != 0) {
        CCU_IF(linkMode != 0) {
            for (uint32_t i = 0; i < arg.channelCount; ++i) {
                remoteCachedAddress[i] = arg.roleLinkTargets[i].address;
                remoteCachedToken[i] = arg.roleLinkTargets[i].token;
            }
        }
    }
    auto publishPeers = [&](const std::vector<uint32_t> &peers) -> CcuResult {
        CCU_IF(linkMode == 0) {
            for (uint32_t i : peers) {
                CCU_CHK_RET(ccu::WriteVariableWithNotify(arg.channels[i], publishedAddress,
                    ADDRESS_SLOT, NOTIFY_INDEX, 1U << ADDRESS_SLOT));
                CCU_CHK_RET(ccu::WriteVariableWithNotify(arg.channels[i], publishedToken,
                    TOKEN_SLOT, NOTIFY_INDEX, 1U << TOKEN_SLOT));
            }
        }
        return CCU_SUCCESS;
    };
    // Mesh 阶段0 额外发布本 rank 的 Output；阶段1 直接用缓存值，不再发布也不再等 READY。
    auto publishMeshPeers = [&](const std::vector<uint32_t> &peers) -> CcuResult {
        CCU_IF(linkMode == 0) {
            CCU_IF(mode[0] != static_cast<uint64_t>(WriteMetadataMode::ReuseOutput)) {
                CCU_CHK_RET(publishPeers(peers));
                CCU_IF(mode[0] != static_cast<uint64_t>(WriteMetadataMode::Refresh)) {
                    for (uint32_t i : peers) {
                        CCU_CHK_RET(ccu::WriteVariableWithNotify(arg.channels[i], cachedAddress[0],
                            CACHED_OUTPUT_SLOT, NOTIFY_INDEX, CACHED_OUTPUT_MASK));
                        CCU_CHK_RET(ccu::WriteVariableWithNotify(arg.channels[i], cachedToken[0],
                            CACHED_TOKEN_SLOT, NOTIFY_INDEX, CACHED_TOKEN_MASK));
                    }
                }
            }
        }
        return CCU_SUCCESS;
    };
    if (mesh) {
        CCU_CHK_RET(publishMeshPeers(allPeers));
        copySource.addr = copySourceAddress;
        copySource.token = sharedToken;
        copyTarget.addr = copyTargetAddress;
        copyTarget.token = publishedToken;
    }
    struct WaveTransfer {
        ccu::LocalAddr local;
        ccu::Variable address, offset, bytes;
    };
    std::vector<std::vector<WaveTransfer>> waves(arg.slots);
    for (auto &wave : waves) {
        wave.resize(arg.channelCount);
    }
    std::vector<ccu::Variable> positions(mesh ? 0 : 9);
    const uint32_t serverBegin = arg.rank / 3 * 3;
    const uint32_t window = mesh ? 2 : 1;
    std::vector<ccu::Event> completed(window);
    ccu::RemoteAddr remote;
    auto waitReady = [&](const std::vector<uint32_t> &peers) -> CcuResult {
        CCU_IF(linkMode == 0) {
            if (mesh) {
                CCU_IF(mode[0] != static_cast<uint64_t>(WriteMetadataMode::ReuseOutput)) {
                    CCU_IF(mode[0] != static_cast<uint64_t>(WriteMetadataMode::Refresh)) {
                        for (uint32_t i : peers) {
                            CCU_CHK_RET(ccu::NotifyWait(arg.channels[i], NOTIFY_INDEX,
                                READY_MASK | CACHED_OUTPUT_MASK | CACHED_TOKEN_MASK));
                        }
                    }
                }
                CCU_IF(mode[0] == static_cast<uint64_t>(WriteMetadataMode::Refresh)) {
                    for (uint32_t i : peers) {
                        CCU_CHK_RET(ccu::NotifyWait(arg.channels[i], NOTIFY_INDEX, READY_MASK));
                    }
                }
            } else {
                for (uint32_t i : peers) {
                    CCU_CHK_RET(ccu::NotifyWait(arg.channels[i], NOTIFY_INDEX, READY_MASK));
                }
            }
        }
        return CCU_SUCCESS;
    };
    auto prepareCommon = [&](uint32_t count) {
        // 发布之后再计算，与远端元数据传输重叠；各 Variable 独立构造，避免复制句柄别名。
        if (mesh) {
            for (auto &lane : first) {
                lane.token = sharedToken;
                lane.offset = sharedOffset;
            }
            if (count == 9 && serverBegin != 0 && serverBegin < 9) {
                // 只有跨过中间本机的九波才使用跳距；单波和纯接收不增加地址运算。
                sourceServerGap = sourceStride;
                sourceServerGap += sourceStride;
                sourceServerGap += sourceStride;
            }
        }
        if (!mesh) {
            positions[0] = first[0].address;
            for (uint32_t i = 1; i < positions.size(); ++i) {
                positions[i] = positions[i - 1];
                positions[i] += sourceStride;
                if (serverBegin != 0 && i == serverBegin) {
                    positions[i] += sourceServerGap;
                }
            }
        }
    };
    auto prepareWave = [&](uint32_t slot) {
        for (uint32_t i = 0; i < arg.channelCount; ++i) {
            auto &transfer = waves[slot][i];
            auto &lane = first[mesh ? i : 0];
            if (!mesh) {
                const uint32_t ordinal = arg.ranks[i] < serverBegin ? arg.ranks[i] : arg.ranks[i] - 3;
                transfer.address = positions[ordinal];
            } else if (slot == 0) {
                transfer.address = lane.address;
            } else {
                transfer.address = waves[slot - 1][i].address;
                transfer.address += sourceStride;
                if (serverBegin != 0 && slot == serverBegin) {
                    transfer.address += sourceServerGap;
                }
            }
            transfer.local.token = lane.token;
            transfer.bytes = lane.bytes;
            if (slot == 0) {
                transfer.offset = lane.offset;
            } else {
                transfer.offset = waves[slot - 1][i].offset;
                transfer.offset += lane.bytes;
            }
        }
    };
    auto waitWave = [&](uint32_t slot) -> CcuResult {
        uint16_t mask = static_cast<uint16_t>((1U << arg.channelCount) - 1);
        if (arg.selfCopy && slot == 0) {
            mask = static_cast<uint16_t>(mask | (1U << arg.channelCount));
        }
        return ccu::EventWait(completed[slot % window], mask);
    };
    auto emitWave = [&](uint32_t slot) -> CcuResult {
        auto &event = completed[slot % window];
        for (uint32_t i = 0; i < arg.channelCount; ++i) {
            auto &transfer = waves[slot][i];
            transfer.local.addr = transfer.address;
            if (mesh) {
                CCU_IF(mode[0] == static_cast<uint64_t>(WriteMetadataMode::ReuseOutput)) {
                    remote.addr = remoteCachedAddress[i];
                    remote.token = remoteCachedToken[i];
                }
                CCU_ELSE {
                    remote.addr = remoteAddress[i];
                    remote.token = remoteToken[i];
                }
            } else {
                remote.addr = remoteAddress[i];
                remote.token = remoteToken[i];
            }
            remote.addr += transfer.offset;
            const uint16_t bit = static_cast<uint16_t>(1U << i);
            CCU_IF(transfer.bytes != 0) {
                CCU_CHK_RET(ccu::Write(arg.channels[i], remote, transfer.local, transfer.bytes, event, bit));
            }
            CCU_IF(transfer.bytes == 0) {
                CCU_CHK_RET(ccu::EventRecord(event, bit));
            }
        }
        if (arg.selfCopy && slot == 0) {
            const uint16_t bit = static_cast<uint16_t>(1U << arg.channelCount);
            CCU_IF(mode[0] == static_cast<uint64_t>(WriteMetadataMode::RootFusedBoth)) {
                // 阶段0只排空 Scratch；真正的复制留到最终 Output 段。
                CCU_CHK_RET(ccu::EventRecord(event, bit));
            }
            CCU_ELSE {
                CCU_IF(copyBytes != 0) {
                    CCU_CHK_RET(ccu::LocalCopy(copyTarget, copySource, copyBytes, event, bit));
                }
                CCU_IF(copyBytes == 0) {
                    CCU_CHK_RET(ccu::EventRecord(event, bit));
                }
            }
        }
        return CCU_SUCCESS;
    };
    auto runWaves = [&](uint32_t count) -> CcuResult {
        prepareCommon(count);
        prepareWave(0);
        CCU_CHK_RET(waitReady(allPeers));
        for (uint32_t slot = 0; slot < count; ++slot) {
            if (slot != 0) {
                // 每波操作数独立；先与在途搬运重叠准备，再消费将复用的窗口完成位。
                prepareWave(slot);
            }
            if (slot >= window) {
                CCU_CHK_RET(waitWave(slot - window));
            }
            CCU_CHK_RET(emitWave(slot));
        }
        const uint32_t tailBegin = count > window ? count - window : 0;
        for (uint32_t slot = tailBegin; slot < count; ++slot) {
            CCU_CHK_RET(waitWave(slot));
        }
        return CCU_SUCCESS;
    };
    auto runActive = [&]() -> CcuResult {
        CCU_IF(activeCount != 0) {
            if (mesh) {
                CCU_IF(activeCount == 9) {
                    CCU_CHK_RET(runWaves(9));
                }
                CCU_ELSE {
                    CCU_CHK_RET(runWaves(1));
                }
            } else {
                CCU_CHK_RET(runWaves(1));
            }
        }
        CCU_ELSE {
            // 无发送者也消费 READY；不访问未准备的发送变量或未记录的 Event。
            CCU_CHK_RET(waitReady(allPeers));
        }
        return CCU_SUCCESS;
    };
    auto finishPeers = [&](const std::vector<uint32_t> &peers) -> CcuResult {
        // 完整排空窗口后再收尾；参与对端的 READY 保留跨调用复用 FINISHED 位的边界。
        for (uint32_t i : peers) {
            CCU_CHK_RET(ccu::NotifyRecord(arg.channels[i], NOTIFY_INDEX, FINISHED_MASK));
        }
        for (uint32_t i : peers) {
            CCU_CHK_RET(ccu::NotifyWait(arg.channels[i], NOTIFY_INDEX, FINISHED_MASK));
        }
        return CCU_SUCCESS;
    };
    if (mesh) {
        CCU_CHK_RET(runActive());
        auto recordMesh = [&](uint16_t mask) -> CcuResult {
            for (uint32_t i : allPeers) {
                CCU_CHK_RET(ccu::NotifyRecord(arg.channels[i], NOTIFY_INDEX, mask));
            }
            return CCU_SUCCESS;
        };
        auto waitMesh = [&](uint16_t mask) -> CcuResult {
            for (uint32_t i : allPeers) {
                CCU_CHK_RET(ccu::NotifyWait(arg.channels[i], NOTIFY_INDEX, mask));
            }
            return CCU_SUCCESS;
        };
        CCU_IF(mode[0] == static_cast<uint64_t>(WriteMetadataMode::RootFusedBoth)) {
            // 九波已排空，立即放行 helper 的原 Mesh→Clos 交接。
            CCU_CHK_RET(recordMesh(FINISHED_MASK));
            ccu::Event finalDone;
            for (uint32_t i = 0; i < 2; ++i) {
                ccu::LocalAddr source;
                source.addr = finalFields[i];
                source.token = sharedToken;
                ccu::RemoteAddr target;
                target.addr = remoteCachedAddress[i];
                target.addr += finalFields[3];
                target.token = remoteCachedToken[i];
                CCU_IF(finalFields[2] != 0) {
                    CCU_CHK_RET(ccu::Write(arg.channels[i], target, source, finalFields[2],
                        finalDone, static_cast<uint16_t>(1U << i)));
                }
                CCU_ELSE {
                    CCU_CHK_RET(ccu::EventRecord(finalDone, static_cast<uint16_t>(1U << i)));
                }
            }
            CCU_IF(copyBytes != 0) {
                CCU_CHK_RET(ccu::LocalCopy(copyTarget, copySource, copyBytes, finalDone, 4));
            }
            CCU_ELSE {
                CCU_CHK_RET(ccu::EventRecord(finalDone, 4));
            }
            CCU_CHK_RET(ccu::EventWait(finalDone, 7));
            CCU_CHK_RET(waitMesh(FINISHED_MASK));
            CCU_CHK_RET(recordMesh(REUSED_FINISHED_MASK));
            CCU_CHK_RET(waitMesh(REUSED_FINISHED_MASK));
        }
        CCU_ELSE {
            CCU_IF(mode[0] == static_cast<uint64_t>(WriteMetadataMode::PublishBoth)) {
                CCU_CHK_RET(recordMesh(FINISHED_MASK));
                CCU_IF(activeCount == 0) {
                    // 中继返回前必须确认 Scratch 已写满；发送者把确认延后到下一阶段收尾。
                    CCU_CHK_RET(waitMesh(FINISHED_MASK));
                }
            }
            CCU_ELSE {
                CCU_IF(mode[0] == static_cast<uint64_t>(WriteMetadataMode::ReuseOutput)) {
                    CCU_IF(activeCount != 0) {
                        CCU_CHK_RET(waitMesh(FINISHED_MASK));
                    }
                    CCU_CHK_RET(recordMesh(REUSED_FINISHED_MASK));
                    CCU_CHK_RET(waitMesh(REUSED_FINISHED_MASK));
                }
                CCU_ELSE {
                    CCU_CHK_RET(finishPeers(allPeers));
                }
            }
        }
    } else {
        CCU_IF(rootServer[0] == arg.rank / 3) {
            CCU_CHK_RET(publishPeers(allPeers));
            CCU_CHK_RET(runActive());
            CCU_CHK_RET(finishPeers(allPeers));
        }
        CCU_ELSE {
            // 远端只与 root 服务器握手；静态生成三组通道，由每轮参数选中其中一组。
            for (uint32_t server = 0; server < 4; ++server) {
                if (server == arg.rank / 3) {
                    continue;
                }
                std::vector<uint32_t> peers;
                for (uint32_t i : allPeers) {
                    if (scatter_plan::TriClosPeerActive(arg.rank, arg.ranks[i], server)) {
                        peers.push_back(i);
                    }
                }
                if (peers.size() != 3) {
                    return CCU_E_PARA;
                }
                CCU_IF(rootServer[0] == server) {
                    CCU_CHK_RET(publishPeers(peers));
                    CCU_CHK_RET(waitReady(peers));
                    CCU_CHK_RET(finishPeers(peers));
                }
            }
        }
    }
    return CCU_SUCCESS;
}

// 2×8 Clos 只保留 root→八对端、helper→buddy 的真实逻辑边；零字节仍完成握手。
CcuResult BuildPairClosKernel(const ScatterKernelArg &arg)
{
    if (arg.channelCount != 8 || arg.slots != 1 || arg.selfCopy) return CCU_E_PARA;
    ccu::Variable publishedAddress, publishedToken, role, sourceBase, sourceToken;
    ccu::Variable targetOffset, sourceStride, normalBytes, buddyBytes, root, pairRelay, linkMode;
    uint32_t index = 0;
    CCU_CHK_RET(ccu::LoadArg(publishedAddress, index++));
    CCU_CHK_RET(ccu::LoadArg(publishedToken, index++));
    CCU_CHK_RET(ccu::LoadArg(role, index++));
    CCU_CHK_RET(ccu::LoadArg(sourceBase, index++));
    CCU_CHK_RET(ccu::LoadArg(sourceToken, index++));
    CCU_CHK_RET(ccu::LoadArg(targetOffset, index++));
    CCU_CHK_RET(ccu::LoadArg(sourceStride, index++));
    CCU_CHK_RET(ccu::LoadArg(normalBytes, index++));
    CCU_CHK_RET(ccu::LoadArg(buddyBytes, index++));
    CCU_CHK_RET(ccu::LoadArg(root, index++));
    CCU_CHK_RET(ccu::LoadArg(pairRelay, index++));
    if (index != scatter_plan::PAIR_CLOS_WRITE_ARGS) return CCU_E_PARA;
    CCU_CHK_RET(ccu::LoadArg(linkMode, index++));
    uint32_t buddy = arg.channelCount;
    std::vector<ccu::Variable> remoteAddress, remoteToken;
    for (uint32_t i = 0; i < arg.channelCount; ++i) {
        if (arg.ranks[i] % 8 == arg.rank % 8) buddy = i;
        remoteAddress.push_back(ccu::GetResByChannel<ccu::Variable>(arg.channels[i], ADDRESS_SLOT));
        remoteToken.push_back(ccu::GetResByChannel<ccu::Variable>(arg.channels[i], TOKEN_SLOT));
    }
    if (buddy == arg.channelCount) return CCU_E_PARA;
    CCU_CHK_RET(BindLinkedTargets(arg, linkMode, remoteAddress, remoteToken));
    ccu::Event completed;
    auto send = [&](uint32_t i, const ccu::Variable &address, ccu::Variable &bytes) -> CcuResult {
        CCU_IF(linkMode == 0) {
            CCU_CHK_RET(ccu::NotifyWait(arg.channels[i], NOTIFY_INDEX, READY_MASK));
        }
        CCU_IF(bytes != 0) {
            ccu::LocalAddr source;
            source.addr = address;
            source.token = sourceToken;
            ccu::RemoteAddr target;
            target.addr = remoteAddress[i];
            target.addr += targetOffset;
            target.token = remoteToken[i];
            CCU_CHK_RET(ccu::Write(arg.channels[i], target, source, bytes,
                completed, static_cast<uint16_t>(1U << i)));
        }
        CCU_ELSE {
            CCU_CHK_RET(ccu::EventRecord(completed, static_cast<uint16_t>(1U << i)));
        }
        return CCU_SUCCESS;
    };
    auto finish = [&](uint32_t i) -> CcuResult {
        CCU_CHK_RET(ccu::EventWait(completed, static_cast<uint16_t>(1U << i)));
        return ccu::NotifyRecord(arg.channels[i], NOTIFY_INDEX, FINISHED_MASK);
    };
    CCU_IF(role == 1) {
        ccu::Variable position;
        position = sourceBase;
        for (uint32_t i = 0; i < arg.channelCount; ++i) {
            // 对端按 rank 排序，首地址归一化到对面服务器的第0个 rank。
            if (arg.ranks[i] % 8 != i) return CCU_E_PARA;
            if (i != 0) position += sourceStride;
            CCU_CHK_RET(send(i, position, i == buddy ? buddyBytes : normalBytes));
        }
        for (uint32_t i = 0; i < arg.channelCount; ++i) CCU_CHK_RET(finish(i));
    }
    CCU_IF(role == 2) {
        CCU_CHK_RET(send(buddy, sourceBase, normalBytes));
        CCU_CHK_RET(finish(buddy));
    }
    CCU_IF(role == 0) {
        auto publish = [&](uint32_t i) -> CcuResult {
            CCU_IF(linkMode == 0) {
                CCU_CHK_RET(ccu::WriteVariableWithNotify(arg.channels[i], publishedAddress,
                    ADDRESS_SLOT, NOTIFY_INDEX, 1U << ADDRESS_SLOT));
                CCU_CHK_RET(ccu::WriteVariableWithNotify(arg.channels[i], publishedToken,
                    TOKEN_SLOT, NOTIFY_INDEX, 1U << TOKEN_SLOT));
            }
            return CCU_SUCCESS;
        };
        auto consume = [&](uint32_t i) -> CcuResult {
            return ccu::NotifyWait(arg.channels[i], NOTIFY_INDEX, FINISHED_MASK);
        };
        // 先发布所有真实来源，再等完成；buddy 与 root 重合时只处理一条边。
        for (uint32_t i = 0; i < arg.channelCount; ++i) {
            CCU_IF(root == arg.ranks[i]) {
                CCU_CHK_RET(publish(i));
            }
            if (i == buddy) {
                CCU_IF(root != arg.ranks[i]) {
                    CCU_IF(pairRelay != 0) {
                        CCU_CHK_RET(publish(i));
                    }
                }
            }
        }
        for (uint32_t i = 0; i < arg.channelCount; ++i) {
            CCU_IF(root == arg.ranks[i]) {
                CCU_CHK_RET(consume(i));
            }
            if (i == buddy) {
                CCU_IF(root != arg.ranks[i]) {
                    CCU_IF(pairRelay != 0) {
                        CCU_CHK_RET(consume(i));
                    }
                }
            }
        }
    }
    // role=3 为 DirectWrite 下同机无 Clos 数据的 rank；不产生任何跨机通知。
    return CCU_SUCCESS;
}

// 优化由注册时的拓扑选择，小消息独立的 Read Kernel 不受影响。
CcuResult BuildWriteKernel(CcuKernelArg opaque, bool relay)
{
    auto *arg = static_cast<ScatterKernelArg *>(opaque);
    if (arg == nullptr || arg->channelCount == 0 || arg->channelCount >= 16 || arg->slots == 0) {
        return CCU_E_PARA;
    }
    if (arg->roleWriteOnly) {
        return arg->roleLinkBytes != 0 ? BuildRoleLinkSendKernel(*arg) : BuildRoleSendKernel(*arg);
    }
    if (arg->optimization == WriteOptimization::Compact) {
        return relay ? BuildCompactWriteKernel(*arg) : CCU_E_PARA;
    }
    if (arg->optimization == WriteOptimization::FusedMetadata &&
        !scatter_plan::SameServer(scatter_plan::Topology::TwoByEight, arg->rank, arg->ranks[0])) {
        return relay ? BuildPairClosKernel(*arg) : CCU_E_PARA;
    }
    std::vector<ccu::Variable> remoteAddress;
    std::vector<ccu::Variable> remoteToken;
    std::vector<ccu::Variable> cachedOutput, cachedToken;
    const bool fusedMetadata = arg->optimization == WriteOptimization::FusedMetadata;
    const bool channelReady = arg->optimization == WriteOptimization::ChannelReady;
    const bool channelFinished = arg->optimization == WriteOptimization::ChannelFinished;
    if (fusedMetadata && (!relay || arg->channelCount != 7 || arg->slots != 1 || !arg->selfCopy)) {
        return CCU_E_PARA;
    }
    for (uint32_t i = 0; i < arg->channelCount; ++i) {
        remoteAddress.push_back(ccu::GetResByChannel<ccu::Variable>(arg->channels[i], ADDRESS_SLOT));
        remoteToken.push_back(ccu::GetResByChannel<ccu::Variable>(arg->channels[i], TOKEN_SLOT));
        if (fusedMetadata) {
            cachedOutput.push_back(ccu::GetResByChannel<ccu::Variable>(arg->channels[i], CACHED_OUTPUT_SLOT));
            cachedToken.push_back(ccu::GetResByChannel<ccu::Variable>(arg->channels[i], CACHED_TOKEN_SLOT));
        }
    }
    ccu::Variable publishedAddress, publishedToken, linkMode;
    ccu::LocalAddr copySource, copyTarget;
    ccu::Variable copyBytes, copySourceAddress, copyTargetAddress;
    uint32_t argIndex = 0;
    CCU_CHK_RET(ccu::LoadArg(publishedAddress, argIndex++));
    CCU_CHK_RET(ccu::LoadArg(publishedToken, argIndex++));
    CCU_CHK_RET(ccu::LoadArg(copySourceAddress, argIndex++));
    if (!fusedMetadata) {
        CCU_CHK_RET(ccu::LoadArg(copySource.token, argIndex++));
    }
    CCU_CHK_RET(ccu::LoadArg(copyTargetAddress, argIndex++));
    if (!fusedMetadata) {
        CCU_CHK_RET(ccu::LoadArg(copyTarget.token, argIndex++));
    }
    CCU_CHK_RET(ccu::LoadArg(copyBytes, argIndex++));
    copySource.addr = copySourceAddress;
    copyTarget.addr = copyTargetAddress;
    struct MetadataArgs {
        ccu::Variable mode, outputAddress, outputToken;
    };
    std::vector<MetadataArgs> metadata(fusedMetadata ? 1 : 0);
    if (fusedMetadata) {
        CCU_CHK_RET(ccu::LoadArg(metadata[0].mode, argIndex++));
        CCU_CHK_RET(ccu::LoadArg(metadata[0].outputAddress, argIndex++));
        CCU_CHK_RET(ccu::LoadArg(metadata[0].outputToken, argIndex++));
    }
    const uint32_t slots = arg->slots;
    std::vector<ccu::Variable> activeSlots(relay ? slots : 0);
    struct WaveTransfer {
        ccu::LocalAddr local;
        ccu::Variable localAddress, remoteOffset, bytes;
    };
    std::vector<std::vector<WaveTransfer>> waves(slots);
    for (auto &wave : waves) {
        wave.resize(arg->channelCount);
    }
    for (auto &active : activeSlots) {
        CCU_CHK_RET(ccu::LoadArg(active, argIndex++));
    }
    struct SharedTransferArgs {
        ccu::Variable token, offset, bytes;
    };
    std::vector<SharedTransferArgs> shared(fusedMetadata ? 1 : 0);
    if (fusedMetadata) {
        CCU_CHK_RET(ccu::LoadArg(shared[0].token, argIndex++));
        CCU_CHK_RET(ccu::LoadArg(shared[0].offset, argIndex++));
        CCU_CHK_RET(ccu::LoadArg(shared[0].bytes, argIndex++));
    }
    for (auto &wave : waves) {
        for (auto &transfer : wave) {
            // 参数全部在条件外加载，未激活波次也保留固定索引和独立 Variable。
            CCU_CHK_RET(ccu::LoadArg(transfer.localAddress, argIndex++));
            if (!fusedMetadata) {
                CCU_CHK_RET(ccu::LoadArg(transfer.local.token, argIndex++));
                CCU_CHK_RET(ccu::LoadArg(transfer.remoteOffset, argIndex++));
                CCU_CHK_RET(ccu::LoadArg(transfer.bytes, argIndex++));
            }
        }
    }
    if (fusedMetadata) {
        if (argIndex != scatter_plan::PAIR_MESH_WRITE_ARGS) {
            return CCU_E_PARA;
        }
        copySource.token = shared[0].token;
        copyTarget.token = publishedToken;
    }
    CCU_CHK_RET(ccu::LoadArg(linkMode, argIndex++));
    CCU_CHK_RET(BindLinkedTargets(*arg, linkMode, remoteAddress, remoteToken));
    if (fusedMetadata && arg->roleLinkBytes != 0) {
        CCU_IF(linkMode != 0) {
            for (uint32_t i = 0; i < arg->channelCount; ++i) {
                cachedOutput[i] = arg->roleLinkTargets[i].address;
                cachedToken[i] = arg->roleLinkTargets[i].token;
            }
        }
    }
    auto publish = [&](uint32_t i) -> CcuResult {
        CCU_CHK_RET(ccu::WriteVariableWithNotify(arg->channels[i], publishedAddress,
                                               ADDRESS_SLOT, NOTIFY_INDEX, 1U << ADDRESS_SLOT));
        CCU_CHK_RET(ccu::WriteVariableWithNotify(arg->channels[i], publishedToken,
                                               TOKEN_SLOT, NOTIFY_INDEX, 1U << TOKEN_SLOT));
        return CCU_SUCCESS;
    };
    CCU_IF(linkMode == 0) {
        if (fusedMetadata) {
            // 所有参数已在条件外加载，避免 CANN 9.1.0 拒绝注册含条件 LoadArg 的 Kernel。
            CCU_IF(metadata[0].mode != static_cast<uint64_t>(WriteMetadataMode::ReuseOutput)) {
                for (uint32_t i = 0; i < arg->channelCount; ++i) {
                    CCU_CHK_RET(publish(i));
                }
                CCU_IF(metadata[0].mode == static_cast<uint64_t>(WriteMetadataMode::PublishBoth)) {
                    for (uint32_t i = 0; i < arg->channelCount; ++i) {
                        CCU_CHK_RET(ccu::WriteVariableWithNotify(arg->channels[i], metadata[0].outputAddress,
                            CACHED_OUTPUT_SLOT, NOTIFY_INDEX, CACHED_OUTPUT_MASK));
                        CCU_CHK_RET(ccu::WriteVariableWithNotify(arg->channels[i], metadata[0].outputToken,
                            CACHED_TOKEN_SLOT, NOTIFY_INDEX, CACHED_TOKEN_MASK));
                    }
                    for (uint32_t i = 0; i < arg->channelCount; ++i) {
                        CCU_CHK_RET(ccu::NotifyWait(arg->channels[i], NOTIFY_INDEX,
                            READY_MASK | CACHED_OUTPUT_MASK | CACHED_TOKEN_MASK));
                    }
                }
                CCU_ELSE {
                    for (uint32_t i = 0; i < arg->channelCount; ++i) {
                        CCU_CHK_RET(ccu::NotifyWait(arg->channels[i], NOTIFY_INDEX, READY_MASK));
                    }
                }
            }
        } else {
            for (uint32_t i = 0; i < arg->channelCount; ++i) {
                CCU_CHK_RET(publish(i));
            }
            if (!channelReady) {
                for (uint32_t i = 0; i < arg->channelCount; ++i) {
                    CCU_CHK_RET(ccu::NotifyWait(arg->channels[i], NOTIFY_INDEX, READY_MASK));
                }
            }
        }
    }

    const uint32_t window = relay && slots > 1 ? 2 : 1;
    std::vector<ccu::Event> completed(window);
    ccu::RemoteAddr remote;
    auto hasCopy = [&](uint32_t slot) { return arg->selfCopy && (relay ? slot == 0 : slot + 1 == slots); };
    auto waitMask = [&](uint32_t slot, uint16_t mask) -> CcuResult {
        if (relay) {
            // 等待由旧波次是否发出决定，不能用即将发出的新波次的标记判断。
            CCU_IF(activeSlots[slot] != 0) {
                CCU_CHK_RET(ccu::EventWait(completed[slot % window], mask));
            }
        } else {
            CCU_CHK_RET(ccu::EventWait(completed[slot % window], mask));
        }
        return CCU_SUCCESS;
    };
    auto waitWave = [&](uint32_t slot) -> CcuResult {
        uint16_t mask = static_cast<uint16_t>((1U << arg->channelCount) - 1);
        if (hasCopy(slot)) {
            mask = static_cast<uint16_t>(mask | (1U << arg->channelCount));
        }
        return waitMask(slot, mask);
    };
    auto emitTransfer = [&](uint32_t slot, uint32_t i) -> CcuResult {
        auto &event = completed[slot % window];
        auto &transfer = waves[slot][i];
        transfer.local.addr = transfer.localAddress;
        if (fusedMetadata) {
            transfer.local.token = shared[0].token;
            CCU_IF(metadata[0].mode == static_cast<uint64_t>(WriteMetadataMode::ReuseOutput)) {
                remote.addr = cachedOutput[i];
                remote.token = cachedToken[i];
            }
            CCU_ELSE {
                remote.addr = remoteAddress[i];
                remote.token = remoteToken[i];
            }
        } else {
            remote.addr = remoteAddress[i];
            remote.addr += transfer.remoteOffset;
            remote.token = remoteToken[i];
        }
        if (fusedMetadata) {
            remote.addr += shared[0].offset;
        }
        const uint16_t bit = static_cast<uint16_t>(1U << i);
        auto &bytes = fusedMetadata ? shared[0].bytes : transfer.bytes;
        CCU_IF(bytes != 0) {
            CCU_CHK_RET(ccu::Write(arg->channels[i], remote, transfer.local, bytes, event, bit));
        }
        CCU_IF(bytes == 0) {
            CCU_CHK_RET(ccu::EventRecord(event, bit));
        }
        return CCU_SUCCESS;
    };
    auto emitCopy = [&](uint32_t slot) -> CcuResult {
        auto &event = completed[slot % window];
        // 共用中继布局的写路径把本地拷贝与首波发送一起发起，其余路径保留原拷贝时机。
        if (hasCopy(slot)) {
            const uint16_t bit = static_cast<uint16_t>(1U << arg->channelCount);
            CCU_IF(copyBytes != 0) {
                CCU_CHK_RET(ccu::LocalCopy(copyTarget, copySource, copyBytes, event, bit));
            }
            CCU_IF(copyBytes == 0) {
                CCU_CHK_RET(ccu::EventRecord(event, bit));
            }
        }
        return CCU_SUCCESS;
    };
    auto emitWave = [&](uint32_t slot) -> CcuResult {
        for (uint32_t i = 0; i < arg->channelCount; ++i) {
            CCU_CHK_RET(emitTransfer(slot, i));
        }
        return emitCopy(slot);
    };
    if (channelReady) {
        for (uint32_t i = 0; i < arg->channelCount; ++i) {
            // Write 异步发起；下一条 channel 的 READY 等待可与前面的数据传输重叠。
            // 纯接收组也消费全部 READY，以维持下一阶段及下一调用的槽位复用顺序。
            CCU_IF(linkMode == 0) {
                CCU_CHK_RET(ccu::NotifyWait(arg->channels[i], NOTIFY_INDEX, READY_MASK));
            }
            if (relay) {
                CCU_IF(activeSlots[0] != 0) {
                    CCU_CHK_RET(emitTransfer(0, i));
                }
            } else {
                CCU_CHK_RET(emitTransfer(0, i));
            }
        }
        if (relay) {
            CCU_IF(activeSlots[0] != 0) {
                CCU_CHK_RET(emitCopy(0));
            }
        } else {
            CCU_CHK_RET(emitCopy(0));
        }
    }
    for (uint32_t slot = channelReady ? 1 : 0; slot < slots; ++slot) {
        if (slot >= window) {
            CCU_CHK_RET(waitWave(slot - window));
        }
        if (relay) {
            // 条件只包住本地发送波次；纯接收组也必须在下方等待对端 FINISHED。
            CCU_IF(activeSlots[slot] != 0) {
                CCU_CHK_RET(emitWave(slot));
            }
        } else {
            CCU_CHK_RET(emitWave(slot));
        }
    }
    const uint32_t tailBegin = slots > window ? slots - window : 0;
    if (channelFinished) {
        // 所有 Write 已发起。逐 channel 排空剩余窗口并通知，避免尾部先等齐所有 channel。
        for (uint32_t i = 0; i < arg->channelCount; ++i) {
            for (uint32_t slot = tailBegin; slot < slots; ++slot) {
                CCU_CHK_RET(waitMask(slot, static_cast<uint16_t>(1U << i)));
            }
            CCU_CHK_RET(ccu::NotifyRecord(arg->channels[i], NOTIFY_INDEX, FINISHED_MASK));
        }
        for (uint32_t slot = tailBegin; slot < slots; ++slot) {
            if (hasCopy(slot)) {
                CCU_CHK_RET(waitMask(slot, static_cast<uint16_t>(1U << arg->channelCount)));
            }
        }
        for (uint32_t i = 0; i < arg->channelCount; ++i) {
            CCU_CHK_RET(ccu::NotifyWait(arg->channels[i], NOTIFY_INDEX, FINISHED_MASK));
        }
    } else {
        for (uint32_t slot = tailBegin; slot < slots; ++slot) {
            CCU_CHK_RET(waitWave(slot));
        }
        auto finish = [&](uint16_t mask) -> CcuResult {
            for (uint32_t i = 0; i < arg->channelCount; ++i) {
                CCU_CHK_RET(ccu::NotifyRecord(arg->channels[i], NOTIFY_INDEX, mask));
            }
            for (uint32_t i = 0; i < arg->channelCount; ++i) {
                CCU_CHK_RET(ccu::NotifyWait(arg->channels[i], NOTIFY_INDEX, mask));
            }
            return CCU_SUCCESS;
        };
        if (fusedMetadata) {
            CCU_IF(metadata[0].mode == static_cast<uint64_t>(WriteMetadataMode::PublishBoth)) {
                for (uint32_t i = 0; i < arg->channelCount; ++i) {
                    CCU_CHK_RET(ccu::NotifyRecord(arg->channels[i], NOTIFY_INDEX, FINISHED_MASK));
                }
                CCU_IF(activeSlots[0] == 0) {
                    // 纯接收者保留暂存区就绪边界，不能提前放行 Clos 转发。
                    for (uint32_t i = 0; i < arg->channelCount; ++i) {
                        CCU_CHK_RET(ccu::NotifyWait(arg->channels[i], NOTIFY_INDEX, FINISHED_MASK));
                    }
                }
            }
            CCU_ELSE {
                // 第二阶段排空后，发送者先消费推迟的阶段0确认，再对称消费本阶段的独立完成位。
                CCU_IF(metadata[0].mode == static_cast<uint64_t>(WriteMetadataMode::ReuseOutput)) {
                    CCU_IF(activeSlots[0] != 0) {
                        for (uint32_t i = 0; i < arg->channelCount; ++i) {
                            CCU_CHK_RET(ccu::NotifyWait(arg->channels[i], NOTIFY_INDEX, FINISHED_MASK));
                        }
                    }
                    CCU_CHK_RET(finish(REUSED_FINISHED_MASK));
                }
                CCU_ELSE {
                    CCU_CHK_RET(finish(FINISHED_MASK));
                }
            }
        } else {
            CCU_CHK_RET(finish(FINISHED_MASK));
        }
    }
    return CCU_SUCCESS;
}
}

CcuResult ScatterWriteKernel(CcuKernelArg arg)
{
    return BuildWriteKernel(arg, false);
}

CcuResult ScatterReadKernel(CcuKernelArg arg)
{
    return BuildReadKernel(arg);
}

CcuResult ScatterRelayKernel(CcuKernelArg arg)
{
    return BuildWriteKernel(arg, true);
}
}
