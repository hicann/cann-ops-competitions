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

#include <vector>

#include "ccu_kernel.h"

namespace ops_hccl {
namespace ccu = AscendC::ccu;

#define P2_CCU_CHK_RET(expr) \
    do { \
        const CcuResult result = (expr); \
        if (result != CCU_SUCCESS) { \
            return result; \
        } \
    } while (false)

namespace {
    constexpr uint32_t kCkeIndex = 0;
    // 相位1（散射）槽/bit：槽 0/1 对端 outputAddr/Token（bit0/1），槽 2/3 helper 发布 scratch（bit2/3），
    // bit7 后同步，bit8 前缀落地。chunk 间复用由相位 1 的全连接后同步锁步保证。
    constexpr uint32_t kOutputAddrSlot = 0;
    constexpr uint32_t kOutputTokenSlot = 1;
    constexpr uint32_t kScratchAddrSlot = 2;
    constexpr uint32_t kScratchTokenSlot = 3;
    constexpr uint16_t kOutputAddrBit = 1U << 0;
    constexpr uint16_t kOutputTokenBit = 1U << 1;
    constexpr uint16_t kScratchAddrBit = 1U << 2;
    constexpr uint16_t kScratchTokenBit = 1U << 3;
    constexpr uint16_t kPostSyncBit = 1U << 7;
    constexpr uint16_t kPrefixReadyBit = 1U << 8;
    // E4.5 小消息 ACK（stageMode=2）：root 经槽 0/1 发布 input addr/token（复用 bit0/1），peer 完成
    // Read 后回 ACK（bit2）。跨次复用安全：root 等齐全部 ACK 才结束本 launch，同 stream 顺序保证
    // 下一次发布发生在所有 peer 消费上一代描述符之后，无 Record 合并。
    constexpr uint16_t kSmallAckBit = 1U << 2;
    // 相位2（点对点中继）槽/bit：槽 4/5 dest 发布 outputAddr/Token（bit4/5），bit6 epoch。
    // 跨 chunk 复用安全：dest 记录 chunk(i+1) 的 addr 必先消费 chunk(i) 的 epoch
    // （relayStream 顺序 + epoch wait 在 P2(i) 末尾）；helper 记录 chunk(i+1) 的 epoch
    // 必先消费 chunk(i+1) 的 addr（wait 在转发之前），两条消费链闭合，无合并丢失。
    constexpr uint32_t kP2AddrSlot = 4;
    constexpr uint32_t kP2TokenSlot = 5;
    constexpr uint16_t kP2AddrBit = 1U << 4;
    constexpr uint16_t kP2TokenBit = 1U << 5;
    constexpr uint16_t kEpochBit = 1U << 6;
} // namespace

// E5.3: register-time role specialization, no runtime stage dispatch.
CcuResult CcuKernelSmall(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgDirectProbe *>(arg);
    if (kernelArg == nullptr || kernelArg->channelCount == 0 || kernelArg->channelCount >= kernelArg->rankSize) {
        return CCU_E_PARA;
    }
    ccu::Variable inputAddr, outputAddr, inputToken, outputToken, totalSliceBytes;
    if (kernelArg->rankId == kernelArg->rootId) {
        P2_CCU_CHK_RET(ccu::LoadArg(inputAddr, 0));
        P2_CCU_CHK_RET(ccu::LoadArg(outputAddr, 1));
        P2_CCU_CHK_RET(ccu::LoadArg(inputToken, 2));
        P2_CCU_CHK_RET(ccu::LoadArg(outputToken, 3));
        P2_CCU_CHK_RET(ccu::LoadArg(totalSliceBytes, 4));
    } else {
        P2_CCU_CHK_RET(ccu::LoadArg(outputAddr, 0));
        P2_CCU_CHK_RET(ccu::LoadArg(outputToken, 1));
    }
    if (kernelArg->rankId == kernelArg->rootId) {
        // P1c 逐 peer 交织（对照 hkx_v1 A916）：每收齐一个 peer 的 READY 立即发该 peer 的 Write，
        // 慢 peer 的就绪等待与已发出 Write 的传输重叠，不再被最慢 READY 整体阻塞。
        // 无环等：peer 的 READY 在 kernel 起始无条件发布，root 的 Wait 只依赖它。
        ccu::Event remoteDone;
        ccu::Event selfDone;
        // 仅静态 selfCopy 组（host 只给第一个非空 die 组置位）有此段；
        // src/dst 构造与原位置逐字节相同，只是发起时机提前到 READY 等待之前。
        if (kernelArg->selfCopy != 0) {
            ccu::LocalAddr src;
            src.addr = inputAddr;
            for (uint32_t peerOffset = 0; peerOffset < kernelArg->rankId; ++peerOffset) {
                src.addr += totalSliceBytes;
            }
            src.token = inputToken;
            ccu::LocalAddr dst;
            dst.addr = outputAddr;
            dst.token = outputToken;
            P2_CCU_CHK_RET(ccu::LocalCopy(dst, src, totalSliceBytes, selfDone, 1));
        }
        // Each group starts a fresh runtime cursor; peer ranks stay in channel order.
        ccu::Address cursor;
        cursor = inputAddr;
        uint32_t previousPeer = 0;
        for (uint32_t i = 0; i < kernelArg->channelCount; ++i) {
            const uint32_t peer = kernelArg->peerRanks[i];
            if (peer >= kernelArg->rankSize || peer == kernelArg->rootId || (i != 0 && peer <= previousPeer)) {
                return CCU_E_PARA;
            }
            for (uint32_t delta = previousPeer; delta < peer; ++delta) {
                cursor += totalSliceBytes;
            }
            previousPeer = peer;
            // Assignment into a freshly allocated Address emits a copy, not an alias.
            ccu::LocalAddr src;
            src.addr = cursor;
            src.token = inputToken;
            P2_CCU_CHK_RET(ccu::NotifyWait(kernelArg->channels[i], kCkeIndex, kOutputAddrBit | kOutputTokenBit));
            ccu::RemoteAddr dst;
            dst.addr = ccu::GetResByChannel<ccu::Variable>(kernelArg->channels[i], kOutputAddrSlot);
            dst.token = ccu::GetResByChannel<ccu::Variable>(kernelArg->channels[i], kOutputTokenSlot);
            const uint16_t peerMask = static_cast<uint16_t>(1U << i);
            P2_CCU_CHK_RET(ccu::Write(kernelArg->channels[i], dst, src, totalSliceBytes, remoteDone, peerMask));
        }
        // All Writes are issued before completion release; consume each bit exactly once.
        for (uint32_t i = 0; i < kernelArg->channelCount; ++i) {
            const uint16_t peerMask = static_cast<uint16_t>(1U << i);
            P2_CCU_CHK_RET(ccu::EventWait(remoteDone, peerMask));
            P2_CCU_CHK_RET(ccu::NotifyRecord(kernelArg->channels[i], kCkeIndex, kPostSyncBit));
        }
        // root 返回前确认自拷贝落地；无 selfCopy 的组不生成该等待（无生产者）。
        if (kernelArg->selfCopy != 0) {
            P2_CCU_CHK_RET(ccu::EventWait(selfDone, 1));
        }
    } else if (kernelArg->smallRootIdx < kernelArg->channelCount) {
        // 非 root：只在持有 root channel 的组翻译本协议；host 只 launch 该组（ctx.smallRootGroup），
        // 其余组 stageMode=4 分支为空——翻译期不能 return 错误码，否则注册直接失败。
        const ChannelHandle channel = kernelArg->channels[kernelArg->smallRootIdx];
        P2_CCU_CHK_RET(ccu::WriteVariableWithNotify(channel, outputAddr, kOutputAddrSlot, kCkeIndex, kOutputAddrBit));
        P2_CCU_CHK_RET(
            ccu::WriteVariableWithNotify(channel, outputToken, kOutputTokenSlot, kCkeIndex, kOutputTokenBit));
        P2_CCU_CHK_RET(ccu::NotifyWait(channel, kCkeIndex, kPostSyncBit));
    }
    return CCU_SUCCESS;
}

// E4 全 helper 中继：按本地 die 分组的自包含子集 scatter（phase 1）+ 点对点中继（phase 2）。
// 角色/变体全部由 kernelArg 常量表达（注册翻译期求值，对 checker 为静态结构）；
// relayY==0 的 chunk 经 CCU_IF 退化为纯直连行为，无需独立 kernel 实例。
CcuResult CcuKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgDirectProbe *>(arg);
    if (kernelArg == nullptr || kernelArg->rankSize == 0 || kernelArg->rankSize > MAX_RANK_SIZE
        || kernelArg->channelCount == 0 || kernelArg->channelCount >= kernelArg->rankSize) {
        return CCU_E_PARA;
    }

    ccu::Variable inputAddr;
    ccu::Variable outputAddr;
    ccu::Variable inputToken;
    ccu::Variable outputToken;
    ccu::Variable totalSliceBytes;
    ccu::Variable chunkOffset;
    ccu::Variable chunkBytes;
    ccu::Variable scratchAddr;
    ccu::Variable scratchToken;
    ccu::Variable relayY;
    ccu::Variable relaySuffix;
    // E4.8 紧凑接收参数（借鉴 R917）：非 root 的相位1 kernel 只用 7 个参数，按连续下标加载，
    // 物理上少发射 5 条 LoadArg（翻译器会把全部 LoadArg 提升到 mission 前导，少声明才是真减载）。
    // 紧凑布局：[0]=outputAddr [1]=outputToken [2]=scratchAddr [3]=scratchToken
    //          [4]=relayY [5]=totalSliceBytes [6]=stageMode
    // root/P2 kernel 保持原 11/12 参布局。host 按 rank 全局角色（isRoot）选择布局，逐组一致。
    const bool compactP1Receiver = kernelArg->phase == 1 && kernelArg->rankId != kernelArg->rootId;
    if (compactP1Receiver) {
        P2_CCU_CHK_RET(ccu::LoadArg(outputAddr, 0));
        P2_CCU_CHK_RET(ccu::LoadArg(outputToken, 1));
        P2_CCU_CHK_RET(ccu::LoadArg(scratchAddr, 2));
        P2_CCU_CHK_RET(ccu::LoadArg(scratchToken, 3));
        P2_CCU_CHK_RET(ccu::LoadArg(relayY, 4));
        P2_CCU_CHK_RET(ccu::LoadArg(totalSliceBytes, 5));
    } else {
        P2_CCU_CHK_RET(ccu::LoadArg(inputAddr, 0));
        P2_CCU_CHK_RET(ccu::LoadArg(outputAddr, 1));
        P2_CCU_CHK_RET(ccu::LoadArg(inputToken, 2));
        P2_CCU_CHK_RET(ccu::LoadArg(outputToken, 3));
        P2_CCU_CHK_RET(ccu::LoadArg(totalSliceBytes, 4));
        P2_CCU_CHK_RET(ccu::LoadArg(chunkOffset, 5));
        P2_CCU_CHK_RET(ccu::LoadArg(chunkBytes, 6));
        P2_CCU_CHK_RET(ccu::LoadArg(scratchAddr, 7));
        P2_CCU_CHK_RET(ccu::LoadArg(scratchToken, 8));
        P2_CCU_CHK_RET(ccu::LoadArg(relayY, 9));
        P2_CCU_CHK_RET(ccu::LoadArg(relaySuffix, 10));
    }

    if (kernelArg->phase == 3) {
        // Every cross-server edge has an independent READY -> DONE handshake.
        // All records precede waits, so no static peer-order cycle is introduced.
        for (uint32_t i = 0; i < kernelArg->channelCount; ++i) {
            const auto ch = kernelArg->channels[i];
            P2_CCU_CHK_RET(ccu::WriteVariableWithNotify(ch, outputAddr, kP2AddrSlot, kCkeIndex, kP2AddrBit));
            P2_CCU_CHK_RET(ccu::WriteVariableWithNotify(ch, outputToken, kP2TokenSlot, kCkeIndex, kP2TokenBit));
        }
        if (kernelArg->relayDestIdx != UINT32_MAX) {
            if (kernelArg->relayDestIdx >= kernelArg->channelCount) {
                return CCU_E_PARA;
            }
            const auto ch = kernelArg->channels[kernelArg->relayDestIdx];
            // L1 目标优先：只等转发目标的 READY 即发起 Write，无关 peer 的 READY 在 Write 发出后消费，
            // 慢 peer 的描述符不再阻塞本 helper 的转发（任务顺序论证见
            // .syncheck/tests/final_ccu/e49/l1-task-order.md）。
            P2_CCU_CHK_RET(ccu::NotifyWait(ch, kCkeIndex, kP2AddrBit | kP2TokenBit));
            ccu::LocalAddr src;
            src.addr = scratchAddr;
            src.token = scratchToken;
            ccu::RemoteAddr dst;
            dst.addr = ccu::GetResByChannel<ccu::Variable>(ch, kP2AddrSlot);
            dst.addr += chunkOffset;
            dst.token = ccu::GetResByChannel<ccu::Variable>(ch, kP2TokenSlot);
            ccu::Event done;
            P2_CCU_CHK_RET(ccu::Write(ch, dst, src, relayY, done, 1));
            for (uint32_t i = 0; i < kernelArg->channelCount; ++i) {
                if (i == kernelArg->relayDestIdx) {
                    continue;
                }
                P2_CCU_CHK_RET(ccu::NotifyWait(kernelArg->channels[i], kCkeIndex, kP2AddrBit | kP2TokenBit));
            }
            P2_CCU_CHK_RET(ccu::EventWait(done, 1));
        } else {
            for (uint32_t i = 0; i < kernelArg->channelCount; ++i) {
                P2_CCU_CHK_RET(ccu::NotifyWait(kernelArg->channels[i], kCkeIndex, kP2AddrBit | kP2TokenBit));
            }
        }
        for (uint32_t i = 0; i < kernelArg->channelCount; ++i) {
            P2_CCU_CHK_RET(ccu::NotifyRecord(kernelArg->channels[i], kCkeIndex, kEpochBit));
        }
        for (uint32_t i = 0; i < kernelArg->channelCount; ++i) {
            P2_CCU_CHK_RET(ccu::NotifyWait(kernelArg->channels[i], kCkeIndex, kEpochBit));
        }
        return CCU_SUCCESS;
    }

    if (kernelArg->phase == 2) {
        // ========== 相位2：点对点中继（单 channel） ==========
        // dest 发布本 chunk 输出地址/token；helper 消费后把 scratch 前缀转发到 dest
        // 输出前缀，完成后 Record epoch；dest 仅等待 epoch。host 仅等待 helper 前缀阶段。
        if (kernelArg->channelCount != 1) {
            return CCU_E_PARA;
        }
        const ChannelHandle channel = kernelArg->channels[0];
        if (kernelArg->relayRole != 0) {
            P2_CCU_CHK_RET(ccu::NotifyWait(channel, kCkeIndex, kP2AddrBit | kP2TokenBit));
            ccu::LocalAddr src;
            src.addr = scratchAddr;
            src.token = scratchToken;
            ccu::RemoteAddr dst;
            dst.addr = ccu::GetResByChannel<ccu::Variable>(channel, kP2AddrSlot);
            dst.addr += chunkOffset;
            dst.token = ccu::GetResByChannel<ccu::Variable>(channel, kP2TokenSlot);
            ccu::Event forwardDone;
            CCU_IF(relayY != 0ULL)
            {
                P2_CCU_CHK_RET(ccu::Write(channel, dst, src, relayY, forwardDone, 0x1));
                P2_CCU_CHK_RET(ccu::EventWait(forwardDone, 0x1));
            }
            P2_CCU_CHK_RET(ccu::NotifyRecord(channel, kCkeIndex, kEpochBit));
        } else {
            P2_CCU_CHK_RET(ccu::WriteVariableWithNotify(channel, outputAddr, kP2AddrSlot, kCkeIndex, kP2AddrBit));
            P2_CCU_CHK_RET(ccu::WriteVariableWithNotify(channel, outputToken, kP2TokenSlot, kCkeIndex, kP2TokenBit));
            P2_CCU_CHK_RET(ccu::NotifyWait(channel, kCkeIndex, kEpochBit));
        }
        return CCU_SUCCESS;
    }

    // Reuse the registered P1 mission: prefix and regular launches are ordered on its stream.
    // stageMode 为运行期分支（0=regular，1=prefix，2=small，3=star），复用相位1 注册，
    // 不新增 per-die mission（独立注册在 128 dest rank 上已触发 CCU_E_UNAVAIL）。
    // 下标按角色：root 为 arg11，紧凑接收端为 arg6。
    ccu::Variable prefixStage;
    P2_CCU_CHK_RET(ccu::LoadArg(prefixStage, compactP1Receiver ? 6 : 11));
    CCU_IF(prefixStage == 2ULL)
    {
        // ========== E4.5 小消息：root 发布 -> peer 主动 Read -> peer ACK ==========
        // 通信与同步只沿 root-peer 边展开；root 先发布全部描述符再做自拷贝，最后统一等 ACK，
        // 避免逐 peer 发送后立即等待造成串行。非 root 只经本组 root channel（host 只 launch 该组）。
        if (kernelArg->rankId == kernelArg->rootId) {
            for (uint32_t i = 0; i < kernelArg->channelCount; ++i) {
                P2_CCU_CHK_RET(ccu::WriteVariableWithNotify(
                    kernelArg->channels[i], inputAddr, kOutputAddrSlot, kCkeIndex, kOutputAddrBit));
                P2_CCU_CHK_RET(ccu::WriteVariableWithNotify(
                    kernelArg->channels[i], inputToken, kOutputTokenSlot, kCkeIndex, kOutputTokenBit));
            }
            if (kernelArg->selfCopy != 0) {
                ccu::LocalAddr src;
                src.addr = inputAddr;
                for (uint32_t peerOffset = 0; peerOffset < kernelArg->rankId; ++peerOffset) {
                    src.addr += totalSliceBytes;
                }
                src.token = inputToken;
                ccu::LocalAddr dst;
                dst.addr = outputAddr;
                dst.token = outputToken;
                ccu::Event selfDone;
                P2_CCU_CHK_RET(ccu::LocalCopy(dst, src, totalSliceBytes, selfDone, 1));
                P2_CCU_CHK_RET(ccu::EventWait(selfDone, 1));
            }
            for (uint32_t i = 0; i < kernelArg->channelCount; ++i) {
                P2_CCU_CHK_RET(ccu::NotifyWait(kernelArg->channels[i], kCkeIndex, kSmallAckBit));
            }
        } else if (kernelArg->smallRootIdx < kernelArg->channelCount) {
            // 本组持有 root channel 才翻译接收协议；host 只 launch 该组（ctx.smallRootGroup），
            // 其余组的 stageMode=2 分支为空——翻译期不能 return 错误码，否则注册直接失败。
            const ChannelHandle channel = kernelArg->channels[kernelArg->smallRootIdx];
            P2_CCU_CHK_RET(ccu::NotifyWait(channel, kCkeIndex, kOutputAddrBit | kOutputTokenBit));
            ccu::RemoteAddr src;
            src.addr = ccu::GetResByChannel<ccu::Variable>(channel, kOutputAddrSlot);
            for (uint32_t peerOffset = 0; peerOffset < kernelArg->rankId; ++peerOffset) {
                src.addr += totalSliceBytes;
            }
            src.token = ccu::GetResByChannel<ccu::Variable>(channel, kOutputTokenSlot);
            ccu::LocalAddr dst;
            dst.addr = outputAddr;
            dst.token = outputToken;
            ccu::Event readDone;
            P2_CCU_CHK_RET(ccu::Read(channel, dst, src, totalSliceBytes, readDone, 1));
            P2_CCU_CHK_RET(ccu::EventWait(readDone, 1));
            P2_CCU_CHK_RET(ccu::NotifyRecord(channel, kCkeIndex, kSmallAckBit));
        }
    }
    // E5.0 小消息短握手 Push（arg11/arg6 stageMode=4 运行期分支，复用相位1 注册）：
    // peer 发布 output 描述符 READY(槽0/1,bit0/1) -> root 逐 peer 收齐即发 Write ->
    // EventWait 远端写落地 -> root 发 DONE(bit7) -> peer 等 DONE 结束。无尾 ACK；
    // 协议自平衡与跨次安全证明见 .syncheck/tests/final_ccu/e50/p1-protocol-proof.md。
    // E5.2-A：selfCopy 提前到 READY 等待之前发起，远端 DONE 与自拷贝完成解耦
    // （remoteDone/selfDone 两个独立 Event 各自等待）；通信条数、READY/DONE 协议、
    // launch 数量与 stream 调度不变，证明见 .syncheck/tests/final_ccu/e52/a-proof.md。
    CCU_IF(prefixStage == 4ULL)
    {
        if (kernelArg->rankId == kernelArg->rootId) {
            // P1c 逐 peer 交织（对照 hkx_v1 A916）：每收齐一个 peer 的 READY 立即发该 peer 的 Write，
            // 慢 peer 的就绪等待与已发出 Write 的传输重叠，不再被最慢 READY 整体阻塞。
            // 无环等：peer 的 READY 在 kernel 起始无条件发布，root 的 Wait 只依赖它。
            ccu::Event remoteDone;
            ccu::Event selfDone;
            // 仅静态 selfCopy 组（host 只给第一个非空 die 组置位）有此段；
            // src/dst 构造与原位置逐字节相同，只是发起时机提前到 READY 等待之前。
            if (kernelArg->selfCopy != 0) {
                ccu::LocalAddr src;
                src.addr = inputAddr;
                for (uint32_t peerOffset = 0; peerOffset < kernelArg->rankId; ++peerOffset) {
                    src.addr += totalSliceBytes;
                }
                src.token = inputToken;
                ccu::LocalAddr dst;
                dst.addr = outputAddr;
                dst.token = outputToken;
                P2_CCU_CHK_RET(ccu::LocalCopy(dst, src, totalSliceBytes, selfDone, 1));
            }
            // Each group starts a fresh runtime cursor; peer ranks stay in channel order.
            ccu::Address cursor;
            cursor = inputAddr;
            uint32_t previousPeer = 0;
            for (uint32_t i = 0; i < kernelArg->channelCount; ++i) {
                const uint32_t peer = kernelArg->peerRanks[i];
                if (peer >= kernelArg->rankSize || peer == kernelArg->rootId || (i != 0 && peer <= previousPeer)) {
                    return CCU_E_PARA;
                }
                for (uint32_t delta = previousPeer; delta < peer; ++delta) {
                    cursor += totalSliceBytes;
                }
                previousPeer = peer;
                // Assignment into a freshly allocated Address emits a copy, not an alias.
                ccu::LocalAddr src;
                src.addr = cursor;
                src.token = inputToken;
                P2_CCU_CHK_RET(ccu::NotifyWait(kernelArg->channels[i], kCkeIndex, kOutputAddrBit | kOutputTokenBit));
                ccu::RemoteAddr dst;
                dst.addr = ccu::GetResByChannel<ccu::Variable>(kernelArg->channels[i], kOutputAddrSlot);
                dst.token = ccu::GetResByChannel<ccu::Variable>(kernelArg->channels[i], kOutputTokenSlot);
                const uint16_t peerMask = static_cast<uint16_t>(1U << i);
                P2_CCU_CHK_RET(ccu::Write(kernelArg->channels[i], dst, src, totalSliceBytes, remoteDone, peerMask));
            }
            // All Writes are issued before completion release; consume each bit exactly once.
            for (uint32_t i = 0; i < kernelArg->channelCount; ++i) {
                const uint16_t peerMask = static_cast<uint16_t>(1U << i);
                P2_CCU_CHK_RET(ccu::EventWait(remoteDone, peerMask));
                P2_CCU_CHK_RET(ccu::NotifyRecord(kernelArg->channels[i], kCkeIndex, kPostSyncBit));
            }
            // root 返回前确认自拷贝落地；无 selfCopy 的组不生成该等待（无生产者）。
            if (kernelArg->selfCopy != 0) {
                P2_CCU_CHK_RET(ccu::EventWait(selfDone, 1));
            }
        } else if (kernelArg->smallRootIdx < kernelArg->channelCount) {
            // 非 root：只在持有 root channel 的组翻译本协议；host 只 launch 该组（ctx.smallRootGroup），
            // 其余组 stageMode=4 分支为空——翻译期不能 return 错误码，否则注册直接失败。
            const ChannelHandle channel = kernelArg->channels[kernelArg->smallRootIdx];
            P2_CCU_CHK_RET(
                ccu::WriteVariableWithNotify(channel, outputAddr, kOutputAddrSlot, kCkeIndex, kOutputAddrBit));
            P2_CCU_CHK_RET(
                ccu::WriteVariableWithNotify(channel, outputToken, kOutputTokenSlot, kCkeIndex, kOutputTokenBit));
            P2_CCU_CHK_RET(ccu::NotifyWait(channel, kCkeIndex, kPostSyncBit));
        }
    }
    // E4.6 大消息纯直连星形（arg11 stageMode=3 运行期分支，复用相位1 注册；
    // 与注册期常量 phase=3 的 passive P2 完全不同）：整次调用无中继时启用，
    // 通信只沿 root-peer 边，peer 互不感知；DONE 复用 bit7（root->peer，单向，无 ACK）。
    CCU_IF(prefixStage == 3ULL)
    {
        if (kernelArg->rankId == kernelArg->rootId) {
            // L3：逐 peer READY->Write 交织（同 P1c 论证：peer 的 READY 无条件首发，无环等），
            // 慢 peer 的就绪等待与已发出 Write 的传输重叠
            ccu::Event completion;
            uint16_t completionMask = 0;
            uint32_t bit = 0;
            // L1 游标：peerRanks 注册期升序，源地址 ADD 从 O(sum(peer)) 降到 O(max(peer))；
            // 与 CcuKernelSmall 同构。乱序/自指在翻译期折叠为常量校验。
            ccu::Address cursor;
            cursor = inputAddr;
            cursor += chunkOffset;
            uint32_t previousPeer = 0;
            for (uint32_t i = 0; i < kernelArg->channelCount; ++i) {
                const uint32_t peer = kernelArg->peerRanks[i];
                if (peer >= kernelArg->rankSize || peer == kernelArg->rootId || (i != 0 && peer <= previousPeer)) {
                    return CCU_E_PARA;
                }
                for (uint32_t delta = previousPeer; delta < peer; ++delta) {
                    cursor += totalSliceBytes;
                }
                previousPeer = peer;
                ccu::LocalAddr src;
                src.addr = cursor;
                src.token = inputToken;
                P2_CCU_CHK_RET(ccu::NotifyWait(kernelArg->channels[i], kCkeIndex, kOutputAddrBit | kOutputTokenBit));
                ccu::RemoteAddr dst;
                dst.addr = ccu::GetResByChannel<ccu::Variable>(kernelArg->channels[i], kOutputAddrSlot);
                dst.addr += chunkOffset;
                dst.token = ccu::GetResByChannel<ccu::Variable>(kernelArg->channels[i], kOutputTokenSlot);
                const uint16_t peerMask = static_cast<uint16_t>(1U << bit);
                P2_CCU_CHK_RET(ccu::Write(kernelArg->channels[i], dst, src, chunkBytes, completion, peerMask));
                completionMask |= peerMask;
                ++bit;
            }
            if (kernelArg->selfCopy != 0) {
                ccu::LocalAddr src;
                src.addr = inputAddr;
                for (uint32_t peerOffset = 0; peerOffset < kernelArg->rankId; ++peerOffset) {
                    src.addr += totalSliceBytes;
                }
                src.addr += chunkOffset;
                src.token = inputToken;
                ccu::LocalAddr dst;
                dst.addr = outputAddr;
                dst.addr += chunkOffset;
                dst.token = outputToken;
                const uint16_t selfMask = static_cast<uint16_t>(1U << bit);
                P2_CCU_CHK_RET(ccu::LocalCopy(dst, src, chunkBytes, completion, selfMask));
                completionMask |= selfMask;
            }
            P2_CCU_CHK_RET(ccu::EventWait(completion, completionMask));
            // L3：数据全部落地后发 DONE，不再等 ACK。跨次安全：peer 只有消费 DONE 后才发布下一次
            // READY（程序序），root 的槽读取经 Write 数据依赖先于 Record DONE，无覆写竞争
            // （与 stageMode=4 小消息无 ACK 证明同构，见 e50/p1-protocol-proof.md）。
            for (uint32_t i = 0; i < kernelArg->channelCount; ++i) {
                P2_CCU_CHK_RET(ccu::NotifyRecord(kernelArg->channels[i], kCkeIndex, kPostSyncBit));
            }
        } else if (kernelArg->smallRootIdx < kernelArg->channelCount) {
            // 非 root：只在持有 root channel 的组翻译本协议；host 只 launch 该组（ctx.smallRootGroup），
            // 其余组 stageMode=3 分支为空——翻译期不能 return 错误码，否则注册直接失败。
            const ChannelHandle channel = kernelArg->channels[kernelArg->smallRootIdx];
            P2_CCU_CHK_RET(
                ccu::WriteVariableWithNotify(channel, outputAddr, kOutputAddrSlot, kCkeIndex, kOutputAddrBit));
            P2_CCU_CHK_RET(
                ccu::WriteVariableWithNotify(channel, outputToken, kOutputTokenSlot, kCkeIndex, kOutputTokenBit));
            P2_CCU_CHK_RET(ccu::NotifyWait(channel, kCkeIndex, kPostSyncBit));
        }
    }
    CCU_IF(prefixStage == 1ULL)
    {
        CCU_IF(relayY != 0ULL)
        {
            if (kernelArg->publishRootIdx != UINT32_MAX) {
                const auto ch = kernelArg->channels[kernelArg->publishRootIdx];
                P2_CCU_CHK_RET(
                    ccu::WriteVariableWithNotify(ch, scratchAddr, kScratchAddrSlot, kCkeIndex, kScratchAddrBit));
                P2_CCU_CHK_RET(
                    ccu::WriteVariableWithNotify(ch, scratchToken, kScratchTokenSlot, kCkeIndex, kScratchTokenBit));
                P2_CCU_CHK_RET(ccu::NotifyWait(ch, kCkeIndex, kPrefixReadyBit));
            }
            if (kernelArg->rankId == kernelArg->rootId) {
                ccu::Event ingressDone;
                if (kernelArg->relayVariant != 0) {
                    // L1 游标：ingressRemoteRank 注册期升序（helpers/remotes 升序配对），同 CcuKernelSmall 模式
                    ccu::Address ingressCursor;
                    ingressCursor = inputAddr;
                    ingressCursor += chunkOffset;
                    uint32_t previousRemote = 0;
                    // L4：逐 helper 收 scratch READY 即发该路 ingress（READY 无条件首发，无环等）；
                    // 全部 Write 发出后再逐 helper EventWait 并放 PREFIX_READY（bit8），
                    // 先落地的 helper 先沿 ACL fork 启动 P2，不再等齐全部 ingress。
                    for (uint32_t j = 0; j < kernelArg->ingressCount; ++j) {
                        const ChannelHandle helperChannel = kernelArg->channels[kernelArg->ingressHelperIdx[j]];
                        const uint32_t remote = kernelArg->ingressRemoteRank[j];
                        if (remote >= kernelArg->rankSize || (j != 0 && remote <= previousRemote)) {
                            return CCU_E_PARA;
                        }
                        for (uint32_t delta = previousRemote; delta < remote; ++delta) {
                            ingressCursor += totalSliceBytes;
                        }
                        previousRemote = remote;
                        ccu::LocalAddr src;
                        src.addr = ingressCursor;
                        src.token = inputToken;
                        P2_CCU_CHK_RET(ccu::NotifyWait(helperChannel, kCkeIndex, kScratchAddrBit | kScratchTokenBit));
                        const uint16_t ingressBit = static_cast<uint16_t>(1U << j);
                        CCU_IF(relayY != 0ULL)
                        {
                            ccu::RemoteAddr dst;
                            dst.addr = ccu::GetResByChannel<ccu::Variable>(helperChannel, kScratchAddrSlot);
                            dst.token = ccu::GetResByChannel<ccu::Variable>(helperChannel, kScratchTokenSlot);
                            P2_CCU_CHK_RET(ccu::Write(helperChannel, dst, src, relayY, ingressDone, ingressBit));
                        }
                    }
                    for (uint32_t j = 0; j < kernelArg->ingressCount; ++j) {
                        const uint16_t ingressBit = static_cast<uint16_t>(1U << j);
                        P2_CCU_CHK_RET(ccu::EventWait(ingressDone, ingressBit));
                        P2_CCU_CHK_RET(ccu::NotifyRecord(
                            kernelArg->channels[kernelArg->ingressHelperIdx[j]], kCkeIndex, kPrefixReadyBit));
                    }
                }
            }
        }
    }
    CCU_IF(prefixStage == 0ULL)
    {
        // ========== 相位1：散射相位（L2 root-star）==========
        // 通信只沿 root-peer 边：非 root 只向 root 发布 output 描述符并等 DONE；
        // root 逐 peer 收 READY 即读描述符发 Write（交织，同 P1c 无环等论证：READY 无条件首发）。
        // 跨次安全：peer 只有消费 DONE 才发布下一次 READY（程序序），root 的槽读取经 Write
        // 数据依赖先于 Record DONE，无覆写竞争。槽0/1 与 bit0/1/7 沿用旧布局，P2/prefix 不变。
        if (kernelArg->rankId == kernelArg->rootId) {
            ccu::Event completion;
            uint16_t completionMask = 0;
            uint32_t bit = 0;
            // L1 游标：peerRanks 注册期升序；served 的 relayY 只加在临时 src 上，不污染游标
            ccu::Address cursor;
            cursor = inputAddr;
            cursor += chunkOffset;
            uint32_t previousPeer = 0;
            for (uint32_t i = 0; i < kernelArg->channelCount; ++i) {
                const uint32_t peer = kernelArg->peerRanks[i];
                if (peer >= kernelArg->rankSize || peer == kernelArg->rootId || (i != 0 && peer <= previousPeer)) {
                    return CCU_E_PARA;
                }
                for (uint32_t delta = previousPeer; delta < peer; ++delta) {
                    cursor += totalSliceBytes;
                }
                previousPeer = peer;
                ccu::LocalAddr src;
                src.addr = cursor;
                src.token = inputToken;
                P2_CCU_CHK_RET(ccu::NotifyWait(kernelArg->channels[i], kCkeIndex, kOutputAddrBit | kOutputTokenBit));
                ccu::RemoteAddr dst;
                dst.addr = ccu::GetResByChannel<ccu::Variable>(kernelArg->channels[i], kOutputAddrSlot);
                dst.addr += chunkOffset;
                dst.token = ccu::GetResByChannel<ccu::Variable>(kernelArg->channels[i], kOutputTokenSlot);
                const uint16_t peerMask = static_cast<uint16_t>(1U << bit);
                if (kernelArg->relayVariant != 0 && kernelArg->servedRemote[i] != 0) {
                    // 被服务 remote：前缀 relayY 由其 helper 中继，这里只写 suffix
                    src.addr += relayY;
                    dst.addr += relayY;
                    CCU_IF(relaySuffix != 0ULL)
                    {
                        P2_CCU_CHK_RET(ccu::Write(kernelArg->channels[i], dst, src, relaySuffix, completion, peerMask));
                    }
                } else {
                    P2_CCU_CHK_RET(ccu::Write(kernelArg->channels[i], dst, src, chunkBytes, completion, peerMask));
                }
                completionMask |= peerMask;
                ++bit;
            }
            if (kernelArg->selfCopy != 0) {
                ccu::LocalAddr src;
                src.addr = inputAddr;
                for (uint32_t peerOffset = 0; peerOffset < kernelArg->rankId; ++peerOffset) {
                    src.addr += totalSliceBytes;
                }
                src.addr += chunkOffset;
                src.token = inputToken;
                ccu::LocalAddr dst;
                dst.addr = outputAddr;
                dst.addr += chunkOffset;
                dst.token = outputToken;
                const uint16_t selfMask = static_cast<uint16_t>(1U << bit);
                P2_CCU_CHK_RET(ccu::LocalCopy(dst, src, chunkBytes, completion, selfMask));
                completionMask |= selfMask;
            }
            // 组内全部 Write 与 selfCopy 落地后统一发 DONE（初版保持批量，不逐 peer 交替）
            P2_CCU_CHK_RET(ccu::EventWait(completion, completionMask));
            for (uint32_t i = 0; i < kernelArg->channelCount; ++i) {
                P2_CCU_CHK_RET(ccu::NotifyRecord(kernelArg->channels[i], kCkeIndex, kPostSyncBit));
            }
        } else if (kernelArg->smallRootIdx < kernelArg->channelCount) {
            // 非 root：只在持有 root channel 的组翻译本协议（host 只 launch 该组与 prefix 组），
            // 其余组 stageMode=0 分支为空——翻译期不能 return 错误码，否则注册直接失败。
            const ChannelHandle channel = kernelArg->channels[kernelArg->smallRootIdx];
            P2_CCU_CHK_RET(
                ccu::WriteVariableWithNotify(channel, outputAddr, kOutputAddrSlot, kCkeIndex, kOutputAddrBit));
            P2_CCU_CHK_RET(
                ccu::WriteVariableWithNotify(channel, outputToken, kOutputTokenSlot, kCkeIndex, kOutputTokenBit));
            P2_CCU_CHK_RET(ccu::NotifyWait(channel, kCkeIndex, kPostSyncBit));
        }
    }
    return CCU_SUCCESS;
}

} // namespace ops_hccl
