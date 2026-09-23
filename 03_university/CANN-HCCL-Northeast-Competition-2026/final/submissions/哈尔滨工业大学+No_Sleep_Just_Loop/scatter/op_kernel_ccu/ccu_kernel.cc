/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <vector>

#include <hcomm/hcomm_primitives.h>

#include "ccu_kernel.h"

#define CCU_RETURN_IF_ERROR(call)         \
    do {                                  \
        const CcuResult _ccuRet = (call); \
        if (_ccuRet != CCU_SUCCESS) {     \
            return _ccuRet;               \
        }                                 \
    } while (0)

namespace ops_hccl {
namespace ccu = ::AscendC::ccu;
namespace {

// ---- 三条路径共用的 channel 资源槽与 CKE 索引 ----
constexpr uint32_t OUTPUT_XN_ID = 0; // 对端 recvBuf 地址
constexpr uint32_t TOKEN_XN_ID = 1;  // 对端 recvBuf token
constexpr uint32_t CKE_IDX_0 = 0;

// 路径 1 / 2 的事件位
constexpr uint32_t POST_SYNC_ID = 3;

// 路径 3 的事件位(必须互不重叠, 且不与上面复用):
//   0..3 给 PreSync 的 4 个变量槽, 4 给代理数据就绪, 5 给后同步
constexpr uint32_t SCRATCH_XN_ID = 2;       // 对端 CCL buffer 地址(代理中转用)
constexpr uint32_t SCRATCH_TOKEN_XN_ID = 3; // 对端 CCL buffer token
constexpr uint32_t PROXY_DATA_READY_ID = 4;  // 分块流水时 = 第 0 块就绪
constexpr uint32_t PROXY_POST_SYNC_ID = 5;
constexpr uint32_t PROXY_CHUNK1_READY_ID = 6; // 分块流水: 第 1 块就绪

// 路径 3 分块流水的 taskArg 下标(接在原有 9 项之后)
constexpr uint32_t PROXY_ARG_STAGE_ID = 9;
constexpr uint32_t PROXY_ARG_CHUNK0_BYTES = 10;
constexpr uint32_t PROXY_ARG_CHUNK1_BYTES = 11;
constexpr uint32_t PROXY_TASK_ARG_COUNT = 12;

// ===========================================================================
// 路径 2 (Wave) 的辅助: 提交一个 wave 内的全部对端传输。
// 调用方可用不同 offset 调两次, channel 与 peer 映射复用。
// ===========================================================================
CcuResult IssueScatterWave(const CcuKernelArgWave &kernelArg,
    const std::vector<ccu::Variable> &peerOutput, const std::vector<ccu::Variable> &peerToken,
    ccu::Variable &input, ccu::Variable &inputToken, ccu::Variable &selfOutput, ccu::Variable &selfToken,
    ccu::Variable &sliceBytes, ccu::Variable &waveOffset, ccu::Variable &waveBytes,
    ccu::Event &event, uint16_t &doneMask, bool waitReady)
{
    // A0b: root 的 Mesh / Clos 出口都是共享瓶颈通道。原来前同步先把整组等齐, 出口
    // 在等最慢那个对端的期间是空转的。把等待折进第一个 wave 的通道循环, 就变成
    // "等一条发一条" —— 出口从第一条通道就绪起就开始忙。
    //
    // 注意: 只能在第一个 wave 里等。本函数被调用两次(wave0 / wave1), 两次共用同一组
    // channels 和同一组 peerOutput/peerToken。第二个 wave 若再等一次, 就是同一通道
    // 同一 CKE 上重复消费同一组位 —— 那正是会让整个算子挂死的模式。
    const uint16_t readyBits = static_cast<uint16_t>((1U << OUTPUT_XN_ID) | (1U << TOKEN_XN_ID));
    for (uint32_t i = 0; i < kernelArg.channelCount; ++i) {
        if (waitReady) {
            CCU_RETURN_IF_ERROR(ccu::NotifyWait(kernelArg.channels[i], CKE_IDX_0, readyBits));
        }
        ccu::Variable peerBlockOffset;
        peerBlockOffset = 0;
        for (uint32_t rank = 0; rank < kernelArg.peerRanks[i]; ++rank) {
            peerBlockOffset += sliceBytes;
        }
        peerBlockOffset += waveOffset;

        ccu::LocalAddr src;
        src.addr = input;
        src.addr += peerBlockOffset;
        src.token = inputToken;

        ccu::RemoteAddr dst;
        dst.addr = peerOutput[i];
        dst.addr += waveOffset;
        dst.token = peerToken[i];

        const uint16_t mask = static_cast<uint16_t>(1U << i);
        CCU_RETURN_IF_ERROR(ccu::Write(kernelArg.channels[i], dst, src, waveBytes, event, mask));
        doneMask |= mask;
    }

    if (kernelArg.doSelfCopy) {
        ccu::Variable selfOffset;
        selfOffset = 0;
        for (uint32_t rank = 0; rank < kernelArg.root; ++rank) {
            selfOffset += sliceBytes;
        }
        selfOffset += waveOffset;

        ccu::LocalAddr localSrc;
        localSrc.addr = input;
        localSrc.addr += selfOffset;
        localSrc.token = inputToken;

        ccu::LocalAddr localDst;
        localDst.addr = selfOutput;
        localDst.addr += waveOffset;
        localDst.token = selfToken;

        const uint16_t selfMask = static_cast<uint16_t>(1U << kernelArg.channelCount);
        CCU_RETURN_IF_ERROR(ccu::LocalCopy(localDst, localSrc, waveBytes, event, selfMask));
        doneMask |= selfMask;
    }
    return CcuResult::CCU_SUCCESS;
}

// ===========================================================================
// 路径 4 (Wave-Star) 的辅助: 对单个 Peer 提交一个 wave。
// 调用方在该 Peer READY 之后立刻调用, 形成 Ready->Write 流水。
// 块基址用注册期固化的 peerOffsets, 不再循环累加 sliceBytes。
// ===========================================================================
CcuResult IssueWaveStarPeerWrite(const CcuKernelArgWaveStar &kernelArg, uint32_t i,
    const ccu::Variable &peerOutput, const ccu::Variable &peerToken,
    ccu::Variable &input, ccu::Variable &inputToken,
    ccu::Variable &waveOffset, ccu::Variable &waveBytes, ccu::Event &event, uint16_t &doneMask)
{
    ccu::Variable peerBlockOffset;
    peerBlockOffset = kernelArg.peerOffsets[i];
    peerBlockOffset += waveOffset;

    ccu::LocalAddr src;
    src.addr = input;
    src.addr += peerBlockOffset;
    src.token = inputToken;

    ccu::RemoteAddr dst;
    dst.addr = peerOutput;
    dst.addr += waveOffset;
    dst.token = peerToken;

    const uint16_t mask = static_cast<uint16_t>(1U << i);
    CCU_RETURN_IF_ERROR(ccu::Write(kernelArg.channels[i], dst, src, waveBytes, event, mask));
    doneMask |= mask;
    return CcuResult::CCU_SUCCESS;
}

CcuResult IssueWaveStarSelfCopy(const CcuKernelArgWaveStar &kernelArg,
    ccu::Variable &input, ccu::Variable &inputToken, ccu::Variable &selfOutput, ccu::Variable &selfToken,
    ccu::Variable &waveOffset, ccu::Variable &waveBytes,
    ccu::Event &event, uint16_t &doneMask)
{
    ccu::Variable selfOffset;
    selfOffset = kernelArg.selfOffset;
    selfOffset += waveOffset;

    ccu::LocalAddr localSrc;
    localSrc.addr = input;
    localSrc.addr += selfOffset;
    localSrc.token = inputToken;

    ccu::LocalAddr localDst;
    localDst.addr = selfOutput;
    localDst.addr += waveOffset;
    localDst.token = selfToken;

    const uint16_t selfMask = static_cast<uint16_t>(1U << kernelArg.channelCount);
    CCU_RETURN_IF_ERROR(ccu::LocalCopy(localDst, localSrc, waveBytes, event, selfMask));
    doneMask |= selfMask;
    return CcuResult::CCU_SUCCESS;
}

// ===========================================================================
// 路径 3 (Proxy) 的辅助
// ===========================================================================
bool IsInList(const uint32_t *list, uint32_t count, uint32_t value, uint32_t &index)
{
    for (uint32_t i = 0; i < count; ++i) {
        if (list[i] == value) {
            index = i;
            return true;
        }
    }
    return false;
}

uint32_t FindPeerChannel(const CcuKernelArgProxy *k, uint32_t rank)
{
    for (uint32_t i = 0; i < k->channelCount; ++i) {
        if (k->peerRanks[i] == rank) {
            return i;
        }
    }
    return INVALID_VALUE_RANKID;
}

void MakeBlockOffset(const ccu::Variable &sliceBytes, ccu::Variable &offset, uint32_t peer)
{
    offset = 0;
    for (uint32_t r = 0; r < peer; ++r) {
        offset += sliceBytes;
    }
}

} // namespace

// ===========================================================================
// 路径 1: Root-Star (512KB)
//   Root  : 自留块拷贝先提交(与 Peer 的地址发布重叠) -> 等各 Peer 发布 -> 并行写各
//           Peer 的块 -> EventWait -> 单向发 DONE(无需 Peer ACK)。
//   Peer  : 只连 Root 一条 channel, 发布自己的 recvBuf 地址/token, 等 Root 的 DONE。
// ===========================================================================
CcuResult CcuScatterRootStarKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgRootStar *>(arg);
    if (kernelArg == nullptr || kernelArg->channelCount == 0) {
        return CcuResult::CCU_E_INTERNAL;
    }
    const uint32_t channelCount = kernelArg->channelCount;

    // ---------- Root Fast Path ----------
    // Root 只等待各 Peer 发布 recvBuf/token，随后向各 Peer 写数据并单向发送 DONE。
    if (kernelArg->rankId == kernelArg->root) {
        ccu::Variable input;
        ccu::Variable inputToken;
        ccu::Variable selfOutput;
        ccu::Variable selfToken;
        ccu::Variable sliceBytes;
        sliceBytes = kernelArg->sliceBytes;
        CCU_RETURN_IF_ERROR(ccu::LoadArg(input, 0));
        CCU_RETURN_IF_ERROR(ccu::LoadArg(inputToken, 1));
        if (kernelArg->doSelfCopy) {
            CCU_RETURN_IF_ERROR(ccu::LoadArg(selfOutput, 2));
            CCU_RETURN_IF_ERROR(ccu::LoadArg(selfToken, 3));
        }

        // 只有 Root 需要读取 channel 中 Peer 发布的接收地址和 token。
        std::vector<ccu::Variable> peerOutput(channelCount);
        std::vector<ccu::Variable> peerToken(channelCount);
        for (uint32_t i = 0; i < channelCount; ++i) {
            peerOutput[i] = ccu::GetResByChannel<ccu::Variable>(kernelArg->channels[i], OUTPUT_XN_ID);
            peerToken[i] = ccu::GetResByChannel<ccu::Variable>(kernelArg->channels[i], TOKEN_XN_ID);
        }

        ccu::Event event;
        uint16_t doneMask = 0;

        // Submit the root-local slice before the READY handshake so it can overlap peer setup.
        if (kernelArg->doSelfCopy) {
            ccu::Variable selfOffset;
            selfOffset = kernelArg->selfOffset;
            ccu::LocalAddr localSrc;
            localSrc.addr = input;
            localSrc.addr += selfOffset;
            localSrc.token = inputToken;

            ccu::LocalAddr localDst;
            localDst.addr = selfOutput;
            localDst.token = selfToken;

            const uint16_t selfMask = static_cast<uint16_t>(1U << channelCount);
            CCU_RETURN_IF_ERROR(ccu::LocalCopy(localDst, localSrc, sliceBytes, event, selfMask));
            doneMask |= selfMask;
        }

        const uint16_t readyBits = static_cast<uint16_t>((1U << OUTPUT_XN_ID) | (1U << TOKEN_XN_ID));
        // 每个 Peer 就绪后立即提交对应 Write，Ready->Write 形成流水；
        // 所有 Write 共用一个 event，循环结束后只做一次 EventWait。
        for (uint32_t i = 0; i < channelCount; ++i) {
            CCU_RETURN_IF_ERROR(ccu::NotifyWait(kernelArg->channels[i], CKE_IDX_0, readyBits));

            ccu::Variable peerBlockOffset;
            peerBlockOffset = kernelArg->peerOffsets[i];
            ccu::LocalAddr src;
            src.addr = input;
            src.addr += peerBlockOffset;
            src.token = inputToken;

            ccu::RemoteAddr dst;
            dst.addr = peerOutput[i];
            dst.token = peerToken[i];

            const uint16_t mask = static_cast<uint16_t>(1U << i);
            CCU_RETURN_IF_ERROR(ccu::Write(kernelArg->channels[i], dst, src, sliceBytes, event, mask));
            doneMask |= mask;
        }

        // DONE 必须排在所有数据写完成之后，Peer 收到后即可返回，无需 ACK。
        CCU_RETURN_IF_ERROR(ccu::EventWait(event, doneMask));
        for (uint32_t i = 0; i < channelCount; ++i) {
            CCU_RETURN_IF_ERROR(
                ccu::NotifyRecord(kernelArg->channels[i], CKE_IDX_0, static_cast<uint16_t>(1U << POST_SYNC_ID)));
        }
    } else {
        // ---------- Peer Fast Path ----------
        // Root-Star 中每个 Peer 只有一条到 Root 的 Channel。
        if (channelCount != 1 || kernelArg->peerRanks[0] != kernelArg->root) {
            return CcuResult::CCU_E_INTERNAL;
        }

        ccu::Variable selfOutput;
        ccu::Variable selfToken;
        CCU_RETURN_IF_ERROR(ccu::LoadArg(selfOutput, 0));
        CCU_RETURN_IF_ERROR(ccu::LoadArg(selfToken, 1));

        // Peer 只向 Root 发布自己的接收地址/token，不交换给其他 Peer。
        CCU_RETURN_IF_ERROR(ccu::WriteVariableWithNotify(kernelArg->channels[0], selfOutput, OUTPUT_XN_ID,
            CKE_IDX_0, static_cast<uint16_t>(1U << OUTPUT_XN_ID)));
        CCU_RETURN_IF_ERROR(ccu::WriteVariableWithNotify(kernelArg->channels[0], selfToken, TOKEN_XN_ID,
            CKE_IDX_0, static_cast<uint16_t>(1U << TOKEN_XN_ID)));

        // Root 在数据 EventWait 后发送单向 DONE；Peer 无需向 Root 回 ACK。
        CCU_RETURN_IF_ERROR(
            ccu::NotifyWait(kernelArg->channels[0], CKE_IDX_0, static_cast<uint16_t>(1U << POST_SYNC_ID)));
    }

    return CcuResult::CCU_SUCCESS;
}

// ===========================================================================
// 路径 1 Push SMALL: CcuScatterRootStarKernel 的协议副本。
// E1 隔离车 + E1b H0: 注册期 offset==0 时不发 AssignImm / Address+=。
// 只给 sendBytes < 1MiB 注册; LARGE 继续走上面的原函数。
// ===========================================================================
CcuResult CcuScatterRootStarSmallKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgRootStar *>(arg);
    if (kernelArg == nullptr || kernelArg->channelCount == 0) {
        return CcuResult::CCU_E_INTERNAL;
    }
    const uint32_t channelCount = kernelArg->channelCount;

    // ---------- Root Fast Path ----------
    // Root 只等待各 Peer 发布 recvBuf/token，随后向各 Peer 写数据并单向发送 DONE。
    if (kernelArg->rankId == kernelArg->root) {
        ccu::Variable input;
        ccu::Variable inputToken;
        ccu::Variable selfOutput;
        ccu::Variable selfToken;
        ccu::Variable sliceBytes;
        sliceBytes = kernelArg->sliceBytes;
        CCU_RETURN_IF_ERROR(ccu::LoadArg(input, 0));
        CCU_RETURN_IF_ERROR(ccu::LoadArg(inputToken, 1));
        if (kernelArg->doSelfCopy) {
            CCU_RETURN_IF_ERROR(ccu::LoadArg(selfOutput, 2));
            CCU_RETURN_IF_ERROR(ccu::LoadArg(selfToken, 3));
        }

        // 只有 Root 需要读取 channel 中 Peer 发布的接收地址和 token。
        std::vector<ccu::Variable> peerOutput(channelCount);
        std::vector<ccu::Variable> peerToken(channelCount);
        for (uint32_t i = 0; i < channelCount; ++i) {
            peerOutput[i] = ccu::GetResByChannel<ccu::Variable>(kernelArg->channels[i], OUTPUT_XN_ID);
            peerToken[i] = ccu::GetResByChannel<ccu::Variable>(kernelArg->channels[i], TOKEN_XN_ID);
        }

        ccu::Event event;
        uint16_t doneMask = 0;

        // E4: skipDeviceSelfCopy 时自留块已在 Host 拷完, 不再发 LocalCopy, 也不把 selfMask
        // 记进 EventWait(没提交的 bit 会挂死)。仍 LoadArg 2/3, 保持 4 项 ABI。
        // E1b: 若仍走 device LocalCopy, offset==0 不发赋常量/加零。
        if (kernelArg->doSelfCopy && !kernelArg->skipDeviceSelfCopy) {
            ccu::LocalAddr localSrc;
            localSrc.addr = input;
            if (kernelArg->selfOffset != 0) {
                ccu::Variable selfOffset;
                selfOffset = kernelArg->selfOffset;
                localSrc.addr += selfOffset;
            }
            localSrc.token = inputToken;

            ccu::LocalAddr localDst;
            localDst.addr = selfOutput;
            localDst.token = selfToken;

            const uint16_t selfMask = static_cast<uint16_t>(1U << channelCount);
            CCU_RETURN_IF_ERROR(ccu::LocalCopy(localDst, localSrc, sliceBytes, event, selfMask));
            doneMask |= selfMask;
        }

        const uint16_t readyBits = static_cast<uint16_t>((1U << OUTPUT_XN_ID) | (1U << TOKEN_XN_ID));
        // 每个 Peer 就绪后立即提交对应 Write，Ready->Write 形成流水；
        // 所有 Write 共用一个 event，循环结束后只做一次 EventWait。
        for (uint32_t i = 0; i < channelCount; ++i) {
            CCU_RETURN_IF_ERROR(ccu::NotifyWait(kernelArg->channels[i], CKE_IDX_0, readyBits));

            ccu::LocalAddr src;
            src.addr = input;
            if (kernelArg->peerOffsets[i] != 0) {
                ccu::Variable peerBlockOffset;
                peerBlockOffset = kernelArg->peerOffsets[i];
                src.addr += peerBlockOffset;
            }
            src.token = inputToken;

            ccu::RemoteAddr dst;
            dst.addr = peerOutput[i];
            dst.token = peerToken[i];

            const uint16_t mask = static_cast<uint16_t>(1U << i);
            CCU_RETURN_IF_ERROR(ccu::Write(kernelArg->channels[i], dst, src, sliceBytes, event, mask));
            doneMask |= mask;
        }

        // DONE 必须排在所有数据写完成之后，Peer 收到后即可返回，无需 ACK。
        CCU_RETURN_IF_ERROR(ccu::EventWait(event, doneMask));
        for (uint32_t i = 0; i < channelCount; ++i) {
            CCU_RETURN_IF_ERROR(
                ccu::NotifyRecord(kernelArg->channels[i], CKE_IDX_0, static_cast<uint16_t>(1U << POST_SYNC_ID)));
        }
    } else {
        // ---------- Peer Fast Path ----------
        // Root-Star 中每个 Peer 只有一条到 Root 的 Channel。
        if (channelCount != 1 || kernelArg->peerRanks[0] != kernelArg->root) {
            return CcuResult::CCU_E_INTERNAL;
        }

        ccu::Variable selfOutput;
        ccu::Variable selfToken;
        CCU_RETURN_IF_ERROR(ccu::LoadArg(selfOutput, 0));
        CCU_RETURN_IF_ERROR(ccu::LoadArg(selfToken, 1));

        // Peer 只向 Root 发布自己的接收地址/token，不交换给其他 Peer。
        CCU_RETURN_IF_ERROR(ccu::WriteVariableWithNotify(kernelArg->channels[0], selfOutput, OUTPUT_XN_ID,
            CKE_IDX_0, static_cast<uint16_t>(1U << OUTPUT_XN_ID)));
        CCU_RETURN_IF_ERROR(ccu::WriteVariableWithNotify(kernelArg->channels[0], selfToken, TOKEN_XN_ID,
            CKE_IDX_0, static_cast<uint16_t>(1U << TOKEN_XN_ID)));

        // Root 在数据 EventWait 后发送单向 DONE；Peer 无需向 Root 回 ACK。
        CCU_RETURN_IF_ERROR(
            ccu::NotifyWait(kernelArg->channels[0], CKE_IDX_0, static_cast<uint16_t>(1U << POST_SYNC_ID)));
    }

    return CcuResult::CCU_SUCCESS;
}

// ===========================================================================
// F-ClosBurst (4x3 SMALL Root Clos only)
// 语义: 不准把 9 次 NotifyWait 挪到 9 次 Write 之后 (Write 的 RemoteAddr 来自
// GetResByChannel, 没有晚绑定保证; Wait-after-Write 会用空 dest 发 DMA)。
// 可执行的削弱只有一种: 用第一条 Clos READY 当作「这一群 Peer 已经在跑 Peer 核」
// 的门闩, 其余 8 次 Wait 从 Write 前面拿掉, 让 9 次 Write 连发。
// Host 保证: Root + channelCount>=2 + doSelfCopy=false (4x3 Clos 实为 9)。
// Peer 支路与旧 Small 逐条原语相同 (门控走不到也保持可审)。
// ===========================================================================
CcuResult CcuScatterRootStarSmallClosBurstKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgRootStar *>(arg);
    if (kernelArg == nullptr || kernelArg->channelCount == 0) {
        return CcuResult::CCU_E_INTERNAL;
    }
    const uint32_t channelCount = kernelArg->channelCount;

    // ---------- Root Fast Path ----------
    if (kernelArg->rankId == kernelArg->root) {
        if (kernelArg->doSelfCopy || channelCount < 2) {
            return CcuResult::CCU_E_INTERNAL;
        }

        ccu::Variable input;
        ccu::Variable inputToken;
        ccu::Variable selfOutput;
        ccu::Variable selfToken;
        ccu::Variable sliceBytes;
        sliceBytes = kernelArg->sliceBytes;
        CCU_RETURN_IF_ERROR(ccu::LoadArg(input, 0));
        CCU_RETURN_IF_ERROR(ccu::LoadArg(inputToken, 1));
        if (kernelArg->doSelfCopy) {
            CCU_RETURN_IF_ERROR(ccu::LoadArg(selfOutput, 2));
            CCU_RETURN_IF_ERROR(ccu::LoadArg(selfToken, 3));
        }

        std::vector<ccu::Variable> peerOutput(channelCount);
        std::vector<ccu::Variable> peerToken(channelCount);
        for (uint32_t i = 0; i < channelCount; ++i) {
            peerOutput[i] = ccu::GetResByChannel<ccu::Variable>(kernelArg->channels[i], OUTPUT_XN_ID);
            peerToken[i] = ccu::GetResByChannel<ccu::Variable>(kernelArg->channels[i], TOKEN_XN_ID);
        }

        ccu::Event event;
        uint16_t doneMask = 0;

        if (kernelArg->doSelfCopy && !kernelArg->skipDeviceSelfCopy) {
            ccu::LocalAddr localSrc;
            localSrc.addr = input;
            if (kernelArg->selfOffset != 0) {
                ccu::Variable selfOffset;
                selfOffset = kernelArg->selfOffset;
                localSrc.addr += selfOffset;
            }
            localSrc.token = inputToken;

            ccu::LocalAddr localDst;
            localDst.addr = selfOutput;
            localDst.token = selfToken;

            const uint16_t selfMask = static_cast<uint16_t>(1U << channelCount);
            CCU_RETURN_IF_ERROR(ccu::LocalCopy(localDst, localSrc, sliceBytes, event, selfMask));
            doneMask |= selfMask;
        }

        const uint16_t readyBits = static_cast<uint16_t>((1U << OUTPUT_XN_ID) | (1U << TOKEN_XN_ID));
        // F-ClosBurst 门闩。默认只 Wait ch[0]；HVM 在 root=5 上错数后按方案 §3.6
        // 收紧为 first+last，其余通道仍连发 Write。kernel 仍以 WAIT 开头。
        CCU_RETURN_IF_ERROR(ccu::NotifyWait(kernelArg->channels[0], CKE_IDX_0, readyBits));
        CCU_RETURN_IF_ERROR(ccu::NotifyWait(kernelArg->channels[channelCount - 1], CKE_IDX_0, readyBits));
        for (uint32_t i = 0; i < channelCount; ++i) {
            ccu::LocalAddr src;
            src.addr = input;
            if (kernelArg->peerOffsets[i] != 0) {
                ccu::Variable peerBlockOffset;
                peerBlockOffset = kernelArg->peerOffsets[i];
                src.addr += peerBlockOffset;
            }
            src.token = inputToken;

            ccu::RemoteAddr dst;
            dst.addr = peerOutput[i];
            dst.token = peerToken[i];

            const uint16_t mask = static_cast<uint16_t>(1U << i);
            CCU_RETURN_IF_ERROR(ccu::Write(kernelArg->channels[i], dst, src, sliceBytes, event, mask));
            doneMask |= mask;
        }

        CCU_RETURN_IF_ERROR(ccu::EventWait(event, doneMask));
        for (uint32_t i = 0; i < channelCount; ++i) {
            CCU_RETURN_IF_ERROR(
                ccu::NotifyRecord(kernelArg->channels[i], CKE_IDX_0, static_cast<uint16_t>(1U << POST_SYNC_ID)));
        }
    } else {
        // ---------- Peer Fast Path (与旧 Small 逐条原语相同) ----------
        if (channelCount != 1 || kernelArg->peerRanks[0] != kernelArg->root) {
            return CcuResult::CCU_E_INTERNAL;
        }

        ccu::Variable selfOutput;
        ccu::Variable selfToken;
        CCU_RETURN_IF_ERROR(ccu::LoadArg(selfOutput, 0));
        CCU_RETURN_IF_ERROR(ccu::LoadArg(selfToken, 1));

        CCU_RETURN_IF_ERROR(ccu::WriteVariableWithNotify(kernelArg->channels[0], selfOutput, OUTPUT_XN_ID,
            CKE_IDX_0, static_cast<uint16_t>(1U << OUTPUT_XN_ID)));
        CCU_RETURN_IF_ERROR(ccu::WriteVariableWithNotify(kernelArg->channels[0], selfToken, TOKEN_XN_ID,
            CKE_IDX_0, static_cast<uint16_t>(1U << TOKEN_XN_ID)));

        CCU_RETURN_IF_ERROR(
            ccu::NotifyWait(kernelArg->channels[0], CKE_IDX_0, static_cast<uint16_t>(1U << POST_SYNC_ID)));
    }

    return CcuResult::CCU_SUCCESS;
}

// ===========================================================================
// 路径 2: Wave (400M+4B)
//   把一个 Slice 拆成两段连续 wave, 两段都先提交再等待; 源与目的区间互不重叠,
//   链路在 wave0 排空时仍可继续跑。chunk1Bytes 为 0 时退化为单 wave。
// ===========================================================================
CcuResult CcuScatterWaveKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgWave *>(arg);
    if (kernelArg == nullptr || kernelArg->channelCount == 0) {
        return CcuResult::CCU_E_INTERNAL;
    }
    const uint32_t channelCount = kernelArg->channelCount;

    // ---------- 1. 取资源: 每个对端的 recvBuf 地址槽与 token 槽 ----------
    std::vector<ccu::Variable> peerOutput(channelCount);
    std::vector<ccu::Variable> peerToken(channelCount);
    for (uint32_t i = 0; i < channelCount; ++i) {
        peerOutput[i] = ccu::GetResByChannel<ccu::Variable>(kernelArg->channels[i], OUTPUT_XN_ID);
        peerToken[i] = ccu::GetResByChannel<ccu::Variable>(kernelArg->channels[i], TOKEN_XN_ID);
    }

    // ---------- 2. 加载参数(顺序须与 host taskArgs 完全一致) ----------
    ccu::Variable input;
    ccu::Variable inputToken;
    ccu::Variable selfOutput;
    ccu::Variable selfToken;
    ccu::Variable sliceBytes;
    ccu::Variable chunk0Bytes;
    ccu::Variable chunk1Bytes;
    CCU_RETURN_IF_ERROR(ccu::LoadArg(input, SCATTER_ARG_INPUT));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(inputToken, SCATTER_ARG_INPUT_TOKEN));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(selfOutput, SCATTER_ARG_OUTPUT));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(selfToken, SCATTER_ARG_OUTPUT_TOKEN));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(sliceBytes, SCATTER_ARG_SLICE_BYTES));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(chunk0Bytes, SCATTER_ARG_CHUNK0_BYTES));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(chunk1Bytes, SCATTER_ARG_CHUNK1_BYTES));

    // ---------- 3. 前同步: 把本 rank 的 recvBuf 地址与 token 推给组内所有对端 ----------
    for (uint32_t i = 0; i < channelCount; ++i) {
        CCU_RETURN_IF_ERROR(ccu::WriteVariableWithNotify(kernelArg->channels[i], selfOutput, OUTPUT_XN_ID,
            CKE_IDX_0, static_cast<uint16_t>(1U << OUTPUT_XN_ID)));
        CCU_RETURN_IF_ERROR(ccu::WriteVariableWithNotify(kernelArg->channels[i], selfToken, TOKEN_XN_ID,
            CKE_IDX_0, static_cast<uint16_t>(1U << TOKEN_XN_ID)));
    }
    const uint16_t readyBits = static_cast<uint16_t>((1U << OUTPUT_XN_ID) | (1U << TOKEN_XN_ID));
    // root 的等待折进数据面的第一个 wave(见 IssueScatterWave 的 waitReady) ——
    // root 的出口是共享瓶颈, 不该空转着等整组就绪。
    // 非 root 没有数据面, 仍在这里等齐(它不写数据, 这一步只为保持任务图自洽)。
    // 发布循环不能省: 对端在等 root 的 recvBuf 地址/token。
    if (kernelArg->rankId != kernelArg->root) {
        for (uint32_t i = 0; i < channelCount; ++i) {
            CCU_RETURN_IF_ERROR(ccu::NotifyWait(kernelArg->channels[i], CKE_IDX_0, readyBits));
        }
    }

    // ---------- 4. 数据面: 只有 root 发数据 ----------
    // rankId / root 均来自 kernelArg(注册期常量), 用普通 if 即可。
    if (kernelArg->rankId == kernelArg->root) {
        ccu::Variable wave0Offset;
        wave0Offset = 0;
        ccu::Variable wave1Offset;
        wave1Offset = chunk0Bytes;
        ccu::Event wave0Event;
        ccu::Event wave1Event;
        uint16_t wave0Mask = 0;
        uint16_t wave1Mask = 0;

        // Submit both waves before waiting.  Their source and destination
        // ranges are disjoint, so the links can keep moving while wave 0
        // drains.  The post-sync remains after both event waits.
        CCU_RETURN_IF_ERROR(IssueScatterWave(*kernelArg, peerOutput, peerToken,
            input, inputToken, selfOutput, selfToken, sliceBytes, wave0Offset, chunk0Bytes,
            wave0Event, wave0Mask, true));
        CCU_IF(chunk1Bytes != 0)
        {
            CCU_RETURN_IF_ERROR(IssueScatterWave(*kernelArg, peerOutput, peerToken,
                input, inputToken, selfOutput, selfToken, sliceBytes, wave1Offset, chunk1Bytes,
                wave1Event, wave1Mask, false));
        }
        CCU_RETURN_IF_ERROR(ccu::EventWait(wave0Event, wave0Mask));
        CCU_IF(chunk1Bytes != 0)
        {
            CCU_RETURN_IF_ERROR(ccu::EventWait(wave1Event, wave1Mask));
        }
    }
    // 非 root 不产生数据面任务: 数据由 root 直接写进本 rank 的 recvBuf。

    // ---------- 5. 后同步: 组内栅栏, 确保所有 rank 数据传输完成 ----------
    for (uint32_t i = 0; i < channelCount; ++i) {
        CCU_RETURN_IF_ERROR(
            ccu::NotifyRecord(kernelArg->channels[i], CKE_IDX_0, static_cast<uint16_t>(1U << POST_SYNC_ID)));
    }
    for (uint32_t i = 0; i < channelCount; ++i) {
        CCU_RETURN_IF_ERROR(
            ccu::NotifyWait(kernelArg->channels[i], CKE_IDX_0, static_cast<uint16_t>(1U << POST_SYNC_ID)));
    }

    return CcuResult::CCU_SUCCESS;
}

// ===========================================================================
// 路径 4: Wave-Star Root
//   数据面: 只有 Root 发数据。
//   Single-Wave fast path: 注册期 singleWave=true (sliceBytes <= MAX_DATA_SIZE)
//     时只发一整块, 不建 wave1, 不用 CCU_IF(chunk1)。
//   Two-wave: sliceBytes > MAX_DATA_SIZE 时保留 chunk0/chunk1。
//   控制面: 只 wait 本组 Peer; 每个 Peer READY 后立刻 Write (Ready->Write)。
//   完成后 EventWait, 再单向 DONE; Peer 不回 ACK。
//   块偏移使用注册期固化的 peerOffsets / selfOffset / sliceBytes。
// ===========================================================================
CcuResult CcuScatterWaveStarRootKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgWaveStar *>(arg);
    if (kernelArg == nullptr || kernelArg->channelCount == 0) {
        return CcuResult::CCU_E_INTERNAL;
    }
    const uint32_t channelCount = kernelArg->channelCount;

    // Host Root 始终下发 WAVESTAR_KERNEL_ARG_COUNT(7) 个 taskArgs。
    // 无论 Mesh/Clos、有无 SelfCopy、Single-Wave/Two-Wave, LoadArg 数量必须与之一致。
    ccu::Variable input;
    ccu::Variable inputToken;
    ccu::Variable selfOutput;
    ccu::Variable selfToken;
    ccu::Variable sliceBytesArg;
    ccu::Variable chunk0Bytes;
    ccu::Variable chunk1Bytes;
    CCU_RETURN_IF_ERROR(ccu::LoadArg(input, WAVESTAR_ARG_INPUT));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(inputToken, WAVESTAR_ARG_INPUT_TOKEN));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(selfOutput, WAVESTAR_ARG_OUTPUT));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(selfToken, WAVESTAR_ARG_OUTPUT_TOKEN));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(sliceBytesArg, WAVESTAR_ARG_SLICE_BYTES));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(chunk0Bytes, WAVESTAR_ARG_CHUNK0_BYTES));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(chunk1Bytes, WAVESTAR_ARG_CHUNK1_BYTES));
    (void)sliceBytesArg;

    std::vector<ccu::Variable> peerOutput(channelCount);
    std::vector<ccu::Variable> peerToken(channelCount);
    for (uint32_t i = 0; i < channelCount; ++i) {
        peerOutput[i] = ccu::GetResByChannel<ccu::Variable>(kernelArg->channels[i], OUTPUT_XN_ID);
        peerToken[i] = ccu::GetResByChannel<ccu::Variable>(kernelArg->channels[i], TOKEN_XN_ID);
    }

    const uint16_t readyBits = static_cast<uint16_t>((1U << OUTPUT_XN_ID) | (1U << TOKEN_XN_ID));

    if (kernelArg->singleWave) {
        ccu::Variable sliceBytes;
        sliceBytes = kernelArg->sliceBytes;

        ccu::Event event;
        uint16_t doneMask = 0;

        if (kernelArg->doSelfCopy) {
            ccu::Variable selfOffset;
            selfOffset = kernelArg->selfOffset;
            ccu::LocalAddr localSrc;
            localSrc.addr = input;
            localSrc.addr += selfOffset;
            localSrc.token = inputToken;

            ccu::LocalAddr localDst;
            localDst.addr = selfOutput;
            localDst.token = selfToken;

            const uint16_t selfMask = static_cast<uint16_t>(1U << channelCount);
            CCU_RETURN_IF_ERROR(ccu::LocalCopy(localDst, localSrc, sliceBytes, event, selfMask));
            doneMask |= selfMask;
        }

        for (uint32_t i = 0; i < channelCount; ++i) {
            CCU_RETURN_IF_ERROR(ccu::NotifyWait(kernelArg->channels[i], CKE_IDX_0, readyBits));

            ccu::Variable peerBlockOffset;
            peerBlockOffset = kernelArg->peerOffsets[i];
            ccu::LocalAddr src;
            src.addr = input;
            src.addr += peerBlockOffset;
            src.token = inputToken;

            ccu::RemoteAddr dst;
            dst.addr = peerOutput[i];
            dst.token = peerToken[i];

            const uint16_t mask = static_cast<uint16_t>(1U << i);
            CCU_RETURN_IF_ERROR(ccu::Write(kernelArg->channels[i], dst, src, sliceBytes, event, mask));
            doneMask |= mask;
        }

        CCU_RETURN_IF_ERROR(ccu::EventWait(event, doneMask));
        for (uint32_t i = 0; i < channelCount; ++i) {
            CCU_RETURN_IF_ERROR(
                ccu::NotifyRecord(kernelArg->channels[i], CKE_IDX_0, static_cast<uint16_t>(1U << POST_SYNC_ID)));
        }
        return CcuResult::CCU_SUCCESS;
    }

    ccu::Variable wave0Offset;
    wave0Offset = 0;
    ccu::Variable wave1Offset;
    wave1Offset = chunk0Bytes;
    ccu::Event wave0Event;
    ccu::Event wave1Event;
    uint16_t wave0Mask = 0;
    uint16_t wave1Mask = 0;

    if (kernelArg->doSelfCopy) {
        CCU_RETURN_IF_ERROR(IssueWaveStarSelfCopy(*kernelArg, input, inputToken, selfOutput, selfToken,
            wave0Offset, chunk0Bytes, wave0Event, wave0Mask));
        CCU_IF(chunk1Bytes != 0)
        {
            CCU_RETURN_IF_ERROR(IssueWaveStarSelfCopy(*kernelArg, input, inputToken, selfOutput, selfToken,
                wave1Offset, chunk1Bytes, wave1Event, wave1Mask));
        }
    }

    for (uint32_t i = 0; i < channelCount; ++i) {
        CCU_RETURN_IF_ERROR(ccu::NotifyWait(kernelArg->channels[i], CKE_IDX_0, readyBits));
        CCU_RETURN_IF_ERROR(IssueWaveStarPeerWrite(*kernelArg, i, peerOutput[i], peerToken[i],
            input, inputToken, wave0Offset, chunk0Bytes, wave0Event, wave0Mask));
        CCU_IF(chunk1Bytes != 0)
        {
            CCU_RETURN_IF_ERROR(IssueWaveStarPeerWrite(*kernelArg, i, peerOutput[i], peerToken[i],
                input, inputToken, wave1Offset, chunk1Bytes, wave1Event, wave1Mask));
        }
    }

    CCU_RETURN_IF_ERROR(ccu::EventWait(wave0Event, wave0Mask));
    CCU_IF(chunk1Bytes != 0)
    {
        CCU_RETURN_IF_ERROR(ccu::EventWait(wave1Event, wave1Mask));
    }
    for (uint32_t i = 0; i < channelCount; ++i) {
        CCU_RETURN_IF_ERROR(
            ccu::NotifyRecord(kernelArg->channels[i], CKE_IDX_0, static_cast<uint16_t>(1U << POST_SYNC_ID)));
    }
    return CcuResult::CCU_SUCCESS;
}

// ===========================================================================
// 路径 4: Wave-Star Peer
//   只连 Root 一条 channel。Load output/token, publish 到 Root, Wait DONE, return。
//   不加载 input / chunk, 不创建 peerOutput[] / Event, 不回 ACK, 不与其他 Peer 同步。
// ===========================================================================
CcuResult CcuScatterWaveStarPeerKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgWaveStar *>(arg);
    if (kernelArg == nullptr || kernelArg->channelCount != 1 ||
        kernelArg->peerRanks[0] != kernelArg->root) {
        return CcuResult::CCU_E_INTERNAL;
    }

    ccu::Variable selfOutput;
    ccu::Variable selfToken;
    CCU_RETURN_IF_ERROR(ccu::LoadArg(selfOutput, 0));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(selfToken, 1));

    CCU_RETURN_IF_ERROR(ccu::WriteVariableWithNotify(kernelArg->channels[0], selfOutput, OUTPUT_XN_ID,
        CKE_IDX_0, static_cast<uint16_t>(1U << OUTPUT_XN_ID)));
    CCU_RETURN_IF_ERROR(ccu::WriteVariableWithNotify(kernelArg->channels[0], selfToken, TOKEN_XN_ID,
        CKE_IDX_0, static_cast<uint16_t>(1U << TOKEN_XN_ID)));
    CCU_RETURN_IF_ERROR(
        ccu::NotifyWait(kernelArg->channels[0], CKE_IDX_0, static_cast<uint16_t>(1U << POST_SYNC_ID)));
    return CcuResult::CCU_SUCCESS;
}

// ===========================================================================
// 路径 3: Proxy (512MB)
//   root            : Mesh kernel 发本 Server 块, 并把被代转的跨服块尾段送到代转者的
//                     CCL buffer; Clos kernel 发跨服块的前段。
//   代转者(本Server) : Mesh kernel 等 root 的中转数据, Clos kernel 用自己的 Clos 口
//                     把尾段转发给目标跨服 rank。两条流由 host 侧串行下发。
//   其他            : 只参与前/后同步。
// ===========================================================================
CcuResult CcuScatterProxyKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgProxy *>(arg);
    if (kernelArg == nullptr || kernelArg->channelCount == 0) {
        return CcuResult::CCU_E_INTERNAL;
    }
    const uint32_t channelCount = kernelArg->channelCount;
    const bool isClosKernel = kernelArg->isClosGroup;
    const bool proxyOn = kernelArg->proxyEnabled;
    const bool isRoot = (kernelArg->rankId == kernelArg->root);
    // 代转者: 非 root 且持有转发目标(proxyTargets 在 Mesh/Clos kernel 上语义一致)
    const bool iAmForwarder = proxyOn && (!isRoot) && (kernelArg->proxyTargetCount > 0);
    // 分块流水(注册期常量, 仅 4x3 为真): 代转者按阶段多次启动, "收 -> 转"重叠。
    const bool multiStage = kernelArg->multiStageRelay && proxyOn;
    // 去串行化(仅 400M 档): 同一条线程上先后放两个内核, 用 kernel 边界把
    // "该早走的先走完"变成硬保证 —— 同一内核内调换语句顺序对链路占用次序无影响。
    //   root  : SEED(前同步 + 只投递代理尾段) -> LOCAL(直发块 + 自留块 + 后同步)
    //   代转者: TAIL(前同步 + 只等尾段就绪) -> SYNC(只做后同步栅栏)
    const uint32_t meshPhase = kernelArg->meshPhase;
    const bool isSeed = (meshPhase == CcuKernelArgProxy::MESH_PHASE_SEED);
    const bool isLocal = (meshPhase == CcuKernelArgProxy::MESH_PHASE_LOCAL);
    const bool isTailOnly = (meshPhase == CcuKernelArgProxy::MESH_PHASE_TAIL);
    const bool isSyncOnly = (meshPhase == CcuKernelArgProxy::MESH_PHASE_SYNC);
    // SEED 之后才轮到 LOCAL, 而 SEED 已经把前同步的等待做完了: NotifyWait 是电平检查,
    // 已置位的 bit 不会因为被等过一次就消失, 所以 LOCAL 再等一次是立即返回的空操作。
    // 保留它(而不是跳过)是为了让每个内核自身都是自洽的 —— LOCAL 也要读对端的
    // OUTPUT/TOKEN 变量, 任务图里应当存在对应的等待节点。
    // LOCAL 不再做前同步: 同线程上更早的 SEED 已经一次等齐了它要读的全部变量。
    // 对端是在同一次循环里按 OUTPUT -> TOKEN -> SCRATCH -> SCRATCH_TOKEN 的顺序发布的,
    // 所以 SEED 等到 scratch 就蕴含了 LOCAL 要读的 OUTPUT/TOKEN; 而代理路径下
    // root 的 Mesh 组**恰好就是** proxyPeers 集合(2x8 的 7 个代转者 / 4x3 的 2 个),
    // 不存在"SEED 跳过的通道"。
    // 这样 root 的关键路径上少 7 次(2x8) / 2 次(4x3) NotifyWait。
    const bool needPreSync = !isSyncOnly && !isLocal;
    const bool needPostSync = !isSeed && !isTailOnly;
    // P2 (Ready->Write): root Mesh SEED 不再整组等齐, 把等待折进中转写循环。
    // 仅 4x3 中档 seedReadyWrite=true; 2x8 / 4x3-512MB 为假时下面分支全部不进入。
    const bool deferProxyWait =
        kernelArg->seedReadyWrite && isRoot && isSeed && proxyOn && !isClosKernel;

    // ---------- 1. 取资源 ----------
    std::vector<ccu::Variable> peerOutput(channelCount);
    std::vector<ccu::Variable> peerToken(channelCount);
    std::vector<ccu::Variable> peerScratch(channelCount);
    std::vector<ccu::Variable> peerScratchToken(channelCount);
    for (uint32_t i = 0; i < channelCount; ++i) {
        peerOutput[i] = ccu::GetResByChannel<ccu::Variable>(kernelArg->channels[i], OUTPUT_XN_ID);
        peerToken[i] = ccu::GetResByChannel<ccu::Variable>(kernelArg->channels[i], TOKEN_XN_ID);
        if (proxyOn) {
            peerScratch[i] = ccu::GetResByChannel<ccu::Variable>(kernelArg->channels[i], SCRATCH_XN_ID);
            peerScratchToken[i] = ccu::GetResByChannel<ccu::Variable>(kernelArg->channels[i], SCRATCH_TOKEN_XN_ID);
        }
    }

    // ---------- 2. 加载参数(顺序须与 host taskArgs 完全一致, 共 9 项) ----------
    ccu::Variable input;
    ccu::Variable inputToken;
    ccu::Variable selfOutput;
    ccu::Variable selfToken;
    ccu::Variable sliceBytes;
    ccu::Variable closDirectBytes;
    ccu::Variable proxyBytes;
    ccu::Variable selfScratch;
    ccu::Variable selfScratchToken;
    ccu::Variable stageId;     // 分块流水阶段号(仅 multiStageRelay 且为代转者时有效)
    ccu::Variable chunk0Bytes; // 分块流水: 第 0 块的字节数
    ccu::Variable chunk1Bytes; // 分块流水: 第 1 块的字节数
    CCU_RETURN_IF_ERROR(ccu::LoadArg(input, 0));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(inputToken, 1));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(selfOutput, 2));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(selfToken, 3));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(sliceBytes, 4));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(closDirectBytes, 5));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(proxyBytes, 6));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(selfScratch, 7));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(selfScratchToken, 8));
    // 分块流水新增的 3 项(非分块路径传占位值, 不参与任何判断)
    CCU_RETURN_IF_ERROR(ccu::LoadArg(stageId, PROXY_ARG_STAGE_ID));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(chunk0Bytes, PROXY_ARG_CHUNK0_BYTES));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(chunk1Bytes, PROXY_ARG_CHUNK1_BYTES));

    // ---------- 3. 前同步 ----------
    // proxyOn 为假时只交换 recvBuf 地址与 token, 与无代理版本逐字节一致。
    //
    // CCL buffer(scratch) 地址只在 root <-> 代转者 这一条链路上存在读取方:
    //   root    : 要拿到代转者的 scratch 地址, 才能把中转尾段写进它的 CCL buffer;
    //   代转者  : 读的是自己的 selfScratch(由 taskArgs 带入), 不需要对端的;
    //   root 自己的 scratch 没有任何读取方(root 不是代转者)。
    // 故只在"代转者 -> root"这一个方向写, root 只在通向代转者的 channel 上等。
    // 其余 channel 上的 scratch 交换没有读取方, 是纯开销。
    //
    // recvBuf 地址(token) 同理: **root 自己的那份没有读取方** —— root 的 recvBuf 只被它
    // 自己的 LocalCopy 写(见第 4 段), 没有任何对端会写进来, 也就没有 rank 需要 root 的地址。
    // 故 proxyOn 时 root 不再广播自己的 OUTPUT/TOKEN, 各 rank 也不再在通向 root 的那条
    // channel 上等这两位(等不到会挂死, 所以两处必须成对改动)。
    // root 自己仍要在所有 channel 上等 —— 它要拿每个对端的地址才能写过去。
    // 本条仅在 proxyOn 时启用: 4x1 / 8+4 的 512MB 走同一内核但 proxyOn=false, 二者
    // 已在榜一分位(2.06ms 并列第一 / 差 0.3%), 无收益空间, 不为它们引入任何变动。
    const bool trimRootPublish = proxyOn;
    const uint32_t myRootChannelIdx = FindPeerChannel(kernelArg, kernelArg->root);
    const uint16_t readyBits = static_cast<uint16_t>((1U << OUTPUT_XN_ID) | (1U << TOKEN_XN_ID));
    const uint16_t readyBitsWithScratch =
        static_cast<uint16_t>(readyBits | (1U << SCRATCH_XN_ID) | (1U << SCRATCH_TOKEN_XN_ID));

    auto doPreSync = [&]() -> CcuResult {
        for (uint32_t i = 0; i < channelCount; ++i) {
            if (!(trimRootPublish && isRoot)) {
                CCU_RETURN_IF_ERROR(ccu::WriteVariableWithNotify(kernelArg->channels[i], selfOutput,
                    OUTPUT_XN_ID, CKE_IDX_0, static_cast<uint16_t>(1U << OUTPUT_XN_ID)));
                CCU_RETURN_IF_ERROR(ccu::WriteVariableWithNotify(kernelArg->channels[i], selfToken,
                    TOKEN_XN_ID, CKE_IDX_0, static_cast<uint16_t>(1U << TOKEN_XN_ID)));
            }
            if (iAmForwarder && !isClosKernel && i == myRootChannelIdx) {
                CCU_RETURN_IF_ERROR(ccu::WriteVariableWithNotify(kernelArg->channels[i], selfScratch,
                    SCRATCH_XN_ID, CKE_IDX_0, static_cast<uint16_t>(1U << SCRATCH_XN_ID)));
                CCU_RETURN_IF_ERROR(ccu::WriteVariableWithNotify(kernelArg->channels[i], selfScratchToken,
                    SCRATCH_TOKEN_XN_ID, CKE_IDX_0, static_cast<uint16_t>(1U << SCRATCH_TOKEN_XN_ID)));
            }
        }
        // A0: root 的 Clos 出口是共享瓶颈, 它的等待折进数据面的写循环(见第 4 段),
        // 这里不再整组等齐 —— 否则出口要空转着等最慢那个跨服对端就绪。
        // 默认只改 root 的 Clos 内核; root 的 Mesh 组与其余角色不动。
        // 例外: seedReadyWrite 时 Mesh SEED 把代转者 scratch 的等待折进写循环。
        // 发布循环在它上面, 原样保留。
        if (isRoot && isClosKernel) {
            return CcuResult::CCU_SUCCESS;
        }
        for (uint32_t i = 0; i < channelCount; ++i) {
            // root 已不广播自己的 OUTPUT/TOKEN, 通向 root 的 channel 上没有这两位可等。
            // (非 root 也从不读 root 的地址, 见上一段的说明)
            if (trimRootPublish && !isRoot && i == myRootChannelIdx) {
                continue;
            }
            uint32_t dummyIdx = 0;
            // 只有 root 的 Mesh 内核会读代转者的 scratch(见第 4 段的中转循环);
            // root 的 Clos 内核、以及其余角色都不读, 故不必等这 2 个 bit。
            const bool expectScratch = isRoot && !isClosKernel && proxyOn &&
                IsInList(kernelArg->proxyPeers, kernelArg->proxyPeerCount, kernelArg->peerRanks[i], dummyIdx);
            if (deferProxyWait && expectScratch) {
                continue;
            }
            CCU_RETURN_IF_ERROR(ccu::NotifyWait(kernelArg->channels[i], CKE_IDX_0,
                expectScratch ? readyBitsWithScratch : readyBits));
        }
        return CcuResult::CCU_SUCCESS;
    };

    // 分块流水下代转者要按阶段多次启动内核, 前同步只能在第 0 阶段做一次 ——
    // 否则后续阶段会去等一个已被消费掉的 bit, 直接挂死。
    if (!needPreSync) {
        // MESH_PHASE_SYNC: 只做后同步栅栏, 前同步已由同线程上更早的 TAIL 内核完成
    } else if (multiStage && iAmForwarder) {
        CCU_IF(stageId == SCATTER_PROXY_STAGE_CHUNK0)
        {
            CCU_RETURN_IF_ERROR(doPreSync());
        }
    } else {
        CCU_RETURN_IF_ERROR(doPreSync());
    }

    // ---------- 4. 数据面 ----------
    // SEED 内核不做直发块: 它的全部意义就是让代理尾段抢在直发块之前占用 Mesh 链路。
    if (isRoot && !isSeed) {
        ccu::Event directEvent;
        uint16_t directMask = 0;
        for (uint32_t i = 0; i < channelCount; ++i) {
            // A0: root 的 Clos 内核在这里逐通道等待 —— 等一条就发一条。
            // 掩码与它原来在前同步里用的完全一致(Clos 内核不读 scratch, 故为 readyBits)。
            // root 的 Mesh 组不做这一步: SEED 的等待要覆盖后续 LOCAL 读取的变量,
            // 而 LOCAL 的启动本来就被 SEED 的 kernel 边界卡住, 折进来没有收益。
            if (isClosKernel) {
                CCU_RETURN_IF_ERROR(ccu::NotifyWait(kernelArg->channels[i], CKE_IDX_0, readyBits));
            }
            const uint32_t peer = kernelArg->peerRanks[i];
            ccu::Variable blockOffset;
            MakeBlockOffset(sliceBytes, blockOffset, peer);
            ccu::LocalAddr src;
            src.addr = input;
            src.addr += blockOffset;
            src.token = inputToken;
            ccu::RemoteAddr dst;
            dst.addr = peerOutput[i];
            dst.token = peerToken[i];

            uint32_t dummyIdx = 0;
            const bool proxied = proxyOn && isClosKernel &&
                IsInList(kernelArg->proxyTargets, kernelArg->proxyTargetCount, peer, dummyIdx);
            // 跨服块中被代转的只发前段; 其余(含全部本 Server 块)整块发
            const ccu::Variable &len = proxied ? closDirectBytes : sliceBytes;
            const uint16_t mask = static_cast<uint16_t>(1U << i);
            CCU_RETURN_IF_ERROR(ccu::Write(kernelArg->channels[i], dst, src, len, directEvent, mask));
            directMask |= mask;
        }

        // root 自留块: 只能由指定的那一个 kernel 做(见 CcuKernelArgProxy::doSelfCopy)
        if (kernelArg->doSelfCopy) {
            ccu::Variable selfOffset;
            MakeBlockOffset(sliceBytes, selfOffset, kernelArg->root);
            ccu::LocalAddr localSrc;
            localSrc.addr = input;
            localSrc.addr += selfOffset;
            localSrc.token = inputToken;
            ccu::LocalAddr localDst;
            localDst.addr = selfOutput;
            localDst.token = selfToken;
            const uint16_t selfMask = static_cast<uint16_t>(1U << channelCount);
            CCU_RETURN_IF_ERROR(ccu::LocalCopy(localDst, localSrc, sliceBytes, directEvent, selfMask));
            directMask |= selfMask;
        }
        CCU_RETURN_IF_ERROR(ccu::EventWait(directEvent, directMask));

    }
    // Mesh kernel 额外把被代转跨服块的尾段送到代转者的 CCL buffer。
    // LOCAL 内核不做这一段(尾段已由 SEED 投递), 否则会重复写同一段 scratch。
    if (isRoot && !isLocal && proxyOn && !isClosKernel && kernelArg->proxyPeerCount > 0) {
        ccu::Event transferEvent;
        uint16_t maskChunk0 = 0;
        uint16_t maskChunk1 = 0;
        const uint32_t k = kernelArg->targetsPerForwarder;
        for (uint32_t i = 0; i < channelCount; ++i) {
            uint32_t t = 0;
            if (!IsInList(kernelArg->proxyPeers, kernelArg->proxyPeerCount, kernelArg->peerRanks[i], t)) {
                continue;
            }
            if (deferProxyWait) {
                CCU_RETURN_IF_ERROR(ccu::NotifyWait(kernelArg->channels[i], CKE_IDX_0,
                    readyBitsWithScratch));
            }
            if (k > 1) {
                for (uint32_t j = 0; j < k; ++j) {
                    const uint32_t target = kernelArg->proxyTargets[t * k + j];
                    ccu::Variable srcOffset;
                    MakeBlockOffset(sliceBytes, srcOffset, target);
                    srcOffset += closDirectBytes;
                    ccu::LocalAddr src;
                    src.addr = input;
                    src.addr += srcOffset;
                    src.token = inputToken;
                    ccu::RemoteAddr dst;
                    dst.addr = peerScratch[i];
                    for (uint32_t s = 0; s <= j; ++s) {
                        dst.addr += proxyBytes;
                    }
                    dst.token = peerScratchToken[i];
                    const uint16_t mask = static_cast<uint16_t>(1U << (i * k + j));
                    CCU_RETURN_IF_ERROR(ccu::Write(kernelArg->channels[i], dst, src, proxyBytes,
                        transferEvent, mask));
                    maskChunk0 |= mask;
                }
                continue;
            }
            const uint32_t target = kernelArg->proxyTargets[t];
            ccu::Variable srcOffset;
            MakeBlockOffset(sliceBytes, srcOffset, target);
            srcOffset += closDirectBytes; // 尾段紧跟在 root 直发的前段之后
            ccu::LocalAddr src;
            src.addr = input;
            src.addr += srcOffset;
            src.token = inputToken;
            ccu::RemoteAddr dst;
            dst.addr = peerScratch[i];
            dst.addr += proxyBytes; // 每个代转者只代转 1 个目标, 固定用第 1 个 slot
            dst.token = peerScratchToken[i];
            const uint16_t mask = static_cast<uint16_t>(1U << i);
            if (multiStage) {
                // 尾段拆成两块, 分别占一个 event bit 并各自通知 —— 第 0 块落位即可
                // 让代转者开始转发, 不必等整段到齐。
                CCU_RETURN_IF_ERROR(ccu::Write(kernelArg->channels[i], dst, src, chunk0Bytes,
                    transferEvent, mask));
                maskChunk0 |= mask;

                ccu::RemoteAddr dst1;
                dst1.addr = peerScratch[i];
                dst1.addr += proxyBytes;
                dst1.addr += chunk0Bytes;
                dst1.token = peerScratchToken[i];
                ccu::LocalAddr src1;
                src1.addr = input;
                src1.addr += srcOffset;
                src1.addr += chunk0Bytes;
                src1.token = inputToken;
                const uint16_t mask1 = static_cast<uint16_t>(mask << 8);
                CCU_RETURN_IF_ERROR(ccu::Write(kernelArg->channels[i], dst1, src1, chunk1Bytes,
                    transferEvent, mask1));
                maskChunk1 |= mask1;
            } else {
                CCU_RETURN_IF_ERROR(ccu::Write(kernelArg->channels[i], dst, src, proxyBytes,
                    transferEvent, mask));
                maskChunk0 |= mask;
            }
        }
        // 第 0 块落位 -> 立刻通知代转者开始转发
        if (maskChunk0 != 0) {
            CCU_RETURN_IF_ERROR(ccu::EventWait(transferEvent, maskChunk0));
        }
        for (uint32_t i = 0; i < channelCount; ++i) {
            uint32_t t = 0;
            if (IsInList(kernelArg->proxyPeers, kernelArg->proxyPeerCount, kernelArg->peerRanks[i], t)) {
                CCU_RETURN_IF_ERROR(ccu::NotifyRecord(kernelArg->channels[i], CKE_IDX_0,
                    static_cast<uint16_t>(1U << PROXY_DATA_READY_ID)));
            }
        }
        // 第 1 块落位 -> 通知代转者收尾
        if (maskChunk1 != 0) {
            CCU_RETURN_IF_ERROR(ccu::EventWait(transferEvent, maskChunk1));
            for (uint32_t i = 0; i < channelCount; ++i) {
                uint32_t t = 0;
                if (IsInList(kernelArg->proxyPeers, kernelArg->proxyPeerCount,
                        kernelArg->peerRanks[i], t)) {
                    CCU_RETURN_IF_ERROR(ccu::NotifyRecord(kernelArg->channels[i], CKE_IDX_0,
                        static_cast<uint16_t>(1U << PROXY_CHUNK1_READY_ID)));
                }
            }
        }
    }
    if (!isRoot && iAmForwarder && !isSyncOnly) {
        if (!isClosKernel) {
            // Mesh kernel: 等 root 把中转数据写进本端 CCL buffer。
            // 分块流水下每个阶段只等"本阶段那一片"。
            const uint32_t rootIdx = FindPeerChannel(kernelArg, kernelArg->root);
            if (rootIdx != INVALID_VALUE_RANKID) {
                if (multiStage) {
                    CCU_IF(stageId == SCATTER_PROXY_STAGE_CHUNK0)
                    {
                        CCU_RETURN_IF_ERROR(ccu::NotifyWait(kernelArg->channels[rootIdx], CKE_IDX_0,
                            static_cast<uint16_t>(1U << PROXY_DATA_READY_ID)));
                    }
                    CCU_IF(stageId == SCATTER_PROXY_STAGE_CHUNK1)
                    {
                        CCU_RETURN_IF_ERROR(ccu::NotifyWait(kernelArg->channels[rootIdx], CKE_IDX_0,
                            static_cast<uint16_t>(1U << PROXY_CHUNK1_READY_ID)));
                    }
                    // TAIL 阶段 Mesh 内核无事可做(数据面由 Clos 内核收尾)
                } else {
                    CCU_RETURN_IF_ERROR(ccu::NotifyWait(kernelArg->channels[rootIdx], CKE_IDX_0,
                        static_cast<uint16_t>(1U << PROXY_DATA_READY_ID)));
                }
            }
        } else {
            // Clos kernel: 用本 rank 自己的 Clos 口把尾段转发给目标跨服 rank
            auto doForward = [&](const ccu::Variable &len, const ccu::Variable &offset) -> CcuResult {
                ccu::Event forwardEvent;
                uint16_t forwardMask = 0;
                for (uint32_t t = 0; t < kernelArg->proxyTargetCount; ++t) {
                    const uint32_t target = kernelArg->proxyTargets[t];
                    const uint32_t targetIdx = FindPeerChannel(kernelArg, target);
                    if (targetIdx == INVALID_VALUE_RANKID) {
                        continue;
                    }
                    ccu::LocalAddr src; // 源是本端 CCL buffer 里 root 写进来的中转段
                    src.addr = selfScratch;
                    src.addr += proxyBytes;
                    for (uint32_t s = 0; s < t; ++s) {
                        src.addr += proxyBytes;
                    }
                    src.addr += offset;
                    src.token = selfScratchToken;
                    ccu::RemoteAddr dst; // 落到目标 recvBuf 直发段之后
                    dst.addr = peerOutput[targetIdx];
                    dst.addr += closDirectBytes;
                    dst.addr += offset;
                    dst.token = peerToken[targetIdx];
                    const uint16_t mask = static_cast<uint16_t>(1U << t);
                    CCU_RETURN_IF_ERROR(ccu::Write(kernelArg->channels[targetIdx], dst, src, len,
                        forwardEvent, mask));
                    forwardMask |= mask;
                }
                if (forwardMask != 0) {
                    CCU_RETURN_IF_ERROR(ccu::EventWait(forwardEvent, forwardMask));
                }
                return CcuResult::CCU_SUCCESS;
            };

            ccu::Variable zeroOffset;
            zeroOffset = 0;
            if (multiStage) {
                // 第 1 阶段: 转第 0 块(此时 Mesh 内核正在收第 1 块 —— 这就是重叠所在)
                CCU_IF(stageId == SCATTER_PROXY_STAGE_CHUNK1)
                {
                    CCU_RETURN_IF_ERROR(doForward(chunk0Bytes, zeroOffset));
                }
                // 第 2 阶段: 转第 1 块收尾
                CCU_IF(stageId == SCATTER_PROXY_STAGE_TAIL)
                {
                    CCU_RETURN_IF_ERROR(doForward(chunk1Bytes, chunk0Bytes));
                }
            } else {
                CCU_RETURN_IF_ERROR(doForward(proxyBytes, zeroOffset));
            }
        }
    }
    // 其他 rank 不产生数据面任务: 数据由 root(及代转者)写来。

    // ---------- 5. 后同步 ----------
    // 分块流水下代转者会被启动多次, 后同步只在各自的最后一个阶段做一次:
    // Mesh 内核在 CHUNK1 阶段就收完了, Clos 内核要到 TAIL 阶段才收尾。
    auto doPostSync = [&]() -> CcuResult {
        for (uint32_t i = 0; i < channelCount; ++i) {
            CCU_RETURN_IF_ERROR(ccu::NotifyRecord(kernelArg->channels[i], CKE_IDX_0,
                static_cast<uint16_t>(1U << PROXY_POST_SYNC_ID)));
        }
        for (uint32_t i = 0; i < channelCount; ++i) {
            CCU_RETURN_IF_ERROR(ccu::NotifyWait(kernelArg->channels[i], CKE_IDX_0,
                static_cast<uint16_t>(1U << PROXY_POST_SYNC_ID)));
        }
        return CcuResult::CCU_SUCCESS;
    };

    if (!needPostSync) {
        // SEED / TAIL: 本内核结束得早正是目的所在, 后同步交给同线程上紧随其后的
        // LOCAL / SYNC 内核 —— 保护没有减少, 只是把栅栏挪到了更晚的一个内核里。
    } else if (multiStage && iAmForwarder) {
        if (isClosKernel) {
            CCU_IF(stageId == SCATTER_PROXY_STAGE_TAIL)
            {
                CCU_RETURN_IF_ERROR(doPostSync());
            }
        } else {
            CCU_IF(stageId == SCATTER_PROXY_STAGE_CHUNK1)
            {
                CCU_RETURN_IF_ERROR(doPostSync());
            }
        }
    } else {
        CCU_RETURN_IF_ERROR(doPostSync());
    }

    return CcuResult::CCU_SUCCESS;
}

} // namespace ops_hccl

#undef CCU_RETURN_IF_ERROR
