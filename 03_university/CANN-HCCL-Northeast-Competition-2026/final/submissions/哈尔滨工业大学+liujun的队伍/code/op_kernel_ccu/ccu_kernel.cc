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

#include "../include/log.h"
#include "ccu_kernel.h"

#ifndef OPS_HCCL_SCATTER_CUSTOM_H_486
#error "DIAG: custom.h not effectively included - guard pre-empted by another header?"
#endif

namespace ops_hccl {
namespace ccu = ::AscendC::ccu;

constexpr uint32_t XN_ROUTE = 0;
constexpr uint16_t BIT_ROUTE = 1u;
constexpr uint32_t XN_BUF_ADDR = 1;
constexpr uint32_t XN_BUF_TOKEN = 2;
constexpr uint16_t BIT_BUF_ADDR = static_cast<uint16_t>(1u << XN_BUF_ADDR);
constexpr uint16_t BIT_BUF_TOKEN = static_cast<uint16_t>(1u << XN_BUF_TOKEN);
constexpr uint16_t BIT_PUB_A = static_cast<uint16_t>(BIT_BUF_ADDR | BIT_BUF_TOKEN); // 1|2
constexpr uint16_t BIT_BUF_ADDR_B = static_cast<uint16_t>(1u << 3);
constexpr uint16_t BIT_BUF_TOKEN_B = static_cast<uint16_t>(1u << 4);
constexpr uint16_t BIT_PUB_B = static_cast<uint16_t>(BIT_BUF_ADDR_B | BIT_BUF_TOKEN_B); // 3|4
constexpr uint16_t BIT_READ_DONE = static_cast<uint16_t>(1u << 5);
constexpr uint16_t BIT_DIRECT_DONE = static_cast<uint16_t>(1u << 7);
constexpr uint16_t BIT_PIECE_DONE = static_cast<uint16_t>(1u << 8);
constexpr uint16_t EV_MAIN = static_cast<uint16_t>(1u << 14);
constexpr uint16_t EV_AUX = static_cast<uint16_t>(1u << 15);
constexpr uint32_t CKE_IDX_0 = 0;

#define SCATTER_CCU_CHK_RET(call)                                                                          \
    do {                                                                                                   \
        CcuResult ccuRet = (call);                                                                         \
        if (UNLIKELY(ccuRet != CCU_SUCCESS)) {                                                             \
            HCCL_ERROR("[%s] ccu call trace: ccuRet -> %d", __func__, ccuRet);                             \
            return ccuRet;                                                                                 \
        }                                                                                                  \
    } while (0)
struct ScatterArgs {
    const std::vector<uint32_t> &ids;
    std::vector<ccu::Variable> values;
    explicit ScatterArgs(ScatterEntry entry) : ids(ScatterEntryArgs(entry)), values(ids.size()) {}
    ccu::Variable &operator[](uint32_t id) {
        for (size_t i = 0; i < ids.size(); ++i) if (ids[i] == id) return values[i];
        std::abort();
    }
    CcuResult Load() {
        for (uint32_t i = 0; i < values.size(); ++i) {
            SCATTER_CCU_CHK_RET(ccu::LoadArg(values[i], i));
        }
        return CCU_SUCCESS;
    }
};

namespace {

struct ChannelVars {
    std::vector<ChannelHandle> channels;
    std::vector<ccu::Variable> bufAddr;
    std::vector<ccu::Variable> bufToken;

    CcuResult Init(CcuKernelArgScatter *ka)
    {
        const uint32_t n = ka->channelCount;
        channels.resize(n);
        bufAddr.reserve(n);
        bufToken.reserve(n);
        for (uint32_t i = 0; i < n; i++) {
            channels[i] = ka->channels[i];
            bufAddr.push_back(ccu::GetResByChannel<ccu::Variable>(ka->channels[i], XN_BUF_ADDR));
            bufToken.push_back(ccu::GetResByChannel<ccu::Variable>(ka->channels[i], XN_BUF_TOKEN));
        }
        return CCU_SUCCESS;
    }
};

CcuKernelArgScatter *Enter(CcuKernelArg arg, const char *name, [[maybe_unused]] const char *role)
{
    auto *ka = static_cast<CcuKernelArgScatter *>(arg);
    if (ka == nullptr || ka->channelCount == 0) {
        HCCL_ERROR("[%s] kernelArg or channels is invalid", name);
        return nullptr;
    }
    return ka;
}

inline CcuResult PublishBuffer(ChannelHandle ch, ccu::Variable &addr, ccu::Variable &token, uint16_t addrMask,
    uint16_t tokenMask)
{
    SCATTER_CCU_CHK_RET(ccu::WriteVariableWithNotify(ch, addr, XN_BUF_ADDR, CKE_IDX_0, addrMask));
    SCATTER_CCU_CHK_RET(ccu::WriteVariableWithNotify(ch, token, XN_BUF_TOKEN, CKE_IDX_0, tokenMask));
    return CCU_SUCCESS;
}

inline CcuResult LocalSliceCopy(CcuKernelArgScatter *ka, ScatterArgs &args)
{
    if (ka->hasLocalSlice == 0) {
        return CCU_SUCCESS;
    }
    ccu::LocalAddr dst;
    dst.addr = args[A_RECV_BUF];
    dst.token = args[A_RECV_TOKEN];
    ccu::LocalAddr src;
    src.addr = args[A_SEND_BUF];
    src.addr += args[A_SELF_OFF];
    src.token = args[A_SEND_TOKEN];
    ccu::Event ev;
    {
        SCATTER_CCU_CHK_RET(ccu::LocalCopy(dst, src, args[A_SLICE_BYTES], ev, EV_MAIN));
    }
    SCATTER_CCU_CHK_RET(ccu::EventWait(ev, EV_MAIN));
    return CCU_SUCCESS;
}

} // namespace

CcuResult CcuScatterRootPub(CcuKernelArg arg)
{
    CcuKernelArgScatter *ka = Enter(arg, "CcuScatterRootPub", "root·发布（小份/中继拓扑的Server内相位）");
    if (ka == nullptr) {
        return CcuResult::CCU_E_INTERNAL;
    }
    const uint32_t chNum = ka->channelCount;
    ChannelVars v;
    SCATTER_CCU_CHK_RET(v.Init(ka));
    ScatterArgs args(ENTRY_ROOT_PUB);
    SCATTER_CCU_CHK_RET(args.Load());

    for (uint32_t i = 0; i < chNum; i++) {
        const bool relayChannel = (ka->isMeshGroup != 0) && (i < ka->relayCount);
        const uint16_t addrMask = relayChannel ? static_cast<uint16_t>(BIT_BUF_ADDR | BIT_BUF_ADDR_B) : BIT_BUF_ADDR;
        const uint16_t tokenMask = relayChannel ? static_cast<uint16_t>(BIT_BUF_TOKEN | BIT_BUF_TOKEN_B)
                                                : BIT_BUF_TOKEN;
        SCATTER_CCU_CHK_RET(PublishBuffer(v.channels[i], args[A_SEND_BUF], args[A_SEND_TOKEN], addrMask, tokenMask));
    }
    SCATTER_CCU_CHK_RET(LocalSliceCopy(ka, args));
    return CcuResult::CCU_SUCCESS;
}
CcuResult CcuScatterPeerPull(CcuKernelArg arg)
{
    CcuKernelArgScatter *ka = Enter(arg, "CcuScatterPeerPull", "非root·拉取自己的份（小份）");
    if (ka == nullptr) {
        return CcuResult::CCU_E_INTERNAL;
    }
    const uint32_t chNum = ka->channelCount;
    ChannelVars v;
    SCATTER_CCU_CHK_RET(v.Init(ka));
    ScatterArgs args(ENTRY_PEER_PULL);
    SCATTER_CCU_CHK_RET(args.Load());

    for (uint32_t i = 0; i < chNum; i++) {
        if (ka->rootId == ka->peerRanks[i])
        {
            SCATTER_CCU_CHK_RET(ccu::NotifyWait(v.channels[i], CKE_IDX_0, BIT_PUB_A));
            ccu::RemoteAddr src;
            src.addr = v.bufAddr[i];
            src.addr += args[A_SELF_OFF];
            src.token = v.bufToken[i];
            ccu::LocalAddr dst;
            dst.addr = args[A_RECV_BUF];
            dst.token = args[A_RECV_TOKEN];
            ccu::Event ev;
            {
                SCATTER_CCU_CHK_RET(ccu::Read(v.channels[i], dst, src, args[A_SLICE_BYTES], ev, EV_MAIN));
            }
            SCATTER_CCU_CHK_RET(ccu::EventWait(ev, EV_MAIN));
        }
    }
    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuScatterRootPush(CcuKernelArg arg)
{
    CcuKernelArgScatter *ka = Enter(arg, "CcuScatterRootPush", "root·推送（跨Server相位）");
    if (ka == nullptr) {
        return CcuResult::CCU_E_INTERNAL;
    }
    const uint32_t chNum = ka->channelCount;
    ChannelVars v;
    SCATTER_CCU_CHK_RET(v.Init(ka));
    ScatterArgs args(ENTRY_ROOT_PUSH);
    SCATTER_CCU_CHK_RET(args.Load());

    SCATTER_CCU_CHK_RET(LocalSliceCopy(ka, args));
    ccu::Event ev;
    uint16_t waitMask = 0;
    ccu::Variable off;
    off = args[A_FIRST_PEER_OFF];
    uint32_t prev = ka->firstPeerRank;
    for (uint32_t i = 0; i < chNum; i++) {
        SCATTER_CCU_CHK_RET(ccu::NotifyWait(v.channels[i], CKE_IDX_0, BIT_PUB_A));
        ccu::Variable route;
        route = static_cast<uint64_t>(0xFFFFFFFFu);
        if (i < ka->relayCount) {
            CCU_IF(args[A_BETA] != 0) { route = static_cast<uint64_t>(ka->relaySourceRanks[i]); }
        }
        SCATTER_CCU_CHK_RET(ccu::WriteVariableWithNotify(v.channels[i], route, XN_ROUTE, CKE_IDX_0, BIT_ROUTE));
        const uint32_t t = ka->peerRanks[i];
        for (uint32_t gap = t - prev; gap > 0; gap--) {
            off += args[A_CHUNK];
        }
        prev = t;
        ccu::RemoteAddr dst;
        dst.addr = v.bufAddr[i];
        dst.token = v.bufToken[i];
        ccu::LocalAddr src;
        src.addr = args[A_SEND_BUF];
        src.addr += off;
        src.addr += args[A_SLICE_OFF];
        src.token = args[A_SEND_TOKEN];
        ccu::Variable len;
        len = (i < ka->relayCount) ? args[A_ALPHA] : args[A_SLICE_BYTES];
        const uint16_t bit = static_cast<uint16_t>(1u << i);
        waitMask = static_cast<uint16_t>(waitMask | bit);
        {
            SCATTER_CCU_CHK_RET(ccu::Write(v.channels[i], dst, src, len, ev, bit));
        }
    }
    SCATTER_CCU_CHK_RET(ccu::EventWait(ev, waitMask));
    for (uint32_t i = 0; i < chNum; i++) {
        SCATTER_CCU_CHK_RET(ccu::NotifyRecord(v.channels[i], CKE_IDX_0, BIT_DIRECT_DONE));
    }
    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuScatterRecvLarge(CcuKernelArg arg)
{
    auto *ka = Enter(arg, "CcuScatterRecvLarge", "sparse receiver");
    if (ka == nullptr) return CCU_E_INTERNAL;
    ChannelVars v;
    SCATTER_CCU_CHK_RET(v.Init(ka));
    ScatterArgs args(ENTRY_RECV_LARGE);
    SCATTER_CCU_CHK_RET(args.Load());
    for (uint32_t r = 0; r < ka->channelCount; ++r) {
        if (ka->rootId != ka->peerRanks[r]) continue;
        SCATTER_CCU_CHK_RET(PublishBuffer(v.channels[r], args[A_RECV_BUF], args[A_RECV_TOKEN], BIT_BUF_ADDR, BIT_BUF_TOKEN));
        SCATTER_CCU_CHK_RET(ccu::NotifyWait(v.channels[r], CKE_IDX_0, BIT_ROUTE));
        auto route = ccu::GetResByChannel<ccu::Variable>(v.channels[r], XN_ROUTE);
        for (uint32_t i = 0; i < ka->channelCount; ++i) {
            if (ka->peerRanks[i] == ka->rootId) continue;
            CCU_IF(route == static_cast<uint64_t>(ka->peerRanks[i])) {
                SCATTER_CCU_CHK_RET(PublishBuffer(v.channels[i], args[A_RECV_BUF], args[A_RECV_TOKEN], BIT_BUF_ADDR, BIT_BUF_TOKEN));
            }
        }
        SCATTER_CCU_CHK_RET(ccu::NotifyWait(v.channels[r], CKE_IDX_0, BIT_DIRECT_DONE));
        for (uint32_t i = 0; i < ka->channelCount; ++i) {
            if (ka->peerRanks[i] == ka->rootId) continue;
            CCU_IF(route == static_cast<uint64_t>(ka->peerRanks[i])) {
                SCATTER_CCU_CHK_RET(ccu::NotifyWait(v.channels[i], CKE_IDX_0, BIT_PIECE_DONE));
            }
        }
    }
    return CCU_SUCCESS;
}

CcuResult CcuScatterRelayRead(CcuKernelArg arg)
{
    CcuKernelArgScatter *ka = Enter(arg, "CcuScatterRelayRead", "中继卡·Server内相位1/2（只读中转片）");
    if (ka == nullptr) {
        return CcuResult::CCU_E_INTERNAL;
    }
    const uint32_t chNum = ka->channelCount;
    ChannelVars v;
    SCATTER_CCU_CHK_RET(v.Init(ka));
    ScatterArgs args(ENTRY_RELAY_READ);
    SCATTER_CCU_CHK_RET(args.Load());

    for (uint32_t i = 0; i < chNum; i++) {
        if (ka->rootId == ka->peerRanks[i])
        {
            SCATTER_CCU_CHK_RET(ccu::NotifyWait(v.channels[i], CKE_IDX_0, BIT_PUB_A));
            ccu::RemoteAddr paySrc;
            paySrc.addr = v.bufAddr[i];
            paySrc.addr += args[A_RELAY_SRC_OFF];
            paySrc.token = v.bufToken[i];
            ccu::LocalAddr stage;
            stage.addr = args[A_CCL_BASE];
            stage.token = args[A_CCL_TOKEN];
            ccu::Event ev;
            CCU_IF(args[A_BETA] != 0)
            {
                SCATTER_CCU_CHK_RET(ccu::Read(v.channels[i], stage, paySrc, args[A_BETA], ev, EV_AUX));
            }
            CCU_IF(args[A_BETA] == 0) { SCATTER_CCU_CHK_RET(ccu::EventRecord(ev, EV_AUX)); }
            SCATTER_CCU_CHK_RET(ccu::EventWait(ev, EV_AUX));
        }
    }
    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuScatterRelayOwn(CcuKernelArg arg)
{
    CcuKernelArgScatter *ka = Enter(arg, "CcuScatterRelayOwn", "中继卡·读自留份（大份档第二段 / 小份档唯一内核）");
    if (ka == nullptr) {
        return CcuResult::CCU_E_INTERNAL;
    }
    const uint32_t chNum = ka->channelCount;
    ChannelVars v;
    SCATTER_CCU_CHK_RET(v.Init(ka));
    ScatterArgs args(ENTRY_RELAY_OWN);
    SCATTER_CCU_CHK_RET(args.Load());

    for (uint32_t i = 0; i < chNum; i++) {
        if (ka->rootId == ka->peerRanks[i])
        {
            CCU_IF(args[A_LARGE_MODE] != 0)
            {
                SCATTER_CCU_CHK_RET(ccu::NotifyWait(v.channels[i], CKE_IDX_0, BIT_PUB_B));
            }
            CCU_IF(args[A_LARGE_MODE] == 0)
            {
                SCATTER_CCU_CHK_RET(ccu::NotifyWait(v.channels[i], CKE_IDX_0,
                    static_cast<uint16_t>(BIT_PUB_A | BIT_PUB_B)));
            }
            ccu::RemoteAddr ownSrc;
            ownSrc.addr = v.bufAddr[i];
            ownSrc.addr += args[A_SELF_OFF];
            ownSrc.token = v.bufToken[i];
            ccu::LocalAddr dst;
            dst.addr = args[A_RECV_BUF];
            dst.token = args[A_RECV_TOKEN];
            ccu::Event ev;
            {
                SCATTER_CCU_CHK_RET(ccu::Read(v.channels[i], dst, ownSrc, args[A_SLICE_BYTES], ev, EV_MAIN));
            }
            SCATTER_CCU_CHK_RET(ccu::EventWait(ev, EV_MAIN));
        }
    }
    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuScatterRelayFwd(CcuKernelArg arg)
{
    auto *ka = Enter(arg, "CcuScatterRelayFwd", "sparse forward");
    if (ka == nullptr) return CCU_E_INTERNAL;
    ChannelVars v;
    SCATTER_CCU_CHK_RET(v.Init(ka));
    ScatterArgs args(ENTRY_RELAY_FWD);
    SCATTER_CCU_CHK_RET(args.Load());
    for (uint32_t i = 0; i < ka->channelCount; ++i) {
        if (ka->targetRank != ka->peerRanks[i]) continue;
        SCATTER_CCU_CHK_RET(ccu::NotifyWait(v.channels[i], CKE_IDX_0, BIT_PUB_A));
        ccu::RemoteAddr dst;
        dst.addr = v.bufAddr[i];
        dst.addr += args[A_ALPHA];
        dst.token = v.bufToken[i];
        ccu::LocalAddr src;
        src.addr = args[A_CCL_BASE];
        src.token = args[A_CCL_TOKEN];
        ccu::Event ev;
        SCATTER_CCU_CHK_RET(ccu::Write(v.channels[i], dst, src, args[A_BETA], ev, EV_MAIN));
        SCATTER_CCU_CHK_RET(ccu::EventWait(ev, EV_MAIN));
        SCATTER_CCU_CHK_RET(ccu::NotifyRecord(v.channels[i], CKE_IDX_0, BIT_PIECE_DONE));
    }
    return CCU_SUCCESS;
}
} // namespace ops_hccl
