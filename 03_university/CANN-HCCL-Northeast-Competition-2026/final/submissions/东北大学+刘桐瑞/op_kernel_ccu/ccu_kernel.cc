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

namespace ops_hccl {
namespace {
struct ScatterChannelVars {
    std::vector<ccu::Variable> base;
    std::vector<ccu::Variable> token;
};

void LoadChannelVars(const ScatterKernelArg *ka, ScatterChannelVars &vars)
{
    const uint32_t chCnt = ka->channelCount;
    vars.base.resize(chCnt);
    vars.token.resize(chCnt);
    for (uint32_t c = 0; c < chCnt; c++) {
        vars.base[c] = ccu::GetResByChannel<ccu::Variable>(ka->channels[c], SCATTER_XN_BASE);
        vars.token[c] = ccu::GetResByChannel<ccu::Variable>(ka->channels[c], SCATTER_XN_TOKEN);
    }
}

bool CheckKernelArg(const ScatterKernelArg *ka)
{
    if (ka == nullptr) {
        return false;
    }
    if (ka->channelCount == 0 || static_cast<uint64_t>(ka->channelCount) >= MAX_RANK_SIZE) {
        return false;
    }
    return true;
}

} // namespace

CcuResult CcuKernelPullRoot(CcuKernelArg arg)
{
    auto *ka = static_cast<ScatterKernelArg *>(arg);
    if (!CheckKernelArg(ka)) {
        return CCU_E_INTERNAL;
    }
    const uint32_t chCnt = ka->channelCount;
    const bool handleSelf = (ka->handleSelfRank != 0);

    ccu::Variable inBase;
    ccu::Variable inToken;
    ccu::LoadArg(inBase, SC_PR_ARG_IN_BASE);
    ccu::LoadArg(inToken, SC_PR_ARG_IN_TOKEN);

    const uint32_t ownBase = static_cast<uint32_t>(SC_PR_ARG_FIXED_NUM);
    ccu::Variable ownDstBase;
    ccu::Variable ownDstToken;
    ccu::Variable ownSrcAddr;
    ccu::Variable ownSize;
    if (handleSelf) {
        ccu::LoadArg(ownDstBase, ownBase + SC_PR_OWN_OFF_DST_BASE);
        ccu::LoadArg(ownDstToken, ownBase + SC_PR_OWN_OFF_DST_TOKEN);
        ccu::LoadArg(ownSrcAddr, ownBase + SC_PR_OWN_OFF_SRC_ADDR);
        ccu::LoadArg(ownSize, ownBase + SC_PR_OWN_OFF_SIZE);
    }

    ccu::Event ev;

    ccu::LocalAddr ownSrc;
    ccu::LocalAddr ownDst;

    for (uint32_t c = 0; c < chCnt; c++) {
        ccu::WriteVariableWithNotify(
            ka->channels[c], inBase, SCATTER_XN_BASE, SCATTER_NOTIFY_IDX, SCATTER_BIT_PUB_BASE);
        ccu::WriteVariableWithNotify(
            ka->channels[c], inToken, SCATTER_XN_TOKEN, SCATTER_NOTIFY_IDX, SCATTER_BIT_PUB_TOKEN);
    }

    if (handleSelf) {
        CCU_IF(ownSize != 0)
        {
            ownSrc.addr = ownSrcAddr;
            ownSrc.token = inToken;
            ownDst.addr = ownDstBase;
            ownDst.token = ownDstToken;
            ccu::LocalCopy(ownDst, ownSrc, ownSize, ev, SCATTER_EV_OWN);
        }
    }

#if SCATTER_WAIT_SLOWEST_LAST

    const uint32_t crossNum = (ka->crossNum <= chCnt) ? ka->crossNum : chCnt;
    for (uint32_t i = crossNum; i < chCnt; i++) {
        ccu::NotifyWait(ka->channels[i], SCATTER_NOTIFY_IDX, SCATTER_BIT_READ_DONE);
    }
    for (uint32_t i = 0; i < crossNum; i++) {
        ccu::NotifyWait(ka->channels[i], SCATTER_NOTIFY_IDX, SCATTER_BIT_READ_DONE);
    }
#else
    for (uint32_t c = 0; c < chCnt; c++) {
        ccu::NotifyWait(ka->channels[c], SCATTER_NOTIFY_IDX, SCATTER_BIT_READ_DONE);
    }
#endif

    if (handleSelf) {
        CCU_IF(ownSize != 0) { ccu::EventWait(ev, SCATTER_EV_OWN); }
    }

    return CCU_SUCCESS;
}

CcuResult CcuKernelPush(CcuKernelArg arg)
{
    auto *ka = static_cast<ScatterKernelArg *>(arg);
    if (!CheckKernelArg(ka)) {
        return CCU_E_INTERNAL;
    }
    const uint32_t chCnt = ka->channelCount;
    const bool handleSelf = (ka->handleSelfRank != 0);

    ScatterChannelVars peer;
    LoadChannelVars(ka, peer);

    ccu::Variable role;
    ccu::Variable hasPub;
    ccu::Variable y1;
    ccu::Variable y2;
    ccu::Variable y3;
    ccu::Variable y4;
    ccu::Variable y5;
    ccu::Variable y6;
    ccu::Variable y7;
    ccu::Variable y8;
    std::vector<ccu::Variable> pc;
    pc.resize(chCnt);

    ccu::LoadArg(role, SC_PUSH_ARG_ROLE);
    ccu::LoadArg(hasPub, SC_PUSH_ARG_HAS_PUB);
    ccu::LoadArg(y1, SC_PUSH_ARG_Y1);
    ccu::LoadArg(y2, SC_PUSH_ARG_Y2);
    ccu::LoadArg(y3, SC_PUSH_ARG_Y3);
    ccu::LoadArg(y4, SC_PUSH_ARG_Y4);
    ccu::LoadArg(y5, SC_PUSH_ARG_Y5);
    ccu::LoadArg(y6, SC_PUSH_ARG_Y6);
    ccu::LoadArg(y7, SC_PUSH_ARG_Y7);
    ccu::LoadArg(y8, SC_PUSH_ARG_Y8);
    const uint32_t pcBase = static_cast<uint32_t>(SC_PUSH_ARG_FIXED_NUM);
    for (uint32_t c = 0; c < chCnt; c++) {
        ccu::LoadArg(pc[c], pcBase + c);
    }

    ccu::Variable &inBase = y1;
    ccu::Variable &inToken = y2;
    ccu::Variable &wrSize = y3;
    ccu::Variable &wrSizeRelay = y4;
    ccu::Variable &ownDstBase = y5;
    ccu::Variable &ownDstToken = y6;
    ccu::Variable &ownSrcAddr = y7;
    ccu::Variable &ownSize = y8;

    ccu::Variable &outBase = y1;
    ccu::Variable &outToken = y2;
    ccu::Variable &bufBase = y3;
    ccu::Variable &bufToken = y4;
    ccu::Variable &pullSrcOff = y5;
    ccu::Variable &fwdDstOff = y6;
    ccu::Variable &relaySize = y7;
    ccu::Variable &phase = y8;

    ccu::Variable &pullCh = y3;

    ccu::Event ev;
    const uint16_t chBits = static_cast<uint16_t>((1U << chCnt) - 1U);
    const uint16_t ownBit = static_cast<uint16_t>(1U << chCnt);

    std::vector<ccu::LocalAddr> src;
    std::vector<ccu::RemoteAddr> dst;
    src.resize(chCnt);
    dst.resize(chCnt);
    ccu::LocalAddr ownSrc;
    ccu::LocalAddr ownDst;
    ccu::LocalAddr stage;
    ccu::LocalAddr selfDst;
    ccu::RemoteAddr pullSrc;
    ccu::RemoteAddr fwdDst;

    CCU_IF(role == SC_PUSH_ROLE_NODE_PULL)
    {
        selfDst.addr = outBase;
        selfDst.token = outToken;
        for (uint32_t c = 0; c < chCnt; c++) {
            CCU_IF(pullCh == static_cast<uint64_t>(c))
            {
                ccu::NotifyWait(ka->channels[c], SCATTER_NOTIFY_IDX, SCATTER_BIT_PRESYNC);
                pullSrc.addr = peer.base[c];
                pullSrc.addr += pullSrcOff;
                pullSrc.token = peer.token[c];
                ccu::Read(ka->channels[c], selfDst, pullSrc, relaySize, ev, SCATTER_EV_SELF);
                ccu::EventWait(ev, SCATTER_EV_SELF);
                ccu::NotifyRecord(ka->channels[c], SCATTER_NOTIFY_IDX, SCATTER_BIT_READ_DONE);
            }
        }
    }

    CCU_IF(hasPub == 1)
    {
        for (uint32_t i = 0; i < ka->pubChannelCount; i++) {
            const ChannelHandle pubCh = ka->channels[ka->pubChannelIdx[i]];
            ccu::WriteVariableWithNotify(pubCh, inBase, SCATTER_XN_BASE, SCATTER_NOTIFY_IDX, SCATTER_BIT_PUB_BASE);
            ccu::WriteVariableWithNotify(pubCh, inToken, SCATTER_XN_TOKEN, SCATTER_NOTIFY_IDX, SCATTER_BIT_PUB_TOKEN);
        }
    }

    CCU_IF(role == SC_PUSH_ROLE_ROOT)
    {
        for (uint32_t c = 0; c < chCnt; c++) {
            src[c].addr = pc[c];
            src[c].token = inToken;
        }

        for (uint32_t c = 0; c < chCnt; c++) {
            ccu::NotifyWait(ka->channels[c], SCATTER_NOTIFY_IDX, SCATTER_BIT_PRESYNC);
        }

        for (uint32_t c = 0; c < chCnt; c++) {
            dst[c].addr = peer.base[c];
            dst[c].token = peer.token[c];
            ccu::Write(ka->channels[c], dst[c], src[c], (ka->wrUseRelaySize[c] != 0) ? wrSizeRelay : wrSize, ev,
                static_cast<uint16_t>(1U << c));
        }

        if (handleSelf) {
            ownSrc.addr = ownSrcAddr;
            ownSrc.token = inToken;
            ownDst.addr = ownDstBase;
            ownDst.token = ownDstToken;
            CCU_IF(ownSize != 0) { ccu::LocalCopy(ownDst, ownSrc, ownSize, ev, ownBit); }
            CCU_IF(ownSize == 0) { ccu::EventRecord(ev, ownBit); }
        }

#if SCATTER_POST_BEFORE_OWNCOPY

        ccu::EventWait(ev, chBits);
        for (uint32_t c = 0; c < chCnt; c++) {
            ccu::NotifyRecord(ka->channels[c], SCATTER_NOTIFY_IDX, SCATTER_BIT_POST);
        }

        if (handleSelf) {
            ccu::EventWait(ev, ownBit);
        }
#else
        const uint16_t allBit = handleSelf ? static_cast<uint16_t>(chBits | ownBit) : chBits;
        ccu::EventWait(ev, allBit);
        for (uint32_t c = 0; c < chCnt; c++) {
            ccu::NotifyRecord(ka->channels[c], SCATTER_NOTIFY_IDX, SCATTER_BIT_POST);
        }
#endif
    }

    CCU_IF(hasPub == 1)
    {
        for (uint32_t i = 0; i < ka->pubChannelCount; i++) {
            ccu::NotifyWait(ka->channels[ka->pubChannelIdx[i]], SCATTER_NOTIFY_IDX, SCATTER_BIT_READ_DONE);
        }
    }

    CCU_IF(role == SC_PUSH_ROLE_NODE)
    {
        for (uint32_t c = 0; c < chCnt; c++) {
            CCU_IF(pc[c] == SC_NODE_MODE_RECV)
            {
                ccu::WriteVariableWithNotify(
                    ka->channels[c], outBase, SCATTER_XN_BASE, SCATTER_NOTIFY_IDX, SCATTER_BIT_PUB_BASE);
                ccu::WriteVariableWithNotify(
                    ka->channels[c], outToken, SCATTER_XN_TOKEN, SCATTER_NOTIFY_IDX, SCATTER_BIT_PUB_TOKEN);
            }
        }

        CCU_IF(phase != SC_NODE_PHASE_NONE)
        {
            for (uint32_t c = 0; c < chCnt; c++) {
                CCU_IF(pc[c] == SC_NODE_MODE_PULL)
                {
                    ccu::WriteVariableWithNotify(
                        ka->channels[c], outBase, SCATTER_XN_BASE, SCATTER_NOTIFY_IDX, SCATTER_BIT_PUB_BASE);
                    ccu::WriteVariableWithNotify(
                        ka->channels[c], outToken, SCATTER_XN_TOKEN, SCATTER_NOTIFY_IDX, SCATTER_BIT_PUB_TOKEN);
                    ccu::NotifyWait(ka->channels[c], SCATTER_NOTIFY_IDX, SCATTER_BIT_PRESYNC);
                }
            }

            stage.addr = bufBase;
            stage.token = bufToken;

            for (uint32_t c = 0; c < chCnt; c++) {
                CCU_IF(pc[c] == SC_NODE_MODE_PULL)
                {
                    pullSrc.addr = peer.base[c];
                    pullSrc.addr += pullSrcOff;
                    pullSrc.token = peer.token[c];
                    ccu::Read(ka->channels[c], stage, pullSrc, relaySize, ev, SCATTER_EV_PULL);
                }
            }
            CCU_IF(phase == SC_NODE_PHASE_FWD) { ccu::EventRecord(ev, SCATTER_EV_PULL); }
            CCU_IF(phase == SC_NODE_PHASE_WAIT) { ccu::EventRecord(ev, SCATTER_EV_PULL); }
            ccu::EventWait(ev, SCATTER_EV_PULL);

            for (uint32_t c = 0; c < chCnt; c++) {
                CCU_IF(pc[c] == SC_NODE_MODE_PULL)
                {
                    ccu::NotifyRecord(ka->channels[c], SCATTER_NOTIFY_IDX, SCATTER_BIT_READ_DONE);
                }
            }

            for (uint32_t c = 0; c < chCnt; c++) {
                CCU_IF(pc[c] == SC_NODE_MODE_FWD)
                {
                    ccu::NotifyWait(ka->channels[c], SCATTER_NOTIFY_IDX, SCATTER_BIT_PRESYNC);
                    fwdDst.addr = peer.base[c];
                    fwdDst.addr += fwdDstOff;
                    fwdDst.token = peer.token[c];
                    ccu::Write(ka->channels[c], fwdDst, stage, relaySize, ev, SCATTER_EV_FWD);
                }
            }
            CCU_IF(phase == SC_NODE_PHASE_PULL) { ccu::EventRecord(ev, SCATTER_EV_FWD); }
            CCU_IF(phase == SC_NODE_PHASE_WAIT) { ccu::EventRecord(ev, SCATTER_EV_FWD); }
            ccu::EventWait(ev, SCATTER_EV_FWD);
            for (uint32_t c = 0; c < chCnt; c++) {
                CCU_IF(pc[c] == SC_NODE_MODE_FWD)
                {
                    ccu::NotifyRecord(ka->channels[c], SCATTER_NOTIFY_IDX, SCATTER_BIT_POST);
                }
            }

            for (uint32_t c = 0; c < chCnt; c++) {
                CCU_IF(pc[c] == SC_NODE_MODE_WAIT)
                {
                    ccu::NotifyWait(ka->channels[c], SCATTER_NOTIFY_IDX, SCATTER_BIT_POST);
                }
            }
        }

        for (uint32_t c = 0; c < chCnt; c++) {
            CCU_IF(pc[c] == SC_NODE_MODE_RECV)
            {
                ccu::NotifyWait(ka->channels[c], SCATTER_NOTIFY_IDX, SCATTER_BIT_POST);
            }
        }
    }

    return CCU_SUCCESS;
}

} // namespace ops_hccl
