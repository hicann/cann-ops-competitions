/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root directory of the software repository for the full text of the License.
 */

// ===== 提交版本 v63-final | v60安全基线 + peer就绪流水规约 | 2026-07-24 =====
// v63-final: notify数量和槽位完全保持v60，仅缩短peer就绪后的等待路径
// 关键修复: 回到v44单槽位模式 (CKE_IDX_0 + 不同mask bit), 与官方demo一致
//   v52-v54 错误地改用多槽位 → 导致不出性能分
//   PostSync必须包含NotifyRecord+NotifyWait, 否则多轮调用时数据竞争
// Event bit: 1u = bit0 (正确, 与官方demo一致)
// 算法特点: LocalCopy与前同步重叠 + 每个peer就绪即串行ReadReduce

#include <hcomm/hcomm_primitives.h>
#include <ccu/ccu_variable.hpp>
#include <ccu/ccu_event.hpp>
#include <ccu/ccu_primitives.hpp>

#include "ccu_kernel.h"
#include "custom.h"

namespace ccu = ::AscendC::ccu;

namespace ops_hccl {

#define CCU_CHK_RET(call)                                                         \
    do {                                                                          \
        CcuResult ccuRet = (call);                                                \
        if (ccuRet != CcuResult::CCU_SUCCESS) {                                   \
            HCCL_ERROR("[%s] ccu call failed: ccuRet -> %d", __func__, ccuRet);  \
            return ccuRet;                                                        \
        }                                                                         \
    } while (0)

CcuResult CcuKernel(::CcuKernelArg argPtr)
{
    auto *arg = static_cast<CcuKernelArg *>(argPtr);
    auto *argBase = static_cast<CcuKernelArgBase *>(argPtr);
    if (argBase->channelCount == 0) {
        HCCL_ERROR("[CcuKernel] channels is empty!");
        return CcuResult::CCU_E_INTERNAL;
    }

    // 1. 加载 5 个参数
    ccu::Variable localDstAddr;
    ccu::Variable exchangeAddr;
    ccu::Variable localDstToken;
    ccu::Variable exchangeToken;
    ccu::Variable sliceSize;

    uint32_t argId = 0;
    CCU_CHK_RET(ccu::LoadArg(localDstAddr, argId++));
    CCU_CHK_RET(ccu::LoadArg(exchangeAddr, argId++));
    CCU_CHK_RET(ccu::LoadArg(localDstToken, argId++));
    CCU_CHK_RET(ccu::LoadArg(exchangeToken, argId++));
    CCU_CHK_RET(ccu::LoadArg(sliceSize, argId++));

    // 2. LocalCopy 发起后不等待, 与前同步重叠 (与v56相同, 与v55不同)
    ccu::LocalAddr dst;
    dst.addr = localDstAddr;
    dst.token = localDstToken;

    ccu::Event initEvent;
    bool needInitWait = false;
    if (arg->phase == 0) {
        ccu::LocalAddr src;
        src.addr = exchangeAddr;
        src.token = exchangeToken;
        CCU_CHK_RET(ccu::LocalCopy(dst, src, sliceSize, initEvent, 1u));
        needInitWait = true;
    }

    // 3. 前同步: 单槽位模式 — CKE_IDX_0, 不同mask bit
    for (uint32_t i = 0; i < argBase->channelCount; i++) {
        CCU_CHK_RET(ccu::WriteVariableWithNotify(argBase->channels[i], exchangeAddr,
            OUTPUT_XN_ID, CKE_IDX_0, 1u << OUTPUT_XN_ID));
        CCU_CHK_RET(ccu::WriteVariableWithNotify(argBase->channels[i], exchangeToken,
            TOKEN_XN_ID, CKE_IDX_0, 1u << TOKEN_XN_ID));
    }

    // LocalCopy 完成后等待 (与WriteVariableWithNotify重叠)
    if (needInitWait) {
        CCU_CHK_RET(ccu::EventWait(initEvent, 1u));
    }

    // 4. peer就绪流水: Wait→GetRes→ReadReduce→EventWait。
    // 所有WriteVariableWithNotify已在上方完成，因此此循环不会阻止任何peer发布地址。
    const uint32_t allBit = (1u << OUTPUT_XN_ID) | (1u << TOKEN_XN_ID);
    ccu::Event event;
    for (uint32_t i = 0; i < argBase->channelCount; i++) {
        CCU_CHK_RET(ccu::NotifyWait(argBase->channels[i], CKE_IDX_0, allBit));

        ccu::Variable peerAddr = ccu::GetResByChannel<ccu::Variable>(
            argBase->channels[i], OUTPUT_XN_ID);
        ccu::Variable peerToken = ccu::GetResByChannel<ccu::Variable>(
            argBase->channels[i], TOKEN_XN_ID);

        ccu::RemoteAddr remoteSrc;
        remoteSrc.addr = peerAddr;
        remoteSrc.token = peerToken;

        CCU_CHK_RET(ccu::ReadReduce(argBase->channels[i],
            dst, remoteSrc, sliceSize,
            arg->dataType, arg->reduceOp,
            event, 1u));
        CCU_CHK_RET(ccu::EventWait(event, 1u));
    }

    // 5. 完整后同步: NotifyRecord + NotifyWait (单槽位, 不同mask bit)
    //    NotifyRecord: 通知对端本 rank 已完成 ReadReduce
    //    NotifyWait: 等待对端完成, 确保下一轮调用时对端不再读本 rank 数据
    for (uint32_t i = 0; i < argBase->channelCount; i++) {
        CCU_CHK_RET(ccu::NotifyRecord(argBase->channels[i], CKE_IDX_0, 1u << POST_SYNC_ID));
    }
    for (uint32_t i = 0; i < argBase->channelCount; i++) {
        CCU_CHK_RET(ccu::NotifyWait(argBase->channels[i], CKE_IDX_0, 1u << POST_SYNC_ID));
    }

    return CcuResult::CCU_SUCCESS;
}

} // namespace ops_hccl
