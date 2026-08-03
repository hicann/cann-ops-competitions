/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_HCCL_CCU_KERNEL_H
#define OPS_HCCL_CCU_KERNEL_H

#include "custom.h"

namespace ops_hccl {

/**
 * @brief 保存单个 AllGather CCU Kernel 变体的固定资源和静态调度信息
 */
struct CcuKernelArgAllGather : CcuKernelArgBase {
    uint32_t myRank;
    uint32_t rankSize;
    uint32_t variantType;
    uint32_t phaseId;
    uint32_t sourceKind;
    uint32_t handleSelf;
    uint32_t peerRanks[MAX_PEER_COUNT];
    uint32_t segmentRanks[MAX_SEGMENT_COUNT];
    uint8_t eventBits[MAX_OPERATION_COUNT];
    uint32_t segmentCount;
    uint32_t operationCount;
    uint32_t selfEventBit;
    uint16_t completionMask;
    uint16_t reserved;
};

static_assert(std::is_trivially_copyable<CcuKernelArgAllGather>::value,
    "CcuKernelArgAllGather must be trivially copyable");

/**
 * @brief 按静态变体描述完成地址交换、AllGather 数据写入、本地拷贝和通道同步
 * @param arg 指向 CcuKernelArgAllGather 固定参数对象的 CCU Kernel 参数
 * @return 成功返回 CCU_SUCCESS，参数或 CCU 指令生成失败返回对应错误码
 */
CcuResult CcuKernel(CcuKernelArg arg);

}

#endif
