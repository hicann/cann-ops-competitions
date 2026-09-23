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

#include <vector>

#include "common.h"
#include "custom.h"
#include "log.h"

namespace ccu = ::AscendC::ccu;

namespace ops_hccl {

enum ScatterLocalRole : uint32_t {
    SCATTER_ROLE_NORMAL = 0,
    SCATTER_ROLE_ROOT = 1,
    SCATTER_ROLE_HELPER = 2,
    SCATTER_ROLE_TARGET = 3,
    SCATTER_ROLE_HUB = 4,
};

enum ScatterPeerRole : uint32_t {
    SCATTER_PEER_NORMAL = 0,
    SCATTER_PEER_RELAY_TARGET = 1,
};

struct CcuKernelArgScatter : public CcuKernelArgBase {
    uint32_t rankSize;
    uint32_t rankId;
    uint32_t rootId;
    uint32_t localRole;
    uint32_t partnerRank;
    // Exactly one active root kernel owns each local output window.
    uint32_t copyRoot = 0U;
    // V23: on selected large 8+4 Direct paths, start remote writes before the local root copy.
    uint32_t copyRootLate = 0U;
    uint32_t peerRanks[MAX_RANK_SIZE];
    uint32_t peerRoles[MAX_RANK_SIZE];
    uint32_t pairedRanks[MAX_RANK_SIZE];
};

// Versioned ABI for V20. The grouped copy uses nine task arguments.
// The seven-target ingress uses ten, adding the packed scratch slot pitch.
struct CcuKernelArgHub : public CcuKernelArgBase {
    uint32_t rankSize = 0;
    uint32_t rankId = 0;
    uint32_t rootId = 0;
    uint32_t isSender = 0;
    uint32_t copyRoot = 0U;
    uint32_t peerRanks[MAX_RANK_SIZE]{};
    uint32_t sourceIndices[MAX_RANK_SIZE]{};
    uint32_t targetRanks[7]{};
};
CcuResult CcuScatterHubCopyKernel(CcuKernelArg arg);
CcuResult CcuScatterHubIngressKernel(CcuKernelArg arg);

struct ScatterContext {
    const CcuKernelArgScatter *arg;
    ccu::Variable input;
    std::vector<ccu::Variable> output;
    std::vector<ccu::Variable> token;
    ccu::Variable inputToken;
    ccu::Variable relayBuffer;
    ccu::Variable relayBufferToken;
    ccu::Variable sliceStride;
    ccu::Variable windowOffset;
    ccu::Variable sliceSize;
    ccu::Variable rootCopyEnabled;
    ccu::Variable directBytes;
    ccu::Variable relayBytes;
    ccu::Variable relayOffset;
    ccu::Variable relayChunkBytes;
    ccu::Variable scratchOffset;
    ccu::Event event;
};

CcuResult CcuScatterDirectLegacyKernel(CcuKernelArg arg);
CcuResult CcuScatterDirectKernel(CcuKernelArg arg);
CcuResult CcuScatterDirectBatchKernel(CcuKernelArg arg);
CcuResult CcuScatterRelayLoadKernel(CcuKernelArg arg);
CcuResult CcuScatterRelayForwardKernel(CcuKernelArg arg);
CcuResult CcuScatterRelayFinalKernel(CcuKernelArg arg);

} // namespace ops_hccl

#endif // OPS_HCCL_CCU_KERNEL_H
