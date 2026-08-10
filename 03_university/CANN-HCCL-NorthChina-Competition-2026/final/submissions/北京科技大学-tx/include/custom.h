/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_HCCL_CUSTOM_H
#define OPS_HCCL_CUSTOM_H

#include <cstdint>
#include <vector>

#include <hccl/hccl_res.h>

#include "binary_stream.h"
#include "common.h"

enum class AllGatherKernelMode : uint64_t {
    DIRECT = 0,
    COLUMN_LAYER0 = 1,
    COLUMN_LAYER1 = 2,
    RD4_XOR_TWO_STAGE = 3,
};

enum class AllGatherKernelLayer : uint32_t {
    LAYER0 = 0,
    LAYER1 = 1,
};

enum class A84PrefixKernelMode : uint64_t {
    LAYER0_OWN = 0,
    LAYER0_RELAY = 1,
    LAYER1_PREFIX = 2,
    LAYER1_SUFFIX = 3,
};

// v6-A direct kernel argument. It is used only by 4x1 and 8+4 in F1.
struct CcuAllGatherKernelArg {
    ChannelHandle channels[MAX_RANK_SIZE]{};
    uint32_t channelCount = 0;
    uint32_t layer = static_cast<uint32_t>(AllGatherKernelLayer::LAYER0);
    uint32_t directCopyLocal = 0;
    uint32_t columnCopyLocal = 0;
    uint32_t columnPeerChannelIndex = 0;
    uint32_t rd4Xor1ChannelIndex = 0;
    uint32_t rd4Xor2ChannelIndex = 0;
};

// Dedicated small-message pull argument. The remote input virtual addresses
// come from custom sendBuf memories exchanged while acquiring the Channels.
// Input tokens remain dynamic and are exchanged once per invocation. Each
// kernel then reads its peers directly into disjoint final-output slots, so no
// data-completion PostSync is required.
struct CcuSmallPullKernelArg {
    ChannelHandle channels[MAX_RANK_SIZE]{};
    uint64_t registeredRemoteInputs[MAX_RANK_SIZE]{};
    uint32_t peerRanks[MAX_RANK_SIZE]{};
    uint32_t channelCount = 0;
    uint32_t directCopyLocal = 0;
};


// Dedicated 8+4 asymmetric Prefix-first kernels. They reuse the existing
// one-Channel-per-peer handles but are registered separately from the D06
// Direct kernels so the 512 KiB path remains byte-for-byte D06.
struct CcuA84PrefixLayer0KernelArg {
    ChannelHandle channels[MAX_RANK_SIZE]{};
    uint32_t channelCount = 0;
    uint32_t myRank = 0;
    uint32_t copyLocal = 0;
    uint32_t relaySourceRanks[2]{};
    uint32_t relaySourceCount = 0;
};

struct CcuA84PrefixLayer1KernelArg {
    ChannelHandle channels[MAX_RANK_SIZE]{};
    uint32_t channelCount = 0;
    uint32_t outgoingProxyChannelIndex = 0;
    uint32_t incomingProxyMask = 0;
    uint32_t copyLocal = 0;
};

// The two 2x8 Stage kernels use disjoint channel sets. The layer-1 kernel
// additionally needs the one symmetric partner channel selected from the
// sorted layer-0/layer-1 topology lists.
struct CcuM1StageKernelArg {
    ChannelHandle channels[MAX_RANK_SIZE]{};
    uint32_t channelCount = 0;
    uint32_t partnerChannelIndex = 0;
};

struct AlgResourceCtx {
    // threads[0] is stream-backed. With two IO Dies, threads[1] owns layer-1.
    std::vector<ThreadHandle> threads;

    // 2x8: [M1 layer-0 Stage, M1 layer-1 Stage].
    // 8+4: [v6-A direct layer-0, v6-A direct layer-1].
    // 4x1: [v6-A direct layer-1].
    std::vector<CcuKernelHandle> ccuKernels;

    uint32_t hasLayer0Kernel = 0;
    uint32_t useTwoStage2x8 = 0;
    uint32_t partnerRank = INVALID_VALUE_RANKID;
    uint32_t useAsymmetricPrefix8Plus4 = 0;
    uint32_t localGroupSize = 0;

    std::vector<char> Serialize()
    {
        BinaryStream binaryStream;
        binaryStream << threads;
        binaryStream << ccuKernels;
        binaryStream << hasLayer0Kernel;
        binaryStream << useTwoStage2x8;
        binaryStream << partnerRank;
        binaryStream << useAsymmetricPrefix8Plus4;
        binaryStream << localGroupSize;
        std::vector<char> result;
        binaryStream.Dump(result);
        return result;
    }

    void DeSerialize(std::vector<char> &data)
    {
        BinaryStream binaryStream(data);
        binaryStream >> threads;
        binaryStream >> ccuKernels;
        binaryStream >> hasLayer0Kernel;
        binaryStream >> useTwoStage2x8;
        binaryStream >> partnerRank;
        binaryStream >> useAsymmetricPrefix8Plus4;
        binaryStream >> localGroupSize;
    }
};

namespace ops_hccl {

// Existing v6-A direct entry points, registered only for 4x1 and 8+4.
CcuResult CcuAllGatherLayer0Kernel(CcuKernelArg arg);
CcuResult CcuAllGatherLayer1Kernel(CcuKernelArg arg);

// Small-message pull entry points. Their remote input addresses are static
// registered-memory resources; only the input token is exchanged at runtime.
CcuResult CcuSmallPullLayer0Kernel(CcuKernelArg arg);
CcuResult CcuSmallPullLayer1Kernel(CcuKernelArg arg);
CcuResult CcuSmallPullWarmLayer0Kernel(CcuKernelArg arg);
CcuResult CcuSmallPullWarmLayer1Kernel(CcuKernelArg arg);

// 8+4 asymmetric Prefix-first entry points. Each kernel references channels
// from exactly one IO Die and is used only for rankSize==12 large messages.
CcuResult CcuA84PrefixLayer0Kernel(CcuKernelArg arg);
CcuResult CcuA84PrefixLayer1Kernel(CcuKernelArg arg);

// F1 2x8 entry points. Each kernel references channels from exactly one IO Die.
CcuResult CcuM1Layer0StageKernel(CcuKernelArg arg);
CcuResult CcuM1Layer1StageKernel(CcuKernelArg arg);

} // namespace ops_hccl

#endif // OPS_HCCL_CUSTOM_H
