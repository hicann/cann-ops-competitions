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

#include <memory>
#include <hccl/hccl_types.h>
#include <hccl/hccl_res.h>

#include "binary_stream.h"
#include "common.h"

typedef struct {
    void *addr;
    uint64_t size;
} CommBuffer;

constexpr uint32_t CUSTOM_MAX_RANK_SIZE = 32;
constexpr uint32_t CUSTOM_MAX_CHANNELS = CUSTOM_MAX_RANK_SIZE - 1;

enum class KernelKind : uint32_t {
    GATHER = 0,
    FINAL_REDUCE = 1,
    LOCAL_SERVER_REDUCE = 2,
    CLOS_PAIR_GATHER = 3,
    FINAL_PAIR_REDUCE = 4,
    GROUP_REDUCE = 5,
    HIERARCHICAL_MESH_PARTIAL = 6,
    HIERARCHICAL_CLOS_PAIR = 7,
    HIERARCHICAL_FINAL = 8,
    NHR = 9,
    STRIPED_CLOS_DIRECT = 10,
    STRIPED_LAYER_DIRECT = 11,
};

struct CcuKernelArgBase {
    ChannelHandle channels[MAX_RANK_SIZE];
    uint32_t channelCount;
};

// ccu kernel register所需信息
struct CcuKernelInfo {
    // kernel名称
    char kernelFuncName[64];
    // kernel函数
    void *kernelFunc;
    // KernelArg实例指针
    void *kernelArg;

private:
    std::shared_ptr<CcuKernelArgBase> kernelArgSmartPtr;

public:
    template <typename T> void setKernelArg(std::shared_ptr<T> arg)
    {
        kernelArgSmartPtr = std::static_pointer_cast<CcuKernelArgBase>(arg);
        kernelArg = static_cast<void *>(arg.get());
    }
};

// The starter base payload has room for 16 channels, while final domains have
// up to 32 ranks.  CCU kernels use this derived payload's channel array.
struct ReduceScatterCcuKernelArg : public CcuKernelArgBase {
    ChannelHandle commChannels[CUSTOM_MAX_CHANNELS] = {};
    uint32_t peerRanks[CUSTOM_MAX_CHANNELS] = {};
    uint64_t slotOffsets[CUSTOM_MAX_RANK_SIZE] = {};
    uint32_t commChannelCount = 0;
    uint32_t rankSize = 0;
    uint32_t rankId = 0;
    HcclDataType dataType = HCCL_DATA_TYPE_RESERVED;
    HcclReduceOp reduceOp = HCCL_REDUCE_SUM;
    KernelKind kind = KernelKind::GATHER;
    bool useSmallTree = false;
    bool stageSelfInput = false;
    bool selfInputStaged = false;
    bool includeSelf = false;
    // The gather kernel reduces its layer's fixed source set into this CCL
    // slot.  Mesh and Clos kernels always use distinct slots.
    uint32_t partialResultSlot = INVALID_VALUE_RANKID;
    bool finalOnClosDie = false;
    ChannelHandle finalDieAnchorChannel = {};
    uint32_t localRanks[CUSTOM_MAX_RANK_SIZE] = {};
    uint32_t localRankCount = 0;
    uint32_t closPeerRank = INVALID_VALUE_RANKID;
    uint32_t nhrDistances[4] = {};
    uint32_t nhrStepCount = 0;
    bool nhrWriteOutput = false;
    bool directInitializeOutput = false;
};

struct AlgResourceCtx {
    ThreadHandle ccuThread;            ///< CCU通信引擎上的thread资源
    CommBuffer localBuffer;            ///< 本端HCCL通信内存
    std::vector<ThreadHandle> threads; ///< CCU通信引擎上的thread资源
    std::vector<CcuKernelHandle> ccuKernels;
    uint32_t gatherKernelIndex[2] = {INVALID_VALUE_RANKID, INVALID_VALUE_RANKID};
    uint32_t finalReduceKernelIndex = INVALID_VALUE_RANKID;
    uint32_t finalReduceThreadIndex = 0;
    uint32_t localServerReduceKernelIndex = INVALID_VALUE_RANKID;
    uint32_t closPairGatherKernelIndex = INVALID_VALUE_RANKID;
    uint32_t finalPairReduceKernelIndex = INVALID_VALUE_RANKID;
    uint32_t directStripeCounts[2] = {0, 0};
    bool useHierarchicalTwoByEight = false;
    bool useCcuBufferGroupReduce = false;
    bool useLayerPartialReduce = false;
    bool useSmallNhr = false;
    bool useSmallGroupFanIn = false;
    bool useStripedClosDirect = false;
    bool useStripedTwoLayerDirect = false;

    // 序列化
    std::vector<char> Serialize()
    {
        BinaryStream binaryStream;
        binaryStream << ccuThread;
        binaryStream << localBuffer;
        binaryStream << threads;
        binaryStream << ccuKernels;
        binaryStream << gatherKernelIndex[0];
        binaryStream << gatherKernelIndex[1];
        binaryStream << finalReduceKernelIndex;
        binaryStream << finalReduceThreadIndex;
        binaryStream << localServerReduceKernelIndex;
        binaryStream << closPairGatherKernelIndex;
        binaryStream << finalPairReduceKernelIndex;
        binaryStream << directStripeCounts[0];
        binaryStream << directStripeCounts[1];
        binaryStream << useHierarchicalTwoByEight;
        binaryStream << useCcuBufferGroupReduce;
        binaryStream << useLayerPartialReduce;
        binaryStream << useSmallNhr;
        binaryStream << useSmallGroupFanIn;
        binaryStream << useStripedClosDirect;
        binaryStream << useStripedTwoLayerDirect;
        std::vector<char> result;
        binaryStream.Dump(result);
        return result;
    }

    // 反序列化
    void DeSerialize(std::vector<char> &data)
    {
        BinaryStream binaryStream(data);
        binaryStream >> ccuThread;
        binaryStream >> localBuffer;
        binaryStream >> threads;
        binaryStream >> ccuKernels;
        binaryStream >> gatherKernelIndex[0];
        binaryStream >> gatherKernelIndex[1];
        binaryStream >> finalReduceKernelIndex;
        binaryStream >> finalReduceThreadIndex;
        binaryStream >> localServerReduceKernelIndex;
        binaryStream >> closPairGatherKernelIndex;
        binaryStream >> finalPairReduceKernelIndex;
        binaryStream >> directStripeCounts[0];
        binaryStream >> directStripeCounts[1];
        binaryStream >> useHierarchicalTwoByEight;
        binaryStream >> useCcuBufferGroupReduce;
        binaryStream >> useLayerPartialReduce;
        binaryStream >> useSmallNhr;
        binaryStream >> useSmallGroupFanIn;
        binaryStream >> useStripedClosDirect;
        binaryStream >> useStripedTwoLayerDirect;
    }
};

#endif // OPS_HCCL_CUSTOM_H
