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
#include <vector>

#include <hccl/hccl_res.h>
#include <hccl/hccl_types.h>

#include "binary_stream.h"
#include "common.h"

namespace final_small512 {
typedef struct {
    void *addr;
    uint64_t size;
} CommBuffer;

enum class Rs2BuildFlavor : uint32_t {
    HIERARCHICAL = 0,
    HCCL_TOPOLOGY = 1,
    GLOBAL_MESH_TREE = 2,
    ADAPTIVE = 3,
    HCCL_TOPOLOGY_HALF_RING = 4,
    HCCL_TOPOLOGY_512K_OPT = 5,
};

enum class Rs2TopologyKind : uint32_t {
    UNKNOWN = 0,
    TWO_BY_EIGHT = 1,
    FOUR_BY_ONE = 2,
    EIGHT_PLUS_FOUR = 3,
};

constexpr Rs2BuildFlavor RS2_BUILD_FLAVOR = Rs2BuildFlavor::HCCL_TOPOLOGY_512K_OPT;
constexpr uint32_t MAX_GLOBAL_RESULT_COUNT = 4;
constexpr uint64_t SMALL_INPUT_BYTES = 1ULL * 1024 * 1024;

struct CcuKernelArgBase {
    ChannelHandle channels[MAX_RANK_SIZE];
    uint32_t channelCount;
};

struct CcuKernelArgReduceScatterTree : public CcuKernelArgBase {
    uint32_t includeLocalInput;
    uint32_t initializeOutput;
    uint32_t scratchSlotBase;
    uint32_t writeOutput;
};

struct CcuKernelArgReduceScatterRing : public CcuKernelArgBase {
    uint32_t rankId;
    uint32_t rankSize;
    uint32_t sendBlockIndices[MAX_RANK_SIZE];
};

struct CcuKernelArgReduceScatterSmall : public CcuKernelArgBase {
    uint32_t includeLocalInput;
    uint32_t scratchSlotBase;
    uint32_t writeOutput;
    uint32_t barrierRoundCount;
    uint32_t barrierSendChannelIndices[MAX_RANK_SIZE];
    uint32_t barrierRecvChannelIndices[MAX_RANK_SIZE];
};

struct CcuKernelInfo {
    char kernelFuncName[64];
    void *kernelFunc;
    void *kernelArg;

private:
    std::shared_ptr<CcuKernelArgBase> kernelArgSmartPtr;

public:
    template <typename T> void SetKernelArg(std::shared_ptr<T> arg)
    {
        kernelArgSmartPtr = std::static_pointer_cast<CcuKernelArgBase>(arg);
        kernelArg = static_cast<void *>(arg.get());
    }
};

struct AlgResourceCtx {
    CommBuffer localBuffer;
    std::vector<ThreadHandle> threads;
    std::vector<CcuKernelHandle> globalKernels;
    std::vector<uint32_t> globalPieceCounts;
    std::vector<uint32_t> globalKernelDieIds;
    std::vector<uint32_t> globalResultSlots;
    uint32_t globalScratchSlotCount = 0;
    std::vector<CcuKernelHandle> localKernels;
    std::vector<uint32_t> localPieceCounts;
    std::vector<CcuKernelHandle> remoteKernels;
    std::vector<uint32_t> remotePieceCounts;
    std::vector<CcuKernelHandle> halfRingKernels;
    std::vector<uint32_t> halfRingKernelDieIds;
    std::vector<CcuKernelHandle> smallKernels;
    std::vector<uint32_t> smallKernelDieIds;
    std::vector<uint32_t> smallResultSlots;
    uint32_t smallScratchSlotCount = 0;
    uint32_t smallDirectKernelIndex = 0;
    CcuKernelHandle smallFinalizeKernel = 0;
    uint32_t topologyKind = static_cast<uint32_t>(Rs2TopologyKind::UNKNOWN);
    uint32_t localGroupSize = 1;

    std::vector<char> Serialize()
    {
        BinaryStream binaryStream;
        binaryStream << localBuffer;
        binaryStream << threads;
        binaryStream << globalKernels;
        binaryStream << globalPieceCounts;
        binaryStream << globalKernelDieIds;
        binaryStream << globalResultSlots;
        binaryStream << globalScratchSlotCount;
        binaryStream << localKernels;
        binaryStream << localPieceCounts;
        binaryStream << remoteKernels;
        binaryStream << remotePieceCounts;
        binaryStream << halfRingKernels;
        binaryStream << halfRingKernelDieIds;
        binaryStream << smallKernels;
        binaryStream << smallKernelDieIds;
        binaryStream << smallResultSlots;
        binaryStream << smallScratchSlotCount;
        binaryStream << smallDirectKernelIndex;
        binaryStream << smallFinalizeKernel;
        binaryStream << topologyKind;
        binaryStream << localGroupSize;
        std::vector<char> result;
        binaryStream.Dump(result);
        return result;
    }

    void DeSerialize(std::vector<char> &data)
    {
        BinaryStream binaryStream(data);
        binaryStream >> localBuffer;
        binaryStream >> threads;
        binaryStream >> globalKernels;
        binaryStream >> globalPieceCounts;
        binaryStream >> globalKernelDieIds;
        binaryStream >> globalResultSlots;
        binaryStream >> globalScratchSlotCount;
        binaryStream >> localKernels;
        binaryStream >> localPieceCounts;
        binaryStream >> remoteKernels;
        binaryStream >> remotePieceCounts;
        binaryStream >> halfRingKernels;
        binaryStream >> halfRingKernelDieIds;
        binaryStream >> smallKernels;
        binaryStream >> smallKernelDieIds;
        binaryStream >> smallResultSlots;
        binaryStream >> smallScratchSlotCount;
        binaryStream >> smallDirectKernelIndex;
        binaryStream >> smallFinalizeKernel;
        binaryStream >> topologyKind;
        binaryStream >> localGroupSize;
    }
};

} // namespace final_small512

namespace final_scratch_lite {
typedef struct {
    void *addr;
    uint64_t size;
} CommBuffer;

enum class Rs2BuildFlavor : uint32_t {
    HIERARCHICAL = 0,
    HCCL_TOPOLOGY = 1,
    GLOBAL_MESH_TREE = 2,
    ADAPTIVE = 3,
    SCRATCH_LITE = 4,
};

enum class Rs2TopologyKind : uint32_t {
    UNKNOWN = 0,
    TWO_BY_EIGHT = 1,
    FOUR_BY_ONE = 2,
    EIGHT_PLUS_FOUR = 3,
};

constexpr Rs2BuildFlavor RS2_BUILD_FLAVOR = Rs2BuildFlavor::SCRATCH_LITE;
constexpr uint32_t MAX_GLOBAL_RESULT_COUNT = 4;

struct CcuKernelArgBase {
    ChannelHandle channels[MAX_RANK_SIZE];
    uint32_t channelCount;
};

struct CcuKernelArgReduceScatterTree : public CcuKernelArgBase {
    uint32_t includeLocalInput;
    uint32_t initializeOutput;
    uint32_t scratchSlotBase;
    uint32_t writeOutput;
};


struct CcuKernelInfo {
    char kernelFuncName[64];
    void *kernelFunc;
    void *kernelArg;

private:
    std::shared_ptr<CcuKernelArgBase> kernelArgSmartPtr;

public:
    template <typename T> void SetKernelArg(std::shared_ptr<T> arg)
    {
        kernelArgSmartPtr = std::static_pointer_cast<CcuKernelArgBase>(arg);
        kernelArg = static_cast<void *>(arg.get());
    }
};

struct AlgResourceCtx {
    CommBuffer localBuffer;
    uint64_t localBufferToken = 0;
    std::vector<ThreadHandle> threads;
    std::vector<CcuKernelHandle> globalKernels;
    std::vector<uint32_t> globalPieceCounts;
    std::vector<uint32_t> globalKernelDieIds;
    std::vector<uint32_t> globalResultSlots;
    uint32_t globalScratchSlotCount = 0;
    std::vector<CcuKernelHandle> localKernels;
    std::vector<uint32_t> localPieceCounts;
    std::vector<CcuKernelHandle> remoteKernels;
    std::vector<uint32_t> remotePieceCounts;
    uint32_t topologyKind = static_cast<uint32_t>(Rs2TopologyKind::UNKNOWN);
    uint32_t localGroupSize = 1;

    std::vector<char> Serialize()
    {
        BinaryStream binaryStream;
        binaryStream << localBuffer;
        binaryStream << localBufferToken;
        binaryStream << threads;
        binaryStream << globalKernels;
        binaryStream << globalPieceCounts;
        binaryStream << globalKernelDieIds;
        binaryStream << globalResultSlots;
        binaryStream << globalScratchSlotCount;
        binaryStream << localKernels;
        binaryStream << localPieceCounts;
        binaryStream << remoteKernels;
        binaryStream << remotePieceCounts;
        binaryStream << topologyKind;
        binaryStream << localGroupSize;
        std::vector<char> result;
        binaryStream.Dump(result);
        return result;
    }

    void DeSerialize(std::vector<char> &data)
    {
        BinaryStream binaryStream(data);
        binaryStream >> localBuffer;
        binaryStream >> localBufferToken;
        binaryStream >> threads;
        binaryStream >> globalKernels;
        binaryStream >> globalPieceCounts;
        binaryStream >> globalKernelDieIds;
        binaryStream >> globalResultSlots;
        binaryStream >> globalScratchSlotCount;
        binaryStream >> localKernels;
        binaryStream >> localPieceCounts;
        binaryStream >> remoteKernels;
        binaryStream >> remotePieceCounts;
        binaryStream >> topologyKind;
        binaryStream >> localGroupSize;
    }
};

} // namespace final_scratch_lite

namespace final_epoch_cache {
typedef struct {
    void *addr;
    uint64_t size;
} CommBuffer;

enum class Rs2BuildFlavor : uint32_t {
    HIERARCHICAL = 0,
    HCCL_TOPOLOGY = 1,
    GLOBAL_MESH_TREE = 2,
    ADAPTIVE = 3,
    SCRATCH_LITE = 4,
    EPOCH_CACHE = 5,
};

enum class Rs2TopologyKind : uint32_t {
    UNKNOWN = 0,
    TWO_BY_EIGHT = 1,
    FOUR_BY_ONE = 2,
    EIGHT_PLUS_FOUR = 3,
};

constexpr Rs2BuildFlavor RS2_BUILD_FLAVOR = Rs2BuildFlavor::EPOCH_CACHE;
constexpr uint32_t MAX_GLOBAL_RESULT_COUNT = 4;

struct CcuKernelArgBase {
    ChannelHandle channels[MAX_RANK_SIZE];
    uint32_t channelCount;
};

struct CcuKernelArgReduceScatterTree : public CcuKernelArgBase {
    uint32_t includeLocalInput;
    uint32_t initializeOutput;
    uint32_t scratchSlotBase;
    uint32_t writeOutput;
};


struct CcuKernelInfo {
    char kernelFuncName[64];
    void *kernelFunc;
    void *kernelArg;

private:
    std::shared_ptr<CcuKernelArgBase> kernelArgSmartPtr;

public:
    template <typename T> void SetKernelArg(std::shared_ptr<T> arg)
    {
        kernelArgSmartPtr = std::static_pointer_cast<CcuKernelArgBase>(arg);
        kernelArg = static_cast<void *>(arg.get());
    }
};

struct AlgResourceCtx {
    CommBuffer localBuffer;
    uint64_t localBufferToken = 0;
    uint64_t inputToken = 0;
    uint64_t outputToken = 0;
    std::vector<ThreadHandle> threads;
    std::vector<CcuKernelHandle> globalKernels;
    std::vector<uint32_t> globalPieceCounts;
    std::vector<uint32_t> globalKernelDieIds;
    std::vector<uint32_t> globalResultSlots;
    uint32_t globalScratchSlotCount = 0;
    std::vector<CcuKernelHandle> localKernels;
    std::vector<uint32_t> localPieceCounts;
    std::vector<CcuKernelHandle> remoteKernels;
    std::vector<uint32_t> remotePieceCounts;
    uint32_t topologyKind = static_cast<uint32_t>(Rs2TopologyKind::UNKNOWN);
    uint32_t localGroupSize = 1;

    std::vector<char> Serialize()
    {
        BinaryStream binaryStream;
        binaryStream << localBuffer;
        binaryStream << localBufferToken;
        binaryStream << inputToken;
        binaryStream << outputToken;
        binaryStream << threads;
        binaryStream << globalKernels;
        binaryStream << globalPieceCounts;
        binaryStream << globalKernelDieIds;
        binaryStream << globalResultSlots;
        binaryStream << globalScratchSlotCount;
        binaryStream << localKernels;
        binaryStream << localPieceCounts;
        binaryStream << remoteKernels;
        binaryStream << remotePieceCounts;
        binaryStream << topologyKind;
        binaryStream << localGroupSize;
        std::vector<char> result;
        binaryStream.Dump(result);
        return result;
    }

    void DeSerialize(std::vector<char> &data)
    {
        BinaryStream binaryStream(data);
        binaryStream >> localBuffer;
        binaryStream >> localBufferToken;
        binaryStream >> inputToken;
        binaryStream >> outputToken;
        binaryStream >> threads;
        binaryStream >> globalKernels;
        binaryStream >> globalPieceCounts;
        binaryStream >> globalKernelDieIds;
        binaryStream >> globalResultSlots;
        binaryStream >> globalScratchSlotCount;
        binaryStream >> localKernels;
        binaryStream >> localPieceCounts;
        binaryStream >> remoteKernels;
        binaryStream >> remotePieceCounts;
        binaryStream >> topologyKind;
        binaryStream >> localGroupSize;
    }
};

} // namespace final_epoch_cache

namespace final_direct_pipeline {
typedef struct {
    void *addr;
    uint64_t size;
} CommBuffer;

enum class Rs2BuildFlavor : uint32_t {
    HIERARCHICAL = 0,
    HCCL_TOPOLOGY = 1,
    GLOBAL_MESH_TREE = 2,
    ADAPTIVE = 3,
    SCRATCH_LITE = 4,
    EPOCH_CACHE = 5,
    DIRECT_PIPELINE = 6,
};

enum class Rs2TopologyKind : uint32_t {
    UNKNOWN = 0,
    TWO_BY_EIGHT = 1,
    FOUR_BY_ONE = 2,
    EIGHT_PLUS_FOUR = 3,
};

constexpr Rs2BuildFlavor RS2_BUILD_FLAVOR = Rs2BuildFlavor::DIRECT_PIPELINE;
constexpr uint32_t MAX_GLOBAL_RESULT_COUNT = 4;
constexpr uint32_t DIRECT_PIPELINE_NOTIFY_COUNT = 4;

struct CcuKernelArgBase {
    ChannelHandle channels[MAX_RANK_SIZE];
    uint32_t channelCount;
};

struct CcuKernelArgReduceScatterTree : public CcuKernelArgBase {
    uint32_t includeLocalInput;
    uint32_t initializeOutput;
    uint32_t scratchSlotBase;
    uint32_t writeOutput;
    uint32_t useFirstLayerPairReady;
};


struct CcuKernelInfo {
    char kernelFuncName[64];
    void *kernelFunc;
    void *kernelArg;

private:
    std::shared_ptr<CcuKernelArgBase> kernelArgSmartPtr;

public:
    template <typename T> void SetKernelArg(std::shared_ptr<T> arg)
    {
        kernelArgSmartPtr = std::static_pointer_cast<CcuKernelArgBase>(arg);
        kernelArg = static_cast<void *>(arg.get());
    }
};

struct AlgResourceCtx {
    CommBuffer localBuffer;
    uint64_t localBufferToken = 0;
    uint64_t inputToken = 0;
    uint64_t outputToken = 0;
    std::vector<ThreadHandle> threads;
    std::vector<CcuKernelHandle> globalKernels;
    std::vector<uint32_t> globalPieceCounts;
    std::vector<uint32_t> globalKernelDieIds;
    std::vector<uint32_t> globalResultSlots;
    uint32_t globalScratchSlotCount = 0;
    std::vector<CcuKernelHandle> localKernels;
    std::vector<uint32_t> localPieceCounts;
    std::vector<CcuKernelHandle> remoteKernels;
    std::vector<uint32_t> remotePieceCounts;
    uint32_t topologyKind = static_cast<uint32_t>(Rs2TopologyKind::UNKNOWN);
    uint32_t localGroupSize = 1;

    std::vector<char> Serialize()
    {
        BinaryStream binaryStream;
        binaryStream << localBuffer;
        binaryStream << localBufferToken;
        binaryStream << inputToken;
        binaryStream << outputToken;
        binaryStream << threads;
        binaryStream << globalKernels;
        binaryStream << globalPieceCounts;
        binaryStream << globalKernelDieIds;
        binaryStream << globalResultSlots;
        binaryStream << globalScratchSlotCount;
        binaryStream << localKernels;
        binaryStream << localPieceCounts;
        binaryStream << remoteKernels;
        binaryStream << remotePieceCounts;
        binaryStream << topologyKind;
        binaryStream << localGroupSize;
        std::vector<char> result;
        binaryStream.Dump(result);
        return result;
    }

    void DeSerialize(std::vector<char> &data)
    {
        BinaryStream binaryStream(data);
        binaryStream >> localBuffer;
        binaryStream >> localBufferToken;
        binaryStream >> inputToken;
        binaryStream >> outputToken;
        binaryStream >> threads;
        binaryStream >> globalKernels;
        binaryStream >> globalPieceCounts;
        binaryStream >> globalKernelDieIds;
        binaryStream >> globalResultSlots;
        binaryStream >> globalScratchSlotCount;
        binaryStream >> localKernels;
        binaryStream >> localPieceCounts;
        binaryStream >> remoteKernels;
        binaryStream >> remotePieceCounts;
        binaryStream >> topologyKind;
        binaryStream >> localGroupSize;
    }
};

} // namespace final_direct_pipeline

namespace final_adaptive41 {
typedef struct {
    void *addr;
    uint64_t size;
} CommBuffer;

enum class Rs2BuildFlavor : uint32_t {
    HIERARCHICAL = 0,
    HCCL_TOPOLOGY = 1,
    GLOBAL_MESH_TREE = 2,
    ADAPTIVE = 3,
    FOUR_BY_ONE_RING = 4,
};

enum class Rs2TopologyKind : uint32_t {
    UNKNOWN = 0,
    TWO_BY_EIGHT = 1,
    FOUR_BY_ONE = 2,
    EIGHT_PLUS_FOUR = 3,
};

constexpr Rs2BuildFlavor RS2_BUILD_FLAVOR = Rs2BuildFlavor::FOUR_BY_ONE_RING;
constexpr uint32_t MAX_GLOBAL_RESULT_COUNT = 4;

struct CcuKernelArgBase {
    ChannelHandle channels[MAX_RANK_SIZE];
    uint32_t channelCount;
};

struct CcuKernelArgReduceScatterTree : public CcuKernelArgBase {
    uint32_t includeLocalInput;
    uint32_t initializeOutput;
    uint32_t scratchSlotBase;
    uint32_t writeOutput;
};

struct CcuKernelArgReduceScatterRing : public CcuKernelArgBase {
    uint32_t rankId;
    uint32_t rankSize;
    uint32_t sendBlockIndices[MAX_RANK_SIZE];
};

struct CcuKernelArgEightPlusFourPreReduce : public CcuKernelArgBase {
    uint32_t groupRank;
};

struct CcuKernelArgEightPlusFourExchange : public CcuKernelArgBase {
    uint32_t groupIndex;
    uint32_t groupRank;
    uint32_t phase;
};

struct CcuKernelArgEightPlusFourRingEdge : public CcuKernelArgBase {
    uint32_t groupIndex;
    uint32_t groupRank;
};

struct CcuKernelInfo {
    char kernelFuncName[64];
    void *kernelFunc;
    void *kernelArg;

private:
    std::shared_ptr<CcuKernelArgBase> kernelArgSmartPtr;

public:
    template <typename T> void SetKernelArg(std::shared_ptr<T> arg)
    {
        kernelArgSmartPtr = std::static_pointer_cast<CcuKernelArgBase>(arg);
        kernelArg = static_cast<void *>(arg.get());
    }
};

struct AlgResourceCtx {
    CommBuffer localBuffer;
    std::vector<ThreadHandle> threads;
    std::vector<CcuKernelHandle> globalKernels;
    std::vector<uint32_t> globalPieceCounts;
    std::vector<uint32_t> globalKernelDieIds;
    std::vector<uint32_t> globalResultSlots;
    uint32_t globalScratchSlotCount = 0;
    std::vector<CcuKernelHandle> localKernels;
    std::vector<uint32_t> localPieceCounts;
    std::vector<CcuKernelHandle> remoteKernels;
    std::vector<uint32_t> remotePieceCounts;
    CcuKernelHandle ringKernel = 0;
    uint32_t ringKernelDieId = 0;
    CcuKernelHandle eightPlusFourPreReduceKernel = 0;
    uint32_t eightPlusFourPreReduceKernelDieId = 0;
    CcuKernelHandle eightPlusFourPairKernel = 0;
    uint32_t eightPlusFourPairKernelDieId = 0;
    CcuKernelHandle eightPlusFourRemoteKernel = 0;
    uint32_t eightPlusFourRemoteKernelDieId = 0;
    std::vector<CcuKernelHandle> eightPlusFourRingEdgeKernels;
    std::vector<uint32_t> eightPlusFourRingEdgeKernelDieIds;
    uint32_t topologyKind = static_cast<uint32_t>(Rs2TopologyKind::UNKNOWN);
    uint32_t localGroupSize = 1;

    std::vector<char> Serialize()
    {
        BinaryStream binaryStream;
        binaryStream << localBuffer;
        binaryStream << threads;
        binaryStream << globalKernels;
        binaryStream << globalPieceCounts;
        binaryStream << globalKernelDieIds;
        binaryStream << globalResultSlots;
        binaryStream << globalScratchSlotCount;
        binaryStream << localKernels;
        binaryStream << localPieceCounts;
        binaryStream << remoteKernels;
        binaryStream << remotePieceCounts;
        binaryStream << ringKernel;
        binaryStream << ringKernelDieId;
        binaryStream << eightPlusFourPreReduceKernel;
        binaryStream << eightPlusFourPreReduceKernelDieId;
        binaryStream << eightPlusFourPairKernel;
        binaryStream << eightPlusFourPairKernelDieId;
        binaryStream << eightPlusFourRemoteKernel;
        binaryStream << eightPlusFourRemoteKernelDieId;
        binaryStream << eightPlusFourRingEdgeKernels;
        binaryStream << eightPlusFourRingEdgeKernelDieIds;
        binaryStream << topologyKind;
        binaryStream << localGroupSize;
        std::vector<char> result;
        binaryStream.Dump(result);
        return result;
    }

    void DeSerialize(std::vector<char> &data)
    {
        BinaryStream binaryStream(data);
        binaryStream >> localBuffer;
        binaryStream >> threads;
        binaryStream >> globalKernels;
        binaryStream >> globalPieceCounts;
        binaryStream >> globalKernelDieIds;
        binaryStream >> globalResultSlots;
        binaryStream >> globalScratchSlotCount;
        binaryStream >> localKernels;
        binaryStream >> localPieceCounts;
        binaryStream >> remoteKernels;
        binaryStream >> remotePieceCounts;
        binaryStream >> ringKernel;
        binaryStream >> ringKernelDieId;
        binaryStream >> eightPlusFourPreReduceKernel;
        binaryStream >> eightPlusFourPreReduceKernelDieId;
        binaryStream >> eightPlusFourPairKernel;
        binaryStream >> eightPlusFourPairKernelDieId;
        binaryStream >> eightPlusFourRemoteKernel;
        binaryStream >> eightPlusFourRemoteKernelDieId;
        binaryStream >> eightPlusFourRingEdgeKernels;
        binaryStream >> eightPlusFourRingEdgeKernelDieIds;
        binaryStream >> topologyKind;
        binaryStream >> localGroupSize;
    }
};

} // namespace final_adaptive41

#endif // OPS_HCCL_CUSTOM_H
