/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 */

#ifndef OPS_HCCL_CUSTOM_H
#define OPS_HCCL_CUSTOM_H

#include <cstdint>
#include <memory>
#include <vector>

#include <hccl/hccl_res.h>
#include <hccl/hccl_types.h>

#include "binary_stream.h"
#include "common.h"

constexpr uint32_t BROADCAST_MAX_WINDOWS = 4;
constexpr uint64_t BROADCAST_WINDOW_BYTES = 128ULL * 1024ULL * 1024ULL;

enum class BroadcastMode : uint32_t {
    ONE_SHOT = 0,
    SCATTER_ALLGATHER = 1,
};

enum class BroadcastPhase : uint32_t {
    ONE_SHOT = 0,
    SCATTER = 1,
    ALLGATHER = 2,
};

typedef struct {
    void *addr;
    uint64_t size;
} CommBuffer;

struct BroadcastWindow {
    uint64_t offset{0};
    uint64_t bytes{0};
};

struct BroadcastKernelMeta {
    uint32_t peerCount{0};
    uint32_t peerRanks[MAX_RANK_SIZE]{};
};

struct CcuKernelArgBase {
    ChannelHandle channels[MAX_RANK_SIZE]{};
    uint32_t channelCount{0};
};

struct CcuKernelArgBroadcast : public CcuKernelArgBase {
    uint32_t rankSize{0};
    uint32_t rankId{INVALID_VALUE_RANKID};
    uint32_t kernelIndex{0};
    uint32_t kernelCount{0};
    uint32_t mode{static_cast<uint32_t>(BroadcastMode::ONE_SHOT)};
    uint32_t windowCount{0};
    uint32_t peerRanks[MAX_RANK_SIZE]{};
    uint32_t rankKernelIndices[MAX_RANK_SIZE]{};
    uint64_t windowBytes[BROADCAST_MAX_WINDOWS]{};
    uint64_t ownerSliceOffsets[BROADCAST_MAX_WINDOWS]{};
    uint64_t ownerSliceBytes[BROADCAST_MAX_WINDOWS]{};
    uint64_t peerSliceOffsets[BROADCAST_MAX_WINDOWS][MAX_RANK_SIZE]{};
    uint64_t peerSliceBytes[BROADCAST_MAX_WINDOWS][MAX_RANK_SIZE]{};
};

struct CcuKernelInfo {
    char kernelFuncName[64]{};
    void *kernelFunc{nullptr};
    void *kernelArg{nullptr};

private:
    std::shared_ptr<CcuKernelArgBase> kernelArgSmartPtr;

public:
    template <typename T> void setKernelArg(std::shared_ptr<T> arg)
    {
        kernelArgSmartPtr = std::static_pointer_cast<CcuKernelArgBase>(arg);
        kernelArg = static_cast<void *>(arg.get());
    }
};

struct AlgResourceCtx {
    uint32_t mode{static_cast<uint32_t>(BroadcastMode::ONE_SHOT)};
    std::vector<BroadcastWindow> windows;
    std::vector<ThreadHandle> extraThreads;
    std::vector<CcuKernelHandle> ccuKernels;
    std::vector<BroadcastKernelMeta> kernelMeta;

    std::vector<char> Serialize()
    {
        BinaryStream binaryStream;
        binaryStream << mode;
        binaryStream << windows;
        binaryStream << extraThreads;
        binaryStream << ccuKernels;
        binaryStream << kernelMeta;
        std::vector<char> result;
        binaryStream.Dump(result);
        return result;
    }

    void DeSerialize(std::vector<char> &data)
    {
        BinaryStream binaryStream(data);
        binaryStream >> mode;
        binaryStream >> windows;
        binaryStream >> extraThreads;
        binaryStream >> ccuKernels;
        binaryStream >> kernelMeta;
    }
};

#endif // OPS_HCCL_CUSTOM_H
