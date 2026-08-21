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

constexpr uint32_t FOREST_STAGE_COUNT = 2;
constexpr uint32_t FOREST_RADIX = 4;
constexpr uint32_t FOREST_RING_COUNT = 4;
constexpr uint64_t DIRECT_MESSAGE_LIMIT = 1024ULL * 1024ULL;

enum class ForestPhase : uint32_t {
    DIRECT = 0,
    SEED = 1,
    LEVEL_ZERO = 2,
    LEVEL_ONE = 3,
    RING = 4,
};

struct ForestKernelConfig {
    ChannelHandle channels[MAX_RANK_SIZE]{};
    uint32_t peerRanks[MAX_RANK_SIZE]{};
    uint32_t channelCount{0};
    uint32_t rankSize{0};
    uint32_t rankId{INVALID_VALUE_RANKID};
    uint32_t rootRank{INVALID_VALUE_RANKID};
    uint32_t directMode{0};
    uint64_t totalBytes{0};
    uint64_t laneOffsets[MAX_RANK_SIZE]{};
    uint64_t laneBytes[MAX_RANK_SIZE]{};
    uint32_t parentRanks[FOREST_STAGE_COUNT][MAX_RANK_SIZE]{};
    uint32_t childRankMasks[FOREST_STAGE_COUNT][MAX_RANK_SIZE]{};
    uint32_t ringNextRanks[FOREST_RING_COUNT]{};
    uint32_t ringPreviousRanks[FOREST_RING_COUNT]{};
};

class ForestKernelRecord {
public:
    char name[64]{};
    void *function{nullptr};
    void *argument{nullptr};

    void Keep(const std::shared_ptr<ForestKernelConfig> &config)
    {
        owner_ = config;
        argument = config.get();
    }

private:
    std::shared_ptr<ForestKernelConfig> owner_;
};

struct ForestResourceContext {
    std::vector<ThreadHandle> auxiliaryThreads;
    std::vector<CcuKernelHandle> kernels;
    std::vector<uint32_t> directActive;
    std::vector<uint32_t> ringOrders;

    std::vector<char> Encode()
    {
        BinaryStream stream;
        stream << auxiliaryThreads << kernels << directActive << ringOrders;
        std::vector<char> bytes;
        stream.Dump(bytes);
        return bytes;
    }

    void Decode(std::vector<char> &bytes)
    {
        BinaryStream stream(bytes);
        stream >> auxiliaryThreads >> kernels >> directActive >> ringOrders;
    }
};

#endif // OPS_HCCL_CUSTOM_H
