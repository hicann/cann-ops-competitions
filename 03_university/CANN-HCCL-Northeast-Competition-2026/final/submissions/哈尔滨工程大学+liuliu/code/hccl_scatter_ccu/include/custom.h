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
#include <vector>
#include <cstring>
#include "common.h"

constexpr uint64_t SCATTER_CCU_CHUNK = 64ULL * 1024 * 1024;
constexpr uint64_t SCATTER_FOUR_RANK_DIRECT_CHUNK = 128ULL * 1024 * 1024;
static_assert(SCATTER_FOUR_RANK_DIRECT_CHUNK <= MAX_DATA_SIZE, "Direct transfer limit");
constexpr uint64_t SCATTER_RELAY_THRESHOLD = 16ULL * 1024 * 1024;
constexpr uint32_t SCATTER_CCU_TASK_ARGS = 7;
constexpr uint32_t SCATTER_RECEIVE_TASK_ARGS = 2;
constexpr uint32_t SCATTER_COMPACT_TASK_ARGS = 8;
constexpr uint32_t SCATTER_RELAY_TASK_ARGS = 14;
constexpr uint32_t SCATTER_ROLE_ROOT = 0;
constexpr uint32_t SCATTER_ROLE_HELPER = 1;
constexpr uint32_t SCATTER_ROLE_TARGET = 2;
constexpr uint32_t SCATTER_ROLE_DIRECT = 3;
static_assert(SCATTER_CCU_CHUNK <= MAX_DATA_SIZE && SCATTER_CCU_CHUNK % 4 == 0,
    "CCU chunk must satisfy the transfer limit and FP32 alignment");
struct CommBuffer {
    void *addr = nullptr;
    uint64_t size = 0;
};
struct CcuKernelArgBase {
    ChannelHandle channels[MAX_RANK_SIZE]{};
    uint32_t peerRanks[MAX_RANK_SIZE]{};
    uint32_t channelCount = 0;
    uint32_t rankSize = 0;
    uint32_t myRank = 0;
    uint32_t rootRank = 0;
    bool singleChunk = false;
    uint32_t role = SCATTER_ROLE_DIRECT;
    uint32_t partnerRank = INVALID_VALUE_RANKID;
    uint32_t groupIndex = 0;
    uint32_t rootGroup = INVALID_VALUE_RANKID;
    uint32_t partnerGroup = INVALID_VALUE_RANKID;
    uint32_t peerRoles[MAX_RANK_SIZE]{};
    uint32_t partners[MAX_RANK_SIZE]{};
};
struct AlgResourceCtx {
    CommBuffer localBuffer;
    // Last kernel is the local overlap-copy kernel; preceding kernels are die groups.
    std::vector<CcuKernelHandle> ccuKernels;
    std::vector<uint64_t> groupMasks;
    // Worker i executes die group i+1; group zero stays on the user stream.
    std::vector<ThreadHandle> workers;
    bool relayEnabled = false;
    std::vector<char> Serialize()
    {
        if (groupMasks.empty() || ccuKernels.size() != groupMasks.size() + 1
            || workers.size() + 1 != groupMasks.size()) {
            return {};
        }
        std::vector<uint64_t> fields = {0x534343553400ULL, reinterpret_cast<uint64_t>(localBuffer.addr),
            localBuffer.size, groupMasks.size(), relayEnabled ? 1ULL : 0ULL};
        fields.insert(fields.end(), ccuKernels.begin(), ccuKernels.end());
        fields.insert(fields.end(), groupMasks.begin(), groupMasks.end());
        fields.insert(fields.end(), workers.begin(), workers.end());
        std::vector<char> data(fields.size() * sizeof(uint64_t));
        std::memcpy(data.data(), fields.data(), data.size());
        return data;
    }
    void DeSerialize(std::vector<char> &data)
    {
        ccuKernels.clear();
        groupMasks.clear();
        workers.clear();
        relayEnabled = false;
        if (data.size() < 8 * sizeof(uint64_t) || data.size() % sizeof(uint64_t) != 0) {
            return;
        }
        std::vector<uint64_t> fields(data.size() / sizeof(uint64_t));
        std::memcpy(fields.data(), data.data(), data.size());
        const uint64_t groups = fields[3];
        if (fields[0] != 0x534343553400ULL || groups == 0 || groups > MAX_RANK_SIZE || fields[4] > 1
            || fields.size() != 5 + groups * 3) {
            return;
        }
        localBuffer = {reinterpret_cast<void *>(fields[1]), fields[2]};
        relayEnabled = fields[4] != 0;
        ccuKernels.assign(fields.begin() + 5, fields.begin() + 6 + groups);
        groupMasks.assign(fields.begin() + 6 + groups, fields.begin() + 6 + groups * 2);
        workers.assign(fields.begin() + 6 + groups * 2, fields.end());
    }
};
#endif
