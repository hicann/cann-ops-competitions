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

#include <algorithm>
#include <memory>
#include <vector>

#include <hccl/hccl_types.h>
#include <hccl/hccl_res.h>

#include "binary_stream.h"
#include "common.h"

typedef struct {
    void *addr;
    uint64_t size;
} CommBuffer;

struct CcuKernelArgBase {
    ChannelHandle channels[MAX_RANK_SIZE];
    uint32_t channelCount;
};

inline uint32_t BuildScatterRelayPairs(const std::vector<uint32_t> &serverIds, uint32_t root,
    std::vector<uint32_t> &relayTargetByRank, std::vector<uint32_t> &relaySourceByRank)
{
    const uint32_t rankSize = static_cast<uint32_t>(serverIds.size());
    relayTargetByRank.assign(rankSize, INVALID_VALUE_RANKID);
    relaySourceByRank.assign(rankSize, INVALID_VALUE_RANKID);
    if (root >= rankSize || rankSize == 0) {
        return 0;
    }

    const uint32_t rootServer = serverIds[root];
    std::vector<uint32_t> localRelays;
    std::vector<uint32_t> remoteServers;
    for (uint32_t rank = 0; rank < rankSize; ++rank) {
        if (rank != root && serverIds[rank] == rootServer) {
            localRelays.push_back(rank);
        } else if (serverIds[rank] != rootServer &&
            std::find(remoteServers.begin(), remoteServers.end(), serverIds[rank]) == remoteServers.end()) {
            remoteServers.push_back(serverIds[rank]);
        }
    }
    std::sort(localRelays.begin(), localRelays.end());
    std::sort(remoteServers.begin(), remoteServers.end());

    // Order remote targets round-robin across remote servers. 2x8 has one
    // remote server; 4x3 has three and this avoids concentrating the relay
    // targets on only one of them.
    std::vector<uint32_t> remoteTargets;
    for (uint32_t offset = 0; remoteTargets.size() < rankSize; ++offset) {
        bool appended = false;
        for (uint32_t serverId : remoteServers) {
            uint32_t seen = 0;
            for (uint32_t rank = 0; rank < rankSize; ++rank) {
                if (serverIds[rank] != serverId) {
                    continue;
                }
                if (seen == offset) {
                    remoteTargets.push_back(rank);
                    appended = true;
                    break;
                }
                ++seen;
            }
        }
        if (!appended) {
            break;
        }
    }

    const uint32_t pairNum = static_cast<uint32_t>(std::min(localRelays.size(), remoteTargets.size()));
    for (uint32_t i = 0; i < pairNum; ++i) {
        relayTargetByRank[localRelays[i]] = remoteTargets[i];
        relaySourceByRank[remoteTargets[i]] = localRelays[i];
    }
    return pairNum;
}

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

struct AlgResourceCtx {
    ThreadHandle ccuThread;            ///< CCU通信引擎上的thread资源
    CommBuffer localBuffer;            ///< 本端HCCL通信内存
    std::vector<ThreadHandle> threads; ///< CCU通信引擎上的thread资源
    std::vector<CcuKernelHandle> ccuKernels;
    // Direct-only metadata. Bit r in directKernelPeerMasks[k] means that
    // ccuKernels[k] owns the channel to peer rank r.
    std::vector<uint32_t> directKernelPeerMasks;
    uint32_t relayNumerator = 0;
    uint32_t relayDenominator = 1;
    std::vector<uint32_t> relayServerIds;

    // 序列化
    std::vector<char> Serialize()
    {
        BinaryStream binaryStream;
        binaryStream << ccuThread;
        binaryStream << localBuffer;
        binaryStream << threads;
        binaryStream << ccuKernels;
        binaryStream << directKernelPeerMasks;
        binaryStream << relayNumerator;
        binaryStream << relayDenominator;
        binaryStream << relayServerIds;
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
        binaryStream >> directKernelPeerMasks;
        binaryStream >> relayNumerator;
        binaryStream >> relayDenominator;
        binaryStream >> relayServerIds;
    }
};

#endif // OPS_HCCL_CUSTOM_H
