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
#include <hccl/hccl_types.h>
#include <hccl/hccl_res.h>

#include "binary_stream.h"
#include "common.h"

typedef struct {
    void *addr = nullptr;
    uint64_t size = 0;
} CommBuffer;

struct CcuKernelArgBase {
    ChannelHandle channels[MAX_RANK_SIZE]{};
    uint32_t channelCount = 0;
};

struct CcuDirectKernelArg : public CcuKernelArgBase {
    uint32_t rankSize = 0;
    uint32_t rankId = INVALID_VALUE_RANKID;
    uint32_t peerRanks[MAX_RANK_SIZE]{};
};

// 4x1和8+4机内阶段共用的Source-Owned Push静态参数。
// handleLocalCopy为true时，sendBuf到本Rank输出块的拷贝与网络Write在同一个Kernel中完成。
struct CcuRotatingPushKernelArg : public CcuKernelArgBase {
    uint32_t rankSize = 0;
    uint32_t rankId = INVALID_VALUE_RANKID;
    uint32_t peerRanks[MAX_RANK_SIZE]{};
    bool handleLocalCopy = false;
    // V27 legacy mode, retained for compatibility with the proven latency code.
    bool latencyPipeline = false;
    // V30D bandwidth-only controls.  Their defaults preserve the V27 graph.
    bool exchangeResources = true;
    bool progressiveAddressReady = false;
    bool finalSync = true;
    bool combineSliceWaits = false;
    bool reuseNetworkEventForLocalCopy = false;
    bool useStaticGeometry = false;
    uint64_t staticRankOutputOffset = 0;
    uint64_t staticFirstBytes = 0;
    uint64_t staticSecondBytes = 0;
};

// 512KB专用的轻量Direct-Push静态参数。每个实例中的Channel严格属于
// 同一个IO Die；4x1只注册一个实例，8+4分别在Intra/Cross Die注册一个实例。
// 与大消息Kernel完全解耦，不携带Chunk、Seed、Relay和phaseMode状态。
struct CcuLatencyPushKernelArg : public CcuKernelArgBase {
    uint32_t rankSize = 0;
    uint32_t rankId = INVALID_VALUE_RANKID;
    uint32_t peerRanks[MAX_RANK_SIZE]{};
    bool handleLocalCopy = false;
};

// 4x1 exact-512 KiB registered source-owned push.  The cold graph republishes
// the output identity once; the warm graph reuses the channel XN variables and
// launches with zero runtime arguments.  Both graphs retain V27's direct Write
// data path and combine the three network completions plus OwnCopy into one
// EventWait.
struct CcuRegistered4x1PushKernelArg : public CcuKernelArgBase {
    uint32_t rankSize = 0;
    uint32_t rankId = INVALID_VALUE_RANKID;
    uint32_t peerRanks[MAX_RANK_SIZE]{};
    uint64_t inputAddr = 0;
    uint64_t outputAddr = 0;
    uint64_t inputToken = 0;
    uint64_t outputToken = 0;
    uint64_t rankOutputOffset = 0;
    uint64_t transferBytes = 0;
    bool publishResources = true;
    bool handleLocalCopy = true;
};

// Final-output 512 KiB resident micro-kernel.  A separate instance is
// registered for every real IO Die, so no Channel ever crosses the Die bound.
// The cold instance publishes its fixed recv buffer identity; the steady
// instance has no LoadArg and reuses the channel XN variables.
struct CcuOutput512StaticPushKernelArg : public CcuKernelArgBase {
    uint32_t rankSize = 0;
    uint32_t rankId = INVALID_VALUE_RANKID;
    uint32_t peerRanks[MAX_RANK_SIZE]{};
    uint64_t inputAddr = 0;
    uint64_t outputAddr = 0;
    uint64_t inputToken = 0;
    uint64_t outputToken = 0;
    uint64_t rankOutputOffset = 0;
    uint64_t transferBytes = 0;
    bool publishResources = true;
    bool handleLocalCopy = false;
};

// 4x1 CLOS公共Die专用Kernel。Host保证三个Channel来自同一网络层和同一IO Die。
// latency版本使用1个Chunk并融合OwnCopy；bandwidth版本使用4个Chunk，
// OwnCopy由独立Thread并行执行。
struct CcuClos4x1PushKernelArg : public CcuKernelArgBase {
    uint32_t rankSize = 0;
    uint32_t rankId = INVALID_VALUE_RANKID;
    uint32_t peerRanks[MAX_RANK_SIZE]{};
    uint32_t chunkCount = 0;
    bool handleLocalCopy = false;
    bool latencyPipeline = false;
};

// 8+4跨Server阶段。经Host严格验证后，rank[0,7]为大Server，
// rank[8,11]为小Server；小Server的每个源Rank对应两个不同条带Seed。
struct CcuDualSeedCrossKernelArg : public CcuKernelArgBase {
    uint32_t rankSize = 0;
    uint32_t rankId = INVALID_VALUE_RANKID;
    uint32_t peerRanks[MAX_RANK_SIZE]{};
    bool sourceIsSmallServer = false;
    uint32_t seedRanks[2]{};
    bool latencyPipeline = false;
};

enum class CcuDualSeedPhaseKind : uint32_t {
    SEED = 0,
    DIRECT = 1,
};

// Bandwidth-only 8+4 cross-server static kernels.  The proven V27 latency
// micro-kernels do not use this type.
struct CcuWideDualSeedCrossKernelArg : public CcuDualSeedCrossKernelArg {
    CcuDualSeedPhaseKind phaseKind = CcuDualSeedPhaseKind::SEED;
    bool exchangeResources = false;
    bool finalSync = false;
    uint32_t incomingSeedRankCount = 0;
    uint32_t incomingSeedRanks[2]{};
};

// 8+4大Server机内Relay。四个小Server源各分配两个Seed：
// rank8->{0,1}, rank9->{2,3}, rank10->{4,5}, rank11->{6,7}。
struct CcuDualSeedRelayKernelArg : public CcuKernelArgBase {
    uint32_t rankSize = 0;
    uint32_t rankId = INVALID_VALUE_RANKID;
    uint32_t peerRanks[MAX_RANK_SIZE]{};
    bool enabled = false;
    bool relayOnLargeServer = false;
    uint32_t sourceRankCount = 0;
    uint32_t sourceRanks[2]{};
    uint32_t stripeIndices[2]{};
};

// ccu kernel register所需信息
struct CcuKernelInfo {
    // kernel名称
    char kernelFuncName[64]{};
    // kernel函数
    void *kernelFunc = nullptr;
    // KernelArg实例指针
    void *kernelArg = nullptr;

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
    ThreadHandle ccuThread{};          ///< CCU通信引擎上的thread资源
    ThreadHandle ownCopyThread{};      ///< 与网络传输并行的本Rank输出拷贝thread
    CommBuffer localBuffer{};          ///< 本端HCCL通信内存
    std::vector<ThreadHandle> threads; ///< 每个Die通信组对应的thread资源
    std::vector<CcuKernelHandle> ccuKernels;
    std::vector<std::vector<uint32_t>> peerRanksByGroup; ///< Host构造绝对目标地址所需的Peer映射
    uint32_t topologyType = 0;
    uint32_t algorithmMode = 0; ///< 0=V10 Pull, 1=4x1 Common-Die, 2=8+4 Relay, 3=4x1 Dual-Plane, 4=2x8 V22, 5=8+4 Latency, 6=8+4 Wide
    uint32_t hasRegistered4x1Exact512 = 0;
    uint64_t registered4x1InputAddr = 0;
    uint64_t registered4x1OutputAddr = 0;
    uint64_t registered4x1InputToken = 0;
    uint64_t registered4x1OutputToken = 0;
    uint32_t hasOutput512StaticSession = 0;
    uint64_t output512InputAddr = 0;
    uint64_t output512OutputAddr = 0;
    uint64_t output512InputToken = 0;
    uint64_t output512OutputToken = 0;
    uint64_t output512RankBytes = 0;

    // 序列化
    std::vector<char> Serialize()
    {
        BinaryStream binaryStream;
        binaryStream << ccuThread;
        binaryStream << ownCopyThread;
        binaryStream << localBuffer;
        binaryStream << threads;
        binaryStream << ccuKernels;
        binaryStream << peerRanksByGroup;
        binaryStream << topologyType;
        binaryStream << algorithmMode;
        binaryStream << hasRegistered4x1Exact512;
        binaryStream << registered4x1InputAddr;
        binaryStream << registered4x1OutputAddr;
        binaryStream << registered4x1InputToken;
        binaryStream << registered4x1OutputToken;
        binaryStream << hasOutput512StaticSession;
        binaryStream << output512InputAddr;
        binaryStream << output512OutputAddr;
        binaryStream << output512InputToken;
        binaryStream << output512OutputToken;
        binaryStream << output512RankBytes;
        std::vector<char> result;
        binaryStream.Dump(result);
        return result;
    }

    // 反序列化
    void DeSerialize(std::vector<char> &data)
    {
        BinaryStream binaryStream(data);
        binaryStream >> ccuThread;
        binaryStream >> ownCopyThread;
        binaryStream >> localBuffer;
        binaryStream >> threads;
        binaryStream >> ccuKernels;
        binaryStream >> peerRanksByGroup;
        binaryStream >> topologyType;
        binaryStream >> algorithmMode;
        binaryStream >> hasRegistered4x1Exact512;
        binaryStream >> registered4x1InputAddr;
        binaryStream >> registered4x1OutputAddr;
        binaryStream >> registered4x1InputToken;
        binaryStream >> registered4x1OutputToken;
        binaryStream >> hasOutput512StaticSession;
        binaryStream >> output512InputAddr;
        binaryStream >> output512OutputAddr;
        binaryStream >> output512InputToken;
        binaryStream >> output512OutputToken;
        binaryStream >> output512RankBytes;
    }
};

#endif // OPS_HCCL_CUSTOM_H
