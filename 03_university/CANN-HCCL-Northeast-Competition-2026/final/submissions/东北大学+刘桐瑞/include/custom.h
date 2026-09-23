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

#if defined(__has_include)
#if __has_include(<ccu/ccu_res.h>)
#include <ccu/ccu_res.h>
#elif __has_include(<ccu_res.h>)
#include <ccu_res.h>
#endif
#if __has_include(<ccu/ccu_launch.h>)
#include <ccu/ccu_launch.h>
#elif __has_include(<ccu_launch.h>)
#include <ccu_launch.h>
#endif
#if __has_include(<hccl/hccl_ccu_res.h>)
#include <hccl/hccl_ccu_res.h>
#elif __has_include(<hccl_ccu_res.h>)
#include <hccl_ccu_res.h>
#endif
#if __has_include(<hccl/hccl_channel.h>)
#include <hccl/hccl_channel.h>
#endif
#endif

extern "C" {
extern CcuResult HcommCcuGetMemToken(uint64_t srcVa, uint64_t size, uint64_t *tokenInfo);

extern HcclResult HcclCommQueryCcuIns(HcclComm comm, CcuInsHandle *insHandles, uint32_t *insNum);

extern CcuResult HcommCcuKernelRegisterStart(CcuInsHandle insHandle);
extern CcuResult HcommCcuKernelRegister(CcuInsHandle insHandle, uint32_t dieId, const char *kernelFuncName,
    const void *kernelFunc, const void **kernelArgs, uint32_t argNum, CcuKernelHandle *kernelHandle);
extern CcuResult HcommCcuKernelRegisterEnd(CcuInsHandle insHandle);
extern CcuResult HcommCcuKernelLaunch(
    ThreadHandle threadHandle, CcuKernelHandle kernelHandle, const void *taskArgs, uint32_t argNum);
}

namespace ccu = ::AscendC::ccu;

typedef struct {
    void *addr;
    uint64_t size;
} CommBuffer;

struct CcuKernelArgBase {
    ChannelHandle channels[MAX_RANK_SIZE];
    uint32_t channelCount;
};

struct CcuKernelInfo {
    char kernelFuncName[64];

    void *kernelFunc;

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

#define SCATTER_DIAG_LOG 0

#define SCATTER_SPREAD_CLOS_DIES 1

#define SCATTER_ENABLE_RELAY 1

constexpr uint64_t SCATTER_RELAY_MAX_NUM = 9;
constexpr uint64_t SCATTER_RELAY_MAX_DEN = 11;

constexpr uint32_t SCATTER_DIE_NUM = 2;

constexpr uint64_t SCATTER_CLOS_BW_NUM = 4;
constexpr uint64_t SCATTER_CLOS_BW_DEN = 1;

constexpr uint64_t SCATTER_RELAY_MIN_SHARE = 1024ULL * 1024ULL;

constexpr uint64_t SCATTER_RELAY_GAIN_NUM = 15;
constexpr uint64_t SCATTER_RELAY_GAIN_DEN = 16;

constexpr uint64_t SCATTER_ALIGN = 512;

constexpr CommProtocol SCATTER_CCU_PROTOCOL = static_cast<CommProtocol>(4);

constexpr uint32_t SCATTER_CHANNEL_NOTIFY_NUM = 3;

#define SCATTER_WAIT_SLOWEST_LAST 1

#define SCATTER_CACHE_COMM_INFO 1

#define SCATTER_POST_BEFORE_OWNCOPY 1

#define SCATTER_PULL_SMALL 1

constexpr uint32_t SCATTER_XN_BASE = 0;
constexpr uint32_t SCATTER_XN_TOKEN = 1;

constexpr uint32_t SCATTER_NOTIFY_IDX = 0;

constexpr uint32_t SCATTER_INTER_THREAD_NOTIFY_IDX = 0;

constexpr uint16_t SCATTER_BIT_PUB_BASE = 0x1;
constexpr uint16_t SCATTER_BIT_PUB_TOKEN = 0x2;
constexpr uint16_t SCATTER_BIT_PRESYNC = SCATTER_BIT_PUB_BASE | SCATTER_BIT_PUB_TOKEN;
constexpr uint16_t SCATTER_BIT_POST = 0x4;
constexpr uint16_t SCATTER_BIT_READ_DONE = 0x8;

constexpr uint16_t SCATTER_EV_PULL = 0x1;
constexpr uint16_t SCATTER_EV_FWD = 0x2;
constexpr uint16_t SCATTER_EV_OWN = 0x1;
constexpr uint16_t SCATTER_EV_SELF = 0x2;

constexpr uint32_t SCATTER_MAX_KERNEL_PER_DIE = 2;

enum ScatterRole : uint32_t {
    SCATTER_ROLE_ROOT = 0,
    SCATTER_ROLE_NODE = 1,
    SCATTER_ROLE_NUM = 2
};

enum ScatterKernelKind : uint32_t {
    SCATTER_KIND_PULL_ROOT = 0,
    SCATTER_KIND_PUSH = 1,
    SCATTER_KIND_NUM = 2
};

static_assert(SCATTER_KIND_NUM <= SCATTER_MAX_KERNEL_PER_DIE, "ccu mission budget is 2 kernels per die");

enum ScatterPullRootArgIdx : uint32_t {
    SC_PR_ARG_IN_BASE = 0,
    SC_PR_ARG_IN_TOKEN,
    SC_PR_ARG_FIXED_NUM
};

enum ScatterPullRootOwnArgOff : uint32_t {
    SC_PR_OWN_OFF_DST_BASE = 0,
    SC_PR_OWN_OFF_DST_TOKEN,
    SC_PR_OWN_OFF_SRC_ADDR,
    SC_PR_OWN_OFF_SIZE,
    SC_PR_OWN_ARG_NUM
};

enum ScatterPushArgIdx : uint32_t {
    SC_PUSH_ARG_ROLE = 0,
    SC_PUSH_ARG_HAS_PUB,
    SC_PUSH_ARG_Y1,
    SC_PUSH_ARG_Y2,
    SC_PUSH_ARG_Y3,
    SC_PUSH_ARG_Y4,
    SC_PUSH_ARG_Y5,
    SC_PUSH_ARG_Y6,
    SC_PUSH_ARG_Y7,
    SC_PUSH_ARG_Y8,
    SC_PUSH_ARG_FIXED_NUM
};

constexpr uint64_t SC_PUSH_ROLE_ROOT = 0;
constexpr uint64_t SC_PUSH_ROLE_NODE = 1;
constexpr uint64_t SC_PUSH_ROLE_NODE_PULL = 2;

constexpr uint64_t SC_NODE_MODE_IDLE = 0;
constexpr uint64_t SC_NODE_MODE_RECV = 1;
constexpr uint64_t SC_NODE_MODE_PULL = 2;
constexpr uint64_t SC_NODE_MODE_FWD = 3;
constexpr uint64_t SC_NODE_MODE_WAIT = 4;

constexpr uint64_t SC_NODE_PHASE_NONE = 0;
constexpr uint64_t SC_NODE_PHASE_PULL = 1;
constexpr uint64_t SC_NODE_PHASE_FWD = 2;
constexpr uint64_t SC_NODE_PHASE_BOTH = 3;
constexpr uint64_t SC_NODE_PHASE_WAIT = 4;

constexpr uint32_t SCATTER_PUSH_ARG_GROUPS = 1;

struct AlgResourceCtx {
    ThreadHandle ccuThread;
    CommBuffer localBuffer;
    std::vector<ThreadHandle> threads;
    std::vector<CcuKernelHandle> ccuKernels;

    uint64_t localBufferToken = 0;
    uint32_t myRank = INVALID_VALUE_RANKID;
    uint32_t rankSize = 0;
    std::vector<ChannelHandle> channels;
    std::vector<uint32_t> channelDie;
    std::vector<uint32_t> serverId;
    std::vector<uint32_t> argNum;

    std::vector<char> Serialize()
    {
        BinaryStream binaryStream;
        binaryStream << ccuThread;
        binaryStream << localBuffer;
        binaryStream << threads;
        binaryStream << ccuKernels;
        binaryStream << localBufferToken;
        binaryStream << myRank;
        binaryStream << rankSize;
        binaryStream << channels;
        binaryStream << channelDie;
        binaryStream << serverId;
        binaryStream << argNum;
        std::vector<char> result;
        binaryStream.Dump(result);
        return result;
    }

    void DeSerialize(std::vector<char> &data)
    {
        BinaryStream binaryStream(data);
        binaryStream >> ccuThread;
        binaryStream >> localBuffer;
        binaryStream >> threads;
        binaryStream >> ccuKernels;
        binaryStream >> localBufferToken;
        binaryStream >> myRank;
        binaryStream >> rankSize;
        binaryStream >> channels;
        binaryStream >> channelDie;
        binaryStream >> serverId;
        binaryStream >> argNum;
    }
};

struct ScatterKernelArg : CcuKernelArgBase {
    uint32_t rankSize;
    uint32_t rankId;

    uint32_t pubChannelIdx[MAX_RANK_SIZE];
    uint32_t pubChannelCount;

    uint8_t wrUseRelaySize[MAX_RANK_SIZE];

    uint32_t crossNum;

    uint32_t handleSelfRank;
};

inline uint32_t ScatterPeerRankOf(uint32_t channelIdx, uint32_t myRank)
{
    return (channelIdx < myRank) ? channelIdx : (channelIdx + 1);
}

inline uint32_t ScatterChannelOf(uint32_t peerRank, uint32_t myRank)
{
    return (peerRank < myRank) ? peerRank : (peerRank - 1);
}

inline void ScatterGroupChannelsByDie(uint32_t myRank, const std::vector<uint32_t> &channelDie,
    const std::vector<uint32_t> &serverId, std::vector<std::vector<uint32_t>> &dieChannelIdx)
{
    dieChannelIdx.assign(SCATTER_DIE_NUM, std::vector<uint32_t>());
    const uint32_t mySid = (myRank < serverId.size()) ? serverId[myRank] : 0U;

    for (uint32_t pass = 0; pass < 2; pass++) {
        for (uint32_t c = 0; c < channelDie.size(); c++) {
            const uint32_t peer = ScatterPeerRankOf(c, myRank);
            const uint32_t sid = (peer < serverId.size()) ? serverId[peer] : 0U;
            const bool cross = (sid != mySid);
            if ((pass == 0) != cross) {
                continue;
            }
            if (channelDie[c] < SCATTER_DIE_NUM) {
                dieChannelIdx[channelDie[c]].push_back(c);
            }
        }
    }
}

inline uint32_t ScatterOwnCopyDie(const std::vector<std::vector<uint32_t>> &dieChannelIdx)
{
    uint32_t ownCopyDie = SCATTER_DIE_NUM;
    for (uint32_t die = 0; die < SCATTER_DIE_NUM && die < dieChannelIdx.size(); die++) {
        if (dieChannelIdx[die].empty()) {
            continue;
        }
        if (ownCopyDie == SCATTER_DIE_NUM || dieChannelIdx[die].size() < dieChannelIdx[ownCopyDie].size()) {
            ownCopyDie = die;
        }
    }
    return ownCopyDie;
}

inline uint32_t ScatterDataTypeSize(HcclDataType dataType)
{
    if (dataType == HCCL_DATA_TYPE_FP32) {
        return static_cast<uint32_t>(sizeof(float));
    }
    auto it = SIZE_TABLE.find(dataType);
    return (it == SIZE_TABLE.end()) ? 0U : it->second;
}

inline bool ScatterPullMode(uint64_t shareBytes, uint64_t pieceBytes)
{
#if SCATTER_PULL_SMALL
    return (shareBytes < SCATTER_RELAY_MIN_SHARE) && (pieceBytes == shareBytes);
#else
    (void)shareBytes;
    (void)pieceBytes;
    return false;
#endif
}

inline uint64_t ScatterAlignUp(uint64_t value, uint64_t align)
{
    if (align == 0) {
        return value;
    }
    return ((value + align - 1) / align) * align;
}

struct ScatterPlan {
    bool relayOn = false;
    uint64_t shareBytes = 0;
    uint64_t directBytes = 0;
    uint64_t relayBytes = 0;
    uint32_t role = SCATTER_ROLE_NODE;
    uint32_t myRelayTarget = INVALID_VALUE_RANKID;
    uint32_t relayParent[MAX_RANK_SIZE];
    uint32_t relayChild[MAX_RANK_SIZE];
    bool sameServerAsRoot[MAX_RANK_SIZE];

    ScatterPlan()
    {
        for (uint32_t i = 0; i < MAX_RANK_SIZE; i++) {
            relayParent[i] = INVALID_VALUE_RANKID;
            relayChild[i] = INVALID_VALUE_RANKID;
            sameServerAsRoot[i] = false;
        }
    }

    bool IsRelay() const { return myRelayTarget != INVALID_VALUE_RANKID; }
};

inline void ScatterBuildPlan(uint32_t rankSize, uint32_t root, uint32_t myRank, const std::vector<uint32_t> &serverId,
    uint64_t shareBytes, uint64_t stageCapacity, bool allowRelay, ScatterPlan &plan)
{
    plan.shareBytes = shareBytes;
    plan.directBytes = shareBytes;
    plan.relayBytes = 0;
    plan.relayOn = false;
    plan.role = (myRank == root) ? SCATTER_ROLE_ROOT : SCATTER_ROLE_NODE;
    plan.myRelayTarget = INVALID_VALUE_RANKID;

    const uint32_t rootServer = (root < serverId.size()) ? serverId[root] : 0U;
    uint32_t localNum = 0;
    for (uint32_t r = 0; r < rankSize; r++) {
        const uint32_t sid = (r < serverId.size()) ? serverId[r] : 0U;
        plan.sameServerAsRoot[r] = (sid == rootServer);
        if (plan.sameServerAsRoot[r]) {
            localNum++;
        }
    }

#if SCATTER_ENABLE_RELAY
    if (!allowRelay || shareBytes < SCATTER_RELAY_MIN_SHARE || localNum < 2 || localNum >= rankSize) {
        return;
    }

    std::vector<uint32_t> neighbors;
    std::vector<uint32_t> remotes;
    for (uint32_t r = 0; r < rankSize; r++) {
        if (plan.sameServerAsRoot[r]) {
            if (r != root) {
                neighbors.push_back(r);
            }
        } else {
            remotes.push_back(r);
        }
    }

    const uint64_t remoteNum = static_cast<uint64_t>(remotes.size());
    const uint64_t pairNum = (neighbors.size() < remotes.size()) ? neighbors.size() : remotes.size();
    if (pairNum == 0) {
        return;
    }

    if (pairNum != static_cast<uint64_t>(neighbors.size())) {
        return;
    }

    const uint64_t numerator = remoteNum * SCATTER_CLOS_BW_DEN;
    if (numerator <= SCATTER_CLOS_BW_NUM) {
        return;
    }

    {
        const uint64_t gainLhs = SCATTER_RELAY_GAIN_DEN * SCATTER_CLOS_BW_NUM * (remoteNum + pairNum);
        const uint64_t gainRhs
            = SCATTER_RELAY_GAIN_NUM * remoteNum * (pairNum * SCATTER_CLOS_BW_DEN + SCATTER_CLOS_BW_NUM);
        if (gainLhs > gainRhs) {
            return;
        }
    }
    const uint64_t coefNum = numerator - SCATTER_CLOS_BW_NUM;
    const uint64_t coefDen = pairNum * SCATTER_CLOS_BW_DEN + SCATTER_CLOS_BW_NUM;
    uint64_t relayRaw = shareBytes / coefDen * coefNum + (shareBytes % coefDen) * coefNum / coefDen;
    if (relayRaw >= shareBytes) {
        relayRaw = shareBytes - 1;
    }

    uint64_t direct = ScatterAlignUp(shareBytes - relayRaw, SCATTER_ALIGN);
    if (direct == 0) {
        direct = SCATTER_ALIGN;
    }

    const uint64_t minDirect = ScatterAlignUp(
        shareBytes - shareBytes / SCATTER_RELAY_MAX_DEN * SCATTER_RELAY_MAX_NUM, SCATTER_ALIGN);
    if (direct < minDirect) {
        direct = minDirect;
    }
    if (direct >= shareBytes) {
        return;
    }
    const uint64_t relay = shareBytes - direct;
    if (relay > stageCapacity || relay > MAX_DATA_SIZE) {
        return;
    }

    plan.relayOn = true;
    plan.directBytes = direct;
    plan.relayBytes = relay;
    for (uint64_t i = 0; i < pairNum; i++) {
        const uint32_t up = neighbors[i];
        const uint32_t down = remotes[i];
        plan.relayChild[up] = down;
        plan.relayParent[down] = up;
    }
    if (myRank != root && plan.relayChild[myRank] != INVALID_VALUE_RANKID) {
        plan.myRelayTarget = plan.relayChild[myRank];
    }
#else
    (void)allowRelay;
    (void)stageCapacity;
    (void)localNum;
#endif
}

#endif
