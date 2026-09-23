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

#ifndef HCCL_SCATTER_HUB_ENABLE
#define HCCL_SCATTER_HUB_ENABLE 1
#endif

#ifndef HCCL_SCATTER_DIAGNOSTICS
#define HCCL_SCATTER_DIAGNOSTICS 0
#endif

// V32 combined performance candidate: retain V30.6 rank12/4x3 RelayFuse+SyncFix2,
// and enable the independently validated V31 rank16/2x8 Hub4 physical route selector.
#ifndef HCCL_SCATTER_2X8_PHYS_ROUTE
#define HCCL_SCATTER_2X8_PHYS_ROUTE 1
#endif

// V21 recovery guard: keep the entire 16-rank/2x8 data path on the
// empirically stable V17.4 Direct protocol while retaining V20 optimizations
// for rankSize 4/12. Disable only for a dedicated 2x8 experiment.
#ifndef HCCL_SCATTER_LEGACY_2X8_SAFE
#define HCCL_SCATTER_LEGACY_2X8_SAFE 1
#endif

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

/*
 * directKernels: V6 correctness baseline.
 * relayLoadKernels: root -> helper, staging the relay tail in the communication-domain HcclBuffer.
 * relayForwardKernels: helper -> target, forwarding data from HcclBuffer to target.recvBuf.
 * relayFinalKernels: reserved compatibility field; V30.5 no longer registers this redundant kernel set.
 *
 * V10 keeps the three-phase correctness protocol but separates relay scratch from recvBuf. This
 * removes the output overwrite dependency and is the prerequisite for a safe ping-pong pipeline.
 */
struct AlgResourceCtx {
    uint32_t schemaVersion = 25U;
    uint32_t directBatchMode = 0U;
    // V23: enable the low-latency Direct scheduler only for the verified 8+4 topology.
    uint32_t directSmallFast = 0U;
    uint32_t localRole = 0U;
    uint64_t relayPlanHash = 0U;
    CommBuffer localBuffer{};
    std::vector<CcuKernelHandle> directKernels;
    // V16: metadata aligned 1:1 with directKernels. It lets the host scheduler
    // place kernels registered on different CCU dies onto independent TS threads.
    std::vector<uint32_t> directKernelDieIds;
    std::vector<uint32_t> directKernelChannelCounts;
    std::vector<CcuKernelHandle> relayLoadKernels;
    std::vector<CcuKernelHandle> relayForwardKernels;
    std::vector<CcuKernelHandle> relayFinalKernels;
    // V17: split relay final traffic so work that does not touch root->helper
    // channels can overlap the staged relay path.
    std::vector<CcuKernelHandle> relayDirectKernels;
    std::vector<uint32_t> relayDirectKernelChannelCounts;
    // V18: approximate twelfth-of-slice traffic weights for each root relay-direct
    // path group.  The root scheduler uses these weights together with the
    // helper Load+Final service-time proxies to reduce the longest TS lane.
    std::vector<uint32_t> relayDirectKernelWeightUnits;
    std::vector<CcuKernelHandle> relayHelperFinalKernels;
    uint32_t relayCapable = 0;
    uint32_t relaySplitCapable = 0;

    // V19 dual-end 2x8 relay. All rank IDs are obtained from RankGraph.
    uint32_t hubEnabled = 0U;
    uint32_t hubRank = INVALID_VALUE_RANKID;
    std::vector<uint32_t> hubHelpers;
    std::vector<uint32_t> hubTargets;
    // Root: matching groups, Load followed by helper own-data delivery.
    // Helper: one corresponding receiver in each vector.
    std::vector<CcuKernelHandle> hubLocalLoad;
    std::vector<CcuKernelHandle> hubLocalOwn;
    // Root/B0: one ingress kernel and one B0 own-data kernel.
    std::vector<CcuKernelHandle> hubIngress;
    std::vector<CcuKernelHandle> hubOwn;
    // Helper/target: one tail kernel; B0: path-grouped prefix kernels;
    // target: its single prefix receiver.
    std::vector<CcuKernelHandle> hubTail;
    std::vector<CcuKernelHandle> hubPrefix;

    std::vector<char> Serialize()
    {
        BinaryStream binaryStream;
        binaryStream << schemaVersion;
        binaryStream << directBatchMode;
        binaryStream << directSmallFast;
        binaryStream << localRole;
        binaryStream << relayPlanHash;
        binaryStream << localBuffer;
        binaryStream << directKernels;
        binaryStream << directKernelDieIds;
        binaryStream << directKernelChannelCounts;
        binaryStream << relayLoadKernels;
        binaryStream << relayForwardKernels;
        binaryStream << relayFinalKernels;
        binaryStream << relayDirectKernels;
        binaryStream << relayDirectKernelChannelCounts;
        binaryStream << relayDirectKernelWeightUnits;
        binaryStream << relayHelperFinalKernels;
        binaryStream << relayCapable;
        binaryStream << relaySplitCapable;
        binaryStream << hubEnabled;
        binaryStream << hubRank;
        binaryStream << hubHelpers;
        binaryStream << hubTargets;
        binaryStream << hubLocalLoad;
        binaryStream << hubLocalOwn;
        binaryStream << hubIngress;
        binaryStream << hubOwn;
        binaryStream << hubTail;
        binaryStream << hubPrefix;

        std::vector<char> result;
        binaryStream.Dump(result);
        return result;
    }

    void DeSerialize(std::vector<char> &data)
    {
        BinaryStream binaryStream(data);
        binaryStream >> schemaVersion;
        binaryStream >> directBatchMode;
        binaryStream >> directSmallFast;
        binaryStream >> localRole;
        binaryStream >> relayPlanHash;
        binaryStream >> localBuffer;
        binaryStream >> directKernels;
        binaryStream >> directKernelDieIds;
        binaryStream >> directKernelChannelCounts;
        binaryStream >> relayLoadKernels;
        binaryStream >> relayForwardKernels;
        binaryStream >> relayFinalKernels;
        binaryStream >> relayDirectKernels;
        binaryStream >> relayDirectKernelChannelCounts;
        binaryStream >> relayDirectKernelWeightUnits;
        binaryStream >> relayHelperFinalKernels;
        binaryStream >> relayCapable;
        binaryStream >> relaySplitCapable;
        binaryStream >> hubEnabled;
        binaryStream >> hubRank;
        binaryStream >> hubHelpers;
        binaryStream >> hubTargets;
        binaryStream >> hubLocalLoad;
        binaryStream >> hubLocalOwn;
        binaryStream >> hubIngress;
        binaryStream >> hubOwn;
        binaryStream >> hubTail;
        binaryStream >> hubPrefix;

    }
};

#endif // OPS_HCCL_CUSTOM_H
