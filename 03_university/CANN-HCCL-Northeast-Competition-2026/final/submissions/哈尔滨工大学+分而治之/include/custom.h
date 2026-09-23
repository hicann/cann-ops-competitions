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

struct CcuKernelArgBase {
    ChannelHandle channels[MAX_RANK_SIZE];
    uint32_t channelCount;
};

struct CcuKernelArgDirectProbe : CcuKernelArgBase {
    uint32_t rankSize;
    uint32_t rankId;
    uint32_t rootId;
    uint32_t groupDie;                 // 本 kernel 全部 channel 所属的本地 IO die
    uint32_t selfCopy;                 // root 自分片 LocalCopy 是否归本 kernel（每变体恰一组持有）
    uint32_t peerRanks[MAX_RANK_SIZE]; // 与 channels[0..channelCount) 对齐的对端 rank
    // E4 全 helper 中继（点对点：helper[i] <-> remote[i]，全部注册期烘焙，kernel 内无运行期角色分支）
    uint32_t relayDestIdx;   // phase3: unique forwarding channel, or UINT32_MAX
    uint32_t phase;          // 1=P1, 2=point-to-point P2, 3=passive Clos P2
    uint32_t relayVariant;   // phase1: 1=two-server relay; 2=uniform multi-server, scratch gated by relayY
    uint32_t relayRole;      // phase2：1=helper（转发完成后 Record epoch），0=dest（Wait epoch）
    uint32_t publishRootIdx; // phase1 中继变体：serving helper 到 root 的 channel 下标，否则 UINT32_MAX
    uint32_t ingressCount;   // phase1 中继变体（仅 root）：本 kernel 内 ingress 写条目数
    uint32_t ingressHelperIdx[MAX_RANK_SIZE];  // 条目 j：helper channel 下标
    uint32_t ingressRemoteRank[MAX_RANK_SIZE]; // 条目 j：该 helper 服务的 remote rank
    uint32_t servedRemote[MAX_RANK_SIZE];      // 与 peerRanks 对齐：1=被服务 remote（仅 root 使用）
    // E4.5 小消息（stageMode=2 运行期分支，复用相位1 注册）：本组内 root channel 下标，无则 UINT32_MAX
    uint32_t smallRootIdx;
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

struct AlgResourceCtx {
    ThreadHandle ccuThread;                    ///< CCU通信引擎上的thread资源
    CommBuffer localBuffer;                    ///< 本端HCCL通信内存
    std::vector<ThreadHandle> threads;         ///< [cpuThread, auxiliaryThread?]（die 分组>1 时含后者）
    std::vector<CcuKernelHandle> ccuKernels;   ///< 相位1 kernel，按本地 die 组（每 rank 至多 2 个）
    std::vector<CcuKernelHandle> relayKernels; ///< P2: pair participants, or all ranks for passive Clos; at most one
    // E4.5 小消息不新增注册（per-die mission 上限），复用相位1 kernel 的 stageMode=2 分支；
    // 非 root 记录 root channel 所在的相位1 die 组下标，root 为 UINT32_MAX（launch 全部组）
    uint32_t smallRootGroup = UINT32_MAX;
    uint32_t smallEntryVersion = 1;
    std::vector<CcuKernelHandle> smallKernels;
    std::vector<uint32_t> smallDedicated;
    // 双 die 分组时的 fork/join 资源；随缓存的引擎上下文存活，生命周期与通信域一致
    aclrtStream auxiliaryStream = nullptr;
    aclrtNotify startNotify = nullptr;
    aclrtNotify doneNotify = nullptr;
    // E4：P2 流水资源（与 P1 流组并行，chunk 级重叠）；仅持有 P2 kernel 的 rank 创建
    aclrtStream relayStream = nullptr;
    aclrtNotify relayForkParity0 = nullptr; // P1(chunk)->P2(chunk) 按奇偶独立，避免合并丢失
    aclrtNotify relayForkParity1 = nullptr;
    aclrtNotify relayDoneNotify = nullptr;
    ThreadHandle relayThread{};
    std::vector<uint32_t> prefixGroups;      // P1 groups with root ingress or helper scratch receipt
    uint32_t helperPrefixGroup = UINT32_MAX; // stream that establishes helper scratch readiness
    uint32_t kRoot = 0;                      // root server size: two-server complement or uniform instance size
    uint32_t relayEligible = 0;              // 0=direct, 1=pair relay, 2=passive relay (single chunk <=256MiB)

    // 序列化
    std::vector<char> Serialize()
    {
        BinaryStream binaryStream;
        binaryStream << ccuThread;
        binaryStream << localBuffer;
        binaryStream << threads;
        binaryStream << ccuKernels;
        binaryStream << relayKernels;
        binaryStream << auxiliaryStream << startNotify << doneNotify;
        binaryStream << relayStream << relayForkParity0 << relayForkParity1 << relayDoneNotify;
        binaryStream << relayThread;
        binaryStream << kRoot << relayEligible;
        binaryStream << prefixGroups << helperPrefixGroup;
        binaryStream << smallRootGroup << smallEntryVersion << smallKernels << smallDedicated;
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
        binaryStream >> relayKernels;
        binaryStream >> auxiliaryStream >> startNotify >> doneNotify;
        binaryStream >> relayStream >> relayForkParity0 >> relayForkParity1 >> relayDoneNotify;
        binaryStream >> relayThread;
        binaryStream >> kRoot >> relayEligible;
        binaryStream >> prefixGroups >> helperPrefixGroup;
        binaryStream >> smallRootGroup >> smallEntryVersion >> smallKernels >> smallDedicated;
    }
};

#endif // OPS_HCCL_CUSTOM_H
