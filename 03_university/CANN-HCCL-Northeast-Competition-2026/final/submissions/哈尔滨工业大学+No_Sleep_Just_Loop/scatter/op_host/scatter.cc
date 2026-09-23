/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <hccl/hccl_res_expt.h>
#include <hccl/hccl_rank_graph.h>
#include <hccl/hccl_diag.h>
#include <hccl/hccl_ccu_res.h>
#include <hcomm/ccu/ccu_launch.h>

#include <algorithm>
#include <array>
#include <cstdio>
#include <string>
#include <vector>

#include "log.h"
#include "common.h"
#include "custom.h"
#include "hccl.h"
#include "exec_op.h"
#include "ccu_kernel.h"

namespace {

constexpr uint32_t CHANNEL_NOTIFY_NUM = 3;
constexpr uint32_t LAYER_MESH = 0;
constexpr uint32_t LAYER_CLOS = 1;

// ---- 三条路径的数据量判据 ----------------------------------------------------
// 512KB 档: sendBytes ≈ 524288;  400M+4B 档: ≈ 419430416~419430464;  512MB 档: ≈ 536870912
// 三档相差 3 个数量级, 用 1MB / 500MB 两个阈值即可干净分开。
constexpr uint64_t WAVE_MIN_SEND_BYTES = 1ULL * 1024 * 1024;
constexpr uint64_t PROXY_MIN_SEND_BYTES = 500ULL * 1024 * 1024;

inline bool IsSmallScatter(uint64_t sendBytes)
{
    return sendBytes < WAVE_MIN_SEND_BYTES; // 1MiB, 已有常量, 禁止另写 512*1024
}

// 512KB 档走 Push RootStar。Pull 实现已删除(开 Pull 到不了官方 9µs, USE_PULL 恒 false)。
// v16f: 在 v16e 上合入 4x3 中档 Multi-Target + seedReadyWrite; 8+4 WAVE_STAR 与 512KB E1/E1b/E4 保持不变。
// v16g: 仅 4x3 SMALL Root Clos 走 F-ClosBurst (Wait ch[0] + 9 Write 连发); Mesh/Peer/LARGE 零协议 diff。
// 未恢复 Pull 死路径; 未把源 zip 的本 rank 选路 / 无条件 >=500MB PROXY 带进来。

// 8+4 等 Mesh-dominant mixed 大包走 Wave-Star (Root-only control + 单向 DONE)。
// 4x1 保持旧 WAVE。关闭本开关即回退到 WAVE(含 Round B SelfCopy 负载均衡)。
constexpr bool USE_WAVE_STAR = true;

// 路径 3 的代理比例 p —— 由 root 的出口带宽下界推出, **与数据量无关**。
//
//   root 必须发出的字节 = (F + R)·S     F 个本 Server peer 各 1 块 + R 个远端块
//   root 的出口总带宽    = (F + 4)·b     F 条 Mesh 链路(各 b) + 1 条 Clos(4b)
//   =>  T_floor = (F + R)/(F + 4) 单位,  1 单位 = S/b
//
// 去串行化(MeshPhase SEED/LOCAL/TAIL/SYNC)之前的时序是
//     T(p) = max( 1 + 1.25p , (R - Fp)/4 )        <- 转发 0.25p 串在 Mesh 数据面之后, 够不到下界
// 去串行化之后:
//     T(p) = max( 1 + p , 1.25p , (R - Fp)/4 )
//   root  : 尾段由 SEED kernel 先发 -> Mesh 链路上先走 p 再走直发块 1, 占用仍是 1+p;
//   代转者: TAIL kernel 在尾段落位即结束 -> Clos kernel 立刻开跑, 转发落在 1.25p, 不再叠加。
// 两项平衡即取下界, 解出
//     kp = (R - 4) / (F + 4)
// 其中 k = 每个代转者带的远端目标数。M(=F·k) 在两边同时约掉, 故 **k 与目标分配方式
// 都不影响最优 p** —— 2 个目标还是 8 个目标, T 都是同一个值。
//
//   2x8 (F=7, R=8): kp = 4/11  ->  p = 4/11,  T = 15/11
//   4x3 (F=2, R=9): kp = 5/6   ->  p = 5/6,   T = 11/6
//
// ⚠️ 最优 p 与数据量无关, 400M 与 512MB **用同一组值** —— 前提是两档都开了去串行化。
//    若拆分被关掉(退回 1+1.25p), 这两个 p 会沿串行曲线**反向**移动
//    (2x8 上 4/11 比 1/3 慢 2.7%), 必须与拆分开关同进同退。
//
// 实测核对(2026-09-17, b ≈ 48.5 GB/s):
//   2x8 400M 下界 736  -> 实测 740  (+0.5%)     4x3 400M 下界 1320 -> 实测 1320 (0%)
//   4x1 400M 下界 1620 -> 实测 1610 (-0.6%)     8+4 400M 下界 720  -> 实测 725  (+0.7%)
// 四个点全部落在下界上 —— 本结构已把 root 的出口带宽榨干, 没有剩余控制开销。
// 而改动前的 512MB 两档高出各自下界 4.9% / 3.0%, 正是差在"没开去串行化"这一步。
constexpr uint32_t PROXY_2X8_NUMERATOR = 4;
constexpr uint32_t PROXY_2X8_DENOMINATOR = 11;

constexpr uint32_t PROXY_4X3_NUMERATOR = 5;
constexpr uint32_t PROXY_4X3_DENOMINATOR = 6;

struct ChannelGroup {
    std::vector<ChannelHandle> channels;
    std::vector<uint32_t> peers;
};

std::string FormatRankList(const std::vector<uint32_t> &ranks)
{
    std::string text = "[";
    for (size_t i = 0; i < ranks.size(); ++i) {
        if (i != 0) {
            text += ",";
        }
        text += std::to_string(ranks[i]);
    }
    text += "]";
    return text;
}

// 在指定 layer 上找 myRank -> peerRank 的第一条 UBC_CTP 链路并填出 channel 描述。
// 该层没有链路时返回成功但 found=false, 交由调用方尝试下一层。
HcclResult GetChannelDesc(HcclComm comm, uint32_t layer, uint32_t myRank, uint32_t peerRank,
    HcclChannelDesc &desc, bool &found)
{
    found = false;
    CommLink *links = nullptr;
    uint32_t linkCount = 0;
    HcclResult linkRet = HcclRankGraphGetLinks(comm, layer, myRank, peerRank, &links, &linkCount);
    if (linkRet != HCCL_SUCCESS || links == nullptr) {
        return HCCL_SUCCESS;
    }
    for (uint32_t index = 0; index < linkCount; ++index) {
        if (links[index].linkAttr.linkProtocol != COMM_PROTOCOL_UBC_CTP) {
            continue;
        }
        CHK_RET(HcclChannelDescInit(&desc, 1));
        desc.remoteRank = peerRank;
        desc.notifyNum = CHANNEL_NOTIFY_NUM;
        desc.channelProtocol = links[index].linkAttr.linkProtocol;
        desc.localEndpoint.protocol = links[index].srcEndpointDesc.protocol;
        desc.localEndpoint.commAddr = links[index].srcEndpointDesc.commAddr;
        desc.localEndpoint.loc = links[index].srcEndpointDesc.loc;
        desc.remoteEndpoint.protocol = links[index].dstEndpointDesc.protocol;
        desc.remoteEndpoint.commAddr = links[index].dstEndpointDesc.commAddr;
        desc.remoteEndpoint.loc = links[index].dstEndpointDesc.loc;
        found = true;
        return HCCL_SUCCESS;
    }
    return HCCL_SUCCESS;
}

// 对 [peerBegin, peerEnd) 内的每个对端申请 1 条 channel, 按所在网络层归入 Mesh/Clos 组。
HcclResult AcquireChannelsRange(HcclComm comm, const OpParam &param, uint32_t peerBegin, uint32_t peerEnd,
    std::array<ChannelGroup, 2> &groups)
{
    CommEngine engine = CommEngine::COMM_ENGINE_CCU;
    for (uint32_t peerRank = peerBegin; peerRank < peerEnd; ++peerRank) {
        if (peerRank == param.myRank) {
            continue;
        }

        HcclChannelDesc desc;
        bool found = false;
        uint32_t groupIndex = LAYER_MESH;
        CHK_RET(GetChannelDesc(comm, LAYER_MESH, param.myRank, peerRank, desc, found));
        if (!found) {
            groupIndex = LAYER_CLOS;
            CHK_RET(GetChannelDesc(comm, LAYER_CLOS, param.myRank, peerRank, desc, found));
        }
        CHK_PRT_RET(!found,
            HCCL_ERROR("[HcclScatter] no CCU-capable link from rank %u to rank %u", param.myRank, peerRank),
            HCCL_E_NOT_FOUND);

        ChannelHandle channel = 0;
        CHK_RET(HcclChannelAcquire(comm, engine, &desc, 1, &channel));
        groups[groupIndex].channels.push_back(channel);
        groups[groupIndex].peers.push_back(peerRank);
        HCCL_INFO("[HcclScatter] rank%u -> peer%u on layer%u", param.myRank, peerRank, groupIndex);
    }
    return HCCL_SUCCESS;
}

// 探测本 rank 的 Mesh(层 0) / Clos(层 1) 对端数量。
// 复用 GetChannelDesc 的判定路径, 保证与随后 AcquireChannelsRange 的分组结果逐条一致。
HcclResult ProbeLayerCounts(HcclComm comm, const OpParam &param, uint32_t &meshCount, uint32_t &closCount)
{
    meshCount = 0;
    closCount = 0;
    for (uint32_t peerRank = 0; peerRank < param.rankSize; ++peerRank) {
        if (peerRank == param.myRank) {
            continue;
        }
        HcclChannelDesc desc;
        bool found = false;
        CHK_RET(GetChannelDesc(comm, LAYER_MESH, param.myRank, peerRank, desc, found));
        if (found) {
            ++meshCount;
            continue;
        }
        CHK_RET(GetChannelDesc(comm, LAYER_CLOS, param.myRank, peerRank, desc, found));
        if (found) {
            ++closCount;
        }
    }
    return HCCL_SUCCESS;
}

// 全集群 Mesh 层 inst 大小。GetLinks 只返回与调用者相关的链路, 不能用来数其他 rank 的度数;
// 该接口给出 rankTable 在 Mesh 层的 group 大小, 同一 comm 的所有 rank 结果一致。
// 这是测 Round B 的选路配套: 8+4 上本 rank 邻居计数会把 8 卡侧判成 WAVE_STAR、4 卡侧判成 PROXY。
HcclResult ProbeMeshInstSizes(HcclComm comm, std::vector<uint32_t> &serverSizes)
{
    serverSizes.clear();
    uint32_t *instSizeList = nullptr;
    uint32_t listSize = 0;
    CHK_RET(HcclRankGraphGetInstSizeListByLayer(comm, LAYER_MESH, &instSizeList, &listSize));
    CHK_PRT_RET(instSizeList == nullptr || listSize == 0,
        HCCL_ERROR("[HcclScatter] mesh inst size list is empty"), HCCL_E_NOT_FOUND);
    serverSizes.assign(instSizeList, instSizeList + listSize);
    return HCCL_SUCCESS;
}

// 全集群对称等大 Server: 例如 4x3 是 4 个 Mesh inst、每个 size=3。
// 不用本 rank 的 mesh/clos 邻居数(8+4 会按侧分裂); 与 SelectPath 同一套 inst size 列表。
bool IsUniformMeshInstTopo(const std::vector<uint32_t> &serverSizes, uint32_t rankSize,
    uint32_t expectedServerCount, uint32_t expectedServerSize)
{
    if (serverSizes.size() != expectedServerCount || expectedServerSize <= 1U) {
        return false;
    }
    uint32_t sizeSum = 0;
    for (uint32_t serverSize : serverSizes) {
        if (serverSize != expectedServerSize) {
            return false;
        }
        sizeSum += serverSize;
    }
    return sizeSum == rankSize;
}

const char *ScatterPathName(ScatterPath path)
{
    switch (path) {
        case ScatterPath::ROOT_STAR:
            return "ROOT_STAR";
        case ScatterPath::WAVE:
            return "WAVE";
        case ScatterPath::PROXY:
            return "PROXY";
        case ScatterPath::ROOT_STAR_PULL:
            return "ROOT_STAR_PULL";
        case ScatterPath::WAVE_STAR:
            return "WAVE_STAR";
        default:
            return "UNKNOWN";
    }
}

// Mixed topology 把 Root self-copy 放到更轻的那一组, 且全局恰好一次。
void DecideSelfCopyOwner(size_t meshChannels, size_t closChannels, bool isRootRank,
    bool &meshSelf, bool &closSelf, bool &selfOnClos)
{
    const bool mixed = meshChannels > 0 && closChannels > 0;
    selfOnClos = mixed && closChannels < meshChannels;
    meshSelf = isRootRank && (!mixed || !selfOnClos);
    closSelf = isRootRank && (!mixed || selfOnClos);
}

// ---------------------------------------------------------------------------
// 选路
//   small (< 1MB): ROOT_STAR, 不探测拓扑
//   large (>= 1MB): 本 rank Mesh/Clos 计数仅用于日志; 真正选路用 Mesh 层 inst size 列表
//     proxy-capable Clos-dominant (对称多 Server 且最大 Server 上 Clos 更重, 如 2x8 / 4x3)
//         -> PROXY
//     mesh-dominant mixed (最大 Server 上 Mesh 更重, 如 8+4 的 [8,4])
//         -> WAVE_STAR (USE_WAVE_STAR=true) 或 WAVE (关闭时回退 Round A)
//     其余 (如 4x1)
//         -> 保持原 fallback: >=500MB PROXY, 否则 WAVE
//
// Probe 只发生在 Engine Context miss 后的 SelectPath 内, 不进每次 collective 热路径。
// 必须保证同一 comm 的所有 rank 选到同一条路径 —— 8+4 上本 rank 邻居计数会分裂:
//   8 卡侧 mesh=7 clos=4 -> WAVE_STAR, 4 卡侧 mesh=3 clos=8 且 12%4==0 -> PROXY。
// 对称性/Mesh-dominant 一律用 GetInstSizeListByLayer 的全集群 server 大小。
// ---------------------------------------------------------------------------
HcclResult SelectPath(HcclComm comm, const OpParam &param, uint64_t sendBytes, ScatterPath &path)
{
    if (sendBytes < WAVE_MIN_SEND_BYTES) {
        // 512KB 档: Pull/Push 由编译期常量二选一, 不需要探测拓扑。
        path = ScatterPath::ROOT_STAR;
        HCCL_INFO("[HcclScatter] rank%u rankSize=%u sendBytes=%llu selectedPath=%s",
            param.myRank, param.rankSize, static_cast<unsigned long long>(sendBytes),
            ScatterPathName(path));
        std::printf("[HcclScatter] pack=v16g_4x3_512kb_closburst rank%u rankSize=%u sendBytes=%llu selectedPath=%s\n",
            param.myRank, param.rankSize, static_cast<unsigned long long>(sendBytes),
            ScatterPathName(path));
        return HCCL_SUCCESS;
    }

    uint32_t meshCount = 0;
    uint32_t closCount = 0;
    CHK_RET(ProbeLayerCounts(comm, param, meshCount, closCount));

    std::vector<uint32_t> serverSizes;
    CHK_RET(ProbeMeshInstSizes(comm, serverSizes));
    uint32_t minServer = serverSizes[0];
    uint32_t maxServer = serverSizes[0];
    uint32_t sizeSum = 0;
    for (uint32_t serverSize : serverSizes) {
        minServer = std::min(minServer, serverSize);
        maxServer = std::max(maxServer, serverSize);
        sizeSum += serverSize;
    }
    const uint32_t serverCount = static_cast<uint32_t>(serverSizes.size());
    // size=1 的 Mesh inst 不是真实 Server(4x1 在本 VM 上会得到 [1,1,1,1]),
    // 不能据此当成对称多 Server 而走 PROXY。
    const bool symmetricServers =
        (minServer == maxServer) && (minServer > 1) && (sizeSum == param.rankSize);
    const uint32_t maxMesh = (maxServer > 0) ? (maxServer - 1U) : 0;
    const uint32_t minMesh = (minServer > 0) ? (minServer - 1U) : 0;
    const uint32_t closAtMaxMesh = (param.rankSize > maxServer) ? (param.rankSize - maxServer) : 0;

    const bool mixed = (maxMesh > 0) && (closAtMaxMesh > 0);
    const bool meshDominantMixed = mixed && (maxMesh > closAtMaxMesh);
    const bool proxyCapable = symmetricServers && (serverCount >= 2) && (closAtMaxMesh > maxMesh);

    if (proxyCapable) {
        path = ScatterPath::PROXY;
    } else if (meshDominantMixed) {
        path = USE_WAVE_STAR ? ScatterPath::WAVE_STAR : ScatterPath::WAVE;
    } else {
        path = (sendBytes >= PROXY_MIN_SEND_BYTES) ? ScatterPath::PROXY : ScatterPath::WAVE;
    }

    HCCL_INFO("[HcclScatter] rank%u rankSize=%u sendBytes=%llu meshCount=%u closCount=%u "
        "minMesh=%u maxMesh=%u closAtMaxMesh=%u serverCount=%u minServer=%u maxServer=%u "
        "proxyCapable=%d meshDominantMixed=%d selectedPath=%s",
        param.myRank, param.rankSize, static_cast<unsigned long long>(sendBytes), meshCount, closCount,
        minMesh, maxMesh, closAtMaxMesh, serverCount, minServer, maxServer,
        static_cast<int>(proxyCapable), static_cast<int>(meshDominantMixed), ScatterPathName(path));
    std::printf("[HcclScatter] pack=v16g_4x3_512kb_closburst rank%u rankSize=%u sendBytes=%llu meshCount=%u closCount=%u "
        "minMesh=%u maxMesh=%u closAtMaxMesh=%u serverCount=%u minServer=%u maxServer=%u "
        "proxyCapable=%d meshDominantMixed=%d selectedPath=%s\n",
        param.myRank, param.rankSize, static_cast<unsigned long long>(sendBytes), meshCount, closCount,
        minMesh, maxMesh, closAtMaxMesh, serverCount, minServer, maxServer,
        static_cast<int>(proxyCapable), static_cast<int>(meshDominantMixed), ScatterPathName(path));
    return HCCL_SUCCESS;
}

// ---- 路径 1 (RootStar) 的 channel 分配: Root 连所有对端; 非 Root 只连 Root ----
HcclResult AcquireChannelsRootStar(HcclComm comm, const OpParam &param, std::array<ChannelGroup, 2> &groups)
{
    const uint32_t peerBegin = (param.myRank == param.root) ? 0 : param.root;
    const uint32_t peerEnd = (param.myRank == param.root) ? param.rankSize : param.root + 1;
    return AcquireChannelsRange(comm, param, peerBegin, peerEnd, groups);
}

// ---- 路径 2 (Wave) 的 channel 分配: 每个对端 1 条 ----
HcclResult AcquireChannelsWave(HcclComm comm, const OpParam &param, std::array<ChannelGroup, 2> &groups)
{
    return AcquireChannelsRange(comm, param, 0, param.rankSize, groups);
}

// ---- 路径 4 (Wave-Star) 的 channel 分配: Root 连所有 Peer; Non-root 只连 Root ----
HcclResult AcquireChannelsWaveStar(HcclComm comm, const OpParam &param, std::array<ChannelGroup, 2> &groups)
{
    const uint32_t peerBegin = (param.myRank == param.root) ? 0 : param.root;
    const uint32_t peerEnd = (param.myRank == param.root) ? param.rankSize : param.root + 1;
    return AcquireChannelsRange(comm, param, peerBegin, peerEnd, groups);
}

// ---- 路径 1: 注册 RootStar kernel ----
HcclResult RegisterRootStarKernel(CcuInsHandle insHandle, const char *name, const ChannelGroup &group,
    const OpParam &param, uint64_t sliceBytes, bool doSelfCopy, CcuKernelHandle &kernelHandle)
{
    auto kernelArg = std::make_shared<CcuKernelArgRootStar>();
    kernelArg->sliceBytes = sliceBytes;
    kernelArg->selfOffset = static_cast<uint64_t>(param.root) * sliceBytes;
    kernelArg->rankSize = param.rankSize;
    kernelArg->rankId = param.myRank;
    kernelArg->root = param.root;
    kernelArg->doSelfCopy = doSelfCopy;
    kernelArg->channelCount = static_cast<uint32_t>(group.channels.size());
    for (uint32_t i = 0; i < kernelArg->channelCount; ++i) {
        kernelArg->channels[i] = group.channels[i];
        kernelArg->peerRanks[i] = group.peers[i];
        kernelArg->peerOffsets[i] = static_cast<uint64_t>(group.peers[i]) * sliceBytes;
    }

    const void *kernelArgs[] = {static_cast<void *>(kernelArg.get())};
    constexpr uint32_t kernelArgNum = 1;
    constexpr uint32_t dieId = 0; // 预留参数, 平台按 channel 集合自动推导 IO Die
    CHK_RET_CCU(HcommCcuKernelRegister(insHandle, dieId, name,
        reinterpret_cast<void *>(ops_hccl::CcuScatterRootStarKernel), kernelArgs, kernelArgNum, &kernelHandle));
    HCCL_INFO("[HcclScatter] registered %s ch=%u", name, kernelArg->channelCount);
    return HCCL_SUCCESS;
}

// ---- 路径 1 Push SMALL: 独立注册函数, 不改 LARGE 的 RegisterRootStarKernel ----
HcclResult RegisterRootStarSmallKernel(CcuInsHandle insHandle, const char *name, const ChannelGroup &group,
    const OpParam &param, uint64_t sliceBytes, bool doSelfCopy, CcuKernelHandle &kernelHandle,
    void *kernelFn = nullptr)
{
    auto kernelArg = std::make_shared<CcuKernelArgRootStar>();
    kernelArg->sliceBytes = sliceBytes;
    kernelArg->selfOffset = static_cast<uint64_t>(param.root) * sliceBytes;
    kernelArg->rankSize = param.rankSize;
    kernelArg->rankId = param.myRank;
    kernelArg->root = param.root;
    kernelArg->doSelfCopy = doSelfCopy;
    // E4: 自留块改由 Exec 在 Launch 前 Host 拷, kernel 仍 LoadArg 2/3 保持 4/2/2 ABI。
    kernelArg->skipDeviceSelfCopy = doSelfCopy;
    kernelArg->channelCount = static_cast<uint32_t>(group.channels.size());
    for (uint32_t i = 0; i < kernelArg->channelCount; ++i) {
        kernelArg->channels[i] = group.channels[i];
        kernelArg->peerRanks[i] = group.peers[i];
        kernelArg->peerOffsets[i] = static_cast<uint64_t>(group.peers[i]) * sliceBytes;
    }

    if (kernelFn == nullptr) {
        kernelFn = reinterpret_cast<void *>(ops_hccl::CcuScatterRootStarSmallKernel);
    }
    const void *kernelArgs[] = {static_cast<void *>(kernelArg.get())};
    constexpr uint32_t kernelArgNum = 1;
    constexpr uint32_t dieId = 0;
    CHK_RET_CCU(HcommCcuKernelRegister(insHandle, dieId, name, kernelFn, kernelArgs, kernelArgNum, &kernelHandle));
    HCCL_INFO("[HcclScatter] registered %s ch=%u doSelfCopy=%d", name, kernelArg->channelCount,
        static_cast<int>(doSelfCopy));
    std::printf("[HcclScatter] registered %s ch=%u doSelfCopy=%d\n", name, kernelArg->channelCount,
        static_cast<int>(doSelfCopy));
    return HCCL_SUCCESS;
}

// ---- 路径 2: 注册 Wave kernel ----
HcclResult RegisterWaveKernel(CcuInsHandle insHandle, const char *name, const ChannelGroup &group,
    const OpParam &param, bool doSelfCopy, CcuKernelHandle &kernelHandle)
{
    auto kernelArg = std::make_shared<CcuKernelArgWave>();
    kernelArg->rankSize = param.rankSize;
    kernelArg->rankId = param.myRank;
    kernelArg->root = param.root;
    kernelArg->doSelfCopy = doSelfCopy;
    kernelArg->channelCount = static_cast<uint32_t>(group.channels.size());
    for (uint32_t i = 0; i < kernelArg->channelCount; ++i) {
        kernelArg->channels[i] = group.channels[i];
        kernelArg->peerRanks[i] = group.peers[i];
    }

    const void *kernelArgs[] = {static_cast<void *>(kernelArg.get())};
    constexpr uint32_t kernelArgNum = 1;
    constexpr uint32_t dieId = 0;
    CHK_RET_CCU(HcommCcuKernelRegister(insHandle, dieId, name,
        reinterpret_cast<void *>(ops_hccl::CcuScatterWaveKernel), kernelArgs, kernelArgNum, &kernelHandle));
    HCCL_INFO("[HcclScatter] registered %s ch=%u doSelfCopy=%d", name, kernelArg->channelCount,
        static_cast<int>(kernelArg->doSelfCopy));
    return HCCL_SUCCESS;
}

void InitWaveStarKernelArg(CcuKernelArgWaveStar &kernelArg, const ChannelGroup &group,
    const OpParam &param, uint64_t sliceBytes, bool doSelfCopy)
{
    kernelArg.sliceBytes = sliceBytes;
    kernelArg.selfOffset = static_cast<uint64_t>(param.root) * sliceBytes;
    kernelArg.singleWave = (sliceBytes <= MAX_DATA_SIZE);
    kernelArg.rankSize = param.rankSize;
    kernelArg.rankId = param.myRank;
    kernelArg.root = param.root;
    kernelArg.doSelfCopy = doSelfCopy;
    kernelArg.channelCount = static_cast<uint32_t>(group.channels.size());
    for (uint32_t i = 0; i < kernelArg.channelCount; ++i) {
        kernelArg.channels[i] = group.channels[i];
        kernelArg.peerRanks[i] = group.peers[i];
        kernelArg.peerOffsets[i] = static_cast<uint64_t>(group.peers[i]) * sliceBytes;
    }
}

// ---- 路径 4: 注册 Wave-Star Root kernel ----
HcclResult RegisterWaveStarRootKernel(CcuInsHandle insHandle, const char *name, const ChannelGroup &group,
    const OpParam &param, uint64_t sliceBytes, bool doSelfCopy, CcuKernelHandle &kernelHandle)
{
    auto kernelArg = std::make_shared<CcuKernelArgWaveStar>();
    InitWaveStarKernelArg(*kernelArg, group, param, sliceBytes, doSelfCopy);

    const void *kernelArgs[] = {static_cast<void *>(kernelArg.get())};
    constexpr uint32_t kernelArgNum = 1;
    constexpr uint32_t dieId = 0;
    CHK_RET_CCU(HcommCcuKernelRegister(insHandle, dieId, name,
        reinterpret_cast<void *>(ops_hccl::CcuScatterWaveStarRootKernel), kernelArgs, kernelArgNum, &kernelHandle));
    HCCL_INFO("[HcclScatter] registered %s ch=%u doSelfCopy=%d singleWave=%d sliceBytes=%llu",
        name, kernelArg->channelCount, static_cast<int>(doSelfCopy),
        static_cast<int>(kernelArg->singleWave), static_cast<unsigned long long>(sliceBytes));
    return HCCL_SUCCESS;
}

// ---- 路径 4: 注册 Wave-Star Peer kernel ----
HcclResult RegisterWaveStarPeerKernel(CcuInsHandle insHandle, const char *name, const ChannelGroup &group,
    const OpParam &param, uint64_t sliceBytes, CcuKernelHandle &kernelHandle)
{
    auto kernelArg = std::make_shared<CcuKernelArgWaveStar>();
    InitWaveStarKernelArg(*kernelArg, group, param, sliceBytes, false);

    const void *kernelArgs[] = {static_cast<void *>(kernelArg.get())};
    constexpr uint32_t kernelArgNum = 1;
    constexpr uint32_t dieId = 0;
    CHK_RET_CCU(HcommCcuKernelRegister(insHandle, dieId, name,
        reinterpret_cast<void *>(ops_hccl::CcuScatterWaveStarPeerKernel), kernelArgs, kernelArgNum, &kernelHandle));
    HCCL_INFO("[HcclScatter] registered %s ch=%u", name, kernelArg->channelCount);
    return HCCL_SUCCESS;
}

// ---- 路径 3: 注册 Proxy kernel ----
HcclResult RegisterProxyKernel(CcuInsHandle insHandle, const char *name, const ChannelGroup &group,
    const OpParam &param, bool doSelfCopy, bool isClosGroup, bool proxyEnabled,
    const std::vector<uint32_t> &proxyPeers, const std::vector<uint32_t> &proxyTargets,
    bool multiStageRelay, CcuKernelHandle &kernelHandle,
    uint32_t meshPhase = CcuKernelArgProxy::MESH_PHASE_FULL, bool seedReadyWrite = false,
    uint32_t targetsPerForwarder = 1)
{
    auto kernelArg = std::make_shared<CcuKernelArgProxy>();
    kernelArg->rankSize = param.rankSize;
    kernelArg->rankId = param.myRank;
    kernelArg->root = param.root;
    kernelArg->doSelfCopy = doSelfCopy;
    kernelArg->isClosGroup = isClosGroup;
    kernelArg->proxyEnabled = proxyEnabled;
    kernelArg->multiStageRelay = multiStageRelay;
    kernelArg->meshPhase = meshPhase;
    kernelArg->seedReadyWrite = seedReadyWrite;
    kernelArg->targetsPerForwarder = targetsPerForwarder;
    kernelArg->proxyPeerCount = static_cast<uint32_t>(proxyPeers.size());
    kernelArg->proxyTargetCount = static_cast<uint32_t>(proxyTargets.size());
    for (uint32_t i = 0; i < kernelArg->proxyPeerCount; ++i) {
        kernelArg->proxyPeers[i] = proxyPeers[i];
    }
    for (uint32_t i = 0; i < kernelArg->proxyTargetCount; ++i) {
        kernelArg->proxyTargets[i] = proxyTargets[i];
    }
    kernelArg->channelCount = static_cast<uint32_t>(group.channels.size());
    for (uint32_t i = 0; i < kernelArg->channelCount; ++i) {
        kernelArg->channels[i] = group.channels[i];
        kernelArg->peerRanks[i] = group.peers[i];
    }

    const void *kernelArgs[] = {static_cast<void *>(kernelArg.get())};
    constexpr uint32_t kernelArgNum = 1;
    constexpr uint32_t dieId = 0;
    CHK_RET_CCU(HcommCcuKernelRegister(insHandle, dieId, name,
        reinterpret_cast<void *>(ops_hccl::CcuScatterProxyKernel), kernelArgs, kernelArgNum, &kernelHandle));
    HCCL_INFO("[HcclScatter] registered %s ch=%u proxyPeers=%u proxyTargets=%u", name,
        kernelArg->channelCount, kernelArg->proxyPeerCount, kernelArg->proxyTargetCount);
    return HCCL_SUCCESS;
}

// ===========================================================================
// 路径 1 (512KB): RootStar
// ===========================================================================
HcclResult CreateRootStarResources(HcclComm comm, OpParam &param, uint64_t sliceBytes,
    uint64_t sendBytes, AlgResourceCtx &resCtxHost, CommEngine ccuEngine)
{
    const bool small = IsSmallScatter(sendBytes);
    const bool isRootRank = (param.myRank == param.root);

    // RootStar 先保留主 thread；LARGE 混合 Mesh+Clos 时再按需申请第二条。
    resCtxHost.threads.resize(1);
    resCtxHost.threads[0] = param.cpuThread;

    std::array<ChannelGroup, 2> groups;
    CHK_RET(AcquireChannelsRootStar(comm, param, groups));

    const bool hasMeshGroup = !groups[LAYER_MESH].channels.empty();
    const bool hasClosGroup = !groups[LAYER_CLOS].channels.empty();
    const uint32_t meshN = static_cast<uint32_t>(groups[LAYER_MESH].peers.size());
    const uint32_t closN = static_cast<uint32_t>(groups[LAYER_CLOS].peers.size());

    // 4x3 SMALL Root Clos 才走 F-ClosBurst。8+4 度数不是 2+9; 4x1 rankSize!=12;
    // 2x8 rankSize==16。再用全集群 Mesh inst [3,3,3,3] 挡住碰巧 12 卡且某 root 2+9 的怪图。
    bool fourByThreeSmallRootClos = false;
    if (small && isRootRank) {
        std::vector<uint32_t> serverSizes;
        CHK_RET(ProbeMeshInstSizes(comm, serverSizes));
        const bool fourByThreeTopo = IsUniformMeshInstTopo(serverSizes, param.rankSize, 4U, 3U);
        fourByThreeSmallRootClos = fourByThreeTopo && (param.rankSize == 12U) &&
            (meshN == 2U) && (closN == 9U);
    }
    std::printf("[HcclScatter] pack=v16g_4x3_512kb_closburst rank%u closBurst=%d ch=%u meshN=%u closN=%u\n",
        param.myRank, static_cast<int>(fourByThreeSmallRootClos), closN, meshN, closN);

    // E2 flatten 已在 8+4 512KB 本地探针上被平台拒绝:
    // GetDieIdByChannels: the dies of channels are not same. 禁止串行双 kernel 冒充 flatten.
    // SMALL mixed 保持 E1 双 kernel / 双 thread, 注册 SmallMesh/SmallClos 独立函数.

    // 纯 Mesh/纯 Clos 只有一个 kernel，保持单 thread；混合拓扑需要两条
    // thread 才能并行下发两个互不冲突的 kernel。
    if (hasMeshGroup && hasClosGroup) {
        resCtxHost.threads.resize(2);
        CHK_RET(HcclThreadAcquire(comm, ccuEngine, 1, 1, &resCtxHost.threads[1]));
    }

    CcuInsHandle insHandle = 0;
    uint32_t insNum = 0;
    CHK_RET(HcclCommQueryCcuIns(comm, &insHandle, &insNum));
    CHK_PRT_RET(insNum != 1,
        HCCL_ERROR("[HcclScatter] unexpected CCU instance count %u", insNum), HCCL_E_INTERNAL);
    CHK_RET_CCU(HcommCcuKernelRegisterStart(insHandle));

    // 自留块只交给其中一个 kernel, 否则两个 kernel 并发写同一段 recvBuf 会被
    // CheckerV3 判 [ErrorCode: 302] Memory conflict。
    if (hasMeshGroup) {
        CcuKernelHandle handle = 0;
        if (small) {
            CHK_RET(RegisterRootStarSmallKernel(insHandle, "CcuScatterRootStarSmallMeshKernel",
                groups[LAYER_MESH], param, sliceBytes, isRootRank, handle));
        } else {
            CHK_RET(RegisterRootStarKernel(insHandle, "CcuScatterRootStarMeshKernel", groups[LAYER_MESH], param,
                sliceBytes, isRootRank, handle));
        }
        resCtxHost.meshKernelIndex = static_cast<uint32_t>(resCtxHost.ccuKernels.size());
        resCtxHost.ccuKernels.push_back(handle);
    }
    if (hasClosGroup) {
        CcuKernelHandle handle = 0;
        if (small) {
            if (fourByThreeSmallRootClos) {
                CHK_RET(RegisterRootStarSmallKernel(insHandle, "CcuScatterRootStarSmallClosBurstKernel",
                    groups[LAYER_CLOS], param, sliceBytes, false, handle,
                    reinterpret_cast<void *>(ops_hccl::CcuScatterRootStarSmallClosBurstKernel)));
            } else {
                CHK_RET(RegisterRootStarSmallKernel(insHandle, "CcuScatterRootStarSmallClosKernel",
                    groups[LAYER_CLOS], param, sliceBytes, isRootRank && !hasMeshGroup, handle));
            }
        } else {
            CHK_RET(RegisterRootStarKernel(insHandle, "CcuScatterRootStarClosKernel", groups[LAYER_CLOS], param,
                sliceBytes, isRootRank && !hasMeshGroup, handle));
        }
        resCtxHost.closKernelIndex = static_cast<uint32_t>(resCtxHost.ccuKernels.size());
        resCtxHost.ccuKernels.push_back(handle);
    }
    CHK_RET_CCU(HcommCcuKernelRegisterEnd(insHandle));
    resCtxHost.ccuThread = param.cpuThread;
    return HCCL_SUCCESS;
}

// ===========================================================================
// 路径 2: Wave
//   大消息直发(含 two-wave 分段能力)。4x1 等非 mixed, 以及 USE_WAVE_STAR 关闭时的
//   mesh-dominant mixed 回退走这里。2x8 / 4x3 仍走 Proxy。Root SelfCopy 归较轻的那一组。
// ===========================================================================
HcclResult CreateWaveResources(HcclComm comm, OpParam &param, AlgResourceCtx &resCtxHost,
    CommEngine ccuEngine)
{
    resCtxHost.threads.resize(2);
    resCtxHost.threads[0] = param.cpuThread;
    CHK_RET(HcclThreadAcquire(comm, ccuEngine, 1, 1, &resCtxHost.threads[1]));

    std::array<ChannelGroup, 2> groups;
    CHK_RET(AcquireChannelsWave(comm, param, groups));
    CHK_PRT_RET(groups[LAYER_MESH].channels.empty() && groups[LAYER_CLOS].channels.empty(),
        HCCL_ERROR("[HcclScatter] rankSize %u but no channel acquired", param.rankSize), HCCL_E_INTERNAL);

    CcuInsHandle insHandle = 0;
    uint32_t insNum = 0;
    CHK_RET(HcclCommQueryCcuIns(comm, &insHandle, &insNum));
    CHK_PRT_RET(insNum != 1,
        HCCL_ERROR("[HcclScatter] unexpected CCU instance count %u", insNum), HCCL_E_INTERNAL);
    CHK_RET_CCU(HcommCcuKernelRegisterStart(insHandle));

    const size_t meshChannels = groups[LAYER_MESH].channels.size();
    const size_t closChannels = groups[LAYER_CLOS].channels.size();
    const bool isRootRank = (param.myRank == param.root);
    bool meshSelf = false;
    bool closSelf = false;
    bool selfOnClos = false;
    DecideSelfCopyOwner(meshChannels, closChannels, isRootRank, meshSelf, closSelf, selfOnClos);
    HCCL_INFO("[HcclScatter] rank%u wave Mesh doSelfCopy=%d Clos doSelfCopy=%d meshCh=%zu closCh=%zu "
        "mixed=%d selfOnClos=%d",
        param.myRank, static_cast<int>(meshSelf), static_cast<int>(closSelf), meshChannels, closChannels,
        static_cast<int>(meshChannels > 0 && closChannels > 0), static_cast<int>(selfOnClos));
    std::printf("[HcclScatter] rank%u wave Mesh doSelfCopy=%d Clos doSelfCopy=%d meshCh=%zu closCh=%zu "
        "mixed=%d selfOnClos=%d\n",
        param.myRank, static_cast<int>(meshSelf), static_cast<int>(closSelf), meshChannels, closChannels,
        static_cast<int>(meshChannels > 0 && closChannels > 0), static_cast<int>(selfOnClos));

    if (meshChannels > 0) {
        CcuKernelHandle handle = 0;
        CHK_RET(RegisterWaveKernel(insHandle, "CcuScatterWaveMeshKernel", groups[LAYER_MESH], param,
            meshSelf, handle));
        resCtxHost.meshKernelIndex = static_cast<uint32_t>(resCtxHost.ccuKernels.size());
        resCtxHost.ccuKernels.push_back(handle);
    }
    if (closChannels > 0) {
        CcuKernelHandle handle = 0;
        CHK_RET(RegisterWaveKernel(insHandle, "CcuScatterWaveClosKernel", groups[LAYER_CLOS], param,
            closSelf, handle));
        resCtxHost.closKernelIndex = static_cast<uint32_t>(resCtxHost.ccuKernels.size());
        resCtxHost.ccuKernels.push_back(handle);
    }
    CHK_RET_CCU(HcommCcuKernelRegisterEnd(insHandle));
    resCtxHost.ccuThread = param.cpuThread;
    return HCCL_SUCCESS;
}

// ===========================================================================
// 路径 4: Wave-Star
//   Root mixed: 按真实 layer 分组, 2 kernel / 2 thread, SelfCopy 归轻组
//   Non-root  : 只 acquire 到 Root 的 1 条 channel, 1 kernel / 1 thread
//   注册期固化 sliceBytes / selfOffset / peerOffsets, 并标记 singleWave。
// ===========================================================================
HcclResult CreateWaveStarResources(HcclComm comm, OpParam &param, uint64_t sliceBytes,
    AlgResourceCtx &resCtxHost, CommEngine ccuEngine)
{
    const bool isRootRank = (param.myRank == param.root);

    std::array<ChannelGroup, 2> groups;
    CHK_RET(AcquireChannelsWaveStar(comm, param, groups));
    CHK_PRT_RET(groups[LAYER_MESH].channels.empty() && groups[LAYER_CLOS].channels.empty(),
        HCCL_ERROR("[HcclScatter] WaveStar rank %u acquired no channel", param.myRank), HCCL_E_INTERNAL);

    const size_t meshChannels = groups[LAYER_MESH].channels.size();
    const size_t closChannels = groups[LAYER_CLOS].channels.size();
    const bool mixed = meshChannels > 0 && closChannels > 0;
    bool meshSelf = false;
    bool closSelf = false;
    bool selfOnClos = false;
    DecideSelfCopyOwner(meshChannels, closChannels, isRootRank, meshSelf, closSelf, selfOnClos);

    // Root mixed 必须保留双 Thread; Peer 只申请 1 条, 不再启动第二 Thread。
    if (isRootRank && mixed) {
        resCtxHost.threads.resize(2);
        resCtxHost.threads[0] = param.cpuThread;
        CHK_RET(HcclThreadAcquire(comm, ccuEngine, 1, 1, &resCtxHost.threads[1]));
    } else {
        resCtxHost.threads.resize(1);
        resCtxHost.threads[0] = param.cpuThread;
    }

    CcuInsHandle insHandle = 0;
    uint32_t insNum = 0;
    CHK_RET(HcclCommQueryCcuIns(comm, &insHandle, &insNum));
    CHK_PRT_RET(insNum != 1,
        HCCL_ERROR("[HcclScatter] unexpected CCU instance count %u", insNum), HCCL_E_INTERNAL);
    CHK_RET_CCU(HcommCcuKernelRegisterStart(insHandle));

    if (isRootRank) {
        if (meshChannels > 0) {
            CcuKernelHandle handle = 0;
            CHK_RET(RegisterWaveStarRootKernel(insHandle, "CcuScatterWaveStarMeshKernel", groups[LAYER_MESH],
                param, sliceBytes, meshSelf, handle));
            resCtxHost.meshKernelIndex = static_cast<uint32_t>(resCtxHost.ccuKernels.size());
            resCtxHost.ccuKernels.push_back(handle);
        }
        if (closChannels > 0) {
            CcuKernelHandle handle = 0;
            CHK_RET(RegisterWaveStarRootKernel(insHandle, "CcuScatterWaveStarClosKernel", groups[LAYER_CLOS],
                param, sliceBytes, closSelf, handle));
            resCtxHost.closKernelIndex = static_cast<uint32_t>(resCtxHost.ccuKernels.size());
            resCtxHost.ccuKernels.push_back(handle);
        }
    } else {
        CHK_PRT_RET((meshChannels + closChannels) != 1,
            HCCL_ERROR("[HcclScatter] WaveStar Peer rank %u expected 1 channel, got mesh=%zu clos=%zu",
                param.myRank, meshChannels, closChannels), HCCL_E_INTERNAL);
        const bool hasMesh = meshChannels > 0;
        const ChannelGroup &group = hasMesh ? groups[LAYER_MESH] : groups[LAYER_CLOS];
        CcuKernelHandle handle = 0;
        CHK_RET(RegisterWaveStarPeerKernel(insHandle, "CcuScatterWaveStarPeerKernel", group, param,
            sliceBytes, handle));
        if (hasMesh) {
            resCtxHost.meshKernelIndex = static_cast<uint32_t>(resCtxHost.ccuKernels.size());
        } else {
            resCtxHost.closKernelIndex = static_cast<uint32_t>(resCtxHost.ccuKernels.size());
        }
        resCtxHost.ccuKernels.push_back(handle);
    }
    CHK_RET_CCU(HcommCcuKernelRegisterEnd(insHandle));
    resCtxHost.ccuThread = param.cpuThread;

    if (isRootRank) {
        HCCL_INFO("[HcclScatter] WaveStar Root rank=%u meshChannelCount=%zu closChannelCount=%zu "
            "Mesh doSelfCopy=%d Clos doSelfCopy=%d threadCount=%zu kernelCount=%zu "
            "singleWave=%d sliceBytes=%llu",
            param.myRank, meshChannels, closChannels, static_cast<int>(meshSelf),
            static_cast<int>(closSelf), resCtxHost.threads.size(), resCtxHost.ccuKernels.size(),
            static_cast<int>(sliceBytes <= MAX_DATA_SIZE), static_cast<unsigned long long>(sliceBytes));
        std::printf("[HcclScatter] WaveStar Root rank=%u meshChannelCount=%zu closChannelCount=%zu "
            "Mesh doSelfCopy=%d Clos doSelfCopy=%d threadCount=%zu kernelCount=%zu "
            "singleWave=%d sliceBytes=%llu\n",
            param.myRank, meshChannels, closChannels, static_cast<int>(meshSelf),
            static_cast<int>(closSelf), resCtxHost.threads.size(), resCtxHost.ccuKernels.size(),
            static_cast<int>(sliceBytes <= MAX_DATA_SIZE), static_cast<unsigned long long>(sliceBytes));
    } else {
        HCCL_INFO("[HcclScatter] WaveStar Peer rank=%u channelCount=%zu threadCount=%zu kernelCount=%zu",
            param.myRank, meshChannels + closChannels, resCtxHost.threads.size(),
            resCtxHost.ccuKernels.size());
    }
    return HCCL_SUCCESS;
}

// ===========================================================================
// 路径 3 (512MB): Proxy
//   channel: root 连所有对端; 代转者连 root + 它的转发目标; 其他非 root 只连 root
// ===========================================================================
HcclResult CreateProxyResources(HcclComm comm, OpParam &param, AlgResourceCtx &resCtxHost,
    CommEngine ccuEngine, uint64_t sendBytes)
{
    resCtxHost.threads.resize(2);
    resCtxHost.threads[0] = param.cpuThread;
    // 第二条流需要 4 个 notify:
    //   idx0     —— "放行"(让从属流以 WAIT 开头, 满足 CheckerV3 首任务约束)
    //   idx1..3  —— 分块流水的 3 个阶段各自的收敛点
    // 阶段间绝不能复用同一个 idx: 同一个 idx 上出现两个 RECORD 会被 CheckerV3 判 many-to-one
    // (实测报错: resource=AICPU_NOTIFY, previousProducer/nextProducer 同 notifyId)。
    // 非分块路径只会用到 idx0/idx1, 多申请的 notify 不产生任务。
    CHK_RET(HcclThreadAcquire(comm, ccuEngine, 1, 4, &resCtxHost.threads[1]));

    std::array<ChannelGroup, 2> groups;
    CHK_RET(AcquireChannelsWave(comm, param, groups));
    CHK_PRT_RET(groups[LAYER_MESH].channels.size() > CUSTOM_MAX_CHANNELS ||
            groups[LAYER_CLOS].channels.size() > CUSTOM_MAX_CHANNELS,
        HCCL_ERROR("[HcclScatter] too many channels in one group"), HCCL_E_INTERNAL);
    CHK_PRT_RET(groups[LAYER_MESH].channels.empty() && groups[LAYER_CLOS].channels.empty(),
        HCCL_ERROR("[HcclScatter] rankSize %u but no channel acquired", param.rankSize), HCCL_E_INTERNAL);

    // ---- 代理分流的拓扑划分 ----
    // 不能用 HcclRankGraphGetLinks(comm, root, x) 判"x 是否与 root 同 Server":
    // 该接口只返回与调用者自身相关的链路, 跨 Server 的 rank 查不到对端的 Mesh 关系。
    // 改为每个 rank 用自己已知的 channel 分组推导:
    //   root      : serverPeers = Mesh 组, remotePeers = Clos 组
    //   本 Server : serverPeers = Mesh 组去掉 root + 自身; remotePeers = Clos 组
    //   跨 Server : serverPeers = Clos 组去掉 root; remotePeers = Mesh 组 + 自身
    //               (仅两 Server 拓扑成立)
    // 4x3 有 4 个对称 Server, 跨 Server rank 的 Clos 组混了 3 个 Server, 无法直接分辨,
    // 故改走"编号连续 + Server 等大"的推导, 并用整除做自洽检查。
    const size_t meshCount = groups[LAYER_MESH].peers.size();
    const size_t closCount = groups[LAYER_CLOS].peers.size();
    const bool twoServerTopo = (param.rankSize == 16 && meshCount == 7 && closCount == 8);
    const bool symmetricFourServerTopo = (param.rankSize == 12 && meshCount == 2 && closCount == 9);
    CHK_PRT_RET(symmetricFourServerTopo && (meshCount == 0 || param.rankSize % (meshCount + 1) != 0),
        HCCL_ERROR("[HcclScatter] 4-server topology is not symmetric (rankSize=%u mesh=%zu)",
            param.rankSize, meshCount), HCCL_E_INTERNAL);

    const bool iAmRoot = (param.myRank == param.root);
    const bool rootInMyMesh =
        std::find(groups[LAYER_MESH].peers.begin(), groups[LAYER_MESH].peers.end(), param.root) !=
        groups[LAYER_MESH].peers.end();
    const bool iAmServerPeer = !iAmRoot && rootInMyMesh;

    std::vector<uint32_t> serverPeers;
    std::vector<uint32_t> remotePeers;
    if (symmetricFourServerTopo) {
        const uint32_t serverSize = static_cast<uint32_t>(meshCount) + 1;
        const uint32_t rootServerStart = (param.root / serverSize) * serverSize;
        const uint32_t rootServerEnd = rootServerStart + serverSize;
        for (uint32_t x = 0; x < param.rankSize; ++x) {
            if (x == param.root) {
                continue;
            }
            if (x >= rootServerStart && x < rootServerEnd) {
                serverPeers.push_back(x);
            } else {
                remotePeers.push_back(x);
            }
        }
    } else if (iAmRoot) {
        serverPeers = groups[LAYER_MESH].peers;
        remotePeers = groups[LAYER_CLOS].peers;
    } else if (iAmServerPeer) {
        for (uint32_t peer : groups[LAYER_MESH].peers) {
            if (peer != param.root) {
                serverPeers.push_back(peer);
            }
        }
        // 本 Server peer 自己的 Mesh 组里没有它自己, 必须补上并重排, 否则它与
        // root 视角的 serverPeers 顺序对不上, 会认领不到代转任务。
        serverPeers.push_back(param.myRank);
        std::sort(serverPeers.begin(), serverPeers.end());
        remotePeers = groups[LAYER_CLOS].peers;
    } else {
        for (uint32_t peer : groups[LAYER_CLOS].peers) {
            if (peer != param.root) {
                serverPeers.push_back(peer);
            }
        }
        remotePeers = groups[LAYER_MESH].peers;
        remotePeers.push_back(param.myRank);
        std::sort(remotePeers.begin(), remotePeers.end());
    }

    // 是否启用代理: 仅两个可推导的拓扑(2x8 / 4x3), 且两侧都有可分流的对象。
    // "是否走本路径"由 HcclScatter 的数据量判据决定, 这里只判结构是否成立。
    resCtxHost.proxyEnabled = (twoServerTopo || symmetricFourServerTopo) && !serverPeers.empty() &&
        !remotePeers.empty();
    // 最优 p 只由拓扑(F, R)决定, 与数据量无关 —— 400M 与 512MB 共用同一组值。
    // 历史上两档取值不同(2x8 1/3 vs 4/11, 4x3 7/10 vs 5/6), 那是因为 512MB 档
    // 当时跑的是旧的串行时序; 去串行化之后这个差异消失了。
    if (symmetricFourServerTopo) {
        resCtxHost.proxyNumerator = PROXY_4X3_NUMERATOR;
        resCtxHost.proxyDenominator = PROXY_4X3_DENOMINATOR;
    } else {
        resCtxHost.proxyNumerator = PROXY_2X8_NUMERATOR;
        resCtxHost.proxyDenominator = PROXY_2X8_DENOMINATOR;
    }

    // 映射: 默认仍是 serverPeers[t] 代转 remotePeers[t]; 4x3 中档走下面的 appendTargets。
    // 分块流水: 2x8 与 4x3 的代理结构同构(root 写尾段到代转者 scratch -> 代转者用
    // 自己的 Clos 口转发), 故两者共用同一套分块机制。非代理拓扑恒为 false。
    // ⚠️ 2026-09-15 上机实测: 分块流水**零收益**。
    //    实测 4x3 @p=0.77 = 1.80ms, 精确等于"无流水"模型预测(1.7989ms), 而非含流水预测(1.711ms);
    //    2x8 上还净亏 11us(7 个代转者的阶段开销)。日志 dataTaskNodeCount 16->20 / 30->44 证明它确实执行了,
    //    但瓶颈其实在 root 侧(A 侧), 代转者的"转"本来就有余量可藏 —— 藏了也不在关键路径上。
    //    机制代码保留(见 ccu_kernel.cc 的 stageId 分支), 用本开关关闭, 便于日后带数据重启。
    resCtxHost.multiStageRelay = false;

    // 4x3 中档专用: Multi-Target + Ready->Write。拓扑用全集群 Mesh inst size
    // (serverCount==4 && minServer==maxServer==3), 不用本 rank mesh/clos 度数。
    // 尺寸用已有 500MB 阈值, 不写死 400MB, 也不发明独立 4B tail kernel。
    std::vector<uint32_t> meshInstSizes;
    CHK_RET(ProbeMeshInstSizes(comm, meshInstSizes));
    const bool fourByThreeTopo = IsUniformMeshInstTopo(meshInstSizes, param.rankSize, 4U, 3U);
    const bool fourByThreeMidBand = fourByThreeTopo && (sendBytes < PROXY_MIN_SEND_BYTES);
    const bool multiTargetProxy = fourByThreeMidBand;
    const bool seedReadyWrite = fourByThreeMidBand;
    const uint32_t serverSizeU = static_cast<uint32_t>(meshCount) + 1;
    const uint32_t targetsPerForwarder = multiTargetProxy
        ? static_cast<uint32_t>(remotePeers.size() / serverSizeU) : 1u;
    if (multiTargetProxy) {
        CHK_PRT_RET(remotePeers.size() % serverSizeU != 0 || targetsPerForwarder == 0,
            HCCL_ERROR("[HcclScatter] multi-target: remotePeers=%zu not a multiple of serverSize=%u",
                remotePeers.size(), serverSizeU), HCCL_E_INTERNAL);
        CHK_PRT_RET(serverPeers.size() + 1 > serverSizeU,
            HCCL_ERROR("[HcclScatter] multi-target: srvPeers=%zu + 1 exceeds serverSize=%u",
                serverPeers.size(), serverSizeU), HCCL_E_INTERNAL);
        CHK_PRT_RET(groups[LAYER_MESH].channels.size() * targetsPerForwarder > 16,
            HCCL_ERROR("[HcclScatter] multi-target: meshCh=%zu * k=%u exceeds the 16-bit event mask",
                groups[LAYER_MESH].channels.size(), targetsPerForwarder), HCCL_E_INTERNAL);
    }
    CHK_PRT_RET(resCtxHost.multiStageRelay && targetsPerForwarder > 1,
        HCCL_ERROR("[HcclScatter] multiStageRelay conflicts with multi-target proxy"), HCCL_E_INTERNAL);

    auto appendTargets = [&](std::vector<uint32_t> &out, size_t t) {
        if (targetsPerForwarder <= 1) {
            out.push_back(remotePeers[t]);
            return;
        }
        for (uint32_t j = 0; j < targetsPerForwarder; ++j) {
            const size_t idx = static_cast<size_t>(j) * serverSizeU + (t + 1);
            if (idx >= remotePeers.size()) {
                break;
            }
            out.push_back(remotePeers[idx]);
        }
    };

    std::vector<uint32_t> proxyPeers;
    std::vector<uint32_t> proxyTargets;
    if (resCtxHost.proxyEnabled && iAmRoot) {
        const size_t pairCount = std::min(serverPeers.size(), remotePeers.size());
        for (size_t t = 0; t < pairCount; ++t) {
            proxyPeers.push_back(serverPeers[t]);
            appendTargets(proxyTargets, t);
        }
    }
    std::vector<uint32_t> myTargets;
    if (resCtxHost.proxyEnabled && iAmServerPeer) {
        const size_t pairCount = std::min(serverPeers.size(), remotePeers.size());
        for (size_t t = 0; t < pairCount; ++t) {
            if (serverPeers[t] == param.myRank) {
                appendTargets(myTargets, t);
            }
        }
    }
    resCtxHost.proxyTargetsPerForwarder = targetsPerForwarder;
    resCtxHost.isForwarder = resCtxHost.proxyEnabled && !myTargets.empty();

    HCCL_INFO("[HcclScatter] rank%u twoServer=%d fourServer=%d fourByThree=%d proxy=%d srvPeers=%zu rmtPeers=%zu "
        "targets=%zu k=%u multiTarget=%d seedReadyWrite=%d",
        param.myRank, static_cast<int>(twoServerTopo), static_cast<int>(symmetricFourServerTopo),
        static_cast<int>(fourByThreeTopo), static_cast<int>(resCtxHost.proxyEnabled),
        serverPeers.size(), remotePeers.size(), myTargets.size(), targetsPerForwarder,
        static_cast<int>(multiTargetProxy), static_cast<int>(seedReadyWrite));
    std::printf("[HcclScatter] pack=%s rank%u fourByThree=%d seedReadyWrite=%d multiTarget=%d k=%u "
        "sendBytes=%llu proxyEnabled=%d pathHint=PROXY\n",
        "v16g_4x3_512kb_closburst", param.myRank, static_cast<int>(fourByThreeTopo),
        static_cast<int>(seedReadyWrite), static_cast<int>(multiTargetProxy), targetsPerForwarder,
        static_cast<unsigned long long>(sendBytes), static_cast<int>(resCtxHost.proxyEnabled));
    if (resCtxHost.proxyEnabled) {
        HCCL_INFO("[HcclScatter] rank%u proxyPeers=%s proxyTargets=%s myTargets=%s", param.myRank,
            FormatRankList(proxyPeers).c_str(), FormatRankList(proxyTargets).c_str(),
            FormatRankList(myTargets).c_str());
    }

    CcuInsHandle insHandle = 0;
    uint32_t insNum = 0;
    CHK_RET(HcclCommQueryCcuIns(comm, &insHandle, &insNum));
    CHK_PRT_RET(insNum != 1,
        HCCL_ERROR("[HcclScatter] unexpected CCU instance count %u", insNum), HCCL_E_INTERNAL);
    CHK_RET_CCU(HcommCcuKernelRegisterStart(insHandle));

    const bool hasMeshGroup = !groups[LAYER_MESH].channels.empty();
    const bool isRootRank = (param.myRank == param.root);
    // 去串行化: 两个代理拓扑的**全部**数据量档都启用。
    //
    // 上一版按 `sendBytes < PROXY_MIN_SEND_BYTES` 把它限制在 400M 档, 理由是
    // "512MB 已在榜一分位附近, 不为它引入执行图变动"。2026-09-17 的实测推翻了这个判断:
    // 400M 档四个点全部落到 root 出口带宽下界上之后, 512MB 两档仍高出各自下界
    //   2x8: 989 vs 942 (+4.9%)      4x3: 1740 vs 1690 (+3.0%)
    // 差的正是这同一步 —— 它们还在跑 `1 + 1.25p` 的串行时序。而榜首的 512MB 成绩
    // (944 / 1.68ms) 恰好压在下界上, 说明这条路是通的。
    // 由于最优 p 与数据量无关(见文件头的推导), 这里放开判据不需要新的常量。
    //
    // 4x1 / 8+4 不受影响: 它们的 proxyEnabled 为假(无同 Server peer 或天然平衡),
    // 走的仍是各自原有的内核与比例。
    const bool detour = resCtxHost.proxyEnabled;
    if (hasMeshGroup) {
        // Mesh kernel 的 proxyTargets 语义随角色而变:
        //   root   -> 哪些跨服 rank 被代转; 代转者 -> 本 rank 要转发到的跨服 rank
        const std::vector<uint32_t> &meshTargets = isRootRank ? proxyTargets : myTargets;
        if (detour && isRootRank) {
            // root 的 Mesh 组拆成 SEED + LOCAL:
            //   SEED  只做前同步 + 投递代理尾段并等它落位, 结束得早
            //         -> 尾段因此在直发块**之前**占用 Mesh 链路;
            //   LOCAL 做直发块 + 自留块 + 后同步, 结束得晚。
            // 两个 kernel 同挂 threads[0] 且按 SEED -> LOCAL 顺序下发, kernel 边界
            // (SEED 的 EventWait 必须完成才结束)给出硬保证 —— 这是同一 kernel 内
            // 调换语句顺序做不到的(平台实测无任何效果)。
            CcuKernelHandle seed = 0;
            CHK_RET(RegisterProxyKernel(insHandle, "CcuScatterProxyMeshSeedKernel", groups[LAYER_MESH],
                param, false, false, resCtxHost.proxyEnabled, proxyPeers, meshTargets,
                resCtxHost.multiStageRelay, seed, CcuKernelArgProxy::MESH_PHASE_SEED,
                seedReadyWrite, targetsPerForwarder));
            resCtxHost.meshKernelIndex = static_cast<uint32_t>(resCtxHost.ccuKernels.size());
            resCtxHost.ccuKernels.push_back(seed);

            CcuKernelHandle local = 0;
            CHK_RET(RegisterProxyKernel(insHandle, "CcuScatterProxyMeshLocalKernel", groups[LAYER_MESH],
                param, true, false, resCtxHost.proxyEnabled, proxyPeers, meshTargets,
                resCtxHost.multiStageRelay, local, CcuKernelArgProxy::MESH_PHASE_LOCAL));
            resCtxHost.meshKernelBIndex = static_cast<uint32_t>(resCtxHost.ccuKernels.size());
            resCtxHost.ccuKernels.push_back(local);
        } else if (detour && resCtxHost.isForwarder) {
            // 代转者的 Mesh 组拆成 TAIL + SYNC:
            //   TAIL 只等 root 的尾段就绪就结束 -> host 据此立刻放行 Clos kernel,
            //        转发不再排在整条 Mesh 数据面之后;
            //   SYNC 只做后同步栅栏, 保证本 rank 自己那块数据已落位 —— 保护没有减少,
            //        只是把栅栏挪到了更晚的一个 kernel 里。
            CcuKernelHandle tail = 0;
            CHK_RET(RegisterProxyKernel(insHandle, "CcuScatterProxyMeshTailKernel", groups[LAYER_MESH],
                param, false, false, resCtxHost.proxyEnabled, proxyPeers, meshTargets,
                resCtxHost.multiStageRelay, tail, CcuKernelArgProxy::MESH_PHASE_TAIL));
            resCtxHost.meshKernelIndex = static_cast<uint32_t>(resCtxHost.ccuKernels.size());
            resCtxHost.ccuKernels.push_back(tail);

            CcuKernelHandle sync = 0;
            CHK_RET(RegisterProxyKernel(insHandle, "CcuScatterProxyMeshSyncKernel", groups[LAYER_MESH],
                param, false, false, resCtxHost.proxyEnabled, proxyPeers, meshTargets,
                resCtxHost.multiStageRelay, sync, CcuKernelArgProxy::MESH_PHASE_SYNC));
            resCtxHost.meshKernelBIndex = static_cast<uint32_t>(resCtxHost.ccuKernels.size());
            resCtxHost.ccuKernels.push_back(sync);
        } else {
            CcuKernelHandle handle = 0;
            CHK_RET(RegisterProxyKernel(insHandle, "CcuScatterProxyMeshKernel", groups[LAYER_MESH], param,
                isRootRank, false, resCtxHost.proxyEnabled, proxyPeers, meshTargets,
                resCtxHost.multiStageRelay, handle, CcuKernelArgProxy::MESH_PHASE_FULL));
            resCtxHost.meshKernelIndex = static_cast<uint32_t>(resCtxHost.ccuKernels.size());
            resCtxHost.ccuKernels.push_back(handle);
        }
    }
    if (!groups[LAYER_CLOS].channels.empty()) {
        const std::vector<uint32_t> &closTargets = isRootRank ? proxyTargets : myTargets;
        CcuKernelHandle handle = 0;
        CHK_RET(RegisterProxyKernel(insHandle, "CcuScatterProxyClosKernel", groups[LAYER_CLOS], param,
            isRootRank && !hasMeshGroup, true, resCtxHost.proxyEnabled, proxyPeers, closTargets,
            resCtxHost.multiStageRelay, handle, CcuKernelArgProxy::MESH_PHASE_FULL));
        resCtxHost.closKernelIndex = static_cast<uint32_t>(resCtxHost.ccuKernels.size());
        resCtxHost.ccuKernels.push_back(handle);
    }
    CHK_RET_CCU(HcommCcuKernelRegisterEnd(insHandle));
    resCtxHost.ccuThread = param.cpuThread;
    return HCCL_SUCCESS;
}

} // namespace

HcclResult HcclScatter(void *sendBuf, void *recvBuf, uint64_t recvCount, HcclDataType dataType, uint32_t root,
    HcclComm comm, aclrtStream stream)
{
    CHK_PTR_NULL(recvBuf);
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(stream);

    // 构造算子参数。最终 tag 在得到 rankSize 和 sendBytes 后生成。
    OpParam param;
    param.inputPtr = sendBuf;
    param.outputPtr = recvBuf;
    param.count = recvCount;
    param.dataType = dataType;
    param.opType = HcclCMDType::HCCL_CMD_SCATTER;

    // 注册算子信息
    HcclDfxOpInfo dfxInfo;
    char commName[COMM_INDENTIFIER_MAX_LENGTH];
    CHK_RET(HcclGetCommName(comm, commName));
    CHK_RET(HcclDfxRegOpInfoByCommId(commName, reinterpret_cast<void *>(&dfxInfo)));

    // ==============================================
    // STEP 1: 解析拓扑信息
    // ==============================================
    CHK_RET(HcclGetRankId(comm, &param.myRank));
    CHK_RET(HcclGetRankSize(comm, &param.rankSize));
    param.root = root; // 模板缺失; 不补则 kernel 恒把 rank 0 当 root
    CHK_PRT_RET(root >= param.rankSize,
        HCCL_ERROR("[HcclScatter] invalid root %u for rankSize %u", root, param.rankSize), HCCL_E_PARA);
    // 非 root 的 sendBuf 无意义, 只做非空校验(runner 会对 root 之外的 sendBuf 填 0)
    if (param.myRank == root) {
        CHK_PTR_NULL(sendBuf);
    }

    // ==============================================
    // STEP 2: 创建资源
    // ==============================================
    CommEngine ccuEngine = CommEngine::COMM_ENGINE_CCU;

    // ==============================================
    // STEP 2.1: 申请用于 Host/Device 同步的通信资源
    // ==============================================
    CHK_RET(HcclThreadAcquireWithStream(comm, ccuEngine, stream, 1, &param.cpuThread));

    // 按数据量选定路径: 三条路径各自独立注册 kernel, 互不影响
    const auto sizeIt = SIZE_TABLE.find(param.dataType);
    const uint64_t sliceBytes = (sizeIt != SIZE_TABLE.end()) ? param.count * sizeIt->second : 0;
    CHK_PRT_RET(sliceBytes == 0,
        HCCL_ERROR("[HcclScatter] zero-size Scatter is not supported"), HCCL_E_PARA);
    const uint64_t sendBytes = sliceBytes * param.rankSize;

    // 注意: 这个 Path 只按 sendBytes 判定, 用途是**生成 ctx tag** —— ctx 查找发生
    // 在 SelectPath(需探测拓扑)之前, 命中时根本不会调 SelectPath, 所以 tag 必须
    // 只依赖尺寸可推导的量。它与真正的选路(内层 ScatterPath)是两件事, 同名易混。
    // 因此 8+4 的 512MB 仍可能挂 hccl_scatter_proxy_* tag; 真正执行路径以
    // resCtx.path 为准(SelectPath 会写成 WAVE_STAR, ExecOp 按它分发)。
    enum class Path { ROOT_STAR, WAVE, PROXY };
    Path path = Path::ROOT_STAR;
    if (sendBytes >= PROXY_MIN_SEND_BYTES) {
        path = Path::PROXY;
    } else if (sendBytes >= WAVE_MIN_SEND_BYTES) {
        path = Path::WAVE;
    }

    // RootStar 的 kernel arg 含有按消息大小固化的 offset，必须让不同消息大小
    // 使用不同 context；Wave/Proxy 保持各自独立的路径 tag，避免跨路径复用资源。
    if (path == Path::ROOT_STAR) {
        if (IsSmallScatter(sendBytes)) {
            // tag 分桶: 4x3 对称, 所有 rank 都是 mesh=2 clos=9。8+4 两侧都不是 2+9,
            // 仍共用旧 tag, 不会造成路径分裂。真正选路 SMALL 仍是 ROOT_STAR。
            uint32_t meshCount = 0;
            uint32_t closCount = 0;
            CHK_RET(ProbeLayerCounts(comm, param, meshCount, closCount));
            if (param.rankSize == 12U && meshCount == 2U && closCount == 9U) {
                std::snprintf(param.tag, sizeof(param.tag), "hccl_scatter_rs_s43b_r%u_b%llu", root,
                    static_cast<unsigned long long>(sendBytes));
            } else {
                std::snprintf(param.tag, sizeof(param.tag), "hccl_scatter_rs_s_r%u_b%llu", root,
                    static_cast<unsigned long long>(sendBytes));
            }
        } else {
            std::snprintf(param.tag, sizeof(param.tag), "hccl_scatter_rs_r%u_b%llu", root,
                static_cast<unsigned long long>(sendBytes));
        }
    } else if (path == Path::WAVE) {
        std::snprintf(param.tag, sizeof(param.tag), "hccl_scatter_wave_r%u", root);
    } else {
        std::snprintf(param.tag, sizeof(param.tag), "hccl_scatter_proxy_r%u", root);
    }
    HCCL_INFO("[HcclScatter] rank%u sendBytes=%llu path=%d tag=%s", param.myRank,
        static_cast<unsigned long long>(sendBytes), static_cast<int>(path), param.tag);
    std::printf("[HcclScatter] pack=v16g_4x3_512kb_closburst rank%u sendBytes=%llu tag=%s\n",
        param.myRank, static_cast<unsigned long long>(sendBytes), param.tag);

    void *ctx = nullptr;
    uint64_t size = 0;
    if (HcclEngineCtxGet(comm, param.tag, ccuEngine, &ctx, &size) == HCCL_SUCCESS) {
        // CCU 资源已经存在, 复用资源
        HCCL_INFO("Engine context already exists");
        param.resCtx = ctx;
        param.ctxSize = size;
    } else {
        AlgResourceCtx resCtxHost;
        void *cclBufferAddr = nullptr;
        uint64_t cclBufferSize = 0;
        CHK_RET(HcclGetHcclBuffer(comm, &cclBufferAddr, &cclBufferSize));
        resCtxHost.localBuffer = CommBuffer{cclBufferAddr, cclBufferSize};

        ScatterPath path = ScatterPath::ROOT_STAR;
        CHK_RET(SelectPath(comm, param, sendBytes, path));
        resCtxHost.path = static_cast<uint32_t>(path);
        HCCL_INFO("[HcclScatter] rank%u sendBytes=%llu path=%s(%u)", param.myRank,
            static_cast<unsigned long long>(sendBytes), ScatterPathName(path), resCtxHost.path);

        if (path == ScatterPath::PROXY) {
            CHK_RET(CreateProxyResources(comm, param, resCtxHost, ccuEngine, sendBytes));
        } else if (path == ScatterPath::WAVE_STAR) {
            CHK_RET(CreateWaveStarResources(comm, param, sliceBytes, resCtxHost, ccuEngine));
        } else if (path == ScatterPath::WAVE) {
            CHK_RET(CreateWaveResources(comm, param, resCtxHost, ccuEngine));
        } else {
            CHK_RET(CreateRootStarResources(comm, param, sliceBytes, sendBytes, resCtxHost, ccuEngine));
        }

        // ==============================================
        // STEP 2.3: 申请通信引擎上下文
        // ==============================================
        std::vector<char> seq = resCtxHost.Serialize();
        uint64_t seqSize = seq.size();
        param.ctxSize = seqSize;
        CHK_RET(HcclEngineCtxCreate(comm, param.tag, ccuEngine, param.ctxSize, &param.resCtx));
        CHK_RET(HcclEngineCtxCopy(comm, ccuEngine, param.tag, seq.data(), seqSize, 0));
    }

    // ==============================================
    // STEP 3: 下发 CCU Kernel
    // ==============================================
    CHK_RET(ops_hccl::ExecOp(param));
    return HCCL_SUCCESS;
}
